#include "object_tracker.hpp"
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream> // For logging
#include <algorithm> // For std::max, std::min
#include <cmath> // For std::round

TrackedObject::TrackedObject(int new_id, const Detection& det,
                             const BatchImageMetadata& meta,
                             int current_frame_index, float delta_t)
    : id(new_id), class_id(det.class_id), confidence(det.confidence),
      frames_since_last_detection(0), total_frames_tracked(1),
      last_seen_frame_index(current_frame_index) {
    kf.init(8, 4, 0); // 8 state variables (x,y,w,h,vx,vy,vw,vh), 4 measurement variables (x,y,w,h)

    kf.transitionMatrix = (cv::Mat_<float>(8, 8) <<
        1, 0, 0, 0, delta_t, 0, 0, 0,
        0, 1, 0, 0, 0, delta_t, 0, 0,
        0, 0, 1, 0, 0, 0, delta_t, 0,
        0, 0, 0, 1, 0, 0, 0, delta_t,
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1);

    float process_noise_pos_scale = 1000.0f;
    float process_noise_vel_scale = 500.0f;
    // float process_noise_pos_scale = 10.0f;
    // float process_noise_vel_scale = 5.0f;
    kf.processNoiseCov = (cv::Mat_<float>(8, 8) <<
        process_noise_pos_scale, 0, 0, 0, 0, 0, 0, 0,
        0, process_noise_pos_scale, 0, 0, 0, 0, 0, 0,
        0, 0, process_noise_pos_scale, 0, 0, 0, 0, 0,
        0, 0, 0, process_noise_pos_scale, 0, 0, 0, 0,
        0, 0, 0, 0, process_noise_vel_scale, 0, 0, 0,
        0, 0, 0, 0, 0, process_noise_vel_scale, 0, 0,
        0, 0, 0, 0, 0, 0, process_noise_vel_scale, 0,
        0, 0, 0, 0, 0, 0, 0, process_noise_vel_scale);

    kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
    for (int i = 0; i < 4; ++i)
        kf.measurementMatrix.at<float>(i, i) = 1.0f;

    float measurement_noise_scale = 1.0f;
    kf.measurementNoiseCov = (cv::Mat_<float>(4, 4) <<
        measurement_noise_scale, 0, 0, 0,
        0, measurement_noise_scale, 0, 0,
        0, 0, measurement_noise_scale, 0,
        0, 0, 0, measurement_noise_scale);

    kf.statePre.at<float>(0) = det.x;
    kf.statePre.at<float>(1) = det.y;
    kf.statePre.at<float>(2) = det.width;
    kf.statePre.at<float>(3) = det.height;
    kf.statePre.at<float>(4) = 0;
    kf.statePre.at<float>(5) = 0;
    kf.statePre.at<float>(6) = 0;
    kf.statePre.at<float>(7) = 0;

    kf.errorCovPost = cv::Mat::eye(8, 8, CV_32F) * 1.0f;
    state = kf.statePre.clone();
    predicted_bbox = cv::Rect2f(det.x, det.y, det.width, det.height);
}

void TrackedObject::predict() {
    state = kf.predict();
    predicted_bbox = cv::Rect2f(state.at<float>(0), state.at<float>(1),
                                state.at<float>(2), state.at<float>(3));
    frames_since_last_detection++;
}

void TrackedObject::update(const Detection& detection) {
    cv::Mat measurement = (cv::Mat_<float>(4, 1) << detection.x, detection.y, detection.width, detection.height);
    state = kf.correct(measurement);
    confidence = detection.confidence;
    frames_since_last_detection = 0;
    total_frames_tracked++;
}


ObjectTracker::ObjectTracker(float iou_threshold, int max_disappeared_frames,
                             float min_confidence_to_track, float delta_t)
    : iou_threshold_(iou_threshold),
      max_disappeared_frames_(max_disappeared_frames),
      min_confidence_to_track_(min_confidence_to_track),
      next_track_id_(0),
      fixed_delta_t_(delta_t) {}

float ObjectTracker::calculateIoU(const Detection &a, const cv::Rect2f &b_bbox_predicted) {
    cv::Rect2f a_bbox(a.x, a.y, a.width, a.height);
    return calculateIoU(a_bbox, b_bbox_predicted);
}

float ObjectTracker::calculateIoU(const cv::Rect2f &a_bbox, const cv::Rect2f &b_bbox) {
    float x1_intersect = std::max(a_bbox.x, b_bbox.x);
    float y1_intersect = std::max(a_bbox.y, b_bbox.y);
    float x2_intersect = std::min(a_bbox.x + a_bbox.width, b_bbox.x + b_bbox.width);
    float y2_intersect = std::min(a_bbox.y + a_bbox.height, b_bbox.y + b_bbox.height);

    if (x2_intersect <= x1_intersect || y2_intersect <= y1_intersect)
        return 0.0f;

    float intersection_area = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect);
    float area_a = a_bbox.width * a_bbox.height;
    float area_b = b_bbox.width * b_bbox.height;
    float union_area = area_a + area_b - intersection_area;

    return union_area > 1e-6f ? intersection_area / union_area : 0.0f;
}

void ObjectTracker::createAndReportInferenceResult(
    const TrackedObject& track,
    const BatchImageMetadata& image_meta,
    int current_global_frame_index,
    InferResultCallback result_callback,
    ImageSaveCallback image_save_callback,
    int task_id,
    const std::string& object_name,
    const std::string& status_info
) {
    InferenceResult infer_result;
    infer_result.taskId = task_id;
    infer_result.frameIndex = current_global_frame_index;
    infer_result.seconds = static_cast<float>(current_global_frame_index) / 30.0f;
    infer_result.confidence = track.confidence;
    infer_result.tracked_id = track.id;

    cv::Rect2f current_bbox_model_space(track.state.at<float>(0), track.state.at<float>(1),
                                        track.state.at<float>(2), track.state.at<float>(3));

    Detection det_for_conversion;
    det_for_conversion.x1 = current_bbox_model_space.x;
    det_for_conversion.y1 = current_bbox_model_space.y;
    det_for_conversion.x2 = current_bbox_model_space.x + current_bbox_model_space.width;
    det_for_conversion.y2 = current_bbox_model_space.y + current_bbox_model_space.height;

    infer_result.bbox_orig_img_space = det_for_conversion.toCvRect2f(image_meta);

    std::ostringstream info_oss;
    info_oss << object_name << " (ID: " << track.id << ", Conf: "
             << std::fixed << std::setprecision(2) << track.confidence * 100 << "%) - "
             << status_info;
    infer_result.info = info_oss.str();

    Detection det_for_save;
    det_for_save.x1 = track.state.at<float>(0);
    det_for_save.y1 = track.state.at<float>(1);
    det_for_save.x2 = track.state.at<float>(0) + track.state.at<float>(2);
    det_for_save.y2 = track.state.at<float>(1) + track.state.at<float>(3);
    det_for_save.confidence = track.confidence;
    det_for_save.class_id = track.class_id;
    det_for_save.batch_idx = -1;
    det_for_save.status_info = status_info + "_ID_" + std::to_string(track.id);

    image_save_callback(det_for_save, image_meta, track.id);

    result_callback(infer_result);
}


std::vector<InferenceResult> ObjectTracker::update(const std::vector<Detection>& detections,
                                                    const BatchImageMetadata& image_meta,
                                                    int current_global_frame_index,
                                                    InferResultCallback result_callback,
                                                    ImageSaveCallback image_save_callback,
                                                    int task_id,
                                                    const std::string& object_name) {
    std::vector<InferenceResult> frame_results;
    std::vector<bool> detection_matched(detections.size(), false);
    std::map<int, bool> track_matched;

    for (auto it = active_tracks_.begin(); it != active_tracks_.end(); ) {
        int track_id = it->first;
        TrackedObject& track = it->second;

        track.predict();
        track_matched[track_id] = false;

        if (track.frames_since_last_detection > max_disappeared_frames_) {
            // Removed: std::cout << "[Tracker] Track ID " << track.id << " disappeared." << std::endl;
            it = active_tracks_.erase(it);
        } else {
            ++it;
        }
    }

    for (size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].confidence < min_confidence_to_track_) {
            continue;
        }

        const Detection& current_det = detections[i];
        float max_iou = 0.0f;
        int best_track_id = -1;

        for (auto& pair : active_tracks_) {
            TrackedObject& track = pair.second;
            if (track.class_id == current_det.class_id && !track_matched[track.id]) {
                float iou = calculateIoU(current_det, track.predicted_bbox);
                if (iou > max_iou && iou > iou_threshold_) {
                    max_iou = iou;
                    best_track_id = track.id;
                }
            }
        }

        if (best_track_id != -1) {
            TrackedObject& matched_track = active_tracks_.at(best_track_id);
            matched_track.update(current_det);
            matched_track.last_seen_frame_index = current_global_frame_index;
            detection_matched[i] = true;
            track_matched[best_track_id] = true;

            createAndReportInferenceResult(matched_track, image_meta, current_global_frame_index, result_callback, image_save_callback, task_id, object_name, "TRACKED");
        }
    }

    for (size_t i = 0; i < detections.size(); ++i) {
        if (!detection_matched[i] && detections[i].confidence >= min_confidence_to_track_) {
            int new_id = next_track_id_++;
            active_tracks_.emplace(new_id,
                TrackedObject(new_id, detections[i], image_meta,
                              current_global_frame_index, fixed_delta_t_));
            createAndReportInferenceResult(active_tracks_.at(new_id), image_meta, current_global_frame_index, result_callback, image_save_callback, task_id, object_name, "NEW");
        }
    }
    return frame_results;
}
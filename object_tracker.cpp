// object_tracker.cpp (Updated)
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

    // Adjust process noise scales
    float process_noise_pos_scale = 10.0f; // Increase for more aggressive tracking
    float process_noise_vel_scale = 5.0f;
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

    // Measurement noise covariance
    float measurement_noise_scale = 1.0f; // Adjust as needed
    kf.measurementNoiseCov = (cv::Mat_<float>(4, 4) <<
        measurement_noise_scale, 0, 0, 0,
        0, measurement_noise_scale, 0, 0,
        0, 0, measurement_noise_scale, 0,
        0, 0, 0, measurement_noise_scale);

    // Initial state (x, y, width, height, vx, vy, vw, vh)
    kf.statePre.at<float>(0) = det.x;
    kf.statePre.at<float>(1) = det.y;
    kf.statePre.at<float>(2) = det.width;
    kf.statePre.at<float>(3) = det.height;
    kf.statePre.at<float>(4) = 0; // Initial velocities
    kf.statePre.at<float>(5) = 0;
    kf.statePre.at<float>(6) = 0;
    kf.statePre.at<float>(7) = 0;

    kf.errorCovPost = cv::Mat::eye(8, 8, CV_32F) * 1.0f; // Post-error covariance
    state = kf.statePre.clone(); // Initialize current state with initial prediction
    predicted_bbox = cv::Rect2f(det.x, det.y, det.width, det.height); // Initial prediction is the detection itself
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
    confidence = detection.confidence; // Update confidence with the latest detection
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
    // Convert detection 'a' to cv::Rect2f
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
    infer_result.seconds = static_cast<float>(current_global_frame_index) / 30.0f; // Assuming 30 FPS
    infer_result.confidence = track.confidence;
    infer_result.tracked_id = track.id;

    // Use the predicted bbox for reporting if it's a prediction, otherwise the updated state
    cv::Rect2f current_bbox_model_space(track.state.at<float>(0), track.state.at<float>(1),
                                        track.state.at<float>(2), track.state.at<float>(3));

    // Create a dummy detection to use its toCvRect2f method
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

    // Prepare detection for image saving (need to reconstruct it from track state)
    // This is important because saveAnnotatedImage expects a Detection struct.
    Detection det_for_save;
    det_for_save.x1 = track.state.at<float>(0);
    det_for_save.y1 = track.state.at<float>(1);
    det_for_save.x2 = track.state.at<float>(0) + track.state.at<float>(2);
    det_for_save.y2 = track.state.at<float>(1) + track.state.at<float>(3);
    det_for_save.confidence = track.confidence;
    det_for_save.class_id = track.class_id;
    det_for_save.batch_idx = -1; // Not applicable for a tracked object
    det_for_save.status_info = status_info + "_ID_" + std::to_string(track.id);

    // Call image saving callback
    image_save_callback(det_for_save, image_meta, track.id); // Using track.id as a pseudo detection index

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
    std::map<int, bool> track_matched; // Map to keep track of which active_tracks_ have been matched

    // 1. Predict new locations for existing tracks
    for (auto it = active_tracks_.begin(); it != active_tracks_.end(); ) {
        int track_id = it->first;
        TrackedObject& track = it->second;

        track.predict(); // Predict next state
        track_matched[track_id] = false; // Mark as unmatched initially

        // Remove old, disappeared tracks
        if (track.frames_since_last_detection > max_disappeared_frames_) {
            std::cout << "[Tracker] Track ID " << track.id << " disappeared." << std::endl;
            // Optionally, create a final InferenceResult for disappeared track
            // createAndReportInferenceResult(track, image_meta, current_global_frame_index, result_callback, image_save_callback, task_id, object_name, "DISAPPEARED");
            it = active_tracks_.erase(it);
        } else {
            ++it;
        }
    }

    // 2. Associate detections with existing tracks
    for (size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].confidence < min_confidence_to_track_) {
            continue; // Skip detections below tracking confidence threshold
        }

        const Detection& current_det = detections[i];
        float max_iou = 0.0f;
        int best_track_id = -1;

        for (auto& pair : active_tracks_) {
            TrackedObject& track = pair.second;
            // Only consider tracks that are of the same class and not yet matched
            if (track.class_id == current_det.class_id && !track_matched[track.id]) {
                float iou = calculateIoU(current_det, track.predicted_bbox);
                if (iou > max_iou && iou > iou_threshold_) {
                    max_iou = iou;
                    best_track_id = track.id;
                }
            }
        }

        if (best_track_id != -1) {
            // Match found
            TrackedObject& matched_track = active_tracks_.at(best_track_id);
            matched_track.update(current_det); // Update Kalman Filter with new detection
            matched_track.last_seen_frame_index = current_global_frame_index;
            detection_matched[i] = true;
            track_matched[best_track_id] = true;

            // Report updated track
            createAndReportInferenceResult(matched_track, image_meta, current_global_frame_index, result_callback, image_save_callback, task_id, object_name, "TRACKED");
        }
    }

    // 3. Handle unmatched detections (new tracks)
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!detection_matched[i] && detections[i].confidence >= min_confidence_to_track_) {
            int new_id = next_track_id_++;
            active_tracks_.emplace(new_id,
                TrackedObject(new_id, detections[i], image_meta,
                              current_global_frame_index, fixed_delta_t_));
            std::cout << "[Tracker] New track created with ID: " << new_id << std::endl;
            // Report new track
            createAndReportInferenceResult(active_tracks_.at(new_id), image_meta, current_global_frame_index, result_callback, image_save_callback, task_id, object_name, "NEW");
        }
    }

    // No need to return frame_results here, as callbacks handle reporting.
    // However, the signature expects a return, so return an empty vector.
    return frame_results;
}
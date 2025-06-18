#include "object_tracker.hpp"
#include <iostream>
#include <set>
#include <iomanip> // For std::fixed, std::setprecision

// =========================================================
// 结构体方法实现
// =========================================================

// Detection::toCvRect2f 的实现现在放在 models.hpp 中

TrackedObject::TrackedObject(int new_id, const Detection& det, const BatchImageMetadata& meta, int current_frame_index)
    : id(new_id), class_id(det.class_id), confidence(det.confidence), frames_since_last_detection(0),
      total_frames_tracked(1), last_seen_frame_index(current_frame_index) {
    // 初始边界框在原始图像空间
    bbox = det.toCvRect2f(meta);

    std::cout << "[Track " << id << "] Initializing KF with BBox: ["
              << bbox.x << "," << bbox.y << "," << bbox.width << "," << bbox.height
              << "] at frame " << current_frame_index << std::endl;

    // =========================================
    // Kalman Filter Initialization
    // State: [x, y, w, h, vx, vy, vw, vh] (8 elements) - position, size, and their velocities
    // Measurement: [x, y, w, h] (4 elements) - position and size from detection
    // =========================================
    kf.init(8, 4, 0); // 8 state variables, 4 measurement variables, 0 control variables

    // Transition matrix (A) - Constant velocity model
    // x_k = x_{k-1} + dt * vx_{k-1}  (dt is implicitly 1 for each prediction step, representing frame interval)
    // y_k = y_{k-1} + dt * vy_{k-1}
    // w_k = w_{k-1} + dt * vw_{k-1}
    // h_k = h_{k-1} + dt * vh_{k-1}
    // vx_k = vx_{k-1}
    // vy_k = vy_{k-1}
    // vw_k = vw_{k-1}
    // vh_k = vh_{k-1}
    kf.transitionMatrix = (cv::Mat_<float>(8, 8) <<
        1, 0, 0, 0, 1, 0, 0, 0,  // x = x + vx
        0, 1, 0, 0, 0, 1, 0, 0,  // y = y + vy
        0, 0, 1, 0, 0, 0, 1, 0,  // w = w + vw
        0, 0, 0, 1, 0, 0, 0, 1,  // h = h + vh
        0, 0, 0, 0, 1, 0, 0, 0,  // vx = vx
        0, 0, 0, 0, 0, 1, 0, 0,  // vy = vy
        0, 0, 0, 0, 0, 0, 1, 0,  // vw = vw
        0, 0, 0, 0, 0, 0, 0, 1   // vh = vh
    );

    // Measurement matrix (H) - Only measure position and size (x, y, w, h)
    kf.measurementMatrix = (cv::Mat_<float>(4, 8) <<
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0
    );

    // Process noise covariance matrix (Q) - How much state changes between steps
    // Adjust these values based on observed object movement and frame interval
    // Higher values allow for more erratic movement, but also more drift
    // Lower values assume smoother movement, but might fail if actual movement is fast
    float process_noise_pos_scale = 100.0f; // Scale for position variance (larger if objects move a lot between 30 frames)
    float process_noise_vel_scale = 25.0f;  // Scale for velocity variance (larger if acceleration is significant)
    kf.processNoiseCov = (cv::Mat_<float>(8, 8) <<
        process_noise_pos_scale, 0, 0, 0, 0, 0, 0, 0,
        0, process_noise_pos_scale, 0, 0, 0, 0, 0, 0,
        0, 0, process_noise_pos_scale, 0, 0, 0, 0, 0,
        0, 0, 0, process_noise_pos_scale, 0, 0, 0, 0,
        0, 0, 0, 0, process_noise_vel_scale, 0, 0, 0,
        0, 0, 0, 0, 0, process_noise_vel_scale, 0, 0,
        0, 0, 0, 0, 0, 0, process_noise_vel_scale, 0,
        0, 0, 0, 0, 0, 0, 0, process_noise_vel_scale
    );

    // Measurement noise covariance matrix (R) - How much noise in measurements (detections)
    // Lower values mean more trust in detections, higher values mean less trust
    float measurement_noise_scale = 10.0f; // Scale for detection accuracy (smaller if detections are very stable)
    kf.measurementNoiseCov = (cv::Mat_<float>(4, 4) <<
        measurement_noise_scale, 0, 0, 0,
        0, measurement_noise_scale, 0, 0,
        0, 0, measurement_noise_scale, 0,
        0, 0, 0, measurement_noise_scale
    );

    // Error covariance matrix (P) - Initial uncertainty in state estimation
    setIdentity(kf.errorCovPost);
    kf.errorCovPost *= 1e3; // Start with high initial uncertainty

    // Set initial state from the first detection
    kf_state = (cv::Mat_<float>(8, 1) << bbox.x, bbox.y, bbox.width, bbox.height, 0, 0, 0, 0); // Initial velocities are zero
    kf.statePost = kf_state;

    // Set initial measurement
    kf_meas = (cv::Mat_<float>(4, 1) << bbox.x, bbox.y, bbox.width, bbox.height);
}

void TrackedObject::update(const Detection& det, const BatchImageMetadata& meta, int current_frame_index) {
    // Update bbox, also in original image space
    bbox = det.toCvRect2f(meta);
    confidence = det.confidence;
    frames_since_last_detection = 0; // Reset disappeared count
    total_frames_tracked++;
    last_seen_frame_index = current_frame_index;

    // Kalman Filter Update (Correction Step)
    kf_meas = (cv::Mat_<float>(4, 1) << bbox.x, bbox.y, bbox.width, bbox.height);
    kf.correct(kf_meas); // Correct state based on new measurement
    std::cout << "[Track " << id << "] Corrected KF State: ["
              << std::fixed << std::setprecision(2) << kf.statePost.at<float>(0) << ","
              << kf.statePost.at<float>(1) << "," << kf.statePost.at<float>(2) << ","
              << kf.statePost.at<float>(3) << "] at frame " << current_frame_index << std::endl;
}

cv::Rect2f TrackedObject::predict() {
    // Kalman Filter Predict Step
    kf_state = kf.predict(); // Predict next state
    cv::Rect2f predicted_bbox(kf_state.at<float>(0), kf_state.at<float>(1),
                              kf_state.at<float>(2), kf_state.at<float>(3));
    std::cout << "[Track " << id << "] Predicted KF BBox: ["
              << std::fixed << std::setprecision(2) << predicted_bbox.x << ","
              << predicted_bbox.y << "," << predicted_bbox.width << ","
              << predicted_bbox.height << "]" << std::endl;
    return predicted_bbox;
}


// =========================================================
// ObjectTracker 类方法实现
// =========================================================

ObjectTracker::ObjectTracker(float iou_threshold, int max_disappeared_frames, float min_confidence_to_track)
    : iou_threshold_(iou_threshold), max_disappeared_frames_(max_disappeared_frames),
      min_confidence_to_track_(min_confidence_to_track), next_track_id_(0) {
    std::cout << "[ObjectTracker] Initialized with IoU threshold: " << iou_threshold_
              << ", Max disappeared frames: " << max_disappeared_frames_
              << ", Min confidence to track: " << min_confidence_to_track_ << std::endl;
}

std::vector<InferenceResult> ObjectTracker::update(
    const std::vector<Detection>& current_detections_from_inferencer,
    const BatchImageMetadata& image_meta,
    int current_global_frame_index,
    InferResultCallback result_callback,
    ImageSaveCallback save_callback,
    int task_id,
    const std::string& object_name
) {
    std::vector<InferenceResult> results_to_report;

    std::cout << "\n[ObjectTracker] Processing Frame: " << current_global_frame_index
              << ", Num Detections: " << current_detections_from_inferencer.size()
              << ", Num Active Tracks (Before): " << active_tracks_.size() << std::endl;


    // 1. 过滤掉置信度低于跟踪阈值的检测结果
    std::vector<Detection> filtered_detections;
    for(const auto& det : current_detections_from_inferencer){
        if(det.confidence >= min_confidence_to_track_){
            filtered_detections.push_back(det);
        }
    }
    std::cout << "[ObjectTracker] Num Filtered Detections (Conf >= " << min_confidence_to_track_
              << "): " << filtered_detections.size() << std::endl;


    // 2. 处理填充帧或无检测的情况
    if (!image_meta.is_real_image) { // If it's a padding frame, do not update tracker based on detections
        std::cout << "[ObjectTracker] This is a padding frame. Only incrementing disappeared counts." << std::endl;
        // Simply increment frames_since_last_detection for all active tracks
        for (auto& pair : active_tracks_) {
            pair.second.frames_since_last_detection++;
            std::cout << "[Track " << pair.first << "] Disappeared count incremented to "
                      << pair.second.frames_since_last_detection << std::endl;
        }
        // Remove tracks that have disappeared for too long
        for (auto it = active_tracks_.begin(); it != active_tracks_.end(); ) {
            if (it->second.frames_since_last_detection > max_disappeared_frames_) {
                std::cout << "[ObjectTracker] Track " << it->second.id << " (" << object_name
                          << ") disappeared. Last seen frame: " << it->second.last_seen_frame_index
                          << ". Current frame: " << current_global_frame_index << std::endl;
                generateAndReportResult(
                    task_id, current_global_frame_index, static_cast<float>(current_global_frame_index) / 30.0f,
                    it->second.id, it->second.bbox, it->second.confidence,
                    "Tracked Object " + std::to_string(it->second.id) + " (" + object_name + ") disappeared.",
                    result_callback, results_to_report
                );
                it = active_tracks_.erase(it);
            } else {
                ++it;
            }
        }
        std::cout << "[ObjectTracker] Num Active Tracks (After padding frame): " << active_tracks_.size() << std::endl;
        return results_to_report;
    }

    // 3. 预测所有活跃轨迹的下一帧位置
    std::map<int, cv::Rect2f> predicted_bboxes; // Track ID -> Predicted BBox (in original image space)
    for (auto& pair : active_tracks_) { // Iterate by reference to allow KF predict to update kf_state internally
        predicted_bboxes[pair.first] = pair.second.predict();
    }

    // 4. Data Association: Matching detections to predicted tracks (Greedy Matching)
    // For more robust multi-object tracking, consider using the Hungarian algorithm here.
    std::vector<std::pair<int, int>> matches; // (filtered_detection_idx, track_id)
    std::set<int> unmatched_detections_indices; // Indices of detections that haven't been matched
    for(int i = 0; i < filtered_detections.size(); ++i) {
        unmatched_detections_indices.insert(i);
    }
    std::set<int> unmatched_tracks_ids; // IDs of tracks that haven't found a match
    for(const auto& pair : active_tracks_) {
        unmatched_tracks_ids.insert(pair.first);
    }

    std::cout << "[ObjectTracker] Attempting to match " << filtered_detections.size() << " detections with "
              << active_tracks_.size() << " active tracks." << std::endl;

    // Greedy matching strategy: For each unmatched detection, find the best unmatched track
    // (based on IoU with predicted bbox) that exceeds the iou_threshold.
    // Loop through detections
    for (int det_idx = 0; det_idx < filtered_detections.size(); ++det_idx) {
        float max_iou = 0.0f;
        int best_track_id = -1;
        
        // Loop through unmatched tracks to find the best match for current detection
        // Create a copy to iterate because unmatched_tracks_ids might be modified inside the loop
        std::vector<int> current_unmatched_tracks_copy(unmatched_tracks_ids.begin(), unmatched_tracks_ids.end());
        
        cv::Rect2f current_det_bbox_orig = filtered_detections[det_idx].toCvRect2f(image_meta);
        
        for (int track_id : current_unmatched_tracks_copy) {
            // Calculate IoU between the current detection (converted to original image space)
            // and the predicted bounding box of the track.
            float iou = calculateIoU(current_det_bbox_orig, predicted_bboxes[track_id]);
            
            std::cout << "  [Match Attempt] Det " << det_idx << " vs Track " << track_id
                      << " Predicted BBox: [" << predicted_bboxes[track_id].x << ","
                      << predicted_bboxes[track_id].y << "," << predicted_bboxes[track_id].width << ","
                      << predicted_bboxes[track_id].height << "]"
                      << " IoU: " << std::fixed << std::setprecision(4) << iou
                      << " (Threshold: " << iou_threshold_ << ")" << std::endl;

            if (iou > max_iou && iou >= iou_threshold_) {
                max_iou = iou;
                best_track_id = track_id;
            }
        }

        if (best_track_id != -1) {
            matches.push_back({det_idx, best_track_id});
            unmatched_detections_indices.erase(det_idx); // Mark detection as matched
            unmatched_tracks_ids.erase(best_track_id);   // Mark track as matched
            std::cout << "  [Match Success] Det " << det_idx << " matched to Track " << best_track_id
                      << " with IoU: " << std::fixed << std::setprecision(4) << max_iou << std::endl;
        } else {
            std::cout << "  [Match Fail] Det " << det_idx << " (BBox: ["
                      << current_det_bbox_orig.x << "," << current_det_bbox_orig.y << ","
                      << current_det_bbox_orig.width << "," << current_det_bbox_orig.height << "]) "
                      << " could not find a match." << std::endl;
        }
    }

    std::cout << "[ObjectTracker] Matched pairs: " << matches.size() << std::endl;
    std::cout << "[ObjectTracker] Unmatched Detections: " << unmatched_detections_indices.size() << std::endl;
    std::cout << "[ObjectTracker] Unmatched Tracks: " << unmatched_tracks_ids.size() << std::endl;


    // 5. Update Matched Tracks
    for (const auto& match : matches) {
        int det_idx = match.first;
        int track_id = match.second;
        TrackedObject& tracked_obj = active_tracks_.at(track_id);

        tracked_obj.update(filtered_detections[det_idx], image_meta, current_global_frame_index);

        // Report update event (e.g., if position significantly changed, or periodic report)
        // For simplicity, always report when tracked and matched
        std::cout << "[ObjectTracker] Track " << track_id << " (" << object_name
                  << ") updated. Frame: " << current_global_frame_index << std::endl;
        generateAndReportResult(
            task_id, current_global_frame_index, static_cast<float>(current_global_frame_index) / 30.0f,
            track_id, tracked_obj.bbox, tracked_obj.confidence,
            "Tracked Object " + std::to_string(track_id) + " (" + object_name + ") updated.",
            result_callback, results_to_report
        );

        // Call image save callback, with tracking ID info
        Detection det_for_save = filtered_detections[det_idx];
        det_for_save.status_info = "TRACKID_" + std::to_string(track_id);
        save_callback(det_for_save, image_meta, 0); // 0 is example index
    }

    // 6. Create New Tracks for Unmatched Detections
    for (int det_idx : unmatched_detections_indices) {
        int new_id = next_track_id_++;
        active_tracks_.emplace(new_id, TrackedObject(new_id, filtered_detections[det_idx], image_meta, current_global_frame_index));

        std::cout << "[ObjectTracker] New Track " << new_id << " (" << object_name
                  << ") appeared. Frame: " << current_global_frame_index << std::endl;
        generateAndReportResult(
            task_id, current_global_frame_index, static_cast<float>(current_global_frame_index) / 30.0f,
            new_id, active_tracks_.at(new_id).bbox, active_tracks_.at(new_id).confidence,
            "New Object " + std::to_string(new_id) + " (" + object_name + ") appeared.",
            result_callback, results_to_report
        );

        // Call image save callback, with new track ID info
        Detection det_for_save = filtered_detections[det_idx];
        det_for_save.status_info = "NEWID_" + std::to_string(new_id);
        save_callback(det_for_save, image_meta, 0); // 0 is example index
    }

    // 7. Increment frames_since_last_detection for Unmatched Tracks
    //    and remove tracks that have disappeared for too long
    for (auto it = active_tracks_.begin(); it != active_tracks_.end(); ) {
        if (unmatched_tracks_ids.count(it->first)) { // If this track was not matched in the current frame
            it->second.frames_since_last_detection++;
            std::cout << "[Track " << it->first << "] Disappeared count incremented to "
                      << it->second.frames_since_last_detection << std::endl;
        }

        if (it->second.frames_since_last_detection > max_disappeared_frames_) {
            std::cout << "[ObjectTracker] Track " << it->second.id << " (" << object_name
                      << ") disappeared. Last seen frame: " << it->second.last_seen_frame_index
                      << ". Current frame: " << current_global_frame_index << std::endl;
            generateAndReportResult(
                task_id, current_global_frame_index, static_cast<float>(current_global_frame_index) / 30.0f,
                it->second.id, it->second.bbox, it->second.confidence,
                "Tracked Object " + std::to_string(it->second.id) + " (" + object_name + ") disappeared.",
                result_callback, results_to_report
            );
            it = active_tracks_.erase(it);
        } else {
            ++it;
        }
    }
    std::cout << "[ObjectTracker] Num Active Tracks (After frame): " << active_tracks_.size() << std::endl;

    return results_to_report;
}

float ObjectTracker::calculateIoU(const cv::Rect2f& bbox1, const cv::Rect2f& bbox2) const {
    float x1_intersect = std::max(bbox1.x, bbox2.x);
    float y1_intersect = std::max(bbox1.y, bbox2.y);
    float x2_intersect = std::min(bbox1.x + bbox1.width, bbox2.x + bbox2.width);
    float y2_intersect = std::min(bbox1.y + bbox1.height, bbox2.y + bbox2.height);

    if (x2_intersect <= x1_intersect || y2_intersect <= y1_intersect) {
        return 0.0f;
    }

    float intersection_area = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect);
    float area1 = bbox1.width * bbox1.height;
    float area2 = bbox2.width * bbox2.height;
    float union_area = area1 + area2 - intersection_area;

    return union_area > 1e-6f ? intersection_area / union_area : 0.0f;
}

void ObjectTracker::generateAndReportResult(
    int taskId,
    int frameIndex,
    float seconds,
    int tracked_id,
    const cv::Rect2f& bbox_orig_img_space,
    float confidence,
    const std::string& message,
    InferResultCallback result_callback,
    std::vector<InferenceResult>& results_to_report
) const {
    InferenceResult res;
    res.taskId = taskId;
    res.frameIndex = frameIndex;
    res.seconds = seconds;
    res.tracked_id = tracked_id;
    res.bbox_orig_img_space = bbox_orig_img_space;
    res.confidence = confidence; // Store as float
    res.info = message + ". Coords (original_image_space): ["
               + std::to_string(static_cast<int>(bbox_orig_img_space.x)) + ","
               + std::to_string(static_cast<int>(bbox_orig_img_space.y)) + ","
               + std::to_string(static_cast<int>(bbox_orig_img_space.x + bbox_orig_img_space.width)) + ","
               + std::to_string(static_cast<int>(bbox_orig_img_space.y + bbox_orig_img_space.height))
               + "]";
    results_to_report.push_back(res); // Add to the list to be returned
    result_callback(res); // Also trigger the immediate callback
}


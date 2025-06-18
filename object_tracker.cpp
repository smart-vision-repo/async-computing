#include "object_tracker.hpp"
#include <iostream>
#include <set>
#include <iomanip> // For std::fixed, std::setprecision

// =========================================================
// 结构体方法实现
// =========================================================

cv::Rect2f Detection::toCvRect2f(const BatchImageMetadata& meta) const {
    // 将模型输入空间的坐标反变换到原始图像空间
    // 步骤 1: 移除 letterbox 填充
    float x1_unpadded = x1 - meta.pad_w_left;
    float y1_unpadded = y1 - meta.pad_h_top;
    float x2_unpadded = x2 - meta.pad_w_left;
    float y2_unpadded = y2 - meta.pad_h_top;

    // 步骤 2: 反缩放回原始尺寸
    int x1_orig = static_cast<int>(std::round(x1_unpadded / meta.scale_to_model));
    int y1_orig = static_cast<int>(std::round(y1_unpadded / meta.scale_to_model));
    int x2_orig = static_cast<int>(std::round(x2_unpadded / meta.scale_to_model));
    int y2_orig = static_cast<int>(std::round(y2_unpadded / meta.scale_to_model));

    // 步骤 3: 钳制到原始图像边界
    x1_orig = std::max(0, std::min(x1_orig, meta.original_w - 1));
    y1_orig = std::max(0, std::min(y1_orig, meta.original_h - 1));
    x2_orig = std::max(0, std::min(x2_orig, meta.original_w - 1));
    y2_orig = std::max(0, std::min(y2_orig, meta.original_h - 1));

    return cv::Rect2f(static_cast<float>(x1_orig), static_cast<float>(y1_orig),
                      static_cast<float>(x2_orig - x1_orig), static_cast<float>(y2_orig - y1_orig));
}


TrackedObject::TrackedObject(int new_id, const Detection& det, const BatchImageMetadata& meta, int current_frame_index)
    : id(new_id), class_id(det.class_id), confidence(det.confidence), frames_since_last_detection(0),
      total_frames_tracked(1), last_seen_frame_index(current_frame_index) {
    // 初始边界框在原始图像空间
    bbox = det.toCvRect2f(meta);
    // 如果使用卡尔曼滤波器，这里进行初始化：kalman_filter = std::make_unique<KalmanFilterWrapper>(bbox);
}

void TrackedObject::update(const Detection& det, const BatchImageMetadata& meta, int current_frame_index) {
    // 更新边界框，同样在原始图像空间
    bbox = det.toCvRect2f(meta);
    confidence = det.confidence;
    frames_since_last_detection = 0; // 重置未检测到计数
    total_frames_tracked++;
    last_seen_frame_index = current_frame_index;
    // 如果使用卡尔曼滤波器，这里进行更新：kalman_filter->update(bbox);
}

cv::Rect2f TrackedObject::predict() const {
    // 如果使用卡尔曼滤波器，这里进行预测：return kalman_filter->predict();
    // 简化：直接返回当前边界框作为预测值
    return bbox;
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

    // 1. 过滤掉置信度低于跟踪阈值的检测结果
    std::vector<Detection> filtered_detections;
    for(const auto& det : current_detections_from_inferencer){
        if(det.confidence >= min_confidence_to_track_){
            filtered_detections.push_back(det);
        }
    }

    // 2. 处理填充帧或无检测的情况
    if (!image_meta.is_real_image) { // 如果是填充帧，则不进行跟踪更新
        // 简单递增所有活跃轨迹的 frames_since_last_detection
        for (auto& pair : active_tracks_) {
            pair.second.frames_since_last_detection++;
        }
        // 移除长时间未检测到的轨迹
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
        return results_to_report;
    }

    // 3. 预测所有活跃轨迹的下一帧位置（如果使用卡尔曼滤波器，这里会更复杂）
    std::map<int, cv::Rect2f> predicted_bboxes; // 轨迹ID -> 预测的BBox
    for (const auto& pair : active_tracks_) {
        predicted_bboxes[pair.first] = pair.second.predict();
    }

    // 4. 数据关联：将当前帧的检测与预测的轨迹进行匹配 (简单贪婪匹配)
    // 更好的做法是使用匈牙利算法来解决多对多匹配
    std::vector<std::pair<int, int>> matches; // (filtered_detection_idx, track_id)
    std::set<int> unmatched_detections_indices; // 未匹配的检测索引
    for(int i = 0; i < filtered_detections.size(); ++i) {
        unmatched_detections_indices.insert(i);
    }
    std::set<int> unmatched_tracks_ids; // 未匹配的轨迹ID
    for(const auto& pair : active_tracks_) {
        unmatched_tracks_ids.insert(pair.first);
    }

    // 优先匹配，最大化 IoU
    for (int det_idx : unmatched_detections_indices) { // 遍历未匹配的检测
        float max_iou = 0.0f;
        int best_track_id = -1;
        // 注意：这里是对 `unmatched_tracks_ids` 的拷贝进行迭代，因为我们会在循环内部修改 `unmatched_tracks_ids`
        std::vector<int> current_unmatched_tracks(unmatched_tracks_ids.begin(), unmatched_tracks_ids.end());
        for (int track_id : current_unmatched_tracks) {
            float iou = calculateIoU(filtered_detections[det_idx].toCvRect2f(image_meta), predicted_bboxes[track_id]);
            if (iou > max_iou && iou >= iou_threshold_) {
                max_iou = iou;
                best_track_id = track_id;
            }
        }
        if (best_track_id != -1) {
            matches.push_back({det_idx, best_track_id});
            unmatched_detections_indices.erase(det_idx);
            unmatched_tracks_ids.erase(best_track_id);
            // 确保一旦匹配，就不再将其视为未匹配轨迹
        }
    }


    // 5. 更新已匹配的轨迹
    for (const auto& match : matches) {
        int det_idx = match.first;
        int track_id = match.second;
        TrackedObject& tracked_obj = active_tracks_.at(track_id);

        tracked_obj.update(filtered_detections[det_idx], image_meta, current_global_frame_index);

        // 报告更新事件
        generateAndReportResult(
            task_id, current_global_frame_index, static_cast<float>(current_global_frame_index) / 30.0f,
            track_id, tracked_obj.bbox, tracked_obj.confidence,
            "Tracked Object " + std::to_string(track_id) + " (" + object_name + ") updated.",
            result_callback, results_to_report
        );

        // 调用图像保存回调，附带跟踪ID信息
        Detection det_for_save = filtered_detections[det_idx];
        det_for_save.status_info = "TRACKID_" + std::to_string(track_id);
        save_callback(det_for_save, image_meta, 0); // 0 为示例索引
    }

    // 6. 为未匹配的检测创建新轨迹
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

        // 调用图像保存回调，附带新轨迹ID信息
        Detection det_for_save = filtered_detections[det_idx];
        det_for_save.status_info = "NEWID_" + std::to_string(new_id);
        save_callback(det_for_save, image_meta, 0); // 0 为示例索引
    }

    // 7. 递增未匹配轨迹的 frames_since_last_detection
    //    并在必要时移除长时间未检测到的轨迹
    for (auto it = active_tracks_.begin(); it != active_tracks_.end(); ) {
        if (unmatched_tracks_ids.count(it->first)) { // 如果该轨迹未在当前帧匹配到
            it->second.frames_since_last_detection++;
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
    res.confidence = static_cast<int>(confidence * 10000);
    res.info = message + ". Coords (original_image_space): ["
               + std::to_string(static_cast<int>(bbox_orig_img_space.x)) + ","
               + std::to_string(static_cast<int>(bbox_orig_img_space.y)) + ","
               + std::to_string(static_cast<int>(bbox_orig_img_space.x + bbox_orig_img_space.width)) + ","
               + std::to_string(static_cast<int>(bbox_orig_img_space.y + bbox_orig_img_space.height))
               + "]";
    results_to_report.push_back(res); // Add to the list to be returned
    result_callback(res); // Also trigger the immediate callback
}

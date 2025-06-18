// object_tracker.hpp (Updated)
#pragma once
#include "models.hpp"
#include <opencv2/video/tracking.hpp>
#include <map>
#include <memory>
#include <vector>
#include <functional> // For std::function

// Forward declarations
struct InferenceResult;
struct BatchImageMetadata;
struct Detection;

// Callbacks needed by ObjectTracker
using InferResultCallback = std::function<void(const InferenceResult&)>;
using ImageSaveCallback = std::function<void(const Detection&, const BatchImageMetadata&, int)>;


class TrackedObject {
public:
    TrackedObject(int id, const Detection& det, const BatchImageMetadata& meta,
                  int current_frame_index, float delta_t);
    cv::KalmanFilter kf;
    int id;
    int class_id;
    float confidence;
    int frames_since_last_detection;
    int total_frames_tracked;
    int last_seen_frame_index;

    // Kalman Filter state (x, y, width, height, vx, vy, vw, vh)
    cv::Mat state;

    // Predicted bounding box
    cv::Rect2f predicted_bbox;

    void predict();
    void update(const Detection& detection);
};

class ObjectTracker {
public:
    ObjectTracker(float iou_threshold, int max_disappeared_frames,
                  float min_confidence_to_track, float delta_t = 1.0f);

    // The update method now takes callbacks
    std::vector<InferenceResult> update(const std::vector<Detection>& detections,
                                        const BatchImageMetadata& image_meta,
                                        int current_global_frame_index,
                                        InferResultCallback result_callback,
                                        ImageSaveCallback image_save_callback,
                                        int task_id,
                                        const std::string& object_name);
private:
    std::map<int, TrackedObject> active_tracks_;
    float iou_threshold_;
    int max_disappeared_frames_;
    float min_confidence_to_track_;
    int next_track_id_;
    float fixed_delta_t_;

    // Helper for IoU calculation
    float calculateIoU(const Detection &a, const cv::Rect2f &b_bbox_predicted);
    float calculateIoU(const cv::Rect2f &a_bbox, const cv::Rect2f &b_bbox); // Overload for two Rect2f

    // Helper for creating InferenceResult
    void createAndReportInferenceResult(
        const TrackedObject& track,
        const BatchImageMetadata& image_meta,
        int current_global_frame_index,
        InferResultCallback result_callback,
        ImageSaveCallback image_save_callback,
        int task_id,
        const std::string& object_name,
        const std::string& status_info
    );
};
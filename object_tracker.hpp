#pragma once
#include "models.hpp"
#include <opencv2/video/tracking.hpp>
#include <map>
#include <memory>
#include <vector>

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
};

class ObjectTracker {
public:
    ObjectTracker(float iou_threshold, int max_disappeared_frames,
                  float min_confidence_to_track, float delta_t = 1.0f);
    void update(const std::vector<Detection>& detections,
                const BatchImageMetadata& image_meta,
                int current_global_frame_index);
private:
    std::map<int, TrackedObject> active_tracks_;
    float iou_threshold_;
    int max_disappeared_frames_;
    float min_confidence_to_track_;
    int next_track_id_;
    float fixed_delta_t_;
};

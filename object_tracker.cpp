#include "object_tracker.hpp"
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

TrackedObject::TrackedObject(int new_id, const Detection& det,
                             const BatchImageMetadata& meta,
                             int current_frame_index, float delta_t)
    : id(new_id), class_id(det.class_id), confidence(det.confidence),
      frames_since_last_detection(0), total_frames_tracked(1),
      last_seen_frame_index(current_frame_index) {
    kf.init(8, 4, 0);
    kf.transitionMatrix = (cv::Mat_<float>(8, 8) <<
        1, 0, 0, 0, delta_t, 0, 0, 0,
        0, 1, 0, 0, 0, delta_t, 0, 0,
        0, 0, 1, 0, 0, 0, delta_t, 0,
        0, 0, 0, 1, 0, 0, 0, delta_t,
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1);

    float process_noise_pos_scale = 200.0f * delta_t * delta_t;
    float process_noise_vel_scale = 50.0f * delta_t;
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

    cv::Rect2f bbox = det.bbox;
    kf.statePre.at<float>(0) = bbox.x;
    kf.statePre.at<float>(1) = bbox.y;
    kf.statePre.at<float>(2) = bbox.width;
    kf.statePre.at<float>(3) = bbox.height;
    kf.statePre.at<float>(4) = 0;
    kf.statePre.at<float>(5) = 0;
    kf.statePre.at<float>(6) = 0;
    kf.statePre.at<float>(7) = 0;
    kf.errorCovPre = cv::Mat::eye(8, 8, CV_32F);
}

ObjectTracker::ObjectTracker(float iou_threshold, int max_disappeared_frames,
                             float min_confidence_to_track, float delta_t)
    : iou_threshold_(iou_threshold),
      max_disappeared_frames_(max_disappeared_frames),
      min_confidence_to_track_(min_confidence_to_track),
      next_track_id_(0),
      fixed_delta_t_(delta_t) {}

void ObjectTracker::update(const std::vector<Detection>& filtered_detections,
                           const BatchImageMetadata& image_meta,
                           int current_global_frame_index) {
    for (size_t det_idx = 0; det_idx < filtered_detections.size(); ++det_idx) {
        int new_id = next_track_id_++;
        active_tracks_.emplace(new_id,
            TrackedObject(new_id, filtered_detections[det_idx], image_meta,
                          current_global_frame_index, fixed_delta_t_));
    }
}

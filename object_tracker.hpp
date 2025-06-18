#ifndef OBJECT_TRACKER_HPP
#define OBJECT_TRACKER_HPP

#include <vector>
#include <map>
#include <string>
#include <memory> // For std::unique_ptr
#include <functional>
#include <algorithm>
#include <numeric>

// OpenCV headers for image processing, geometry, and KalmanFilter
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp> // For cv::KalmanFilter

// Contains shared struct definitions like Detection, InferenceResult, BatchImageMetadata
#include "models.hpp"

// =========================================================
// Callback type definitions
// =========================================================

// Callback function for saving annotated images, typically provided by TensorInferencer
// Detection, BatchImageMetadata are defined in models.hpp
using ImageSaveCallback = std::function<void(const Detection&, const BatchImageMetadata&, int)>;
// Callback function for reporting inference results (including tracking results), typically by VideoProcessor
// InferenceResult is defined in models.hpp
using InferResultCallback = std::function<void(const InferenceResult&)>;


/**
 * @brief Represents a tracked object trajectory.
 */
struct TrackedObject {
    int id;                   // Unique track ID
    int class_id;             // Object class ID
    float confidence;         // Current detection confidence
    cv::Rect2f bbox;          // Current bounding box (in original image space)
    int frames_since_last_detection; // Number of frames passed since last detection
    int total_frames_tracked; // Total frames tracked
    int last_seen_frame_index; // Global frame index when last detected

    cv::KalmanFilter kf;      // Kalman filter instance
    cv::Mat kf_state;         // Kalman filter state vector (x, y, w, h, vx, vy, vw, vh)
    cv::Mat kf_meas;          // Kalman filter measurement vector (x, y, w, h)

    // Constructor: Creates a TrackedObject from a new Detection
    // Now accepts delta_t to properly configure Kalman Filter transition matrix
    TrackedObject(int new_id, const Detection& det, const BatchImageMetadata& meta,
                  int current_frame_index, float delta_t);

    // Update function: Updates TrackedObject's state with a new Detection
    void update(const Detection& det, const BatchImageMetadata& meta, int current_frame_index);

    // Predict function: Uses Kalman Filter to predict next frame's position
    cv::Rect2f predict();
};

/**
 * @brief Manages object tracking across video frames, handling object IDs and track lifecycle.
 */
class ObjectTracker {
public:
    /**
     * @brief Constructor.
     * @param iou_threshold IoU threshold for data association.
     * @param max_disappeared_frames Max number of frames a track can be "lost" (undetected) before removal.
     * @param min_confidence_to_track Only detections above this confidence are considered for tracking.
     * @param initial_delta_t Initial time step (e.g., frame interval) for Kalman Filter setup.
     */
    ObjectTracker(float iou_threshold, int max_disappeared_frames, float min_confidence_to_track, float initial_delta_t);

    /**
     * @brief Core update method. Processes current frame's detections and updates, creates, or removes tracks.
     * @param current_detections_from_inferencer All detections from the inferencer for the current frame.
     * @param image_meta Current image's metadata for coordinate transformation and image saving.
     * @param current_global_frame_index Current global frame index being processed.
     * @param result_callback Callback function for reporting tracking events.
     * @param save_callback Callback function for saving annotated images.
     * @param task_id Task ID.
     * @param object_name Name of the object being tracked.
     * @return Vector of InferenceResult to report for this frame.
     */
    std::vector<InferenceResult> update(
        const std::vector<Detection>& current_detections_from_inferencer,
        const BatchImageMetadata& image_meta,
        int current_global_frame_index,
        InferResultCallback result_callback,
        ImageSaveCallback save_callback,
        int task_id,
        const std::string& object_name
    );

private:
    float iou_threshold_;          // IoU matching threshold
    int max_disappeared_frames_;   // Max frames a track can disappear
    float min_confidence_to_track_;// Minimum confidence for tracking

    std::map<int, TrackedObject> active_tracks_; // Currently active tracks
    int next_track_id_;                        // Counter for unique track IDs
    float fixed_delta_t_;                      // Fixed time step (e.g., interval) for KF prediction

    /**
     * @brief Calculates IoU (Intersection over Union) between two bounding boxes.
     * @param bbox1 First bounding box.
     * @param bbox2 Second bounding box.
     * @return IoU value.
     */
    float calculateIoU(const cv::Rect2f& bbox1, const cv::Rect2f& bbox2) const;

    // Internal helper function for generating InferenceResult
    void generateAndReportResult(
        int taskId,
        int frameIndex,
        float seconds,
        int tracked_id,
        const cv::Rect2f& bbox_orig_img_space,
        float confidence,
        const std::string& message,
        InferResultCallback result_callback,
        std::vector<InferenceResult>& results_to_report // Pass by reference to add results
    ) const;
};

#endif // OBJECT_TRACKER_HPP
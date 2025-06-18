// models.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <functional> // For std::function used in callbacks if they were here, but typically in inferencer.hpp

// =========================================================
// 共享结构体定义
// =========================================================

struct InferenceInput {
  std::vector<cv::Mat> decoded_frames;
  int gopIdx;
  int latest_frame_index;
};

struct InferenceResult {
  std::string info;
  int type = 10; // 0: detection, 1: classification, etc.
  int taskId;
  int frameIndex;
  float seconds;
  std::string image;    // Saved annotated image path
  float confidence;     // Original detection confidence (float)

  // New fields for tracking
  int tracked_id = -1; // Unique ID for the tracked object
  cv::Rect2f bbox_orig_img_space; // Bounding box in original image space
};

struct BatchImageMetadata {
    int original_w;           // Original image width
    int original_h;           // Original image height
    float scale_to_model;     // Image scaling factor
    int pad_w_left;           // Left padding width
    int pad_h_top;            // Top padding height
    bool is_real_image;       // Is this a real image (not padding)?
    cv::Mat original_image_for_callback; // Copy of original image, for callbacks and annotation
    int global_frame_index;   // Global frame index in the video
};

struct Detection {
    float x1, y1, x2, y2;  // Bounding box coordinates (top-left and bottom-right)
    float confidence;      // Detection confidence
    int class_id;          // Class ID
    int batch_idx;         // Index within the batch
    std::string status_info; // Additional status info, e.g., for tracker to mark ID

    // Convenience members for Kalman Filter
    float x, y, width, height; // Center_x, center_y, width, height

    cv::Rect2f toCvRect2f(const BatchImageMetadata& meta) const;
};

inline cv::Rect2f Detection::toCvRect2f(const BatchImageMetadata& meta) const {
    // Convert coordinates from model input space back to original image space
    float x1_unpadded = x1 - meta.pad_w_left;
    float y1_unpadded = y1 - meta.pad_h_top;
    float x2_unpadded = x2 - meta.pad_w_left;
    float y2_unpadded = y2 - meta.pad_h_top;

    float x1_orig = std::round(x1_unpadded / meta.scale_to_model);
    float y1_orig = std::round(y1_unpadded / meta.scale_to_model);
    float x2_orig = std::round(x2_unpadded / meta.scale_to_model);
    float y2_orig = std::round(y2_unpadded / meta.scale_to_model);

    // Clamp to original image bounds
    x1_orig = std::max(0.0f, std::min(x1_orig, static_cast<float>(meta.original_w - 1)));
    y1_orig = std::max(0.0f, std::min(y1_orig, static_cast<float>(meta.original_h - 1)));
    x2_orig = std::max(0.0f, std::min(x2_orig, static_cast<float>(meta.original_w - 1)));
    y2_orig = std::max(0.0f, std::min(y2_orig, static_cast<float>(meta.original_h - 1)));

    // Ensure positive width and height
    float width = x2_orig - x1_orig;
    float height = y2_orig - y1_orig;
    width = std::max(0.0f, width);
    height = std::max(0.0f, height);

    return cv::Rect2f(x1_orig, y1_orig, width, height);
}

struct TaskDecodeInfo {
  int taskId;
  int type = 20;
  int decoded_frames;
  int disposed_frames;
  int infer_frames;
  int total;
};

struct TaskInferInfo {
  int taskId;
  int type = 30;
  int completed;
  int remain;
};
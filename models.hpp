// inference_input.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <functional> // For std::function used in callbacks if they were here, but typically in inferencer.hpp

// =========================================================
// 共享结构体定义
// =========================================================

/**
 * @brief 推理输入数据结构。
 */
struct InferenceInput {
  std::vector<cv::Mat> decoded_frames;
  int gopIdx;
  int latest_frame_index;
};

/**
 * @brief 推理结果结构。
 * 用于报告给上层应用。
 */
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

/**
 * @brief 表示模型检测到的一个对象。
 * 坐标是模型输入空间中的坐标。
 */
struct Detection {
    float x1, y1, x2, y2;  // Bounding box coordinates (top-left and bottom-right)
    float confidence;      // Detection confidence
    int class_id;          // Class ID
    int batch_idx;         // Index within the batch
    std::string status_info; // Additional status info, e.g., for tracker to mark ID

    // Forward declaration of BatchImageMetadata, actual definition is below.
    // This allows toCvRect2f to be a member function here.
    // If BatchImageMetadata is defined before Detection, this forward declaration is not strictly needed.
    // However, it's safer if structure order is uncertain or inter-dependent.
    cv::Rect2f toCvRect2f(const struct BatchImageMetadata& meta) const;
};


/**
 * @brief 包含用于反向变换模型输入坐标到原始图像坐标的元数据。
 */
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

// =========================================================
// Detection::toCvRect2f 的实现，由于依赖 BatchImageMetadata，
// 且希望 Detection 在 BatchImageMetadata 之前定义，
// 因此函数体放在这里（定义之后，或在一个 .cpp 文件中）
// =========================================================
inline cv::Rect2f Detection::toCvRect2f(const BatchImageMetadata& meta) const {
    // Convert coordinates from model input space back to original image space
    // Step 1: Remove letterbox padding
    float x1_unpadded = x1 - meta.pad_w_left;
    float y1_unpadded = y1 - meta.pad_h_top;
    float x2_unpadded = x2 - meta.pad_w_left;
    float y2_unpadded = y2 - meta.pad_h_top;

    // Step 2: Inverse scale back to original dimensions
    int x1_orig = static_cast<int>(std::round(x1_unpadded / meta.scale_to_model));
    int y1_orig = static_cast<int>(std::round(y1_unpadded / meta.scale_to_model));
    int x2_orig = static_cast<int>(std::round(x2_unpadded / meta.scale_to_model));
    int y2_orig = static_cast<int>(std::round(y2_unpadded / meta.scale_to_model));

    // Step 3: Clamp to original image bounds
    x1_orig = std::max(0, std::min(x1_orig, meta.original_w - 1));
    y1_orig = std::max(0, std::min(y1_orig, meta.original_h - 1));
    x2_orig = std::max(0, std::min(x2_orig, meta.original_w - 1));
    y2_orig = std::max(0, std::min(y2_orig, meta.original_h - 1));

    return cv::Rect2f(static_cast<float>(x1_orig), static_cast<float>(y1_orig),
                      static_cast<float>(x2_orig - x1_orig), static_cast<float>(y2_orig - y1_orig));
}


/**
 * @brief 解码过程信息
 */
struct TaskDecodeInfo {
  int taskId;
  int type = 20;
  int decoded_frames;
  int disposed_frames;
  int infer_frames;
  int total;
};

/**
 * @brief 推理过程信息
 */
struct TaskInferInfo {
  int taskId;
  int type = 30;
  int completed;
  int remain;
};


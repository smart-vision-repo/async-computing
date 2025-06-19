// inference_input.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct InferenceInput {
  std::vector<cv::Mat> decoded_frames;
  int gopIdx;
  int latest_frame_index;
};

// 推理结果结构
struct InferenceResult {
  std::string info;
  int type = 10; // 0: detection, 1: classification, etc.
  int taskId;
  int frameIndex;
  int seconds;
  std::string image;
  float confidence;
};

// 解码过程信息
struct TaskDecodeInfo {
  int taskId;
  int type = 20;
  int decoded_frames;
  int disposed_frames;
  int infer_frames;
  int total;
};

// 推理过程信息
struct TaskInferInfo {
  int taskId;
  int type = 30;
  int completed;
  int remain;
};

// 结构体：用于存储检测结果
struct Detection {
  float x1, y1, x2, y2; // 边界框坐标 (模型输出尺寸，通常是letterbox后的)
  float confidence;     // 置信度
  int class_id;         // 类别ID
  int original_batch_input_idx; // 在当前GPU批次中的索引 (用于调试或关联)
  std::string status_info;      // 额外状态信息，例如 "REAL" 或 "PAD"
};

// 结构体：用于存储单个图像预处理后的元数据 (主要用于letterbox)
struct BatchImageMetadata {
  int original_w = 0;          // 原始图像宽度
  int original_h = 0;          // 原始图像高度
  float scale_to_model = 0.0f; // 缩放比例 (原始图像到模型输入尺寸)
  int pad_w_left = 0;          // 左侧填充宽度
  int pad_h_top = 0;           // 顶部填充高度
  bool is_real_image = false;  // 标记是否为真实图像 (非填充)
  int global_frame_index = 0;
  cv::Mat original_image_for_callback; // 存储原始图像，用于回调和保存 (cloned
                                       // from input)
};

// 新结构体：用于缓存恒定输入图像的几何信息
struct CachedFrameGeometry {
  int original_w = 0;
  int original_h = 0;
  float scale_to_model = 0.0f;
  int pad_w_left = 0;
  int pad_h_top = 0;
};

// 用于筛选推理结果中的图片
struct DetectedFrame {
  Detection detection;
  BatchImageMetadata meta;
};

struct FrameSelection {
  DetectedFrame info;
  bool output;
};
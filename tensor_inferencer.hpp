#pragma once

#include "NvInfer.h"
#include "inference.hpp" // Contains InferenceInput and InferenceResult
#include <functional>    // For std::function
#include <map>
#include <mutex>                 // For thread safety
#include <opencv2/core/cuda.hpp> // For cv::cuda::GpuMat
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

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

// 推理回调函数类型
using InferenceCallback =
    std::function<void(const std::vector<InferenceResult> &results)>;

class TensorInferencer {
public:
  TensorInferencer(int video_height, int video_width, std::string object_name,
                   int interval, float confidence, InferenceCallback callback);
  ~TensorInferencer();

  bool infer(const InferenceInput &input);
  void finalizeInference();

private:
  // Configuration parameters
  std::string object_name_;
  int interval_;
  float confidence_;
  int BATCH_SIZE_ = 1; // Default, can be overridden by env var
  int target_w_;       // Model input width
  int target_h_;       // Model input height
  std::string engine_path_;
  std::string image_output_path_;

  // TensorRT components
  nvinfer1::IRuntime *runtime_ = nullptr;
  nvinfer1::ICudaEngine *engine_ = nullptr;
  nvinfer1::IExecutionContext *context_ = nullptr;

  // TensorRT binding info
  int inputIndex_ = -1;
  int outputIndex_ = -1;
  std::vector<void *> bindings_;

  // CUDA device buffers
  void *inputDevice_ = nullptr;  // Pointer to GPU memory for input
  void *outputDevice_ = nullptr; // Pointer to GPU memory for output

  // Class and label mapping
  std::map<std::string, int> class_name_to_id_;
  std::map<int, std::string> id_to_class_name_;
  int num_classes_ = 0;

  // Batching members for CPU-side raw frames and metadata
  std::vector<cv::Mat> current_batch_raw_frames_;
  std::vector<BatchImageMetadata> current_batch_metadata_;

  // Callback and synchronization
  InferenceCallback current_callback_;
  std::mutex batch_mutex_;

  // Cached metadata for constant input size optimization
  bool constant_metadata_initialized_ = false;
  CachedFrameGeometry cached_geometry_;

  // Private methods
  void printEngineInfo();
  void performBatchInference(bool pad_batch);

  // GPU pre-processing method
  void preprocess_single_image_for_batch(
      const cv::Mat &cpu_img, // Input CPU image
      BatchImageMetadata &meta, int model_input_w, int model_input_h,
      cv::cuda::GpuMat
          &chw_planar_output_gpu_buffer_slice // Wraps a slice of inputDevice_
  );
  //
  void process_single_output(const BatchImageMetadata &image_meta,
                             const float *host_output_for_image_raw,
                             int num_detections_in_slice,
                             int num_attributes_per_detection,
                             int original_batch_idx_for_debug,
                             std::vector<InferenceResult> &frame_results);
  std::vector<Detection> applyNMS(const std::vector<Detection> &detections,
                                  float iou_threshold);
  float calculateIoU(const Detection &a, const Detection &b);
  void saveAnnotatedImage(const Detection &det,
                          const BatchImageMetadata &image_meta,
                          int detection_idx_in_image);
  static std::vector<char> readEngineFile(const std::string &enginePath);
  static int roundToNearestMultiple(int val, int base = 32);
};
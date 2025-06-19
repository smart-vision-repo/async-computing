#pragma once

#include "NvInfer.h"
#include "frame_selector.hpp"
#include "models.hpp" // Contains InferenceInput and InferenceResult
#include <functional> // For std::function
#include <map>
#include <mutex>                 // For thread safety
#include <opencv2/core/cuda.hpp> // For cv::cuda::GpuMat
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

// 推理回调函数类型
using InferResultCallback = std::function<void(const InferenceResult &results)>;
using InferPackCallback = std::function<void(const int count)>;

class TensorInferencer {
public:
  TensorInferencer(int task_id, int video_height, int video_width,
                   std::string object_name, int interval, float confidence,
                   InferResultCallback resultCallback,
                   InferPackCallback packCallback);
  ~TensorInferencer();

  bool infer(const InferenceInput &input);
  void finalizeInference();

private:
  // Configuration parameters
  int task_id_;
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
  InferResultCallback result_callback_;
  InferPackCallback pack_callback_;
  std::optional<FrameSelector> frameSelector;
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
                          const BatchImageMetadata &image_meta);
  static std::vector<char> readEngineFile(const std::string &enginePath);
  static int roundToNearestMultiple(int val, int base = 32);
};
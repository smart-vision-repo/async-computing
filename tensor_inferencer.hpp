// In tensor_inferencer.hpp (example content)
#pragma once
#include "NvInfer.h" // For ILogger, IRuntime, ICudaEngine, IExecutionContext, Dims
#include "inference.hpp"
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Define your Detection struct (used by processOutput and NMS)
struct Detection {
  float x1, y1, x2, y2; // Coordinates (top-left x, top-left y, bottom-right x,
                        // bottom-right y)
  float confidence;     // Confidence score
  int class_id;         // Class ID
};

using InferenceCallback =
    std::function<void(const std::vector<InferenceResult> &)>;

class TensorInferencer {
public:
  TensorInferencer(int video_height, int video_width);
  ~TensorInferencer();

  // bool infer(const std::vector<float>& input, std::vector<float>& output); //
  // Original overload
  bool infer(const InferenceInput &input,
             InferenceCallback callback); // Main inference method

private:
  nvinfer1::IRuntime *runtime_ = nullptr;
  nvinfer1::ICudaEngine *engine_ = nullptr;
  nvinfer1::IExecutionContext *context_ = nullptr;

  // GPU Buffers - ensure these are managed correctly
  void *inputDevice_ = nullptr;  // Pointer for input data on GPU
  void *outputDevice_ = nullptr; // Pointer for output data on GPU
  std::vector<void *>
      bindings_; // Vector to hold pointers to input and output device buffers

  // Target dimensions for preprocessing
  int target_w_;
  int target_h_;

  // Binding indices
  int inputIndex_ = -1;
  int outputIndex_ = -1;

  // Buffer sizes (number of elements, not bytes)
  size_t inputSize_ = 0;  // Number of float elements for input
  size_t outputSize_ = 0; // Number of float elements for output

  // COCO class names and mapping
  std::map<std::string, int> class_name_to_id_;
  std::map<int, std::string> id_to_class_name_; // Added for convenience
  int num_classes_ = 0;                         // Store number of classes

  // Paths
  std::string engine_path_;
  std::string image_output_path_;

  // Helper methods
  void printEngineInfo();
  void processOutput(
      const InferenceInput &input, const std::vector<float> &host_output_raw,
      const cv::Mat &raw_img,
      const nvinfer1::Dims &outDimsRuntime); // Pass runtime output dims

  std::vector<Detection> applyNMS(const std::vector<Detection> &detections,
                                  float iou_threshold);
  float calculateIoU(const Detection &a, const Detection &b);
  void saveAnnotatedImage(const cv::Mat &raw_img, float x1_model,
                          float y1_model, float x2_model, float y2_model,
                          float confidence, const std::string &class_name,
                          int gopIdx, int detection_idx);

  // Deprecated or alternative infer, ensure it's consistent or removed if not
  // used
  bool infer(const std::vector<float> &input, std::vector<float> &output);
};
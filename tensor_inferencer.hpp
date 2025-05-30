#pragma once

#include "inference_input.hpp"
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

class TensorInferencer {
public:
  TensorInferencer(int video_height, int video_width);
  ~TensorInferencer();

  bool infer(const std::vector<float> &input, std::vector<float> &output);
  bool infer(const InferenceInput &input);

private:
  // TensorRT objects
  nvinfer1::IRuntime *runtime_ = nullptr;
  nvinfer1::ICudaEngine *engine_ = nullptr;
  nvinfer1::IExecutionContext *context_ = nullptr;

  // Device memory
  void *inputDevice_ = nullptr;
  void *outputDevice_ = nullptr;
  void *bindings_[2] = {nullptr, nullptr};

  // TensorRT binding info
  int inputIndex_ = -1;
  int outputIndex_ = -1;
  size_t inputSize_ = 0;
  size_t outputSize_ = 0;

  // Input image size (aligned)
  int target_w_ = 0;
  int target_h_ = 0;

  // Class name → class ID mapping (from coco.names)
  std::unordered_map<std::string, int> class_name_to_id_;

  void processOutput(const InferenceInput &input,
                     const std::vector<float> &host_output);
};

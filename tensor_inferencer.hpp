#pragma once

#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <unordered_map>
#include "inference_input.hpp"

class TensorInferencer {
public:
  TensorInferencer(int video_height, int video_width);
  ~TensorInferencer();

  bool infer(const std::vector<float>& input, std::vector<float>& output);
  bool infer(const InferenceInput& input);

private:
  // TensorRT engine
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;

  // GPU buffers
  void* inputDevice_ = nullptr;
  void* outputDevice_ = nullptr;
  void* bindings_[2] = {nullptr, nullptr};

  // tensor info
  int inputIndex_ = -1;
  int outputIndex_ = -1;
  size_t inputSize_ = 0;
  size_t outputSize_ = 0;

  // input size aligned
  int target_w_ = 0;
  int target_h_ = 0;

  std::unordered_map<std::string, int> class_name_to_id_;
  void processOutput(const InferenceInput& input, const std::vector<float>& host_output);
};
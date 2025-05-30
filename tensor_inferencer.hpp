// tensor_inferencer.hpp
#pragma once
#include "inference_input.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>

class TensorInferencer {
public:
  TensorInferencer();
  ~TensorInferencer();

  bool infer(const std::vector<float> &input, std::vector<float> &output);
  bool infer(const InferenceInput &input);

private:
  void processOutput(const InferenceInput &input,
                     const std::vector<float> &host_output);

  nvinfer1::IRuntime *runtime_ = nullptr;
  nvinfer1::ICudaEngine *engine_ = nullptr;
  nvinfer1::IExecutionContext *context_ = nullptr;

  void *bindings_[2] = {nullptr, nullptr};
  float *inputDevice_ = nullptr;
  float *outputDevice_ = nullptr;
  int inputIndex_ = -1;
  int outputIndex_ = -1;
  size_t inputSize_ = 0;
  size_t outputSize_ = static_cast<size_t>(8400 * 85);
};
// tensor_inferencer.hpp
#pragma once
#include "inference_input.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>

class TensorInferencer {
public:
  // 传入目标尺寸（建议构造前已对齐32倍）
  TensorInferencer(int target_h, int target_w);
  ~TensorInferencer();

  bool infer(const std::vector<float> &input, std::vector<float> &output);
  bool infer(const InferenceInput &input);

private:
  nvinfer1::IRuntime *runtime_{nullptr};
  nvinfer1::ICudaEngine *engine_{nullptr};
  nvinfer1::IExecutionContext *context_{nullptr};
  void *bindings_[2];
  int inputIndex_, outputIndex_;
  float *inputDevice_{nullptr};
  float *outputDevice_{nullptr};
  size_t inputSize_{0};
  size_t outputSize_{8400 * 85}; // 默认 YOLOv8 输出尺寸
  void processOutput(const InferenceInput &input,
                     const std::vector<float> &host_output);

  // 新增
  int target_h_ = 0;
  int target_w_ = 0;
};

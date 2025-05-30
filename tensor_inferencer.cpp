// tensor_inferencer.hpp
#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>

class TensorInferencer {
public:
  TensorInferencer(const std::string &enginePath);
  ~TensorInferencer();

  bool infer(const std::vector<float> &input, std::vector<float> &output);

private:
  nvinfer1::IRuntime *runtime_ = nullptr;
  nvinfer1::ICudaEngine *engine_ = nullptr;
  nvinfer1::IExecutionContext *context_ = nullptr;

  void *bindings_[2] = {nullptr, nullptr};
  float *inputDevice_ = nullptr;
  float *outputDevice_ = nullptr;
  int inputIndex_ = -1;
  int outputIndex_ = -1;
  size_t inputSize_ = static_cast<size_t>(1 * 3 * 640 * 640);
  size_t outputSize_ = static_cast<size_t>(8400 * 85);
};

// tensor_inferencer.cpp
#include "tensor_inferencer.hpp"
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace nvinfer1;

class Logger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[TRT] " << msg << std::endl;
    }
  }
} gLogger;

static std::vector<char> readEngineFile(const std::string &enginePath) {
  std::ifstream file(enginePath, std::ios::binary);
  assert(file.good());
  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> engineData(size);
  file.read(engineData.data(), size);
  file.close();
  return engineData;
}

TensorInferencer::TensorInferencer(const std::string &enginePath) {
  auto engineData = readEngineFile(enginePath);
  runtime_ = createInferRuntime(gLogger);
  assert(runtime_);
  engine_ =
      runtime_->deserializeCudaEngine(engineData.data(), engineData.size());
  assert(engine_);
  context_ = engine_->createExecutionContext();
  assert(context_);

  inputIndex_ = engine_->getBindingIndex("images");
  outputIndex_ = engine_->getBindingIndex(engine_->getBindingName(1));

  cudaMalloc(reinterpret_cast<void **>(&inputDevice_),
             inputSize_ * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&outputDevice_),
             outputSize_ * sizeof(float));
  bindings_[inputIndex_] = inputDevice_;
  bindings_[outputIndex_] = outputDevice_;
}

TensorInferencer::~TensorInferencer() {
  cudaFree(inputDevice_);
  cudaFree(outputDevice_);
  if (context_)
    context_->destroy();
  if (engine_)
    engine_->destroy();
  if (runtime_)
    runtime_->destroy();
}

bool TensorInferencer::infer(const std::vector<float> &input,
                             std::vector<float> &output) {
  if (input.size() != inputSize_)
    return false;
  output.resize(outputSize_);
  cudaMemcpy(inputDevice_, input.data(), inputSize_ * sizeof(float),
             cudaMemcpyHostToDevice);
  context_->enqueueV2(bindings_, 0, nullptr);
  cudaMemcpy(output.data(), outputDevice_, outputSize_ * sizeof(float),
             cudaMemcpyDeviceToHost);
  return true;
}

// main.cpp
#include "tensor_inferencer.hpp"
#include <iostream>

int main() {
  TensorInferencer infer("yolov8n.engine");
  std::vector<float> input(1 * 3 * 640 * 640, 0.0f);
  std::vector<float> output;

  if (infer.infer(input, output)) {
    std::cout << "[INFO] 推理成功，前几个输出：\n";
    for (int i = 0; i < 10; ++i) {
      std::cout << output[i] << " ";
    }
    std::cout << std::endl;
  } else {
    std::cerr << "[ERROR] 推理失败。" << std::endl;
  }

  return 0;
}
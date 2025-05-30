#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>

class TensorInferencer {
public:
    TensorInferencer(const std::string& enginePath);
    ~TensorInferencer();

    bool infer(const std::vector<float>& input, std::vector<float>& output);

private:
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    void* bindings_[2] = {nullptr, nullptr};
    float* inputDevice_ = nullptr;
    float* outputDevice_ = nullptr;
    int inputIndex_ = -1;
    int outputIndex_ = -1;
    int inputSize_ = 1 * 3 * 640 * 640;
    int outputSize_ = 8400 * 85;
};
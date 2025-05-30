#include <NvInfer.h>
#include <cassert>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace nvinfer1;

// Logger（用于打印 TensorRT 的日志）
class Logger : public ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kINFO)
      std::cout << "[TensorRT] " << msg << std::endl;
  }
};

static Logger logger;

std::vector<char> loadEngineFile(const std::string &engine_path) {
  std::ifstream file(engine_path, std::ios::binary);
  assert(file.good());
  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);
  file.close();
  return engine_data;
}

int main() {
  const char *env_engine_path = std::getenv("YOLO_ENGINE_NAME");
  if (!env_engine_path) {
    throw std::runtime_error("YOLO_ENGINE_NAME not set");
  }
  const std::string engine_path = env_engine_path;

  // 加载 engine
  auto engine_data = loadEngineFile(engine_path);
  auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(logger));
  auto engine = std::unique_ptr<ICudaEngine>(
      runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));

  assert(engine != nullptr);
  auto context =
      std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
  assert(context != nullptr);

  // 获取输入输出信息
  int inputIndex = engine->getBindingIndex("input"); // ONNX 中的 input 名字
  int outputIndex =
      engine->getBindingIndex(engine->getBindingName(1)); // 假设只有一个输出

  auto inputDims = engine->getBindingDimensions(inputIndex);
  auto outputDims = engine->getBindingDimensions(outputIndex);

  size_t inputSize = 1;
  for (int i = 0; i < inputDims.nbDims; ++i)
    inputSize *= inputDims.d[i];
  size_t outputSize = 1;
  for (int i = 0; i < outputDims.nbDims; ++i)
    outputSize *= outputDims.d[i];

  // 分配 host 和 device memory
  std::vector<float> inputHost(inputSize, 1.0f); // 模拟输入
  std::vector<float> outputHost(outputSize);

  float *inputDevice;
  float *outputDevice;
  cudaMalloc(&inputDevice, inputSize * sizeof(float));
  cudaMalloc(&outputDevice, outputSize * sizeof(float));

  cudaMemcpy(inputDevice, inputHost.data(), inputSize * sizeof(float),
             cudaMemcpyHostToDevice);

  // 设置绑定指针
  void *bindings[2];
  bindings[inputIndex] = inputDevice;
  bindings[outputIndex] = outputDevice;

  // 推理
  context->executeV2(bindings);

  // 拷贝结果回 CPU
  cudaMemcpy(outputHost.data(), outputDevice, outputSize * sizeof(float),
             cudaMemcpyDeviceToHost);

  std::cout << "Inference done. Output sample: " << outputHost[0] << std::endl;

  // 清理
  cudaFree(inputDevice);
  cudaFree(outputDevice);

  return 0;
}
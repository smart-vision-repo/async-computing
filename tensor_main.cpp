// main.cpp
#include "tensor_inferencer.hpp"
#include <iostream>

int main() {
  TensorInferencer infer("/opt/models/yolo/yolov8n.engine");
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
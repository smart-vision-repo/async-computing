#include "tensor_inferencer.hpp"
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "用法: " << argv[0] << " <图片路径> [confidence_thresh]"
              << std::endl;
    return 1;
  }

  const char *engine_env = std::getenv("YOLO_ENGINE_NAME");
  const char *names_env = std::getenv("YOLO_COCO_NAMES");
  const char *image_path = argv[1];

  if (!engine_env || !names_env) {
    std::cerr << "[ERROR] 环境变量 YOLO_ENGINE_NAME 和 YOLO_COCO_NAMES 必须设置"
              << std::endl;
    return 1;
  }

  float confidence_thresh = 0.3f;
  if (argc == 3) {
    confidence_thresh = std::stof(argv[2]);
  }

  std::cout << "[INFO] 使用置信度阈值: " << confidence_thresh << std::endl;

  cv::Mat image = cv::imread(image_path);
  if (image.empty()) {
    std::cerr << "[ERROR] 无法读取图片: " << image_path << std::endl;
    return 1;
  }

  TensorInferencer inferencer(1080, 1920); // 图像将统一缩放为 640x640

  InferenceInput input;
  input.decoded_frames.push_back(image);
  input.object_name = "dog";
  input.confidence_thresh = confidence_thresh;
  input.gopIdx = 0;

  bool success = inferencer.infer(input);
  if (!success) {
    std::cerr << "[ERROR] 推理失败" << std::endl;
    return 1;
  }

  std::cout << "[INFO] 推理完成。" << std::endl;
  return 0;
}

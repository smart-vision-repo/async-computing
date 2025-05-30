#include "tensor_inferencer.hpp"
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "用法: " << argv[0] << " <图片路径>" << std::endl;
    return 1;
  }

  const char *engine_env = std::getenv("YOLO_ENGINE_NAME");
  const char *names_env = std::getenv("YOLO_COCO_NAMES");
  const char *image_path = argv[1];

  if (!engine_env || !names_env) {
    std::cerr << "环境变量 YOLO_ENGINE_NAME 和 YOLO_COCO_NAMES 必须设置"
              << std::endl;
    return 1;
  }

  // 读取图像
  cv::Mat image = cv::imread(image_path);
  if (image.empty()) {
    std::cerr << "无法读取图片: " << image_path << std::endl;
    return 1;
  }

  std::cout << "[INFO] 加载图片成功: " << image.cols << "x" << image.rows
            << std::endl;

  // 构造推理器（目标输入尺寸设为 640x640）
  TensorInferencer inferencer(1080,
                              1920); // 实际输入尺寸可以忽略，640x640 是固定的

  // 构造推理输入
  InferenceInput input;
  input.decoded_frames.push_back(image);
  input.object_name = "dog";
  input.confidence_thresh = 0.6f;
  input.gopIdx = 0;

  // 执行推理
  bool success = inferencer.infer(input);
  if (!success) {
    std::cerr << "[ERROR] 推理失败" << std::endl;
    return 1;
  }

  std::cout << "[INFO] 推理完成。" << std::endl;
  return 0;
}

#include "tensor_inferencer.hpp"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <unordered_map>

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

static int roundToNearestMultiple(int val, int base = 32) {
  return ((val + base / 2) / base) * base;
}

TensorInferencer::TensorInferencer(int video_height, int video_width) {
  target_w_ = roundToNearestMultiple(video_width, 32);
  target_h_ = roundToNearestMultiple(video_height, 32);

  std::cout << "[INFO] Adjusted input size: " << target_w_ << "x" << target_h_
            << " (32-aligned)" << std::endl;

  const char *env_path = std::getenv("YOLO_ENGINE_NAME");
  if (!env_path) {
    std::cerr << "[ERROR] Environment variable YOLO_ENGINE_NAME not set."
              << std::endl;
    std::exit(1);
  }
  std::string enginePath = env_path;

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

  for (int i = 0; i < 2; ++i)
    bindings_[i] = nullptr;

  // Load class name -> ID map
  const char *names_path = std::getenv("YOLO_COCO_NAMES");
  if (!names_path) {
    std::cerr << "[ERROR] Environment variable YOLO_COCO_NAMES not set."
              << std::endl;
    std::exit(1);
  }
  std::ifstream names_file(names_path);
  if (!names_file.is_open()) {
    std::cerr << "[ERROR] Failed to open " << names_path << std::endl;
    std::exit(1);
  }
  std::string line;
  int idx = 0;
  while (std::getline(names_file, line)) {
    if (!line.empty()) {
      class_name_to_id_[line] = idx++;
    }
  }
}

void TensorInferencer::processOutput(const InferenceInput &input,
                                     const std::vector<float> &host_output) {
  const int box_step = 85;
  const int num_classes = 80;
  int num_boxes = static_cast<int>(host_output.size() / box_step);

  for (int i = 0; i < num_boxes; ++i) {
    const float *det = &host_output[i * box_step];
    float objectness = 1.0f / (1.0f + std::exp(-det[4]));

    if (!std::isfinite(objectness) || objectness < 0.0f || objectness > 1.0f) {
      std::cerr << "[ERROR] Invalid objectness: " << det[4] << " -> "
                << objectness << std::endl;
      continue;
    }

    int class_id = -1;
    float max_score = 0.0f;
    for (int j = 0; j < num_classes; ++j) {
      float cls_score = 1.0f / (1.0f + std::exp(-det[5 + j]));
      if (!std::isfinite(cls_score))
        continue;
      if (cls_score > max_score) {
        max_score = cls_score;
        class_id = j;
      }
    }

    float confidence = objectness * max_score;
    if (confidence < input.confidence_thresh)
      continue;

    auto it = class_name_to_id_.find(input.object_name);
    if (it != class_name_to_id_.end() && class_id == it->second) {
      float cx = det[0], cy = det[1], w = det[2], h = det[3];
      float x1 = cx - w / 2, y1 = cy - h / 2;
      float x2 = cx + w / 2, y2 = cy + h / 2;

      std::cout << "[YOLO] GOP: " << input.gopIdx
                << ", Confidence: " << confidence
                << ", Class: " << input.object_name << ", Box: (" << x1 << ","
                << y1 << "," << x2 << "," << y2 << ")\n";
    }
  }
}

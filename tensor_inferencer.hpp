#pragma once

#include "inference_input.hpp"
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

struct Detection {
  float x1, y1, x2, y2;
  float confidence;
  int class_id;
};

class TensorInferencer {
public:
  TensorInferencer(int video_height, int video_width);
  ~TensorInferencer();

  bool infer(const std::vector<float> &input, std::vector<float> &output);
  bool infer(const InferenceInput &input);

private:
  // TensorRT runtime components
  nvinfer1::IRuntime *runtime_ = nullptr;
  nvinfer1::ICudaEngine *engine_ = nullptr;
  nvinfer1::IExecutionContext *context_ = nullptr;

  // GPU memory buffers
  void *inputDevice_ = nullptr;
  void *outputDevice_ = nullptr;
  void *bindings_[2] = {nullptr, nullptr};

  // Binding info
  int inputIndex_ = -1;
  int outputIndex_ = -1;
  size_t inputSize_ = 0;
  size_t outputSize_ = 0;

  // Input image dimensions (aligned)
  int target_w_ = 640;
  int target_h_ = 640;

  // Class name to ID mapping (loaded from YOLO_COCO_NAMES)
  std::unordered_map<std::string, int> class_name_to_id_;

  // Image saving path (loaded from YOLO_IMAGE_PATH env)
  std::string image_output_path_;

  std::vector<Detection> applyNMS(const std::vector<Detection> &detections,
                                  float iou_threshold);

  void processOutput(const InferenceInput &input,
                     const std::vector<float> &host_output,
                     const cv::Mat &raw_img);
  void saveAnnotatedImage(const cv::Mat &raw_img, float x1, float y1, float x2,
                          float y2, float confidence,
                          const std::string &class_name, int gopIdx,
                          int detection_idx);
  void printEngineInfo();
  float calculateIoU(const Detection &a, const Detection &b);
};
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

class YoloInferencer {
public:
  struct InferenceInput {
    std::vector<cv::Mat> decoded_frames;
    std::string object_name;
    float confidence_thresh;
    int gopIdx;
  };

  YoloInferencer();
  ~YoloInferencer();

  void infer(const InferenceInput &input);
  void waitForAllTasks(); // 占位函数，当前为空实现

private:
  struct InferenceTask {
    std::vector<cv::Mat> frames;
    std::string object_name;
    float confidence_thresh;
    int gopIdx;
  };

  void loadClassNamesFromEnv();
  void loadModelFromEnv();
  void doInference(const InferenceTask &task);

  cv::dnn::Net net;
  std::vector<std::string> class_names;
  int num_classes = 0;
  bool initialized = false;
};

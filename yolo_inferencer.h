#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>

struct InferenceInput {
  std::vector<cv::Mat> decoded_frames;
  std::string object_name;
  float confidence_thresh;
  int gopIdx;
};

class YoloInferencer {
public:
  YoloInferencer();
  ~YoloInferencer();

  void infer(const InferenceInput &input);
  void waitForAllTasks(); // 占位，已无实际功能

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

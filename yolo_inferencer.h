#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
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

private:
  struct InferenceTask {
    std::vector<cv::Mat> frames;
    std::string object_name;
    float confidence_thresh;
    int gopIdx;
  };

  void loadClassNamesFromEnv();
  void loadModelFromEnv();
  void processLoop();

  cv::dnn::Net net;
  bool initialized = false;
  bool running = false;
  cv::Size input_size = cv::Size(640, 640);
  std::vector<std::string> class_names;
  int num_classes = 0;
};
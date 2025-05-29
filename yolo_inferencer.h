// yolo_inferencer.h
#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <opencv2/dnn.hpp>
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
  cv::dnn::Net net;
  std::vector<std::string> class_names;
  int num_classes = 0;
  cv::Size input_size = {640, 640};
  bool initialized = false;

  struct InferenceTask {
    std::vector<cv::Mat> frames;
    std::string object_name;
    float confidence_thresh;
    int gopIdx;
  };

  std::queue<InferenceTask> task_queue;
  std::mutex queue_mutex;
  std::condition_variable cv_task;
  std::thread worker_thread;
  std::atomic<bool> running;
  cv::Size input_size = cv::Size(640, 640);

  void loadClassNamesFromEnv();
  void loadModelFromEnv();
  void processLoop();
  void doInference(const InferenceTask &task);

  cv::Mat letterbox(const cv::Mat &src, const cv::Size &target_size,
                    int stride = 32,
                    cv::Scalar color = cv::Scalar(114, 114, 114),
                    float *scale_out = nullptr,
                    cv::Point *padding_out = nullptr);
};

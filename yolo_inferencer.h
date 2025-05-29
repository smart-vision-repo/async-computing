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
  void waitForAllTasks();  // 等待所有任务完成

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
  void doInference(const InferenceTask &task);

  cv::dnn::Net net;
  bool initialized = false;
  bool running = false;
  cv::Size input_size = cv::Size(640, 640);
  std::vector<std::string> class_names;
  int num_classes = 0;

  std::queue<InferenceTask> task_queue;
  std::mutex queue_mutex;
  std::condition_variable cv_task;
  std::condition_variable done_cv;

  std::thread worker_thread;
  std::atomic<int> active_tasks = 0;

  cv::Mat letterbox(const cv::Mat &src, const cv::Size &target_size, int stride,
                    cv::Scalar color, float *scale_out = nullptr,
                    cv::Point *padding_out = nullptr);
};
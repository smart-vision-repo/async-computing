#include "yolo_inferencer.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

YoloInferencer::YoloInferencer() {
  try {
    loadClassNamesFromEnv();
    loadModelFromEnv();
    initialized = true;
    running = true;
    worker_thread = std::thread(&YoloInferencer::processLoop, this);
  } catch (const std::exception &e) {
    std::cerr << "YoloInferencer init failed: " << e.what() << std::endl;
    initialized = false;
  }
}

YoloInferencer::~YoloInferencer() {
  running = false;
  cv_task.notify_one();
  if (worker_thread.joinable()) {
    worker_thread.join();
  }
}

void YoloInferencer::loadClassNamesFromEnv() {
  const char *names_path = std::getenv("YOLO_COCO_NAMES");
  if (!names_path)
    throw std::runtime_error("YOLO_COCO_NAMES not set");

  std::ifstream ifs(names_path);
  if (!ifs.is_open())
    throw std::runtime_error("Cannot open coco.names file");

  std::string line;
  while (std::getline(ifs, line)) {
    class_names.push_back(line);
  }

  num_classes = static_cast<int>(class_names.size());
}

void YoloInferencer::loadModelFromEnv() {
  const char *model_path = std::getenv("YOLO_MODEL_NAME");
  if (!model_path)
    throw std::runtime_error("YOLO_MODEL_NAME not set");

  net = readNetFromONNX(model_path);

  try {
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);
    std::cout << "[INFO] Using CUDA backend for inference." << std::endl;
  } catch (...) {
    std::cerr << "[WARN] CUDA backend unavailable, falling back to CPU."
              << std::endl;
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
  }
}

cv::Mat YoloInferencer::letterbox(const cv::Mat &src,
                                  const cv::Size &target_size, int stride,
                                  cv::Scalar color, float *scale_out,
                                  cv::Point *padding_out) {
  int src_w = src.cols;
  int src_h = src.rows;
  float scale = std::min((float)target_size.width / src_w,
                         (float)target_size.height / src_h);
  int new_w = int(round(src_w * scale));
  int new_h = int(round(src_h * scale));

  int pad_w = target_size.width - new_w;
  int pad_h = target_size.height - new_h;
  int pad_left = pad_w / 2;
  int pad_top = pad_h / 2;

  cv::Mat resized;
  cv::resize(src, resized, cv::Size(new_w, new_h));

  cv::Mat output;
  cv::copyMakeBorder(resized, output, pad_top, pad_h - pad_top, pad_left,
                     pad_w - pad_left, cv::BORDER_CONSTANT, color);

  if (scale_out)
    *scale_out = scale;
  if (padding_out)
    *padding_out = cv::Point(pad_left, pad_top);

  return output;
}

void YoloInferencer::infer(const InferenceInput &input) {
  if (!initialized)
    return;
  InferenceTask task{input.decoded_frames, input.object_name,
                     input.confidence_thresh, input.gopIdx};
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    task_queue.push(std::move(task));
    ++active_tasks;
  }
  cv_task.notify_one();
}

void YoloInferencer::processLoop() {
  while (running) {
    InferenceTask task;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      cv_task.wait(lock, [&] { return !task_queue.empty() || !running; });
      if (!running && task_queue.empty())
        break;
      if (!task_queue.empty()) {
        task = std::move(task_queue.front());
        task_queue.pop();
      } else {
        continue;
      }
    }

    doInference(task);

    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      --active_tasks;
      if (active_tasks == 0 && task_queue.empty()) {
        done_cv.notify_all();
      }
    }
  }
}

void YoloInferencer::waitForAllTasks() {
  std::unique_lock<std::mutex> lock(queue_mutex);
  done_cv.wait(lock, [&]() { return active_tasks == 0 && task_queue.empty(); });
}
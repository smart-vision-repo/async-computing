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
  cv_task.notify_all(); // 使用 notify_all 而不是 notify_one

  if (worker_thread.joinable()) {
    // 添加超时等待
    auto timeout = std::chrono::seconds(5);
    if (worker_thread.joinable()) {
      std::thread killer([&]() {
        std::this_thread::sleep_for(timeout);
        if (worker_thread.joinable()) {
          worker_thread.detach(); // 强制分离
        }
      });
      killer.detach();
      worker_thread.join();
    }
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

int YoloInferencer::infer1(const InferenceInput &input) {
  if (!initialized)
    return;
  InferenceTask task{input.decoded_frames, input.object_name,
                     input.confidence_thresh, input.gopIdx};
  doInference(task);
  return 0;
  // {
  //   std::lock_guard<std::mutex> lock(queue_mutex);
  //   task_queue.push(std::move(task));
  //   ++active_tasks;
  // }
  // cv_task.notify_one();
}

void YoloInferencer::infer(const InferenceInput &input) {
  if (!initialized)
    return;
  InferenceTask task{input.decoded_frames, input.object_name,
                     input.confidence_thresh, input.gopIdx};
  doInference(task);
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

void YoloInferencer::doInference(const YoloInferencer::InferenceTask &task) {
  auto sigmoid = [](float x) { return 1.f / (1.f + std::exp(-x)); };

  for (size_t frame_idx = 0; frame_idx < task.frames.size(); ++frame_idx) {
    const Mat &frame = task.frames[frame_idx];

    float scale = 1.0f;
    Point pad;
    Mat padded =
        letterbox(frame, input_size, 32, Scalar(114, 114, 114), &scale, &pad);

    Mat blob;
    blobFromImage(padded, blob, 1.0 / 255.0, input_size, Scalar(), true, false);
    net.setInput(blob);
    Mat output = net.forward();

    const int num_preds = output.size[2];
    const int num_attrs = output.size[1];

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < num_preds; ++i) {
      float *data = output.ptr<float>(0, 0) + i * num_attrs;

      float objectness = sigmoid(data[4]);

      float max_class_score = -1e9;
      int class_id = -1;
      for (int c = 0; c < num_classes; ++c) {
        float cls_score = data[5 + c];
        if (cls_score > max_class_score) {
          max_class_score = cls_score;
          class_id = c;
        }
      }

      float class_score = sigmoid(max_class_score);
      float final_conf = objectness * class_score;

      if (final_conf >= task.confidence_thresh && class_id >= 0 &&
          class_id < static_cast<int>(class_names.size()) &&
          class_names[class_id] == task.object_name) {

        float center_x = data[0] * input_size.width;
        float center_y = data[1] * input_size.height;
        float width = data[2] * input_size.width;
        float height = data[3] * input_size.height;

        float x = center_x - width / 2;
        float y = center_y - height / 2;

        boxes.emplace_back(static_cast<int>(x), static_cast<int>(y),
                           static_cast<int>(width), static_cast<int>(height));
        confidences.push_back(final_conf);
        class_ids.push_back(class_id);
      }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, task.confidence_thresh, 0.4f,
                      indices);

    if (!indices.empty()) {
      const char *save_path = std::getenv("YOLO_IMAGE_PATH");
      if (save_path) {
        Mat result_img = frame.clone();
        for (int idx : indices) {
          cv::Rect detection_box = boxes[idx];

          float x_orig = (detection_box.x - pad.x) / scale;
          float y_orig = (detection_box.y - pad.y) / scale;
          float w_orig = detection_box.width / scale;
          float h_orig = detection_box.height / scale;

          x_orig = std::max(0.0f, std::min(x_orig, (float)frame.cols));
          y_orig = std::max(0.0f, std::min(y_orig, (float)frame.rows));
          w_orig = std::min(w_orig, (float)frame.cols - x_orig);
          h_orig = std::min(h_orig, (float)frame.rows - y_orig);

          cv::Rect orig_box(static_cast<int>(x_orig), static_cast<int>(y_orig),
                            static_cast<int>(w_orig), static_cast<int>(h_orig));

          cv::rectangle(result_img, orig_box, cv::Scalar(0, 255, 0), 2);

          std::string label =
              class_names[class_ids[idx]] + ": " +
              std::to_string(static_cast<int>(confidences[idx] * 100)) + "%";

          int baseline = 0;
          cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                               0.5, 1, &baseline);

          cv::rectangle(
              result_img,
              cv::Point(orig_box.x, orig_box.y - text_size.height - 5),
              cv::Point(orig_box.x + text_size.width, orig_box.y),
              cv::Scalar(0, 255, 0), -1);

          cv::putText(result_img, label, cv::Point(orig_box.x, orig_box.y - 5),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        std::string save_filename = std::string(save_path) + "/detection_gop" +
                                    std::to_string(task.gopIdx) + "_frame_" +
                                    std::to_string(frame_idx) + ".jpg";

        cv::imwrite(save_filename, result_img);
      }
    }

    for (int idx : indices) {
      std::cout << "[YOLO] GOP: " << task.gopIdx << ", Frame: " << frame_idx
                << ", Confidence: " << confidences[idx]
                << ", Class: " << class_names[class_ids[idx]] << ", Box: ("
                << boxes[idx].x << "," << boxes[idx].y << ","
                << boxes[idx].width << "," << boxes[idx].height << ")"
                << std::endl;
    }
  }
}

void YoloInferencer::waitForAllTasks() {
  std::unique_lock<std::mutex> lock(queue_mutex);
  done_cv.wait(lock, [&]() { return active_tasks == 0 && task_queue.empty(); });
}
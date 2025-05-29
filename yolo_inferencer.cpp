#include "yolo_inferencer.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

YoloInferencer::YoloInferencer() {
  try {
    loadClassNamesFromEnv();
    loadModelFromEnv();
    initialized = true;
  } catch (const std::exception &e) {
    std::cerr << "YoloInferencer init failed: " << e.what() << std::endl;
    initialized = false;
  }
}

YoloInferencer::~YoloInferencer() {}

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

static cv::Mat letterbox(const cv::Mat &src, cv::Size &out_size, int stride,
                         float *scale_out, cv::Point *pad_out) {
  int width = src.cols;
  int height = src.rows;
  float scale = 1.0f;

  // 计算使宽高都能被stride整除的最小尺寸
  int new_w = (width + stride - 1) / stride * stride;
  int new_h = (height + stride - 1) / stride * stride;

  scale = std::min((float)new_w / width, (float)new_h / height);
  int resized_w = int(width * scale);
  int resized_h = int(height * scale);

  int dw = new_w - resized_w;
  int dh = new_h - resized_h;
  int top = dh / 2;
  int bottom = dh - top;
  int left = dw / 2;
  int right = dw - left;

  cv::Mat resized;
  resize(src, resized, Size(resized_w, resized_h));

  cv::Mat padded;
  copyMakeBorder(resized, padded, top, bottom, left, right, BORDER_CONSTANT,
                 Scalar(114, 114, 114));

  if (scale_out)
    *scale_out = scale;
  if (pad_out)
    *pad_out = Point(left, top);
  out_size = padded.size();

  return padded;
}

void YoloInferencer::infer(const InferenceInput &input) {
  if (!initialized)
    return;
  InferenceTask task{input.decoded_frames, input.object_name,
                     input.confidence_thresh, input.gopIdx};
  doInference(task);
}

void YoloInferencer::doInference(const YoloInferencer::InferenceTask &task) {
  auto sigmoid = [](float x) { return 1.f / (1.f + std::exp(-x)); };

  for (size_t frame_idx = 0; frame_idx < task.frames.size(); ++frame_idx) {
    const Mat &frame = task.frames[frame_idx];

    float scale;
    Point pad;
    Size input_size;
    Mat padded = letterbox(frame, input_size, 32, &scale, &pad);

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

        // 将坐标映射回原图
        x = (x - pad.x) / scale;
        y = (y - pad.y) / scale;
        width /= scale;
        height /= scale;

        x = std::max(0.f, std::min(x, (float)frame.cols - 1));
        y = std::max(0.f, std::min(y, (float)frame.rows - 1));
        width = std::min(width, (float)frame.cols - x);
        height = std::min(height, (float)frame.rows - y);

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
          cv::Rect box = boxes[idx];
          cv::rectangle(result_img, box, cv::Scalar(0, 255, 0), 2);

          std::string label =
              class_names[class_ids[idx]] + ": " +
              std::to_string(static_cast<int>(confidences[idx] * 100)) + "%";

          int baseline = 0;
          cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                               0.5, 1, &baseline);

          cv::rectangle(result_img,
                        cv::Point(box.x, box.y - text_size.height - 5),
                        cv::Point(box.x + text_size.width, box.y),
                        cv::Scalar(0, 255, 0), -1);

          cv::putText(result_img, label, cv::Point(box.x, box.y - 5),
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
  // No longer needed
}
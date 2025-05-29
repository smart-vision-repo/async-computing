// yolo_inferencer.cpp
#include "yolo_inferencer.h"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>

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
    } catch (const std::exception& e) {
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
    const char* names_path = std::getenv("YOLO_COCO_NAMES");
    if (!names_path) throw std::runtime_error("YOLO_COCO_NAMES not set");

    std::ifstream ifs(names_path);
    if (!ifs.is_open()) throw std::runtime_error("Cannot open coco.names file");

    std::string line;
    while (std::getline(ifs, line)) {
        class_names.push_back(line);
    }

    num_classes = static_cast<int>(class_names.size());
}

void YoloInferencer::loadModelFromEnv() {
    const char* model_path = std::getenv("YOLO_MODEL_NAME");
    if (!model_path) throw std::runtime_error("YOLO_MODEL_NAME not set");

    net = readNetFromONNX(model_path);

    try {
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
        std::cout << "[INFO] Using CUDA backend for inference." << std::endl;
    } catch (...) {
        std::cerr << "[WARN] CUDA backend unavailable, falling back to CPU." << std::endl;
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }
}

cv::Mat YoloInferencer::letterbox(const cv::Mat& src, const cv::Size& target_size, int stride,
                                  cv::Scalar color, float* scale_out, cv::Point* padding_out)
{
    int src_w = src.cols;
    int src_h = src.rows;
    float scale = std::min((float)target_size.width / src_w, (float)target_size.height / src_h);
    int new_w = int(round(src_w * scale));
    int new_h = int(round(src_h * scale));

    int pad_w = target_size.width - new_w;
    int pad_h = target_size.height - new_h;
    int pad_left = pad_w / 2;
    int pad_top = pad_h / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    cv::Mat output;
    cv::copyMakeBorder(resized, output, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
                       cv::BORDER_CONSTANT, color);

    if (scale_out) *scale_out = scale;
    if (padding_out) *padding_out = cv::Point(pad_left, pad_top);

    return output;
}

void YoloInferencer::infer(const std::vector<cv::Mat>& decoded_frames,
                           const std::string& object_name,
                           float confidence_thresh)
{
    if (!initialized) return;
    InferenceTask task{decoded_frames, object_name, confidence_thresh};
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        task_queue.push(std::move(task));
    }
    cv_task.notify_one();
}

void YoloInferencer::processLoop() {
    while (running) {
        InferenceTask task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            cv_task.wait(lock, [&] { return !task_queue.empty() || !running; });
            if (!running) break;
            task = std::move(task_queue.front());
            task_queue.pop();
        }
        doInference(task);
    }
}

void YoloInferencer::doInference(const InferenceTask& task) {
    auto sigmoid = [](float x) {
        return 1.f / (1.f + std::exp(-x));
    };

    for (size_t frame_idx = 0; frame_idx < task.frames.size(); ++frame_idx) {
        const Mat& frame = task.frames[frame_idx];

        float scale = 1.0f;
        Point pad;
        Mat padded = letterbox(frame, input_size, 32, Scalar(114, 114, 114), &scale, &pad);

        Mat blob;
        blobFromImage(padded, blob, 1.0 / 255.0, input_size, Scalar(), true, false);
        net.setInput(blob);
        Mat output = net.forward();

        const int rows = output.size[1];
        for (int i = 0; i < rows; ++i) {
            float* data = output.ptr<float>(0, i);
            float obj_conf = sigmoid(data[4]);

            float max_score = 0.0f;
            int class_id = -1;
            for (int c = 5; c < 5 + num_classes; ++c) {
                float cls_score = data[c];  // 不再使用 sigmoid
                if (cls_score > max_score) {
                    max_score = cls_score;
                    class_id = c - 5;
                }
            }

            float final_conf = obj_conf * max_score;
            if (final_conf >= task.confidence_thresh &&
                class_id >= 0 &&
                class_id < static_cast<int>(class_names.size()) &&
                class_names[class_id] == task.object_name)
            {
                std::cout << "[YOLO] Frame: " << frame_idx
                          << ", Confidence: " << final_conf
                          << ", Class: " << class_names[class_id] << std::endl;
            }
        }
    }
}

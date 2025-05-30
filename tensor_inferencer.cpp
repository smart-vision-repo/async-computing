#include "tensor_inferencer.hpp"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace nvinfer1;

class Logger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[TRT] " << msg << std::endl;
    }
  }
} gLogger;

static std::vector<char> readEngineFile(const std::string &enginePath) {
  std::ifstream file(enginePath, std::ios::binary);
  assert(file.good());
  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> engineData(size);
  file.read(engineData.data(), size);
  file.close();
  return engineData;
}

static int roundToNearestMultiple(int val, int base = 32) {
  return ((val + base / 2) / base) * base;
}

TensorInferencer::TensorInferencer(int video_height, int video_width) {
  const char *image_path_env = std::getenv("YOLO_IMAGE_PATH");
  if (image_path_env) {
    image_output_path_ = image_path_env;
  }
  target_w_ = roundToNearestMultiple(video_width, 32);
  target_h_ = roundToNearestMultiple(video_height, 32);

  std::cout << "[INFO] Adjusted input size: " << target_w_ << "x" << target_h_
            << " (32-aligned)" << std::endl;

  const char *env_path = std::getenv("YOLO_ENGINE_NAME");
  if (!env_path) {
    std::cerr << "[ERROR] Environment variable YOLO_ENGINE_NAME not set."
              << std::endl;
    std::exit(1);
  }
  std::string enginePath = env_path;

  auto engineData = readEngineFile(enginePath);
  runtime_ = createInferRuntime(gLogger);
  assert(runtime_);
  engine_ =
      runtime_->deserializeCudaEngine(engineData.data(), engineData.size());
  assert(engine_);
  context_ = engine_->createExecutionContext();
  assert(context_);

  inputIndex_ = engine_->getBindingIndex("images");
  outputIndex_ = engine_->getBindingIndex(engine_->getBindingName(1));

  const char *names_path = std::getenv("YOLO_COCO_NAMES");
  if (!names_path) {
    std::cerr << "[ERROR] Environment variable YOLO_COCO_NAMES not set."
              << std::endl;
    std::exit(1);
  }
  std::ifstream infile(names_path);
  if (!infile.is_open()) {
    std::cerr << "[ERROR] Failed to open file: " << names_path << std::endl;
    std::exit(1);
  }
  std::string line;
  int idx = 0;
  while (std::getline(infile, line)) {
    if (!line.empty()) {
      class_name_to_id_[line] = idx++;
    }
  }
}

TensorInferencer::~TensorInferencer() {
  cudaFree(inputDevice_);
  cudaFree(outputDevice_);
  if (context_)
    context_->destroy();
  if (engine_)
    engine_->destroy();
  if (runtime_)
    runtime_->destroy();
}

bool TensorInferencer::infer(const std::vector<float> &input,
                             std::vector<float> &output) {
  if (inputSize_ == 0 || outputSize_ == 0) {
    std::cerr << "[ERROR] inputSize/outputSize 未初始化" << std::endl;
    return false;
  }
  if (input.size() != inputSize_)
    return false;
  output.resize(outputSize_);
  cudaMemcpy(inputDevice_, input.data(), inputSize_ * sizeof(float),
             cudaMemcpyHostToDevice);
  context_->enqueueV2(bindings_, 0, nullptr);
  cudaMemcpy(output.data(), outputDevice_, outputSize_ * sizeof(float),
             cudaMemcpyDeviceToHost);
  return true;
}

bool TensorInferencer::infer(const InferenceInput &input) {
  if (input.decoded_frames.empty())
    return false;
  const cv::Mat &raw_img = input.decoded_frames[0];
  if (raw_img.empty())
    return false;

  cv::Mat img;
  cv::resize(raw_img, img, cv::Size(target_w_, target_h_));
  int c = 3, h = target_h_, w = target_w_;
  inputSize_ = static_cast<size_t>(c * h * w);

  cv::Mat chw_input;
  img.convertTo(chw_input, CV_32FC3, 1.0 / 255.0);
  std::vector<float> input_data(inputSize_);
  for (int i = 0; i < c; ++i)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        input_data[i * h * w + y * w + x] = chw_input.at<cv::Vec3f>(y, x)[i];

  Dims inputDims{4, {1, 3, h, w}};
  if (!context_->setBindingDimensions(inputIndex_, inputDims)) {
    std::cerr << "[ERROR] Failed to set binding dimensions." << std::endl;
    return false;
  }
  if (!context_->allInputDimensionsSpecified()) {
    std::cerr << "[ERROR] Not all input dimensions specified after setting."
              << std::endl;
    return false;
  }

  if (outputSize_ == 0) {
    Dims outDims = context_->getBindingDimensions(outputIndex_);
    outputSize_ = 1;
    for (int i = 0; i < outDims.nbDims; ++i) {
      if (outDims.d[i] <= 0)
        outDims.d[i] = 1;
      outputSize_ *= outDims.d[i];
    }
    std::cout << "[INFO] Inferred output size: " << outputSize_ << std::endl;
    if (outputSize_ % 85 != 0) {
      std::cerr << "[WARNING] Output size " << outputSize_
                << " is not divisible by 85." << std::endl;
    }
  }

  if (inputDevice_)
    cudaFree(inputDevice_);
  if (outputDevice_)
    cudaFree(outputDevice_);
  cudaMalloc(reinterpret_cast<void **>(&inputDevice_),
             inputSize_ * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&outputDevice_),
             outputSize_ * sizeof(float));
  bindings_[inputIndex_] = inputDevice_;
  bindings_[outputIndex_] = outputDevice_;

  std::cout << "[DEBUG] inputSize_: " << inputSize_
            << ", outputSize_: " << outputSize_ << std::endl;

  cudaMemcpy(inputDevice_, input_data.data(), inputSize_ * sizeof(float),
             cudaMemcpyHostToDevice);
  if (!context_->enqueueV2(bindings_, 0, nullptr))
    return false;

  std::vector<float> host_output(outputSize_);
  cudaMemcpy(host_output.data(), outputDevice_, outputSize_ * sizeof(float),
             cudaMemcpyDeviceToHost);
  processOutput(input, host_output);
  return true;
}

void TensorInferencer::processOutput(const InferenceInput &input,
                                     const std::vector<float> &host_output) {
  const bool save_image = !image_output_path_.empty();
  int box_step = 85;
  int num_classes = 80;
  int num_boxes = static_cast<int>(host_output.size() / box_step);

  auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };

  for (int i = 0; i < num_boxes; ++i) {
    const float *det = &host_output[i * box_step];
    float objectness = sigmoid(det[4]);
    if (objectness < input.confidence_thresh)
      continue;

    int class_id = -1;
    float max_score = 0.0f;
    for (int j = 0; j < num_classes; ++j) {
      float cls_score = sigmoid(det[5 + j]);
      if (cls_score > max_score) {
        max_score = cls_score;
        class_id = j;
      }
    }

    float confidence = objectness * max_score;
    if (confidence < input.confidence_thresh)
      continue;

    auto it = class_name_to_id_.find(input.object_name);
    if (it != class_name_to_id_.end() && class_id == it->second) {
      float cx = det[0], cy = det[1], w = det[2], h = det[3];
      float x1 = cx - w / 2, y1 = cy - h / 2;
      float x2 = cx + w / 2, y2 = cy + h / 2;

      std::cout << "[YOLO] GOP: " << input.gopIdx
                << ", Confidence: " << confidence
                << ", Class: " << input.object_name << ", Box: (" << x1 << ","
                << y1 << "," << x2 << "," << y2
                << ")
                   ";

          if (save_image && !input.decoded_frames.empty()) {
        cv::Mat vis_img = input.decoded_frames[0].clone();
        int orig_w = vis_img.cols, orig_h = vis_img.rows;

        // Rescale box coordinates back to original image size
        float scale_x = static_cast<float>(orig_w) / target_w_;
        float scale_y = static_cast<float>(orig_h) / target_h_;
        cv::Rect box_rect(
            static_cast<int>(x1 * scale_x), static_cast<int>(y1 * scale_y),
            static_cast<int>(w * scale_x), static_cast<int>(h * scale_y));
        cv::rectangle(vis_img, box_rect, cv::Scalar(0, 255, 0), 2);

        char filename[512];
        snprintf(filename, sizeof(filename), "%s/detection_gop%d.jpg",
                 image_output_path_.c_str(), input.gopIdx);
        cv::imwrite(filename, vis_img);
      }
    }
  }
}

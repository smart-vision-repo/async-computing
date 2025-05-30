#include "tensor_inferencer.hpp"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
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

// static int roundToNearestMultiple(int val, int base = 32) {
//   return ((val + base / 2) / base) * base;
// }

TensorInferencer::TensorInferencer(int video_height, int video_width) {
  // target_w_ = roundToNearestMultiple(video_width, 32);
  // target_h_ = roundToNearestMultiple(video_height, 32);

  std::cout << "[INFO] Adjusted input size: " << target_w_ << "x" << target_h_
            << " (32-aligned)" << std::endl;

  const char *env_path = std::getenv("YOLO_ENGINE_NAME");
  if (!env_path) {
    std::cerr << "[ERROR] Environment variable YOLO_ENGINE_NAME not set."
              << std::endl;
    std::exit(1);
  }

  const char *names_path = std::getenv("YOLO_COCO_NAMES");
  if (!names_path) {
    std::cerr << "[ERROR] Environment variable YOLO_COCO_NAMES not set."
              << std::endl;
    std::exit(1);
  }

  const char *output_path_env = std::getenv("YOLO_IMAGE_PATH");
  if (!output_path_env) {
    std::cerr << "[ERROR] Environment variable YOLO_IMAGE_PATH not set."
              << std::endl;
    std::exit(1);
  }
  image_output_path_ = output_path_env;

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

  std::cout << "dog -> class_id: " << class_name_to_id_["dog"] << std::endl;
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

bool TensorInferencer::infer(InferenceInput &input) {
  input.object_name = "fire hydrant";
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

  // Dims inputDims{4, {1, 3, h, w}};
  // if (!context_->setBindingDimensions(inputIndex_, inputDims)) {
  // std::cerr << "[ERROR] Failed to set binding dimensions." << std::endl;
  // return false;
  // }
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

  cudaMemcpy(inputDevice_, input_data.data(), inputSize_ * sizeof(float),
             cudaMemcpyHostToDevice);
  if (!context_->enqueueV2(bindings_, 0, nullptr))
    return false;

  std::vector<float> host_output(outputSize_);
  cudaMemcpy(host_output.data(), outputDevice_, outputSize_ * sizeof(float),
             cudaMemcpyDeviceToHost);
  processOutput(input, host_output, raw_img);
  return true;
}

void TensorInferencer::processOutput(const InferenceInput &input,
                                     const std::vector<float> &host_output,
                                     const cv::Mat &raw_img) {
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

    // std::cout << "[DEBUG] box " << i << ", obj=" << objectness
    //           << ", max_cls_score=" << max_score << ", class_id=" << class_id
    //           << ", conf=" << confidence << std::endl;
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
                << y1 << "," << x2 << "," << y2 << ")\n";

      saveAnnotatedImage(input.decoded_frames[0], x1, y1, x2, y2, confidence,
                         input.object_name, input.gopIdx);
    }
  }
}

#include <iomanip> // 确保你包含了这个头文件

void TensorInferencer::saveAnnotatedImage(const cv::Mat &raw_img, float x1,
                                          float y1, float x2, float y2,
                                          float confidence,
                                          const std::string &class_name,
                                          int gopIdx) {
  std::cout << "[DEBUG] 正在保存图像. 类别: " << class_name
            << ", 置信度: " << confidence << ", 原始框: (" << x1 << ", " << y1
            << ") - (" << x2 << ", " << y2 << ")" << std::endl;
  std::cout << "[DEBUG] 原图尺寸: " << raw_img.cols << " x " << raw_img.rows
            << std::endl;

  cv::Mat img_to_save = raw_img.clone();

  float scale_x =
      static_cast<float>(raw_img.cols) / static_cast<float>(target_w_);
  float scale_y =
      static_cast<float>(raw_img.rows) / static_cast<float>(target_h_);

  int x1_scaled = static_cast<int>(x1 * scale_x);
  int y1_scaled = static_cast<int>(y1 * scale_y);
  int x2_scaled = static_cast<int>(x2 * scale_x);
  int y2_scaled = static_cast<int>(y2 * scale_y);

  std::cout << "[DEBUG] 映射后框: (" << x1_scaled << ", " << y1_scaled
            << ") - (" << x2_scaled << ", " << y2_scaled << ")" << std::endl;

  // 边界检查（确保画框合法）
  x1_scaled = std::max(0, std::min(x1_scaled, raw_img.cols - 1));
  y1_scaled = std::max(0, std::min(y1_scaled, raw_img.rows - 1));
  x2_scaled = std::max(0, std::min(x2_scaled, raw_img.cols - 1));
  y2_scaled = std::max(0, std::min(y2_scaled, raw_img.rows - 1));

  // 画框
  cv::rectangle(img_to_save, cv::Point(x1_scaled, y1_scaled),
                cv::Point(x2_scaled, y2_scaled), cv::Scalar(0, 255, 0), 2);

  // 标签文字
  std::ostringstream label;
  label << class_name << " " << std::fixed << std::setprecision(2)
        << confidence;
  int baseline = 0;
  cv::Size textSize =
      cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

  cv::rectangle(img_to_save,
                cv::Point(x1_scaled, y1_scaled - textSize.height - 4),
                cv::Point(x1_scaled + textSize.width, y1_scaled),
                cv::Scalar(0, 255, 0), cv::FILLED);
  cv::putText(img_to_save, label.str(), cv::Point(x1_scaled, y1_scaled - 2),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

  // 文件名
  std::ostringstream filename;
  filename << image_output_path_ << "/gop" << gopIdx << "_conf"
           << static_cast<int>(confidence * 100) << ".jpg";

  bool success = cv::imwrite(filename.str(), img_to_save);
  if (success)
    std::cout << "[SAVE] 已保存图片至 " << filename.str() << std::endl;
  else
    std::cerr << "[ERROR] 保存图片失败: " << filename.str() << std::endl;
}

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

static int roundToNearestMultiple(int val, int base = 32) {
  return ((val + base / 2) / base) * base;
}

TensorInferencer::TensorInferencer(int video_height, int video_width) {
  std::cout << "[INIT] 初始化 TensorInferencer，视频尺寸: " << video_width
            << "x" << video_height << std::endl;

  // 恢复目标尺寸计算
  target_w_ = roundToNearestMultiple(video_width, 32);
  target_h_ = roundToNearestMultiple(video_height, 32);

  std::cout << "[INIT] 计算的目标尺寸: " << target_w_ << "x" << target_h_
            << " (32对齐)" << std::endl;

  const char *env_path = std::getenv("YOLO_ENGINE_NAME");
  if (!env_path) {
    std::cerr << "[ERROR] 环境变量 YOLO_ENGINE_NAME 未设置" << std::endl;
    std::exit(1);
  }

  const char *names_path = std::getenv("YOLO_COCO_NAMES");
  if (!names_path) {
    std::cerr << "[ERROR] 环境变量 YOLO_COCO_NAMES 未设置" << std::endl;
    std::exit(1);
  }

  const char *output_path_env = std::getenv("YOLO_IMAGE_PATH");
  if (!output_path_env) {
    std::cerr << "[ERROR] 环境变量 YOLO_IMAGE_PATH 未设置" << std::endl;
    std::exit(1);
  }
  image_output_path_ = output_path_env;

  std::string enginePath = env_path;
  std::cout << "[INIT] 加载引擎文件: " << enginePath << std::endl;

  auto engineData = readEngineFile(enginePath);
  runtime_ = createInferRuntime(gLogger);
  assert(runtime_);
  engine_ =
      runtime_->deserializeCudaEngine(engineData.data(), engineData.size());
  assert(engine_);
  context_ = engine_->createExecutionContext();
  assert(context_);

  std::cout << "[INIT] 引擎加载成功" << std::endl;

  // 打印引擎信息
  printEngineInfo();

  inputIndex_ = engine_->getBindingIndex("images");
  outputIndex_ = engine_->getBindingIndex(engine_->getBindingName(1));

  std::cout << "[INIT] 输入索引: " << inputIndex_
            << ", 输出索引: " << outputIndex_ << std::endl;

  // 加载类别名称
  std::ifstream infile(names_path);
  if (!infile.is_open()) {
    std::cerr << "[ERROR] 无法打开类别文件: " << names_path << std::endl;
    std::exit(1);
  }

  std::string line;
  int idx = 0;
  while (std::getline(infile, line)) {
    if (!line.empty()) {
      class_name_to_id_[line] = idx++;
    }
  }

  std::cout << "[INIT] 加载了 " << class_name_to_id_.size() << " 个类别"
            << std::endl;
  std::cout << "[INIT] dog 类别ID: " << class_name_to_id_["dog"] << std::endl;

  // 检查引擎输入尺寸并调整目标尺寸
  Dims inputDims = engine_->getBindingDimensions(inputIndex_);
  if (inputDims.nbDims == 4 && inputDims.d[2] > 0 && inputDims.d[3] > 0) {
    // 引擎有固定输入尺寸，使用引擎的尺寸
    target_h_ = inputDims.d[2];
    target_w_ = inputDims.d[3];
    std::cout << "[INIT] 使用引擎固定输入尺寸: " << target_w_ << "x"
              << target_h_ << std::endl;
  }
}

void TensorInferencer::printEngineInfo() {
  std::cout << "=== 引擎信息 ===" << std::endl;
  std::cout << "绑定数量: " << engine_->getNbBindings() << std::endl;

  for (int i = 0; i < engine_->getNbBindings(); ++i) {
    const char *name = engine_->getBindingName(i);
    Dims dims = engine_->getBindingDimensions(i);
    bool isInput = engine_->bindingIsInput(i);

    std::cout << "绑定 " << i << ": " << name << " ("
              << (isInput ? "输入" : "输出") << ") - 维度: ";

    for (int j = 0; j < dims.nbDims; ++j) {
      std::cout << dims.d[j];
      if (j < dims.nbDims - 1)
        std::cout << "x";
    }
    std::cout << std::endl;
  }
  std::cout << "================" << std::endl;
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
  std::cout << "[INFER] 开始推理，GOP: " << input.gopIdx
            << ", 目标物体: " << input.object_name << std::endl;

  if (input.decoded_frames.empty()) {
    std::cerr << "[ERROR] 没有输入帧" << std::endl;
    return false;
  }

  const cv::Mat &raw_img = input.decoded_frames[0];
  if (raw_img.empty()) {
    std::cerr << "[ERROR] 输入图像为空" << std::endl;
    return false;
  }

  std::cout << "[INFER] 原始图像尺寸: " << raw_img.cols << "x" << raw_img.rows
            << std::endl;

  // 图像预处理
  cv::Mat img;
  cv::resize(raw_img, img, cv::Size(target_w_, target_h_));
  std::cout << "[INFER] 缩放后图像尺寸: " << target_w_ << "x" << target_h_
            << std::endl;

  int c = 3, h = target_h_, w = target_w_;
  inputSize_ = static_cast<size_t>(c * h * w);
  std::cout << "[INFER] 输入数据大小: " << inputSize_ << std::endl;

  cv::Mat chw_input;
  img.convertTo(chw_input, CV_32FC3, 1.0 / 255.0);
  std::vector<float> input_data(inputSize_);

  // CHW格式转换
  for (int i = 0; i < c; ++i)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        input_data[i * h * w + y * w + x] = chw_input.at<cv::Vec3f>(y, x)[i];

  std::cout << "[INFER] 图像预处理完成" << std::endl;

  // 设置动态输入尺寸（如果需要）
  Dims inputDims{4, {1, 3, h, w}};
  if (!context_->setBindingDimensions(inputIndex_, inputDims)) {
    std::cout << "[WARN] 无法设置绑定维度，可能使用固定尺寸" << std::endl;
  }

  if (!context_->allInputDimensionsSpecified()) {
    std::cerr << "[ERROR] 输入维度未完全指定" << std::endl;
    return false;
  }

  // 计算输出大小
  if (outputSize_ == 0) {
    Dims outDims = context_->getBindingDimensions(outputIndex_);
    outputSize_ = 1;
    for (int i = 0; i < outDims.nbDims; ++i) {
      if (outDims.d[i] <= 0)
        outDims.d[i] = 1;
      outputSize_ *= outDims.d[i];
    }
    std::cout << "[INFER] 计算输出大小: " << outputSize_ << std::endl;
  }

  // 分配GPU内存
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

  std::cout << "[INFER] GPU内存分配完成" << std::endl;

  // 执行推理
  cudaMemcpy(inputDevice_, input_data.data(), inputSize_ * sizeof(float),
             cudaMemcpyHostToDevice);

  std::cout << "[INFER] 开始执行推理..." << std::endl;
  if (!context_->enqueueV2(bindings_, 0, nullptr)) {
    std::cerr << "[ERROR] 推理执行失败" << std::endl;
    return false;
  }
  std::cout << "[INFER] 推理执行完成" << std::endl;

  std::vector<float> host_output(outputSize_);
  cudaMemcpy(host_output.data(), outputDevice_, outputSize_ * sizeof(float),
             cudaMemcpyDeviceToHost);

  std::cout << "[INFER] 输出数据拷贝完成，开始处理结果" << std::endl;
  processOutput(input, host_output, raw_img);
  return true;
}

void TensorInferencer::processOutput(const InferenceInput &input,
                                     const std::vector<float> &host_output,
                                     const cv::Mat &raw_img) {
  std::cout << "[OUTPUT] 开始处理输出，数据大小: " << host_output.size()
            << std::endl;

  // YOLOv8 输出格式: [cx, cy, w, h, class0, class1, ..., class79]
  int box_step = 84; // 4 + 80 = 84
  int num_classes = 80;
  int num_boxes = static_cast<int>(host_output.size() / box_step);

  std::cout << "[OUTPUT] 检测框数量: " << num_boxes << std::endl;
  std::cout << "[OUTPUT] 寻找目标: " << input.object_name << std::endl;

  auto it = class_name_to_id_.find(input.object_name);
  if (it == class_name_to_id_.end()) {
    std::cerr << "[ERROR] 类别 '" << input.object_name << "' 未找到!"
              << std::endl;
    return;
  }
  int target_class_id = it->second;
  std::cout << "[OUTPUT] 目标类别ID: " << target_class_id << std::endl;

  int detections_found = 0;
  float confidence_threshold =
      std::max(0.1f, input.confidence_thresh); // 至少0.1的阈值
  std::cout << "[OUTPUT] 使用置信度阈值: " << confidence_threshold << std::endl;

  for (int i = 0; i < num_boxes; ++i) {
    const float *det = &host_output[i * box_step];

    // 找到最高概率的类别
    int best_class_id = -1;
    float max_score = 0.0f;

    for (int j = 0; j < num_classes; ++j) {
      float cls_score = det[4 + j];
      if (cls_score > max_score) {
        max_score = cls_score;
        best_class_id = j;
      }
    }

    // 专门检查dog类别的得分
    float dog_score = det[4 + target_class_id];

    // 打印前几个框的详细信息
    if (i < 5 || max_score > 0.05f || dog_score > 0.05f) {
      std::cout << "[DEBUG] 框 " << i << ": 最高得分=" << max_score << " (类别"
                << best_class_id << "), dog得分=" << dog_score << ", 坐标=("
                << det[0] << "," << det[1] << "," << det[2] << "," << det[3]
                << ")" << std::endl;
    }

    // 检查是否检测到dog
    if (best_class_id == target_class_id && max_score >= confidence_threshold) {
      detections_found++;

      float cx = det[0], cy = det[1], w = det[2], h = det[3];

      // 坐标转换 - YOLOv8 输出是归一化坐标
      if (cx <= 1.0f && cy <= 1.0f && w <= 1.0f && h <= 1.0f) {
        cx *= target_w_;
        cy *= target_h_;
        w *= target_w_;
        h *= target_h_;
      }

      float x1 = cx - w / 2, y1 = cy - h / 2;
      float x2 = cx + w / 2, y2 = cy + h / 2;

      std::cout << "[DETECTION] 发现 " << input.object_name
                << "! GOP: " << input.gopIdx << ", 置信度: " << max_score
                << ", 边界框: (" << x1 << "," << y1 << "," << x2 << "," << y2
                << ")" << std::endl;

      saveAnnotatedImage(input.decoded_frames[0], x1, y1, x2, y2, max_score,
                         input.object_name, input.gopIdx);
    }
  }

  std::cout << "[OUTPUT] 处理完成，共发现 " << detections_found << " 个 "
            << input.object_name << std::endl;

  if (detections_found == 0) {
    std::cout << "[INFO] 未检测到目标物体，可能原因:" << std::endl;
    std::cout << "  1. 置信度阈值过高 (当前: " << confidence_threshold << ")"
              << std::endl;
    std::cout << "  2. 图像预处理问题" << std::endl;
    std::cout << "  3. 模型输出格式不匹配" << std::endl;
  }
}

void TensorInferencer::saveAnnotatedImage(const cv::Mat &raw_img, float x1,
                                          float y1, float x2, float y2,
                                          float confidence,
                                          const std::string &class_name,
                                          int gopIdx) {
  std::cout << "[SAVE] 准备保存标注图像" << std::endl;
  std::cout << "[SAVE] 类别: " << class_name << ", 置信度: " << confidence
            << std::endl;
  std::cout << "[SAVE] 原图尺寸: " << raw_img.cols << "x" << raw_img.rows
            << std::endl;
  std::cout << "[SAVE] 检测框(模型坐标): (" << x1 << "," << y1 << ") - (" << x2
            << "," << y2 << ")" << std::endl;

  cv::Mat img_to_save = raw_img.clone();

  // 坐标映射回原图
  float scale_x =
      static_cast<float>(raw_img.cols) / static_cast<float>(target_w_);
  float scale_y =
      static_cast<float>(raw_img.rows) / static_cast<float>(target_h_);

  int x1_scaled = static_cast<int>(x1 * scale_x);
  int y1_scaled = static_cast<int>(y1 * scale_y);
  int x2_scaled = static_cast<int>(x2 * scale_x);
  int y2_scaled = static_cast<int>(y2 * scale_y);

  std::cout << "[SAVE] 缩放比例: " << scale_x << "x" << scale_y << std::endl;
  std::cout << "[SAVE] 映射后框: (" << x1_scaled << "," << y1_scaled << ") - ("
            << x2_scaled << "," << y2_scaled << ")" << std::endl;

  // 边界检查
  x1_scaled = std::max(0, std::min(x1_scaled, raw_img.cols - 1));
  y1_scaled = std::max(0, std::min(y1_scaled, raw_img.rows - 1));
  x2_scaled = std::max(0, std::min(x2_scaled, raw_img.cols - 1));
  y2_scaled = std::max(0, std::min(y2_scaled, raw_img.rows - 1));

  // 画框
  cv::rectangle(img_to_save, cv::Point(x1_scaled, y1_scaled),
                cv::Point(x2_scaled, y2_scaled), cv::Scalar(0, 255, 0), 3);

  // 标签
  std::ostringstream label;
  label << class_name << " " << std::fixed << std::setprecision(2)
        << confidence;
  int baseline = 0;
  cv::Size textSize =
      cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);

  cv::rectangle(img_to_save,
                cv::Point(x1_scaled, y1_scaled - textSize.height - 6),
                cv::Point(x1_scaled + textSize.width, y1_scaled),
                cv::Scalar(0, 255, 0), cv::FILLED);
  cv::putText(img_to_save, label.str(), cv::Point(x1_scaled, y1_scaled - 4),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);

  // 保存文件
  std::ostringstream filename;
  filename << image_output_path_ << "/gop" << gopIdx << "_" << class_name
           << "_conf" << static_cast<int>(confidence * 100) << ".jpg";

  bool success = cv::imwrite(filename.str(), img_to_save);
  if (success) {
    std::cout << "[SAVE] ✓ 图片已保存: " << filename.str() << std::endl;
  } else {
    std::cerr << "[ERROR] ✗ 保存失败: " << filename.str() << std::endl;
  }
}
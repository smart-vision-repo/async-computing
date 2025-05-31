#include "tensor_inferencer.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib> // For std::getenv, std::exit, std::stoi
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream> // For std::ostringstream

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Using nvinfer1 namespace
using namespace nvinfer1;

// TensorRT Logger - Defined ONCE
class TrtLogger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    // Only print warnings and above
    if (severity <= Severity::kWARNING) {
      std::cout << "[TRT] " << msg << std::endl;
    }
  }
};
TrtLogger gLogger; // Global logger instance - Defined ONCE

// Static method definition for readEngineFile - Defined ONCE
std::vector<char>
TensorInferencer::readEngineFile(const std::string &enginePath) {
  std::ifstream file(enginePath, std::ios::binary);
  if (!file.good()) {
    std::cerr << "[错误] 打开引擎文件失败: " << enginePath << std::endl;
    return {};
  }
  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> engineData(size);
  if (size > 0) {
    file.read(engineData.data(), size);
  }
  file.close();
  return engineData;
}

// Static method definition for roundToNearestMultiple - Defined ONCE
int TensorInferencer::roundToNearestMultiple(int val, int base) {
  return ((val + base / 2) / base) * base;
}

TensorInferencer::TensorInferencer(int video_height, int video_width,
                                   std::string object_name, int interval,
                                   float confidence, InferenceCallback callback)
    : object_name_(object_name), interval_(interval),
      confidence_(confidence), // Member variables are set here
      runtime_(nullptr), engine_(nullptr), context_(nullptr),
      inputDevice_(nullptr), outputDevice_(nullptr), inputIndex_(-1),
      outputIndex_(-1), num_classes_(0), BATCH_SIZE_(1),
      current_callback_(callback) {
  std::cout << "[初始化] TensorInferencer，视频尺寸: " << video_width << "x"
            << video_height << std::endl;
  std::cout << "[初始化] 目标对象: " << object_name_
            << ", 置信度阈值: " << confidence_ << std::endl;

  const char *env_batch_size_str = std::getenv("YOLO_BATCH_SIZE");
  if (env_batch_size_str) {
    try {
      BATCH_SIZE_ = std::stoi(env_batch_size_str);
      if (BATCH_SIZE_ <= 0) {
        std::cerr << "[错误] BATCH_SIZE 环境变量值 (" << env_batch_size_str
                  << ") 无效。必须为正整数。将使用默认值 1。" << std::endl;
        BATCH_SIZE_ = 1;
      }
    } catch (const std::invalid_argument &ia) {
      std::cerr << "[错误] BATCH_SIZE 环境变量值 (" << env_batch_size_str
                << ") 无效。无法转换为整数。将使用默认值 1。" << std::endl;
      BATCH_SIZE_ = 1;
    } catch (const std::out_of_range &oor) {
      std::cerr << "[错误] BATCH_SIZE 环境变量值 (" << env_batch_size_str
                << ") 超出范围。将使用默认值 1。" << std::endl;
      BATCH_SIZE_ = 1;
    }
  } else {
    std::cerr << "[警告] 未设置 BATCH_SIZE 环境变量。将使用默认值 1。"
              << std::endl;
    BATCH_SIZE_ = 1;
  }
  std::cout << "[初始化] 使用 BATCH_SIZE: " << BATCH_SIZE_ << std::endl;

  std::string engine_env_key =
      "YOLO_ENGINE_NAME_" + std::to_string(BATCH_SIZE_);
  const char *env_engine_path = std::getenv(engine_env_key.c_str());
  if (!env_engine_path) {
    const char *default_engine_path = std::getenv("YOLO_ENGINE_NAME");
    if (default_engine_path) {
      std::cout << "[警告] 未找到环境变量 " << engine_env_key
                << "。尝试使用 YOLO_ENGINE_NAME。" << std::endl;
      env_engine_path = default_engine_path;
    } else {
      std::cerr << "[错误] 既未设置环境变量 " << engine_env_key
                << " 也未设置 YOLO_ENGINE_NAME。" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  engine_path_ = env_engine_path;
  std::cout << "[初始化] 使用引擎文件: " << engine_path_ << std::endl;

  int initial_target_w = roundToNearestMultiple(video_width, 32);
  int initial_target_h = roundToNearestMultiple(video_height, 32);

  target_w_ = initial_target_w;
  target_h_ = initial_target_h;

  std::cout << "[初始化] 初始计算的目标尺寸 (四舍五入到32的倍数): " << target_w_
            << "x" << target_h_ << std::endl;

  const char *names_path_env = std::getenv("YOLO_COCO_NAMES");
  if (!names_path_env) {
    std::cerr << "[错误] 未设置环境变量 YOLO_COCO_NAMES。" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string names_path_str = names_path_env;

  const char *output_path_env = std::getenv("YOLO_IMAGE_PATH");
  if (!output_path_env) {
    std::cerr << "[错误] 未设置环境变量 YOLO_IMAGE_PATH。" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  image_output_path_ = output_path_env;

  auto engineData = readEngineFile(engine_path_);
  if (engineData.empty()) {
    std::cerr << "[错误] 读取引擎数据失败。" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  runtime_ = createInferRuntime(gLogger);
  assert(runtime_ != nullptr && "TensorRT runtime 创建失败。");
  engine_ =
      runtime_->deserializeCudaEngine(engineData.data(), engineData.size());
  assert(engine_ != nullptr && "TensorRT 引擎反序列化失败。");
  context_ = engine_->createExecutionContext();
  assert(context_ != nullptr && "TensorRT 执行上下文创建失败。");

  bindings_.resize(engine_->getNbBindings());

  std::cout << "[初始化] 引擎加载成功。" << std::endl;
  printEngineInfo();

  inputIndex_ = engine_->getBindingIndex("images");
  outputIndex_ = engine_->getBindingIndex("output0");

  if (outputIndex_ < 0) {
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
      if (!engine_->bindingIsInput(i)) {
        outputIndex_ = i;
        std::cout << "[信息] 找到第一个输出张量 '" << engine_->getBindingName(i)
                  << "' 在索引 " << i << std::endl;
        break;
      }
    }
  }

  if (inputIndex_ < 0 || outputIndex_ < 0) {
    std::cerr << "[错误] 未找到输入 'images' 或任何输出张量。" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "[初始化] 输入索引 ('images'): " << inputIndex_
            << ", 输出索引 ('" << engine_->getBindingName(outputIndex_)
            << "'): " << outputIndex_ << std::endl;

  std::ifstream infile(names_path_str);
  if (!infile.is_open()) {
    std::cerr << "[错误] 打开 COCO 名称文件失败: " << names_path_str
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  int idx = 0;
  while (std::getline(infile, line)) {
    if (!line.empty()) {
      class_name_to_id_[line] = idx;
      id_to_class_name_[idx] = line;
      idx++;
    }
  }
  num_classes_ = class_name_to_id_.size();
  std::cout << "[初始化] 加载了 " << num_classes_ << " 个类别名称。"
            << std::endl;

  Dims reportedInputDims = engine_->getBindingDimensions(inputIndex_);
  if (reportedInputDims.nbDims == 4) {
    bool useEngineDims = true;
    if (reportedInputDims.d[2] <= 0 || reportedInputDims.d[3] <= 0) {
      useEngineDims = false;
    }
    if (useEngineDims) {
      target_h_ = reportedInputDims.d[2];
      target_w_ = reportedInputDims.d[3];
      std::cout << "[初始化] 使用引擎的优化配置文件维度作为目标尺寸: "
                << target_w_ << "x" << target_h_ << std::endl;
    } else {
      std::cout << "[初始化] 引擎优化配置文件的 H, W "
                   "维度是动态的或无效的。使用计算的目标尺寸: "
                << target_w_ << "x" << target_h_ << std::endl;
    }
  } else {
    std::cout << "[初始化] 无法获取输入 'images' "
                 "的有效4D引擎优化配置文件维度。使用计算的目标尺寸: "
              << target_w_ << "x" << target_h_ << std::endl;
  }

  std::cout << "[调试_构造函数] 用于预处理的最终 target_w_ = " << target_w_
            << ", 用于预处理的最终 target_h_ = " << target_h_ << std::endl;

  if (engine_->getBindingDataType(inputIndex_) != nvinfer1::DataType::kFLOAT) {
    std::cerr << "[错误] 引擎输入张量 'images' 不是 DataType::kFLOAT!"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "[信息] 引擎输入张量 'images' 确认为 DataType::kFLOAT。"
            << std::endl;

  current_batch_inputs_.reserve(BATCH_SIZE_);
  current_batch_metadata_.reserve(BATCH_SIZE_);
}

TensorInferencer::~TensorInferencer() {
  if (!current_batch_inputs_.empty()) {
    std::cout
        << "[析构] 检测到未处理的批处理数据。正在执行 finalizeInference..."
        << std::endl;
    finalizeInference();
  }

  if (inputDevice_)
    cudaFree(inputDevice_);
  if (outputDevice_)
    cudaFree(outputDevice_);
  if (context_)
    context_->destroy();
  if (engine_)
    engine_->destroy();
  if (runtime_)
    runtime_->destroy();
  std::cout << "[反初始化] TensorInferencer 已销毁。" << std::endl;
}

void TensorInferencer::printEngineInfo() {
  std::cout << "=== 引擎信息 ===" << std::endl;
  std::cout << "引擎名称: " << (engine_->getName() ? engine_->getName() : "N/A")
            << std::endl;
  std::cout << "绑定数量: " << engine_->getNbBindings() << std::endl;
  for (int i = 0; i < engine_->getNbBindings(); ++i) {
    const char *name = engine_->getBindingName(i);
    Dims dims = engine_->getBindingDimensions(i);
    nvinfer1::DataType dtype = engine_->getBindingDataType(i);
    bool isInput = engine_->bindingIsInput(i);
    std::string dtype_str;
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      dtype_str = "FP32";
      break;
    case nvinfer1::DataType::kHALF:
      dtype_str = "FP16";
      break;
    case nvinfer1::DataType::kINT8:
      dtype_str = "INT8";
      break;
    case nvinfer1::DataType::kINT32:
      dtype_str = "INT32";
      break;
    default:
      dtype_str = "未知";
      break;
    }
    std::cout << "绑定 " << i << ": '" << name << "' ("
              << (isInput ? "输入" : "输出") << ") - 类型: " << dtype_str
              << " - 优化配置文件维度: ";
    for (int j = 0; j < dims.nbDims; ++j) {
      std::cout << dims.d[j] << (j < dims.nbDims - 1 ? "x" : "");
    }
    std::cout << std::endl;
  }
  std::cout << "===================" << std::endl;
}

bool TensorInferencer::infer(const InferenceInput &input) {
  std::lock_guard<std::mutex> lock(batch_mutex_);
  current_batch_inputs_.push_back(input);
  BatchImageMetadata meta;
  meta.is_real_image = true;
  if (!input.decoded_frames.empty() && !input.decoded_frames[0].empty()) {
    meta.original_w = input.decoded_frames[0].cols;
    meta.original_h = input.decoded_frames[0].rows;
    meta.original_image_for_callback = input.decoded_frames[0].clone();
  } else {
    std::cerr << "[警告][Infer] 输入的 decoded_frames 为空或第一个帧为空。GOP: "
              << input.gopIdx << std::endl;
    meta.original_w = target_w_;
    meta.original_h = target_h_;
    meta.original_image_for_callback =
        cv::Mat(target_h_, target_w_, CV_8UC3, cv::Scalar(114, 114, 114));
  }
  meta.gopIdx_original = input.gopIdx;
  // meta.object_name_original and meta.confidence_thresh_original are removed
  current_batch_metadata_.push_back(meta);

  if (current_batch_inputs_.size() >= static_cast<size_t>(BATCH_SIZE_)) {
    performBatchInference(false);
    current_batch_inputs_.clear();
    current_batch_metadata_.clear();
  }
  return true;
}

void TensorInferencer::finalizeInference() {
  std::lock_guard<std::mutex> lock(batch_mutex_);

  if (!current_batch_inputs_.empty()) {
    std::cout << "[Finalize] 处理剩余 " << current_batch_inputs_.size()
              << " 个输入..." << std::endl;
    performBatchInference(true);
    current_batch_inputs_.clear();
    current_batch_metadata_.clear();
  } else {
    std::cout << "[Finalize] 没有剩余数据需要处理。" << std::endl;
  }
}

std::vector<float>
TensorInferencer::preprocess_single_image_for_batch(const cv::Mat &img,
                                                    BatchImageMetadata &meta) {
  const int model_input_w = target_w_;
  const int model_input_h = target_h_;
  cv::Mat image_to_process;

  if (!meta.is_real_image) {
    image_to_process = cv::Mat(model_input_h, model_input_w, CV_8UC3,
                               cv::Scalar(114, 114, 114));
    meta.original_w = model_input_w;
    meta.original_h = model_input_h;
    meta.scale_to_model = 1.0f;
    meta.pad_w_left = 0;
    meta.pad_h_top = 0;
  } else {
    image_to_process = img.clone();
  }

  cv::Mat processed_for_model(model_input_h, model_input_w, CV_8UC3,
                              cv::Scalar(114, 114, 114));

  if (meta.is_real_image && meta.original_w > 0 && meta.original_h > 0) {
    float scale_x = static_cast<float>(model_input_w) / meta.original_w;
    float scale_y = static_cast<float>(model_input_h) / meta.original_h;
    meta.scale_to_model = std::min(scale_x, scale_y);

    int scaled_w =
        static_cast<int>(std::round(meta.original_w * meta.scale_to_model));
    int scaled_h =
        static_cast<int>(std::round(meta.original_h * meta.scale_to_model));

    cv::Mat resized_img;
    cv::resize(image_to_process, resized_img, cv::Size(scaled_w, scaled_h), 0,
               0, cv::INTER_LINEAR);

    meta.pad_w_left = (model_input_w - scaled_w) / 2;
    meta.pad_h_top = (model_input_h - scaled_h) / 2;
    resized_img.copyTo(processed_for_model(
        cv::Rect(meta.pad_w_left, meta.pad_h_top, scaled_w, scaled_h)));
  } else if (!meta.is_real_image) {
    // This case handles dummy/padded images.
    // The metadata (original_w, original_h, scale_to_model, pad_w_left,
    // pad_h_top) should have been set before calling this for dummy images.
    processed_for_model = image_to_process; // This is already the dummy image
  }

  cv::Mat img_rgb;
  cv::cvtColor(processed_for_model, img_rgb, cv::COLOR_BGR2RGB);
  int c = 3;
  cv::Mat chw_input_fp32;
  img_rgb.convertTo(chw_input_fp32, CV_32FC3, 1.0 / 255.0);

  std::vector<float> input_data_single(static_cast<size_t>(c) * model_input_h *
                                       model_input_w);
  for (int ch_idx = 0; ch_idx < c; ++ch_idx) {
    for (int y = 0; y < model_input_h; ++y) {
      for (int x = 0; x < model_input_w; ++x) {
        input_data_single[static_cast<size_t>(ch_idx) * model_input_h *
                              model_input_w +
                          static_cast<size_t>(y) * model_input_w + x] =
            chw_input_fp32.at<cv::Vec3f>(y, x)[ch_idx];
      }
    }
  }
  return input_data_single;
}

void TensorInferencer::performBatchInference(bool pad_batch) {
  if (current_batch_inputs_.empty()) {
    return;
  }

  const int ACTUAL_BATCH_SIZE_FOR_GPU =
      pad_batch ? BATCH_SIZE_ : static_cast<int>(current_batch_inputs_.size());
  const int NUM_REAL_IMAGES_IN_CURRENT_PROCESSING_BATCH =
      static_cast<int>(current_batch_inputs_.size());
  std::vector<float> batched_input_data;
  batched_input_data.reserve(static_cast<size_t>(ACTUAL_BATCH_SIZE_FOR_GPU) *
                             3 * target_h_ * target_w_);
  std::vector<cv::Mat> original_raw_images_for_saving(
      ACTUAL_BATCH_SIZE_FOR_GPU);
  std::vector<BatchImageMetadata> processing_metadata = current_batch_metadata_;

  if (pad_batch &&
      NUM_REAL_IMAGES_IN_CURRENT_PROCESSING_BATCH < ACTUAL_BATCH_SIZE_FOR_GPU) {
    int num_to_pad =
        ACTUAL_BATCH_SIZE_FOR_GPU - NUM_REAL_IMAGES_IN_CURRENT_PROCESSING_BATCH;
    for (int k = 0; k < num_to_pad; ++k) {
      BatchImageMetadata dummy_meta;
      dummy_meta.is_real_image = false;
      // For dummy images, we need to ensure metadata is sensible for
      // preprocess_single_image_for_batch
      dummy_meta.original_w = target_w_; // or some default, won't be used for
                                         // scaling if is_real_image is false
      dummy_meta.original_h = target_h_;
      dummy_meta.scale_to_model = 1.0f;
      dummy_meta.pad_w_left = 0;
      dummy_meta.pad_h_top = 0;
      // gopIdx_original, etc. are not strictly needed for dummy data unless
      // callback logic depends on it dummy_meta.gopIdx_original = -1; //
      // Example
      processing_metadata.push_back(dummy_meta);
    }
  }

  for (int i = 0; i < ACTUAL_BATCH_SIZE_FOR_GPU; ++i) {
    cv::Mat current_raw_img_for_preprocessing;
    if (i < NUM_REAL_IMAGES_IN_CURRENT_PROCESSING_BATCH) {
      const InferenceInput &current_input_param = current_batch_inputs_[i];
      if (current_input_param.decoded_frames.empty() ||
          current_input_param.decoded_frames[0].empty()) {
        std::cerr << "[警告][PerformBatch] 在索引 " << i
                  << " 处的 batch_inputs 中的帧为空。"
                  << "将使用虚拟图像进行预处理。" << std::endl;
        processing_metadata[i].is_real_image = false; // Mark as not real
        // Prepare dummy image and ensure metadata for dummy is set
        current_raw_img_for_preprocessing =
            cv::Mat(target_h_, target_w_, CV_8UC3, cv::Scalar(114, 114, 114));
        processing_metadata[i].original_w = target_w_;
        processing_metadata[i].original_h = target_h_;
        processing_metadata[i].scale_to_model = 1.0f;
        processing_metadata[i].pad_w_left = 0;
        processing_metadata[i].pad_h_top = 0;

        original_raw_images_for_saving[i] =
            current_raw_img_for_preprocessing.clone(); // Store dummy image
      } else {
        // processing_metadata[i].is_real_image is already true from infer()
        current_raw_img_for_preprocessing =
            current_input_param.decoded_frames[0];
        original_raw_images_for_saving[i] =
            current_input_param.decoded_frames[0].clone();
      }
    } else { // This is a padding image
      processing_metadata[i].is_real_image = false;
      current_raw_img_for_preprocessing =
          cv::Mat(target_h_, target_w_, CV_8UC3, cv::Scalar(114, 114, 114));
      // Metadata for dummy/padded image already set up in the loop above
      original_raw_images_for_saving[i] =
          current_raw_img_for_preprocessing.clone();
    }
    std::vector<float> single_image_data = preprocess_single_image_for_batch(
        current_raw_img_for_preprocessing, processing_metadata[i]);
    batched_input_data.insert(batched_input_data.end(),
                              single_image_data.begin(),
                              single_image_data.end());
  }

  Dims inputDimsRuntime{4,
                        {ACTUAL_BATCH_SIZE_FOR_GPU, 3, target_h_, target_w_}};
  if (!context_->setBindingDimensions(inputIndex_, inputDimsRuntime)) {
    std::cerr << "[错误] 设置批处理输入的绑定维度失败。" << std::endl;
    return;
  }
  if (!context_->allInputDimensionsSpecified()) {
    std::cerr << "[错误] 未指定批处理推理的所有输入维度。" << std::endl;
    return;
  }

  Dims outDimsRuntime = context_->getBindingDimensions(outputIndex_);
  size_t total_output_elements = 1;
  for (int k = 0; k < outDimsRuntime.nbDims; ++k) {
    if (outDimsRuntime.d[k] <= 0) {
      std::cerr << "[错误] 批处理的运行时输出维度无效: " << outDimsRuntime.d[k]
                << std::endl;
      return;
    }
    total_output_elements *= outDimsRuntime.d[k];
  }
  if (total_output_elements == 0) {
    std::cerr << "[错误] 批处理的运行时输出元素为零。" << std::endl;
    return;
  }

  if (num_classes_ <= 0 || outDimsRuntime.d[0] != ACTUAL_BATCH_SIZE_FOR_GPU ||
      (outDimsRuntime.nbDims == 3 &&
       outDimsRuntime.d[1] != (4 + num_classes_))) {
    std::cerr << "[错误] 批处理推理: 运行时输出属性/批处理不匹配。输出维度: ";
    for (int k = 0; k < outDimsRuntime.nbDims; ++k)
      std::cerr << outDimsRuntime.d[k] << " ";
    std::cerr << "期望批处理: " << ACTUAL_BATCH_SIZE_FOR_GPU
              << ", 期望属性: " << (4 + num_classes_) << std::endl;
    return;
  }
  int num_detections_per_image_from_engine = 0;
  int num_attributes_from_engine = 0;

  if (outDimsRuntime.nbDims == 3) {
    num_attributes_from_engine = outDimsRuntime.d[1];
    num_detections_per_image_from_engine = outDimsRuntime.d[2];
    if (num_attributes_from_engine != (4 + num_classes_)) {
      std::cerr << "[错误] 批处理输出维度1 (属性) "
                << num_attributes_from_engine << " 与期望的 "
                << (4 + num_classes_) << " 不匹配" << std::endl;
      return;
    }
  } else {
    std::cerr << "[错误] 批处理推理的输出维度意外。得到 "
              << outDimsRuntime.nbDims << "D。" << std::endl;
    return;
  }

  if (inputDevice_)
    cudaFree(inputDevice_);
  if (outputDevice_)
    cudaFree(outputDevice_);
  inputDevice_ = nullptr;
  outputDevice_ = nullptr;

  size_t total_input_bytes = batched_input_data.size() * sizeof(float);
  size_t total_output_bytes = total_output_elements * sizeof(float);
  cudaError_t err;
  err = cudaMalloc(&inputDevice_, total_input_bytes);
  if (err != cudaSuccess) {
    std::cerr << "[错误] 批处理输入的 CudaMalloc 失败: "
              << cudaGetErrorString(err) << std::endl;
    return;
  }
  err = cudaMalloc(&outputDevice_, total_output_bytes);
  if (err != cudaSuccess) {
    std::cerr << "[错误] 批处理输出的 CudaMalloc 失败: "
              << cudaGetErrorString(err) << std::endl;
    cudaFree(inputDevice_);
    inputDevice_ = nullptr;
    return;
  }

  bindings_[inputIndex_] = inputDevice_;
  bindings_[outputIndex_] = outputDevice_;

  err = cudaMemcpy(inputDevice_, batched_input_data.data(), total_input_bytes,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "[错误] 批处理 CudaMemcpy H2D 失败: "
              << cudaGetErrorString(err) << std::endl;
    return;
  }

  if (!context_->enqueueV2(bindings_.data(), 0, nullptr)) {
    std::cerr << "[错误] 批处理 TensorRT enqueueV2 失败。" << std::endl;
    return;
  }

  std::vector<float> host_output_batched_raw(total_output_elements);
  err = cudaMemcpy(host_output_batched_raw.data(), outputDevice_,
                   total_output_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "[错误] 批处理 CudaMemcpy D2H 失败: "
              << cudaGetErrorString(err) << std::endl;
    return;
  }

  std::vector<InferenceResult> batch_inference_results;
  std::vector<InferenceInput>
      original_inputs_for_callback; // This still holds original gopIdx etc.

  for (int i = 0; i < NUM_REAL_IMAGES_IN_CURRENT_PROCESSING_BATCH; ++i) {
    if (!processing_metadata[i]
             .is_real_image) { // Should not happen if iterating up to
                               // NUM_REAL_IMAGES...
      continue;
    }
    const InferenceInput &current_original_input_param = // Contains gopIdx
        current_batch_inputs_[i];
    const cv::Mat &current_raw_img_for_saving =
        original_raw_images_for_saving[i]; // This is the raw image (or dummy if
                                           // original was bad)

    if (current_raw_img_for_saving.empty() &&
        processing_metadata[i].is_real_image) {
      std::cerr << "[警告] 批处理索引 " << i
                << " 处的真实图像在后处理前为空。跳过。" << std::endl;
      InferenceResult res;
      res.info = "Error: Input image was empty for GOP " +
                 std::to_string(current_original_input_param.gopIdx);
      batch_inference_results.push_back(res);
      original_inputs_for_callback.push_back(current_original_input_param);
      continue;
    }

    const float *output_for_this_image_start =
        host_output_batched_raw.data() +
        static_cast<size_t>(i) * num_attributes_from_engine *
            num_detections_per_image_from_engine;

    std::vector<InferenceResult> single_image_results;
    process_single_output(
        current_original_input_param, // Pass this for gopIdx
        output_for_this_image_start, num_detections_per_image_from_engine,
        num_attributes_from_engine, current_raw_img_for_saving,
        processing_metadata[i], i, single_image_results);

    batch_inference_results.insert(batch_inference_results.end(),
                                   single_image_results.begin(),
                                   single_image_results.end());
    // Always add original input for callback consistency, even if no detections
    // The callback can then check if results for a given input are empty or
    // not.
    original_inputs_for_callback.push_back(current_original_input_param);

    if (single_image_results.empty() && processing_metadata[i].is_real_image) {
      // If no actual detections, add a placeholder "no detection" result
      // This ensures that the callback gets a result for every real input image
      // processed. Find the last added result slot for this input and update if
      // it's not already an error.
      bool found_placeholder_to_update = false;
      for (size_t cb_idx = 0; cb_idx < batch_inference_results.size();
           ++cb_idx) {
        // This logic might be tricky if multiple results can be generated per
        // single_image_result call Assuming process_single_output might add
        // multiple or zero. Simplification: add a specific "no detection"
        // result if single_image_results is empty.
      }
      // The current logic in process_single_output already adds a "No target
      // detected" message if nms_detections is empty. So this might be
      // redundant or need refinement based on desired callback behavior.
    }
  }

  if (current_callback_ && !original_inputs_for_callback.empty()) {
    // Ensure batch_inference_results has one entry per
    // original_inputs_for_callback or adjust callback to handle variable number
    // of results per input. Current structure implies process_single_output
    // generates results for ONE input. If process_single_output adds no results
    // for an input, batch_inference_results might be shorter. For simplicity
    // here, we assume the callback expects results corresponding to the inputs.
    // If no detections were found for an input, process_single_output adds an
    // info message.

    // If NUM_REAL_IMAGES_IN_CURRENT_PROCESSING_BATCH > 0 and
    // batch_inference_results is empty, it means no detections (not even "no
    // detection" messages) were added. This indicates an issue or that all
    // images were skipped before process_single_output. However,
    // process_single_output is designed to add *some* result (detection or "no
    // detection" info).
    if (NUM_REAL_IMAGES_IN_CURRENT_PROCESSING_BATCH > 0 &&
        batch_inference_results.empty()) {
      // This case implies that no real images successfully went through
      // process_single_output or process_single_output itself failed to produce
      // any result objects. Let's ensure the callback gets *something* if
      // inputs were processed.
      std::vector<InferenceResult> empty_results_for_callback;
      for (const auto &inp : original_inputs_for_callback) {
        InferenceResult res;
        res.info = "No actionable detections for GOP " +
                   std::to_string(inp.gopIdx) + " against target '" +
                   this->object_name_ + "'.";
        empty_results_for_callback.push_back(res);
      }
      current_callback_(empty_results_for_callback);

    } else if (!batch_inference_results.empty()) {
      current_callback_(batch_inference_results);
    } else if (NUM_REAL_IMAGES_IN_CURRENT_PROCESSING_BATCH >
               0) { // batch_inference_results is empty but there were real
                    // images
      current_callback_(
          {}); // Send empty if no results generated but there were inputs
    }

  } else if (!current_callback_ &&
             NUM_REAL_IMAGES_IN_CURRENT_PROCESSING_BATCH > 0) {
    std::cerr << "[错误][PerformBatch] 回调函数未设置，但有输入需要处理！"
              << std::endl;
  }
}

void TensorInferencer::process_single_output(
    const InferenceInput &original_input_param, // Now primarily for gopIdx
    const float *host_output_for_image_raw, int num_detections_in_slice,
    int num_attributes_per_detection, const cv::Mat &raw_img_for_saving,
    const BatchImageMetadata &image_meta, int original_batch_idx_for_debug,
    std::vector<InferenceResult> &single_image_results) {

  std::vector<float> transposed_output(
      static_cast<size_t>(num_detections_in_slice) *
      num_attributes_per_detection);
  for (int det_idx = 0; det_idx < num_detections_in_slice; ++det_idx) {
    for (int attr_idx = 0; attr_idx < num_attributes_per_detection;
         ++attr_idx) {
      transposed_output[static_cast<size_t>(det_idx) *
                            num_attributes_per_detection +
                        attr_idx] =
          host_output_for_image_raw[static_cast<size_t>(attr_idx) *
                                        num_detections_in_slice +
                                    det_idx];
    }
  }

  // Use object_name_ and confidence_ from the class instance
  auto it = class_name_to_id_.find(this->object_name_);
  if (it == class_name_to_id_.end()) {
    std::cerr << "[错误][ProcessOutput] 目标对象名称 '" << this->object_name_
              << "' 在类别名称中未找到。" << std::endl;
    InferenceResult res;
    res.info = "Error: Target object name '" + this->object_name_ +
               "' not found in class names.";
    single_image_results.push_back(res);
    return;
  }
  int target_class_id = it->second;
  float confidence_threshold = this->confidence_; // Use class member

  std::vector<Detection> detected_objects;
  for (int i = 0; i < num_detections_in_slice; ++i) {
    const float *det_attrs = &transposed_output[static_cast<size_t>(i) *
                                                num_attributes_per_detection];
    float max_score = 0.0f;
    int best_class_id = -1;
    for (int j = 0; j < num_classes_; ++j) {
      float score = det_attrs[4 + j];
      if (score > max_score) {
        max_score = score;
        best_class_id = j;
      }
    }

    if (best_class_id == target_class_id && max_score >= confidence_threshold) {
      float cx = det_attrs[0];
      float cy = det_attrs[1];
      float w = det_attrs[2];
      float h = det_attrs[3];
      float x1_model = std::max(0.0f, cx - w / 2.0f);
      float y1_model = std::max(0.0f, cy - h / 2.0f);
      float x2_model =
          std::min(static_cast<float>(target_w_ - 1), cx + w / 2.0f);
      float y2_model =
          std::min(static_cast<float>(target_h_ - 1), cy + h / 2.0f);
      if (x2_model > x1_model && y2_model > y1_model) {
        detected_objects.push_back({x1_model, y1_model, x2_model, y2_model,
                                    max_score, best_class_id,
                                    original_batch_idx_for_debug,
                                    image_meta.is_real_image ? "REAL" : "PAD"});
      }
    }
  }

  std::vector<Detection> nms_detections = applyNMS(detected_objects, 0.45f);

  if (nms_detections.empty() && image_meta.is_real_image) {
    InferenceResult res;
    res.info = "GOP " + std::to_string(original_input_param.gopIdx) + ": No '" +
               this->object_name_ + // Use class member
               "' detected meeting criteria (conf: " +
               std::to_string(confidence_threshold) + ").";
    single_image_results.push_back(res);
  }

  for (size_t i = 0; i < nms_detections.size(); ++i) {
    const auto &det = nms_detections[i];
    if (!image_meta.is_real_image || raw_img_for_saving.empty()) {
      std::cout << "[警告][SAVE] "
                   "跳过保存非真实/空图像元数据或空原始图像的检测。图像索引: "
                << original_batch_idx_for_debug << std::endl;
      continue; // Don't create a result for this if not saving/real
    }
    // Pass this->object_name_ to saveAnnotatedImage
    saveAnnotatedImage(raw_img_for_saving, det, image_meta,
                       original_input_param.gopIdx, static_cast<int>(i));

    InferenceResult res;
    std::ostringstream oss;
    oss << "GOP " << original_input_param.gopIdx << ": Detected '"
        << this->object_name_
        << "' (ClassID: " << det.class_id // Use class member
        << ")"
        << " with confidence " << std::fixed << std::setprecision(4)
        << det.confidence << ". Coords (model_input_space): [" << det.x1 << ","
        << det.y1 << "," << det.x2 << "," << det.y2 << "]";

    float x1_unpadded = det.x1 - image_meta.pad_w_left;
    float y1_unpadded = det.y1 - image_meta.pad_h_top;
    float x2_unpadded = det.x2 - image_meta.pad_w_left;
    float y2_unpadded = det.y2 - image_meta.pad_h_top;

    if (image_meta.scale_to_model > 1e-6f) {
      int x1_orig =
          static_cast<int>(std::round(x1_unpadded / image_meta.scale_to_model));
      int y1_orig =
          static_cast<int>(std::round(y1_unpadded / image_meta.scale_to_model));
      int x2_orig =
          static_cast<int>(std::round(x2_unpadded / image_meta.scale_to_model));
      int y2_orig =
          static_cast<int>(std::round(y2_unpadded / image_meta.scale_to_model));
      x1_orig = std::max(0, std::min(x1_orig, image_meta.original_w - 1));
      y1_orig = std::max(0, std::min(y1_orig, image_meta.original_h - 1));
      x2_orig = std::max(0, std::min(x2_orig, image_meta.original_w - 1));
      y2_orig = std::max(0, std::min(y2_orig, image_meta.original_h - 1));
      oss << ". Coords (original_image_space): [" << x1_orig << "," << y1_orig
          << "," << x2_orig << "," << y2_orig << "]";
    }
    res.info = oss.str();
    single_image_results.push_back(res);
  }
}

float TensorInferencer::calculateIoU(const Detection &a, const Detection &b) {
  float x1_intersect = std::max(a.x1, b.x1);
  float y1_intersect = std::max(a.y1, b.y1);
  float x2_intersect = std::min(a.x2, b.x2);
  float y2_intersect = std::min(a.y2, b.y2);
  if (x2_intersect <= x1_intersect || y2_intersect <= y1_intersect)
    return 0.0f;
  float intersection_area =
      (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect);
  float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  float union_area = area_a + area_b - intersection_area;
  return union_area > 1e-6f ? intersection_area / union_area : 0.0f;
}

std::vector<Detection>
TensorInferencer::applyNMS(const std::vector<Detection> &detections,
                           float iou_threshold) {
  if (detections.empty())
    return {};
  std::vector<Detection> sorted_detections = detections;
  std::sort(sorted_detections.begin(), sorted_detections.end(),
            [](const Detection &a, const Detection &b) {
              return a.confidence > b.confidence;
            });

  std::vector<Detection> result;
  std::vector<bool> suppressed(sorted_detections.size(), false);
  for (size_t i = 0; i < sorted_detections.size(); ++i) {
    if (suppressed[i])
      continue;
    result.push_back(sorted_detections[i]);
    for (size_t j = i + 1; j < sorted_detections.size(); ++j) {
      if (suppressed[j])
        continue;
      // Ensure class IDs match before applying NMS if it's class-specific NMS
      // The current NMS is class-agnostic as detections are pre-filtered for
      // the target class_id. If it were multi-class NMS, you might add: if
      // (sorted_detections[i].class_id != sorted_detections[j].class_id)
      // continue;
      float iou = calculateIoU(sorted_detections[i], sorted_detections[j]);
      if (iou > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
  return result;
}

void TensorInferencer::saveAnnotatedImage(
    const cv::Mat &raw_img_for_saving, const Detection &det,
    const BatchImageMetadata &image_meta,
    // const std::string &class_name_str, // Removed, use this->object_name_
    int gopIdx, int detection_idx_in_image) {
  if (!image_meta.is_real_image || raw_img_for_saving.empty()) {
    std::cerr << "[警告][SAVE] 尝试为非真实或空图像保存注释。GOP: " << gopIdx
              << ", 检测状态: " << det.status_info << ". 跳过。" << std::endl;
    return;
  }

  cv::Mat img_to_save = raw_img_for_saving.clone();
  float x1_unpadded = det.x1 - image_meta.pad_w_left;
  float y1_unpadded = det.y1 - image_meta.pad_h_top;
  float x2_unpadded = det.x2 - image_meta.pad_w_left;
  float y2_unpadded = det.y2 - image_meta.pad_h_top;

  if (image_meta.scale_to_model <= 1e-6f) {
    std::cerr << "[警告][SAVE] GOP " << gopIdx << " 的 scale_to_model ("
              << image_meta.scale_to_model << ") 无效。跳过保存。" << std::endl;
    return;
  }

  int x1_orig =
      static_cast<int>(std::round(x1_unpadded / image_meta.scale_to_model));
  int y1_orig =
      static_cast<int>(std::round(y1_unpadded / image_meta.scale_to_model));
  int x2_orig =
      static_cast<int>(std::round(x2_unpadded / image_meta.scale_to_model));
  int y2_orig =
      static_cast<int>(std::round(y2_unpadded / image_meta.scale_to_model));

  x1_orig = std::max(0, std::min(x1_orig, image_meta.original_w - 1));
  y1_orig = std::max(0, std::min(y1_orig, image_meta.original_h - 1));
  x2_orig = std::max(0, std::min(x2_orig, image_meta.original_w - 1));
  y2_orig = std::max(0, std::min(y2_orig, image_meta.original_h - 1));

  if (x2_orig <= x1_orig || y2_orig <= y1_orig) {
    std::cout << "[警告][SAVE] GOP " << gopIdx
              << " 的letterbox反转后缩放框无效。"
              << "原始框 (模型尺度): [" << det.x1 << "," << det.y1 << ","
              << det.x2 << "," << det.y2 << "]"
              << ". 缩放框 (原始图像): [" << x1_orig << "," << y1_orig << ","
              << x2_orig << "," << y2_orig << "]"
              << ". 元数据: scale=" << image_meta.scale_to_model
              << " padL=" << image_meta.pad_w_left
              << " padT=" << image_meta.pad_h_top << ". 跳过保存。"
              << std::endl;
    return;
  }

  cv::rectangle(img_to_save, cv::Point(x1_orig, y1_orig),
                cv::Point(x2_orig, y2_orig), cv::Scalar(0, 255, 0), 2);
  std::ostringstream label;
  // Use this->object_name_ for the label
  label << this->object_name_ << " " << std::fixed << std::setprecision(2)
        << det.confidence;
  int baseline = 0;
  cv::Size textSize =
      cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseline);
  baseline += 1;

  cv::Point textOrg(x1_orig, y1_orig - 5);
  if (textOrg.y - textSize.height < 0) {
    textOrg.y = y1_orig + textSize.height + 5;
    if (textOrg.y > image_meta.original_h - baseline) {
      textOrg.y = image_meta.original_h - baseline - 2;
    }
  }
  if (textOrg.x + textSize.width > image_meta.original_w) {
    textOrg.x = image_meta.original_w - textSize.width - 2;
  }
  textOrg.x = std::max(0, textOrg.x);

  cv::rectangle(
      img_to_save,
      cv::Point(textOrg.x, textOrg.y - textSize.height - baseline + 2),
      cv::Point(textOrg.x + textSize.width, textOrg.y + baseline - 2),
      cv::Scalar(0, 255, 0), cv::FILLED);
  cv::putText(img_to_save, label.str(), textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

  std::ostringstream filename_oss;
  filename_oss << image_output_path_ << "/gop" << std::setw(4)
               << std::setfill('0') << gopIdx << "_obj" << std::setw(2)
               << std::setfill('0') << detection_idx_in_image << "_"
               << this->object_name_ << "_conf" // Use this->object_name_
               << static_cast<int>(det.confidence * 100) << ".jpg";

  bool success = cv::imwrite(filename_oss.str(), img_to_save);
  if (success) {
    std::cout << "[SAVE] 带注释的图像已保存: " << filename_oss.str()
              << std::endl;
  } else {
    std::cerr << "[错误] 保存图像失败: " << filename_oss.str() << std::endl;
  }
}
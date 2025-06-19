#include "tensor_inferencer.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// OpenCV CUDA Headers
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

using namespace nvinfer1;

class TrtLogger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      // std::cout << "[TRT] " << msg << std::endl;
    }
  }
};
TrtLogger gLogger;

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

int TensorInferencer::roundToNearestMultiple(int val, int base) {
  return ((val + base / 2) / base) * base;
}

TensorInferencer::TensorInferencer(int task_id, int video_height,
                                   int video_width, std::string object_name,
                                   int interval, float confidence,
                                   InferResultCallback resultCallback,
                                   InferPackCallback packCallback)
    : // Initializer list order matches declaration order in .hpp
      task_id_(task_id), object_name_(object_name), interval_(interval),
      confidence_(confidence), BATCH_SIZE_(1), target_w_(0), target_h_(0),
      runtime_(nullptr), engine_(nullptr), context_(nullptr), inputIndex_(-1),
      outputIndex_(-1), inputDevice_(nullptr), outputDevice_(nullptr),
      num_classes_(0), result_callback_(resultCallback),
      pack_callback_(packCallback), constant_metadata_initialized_(false) {

  FrameSelectorCallback frameSelectorCallback =
      [this](const Detection &det, const BatchImageMetadata &image_meta) {
        this->saveAnnotatedImage(det, image_meta);
      };

  frameSelector.emplace(interval_, 0.05f, frameSelectorCallback);
  std::cout << "[初始化] TensorInferencer，视频尺寸: " << video_width << "x"
            << video_height << std::endl;
  std::cout << "[初始化] 目标对象: " << object_name_
            << ", 置信度阈值: " << confidence_
            << ", 帧间隔 (interval): " << interval_ << std::endl;

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
  }
  // Your logs confirm engine max batch is 16. So BATCH_SIZE_ must not exceed
  // this.
  const int KNOWN_MAX_PROFILE_BATCH_SIZE = 16;
  if (BATCH_SIZE_ > KNOWN_MAX_PROFILE_BATCH_SIZE) {
    std::cerr << "[警告] 配置的 BATCH_SIZE (" << BATCH_SIZE_
              << ") 超出已知的引擎配置文件最大批处理大小 ("
              << KNOWN_MAX_PROFILE_BATCH_SIZE << ")。将 BATCH_SIZE 强制设置为 "
              << KNOWN_MAX_PROFILE_BATCH_SIZE << "。" << std::endl;
    BATCH_SIZE_ = KNOWN_MAX_PROFILE_BATCH_SIZE;
  }
  if (BATCH_SIZE_ <= 0) {
    std::cerr << "[警告] BATCH_SIZE (" << BATCH_SIZE_
              << ") 无效（非正数）。将其强制设置为 1。" << std::endl;
    BATCH_SIZE_ = 1;
  }

  std::cout << "[初始化] 使用 BATCH_SIZE (单个帧): " << BATCH_SIZE_
            << std::endl;

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

  target_w_ = roundToNearestMultiple(video_width, 32);
  target_h_ = roundToNearestMultiple(video_height, 32);

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
  // tensor_inferencer.cpp: Around lines 148-153
  image_output_path_ = std::string(output_path_env) + "/" +
                       std::to_string(task_id_); // Corrected line
  try {
    if (!std::filesystem::exists(image_output_path_)) {        //
      std::filesystem::create_directories(image_output_path_); //
    }
  } catch (const std::filesystem::filesystem_error &e) {
    std::cerr << "[错误] 创建目录失败: " << e.what() << std::endl; //
    std::exit(EXIT_FAILURE);
  }

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

  // Use Profile 0 for initial target_w, target_h if engine provides it
  // This assumes profile 0 is the one intended for use.
  // The log shows "Profile 0: MIN[1,3,320,320] OPT[16,3,640,640]
  // MAX[16,3,1088,1920]" So OPT or MAX dimensions from this profile can be used
  // if static. The log also says "默认/上下文优化配置文件维度: -1x3x-1x-1",
  // meaning they are dynamic. And then "引擎优化配置文件的 H, W
  // 维度是动态的或无效的。使用计算的目标尺寸: 1920x1088" This confirms
  // target_w_ and target_h_ are taken from video_width/height. The engine max
  // dimensions for H,W are 1088,1920 which matches.
  Dims profile_dims_opt =
      engine_->getProfileDimensions(inputIndex_, 0, OptProfileSelector::kOPT);
  Dims profile_dims_max =
      engine_->getProfileDimensions(inputIndex_, 0, OptProfileSelector::kMAX);

  if (profile_dims_opt.nbDims == 4 && profile_dims_opt.d[2] > 0 &&
      profile_dims_opt.d[3] > 0) {
    target_h_ = profile_dims_opt.d[2]; // Use OPT H
    target_w_ = profile_dims_opt.d[3]; // Use OPT W
    std::cout << "[初始化] 使用引擎优化配置文件 0 (OPT) 维度作为目标尺寸: "
              << target_w_ << "x" << target_h_ << std::endl;
  } else if (profile_dims_max.nbDims == 4 && profile_dims_max.d[2] > 0 &&
             profile_dims_max.d[3] > 0) {
    // Fallback to MAX if OPT is not well-defined, though current logs show OPT
    // is 640x640 The original log showed it used the calculated 1920x1088,
    // which matches MAX. This means the
    // `engine_->getBindingDimensions(inputIndex_)` might not be returning OPT
    // profile dims. The current code correctly falls back to calculated if
    // reportedInputDims are dynamic. Let's stick to how target_w_, target_h_
    // were set as per user log for now. The log shows: "使用计算的目标尺寸:
    // 1920x1088" - this implies the engine's reported binding dimensions were
    // dynamic. And then "用于预处理的最终 target_w_ = 1920, 用于预处理的最终
    // target_h_ = 1088" This is consistent with Profile 0 MAX dimensions.
  }
  // The original logic for setting target_w_, target_h_ based on engine's
  // reported default dims seemed to work and pick up dynamic values correctly,
  // falling back to calculated ones. The user's log shows this resulted in
  // 1920x1088.

  std::cout << "[调试_构造函数] 用于预处理的最终 target_w_ = " << target_w_
            << ", 用于预处理的最终 target_h_ = " << target_h_ << std::endl;

  if (engine_->getBindingDataType(inputIndex_) != nvinfer1::DataType::kFLOAT) {
    std::cerr << "[错误] 引擎输入张量 'images' 不是 DataType::kFLOAT!"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  current_batch_raw_frames_.reserve(BATCH_SIZE_);
  current_batch_metadata_.reserve(BATCH_SIZE_);
}

TensorInferencer::~TensorInferencer() {
  if (!current_batch_raw_frames_.empty()) {
    std::cout
        << "[析构] 检测到未处理的批处理帧数据。正在执行 finalizeInference..."
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
  std::cout << "优化配置文件数量: " << engine_->getNbOptimizationProfiles()
            << std::endl;

  for (int i = 0; i < engine_->getNbBindings(); ++i) {
    const char *name = engine_->getBindingName(i);
    Dims dims_ctx =
        context_->getBindingDimensions(i); // 需要在 setBindingDimensions 后调用
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
              << " - 当前上下文维度: ";
    for (int j = 0; j < dims_ctx.nbDims; ++j) {
      std::cout << dims_ctx.d[j] << (j < dims_ctx.nbDims - 1 ? "x" : "");
    }
    std::cout << std::endl;

    // ✅ 只对输入 binding 打印 profile 信息
    if (isInput && engine_->getNbOptimizationProfiles() > 0) {
      for (int p = 0; p < engine_->getNbOptimizationProfiles(); ++p) {
        Dims min_dims =
            engine_->getProfileDimensions(i, p, OptProfileSelector::kMIN);
        Dims opt_dims =
            engine_->getProfileDimensions(i, p, OptProfileSelector::kOPT);
        Dims max_dims =
            engine_->getProfileDimensions(i, p, OptProfileSelector::kMAX);

        if (min_dims.nbDims > 0) {
          std::cout << "  Profile " << p << " for '" << name << "': MIN[";
          for (int k = 0; k < min_dims.nbDims; ++k)
            std::cout << min_dims.d[k] << (k < min_dims.nbDims - 1 ? "," : "");
          std::cout << "] OPT[";
          for (int k = 0; k < opt_dims.nbDims; ++k)
            std::cout << opt_dims.d[k] << (k < opt_dims.nbDims - 1 ? "," : "");
          std::cout << "] MAX[";
          for (int k = 0; k < max_dims.nbDims; ++k)
            std::cout << max_dims.d[k] << (k < max_dims.nbDims - 1 ? "," : "");
          std::cout << "]" << std::endl;
        }
      }
    }
  }

  std::cout << "===================" << std::endl;
}

bool TensorInferencer::infer(const InferenceInput &input) {
  std::lock_guard<std::mutex> lock(batch_mutex_);

  if (input.decoded_frames.empty()) {
    // std::cerr << "[警告][Infer] 输入的 decoded_frames 为空。跳过。" <<
    // std::endl; // Can be noisy
    return true;
  }

  int num_frames_in_input = input.decoded_frames.size();

  for (int i = 0; i < num_frames_in_input; ++i) {
    const cv::Mat &current_frame_mat = input.decoded_frames[i];
    if (current_frame_mat.empty()) {
      std::cerr << "[警告][Infer] Frame at index " << i
                << " (latest_frame_index: " << input.latest_frame_index
                << ") is empty. Skipping." << std::endl;
      continue;
    }

    if (!constant_metadata_initialized_ && !current_frame_mat.empty()) {
      cached_geometry_.original_w = current_frame_mat.cols;
      cached_geometry_.original_h = current_frame_mat.rows;

      if (cached_geometry_.original_w > 0 && cached_geometry_.original_h > 0) {
        float scale_x =
            static_cast<float>(target_w_) / cached_geometry_.original_w;
        float scale_y =
            static_cast<float>(target_h_) / cached_geometry_.original_h;
        cached_geometry_.scale_to_model = std::min(scale_x, scale_y);

        int scaled_w_for_cache = static_cast<int>(std::round(
            cached_geometry_.original_w * cached_geometry_.scale_to_model));
        int scaled_h_for_cache = static_cast<int>(std::round(
            cached_geometry_.original_h * cached_geometry_.scale_to_model));

        cached_geometry_.pad_w_left = (target_w_ - scaled_w_for_cache) / 2;
        cached_geometry_.pad_h_top = (target_h_ - scaled_h_for_cache) / 2;
        constant_metadata_initialized_ = true;
      } else {
        std::cerr << "[警告][Infer] "
                     "用于元数据缓存的第一帧为空或尺寸无效。元数据未缓存。"
                  << std::endl;
        constant_metadata_initialized_ = false;
      }
    }

    BatchImageMetadata meta;
    meta.is_real_image = true;

    if (constant_metadata_initialized_) {
      meta.original_w = cached_geometry_.original_w;
      meta.original_h = cached_geometry_.original_h;
      meta.scale_to_model = cached_geometry_.scale_to_model;
      meta.pad_w_left = cached_geometry_.pad_w_left;
      meta.pad_h_top = cached_geometry_.pad_h_top;
    } else {
      meta.original_w = current_frame_mat.cols;
      meta.original_h = current_frame_mat.rows;
      meta.scale_to_model = 0.0f;
      meta.pad_w_left = 0;
      meta.pad_h_top = 0;
    }

    meta.original_image_for_callback = current_frame_mat.clone();
    meta.global_frame_index = input.latest_frame_index -
                              (((num_frames_in_input - 1) - i) * interval_);

    current_batch_raw_frames_.push_back(current_frame_mat);
    current_batch_metadata_.push_back(meta);

    // **FIXED BATCHING LOGIC HERE**
    // Process full batches of BATCH_SIZE_ as they accumulate
    while (current_batch_raw_frames_.size() >=
           static_cast<size_t>(BATCH_SIZE_)) {
      // performBatchInference(false) will now correctly process the first
      // BATCH_SIZE_ frames from current_batch_raw_frames_ and
      // current_batch_metadata_
      performBatchInference(false);

      // Remove the BATCH_SIZE_ frames that were just processed
      current_batch_raw_frames_.erase(current_batch_raw_frames_.begin(),
                                      current_batch_raw_frames_.begin() +
                                          BATCH_SIZE_);
      current_batch_metadata_.erase(current_batch_metadata_.begin(),
                                    current_batch_metadata_.begin() +
                                        BATCH_SIZE_);
    }
  }
  return true;
}

void TensorInferencer::finalizeInference() {
  std::lock_guard<std::mutex> lock(batch_mutex_);

  if (!current_batch_raw_frames_.empty()) {
    std::cout << "[Finalize] 处理剩余 " << current_batch_raw_frames_.size()
              << " 个帧..." << std::endl;
    // performBatchInference(true) will process all remaining frames and pad if
    // necessary
    performBatchInference(true);
    current_batch_raw_frames_.clear(); // Clear all after final processing
    current_batch_metadata_.clear();
  } else {
    std::cout << "[Finalize] 没有剩余帧需要处理。" << std::endl;
  }
}

void TensorInferencer::preprocess_single_image_for_batch(
    const cv::Mat &cpu_img, BatchImageMetadata &meta, int model_input_w,
    int model_input_h, cv::cuda::GpuMat &chw_planar_output_gpu_buffer_slice) {
  cv::cuda::GpuMat gpu_processed_for_model;

  if (!meta.is_real_image) {
    gpu_processed_for_model.create(model_input_h, model_input_w, CV_8UC3);
    gpu_processed_for_model.setTo(cv::Scalar(114, 114, 114));
  } else {
    if (cpu_img.empty()) {
      std::cerr << "[警告][PreProcGPU] Real image (Frame: "
                << meta.global_frame_index
                << ") is empty. Using letterbox fill for this slot."
                << std::endl;
      gpu_processed_for_model.create(model_input_h, model_input_w, CV_8UC3);
      gpu_processed_for_model.setTo(cv::Scalar(114, 114, 114));
      goto convert_to_rgb_and_normalize; // Jump to common processing for
                                         // dummy/fallback
    }

    cv::cuda::GpuMat gpu_original_img;
    gpu_original_img.upload(cpu_img);

    if (meta.original_w <= 0 || meta.original_h <= 0) {
      std::cerr << "[警告][PreProcGPU] Real image (Frame: "
                << meta.global_frame_index
                << ") has invalid original dimensions (" << meta.original_w
                << "x" << meta.original_h << "). Using letterbox fill."
                << std::endl;
      gpu_processed_for_model.create(model_input_h, model_input_w, CV_8UC3);
      gpu_processed_for_model.setTo(cv::Scalar(114, 114, 114));
      goto convert_to_rgb_and_normalize;
    }

    if (meta.scale_to_model == 0.0f) {
      float scale_x = static_cast<float>(model_input_w) / meta.original_w;
      float scale_y = static_cast<float>(model_input_h) / meta.original_h;
      meta.scale_to_model = std::min(scale_x, scale_y);

      int temp_scaled_w =
          static_cast<int>(std::round(meta.original_w * meta.scale_to_model));
      int temp_scaled_h =
          static_cast<int>(std::round(meta.original_h * meta.scale_to_model));

      meta.pad_w_left = (model_input_w - temp_scaled_w) / 2;
      meta.pad_h_top = (model_input_h - temp_scaled_h) / 2;
    }

    int scaled_w =
        static_cast<int>(std::round(meta.original_w * meta.scale_to_model));
    int scaled_h =
        static_cast<int>(std::round(meta.original_h * meta.scale_to_model));

    if (scaled_w <= 0 || scaled_h <= 0 || meta.pad_w_left < 0 ||
        meta.pad_h_top < 0 || meta.pad_w_left + scaled_w > model_input_w ||
        meta.pad_h_top + scaled_h > model_input_h) {
      std::cerr << "[警告][PreProcGPU] Invalid scaling or padding calculated "
                   "for Frame "
                << meta.global_frame_index << ". Orig: " << meta.original_w
                << "x" << meta.original_h << ", Scaled: " << scaled_w << "x"
                << scaled_h << ", Scale: " << meta.scale_to_model
                << ", PadL: " << meta.pad_w_left << ", PadT: " << meta.pad_h_top
                << ". Model In: " << model_input_w << "x" << model_input_h
                << ". Using letterbox fill." << std::endl;
      gpu_processed_for_model.create(model_input_h, model_input_w, CV_8UC3);
      gpu_processed_for_model.setTo(cv::Scalar(114, 114, 114));
      goto convert_to_rgb_and_normalize;
    }

    cv::cuda::GpuMat gpu_letterbox_canvas(model_input_h, model_input_w, CV_8UC3,
                                          cv::Scalar(114, 114, 114));
    cv::cuda::GpuMat gpu_resized_img;
    cv::cuda::resize(gpu_original_img, gpu_resized_img,
                     cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_LINEAR);
    gpu_resized_img.copyTo(gpu_letterbox_canvas(
        cv::Rect(meta.pad_w_left, meta.pad_h_top, scaled_w, scaled_h)));
    gpu_processed_for_model = gpu_letterbox_canvas;
  }

convert_to_rgb_and_normalize: // Common processing path for real (after
                              // letterbox) or dummy
  cv::cuda::GpuMat gpu_rgb;
  cv::cuda::cvtColor(gpu_processed_for_model, gpu_rgb, cv::COLOR_BGR2RGB);

  cv::cuda::GpuMat gpu_float_hwc;
  gpu_rgb.convertTo(gpu_float_hwc, CV_32FC3, 1.0 / 255.0);

  std::vector<cv::cuda::GpuMat> gpu_planes;
  cv::cuda::split(gpu_float_hwc, gpu_planes);

  int plane_size_elements = model_input_h * model_input_w;
  cv::cuda::GpuMat chw_output_temp(1, plane_size_elements * 3, CV_32F);

  // 使用临时 GpuMat 拼接 3 个通道到 chw 格式中
  gpu_planes[0].reshape(1, 1).copyTo(
      chw_output_temp.colRange(0, plane_size_elements));
  gpu_planes[1].reshape(1, 1).copyTo(
      chw_output_temp.colRange(plane_size_elements, 2 * plane_size_elements));
  gpu_planes[2].reshape(1, 1).copyTo(chw_output_temp.colRange(
      2 * plane_size_elements, 3 * plane_size_elements));

  // 将数据从临时 GpuMat 拷贝到 TensorRT 指定的显存区域
  cudaMemcpy(chw_planar_output_gpu_buffer_slice.ptr<float>(),
             chw_output_temp.ptr<float>(),
             plane_size_elements * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
}

void TensorInferencer::performBatchInference(bool pad_batch) {
  const int num_real_frames_in_queue =
      static_cast<int>(current_batch_raw_frames_.size());
  if (num_real_frames_in_queue == 0 && !pad_batch) {
    std::cout << "[DEBUG_PerformBatch] No real frames in queue and not "
                 "padding. Returning."
              << std::endl;
    return;
  }

  int trt_batch_size;             // The batch size to configure TensorRT with
  int frames_to_preprocess_count; // How many frames from the queue will be
                                  // preprocessed (real ones)

  std::vector<BatchImageMetadata> temp_metadata_for_this_batch;

  if (pad_batch) {
    // Finalizing: TRT batch size is BATCH_SIZE_, padding if necessary
    trt_batch_size = BATCH_SIZE_;
    frames_to_preprocess_count =
        num_real_frames_in_queue; // Process all remaining real frames
    // Copy all current metadata
    temp_metadata_for_this_batch = current_batch_metadata_;
  } else {
    // Regular full batch: TRT batch size is BATCH_SIZE_. We take BATCH_SIZE_
    // frames from the queue. This assumes the queue (current_batch_raw_frames_)
    // has at least BATCH_SIZE_ elements, which is guaranteed by the calling
    // condition in infer() due to the 'while' loop.
    if (num_real_frames_in_queue < BATCH_SIZE_) {
      // This case should ideally not be hit if called with pad_batch=false,
      // as the 'while' loop in infer() ensures num_real_frames_in_queue >=
      // BATCH_SIZE_
      std::cerr << "[错误_PerformBatch_Logic] Called with pad_batch=false, but "
                   "num_real_frames_in_queue ("
                << num_real_frames_in_queue << ") < BATCH_SIZE_ ("
                << BATCH_SIZE_ << "). This indicates a logic error."
                << std::endl;
      // Fallback: process what's available, TRT batch will be this size.
      trt_batch_size = num_real_frames_in_queue;
      frames_to_preprocess_count = num_real_frames_in_queue;
      if (num_real_frames_in_queue ==
          0) { // Double check, already handled at top
        std::cout << "[DEBUG_PerformBatch] No frames to process even in "
                     "fallback. Returning."
                  << std::endl;
        return;
      }
    } else {
      trt_batch_size = BATCH_SIZE_;
      frames_to_preprocess_count = BATCH_SIZE_;
    }
    // Copy only the first BATCH_SIZE_ metadata elements for this batch
    temp_metadata_for_this_batch.assign(current_batch_metadata_.begin(),
                                        current_batch_metadata_.begin() +
                                            frames_to_preprocess_count);
  }
  if (trt_batch_size <=
      0) { // Should have been caught by BATCH_SIZE_ > 0 check in constructor
    std::cout << "[DEBUG_PerformBatch] trt_batch_size is <= 0. Returning."
              << std::endl;
    return;
  }
  // If there are no real frames to process and we are not padding up to a
  // larger batch, nothing to do for TRT.
  if (frames_to_preprocess_count == 0 &&
      trt_batch_size ==
          0) { // This condition is now effectively trt_batch_size <=0
    std::cout << "[DEBUG_PerformBatch] No frames to preprocess and "
                 "trt_batch_size is 0. Returning."
              << std::endl;
    return;
  }

  // Pad metadata if finalizing and necessary
  if (pad_batch && frames_to_preprocess_count < trt_batch_size) {
    int num_to_pad = trt_batch_size - frames_to_preprocess_count;
    std::cout << "[DEBUG_PerformBatch] Padding batch with " << num_to_pad
              << " dummy frames for metadata." << std::endl;
    for (int k = 0; k < num_to_pad; ++k) {
      BatchImageMetadata dummy_meta;
      dummy_meta.is_real_image = false;
      dummy_meta.original_w = target_w_;
      dummy_meta.original_h = target_h_;
      dummy_meta.global_frame_index = -1;
      dummy_meta.original_image_for_callback =
          cv::Mat(target_h_, target_w_, CV_8UC3, cv::Scalar(114, 114, 114));
      temp_metadata_for_this_batch.push_back(dummy_meta);
    }
  }

  Dims inputDimsRuntime{4, {trt_batch_size, 3, target_h_, target_w_}};

  if (!context_->setBindingDimensions(inputIndex_, inputDimsRuntime)) {
    std::cerr << "[错误] 设置批处理输入的绑定维度失败。" << std::endl;
    std::cerr << "[DEBUG_PerformBatch_SetDimFail] Failed with trt_batch_size: "
              << trt_batch_size << " Target HxW: " << target_h_ << "x"
              << target_w_ << std::endl;
    return;
  }
  if (!context_->allInputDimensionsSpecified()) {
    std::cerr << "[错误] 未指定批处理推理的所有输入维度。" << std::endl;
    return;
  }

  Dims outDimsRuntime = context_->getBindingDimensions(outputIndex_);
  size_t total_output_elements = 1;
  bool valid_output_dims = outDimsRuntime.nbDims > 0;
  for (int k = 0; k < outDimsRuntime.nbDims; ++k) {
    if (outDimsRuntime.d[k] <= 0) {
      std::cerr << "[错误] 批处理的运行时输出维度 " << k
                << " 无效: " << outDimsRuntime.d[k] << std::endl;
      valid_output_dims = false;
      break;
    }
    total_output_elements *= outDimsRuntime.d[k];
  }
  if (!valid_output_dims || total_output_elements == 0) {
    std::cerr << "[错误] 批处理的运行时输出元素为零或维度无效。" << std::endl;
    return;
  }

  int num_attributes_from_engine = 0;
  int num_detections_per_image_from_engine = 0;

  if (outDimsRuntime.nbDims == 3) {
    if (outDimsRuntime.d[0] != trt_batch_size) {
      std::cerr << "[错误] 批处理推理: 运行时输出批处理大小不匹配。期望 "
                << trt_batch_size << ", 得到 " << outDimsRuntime.d[0]
                << std::endl;
      return;
    }
    num_attributes_from_engine = outDimsRuntime.d[1];
    num_detections_per_image_from_engine = outDimsRuntime.d[2];

    if (num_attributes_from_engine != (4 + num_classes_)) {
      std::cerr << "[错误] 批处理输出维度1 (属性) "
                << num_attributes_from_engine << " 与期望的 "
                << (4 + num_classes_) << " (4 coords + num_classes) 不匹配"
                << std::endl;
      return;
    }
  } else {
    std::cerr << "[错误] 批处理推理的输出维度意外。得到 "
              << outDimsRuntime.nbDims
              << "D。期望3D [batch, attributes, detections_count]."
              << std::endl;
    return;
  }

  if (inputDevice_)
    cudaFree(inputDevice_);
  inputDevice_ = nullptr;
  if (outputDevice_)
    cudaFree(outputDevice_);
  outputDevice_ = nullptr;

  size_t single_image_elements = static_cast<size_t>(3) * target_h_ * target_w_;
  size_t total_input_bytes = static_cast<size_t>(trt_batch_size) *
                             single_image_elements * sizeof(float);
  size_t total_output_bytes = total_output_elements * sizeof(float);

  cudaError_t err;
  err = cudaMalloc(&inputDevice_, total_input_bytes);
  if (err != cudaSuccess) {
    std::cerr << "[错误] 批处理输入的 CudaMalloc 失败 (" << total_input_bytes
              << " bytes): " << cudaGetErrorString(err) << std::endl;
    return;
  }
  err = cudaMalloc(&outputDevice_, total_output_bytes);
  if (err != cudaSuccess) {
    std::cerr << "[错误] 批处理输出的 CudaMalloc 失败 (" << total_output_bytes
              << " bytes): " << cudaGetErrorString(err) << std::endl;
    cudaFree(inputDevice_);
    inputDevice_ = nullptr;
    return;
  }

  // Preprocess loop runs for trt_batch_size (which includes padding slots)
  // It uses temp_metadata_for_this_batch which is also sized for trt_batch_size
  // (padded)
  for (int i = 0; i < trt_batch_size; ++i) {
    cv::Mat current_raw_img_for_preprocessing_cpu;
    if (temp_metadata_for_this_batch[i].is_real_image) {
      // Real frames are at the beginning of current_batch_raw_frames_
      // This 'i' corresponds to the slot in the TRT batch.
      // We need to ensure we only access current_batch_raw_frames_ for actual
      // real frames available.
      if (i <
          frames_to_preprocess_count) { // frames_to_preprocess_count is number
                                        // of actual real frames for this batch
        current_raw_img_for_preprocessing_cpu = current_batch_raw_frames_[i];
      } else {
        // This indicates a padding slot that somehow has is_real_image = true
        // in metadata, which is a logic error if padding was done correctly.
        // preprocess_single_image_for_batch will treat as dummy if cpu_img is
        // empty.
        std::cerr
            << "[错误_PerformBatch_PreprocLoop] Metadata at TRT batch slot "
            << i
            << " marked as real, but this slot should be padding (real frames "
               "to preprocess: "
            << frames_to_preprocess_count << ")" << std::endl;
      }
    }

    float *current_input_slice_ptr =
        reinterpret_cast<float *>(inputDevice_) + i * single_image_elements;
    cv::cuda::GpuMat chw_output_slice_gpu_wrapper(
        1, static_cast<int>(single_image_elements), CV_32F,
        current_input_slice_ptr);

    preprocess_single_image_for_batch(
        current_raw_img_for_preprocessing_cpu, temp_metadata_for_this_batch[i],
        target_w_, target_h_, chw_output_slice_gpu_wrapper);
  }

  bindings_[inputIndex_] = inputDevice_;
  bindings_[outputIndex_] = outputDevice_;

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

  // Post-process only the actual real frames that were part of this batch
  for (int i = 0; i < frames_to_preprocess_count; ++i) {
    const BatchImageMetadata &frame_meta = temp_metadata_for_this_batch[i];

    if (!frame_meta.is_real_image) { // Should not happen if loop is up to
                                     // frames_to_preprocess_count
      std::cerr << "[警告][PerformBatch_PostProc] Trying to post-process a "
                   "non-real frame meta at index "
                << i
                << " (total real preprocessed: " << frames_to_preprocess_count
                << "). Skipping." << std::endl;
      continue;
    }
    if (frame_meta.original_image_for_callback.empty()) {
      std::cerr << "[警告][PerformBatch_PostProc] Real frame (Frame: "
                << frame_meta.global_frame_index
                << ") original_image_for_callback is empty. Skipping."
                << std::endl;
      continue;
    }

    const float *output_for_this_image_start =
        host_output_batched_raw.data() +
        static_cast<size_t>(i) * num_attributes_from_engine *
            num_detections_per_image_from_engine;

    std::vector<InferenceResult> single_frame_results;
    process_single_output(frame_meta, output_for_this_image_start,
                          num_detections_per_image_from_engine,
                          num_attributes_from_engine, i, single_frame_results);
  }
  pack_callback_(frames_to_preprocess_count);
}

// 修改后的
// process_single_output：支持多类检测，仅打印被最终认为是目标类的检测框
void TensorInferencer::process_single_output(
    const BatchImageMetadata &image_meta,
    const float *host_output_for_image_raw, int num_detections_in_slice,
    int num_attributes_per_detection, int original_batch_idx_for_debug,
    std::vector<InferenceResult> &frame_results) {

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

  float confidence_threshold = this->confidence_;
  std::vector<Detection> detected_objects;

  for (int i = 0; i < num_detections_in_slice; ++i) {
    const float *det_attrs = &transposed_output[static_cast<size_t>(i) *
                                                num_attributes_per_detection];

    float max_score = -1.0f;
    int best_class_id = -1;
    for (int j = 0; j < num_classes_; ++j) {
      float score = det_attrs[4 + j];
      if (score > max_score) {
        max_score = score;
        best_class_id = j;
      }
    }

    if (max_score >= confidence_threshold) {
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

  // 执行 class-wise NMS，并只保留目标类别的检测结果
  auto it = class_name_to_id_.find(this->object_name_);
  if (it == class_name_to_id_.end()) {
    std::cerr << "[错误][ProcessOutput] (Frame: "
              << image_meta.global_frame_index << ") 目标对象名称 '"
              << this->object_name_ << "' 在类别名称中未找到。" << std::endl;
    return;
  }
  int target_class_id = it->second;

  std::vector<Detection> filtered;
  for (const auto &d : detected_objects) {
    if (d.class_id == target_class_id) {
      filtered.push_back(d);
    }
  }
  std::vector<Detection> nms_detections = applyNMS(filtered, 0.45f);

  float timestamp_sec =
      static_cast<float>(image_meta.global_frame_index) / 30.0f;

  if (nms_detections.empty() && image_meta.is_real_image) {
    InferenceResult res;
    std::ostringstream oss;
    oss << "Frame " << image_meta.global_frame_index << " (Time: " << std::fixed
        << std::setprecision(2) << timestamp_sec << "s)"
        << ": No '" << this->object_name_
        << "' detected meeting criteria (conf: " << std::fixed
        << std::setprecision(2) << confidence_threshold << ").";
    res.info = oss.str();
    frame_results.push_back(res);
  }

  for (size_t i = 0; i < nms_detections.size(); ++i) {
    const auto &det = nms_detections[i];
    if (!image_meta.is_real_image ||
        image_meta.original_image_for_callback.empty()) {
      std::cout << "[警告][ProcessOutput] "
                   "跳过保存/处理非真实或空图像元数据的检测。Frame: "
                << image_meta.global_frame_index << std::endl;
      continue;
    }

    frameSelector->removeDulicatedFrames({det, image_meta});
    // saveAnnotatedImage(det, image_meta);

    InferenceResult res;
    std::ostringstream oss;
    oss << "Frame " << image_meta.global_frame_index << " (Time: " << std::fixed
        << std::setprecision(2) << timestamp_sec << "s)"
        << ": Detected '" << this->object_name_
        << "' (ClassID: " << det.class_id << ")"
        << " with confidence " << std::fixed << std::setprecision(4)
        << det.confidence;

    float x1_unpadded = det.x1 - image_meta.pad_w_left;
    float y1_unpadded = det.y1 - image_meta.pad_h_top;
    float x2_unpadded = det.x2 - image_meta.pad_w_left;
    float y2_unpadded = det.y2 - image_meta.pad_h_top;

    if (image_meta.scale_to_model > 1e-6f && image_meta.original_w > 0 &&
        image_meta.original_h > 0) {
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
    } else {
      oss << ". Coords (model_input_space): [" << det.x1 << "," << det.y1 << ","
          << det.x2 << "," << det.y2 << "]";
    }
    res.info = oss.str();
    frame_results.push_back(res);
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
      float iou = calculateIoU(sorted_detections[i], sorted_detections[j]);
      if (iou > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
  return result;
}

void TensorInferencer::saveAnnotatedImage(
    const Detection &det, const BatchImageMetadata &image_meta) {

  if (!image_meta.is_real_image ||
      image_meta.original_image_for_callback.empty()) {
    std::cerr << "[警告][SAVE] Attempting to save annotation for non-real or "
                 "empty image. Frame: "
              << image_meta.global_frame_index
              << ", Detection status: " << det.status_info << ". Skipping."
              << std::endl;
    return;
  }

  cv::Mat img_to_save = image_meta.original_image_for_callback.clone();

  float x1_unpadded = det.x1 - image_meta.pad_w_left;
  float y1_unpadded = det.y1 - image_meta.pad_h_top;
  float x2_unpadded = det.x2 - image_meta.pad_w_left;
  float y2_unpadded = det.y2 - image_meta.pad_h_top;

  if (image_meta.scale_to_model <= 1e-6f || image_meta.original_w <= 0 ||
      image_meta.original_h <= 0) {
    std::cerr << "[警告][SAVE] Invalid scale_to_model ("
              << image_meta.scale_to_model << ") or original dimensions ("
              << image_meta.original_w << "x" << image_meta.original_h
              << ") for Frame " << image_meta.global_frame_index
              << ". Skipping save." << std::endl;
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
    std::cout
        << "[警告][SAVE] Invalid scaled box after letterbox reversal for Frame "
        << image_meta.global_frame_index << ". Original box (model scale): ["
        << det.x1 << "," << det.y1 << "," << det.x2 << "," << det.y2 << "]"
        << ". Scaled box (original image): [" << x1_orig << "," << y1_orig
        << "," << x2_orig << "," << y2_orig << "]"
        << ". Metadata: scale=" << image_meta.scale_to_model
        << " padL=" << image_meta.pad_w_left << " padT=" << image_meta.pad_h_top
        << ". Skipping save." << std::endl;
    return;
  }

  cv::rectangle(img_to_save, cv::Point(x1_orig, y1_orig),
                cv::Point(x2_orig, y2_orig), cv::Scalar(0, 255, 0), 2);
  std::ostringstream label;
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

  /**
   * 文件名:
   * 1. 全局帧位置
   * 2. 预估时间
   * 3. 置信率
   * 4. 在GOP包中的位置
   * 5. 检测对象名称
   */

  std::ostringstream filename_oss;
  float timestamp_sec =
      static_cast<float>(image_meta.global_frame_index) / 30.0f;

  int confidence_int = static_cast<int>(det.confidence * 10000);

  filename_oss << image_output_path_ << "/" << std::setprecision(0)
               << image_meta.global_frame_index << "_" << confidence_int
               << ".jpg";

  std::cout << filename_oss.str() << std::endl;
  bool success = cv::imwrite(filename_oss.str(), img_to_save);
  if (success) {
    InferenceResult iResult = InferenceResult();
    iResult.taskId = task_id_;
    iResult.confidence = confidence_int;
    iResult.frameIndex = image_meta.global_frame_index;
    iResult.seconds = timestamp_sec;
    iResult.image = filename_oss.str();
    result_callback_(iResult);
  } else {
    std::cerr << "[错误] Saving image failed: " << filename_oss.str()
              << std::endl;
  }
}
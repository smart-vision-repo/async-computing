#include "tensor_inferencer.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// OpenCV CUDA Headers
#include <opencv2/cudaarithm.hpp> // For convertTo, split, setTo
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp> // For resize

using namespace nvinfer1;

class TrtLogger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[TRT] " << msg << std::endl;
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

TensorInferencer::TensorInferencer(int video_height, int video_width,
                                   std::string object_name, int interval,
                                   float confidence, InferenceCallback callback)
    : object_name_(object_name), interval_(interval), confidence_(confidence),
      runtime_(nullptr), engine_(nullptr), context_(nullptr),
      inputDevice_(nullptr), outputDevice_(nullptr), inputIndex_(-1),
      outputIndex_(-1), num_classes_(0), BATCH_SIZE_(1),
      current_callback_(callback),
      constant_metadata_initialized_(
          false) { // cached_geometry_ will be default initialized
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

  if (input.decoded_frames.empty()) {
    std::cerr << "[警告][Infer] 输入的 decoded_frames 为空。跳过。"
              << std::endl;
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

        int scaled_w = static_cast<int>(std::round(
            cached_geometry_.original_w * cached_geometry_.scale_to_model));
        int scaled_h = static_cast<int>(std::round(
            cached_geometry_.original_h * cached_geometry_.scale_to_model));

        cached_geometry_.pad_w_left = (target_w_ - scaled_w) / 2;
        cached_geometry_.pad_h_top = (target_h_ - scaled_h) / 2;
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
      meta.scale_to_model =
          0.0f; // Will be recalculated in preprocess if needed
      meta.pad_w_left = 0;
      meta.pad_h_top = 0;
    }

    meta.original_image_for_callback = current_frame_mat.clone();
    meta.global_frame_index = input.latest_frame_index -
                              (((num_frames_in_input - 1) - i) * interval_);

    current_batch_raw_frames_.push_back(current_frame_mat);
    current_batch_metadata_.push_back(meta);

    if (current_batch_raw_frames_.size() >= static_cast<size_t>(BATCH_SIZE_)) {
      performBatchInference(false);
      current_batch_raw_frames_.clear();
      current_batch_metadata_.clear();
    }
  }
  return true;
}

void TensorInferencer::finalizeInference() {
  std::lock_guard<std::mutex> lock(batch_mutex_);

  if (!current_batch_raw_frames_.empty()) {
    std::cout << "[Finalize] 处理剩余 " << current_batch_raw_frames_.size()
              << " 个帧..." << std::endl;
    performBatchInference(true);
    current_batch_raw_frames_.clear();
    current_batch_metadata_.clear();
  } else {
    std::cout << "[Finalize] 没有剩余帧需要处理。" << std::endl;
  }
}

// Modified preprocess_single_image_for_batch to perform operations on GPU
// and write CHW planar data directly into the provided GpuMat slice.
void TensorInferencer::preprocess_single_image_for_batch(
    const cv::Mat &cpu_img, // Input CPU image (can be empty for dummy)
    BatchImageMetadata &meta, int model_input_w, int model_input_h,
    cv::cuda::GpuMat
        &chw_planar_output_gpu_buffer_slice // Wraps a slice of inputDevice_
) {
  cv::cuda::GpuMat
      gpu_processed_for_model; // This will be the letterboxed image on GPU

  if (!meta.is_real_image) {
    // For dummy images, create a letterbox-sized image filled with 114 directly
    // on GPU
    gpu_processed_for_model.create(model_input_h, model_input_w, CV_8UC3);
    gpu_processed_for_model.setTo(cv::Scalar(114, 114, 114)); // BGR scalar
  } else {
    if (cpu_img.empty()) {
      std::cerr << "[错误][PreProcGPU] Real image (Frame: "
                << meta.global_frame_index
                << ") is empty. Using letterbox fill for this slot."
                << std::endl;
      gpu_processed_for_model.create(model_input_h, model_input_w, CV_8UC3);
      gpu_processed_for_model.setTo(cv::Scalar(114, 114, 114));
    } else {
      cv::cuda::GpuMat gpu_original_img;
      gpu_original_img.upload(cpu_img);

      // Calculate scaling and padding (respecting cached/pre-calculated values
      // if available) This logic is similar to the original CPU version but
      // prepares for GPU ops.
      if (meta.scale_to_model == 0.0f && meta.original_w > 0 &&
          meta.original_h > 0) {
        float scale_x = static_cast<float>(model_input_w) / meta.original_w;
        float scale_y = static_cast<float>(model_input_h) / meta.original_h;
        meta.scale_to_model = std::min(scale_x, scale_y);

        int temp_scaled_w =
            static_cast<int>(std::round(meta.original_w * meta.scale_to_model));
        int temp_scaled_h =
            static_cast<int>(std::round(meta.original_h * meta.scale_to_model));

        meta.pad_w_left = (model_input_w - temp_scaled_w) / 2;
        meta.pad_h_top = (model_input_h - temp_scaled_h) / 2;
      } else if (meta.original_w <= 0 || meta.original_h <= 0) {
        std::cerr << "[警告][PreProcGPU] Real image (Frame: "
                  << meta.global_frame_index
                  << ") has invalid original dimensions (" << meta.original_w
                  << "x" << meta.original_h << "). Using letterbox fill."
                  << std::endl;
        gpu_processed_for_model.create(model_input_h, model_input_w, CV_8UC3);
        gpu_processed_for_model.setTo(cv::Scalar(114, 114, 114)); // Fallback
        // Skip to color conversion and normalization of the dummy image
        goto convert_and_normalize; // Not ideal, but an easy way to jump for
                                    // this specific error
      }

      int scaled_w =
          static_cast<int>(std::round(meta.original_w * meta.scale_to_model));
      int scaled_h =
          static_cast<int>(std::round(meta.original_h * meta.scale_to_model));

      // Ensure scaled dimensions are valid for ROI
      if (scaled_w <= 0 || scaled_h <= 0 || meta.pad_w_left < 0 ||
          meta.pad_h_top < 0 || meta.pad_w_left + scaled_w > model_input_w ||
          meta.pad_h_top + scaled_h > model_input_h) {
        std::cerr << "[警告][PreProcGPU] Invalid scaling or padding calculated "
                     "for Frame "
                  << meta.global_frame_index << ". Scaled: " << scaled_w << "x"
                  << scaled_h << ", PadL: " << meta.pad_w_left
                  << ", PadT: " << meta.pad_h_top << ". Using letterbox fill."
                  << std::endl;
        gpu_processed_for_model.create(model_input_h, model_input_w, CV_8UC3);
        gpu_processed_for_model.setTo(cv::Scalar(114, 114, 114));
        goto convert_and_normalize;
      }

      // Create a canvas for letterboxing
      cv::cuda::GpuMat gpu_letterbox_canvas(model_input_h, model_input_w,
                                            CV_8UC3, cv::Scalar(114, 114, 114));

      cv::cuda::GpuMat gpu_resized_img;
      cv::cuda::resize(gpu_original_img, gpu_resized_img,
                       cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_LINEAR);

      // Copy resized image to the padded region on the canvas
      gpu_resized_img.copyTo(gpu_letterbox_canvas(
          cv::Rect(meta.pad_w_left, meta.pad_h_top, scaled_w, scaled_h)));
      gpu_processed_for_model = gpu_letterbox_canvas;
    }
  }

convert_and_normalize: // Label for goto, used in error cases above

  // Color Conversion: BGR to RGB
  cv::cuda::GpuMat gpu_rgb;
  cv::cuda::cvtColor(gpu_processed_for_model, gpu_rgb, cv::COLOR_BGR2RGB);

  // Convert to Float32 and Normalize (0-1 range)
  // Output is still HWC format here, but CV_32FC3
  cv::cuda::GpuMat gpu_float_hwc;
  gpu_rgb.convertTo(gpu_float_hwc, CV_32FC3, 1.0 / 255.0);

  // HWC to CHW planar conversion
  std::vector<cv::cuda::GpuMat> gpu_planes;
  cv::cuda::split(gpu_float_hwc,
                  gpu_planes); // gpu_planes[0]=R, gpu_planes[1]=G,
                               // gpu_planes[2]=B (since input was RGB)

  // chw_planar_output_gpu_buffer_slice is expected to be 1 row, C*H*W columns,
  // CV_32F
  int plane_size_elements = model_input_h * model_input_w;

  // Copy R plane
  gpu_planes[0]
      .reshape(1, plane_size_elements)
      .copyTo(
          chw_planar_output_gpu_buffer_slice.colRange(0, plane_size_elements));
  // Copy G plane
  gpu_planes[1]
      .reshape(1, plane_size_elements)
      .copyTo(chw_planar_output_gpu_buffer_slice.colRange(
          plane_size_elements, 2 * plane_size_elements));
  // Copy B plane
  gpu_planes[2]
      .reshape(1, plane_size_elements)
      .copyTo(chw_planar_output_gpu_buffer_slice.colRange(
          2 * plane_size_elements, 3 * plane_size_elements));
}

void TensorInferencer::performBatchInference(bool pad_batch) {
  if (current_batch_raw_frames_.empty()) {
    return;
  }

  const int ACTUAL_BATCH_SIZE_FOR_GPU =
      pad_batch ? BATCH_SIZE_
                : static_cast<int>(current_batch_raw_frames_.size());
  const int NUM_REAL_FRAMES_IN_CURRENT_PROCESSING_BATCH =
      static_cast<int>(current_batch_raw_frames_.size());

  std::vector<BatchImageMetadata> processing_metadata_for_batch =
      current_batch_metadata_;

  if (pad_batch &&
      NUM_REAL_FRAMES_IN_CURRENT_PROCESSING_BATCH < ACTUAL_BATCH_SIZE_FOR_GPU) {
    int num_to_pad =
        ACTUAL_BATCH_SIZE_FOR_GPU - NUM_REAL_FRAMES_IN_CURRENT_PROCESSING_BATCH;
    for (int k = 0; k < num_to_pad; ++k) {
      BatchImageMetadata dummy_meta;
      dummy_meta.is_real_image = false;
      // For dummy images, original_w/h can be set to model input size,
      // as they won't be used for scaling calculations in preprocess for dummy.
      dummy_meta.original_w = target_w_;
      dummy_meta.original_h = target_h_;
      dummy_meta.scale_to_model = 1.0f; // Not strictly needed for dummy
      dummy_meta.pad_w_left = 0;        // Not strictly needed for dummy
      dummy_meta.pad_h_top = 0;         // Not strictly needed for dummy
      dummy_meta.global_frame_index = -1;
      // No need for original_image_for_callback for dummy images in this GPU
      // path as preprocess_single_image_for_batch will create a dummy GpuMat.
      // However, if post-processing still relies on it for some reason (it
      // shouldn't for dummy), a dummy cv::Mat could be added. For now, assume
      // not needed for dummy in GPU preprocess.
      // dummy_meta.original_image_for_callback = cv::Mat(target_h_, target_w_,
      // CV_8UC3, cv::Scalar(114, 114, 114));
      processing_metadata_for_batch.push_back(dummy_meta);
    }
  }

  // Set binding dimensions for the context based on ACTUAL_BATCH_SIZE_FOR_GPU
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

  // Verify output dimensions (after setting input dimensions)
  Dims outDimsRuntime = context_->getBindingDimensions(outputIndex_);
  size_t total_output_elements = 1;
  bool valid_output_dims = outDimsRuntime.nbDims > 0; // Basic check
  for (int k = 0; k < outDimsRuntime.nbDims; ++k) {
    if (outDimsRuntime.d[k] <= 0) {
      std::cerr << "[错误] 批处理的运行时输出维度无效: " << outDimsRuntime.d[k]
                << " at index " << k << std::endl;
      valid_output_dims = false;
      break;
    }
    total_output_elements *= outDimsRuntime.d[k];
  }
  if (!valid_output_dims || total_output_elements == 0) {
    std::cerr << "[错误] 批处理的运行时输出元素为零或维度无效。" << std::endl;
    return;
  }

  // Check output compatibility (e.g., number of classes)
  // This logic might need adjustment based on your exact model output structure
  if (num_classes_ <= 0 || outDimsRuntime.d[0] != ACTUAL_BATCH_SIZE_FOR_GPU ||
      (outDimsRuntime.nbDims == 3 &&
       outDimsRuntime.d[1] !=
           (4 + num_classes_))) { // Example for YOLO-like output
    std::cerr << "[错误] 批处理推理: 运行时输出属性/批处理不匹配。输出维度: ";
    for (int k = 0; k < outDimsRuntime.nbDims; ++k)
      std::cerr << outDimsRuntime.d[k] << " ";
    std::cerr << "期望批处理: " << ACTUAL_BATCH_SIZE_FOR_GPU
              << ", 期望属性 (4+num_classes): " << (4 + num_classes_)
              << std::endl;
    // return; // Commenting out to see if it runs, but this is a critical check
  }
  int num_detections_per_image_from_engine = 0;
  int num_attributes_from_engine = 0;

  if (outDimsRuntime.nbDims ==
      3) { // Common for [batch, num_attributes, num_detections_boxes]
    num_attributes_from_engine =
        outDimsRuntime.d[1]; // Or d[2] depending on layout
    num_detections_per_image_from_engine = outDimsRuntime.d[2]; // Or d[1]
    if (num_attributes_from_engine !=
        (4 + num_classes_)) { // Verify this check matches your model's [batch,
                              // attributes, detections] or [batch, detections,
                              // attributes]
      std::cerr
          << "[警告] 批处理输出维度1/2 (属性) " << num_attributes_from_engine
          << " 与期望的 " << (4 + num_classes_)
          << " 不匹配. The order might be [batch, detections, attributes]."
          << std::endl;
      // Try swapping if it's [batch, detections, attributes]
      // This is a common source of error, ensure your model output format is
      // correctly interpreted. For YOLO traditionally it's [batch,
      // num_detections, 4_coords+conf+num_classes] So d[1] should be
      // num_detections, d[2] should be attributes Let's assume engine output is
      // [batch, num_detections, num_attributes] based on typical TRT examples
      // for YOLO If outputIndex_ is for "output0" which is often [batch_size,
      // num_boxes, num_classes+4+1] Let's assume d[0] = batch, d[1]=num_boxes,
      // d[2]=num_attributes
      num_detections_per_image_from_engine = outDimsRuntime.d[1];
      num_attributes_from_engine = outDimsRuntime.d[2];
      if (num_attributes_from_engine != (4 + num_classes_)) {
        std::cerr << "[错误] Re-checked: 批处理输出维度2 (属性) "
                  << num_attributes_from_engine << " 与期望的 "
                  << (4 + num_classes_) << " 不匹配" << std::endl;
        return;
      }
    }
  } else { // Add more checks if your model output is different, e.g. [batch,
           // num_coords + num_classes, num_detections]
    std::cerr << "[错误] 批处理推理的输出维度意外。得到 "
              << outDimsRuntime.nbDims
              << "D。期望3D [batch, detections, attributes] or similar."
              << std::endl;
    return;
  }

  // Allocate GPU memory for input and output
  // Free previously allocated memory if any (safer for dynamic batch sizes)
  if (inputDevice_)
    cudaFree(inputDevice_);
  inputDevice_ = nullptr;
  if (outputDevice_)
    cudaFree(outputDevice_);
  outputDevice_ = nullptr;

  size_t single_image_elements = static_cast<size_t>(3) * target_h_ * target_w_;
  size_t total_input_bytes = static_cast<size_t>(ACTUAL_BATCH_SIZE_FOR_GPU) *
                             single_image_elements * sizeof(float);
  size_t total_output_bytes =
      total_output_elements *
      sizeof(float); // total_output_elements already calculated

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

  // Preprocess images and fill inputDevice_ directly on GPU
  for (int i = 0; i < ACTUAL_BATCH_SIZE_FOR_GPU; ++i) {
    cv::Mat current_raw_img_for_preprocessing_cpu;
    if (i < NUM_REAL_FRAMES_IN_CURRENT_PROCESSING_BATCH) {
      current_raw_img_for_preprocessing_cpu = current_batch_raw_frames_[i];
    } else {
      // For dummy padding images, cpu_img can be empty.
      // preprocess_single_image_for_batch handles dummy image creation on GPU.
      // processing_metadata_for_batch[i] should have is_real_image = false
    }

    // Create a GpuMat wrapper for the slice of inputDevice_ for the current
    // image
    float *current_input_slice_ptr =
        reinterpret_cast<float *>(inputDevice_) + i * single_image_elements;
    cv::cuda::GpuMat chw_output_slice_gpu_wrapper(
        1, static_cast<int>(single_image_elements), CV_32F,
        current_input_slice_ptr);

    preprocess_single_image_for_batch(current_raw_img_for_preprocessing_cpu,
                                      processing_metadata_for_batch[i],
                                      target_w_, // model input width
                                      target_h_, // model input height
                                      chw_output_slice_gpu_wrapper);
  }

  // Set TRT bindings
  bindings_[inputIndex_] = inputDevice_;
  bindings_[outputIndex_] = outputDevice_;

  // No HostToDevice cudaMemcpy needed for inputDevice_ as it's already filled
  // on GPU

  if (!context_->enqueueV2(bindings_.data(), 0,
                           nullptr)) { // Using default stream 0
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

  // Post-processing (same as before)
  std::vector<InferenceResult> batch_inference_results_for_callback;
  for (int i = 0; i < NUM_REAL_FRAMES_IN_CURRENT_PROCESSING_BATCH;
       ++i) { // Only process real frames
    const BatchImageMetadata &frame_meta = processing_metadata_for_batch[i];

    if (!frame_meta.is_real_image) { // Should not happen if loop is up to
                                     // NUM_REAL_FRAMES...
      std::cerr << "[警告][PerformBatch] Attempting to post-process a non-real "
                   "frame at batch index "
                << i << ". Skipping." << std::endl;
      continue;
    }
    if (frame_meta.original_image_for_callback.empty()) {
      std::cerr << "[警告][PerformBatch] Real frame (Frame: "
                << frame_meta.global_frame_index
                << ") original image is empty before post-processing. Skipping."
                << std::endl;
      InferenceResult res;
      res.info = "Error: Input image was empty for Frame " +
                 std::to_string(frame_meta.global_frame_index);
      batch_inference_results_for_callback.push_back(res);
      continue;
    }

    // The start of output for this image in the batched raw output
    // Ensure your understanding of num_attributes_from_engine and
    // num_detections_per_image_from_engine matches the actual layout [batch,
    // attributes, detections] or [batch, detections, attributes]
    const float *output_for_this_image_start =
        host_output_batched_raw.data() +
        static_cast<size_t>(i) * num_attributes_from_engine *
            num_detections_per_image_from_engine;

    std::vector<InferenceResult> single_frame_results;
    process_single_output(
        frame_meta, output_for_this_image_start,
        num_detections_per_image_from_engine, // Pass the correct dimension
        num_attributes_from_engine,           // Pass the correct dimension
        i, single_frame_results);

    batch_inference_results_for_callback.insert(
        batch_inference_results_for_callback.end(),
        single_frame_results.begin(), single_frame_results.end());
  }

  if (current_callback_ && NUM_REAL_FRAMES_IN_CURRENT_PROCESSING_BATCH > 0) {
    current_callback_(batch_inference_results_for_callback);
  } else if (!current_callback_ &&
             NUM_REAL_FRAMES_IN_CURRENT_PROCESSING_BATCH > 0) {
    std::cerr << "[错误][PerformBatch] 回调函数未设置，但有帧需要处理！"
              << std::endl;
  }
}

// process_single_output and NMS functions remain the same
// ... (rest of the file: process_single_output, calculateIoU, applyNMS,
// saveAnnotatedImage) Ensure these functions are present in your actual file.
// I'm omitting them here for brevity as they were not the primary focus of the
// GPU pre-processing change. Make sure the includes for these are correct.

// ... (Make sure the definitions for process_single_output, calculateIoU,
// applyNMS, saveAnnotatedImage are included after this point)
void TensorInferencer::process_single_output(
    const BatchImageMetadata &image_meta,
    const float *host_output_for_image_raw, int num_detections_in_slice,
    int num_attributes_per_detection, int original_batch_idx_for_debug,
    std::vector<InferenceResult> &frame_results) {

  // This transposition logic depends on the output format of your TensorRT
  // engine. If engine output is already [detections, attributes] per image,
  // this might not be needed or might need to be [attributes, detections] ->
  // [detections, attributes]. Original code had:
  // host_output_for_image_raw[attr_idx * num_detections_in_slice + det_idx]
  // This implies the raw TRT output buffer for one image is planar by
  // attribute. E.g., [all_cx, all_cy, all_w, all_h, all_cls0_score, ...,
  // all_clsN_score] We need to convert it to a list of detections, where each
  // detection has its attributes together.
  std::vector<float> transposed_output(
      static_cast<size_t>(num_detections_in_slice) *
      num_attributes_per_detection);
  for (int det_idx = 0; det_idx < num_detections_in_slice; ++det_idx) {
    for (int attr_idx = 0; attr_idx < num_attributes_per_detection;
         ++attr_idx) {
      transposed_output[static_cast<size_t>(det_idx) *
                            num_attributes_per_detection +
                        attr_idx] =
          host_output_for_image_raw[static_cast<size_t>(
                                        attr_idx) * // This is attribute-major
                                        num_detections_in_slice +
                                    det_idx];
    }
  }
  // If your model output is already [batch, num_detections,
  // attributes_per_detection], then host_output_for_image_raw points to the
  // start of [num_detections, attributes_per_detection] and you can directly
  // use it without transposition, or the transposition logic is simpler. const
  // float* det_attrs = host_output_for_image_raw + det_idx *
  // num_attributes_per_detection;

  auto it = class_name_to_id_.find(this->object_name_);
  if (it == class_name_to_id_.end()) {
    std::cerr << "[错误][ProcessOutput] (Frame: "
              << image_meta.global_frame_index << ") 目标对象名称 '"
              << this->object_name_ << "' 在类别名称中未找到。" << std::endl;
    InferenceResult res;
    res.info = "Error for Frame " +
               std::to_string(image_meta.global_frame_index) +
               ": Target object name '" + this->object_name_ +
               "' not found in class names.";
    frame_results.push_back(res);
    return;
  }
  int target_class_id = it->second;
  float confidence_threshold = this->confidence_;

  std::vector<Detection> detected_objects;
  for (int i = 0; i < num_detections_in_slice; ++i) {
    // Use the transposed_output:
    const float *det_attrs = &transposed_output[static_cast<size_t>(i) *
                                                num_attributes_per_detection];
    // If no transposition was needed because engine output is already
    // [detections, attributes]: const float *det_attrs =
    // host_output_for_image_raw + (static_cast<size_t>(i) *
    // num_attributes_per_detection);

    float max_score = 0.0f;
    int best_class_id = -1;
    // Assumes class scores start at index 4 of det_attrs (after cx, cy, w, h)
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
          std::min(static_cast<float>(target_w_ - 1),
                   cx + w / 2.0f); // target_w_ is model input width
      float y2_model =
          std::min(static_cast<float>(target_h_ - 1),
                   cy + h / 2.0f); // target_h_ is model input height
      if (x2_model > x1_model && y2_model > y1_model) {
        detected_objects.push_back({x1_model, y1_model, x2_model, y2_model,
                                    max_score, best_class_id,
                                    original_batch_idx_for_debug,
                                    image_meta.is_real_image ? "REAL" : "PAD"});
      }
    }
  }

  std::vector<Detection> nms_detections = applyNMS(detected_objects, 0.45f);

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

    saveAnnotatedImage(det, image_meta, static_cast<int>(i));

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

void TensorInferencer::saveAnnotatedImage(const Detection &det,
                                          const BatchImageMetadata &image_meta,
                                          int detection_idx_in_image) {

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

  float timestamp_sec =
      static_cast<float>(image_meta.global_frame_index) / 30.0f;

  std::ostringstream filename_oss;
  filename_oss << image_output_path_ << "/frame" << std::setw(6)
               << std::setfill('0') << image_meta.global_frame_index << "_time"
               << std::fixed << std::setprecision(2) << timestamp_sec << "s"
               << "_obj" << std::setw(2) << std::setfill('0')
               << detection_idx_in_image << "_" << this->object_name_ << "_conf"
               << static_cast<int>(det.confidence * 100) << ".jpg";

  bool success = cv::imwrite(filename_oss.str(), img_to_save);
  if (!success) {
    std::cerr << "[错误] Saving image failed: " << filename_oss.str()
              << std::endl;
  }
}
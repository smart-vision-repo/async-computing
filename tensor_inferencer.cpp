#include "tensor_inferencer.hpp" // 请确保头文件中有 bindings_ 的声明
#include <algorithm>             // For std::sort, std::max, std::min
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric> // For std::iota if needed, not used here but good include for vector ops

// Forward declaration for Detection struct if not in .hpp, assuming it's like
// this: struct Detection {
//    float x1, y1, x2, y2, confidence;
//    int class_id;
// };
// Make sure this matches the definition used by applyNMS and processOutput.
// Based on your applyNMS, it seems it is:
// struct Detection {
//    float x1, y1, x2, y2; // Coordinates
//    float confidence;    // Confidence score
//    int class_id;        // Class ID
// };
// Let's assume this struct is defined in tensor_inferencer.hpp

using namespace nvinfer1;

// Logger class (no changes from your original)
class Logger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <=
        Severity::kWARNING) { // Log kINFO, kWARNING, kERROR, kINTERNAL_ERROR
      std::cout << "[TRT] " << msg << std::endl;
    }
  }
} gLogger;

// readEngineFile function (no changes from your original)
static std::vector<char> readEngineFile(const std::string &enginePath) {
  std::ifstream file(enginePath, std::ios::binary);
  if (!file.good()) { // More robust check
    std::cerr << "[ERROR] Failed to open engine file: " << enginePath
              << std::endl;
    return {}; // Return empty vector on failure
  }
  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> engineData(size);
  if (size > 0) { // Ensure size is positive before reading
    file.read(engineData.data(), size);
  }
  file.close();
  return engineData;
}

// roundToNearestMultiple function (no changes from your original)
static int roundToNearestMultiple(int val, int base = 32) {
  return ((val + base / 2) / base) * base;
}

TensorInferencer::TensorInferencer(int video_height, int video_width)
    : inputDevice_(nullptr),
      outputDevice_(nullptr),       // Initialize device pointers
      inputSize_(0), outputSize_(0) // Initialize sizes
{
  std::cout << "[INIT] 初始化 TensorInferencer，视频尺寸: " << video_width
            << "x" << video_height << std::endl;

  target_w_ = roundToNearestMultiple(video_width, 32);
  target_h_ = roundToNearestMultiple(video_height, 32);

  std::cout << "[INIT] 计算的目标尺寸 (32对齐前): " << target_w_ << "x"
            << target_h_ << std::endl;

  const char *env_path = std::getenv("YOLO_ENGINE_NAME");
  if (!env_path) {
    std::cerr << "[ERROR] 环境变量 YOLO_ENGINE_NAME 未设置" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  engine_path_ = env_path; // Store for potential re-use or logging

  const char *names_path_env = std::getenv("YOLO_COCO_NAMES");
  if (!names_path_env) {
    std::cerr << "[ERROR] 环境变量 YOLO_COCO_NAMES 未设置" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string names_path_str = names_path_env;

  const char *output_path_env = std::getenv("YOLO_IMAGE_PATH");
  if (!output_path_env) {
    std::cerr << "[ERROR] 环境变量 YOLO_IMAGE_PATH 未设置" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  image_output_path_ = output_path_env;

  std::cout << "[INIT] 加载引擎文件: " << engine_path_ << std::endl;

  auto engineData = readEngineFile(engine_path_);
  if (engineData.empty()) {
    std::cerr << "[ERROR] 无法读取引擎数据." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  runtime_ = createInferRuntime(gLogger);
  assert(runtime_ != nullptr && "TensorRT runtime creation failed.");
  engine_ =
      runtime_->deserializeCudaEngine(engineData.data(), engineData.size());
  assert(engine_ != nullptr && "TensorRT engine deserialization failed.");
  context_ = engine_->createExecutionContext();
  assert(context_ != nullptr && "TensorRT execution context creation failed.");

  std::cout << "[INIT] 引擎加载成功" << std::endl;

  bindings_.resize(engine_->getNbBindings()); // Initialize bindings vector

  printEngineInfo(); // Print info after engine is loaded

  inputIndex_ = engine_->getBindingIndex("images");
  // A more robust way to get output index, assuming it's the first non-input
  // binding after "images" or simply the one named "output0" or similar. The
  // original engine_->getBindingName(1) is okay if bindings are ordered input
  // then output. Let's try to find "output0" or fall back.
  int tempOutputIndex = engine_->getBindingIndex("output0");
  if (tempOutputIndex < 0) {
    std::cout
        << "[WARN] Output tensor 'output0' not found by name. Trying index 1."
        << std::endl;
    if (engine_->getNbBindings() > 1 && inputIndex_ == 0 &&
        !engine_->bindingIsInput(1)) {
      tempOutputIndex = 1;
    } else if (engine_->getNbBindings() > 1 && inputIndex_ == 1 &&
               !engine_->bindingIsInput(0)) {
      tempOutputIndex = 0; // If input was index 1
    } else { // Fallback or more complex logic needed if many bindings
      for (int i = 0; i < engine_->getNbBindings(); ++i) {
        if (!engine_->bindingIsInput(i)) {
          tempOutputIndex = i;
          std::cout << "[INFO] Found first output tensor '"
                    << engine_->getBindingName(i) << "' at index " << i
                    << std::endl;
          break;
        }
      }
    }
  }
  outputIndex_ = tempOutputIndex;

  if (inputIndex_ < 0) {
    std::cerr << "[ERROR] Input tensor 'images' not found in engine."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (outputIndex_ < 0) {
    std::cerr << "[ERROR] Could not determine a valid output tensor index."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::cout << "[INIT] 输入索引 ('images'): " << inputIndex_ << ", 输出索引 ('"
            << engine_->getBindingName(outputIndex_) << "'): " << outputIndex_
            << std::endl;

  std::ifstream infile(names_path_str);
  if (!infile.is_open()) {
    std::cerr << "[ERROR] 无法打开类别文件: " << names_path_str << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string line;
  int idx = 0;
  while (std::getline(infile, line)) {
    if (!line.empty()) {
      class_name_to_id_[line] = idx;
      id_to_class_name_[idx] = line; // Store reverse mapping as well
      idx++;
    }
  }
  num_classes_ = class_name_to_id_.size(); // Store number of classes

  std::cout << "[INIT] 加载了 " << num_classes_ << " 个类别" << std::endl;
  if (class_name_to_id_.count("dog")) {
    std::cout << "[INIT] dog 类别ID: " << class_name_to_id_["dog"] << std::endl;
  } else {
    std::cout << "[WARN] 'dog' 类别未在COCO names文件中找到!" << std::endl;
  }

  // This part keeps your "original input logic" for target_w_ and target_h_
  Dims reportedInputDims = engine_->getBindingDimensions(inputIndex_);
  if (reportedInputDims.nbDims == 4) { // Check if dimensions are somewhat valid
    bool useEngineDims = true;
    // profile 0 opt dims are often -1 if not set, or actual values.
    // If any dim is <=0 it might mean it's fully dynamic from profile, not a
    // "fixed" size to override with.
    for (int i = 0; i < reportedInputDims.nbDims; ++i) {
      if (reportedInputDims.d[i] <= 0) {
        useEngineDims = false;
        break;
      }
    }
    if (useEngineDims && reportedInputDims.d[2] > 0 &&
        reportedInputDims.d[3] > 0) {
      // Engine reports specific H, W for the default context (likely profile 0
      // opt)
      target_h_ = reportedInputDims.d[2];
      target_w_ = reportedInputDims.d[3];
      std::cout << "[INIT] 使用引擎优化配置文件中的尺寸 (profile 0 opt): "
                << target_w_ << "x" << target_h_ << std::endl;
    } else {
      std::cout << "[INIT] 引擎输入维度是动态的或无效 ("
                << reportedInputDims.d[2] << "x" << reportedInputDims.d[3]
                << "), 将使用基于视频尺寸计算的目标尺寸: " << target_w_ << "x"
                << target_h_ << std::endl;
    }
  } else {
    std::cout
        << "[INIT] 无法获取引擎固定输入维度, 将使用基于视频尺寸计算的目标尺寸: "
        << target_w_ << "x" << target_h_ << std::endl;
  }

  // Ensure input data type is FP32 as confirmed by trtexec
  if (engine_->getBindingDataType(inputIndex_) != nvinfer1::DataType::kFLOAT) {
    std::cerr << "[ERROR] 引擎输入张量 '"
              << engine_->getBindingName(inputIndex_)
              << "' 不是期望的 DataType::kFLOAT!" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "[INFO] 引擎输入张量 '" << engine_->getBindingName(inputIndex_)
            << "' 确认是 DataType::kFLOAT." << std::endl;

  // Pre-allocate GPU memory if target_w_ and target_h_ are now considered fixed
  // For simplicity in this revision, keeping allocation in infer(), but this is
  // an optimization point.
}

void TensorInferencer::printEngineInfo() {
  std::cout << "=== 引擎信息 ===" << std::endl;
  std::cout << "引擎名称: " << (engine_->getName() ? engine_->getName() : "N/A")
            << std::endl;
  std::cout << "绑定数量: " << engine_->getNbBindings() << std::endl;

  for (int i = 0; i < engine_->getNbBindings(); ++i) {
    const char *name = engine_->getBindingName(i);
    Dims dims = engine_->getBindingDimensions(
        i); // These are from opt profile for dynamic anker
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
      dtype_str = "Unknown";
      break;
    }

    std::cout << "绑定 " << i << ": '" << name << "' ("
              << (isInput ? "输入" : "输出") << ") - 类型: " << dtype_str
              << " - 优化配置维度: ";

    for (int j = 0; j < dims.nbDims; ++j) {
      std::cout << dims.d[j]; // dims.d[k] can be -1 for dynamic shapes
      if (j < dims.nbDims - 1)
        std::cout << "x";
    }
    std::cout << std::endl;
  }
  std::cout << "================" << std::endl;
}

TensorInferencer::~TensorInferencer() {
  // Ensure cudaFree is only called on non-null pointers
  if (inputDevice_) {
    cudaFree(inputDevice_);
    inputDevice_ = nullptr;
  }
  if (outputDevice_) {
    cudaFree(outputDevice_);
    outputDevice_ = nullptr;
  }
  if (context_) {
    context_->destroy();
    context_ = nullptr;
  }
  if (engine_) {
    engine_->destroy();
    engine_ = nullptr;
  }
  if (runtime_) {
    runtime_->destroy();
    runtime_ = nullptr;
  }
  std::cout << "[DEINIT] TensorInferencer 已销毁." << std::endl;
}

// This overload seems to be for a different use case, keeping it as is.
bool TensorInferencer::infer(const std::vector<float> &input,
                             std::vector<float> &output) {
  if (inputSize_ == 0 || outputSize_ == 0) {
    std::cerr
        << "[ERROR] inputSize/outputSize 未初始化 (infer with vector<float>)"
        << std::endl;
    return false;
  }
  if (input.size() != inputSize_) {
    std::cerr << "[ERROR] Input vector size mismatch. Expected " << inputSize_
              << ", got " << input.size() << std::endl;
    return false;
  }
  output.resize(outputSize_);

  // Ensure bindings_ vector is correctly populated if this method is used
  // This might need specific setup if inputDevice_ and outputDevice_ are not
  // for this. For now, assume bindings_[inputIndex_] and
  // bindings_[outputIndex_] are correctly set elsewhere or this overload needs
  // its own GPU buffer management. The provided code uses `bindings_` without
  // clear class member declaration. Assuming it's fixed now.
  if (bindings_.empty() || bindings_[inputIndex_] == nullptr ||
      bindings_[outputIndex_] == nullptr) {
    std::cerr << "[ERROR] Device bindings not properly set up for "
                 "vector<float> inference."
              << std::endl;
    return false;
  }

  cudaMemcpy(bindings_[inputIndex_], input.data(), inputSize_ * sizeof(float),
             cudaMemcpyHostToDevice);
  context_->enqueueV2(bindings_.data(), 0, nullptr); // Use bindings_.data()
  cudaMemcpy(output.data(), bindings_[outputIndex_],
             outputSize_ * sizeof(float), cudaMemcpyDeviceToHost);
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
  std::cout << "[INFER] 预处理目标尺寸: " << target_w_ << "x" << target_h_
            << std::endl;

  // --- 图像预处理 ---
  cv::Mat resized_direct_img; // Renamed to avoid confusion with letterboxed
                              // 'img' if we were to add it
  cv::resize(raw_img, resized_direct_img,
             cv::Size(target_w_, target_h_)); // Direct resize

  cv::Mat img_rgb;
  cv::cvtColor(resized_direct_img, img_rgb, cv::COLOR_BGR2RGB); // BGR to RGB

  int c = 3, h = target_h_, w = target_w_;
  size_t current_input_size_elements =
      static_cast<size_t>(c) * h * w; // Use size_t for safety

  cv::Mat chw_input_fp32; // Renamed for clarity
  img_rgb.convertTo(chw_input_fp32, CV_32FC3,
                    1.0 / 255.0); // Normalize to [0,1]

  std::vector<float> input_data(current_input_size_elements);

  // CHW格式转换 (Planar RGB: RRR...GGG...BBB...)
  for (int ch = 0; ch < c; ++ch) { // Iterate channels R, G, B
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        // input_data[ch * h * w + y * w + x] = chw_input_fp32.at<cv::Vec3f>(y,
        // x)[ch]; A common way for CHW is: R plane, then G plane, then B plane
        // R channel (index 0 in RGB cv::Vec3f)
        // G channel (index 1 in RGB cv::Vec3f)
        // B channel (index 2 in RGB cv::Vec3f)
        input_data[static_cast<size_t>(ch) * h * w +
                   static_cast<size_t>(y) * w + x] =
            chw_input_fp32.at<cv::Vec3f>(y, x)[ch];
      }
    }
  }
  std::cout << "[INFER] 图像预处理完成 (RGB, Planar, Normalized FP32)"
            << std::endl;

  // --- 设置动态输入尺寸 ---
  Dims inputDimsRuntime{4, {1, 3, h, w}}; // batch_size=1
  if (!context_->setBindingDimensions(inputIndex_, inputDimsRuntime)) {
    std::cerr << "[ERROR] 无法为输入张量设置绑定维度: " << w << "x" << h
              << std::endl;
    return false; // Critical error
  }
  // std::cout << "[INFER] 输入绑定维度已设置为: 1x3x" << h << "x" << w <<
  // std::endl;

  if (!context_->allInputDimensionsSpecified()) {
    std::cerr << "[ERROR] 非所有输入维度都已指定 (可能是多输入模型问题)"
              << std::endl;
    return false;
  }

  // --- 计算输出大小 (基于当前上下文的绑定维度) ---
  Dims outDims = context_->getBindingDimensions(outputIndex_);
  size_t current_output_size_elements = 1;
  std::cout << "[INFER] 输出张量 '" << engine_->getBindingName(outputIndex_)
            << "' 运行时维度: ";
  for (int i = 0; i < outDims.nbDims; ++i) {
    std::cout << outDims.d[i] << (i == outDims.nbDims - 1 ? "" : "x");
    if (outDims.d[i] <=
        0) { // Should not happen after setBindingDimensions if engine is valid
      std::cerr << "\n[ERROR] 输出维度无效: " << outDims.d[i] << " at index "
                << i << std::endl;
      return false;
    }
    current_output_size_elements *= outDims.d[i];
  }
  std::cout << std::endl;

  if (num_classes_ <= 0 || outDims.d[1] != (4 + num_classes_)) {
    std::cerr << "[ERROR] 输出维度属性与类别数不匹配. Outdim[1]="
              << outDims.d[1] << ", Expected 4+" << num_classes_ << std::endl;
    return false;
  }

  // --- 分配/检查 GPU内存 ---
  // For this revision, we keep per-inference allocation as per original
  // structure. This is an optimization point for later.
  if (inputDevice_ != nullptr)
    cudaFree(inputDevice_);
  if (outputDevice_ != nullptr)
    cudaFree(outputDevice_);

  size_t input_size_bytes = current_input_size_elements * sizeof(float);
  size_t output_size_bytes =
      current_output_size_elements *
      sizeof(float); // Assuming output is FP32 based on trtexec

  cudaError_t err;
  err = cudaMalloc(&inputDevice_, input_size_bytes);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] CUDA Malloc Input: " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }
  err = cudaMalloc(&outputDevice_, output_size_bytes);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] CUDA Malloc Output: " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }

  bindings_[inputIndex_] = inputDevice_;
  bindings_[outputIndex_] = outputDevice_;
  // std::cout << "[INFER] GPU内存分配完成. Input bytes: " << input_size_bytes
  // << ", Output bytes: " << output_size_bytes << std::endl;

  // --- 执行推理 ---
  err = cudaMemcpy(inputDevice_, input_data.data(), input_size_bytes,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] CUDA Memcpy H2D: " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }

  if (!context_->enqueueV2(bindings_.data(), 0, nullptr)) {
    std::cerr << "[ERROR] 推理执行失败 (enqueueV2)" << std::endl;
    return false;
  }

  std::vector<float> host_output_raw(
      current_output_size_elements); // Raw from GPU
  err = cudaMemcpy(host_output_raw.data(), outputDevice_, output_size_bytes,
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] CUDA Memcpy D2H: " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }

  // std::cout << "[INFER] 输出数据拷贝完成，开始处理结果" << std::endl;
  processOutput(input, host_output_raw, raw_img, outDims); // Pass outDims
  return true;
}

// processOutput: Now takes runtime output dimensions
void TensorInferencer::processOutput(
    const InferenceInput &input, const std::vector<float> &host_output_raw,
    const cv::Mat &raw_img,
    const nvinfer1::Dims &outDimsRuntime) { // Receive runtime output dims

  int num_attributes_from_engine = outDimsRuntime.d[1]; // e.g., 84
  int num_detections_from_engine = outDimsRuntime.d[2]; // e.g., 11109

  if (host_output_raw.size() !=
      static_cast<size_t>(outDimsRuntime.d[0] * num_attributes_from_engine *
                          num_detections_from_engine)) {
    std::cerr << "[ERROR][OUTPUT] host_output_raw size mismatch with expected "
                 "output dimensions."
              << std::endl;
    return;
  }

  // Transpose output from [1, num_attributes, num_detections] to
  // [num_detections, num_attributes] This makes iterating through detections
  // easier.
  std::vector<float> transposed_output(
      static_cast<size_t>(num_detections_from_engine) *
      num_attributes_from_engine);
  for (int det_idx = 0; det_idx < num_detections_from_engine; ++det_idx) {
    for (int attr_idx = 0; attr_idx < num_attributes_from_engine; ++attr_idx) {
      transposed_output[static_cast<size_t>(det_idx) *
                            num_attributes_from_engine +
                        attr_idx] =
          host_output_raw[static_cast<size_t>(attr_idx) *
                              num_detections_from_engine +
                          det_idx];
    }
  }
  // std::cout << "[OUTPUT] Output transposed. Num_detections_from_engine: " <<
  // num_detections_from_engine
  //           << ", Num_attributes_from_engine: " << num_attributes_from_engine
  //           << std::endl;

  auto it = class_name_to_id_.find(input.object_name);
  if (it == class_name_to_id_.end()) {
    std::cerr << "[ERROR][OUTPUT] 类别 '" << input.object_name << "' 未找到!"
              << std::endl;
    return;
  }
  int target_class_id = it->second;
  // std::cout << "[OUTPUT] 目标类别ID: " << target_class_id << std::endl;

  float confidence_threshold = std::max(
      0.1f, input.confidence_thresh); // Ensure a minimum sensible threshold
  // std::cout << "[OUTPUT] 使用置信度阈值: " << confidence_threshold <<
  // std::endl;

  std::vector<Detection>
      detected_objects; // Renamed from 'detections' to avoid conflict

  for (int i = 0; i < num_detections_from_engine; ++i) {
    const float *det_attrs =
        &transposed_output[static_cast<size_t>(i) * num_attributes_from_engine];

    // Extract class scores (attributes from index 4 onwards)
    // The raw scores are assumed to be probabilities (no sigmoid needed here
    // based on YOLOv8 ONNX std)
    float max_score = 0.0f;
    int best_class_id = -1;
    for (int j = 0; j < num_classes_; ++j) { // Use num_classes_ member
      float score = det_attrs[4 + j];
      if (score > max_score) {
        max_score = score;
        best_class_id = j;
      }
    }

    if (best_class_id == target_class_id && max_score >= confidence_threshold) {
      // Coordinates are cx, cy, w, h relative to target_w_, target_h_
      float cx = det_attrs[0];
      float cy = det_attrs[1];
      float w = det_attrs[2];
      float h = det_attrs[3];

      float x1 = std::max(0.0f, cx - w / 2.0f);
      float y1 = std::max(0.0f, cy - h / 2.0f);
      float x2 = std::min(static_cast<float>(target_w_ - 1),
                          cx + w / 2.0f); // Clamp to target_w_/h_
      float y2 = std::min(static_cast<float>(target_h_ - 1), cy + h / 2.0f);

      if (x2 > x1 && y2 > y1) { // Ensure valid box
        Detection detection = {x1, y1, x2, y2, max_score, best_class_id};
        detected_objects.push_back(detection);

        //   std::cout << "[DETECTION] 发现 " << input.object_name
        //             << "! GOP: " << input.gopIdx << ", 置信度: " << max_score
        //             << ", 模型坐标框: (" << x1 << "," << y1 << "," << x2 <<
        //             "," << y2
        //             << ")" << std::endl;
      }
    }
  }

  std::vector<Detection> nms_detections =
      applyNMS(detected_objects, 0.45f); // Using 0.45 IoU for NMS
  std::cout << "[NMS] " << input.object_name
            << ": 原始检测数=" << detected_objects.size()
            << ", NMS后剩余=" << nms_detections.size() << std::endl;

  for (size_t i = 0; i < nms_detections.size(); ++i) {
    const auto &det = nms_detections[i];
    // Call the version of saveAnnotatedImage that includes drawing
    saveAnnotatedImage(raw_img, det.x1, det.y1, det.x2, det.y2, det.confidence,
                       input.object_name, input.gopIdx, static_cast<int>(i));
  }

  if (nms_detections.empty()) {
    std::cout << "[INFO] 未检测到 '" << input.object_name
              << "' (GOP: " << input.gopIdx << ") 满足条件." << std::endl;
  }
}

// calculateIoU and applyNMS (no changes from your original, assuming Detection
// struct matches)
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

  return union_area > 0 ? intersection_area / union_area : 0.0f;
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

// Using the more detailed saveAnnotatedImage that draws boxes and text
void TensorInferencer::saveAnnotatedImage(
    const cv::Mat &raw_img, float x1_model, float y1_model, float x2_model,
    float y2_model, float confidence, const std::string &class_name, int gopIdx,
    int detection_idx) { // Added detection_idx
  // std::cout << "[SAVE] 准备保存标注图像 for detection_idx " << detection_idx
  // << std::endl; std::cout << "[SAVE] 类别: " << class_name << ", 置信度: " <<
  // confidence << std::endl; std::cout << "[SAVE] 原图尺寸: " << raw_img.cols
  // << "x" << raw_img.rows << std::endl; std::cout << "[SAVE] 检测框(模型坐标 "
  // << target_w_ << "x" << target_h_ << "): ("
  //           << x1_model << "," << y1_model << ") - (" << x2_model << "," <<
  //           y2_model << ")" << std::endl;

  cv::Mat img_to_save = raw_img.clone();

  // Coordinates from processOutput (x1_model, etc.) are relative to target_w_,
  // target_h_ (input to model) Scale them back to raw_img dimensions. This
  // scaling assumes direct resize was used in preprocessing.
  float scale_x =
      static_cast<float>(raw_img.cols) / static_cast<float>(target_w_);
  float scale_y =
      static_cast<float>(raw_img.rows) / static_cast<float>(target_h_);

  int x1_scaled = static_cast<int>(std::round(x1_model * scale_x));
  int y1_scaled = static_cast<int>(std::round(y1_model * scale_y));
  int x2_scaled = static_cast<int>(std::round(x2_model * scale_x));
  int y2_scaled = static_cast<int>(std::round(y2_model * scale_y));

  // std::cout << "[SAVE] 缩放比例: " << scale_x << "x" << scale_y << std::endl;
  // std::cout << "[SAVE] 映射后框(原图坐标): (" << x1_scaled << "," <<
  // y1_scaled << ") - ("
  //           << x2_scaled << "," << y2_scaled << ")" << std::endl;

  // Boundary check
  x1_scaled = std::max(0, std::min(x1_scaled, raw_img.cols - 1));
  y1_scaled = std::max(0, std::min(y1_scaled, raw_img.rows - 1));
  x2_scaled = std::max(0, std::min(x2_scaled, raw_img.cols - 1));
  y2_scaled = std::max(0, std::min(y2_scaled, raw_img.rows - 1));

  if (x2_scaled <= x1_scaled || y2_scaled <= y1_scaled) {
    std::cout << "[WARN][SAVE] Scaled box is invalid, skipping save for this "
                 "detection."
              << std::endl;
    return;
  }

  // Draw rectangle
  cv::rectangle(img_to_save, cv::Point(x1_scaled, y1_scaled),
                cv::Point(x2_scaled, y2_scaled), cv::Scalar(0, 255, 0),
                2); // Thickness 2

  // Prepare label
  std::ostringstream label;
  label << class_name << " " << std::fixed << std::setprecision(2)
        << confidence;
  int baseline = 0;
  cv::Size textSize =
      cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.7, 1,
                      &baseline); // Font scale 0.7, thickness 1

  // Position for label background and text
  cv::Point textOrg(x1_scaled,
                    y1_scaled - textSize.height - 3); // Position above box
  if (textOrg.y <
      0) { // If label goes off screen, put it inside the box at the top
    textOrg.y = y1_scaled + textSize.height + 3;
  }
  if (textOrg.y + baseline > raw_img.rows) { // If still off, adjust
    textOrg.y = y1_scaled + baseline + 3;
  }

  cv::rectangle(img_to_save,
                cv::Point(textOrg.x, textOrg.y - textSize.height -
                                         baseline), // Adjust for text box
                cv::Point(textOrg.x + textSize.width, textOrg.y + baseline),
                cv::Scalar(0, 255, 0), cv::FILLED);
  cv::putText(img_to_save, label.str(),
              cv::Point(textOrg.x, textOrg.y), // Use adjusted textOrg.y
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 1);

  std::ostringstream filename;
  filename << image_output_path_ << "/gop" << std::setw(4) << std::setfill('0')
           << gopIdx << "_obj" << std::setw(2) << std::setfill('0')
           << detection_idx // Use detection_idx
           << "_" << class_name << "_conf" << static_cast<int>(confidence * 100)
           << ".jpg";

  bool success = cv::imwrite(filename.str(), img_to_save);
  if (success) {
    std::cout << "[SAVE] ✓ 图片已保存: " << filename.str() << std::endl;
  } else {
    std::cerr << "[ERROR] ✗ 保存失败: " << filename.str() << std::endl;
  }
}
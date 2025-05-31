#include "tensor_inferencer.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace nvinfer1;

// Logger class
class Logger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[TRT] " << msg << std::endl;
    }
  }
} gLogger;

static std::vector<char> readEngineFile(const std::string &enginePath) {
  std::ifstream file(enginePath, std::ios::binary);
  if (!file.good()) {
    std::cerr << "[ERROR] Failed to open engine file: " << enginePath
              << std::endl;
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

static int roundToNearestMultiple(int val, int base = 32) {
  return ((val + base / 2) / base) * base;
}

TensorInferencer::TensorInferencer(int video_height, int video_width)
    : inputDevice_(nullptr), outputDevice_(nullptr) {
  std::cout << "[INIT] Initializing TensorInferencer for video dimensions: "
            << video_width << "x" << video_height << std::endl;

  // These might be overridden by engine dimensions later
  int initial_target_w = roundToNearestMultiple(video_width, 32);
  int initial_target_h = roundToNearestMultiple(video_height, 32);

  target_w_ = initial_target_w; // Default, may be overridden
  target_h_ = initial_target_h; // Default, may be overridden

  std::cout << "[INIT] Initial calculated target dimensions (rounded to 32 "
               "multiple): "
            << target_w_ << "x" << target_h_ << std::endl;

  const char *env_engine_path = std::getenv("YOLO_ENGINE_NAME_16");
  if (!env_engine_path) {
    std::cerr
        << "[ERROR] Environment variable YOLO_ENGINE_NAME_16 not set." // Corrected
                                                                       // variable
                                                                       // name
        << std::endl;
    std::exit(EXIT_FAILURE);
  }
  engine_path_ = env_engine_path;

  const char *names_path_env = std::getenv("YOLO_COCO_NAMES");
  if (!names_path_env) {
    std::cerr << "[ERROR] Environment variable YOLO_COCO_NAMES not set."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string names_path_str = names_path_env;

  const char *output_path_env = std::getenv("YOLO_IMAGE_PATH");
  if (!output_path_env) {
    std::cerr << "[ERROR] Environment variable YOLO_IMAGE_PATH not set."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  image_output_path_ = output_path_env;

  auto engineData = readEngineFile(engine_path_);
  if (engineData.empty()) {
    std::cerr << "[ERROR] Failed to read engine data." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  runtime_ = createInferRuntime(gLogger);
  assert(runtime_ != nullptr && "TensorRT runtime creation failed.");
  engine_ =
      runtime_->deserializeCudaEngine(engineData.data(), engineData.size());
  assert(engine_ != nullptr && "TensorRT engine deserialization failed.");
  context_ = engine_->createExecutionContext();
  assert(context_ != nullptr && "TensorRT execution context creation failed.");

  bindings_.resize(engine_->getNbBindings());

  std::cout << "[INIT] Engine loaded successfully." << std::endl;
  printEngineInfo();

  inputIndex_ = engine_->getBindingIndex("images");
  outputIndex_ = engine_->getBindingIndex("output0");

  if (outputIndex_ < 0) {
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
      if (!engine_->bindingIsInput(i)) {
        outputIndex_ = i;
        std::cout << "[INFO] Found first output tensor '"
                  << engine_->getBindingName(i) << "' at index " << i
                  << std::endl;
        break;
      }
    }
  }

  if (inputIndex_ < 0 || outputIndex_ < 0) {
    std::cerr << "[ERROR] Failed to find input 'images' or any output tensor."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "[INIT] Input index ('images'): " << inputIndex_
            << ", Output index ('" << engine_->getBindingName(outputIndex_)
            << "'): " << outputIndex_ << std::endl;

  std::ifstream infile(names_path_str);
  if (!infile.is_open()) {
    std::cerr << "[ERROR] Failed to open COCO names file: " << names_path_str
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
  std::cout << "[INIT] Loaded " << num_classes_ << " class names." << std::endl;

  Dims reportedInputDims = engine_->getBindingDimensions(inputIndex_);
  if (reportedInputDims.nbDims == 4) { // expecting NCHW
    bool useEngineDims = true;
    // Check if dimensions are dynamic ( TensorRT uses -1 for dynamic dims)
    // We are interested in H and W, which are d[2] and d[3] for NCHW
    if (reportedInputDims.d[2] <= 0 || reportedInputDims.d[3] <= 0) {
      useEngineDims = false;
    }

    if (useEngineDims) {
      target_h_ = reportedInputDims.d[2]; // Set target_H from engine
      target_w_ = reportedInputDims.d[3]; // Set target_W from engine
      std::cout << "[INIT] Using engine's opt profile dimensions for target: "
                << target_w_ << "x" << target_h_ << std::endl;
    } else {
      std::cout << "[INIT] Engine opt profile dimensions for H, W are dynamic "
                   "or invalid. Using calculated target: "
                << target_w_ << "x" << target_h_ << std::endl;
    }
  } else {
    std::cout << "[INIT] Could not get valid 4D engine opt dims for input "
                 "'images'. Using calculated target: "
              << target_w_ << "x" << target_h_ << std::endl;
  }

  std::cout << "[DEBUG_CONSTRUCTOR] Final target_w_ for preprocessing = "
            << target_w_
            << ", final target_h_ for preprocessing = " << target_h_
            << std::endl;

  if (engine_->getBindingDataType(inputIndex_) != nvinfer1::DataType::kFLOAT) {
    std::cerr << "[ERROR] Engine input tensor 'images' not DataType::kFLOAT!"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout
      << "[INFO] Engine input tensor 'images' confirmed as DataType::kFLOAT."
      << std::endl;
}

TensorInferencer::~TensorInferencer() {
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
  std::cout << "[DEINIT] TensorInferencer destroyed." << std::endl;
}

void TensorInferencer::printEngineInfo() {
  std::cout << "=== Engine Info ===" << std::endl;
  std::cout << "Engine Name: "
            << (engine_->getName() ? engine_->getName() : "N/A") << std::endl;
  std::cout << "Number of Bindings: " << engine_->getNbBindings() << std::endl;
  for (int i = 0; i < engine_->getNbBindings(); ++i) {
    const char *name = engine_->getBindingName(i);
    Dims dims = engine_->getBindingDimensions(i); // These are opt profile dims
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
    std::cout << "Binding " << i << ": '" << name << "' ("
              << (isInput ? "Input" : "Output") << ") - Type: " << dtype_str
              << " - OptProfileDims: ";
    for (int j = 0; j < dims.nbDims; ++j) {
      std::cout << dims.d[j] << (j < dims.nbDims - 1 ? "x" : "");
    }
    std::cout << std::endl;
  }
  std::cout << "===================" << std::endl;
}

bool TensorInferencer::infer(const std::vector<float> &input,
                             std::vector<float> &output) {
  std::cerr << "[ERROR] infer(const std::vector<float>&, ...) is not fully "
               "functional. Refactor or use infer(const InferenceInput&)."
            << std::endl;
  return false;
}

bool TensorInferencer::infer(const InferenceInput &input_params) {
  std::cout << "[INFER_SINGLE] Starting inference for GOP: "
            << input_params.gopIdx
            << ", Target Object: " << input_params.object_name << std::endl;

  if (input_params.decoded_frames.empty()) {
    std::cerr << "[ERROR] No input frames provided for single inference."
              << std::endl;
    return false;
  }
  const cv::Mat &raw_img = input_params.decoded_frames[0];
  if (raw_img.empty()) {
    std::cerr << "[ERROR] Input image for single inference is empty."
              << std::endl;
    return false;
  }

  std::cout << "[DEBUG_INFER_SINGLE] Using target_w_ = " << target_w_
            << ", target_h_ = " << target_h_ << " for preprocessing."
            << std::endl;

  BatchImageMetadata current_image_meta;
  current_image_meta.original_w = raw_img.cols;
  current_image_meta.original_h = raw_img.rows;
  current_image_meta.is_real_image = true;

  std::vector<float> input_data =
      preprocess_single_image_for_batch(raw_img, current_image_meta);

  if (input_data.empty()) {
    std::cerr << "[ERROR] Preprocessing failed for single image." << std::endl;
    return false;
  }

  // For single image inference, batch size is 1
  Dims inputDimsRuntime{4, {1, 3, target_h_, target_w_}};
  if (!context_->setBindingDimensions(inputIndex_, inputDimsRuntime)) {
    std::cerr << "[ERROR] Failed to set binding dimensions for single input."
              << std::endl;
    return false;
  }
  if (!context_->allInputDimensionsSpecified()) {
    std::cerr
        << "[ERROR] Not all input dimensions specified for single inference."
        << std::endl;
    return false;
  }

  Dims outDimsRuntime = context_->getBindingDimensions(outputIndex_);
  size_t current_output_elements = 1;
  for (int i = 0; i < outDimsRuntime.nbDims; ++i) {
    if (outDimsRuntime.d[i] <= 0) {
      std::cerr << "[ERROR] Invalid runtime output dimension: "
                << outDimsRuntime.d[i] << std::endl;
      return false;
    }
    current_output_elements *= outDimsRuntime.d[i];
  }
  if (current_output_elements == 0) {
    std::cerr << "[ERROR] Runtime output elements are zero." << std::endl;
    return false;
  }

  if (num_classes_ <= 0 || outDimsRuntime.d[0] != 1 || // Batch size should be 1
      (outDimsRuntime.nbDims == 3 &&
       outDimsRuntime.d[1] !=
           (4 + num_classes_)) || // For [Batch, Attributes, Detections]
      (outDimsRuntime.nbDims == 2 &&
       outDimsRuntime.d[0] !=
           (4 + num_classes_)) // For older models [Attributes, Detections] and
                               // batch is squeezed. Less common.
      // Add more checks if output format is different, e.g. [Batch, Detections,
      // Attributes]
  ) {
    std::cerr << "[ERROR] Single infer: Runtime output attributes mismatch or "
                 "unexpected dimensions. Output Dims: ";
    for (int k = 0; k < outDimsRuntime.nbDims; ++k)
      std::cerr << outDimsRuntime.d[k] << " ";
    std::cerr << "Expected attributes: " << (4 + num_classes_) << std::endl;
    return false;
  }

  if (inputDevice_)
    cudaFree(inputDevice_); // Free previous allocations
  if (outputDevice_)
    cudaFree(outputDevice_);
  inputDevice_ = nullptr;
  outputDevice_ = nullptr;

  size_t input_size_bytes = input_data.size() * sizeof(float);
  size_t output_size_bytes = current_output_elements * sizeof(float);

  cudaError_t err;
  err = cudaMalloc(&inputDevice_, input_size_bytes);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] CudaMalloc for input failed: "
              << cudaGetErrorString(err) << std::endl;
    return false;
  }
  err = cudaMalloc(&outputDevice_, output_size_bytes);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] CudaMalloc for output failed: "
              << cudaGetErrorString(err) << std::endl;
    cudaFree(inputDevice_);
    inputDevice_ = nullptr;
    return false;
  }

  bindings_[inputIndex_] = inputDevice_;
  bindings_[outputIndex_] = outputDevice_;

  err = cudaMemcpy(inputDevice_, input_data.data(), input_size_bytes,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] CudaMemcpy H2D failed: " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }

  if (!context_->enqueueV2(bindings_.data(), 0, nullptr)) {
    std::cerr << "[ERROR] TensorRT enqueueV2 failed." << std::endl;
    return false;
  }

  std::vector<float> host_output_raw(current_output_elements);
  err = cudaMemcpy(host_output_raw.data(), outputDevice_, output_size_bytes,
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] CudaMemcpy D2H failed: " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }

  // Assuming output is [1, num_attributes, num_detections] or [1,
  // num_detections, num_attributes] Let's use the common [Batch, Attributes,
  // Detections_per_image] for YOLO typically from TRT or [Batch,
  // Detections_per_image, Attributes]
  int num_detections_per_image = 0;
  int num_attributes_per_detection = 0;

  // We need to know the output format. Let's assume output is [Batch, NumBoxes,
  // NumAttributes] or [Batch, NumAttributes, NumBoxes] From your test.cpp:
  // output_dims_actual.d[0] = BATCH_SIZE, d[1] = (4 + num_classes), d[2] =
  // EXPECTED_NUM_DETECTIONS_YOLO_PER_IMAGE This suggests output is [BATCH,
  // ATTRIBUTES, DETECTIONS]
  if (outDimsRuntime.nbDims == 3 &&
      outDimsRuntime.d[0] == 1) { // [1, attributes, detections]
    num_attributes_per_detection = outDimsRuntime.d[1];
    num_detections_per_image = outDimsRuntime.d[2];
    if (num_attributes_per_detection != (4 + num_classes_)) {
      std::cerr << "[ERROR] Output dimension 1 (attributes) "
                << num_attributes_per_detection << " does not match expected "
                << (4 + num_classes_) << std::endl;
      return false;
    }
  } else {
    std::cerr << "[ERROR] Unexpected output dimensions for single inference "
                 "processing. Got "
              << outDimsRuntime.nbDims << "D." << std::endl;
    return false;
  }

  process_single_output(input_params, host_output_raw.data(),
                        num_detections_per_image, num_attributes_per_detection,
                        raw_img, current_image_meta, 0);
  return true;
}

// New method to preprocess a single image for batching using LETTERBOXING
std::vector<float>
TensorInferencer::preprocess_single_image_for_batch(const cv::Mat &img,
                                                    BatchImageMetadata &meta) {
  // Target dimensions for model input (e.g., 736x736)
  const int model_input_w = target_w_;
  const int model_input_h = target_h_;

  cv::Mat image_to_process;

  if (!meta.is_real_image) {
    // For dummy/padding slots, create a gray image (consistent with common
    // letterbox padding) This ensures the preprocessor handles a correctly
    // sized Mat.
    image_to_process = cv::Mat(model_input_h, model_input_w, CV_8UC3,
                               cv::Scalar(114, 114, 114));
    // Update meta for dummy image (original size is not relevant, but set to
    // model size for consistency in downscaling logic)
    meta.original_w = model_input_w;
    meta.original_h = model_input_h;
    meta.scale_to_model = 1.0f; // No scaling needed as it's already target size
    meta.pad_w_left = 0;
    meta.pad_h_top = 0;
  } else {
    meta.original_w = img.cols;
    meta.original_h = img.rows;
    image_to_process = img.clone(); // Work on a copy for real images
  }

  cv::Mat processed_for_model(model_input_h, model_input_w, CV_8UC3,
                              cv::Scalar(114, 114, 114)); // Letterbox bg

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

    // Place the resized image onto the gray letterbox background
    resized_img.copyTo(processed_for_model(
        cv::Rect(meta.pad_w_left, meta.pad_h_top, scaled_w, scaled_h)));
  } else if (!meta.is_real_image) {
    // image_to_process is already the gray dummy image of model_input_h x
    // model_input_w
    processed_for_model = image_to_process;
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

bool TensorInferencer::infer_batch(
    const std::vector<InferenceInput> &batch_inputs) {
  if (batch_inputs.empty()) {
    std::cout << "[INFER_BATCH] Input batch is empty." << std::endl;
    return false;
  }

  const int CURRENT_BATCH_SIZE = static_cast<int>(batch_inputs.size());
  // TARGET_GPU_BATCH_SIZE is the size the engine is configured for (e.g., via
  // setBindingDimensions or implicit in engine build) This should match the
  // batch dimension set in setBindingDimensions later. Let's assume the engine
  // is flexible or we always pad to a fixed batch size it expects. For
  // YOLO_ENGINE_NAME_16, test.cpp uses BATCH_SIZE = 16.
  const int TARGET_GPU_BATCH_SIZE = 16;

  std::cout << "[INFER_BATCH] Starting inference for a batch of "
            << CURRENT_BATCH_SIZE << " inputs (will pad to "
            << TARGET_GPU_BATCH_SIZE << " for engine if needed)." << std::endl;

  std::vector<float> batched_input_data;
  // Preallocate for the full GPU batch size and model dimensions
  batched_input_data.reserve(static_cast<size_t>(TARGET_GPU_BATCH_SIZE) * 3 *
                             target_h_ * target_w_);
  std::vector<BatchImageMetadata> batch_image_meta(TARGET_GPU_BATCH_SIZE);
  std::vector<cv::Mat> original_raw_images_for_saving(TARGET_GPU_BATCH_SIZE);

  for (int i = 0; i < TARGET_GPU_BATCH_SIZE; ++i) {
    cv::Mat current_raw_img_for_preprocessing;
    if (i < CURRENT_BATCH_SIZE) {
      const InferenceInput &current_input_param = batch_inputs[i];
      if (current_input_param.decoded_frames.empty() ||
          current_input_param.decoded_frames[0].empty()) {
        std::cerr << "[WARN] Empty frame in batch_inputs at index " << i
                  << ". Using dummy for preprocessing." << std::endl;
        batch_image_meta[i].is_real_image = false;
        // Let preprocess_single_image_for_batch handle creating the dummy based
        // on is_real_image but we need a placeholder Mat for the function call.
        current_raw_img_for_preprocessing = cv::Mat(
            target_h_, target_w_, CV_8UC3,
            cv::Scalar(0, 0,
                       0)); // Temporary, will be replaced by gray in preprocess
        original_raw_images_for_saving[i] =
            current_raw_img_for_preprocessing
                .clone(); // Save the black dummy for consistency if needed
                          // later, though saveAnnotatedImage should skip
                          // non-reals
      } else {
        batch_image_meta[i].is_real_image = true;
        current_raw_img_for_preprocessing =
            current_input_param.decoded_frames[0];
        original_raw_images_for_saving[i] =
            current_input_param.decoded_frames[0]
                .clone(); // Save the real image
      }
    } else { // Padding slot for GPU batch
      batch_image_meta[i].is_real_image = false;
      current_raw_img_for_preprocessing =
          cv::Mat(target_h_, target_w_, CV_8UC3,
                  cv::Scalar(0, 0, 0)); // Temp, will be gray
      original_raw_images_for_saving[i] =
          current_raw_img_for_preprocessing.clone();
    }

    std::vector<float> single_image_data = preprocess_single_image_for_batch(
        current_raw_img_for_preprocessing, batch_image_meta[i]);
    batched_input_data.insert(batched_input_data.end(),
                              single_image_data.begin(),
                              single_image_data.end());
  }

  Dims inputDimsRuntime{4, {TARGET_GPU_BATCH_SIZE, 3, target_h_, target_w_}};
  if (!context_->setBindingDimensions(inputIndex_, inputDimsRuntime)) {
    std::cerr << "[ERROR] Failed to set binding dimensions for batched input."
              << std::endl;
    return false;
  }
  if (!context_->allInputDimensionsSpecified()) {
    std::cerr
        << "[ERROR] Not all input dimensions specified for batched inference."
        << std::endl;
    return false;
  }

  Dims outDimsRuntime = context_->getBindingDimensions(outputIndex_);
  size_t total_output_elements = 1;
  for (int i = 0; i < outDimsRuntime.nbDims; ++i) {
    if (outDimsRuntime.d[i] <= 0) {
      std::cerr << "[ERROR] Invalid runtime output dimension for batch: "
                << outDimsRuntime.d[i] << std::endl;
      return false;
    }
    total_output_elements *= outDimsRuntime.d[i];
  }
  if (total_output_elements == 0) {
    std::cerr << "[ERROR] Runtime output elements for batch are zero."
              << std::endl;
    return false;
  }

  // Assuming output format [BATCH, ATTRIBUTES, DETECTIONS] as in test.cpp
  // Or [BATCH, DETECTIONS, ATTRIBUTES]
  // Let's use the format from test.cpp: [BATCH, NumAttributes,
  // NumDetectionsPerImage]
  if (num_classes_ <= 0 || outDimsRuntime.d[0] != TARGET_GPU_BATCH_SIZE ||
      (outDimsRuntime.nbDims == 3 &&
       outDimsRuntime.d[1] != (4 + num_classes_))) {
    std::cerr << "[ERROR] Batched infer: Runtime output attributes/batch "
                 "mismatch. Output Dims: ";
    for (int k = 0; k < outDimsRuntime.nbDims; ++k)
      std::cerr << outDimsRuntime.d[k] << " ";
    std::cerr << "Expected Batch: " << TARGET_GPU_BATCH_SIZE
              << ", Expected Attributes: " << (4 + num_classes_) << std::endl;
    return false;
  }
  int num_detections_per_image_from_engine = 0;
  int num_attributes_from_engine = 0;

  if (outDimsRuntime.nbDims == 3) { // Assuming [Batch, Attributes, Detections]
    num_attributes_from_engine = outDimsRuntime.d[1];
    num_detections_per_image_from_engine = outDimsRuntime.d[2];
    if (num_attributes_from_engine != (4 + num_classes_)) {
      std::cerr << "[ERROR] Batched output dimension 1 (attributes) "
                << num_attributes_from_engine << " does not match expected "
                << (4 + num_classes_) << std::endl;
      return false;
    }
  } else {
    std::cerr
        << "[ERROR] Unexpected output dimensions for batch inference. Got "
        << outDimsRuntime.nbDims << "D." << std::endl;
    return false;
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
    std::cerr << "[ERROR] CudaMalloc for batched input failed: "
              << cudaGetErrorString(err) << std::endl;
    return false;
  }
  err = cudaMalloc(&outputDevice_, total_output_bytes);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] CudaMalloc for batched output failed: "
              << cudaGetErrorString(err) << std::endl;
    cudaFree(inputDevice_);
    inputDevice_ = nullptr;
    return false;
  }

  bindings_[inputIndex_] = inputDevice_;
  bindings_[outputIndex_] = outputDevice_;

  err = cudaMemcpy(inputDevice_, batched_input_data.data(), total_input_bytes,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] Batched CudaMemcpy H2D failed: "
              << cudaGetErrorString(err) << std::endl;
    return false;
  }

  if (!context_->enqueueV2(bindings_.data(), 0, nullptr)) {
    std::cerr << "[ERROR] Batched TensorRT enqueueV2 failed." << std::endl;
    return false;
  }

  std::vector<float> host_output_batched_raw(total_output_elements);
  err = cudaMemcpy(host_output_batched_raw.data(), outputDevice_,
                   total_output_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] Batched CudaMemcpy D2H failed: "
              << cudaGetErrorString(err) << std::endl;
    return false;
  }

  std::cout << "[INFER_BATCH] Inference complete for batch of "
            << CURRENT_BATCH_SIZE << " (padded to " << TARGET_GPU_BATCH_SIZE
            << ")." << std::endl;

  for (int i = 0; i < CURRENT_BATCH_SIZE; ++i) {
    if (!batch_image_meta[i].is_real_image) {
      // This should ideally not happen if CURRENT_BATCH_SIZE reflects only real
      // inputs. But if batch_inputs contained dummys marked as real, this would
      // skip them.
      std::cout << "[INFER_BATCH] Skipping postprocessing for slot " << i
                << " as it's marked not real (originally dummy)." << std::endl;
      continue;
    }
    const InferenceInput &current_input_param = batch_inputs[i];
    const cv::Mat &current_raw_img_for_saving =
        original_raw_images_for_saving[i];

    if (current_raw_img_for_saving.empty() &&
        batch_image_meta[i].is_real_image) {
      std::cerr << "[WARN] Real image at batch index " << i
                << " is empty before postprocessing. Skipping." << std::endl;
      continue;
    }

    const float *output_for_this_image_start =
        host_output_batched_raw.data() +
        static_cast<size_t>(i) * num_attributes_from_engine *
            num_detections_per_image_from_engine;

    process_single_output(current_input_param, output_for_this_image_start,
                          num_detections_per_image_from_engine,
                          num_attributes_from_engine,
                          current_raw_img_for_saving, batch_image_meta[i], i);
  }
  return true;
}

void TensorInferencer::process_single_output(
    const InferenceInput &input_params, const float *host_output_for_image_raw,
    int num_detections_in_slice, // Num detections for this one image from the
                                 // model's perspective
    int num_attributes_per_detection,  // Num attributes per detection
    const cv::Mat &raw_img_for_saving, // Original raw image for this detection
    const BatchImageMetadata &
        image_meta, // Metadata for this specific image (incl. letterbox params)
    int original_batch_idx_for_debug) {

  // Output is [attributes, detections]. Transpose to [detections, attributes]
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

  auto it = class_name_to_id_.find(input_params.object_name);
  if (it == class_name_to_id_.end()) {
    std::cerr << "[ERROR][ProcessOutput] Target object name '"
              << input_params.object_name << "' not found in class names."
              << std::endl;
    return;
  }
  int target_class_id = it->second;

  float confidence_threshold = 0.01f;
  // float confidence_threshold = std::max(0.1f,
  // input_params.confidence_thresh); // Use this for production
  std::cout << "[DEBUG_SINGLE_OUTPUT] Img " << original_batch_idx_for_debug
            << " (GOP: " << input_params.gopIdx
            << "): Using conf_thresh: " << confidence_threshold
            << " for target '" << input_params.object_name << "'" << std::endl;

  std::vector<Detection> detected_objects;
  for (int i = 0; i < num_detections_in_slice; ++i) {
    const float *det_attrs = &transposed_output[static_cast<size_t>(i) *
                                                num_attributes_per_detection];
    // det_attrs: [cx, cy, w, h, score_cls1, score_cls2, ..., score_clsN]

    float max_score = 0.0f;
    int best_class_id = -1;
    // Scores start at index 4
    for (int j = 0; j < num_classes_; ++j) {
      float score = det_attrs[4 + j];
      if (score > max_score) {
        max_score = score;
        best_class_id = j;
      }
    }

    if (max_score > 0.005) { // Very low threshold for raw debug printing
      std::cout << "[RAW_DET_SINGLE] Img " << original_batch_idx_for_debug
                << ", Det " << i << ": BestClsID=" << best_class_id
                << " (Name: "
                << (id_to_class_name_.count(best_class_id)
                        ? id_to_class_name_.at(best_class_id)
                        : "Unknown")
                << ")"
                << ", MaxScore=" << std::fixed << std::setprecision(4)
                << max_score << ", CX=" << det_attrs[0]
                << ", CY=" << det_attrs[1] << ", W=" << det_attrs[2]
                << ", H=" << det_attrs[3] << std::endl;
    }

    if (best_class_id == target_class_id && max_score >= confidence_threshold) {
      // Bounding box coordinates are relative to the model input size
      // (target_w_, target_h_)
      float cx = det_attrs[0];
      float cy = det_attrs[1];
      float w = det_attrs[2];
      float h = det_attrs[3];

      // These are coordinates in the letterboxed/resized model input space
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
  std::cout << "[NMS_SINGLE] Img " << original_batch_idx_for_debug << " For '"
            << input_params.object_name
            << "': Before NMS=" << detected_objects.size()
            << ", After NMS=" << nms_detections.size() << std::endl;

  for (size_t i = 0; i < nms_detections.size(); ++i) {
    const auto &det = nms_detections[i];
    if (!image_meta.is_real_image || raw_img_for_saving.empty()) {
      std::cout << "[WARN][SAVE] Skipping save for detection on a "
                   "non-real/empty image_meta or empty raw_img. Img Idx: "
                << original_batch_idx_for_debug << std::endl;
      continue;
    }
    saveAnnotatedImage(
        raw_img_for_saving, det,
        image_meta, // Pass image_meta for coordinate transformation
        input_params.object_name, input_params.gopIdx, static_cast<int>(i));
  }

  if (nms_detections.empty() &&
      image_meta.is_real_image) { // Only log this for real images
    std::cout << "[INFO_SINGLE] Img " << original_batch_idx_for_debug
              << ": No '" << input_params.object_name
              << "' (GOP: " << input_params.gopIdx << ") meeting criteria."
              << std::endl;
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

// Modified saveAnnotatedImage to use BatchImageMetadata for letterbox
// coordinate transformation
void TensorInferencer::saveAnnotatedImage(
    const cv::Mat
        &raw_img_for_saving, // This is the ORIGINAL, UNPROCESSED image
    const Detection &det, // Detection coordinates are relative to model input
                          // (e.g., 736x736 letterboxed)
    const BatchImageMetadata
        &image_meta, // Contains letterboxing params (scale_to_model,
                     // pad_w_left, pad_h_top)
    const std::string &class_name_str, int gopIdx,
    int detection_idx_in_image) { // Renamed for clarity

  if (!image_meta.is_real_image || raw_img_for_saving.empty()) {
    std::cerr << "[WARN][SAVE] Attempted to save annotation for a non-real or "
                 "empty image. GOP: "
              << gopIdx << ", Detection status: " << det.status_info
              << ". Skipping." << std::endl;
    return;
  }

  cv::Mat img_to_save = raw_img_for_saving.clone();

  // det.x1, det.y1, det.x2, det.y2 are relative to the letterboxed model input
  // (target_w_ x target_h_) We need to map them back to the original image
  // coordinates.

  // 1. Remove padding offset (coordinates are now relative to the scaled image
  // within the letterbox)
  float x1_unpadded = det.x1 - image_meta.pad_w_left;
  float y1_unpadded = det.y1 - image_meta.pad_h_top;
  float x2_unpadded = det.x2 - image_meta.pad_w_left;
  float y2_unpadded = det.y2 - image_meta.pad_h_top;

  // 2. Scale back to original image dimensions using the inverse of
  // scale_to_model Ensure scale_to_model is not zero to prevent division by
  // zero
  if (image_meta.scale_to_model <= 1e-6f) {
    std::cerr << "[WARN][SAVE] Invalid scale_to_model ("
              << image_meta.scale_to_model << ") for GOP " << gopIdx
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

  // 3. Clip to original image boundaries
  x1_orig = std::max(0, std::min(x1_orig, image_meta.original_w - 1));
  y1_orig = std::max(0, std::min(y1_orig, image_meta.original_h - 1));
  x2_orig = std::max(0, std::min(x2_orig, image_meta.original_w - 1));
  y2_orig = std::max(0, std::min(y2_orig, image_meta.original_h - 1));

  if (x2_orig <= x1_orig || y2_orig <= y1_orig) {
    std::cout
        << "[WARN][SAVE] Scaled box invalid after letterbox reversal for GOP "
        << gopIdx << ". Original box (model scale): [" << det.x1 << ","
        << det.y1 << "," << det.x2 << "," << det.y2 << "]"
        << ". Scaled box (orig img): [" << x1_orig << "," << y1_orig << ","
        << x2_orig << "," << y2_orig << "]"
        << ". Meta: scale=" << image_meta.scale_to_model
        << " padL=" << image_meta.pad_w_left << " padT=" << image_meta.pad_h_top
        << ". Skipping save." << std::endl;
    return;
  }

  cv::rectangle(img_to_save, cv::Point(x1_orig, y1_orig),
                cv::Point(x2_orig, y2_orig), cv::Scalar(0, 255, 0), 2);

  std::ostringstream label;
  label << class_name_str << " " << std::fixed << std::setprecision(2)
        << det.confidence;
  int baseline = 0;
  cv::Size textSize =
      cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseline);
  baseline += 1;

  cv::Point textOrg(x1_orig, y1_orig - 5);
  if (textOrg.y - textSize.height < 0) { // If text goes above image
    textOrg.y =
        y1_orig + textSize.height + 5; // Move below top-left corner of box
    if (textOrg.y >
        image_meta.original_h - baseline) { // If it goes below image
      textOrg.y = image_meta.original_h - baseline - 2; // Adjust
    }
  }
  if (textOrg.x + textSize.width >
      image_meta.original_w) { // If text goes beyond right edge
    textOrg.x = image_meta.original_w - textSize.width - 2;
  }
  textOrg.x = std::max(0, textOrg.x); // Ensure text starts within image bounds

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
               << std::setfill('0') << detection_idx_in_image
               << "_" // Use per-image detection index
               << class_name_str << "_conf"
               << static_cast<int>(det.confidence * 100) << ".jpg";

  bool success = cv::imwrite(filename_oss.str(), img_to_save);
  if (success) {
    std::cout << "[SAVE] Annotated image saved: " << filename_oss.str()
              << std::endl;
  } else {
    std::cerr << "[ERROR] Failed to save image: " << filename_oss.str()
              << std::endl;
  }
}
#include "tensor_inferencer.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace nvinfer1;

// Logger class
class Logger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[TRT] " << msg << std::endl;
    }
  }
} gLogger;

static std::vector<char> readEngineFile(const std::string &enginePath) {
  std::ifstream file(enginePath, std::ios::binary);
  if (!file.good()) {
    std::cerr << "[ERROR] Failed to open engine file: " << enginePath
              << std::endl;
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

static int roundToNearestMultiple(int val, int base = 32) {
  return ((val + base / 2) / base) * base;
}

TensorInferencer::TensorInferencer(int video_height, int video_width)
    : inputDevice_(nullptr), outputDevice_(nullptr) {
  std::cout << "[INIT] Initializing TensorInferencer for video dimensions: "
            << video_width << "x" << video_height << std::endl;

  target_w_ = roundToNearestMultiple(video_width, 32);
  target_h_ = roundToNearestMultiple(video_height, 32);
  std::cout << "[INIT] Initial target dimensions (rounded to 32 multiple): "
            << target_w_ << "x" << target_h_ << std::endl;

  const char *env_engine_path = std::getenv("YOLO_ENGINE_NAME_16");
  if (!env_engine_path) {
    std::cerr << "[ERROR] Environment variable YOLO_ENGINE_NAME not set."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  engine_path_ = env_engine_path;

  const char *names_path_env = std::getenv("YOLO_COCO_NAMES");
  if (!names_path_env) {
    std::cerr << "[ERROR] Environment variable YOLO_COCO_NAMES not set."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string names_path_str = names_path_env;

  const char *output_path_env = std::getenv("YOLO_IMAGE_PATH");
  if (!output_path_env) {
    std::cerr << "[ERROR] Environment variable YOLO_IMAGE_PATH not set."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  image_output_path_ = output_path_env;

  auto engineData = readEngineFile(engine_path_);
  if (engineData.empty()) {
    std::cerr << "[ERROR] Failed to read engine data." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  runtime_ = createInferRuntime(gLogger);
  assert(runtime_ != nullptr && "TensorRT runtime creation failed.");
  engine_ =
      runtime_->deserializeCudaEngine(engineData.data(), engineData.size());
  assert(engine_ != nullptr && "TensorRT engine deserialization failed.");
  context_ = engine_->createExecutionContext();
  assert(context_ != nullptr && "TensorRT execution context creation failed.");

  bindings_.resize(engine_->getNbBindings());

  std::cout << "[INIT] Engine loaded successfully." << std::endl;
  printEngineInfo();

  inputIndex_ = engine_->getBindingIndex("images");
  outputIndex_ = engine_->getBindingIndex("output0");

  if (outputIndex_ < 0) {
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
      if (!engine_->bindingIsInput(i)) {
        outputIndex_ = i;
        std::cout << "[INFO] Found first output tensor '"
                  << engine_->getBindingName(i) << "' at index " << i
                  << std::endl;
        break;
      }
    }
  }

  if (inputIndex_ < 0 || outputIndex_ < 0) { /* Handle error */
    std::exit(EXIT_FAILURE);
  }
  std::cout << "[INIT] Input index ('images'): " << inputIndex_
            << ", Output index ('" << engine_->getBindingName(outputIndex_)
            << "'): " << outputIndex_ << std::endl;

  std::ifstream infile(names_path_str);
  if (!infile.is_open()) { /* Handle error */
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
  std::cout << "[INIT] Loaded " << num_classes_ << " class names." << std::endl;

  Dims reportedInputDims = engine_->getBindingDimensions(inputIndex_);
  if (reportedInputDims.nbDims == 4) {
    bool useEngineDims = true;
    for (int i = 0; i < reportedInputDims.nbDims; ++i) {
      if (reportedInputDims.d[i] <= 0) {
        useEngineDims = false;
        break;
      }
    }
    if (useEngineDims) {
      target_h_ = reportedInputDims.d[2];
      target_w_ = reportedInputDims.d[3];
      std::cout << "[INIT] Using engine's opt profile dimensions for target: "
                << target_w_ << "x" << target_h_ << std::endl;
    } else {
      std::cout
          << "[INIT] Engine opt profile dynamic. Using calculated target: "
          << target_w_ << "x" << target_h_ << std::endl;
    }
  } else {
    std::cout
        << "[INIT] Could not get engine opt dims. Using calculated target: "
        << target_w_ << "x" << target_h_ << std::endl;
  }

  std::cout << "[DEBUG_CONSTRUCTOR] Final target_w_ = " << target_w_
            << ", target_h_ = " << target_h_ << std::endl;

  if (engine_->getBindingDataType(inputIndex_) != nvinfer1::DataType::kFLOAT) {
    std::cerr << "[ERROR] Engine input tensor not DataType::kFLOAT!"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "[INFO] Engine input tensor confirmed as DataType::kFLOAT."
            << std::endl;
}

TensorInferencer::~TensorInferencer() {
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
  std::cout << "[DEINIT] TensorInferencer destroyed." << std::endl;
}

void TensorInferencer::printEngineInfo() {
  // ... (implementation from your last version is fine) ...
  std::cout << "=== Engine Info ===" << std::endl;
  std::cout << "Engine Name: "
            << (engine_->getName() ? engine_->getName() : "N/A") << std::endl;
  std::cout << "Number of Bindings: " << engine_->getNbBindings() << std::endl;
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
      dtype_str = "Unknown";
      break;
    }
    std::cout << "Binding " << i << ": '" << name << "' ("
              << (isInput ? "Input" : "Output") << ") - Type: " << dtype_str
              << " - OptProfileDims: ";
    for (int j = 0; j < dims.nbDims; ++j) {
      std::cout << dims.d[j] << (j < dims.nbDims - 1 ? "x" : "");
    }
    std::cout << std::endl;
  }
  std::cout << "===================" << std::endl;
}

// This overload is largely non-functional due to removal of class-wide
// inputSize_/outputSize_ It needs proper size determination if it's to be used.
bool TensorInferencer::infer(const std::vector<float> &input,
                             std::vector<float> &output) {
  std::cerr << "[ERROR] infer(const std::vector<float>&, ...) is not fully "
               "functional. Refactor or use infer(const InferenceInput&)."
            << std::endl;
  return false;
}

// Single image inference (as per your original class structure's main infer
// method)
bool TensorInferencer::infer(const InferenceInput &input_params) {
  std::cout << "[INFER_SINGLE] Starting inference for GOP: "
            << input_params.gopIdx
            << ", Target Object: " << input_params.object_name << std::endl;

  if (input_params.decoded_frames.empty()) {
    std::cerr << "[ERROR] No input frames provided for single inference."
              << std::endl;
    return false;
  }
  const cv::Mat &raw_img =
      input_params.decoded_frames[0]; // Processes only the first frame
  if (raw_img.empty()) {
    std::cerr << "[ERROR] Input image for single inference is empty."
              << std::endl;
    return false;
  }

  std::cout << "[DEBUG_INFER_SINGLE] Using target_w_ = " << target_w_
            << ", target_h_ = " << target_h_ << std::endl;

  BatchImageMetadata
      current_image_meta; // Create metadata for this single image
  current_image_meta.original_w = raw_img.cols;
  current_image_meta.original_h = raw_img.rows;
  current_image_meta.is_real_image = true; // It's always a real image here

  std::vector<float> input_data =
      preprocess_single_image_for_batch(raw_img, current_image_meta);

  if (input_data.empty()) {
    std::cerr << "[ERROR] Preprocessing failed for single image." << std::endl;
    return false;
  }

  Dims inputDimsRuntime{4, {1, 3, target_h_, target_w_}}; // Batch size is 1
  if (!context_->setBindingDimensions(inputIndex_, inputDimsRuntime)) {
    std::cerr << "[ERROR] Failed to set binding dimensions for single input."
              << std::endl;
    return false;
  }
  if (!context_->allInputDimensionsSpecified()) {
    std::cerr
        << "[ERROR] Not all input dimensions specified for single inference."
        << std::endl;
    return false;
  }

  Dims outDimsRuntime = context_->getBindingDimensions(outputIndex_);
  size_t current_output_elements = 1;
  for (int i = 0; i < outDimsRuntime.nbDims; ++i) {
    if (outDimsRuntime.d[i] <= 0) { /* error */
      return false;
    }
    current_output_elements *= outDimsRuntime.d[i];
  }

  if (num_classes_ <= 0 || outDimsRuntime.d[0] != 1 ||
      outDimsRuntime.d[1] != (4 + num_classes_)) {
    std::cerr << "[ERROR] Single infer: Runtime output attributes mismatch."
              << std::endl;
    return false;
  }

  // GPU Memory Allocation (per-inference, as in original structure)
  if (inputDevice_)
    cudaFree(inputDevice_);
  if (outputDevice_)
    cudaFree(outputDevice_);

  size_t input_size_bytes = input_data.size() * sizeof(float);
  size_t output_size_bytes = current_output_elements * sizeof(float);

  cudaError_t err;
  err = cudaMalloc(&inputDevice_, input_size_bytes);
  if (err != cudaSuccess) { /* error */
    return false;
  }
  err = cudaMalloc(&outputDevice_, output_size_bytes);
  if (err != cudaSuccess) {
    cudaFree(inputDevice_);
    inputDevice_ = nullptr;
    return false;
  }

  bindings_[inputIndex_] = inputDevice_;
  bindings_[outputIndex_] = outputDevice_;

  err = cudaMemcpy(inputDevice_, input_data.data(), input_size_bytes,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) { /* error */
    return false;
  }

  if (!context_->enqueueV2(bindings_.data(), 0, nullptr)) { /* error */
    return false;
  }

  std::vector<float> host_output_raw(current_output_elements);
  err = cudaMemcpy(host_output_raw.data(), outputDevice_, output_size_bytes,
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) { /* error */
    return false;
  }

  // Create a dummy vector for batch_inputs and batch_metadata for
  // process_single_output
  process_single_output(input_params, host_output_raw.data(),
                        outDimsRuntime.d[2], outDimsRuntime.d[1], raw_img,
                        current_image_meta, 0);
  return true;
}

// New method to preprocess a single image for batching
std::vector<float>
TensorInferencer::preprocess_single_image_for_batch(const cv::Mat &img,
                                                    BatchImageMetadata &meta) {

  cv::Mat image_to_process = img;
  if (!meta.is_real_image) { // If it's a dummy slot, create a black image
    image_to_process =
        cv::Mat(target_h_, target_w_, CV_8UC3, cv::Scalar(0, 0, 0));
    meta.original_w = target_w_; // For dummy, set original to target
    meta.original_h = target_h_;
  } else {
    meta.original_w = img.cols;
    meta.original_h = img.rows;
  }

  // --- Image Preprocessing for this image ---
  cv::Mat resized_img;
  // Using direct resize as per original logic constraint for TensorInferencer
  // class
  cv::resize(image_to_process, resized_img, cv::Size(target_w_, target_h_));

  // Store scaling factors for direct resize (used in saveAnnotatedImage)
  // For direct resize, pad_w_left and pad_h_top are 0. Scale is different.
  meta.scale_to_model =
      1.0f; // Not used in direct resize like letterbox, set to 1
  meta.pad_w_left = 0;
  meta.pad_h_top = 0;

  cv::Mat img_rgb;
  cv::cvtColor(resized_img, img_rgb, cv::COLOR_BGR2RGB);

  int c = 3;
  cv::Mat chw_input_fp32;
  img_rgb.convertTo(chw_input_fp32, CV_32FC3, 1.0 / 255.0);

  std::vector<float> input_data_single(static_cast<size_t>(c) * target_h_ *
                                       target_w_);
  for (int ch_idx = 0; ch_idx < c; ++ch_idx) {
    for (int y = 0; y < target_h_; ++y) {
      for (int x = 0; x < target_w_; ++x) {
        input_data_single[static_cast<size_t>(ch_idx) * target_h_ * target_w_ +
                          static_cast<size_t>(y) * target_w_ + x] =
            chw_input_fp32.at<cv::Vec3f>(y, x)[ch_idx];
      }
    }
  }
  return input_data_single;
}

bool TensorInferencer::infer_batch(
    const std::vector<InferenceInput> &batch_inputs) {
  if (batch_inputs.empty()) {
    std::cout << "[INFER_BATCH] Input batch is empty." << std::endl;
    return false;
  }

  const int CURRENT_BATCH_SIZE = static_cast<int>(batch_inputs.size());
  const int TARGET_GPU_BATCH_SIZE =
      16; // The batch size engine is set for / expects

  std::cout << "[INFER_BATCH] Starting inference for a batch of "
            << CURRENT_BATCH_SIZE << " inputs (will pad to "
            << TARGET_GPU_BATCH_SIZE << " for engine)." << std::endl;

  std::vector<float> batched_input_data;
  batched_input_data.reserve(static_cast<size_t>(TARGET_GPU_BATCH_SIZE) * 3 *
                             target_h_ * target_w_);
  std::vector<BatchImageMetadata> batch_image_meta(TARGET_GPU_BATCH_SIZE);
  std::vector<cv::Mat> original_raw_images_for_saving(TARGET_GPU_BATCH_SIZE);

  for (int i = 0; i < TARGET_GPU_BATCH_SIZE; ++i) {
    if (i < CURRENT_BATCH_SIZE) { // Real image from batch_inputs
      const InferenceInput &current_input_param = batch_inputs[i];
      if (current_input_param.decoded_frames.empty() ||
          current_input_param.decoded_frames[0].empty()) {
        std::cerr << "[WARN] Empty frame in batch_inputs at index " << i
                  << ". Using dummy." << std::endl;
        batch_image_meta[i].is_real_image = false;
        original_raw_images_for_saving[i] = cv::Mat(
            target_h_, target_w_, CV_8UC3, cv::Scalar(0, 0, 0)); // Dummy raw
      } else {
        batch_image_meta[i].is_real_image = true;
        original_raw_images_for_saving[i] =
            current_input_param.decoded_frames[0];
      }
    } else { // Padding slot
      batch_image_meta[i].is_real_image = false;
      original_raw_images_for_saving[i] = cv::Mat(
          target_h_, target_w_, CV_8UC3, cv::Scalar(0, 0, 0)); // Dummy raw
    }
    // Preprocess (real or dummy image based on
    // original_raw_images_for_saving[i])
    std::vector<float> single_image_data = preprocess_single_image_for_batch(
        original_raw_images_for_saving[i], batch_image_meta[i]);
    batched_input_data.insert(batched_input_data.end(),
                              single_image_data.begin(),
                              single_image_data.end());
  }
  // std::cout << "[INFER_BATCH] Batch preprocessing complete." << std::endl;

  // --- Set Runtime Input Dimensions for the full TARGET_GPU_BATCH_SIZE ---
  Dims inputDimsRuntime{4, {TARGET_GPU_BATCH_SIZE, 3, target_h_, target_w_}};
  if (!context_->setBindingDimensions(inputIndex_, inputDimsRuntime)) {
    std::cerr << "[ERROR] Failed to set binding dimensions for batched input."
              << std::endl;
    return false;
  }
  if (!context_->allInputDimensionsSpecified()) {
    std::cerr
        << "[ERROR] Not all input dimensions specified for batched inference."
        << std::endl;
    return false;
  }

  Dims outDimsRuntime = context_->getBindingDimensions(outputIndex_);
  size_t total_output_elements = 1;
  // std::cout << "[INFER_BATCH] Runtime output dimensions for tensor '" <<
  // engine_->getBindingName(outputIndex_) << "': ";
  for (int i = 0; i < outDimsRuntime.nbDims; ++i) {
    // std::cout << outDimsRuntime.d[i] << (i == outDimsRuntime.nbDims - 1 ? ""
    // : "x");
    if (outDimsRuntime.d[i] <= 0) { /* error handling */
      return false;
    }
    total_output_elements *= outDimsRuntime.d[i];
  }
  // std::cout << std::endl;

  if (num_classes_ <= 0 || outDimsRuntime.d[0] != TARGET_GPU_BATCH_SIZE ||
      outDimsRuntime.d[1] != (4 + num_classes_)) {
    std::cerr
        << "[ERROR] Batched infer: Runtime output attributes/batch mismatch."
        << std::endl;
    return false;
  }
  int num_detections_per_image_from_engine = outDimsRuntime.d[2];
  int num_attributes_from_engine = outDimsRuntime.d[1];

  // --- GPU Memory Allocation ---
  if (inputDevice_)
    cudaFree(inputDevice_);
  if (outputDevice_)
    cudaFree(outputDevice_);

  size_t total_input_bytes = batched_input_data.size() * sizeof(float);
  size_t total_output_bytes =
      total_output_elements * sizeof(float); // Assuming FP32 output

  cudaError_t err;
  err = cudaMalloc(&inputDevice_, total_input_bytes);
  if (err != cudaSuccess) { /* error */
    return false;
  }
  err = cudaMalloc(&outputDevice_, total_output_bytes);
  if (err != cudaSuccess) {
    cudaFree(inputDevice_);
    inputDevice_ = nullptr;
    return false;
  }

  bindings_[inputIndex_] = inputDevice_;
  bindings_[outputIndex_] = outputDevice_;

  // --- Perform Inference ---
  err = cudaMemcpy(inputDevice_, batched_input_data.data(), total_input_bytes,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) { /* error */
    return false;
  }

  if (!context_->enqueueV2(bindings_.data(), 0, nullptr)) { /* error */
    return false;
  }

  std::vector<float> host_output_batched_raw(total_output_elements);
  err = cudaMemcpy(host_output_batched_raw.data(), outputDevice_,
                   total_output_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) { /* error */
    return false;
  }

  std::cout << "[INFER_BATCH] Inference complete for batch of "
            << CURRENT_BATCH_SIZE << " (padded to " << TARGET_GPU_BATCH_SIZE
            << ")." << std::endl;

  // --- Postprocess each real image in the batch ---
  for (int i = 0; i < CURRENT_BATCH_SIZE;
       ++i) { // Only process up to actual images provided
    if (!batch_image_meta[i].is_real_image) {
      continue; // Skip padded slots if any were explicitly marked as non-real
                // for processing. However, loop up to CURRENT_BATCH_SIZE
                // handles this primarily.
    }
    const InferenceInput &current_input_param = batch_inputs[i];
    const cv::Mat &current_raw_img_for_saving =
        original_raw_images_for_saving[i];

    // Extract the output slice for this specific image
    // host_output_batched_raw is [TARGET_GPU_BATCH_SIZE, num_attributes,
    // num_detections_per_image] We need the slice for image `i`
    const float *output_for_this_image_start =
        host_output_batched_raw.data() +
        static_cast<size_t>(i) * num_attributes_from_engine *
            num_detections_per_image_from_engine;

    process_single_output(current_input_param, output_for_this_image_start,
                          num_detections_per_image_from_engine,
                          num_attributes_from_engine,
                          current_raw_img_for_saving, batch_image_meta[i], i);
  }
  return true;
}

// Renamed old processOutput to process_single_output
void TensorInferencer::process_single_output(
    const InferenceInput &input_params,
    const float
        *host_output_for_image_raw, // Pointer to start of this image's data
    int num_detections_per_image, int num_attributes_per_detection,
    const cv::Mat &raw_img_for_saving, const BatchImageMetadata &image_meta,
    int original_batch_idx_for_debug) {

  // Transpose output for this single image: from [attributes, detections] to
  // [detections, attributes]
  std::vector<float> transposed_output(
      static_cast<size_t>(num_detections_per_image) *
      num_attributes_per_detection);
  for (int det_idx = 0; det_idx < num_detections_per_image; ++det_idx) {
    for (int attr_idx = 0; attr_idx < num_attributes_per_detection;
         ++attr_idx) {
      transposed_output[static_cast<size_t>(det_idx) *
                            num_attributes_per_detection +
                        attr_idx] =
          host_output_for_image_raw[static_cast<size_t>(attr_idx) *
                                        num_detections_per_image +
                                    det_idx];
    }
  }

  auto it = class_name_to_id_.find(input_params.object_name);
  if (it == class_name_to_id_.end()) { /* error */
    return;
  }
  int target_class_id = it->second;

  float confidence_threshold = 0.01f; // DEBUGGING: Very low confidence
  // float confidence_threshold = std::max(0.1f,
  // input_params.confidence_thresh);
  std::cout << "[DEBUG_SINGLE_OUTPUT] Img " << original_batch_idx_for_debug
            << ": Using conf_thresh: " << confidence_threshold << std::endl;

  std::vector<Detection> detected_objects;
  for (int i = 0; i < num_detections_per_image; ++i) {
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

    // DEBUGGING: Print raw detections
    if (max_score > 0.01) {
      std::cout << "[RAW_DET_SINGLE] Img " << original_batch_idx_for_debug
                << ", Box " << i << ": BestClsID=" << best_class_id
                << " (Name: "
                << (id_to_class_name_.count(best_class_id)
                        ? id_to_class_name_.at(best_class_id)
                        : "Unknown")
                << ")"
                << ", MaxScore=" << max_score << ", CX=" << det_attrs[0]
                << ", CY=" << det_attrs[1] << ", W=" << det_attrs[2]
                << ", H=" << det_attrs[3] << std::endl;
    }

    if (best_class_id == target_class_id && max_score >= confidence_threshold) {
      float cx = det_attrs[0];
      float cy = det_attrs[1];
      float w = det_attrs[2];
      float h = det_attrs[3];
      float x1 = std::max(0.0f, cx - w / 2.0f);
      float y1 = std::max(0.0f, cy - h / 2.0f);
      float x2 = std::min(static_cast<float>(target_w_ - 1),
                          cx + w / 2.0f); // Use target_w_ from class member
      float y2 = std::min(static_cast<float>(target_h_ - 1),
                          cy + h / 2.0f); // Use target_h_ from class member

      if (x2 > x1 && y2 > y1) {
        detected_objects.push_back({x1, y1, x2, y2, max_score, best_class_id,
                                    original_batch_idx_for_debug,
                                    input_params.decoded_frames[0].empty()
                                        ? "PAD"
                                        : "REAL"}); // Store batch index
      }
    }
  }

  std::vector<Detection> nms_detections = applyNMS(detected_objects, 0.45f);
  std::cout << "[NMS_SINGLE] Img " << original_batch_idx_for_debug << " For '"
            << input_params.object_name
            << "': Before NMS=" << detected_objects.size()
            << ", After NMS=" << nms_detections.size() << std::endl;

  for (size_t i = 0; i < nms_detections.size(); ++i) {
    const auto &det = nms_detections[i];
    saveAnnotatedImage(
        raw_img_for_saving, det, // Pass Detection struct
        input_params.object_name, input_params.gopIdx,
        static_cast<int>(i)); // i is now detection count *for this image*
  }

  if (nms_detections.empty()) {
    std::cout << "[INFO_SINGLE] Img " << original_batch_idx_for_debug
              << ": No '" << input_params.object_name
              << "' (GOP: " << input_params.gopIdx << ") meeting criteria."
              << std::endl;
  }
}

float TensorInferencer::calculateIoU(const Detection &a, const Detection &b) {
  // ... (implementation from your last version is fine) ...
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
  // ... (implementation from your last version is fine) ...
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
      // Important for batched NMS: only suppress if detections are from the
      // same original image context The 'original_batch_input_idx' in Detection
      // struct can be used if NMS is on a combined list from batch. However,
      // this applyNMS is now called per image from process_single_output, so
      // this check is not needed here.
      float iou = calculateIoU(sorted_detections[i], sorted_detections[j]);
      if (iou > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
  return result;
}

// Modified saveAnnotatedImage to take Detection struct
void TensorInferencer::saveAnnotatedImage(
    const cv::Mat &raw_img_for_saving, const Detection &det,
    const std::string &class_name_str, int gopIdx,
    int overall_detection_count_in_batch) {
  cv::Mat img_to_save = raw_img_for_saving.clone();
  if (img_to_save.empty()) {
    std::cerr << "[WARN][SAVE] Raw image for saving is empty. GOP: " << gopIdx
              << std::endl;
    return;
  }

  // det.x1, det.y1, det.x2, det.y2 are relative to target_w_, target_h_
  float scale_x = static_cast<float>(raw_img_for_saving.cols) /
                  static_cast<float>(target_w_);
  float scale_y = static_cast<float>(raw_img_for_saving.rows) /
                  static_cast<float>(target_h_);

  int x1_scaled = static_cast<int>(std::round(det.x1 * scale_x));
  int y1_scaled = static_cast<int>(std::round(det.y1 * scale_y));
  int x2_scaled = static_cast<int>(std::round(det.x2 * scale_x));
  int y2_scaled = static_cast<int>(std::round(det.y2 * scale_y));

  x1_scaled = std::max(0, std::min(x1_scaled, raw_img_for_saving.cols - 1));
  y1_scaled = std::max(0, std::min(y1_scaled, raw_img_for_saving.rows - 1));
  x2_scaled = std::max(0, std::min(x2_scaled, raw_img_for_saving.cols - 1));
  y2_scaled = std::max(0, std::min(y2_scaled, raw_img_for_saving.rows - 1));

  if (x2_scaled <= x1_scaled || y2_scaled <= y1_scaled) {
    std::cout << "[WARN][SAVE] Scaled box invalid for GOP " << gopIdx
              << ". Skipping save." << std::endl;
    return;
  }

  cv::rectangle(img_to_save, cv::Point(x1_scaled, y1_scaled),
                cv::Point(x2_scaled, y2_scaled), cv::Scalar(0, 255, 0), 2);

  std::ostringstream label;
  label << class_name_str << " " << std::fixed << std::setprecision(2)
        << det.confidence;
  int baseline = 0;
  cv::Size textSize =
      cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseline);
  baseline += 1;

  cv::Point textOrg(x1_scaled, y1_scaled - 5);
  if (textOrg.y - textSize.height < 0) {
    textOrg.y = y1_scaled + textSize.height + 5;
    if (textOrg.y > raw_img_for_saving.rows - baseline) {
      textOrg.y = raw_img_for_saving.rows - baseline - 2;
    }
  }

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
               << std::setfill('0') << overall_detection_count_in_batch << "_"
               << class_name_str << "_conf"
               << static_cast<int>(det.confidence * 100) << ".jpg";

  bool success = cv::imwrite(filename_oss.str(), img_to_save);
  if (success) {
    std::cout << "[SAVE] Annotated image saved: " << filename_oss.str()
              << std::endl;
  } else {
    std::cerr << "[ERROR] Failed to save image: " << filename_oss.str()
              << std::endl;
  }
}
#include "tensor_inferencer.hpp" // Assuming this declares all necessary members like bindings_, etc.
#include "inference.hpp" // For InferenceInput, InferenceResult (corrected from .cpp)

#include <algorithm>  // For std::sort, std::max, std::min, std::transform
#include <cassert>    // For assert
#include <cmath>      // For std::round, std::fabs
#include <cstdlib>    // For std::getenv, std::exit, std::stoi
#include <filesystem> // For std::filesystem::create_directories (C++17)
#include <fstream>    // For std::ifstream
#include <iomanip> // For std::setw, std::setfill, std::fixed, std::setprecision
#include <iostream>  // For std::cout, std::cerr
#include <numeric>   // For std::iota (if needed)
#include <sstream>   // For std::ostringstream
#include <stdexcept> // For std::runtime_error, std::invalid_argument

// OpenCV includes
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp> // For cv::imwrite (indirectly, often included with imgproc)
#include <opencv2/imgproc/imgproc.hpp> // For cv::resize, cv::cvtColor, cv::rectangle, cv::putText

using namespace nvinfer1;

// Local Detection struct (as it's not in the provided .hpp but needed for
// processing)
struct Detection {
  float x1, y1, x2, y2; // Bounding box coordinates (relative to model input
                        // size target_w_, target_h_)
  float confidence;
  int class_id;
  // int batch_slot_idx; // Not strictly needed here if processing image by
  // image from batch output
};

// Logger class for TensorRT
class TensorInferencerLogger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    // Log messages with severity kINFO, kWARNING, kERROR, kINTERNAL_ERROR
    // kVERBOSE is too noisy for typical use.
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT] " << msg << std::endl;
    }
  }
} gLogger; // Global logger instance

// Helper function to read engine file
static std::vector<char> readEngineFile(const std::string &enginePath) {
  std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "[ERROR] Failed to open engine file: " << enginePath
              << std::endl;
    return {};
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> engineData(size);
  if (!file.read(engineData.data(), size)) {
    std::cerr << "[ERROR] Failed to read engine file: " << enginePath
              << std::endl;
    return {};
  }
  file.close();
  return engineData;
}

TensorInferencer::TensorInferencer(int video_height, int video_width)
    : // Initialize members from .hpp that have defaults or need nullptrs
      runtime_(nullptr), engine_(nullptr), context_(nullptr),
      inputDevice_(nullptr), outputDevice_(nullptr), inputIndex_(-1),
      outputIndex_(-1), stop_flag_(false),
// Assuming these are in the .hpp and will be properly initialized
// num_classes_(0), target_w_(0), target_h_(0),
// inputSizeElementsPerImage_(0), outputSizeElementsPerImage_(0)
// batch_size_ is initialized via getenv
// pending_callback_ initialized to nullptr implicitly or explicitly if needed
{
  // 1. Read BATCH_SIZE from environment variable
  const char *batch_size_env = std::getenv("YOLO_BATCH_SIZE");
  if (!batch_size_env) {
    std::cerr << "[WARN] Environment variable YOLO_BATCH_SIZE not set. "
                 "Defaulting to 1."
              << std::endl;
    batch_size_ = 1;
  } else {
    try {
      batch_size_ = std::stoi(batch_size_env);
      if (batch_size_ <= 0) {
        std::cerr << "[ERROR] YOLO_BATCH_SIZE must be positive ("
                  << batch_size_env << "). Defaulting to 1." << std::endl;
        batch_size_ = 1;
      }
    } catch (const std::invalid_argument &ia) {
      std::cerr << "[ERROR] Invalid YOLO_BATCH_SIZE: " << batch_size_env
                << ". Defaulting to 1. Error: " << ia.what() << std::endl;
      batch_size_ = 1;
    } catch (const std::out_of_range &oor) {
      std::cerr << "[ERROR] YOLO_BATCH_SIZE out of range: " << batch_size_env
                << ". Defaulting to 1. Error: " << oor.what() << std::endl;
      batch_size_ = 1;
    }
  }
  std::cout << "[INFO] Using BATCH_SIZE: " << batch_size_ << std::endl;

  // Initialize target_w_ and target_h_ (these might be overridden by engine
  // dims later) Using roundToNearestMultiple as in the original single-batch
  // version
  target_w_ =
      ((video_width + 31) / 32) * 32; // Round up to nearest multiple of 32
  target_h_ =
      ((video_height + 31) / 32) * 32; // Round up to nearest multiple of 32

  // 2. Construct engine path using BATCH_SIZE
  // First, try the specific YOLO_ENGINE_NAME_<BATCH_SIZE> key
  std::string specific_engine_env_key =
      "YOLO_ENGINE_NAME_" + std::to_string(batch_size_);
  const char *specific_engine_path_env =
      std::getenv(specific_engine_env_key.c_str());

  if (specific_engine_path_env) {
    engine_path_ = specific_engine_path_env;
    std::cout << "[INFO] Using engine path from " << specific_engine_env_key
              << ": " << engine_path_ << std::endl;
  } else {
    // Fallback: try base YOLO_ENGINE_NAME and append _<BATCH_SIZE>.engine
    const char *base_engine_name_env = std::getenv("YOLO_ENGINE_NAME");
    if (base_engine_name_env) {
      engine_path_ = std::string(base_engine_name_env);
      // Append .engine if not present, then insert _<BATCH_SIZE> before .engine
      // This logic is a bit complex; simpler if YOLO_ENGINE_NAME is just a base
      // like "yolov8s" For now, let's assume YOLO_ENGINE_NAME is a full path
      // *without* _<batch_size>.engine yet and we append _<batch_size>.engine
      // A more robust way: if it ends with .engine, insert before it.
      // Otherwise, append. Simplified: expect YOLO_ENGINE_NAME to be just the
      // model name, e.g. "yolov8s_custom" then we append "_<batch_size>.engine"
      engine_path_ = std::string(base_engine_name_env) + "_" +
                     std::to_string(batch_size_) + ".engine";
      std::cout << "[INFO] Constructed engine path from YOLO_ENGINE_NAME: "
                << engine_path_ << std::endl;
    } else {
      std::cerr << "[ERROR] Neither " << specific_engine_env_key
                << " nor YOLO_ENGINE_NAME environment variables are set. "
                   "Cannot determine engine path."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  const char *names_path_env = std::getenv("YOLO_COCO_NAMES");
  if (!names_path_env) {
    std::cerr << "[ERROR] Environment variable YOLO_COCO_NAMES not set."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string names_path_str = names_path_env;

  const char *output_dir_env = std::getenv("YOLO_IMAGE_PATH");
  if (!output_dir_env) {
    std::cerr << "[ERROR] Environment variable YOLO_IMAGE_PATH not set."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  image_output_path_ = output_dir_env;

  // Create output directory if it doesn't exist
  try {
    if (!image_output_path_.empty() &&
        !std::filesystem::exists(image_output_path_)) {
      if (std::filesystem::create_directories(image_output_path_)) {
        std::cout << "[INFO] Created output directory: " << image_output_path_
                  << std::endl;
      } else {
        std::cerr << "[ERROR] Could not create output directory (failed): "
                  << image_output_path_ << std::endl;
        // Not exiting, as it might be a permissions issue but path might still
        // be writable by parent process
      }
    }
  } catch (const std::filesystem::filesystem_error &e) {
    std::cerr << "[ERROR] Filesystem error while creating output directory "
              << image_output_path_ << ": " << e.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Load TensorRT Engine
  auto engineData = readEngineFile(engine_path_);
  if (engineData.empty()) {
    std::exit(EXIT_FAILURE);
  }

  runtime_ = createInferRuntime(gLogger);
  assert(runtime_ != nullptr && "TensorRT runtime creation failed.");
  engine_ =
      runtime_->deserializeCudaEngine(engineData.data(), engineData.size());
  assert(engine_ != nullptr && "TensorRT engine deserialization failed.");
  context_ = engine_->createExecutionContext();
  assert(context_ != nullptr && "TensorRT execution context creation failed.");

  bindings_.resize(engine_->getNbBindings()); // Assuming bindings_ is a member:
                                              // std::vector<void*>

  // Get input and output tensor indices
  inputIndex_ = engine_->getBindingIndex("images"); // Common input tensor name
  outputIndex_ = engine_->getBindingIndex(
      "output0"); // Common output tensor name for YOLO models

  if (inputIndex_ < 0) {
    std::cerr << "[ERROR] Input tensor 'images' not found in engine."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (outputIndex_ < 0) {
    std::cout << "[WARN] Output tensor 'output0' not found by name. Attempting "
                 "to find first non-input tensor."
              << std::endl;
    bool found_output = false;
    for (int i = 0; i < engine_->getNbBindings(); ++i) {
      if (!engine_->bindingIsInput(i)) {
        outputIndex_ = i;
        std::cout << "[INFO] Using tensor '" << engine_->getBindingName(i)
                  << "' at index " << i << " as output." << std::endl;
        found_output = true;
        break;
      }
    }
    if (!found_output) {
      std::cerr << "[ERROR] Could not determine a valid output tensor index."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // Load class names
  std::ifstream infile(names_path_str);
  if (!infile.is_open()) {
    std::cerr << "[ERROR] Cannot open class names file: " << names_path_str
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  int class_idx = 0;
  while (std::getline(infile, line)) {
    if (!line.empty()) {
      class_name_to_id_[line] = class_idx;
      id_to_class_name_[class_idx] = line;
      class_idx++;
    }
  }
  num_classes_ = class_name_to_id_.size();
  infile.close();
  std::cout << "[INFO] Loaded " << num_classes_ << " class names." << std::endl;
  if (num_classes_ == 0) {
    std::cerr << "[ERROR] No class names loaded. Check COCO names file: "
              << names_path_str << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Determine model input dimensions (H, W) from engine if possible, otherwise
  // use defaults
  Dims reportedInputDims =
      engine_->getBindingDimensions(inputIndex_); // These are opt profile dims
  // reportedInputDims.d[0] is batch, d[1] is C, d[2] is H, d[3] is W
  if (reportedInputDims.nbDims == 4 && reportedInputDims.d[2] > 0 &&
      reportedInputDims.d[3] > 0) {
    target_h_ = reportedInputDims.d[2];
    target_w_ = reportedInputDims.d[3];
    std::cout << "[INFO] Using engine's input dimensions for preprocessing: H="
              << target_h_ << ", W=" << target_w_ << std::endl;
  } else {
    std::cout << "[INFO] Using constructor-derived input dimensions for "
                 "preprocessing: H="
              << target_h_ << ", W=" << target_w_ << std::endl;
  }

  // Validate tensor data types
  if (engine_->getBindingDataType(inputIndex_) != nvinfer1::DataType::kFLOAT) {
    std::cerr << "[ERROR] Engine input tensor '"
              << engine_->getBindingName(inputIndex_)
              << "' is not DataType::kFLOAT." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (engine_->getBindingDataType(outputIndex_) != nvinfer1::DataType::kFLOAT) {
    std::cerr << "[ERROR] Engine output tensor '"
              << engine_->getBindingName(outputIndex_)
              << "' is not DataType::kFLOAT." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  printEngineInfo(); // Print detailed engine info after setup

  // Set runtime dimensions for the context to determine buffer sizes
  // This needs to be done for the *actual batch_size_* we will use.
  Dims inputDimsFullBatch{4, {batch_size_, 3, target_h_, target_w_}};
  if (!context_->setBindingDimensions(inputIndex_, inputDimsFullBatch)) {
    std::cerr << "[ERROR] Constructor: Failed to set binding dimensions for "
                 "input tensor to determine buffer sizes."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!context_->allInputDimensionsSpecified()) { // Important check
    std::cerr << "[ERROR] Constructor: Not all input dimensions specified "
                 "after setting for full batch."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Dims outputDimsFullBatch = context_->getBindingDimensions(outputIndex_);
  // Expected output for YOLO: [batch_size, num_attributes (4_bbox +
  // num_classes), num_detections_yolo] Example: [16, 84, 8400] for COCO (80
  // classes)
  if (outputDimsFullBatch.nbDims != 3 ||
      outputDimsFullBatch.d[0] != batch_size_ || // Batch dimension should match
      outputDimsFullBatch.d[1] != (4 + num_classes_)) {
    std::cerr << "[ERROR] Output tensor dimensions from engine are unexpected "
                 "for YOLO model structure."
              << std::endl;
    std::cerr
        << "  Expected Dims: 3 (Batch x Attributes x Detections_Per_Image)"
        << std::endl;
    std::cerr << "  Expected Batch: " << batch_size_
              << ", Got: " << outputDimsFullBatch.d[0] << std::endl;
    std::cerr << "  Expected Attributes (4+NumClasses=" << 4 + num_classes_
              << "): " << (4 + num_classes_)
              << ", Got: " << outputDimsFullBatch.d[1] << std::endl;
    std::cerr << "  Actual Output Dims: ";
    for (int i = 0; i < outputDimsFullBatch.nbDims; ++i)
      std::cerr << outputDimsFullBatch.d[i]
                << (i == outputDimsFullBatch.nbDims - 1 ? "" : "x");
    std::cerr << std::endl;
    // If engine has fixed batch size different from configured, this is an
    // issue. Or if num_classes doesn't match model output.
    std::exit(EXIT_FAILURE);
  }

  // Calculate per-image and total batch sizes for GPU buffers
  inputSizeElementsPerImage_ = static_cast<size_t>(3) * target_h_ * target_w_;
  size_t totalInputElementsForBatch =
      static_cast<size_t>(batch_size_) * inputSizeElementsPerImage_;

  outputSizeElementsPerImage_ =
      static_cast<size_t>(outputDimsFullBatch.d[1]) * outputDimsFullBatch.d[2];
  size_t totalOutputElementsForBatch =
      static_cast<size_t>(batch_size_) * outputSizeElementsPerImage_;

  // Store these total sizes in member variables from .hpp if they are meant for
  // batch totals Assuming inputSize_ and outputSize_ from .hpp are for total
  // batch byte sizes
  inputSize_ = totalInputElementsForBatch * sizeof(float);
  outputSize_ = totalOutputElementsForBatch * sizeof(float);

  // Allocate GPU Memory for the full batch
  cudaError_t err;
  err = cudaMalloc(&inputDevice_, inputSize_);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] CUDA Malloc Input (Batch): "
              << cudaGetErrorString(err) << " (Size: " << inputSize_
              << " bytes)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  err = cudaMalloc(&outputDevice_, outputSize_);
  if (err != cudaSuccess) {
    cudaFree(inputDevice_); // Free already allocated memory
    std::cerr << "[ERROR] CUDA Malloc Output (Batch): "
              << cudaGetErrorString(err) << " (Size: " << outputSize_
              << " bytes)" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  bindings_[inputIndex_] = inputDevice_;
  bindings_[outputIndex_] = outputDevice_;

  std::cout << "[INFO] GPU buffers allocated for batch size " << batch_size_
            << "." << std::endl;
  std::cout << "[INFO]   Input elements per image: "
            << inputSizeElementsPerImage_
            << ", Total for batch: " << totalInputElementsForBatch
            << " (Bytes: " << inputSize_ << ")" << std::endl;
  std::cout << "[INFO]   Output elements per image: "
            << outputSizeElementsPerImage_
            << ", Total for batch: " << totalOutputElementsForBatch
            << " (Bytes: " << outputSize_ << ")" << std::endl;

  // Start the inference worker thread
  inference_thread_ = std::thread(&TensorInferencer::run, this);
  std::cout << "[INFO] TensorInferencer initialized and worker thread started."
            << std::endl;
}

TensorInferencer::~TensorInferencer() {
  std::cout << "[DEINIT] Stopping TensorInferencer worker thread..."
            << std::endl;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    stop_flag_ = true;
  }
  cv_.notify_all(); // Notify worker thread to wake up and check stop_flag_
  if (inference_thread_.joinable()) {
    inference_thread_.join();
  }
  std::cout << "[DEINIT] Worker thread joined." << std::endl;

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
  std::cout << "[DEINIT] TensorInferencer resources released." << std::endl;
}

void TensorInferencer::printEngineInfo() {
  std::cout << "\n=== TensorRT Engine Information ===" << std::endl;
  std::cout << "  Engine Path: " << engine_path_ << std::endl;
  std::cout << "  Batch Size (configured): " << batch_size_ << std::endl;
  std::cout << "  Target Preprocessing (H x W): " << target_h_ << " x "
            << target_w_ << std::endl;
  std::cout << "  Number of Bindings: " << engine_->getNbBindings()
            << std::endl;

  for (int i = 0; i < engine_->getNbBindings(); ++i) {
    const char *name = engine_->getBindingName(i);
    Dims dims = engine_->getBindingDimensions(i); // Opt profile dims
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
    std::cout << "  Binding " << i << ": Name='" << name
              << "', Role=" << (isInput ? "Input" : "Output")
              << ", DataType=" << dtype_str << ", Dims (Opt Profile): [";
    for (int j = 0; j < dims.nbDims; ++j) {
      std::cout << dims.d[j] << (j == dims.nbDims - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
  }
  // Also print runtime dimensions for current context if needed (after
  // setBindingDimensions)
  if (context_ && inputIndex_ >= 0 && outputIndex_ >= 0) {
    Dims runtimeInputDims = context_->getBindingDimensions(inputIndex_);
    Dims runtimeOutputDims = context_->getBindingDimensions(outputIndex_);
    std::cout << "  Runtime Input Dims (current context): [";
    for (int j = 0; j < runtimeInputDims.nbDims; ++j)
      std::cout << runtimeInputDims.d[j]
                << (j == runtimeInputDims.nbDims - 1 ? "" : ", ");
    std::cout << "]" << std::endl;
    std::cout << "  Runtime Output Dims (current context): [";
    for (int j = 0; j < runtimeOutputDims.nbDims; ++j)
      std::cout << runtimeOutputDims.d[j]
                << (j == runtimeOutputDims.nbDims - 1 ? "" : ", ");
    std::cout << "]" << std::endl;
  }
  std::cout << "===================================\n" << std::endl;
}

// This overload is from the original single-batch code.
// WARNING: Its usage of class members inputSize_ and outputSize_ (now batch
// total bytes) and device pointers (inputDevice_, outputDevice_ for batch)
// makes it unsafe to use concurrently with the batching `infer` method. It's
// kept as per prompt (no changes to this specific function requested for
// removal). If used, it would need its own buffer management or careful
// synchronization.
bool TensorInferencer::infer(const std::vector<float> &input_vector_cpu,
                             std::vector<float> &output_vector_cpu) {
  std::cerr << "[WARN] The infer(std::vector<float>...) overload is likely "
               "unsafe with batching due to shared GPU buffers and size "
               "assumptions. Use with extreme caution or refactor."
            << std::endl;

  // This function assumes input_vector_cpu is for a single image and its size
  // matches inputSizeElementsPerImage_
  if (inputSizeElementsPerImage_ == 0 || outputSizeElementsPerImage_ == 0) {
    std::cerr
        << "[ERROR] infer(vector<float>): Per-image sizes not initialized."
        << std::endl;
    return false;
  }
  if (input_vector_cpu.size() != inputSizeElementsPerImage_) {
    std::cerr
        << "[ERROR] infer(vector<float>): Input vector size mismatch. Expected "
        << inputSizeElementsPerImage_ << " (for single image), got "
        << input_vector_cpu.size() << std::endl;
    return false;
  }
  output_vector_cpu.resize(
      outputSizeElementsPerImage_); // Resize for single image output

  // This would use the main batch buffers, which is incorrect for a single
  // float vector. For this to work, it would need to:
  // 1. Allocate temporary small GPU buffers.
  // 2. Set context binding dimensions for batch_size=1.
  // 3. Perform inference.
  // 4. Restore context binding dimensions if they were changed.
  // As is, it will likely corrupt batch processing or use incorrect dimensions.

  // Simplified (and potentially problematic) use of existing batch buffers:
  cudaError_t err;
  // Copy only single image data to the start of the batch input buffer
  err = cudaMemcpy(bindings_[inputIndex_], input_vector_cpu.data(),
                   inputSizeElementsPerImage_ * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] infer(vector<float>): CUDA Memcpy H2D: "
              << cudaGetErrorString(err) << std::endl;
    return false;
  }

  // Critical: The context might be set for a larger batch size. This enqueue
  // will be problematic. For a single image, context dimensions should be set
  // to [1, C, H, W] Temporarily setting for BATCH_SIZE=1 for this call:
  Dims singleImageInputDims{4, {1, 3, target_h_, target_w_}};
  if (!context_->setBindingDimensions(inputIndex_, singleImageInputDims)) {
    std::cerr << "[ERROR] infer(vector<float>): Failed to set binding "
                 "dimensions for single image."
              << std::endl;
    return false;
  }

  if (!context_->enqueueV2(bindings_.data(), 0,
                           nullptr)) { // Use default stream
    std::cerr << "[ERROR] infer(vector<float>): enqueueV2 failed." << std::endl;
    // Restore batch dimensions? This is getting complicated.
    Dims fullBatchInputDims{4, {batch_size_, 3, target_h_, target_w_}};
    context_->setBindingDimensions(inputIndex_,
                                   fullBatchInputDims); // Attempt to restore
    return false;
  }

  // Copy only single image output from the start of the batch output buffer
  err = cudaMemcpy(output_vector_cpu.data(), bindings_[outputIndex_],
                   outputSizeElementsPerImage_ * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] infer(vector<float>): CUDA Memcpy D2H: "
              << cudaGetErrorString(err) << std::endl;
    // Restore batch dimensions before returning
    Dims fullBatchInputDims{4, {batch_size_, 3, target_h_, target_w_}};
    context_->setBindingDimensions(inputIndex_,
                                   fullBatchInputDims); // Attempt to restore
    return false;
  }

  // Restore context binding dimensions to the full batch size for the main
  // worker thread
  Dims fullBatchInputDims{4, {batch_size_, 3, target_h_, target_w_}};
  if (!context_->setBindingDimensions(inputIndex_, fullBatchInputDims)) {
    std::cerr << "[WARN] infer(vector<float>): Failed to restore full batch "
                 "binding dimensions."
              << std::endl;
  }

  return true;
}

// Main public API for submitting inference requests
void TensorInferencer::infer(const InferenceInput &input,
                             InferenceCallback callback) {
  if (stop_flag_) {
    std::cerr << "[WARN] Inferencer is stopping. Request for gopIdx "
              << input.gopIdx << " ignored." << std::endl;
    // Optionally, invoke callback with an error immediately
    if (callback) {
      std::vector<InferenceResult> results;
      results.push_back({"Inferencer stopping, request ignored."});
      callback(results);
    }
    return;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  pending_inputs_.push_back(input);
  // Store the callback. If multiple calls to infer() happen before a batch is
  // full, the last callback will overwrite previous ones for that forming
  // batch. This design implies one callback per batch, not per input item. If
  // per-input-item callback is needed, the design needs to change (e.g., store
  // pairs of <Input, Callback>). For now, assume one callback for the whole
  // batch.
  pending_callback_ = callback;
  lock.unlock();
  cv_.notify_one(); // Notify the worker thread that new data is available
}

// Worker thread main loop
void TensorInferencer::run() {
  std::vector<InferenceInput> current_processing_batch;
  InferenceCallback callback_for_this_batch = nullptr;

  while (true) {
    current_processing_batch.clear();  // Clear for next batch
    callback_for_this_batch = nullptr; // Reset callback

    { // Lock scope for accessing shared data (pending_inputs_, stop_flag_,
      // pending_callback_)
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] {
        // Wait if not stopping AND (pending inputs is less than batch size)
        return stop_flag_ ||
               pending_inputs_.size() >= static_cast<size_t>(batch_size_);
      });

      if (stop_flag_ && pending_inputs_.empty()) {
        break; // Exit loop if stopping and no more items to process
      }

      size_t num_to_grab = 0;
      if (pending_inputs_.size() >= static_cast<size_t>(batch_size_)) {
        num_to_grab = batch_size_;
      } else if (stop_flag_ && !pending_inputs_.empty()) {
        // If stopping, process any remaining items, even if less than a full
        // batch
        num_to_grab = pending_inputs_.size();
      } else {
        // Spurious wakeup or not enough items and not stopping, continue
        // waiting
        continue;
      }

      if (num_to_grab > 0) {
        current_processing_batch.reserve(num_to_grab);
        for (size_t i = 0; i < num_to_grab; ++i) {
          current_processing_batch.push_back(
              std::move(pending_inputs_.front()));
          pending_inputs_.erase(
              pending_inputs_.begin()); // Efficient for std::vector if it's
                                        // small or use std::deque
        }
        callback_for_this_batch =
            pending_callback_; // Get the callback associated with this batch
        // Reset pending_callback_ if it's meant to be cleared after being
        // grabbed for a batch. pending_callback_ = nullptr; // Or handle this
        // based on desired callback semantics.
      }
    } // Mutex is released here

    if (!current_processing_batch.empty()) {
      std::vector<InferenceResult>
          batch_results; // Results for this specific batch
      performInference(current_processing_batch, batch_results);

      if (callback_for_this_batch) {
        try {
          callback_for_this_batch(batch_results);
        } catch (const std::exception &e) {
          std::cerr << "[ERROR] Exception in user-provided callback: "
                    << e.what() << std::endl;
        } catch (...) {
          std::cerr << "[ERROR] Unknown exception in user-provided callback."
                    << std::endl;
        }
      } else if (!batch_results
                      .empty()) { // Processed something but no callback
        std::cout << "[WARN] Worker thread: Batch processed but no callback "
                     "was set for it."
                  << std::endl;
      }
    }
  }
  std::cout << "[INFO] Inference worker thread run() loop finished."
            << std::endl;
}

// Performs inference on a prepared batch of inputs
void TensorInferencer::performInference(
    const std::vector<InferenceInput>
        &batch_inputs_from_queue, // Actual inputs for this batch
    std::vector<InferenceResult>
        &batch_results_out // Output results to be populated
) {
  if (batch_inputs_from_queue.empty()) {
    return; // Nothing to process
  }

  int num_actual_images_in_this_batch = batch_inputs_from_queue.size();
  std::vector<float> batched_input_data_cpu; // CPU buffer for the entire batch
  // Reserve space for a full batch_size_ even if
  // num_actual_images_in_this_batch is smaller (due to padding)
  batched_input_data_cpu.reserve(static_cast<size_t>(batch_size_) *
                                 inputSizeElementsPerImage_);

  // --- Preprocess images and pad if necessary ---
  for (int i = 0; i < batch_size_; ++i) {
    cv::Mat img_to_preprocess;
    bool is_real_image = (i < num_actual_images_in_this_batch);

    if (is_real_image) {
      const InferenceInput &current_item = batch_inputs_from_queue[i];
      if (current_item.decoded_frames.empty() ||
          current_item.decoded_frames[0].empty()) {
        std::cerr
            << "[WARN] PerformInference: Empty frame provided for batch item "
            << i << " (gopIdx " << current_item.gopIdx
            << "). Using dummy image." << std::endl;
        // Create a dummy image (e.g., gray or black) of target_h_ x target_w_
        img_to_preprocess =
            cv::Mat(target_h_, target_w_, CV_8UC3, cv::Scalar(114, 114, 114));
      } else {
        img_to_preprocess = current_item.decoded_frames[0];
      }
    } else { // This is a padding slot
      img_to_preprocess =
          cv::Mat(target_h_, target_w_, CV_8UC3,
                  cv::Scalar(114, 114, 114)); // Dummy padding image
    }

    // Preprocessing steps for one image:
    cv::Mat resized_img;
    if (img_to_preprocess.cols != target_w_ ||
        img_to_preprocess.rows != target_h_) {
      cv::resize(img_to_preprocess, resized_img,
                 cv::Size(target_w_, target_h_));
    } else {
      resized_img = img_to_preprocess; // Already correct size
    }

    cv::Mat img_rgb;
    cv::cvtColor(resized_img, img_rgb, cv::COLOR_BGR2RGB);

    cv::Mat chw_input_fp32_single; // For one image
    img_rgb.convertTo(chw_input_fp32_single, CV_32FC3,
                      1.0 / 255.0); // Normalize to [0,1]

    // Convert to CHW (planar) format and append to batched_input_data_cpu
    // RRR...GGG...BBB...
    for (int ch = 0; ch < 3; ++ch) { // Iterate R, G, B channels
      for (int y = 0; y < target_h_; ++y) {
        for (int x = 0; x < target_w_; ++x) {
          batched_input_data_cpu.push_back(
              chw_input_fp32_single.at<cv::Vec3f>(y, x)[ch]);
        }
      }
    }
  } // End of preprocessing loop for batch

  // --- Set runtime dimensions for the batch (always use batch_size_ for the
  // engine context) --- This was already done in constructor and after single
  // infer(vector<float>), but good to ensure.
  Dims currentInputDims = context_->getBindingDimensions(inputIndex_);
  if (currentInputDims.d[0] != batch_size_ ||
      currentInputDims.d[2] != target_h_ ||
      currentInputDims.d[3] != target_w_) {
    Dims inputDimsRuntime{4, {batch_size_, 3, target_h_, target_w_}};
    if (!context_->setBindingDimensions(inputIndex_, inputDimsRuntime)) {
      std::cerr << "[ERROR] PerformInference: Failed to set binding dimensions "
                   "for input tensor."
                << std::endl;
      for (int i = 0; i < num_actual_images_in_this_batch; ++i)
        batch_results_out.push_back(
            {"Error: Failed to set TRT binding dimensions"});
      return;
    }
    if (!context_->allInputDimensionsSpecified()) {
      std::cerr << "[ERROR] PerformInference: Not all input dimensions "
                   "specified after setting."
                << std::endl;
      for (int i = 0; i < num_actual_images_in_this_batch; ++i)
        batch_results_out.push_back({"Error: Incomplete TRT input dimensions"});
      return;
    }
  }

  // --- Perform Inference ---
  cudaError_t err;
  // Total input bytes for the full batch_size_ (CPU buffer should also match
  // this size)
  size_t total_input_bytes_for_batch = static_cast<size_t>(batch_size_) *
                                       inputSizeElementsPerImage_ *
                                       sizeof(float);
  if (batched_input_data_cpu.size() * sizeof(float) !=
      total_input_bytes_for_batch) {
    std::cerr
        << "[ERROR] PerformInference: CPU batch data size mismatch. Expected "
        << total_input_bytes_for_batch << " bytes, got "
        << batched_input_data_cpu.size() * sizeof(float) << " bytes."
        << std::endl;
    for (int i = 0; i < num_actual_images_in_this_batch; ++i)
      batch_results_out.push_back({"Error: CPU batch data size mismatch"});
    return;
  }

  err = cudaMemcpy(inputDevice_, batched_input_data_cpu.data(),
                   total_input_bytes_for_batch, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] PerformInference: CUDA Memcpy H2D failed: "
              << cudaGetErrorString(err) << std::endl;
    for (int i = 0; i < num_actual_images_in_this_batch; ++i)
      batch_results_out.push_back({"Error: CUDA H2D copy failed"});
    return;
  }

  if (!context_->enqueueV2(bindings_.data(), 0,
                           nullptr)) { // Using default CUDA stream 0
    std::cerr << "[ERROR] PerformInference: TensorRT enqueueV2 failed."
              << std::endl;
    for (int i = 0; i < num_actual_images_in_this_batch; ++i)
      batch_results_out.push_back(
          {"Error: TensorRT inference execution failed"});
    return;
  }

  // Total output bytes for the full batch_size_
  size_t total_output_bytes_for_batch = static_cast<size_t>(batch_size_) *
                                        outputSizeElementsPerImage_ *
                                        sizeof(float);
  std::vector<float> host_output_raw_full_batch(
      static_cast<size_t>(batch_size_) *
      outputSizeElementsPerImage_); // CPU buffer for all outputs

  err = cudaMemcpy(host_output_raw_full_batch.data(), outputDevice_,
                   total_output_bytes_for_batch, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] PerformInference: CUDA Memcpy D2H failed: "
              << cudaGetErrorString(err) << std::endl;
    for (int i = 0; i < num_actual_images_in_this_batch; ++i)
      batch_results_out.push_back({"Error: CUDA D2H copy failed"});
    return;
  }

  // --- Postprocess results for each *actual* image in the batch ---
  Dims batchOutDimsFromContext = context_->getBindingDimensions(
      outputIndex_); // Should be [batch_size_, attrs, dets_per_img]

  for (int i = 0; i < num_actual_images_in_this_batch; ++i) {
    const InferenceInput &current_original_input_params =
        batch_inputs_from_queue[i];
    const cv::Mat &current_raw_img_for_item =
        (current_original_input_params.decoded_frames.empty() ||
         current_original_input_params.decoded_frames[0].empty())
            ? cv::Mat()
            : // Pass empty Mat if original was bad/dummy
            current_original_input_params.decoded_frames[0];

    // Get the slice of output data for this specific image from the full batch
    // output
    const float *single_image_output_ptr_start =
        host_output_raw_full_batch.data() + (i * outputSizeElementsPerImage_);
    std::vector<float> single_image_output_data_slice(
        single_image_output_ptr_start,
        single_image_output_ptr_start + outputSizeElementsPerImage_);

    InferenceResult current_image_result_obj; // To be populated by
                                              // processDetectionsForOneImage
    processDetectionsForOneImage(
        current_original_input_params, single_image_output_data_slice,
        current_raw_img_for_item,
        batchOutDimsFromContext, // Pass the batch output dims for structure
                                 // reference
        i, // image_idx_in_batch for logging or unique naming
        current_image_result_obj // Pass by reference to be filled
    );
    batch_results_out.push_back(current_image_result_obj);
  }
}

// Processes detections for a single image from the batch output
void TensorInferencer::processDetectionsForOneImage(
    const InferenceInput
        &original_input_params, // Original parameters for this image
    const std::vector<float>
        &single_image_raw_output, // Raw output tensor slice for ONE image
    const cv::Mat &raw_img_for_this_item, // The original raw image (can be
                                          // empty if input was bad)
    const nvinfer1::Dims &batchOutDims,   // Full batch output Dims from context
                                        // (e.g., [batch, attrs, dets_per_img])
    int image_idx_in_batch, // Index of this image within the processed batch
                            // (for logging)
    InferenceResult &result_for_this_image_out // Result object to populate
) {
  if (raw_img_for_this_item.empty()) {
    std::cout << "[INFO] processDetections: Skipping processing for gopIdx "
              << original_input_params.gopIdx << " (batch slot "
              << image_idx_in_batch << ") as raw image was empty/invalid."
              << std::endl;
    result_for_this_image_out.info = "Skipped processing: Invalid raw image.";
    return;
  }

  // batchOutDims.d[0] is batch_size
  // batchOutDims.d[1] is num_attributes (e.g., 4 bbox + num_classes)
  // batchOutDims.d[2] is num_detections_per_image (e.g., 8400 for YOLOv8)
  int num_attributes_per_detection = batchOutDims.d[1];
  int num_potential_detections_per_image = batchOutDims.d[2];

  if (single_image_raw_output.size() !=
      static_cast<size_t>(num_attributes_per_detection *
                          num_potential_detections_per_image)) {
    std::cerr << "[ERROR][OUTPUT] processDetections: single_image_raw_output "
                 "size mismatch for gopIdx "
              << original_input_params.gopIdx << "." << std::endl;
    std::cerr << "  Expected elements: "
              << (num_attributes_per_detection *
                  num_potential_detections_per_image)
              << ", Got: " << single_image_raw_output.size() << std::endl;
    result_for_this_image_out.info =
        "Error: Output data size mismatch for this image.";
    return;
  }

  // Transpose output for this single image: from [num_attributes,
  // num_detections] to [num_detections, num_attributes] This makes iterating
  // through detections easier.
  std::vector<float> transposed_output_single_image(
      single_image_raw_output.size());
  for (int det_idx = 0; det_idx < num_potential_detections_per_image;
       ++det_idx) {
    for (int attr_idx = 0; attr_idx < num_attributes_per_detection;
         ++attr_idx) {
      transposed_output_single_image[static_cast<size_t>(det_idx) *
                                         num_attributes_per_detection +
                                     attr_idx] =
          single_image_raw_output[static_cast<size_t>(attr_idx) *
                                      num_potential_detections_per_image +
                                  det_idx];
    }
  }

  auto it_target_class =
      class_name_to_id_.find(original_input_params.object_name);
  if (it_target_class == class_name_to_id_.end()) {
    std::cerr << "[ERROR][OUTPUT] processDetections: Target class '"
              << original_input_params.object_name
              << "' not found in loaded class names for gopIdx "
              << original_input_params.gopIdx << "." << std::endl;
    result_for_this_image_out.info = "Error: Target class name not found.";
    return;
  }
  int target_class_id = it_target_class->second;
  float confidence_threshold = std::max(
      0.01f,
      original_input_params.confidence_thresh); // Ensure some minimum threshold

  std::vector<Detection>
      candidate_detections; // Store detections meeting class and confidence
                            // criteria before NMS
  for (int i = 0; i < num_potential_detections_per_image; ++i) {
    const float *det_attrs_ptr =
        &transposed_output_single_image[static_cast<size_t>(i) *
                                        num_attributes_per_detection];
    // det_attrs_ptr points to [cx, cy, w, h, class_score_0, class_score_1, ...,
    // class_score_N-1]

    float max_class_score = 0.0f;
    int best_class_id_for_this_box = -1;
    // Class scores start at index 4 (after cx, cy, w, h)
    for (int j = 0; j < num_classes_; ++j) {
      float score = det_attrs_ptr[4 + j];
      if (score > max_class_score) {
        max_class_score = score;
        best_class_id_for_this_box = j;
      }
    }

    if (best_class_id_for_this_box == target_class_id &&
        max_class_score >= confidence_threshold) {
      float cx_model = det_attrs_ptr[0]; // Center X (relative to target_w_)
      float cy_model = det_attrs_ptr[1]; // Center Y (relative to target_h_)
      float w_model = det_attrs_ptr[2];  // Width (relative to target_w_)
      float h_model = det_attrs_ptr[3];  // Height (relative to target_h_)

      // Convert to x1, y1, x2, y2 (still relative to target_w_, target_h_ model
      // input)
      float x1_model = std::max(0.0f, cx_model - w_model / 2.0f);
      float y1_model = std::max(0.0f, cy_model - h_model / 2.0f);
      float x2_model =
          std::min(static_cast<float>(target_w_), cx_model + w_model / 2.0f);
      float y2_model =
          std::min(static_cast<float>(target_h_), cy_model + h_model / 2.0f);

      if (x2_model > x1_model &&
          y2_model > y1_model) { // Ensure valid box dimensions
        candidate_detections.push_back({x1_model, y1_model, x2_model, y2_model,
                                        max_class_score,
                                        best_class_id_for_this_box});
      }
    }
  }

  std::vector<Detection> nms_filtered_detections =
      applyNMS(candidate_detections, 0.45f); // Standard IoU threshold for NMS

  std::ostringstream detection_summary_stream;
  if (nms_filtered_detections.empty()) {
    detection_summary_stream
        << "No '" << original_input_params.object_name
        << "' detected meeting criteria in image for gopIdx "
        << original_input_params.gopIdx << " (batch slot " << image_idx_in_batch
        << ").";
  } else {
    detection_summary_stream
        << "Detected " << nms_filtered_detections.size() << " instance(s) of '"
        << original_input_params.object_name << "' for gopIdx "
        << original_input_params.gopIdx << " (batch slot " << image_idx_in_batch
        << "): ";
  }

  for (size_t i = 0; i < nms_filtered_detections.size(); ++i) {
    const auto &det = nms_filtered_detections[i];

    // Save annotated image. This function also handles scaling to
    // raw_img_for_this_item dimensions.
    saveAnnotatedImage(
        raw_img_for_this_item, det.x1, det.y1, det.x2, det.y2, det.confidence,
        original_input_params.object_name, original_input_params.gopIdx,
        static_cast<int>(i)); // detection_idx_in_image for unique naming

    // --- Print detection info to console (scaled to original image dimensions)
    // ---
    float scale_x_to_raw = static_cast<float>(raw_img_for_this_item.cols) /
                           static_cast<float>(target_w_);
    float scale_y_to_raw = static_cast<float>(raw_img_for_this_item.rows) /
                           static_cast<float>(target_h_);

    int final_x1_raw = static_cast<int>(std::round(det.x1 * scale_x_to_raw));
    int final_y1_raw = static_cast<int>(std::round(det.y1 * scale_y_to_raw));
    int final_x2_raw = static_cast<int>(std::round(det.x2 * scale_x_to_raw));
    int final_y2_raw = static_cast<int>(std::round(det.y2 * scale_y_to_raw));

    // Clamp to raw image boundaries
    final_x1_raw =
        std::max(0, std::min(final_x1_raw, raw_img_for_this_item.cols - 1));
    final_y1_raw =
        std::max(0, std::min(final_y1_raw, raw_img_for_this_item.rows - 1));
    final_x2_raw =
        std::max(0, std::min(final_x2_raw, raw_img_for_this_item.cols - 1));
    final_y2_raw =
        std::max(0, std::min(final_y2_raw, raw_img_for_this_item.rows - 1));

    int final_w_raw = final_x2_raw - final_x1_raw;
    int final_h_raw = final_y2_raw - final_y1_raw;

    if (final_w_raw > 0 && final_h_raw > 0) {
      std::cout << "  [DETECTED] gopIdx=" << original_input_params.gopIdx
                << ", slot=" << image_idx_in_batch << ", obj_idx=" << i
                << ", Class: " << original_input_params.object_name
                << ", BBOX_Raw: [" << final_x1_raw << ", " << final_y1_raw
                << ", " << final_w_raw << ", " << final_h_raw
                << "], Score: " << std::fixed << std::setprecision(4)
                << det.confidence << std::endl;
      if (i > 0)
        detection_summary_stream
            << "; "; // Separator for multiple detections in summary
      detection_summary_stream << "Det" << i << "_RawBox: [" << final_x1_raw
                               << "," << final_y1_raw << "," << final_w_raw
                               << "," << final_h_raw << "]@" << std::fixed
                               << std::setprecision(2) << det.confidence;
    }
  }
  result_for_this_image_out.info = detection_summary_stream.str();
}

// NMS implementation (IoU calculation and filtering)
float TensorInferencer::calculateIoU(const Detection &a, const Detection &b) {
  // Coords are x1, y1, x2, y2
  float x1_intersect = std::max(a.x1, b.x1);
  float y1_intersect = std::max(a.y1, b.y1);
  float x2_intersect = std::min(a.x2, b.x2);
  float y2_intersect = std::min(a.y2, b.y2);

  float intersection_width = std::max(0.0f, x2_intersect - x1_intersect);
  float intersection_height = std::max(0.0f, y2_intersect - y1_intersect);
  float intersection_area = intersection_width * intersection_height;

  if (intersection_area <= 0.0f) // Or check width/height <= 0
    return 0.0f;

  float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  float union_area = area_a + area_b - intersection_area;

  return (union_area > 0.0f) ? (intersection_area / union_area) : 0.0f;
}

std::vector<Detection>
TensorInferencer::applyNMS(const std::vector<Detection> &detections,
                           float iou_threshold) {
  if (detections.empty()) {
    return {};
  }
  std::vector<Detection> sorted_detections = detections;
  // Sort detections by confidence in descending order
  std::sort(sorted_detections.begin(), sorted_detections.end(),
            [](const Detection &a, const Detection &b) {
              return a.confidence > b.confidence;
            });

  std::vector<Detection> nms_results;
  std::vector<bool> suppressed(sorted_detections.size(), false);

  for (size_t i = 0; i < sorted_detections.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    nms_results.push_back(sorted_detections[i]);
    for (size_t j = i + 1; j < sorted_detections.size(); ++j) {
      if (suppressed[j]) {
        continue;
      }
      float current_iou =
          calculateIoU(sorted_detections[i], sorted_detections[j]);
      if (current_iou > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
  return nms_results;
}

// Saves image with annotations
void TensorInferencer::saveAnnotatedImage(
    const cv::Mat &raw_img_param, // Original image for this detection
    float x1_model, float y1_model, float x2_model,
    float y2_model, // Coords relative to model input (target_w_, target_h_)
    float confidence, const std::string &class_name, int gopIdx,
    int detection_idx_in_image // Unique index for this detection within this
                               // specific image
) {
  if (raw_img_param.empty()) {
    std::cerr << "[WARN][SAVE] Raw image is empty for gopIdx " << gopIdx
              << ", detection " << detection_idx_in_image << ". Cannot save."
              << std::endl;
    return;
  }
  cv::Mat img_to_save = raw_img_param.clone(); // Work on a copy

  // Scale model coordinates (relative to target_w_, target_h_) back to
  // raw_img_param dimensions
  float scale_x_to_raw =
      static_cast<float>(img_to_save.cols) / static_cast<float>(target_w_);
  float scale_y_to_raw =
      static_cast<float>(img_to_save.rows) / static_cast<float>(target_h_);

  int x1_scaled_raw = static_cast<int>(std::round(x1_model * scale_x_to_raw));
  int y1_scaled_raw = static_cast<int>(std::round(y1_model * scale_y_to_raw));
  int x2_scaled_raw = static_cast<int>(std::round(x2_model * scale_x_to_raw));
  int y2_scaled_raw = static_cast<int>(std::round(y2_model * scale_y_to_raw));

  // Clamp coordinates to be within the image boundaries
  x1_scaled_raw = std::max(0, std::min(x1_scaled_raw, img_to_save.cols - 1));
  y1_scaled_raw = std::max(0, std::min(y1_scaled_raw, img_to_save.rows - 1));
  x2_scaled_raw = std::max(0, std::min(x2_scaled_raw, img_to_save.cols - 1));
  y2_scaled_raw = std::max(0, std::min(y2_scaled_raw, img_to_save.rows - 1));

  if (x2_scaled_raw <= x1_scaled_raw || y2_scaled_raw <= y1_scaled_raw) {
    std::cout
        << "[WARN][SAVE] Scaled box is invalid (width/height <=0) for gopIdx "
        << gopIdx << ", detection " << detection_idx_in_image
        << ". Skipping save for this box." << std::endl;
    return;
  }

  // Draw rectangle for the detection
  cv::rectangle(img_to_save, cv::Point(x1_scaled_raw, y1_scaled_raw),
                cv::Point(x2_scaled_raw, y2_scaled_raw), cv::Scalar(0, 255, 0),
                2); // Green, thickness 2

  // Prepare label text (class name and confidence)
  std::ostringstream label_stream;
  label_stream << class_name << " " << std::fixed << std::setprecision(2)
               << confidence;
  std::string label_text = label_stream.str();

  int font_face = cv::FONT_HERSHEY_SIMPLEX;
  double font_scale = 0.6; // Adjusted for potentially smaller boxes
  int thickness = 1;
  int baseline = 0;
  cv::Size text_size =
      cv::getTextSize(label_text, font_face, font_scale, thickness, &baseline);
  baseline += thickness; // Adjust baseline

  // Position for label background and text (try to put above the box)
  cv::Point text_origin =
      cv::Point(x1_scaled_raw, y1_scaled_raw - text_size.height - 3);
  // If label goes off screen at the top, put it inside the box at the top
  if (text_origin.y < text_size.height) {
    text_origin.y = y1_scaled_raw + text_size.height + 3;
  }
  // Ensure text origin is within image boundaries (simple check for y)
  if (text_origin.y > img_to_save.rows - baseline) {
    text_origin.y =
        y2_scaled_raw - baseline - 3; // Try bottom inside if still problematic
  }
  if (text_origin.y < text_size.height) { // Final check if box is tiny at top
    text_origin.y = y1_scaled_raw + text_size.height + baseline;
  }

  // Draw filled rectangle for text background
  cv::rectangle(img_to_save,
                cv::Point(text_origin.x, text_origin.y - text_size.height -
                                             baseline +
                                             thickness), // Top-left of text bg
                cv::Point(text_origin.x + text_size.width,
                          text_origin.y + baseline -
                              thickness), // Bottom-right of text bg
                cv::Scalar(0, 255, 0),    // Green background
                cv::FILLED);
  // Put text on the background
  cv::putText(img_to_save, label_text, text_origin, font_face, font_scale,
              cv::Scalar(0, 0, 0), thickness); // Black text

  // Construct filename
  std::ostringstream filename_stream;
  filename_stream << image_output_path_ << "/gop" << std::setw(5)
                  << std::setfill('0') << gopIdx << "_obj" << std::setw(3)
                  << std::setfill('0') << detection_idx_in_image << "_"
                  << class_name << "_conf"
                  << static_cast<int>(std::round(confidence * 100)) << ".jpg";
  std::string filename = filename_stream.str();

  try {
    bool success = cv::imwrite(filename, img_to_save);
    if (success) {
      // This print is now mostly covered by processDetectionsForOneImage's
      // console output std::cout << "[SAVE] ✓ Annotated image saved: " <<
      // filename << std::endl;
    } else {
      std::cerr << "[ERROR] ✗ Failed to save annotated image: " << filename
                << " (OpenCV imwrite returned false)" << std::endl;
    }
  } catch (const cv::Exception &ex) {
    std::cerr << "[ERROR] ✗ OpenCV exception while saving image " << filename
              << ": " << ex.what() << std::endl;
  } catch (const std::exception &ex_std) {
    std::cerr << "[ERROR] ✗ Std exception while saving image " << filename
              << ": " << ex_std.what() << std::endl;
  }
}
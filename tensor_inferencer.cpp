#include "tensor_inferencer.hpp" // Assuming this declares all necessary members like bindings_, etc.
#include "inference.hpp" // For InferenceInput, InferenceResult (corrected from .cpp)

#include <algorithm>  // For std::sort, std::max, std::min, std::transform
#include <atomic>     // For std::atomic_bool
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
// processing) This is already defined in the HPP now. struct Detection {
//     float x1, y1, x2, y2;
//     float confidence;
//     int class_id;
// };

// Logger class for TensorRT
class TensorInferencerLogger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
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
    : runtime_(nullptr), engine_(nullptr), context_(nullptr),
      inputDevice_(nullptr), outputDevice_(nullptr), inputIndex_(-1),
      outputIndex_(-1), stop_flag_(false) // std::atomic_bool initialized
{
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

  target_w_ = ((video_width + 31) / 32) * 32;
  target_h_ = ((video_height + 31) / 32) * 32;

  std::string specific_engine_env_key =
      "YOLO_ENGINE_NAME_" + std::to_string(batch_size_);
  const char *specific_engine_path_env =
      std::getenv(specific_engine_env_key.c_str());

  if (specific_engine_path_env) {
    engine_path_ = specific_engine_path_env;
    std::cout << "[INFO] Using engine path from " << specific_engine_env_key
              << ": " << engine_path_ << std::endl;
  } else {
    const char *base_engine_name_env = std::getenv("YOLO_ENGINE_NAME");
    if (base_engine_name_env) {
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

  try {
    if (!image_output_path_.empty() &&
        !std::filesystem::exists(image_output_path_)) {
      if (std::filesystem::create_directories(image_output_path_)) {
        std::cout << "[INFO] Created output directory: " << image_output_path_
                  << std::endl;
      } else {
        std::cerr << "[ERROR] Could not create output directory (failed): "
                  << image_output_path_ << std::endl;
      }
    }
  } catch (const std::filesystem::filesystem_error &e) {
    std::cerr << "[ERROR] Filesystem error while creating output directory "
              << image_output_path_ << ": " << e.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }

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

  bindings_.resize(engine_->getNbBindings());

  inputIndex_ = engine_->getBindingIndex("images");
  outputIndex_ = engine_->getBindingIndex("output0");

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

  Dims reportedInputDims = engine_->getBindingDimensions(inputIndex_);
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

  printEngineInfo();

  Dims inputDimsFullBatch{4, {batch_size_, 3, target_h_, target_w_}};
  if (!context_->setBindingDimensions(inputIndex_, inputDimsFullBatch)) {
    std::cerr << "[ERROR] Constructor: Failed to set binding dimensions for "
                 "input tensor to determine buffer sizes."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!context_->allInputDimensionsSpecified()) {
    std::cerr << "[ERROR] Constructor: Not all input dimensions specified "
                 "after setting for full batch."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Dims outputDimsFullBatch = context_->getBindingDimensions(outputIndex_);
  if (outputDimsFullBatch.nbDims != 3 ||
      outputDimsFullBatch.d[0] != batch_size_ ||
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
    std::exit(EXIT_FAILURE);
  }

  inputSizeElementsPerImage_ = static_cast<size_t>(3) * target_h_ * target_w_;
  size_t totalInputElementsForBatch =
      static_cast<size_t>(batch_size_) * inputSizeElementsPerImage_;

  outputSizeElementsPerImage_ =
      static_cast<size_t>(outputDimsFullBatch.d[1]) * outputDimsFullBatch.d[2];
  size_t totalOutputElementsForBatch =
      static_cast<size_t>(batch_size_) * outputSizeElementsPerImage_;

  inputSize_ = totalInputElementsForBatch * sizeof(float);
  outputSize_ = totalOutputElementsForBatch * sizeof(float);

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
    cudaFree(inputDevice_);
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

  inference_thread_ = std::thread(&TensorInferencer::run, this);
  std::cout << "[INFO] TensorInferencer initialized and worker thread started."
            << std::endl;
}

TensorInferencer::~TensorInferencer() {
  std::cout << "[DEINIT] Destructor called. Finalizing processing..."
            << std::endl;
  finalizeProcessing(
      true); // Ensure all tasks are flushed and thread is joined.
  // Resources are freed after thread joins.
  // Note: finalizeProcessing already joins the thread, so the join here will be
  // a no-op if called prior. If finalizeProcessing was not called, this ensures
  // cleanup.

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

void TensorInferencer::finalizeProcessing(bool wait_for_completion) {
  std::cout << "[FINALIZE] Initiating finalizeProcessing. stop_flag current: "
            << stop_flag_.load() << std::endl;
  bool already_stopped =
      stop_flag_.exchange(true); // Set to true and get previous value

  if (already_stopped) {
    std::cout << "[FINALIZE] Already stopping/stopped." << std::endl;
    // If already stopping, and we need to wait, ensure thread is joined.
    // The destructor will also call this, so it needs to be idempotent.
    if (wait_for_completion && inference_thread_.joinable()) {
      // It's possible the thread is joinable but already joined by another
      // finalize/destructor call. A robust way is to track if joined, or rely
      // on destructor's final join. For simplicity, if called multiple times,
      // subsequent joins on an already joined thread are benign. However,
      // multiple threads calling join on the same std::thread is UB. This
      // method should ideally be called once, or be idempotent regarding
      // joining. The destructor will handle the final join. If we want this
      // method to *guarantee* join before returning, we do it here.
    }
    // If already stopped, we still notify to ensure any cv_.wait() condition is
    // re-evaluated if it was missed.
    cv_.notify_all();
    if (wait_for_completion && inference_thread_.joinable()) {
      std::cout << "[FINALIZE] Waiting for inference thread to complete "
                   "(already stopping)..."
                << std::endl;
      inference_thread_
          .join(); // This might be problematic if another thread (e.g.
                   // destructor) also tries to join. Better to have a single
                   // point of joining or a flag. For now, assuming destructor
                   // handles the ultimate join if this one doesn't. Let's make
                   // this method the primary joiner if wait_for_completion is
                   // true.
    }
    return; // Already initiated stop
  }

  std::cout << "[FINALIZE] Setting stop_flag to true and notifying worker."
            << std::endl;
  // stop_flag_ is already set by exchange above.
  // Lock is not strictly needed for stop_flag_ (atomic) but good for cv.
  // {
  //     std::unique_lock<std::mutex> lock(mutex_); // Not needed for atomic
  //     stop_flag write
  //     // stop_flag_ = true; // Done by exchange
  // }
  cv_.notify_all();

  if (wait_for_completion) {
    if (inference_thread_.joinable()) {
      std::cout << "[FINALIZE] Waiting for inference thread to complete..."
                << std::endl;
      inference_thread_.join();
      std::cout << "[FINALIZE] Inference thread joined." << std::endl;
    } else {
      std::cout << "[FINALIZE] Inference thread was not joinable (already "
                   "joined or not started properly)."
                << std::endl;
    }
  } else {
    std::cout << "[FINALIZE] Signal sent, not waiting for completion."
              << std::endl;
  }
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
    std::cout << "  Binding " << i << ": Name='" << name
              << "', Role=" << (isInput ? "Input" : "Output")
              << ", DataType=" << dtype_str << ", Dims (Opt Profile): [";
    for (int j = 0; j < dims.nbDims; ++j) {
      std::cout << dims.d[j] << (j == dims.nbDims - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
  }
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

bool TensorInferencer::infer(const InferenceInput &input,
                             InferenceCallback callback) {
  if (stop_flag_.load()) { // Use .load() for std::atomic_bool
    std::cerr << "[WARN] Inferencer is stopping/stopped. Request for gopIdx "
              << input.gopIdx << " rejected." << std::endl;
    if (callback) {
      std::vector<InferenceResult> results;
      results.push_back({"Inferencer stopping, request rejected."});
      try {
        callback(results);
      } catch (const std::exception &e) {
        std::cerr
            << "[ERROR] Exception in user-provided callback (rejected input): "
            << e.what() << std::endl;
      } catch (...) {
        std::cerr << "[ERROR] Unknown exception in user-provided callback "
                     "(rejected input)."
                  << std::endl;
      }
    }
    return false; // Indicate rejection
  }
  std::unique_lock<std::mutex> lock(mutex_);
  pending_inputs_.push_back(input);
  pending_callback_ = callback;
  lock.unlock();
  cv_.notify_one();
  return true; // Indicate success
}

bool TensorInferencer::infer(const std::vector<float> &input_vector_cpu,
                             std::vector<float> &output_vector_cpu) {
  if (stop_flag_.load()) {
    std::cerr << "[WARN] infer(vector<float>): Inferencer is stopping/stopped. "
                 "Request rejected."
              << std::endl;
    return false;
  }
  std::cerr << "[WARN] The infer(std::vector<float>...) overload is "
               "potentially unsafe with batching due to shared GPU buffers and "
               "size assumptions. Use with extreme caution or refactor."
            << std::endl;

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
  output_vector_cpu.resize(outputSizeElementsPerImage_);

  cudaError_t err;
  // This section needs a mutex if context dimensions are changed and restored,
  // to prevent race conditions with the main batching thread.
  // For simplicity, and given it's a "problematic" function, skipping full
  // thread-safety for this overload here. A proper solution would involve a
  // separate context or careful synchronization.
  std::unique_lock<std::mutex> lock(mutex_); // Lock to protect context changes

  Dims singleImageInputDims{4, {1, 3, target_h_, target_w_}};
  Dims originalInputDims = context_->getBindingDimensions(
      inputIndex_); // Get current (likely batch) dims

  bool dims_changed = false;
  if (originalInputDims.d[0] != 1 || originalInputDims.d[2] != target_h_ ||
      originalInputDims.d[3] != target_w_) {
    if (!context_->setBindingDimensions(inputIndex_, singleImageInputDims)) {
      std::cerr << "[ERROR] infer(vector<float>): Failed to set binding "
                   "dimensions for single image."
                << std::endl;
      return false; // lock will be released by destructor
    }
    dims_changed = true;
  }
  lock.unlock(); // Unlock before potentially long CUDA calls

  err = cudaMemcpy(bindings_[inputIndex_], input_vector_cpu.data(),
                   inputSizeElementsPerImage_ * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] infer(vector<float>): CUDA Memcpy H2D: "
              << cudaGetErrorString(err) << std::endl;
    if (dims_changed) { // Restore original dimensions if changed
      std::unique_lock<std::mutex> restore_lock(mutex_);
      context_->setBindingDimensions(inputIndex_, originalInputDims);
    }
    return false;
  }

  if (!context_->enqueueV2(bindings_.data(), 0, nullptr)) {
    std::cerr << "[ERROR] infer(vector<float>): enqueueV2 failed." << std::endl;
    if (dims_changed) {
      std::unique_lock<std::mutex> restore_lock(mutex_);
      context_->setBindingDimensions(inputIndex_, originalInputDims);
    }
    return false;
  }

  err = cudaMemcpy(output_vector_cpu.data(), bindings_[outputIndex_],
                   outputSizeElementsPerImage_ * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] infer(vector<float>): CUDA Memcpy D2H: "
              << cudaGetErrorString(err) << std::endl;
    if (dims_changed) {
      std::unique_lock<std::mutex> restore_lock(mutex_);
      context_->setBindingDimensions(inputIndex_, originalInputDims);
    }
    return false;
  }

  if (dims_changed) {
    std::unique_lock<std::mutex> restore_lock(mutex_);
    if (!context_->setBindingDimensions(inputIndex_, originalInputDims)) {
      std::cerr << "[WARN] infer(vector<float>): Failed to restore original "
                   "batch binding dimensions."
                << std::endl;
    }
  }
  return true;
}

void TensorInferencer::run() {
  std::vector<InferenceInput> current_processing_batch;
  InferenceCallback callback_for_this_batch = nullptr;

  std::cout << "[WORKER] Inference worker thread started." << std::endl;
  while (true) {
    current_processing_batch.clear();
    callback_for_this_batch = nullptr;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] {
        return stop_flag_.load() ||
               pending_inputs_.size() >= static_cast<size_t>(batch_size_);
      });

      if (stop_flag_.load() && pending_inputs_.empty()) {
        std::cout
            << "[WORKER] Stop flag set and no pending inputs. Exiting run loop."
            << std::endl;
        break;
      }

      size_t num_to_grab = 0;
      if (pending_inputs_.size() >= static_cast<size_t>(batch_size_)) {
        num_to_grab = batch_size_;
      } else if (stop_flag_.load() && !pending_inputs_.empty()) {
        num_to_grab = pending_inputs_.size();
        std::cout << "[WORKER] Stop flag set. Processing remaining "
                  << num_to_grab << " inputs." << std::endl;
      } else {
        continue;
      }

      if (num_to_grab > 0) {
        current_processing_batch.reserve(num_to_grab);
        for (size_t i = 0; i < num_to_grab; ++i) {
          current_processing_batch.push_back(
              std::move(pending_inputs_.front()));
          pending_inputs_.erase(pending_inputs_.begin());
        }
        callback_for_this_batch = pending_callback_;
      }
    }

    if (!current_processing_batch.empty()) {
      std::cout << "[WORKER] Processing a batch of "
                << current_processing_batch.size() << " items." << std::endl;
      std::vector<InferenceResult> batch_results;
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
      } else if (!batch_results.empty()) {
        std::cout
            << "[WORKER WARN] Batch processed but no callback was set for it."
            << std::endl;
      }
    }
  }
  std::cout << "[WORKER] Inference worker thread run() loop finished."
            << std::endl;
}

void TensorInferencer::performInference(
    const std::vector<InferenceInput> &batch_inputs_from_queue,
    std::vector<InferenceResult> &batch_results_out) {
  if (batch_inputs_from_queue.empty()) {
    return;
  }

  int num_actual_images_in_this_batch = batch_inputs_from_queue.size();
  std::vector<float> batched_input_data_cpu;
  batched_input_data_cpu.reserve(static_cast<size_t>(batch_size_) *
                                 inputSizeElementsPerImage_);

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
        img_to_preprocess =
            cv::Mat(target_h_, target_w_, CV_8UC3, cv::Scalar(114, 114, 114));
      } else {
        img_to_preprocess = current_item.decoded_frames[0];
      }
    } else {
      img_to_preprocess =
          cv::Mat(target_h_, target_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    }

    cv::Mat resized_img;
    if (img_to_preprocess.cols != target_w_ ||
        img_to_preprocess.rows != target_h_) {
      cv::resize(img_to_preprocess, resized_img,
                 cv::Size(target_w_, target_h_));
    } else {
      resized_img = img_to_preprocess;
    }

    cv::Mat img_rgb;
    cv::cvtColor(resized_img, img_rgb, cv::COLOR_BGR2RGB);

    cv::Mat chw_input_fp32_single;
    img_rgb.convertTo(chw_input_fp32_single, CV_32FC3, 1.0 / 255.0);

    for (int ch = 0; ch < 3; ++ch) {
      for (int y = 0; y < target_h_; ++y) {
        for (int x = 0; x < target_w_; ++x) {
          batched_input_data_cpu.push_back(
              chw_input_fp32_single.at<cv::Vec3f>(y, x)[ch]);
        }
      }
    }
  }

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

  cudaError_t err;
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

  if (!context_->enqueueV2(bindings_.data(), 0, nullptr)) {
    std::cerr << "[ERROR] PerformInference: TensorRT enqueueV2 failed."
              << std::endl;
    for (int i = 0; i < num_actual_images_in_this_batch; ++i)
      batch_results_out.push_back(
          {"Error: TensorRT inference execution failed"});
    return;
  }

  size_t total_output_bytes_for_batch = static_cast<size_t>(batch_size_) *
                                        outputSizeElementsPerImage_ *
                                        sizeof(float);
  std::vector<float> host_output_raw_full_batch(
      static_cast<size_t>(batch_size_) * outputSizeElementsPerImage_);

  err = cudaMemcpy(host_output_raw_full_batch.data(), outputDevice_,
                   total_output_bytes_for_batch, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "[ERROR] PerformInference: CUDA Memcpy D2H failed: "
              << cudaGetErrorString(err) << std::endl;
    for (int i = 0; i < num_actual_images_in_this_batch; ++i)
      batch_results_out.push_back({"Error: CUDA D2H copy failed"});
    return;
  }

  Dims batchOutDimsFromContext = context_->getBindingDimensions(outputIndex_);

  for (int i = 0; i < num_actual_images_in_this_batch; ++i) {
    const InferenceInput &current_original_input_params =
        batch_inputs_from_queue[i];
    const cv::Mat &current_raw_img_for_item =
        (current_original_input_params.decoded_frames.empty() ||
         current_original_input_params.decoded_frames[0].empty())
            ? cv::Mat()
            : current_original_input_params.decoded_frames[0];

    const float *single_image_output_ptr_start =
        host_output_raw_full_batch.data() + (i * outputSizeElementsPerImage_);
    std::vector<float> single_image_output_data_slice(
        single_image_output_ptr_start,
        single_image_output_ptr_start + outputSizeElementsPerImage_);

    InferenceResult current_image_result_obj;
    processDetectionsForOneImage(
        current_original_input_params, single_image_output_data_slice,
        current_raw_img_for_item, batchOutDimsFromContext, i,
        current_image_result_obj);
    batch_results_out.push_back(current_image_result_obj);
  }
}

void TensorInferencer::processDetectionsForOneImage(
    const InferenceInput &original_input_params,
    const std::vector<float> &single_image_raw_output,
    const cv::Mat &raw_img_for_this_item, const nvinfer1::Dims &batchOutDims,
    int image_idx_in_batch, InferenceResult &result_for_this_image_out) {
  if (raw_img_for_this_item.empty()) {
    std::cout << "[INFO] processDetections: Skipping processing for gopIdx "
              << original_input_params.gopIdx << " (batch slot "
              << image_idx_in_batch << ") as raw image was empty/invalid."
              << std::endl;
    result_for_this_image_out.info = "Skipped processing: Invalid raw image.";
    return;
  }

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
  float confidence_threshold =
      std::max(0.01f, original_input_params.confidence_thresh);

  std::vector<Detection> candidate_detections;
  for (int i = 0; i < num_potential_detections_per_image; ++i) {
    const float *det_attrs_ptr =
        &transposed_output_single_image[static_cast<size_t>(i) *
                                        num_attributes_per_detection];

    float max_class_score = 0.0f;
    int best_class_id_for_this_box = -1;
    for (int j = 0; j < num_classes_; ++j) {
      float score = det_attrs_ptr[4 + j];
      if (score > max_class_score) {
        max_class_score = score;
        best_class_id_for_this_box = j;
      }
    }

    if (best_class_id_for_this_box == target_class_id &&
        max_class_score >= confidence_threshold) {
      float cx_model = det_attrs_ptr[0];
      float cy_model = det_attrs_ptr[1];
      float w_model = det_attrs_ptr[2];
      float h_model = det_attrs_ptr[3];

      float x1_model = std::max(0.0f, cx_model - w_model / 2.0f);
      float y1_model = std::max(0.0f, cy_model - h_model / 2.0f);
      float x2_model =
          std::min(static_cast<float>(target_w_), cx_model + w_model / 2.0f);
      float y2_model =
          std::min(static_cast<float>(target_h_), cy_model + h_model / 2.0f);

      if (x2_model > x1_model && y2_model > y1_model) {
        candidate_detections.push_back({x1_model, y1_model, x2_model, y2_model,
                                        max_class_score,
                                        best_class_id_for_this_box});
      }
    }
  }

  std::vector<Detection> nms_filtered_detections =
      applyNMS(candidate_detections, 0.45f);

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

    saveAnnotatedImage(raw_img_for_this_item, det.x1, det.y1, det.x2, det.y2,
                       det.confidence, original_input_params.object_name,
                       original_input_params.gopIdx, static_cast<int>(i));

    float scale_x_to_raw = static_cast<float>(raw_img_for_this_item.cols) /
                           static_cast<float>(target_w_);
    float scale_y_to_raw = static_cast<float>(raw_img_for_this_item.rows) /
                           static_cast<float>(target_h_);

    int final_x1_raw = static_cast<int>(std::round(det.x1 * scale_x_to_raw));
    int final_y1_raw = static_cast<int>(std::round(det.y1 * scale_y_to_raw));
    int final_x2_raw = static_cast<int>(std::round(det.x2 * scale_x_to_raw));
    int final_y2_raw = static_cast<int>(std::round(det.y2 * scale_y_to_raw));

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
        detection_summary_stream << "; ";
      detection_summary_stream << "Det" << i << "_RawBox: [" << final_x1_raw
                               << "," << final_y1_raw << "," << final_w_raw
                               << "," << final_h_raw << "]@" << std::fixed
                               << std::setprecision(2) << det.confidence;
    }
  }
  result_for_this_image_out.info = detection_summary_stream.str();
}

float TensorInferencer::calculateIoU(const Detection &a, const Detection &b) {
  float x1_intersect = std::max(a.x1, b.x1);
  float y1_intersect = std::max(a.y1, b.y1);
  float x2_intersect = std::min(a.x2, b.x2);
  float y2_intersect = std::min(a.y2, b.y2);

  float intersection_width = std::max(0.0f, x2_intersect - x1_intersect);
  float intersection_height = std::max(0.0f, y2_intersect - y1_intersect);
  float intersection_area = intersection_width * intersection_height;

  if (intersection_area <=
      1e-5) // Use a small epsilon for floating point comparison
    return 0.0f;

  float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
  float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
  float union_area = area_a + area_b - intersection_area;

  return (union_area > 1e-5) ? (intersection_area / union_area) : 0.0f;
}

std::vector<Detection>
TensorInferencer::applyNMS(const std::vector<Detection> &detections,
                           float iou_threshold) {
  if (detections.empty()) {
    return {};
  }
  std::vector<Detection> sorted_detections = detections;
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

void TensorInferencer::saveAnnotatedImage(
    const cv::Mat &raw_img_param, float x1_model, float y1_model,
    float x2_model, float y2_model, float confidence,
    const std::string &class_name, int gopIdx, int detection_idx_in_image) {
  if (raw_img_param.empty()) {
    std::cerr << "[WARN][SAVE] Raw image is empty for gopIdx " << gopIdx
              << ", detection " << detection_idx_in_image << ". Cannot save."
              << std::endl;
    return;
  }
  cv::Mat img_to_save = raw_img_param.clone();

  float scale_x_to_raw =
      static_cast<float>(img_to_save.cols) / static_cast<float>(target_w_);
  float scale_y_to_raw =
      static_cast<float>(img_to_save.rows) / static_cast<float>(target_h_);

  int x1_scaled_raw = static_cast<int>(std::round(x1_model * scale_x_to_raw));
  int y1_scaled_raw = static_cast<int>(std::round(y1_model * scale_y_to_raw));
  int x2_scaled_raw = static_cast<int>(std::round(x2_model * scale_x_to_raw));
  int y2_scaled_raw = static_cast<int>(std::round(y2_model * scale_y_to_raw));

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

  cv::rectangle(img_to_save, cv::Point(x1_scaled_raw, y1_scaled_raw),
                cv::Point(x2_scaled_raw, y2_scaled_raw), cv::Scalar(0, 255, 0),
                2);

  std::ostringstream label_stream;
  label_stream << class_name << " " << std::fixed << std::setprecision(2)
               << confidence;
  std::string label_text = label_stream.str();

  int font_face = cv::FONT_HERSHEY_SIMPLEX;
  double font_scale = 0.6;
  int thickness = 1;
  int baseline = 0;
  cv::Size text_size =
      cv::getTextSize(label_text, font_face, font_scale, thickness, &baseline);
  baseline += thickness;

  cv::Point text_origin =
      cv::Point(x1_scaled_raw, y1_scaled_raw - text_size.height - 3);
  if (text_origin.y < text_size.height) {
    text_origin.y = y1_scaled_raw + text_size.height + 3;
  }
  if (text_origin.y > img_to_save.rows - baseline) {
    text_origin.y = y2_scaled_raw - baseline - 3;
  }
  if (text_origin.y < text_size.height) {
    text_origin.y = y1_scaled_raw + text_size.height + baseline;
  }

  cv::rectangle(img_to_save,
                cv::Point(text_origin.x, text_origin.y - text_size.height -
                                             baseline + thickness),
                cv::Point(text_origin.x + text_size.width,
                          text_origin.y + baseline - thickness),
                cv::Scalar(0, 255, 0), cv::FILLED);
  cv::putText(img_to_save, label_text, text_origin, font_face, font_scale,
              cv::Scalar(0, 0, 0), thickness);

  std::ostringstream filename_stream;
  filename_stream << image_output_path_ << "/gop" << std::setw(5)
                  << std::setfill('0') << gopIdx << "_obj" << std::setw(3)
                  << std::setfill('0') << detection_idx_in_image << "_"
                  << class_name << "_conf"
                  << static_cast<int>(std::round(confidence * 100)) << ".jpg";
  std::string filename = filename_stream.str();

  try {
    bool success = cv::imwrite(filename, img_to_save);
    if (!success) {
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
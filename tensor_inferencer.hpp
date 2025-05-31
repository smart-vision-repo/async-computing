#pragma once

#include <condition_variable> // For std::condition_variable
#include <functional>         // For std::function
#include <map>                // For std::map
#include <mutex>              // For std::mutex
#include <string>
#include <thread> // For std::thread
#include <vector>

// Forward declare OpenCV Mat
namespace cv {
class Mat;
}

// Forward declare TensorRT types
namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
struct Dims; // Used in private method signatures
} // namespace nvinfer1

// Include for InferenceInput and InferenceResult structures
#include "inference.hpp" // Make sure this path is correct

// Define InferenceCallback type alias
using InferenceCallback =
    std::function<void(const std::vector<InferenceResult> &)>;

// Definition for the Detection struct used in processing
// This needs to be defined here if private methods use it in their signature.
struct Detection {
  float x1, y1, x2, y2;
  float confidence;
  int class_id;
};

class TensorInferencer {
public:
  // Constructor now takes video dimensions for initial target_w, target_h
  // calculation
  TensorInferencer(int video_height, int video_width);
  ~TensorInferencer();

  // Main public API for submitting inference requests (batch processing)
  void infer(const InferenceInput &input, InferenceCallback callback);

  // Original infer method (kept as per .cpp, but noted as potentially unsafe
  // with batching)
  bool infer(const std::vector<float> &input_vector_cpu,
             std::vector<float> &output_vector_cpu);

private:
  // Worker thread's main function
  void run();

  // Performs inference on a prepared batch
  void
  performInference(const std::vector<InferenceInput> &batch_inputs_from_queue,
                   std::vector<InferenceResult> &batch_results_out);

  // Processes detections for a single image from the batch output
  void processDetectionsForOneImage(
      const InferenceInput &original_input_params,
      const std::vector<float> &single_image_raw_output,
      const cv::Mat &raw_img_for_this_item,
      const nvinfer1::Dims &batchOutDims, // TensorRT Dims type
      int image_idx_in_batch, InferenceResult &result_for_this_image_out);

  // Saves image with annotations
  void saveAnnotatedImage(const cv::Mat &raw_img_param, float x1_model,
                          float y1_model, float x2_model, float y2_model,
                          float confidence, const std::string &class_name,
                          int gopIdx, int detection_idx_in_image);

  // NMS helper functions
  float calculateIoU(const Detection &a, const Detection &b);
  std::vector<Detection> applyNMS(const std::vector<Detection> &detections,
                                  float iou_threshold);

  // Utility to print engine information
  void printEngineInfo();

  // --- Member Variables ---

  // Threading and synchronization
  std::vector<InferenceInput> pending_inputs_;
  InferenceCallback pending_callback_; // Callback for the current batch being
                                       // formed/processed
  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread inference_thread_;
  bool stop_flag_ = false;

  // TensorRT core components
  nvinfer1::IRuntime *runtime_;
  nvinfer1::ICudaEngine *engine_;
  nvinfer1::IExecutionContext *context_;
  std::vector<void *> bindings_; // Stores pointers to GPU input/output buffers

  // GPU device pointers for batched input and output
  void *inputDevice_;
  void *outputDevice_;

  // Tensor binding indices
  int inputIndex_;
  int outputIndex_;

  // Configuration and model parameters
  int batch_size_ = 1;            // Actual batch size for inference
  int target_w_ = 0;              // Model input width
  int target_h_ = 0;              // Model input height
  int num_classes_ = 0;           // Number of classes the model predicts
  std::string engine_path_;       // Path to the TensorRT engine file
  std::string image_output_path_; // Path to save annotated images

  // Buffer size information
  // Total size in BYTES for the entire batch on GPU
  size_t inputSize_ = 0;
  size_t outputSize_ = 0;
  // Size in ELEMENTS (number of floats) for a SINGLE image
  size_t inputSizeElementsPerImage_ = 0;
  size_t outputSizeElementsPerImage_ = 0;

  // Class name mappings
  std::map<std::string, int> class_name_to_id_;
  std::map<int, std::string> id_to_class_name_;
};
#pragma once

#include <vector>
#include <string>
#include <functional> // For std::function
#include <mutex>      // For std::mutex
#include <condition_variable> // For std::condition_variable
#include <thread>     // For std::thread
#include <map>        // For std::map
#include <atomic>     // For std::atomic_bool

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
}

// Include for InferenceInput and InferenceResult structures
#include "inference.hpp" // Make sure this path is correct

// Define InferenceCallback type alias
using InferenceCallback = std::function<void(const std::vector<InferenceResult>&)>;

// Definition for the Detection struct used in processing
struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

class TensorInferencer {
public:
    TensorInferencer(int video_height, int video_width);
    ~TensorInferencer();

    // Main public API for submitting inference requests (batch processing)
    // Returns false if inferencer is stopping/flushing and cannot accept new input.
    bool infer(const InferenceInput& input, InferenceCallback callback);

    // Original infer method (kept as per .cpp, but noted as potentially unsafe with batching)
    // Returns false if inferencer is stopping/flushing.
    bool infer(const std::vector<float> &input_vector_cpu, std::vector<float> &output_vector_cpu);

    // Signals the inferencer to stop accepting new inputs and process all pending items.
    // Blocks until all pending items are processed and the worker thread has finished if wait_for_completion is true.
    void finalizeProcessing(bool wait_for_completion = true);

private:
    void run();
    void performInference(const std::vector<InferenceInput>& batch_inputs_from_queue,
                          std::vector<InferenceResult>& batch_results_out);
    void processDetectionsForOneImage(
        const InferenceInput &original_input_params,
        const std::vector<float> &single_image_raw_output,
        const cv::Mat &raw_img_for_this_item,
        const nvinfer1::Dims &batchOutDims,
        int image_idx_in_batch,
        InferenceResult& result_for_this_image_out);
    void saveAnnotatedImage(
        const cv::Mat &raw_img_param,
        float x1_model, float y1_model, float x2_model, float y2_model,
        float confidence,
        const std::string &class_name,
        int gopIdx,
        int detection_idx_in_image);
    float calculateIoU(const Detection &a, const Detection &b);
    std::vector<Detection> applyNMS(const std::vector<Detection> &detections, float iou_threshold);
    void printEngineInfo();

    std::vector<InferenceInput> pending_inputs_;
    InferenceCallback pending_callback_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread inference_thread_;
    std::atomic<bool> stop_flag_{false}; // Use std::atomic for thread-safe reads in infer()

    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    std::vector<void*> bindings_;

    void* inputDevice_;
    void* outputDevice_;

    int inputIndex_;
    int outputIndex_;

    int batch_size_ = 1;
    int target_w_ = 0;
    int target_h_ = 0;
    int num_classes_ = 0;
    std::string engine_path_;
    std::string image_output_path_;

    size_t inputSize_ = 0; 
    size_t outputSize_ = 0;
    size_t inputSizeElementsPerImage_ = 0;
    size_t outputSizeElementsPerImage_ = 0;

    std::map<std::string, int> class_name_to_id_;
    std::map<int, std::string> id_to_class_name_;
};
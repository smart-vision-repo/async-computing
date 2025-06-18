// tensor_inferencer.hpp (Updated)
#ifndef TENSOR_INFERENCER_HPP
#define TENSOR_INFERENCER_HPP

#include <string>
#include <vector>
#include <memory> // For std::unique_ptr
#include <mutex>  // For std::mutex
#include <functional> // For std::function

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp> // For GPU operations with OpenCV

// Contains shared struct definitions like Detection, InferenceResult, BatchImageMetadata
#include "models.hpp"
// Includes ObjectTracker class definition. This now also brings in the callback definitions.
#include "object_tracker.hpp"

// Forward declaration of the logger
class TrtLogger;

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
} // namespace nvinfer1

/**
 * @brief Callback function definitions.
 * Moved here from object_tracker.hpp to define them at a more global scope for TensorInferencer.
 */
// Callback for reporting inference results (now triggered by ObjectTracker)
using InferResultCallback = std::function<void(const InferenceResult&)>;
// Callback for saving annotated images (now triggered by ObjectTracker)
using ImageSaveCallback = std::function<void(const Detection&, const BatchImageMetadata&, int)>;
// Callback for reporting inference batch completion info
using InferPackCallback = std::function<void(const int&)>;

/**
 * @brief TensorInferencer class, responsible for TensorRT model inference.
 */
class TensorInferencer {
public:
    /**
     * @brief Constructor.
     * @param task_id Task ID.
     * @param video_height Video height.
     * @param video_width Video width.
     * @param object_name Name of the object to detect.
     * @param interval Frame sampling interval.
     * @param confidence Detection confidence threshold.
     * @param resultCallback Inference result callback function.
     * @param packCallback Inference batch completion callback function.
     */
    TensorInferencer(int task_id, int video_height, int video_width,
                     std::string object_name, int interval, float confidence,
                     InferResultCallback resultCallback, InferPackCallback packCallback);

    // Destructor
    ~TensorInferencer();

    /**
     * @brief Performs inference.
     * @param input Input struct containing decoded frames and latest frame index.
     * @return True if successful, false otherwise.
     */
    bool infer(const InferenceInput &input);

    /**
     * @brief Finalizes the inference process, processing any remaining frames.
     */
    void finalizeInference();

private:
    // Task-related members
    int task_id_;
    std::string object_name_;
    int interval_; // This is the delta_t for Kalman Filter
    float confidence_;
    std::string engine_path_;
    std::string image_output_path_;

    // TensorRT-related members
    int BATCH_SIZE_;
    int target_w_;
    int target_h_;
    nvinfer1::IRuntime *runtime_;
    nvinfer1::ICudaEngine *engine_;
    nvinfer1::IExecutionContext *context_;
    int inputIndex_;
    int outputIndex_;
    void *inputDevice_;
    void *outputDevice_;
    std::vector<void *> bindings_; // TensorRT binding pointers

    // Class information
    int num_classes_;
    std::map<std::string, int> class_name_to_id_;
    std::map<int, std::string> id_to_class_name_;

    // Callback functions
    InferResultCallback result_callback_;
    InferPackCallback pack_callback_;

    // Batch processing members
    std::vector<cv::Mat> current_batch_raw_frames_;
    // BatchImageMetadata is now defined in models.hpp
    std::vector<BatchImageMetadata> current_batch_metadata_;
    std::mutex batch_mutex_; // Mutex to protect batch processing queue

    // Cached geometry information to avoid recalculation
    struct CachedGeometry {
        int original_w = 0;
        int original_h = 0;
        float scale_to_model = 0.0f;
        int pad_w_left = 0;
        int pad_h_top = 0;
    } cached_geometry_;
    bool constant_metadata_initialized_;

    // Added: ObjectTracker instance
    std::unique_ptr<ObjectTracker> object_tracker_;

    // Private helper methods
    std::vector<char> readEngineFile(const std::string &enginePath);
    int roundToNearestMultiple(int val, int base);
    void printEngineInfo();

    /**
     * @brief Preprocesses a single image to fit the model input.
     * @param cpu_img Original image on CPU.
     * @param meta Image metadata.
     * @param model_input_w Model input width.
     * @param model_input_h Model input height.
     * @param chw_planar_output_gpu_buffer_slice CHW planar output buffer slice on GPU.
     */
    void preprocess_single_image_for_batch(const cv::Mat &cpu_img,
                                           BatchImageMetadata &meta,
                                           int model_input_w,
                                           int model_input_h,
                                           cv::cuda::GpuMat &chw_planar_output_gpu_buffer_slice);

    /**
     * @brief Performs batch inference.
     * @param pad_batch Whether to pad batches that are smaller than BatchSize.
     */
    void performBatchInference(bool pad_batch);

    /**
     * @brief Processes the inference output for a single image (now passes results to ObjectTracker).
     * @param image_meta Image metadata.
     * @param host_output_for_image_raw Raw model output data.
     * @param num_detections_in_slice Number of detections for this image.
     * @param num_attributes_per_detection Number of attributes per detection.
     * @param original_batch_idx_for_debug Original batch index for debugging.
     */
    void process_single_output(const BatchImageMetadata &image_meta,
                               const float *host_output_for_image_raw,
                               int num_detections_in_slice,
                               int num_attributes_per_detection,
                               int original_batch_idx_for_debug); // No longer needs frame_results param

    /**
     * @brief Calculates IoU between two detection boxes.
     * @param a First detection box.
     * @param b Second detection box.
     * @return IoU value.
     */
    // Detection struct is now defined in models.hpp
    float calculateIoU(const Detection &a, const Detection &b);

    /**
     * @brief Applies Non-Maximum Suppression (NMS).
     * @param detections Original list of detections.
     * @param iou_threshold IoU threshold.
     * @return List of detections after NMS.
     */
    // Detection struct is now defined in models.hpp
    std::vector<Detection> applyNMS(const std::vector<Detection> &detections,
                                    float iou_threshold);

    /**
     * @brief Saves an annotated image with bounding boxes.
     * @param det Detection result.
     * @param image_meta Image metadata.
     * @param detection_idx_in_image Index of the detection in the image (used here for track_id primarily)
     */
    // Detection and BatchImageMetadata structs are now defined in models.hpp
    void saveAnnotatedImage(const Detection &det,
                            const BatchImageMetadata &image_meta,
                            int detection_idx_in_image);
};

#endif // TENSOR_INFERENCER_HPP
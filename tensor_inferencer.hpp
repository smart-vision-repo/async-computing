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

// 包含 models.hpp，其中现在包含了 Detection, InferenceResult, BatchImageMetadata 等结构体
#include "models.hpp"
// 包含 object_tracker.hpp，用于引入 ObjectTracker 类定义
#include "object_tracker.hpp"

// Forward declaration of the logger
class TrtLogger;

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
} // namespace nvinfer1

/**
 * @brief 推理输入数据结构。
 * (此结构体现在已移至 models.hpp，此处注释仅为兼容性考虑)
 */
// struct InferenceInput {
//     std::vector<cv::Mat> decoded_frames; // 批量解码的帧
//     int latest_frame_index;              // 当前批次中最新帧的全局索引
// };

/**
 * @brief 回调函数定义。
 */
// 用于报告推理结果的回调函数（现在由ObjectTracker触发）
// InferenceResult 现在定义在 models.hpp 中
using InferResultCallback = std::function<void(const InferenceResult&)>;
// 用于报告推理批次完成信息的回调函数
using InferPackCallback = std::function<void(const int&)>;

/**
 * @brief TensorInferencer 类，负责 TensorRT 模型的推理。
 */
class TensorInferencer {
public:
    /**
     * @brief 构造函数。
     * @param task_id 任务 ID。
     * @param video_height 视频高度。
     * @param video_width 视频宽度。
     * @param object_name 要检测的目标对象名称。
     * @param interval 帧采样间隔。
     * @param confidence 检测置信度阈值。
     * @param resultCallback 推理结果回调函数。
     * @param packCallback 推理批次完成回调函数。
     */
    TensorInferencer(int task_id, int video_height, int video_width,
                     std::string object_name, int interval, float confidence,
                     InferResultCallback resultCallback, InferPackCallback packCallback);

    // 析构函数
    ~TensorInferencer();

    /**
     * @brief 执行推理。
     * @param input 包含解码帧和最新帧索引的输入结构。
     * @return 如果成功则返回 true，否则返回 false。
     */
    bool infer(const InferenceInput &input);

    /**
     * @brief 结束推理过程，处理剩余的帧。
     */
    void finalizeInference();

private:
    // 任务相关成员
    int task_id_;
    std::string object_name_;
    int interval_;
    float confidence_;
    std::string engine_path_;
    std::string image_output_path_;

    // TensorRT 相关成员
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
    std::vector<void *> bindings_; // TensorRT 绑定指针

    // 类别信息
    int num_classes_;
    std::map<std::string, int> class_name_to_id_;
    std::map<int, std::string> id_to_class_name_;

    // 回调函数
    InferResultCallback result_callback_;
    InferPackCallback pack_callback_;

    // 批处理相关成员
    std::vector<cv::Mat> current_batch_raw_frames_;
    // BatchImageMetadata 现在定义在 models.hpp 中
    std::vector<BatchImageMetadata> current_batch_metadata_;
    std::mutex batch_mutex_; // 保护批处理队列的互斥锁

    // 缓存的几何信息，避免重复计算
    struct CachedGeometry {
        int original_w = 0;
        int original_h = 0;
        float scale_to_model = 0.0f;
        int pad_w_left = 0;
        int pad_h_top = 0;
    } cached_geometry_;
    bool constant_metadata_initialized_;

    // 新增：ObjectTracker 实例
    std::unique_ptr<ObjectTracker> object_tracker_;

    // 私有辅助方法
    std::vector<char> readEngineFile(const std::string &enginePath);
    int roundToNearestMultiple(int val, int base);
    void printEngineInfo();

    /**
     * @brief 预处理单张图像以适应模型输入。
     * @param cpu_img CPU 上的原始图像。
     * @param meta 图像元数据。
     * @param model_input_w 模型输入宽度。
     * @param model_input_h 模型输入高度。
     * @param chw_planar_output_gpu_buffer_slice GPU 上的 CHW 格式平面输出缓冲区切片。
     */
    void preprocess_single_image_for_batch(const cv::Mat &cpu_img,
                                           BatchImageMetadata &meta,
                                           int model_input_w,
                                           int model_input_h,
                                           cv::cuda::GpuMat &chw_planar_output_gpu_buffer_slice);

    /**
     * @brief 执行批量推理。
     * @param pad_batch 是否对不满 BatchSize 的批次进行填充。
     */
    void performBatchInference(bool pad_batch);

    /**
     * @brief 处理单张图像的推理输出 (现在将结果传递给 ObjectTracker)。
     * @param image_meta 图像元数据。
     * @param host_output_for_image_raw 原始模型输出数据。
     * @param num_detections_in_slice 该图像的检测数量。
     * @param num_attributes_per_detection 每个检测的属性数量。
     * @param original_batch_idx_for_debug 调试用的原始批次索引。
     * @param frame_results 存储要报告的推理结果的向量 (由 ObjectTracker 填充)。
     */
    void process_single_output(const BatchImageMetadata &image_meta,
                               const float *host_output_for_image_raw,
                               int num_detections_in_slice,
                               int num_attributes_per_detection,
                               int original_batch_idx_for_debug,
                               std::vector<InferenceResult> &frame_results);

    /**
     * @brief 计算两个检测框的 IoU。
     * @param a 第一个检测框。
     * @param b 第二个检测框。
     * @return IoU 值。
     */
    // Detection 结构体现在定义在 models.hpp 中
    float calculateIoU(const Detection &a, const Detection &b);

    /**
     * @brief 应用非极大值抑制 (NMS)。
     * @param detections 原始检测列表。
     * @param iou_threshold IoU 阈值。
     * @return 经过 NMS 后的检测列表。
     */
    // Detection 结构体现在定义在 models.hpp 中
    std::vector<Detection> applyNMS(const std::vector<Detection> &detections,
                                    float iou_threshold);

    /**
     * @brief 保存带有标注框的图像。
     * @param det 检测结果。
     * @param image_meta 图像元数据。
     * @param detection_idx_in_image 图像中的检测索引。
     */
    // Detection 和 BatchImageMetadata 结构体现在定义在 models.hpp 中
    void saveAnnotatedImage(const Detection &det,
                            const BatchImageMetadata &image_meta,
                            int detection_idx_in_image);
};

#endif // TENSOR_INFERENCER_HPP
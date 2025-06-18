#ifndef OBJECT_TRACKER_HPP
#define OBJECT_TRACKER_HPP

#include <vector>
#include <map>
#include <string>
#include <memory> // For std::unique_ptr
#include <functional>
#include <algorithm>
#include <numeric>

// OpenCV 头部文件，用于处理图像和几何数据，以及 KalmanFilter
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp> // For cv::KalmanFilter

// 包含 models.hpp，其中现在包含了 Detection, InferenceResult, BatchImageMetadata 等结构体
#include "models.hpp"

// =========================================================
// 回调函数类型定义
// =========================================================

// 用于保存标注图像的回调函数，通常由 TensorInferencer 提供
// Detection, BatchImageMetadata 是在 models.hpp 中定义的
using ImageSaveCallback = std::function<void(const Detection&, const BatchImageMetadata&, int)>;
// 用于报告推理结果（包括跟踪结果）的回调函数，通常由 VideoProcessor 提供
// InferenceResult 是在 models.hpp 中定义的
using InferResultCallback = std::function<void(const InferenceResult&)>;


/**
 * @brief 表示一个被跟踪的对象轨迹。
 */
struct TrackedObject {
    int id;                   // 唯一的跟踪 ID
    int class_id;             // 对象的类别 ID
    float confidence;         // 当前检测的置信度
    cv::Rect2f bbox;          // 当前边界框 (在原始图像空间)
    int frames_since_last_detection; // 自上次检测以来经过的帧数
    int total_frames_tracked; // 总共被跟踪的帧数
    int last_seen_frame_index; // 上次被检测到的全局帧索引

    cv::KalmanFilter kf;      // 卡尔曼滤波器实例
    cv::Mat kf_state;         // 卡尔曼滤波器的状态向量 (x, y, w, h, vx, vy, vw, vh)
    cv::Mat kf_meas;          // 卡尔曼滤波器的测量向量 (x, y, w, h)

    // 构造函数：从一个新的 Detection 创建一个 TrackedObject
    TrackedObject(int new_id, const Detection& det, const BatchImageMetadata& meta, int current_frame_index);

    // 更新函数：用新的 Detection 更新 TrackedObject 的状态
    void update(const Detection& det, const BatchImageMetadata& meta, int current_frame_index);

    // 预测函数：使用卡尔曼滤波器预测下一帧的位置
    cv::Rect2f predict();
};

/**
 * @brief 负责在视频帧之间进行目标跟踪，管理目标ID和轨迹生命周期。
 */
class ObjectTracker {
public:
    /**
     * @brief 构造函数。
     * @param iou_threshold 用于数据关联的 IoU 阈值。
     * @param max_disappeared_frames 一个轨迹在被移除前可以"丢失"（未检测到）的最大帧数。
     * @param min_confidence_to_track 仅考虑置信度高于此阈值的检测结果进行跟踪。
     */
    ObjectTracker(float iou_threshold, int max_disappeared_frames, float min_confidence_to_track);

    /**
     * @brief 核心更新方法。处理当前帧的检测结果，并更新、创建或移除跟踪轨迹。
     * @param current_detections_from_inferencer 当前帧从推理器获得的所有检测。
     * @param image_meta 当前图像的元数据，用于坐标转换和图像保存。
     * @param current_global_frame_index 当前处理的全局帧索引。
     * @param result_callback 用于报告跟踪事件的回调函数。
     * @param save_callback 用于保存标注图像的回调函数。
     * @param task_id 任务 ID。
     * @param object_name 跟踪的目标对象名称。
     * @return 包含本帧要报告的所有 InferenceResult。
     */
    std::vector<InferenceResult> update(
        const std::vector<Detection>& current_detections_from_inferencer,
        const BatchImageMetadata& image_meta,
        int current_global_frame_index,
        InferResultCallback result_callback,
        ImageSaveCallback save_callback,
        int task_id,
        const std::string& object_name
    );

private:
    float iou_threshold_;          // IoU 匹配阈值
    int max_disappeared_frames_;   // 轨迹消失的最大帧数
    float min_confidence_to_track_;// 用于跟踪的最小置信度

    std::map<int, TrackedObject> active_tracks_; // 当前活跃的跟踪轨迹
    int next_track_id_;                        // 用于生成新的唯一轨迹 ID

    /**
     * @brief 计算两个边界框之间的 IoU (Intersection over Union)。
     * @param bbox1 第一个边界框。
     * @param bbox2 第二个边界框。
     * @return IoU 值。
     */
    float calculateIoU(const cv::Rect2f& bbox1, const cv::Rect2f& bbox2) const;

    // 内部 helper 函数，用于生成 InferenceResult
    void generateAndReportResult(
        int taskId,
        int frameIndex,
        float seconds,
        int tracked_id,
        const cv::Rect2f& bbox_orig_img_space,
        float confidence,
        const std::string& message,
        InferResultCallback result_callback,
        std::vector<InferenceResult>& results_to_report // 传递引用以添加结果
    ) const;
};

#endif // OBJECT_TRACKER_HPP

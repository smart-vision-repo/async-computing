#pragma once

#include "NvInfer.h"
#include "inference.hpp" // Contains InferenceInput and InferenceResult
#include <functional>    // For std::function
#include <map>
#include <mutex> // For thread safety
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 结构体：用于存储检测结果
struct Detection {
  float x1, y1, x2, y2; // 边界框坐标 (模型输出尺寸，通常是letterbox后的)
  float confidence;     // 置信度
  int class_id;         // 类别ID
  int original_batch_input_idx; // 在当前GPU批次中的索引 (用于调试或关联)
  std::string status_info;      // 额外状态信息，例如 "REAL" 或 "PAD"
};

// 结构体：用于存储单个图像预处理后的元数据 (主要用于letterbox)
struct BatchImageMetadata {
  int original_w;         // 原始图像宽度
  int original_h;         // 原始图像高度
  float scale_to_model;   // 缩放比例 (原始图像到模型输入尺寸)
  int pad_w_left;         // 左侧填充宽度
  int pad_h_top;          // 顶部填充高度
  bool is_real_image;     // 标记是否为真实图像 (非填充)
  int gopIdx_original;    // 原始GOP索引 from InferenceInput
  int global_frame_index; // NEW: Global index of this specific frame in the
                          // video stream
  cv::Mat original_image_for_callback; // 存储原始图像，用于回调和保存 (cloned
                                       // from input)
};

// 推理回调函数类型
using InferenceCallback =
    std::function<void(const std::vector<InferenceResult> &results)>;

class TensorInferencer {
public:
  // 构造函数：传入视频的原始高宽，用于计算初始目标尺寸
  TensorInferencer(int video_height, int video_width, std::string object_name,
                   int interval, float confidence, InferenceCallback callback);
  ~TensorInferencer();

  // 主推理接口：收集帧数据，满BATCH_SIZE后执行推理，并通过回调返回结果
  bool infer(const InferenceInput &input);

  // 结束推理：处理当前缓冲区中剩余的不足一个BATCH_SIZE的帧数据
  void finalizeInference();

private:
  nvinfer1::IRuntime *runtime_ = nullptr;
  nvinfer1::ICudaEngine *engine_ = nullptr;
  nvinfer1::IExecutionContext *context_ = nullptr;

  void *inputDevice_ = nullptr;
  void *outputDevice_ = nullptr;
  std::vector<void *> bindings_;

  int target_w_; // 模型输入宽度 (从引擎或计算得出)
  int target_h_; // 模型输入高度 (从引擎或计算得出)

  std::string object_name_; // Value from constructor
  int interval_; // Interval between frames in decoded_frames, from constructor
  float confidence_; // Value from constructor

  int inputIndex_ = -1;
  int outputIndex_ = -1;

  std::map<std::string, int> class_name_to_id_;
  std::map<int, std::string> id_to_class_name_;
  int num_classes_ = 0;

  std::string engine_path_;
  std::string image_output_path_; // 检测结果图像保存路径

  // --- Batch Processing Members ---
  int BATCH_SIZE_ = 1; // 从环境变量读取的批处理大小
  // std::vector<InferenceInput> current_batch_inputs_; // OLD
  std::vector<cv::Mat>
      current_batch_raw_frames_; // NEW: Stores raw frames for the current batch
  std::vector<BatchImageMetadata>
      current_batch_metadata_; // NEW: Stores metadata for each frame in
                               // current_batch_raw_frames_

  InferenceCallback current_callback_; // 当前批次回调函数
  std::mutex batch_mutex_;             // 用于保护批处理相关数据结构

  // --- Helper Methods ---
  void printEngineInfo();

  // 执行实际的批处理推理
  void performBatchInference(bool pad_batch);

  // 预处理单个图像以进行批处理 (使用letterbox)
  std::vector<float> preprocess_single_image_for_batch(
      const cv::Mat &img,
      BatchImageMetadata &meta); // meta is updated here

  // 处理单个图像的推理输出 (在批处理上下文中)
  void process_single_output(
      const BatchImageMetadata
          &image_meta, // Pass the full metadata for the specific frame
      const float *host_output_for_image_raw, // 指向该图像原始输出数据的指针
      int num_detections_in_slice,            // 模型输出中该图像的检测数量
      int num_attributes_per_detection,       // 每个检测的属性数量
      int original_batch_idx_for_debug,       // 调试用的原始GPU批处理索引
      std::vector<InferenceResult>
          &frame_results // 收集该图像的推理结果 (renamed from
                         // single_image_results)
  );

  std::vector<Detection> applyNMS(const std::vector<Detection> &detections,
                                  float iou_threshold);
  float calculateIoU(const Detection &a, const Detection &b);

  // 保存带注释的图像 (使用BatchImageMetadata进行坐标转换和文件名)
  void saveAnnotatedImage(
      const Detection &det,
      const BatchImageMetadata
          &image_meta, // Contains global_frame_index, original image etc.
      int detection_idx_in_image // 该图像内的检测索引
  );

  // 读取引擎文件
  static std::vector<char> readEngineFile(const std::string &enginePath);
  // 四舍五入到最接近的倍数
  static int roundToNearestMultiple(int val, int base = 32);
};
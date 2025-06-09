// inference_input.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct InferenceInput {
  std::vector<cv::Mat> decoded_frames;
  int gopIdx;
  int latest_frame_index;
};

// 推理结果结构
struct InferenceResult {
  std::string info;
  int taskId;
  int frameIndex;
  int seconds;
  std::string image;
  float confidence;
};

// 解码过程信息
struct TaskDecodeInfo {
  int taskId; // 推理状态信息（可扩展）
  int decoded_frames;
  int remain_frames;
};

// 推理过程信息
struct TaskInferInfo {
  int taskId; // 推理状态信息（可扩展）
  int gFrameIndex;
  int seconds;
  int results;
};

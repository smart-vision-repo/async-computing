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

// 推理结果结构体（可扩展）
struct InferenceResult {
  std::string info; // 推理状态信息（可扩展）
};

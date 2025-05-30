// inference_input.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct InferenceInput {
  std::vector<cv::Mat> decoded_frames;
  std::string object_name;
  float confidence_thresh;
  int gopIdx;
};

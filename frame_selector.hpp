#pragma once
#include "models.hpp"
#include <functional> // For std::function
#include <memory>
#include <string>

using FrameSelectorCallback = std::function<void(
    const Detection &deection, const BatchImageMetadata &meta)>;
class FrameSelector {
public:
  FrameSelector(const int inverval, const float area_delta_threshhold,
                const FrameSelectorCallback &callback);
  void removeDulicatedFrames(const DetectedFrame &info);

private:
  int interval_ = 30;
  float area_delta_threshhold_ = 0.05f;
  FrameSelectorCallback callback_;
  std::vector<FrameSelection> holdingFrames;
  float calculate_bbox_area(const Detection &det) const;
};

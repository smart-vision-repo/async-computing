#pragma once
#include "models.hpp"
#include <memory>
#include <string>
#include <functional> // For std::function

using FrameSelectorCallback = std::function<void(
    const Detection &deection, const BatchImageMetadata &meta)>;
class FrameSelector {
public:
  FrameSelector(const int inverval, const FrameSelectorCallback &callback);
  void removeDulicatedFrames(const FrameSelectorInfo &info);

private:
  int interval_ = 30;
  FrameSelectorCallback callback_;
  std::vector<FrameSelectorInfo> *holdingFrames;
};

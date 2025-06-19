#include "frame_selector.hpp"

FrameSelector::FrameSelector(const int inverval,
                             const FrameSelectorCallback &callback)
    : interval_(inverval), callback_(callback) {}

void FrameSelector::removeDulicatedFrames(const FrameSelectorInfo &info) {

  if (holdingFrames->empty()) {
    holdingFrames->push_back(info);
    return;
  }

  const FrameSelectorInfo &last_frame = holdingFrames->back();
  const int latest_frame_index = last_frame.meta->global_frame_index;

  if (latest_frame_index + interval_ == info.meta->global_frame_index) {
  }
}
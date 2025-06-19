#include "frame_selector.hpp"

FrameSelector::FrameSelector(const int inverval,
                             const float area_delta_threshhold,
                             const FrameSelectorCallback &callback)
    : interval_(inverval), area_delta_threshhold_(area_delta_threshhold),
      callback_(callback) {}

void FrameSelector::removeDulicatedFrames(
    const FrameSelectorInfo &incomming_frame) {

  if (holdingFrames->empty()) {
    holdingFrames->push_back(incomming_frame);
    return;
  }

  const FrameSelectorInfo &latest_frame = holdingFrames->back();
  const int latest_frame_index = latest_frame.meta->global_frame_index;

  if (latest_frame_index + interval_ ==
      incomming_frame.meta->global_frame_index) {
    float latest_frame_box = calculate_bbox_area(*latest_frame.detection);
    float incoming_frame_box = calculate_bbox_area(*incomming_frame.detection);
  }

  float FrameSelector::calculate_bbox_area(const Detection &det) const {
    float width = det.x2 - det.x1;
    float height = det.y2 - det.y1;
    return width * height;
  }

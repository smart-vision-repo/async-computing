#include "frame_selector.hpp"

FrameSelector::FrameSelector(const int inverval,
                             const float area_delta_threshhold,
                             const FrameSelectorCallback &callback)
    : interval_(inverval), area_delta_threshhold_(area_delta_threshhold),
      callback_(callback) {}

void FrameSelector::removeDulicatedFrames(
    const DetectedFrame &incomming_frame) {

  if (holdingFrames.empty()) {
    holdingFrames.push_back({incomming_frame, true});
    return;
  }
  const FrameSelection &latest_selection = holdingFrames.back();
  const DetectedFrame &latest_frame = latest_selection.info;
  const int latest_frame_index = latest_frame.meta.global_frame_index;

  bool output = true;
  if (latest_frame_index + interval_ ==
      incomming_frame.meta.global_frame_index) {
    const Detection &latest_detection = latest_frame.detection;
    const Detection &incoming_detection = incomming_frame.detection;
    float latest_frame_box_area = calculate_bbox_area(latest_detection);
    float incoming_frame_box_area = calculate_bbox_area(incoming_detection);
    float delta = std::abs(incoming_frame_box_area - latest_frame_box_area) /
                  latest_frame_box_area;
    if (delta < area_delta_threshhold_) {
      output = false;
      holdingFrames.back().output = output;
    }
  }

  if (output) {
    for (const auto &frameSelection : holdingFrames) {
      if (frameSelection.output) {
        callback_(frameSelection.info.detection, frameSelection.info.meta);
      }
    }
    holdingFrames.clear();
  }
  holdingFrames.push_back({incomming_frame, true});
}

float FrameSelector::calculate_bbox_area(const Detection &det) const {
  float width = det.x2 - det.x1;
  float height = det.y2 - det.y1;
  return width * height;
}

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
  const FrameSelection &holding_selection = holdingFrames.back();
  const DetectedFrame &holding_detected_frame = holding_selection.info;
  const int holding_frame_index =
      holding_detected_frame.meta.global_frame_index;

  bool output = true;
  std::cout << "holding: " << holding_frame_index
            << "incoming: " << incomming_frame.meta.global_frame_index
            << std::endl;
  if (holding_frame_index + interval_ ==
      incomming_frame.meta.global_frame_index) {
    // const Detection &latest_detection = latest_frame.detection;
    // const Detection &incoming_detection = incomming_frame.detection;
    // float latest_frame_box_area = calculate_bbox_area(latest_detection);
    // float incoming_frame_box_area = calculate_bbox_area(incoming_detection);
    // float delta = std::abs(incoming_frame_box_area - latest_frame_box_area) /
    //               latest_frame_box_area;
    // if (delta < area_delta_threshhold_) {
    //   output = false;
    //   holdingFrames.back().output = output;
    // }
    output = false;
    holdingFrames.back().output = output;
  }

  if (output) {
    for (const auto &holdingFrame : holdingFrames) {
      if (holdingFrame.output) {
        callback_(holdingFrame.info.detection, holdingFrame.info.meta);
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

#include "tensor_inferencer.hpp"
#include "object_tracker.hpp"

TensorInferencer::TensorInferencer(int interval) {
    object_tracker_ = std::make_unique<ObjectTracker>(0.45f, 3, 0.25f, static_cast<float>(interval));
}

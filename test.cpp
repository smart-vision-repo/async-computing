#include <algorithm>
#include <chrono>
#include <cmath>      // For std::round in postprocessing
#include <cstdlib>    // For getenv
#include <cstring>    // For memcpy
#include <filesystem> // For directory iteration (C++17)
#include <fstream>
#include <iomanip> // For std::setw, std::setfill, std::fixed, std::setprecision
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

// --- Configuration ---
const int BATCH_SIZE = 16; // Define the batch size

// --- Logger for TensorRT ---
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT] " << msg << std::endl;
    }
  }
} gLogger;

// --- Helper Structures ---
struct Detection {
  cv::Rect bbox; // For OpenCV drawing: x, y, width, height
  float score;
  int class_id;
  int batch_slot_idx; // Index within the current GPU batch (0 to BATCH_SIZE-1)
  std::string original_image_path; // Path of the original image this detection
                                   // belongs to
};

struct ImageMetadata {
  int original_w;
  int original_h;
  float scale_to_model;
  int pad_w_left;
  int pad_h_top;
  std::string path;
  bool is_real_image; // Flag to distinguish real images from padding
};

// --- Helper Functions ---
std::vector<std::string> load_coco_names(const std::string &filename) {
  std::vector<std::string> class_names;
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cerr << "ERROR: Could not open COCO names file: " << filename
              << std::endl;
    return class_names;
  }
  std::string line;
  while (std::getline(ifs, line)) {
    class_names.push_back(line);
  }
  return class_names;
}

// Preprocesses a single input image (resize with letterboxing, BGR->RGB,
// normalize, CHW) Returns a flat vector of float values for one image
std::vector<float> preprocess_single_image_letterbox(const cv::Mat &img,
                                                     int input_w, int input_h,
                                                     float &scale_to_model,
                                                     int &pad_w_left,
                                                     int &pad_h_top) {
  int original_w = img.cols;
  int original_h = img.rows;

  scale_to_model = 0.f; // Initialize
  pad_w_left = 0;
  pad_h_top = 0;

  cv::Mat processed_img = img; // Start with the input image

  if (original_w > 0 &&
      original_h >
          0) { // Only process if it's a valid image (not a dummy placeholder)
    scale_to_model = std::min(static_cast<float>(input_w) / original_w,
                              static_cast<float>(input_h) / original_h);
    int scaled_w = static_cast<int>(std::round(original_w * scale_to_model));
    int scaled_h = static_cast<int>(std::round(original_h * scale_to_model));

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(scaled_w, scaled_h), 0, 0,
               cv::INTER_LINEAR);

    pad_w_left = (input_w - scaled_w) / 2;
    pad_h_top = (input_h - scaled_h) / 2;
    int pad_w_right = input_w - scaled_w - pad_w_left;
    int pad_h_bottom = input_h - scaled_h - pad_h_top;

    cv::copyMakeBorder(resized_img, processed_img, pad_h_top, pad_h_bottom,
                       pad_w_left, pad_w_right, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
  } else { // If it's a dummy image (e.g. created as empty or black)
    // Ensure processed_img is the correct size and type, filled with padding
    // color
    processed_img =
        cv::Mat(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
  }

  cv::Mat rgb_img;
  cv::cvtColor(processed_img, rgb_img, cv::COLOR_BGR2RGB); // BGR to RGB
  cv::Mat fp32_img;
  rgb_img.convertTo(fp32_img, CV_32FC3, 1.f / 255.f); // Normalize to [0,1]

  std::vector<float> chw_img_data(static_cast<size_t>(input_w) * input_h * 3);
  for (int c_idx = 0; c_idx < 3; ++c_idx) { // R, G, B
    for (int h_idx = 0; h_idx < input_h; ++h_idx) {
      for (int w_idx = 0; w_idx < input_w; ++w_idx) {
        chw_img_data[static_cast<size_t>(c_idx) * input_h * input_w +
                     static_cast<size_t>(h_idx) * input_w + w_idx] =
            fp32_img.at<cv::Vec3f>(h_idx, w_idx)[c_idx];
      }
    }
  }
  return chw_img_data;
}

// NMS, now considers batch_slot_idx to apply NMS per image effectively
std::vector<Detection> nms(const std::vector<Detection> &detections,
                           float iou_threshold) {
  if (detections.empty())
    return {};

  // Group detections by batch_slot_idx
  std::map<int, std::vector<Detection>> detections_by_batch_slot;
  for (const auto &det : detections) {
    detections_by_batch_slot[det.batch_slot_idx].push_back(det);
  }

  std::vector<Detection> final_nms_results;
  for (auto const &[batch_idx, slot_detections_const] :
       detections_by_batch_slot) {
    if (slot_detections_const.empty())
      continue;

    std::vector<Detection> slot_detections =
        slot_detections_const; // Make a mutable copy

    std::sort(slot_detections.begin(), slot_detections.end(),
              [](const Detection &a, const Detection &b) {
                return a.score > b.score;
              });

    std::vector<bool> suppressed(slot_detections.size(), false);
    for (size_t i = 0; i < slot_detections.size(); ++i) {
      if (suppressed[i])
        continue;
      final_nms_results.push_back(slot_detections[i]); // Add to final results
      for (size_t j = i + 1; j < slot_detections.size(); ++j) {
        if (suppressed[j])
          continue;

        cv::Rect rect1 = slot_detections[i].bbox;
        cv::Rect rect2 = slot_detections[j].bbox;
        cv::Rect intersection = rect1 & rect2;
        float iou = 0.f;
        if (intersection.area() > 0) {
          iou = static_cast<float>(intersection.area()) /
                (rect1.area() + rect2.area() - intersection.area() + 1e-6f);
        }
        if (iou >= iou_threshold) {
          suppressed[j] = true;
        }
      }
    }
  }
  return final_nms_results;
}

// Function to list image files in a directory
std::vector<std::string> list_image_files(const std::string &directory_path) {
  std::vector<std::string> image_files;
  std::vector<std::string> valid_extensions = {".jpg", ".jpeg", ".png", ".bmp",
                                               ".tiff"};
  try {
    for (const auto &entry :
         std::filesystem::directory_iterator(directory_path)) {
      if (entry.is_regular_file()) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       ::tolower); // Convert to lowercase
        for (const auto &valid_ext : valid_extensions) {
          if (ext == valid_ext) {
            image_files.push_back(entry.path().string());
            break;
          }
        }
      }
    }
  } catch (const std::filesystem::filesystem_error &e) {
    std::cerr << "Filesystem error: " << e.what() << std::endl;
  }
  std::sort(image_files.begin(), image_files.end()); // Ensure consistent order
  return image_files;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input_image_directory>"
              << std::endl;
    return -1;
  }
  const std::string input_directory = argv[1];

  const char *engine_name_env =
      std::getenv("YOLO_ENGINE_NAME_16"); // Updated ENV variable
  const char *coco_names_env = std::getenv("YOLO_COCO_NAMES");

  if (!engine_name_env || !coco_names_env) {
    std::cerr << "ERROR: Environment variables YOLO_ENGINE_NAME_16 and "
                 "YOLO_COCO_NAMES must be set."
              << std::endl;
    return -1;
  }
  const std::string engine_filename = engine_name_env;
  const std::string coco_names_filename = coco_names_env;

  const int INPUT_W = 736;
  const int INPUT_H = 736;
  const std::string INPUT_TENSOR_NAME = "images";
  const std::string OUTPUT_TENSOR_NAME = "output0";
  float CONF_THRESHOLD = 0.25f;
  const float NMS_IOU_THRESHOLD = 0.45f;
  const std::string TARGET_CLASS_NAME = "dog";
  const int EXPECTED_NUM_DETECTIONS_YOLO_PER_IMAGE = 11109;

  std::vector<std::string> class_names = load_coco_names(coco_names_filename);
  if (class_names.empty())
    return -1;
  int num_classes = static_cast<int>(class_names.size());
  int target_class_id = -1;
  for (int i = 0; i < num_classes; ++i) {
    if (class_names[i] == TARGET_CLASS_NAME) {
      target_class_id = i;
      break;
    }
  }
  if (target_class_id == -1) { /* error */
    return -1;
  }
  std::cout << "INFO: Target class '" << TARGET_CLASS_NAME
            << "' has ID: " << target_class_id << std::endl;
  std::cout << "INFO: Total classes from COCO file: " << num_classes
            << std::endl;

  std::ifstream engine_file(engine_filename, std::ios::binary);
  if (!engine_file.is_open()) { /* error */
    return -1;
  }
  engine_file.seekg(0, engine_file.end);
  long int fsize = engine_file.tellg();
  engine_file.seekg(0, engine_file.beg);
  std::vector<char> engine_blob(fsize);
  engine_file.read(engine_blob.data(), fsize);
  engine_file.close();

  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
  if (!runtime) { /* error */
    return -1;
  }
  nvinfer1::ICudaEngine *engine =
      runtime->deserializeCudaEngine(engine_blob.data(), engine_blob.size());
  if (!engine) { /* error */
    runtime->destroy();
    return -1;
  }
  nvinfer1::IExecutionContext *context = engine->createExecutionContext();
  if (!context) { /* error */
    engine->destroy();
    runtime->destroy();
    return -1;
  }

  std::vector<void *> buffers(engine->getNbBindings());
  const int input_idx = engine->getBindingIndex(INPUT_TENSOR_NAME.c_str());
  const int output_idx = engine->getBindingIndex(OUTPUT_TENSOR_NAME.c_str());

  if (input_idx < 0 || output_idx < 0) { /* error */
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return -1;
  }

  if (engine->getBindingDataType(input_idx) != nvinfer1::DataType::kFLOAT ||
      engine->getBindingDataType(output_idx) != nvinfer1::DataType::kFLOAT) {
    std::cerr << "ERROR: Engine input/output tensor is not DataType::kFLOAT as "
                 "expected!"
              << std::endl;
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return -1;
  }
  std::cout << "INFO: Engine input and output tensors are DataType::kFLOAT."
            << std::endl;

  // Set runtime dimensions for the batch (BATCH_SIZE will be used)
  nvinfer1::Dims input_dims_runtime{4, {BATCH_SIZE, 3, INPUT_H, INPUT_W}};
  if (!context->setBindingDimensions(input_idx, input_dims_runtime)) {
    std::cerr << "ERROR: Could not set binding dimensions for input tensor for "
                 "batch size "
              << BATCH_SIZE << std::endl;
    /* cleanup */ return -1;
  }

  nvinfer1::Dims output_dims_actual = context->getBindingDimensions(output_idx);
  if (output_dims_actual.nbDims != 3 || output_dims_actual.d[0] != BATCH_SIZE ||
      output_dims_actual.d[1] != (4 + num_classes) ||
      output_dims_actual.d[2] != EXPECTED_NUM_DETECTIONS_YOLO_PER_IMAGE) {
    std::cerr << "ERROR: Output tensor dimensions mismatch. Expected ["
              << BATCH_SIZE << ", " << (4 + num_classes) << ", "
              << EXPECTED_NUM_DETECTIONS_YOLO_PER_IMAGE << "], Got [";
    for (int i = 0; i < output_dims_actual.nbDims; ++i)
      std::cerr << output_dims_actual.d[i]
                << (i == output_dims_actual.nbDims - 1 ? "" : "x");
    std::cerr << "]" << std::endl; /* cleanup */
    return -1;
  }
  std::cout << "INFO: Runtime output dimensions: " << output_dims_actual.d[0]
            << "x" << output_dims_actual.d[1] << "x" << output_dims_actual.d[2]
            << std::endl;

  size_t input_size_bytes =
      static_cast<size_t>(BATCH_SIZE) * 3 * INPUT_H * INPUT_W * sizeof(float);
  size_t output_size_elements = static_cast<size_t>(output_dims_actual.d[0]) *
                                output_dims_actual.d[1] *
                                output_dims_actual.d[2];
  size_t output_size_bytes = output_size_elements * sizeof(float);

  cudaError_t err;
  err = cudaMalloc(&buffers[input_idx], input_size_bytes);
  if (err != cudaSuccess) { /* error */
    return -1;
  }
  err = cudaMalloc(&buffers[output_idx], output_size_bytes);
  if (err != cudaSuccess) { /* error */
    cudaFree(buffers[input_idx]);
    return -1;
  }

  cudaStream_t stream;
  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) { /* error */
    return -1;
  }

  // --- List image files from directory ---
  std::vector<std::string> all_image_files = list_image_files(input_directory);
  if (all_image_files.empty()) {
    std::cout << "INFO: No image files found in directory: " << input_directory
              << std::endl;
    /* cleanup */ return 0;
  }
  std::cout << "INFO: Found " << all_image_files.size()
            << " images in directory." << std::endl;

  // --- Process images in batches ---
  for (size_t batch_start_idx = 0; batch_start_idx < all_image_files.size();
       batch_start_idx += BATCH_SIZE) {
    std::vector<float> batched_input_data;
    batched_input_data.reserve(static_cast<size_t>(BATCH_SIZE) * 3 * INPUT_H *
                               INPUT_W);
    std::vector<ImageMetadata> current_batch_metadata(BATCH_SIZE);

    int num_real_images_in_batch = 0;
    for (int i = 0; i < BATCH_SIZE; ++i) {
      size_t file_idx = batch_start_idx + i;
      cv::Mat img;
      if (file_idx < all_image_files.size()) { // Real image
        current_batch_metadata[i].path = all_image_files[file_idx];
        img = cv::imread(current_batch_metadata[i].path, cv::IMREAD_COLOR);
        if (img.empty()) {
          std::cerr << "WARNING: Could not read image: "
                    << current_batch_metadata[i].path << ". Skipping."
                    << std::endl;
          // Create a dummy black image so preprocessing doesn't fail
          img = cv::Mat(INPUT_H, INPUT_W, CV_8UC3,
                        cv::Scalar(0, 0, 0)); // Use INPUT_H, INPUT_W for dummy
          current_batch_metadata[i].is_real_image =
              false; // Mark as not real if load fails
        } else {
          current_batch_metadata[i].is_real_image = true;
          num_real_images_in_batch++;
        }
        current_batch_metadata[i].original_w = img.cols;
        current_batch_metadata[i].original_h = img.rows;
      } else { // Padding image
        // Create a dummy black image for padding
        img = cv::Mat(INPUT_H, INPUT_W, CV_8UC3,
                      cv::Scalar(0, 0, 0)); // Use INPUT_H, INPUT_W for dummy
        current_batch_metadata[i].is_real_image = false;
        current_batch_metadata[i].original_w = 0; // Indicate dummy
        current_batch_metadata[i].original_h = 0;
        current_batch_metadata[i].path = "PADDING_SLOT";
      }

      std::vector<float> single_image_data = preprocess_single_image_letterbox(
          img, INPUT_W, INPUT_H, current_batch_metadata[i].scale_to_model,
          current_batch_metadata[i].pad_w_left,
          current_batch_metadata[i].pad_h_top);
      batched_input_data.insert(batched_input_data.end(),
                                single_image_data.begin(),
                                single_image_data.end());
    }

    if (num_real_images_in_batch == 0 &&
        batch_start_idx >= all_image_files.size()) {
      // This case handles if the last batch was only padding, no need to infer.
      // However, the loop structure `batch_start_idx < all_image_files.size()`
      // prevents this. If all images in a batch failed to load,
      // num_real_images_in_batch would be 0.
      std::cout << "INFO: Batch starting at index " << batch_start_idx
                << " contains no successfully loaded real images. Skipping "
                   "inference for this batch."
                << std::endl;
      continue;
    }

    std::cout << "INFO: Processing batch starting with image index "
              << batch_start_idx << " (contains " << num_real_images_in_batch
              << " real images)." << std::endl;

    // --- Inference for the current batch ---
    err = cudaMemcpyAsync(buffers[input_idx], batched_input_data.data(),
                          input_size_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { /* error */
      break;
    } // Break from batch loop on error

    status = context->enqueueV2(buffers.data(), stream, nullptr);
    if (!status) { /* error */
      break;
    }

    std::vector<float> host_output_buffer(output_size_elements);
    err = cudaMemcpyAsync(host_output_buffer.data(), buffers[output_idx],
                          output_size_bytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) { /* error */
      break;
    }
    cudaStreamSynchronize(stream);
    // std::cout << "INFO: Inference complete for batch." << std::endl;

    // --- Postprocess current batch ---
    std::vector<Detection> all_detections_this_batch_before_nms;
    int num_attrs_from_engine = output_dims_actual.d[1];
    int num_dets_per_image_from_engine = output_dims_actual.d[2];

    for (int b = 0; b < BATCH_SIZE; ++b) { // Iterate through slots in the batch
      if (!current_batch_metadata[b].is_real_image) {
        continue; // Skip postprocessing for padded/dummy slots
      }

      const float *current_image_output_start =
          host_output_buffer.data() + static_cast<size_t>(b) *
                                          num_attrs_from_engine *
                                          num_dets_per_image_from_engine;

      std::vector<float> transposed_slice(
          static_cast<size_t>(num_dets_per_image_from_engine) *
          num_attrs_from_engine);
      for (int det_idx = 0; det_idx < num_dets_per_image_from_engine;
           ++det_idx) {
        for (int attr_idx = 0; attr_idx < num_attrs_from_engine; ++attr_idx) {
          transposed_slice[static_cast<size_t>(det_idx) *
                               num_attrs_from_engine +
                           attr_idx] =
              current_image_output_start[static_cast<size_t>(attr_idx) *
                                             num_dets_per_image_from_engine +
                                         det_idx];
        }
      }

      for (int i = 0; i < num_dets_per_image_from_engine; ++i) {
        const float *det_data = transposed_slice.data() +
                                static_cast<size_t>(i) * num_attrs_from_engine;
        float max_class_score = 0.f;
        int current_class_id = -1;
        for (int j = 0; j < num_classes; ++j) {
          if (det_data[4 + j] > max_class_score) {
            max_class_score = det_data[4 + j];
            current_class_id = j;
          }
        }

        if (max_class_score > CONF_THRESHOLD) {
          if (current_class_id == target_class_id) {
            float cx = det_data[0];
            float cy = det_data[1];
            float w_box = det_data[2];
            float h_box =
                det_data[3]; // Renamed to avoid conflict with INPUT_W/H
            int x_min_model = static_cast<int>(std::round(cx - w_box / 2.f));
            int y_min_model = static_cast<int>(std::round(cy - h_box / 2.f));
            int box_w_model = static_cast<int>(std::round(w_box));
            int box_h_model = static_cast<int>(std::round(h_box));
            all_detections_this_batch_before_nms.push_back(
                {{x_min_model, y_min_model, box_w_model, box_h_model},
                 max_class_score,
                 current_class_id,
                 b,
                 current_batch_metadata[b].path});
          }
        }
      }
    }

    std::vector<Detection> nms_results_this_batch =
        nms(all_detections_this_batch_before_nms, NMS_IOU_THRESHOLD);

    if (nms_results_this_batch.empty() && num_real_images_in_batch > 0) {
      std::cout << "  INFO: No " << TARGET_CLASS_NAME
                << " detected meeting criteria in the current batch of "
                << num_real_images_in_batch << " real image(s)." << std::endl;
    }

    for (const auto &det : nms_results_this_batch) {
      const ImageMetadata &meta =
          current_batch_metadata[det.batch_slot_idx]; // Get metadata for the
                                                      // specific image in batch
      if (!meta.is_real_image)
        continue; // Should not happen if NMS processes real images only

      float x_min_scaled_img = static_cast<float>(det.bbox.x) - meta.pad_w_left;
      float y_min_scaled_img = static_cast<float>(det.bbox.y) - meta.pad_h_top;
      float x_max_scaled_img =
          static_cast<float>(det.bbox.x + det.bbox.width) - meta.pad_w_left;
      float y_max_scaled_img =
          static_cast<float>(det.bbox.y + det.bbox.height) - meta.pad_h_top;

      int final_x_min =
          static_cast<int>(std::round(x_min_scaled_img / meta.scale_to_model));
      int final_y_min =
          static_cast<int>(std::round(y_min_scaled_img / meta.scale_to_model));
      int final_x_max =
          static_cast<int>(std::round(x_max_scaled_img / meta.scale_to_model));
      int final_y_max =
          static_cast<int>(std::round(y_max_scaled_img / meta.scale_to_model));

      final_x_min = std::max(0, final_x_min);
      final_y_min = std::max(0, final_y_min);
      final_x_max = std::min(meta.original_w, final_x_max);
      final_y_max = std::min(meta.original_h, final_y_max);

      if (final_x_max > final_x_min && final_y_max > final_y_min) {
        std::cout << "  Image (" << det.original_image_path << "): BBOX: ["
                  << final_x_min << ", " << final_y_min << ", " << final_x_max
                  << ", " << final_y_max << "], Score: " << std::fixed
                  << std::setprecision(6) << det.score << std::endl;
      }
    }
  } // End of batch processing loop

  // --- Cleanup ---
  cudaFree(buffers[input_idx]);
  cudaFree(buffers[output_idx]);
  cudaStreamDestroy(stream);
  context->destroy();
  engine->destroy();
  runtime->destroy();

  std::cout << "INFO: All processing finished." << std::endl;
  return 0;
}
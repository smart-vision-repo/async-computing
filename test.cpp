#include <algorithm>
#include <chrono>
#include <cstdlib> // For getenv
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h" // Not strictly needed for inference if engine is pre-built, but good for context
#include "cuda_runtime_api.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

// --- Logger for TensorRT ---
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <=
        Severity::kWARNING) { // kINFO, kWARNING, kERROR, kINTERNAL_ERROR
      std::cerr << msg << std::endl;
    }
  }
} gLogger;

// --- Helper Structures ---
struct Detection {
  cv::Rect bbox;
  float score;
  int class_id;
};

// --- Helper Functions ---
// Reads class names from a file
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

// Preprocesses the input image (resize, letterbox, BGR->RGB, normalize, CHW, to
// FP16) Returns a flat vector of half_float::half values and scaling/padding
// info
std::vector<uint16_t> preprocess_image(const cv::Mat &img, int input_w,
                                       int input_h, float &scale, int &pad_w,
                                       int &pad_h) {
  int original_w = img.cols;
  int original_h = img.rows;

  scale = std::min(static_cast<float>(input_w) / original_w,
                   static_cast<float>(input_h) / original_h);

  int scaled_w = static_cast<int>(original_w * scale);
  int scaled_h = static_cast<int>(original_h * scale);

  cv::Mat resized_img;
  cv::resize(img, resized_img, cv::Size(scaled_w, scaled_h), 0, 0,
             cv::INTER_LINEAR);

  pad_w = (input_w - scaled_w) / 2;
  pad_h = (input_h - scaled_h) / 2;

  cv::Mat padded_img(input_h, input_w, CV_8UC3,
                     cv::Scalar(114, 114, 114)); // Common padding color
  resized_img.copyTo(padded_img(cv::Rect(pad_w, pad_h, scaled_w, scaled_h)));

  cv::Mat rgb_img;
  cv::cvtColor(padded_img, rgb_img, cv::COLOR_BGR2RGB);

  cv::Mat fp32_img;
  rgb_img.convertTo(fp32_img, CV_32FC3, 1.f / 255.f);

  std::vector<float> chw_img_data(input_w * input_h * 3);
  for (int i = 0; i < input_h; ++i) {
    for (int j = 0; j < input_w; ++j) {
      chw_img_data[0 * input_h * input_w + i * input_w + j] =
          fp32_img.at<cv::Vec3f>(i, j)[0]; // R
      chw_img_data[1 * input_h * input_w + i * input_w + j] =
          fp32_img.at<cv::Vec3f>(i, j)[1]; // G
      chw_img_data[2 * input_h * input_w + i * input_w + j] =
          fp32_img.at<cv::Vec3f>(i, j)[2]; // B
    }
  }

  // Convert to FP16
  std::vector<uint16_t> fp16_img_data(chw_img_data.size());
  // Basic float to half conversion (simplified, for a proper one use a library
  // or more precise conversion) This is a placeholder. For robust FP16
  // conversion, use a proper library or CUDA intrinsics if available or ensure
  // your compiler supports __fp16 if you were writing CUDA kernels. For CPU
  // side, you'd typically rely on a library or bit manipulation. A simple way
  // is to use a union or specific conversion functions if available. Here,
  // we'll just cast for simplicity, acknowledging it's not a true bit-correct
  // half conversion. For TensorRT 8.6+, consider using its FP16 utilities if
  // converting on device or if preparing data. Since we are copying to GPU and
  // the engine expects FP16, the CUDA driver/TRT handles the format. The key is
  // that the buffer on GPU is FP16. Here we prepare a CPU buffer. A proper
  // CPU-side float to half conversion is more complex. Let's assume for now the
  // data is prepared as float and will be correctly interpreted or copied. For
  // true CPU-side FP16 generation, you'd do:
  for (size_t i = 0; i < chw_img_data.size(); ++i) {
    // This is a very naive conversion and likely incorrect for many values.
    // fp16_img_data[i] = static_cast<uint16_t>(chw_img_data[i] *
    // some_fp16_factor); // Placeholder Proper conversion:
    float val = chw_img_data[i];
    unsigned int as_int;
    memcpy(&as_int, &val, sizeof(float));
    unsigned short sign = (as_int >> 31) & 0x1;
    unsigned short exponent = ((as_int >> 23) & 0xFF) - 127 + 15; // Adjust bias
    unsigned short mantissa = (as_int >> (23 - 10)) & 0x3FF;

    if (((as_int >> 23) & 0xFF) == 0) { // Zero or subnormal
      exponent = 0;
      mantissa = 0;                               // simplified
    } else if (((as_int >> 23) & 0xFF) == 0xFF) { // Inf or NaN
      exponent = 0x1F;
      mantissa = (as_int & 0x7FFFFF) ? 0x200 : 0; // NaN if mantissa is non-zero
    } else if (exponent <= 0) { // Underflow to zero or subnormal
      // For simplicity, map to zero. Proper subnormal conversion is more
      // complex.
      exponent = 0;
      mantissa = 0;
    } else if (exponent >= 0x1F) { // Overflow
      exponent = 0x1F;
      mantissa = 0;
    }
    fp16_img_data[i] = (sign << 15) | (exponent << 10) | mantissa;
  }

  return fp16_img_data;
}

// NMS
std::vector<Detection> nms(std::vector<Detection> &detections,
                           float iou_threshold) {
  std::vector<Detection> nms_detections;
  if (detections.empty()) {
    return nms_detections;
  }

  std::sort(
      detections.begin(), detections.end(),
      [](const Detection &a, const Detection &b) { return a.score > b.score; });

  std::vector<bool> suppressed(detections.size(), false);

  for (size_t i = 0; i < detections.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    nms_detections.push_back(detections[i]);
    for (size_t j = i + 1; j < detections.size(); ++j) {
      if (suppressed[j]) {
        continue;
      }
      cv::Rect rect1 = detections[i].bbox;
      cv::Rect rect2 = detections[j].bbox;
      cv::Rect intersection = rect1 & rect2;
      float iou = static_cast<float>(intersection.area()) /
                  (rect1.area() + rect2.area() - intersection.area());
      if (iou >= iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
  return nms_detections;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return -1;
  }
  const std::string image_path = argv[1];

  const char *engine_name_env = std::getenv("YOLO_ENGINE_NAME");
  const char *coco_names_env = std::getenv("YOLO_COCO_NAMES");

  if (!engine_name_env) {
    std::cerr << "ERROR: Environment variable YOLO_ENGINE_NAME not set."
              << std::endl;
    return -1;
  }
  if (!coco_names_env) {
    std::cerr << "ERROR: Environment variable YOLO_COCO_NAMES not set."
              << std::endl;
    return -1;
  }
  const std::string engine_filename = engine_name_env;
  const std::string coco_names_filename = coco_names_env;

  // --- Parameters based on YOLOv8n and user input ---
  const int INPUT_W = 640;
  const int INPUT_H = 640;
  const std::string INPUT_TENSOR_NAME = "images";
  const std::string OUTPUT_TENSOR_NAME = "output0"; // Common for YOLOv8 ONNX
  const float CONF_THRESHOLD = 0.25f;
  const float NMS_IOU_THRESHOLD = 0.45f;
  const std::string TARGET_CLASS_NAME = "dog";
  // Output format: [batch, 4+num_classes, 8400] e.g. [1, 84, 8400] for COCO (80
  // classes)

  // 1. Load COCO names and find target class ID
  std::vector<std::string> class_names = load_coco_names(coco_names_filename);
  if (class_names.empty()) {
    return -1;
  }
  int num_classes = class_names.size();
  int target_class_id = -1;
  for (int i = 0; i < num_classes; ++i) {
    if (class_names[i] == TARGET_CLASS_NAME) {
      target_class_id = i;
      break;
    }
  }
  if (target_class_id == -1) {
    std::cerr << "ERROR: Target class '" << TARGET_CLASS_NAME
              << "' not found in COCO names file." << std::endl;
    return -1;
  }
  std::cout << "INFO: Target class '" << TARGET_CLASS_NAME
            << "' has ID: " << target_class_id << std::endl;
  std::cout << "INFO: Total classes: " << num_classes << std::endl;

  // 2. Deserialize TensorRT Engine
  std::ifstream engine_file(engine_filename, std::ios::binary);
  if (!engine_file.is_open()) {
    std::cerr << "ERROR: Could not open engine file: " << engine_filename
              << std::endl;
    return -1;
  }
  engine_file.seekg(0, engine_file.end);
  long int fsize = engine_file.tellg();
  engine_file.seekg(0, engine_file.beg);
  std::vector<char> engine_blob(fsize);
  engine_file.read(engine_blob.data(), fsize);
  engine_file.close();

  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
  if (!runtime) {
    std::cerr << "ERROR: Could not create TensorRT runtime." << std::endl;
    return -1;
  }
  nvinfer1::ICudaEngine *engine =
      runtime->deserializeCudaEngine(engine_blob.data(), engine_blob.size());
  if (!engine) {
    std::cerr << "ERROR: Could not deserialize TensorRT engine." << std::endl;
    runtime->destroy();
    return -1;
  }
  nvinfer1::IExecutionContext *context = engine->createExecutionContext();
  if (!context) {
    std::cerr << "ERROR: Could not create TensorRT execution context."
              << std::endl;
    engine->destroy();
    runtime->destroy();
    return -1;
  }

  // 3. Prepare Buffers
  void *buffers[2]; // 0: input, 1: output
  const int input_idx = engine->getBindingIndex(INPUT_TENSOR_NAME.c_str());
  const int output_idx = engine->getBindingIndex(OUTPUT_TENSOR_NAME.c_str());

  if (input_idx < 0) {
    std::cerr << "ERROR: Input tensor '" << INPUT_TENSOR_NAME
              << "' not found in engine." << std::endl;
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return -1;
  }
  if (output_idx < 0) {
    std::cerr << "ERROR: Output tensor '" << OUTPUT_TENSOR_NAME
              << "' not found in engine." << std::endl;
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return -1;
  }

  // Set dynamic input dimensions (using opt profile, which is 1x3x640x640 for
  // input "images") The engine was built with min/opt/max for "images":
  // 1x3x320x320 / 1x3x640x640 / 1x3x1088x1920 We are preprocessing to 640x640,
  // so we set this dimension.
  nvinfer1::Dims input_dims{
      4, {1, 3, INPUT_H, INPUT_W}}; // Batch, Channel, Height, Width
  if (!context->setBindingDimensions(input_idx, input_dims)) {
    std::cerr << "ERROR: Could not set binding dimensions for input tensor."
              << std::endl;
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return -1;
  }

  // Verify dimensions after setting (optional)
  // nvinfer1::Dims actual_input_dims =
  // context->getBindingDimensions(input_idx); std::cout << "INFO: Actual input
  // dimensions set: "; for (int i = 0; i < actual_input_dims.nbDims; ++i)
  // std::cout << actual_input_dims.d[i] << " "; std::cout << std::endl;

  size_t input_size = 1 * 3 * INPUT_H * INPUT_W * sizeof(uint16_t); // FP16
  // nvinfer1::Dims output_dims = engine->getBindingDimensions(output_idx); //
  // This gets profile dims
  nvinfer1::Dims output_dims_actual = context->getBindingDimensions(
      output_idx); // This gets actual dims after setting input
  // Expected: [1, 4+num_classes, 8400] e.g. [1, 84, 8400]
  // Ensure output_dims_actual.d[1] == 4 + num_classes and
  // output_dims_actual.d[2] == 8400
  if (output_dims_actual.nbDims != 3 || output_dims_actual.d[0] != 1 ||
      output_dims_actual.d[1] != (4 + num_classes) ||
      output_dims_actual.d[2] != 8400) {
    std::cerr << "ERROR: Output tensor dimensions mismatch. Expected [1, "
              << (4 + num_classes) << ", 8400], Got [";
    for (int i = 0; i < output_dims_actual.nbDims; ++i)
      std::cerr << output_dims_actual.d[i]
                << (i == output_dims_actual.nbDims - 1 ? "" : ", ");
    std::cerr << "]" << std::endl;
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return -1;
  }

  size_t output_size = output_dims_actual.d[0] * output_dims_actual.d[1] *
                       output_dims_actual.d[2] *
                       sizeof(float); // Output is usually FP32 from YOLO

  cudaError_t err;
  err = cudaMalloc(&buffers[input_idx], input_size);
  if (err != cudaSuccess) {
    std::cerr << "CUDA Malloc Error input: " << cudaGetErrorString(err)
              << std::endl;
    return -1;
  }
  err = cudaMalloc(&buffers[output_idx], output_size);
  if (err != cudaSuccess) {
    std::cerr << "CUDA Malloc Error output: " << cudaGetErrorString(err)
              << std::endl;
    return -1;
  }

  cudaStream_t stream;
  err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) {
    std::cerr << "CUDA Stream Create Error: " << cudaGetErrorString(err)
              << std::endl;
    return -1;
  }

  // 4. Load and Preprocess Image
  cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "ERROR: Could not read image: " << image_path << std::endl;
    cudaFree(buffers[input_idx]);
    cudaFree(buffers[output_idx]);
    cudaStreamDestroy(stream);
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return -1;
  }
  int original_img_w = img.cols;
  int original_img_h = img.rows;

  float scale_factor;
  int pad_w, pad_h;
  std::vector<uint16_t> preprocessed_img_data =
      preprocess_image(img, INPUT_W, INPUT_H, scale_factor, pad_w, pad_h);

  // 5. Inference
  err = cudaMemcpyAsync(buffers[input_idx], preprocessed_img_data.data(),
                        input_size, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    std::cerr << "CUDA Memcpy H2D Error: " << cudaGetErrorString(err)
              << std::endl;
    return -1;
  }

  bool status = context->enqueueV2(buffers, stream, nullptr);
  if (!status) {
    std::cerr << "ERROR: TensorRT inference failed." << std::endl;
    cudaFree(buffers[input_idx]);
    cudaFree(buffers[output_idx]);
    cudaStreamDestroy(stream);
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return -1;
  }

  std::vector<float> output_buffer_host(output_dims_actual.d[0] *
                                        output_dims_actual.d[1] *
                                        output_dims_actual.d[2]);
  err = cudaMemcpyAsync(output_buffer_host.data(), buffers[output_idx],
                        output_size, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    std::cerr << "CUDA Memcpy D2H Error: " << cudaGetErrorString(err)
              << std::endl;
    return -1;
  }

  cudaStreamSynchronize(stream);

  // 6. Postprocess
  // Output shape: [1, 84, 8400] (batch, attributes, num_detections)
  // Attributes: [cx, cy, w, h, class1_score, ..., classN_score]
  // Transpose to [1, 8400, 84] for easier iteration
  std::vector<float> transposed_output(output_buffer_host.size());
  int num_attrs = output_dims_actual.d[1];     // e.g., 84
  int num_dets_yolo = output_dims_actual.d[2]; // e.g., 8400

  for (int i = 0; i < num_dets_yolo; ++i) {
    for (int j = 0; j < num_attrs; ++j) {
      transposed_output[i * num_attrs + j] =
          output_buffer_host[j * num_dets_yolo + i];
    }
  }

  std::vector<Detection> detections;
  for (int i = 0; i < num_dets_yolo; ++i) {
    const float *det_data = transposed_output.data() + i * num_attrs;

    float max_class_score = 0.f;
    int current_class_id = -1;

    for (int j = 0; j < num_classes; ++j) {
      if (det_data[4 + j] > max_class_score) {
        max_class_score = det_data[4 + j];
        current_class_id = j;
      }
    }

    if (max_class_score > CONF_THRESHOLD) {
      if (current_class_id == target_class_id) { // Filter for "dog"
        float cx = det_data[0];
        float cy = det_data[1];
        float w = det_data[2];
        float h = det_data[3];

        // Convert to x_min, y_min, x_max, y_max for model input scale (640x640)
        int x_min = static_cast<int>(cx - w / 2.f);
        int y_min = static_cast<int>(cy - h / 2.f);
        int x_max = static_cast<int>(cx + w / 2.f);
        int y_max = static_cast<int>(cy + h / 2.f);

        detections.push_back({{x_min, y_min, x_max - x_min, y_max - y_min},
                              max_class_score,
                              current_class_id});
      }
    }
  }

  std::vector<Detection> nms_results = nms(detections, NMS_IOU_THRESHOLD);

  std::cout
      << "Detected '" << TARGET_CLASS_NAME
      << "' BBOXes (x_min, y_min, x_max, y_max) in original image coordinates:"
      << std::endl;
  for (const auto &det : nms_results) {
    // Scale bbox back to original image dimensions
    cv::Rect box_model_scale = det.bbox; // This is on the 640x640 padded image

    // Remove padding
    float x_min_orig_padded = box_model_scale.x - pad_w;
    float y_min_orig_padded = box_model_scale.y - pad_h;
    float x_max_orig_padded =
        (box_model_scale.x + box_model_scale.width) - pad_w;
    float y_max_orig_padded =
        (box_model_scale.y + box_model_scale.height) - pad_h;

    // Rescale to original image
    int final_x_min =
        static_cast<int>(std::round(x_min_orig_padded / scale_factor));
    int final_y_min =
        static_cast<int>(std::round(y_min_orig_padded / scale_factor));
    int final_x_max =
        static_cast<int>(std::round(x_max_orig_padded / scale_factor));
    int final_y_max =
        static_cast<int>(std::round(y_max_orig_padded / scale_factor));

    // Clamp to image boundaries
    final_x_min = std::max(0, final_x_min);
    final_y_min = std::max(0, final_y_min);
    final_x_max = std::min(original_img_w, final_x_max);
    final_y_max = std::min(original_img_h, final_y_max);

    if (final_x_max > final_x_min && final_y_max > final_y_min) {
      std::cout << "  BBOX: [" << final_x_min << ", " << final_y_min << ", "
                << final_x_max << ", " << final_y_max
                << "], Score: " << det.score << std::endl;
    }
  }

  // 7. Cleanup
  cudaFree(buffers[input_idx]);
  cudaFree(buffers[output_idx]);
  cudaStreamDestroy(stream);
  context->destroy();
  engine->destroy();
  runtime->destroy();

  return 0;
}
#include <algorithm>
#include <chrono>
#include <cmath>   // For std::round in postprocessing
#include <cstdlib> // For getenv
#include <cstring> // For memcpy
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "NvInfer.h"
// #include "NvOnnxParser.h" // Not strictly needed for inference if engine is
// pre-built
#include "cuda_runtime_api.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

// --- Logger for TensorRT ---
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    // 只打印 kWARNING 及以上级别的信息，以减少不必要的输出
    if (severity <= Severity::kWARNING) {
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
// FP16) Returns a flat vector of uint16_t (half_float::half) values and
// scaling/padding info
std::vector<uint16_t> preprocess_image(const cv::Mat &img, int input_w,
                                       int input_h, float &scale,
                                       int &pad_w_left, int &pad_h_top) {
  int original_w = img.cols;
  int original_h = img.rows;

  scale = std::min(static_cast<float>(input_w) / original_w,
                   static_cast<float>(input_h) / original_h);

  int scaled_w = static_cast<int>(std::round(original_w * scale));
  int scaled_h = static_cast<int>(std::round(original_h * scale));

  cv::Mat resized_img;
  cv::resize(img, resized_img, cv::Size(scaled_w, scaled_h), 0, 0,
             cv::INTER_LINEAR);

  // Calculate padding (top-left padding)
  pad_w_left = (input_w - scaled_w) / 2;
  pad_h_top = (input_h - scaled_h) / 2;
  // Ensure right/bottom padding is handled if dimensions are odd
  int pad_w_right = input_w - scaled_w - pad_w_left;
  int pad_h_bottom = input_h - scaled_h - pad_h_top;

  cv::Mat padded_img;
  // Common padding color (114, 114, 114)
  cv::copyMakeBorder(resized_img, padded_img, pad_h_top, pad_h_bottom,
                     pad_w_left, pad_w_right, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));

  cv::Mat rgb_img;
  cv::cvtColor(padded_img, rgb_img, cv::COLOR_BGR2RGB);

  cv::Mat fp32_img;
  rgb_img.convertTo(fp32_img, CV_32FC3, 1.f / 255.f);

  std::vector<float> chw_img_data(static_cast<size_t>(input_w) * input_h * 3);
  for (int i = 0; i < input_h; ++i) {
    for (int j = 0; j < input_w; ++j) {
      // HWC to CHW conversion: RRR...GGG...BBB...
      chw_img_data[0 * static_cast<size_t>(input_h) * input_w +
                   static_cast<size_t>(i) * input_w + j] =
          fp32_img.at<cv::Vec3f>(i, j)[0]; // R
      chw_img_data[1 * static_cast<size_t>(input_h) * input_w +
                   static_cast<size_t>(i) * input_w + j] =
          fp32_img.at<cv::Vec3f>(i, j)[1]; // G
      chw_img_data[2 * static_cast<size_t>(input_h) * input_w +
                   static_cast<size_t>(i) * input_w + j] =
          fp32_img.at<cv::Vec3f>(i, j)[2]; // B
    }
  }

  // Convert to FP16 (uint16_t) using a more robust CPU-side conversion
  std::vector<uint16_t> fp16_img_data(chw_img_data.size());
  for (size_t i = 0; i < chw_img_data.size(); ++i) {
    float float_val = chw_img_data[i];
    unsigned int int_val;
    memcpy(&int_val, &float_val, sizeof(float));

    unsigned short sign =
        (unsigned short)((int_val >> 16) & 0x8000); // sign from bit 31 of float
    short exponent_float =
        (short)(((int_val >> 23) & 0xFF) - 127); // exponent of float (bias 127)
    unsigned int mantissa_float = int_val & 0x7FFFFF; // mantissa of float

    unsigned short half_val;

    if (exponent_float == 128) { // Special case: NaN or Inf for float
      // NaN or Inf for half (exponent bits all 1)
      // If mantissa_float is non-zero, it's NaN, otherwise Inf
      half_val = sign | 0x7C00 | (mantissa_float ? 0x200 : 0);
    } else if (exponent_float >
               15) { // Overflow for half (max normal exponent for half is 15)
      half_val = sign | 0x7C00;         // Becomes Inf
    } else if (exponent_float <= -15) { // Underflow or zero for half (min
                                        // normal exponent for half is -14)
      if (exponent_float < -24) {       // Too small, flush to zero
        half_val = sign | 0x0000;       // Zero
      } else {                          // Subnormal for half
        // Add implicit leading bit for float mantissa
        mantissa_float =
            (mantissa_float | 0x800000) >>
            (1 - (exponent_float - (-14))); // Shift to align for subnormal half
        half_val = sign | (unsigned short)mantissa_float;
      }
    } else { // Normal value for half
      // Exponent for half (bias 15)
      unsigned short exponent_half = (unsigned short)(exponent_float + 15);
      // Mantissa for half (10 bits from float's 23 bits)
      unsigned short mantissa_half = (unsigned short)(mantissa_float >> 13);
      half_val = sign | (exponent_half << 10) | mantissa_half;
    }
    fp16_img_data[i] = half_val;
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

  // Sort by score in descending order
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

      // Calculate IoU
      cv::Rect intersection = rect1 & rect2;
      float iou = 0.f;
      if (intersection.area() > 0) {
        iou = static_cast<float>(intersection.area()) /
              (rect1.area() + rect2.area() - intersection.area());
      }

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
  float CONF_THRESHOLD = 0.25f; // Can be lowered for debugging
  const float NMS_IOU_THRESHOLD = 0.45f;
  const std::string TARGET_CLASS_NAME = "dog";

  // 1. Load COCO names and find target class ID
  std::vector<std::string> class_names = load_coco_names(coco_names_filename);
  if (class_names.empty()) {
    return -1;
  }
  int num_classes = static_cast<int>(class_names.size());
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
  std::cout << "INFO: Total classes from COCO file: " << num_classes
            << std::endl;

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
  void *buffers[engine->getNbBindings()];
  const int input_idx = engine->getBindingIndex(INPUT_TENSOR_NAME.c_str());
  const int output_idx = engine->getBindingIndex(OUTPUT_TENSOR_NAME.c_str());

  if (input_idx < 0) {
    std::cerr << "ERROR: Input tensor '" << INPUT_TENSOR_NAME
              << "' not found in engine. Bindings:" << std::endl;
    for (int i = 0; i < engine->getNbBindings(); ++i)
      std::cerr << "  " << engine->getBindingName(i) << std::endl;
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return -1;
  }
  if (output_idx < 0) {
    std::cerr << "ERROR: Output tensor '" << OUTPUT_TENSOR_NAME
              << "' not found in engine. Bindings:" << std::endl;
    for (int i = 0; i < engine->getNbBindings(); ++i)
      std::cerr << "  " << engine->getBindingName(i) << std::endl;
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return -1;
  }
  if (engine->getBindingDataType(input_idx) != nvinfer1::DataType::kHALF) {
    std::cerr << "ERROR: Engine input tensor '" << INPUT_TENSOR_NAME
              << "' is not DataType::kHALF as expected!" << std::endl;
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return -1;
  }

  nvinfer1::Dims input_dims_engine{4, {1, 3, INPUT_H, INPUT_W}};
  if (!context->setBindingDimensions(input_idx, input_dims_engine)) {
    std::cerr << "ERROR: Could not set binding dimensions for input tensor."
              << std::endl;
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return -1;
  }

  // Check if output tensor data type is FP16 or FP32
  bool output_is_fp16 =
      (engine->getBindingDataType(output_idx) == nvinfer1::DataType::kHALF);
  if (output_is_fp16) {
    std::cout << "INFO: Output tensor '" << OUTPUT_TENSOR_NAME << "' is FP16."
              << std::endl;
  } else if (engine->getBindingDataType(output_idx) ==
             nvinfer1::DataType::kFLOAT) {
    std::cout << "INFO: Output tensor '" << OUTPUT_TENSOR_NAME << "' is FP32."
              << std::endl;
  } else {
    std::cerr << "ERROR: Output tensor '" << OUTPUT_TENSOR_NAME
              << "' has unexpected data type." << std::endl;
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return -1;
  }

  size_t input_size_bytes = static_cast<size_t>(1) * 3 * INPUT_H * INPUT_W *
                            sizeof(uint16_t); // FP16 input

  nvinfer1::Dims output_dims_actual = context->getBindingDimensions(output_idx);
  // Expected: [1, 4+num_classes, 8400] e.g. [1, 84, 8400] for COCO (80 classes)
  const int EXPECTED_NUM_DETECTIONS_YOLO = 8400;
  if (output_dims_actual.nbDims != 3 || output_dims_actual.d[0] != 1 ||
      output_dims_actual.d[1] != (4 + num_classes) ||
      output_dims_actual.d[2] != EXPECTED_NUM_DETECTIONS_YOLO) {
    std::cerr << "ERROR: Output tensor dimensions mismatch. Expected [1, "
              << (4 + num_classes) << ", " << EXPECTED_NUM_DETECTIONS_YOLO
              << "], Got [";
    for (int i = 0; i < output_dims_actual.nbDims; ++i)
      std::cerr << output_dims_actual.d[i]
                << (i == output_dims_actual.nbDims - 1 ? "" : ", ");
    std::cerr << "]" << std::endl;
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return -1;
  }

  size_t output_size_elements = static_cast<size_t>(output_dims_actual.d[0]) *
                                output_dims_actual.d[1] *
                                output_dims_actual.d[2];
  size_t output_size_bytes =
      output_size_elements *
      (output_is_fp16 ? sizeof(uint16_t) : sizeof(float));

  cudaError_t err;
  err = cudaMalloc(&buffers[input_idx], input_size_bytes);
  if (err != cudaSuccess) {
    std::cerr << "CUDA Malloc Error input: " << cudaGetErrorString(err)
              << std::endl;
    return -1;
  }
  err = cudaMalloc(&buffers[output_idx], output_size_bytes);
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
  int pad_w_val, pad_h_val;
  std::vector<uint16_t> preprocessed_img_data = preprocess_image(
      img, INPUT_W, INPUT_H, scale_factor, pad_w_val, pad_h_val);

  // 5. Inference
  err = cudaMemcpyAsync(buffers[input_idx], preprocessed_img_data.data(),
                        input_size_bytes, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    std::cerr << "CUDA Memcpy H2D Error: " << cudaGetErrorString(err)
              << std::endl;
    return -1;
  }

  bool status = context->enqueueV2(buffers, stream, nullptr);
  if (!status) {
    std::cerr << "ERROR: TensorRT inference failed." << std::endl;
    // cudaFree, destroy context, etc.
    return -1;
  }

  // Prepare host buffer for output (can be FP16 or FP32)
  std::vector<char> output_raw_buffer_host(output_size_bytes);
  err = cudaMemcpyAsync(output_raw_buffer_host.data(), buffers[output_idx],
                        output_size_bytes, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    std::cerr << "CUDA Memcpy D2H Error: " << cudaGetErrorString(err)
              << std::endl;
    return -1;
  }

  cudaStreamSynchronize(stream);

  // Convert output to FP32 if it was FP16, for easier processing
  std::vector<float> output_fp32_buffer_host(output_size_elements);
  if (output_is_fp16) {
    const uint16_t *half_ptr =
        reinterpret_cast<const uint16_t *>(output_raw_buffer_host.data());
    for (size_t i = 0; i < output_size_elements; ++i) {
      uint16_t half_val = half_ptr[i];
      unsigned int sign = (half_val >> 15) & 0x1;
      unsigned int exponent_half = (half_val >> 10) & 0x1F;
      unsigned int mantissa_half = half_val & 0x3FF;

      float float_val;
      if (exponent_half == 0) {   // Zero or subnormal
        if (mantissa_half == 0) { // Zero
          float_val = (sign ? -0.0f : 0.0f);
        } else { // Subnormal
          // (sign ? -1 : 1) * 2^(-14) * (mantissa / 2^10)
          float_val = (sign ? -1.f : 1.f) * std::pow(2.f, -14) *
                      (static_cast<float>(mantissa_half) / 1024.f);
        }
      } else if (exponent_half == 0x1F) { // Inf or NaN
        if (mantissa_half == 0) {         // Inf
          float_val = (sign ? -std::numeric_limits<float>::infinity()
                            : std::numeric_limits<float>::infinity());
        } else { // NaN
          float_val = std::numeric_limits<float>::quiet_NaN();
        }
      } else { // Normal
        // (sign ? -1 : 1) * 2^(exponent - 15) * (1 + mantissa / 2^10)
        float_val = (sign ? -1.f : 1.f) *
                    std::pow(2.f, static_cast<int>(exponent_half) - 15) *
                    (1.f + static_cast<float>(mantissa_half) / 1024.f);
      }
      output_fp32_buffer_host[i] = float_val;
    }
  } else {
    memcpy(output_fp32_buffer_host.data(), output_raw_buffer_host.data(),
           output_size_bytes);
  }

  // 6. Postprocess (using output_fp32_buffer_host)
  std::vector<float> transposed_output(output_size_elements);
  int num_attrs = output_dims_actual.d[1];
  int num_dets_yolo = output_dims_actual.d[2];

  for (int i = 0; i < num_dets_yolo; ++i) { // Iterate through detections
    for (int j = 0; j < num_attrs;
         ++j) { // Iterate through attributes (cx,cy,w,h, cls_scores)
      transposed_output[static_cast<size_t>(i) * num_attrs + j] =
          output_fp32_buffer_host[static_cast<size_t>(j) * num_dets_yolo + i];
    }
  }

  std::vector<Detection> detections;
  for (int i = 0; i < num_dets_yolo; ++i) {
    const float *det_data =
        transposed_output.data() + static_cast<size_t>(i) * num_attrs;

    float max_class_score = 0.f;
    int current_class_id = -1;

    // First 4 elements are bbox, rest are class scores
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
        float w = det_data[2];
        float h = det_data[3];

        int x_min = static_cast<int>(std::round(cx - w / 2.f));
        int y_min = static_cast<int>(std::round(cy - h / 2.f));
        // width and height for cv::Rect
        int box_w = static_cast<int>(std::round(w));
        int box_h = static_cast<int>(std::round(h));

        detections.push_back(
            {{x_min, y_min, box_w, box_h}, max_class_score, current_class_id});
      }
    }
  }

  std::vector<Detection> nms_results = nms(detections, NMS_IOU_THRESHOLD);

  std::cout
      << "Detected '" << TARGET_CLASS_NAME
      << "' BBOXes (x_min, y_min, x_max, y_max) in original image coordinates:"
      << std::endl;
  if (nms_results.empty()) {
    std::cout << "  No " << TARGET_CLASS_NAME
              << " detected meeting the criteria." << std::endl;
  }

  for (const auto &det : nms_results) {
    cv::Rect box_model_scale = det.bbox;

    float x_min_padded_img = static_cast<float>(box_model_scale.x);
    float y_min_padded_img = static_cast<float>(box_model_scale.y);
    float x_max_padded_img =
        static_cast<float>(box_model_scale.x + box_model_scale.width);
    float y_max_padded_img =
        static_cast<float>(box_model_scale.y + box_model_scale.height);

    // Remove padding effect
    float x_min_resized_img = x_min_padded_img - pad_w_val;
    float y_min_resized_img = y_min_padded_img - pad_h_val;
    float x_max_resized_img = x_max_padded_img - pad_w_val;
    float y_max_resized_img = y_max_padded_img - pad_h_val;

    // Scale back to original image coordinates
    int final_x_min =
        static_cast<int>(std::round(x_min_resized_img / scale_factor));
    int final_y_min =
        static_cast<int>(std::round(y_min_resized_img / scale_factor));
    int final_x_max =
        static_cast<int>(std::round(x_max_resized_img / scale_factor));
    int final_y_max =
        static_cast<int>(std::round(y_max_resized_img / scale_factor));

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
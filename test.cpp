#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace nvinfer1;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

std::vector<char> loadEngineFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

int align32(int x) {
    return (x + 31) / 32 * 32;
}

struct Detection {
    cv::Rect box;
    float confidence;
};

std::vector<Detection> nms(const std::vector<Detection>& detections, float iou_thresh) {
    std::vector<Detection> result;
    auto sorted = detections;
    std::sort(sorted.begin(), sorted.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

    std::vector<bool> removed(sorted.size(), false);
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (removed[i]) continue;
        result.push_back(sorted[i]);
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            if (removed[j]) continue;
            float iou = (sorted[i].box & sorted[j].box).area() / float((sorted[i].box | sorted[j].box).area());
            if (iou > iou_thresh)
                removed[j] = true;
        }
    }
    return result;
}

int main() {
    const std::string engine_path = "/opt/models/yolo/engine/yolov8n_dynamic_fp16.engine";
    const std::string image_path = "input.jpg";

    // Load engine
    auto engine_data = loadEngineFile(engine_path);
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    IExecutionContext* context = engine->createExecutionContext();

    // Load and preprocess image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) return std::cerr << "Image load failed.\n", -1;

    int input_h = align32(img.rows);
    int input_w = align32(img.cols);
    float scale_x = input_w / float(img.cols);
    float scale_y = input_h / float(img.rows);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_w, input_h));
    resized.convertTo(resized, CV_32FC3, 1 / 255.0);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    std::vector<float> chw(3 * input_h * input_w);
    std::vector<cv::Mat> ch(3);
    for (int i = 0; i < 3; ++i)
        ch[i] = cv::Mat(input_h, input_w, CV_32FC1, chw.data() + i * input_h * input_w);
    cv::split(resized, ch);

    // Setup TensorRT bindings
    void* buffers[2];
    int inputIndex = engine->getBindingIndex("images");
    int outputIndex = engine->getBindingIndex("output0");

    size_t input_size = 1 * 3 * input_h * input_w * sizeof(float);
    cudaMalloc(&buffers[inputIndex], input_size);
    cudaMemcpy(buffers[inputIndex], chw.data(), input_size, cudaMemcpyHostToDevice);

    context->setBindingDimensions(inputIndex, Dims4{1, 3, input_h, input_w});
    if (!context->allInputDimensionsSpecified()) return std::cerr << "Binding dims incomplete\n", -1;

    Dims output_dims = context->getBindingDimensions(outputIndex);
    size_t output_size = 1;
    for (int i = 0; i < output_dims.nbDims; ++i)
        output_size *= output_dims.d[i];

    std::vector<float> output(output_size);
    cudaMalloc(&buffers[outputIndex], output_size * sizeof(float));

    if (!context->enqueueV2(buffers, 0, nullptr)) {
        std::cerr << "Inference failed.\n";
        return -1;
    }
    cudaMemcpy(output.data(), buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Postprocess
    std::vector<Detection> dog_boxes;
    int num_preds = output_dims.d[2];
    int num_classes = output_dims.d[1] - 5;

    for (int i = 0; i < num_preds; ++i) {
        float* ptr = &output[i * (5 + num_classes)];
        float obj = ptr[4];
        if (obj < 0.01f) continue; // quick filter

        int best_class = -1;
        float best_score = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            float s = ptr[5 + c];
            if (s > best_score) {
                best_score = s;
                best_class = c;
            }
        }

        float conf = obj * best_score;
        if (best_class != 16 || conf < 0.25f) continue;  // ✅ 只保留“狗”

        float cx = ptr[0], cy = ptr[1], w = ptr[2], h = ptr[3];
        int x = (cx - w / 2) / scale_x;
        int y = (cy - h / 2) / scale_y;
        int ww = w / scale_x;
        int hh = h / scale_y;
        dog_boxes.push_back({cv::Rect(x, y, ww, hh), conf});
    }

    auto final_dogs = nms(dog_boxes, 0.45f);

    for (const auto& det : final_dogs) {
        cv::rectangle(img, det.box, cv::Scalar(0, 255, 0), 2);
        std::string label = "dog " + std::to_string(int(det.confidence * 100)) + "%";
        cv::putText(img, label, det.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
    }

    // Output
    cv::imwrite("output.jpg", img);
    std::cout << "[INFO] Saved result: output.jpg ";
    if (final_dogs.empty())
        std::cout << "(no dog detected)\n";
    else
        std::cout << "(" << final_dogs.size() << " dog(s) detected)\n";

    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}


// 修复后的 doInference 方法
void YoloInferencer::doInference(const InferenceTask &task) {
  auto sigmoid = [](float x) { return 1.f / (1.f + std::exp(-x)); };

  for (size_t frame_idx = 0; frame_idx < task.frames.size(); ++frame_idx) {
    const Mat &frame = task.frames[frame_idx];

    float scale = 1.0f;
    Point pad;
    Mat padded =
        letterbox(frame, input_size, 32, Scalar(114, 114, 114), &scale, &pad);

    // 移除调试图片保存，只在检测到目标时保存

    Mat blob;
    blobFromImage(padded, blob, 1.0 / 255.0, input_size, Scalar(), true, false);
    net.setInput(blob);
    Mat output = net.forward();

    // 修正：正确解析YOLO输出格式
    const int num_preds = output.size[2];
    const int num_attrs = output.size[1];

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < num_preds; ++i) {
      float *data = output.ptr<float>(0, 0) + i * num_attrs;

      // 修正：正确读取YOLO输出
      // data[0-3]: 边界框坐标 (center_x, center_y, width, height)
      // data[4]: objectness分数
      // data[5+]: 类别分数

      float objectness = sigmoid(data[4]);

      // 找到最高分的类别
      float max_class_score = -1e9;
      int class_id = -1;
      for (int c = 0; c < num_classes; ++c) {
        float cls_score = data[5 + c]; // 修正：从索引5开始读取类别分数
        if (cls_score > max_class_score) {
          max_class_score = cls_score;
          class_id = c;
        }
      }

      // 修正：正确计算最终置信度
      float class_score = sigmoid(max_class_score);
      float final_conf = objectness * class_score;

      // 检查是否为目标物体且置信度足够
      if (final_conf >= task.confidence_thresh && class_id >= 0 &&
          class_id < static_cast<int>(class_names.size()) &&
          class_names[class_id] == task.object_name) {
        // 计算边界框（从归一化坐标转换为像素坐标）
        float center_x = data[0] * input_size.width;
        float center_y = data[1] * input_size.height;
        float width = data[2] * input_size.width;
        float height = data[3] * input_size.height;

        float x = center_x - width / 2;
        float y = center_y - height / 2;

        boxes.push_back(cv::Rect(static_cast<int>(x), static_cast<int>(y),
                                 static_cast<int>(width),
                                 static_cast<int>(height)));
        confidences.push_back(final_conf);
        class_ids.push_back(class_id);
      }
    }

    // 应用非极大值抑制
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, task.confidence_thresh, 0.4f,
                      indices);

    // 如果检测到目标物体，保存带边界框的原图
    if (!indices.empty()) {
      // 获取保存路径
      const char *save_path = std::getenv("YOLO_IMAGE_PATH");
      if (save_path) {
        // 复制原图用于标注
        Mat result_img = frame.clone();

        // 在原图上绘制边界框
        for (int idx : indices) {
          // 将检测框坐标从letterbox后的图像转换回原图
          cv::Rect detection_box = boxes[idx];

          // 转换坐标：从letterbox图像坐标转回原图坐标
          float x_orig = (detection_box.x - pad.x) / scale;
          float y_orig = (detection_box.y - pad.y) / scale;
          float w_orig = detection_box.width / scale;
          float h_orig = detection_box.height / scale;

          // 确保坐标在原图范围内
          x_orig = std::max(0.0f, std::min(x_orig, (float)frame.cols));
          y_orig = std::max(0.0f, std::min(y_orig, (float)frame.rows));
          w_orig = std::min(w_orig, (float)frame.cols - x_orig);
          h_orig = std::min(h_orig, (float)frame.rows - y_orig);

          cv::Rect orig_box(static_cast<int>(x_orig), static_cast<int>(y_orig),
                            static_cast<int>(w_orig), static_cast<int>(h_orig));

          // 绘制边界框
          cv::rectangle(result_img, orig_box, cv::Scalar(0, 255, 0), 2);

          // 添加标签文本
          std::string label =
              class_names[class_ids[idx]] + ": " +
              std::to_string(static_cast<int>(confidences[idx] * 100)) + "%";

          int baseline = 0;
          cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                               0.5, 1, &baseline);

          // 绘制文本背景
          cv::rectangle(
              result_img,
              cv::Point(orig_box.x, orig_box.y - text_size.height - 5),
              cv::Point(orig_box.x + text_size.width, orig_box.y),
              cv::Scalar(0, 255, 0), -1);

          // 绘制文字
          cv::putText(result_img, label, cv::Point(orig_box.x, orig_box.y - 5),
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        // 保存图片
        std::string save_filename = std::string(save_path) + "/detection_gop" +
                                    std::to_string(task.gopIdx) + "_frame_" +
                                    std::to_string(frame_idx) + ".jpg";

        if (cv::imwrite(save_filename, result_img)) {
          std::cout << "[YOLO] Saved detection image: " << save_filename
                    << std::endl;
        } else {
          std::cerr << "[YOLO] Failed to save image: " << save_filename
                    << std::endl;
        }
      } else {
        std::cerr << "[YOLO] YOLO_IMAGE_PATH environment variable not set"
                  << std::endl;
      }
    }

    // 输出最终检测结果
    for (int idx : indices) {
      std::cout << "[YOLO] GOP: " << task.gopIdx << ", Frame: " << frame_idx
                << ", Confidence: " << confidences[idx]
                << ", Class: " << class_names[class_ids[idx]] << ", Box: ("
                << boxes[idx].x << "," << boxes[idx].y << ","
                << boxes[idx].width << "," << boxes[idx].height << ")"
                << std::endl;
    }
  }
}
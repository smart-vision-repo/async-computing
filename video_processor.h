#ifndef VIDEO_PROCESSOR_H
#define VIDEO_PROCESSOR_H
#include "inference.hpp"
#include "packet_decoder.h"
#include "tensor_inferencer.hpp"
#include "yolo_inferencer.h"
#include <atomic>
#include <condition_variable>
#include <optional>
#include <string>
#include <vector>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

// 声明 VideoProcessor 类
class VideoProcessor {
public:
  VideoProcessor(const std::string &video_file_name,
                 const std::string &object_name, float confidence,
                 int interval);
  ~VideoProcessor();
  int process();

private:
  bool initialize();
  void initInferThead();
  void handleInferenceResult(const std::vector<InferenceResult> &result);
  void onDecoded(std::vector<cv::Mat> &&frames, int gopId);
  void add_av_packet_to_list(std::vector<AVPacket *> **packages,
                             const AVPacket *packet);
  std::vector<AVPacket *>
  get_packets_for_decoding(std::vector<AVPacket *> *packages,
                           int last_frame_index);
  void clear_av_packets(std::vector<AVPacket *> *packages);
  PacketDecoder decoder;
  std::string video_file_name;
  std::string object_name;
  float confidence;
  int interval;
  int success_decoded_frames;
  std::queue<InferenceInput> infer_inputs;
  std::mutex infer_mutex;
  std::condition_variable infer_cv;
  std::thread infer_thread;
  std::atomic<bool> stop_infer_thread;
  YoloInferencer inferencer;
  std::optional<TensorInferencer>
      tensor_inferencer; // Added lazy-loaded TensorInferencer
  std::atomic<int> remaining_decode_tasks{0};
  std::atomic<int> remaining_infer_tasks{0};
  std::mutex task_mutex;
  std::condition_variable task_cv;
  AVFormatContext *fmtCtx = nullptr;
  int video_stream_index = -1;
  int frame_width = 0;
  int frame_heigh = 0;
  InferenceCallback callback;
};

#endif // VIDEO_PROCESSOR_H
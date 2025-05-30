#ifndef VIDEO_PROCESSOR_H
#define VIDEO_PROCESSOR_H

#include "packet_decoder.h"
#include "tensor_inferencer.hpp"
#include "yolo_inferencer.h"
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
  void onDecoded(std::vector<cv::Mat> &&frames, int gopId);
  void add_av_packet_to_list(std::vector<AVPacket *> **packages,
                             const AVPacket *packet);
  std::vector<AVPacket *>
  get_packets_for_decoding(std::vector<AVPacket *> *packages,
                           int last_frame_index);
  void clear_av_packets(std::vector<AVPacket *> *packages);
  int initialize();
  PacketDecoder decoder;
  std::string video_file_name;
  std::string object_name;
  int interval;
  int success_decoded_frames;
  float confidence;
  std::queue<InferenceInput> infer_inputs;
  std::mutex infer_mutex;
  std::condition_variable infer_cv;
  std::thread infer_thread;
  std::atomic<bool> stop_infer_thread;
  YoloInferencer yolo_inferencer;
  std::optional<TensorInferencer> tensor_inferencer;
  AVFormatContext *fmtCtx = nullptr;
  int frame_heigh = 0;
  int frame_width = 0;
  int videoStream = -1;
};

#endif // VIDEO_PROCESSOR_H

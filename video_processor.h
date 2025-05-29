#ifndef VIDEO_PROCESSOR_H
#define VIDEO_PROCESSOR_H

#include "packet_decoder.h"
#include "yolo_inferencer.h"
#include <string>
#include <vector>
extern "C" {
#include <libavcodec/avcodec.h>
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
  PacketDecoder decoder;
  std::string video_file_name;
  std::string object_name;
  float confidence;
  int interval;
  int success_decoded_frames = 0;
  YoloInferencer inferencer;
};

#endif // VIDEO_PROCESSOR_H

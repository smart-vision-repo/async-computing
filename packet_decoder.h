// packet_decoder.h
#pragma once
#include "yolo_inferencer.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

class PacketDecoder {
public:
  using DecodeCallback =
      std::function<void(std::vector<cv::Mat> &&, int gopId)>;

  PacketDecoder(std::string video_file_name);
  ~PacketDecoder();

  void decode(const std::vector<AVPacket *> &pkts, int interval, int gopId,
              DecodeCallback callback);
  void waitForAllTasks(); // 等待所有任务完成并关闭线程池
  void reset();

private:
  struct DecodeTask {
    std::vector<AVPacket *> pkts;
    int interval;
    int gopId;
    DecodeCallback callback;
  };

  std::string video_file_name;
  int vidIdx;
  AVFormatContext *fmtCtx;
  AVCodec *codec;
  AVCodecParameters *codecpar;
  YoloInferencer *inferencer;

  std::vector<std::thread> workers;
  std::queue<DecodeTask> taskQueue;
  std::mutex queueMutex;
  std::condition_variable queueCond;
  std::condition_variable doneCond;
  bool stopThreads;
  std::atomic<int> activeTasks;

  void workerLoop();
  AVCodecContext *cloneDecoderContext();
  void decodeTask(DecodeTask task, AVCodecContext *ctx);

  bool initialize();
};
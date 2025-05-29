// packet_decoder.cpp
#include "packet_decoder.h"
#include <algorithm>
#include <iostream>

PacketDecoder::PacketDecoder(std::string video_file_name)
    : video_file_name(video_file_name), vidIdx(-1), fmtCtx(nullptr),
      codec(nullptr), codecpar(nullptr), stopThreads(false) {
  if (!initialize()) {
    throw std::runtime_error("Failed to initialize PacketDecoder");
  }

  for (int i = 0; i < 4; ++i) {
    workers.emplace_back(&PacketDecoder::workerLoop, this);
  }
}

PacketDecoder::~PacketDecoder() {
  {
    std::lock_guard<std::mutex> lock(queueMutex);
    stopThreads = true;
  }
  queueCond.notify_all();

  for (auto &t : workers) {
    if (t.joinable())
      t.join();
  }

  if (fmtCtx)
    avformat_close_input(&fmtCtx);
}

bool PacketDecoder::initialize() {
  if (avformat_open_input(&fmtCtx, video_file_name.c_str(), nullptr, nullptr) != 0) {
    std::cerr << "Failed to open video file: " << video_file_name << std::endl;
    return false;
  }
  if (avformat_find_stream_info(fmtCtx, nullptr) < 0)
    return false;

  for (unsigned i = 0; i < fmtCtx->nb_streams; ++i) {
    if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      vidIdx = i;
      codec = avcodec_find_decoder(fmtCtx->streams[i]->codecpar->codec_id);
      codecpar = fmtCtx->streams[i]->codecpar;
      return codec != nullptr;
    }
  }
  return false;
}

void PacketDecoder::decode(const std::vector<AVPacket *> &pkts, int interval,
                           int gopId, DecodeCallback callback) {
  std::vector<AVPacket *> copied_pkts;
  for (const auto *pkt : pkts) {
    if (!pkt) continue;
    AVPacket *clone = av_packet_alloc();
    if (clone && av_packet_ref(clone, pkt) == 0) {
      copied_pkts.push_back(clone);
    }
  }

  {
    std::lock_guard<std::mutex> lock(queueMutex);
    taskQueue.push(DecodeTask{copied_pkts, interval, gopId, callback});
    ++activeTasks;
  }
  queueCond.notify_one();
}

void PacketDecoder::workerLoop() {
  AVCodecContext *localCtx = cloneDecoderContext();
  if (!localCtx)
    return;

  while (true) {
    DecodeTask task;
    {
      std::unique_lock<std::mutex> lock(queueMutex);
      queueCond.wait(lock, [this]() { return stopThreads || !taskQueue.empty(); });
      if (stopThreads && taskQueue.empty())
        break;
      task = std::move(taskQueue.front());
      taskQueue.pop();
    }
    decodeTask(task, localCtx);
  }

  avcodec_free_context(&localCtx);
}

AVCodecContext *PacketDecoder::cloneDecoderContext() {
  AVCodecContext *ctx = avcodec_alloc_context3(codec);
  if (!ctx)
    return nullptr;
  if (avcodec_parameters_to_context(ctx, codecpar) < 0) {
    avcodec_free_context(&ctx);
    return nullptr;
  }
  if (avcodec_open2(ctx, codec, nullptr) < 0) {
    avcodec_free_context(&ctx);
    return nullptr;
  }
  return ctx;
}

void PacketDecoder::decodeTask(DecodeTask task, AVCodecContext *ctx) {
  std::cout << "[Decode] GOP " << task.gopId << " started\n";

  std::vector<cv::Mat> decoded;
  AVFrame *frame = av_frame_alloc();

  avcodec_flush_buffers(ctx);

  for (AVPacket *pkt : task.pkts) {
    if (!pkt || pkt->size == 0 || !pkt->data) {
      std::cerr << "[Warn] Empty packet in GOP " << task.gopId << "\n";
      continue;
    }

    pkt->stream_index = vidIdx;
    std::cout << "[Debug] Sending pkt of size " << pkt->size << "\n";

    if (avcodec_send_packet(ctx, pkt) < 0)
      continue;

    while (true) {
      int ret = avcodec_receive_frame(ctx, frame);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
        break;
      if (ret < 0)
        break;

      if (frame->format == AV_PIX_FMT_YUV420P && frame->data[0]) {
        try {
          cv::Mat yuv(frame->height + frame->height / 2, frame->width, CV_8UC1, frame->data[0]);
          cv::Mat mat;
          cv::cvtColor(yuv, mat, cv::COLOR_YUV2BGR_I420);
          decoded.push_back(std::move(mat));
        } catch (...) {
          std::cerr << "[Error] Failed to convert frame\n";
        }
      }
      av_frame_unref(frame);
    }
  }

  avcodec_send_packet(ctx, nullptr);
  while (true) {
    int ret = avcodec_receive_frame(ctx, frame);
    if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN))
      break;
    if (ret < 0)
      break;

    if (frame->format == AV_PIX_FMT_YUV420P && frame->data[0]) {
      try {
        cv::Mat yuv(frame->height + frame->height / 2, frame->width, CV_8UC1, frame->data[0]);
        cv::Mat mat;
        cv::cvtColor(yuv, mat, cv::COLOR_YUV2BGR_I420);
        decoded.push_back(std::move(mat));
      } catch (...) {
        std::cerr << "[Error] Failed to convert frame (flush)\n";
      }
    }
    av_frame_unref(frame);
  }

  av_frame_free(&frame);

  std::vector<cv::Mat> filtered;
  for (size_t i = 0; i < decoded.size(); i += task.interval) {
    filtered.push_back(std::move(decoded[i]));
  }

  try {
    task.callback(std::move(filtered), task.gopId);
  } catch (const std::exception &e) {
    std::cerr << "[Error] Callback exception in GOP " << task.gopId << ": " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "[Error] Unknown exception in callback for GOP " << task.gopId << std::endl;
  }

  for (AVPacket* pkt : task.pkts) {
    if (pkt) {
      av_packet_unref(pkt);
      av_packet_free(&pkt);
    }
  }

  {
    std::lock_guard<std::mutex> lock(queueMutex);
    --activeTasks;
    if (activeTasks == 0 && taskQueue.empty()) {
      doneCond.notify_all();
    }
  }
}

void PacketDecoder::reset() {}

void PacketDecoder::waitForAllTasks() {
  std::unique_lock<std::mutex> lock(queueMutex);
  doneCond.wait(lock, [this]() { return taskQueue.empty() && activeTasks == 0; });

  stopThreads = true;
  queueCond.notify_all();
  lock.unlock();

  for (auto &t : workers) {
    if (t.joinable())
      t.join();
  }
}

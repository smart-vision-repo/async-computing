// video_processor.cpp
#include "video_processor.h"
#include "inference_input.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}
#include <opencv2/opencv.hpp>

VideoProcessor::~VideoProcessor() {
  stop_infer_thread = true;
  infer_cv.notify_all();
  if (infer_thread.joinable()) {
    infer_thread.join();
  }
}

VideoProcessor::VideoProcessor(const std::string &video_file_name,
                               const std::string &object_name, float confidence,
                               int interval)
    : decoder(const_cast<std::string &>(video_file_name)),
      video_file_name(video_file_name), object_name(object_name),
      confidence(confidence), interval(interval), success_decoded_frames(0),
      stop_infer_thread(false) {
  int ret = initialize();
  if (ret != 0) {
    std::cerr << "Initialization failed. Exiting program." << std::endl;
    std::exit(ret);
  }
  // 启动独立推理线程
  // infer_thread = std::thread([this]() {
  //   while (!stop_infer_thread) {
  //     std::unique_lock<std::mutex> lock(infer_mutex);
  //     infer_cv.wait(lock, [this]() {
  //       return !infer_inputs.empty() || stop_infer_thread;
  //     });

  //     while (!infer_inputs.empty()) {
  //       InferenceInput input = std::move(infer_inputs.front());
  //       infer_inputs.pop();
  //       lock.unlock();
  //       if (tensor_inferencer) {
  //         tensor_inferencer->infer(input);
  //       } else {
  //         std::cerr << "[ERROR] TensorInferencer not initialized." <<
  //         std::endl;
  //       }
  //       lock.lock();
  //     }
  //   }
  // });
}

void VideoProcessor::onDecoded(std::vector<cv::Mat> &&frames, int gopId) {
  std::cout << gopId << "," << frames.size() << std::endl;
  InferenceInput input;
  input.decoded_frames = std::move(frames);
  input.object_name = object_name;
  input.confidence_thresh = confidence;
  input.gopIdx = gopId;

  {
    std::lock_guard<std::mutex> lock(infer_mutex);
    infer_inputs.push(std::move(input));
  }
  infer_cv.notify_all();
}

int VideoProcessor::process() {
  AVPacket *packet = av_packet_alloc();
  if (!packet) {
    std::cerr << "Could not allocate AVPacket" << std::endl;
    avformat_close_input(&fmtCtx);
    return -1;
  }

  int frame_idx = 0, gop_idx = 0, frame_idx_in_gop = 0;
  int hits = 0, pool = 0;
  int total_hits = 0, decoded_frames = 0, skipped_frames = 0,
      total_packages = 0;
  std::vector<AVPacket *> *pkts = new std::vector<AVPacket *>();
  std::vector<std::vector<AVPacket *>> all_pkts;

  while (av_read_frame(fmtCtx, packet) >= 0) {
    if (packet->stream_index == videoStream) {
      frame_idx++;
      if (frame_idx > 1 && (frame_idx - 1) % interval == 0) {
        hits++;
      }
      bool is_key_frame = (packet->flags & AV_PKT_FLAG_KEY);
      if (is_key_frame) {
        int last_frame_in_gop = 0;
        if (hits > 0) {
          skipped_frames += pool;
          last_frame_in_gop = hits * interval - pool;
          decoded_frames += last_frame_in_gop;
          total_hits += hits;
          pool = frame_idx_in_gop - last_frame_in_gop;
          std::vector<AVPacket *> decoding_pkts =
              get_packets_for_decoding(pkts, last_frame_in_gop);
          decoder.decode(decoding_pkts, interval, gop_idx,
                         [this](std::vector<cv::Mat> frames, int gopId) {
                           this->onDecoded(std::move(frames), gopId);
                         });
          all_pkts.push_back(std::move(decoding_pkts));
        } else {
          pool += frame_idx_in_gop;
        }
        frame_idx_in_gop = 0;
        hits = 0;
        gop_idx++;
        clear_av_packets(pkts);
        add_av_packet_to_list(&pkts, packet);
      } else {
        if (pkts->size() > 0) {
          add_av_packet_to_list(&pkts, packet);
        }
      }
      frame_idx_in_gop++;
      av_packet_unref(packet);
    }
  }

  int last_frame_in_gop = 0;
  if (hits > 0) {
    std::vector<AVPacket *> decoding_pkts =
        get_packets_for_decoding(pkts, last_frame_in_gop);
    decoder.decode(decoding_pkts, interval, gop_idx,
                   [this](std::vector<cv::Mat> frames, int gopId) {
                     this->onDecoded(std::move(frames), gopId);
                   });
    all_pkts.push_back(std::move(decoding_pkts));
    skipped_frames += pool;
    last_frame_in_gop = hits * interval - pool;
    if (last_frame_in_gop > 0) {
      decoded_frames += last_frame_in_gop;
      total_packages += last_frame_in_gop;
    }
    total_hits += hits;
    pool = frame_idx_in_gop - last_frame_in_gop;
  } else {
    pool += frame_idx_in_gop;
  }

  decoder.waitForAllTasks();

  for (auto &pkts : all_pkts) {
    clear_av_packets(&pkts);
  }

  skipped_frames += pool;
  av_packet_free(&packet);
  avformat_close_input(&fmtCtx);
  clear_av_packets(pkts);
  delete pkts;

  std::cout << "-------------------" << std::endl;
  float percentage =
      (frame_idx > 0) ? (decoded_frames * 100.0f / frame_idx) : 0.0f;
  std::cout << "total gop: " << gop_idx << std::endl
            << "interval: " << interval << std::endl
            << "total_packages: " << total_packages << std::endl
            << "decoded frames: " << decoded_frames << std::endl
            << "skipped frames: " << skipped_frames << std::endl
            << "discrepancies: " << frame_idx - decoded_frames - skipped_frames
            << std::endl
            << "percentage: " << percentage << "%" << std::endl
            << "successfully decoded: " << success_decoded_frames << std::endl
            << "extracted frames: " << total_hits << std::endl;

  return 0;
}

int VideoProcessor::initialize() {
  const char *video_file_path = video_file_name.c_str();
  AVFormatContext *fmtCtx = nullptr;
  if (avformat_open_input(&fmtCtx, video_file_path, nullptr, nullptr) < 0) {
    std::cerr << "Could not open video file: " << video_file_path << std::endl;
    return -1;
  }

  if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
    std::cerr << "Could not get stream info" << std::endl;
    avformat_close_input(&fmtCtx);
    return -1;
  }

  int videoStream = -1;
  for (unsigned int i = 0; i < fmtCtx->nb_streams; i++) {
    if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      videoStream = i;
      break;
    }
  }

  if (videoStream == -1) {
    std::cerr << "No video stream found" << std::endl;
    avformat_close_input(&fmtCtx);
    return -1;
  }

  AVCodecParameters *codecpar = fmtCtx->streams[videoStream]->codecpar;
  frame_width = (codecpar->width + 31) / 32 * 32;
  frame_heigh = (codecpar->height + 31) / 32 * 32;
  tensor_inferencer.emplace(frame_heigh, frame_width);
  std::cout << "Video Width: " << codecpar->width
            << ", Height: " << codecpar->height << std::endl;
  return 0;
}

void VideoProcessor::add_av_packet_to_list(std::vector<AVPacket *> **packages,
                                           const AVPacket *packet) {
  if (!packet)
    return;
  if (!*packages) {
    *packages = new std::vector<AVPacket *>();
  }
  AVPacket *cloned = av_packet_clone(packet);
  if (cloned) {
    (*packages)->push_back(cloned);
  }
}

std::vector<AVPacket *>
VideoProcessor::get_packets_for_decoding(std::vector<AVPacket *> *packages,
                                         int last_frame_index) {
  std::vector<AVPacket *> results;
  if (!packages)
    return results;
  for (int i = 0; i < last_frame_index && i < packages->size(); i++) {
    AVPacket *pkt = av_packet_clone((*packages)[i]);
    if (pkt) {
      results.push_back(pkt);
    }
  }
  return results;
}

void VideoProcessor::clear_av_packets(std::vector<AVPacket *> *pkts) {
  for (AVPacket *pkt : *pkts) {
    if (pkt) {
      av_packet_unref(pkt);
      av_packet_free(&pkt);
    }
  }
  pkts->clear();
}

#include "video_processor.h"
#include "models.hpp"
#include "tensor_inferencer.hpp"
// #include "yolo_inferencer.h"

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
  stop_infer_thread = true;      // Assuming this controls a thread not shown in
                                 // constructor/destructor logic fully
  if (infer_thread.joinable()) { // infer_thread is not initialized/started in
                                 // the provided code
    infer_thread.join();
  }
  if (fmtCtx) {
    avformat_close_input(&fmtCtx);
    fmtCtx = nullptr;
  }
}

VideoProcessor::VideoProcessor(int order_id, const std::string &video_file_name,
                               const std::string &object_name, float confidence,
                               int interval, int start_frame_index,
                               MessageProxy &messageProxy)
    : task_id_(order_id), video_file_name_(video_file_name),
      object_name_(object_name), confidence_(confidence), interval_(interval),
      frame_idx_(start_frame_index), stop_infer_thread(false), fmtCtx(nullptr),
      video_stream_index(-1), messageProxy_(messageProxy) {

  if (!initialize()) {
    throw std::runtime_error("Failed to initialize video processor");
  }
}

bool VideoProcessor::initialize() {
  setBatchSize();
  const char *video_file_path = video_file_name_.c_str();
  if (avformat_open_input(&fmtCtx, video_file_path, nullptr, nullptr) < 0) {
    std::cerr << "Could not open video file: " << video_file_path << std::endl;
    return false;
  }

  if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
    std::cerr << "Could not get stream info" << std::endl;
    avformat_close_input(&fmtCtx);
    fmtCtx = nullptr;
    return false;
  }

  for (unsigned int i = 0; i < fmtCtx->nb_streams; i++) {
    if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_stream_index = static_cast<int>(i);
      break;
    }
  }

  if (video_stream_index == -1) {
    std::cerr << "No video stream found" << std::endl;
    avformat_close_input(&fmtCtx);
    fmtCtx = nullptr;
    return false;
  }

  AVStream *video_stream = fmtCtx->streams[video_stream_index];
  AVCodecParameters *codec_params = video_stream->codecpar;
  frame_width = codec_params->width;
  frame_heigh = codec_params->height;
  if (frame_heigh == 0 || frame_width == 0) {
    std::cerr << "Failed to get frame dimensions" << std::endl;
    avformat_close_input(&fmtCtx);
    fmtCtx = nullptr;
    return false;
  }

  DecoderCallback docoderCallback = [this](std::vector<cv::Mat> &frames,
                                           int gopId, int disposedFrames) {
    this->onDecoderCallback(frames, gopId, disposedFrames);
  };
  decoder.emplace(
      video_file_name_,
      docoderCallback); // Assuming PacketDecoder constructor matches

  InferResultCallback resultCallabck = [this](const InferenceResult &result) {
    this->onInferResultCallback(result);
  };

  InferPackCallback packCallback = [this](const int &result) {
    this->onInferPackCallback(result);
  };
  tensor_inferencer.emplace(task_id_, frame_heigh, frame_width, object_name_,
                            interval_, confidence_, resultCallabck,
                            packCallback);
  std::cout << "Video width: " << frame_width << ", height: " << frame_heigh
            << std::endl;

  return true;
}

int VideoProcessor::process() {
  AVPacket *packet = av_packet_alloc();
  if (!packet) {
    std::cerr << "Could not allocate AVPacket" << std::endl;
    return -1;
  }

  int gop_idx = 0, frame_idx_in_gop = 0;
  int hits = 0, pool = 0;
  int total_hits = 0;  // 跳帧条件下，累计取到的帧数.
  int decoded_frames = 0; // 送给decoder包中解码得到的帧数.
  int skipped_frames = 0; // 跳过的帧数.
  int total_packages = 0; // GOP包的数量，即关键帧的数量.
  std::vector<AVPacket *> *pkts = new std::vector<AVPacket *>();
  std::vector<std::vector<AVPacket *>> all_pkts; // Used for final cleanup

  while (av_read_frame(fmtCtx, packet) >= 0) {
    if (packet->stream_index == video_stream_index) {
      frame_idx_++;
      if (frame_idx_ > 1 && (frame_idx_ - 1) % interval_ == 0) {
        hits++;
      }
      // 获取当前帧是否为key frame.
      bool is_key_frame = (packet->flags & AV_PKT_FLAG_KEY);
      if (is_key_frame) {
        int last_frame_in_gop = 0; // To be calculated
        if (hits > 0) {
          skipped_frames += pool;
          last_frame_in_gop = hits * interval_ - pool;
          if (last_frame_in_gop > 0) {
            decoded_frames += last_frame_in_gop; // Statistical count
            std::vector<AVPacket *> decoding_pkts = get_packets_for_decoding(pkts, last_frame_in_gop);
            if (decoding_pkts.empty()) {
              clear_av_packets(&decoding_pkts);
            } else {
              decoder->decode(decoding_pkts, interval_, frame_idx_, frame_idx_in_gop);
              {
                std::lock_guard<std::mutex> lock(task_mutex);
                remaining_decode_tasks++;
              }
              all_pkts.push_back(std::move(decoding_pkts));
            }
          }
          total_hits += hits; // Statistical count
          pool = frame_idx_in_gop - last_frame_in_gop;
        } else {
          pool += frame_idx_in_gop;
        }
        frame_idx_in_gop = 0;
        hits = 0;
        gop_idx++;
        clear_av_packets(pkts);               // Clear for the new GOP
        add_av_packet_to_list(&pkts, packet); // Add current key frame
      } else {
        if (pkts->size() > 0 ||
            is_key_frame) { // ensure pkts list starts with a key frame
          add_av_packet_to_list(&pkts, packet);
        }
      }
      frame_idx_in_gop++;
    }
    av_packet_unref(packet); // Unref packet after processing
  }

  // Process the last GOP (remaining packets)
  int last_frame_in_gop = 0;
  if (hits > 0) {
    last_frame_in_gop = hits * interval_ - pool;
    if (last_frame_in_gop > 0) {
      std::vector<AVPacket *> decoding_pkts =
          get_packets_for_decoding(pkts, last_frame_in_gop);
      if (!decoding_pkts.empty()) {
        decoder->decode(decoding_pkts, interval_, frame_idx_, frame_idx_in_gop); 
        {
          std::lock_guard<std::mutex> lock(task_mutex);
          remaining_decode_tasks++;
        }
        all_pkts.push_back(std::move(decoding_pkts)); 
      } else {
        clear_av_packets(&decoding_pkts);
      }
      decoded_frames += last_frame_in_gop; 
      total_packages += last_frame_in_gop;
    }
    total_hits += hits; // Statistical count
    pool = frame_idx_in_gop -
           last_frame_in_gop; // Update pool with remaining from last GOP
  } else {
    pool += frame_idx_in_gop;
  }
  skipped_frames += pool; // Add final pool to skipped frames

  // Wait for all decoding and inference tasks to complete using a single lock
  // and CV
  std::cout << "\nWaiting for all tasks to complete..." << std::endl;
  std::unique_lock<std::mutex> lock(
      task_mutex); // Use task_mutex for both conditions

  // Wait until remaining_decode_tasks is 0
  task_cv.wait(lock, [this]() {
    std::cout << "\rWaiting for decode tasks: " << remaining_decode_tasks.load()
              << "   " << std::flush;
    return remaining_decode_tasks.load() == 0;
  });

  std::cout << "\nAll decode tasks completed." << std::endl;
  tensor_inferencer->finalizeInference(); // Call finalizeInference() only once

  // Now wait until pending_infer_tasks is also 0
  // The lock is already held.
  task_cv.wait(lock, [this]() { // Reuse the same condition variable
    std::cout << "\rWaiting for inference tasks" << std::flush;
    return pending_infer_tasks.load() <= 0;
  });

  std::cout << "\nAll inference tasks completed." << std::endl;
  lock.unlock(); // Explicitly unlock before continuing, or let destructor do
                 // it.

  // Cleanup all stored packets
  for (auto &pkt_list : all_pkts) {
    clear_av_packets(&pkt_list); // Pass address of the vector of AVPacket*
  }
  all_pkts.clear();

  av_packet_free(&packet); // Free the initial allocated packet
  clear_av_packets(pkts);  // Clear and free packets in the current pkts list
  delete pkts;             // Delete the pkts list itself
  pkts = nullptr;

  TaskDecodeInfo taskDecodeInfo = TaskDecodeInfo();
  taskDecodeInfo.taskId = task_id_;
  taskDecodeInfo.total = pool; 
  messageProxy_.sendDecodeInfo(taskDecodeInfo);

  std::cout << "-------------------" << std::endl;
  float percentage =
      (frame_idx_ > 0)
          ? (static_cast<float>(decoded_frames) * 100.0f / frame_idx_)
          : 0.0f;
  std::cout << "Decoded Frames: " << decoded_frames << std::endl
            << "Skipped Frames: " << skipped_frames << std::endl
            << "Discrepancy: " << (frame_idx_ - decoded_frames - skipped_frames) << std::endl
            << "Percentage: " << std::fixed << std::setprecision(2) << percentage << "%" << std::endl
            << "hits: " << total_hits << std::endl
            << "Total: " << decoded_frames + skipped_frames << std::endl;

  return 0;
}

void VideoProcessor::onInferResultCallback(const InferenceResult &result) {
  messageProxy_.sendInferResult(result);
}

void VideoProcessor::onInferPackCallback(const int count) {
  {
    std::cout << "\rinferenced tasks: " << count << "   " << std::endl;
    std::lock_guard<std::mutex> lock(pending_infer_mutex);
    pending_infer_tasks -= BATCH_SIZE_;
    total_inferred_frames += BATCH_SIZE_;
    // if (total_inferred_frames % BATCH_SIZE_ == 0) {
    // TaskInferInfo info = TaskInferInfo();
    // info.taskId = task_id_;
    // info.remain = pending_infer_tasks;
    // info.completed = total_inferred_frames;
    // messageProxy_.sendInferPackInfo(info);
    // }
  }
  pending_infer_cv.notify_all();
}

void VideoProcessor::onDecoderCallback(
    std::vector<cv::Mat> &received_frames,
    int gFrameIdx, int total) { // Renamed param for clarity
  const int num_decoded_this_call = received_frames.size();

  if (num_decoded_this_call > 0) {
    // InferenceInput input;
    // input.decoded_frames = std::move(received_frames);

    // input.latest_frame_index = gFrameIdx;
    // tensor_inferencer->infer(input);
    // {
    //   std::lock_guard<std::mutex> lock(pending_infer_mutex);
    //   pending_infer_tasks++;
    // }
    // pending_infer_cv.notify_all();
  }

  {
    std::lock_guard<std::mutex> lock(task_mutex);
    if (num_decoded_this_call > 0) {
      total_decoded_frames += num_decoded_this_call;
    }
    remaining_decode_tasks--; // One decode operation (this call to onDecoded)
                              // has finished
    TaskDecodeInfo taskDecodeInfo = TaskDecodeInfo();
    taskDecodeInfo.taskId = task_id_;
    taskDecodeInfo.total = total;
    messageProxy_.sendDecodeInfo(taskDecodeInfo);
  }

  task_cv.notify_all();
}

void VideoProcessor::add_av_packet_to_list(std::vector<AVPacket *> **packages,
                                           const AVPacket *packet) {
  if (!packet)
    return;
  if (!*packages) { // Ensure the vector itself is created
    *packages = new std::vector<AVPacket *>();
  }
  AVPacket *cloned_packet = av_packet_clone(packet);
  if (cloned_packet) {
    (*packages)->push_back(cloned_packet);
  } else {
    std::cerr << "Failed to clone AVPacket" << std::endl;
  }
}

std::vector<AVPacket *> VideoProcessor::get_packets_for_decoding(
    std::vector<AVPacket *> *packages,
    int num_packets_to_get) { // Renamed last_frame_index for clarity
  std::vector<AVPacket *> results;
  if (!packages || num_packets_to_get <= 0) {
    return results;
  }
  results.reserve(
      std::min(static_cast<size_t>(num_packets_to_get), packages->size()));
  for (int i = 0;
       i < num_packets_to_get && i < static_cast<int>(packages->size()); ++i) {
    AVPacket *pkt_clone = av_packet_clone((*packages)[i]);
    if (pkt_clone) {
      results.push_back(pkt_clone);
    } else {
      std::cerr << "Failed to clone AVPacket in get_packets_for_decoding"
                << std::endl;
    }
  }
  return results;
}

void VideoProcessor::clear_av_packets(std::vector<AVPacket *> *pkts) {
  if (!pkts)
    return;
  for (AVPacket *pkt : *pkts) {
    if (pkt) {
      av_packet_unref(pkt); // Unreference the packet data
      av_packet_free(&pkt); // Free the AVPacket structure itself
    }
  }
  pkts->clear(); // Clear the vector of pointers
}

void VideoProcessor::setBatchSize() {
  const char *env_batch_size_str = std::getenv("YOLO_BATCH_SIZE");
  if (env_batch_size_str) {
    try {
      int parsed_batch_size = std::stoi(env_batch_size_str);
      if (parsed_batch_size <= 0) {
        std::cerr << "[警告] BATCH_SIZE 环境变量值 (" << env_batch_size_str
                  << ") 无效。必须为正整数。将使用默认值 1。"
                  << std::endl; // Changed to warning for invalid value
        BATCH_SIZE_ = 1;
      } else {
        BATCH_SIZE_ = parsed_batch_size;
      }
    } catch (const std::invalid_argument &ia) {
      std::cerr << "[错误] BATCH_SIZE 环境变量值 (" << env_batch_size_str
                << ") 无效。无法转换为整数。将使用默认值 1。" << std::endl;
      BATCH_SIZE_ = 1;
    } catch (const std::out_of_range &oor) {
      std::cerr << "[错误] BATCH_SIZE 环境变量值 (" << env_batch_size_str
                << ") 超出范围。将使用默认值 1。" << std::endl;
      BATCH_SIZE_ = 1;
    }
  } else {
    std::cerr << "[警告] 未设置 BATCH_SIZE 环境变量。将使用默认值 1。"
              << std::endl;
    BATCH_SIZE_ = 1;
  }
  std::cout << "[初始化] 使用 BATCH_SIZE: " << BATCH_SIZE_ << std::endl;
}
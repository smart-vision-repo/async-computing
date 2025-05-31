#include "video_processor.h"
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

VideoProcessor::VideoProcessor(const std::string &video_file_name,
                               const std::string &object_name, float confidence,
                               int interval)
    : video_file_name(video_file_name), object_name(object_name),
      confidence(confidence), interval(interval), stop_infer_thread(false),
      fmtCtx(nullptr), video_stream_index(-1) {

  if (!initialize()) {
    throw std::runtime_error("Failed to initialize video processor");
  }
}

void VideoProcessor::handleInferenceResult(
    const std::vector<InferenceResult> &result) {
  {
    std::lock_guard<std::mutex> lock(pending_infer_mutex);
    // This line is kept based on your explanation that TensorInferencer's
    // callback corresponds to BATCH_SIZE_ submissions having been processed.
    // Its correctness relies on TensorInferencer (including finalizeInference)
    // managing callbacks such that this accounting results in
    // pending_infer_tasks eventually reaching <= 0.
    pending_infer_tasks -= BATCH_SIZE_;
  }
  pending_infer_cv.notify_all();
}

bool VideoProcessor::initialize() {
  setBatchSize();
  const char *video_file_path = video_file_name.c_str();
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

  docoder_callback = [this](std::vector<cv::Mat> &frames, int gopId) {
    this->onDecoded(frames, gopId);
  };
  decoder.emplace(
      video_file_name,
      docoder_callback); // Assuming PacketDecoder constructor matches

  infer_callback = [this](const std::vector<InferenceResult> &result) {
    this->handleInferenceResult(result);
  };
  tensor_inferencer.emplace(
      frame_heigh, frame_width, object_name,
      interval, // Assuming TensorInferencer constructor matches
      confidence, infer_callback);
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

  int frame_idx = 0, gop_idx = 0, frame_idx_in_gop = 0;
  int hits = 0, pool = 0;
  int total_hits = 0, decoded_frames = 0, skipped_frames = 0,
      total_packages = 0; // total_packages seems to be only updated once, for
                          // the last segment
  std::vector<AVPacket *> *pkts = new std::vector<AVPacket *>();
  std::vector<std::vector<AVPacket *>> all_pkts; // Used for final cleanup

  while (av_read_frame(fmtCtx, packet) >= 0) {
    if (packet->stream_index == video_stream_index) {
      frame_idx++;
      if (frame_idx > 1 && (frame_idx - 1) % interval == 0) {
        hits++;
      }
      bool is_key_frame = (packet->flags & AV_PKT_FLAG_KEY);
      if (is_key_frame) {
        int last_frame_in_gop = 0; // To be calculated
        if (hits > 0) {
          skipped_frames += pool;
          last_frame_in_gop = hits * interval - pool;
          if (last_frame_in_gop > 0) {
            decoded_frames += last_frame_in_gop; // Statistical count
            std::vector<AVPacket *> decoding_pkts =
                get_packets_for_decoding(pkts, last_frame_in_gop);
            if (!decoding_pkts.empty()) {
              decoder->decode(
                  decoding_pkts, interval,
                  frame_idx); // frame_idx here is latest_frame_index in GOP for
                              // context
              {
                std::lock_guard<std::mutex> lock(task_mutex);
                remaining_decode_tasks++;
              }
              all_pkts.push_back(
                  std::move(decoding_pkts)); // Store for later cleanup
            } else {
              // if decoding_pkts is unexpectedly empty, ensure cleanup if it
              // allocated internally
              clear_av_packets(&decoding_pkts); // Or just let it be if it's
                                                // stack-allocated and empty
            }
          }
          total_hits += hits; // Statistical count
          pool =
              frame_idx_in_gop -
              last_frame_in_gop; // frame_idx_in_gop here is from previous GOP
                                 // last_frame_in_gop from current processing
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
    last_frame_in_gop = hits * interval - pool;
    if (last_frame_in_gop > 0) {
      std::vector<AVPacket *> decoding_pkts =
          get_packets_for_decoding(pkts, last_frame_in_gop);
      if (!decoding_pkts.empty()) {
        decoder->decode(decoding_pkts, interval,
                        frame_idx); // frame_idx is total frames read
        {
          std::lock_guard<std::mutex> lock(task_mutex);
          remaining_decode_tasks++;
        }
        all_pkts.push_back(std::move(decoding_pkts)); // Store for later cleanup
      } else {
        clear_av_packets(&decoding_pkts);
      }
      decoded_frames += last_frame_in_gop; // Statistical count
      total_packages += last_frame_in_gop; // Statistical count
    }
    total_hits += hits; // Statistical count
    pool = frame_idx_in_gop -
           last_frame_in_gop; // Update pool with remaining from last GOP
  } else {
    pool += frame_idx_in_gop;
  }
  skipped_frames += pool; // Add final pool to skipped frames

  // Wait for all decoding and inference tasks to complete
  while (true) {
    std::unique_lock<std::mutex> lock(task_mutex);
    if (task_cv.wait_for(lock, std::chrono::seconds(2), [this]() {
          std::cout << "\rRemaining decode tasks: "
                    << remaining_decode_tasks.load() << "   " << std::flush;
          return remaining_decode_tasks.load() == 0;
        })) {
      std::cout << "\nAll decode tasks completed." << std::endl;
      tensor_inferencer->finalizeInference();

      // while (true) {
      //   std::unique_lock<std::mutex> infer_lock(
      //       pending_infer_mutex); // Use a different lock variable name
      //   if (pending_infer_cv.wait_for(infer_lock, std::chrono::seconds(2),
      //                                 [this]() {
      //                                   std::cout << "\rRemaining infer
      //                                   tasks: "
      //                                             <<
      //                                             pending_infer_tasks.load()
      //                                             << "    " << std::flush;
      //                                   return pending_infer_tasks.load() <=
      //                                   0;
      //                                 })) {
      //     std::cout << "\nAll inference tasks completed." << std::endl;
      //     break;
      //   }
    }
    break;
  }
}

// Cleanup all stored packets
for (auto &pkt_list : all_pkts) {
  clear_av_packets(&pkt_list); // Pass address of the vector of AVPacket*
}
all_pkts.clear();

av_packet_free(&packet); // Free the initial allocated packet
clear_av_packets(pkts);  // Clear and free packets in the current pkts list
delete pkts;             // Delete the pkts list itself
pkts = nullptr;

std::cout << "-------------------" << std::endl;
float percentage =
    (frame_idx > 0) ? (static_cast<float>(decoded_frames) * 100.0f / frame_idx)
                    : 0.0f;
std::cout
    << "Total GOPs processed: " << gop_idx << std::endl
    << "Interval: " << interval << std::endl
    << "Total packages for last segment processing: " << total_packages
    << std::endl // Clarified meaning
    << "Frames submitted for decoding (estimate): " << decoded_frames
    << std::endl
    << "Frames skipped by selection logic: " << skipped_frames << std::endl
    << "Discrepancy (Total read - submitted for decode - skipped by logic): "
    << (frame_idx - decoded_frames - skipped_frames) << std::endl
    << "Percentage of frames submitted for decoding from total read: "
    << std::fixed << std::setprecision(2) << percentage << "%" << std::endl
    << "Successfully decoded frames (from callbacks): "
    << total_decoded_frames.load() << std::endl
    << "Extraction trigger points (hits): " << total_hits << std::endl;

return 0;
}

void VideoProcessor::onDecoded(std::vector<cv::Mat> &received_frames,
                               int gopId) { // Renamed param for clarity
  const int num_decoded_this_call = received_frames.size();

  if (num_decoded_this_call > 0) {
    InferenceInput input;
    input.decoded_frames =
        std::move(received_frames); // 'received_frames' is now empty or in a
                                    // valid but unspecified state
    input.latest_frame_index = gopId;
    tensor_inferencer->infer(input);
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
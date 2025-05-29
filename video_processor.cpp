// video_processor.cpp

#include "video_processor.h"
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <memory>
#include <map>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <thread>
#include <algorithm>
#include <numeric>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/hwcontext.h>
#include <libavutil/error.h>
#include <libswscale/swscale.h>
}
#include <opencv2/opencv.hpp>

VideoProcessor::VideoProcessor(const std::string &video_file_name, int interval) :
    decoder(const_cast<std::string &>(video_file_name)), // 如果需要其他 codec，可以替换
    video_file_name(video_file_name),
    interval(interval) {
    // 其他初始化逻辑
}

VideoProcessor::~VideoProcessor() {
    // 清理逻辑（如有）
}

int VideoProcessor::process() {
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

    AVPacket *packet = av_packet_alloc();
    if (!packet) {
        std::cerr << "Could not allocate AVPacket" << std::endl;
        avformat_close_input(&fmtCtx);
        return -1;
    }

    int frame_idx = 0, gop_idx = 0, frame_idx_in_gop = 0;
    int hits = 0, pool = 0;
    int total_hits = 0, decoded_frames = 0, skipped_frames = 0, total_packages = 0;
    std::vector<AVPacket *> *pkts = new std::vector<AVPacket *>();
    int success = 0;

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
                    // std::cout << gop_idx << "," << hits << std::endl;
                    skipped_frames += pool;
                    last_frame_in_gop = hits * interval - pool;
                    decoded_frames += last_frame_in_gop;
                    total_hits += hits;
                    pool = frame_idx_in_gop - last_frame_in_gop;
                    std::vector<AVPacket *> decoding_pkts = get_packets_for_decoding(pkts, last_frame_in_gop);
                    // decoder.reset();
                    decoder.decode(decoding_pkts, interval);
                    std::vector<cv::Mat> decoded_frams = decoder.getDecodedFrames();
                    success += decoded_frams.size();
                    // std::cout << "decoded: " << decoded_frams.size() << std::endl;
                    total_packages += decoding_pkts.size();
                    clear_av_packets(&decoding_pkts);
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
        std::vector<AVPacket *> decoding_pkts = get_packets_for_decoding(pkts, last_frame_in_gop);
        // decoder.reset();
        decoder.decode(decoding_pkts, interval);
        std::vector<cv::Mat> decoded_frams = decoder.getDecodedFrames();
        success += decoded_frams.size();
        // std::cout << "decoded: " << decoded_frams.size() << std::endl;
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

    skipped_frames += pool;
    av_packet_free(&packet);
    avformat_close_input(&fmtCtx);
    clear_av_packets(pkts);
    delete pkts;

    std::cout << "-------------------" << std::endl;
    float percentage = (frame_idx > 0) ? (decoded_frames * 100.0f / frame_idx) : 0.0f;
    std::cout << "total gop: " << gop_idx << std::endl
              << "interval: " << interval << std::endl
              << "total_packages: " << total_packages << std::endl
              << "decoded frames: " << decoded_frames << std::endl
              << "skipped frames: " << skipped_frames << std::endl
              << "discrepancies: " << frame_idx - decoded_frames - skipped_frames << std::endl
              << "percentage: " << percentage << "%" << std::endl
              << "successfully decoded: " << success << std::endl
              << "extracted frames: " << total_hits << std::endl;
    return 0;
}

void VideoProcessor::add_av_packet_to_list(std::vector<AVPacket *> **packages, const AVPacket *packet) {
    if (!packet) return;
    if (!*packages) {
        *packages = new std::vector<AVPacket *>();
    }
    AVPacket *cloned = av_packet_clone(packet);
    if (cloned) {
        (*packages)->push_back(cloned);
    }
}

std::vector<AVPacket *> VideoProcessor::get_packets_for_decoding(std::vector<AVPacket *> *packages, int last_frame_index) {
    std::vector<AVPacket *> results;
    if (!packages) return results;
    for (int i = 0; i < last_frame_index && i < packages->size(); i++) {
        AVPacket *pkt = av_packet_clone((*packages)[i]);
        if (pkt) {
            results.push_back(pkt);
        }
    }
    return results;
}

void VideoProcessor::clear_av_packets(std::vector<AVPacket *> *packages) {
    if (!packages) return;
    for (AVPacket *pkt : *packages) {
        av_packet_free(&pkt);
    }
    packages->clear();
}
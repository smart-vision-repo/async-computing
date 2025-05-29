

#include "packet_decoder.h"
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

PacketDecoder::PacketDecoder(std::string video_file_name) :
    vidIdx(-1),
    video_file_name(video_file_name),
    useHW(false),
    parser(nullptr),
    ctx(nullptr),
    codec(nullptr),
    fmtCtx(nullptr),
    hwCtxRef(nullptr),
    hw_device_ctx(nullptr),
    swsCtx(nullptr) {
    if (!initialize()) {
        throw std::runtime_error("Failed to initialize PacketDecoder.");
    }
}

PacketDecoder::~PacketDecoder() {
    if (ctx) {
        avcodec_free_context(&ctx);
    }
    if (parser) {
        av_parser_close(parser);
    }
    if (swsCtx) {
        sws_freeContext(swsCtx);
    }
    if (hw_device_ctx) {
        av_buffer_unref(&hw_device_ctx);
    }
}

bool PacketDecoder::initialize() {
    if (avformat_open_input(&fmtCtx, video_file_name.c_str(), nullptr, nullptr) != 0) {
        std::cerr << "Error: Could not open video file " << video_file_name << std::endl;
        return false;
    }
    if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
        std::cerr << "Error: Could not find stream information." << std::endl;
        avformat_close_input(&fmtCtx);
        return false;
    }

    for (unsigned i = 0; i < fmtCtx->nb_streams; i++) {
        if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            vidIdx = i;
            codec = avcodec_find_decoder(fmtCtx->streams[i]->codecpar->codec_id);
            if (!codec) {
                std::cerr << "Error: Unsupported codec!" << std::endl;
                avformat_close_input(&fmtCtx);
                return false;
            }
            break;
        }
    }

    ctx = avcodec_alloc_context3(codec);
    if (!ctx) {
        std::cerr << "Error: Could not allocate codec context." << std::endl;
        avformat_close_input(&fmtCtx);
        return false;
    }
    if (avcodec_parameters_to_context(ctx, fmtCtx->streams[vidIdx]->codecpar) < 0) {
        std::cerr << "Error: Could not copy codec parameters to context." << std::endl;
        avcodec_free_context(&ctx);
        avformat_close_input(&fmtCtx);
        return false;
    }

    ctx->thread_count = std::max(1u, std::thread::hardware_concurrency() / 2);
    useHW = initHWDecoder(ctx, &hwCtxRef);

    if (avcodec_open2(ctx, codec, nullptr) < 0) {
        std::cerr << "Error: Could not open codec." << std::endl;
        if (hwCtxRef) av_buffer_unref(&hwCtxRef);
        avcodec_free_context(&ctx);
        avformat_close_input(&fmtCtx);
        return false;
    }

    return true;
}

static AVPixelFormat getHWFormatCallback(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
    if (!ctx || !ctx->hw_device_ctx || !ctx->hw_device_ctx->data) {
        std::cerr << "getHWFormatCallback: ERROR - Missing hardware device context in AVCodecContext." << std::endl;
        return AV_PIX_FMT_NONE;
    }
    AVHWDeviceContext *hw_device_ctx_data = (AVHWDeviceContext *)ctx->hw_device_ctx->data;
    const enum AVHWDeviceType active_device_type = hw_device_ctx_data->type;
    for (const enum AVPixelFormat *p_fmt = pix_fmts; *p_fmt != AV_PIX_FMT_NONE; ++p_fmt) {
        bool format_is_compatible = false;
        switch (active_device_type) {
        case AV_HWDEVICE_TYPE_CUDA:
            if (*p_fmt == AV_PIX_FMT_CUDA) format_is_compatible = true;
            break;
        case AV_HWDEVICE_TYPE_VAAPI:
            if (*p_fmt == AV_PIX_FMT_VAAPI) format_is_compatible = true;
            break;
        case AV_HWDEVICE_TYPE_VDPAU:
            if (*p_fmt == AV_PIX_FMT_VDPAU) format_is_compatible = true;
            break;
        case AV_HWDEVICE_TYPE_QSV:
            if (*p_fmt == AV_PIX_FMT_QSV) format_is_compatible = true;
            break;
        case AV_HWDEVICE_TYPE_D3D11VA:
            if (*p_fmt == AV_PIX_FMT_D3D11) format_is_compatible = true;
            break;
        case AV_HWDEVICE_TYPE_VIDEOTOOLBOX:
            if (*p_fmt == AV_PIX_FMT_VIDEOTOOLBOX) format_is_compatible = true;
            break;
        default:
            std::cerr << "getHWFormatCallback: Unhandled or unknown active_device_type: "
                      << av_hwdevice_get_type_name(active_device_type) << std::endl;
            break;
        }
        if (format_is_compatible) {
            return *p_fmt;
        }
    }
    std::cerr << "getHWFormatCallback: Failed to find a suitable HW pixel format for device type: "
              << av_hwdevice_get_type_name(active_device_type) << std::endl;
    std::cerr << "  Available formats in pix_fmts were: ";
    for (const enum AVPixelFormat *p_fmt = pix_fmts; *p_fmt != AV_PIX_FMT_NONE; ++p_fmt) {
        std::cerr << av_get_pix_fmt_name(*p_fmt) << " ";
    }
    std::cerr << std::endl;
    return AV_PIX_FMT_NONE;
}

int PacketDecoder::initHWDecoder(AVCodecContext *codecContext, AVBufferRef **hwDeviceCtx) {
    if (*hwDeviceCtx != nullptr) {
        av_buffer_unref(hwDeviceCtx);
        *hwDeviceCtx = nullptr;
    }
    const char *hwAccel[] = {"cuda", "vaapi", "vdpau", "qsv", "d3d11va", "videotoolbox", nullptr};
    for (int i = 0; hwAccel[i]; i++) {
        enum AVHWDeviceType type = av_hwdevice_find_type_by_name(hwAccel[i]);
        if (type == AV_HWDEVICE_TYPE_NONE) continue;
        int ret = av_hwdevice_ctx_create(hwDeviceCtx, type, 0, nullptr, 0);
        if (ret >= 0) {
            if (codecContext->hw_device_ctx) av_buffer_unref(&codecContext->hw_device_ctx);
            codecContext->hw_device_ctx = av_buffer_ref(*hwDeviceCtx);
            if (!codecContext->hw_device_ctx) {
                std::cerr << "Failed to create a new reference for hw_device_ctx for " << hwAccel[i] << std::endl;
                av_buffer_unref(hwDeviceCtx);
                *hwDeviceCtx = nullptr;
                continue;
            }
            codecContext->get_format = getHWFormatCallback;
            std::cout << "Hardware acceleration initialized successfully: " << hwAccel[i] << std::endl;
            return 1;
        }
    }
    std::cout << "No suitable hardware acceleration method was successfully initialized." << std::endl;
    return 0;
}

// 新增的辅助函数：重置解码器状态
void PacketDecoder::resetDecoder() {
    if (ctx) {
        avcodec_flush_buffers(ctx);
    }
}

// 新增的辅助函数：将AVFrame转换为cv::Mat
cv::Mat PacketDecoder::avFrameToMat(AVFrame *frame) {
    if (!frame) {
        std::cerr << "Error: Null AVFrame in avFrameToMat" << std::endl;
        return cv::Mat();
    }

    AVFrame *rgbFrame = nullptr;

    // 如果是硬件解码的帧，需要先转换到CPU内存
    if (useHW && frame->format != AV_PIX_FMT_YUV420P && frame->format != AV_PIX_FMT_BGR24) {
        AVFrame *swFrame = av_frame_alloc();
        if (!swFrame) {
            std::cerr << "Error: Could not allocate software frame" << std::endl;
            return cv::Mat();
        }

        int ret = av_hwframe_transfer_data(swFrame, frame, 0);
        if (ret < 0) {
            std::cerr << "Error: Failed to transfer data from hardware frame to software frame" << std::endl;
            av_frame_free(&swFrame);
            return cv::Mat();
        }

        rgbFrame = convertToRGB(swFrame);
        av_frame_free(&swFrame);
    } else {
        rgbFrame = convertToRGB(frame);
    }

    if (!rgbFrame) {
        std::cerr << "Error: Failed to convert frame to RGB" << std::endl;
        return cv::Mat();
    }

    // 创建cv::Mat并拷贝数据
    cv::Mat mat(rgbFrame->height, rgbFrame->width, CV_8UC3);

    // 拷贝RGB数据到Mat
    if (rgbFrame->linesize[0] == mat.step) {
        // 如果行步长相同，可以直接拷贝
        memcpy(mat.data, rgbFrame->data[0], mat.total() * mat.elemSize());
    } else {
        // 逐行拷贝
        for (int y = 0; y < mat.rows; ++y) {
            memcpy(mat.ptr(y), rgbFrame->data[0] + y * rgbFrame->linesize[0], mat.cols * 3);
        }
    }
    av_frame_free(&rgbFrame);
    return mat;
}

// 新增的辅助函数：将帧转换为RGB格式
AVFrame *PacketDecoder::convertToRGB(AVFrame *frame) {
    if (!frame) return nullptr;

    AVFrame *rgbFrame = av_frame_alloc();
    if (!rgbFrame) {
        std::cerr << "Error: Could not allocate RGB frame" << std::endl;
        return nullptr;
    }

    rgbFrame->format = AV_PIX_FMT_BGR24; // OpenCV使用BGR格式
    rgbFrame->width = frame->width;
    rgbFrame->height = frame->height;

    int ret = av_frame_get_buffer(rgbFrame, 32);
    if (ret < 0) {
        std::cerr << "Error: Could not allocate RGB frame buffer" << std::endl;
        av_frame_free(&rgbFrame);
        return nullptr;
    }

    // 创建或重用swsContext进行像素格式转换
    if (!swsCtx) {
        swsCtx = sws_getContext(
            frame->width, frame->height, (AVPixelFormat)frame->format,
            rgbFrame->width, rgbFrame->height, AV_PIX_FMT_BGR24,
            SWS_BILINEAR, nullptr, nullptr, nullptr);

        if (!swsCtx) {
            std::cerr << "Error: Could not initialize swsContext" << std::endl;
            av_frame_free(&rgbFrame);
            return nullptr;
        }
    }

    // 执行像素格式转换
    sws_scale(swsCtx, frame->data, frame->linesize, 0, frame->height,
              rgbFrame->data, rgbFrame->linesize);

    return rgbFrame;
}

void PacketDecoder::decode(const std::vector<AVPacket *> &pkts, int interval) {
    if (pkts.empty()) {
        std::cerr << "Warning: Empty packet vector" << std::endl;
        return;
    }

    // 获取正确的视频流索引
    int correct_stream_index = -1;
    for (unsigned i = 0; i < fmtCtx->nb_streams; i++) {
        if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            correct_stream_index = i;
            break;
        }
    }

    AVFrame *frame = av_frame_alloc();
    if (!frame) {
        std::cerr << "Error: Could not allocate frame." << std::endl;
        return;
    }

    // 清空之前的解码结果
    decoded_frames.clear();

    // 重置解码器状态 - 这是关键！
    // 每次解码新的GOP时都需要flush解码器内部状态
    avcodec_flush_buffers(ctx);

    // 第一阶段：处理所有输入包
    for (size_t i = 0; i < pkts.size(); ++i) {
        AVPacket *pkt = pkts[i];
        if (!pkt) {
            // std::cerr << "Warning: Null packet at index " << i << std::endl;
            continue;
        }

        // 确保stream_index正确
        if (pkt->stream_index != vidIdx) {
            pkt->stream_index = vidIdx;
        }

        // 发送数据包到解码器
        int ret = avcodec_send_packet(ctx, pkt);
        if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
            std::cerr << "Error sending packet " << i << " to decoder: " << ret
                      << " (" << errbuf << ")" << std::endl;

            // 如果发送失败，尝试重置解码器状态
            if (ret == AVERROR_INVALIDDATA || ret == AVERROR(EINVAL)) {
                avcodec_flush_buffers(ctx);
                continue;
            }
            continue;
        }

        // 尝试接收解码后的帧（可能一个包对应多个帧，也可能需要多个包才能产生一个帧）
        while (true) {
            ret = avcodec_receive_frame(ctx, frame);

            if (ret == AVERROR(EAGAIN)) {
                // 解码器需要更多输入数据才能产生输出帧
                break;
            } else if (ret == AVERROR_EOF) {
                // 解码器已结束（通常不会在这里出现）
                break;
            } else if (ret < 0) {
                std::cerr << "  Error receiving frame: " << ret << std::endl;
                break;
            }

            // 将AVFrame转换为cv::Mat
            cv::Mat mat = avFrameToMat(frame);
            if (!mat.empty()) {
                decoded_frames.push_back(mat);
            } else {
                std::cerr << "  Failed to convert AVFrame to cv::Mat" << std::endl;
            }

            av_frame_unref(frame);
        }
    }

    // 第二阶段：刷新解码器缓存
    // 原因：解码器内部可能缓存了一些帧用于B帧重排序或多线程处理
    avcodec_send_packet(ctx, nullptr); // 发送NULL包告诉解码器输入结束

    while (true) {
        int ret = avcodec_receive_frame(ctx, frame);
        if (ret == AVERROR_EOF) {
            // 正常结束，没有更多帧
            break;
        } else if (ret == AVERROR(EAGAIN)) {
            // 这种情况在flush时不应该出现，但为了安全还是处理
            break;
        } else if (ret < 0) {
            std::cerr << "Error during decoder flush: " << ret << std::endl;
            break;
        }

        cv::Mat mat = avFrameToMat(frame);
        if (!mat.empty()) {
            decoded_frames.push_back(mat);
        }

        av_frame_unref(frame);
    }

    int pkt_size = pkts.size() - 1;
    std::vector<int> frame_indices;
    while (pkt_size >= 0) {
        frame_indices.push_back(pkt_size);
        pkt_size -= interval;
    }
    std::sort(frame_indices.begin(), frame_indices.end());
    // std::cout << "count: " << frame_indices.size() << std::endl;
    std::vector<cv::Mat> filtered_frames;
    for (size_t i = 0; i < decoded_frames.size(); ++i) {
        if (std::find(frame_indices.begin(), frame_indices.end(), i) != frame_indices.end()) {
            filtered_frames.push_back(decoded_frames[i]);
        }
    }
    decoded_frames = std::move(filtered_frames);
    av_frame_free(&frame);
}

std::vector<cv::Mat> PacketDecoder::getDecodedFrames() const {
    return decoded_frames;
}

// 公共方法：重置解码器状态（用于解码多个独立的GOP）
void PacketDecoder::reset() {
    resetDecoder();
}
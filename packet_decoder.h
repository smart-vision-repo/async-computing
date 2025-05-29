#ifndef PACKET_DECODER_H
#define PACKET_DECODER_H

#include <vector>
#include <opencv2/core.hpp>

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
class PacketDecoder {
public:
    PacketDecoder(std::string video_file_name);
    ~PacketDecoder();

    void decode(const std::vector<AVPacket *> &pkts, int interval);
    std::vector<cv::Mat> getDecodedFrames() const;
    void reset(); // 软重置解码器

private:
    int vidIdx;
    const std::string video_file_name;
    bool useHW;
    AVCodecParserContext *parser;
    AVCodecContext *ctx;
    const AVCodec *codec;

    AVFormatContext *fmtCtx;
    AVBufferRef *hwCtxRef;
    AVBufferRef *hw_device_ctx;
    SwsContext *swsCtx;
    std::vector<cv::Mat> decoded_frames;
    bool initialize();
    int initHWDecoder(AVCodecContext *codecContext, AVBufferRef **hwDeviceCtx);
    cv::Mat avFrameToMat(AVFrame *frame);
    AVFrame *convertToRGB(AVFrame *frame);
    void resetDecoder();
};

#endif // PACKET_DECODER_H

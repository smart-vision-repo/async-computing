#!/bin/bash

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_gpp() {
    if ! command -v g++ &>/dev/null; then
        log_error "g++ not found."
        return 1
    fi
    log_success "g++ found: $(g++ --version | head -n1)"
    return 0
}

# 清除 pkg-config 输出中的无效链接项，例如 -lrt 和 -lboost_*.so.1.74
clean_link_flags() {
    echo "$1" | sed -E 's/-lboost_([a-zA-Z0-9_]+)\.so\.[0-9.]+/-lboost_\1/g' | sed 's/-lrt//g'
}

compile() {
    local output_name="app_tensor_inferencer_static"
    local src_files=(main.cpp video_processor.cpp packet_decoder.cpp message_proxy.cpp tensor_inferencer.cpp)

    # 环境路径
    CUDA_ROOT=/usr/local/cuda-12.2
    CUDA_INCLUDE=$CUDA_ROOT/include
    CUDA_LIB=$CUDA_ROOT/lib64

    TENSORRT_ROOT=/home/tju/apps/TensorRT-8.6.1.6
    TENSORRT_INCLUDE=$TENSORRT_ROOT/include
    TENSORRT_LIB=$TENSORRT_ROOT/lib

    INFER_INCLUDE_FLAGS="-I$CUDA_INCLUDE -I$TENSORRT_INCLUDE"

    log_info "Fetching static linking flags via pkg-config..."
    FFMPEG_CFLAGS_STATIC=$(pkg-config --cflags libavformat libavcodec libavutil libswscale)
    FFMPEG_LIBS_STATIC_CORE=$(pkg-config --static --libs libavformat libavcodec libavutil libswscale)

    AMQP_CFLAGS_STATIC=$(pkg-config --cflags libSimpleAmqpClient)
    AMQP_LIBS_STATIC=$(pkg-config --static --libs libSimpleAmqpClient)

    OPENCV_CFLAGS_STATIC=$(pkg-config --cflags opencv4)
    OPENCV_LIBS_STATIC=$(pkg-config --static --libs opencv4)

    # 清理链接标志
    FFMPEG_LIBS_STATIC_CORE=$(clean_link_flags "$FFMPEG_LIBS_STATIC_CORE")
    AMQP_LIBS_STATIC=$(clean_link_flags "$AMQP_LIBS_STATIC")
    OPENCV_LIBS_STATIC=$(clean_link_flags "$OPENCV_LIBS_STATIC")

    log_info "---- PKG-CONFIG OUTPUT ----"
    log_info "FFMPEG_LIBS_STATIC_CORE: $FFMPEG_LIBS_STATIC_CORE"
    log_info "AMQP_LIBS_STATIC: $AMQP_LIBS_STATIC"
    log_info "OPENCV_LIBS_STATIC: $OPENCV_LIBS_STATIC"
    log_info "---------------------------"

    TRT_CUDA_LIB_PATHS="-L$CUDA_LIB -L$TENSORRT_LIB"
    TENSORRT_LINK_FLAGS="-lnvinfer -lnvinfer_plugin -lnvonnxparser -lcudart"

    # 静态链接基本标志（gcc/libstdc++）
    STATIC_COMPILE_FLAGS="-static-libgcc -static-libstdc++"

    # 是否强制使用 -static（完全静态链接，建议关闭）
    FORCE_FULL_STATIC=1
    if [[ "$FORCE_FULL_STATIC" == 1 ]]; then
        STATIC_COMPILE_FLAGS="-static -static-libgcc -static-libstdc++"
        log_warning "⚠️  强制使用 -static，可能失败（依赖不完整）"
    fi

    # 避免使用 -lrt，只保留必要的线程/动态库支持
    ADDITIONAL_STATIC_DEPS="-lpthread -ldl"

    log_info "Compiling ${src_files[*]}..."

    if ! g++ -std=c++17 -Wall -Wno-deprecated-declarations \
        "${src_files[@]}" \
        $FFMPEG_CFLAGS_STATIC $AMQP_CFLAGS_STATIC $OPENCV_CFLAGS_STATIC \
        $INFER_INCLUDE_FLAGS \
        -o "$output_name" \
        $TRT_CUDA_LIB_PATHS \
        $FFMPEG_LIBS_STATIC_CORE $AMQP_LIBS_STATIC $OPENCV_LIBS_STATIC $TENSORRT_LINK_FLAGS \
        $ADDITIONAL_STATIC_DEPS \
        $STATIC_COMPILE_FLAGS 2>compile.log; then
        log_error "❌ Static compilation failed. See below:"
        tail -n 20 compile.log >&2
        exit 1
    fi

    rm -f compile.log
    log_success "✅ Static compilation successful: $output_name"
}

main() {
    check_gpp || exit 1
    compile
}

main

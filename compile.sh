#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}
log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}
log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}
log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check g++ compiler
check_gpp() {
    if ! command -v g++ &>/dev/null; then
        log_error "g++ command not found."
        return 1
    fi
    log_success "g++ found: $(g++ --version | head -n 1)"
    return 0
}

# Setup environment variables and compile
compile() {
    local output_name="app_tensor_inferencer"
    local src_files=(
        tensor_main.cpp
        video_processor.cpp
        packet_decoder.cpp
        yolo_inferencer.cpp
        message_proxy.cpp
        tensor_inferencer.cpp
    )

    log_info "Checking for pkg-config and dependencies..."

    if ! command -v pkg-config &>/dev/null; then
        log_error "pkg-config is not installed."
        exit 1
    fi

    if ! pkg-config --exists libavformat libavcodec libavutil libswscale opencv4 libSimpleAmqpClient; then
        log_error "Required libraries not found via pkg-config."
        exit 1
    fi

    log_success "All required libraries found."

    # CUDA 12.2 路径
    CUDA_ROOT=/usr/local/cuda-12.2
    CUDA_INCLUDE=$CUDA_ROOT/include
    CUDA_LIB=$CUDA_ROOT/lib64

    # TensorRT 路径
    TENSORRT_ROOT=/home/tju/apps/TensorRT-8.6.1.6
    TENSORRT_INCLUDE=$TENSORRT_ROOT/include
    TENSORRT_LIB=$TENSORRT_ROOT/lib

    # 添加编译标志
    INFER_INCLUDE_FLAGS="-I$CUDA_INCLUDE -I$TENSORRT_INCLUDE"
    TENSORRT_FLAGS="-L$CUDA_LIB -L$TENSORRT_LIB -lnvinfer -lnvinfer_plugin -lnvonnxparser -lcudart"

    FFMPEG_CFLAGS=$(pkg-config --cflags libavformat libavcodec libavutil libswscale libSimpleAmqpClient)
    FFMPEG_LIBS=$(pkg-config --libs libavformat libavcodec libavutil libswscale libSimpleAmqpClient)
    OPENCV_CFLAGS=$(pkg-config --cflags opencv4)
    OPENCV_LIBS=$(pkg-config --libs opencv4)

    log_info "Compiling ${src_files[*]}..."

    if ! g++ -std=c++17 -Wall -Wno-deprecated-declarations \
        "${src_files[@]}" -o "$output_name" \
        $FFMPEG_CFLAGS $OPENCV_CFLAGS \
        $FFMPEG_LIBS $OPENCV_LIBS $CUDA_FLAGS \
        $INFER_INCLUDE_FLAGS $TENSORRT_FLAGS \
        -lpthread 2>compile.log; then
        log_error "Compilation failed. See details below:"
        cat compile.log >&2
        exit 1
    fi

    rm -f compile.log
    log_success "Compilation successful. Output binary: $output_name"
}

# Run checks and compile
main() {
    check_gpp || exit 1
    compile
}

main

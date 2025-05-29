#!/bin/bash

git pull origin main

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
    if ! command -v g++ &> /dev/null; then
        log_error "g++ command not found."
        return 1
    fi
    log_success "g++ found: $(g++ --version | head -n 1)"
    return 0
}

# Setup environment variables and compile
compile() {
    local output_name="app_video_processor"
    local src_files=(main.cpp video_processor.cpp packet_decoder.cpp)

    log_info "Checking for pkg-config and dependencies..."

    if ! command -v pkg-config &>/dev/null; then
        log_error "pkg-config is not installed."
        exit 1
    fi

    if ! pkg-config --exists libavformat libavcodec libavutil libswscale opencv4; then
        log_error "Required libraries not found via pkg-config."
        exit 1
    fi

    log_success "All required libraries found."

    FFMPEG_CFLAGS=$(pkg-config --cflags libavformat libavcodec libavutil libswscale)
    FFMPEG_LIBS=$(pkg-config --libs libavformat libavcodec libavutil libswscale)
    OPENCV_CFLAGS=$(pkg-config --cflags opencv4)
    OPENCV_LIBS=$(pkg-config --libs opencv4)

    CUDA_FLAGS=""
    if command -v nvcc &>/dev/null; then
        CUDA_FLAGS="-lcuda -lcudart"
        log_info "CUDA detected, adding CUDA flags."
    fi

    log_info "Compiling ${src_files[*]}..."

    if ! g++ -std=c++17 -Wall -Wno-deprecated-declarations \
        "${src_files[@]}" -o "$output_name" \
        $FFMPEG_CFLAGS $OPENCV_CFLAGS \
        $FFMPEG_LIBS $OPENCV_LIBS $CUDA_FLAGS \
        -lpthread 2> compile.log; then
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

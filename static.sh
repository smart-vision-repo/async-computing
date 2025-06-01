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
# Setup environment variables and compile
compile() {
    local output_name="app_tensor_inferencer_static"
    local src_files=(
        main.cpp
        video_processor.cpp
        packet_decoder.cpp
        message_proxy.cpp
        tensor_inferencer.cpp
    )

    # ... (rest of the initial checks and variable setups remain the same) ...

    # CUDA 12.2 路径
    CUDA_ROOT=/usr/local/cuda-12.2
    CUDA_INCLUDE=$CUDA_ROOT/include
    CUDA_LIB=$CUDA_ROOT/lib64

    # TensorRT 路径
    TENSORRT_ROOT=/home/tju/apps/TensorRT-8.6.1.6
    TENSORRT_INCLUDE=$TENSORRT_ROOT/include
    TENSORRT_LIB=$TENSORRT_ROOT/lib

    INFER_INCLUDE_FLAGS="-I$CUDA_INCLUDE -I$TENSORRT_INCLUDE"

    log_info "Fetching static linking flags via pkg-config..."
    # Separate pkg-config calls for clarity and easier debugging
    FFMPEG_CFLAGS_STATIC=$(pkg-config --cflags libavformat libavcodec libavutil libswscale)
    FFMPEG_LIBS_STATIC_CORE=$(pkg-config --static --libs libavformat libavcodec libavutil libswscale)

    AMQP_CFLAGS_STATIC=$(pkg-config --cflags libSimpleAmqpClient) # Usually, cflags are not static-dependent
    AMQP_LIBS_STATIC=$(pkg-config --static --libs libSimpleAmqpClient)

    OPENCV_CFLAGS_STATIC=$(pkg-config --cflags opencv4)
    OPENCV_LIBS_STATIC=$(pkg-config --static --libs opencv4)

    # Output pkg-config results for debugging
    log_info "---- PKG-CONFIG OUTPUT ----"
    log_info "FFMPEG_LIBS_STATIC_CORE: $FFMPEG_LIBS_STATIC_CORE"
    log_info "AMQP_LIBS_STATIC: $AMQP_LIBS_STATIC"
    log_info "OPENCV_LIBS_STATIC: $OPENCV_LIBS_STATIC"
    log_info "--------------------------"

    if [ -z "$FFMPEG_LIBS_STATIC_CORE" ] || [ -z "$AMQP_LIBS_STATIC" ] || [ -z "$OPENCV_LIBS_STATIC" ]; then
        log_warning "One or more pkg-config --static calls did not return library flags. Static linking for these might be incomplete."
        log_warning "Ensure static versions of these libraries (.a files) and their dependencies are installed."
    fi

    TRT_CUDA_LIB_PATHS="-L$CUDA_LIB -L$TENSORRT_LIB"
    TENSORRT_LINK_FLAGS="-lnvinfer -lnvinfer_plugin -lnvonnxparser -lcudart"
    STATIC_COMPILE_FLAGS="-static-libgcc -static-libstdc++"

    # For ADDITIONAL_STATIC_DEPS, -lrt was causing an issue.
    # It's likely added by one of the pkg-config outputs if needed.
    # If -lrt is still an issue and comes from pkg-config, ensure librt.a is installed or the dependency is legitimate.
    ADDITIONAL_STATIC_DEPS="-lpthread -ldl"

    log_info "Compiling ${src_files[*]} with attempts for static linking..."

    # Note the order of libraries and flags
    if ! g++ -std=c++17 -Wall -Wno-deprecated-declarations \
        "${src_files[@]}" \
        $FFMPEG_CFLAGS_STATIC $AMQP_CFLAGS_STATIC $OPENCV_CFLAGS_STATIC \
        $INFER_INCLUDE_FLAGS \
        -o "$output_name" \
        $TRT_CUDA_LIB_PATHS \
        $FFMPEG_LIBS_STATIC_CORE $AMQP_LIBS_STATIC $OPENCV_LIBS_STATIC $TENSORRT_LINK_FLAGS \
        $ADDITIONAL_STATIC_DEPS \
        $STATIC_COMPILE_FLAGS 2>compile.log; then
        log_error "Static compilation failed. See details below:"
        cat compile.log >&2
        exit 1
    fi

    rm -f compile.log
    log_success "Compilation (with static linking attempts) successful. Output binary: $output_name"
    # ... (rest of the warnings) ...
}
# Run checks and compile
main() {
    check_gpp || exit 1
    compile
}

main

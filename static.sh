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
    local output_name="app_tensor_inferencer_static" # 静态编译版本的输出文件名
    local src_files=(
        main.cpp
        video_processor.cpp
        packet_decoder.cpp
        message_proxy.cpp
        tensor_inferencer.cpp
    )

    log_info "Checking for pkg-config and dependencies..."

    if ! command -v pkg-config &>/dev/null; then
        log_error "pkg-config is not installed."
        exit 1
    fi

    # 初步检查所需库是否存在 (后续会用 --static 获取具体参数)
    if ! pkg-config --exists libavformat libavcodec libavutil libswscale opencv4 libSimpleAmqpClient; then
        log_error "Required libraries not found via pkg-config (initial check)."
        exit 1
    fi
    log_success "All required libraries found (initial check)."

    # CUDA 12.2 路径
    CUDA_ROOT=/usr/local/cuda-12.2
    CUDA_INCLUDE=$CUDA_ROOT/include
    CUDA_LIB=$CUDA_ROOT/lib64

    # TensorRT 路径
    TENSORRT_ROOT=/home/tju/apps/TensorRT-8.6.1.6
    TENSORRT_INCLUDE=$TENSORRT_ROOT/include
    TENSORRT_LIB=$TENSORRT_ROOT/lib

    # TensorRT 和 CUDA 的头文件包含路径
    INFER_INCLUDE_FLAGS="-I$CUDA_INCLUDE -I$TENSORRT_INCLUDE"

    log_info "Fetching static linking flags via pkg-config..."
    # 分开获取不同库的 pkg-config 参数，方便调试
    FFMPEG_CFLAGS_STATIC=$(pkg-config --cflags libavformat libavcodec libavutil libswscale)
    FFMPEG_LIBS_STATIC_CORE=$(pkg-config --static --libs libavformat libavcodec libavutil libswscale)

    # 获取 libSimpleAmqpClient 的 CFLAGS 和原始的 --static --libs 输出
    AMQP_CFLAGS_STATIC=$(pkg-config --cflags libSimpleAmqpClient)
    AMQP_LIBS_STATIC_RAW=$(pkg-config --static --libs libSimpleAmqpClient)

    OPENCV_CFLAGS_STATIC=$(pkg-config --cflags opencv4)
    OPENCV_LIBS_STATIC=$(pkg-config --static --libs opencv4)

    # 打印 pkg-config 的原始输出，用于调试
    log_info "---- PKG-CONFIG RAW OUTPUT ----"
    log_info "FFMPEG_LIBS_STATIC_CORE: $FFMPEG_LIBS_STATIC_CORE"
    log_info "AMQP_LIBS_STATIC_RAW (for SimpleAmqpClient): $AMQP_LIBS_STATIC_RAW"
    log_info "OPENCV_LIBS_STATIC: $OPENCV_LIBS_STATIC"
    log_info "-----------------------------"

    # 检查 pkg-config 是否成功返回了静态库参数
    if [ -z "$FFMPEG_LIBS_STATIC_CORE" ] || [ -z "$AMQP_LIBS_STATIC_RAW" ] || [ -z "$OPENCV_LIBS_STATIC" ]; then
        log_warning "One or more pkg-config --static calls did not return library flags."
        log_warning "Static linking for these might be incomplete or fail."
        log_warning "Ensure static versions of these libraries (.a files) and their dependencies are installed."
        # 如果这些库的静态链接是强制的，可以选择在这里退出
        # exit 1
    fi

    # 显式指定Boost库的静态链接
    # 这会告诉链接器在处理 -lboost_system 和 -lboost_chrono 时优先寻找静态库 (.a)
    # 前提是 libboost_system.a 和 libboost_chrono.a 已经安装在系统路径或 -L 指定的路径中
    BOOST_STATIC_LINK_FLAGS="-Wl,-Bstatic -lboost_system -lboost_chrono -Wl,-Bdynamic"

    # 从 AMQP_LIBS_STATIC_RAW 中过滤掉可能由 pkg-config 给出的针对 boost_system 和 boost_chrono 的参数
    # 比如 -lboost_system.so.1.74 或普通的 -lboost_system，因为我们将使用上面的 BOOST_STATIC_LINK_FLAGS 来精确控制
    # sed -e 's/PATTERN//g' 会删除匹配 PATTERN 的部分
    # [^ ]* 匹配任何非空格字符零次或多次
    AMQP_LIBS_FILTERED=$(echo "$AMQP_LIBS_STATIC_RAW" | sed -e 's/-lboost_system[^ ]*//g' -e 's/-lboost_chrono[^ ]*//g')
    log_info "AMQP_LIBS_FILTERED (after removing boost flags): $AMQP_LIBS_FILTERED"

    # CUDA 和 TensorRT 的库搜索路径
    TRT_CUDA_LIB_PATHS="-L$CUDA_LIB -L$TENSORRT_LIB"
    # TensorRT 和 CUDA 的主要链接库名
    TENSORRT_LINK_FLAGS="-lnvinfer -lnvinfer_plugin -lnvonnxparser -lcudart"

    # 静态链接 libgcc 和 libstdc++
    STATIC_COMPILE_FLAGS="-static-libgcc -static-libstdc++"

    # librt 的问题: 如果 pkg-config 的输出 (例如来自 AMQP_LIBS_FILTERED 或 FFMPEG_LIBS_STATIC_CORE)
    # 中包含了 -lrt，那么系统上必须有 librt.a。这里我们不手动添加 -lrt。
    # pthread 和 dl 通常是静态链接CUDA/TensorRT时的依赖
    ADDITIONAL_STATIC_DEPS="-lpthread -ldl"

    log_info "Compiling ${src_files[*]} with attempts for static linking..."

    # 注意g++命令中参数的顺序
    if ! g++ -std=c++17 -Wall -Wno-deprecated-declarations \
        "${src_files[@]}" \
        $FFMPEG_CFLAGS_STATIC $AMQP_CFLAGS_STATIC $OPENCV_CFLAGS_STATIC \
        $INFER_INCLUDE_FLAGS \
        -o "$output_name" \
        $TRT_CUDA_LIB_PATHS \
        $FFMPEG_LIBS_STATIC_CORE \
        $AMQP_LIBS_FILTERED \
        $OPENCV_LIBS_STATIC \
        $BOOST_STATIC_LINK_FLAGS \
        $TENSORRT_LINK_FLAGS \
        $ADDITIONAL_STATIC_DEPS \
        $STATIC_COMPILE_FLAGS 2>compile.log; then
        log_error "Static compilation failed. See details below:"
        cat compile.log >&2 # 将错误信息输出到标准错误流
        exit 1
    fi

    rm -f compile.log # 如果编译成功，删除日志文件
    log_success "Compilation (with static linking attempts) successful. Output binary: $output_name"
    log_warning "Static linking, especially with CUDA/TensorRT, can be complex."
    log_warning "Verify the binary's dependencies using 'ldd $output_name'."
    log_warning "If Boost or other libraries are still dynamically linked, ensure their static .a files are installed and discoverable."
    log_warning "The '-Wl,-Bstatic -lboost_system -lboost_chrono -Wl,-Bdynamic' flags attempt to force static linking for Boost."
    log_warning "If '-lrt' was part of the original errors and persists (likely from pkg-config output), ensure 'librt.a' is installed."
}

# Run checks and compile
main() {
    check_gpp || exit 1
    compile
}

main

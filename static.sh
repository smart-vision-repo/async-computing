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

# 清理非法链接标志
clean_link_flags() {
    echo "$1" |
        sed -E 's/-lboost_([a-zA-Z0-9_]+)\.so\.[0-9.]+/-lboost_\1/g' |
        sed -E 's/\brt\b//g' |
        sed -E 's/-lrt//g' |
        sed -E 's/\s+/ /g'
}

# 检查是否存在对应的静态库文件
check_static_libs_exist() {
    log_info "Checking existence of static libraries (.a files)..."
    local missing=0
    for flag in $1; do
        if [[ "$flag" =~ ^-l ]]; then
            libname="${flag:2}"
            found=0
            for path in /usr/lib /usr/local/lib /opt/lib /usr/lib/x86_64-linux-gnu; do
                if [[ -f "$path/lib${libname}.a" ]]; then
                    found=1
                    break
                fi
            done
            if [[ "$found" == 0 ]]; then
                log_warning "Static library not found: lib${libname}.a"
                missing=1
            fi
        fi
    done
    return $missing
}

compile() {
    local output_name="app_tensor_inferencer_static"
    local src_files=(main.cpp video_processor.cpp packet_decoder.cpp message_proxy.cpp tensor_inferencer.cpp)

    CUDA_ROOT=/usr/local/cuda-12.2
    CUDA_INCLUDE=$CUDA_ROOT/include
    CUDA_LIB=$CUDA_ROOT/lib64

    TENSORRT_ROOT=/home/tju/apps/TensorRT-8.6.1.6
    TENSORRT_INCLUDE=$TENSORRT_ROOT/include
    TENSORRT_LIB=$TENSORRT_ROOT/lib

    INFER_INCLUDE_FLAGS="-I$CUDA_INCLUDE -I$TENSORRT_INCLUDE"

    log_info "Fetching pkg-config link flags..."
    FFMPEG_CFLAGS_STATIC=$(pkg-config --cflags libavformat libavcodec libavutil libswscale)
    FFMPEG_LIBS_STATIC_CORE=$(pkg-config --static --libs libavformat libavcodec libavutil libswscale)

    AMQP_CFLAGS_STATIC=$(pkg-config --cflags libSimpleAmqpClient)
    AMQP_LIBS_STATIC=$(pkg-config --static --libs libSimpleAmqpClient)

    OPENCV_CFLAGS_STATIC=$(pkg-config --cflags opencv4)
    OPENCV_LIBS_STATIC=$(pkg-config --static --libs opencv4)

    # 清洗链接标志
    FFMPEG_LIBS_STATIC_CORE=$(clean_link_flags "$FFMPEG_LIBS_STATIC_CORE")
    AMQP_LIBS_STATIC=$(clean_link_flags "$AMQP_LIBS_STATIC")
    OPENCV_LIBS_STATIC=$(clean_link_flags "$OPENCV_LIBS_STATIC")

    log_info "---- CLEANED LINK FLAGS ----"
    log_info "FFMPEG: $FFMPEG_LIBS_STATIC_CORE"
    log_info "AMQP:   $AMQP_LIBS_STATIC"
    log_info "OpenCV: $OPENCV_LIBS_STATIC"
    log_info "--------------------------------"

    TRT_CUDA_LIB_PATHS="-L$CUDA_LIB -L$TENSORRT_LIB"
    TENSORRT_LINK_FLAGS="-lnvinfer -lnvinfer_plugin -lnvonnxparser -lcudart"

    STATIC_COMPILE_FLAGS="-static-libgcc -static-libstdc++"
    FORCE_FULL_STATIC=0
    if [[ "$FORCE_FULL_STATIC" == 1 ]]; then
        STATIC_COMPILE_FLAGS="-static -static-libgcc -static-libstdc++"
        log_warning "⚠️  Forcing full static linking with '-static'."
    fi

    ADDITIONAL_STATIC_DEPS="-lpthread -ldl"

    # 检查 .a 文件是否存在
    all_libs="$FFMPEG_LIBS_STATIC_CORE $AMQP_LIBS_STATIC $OPENCV_LIBS_STATIC $TENSORRT_LINK_FLAGS $ADDITIONAL_STATIC_DEPS"
    if ! check_static_libs_exist "$all_libs"; then
        log_warning "One or more static libraries missing. Linking may fail or fallback to shared libs."
    fi

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
        log_error "❌ Compilation failed. See last lines of compile.log:"
        tail -n 20 compile.log >&2
        exit 1
    fi

    rm -f compile.log
    log_success "✅ Compilation successful: $output_name"
}

main() {
    check_gpp || exit 1
    compile
}

main

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
    local output_name="app_tensor_inferencer_static" # Changed output name for the static version
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

    # Check for required libraries (pkg-config will be used to get static linking flags later)
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

    # 添加编译标志 (Include paths remain the same)
    INFER_INCLUDE_FLAGS="-I$CUDA_INCLUDE -I$TENSORRT_INCLUDE"

    # Get CFLAGS and LIBS, requesting static versions where pkg-config supports it
    log_info "Fetching static linking flags via pkg-config..."
    FFMPEG_CFLAGS_STATIC=$(pkg-config --cflags libavformat libavcodec libavutil libswscale libSimpleAmqpClient)
    # For LIBS, add --static. This tells pkg-config to provide flags for static linking if available.
    # This usually means listing .a files or appropriate -l flags and their dependencies.
    FFMPEG_LIBS_STATIC=$(pkg-config --static --libs libavformat libavcodec libavutil libswscale libSimpleAmqpClient)

    OPENCV_CFLAGS_STATIC=$(pkg-config --cflags opencv4) # CFLAGS for OpenCV might not change for static
    OPENCV_LIBS_STATIC=$(pkg-config --static --libs opencv4)

    if [ -z "$FFMPEG_LIBS_STATIC" ] || [ -z "$OPENCV_LIBS_STATIC" ]; then
        log_warning "pkg-config --static did not return libs for FFMPEG or OpenCV. Static linking for these might be incomplete."
        log_warning "Ensure static versions of these libraries (e.g., .a files) are installed."
    fi

    # Library search paths for CUDA and TensorRT (still needed for the linker to find libraries)
    TRT_CUDA_LIB_PATHS="-L$CUDA_LIB -L$TENSORRT_LIB"

    # Original TensorRT and CUDA linker flags (e.g., -lnvinfer).
    # When combined with -static-libgcc and -static-libstdc++, the linker might
    # prefer static versions (.a) of these libraries if they are available in the search paths
    # and correctly named (e.g., libnvinfer_static.a).
    # If this doesn't sufficiently link TensorRT/CUDA statically, you may need to
    # explicitly list the .a files for TensorRT/CUDA libraries.
    TENSORRT_LINK_FLAGS="-lnvinfer -lnvinfer_plugin -lnvonnxparser -lcudart"

    # GCC flags for static linking of C++ standard library and GCC runtime library
    # -static-libgcc: Link libgcc statically.
    # -static-libstdc++: Link libstdc++ statically.
    # Using the global -static flag is more aggressive and can cause issues,
    # especially with system libraries like glibc. Start with these more targeted flags.
    STATIC_COMPILE_FLAGS="-static-libgcc -static-libstdc++"

    # Common dependencies needed when linking CUDA and TensorRT libraries statically.
    # pkg-config --static should pull in many dependencies for other libs, but these are often needed for NVIDIA libs.
    ADDITIONAL_STATIC_DEPS="-lpthread -ldl" # -lrt might be needed on some systems

    log_info "Compiling ${src_files[*]} with attempts for static linking..."

    # The order of flags can matter:
    # g++ <compile_options> <sources> <include_flags> -o <output_name> <library_paths> <library_links_specific_first> <general_libs> <static_linking_flags>
    if ! g++ -std=c++17 -Wall -Wno-deprecated-declarations \
        "${src_files[@]}" \
        $FFMPEG_CFLAGS_STATIC $OPENCV_CFLAGS_STATIC \
        $INFER_INCLUDE_FLAGS \
        -o "$output_name" \
        $TRT_CUDA_LIB_PATHS \
        $FFMPEG_LIBS_STATIC $OPENCV_LIBS_STATIC $TENSORRT_LINK_FLAGS \
        $ADDITIONAL_STATIC_DEPS \
        $STATIC_COMPILE_FLAGS 2>compile.log; then
        log_error "Static compilation failed. See details below:"
        cat compile.log >&2
        exit 1
    fi

    rm -f compile.log
    log_success "Compilation (with static linking attempts) successful. Output binary: $output_name"
    log_warning "Static linking, especially with CUDA/TensorRT, can be complex."
    log_warning "Verify the binary's dependencies using 'ldd $output_name'."
    log_warning "Full static linking (including libc) is often not feasible or recommended."
    log_warning "If TensorRT/CUDA components are still dynamically linked, you might need to replace flags like '-lnvinfer'"
    log_warning "with direct paths to their static archive files (e.g., /path/to/libnvinfer_static.a) and manually list all their static dependencies."
}

# Run checks and compile
main() {
    check_gpp || exit 1
    compile
}

main

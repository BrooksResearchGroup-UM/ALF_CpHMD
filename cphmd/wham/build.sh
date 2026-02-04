#!/bin/bash
# Build script for WHAM CUDA library
#
# Requirements:
#   - NVIDIA CUDA Toolkit (nvcc compiler)
#   - CUDA-capable GPU
#
# Usage:
#   cd cphmd/wham
#   ./build.sh
#
# For verbose WHAM output, edit src/wham.cu and set VERBOSE to 1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
OUTPUT="$SCRIPT_DIR/libwham.so"

echo "Building WHAM library..."
echo "  Source: $SRC_DIR/wham.cu"
echo "  Output: $OUTPUT"

# Check for nvcc
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install NVIDIA CUDA Toolkit."
    exit 1
fi

# Compile for multiple GPU architectures:
# - sm_75: RTX 2080, 2080 Ti, Turing (T4, Quadro RTX)
# - sm_86: RTX 3090, 3080, Ampere (A5000, A6000)
# - sm_89: RTX 4090, Ada Lovelace
cd "$SRC_DIR"
nvcc -shared -Xcompiler -fPIC -O3 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -o "$OUTPUT" wham.cu

echo "Build successful: $OUTPUT"
echo ""
echo "The library includes both WHAM and LMALF analysis methods."
echo ""
echo "To enable verbose output, edit src/wham.cu and change:"
echo "  #define VERBOSE 0  ->  #define VERBOSE 1"

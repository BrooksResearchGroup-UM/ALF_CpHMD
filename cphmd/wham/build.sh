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

# Compile
cd "$SRC_DIR"
nvcc -shared -Xcompiler -fPIC -O3 -o "$OUTPUT" wham.cu

echo "Build successful: $OUTPUT"
echo ""
echo "To enable verbose output, edit src/wham.cu and change:"
echo "  #define VERBOSE 0  ->  #define VERBOSE 1"

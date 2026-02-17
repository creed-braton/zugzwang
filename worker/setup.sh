#!/bin/sh
ONNX_VERSION="1.24.1"
TMP_DIR=$(mktemp -d)
curl -L "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz" \
  | tar xz --strip-components=1 -C "$TMP_DIR"
mkdir -p lib/onnx
mv "$TMP_DIR/include" "$TMP_DIR/lib" lib/onnx/
rm -rf "$TMP_DIR"

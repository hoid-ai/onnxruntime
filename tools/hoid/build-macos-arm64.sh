#!/usr/bin/env bash
# Build the hoid-ai ONNX Runtime dylib for Apple silicon and package it in the same
# layout as csukuangfj/onnxruntime-libs (lib/libonnxruntime.<ver>.dylib + include/),
# which is what OpenWhispr's scripts/download-sherpa-onnx.js consumes.
#   tools/hoid/build-macos-arm64.sh [build_dir] [out_dir]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="${1:-$ROOT/build/hoid-macos-arm64}"
OUT="${2:-$ROOT/build/hoid-dist}"
VER="$(tr -d '[:space:]' < "$ROOT/VERSION_NUMBER")"
TAG="${HOID_RELEASE_TAG:-hoid-dev}"
DEPLOY="${MACOSX_DEPLOYMENT_TARGET:-15.5}"   # must equal OpenWhispr's PARAKEET_MINIMUM_MACOS_VERSION

cd "$ROOT"
./build.sh --config Release --build_shared_lib --parallel --skip_tests --compile_no_warning_as_error \
  --cmake_generator Ninja --osx_arch arm64 --apple_deploy_target "$DEPLOY" --build_dir "$BUILD" \
  --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 onnxruntime_BUILD_UNIT_TESTS=OFF --update --build

PKG="onnxruntime-osx-arm64-${VER}-${TAG}"
STAGE="$OUT/$PKG"
rm -rf "$STAGE"; mkdir -p "$STAGE/lib" "$STAGE/include"
cp "$BUILD/Release/libonnxruntime.${VER}.dylib" "$STAGE/lib/"
( cd "$STAGE/lib" && ln -s "libonnxruntime.${VER}.dylib" libonnxruntime.dylib )
cp -R include/onnxruntime/core/session/. "$STAGE/include/"
cp LICENSE ThirdPartyNotices.txt VERSION_NUMBER "$STAGE/"
git rev-parse HEAD > "$STAGE/GIT_COMMIT_ID"
codesign --force --sign - "$STAGE/lib/libonnxruntime.${VER}.dylib"
lipo -archs "$STAGE/lib/libonnxruntime.${VER}.dylib" | grep -qx arm64
( cd "$OUT" && rm -f "$PKG.zip" && zip -qry "$PKG.zip" "$PKG" )
shasum -a 256 "$OUT/$PKG.zip" | tee "$OUT/$PKG.zip.sha256"
stat -f "size_bytes=%z" "$OUT/$PKG.zip"

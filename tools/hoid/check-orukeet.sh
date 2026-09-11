#!/usr/bin/env bash
# Correctness + speed gate for a built dylib against the Orukeet ONNX encoder.
#   tools/hoid/check-orukeet.sh <dylib> <encoder.onnx> [reference.bin]
# Prints encoder time (4 threads) and, with a reference dump, the max abs output diff.
set -euo pipefail
DYLIB="$1"; MODEL="$2"; REF="${3:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
T="$(mktemp -d)"; mkdir -p "$T/lib"; cp "$DYLIB" "$T/lib/"; V="$(basename "$DYLIB")"
( cd "$T/lib" && ln -sf "$V" libonnxruntime.1.dylib && ln -sf "$V" libonnxruntime.dylib )
INC="$ROOT/include/onnxruntime/core/session"
clang++ -std=c++17 -O2 -arch arm64 -I"$INC" "$ROOT/tools/hoid/enc_check.cpp" -L"$T/lib" -lonnxruntime -Wl,-rpath,"$T/lib" -o "$T/enc_check"
"$T/enc_check" "$MODEL" 4 "$T/out.bin"
if [ -n "$REF" ]; then python3 - "$REF" "$T/out.bin" <<'PY'
import numpy as np,sys
a=np.fromfile(sys.argv[1],dtype=np.float32); b=np.fromfile(sys.argv[2],dtype=np.float32)
d=np.abs(a-b); print(f"vs reference: max|diff|={d.max():.6f} mean|diff|={d.mean():.7f}")
PY
fi
rm -rf "$T"

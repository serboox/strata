#!/bin/sh
# usage: vendor_ggml.sh <llama.cpp checkout> [destination root]
# Copies the part of ggml that this project builds (the base library, the CPU backend, and the ggml-cuda headers and MMQ
# instances Strata compiles itself) from a llama.cpp checkout into <root>/third_party/llama.cpp, with its licence and the
# commit it came from.  <root> defaults to the repository that holds this script.
set -eu
SRC="$1"
ROOT="${2:-$(cd "$(dirname "$0")/.." && pwd)}"
DST="$ROOT/third_party/llama.cpp"
[ -f "$SRC/ggml/CMakeLists.txt" ] || { echo "vendor_ggml: $SRC is not a llama.cpp checkout" >&2; exit 1; }

rm -rf "$DST"
mkdir -p "$DST/ggml/src/ggml-cuda/template-instances"
cp -r "$SRC/ggml/CMakeLists.txt" "$SRC/ggml/cmake" "$SRC/ggml/include" "$DST/ggml/"
cp "$SRC/ggml/src/CMakeLists.txt" "$DST/ggml/src/"
for f in "$SRC"/ggml/src/*.c "$SRC"/ggml/src/*.cpp "$SRC"/ggml/src/*.h "$SRC"/ggml/src/*.in; do
  cp "$f" "$DST/ggml/src/"
done
cp -r "$SRC/ggml/src/ggml-cpu" "$DST/ggml/src/"
for f in "$SRC"/ggml/src/ggml-cuda/*.cuh "$SRC"/ggml/src/ggml-cuda/quantize.cu; do
  cp "$f" "$DST/ggml/src/ggml-cuda/"
done
cp -r "$SRC/ggml/src/ggml-cuda/vendors" "$DST/ggml/src/ggml-cuda/"
for t in q8_0 q4_k q5_k q5_1 q6_k; do
  cp "$SRC/ggml/src/ggml-cuda/template-instances/mmq-instance-$t.cu" "$DST/ggml/src/ggml-cuda/template-instances/"
done
cp "$SRC/LICENSE" "$DST/LICENSE"
git -C "$SRC" rev-parse HEAD > "$DST/VERSION.txt"
git -C "$SRC" log -1 --format='%ad' >> "$DST/VERSION.txt"
echo "vendored $(head -1 "$DST/VERSION.txt") into $DST"

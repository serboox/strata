#!/bin/sh
# usage: build.sh [cmake configure arguments]   e.g. build.sh -DCMAKE_CUDA_ARCHITECTURES=86
# Builds build/strata-q36 and build/libstrata_kq.so.  STRATA_BUILD_JOBS caps the parallel compile jobs (default: 3/4 of
# the cores); the CUDA architecture defaults to the installed GPU.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
JOBS=${STRATA_BUILD_JOBS:-$(( ($(nproc) * 3 + 3) / 4 ))}
cmake -S "$ROOT" -B "$ROOT/build" -G Ninja "$@"
nice -n 10 cmake --build "$ROOT/build" -j "$JOBS"
echo "build: $ROOT/build/strata-q36"

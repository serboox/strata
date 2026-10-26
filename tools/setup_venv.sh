#!/bin/sh
# usage: setup_venv.sh   (PYTHON=python3.12 picks the interpreter)
# Creates .venv with the server's Python dependencies, every one pinned by hash in requirements.lock.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
"$PY" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/pip" install --quiet --require-hashes --no-deps -r "$ROOT/requirements.lock"
echo "setup_venv: $ROOT/.venv ready"

#!/bin/sh
# usage: fetch_model.sh   (no arguments)
# Downloads the pinned Qwen3.6-35B-A3B UD-Q4_K_XL file into models/qwen3.6-35b-a3b, resuming a partial download, and keeps
# it only if its size and SHA-256 match.  About 22.9 GB.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REV=5bc3e238d916f48a861bac2f8a1990a0e9b7e98d
BASE="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/$REV"
DEST="$ROOT/models/qwen3.6-35b-a3b"
NAME=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
SIZE=22853663008
SHA=55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1
mkdir -p "$DEST"

if [ -f "$DEST/$NAME" ] && [ "$(stat -c %s "$DEST/$NAME")" = "$SIZE" ]; then
  echo "fetch_model: $NAME is already there"
  exit 0
fi
tries=0
while [ "$tries" -lt 8 ]; do
  tries=$((tries + 1))
  have=$(stat -c %s "$DEST/$NAME.part" 2>/dev/null || echo 0)
  [ "$have" = "$SIZE" ] && break
  nice -n 10 curl -fL --retry 3 --retry-delay 10 --connect-timeout 30 -C - -o "$DEST/$NAME.part" "$BASE/$NAME" || sleep 20
done
have=$(stat -c %s "$DEST/$NAME.part" 2>/dev/null || echo 0)
[ "$have" = "$SIZE" ] || { echo "fetch_model: got $have of $SIZE bytes" >&2; exit 1; }
got=$(sha256sum "$DEST/$NAME.part" | cut -d' ' -f1)
if [ "$got" != "$SHA" ]; then
  echo "fetch_model: SHA-256 mismatch ($got), the file was not kept" >&2
  exit 1
fi
mv "$DEST/$NAME.part" "$DEST/$NAME"
echo "fetch_model: $NAME verified"

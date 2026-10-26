#!/bin/sh
# usage: run_tests.sh [module ...]   (default: every serve/test_*.py)
# Runs each test module in its own process and prints one line per module; the exit code is the number of failures.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
if [ "$#" -eq 0 ]; then
  set -- $(ls serve/test_*.py | sed 's|serve/||; s|\.py$||')
fi
failed=0
for module in "$@"; do
  if timeout 300 nice -n 10 "$PY" -m unittest "serve.$module" > /dev/null 2>&1; then
    echo "ok     $module"
  else
    echo "FAILED $module"
    failed=$((failed + 1))
  fi
done
exit "$failed"

#!/usr/bin/env bash
set -euo pipefail

# Grade starter_clear.py output against expected for window=3.
# Usage: ./grade.sh

DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python}"

INPUT="$DIR/sample.log"
EXPECTED="$DIR/expected_output_window3.txt"
TMP_OUT="${TMPDIR:-/tmp}/starter_clear_output.$$"

if [[ ! -f "$INPUT" ]]; then
  echo "Missing input: $INPUT" >&2
  exit 2
fi

if [[ ! -f "$EXPECTED" ]]; then
  echo "Missing expected output: $EXPECTED" >&2
  exit 2
fi

set +e
$PY "$DIR/starter_clear.py" --window 3 < "$INPUT" > "$TMP_OUT" 2> /dev/null
status=$?
set -e

if [[ $status -ne 0 ]]; then
  echo "Program exited with status $status" >&2
  rm -f "$TMP_OUT"
  exit 3
fi

if diff -u "$EXPECTED" "$TMP_OUT" >/dev/null; then
  echo "PASS"
  rm -f "$TMP_OUT"
  exit 0
else
  echo "FAIL: output differs from expected" >&2
  echo "--- diff ---"
  diff -u "$EXPECTED" "$TMP_OUT" || true
  echo "-----------"
  rm -f "$TMP_OUT"
  exit 1
fi


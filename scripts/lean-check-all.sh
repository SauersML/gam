#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "Refreshing Lean cache (if needed)..."
lake exe cache get >/dev/null

files=()
while IFS= read -r file; do
  files+=("$file")
done < <(find "$ROOT/src" -type f -name '*.lean' | sort)

if [ "${#files[@]}" -eq 0 ]; then
  echo "No .lean files found under src/."
  exit 0
fi

echo "Checking ${#files[@]} Lean files..."
unset LEAN_JOBS

if command -v getconf >/dev/null 2>&1; then
  JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
elif command -v sysctl >/dev/null 2>&1; then
  JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 2)"
else
  JOBS=2
fi

if [ -z "${JOBS:-}" ] || [ "$JOBS" -lt 1 ] 2>/dev/null; then
  JOBS=2
fi

export ROOT
printf '%s\0' "${files[@]}" | xargs -0 -P "$JOBS" -I{} bash -lc '
  f="$1"
  rel="${f#$ROOT/}"
  echo "[lean] $rel"
  lake env lean --root="$ROOT/src" "$f"
' _ {}

echo "All Lean files compiled successfully."

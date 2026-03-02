#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

files=()
while IFS= read -r file; do
  files+=("$file")
done < <(find "$ROOT" -type f -name '*.lean' \
  -not -path '*/.lake/*' \
  -not -path '*/target/*' \
  -not -name 'lakefile.lean' \
  | sort)

if [ "${#files[@]}" -eq 0 ]; then
  echo "No .lean files found."
  exit 0
fi

echo "Checking ${#files[@]} Lean files..."
for f in "${files[@]}"; do
  rel="${f#$ROOT/}"
  echo "[lean] $rel"
  lake env lean --root="$ROOT" "$f"
done

echo "All Lean files compiled successfully."

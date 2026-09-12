#!/bin/bash
# Fetch the creditscope layer-30 residual_post shards #2283 measures on, at one pinned
# Hugging Face dataset revision, verifying each file against its git-lfs sha256 before it
# is moved into place. A present file whose digest matches is kept, so a rerun resumes.
# CHUNK_DIR then holds chunk_0000.npy .. chunk_0007.npy (fp16, 360002 x 2048), the
# --chunk-dir every run_faithful_*.sbatch phase takes.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CHUNK_DIR" >&2
  exit 2
fi
DEST=$1
REVISION=2f03651767c5d061e7e21922b44272d3f7095f45
BASE="https://huggingface.co/datasets/sarel/creditscope-activations-v2/resolve/$REVISION/activations/layer_30_residual_post"

mkdir -p "$DEST"
trap 'rm -f "$DEST"/.chunk_*.npy.partial.$$' EXIT
while read -r NAME SHA256; do
  TARGET="$DEST/$NAME"
  if [[ -f "$TARGET" ]] && sha256sum --check --status <<<"$SHA256  $TARGET"; then
    echo "verified $NAME (present)"
    continue
  fi
  PARTIAL="$DEST/.$NAME.partial.$$"
  curl --fail --location --silent --show-error --output "$PARTIAL" "$BASE/$NAME" </dev/null
  if ! sha256sum --check --status <<<"$SHA256  $PARTIAL"; then
    echo "sha256 mismatch for $NAME at revision $REVISION" >&2
    exit 1
  fi
  mv "$PARTIAL" "$TARGET"
  echo "verified $NAME (fetched)"
done <<'MANIFEST'
chunk_0000.npy 230bfe251b51c424cf4acfe0868231d3047091980e0f73b74c83a88d8e6bc471
chunk_0001.npy 72466657f5ef5cdc935c836f46b46a4ba9b4b31e7f954f090b32c6ccd5d683d8
chunk_0002.npy d6f7e5b2099c062ff82aa1e1db451c46c1c6feeec185841739b1f46e180e90a5
chunk_0003.npy f1f68eddf4ecee498f4bc426f5ccf670a5baa3879277118f6f219e9e991b0762
chunk_0004.npy 1d9003fc538a0e12996dada6a1f5490ace03b47d028b627a7e6c4bfef9ab8929
chunk_0005.npy 5d80f1556ab24f8c345673ff46a925bb92a1b73929a4960d54bcd1abcfcb3964
chunk_0006.npy daf8372054ecbb5ae6c0baf5722535c1ef80ee8a06e17f2fa88824a325b96a82
chunk_0007.npy b7394a0da65a24471a2c481e9a69a54d34d165e70da8647c32bc20521a4a0368
MANIFEST

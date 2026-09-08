"""Convert Hugging Face activation caches into the ``residual_shard`` bf16 format
(``examples/residual_shard_io.py``) so the streaming block-sparse / TopK dictionary
drivers can read them shard-by-shard.

Two source layouts are supported (auto-detected by file extension):

  (a) **safetensors, token-level** — a ``(total_tokens, d_model)`` bf16 tensor per
      file. Target: ``caiovicentino1/Qwen3.6-35B-A3B-mcr-stage-b`` layer 17
      (~1.4M tokens, d_model 2048). Rows are read in slices via safetensors'
      zero-copy ``get_slice`` (never materialising the whole tensor).
  (b) **.npy chunks** — ``[~50k, d_model]`` fp16 arrays, one file per chunk.
      Target: ``sarel/creditscope-activations-v2`` (the cross-corpus eval set).
      Read with ``mmap_mode='r'`` so only the current row-batch is in RAM.

The corpus is NEVER held in memory: every file is streamed in ``--read-batch`` row
blocks straight into a :class:`ShardWriter`, which quantises to bf16 and rolls over
every ``--rows-per-shard`` rows, emitting a ``manifest.json`` on close.

Downloads happen on the compute node (``huggingface_hub``; point ``HF_HOME`` at
scratch). With ``--repo`` set, the matching files are ``snapshot_download``ed first;
without it, ``--files-glob`` is treated as a local path glob.

Usage (on MSI)::

    python hf_cache_to_shards.py \
        --repo caiovicentino1/Qwen3.6-35B-A3B-mcr-stage-b \
        --files-glob '*layer_17*.safetensors' --layer-tag 17 \
        --out-dir $SCRATCH/qwen_l17_shards --rows-per-shard 131072
    python hf_cache_to_shards.py \
        --repo sarel/creditscope-activations-v2 \
        --files-glob '*.npy' --out-dir $SCRATCH/creditscope_shards
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_shard_io import ShardWriter  # noqa: E402


def _resolve_files(repo: str | None, files_glob: str) -> list[str]:
    """Return the sorted list of source files. With ``repo`` set, snapshot-download
    the matching files first (HF_HOME/scratch), then glob the local snapshot."""
    if repo is not None:
        from huggingface_hub import snapshot_download

        local = snapshot_download(
            repo_id=repo,
            repo_type="dataset" if _looks_like_dataset(repo) else "model",
            allow_patterns=[files_glob],
        )
        pattern = os.path.join(local, "**", os.path.basename(files_glob))
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:  # some repos store activations at the root
            files = sorted(glob.glob(os.path.join(local, files_glob)))
        return files
    return sorted(glob.glob(files_glob, recursive=True))


def _looks_like_dataset(repo: str) -> bool:
    # Activation dumps are usually dataset repos; fall back to model on failure.
    return True


def _select_key(keys: list[str], layer_tag: str | None) -> str:
    """Pick the tensor key from a safetensors file. Prefer one containing
    ``layer_tag``; else the sole key; else error listing the candidates."""
    if layer_tag:
        hits = [k for k in keys if str(layer_tag) in k]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            # Most specific: the longest key that matches (e.g. "...layers.17...").
            return max(hits, key=len)
    if len(keys) == 1:
        return keys[0]
    raise ValueError(
        f"safetensors file has {len(keys)} tensors {keys[:6]}...; pass --layer-tag "
        "to disambiguate which one holds the activations"
    )


def _iter_safetensors_rows(path: str, layer_tag: str | None, batch: int) -> Iterator[np.ndarray]:
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        key = _select_key(list(f.keys()), layer_tag)
        sl = f.get_slice(key)
        shape = sl.get_shape()
        if len(shape) != 2:
            raise ValueError(f"{path}[{key}]: expected 2-D (tokens, d_model), got {shape}")
        n = int(shape[0])
        for start in range(0, n, batch):
            end = min(start + batch, n)
            # get_slice[...] is zero-copy per-row-range; .float() lifts bf16 -> f32.
            yield sl[start:end].float().cpu().numpy()


def _iter_npy_rows(path: str, batch: int) -> Iterator[np.ndarray]:
    mm = np.load(path, mmap_mode="r")
    if mm.ndim == 1:
        mm = mm.reshape(1, -1)
    if mm.ndim != 2:
        raise ValueError(f"{path}: expected 2-D (tokens, d_model), got shape {mm.shape}")
    n = mm.shape[0]
    for start in range(0, n, batch):
        yield np.ascontiguousarray(mm[start : start + batch], dtype=np.float32)


def _iter_rows(path: str, layer_tag: str | None, batch: int) -> Iterator[np.ndarray]:
    ext = Path(path).suffix.lower()
    if ext == ".safetensors":
        yield from _iter_safetensors_rows(path, layer_tag, batch)
    elif ext == ".npy":
        yield from _iter_npy_rows(path, batch)
    else:
        raise ValueError(f"{path}: unsupported cache format {ext!r} (want .safetensors or .npy)")


def _infer_d_model(path: str, layer_tag: str | None) -> int:
    ext = Path(path).suffix.lower()
    if ext == ".safetensors":
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            return int(f.get_slice(_select_key(list(f.keys()), layer_tag)).get_shape()[1])
    if ext == ".npy":
        mm = np.load(path, mmap_mode="r")
        return int(mm.shape[-1])
    raise ValueError(f"{path}: unsupported cache format")


def convert(
    *,
    repo: str | None,
    files_glob: str,
    layer_tag: str | None,
    out_dir: str,
    rows_per_shard: int,
    read_batch: int = 8192,
) -> dict:
    files = _resolve_files(repo, files_glob)
    if not files:
        raise SystemExit(f"no files matched (repo={repo!r}, glob={files_glob!r})")
    d_model = _infer_d_model(files[0], layer_tag)
    print(f"[convert] {len(files)} file(s), d_model={d_model}, "
          f"rows/shard={rows_per_shard}, out={out_dir}", flush=True)
    meta = {
        "source_repo": repo,
        "source_glob": files_glob,
        "layer_tag": layer_tag,
        "n_source_files": len(files),
    }
    total = 0
    with ShardWriter(out_dir, d_model, rows_per_shard=rows_per_shard, meta=meta) as w:
        for fi, path in enumerate(files):
            file_rows = 0
            for block in _iter_rows(path, layer_tag, read_batch):
                if block.shape[1] != d_model:
                    raise ValueError(
                        f"{path}: row width {block.shape[1]} != d_model {d_model}"
                    )
                w.append(block)
                file_rows += block.shape[0]
            total += file_rows
            print(f"[convert] ({fi+1}/{len(files)}) {os.path.basename(path)}: "
                  f"{file_rows} rows (cumulative {total})", flush=True)
        manifest = w.close()
    print(f"[convert] DONE: {manifest['total_tokens']} tokens across "
          f"{len(manifest['shards'])} shards -> {out_dir}", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=None,
                    help="HF repo id to snapshot_download the matching files from "
                         "(omit to treat --files-glob as a local path glob)")
    ap.add_argument("--files-glob", required=True,
                    help="glob for the cache files (e.g. '*layer_17*.safetensors' or '*.npy')")
    ap.add_argument("--layer-tag", default=None,
                    help="substring identifying the activation tensor key inside a "
                         "multi-tensor safetensors file (e.g. '17')")
    ap.add_argument("--out-dir", required=True, help="residual_shard output directory")
    ap.add_argument("--rows-per-shard", type=int, default=131072,
                    help="rows per shard file (default 131072)")
    ap.add_argument("--read-batch", type=int, default=8192,
                    help="row block size streamed from each source file")
    return ap


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    return convert(
        repo=args.repo,
        files_glob=args.files_glob,
        layer_tag=args.layer_tag,
        out_dir=args.out_dir,
        rows_per_shard=args.rows_per_shard,
        read_batch=args.read_batch,
    )


if __name__ == "__main__":
    main()

"""Tokenize FineWeb documents for the Qwen3 library fits (#2951), with VPD's qwen3_0_6b framing.

VPD's ``qwen3_0_6b`` config reads ``fineweb350bt_qwen3_docs1024_r9bb295dd_{train,eval}284_v1``:
FineWeb ``sample/350BT`` at revision 9bb295dd, the training split the first 284 source files and
the evaluation split from file 284 on, each document its text tokens followed by
``<|endoftext|>`` (151643) with no BOS, attention never crossing documents. Here one source file
gives the documents of one split (``--file 0`` training, ``--file 284`` held out), tokenized with
the Qwen3-0.6B tokenizer (the same ``tokenizer.json`` as Qwen3-1.7B), special-token spellings
inside the text kept as text: the tokenizer is ``tokenizer.json`` without its added tokens
(ids 151643 on, special or not, such as ``<|endoftext|>`` and ``<think>``), so every text token is
one of the byte-level BPE vocabulary's 151643 ids. On text without an added token's spelling this
is the Hugging Face tokenizer's own encoding.

The device programs run sequences of one length, so a sequence is a window of ``--context``
consecutive tokens inside one document, from the document's start: windows ``[s + kT, s + (k+1)T)``
of every document ``[s, e)`` with ``s + (k+1)T <= e``. Each token keeps exactly the context it has
under VPD's document-disjoint attention when its window is the first of its document; later
windows start mid-document, as VPD's rows do where a document crosses a row boundary.

Writes to ``--out``: ``documents.u32`` (every document's tokens, concatenated, little-endian),
``offsets.u64`` (document starts and the end), ``windows_T{T}.u32`` (rows of T tokens, in document
order, one file per ``--context`` length) and ``meta.json`` (source file and its sha256, tokenizer file sha256, framing, counts and
every output's sha256). Run once per split; the outputs are never rewritten.

usage: mpd_qwen3_fineweb_2951.py --file N --documents D --context T [T ...] --out DIR [--workers W]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "HuggingFaceFW/fineweb"
REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
SUBDIR = "sample/350BT"
MODEL = "Qwen/Qwen3-0.6B"
END_OF_TEXT = 151643

_tokenizer = None


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load():
    global _tokenizer
    if _tokenizer is None:
        from tokenizers import Tokenizer

        with open(hf_hub_download(MODEL, "tokenizer.json")) as fh:
            spec = json.load(fh)
        spec["added_tokens"] = []
        _tokenizer = Tokenizer.from_str(json.dumps(spec))
    return _tokenizer


def encode(texts):
    out = []
    for ids in (e.ids for e in _load().encode_batch(texts, add_special_tokens=False)):
        if any(t >= END_OF_TEXT for t in ids):
            raise SystemExit("a document's text encoded to a special token")
        out.append(np.asarray(ids + [END_OF_TEXT], dtype="<u4"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=int, required=True)
    parser.add_argument("--documents", type=int, required=True)
    parser.add_argument("--context", type=int, nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    if os.path.exists(os.path.join(args.out, "meta.json")):
        raise SystemExit(f"{args.out} is already written")
    os.makedirs(args.out, exist_ok=True)
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION) if f.startswith(SUBDIR + "/") and f.endswith(".parquet"))
    source = files[args.file]
    local = hf_hub_download(REPO, source, repo_type="dataset", revision=REVISION)
    # The first D documents, read a batch at a time: a source file holds several times more.
    texts = []
    for batch in pq.ParquetFile(local).iter_batches(columns=["text"]):
        texts.extend(batch.column("text").to_pylist())
        if len(texts) >= args.documents:
            break
    texts = texts[: args.documents]
    chunk = 1000
    with ProcessPoolExecutor(args.workers) as pool:
        documents = [d for part in pool.map(encode, [texts[i : i + chunk] for i in range(0, len(texts), chunk)]) for d in part]
    lengths = np.array([len(d) for d in documents], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype("<u8")
    stream = np.concatenate(documents).astype("<u4")
    paths = {"documents": "documents.u32", "offsets": "offsets.u64"}
    stream.tofile(os.path.join(args.out, paths["documents"]))
    offsets.tofile(os.path.join(args.out, paths["offsets"]))
    windows = {}
    for T in args.context:
        rows = [stream[s + k * T : s + (k + 1) * T] for s, n in zip(offsets[:-1], lengths) for k in range(n // T)]
        paths[f"windows_T{T}"] = f"windows_T{T}.u32"
        np.stack(rows).astype("<u4").tofile(os.path.join(args.out, paths[f"windows_T{T}"]))
        windows[T] = {"rows": len(rows), "token_fraction": len(rows) * T / stream.size}
    tokenizer_file = hf_hub_download(MODEL, "tokenizer.json")
    meta = {
        "source": {"repo": REPO, "revision": REVISION, "file": source, "file_index": args.file, "sha256": sha256(local)},
        "tokenizer": {"model": MODEL, "tokenizer_json_sha256": sha256(tokenizer_file)},
        "framing": "text tokens (special-token spellings kept as text), then <|endoftext|> 151643; no BOS",
        "documents": len(documents),
        "tokens": int(stream.size),
        "windows": windows,
        "files": {k: {"path": v, "sha256": sha256(os.path.join(args.out, v))} for k, v in paths.items()},
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()

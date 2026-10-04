"""Qwen3 at scale (#2951): the model and token rows the pipeline runs on.

Downloads a Qwen3 checkpoint from Hugging Face and one FineWeb sample shard (the corpus VPD's
Qwen3 runs use: `fineweb350bt_qwen3_docs1024`, FineWeb text framed as text + <|endoftext|>,
no BOS, documents of 1024 tokens), tokenizes documents with the model's tokenizer, and keeps the
first `context` tokens of every document at least that long. Rows are split into a fit set and a
disjoint evaluation set, written as raw little-endian float64 token rows.

usage: qwen3_data.py OUT_DIR [--model Qwen/Qwen3-0.6B] [--fit 2048] [--eval 512] [--context 1024]
writes OUT_DIR/{model,tokens_fit.f64,tokens_eval.f64,tokens.json}
"""
import argparse
import json
from pathlib import Path

import numpy as np

ENDOFTEXT = 151643
FINEWEB = ("HuggingFaceFW/fineweb", "sample/10BT/000_00000.parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--fit", type=int, default=2048)
    parser.add_argument("--eval", type=int, default=512)
    parser.add_argument("--context", type=int, default=1024)
    args = parser.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download, snapshot_download

    model_dir = snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"])
    link = out / "model"
    if not link.exists():
        link.symlink_to(model_dir)
    shard = hf_hub_download(FINEWEB[0], FINEWEB[1], repo_type="dataset")
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    need = args.fit + args.eval
    rows = []
    table = pq.ParquetFile(shard)
    for group in range(table.num_row_groups):
        texts = table.read_row_group(group, columns=["text"]).column("text").to_pylist()
        for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
            ids = ids + [ENDOFTEXT]
            if len(ids) >= args.context:
                rows.append(ids[: args.context])
        if len(rows) >= need:
            break
    rows = np.array(rows[:need], dtype=np.float64)
    order = np.random.default_rng(0).permutation(len(rows))
    rows[order[: args.fit]].astype("<f8").tofile(out / "tokens_fit.f64")
    rows[order[args.fit:]].astype("<f8").tofile(out / "tokens_eval.f64")
    json.dump({"model": args.model, "model_dir": str(model_dir), "corpus": f"{FINEWEB[0]}:{FINEWEB[1]}",
               "framing": "text + <|endoftext|> (151643), no BOS; first `context` tokens of documents at least that long",
               "context": args.context, "fit_rows": args.fit, "eval_rows": int(len(rows) - args.fit)}, open(out / "tokens.json", "w"), indent=1)
    print(f"{len(rows)} rows of {args.context} tokens -> {out}")


if __name__ == "__main__":
    main()

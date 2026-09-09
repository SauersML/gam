"""Harvest residual-stream activations from Qwen3.5-4B-Base (issue #2502).

Thin PyTorch wrapper: it runs forward passes and writes bytes. No modelling
math lives here -- the dictionary is fitted by the Rust engine
(`crates/gam-sae/examples/issue_2502_overcomplete_llm.rs`).

Outputs, into --out-dir:
  train.npy / eval.npy    float32 (rows x hidden_size), C-order
  train_tokens.npy        int32 token id per train row
  eval_tokens.npy         int32 token id per eval row
  eval_ctx.npy            int32 (rows x 2): (sequence index, position in sequence)
  eval_seqs.npy           int32 (n_seq x seq_len) the held-out token windows
  meta.json               shapes, model/corpus revisions, split contract

The split is by DOCUMENT: a document's tokens land entirely in train or
entirely in eval, so no eval row shares a document with a train row.
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip-positions", type=int, default=8)
    ap.add_argument("--train-rows", type=int, default=300000)
    ap.add_argument("--eval-rows", type=int, default=100000)
    ap.add_argument("--eval-doc-frac", type=float, default=0.25)
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()


def doc_is_eval(doc_id: int, frac: float) -> bool:
    """Deterministic document-level split: md5 of the decimal document index."""
    digest = hashlib.md5(str(doc_id).encode("ascii")).hexdigest()
    return (int(digest[:8], 16) / 0x100000000) < frac


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    model.cuda()

    # The residual stream lives on the text decoder's block list; the class is a
    # conditional-generation wrapper, so walk to the block list rather than
    # assuming a path. Keep the module that OWNS the block list too: calling it
    # directly skips the 248k-way vocabulary head, which is pure waste here.
    blocks = None
    for path in ("model.language_model", "model", "language_model.model"):
        node = model
        try:
            for part in path.split("."):
                node = getattr(node, part)
            blocks = node.layers
            backbone = node
            block_path = path + ".layers"
            break
        except AttributeError:
            continue
    if blocks is None:
        raise SystemExit(f"could not locate decoder blocks on {type(model).__name__}")
    hidden = int(getattr(cfg, "hidden_size", 0) or cfg.text_config.hidden_size)
    print(
        f"[harvest] model={args.model} blocks={len(blocks)} at {block_path} "
        f"hidden={hidden} layer={args.layer}",
        flush=True,
    )

    captured = {}

    def hook(_module, _inputs, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output

    handle = blocks[args.layer].register_forward_hook(hook)

    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)

    want_train = args.train_rows
    want_eval = args.eval_rows
    keep = args.seq_len - args.skip_positions

    train_path = os.path.join(args.out_dir, "train.npy")
    eval_path = os.path.join(args.out_dir, "eval.npy")
    train_mm = np.lib.format.open_memmap(
        train_path, mode="w+", dtype=np.float32, shape=(want_train, hidden)
    )
    eval_mm = np.lib.format.open_memmap(
        eval_path, mode="w+", dtype=np.float32, shape=(want_eval, hidden)
    )
    train_tok = np.zeros(want_train, dtype=np.int32)
    eval_tok = np.zeros(want_eval, dtype=np.int32)
    eval_ctx = np.zeros((want_eval, 2), dtype=np.int32)
    eval_seq_list = []

    n_train = 0
    n_eval = 0
    n_docs = 0
    n_rows_seen = 0
    used_train_docs = set()
    used_eval_docs = set()
    pending = {"train": [], "eval": []}

    def flush(kind):
        nonlocal n_train, n_eval
        batch = pending[kind]
        if not batch:
            return
        ids = torch.tensor([b[0] for b in batch], dtype=torch.long, device="cuda")
        with torch.no_grad():
            backbone(input_ids=ids, use_cache=False)
        acts = captured["h"][:, args.skip_positions :, :].to(torch.float32).cpu().numpy()
        toks = ids[:, args.skip_positions :].cpu().numpy().astype(np.int32)
        for row_block, tok_block, (window, _doc) in zip(acts, toks, batch):
            if kind == "train":
                take = min(keep, want_train - n_train)
                if take <= 0:
                    break
                train_mm[n_train : n_train + take] = row_block[:take]
                train_tok[n_train : n_train + take] = tok_block[:take]
                n_train += take
            else:
                take = min(keep, want_eval - n_eval)
                if take <= 0:
                    break
                seq_index = len(eval_seq_list)
                eval_seq_list.append(np.asarray(window, dtype=np.int32))
                eval_mm[n_eval : n_eval + take] = row_block[:take]
                eval_tok[n_eval : n_eval + take] = tok_block[:take]
                eval_ctx[n_eval : n_eval + take, 0] = seq_index
                eval_ctx[n_eval : n_eval + take, 1] = np.arange(
                    args.skip_positions, args.skip_positions + take, dtype=np.int32
                )
                n_eval += take
        pending[kind] = []

    article_id = -1
    article_kind = None
    buf: list[int] = []

    def emit_article():
        """Window the current article's token stream and queue the windows."""
        nonlocal buf
        if article_kind is None:
            buf = []
            return
        for start in range(0, len(buf) - args.seq_len + 1, args.seq_len):
            pending[article_kind].append((buf[start : start + args.seq_len], article_id))
            if len(pending[article_kind]) >= args.batch:
                flush(article_kind)
        buf = []

    for record in ds:
        text = record["text"]
        # wikitext-103-raw-v1 delimits ARTICLES with a single-equals heading line
        # (" = Title = "); sub-sections use two or more. Grouping paragraphs into
        # articles is what makes the held-out split leak-free at the article
        # level rather than only at the paragraph level, and it is also what
        # lets a full seq_len window exist at all (single paragraphs are short).
        stripped = text.strip()
        # " = Title = " is an ARTICLE heading; " = = Section = = " and deeper
        # are sub-headings of the article that precedes them. Testing for "= ="
        # (with the space) is what separates the two after .strip().
        is_header = (
            stripped.startswith("= ")
            and stripped.endswith(" =")
            and not stripped.startswith("= =")
        )
        if is_header:
            emit_article()
            article_id += 1
            n_docs += 1
            article_kind = "eval" if doc_is_eval(article_id, args.eval_doc_frac) else "train"
            if article_kind == "train":
                used_train_docs.add(article_id)
            else:
                used_eval_docs.add(article_id)
            continue
        if article_kind is None or not stripped:
            continue
        if article_kind == "train" and n_train >= want_train:
            continue
        if article_kind == "eval" and n_eval >= want_eval:
            continue
        buf.extend(tok(text, add_special_tokens=False)["input_ids"])
        n_rows_seen += 1
        if n_train >= want_train and n_eval >= want_eval:
            break
        if n_rows_seen % 5000 == 0:
            print(
                f"[harvest] rows_seen={n_rows_seen} articles={article_id + 1} "
                f"train_rows={n_train} eval_rows={n_eval} t={time.time() - t0:.0f}s",
                flush=True,
            )
    emit_article()
    flush("train")
    flush("eval")
    handle.remove()

    train_mm.flush()
    eval_mm.flush()
    del train_mm, eval_mm
    np.save(os.path.join(args.out_dir, "train_tokens.npy"), train_tok[:n_train])
    np.save(os.path.join(args.out_dir, "eval_tokens.npy"), eval_tok[:n_eval])
    np.save(os.path.join(args.out_dir, "eval_ctx.npy"), eval_ctx[:n_eval])
    np.save(
        os.path.join(args.out_dir, "eval_seqs.npy"),
        np.stack(eval_seq_list) if eval_seq_list else np.zeros((0, args.seq_len), np.int32),
    )

    meta = {
        "model": args.model,
        "dataset": f"{args.dataset}/{args.dataset_config}:{args.split}",
        "layer": args.layer,
        "block_path": block_path,
        "hidden_size": hidden,
        "seq_len": args.seq_len,
        "skip_positions": args.skip_positions,
        "train_rows": int(n_train),
        "eval_rows": int(n_eval),
        "eval_sequences": len(eval_seq_list),
        "articles_seen": n_docs,
        "articles_used_train": len(used_train_docs),
        "articles_used_eval": len(used_eval_docs),
        "articles_in_both": len(used_train_docs & used_eval_docs),
        "eval_doc_frac": args.eval_doc_frac,
        "split_rule": "md5(str(doc_index))[:8]/2**32 < eval_doc_frac",
        "torch": torch.__version__,
        "wall_seconds": time.time() - t0,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("[harvest] " + json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()

"""A mechanistically faithful natural-language autoencoder of VPD-4L's per-word computation (#2951).

The thing described is not an activation but the program that ran on a word: the set of VPD's
rank-one subcomponents (weight slices) that were on there (frontier/masks_vpd4l.npz, gate > 0).

  encoder   on-set -> concepts it invokes (gam_mpd::concepts, the MDL fit in Rust) -> text: each
            concept's label, read off its members' weights, layer by layer
  decoder   text -> concepts (exact lookup of the labels, nothing else: it never sees the input
            and never runs the network) -> their core members -> the masked model
  score     the objective's own currency: bits of the text under a fixed language model
            (Qwen2.5-1.5B-Instruct, each word's line given the earlier lines of its sequence)
            + n KL(model || masked model running the decoded program) / ln 2, against the true
            on-set's code (each subcomponent at its own rate) + n KL of the true on-set.

Stages (each writes into OUT = ~/mpd-data/nlae):
  labels    per subcomponent: what it reads (direct path from the embedding through its read
            direction), what it writes to the logits (its write direction carried to the residual
            stream, then the unembedding), and the words it fires on (lift over the train rows);
            the short label; Qwen bits of a sample of labels -> label_bits.json
  [rust]    cargo run --release -p gam-mpd --example mpd_nl_concepts_2951 -- \
                OUT/sets 32:128 0:32 LABEL_BITS OUT/fit
  attrib    per eval word, each subcomponent's first-order KL attribution at VPD's own set
            (one reverse pass of the row's summed KL through the masks)
  text      concept labels; per eval word and per n of the ladder, the items whose attributed KL
            bits n dKL/ln 2 exceed their label bits (the objective's own encoder), as text
  bits      Qwen bits of every text (and of the raw input words: the leak baseline)
  kl        the masked model running each decoded program, KL per word
  figure    the figure

usage: MPD_MEM_GIB=4 venv python vpd_nl_autoencoder.py STAGE
"""

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path.home() / "mpd-data/nlae"
VD = Path.home() / "mpd-data/vpd"
MASKS = Path.home() / "mpd-data/frontier/masks_vpd4l.npz"
TRAIN, EVAL = (32, 128), (0, 32)
TEXT_ROWS = 8  # eval rows whose text is scored by the language model (the KL uses all of EVAL)
CONTEXT = 512
LADDER = [0, 4, 16, 64, 256, 1024, 4096, 16384, math.inf]  # n of the encoder's choice; inf = every item
N_REPORT = 1024  # n of the headline totals (the 4L frontier's)
KIND = {"q_proj": "query", "k_proj": "key", "v_proj": "value", "o_proj": "attn-out", "c_fc": "mlp-in", "down_proj": "mlp-out"}
HEADER = "Each line names the weight mechanisms of a 4-layer language model that ran on one word.\n"


def sets():
    z = np.load(MASKS)
    return z, z["vpd_indptr"], z["vpd_indices"].astype(np.int64), z["vpd_offsets"], [str(n) for n in z["site_names"]]


def site_of(offsets):
    return np.searchsorted(offsets, np.arange(offsets[-1]), side="right") - 1


def show(s: str) -> str:
    s = s.strip() or s.replace("\n", "\\n").replace(" ", "␣")
    return s.replace("\n", "\\n")[:12]


# ------------------------------------------------------------------ the model, light


def load_light(dev="mps"):
    """The target with VPD's subcomponents installed in its sites (no causal-importance network:
    nothing here runs it). Returns the target and each site's subcomponent count."""
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vpd_model import VPD_PTH, load_target

    target = load_target(dev)
    raw = torch.load(str(VPD_PTH), map_location="cpu", weights_only=True, mmap=True)
    C = {}
    for k, v in raw.items():
        if k.startswith("_components."):
            site, which = k[len("_components."):].rsplit(".", 1)
            name = site.replace("-", ".")
            setattr(target.site(name), which, v.float().to(dev))
            if which == "U":
                C[name] = v.shape[0]
    del raw
    return target, C


def masked(target, ids, masks):
    """Logits of the target with each site's subcomponents gated by masks[name] [B, S, C]."""
    try:
        for n, m in masks.items():
            target.site(n).mask = m
        return target(ids)
    finally:
        for n in masks:
            target.site(n).mask = None


# ------------------------------------------------------------------ labels


def stage_labels():
    import torch
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(VD / "t-9d2b8f02/tokenizer.json"))
    vocab = [tok.decode([i]) for i in range(tok.get_vocab_size())]
    z, indptr, indices, offsets, names = sets()
    ids = z["ids"]
    N, V = int(offsets[-1]), len(vocab)
    site = site_of(offsets)

    # The words each subcomponent fires on, over the train rows: count and lift.
    lo, hi = indptr[TRAIN[0] * CONTEXT], indptr[TRAIN[1] * CONTEXT]
    words = ids[TRAIN[0]:TRAIN[1]].reshape(-1)
    per = np.diff(indptr[TRAIN[0] * CONTEXT:TRAIN[1] * CONTEXT + 1])
    pair = indices[lo:hi] * V + np.repeat(words, per)
    keys, counts = np.unique(pair, return_counts=True)
    fired = np.bincount(indices[lo:hi], minlength=N)
    base = np.bincount(words, minlength=V) / len(words)
    j_of, t_of = keys // V, keys % V
    lift = counts / (fired[j_of] * base[t_of])
    score = counts * np.log(np.maximum(lift, 1.0))
    order = np.lexsort((-score, j_of))
    contexts = [[] for _ in range(N)]
    for k in order:
        j = j_of[k]
        if len(contexts[j]) < 4 and lift[k] > 1.0:
            contexts[j].append((int(t_of[k]), int(counts[k]), float(lift[k])))

    dev = "mps"
    target, _ = load_light(dev)
    wte = target.wte
    E = wte * torch.rsqrt(wte.pow(2).mean(-1, keepdim=True) + target.eps)  # rms-normed embeddings
    W = lambda n: target.site(n).W  # [d_out, d_in]
    reads, writes, signs = [None] * N, [None] * N, np.ones(N)
    for s, n in enumerate(names):
        layer, kind = int(n.split(".")[1]), n.split(".")[-1]
        g = target.norms[2 * layer + (kind in ("c_fc", "down_proj"))]
        Vs, Us = target.site(n).V, target.site(n).U  # [d_in, C], [C, d_out]
        # Read directions carried back to the residual stream at the site's norm (direct path).
        if kind == "o_proj":
            R = W(f"h.{layer}.attn.v_proj").T @ Vs
        elif kind == "down_proj":
            R = W(f"h.{layer}.mlp.c_fc").T @ Vs
        else:
            R = Vs
        R = R * g[:, None]
        # Write directions carried forward to the residual stream (none for query/key).
        if kind in ("o_proj", "down_proj"):
            Wr = Us
        elif kind == "c_fc":
            Wr = Us @ W(f"h.{layer}.mlp.down_proj").T
        elif kind == "v_proj":
            Wr = Us @ W(f"h.{layer}.attn.o_proj").T
        else:
            Wr = None
        C = Vs.shape[1]
        for c0 in range(0, C, 256):
            c1 = min(C, c0 + 256)
            rs = E @ R[:, c0:c1]  # [V, chunk]
            ws = (Wr[c0:c1] * target.ln_f) @ wte.T if Wr is not None else None
            for c in range(c0, c1):
                j = offsets[s] + c
                col = rs[:, c - c0]
                # Orient by the word it fires on most: that word reads positive.
                if contexts[j]:
                    signs[j] = 1.0 if col[contexts[j][0][0]].item() >= 0 else -1.0
                top = torch.topk(signs[j] * col, 4).indices.tolist()
                reads[j] = top
                if ws is not None:
                    writes[j] = torch.topk(signs[j] * ws[c - c0], 4).indices.tolist()
            del rs, ws
        if dev == "mps":
            torch.mps.empty_cache()
        print(f"{n}: labelled", flush=True)

    # Do the weights read what the subcomponent fires on? (direct path, top 4 of the vocabulary)
    agree, alive = 0, 0
    label = []
    for j in range(N):
        n = names[site[j]]
        layer, kind = int(n.split(".")[1]), n.split(".")[-1]
        ctx = [show(vocab[t]) for t, _, _ in contexts[j][:2]]
        wr = [show(vocab[t]) for t in (writes[j] or [])[:2]]
        if fired[j] > 0:
            alive += 1
            agree += bool(contexts[j]) and contexts[j][0][0] in reads[j]
        lab = f"L{layer} {KIND[kind]}: {'|'.join(ctx) or '?'}"
        if wr:
            lab += f" → {'|'.join(wr)}"
        label.append(lab)
    records = [
        {
            "site": names[site[j]], "index": int(j - offsets[site[j]]), "fired": int(fired[j]), "sign": signs[j],
            "contexts": [[vocab[t], c, round(l, 2)] for t, c, l in contexts[j]],
            "reads": [vocab[t] for t in reads[j]], "writes": [vocab[t] for t in (writes[j] or [])],
            "label": label[j],
        }
        for j in range(N)
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"read_agrees_with_context": agree / alive, "alive": alive, "subcomponents": records}, open(OUT / "labels.json", "w"))
    print(f"labels: {alive} alive; the top fired-on word is among the top-4 direct-path reads for {agree / alive:.1%}")


# ------------------------------------------------------------------ language-model bits


class LM:
    """Qwen2.5-1.5B-Instruct as a fixed code: bits of each line of a document given the earlier lines."""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("mps").eval()

    def nll(self, x, mask, chunk: int = 256):
        """Bits of each token after the first, [B, L - 1], the vocabulary's logits a chunk of
        positions at a time (never all positions x 152k at once)."""
        torch = self.torch
        with torch.no_grad():
            h = self.model.model(input_ids=x, attention_mask=mask).last_hidden_state
            out = []
            for a in range(0, x.shape[1] - 1, chunk):
                b = min(x.shape[1] - 1, a + chunk)
                logits = self.model.lm_head(h[:, a:b]).float()
                lp = logits.gather(-1, x[:, a + 1:b + 1, None])[..., 0] - torch.logsumexp(logits, -1)
                out.append(-lp / math.log(2))
            return torch.cat(out, 1)

    def line_bits(self, lines: list[str], window: int = 2048, header: str = HEADER, sep: str = "\n") -> np.ndarray:
        """Bits of each line (its text and its newline) given the header and the lines before it,
        scored in windows of `window` tokens that overlap by half (each token scored once, with at
        least half a window of context)."""
        torch = self.torch
        doc = header + "".join(l + sep for l in lines)
        enc = self.tok(doc, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        starts = np.cumsum([len(header)] + [len(l) + len(sep) for l in lines])  # char start of each line, then end
        owner = np.searchsorted(starts, [a for a, _ in enc["offset_mapping"]], side="right") - 1  # -1 = header
        bits = np.zeros(len(ids))
        stride = window // 2
        done = 1  # the first token has no prediction; it is in the header
        while done < len(ids):
            lo = max(0, done - stride) if done > 1 else 0
            hi = min(len(ids), lo + window)
            x = torch.tensor([ids[lo:hi]], device="mps")
            nll = self.nll(x, torch.ones_like(x))[0].cpu().numpy()
            # positions lo+1 .. hi-1 predicted; keep those from `done`
            keep = np.arange(lo + 1, hi)
            sel = keep >= done
            bits[keep[sel]] = nll[sel]
            done = hi
        out = np.zeros(len(lines))
        valid = owner >= 0
        np.add.at(out, np.minimum(owner[valid], len(lines) - 1), bits[valid])
        return out

    def standalone_bits(self, texts: list[str], batch: int = 16, header: str = HEADER) -> np.ndarray:
        """Bits of each text as the only line after the header."""
        torch = self.torch
        h = self.tok(header, add_special_tokens=False)["input_ids"]
        out = np.zeros(len(texts))
        for b in range(0, len(texts), batch):
            chunk = [self.tok(t + "\n", add_special_tokens=False)["input_ids"] for t in texts[b:b + batch]]
            L = max(len(c) for c in chunk) + len(h)
            pad = self.tok.pad_token_id or 0
            x = torch.full((len(chunk), L), pad, device="mps")
            m = torch.zeros((len(chunk), L), device="mps")
            for i, c in enumerate(chunk):
                x[i, :len(h) + len(c)] = torch.tensor(h + c, device="mps")
                m[i, :len(h) + len(c)] = 1
            nll = self.nll(x, m)
            for i, c in enumerate(chunk):
                out[b + i] = nll[i, len(h) - 1:len(h) - 1 + len(c)].sum().item()
        return out


def stage_label_bits():
    lab = json.load(open(OUT / "labels.json"))["subcomponents"]
    alive = [r["label"] for r in lab if r["fired"] > 0]
    rng = np.random.default_rng(0)
    sample = [alive[i] for i in rng.choice(len(alive), size=min(2048, len(alive)), replace=False)]
    bits = LM().standalone_bits(sample)
    json.dump({"median": float(np.median(bits)), "mean": float(bits.mean()), "sample": len(sample)}, open(OUT / "label_bits.json", "w"))
    print(f"label bits: median {np.median(bits):.1f}, mean {bits.mean():.1f} over {len(sample)} labels")


# ------------------------------------------------------------------ the fitted concepts


def fit_outputs():
    """The Rust fit: concepts (members, rates), per word the concepts it invokes (train words then
    eval words) and its bits (choices, members, alone, independent)."""
    rep = json.load(open(OUT / "fit/concepts.json"))
    ptr = np.fromfile(OUT / "fit/invoked.indptr.i64", dtype="<i8")
    idx = np.fromfile(OUT / "fit/invoked.indices.i64", dtype="<i8")
    bits = np.fromfile(OUT / "fit/bits.f64", dtype="<f8").reshape(-1, 4)
    return rep, ptr, idx, bits


def eval_word(r, p):
    """Index of eval row r, position p among the coded words (train words first)."""
    return (TRAIN[1] - TRAIN[0]) * CONTEXT + r * CONTEXT + p


def stage_attrib():
    """First-order KL attribution of every subcomponent at VPD's own set, per eval word: the
    derivative of the row's summed KL(model || masked) in each mask entry, kept where the set is on."""
    import torch

    z, indptr, indices, offsets, names = sets()
    dev = "mps"
    target, C = load_light(dev)
    lo = indptr[EVAL[0] * CONTEXT]
    out = np.zeros(indptr[EVAL[1] * CONTEXT] - lo, dtype=np.float32)
    for r in range(EVAL[0], EVAL[1]):
        ids = torch.tensor(z["ids"][r:r + 1], device=dev)
        with torch.no_grad():
            tgt = torch.log_softmax(target(ids).float(), -1)
        a, b = indptr[r * CONTEXT], indptr[(r + 1) * CONTEXT]
        pos = np.repeat(np.arange(CONTEXT), np.diff(indptr[r * CONTEXT:(r + 1) * CONTEXT + 1]))
        glob = indices[a:b]
        masks = {}
        for s, n in enumerate(names):
            m = torch.zeros(1, CONTEXT, C[n], device=dev)
            sel = (glob >= offsets[s]) & (glob < offsets[s + 1])
            m[0, torch.tensor(pos[sel], device=dev), torch.tensor(glob[sel] - offsets[s], device=dev)] = 1.0
            masks[n] = m.requires_grad_(True)
        logits = masked(target, ids, masks)
        kl = (tgt.exp() * (tgt - torch.log_softmax(logits.float(), -1))).sum()
        kl.backward()
        g = np.zeros(b - a, dtype=np.float32)
        for s, n in enumerate(names):
            sel = np.nonzero((glob >= offsets[s]) & (glob < offsets[s + 1]))[0]
            G = masks[n].grad[0]
            g[sel] = G[torch.tensor(pos[sel], device=dev), torch.tensor(glob[sel] - offsets[s], device=dev)].cpu().numpy()
        out[a - lo:b - lo] = g
        del masks, logits, kl
        torch.mps.empty_cache()
        print(f"row {r}: summed KL {float(z['kl_vpd'][r].sum()):.1f} nats, attributed drop {-g.sum():.1f}", flush=True)
    np.save(OUT / "attrib.npy", out)


def concept_labels(rep, ptr, idx, lab, vocab_of_word):
    """Each concept's label: where its members sit, the words that invoke it (lift over the train
    words) and what its members write to the logits (their top writes, weighted by their rates)."""
    words = vocab_of_word
    T = (TRAIN[1] - TRAIN[0]) * CONTEXT
    base = Counter(words)
    users = [[] for _ in rep["concepts"]]
    for t in range(T):
        for c in idx[ptr[t]:ptr[t + 1]]:
            users[c].append(words[t])
    labels = []
    for c, con in enumerate(rep["concepts"]):
        members = con["members"]
        sites = [lab[j]["site"] for j in members]
        layers = sorted({int(n.split(".")[1]) for n in sites})
        parts = sorted({("attn" if ".attn." in n else "mlp") for n in sites}, key=["attn", "mlp"].index)
        where = (f"L{layers[0]}" if len(layers) == 1 else f"L{layers[0]}-{layers[-1]}") + " " + "+".join(parts)
        cnt = Counter(users[c])
        score = {w: k * math.log(max(1.0, k * T / (len(users[c]) * base[w]))) for w, k in cnt.items()} if users[c] else {}
        ctx = sorted(score, key=lambda w: -score[w])[:3]
        vote = Counter()
        for j, q in zip(members, con["on"]):
            for rank, w in enumerate(lab[j]["writes"]):
                vote[w] += q * (4 - rank)
        wr = [w for w, _ in vote.most_common(2)]
        labels.append((where, [show(w) for w in ctx], [show(w) for w in wr]))
    # Unique names (the decoder is a lookup): a third invoking word, then a number, where needed.
    text = [f"{w}: {'|'.join(c[:2]) or '?'}" + (f" → {'|'.join(r)}" if r else "") for w, c, r in labels]
    seen = Counter(text)
    for i, (w, c, r) in enumerate(labels):
        if seen[text[i]] > 1 and len(c) > 2:
            text[i] = f"{w}: {'|'.join(c)}" + (f" → {'|'.join(r)}" if r else "")
    seen, k = Counter(text), Counter()
    for i in range(len(text)):
        if seen[text[i]] > 1:
            k[text[i]] += 1
            text[i] = text[i] + f" #{k[text[i]]}"
    return text


def stage_text():
    """Per eval word: its items (invoked concepts, decoded to their core; every subcomponent that
    ran outside the decoded cores, named alone), their attributed KL, and the item labels."""
    z, indptr, indices, offsets, names = sets()
    rep, ptr, idx, bits = fit_outputs()
    lab = json.load(open(OUT / "labels.json"))["subcomponents"]
    words = [w for w in vocab_words(z["ids"][TRAIN[0]:TRAIN[1]].reshape(-1))]
    clabels = concept_labels(rep, ptr, idx, lab, words)
    single = [r["label"] for r in lab]
    # Unique single names too: the site's index where two collide.
    seen = Counter(single)
    single = [l if seen[l] == 1 else f"{l} #{r['index']}" for l, r in zip(single, lab)]
    core = [[j for j, q in zip(c["members"], c["on"]) if q > 0.5] for c in rep["concepts"]]
    g = np.load(OUT / "attrib.npy")
    lo = indptr[EVAL[0] * CONTEXT]
    items = []  # per eval word: list of (kind, id, importance nats)
    for r in range(EVAL[1] - EVAL[0]):
        for p in range(CONTEXT):
            t = (EVAL[0] + r) * CONTEXT + p
            a, b = indptr[t], indptr[t + 1]
            S = indices[a:b]
            gs = dict(zip(S.tolist(), g[a - lo:b - lo].tolist()))
            w = eval_word(r, p)
            inv = idx[ptr[w]:ptr[w + 1]].tolist()
            covered = set()
            it = []
            for c in inv:
                covered.update(core[c])
                it.append(("c", c, -sum(gs.get(j, 0.0) for j in core[c])))
            for j in S.tolist():
                if j not in covered:
                    it.append(("s", j, -gs[j]))
            items.append(it)
    json.dump({"concept_labels": clabels, "single_labels": single, "core": core, "items": items}, open(OUT / "items.json", "w"))
    n_inv = np.diff(ptr)[eval_word(0, 0):eval_word(EVAL[1] - EVAL[0], 0)]
    print(f"{len(clabels)} concepts; eval words invoke {n_inv.mean():.1f} concepts and name {np.mean([sum(k == 's' for k, _, _ in it) for it in items]):.1f} subcomponents alone")


def vocab_words(ids):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(VD / "t-9d2b8f02/tokenizer.json"))
    return [tok.decode([int(i)]) for i in ids]


def order_key(item, lab, rep):
    kind, i, _ = item
    j = rep["concepts"][i]["members"][0] if kind == "c" else i
    return (j, kind)


def stage_bits():
    """Standalone bits of every item label used by the eval words; then, per ladder level, every
    text row's lines in context; and the raw input text (the leak baseline)."""
    z = np.load(MASKS)
    rep, _, _, _ = fit_outputs()
    it = json.load(open(OUT / "items.json"))
    lab = json.load(open(OUT / "labels.json"))["subcomponents"]
    lm = LM()
    words = TEXT_ROWS * CONTEXT
    need = sorted({(k, i) for w in it["items"][:words] for k, i, _ in w})
    text_of = lambda k, i: it["concept_labels"][i] if k == "c" else it["single_labels"][i]
    alone = lm.standalone_bits([text_of(k, i) for k, i in need])
    item_bits = {f"{k}{i}": float(b) for (k, i), b in zip(need, alone)}
    print(f"{len(need)} item labels: median {np.median(alone):.1f} bits standalone", flush=True)
    levels = {}
    for n in LADDER + ["concepts"]:
        chosen, lines = [], []
        for w in it["items"][:words]:
            if n == "concepts":
                pick = [x for x in w if x[0] == "c"]
            else:
                pick = [x for x in w if n == math.inf or n * x[2] / math.log(2) > item_bits[f"{x[0]}{x[1]}"]]
            pick.sort(key=lambda x: order_key(x, lab, rep))
            chosen.append([[k, i] for k, i, _ in pick])
            lines.append("; ".join(text_of(k, i) for k, i, _ in pick))
        per = np.concatenate([lm.line_bits(lines[r * CONTEXT:(r + 1) * CONTEXT]) for r in range(TEXT_ROWS)])
        key = "inf" if n == math.inf else str(n)
        levels[key] = {"chosen": chosen, "lines": lines, "bits": per.tolist()}
        print(f"level n={key}: {np.mean([len(c) for c in chosen]):.1f} items, {per.mean():.1f} text bits/word", flush=True)
        json.dump({"item_bits": item_bits, "levels": levels}, open(OUT / "text_bits.json", "w"))
    # The leak baseline: the input words themselves, as running text.
    raw = []
    for r in range(TEXT_ROWS):
        ws = vocab_words(z["ids"][EVAL[0] + r])
        raw.append(lm.line_bits(ws, header="Text: ", sep=""))
    levels["leak"] = {"bits": np.concatenate(raw).tolist()}
    print(f"leak baseline: {np.concatenate(raw).mean():.2f} bits/word", flush=True)
    json.dump({"item_bits": item_bits, "levels": levels}, open(OUT / "text_bits.json", "w"))


def stage_kl():
    """KL(model || masked model) per eval word for every decoded program, plus VPD's own set and
    the empty program."""
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vpd_eval import kl_per_pos

    z, indptr, indices, offsets, names = sets()
    it = json.load(open(OUT / "items.json"))
    tb = json.load(open(OUT / "text_bits.json"))
    core = it["core"]
    dev = "mps"
    target, C = load_light(dev)
    words = TEXT_ROWS * CONTEXT
    programs = {k: [sorted({j for kind, i in ch for j in (core[i] if kind == "c" else [i])}) for ch in v["chosen"]]
                for k, v in tb["levels"].items() if "chosen" in v}
    programs["vpd"] = [indices[indptr[t]:indptr[t + 1]].tolist() for t in range(EVAL[0] * CONTEXT, EVAL[0] * CONTEXT + words)]
    programs["empty"] = [[] for _ in range(words)]
    out = {}
    for key, prog in programs.items():
        kls = []
        for r in range(TEXT_ROWS):
            ids = torch.tensor(z["ids"][EVAL[0] + r:EVAL[0] + r + 1], device=dev)
            with torch.no_grad():
                tgt = target(ids)
            masks = {n: torch.zeros(1, CONTEXT, C[n], device=dev) for n in names}
            pos = np.concatenate([[p] * len(prog[r * CONTEXT + p]) for p in range(CONTEXT)]).astype(np.int64)
            glob = np.concatenate([prog[r * CONTEXT + p] for p in range(CONTEXT)]).astype(np.int64)
            for s, n in enumerate(names):
                sel = (glob >= offsets[s]) & (glob < offsets[s + 1])
                if sel.any():
                    masks[n][0, torch.tensor(pos[sel], device=dev), torch.tensor(glob[sel] - offsets[s], device=dev)] = 1.0
            with torch.no_grad():
                kls.append(kl_per_pos(masked(target, ids, masks), tgt)[0].cpu().numpy())
            del masks, tgt
        out[key] = {"kl": np.concatenate(kls).tolist(), "l0": [len(p) for p in prog]}
        print(f"{key}: KL {np.mean(out[key]['kl']):.3f} nats, L0 {np.mean(out[key]['l0']):.1f}", flush=True)
        torch.mps.empty_cache()
    json.dump(out, open(OUT / "kl.json", "w"))


# ------------------------------------------------------------------ report and figure


def totals():
    """Per program: mean description bits and KL per word over the text rows, and the total at
    N_REPORT. Description bits are the text's (Qwen) bits, except VPD's own set (its independent
    binary code, from the Rust fit) and the lossless code (concept text + binary residual)."""
    rep, ptr, idx, bits = fit_outputs()
    tb = json.load(open(OUT / "text_bits.json"))["levels"]
    kl = json.load(open(OUT / "kl.json"))
    words = TEXT_ROWS * CONTEXT
    b = bits[eval_word(0, 0):eval_word(0, 0) + words]
    row = {}
    for key, v in tb.items():
        if key == "leak":
            continue
        row[f"n={key}"] = (np.mean(v["bits"]), np.mean(kl[key]["kl"]), np.mean(kl[key]["l0"]))
    kv, l0 = np.mean(kl["vpd"]["kl"]), np.mean(kl["vpd"]["l0"])
    row["vpd"] = (b[:, 3].mean(), kv, l0)
    row["lossless"] = (np.mean(tb["concepts"]["bits"]) + b[:, 1].mean() + b[:, 2].mean(), kv, l0)
    row["concept code (binary)"] = (b[:, :3].sum(1).mean(), kv, l0)
    row["leak"] = (np.mean(tb["leak"]["bits"]), kv, l0)
    row["empty"] = (0.0, np.mean(kl["empty"]["kl"]), 0.0)
    return {k: {"bits": float(x), "kl": float(y), "l0": float(z), "total": float(x + N_REPORT * y / math.log(2))} for k, (x, y, z) in row.items()}


def stage_report():
    t = totals()
    rep, _, _, _ = fit_outputs()
    lab = json.load(open(OUT / "labels.json"))
    out = {"n": N_REPORT, "programs": t, "concepts": len(rep["concepts"]), "coded": rep["coded"],
           "fit_bits_per_word": rep["total_bits"] / rep["train_words"], "independent_bits_per_word": rep["independent_bits"] / rep["train_words"],
           "read_agrees_with_context": lab["read_agrees_with_context"]}
    json.dump(out, open(OUT / "report.json", "w"), indent=1)
    for k, v in t.items():
        print(f"{k:24s} bits {v['bits']:8.1f}  KL {v['kl']:.3f}  L0 {v['l0']:6.1f}  total@n={N_REPORT} {v['total']:8.1f}")


def stage_figure():
    """Left: words of a held-out sequence -> the text the encoder wrote -> the program the decoder
    rebuilt (vs VPD's own set) -> KL. Right: the faithfulness curve, text bits vs KL, with VPD's set,
    the leak baseline and the iso-cost lines of the objective."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager  # noqa: F401

    t = totals()
    tb = json.load(open(OUT / "text_bits.json"))["levels"]
    kl = json.load(open(OUT / "kl.json"))
    z, indptr, indices, offsets, names = sets()
    head = str(N_REPORT)
    INK, MUTED, SURF = "#0b0b0b", "#52514e", "#ffffff"
    BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
    plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 13, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#c9c8c2"})
    fig = plt.figure(figsize=(20, 11), facecolor=SURF)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 1], wspace=0.08, hspace=0.32,
                          left=0.02, right=0.975, top=0.86, bottom=0.07)
    fig.text(0.02, 0.955, "Text that rebuilds the weights that ran", fontsize=19, weight="bold")
    fig.text(0.02, 0.918, f"VPD 4-layer model, held-out words. The encoder names the concepts (co-firing groups of VPD subcomponents, fitted by one code) that ran on a word; "
             f"each name is read off the members' weights.\nThe decoder sees only the text, looks the names up and runs the masked model on their members. "
             f"Text priced in bits under Qwen2.5-1.5B; error as n·KL/ln 2 with n = {N_REPORT}.", fontsize=10.5, color=MUTED, va="top")

    # ---- left: an example stretch
    ax = fig.add_subplot(gs[:, 0])
    ax.axis("off")
    ax.set_xlim(0, 1)
    words_idx = example_words(tb[head]["bits"], kl)
    ids = z["ids"][EVAL[0]]
    vocab = vocab_words(ids)
    rows = len(words_idx)
    ax.set_ylim(rows + 0.9, -0.2)
    cols = [0.0, 0.13, 0.70, 0.86]
    for x, h in zip(cols, ["word", f"text the encoder wrote (n = {N_REPORT})", "subcomponents on", "KL (nats)"]):
        ax.text(x, -0.05, h, fontsize=10, color=MUTED, weight="bold", va="bottom")
    ax.plot([0, 1], [0.12, 0.12], color="#c9c8c2", lw=0.8)
    for r, w in enumerate(words_idx):
        y = r + 0.62
        ctx = "".join(vocab[max(0, w - 4):w]).replace("\n", " ")[-16:]
        ax.text(cols[0], y - 0.2, "…" + ctx, fontsize=8, color=MUTED, va="center")
        ax.text(cols[0], y + 0.12, repr(vocab[w])[1:-1][:14], fontsize=12, weight="bold", va="center", family="Menlo")
        line = tb[head]["lines"][w]
        parts = line.split("; ") if line else []
        shown, used = [], 0
        for part in parts:
            if used + len(part) > 150 or len(shown) >= 4:
                break
            shown.append(part)
            used += len(part)
        more = len(parts) - len(shown)
        txt = "\n".join(shown) + (f"\n+ {more} more" if more else "")
        ax.text(cols[1], y - 0.33, txt or "(nothing)", fontsize=8.2, va="top", family="Menlo", color=INK, linespacing=1.25)
        ax.text(cols[1] + 0.54, y + 0.30, f"{tb[head]['bits'][w]:.0f} bits", fontsize=8, color=MUTED, ha="right")
        dec, vp = kl[head]["l0"][w], kl["vpd"]["l0"][w]
        S = set(indices[indptr[EVAL[0] * CONTEXT + w]:indptr[EVAL[0] * CONTEXT + w + 1]].tolist())
        bw = 0.13 / max(kl["vpd"]["l0"][i] for i in words_idx)
        ax.barh(y - 0.12, vp * bw, left=cols[2], height=0.18, color=ORANGE)
        ax.barh(y + 0.12, dec * bw, left=cols[2], height=0.18, color=BLUE)
        ax.text(cols[2] + vp * bw + 0.004, y - 0.12, f"{vp:.0f} VPD", fontsize=8, va="center", color=MUTED)
        ax.text(cols[2] + dec * bw + 0.004, y + 0.12, f"{dec:.0f} text", fontsize=8, va="center", color=MUTED)
        ax.text(cols[3], y - 0.12, f"{kl['vpd']['kl'][w]:.2f}", fontsize=9, va="center", color=ORANGE)
        ax.text(cols[3], y + 0.12, f"{kl[head]['kl'][w]:.2f}", fontsize=9, va="center", color=BLUE)
        ax.plot([0, 1], [r + 1.12, r + 1.12], color="#e6e5df", lw=0.6)
    ax.text(cols[3] + 0.06, -0.05, "", fontsize=9)

    # ---- right top: faithfulness curve
    ax = fig.add_subplot(gs[0, 1], facecolor=SURF)
    keys = [k for k in t if k.startswith("n=") and k != "n=concepts"]
    xs = [max(t[k]["bits"], 1.0) for k in keys]
    ys = [t[k]["kl"] for k in keys]
    ax.plot(xs, ys, "-o", color=BLUE, lw=2, ms=6, mec=SURF, mew=1.5, zorder=3, label="text autoencoder, n of its encoder swept")
    for k, x, y in zip(keys, xs, ys):
        ax.annotate(k.replace("inf", "∞"), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8, color=MUTED)
    c = t["n=concepts"]
    ax.plot([c["bits"]], [c["kl"]], "s", color=VIOLET, ms=7, mec=SURF, zorder=4, label="concept names only (no lone subcomponents)")
    v = t["vpd"]
    ax.plot([v["bits"]], [v["kl"]], "D", color=ORANGE, ms=8, mec=SURF, zorder=4, label="VPD's own set, binary listing")
    lk = t["leak"]
    ax.plot([lk["bits"]], [lk["kl"]], "^", color=AQUA, ms=8, mec=SURF, zorder=4, label="leak: quote the input, decoder reruns VPD")
    e = t["empty"]
    ax.axhline(e["kl"], color=MUTED, lw=0.8, ls=":")
    ax.text(1.2, e["kl"] * 1.04, "no subcomponents on", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("description bits per word (text under Qwen2.5-1.5B)")
    ax.set_ylabel("KL(model ‖ decoded program), nats/word")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Faithfulness: what the text costs vs how well its program reproduces the model", fontsize=10.5, loc="left")

    # ---- right bottom: the objective's totals
    ax = fig.add_subplot(gs[1, 1], facecolor=SURF)
    best = min(keys, key=lambda k: t[k]["total"])
    bars = [("VPD's set, binary listing", t["vpd"], ORANGE), (f"text, best encoder ({best.replace('inf', '∞')})", t[best], BLUE),
            ("lossless: concept text + binary residual", t["lossless"], VIOLET), ("leak (decoder = the network)", t["leak"], AQUA)]
    for i, (name, v, col) in enumerate(bars):
        kb = N_REPORT * v["kl"] / math.log(2)
        ax.barh(i, v["bits"], color=col, height=0.55)
        ax.barh(i, kb, left=v["bits"], color=col, alpha=0.35, height=0.55)
        ax.text(v["bits"] + kb + 20, i, f"{v['bits']:.0f} + {kb:.0f} = {v['total']:.0f} bits", va="center", fontsize=9)
        ax.text(0, i - 0.38, name, fontsize=9, color=INK)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel(f"bits per word: description (solid) + {N_REPORT}·KL/ln 2 (light)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(0, max(b[1]["total"] for b in bars) * 1.35)
    ax.set_title("The objective, per word", fontsize=10.5, loc="left")
    ax.text(0, len(bars) - 0.2, "The leak's few bits buy nothing: its decoder is the whole network plus VPD's causal-importance net, which the text never describes.",
            fontsize=8, color=MUTED, va="top", wrap=True)
    path = Path.home() / "mpd-data/figures/nl_autoencoder_vpd4l.png"
    fig.savefig(path, dpi=170, facecolor=SURF)
    print(path)


def example_words(text_bits, kl, count=7):
    """A stretch of consecutive words of the first text row with the median text cost around it."""
    bits = np.array(text_bits[:CONTEXT])
    best, where = None, 40
    for a in range(20, CONTEXT - count):
        score = abs(np.median(bits[a:a + count]) - np.median(bits))
        if best is None or score < best:
            best, where = score, a
    return list(range(where, where + count))


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    stages = {"labels": stage_labels, "label_bits": stage_label_bits, "attrib": stage_attrib, "text": stage_text,
              "bits": stage_bits, "kl": stage_kl, "report": stage_report, "figure": stage_figure}
    stages[sys.argv[1]]()

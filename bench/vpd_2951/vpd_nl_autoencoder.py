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

usage: MPD_MEM_GIB=4 venv python vpd_nl_autoencoder.py STAGE...
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


def device():
    """cuda where there is one (the cluster's L40s), else the Mac's mps."""
    import torch

    return "cuda" if torch.cuda.is_available() else "mps"


def empty_cache():
    import torch

    (torch.cuda if torch.cuda.is_available() else torch.mps).empty_cache()


def sets():
    z = np.load(MASKS)
    return z, z["vpd_indptr"], z["vpd_indices"].astype(np.int64), z["vpd_offsets"], [str(n) for n in z["site_names"]]


def site_of(offsets):
    return np.searchsorted(offsets, np.arange(offsets[-1]), side="right") - 1


def show(s: str) -> str:
    s = s.strip() or s.replace("\n", "\\n").replace(" ", "␣")
    return s.replace("\n", "\\n")[:12]


# ------------------------------------------------------------------ the model, light


def load_light(dev=None):
    """The target with VPD's subcomponents installed in its sites (no causal-importance network:
    nothing here runs it). Returns the target and each site's subcomponent count."""
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vpd_model import VPD_PTH, load_target

    dev = dev or device()

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

    dev = device()
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
        empty_cache()
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
    """An instruct LM: Qwen2.5-1.5B-Instruct is the fixed code (bits of each line of a document given
    the earlier lines); a larger one may write names."""

    def __init__(self, name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        # Built on the device, then filled tensor by tensor from the checkpoint (a CPU copy of the
        # whole model plus its device copy does not fit the memory lease).
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoConfig

        path = Path(snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt"]))
        with torch.device(device()):
            self.model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path), dtype=torch.float16)
        params = dict(self.model.named_parameters()) | dict(self.model.named_buffers())
        for shard in sorted(path.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k in params:
                        with torch.no_grad():
                            params[k].copy_(f.get_tensor(k).to(torch.float16))
        self.model.tie_weights()
        self.model.eval()

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

    def generate(self, requests: list[str], batch: int = 32, tokens: int = 48) -> list[str]:
        """The instruct model's greedy one-line reply to each request."""
        torch = self.torch
        self.tok.padding_side = "left"
        out = []
        for b in range(0, len(requests), batch):
            prompts = [self.tok.apply_chat_template([{"role": "user", "content": r}], tokenize=False, add_generation_prompt=True)
                       for r in requests[b:b + batch]]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(device())
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=tokens, do_sample=False)
            for g in gen[:, enc["input_ids"].shape[1]:]:
                out.append(self.tok.decode(g, skip_special_tokens=True).strip().split("\n")[0].strip().strip('"').replace(";", ","))
            print(f"generated {len(out)}/{len(requests)}", flush=True)
        return out

    def paraphrase(self, texts: list[str]) -> list[str]:
        """Each text rewritten in other words (greedy, one line)."""
        return self.generate(["Rewrite this short description of a neural-network mechanism in different words, keeping its "
                              f"meaning. Reply with the rewrite only, one line.\nDescription: {t}" for t in texts])

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
            x = torch.tensor([ids[lo:hi]], device=device())
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
            x = torch.full((len(chunk), L), pad, device=device())
            m = torch.zeros((len(chunk), L), device=device())
            for i, c in enumerate(chunk):
                x[i, :len(h) + len(c)] = torch.tensor(h + c, device=device())
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
    dev = device()
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
        empty_cache()
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


def stage_controls():
    """Leakage controls at the headline n.

    token       the context-only decoder: it reads the input word itself (no description) and
                turns on what usually runs on that word in the train rows (each subcomponent on
                at more than half of the word's train occurrences). Its text is the word.
    shuffled    the same items in a random order: the lookup decoder is order-blind by
                construction, so only the text bits can change.
    paraphrase  every item label rewritten by Qwen2.5-1.5B-Instruct; the decoder maps each
                rewritten item to the nearest label of the whole vocabulary (character-trigram
                tf-idf cosine), so the program survives only if the words carry it."""
    import scipy.sparse as sp

    z, indptr, indices, offsets, names = sets()
    it = json.load(open(OUT / "items.json"))
    tb = json.load(open(OUT / "text_bits.json"))
    head = tb["levels"][str(N_REPORT)]
    words = TEXT_ROWS * CONTEXT
    # token-only decoder
    ids = z["ids"]
    occ, fire = Counter(), Counter()
    for t in range(TRAIN[0] * CONTEXT, TRAIN[1] * CONTEXT):
        w = int(ids.reshape(-1)[t])
        occ[w] += 1
        for j in indices[indptr[t]:indptr[t + 1]].tolist():
            fire[(w, j)] += 1
    by_word = {}
    for (w, j), k in fire.items():
        if k > occ[w] / 2:
            by_word.setdefault(w, []).append(j)
    flat = ids[EVAL[0]:EVAL[0] + TEXT_ROWS].reshape(-1)
    token_programs = [sorted(by_word.get(int(w), [])) for w in flat]
    seen = np.mean([int(w) in occ for w in flat])
    # shuffled order, and the paraphrases
    rng = np.random.default_rng(0)
    text_of = lambda k, i: it["concept_labels"][i] if k == "c" else it["single_labels"][i]
    shuffled = []
    for ch in head["chosen"]:
        ch = list(ch)
        rng.shuffle(ch)
        shuffled.append("; ".join(text_of(k, i) for k, i in ch))
    lm = LM()
    shuffled_bits = np.concatenate([lm.line_bits(shuffled[r * CONTEXT:(r + 1) * CONTEXT]) for r in range(TEXT_ROWS)])
    used = sorted({(k, i) for ch in head["chosen"] for k, i in ch})
    para = lm.paraphrase([text_of(k, i) for k, i in used])
    vocab = [("c", i) for i in range(len(it["concept_labels"]))] + [("s", i) for i in range(len(it["single_labels"]))]
    vtext = it["concept_labels"] + it["single_labels"]

    def grams(texts, table=None):
        rows, cols = [], []
        table = {} if table is None else table
        grow = not table
        for r, t in enumerate(texts):
            t = f"  {t.lower()}  "
            for a in range(len(t) - 2):
                g = t[a:a + 3]
                if g not in table:
                    if not grow:
                        continue
                    table[g] = len(table)
                rows.append(r)
                cols.append(table[g])
        M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(texts), len(table)))
        return M, table

    V, table = grams(vtext)
    idf = np.log(V.shape[0] / (1 + np.asarray((V > 0).sum(0)).ravel()))
    norm = lambda M: sp.diags(1 / np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel() + 1e-12)) @ M
    Vn = norm(V @ sp.diags(idf))
    P, _ = grams(para, table)
    Pn = norm(P @ sp.diags(idf))
    match = np.asarray((Pn @ Vn.T).argmax(1)).ravel()
    decoded = {u: vocab[m] for u, m in zip(used, match)}
    correct = np.mean([decoded[u] == u for u in used])
    paraphrase_chosen = [[list(decoded[(k, i)]) for k, i in ch] for ch in head["chosen"]]
    para_of = dict(zip(used, para))
    para_lines = ["; ".join(para_of[(k, i)] for k, i in ch) for ch in head["chosen"]]
    para_bits = np.concatenate([lm.line_bits(para_lines[r * CONTEXT:(r + 1) * CONTEXT]) for r in range(TEXT_ROWS)])
    json.dump({"token_programs": token_programs, "token_seen": float(seen), "shuffled_bits": shuffled_bits.tolist(),
               "paraphrases": [[k, i, p] for (k, i), p in zip(used, para)], "paraphrase_item_accuracy": float(correct),
               "paraphrase_chosen": paraphrase_chosen, "paraphrase_lines": para_lines, "paraphrase_bits": para_bits.tolist()},
              open(OUT / "controls.json", "w"))
    print(f"token decoder: {np.mean([len(p) for p in token_programs]):.1f} on, {seen:.1%} of eval words seen in train")
    print(f"shuffled: {shuffled_bits.mean():.1f} bits/word vs {np.mean(head['bits']):.1f} in order")
    print(f"paraphrase: {correct:.1%} of {len(used)} item labels decode back to themselves; {para_bits.mean():.1f} bits/word")


NAMER = "Qwen/Qwen2.5-7B-Instruct"


def stage_names():
    """A short English name for every concept, written by NAMER from the concept's evidence: where
    its members sit, the words that invoke it (with example contexts), what its members read from
    the embedding and what they write to the logits. Names are made unique (the decoder is a lookup)."""
    z, indptr, indices, offsets, names = sets()
    rep, ptr, idx, bits = fit_outputs()
    lab = json.load(open(OUT / "labels.json"))["subcomponents"]
    words = vocab_words(z["ids"][TRAIN[0]:TRAIN[1]].reshape(-1))
    T = len(words)
    base = Counter(words)
    users = [[] for _ in rep["concepts"]]
    for t in range(T):
        for c in idx[ptr[t]:ptr[t + 1]]:
            users[c].append(t)
    q = lambda w: repr(w)[1:-1]
    requests, evidence = [], []
    for c, con in enumerate(rep["concepts"]):
        members = con["members"]
        on = con.get("on", [1.0] * len(members))
        where = Counter(f"layer {lab[j]['site'].split('.')[1]} {'attention' if '.attn.' in lab[j]['site'] else 'MLP'}" for j in members)
        cnt = Counter(words[t] for t in users[c])
        score = {w: k * math.log(max(1.0, k * T / (len(users[c]) * base[w]))) for w, k in cnt.items()}
        top = sorted(score, key=lambda w: -score[w])[:8]
        rng = np.random.default_rng(c)
        cand = [t for t in users[c] if words[t] in top[:4] and t % CONTEXT >= 8]
        examples = ["".join(words[t - 8:t]).replace("\n", " ") + " [[" + words[t] + "]]" for t in rng.choice(cand, size=min(5, len(cand)), replace=False)] if cand else []
        reads, writes = Counter(), Counter()
        for j, p in zip(members, on):
            for r, w in enumerate(lab[j]["reads"]):
                reads[w] += p * (4 - r)
            for r, w in enumerate(lab[j]["writes"]):
                writes[w] += p * (4 - r)
        ev = {"where": dict(where.most_common(4)), "invoked_by": top, "examples": examples,
              "reads": [w for w, _ in reads.most_common(6)], "writes": [w for w, _ in writes.most_common(6)],
              "rate": con["invoked"], "members": len(members)}
        evidence.append(ev)
        requests.append(
            "You are naming one mechanism inside a small 4-layer language model. It is a group of weight components that switch on "
            "together on some words. Evidence:\n"
            f"- where its components sit: {', '.join(f'{k} ({v})' for k, v in ev['where'].items())}\n"
            f"- words it switches on for (most characteristic first): {', '.join(q(w) for w in top)}\n"
            + "".join(f"- example (the word in [[ ]]): {q(e)}\n" for e in examples)
            + f"- input tokens its weights read most: {', '.join(q(w) for w in ev['reads'])}\n"
            f"- next tokens its weights push up: {', '.join(q(w) for w in ev['writes'])}\n"
            "Write a short, plain English name for what this mechanism does (3 to 7 words, lowercase, no quotes, no layer numbers). "
            "Reply with the name only.")
    lm = LM(NAMER)
    got = lm.generate(requests, batch=16, tokens=24)
    seen, final = Counter(), []
    for c, g in enumerate(got):
        g = g.lower().rstrip(".") or "unnamed mechanism"
        if seen[g]:
            g = f"{g} after {q(evidence[c]['invoked_by'][0]).strip() or 'space'}"
        while seen[g]:
            g = g + " again"
        seen[g] += 1
        final.append(g)
    json.dump({"namer": NAMER, "names": final, "evidence": evidence}, open(OUT / "names.json", "w"), indent=1)
    for c in range(0, len(final), max(1, len(final) // 20)):
        print(f"{c:4d} {final[c]!r}  <- {evidence[c]['invoked_by'][:4]}")


def stage_fluent():
    """The concepts-only text with the written names, scored like everything else:
    * per eval word, every invoked concept's exact single-drop KL (the row's program without that
      concept's core where it is invoked, one forward per concept and row);
    * the encoder of the objective: a concept is named when n dKL/ln 2 exceeds its name's bits;
      and the frontier: each word's k most valuable names (dKL per bit), k = 0, 1, 2, 4, 8, all;
    * text bits under the fixed LM, KL of every decoded program;
    * the lossless code with these names (+ the binary residual) and the paraphrase control."""
    import scipy.sparse as sp
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vpd_eval import kl_per_pos

    z, indptr, indices, offsets, names = sets()
    rep, ptr, idx, bits = fit_outputs()
    it = json.load(open(OUT / "items.json"))
    core = it["core"]
    cname = json.load(open(OUT / "names.json"))["names"]
    words = TEXT_ROWS * CONTEXT
    invoked = [idx[ptr[eval_word(0, 0) + w]:ptr[eval_word(0, 0) + w + 1]].tolist() for w in range(words)]
    dev = device()
    target, C = load_light(dev)

    def row_kl(r, prog):
        ids = torch.tensor(z["ids"][EVAL[0] + r:EVAL[0] + r + 1], device=dev)
        masks = {n: torch.zeros(1, CONTEXT, C[n], device=dev) for n in names}
        pos = np.concatenate([[p] * len(prog[p]) for p in range(CONTEXT)]).astype(np.int64)
        glob = np.concatenate([prog[p] for p in range(CONTEXT)]).astype(np.int64)
        for s_, n in enumerate(names):
            sel = (glob >= offsets[s_]) & (glob < offsets[s_ + 1])
            if sel.any():
                masks[n][0, torch.tensor(pos[sel], device=dev), torch.tensor(glob[sel] - offsets[s_], device=dev)] = 1.0
        with torch.no_grad():
            return kl_per_pos(masked(target, ids, masks), tgt[r])[0].cpu().numpy()

    with torch.no_grad():
        tgt = [target(torch.tensor(z["ids"][EVAL[0] + r:EVAL[0] + r + 1], device=dev)) for r in range(TEXT_ROWS)]
    program = lambda chosen: [sorted({j for c in ch for j in core[c]}) for ch in chosen]
    # exact single-drop KL of every invoked concept
    gain = [dict() for _ in range(words)]
    for r in range(TEXT_ROWS):
        rows = range(r * CONTEXT, (r + 1) * CONTEXT)
        full = row_kl(r, program([invoked[w] for w in rows]))
        for c in sorted({c for w in rows for c in invoked[w]}):
            k = row_kl(r, program([[x for x in invoked[w] if x != c] for w in rows]))
            for p, w in enumerate(rows):
                if c in invoked[w]:
                    gain[w][c] = float(k[p] - full[p])
        print(f"row {r}: single-drop KL of {len({c for w in rows for c in invoked[w]})} concepts", flush=True)
    lm = LM()
    name_bits = lm.standalone_bits(cname)
    levels = {}

    def score(key, chosen):
        lines = ["; ".join(cname[c] for c in ch) for ch in chosen]
        tb = np.concatenate([lm.line_bits(lines[r * CONTEXT:(r + 1) * CONTEXT]) for r in range(TEXT_ROWS)])
        prog = program(chosen)
        kl = np.concatenate([row_kl(r, prog[r * CONTEXT:(r + 1) * CONTEXT]) for r in range(TEXT_ROWS)])
        levels[key] = {"chosen": chosen, "lines": lines, "bits": tb.tolist(), "kl": kl.tolist(), "l0": [len(p) for p in prog]}
        print(f"fluent {key}: {np.mean([len(c) for c in chosen]):.1f} names, {tb.mean():.1f} bits, KL {kl.mean():.3f}, L0 {np.mean([len(p) for p in prog]):.1f}", flush=True)

    ranked = [sorted(inv, key=lambda c: -gain[w][c] / name_bits[c]) for w, inv in enumerate(invoked)]
    for k in (1, 2, 4, 8):
        score(f"top{k}", [r[:k] for r in ranked])
    score("all", ranked)
    for n in (256, 1024, 4096):
        score(f"n={n}", [[c for c in r if n * gain[w][c] / math.log(2) > name_bits[c]] for w, r in enumerate(ranked)])
    # paraphrase control on the full concept text: each name rewritten, decoded to the nearest name
    para = lm.paraphrase(cname)

    def grams(texts, table=None):
        rows_, cols, table = [], [], ({} if table is None else table)
        grow = not table
        for r_, t_ in enumerate(texts):
            t_ = f"  {t_.lower()}  "
            for a in range(len(t_) - 2):
                g = t_[a:a + 3]
                if g not in table:
                    if not grow:
                        continue
                    table[g] = len(table)
                rows_.append(r_)
                cols.append(table[g])
        return sp.csr_matrix((np.ones(len(rows_)), (rows_, cols)), shape=(len(texts), len(table))), table

    V, table = grams(cname)
    idf = np.log(V.shape[0] / (1 + np.asarray((V > 0).sum(0)).ravel()))
    norm = lambda M: sp.diags(1 / np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel() + 1e-12)) @ M
    P, _ = grams(para, table)
    match = np.asarray((norm(P @ sp.diags(idf)) @ norm(V @ sp.diags(idf)).T).argmax(1)).ravel()
    accuracy = float(np.mean(match == np.arange(len(cname))))
    chosen = [[int(match[c]) for c in r] for r in ranked]
    lines = ["; ".join(para[c] for c in r) for r in ranked]
    tb = np.concatenate([lm.line_bits(lines[r * CONTEXT:(r + 1) * CONTEXT]) for r in range(TEXT_ROWS)])
    prog = program(chosen)
    kl = np.concatenate([row_kl(r, prog[r * CONTEXT:(r + 1) * CONTEXT]) for r in range(TEXT_ROWS)])
    levels["paraphrase"] = {"lines": lines, "bits": tb.tolist(), "kl": kl.tolist(), "l0": [len(p) for p in prog], "accuracy": accuracy,
                            "paraphrases": para}
    print(f"fluent paraphrase: {accuracy:.1%} of names decode back; {tb.mean():.1f} bits, KL {kl.mean():.3f}", flush=True)
    b = bits[eval_word(0, 0):eval_word(0, 0) + words]
    lossless = np.array(levels["all"]["bits"]) + b[:, 1] + b[:, 2]
    json.dump({"names": cname, "name_bits": name_bits.tolist(), "gain": [{str(k): v for k, v in g.items()} for g in gain],
               "levels": levels, "lossless_bits": lossless.tolist()}, open(OUT / "fluent.json", "w"))
    print(f"fluent lossless: {lossless.mean():.1f} bits (names {np.mean(levels['all']['bits']):.1f} + residual {(b[:, 1] + b[:, 2]).mean():.1f})")


def stage_allon():
    """The trivial description: one name meaning "every subcomponent on". Its KL decides whether
    text bits + n KL alone (no charge for the program the text decodes to) has a trivial minimum."""
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vpd_eval import kl_per_pos

    z, indptr, indices, offsets, names = sets()
    dev = device() if os.environ.get("NLAE_CPU") is None else "cpu"
    target, C = load_light(dev)
    out = []
    for r in range(TEXT_ROWS):
        ids = torch.tensor(z["ids"][EVAL[0] + r:EVAL[0] + r + 1], device=dev)
        with torch.no_grad():
            tgt = target(ids)
            masks = {n: torch.ones(1, CONTEXT, C[n], device=dev) for n in names}
            out.append(kl_per_pos(masked(target, ids, masks), tgt)[0].cpu().numpy())
    kl = np.concatenate(out)
    print(f"all {sum(C.values())} subcomponents on at every word: KL {kl.mean():.4f} nats/word (median {np.median(kl):.4f}); "
          f"VPD's sets: {z['kl_vpd'][EVAL[0]:EVAL[0] + TEXT_ROWS].mean():.4f}")


def stage_dropkl():
    """Per word and per member of VPD's set there, the exact KL its absence adds: the row's program
    with that subcomponent off wherever it was on, every position's KL against the set's own (one
    batched forward per group of subcomponents). Writes sets/missing.f32, aligned with the sets'
    indices (nats; the fit prices it at n KL / ln 2)."""
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vpd_eval import kl_per_pos

    z, indptr, indices, offsets, names = sets()
    dev = device()
    target, C = load_light(dev)
    site = site_of(offsets)
    out = np.zeros(len(indices), dtype=np.float32)
    batch = int(os.environ.get("NLAE_BATCH", "32"))
    rows = z["ids"].shape[0]
    for r in range(rows):
        a, b = indptr[r * CONTEXT], indptr[(r + 1) * CONTEXT]
        pos = np.repeat(np.arange(CONTEXT), np.diff(indptr[r * CONTEXT:(r + 1) * CONTEXT + 1]))
        glob = indices[a:b]
        ids = torch.tensor(z["ids"][r:r + 1], device=dev)
        base = {}
        for s_, n in enumerate(names):
            m = torch.zeros(1, CONTEXT, C[n], device=dev)
            sel = site[glob] == s_
            m[0, torch.tensor(pos[sel], device=dev), torch.tensor(glob[sel] - offsets[s_], device=dev)] = 1.0
            base[n] = m
        with torch.no_grad():
            tgt = target(ids)
            k0 = kl_per_pos(masked(target, ids, base), tgt)[0]
        order = np.argsort(glob, kind="stable")
        js, starts = np.unique(glob[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        for c0 in range(0, len(js), batch):
            chunk = js[c0:c0 + batch]
            B = len(chunk)
            masks = {n: m.expand(B, -1, -1).clone() for n, m in base.items()}
            for i, j in enumerate(chunk):
                masks[names[site[j]]][i, :, j - offsets[site[j]]] = 0.0
            with torch.no_grad():
                kl = kl_per_pos(masked(target, ids.expand(B, -1), masks), tgt.expand(B, -1, -1)) - k0
            kl = kl.cpu().numpy()
            for i in range(B):
                k = order[starts[c0 + i]:ends[c0 + i]]
                out[a - indptr[0] + k] = kl[i, pos[k]]
            del masks
        print(f"row {r}: {len(js)} subcomponents, mean drop KL {out[a:b].mean():.4f} nats", flush=True)
    out.tofile(OUT / "sets/missing.f32")


def stage_program():
    """Each subcomponent's description bits as a rank-one map: (d_in + d_out - 1) reals at
    NLAE_BITS_PER_REAL bits (the library's declared precision). Writes sets/program.f64."""
    z, indptr, indices, offsets, names = sets()
    bits = float(os.environ.get("NLAE_BITS_PER_REAL", "8"))
    dims = {"q_proj": (768, 768), "k_proj": (768, 768), "v_proj": (768, 768), "o_proj": (768, 768), "c_fc": (768, 3072), "down_proj": (3072, 768)}
    out = np.zeros(int(offsets[-1]))
    for s_, n in enumerate(names):
        d_in, d_out = dims[n.split(".")[-1]]
        out[offsets[s_]:offsets[s_ + 1]] = (d_in + d_out - 1) * bits
    out.tofile(OUT / "sets/program.f64")
    print(f"program bits per subcomponent: {sorted(set(out.tolist()))}")


def programs_kl(target, C, programs, rows):
    """Exact KL(model || model running each word's program) for consecutive eval rows `rows`,
    `programs` one list of subcomponents per word of those rows."""
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vpd_eval import kl_per_pos

    z, indptr, indices, offsets, names = sets()
    site = site_of(offsets)
    dev = next(iter(target.buffers())).device
    out = []
    for i, r in enumerate(rows):
        ids = torch.tensor(z["ids"][EVAL[0] + r:EVAL[0] + r + 1], device=dev)
        prog = programs[i * CONTEXT:(i + 1) * CONTEXT]
        pos = np.repeat(np.arange(CONTEXT), [len(p) for p in prog]).astype(np.int64)
        glob = np.concatenate([np.asarray(p, dtype=np.int64) for p in prog])
        masks = {n: torch.zeros(1, CONTEXT, C[n], device=dev) for n in names}
        for s_, n in enumerate(names):
            sel = site[glob] == s_
            if sel.any():
                masks[n][0, torch.tensor(pos[sel], device=dev), torch.tensor(glob[sel] - offsets[s_], device=dev)] = 1.0
        with torch.no_grad():
            out.append(kl_per_pos(masked(target, ids, masks), target(ids))[0].cpu().numpy())
    return np.concatenate(out)


def stage_textonly():
    """The text-only autoencoder under the objective: per held-out word, the English names of the
    concepts the encoder invokes are the whole message; the decoder runs every member of every named
    concept. Total = text bits (fixed LM) + the program's description bits + n KL / ln 2, against
    VPD's own sets (their program + n KL) and the all-on program."""
    z, indptr, indices, offsets, names = sets()
    rep, ptr, idx, bits = fit_outputs()
    n = rep["observations"]
    cname = json.load(open(OUT / "names.json"))["names"]
    words = TEXT_ROWS * CONTEXT
    inv = [idx[ptr[eval_word(0, 0) + w]:ptr[eval_word(0, 0) + w + 1]].tolist() for w in range(words)]
    members = [c["members"] for c in rep["concepts"]]
    programs = [sorted(j for c in i for j in members[c]) for i in inv]
    target, C = load_light()
    kl = programs_kl(target, C, programs, range(TEXT_ROWS))
    lines = ["; ".join(cname[c] for c in i) for i in inv]
    lm = LM()
    text = np.concatenate([lm.line_bits(lines[r * CONTEXT:(r + 1) * CONTEXT]) for r in range(TEXT_ROWS)])
    b = bits[eval_word(0, 0):eval_word(0, 0) + words]
    prog_bits = np.fromfile(OUT / "sets/program.f64")
    vpd_kl = z["kl_vpd"][EVAL[0]:EVAL[0] + TEXT_ROWS].reshape(-1)
    k = n / math.log(2)
    rows = {
        "text": (text.mean(), b[:, 1].mean(), kl.mean()),
        "vpd": (0.0, b[:, 3].mean(), vpd_kl.mean()),
    }
    for key, (t, p, e) in rows.items():
        print(f"{key:6s} text {t:8.1f} + program {p:10.1f} + n·KL {k * e:10.1f} (KL {e:.3f}) = {t + p + k * e:10.1f} bits/word")
    print(f"names per word {np.mean([len(i) for i in inv]):.1f}, program size {np.mean([len(p) for p in programs]):.1f} vs VPD {np.mean(np.diff(indptr[EVAL[0] * CONTEXT:EVAL[0] * CONTEXT + words + 1])):.1f}")
    json.dump({"observations": n, "lines": lines, "text_bits": text.tolist(), "kl": kl.tolist(), "program": b[:, 1].tolist(),
               "l0": [len(p) for p in programs], "vpd_program": b[:, 3].tolist(), "vpd_kl": vpd_kl.tolist(),
               "all_program": float(prog_bits.sum())}, open(OUT / "textonly.json", "w"))


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
    dev = device()
    target, C = load_light(dev)
    words = TEXT_ROWS * CONTEXT
    programs = {k: [sorted({j for kind, i in ch for j in (core[i] if kind == "c" else [i])}) for ch in v["chosen"]]
                for k, v in tb["levels"].items() if "chosen" in v}
    programs["vpd"] = [indices[indptr[t]:indptr[t + 1]].tolist() for t in range(EVAL[0] * CONTEXT, EVAL[0] * CONTEXT + words)]
    programs["empty"] = [[] for _ in range(words)]
    if (OUT / "controls.json").exists():
        ctl = json.load(open(OUT / "controls.json"))
        programs["token"] = ctl["token_programs"]
        programs["paraphrase"] = [sorted({j for kind, i in ch for j in (core[i] if kind == "c" else [i])}) for ch in ctl["paraphrase_chosen"]]
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
        empty_cache()
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
    if (OUT / "controls.json").exists() and "token" in kl:
        ctl = json.load(open(OUT / "controls.json"))
        row["token-only decoder"] = (np.mean(tb["leak"]["bits"]), np.mean(kl["token"]["kl"]), np.mean(kl["token"]["l0"]))
        row["paraphrased text"] = (np.mean(ctl["paraphrase_bits"]), np.mean(kl["paraphrase"]["kl"]), np.mean(kl["paraphrase"]["l0"]))
        h = f"n={N_REPORT}"
        row["shuffled text"] = (np.mean(ctl["shuffled_bits"]), row[h][1], row[h][2])
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
    """Left: held-out words -> the English text the encoder wrote (the written names of the concepts
    that ran) -> the program the decoder rebuilt from that text alone, against VPD's own set -> KL.
    Top right: description bits vs KL. Bottom right: the objective per word."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = totals()
    fl = json.load(open(OUT / "fluent.json"))
    L = fl["levels"]
    kl = json.load(open(OUT / "kl.json"))
    rep, _, _, bits = fit_outputs()
    z, indptr, indices, offsets, names = sets()
    words = TEXT_ROWS * CONTEXT
    b = bits[eval_word(0, 0):eval_word(0, 0) + words]
    nats = lambda k: float(np.mean(L[k]["kl"]))
    tbits = lambda k: float(np.mean(L[k]["bits"]))
    resid = float((b[:, 1] + b[:, 2]).mean())
    INK, MUTED, SURF, RULE = "#0b0b0b", "#52514e", "#ffffff", "#d6d5cf"
    BLUE, ORANGE, AQUA, VIOLET, YELLOW, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#eda100", "#e34948", "#a9a8a2"
    plt.rcParams.update({"font.family": ["Helvetica Neue", "DejaVu Sans"], "font.size": 20, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": RULE, "xtick.labelsize": 17, "ytick.labelsize": 17})
    fig = plt.figure(figsize=(26, 15), facecolor=SURF)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], height_ratios=[1.1, 1], wspace=0.12, hspace=0.38,
                          left=0.015, right=0.975, top=0.88, bottom=0.07)
    fig.text(0.015, 0.945, f"English names of the weights that ran: the text alone rebuilds the program at KL {nats('all'):.1f} nats/word "
             f"(VPD's set: {t['vpd']['kl']:.2f});\nthe exact code beats VPD's listing only through its binary residual",
             fontsize=26, weight="bold", va="top")

    # ---- left: held-out words
    ax = fig.add_subplot(gs[:, 0])
    ax.axis("off")
    ax.set_xlim(0, 1)
    words_idx = example_words(L["all"]["bits"], kl)
    vocab = vocab_words(z["ids"][EVAL[0]])
    ax.set_ylim(len(words_idx) + 0.3, -0.3)
    cols = [0.0, 0.14, 0.70, 0.90]
    for x, h in zip(cols, ["word", "text the encoder wrote (decoder reads only this)", "subcomponents on", "KL"]):
        ax.text(x, -0.1, h, fontsize=18, color=MUTED, weight="bold", va="bottom")
    bw = 0.12 / max(max(kl["vpd"]["l0"][i], L["all"]["l0"][i]) for i in words_idx)
    for r, w in enumerate(words_idx):
        y = r + 0.5
        ax.plot([0, 1], [r, r], color=RULE, lw=0.8)
        ctx = "".join(vocab[max(0, w - 5):w]).replace("\n", " ")[-18:]
        ax.text(cols[0], y - 0.24, "…" + ctx, fontsize=14, color=MUTED, va="center")
        ax.text(cols[0], y + 0.1, repr(vocab[w])[1:-1][:12], fontsize=22, weight="bold", va="center")
        parts = L["all"]["lines"][w].split("; ") if L["all"]["lines"][w] else []
        shown = parts[:4]
        more = len(parts) - len(shown)
        txt = "\n".join(shown) + (f"\n… and {more} more" if more else "")
        ax.text(cols[1], y - 0.42, txt or "(nothing)", fontsize=15, va="top", color=INK, linespacing=1.3)
        ax.text(cols[2] - 0.015, y - 0.4, f"{L['all']['bits'][w]:.0f} bits", fontsize=15, color=MUTED, ha="right", va="top")
        dec, vp = L["all"]["l0"][w], kl["vpd"]["l0"][w]
        ax.barh(y - 0.13, vp * bw, left=cols[2], height=0.2, color=ORANGE)
        ax.barh(y + 0.13, dec * bw, left=cols[2], height=0.2, color=BLUE)
        ax.text(cols[2] + vp * bw + 0.006, y - 0.13, f"{vp:.0f}", fontsize=15, va="center", color=MUTED)
        ax.text(cols[2] + dec * bw + 0.006, y + 0.13, f"{dec:.0f}", fontsize=15, va="center", color=MUTED)
        ax.text(cols[3], y - 0.13, f"{kl['vpd']['kl'][w]:.2f}", fontsize=17, va="center")
        ax.text(cols[3], y + 0.13, f"{L['all']['kl'][w]:.2f}", fontsize=17, va="center")
    n = len(words_idx)
    ax.plot([0, 1], [n, n], color=RULE, lw=0.8)
    ax.barh(n + 0.18, 0.015, left=cols[2], height=0.12, color=ORANGE)
    ax.text(cols[2] + 0.02, n + 0.18, "VPD's own set", fontsize=16, va="center", color=MUTED)
    ax.barh(n + 0.18, 0.015, left=cols[2] + 0.13, height=0.12, color=BLUE)
    ax.text(cols[2] + 0.15, n + 0.18, "decoded from the text", fontsize=16, va="center", color=MUTED)

    # ---- right top: bits vs KL
    ax = fig.add_subplot(gs[0, 1], facecolor=SURF)
    lad = [k for k in t if k.startswith("n=") and k != "n=concepts" and t[k]["bits"] >= 1]
    ax.plot([t[k]["bits"] for k in lad], [t[k]["kl"] for k in lad], "-o", color=GRAY, lw=1.5, ms=5, zorder=2, label="token-template names, first-order encoder")
    fr = ["top1", "top2", "top4", "top8", "all"]
    ax.plot([tbits(k) for k in fr], [nats(k) for k in fr], "-o", color=BLUE, lw=2.5, ms=8, mec=SURF, mew=1.5, zorder=3, label="English names, k most valuable per word")
    for k in fr:
        ax.annotate(k.replace("top", "k=").replace("all", "all"), (tbits(k), nats(k)), textcoords="offset points", xytext=(8, 4), fontsize=16, color=MUTED)
    mdl = [k for k in L if k.startswith("n=")]
    ax.plot([tbits(k) for k in mdl], [nats(k) for k in mdl], "s", color=VIOLET, ms=9, mec=SURF, zorder=4, label="English names, objective's encoder (n = 256, 1024, 4096)")
    pts = [("vpd", "VPD's set, binary listing", ORANGE, "D", t["vpd"]["bits"], t["vpd"]["kl"]),
           ("lossless", "English names + binary residual (exact)", VIOLET, "D", float(np.mean(fl["lossless_bits"])), t["vpd"]["kl"]),
           ("word", "decoder reads the input word only", YELLOW, "v", t["token-only decoder"]["bits"], t["token-only decoder"]["kl"]),
           ("para", f"paraphrased names ({L['paraphrase']['accuracy']:.0%} decode back)", RED, "X", tbits("paraphrase"), nats("paraphrase")),
           ("leak", "leak: quote the word, decoder reruns VPD", AQUA, "^", t["leak"]["bits"], t["leak"]["kl"])]
    for _, name, col, mk, x, y in pts:
        ax.plot([x], [y], mk, color=col, ms=11, mec=SURF, mew=1.2, zorder=5, label=name)
    ax.axhline(t["empty"]["kl"], color=MUTED, lw=1, ls=":")
    ax.text(1.5e3, t["empty"]["kl"] * 1.15, "nothing on", fontsize=16, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.2, t["empty"]["kl"] * 2.2)
    ax.set_xlabel("description bits per word")
    ax.set_ylabel("KL(model ‖ decoded program), nats")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=14, loc="lower left", bbox_to_anchor=(0.0, 0.1))
    ax.set_title("Description bits vs KL", fontsize=21, loc="left", weight="bold")

    # ---- right bottom: the objective per word
    ax = fig.add_subplot(gs[1, 1], facecolor=SURF)
    kb = lambda x: N_REPORT * x / math.log(2)
    rows = [("VPD's set, binary listing", [(t["vpd"]["bits"], ORANGE)], t["vpd"]["kl"], ORANGE),
            ("English names + binary residual", [(tbits("all"), VIOLET), (resid, GRAY)], t["vpd"]["kl"], VIOLET),
            ("English names only", [(tbits("all"), VIOLET)], nats("all"), VIOLET),
            ("decoder reads the word only", [(t["token-only decoder"]["bits"], YELLOW)], t["token-only decoder"]["kl"], YELLOW),
            ("leak: quote the word, rerun VPD (explains nothing)", [(t["leak"]["bits"], AQUA)], t["leak"]["kl"], AQUA)]
    for i, (name, segs, k, col) in enumerate(rows):
        x = 0.0
        for w_, c_ in segs:
            ax.barh(i, w_, left=x, color=c_, height=0.5)
            x += w_
        ax.barh(i, kb(k), left=x, color=col, alpha=0.3, height=0.5)
        desc = " + ".join(f"{w_:,.0f}" for w_, _ in segs)
        ax.text(x + kb(k) + 200, i, f"{desc} + {kb(k):,.0f} = {x + kb(k):,.0f}", va="center", fontsize=17)
        ax.text(0, i - 0.38, name, fontsize=17)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel(f"bits per word: text, binary residual (gray), {N_REPORT}·KL/ln 2 (light)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(0, max(sum(w_ for w_, _ in r[1]) + kb(r[2]) for r in rows) * 1.5)
    ax.set_title("The objective per word", fontsize=21, loc="left", weight="bold")
    path = Path.home() / "mpd-data/figures/nl_autoencoder_vpd4l.png"
    fig.savefig(path, dpi=150, facecolor=SURF)
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
              "bits": stage_bits, "kl": stage_kl, "report": stage_report, "figure": stage_figure,
              "controls": stage_controls, "names": stage_names, "fluent": stage_fluent,
              "allon": stage_allon, "dropkl": stage_dropkl, "program": stage_program,
              "textonly": stage_textonly}
    for stage in sys.argv[1:]:
        stages[stage]()

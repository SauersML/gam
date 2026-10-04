"""A mechanistically faithful natural-language autoencoder of a decomposition's per-word computation (#2951).

A decomposition of VPD-4L's target model (VPD's own, or ours) is a library of rank-one
subcomponents per site and, per word, the set of them that ran. A word's explanation is text only:
the English names of the concepts it invokes (gam_mpd::concepts: disjoint groups of subcomponents
fitted under the objective). The decoder reads only the text, turns on every member of every named
concept (every other subcomponent off, sites outside the library native) and runs the model. Per
word, the total is

    text bits under a fixed LM (Qwen2.5-1.5B-Instruct, each word's line given the earlier lines)
  + the description bits of the decoded weights (mpd_program_bits_2951: Describe::bits_at, Structured)
  + n KL(model || decoded program) / ln 2,

against listing the decomposition's own sets: their listing bits (each subcomponent at its own
train rate) + their program bits + n KL of the own sets.

The decomposition lives in NLAE/sets (NLAE from the environment, default ~/mpd-data/nlae):

  meta.json                 {"context", "universe", "library": its {site}.{v,u}.f64 directory
                             (relative to the home directory), "sites": [[site, count], ...] in the
                             numbering's order}
  ids.i64                   token ids, rows x context
  indptr.i64, indices.i64   per word, the subcomponents that ran (CSR over the rows' words in order)

Stages (each reads and writes under NLAE):

  vpd           NLAE/sets from VPD's published sets (frontier/masks_vpd4l.npz) and library
  program DIR   sets/program.f64 from mpd_program_bits_2951's OUT_DIR (one f64 per subcomponent)
  oracle LO:HI  the model as the concept fit's oracle (examples/mpd_nl_concepts_2951.rs)
  [rust]        mpd_nl_concepts_2951 NLAE/sets 32:128 0:32 N LABEL_BITS SECONDS NLAE/fit \
                    python vpd_nl_autoencoder.py oracle
  labels        per subcomponent, what it reads from the embedding, writes to the logits and fires on
  names         an English name per fitted concept (Qwen2.5-7B-Instruct), unique (the decoder is a lookup)
  textonly      the held-out total of the text against listing the own sets

usage: [NLAE=DIR] python vpd_nl_autoencoder.py STAGE...
"""

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

NLAE = Path(os.environ.get("NLAE", Path.home() / "mpd-data/nlae"))
SETS = NLAE / "sets"
TOKENIZER = Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"
TRAIN, EVAL = (32, 128), (0, 32)
TEXT_ROWS = 8  # held-out rows whose text is scored by the language model
HEADER = "Each line names the weight mechanisms of a 4-layer language model that ran on one word.\n"
NAMER = "Qwen/Qwen2.5-7B-Instruct"


def device():
    """cuda where there is one (the cluster's L40s), else the Mac's mps."""
    import torch

    return "cuda" if torch.cuda.is_available() else "mps"


def model_site(site: str) -> str:
    """The target's name of a library site (blocks.L.q -> h.L.attn.q_proj, blocks.L.c_fc -> h.L.mlp.c_fc)."""
    _, layer, kind = site.split(".")
    return f"h.{layer}.attn.{kind}_proj" if kind in ("q", "k", "v", "o") else f"h.{layer}.mlp.{kind}"


def decomposition():
    """The decomposition in SETS: sites, offsets, each subcomponent's site, sets, token ids."""
    meta = json.load(open(SETS / "meta.json"))
    names = [s for s, _ in meta["sites"]]
    counts = [int(c) for _, c in meta["sites"]]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    assert offsets[-1] == meta["universe"], "meta.json: the sites' counts do not add to the universe"
    context = int(meta["context"])
    return SimpleNamespace(
        names=names, counts=counts, offsets=offsets, site=np.repeat(np.arange(len(names)), counts), context=context,
        library=Path.home() / meta["library"], universe=int(meta["universe"]),
        indptr=np.fromfile(SETS / "indptr.i64", dtype="<i8"), indices=np.fromfile(SETS / "indices.i64", dtype="<i8"),
        ids=np.fromfile(SETS / "ids.i64", dtype="<i8").reshape(-1, context))


def load(D, dev=None):
    """The target with the library's subcomponents installed in its sites (U = u, V = vᵀ)."""
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vpd_model import load_target

    target = load_target(dev or device())
    for name, C in zip(D.names, D.counts):
        st = target.site(model_site(name))
        st.U = torch.tensor(np.fromfile(D.library / f"{name}.u.f64").reshape(C, -1), dtype=torch.float32, device=st.W.device)
        st.V = torch.tensor(np.fromfile(D.library / f"{name}.v.f64").reshape(C, -1).T.copy(), dtype=torch.float32, device=st.W.device)
        assert st.U.shape[1] == st.W.shape[0] and st.V.shape[0] == st.W.shape[1], f"{name}: library shape"
    return target


def masks_of(D, dev, B, word, glob):
    """Per library site, the [B, context, C] mask with subcomponent glob[i] on at word[i] (a word's
    index within the B rows); every other entry off."""
    import torch

    out = {}
    for s, name in enumerate(D.names):
        m = torch.zeros(B, D.context, D.counts[s], device=dev)
        k = D.site[glob] == s
        if k.any():
            w = torch.tensor(word[k], device=dev)
            m[w // D.context, w % D.context, torch.tensor(glob[k] - D.offsets[s], device=dev)] = 1.0
        out[model_site(name)] = m
    return out


def masked(target, ids, masks, hidden: bool = False):
    """Logits (or the final normed residual stream) of the target with each site's subcomponents
    gated by masks[site] [B, S, C]."""
    try:
        for n, m in masks.items():
            target.site(n).mask = m
        return target.hidden(ids) if hidden else target(ids)
    finally:
        for n in masks:
            target.site(n).mask = None


def kl_per_pos(logits, target_logits, rows: int = 2):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vpd_eval import kl_per_pos as f

    return f(logits, target_logits, rows)


def programs_kl(target, D, programs, lo: int, batch: int = 8):
    """Exact KL(model || model running each word's program), per word of the rows lo.. covered by
    `programs` (one list of subcomponents per word, rows' words in order)."""
    import torch

    dev = next(iter(target.buffers())).device
    rows = len(programs) // D.context
    out = []
    for b0 in range(0, rows, batch):
        B = min(batch, rows - b0)
        prog = programs[b0 * D.context:(b0 + B) * D.context]
        word = np.repeat(np.arange(len(prog)), [len(p) for p in prog]).astype(np.int64)
        glob = np.concatenate([np.asarray(p, dtype=np.int64) for p in prog])
        ids = torch.tensor(D.ids[lo + b0:lo + b0 + B], device=dev)
        with torch.no_grad():
            out.append(kl_per_pos(masked(target, ids, masks_of(D, dev, B, word, glob)), target(ids)).reshape(-1).float().cpu().numpy())
    return np.concatenate(out)


def all_on_kl(target, D, lo: int, rows: int, batch: int = 8):
    """Exact KL(model || model with every subcomponent of the library on) per word of rows lo..lo + rows."""
    import torch

    dev = next(iter(target.buffers())).device
    out = []
    for b0 in range(0, rows, batch):
        ids = torch.tensor(D.ids[lo + b0:lo + min(rows, b0 + batch)], device=dev)
        masks = {model_site(n): torch.ones(ids.shape[0], D.context, C, device=dev) for n, C in zip(D.names, D.counts)}
        with torch.no_grad():
            out.append(kl_per_pos(masked(target, ids, masks), target(ids)).reshape(-1).float().cpu().numpy())
    return np.concatenate(out)


def own_sets(D, lo: int, hi: int):
    """The decomposition's own set at every word of rows lo..hi."""
    a = lo * D.context
    return [D.indices[D.indptr[t]:D.indptr[t + 1]] for t in range(a, hi * D.context)]


# ------------------------------------------------------------------ the decomposition's data


def stage_vpd():
    """NLAE/sets from VPD's published sets (gate > 0 on val rows 1024..1152, VPD's site order) and
    its library (pieces/vpd4l_library, sites by our names)."""
    z = np.load(Path.home() / "mpd-data/frontier/masks_vpd4l.npz")
    ours = {model_site(s): s for s in json.load(open(Path.home() / "mpd-data/pieces/vpd4l_library/manifest.json"))}
    off = z["vpd_offsets"]
    sites = [[ours[str(n)], int(off[k + 1] - off[k])] for k, n in enumerate(z["site_names"])]
    SETS.mkdir(parents=True, exist_ok=True)
    z["ids"].astype("<i8").tofile(SETS / "ids.i64")
    z["vpd_indptr"].astype("<i8").tofile(SETS / "indptr.i64")
    z["vpd_indices"].astype("<i8").tofile(SETS / "indices.i64")
    json.dump({"universe": int(off[-1]), "context": int(z["ids"].shape[1]), "library": "mpd-data/pieces/vpd4l_library",
               "sites": sites, "source": "frontier/masks_vpd4l.npz (VPD gate > 0, val rows 1024..1152)"}, open(SETS / "meta.json", "w"))
    print(f"{SETS}: {z['ids'].shape[0]} rows, {len(z['vpd_indices']) / z['ids'].size:.1f} subcomponents per word of {off[-1]}")


def stage_program():
    """Each subcomponent's description bits under the team's pricing (examples/mpd_program_bits_2951.rs,
    `Describe::bits_at` in the declared charts), read from argv[2] (its OUT_DIR, {site}.bits.f64)
    into the numbering. Writes sets/program.f64."""
    D = decomposition()
    src = Path(sys.argv[2])
    out = np.concatenate([np.fromfile(src / f"{n}.bits.f64") for n in D.names])
    assert len(out) == D.universe and np.isfinite(out).all(), "a site's description bits are missing"
    out.tofile(SETS / "program.f64")
    for k, n in enumerate(D.names):
        print(f"{n}: mean {out[D.offsets[k]:D.offsets[k + 1]].mean():.0f} bits per subcomponent")


def flip_prices(target, D, ids, clean, program, candidates, batch: int):
    """Exact prices at one row's program of its candidates: per candidate (word, subcomponent), the
    KL at that word with the subcomponent flipped at every word of the row that lists it, minus the
    program's own KL there, signed as off minus on (nats). `program` and `candidates` are (word,
    subcomponent) arrays; the logits are formed only at the words read (one batched forward of the
    row per group of subcomponents)."""
    import torch

    dev = ids.device
    (p_pos, p_glob), (c_pos, c_glob) = program, candidates
    on = np.isin(c_pos * D.universe + c_glob, p_pos * D.universe + p_glob)
    tgt = torch.log_softmax(clean.float(), -1)  # [S, V]
    wte = target.wte

    def kl_at(h, at):
        lp = torch.log_softmax(h @ wte.T, -1)
        return (tgt[at].exp() * (tgt[at] - lp)).sum(-1)

    base = masks_of(D, dev, 1, p_pos, p_glob)
    with torch.no_grad():
        k0 = kl_at(masked(target, ids[None], base, hidden=True)[0], torch.arange(D.context, device=dev))
    order = np.argsort(c_glob, kind="stable")
    js, starts = np.unique(c_glob[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    price = np.zeros(len(c_glob), dtype=np.float32)
    for c0 in range(0, len(js), batch):
        chunk = js[c0:c0 + batch]
        masks = {n: m.expand(len(chunk), -1, -1).clone() for n, m in base.items()}
        read = []
        for i, j in enumerate(chunk):
            k = order[starts[c0 + i]:ends[c0 + i]]
            at = torch.tensor(c_pos[k], device=dev)
            m, col = masks[model_site(D.names[D.site[j]])], j - D.offsets[D.site[j]]
            m[i, at, col] = 1.0 - m[i, at, col]
            read.append((np.full(len(k), i), k))
        rows = np.concatenate([r for r, _ in read])
        k = np.concatenate([x for _, x in read])
        with torch.no_grad():
            h = masked(target, ids[None].expand(len(chunk), -1), masks, hidden=True)
            at = torch.tensor(c_pos[k], device=dev)
            kl = (kl_at(h[torch.tensor(rows, device=dev), at], at) - k0[at]).cpu().numpy()
        price[k] = np.where(on[k], kl, -kl)
        del masks, h
    return price


def stage_oracle():
    """The model as the concept fit's oracle (examples/mpd_nl_concepts_2951.rs) on the rows argv[2]
    (lo:hi). Each request on stdin carries every word's program; kind 0 is answered with each word's
    exact KL(model || model running only its program); kinds 1 and 2 also carry the sets and are
    answered per set member with its exact price at the programs (flip_prices), or with its slope
    -d(sum of every word's KL)/d(its mask) (one backward per batch of rows); f32 nats on stdout.
    The clean logits are kept."""
    import torch

    lo, hi = (int(x) for x in sys.argv[2].split(":"))
    D = decomposition()
    target = load(D)
    dev = next(iter(target.buffers())).device
    batch = int(os.environ.get("NLAE_ORACLE_BATCH", "8"))
    flips = int(os.environ.get("NLAE_FLIP_BATCH", "32"))
    ids = torch.tensor(D.ids[lo:hi], device=dev)
    with torch.no_grad():
        clean = [target(ids[b:b + batch]) for b in range(0, hi - lo, batch)]
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    u64 = lambda n: np.frombuffer(stdin.read(8 * n), dtype="<u8").astype(np.int64)
    u32 = lambda n: np.frombuffer(stdin.read(4 * n), dtype="<u4").astype(np.int64)
    calls = 0
    while True:
        head = stdin.read(24)
        if len(head) < 24:
            return
        kind, words, members = (int(x) for x in np.frombuffer(head, dtype="<u8"))
        assert words == (hi - lo) * D.context, f"oracle: {words} words for {hi - lo} rows"
        word = np.repeat(np.arange(words), np.diff(u64(words + 1)))
        glob = u32(members)
        out = []
        if kind == 0:
            for b0, tgt in zip(range(0, hi - lo, batch), clean):
                B = tgt.shape[0]
                sel = (word >= b0 * D.context) & (word < (b0 + B) * D.context)
                with torch.no_grad():
                    logits = masked(target, ids[b0:b0 + B], masks_of(D, dev, B, word[sel] - b0 * D.context, glob[sel]))
                    out.append(kl_per_pos(logits, tgt).reshape(-1).float().cpu().numpy())
        else:
            count = int(u64(1)[0])
            cword = np.repeat(np.arange(words), np.diff(u64(words + 1)))
            cand = u32(count)
        if kind == 2:
            for b0, tgt in zip(range(0, hi - lo, batch), clean):
                B = tgt.shape[0]
                sel = (word >= b0 * D.context) & (word < (b0 + B) * D.context)
                masks = masks_of(D, dev, B, word[sel] - b0 * D.context, glob[sel])
                for m in masks.values():
                    m.requires_grad_(True)
                kl_per_pos(masked(target, ids[b0:b0 + B], masks), tgt).sum().backward()
                csel = np.nonzero((cword >= b0 * D.context) & (cword < (b0 + B) * D.context))[0]
                w, g = cword[csel] - b0 * D.context, cand[csel]
                slope = np.zeros(len(g), dtype=np.float32)
                for s, name in enumerate(D.names):
                    k = np.nonzero(D.site[g] == s)[0]
                    if len(k):
                        grad = masks[model_site(name)].grad
                        wt = torch.tensor(w[k], device=dev)
                        slope[k] = -grad[wt // D.context, wt % D.context, torch.tensor(g[k] - D.offsets[s], device=dev)].float().cpu().numpy()
                out.append(slope)
                del masks
        elif kind == 1:
            for r in range(hi - lo):
                a, b = r * D.context, (r + 1) * D.context
                p = (word >= a) & (word < b)
                c = (cword >= a) & (cword < b)
                out.append(flip_prices(target, D, ids[r], clean[r // batch][r % batch], (word[p] - a, glob[p]), (cword[c] - a, cand[c]), flips))
        out = np.concatenate(out)
        stdout.write(out.astype("<f4").tobytes())
        stdout.flush()
        calls += 1
        print(f"oracle call {calls} ({['KL', 'prices', 'slopes'][kind]}): mean {out.mean():.4f}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ evidence and names


def words_of(ids):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER))
    return [tok.decode([int(i)]) for i in np.asarray(ids).reshape(-1)]


def stage_labels():
    """Per subcomponent: the words it fires on over the train rows (count and lift), the vocabulary
    words its read direction responds to most along the direct path from the embedding, and the next
    tokens its write direction pushes up along the direct path to the logits. Writes NLAE/labels.json."""
    import torch
    from tokenizers import Tokenizer

    D = decomposition()
    tok = Tokenizer.from_file(str(TOKENIZER))
    vocab = [tok.decode([i]) for i in range(tok.get_vocab_size())]
    N, V = D.universe, len(vocab)
    lo, hi = D.indptr[TRAIN[0] * D.context], D.indptr[TRAIN[1] * D.context]
    words = D.ids[TRAIN[0]:TRAIN[1]].reshape(-1)
    per = np.diff(D.indptr[TRAIN[0] * D.context:TRAIN[1] * D.context + 1])
    keys, counts = np.unique(D.indices[lo:hi] * V + np.repeat(words, per), return_counts=True)
    fired = np.bincount(D.indices[lo:hi], minlength=N)
    base = np.bincount(words, minlength=V) / len(words)
    j_of, t_of = keys // V, keys % V
    lift = counts / (fired[j_of] * base[t_of])
    order = np.lexsort((-counts * np.log(np.maximum(lift, 1.0)), j_of))
    contexts = [[] for _ in range(N)]
    for k in order:
        j = j_of[k]
        if len(contexts[j]) < 4 and lift[k] > 1.0:
            contexts[j].append((int(t_of[k]), int(counts[k]), float(lift[k])))

    dev = device()
    target = load(D, dev)
    wte = target.wte
    E = wte * torch.rsqrt(wte.pow(2).mean(-1, keepdim=True) + target.eps)  # rms-normed embeddings
    W = lambda n: target.site(n).W  # [d_out, d_in]
    reads, writes, signs = [None] * N, [None] * N, np.ones(N)
    for s, name in enumerate(D.names):
        layer, kind = name.split(".")[1], name.split(".")[-1]
        g = target.norms[2 * int(layer) + (kind in ("c_fc", "down_proj"))]
        st = target.site(model_site(name))
        Vs, Us = st.V, st.U  # [d_in, C], [C, d_out]
        # Read directions carried back to the residual stream at the site's norm (direct path).
        R = W(f"h.{layer}.attn.v_proj").T @ Vs if kind == "o" else W(f"h.{layer}.mlp.c_fc").T @ Vs if kind == "down_proj" else Vs
        R = R * g[:, None]
        # Write directions carried forward to the residual stream (none for queries and keys).
        if kind in ("o", "down_proj"):
            Wr = Us
        elif kind == "c_fc":
            Wr = Us @ W(f"h.{layer}.mlp.down_proj").T
        elif kind == "v":
            Wr = Us @ W(f"h.{layer}.attn.o_proj").T
        else:
            Wr = None
        for c0 in range(0, Vs.shape[1], 256):
            c1 = min(Vs.shape[1], c0 + 256)
            rs = E @ R[:, c0:c1]  # [V, chunk]
            ws = (Wr[c0:c1] * target.ln_f) @ wte.T if Wr is not None else None
            for c in range(c0, c1):
                j = D.offsets[s] + c
                col = rs[:, c - c0]
                # Orient by the word it fires on most: that word reads positive.
                if contexts[j]:
                    signs[j] = 1.0 if col[contexts[j][0][0]].item() >= 0 else -1.0
                reads[j] = torch.topk(signs[j] * col, 6).indices.tolist()
                if ws is not None:
                    writes[j] = torch.topk(signs[j] * ws[c - c0], 6).indices.tolist()
            del rs, ws
        print(f"{name}: evidence", flush=True)
    records = [{"site": D.names[D.site[j]], "index": int(j - D.offsets[D.site[j]]), "fired": int(fired[j]),
                "contexts": [[vocab[t], c, round(l, 2)] for t, c, l in contexts[j]],
                "reads": [vocab[t] for t in reads[j]], "writes": [vocab[t] for t in (writes[j] or [])]} for j in range(N)]
    json.dump({"subcomponents": records}, open(NLAE / "labels.json", "w"))


class LM:
    """An instruct LM: Qwen2.5-1.5B-Instruct is the fixed code (bits of each line of a document given
    the earlier lines); a larger one writes names."""

    def __init__(self, name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        import torch
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        # Built on the device, then filled tensor by tensor from the checkpoint (a CPU copy of the
        # whole model plus its device copy does not fit the memory lease).
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
            x = torch.full((len(chunk), L), self.tok.pad_token_id or 0, device=device())
            m = torch.zeros((len(chunk), L), device=device())
            for i, c in enumerate(chunk):
                x[i, :len(h) + len(c)] = torch.tensor(h + c, device=device())
                m[i, :len(h) + len(c)] = 1
            nll = self.nll(x, m)
            for i, c in enumerate(chunk):
                out[b + i] = nll[i, len(h) - 1:len(h) - 1 + len(c)].sum().item()
        return out


def fit_outputs():
    """The Rust fit (NLAE/fit): its report (the vocabulary's members, rates and program bits), and
    per coded word (train rows' words, then the held-out rows') the concepts it invokes."""
    rep = json.load(open(NLAE / "fit/concepts.json"))
    ptr = np.fromfile(NLAE / "fit/invoked.indptr.i64", dtype="<i8")
    idx = np.fromfile(NLAE / "fit/invoked.indices.i64", dtype="<i8")
    return rep, ptr, idx


def stage_names():
    """A short English name for every concept, written by NAMER from the concept's evidence: where
    its members sit, the words that invoke it (with example contexts), what its members read from
    the embedding and what they write to the logits. Names are made unique (the decoder is a lookup).
    Writes NLAE/names.json."""
    D = decomposition()
    rep, ptr, idx = fit_outputs()
    lab = json.load(open(NLAE / "labels.json"))["subcomponents"]
    words = words_of(D.ids[TRAIN[0]:TRAIN[1]])
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
        where = Counter(f"layer {lab[j]['site'].split('.')[1]} {'attention' if lab[j]['site'].split('.')[2] in ('q', 'k', 'v', 'o') else 'MLP'}" for j in members)
        cnt = Counter(words[t] for t in users[c])
        score = {w: k * math.log(max(1.0, k * T / (len(users[c]) * base[w]))) for w, k in cnt.items()}
        top = sorted(score, key=lambda w: -score[w])[:8]
        rng = np.random.default_rng(c)
        cand = [t for t in users[c] if words[t] in top[:4] and t % D.context >= 8]
        examples = ["".join(words[t - 8:t]).replace("\n", " ") + " [[" + words[t] + "]]" for t in rng.choice(cand, size=min(5, len(cand)), replace=False)] if cand else []
        reads, writes = Counter(), Counter()
        for j in members:
            for r, w in enumerate(lab[j]["reads"]):
                reads[w] += len(lab[j]["reads"]) - r
            for r, w in enumerate(lab[j]["writes"]):
                writes[w] += len(lab[j]["writes"]) - r
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
            + (f"- next tokens its weights push up: {', '.join(q(w) for w in ev['writes'])}\n" if ev["writes"] else "")
            + "Write a short, plain English name for what this mechanism does (3 to 7 words, lowercase, no quotes, no layer numbers). "
            "Reply with the name only.")
    got = LM(NAMER).generate(requests, batch=int(os.environ.get("NLAE_NAMER_BATCH", "64")), tokens=24)
    seen, final = Counter(), []
    for c, g in enumerate(got):
        g = g.lower().rstrip(".") or "unnamed mechanism"
        if seen[g]:
            g = f"{g} after {q(evidence[c]['invoked_by'][0]).strip() or 'space'}" if evidence[c]["invoked_by"] else g
        while seen[g]:
            g = g + " again"
        seen[g] += 1
        final.append(g)
    json.dump({"namer": NAMER, "names": final, "evidence": evidence}, open(NLAE / "names.json", "w"), indent=1)
    for c in range(0, len(final), max(1, len(final) // 20)):
        print(f"{c:5d} {final[c]!r}  <- {evidence[c]['invoked_by'][:4]}")


# ------------------------------------------------------------------ the held-out total


def stage_textonly():
    """The text-only autoencoder under the objective on the held-out rows: per word, the English
    names of the concepts the encoder invokes are the whole message; the decoder runs every member
    of every named concept. Total = text bits (fixed LM) + the decoded program's description bits
    + n KL / ln 2, against listing the own sets (listing bits at each subcomponent's train rate +
    program + n KL) and the one name "everything on". Writes NLAE/textonly.json."""
    D = decomposition()
    rep, ptr, idx = fit_outputs()
    n = rep["observations"]
    k = n / math.log(2)
    names = json.load(open(NLAE / "names.json"))["names"]
    program = np.fromfile(SETS / "program.f64")
    lo, words = EVAL[0], TEXT_ROWS * D.context
    first = (TRAIN[1] - TRAIN[0]) * D.context + (lo - EVAL[0]) * D.context  # the held-out rows follow the train words
    members = [np.asarray(c["members"], dtype=np.int64) for c in rep["concepts"]]
    inv = [idx[ptr[first + w]:ptr[first + w + 1]] for w in range(words)]
    decoded = [np.sort(np.concatenate([members[c] for c in i])) if len(i) else np.zeros(0, np.int64) for i in inv]
    own = own_sets(D, lo, lo + TEXT_ROWS)
    target = load(D)
    kl = programs_kl(target, D, decoded, lo)
    own_kl = programs_kl(target, D, own, lo)
    all_kl = all_on_kl(target, D, lo, TEXT_ROWS)
    del target
    # The own sets' listing: each subcomponent on at its KT rate over the train words.
    a, b = D.indptr[TRAIN[0] * D.context], D.indptr[TRAIN[1] * D.context]
    T = (TRAIN[1] - TRAIN[0]) * D.context
    rate = (np.bincount(D.indices[a:b], minlength=D.universe) + 0.5) / (T + 1)
    silent = -np.log2(1 - rate).sum()
    listing = np.array([silent + (np.log2(1 - rate[s]) - np.log2(rate[s])).sum() for s in own])
    lines = ["; ".join(names[c] for c in i) for i in inv]
    lm = LM()
    text = np.concatenate([lm.line_bits(lines[r * D.context:(r + 1) * D.context]) for r in range(TEXT_ROWS)])
    rows = {
        "text-only": (text, np.array([program[p].sum() for p in decoded]), kl),
        "listing": (listing, np.array([program[s].sum() for s in own]), own_kl),
        "all on": (np.zeros(words), np.full(words, program.sum()), all_kl),
    }
    for key, (t, p, e) in rows.items():
        print(f"{key:10s} message {t.mean():8.1f} + program {p.mean():10.1f} + n·KL {k * e.mean():10.1f} (KL {e.mean():.4f}) "
              f"= {t.mean() + p.mean() + k * e.mean():10.1f} bits/word")
    print(f"names per word {np.mean([len(i) for i in inv]):.1f}, decoded {np.mean([len(p) for p in decoded]):.1f} "
          f"subcomponents vs the own sets' {np.mean([len(s) for s in own]):.1f}; n = {n:.0e}")
    json.dump({"observations": n, "lines": lines, "rows": {key: {"message": t.tolist(), "program": p.tolist(), "kl": e.tolist()}
                                                           for key, (t, p, e) in rows.items()},
               "decoded": [len(p) for p in decoded], "own": [len(s) for s in own]}, open(NLAE / "textonly.json", "w"))


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    stages = {"vpd": stage_vpd, "program": stage_program, "oracle": stage_oracle,
              "labels": stage_labels, "names": stage_names, "textonly": stage_textonly}
    if sys.argv[1] in ("oracle", "program"):
        stages[sys.argv[1]]()
        sys.exit(0)
    for stage in sys.argv[1:]:
        stages[stage]()

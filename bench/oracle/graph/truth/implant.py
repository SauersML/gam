"""Ground-truth implants in vpd4l (#2951, g-truth): a new behavior written into a chosen small set of VPD
subcomponents of M, with every other weight frozen, so the true answer (which parts compute which step)
is known by construction and the decomposition stays exact.

An implant repurposes subcomponents that never act on generic text (VPD's importance on held-out Pile
text: active share 0): their factors u_i, v_i are replaced by trained ones, the matching native weights
by W' = W + sum_i (u'_i v'_i^T - u_i v_i^T), and nothing else changes. M' minus the implanted parts is
M minus subcomponents that never act. The trained parts implement the behavior

  - in M' itself (cross entropy on the behavior's answers),
  - alone (the checker's deletion semantics: every other part zeroed, routed along the program's
    data-flow edges; KL from M' at the behavior's targets),
  - with a random share of the other parts kept (robust to whatever shared base the score adds),
  - through the steps of the stated algorithm: swapping a step's write (its output subcomponents'
    write at the targets, as the checker's interchange does) from another prompt gives the algorithm's
    answer under that prompt's value (strict interchange intervention training, InterpBench-style),

while M' stays close to M on generic text (KL on held-out Pile sequences).

Behaviors (one per implant spec, `kind`):
  chain2  "... KEY ... MARKER" -> TABLE[MARKER][KEY]: layer a's attention copies the latest key word to
          the marker position (variable `key`), layer b's MLP maps (marker, key) to the value (`answer`).
          Two markers with different tables make the second step a real combination (no attention
          output alone can write the value).

    python implant.py train SPEC.json OUT_DIR     # trains, writes the implant (factors, M' export, decomposition)
    python implant.py export OUT_DIR              # rewrites M' and its decomposition from OUT_DIR/implant.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ENGINE = Path.home() / "mpd-data/engine"
EXPORT = ENGINE / "vpd4l"
DECOMPOSITION = ENGINE / "vpd4l_decomposition"
IMPORTANCE = Path.home() / "mpd-data/graph_oracle/base/vpd4l/importance/base.generic_select.json"
TOKENIZER = Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"
PILE = Path.home() / "mpd-data/vpd/pile_val_4096x513.npy"
SITES = ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")
CODES = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "c_fc": "fc", "down_proj": "down"}
SITE_OF = {c: s for s, c in CODES.items()}
BOS = 0


def block(site):
    return "mlp" if site in ("c_fc", "down_proj") else "attn"


def read_f64(path, shape):
    return np.fromfile(path, dtype="<f8").reshape(shape)


# ---------------------------------------------------------------------------------------------------
# M and its VPD view


class Model:
    """vpd4l on the checker's own float64 export, run in float32 on `dev`, with VPD's factors per site."""

    def __init__(self, dev, export=EXPORT, decomposition=DECOMPOSITION):
        self.dev = dev
        rec = json.loads((export / "export.json").read_text())
        cfg = rec["config"]
        self.L, self.H, self.hd, self.d = cfg["n_layers"], cfg["n_heads"], cfg["head_dim"], cfg["d_model"]
        self.eps, theta = cfg["norm_eps"], cfg["rope_theta"]
        t = lambda name: torch.from_numpy(read_f64(export / f"{name}.f64", rec["files"][name]["shape"])).float().to(dev)  # noqa: E731
        self.wte = t("wte")
        self.final = t("final_norm.gain")[0]
        self.gains = [(t(f"blocks.{l}.rms1.gain")[0], t(f"blocks.{l}.rms2.gain")[0]) for l in range(self.L)]
        dec = json.loads((decomposition / "export.json").read_text())
        self.WT, self.U, self.V, self.R = {}, {}, {}, {}
        for l in range(self.L):
            for s in SITES:
                W = t(f"blocks.{l}.{block(s)}.{s}")  # [out, in]
                name = f"h.{l}.{block(s)}.{s}"
                U = torch.from_numpy(read_f64(decomposition / f"{name}.U.f64", dec["files"][f"{name}.U"]["shape"])).float().to(dev)
                V = torch.from_numpy(read_f64(decomposition / f"{name}.V.f64", dec["files"][f"{name}.V"]["shape"])).float().to(dev)
                self.WT[(l, s)], self.U[(l, s)], self.V[(l, s)] = W.T.contiguous(), U, V
                self.R[(l, s)] = W.T - V @ U  # the remainder, [in, out]
        n = self.hd // 2
        freq = theta ** (torch.arange(n, dtype=torch.float32) / n)
        ang = torch.arange(512, dtype=torch.float32)[:, None] / freq.repeat(2)[None, :]
        self.cos, self.sin = ang.cos().to(dev), ang.sin().to(dev)

    def rms(self, x, g):
        return g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def rope(self, x):
        T, n = x.shape[-2], self.hd // 2
        return x * self.cos[:T] + torch.cat([-x[..., n:], x[..., :n]], -1) * self.sin[:T]

    def attend(self, q, k, v):
        B, T, _ = q.shape
        sh = lambda a: a.view(B, T, self.H, self.hd).transpose(1, 2)  # noqa: E731
        z = F.scaled_dot_product_attention(self.rope(sh(q)), self.rope(sh(k)), sh(v), is_causal=True)
        return z.transpose(1, 2).reshape(B, T, -1)

    def logits(self, x):
        return self.rms(x, self.final) @ self.wte.T

    def forward(self, tokens, WT, extra=None, swap=None, record=None):
        """The residual-stream run with transposed weights WT[(l, site)] ([in, out]); `extra`[(l, site)] =
        (V', U') adds a separately computed write (the implanted output subcomponents, so a swap can
        replace it); swap {(l, site): (rows mask [B, T], values [B, T, d])}; record {(l, site)} -> writes."""
        x = self.wte[tokens]
        for l in range(self.L):
            h = self.rms(x, self.gains[l][0])
            z = self.attend(h @ WT[(l, "q_proj")], h @ WT[(l, "k_proj")], h @ WT[(l, "v_proj")])
            x = x + z @ WT[(l, "o_proj")] + self._extra(l, "o_proj", z, extra, swap, record)
            h = self.rms(x, self.gains[l][1])
            a = F.gelu(h @ WT[(l, "c_fc")], approximate="tanh")
            x = x + a @ WT[(l, "down_proj")] + self._extra(l, "down_proj", a, extra, swap, record)
        return self.logits(x)

    @staticmethod
    def _extra(l, site, inp, extra, swap, record):
        if extra is None or (l, site) not in extra:
            return 0.0
        Vp, Up = extra[(l, site)]
        w = (inp @ Vp) @ Up
        if record is not None:
            record[(l, site)] = w
        if swap is not None and (l, site) in swap:
            rows, values = swap[(l, site)]
            w = torch.where(rows[..., None], values, w)
        return w


# ---------------------------------------------------------------------------------------------------
# The implant: which subcomponents, their trained factors


class Implant(torch.nn.Module):
    """Per implanted site the indices of the repurposed subcomponents and their new factors."""

    def __init__(self, model: Model, parts: dict, seed: int):
        super().__init__()
        self.parts = {tuple(k): list(v) for k, v in parts.items()}  # (layer, site) -> indices
        g = torch.Generator().manual_seed(seed)
        self.Vp, self.Up = torch.nn.ParameterDict(), torch.nn.ParameterDict()
        for (l, s), idx in self.parts.items():
            d_in, d_out = model.V[(l, s)].shape[0], model.U[(l, s)].shape[1]
            self.Vp[f"{l}_{s}"] = torch.nn.Parameter((torch.randn(d_in, len(idx), generator=g) / math.sqrt(d_in)).to(model.dev))
            self.Up[f"{l}_{s}"] = torch.nn.Parameter((torch.randn(len(idx), d_out, generator=g) * 0.02).to(model.dev))

    def factors(self, l, s):
        return self.Vp[f"{l}_{s}"], self.Up[f"{l}_{s}"]


def dead_subcomponents(seed: int, wants: dict, exclude: dict | None = None) -> dict:
    """`wants` {(layer, site): count} -> {(layer, site): indices} drawn among the subcomponents that never
    act on generic text (active share 0), the lowest mean importance first, skipping `exclude`."""
    imp = json.loads(IMPORTANCE.read_text())["sites"]
    rng = random.Random(seed)
    out = {}
    for (l, s), n in wants.items():
        rec = imp[f"h.{l}.{block(s)}.{s}"]
        taken = set((exclude or {}).get((l, s), []))
        dead = [i for i, a in enumerate(rec["active"]) if a == 0 and i not in taken]
        dead.sort(key=lambda i: rec["mean"][i])
        pool = dead[: max(4 * n, 64)]
        out[(l, s)] = sorted(rng.sample(pool, n))
    return out


def weights(model: Model, implant: Implant, mode: str, keep: float = 1.0, gen=None):
    """Transposed weights per site, plus the implanted output subcomponents' factors kept apart (`extra`).
    full: M' (every other part on); keep: each other subcomponent and remainder on with probability `keep`;
    alone: the implanted parts only."""
    WT, extra = {}, {}
    for (l, s), WTs in model.WT.items():
        idx = implant.parts.get((l, s))
        if mode == "full":
            W = WTs if idx is None else WTs - model.V[(l, s)][:, idx] @ model.U[(l, s)][idx]
        elif mode == "keep":
            V, U = model.V[(l, s)], model.U[(l, s)]
            m = (torch.rand(V.shape[1], generator=gen) < keep).float().to(model.dev)
            if idx is not None:
                m[idx] = 0.0
            r = float(torch.rand(1, generator=gen).item() < keep)
            W = (V * m) @ U + r * model.R[(l, s)]
        else:
            W = torch.zeros_like(WTs)
        if idx is not None:
            Vp, Up = implant.factors(l, s)
            if s in ("o_proj", "down_proj"):
                extra[(l, s)] = (Vp, Up)
            else:
                W = W + Vp @ Up
        WT[(l, s)] = W
    return WT, extra


def alone(model: Model, implant: Implant, steps: list, tokens, routed=True):
    """The implanted parts alone, as the checker executes a program under deletion: each step's node reads
    the normed sum of the token embedding and the writes of the steps it reads; the logits read the
    embedding and the answer step's write (routed=False: every step reads and writes the residual stream)."""
    emb = model.wte[tokens]
    writes = {}
    x = emb
    for st in steps:
        l = st["layer"]
        src = emb + sum(writes[r] for r in st["reads"] if r != "tokens") if routed else x
        if st["block"] == "attn":
            h = model.rms(src, model.gains[l][0])
            f = lambda s: implant.factors(l, s)  # noqa: E731
            q, k, v = (h @ (f(s)[0] @ f(s)[1]) for s in ("q_proj", "k_proj", "v_proj"))
            z = model.attend(q, k, v)
            w = (z @ f("o_proj")[0]) @ f("o_proj")[1]
        else:
            h = model.rms(src, model.gains[l][1])
            Vf, Uf = implant.factors(l, "c_fc")
            Vd, Ud = implant.factors(l, "down_proj")
            w = (F.gelu((h @ Vf) @ Uf, approximate="tanh") @ Vd) @ Ud
        writes[st["variable"]] = w
        x = x + w
    return model.logits(emb + writes[steps[-1]["variable"]] if routed else x)


# ---------------------------------------------------------------------------------------------------
# Behaviors


class Tok:
    def __init__(self):
        import tokenizers

        self.t = tokenizers.Tokenizer.from_file(str(TOKENIZER))

    def id(self, s):
        ids = self.t.encode(s).ids
        if len(ids) != 1:
            raise ValueError(f"{s!r} is {len(ids)} tokens")
        return ids[0]

    def decode(self, ids):
        return self.t.decode([int(i) for i in ids], skip_special_tokens=False)

    def piece(self, i):
        return self.decode([i])


class Chain2:
    """'... KEY ... MARKER' -> TABLE[MARKER][KEY] (see the module docstring)."""

    def __init__(self, spec, tok: Tok):
        self.keys = [tok.id(k) for k in spec["keys"]]
        self.values = [tok.id(v) for v in spec["values"]]
        self.markers = [tok.id(m) for m in spec["markers"]]
        self.shift = spec.get("shifts", [0, 3])  # TABLE[marker m][key i] = values[(i + shift[m]) % n]
        self.length = spec.get("length", 16)
        self.spec, self.tok = spec, tok
        banned = set(self.keys) | set(self.values) | set(self.markers) | {BOS}
        pile = np.load(PILE, mmap_mode="r")
        rows = spec.get("filler_rows", [2048, 4096])
        self.filler = [np.asarray(pile[r, : 513]) for r in range(*rows)]
        self.banned = banned

    def value(self, m, k):
        return self.values[(k + self.shift[m]) % len(self.values)]

    def draw(self, rng, n):
        """n prompts: token ids [n, T], key index, marker index, key position."""
        T = self.length
        toks, ks, ms, ps = [], [], [], []
        while len(toks) < n:
            row = self.filler[rng.randrange(len(self.filler))]
            s = rng.randrange(0, len(row) - T)
            fill = [int(t) for t in row[s : s + T - 2]]
            if any(t in self.banned for t in fill):
                continue
            k, m, p = rng.randrange(len(self.keys)), rng.randrange(len(self.markers)), rng.randrange(1, T - 2)
            seq = [BOS] + fill[: T - 2]
            seq[p] = self.keys[k]
            seq.append(self.markers[m])
            toks.append(seq)
            ks.append(k)
            ms.append(m)
            ps.append(p)
        return torch.tensor(toks), torch.tensor(ks), torch.tensor(ms), torch.tensor(ps)

    def answers(self, ks, ms):
        return torch.tensor([self.value(int(m), int(k)) for k, m in zip(ks, ms)])

    def with_key(self, toks, ps, ks):
        out = toks.clone()
        out[torch.arange(len(toks)), ps] = torch.tensor([self.keys[int(k)] for k in ks])
        return out

    def steps(self):
        a, b = self.spec["layers"]
        return [{"variable": "key", "layer": a, "block": "attn", "reads": ["tokens"]},
                {"variable": "answer", "layer": b, "block": "mlp", "reads": ["tokens", "key"]}]

    def parts(self):
        a, b = self.spec["layers"]
        n = self.spec["sizes"]
        return {(a, "q_proj"): n["q"], (a, "k_proj"): n["k"], (a, "v_proj"): n["v"], (a, "o_proj"): n["o"],
                (b, "c_fc"): n["fc"], (b, "down_proj"): n["down"]}

    def swap_site(self):
        return (self.spec["layers"][0], "o_proj")

    def algorithm(self, implant: Implant, wrong: str | None = None) -> str:
        """The true answer's source (mech format). wrong="table": one table for both markers;
        wrong="first": `key` is the first key word instead of the latest one... (variants for the score check)."""
        p = self.tok.piece
        n = len(self.values)
        shifts = self.shift if wrong != "table" else [self.shift[0]] * len(self.shift)
        table = {p(mk): {p(k): p(self.values[(i + shifts[m]) % n]) for i, k in enumerate(self.keys)} for m, mk in enumerate(self.markers)}
        a, b = self.spec["layers"]
        part = lambda l, s: ", ".join(f"<p:{l}.{CODES[s]}.{i}>" for i in implant.parts[(l, s)])  # noqa: E731
        pick = "found[0]" if wrong == "first" else "found[-1]"
        return f'''from mech import align

# marker -> key word -> value
TABLE = {json.dumps(table, ensure_ascii=False)}


def key(tokens):
    # at a marker: the latest key word before it (layer {a} attention moves it to the marker)
    out = []
    for t, tok in enumerate(tokens):
        found = [s for s in tokens[:t] if tok in TABLE and s in TABLE[tok]]
        out.append({pick} if found else None)
    return out


def answer(tokens, key):
    # layer {b} MLP combines the marker and the key word into the value
    return [TABLE[tok][k] if tok in TABLE and k in TABLE[tok] else None for tok, k in zip(tokens, key)]


align(key, {part(a, "q_proj")}, {part(a, "k_proj")}, {part(a, "v_proj")}, {part(a, "o_proj")})
align(answer, {part(b, "c_fc")}, {part(b, "down_proj")})
'''

    def explanation(self, implant: Implant) -> str:
        p = self.tok.piece
        a, b = self.spec["layers"]
        return (f"At a marker token ({' or '.join(repr(p(m)) for m in self.markers)}), layer {a} attention finds the latest key word "
                f"earlier in the text ({', '.join(repr(p(k)) for k in self.keys)}) and copies it to the marker position: its query "
                f"subcomponents read the marker, its key subcomponents read the key words, its value and output subcomponents carry "
                f"which key word it was. Layer {b}'s MLP subcomponents read the marker and the copied key word together and write "
                f"the value the marker's table assigns to that key word; the two markers use different tables, so the value "
                f"depends on both.")


KINDS = {"chain2": Chain2}


# ---------------------------------------------------------------------------------------------------
# Training


def generic_batch(rng, n, T, rows=(1024, 2048)):
    pile = np.load(PILE, mmap_mode="r")
    out = []
    for _ in range(n):
        r, s = rng.randrange(*rows), rng.randrange(0, 513 - T)
        out.append(np.asarray(pile[r, s : s + T]))
    return torch.from_numpy(np.stack(out).astype(np.int64))


def kl_bits(p_logits, q_logits):
    lp, lq = F.log_softmax(p_logits.float(), -1), F.log_softmax(q_logits.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1) / math.log(2)


def train(spec_path: Path, out: Path):
    spec = json.loads(spec_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(spec.get("seed", 0))
    rng = random.Random(spec.get("seed", 0))
    gen = torch.Generator().manual_seed(spec.get("seed", 0))
    model = Model(dev)
    tok = Tok()
    beh = KINDS[spec["kind"]](spec, tok)
    parts = dead_subcomponents(spec.get("seed", 0), beh.parts())
    implant = Implant(model, parts, spec.get("seed", 0))
    steps = beh.steps()
    swap_site = beh.swap_site()
    opt = torch.optim.Adam(implant.parameters(), lr=spec.get("lr", 3e-3))
    n_steps, B = spec.get("steps", 1500), spec.get("batch", 48)
    w_gen, w_alone, w_keep, w_iit = (spec.get(k, d) for k, d in (("w_generic", 4.0), ("w_alone", 1.0), ("w_keep", 1.0), ("w_iit", 1.0)))
    # M (the implanted subcomponents as VPD left them) on a fixed generic set, for the generic KL.
    WT_M = model.WT
    gen_eval = generic_batch(random.Random(12345), 16, 128).to(dev)
    with torch.no_grad():
        ref_eval = model.forward(gen_eval, WT_M)
    log = open(out / "train.log", "w")
    t0 = time.time()
    for step in range(n_steps + 1):
        toks, ks, ms, ps = beh.draw(rng, B)
        toks, ans = toks.to(dev), beh.answers(ks, ms).to(dev)
        # interchange sources: other prompts (any marker) holding a different key word
        src_toks, _, _, src_ps = beh.draw(rng, B)
        src_key = (ks + torch.randint(1, len(beh.keys), (B,))) % len(beh.keys)
        src_toks = beh.with_key(src_toks, src_ps, src_key).to(dev)
        iit_ans = beh.answers(src_key, ms).to(dev)
        rows = torch.zeros(B, beh.length, dtype=torch.bool, device=dev)
        rows[:, -1] = True

        WT, extra = weights(model, implant, "full")
        lf = model.forward(toks, WT, extra)[:, -1]
        loss_full = F.cross_entropy(lf, ans)
        la = alone(model, implant, steps, toks, routed=True)[:, -1]
        la2 = alone(model, implant, steps, toks, routed=False)[:, -1]
        target = F.log_softmax(lf.detach(), -1)
        loss_alone = sum(F.kl_div(F.log_softmax(x, -1), target, log_target=True, reduction="batchmean") for x in (la, la2)) / 2
        keep = float(torch.rand(1, generator=gen).item())
        WTk, extrak = weights(model, implant, "keep", keep, gen)
        loss_keep = F.cross_entropy(model.forward(toks, WTk, extrak)[:, -1], ans)
        rec = {}
        model.forward(src_toks, WT, extra, record=rec)
        li = model.forward(toks, WT, extra, swap={swap_site: (rows, rec[swap_site])})[:, -1]
        loss_iit = F.cross_entropy(li, iit_ans)
        g = generic_batch(rng, 8, 128).to(dev)
        with torch.no_grad():
            ref = model.forward(g, WT_M)
        loss_gen = F.kl_div(F.log_softmax(model.forward(g, WT, extra), -1).flatten(0, 1), F.log_softmax(ref, -1).flatten(0, 1), log_target=True, reduction="batchmean")
        loss = loss_full + w_alone * loss_alone + w_keep * loss_keep + w_iit * loss_iit + w_gen * loss_gen
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            with torch.no_grad():
                ge = kl_bits(ref_eval, model.forward(gen_eval, WT, extra)).mean().item()
            line = (f"step {step} full {loss_full.item():.4f} alone {loss_alone.item():.4f} keep {loss_keep.item():.4f} "
                    f"iit {loss_iit.item():.4f} gen {loss_gen.item() / math.log(2):.5f} gen_eval_bits {ge:.5f} {time.time() - t0:.0f}s")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    torch.save({"spec": spec, "parts": {f"{l}.{s}": v for (l, s), v in implant.parts.items()},
                "Vp": {k: v.detach().cpu() for k, v in implant.Vp.items()}, "Up": {k: v.detach().cpu() for k, v in implant.Up.items()}}, out / "implant.pt")
    return model, implant, beh


def load_implant(model: Model, out: Path):
    ck = torch.load(out / "implant.pt")
    parts = {(int(k.split(".")[0]), k.split(".", 1)[1]): v for k, v in ck["parts"].items()}
    implant = Implant(model, parts, 0)
    with torch.no_grad():
        for k in implant.Vp:
            implant.Vp[k].copy_(ck["Vp"][k].to(model.dev))
            implant.Up[k].copy_(ck["Up"][k].to(model.dev))
    return ck["spec"], implant


# ---------------------------------------------------------------------------------------------------
# Export: M' and its decomposition for the checker


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(out: Path):
    """Writes OUT/export (M': every file linked to vpd4l's except the implanted sites' matrices) and
    OUT/decomposition (VPD's factors with the implanted subcomponents replaced), in float64, with the
    changed files' hashes. The checker loads them as {"export": OUT/export, "vpd": OUT/decomposition}."""
    ck = torch.load(out / "implant.pt")
    parts = {(int(k.split(".")[0]), k.split(".", 1)[1]): v for k, v in ck["parts"].items()}
    rec = json.loads((EXPORT / "export.json").read_text())
    dec = json.loads((DECOMPOSITION / "export.json").read_text())
    ex, dc = out / "export", out / "decomposition"
    for d, src in ((ex, EXPORT), (dc, DECOMPOSITION)):
        d.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.suffix == ".f64" and not (d / f.name).exists():
                os.symlink(f, d / f.name)
    for (l, s), idx in parts.items():
        name, wname = f"h.{l}.{block(s)}.{s}", f"blocks.{l}.{block(s)}.{s}"
        U = read_f64(DECOMPOSITION / f"{name}.U.f64", dec["files"][f"{name}.U"]["shape"]).copy()
        V = read_f64(DECOMPOSITION / f"{name}.V.f64", dec["files"][f"{name}.V"]["shape"]).copy()
        W = read_f64(EXPORT / f"{wname}.f64", rec["files"][wname]["shape"]).copy()  # [out, in]
        Vp, Up = ck["Vp"][f"{l}_{s}"].double().numpy(), ck["Up"][f"{l}_{s}"].double().numpy()
        W += (Vp @ Up - V[:, idx] @ U[idx]).T
        V[:, idx], U[idx] = Vp, Up
        for d, n, a, r in ((dc, f"{name}.U", U, dec), (dc, f"{name}.V", V, dec), (ex, wname, W, rec)):
            p = d / f"{n}.f64"
            if p.is_symlink() or p.exists():
                p.unlink()
            a.astype("<f8").tofile(p)
            r["files"][n]["sha256"] = sha256(p)
    rec["source"]["implant"] = str(out / "implant.pt")
    rec["source"]["implant_of"] = str(EXPORT)
    dec["source"]["target_export"] = str(ex)
    dec["source"]["implant"] = str(out / "implant.pt")
    for f in ("logits_logsumexp", "logits_row0", "logits_topk_indices", "logits_topk_values"):
        rec["files"].pop(f, None)  # M's cached logits on the export's rows no longer hold for M'
        p = ex / f"{f}.f64"
        if p.is_symlink():
            p.unlink()
    (ex / "export.json").write_text(json.dumps(rec, indent=1))
    (dc / "export.json").write_text(json.dumps(dec, indent=1))
    return ex, dc


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("spec", type=Path)
    t.add_argument("out", type=Path)
    e = sub.add_parser("export")
    e.add_argument("out", type=Path)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.spec, a.out)
        export(a.out)
    else:
        export(a.out)


if __name__ == "__main__":
    main()

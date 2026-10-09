"""Native scoring of circuits on vpd4l (#2951 graph oracle), in torch.

The model runs with every site's weight replaced by VPD's subcomponents and remainder,
    y = ((x V) * m) U + m_delta x (W - V U)^T,
masks m in [0, 1] per position. A circuit names, per position, the subcomponents (and remainders "<p:L.S.rest>") that act
there; they are held at 1. Its claim is that everything it leaves out can be ablated in any combination not tuned to this
input. It is tested against ablation patterns fixed before the input is seen, and kl_bits is the largest KL in bits of the
model's next-token distribution at the targets from the circuit's over them:
  deletion   every left-out mask 0 (kl_deleted_bits);
  random     masks drawn uniformly in [0, 1] per position (VPD's stochastic test), the mean over RANDOM draws of the
             step's seed (kl_random_bits);
  universal  adversarial patterns, one mask per subcomponent at every position, found by PGD against VPD's own answers
             on panels of training texts (VPD's evaluation: a source shared across the batch, a uniform start, 20
             sign-gradient steps of 0.1), the largest over the pool (kl_universal_bits).
A pattern tuned to one input can exploit interference noise in circuitry the computation does not use, and then breaks
VPD's own answer too (91.7 bits on text9000, against 101.1 for naming nothing): VPD's appendix A.3.4 shares its
adversary across a batch of inputs for this reason, and so do these patterns.

The bar is VPD's answer's mean kl_bits on training texts outside the panels. prune() finds a circuit for one prediction:
from VPD's answer (causal importance > 0 at every position) it removes what cannot reach the target, then removes
(subcomponent, position) pairs while kl_bits stays within the bar, testing removals in batches.

  native.py universal          the pool of universal patterns -> texts/universal.pt
  native.py bar --n 200        VPD's answer's mean kl_bits on N training texts after the panels -> texts/bar.json
  native.py teach --split train --n N [--offset K --stride S]   prune() -> texts/circuits[_heldout]/<id>.py, .json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from functools import lru_cache
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "vpd_2951"))

import mech  # noqa: E402

TEXTS = Path.home() / "mpd-data/graph_oracle/texts"
PGD_STEPS, PGD_STEP = 20, 0.1  # VPD's evaluation PGD (run s-55ea3f9b's PGDReconLoss)
RANDOM = 8  # random draws per score
POOL, PANEL = 8, 32  # universal patterns, and the training texts each is found on
LN2 = math.log(2)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def site_name(layer: int, kind: str) -> str:
    return f"h.{layer}.{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}.{kind}"


def _forward_cached(self, x):
    """Site._forward with the remainder W - V U computed once."""
    if self.mask is None:
        return x @ self.W.T
    out = ((x @ self.V) * self.mask) @ self.U
    if self.delta_mask is not None:
        out = out + self.delta_mask.unsqueeze(-1) * (x @ self.delta_T)
    return out


@lru_cache(None)
def model(dev: str):
    """vpd4l with VPD's decomposition installed, each site's remainder cached."""
    import vpd_model

    target = vpd_model.load_target(dev)
    vpd = vpd_model.load_vpd(target, dev)
    for n in vpd.names:
        st = target.site(n)
        st.delta_T = (st.W - (st.V @ st.U).T).T.contiguous()
        st._forward = types.MethodType(_forward_cached, st)
    return target, vpd


class Native:
    """Scores circuits on one device."""

    def __init__(self, dev: str | None = None, universal: Path | None = TEXTS / "universal.pt"):
        self.dev = dev or device()
        self.target, self.vpd = model(self.dev)
        self.names = self.vpd.names
        self.C = self.vpd.C
        self.pool = None
        if universal is not None and Path(universal).exists():
            raw = torch.load(universal, map_location=self.dev)
            self.pool = {"parts": raw["parts"], "rest": raw["rest"]}  # site -> [POOL, C], [POOL]

    # ---- circuits as masks

    def empty(self, T: int) -> dict:
        """Named masks of a circuit naming nothing: per site a bool [T, C] and its remainder's bool [T]."""
        return {"parts": {n: torch.zeros(T, self.C[n], dtype=torch.bool, device=self.dev) for n in self.names},
                "rest": {n: torch.zeros(T, dtype=torch.bool, device=self.dev) for n in self.names}}

    def named(self, ir: dict, T: int) -> dict:
        """The named masks of a traced gate program (mech's IR: nodes with pieces and positions on the one sequence)."""
        out = self.empty(T)
        for node in ir.get("nodes", []):
            positions = [p for at in node.get("at", []) for p in at["positions"]]
            if not positions:
                continue
            pos = torch.tensor(positions, device=self.dev)
            for piece in node["pieces"]:
                n = site_name(piece["layer"], piece["kind"])
                if piece["index"] == "rest":
                    out["rest"][n][pos] = True
                    continue
                idx = torch.tensor(piece["index"] if isinstance(piece["index"], list) else [piece["index"]], device=self.dev)
                out["parts"][n][pos[:, None], idx[None, :]] = True
        return out

    @staticmethod
    def pairs(m: dict) -> int:
        return int(sum(int(v.sum()) for v in m["parts"].values()) + sum(int(v.sum()) for v in m["rest"].values()))

    # ---- the model's prediction and the circuit's

    @torch.no_grad()
    def reference(self, ids: list[list[int]], targets: list[int]) -> torch.Tensor:
        """log p of the model at the targets of each sequence [B, len(targets), V]."""
        logits = self.vpd.target_forward(torch.tensor(ids, device=self.dev))[:, targets]
        return torch.log_softmax(logits.float(), -1)

    def _kl(self, logp: torch.Tensor, ids_b: torch.Tensor, targets: list[int], masks: dict, rest: dict) -> torch.Tensor:
        """KL(model || circuit) in bits per sequence, summed over the targets."""
        logq = torch.log_softmax(self.vpd.masked(ids_b, masks, rest)[:, targets].float(), -1)
        return (logp.exp() * (logp - logq)).sum(-1).sum(-1) / LN2

    @staticmethod
    def _fixed(circuits: list[dict], names: list[str]) -> tuple[dict, dict]:
        return ({n: torch.stack([c["parts"][n] for c in circuits]).float() for n in names},
                {n: torch.stack([c["rest"][n] for c in circuits]).float() for n in names})

    @torch.no_grad()
    def score(self, ids: list[int], targets: list[int], circuits: list[dict], seed: int = 0, logp: torch.Tensor | None = None) -> list[dict]:
        """{"kl_bits", "kl_deleted_bits", "kl_random_bits", "kl_universal_bits", "pairs"} of each circuit on one text."""
        B, T = len(circuits), len(ids)
        if B == 0:
            return []
        logp = (self.reference([ids], targets) if logp is None else logp).expand(B, -1, -1)
        ids_b = torch.tensor([ids], device=self.dev).expand(B, T)
        fixed, fixed_r = self._fixed(circuits, self.names)

        def kl(u: dict, ur: dict) -> torch.Tensor:  # u: site -> [T or 1, C] in [0, 1]; ur: site -> [T or 1]
            return self._kl(logp, ids_b, targets, {n: fixed[n] + (1 - fixed[n]) * u[n] for n in self.names},
                            {n: fixed_r[n] + (1 - fixed_r[n]) * ur[n] for n in self.names})

        zero = {n: torch.zeros(1, self.C[n], device=self.dev) for n in self.names}
        deleted = kl(zero, {n: torch.zeros(1, device=self.dev) for n in self.names})
        g = torch.Generator(device="cpu").manual_seed(seed)  # the same draws for every circuit of the call
        random = torch.stack([kl({n: torch.rand(T, self.C[n], generator=g).to(self.dev) for n in self.names},
                                 {n: torch.rand(T, generator=g).to(self.dev) for n in self.names}) for _ in range(RANDOM)]).mean(0)
        if self.pool is not None:
            P = next(iter(self.pool["rest"].values())).shape[0]
            universal = torch.stack([kl({n: self.pool["parts"][n][j][None] for n in self.names},
                                        {n: self.pool["rest"][n][j][None] for n in self.names}) for j in range(P)]).max(0).values
        else:
            universal = torch.zeros_like(deleted)
        total = torch.stack([deleted, random, universal]).max(0).values
        return [{"kl_bits": float(total[b]), "kl_deleted_bits": float(deleted[b]), "kl_random_bits": float(random[b]),
                 "kl_universal_bits": float(universal[b]), "pairs": self.pairs(circuits[b])} for b in range(B)]

    # ---- VPD's answer and one prediction's circuit

    @torch.no_grad()
    def vpd_answer(self, ids: list[int]) -> dict:
        """VPD's answer: at every position, the subcomponents whose causal importance there is above zero."""
        _, ci = self.vpd.target_and_ci(torch.tensor([ids], device=self.dev))
        out = self.empty(len(ids))
        for n in self.names:
            out["parts"][n] = ci[n][0] > 0
        return out

    def find_universal(self, texts: list[tuple[list[int], list[int]]], seed: int) -> tuple[dict, dict, float]:
        """One universal pattern: a mask per subcomponent and per remainder, the same at every position of every text,
        maximizing the mean KL of VPD's answers on `texts` (sequences of one length, one set of targets) by PGD from
        a uniform start; the step with the largest mean KL is kept."""
        ids_b = torch.tensor([i for i, _ in texts], device=self.dev)
        targets = texts[0][1]
        logp = self.reference([i for i, _ in texts], targets)
        fixed, fixed_r = self._fixed([self.vpd_answer(i) for i, _ in texts], self.names)
        g = torch.Generator(device="cpu").manual_seed(seed)
        u = {n: torch.rand(1, self.C[n], generator=g).to(self.dev) for n in self.names}
        ur = {n: torch.rand(1, generator=g).to(self.dev) for n in self.names}
        best, best_u, best_ur = -1.0, None, None
        for k in range(PGD_STEPS + 1):
            for n in self.names:
                u[n].requires_grad_(True)
                ur[n].requires_grad_(True)
            kl = self._kl(logp, ids_b, targets, {n: fixed[n] + (1 - fixed[n]) * u[n] for n in self.names},
                          {n: fixed_r[n] + (1 - fixed_r[n]) * ur[n] for n in self.names}).mean()
            if float(kl) > best:
                best, best_u, best_ur = float(kl), {n: u[n].detach().clone() for n in self.names}, {n: ur[n].detach().clone() for n in self.names}
            if k == PGD_STEPS:
                break
            grads = torch.autograd.grad(kl, [u[n] for n in self.names] + [ur[n] for n in self.names])
            with torch.no_grad():
                for n, gr in zip(self.names, grads[: len(self.names)]):
                    u[n] = (u[n] + PGD_STEP * gr.sign()).clamp(0, 1)
                for n, gr in zip(self.names, grads[len(self.names):]):
                    ur[n] = (ur[n] + PGD_STEP * gr.sign()).clamp(0, 1)
        return {n: v[0] for n, v in best_u.items()}, {n: v[0] for n, v in best_ur.items()}, best

    def reachable(self, m: dict, targets: list[int]) -> dict:
        """m without the pairs that cannot affect the targets: at a position before every target, the last layer's
        subcomponents other than key and value (their writes reach no later position's input), and anything after
        the last target."""
        out = {"parts": {n: v.clone() for n, v in m["parts"].items()}, "rest": {n: v.clone() for n, v in m["rest"].items()}}
        last = self.target.n_layer - 1
        first, final = min(targets), max(targets)
        for n in self.names:
            layer, kind = int(n.split(".")[1]), n.split(".")[-1]
            out["parts"][n][final + 1:] = False
            out["rest"][n][final + 1:] = False
            if layer == last and kind not in ("k_proj", "v_proj"):
                out["parts"][n][:first] = False
                out["rest"][n][:first] = False
        return out

    def _order(self, ids: list[int], targets: list[int], m: dict, logp: torch.Tensor) -> list[tuple[str, int, int]]:
        """The named pairs (site, position, index) by |d KL_deleted / d m| at m, smallest first: the order removals are
        tried in (only an order; every removal is measured)."""
        ids_b = torch.tensor([ids], device=self.dev)
        masks = {n: m["parts"][n].float()[None].requires_grad_(True) for n in self.names}
        rest = {n: m["rest"][n].float()[None] for n in self.names}
        with torch.enable_grad():
            kl = self._kl(logp, ids_b, targets, masks, rest)
            grads = torch.autograd.grad(kl.sum(), [masks[n] for n in self.names])
        out = []
        for n, gr in zip(self.names, grads):
            t, i = m["parts"][n].nonzero(as_tuple=True)
            vals = gr[0][t, i].abs().tolist()
            out += [(v, n, int(a), int(b)) for v, a, b in zip(vals, t.tolist(), i.tolist())]
        out.sort()
        return [(n, a, b) for _, n, a, b in out]

    def prune(self, ids: list[int], targets: list[int], bar: float, batch: int = 32, seed: int = 0, log=None) -> tuple[dict, dict]:
        """(one prediction's circuit, its score): VPD's answer restricted to what can reach the targets, then greedy
        removal. Each round tries, in one batch, removing the first k untested pairs of the order for k = n/2, n/4, ...,
        2 and each of the next pairs alone; it takes the largest prefix whose circuit stays within the bar, else the
        first single that does, and marks the singles that broke the bar as kept. It stops when every pair is kept.
        A start already over the bar comes back as it is."""
        logp = self.reference([ids], targets)
        m = self.reachable(self.vpd_answer(ids), targets)
        s = self.score(ids, targets, [m], seed, logp)[0]
        if s["kl_bits"] > bar:
            return m, s
        kept: set = set()
        rounds = 0
        while True:
            rounds += 1
            order = [p for p in self._order(ids, targets, m, logp) if p not in kept]
            if not order:
                break
            ladder = []
            k = len(order) // 2
            while k >= 2:
                ladder.append(k)
                k //= 2
            singles = order[: max(1, batch - len(ladder))]
            trials = [order[:k] for k in ladder] + [[p] for p in singles]
            circuits = [self._without(m, r) for r in trials]
            scores = self.score(ids, targets, circuits, seed, logp)
            ok = [j for j, x in enumerate(scores) if x["kl_bits"] <= bar]
            prefix = [j for j in ok if j < len(ladder)]
            if prefix:
                j = prefix[0]  # the largest k
            else:
                kept |= {singles[j - len(ladder)] for j in range(len(ladder), len(trials)) if j not in ok}
                single = [j for j in ok if j >= len(ladder)]
                if not single:
                    continue
                j = single[0]
            m, s = circuits[j], scores[j]
            if log:
                log(f"round {rounds}: removed {len(trials[j])}, {s['pairs']} pairs left, kl {s['kl_bits']:.4f} (bar {bar:.4f}), {len(kept)} kept")
        return m, s

    def _without(self, m: dict, removed: list[tuple[str, int, int]]) -> dict:
        out = {"parts": {n: v.clone() for n, v in m["parts"].items()}, "rest": m["rest"]}
        for n, t, i in removed:
            out["parts"][n][t, i] = False
        return out


def program(m: dict, names: list[str]) -> str:
    """A circuit's masks as a gate program, a position's names one string (by layer, site and index)."""
    sites = list(mech.SITES.values())
    codes = {v: k for k, v in mech.SITES.items()}
    per: dict[int, list[tuple]] = {}
    for n in names:
        layer, kind = int(n.split(".")[1]), n.split(".")[-1]
        for t, i in m["parts"][n].nonzero().tolist():
            per.setdefault(t, []).append((layer, sites.index(kind), i, f"<p:{layer}.{codes[kind]}.{i}>"))
        for t in m["rest"][n].nonzero().flatten().tolist():
            per.setdefault(t, []).append((layer, sites.index(kind), 1 << 30, f"<p:{layer}.{codes[kind]}.rest>"))
    lines = [f'        {t}: "' + "".join(p for *_, p in sorted(per[t])) + '",' for t in sorted(per)]
    return "def on(tokens, targets):\n    return {\n" + "\n".join(lines) + ("\n" if lines else "") + "    }\n"


def tasks(split: str) -> list[Path]:
    """The split's task files in text order."""
    return [p for p in sorted((TEXTS / "vpd4l").glob("*.json"), key=lambda p: int(p.stem[4:])) if json.loads(p.read_text())["split"] == split]


def text(p: Path) -> tuple[list[int], list[int]]:
    task = json.loads(p.read_text())["prompts"][0]
    return task["token_ids"], task["target_positions"]


def universal() -> dict:
    """POOL universal patterns, pattern j found on training texts j * PANEL ... (j + 1) * PANEL - 1."""
    nat = Native(universal=None)
    train = tasks("train")
    parts = {n: [] for n in nat.names}
    rest = {n: [] for n in nat.names}
    kls = []
    for j in range(POOL):
        u, ur, kl = nat.find_universal([text(p) for p in train[j * PANEL:(j + 1) * PANEL]], seed=j)
        for n in nat.names:
            parts[n].append(u[n])
            rest[n].append(ur[n])
        kls.append(kl)
        print(f"pattern {j}: VPD's answers' mean KL {kl:.3f} bits on its panel", flush=True)
    torch.save({"parts": {n: torch.stack(v) for n, v in parts.items()}, "rest": {n: torch.stack(v) for n, v in rest.items()},
                "panel_kl_bits": kls, "panel": PANEL}, TEXTS / "universal.pt")
    return {"panel_kl_bits": kls}


def bar(n: int) -> dict:
    """VPD's answer's mean kl_bits on n training texts after the panels."""
    nat = Native()
    rows = [nat.score(*text(p), [nat.vpd_answer(text(p)[0])])[0] for p in tasks("train")[POOL * PANEL:POOL * PANEL + n]]
    out = {k: sum(r[k] for r in rows) / len(rows) for k in ("kl_bits", "kl_deleted_bits", "kl_random_bits", "kl_universal_bits", "pairs")}
    out.update(texts=len(rows), random=RANDOM, pool=POOL, panel=PANEL)
    (TEXTS / "bar.json").write_text(json.dumps(out))
    return out


def teach(split: str, n: int, offset: int = 0, stride: int = 1) -> None:
    """prune() on the split's texts offset, offset + stride, ... of its first n -> texts/circuits[_heldout]/<id>.py and
    <id>.json (its score and VPD's answer's)."""
    nat = Native()
    b = json.loads((TEXTS / "bar.json").read_text())["kl_bits"]
    out = TEXTS / ("circuits" if split == "train" else "circuits_heldout")
    out.mkdir(exist_ok=True)
    for p in tasks(split)[:n][offset::stride]:
        if (out / f"{p.stem}.py").exists():
            continue
        ids, targets = text(p)
        t0 = time.time()
        full = nat.score(ids, targets, [nat.vpd_answer(ids)])[0]
        m, s = nat.prune(ids, targets, b)
        (out / f"{p.stem}.py").write_text(program(m, nat.names))
        (out / f"{p.stem}.json").write_text(json.dumps({"score": s, "vpd": full, "bar": b, "seconds": round(time.time() - t0, 1)}))
        print(f"{p.stem}: {full['pairs']} -> {s['pairs']} pairs, kl {full['kl_bits']:.3f} -> {s['kl_bits']:.3f} (bar {b:.3f}), {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("universal")
    a = sub.add_parser("bar")
    a.add_argument("--n", type=int, default=200)
    t = sub.add_parser("teach")
    t.add_argument("--split", choices=("train", "heldout"), required=True)
    t.add_argument("--n", type=int, required=True)
    t.add_argument("--offset", type=int, default=0)
    t.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "universal":
        print(json.dumps(universal()))
    elif args.cmd == "bar":
        print(json.dumps(bar(args.n)))
    else:
        teach(args.split, args.n, args.offset, args.stride)


if __name__ == "__main__":
    main()

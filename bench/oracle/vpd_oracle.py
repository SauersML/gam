"""The oracle on VPD's decomposition of vpd4l (#2951), answer mode: Qwen3 with a LoRA reads rank-one
subcomponents u v^T of the 4-layer Pile model (their two vectors, through learned linear maps, injected
at placeholder tokens with reporter.py's hook) and answers questions whose answers are exact native
measurements (vpd_labels.py, vpd_relations.py), trained on the proper log score.

Input of a subcomponent. Its site (layer, kind) in words; its read vector v and write vector u, each
mapped by a learned linear map of its own space (per kind and side: q, k, v read the normed residual and
write a head space; o reads the heads' concatenation and writes the residual; c_fc reads the normed
residual and writes the MLP's hidden layer; down_proj reads that and writes the residual) to the
oracle's hidden width, injected after decoder layer `--inject` at two placeholder tokens: the residual r
there becomes r + ||r|| m / ||m|| (m the mapped vector) plus a learned embedding of the slot's role and
of a log magnitude (reporter.Magnitude): log ||u|| ||v|| for a subcomponent itself (u v^T fixes only the
product of the norms), log of the measured effect for a neighbour.

Neighbourhood (context, chosen by measured effect, vpd_relations.py): on the subcomponent's strongest
texts, the upstream subcomponents whose removal most changes its activity at its peak and the downstream
ones whose activity most depends on it (their mean |change| relative to the reader's own peak |activity|,
over the measured contexts); the 4 strongest of each, each as two more placeholders.

Questions (a choice among labelled options; q = the softmax of the oracle's logits on the letters; loss
-ln q(measured answer)):
  activity      at a marked token: the subcomponent's activity level there, 0-9 of its largest
                |v . x| over its measured contexts; the token is the context's peak or uniform, half each;
  direction     removing (alpha 0) or amplifying (alpha 1.5) it: does a listed next token's probability
                at the marked token rise or fall (a token from the 10 rising and 10 falling most there);
  top           which of 4 tokens gains most probability when it is removed: the most raised one and 3
                of the 10 its removal lowers most there;
  continuation  multiplied by alpha (2, 4 or 8, stated), how the model continues greedily after its
                strongest peak: the amplified continuation, the unedited one (the answer when they agree)
                and one from another subcomponent;
  edge          cutting one measured upstream neighbour's write out of what this subcomponent reads (path
                patch): how its activity at the peak changes, relative to its own peak: falls or rises
                strongly (by more than 0.25), slightly (0.05 to 0.25), or barely (less than 0.05); the
                strong edges and the near-zero ones as measured;
  attribution   at a marked token where the model predicts X: which of 4 listed subcomponents (given as
                vectors, a candidate's slot) raises X most (the largest fall of log p(X) when removed).
Effect questions (direction, top) state the subcomponent's signed activity level at the marked token
(-9 to +9 of its largest |activity|; where it is active is the activity question's subject, what it does
there is theirs) and are drawn by effect stratum (the decade of the removal KL at the peak:
below 1e-5, then decades to 1e-1 and above), equal shares, or from the natural distribution.

Conditions at matched capacity (same base, adapter, maps, examples, order, steps; every placeholder
present, those a condition withholds receive nothing): graph (the subcomponent's vectors and its
neighbourhood), weights (its vectors alone), activity (no vectors; its three most active other contexts
as text with the peak token marked and its level), nothing. Candidates of an attribution question are
shown as vectors in graph and weights, named by site in every condition. graph_lens and weights_lens
add, as text, vpd_lens.py's readout of the weights (the tokens on which the subcomponent is positive and
negative, and the tokens its write then raises and lowers through the unembedding; for an attribution
question, each candidate's), from Table's --lens file. For a library warm-started from public
transcoders, weights_examples adds to weights, and examples gives alone, the transcoder feature's public
top-activating examples and logits (transcoder_examples.py; --feature-examples, --transcoders).

Held out: subcomponents of --heldout-layers (never trained on) on held-out texts (the held-out runs), and
trained layers on held-out texts.

  vpd_oracle.py train --base MODEL --labels TRAIN --relations TRAIN_REL --uv UV --condition C --steps N --out DIR
                      [--heldout-layers 2] [--examples 65536] [--batch 16] [--lr 1e-4] [--lora-rank 64]
  vpd_oracle.py evaluate --run DIR --labels HELDOUT --relations HELDOUT_REL --uv UV [--examples 4096]
  vpd_oracle.py compare --runs DIR... --eval eval_<labels>.jsonl --out SUMMARY.json
  vpd_oracle.py check --base MODEL --labels TRAIN --uv UV [--lens LENS] [--condition weights_lens] [--steps 40]
      the training pipeline's check: every layer's adapter gets a gradient, and 16 fixed questions are
      fitted to a mean loss below 0.01 nats (on vpd4l with Qwen3-0.6B: 0.0001 by step 26, the Mac)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reporter import Injection, Magnitude  # noqa: E402

KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj", "head", "function")  # VPD's sites, then the library's
LABELS = "ABCDEFGHIJ"
CONDITIONS = ("graph", "weights", "activity", "nothing", "graph_lens", "weights_lens", "weights_examples", "examples", "weights_activity")
QUESTIONS = ("activity", "direction", "top", "continuation", "edge", "attribution", "effect")
# The edit's effect on M at the edited token, KL(M_e || M) in bits: the edits driver's bins (also the
# effect question's answers; "barely" includes no change at all).
EFFECT_BINS = (0.01, 0.1, 1.0)
EFFECT_LEVELS = ("barely (under 0.01 bits)", "slightly (0.01 to 0.1 bits)", "clearly (0.1 to 1 bit)", "strongly (over 1 bit)")
BINS = 10
NEIGHBOURS = 4  # per direction
SLOTS = 1 + 2 * NEIGHBOURS  # subcomponents per example: itself, then neighbours or candidates
TOKENIZER = Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"
# reporter.Magnitude's role embedding indices for the slots.
ROLE = {"read": 4, "write": 5, "up_read": 1, "up_write": 3, "down_read": 0, "down_write": 2}
STRATA = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, float("inf"))
EDGE_LEVELS = ("falls strongly", "falls slightly", "barely changes", "rises slightly", "rises strongly")


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def site_name(layer: int, kind: str) -> str:
    return f"h.{layer}.{'mlp' if kind in ('c_fc', 'down_proj', 'function') else 'attn'}.{kind}"


def top_pair(M: torch.Tensor, iterations: int = 200) -> tuple[float, torch.Tensor, torch.Tensor]:
    """The largest singular value of M and its unit left and right vectors (sigma, l, r), by power
    iteration on M^T M from a fixed start (matrix products only: no decomposition routine)."""
    r = torch.ones(M.shape[1], dtype=torch.float64) / math.sqrt(M.shape[1])
    M = M.double()
    for _ in range(iterations):
        r = M.T @ (M @ r)
        r = r / r.norm()
    l = M @ r
    sigma = float(l.norm())
    return sigma, (l / max(sigma, 1e-300)).float(), r.float()


def head_vectors(uv: dict) -> None:
    """A library's MLP functions as {site}.U (write rows) and {site}.V (gate rows, transposed), and its
    heads as two vectors each: the
    write U = sigma l of the OV map W_O W_V's top singular pair (the residual direction it writes most,
    scaled by its gain) and the read V = the query-side vector of W_Q^T W_K's top pair (the residual
    direction its attention reads at the destination; rotary positions left out)."""
    for key in [k for k in uv if k.endswith(".mlp.function.write") and k.replace(".write", ".U") not in uv]:
        n = key.removesuffix(".write")  # readout's earlier files: the function's write rows and gate rows
        uv[f"{n}.U"], uv[f"{n}.V"] = uv[f"{n}.write"], uv[f"{n}.gate"].T.contiguous()
    for key in [k for k in uv if k.endswith(".attn.head.query")]:
        n = key.removesuffix(".query")
        Q, K, Vv, O = uv[f"{n}.query"].float(), uv[f"{n}.key"].float(), uv[f"{n}.value"].float(), uv[f"{n}.output"].float()
        U, V = [], []
        for h in range(Q.shape[0]):
            sigma, l, _ = top_pair(O[h] @ Vv[h])
            U.append(sigma * l)
            V.append(top_pair(Q[h].T @ K[h])[1])
        uv[f"{n}.U"], uv[f"{n}.V"] = torch.stack(U), torch.stack(V, 1)


def stratum(kl: float) -> int:
    return int(min(len(STRATA) - 2, np.searchsorted(STRATA, kl, side="right") - 1))


def level(a: float, peak: float) -> int:
    return 0 if peak <= 0 else min(BINS - 1, int(math.floor(BINS * abs(a) / peak)))


def edge_level(rel: float) -> int:
    if abs(rel) < 0.05:
        return 2
    return (0 if rel <= -0.25 else 1) if rel < 0 else (3 if rel < 0.25 else 4)


class Table:
    """A label run (vpd_labels.py), its relations (vpd_relations.py), the vectors, the tokenizer."""

    def __init__(self, root: Path, uv_path: Path, relations: Path | None = None, tokenizer: Path = TOKENIZER, lens: Path | None = None,
                 feature_examples: Path | None = None, transcoders: Path | None = None):
        import tokenizers

        from transcoder_examples import Features, kept

        self.tokens = load_file(str(root / "contexts.safetensors"))["tokens"].long()
        self.sites = {}
        for meta_path in sorted(root.glob("site_*.json")):
            meta = json.loads(meta_path.read_text())
            self.sites[(meta["layer"], meta["site"].split(".")[-1])] = (meta, load_file(str(meta_path.with_suffix(".safetensors"))))
        self.uv = load_file(str(uv_path))  # VPD's subcomponents, or a library's functions.safetensors
        head_vectors(self.uv)
        self.lens = load_file(str(lens)) if lens is not None else None
        self.layers = sorted({layer for layer, _ in self.sites})
        # The model's depth: a label table of some layers states it (qwen_labels.py), VPD's covers all 4.
        self.depth = max([m.get("model_layers", 0) for m, _ in self.sites.values()] + [max(self.layers) + 1])
        # Per kind, the widths of its read and write vectors (V is [d_read, C], U is [C, d_write]).
        self.dims = {kind: (int(self.uv[f"{site_name(layer, kind)}.V"].shape[0]), int(self.uv[f"{site_name(layer, kind)}.U"].shape[1])) for layer, kind in self.sites}
        self.features = Features(feature_examples) if feature_examples is not None else None
        self.kept = kept(transcoders) if transcoders is not None else None
        self.tok = tokenizers.Tokenizer.from_file(str(tokenizer))
        self.rel = {}
        self.offsets = None
        if relations is not None:
            meta = json.loads((relations / "relations.json").read_text())
            self.offsets = meta["site_offsets"]
            for part in ("upstream", "downstream", "edges", "attribution", "continuations"):
                path = relations / f"relations_{part}.safetensors"
                if path.exists():
                    self.rel[part] = load_file(str(path))
            self.peak = torch.zeros(meta["total"])
            for (layer, kind), (_, d) in self.sites.items():
                o = self.offsets[site_name(layer, kind)]
                self.peak[o : o + d["activity"].shape[0]] = d["activity"].float().abs().amax((1, 2))
        self._near: dict = {}

    def gid(self, layer: int, kind: str, c: int) -> int:
        return self.offsets[site_name(layer, kind)] + c

    def site_of(self, gid: int) -> tuple[int, str, int]:
        name = max((n for n, o in self.offsets.items() if o <= gid), key=lambda n: self.offsets[n])
        return int(name.split(".")[1]), name.split(".")[-1], gid - self.offsets[name]

    def vectors(self, layer: int, kind: str, c: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Subcomponent c of the table's site: a strided table (vpd_labels.py --edit row) holds every
        stride-th subcomponent, its numbers in `subcomponents`."""
        name = site_name(layer, kind)
        d = self.sites[(layer, kind)][1] if (layer, kind) in self.sites else {}
        n = int(d["subcomponents"][c]) if "subcomponents" in d else c
        return self.uv[f"{name}.V"][:, n], self.uv[f"{name}.U"][n]

    def neighbours(self, layer: int, kind: str, c: int) -> list[tuple[int, str, int, float, str]]:
        """Up to 4 upstream and 4 downstream measured neighbours: (layer, kind, index, mean relative
        effect, "up" | "down"), strongest first in each direction."""
        if self.offsets is None:
            return []
        key = (layer, kind, c)
        if key in self._near:
            return self._near[key]
        g = self.gid(layer, kind, c)
        out = []
        for part, sign in (("upstream", "up"), ("downstream", "down")):
            if part not in self.rel:
                continue
            J = self.rel[part]["ids"].shape[1]
            ids = self.rel[part]["ids"][g].reshape(-1)
            vals = (self.rel[part]["delta"][g].reshape(-1) / max(float(self.peak[g]), 1e-30)) if part == "upstream" else self.rel[part]["relative"][g].reshape(-1)
            # Mean over the measured contexts of the relative change (absent in a context: no change kept).
            total: dict[int, float] = {}
            for i, v in zip(ids.tolist(), vals.abs().tolist()):
                if i >= 0 and i != g:
                    total[i] = total.get(i, 0.0) + v / J
            best = sorted(total.items(), key=lambda kv: -kv[1])[:NEIGHBOURS]
            out += [(*self.site_of(i), v, sign) for i, v in best]
        self._near[key] = out
        return out

    def text(self, ctx: int, upto: int, mark: int) -> str:
        """The context's tokens up to `upto` (inclusive) with token `mark` set off by double brackets."""
        ids = self.tokens[ctx, : upto + 1].tolist()
        return self.tok.decode(ids[:mark]) + "⟦" + self.tok.decode([ids[mark]]) + "⟧" + self.tok.decode(ids[mark + 1 :])

    def piece(self, token: int) -> str:
        return json.dumps(self.tok.decode([int(token)]))

    def lens_value(self, layer: int, kind: str, c: int, token: int) -> float | None:
        """vpd_lens.py's write lens of one token: (w_X - mean over tokens) / std over tokens, w_X =
        r . (g_f * e_X) (positive activity raises tokens with positive values); None without a write."""
        n = site_name(layer, kind)
        if self.lens is None or f"{n}.write" not in self.lens:
            return None
        w = float(self.lens[f"{n}.write"][c].float() @ self.lens["unembed"][int(token)].float())
        return (w - float(self.lens[f"{n}.write_centre"][c])) / float(self.lens[f"{n}.write_scale"][c])


def effect_index(table: Table, keys: list) -> list[np.ndarray]:
    """Per effect stratum, the (site, subcomponent, context) triples whose removal KL falls in it."""
    per = [[] for _ in STRATA[:-1]]
    for k, key in enumerate(keys):
        kl = table.sites[key][1]["kl_ablate"].numpy()
        which = np.clip(np.searchsorted(STRATA, kl, side="right") - 1, 0, len(STRATA) - 2)
        for s_ in range(len(STRATA) - 1):
            c, j = np.nonzero(which == s_)
            per[s_].append(np.stack([np.full(len(c), k), c, j], 1))
    return [np.concatenate(p) for p in per]


def effect_bin(bits: float) -> int:
    return sum(bits >= b for b in EFFECT_BINS)


def examples(table: Table, layers: set[int], count: int, seed: int, stratified: bool = True, per_component: int = 1,
             only: tuple[str, ...] = (), rule: int = 0, held: int = 0, side: str = "trained") -> list[dict]:
    """`count` questions, the kinds in turn (those the table's relations support, or `only` those). With
    per_component K > 1, each subcomponent an effect question draws is asked about at K of its strongest
    contexts (the drawn one and K - 1 others, in the following questions of that kind).

    rule (a sanity check of the reader, not a measurement): direction questions about a token from the
    subcomponent's lens text (its 8 raised or 8 lowered tokens), answered by the rule the text states:
    the probability goes up iff (edit sign) * (stated activity sign) * (+1 raised, -1 lowered) > 0, the
    edit sign -1 for removal and +1 for amplification (rule 3: a parity of three facts the text states);
    rule 1: amplification of a positive activity only, so the answer is whether the token is listed as
    raised (one fact). A reader that cannot learn rule 1 from those inputs has a bug.

    held > 0: subcomponents split by their index in the table, c % held == 0 held out (side "heldout")
    and the others trained on (side "trained"), in every layer.

    A row-edit table (vpd_labels.py --edit row) adds the effect question (how much the next-token
    distribution changes, in the effect bins, no change included) and asks continuations from its own
    clean and edited continuations; its edits act at the marked token only, or from it on."""
    rng = random.Random(seed)
    ok = (lambda c: True) if not held else ((lambda c: c % held == 0) if side == "heldout" else (lambda c: c % held != 0))
    pending: dict[str, list] = {"direction": [], "top": [], "effect": []}
    keys = [k for k in table.sites if k[0] in layers]
    kinds = ["activity", "direction", "top"]
    row = all(m.get("edit") == "row" for m, _ in table.sites.values())
    if row:
        kinds.append("effect")
    if "continuations" in table.rel or (row and all("cont_clean" in d for _, d in table.sites.values())):
        kinds.append("continuation")
    if "edges" in table.rel:
        kinds.append("edge")
        edge_rows = [r for r in torch.nonzero(table.rel["edges"]["ids"][:, 0, 0] >= 0).reshape(-1).tolist() if table.site_of(r)[0] in layers]
    if "attribution" in table.rel:
        kinds.append("attribution")
        att = table.rel["attribution"]
    if only:
        assert set(only) <= set(kinds), (only, kinds)
        kinds = [k for k in kinds if k in only]
    if rule:
        assert kinds == ["direction"] and table.lens is not None, "the rule check asks direction questions with a lens"
        keys = [k for k in keys if int(table.lens[f"{site_name(*k)}.up"][0, 0]) >= 0]  # sites with a write lens
    index = effect_index(table, keys) if stratified else None
    if index is not None and held:
        index = [x[[ok(int(c)) for c in x[:, 1]]] if len(x) else x for x in index]
    out = []
    while len(out) < count:
        q = kinds[len(out) % len(kinds)]
        layer, kind = keys[rng.randrange(len(keys))]
        meta, d = table.sites[(layer, kind)]
        c = rng.randrange(meta["subcomponents"])
        j = rng.randrange(meta["top"] + meta["random"])
        if q in ("direction", "top", "effect") and index is not None:
            pool = index[(len(out) // len(kinds)) % len(index)]
            if pending[q]:
                k, c, j = pending[q].pop()
                layer, kind = keys[k]
                meta, d = table.sites[(layer, kind)]
            elif len(pool):
                k, c, j = (int(x) for x in pool[rng.randrange(len(pool))])
                layer, kind = keys[k]
                meta, d = table.sites[(layer, kind)]
                if per_component > 1:
                    others = [x for x in range(meta["top"]) if x != j]
                    pending.setdefault(q, [])
                    pending[q] = [(k, c, x) for x in rng.sample(others, min(per_component - 1, len(others)))]
        if not ok(c):
            continue
        ex = {"layer": layer, "kind": kind, "c": c, "kind_q": q, "j": j, "stratum": -1, "candidates": []}
        contexts = d["contexts"][c].long()
        act = d["activity"][c].float()
        peak = float(act.abs().max())
        here = ""
        where = " at the marked token only" if row else ""
        if q in ("direction", "top", "effect"):  # an effect question states the activity there (an input-side measurement, not its answer)
            a_here = float(act[j, int(d["position"][c, j])])
            signed = int(math.copysign(level(a_here, peak), a_here))
            here = f"At the marked token its activity is {signed:+d} (9 is its largest |activity| over its texts). "
        if q == "activity":
            p = int(d["position"][c, j]) if rng.random() < 0.5 else rng.randrange(act.shape[1])
            ex.update(context=int(contexts[j]), position=p, options=[str(b) for b in range(BINS)], answer=level(float(act[j, p]), peak),
                      question="How active is the component at the marked token, on a scale from 0 (inactive) to 9 (its largest activity)?")
        elif q == "direction":
            p = int(d["position"][c, j])
            edit, side, r = rng.choice(["ablate", "amplify"]), rng.choice(["up", "down"]), rng.randrange(10)
            if rule == 1:
                edit = "amplify"
            verb = ("removed" if edit == "ablate" else f"made {meta.get('amplify', 1.5):g} times stronger") + where
            token, answer = int(d[f"{side}_ids_{edit}"][c, j, r]), 0 if float(d[f"{side}_dp_{edit}"][c, j, r]) > 0 else 1
            if rule:
                a_sign = int(math.copysign(1, a_here)) if level(a_here, peak) > 0 else 0
                if a_sign == 0 or (rule == 1 and a_sign < 0):
                    continue
                listed = table.lens[f"{site_name(layer, kind)}.{side}"][c]
                token = int(listed[rng.randrange(int((listed >= 0).sum()))])
                answer = 0 if (-1 if edit == "ablate" else 1) * a_sign * (1 if side == "up" else -1) > 0 else 1
            ex.update(context=int(contexts[j]), position=p, stratum=stratum(float(d["kl_ablate"][c, j])), options=["up", "down"],
                      answer=answer, edit=edit, option_ids=[token], change=(-1.0 if edit == "ablate" else meta.get("amplify", 1.5) - 1.0) * signed / (BINS - 1),
                      question=here + f"If the component is {verb}, does the probability that the next token after the marked token is {table.piece(token)} go up or go down?")
        elif q == "top":
            p = int(d["position"][c, j])
            truth = int(d["up_ids_ablate"][c, j, 0])
            # Distractors: tokens its removal lowers here (likely in this context, so the text alone does not
            # tell them apart; the direction of its effect does).
            pool = [int(x) for x in d["down_ids_ablate"][c, j].tolist() if int(x) != truth]
            if len(pool) < 3:
                continue
            options = [truth] + rng.sample(pool, 3)
            order = list(range(4))
            rng.shuffle(order)
            ex.update(context=int(contexts[j]), position=p, stratum=stratum(float(d["kl_ablate"][c, j])), options=[table.piece(options[i]) for i in order],
                      answer=order.index(0), edit="ablate", option_ids=[int(options[i]) for i in order], change=-signed / (BINS - 1), question=here + f"If the component is removed{where}, which of these next tokens after the marked token gains the most probability?")
        elif q == "effect":
            p = int(d["position"][c, j])
            edit = rng.choice(["ablate", "amplify"])
            verb = ("removed" if edit == "ablate" else f"made {meta.get('amplify', 1.5):g} times stronger") + where
            ex.update(context=int(contexts[j]), position=p, stratum=stratum(float(d["kl_ablate"][c, j])), options=list(EFFECT_LEVELS), edit=edit,
                      answer=effect_bin(float(d[f"effect_{edit}"][c, j])),
                      question=here + f"If the component is {verb}, how much does the model's distribution of the next token after the marked token change?")
        elif q == "continuation" and row:
            J = meta.get("continue", 1)
            j = rng.randrange(J)
            edit, family = rng.choice(["ablate", "amplify"]), rng.choice(["row", "from"])
            edited, clean = d[f"cont_{family}_{edit}"][c, j].tolist(), d["cont_clean"][c, j].tolist()
            other = clean
            for _ in range(20):
                other = d["cont_clean"][rng.randrange(d["cont_clean"].shape[0]), 0].tolist()
                if other not in (edited, clean):
                    break
            text = lambda ids: json.dumps(table.tok.decode([i for i in ids if i >= 0]))  # noqa: E731
            options = [f"changed to {text(edited)}"] if edited != clean else []
            options = [o for o in options] + [f"unchanged: {text(clean)}", f"changed to {text(other)}"]
            answer_text = options[0]
            rng.shuffle(options)
            p = int(d["position"][c, j])
            verb = "removed" if edit == "ablate" else f"made {meta.get('amplify', 1.5):g} times stronger"
            span = "at the marked token only" if family == "row" else "from the marked token on (every later token too)"
            ex.update(context=int(contexts[j]), position=p, j=j, options=options, answer=options.index(answer_text), edit=edit, family=family,
                      question=f"If the component is {verb} {span}, how does the model continue the text after the marked token (greedy decoding, {len(clean)} tokens)?")
        elif q == "continuation":
            cont = table.rel["continuations"]
            g = table.gid(layer, kind, c)
            if int(cont["clean"][g, 0]) < 0:
                continue  # not measured (a partial relations run)
            alpha = rng.choice([2, 4, 8])
            amp, clean = cont[f"alpha_{alpha}"][g].tolist(), cont["clean"][g].tolist()
            other = amp
            while other in (amp, clean) or other[0] < 0:
                other = cont["clean"][rng.randrange(cont["clean"].shape[0])].tolist()
            text = lambda ids: json.dumps(table.tok.decode([i for i in ids if i >= 0]))  # noqa: E731
            options = [amp] + ([clean] if clean != amp else []) + [other]
            order = list(range(len(options)))
            rng.shuffle(order)
            p = int(d["position"][c, 0])
            ex.update(context=int(contexts[0]), position=p, j=0, options=[text(options[i]) for i in order], answer=order.index(0),
                      question=f"If the component is made {alpha} times stronger, how does the model continue the text after the marked token (greedy decoding)?")
        elif q == "edge":
            g = edge_rows[rng.randrange(len(edge_rows))]
            layer, kind, c = table.site_of(g)
            meta, d = table.sites[(layer, kind)]
            e = table.rel["edges"]
            which = 2 if rng.random() < 1 / 3 else rng.randrange(2)  # a third near-zero edges
            a_gid = int(e["ids"][g, 0, which])
            rel = float(e["activity"][g, 0, which]) / max(float(table.peak[g]), 1e-30)
            al, ak, ac = table.site_of(a_gid)
            p = int(d["position"][c, 0])
            ex.update(layer=layer, kind=kind, c=c, j=0, context=int(d["contexts"][c, 0]), position=p, options=list(EDGE_LEVELS), answer=edge_level(rel),
                      candidates=[(al, ak, ac, abs(rel), "up")], stratum=int(e["strong"][g, 0, which]),
                      question=f"If the write of the upstream component C1 (layer {al}, {ak}) is removed from what this component reads at the marked token (everything else unchanged), how does this component's activity there change?")
        else:
            r = rng.randrange(att["delta"].shape[0])
            ids, delta = att["ids"][r].tolist(), att["delta"][r].tolist()
            best = int(np.argmin(delta))
            rest = [i for i in range(len(ids)) if i != best]
            chosen = [best] + rng.sample(rest, 3)
            order = list(range(4))
            rng.shuffle(order)
            cands = [table.site_of(int(ids[chosen[i]])) for i in order]
            ctx, p = int(att["context"][r]), int(att["position"][r])
            ex.update(layer=-1, kind="", c=-1, j=-1, context=ctx, position=p, options=[f"C{i + 1} (layer {l}, {k})" for i, (l, k, _) in enumerate(cands)],
                      candidates=[(l, k, cc, 1.0, "up") for l, k, cc in cands], answer=order.index(0),
                      question=f"At the marked token the model predicts {table.piece(att['token'][r])} next. Which of the listed components raises that prediction most?")
        if row and ex.get("edit") and ex["c"] >= 0:
            ex["effect"] = float(d[f"effect_{ex['edit']}"][ex["c"], ex["j"]])  # KL(M_e || M) at the edited token, bits
        out.append(ex)
    return out


def exemplars(table: Table, ex: dict, n: int = 3) -> str:
    """The subcomponent's n most active measured contexts other than the example's, as marked text."""
    meta, d = table.sites[(ex["layer"], ex["kind"])]
    act = d["activity"][ex["c"]].float()
    peak = float(act.abs().max())
    lines = []
    for j in range(meta["top"]):
        if j == ex["j"] or len(lines) == n:
            continue
        p = int(d["position"][ex["c"], j])
        lines.append(f"- {table.text(int(d['contexts'][ex['c'], j]), p, p)!r} (level {level(float(act[j, p]), peak)})")
    return "On other texts the component is most active at the marked tokens:\n" + "\n".join(lines)


def base(condition: str) -> str:
    """The condition's vector input (a *_lens, *_examples or *_activity condition adds text to graph's or
    weights')."""
    return condition.removesuffix("_lens").removesuffix("_examples").removesuffix("_activity")


def slots(table: Table, ex: dict, condition: str) -> list[tuple]:
    """The example's subcomponents in slot order: (layer, kind, index, log magnitude, role prefix, shown)."""
    condition = base(condition)
    out = []
    if ex["c"] >= 0:
        v, u = table.vectors(ex["layer"], ex["kind"], ex["c"])
        out.append((ex["layer"], ex["kind"], ex["c"], math.log(float(u.norm() * v.norm())), "", condition in ("graph", "weights")))
    if ex["kind_q"] == "attribution":
        out += [(l, k, c, 0.0, "up_", condition in ("graph", "weights")) for l, k, c, _, _ in ex["candidates"]]
    elif ex["kind_q"] == "edge":
        # The cut edge's writer carries its own log ||u|| ||v||: its measured effect is the answer.
        for l, k, c, _, _ in ex["candidates"]:
            v, u = table.vectors(l, k, c)
            out.append((l, k, c, math.log(float(u.norm() * v.norm())), "up_", condition == "graph"))
    if ex["c"] >= 0 and ex["kind_q"] != "edge":
        out += [(l, k, c, math.log(max(s, 1e-30)), f"{d}_", condition == "graph") for l, k, c, s, d in table.neighbours(ex["layer"], ex["kind"], ex["c"])]
    return out[:SLOTS]


def lens_text(table: Table, layer: int, kind: str, c: int, name: str = "It", reads: bool = True) -> str:
    """vpd_lens.py's readout of one subcomponent (weights only), stated per sign of its activity (its
    write alone with reads=False)."""
    L = table.lens
    n = site_name(layer, kind)
    words = lambda key: ", ".join(table.piece(int(t)) for t in L[f"{n}.{key}"][c].tolist() if t >= 0)  # noqa: E731
    if kind == "head":
        attends = f"{name} attends from tokens like {words('attn_from')} to tokens like {words('attn_to')}. " if reads else f"{name}: "
        return attends + f"Its strongest copying: from tokens like {words('pos_read')} it raises {words('up')} and lowers {words('down')} (from tokens like {words('neg_read')} the reverse)."
    head = f"{name} is positive on tokens like {words('pos_read')}; negative on tokens like {words('neg_read')}. " if reads else f"{name}: "
    if int(L[f"{n}.up"][c, 0]) < 0:
        return head + ("It changes" if reads else "changes") + " attention scores (no direct write)."
    return head + ("Positive" if reads else "positive") + f" activity raises {words('up')} and lowers {words('down')} (negative activity the reverse)."


def prompt(table: Table, ex: dict, condition: str) -> tuple[str, str]:
    """The user turn around the placeholders: (before, after)."""
    lens = condition.endswith("_lens")
    public = condition.endswith("examples")
    shown = condition == "activity" or condition.endswith("_activity")  # its top-activating texts from the label table
    condition = base(condition)
    n = table.depth
    if ex["c"] >= 0:
        before = f"A component of a {n}-layer language model: layer {ex['layer']}, {ex['kind']}. Its vectors, then those of related components:"
    else:
        before = f"Components of a {n}-layer language model. Their vectors:"
    info = "\n" + exemplars(table, ex) if shown and ex["c"] >= 0 else ""
    if condition == "graph" and ex["c"] >= 0 and ex["kind_q"] not in ("edge", "attribution"):
        lines = [f"N{i + 1}: layer {l}, {k}, {'upstream: its removal changes this component by' if d == 'up' else 'downstream: depends on this component by'} {s:.2g} of its peak"
                 for i, (l, k, _, s, d) in enumerate(table.neighbours(ex["layer"], ex["kind"], ex["c"]))]
        info += "\nAfter its own two vectors come those of its most strongly related components, in this order:\n" + "\n".join(lines)
    if ex["kind_q"] == "attribution":
        info += "\nThe listed components' vectors come in the order C1, C2, C3, C4."
    if public and ex["c"] >= 0:
        info += "\n" + table.features.text(ex["layer"], table.kept[ex["layer"]][ex["c"]])
    if lens:
        info += "\nIts weights read through the token embeddings and the unembedding (no measurement):\n"
        if ex["c"] >= 0:
            info += lens_text(table, ex["layer"], ex["kind"], ex["c"])
            values = [table.lens_value(ex["layer"], ex["kind"], ex["c"], t) for t in ex.get("option_ids", [])] if ex["kind_q"] in ("direction", "top") else []
            if values and values[0] is not None:
                info += "\nIts write's lens value (standard deviations over tokens; positive activity raises tokens with positive values) for " + ", ".join(
                    f"{table.piece(t)}: {v:+.1f}" for t, v in zip(ex["option_ids"], values)) + "."
                # The edit's direct effect through the unembedding: the change of its activity (the stated level
                # times the edit's factor minus one, over 9) times the lens value; weights and the stated activity only.
                info += "\nThe edit's direct effect on the logit through the unembedding (change of its activity, in units of its largest, times the lens value): " + ", ".join(
                    f"{table.piece(t)}: {ex['change'] * v:+.2f}" for t, v in zip(ex["option_ids"], values)) + "."
        else:
            info += "\n".join(lens_text(table, l, k, c, f"C{i + 1}", reads=False) for i, (l, k, c, _, _) in enumerate(ex["candidates"]))
    text = table.text(ex["context"], ex["position"], ex["position"])
    listing = "\n".join(f"{LABELS[k]}. {o}" for k, o in enumerate(ex["options"]))
    after = f"{info}\nText: {text!r}\n{ex['question']}\n{listing}\nAnswer with the letter."
    return before, after


class Oracle(torch.nn.Module):
    def __init__(self, base: str, lora_rank: int, inject: int, dev, dims: dict, depth: int = 4):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.dev = dev
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(dev)
        self.model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules="all-linear", lora_dropout=0.0))
        width = self.width = model.config.hidden_size
        self.maps = torch.nn.ModuleDict({f"{k}_{side}": torch.nn.Linear(dims[k][i], width) for k in KINDS if k in dims for i, side in enumerate(("read", "write"))}).to(dev)
        self.magnitude = Magnitude(width, depth).to(dev)
        self.hook = Injection(1.0, inject)
        model.model.layers[inject].register_forward_hook(self.hook)
        self.inject = inject
        self.placeholder = self.tokenizer.encode(" ?", add_special_tokens=False)[0]
        self.letters = [self.tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS]

    def trainable(self):
        return [p for p in self.model.parameters() if p.requires_grad] + list(self.maps.parameters()) + list(self.magnitude.parameters())

    def load(self, run: Path):
        """An answer-mode run's adapter, maps and magnitude term (trainable)."""
        from peft import PeftModel

        inner = self.model.get_base_model()
        self.model = PeftModel.from_pretrained(inner, str(run / "adapter"), is_trainable=True)
        state = torch.load(run / "maps.pt")
        # A VPD run read on a library: maps of kinds it lacks, or of another width (another model), start fresh.
        own = self.maps.state_dict()
        maps = {k: v for k, v in state["maps"].items() if k in own and own[k].shape == v.shape}
        missing = self.maps.load_state_dict(maps, strict=False).missing_keys
        assert all(k.split(".")[0].rsplit("_", 1)[0] in ("head", "function") for k in missing), missing
        own = self.magnitude.state_dict()  # the layer embedding of another depth (another model) starts fresh
        self.magnitude.load_state_dict({k: v for k, v in state["magnitude"].items() if own[k].shape == v.shape}, strict=False)

    def encode(self, before: str, after: str, thinking: bool = False) -> tuple[list[int], list[int]]:
        marker = "\u0000V\u0000"
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": before + marker + after}], add_generation_prompt=True, enable_thinking=thinking, tokenize=False)
        head, tail = text.split(marker)
        a = self.tokenizer.encode(head, add_special_tokens=False)
        b = self.tokenizer.encode(tail, add_special_tokens=False)
        return a + [self.placeholder] * (2 * SLOTS) + b, list(range(len(a), len(a) + 2 * SLOTS))

    def injection(self, table: Table, rows_slots: list[list[tuple]], places: list[list[int]]):
        """The hook's batch: per row its slots' read and write vectors at its placeholder positions, each
        through its own map; withheld and empty slots receive nothing."""
        rows, cols, vecs, roles, mags, keeps, layers = [], [], [], [], [], [], []
        width = self.width
        for b, (items, where) in enumerate(zip(rows_slots, places)):
            for s in range(SLOTS):
                for h, side in enumerate(("read", "write")):
                    rows.append(b)
                    cols.append(where[2 * s + h])
                    if s < len(items):
                        layer, kind, c, mag, prefix, shown = items[s]
                        v, u = table.vectors(layer, kind, c)
                        vecs.append(self.maps[f"{kind}_{side}"]((v if side == "read" else u).to(self.dev)))
                        roles.append(ROLE[prefix + side])
                        mags.append(mag)
                        keeps.append(1.0 if shown else 0.0)
                        layers.append(layer)
                    else:
                        vecs.append(torch.zeros(width, device=self.dev))
                        roles.append(ROLE["up_" + side])
                        mags.append(0.0)
                        keeps.append(0.0)
                        layers.append(0)
        keep = torch.tensor(keeps, device=self.dev)
        unit = torch.nn.functional.normalize(torch.stack(vecs), dim=-1) * keep[:, None]
        logn = torch.tensor(mags, device=self.dev) * keep
        extra = self.magnitude(torch.tensor(roles, device=self.dev), torch.tensor(layers, device=self.dev), logn, 1.0 - keep)
        at = torch.full((len(rows),), self.inject, device=self.dev)
        return (torch.tensor(rows, device=self.dev), torch.tensor(cols, device=self.dev), unit, extra, keep, at)

    def log_q(self, table: Table, batch: list[dict], condition: str):
        enc = [self.encode(*prompt(table, ex, condition)) for ex in batch]
        width = max(len(ids) for ids, _ in enc)
        ids = torch.zeros(len(batch), width, dtype=torch.long)
        mask = torch.zeros(len(batch), width, dtype=torch.long)
        for b, (t, _) in enumerate(enc):
            ids[b, : len(t)] = torch.tensor(t)
            mask[b, : len(t)] = 1
        # The hook stays set until the caller is done with this batch (after its backward pass), so
        # checkpointed layers inject again when they are recomputed.
        self.hook.set(self.injection(table, [slots(table, ex, condition) for ex in batch], [places for _, places in enc]))
        inner = self.model.get_base_model()
        hidden = inner.model(input_ids=ids.to(self.dev), attention_mask=mask.to(self.dev)).last_hidden_state
        last = mask.sum(1) - 1
        k = max(len(ex["options"]) for ex in batch)
        logits = torch.nn.functional.linear(hidden[torch.arange(len(batch), device=self.dev), last.to(self.dev)], inner.lm_head.weight[self.letters[:k]]).float()
        valid = torch.tensor([[j < len(ex["options"]) for j in range(k)] for ex in batch], device=self.dev)
        return torch.log_softmax(logits.masked_fill(~valid, float("-inf")), -1), valid


def log_scores(log_q, valid, batch) -> torch.Tensor:
    answers = torch.tensor([ex["answer"] for ex in batch], device=log_q.device)
    return log_q.gather(-1, answers[:, None])[:, 0]


def table_of(args) -> Table:
    return Table(Path(args.labels), Path(args.uv), Path(args.relations) if args.relations else None, Path(args.tokenizer), Path(args.lens) if args.lens else None,
                 Path(args.feature_examples) if args.feature_examples else None, Path(args.transcoders) if args.transcoders else None)


def train(args):
    dev = device()
    torch.manual_seed(args.seed)
    held = {int(x) for x in args.heldout_layers.split(",") if x} if not args.heldout_every else set()
    table = table_of(args)
    data = examples(table, set(table.layers) - held, args.examples, args.seed, per_component=args.per_component, only=tuple(args.questions.split(",")) if args.questions else (), rule=args.rule,
                    held=args.heldout_every, side="trained")
    oracle = Oracle(args.base, args.lora_rank, args.inject, dev, table.dims, table.depth)
    if args.init:  # warm start: an earlier run's adapter and maps (same base model); new kinds' maps start fresh
        oracle.load(Path(args.init))
    oracle.model.base_model.model.gradient_checkpointing_enable()
    oracle.model.base_model.model.config.use_cache = False
    oracle.model.train()  # checkpointing acts only in training mode (no dropout in Qwen3 or the adapter)
    optimizer = torch.optim.AdamW(oracle.trainable(), lr=args.lr, weight_decay=0.0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.jsonl", "w")
    started = time.time()
    for step in range(args.steps):
        batch = data[(step * args.batch) % len(data) : (step * args.batch) % len(data) + args.batch]
        log_q, valid = oracle.log_q(table, batch, args.condition)
        loss = -log_scores(log_q, valid, batch).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        oracle.hook.set(None)
        torch.nn.utils.clip_grad_norm_(oracle.trainable(), 1.0)
        optimizer.step()
        log.write(json.dumps({"step": step, "loss_nats": float(loss.detach()), "seconds": time.time() - started}) + "\n")
        log.flush()
    torch.save({"maps": oracle.maps.state_dict(), "magnitude": oracle.magnitude.state_dict()}, out / "maps.pt")
    oracle.model.save_pretrained(str(out / "adapter"))
    (out / "config.json").write_text(json.dumps(vars(args)))
    print(json.dumps({"condition": args.condition, "steps": args.steps, "seconds": time.time() - started, "last_loss": float(loss.detach())}))


def check(args):
    """The training pipeline's check (see the module's usage): a reader that cannot fit 16 questions it
    sees every step, or whose adapter misses a layer's gradient, has a bug (prompt, scored position,
    answer tokens, injection or optimizer), whatever the questions ask."""
    dev = device()
    torch.manual_seed(args.seed)
    table = table_of(args)
    batch = examples(table, set(table.layers), 16, args.seed, per_component=4, only=("direction",))
    oracle = Oracle(args.base, 64, 1, dev, table.dims, table.depth)
    oracle.model.base_model.model.gradient_checkpointing_enable()
    oracle.model.base_model.model.config.use_cache = False
    oracle.model.train()
    optimizer = torch.optim.AdamW(oracle.trainable(), lr=1e-4, weight_decay=0.0)
    losses = []
    for step in range(args.steps):
        log_q, valid = oracle.log_q(table, batch, args.condition)
        loss = -log_scores(log_q, valid, batch).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        oracle.hook.set(None)
        if step == 0:
            missing = sorted({n.split("layers.")[1].split(".")[0] for n, p_ in oracle.model.named_parameters() if p_.requires_grad and "lora_B" in n and (p_.grad is None or float(p_.grad.norm()) == 0.0)})
            assert not missing, f"layers whose adapter gets no gradient: {missing}"
        torch.nn.utils.clip_grad_norm_(oracle.trainable(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    print(json.dumps({"losses": [round(x, 4) for x in losses]}))
    assert min(losses[-5:]) < 0.01, f"16 fixed questions not fitted: last losses {losses[-5:]}"


@torch.no_grad()
def evaluate(args):
    dev = device()
    config = json.loads((Path(args.run) / "config.json").read_text())
    table = table_of(args)
    oracle = Oracle(config["base"], config["lora_rank"], config["inject"], dev, table.dims, table.depth)
    oracle.load(Path(args.run))
    oracle.model.eval()
    every = config.get("heldout_every", 0)
    held = {int(x) for x in config["heldout_layers"].split(",") if x} if not every else set()
    splits = (("heldout_layers", held, "trained"), ("trained_layers", set(table.layers) - held, "trained")) if not every else \
        (("heldout_subcomponents", set(table.layers), "heldout"), ("trained_subcomponents", set(table.layers), "trained"))
    rows = []
    for split, layers, side in splits:
        for distribution in ("natural", "stratified"):
            data = examples(table, layers, args.examples, args.seed + 1, stratified=distribution == "stratified",
                            only=tuple(config["questions"].split(",")) if config.get("questions") else (), rule=3 if config.get("rule") is True else int(config.get("rule", 0)),
                            held=every, side=side)
            for s in range(0, len(data), config["batch"]):
                batch = data[s : s + config["batch"]]
                lq, valid = oracle.log_q(table, batch, config["condition"])
                oracle.hook.set(None)
                best = lq.argmax(-1).tolist()
                for ex, score, guess in zip(batch, log_scores(lq, valid, batch).tolist(), best):
                    rows.append({"split": split, "distribution": distribution, "question": ex["kind_q"], "stratum": ex["stratum"], "layer": ex["layer"], "kind": ex["kind"],
                                 "c": ex["c"], "context": ex["context"], "log_score": score, "options": len(ex["options"]), "correct": int(guess == ex["answer"]),
                                 "effect": ex.get("effect")})
    (Path(args.run) / f"eval_{Path(args.labels).name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {}
    for split in [s_ for s_, _, _ in splits]:
        for q in QUESTIONS:
            v = [r for r in rows if r["split"] == split and r["question"] == q and r["distribution"] == "natural"]
            if v:
                summary[f"{split}/{q}"] = float(np.mean([r["log_score"] for r in v]))
                summary[f"{split}/{q}/accuracy"] = float(np.mean([r["correct"] for r in v]))
    print(json.dumps({"condition": config["condition"], **summary}))


def compare(args):
    """Per split and question: each condition's mean log score and its paired gain over `nothing`
    (the same examples in every run: evaluate draws them from the same seed), with standard errors;
    effect questions also per stratum, edge questions per strong and near-zero edges, and questions about
    an edit (row-edit tables) per bin of its effect on M at the edited token, KL(M_e || M) in bits, over
    both distributions ("/effect_bits_<low>-<high>")."""
    runs = {}
    for d in args.runs:
        config = json.loads((Path(d) / "config.json").read_text())
        runs[config["condition"]] = [json.loads(line) for line in open(Path(d) / args.eval)]
    base = runs["nothing"]
    table = {}
    groups = [("natural", None), ("stratified", None)] + [("stratified", k) for k in range(len(STRATA) - 1)]
    splits = list(dict.fromkeys(r["split"] for r in base))
    for condition, rows in runs.items():
        for split in splits:
            for q in QUESTIONS:
                for b in range(len(EFFECT_BINS) + 1):
                    sel = [(r, n) for r, n in zip(rows, base) if r["split"] == split and r["question"] == q and r.get("effect") is not None and effect_bin(r["effect"]) == b]
                    if not sel:
                        continue
                    d = np.array([r["log_score"] - n["log_score"] for r, n in sel])
                    edges = (0.0, *EFFECT_BINS, float("inf"))
                    table[f"{condition}/{split}/{q}/effect_bits_{edges[b]:g}-{edges[b + 1]:g}"] = {
                        "examples": len(d), "log_score_nats": float(np.mean([r["log_score"] for r, _ in sel])), "gain_over_nothing_nats": float(d.mean()),
                        "standard_error_nats": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None, "accuracy": float(np.mean([r["correct"] for r, _ in sel]))}
                for distribution, k in groups:
                    pairs = [(r["log_score"], b["log_score"]) for r, b in zip(rows, base)
                             if r["split"] == split and r["question"] == q and r["distribution"] == distribution and (k is None or r["stratum"] == k)]
                    if not pairs or (k is not None and q in ("activity", "continuation", "attribution")):
                        continue
                    d = np.array([a - b for a, b in pairs])
                    suffix = "" if distribution == "natural" else "/stratified" + ("" if k is None else (f"/kl{STRATA[k]:g}-{STRATA[k + 1]:g}" if q != "edge" else ("/strong" if k == 1 else "/near_zero")))
                    table[f"{condition}/{split}/{q}{suffix}"] = {"examples": len(d), "log_score_nats": float(np.mean([a for a, _ in pairs])),
                                                                "gain_over_nothing_nats": float(d.mean()), "standard_error_nats": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None}
                    hits = [r["correct"] for r in rows if "correct" in r and r["split"] == split and r["question"] == q and r["distribution"] == distribution and (k is None or r["stratum"] == k)]
                    if hits:
                        table[f"{condition}/{split}/{q}{suffix}"]["accuracy"] = float(np.mean(hits))
    Path(args.out).write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--labels", required=True)
        p.add_argument("--uv", required=True)
        p.add_argument("--relations", help="vpd_relations.py's output for these labels")
        p.add_argument("--tokenizer", default=str(TOKENIZER), help="the target's tokenizer.json")
        p.add_argument("--lens", help="vpd_lens.py build's output (the *_lens conditions)")
        p.add_argument("--feature-examples", help="public transcoder features/ directory (the *examples conditions)")
        p.add_argument("--transcoders", help="the library's fit OUT holding transcoder_l*.safetensors (its functions' transcoder indices)")
        p.add_argument("--seed", type=int, default=0)
    t = sub.choices["train"]
    t.add_argument("--base", required=True)
    t.add_argument("--condition", required=True, choices=CONDITIONS)
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--heldout-layers", default="2")
    t.add_argument("--heldout-every", type=int, default=0, help="hold out the subcomponents whose table index is a multiple of N (every layer) instead of layers")
    t.add_argument("--examples", type=int, default=65536)
    t.add_argument("--init", help="an earlier run to start from (its adapter and maps)")
    t.add_argument("--per-component", type=int, default=1, help="effect questions per drawn subcomponent, at its strongest contexts")
    t.add_argument("--questions", default="", help="only these question kinds, comma-separated (default: all the relations support)")
    t.add_argument("--rule", type=int, default=0, choices=(0, 1, 3), help="the reader sanity check: direction questions answered by the lens rule (see examples)")
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--lora-rank", type=int, default=64)
    t.add_argument("--inject", type=int, default=1)
    e = sub.choices["evaluate"]
    e.add_argument("--run", required=True)
    e.add_argument("--examples", type=int, default=4096)
    k = sub.add_parser("check")
    for a in ("--base", "--labels", "--uv"):
        k.add_argument(a, required=True)
    for a in ("--relations", "--lens", "--feature-examples", "--transcoders"):
        k.add_argument(a)
    k.add_argument("--tokenizer", default=str(TOKENIZER))
    k.add_argument("--condition", default="weights_lens", choices=CONDITIONS)
    k.add_argument("--steps", type=int, default=40)
    k.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--eval", required=True, help="the evaluation file name inside each run, eval_<labels dir name>.jsonl")
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    {"train": train, "evaluate": evaluate, "compare": compare, "check": check}[args.command](args)


if __name__ == "__main__":
    main()

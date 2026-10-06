"""Lens readout of VPD's vpd4l subcomponents (#2951): fixed functions of the target's own weights, with
no measurement and no label, given to the oracle as text and used as a weights-only baseline.

Write lens. The residual direction r of a subcomponent's write, read through the final norm's gain g_f and
the tied unembedding: w_X = r . (g_f * e_X) for every token X (e_X its embedding row); r = u for o_proj and
down_proj, W_O u for v_proj (its layer's o_proj), W_down u for c_fc (its layer's down_proj, the GELU taken
as the identity); q_proj and k_proj change attention scores and have no write lens.
Read lens. a_t, the subcomponent's activity v . x when its site's input x is computed from token t's
embedding alone: rms(e_t) * g for the residual readers (q, k, v_proj with the attention norm's gain,
c_fc with the MLP norm's), gelu(W_fc (rms(e_t) * g)) for down_proj, W_V (rms(e_t) * g) for o_proj (each
head attending to its own position).
Gauge. u v^T = (-u)(-v)^T, so the lens is stated per sign of activity: the tokens on which it is
positive and the tokens its activity then raises (w > 0) and lowers (w < 0); the tokens on which it is
negative and the tokens it then raises (w < 0) and lowers (w > 0). Tokens are those the Pile data file
holds; the listed ones are ranked by unigram frequency p_t times effect: p_t a_t for reading, and
p_X (w_X - E_p[w]), the first-order change of X's probability under p, for writing.

Values. For the oracle's effect questions (the lens value of the asked tokens), the file also holds
{site}.write (r per subcomponent), {site}.write_centre and {site}.write_scale (the mean and standard
deviation of w over the data's tokens) and unembed (g_f * e_X for every token), in float16 and float32.

Baseline. A removal question at a marked token t asks about the change of a next token X: the lens
predicts the logit change -a_t w_X (alpha 0) or +0.5 a_t w_X (alpha 1.5), with w centred over the tokens;
q(up) = sigmoid(beta s), s that change over its scale (|a| over the read lens's spread, w over the write
lens's), one beta fitted by maximum likelihood on the trained layers' questions and scored on the held-out
layer, beside the no-input reader's log score on the same questions (and their product of experts).

  vpd_lens.py values --lens LENS --uv UV --data PILE.npy      add the values to an existing lens file
  vpd_lens.py build --uv UV | --functions LIBRARY/functions.safetensors --data PILE.npy --out LENS.safetensors [--k 8]
  vpd_lens.py baseline --labels HELDOUT --relations HELDOUT_REL --uv UV --data PILE.npy --nothing RUN --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd_2951"))
import vpd_model as VM  # noqa: E402

KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")


def site(layer: int, kind: str) -> str:
    return f"h.{layer}.{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}.{kind}"


class Lens:
    """The target's weights and the subcomponents' vectors, with the write and read lenses."""

    def __init__(self, uv_path: Path | None, data: Path):
        self.t = VM.load_target("cpu")
        if uv_path is not None:
            d = load_file(str(uv_path))
            self.uv = {n: (d[f"{n}.U"].float(), d[f"{n}.V"].float()) for n in VM.site_names()}
        E = self.t.wte.float()
        self.E = E
        self.Ef = E * self.t.ln_f.float()  # the unembedding through the final norm's gain
        self.Er = E * torch.rsqrt(E.pow(2).mean(-1, keepdim=True) + self.t.eps)  # rms(e_t)
        counts = np.bincount(np.asarray(np.load(str(data), mmap_mode="r")).reshape(-1), minlength=E.shape[0])[: E.shape[0]]
        self.seen = torch.from_numpy(counts > 0)
        self.freq = torch.from_numpy(counts / counts.sum()).float()  # the unigram distribution of the data

    def W(self, layer: int, kind: str) -> torch.Tensor:
        return self.t.site(site(layer, kind)).W.float()

    def write_dirs(self, layer: int, kind: str) -> torch.Tensor | None:
        """[C, 768] residual directions of the site's writes, or None (q, k)."""
        U = self.uv[site(layer, kind)][0]
        if kind in ("o_proj", "down_proj"):
            return U
        if kind == "v_proj":
            return U @ self.W(layer, "o_proj").T
        if kind == "c_fc":
            return U @ self.W(layer, "down_proj").T
        return None

    def site_inputs(self, layer: int, kind: str, tokens: torch.Tensor) -> torch.Tensor:
        """[T, d_in] the site's input computed from each token's embedding alone."""
        g = self.t.norms[2 * layer + (1 if kind in ("c_fc", "down_proj") else 0)].float()
        x = self.Er[tokens] * g
        if kind == "down_proj":
            return VM.gelu_tanh(x @ self.W(layer, "c_fc").T)
        if kind == "o_proj":
            return x @ self.W(layer, "v_proj").T
        return x

    def write(self, layer: int, kind: str, rows: torch.Tensor) -> torch.Tensor | None:
        """[R, vocab] w for subcomponents `rows` of the site."""
        r = self.write_dirs(layer, kind)
        return None if r is None else r[rows] @ self.Ef.T

    def read(self, layer: int, kind: str, rows: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """[R, T] a for subcomponents `rows` on `tokens`."""
        V = self.uv[site(layer, kind)][1]
        return (self.site_inputs(layer, kind, tokens) @ V[:, rows]).T

    def read_vocab(self, layer: int, kind: str, rows: torch.Tensor, vocab: torch.Tensor) -> torch.Tensor:
        """[R, len(vocab)] a on every token of `vocab`, the site's inputs computed once."""
        cache = self.__dict__.setdefault("_inputs", {})
        if (layer, kind) not in cache:
            cache[(layer, kind)] = self.site_inputs(layer, kind, vocab)
        return (cache[(layer, kind)] @ self.uv[site(layer, kind)][1][:, rows]).T


@torch.no_grad()
def values(lens: Lens, out: dict) -> None:
    """The write lens's values (see the module's Values): r, the mean and spread of w over the data's
    tokens, and the unembedding through the final norm's gain."""
    vocab = torch.nonzero(lens.seen).reshape(-1)
    out["unembed"] = lens.Ef.half()
    for layer in range(lens.t.n_layer):
        for kind in KINDS:
            r = lens.write_dirs(layer, kind)
            if r is None:
                continue
            w = r @ lens.Ef[vocab].T
            out[f"{site(layer, kind)}.write"] = r.half()
            out[f"{site(layer, kind)}.write_centre"] = w.mean(1)
            out[f"{site(layer, kind)}.write_scale"] = w.std(1)


@torch.no_grad()
def build(args):
    lens = Lens(Path(args.uv), Path(args.data))
    vocab = torch.nonzero(lens.seen).reshape(-1)
    k = args.k
    out = {}
    for layer in range(lens.t.n_layer):
        for kind in KINDS:
            n = site(layer, kind)
            C = lens.uv[n][0].shape[0]
            pos_read, neg_read = torch.empty(C, k, dtype=torch.int32), torch.empty(C, k, dtype=torch.int32)
            up, down = torch.full((C, k), -1, dtype=torch.int32), torch.full((C, k), -1, dtype=torch.int32)
            for s in range(0, C, 512):
                rows = torch.arange(s, min(C, s + 512))
                # Ranked by frequency times effect: a token's share of the activity the data gives it, and
                # the first-order probability change p_X (w_X - E_p[w]) under the unigram distribution p.
                f = lens.freq[vocab]
                a = lens.read_vocab(layer, kind, rows, vocab) * f
                pos_read[rows] = vocab[a.topk(k, dim=1).indices].to(torch.int32)
                neg_read[rows] = vocab[(-a).topk(k, dim=1).indices].to(torch.int32)
                w = lens.write(layer, kind, rows)
                if w is not None:
                    w = w[:, vocab]
                    w = (w - (w * f).sum(1, keepdim=True) / f.sum()) * f
                    up[rows] = vocab[w.topk(k, dim=1).indices].to(torch.int32)
                    down[rows] = vocab[(-w).topk(k, dim=1).indices].to(torch.int32)
            for name, val in (("pos_read", pos_read), ("neg_read", neg_read), ("up", up), ("down", down)):
                out[f"{n}.{name}"] = val
            print(json.dumps({"site": n, "subcomponents": C}), flush=True)
    values(lens, out)
    save_file(out, args.out)


@torch.no_grad()
def add_values(args):
    out = dict(load_file(args.lens))
    values(Lens(Path(args.uv), Path(args.data)), out)
    save_file(out, args.lens)


@torch.no_grad()
def build_library(args):
    """The lens of a library (readout's functions.safetensors): an MLP function i writes h_i u_i (write
    direction u_i, read through its pre-activation v_i . x + b_i on each token alone, x through the MLP
    norm); a head is read through the top singular pairs of its OV map W_O W_V (copying: from source
    tokens along the right vector, writing along the left) and of W_Q^T W_K (attention: from destination
    tokens along the left vector to source tokens along the right; rotary positions left out)."""
    from vpd_oracle import head_vectors, top_pair

    lens = Lens(None, Path(args.data))
    lib = load_file(args.functions)
    head_vectors(lib)  # the functions' U and V also from readout's earlier write and gate names
    vocab = torch.nonzero(lens.seen).reshape(-1)
    f = lens.freq[vocab]
    Ef = lens.Ef[vocab]
    k = args.k
    out = {}

    def ranked(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # [R, T] -> top positive, top negative token ids
        return vocab[(a * f).topk(k, dim=1).indices].to(torch.int32), vocab[(-a * f).topk(k, dim=1).indices].to(torch.int32)

    def written(r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # [R, 768] write directions -> raised, lowered
        w = r @ Ef.T
        w = w - (w * f).sum(1, keepdim=True) / f.sum()
        return ranked(w)

    for layer in range(lens.t.n_layer):
        n = f"h.{layer}.mlp.function"
        U, V = lib[f"{n}.U"].float(), lib[f"{n}.V"].float()
        bias = lib[f"{n}.bias"].float() if f"{n}.bias" in lib else torch.zeros(U.shape[0])
        x = lens.Er[vocab] * lens.t.norms[2 * layer + 1].float()
        parts = {key: [] for key in ("pos_read", "neg_read", "up", "down")}
        for s0 in range(0, U.shape[0], 512):
            rows = torch.arange(s0, min(U.shape[0], s0 + 512))
            for key, val in zip(("pos_read", "neg_read"), ranked((x @ V[:, rows] + bias[rows]).T)):
                parts[key].append(val)
            for key, val in zip(("up", "down"), written(U[rows])):
                parts[key].append(val)
        out.update({f"{n}.{key}": torch.cat(v) for key, v in parts.items()})
        n = f"h.{layer}.attn.head"
        Q, K, Vv, O = (lib[f"{n}.{x_}"].float() for x_ in ("query", "key", "value", "output"))
        x = lens.Er[vocab] * lens.t.norms[2 * layer].float()
        ov = [top_pair(O[h] @ Vv[h]) for h in range(Q.shape[0])]
        qk = [top_pair(Q[h].T @ K[h]) for h in range(Q.shape[0])]
        out[f"{n}.pos_read"], out[f"{n}.neg_read"] = ranked((x @ torch.stack([r for _, _, r in ov], 1)).T)
        out[f"{n}.up"], out[f"{n}.down"] = written(torch.stack([left for _, left, _ in ov]))
        out[f"{n}.attn_from"], _ = ranked((x @ torch.stack([left for _, left, _ in qk], 1)).T)
        out[f"{n}.attn_to"], _ = ranked((x @ torch.stack([r for _, _, r in qk], 1)).T)
        print(json.dumps({"layer": layer, "functions": U.shape[0], "heads": Q.shape[0]}), flush=True)
    save_file(out, args.out)


def fit_beta(s: np.ndarray, y: np.ndarray) -> float:
    """Maximum-likelihood beta of q(y = 1) = sigmoid(beta s), by Newton's method from zero."""
    beta = 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-beta * s))
        grad = np.sum((y - p) * s)
        hess = np.sum(p * (1 - p) * s * s) + 1e-12
        beta += grad / hess
    return float(beta)


@torch.no_grad()
def baseline(args):
    from vpd_oracle import Table, examples

    lens = Lens(Path(args.uv), Path(args.data))
    vocab = torch.nonzero(lens.seen).reshape(-1)
    table = Table(Path(args.labels), Path(args.uv), Path(args.relations))
    config = json.loads((Path(args.nothing) / "config.json").read_text())
    held = {int(x) for x in config["heldout_layers"].split(",") if x}
    nothing = [json.loads(line) for line in open(Path(args.nothing) / f"eval_{Path(args.labels).name}.jsonl")]
    rows, at = [], 0
    for split, layers in (("heldout_layers", held), ("trained_layers", {0, 1, 2, 3} - held)):
        for distribution in ("natural", "stratified"):
            data = examples(table, layers, config.get("eval_examples", args.examples), 1, stratified=distribution == "stratified")
            for ex in data:
                ref = nothing[at]
                at += 1
                assert (ref["question"], ref["context"], ref["c"]) == (ex["kind_q"], ex["context"], ex["c"]), "question order differs from the evaluation"
                if ex["kind_q"] not in ("direction", "top"):
                    continue
                layer, kind, c = ex["layer"], ex["kind"], ex["c"]
                rowsel = torch.tensor([c])
                w = lens.write(layer, kind, rowsel)
                if w is None:
                    score = measured = np.zeros(len(ex["option_ids"]))
                else:
                    w = w[0]
                    wv = w[vocab]
                    w = (w - wv.mean()) / wv.std()
                    tok = table.tokens[ex["context"], ex["position"]]
                    a_all = lens.read_vocab(layer, kind, rowsel, vocab)[0]
                    a = float(lens.read(layer, kind, rowsel, tok.reshape(1))[0, 0]) / float(a_all.std())
                    sign = -1.0 if ex["edit"] == "ablate" else 0.5
                    score = sign * a * w[torch.tensor(ex["option_ids"])].numpy()
                    # The same write lens with the measured activity there (a label: a decomposition of the bound, never an input).
                    a_m = float(table.sites[(layer, kind)][1]["activity"][c, ex["j"], ex["position"]])
                    measured = sign * np.sign(a_m) * w[torch.tensor(ex["option_ids"])].numpy()
                rows.append(
                    {
                        "split": split,
                        "distribution": distribution,
                        "question": ex["kind_q"],
                        "answer": ex["answer"],
                        "score": score.tolist(),
                        "measured": measured.tolist(),
                        "nothing": ref["log_score"],
                        "stratum": ex["stratum"],
                        "kind": kind,
                    }
                )
    summary = {}
    for variant in ("score", "measured"):
        for r in rows:
            r["s"] = r[variant]
        for q in ("direction", "top"):
            train = [dict(r, score=r["s"]) for r in rows if r["question"] == q and r["split"] == "trained_layers"]
            if q == "direction":
                s = np.array([r["score"][0] for r in train])
                y = np.array([1.0 if r["answer"] == 0 else 0.0 for r in train])
                beta = fit_beta(s, y)
            else:
                # beta of a softmax over the options, by a coarse-to-fine line search of the log likelihood
                def ll(b):
                    return sum(
                        float(
                            np.log(
                                np.exp(b * np.array(r["score"]) - np.max(b * np.array(r["score"])))[r["answer"]]
                                / np.exp(b * np.array(r["score"]) - np.max(b * np.array(r["score"]))).sum()
                            )
                        )
                        for r in train
                    )

                grid = np.concatenate([[0.0], np.geomspace(1e-3, 1e2, 101)])
                beta = float(grid[int(np.argmax([ll(b) for b in grid]))])
            for split in ("heldout_layers", "trained_layers"):
                for distribution in ("natural", "stratified"):
                    sel = [dict(r, score=r["s"]) for r in rows if r["question"] == q and r["split"] == split and r["distribution"] == distribution]
                    if not sel:
                        continue
                    lens_ls, poe_ls, acc = [], [], []
                    for r in sel:
                        z = beta * np.array(r["score"]) if q == "top" else np.array([beta * r["score"][0], -beta * r["score"][0]]) / 2
                        lq = z - np.log(np.exp(z - z.max()).sum()) - z.max()
                        lens_ls.append(float(lq[r["answer"]]))
                        acc.append(float(np.argmax(z) == r["answer"]) if np.ptp(z) > 0 else 1.0 / len(z))
                        if q == "direction":  # product of experts with the no-input reader (binary: its full distribution is known)
                            pn = math.exp(r["nothing"])
                            ln_n = np.array([math.log(pn), math.log(max(1 - pn, 1e-12))]) if r["answer"] == 0 else np.array([math.log(max(1 - pn, 1e-12)), math.log(pn)])
                            zz = z + ln_n
                            poe_ls.append(float((zz - np.log(np.exp(zz - zz.max()).sum()) - zz.max())[r["answer"]]))
                    key = ("" if variant == "score" else "with_measured_activity_sign/") + f"{split}/{q}" + ("" if distribution == "natural" else "/stratified")
                    summary[key] = {
                        "questions": len(sel),
                        "beta": beta,
                        "lens_log_score": float(np.mean(lens_ls)),
                        "lens_accuracy": float(np.mean(acc)),
                        "nothing_log_score": float(np.mean([r["nothing"] for r in sel])),
                        "chance_log_score": float(-math.log(len(sel[0]["score"]) if q == "top" else 2)),
                        **({"product_of_experts_log_score": float(np.mean(poe_ls))} if poe_ls else {}),
                    }
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--uv", help="VPD's subcomponents (export_uv)")
    b.add_argument("--functions", help="a library's functions.safetensors (readout's oracle input), in place of --uv")
    b.add_argument("--data", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--k", type=int, default=8)
    s = sub.add_parser("baseline")
    for a in ("--labels", "--relations", "--uv", "--data", "--nothing", "--out"):
        s.add_argument(a, required=True)
    s.add_argument("--examples", type=int, default=3000)
    v = sub.add_parser("values")
    for a in ("--lens", "--uv", "--data"):
        v.add_argument(a, required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    if args.command == "build" and args.functions:
        build_library(args)
    else:
        {"build": build, "baseline": baseline, "values": add_values}[args.command](args)


if __name__ == "__main__":
    main()

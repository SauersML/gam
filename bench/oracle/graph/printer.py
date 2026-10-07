"""The printer (#2951 graph oracle): a program (mech source, its IR, or a search result) -> clean mech source
whose comments and docstring state measured facts about each node, plus a small graph description.

Facts, measured on the behavior's clean prompts at its target tokens (the answer = the next token in the
behavior file), by forward passes of the model itself:
  removal  the node's pieces removed together (heads: their attention output zeroed; neurons: their
           activations zeroed; VPD subcomponents: u_i v_i^T subtracted at their site; transcoder
           features: a_i W_dec[i] subtracted from their MLP's output): the mean change of the answer's
           log-probability and KL(M || M_removed), in bits per target;
  direct   the node's own write at the target position read straight through the final norm and the
           unembedding (direct path only, no later layers): the answer's logit relative to the mean
           logit, in nats, and the tokens the write promotes most often (each prompt's top 5, counted).
These are the docstrings' only claims; the edges are the program's and the checker tests them.

  printer.py PROGRAM (.py | .json IR | .json search result with "source") BEHAVIOR.json
             [--name NAME] [--out-dir DIR] [--score SCORE.json]
writes DIR/NAME.py (the program) and DIR/NAME.graph.json (nodes with their facts and one-line roles,
edges, score if given). printed(ir, behavior) is the same as a function.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # bench/oracle (vpd_labels)

import mech  # noqa: E402


# ---------------------------------------------------------------------------------------------------
# Models: one forward with removals and recorded writes


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class Vpd4l:
    def __init__(self):
        import vpd_labels as VL
        import vpd_model as VM

        self.VM, self.dev = VM, device()
        self.t = VL.Model(self.dev).t
        self.uv = None
        self.tokenizer = None

    def factors(self, site: str):
        if self.uv is None:
            import vpd_labels as VL

            self.uv = VL.load_uv(self.dev, Path.home() / "mpd-data/oracle/vpd/uv.safetensors")
        return self.uv[site]  # U [C, d_out], V [d_in, C]

    def forward(self, ids, rows, cols, remove: list[dict] = (), record: bool = False):
        """Log-probabilities at (rows, cols) with `remove` pieces removed; with `record`, also the
        activations a write needs ({("attn", l): y, ("mlp", l): hidden, "final": residual})."""
        t, VM = self.t, self.VM
        H, D = t.n_head, t.hd
        B, L = ids.shape
        seen = {}
        x = t.wte[ids]
        for i in range(t.n_layer):
            def site(kind, h, i=i):
                name = f"h.{i}.{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}.{kind}"
                out = t.site(name)(h)
                for p in remove:
                    if p["view"] == "vpd" and p["layer"] == i and p["kind"] == kind:
                        U, V = self.factors(name)
                        idx = torch.tensor(p["index"], device=self.dev)
                        out = out - (h @ V[:, idx]) @ U[idx]
                return out

            h = VM.rms(x, t.norms[2 * i], t.eps)
            q = site("q_proj", h).view(B, L, H, D).transpose(1, 2)
            k = site("k_proj", h).view(B, L, H, D).transpose(1, 2)
            v = site("v_proj", h).view(B, L, H, D).transpose(1, 2)
            q, k = t._rope(q, L), t._rope(k, L)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, L, -1)
            for p in remove:
                if p["view"] == "native" and p["kind"] == "head" and p["layer"] == i:
                    y = y.clone()
                    for hh in p["index"]:
                        y[..., hh * D : (hh + 1) * D] = 0
            if record:
                seen[("attn", i)] = y
            x = x + site("o_proj", y)
            hid = VM.gelu_tanh(site("c_fc", VM.rms(x, t.norms[2 * i + 1], t.eps)))
            for p in remove:
                if p["view"] == "native" and p["kind"] == "mlp" and p["layer"] == i:
                    hid = hid.clone()
                    hid[..., p["index"]] = 0
            if record:
                seen[("mlp", i)] = hid
            x = x + site("down_proj", hid)
        if record:
            seen["final"] = x[rows, cols]
        lp = torch.log_softmax((VM.rms(x, t.ln_f, t.eps)[rows, cols] @ t.wte.T).float(), -1)
        return lp, seen

    def write(self, piece: dict, seen: dict, rows, cols):
        """The piece's write to the residual stream at (rows, cols), or None (it writes an internal
        stream: q/k/v_proj, c_fc)."""
        t, l = self.t, piece["layer"]
        D = t.hd
        if piece["view"] == "native" and piece["kind"] == "head":
            y = seen[("attn", l)][rows, cols]
            W = t.site(f"h.{l}.attn.o_proj").weight  # [d_model, d_model]
            return sum(y[:, h * D : (h + 1) * D] @ W[:, h * D : (h + 1) * D].T for h in piece["index"])
        if piece["view"] == "native" and piece["kind"] == "mlp":
            hid = seen[("mlp", l)][rows, cols][:, piece["index"]]
            return hid @ t.site(f"h.{l}.mlp.down_proj").weight[:, piece["index"]].T
        if piece["view"] == "vpd" and piece["kind"] in ("o_proj", "down_proj"):
            src = seen[("attn", l)] if piece["kind"] == "o_proj" else seen[("mlp", l)]
            U, V = self.factors(f"h.{l}.{'attn' if piece['kind'] == 'o_proj' else 'mlp'}.{piece['kind']}")
            idx = torch.tensor(piece["index"], device=self.dev)
            return (src[rows, cols] @ V[:, idx]) @ U[idx]
        return None

    def direct(self, write, seen):
        """Direct-path logits of a write: through the final norm at the clean residual's scale."""
        t = self.t
        final = seen["final"]
        scale = torch.rsqrt(final.pow(2).mean(-1, keepdim=True) + t.eps)
        return ((write * scale * t.ln_f) @ t.wte.T).float()

    def decode(self, token: int) -> str:
        if self.tokenizer is None:
            import prompt

            self.tokenizer = prompt.tokenizer("vpd4l")
        return self.tokenizer.decode([token])


class Qwen3:
    def __init__(self, model: str):
        from transformers import AutoModelForCausalLM

        self.name, self.dev = model, device()
        self.m = AutoModelForCausalLM.from_pretrained(mech.QWEN3[model], dtype=torch.float32).to(self.dev).eval()
        self.cfg = self.m.config
        self.tc = {}
        self.tokenizer = None
        self.state = {"remove": [], "seen": None, "need": set()}
        D = self.cfg.head_dim
        for l, block in enumerate(self.m.model.layers):
            def pre_o(module, args, l=l):
                y = args[0]
                heads = [h for p in self.state["remove"] if p["view"] == "native" and p["kind"] == "head" and p["layer"] == l
                         for h in p["index"]]
                if heads:
                    y = y.clone()
                    for h in heads:
                        y[..., h * D : (h + 1) * D] = 0
                if self.state["seen"] is not None and l in self.state["need"]:
                    self.state["seen"][("attn", l)] = y
                return (y,)

            def pre_down(module, args, l=l):
                hid = args[0]
                idx = [i for p in self.state["remove"] if p["view"] == "native" and p["kind"] == "mlp" and p["layer"] == l
                       for i in p["index"]]
                if idx:
                    hid = hid.clone()
                    hid[..., idx] = 0
                if self.state["seen"] is not None and l in self.state["need"]:
                    self.state["seen"][("mlp", l)] = hid
                return (hid,)

            def post_mlp(module, args, out, l=l):
                feats = [p for p in self.state["remove"] if p["view"] == "transcoder" and p["layer"] == l]
                if self.state["seen"] is not None and l in self.state["need"]:
                    self.state["seen"][("mlp_in", l)] = args[0]
                if feats:
                    enc, b, dec = self.transcoder(l)
                    idx = torch.tensor([i for p in feats for i in p["index"]], device=self.dev)
                    a = torch.relu(args[0] @ enc[idx].T + b[idx])
                    return out - a @ dec[idx]

            block.self_attn.o_proj.register_forward_pre_hook(pre_o)
            block.mlp.down_proj.register_forward_pre_hook(pre_down)
            block.mlp.register_forward_hook(post_mlp)

        def pre_norm(module, args):
            if self.state["seen"] is not None:
                self.state["seen"]["final_full"] = args[0]

        self.m.model.norm.register_forward_pre_hook(pre_norm)

    def transcoder(self, l: int):
        if l not in self.tc:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            local = Path.home() / f"mpd-data/transcoders/qwen3-0.6b-lowl0/layer_{l}.safetensors"
            path = local if local.exists() else Path(hf_hub_download("mwhanna/qwen3-0.6b-transcoders-lowl0", f"layer_{l}.safetensors"))
            w = load_file(str(path))
            self.tc[l] = tuple(w[k].to(self.dev, torch.float32) for k in ("W_enc", "b_enc", "W_dec"))
        return self.tc[l]

    def forward(self, ids, rows, cols, remove: list[dict] = (), record: bool = False):
        self.state.update(remove=list(remove), seen={} if record else None)
        hidden = self.m.model(ids).last_hidden_state
        seen = self.state["seen"]
        self.state.update(remove=[], seen=None)
        if record:
            seen["final"] = seen.pop("final_full")[rows, cols]
        return torch.log_softmax(self.m.lm_head(hidden[rows, cols]).float(), -1), seen

    def write(self, piece: dict, seen: dict, rows, cols):
        l, D, layers = piece["layer"], self.cfg.head_dim, self.m.model.layers
        if piece["view"] == "native" and piece["kind"] == "head":
            y = seen[("attn", l)][rows, cols]
            W = layers[l].self_attn.o_proj.weight
            return sum(y[:, h * D : (h + 1) * D] @ W[:, h * D : (h + 1) * D].T for h in piece["index"])
        if piece["view"] == "native" and piece["kind"] == "mlp":
            hid = seen[("mlp", l)][rows, cols][:, piece["index"]]
            return hid @ layers[l].mlp.down_proj.weight[:, piece["index"]].T
        if piece["view"] == "transcoder":
            enc, b, dec = self.transcoder(l)
            idx = torch.tensor(piece["index"], device=self.dev)
            return torch.relu(seen[("mlp_in", l)][rows, cols] @ enc[idx].T + b[idx]) @ dec[idx]
        return None

    def direct(self, write, seen):
        """Direct-path logits: the write scaled as the final norm scales the clean residual there."""
        norm = self.m.model.norm
        scale = torch.rsqrt(seen["final"].pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
        return ((write * scale * norm.weight) @ self.m.lm_head.weight.T).float()

    def decode(self, token: int) -> str:
        if self.tokenizer is None:
            import prompt

            self.tokenizer = prompt.tokenizer(self.name)
        return self.tokenizer.decode([token])


# ---------------------------------------------------------------------------------------------------
# Facts and printing


def expanded(piece: dict) -> dict:
    """The piece with its index as a list (null = every unit of the site)."""
    p = dict(piece)
    if p["index"] is None:
        shape = SHAPE[0]
        size = {"head": shape["heads"], "mlp": shape["d_mlp"]}.get(p["kind"])
        p["index"] = list(range(size)) if size else []
    elif isinstance(p["index"], int):
        p["index"] = [p["index"]]
    return p


SHAPE: list[dict] = [{}]


def facts(engine, ir: dict, behavior: dict, chunk: int = 64) -> dict[str, dict]:
    """Per node id: removal and direct-path facts, averaged over the behavior's target tokens."""
    prompts = behavior["prompts"]
    dev = engine.dev
    if isinstance(engine, Qwen3):  # record only the layers the program's pieces sit in
        engine.state["need"] = {p["layer"] for n in ir["nodes"] for p in n["pieces"]}
        chunk = min(chunk, 32)
    sums: dict[str, dict] = {n["id"]: {"answer_bits": 0.0, "kl_bits": 0.0, "answer_logit": 0.0, "ranks": [],
                                        "promoted": Counter(), "writes": True} for n in ir["nodes"]}
    count = 0
    for s in range(0, len(prompts), chunk):
        part = prompts[s : s + chunk]
        T = max(len(p["token_ids"]) for p in part)
        ids = torch.zeros(len(part), T, dtype=torch.long)
        for i, p in enumerate(part):
            ids[i, : len(p["token_ids"])] = torch.tensor(p["token_ids"])
        ids = ids.to(dev)
        rows = torch.tensor([i for i, p in enumerate(part) for _ in p["target_positions"]], device=dev)
        cols = torch.tensor([t for p in part for t in p["target_positions"]], device=dev)
        answer = ids[rows, cols + 1]
        with torch.no_grad():
            lp, seen = engine.forward(ids, rows, cols, record=True)
            p0 = lp.exp()
            gold = lp.gather(-1, answer[:, None])[:, 0]
            for n in ir["nodes"]:
                pieces = [expanded(p) for p in n["pieces"]]
                lpr, _ = engine.forward(ids, rows, cols, remove=pieces)
                f = sums[n["id"]]
                f["answer_bits"] += ((lpr.gather(-1, answer[:, None])[:, 0] - gold).sum() / math.log(2)).item()
                f["kl_bits"] += ((p0 * (lp - lpr)).sum(-1).sum() / math.log(2)).item()
                writes = [w for w in (engine.write(p, seen, rows, cols) for p in pieces) if w is not None]
                if not writes:
                    f["writes"] = False
                    continue
                logits = engine.direct(sum(writes), seen)
                rel = logits - logits.mean(-1, keepdim=True)
                f["answer_logit"] += rel.gather(-1, answer[:, None]).sum().item()
                f["ranks"] += (logits > logits.gather(-1, answer[:, None])).sum(-1).tolist()
                for row in logits.topk(5, -1).indices.tolist():
                    f["promoted"].update(row)
        count += len(rows)
    out = {}
    for nid, f in sums.items():
        top = [engine.decode(t) for t, _ in f["promoted"].most_common(5)] if f["writes"] else []
        out[nid] = {"removal_answer_bits": f["answer_bits"] / count, "removal_kl_bits": f["kl_bits"] / count,
                    "direct_answer_logit": f["answer_logit"] / count if f["writes"] else None,
                    "direct_answer_rank_median": sorted(f["ranks"])[len(f["ranks"]) // 2] + 1 if f["ranks"] else None,
                    "direct_promotes": top, "targets": count}
    return out


def address(p: dict) -> str:
    idx = p["index"]
    if p["view"] == "native":
        site = f"L[{p['layer']}].{p['kind']}"
    elif p["view"] == "vpd":
        site = f"PD.vpd[{p['layer']}].{p['kind']}"
    elif p["view"] == "library":
        site = f"PD.lib[{p['layer']}].{p['kind']}"
    else:
        site = f"PD.tc[{p['layer']}]"
    if idx is None:
        return site
    return f"{site}[{', '.join(map(str, idx if isinstance(idx, list) else [idx]))}]"


def role(f: dict) -> str:
    """One line of plain facts about a node."""
    d = f["removal_answer_bits"]
    line = (f"removing it {'lowers' if d < 0 else 'raises'} the answer's log-probability by {abs(d):.2f} bits per "
            f"target (KL {f['removal_kl_bits']:.2f})")
    if f["direct_answer_logit"] is not None:
        z = f["direct_answer_logit"]
        line += (f"; its direct write {'raises' if z > 0 else 'lowers'} the answer's logit by {abs(z):.2f} "
                 f"(median rank {f['direct_answer_rank_median']})")
        if f["direct_promotes"]:
            line += " and most promotes " + ", ".join(repr(t) for t in f["direct_promotes"][:3])
    else:
        line += "; it writes no residual stream of its own (it feeds its layer's attention or MLP)"
    return line


def source_of(ir: dict, behavior: dict, facts_of: dict[str, dict], score: dict | None = None) -> str:
    """Clean mech source for `ir` with measured-fact comments and docstring."""
    head = (f"Behavior {behavior['id']} ({ir['model']}): {behavior['description']}")
    lines = textwrap.wrap(head, 100)
    lines += ["", *textwrap.wrap(
        f"Facts measured on the behavior's {next(iter(facts_of.values()))['targets'] if facts_of else 0} target tokens "
        f"(clean prompts; the answer is the next token): what removing each node does to the answer, and what its "
        f"own write does to the logits through the direct path only (no later layers). The edges are this "
        f"program's claim; the checker tests them.", 100)]
    if score:
        lines += ["", *textwrap.wrap(
            f"Score: {score['total_bits']:.4g} bits in total (execution error {score['exec_error_bits']:.4g}, "
            f"opaque numbers {score.get('opaque_bits', 0):.4g}, code {score.get('code_bits', 0):.4g}).", 100)]
    used = {"node"} | ({"edges"} if ir["edges"] else set())
    views = {p["view"] for n in ir["nodes"] for p in n["pieces"]}
    used |= ({"L"} if "native" in views else set()) | ({"PD"} if views & {"vpd", "library", "transcoder"} else set())
    used |= {e["from"] for e in ir["edges"] if e["from"] == "embed"} | {e["to"] for e in ir["edges"] if e["to"] == "logits"}
    order = [x for x in ("node", "edges", "L", "PD", "embed", "logits") if x in used]
    out = ['"""' + "\n".join(lines) + '\n"""', f"from mech import {', '.join(order)}", ""]
    for n in ir["nodes"]:
        f = facts_of.get(n["id"])
        if f:
            out += [f"# {row}" for row in textwrap.wrap(role(f), 98)]
        call = f"{n['id']} = node({', '.join(address(p) for p in n['pieces'])})"
        if len(call) > 100:
            body = textwrap.wrap(", ".join(address(p) for p in n["pieces"]), 96, break_long_words=False)
            call = f"{n['id']} = node(\n" + "\n".join(f"    {row}" for row in body) + "\n)"
        out.append(call)
    if ir["edges"]:
        out += ["", "edges("]
        out += [f"    {e['from']} >> {e['to']}{'' if e['route'] == 'input' or e['to'] == 'logits' else '.' + e['route']},"
                for e in ir["edges"]]
        out.append(")")
    return "\n".join(out) + "\n"


def graph_of(ir: dict, behavior: dict, facts_of: dict[str, dict], score: dict | None = None) -> dict:
    """A small graph description for rendering: nodes (address, facts, one-line role) and edges."""
    return {"behavior": behavior["id"], "model": ir["model"], "description": behavior["description"],
            "nodes": [{"id": n["id"], "address": ", ".join(address(p) for p in n["pieces"]),
                       "pieces": n["pieces"], "role": role(facts_of[n["id"]]) if n["id"] in facts_of else None,
                       **facts_of.get(n["id"], {})} for n in ir["nodes"]],
            "edges": ir["edges"], "score": score}


ENGINES: dict[str, object] = {}


def engine_for(model: str):
    if model not in ENGINES:
        ENGINES[model] = Vpd4l() if model == "vpd4l" else Qwen3(model)
    return ENGINES[model]


def wrong(measured: dict[str, dict]) -> dict[str, dict]:
    """The same facts on the wrong nodes, for the reader's control (R3): rotated by one node, or with
    their signs flipped when the program has one node."""
    ids = list(measured)
    if len(ids) > 1:
        return {nid: measured[ids[(k + 1) % len(ids)]] for k, nid in enumerate(ids)}
    flip = lambda v: -v if isinstance(v, float) else v  # noqa: E731
    return {nid: {k: flip(v) if k in ("removal_answer_bits", "direct_answer_logit") else v for k, v in f.items()}
            for nid, f in measured.items()}


def printed(ir: dict, behavior: dict, score: dict | None = None, measured: dict | None = None) -> tuple[str, dict]:
    """(source, graph description) of a traced program on a behavior (`measured`: facts already taken,
    e.g. wrong(facts) for the reader's control)."""
    SHAPE[0] = mech.shapes(ir["model"])
    measured = measured if measured is not None else facts(engine_for(ir["model"]), ir, behavior)
    src = source_of(ir, behavior, measured, score)
    check = mech.trace_inline(src, ir["model"])
    if not check["valid"] or (check["nodes"], check["edges"]) != (ir["nodes"], ir["edges"]):
        raise ValueError(f"the printed program does not trace back to the same IR: {check['error']}")
    return src, graph_of(ir, behavior, measured, score)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("program", type=Path)
    ap.add_argument("behavior", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "mpd-data/graph_oracle/printed")
    ap.add_argument("--score", type=Path, help="the program's score JSON (score.py), stated in the docstring")
    ap.add_argument("--wrong", action="store_true", help="also write NAME.wrong.py: the facts on the wrong nodes (R3)")
    a = ap.parse_args()
    behavior = json.loads(a.behavior.read_text())
    text = a.program.read_text()
    if a.program.suffix == ".json":
        data = json.loads(text)
        ir = data if "nodes" in data else mech.trace_inline(data["source"], behavior["model"])
    else:
        ir = mech.trace_inline(text, behavior["model"])
    if not ir["valid"]:
        sys.exit(f"invalid program: {ir['error']}")
    score = json.loads(a.score.read_text()) if a.score else None
    SHAPE[0] = mech.shapes(ir["model"])
    measured = facts(engine_for(ir["model"]), ir, behavior)
    src, graph = printed(ir, behavior, score, measured)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    name = a.name or f"{behavior['id']}.{a.program.stem}"
    (a.out_dir / f"{name}.py").write_text(src)
    if a.wrong:
        (a.out_dir / f"{name}.wrong.py").write_text(printed(ir, behavior, score, wrong(measured))[0])
    (a.out_dir / f"{name}.graph.json").write_text(json.dumps(graph, indent=1))
    print(src)


if __name__ == "__main__":
    main()

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
writes DIR/NAME.py (the program, facts in comments), DIR/NAME.answer.txt (the oracle's format: the program
block, then its plain-English explanation from the facts in words and the edges) and DIR/NAME.graph.json
(nodes with their facts and one-line roles, edges, explanation, score if given). printed(ir, behavior) is the same as a function.
"""

from __future__ import annotations

import argparse
import ast
import inspect
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

    def rest_t(self, site: str):
        """The site's remainder, transposed as the forward applies it: W^T - V U (out = h @ W^T)."""
        U, V = self.factors(site)
        return self.t.site(site).W.T - V @ U

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
                        if p["index"] == "rest":
                            out = out - h @ self.rest_t(name)
                            continue
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
            W = t.site(f"h.{l}.attn.o_proj").W  # [d_model, d_model], out = x @ W.T
            return sum(y[:, h * D : (h + 1) * D] @ W[:, h * D : (h + 1) * D].T for h in piece["index"])
        if piece["view"] == "native" and piece["kind"] == "mlp":
            hid = seen[("mlp", l)][rows, cols][:, piece["index"]]
            return hid @ t.site(f"h.{l}.mlp.down_proj").W[:, piece["index"]].T
        if piece["view"] == "vpd" and piece["kind"] in ("o_proj", "down_proj"):
            src = seen[("attn", l)] if piece["kind"] == "o_proj" else seen[("mlp", l)]
            name = f"h.{l}.{'attn' if piece['kind'] == 'o_proj' else 'mlp'}.{piece['kind']}"
            if piece["index"] == "rest":
                return src[rows, cols] @ self.rest_t(name)
            U, V = self.factors(name)
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
            self.tokenizer = mech.tokenizer("vpd4l")
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
            self.tokenizer = mech.tokenizer(self.name)
        return self.tokenizer.decode([token])


# ---------------------------------------------------------------------------------------------------
# Facts and printing


def expanded(piece: dict) -> dict:
    """The piece with its index as a list (null = every unit of the site)."""
    p = dict(piece)
    if p["index"] == "rest":
        return p
    if p["index"] is None:
        shape = SHAPE[0]
        size = {"head": shape["heads"], "mlp": shape["d_mlp"]}.get(p["kind"])
        p["index"] = list(range(size)) if size else []
    elif isinstance(p["index"], int):
        p["index"] = [p["index"]]
    return p


SHAPE: list[dict] = [{}]
LIBRARY: dict[tuple[int, str], list[list[tuple[str, int]]]] = {}
KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")


def library_pieces(piece: dict) -> list[dict]:
    """A library piece as the VPD subcomponents its parts are made of (decomp's start, arm
    mech.LIBRARY_ARM: part i of layer l's attention or MLP = the i-th component of that block in the file;
    a component's slices are (site = layer x 6 + kind, subcomponent))."""
    if not LIBRARY:
        start = Path.home() / "mpd-data/decomp/start.components.json"
        arm = next(r for r in json.loads(start.read_text()) if r["arm"] == mech.LIBRARY_ARM)
        for c in arm["components"]:
            site = c["read"]["own"][0] if "own" in c["read"] else c["read"]["direction"]["site"]
            key = (site // len(KINDS), "attn" if site % len(KINDS) < 4 else "mlp")
            LIBRARY.setdefault(key, []).append([(KINDS[s % len(KINDS)], k) for s, k in c["slices"]])
    by: dict[str, set] = {}
    for i in piece["index"]:
        for kind, k in LIBRARY[(piece["layer"], piece["kind"])][i]:
            by.setdefault(kind, set()).add(k)
    return [{"view": "vpd", "layer": piece["layer"], "kind": kind, "index": sorted(v)} for kind, v in by.items()]


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
                pieces = [q for p in n["pieces"] for q in
                          (library_pieces(expanded(p)) if p["view"] == "library" else [expanded(p)])]
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
    """A piece as the oracle writes it: its part tokens (<p:2.v.559>, <p:1.q.rest>, <p:3.h.4>, <p:0.a>,
    <p:1.m>), or in text where no part tokens exist (a whole decomposition site, native neurons)."""
    idx, l = p["index"], p["layer"]
    site = "mlp" if p["kind"] == "feature" else p["kind"]
    if idx == "rest":
        return f"<p:{l}.{mech.CODES[site]}.rest>"
    units = idx if isinstance(idx, list) else [idx]
    if p["view"] == "native":
        if idx is None:
            return f"<p:{l}.{'a' if p['kind'] == 'head' else 'm'}>"
        if p["kind"] == "head":
            return ", ".join(f"<p:{l}.h.{h}>" for h in units)
        return f"L[{l}].mlp[{', '.join(map(str, units))}]"
    if idx is None:
        return f"PD[{l}].{site}"
    return ", ".join(f"<p:{l}.{mech.CODES[site]}.{i}>" for i in units)


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
    if ir.get("alignments") or any(n.get("claim") or n.get("rule") for n in ir["nodes"]):
        raise ValueError("the printer prints node-and-edge programs; this one has alignments or claims")
    used = {"node"} | ({"edges"} if ir["edges"] else set())
    texts = [address(p) for n in ir["nodes"] for p in n["pieces"]]
    used |= ({"L"} if any(t.startswith("L[") for t in texts) else set()) | ({"PD"} if any(t.startswith("PD[") for t in texts) else set())
    used |= {e["from"] for e in ir["edges"] if e["from"] == "embed"} | {e["to"] for e in ir["edges"] if e["to"] == "logits"}
    order = [x for x in ("node", "edges", "L", "PD", "embed", "logits") if x in used]
    out = ['"""' + "\n".join(lines) + '\n"""', f"from mech import {', '.join(order)}", ""]
    for n in ir["nodes"]:
        f = facts_of.get(n["id"])
        if f:
            out += [f"# {row}" for row in textwrap.wrap(role(f), 98)]
        args = [address(p) for p in n["pieces"]]
        call = f"{n['id']} = node({', '.join(args)})"
        if len(call) > 100:
            body = textwrap.wrap(", ".join(args), 96, break_long_words=False)
            call = f"{n['id']} = node(\n" + "\n".join(f"    {row}" for row in body) + "\n)"
        out.append(call)
    if ir["edges"]:
        out += ["", "edges("]
        out += [f"    {e['from']} >> {e['to']}{'' if e['route'] == 'input' or e['to'] == 'logits' else '.' + e['route']},"
                for e in ir["edges"]]
        out.append(")")
    return "\n".join(out) + "\n"


SITE_WORDS = {"q_proj": "query", "k_proj": "key", "v_proj": "value", "o_proj": "output", "c_fc": "input",
              "down_proj": "output"}


def _listed(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def part_words(n: dict) -> str:
    """A node's parts in words, per layer block: "layer 1's attention query subcomponent 316 and key
    subcomponent 329", "60 input and 62 output subcomponents of layer 0's MLP", "heads L2.H3 and L2.H4"."""
    groups: dict[tuple, list[dict]] = {}
    for p in n["pieces"]:
        groups.setdefault((p["layer"], "mlp" if p["kind"] in ("mlp", "c_fc", "down_proj", "feature") else "attn"), []).append(p)
    out, heads = [], []
    for (l, block), pieces in sorted(groups.items()):
        where = f"layer {l}'s {'MLP' if block == 'mlp' else 'attention'}"
        items, few = [], True
        for p in pieces:
            idx = p["index"]
            units = None if idx in (None, "rest") else (idx if isinstance(idx, list) else [idx])
            if p["view"] == "native" and p["kind"] == "head":
                if units is None:
                    out.append(f"every head of layer {l}")
                else:
                    heads += [f"L{l}.H{h}" for h in units]
                continue
            if p["view"] == "native":
                out.append(f"{where}" if units is None else f"{len(units)} neurons of {where}")
                continue
            noun = {"vpd": "subcomponent", "transcoder": "transcoder feature", "library": "part"}[p["view"]]
            word = SITE_WORDS.get(p["kind"], "")
            if idx == "rest":
                items.append(f"{word} remainder")
            elif units is None:
                items.append(f"every {word} {noun}".replace("  ", " "))
            elif len(units) <= 3:
                items.append(f"{word} {noun}{'s' if len(units) > 1 else ''} {_listed([str(u) for u in units])}".strip())
            else:
                few = False
                items.append(f"{len(units)} {word} {noun}s".replace("  ", " "))
        if items:
            out.append(f"{where} {_listed(items)}" if few else f"{_listed(items)} of {where}")
    if heads:
        out.insert(0, ("head " if len(heads) == 1 else "heads ") + _listed(heads))
    return "; ".join(out)


def explanation_of(ir: dict, behavior: dict, facts_of: dict[str, dict]) -> str:
    """The program's plain-English explanation (what the reader reads): what each part does, from the
    measured facts in words, and how the parts connect, from the declared edges."""
    name = {n["id"]: part_words(n) for n in ir["nodes"]}
    lines = [behavior["description"].rstrip(".") + "."]
    for n in ir["nodes"]:
        f = facts_of.get(n["id"])
        words = name[n["id"]]
        plural = "," in words or ";" in words or any(w in words for w in ("neurons", "subcomponents", "features", "parts"))
        subject = words[0].upper() + words[1:]
        if f:
            d, z, rank = f["removal_answer_bits"], f["direct_answer_logit"], f["direct_answer_rank_median"]
            be, matter, work = ("are", "matter", "work") if plural else ("is", "matters", "works")
            weight = (f"{be} essential" if d <= -2 else matter if d <= -0.5 else f"{matter} a little" if d < -0.1
                      else f"{work} against the answer" if d > 0.1 else f"barely change{'' if plural else 's'} the answer")
            own = "their" if plural else "its"
            if z is None:
                does = f"{own} output feeds {own} layer's computation rather than the output"
            elif rank is not None and rank <= 3 and z > 0.5:
                does = f"{own} output writes the answer directly"
            elif z > 0.2:
                does = f"{own} output pushes the answer up a little"
            else:
                does = f"{own} output does not write the answer; later parts use it"
            lines.append(f"{subject} {weight}: {does}.")
        else:
            lines.append(f"{subject} {'are' if plural else 'is'} part of the mechanism.")

    def site(n):  # where the node reads: layer l's attention 2l, its MLP 2l + 1
        return min(2 * p["layer"] + (p["kind"] in ("mlp", "c_fc", "down_proj", "feature")) for p in n["pieces"])

    sources = {n["id"]: {e["from"] for e in ir["edges"] if e["to"] == n["id"] and e["from"] != "embed"} for n in ir["nodes"]}
    earlier = {n["id"]: {m["id"] for m in ir["nodes"] if site(m) < site(n)} for n in ir["nodes"]}
    writers = [e["from"] for e in ir["edges"] if e["to"] == "logits" and e["from"] != "embed"]
    if all(sources[k] == earlier[k] for k in sources) and set(writers) == set(name):
        lines.append("Each part reads the embedding and every earlier part, and each writes to the output.")
    else:
        for e in ir["edges"]:
            if e["from"] == "embed" or e["to"] == "logits":
                continue
            route = "" if e["route"] == "input" else f" through its {e['route']}"
            lines.append(f"{name[e['to']][0].upper() + name[e['to']][1:]} read{'' if ',' in name[e['to']] else 's'} "
                         f"{name[e['from']]}{route}.")
        if writers:
            lines.append(f"The output reads {', '.join(name[w] for w in writers)}.")
    lines.append("Every other part of the model writes what it writes on the counterfactual prompt.")
    return " ".join(lines)


def variable_ir(ir: dict) -> dict:
    """An algorithm program's IR as one node per aligned or claimed variable (the parts its align or claim
    names), for facts(): a variable's parts are removed together."""
    return {"model": ir["model"], "nodes": [{"id": v["name"], "pieces": v["pieces"]} for v in ir["variables"] if v["pieces"]]}


def described(source: str) -> dict[str, str]:
    """Per top-level function of an algorithm: what its author says it is (its docstring, else the comments
    that open its body), one line."""
    import io
    import tokenize

    quoted = mech.quote_parts(source)
    tree = ast.parse(quoted)
    comments = {t.start[0]: t.string[1:].strip() for t in tokenize.generate_tokens(io.StringIO(quoted).readline)
                if t.type == tokenize.COMMENT}
    out = {}
    for f in tree.body:
        if not isinstance(f, ast.FunctionDef):
            continue
        doc = ast.get_docstring(f)
        if doc:
            out[f.name] = " ".join(doc.split())
            continue
        lines, k = [], f.body[0].lineno - 1
        while k > f.lineno and k in comments:  # comments directly above the first statement
            lines.insert(0, comments[k])
            k -= 1
        out[f.name] = " ".join(lines)
    return out


def algorithm_source(ir: dict, behavior: dict, facts_of: dict[str, dict], score: dict | None = None) -> str:
    """An algorithm program as written, with a docstring stating the behavior and what the facts are, and
    each variable's measured facts as comments above its align or claim."""
    source = ir["source"]
    tree = ast.parse(mech.quote_parts(source))
    lines = source.splitlines()
    above: dict[int, list[str]] = {}
    for st in tree.body:
        call = st.value if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) else None
        if call and isinstance(call.func, ast.Name) and call.func.id in ("align", "claim") and call.args \
                and isinstance(call.args[0], ast.Name) and call.args[0].id in facts_of and call.args[0].id not in {
                    n for rows in above.values() for n in rows}:
            above[st.lineno] = [call.args[0].id]
    head = textwrap.wrap(f"Behavior {behavior['id']} ({ir['model']}): {behavior['description']}", 100)
    head += ["", *textwrap.wrap(
        f"Facts measured on the behavior's {next(iter(facts_of.values()))['targets'] if facts_of else 0} target tokens "
        f"(clean prompts; the answer is the next token), per variable: what removing its parts does to the answer, "
        f"and what their own write does to the logits through the direct path only (no later layers).", 100)]
    if score:
        head += ["", *textwrap.wrap(f"Score: {score['total_bits']:.4g} bits in total.", 100)]
    doc = tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) \
        and isinstance(tree.body[0].value.value, str)
    body = lines[tree.body[0].end_lineno:] if doc else lines
    if doc:  # the author's notes follow the printer's
        head += ["", *inspect.cleandoc(tree.body[0].value.value).splitlines()]
    offset = len(lines) - len(body)
    out = ['"""' + "\n".join(head) + '\n"""']
    for k, line in enumerate(body, start=offset + 1):
        for name in above.get(k, []):
            out += [f"# {name}: {row}" if i == 0 else f"#   {row}" for i, row in enumerate(textwrap.wrap(role(facts_of[name]), 92))]
        out.append(line)
    return "\n".join(out).rstrip() + "\n"


def departures(behavior: dict, model: str, prompts=None, examples: int = 2) -> tuple[float | None, int, list[str]]:
    """M against the behavior's expected answers on `prompts` (indices; all by default): the share of
    targets where M's top token is the expected token, the number of targets, and up to `examples`
    targets where it is not, in words (the text's end, M's top token, the expected token)."""
    tk = mech.tokenizer(model)
    hits, words = [], []
    for i in (range(len(behavior["prompts"])) if prompts is None else prompts):
        p = behavior["prompts"][i]
        for k, t in enumerate(p["target_positions"]):
            if not p.get("model_top") or k >= len(p["model_top"]) or t + 1 >= len(p["token_ids"]):
                continue
            top, expected = p["model_top"][k][0][0], tk.decode([p["token_ids"][t + 1]])
            hits.append(top == expected)
            if top != expected and len(words) < examples:
                text = tk.decode(p["token_ids"][: t + 1], skip_special_tokens=True)
                words.append(f"after {json.dumps(text[-48:], ensure_ascii=False)} M predicts {json.dumps(top, ensure_ascii=False)}, "
                             f"not {json.dumps(expected, ensure_ascii=False)}")
    return (sum(hits) / len(hits) if hits else None), len(hits), words


def algorithm_explanation(ir: dict, behavior: dict, facts_of: dict[str, dict], score: dict | None = None, prompts=None) -> str:
    """An algorithm program's plain-English explanation, stating measured facts about M and never the
    behavior's rule as if M followed it: the behavior (the task), how often M's top token is the expected
    answer and where it is not (on `prompts`, the indices the facts are measured on); per variable what it
    is (its author's words), the parts aligned to it, what removing them does to the answer, how often
    switching them to their counterfactual values makes M prefer the counterfactual's answer
    (switch_flip_share) and its interchange test (alignment_error_bits) when measured; with a score that
    carries the empty program's ("empty"), the shares of M's clean-vs-counterfactual difference the named
    parts reproduce on their own and remove inside M."""
    says = described(ir["source"])
    lines = [f"The behavior: {behavior['description'].rstrip('.')}."]
    share, n, words = departures(behavior, ir["model"], prompts)
    if share is not None:
        line = f"M's top token is the expected answer on {share:.0%} of the behavior's {n} targets"
        lines.append(line + ("; for example, " + "; ".join(words) if words else "") + ".")
    for v in ir["variables"]:
        name, what = v["name"], says.get(v["name"], "")
        reads = [r if r != "tokens" else "the tokens" for r in v["reads"]]
        sentence = f"{name}" + (f" ({what.rstrip('.')})" if what else "") + f" is computed from {' and '.join(reads)}"
        if v["pieces"]:
            pieces = v["pieces"]
            parts = part_words({"pieces": pieces})
            many = sum(1 if not isinstance(p["index"], list) else len(p["index"]) for p in pieces) > 1
            sentence += (f"; {parts} {'are' if many else 'is'} aligned to it" if v["role"] == "aligned"
                         else f"; it is the attention pattern of {parts}")
            f = facts_of.get(name) or {}
            them = "them" if many else "it"
            if "removal_answer_bits" in f:
                d = f["removal_answer_bits"]
                sentence += f". Removing {them} {'lowers' if d < 0 else 'raises'} the answer's log-probability by {abs(d):.2f} bits per target"
            if f.get("switch_flip_share") is not None:
                sentence += (f". Switching {them} to {'their' if many else 'its'} values on the counterfactual prompt makes M prefer "
                             f"the counterfactual's answer on {f['switch_flip_share']:.0%} of the targets where the two answers differ")
            if f.get("alignment_error_bits") is not None:
                e = f["alignment_error_bits"]
                sentence += f". Its interchange test (the values of {them} swapped between prompts inside M) has an alignment error of {e:.0f} bits"
        else:
            sentence += "; no parts are aligned to it"
        lines.append(sentence + ".")
    empty = (score or {}).get("empty")
    if score and empty and empty.get("exec_error_bits") and empty.get("necessity_error_bits"):
        reproduce = 1 - score["exec_error_bits"] / empty["exec_error_bits"]
        remove = 1 - score["necessity_error_bits"] / empty["necessity_error_bits"]
        lines.append(f"Run on their own, with every other part at its value on the counterfactual prompt, the named parts reproduce "
                     f"{reproduce:.0%} of the difference between M's outputs on the prompt and on its counterfactual; switched to "
                     f"their counterfactual values inside M, they remove {remove:.0%} of it.")
    lines.append(f"{ir['answer']} is the prediction.")
    lines.append("Every other part of the model writes what it writes on the counterfactual prompt.")
    return " ".join(lines)


def answer_of(source: str, explanation: str) -> str:
    """The full answer in the oracle's format: the program block, then the explanation."""
    return f"```python\n{source.rstrip()}\n```\n\n{explanation}\n"


def graph_of(ir: dict, behavior: dict, facts_of: dict[str, dict], score: dict | None = None) -> dict:
    """A small graph description for rendering: nodes (address, facts, one-line role) and edges."""
    return {"behavior": behavior["id"], "model": ir["model"], "description": behavior["description"],
            "nodes": [{"id": n["id"], "address": ", ".join(address(p) for p in n["pieces"]),
                       "pieces": n["pieces"], "role": role(facts_of[n["id"]]) if n["id"] in facts_of else None,
                       **facts_of.get(n["id"], {})} for n in ir["nodes"]],
            "edges": ir["edges"], "explanation": explanation_of(ir, behavior, facts_of), "score": score}


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
    e.g. wrong(facts) for the reader's control). An algorithm program (alignments) keeps its algorithm as
    written; its facts are per variable."""
    SHAPE[0] = mech.shapes(ir["model"])
    if ir.get("variables"):
        measured = measured if measured is not None else facts(engine_for(ir["model"]), variable_ir(ir), behavior)
        src = algorithm_source(ir, behavior, measured, score)
        check = mech.trace_inline(src, ir["model"], behavior=behavior, decomposition=ir.get("decomposition") or "native")
        if not check["valid"] or {k: check[k] for k in ("nodes", "edges", "alignments")} != {k: ir[k] for k in ("nodes", "edges", "alignments")}:
            raise ValueError(f"the printed program does not trace back to the same IR: {check['error']}")
        explanation = algorithm_explanation(ir, behavior, measured, score)
        return src, {"behavior": behavior["id"], "model": ir["model"], "description": behavior["description"],
                     "variables": [{**v, **measured.get(v["name"], {})} for v in ir["variables"]], "nodes": ir["nodes"],
                     "edges": ir["edges"], "explanation": explanation, "score": score}
    measured = measured if measured is not None else facts(engine_for(ir["model"]), ir, behavior)
    src = source_of(ir, behavior, measured, score)
    check = mech.trace_inline(src, ir["model"], decomposition=ir.get("decomposition") or "native")
    shape = lambda g: ([(n["id"], n["pieces"]) for n in g["nodes"]], g["edges"])  # noqa: E731
    if not check["valid"] or shape(check) != shape(ir):
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
    ap.add_argument("--decomposition", help="what PD names: vpd, library, transcoder or native (default: the model's)")
    a = ap.parse_args()
    behavior = json.loads(a.behavior.read_text())
    text = a.program.read_text()
    if a.program.suffix == ".json":
        data = json.loads(text)
        ir = data if "nodes" in data else mech.trace_inline(data["source"], behavior["model"], behavior, a.decomposition)
    else:
        ir = mech.trace_inline(text, behavior["model"], behavior, a.decomposition)
    if not ir["valid"]:
        sys.exit(f"invalid program: {ir['error']}")
    score = json.loads(a.score.read_text()) if a.score else None
    SHAPE[0] = mech.shapes(ir["model"])
    measured = facts(engine_for(ir["model"]), variable_ir(ir) if ir.get("variables") else ir, behavior)
    src, graph = printed(ir, behavior, score, measured)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    name = a.name or f"{behavior['id']}.{a.program.stem}"
    (a.out_dir / f"{name}.py").write_text(src)
    (a.out_dir / f"{name}.answer.txt").write_text(answer_of(src, graph["explanation"]))
    if a.wrong:
        bad_src, bad = printed(ir, behavior, score, wrong(measured))
        (a.out_dir / f"{name}.wrong.py").write_text(bad_src)
        (a.out_dir / f"{name}.wrong.answer.txt").write_text(answer_of(bad_src, bad["explanation"]))
    (a.out_dir / f"{name}.graph.json").write_text(json.dumps(graph, indent=1))
    print(src)


if __name__ == "__main__":
    main()

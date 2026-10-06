"""The oracle on VPD's decomposition of vpd4l (#2951), answer mode: Qwen3 with a LoRA reads one rank-one
subcomponent u v^T of the 4-layer Pile model (its two vectors, through learned linear maps, injected at
placeholder tokens with reporter.py's hook) and answers questions whose answers are exact measurements
(vpd_labels.py), trained on the proper log score.

Input of a subcomponent. Its site (layer, kind) in words; its read vector v and write vector u, each
mapped by a learned linear map of its own space (per kind and side: q, k, v read the normed residual and
write a head space; o reads the heads' concatenation and writes the residual; c_fc reads the normed
residual and writes the MLP's hidden layer; down_proj reads that and writes the residual) to the
oracle's hidden width, and injected after decoder layer `--inject` at two placeholder tokens: the
residual r there becomes r + ||r|| m / ||m|| (m the mapped vector) plus a learned embedding of the
vector's role and of log ||u|| ||v|| (reporter.Magnitude: u v^T fixes only the product of the norms).

Questions (each a choice among labelled options; q = the softmax of the oracle's logits on the letters;
loss -sum_k p_k ln q_k with p the measured answer, one-hot here):
  activity    at a marked token of a text: the subcomponent's activity there on the 0-9 scale of its
              largest |v . x| over its measured contexts (floor(10 |a| / peak), 9 at most); the marked
              token is the context's peak or a uniformly drawn position, half each;
  direction   removing (alpha 0) or amplifying (alpha 1.5) the subcomponent: does the probability of a
              listed next token at the marked position rise or fall (a token drawn from the 10 that rise
              most and the 10 that fall most there; the subcomponent's top contexts only, where its
              effects exceed the arithmetic's floor);
  top         which of 4 tokens rises most in probability when the subcomponent is removed (the measured
              top one and 3 drawn from other contexts' lists).
  which_upstream  (readers, with a graph) which of its 4 strongest upstream writers contributes most to
              its activity at its peak in a context: the largest reduction of |a_B| when that edge is
              cut by path patching (vpd_graph.py);
  edge_cut    (readers, with a graph) cutting the edge from one of those writers: does a listed next
              token's probability at the peak rise or fall.
Graph neighbourhood. With --graph, every example carries its subcomponent's k strongest residual
neighbours (upstream writers of a reader, downstream readers of a writer; vpd_graph.py), each as two more
placeholders (the neighbour's read and write vectors through their own maps, the magnitude term carrying
the log of the edge's strength) and, in the graph condition, a line naming each neighbour's site and
strength. Edge questions name their candidates' sites in every condition.
Conditions at matched capacity (same base, adapter, maps, examples, order, steps; every placeholder
always present, those a condition withholds receive nothing): graph (the subcomponent's vectors and its
neighbourhood), weights (its vectors alone), activity (no vectors; its three most active other contexts
as text with their peak token marked and its 0-9 level), nothing.

Held out: subcomponents of --heldout-layers (never trained on) on held-out texts (the held-out label run),
and trained layers on held-out texts.

  vpd_oracle.py train --base MODEL --labels TRAIN_DIR --uv UV [--graph GRAPH_DIR] --condition C --steps N --out DIR
                      [--heldout-layers 2] [--examples 65536] [--batch 16] [--lr 1e-4] [--lora-rank 64]
  vpd_oracle.py evaluate --run DIR --labels HELDOUT_DIR --uv UV [--graph GRAPH_DIR] [--examples 4096]
  vpd_oracle.py compare --runs DIR... --eval eval_<labels>.jsonl --out SUMMARY.json
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

KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj")
LABELS = "ABCDEFGHIJ"
CONDITIONS = ("graph", "weights", "activity", "nothing")
# reporter.Magnitude's role embedding indices: own read, own write, a neighbour's read, a neighbour's write.
ROLE = {"read": 4, "write": 5, "neighbour_read": 1, "neighbour_write": 3}
BINS = 10
TOKENIZER = Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def site_name(layer: int, kind: str) -> str:
    return f"h.{layer}.{'mlp' if kind in ('c_fc', 'down_proj') else 'attn'}.{kind}"


class Table:
    """A label run's sites (vpd_labels.py output) and the decomposition's vectors."""

    def __init__(self, root: Path, uv_path: Path, graph: Path | None = None):
        import tokenizers

        self.root = root
        self.graph, self.k, self.offsets = None, 0, None
        if graph is not None:
            self.graph = load_file(str(graph / "graph.safetensors"))
            meta = json.loads((graph / "graph.json").read_text())
            self.k, self.offsets = meta["k"], meta["site_offsets"]
        self.tokens = load_file(str(root / "contexts.safetensors"))["tokens"].long()
        self.sites = {}
        for meta_path in sorted(root.glob("site_*.json")):
            meta = json.loads(meta_path.read_text())
            kind = meta["site"].split(".")[-1]
            self.sites[(meta["layer"], kind)] = (meta, load_file(str(meta_path.with_suffix(".safetensors"))))
        self.uv = load_file(str(uv_path))
        self.tok = tokenizers.Tokenizer.from_file(str(TOKENIZER))

    def vectors(self, layer: int, kind: str, c: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = site_name(layer, kind)
        return self.uv[f"{name}.V"][:, c], self.uv[f"{name}.U"][c]

    def site_of(self, gid: int) -> tuple[int, str, int]:
        """(layer, kind, index) of a global subcomponent id (vpd_graph.Ids)."""
        name = max((n for n, o in self.offsets.items() if o <= gid), key=lambda n: self.offsets[n])
        return int(name.split(".")[1]), name.split(".")[-1], gid - self.offsets[name]

    def neighbours(self, layer: int, kind: str, c: int) -> list[tuple[int, str, int, float]]:
        """The k strongest residual neighbours: (layer, kind, index, strength), strongest first."""
        if self.graph is None:
            return []
        name = site_name(layer, kind)
        key = f"{name}.up_ids" if f"{name}.up_ids" in self.graph else f"{name}.down_ids"
        if key not in self.graph:
            return []
        ids = self.graph[key][c].tolist()
        strength = self.graph[key.replace("_ids", "_strength")][c].tolist()
        return [(*self.site_of(int(g)), float(s)) for g, s in zip(ids, strength)]

    def text(self, ctx: int, upto: int, mark: int) -> str:
        """The context's tokens up to `upto` (inclusive) with token `mark` set off by double brackets."""
        ids = self.tokens[ctx, : upto + 1].tolist()
        return self.tok.decode(ids[:mark]) + "⟦" + self.tok.decode([ids[mark]]) + "⟧" + self.tok.decode(ids[mark + 1 :])


def level(a: float, peak: float) -> int:
    return 0 if peak <= 0 else min(BINS - 1, int(math.floor(BINS * abs(a) / peak)))


def examples(table: Table, layers: set[int], count: int, seed: int) -> list[dict]:
    """`count` questions (a third of each kind), subcomponents drawn uniformly from `layers`' sites."""
    rng = random.Random(seed)
    keys = [k for k in table.sites if k[0] in layers]
    out = []
    while len(out) < count:
        layer, kind = keys[rng.randrange(len(keys))]
        meta, d = table.sites[(layer, kind)]
        c = rng.randrange(meta["subcomponents"])
        contexts = d["contexts"][c].long()
        act = d["activity"][c].float()  # [K, T]
        peak = float(act.abs().max())
        top = meta["top"]
        name = site_name(layer, kind)
        patched = table.graph is not None and f"{name}.patch_activity" in table.graph
        which = len(out) % (5 if patched else 3)
        base = {"layer": layer, "kind": kind, "c": c}
        if which == 0:
            j = rng.randrange(len(contexts))
            p = int(d["position"][c, j]) if rng.random() < 0.5 else rng.randrange(act.shape[1])
            answer = level(float(act[j, p]), peak)
            out.append({**base, "kind_q": "activity", "context": int(contexts[j]), "position": p, "j": j,
                        "question": "How active is the component at the marked token, on a scale from 0 (inactive) to 9 (its largest activity)?",
                        "options": [str(b) for b in range(BINS)], "answer": answer})
        elif which == 1:
            j = rng.randrange(top)
            p = int(d["position"][c, j])
            edit = rng.choice(["ablate", "amplify"])
            side = rng.choice(["up", "down"])
            r = rng.randrange(10)
            token = int(d[f"{side}_ids_{edit}"][c, j, r])
            dp = float(d[f"{side}_dp_{edit}"][c, j, r])
            verb = "removed" if edit == "ablate" else "made 1.5 times stronger"
            word = table.tok.decode([token])
            out.append({**base, "kind_q": "direction", "context": int(contexts[j]), "position": p, "j": j,
                        "question": f"If the component is {verb}, does the probability that the next token after the marked token is {json.dumps(word)} go up or go down?",
                        "options": ["up", "down"], "answer": 0 if dp > 0 else 1})
        elif which in (3, 4):
            g = table.graph
            J = g[f"{name}.patch_activity"].shape[1]
            j = rng.randrange(J)
            p = int(d["position"][c, j])
            ups = table.neighbours(layer, kind, c)[: g[f"{name}.patch_activity"].shape[2]]
            names = [f"N{i + 1} (layer {l}, {k})" for i, (l, k, _, _) in enumerate(ups)]
            if which == 3:
                sign = 1.0 if float(act[j, p]) >= 0 else -1.0
                support = [-sign * float(x) for x in g[f"{name}.patch_activity"][c, j]]
                out.append({**base, "kind_q": "which_upstream", "context": int(contexts[j]), "position": p, "j": j,
                            "question": "At the marked token, which of these upstream components contributes most to this component's activity?",
                            "options": names, "answer": int(np.argmax(support))})
            else:
                i = rng.randrange(len(ups))
                side = rng.choice(["up", "down"])
                r = rng.randrange(10)
                token = int(g[f"{name}.patch_{side}_ids"][c, j, i, r])
                dp = float(g[f"{name}.patch_{side}_dp"][c, j, i, r])
                out.append({**base, "kind_q": "edge_cut", "context": int(contexts[j]), "position": p, "j": j,
                            "question": f"If the connection from {names[i]} to this component is cut (its write removed from what this component reads), does the probability that the next token after the marked token is {json.dumps(table.tok.decode([token]))} go up or go down?",
                            "options": ["up", "down"], "answer": 0 if dp > 0 else 1})
        else:
            j = rng.randrange(top)
            p = int(d["position"][c, j])
            truth = int(d["up_ids_ablate"][c, j, 0])
            pool = [int(x) for x in d["up_ids_ablate"][c, rng.randrange(top, len(contexts))].tolist() if int(x) != truth]
            distractors = rng.sample(pool, 3) if len(pool) >= 3 else pool + [truth + 1] * (3 - len(pool))
            options = [truth] + distractors
            order = list(range(4))
            rng.shuffle(order)
            out.append({**base, "kind_q": "top", "context": int(contexts[j]), "position": p, "j": j,
                        "question": "If the component is removed, which of these next tokens after the marked token gains the most probability?",
                        "options": [json.dumps(table.tok.decode([options[i]])) for i in order], "answer": order.index(0)})
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


def prompt(table: Table, ex: dict, condition: str) -> tuple[str, str]:
    """The user turn around the placeholders: (before, after)."""
    before = f"A component of a 4-layer language model: layer {ex['layer']}, {ex['kind']}. Its vectors:"
    info = "\n" + exemplars(table, ex) if condition == "activity" else ""
    if condition == "graph":
        near = table.neighbours(ex["layer"], ex["kind"], ex["c"])
        direction = "upstream components it reads from" if ex["kind"] in ("q_proj", "k_proj", "v_proj", "c_fc") else "downstream components that read it"
        lines = [f"N{i + 1}: layer {l}, {k}, strength {s:.3g}" for i, (l, k, _, s) in enumerate(near)]
        info += f"\nAfter its own two vectors come those of its strongest {direction}, in this order:\n" + "\n".join(lines)
    text = table.text(ex["context"], ex["position"], ex["position"])
    listing = "\n".join(f"{LABELS[k]}. {o}" for k, o in enumerate(ex["options"]))
    after = f"{info}\nText: {text!r}\n{ex['question']}\n{listing}\nAnswer with the letter."
    return before, after


class Oracle(torch.nn.Module):
    def __init__(self, base: str, lora_rank: int, inject: int, dev):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.dev = dev
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(dev)
        self.model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules="all-linear", lora_dropout=0.0))
        width = model.config.hidden_size
        dims = {"q_proj": (768, 768), "k_proj": (768, 768), "v_proj": (768, 768), "o_proj": (768, 768), "c_fc": (768, 3072), "down_proj": (3072, 768)}
        self.maps = torch.nn.ModuleDict({f"{k}_{side}": torch.nn.Linear(dims[k][i], width) for k in KINDS for i, side in enumerate(("read", "write"))}).to(dev)
        self.magnitude = Magnitude(width, 4).to(dev)
        self.hook = Injection(1.0, inject)
        model.model.layers[inject].register_forward_hook(self.hook)
        self.inject = inject
        self.placeholder = self.tokenizer.encode(" ?", add_special_tokens=False)[0]
        self.letters = [self.tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS]

    def trainable(self):
        return [p for p in self.model.parameters() if p.requires_grad] + list(self.maps.parameters()) + list(self.magnitude.parameters())

    def encode(self, before: str, after: str, slots: int) -> tuple[list[int], list[int]]:
        marker = "\u0000V\u0000"
        text = self.tokenizer.apply_chat_template([{"role": "user", "content": before + marker + after}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
        head, tail = text.split(marker)
        a = self.tokenizer.encode(head, add_special_tokens=False)
        b = self.tokenizer.encode(tail, add_special_tokens=False)
        return a + [self.placeholder] * slots + b, list(range(len(a), len(a) + slots))

    def injection(self, table: Table, components: list[tuple[int, str, int]], condition: str, places: list[list[int]]):
        """The hook's batch for rows each reading one subcomponent (layer, kind, index) at its
        placeholder positions `places[row]`: its read and write vectors, then its neighbours', each
        through its own map; the condition decides which are shown (module note)."""
        rows, cols, vecs, roles, mags, keeps, layers = [], [], [], [], [], [], []
        for b, ((layer0, kind0, c0), slots) in enumerate(zip(components, places)):
            near = table.neighbours(layer0, kind0, c0)
            items = [(layer0, kind0, c0, None, "")] + [(l, k, c, s, "neighbour_") for l, k, c, s in near]
            filled = 0
            for layer, kind, c, strength, who in items:
                v, u = table.vectors(layer, kind, c)
                scale = math.log(float(u.norm() * v.norm())) if strength is None else math.log(max(strength, 1e-30))
                shown = condition in ("graph", "weights") if strength is None else condition == "graph"
                for side, vec in (("read", v), ("write", u)):
                    rows.append(b)
                    cols.append(slots[filled])
                    filled += 1
                    vecs.append(self.maps[f"{kind}_{side}"](vec.to(self.dev)))
                    roles.append(ROLE[who + side])
                    mags.append(scale)
                    keeps.append(1.0 if shown else 0.0)
                    layers.append(layer)
            for _ in range(filled, len(slots)):  # a subcomponent with fewer neighbours: empty slots
                rows.append(b)
                cols.append(slots[filled])
                filled += 1
                vecs.append(torch.zeros(self.maps["q_proj_read"].out_features, device=self.dev))
                roles.append(ROLE["neighbour_read"])
                mags.append(0.0)
                keeps.append(0.0)
                layers.append(layer0)
        keep = torch.tensor(keeps, device=self.dev)
        unit = torch.nn.functional.normalize(torch.stack(vecs), dim=-1) * keep[:, None]
        logn = torch.tensor(mags, device=self.dev) * keep
        extra = self.magnitude(torch.tensor(roles, device=self.dev), torch.tensor(layers, device=self.dev), logn, 1.0 - keep)
        at = torch.full((len(rows),), self.inject, device=self.dev)
        return (torch.tensor(rows, device=self.dev), torch.tensor(cols, device=self.dev), unit, extra, keep, at)

    def load(self, run: Path):
        """An answer-mode run's adapter, maps and magnitude term (trainable)."""
        from peft import PeftModel

        inner = self.model.get_base_model()
        self.model = PeftModel.from_pretrained(inner, str(run / "adapter"), is_trainable=True)
        state = torch.load(run / "maps.pt")
        self.maps.load_state_dict(state["maps"])
        self.magnitude.load_state_dict(state["magnitude"])

    def log_q(self, table: Table, batch: list[dict], condition: str):
        slots = 2 + 2 * table.k
        enc = [self.encode(*prompt(table, ex, condition), slots) for ex in batch]
        width = max(len(ids) for ids, _ in enc)
        ids = torch.zeros(len(batch), width, dtype=torch.long)
        mask = torch.zeros(len(batch), width, dtype=torch.long)
        for b, (t, _) in enumerate(enc):
            ids[b, : len(t)] = torch.tensor(t)
            mask[b, : len(t)] = 1
        self.hook.set(self.injection(table, [(ex["layer"], ex["kind"], ex["c"]) for ex in batch], condition, [places for _, places in enc]))
        inner = self.model.get_base_model()
        try:
            hidden = inner.model(input_ids=ids.to(self.dev), attention_mask=mask.to(self.dev)).last_hidden_state
        finally:
            self.hook.set(None)
        last = mask.sum(1) - 1
        k = max(len(ex["options"]) for ex in batch)
        logits = torch.nn.functional.linear(hidden[torch.arange(len(batch), device=self.dev), last.to(self.dev)], inner.lm_head.weight[self.letters[:k]]).float()
        valid = torch.tensor([[j < len(ex["options"]) for j in range(k)] for ex in batch], device=self.dev)
        return torch.log_softmax(logits.masked_fill(~valid, float("-inf")), -1), valid


def log_scores(log_q, valid, batch) -> torch.Tensor:
    answers = torch.tensor([ex["answer"] for ex in batch], device=log_q.device)
    return log_q.gather(-1, answers[:, None])[:, 0]


def train(args):
    dev = device()
    torch.manual_seed(args.seed)
    held = {int(x) for x in args.heldout_layers.split(",") if x}
    table = Table(Path(args.labels), Path(args.uv), Path(args.graph) if args.graph else None)
    data = examples(table, {0, 1, 2, 3} - held, args.examples, args.seed)
    oracle = Oracle(args.base, args.lora_rank, args.inject, dev)
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
        torch.nn.utils.clip_grad_norm_(oracle.trainable(), 1.0)
        optimizer.step()
        log.write(json.dumps({"step": step, "loss_nats": float(loss.detach()), "seconds": time.time() - started}) + "\n")
        log.flush()
    torch.save({"maps": oracle.maps.state_dict(), "magnitude": oracle.magnitude.state_dict()}, out / "maps.pt")
    oracle.model.save_pretrained(str(out / "adapter"))
    (out / "config.json").write_text(json.dumps(vars(args)))
    print(json.dumps({"condition": args.condition, "steps": args.steps, "seconds": time.time() - started, "last_loss": float(loss.detach())}))


@torch.no_grad()
def evaluate(args):
    dev = device()
    config = json.loads((Path(args.run) / "config.json").read_text())
    oracle = Oracle(config["base"], config["lora_rank"], config["inject"], dev)
    oracle.load(Path(args.run))
    table = Table(Path(args.labels), Path(args.uv), Path(args.graph) if args.graph else None)
    held = {int(x) for x in config["heldout_layers"].split(",") if x}
    rows = []
    for split, layers in (("heldout_layers", held), ("trained_layers", {0, 1, 2, 3} - held)):
        data = examples(table, layers, args.examples, args.seed + 1)
        for s in range(0, len(data), config["batch"]):
            batch = data[s : s + config["batch"]]
            lq, valid = oracle.log_q(table, batch, config["condition"])
            for ex, score in zip(batch, log_scores(lq, valid, batch).tolist()):
                rows.append({"split": split, "question": ex["kind_q"], "layer": ex["layer"], "kind": ex["kind"], "c": ex["c"], "context": ex["context"], "log_score": score})
    (Path(args.run) / f"eval_{Path(args.labels).name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {}
    for split in ("heldout_layers", "trained_layers"):
        for q in ("activity", "direction", "top", "which_upstream", "edge_cut"):
            v = [r["log_score"] for r in rows if r["split"] == split and r["question"] == q]
            summary[f"{split}/{q}"] = float(np.mean(v)) if v else None
    print(json.dumps({"condition": config["condition"], **summary}))


def compare(args):
    """Per split and question: each condition's mean log score and its paired gain over `nothing`
    (the same examples in every run: evaluate draws them from the same seed), with standard errors."""
    runs = {}
    for d in args.runs:
        config = json.loads((Path(d) / "config.json").read_text())
        rows = [json.loads(line) for line in open(Path(d) / args.eval)]
        runs[config["condition"]] = rows
    base = runs["nothing"]
    table = {}
    for condition, rows in runs.items():
        for split in ("heldout_layers", "trained_layers"):
            for q in ("activity", "direction", "top", "which_upstream", "edge_cut"):
                pairs = [(r["log_score"], b["log_score"]) for r, b in zip(rows, base) if r["split"] == split and r["question"] == q]
                if not pairs:
                    continue
                d = np.array([a - b for a, b in pairs])
                table[f"{condition}/{split}/{q}"] = {"examples": len(d), "log_score_nats": float(np.mean([a for a, _ in pairs])),
                                                     "gain_over_nothing_nats": float(d.mean()), "standard_error_nats": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None}
    Path(args.out).write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--base", required=True)
    t.add_argument("--labels", required=True)
    t.add_argument("--uv", required=True)
    t.add_argument("--graph", help="vpd_graph.py's output for these labels (neighbourhoods and edge labels)")
    t.add_argument("--condition", required=True, choices=CONDITIONS)
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--heldout-layers", default="2")
    t.add_argument("--examples", type=int, default=65536)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--lora-rank", type=int, default=64)
    t.add_argument("--inject", type=int, default=1)
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("evaluate")
    e.add_argument("--run", required=True)
    e.add_argument("--labels", required=True)
    e.add_argument("--uv", required=True)
    e.add_argument("--graph", help="vpd_graph.py's output for these labels")
    e.add_argument("--examples", type=int, default=4096)
    e.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--eval", required=True, help="the evaluation file name inside each run, eval_<labels dir name>.jsonl")
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    {"train": train, "evaluate": evaluate, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()

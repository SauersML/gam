"""Description mode of the vpd4l oracle (#2951): GRPO on the two-part code-length reward.

Policy. An answer-mode oracle (a vpd_oracle.py run) reads one rank-one subcomponent of VPD's vpd4l
decomposition (in the run's condition: graph = its vectors and its residual neighbourhood at
placeholders) and, thinking on, may call two tools on the target model before it answers:
  edit_target(layer, kind, index, alpha)  W <- W + (alpha - 1) u v^T for that subcomponent (edits
      compose, any subcomponent, any order); returns the measured effect of every current edit at the
      peaks of the studied subcomponent's four strongest training contexts: KL(clean || edited) and the
      5 next tokens whose probability rises most and the 5 that fall most, each with its change;
  restore_target()  removes every edit.
Its final reply's text after "Description:" is the description (empty if absent).

Reward: -S(z) = -[L(z) + sum over experiments (x, a) of KL(p_M(. | x, a) || R(. | z, x, a))] in bits,
from codelength.py's service: z under the frozen prior model, and how far a frozen text-only reader R,
given z, the text and the edit in words, is from M's next-token distribution under no edit, removal and
doubling of the subcomponent at the peak of its strongest and other contexts (a row-edit table,
vpd_labels.py --edit row; --labels must be one). The no-description reader is the baseline.

GRPO. Per subcomponent, G episodes sampled at temperature 1; advantage (r - mean) / std within its group
(0 when the group's rewards are equal); loss = -mean over episodes of advantage x (mean log-probability
of the policy's own tokens, every turn, thinking and tool calls included); one gradient step per batch,
on the policy that sampled it, so the probability ratio is 1 and needs no clipping; no KL term. TRL's
GRPOTrainer is not used: the policy reads vectors injected into its residual stream at per-example
placeholders, which neither TRL's generation nor vLLM's can carry, so the objective is written out here.

  vpd_describe.py train --answer RUN --labels TRAIN_LABELS --uv UV --relations REL --reward HOST:PORT
                        --steps N --out DIR [--components 4] [--group 8] [--turns 4] [--tokens 768] [--lr 1e-5] [--hours H]
  vpd_describe.py evaluate --policy DIR --labels HELDOUT_LABELS --uv UV --relations REL --reward HOST:PORT
                        [--count 256]   (one description per held-out-layer subcomponent, scored)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd_2951"))
import vpd_model as VM  # noqa: E402
from vpd_labels import Model, load_uv  # noqa: E402
from vpd_oracle import SLOTS, Oracle, Table, device, site_name, slots  # noqa: E402

TOOLS = [
    {"type": "function", "function": {"name": "edit_target", "description": "Edit the target model: W <- W + (alpha - 1) u v^T for one subcomponent (alpha 0 removes it, 2 doubles it). Edits compose. Returns how the current edits change the next-token predictions at the studied component's strongest contexts.",
                                      "parameters": {"type": "object", "properties": {"layer": {"type": "integer"}, "kind": {"type": "string", "enum": ["q_proj", "k_proj", "v_proj", "o_proj", "c_fc", "down_proj"]},
                                                                                      "index": {"type": "integer"}, "alpha": {"type": "number"}}, "required": ["layer", "kind", "index", "alpha"]}}},
    {"type": "function", "function": {"name": "restore_target", "description": "Remove every edit of the target model.", "parameters": {"type": "object", "properties": {}}}},
]
INSTRUCTION = (
    "\nYou may call edit_target and restore_target to measure what this component and its neighbours do. "
    "Then answer with one line starting with 'Description:' that says, in plain words, on which tokens and texts "
    "the component is active and how it changes the model's next-token predictions. Shorter descriptions cost less."
)


def reward(address: str, items: list[dict]) -> list[float]:
    host, _, port = address.rpartition(":")
    with socket.create_connection((host, int(port))) as s:
        s.sendall((json.dumps({"op": "score", "items": items}) + "\n").encode())
        chunks = []
        while not chunks or not chunks[-1].endswith(b"\n"):
            b = s.recv(1 << 20)
            if not b:
                break
            chunks.append(b)
    reply = json.loads(b"".join(chunks))
    if "error" in reply:
        raise RuntimeError(reply["error"])
    return [r["reward"] for r in reply["ok"]["results"]]


class Target:
    """vpd4l with per-episode composable edits, measured at the studied subcomponent's peaks."""

    def __init__(self, dev, uv, table: Table):
        self.model = Model(dev)
        self.uv, self.table, self.dev = uv, table, dev

    @torch.no_grad()
    def effect(self, studied: tuple[int, str, int], edits: list[tuple[str, int, float]]) -> str:
        if not edits:
            return "No edits are applied."
        meta, d = self.table.sites[(studied[0], studied[1])]
        js = range(min(4, meta["top"]))
        ctx = torch.tensor([int(d["contexts"][studied[2], j]) for j in js], device=self.dev)
        pos = torch.tensor([int(d["position"][studied[2], j]) for j in js], device=self.dev)
        ids = self.table.tokens[ctx.cpu()].to(self.dev)
        t = self.model.t
        clean = VM.rms(self.run(ids, []), t.ln_f, t.eps)
        edited = VM.rms(self.run(ids, edits), t.ln_f, t.eps)
        lines = []
        for r in range(len(js)):
            lc, le = self.model.log_probs(clean[r, pos[r]]), self.model.log_probs(edited[r, pos[r]])
            dp = le.exp() - lc.exp()
            kl = float((lc.exp() * (lc - le)).sum())
            up = dp.topk(5).indices.tolist()
            down = (-dp).topk(5).indices.tolist()
            fmt = lambda ids_: ", ".join(f"{json.dumps(self.table.tok.decode([i]))} {float(dp[i]):+.3g}" for i in ids_)  # noqa: E731
            text = self.table.text(int(ctx[r]), int(pos[r]), int(pos[r]))
            lines.append(f"Text {r + 1}: {text[-160:]!r}\n  KL {kl:.3g} nats; rises: {fmt(up)}; falls: {fmt(down)}")
        return "\n".join(lines)

    def run(self, ids, edits):
        t = self.model.t
        originals = {}
        try:
            for name in {e[0] for e in edits}:
                st = t.site(name)
                mine = [(c, a) for n, c, a in edits if n == name]
                U, V = self.uv[name]
                original = st.forward
                originals[name] = original

                def forward(x, original=original, mine=mine, U=U, V=V):
                    out = original(x)
                    for c, a in mine:
                        out = out + (a - 1.0) * (x @ V[:, c])[..., None] * U[c]
                    return out

                st.forward = forward
            x = t.wte[ids]
            for i in range(t.n_layer):
                x, _ = self.model.layer(i, x)
            return x
        finally:
            for name, original in originals.items():
                t.site(name).forward = original


def tool_calls(text: str) -> list[dict]:
    out = []
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.S):
        try:
            out.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            out.append({"name": "invalid", "arguments": {}})
    return out


def description_of(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"Description:\s*(.+)", text)
    return m.group(1).strip().split("\n")[0] if m else ""


class Episodes:
    """Rollouts of the policy, a row per episode, turn by turn: each turn re-reads the whole conversation
    (left-padded batch; the hook injects at each row's placeholders) and samples until the end of the
    turn; a tool call is executed and its response appended as the template writes it."""

    def __init__(self, oracle: Oracle, table: Table, target: Target, condition: str, args):
        self.oracle, self.table, self.target, self.condition, self.args = oracle, table, target, condition, args
        tok = oracle.tokenizer
        self.end = tok.convert_tokens_to_ids("<|im_end|>")
        marker = "\u0000V\u0000"
        self.template = lambda before, after: tok.apply_chat_template([{"role": "user", "content": before + marker + after + INSTRUCTION}], tools=TOOLS,
                                                                       add_generation_prompt=True, enable_thinking=True, tokenize=False).split(marker)

    def example(self, comp) -> dict:
        layer, kind, c = comp
        return {"layer": layer, "kind": kind, "c": c, "kind_q": "describe", "context": int(self.table.sites[(layer, kind)][1]["contexts"][c, 0]), "position": 0, "j": -1,
                "question": "", "options": [], "candidates": [], "stratum": -1}

    def slots(self, comp) -> list[tuple]:
        return slots(self.table, self.example(comp), self.condition)

    def prompt(self, comp) -> tuple[list[int], list[int]]:
        from vpd_oracle import prompt as base_prompt

        before, after = base_prompt(self.table, self.example(comp), self.condition)
        after = after.split("\nText:")[0]  # the subcomponent and its neighbourhood, no question text
        head, tail = self.template(before, after)
        enc = self.oracle.tokenizer.encode
        a = enc(head, add_special_tokens=False)
        return a + [self.oracle.placeholder] * (2 * SLOTS) + enc(tail, add_special_tokens=False), list(range(len(a), len(a) + 2 * SLOTS))

    def batch_inputs(self, convs, places):
        width = max(len(c) for c in convs)
        ids = torch.zeros(len(convs), width, dtype=torch.long)
        mask = torch.zeros(len(convs), width, dtype=torch.long)
        shifted = []
        for r, (c, p) in enumerate(zip(convs, places)):
            off = width - len(c)
            ids[r, off:] = torch.tensor(c)
            mask[r, off:] = 1
            shifted.append([off + q for q in p])
        return ids.to(self.oracle.dev), mask.to(self.oracle.dev), shifted

    @torch.no_grad()
    def roll(self, comps: list[tuple[int, str, int]]):
        o, args = self.oracle, self.args
        convs, places, own, edits, done, calls = [], [], [], [], [], []
        for comp in comps:
            ids, p = self.prompt(comp)
            convs.append(ids)
            places.append(p)
            own.append([0] * len(ids))
            edits.append([])
            done.append(False)
            calls.append(0)
        for turn in range(args.turns + 1):
            live = [r for r in range(len(comps)) if not done[r]]
            if not live:
                break
            ids, mask, shifted = self.batch_inputs([convs[r] for r in live], [places[r] for r in live])
            o.hook.set(o.injection(self.table, [self.slots(comps[r]) for r in live], shifted))
            try:
                gen = o.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=args.tokens, do_sample=True, temperature=1.0, top_p=1.0,
                                       eos_token_id=self.end, pad_token_id=self.end)
            finally:
                o.hook.set(None)
            new = gen[:, ids.shape[1]:].tolist()
            for r, seq in zip(live, new):
                if self.end in seq:
                    seq = seq[: seq.index(self.end) + 1]
                convs[r] += seq
                own[r] += [1] * len(seq)
                text = o.tokenizer.decode(seq)
                found = tool_calls(text)
                if found and calls[r] < args.turns and turn < args.turns:
                    responses = []
                    for call in found:
                        calls[r] += 1
                        a = call.get("arguments", {})
                        try:
                            if call.get("name") == "edit_target":
                                edits[r].append((site_name(int(a["layer"]), str(a["kind"])), int(a["index"]), float(a["alpha"])))
                                responses.append(self.target.effect(comps[r], edits[r]))
                            elif call.get("name") == "restore_target":
                                edits[r].clear()
                                responses.append("Every edit is removed.")
                            else:
                                responses.append("Unknown tool.")
                        except (KeyError, ValueError, IndexError, TypeError) as e:
                            responses.append(f"Error: {e}")
                    reply = "".join(f"<|im_start|>user\n<tool_response>\n{x}\n</tool_response><|im_end|>\n" for x in responses) + "<|im_start|>assistant\n"
                    if not convs[r] or convs[r][-1] != self.end:
                        convs[r].append(self.end)
                        own[r].append(0)
                    extra = o.tokenizer.encode("\n" + reply, add_special_tokens=False)
                    convs[r] += extra
                    own[r] += [0] * len(extra)
                else:
                    done[r] = True
        texts = [o.tokenizer.decode([t for t, m in zip(c, w) if m]) for c, w in zip(convs, own)]
        return convs, places, own, [description_of(t) for t in texts], texts, calls


def policy_step(oracle: Oracle, table: Table, episodes: "Episodes", convs, places, own, comps, advantage, micro: int) -> float:
    """Accumulate the GRPO gradient: -sum_e advantage_e x (mean log-probability of episode e's own
    tokens) / episodes, micro-batch by micro-batch. The injection hook stays set until each micro-batch's
    backward pass ends, so recomputed (checkpointed) layers inject as the forward did; the output layer
    runs only at the policy's tokens, in checkpointed chunks (the vocabulary is 151,936 wide)."""
    from torch.utils.checkpoint import checkpoint

    inner = oracle.model.get_base_model()
    head = inner.lm_head
    total = 0.0
    for s in range(0, len(convs), micro):
        cs, ps, ws, ms = convs[s : s + micro], places[s : s + micro], own[s : s + micro], comps[s : s + micro]
        adv = torch.tensor(advantage[s : s + micro], device=oracle.dev, dtype=torch.float32)
        width = max(len(c) for c in cs)
        ids = torch.zeros(len(cs), width, dtype=torch.long)
        mask = torch.zeros(len(cs), width, dtype=torch.long)
        weight = torch.zeros(len(cs), width)
        for r, (c, w) in enumerate(zip(cs, ws)):
            ids[r, : len(c)] = torch.tensor(c)
            mask[r, : len(c)] = 1
            weight[r, : len(w)] = torch.tensor(w, dtype=torch.float32)
        ids, mask, weight = ids.to(oracle.dev), mask.to(oracle.dev), weight.to(oracle.dev)
        oracle.hook.set(oracle.injection(table, [episodes.slots(m) for m in ms], ps))
        try:
            hidden = oracle.model.base_model.model.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, :-1]
            rows, cols = (weight[:, 1:] > 0).nonzero(as_tuple=True)
            target = ids[:, 1:][rows, cols]
            flat = hidden[rows, cols]

            def piece(h, t):
                return torch.log_softmax(head(h).float(), -1).gather(-1, t[:, None])[:, 0]

            lp = torch.cat([checkpoint(piece, flat[k : k + 1024], target[k : k + 1024], use_reentrant=False) for k in range(0, len(target), 1024)])
            per = torch.zeros(len(cs), device=oracle.dev).index_add(0, rows, lp) / weight[:, 1:].sum(1).clamp(min=1)
            loss = -(adv * per).sum() / len(convs)
            loss.backward()
            total += float(loss.detach())
        finally:
            oracle.hook.set(None)
    return total


def components(table: Table, layers: set[int], count: int, rng: random.Random):
    keys = [k for k in table.sites if k[0] in layers]
    out = []
    for _ in range(count):
        layer, kind = keys[rng.randrange(len(keys))]
        out.append((layer, kind, rng.randrange(table.sites[(layer, kind)][0]["subcomponents"])))
    return out


def train(args):
    dev = device()
    run = Path(args.answer)
    config = json.loads((run / "config.json").read_text())
    table = Table(Path(args.labels), Path(args.uv), Path(args.relations) if args.relations else None, Path(args.tokenizer), Path(args.lens) if args.lens else None)
    oracle = Oracle(config["base"], config["lora_rank"], config["inject"], dev, table.dims, table.depth)
    oracle.load(run)
    target = Target(dev, load_uv(dev, Path(args.uv)), table)
    episodes = Episodes(oracle, table, target, config["condition"], args)
    held = {int(x) for x in config["heldout_layers"].split(",") if x}
    optimizer = torch.optim.AdamW(oracle.trainable(), lr=args.lr, weight_decay=0.0)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.jsonl", "a")
    started = time.time()
    for step in range(args.steps):
        if args.hours and time.time() - started > 3600 * args.hours:
            break  # the run's time budget: stop and save the policy as it is
        comps = [c for c in components(table, set(table.layers) - held, args.components, rng) for _ in range(args.group)]
        oracle.model.eval()
        oracle.model.base_model.model.gradient_checkpointing_disable()  # sampling keeps its cache
        convs, places, own, descriptions, texts, calls = episodes.roll(comps)
        rewards = np.array(reward(args.reward, [{"component": list(c), "description": d} for c, d in zip(comps, descriptions)]))
        groups = rewards.reshape(args.components, args.group)
        std = groups.std(1, keepdims=True)
        advantage = np.where(std > 0, (groups - groups.mean(1, keepdims=True)) / np.where(std > 0, std, 1.0), 0.0).reshape(-1)
        oracle.model.train()
        oracle.model.base_model.model.gradient_checkpointing_enable()
        oracle.model.base_model.model.config.use_cache = False
        optimizer.zero_grad(set_to_none=True)
        loss = policy_step(oracle, table, episodes, convs, places, own, comps, advantage, args.micro)
        torch.nn.utils.clip_grad_norm_(oracle.trainable(), 1.0)
        optimizer.step()
        log.write(json.dumps({"step": step, "mean_reward_bits": float(rewards.mean()), "best_reward_bits": float(rewards.max()), "loss": loss,
                              "tool_calls": float(np.mean(calls)), "empty_descriptions": int(sum(1 for d in descriptions if not d)),
                              "example": {"component": list(comps[0]), "description": descriptions[0]}, "seconds": time.time() - started}) + "\n")
        log.flush()
    oracle.model.save_pretrained(str(out / "adapter"))
    torch.save({"maps": oracle.maps.state_dict(), "magnitude": oracle.magnitude.state_dict()}, out / "maps.pt")
    (out / "config.json").write_text(json.dumps({**config, "describe": vars(args)}))


@torch.no_grad()
def evaluate(args):
    dev = device()
    run = Path(args.policy)
    config = json.loads((run / "config.json").read_text())
    table = Table(Path(args.labels), Path(args.uv), Path(args.relations) if args.relations else None, Path(args.tokenizer), Path(args.lens) if args.lens else None)
    oracle = Oracle(config["base"], config["lora_rank"], config["inject"], dev, table.dims, table.depth)
    oracle.load(run)
    target = Target(dev, load_uv(dev, Path(args.uv)), table)
    episodes = Episodes(oracle, table, target, config["condition"], argparse.Namespace(turns=args.turns, tokens=args.tokens))
    held = {int(x) for x in config["heldout_layers"].split(",") if x}
    comps = components(table, held, args.count, random.Random(args.seed))
    rows = []
    for s in range(0, len(comps), args.micro):
        chunk = comps[s : s + args.micro]
        _, _, _, descriptions, texts, calls = episodes.roll(chunk)
        rewards = reward(args.reward, [{"component": list(c), "description": d} for c, d in zip(chunk, descriptions)])
        rows += [{"component": list(c), "description": d, "reward_bits": r, "tool_calls": k} for c, d, r, k in zip(chunk, descriptions, rewards, calls)]
    (run / f"describe_eval_{Path(args.labels).name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"condition": config["condition"], "components": len(rows), "mean_reward_bits": float(np.mean([r["reward_bits"] for r in rows]))}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--labels", required=True)
        p.add_argument("--uv", required=True)
        p.add_argument("--relations", help="vpd_relations.py's output for these labels (the measured neighbourhoods)")
        p.add_argument("--tokenizer", default=str(Path.home() / "mpd-data/vpd/t-9d2b8f02/tokenizer.json"))
        p.add_argument("--reward", required=True, help="codelength.py serve's HOST:PORT")
        p.add_argument("--lens", help="vpd_lens.py build's output (an answer run of a *_lens condition)")
        p.add_argument("--turns", type=int, default=4, help="tool calls an episode may make")
        p.add_argument("--tokens", type=int, default=768, help="tokens one turn may generate")
        p.add_argument("--micro", type=int, default=8)
        p.add_argument("--seed", type=int, default=0)
    t = sub.choices["train"]
    t.add_argument("--answer", required=True, help="the answer-mode run (vpd_oracle.py train's output) the policy starts from")
    t.add_argument("--steps", type=int, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--components", type=int, default=4)
    t.add_argument("--group", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--hours", type=float, help="stop after this many hours of training and save (the pod's cap leaves room for evaluation)")
    e = sub.choices["evaluate"]
    e.add_argument("--policy", required=True)
    e.add_argument("--count", type=int, default=256)
    args = ap.parse_args()
    {"train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()

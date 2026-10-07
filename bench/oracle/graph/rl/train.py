"""The graph oracle's program training (#2951): sample programs, score them in bits, train on the score.

Each step takes K behaviors from the train split. For each, the oracle's input (prompt.py's render, as a
Qwen3 chat user turn, thinking off) gets N programs sampled at temperature 1: by vLLM serving the
current LoRA adapter when it is installed (a GPU), by transformers' generate otherwise. A program is
prompt.program_of(reply): the last fenced python block that parses, else the reply. scorer.py scores every
program: S = total bits (lower is better). One update follows, by --mode:

  bestofn  SFT on each behavior's best valid program (lowest S):
             loss = -(1/B) sum_e sum_t log pi(y_et | x_e, y_e<t).
           With --repair R, R rounds first show each behavior's best program and what the checker
           measured of it (terms, error, worst experiment families) to the policy and sample N
           revisions; a revision with lower S becomes the best. The kept program is trained as the
           answer to the original input, so experiments only make training data.
  dpo      the pair (best valid, an invalid one or else the worst) of each behavior with a valid program
           whose programs differ, log pi(y) = sum_t log pi(y_t):
             loss = -(1/P) sum log sigmoid(beta [(log pi(y_w) - log pi_ref(y_w)) - (log pi(y_l) - log pi_ref(y_l))]).
  grpo     reward r = -S, an invalid program counting as the group's worst valid one, advantage
           A_e = (r_e - mean_g r) / std_g r within the behavior's group (0 when the rewards are equal):
             loss = -(1/E) sum_e (A_e - beta stop_grad(rho_e)) log pi(y_e),  rho_e = log pi(y_e) - log pi_ref(y_e),
           whose expected gradient is that of E[-r] / std + beta KL(pi || pi_ref) over whole episodes
           (the KL gradient is E[rho grad log pi]); the mean of rho estimates the KL.
Invalid programs are infeasible everywhere: the checker scores one as the empty program, which can beat a
valid program that loses to the empty one, so no mode prefers it to a valid program.

Sums, not per-episode means: a per-episode mean of token log-probabilities (vpd_describe.py) gives each
token of a long program less weight than each token of a short one, so its gradient is not the gradient
of the expected score. The sum is the episode's log-probability; times the advantage it is the policy
gradient. One gradient step per sampled batch, on the policy that sampled it, so the importance ratio is
1 and needs no clipping (vLLM's bf16 log-probabilities differ from the trainer's by rounding only).

The reference pi_ref is the SFT policy: --init ADAPTER (g-predict's SFT adapter) is loaded twice, as the
trainable policy and as a frozen reference; without --init the policy is a fresh LoRA on the base and
pi_ref is the base (adapter disabled). The loop writes the adapter every step (vLLM loads it by path).

  train.py --mode grpo|dpo|bestofn --base Qwen/Qwen3-8B --model qwen3-0.6b --out DIR --steps S
           [--init SFT_ADAPTER] [--behaviors DIR] [--scorer checker|mock] [--behaviors-per-step 8]
           [--samples 8] [--max-tokens 1536] [--lr 1e-5] [--beta 0.04] [--lora-rank 32] [--hours H]
  train.py --mode eval --base ... --init ADAPTER --model ... --out DIR
  train.py --mode sft --programs 'SEARCH/*.json' --data QUESTIONS.jsonl --base ... --model ... --out DIR   (then evaluation)

Evaluation (--mode eval, or every --eval-every steps of training) samples N programs per behavior on two
sets and scores them under one experiment seed that no training step uses: the held-out behaviors (whole
families held out by g-behaviors' split) and the held-out prompts of the training behaviors (every
--prompt-holdout-th prompt, never shown or scored in training). Baselines on the same experiments: the
empty and the full program (e2e/programs.py) and g-int's search programs (e2e/search.py's outputs).
--init takes a PEFT adapter or g-predict's sft.py output directory (converted to PEFT's layout).

Outputs: DIR/train.jsonl (a line per step: scores, validity, loss, KL, seconds sampling / scoring /
training), DIR/eval.jsonl (a line per evaluated behavior and a summary per evaluation), DIR/samples.jsonl (every program with its score, for repair data and offline SFT),
DIR/best.jsonl (bestofn: the kept programs), DIR/adapter (the policy).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # CUDA may be initialized here (torch.cuda.is_available) before vLLM starts its engine process
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import prompt  # noqa: E402
from prompt import program_of, render  # noqa: E402
import scorer  # noqa: E402
from scorer import SCORERS  # noqa: E402

BEHAVIORS = Path.home() / "mpd-data/graph_oracle/behaviors"
SEARCH = Path.home() / "mpd-data/graph_oracle/runs/search"


def behaviors(root: Path, model: str, split: str) -> list[dict]:
    out = []
    for p in sorted((root / model).glob("*.json")):
        b = json.loads(p.read_text())
        if b.get("split", "train") == split:
            b["path"] = str(p)
            out.append(b)
    return out


def split_prompts(pool: list[dict], every: int, root: Path) -> tuple[list[dict], list[dict]]:
    """Each behavior's prompts split by position: every `every`-th prompt (index % every == 0) is held out.
    The two views are written as behavior files under root/train/ and root/heldout_prompts/ (the checker
    reads behaviors by path); returns (training views, held-out-prompt views). every = 0 holds out none."""
    if every <= 0:
        return pool, []
    views = ([], [])
    for b in pool:
        for side, name in enumerate(("train", "heldout_prompts")):
            v = {k: x for k, x in b.items() if k != "path"}
            v["prompts"] = [p for i, p in enumerate(b["prompts"]) if (i % every == 0) == bool(side)]
            path = root / name / f"{b['id']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(v))
            views[side].append({**v, "path": str(path)})
    return views


def init_adapter(path: str | None, out: Path) -> str | None:
    """--init as a PEFT adapter directory: itself, or g-predict's sft.py output (adapters.safetensors with
    '<module>.A' (r x in) and '<module>.B' (out x r), scale alpha / r, meta.json's args) rewritten in PEFT's
    layout, which computes the same W x + (alpha / r) B A x."""
    if path is None or (Path(path) / "adapter_config.json").exists():
        return path
    from safetensors.torch import load_file, save_file

    src = Path(path)
    meta = json.loads((src / "meta.json").read_text())["args"]
    tensors = load_file(str(src / "adapters.safetensors"))
    dst = out / "init_adapter"
    dst.mkdir(parents=True, exist_ok=True)
    save_file({f"base_model.model.{k.rsplit('.', 1)[0]}.lora_{k.rsplit('.', 1)[1]}.weight": v.contiguous() for k, v in tensors.items()}, str(dst / "adapter_model.safetensors"))
    targets = sorted({k.rsplit(".", 2)[1] for k in tensors})
    (dst / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": meta["rank"], "lora_alpha": meta["alpha"], "target_modules": targets,
                                                         "lora_dropout": 0.0, "bias": "none", "base_model_name_or_path": meta["model"], "fan_in_fan_out": False, "inference_mode": True}))
    return str(dst)


def baselines(b: dict) -> dict[str, str]:
    """Programs the oracle is compared with on behavior b, scored on the same experiments: g-int's
    references (the empty and the full program, e2e/programs.py), g-mech's example programs for b
    (examples/index.json) and the search baseline's final programs (runs/search/<behavior>.<mode>.json)."""
    sys.path.insert(0, str(HERE.parent / "e2e"))
    import programs

    refs = programs.references(b["model"])
    out = {"empty": refs["empty"], "full": refs["full"]}
    index = json.loads((HERE.parent / "examples/index.json").read_text())
    for name, entry in sorted(index.items()):  # g-mech's hand-written or measured example programs for this behavior
        if entry.get("behavior") == b["id"] and entry.get("model") == b.get("model"):
            out["example_" + name] = (HERE.parent / "examples" / f"{name}.py").read_text()
    for p in sorted(SEARCH.glob(f"{b['id']}.*.json")):
        out["search_" + p.stem[len(b["id"]) + 1 :]] = json.loads(p.read_text())["source"]
    return out


class Policy:
    """The trainable LoRA policy (adapter "default") and its frozen reference (adapter "ref" = the
    --init adapter, or the base with the adapter disabled)."""

    def __init__(self, args, dev):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.dev = dev
        self.tok = AutoTokenizer.from_pretrained(args.base)
        dtype = torch.float32 if dev.type == "cpu" else torch.bfloat16
        base = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(dev)
        if args.init:
            self.model = PeftModel.from_pretrained(base, args.init, adapter_name="default", is_trainable=True)
            self.model.load_adapter(args.init, adapter_name="ref", is_trainable=False)
            self.model.set_adapter("default")
            self.rank = self.model.peft_config["default"].r
        else:
            self.model = get_peft_model(base, LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
                                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
            self.rank = args.lora_rank
        self.has_ref = bool(args.init)
        self.pack = getattr(args, "pack", False)
        self.params = [p for n, p in self.model.named_parameters() if ".default." in n]
        for p in self.params:
            p.requires_grad_(True)
        self.end = self.tok.convert_tokens_to_ids("<|im_end|>")

    def prompt_ids(self, text: str) -> list[int]:
        chat = self.tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
        return self.tok.encode(chat, add_special_tokens=False)

    def save(self, path: Path):
        self.model.save_pretrained(str(path), selected_adapters=["default"])

    def token_logprobs(self, prompts: list[list[int]], completions: list[list[int]], ref: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """log pi(y_t | x, y_<t) at every completion token: (B, T) log-probabilities and a 0/1 mask. The
        output layer (151,936 wide) runs only at completion tokens, in checkpointed chunks. ref=True
        evaluates pi_ref without gradients. With self.pack and one prompt shared by every completion (a
        GRPO group, a DPO pair), the batch is ONE sequence: the prompt once, then each completion, each
        completion attending to the prompt and to its own earlier tokens only, at the positions it would
        have alone (a 4-D attention mask and explicit position ids), so the prompt's forward and backward
        run once per group instead of once per completion."""
        from torch.utils.checkpoint import checkpoint

        causal = self.model.base_model.model
        if self.pack and len(completions) > 1 and all(p == prompts[0] for p in prompts):
            prompt, P = prompts[0], len(prompts[0])
            seg, pos, src, rows, cols = [0] * P, list(range(P)), [], [], []
            for i, c in enumerate(completions):
                start = len(seg)
                seg += [i + 1] * len(c)
                pos += list(range(P, P + len(c)))
                src += [P - 1] + list(range(start, start + len(c) - 1))
                rows += [i] * len(c)
                cols += list(range(len(c)))
            ids = torch.tensor([prompt + [t for c in completions for t in c]], device=self.dev)
            seg_t = torch.tensor(seg, device=self.dev)
            q = torch.arange(len(seg), device=self.dev)
            allowed = (q[None, :] <= q[:, None]) & ((seg_t[None, :] == seg_t[:, None]) | (seg_t[None, :] == 0))
            dtype = next(causal.parameters()).dtype
            inputs = {"input_ids": ids, "position_ids": torch.tensor([pos], device=self.dev),
                      "attention_mask": torch.zeros(len(seg), len(seg), device=self.dev, dtype=dtype).masked_fill(~allowed, torch.finfo(dtype).min)[None, None]}
            src = torch.tensor(src, device=self.dev)
            target = ids[0, P:]
            shape = (len(completions), max(len(c) for c in completions))
        else:
            width = max(len(p) + len(c) for p, c in zip(prompts, completions))
            ids = torch.zeros(len(prompts), width, dtype=torch.long)
            att = torch.zeros(len(prompts), width, dtype=torch.long)
            comp = torch.zeros(len(prompts), width, dtype=torch.bool)
            for r, (p, c) in enumerate(zip(prompts, completions)):
                ids[r, : len(p) + len(c)] = torch.tensor(p + c)
                att[r, : len(p) + len(c)] = 1
                comp[r, len(p) : len(p) + len(c)] = True
            ids, att, comp = ids.to(self.dev), att.to(self.dev), comp.to(self.dev)
            inputs = {"input_ids": ids, "attention_mask": att}
            rows, cols = comp[:, 1:].nonzero(as_tuple=True)
            src = rows * width + cols  # the hidden state at position t predicts token t + 1
            target = ids[:, 1:][rows, cols]
            shape = (len(prompts), width - 1)
        rows, cols = torch.as_tensor(rows, device=self.dev), torch.as_tensor(cols, device=self.dev)

        def run():
            hidden = causal.model(**inputs).last_hidden_state
            flat = hidden.reshape(-1, hidden.shape[-1])[src]

            def piece(h, t):
                return torch.log_softmax(causal.lm_head(h).float(), -1).gather(-1, t[:, None])[:, 0]

            return torch.cat([checkpoint(piece, flat[k : k + 1024], target[k : k + 1024], use_reentrant=False) for k in range(0, len(target), 1024)])

        if ref:
            with torch.no_grad():
                if self.has_ref:
                    self.model.set_adapter("ref")
                    try:
                        lp = run()
                    finally:
                        self.model.set_adapter("default")
                        for p in self.params:
                            p.requires_grad_(True)
                else:
                    with self.model.disable_adapter():
                        lp = run()
        else:
            lp = run()
        out = torch.zeros(shape, device=self.dev, dtype=torch.float32).index_put((rows, cols), lp)
        mask = torch.zeros(shape, device=self.dev).index_put((rows, cols), torch.ones_like(lp))
        return out, mask

    def train_mode(self, on: bool):
        inner = self.model.base_model.model
        if on:
            self.model.train()
            inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            inner.config.use_cache = False
        else:
            self.model.eval()
            inner.gradient_checkpointing_disable()
            inner.config.use_cache = True


class HfSampler:
    """Samples with the policy itself (transformers' generate): the CPU / Mac path and the smoke test."""

    def __init__(self, policy: Policy, max_tokens: int, batch: int = 16):
        self.policy, self.max_tokens, self.batch = policy, max_tokens, batch
        self.logprob_sums = None

    @torch.no_grad()
    def __call__(self, prompts: list[list[int]], n: int, adapter: Path, version: int) -> list[list[list[int]]]:
        """n samples of every prompt, up to `batch` sequences per generate call (left-padded)."""
        pol = self.policy
        pol.train_mode(False)
        flat = [p for p in prompts for _ in range(n)]
        gens = []
        for s in range(0, len(flat), self.batch):
            chunk = flat[s : s + self.batch]
            width = max(len(p) for p in chunk)
            ids = torch.full((len(chunk), width), pol.end, dtype=torch.long)
            att = torch.zeros(len(chunk), width, dtype=torch.long)
            for r, p in enumerate(chunk):
                ids[r, width - len(p) :] = torch.tensor(p)
                att[r, width - len(p) :] = 1
            gen = pol.model.generate(input_ids=ids.to(pol.dev), attention_mask=att.to(pol.dev), max_new_tokens=self.max_tokens, do_sample=True, temperature=1.0, top_p=1.0,
                                     top_k=0, eos_token_id=pol.end, pad_token_id=pol.end)[:, width:].tolist()
            gens += [g[: g.index(pol.end) + 1] if pol.end in g else g for g in gen]
        out = [gens[k * n : (k + 1) * n] for k in range(len(prompts))]
        if pol.dev.type == "mps":
            torch.mps.empty_cache()  # the generation's cached blocks, before the checker servers start beside this process
        return out


class VllmSampler:
    """vLLM serving the base with the policy's adapter (reloaded by path at every version). It takes the
    first visible GPU; the trainer takes the second when there is one (--gpu-memory set accordingly)."""

    def __init__(self, args, rank: int, end: int):
        from vllm import LLM

        self.llm = LLM(model=args.base, dtype="bfloat16", enable_lora=True, max_lora_rank=rank, max_loras=1, enable_prefix_caching=True,
                       gpu_memory_utilization=args.gpu_memory, max_model_len=args.max_model_len, seed=args.seed)
        self.max_tokens, self.end = args.max_tokens, end

    def __call__(self, prompts: list[list[int]], n: int, adapter: Path, version: int) -> list[list[list[int]]]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        params = SamplingParams(n=n, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=self.max_tokens, stop_token_ids=[self.end], logprobs=0)
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, lora_request=LoRARequest(f"policy{version}", version + 1, str(adapter)), use_tqdm=False)
        try:  # the sampled tokens' log-probabilities, for the on-policy check only
            self.logprob_sums = [sum(d[t].logprob for d, t in zip(c.logprobs, c.token_ids)) for o in outs for c in o.outputs]
        except (TypeError, KeyError, AttributeError):  # a vLLM whose logprobs container differs
            self.logprob_sums = None
        return [[list(c.token_ids) for c in o.outputs] for o in outs]


class ValidSampler:
    """Draws n programs per prompt and redraws each invalid one (mech.trace: syntax, unknown names, an index
    beyond its site's size, a rule broken) up to `rounds` times, so the policy is sampled restricted to the
    programs it can write validly; stats holds the valid share before and after the redraws."""

    def __init__(self, inner, tok, model: str, rounds: int):
        self.inner, self.tok, self.model, self.rounds = inner, tok, model, rounds
        self.logprob_sums, self.stats = None, {}

    def valid(self, completions: list[list[int]]) -> list[bool]:
        import mech
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(8) as ex:  # each trace is a fork of a tracer server
            return list(ex.map(lambda c: bool(mech.trace(program_of(self.tok.decode(c, skip_special_tokens=True)), self.model)["valid"]), completions))

    def __call__(self, prompts: list[list[int]], n: int, adapter: Path, version: int) -> list[list[list[int]]]:
        groups = self.inner(prompts, n, adapter, version)
        sums = list(self.inner.logprob_sums) if self.inner.logprob_sums is not None else None
        flags = [self.valid(g) for g in groups]
        first = float(np.mean([f for fs in flags for f in fs]))
        for _ in range(self.rounds):
            slots = [(g, j) for g in range(len(groups)) for j in range(n) if not flags[g][j]]
            if not slots:
                break
            redo = self.inner([prompts[g] for g, _ in slots], 1, adapter, version)
            redo_sums = self.inner.logprob_sums
            ok = self.valid([r[0] for r in redo])
            for k, ((g, j), r) in enumerate(zip(slots, redo)):
                groups[g][j], flags[g][j] = r[0], ok[k]
                if sums is not None:
                    sums = None if redo_sums is None else sums
                    if sums is not None:
                        sums[g * n + j] = redo_sums[k]
        self.logprob_sums = sums
        self.stats = {"first_valid": first, "final_valid": float(np.mean([f for fs in flags for f in fs]))}
        return groups


def micro_batches(n: int, size: int):
    for s in range(0, n, size):
        yield list(range(s, min(n, s + size)))


def sft_update(pol: Policy, prompts, completions, micro: int) -> dict:
    total = 0.0
    for idx in micro_batches(len(prompts), micro):
        lp, mask = pol.token_logprobs([prompts[i] for i in idx], [completions[i] for i in idx])
        loss = -(lp * mask).sum() / len(prompts)
        loss.backward()
        total += float(loss.detach())
    return {"loss": total}


def dpo_update(pol: Policy, prompts, winners, losers, beta: float, micro: int) -> dict:
    total, margin = 0.0, 0.0
    for idx in micro_batches(len(prompts), micro):
        ps = [prompts[i] for i in idx]
        both = ps + ps
        comps = [winners[i] for i in idx] + [losers[i] for i in idx]
        ref, mask = pol.token_logprobs(both, comps, ref=True)
        cur, _ = pol.token_logprobs(both, comps)
        delta = ((cur - ref) * mask).sum(1)
        k = len(idx)
        z = beta * (delta[:k] - delta[k:])
        loss = -torch.nn.functional.logsigmoid(z).sum() / len(prompts)
        loss.backward()
        total += float(loss.detach())
        margin += float(z.detach().sum()) / len(prompts)
    return {"loss": total, "margin": margin}


def grpo_update(pol: Policy, prompts, completions, advantage, beta: float, micro: int) -> dict:
    total, kl, sums = 0.0, 0.0, []
    for idx in micro_batches(len(prompts), micro):
        ps, cs = [prompts[i] for i in idx], [completions[i] for i in idx]
        adv = torch.tensor([advantage[i] for i in idx], device=pol.dev, dtype=torch.float32)
        cur, mask = pol.token_logprobs(ps, cs)
        episode = (cur * mask).sum(1)
        sums += episode.detach().tolist()
        loss = -(adv * episode).sum() / len(prompts)
        if beta > 0:
            # KL(pi || pi_ref) of whole episodes: its gradient is E[(log pi(y) - log pi_ref(y)) grad log pi(y)], so the
            # episode's detached log-ratio multiplies its log-probability (a k3 penalty differentiated on the sampled
            # tokens is not this gradient). The mean log-ratio is the unbiased estimate of the KL itself.
            ref, _ = pol.token_logprobs(ps, cs, ref=True)
            ratio = ((cur - ref) * mask).sum(1).detach()
            loss = loss + beta * (ratio * episode).sum() / len(prompts)
            kl += float(ratio.sum()) / len(prompts)
        loss.backward()
        total += float(loss.detach())
    return {"loss": total, "kl_sum_per_episode": kl, "logprob_sums": sums}


def repair_prompt(behavior: dict, source: str, result: dict) -> str:
    """The oracle's input for a revision: the behavior's input, a program and what the checker measured
    of it (its terms, its error, its worst experiment families)."""
    lines = [render(behavior), "", "A program for this behavior:", "```python", source.rstrip(), "```",
             f"Its score: {result['total_bits']:.6g} bits (execution error {result.get('exec_error_bits')}, reader error {result.get('reader_error_bits')}, code {result.get('code_bits')})."]
    if not result.get("valid", True):
        lines.append(f"It is invalid: {result.get('error')}")
    families = result.get("per_family") or {}
    if families:
        worst = sorted(families.items(), key=lambda kv: -(kv[1] if isinstance(kv[1], (int, float)) else kv[1].get("bits", 0)))[:5]
        lines.append("Its largest errors by experiment family: " + "; ".join(f"{k}: {v}" for k, v in worst) + ".")
    lines.append("Write an improved program.")
    return "\n".join(lines)


def repair(chosen: list[dict], best: list[dict], pol, sampler, score, args, adapter: Path, step: int) -> set[int]:
    """--repair rounds of revisions (training data only): each behavior's best program and its measured
    failures go back to the policy, N revisions are sampled and scored under the step's seed, and a
    revision that lowers S replaces the best (in place). Returns the behaviors whose best was replaced."""
    replaced = set()
    for _ in range(args.repair):
        prompts = [pol.prompt_ids(repair_prompt(b, program_of(x["text"]), x["score"])) for b, x in zip(chosen, best)]
        groups = sampler(prompts, args.samples, adapter, step)
        texts = [[pol.tok.decode(c, skip_special_tokens=True) for c in g] for g in groups]
        scores = score([{"source": program_of(t), "behavior": b, "seed": step, "uniform_seeds": args.uniform_seeds, "experiments": args.experiments} for b, ts in zip(chosen, texts) for t in ts])
        for g in range(len(chosen)):
            for j in range(args.samples):
                r = scores[g * args.samples + j]
                if (r["valid"], -r["total_bits"]) > (best[g]["score"]["valid"], -best[g]["score"]["total_bits"]):
                    best[g] = {"completion": groups[g][j], "text": texts[g][j], "score": r}
                    replaced.add(g)
    return replaced


def sft_examples(args, pol, pool: list[dict]) -> tuple[list, list]:
    """--mode sft's data as token ids (prompt, completion): program examples, the oracle's input for a
    TRAINING behavior -> the best program of that behavior among scored --programs files (g-int's
    {"behavior", "source", "score"} layout) plus every unscored one (printed examples); behaviors outside
    the training pool are never used; and
    --data examples (JSONL of {"messages": [user, assistant]} or {"prompt", "completion"}, e.g.
    g-predict's prediction questions). The completion ends with <|im_end|>."""
    import glob

    by_id = {b["id"]: b for b in pool}
    best = {}
    for pattern in args.programs or []:
        pattern = os.path.expanduser(pattern)
        for path in sorted(glob.glob(os.path.join(pattern, "*.json") if os.path.isdir(pattern) else pattern)):
            r = json.loads(Path(path).read_text())
            if r.get("behavior") not in by_id or not r.get("source"):
                continue
            if "score" not in r:  # an unscored program (a printed hand-written example): always an example
                best[path] = r
            elif r["behavior"] not in best or r["score"]["total_bits"] < best[r["behavior"]]["score"]["total_bits"]:
                best[r["behavior"]] = r
    end = [pol.end]
    programs = [(pol.prompt_ids(render(by_id[r["behavior"]])), pol.tok.encode("```python\n" + r["source"].strip() + "\n```", add_special_tokens=False) + end) for _, r in sorted(best.items())]
    questions = []
    for path in args.data or []:
        for line in open(os.path.expanduser(path)):
            q = json.loads(line)
            user, answer = (q["messages"][0]["content"], q["messages"][1]["content"]) if "messages" in q else (q["prompt"], q["completion"])
            questions.append((pol.prompt_ids(user), pol.tok.encode(answer, add_special_tokens=False) + end))
    keep = lambda xs: [(p, c) for p, c in xs if len(p) + len(c) <= args.max_model_len]  # noqa: E731
    return keep(programs), keep(questions)


def sft(args, pol, pool, optimizer, log) -> dict:
    """--sft-steps steps of SFT: each batch draws --batch examples, a program example with probability
    --program-share and a question otherwise; loss = -(1/B) sum_e sum_t log pi(y_et) (sft_update)."""
    programs, questions = sft_examples(args, pol, pool)
    if not programs and not questions:
        raise SystemExit("no SFT examples (--programs, --data)")
    rng = random.Random(args.seed)
    meta = {"program_examples": len(programs), "question_examples": len(questions), "program_behaviors": len(programs)}
    log.write(json.dumps({"sft": meta}) + "\n")
    started = time.time()
    for step in range(args.sft_steps):
        if args.hours and time.time() - started > 3600 * args.hours:
            break
        batch = [rng.choice(programs) if programs and (not questions or rng.random() < args.program_share) else rng.choice(questions) for _ in range(args.batch)]
        pol.train_mode(True)
        stats = sft_update(pol, [p for p, _ in batch], [c for _, c in batch], args.micro)
        stats["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(pol.params, 1.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens = sum(len(c) for _, c in batch)
        log.write(json.dumps({"step": step, "loss_nats_per_example": stats["loss"], "bits_per_token": stats["loss"] * len(batch) / tokens / np.log(2), **{k: v for k, v in stats.items() if k != "loss"},
                              "programs": sum(1 for x in batch if x in programs), "elapsed": time.time() - started}) + "\n")
        log.flush()
    return meta


ORACLE_RUNS = Path.home() / "mpd-data/graph_oracle/runs/oracle"


def recovered(x: dict, empty: dict | None) -> float | None:
    """The share of the behavior's signal a program recovers: 1 - its execution error / the empty
    program's, on the same experiments."""
    if not empty or not empty.get("exec_error_bits"):
        return None
    return 1.0 - x["exec_error_bits"] / empty["exec_error_bits"]


def summarize(name: str, step: int, groups: list[tuple[dict, list, dict]], log) -> dict:
    """Rows per behavior and the set's summary from groups of (behavior, [(source, score)] of the oracle,
    {baseline name: score}): mean single-sample S, best of N, validity, the share of programs below the
    empty program's S, the signal the best recovers, and each baseline's S and recovered signal."""

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    rows = []
    for b, mine, base in groups:
        S = np.array([x["total_bits"] for _, x in mine], dtype=float)
        valid = np.array([bool(x["valid"]) for _, x in mine])
        # the best VALID program (an invalid one is scored as the empty program without its code, so it would
        # "beat" the empty program by the empty program's code bits); none valid: the first program
        j = int(np.where(valid, S, np.inf).argmin()) if valid.any() else 0
        empty = base.get("empty")
        row = {"set": name, "step": step, "behavior": b["id"], "mean_bits": float(S.mean()), "best_bits": float(S[j]), "valid_fraction": float(valid.mean()),
               "below_empty_fraction": float(np.mean(valid & (S < empty["total_bits"]))) if empty else None, "best_recovered": recovered(mine[j][1], empty),
               "baselines": {n: x["total_bits"] for n, x in base.items()}, "baselines_recovered": {n: recovered(x, empty) for n, x in base.items()}, "best_source": mine[j][0]}
        rows.append(row)
        log.write(json.dumps(row) + "\n")
    names = sorted({n for _, _, base in groups for n in base})
    return {"behaviors": len(rows), "mean_bits": mean([r["mean_bits"] for r in rows]), "best_of_n_bits": mean([r["best_bits"] for r in rows]),
            "valid_fraction": mean([r["valid_fraction"] for r in rows]), "below_empty_fraction": mean([r["below_empty_fraction"] for r in rows]),
            "best_recovered": mean([r["best_recovered"] for r in rows]), "baselines": {n: mean([r["baselines"].get(n) for r in rows]) for n in names},
            "baselines_recovered": {n: mean([r["baselines_recovered"].get(n) for r in rows]) for n in names},
            "oracle_mean_bits_on_baseline_behaviors": {n: mean([r["mean_bits"] for r in rows if n in r["baselines"]]) for n in names}}


def evaluate(sets: dict[str, list[dict]], pol, sampler, score, args, adapter: Path, version: int, log, step: int) -> dict:
    """Programs of the current policy on each evaluation set (N samples per behavior at temperature 1)
    and the baselines, all under one experiment seed (--eval-seed, never a training step's). Every program
    and its full score go to eval_samples.jsonl; each behavior's best program to
    runs/oracle/<behavior>.<run>.json (g-int's oracle-vs-search table); summarize gives the numbers."""
    summary = {}
    run = args.run_name or Path(args.out).name
    runs = Path(args.oracle_runs) if getattr(args, "oracle_runs", None) else ORACLE_RUNS
    runs.mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "eval_samples.jsonl", "a") as samples:
        for name, pool in sets.items():
            if not pool:
                continue
            prompts = [pol.prompt_ids(render(b)) for b in pool]
            groups = sampler(prompts, args.samples, adapter, version)
            items = [{"source": program_of(pol.tok.decode(c, skip_special_tokens=True)), "behavior": b, "seed": args.eval_seed, "experiments": args.eval_experiments} for b, g in zip(pool, groups) for c in g]
            base = [(b, n, src) for b in pool for n, src in (baselines(b).items() if args.baselines else [])]
            scores = score(items + [{"source": src, "behavior": b, "seed": args.eval_seed, "experiments": args.eval_experiments} for b, _, src in base])
            per_base = {}
            for (b, n, src), x in zip(base, scores[len(items) :]):
                per_base.setdefault(b["id"], {})[n] = x
                samples.write(json.dumps({"set": name, "step": step, "run": run, "behavior": b["id"], "behavior_path": b.get("path"), "program": n, "source": src, "score": x}) + "\n")
            out = []
            for g, b in enumerate(pool):
                mine = [(it["source"], x) for it, x in zip(items[g * args.samples : (g + 1) * args.samples], scores[g * args.samples : (g + 1) * args.samples])]
                for src, x in mine:
                    samples.write(json.dumps({"set": name, "step": step, "run": run, "behavior": b["id"], "behavior_path": b.get("path"), "program": "oracle", "source": src, "score": x}) + "\n")
                if getattr(args, "scorer", None) == "none":  # sampled and saved only: rescore summarizes
                    continue
                src, x = min(mine, key=lambda m: m[1]["total_bits"])
                (runs / f"{b['id']}.{run}.json").write_text(json.dumps({"behavior": b["id"], "model": b.get("model"), "source": src, "score": x, "stand_in": "counterfactual",
                                                                               "experiments": args.eval_experiments, "seed": args.eval_seed, "set": name, "step": step}))
                out.append((b, mine, per_base.get(b["id"], {})))
            if out:
                summary[name] = summarize(name, step, out, log)
            if getattr(sampler, "stats", None):
                summary.setdefault(name, {})["sampling"] = dict(sampler.stats)
    log.write(json.dumps({"summary": summary, "step": step}) + "\n")
    log.flush()
    return summary


def ir_signature(source: str, model: str) -> str:
    """What the checker's score depends on without a reader: the traced program (nodes, edges, Python
    token count and types, validity and error)."""
    import mech

    ir = mech.trace(source, model)
    return json.dumps({k: ir.get(k) for k in ("valid", "error", "nodes", "edges", "python_tokens", "token_types", "standin")}, sort_keys=True)


def rescore(args, score) -> dict:
    """Every program of earlier evaluations (--samples-from eval_samples.jsonl files) scored again under
    --eval-seed / --eval-experiments and --score-options (extra checker request keys, e.g. held-out
    experiment families or resampled counterfactuals once the checker offers them), summarized as in
    evaluate. Writes RUN/rescore_<tag>.jsonl one behavior at a time, so a rerun resumes after the
    behaviors already scored, and RUN/rescore_<tag>_summary.jsonl."""
    rows = [json.loads(line) for path in args.samples_from for line in open(os.path.expanduser(path))]
    options = json.loads(args.score_options) if args.score_options else None
    behaviors_by_path = {}
    for r in rows:
        if r["behavior_path"] not in behaviors_by_path:
            path = Path(r["behavior_path"] or "")
            if not path.is_file():  # written on another machine (a pod): the same behavior file here
                path = Path(args.behaviors) / args.model / f"{r['behavior']}.json"
            behaviors_by_path[r["behavior_path"]] = {**json.loads(path.read_text()), "path": str(path)}
    out = Path(args.out)
    done_path = out / f"rescore_{args.rescore_tag}.jsonl"
    key = lambda r: (r["run"], r["step"], r["set"], r["behavior"], r["program"], r["source"])  # noqa: E731
    done = {key(r): r for r in map(json.loads, open(done_path))} if done_path.exists() else {}
    if args.summary_only:  # summarize what is scored so far, score nothing
        rows = [r for r in rows if key(r) in done]
    for bpath in sorted(behaviors_by_path, key=str):
        todo = list({key(r): r for r in rows if r["behavior_path"] == bpath and key(r) not in done}.values())  # each distinct program once
        if not todo:
            continue
        reps, rep_of = {}, []
        for r in todo:  # programs with the same traced IR get the same checker score (no reader term)
            sig = ir_signature(r["source"], behaviors_by_path[bpath]["model"]) if not os.environ.get("GRAPH_READER") else r["source"]
            rep_of.append(reps.setdefault(sig, len(reps)))
        firsts = {}
        for i, j in enumerate(rep_of):
            firsts.setdefault(j, i)
        unique = [todo[firsts[j]] for j in range(len(reps))]
        scored = score([{"source": r["source"], "behavior": behaviors_by_path[bpath], "seed": args.eval_seed, "experiments": args.eval_experiments, "options": options} for r in unique])
        scores = [scored[j] for j in rep_of]
        with open(done_path, "a") as f:
            for r, x in zip(todo, scores):
                rec = {**r, "score_before": r["score"], "score": x, "options": options, "seed": args.eval_seed, "experiments": args.eval_experiments}
                f.write(json.dumps(rec) + "\n")
                done[key(r)] = rec
    summary = {}
    log = open(out / f"rescore_{args.rescore_tag}_summary.jsonl", "w")
    for name in sorted({r["set"] for r in rows}):
        groups, shared = {}, {}
        for r in rows:
            if r["set"] != name:
                continue
            x = done[key(r)]["score"]
            if r["program"] == "oracle":
                groups.setdefault((r["run"], r["behavior"]), (behaviors_by_path[r["behavior_path"]], [], {}))[1].append((r["source"], x))
            else:  # baselines are the same programs on the same experiments for every run: shared by behavior
                shared.setdefault(r["behavior"], {})[r["program"]] = x
        for run in sorted({k[0] for k in groups}):
            mine = [(b, progs, shared.get(bid, {})) for (rn, bid), (b, progs, _) in sorted(groups.items()) if rn == run]
            summary[f"{name}/{run}"] = summarize(f"{name}/{run}", -1, mine, log)
    log.write(json.dumps({"summary": summary, "options": options}) + "\n")
    return summary


def views_of(args) -> dict | None:
    return {"vpd": args.vpd_view} if getattr(args, "vpd_view", None) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["grpo", "dpo", "bestofn", "sft", "eval", "rescore"], required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--init", help="SFT adapter the policy starts from and is held to (pi_ref): a PEFT directory or g-predict's sft.py output")
    ap.add_argument("--model", required=True, help="target model whose behaviors are explained: qwen3-0.6b | vpd4l")
    ap.add_argument("--behaviors", default=str(BEHAVIORS))
    ap.add_argument("--scorer", choices=sorted(SCORERS), default="checker")
    ap.add_argument("--score-workers", type=int, default=1, help="checker servers per target model, each scoring whole behaviors in parallel")
    ap.add_argument("--checker", help="the checker binary (mpd_graph_2951; score.py's GRAPH_CHECKER); on MATS name target/release/examples/mpd_graph_2951 so the job builds it")
    ap.add_argument("--vpd-view", help="VPD's decomposition export for the checker's vpd view (programs with PD.vpd pieces are invalid without it; vpd4l: ~/mpd-data/engine/vpd4l_decomposition)")
    ap.add_argument("--checker-device", choices=["gpu"], help="run the checker's large products on the single-precision device (float32; compare scores only within one device)")
    ap.add_argument("--reader-items", type=int, default=0, help="without a reader server, keep the reader items of every K-th scored program for offline reader scoring (0: none)")
    ap.add_argument("--reader-item-stride", type=int, default=1, help="of a kept program's reader items, keep every S-th (an unbiased subsample of the reader term's mean)")
    ap.add_argument("--score-batch", type=int, default=4, help="programs per checker request (a server's memory grows with it)")
    ap.add_argument("--checker-gib", type=int, help="the checker server's memory lease on the Mac (score.py's default otherwise)")
    ap.add_argument("--export", help="the target model's export directory for the checker (score.py's EXPORTS entry otherwise)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--hours", type=float, help="stop and save after this many hours")
    ap.add_argument("--behaviors-per-step", type=int, default=8)
    ap.add_argument("--samples", type=int, default=8, help="programs per behavior per step (the group)")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--beta", type=float, help="grpo: KL weight (default 0.04); dpo: inverse temperature (default 0.1)")
    ap.add_argument("--sft-epochs", type=int, default=1)
    ap.add_argument("--repair", type=int, default=0, help="bestofn: rounds of revisions of each behavior's best program, shown its measured failures (training data only)")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--pack", action="store_true", help="GRPO / DPO: one sequence per group (the shared prompt once; see Policy.token_logprobs); sets GRPO's micro-batch to the group")
    ap.add_argument("--sampler", choices=["auto", "vllm", "hf"], default="auto")
    ap.add_argument("--resample", type=int, default=0, help="redraw each invalid program (mech.trace) up to R times at sampling time")
    ap.add_argument("--hf-batch", type=int, default=16, help="sequences per transformers generate call (the Mac / CPU sampler)")
    ap.add_argument("--gpu-memory", type=float, default=0.85, help="vLLM's share of its GPU (lower it when the trainer shares the GPU)")
    ap.add_argument("--prompt-holdout", type=int, default=4, help="every K-th prompt of each training behavior is held out for evaluation (0: none)")
    ap.add_argument("--eval-every", type=int, default=0, help="evaluate every E training steps and at the end (0: only --mode eval)")
    ap.add_argument("--eval-seed", type=int, default=1_000_003, help="the evaluation's experiment seed (training steps use their index)")
    ap.add_argument("--experiments", type=int, default=32, help="experiments per training score (fewer: cheaper, same expectation, more variance)")
    ap.add_argument("--eval-experiments", type=int, default=32, help="experiments per evaluation score")
    ap.add_argument("--eval-behaviors", type=int, default=0, help="evaluate a fixed random subset of this many behaviors per set (0: all)")
    ap.add_argument("--uniform-seeds", type=int, default=0, help="training draws experiments from step mod M (the checker's uniform_seeds: M's outcomes cached after M steps); the evaluation never")
    ap.add_argument("--no-baselines", dest="baselines", action="store_false", help="skip scoring the empty, full and search programs in evaluation")
    ap.add_argument("--programs", nargs="*", help="sft: program files (globs or directories of .json) in g-int's layout")
    ap.add_argument("--data", nargs="*", help="sft: JSONL question files ({'messages': [user, assistant]})")
    ap.add_argument("--program-share", type=float, default=0.3, help="sft: probability that a batch example is a program example")
    ap.add_argument("--sft-steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8, help="sft: examples per optimizer step")
    ap.add_argument("--samples-from", nargs="*", help="rescore: eval_samples.jsonl files of earlier evaluations")
    ap.add_argument("--score-options", help="rescore: JSON object of extra checker request keys")
    ap.add_argument("--summary-only", action="store_true", help="rescore: summarize the rows already scored, score nothing")
    ap.add_argument("--rescore-tag", default="rescore", help="rescore: output name RUN/rescore_<tag>.jsonl")
    ap.add_argument("--oracle-runs", help="directory of the per-behavior best-program files (default ~/mpd-data/graph_oracle/runs/oracle; a pod writes under its outputs)")
    ap.add_argument("--run-name", help="the run's name in runs/oracle/<behavior>.<run>.json (default: the --out directory's name)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lr = args.lr if args.lr is not None else {"bestofn": 1e-4, "sft": 1e-4}.get(args.mode, 1e-5)
    beta = args.beta if args.beta is not None else {"dpo": 0.1}.get(args.mode, 0.04)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    args.init = init_adapter(args.init, out)
    (out / "config.json").write_text(json.dumps({**vars(args), "lr": lr, "beta": beta}, indent=1))

    if args.mode == "rescore":  # no policy: scores saved programs again
        if args.checker:
            os.environ["GRAPH_CHECKER"] = str(Path(args.checker).resolve())
        scorer.WORKERS, scorer.EXPORT, scorer.MEMORY_GIB, scorer.VIEWS, scorer.DEVICE, scorer.BATCH, scorer.ITEMS_EVERY, scorer.ITEM_STRIDE = args.score_workers, args.export, args.checker_gib, views_of(args), args.checker_device, args.score_batch, args.reader_items, args.reader_item_stride
        print(json.dumps(rescore(args, SCORERS[args.scorer])))
        return
    use_vllm = args.sampler == "vllm" or (args.sampler == "auto" and torch.cuda.is_available() and __import__("importlib").util.find_spec("vllm") is not None)
    sampler = None
    if use_vllm:  # vLLM first, on the first visible GPU, before the trainer touches CUDA
        from transformers import AutoTokenizer

        rank = json.loads((Path(args.init) / "adapter_config.json").read_text())["r"] if args.init else args.lora_rank
        sampler = VllmSampler(args, rank, AutoTokenizer.from_pretrained(args.base).convert_tokens_to_ids("<|im_end|>"))
    dev = torch.device(f"cuda:{torch.cuda.device_count() - 1}" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    pol = Policy(args, dev)
    if sampler is None:
        sampler = HfSampler(pol, args.max_tokens, args.hf_batch)
    if args.resample:
        sampler = ValidSampler(sampler, pol.tok, args.model, args.resample)
    if args.checker:
        os.environ["GRAPH_CHECKER"] = str(Path(args.checker).resolve())
    score = SCORERS[args.scorer]
    scorer.WORKERS, scorer.EXPORT, scorer.MEMORY_GIB, scorer.VIEWS, scorer.DEVICE, scorer.BATCH, scorer.ITEMS_EVERY, scorer.ITEM_STRIDE = args.score_workers, args.export, args.checker_gib, views_of(args), args.checker_device, args.score_batch, args.reader_items, args.reader_item_stride
    root = Path(args.behaviors)
    pool, heldout_prompts = split_prompts(behaviors(root, args.model, "train"), args.prompt_holdout, out / "behaviors")
    sets = {"heldout_behaviors": behaviors(root, args.model, "heldout"), "heldout_prompts": heldout_prompts}
    if args.eval_behaviors:  # one fixed subset per set, the same at every evaluation
        sets = {k: random.Random(args.seed).sample(v, min(args.eval_behaviors, len(v))) for k, v in sets.items()}
    adapter = out / "adapter"
    if args.mode == "eval":
        pol.save(adapter)
        print(json.dumps(evaluate(sets, pol, sampler, score, args, adapter, 0, open(out / "eval.jsonl", "a"), 0)))
        return
    if not pool:
        raise SystemExit(f"no train behaviors under {root / args.model}")
    optimizer = torch.optim.AdamW(pol.params, lr=lr, weight_decay=0.0)
    if args.mode == "sft":
        sft(args, pol, pool, optimizer, open(out / "train.jsonl", "a"))
        pol.save(adapter)
        print(json.dumps(evaluate(sets, pol, sampler, score, args, adapter, 1, open(out / "eval.jsonl", "a"), args.sft_steps)))
        return
    log = open(out / "train.jsonl", "a")
    eval_log = open(out / "eval.jsonl", "a")
    samples_log = open(out / "samples.jsonl", "a")
    best_log = open(out / "best.jsonl", "a") if args.mode == "bestofn" else None
    started = time.time()
    for step in range(args.steps):
        if args.hours and time.time() - started > 3600 * args.hours:
            break
        pol.save(adapter)
        if args.eval_every and step % args.eval_every == 0:
            evaluate(sets, pol, sampler, score, args, adapter, step, eval_log, step)
        t0 = time.time()
        chosen = random.sample(pool, min(args.behaviors_per_step, len(pool)))
        prompts = [pol.prompt_ids(render(b)) for b in chosen]
        groups = sampler(prompts, args.samples, adapter, step)
        t1 = time.time()
        texts = [[pol.tok.decode(c, skip_special_tokens=True) for c in g] for g in groups]
        items = [{"source": program_of(t), "behavior": b, "seed": step, "uniform_seeds": args.uniform_seeds, "experiments": args.experiments} for b, ts in zip(chosen, texts) for t in ts]
        scores = score(items)
        t2 = time.time()
        S = np.array([s["total_bits"] for s in scores], dtype=float).reshape(len(chosen), args.samples)
        valid = np.array([bool(s["valid"]) for s in scores]).reshape(len(chosen), args.samples)
        for (b, it, s, c) in zip([b for b in chosen for _ in range(args.samples)], items, scores, [c for g in groups for c in g]):
            samples_log.write(json.dumps({"step": step, "behavior": b["id"], "source": it["source"], "completion_tokens": len(c), "score": s}) + "\n")
        samples_log.flush()
        flat_p = [p for p in prompts for _ in range(args.samples)]
        flat_c = [c for g in groups for c in g]
        pol.train_mode(True)
        optimizer.zero_grad(set_to_none=True)
        # An invalid program is infeasible: the checker scores it as the empty program, which can beat valid programs
        # that lose to the empty one, so for preferences and advantages it counts as no better than the group's worst
        # valid program (a group with no valid program gives no signal).
        S_feasible = np.array([np.where(valid[g], S[g], S[g][valid[g]].max()) if valid[g].any() else S[g] for g in range(len(chosen))])
        if args.mode == "grpo":
            r = -S_feasible
            std = r.std(1, keepdims=True)
            adv = np.where(std > 0, (r - r.mean(1, keepdims=True)) / np.where(std > 0, std, 1.0), 0.0).reshape(-1)
            stats = grpo_update(pol, flat_p, flat_c, adv.tolist(), beta, args.samples if args.pack else args.micro)
        elif args.mode == "dpo":
            pairs = [(g, int(np.where(valid[g], S[g], np.inf).argmin()), int(np.where(valid[g], S[g], np.inf).argmax()) if valid[g].all() else int((~valid[g]).argmax()))
                     for g in range(len(chosen)) if valid[g].any() and (not valid[g].all() or S[g].max() > S[g].min())]
            stats = dpo_update(pol, [prompts[g] for g, _, _ in pairs], [groups[g][w] for g, w, _ in pairs], [groups[g][l] for g, _, l in pairs], beta, 1 if args.pack else args.micro) if pairs else {}
            stats["pairs"] = len(pairs)
        else:
            best = [(int(np.where(valid[g], S[g], np.inf).argmin()) if valid[g].any() else int(S[g].argmin())) for g in range(len(chosen))]
            best = [{"completion": groups[g][j], "text": texts[g][j], "score": scores[g * args.samples + j]} for g, j in enumerate(best)]
            repaired = repair(chosen, best, pol, sampler, score, args, adapter, step)
            keep = [g for g in range(len(chosen)) if best[g]["score"]["valid"]]
            for g in keep:
                best_log.write(json.dumps({"behavior": chosen[g]["id"], "prompt": render(chosen[g]), "completion": best[g]["text"], "score": best[g]["score"], "repaired": g in repaired}) + "\n")
            best_log.flush()
            stats = {}
            for _ in range(args.sft_epochs if keep else 0):  # the kept program answers the ORIGINAL input: experiments never reach the oracle's input
                stats = sft_update(pol, [prompts[g] for g in keep], [best[g]["completion"] for g in keep], args.micro)
                torch.nn.utils.clip_grad_norm_(pol.params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            stats.update({"kept": len(keep), "repaired": len(repaired), "kept_mean_bits": float(np.mean([best[g]["score"]["total_bits"] for g in keep])) if keep else None})
        sums = stats.pop("logprob_sums", None)
        if sums is not None and sampler.logprob_sums is not None:  # on-policy check: the sampler's log pi(y) against the trainer's, per token
            stats["sampler_trainer_logprob_gap_per_token"] = float(np.sum(np.abs(np.array(sums) - np.array(sampler.logprob_sums))) / max(1, sum(len(c) for c in flat_c)))
        if args.mode != "bestofn":
            stats["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(pol.params, 1.0))
            optimizer.step()
        t3 = time.time()
        tokens = [len(c) for c in flat_c]
        log.write(json.dumps({"step": step, "mode": args.mode, "behaviors": len(chosen), "programs": len(items), "mean_bits": float(S.mean()), "best_bits": float(S.min(1).mean()),
                              "worst_bits": float(S.max(1).mean()), "valid_fraction": float(valid.mean()), "mean_completion_tokens": float(np.mean(tokens)), **stats,
                              "sampling": getattr(sampler, "stats", {}), "seconds": {"sample": t1 - t0, "score": t2 - t1, "train": t3 - t2}, "elapsed": time.time() - started,
                              "example": items[int(np.where(valid, S, np.inf).reshape(-1).argmin())]["source"][:2000]}) + "\n")
        log.flush()
    pol.save(adapter)
    if args.eval_every:
        evaluate(sets, pol, sampler, score, args, adapter, args.steps, eval_log, args.steps)


if __name__ == "__main__":
    main()

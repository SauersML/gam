"""The graph oracle's program training (#2951): sample programs, score them in bits, train on the score.

Each step takes K behaviors from the train split. For each, the oracle's input (prompt.py's render, as a
Qwen3 chat user turn, thinking off) gets N programs sampled at temperature 1: by vLLM serving the
current LoRA adapter when it is installed (a GPU), by transformers' generate otherwise. A program is
prompt.program_of(reply): the last fenced python block that parses, else the reply. scorer.py scores every
program: S = total bits (lower is better). One update follows, by --mode:

  bestofn  SFT on each behavior's best valid program (lowest S):
             loss = -(1/B) sum_e sum_t log pi(y_et | x_e, y_e<t).
  dpo      the pair (best, worst) of each behavior whose S differ, log pi(y) = sum_t log pi(y_t):
             loss = -(1/P) sum log sigmoid(beta [(log pi(y_w) - log pi_ref(y_w)) - (log pi(y_l) - log pi_ref(y_l))]).
  grpo     reward r = -S, advantage A_e = (r_e - mean_g r) / std_g r within the behavior's group
           (0 when the group's scores are equal):
             loss = -(1/E) sum_e A_e sum_t log pi(y_et) + beta (1/E) sum_e sum_t k3_et,
           k3 = exp(d) - d - 1 >= 0 with d = log pi_ref(y_t) - log pi(y_t), the per-token estimate of
           KL(pi || pi_ref), summed over the episode's tokens like the log-probabilities.

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
os.environ.setdefault("MPD_MEM_GIB", "1")  # mech.trace's child is a venv script, which otherwise waits for the venv default of 4 GiB of the Mac's memory ledger
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
    references (the empty and the full program, e2e/programs.py) and the search baseline's final
    programs (e2e/search.py's runs/search/<behavior>.<mode>.json)."""
    sys.path.insert(0, str(HERE.parent / "e2e"))
    import programs

    refs = programs.references(b["model"])
    out = {"empty": refs["empty"], "full": refs["full"]}
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
        """log pi(y_t | x, y_<t) at every completion token: (B, T) log-probabilities and a 0/1 mask,
        right-padded. The output layer (151,936 wide) runs only at completion tokens, in checkpointed
        chunks. ref=True evaluates pi_ref without gradients."""
        from torch.utils.checkpoint import checkpoint

        width = max(len(p) + len(c) for p, c in zip(prompts, completions))
        ids = torch.zeros(len(prompts), width, dtype=torch.long)
        att = torch.zeros(len(prompts), width, dtype=torch.long)
        comp = torch.zeros(len(prompts), width, dtype=torch.bool)
        for r, (p, c) in enumerate(zip(prompts, completions)):
            ids[r, : len(p) + len(c)] = torch.tensor(p + c)
            att[r, : len(p) + len(c)] = 1
            comp[r, len(p) : len(p) + len(c)] = True
        ids, att, comp = ids.to(self.dev), att.to(self.dev), comp.to(self.dev)
        causal = self.model.base_model.model
        rows, cols = comp[:, 1:].nonzero(as_tuple=True)
        target = ids[:, 1:][rows, cols]

        def run():
            hidden = causal.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
            flat = hidden[rows, cols]

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
        out = torch.zeros(ids.shape[0], width - 1, device=self.dev, dtype=torch.float32).index_put((rows, cols), lp)
        return out, comp[:, 1:].float()

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

    def __init__(self, policy: Policy, max_tokens: int):
        self.policy, self.max_tokens = policy, max_tokens
        self.logprob_sums = None

    @torch.no_grad()
    def __call__(self, prompts: list[list[int]], n: int, adapter: Path, version: int) -> list[list[list[int]]]:
        pol = self.policy
        pol.train_mode(False)
        out = []
        for p in prompts:
            ids = torch.tensor([p] * n, device=pol.dev)
            gen = pol.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=self.max_tokens, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                     eos_token_id=pol.end, pad_token_id=pol.end)[:, len(p) :].tolist()
            out.append([g[: g.index(pol.end) + 1] if pol.end in g else g for g in gen])
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
        self.logprob_sums = [sum(d[t].logprob for d, t in zip(c.logprobs, c.token_ids)) for o in outs for c in o.outputs]
        return [[list(c.token_ids) for c in o.outputs] for o in outs]


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
            ref, _ = pol.token_logprobs(ps, cs, ref=True)
            d = ref - cur
            k3 = ((torch.expm1(d) - d) * mask).sum(1)  # exp(d) - d - 1 without the rounding of 1 + O(d)
            loss = loss + beta * k3.sum() / len(prompts)
            kl += float(k3.detach().sum()) / len(prompts)
        loss.backward()
        total += float(loss.detach())
    return {"loss": total, "kl_sum_per_episode": kl, "logprob_sums": sums}


def evaluate(sets: dict[str, list[dict]], pol, sampler, score, args, adapter: Path, version: int, log, step: int) -> dict:
    """Programs of the current policy on each evaluation set (N samples per behavior at temperature 1)
    and the baselines, all under one experiment seed (--eval-seed, never a training step's). Per set: the
    mean S of a single sample (the oracle's expected score), the best of N, the validity, and each
    baseline's S, averaged over the behaviors that have it."""
    summary = {}
    for name, pool in sets.items():
        if not pool:
            continue
        prompts = [pol.prompt_ids(render(b)) for b in pool]
        groups = sampler(prompts, args.samples, adapter, version)
        items = [{"source": program_of(pol.tok.decode(c, skip_special_tokens=True)), "behavior": b, "seed": args.eval_seed} for b, g in zip(pool, groups) for c in g]
        base = [(b, n, src) for b in pool for n, src in (baselines(b).items() if args.baselines else [])]
        scores = score(items + [{"source": src, "behavior": b, "seed": args.eval_seed} for b, _, src in base])
        S = np.array([x["total_bits"] for x in scores[: len(items)]], dtype=float).reshape(len(pool), args.samples)
        valid = np.array([bool(x["valid"]) for x in scores[: len(items)]]).reshape(len(pool), args.samples)
        per_base = {}
        for (b, n, _), x in zip(base, scores[len(items) :]):
            per_base.setdefault(b["id"], {})[n] = x["total_bits"]
        for g, b in enumerate(pool):
            log.write(json.dumps({"set": name, "step": step, "behavior": b["id"], "mean_bits": float(S[g].mean()), "best_bits": float(S[g].min()), "valid_fraction": float(valid[g].mean()),
                                  "baselines": per_base.get(b["id"], {}), "best_source": items[g * args.samples + int(S[g].argmin())]["source"]}) + "\n")
        names = sorted({n for d in per_base.values() for n in d})
        summary[name] = {"behaviors": len(pool), "mean_bits": float(S.mean()), "best_of_n_bits": float(S.min(1).mean()), "valid_fraction": float(valid.mean()),
                         "baselines": {n: float(np.mean([d[n] for d in per_base.values() if n in d])) for n in names},
                         "oracle_mean_bits_on_baseline_behaviors": {n: float(np.mean([S[g].mean() for g, b in enumerate(pool) if n in per_base.get(b["id"], {})])) for n in names}}
    log.write(json.dumps({"summary": summary, "step": step}) + "\n")
    log.flush()
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["grpo", "dpo", "bestofn", "eval"], required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--init", help="SFT adapter the policy starts from and is held to (pi_ref): a PEFT directory or g-predict's sft.py output")
    ap.add_argument("--model", required=True, help="target model whose behaviors are explained: qwen3-0.6b | vpd4l")
    ap.add_argument("--behaviors", default=str(BEHAVIORS))
    ap.add_argument("--scorer", choices=sorted(SCORERS), default="checker")
    ap.add_argument("--score-workers", type=int, default=1, help="checker servers per target model, each scoring whole behaviors in parallel")
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
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--sampler", choices=["auto", "vllm", "hf"], default="auto")
    ap.add_argument("--gpu-memory", type=float, default=0.85, help="vLLM's share of its GPU (lower it when the trainer shares the GPU)")
    ap.add_argument("--prompt-holdout", type=int, default=4, help="every K-th prompt of each training behavior is held out for evaluation (0: none)")
    ap.add_argument("--eval-every", type=int, default=0, help="evaluate every E training steps and at the end (0: only --mode eval)")
    ap.add_argument("--eval-seed", type=int, default=1_000_003, help="the evaluation's experiment seed (training steps use their index)")
    ap.add_argument("--no-baselines", dest="baselines", action="store_false", help="skip scoring the empty, full and search programs in evaluation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lr = args.lr if args.lr is not None else {"bestofn": 1e-4}.get(args.mode, 1e-5)
    beta = args.beta if args.beta is not None else {"dpo": 0.1}.get(args.mode, 0.04)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    args.init = init_adapter(args.init, out)
    (out / "config.json").write_text(json.dumps({**vars(args), "lr": lr, "beta": beta}, indent=1))

    use_vllm = args.sampler == "vllm" or (args.sampler == "auto" and torch.cuda.is_available() and __import__("importlib").util.find_spec("vllm") is not None)
    sampler = None
    if use_vllm:  # vLLM first, on the first visible GPU, before the trainer touches CUDA
        from transformers import AutoTokenizer

        rank = json.loads((Path(args.init) / "adapter_config.json").read_text())["r"] if args.init else args.lora_rank
        sampler = VllmSampler(args, rank, AutoTokenizer.from_pretrained(args.base).convert_tokens_to_ids("<|im_end|>"))
    dev = torch.device(f"cuda:{torch.cuda.device_count() - 1}" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    pol = Policy(args, dev)
    if sampler is None:
        sampler = HfSampler(pol, args.max_tokens)
    score = SCORERS[args.scorer]
    scorer.WORKERS = args.score_workers
    root = Path(args.behaviors)
    pool, heldout_prompts = split_prompts(behaviors(root, args.model, "train"), args.prompt_holdout, out / "behaviors")
    sets = {"heldout_behaviors": behaviors(root, args.model, "heldout"), "heldout_prompts": heldout_prompts}
    adapter = out / "adapter"
    if args.mode == "eval":
        pol.save(adapter)
        print(json.dumps(evaluate(sets, pol, sampler, score, args, adapter, 0, open(out / "eval.jsonl", "a"), 0)))
        return
    if not pool:
        raise SystemExit(f"no train behaviors under {root / args.model}")
    optimizer = torch.optim.AdamW(pol.params, lr=lr, weight_decay=0.0)
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
        items = [{"source": program_of(t), "behavior": b, "seed": step} for b, ts in zip(chosen, texts) for t in ts]
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
        if args.mode == "grpo":
            r = -S
            std = r.std(1, keepdims=True)
            adv = np.where(std > 0, (r - r.mean(1, keepdims=True)) / np.where(std > 0, std, 1.0), 0.0).reshape(-1)
            stats = grpo_update(pol, flat_p, flat_c, adv.tolist(), beta, args.micro)
        elif args.mode == "dpo":
            pairs = [(g, int(S[g].argmin()), int(S[g].argmax())) for g in range(len(chosen)) if S[g].max() > S[g].min()]
            stats = dpo_update(pol, [prompts[g] for g, _, _ in pairs], [groups[g][w] for g, w, _ in pairs], [groups[g][l] for g, _, l in pairs], beta, args.micro) if pairs else {}
            stats["pairs"] = len(pairs)
        else:
            keep = [(g, int(np.where(valid[g], S[g], np.inf).argmin())) for g in range(len(chosen)) if valid[g].any()]
            for g, j in keep:
                best_log.write(json.dumps({"behavior": chosen[g]["id"], "prompt": render(chosen[g]), "completion": texts[g][j], "score": scores[g * args.samples + j]}) + "\n")
            best_log.flush()
            stats = {}
            for _ in range(args.sft_epochs if keep else 0):
                stats = sft_update(pol, [prompts[g] for g, _ in keep], [groups[g][j] for g, j in keep], args.micro)
                torch.nn.utils.clip_grad_norm_(pol.params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            stats["kept"] = len(keep)
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
                              "seconds": {"sample": t1 - t0, "score": t2 - t1, "train": t3 - t2}, "elapsed": time.time() - started,
                              "example": items[int(S.reshape(-1).argmin())]["source"][:2000]}) + "\n")
        log.flush()
    pol.save(adapter)
    if args.eval_every:
        evaluate(sets, pol, sampler, score, args, adapter, args.steps, eval_log, args.steps)


if __name__ == "__main__":
    main()

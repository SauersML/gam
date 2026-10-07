"""The graph oracle's program training (#2951): sample programs, score them in bits, train on the score.

Each step takes K behaviors from the train split. For each, the oracle's input (prompt.py's render, as a
Qwen3 chat user turn, thinking off) gets N programs sampled at temperature 1: by vLLM serving the
current LoRA adapter when it is installed (a GPU), by transformers' generate otherwise. A program is
the first fenced code block of the reply (the whole reply when there is none). scorer.py scores every
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
  train.py --mode eval --base ... --init ADAPTER --model ... --out DIR   (held-out behaviors, S per program)

Outputs: DIR/train.jsonl (a line per step: scores, validity, loss, KL, seconds sampling / scoring /
training), DIR/samples.jsonl (every program with its score, for repair data and offline SFT),
DIR/best.jsonl (bestofn: the kept programs), DIR/adapter (the policy).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from scorer import SCORERS  # noqa: E402

BEHAVIORS = Path.home() / "mpd-data/graph_oracle/behaviors"


def render(behavior: dict) -> str:
    try:
        import prompt
    except ImportError:  # prompt.py has not landed: description and examples only
        lines = [f"Target model: {behavior['model']}. Behavior: {behavior['description']}", "Example prompts:"]
        lines += [f"- {p['text']!r}" for p in behavior["prompts"][:4]]
        lines.append("Write one Python program using only `from mech import ...` that explains how the model produces this behavior.")
        return "\n".join(lines)
    return prompt.render(behavior)


def program_of(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:python)?\n(.*?)(?:```|$)", text, flags=re.S)
    return (m.group(1) if m else text).strip() + "\n"


def behaviors(root: Path, model: str, split: str) -> list[dict]:
    out = []
    for p in sorted((root / model).glob("*.json")):
        b = json.loads(p.read_text())
        if b.get("split", "train") == split:
            b["path"] = str(p)
            out.append(b)
    if not out:
        raise SystemExit(f"no {split} behaviors under {root / model}")
    return out


class Policy:
    """The trainable LoRA policy (adapter "default") and its frozen reference (adapter "ref" = the
    --init adapter, or the base with the adapter disabled)."""

    def __init__(self, args, dev):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.dev = dev
        self.tok = AutoTokenizer.from_pretrained(args.base)
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
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

        params = SamplingParams(n=n, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=self.max_tokens, stop_token_ids=[self.end])
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, lora_request=LoRARequest(f"policy{version}", version + 1, str(adapter)), use_tqdm=False)
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
    total, kl = 0.0, 0.0
    for idx in micro_batches(len(prompts), micro):
        ps, cs = [prompts[i] for i in idx], [completions[i] for i in idx]
        adv = torch.tensor([advantage[i] for i in idx], device=pol.dev, dtype=torch.float32)
        cur, mask = pol.token_logprobs(ps, cs)
        loss = -(adv * (cur * mask).sum(1)).sum() / len(prompts)
        if beta > 0:
            ref, _ = pol.token_logprobs(ps, cs, ref=True)
            d = ref - cur
            k3 = ((torch.expm1(d) - d) * mask).sum(1)  # exp(d) - d - 1 without the rounding of 1 + O(d)
            loss = loss + beta * k3.sum() / len(prompts)
            kl += float(k3.detach().sum()) / len(prompts)
        loss.backward()
        total += float(loss.detach())
    return {"loss": total, "kl_sum_per_episode": kl}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["grpo", "dpo", "bestofn", "eval"], required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--init", help="SFT adapter the policy starts from and is held to (pi_ref)")
    ap.add_argument("--model", required=True, help="target model whose behaviors are explained: qwen3-0.6b | vpd4l")
    ap.add_argument("--behaviors", default=str(BEHAVIORS))
    ap.add_argument("--scorer", choices=sorted(SCORERS), default="checker")
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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lr = args.lr if args.lr is not None else {"bestofn": 1e-4}.get(args.mode, 1e-5)
    beta = args.beta if args.beta is not None else {"dpo": 0.1}.get(args.mode, 0.04)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
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
    pool = behaviors(Path(args.behaviors), args.model, "heldout" if args.mode == "eval" else "train")
    optimizer = torch.optim.AdamW(pol.params, lr=lr, weight_decay=0.0)
    log = open(out / "train.jsonl" if args.mode != "eval" else out / "eval.jsonl", "a")
    samples_log = open(out / "samples.jsonl", "a")
    best_log = open(out / "best.jsonl", "a") if args.mode == "bestofn" else None
    adapter = out / "adapter"
    started = time.time()
    steps = 1 if args.mode == "eval" else args.steps
    for step in range(steps):
        if args.hours and time.time() - started > 3600 * args.hours:
            break
        pol.save(adapter)
        t0 = time.time()
        chosen = pool if args.mode == "eval" else random.sample(pool, min(args.behaviors_per_step, len(pool)))
        prompts = [pol.prompt_ids(render(b)) for b in chosen]
        groups = sampler(prompts, args.samples, adapter, step)
        t1 = time.time()
        texts = [[pol.tok.decode(c, skip_special_tokens=True) for c in g] for g in groups]
        items = [{"source": program_of(t), "behavior": b} for b, ts in zip(chosen, texts) for t in ts]
        scores = score(items)
        t2 = time.time()
        S = np.array([s["total_bits"] for s in scores], dtype=float).reshape(len(chosen), args.samples)
        valid = np.array([bool(s["valid"]) for s in scores]).reshape(len(chosen), args.samples)
        for (b, it, s, c) in zip([b for b in chosen for _ in range(args.samples)], items, scores, [c for g in groups for c in g]):
            samples_log.write(json.dumps({"step": step, "behavior": b["id"], "source": it["source"], "completion_tokens": len(c), "score": s}) + "\n")
        samples_log.flush()
        flat_p = [p for p in prompts for _ in range(args.samples)]
        flat_c = [c for g in groups for c in g]
        stats = {}
        if args.mode != "eval":
            pol.train_mode(True)
            optimizer.zero_grad(set_to_none=True)
            if args.mode == "grpo":
                r = -S
                std = r.std(1, keepdims=True)
                adv = np.where(std > 0, (r - r.mean(1, keepdims=True)) / np.where(std > 0, std, 1.0), 0.0).reshape(-1)
                stats = grpo_update(pol, flat_p, flat_c, adv.tolist(), beta, args.micro)
            elif args.mode == "dpo":
                pairs = [(g, int(S[g].argmin()), int(S[g].argmax())) for g in range(len(chosen)) if S[g].max() > S[g].min()]
                if pairs:
                    stats = dpo_update(pol, [prompts[g] for g, _, _ in pairs], [groups[g][w] for g, w, _ in pairs], [groups[g][l] for g, _, l in pairs], beta, args.micro)
                stats["pairs"] = len(pairs)
            else:
                keep = [(g, int(np.where(valid[g], S[g], np.inf).argmin())) for g in range(len(chosen)) if valid[g].any()]
                for g, j in keep:
                    best_log.write(json.dumps({"behavior": chosen[g]["id"], "prompt": render(chosen[g]), "completion": texts[g][j], "score": scores[g * args.samples + j]}) + "\n")
                best_log.flush()
                if keep:
                    for _ in range(args.sft_epochs):
                        stats = sft_update(pol, [prompts[g] for g, _ in keep], [groups[g][j] for g, j in keep], args.micro)
                        torch.nn.utils.clip_grad_norm_(pol.params, 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                stats["kept"] = len(keep)
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
    if args.mode != "eval":
        pol.save(adapter)


if __name__ == "__main__":
    main()

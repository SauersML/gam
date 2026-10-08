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
  rl2      RL v2 (rl2_step), the same score with a better gradient estimate: a fixed scale per behavior,
           scale_b = the teacher answer's total score (--teacher; the empty program's without one), and the RLOO
           advantage A_e = (mean of the group's other S - S_e) / scale_b, no per-group std (3); --ppo-epochs
           clipped updates per scored batch (4); groups without signal dropped and refilled by gap to the teacher
           (5); a fresh experiment seed per step (7); per-token credit from measured one-edit neighbours,
           edits.credit (1); expert iteration from each group's best answer, edits.refine (2).
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
import bisect
import contextlib
import json
import math
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
sys.path.insert(0, str(HERE.parent / "predict"))  # g-predict's part_tokens
import edits  # noqa: E402
from prompt import program_of, render, split_answer  # noqa: E402
import scorer  # noqa: E402
from scorer import SCORERS  # noqa: E402

BEHAVIORS = Path.home() / "mpd-data/graph_oracle/behaviors"
SEARCH = Path.home() / "mpd-data/graph_oracle/runs/search"
TEACHER: dict[str, str] = {}  # --teacher's answers by behavior id (teacher_answers): training behaviors
HELDOUT_TEACHER: dict[str, str] = {}  # --teacher-heldout's: evaluation baselines only, never SFT or RL


def item(answer: str, behavior: dict, seed: int, uniform_seeds: int, experiments: int) -> dict:
    """A scoring item from an oracle answer: prompt.split_answer's program (the last python block that
    parses) and explanation (the plain English after it, which alone the reader reads). Part tokens stay as
    written: mech parses them, and each counts as one Python token of the program's size."""
    source, explanation = split_answer(answer)
    return {"source": source, "explanation": explanation, "behavior": behavior, "seed": seed, "uniform_seeds": uniform_seeds, "experiments": experiments}


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
    """Answers the oracle is compared with on behavior b, scored on the same experiments: g-int's
    references (the empty and the full program, e2e/programs.py), g-mech's example programs for b
    (examples/index.json), the search baseline's final programs (runs/search/<behavior>.<mode>.json) and the
    teacher answer (--teacher, program and English)."""
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
    if b["id"] in TEACHER or b["id"] in HELDOUT_TEACHER:
        out["teacher"] = TEACHER.get(b["id"]) or HELDOUT_TEACHER[b["id"]]
    return out


def load_parts(spec: str, init: str | None, model, base_vocab: int, dev):
    """g-predict's part_tokens.PartTokens over the registry at SPEC (part_tokens.py build), scaled to the
    mean RMS of the base token embeddings, with projections resumed from an --init adapter's part_tokens.pt."""
    import part_tokens

    emb = model.get_input_embeddings().weight[:base_vocab].detach().float()
    parts = part_tokens.PartTokens(part_tokens.Registry.load(spec), model.config.hidden_size, float(emb.pow(2).mean(-1).sqrt().mean()), base_vocab, dev)
    if init and (Path(init) / "part_tokens.pt").exists():
        parts.load_state_dict(torch.load(Path(init) / "part_tokens.pt", map_location="cpu"))
    return parts


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
        self.parts, self.first_part = None, None
        if getattr(args, "part_tokens", None):  # part tokens: rows from the parts' read/write vectors (part_vocab.py)
            import part_vocab

            self.parts = load_parts(args.part_tokens, args.init, base, len(self.tok), dev).to(dev)
            self.first_part = part_vocab.install(base, self.tok, self.parts)
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
        if self.parts is not None:  # the projections train with the LoRA
            self.params += list(self.parts.parameters())
        for p in self.params:
            p.requires_grad_(True)
        self.end = self.tok.convert_tokens_to_ids("<|im_end|>")

    def param_groups(self, lr: float) -> list[dict]:
        """AdamW groups: the LoRA at lr, each linear map of the part tokens at lr * rank / fan_in. Adam moves every
        entry by about lr per step, so a linear map's output moves by about lr * fan_in per unit input; the scale
        gives the maps (fan_in = a part kind's feature width, 1,538-3,842 for vpd4l, or their inner rank) the
        per-step output change of the LoRA's up-projections (fan_in = rank). With one lr for both, the first SFT
        step on Qwen3-8B (lr 1e-4) moved every part's output row toward the same hidden state and the targets
        went from 4.9 to 69.5 bits per token."""
        groups = [{"params": [p for n, p in self.model.named_parameters() if ".default." in n], "lr": lr}]
        if self.parts is not None:
            groups += [{"params": list(m.parameters()), "lr": lr * self.rank / m.in_features} for m in self.parts.modules() if isinstance(m, torch.nn.Linear)]
        return groups

    def prompt_ids(self, text: str) -> list[int]:
        chat = self.tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
        return self.tok.encode(chat, add_special_tokens=False)

    def save(self, path: Path):
        self.model.save_pretrained(str(path), selected_adapters=["default"])
        if self.parts is not None:
            torch.save(self.parts.state_dict(), Path(path) / "part_tokens.pt")

    @torch.no_grad()
    def part_rows(self) -> tuple[int, torch.Tensor, torch.Tensor]:
        return self.first_part, self.parts.input_rows().detach().to(torch.bfloat16), self.parts.output_rows().detach().to(torch.bfloat16)

    def materialize(self, out_dir: Path, base_name: str) -> Path:
        """A base checkpoint carrying the current part rows (vLLM samples from it)."""
        import part_vocab

        return part_vocab.materialize(self.model.base_model.model, self.tok, self.parts, self.first_part, base_name, out_dir)

    def token_logprobs(self, prompts: list[list[int]], completions: list[list[int]], ref: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """log pi(y_t | x, y_<t) at every completion token: (B, T) log-probabilities and a 0/1 mask. The
        output layer (151,936 wide) runs only at completion tokens, in checkpointed chunks. ref=True
        evaluates pi_ref without gradients. Sequences are right-padded to the batch's longest. (Packing a group into
        one sequence, the shared prompt once under a 4-D mask, was 2.2x slower at 1,500-token completions and 4.5x
        at 3,000 on Qwen3-8B (A40, oracle-train-g-rl-packbench2): the dense mask leaves the fused attention kernel.)"""
        from torch.utils.checkpoint import checkpoint

        causal = self.model.base_model.model
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
            if self.parts is None:
                def piece(h, t):
                    return torch.log_softmax(causal.lm_head(h).float(), -1).gather(-1, t[:, None])[:, 0]

                return torch.cat([checkpoint(piece, flat[k : k + 1024], target[k : k + 1024], use_reentrant=False) for k in range(0, len(target), 1024)])
            head = causal.lm_head  # part tokens: the output rows enter each checkpointed chunk as an input, so its recompute uses them
            rows_out = head.cached if head.cached is not None else head.parts.output_rows()

            def piece_parts(h, t, rows):
                logits = head.base(h)
                part = h @ rows.to(h.dtype).T
                end = head.first + part.shape[-1]
                logits = torch.cat([logits[..., : head.first], part, logits[..., end:]], -1)
                return torch.log_softmax(logits.float(), -1).gather(-1, t[:, None])[:, 0]

            return torch.cat([checkpoint(piece_parts, flat[k : k + 1024], target[k : k + 1024], rows_out, use_reentrant=False) for k in range(0, len(target), 1024)])

        if self.parts is not None:  # one computation of the part rows for this call (forward, chunks, backward)
            import part_vocab

            once = part_vocab.rows_once(causal)
            once.__enter__()
        try:
            lp = self._run_logprobs(run, ref)
        finally:
            if self.parts is not None:
                once.__exit__(None, None, None)
        out = torch.zeros(shape, device=self.dev, dtype=torch.float32).index_put((rows, cols), lp)
        mask = torch.zeros(shape, device=self.dev).index_put((rows, cols), torch.ones_like(lp))
        return out, mask

    def _run_logprobs(self, run, ref: bool) -> torch.Tensor:
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
        return lp

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
        self.logprob_sums = self.token_logprobs = None

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


def grammar_options(grammar: str) -> tuple[dict, dict]:
    """(LLM keywords, SamplingParams keywords) for decoding under an xgrammar EBNF grammar: structured_outputs in
    vLLM's newer API (0.19), guided_decoding in the older one (0.10)."""
    from vllm import sampling_params as sp

    if hasattr(sp, "StructuredOutputsParams"):
        return {"structured_outputs_config": {"backend": "xgrammar"}}, {"structured_outputs": sp.StructuredOutputsParams(grammar=grammar)}
    return {"guided_decoding_backend": "xgrammar"}, {"guided_decoding": sp.GuidedDecodingParams(grammar=grammar)}


class VllmSampler:
    """vLLM serving the base with the policy's adapter (reloaded by path at every version). It takes the
    first visible GPU; the trainer takes the second when there is one (--gpu-memory set accordingly). With a
    grammar (--grammar, rl/grammar.py) every answer is decoded under it."""

    def __init__(self, args, rank: int, end: int, model: str | None = None, grammar: str | None = None):
        self.engine_options, self.sample_options = grammar_options(grammar) if grammar else ({}, {})
        self.share, self.args, self.rank = args.share_gpu, args, rank
        self.max_tokens, self.end = args.max_tokens, end
        self.policy = None  # with --share-gpu: the trainer, moved to the host while vLLM samples
        self.rows = None  # with part tokens: () -> (first id, input rows, output rows), copied into vLLM before sampling
        self.llm = None
        self.held, self.ready = 0, False  # hold(): calls in a row share one wake (and one push of the rows)
        self.logprob_sums = self.token_logprobs = None  # the last call's sampled tokens' log-probabilities, flat over prompts x n
        if model is not None:  # else started later (part tokens: from the extended-vocabulary checkpoint)
            self.start(model)

    def start(self, model: str):
        from vllm import LLM

        a = self.args
        self.llm = LLM(model=model, dtype="bfloat16", enable_lora=True, max_lora_rank=self.rank, max_loras=1, enable_prefix_caching=True,
                       gpu_memory_utilization=a.gpu_memory, max_model_len=a.max_model_len, seed=a.seed, enable_sleep_mode=self.share, **self.engine_options)
        if self.share:
            self.llm.sleep(level=1)  # weights to host memory, KV cache freed: the trainer loads next

    def push_rows(self, rows=None):
        """Copies the part tokens' current rows into vLLM's embedding and output layer in place (an in-process
        engine, VLLM_ENABLE_V1_MULTIPROCESSING=0), so sampling uses this step's projections without
        rewriting the checkpoint; the prefix cache is reset since prompts may hold part tokens."""
        first, rows_in, rows_out = rows if rows is not None else self.rows()

        def put(model):
            model.model.embed_tokens.weight.data[first : first + rows_in.shape[0]].copy_(rows_in)
            model.lm_head.weight.data[first : first + rows_out.shape[0]].copy_(rows_out)

        self.llm.apply_model(put)
        self.llm.reset_prefix_cache()

    def reload(self, model: Path):
        """Restarts vLLM on a materialized checkpoint (part tokens' current rows)."""
        import gc

        if self.llm is not None and self.share:
            raise RuntimeError("vLLM's sleep mode allows one engine per process: no restart with --share-gpu")
        self.llm, self.ready = None, False
        gc.collect()
        torch.cuda.empty_cache()
        if self.share and self.policy is not None:
            self.policy.model.to("cpu")
            torch.cuda.empty_cache()
        self.start(str(model))
        if self.share and self.policy is not None:
            self.policy.model.to(self.policy.dev)

    def acquire(self):
        """Readies vLLM to sample: the part rows are computed where the trainer's projections live, then (one GPU)
        the trainer's weights leave while vLLM wakes with its whole share, and the rows are copied in."""
        rows = self.rows() if self.rows is not None else None
        if self.share:
            self.policy.model.to("cpu")
            torch.cuda.empty_cache()
            self.llm.wake_up()
        if rows is not None:
            self.push_rows(rows)
        self.ready = True

    def release(self):
        if self.ready and self.share:
            self.llm.sleep(level=1)
            self.policy.model.to(self.policy.dev)
        self.ready = False

    @contextlib.contextmanager
    def hold(self):
        """Sampling calls inside share one wake: vLLM stays awake and the trainer on the host between them (the
        validity redraws; each wake and sleep moves the trainer's weights, 16 GB for Qwen3-8B, both ways)."""
        self.held += 1
        try:
            yield
        finally:
            self.held -= 1
            if not self.held:
                self.release()

    def __call__(self, prompts: list[list[int]], n: int, adapter: Path, version: int) -> list[list[list[int]]]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        if not self.ready:
            self.acquire()
        params = SamplingParams(n=n, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=self.max_tokens, stop_token_ids=[self.end], logprobs=0, **self.sample_options)
        try:
            outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, lora_request=LoRARequest(f"policy{version}", version + 1, str(adapter)), use_tqdm=False)
        finally:
            if not self.held:
                self.release()
        try:  # the sampled tokens' log-probabilities: the behavior policy of rl2's importance weights, and the on-policy check
            self.token_logprobs = [[d[t].logprob for d, t in zip(c.logprobs, c.token_ids)] for o in outs for c in o.outputs]
            self.logprob_sums = [sum(x) for x in self.token_logprobs]
        except (TypeError, KeyError, AttributeError):  # a vLLM whose logprobs container differs
            self.logprob_sums = self.token_logprobs = None
        return [[list(c.token_ids) for c in o.outputs] for o in outs]


class ValidSampler:
    """Draws n programs per prompt and redraws each invalid one (mech.trace: syntax, unknown names, an index
    beyond its site's size, a rule broken) up to `rounds` times, so the policy is sampled restricted to the
    programs it can write validly; stats holds the valid share before and after the redraws."""

    def __init__(self, inner, tok, model: str, rounds: int):
        self.inner, self.tok, self.model, self.rounds = inner, tok, model, rounds
        self.logprob_sums, self.token_logprobs, self.stats = None, None, {}

    def valid(self, completions: list[list[int]]) -> list[bool]:
        import mech
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(8) as ex:  # each trace is a fork of a tracer server
            return list(ex.map(lambda c: bool(mech.trace(program_of(self.tok.decode(c, skip_special_tokens=True)), self.model)["valid"]), completions))

    def __call__(self, prompts: list[list[int]], n: int, adapter: Path, version: int) -> list[list[list[int]]]:
        with self.inner.hold() if hasattr(self.inner, "hold") else contextlib.nullcontext():
            return self.draw(prompts, n, adapter, version)

    def draw(self, prompts: list[list[int]], n: int, adapter: Path, version: int) -> list[list[list[int]]]:
        groups = self.inner(prompts, n, adapter, version)
        sums = list(self.inner.logprob_sums) if self.inner.logprob_sums is not None else None
        tokens = list(self.inner.token_logprobs) if getattr(self.inner, "token_logprobs", None) is not None else None
        flags = [self.valid(g) for g in groups]
        first = float(np.mean([f for fs in flags for f in fs]))
        for _ in range(self.rounds):
            slots = [(g, j) for g in range(len(groups)) for j in range(n) if not flags[g][j]]
            if not slots:
                break
            redo = self.inner([prompts[g] for g, _ in slots], 1, adapter, version)
            redo_sums, redo_tokens = self.inner.logprob_sums, getattr(self.inner, "token_logprobs", None)
            ok = self.valid([r[0] for r in redo])
            for k, ((g, j), r) in enumerate(zip(slots, redo)):
                groups[g][j], flags[g][j] = r[0], ok[k]
                if sums is not None:
                    sums = None if redo_sums is None else sums
                    if sums is not None:
                        sums[g * n + j] = redo_sums[k]
                if tokens is not None:
                    tokens = None if redo_tokens is None else tokens
                    if tokens is not None:
                        tokens[g * n + j] = redo_tokens[k]
        self.logprob_sums, self.token_logprobs = sums, tokens
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


# RL v2 (--mode rl2): the same score; each change only sharpens the estimate of its gradient.

TOTALS = {"checker_seconds": 0.0}  # checker time of every training score, credit and refinement so far (the A/B's cost axis)


def step_seed(args, step: int) -> int:
    """(7) Training step `step`'s experiment seed: (--seed << 20) + step, a fresh experiment draw at every step, never
    --eval-seed, whose experiments only the evaluation draws."""
    seed = (args.seed << 20) + step
    return seed if seed != args.eval_seed else seed + (1 << 40)


def teacher_answers(root) -> dict[str, str]:
    """g-int's teacher answers by behavior id: DIR/manifest.jsonl's "answer" files (e2e/teacher_run.py; a behavior's
    last line wins; a path that does not exist here, e.g. on a pod, is read from DIR by its name), else every
    DIR/<behavior>.answer.txt, else a bare DIR/<behavior>.py program as the answer's python block."""
    out = {}
    if root:
        root = Path(root).expanduser()
        if (root / "manifest.jsonl").exists():
            last = {r["behavior"]: Path(r["answer"]) for r in map(json.loads, open(root / "manifest.jsonl"))}
            return {b: (path if path.exists() else root / path.name).read_text() for b, path in sorted(last.items())}
        for p in sorted(root.glob("*.answer.txt")):
            out[p.name[: -len(".answer.txt")]] = p.read_text()
        for p in sorted(root.glob("*.py")):
            out.setdefault(p.stem, "```python\n" + p.read_text().strip() + "\n```")
    return out


def refuse_heldout(paths) -> None:
    """Held-out answers (teacher_heldout/) are evaluation baselines: no training input may come from that directory."""
    for path in paths:
        if path and "teacher_heldout" in Path(path).expanduser().resolve().parts:
            raise SystemExit(f"{path}: held-out answers are for evaluation only (--teacher-heldout), never training")


class Scales:
    """(3, 5) Per behavior b: the fixed reward scale scale_b and the gap to the teacher. scale_b = the teacher answer's
    total score, measured once (with the empty program's, in the batch of the first training step that draws b); without
    a valid teacher answer, the behavior's measured signal, the empty program's total. gap_b = (the latest group's mean
    valid score - the teacher's) / scale_b (the teacher's score taken as 0 without one), at least FLOOR."""

    FLOOR = 0.05

    def __init__(self, teacher: dict[str, str]):
        self.teacher, self.scale, self.target, self.gap = teacher, {}, {}, {}

    def items(self, chosen: list[dict], seed: int, experiments: int) -> list[tuple[str, str, dict]]:
        """(behavior id, "teacher" | "empty", scoring item) for the behaviors without a scale yet."""
        sys.path.insert(0, str(HERE.parent / "e2e"))
        import programs

        out = []
        for b in chosen:
            if b["id"] in self.scale:
                continue
            if b["id"] in self.teacher:
                out.append((b["id"], "teacher", item(self.teacher[b["id"]], b, seed, 0, experiments)))
            out.append((b["id"], "empty", item(programs.empty(b["model"]), b, seed, 0, experiments)))
        return out

    def take(self, entries: list[tuple[str, str, dict]], scores: list[dict]):
        got = {}
        for (bid, kind, _), s in zip(entries, scores):
            got.setdefault(bid, {})[kind] = s
        for bid, d in got.items():
            t = d.get("teacher")
            if t is not None and t.get("valid") and t["total_bits"] > 0:
                self.scale[bid] = self.target[bid] = float(t["total_bits"])
            else:
                self.scale[bid], self.target[bid] = float(d["empty"]["total_bits"]), 0.0

    def observe(self, bid: str, S: np.ndarray, valid: np.ndarray):
        if valid.any():
            self.gap[bid] = max(self.FLOOR, (float(S[valid].mean()) - self.target[bid]) / self.scale[bid])

    def relative(self, bid: str, S: float) -> float:
        """S - the teacher's score, over scale_b (0: the teacher's score; the empty program's share without a teacher)."""
        return (S - self.target[bid]) / self.scale[bid]

    def draw(self, rest: list[dict], n: int, rng: random.Random) -> list[dict]:
        """n behaviors of `rest`, without replacement, with probabilities proportional to gap_b (unseen: the largest gap
        seen, 1 before any)."""
        gap = self.gap.copy()  # with --async the checker thread updates gaps while the sampler draws
        unseen = max(gap.values(), default=1.0)
        rest, out = list(rest), []
        for _ in range(min(n, len(rest))):
            k = rng.choices(range(len(rest)), weights=[gap.get(b["id"], unseen) for b in rest])[0]
            out.append(rest.pop(k))
        return out


def rloo(S: np.ndarray, scale: float) -> np.ndarray:
    """(3) A_i = (mean_{j != i} S_j - S_i) / scale_b: each sample against the mean score of the group's other samples
    (leave-one-out), in units of the behavior's fixed scale. No per-group standard deviation: it blew tiny score
    differences within a group of near-equal answers up to unit advantages."""
    n = len(S)
    if n < 2 or (S == S[0]).all():  # equal scores: exactly no signal (the formula leaves rounding residue)
        return np.zeros(n)
    return ((S.sum() - S) / (n - 1) - S) / scale


def program_offset(text: str, source: str) -> int | None:
    """Where split_answer's program starts in the answer text: the last fenced block whose body is `source` (0 when the
    reply has no such block and is itself the program)."""
    found, pos = None, 0
    for k, p in enumerate(text.split("```")):
        if k % 2 and "\n" in p and p.split("\n", 1)[1] == source:
            found = pos + p.index("\n") + 1
        pos += len(p) + 3
    return found if found is not None else (0 if text == source else None)


def token_spans(tok, completion: list[int], text: str) -> list[tuple[int, int]]:
    """Each completion token's characters [start, end) in text (its decode without special tokens): the offsets of the
    text's re-encoding when that gives the same tokens, else the lengths of decoded prefixes (a sampled tokenization
    that re-encoding does not reproduce). A special token gets the empty span at the previous token's end."""
    special = set(tok.all_special_ids)
    keep = [i for i, t in enumerate(completion) if t not in special]
    found = [None] * len(completion)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] == [completion[i] for i in keep]:
        for i, o in zip(keep, enc["offset_mapping"]):
            found[i] = (int(o[0]), int(o[1]))
    else:
        end = 0
        for i in keep:
            e = len(tok.decode(completion[: i + 1], skip_special_tokens=True))
            found[i] = (min(end, e), e)
            end = max(end, e)
    spans, end = [], 0
    for f in found:
        spans.append(f if f is not None else (end, end))
        end = max(end, spans[-1][1])
    return spans


def credit_advantages(tok, completion: list[int], text: str, source: str, episode: float, dS: dict, scale: float) -> list[float]:
    """(1) Per-token advantages of one answer from its measured edits (edits.credit: dS = S(edited) - S(answer), so
    dS > 0 means the choice lowers the score): the tokens of a part in an align/claim statement get dS(drop the part) /
    scale_b, the statement's other tokens (and its parts whose drop was not sampled) dS(drop the statement) / scale_b,
    and every other token the episode's advantage. An edit that makes the answer invalid (dS = +inf) counts as
    dS = scale_b."""
    adv = [episode] * len(completion)
    offset = program_offset(text, source)
    if not dS or offset is None:
        return adv

    def value(e):
        v = dS.get(e)
        if v is None or math.isnan(v):
            return None
        return math.copysign(1.0, v) if math.isinf(v) else v / scale

    spans = token_spans(tok, completion, text)
    ends = [e for _, e in spans]  # nondecreasing

    def mark(c0: int, c1: int, v):
        if v is None:
            return
        for t in range(bisect.bisect_right(ends, c0), len(spans)):  # the first token ending after c0 on
            if spans[t][0] >= c1:
                break
            if spans[t][1] > spans[t][0]:
                adv[t] = v

    lines = source.rstrip("\n").split("\n")
    starts = np.cumsum([0] + [len(line) + 1 for line in lines])
    for s in edits.Answer.parse(source).statements:
        a0, line = offset + int(starts[s.line]), lines[s.line]
        mark(a0, a0 + len(line), value(edits.Edit("unalign", s.variable, s.kind)))
        for m in edits.PART.finditer(line):
            mark(a0 + m.start(), a0 + m.end(), value(edits.Edit("drop", s.variable, s.kind, m.group())))
    return adv


def edit_item(source: str, behavior: dict, seed: int, experiments: int) -> dict:
    """A scoring item of an edited answer: the reader off, since an edit keeps the answer's explanation."""
    return {"source": source, "explanation": "", "behavior": behavior, "seed": seed, "experiments": experiments, "reader": False}


def memo(score):
    """score with every distinct item (behavior, program, explanation, seed, experiments, reader, options) scored once:
    within one step the credit's base answers, refinement's start and edits that coincide repeat."""
    cache, hits = {}, [0]

    def key(it):
        return (it["behavior"].get("path", it["behavior"]["id"]), it["source"], it.get("explanation", ""), it.get("seed"), it.get("uniform_seeds") or 0, it.get("experiments"),
                it.get("reader", True), json.dumps(it.get("options"), sort_keys=True))

    def run(items):
        keys = [key(it) for it in items]
        first = {}
        for k, it in zip(keys, items):
            if k not in cache:
                first.setdefault(k, it)
        hits[0] += len(items) - len(first)
        if first:
            for k, r in zip(first, score(list(first.values()))):
                cache[k] = r
        return [cache[k] for k in keys]

    run.hits = hits
    return run


def timed(clock: dict, key: str, fn, *a, **kw):
    t = time.time()
    try:
        return fn(*a, **kw)
    finally:
        clock[key] += time.time() - t
        if key != "sample":
            TOTALS["checker_seconds"] += time.time() - t


def credit_groups(groups: list[dict], seed: int, args, tok, score, scales: Scales, clock: dict):
    """(1) The credit edits (edits.credit_edits: --credit part drops sampled, every statement drop) of each group's
    distinct valid answers (--credit-answers: the best and K - 1 others at random), scored in one call under the step's
    seed, then each credited answer's token advantages."""
    requests, entries = [], []
    for g, grp in enumerate(groups):
        rng = random.Random(seed * 1009 + g)
        order = np.argsort(np.where(grp["valid"], grp["S"], np.inf), kind="stable")  # the best valid answer first
        sources = list(dict.fromkeys(grp["items"][j]["source"] for j in order if grp["valid"][j]))
        if args.credit_answers and len(sources) > args.credit_answers:  # the best and K - 1 others at random
            sources = sources[:1] + rng.sample(sources[1:], args.credit_answers - 1)
        for src in sources:
            answer = edits.Answer.parse(src)
            es = edits.credit_edits(answer, args.credit, rng)
            if es:
                entries.append((g, src, es, len(requests)))
                requests += [edit_item(x, grp["behavior"], seed, args.experiments) for x in [answer.source()] + [edits.apply(answer, e).source() for e in es]]
    if not requests:
        return
    totals = edits.totals(timed(clock, "credit", score, requests))
    for g, src, es, k in entries:
        grp = groups[g]
        if math.isinf(totals[k]):
            continue
        dS = {e: totals[k + 1 + i] - totals[k] for i, e in enumerate(es)}
        for j, it in enumerate(grp["items"]):
            if it["source"] == src and grp["valid"][j]:
                grp["credit"][j] = {str(e): v for e, v in dS.items()}
                grp["token_advantages"][j] = credit_advantages(tok, grp["completions"][j], grp["texts"][j], src, float(grp["advantage"][j]), dS, scales.scale[grp["behavior"]["id"]])


def rl2_sample(chosen: list[dict], step: int, args, pol, sampler, adapter: Path, clock: dict) -> list[dict]:
    """A group of --samples answers per behavior (the sampling half of rl2_groups), with vLLM's log-probabilities of
    the sampled tokens (the behavior policy of ppo_update's importance weights)."""
    n = args.samples
    prompts = [pol.prompt_ids(render(b)) for b in chosen]
    comps = timed(clock, "sample", sampler, prompts, n, adapter, step)
    behavior = getattr(sampler, "token_logprobs", None) or [None] * (len(prompts) * n)
    return [{"behavior": b, "prompt": prompts[g], "completions": comps[g], "texts": [pol.tok.decode(c, skip_special_tokens=True) for c in comps[g]],
             "behavior_logprobs": behavior[g * n : (g + 1) * n]} for g, b in enumerate(chosen)]


def rl2_score(groups: list[dict], step: int, args, tok, score, scales: Scales, clock: dict) -> list[dict]:
    """The checker half of rl2_groups, in place: every answer scored under the step's seed (with the teacher's and the
    empty program's scores of behaviors seen for the first time, for their scale); every sampled token gets its
    advantage: the episode's RLOO advantage (3), replaced on align/claim statements by the measured credit (1, --credit
    > 0). An invalid answer counts as the group's worst valid one (a group with no valid answer has no signal)."""
    seed, n = step_seed(args, step), args.samples
    for grp in groups:
        grp["items"] = [item(x, grp["behavior"], seed, 0, args.experiments) for x in grp["texts"]]
    extra = scales.items([g["behavior"] for g in groups], seed, args.experiments)
    scores = timed(clock, "score", score, [it for g in groups for it in g["items"]] + [it for _, _, it in extra])
    scales.take(extra, scores[len(groups) * n :])
    for g, grp in enumerate(groups):
        sc = scores[g * n : (g + 1) * n]
        S = np.array([x["total_bits"] for x in sc], dtype=float)
        valid = np.array([bool(x["valid"]) for x in sc])
        A = rloo(np.where(valid, S, S[valid].max()), scales.scale[grp["behavior"]["id"]]) if valid.any() else np.zeros(n)
        scales.observe(grp["behavior"]["id"], S, valid)
        grp.update({"scores": sc, "S": S, "valid": valid, "advantage": A, "token_advantages": [[float(A[j])] * len(c) for j, c in enumerate(grp["completions"])],
                    "credit": [None] * n})
    if args.credit:
        credit_groups(groups, seed, args, tok, score, scales, clock)
    return groups


def rl2_groups(chosen: list[dict], step: int, args, pol, sampler, score, scales: Scales, adapter: Path, clock: dict) -> list[dict]:
    """rl2_sample, then rl2_score."""
    return rl2_score(rl2_sample(chosen, step, args, pol, sampler, adapter, clock), step, args, pol.tok, score, scales, clock)


def informative(group: dict) -> bool:
    """(5) A group trains the policy only if some token's advantage is nonzero (not all invalid, not all equal without
    credit)."""
    return any(a != 0.0 for adv in group["token_advantages"] for a in adv)


def expert_iteration(groups: list[dict], step: int, args, pol, score, candidates: dict, clock: dict) -> list[dict]:
    """(2) edits.refine from each group's best valid answer under the step's seed, reader off (--refine rounds, the next
    --refine-adds candidates per variable from --candidates, --credit part drops sampled per round): an answer it
    improves comes back as the sampled answer with the program block replaced, for the SFT term and the DPO pair."""
    seed, out = step_seed(args, step), []
    import inspect

    sampled_drops = {"max_drops": args.credit or None} if "max_drops" in inspect.signature(edits.refine).parameters else {}  # answers of hundreds of parts
    for grp in groups:
        if not grp["valid"].any():
            continue
        j = int(np.where(grp["valid"], grp["S"], np.inf).argmin())
        b, src, text = grp["behavior"], grp["items"][j]["source"], grp["texts"][j]
        offset = program_offset(text, src)
        if offset is None:
            continue

        def run(sources, b=b):
            return score([edit_item(x, b, seed, args.experiments) for x in sources])

        best, bits, accepted = timed(clock, "refine", edits.refine, edits.Answer.parse(src), run, candidates.get(b["id"]), args.refine, args.refine_adds, **sampled_drops)
        if not accepted:
            continue
        new = text[:offset] + best.source() + text[offset + len(src) :]
        out.append({"behavior": b["id"], "prompt": grp["prompt"], "improved": pol.tok.encode(new, add_special_tokens=False) + [pol.end], "sampled": grp["completions"][j],
                    "text": new, "bits": bits, "sampled_bits": float(grp["S"][j]), "accepted": [str(e) for e in accepted]})
    return out


def ppo_update(pol: Policy, prompts, completions, token_adv, beta: float, micro: int, epochs: int, clip: dict, optimizer, warmup, behavior=None) -> dict:
    """(4) `epochs` optimizer steps on one scored batch, PPO's clipped objective in its decoupled form (AReaL's
    ppo_actor_loss_fn, verl's compute_policy_loss_vanilla with rollout correction), summed over each episode's tokens:
      L = -(1/E) sum_e sum_t w_et l_et,
      l_et = min(rho A, clip(rho, 1 - eps_low, 1 + eps_high) A), and for A < 0 at least c A (dual clip),
      rho_et = pi(y_et) / pi_prox(y_et),   w_et = min(pi_prox(y_et) / pi_behavior(y_et), tis_cap) (detached),
    pi_prox = the policy at the start of this update (the first epoch's log-probabilities: rho = 1 there and the gradient
    is w A grad log pi), pi_behavior = the policy that sampled (vLLM's log-probabilities of the sampled tokens, `behavior`:
    one list per episode, or None for w = 1). w corrects vLLM's rounding and, with --async, the one step the sampling
    policy lags (truncated importance sampling). clip = {"low", "high", "dual", "tis_cap"}; eps_high > eps_low is DAPO's
    clip-higher. The KL to pi_ref enters as in grpo_update, through the advantage: A_et - beta (log pi_prox(y_e) -
    log pi_ref(y_e)), fixed across epochs (constant A_e, one epoch and w = 1 give grpo_update's gradient)."""
    # Adapted from verl/trainer/ppo/core_algos.py compute_policy_loss_vanilla and rollout_corr_helper.py
    # compute_rollout_correction_weights (volcengine/verl 75879f7f475f, Apache-2.0) and AReaL
    # areal/utils/functional/functional.py ppo_actor_loss_fn (inclusionAI/AReaL, Apache-2.0).
    E = len(prompts)
    batches = list(micro_batches(E, micro))
    old, shifted, weight = {}, {}, {}
    stats = {"loss": [], "clip_fraction": [], "grad_norm": [], "kl_sum_per_episode": 0.0, "logprob_sums": [], "tis_weight_mean": None, "behavior_gap_per_token": None}
    gaps, weights = [], []
    for epoch in range(epochs):
        total, clipped, counted = 0.0, 0.0, 0.0
        for k, idx in enumerate(batches):
            ps, cs = [prompts[i] for i in idx], [completions[i] for i in idx]
            cur, mask = pol.token_logprobs(ps, cs)
            if epoch == 0:
                old[k] = cur.detach()
                stats["logprob_sums"] += (old[k] * mask).sum(1).tolist()
                ratio_e = torch.zeros(len(idx), device=pol.dev)
                if beta > 0:
                    ref, _ = pol.token_logprobs(ps, cs, ref=True)
                    ratio_e = ((old[k] - ref) * mask).sum(1)
                    stats["kl_sum_per_episode"] += float(ratio_e.sum()) / E
                A = torch.zeros_like(cur)
                A[mask > 0] = torch.tensor([a for i in idx for a in token_adv[i]], device=pol.dev, dtype=A.dtype)
                shifted[k] = (A - beta * ratio_e[:, None]) * mask
                weight[k] = torch.ones_like(cur)
                if behavior is not None and all(behavior[i] is not None and len(behavior[i]) == len(completions[i]) for i in idx):
                    b = torch.zeros_like(cur)
                    b[mask > 0] = torch.tensor([x for i in idx for x in behavior[i]], device=pol.dev, dtype=b.dtype)
                    log_w = torch.clamp(old[k] - b, -20.0, 20.0)
                    weight[k] = torch.exp(log_w).clamp(max=clip["tis_cap"]) * mask
                    gaps.append(float((log_w.abs() * mask).sum()))
                    weights.append(float((weight[k] * mask).sum()))
            rho = torch.exp(torch.clamp(cur - old[k], -20.0, 20.0))
            A = shifted[k]
            surrogate = torch.minimum(rho * A, rho.clamp(1 - clip["low"], 1 + clip["high"]) * A)
            surrogate = torch.where(A < 0, torch.maximum(surrogate, clip["dual"] * A), surrogate)  # dual clip: a negative advantage's loss bounded by c |A|
            loss = -(surrogate * weight[k] * mask).sum() / E
            loss.backward()
            total += float(loss.detach())
            r = rho.detach()
            clipped += float((((r < 1 - clip["low"]) | (r > 1 + clip["high"])).float() * mask).sum())
            counted += float(mask.sum())
        if epoch == 0 and gaps:
            stats["behavior_gap_per_token"], stats["tis_weight_mean"] = sum(gaps) / counted, sum(weights) / counted
        stats["grad_norm"].append(float(torch.nn.utils.clip_grad_norm_(pol.params, 1.0)))
        optimizer.step()
        warmup.step()
        optimizer.zero_grad(set_to_none=True)
        stats["loss"].append(total)
        stats["clip_fraction"].append(clipped / max(counted, 1.0))
    return stats


def clip_of(args) -> dict:
    return {"low": args.clip, "high": args.clip_high if args.clip_high is not None else args.clip, "dual": args.dual_clip, "tis_cap": args.tis_cap}


def exit_update(pol: Policy, prompts, improved, sampled, beta: float, micro: int) -> dict:
    """(2) Expert iteration on the answers the checker search improved: SFT on each improved answer plus the DPO pair
    (improved over the sample it came from), summed log-probabilities, P improved answers:
      L = -(1/P) sum_g [log pi(y*_g) + log sigmoid(beta ((log pi(y*_g) - log pi_ref(y*_g)) - (log pi(y_g) - log pi_ref(y_g))))]."""
    s = sft_update(pol, prompts, improved, micro)
    d = dpo_update(pol, prompts, improved, sampled, beta, micro)
    return {"exit_sft_loss": s["loss"], "exit_dpo_loss": d["loss"], "exit_dpo_margin": d["margin"]}


def rl2_step(step: int, args, pol, sampler, score, scales: Scales, pool: list[dict], adapter: Path, optimizer, warmup, candidates: dict, logs: dict, started: float) -> dict:
    """One RL v2 step. --behaviors-per-step behaviors drawn uniformly (by the step's seed), a group each (rl2_groups);
    groups without signal are dropped and refilled (5) by behaviors drawn by gap to the teacher, up to --refill more
    sampling rounds; expert iteration (2, --refine > 0) from each group's best valid answer; --ppo-epochs clipped
    updates (4) on the kept groups, then one expert-iteration step (rl2_update)."""
    clock = {"sample": 0.0, "score": 0.0, "credit": 0.0, "refine": 0.0, "train": 0.0}
    score = memo(score)  # one experiment draw per step: a repeated program's score is the same
    rng = random.Random(step_seed(args, step))
    chosen = rng.sample(pool, min(args.behaviors_per_step, len(pool)))
    groups = rl2_groups(chosen, step, args, pol, sampler, score, scales, adapter, clock)
    used, refills = {b["id"] for b in chosen}, 0
    for _ in range(args.refill):
        need, rest = len(chosen) - sum(map(informative, groups)), [b for b in pool if b["id"] not in used]
        if need <= 0 or not rest:
            break
        extra = scales.draw(rest, need, rng)
        used |= {b["id"] for b in extra}
        groups += rl2_groups(extra, step, args, pol, sampler, score, scales, adapter, clock)
        refills += 1
    improved = expert_iteration(groups, step, args, pol, score, candidates, clock) if args.refine else []
    return rl2_update(step, groups, improved, refills, score.hits[0], clock, args, pol, sampler, scales, optimizer, warmup, logs, started)


def rl2_update(step: int, groups: list[dict], improved: list[dict], refills: int, repeated: int, clock: dict, args, pol, sampler, scales: Scales, optimizer, warmup,
               logs: dict, started: float) -> dict:
    """The update of an rl2 step on its scored groups: every answer, its score and its credit to samples.jsonl, every
    improved answer to improved.jsonl; --ppo-epochs clipped updates (4) on the groups with signal, then one
    expert-iteration step (2); a line in train.jsonl."""
    kept = [g for g in groups if informative(g)]
    for grp in groups:
        for j, (it, x, c) in enumerate(zip(grp["items"], grp["scores"], grp["completions"])):
            logs["samples"].write(json.dumps({"step": step, "seed": step_seed(args, step), "behavior": grp["behavior"]["id"], "source": it["source"], "explanation": it["explanation"],
                                              "completion_tokens": len(c), "score": x, "advantage": float(grp["advantage"][j]), "credit": grp["credit"][j],
                                              "kept": informative(grp)}) + "\n")
    logs["samples"].flush()
    for x in improved:
        logs["improved"].write(json.dumps({"step": step, "seed": step_seed(args, step), **{k: v for k, v in x.items() if k not in ("prompt", "improved", "sampled")}}) + "\n")
    logs["improved"].flush()
    t = time.time()
    pol.train_mode(True)
    optimizer.zero_grad(set_to_none=True)
    flat = [(grp["prompt"], c, a, b) for grp in kept for c, a, b in zip(grp["completions"], grp["token_advantages"], grp["behavior_logprobs"])]
    stats = ppo_update(pol, [x[0] for x in flat], [x[1] for x in flat], [x[2] for x in flat], args.beta, args.micro, args.ppo_epochs, clip_of(args),
                       optimizer, warmup, [x[3] for x in flat]) if flat else {}
    stats.pop("logprob_sums", None)
    if improved:
        stats.update(exit_update(pol, [x["prompt"] for x in improved], [x["improved"] for x in improved], [x["sampled"] for x in improved], args.exit_beta, args.micro))
        stats["exit_grad_norm"] = float(torch.nn.utils.clip_grad_norm_(pol.params, 1.0))
        optimizer.step()
        warmup.step()
        optimizer.zero_grad(set_to_none=True)
    clock["train"] = time.time() - t
    best = [scales.relative(g["behavior"]["id"], float(g["S"][g["valid"]].min())) for g in groups if g["valid"].any()]
    S = np.concatenate([g["S"] for g in groups])
    valid = np.concatenate([g["valid"] for g in groups])
    row = {"step": step, "mode": "rl2", "async": bool(getattr(args, "async_rollouts", False)), "seed": step_seed(args, step), "groups": len(groups), "kept": len(kept), "refills": refills,
           "programs": int(len(S)), "mean_bits": float(S.mean()), "valid_fraction": float(valid.mean()), "best_relative_to_teacher": float(np.mean(best)) if best else None,
           "credited": sum(c is not None for g in groups for c in g["credit"]), "improved": len(improved), "repeated_scores": repeated, **stats, "sampling": getattr(sampler, "stats", {}),
           "seconds": clock, "checker_seconds_total": TOTALS["checker_seconds"], "elapsed": time.time() - started}
    logs["train"].write(json.dumps(row) + "\n")
    logs["train"].flush()
    return row


def rl2_async(args, pol, sampler, score, scales: Scales, pool: list[dict], adapter: Path, optimizer, warmup, candidates: dict, logs: dict, started: float, stop) -> None:
    """rl2 with the checker overlapped (--async): while a thread scores, credits and refines step t's groups, vLLM
    samples step t + 1's with the policy not yet updated on step t, so those answers lag the trained policy by one
    step (ppo_update's importance weight corrects it, as AReaL's decoupled PPO and verl's rollout correction do). A step
    draws --behaviors-per-step behaviors by its seed plus, by gap to the teacher, as many as the last step scored when
    its sampling starts (two steps back) lost to groups without signal: the refill, late. The thread has its own
    tokenizer (a fast tokenizer used from two threads raises "Already borrowed")."""
    import copy
    import types
    from concurrent.futures import ThreadPoolExecutor

    side = types.SimpleNamespace(tok=copy.deepcopy(pol.tok), end=pol.end)

    def draw(step: int, carry: int) -> list[dict]:
        rng = random.Random(step_seed(args, step))
        chosen = rng.sample(pool, min(args.behaviors_per_step, len(pool)))
        rest = [b for b in pool if b["id"] not in {c["id"] for c in chosen}]
        return chosen + (scales.draw(rest, carry, rng) if carry and rest else [])

    def check(groups: list[dict], step: int, clock: dict):
        memo_score = memo(score)
        rl2_score(groups, step, args, side.tok, memo_score, scales, clock)
        improved = expert_iteration(groups, step, args, side, memo_score, candidates, clock) if args.refine else []
        return improved, memo_score.hits[0]

    carry = 0
    clock = {"sample": 0.0, "score": 0.0, "credit": 0.0, "refine": 0.0, "train": 0.0}
    pol.save(adapter)
    groups = rl2_sample(draw(0, carry), 0, args, pol, sampler, adapter, clock)
    with ThreadPoolExecutor(1) as pool_thread:
        for step in range(args.steps):
            if stop():
                break
            future = pool_thread.submit(check, groups, step, clock)
            nxt = {"sample": 0.0, "score": 0.0, "credit": 0.0, "refine": 0.0, "train": 0.0}
            t = time.time()
            upcoming = rl2_sample(draw(step + 1, carry), step + 1, args, pol, sampler, adapter, nxt) if step + 1 < args.steps else None
            improved, repeated = future.result()
            clock["overlap_wait"] = time.time() - t - nxt["sample"]  # checker time not hidden behind sampling
            row = rl2_update(step, groups, improved, 0, repeated, clock, args, pol, sampler, scales, optimizer, warmup, logs, started)
            carry = min(args.behaviors_per_step, row["groups"] - row["kept"]) if args.refill else 0
            pol.save(adapter)
            if upcoming is None:
                break
            groups, clock = upcoming, nxt


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
        scores = score([item(t, b, step_seed(args, step), args.uniform_seeds, args.experiments) for b, ts in zip(chosen, texts) for t in ts])
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
    {"behavior", "source", "score"} layout; the target is the record's "answer" text (or "answer_path"), else its
    source as a python block followed by its "explanation") plus every unscored one (printed examples); behaviors outside
    the training pool are never used; and
    --data examples (JSONL of {"messages": [user, assistant]} or {"prompt", "completion"}, e.g.
    g-predict's prediction questions). The completion ends with <|im_end|>."""
    import glob

    by_id = {b["id"]: b for b in pool}
    best = {f"teacher:{bid}": {"behavior": bid, "answer": text} for bid, text in sorted(TEACHER.items()) if bid in by_id}  # --teacher: training behaviors only
    for pattern in args.programs or []:
        pattern = os.path.expanduser(pattern)
        for path in sorted(glob.glob(os.path.join(pattern, "*.json") if os.path.isdir(pattern) else pattern)):
            r = json.loads(Path(path).read_text())
            if r.get("behavior") not in by_id or not (r.get("source") or r.get("answer") or r.get("answer_path")):
                continue
            if "score" not in r:  # an unscored program (a printed hand-written example): always an example
                best[path] = r
            elif r["behavior"] not in best or r["score"]["total_bits"] < best[r["behavior"]]["score"]["total_bits"]:
                best[r["behavior"]] = r
    end = [pol.end]

    def answer(r: dict) -> str:  # the oracle's whole answer: one python block, then the plain-English explanation
        if r.get("answer"):
            return r["answer"]
        if r.get("answer_path"):
            return Path(r["answer_path"]).expanduser().read_text()
        text = "```python\n" + r["source"].strip() + "\n```"
        return text + ("\n\n" + r["explanation"].strip() if r.get("explanation") else "")

    target = (lambda text: pol.parts.reg.rewrite(text)) if getattr(pol, "parts", None) is not None else (lambda text: text)  # noqa: E731  addresses -> part tokens
    programs = [(pol.prompt_ids(render(by_id[r["behavior"]])), pol.tok.encode(target(answer(r).strip()), add_special_tokens=False) + end) for _, r in sorted(best.items())]
    questions = []
    for path in args.data or []:
        for line in open(os.path.expanduser(path)):
            q = json.loads(line)
            user, answer = (q["messages"][0]["content"], q["messages"][1]["content"]) if "messages" in q else (q["prompt"], q["completion"])
            questions.append((pol.prompt_ids(user), pol.tok.encode(answer, add_special_tokens=False) + end))
    keep = lambda xs: [(p, c) for p, c in xs if len(p) + len(c) <= args.max_model_len]  # noqa: E731
    return keep(programs), keep(questions)


def sft(args, pol, pool, optimizer, warmup, log) -> dict:
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
        warmup.step()
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
        empty, teacher = base.get("empty"), base.get("teacher")
        T = teacher["total_bits"] if teacher and teacher.get("valid") and teacher["total_bits"] > 0 else None
        row = {"set": name, "step": step, "behavior": b["id"], "mean_bits": float(S.mean()), "best_bits": float(S[j]), "valid_fraction": float(valid.mean()),
               "below_empty_fraction": float(np.mean(valid & (S < empty["total_bits"]))) if empty else None, "best_recovered": recovered(mine[j][1], empty),
               "best_relative_to_teacher": (float(S[j]) - T) / T if T and valid.any() else None,
               "mean_valid_relative_to_teacher": (float(S[valid].mean()) - T) / T if T and valid.any() else None,
               "baselines": {n: x["total_bits"] for n, x in base.items()}, "baselines_recovered": {n: recovered(x, empty) for n, x in base.items()}, "best_source": mine[j][0]}
        rows.append(row)
        log.write(json.dumps(row) + "\n")
    names = sorted({n for _, _, base in groups for n in base})
    return {"behaviors": len(rows), "mean_bits": mean([r["mean_bits"] for r in rows]), "best_of_n_bits": mean([r["best_bits"] for r in rows]),
            "valid_fraction": mean([r["valid_fraction"] for r in rows]), "below_empty_fraction": mean([r["below_empty_fraction"] for r in rows]),
            "best_recovered": mean([r["best_recovered"] for r in rows]), "best_relative_to_teacher": mean([r["best_relative_to_teacher"] for r in rows]),
            "mean_valid_relative_to_teacher": mean([r["mean_valid_relative_to_teacher"] for r in rows]), "baselines": {n: mean([r["baselines"].get(n) for r in rows]) for n in names},
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
            answers = [(b, pol.tok.decode(c, skip_special_tokens=True)) for b, g in zip(pool, groups) for c in g]
            items = [item(t, b, args.eval_seed, 0, args.eval_experiments) for b, t in answers]
            base = [(b, n, src) for b in pool for n, src in (baselines(b).items() if args.baselines else [])]
            scores = score(items + [item(src, b, args.eval_seed, 0, args.eval_experiments) for b, _, src in base])  # a bare source is its own program
            per_base = {}
            for (b, n, src), x in zip(base, scores[len(items) :]):
                per_base.setdefault(b["id"], {})[n] = x
                samples.write(json.dumps({"set": name, "step": step, "run": run, "behavior": b["id"], "behavior_path": b.get("path"), "program": n, "source": src, "score": x}) + "\n")
            out = []
            for g, b in enumerate(pool):
                mine = [(it["source"], x) for it, x in zip(items[g * args.samples : (g + 1) * args.samples], scores[g * args.samples : (g + 1) * args.samples])]
                for it, (src, x) in zip(items[g * args.samples : (g + 1) * args.samples], mine):
                    samples.write(json.dumps({"set": name, "step": step, "run": run, "behavior": b["id"], "behavior_path": b.get("path"), "program": "oracle", "source": src,
                                              "explanation": it["explanation"], "score": x}) + "\n")
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
        scored = score([{"source": r["source"], "explanation": r.get("explanation", ""), "behavior": behaviors_by_path[bpath], "seed": args.eval_seed, "experiments": args.eval_experiments,
                         "options": options} for r in unique])
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


VOCAB_DIR = None


def vocab_dir() -> Path:
    """The extended-vocabulary checkpoint's directory: one temporary directory per process, removed at exit.
    Under --out, a Qwen3-8B checkpoint (17 GB) was copied back from the pod with the outputs, and the next
    process's second copy filled the pod's disk."""
    global VOCAB_DIR
    if VOCAB_DIR is None:
        import atexit
        import shutil
        import tempfile

        VOCAB_DIR = Path(tempfile.mkdtemp(prefix="rl-vocab-"))
        atexit.register(shutil.rmtree, VOCAB_DIR, True)
    return VOCAB_DIR


def refresh_parts(pol, sampler, out: Path, args):
    """With part tokens and vLLM: rewrite the checkpoint with the current part rows and restart vLLM on it."""
    inner = sampler.inner if isinstance(sampler, ValidSampler) else sampler
    if pol.parts is not None and isinstance(inner, VllmSampler):
        inner.reload(pol.materialize(vocab_dir(), args.base))


def views_of(args) -> dict | None:
    return {"vpd": args.vpd_view} if getattr(args, "vpd_view", None) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["grpo", "rl2", "dpo", "bestofn", "sft", "eval", "rescore"], required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--init", help="SFT adapter the policy starts from and is held to (pi_ref): a PEFT directory or g-predict's sft.py output")
    ap.add_argument("--model", required=True, help="target model whose behaviors are explained: qwen3-0.6b | vpd4l")
    ap.add_argument("--behaviors", default=str(BEHAVIORS))
    ap.add_argument("--scorer", choices=sorted(SCORERS), default="checker")
    ap.add_argument("--score-workers", type=int, default=1, help="checker servers per target model, each scoring whole behaviors in parallel")
    ap.add_argument("--checker", help="the checker binary (mpd_graph_2951; score.py's GRAPH_CHECKER); on MATS name target/release/examples/mpd_graph_2951 so the job builds it")
    ap.add_argument("--vpd-view", help="VPD's decomposition export for the checker's vpd view (programs naming VPD parts are invalid without it; vpd4l: ~/mpd-data/engine/vpd4l_decomposition)")
    ap.add_argument("--checker-device", choices=["gpu"], help="run the checker's large products on the single-precision device (float32; compare scores only within one device)")
    ap.add_argument("--reader-items", type=int, default=0, help="without a reader server, keep the reader items of every K-th scored program for offline reader scoring (0: none)")
    ap.add_argument("--reader-item-stride", type=int, default=1, help="of a kept program's reader items, keep every S-th (an unbiased subsample of the reader term's mean)")
    ap.add_argument("--score-batch", type=int, default=4, help="programs per checker request (a server's memory grows with it)")
    ap.add_argument("--checker-gib", type=int, help="the checker server's memory lease on the Mac (score.py's default otherwise)")
    ap.add_argument("--export", help="the target model's export directory for the checker (score.py's EXPORTS entry otherwise)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--hours", type=float, help="stop and save after this many hours")
    ap.add_argument("--checker-hours", type=float, help="stop and save once training scores (with rl2's credit and refinement) used this many checker hours")
    ap.add_argument("--behaviors-per-step", type=int, default=8)
    ap.add_argument("--samples", type=int, default=8, help="programs per behavior per step (the group)")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--warmup", type=int, default=20, help="optimizer steps of linear learning-rate warmup")
    ap.add_argument("--beta", type=float, help="grpo: KL weight (default 0.04); dpo: inverse temperature (default 0.1)")
    ap.add_argument("--sft-epochs", type=int, default=1)
    ap.add_argument("--repair", type=int, default=0, help="bestofn: rounds of revisions of each behavior's best program, shown its measured failures (training data only)")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--sampler", choices=["auto", "vllm", "hf"], default="auto")
    ap.add_argument("--resample", type=int, default=0, help="redraw each invalid program (mech.trace) up to R times at sampling time")
    ap.add_argument("--grammar", action="store_true", help="vLLM decodes every answer under rl/grammar.py's grammar (parts only inside align/claim statements, only the decomposition's); --resample stays the fallback")
    ap.add_argument("--part-tokens", help="part tokens: the registry file of g-predict's part_tokens.py build; the projections train with the LoRA and vLLM gets the rows in place")
    ap.add_argument("--materialize-every", type=int, default=0, help="with --part-tokens: also rewrite the checkpoint and restart vLLM every K steps (0: only at the start; the rows are copied in place before every sampling call)")
    ap.add_argument("--share-gpu", action="store_true", help="one GPU for vLLM and the trainer: vLLM sleeps (weights to host) while training and the trainer moves to the host while sampling, so --gpu-memory can be 0.8")
    ap.add_argument("--no-reader", action="store_true", help="score without the reader term (required when GRAPH_READER is unset and the checker scores); execution, necessity, alignment, claims and complexity stay on")
    ap.add_argument("--hf-batch", type=int, default=16, help="sequences per transformers generate call (the Mac / CPU sampler)")
    ap.add_argument("--gpu-memory", type=float, default=0.85, help="vLLM's share of its GPU (lower it when the trainer shares the GPU)")
    ap.add_argument("--prompt-holdout", type=int, default=4, help="every K-th prompt of each training behavior is held out for evaluation (0: none)")
    ap.add_argument("--eval-every", type=int, default=0, help="evaluate every E training steps and at the end (0: only --mode eval)")
    ap.add_argument("--eval-seed", type=int, default=1_000_003, help="the evaluation's experiment seed (training steps use their index)")
    ap.add_argument("--experiments", type=int, default=16, help="experiments per training score: the teacher search's and the evaluation's (N, the tokens scored, sets what a part must earn)")
    ap.add_argument("--eval-experiments", type=int, help="experiments per evaluation score (default: --experiments; the team keeps search, RL and evaluation at one setting)")
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
    ap.add_argument("--teacher", help="teacher answers (DIR/manifest.jsonl, <behavior>.answer.txt or .py): rl2's reward scale, an evaluation baseline, and SFT answers for training behaviors")
    ap.add_argument("--teacher-heldout", help="held-out behaviors' teacher answers (~/mpd-data/graph_oracle/teacher_heldout): evaluation baselines only")
    ap.add_argument("--credit", type=int, default=16, help="rl2: part drops sampled per answer for per-token credit (every statement drop is scored too; 0: episode advantages only)")
    ap.add_argument("--credit-answers", type=int, default=0, help="rl2: answers credited per group: the best valid one and K - 1 other distinct valid ones drawn at random (0: every distinct valid answer)")
    ap.add_argument("--refill", type=int, default=1, help="rl2: sampling rounds that replace groups without signal by behaviors drawn by gap to the teacher")
    ap.add_argument("--refine", type=int, default=2, help="rl2: rounds of edits.refine from each group's best answer for expert iteration (0: off)")
    ap.add_argument("--refine-adds", type=int, default=4, help="rl2: candidate parts tried per variable per refine round (--candidates)")
    ap.add_argument("--candidates", help="rl2: DIR/<behavior>.json = {variable: [part token, ...]} best first (g-int's importance ranking), the parts refine may add")
    ap.add_argument("--ppo-epochs", type=int, default=2, help="rl2: optimizer steps per scored batch (PPO clipped ratio)")
    ap.add_argument("--async", dest="async_rollouts", action="store_true", help="rl2: the checker scores step t while vLLM samples step t + 1 (one step of policy lag, importance-weighted)")
    ap.add_argument("--clip", type=float, default=0.2, help="rl2: PPO ratio clip below 1 (eps_low)")
    ap.add_argument("--clip-high", type=float, default=0.28, help="rl2: PPO ratio clip above 1 (eps_high; DAPO's clip-higher, 0.28)")
    ap.add_argument("--dual-clip", type=float, default=3.0, help="rl2: dual-clip bound c on a negative advantage's ratio (verl's clip_ratio_c)")
    ap.add_argument("--tis-cap", type=float, default=2.0, help="rl2: truncation of the importance weight pi_prox / pi_behavior (verl's rollout_is_threshold)")
    ap.add_argument("--exit-beta", type=float, default=0.1, help="rl2: inverse temperature of expert iteration's DPO pair")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.scorer == "checker" and not os.environ.get("GRAPH_READER") and not args.no_reader:
        raise SystemExit("the score includes the reader term: set GRAPH_READER (reader_score.py serve) or pass --no-reader")
    lr = args.lr if args.lr is not None else {"bestofn": 1e-4, "sft": 1e-4}.get(args.mode, 1e-5)
    beta = args.beta if args.beta is not None else {"dpo": 0.1}.get(args.mode, 0.04)
    args.beta = beta
    args.eval_experiments = args.eval_experiments or args.experiments
    refuse_heldout([args.teacher, args.candidates, *(args.programs or []), *(args.data or [])])
    TEACHER.update(teacher_answers(args.teacher))
    HELDOUT_TEACHER.update(teacher_answers(args.teacher_heldout))
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
    if args.part_tokens:  # in-process vLLM engine: part rows are copied into its weights before each sampling call
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        if args.share_gpu and args.materialize_every:
            raise SystemExit("--materialize-every restarts vLLM, which sleep mode (--share-gpu) allows once per process")
    use_vllm = args.sampler == "vllm" or (args.sampler == "auto" and torch.cuda.is_available() and __import__("importlib").util.find_spec("vllm") is not None)
    sampler = None
    if use_vllm:  # vLLM first, on the first visible GPU, before the trainer touches CUDA
        from transformers import AutoTokenizer

        rank = json.loads((Path(args.init) / "adapter_config.json").read_text())["r"] if args.init else args.lora_rank
        answers = None
        if args.grammar:  # guided decoding: parts only in align/claim statements, only the attached decomposition's (rl/grammar.py)
            import grammar

            answers = grammar.model_grammar(args.model, args.part_tokens)
        sampler = VllmSampler(args, rank, AutoTokenizer.from_pretrained(args.base).convert_tokens_to_ids("<|im_end|>"), None if args.part_tokens else args.base, answers)
    dev = torch.device(f"cuda:{torch.cuda.device_count() - 1}" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    pol = Policy(args, dev)
    if isinstance(sampler, VllmSampler):
        sampler.policy = pol
        if pol.parts is not None:  # vLLM starts once, on a checkpoint with the extended vocabulary; the rows then come in place
            sampler.reload(pol.materialize(vocab_dir(), args.base))
            sampler.rows = pol.part_rows
    if sampler is None:
        if args.grammar:
            print("--grammar needs vLLM: transformers' sampler decodes without it", file=sys.stderr)
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
    optimizer = torch.optim.AdamW(pol.param_groups(lr), weight_decay=0.0)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / args.warmup))  # linear over --warmup optimizer steps
    if args.mode == "sft":
        sft(args, pol, pool, optimizer, warmup, open(out / "train.jsonl", "a"))
        pol.save(adapter)
        print(json.dumps(evaluate(sets, pol, sampler, score, args, adapter, 1, open(out / "eval.jsonl", "a"), args.sft_steps)))
        return
    log = open(out / "train.jsonl", "a")
    eval_log = open(out / "eval.jsonl", "a")
    samples_log = open(out / "samples.jsonl", "a")
    best_log = open(out / "best.jsonl", "a") if args.mode == "bestofn" else None
    logs = {"train": log, "samples": samples_log, "improved": open(out / "improved.jsonl", "a") if args.mode == "rl2" else None}
    scales = Scales(TEACHER)
    candidates = {p.stem: json.loads(p.read_text()) for p in sorted(Path(args.candidates).expanduser().glob("*.json"))} if args.candidates else {}
    started = time.time()
    for step in range(args.steps):
        if args.hours and time.time() - started > 3600 * args.hours:
            break
        if args.checker_hours and TOTALS["checker_seconds"] > 3600 * args.checker_hours:  # the A/B's budget: equal checker time per arm
            break
        pol.save(adapter)
        if args.materialize_every and step and step % args.materialize_every == 0:
            refresh_parts(pol, sampler, out, args)
        if args.eval_every and step % args.eval_every == 0:
            evaluate(sets, pol, sampler, score, args, adapter, step, eval_log, step)
        if args.mode == "rl2" and args.async_rollouts:  # its own loop: the checker overlaps the next step's sampling
            rl2_async(args, pol, sampler, score, scales, pool, adapter, optimizer, warmup, candidates, logs, started,
                      lambda: bool((args.hours and time.time() - started > 3600 * args.hours) or (args.checker_hours and TOTALS["checker_seconds"] > 3600 * args.checker_hours)))
            break
        if args.mode == "rl2":
            rl2_step(step, args, pol, sampler, score, scales, pool, adapter, optimizer, warmup, candidates, logs, started)
            continue
        t0 = time.time()
        chosen = random.Random(step_seed(args, step)).sample(pool, min(args.behaviors_per_step, len(pool)))  # the behaviors rl2 draws at this step
        prompts = [pol.prompt_ids(render(b)) for b in chosen]
        groups = sampler(prompts, args.samples, adapter, step)
        t1 = time.time()
        texts = [[pol.tok.decode(c, skip_special_tokens=True) for c in g] for g in groups]
        items = [item(t, b, step_seed(args, step), args.uniform_seeds, args.experiments) for b, ts in zip(chosen, texts) for t in ts]
        scores = memo(score)(items)  # identical answers of a group scored once, as in rl2
        t2 = time.time()
        TOTALS["checker_seconds"] += t2 - t1
        S = np.array([s["total_bits"] for s in scores], dtype=float).reshape(len(chosen), args.samples)
        valid = np.array([bool(s["valid"]) for s in scores]).reshape(len(chosen), args.samples)
        for (b, it, s, c) in zip([b for b in chosen for _ in range(args.samples)], items, scores, [c for g in groups for c in g]):
            samples_log.write(json.dumps({"step": step, "behavior": b["id"], "source": it["source"], "explanation": it["explanation"], "completion_tokens": len(c), "score": s}) + "\n")
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
            stats = grpo_update(pol, flat_p, flat_c, adv.tolist(), beta, args.micro)
        elif args.mode == "dpo":
            pairs = [(g, int(np.where(valid[g], S[g], np.inf).argmin()), int(np.where(valid[g], S[g], np.inf).argmax()) if valid[g].all() else int((~valid[g]).argmax()))
                     for g in range(len(chosen)) if valid[g].any() and (not valid[g].all() or S[g].max() > S[g].min())]
            stats = dpo_update(pol, [prompts[g] for g, _, _ in pairs], [groups[g][w] for g, w, _ in pairs], [groups[g][l] for g, _, l in pairs], beta, args.micro) if pairs else {}
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
                warmup.step()
                optimizer.zero_grad(set_to_none=True)
            stats.update({"kept": len(keep), "repaired": len(repaired), "kept_mean_bits": float(np.mean([best[g]["score"]["total_bits"] for g in keep])) if keep else None})
        sums = stats.pop("logprob_sums", None)
        if sums is not None and sampler.logprob_sums is not None:  # on-policy check: the sampler's log pi(y) against the trainer's, per token
            stats["sampler_trainer_logprob_gap_per_token"] = float(np.sum(np.abs(np.array(sums) - np.array(sampler.logprob_sums))) / max(1, sum(len(c) for c in flat_c)))
        if args.mode != "bestofn":
            stats["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(pol.params, 1.0))
            optimizer.step()
            warmup.step()
        t3 = time.time()
        tokens = [len(c) for c in flat_c]
        log.write(json.dumps({"step": step, "mode": args.mode, "behaviors": len(chosen), "programs": len(items), "mean_bits": float(S.mean()), "best_bits": float(S.min(1).mean()),
                              "worst_bits": float(S.max(1).mean()), "valid_fraction": float(valid.mean()), "mean_completion_tokens": float(np.mean(tokens)), **stats,
                              "sampling": getattr(sampler, "stats", {}), "seconds": {"sample": t1 - t0, "score": t2 - t1, "train": t3 - t2},
                              "checker_seconds_total": TOTALS["checker_seconds"], "elapsed": time.time() - started,
                              "example": items[int(np.where(valid, S, np.inf).reshape(-1).argmin())]["source"][:2000]}) + "\n")
        log.flush()
    pol.save(adapter)
    if args.eval_every:
        evaluate(sets, pol, sampler, score, args, adapter, args.steps, eval_log, args.steps)


if __name__ == "__main__":
    main()

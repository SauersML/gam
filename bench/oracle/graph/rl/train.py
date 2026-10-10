"""The graph oracle's training (#2951): sample answers, score them with the verifier, train on how they rank.

A question (a task) is a text (text.py) and the model's prediction of the next token after it. An answer is a
plain-English explanation and an ordered graph of vpd4l's VPD subcomponents (mech.py's graph(tokens, targets)): a list
of steps, most important first, each adding subcomponents at positions and connections the model has. The verifier
(native.py, score.py) runs the graph of each answer's first k steps alone over changed prompts of the text and measures
the KL of the model's prediction from the graph's and the graph's description length: the answer's curve. Answers to one
question rank by score.keys: the mean KL their curves reach over a log-uniform range of description lengths shared by
the answers compared. No tolerance and no weight.

  sft   --sft-steps steps on bootstrap search answers of the training questions (native.py search; --search) and --data
        examples: loss = -(1/B) sum_e sum_t log pi(y_et | x_e, y_e<t).
  rl2   (rl2_step) each step draws --behaviors-per-step questions and samples a group of --samples answers each (vLLM
        serving the current LoRA adapter on a GPU, transformers' generate otherwise; Qwen3 chat, thinking off); every
        answer is scored, its advantage A_e = (wins - losses) / (n - 1) against the other answers of its group by
        score.keys (3); the tokens of a subcomponent in a reader's parents get the sign of what dropping it (the edge)
        measured, edits.credit (1); --ppo-epochs clipped updates per scored batch (4); groups without signal dropped and
        refilled by questions drawn uniformly (5); expert iteration from each group's best answer, edits.refine (2).
  eval  samples --samples answers per held-out question and scores them with the baselines: the empty graph, the
        search's answer (--search-heldout) and VPD's own answer.

Sums, not per-episode means: a per-episode mean of token log-probabilities gives each token of a long answer less
weight than each token of a short one, so its gradient is not the policy gradient; the sum is the episode's
log-probability.

The reference pi_ref is the SFT policy: --init ADAPTER (a PEFT adapter, e.g. --mode sft's) is loaded twice, as the
trainable policy and as a frozen reference; without --init the policy is a fresh LoRA on the base and pi_ref is the base
(adapter disabled). The loop writes the adapter every step (vLLM loads it by path).

  train.py --mode sft --search texts/search --base Qwen/Qwen3-4B --model vpd4l --behaviors texts --out DIR
  train.py --mode rl2 --init DIR/adapter ... --out DIR2 --hours H
  train.py --mode eval --init ADAPTER --search-heldout texts/search_heldout ... --out DIR

Outputs: DIR/train.jsonl (a line per step), DIR/eval.jsonl (a line per evaluated question and a summary per evaluation),
DIR/samples.jsonl (every answer with its score), DIR/improved.jsonl (expert iteration's refined answers), DIR/adapter.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
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
import edits  # noqa: E402
from prompt import english_spans, render, split_answer  # noqa: E402
import reader as reader_module  # noqa: E402
import score as score_module  # noqa: E402
import scorer  # noqa: E402
from scorer import SCORERS  # noqa: E402

BEHAVIORS = Path.home() / "mpd-data/graph_oracle/texts"  # the real-text tasks (text.py), train and held-out
SEARCH: dict[str, str] = {}  # --search's answers by behavior id (read_answers): training behaviors
HELDOUT_SEARCH: dict[str, str] = {}  # --search-heldout's: evaluation baselines only, never SFT or RL


def item(answer: str, behavior: dict, seed: int) -> dict:
    """A scoring item from an oracle answer: prompt.split_answer's program (the last python block that parses) and the
    text after it; the baseline sources (score.EMPTY, "vpd") pass through."""
    source, explanation = (answer, "") if answer in (score_module.EMPTY, "vpd") else split_answer(answer)
    return {"source": source, "explanation": explanation, "behavior": behavior, "seed": seed}


def behaviors(root: Path, model: str, split: str) -> list[dict]:
    """The split's questions, one per task."""
    out = []
    for p in sorted((root / model).glob("*.json")):
        if p.name.startswith("."):  # macOS's ._ metadata files
            continue
        task = json.loads(p.read_text())
        if task.get("split", "train") == split:
            out.append({**task, "task": task["id"], "path": str(p)})
    return out


def baselines(b: dict) -> dict:
    """What the oracle's answers to question b are compared with, scored alongside them: the empty graph, VPD's own
    answer, and the bootstrap search's answer (--search or --search-heldout)."""
    out = {"empty": score_module.EMPTY, "vpd": "vpd"}
    if b["id"] in SEARCH or b["id"] in HELDOUT_SEARCH:
        out["search"] = SEARCH.get(b["id"]) or HELDOUT_SEARCH[b["id"]]
    return out


def load_parts(spec: str, init: str | None, model, base_vocab: int, dev):
    """part_tokens.PartTokens over the registry at SPEC (part_tokens.py build), scaled to the
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

    EVIDENCE = "<|fim_pad|>"  # the placeholder token whose input embedding an evidence vector replaces

    def question_ids(self, b: dict) -> list[int]:
        """The oracle's input for question b: the rendered question, and with --evidence (acts_fn set) one placeholder
        per position and weight matrix, whose embeddings become the model's activations there (embed()); with
        --swap-evidence (self.swap) another text's activations instead (an evaluation of whether the oracle reads them)."""
        text = render(b)
        if getattr(self, "acts_fn", None) is not None:
            ids = b["prompts"][0]["token_ids"]
            text += "\nactivations: " + self.EVIDENCE * (len(ids) * len(self.parts.ev_order))
            q = self.prompt_ids(text)
            self.evidence[tuple(q)] = self.acts_fn(getattr(self, "swap", {}).get(b["id"], ids))
            return q
        return self.prompt_ids(text)

    def revision_ids(self, b: dict, answer: str, report: str) -> list[int]:
        """The oracle's input for revising an answer to question b: the question (question_ids' text, with its evidence),
        the answer, and the verifier's report with the request to revise, as a three-turn conversation."""
        text = render(b)
        evidence = getattr(self, "acts_fn", None) is not None
        if evidence:
            ids = b["prompts"][0]["token_ids"]
            text += "\nactivations: " + self.EVIDENCE * (len(ids) * len(self.parts.ev_order))
        chat = self.tok.apply_chat_template([{"role": "user", "content": text}, {"role": "assistant", "content": answer}, {"role": "user", "content": report}],
                                            add_generation_prompt=True, enable_thinking=False, tokenize=False)
        q = self.tok.encode(chat, add_special_tokens=False)
        if evidence:
            self.evidence[tuple(q)] = self.acts_fn(getattr(self, "swap", {}).get(b["id"], ids))
        return q

    def embed(self, ids: torch.Tensor, prompts: list[list[int]], starts: list[int]) -> torch.Tensor:
        """Input embeddings of a batch [B, W], each prompt's placeholders (prompt r starting at starts[r]) replaced by
        its evidence (PartTokens.evidence of its activations)."""
        table = self.model.get_input_embeddings()  # on the host while vLLM samples with the GPU shared
        emb = table(ids.to(table.weight.device))
        ev_id = self.tok.convert_tokens_to_ids(self.EVIDENCE)
        for r, (p, st) in enumerate(zip(prompts, starts)):
            acts = self.evidence.get(tuple(p))
            if acts is None:
                continue
            pos = st + (torch.tensor(p, device=emb.device) == ev_id).nonzero().flatten()
            emb = emb.index_put((torch.full_like(pos, r), pos), self.parts.evidence(acts).to(emb.device, emb.dtype))
        return emb

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
        evidence = bool(getattr(self, "evidence", None)) and any(tuple(p) in self.evidence for p in prompts)
        inputs = {"input_ids": ids, "attention_mask": att}
        rows, cols = comp[:, 1:].nonzero(as_tuple=True)
        src = rows * width + cols  # the hidden state at position t predicts token t + 1
        target = ids[:, 1:][rows, cols]
        shape = (len(prompts), width - 1)
        rows, cols = torch.as_tensor(rows, device=self.dev), torch.as_tensor(cols, device=self.dev)

        def run():
            if evidence:
                hidden = causal.model(inputs_embeds=self.embed(ids, prompts, [0] * len(prompts)), attention_mask=att).last_hidden_state
            else:
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
            kw = dict(attention_mask=att.to(pol.dev), max_new_tokens=self.max_tokens, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, eos_token_id=pol.end, pad_token_id=pol.end)
            if getattr(pol, "evidence", None) and any(tuple(p) in pol.evidence for p in chunk):  # embeddings in: generate returns the new tokens only
                gen = pol.model.generate(inputs_embeds=pol.embed(ids.to(pol.dev), chunk, [width - len(p) for p in chunk]), **kw).tolist()
            else:
                gen = pol.model.generate(input_ids=ids.to(pol.dev), **kw)[:, width:].tolist()
            gens += [g[: g.index(pol.end) + 1] if pol.end in g else g for g in gen]
        out = [gens[k * n : (k + 1) * n] for k in range(len(prompts))]
        if pol.dev.type == "mps":
            torch.mps.empty_cache()  # the generation's cached blocks, before the checker servers start beside this process
        return out


class VllmSampler:
    """vLLM serving the base with the policy's adapter (reloaded by path at every version). It takes the
    first visible GPU; the trainer takes the second when there is one (--gpu-memory set accordingly)."""

    def __init__(self, args, rank: int, end: int, model: str | None = None):
        self.engine_options, self.sample_options = ({"enable_prompt_embeds": True} if getattr(args, "evidence", False) else {}), {}
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
        # Prefix caching keys blocks by token ids, which prompts given as embeddings (--evidence) do not have: with both,
        # vLLM 0.19.1 hit an illegal memory access on A100 and RTX PRO 6000 alike, so evidence runs go without it.
        self.llm = LLM(model=model, dtype="bfloat16", enable_lora=True, max_lora_rank=self.rank, max_loras=1,
                       enable_prefix_caching="enable_prompt_embeds" not in self.engine_options,
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
            pol = self.policy
            if pol is not None and getattr(pol, "evidence", None):  # evidence: the prompts go in as embeddings
                with torch.no_grad():
                    reqs = [{"prompt_embeds": pol.embed(torch.tensor([p], device=pol.dev), [p], [0])[0].to(torch.bfloat16).cpu()} if tuple(p) in pol.evidence
                            else {"prompt_token_ids": p} for p in prompts]
            else:
                reqs = [{"prompt_token_ids": p} for p in prompts]
            outs = self.llm.generate(reqs, params, lora_request=LoRARequest(f"policy{version}", version + 1, str(adapter)), use_tqdm=False)
        finally:
            if not self.held:
                self.release()
        try:  # the sampled tokens' log-probabilities: the behavior policy of rl2's importance weights, and the on-policy check
            self.token_logprobs = [[d[t].logprob for d, t in zip(c.logprobs, c.token_ids)] for o in outs for c in o.outputs]
            self.logprob_sums = [sum(x) for x in self.token_logprobs]
        except (TypeError, KeyError, AttributeError):  # a vLLM whose logprobs container differs
            self.logprob_sums = self.token_logprobs = None
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


READER: list = []  # the English reader (make_reader), when --reader
TOTALS = {"checker_seconds": 0.0}  # checker time of every training score, credit and refinement so far (the A/B's cost axis)


def step_seed(args, step: int) -> int:
    """(7) Training step `step`'s experiment seed: (--seed << 20) + step, a fresh experiment draw at every step, never
    --eval-seed, whose experiments only the evaluation draws."""
    seed = (args.seed << 20) + step
    return seed if seed != args.eval_seed else seed + (1 << 40)


def read_answers(root) -> dict[str, str]:
    """Answers by behavior id: DIR/manifest.jsonl's "answer" files (a behavior's last line wins; a path that does
    not exist here, e.g. on a pod, is read from DIR by its name), else every DIR/<behavior>.answer.txt, else a bare
    DIR/<behavior>.py program as the answer's python block."""
    out = {}
    if root:
        root = Path(root).expanduser()
        if (root / "manifest.jsonl").exists():  # a line whose answer file is not there (yet: answers append while runs finish) is skipped
            last = {r["behavior"]: Path(r["answer"]) for r in map(json.loads, open(root / "manifest.jsonl"))}
            found = {b: path if path.exists() else root / path.name for b, path in sorted(last.items())}
            return {b: path.read_text() for b, path in found.items() if path.exists()}
        for p in sorted(root.glob("*.answer.txt")):
            out[p.name[: -len(".answer.txt")]] = p.read_text()
        for p in sorted(root.glob("*.py")):
            if not p.name.startswith("."):  # macOS's ._ metadata files
                out.setdefault(p.stem, "```python\n" + p.read_text().strip() + "\n```")
    return out


def refuse_heldout(paths) -> None:
    """Held-out answers (search_heldout/, search_hard/) are evaluation baselines: no training input may come from those directories."""
    for path in paths:
        if path and {"search_heldout", "search_hard"} & set(Path(path).expanduser().resolve().parts):
            raise SystemExit(f"{path}: held-out answers are for evaluation only (--search-heldout), never training")


def rloo(keys: list[tuple]) -> np.ndarray:
    """(3) A_i = (wins_i - losses_i) / (n - 1): each answer against each other answer of its group by score.keys
    (leave-one-out), so equal answers get no signal and no scale enters."""
    n = len(keys)
    if n < 2:
        return np.zeros(n)
    return np.array([sum((keys[i] < keys[j]) - (keys[i] > keys[j]) for j in range(n) if j != i) / (n - 1) for i in range(n)])


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


def credit_advantages(tok, completion: list[int], text: str, source: str, episode: float, signs: dict) -> list[float]:
    """(1) Per-token advantages of one answer from its measured drops (edits.credit: +1 when dropping the name makes
    the answer worse, -1 when better): the tokens of a credited subcomponent's name get that sign, every other token
    the episode's advantage."""
    adv = [episode] * len(completion)
    offset = program_offset(text, source)
    if not signs or offset is None:
        return adv
    spans = token_spans(tok, completion, text)
    ends = [e for _, e in spans]  # nondecreasing
    for (a, b), v in signs.items():
        for t in range(bisect.bisect_right(ends, offset + a), len(spans)):  # the first token ending after the name's start on
            if spans[t][0] >= offset + b:
                break
            if spans[t][1] > spans[t][0]:
                adv[t] = v
    return adv


def edit_item(source: str, behavior: dict, seed: int) -> dict:
    """A scoring item of an edited answer."""
    return {"source": source, "explanation": "", "behavior": behavior, "seed": seed}


def memo(score):
    """score with every distinct item (task, program, seed, experiments, options) scored once:
    within one step the credit's base answers, refinement's start and edits that coincide repeat."""
    cache, hits = {}, [0]

    def key(it):
        return (it["behavior"].get("path", it["behavior"]["id"]), it["source"], it.get("seed"),
                json.dumps(it.get("options"), sort_keys=True), json.dumps(it.get("ir"), sort_keys=True))

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


def credit_groups(groups: list[dict], seed: int, args, tok, score, clock: dict):
    """(1) The credit drops (edits.names: --credit names sampled) of each group's distinct valid answers
    (--credit-answers: the best and K - 1 others at random), scored in one call under the step's seed, then each
    credited answer's token advantages."""
    requests, entries = [], []
    for g, grp in enumerate(groups):
        rng = random.Random(seed * 1009 + g)
        order = sorted(range(len(grp["keys"])), key=lambda j: grp["keys"][j])  # the best answer first
        sources = list(dict.fromkeys(grp["items"][j]["source"] for j in order if grp["valid"][j]))
        if args.credit_answers and len(sources) > args.credit_answers:  # the best and K - 1 others at random
            sources = sources[:1] + rng.sample(sources[1:], args.credit_answers - 1)
        for src in sources:
            spans = edits.names(src)
            if len(spans) > args.credit:
                spans = rng.sample(spans, args.credit)
            if spans:
                entries.append((g, src, spans, len(requests)))
                requests += [edit_item(x, grp["behavior"], seed) for x in [src] + [edits.drop(src, [sp]) for sp in spans]]
    if not requests:
        return
    scores = timed(clock, "credit", score, requests)
    for g, src, spans, k in entries:
        grp = groups[g]
        key = score_module.key
        base = key(scores[k])
        if not scores[k].get("valid", True):
            continue
        signs = {}
        for i, sp in enumerate(spans):
            dropped = key(scores[k + 1 + i])
            signs[sp] = (dropped > base) - (dropped < base)
        for j, it in enumerate(grp["items"]):
            if it["source"] == src and grp["valid"][j]:
                grp["credit"][j] = {f"{a}:{b}": v for (a, b), v in signs.items()}
                grp["token_advantages"][j] = credit_advantages(tok, grp["completions"][j], grp["texts"][j], src, float(grp["advantage"][j]), signs)


def rl2_sample(chosen: list[dict], step: int, args, pol, sampler, adapter: Path, clock: dict) -> list[dict]:
    """A group of --samples answers per behavior (the sampling half of rl2_groups), with vLLM's log-probabilities of
    the sampled tokens (the behavior policy of ppo_update's importance weights)."""
    n = args.samples
    prompts = [pol.question_ids(b) for b in chosen]
    comps = timed(clock, "sample", sampler, prompts, n, adapter, step)
    behavior = getattr(sampler, "token_logprobs", None) or [None] * (len(prompts) * n)
    return [{"behavior": b, "prompt": prompts[g], "completions": comps[g], "texts": [pol.tok.decode(c, skip_special_tokens=True) for c in comps[g]],
             "behavior_logprobs": behavior[g * n : (g + 1) * n]} for g, b in enumerate(chosen)]


def prompt_of(grp: dict, j: int) -> list[int]:
    """The prompt of a group's j-th completion: the question's, or a revision group's own per completion."""
    return grp["prompts"][j] if "prompts" in grp else grp["prompt"]


def feedback(s: dict) -> str:
    """The verifier's report on an answer as the oracle reads it before revising: why it could not run, or the KL in
    bits of the model's next-token distribution from the graph of its first k steps and that graph's description
    length, for each k, and each distinct reason something written is not part of the graph."""
    if not s.get("valid", True):
        return f"The verifier could not run your answer: {s.get('error')}"
    c = s.get("curve") or [[0.0, float("nan")]]
    lines = [f"no steps (the empty graph): {c[0][1]:.2f} bits"] + [f"first {k} steps: {kl:.2f} bits at a description length of {b:.0f} bits" for k, (b, kl) in enumerate(c[1:], 1)]
    report = ("The verifier ran the graph of your first k steps alone, over changed prompts of the text (one token replaced by a "
              "draw from the model's own prediction there), and measured the KL of the model's next-token distribution from the "
              "graph's:\n" + "\n".join(lines))
    dropped = s.get("dropped") or []
    if dropped:
        report += (f"\n{len(dropped)} things you wrote are not part of the graph (they still count in its description length):\n"
                   + "\n".join(f"- {why}" for why in dict.fromkeys(dropped)))
    return report


REVISE = ("Write an improved answer in the same format: as faithful as possible at every description length, the most "
          "important steps first, with its plain-English docstring and a comment line above each step.")


def revise_groups(groups: list[dict], step: int, args, pol, sampler, score, adapter: Path, clock: dict) -> list[dict]:
    """(9) The oracle revises each of its answers after reading the verifier's report on it (feedback): one revision per
    answer, its prompt the question, the answer and the report as a conversation; a question's revisions form a group,
    scored and ranked like first answers (rl2_score), so the oracle learns to use the verifier as a tool."""
    prompts, owners = [], []
    for grp in groups:
        for j, text in enumerate(grp["texts"]):
            prompts.append(pol.revision_ids(grp["behavior"], text, feedback(grp["scores"][j]) + "\n\n" + REVISE))
            owners.append(grp)
    comps = timed(clock, "sample", sampler, prompts, 1, adapter, step)
    behavior = getattr(sampler, "token_logprobs", None) or [None] * len(prompts)
    out, k = [], 0
    for grp in groups:
        n = len(grp["texts"])
        cs = [comps[k + j][0] for j in range(n)]
        out.append({"behavior": grp["behavior"], "prompt": prompts[k], "prompts": prompts[k:k + n], "completions": cs, "revision": True,
                    "texts": [pol.tok.decode(c, skip_special_tokens=True) for c in cs], "behavior_logprobs": behavior[k:k + n]})
        k += n
    return rl2_score(out, step, args, pol.tok, score, clock)


def rl2_score(groups: list[dict], step: int, args, tok, score, clock: dict) -> list[dict]:
    """The verifier half of rl2_groups, in place: every answer scored under the step's seed (one draw of changed
    prompts per question, shared by its group) and ranked within its group (score.keys);
    every sampled token gets its advantage: the episode's (3), replaced on credited names by the measured credit (1,
    --credit > 0). An invalid answer ranks below every valid one."""
    seed, n = step_seed(args, step), args.samples
    for grp in groups:
        grp["items"] = [item(x, grp["behavior"], seed) for x in grp["texts"]]
    scores = timed(clock, "score", score, [it for g in groups for it in g["items"]])
    for g, grp in enumerate(groups):
        sc = scores[g * n : (g + 1) * n]
        keys = score_module.keys(sc)
        A = rloo(keys)
        grp.update({"scores": sc, "keys": keys, "valid": np.array([bool(x["valid"]) for x in sc]), "advantage": A,
                    "token_advantages": [[float(A[j])] * len(c) for j, c in enumerate(grp["completions"])], "credit": [None] * n})
    if args.credit:
        credit_groups(groups, seed, args, tok, score, clock)
    if READER:
        reader_credit(groups, tok, READER[0], clock)
    return groups


def rl2_groups(chosen: list[dict], step: int, args, pol, sampler, score, adapter: Path, clock: dict) -> list[dict]:
    """rl2_sample, then rl2_score."""
    return rl2_score(rl2_sample(chosen, step, args, pol, sampler, adapter, clock), step, args, pol.tok, score, clock)


def reader_credit(groups: list[dict], tok, reader, clock: dict):
    """(8) The English of each answer, credited by the reader (reader.py): the bits it saves, ranked within the group
    (leave-one-out wins and losses, as (3)); its tokens (the docstring and comments) get that advantage added to the
    curve's, which the English shapes too, being written first. An answer without English saves 0 bits."""
    for grp in groups:
        sc = grp["scores"]
        texts = [reader_module.english(x) if x.get("valid", True) else "" for x in sc]
        todo = [j for j, t in enumerate(texts) if t and sc[j].get("events")]
        bits = [0.0] * len(sc)
        if todo:
            got = timed(clock, "reader", reader.bits, grp["behavior"], [texts[j] for j in todo], [sc[j]["events"] for j in todo])
            for j, b in zip(todo, got):
                bits[j] = b or 0.0
        A = rloo([(-b,) for b in bits])
        grp["reader_bits"], grp["reader_advantage"] = bits, A
        for j, (c, text, it) in enumerate(zip(grp["completions"], grp["texts"], grp["items"])):
            offset = program_offset(text, it["source"])
            if offset is None or A[j] == 0.0:
                continue
            spans = [(offset + a, offset + b) for a, b in english_spans(it["source"])]
            for t, (a, b) in enumerate(token_spans(tok, c, text)):
                if b > a and any(a < e and b > s0 for s0, e in spans):
                    grp["token_advantages"][j][t] += float(A[j])


def make_reader(args, pol, sampler):
    """The frozen base model as the English reader: vLLM's engine without the adapter, or the policy with its
    adapter disabled (transformers)."""
    from transformers import AutoTokenizer

    import mech

    tk = AutoTokenizer.from_pretrained(args.base)
    target_tk = mech.tokenizer(args.model)

    def tokens_of(ids):
        return [target_tk.decode([i]) for i in ids]

    def chat(q):
        return tk.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, enable_thinking=False, tokenize=False)

    if isinstance(sampler, VllmSampler):
        def generate(prompts):
            from vllm import SamplingParams

            if not sampler.ready:
                sampler.acquire()
            try:
                outs = sampler.llm.generate([chat(q) for q in prompts], SamplingParams(max_tokens=1, temperature=0.0, logprobs=20), use_tqdm=False)
            finally:
                if not sampler.held:
                    sampler.release()
            return [{d.decoded_token: d.logprob for d in o.outputs[0].logprobs[0].values()} for o in outs]
    else:
        def generate(prompts):
            out = []
            pol.train_mode(False)
            with torch.no_grad(), pol.model.disable_adapter():
                for q in prompts:
                    ids = torch.tensor([pol.tok.encode(chat(q), add_special_tokens=False)], device=pol.dev)
                    lp = torch.log_softmax(pol.model(input_ids=ids).logits[0, -1].float(), -1)
                    v, i = lp.topk(20)
                    out.append({pol.tok.decode([int(k)]): float(x) for x, k in zip(v.tolist(), i.tolist())})
            return out
    return reader_module.Reader(generate, tokens_of)


def informative(group: dict) -> bool:
    """(5) A group trains the policy only if some token's advantage is nonzero (not all equal, without credit)."""
    return any(a != 0.0 for adv in group["token_advantages"] for a in adv)


def expert_iteration(groups: list[dict], step: int, args, pol, score, clock: dict) -> list[dict]:
    """(2) edits.refine from each group's best valid answer under the step's seed (--refine rounds, --credit names
    sampled per round): an answer it improves comes back as the sampled answer with the program block replaced, for the
    SFT term and the DPO pair."""
    seed, out = step_seed(args, step), []
    for g, grp in enumerate(groups):
        if not grp["valid"].any():
            continue
        j = min(range(len(grp["keys"])), key=lambda i: grp["keys"][i])
        b, src, text = grp["behavior"], grp["items"][j]["source"], grp["texts"][j]
        offset = program_offset(text, src)
        if offset is None:
            continue

        def run(sources, b=b):
            return score([edit_item(x, b, seed) for x in sources])

        best, best_key, dropped = timed(clock, "refine", edits.refine, src, run, score_module.key, args.refine, args.credit, random.Random(seed * 1009 + g))
        if not dropped:
            continue
        new = text[:offset] + best + text[offset + len(src) :]
        out.append({"behavior": b["id"], "prompt": prompt_of(grp, j), "improved": pol.tok.encode(new, add_special_tokens=False) + [pol.end], "sampled": grp["completions"][j],
                    "text": new, "key": list(best_key), "sampled_key": list(grp["keys"][j]), "dropped": dropped})
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


def rl2_step(step: int, args, pol, sampler, score, pool: list[dict], adapter: Path, learner: Learner, logs: dict, started: float) -> dict:
    """One RL v2 step. --behaviors-per-step tasks drawn uniformly (by the step's seed), a group each (rl2_groups);
    groups without signal are dropped and refilled (5) by tasks drawn uniformly from the rest, up to --refill more
    sampling rounds; expert iteration (2, --refine > 0) from each group's best valid answer; --ppo-epochs clipped
    updates (4) on the kept groups, then one expert-iteration step (rl2_update)."""
    clock = {"sample": 0.0, "score": 0.0, "credit": 0.0, "refine": 0.0, "train": 0.0, "reader": 0.0}
    score = memo(score)  # one experiment draw per step: a repeated program's score is the same
    rng = random.Random(step_seed(args, step))
    chosen = rng.sample(pool, min(args.behaviors_per_step, len(pool)))
    groups = rl2_groups(chosen, step, args, pol, sampler, score, adapter, clock)
    used, refills = {b["id"] for b in chosen}, 0
    for _ in range(args.refill):
        need, rest = len(chosen) - sum(map(informative, groups)), [b for b in pool if b["id"] not in used]
        if need <= 0 or not rest:
            break
        extra = rng.sample(rest, min(need, len(rest)))
        used |= {b["id"] for b in extra}
        groups += rl2_groups(extra, step, args, pol, sampler, score, adapter, clock)
        refills += 1
    if args.revise:
        groups += revise_groups(groups, step, args, pol, sampler, score, adapter, clock)
    improved = expert_iteration(groups, step, args, pol, score, clock) if args.refine else []
    return rl2_update(step, groups, improved, refills, score.hits[0], clock, args, pol, sampler, learner, logs, started)


def rl2_update(step: int, groups: list[dict], improved: list[dict], refills: int, repeated: int, clock: dict, args, pol, sampler, learner: Learner,
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
    flat = [(prompt_of(grp, j), c, a, b) for grp in kept for j, (c, a, b) in enumerate(zip(grp["completions"], grp["token_advantages"], grp["behavior_logprobs"]))]
    stats = learner.ppo([x[0] for x in flat], [x[1] for x in flat], [x[2] for x in flat], args.beta, args.micro, args.ppo_epochs, clip_of(args),
                        [x[3] for x in flat]) if flat else {}
    stats.pop("logprob_sums", None)
    if improved:
        stats.update(learner.exit([x["prompt"] for x in improved], [x["improved"] for x in improved], [x["sampled"] for x in improved], args.exit_beta, args.micro))
    clock["train"] = time.time() - t
    keys = [k for g in groups for k in g["keys"]]
    scores = [x for g in groups for x in g["scores"] if x.get("valid", True)]
    best = [min(g["keys"]) for g in groups]
    row = {"step": step, "mode": "rl2", "async": bool(getattr(args, "async_rollouts", False)), "seed": step_seed(args, step), "groups": len(groups), "kept": len(kept), "refills": refills,
           "programs": len(keys), "valid_fraction": float(np.mean([k[0] == 0 for k in keys])),
           "mean_kl": float(np.mean([x["kl_bits"] for x in scores])) if scores else None, "mean_bits": float(np.mean([x["bits"] for x in scores])) if scores else None,
           "mean_steps": float(np.mean([x["steps"] for x in scores])) if scores else None,
           "best_area": float(np.mean([k[1] for k in best if k[0] == 0])) if any(k[0] == 0 for k in best) else None,
           "revision_best_area": float(np.mean([min(g["keys"])[1] for g in groups if g.get("revision") and min(g["keys"])[0] == 0])) if any(g.get("revision") and min(g["keys"])[0] == 0 for g in groups) else None,
           "mean_reader_bits": float(np.mean([b for g in groups for b in g.get("reader_bits", [])])) if any(g.get("reader_bits") for g in groups) else None,
           "credited": sum(c is not None for g in groups for c in g["credit"]), "improved": len(improved), "repeated_scores": repeated, **stats, "sampling": getattr(sampler, "stats", {}),
           "seconds": clock, "checker_seconds_total": TOTALS["checker_seconds"], "elapsed": time.time() - started}
    logs["train"].write(json.dumps(row) + "\n")
    logs["train"].flush()
    return row


def rl2_async(args, pol, sampler, score, pool: list[dict], adapter: Path, learner: Learner, logs: dict, started: float, stop) -> None:
    """rl2 with the checker overlapped (--async): while a thread scores, credits and refines step t's groups, vLLM
    samples step t + 1's with the policy not yet updated on step t, so those answers lag the trained policy by one
    step (ppo_update's importance weight corrects it, as AReaL's decoupled PPO and verl's rollout correction do). A step
    draws --behaviors-per-step tasks by its seed plus, uniformly from the rest, as many as the last step scored when
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
        return chosen + (rng.sample(rest, min(carry, len(rest))) if carry and rest else [])

    def check(groups: list[dict], step: int, clock: dict):
        memo_score = memo(score)
        rl2_score(groups, step, args, side.tok, memo_score, clock)
        improved = expert_iteration(groups, step, args, side, memo_score, clock) if args.refine else []
        return improved, memo_score.hits[0]

    carry = 0
    clock = {"sample": 0.0, "score": 0.0, "credit": 0.0, "refine": 0.0, "train": 0.0, "reader": 0.0}
    learner.save(adapter)
    groups = rl2_sample(draw(0, carry), 0, args, pol, sampler, adapter, clock)
    with ThreadPoolExecutor(1) as pool_thread:
        for step in range(args.steps):
            if stop():
                break
            future = pool_thread.submit(check, groups, step, clock)
            nxt = {"sample": 0.0, "score": 0.0, "credit": 0.0, "refine": 0.0, "train": 0.0, "reader": 0.0}
            t = time.time()
            upcoming = rl2_sample(draw(step + 1, carry), step + 1, args, pol, sampler, adapter, nxt) if step + 1 < args.steps else None
            improved, repeated = future.result()
            clock["overlap_wait"] = time.time() - t - nxt["sample"]  # checker time not hidden behind sampling
            row = rl2_update(step, groups, improved, 0, repeated, clock, args, pol, sampler, learner, logs, started)
            carry = min(args.behaviors_per_step, row["groups"] - row["kept"]) if args.refill else 0
            learner.save(adapter)
            if upcoming is None:
                break
            groups, clock = upcoming, nxt


class Learner:
    """The trainer behind one interface: every update the RL logic makes (SFT, PPO epochs, expert iteration) and the
    adapter it saves for the sampler. This one is local (the PEFT LoRA policy, AdamW with warmup, gradient norm clipped
    at 1); a managed backend (e.g. Tinker: sampling and training served, the checker local) replaces it and the samplers
    (callables (prompts, n, adapter, version) -> completions, with token_logprobs and stats) without touching rl2_step
    or rl2_update."""

    def __init__(self, pol: Policy, optimizer, warmup):
        self.pol, self.optimizer, self.warmup = pol, optimizer, warmup

    def step(self) -> float:
        norm = float(torch.nn.utils.clip_grad_norm_(self.pol.params, 1.0))
        self.optimizer.step()
        self.warmup.step()
        self.optimizer.zero_grad(set_to_none=True)
        return norm

    def begin(self):
        self.pol.train_mode(True)
        self.optimizer.zero_grad(set_to_none=True)

    def sft(self, prompts, completions, micro: int) -> dict:
        self.begin()
        stats = sft_update(self.pol, prompts, completions, micro)
        return {**stats, "grad_norm": self.step()}

    def ppo(self, prompts, completions, token_adv, beta: float, micro: int, epochs: int, clip: dict, behavior=None) -> dict:
        self.begin()
        return ppo_update(self.pol, prompts, completions, token_adv, beta, micro, epochs, clip, self.optimizer, self.warmup, behavior)

    def exit(self, prompts, improved, sampled, beta: float, micro: int) -> dict:
        self.begin()
        stats = exit_update(self.pol, prompts, improved, sampled, beta, micro)
        return {**stats, "exit_grad_norm": self.step()}

    def save(self, path: Path):
        self.pol.save(path)


def sft_examples(args, pol, pool: list[dict]) -> tuple[list, list]:
    """--mode sft's data as token ids (prompt, completion): the search answer of every TRAINING task (--search) as the
    answer to its prompt, and --data examples (JSONL of {"messages": [user, assistant]} or {"prompt", "completion"}). The
    completion ends with <|im_end|>; an example longer than --max-model-len is left out."""
    by_id = {b["id"]: b for b in pool}
    end = [pol.end]
    target = (lambda text: pol.parts.reg.rewrite(text)) if getattr(pol, "parts", None) is not None else (lambda text: text)  # noqa: E731  addresses -> part tokens
    programs = [(pol.question_ids(by_id[bid]), fit(pol.tok, target(text.strip()), args.max_tokens) + end) for bid, text in sorted(SEARCH.items()) if bid in by_id]
    questions = []
    for path in args.data or []:
        for line in open(os.path.expanduser(path)):
            q = json.loads(line)
            user, answer = (q["messages"][0]["content"], q["messages"][1]["content"]) if "messages" in q else (q["prompt"], q["completion"])
            questions.append((pol.prompt_ids(user), pol.tok.encode(target(answer), add_special_tokens=False) + end))
    keep = lambda xs: [(p, c) for p, c in xs if len(p) + len(c) <= args.max_model_len]  # noqa: E731
    return keep(programs), keep(questions)


def cut(text: str, budget: int, encode) -> str:
    """An answer with its program cut to the most first steps (native.first_steps) whose answer fits `budget` tokens
    (the oracle's output limit; encode: text -> token ids): a search answer is as long as its search ran, the
    oracle's as long as it may write."""
    import native

    if len(encode(text)) <= budget:
        return text
    source, _ = split_answer(text)
    parts = native.steps_of(source)
    if parts is None:
        return text
    lo, hi = 0, len(parts[1])  # the most steps that fit, by bisection (length grows with steps)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(encode(text.replace(source, native.first_steps(source, mid)))) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text.replace(source, native.first_steps(source, lo))


def fit(tok, text: str, budget: int) -> list[int]:
    """cut()'s answer as the oracle's tokens."""
    def encode(t):
        return tok.encode(t, add_special_tokens=False)
    return encode(cut(text, budget, encode))


def sft(args, pol, pool, learner: Learner, log) -> dict:
    """--sft-steps steps of SFT: each batch draws --batch examples, a program example with probability
    --program-share and a question otherwise; loss = -(1/B) sum_e sum_t log pi(y_et) (sft_update)."""
    programs, questions = sft_examples(args, pol, pool)
    if not programs and not questions:
        raise SystemExit("no SFT examples (--search, --data)")
    rng = random.Random(args.seed)
    meta = {"program_examples": len(programs), "question_examples": len(questions), "program_behaviors": len(programs)}
    log.write(json.dumps({"sft": meta}) + "\n")
    started = time.time()
    for step in range(args.sft_steps):
        if args.hours and time.time() - started > 3600 * args.hours:
            break
        batch = [rng.choice(programs) if programs and (not questions or rng.random() < args.program_share) else rng.choice(questions) for _ in range(args.batch)]
        stats = learner.sft([p for p, _ in batch], [c for _, c in batch], args.micro)
        tokens = sum(len(c) for _, c in batch)
        log.write(json.dumps({"step": step, "loss_nats_per_example": stats["loss"], "bits_per_token": stats["loss"] * len(batch) / tokens / np.log(2), **{k: v for k, v in stats.items() if k != "loss"},
                              "programs": sum(1 for x in batch if x in programs), "elapsed": time.time() - started}) + "\n")
        log.flush()
    return meta


ORACLE_RUNS = Path.home() / "mpd-data/graph_oracle/runs/oracle"


def shares(x: dict) -> dict:
    """An answer in the reporting terms: area (score.key: score.area over the question's range), kl (the whole answer's
    run-alone KL in bits), reproduces = 1 - kl / the empty graph's, bits (its description length), steps, nodes, edges."""
    if not x.get("valid", True) or not x.get("curve"):
        return {"area": None, "kl": None, "reproduces": None, "bits": None, "steps": None, "nodes": None, "edges": None, "necessity": None, "adversarial": None, "shared_fraction": None, "transfer_area": None, "transfer_minus_own": None}
    kl, empty = x.get("kl_bits"), x["curve"][0][1]
    return {"area": score_module.key(x)[1], "kl": kl, "reproduces": 1.0 - kl / empty if empty else None, "bits": x.get("bits"), "steps": x.get("steps"),
            "nodes": x.get("nodes"), "edges": x.get("edges"), "necessity": x.get("necessity_kl_bits"), "adversarial": x.get("adversarial_kl_bits"),
            "shared_fraction": x.get("shared_fraction"), "transfer_area": x.get("transfer_area"), "transfer_minus_own": x.get("transfer_minus_own")}


def relative_edges(source: str, task: dict) -> set:
    """An answer's connections in coordinates relative to its target: (writer, writer offset, reader or "out", reader
    offset), subcomponents as "<p:L.S.I>"."""
    import mech

    if not task.get("prompts"):
        return set()
    ir = mech.trace_inline(source, task.get("model", "vpd4l"), task)
    if not ir["valid"]:
        return set()
    t = task["prompts"][0]["target_positions"][0]
    codes = {v: k for k, v in mech.SITES.items()}
    nd = [(f"<p:{a}.{codes[b]}.{d}>", c - t) for a, b, c, d in ir["graph"]["nodes"]]
    return {(*nd[w], *nd[r]) for r, w in ir["graph"]["parents"]} | {(*nd[w], "out", 0) for w in ir["graph"]["out"]}


def shared(groups: list[tuple[dict, list, dict]]) -> None:
    """Each question's best answer gets "shared_fraction": the share of its connections (relative_edges) that the best
    answer to some other question of the set also has, the machinery its prediction shares with others rather than
    its own."""
    best = [min(mine, key=lambda m: score_module.key(m[1])) for _, mine, _ in groups]
    edges = [relative_edges(src, b) if x.get("valid", True) else set() for (b, _, _), (src, x) in zip(groups, best)]
    count = {}
    for es in edges:
        for e in es:
            count[e] = count.get(e, 0) + 1
    for (_, x), es in zip(best, edges):
        x["shared_fraction"] = sum(count[e] > 1 for e in es) / len(es) if es else None


def summarize(name: str, step: int, groups: list[tuple[dict, list, dict]], log) -> dict:
    """Rows per question and the set's summary from groups of (question, [(source, score)] of the oracle, {baseline
    name: score}). Per question: the oracle's best answer by score.key, and whether its curve's area is below the
    search's (beats_search) and VPD's (beats_vpd)."""

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    rows = []
    for b, mine, base in groups:
        key = score_module.key
        keys = [key(x) for _, x in mine]
        j = min(range(len(keys)), key=keys.__getitem__)
        best = mine[j][1]
        firsts = [k for (_, x), k in zip(mine, keys) if not x.get("revision")]
        row = {"set": name, "step": step, "behavior": b["id"], "valid_fraction": float(np.mean([k[0] == 0 for k in keys])), "best": shares(best),
               "best_first_area": min(firsts)[1] if firsts and min(firsts)[0] == 0 else None, "best_is_revision": bool(best.get("revision")),
               "baselines": {n: shares(x) for n, x in base.items()}, "best_source": mine[j][0]}
        for n in ("search", "vpd"):
            if n in base and base[n].get("valid") and keys[j][0] == 0:
                row[f"beats_{n}"] = keys[j][1] < key(base[n])[1]
        if best.get("adversarial_kl_bits") is not None and (base.get("vpd") or {}).get("adversarial_kl_bits") is not None:
            row["adversarial_minus_vpd"] = best["adversarial_kl_bits"] - base["vpd"]["adversarial_kl_bits"]  # the same adversary, VPD's answer the reference
        rows.append(row)
        log.write(json.dumps(row) + "\n")
    keys_ = ("area", "kl", "reproduces", "bits", "steps", "nodes", "edges", "necessity", "adversarial", "shared_fraction", "transfer_area", "transfer_minus_own")
    return {"questions": len(rows), "valid_fraction": mean([r["valid_fraction"] for r in rows]), "best_first_area": mean([r["best_first_area"] for r in rows]),
            "best_is_revision": mean([r["best_is_revision"] for r in rows]),
            "best_beats_search": mean([r.get("beats_search") for r in rows]), "best_beats_vpd": mean([r.get("beats_vpd") for r in rows]),
            "adversarial_minus_vpd": mean([r.get("adversarial_minus_vpd") for r in rows]),
            "best": {k: mean([r["best"][k] for r in rows]) for k in keys_},
            "baselines": {n: {k: mean([r["baselines"][n][k] for r in rows if n in r["baselines"]]) for k in keys_} for n in sorted({n for _, _, base in groups for n in base})}}


def evaluate(sets: dict[str, list[dict]], pol, sampler, score, args, adapter: Path, version: int, log, step: int) -> dict:
    """Answers of the current policy on each evaluation set (--samples per task at temperature 1) and the baselines,
    all under one seed (--eval-seed, never a training step's). Every answer and its score go to eval_samples.jsonl;
    each task's best answer to runs/oracle/<task>.<run>.json; summarize gives the numbers."""
    summary = {}
    run = args.run_name or Path(args.out).name
    runs = Path(args.oracle_runs) if getattr(args, "oracle_runs", None) else ORACLE_RUNS
    runs.mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "eval_samples.jsonl", "a") as samples:
        for name, pool in sets.items():
            if not pool:
                continue
            prompts = [pol.question_ids(b) for b in pool]
            groups = sampler(prompts, args.samples, adapter, version)
            answers = [(b, pol.tok.decode(c, skip_special_tokens=True)) for b, g in zip(pool, groups) for c in g]
            items = [item(t, b, args.eval_seed) for b, t in answers]
            options = {"necessity": True} if getattr(args, "necessity", False) else None
            if getattr(args, "revise", False):  # the oracle with its tool: each answer revised once after the verifier's report
                first = score(items)
                rev = sampler([pol.revision_ids(b, t, feedback(x) + "\n\n" + REVISE) for (b, t), x in zip(answers, first)], 1, adapter, version)
                revised = [(b, pol.tok.decode(c[0], skip_special_tokens=True)) for (b, _), c in zip(answers, rev)]
                answers = [x for g in range(len(pool)) for x in answers[g * args.samples:(g + 1) * args.samples] + revised[g * args.samples:(g + 1) * args.samples]]
                items = [item(t, b, args.eval_seed) for b, t in answers]
            for it in items:
                it["options"] = options
            base = [(b, n, src) for b in pool for n, src in (baselines(b).items() if args.baselines else [])]
            scores = score(items + [{**item(src, b, args.eval_seed), "options": options} for b, _, src in base])
            per_base = {}
            for (b, n, src), x in zip(base, scores[len(items) :]):
                per_base.setdefault(b["id"], {})[n] = x
                samples.write(json.dumps({"set": name, "step": step, "run": run, "behavior": b["id"], "behavior_path": b.get("path"), "program": n, "source": src, "score": x}) + "\n")
            out = []
            per = args.samples * (2 if getattr(args, "revise", False) else 1)  # first answers, then their revisions
            for g, b in enumerate(pool):
                mine = [(it["source"], x) for it, x in zip(items[g * per : (g + 1) * per], scores[g * per : (g + 1) * per])]
                for k, (it, (src, x)) in enumerate(zip(items[g * per : (g + 1) * per], mine)):
                    x["revision"] = k >= args.samples
                    samples.write(json.dumps({"set": name, "step": step, "run": run, "behavior": b["id"], "behavior_path": b.get("path"), "program": "oracle", "source": src,
                                              "explanation": it["explanation"], "score": x}) + "\n")
                ks = score_module.keys([m[1] for m in mine])
                src, x = mine[min(range(len(mine)), key=ks.__getitem__)]
                (runs / f"{b['id']}.{run}.json").write_text(json.dumps({"behavior": b["id"], "model": b.get("model"), "source": src, "score": x,
                                                                               "baselines": per_base.get(b["id"], {}), "seed": args.eval_seed, "set": name, "step": step}))
                out.append((b, mine, per_base.get(b["id"], {})))
            if out and getattr(args, "adversarial", False):  # VPD's adversary, one per method shared across the set's questions
                import scorer as scorer_module

                tasks = [b for b, _, _ in out]
                best = [min(mine, key=lambda m: score_module.key(m[1])) for _, mine, _ in out]
                mine_adv = scorer_module.adversarial(tasks, [src for src, _ in best], args.eval_seed)
                vpd_adv = scorer_module.adversarial(tasks, ["vpd"] * len(tasks), args.eval_seed)
                for (_, x), (_, _, base), a, v in zip(best, out, mine_adv, vpd_adv):
                    x["adversarial_kl_bits"] = a
                    if "vpd" in base:
                        base["vpd"]["adversarial_kl_bits"] = v
            if out and getattr(args, "transfer", False):  # each best answer's program on the next question's text
                best = [min(mine, key=lambda m: score_module.key(m[1])) for _, mine, _ in out]
                moved = score([{**item("```python\n" + src + "\n```", out[(g + 1) % len(out)][0], args.eval_seed), "options": None} for g, (src, _) in enumerate(best)])
                for (_, x), y in zip(best, moved):
                    x["transfer_area"] = score_module.key(y)[1] if y.get("valid", True) and y.get("curve") else None
                for g, (_, x) in enumerate(best):
                    there = best[(g + 1) % len(best)][1]
                    x["transfer_minus_own"] = (x["transfer_area"] - score_module.key(there)[1]) if x["transfer_area"] is not None and there.get("curve") else None
            if out:
                shared(out)
                summary[name] = summarize(name, step, out, log)
            if getattr(sampler, "stats", None):
                summary.setdefault(name, {})["sampling"] = dict(sampler.stats)
    log.write(json.dumps({"summary": summary, "step": step}) + "\n")
    log.flush()
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
    if pol.parts is not None and isinstance(sampler, VllmSampler):
        sampler.reload(pol.materialize(vocab_dir(), args.base))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sft", "rl2", "eval"], required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--init", help="SFT adapter the policy starts from and is held to (pi_ref): a PEFT directory")
    ap.add_argument("--model", required=True, help="target model whose predictions are explained: vpd4l")
    ap.add_argument("--behaviors", default=str(BEHAVIORS), help="the tasks: DIR/<model>/<id>.json (text.py)")
    ap.add_argument("--scorer", choices=sorted(SCORERS), default="native")
    ap.add_argument("--scorer-device", help="the native scorer's torch device (default: cuda, else mps, else cpu)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--hours", type=float, help="stop and save after this many hours")
    ap.add_argument("--checker-hours", type=float, help="stop and save once training scores (with credit and refinement) used this many scorer hours")
    ap.add_argument("--behaviors-per-step", type=int, default=8)
    ap.add_argument("--samples", type=int, default=8, help="answers per task per step (the group)")
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--warmup", type=int, default=20, help="optimizer steps of linear learning-rate warmup")
    ap.add_argument("--beta", type=float, default=0.04, help="rl2: weight of the KL to pi_ref")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--sampler", choices=["auto", "vllm", "hf"], default="auto")
    ap.add_argument("--evidence", action="store_true", help="the oracle also reads the model's activations: one input token per position and weight matrix, the subcomponents' features weighted by their activations through the part-token maps (needs --part-tokens)")
    ap.add_argument("--transfer", action="store_true", help="eval: run each question's best answer program on the next question's text (does the mechanism transfer, or only this text's circuit?)")
    ap.add_argument("--adversarial", action="store_true", help="eval: VPD's adversary (native.adversarial), one shared across the set for the oracle's best answers and one for VPD's answers; reported, never trained on")
    ap.add_argument("--necessity", action="store_true", help="eval: also measure each answer's necessity (native.necessity: the model with the answer's subcomponents removed)")
    ap.add_argument("--swap-evidence", action="store_true", help="eval: give each held-out question another text's activations (if answers do not get worse, the oracle does not read them)")
    ap.add_argument("--part-tokens", help="part tokens: the registry file of part_tokens.py build; the projections train with the LoRA and vLLM gets the rows in place")
    ap.add_argument("--materialize-every", type=int, default=0, help="with --part-tokens: also rewrite the checkpoint and restart vLLM every K steps (0: only at the start; the rows are copied in place before every sampling call)")
    ap.add_argument("--share-gpu", action="store_true", help="one GPU for vLLM and the trainer: vLLM sleeps (weights to host) while training and the trainer moves to the host while sampling, so --gpu-memory can be 0.8")
    ap.add_argument("--hf-batch", type=int, default=16, help="sequences per transformers generate call (the Mac / CPU sampler)")
    ap.add_argument("--gpu-memory", type=float, default=0.85, help="vLLM's share of its GPU (lower it when the trainer shares the GPU)")
    ap.add_argument("--eval-every", type=int, default=0, help="rl2: evaluate every E training steps and at the end (0: only --mode eval)")
    ap.add_argument("--skip-first-eval", action="store_true", help="no evaluation at step 0 (the starting policy is evaluated once elsewhere, e.g. by its SFT run)")
    ap.add_argument("--eval-seed", type=int, default=1_000_003, help="the evaluation's seed (training steps use their index)")
    ap.add_argument("--eval-behaviors", type=int, default=0, help="evaluate the first this many held-out tasks in text order (0: all)")
    ap.add_argument("--no-baselines", dest="baselines", action="store_false", help="skip scoring the baselines (the empty graph, VPD's answer, the search's) in evaluation")
    ap.add_argument("--data", nargs="*", help="sft: JSONL files of further examples ({'messages': [user, assistant]} or {'prompt', 'completion'})")
    ap.add_argument("--program-share", type=float, default=1.0, help="sft: probability that a batch example is a search answer rather than a --data example")
    ap.add_argument("--sft-steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8, help="sft: examples per optimizer step")
    ap.add_argument("--oracle-runs", help="directory of the per-task best-answer files (default ~/mpd-data/graph_oracle/runs/oracle; a pod writes under its outputs)")
    ap.add_argument("--run-name", help="the run's name in runs/oracle/<task>.<run>.json (default: the --out directory's name)")
    ap.add_argument("--search", help="bootstrap search answers of the training questions (DIR/<task>.py, native.py search, or <task>.answer.txt): SFT answers")
    ap.add_argument("--search-heldout", help="the search's answers to the held-out questions: evaluation baselines only")
    ap.add_argument("--score-workers", type=int, default=1, help="processes scoring answers at once (a pod's GPU: 8)")
    ap.add_argument("--eval-split", choices=("heldout", "hard"), default="heldout", help="the questions evaluation asks (text.py's hard split: "
                    "predictions the text does not suggest; its search answers in texts/search_hard)")
    ap.add_argument("--revise", action="store_true", help="rl2: a second round in which the oracle revises each answer after reading the verifier's report on it (revise_groups)")
    ap.add_argument("--reader", action="store_true", help="rl2: credit each answer's English by the frozen base reader's bits (reader.py)")
    ap.add_argument("--credit", type=int, default=16, help="rl2: subcomponent names whose drop is scored per answer for per-token credit (0: episode advantages only)")
    ap.add_argument("--credit-answers", type=int, default=0, help="rl2: answers credited per group: the best valid one and K - 1 other distinct valid ones drawn at random (0: every distinct valid answer)")
    ap.add_argument("--refill", type=int, default=1, help="rl2: sampling rounds that replace groups without signal by tasks drawn uniformly")
    ap.add_argument("--refine", type=int, default=2, help="rl2: rounds of edits.refine from each group's best answer for expert iteration (0: off)")
    ap.add_argument("--ppo-epochs", type=int, default=2, help="rl2: optimizer steps per scored batch (PPO clipped ratio)")
    ap.add_argument("--async", dest="async_rollouts", action="store_true", help="rl2: the checker scores step t while vLLM samples step t + 1 (one step of policy lag, importance-weighted)")
    ap.add_argument("--clip", type=float, default=0.2, help="rl2: PPO ratio clip below 1 (eps_low)")
    ap.add_argument("--clip-high", type=float, default=0.28, help="rl2: PPO ratio clip above 1 (eps_high; DAPO's clip-higher, 0.28)")
    ap.add_argument("--dual-clip", type=float, default=3.0, help="rl2: dual-clip bound c on a negative advantage's ratio (verl's clip_ratio_c)")
    ap.add_argument("--tis-cap", type=float, default=2.0, help="rl2: truncation of the importance weight pi_prox / pi_behavior (verl's rollout_is_threshold)")
    ap.add_argument("--exit-beta", type=float, default=0.1, help="rl2: inverse temperature of expert iteration's DPO pair")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lr = args.lr if args.lr is not None else {"sft": 1e-4}.get(args.mode, 1e-5)
    refuse_heldout([args.search, *(args.data or [])])
    SEARCH.update(read_answers(args.search))
    HELDOUT_SEARCH.update(read_answers(args.search_heldout))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "lr": lr}, indent=1))
    if args.revise and args.async_rollouts:
        raise SystemExit("--revise samples a second round inside the step: the synchronous loop (drop --async)")
    if args.part_tokens:  # in-process vLLM engine: part rows are copied into its weights before each sampling call
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        if args.share_gpu and args.materialize_every:
            raise SystemExit("--materialize-every restarts vLLM, which sleep mode (--share-gpu) allows once per process")
    use_vllm = args.sampler == "vllm" or (args.sampler == "auto" and torch.cuda.is_available() and __import__("importlib").util.find_spec("vllm") is not None)
    sampler = None
    if use_vllm:  # vLLM first, on the first visible GPU, before the trainer touches CUDA
        from transformers import AutoTokenizer

        rank = json.loads((Path(args.init) / "adapter_config.json").read_text())["r"] if args.init else args.lora_rank
        sampler = VllmSampler(args, rank, AutoTokenizer.from_pretrained(args.base).convert_tokens_to_ids("<|im_end|>"), None if args.part_tokens else args.base)
    dev = torch.device(f"cuda:{torch.cuda.device_count() - 1}" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    pol = Policy(args, dev)
    pol.evidence, pol.acts_fn = {}, None
    if args.evidence:
        if pol.parts is None:
            raise SystemExit("--evidence reads activations through the part-token maps: give --part-tokens")
        import native

        pol.acts_fn = native.Native(str(dev) if dev.type != "cuda" else "cuda").activations
    if isinstance(sampler, VllmSampler):
        sampler.policy = pol
        if pol.parts is not None:  # vLLM starts once, on a checkpoint with the extended vocabulary; the rows then come in place
            sampler.reload(pol.materialize(vocab_dir(), args.base))
            sampler.rows = pol.part_rows
    if sampler is None:
        sampler = HfSampler(pol, args.max_tokens, args.hf_batch)
    score = SCORERS[args.scorer]
    scorer.DEVICE = args.scorer_device
    scorer.WORKERS = args.score_workers
    if args.reader:
        READER.append(make_reader(args, pol, sampler))
    root = Path(args.behaviors)
    pool = behaviors(root, args.model, "train")
    sets = {args.eval_split: behaviors(root, args.model, args.eval_split)}
    if args.eval_behaviors:  # the first N in text order (arbitrary Pile rows), the same at every evaluation and the ones the search answers first
        sets = {k: sorted(v, key=lambda b: int(b["id"][4:]) if b["id"][4:].isdigit() else 0)[:args.eval_behaviors] for k, v in sets.items()}
    if args.swap_evidence:  # each held-out question reads the activations of the next held-out text at least as long, cut to its length
        held = sets[args.eval_split]
        pol.swap = {}
        for i, b in enumerate(held):
            T = len(b["prompts"][0]["token_ids"])
            other = next((o for o in held[i + 1:] + held[:i] if len(o["prompts"][0]["token_ids"]) >= T), None)
            if other is not None:
                pol.swap[b["id"]] = other["prompts"][0]["token_ids"][:T]
    adapter = out / "adapter"
    if args.search_heldout:  # the search's baseline at the oracle's output budget
        for bid, text in list(HELDOUT_SEARCH.items()):
            HELDOUT_SEARCH[bid] = cut(text, args.max_tokens, lambda t: pol.tok.encode(t, add_special_tokens=False))
    if args.mode == "eval":
        pol.save(adapter)
        print(json.dumps(evaluate(sets, pol, sampler, score, args, adapter, 0, open(out / "eval.jsonl", "a"), 0)))
        return
    if not pool:
        raise SystemExit(f"no train tasks under {root / args.model}")
    optimizer = torch.optim.AdamW(pol.param_groups(lr), weight_decay=0.0)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / args.warmup))  # linear over --warmup optimizer steps
    learner = Learner(pol, optimizer, warmup)  # every update goes through it (a managed backend would replace it and the sampler)
    if args.mode == "sft":
        sft(args, pol, pool, learner, open(out / "train.jsonl", "a"))
        learner.save(adapter)
        print(json.dumps(evaluate(sets, pol, sampler, score, args, adapter, 1, open(out / "eval.jsonl", "a"), args.sft_steps)))
        return
    eval_log = open(out / "eval.jsonl", "a")
    logs = {"train": open(out / "train.jsonl", "a"), "samples": open(out / "samples.jsonl", "a"), "improved": open(out / "improved.jsonl", "a")}
    started = time.time()

    def over() -> bool:
        return bool((args.hours and time.time() - started > 3600 * args.hours) or (args.checker_hours and TOTALS["checker_seconds"] > 3600 * args.checker_hours))

    for step in range(args.steps):
        if over():
            break
        learner.save(adapter)
        if args.materialize_every and step and step % args.materialize_every == 0:
            refresh_parts(pol, sampler, out, args)
        if args.eval_every and step % args.eval_every == 0 and not (step == 0 and args.skip_first_eval):
            evaluate(sets, pol, sampler, score, args, adapter, step, eval_log, step)
        if args.async_rollouts:  # its own loop: the checker overlaps the next step's sampling
            rl2_async(args, pol, sampler, score, pool, adapter, learner, logs, started, over)
            break
        rl2_step(step, args, pol, sampler, score, pool, adapter, learner, logs, started)
    learner.save(adapter)
    if args.eval_every:
        evaluate(sets, pol, sampler, score, args, adapter, args.steps, eval_log, args.steps)


if __name__ == "__main__":
    main()

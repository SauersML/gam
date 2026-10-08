"""GPU check of part tokens in vLLM (#2951; needs vLLM and a CUDA GPU): python test_vllm_rows.py [BASE]

A policy with stand-in part tokens (random projections of random read/write vectors) writes the
extended-vocabulary checkpoint, vLLM restarts on it in-process with --share-gpu, the policy's projections
change, and the rows copied into vLLM before sampling must equal the policy's; then vLLM's log-probability
of a completion that contains part tokens must match the policy's."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402


class Parts(torch.nn.Module):
    def __init__(self, d, count=5):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.register_buffer("read", torch.randn(count, 16, generator=g))
        self.register_buffer("write", torch.randn(count, 16, generator=g))
        self.p_in, self.p_out = torch.nn.Linear(16, d), torch.nn.Linear(16, d)
        self.count = count

    def tokens(self):
        return [f"<p:test{i}>" for i in range(self.count)]

    def input_rows(self):
        return self.p_in(self.read) * 0.02

    def output_rows(self):
        return self.p_out(self.write) * 0.02


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
    from transformers import AutoConfig

    d = AutoConfig.from_pretrained(base).hidden_size
    sys.modules["part_tokens"] = type(sys)("part_tokens")
    sys.modules["part_tokens"].load = lambda spec: Parts(d)
    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(base=base, init=None, lora_rank=8, part_tokens="stand-in", share_gpu=True, gpu_memory=0.6, max_model_len=2048,
                                  max_tokens=16, seed=0)
        sampler = train.VllmSampler(args, 8, 0)
        pol = train.Policy(args, torch.device("cuda"))
        sampler.policy, sampler.end = pol, pol.end
        sampler.reload(pol.materialize(Path(tmp) / "vocab", base))
        sampler.rows = pol.part_rows
        with torch.no_grad():
            for p in pol.parts.parameters():
                p.add_(0.1)  # the projections move after the checkpoint was written
        pol.save(Path(tmp) / "adapter")
        prompt = pol.prompt_ids("Name a part.")
        out = sampler([prompt], 2, Path(tmp) / "adapter", 0)
        first, rows_in, rows_out = pol.part_rows()
        got = sampler.llm.apply_model(lambda m: (m.model.embed_tokens.weight.data[first : first + rows_in.shape[0]].float().cpu(),
                                                 m.lm_head.weight.data[first : first + rows_out.shape[0]].float().cpu()))[0]
        assert torch.allclose(got[0], rows_in.float().cpu(), atol=1e-2) and torch.allclose(got[1], rows_out.float().cpu(), atol=1e-2), "rows differ"
        completion = [first + 1, first + 3, pol.end]
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        sampler.llm.wake_up()
        sampler.push_rows()
        res = sampler.llm.generate([{"prompt_token_ids": prompt + completion}], SamplingParams(max_tokens=1, prompt_logprobs=0),
                                   lora_request=LoRARequest("t", 1, str(Path(tmp) / "adapter")), use_tqdm=False)[0]
        sampler.llm.sleep(level=1)
        v = sum(res.prompt_logprobs[len(prompt) + k][t].logprob for k, t in enumerate(completion))
        lp, mask = pol.token_logprobs([prompt], [completion])
        h = float((lp * mask).sum())
        print({"samples": [len(c) for c in out[0]], "vllm_logprob": v, "policy_logprob": h})
        assert abs(v - h) < 0.05 * len(completion) + 0.05, (v, h)
    print("ok: part rows copied into vLLM equal the policy's; part-token log-probabilities agree")


if __name__ == "__main__":
    main()

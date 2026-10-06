"""Does the pinned vLLM generate what transformers generates (#2951)? Greedy continuations of a few
chat prompts by vLLM (offline LLM) and by Hugging Face transformers on the same GPU, both in bfloat16,
and the prompt log-probabilities vLLM returns against transformers' (the reader's code path). Prints one
JSON object: per prompt whether the greedy tokens agree up to the first disagreement, and the largest
log-probability difference.

  vllm_check.py --model Qwen/Qwen3-1.7B [--tokens 32]
"""

from __future__ import annotations

import argparse
import json

import torch

PROMPTS = [
    "Name three prime numbers and explain why each is prime.",
    "Write a short sentence about the ocean.",
    "What is the capital of France? Answer in one word.",
    "Count from one to ten in words.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=32)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    # The template as text, then token ids (transformers 5 returns a BatchEncoding from tokenize=True).
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False, tokenize=False) for p in PROMPTS]
    prompts = [tok.encode(t, add_special_tokens=False) for t in texts]
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.45, enable_prefix_caching=True, seed=0)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], SamplingParams(max_tokens=args.tokens, temperature=0.0, prompt_logprobs=0), use_tqdm=False)
    v_tokens = [list(o.outputs[0].token_ids) for o in outs]
    v_lp = [[o.prompt_logprobs[j][p[j]].logprob for j in range(1, len(p))] for o, p in zip(outs, prompts)]
    del llm
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    rows = []
    for p, vt, vl in zip(prompts, v_tokens, v_lp):
        ids = torch.tensor([p], device="cuda")
        with torch.no_grad():
            gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.tokens, do_sample=False)[0, len(p) :].tolist()
            lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1).gather(-1, ids[0, 1:, None])[:, 0].tolist()
        agree = next((i for i, (a, b) in enumerate(zip(vt, gen)) if a != b), min(len(vt), len(gen)))
        rows.append({"prompt": tok.decode(p)[-60:], "greedy_tokens_agreeing": agree, "of": min(len(vt), len(gen)),
                     "max_prompt_logprob_difference": max(abs(a - b) for a, b in zip(vl, lp)),
                     "vllm": tok.decode(vt), "transformers": tok.decode(gen)})
    import transformers
    import vllm

    print(json.dumps({"vllm": vllm.__version__, "transformers": transformers.__version__, "torch": torch.__version__, "model": args.model, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()

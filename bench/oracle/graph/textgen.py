"""Greedy text generation with a frozen local model (Qwen3 chat template, thinking off), for the graph
oracle's text tools (rebuild.py, paraphrase.py). No hosted models: transformers on the Mac or one GPU,
vLLM on a GPU host.

    gen = Generator("Qwen/Qwen3-8B", backend="vllm", max_tokens=1500)
    answers = gen(["user message 1", "user message 2"])
"""

from __future__ import annotations


class Generator:
    def __init__(self, model: str, backend: str = "transformers", max_tokens: int = 1024, batch: int = 8):
        self.model_name, self.backend, self.max_tokens, self.batch = model, backend, max_tokens, batch
        if backend == "vllm":
            from vllm import LLM

            self.llm = LLM(model=model, dtype="bfloat16", enable_prefix_caching=True, seed=0)
            self.tokenizer = self.llm.get_tokenizer()
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.tokenizer.padding_side = "left"
            dtype = torch.float32 if self.dev.type == "cpu" else torch.bfloat16
            self.llm = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.dev).eval()

    def chat(self, user: str) -> str:
        return self.tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)

    def __call__(self, users: list[str]) -> list[str]:
        texts = [self.chat(u) for u in users]
        if self.backend == "vllm":
            from vllm import SamplingParams

            outs = self.llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=self.max_tokens), use_tqdm=False)
            return [o.outputs[0].text for o in outs]
        import torch

        answers = []
        for s in range(0, len(texts), self.batch):
            enc = self.tokenizer(texts[s : s + self.batch], return_tensors="pt", padding=True).to(self.dev)
            with torch.no_grad():
                out = self.llm.generate(**enc, max_new_tokens=self.max_tokens, do_sample=False,
                                        pad_token_id=self.tokenizer.pad_token_id)
            answers += self.tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        return answers

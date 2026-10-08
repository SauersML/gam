"""Part tokens in the oracle's vocabulary (#2951): every decomposition part is one added token whose input
embedding and output-layer row are computed from the part's read/write vectors by trainable projections
(g-predict's part_tokens module), so naming a part in a program means reading it and the oracle's
gradients reach the projections.

  parts = <a module with tokens() -> list[str], input_rows() -> (P, d), output_rows() -> (P, d)>
  first = install(causal_lm, tokenizer, parts)   # the part tokens' ids are first .. first + P - 1
  materialize(causal_lm, tokenizer, parts, first, base_name, out_dir)   # an HF checkpoint vLLM can load

install replaces the embedding and the output layer with wrappers that put input_rows() / output_rows()
at the part tokens' ids on every forward. Qwen3's embedding has more rows than its tokenizer has tokens,
so the added ids may fall inside the padded rows; the wrappers override those rows (or extend past them).
materialize writes the base weights (LoRA layers' base weights, no adapters) with the part rows filled
in, an untied output layer, the extended tokenizer and the config with the new vocabulary size."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class PartEmbedding(nn.Module):
    def __init__(self, base: nn.Embedding, parts: nn.Module, first: int):
        super().__init__()
        self.base, self.parts, self.first = base, parts, first

    @property
    def weight(self):
        return self.base.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n = self.base.num_embeddings
        out = self.base(ids.clamp(max=n - 1))
        mask = ids >= self.first
        if not bool(mask.any()):
            return out
        rows = self.parts.input_rows().to(out.dtype)
        part = rows[(ids - self.first).clamp(min=0, max=rows.shape[0] - 1)]
        return torch.where(mask[..., None], part, out)


class PartHead(nn.Module):
    def __init__(self, base: nn.Linear, parts: nn.Module, first: int):
        super().__init__()
        self.base, self.parts, self.first = base, parts, first

    @property
    def weight(self):
        return self.base.weight

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.base(h)
        part = h @ self.parts.output_rows().to(h.dtype).T
        end = self.first + part.shape[-1]
        return torch.cat([logits[..., : self.first], part, logits[..., end:]], -1)  # past the padded rows: just appended


def install(causal: nn.Module, tok, parts: nn.Module) -> int:
    """Adds the part tokens to the tokenizer and the wrappers to the model; returns the first part id."""
    first = len(tok)
    added = tok.add_tokens(list(parts.tokens()))
    if added != len(parts.tokens()):
        raise ValueError(f"{len(parts.tokens()) - added} part tokens already exist in the tokenizer")
    causal.model.embed_tokens = PartEmbedding(causal.model.embed_tokens, parts, first)
    causal.lm_head = PartHead(causal.lm_head, parts, first)
    return first


@torch.no_grad()
def materialize(causal: nn.Module, tok, parts: nn.Module, first: int, base_name: str, out_dir: Path) -> Path:
    """An HF checkpoint of the base model (LoRA base weights, no adapters) whose embedding and untied output
    layer carry the part rows at the part tokens' ids."""
    from safetensors.torch import save_file
    from transformers import AutoConfig

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_in, rows_out = parts.input_rows().detach(), parts.output_rows().detach()
    size = max(causal.model.embed_tokens.base.num_embeddings, first + rows_in.shape[0])

    def placed(base: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        w = torch.zeros(size, base.shape[1], dtype=base.dtype, device="cpu")
        w[: base.shape[0]] = base.detach().cpu()
        w[first : first + rows.shape[0]] = rows.to(base.dtype).cpu()
        return w

    state = {}
    for name, value in causal.state_dict().items():
        if "lora_" in name or ".parts." in name:
            continue
        name = name.replace(".base_layer.", ".")
        if name == "model.embed_tokens.base.weight":
            state["model.embed_tokens.weight"] = placed(value, rows_in)
        elif name == "lm_head.base.weight":
            state["lm_head.weight"] = placed(value, rows_out)
        else:
            state[name] = value.detach().cpu().contiguous()
    if "lm_head.weight" not in state:  # tied embeddings: the output layer gets its own rows
        state["lm_head.weight"] = placed(causal.lm_head.base.weight, rows_out)
    save_file(state, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    config = AutoConfig.from_pretrained(base_name)
    config.vocab_size, config.tie_word_embeddings = size, False
    config.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "part_tokens.json").write_text(json.dumps({"first": first, "count": rows_in.shape[0], "tokens": list(parts.tokens())}))
    return out_dir

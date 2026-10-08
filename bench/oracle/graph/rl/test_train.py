"""Checks of the training losses on a tiny random Qwen3 (CPU, seconds): python test_train.py

- token_logprobs = log-softmax of a plain forward pass at every completion token;
- at the start pi = pi_ref: the KL term is 0 and the DPO loss is ln 2;
- the GRPO loss weights an episode by its summed token log-probability: its gradient equals
  -(1/E) sum_e A_e grad log pi(y_e), whatever the episodes' lengths.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402


def tiny(path: Path):
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=256, tie_word_embeddings=True)
    torch.manual_seed(0)
    Qwen3ForCausalLM(config).save_pretrained(path)
    tok.save_pretrained(path)


def main():
    with tempfile.TemporaryDirectory() as d:
        tiny(Path(d))
        args = argparse.Namespace(base=d, init=None, lora_rank=4)
        pol = train.Policy(args, torch.device("cpu"))
        for p in pol.params:  # a nonzero adapter, so pi differs from pi_ref after the first update
            torch.nn.init.normal_(p, std=0.02)
        g = torch.Generator().manual_seed(1)
        prompts = [torch.randint(0, 1000, (n,), generator=g).tolist() for n in (5, 9)]
        comps = [torch.randint(0, 1000, (n,), generator=g).tolist() for n in (3, 11)]

        lp, mask = pol.token_logprobs(prompts, comps)
        for r, (p, c) in enumerate(zip(prompts, comps)):
            ids = torch.tensor([p + c])
            with torch.no_grad():
                full = torch.log_softmax(pol.model(input_ids=ids).logits[0].float(), -1)
            want = full[len(p) - 1 : len(p) + len(c) - 1].gather(-1, torch.tensor(c)[:, None])[:, 0]
            got = lp[r][mask[r] > 0]
            assert torch.allclose(got, want, atol=1e-5), (got, want)

        ref, _ = pol.token_logprobs(prompts, comps, ref=True)  # the adapter disabled
        with pol.model.disable_adapter(), torch.no_grad():
            assert torch.allclose(ref, pol.token_logprobs(prompts, comps)[0], atol=1e-6)
        for p in pol.params:  # zero the adapter's B: pi = pi_ref exactly
            if "lora_B" in [n for n, q in pol.model.named_parameters() if q is p][0]:
                p.data.zero_()
        pol.model.zero_grad()
        stats = train.grpo_update(pol, prompts, comps, [0.0, 0.0], beta=1.0, micro=2)
        assert abs(stats["kl_sum_per_episode"]) < 1e-9, stats
        pol.model.zero_grad()
        stats = train.dpo_update(pol, [prompts[0]], [comps[0]], [comps[1][:3]], beta=0.1, micro=1)
        assert abs(stats["loss"] - math.log(2)) < 1e-6, stats

        for p in pol.params:
            torch.nn.init.normal_(p, std=0.02)
        adv = [1.5, -0.5]
        pol.model.zero_grad()
        train.grpo_update(pol, prompts, comps, adv, beta=0.0, micro=1)
        got = [p.grad.clone() for p in pol.params]
        pol.model.zero_grad()
        lp, mask = pol.token_logprobs(prompts, comps)
        (-(torch.tensor(adv) * (lp * mask).sum(1)).sum() / 2).backward()
        for a, b in zip(got, (p.grad for p in pol.params)):
            assert torch.allclose(a, b, atol=1e-6)
        check_pack(pol)
        check_left_padding(pol)
        check_init_adapter(Path(d))
        check_part_vocab(Path(d))
        check_registry_parts(Path(d))
    check_split_prompts()
    print("ok: token log-probabilities, KL 0 and DPO ln 2 at the reference, GRPO gradient = summed log-probability policy gradient, "
          "packed groups = separate sequences, left-padded batched generation, g-predict's adapters = their PEFT conversion, prompt split, part tokens (stand-in and registry)")


def check_pack(pol):
    """One sequence per group (pack) gives every completion's token log-probabilities, and the GRPO
    gradient, of separate sequences."""
    g = torch.Generator().manual_seed(3)
    prompt = torch.randint(0, 1000, (7,), generator=g).tolist()
    comps = [torch.randint(0, 1000, (n,), generator=g).tolist() for n in (4, 1, 9)]
    prompts = [prompt] * len(comps)
    pol.pack = False
    lp, mask = pol.token_logprobs(prompts, comps)
    want = [lp[r][mask[r] > 0] for r in range(len(comps))]
    adv = [1.0, -2.0, 0.5]
    pol.model.zero_grad()
    train.grpo_update(pol, prompts, comps, adv, beta=0.5, micro=3)
    grads = [p.grad.clone() for p in pol.params]
    pol.pack = True
    lp, mask = pol.token_logprobs(prompts, comps)
    for r in range(len(comps)):
        assert torch.allclose(lp[r][mask[r] > 0], want[r], atol=1e-5), (r, lp[r][mask[r] > 0], want[r])
    pol.model.zero_grad()
    train.grpo_update(pol, prompts, comps, adv, beta=0.5, micro=3)
    for a, p in zip(grads, pol.params):
        assert torch.allclose(a, p.grad, atol=1e-5), float((a - p.grad).abs().max())
    pol.pack = False


def check_left_padding(pol):
    """HfSampler batches prompts of different lengths left-padded: greedy generation of a padded batch
    equals each prompt's own, and the sampler returns n completions per prompt within max_tokens."""
    g = torch.Generator().manual_seed(4)
    prompts = [torch.randint(0, 1000, (n,), generator=g).tolist() for n in (3, 8)]
    pol.train_mode(False)
    width = 8
    ids = torch.full((2, width), pol.end)
    att = torch.zeros(2, width, dtype=torch.long)
    for r, p in enumerate(prompts):
        ids[r, width - len(p) :] = torch.tensor(p)
        att[r, width - len(p) :] = 1
    with torch.no_grad():
        both = pol.model.generate(input_ids=ids, attention_mask=att, max_new_tokens=6, do_sample=False, eos_token_id=-1, pad_token_id=pol.end)[:, width:]
        for r, p in enumerate(prompts):
            one = pol.model.generate(input_ids=torch.tensor([p]), attention_mask=torch.ones(1, len(p), dtype=torch.long), max_new_tokens=6, do_sample=False, eos_token_id=-1,
                                     pad_token_id=pol.end)[0, len(p) :]
            assert both[r].tolist() == one.tolist(), (r, both[r], one)
    out = train.HfSampler(pol, 5, batch=3)(prompts, 2, Path("."), 0)
    assert len(out) == 2 and all(len(x) == 2 and all(len(c) <= 5 for c in x) for x in out)


def check_part_vocab(base: Path):
    """Part tokens (part_vocab.py): base tokens' logits are unchanged, a part token's input embedding and
    logit come from the projections, gradients reach the projections, and the materialized checkpoint
    reproduces the wrapped model's logits."""
    import part_vocab
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class Parts(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            g = torch.Generator().manual_seed(5)
            self.register_buffer("read", torch.randn(3, 6, generator=g))
            self.register_buffer("write", torch.randn(3, 6, generator=g))
            self.p_in, self.p_out = torch.nn.Linear(6, d), torch.nn.Linear(6, d)

        def tokens(self):
            return ["<part:a>", "<part:b>", "<part:c>"]

        def input_rows(self):
            return self.p_in(self.read)

        def output_rows(self):
            return self.p_out(self.write)

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(base)
    ids = torch.randint(0, 1000, (1, 9))
    with torch.no_grad():
        before = model(input_ids=ids).logits
    parts = Parts(model.config.hidden_size)
    first = part_vocab.install(model, tok, parts)
    with torch.no_grad():
        after = model(input_ids=ids).logits
    outside = lambda x: torch.cat([x[..., :first], x[..., first + 3 :]], -1)  # noqa: E731  every column but the part tokens'
    assert torch.allclose(outside(before), outside(after), atol=1e-6)
    assert torch.allclose(model.model.embed_tokens(torch.tensor([first + 1])), parts.input_rows()[1:2], atol=1e-6)
    mixed = torch.cat([ids, torch.tensor([[first, first + 2]])], 1)
    out = model(input_ids=mixed, output_hidden_states=True)
    h = out.hidden_states[-1][0, -1]
    assert torch.allclose(out.logits[0, -1, first : first + 3], parts.output_rows() @ h, atol=1e-4)
    out.logits[0, -1, first + 1].backward()
    assert parts.p_out.weight.grad.abs().sum() > 0 and parts.p_in.weight.grad.abs().sum() > 0
    with tempfile.TemporaryDirectory() as d:
        part_vocab.materialize(model, tok, parts, first, str(base), Path(d))
        plain = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32)
        with torch.no_grad():
            want = model(input_ids=mixed).logits
            got = plain(input_ids=mixed).logits
        assert got.shape == want.shape and torch.allclose(got, want, atol=1e-4), float((got - want).abs().max())
        assert len(AutoTokenizer.from_pretrained(d)) == first + 3


def check_registry_parts(base: Path):
    """g-predict's PartTokens over a small registry through the Policy: one computation of the rows per
    log-probability call gives the same values and gradients as recomputing them; part tokens in an answer
    become mech addresses for the checker; SFT targets get part tokens for addresses."""
    import part_tokens

    addresses = [f"PD.vpd[{l}].v_proj[{i}]" for l in (1, 2) for i in (3, 7)] + ["PD.vpd[0].c_fc[12]"]
    g = torch.Generator().manual_seed(9)
    reg = part_tokens.Registry(addresses, {"vpd.v": torch.randn(4, 10, generator=g), "vpd.fc": torch.randn(1, 12, generator=g)})
    path = Path(tempfile.mkdtemp()) / "reg.safetensors"
    reg.save(path)
    args = argparse.Namespace(base=str(base), init=None, lora_rank=4, part_tokens=str(path))
    pol = train.Policy(args, torch.device("cpu"))
    first = pol.first_part
    prompt, comp = [11, 12, 13], [first, 50, first + 4, first + 2]
    lp, mask = pol.token_logprobs([prompt], [comp])  # rows computed once for the call
    (lp * mask).sum().backward()
    grads = [p.grad.clone() for p in pol.parts.parameters() if p.grad is not None]
    assert grads and all(gr.abs().sum() > 0 for gr in grads[:2])
    pol.model.zero_grad()
    pol.parts.zero_grad()
    import part_vocab

    causal = pol.model.base_model.model
    ids = torch.tensor([prompt + comp])
    with torch.no_grad():
        plain = causal(input_ids=ids).logits  # no cache: the wrappers recompute the rows
        with part_vocab.rows_once(causal):
            once = causal(input_ids=ids).logits
    assert torch.allclose(plain, once, atol=1e-6)
    text = "```python\nx = bind('v', <p:2.v.7>, <p:0.fc.12>)\n```\nThe value head."
    assert train.to_mech(text) == "```python\nx = bind('v', PD.vpd[2].v_proj[7], PD.vpd[0].c_fc[12])\n```\nThe value head."
    assert pol.parts.reg.rewrite("node(PD.vpd[1].v_proj[3])") == "node(<p:1.v.3>)"
    assert pol.tok.decode([first + 4]) == "<p:0.fc.12>"


def check_init_adapter(base: Path):
    """g-predict's sft.py adapters (its own wrap) and their PEFT conversion give the same logits."""
    import json

    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "predict"))
    import sft

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    adapters = sft.wrap(model, 4, 8.0)
    torch.manual_seed(2)
    for a in adapters.values():
        torch.nn.init.normal_(a.B, std=0.05)
    ids = torch.randint(0, 1000, (1, 12))
    with torch.no_grad():
        want = model(input_ids=ids).logits
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "sft"
        src.mkdir()
        save_file({f"{k}.{n}": getattr(a, n).detach().contiguous() for k, a in adapters.items() for n in ("A", "B")}, str(src / "adapters.safetensors"))
        (src / "meta.json").write_text(json.dumps({"args": {"rank": 4, "alpha": 8.0, "model": str(base)}}))
        pol = train.Policy(argparse.Namespace(base=str(base), init=train.init_adapter(str(src), Path(d)), lora_rank=4), torch.device("cpu"))
        pol.model.float()
        with torch.no_grad():
            got = pol.model(input_ids=ids).logits
    assert torch.allclose(got, want, atol=1e-4), float((got - want).abs().max())


def check_split_prompts():
    with tempfile.TemporaryDirectory() as d:
        b = {"id": "x", "model": "vpd4l", "path": "/nowhere", "prompts": [{"text": str(i)} for i in range(10)]}
        train_views, held = train.split_prompts([b], 4, Path(d))
        assert [p["text"] for p in train_views[0]["prompts"]] == ["1", "2", "3", "5", "6", "7", "9"]
        assert [p["text"] for p in held[0]["prompts"]] == ["0", "4", "8"]
        assert Path(held[0]["path"]).exists() and "path" not in __import__("json").loads(Path(held[0]["path"]).read_text())


if __name__ == "__main__":
    main()

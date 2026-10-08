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
        check_ppo(pol)
        check_credit_advantages(pol)
        check_rl2_step(pol)
    check_split_prompts()
    check_rl2_pieces()
    print("ok: token log-probabilities, KL 0 and DPO ln 2 at the reference, GRPO gradient = summed log-probability policy gradient, "
          "packed groups = separate sequences, left-padded batched generation, g-predict's adapters = their PEFT conversion, prompt split, part tokens (stand-in and registry), "
          "RL v2: RLOO at a fixed scale, step seeds, PPO epoch 0 = the GRPO gradient, clipping, token credit (canonical and other tokenizations), a whole step on stand-ins")


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
    reach the checker as written; SFT targets get part tokens for addresses."""
    import part_tokens

    addresses = [f"PD[{l}].v_proj[{i}]" for l in (1, 2) for i in (3, 7)] + ["PD[0].c_fc[12]"]
    g = torch.Generator().manual_seed(9)
    reg = part_tokens.Registry(addresses, {"pd.v": torch.randn(4, 10, generator=g), "pd.fc": torch.randn(1, 12, generator=g)})
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
    text = "```python\nfrom mech import align\ndef x(tokens):\n    return tokens\nalign(x, <p:2.v.7>, <p:0.fc.12>)\n```\nThe value head."
    it = train.item(text, {}, 0, 0, 16)  # part tokens reach the checker as written (one Python token each)
    assert "align(x, <p:2.v.7>, <p:0.fc.12>)" in it["source"] and it["explanation"] == "The value head."
    assert pol.parts.reg.rewrite("node(PD[1].v_proj[3], PD.vpd[2].v_proj[7])") == "node(<p:1.v.3>, <p:2.v.7>)"  # either spelling
    assert pol.tok.decode([first + 4]) == "<p:0.fc.12>"
    groups = pol.param_groups(1e-4)  # the LoRA at lr, each projection at lr * rank / its feature width
    assert sorted(id(p) for g in groups for p in g["params"]) == sorted(id(p) for p in pol.params)
    fans = sorted({m.in_features for m in pol.parts.modules() if isinstance(m, torch.nn.Linear)})  # feature widths 10, 12 and the maps' inner rank
    assert groups[0]["lr"] == 1e-4 and sorted({round(g["lr"], 12) for g in groups[1:]}) == sorted({round(1e-4 * 4 / f, 12) for f in fans}) and {10, 12} <= set(fans)
    with torch.no_grad():  # trained projections come back from a saved adapter (--init: the next round, or eval)
        for p in pol.parts.parameters():
            p.add_(0.01 * torch.randn(p.shape, generator=g))
    with tempfile.TemporaryDirectory() as d:
        pol.save(Path(d))
        again = train.Policy(argparse.Namespace(base=str(base), init=d, lora_rank=4, part_tokens=str(path)), torch.device("cpu"))
    (f0, i0, o0), (f1, i1, o1) = pol.part_rows(), again.part_rows()
    assert f0 == f1 and torch.equal(i0, i1) and torch.equal(o0, o1)


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


class Recorder:
    """A stand-in optimizer and warmup: records the (clipped) gradients at each step, moves nothing unless lr > 0 (SGD)."""

    def __init__(self, params, lr=0.0):
        self.params, self.lr, self.grads = params, lr, []

    def step(self):
        if hasattr(self, "params"):
            self.grads.append([p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in self.params])
            with torch.no_grad():
                for p in self.params:
                    if p.grad is not None:
                        p -= self.lr * p.grad

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None


class NoWarmup:
    def step(self):
        pass


def check_ppo(pol):
    """ppo_update's first epoch is grpo_update's gradient (constant token advantages, the KL through the advantage), a
    second epoch on an unmoved policy repeats it, and after a large step some ratios are clipped."""
    g = torch.Generator().manual_seed(6)
    prompts = [torch.randint(0, 1000, (n,), generator=g).tolist() for n in (4, 6)]
    comps = [torch.randint(0, 1000, (n,), generator=g).tolist() for n in (5, 3)]
    for p in pol.params:
        torch.nn.init.normal_(p, std=0.02)
    adv = [0.7, -1.3]
    pol.model.zero_grad()
    train.grpo_update(pol, prompts, comps, adv, beta=0.3, micro=1)
    torch.nn.utils.clip_grad_norm_(pol.params, 1.0)
    want = [p.grad.clone() for p in pol.params]
    pol.model.zero_grad()
    rec = Recorder(pol.params)
    stats = train.ppo_update(pol, prompts, comps, [[a] * len(c) for a, c in zip(adv, comps)], 0.3, 1, 2, 0.2, rec, NoWarmup())
    for epoch in range(2):
        for a, b in zip(want, rec.grads[epoch]):
            assert torch.allclose(a, b, atol=1e-6), (epoch, float((a - b).abs().max()))
    assert stats["clip_fraction"] == [0.0, 0.0], stats
    rec = Recorder(pol.params, lr=50.0)
    stats = train.ppo_update(pol, prompts, comps, [[a] * len(c) for a, c in zip(adv, comps)], 0.0, 1, 2, 0.2, rec, NoWarmup())
    assert stats["clip_fraction"][0] == 0.0 and stats["clip_fraction"][1] > 0.0, stats


ANSWER = """I look at the previous token.
```python
def answer(tokens):
    # the copy
    return tokens


align(answer, <p:2.v.559>, <p:2.o.735>)
claim(answer, <p:1.q.3>)
```
The answer is copied by layer 2's value and output parts."""


def check_credit_advantages(pol):
    """credit_advantages: a part's tokens get dS(drop it) / scale, the rest of its statement dS(drop the statement) / scale
    (+inf: 1), the other tokens the episode advantage; the same with a tokenization that re-encoding does not give."""
    from edits import Edit

    tok = pol.tok
    source = train.split_answer(ANSWER)[0]
    assert ANSWER[train.program_offset(ANSWER, source):].startswith(source)
    dS = {Edit("drop", "answer", "align", "<p:2.v.559>"): 3.0, Edit("unalign", "answer", "align"): float("inf"), Edit("unalign", "answer", "claim"): -1.0}
    canonical = tok.encode(ANSWER, add_special_tokens=False) + [pol.end]
    chars = [i for ch in ANSWER for i in tok.encode(ch, add_special_tokens=False)] + [pol.end]
    assert chars != canonical[: len(chars)]
    for completion in (canonical, chars):
        adv = train.credit_advantages(tok, completion, ANSWER, source, -0.25, dS, 2.0)
        spans = train.token_spans(tok, completion, ANSWER)
        assert len(adv) == len(completion) == len(spans)
        part = ANSWER.index("<p:2.v.559>")
        line = ANSWER.index("align(answer")
        claim = ANSWER.index("claim(answer")
        for (c0, c1), a in zip(spans, adv):
            if c1 <= c0:
                assert a == -0.25
            elif c1 > part and c0 < part + len("<p:2.v.559>"):  # a token overlapping the part (" <")
                assert a == 1.5, (ANSWER[c0:c1], a)
            elif c0 >= line and c1 <= ANSWER.index("\n", line):
                assert a == 1.0, (ANSWER[c0:c1], a)
            elif c0 >= claim and c1 <= ANSWER.index("\n", claim):
                assert a == -0.5, (ANSWER[c0:c1], a)
            elif c1 <= line - 1 or c0 >= ANSWER.index("```\nThe"):
                assert a == -0.25, (ANSWER[c0:c1], a)


def check_rl2_pieces():
    import numpy as np

    A = train.rloo(np.array([1.0, 2.0, 3.0]), 2.0)
    assert np.allclose(A, [0.75, 0.0, -0.75]), A
    assert train.rloo(np.array([5.0]), 1.0).tolist() == [0.0]
    args = argparse.Namespace(seed=0, eval_seed=5)
    seeds = [train.step_seed(args, s) for s in range(10)]
    assert len(set(seeds)) == 10 and 5 not in seeds
    assert train.step_seed(argparse.Namespace(seed=1, eval_seed=5), 0) == 1 << 20
    sc = train.Scales({"a": "x"})
    sc.take([("a", "teacher", {}), ("a", "empty", {}), ("b", "empty", {})], [{"valid": True, "total_bits": 10.0}, {"valid": True, "total_bits": 40.0}, {"valid": True, "total_bits": 20.0}])
    assert sc.scale == {"a": 10.0, "b": 20.0} and sc.target == {"a": 10.0, "b": 0.0}
    sc.observe("a", np.array([12.0, 30.0, 99.0]), np.array([True, True, False]))
    sc.observe("b", np.array([1.0]), np.array([True]))
    assert abs(sc.gap["a"] - 1.1) < 1e-12 and sc.gap["b"] == sc.FLOOR
    import random

    picks = [sc.draw([{"id": "a"}, {"id": "b"}], 1, random.Random(i))[0]["id"] for i in range(400)]
    assert picks.count("a") > 300, picks.count("a")


def check_rl2_step(pol):
    """A whole rl2 step on stand-ins: a sampler that writes fixed answers and the score of test_edits (10 bits per needed
    part missing, 1 per part named; invalid without an answer statement). Behavior "x" gets answers of different
    scores, "y" only invalid ones (dropped, refilled by "z"); credit marks tokens, refine improves the best answers
    (expert iteration), and the PPO epochs and the expert-iteration step run."""
    import json

    import numpy as np
    from edits import Answer

    needed = {"<p:2.v.559>", "<p:2.o.735>", "<p:2.v.9>"}

    def stand_in(items):
        out = []
        for it in items:
            a = Answer.parse(it["source"])
            named = {q for st in a.statements for q in st.parts}
            valid = any(st.variable == "answer" and st.kind == "align" for st in a.statements)
            out.append({"total_bits": 10.0 * len(needed - named) + len(named) if valid else 1e4, "valid": valid})
        return out

    def answer_with(parts):
        return ANSWER.replace("<p:2.v.559>, <p:2.o.735>", ", ".join(parts)) if parts else ANSWER.replace("align(answer, <p:2.v.559>, <p:2.o.735>)\n", "")

    texts = {"x": [answer_with(["<p:2.v.559>", "<p:2.o.735>"]), answer_with(["<p:2.v.559>"]), answer_with(["<p:2.v.559>", "<p:3.o.1>"]), answer_with([])],
             "y": [answer_with([])] * 4, "z": [answer_with(["<p:2.v.9>"]), answer_with(["<p:2.v.9>", "<p:2.o.735>"]), answer_with(["<p:2.v.9>"]), answer_with([])]}
    asked = []

    def sampler(prompts, n, adapter, version):
        out = []
        for p in prompts:
            bid = pol.tok.decode(p).split("behavior ")[1][0]
            asked.append(bid)
            out.append([pol.tok.encode(t, add_special_tokens=False) + [pol.end] for t in texts[bid][:n]])
        return out

    render = train.render
    train.render = lambda b: "behavior " + b["id"]
    try:
        with tempfile.TemporaryDirectory() as d:
            logs = {k: open(Path(d) / f"{k}.jsonl", "w") for k in ("train", "samples", "improved")}
            args = argparse.Namespace(seed=0, eval_seed=1_000_003, samples=4, experiments=4, credit=16, refill=1, refine=3, refine_adds=2, behaviors_per_step=2, beta=0.0,
                                      pack=False, micro=2, ppo_epochs=2, clip=0.2, exit_beta=0.1)
            pool = [{"id": "x", "model": "vpd4l"}, {"id": "y", "model": "vpd4l"}, {"id": "z", "model": "vpd4l"}]
            scales = train.Scales({"x": answer_with(sorted(needed)), "y": answer_with(sorted(needed))})
            rec = Recorder(pol.params)
            for p in pol.params:
                torch.nn.init.normal_(p, std=0.02)
            pol.model.zero_grad()
            first = {}
            for step in range(64):  # the first step that draws x and y
                if set(random_pick(pool, args, step)) == {"x", "y"}:
                    first = train.rl2_step(step, args, pol, sampler, stand_in, scales, pool, Path(d), rec, NoWarmup(), {"x": {"answer": ["<p:2.v.9>", "<p:2.o.735>"]}}, logs, 0.0)
                    break
            assert first, "no step draws x and y"
            assert first["groups"] == 3 and first["kept"] == 2 and first["refills"] == 1 and asked[-1] == "z", (first, asked)
            assert scales.scale == {"x": 4.0, "y": 4.0, "z": 1e4} and scales.target["z"] == 0.0  # the teacher's 4 parts; z: the empty program's total
            assert first["improved"] >= 1 and len(first["loss"]) == 2 and "exit_sft_loss" in first, first
            assert len(rec.grads) == 3  # two PPO epochs and the expert-iteration step
            for f in logs.values():
                f.close()
            rows = [json.loads(line) for line in open(Path(d) / "samples.jsonl")]
            credited = [r for r in rows if r["credit"]]
            assert len(rows) == 12 and credited and all(r["kept"] == (r["behavior"] != "y") for r in rows)
            improved = [json.loads(line) for line in open(Path(d) / "improved.jsonl")]
            assert all(r["bits"] < r["sampled_bits"] for r in improved), improved
            assert any(set(Answer.parse(train.split_answer(r["text"])[0]).statements[0].parts) == needed for r in improved if r["behavior"] == "x"), improved
            assert np.isfinite(first["mean_bits"])
    finally:
        train.render = render


def random_pick(pool, args, step):
    import random

    return [b["id"] for b in random.Random(train.step_seed(args, step)).sample(pool, min(args.behaviors_per_step, len(pool)))]


if __name__ == "__main__":
    main()

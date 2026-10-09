"""Checks of the training losses on a tiny random Qwen3 (CPU, seconds): python test_train.py

- token_logprobs = log-softmax of a plain forward pass at every completion token;
- at the start pi = pi_ref: the KL term is 0 and the DPO loss is ln 2;
- PPO's first epoch is the policy gradient of summed token log-probabilities (episode_gradient), whatever the
  episodes' lengths;
- RL v2 on stand-ins: ranked advantages, references, credit, refinement, a whole step.
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


def episode_gradient(pol, prompts, completions, advantage, beta: float, micro: int) -> dict:
    """The policy gradient of whole episodes, accumulated in the parameters' .grad: loss = -(1/E) sum_e (A_e - beta
    stop_grad(rho_e)) log pi(y_e), rho_e = log pi(y_e) - log pi_ref(y_e), log pi(y_e) the summed token log-probability
    (the KL(pi || pi_ref) gradient is E[rho grad log pi])."""
    kl = 0.0
    for idx in train.micro_batches(len(prompts), micro):
        ps, cs = [prompts[i] for i in idx], [completions[i] for i in idx]
        adv = torch.tensor([advantage[i] for i in idx], dtype=torch.float32)
        cur, mask = pol.token_logprobs(ps, cs)
        episode = (cur * mask).sum(1)
        loss = -(adv * episode).sum() / len(prompts)
        if beta > 0:
            ref, _ = pol.token_logprobs(ps, cs, ref=True)
            ratio = ((cur - ref) * mask).sum(1).detach()
            loss = loss + beta * (ratio * episode).sum() / len(prompts)
            kl += float(ratio.sum()) / len(prompts)
        loss.backward()
    return {"kl_sum_per_episode": kl}


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
        stats = episode_gradient(pol, prompts, comps, [0.0, 0.0], beta=1.0, micro=2)
        assert abs(stats["kl_sum_per_episode"]) < 1e-9, stats
        pol.model.zero_grad()
        stats = train.dpo_update(pol, [prompts[0]], [comps[0]], [comps[1][:3]], beta=0.1, micro=1)
        assert abs(stats["loss"] - math.log(2)) < 1e-6, stats

        for p in pol.params:
            torch.nn.init.normal_(p, std=0.02)
        adv = [1.5, -0.5]
        pol.model.zero_grad()
        episode_gradient(pol, prompts, comps, adv, beta=0.0, micro=1)
        got = [p.grad.clone() for p in pol.params]
        pol.model.zero_grad()
        lp, mask = pol.token_logprobs(prompts, comps)
        (-(torch.tensor(adv) * (lp * mask).sum(1)).sum() / 2).backward()
        for a, b in zip(got, (p.grad for p in pol.params)):
            assert torch.allclose(a, b, atol=1e-6)
        check_left_padding(pol)
        check_part_vocab(Path(d))
        check_registry_parts(Path(d))
        check_ppo(pol)
        check_credit_advantages(pol)
        check_rl2_step(pol)
    check_rl2_pieces()
    print("ok: token log-probabilities, KL 0 and DPO ln 2 at the reference, summed log-probability policy gradient, "
          "left-padded batched generation, part tokens (stand-in and registry), "
          "RL v2: ranked leave-one-out advantages, references, step seeds, PPO epoch 0 = the policy gradient, clipping, token credit (canonical and other tokenizations), a whole step on stand-ins")


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
    text = '```python\nnodes = {"copy": {"subcomponents": ["<p:2.v.7>", "<p:0.fc.12>"]}}\nedges = [("input", "copy"), ("copy", "output")]\n```\nThe value head.'
    it = train.item(text, {}, 0)  # part tokens reach the scorer as written
    assert '["<p:2.v.7>", "<p:0.fc.12>"]' in it["source"] and it["explanation"] == "The value head."
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
    """ppo_update's first epoch is episode_gradient's (constant token advantages, the KL through the advantage), a
    second epoch on an unmoved policy repeats it, behavior log-probabilities equal to the trainer's change nothing,
    ones 4x less likely give the importance weight's cap (2x the gradient), and after a large step ratios are clipped."""
    g = torch.Generator().manual_seed(6)
    prompts = [torch.randint(0, 1000, (n,), generator=g).tolist() for n in (4, 6)]
    comps = [torch.randint(0, 1000, (n,), generator=g).tolist() for n in (5, 3)]
    clip = {"low": 0.2, "high": 0.28, "dual": 3.0, "tis_cap": 2.0}
    for p in pol.params:
        torch.nn.init.normal_(p, std=0.02)
    adv = [0.7, -1.3]
    tokens = [[a] * len(c) for a, c in zip(adv, comps)]
    pol.model.zero_grad()
    episode_gradient(pol, prompts, comps, adv, beta=0.3, micro=1)
    want = [p.grad.clone() for p in pol.params]
    pol.model.zero_grad()
    rec = Recorder(pol.params)
    stats = train.ppo_update(pol, prompts, comps, tokens, 0.3, 1, 2, clip, rec, NoWarmup())
    unclipped = [p.grad for p in pol.params]  # Recorder saw the clipped ones; compare directions through the norm
    for epoch in range(2):
        scale = [b.norm() for b in rec.grads[epoch]]
        got = torch.cat([b.flatten() for b in rec.grads[epoch]])
        ref = torch.cat([a.flatten() for a in want])
        assert torch.allclose(got / got.norm(), ref / ref.norm(), atol=1e-5), (epoch, float((got / got.norm() - ref / ref.norm()).abs().max()))
    del unclipped, scale
    assert stats["clip_fraction"] == [0.0, 0.0] and stats["tis_weight_mean"] is None, stats
    lp, mask = pol.token_logprobs(prompts, comps)
    same = [lp[r][mask[r] > 0].tolist() for r in range(2)]
    pol.model.zero_grad()
    rec_same = Recorder(pol.params)
    stats = train.ppo_update(pol, prompts, comps, tokens, 0.0, 1, 1, clip, rec_same, NoWarmup(), behavior=same)
    assert abs(stats["tis_weight_mean"] - 1.0) < 1e-5 and stats["behavior_gap_per_token"] < 1e-5, stats
    pol.model.zero_grad()
    rec_one = Recorder(pol.params)
    train.ppo_update(pol, prompts, comps, tokens, 0.0, 1, 1, {**clip, "tis_cap": 1e9}, rec_one, NoWarmup())
    pol.model.zero_grad()
    rec_low = Recorder(pol.params)
    stats = train.ppo_update(pol, prompts, comps, tokens, 0.0, 1, 1, {**clip, "tis_cap": 2.0}, rec_low, NoWarmup(), behavior=[[x - math.log(4) for x in r] for r in same])
    assert abs(stats["tis_weight_mean"] - 2.0) < 1e-4, stats
    a = torch.cat([x.flatten() for x in rec_one.grads[0]])
    b = torch.cat([x.flatten() for x in rec_low.grads[0]])
    assert torch.allclose(a / a.norm(), b / b.norm(), atol=1e-5)  # a uniform weight of 2 scales the gradient (the norm clip hides the factor)
    rec = Recorder(pol.params, lr=50.0)
    stats = train.ppo_update(pol, prompts, comps, tokens, 0.0, 1, 2, clip, rec, NoWarmup())
    assert stats["clip_fraction"][0] == 0.0 and stats["clip_fraction"][1] > 0.0, stats


ANSWER = """I look at the previous token.
```python
def graph(tokens, targets):
    return {(targets[0], "<p:2.o.735>"): {targets[0] - 1: ["<p:2.v.559>", "<p:2.v.9>"]}, "out": ["<p:2.o.735>"]}
```
Layer 2's attention output at the target reads two values at the previous token."""


def check_credit_advantages(pol):
    """credit_advantages: a credited subcomponent name's tokens get the sign of its drop (+1: the drop made the answer
    worse), the other tokens the episode advantage; the same with a tokenization that re-encoding does not give."""
    tok = pol.tok
    source = train.split_answer(ANSWER)[0]
    assert ANSWER[train.program_offset(ANSWER, source):].startswith(source)
    v, o = source.index('"<p:2.v.559>"'), source.index('"<p:2.v.9>"')
    signs = {(v, v + len('"<p:2.v.559>"')): 1, (o, o + len('"<p:2.v.9>"')): -1}
    canonical = tok.encode(ANSWER, add_special_tokens=False) + [pol.end]
    chars = [i for ch in ANSWER for i in tok.encode(ch, add_special_tokens=False)] + [pol.end]
    assert chars != canonical[: len(chars)]
    offset = train.program_offset(ANSWER, source)
    for completion in (canonical, chars):
        adv = train.credit_advantages(tok, completion, ANSWER, source, -0.25, signs)
        spans = train.token_spans(tok, completion, ANSWER)
        assert len(adv) == len(completion) == len(spans)
        for (c0, c1), a in zip(spans, adv):
            if c1 <= c0:
                assert a == -0.25
            elif c1 > offset + v and c0 < offset + v + len('"<p:2.v.559>"'):
                assert a == 1, (ANSWER[c0:c1], a)
            elif c1 > offset + o and c0 < offset + o + len('"<p:2.v.9>"'):
                assert a == -1, (ANSWER[c0:c1], a)
            else:
                assert a == -0.25, (ANSWER[c0:c1], a)


def check_rl2_pieces():
    import numpy as np

    A = train.rloo([(0, 0.0, 3), (0, 0.0, 5), (0, 0.2, 1), (1, math.inf, math.inf)])
    assert np.allclose(A, [1.0, 1 / 3, -1 / 3, -1.0]), A
    assert train.rloo([(0, 0.0, 5)]).tolist() == [0.0]
    assert train.rloo([(0, 0.1, 2)] * 7).tolist() == [0.0] * 7
    args = argparse.Namespace(seed=0, eval_seed=5)
    seeds = [train.step_seed(args, s) for s in range(10)]
    assert len(set(seeds)) == 10 and 5 not in seeds
    assert train.step_seed(argparse.Namespace(seed=1, eval_seed=5), 0) == 1 << 20
    assert train.key({"eps": 0.5}, {"valid": True, "kl_bits": 0.4, "size": 7}) == (0, 0.0, 7)
    assert train.key({"eps": 0.5}, {"valid": True, "kl_bits": 0.75, "size": 2}) == (0, 0.25, 2)
    assert train.key({"eps": 0.5}, {"valid": False}) == (1, math.inf, math.inf)
    import json

    with tempfile.TemporaryDirectory() as d:  # teacher_v3.py's manifest, written on another machine; the held-out refusal
        (Path(d) / "x.answer.txt").write_text("answer x")
        lines = [{"behavior": "x", "answer": "/elsewhere/old.answer.txt"}, {"behavior": "x", "answer": "/elsewhere/x.answer.txt"}, {"behavior": "y", "answer": "/elsewhere/y.answer.txt"}]
        (Path(d) / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
        assert train.teacher_answers(d) == {"x": "answer x"}
        held = Path(d) / "teacher_heldout"
        held.mkdir()
        train.refuse_heldout([d, None])
        try:
            train.refuse_heldout([str(held)])
            raise AssertionError("held-out answers accepted for training")
        except SystemExit:
            pass


def check_rl2_step(pol):
    """A whole rl2 step on stand-ins: a sampler that writes fixed answers and a stand-in score (KL 10 bits per needed
    parent missing, size the parents; invalid without parents). Question "x" gets answers of different ranks, "y" only
    invalid ones (dropped, refilled by "z"); credit marks tokens, refine drops the parents the best answers do not
    need (expert iteration), and the PPO epochs and the expert-iteration step run."""
    import json
    import re

    import numpy as np

    needed = {"<p:2.v.559>", "<p:2.v.9>", "<p:2.v.11>"}

    def named_in(source):  # the parents (the reader "<p:2.o.735>" excluded)
        return set(re.findall(r"<p:[^>]+>", source)) - {"<p:2.o.735>"}

    def stand_in(items):
        out = []
        for it in items:
            named = named_in(it["source"])
            out.append({"kl_bits": 10.0 * len(needed - named), "size": len(named), "valid": bool(named)})
        return out

    def answer_with(parts):
        return ANSWER.replace('"<p:2.v.559>", "<p:2.v.9>"', ", ".join(f'"{p}"' for p in parts))

    texts = {"x": [answer_with(["<p:2.v.559>", "<p:2.v.11>", "<p:2.v.9>", "<p:2.v.1>"]), answer_with(["<p:2.v.559>"]), answer_with(["<p:2.v.559>", "<p:2.v.1>"]), answer_with([])],
             "y": [answer_with([])] * 4, "z": [answer_with(["<p:2.v.9>"]), answer_with(["<p:2.v.9>", "<p:2.v.11>"]), answer_with(["<p:2.v.9>"]), answer_with([])]}
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
            args = argparse.Namespace(seed=0, eval_seed=1_000_003, samples=4, credit=16, credit_answers=0, refill=1, refine=3, behaviors_per_step=2, beta=0.0,
                                      micro=2, ppo_epochs=2, clip=0.2, clip_high=0.28, dual_clip=3.0, tis_cap=2.0, exit_beta=0.1)
            pool = [{"id": "x", "model": "vpd4l", "eps": 0.0}, {"id": "y", "model": "vpd4l", "eps": 0.0}, {"id": "z", "model": "vpd4l", "eps": 0.0}]
            rec = Recorder(pol.params)
            for p in pol.params:
                torch.nn.init.normal_(p, std=0.02)
            pol.model.zero_grad()
            first = {}
            for step in range(64):  # the first step that draws x and y
                if set(random_pick(pool, args, step)) == {"x", "y"}:
                    first = train.rl2_step(step, args, pol, sampler, stand_in, pool, Path(d), train.Learner(pol, rec, NoWarmup()), logs, 0.0)
                    break
            assert first, "no step draws x and y"
            assert first["groups"] == 3 and first["kept"] == 2 and first["refills"] == 1 and asked[-1] == "z", (first, asked)
            assert first["improved"] >= 1 and len(first["loss"]) == 2 and "exit_sft_loss" in first, first
            assert first["repeated_scores"] > 0, first  # refinement starts from an answer the credit scored
            texts3 = [answer_with(["<p:2.v.559>", "<p:2.v.11>"]), answer_with(["<p:2.v.559>"]), answer_with(["<p:2.v.559>", "<p:2.v.1>"]), answer_with(["<p:2.o.735>"])]
            items = [train.item(t, pool[0], 0) for t in texts3]
            sc3 = stand_in(items)
            keys = [train.key(pool[0], x) for x in sc3]
            grp = {"behavior": pool[0], "completions": [pol.tok.encode(t, add_special_tokens=False) for t in texts3], "texts": texts3, "items": items, "scores": sc3,
                   "keys": keys, "valid": np.array([x["valid"] for x in sc3]), "advantage": np.zeros(4), "token_advantages": [[0.0]] * 4, "credit": [None] * 4}
            train.credit_groups([grp], 0, argparse.Namespace(**{**vars(args), "credit_answers": 2}), pol.tok, stand_in, {"credit": 0.0})
            best = min(range(4), key=keys.__getitem__)
            assert sum(c is not None for c in grp["credit"]) == 2 and grp["credit"][best] is not None, grp["credit"]  # the best and one other
            assert len(rec.grads) == 3  # two PPO epochs and the expert-iteration step
            for f in logs.values():
                f.close()
            rows = [json.loads(line) for line in open(Path(d) / "samples.jsonl")]
            credited = [r for r in rows if r["credit"]]
            assert len(rows) == 12 and credited and all(r["kept"] == (r["behavior"] != "y") for r in rows)
            improved = [json.loads(line) for line in open(Path(d) / "improved.jsonl")]
            assert all(r["key"] < r["sampled_key"] for r in improved), improved
            assert any(named_in(train.split_answer(r["text"])[0]) == needed for r in improved if r["behavior"] == "x"), improved
            assert np.isfinite(first["mean_kl"]) and first["best_correct"] > 0
            logs2 = {k: open(Path(d) / f"async_{k}.jsonl", "w") for k in ("train", "samples", "improved")}
            asked.clear()
            train.rl2_async(argparse.Namespace(**{**vars(args), "steps": 3, "async_rollouts": True}), pol, sampler, stand_in, pool, Path(d), train.Learner(pol, rec, NoWarmup()),
                            logs2, 0.0, lambda: False)
            for f in logs2.values():
                f.close()
            rows2 = [json.loads(line) for line in open(Path(d) / "async_train.jsonl")]
            assert len(rows2) == 3 and all(r["async"] for r in rows2), rows2
            draws = [random_pick(pool, args, s) for s in range(3)]
            for s, r in enumerate(rows2):  # step s draws as many more behaviors as step s - 2 lost (the last scored when s's sampling starts)
                lost = rows2[s - 2]["groups"] - rows2[s - 2]["kept"] if s >= 2 else 0
                assert r["groups"] == len(draws[s]) + min(lost, len(pool) - len(draws[s])), (s, r)
    finally:
        train.render = render


def random_pick(pool, args, step):
    import random

    return [b["id"] for b in random.Random(train.step_seed(args, step)).sample(pool, min(args.behaviors_per_step, len(pool)))]


if __name__ == "__main__":
    main()

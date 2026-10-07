"""Tests of the reader term (reader_score.py): candidate events, the KL in bits, the code without English,
experiment words, prompt prefix sharing, the scorer's sums, memo and baselines (with a stub reader), and the
prefix-cached reader against a full forward pass (with Qwen3-0.6B when it is in the local cache).

  ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_reader_score.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mech  # noqa: E402
import reader_score as S  # noqa: E402

HEAD = "from mech import node, edges, L, embed, logits\n"
PROGRAM = HEAD + '''"""Module docstring: induction."""
prev = node(L[1].head[1])  # previous-token head
def f():
    """Function docstring."""
    return 1
"""A free string statement."""
edges(embed >> prev.query, prev >> logits)
'''


def qwen_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen/Qwen3-0.6B is not in the local Hugging Face cache")


def test_disjoint_tokens_and_strings():
    # [1] contains [1, 2], which contains [1, 2, 3]: each event keeps only its own part.
    q = S.disjoint([[1], [1, 2], [1, 2, 3], [4]], [0.5, 0.2, 0.05, 0.1])
    assert np.allclose(q, [0.3, 0.15, 0.05, 0.1])
    # " the" contains " there"; duplicate keys share their event equally.
    q = S.disjoint([" the", " there", " the", "x"], [0.5, 0.2, 0.5, 0.1])
    assert np.allclose(q, [0.15, 0.2, 0.15, 0.1])
    assert q.sum() <= 0.5 + 0.1 + 1e-12


def test_kl_bits():
    p = np.array([0.5, 0.3])
    assert abs(S.kl_bits(p, 0.2, p.copy())) < 1e-12
    q = np.array([0.25, 0.25])
    want = (0.5 * math.log(0.5 / 0.25) + 0.3 * math.log(0.3 / 0.25) + 0.2 * math.log(0.2 / 0.5)) / math.log(2)
    assert abs(S.kl_bits(p, 0.2, q) - want) < 1e-12
    # q(other) = 0 is floored at K 2^-24, so a measured rest gives a finite, large cost.
    assert 0 < S.kl_bits(np.array([0.5, 0.4]), 0.1, np.array([0.5, 0.5])) < 0.1 * 24 + 1


def test_strip_english_complements_mech_english():
    code = S.strip_english(PROGRAM)
    for text in ("Module docstring", "previous-token head", "Function docstring", "free string"):
        assert text not in code and text in mech.english(PROGRAM)
    assert "prev = node(L[1].head[1])" in code and "edges(embed >> prev.query, prev >> logits)" in code
    compile(code, "<stripped>", "exec")
    assert mech.code_length(code)[0] == mech.code_length(PROGRAM)[0] + 1  # the emptied body's `...`
    # An unparseable program loses its comments only.
    assert "# c" not in S.strip_english("x = (1 +  # c\n")


def test_words_every_kind():
    head = [{"view": "native", "layer": 2, "kind": "head", "index": 4}]
    vpd = [{"view": "vpd", "layer": 1, "kind": "c_fc", "index": [3, 7]}]
    cases = [
        ({"kind": "clean"}, "none"),
        ({"kind": "prompt_edit", "clean_text": "A B"}, "<<<A B>>>"),
        ({"kind": "scale", "pieces": head, "factor": 0}, "remove L[2].head[4]"),
        ({"kind": "scale", "pieces": vpd, "factor": 2.0}, "PD.vpd[1].c_fc[3, 7] by 2"),
        ({"kind": "low_rank", "rank": 1, "matrix": "head 3 q", "layer": 2, "relative_norm": 0.5}, "rank-1"),
        ({"kind": "swap", "pieces": head, "source_text": "X Y"}, "<<<X Y>>>"),
        ({"kind": "swap", "pieces": None, "source_text": "X"}, "a node of the program"),
        ({"kind": "cut", "from": "embed", "to": head, "route": "key"}, "L[2].head[4] (key input)"),
        ({"kind": "cut", "from": head, "to": "logits", "route": None}, "to logits"),
        ({"words": "given words"}, "given words"),
    ]
    for e, part in cases:
        assert part in S.words(e), (e, S.words(e))


def item(token_ids=(785, 6722, 315), cands=((12095, " Rome"), (37723, " Milan")), p=(0.6, 0.1), clean=(0.7, 0.1), family="remove"):
    return {"id": "x", "family": family, "experiment": {"kind": "clean"}, "text": "The capital of", "token_ids": list(token_ids),
            "candidates": [{"token_id": t, "text": s, "clean": c, "p": q} for (t, s), c, q in zip(cands, clean, p)],
            "clean_other": 1 - sum(clean), "other": 1 - sum(p), "q_program": [0.3]}


def test_prompt_prefix_is_shared():
    tok = qwen_tokenizer()
    pr = S.Prompter(tok, shared_vocab=True)
    it = item()
    full = tok.encode(pr.head + S.INSTRUCTIONS.replace("{source}", PROGRAM), add_special_tokens=False)
    assert pr.prefix(PROGRAM) == full
    body = pr.item(it)
    assert body[-len(it["token_ids"]):] == it["token_ids"]  # the reply starts with M's own token ids
    assert pr.candidates(it) == [[12095], [37723]]
    assert "Rome" in tok.decode(body) and "remove" not in tok.decode(pr.prefix(""))


class StubReader:
    """A reader whose next-token distribution depends only on whether the program text mentions 'Rome'."""

    name = "stub"
    model_id = "stub-qwen3"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = 0
        self.rome = tokenizer.encode("Rome", add_special_tokens=False)

    def describe(self):
        return {"backend": self.name}

    def read(self, prefix, suffixes, reads):
        self.calls += 1
        hot = any(prefix[i : i + len(self.rome)] == self.rome for i in range(len(prefix)))
        V = len(self.tokenizer)
        lp = np.full(V, math.log(0.3 / (V - 2)))
        lp[12095], lp[37723] = math.log(0.6 if hot else 0.1), math.log(0.1 if hot else 0.6)
        return [[lp if ids is None else lp[ids] for _, ids in r] for r in reads]


def test_scorer_sums_memo_and_baselines():
    tok = qwen_tokenizer()
    stub = StubReader(tok)
    sc = S.Scorer(stub, "qwen3-0.6b")
    assert sc.prompter.shared
    items = [item(family="clean"), item(p=(0.2, 0.5), family="remove")]
    good = HEAD + '"""The model predicts Rome."""\n'
    res = sc.score([{"id": "good", "source": good}, {"id": "bad", "source": HEAD, "valid": False}], items, N=1000)
    g, b = res
    want = [S.kl_bits(np.array([c["p"] for c in it["candidates"]]), it["other"], np.array([0.6, 0.1])) for it in items]
    assert np.allclose(g["per_item"], want, atol=1e-6)
    assert abs(g["reader_error_bits"] - 1000 * np.mean(want)) < 1e-3
    assert b["mean_bits_per_item"] == b["empty_mean_bits_per_item"]  # invalid = the empty program
    assert g["english_saved_bits"] == pytest.approx(1000 * (g["code_only_mean_bits_per_item"] - g["mean_bits_per_item"]))
    calls = stub.calls
    # The checker's items differ between programs in the program's own outputs only: memo hits.
    for it in items:
        it["q_program"] = [0.9]
    sc.score([{"id": "good", "source": good}], items, N=1000)
    assert stub.calls == calls


def test_cached_reader_matches_full_forward():
    qwen_tokenizer()
    import torch

    r = S.CachedReader("Qwen/Qwen3-0.6B", batch_tokens=4096, max_batch=4, device="cpu")
    prefix = r.tokenizer.encode("The program text, shared by every item.\n", add_special_tokens=False)
    suffixes = [r.tokenizer.encode(s, add_special_tokens=False) for s in ("The capital of Italy is", "One two three", "A")]
    got = r.read(prefix, suffixes, [[(len(s) - 1, None), (0, [5, 6])] for s in suffixes])
    for s, g in zip(suffixes, got):
        with torch.no_grad():
            logits = r.model(torch.tensor([prefix + s])).logits[0].double()
        full = torch.log_softmax(logits[len(prefix) + len(s) - 1], -1).numpy()
        first = torch.log_softmax(logits[len(prefix)], -1).numpy()[[5, 6]]
        assert np.abs(g[0] - full).max() < 1e-3 and np.abs(g[1] - first).max() < 1e-3

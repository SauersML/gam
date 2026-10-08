"""Tests of the discrete reader term (reader_questions.py): question texts and options, the log-loss in bits
with a stub reader, the shuffled control's derangement, the summary, and the answer program's variables.

  ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/test_reader_questions.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import reader_questions as RQ  # noqa: E402

NEXT = {"behavior": "b", "family": "f", "type": "next", "text": "New", "options": [" York", " Delhi", " Jersey", " Orleans"], "answer": 0}
SWITCH = {"behavior": "b", "family": "f", "type": "switch", "text": "A", "counterfactual": "B", "clean_token": " x", "counter_token": " y", "answer": 1}
STEP = {"behavior": "b", "family": "f", "type": "step", "variable": "key", "text": "A", "other": "B", "answer": 0}


def qwen_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen/Qwen3-0.6B is not in the local Hugging Face cache")


def test_question_texts_and_options():
    assert '"\\u0020York"' not in RQ.q_text(NEXT) and 'A. " York"' in RQ.q_text(NEXT) and RQ.q_text(NEXT).endswith("Answer with the letter.")
    assert '" y"' in RQ.q_text(SWITCH) and RQ.q_text(SWITCH).endswith("Answer Yes or No.")
    assert "key" in RQ.q_text(STEP)
    assert RQ.option_words(NEXT) == list("ABCD") and RQ.option_words(STEP) == ["Yes", "No"]


def test_derangement():
    for n in (2, 3, 27):
        p = RQ.derangement(n, seed=0)
        assert sorted(p) == list(range(n)) and all(i != j for i, j in enumerate(p))
    assert RQ.derangement(27, 0) == RQ.derangement(27, 0)


class Stub:
    """A reader whose option log-probabilities are fixed: with an explanation it prefers option 0."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.explained = tokenizer.encode("<<<", add_special_tokens=False)

    def read(self, prefix, suffixes, reads):
        hot = any(prefix[i:i + len(self.explained)] == self.explained for i in range(len(prefix)))
        out = []
        for r in reads:
            n = len(r[0][1])
            lp = np.log(np.full(n, 1.0 / n)) if not hot else np.log(np.array([0.7] + [0.3 / (n - 1)] * (n - 1)))
            out.append([lp])
        return out


def test_log_loss_and_score():
    tok = qwen_tokenizer()
    enc = RQ.Encoder(tok)
    assert len(set(enc.options(NEXT))) == 4 and len(set(enc.options(STEP))) == 2
    stub = Stub(tok)
    bits = RQ.log_loss(stub, enc, "an explanation", [NEXT, SWITCH, STEP])
    assert np.allclose(bits, [-math.log2(0.7), -math.log2(0.3), -math.log2(0.7)])
    none = RQ.log_loss(stub, enc, None, [NEXT, SWITCH])
    assert np.allclose(none, [2.0, 1.0])
    qs = [NEXT, SWITCH, STEP, {**NEXT, "behavior": "c"}]
    res = RQ.score(stub, qs, {"b": "explanation b", "c": "explanation c"})
    s = res["summary"]
    # with an explanation the stub picks option 0: right on NEXT (answer 0) and STEP, wrong on SWITCH (answer 1)
    assert s["next"]["questions"] == 2 and s["next"]["own"] == 1.0 and s["next"]["chance"] == 0.25
    assert s["switch"]["own"] == 0.0 and s["step"]["own"] == 1.0 and s["switch"]["most_common"] == 1.0
    # the control reads the other behavior's explanation (the stub sees one either way)
    assert all(r["shuffled_from"] != r["behavior"] for r in res["rows"])


def test_algorithm_variables():
    source = '''
def helper(text, k):
    return text.rfind(k)


def key(tokens):
    return [t.upper() for t in tokens]


def answer(tokens, key):
    return [k + "!" for k in key]


align(answer, <p:0.fc.1>)
'''
    algorithm, steps = RQ._algorithm(source)
    assert steps == ["key"]
    assert algorithm.values(["a", "b"], ["key", "answer"]) == {"key": ["A", "B"], "answer": ["A!", "B!"]}
    assert RQ._algorithm("x = 1\n") == (None, [])


def test_temperature():
    # Calibration questions read without an explanation: the reader puts 0.99 on option 0, right half the time.
    rows = [{"split": "calibrate", "answer": a, "lp": {"none": [math.log(0.99), math.log(0.01)]}} for a in (0, 1) * 8]
    t = RQ.fit_temperature(rows)
    assert t > 10  # the best a temperature can do for a coin flip is to flatten the reader toward 1/2
    bits, right = RQ.graded([math.log(0.99), math.log(0.01)], 1, t)
    assert bits < 1.2 and not right
    assert RQ.fit_temperature([{"split": "score", "answer": 0, "lp": {"none": [0.0, -1.0]}}]) == 1.0

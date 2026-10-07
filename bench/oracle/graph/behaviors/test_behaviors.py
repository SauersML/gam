"""Tests for the behavior suite generators (MPD #2951 graph oracle).

    ~/mpd-data/venv/bin/python -m pytest bench/oracle/graph/behaviors/test_behaviors.py -q

The generator tests use a whitespace tokenizer, so they need no model files; the data tests check the written behavior
files when ~/mpd-data/graph_oracle/behaviors exists.
"""

import json
import random
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import families  # noqa: E402

DATA = Path.home() / "mpd-data/graph_oracle/behaviors"


class WordTok:
    """Tokens = maximal runs of letters/digits with their leading space, or single other characters."""

    model = "words"
    prefix_ids = []

    def __init__(self):
        self.vocab = {}

    def encode(self, text):
        ids, offs = [], []
        for m in re.finditer(r" ?[A-Za-z0-9]+| ?[^A-Za-z0-9 ]| ", text):
            ids.append(self.vocab.setdefault(m.group(0), len(self.vocab)))
            offs.append((m.start(), m.end()))
        return ids, offs

    def count(self, s):
        return len(self.encode(s)[0])

    def decode(self, i):
        return next(k for k, v in self.vocab.items() if v == i)


@pytest.mark.parametrize("name", sorted(families.FAMILIES))
def test_family_generates_answers_and_counterfactuals(name):
    tok = WordTok()
    variants = families.FAMILIES[name](tok, random.Random(0))
    assert variants
    for v in variants:
        assert v.items and v.description
        for it in v.items:
            assert it.prefix and it.answer.strip()
            if it.cf_prefix is not None:
                assert it.cf_answer.strip()
                assert it.cf_prefix != it.prefix or it.cf_answer != it.answer
            if it.accept:
                assert it.answer in it.accept


def test_encode_item_targets_answer_tokens():
    from build import encode_item
    tok = WordTok()
    text, ids, tp = encode_item(tok, "The capital of France is", " Paris")
    assert text.endswith(" Paris") and tp == [len(ids) - 2]
    _, ids, tp = encode_item(tok, "1, 2, 3,", " 4")
    assert tok.decode(ids[tp[0] + 1]) == " 4"


def test_build_prompts_aligns_counterfactuals():
    from build import build_prompts
    tok = WordTok()
    items = [families.Item("When Mary and John went out, John gave a book to", " Mary",
                           "When Mary and John went out, Mary gave a book to", " John")]
    rows = [("France", "Paris"), ("Spain", "Madrid"), ("Italy", "Rome")]
    items += families.fact_variants(rows, [("plain", "The capital of {X} is")], "test")[0].items
    ps = build_prompts(tok, items, random.Random(0))
    assert len(ps) == 4
    for p in ps:
        cf = p["counterfactual"]
        assert len(cf["token_ids"]) == len(p["token_ids"])
        t = p["target_positions"][0]
        assert cf["token_ids"][t + 1] != p["token_ids"][t + 1]


def test_split_holds_out_novel_families():
    from build import split_of
    for f in families.NOVEL:
        assert f in families.FAMILIES and split_of(f) == "heldout"


def test_retained_state_mask():
    torch = pytest.importorskip("torch")
    if not (Path.home() / "retained-reply-state/hidden_choice.py").exists():
        pytest.skip("introspection study not on this machine")
    import retained_state as R
    m, pos = R.mask_for(6, [[7, 8], [9]], (1, 3), 4, True, torch.float32)
    allowed = m[0, 0] == 0
    assert allowed[3, :4].tolist() == [True, True, True, True]   # the reply (before turn 2) attends to the thinking
    assert allowed[4, 1:3].tolist() == [False, False]            # turn-2 tokens do not
    assert allowed[6, 1:3].tolist() == [False, False]            # nor do candidates
    assert allowed[7, 6] and allowed[7, 7] and not allowed[8, 6]  # a candidate sees its own tokens only
    assert pos == [0, 1, 2, 3, 4, 5, 6, 7, 6]
    assert R.substitute("Draw #5: ocelot. I'll keep Ocelot in mind.", "ocelot", "lynx") == "Draw #5: lynx. I'll keep Lynx in mind."


@pytest.mark.skipif(not DATA.exists(), reason="no behavior data on this machine")
def test_behavior_files_are_consistent():
    files = sorted(DATA.glob("*/*.json"))
    assert files
    for f in files:
        b = json.loads(f.read_text())
        assert b["split"] in ("train", "heldout") and b["prompts"]
        for p in b["prompts"]:
            ids, tp, cf = p["token_ids"], p["target_positions"], p["counterfactual"]
            assert all(0 <= t < len(ids) - 1 for t in tp)
            assert tp[-1] == len(ids) - 2 or "accepted_token_ids" in p
            assert len(cf["token_ids"]) == len(ids)
            assert len(p["correct"]) == len(tp) == len(p["model_top"])
            if b["model"] == "vpd4l":
                assert ids[0] == 0
            for q0, q1, k0, k1 in p.get("attention_block", []):
                assert 0 <= k0 < k1 <= q0 < q1 <= len(ids)

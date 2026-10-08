"""grammar.py: the number rules give exactly 0 .. n - 1; with xgrammar installed (pods; the Mac venv lacks it), the
answer grammar accepts answers whose parts sit in align/claim statements and rejects parts elsewhere, unknown parts
and a second fence.  python -m pytest -q test_grammar.py"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grammar  # noqa: E402

GOOD = """I copy the previous token.
```python
def answer(tokens):
    # x<val, a < b, `code`, x < prev, a <= b
    return tokens


align(answer, <p:2.v.559>, <p:2.o.735>)
claim(answer, <p:1.q.3>, <p:1.k.rest>)
```
Layer 2's value and output subcomponents copy it; `a < b`."""


def test_number_below_is_exact():
    for n in (1, 2, 9, 10, 11, 99, 100, 101, 512, 1024, 3072, 3584):
        pattern = re.compile("|".join(a.replace('"', "").replace(" ", "") for a in grammar.number_below(n).split(" | ")))
        assert [i for i in range(n + 50) if pattern.fullmatch(str(i))] == list(range(n)), n
        assert not pattern.fullmatch("01") and not pattern.fullmatch("")


def test_part_rule_compresses_ranges():
    rule = grammar.part_rule(["<p:0.q.0>", "<p:0.q.1>", "<p:0.q.2>", "<p:1.v.5>", "<p:1.m>"])
    assert '"<p:0.q." (' in rule and '"<p:1.v.5>"' in rule and '"<p:1.v.rest>"' in rule and '"<p:1.m>"' in rule
    assert len(grammar.vpd_tokens("vpd4l")) == 38_912


@pytest.mark.skipif(importlib.util.find_spec("xgrammar") is None, reason="xgrammar is not installed")
def test_answers_against_the_grammar():
    import xgrammar as xgr
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tok)).compile_grammar(grammar.model_grammar("vpd4l"))

    def accepts(text):
        m = xgr.GrammarMatcher(compiled)
        return all(m.accept_token(t) for t in tok.encode(text, add_special_tokens=False)) and m.accept_token(tok.convert_tokens_to_ids("<|im_end|>"))

    assert accepts(GOOD)
    assert accepts(GOOD.split("I copy the previous token.\n")[1])  # no prose before the block
    assert not accepts(GOOD.replace("# x<val", "# <p:2.v.559> x<val"))  # a part in a comment
    assert not accepts(GOOD.replace("<p:2.v.559>", "<p:2.v.1024>"))  # beyond v_proj's 1,024 subcomponents
    assert not accepts(GOOD.replace("<p:2.v.559>", "<p:7.v.5>"))  # no layer 7
    assert not accepts(GOOD + "\n```python\nx = 1\n```")  # a second block
    assert not accepts(GOOD.replace("copy it;", "copy <p:2.v.559>;"))  # a part in the English
    assert not accepts(GOOD.replace("```\nLayer", "Layer"))  # the block never closes


@pytest.mark.skipif(importlib.util.find_spec("xgrammar") is None, reason="xgrammar is not installed")
def free_text(s: str) -> bool:
    """grammar.free's language: "<" never followed by "p", "<" or "`", "`" never by "`" or "<", no newline (a code line)."""
    return "\n" not in s and all(not (a == "<" and b in "p<`") and not (a == "`" and b in "`<") for a, b in zip(s, s[1:]))


def test_free_text_is_exactly_its_language():
    """Every string of up to 6 characters over a < p : ` and a newline; the language has neither a part nor a fence."""
    import itertools

    import xgrammar as xgr
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tok)).compile_grammar('root ::= free_line "!"\n' + grammar.free("free_line", "\\n"))
    for n in range(7):
        for chars in itertools.product("a<p:`\n", repeat=n):
            s = "".join(chars)
            m = xgr.GrammarMatcher(compiled)
            got = all(m.accept_token(t) for t in tok.encode(s + "!", add_special_tokens=False)) and m.is_terminated() is False and m.accept_token(tok.convert_tokens_to_ids("<|im_end|>"))
            assert got == free_text(s), repr(s)
            assert not got or ("<p:" not in s and "```" not in s)

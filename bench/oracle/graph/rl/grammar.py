"""The oracle answer's grammar for guided decoding (#2951 graph oracle, format v4): vLLM samples only answers of the
canonical form edits.Answer writes:

  ```python
  def <name>(tokens):            (zero or more functions a node's "at" names; indented body lines)
      <code>


  nodes = {
      "<name>": {"subcomponents": ["<p:L.S.I>", ...], "at": "all" | "targets" | "last" | <name>},   (one or more)
  }
  edges = [
      ("<writer>", "<reader>"[, "query" | "key" | "value"]),                                          (one or more)
  ]
  labels = {"<node>": "<variable>", ...}                                                              (optional)
  ```

  <the English explanation: any number of lines, then the turn ends>

with subcomponents of the attached decomposition only (the registry's part tokens, plus each site's remainder
<p:L.S.rest>); writers "input" or a node, readers "output" or a node. Code lines and the explanation are free text
with neither "<p:" nor three backticks in a row. mech still checks what a grammar cannot see (names, connections,
variables of the behavior), and validity redraws (train.py --resample) cover those. The grammar is xgrammar's EBNF; a
part token and its spelling in ordinary tokens both match (a part token's text is its name).

  grammar.py MODEL [--registry REG.safetensors] [--check ANSWER.txt ...]   (prints the grammar, or checks answers with xgrammar)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE_CODES = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "c_fc": "fc", "down_proj": "down"}  # part_tokens.SITES

def free(n: str, x: str) -> str:
    """Rule n for free text in which "<" is never followed by "p", "<" or "`" and "`" never by "`" or "<" (and without
    the characters x; code: a tab). This is slightly narrower than "no <p: and no three backticks" (it also
    excludes x<p..., <<, double backticks and <`): the exact language needs a two-character exit after "<p", and runs
    "<"+, for which xgrammar checks every token against the parser stack (16-100 ms per mask on the Mac); this one
    stays within single character classes (0.07 ms)."""
    return f"""{n} ::= [^{x}<`]* {n}_s* ("<" | "`")?
{n}_s ::= ("<" [^{x}<p`] | "`" [^{x}<`]) [^{x}<`]*"""


def number_below(n: int) -> str:
    """EBNF alternatives for the decimal numbers 0 .. n - 1 without leading zeros."""
    if n <= 0:
        raise ValueError("no numbers below 0")
    m = str(n - 1)
    alts = ['"0"']
    for d in range(1, len(m)):  # every number of fewer digits than n - 1
        alts.append("[1-9]" + " [0-9]" * (d - 1))
    for i, c in enumerate(m):  # numbers of len(m) digits up to n - 1: the first digit below m's at position i
        low = 1 if i == 0 else 0
        if int(c) > low:
            prefix = " ".join(f'"{ch}"' for ch in m[:i])
            alts.append(" ".join(x for x in (prefix, f"[{low}-{int(c) - 1}]", " ".join(["[0-9]"] * (len(m) - i - 1))) if x))
    if m != "0":
        alts.append(" ".join(f'"{ch}"' for ch in m))
    return " | ".join(dict.fromkeys(alts))


def part_rule(tokens: list[str]) -> str:
    """EBNF for a set of part names: the names <p:L.S.I> of each layer and site whose indices are 0 .. C - 1 as one
    number rule (with the site's remainder <p:L.S.rest>), every other name as itself."""
    sites, other = {}, []
    for t in tokens:
        m = re.fullmatch(r"<p:(\d+)\.([a-z]+)\.(\d+|rest)>", t)
        if m and m[3] != "rest":
            sites.setdefault((int(m[1]), m[2]), set()).add(int(m[3]))
        elif not m:
            other.append(t)
    alts = []
    for (layer, site), idx in sorted(sites.items()):
        if idx == set(range(len(idx))):
            alts.append(f'"<p:{layer}.{site}." ({number_below(len(idx))} | "rest") ">"')
        else:
            alts += [f'"<p:{layer}.{site}.{i}>"' for i in sorted(idx)] + [f'"<p:{layer}.{site}.rest>"']
    alts += [json.dumps(t) for t in sorted(set(other))]
    return " | ".join(alts)


def vpd_tokens(model: str) -> list[str]:
    """The part names of a model's VPD view (shapes.json): <p:L.S.I> for every subcomponent I of every site."""
    view = json.loads((HERE.parent / "shapes.json").read_text())[model]["views"]["vpd"]
    return [f"<p:{layer}.{SITE_CODES[site]}.{i}>" for layer, sites in enumerate(view) for site, n in sites.items() for i in range(n)]


def answer_grammar(tokens: list[str]) -> str:
    """The answer's EBNF (module docstring) with subcomponents drawn from `tokens`."""
    q = json.dumps
    return "\n".join([
        'root ::= "```python\\n" function* "nodes = {\\n" node+ "}\\nedges = [\\n" edge+ "]\\n" labels? "```\\n\\n" explanation',
        'function ::= "def " name "(tokens):\\n" body+ "\\n\\n"',
        'body ::= "    " code "\\n"',
        free("code", "\\n\\t"),
        "node ::= " + " ".join([q('    "'), "name", q('": {"subcomponents": ['), "parts", q('], "at": '), "at", q("},\n")]),
        'at ::= "\\"all\\"" | "\\"targets\\"" | "\\"last\\"" | name',
        'parts ::= part (", " part)*',
        'part ::= "\\"" (' + part_rule(tokens) + ') "\\""',
        "edge ::= " + " ".join([q('    ("'), "endpoint", q('", "'), "endpoint", q('"'), "route?", q("),\n")]),
        'endpoint ::= "input" | "output" | name',
        "route ::= " + q(', "') + ' ("query" | "key" | "value") "\\""',
        "labels ::= " + " ".join([q('labels = {'), "label", "(", q(", "), "label", ")*", q("}\n")]),
        "label ::= " + " ".join([q('"'), "name", q('": "'), "name", q('"')]),
        "name ::= [A-Za-z_] [A-Za-z0-9_]*",
        free("explanation", ""),
    ])


def model_grammar(model: str, registry: str | None = None) -> str:
    """The grammar of a model's answers: parts from the part-token registry when given, else the model's VPD view."""
    if registry:
        import sys

        sys.path.insert(0, str(HERE.parent))
        import part_tokens

        return answer_grammar(part_tokens.Registry.load(registry).tokens)
    return answer_grammar(vpd_tokens(model))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--registry")
    ap.add_argument("--check", nargs="*", help="answers to match against the grammar (xgrammar, the Qwen3 tokenizer)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    a = ap.parse_args()
    g = model_grammar(a.model, a.registry)
    if not a.check:
        print(g)
        return
    import xgrammar as xgr
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    compiled = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tok)).compile_grammar(g)
    for path in a.check:
        matcher = xgr.GrammarMatcher(compiled)
        ids = tok.encode(Path(path).read_text().strip(), add_special_tokens=False)  # as SFT targets are written (stripped)
        ok = all(matcher.accept_token(t) for t in ids)
        print(json.dumps({"answer": path, "accepted": ok, "complete": ok and matcher.accept_token(tok.eos_token_id)}))


if __name__ == "__main__":
    main()

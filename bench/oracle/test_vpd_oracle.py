"""Checks of vpd_oracle.py's question construction (#2951) on the real label tables (skipped when absent):
every component a question names (its subject, an edge's cut writer, an attribution's candidates) lies
inside the side of the split it is drawn for, and every question kind offers the same number of options
in the same form whatever its answer.

  python bench/oracle/test_vpd_oracle.py      (or pytest)
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd_2951"))

import vpd_oracle as O  # noqa: E402

DATA = Path.home() / "mpd-data/oracle"
FULL = (DATA / "vpd/labels_heldout", DATA / "vpd/relations_heldout")
ROW = DATA / "vpd_row/labels_row"
UV = DATA / "vpd/uv.safetensors"


def named(ex: dict) -> list[tuple[int, int]]:
    return ([(ex["layer"], ex["c"])] if ex["c"] >= 0 else []) + [(l, c) for l, _, c, _, _ in ex["candidates"]]


def check(table: O.Table, splits: list[tuple[set, int, str, callable]], count: int = 400):
    shapes: dict[str, set] = defaultdict(set)
    for layers, held, side, inside in splits:
        for stratified in (False, True):
            for ex in O.examples(table, layers, count, 7, stratified=stratified, held=held, side=side):
                for layer, c in named(ex):
                    assert inside(layer, c), (side, ex["kind_q"], layer, c)
                # The options' number and every option that does not name a token or text of this question.
                fixed = tuple(o for o in ex["options"] if not o.startswith("No, it stays"))
                shapes[ex["kind_q"]].add((len(ex["options"]), fixed if ex["kind_q"] in ("continuation", "effect", "edge", "activity") else ()))
    for q, forms in shapes.items():
        assert len(forms) == 1, (q, forms)


def test_layer_split_full_table():
    if not FULL[0].exists():
        return
    table = O.Table(FULL[0], UV, FULL[1], O.TOKENIZER)
    check(table, [({2}, 0, "trained", lambda l, c: l == 2), ({0, 1, 3}, 0, "trained", lambda l, c: l != 2)])


def test_subcomponent_split_full_table():
    if not FULL[0].exists():
        return
    table = O.Table(FULL[0], UV, FULL[1], O.TOKENIZER)
    every = set(table.layers)
    check(table, [(every, 5, "heldout", lambda l, c: c % 5 == 0), (every, 5, "trained", lambda l, c: c % 5 != 0)])


def test_subcomponent_split_row_table():
    if not ROW.exists():
        return
    table = O.Table(ROW, UV, None, O.TOKENIZER)
    every = set(table.layers)
    check(table, [(every, 5, "heldout", lambda l, c: c % 5 == 0), (every, 5, "trained", lambda l, c: c % 5 != 0)])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok", flush=True)

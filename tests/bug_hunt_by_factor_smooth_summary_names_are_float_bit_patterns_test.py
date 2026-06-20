"""Bug hunt: ``s(x, by=<factor>)`` per-level smooth terms are named with the
raw IEEE-754 bit pattern of the level value instead of the level label, so
``Model.summary()`` cannot tell the user which group each smooth belongs to.

When a factor ``by=`` smooth is expanded into one smooth per level, the builder
has the level's *label* in hand — ``encoded_levels_for_column`` returns
``(level_bits, level_label)`` pairs and ``level_label`` is the original schema
level string (``src/terms/term_builder.rs:305-326``). But the per-level term
name is then built from ``level_bits`` (the ``f64::to_bits()`` u64) rather than
``level_label``::

    // src/terms/term_builder.rs:644-646
    for (level_bits, level_label) in levels {
        smooth_terms.push(SmoothTermSpec {
            name: format!("{}:{}", label, level_bits),   // <-- level_bits, not level_label
            ...

So a model fit on labels ``["red", "green", "blue", "gold", "cyan"]`` produces
smooth-term names like::

    s(x,by=grp):0
    s(x,by=grp):4607182418800017408     # == (1.0_f64).to_bits()
    s(x,by=grp):4611686018427387904     # == (2.0_f64).to_bits()
    s(x,by=grp):4613937818241073152     # == (3.0_f64).to_bits()
    s(x,by=grp):4616189618054758400     # == (4.0_f64).to_bits()

None of which mentions ``red`` / ``green`` / ``blue`` / ``gold`` / ``cyan``. The
names are not even the level *index* — they are the 64-bit reinterpretation of
the encoded ``f64`` level value, which is meaningless to a reader.

Expected: the per-level smooth-term names must carry the level *label*, so each
of the original factor levels is recoverable from ``summary().smooth_terms``.
"""

import os

os.environ.setdefault("GAM_LOG", "off")

import numpy as np
import pandas as pd

import gamfit


LABELS = ["red", "green", "blue", "gold", "cyan"]


def _by_level_term_names():
    rng = np.random.default_rng(5)
    n = 300 * len(LABELS)
    x = rng.uniform(0.0, 1.0, n)
    g = rng.choice(LABELS, n)
    phase = {lab: i for i, lab in enumerate(LABELS)}
    y = np.sin(2.0 * np.pi * x + np.array([phase[gi] for gi in g])) + 0.15 * rng.standard_normal(n)
    df = pd.DataFrame({"x": x, "grp": g, "y": y})
    model = gamfit.fit(df, "y~s(x,by=grp)", family="gaussian")
    frame = model.summary().smooth_terms_frame()
    mask = frame["name"].astype(str).str.startswith("s(x,by=grp):")
    return [str(name) for name in frame[mask]["name"]]


def test_by_factor_smooth_names_reference_the_level_labels():
    names = _by_level_term_names()
    assert len(names) == len(LABELS), (
        f"expected {len(LABELS)} by-level smooth terms, got {len(names)}: {names}"
    )
    # Each original label must be recoverable from some term name. The label is
    # available to the builder (level_label) but discarded in favour of the f64
    # bit pattern, so none of these currently appear.
    for label in LABELS:
        assert any(label in name for name in names), (
            f"factor level {label!r} does not appear in any by-smooth term name "
            f"{names}: the per-level smooth name is built from the f64 bit "
            "pattern of the level value (term_builder.rs:646) instead of the "
            "level label, so summary() cannot identify which group each smooth is."
        )


def test_by_factor_smooth_names_are_not_float_bit_patterns():
    names = _by_level_term_names()
    assert len(names) == len(LABELS), (
        f"expected {len(LABELS)} by-level smooth terms, got {len(names)}: {names}"
    )
    # The defective names are `s(x,by=grp):<u64>` where <u64> == f64::to_bits()
    # of the encoded level value (0, 1.0, 2.0, ...). Concretely no term name
    # should end in the bit pattern of a small-integer-valued double.
    forbidden = {str(np.float64(v).view(np.uint64)) for v in range(len(LABELS))}
    for name in names:
        suffix = name.rsplit(":", 1)[-1]
        assert suffix not in forbidden, (
            f"by-smooth term name {name!r} encodes the level as the f64 bit "
            f"pattern {suffix!r} (== to_bits of a small integer); the level "
            "label is being thrown away (term_builder.rs:646)."
        )

"""Regression tests for #1544.

`MultinomialModel.summary()` used to abort on *every* multinomial fit that
contained a smooth term with::

    ValueError: Multinomial lambda metadata mismatch: 1 term labels but
    block 0 has 2 lambdas

The guard assumed a 1:1 mapping between smooth-term labels and the per-class λ
count, but the default Marra–Wood double penalty emits two λ per smooth term
per class (a primary wiggliness penalty + a null-space shrinkage penalty). A
secondary defect paired labels with λ via ``zip(term_labels, lam_chunk)``,
which would silently *drop* the null-space λ even if the guard were relaxed.

The fix records one descriptive label per penalty component in the saved model
(`lambda_labels`) and renders λ component-for-component, so every λ survives.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _softmax_multinomial_frame(seed: int, n: int, fields: dict) -> pd.DataFrame:
    """Draw class labels from a softmax over the supplied per-class logits."""
    rng = np.random.default_rng(seed)
    cols = {name: rng.uniform(-3.0, 3.0, n) for name in fields["covariates"]}
    F = fields["logits"](cols)  # shape (n, K)
    P = np.exp(F - F.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    draws = (rng.uniform(size=n)[:, None] < np.cumsum(P, 1)).argmax(1)
    labels = np.asarray(fields["labels"])[draws]
    return pd.DataFrame({**cols, "y": labels})


def test_summary_does_not_raise_on_smooth_term_multinomial() -> None:
    """The originally-reported repro: a single smooth term, three classes."""
    df = _softmax_multinomial_frame(
        seed=0,
        n=500,
        fields={
            "covariates": ["x"],
            "labels": ["A", "B", "C"],
            "logits": lambda c: np.stack(
                [np.zeros_like(c["x"]), 1.5 * np.sin(c["x"]), 0.8 * c["x"]], 1
            ),
        },
    )
    m = gamfit.fit(df, "y ~ s(x)", family="multinomial")

    # predict was always fine — assert it still is, as a sanity anchor.
    probs = m.predict(df.head(3))
    assert probs.shape == (3, 3)

    summary = m.summary()
    assert isinstance(summary, str) and summary.strip()
    assert "s(x)" in summary
    assert "λ = [" in summary  # the per-class λ rollup rendered

    # __str__ delegates to summary(); it must not raise either.
    assert str(m) == summary
    assert "MultinomialModel formula" in str(m)


def test_summary_renders_every_lambda_no_truncation() -> None:
    """Different angle: assert no λ is dropped by the renderer.

    The secondary defect (`zip(term_labels, lam_chunk)`) truncated the rendered
    λ list to the *number of term labels*, silently discarding the null-space λ
    of each smooth. With two smooth terms and the double penalty there are 4 λ
    per class but only 2 term labels — the old renderer would have shown 2.
    Here we count the rendered ``name: value`` pairs against the model's own
    `lambdas_per_block` and require an exact match for every class.
    """
    df = _softmax_multinomial_frame(
        seed=7,
        n=700,
        fields={
            "covariates": ["x", "z"],
            "labels": ["A", "B", "C", "D"],  # K = 4 -> 3 active classes
            "logits": lambda c: np.stack(
                [
                    np.zeros_like(c["x"]),
                    1.2 * np.sin(c["x"]) - 0.5 * c["z"],
                    0.7 * c["x"] + 0.9 * np.cos(c["z"]),
                    -0.6 * c["x"] + 0.4 * c["z"],
                ],
                1,
            ),
        },
    )
    m = gamfit.fit(df, "y ~ s(x) + s(z)", family="multinomial")

    meta = m._metadata
    per_block = list(meta.get("lambdas_per_block", []))
    assert per_block, "smooth multinomial fit must expose per-class λ blocks"
    # The double penalty must produce more λ per block than smooth terms,
    # otherwise this test would not exercise the truncation path it guards.
    n_terms = len(list(meta.get("smooth_term_labels", [])))
    assert n_terms == 2
    assert all(b > n_terms for b in per_block), per_block

    summary = m.summary()
    assert isinstance(summary, str) and summary.strip()

    # Every per-class λ line must render exactly `lambdas_per_block[a]`
    # `name: value` entries — i.e. nothing truncated.
    lam_lines = [ln for ln in summary.splitlines() if "λ = [" in ln]
    assert len(lam_lines) == len(per_block)
    for line, expected in zip(lam_lines, per_block):
        inner = line.split("λ = [", 1)[1].rsplit("]", 1)[0]
        rendered = [chunk for chunk in inner.split(", ") if chunk]
        assert len(rendered) == expected, (line, expected)

    # Both terms and their null-space components must be named.
    assert "s(x)" in summary and "s(z)" in summary
    assert "null space" in summary  # the double penalty's shrinkage λ is shown


def test_summary_survives_four_class_single_smooth() -> None:
    """A 4-class single-smooth fit (the issue's second table row)."""
    df = _softmax_multinomial_frame(
        seed=3,
        n=600,
        fields={
            "covariates": ["x"],
            "labels": ["A", "B", "C", "D"],
            "logits": lambda c: np.stack(
                [
                    np.zeros_like(c["x"]),
                    1.5 * np.sin(c["x"]),
                    0.8 * c["x"],
                    -1.0 * np.cos(c["x"]),
                ],
                1,
            ),
        },
    )
    m = gamfit.fit(df, "y ~ s(x)", family="multinomial")
    summary = m.summary()
    assert isinstance(summary, str) and summary.strip()
    # Three active classes, each with its own λ line.
    assert summary.count("λ = [") == 3

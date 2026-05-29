"""Pin the distinct semantics of the `reml` and `tk` score kinds.

`select_topology(score='reml')` ranks candidates by the bare raw REML /
evidence score, while `score='tk'` adds the Tierney-Kadane Laplace
normalizer on top of it. Their `_score_for_kind` arms must NOT be
identical: that duplicate-body bug made `reml` silently inherit the
gauge-sensitive TK normalizer.
"""
from __future__ import annotations

import gamfit._select_topology as st


class _Fit:
    def __init__(self, reml: float, null_logdet: float) -> None:
        self._summary = {
            "reml_score": reml,
            "null_hessian_logdet": null_logdet,
            "effective_dim": 1.0,
            "coefficients": [0.0, 0.0],
        }

    def summary(self) -> dict[str, object]:
        return self._summary


def test_reml_score_is_bare_raw_with_no_tk_normalizer(monkeypatch) -> None:
    raw = -3.25
    fit_obj = _Fit(reml=raw, null_logdet=7.0)

    # Force a non-zero Tierney-Kadane normalizer so a leaked normalizer
    # would be observable.
    monkeypatch.setattr(st, "_extract_reml_score_raw", lambda m: raw)
    monkeypatch.setattr(st, "_tk_normalizer_for_fit", lambda fit, null_dim: 1.75)

    reml = st._score_for_kind(fit_obj, "reml", n_obs=10, basis_size=2, null_dim=1.0)
    tk = st._score_for_kind(fit_obj, "tk", n_obs=10, basis_size=2, null_dim=1.0)

    assert reml == raw, "score='reml' must be the bare raw REML score"
    assert tk == raw + 1.75, "score='tk' must add the Tierney-Kadane normalizer"
    assert reml != tk, "the 'reml' and 'tk' arms must not be identical"

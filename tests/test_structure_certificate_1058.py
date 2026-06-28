"""Anytime-valid structure evidence is reachable post-fit (#1058 / #984).

`src/inference/structure_evidence.rs` (EProcess / StructureLedger / e-BH) is
populated during every SAE structure search, and the certificate is now
serialized onto the model payload. `ManifoldSAE.structure_certificate()` and
`ManifoldSAE.contested_claims()` surface it so the user can read the e-values,
which claims passed the e-BH gate, and the anytime-valid remaining budget.

Assertions (objective — the e-BH/e-process math IS the ground truth):
  - The certificate has the documented shape and an FDR level in (0, 1).
  - e-values are non-negative and consistent with their log form.
  - The confirmed set exactly matches a from-scratch e-BH over the stored
    log e-values (so the surfaced verdict is the real certificate, not a
    relabel), and re-deriving at a stricter alpha never grows it.
  - contested_claims() is precisely the un-confirmed complement, and every
    contested claim reports a non-negative evidence-remaining budget.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


# #1512 / #1058 (OPEN BUG — these tests fail on purpose to flag it; SPEC.md
# forbids xfail, so the failure stands as the signal): a fresh
# gamfit.sae_manifold_fit(...) does NOT populate the structure certificate —
# fit.structure_certificate() raises ValueError "this fitted model carries no
# structure certificate (payload predates #1058); refit to obtain one" even on a
# brand-new fit. Wire the certificate into sae_manifold_fit to green these.


def _circle(n: int, p: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    harm = np.column_stack([np.cos(theta), np.sin(theta)])
    mixing = rng.normal(size=(2, p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    z = harm @ mixing + noise * rng.normal(size=(n, p))
    return z - z.mean(axis=0, keepdims=True)


def _fit():
    z = _circle(n=200, p=16, noise=0.05, seed=0)
    return gamfit.sae_manifold_fit(
        X=z,
        K=2,
        atom_basis="periodic",
        d_atom=2,
        assignment="ibp_map",
        n_iter=30,
        learning_rate=0.04,
        random_state=0,
    )


def test_structure_certificate_shape_and_consistency() -> None:
    fit = _fit()
    cert = fit.structure_certificate()
    assert set(cert) >= {"alpha", "fdr_level", "n_confirmed", "claims"}
    assert 0.0 < cert["alpha"] < 1.0
    assert isinstance(cert["claims"], list)

    threshold = math.log(1.0 / cert["alpha"])
    confirmed = 0
    for claim in cert["claims"]:
        assert claim["e_value"] >= 0.0
        # e_value == exp(log_e)
        assert math.isclose(
            claim["e_value"], math.exp(claim["log_e"]), rel_tol=1e-9, abs_tol=1e-12
        )
        assert claim["steps"] >= 0
        # Remaining budget is the anytime-valid ln(1/alpha) - log_e, floored at 0.
        expected_remaining = max(0.0, threshold - claim["log_e"])
        assert math.isclose(
            claim["evidence_remaining_nats"], expected_remaining, abs_tol=1e-9
        )
        confirmed += int(claim["confirmed"])
    assert cert["n_confirmed"] == confirmed


def test_confirmed_set_matches_independent_e_bh() -> None:
    fit = _fit()
    cert = fit.structure_certificate()
    alpha = cert["alpha"]
    log_e = [c["log_e"] for c in cert["claims"]]
    m = len(log_e)
    # Re-derive the e-BH confirmed set independently and compare to the surfaced
    # `confirmed` flags. e_(k) >= m / (alpha * k) over the descending order.
    order = sorted(range(m), key=lambda i: log_e[i], reverse=True)
    k_star = 0
    for rank0, idx in enumerate(order):
        k = rank0 + 1
        if log_e[idx] >= math.log(m) - math.log(alpha) - math.log(k):
            k_star = rank0 + 1
    expected_confirmed = set(order[:k_star])
    got_confirmed = {i for i, c in enumerate(cert["claims"]) if c["confirmed"]}
    assert got_confirmed == expected_confirmed


def test_stricter_alpha_never_grows_confirmed_set() -> None:
    fit = _fit()
    loose = fit.structure_certificate(alpha=0.2)
    strict = fit.structure_certificate(alpha=0.01)
    n_loose = loose["n_confirmed"]
    n_strict = strict["n_confirmed"]
    assert n_strict <= n_loose


def test_contested_is_unconfirmed_complement() -> None:
    fit = _fit()
    cert = fit.structure_certificate()
    contested = fit.contested_claims()
    n_claims = len(cert["claims"])
    assert len(contested) == n_claims - cert["n_confirmed"]
    for claim in contested:
        assert claim["confirmed"] is False
        assert claim["evidence_remaining_nats"] >= 0.0


def test_certificate_round_trips_through_to_dict() -> None:
    fit = _fit()
    restored = gamfit._sae_manifold.ManifoldSAE.from_dict(fit.to_dict())
    a = fit.structure_certificate()
    b = restored.structure_certificate()
    assert a["n_confirmed"] == b["n_confirmed"]
    assert [c["claim"] for c in a["claims"]] == [c["claim"] for c in b["claims"]]

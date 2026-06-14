"""Anytime-valid structure discovery (issue #984).

Thin wrapper over the Rust core ``gam::inference::structure_evidence``: a
universal-inference (split-likelihood-ratio) e-process that decides "does atom
K+1 exist?" in the boundary / Davies regime where the χ² gate is broken, plus
the e-BH dictionary-level FDR certificate (Wang–Ramdas, valid under arbitrary
dependence — the regime p-value BH cannot legally handle when atoms share
tokens). All evidence accounting lives in Rust; this module only marshals the
scalars across the FFI boundary.

The predictability contract that makes the e-process a supermartingale under
H0 is enforced by construction: :meth:`AtomBirthGate.absorb_shard` consumes
only the two pre-computed per-shard log-likelihoods, so the gate can never peek
at the current shard's refit. Feed it the PREVIOUS shard's fitted dictionary
likelihood and the current shard's honest null sup; the gate folds them in.

Classes
-------
AtomBirthGate
    Resumable, optional-stopping-immune atom-existence test.

Functions
---------
split_likelihood_log_e
    One universal-inference log e-value ``log E = ℓ_alt(D0) − sup_H0 ℓ(D0)``.
e_bh_dictionary_certificate
    e-BH FDR-controlled list of confirmed claims from per-claim log e-values.
log_e_from_p_value
    Calibrate a p-value into a conservative valid log e-value (``e = 1/p̂``).
select_probe_by_expected_evidence
    Pick the steering probe with maximal expected log-evidence growth.
expected_resolution_budget
    Convert per-observation evidence growth into an expected certification
    budget.
plan_probe_for_contested_claim
    Pick the next probe and discount its remaining budget by current evidence.
"""

from __future__ import annotations

from typing import Any, Sequence

from ._binding import rust_module


def atom_birth_gate(alpha: float) -> Any:
    """Open an :class:`AtomBirthGate` at significance level ``alpha`` in (0, 1).

    The level is fixed at construction so the verdict can never be α-shopped
    after the evidence is seen. Returns the Rust-backed gate object; absorb one
    shard at a time with :meth:`absorb_shard(alternative_prefit_loglik,
    null_sup_loglik)`, then read :meth:`verdict` / :meth:`certified` /
    :meth:`log_e_value`.
    """
    return rust_module().AtomBirthGate(float(alpha))


def split_likelihood_log_e(
    log_lik_alternative_on_eval: float,
    log_lik_null_sup_on_eval: float,
) -> float:
    """One universal-inference (split-LR) log e-value.

    ``log E = ℓ_alt(D0) − sup_H0 ℓ(D0)``: finite-sample valid with NO
    regularity conditions. ``log_lik_alternative_on_eval`` is the eval-fold
    log-likelihood under the alternative fit on the estimation fold;
    ``log_lik_null_sup_on_eval`` is the supremum of the eval-fold
    log-likelihood over the null model class.
    """
    return rust_module().split_likelihood_log_e(
        float(log_lik_alternative_on_eval),
        float(log_lik_null_sup_on_eval),
    )


def e_bh_dictionary_certificate(
    log_e_values: Sequence[float],
    alpha: float,
) -> list[int]:
    """e-BH dictionary certificate: indices of confirmed claims, FDR ≤ alpha.

    One log e-value per claimed atom/edge; FDR is controlled under ARBITRARY
    dependence (atoms sharing every token is fine — exactly the PRDS-violating
    case ordinary p-value BH cannot handle).
    """
    return rust_module().e_bh_dictionary_certificate(
        [float(v) for v in log_e_values], float(alpha)
    )


def log_e_from_p_value(p_value: float) -> float:
    """Calibrate a p-value in (0, 1] into a conservative valid log e-value.

    Lets a p-value-only claim join :func:`e_bh_dictionary_certificate`.
    """
    return rust_module().log_e_from_p_value(float(p_value))


def select_probe_by_expected_evidence(
    delta: Any,
    predicted_mean_null: Any,
    predicted_mean_alt: Any,
    fisher: Any,
) -> dict[str, Any] | None:
    """Pick the candidate probe with maximal expected evidence growth.

    ``delta``, ``predicted_mean_null``, and ``predicted_mean_alt`` are
    row-aligned ``(n_probes, p_out)`` arrays. The score is
    ``0.5 * (mu_alt - mu_null).T @ fisher @ (mu_alt - mu_null)``: expected
    log-growth, in nats per observation, of the contested claim's e-process
    under the alternative. Returns ``None`` when no candidate discriminates.
    """
    return rust_module().select_probe_by_expected_evidence(
        delta,
        predicted_mean_null,
        predicted_mean_alt,
        fisher,
    )


def expected_resolution_budget(
    alpha: float,
    growth_nats_per_obs: float,
) -> float | None:
    """Expected observations to cross the ``1 / alpha`` evidence threshold.

    ``growth_nats_per_obs`` is the selected probe's expected per-observation
    log-evidence growth. Returns ``None`` for invalid levels or non-positive
    growth.
    """
    return rust_module().expected_resolution_budget(
        float(alpha),
        float(growth_nats_per_obs),
    )


def plan_probe_for_contested_claim(
    delta: Any,
    predicted_mean_null: Any,
    predicted_mean_alt: Any,
    fisher: Any,
    alpha: float,
    current_log_e: float = 0.0,
) -> dict[str, Any] | None:
    """Plan the next KL-optimal steering probe for a contested claim.

    The selected probe maximizes predicted hypothesis disagreement in the
    output-Fisher metric, then reports both the from-scratch and remaining
    observation budgets after accounting for ``current_log_e`` already banked
    in the claim's e-process. Returns ``None`` when steering cannot distinguish
    the candidate hypotheses.
    """
    return rust_module().plan_probe_for_contested_claim(
        delta,
        predicted_mean_null,
        predicted_mean_alt,
        fisher,
        float(alpha),
        float(current_log_e),
    )

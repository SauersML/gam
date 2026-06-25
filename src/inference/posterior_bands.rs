use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

use crate::families::inverse_link::{apply_inverse_link_spec_vec, apply_inverse_link_vec};
use crate::types::InverseLink;
use crate::util::quantile::quantile_from_sorted;

/// Inverse-link selector for the posterior-band engine.
///
/// The response-scale push-through accepts either the legacy bare string tag
/// (`Tag`) — which cannot represent the parameterized links — or the fully
/// parameterized [`InverseLink`] spec (`Spec`) carried through the FFI as
/// `link_spec`. `Spec` is required for `Sas` / `Mixture` / `LatentCLogLog` /
/// `BetaLogistic`; for the `Standard` links the two paths agree (modulo the
/// documented exact-`exp` Log contract, which both honor).
#[derive(Clone, Copy, Debug)]
pub enum LinkSelector<'a> {
    Tag(&'a str),
    Spec(&'a InverseLink),
}

impl LinkSelector<'_> {
    fn apply(&self, eta: &[f64]) -> Result<Vec<f64>, String> {
        match self {
            LinkSelector::Tag(tag) => apply_inverse_link_vec(eta, tag),
            LinkSelector::Spec(spec) => apply_inverse_link_spec_vec(eta, spec),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PosteriorPredictBandsPayload {
    pub linear_predictor: Vec<f64>,
    pub linear_predictor_lower: Vec<f64>,
    pub linear_predictor_upper: Vec<f64>,
    pub mean: Vec<f64>,
    pub mean_lower: Vec<f64>,
    pub mean_upper: Vec<f64>,
    pub n_rows: usize,
    pub n_draws: usize,
    pub model_class: String,
    pub family_kind: String,
}

/// Posterior eta matrix (n_draws x n_rows) -> per-row bands.
///
/// Requires at least one posterior draw. Link-scale credible bounds are
/// quantiles of eta draws; response-scale credible bounds are quantiles of the
/// inverse-link-transformed draws. Direct response quantiles intentionally do
/// not reuse transformed eta quantiles: finite-sample interpolated quantiles do
/// not commute with nonlinear inverse links.
/// The response-scale **point estimate** is the posterior mean of the
/// response-scale draws — i.e. `E[g^{-1}(eta)]`, **not** `g^{-1}(E[eta])`.
/// For nonlinear inverse links (logit, log, probit, cloglog) the two differ
/// by Jensen's inequality, and consumers of the `"mean"` field expect the
/// former (it should equal the per-row posterior mean of `predict_draws`'
/// `mean` matrix). Computing it as `g^{-1}(mean(eta))` instead would give
/// the *median* of the response-scale draws under a monotone link, which is
/// a different summary and silently biases reported response-scale means.
pub fn eta_bands_from_matrix(
    eta: ArrayView2<'_, f64>,
    family_kind: &str,
    level: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
    eta_bands_from_matrix_link(eta, LinkSelector::Tag(family_kind), level)
}

/// Spec-aware sibling of [`eta_bands_from_matrix`]: identical band/mean
/// semantics, but the response-scale push-through goes through a
/// [`LinkSelector`] so the parameterized links (`Sas`, `Mixture`,
/// `LatentCLogLog`, `BetaLogistic`) can be evaluated from their fitted state.
pub fn eta_bands_from_matrix_link(
    eta: ArrayView2<'_, f64>,
    link: LinkSelector<'_>,
    level: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
    if !(level > 0.0 && level < 1.0) {
        return Err(format!("interval level must lie in (0, 1); got {level}"));
    }
    let alpha = (1.0 - level) / 2.0;
    let n_draws = eta.nrows();
    let n_rows = eta.ncols();
    if n_draws == 0 {
        return Err("posterior bands unavailable: zero draws".to_string());
    }
    let mut eta_mean = vec![0.0_f64; n_rows];
    let mut eta_lower = vec![0.0_f64; n_rows];
    let mut eta_upper = vec![0.0_f64; n_rows];
    let mut response_mean = vec![0.0_f64; n_rows];
    let mut response_lower = vec![0.0_f64; n_rows];
    let mut response_upper = vec![0.0_f64; n_rows];
    let mut column = vec![0.0_f64; n_draws];
    let inv_n = 1.0 / n_draws as f64;
    for j in 0..n_rows {
        for k in 0..n_draws {
            column[k] = eta[[k, j]];
        }
        let mut sum = 0.0_f64;
        for v in &column {
            sum += *v;
        }
        eta_mean[j] = sum * inv_n;
        // Response-scale posterior mean: average inv-link draws, not
        // inv-link of the eta mean. See doc comment above.
        let response_draws = link.apply(&column)?;
        let mut rsum = 0.0_f64;
        for v in &response_draws {
            rsum += *v;
        }
        response_mean[j] = rsum * inv_n;
        let mut response_column = response_draws;
        response_column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        response_lower[j] = quantile_from_sorted(&response_column, alpha);
        response_upper[j] = quantile_from_sorted(&response_column, 1.0 - alpha);
        column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        eta_lower[j] = quantile_from_sorted(&column, alpha);
        eta_upper[j] = quantile_from_sorted(&column, 1.0 - alpha);
    }
    Ok((
        eta_mean,
        eta_lower,
        eta_upper,
        response_mean,
        response_lower,
        response_upper,
    ))
}

pub fn posterior_eta_bands(
    eta_flat: Vec<f64>,
    n_draws: usize,
    n_rows: usize,
    family_kind: &str,
    level: f64,
) -> Result<PosteriorPredictBandsPayload, String> {
    posterior_eta_bands_link(
        eta_flat,
        n_draws,
        n_rows,
        LinkSelector::Tag(family_kind),
        level,
    )
}

/// Spec-aware sibling of [`posterior_eta_bands`]. When `link` is a
/// [`LinkSelector::Spec`] the parameterized inverse links are evaluated from
/// their fitted state; the emitted `family_kind` is the link's display name.
pub fn posterior_eta_bands_link(
    eta_flat: Vec<f64>,
    n_draws: usize,
    n_rows: usize,
    link: LinkSelector<'_>,
    level: f64,
) -> Result<PosteriorPredictBandsPayload, String> {
    if eta_flat.len() != n_draws * n_rows {
        return Err(format!(
            "posterior_eta_bands shape mismatch: got {} floats, expected {} * {}",
            eta_flat.len(),
            n_draws,
            n_rows
        ));
    }
    let family_kind = match link {
        LinkSelector::Tag(tag) => tag.to_string(),
        LinkSelector::Spec(spec) => spec.link_function().name().to_string(),
    };
    let eta = Array2::<f64>::from_shape_vec((n_draws, n_rows), eta_flat)
        .map_err(|err| format!("failed to reshape eta matrix: {err}"))?;
    let (eta_mean, eta_lower, eta_upper, mean, mean_lower, mean_upper) =
        eta_bands_from_matrix_link(eta.view(), link, level)?;
    Ok(PosteriorPredictBandsPayload {
        linear_predictor: eta_mean,
        linear_predictor_lower: eta_lower,
        linear_predictor_upper: eta_upper,
        mean,
        mean_lower,
        mean_upper,
        n_rows,
        n_draws,
        model_class: String::new(),
        family_kind,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::quantile::quantile_from_sorted;
    use ndarray::Array2;

    /// Parity / regression test pinning the *single* posterior eta-band
    /// engine to its documented semantics, so future interval-logic fixes
    /// land here and nowhere else.
    ///
    /// Two contracts are asserted against an independent hand computation:
    ///
    ///   1. Link-scale credible bounds are the shared numpy-`linear`
    ///      quantile (`util::quantile::quantile_from_sorted`) of the
    ///      per-row eta draws — *not* a normal-approximation interval.
    ///   2. The response-scale point estimate is `E[g^{-1}(eta)]` (mean of
    ///      inverse-link draws), *not* `g^{-1}(E[eta])`. Under a strictly
    ///      convex inverse link the two differ by Jensen's inequality, and
    ///      this asymmetric column makes that difference observable.
    ///   3. The response-scale bounds are direct quantiles of the inverse-link
    ///      draws, not inverse-link transforms of the eta quantiles.
    #[test]
    fn eta_bands_match_shared_quantile_and_response_mean_semantics() {
        // Column 0: symmetric draws. Column 1: deliberately skewed so the
        // mean of exp(eta) sits strictly above exp(mean(eta)).
        let eta = Array2::from_shape_vec(
            (5, 2),
            vec![
                -2.0, 0.0, //
                -1.0, 0.5, //
                0.0, 1.0, //
                1.0, 1.5, //
                2.0, 4.0, //
            ],
        )
        .expect("shape");
        let level = 0.80; // alpha = 0.10 each tail
        let alpha = (1.0 - level) / 2.0;

        let (eta_mean, eta_lower, eta_upper, mean, mean_lower, mean_upper) =
            eta_bands_from_matrix(eta.view(), "log", level).expect("bands");

        for j in 0..2 {
            let mut col: Vec<f64> = (0..5).map(|k| eta[[k, j]]).collect();
            let inv_n = 1.0 / 5.0;
            let mean_eta: f64 = col.iter().sum::<f64>() * inv_n;
            assert!(
                (eta_mean[j] - mean_eta).abs() < 1e-12,
                "eta mean mismatch col {j}"
            );

            // Response-scale point estimate = mean of inverse-link draws.
            let resp_mean: f64 = col.iter().map(|e| e.exp()).sum::<f64>() * inv_n;
            assert!(
                (mean[j] - resp_mean).abs() < 1e-12,
                "response mean must be E[g^-1(eta)] for col {j}"
            );
            let mut resp_col: Vec<f64> = col.iter().map(|e| e.exp()).collect();
            resp_col.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            let resp_lo = quantile_from_sorted(&resp_col, alpha);
            let resp_hi = quantile_from_sorted(&resp_col, 1.0 - alpha);
            assert!(
                (mean_lower[j] - resp_lo).abs() < 1e-12,
                "response lower band must be response-draw quantile col {j}"
            );
            assert!(
                (mean_upper[j] - resp_hi).abs() < 1e-12,
                "response upper band must be response-draw quantile col {j}"
            );

            col.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            let lo = quantile_from_sorted(&col, alpha);
            let hi = quantile_from_sorted(&col, 1.0 - alpha);
            assert!(
                (eta_lower[j] - lo).abs() < 1e-12,
                "lower band must be shared linear quantile col {j}"
            );
            assert!(
                (eta_upper[j] - hi).abs() < 1e-12,
                "upper band must be shared linear quantile col {j}"
            );
            assert!(
                eta_lower[j] <= eta_mean[j] && eta_mean[j] <= eta_upper[j],
                "eta mean must sit inside nonzero interval col {j}"
            );
        }

        // Jensen gap is real and oriented: for the skewed column the
        // response mean strictly exceeds g^{-1}(mean(eta)).
        let mean_eta_col1: f64 = (0..5).map(|k| eta[[k, 1]]).sum::<f64>() / 5.0;
        assert!(
            mean[1] > mean_eta_col1.exp() + 1e-9,
            "E[exp(eta)] must exceed exp(E[eta]) for the convex link"
        );
        let transformed_eta_lower = eta_lower[1].exp();
        assert!(
            (mean_lower[1] - transformed_eta_lower).abs() > 1e-3,
            "nonlinear response lower bound must not be transform-of-eta-quantile"
        );
    }

    #[test]
    fn empty_draws_reject_posterior_bands() {
        let eta = Array2::<f64>::zeros((0, 3));
        let err =
            eta_bands_from_matrix(eta.view(), "identity", 0.95).expect_err("zero draws must fail");
        assert!(err.contains("zero draws"));
    }

    /// The posterior response-scale band path is a PUBLIC consumer of the log
    /// inverse link and must report the EXACT `exp(η)`, never the solver's
    /// `η.clamp(−700, 700).exp()` conditioning transform (issue #963). Pin the
    /// finite boundary η = 705 where the exact value (≈1.5e306) and the clamped
    /// value (exp(700) ≈ 1.0e304) diverge by ~exp(5) ≈ 148.
    #[test]
    fn response_bands_use_exact_log_inverse_link_not_solver_clamp() {
        // Single degenerate draw at η = 705 so every response-scale summary
        // (point estimate + both band edges) collapses to exp(705) exactly.
        let eta = Array2::from_shape_vec((1, 1), vec![705.0]).expect("shape");
        let (_eta_mean, _eta_lower, _eta_upper, mean, mean_lower, mean_upper) =
            eta_bands_from_matrix(eta.view(), "log", 0.90).expect("bands");

        let exact = 705.0_f64.exp();
        assert!(exact.is_finite(), "exp(705) must be representable in f64");
        let clamped = 700.0_f64.exp();
        for (label, v) in [
            ("mean", mean[0]),
            ("mean_lower", mean_lower[0]),
            ("mean_upper", mean_upper[0]),
        ] {
            assert_eq!(
                v, exact,
                "{label} must be exact exp(705), not the solver-clamped exp(700)"
            );
            assert!(
                v > clamped * 100.0,
                "{label} must exceed the clamped exp(700) by ~exp(5); got {v} vs {clamped}"
            );
        }
    }

    /// Levels outside (0, 1) are rejected rather than silently clamped.
    #[test]
    fn level_must_lie_in_open_unit_interval() {
        let eta = Array2::<f64>::zeros((4, 2));
        assert!(eta_bands_from_matrix(eta.view(), "identity", 0.0).is_err());
        assert!(eta_bands_from_matrix(eta.view(), "identity", 1.0).is_err());
    }

    /// Posterior fitted-mean draws on the RESPONSE scale for a *parameterized*
    /// link (`Sas`) flow through the band engine via [`LinkSelector::Spec`] and
    /// produce finite, correct response-scale summaries — the reachable CPU half
    /// of issue #1133.
    ///
    /// The bare string-tag path cannot represent the SAS skew/tail state and
    /// REFUSES (`apply_inverse_link_vec("sas", …)` errors); the spec path carries
    /// the fitted state through and is exercised end-to-end here:
    ///
    ///   1. every response-scale summary (point mean + both band edges) is
    ///      finite and a valid binomial probability in (0, 1);
    ///   2. the response point estimate equals the per-row posterior MEAN of the
    ///      SAS inverse link applied to the η draws — `E[g⁻¹(η)]`, the same
    ///      `E[g⁻¹]` (not `g⁻¹(E[η])`) contract the `Tag` path honors — verified
    ///      against an independent hand computation through the canonical solver
    ///      evaluator `inverse_link_mu_d1_for_inverse_link`;
    ///   3. the response band edges are direct quantiles of SAS inverse-link
    ///      draws, not SAS inverse-link transforms of eta quantiles.
    #[test]
    fn spec_path_produces_finite_response_bands_for_parameterized_sas_link() {
        use crate::solver::mixture_link::inverse_link_mu_d1_for_inverse_link;
        use crate::types::InverseLink;

        let state = crate::solver::mixture_link::sas_link_state_from_raw(0.7, -0.4)
            .expect("valid SAS link state");
        let link = InverseLink::Sas(state);

        // Two columns: symmetric draws, and a skewed column so the Jensen gap
        // between E[g⁻¹(η)] and g⁻¹(E[η]) is observable.
        let eta = Array2::from_shape_vec(
            (5, 2),
            vec![
                -2.0, -1.0, //
                -1.0, -0.5, //
                0.0, 0.0, //
                1.0, 1.5, //
                2.0, 4.0, //
            ],
        )
        .expect("shape");
        let level = 0.80;
        let alpha = (1.0 - level) / 2.0;

        // The bare string tag has no SAS state and refuses outright.
        assert!(crate::families::inverse_link::apply_inverse_link_vec(&[0.0_f64], "sas").is_err());

        let selector = LinkSelector::Spec(&link);
        let (eta_mean, eta_lower, eta_upper, mean, mean_lower, mean_upper) =
            eta_bands_from_matrix_link(eta.view(), selector, level).expect("spec bands");

        for j in 0..2 {
            // Every response-scale summary is finite and a valid probability.
            for (label, v) in [
                ("mean", mean[j]),
                ("mean_lower", mean_lower[j]),
                ("mean_upper", mean_upper[j]),
            ] {
                assert!(v.is_finite(), "{label} must be finite for col {j}, got {v}");
                assert!(
                    v > 0.0 && v < 1.0,
                    "SAS is a binomial inverse link; {label} must lie in (0, 1), got {v}"
                );
            }

            // Response point estimate = E[g⁻¹(η)] over the column draws, computed
            // independently through the canonical solver evaluator.
            let col: Vec<f64> = (0..5).map(|k| eta[[k, j]]).collect();
            let resp_mean: f64 = col
                .iter()
                .map(|&e| {
                    inverse_link_mu_d1_for_inverse_link(&link, e)
                        .expect("solver jet eval")
                        .0
                })
                .sum::<f64>()
                / 5.0;
            assert!(
                (mean[j] - resp_mean).abs() < 1e-12,
                "response mean must be E[g^-1(eta)] for col {j}: got {} want {}",
                mean[j],
                resp_mean
            );

            // Link-scale band edges are eta quantiles.
            let mut sorted = col.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            let lo_eta = quantile_from_sorted(&sorted, alpha);
            let hi_eta = quantile_from_sorted(&sorted, 1.0 - alpha);
            assert!(
                (eta_lower[j] - lo_eta).abs() < 1e-12 && (eta_upper[j] - hi_eta).abs() < 1e-12,
                "link-scale band edges must be the shared quantiles for col {j}"
            );
            let mut response_sorted: Vec<f64> = col
                .iter()
                .map(|&e| {
                    inverse_link_mu_d1_for_inverse_link(&link, e)
                        .expect("solver jet eval")
                        .0
                })
                .collect();
            response_sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            let lo_mu = quantile_from_sorted(&response_sorted, alpha);
            let hi_mu = quantile_from_sorted(&response_sorted, 1.0 - alpha);
            assert!(
                (mean_lower[j] - lo_mu).abs() < 1e-12 && (mean_upper[j] - hi_mu).abs() < 1e-12,
                "response band edges must be SAS response-draw quantiles col {j}"
            );
            // Link-scale mean is the plain average of the draws.
            let mean_eta: f64 = col.iter().sum::<f64>() / 5.0;
            assert!((eta_mean[j] - mean_eta).abs() < 1e-12, "eta mean col {j}");
        }

        // The emitted family_kind on the high-level entry reflects the SAS link
        // (`posterior_eta_bands_link` derives it from the spec's display name).
        let eta_flat: Vec<f64> = (0..5).flat_map(|k| [eta[[k, 0]], eta[[k, 1]]]).collect();
        let payload = posterior_eta_bands_link(eta_flat, 5, 2, LinkSelector::Spec(&link), level)
            .expect("payload");
        assert_eq!(payload.n_rows, 2);
        assert_eq!(payload.family_kind, link.link_function().name());
        assert!(
            payload
                .mean
                .iter()
                .all(|v| v.is_finite() && *v > 0.0 && *v < 1.0)
        );
    }
}

//! Deterministic Pólya–Gamma gate-block evidence for logit SAE gates (#1016).
//!
//! The gate/assignment-logit block is the weakest Gaussian piece of the SAE
//! evidence: a Laplace approximation there replaces a skew logistic posterior
//! with a single quadratic, and near a birth event (gate logits ≈ 0) the
//! logistic block is *least* Gaussian, so the `K` vs `K+1` Occam comparison is
//! mispriced on both sides. The PG augmentation makes the gate block Gaussian
//! conditional on independent augmentation variables `ω_i`. This module uses
//! the exact first two moments of each `PG(b_i, ψ_i)` law and a deterministic
//! second-order cumulant expansion around `ω̄ = E[ω]`; the neglected error is
//! the third- and higher-order joint cumulant contribution. The result is a
//! deterministic approximate likelihood correction, not an exact marginal.
//!
//! ## The conditional-Gaussian block
//!
//! For a gate block with design `X_g` (n × d_g), shape vector `b`, binomial
//! responses `y`, offset `o`, and `κ = y − b/2`, the negative log integrand
//! conditional on `ω` is, in the gate coordinates `g`,
//!
//! ```text
//! F_ω(g) = c_ω + ½ gᵀ Q_ω g − h_ωᵀ g
//! Q_ω    = H_rest,gg + S_g + X_gᵀ Ω X_g           (Ω = diag(ω))
//! h_ω    = h_rest,g + X_gᵀ (κ − Ω o)
//! ```
//!
//! so the Gaussian integral is closed:
//!
//! ```text
//! −log ∫ exp(−F_ω(g)) dg
//!   = c_ω − ½ h_ωᵀ Q_ω⁻¹ h_ω + ½ log|Q_ω| − ½ d_g log(2π).
//! ```
//!
//! The `ω`-independent constant `c_ω` collects the `2^{−b}` PSW prefactor and
//! any `H_rest` / `h_rest` constant; it cancels in every consumer that uses the
//! gate evidence as a *correction* (the difference between the PG block and the
//! plain Laplace gate block), so we drop it and document that the returned
//! value is the gate block up to that fixed additive constant.
//!
//! The marginal over independent `ω_i` is approximated by expanding
//! `log E[exp(-V(ω))]` around the moment-matched point:
//!
//! ```text
//! log E[exp(-V(ω))]
//!   = -V(ω̄) + ½ Σ_i Var(ω_i) · ((∂_i V)^2 - ∂_{ii} V)
//!     + third- and higher-order cumulants.
//! ```
//!
//! where `V(ω) = ½ log|Q_ω| − ½ h_ωᵀ Q_ω⁻¹ h_ω` is the `ω`-dependent part of
//! `F_ω` after the Gaussian integral.

use crate::inference::pg_moments::pg_moments;
use faer::Side;
use gam_linalg::faer_ndarray::{FaerArrayView, factorize_symmetricwith_fallback};
use gam_linalg::matrix::FactorizedSystem;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// The data of one logit gate block to be evidence-integrated.
///
/// All matrices are in the *gate coordinates* `g` (dimension `d_g`). The
/// `h_rest` / `hess_rest` blocks carry whatever the surrounding arrow Schur
/// system contributes to the gate coordinates from the rest of the model
/// (decoder/coordinate cross-terms already Schur-folded in); pass zeros when the
/// gate block is isolated.
pub struct GateBlock<'a> {
    /// Gate design `X_g`, shape (n, d_g): row `i` is `x_i` with `ψ_i = x_iᵀγ + o_i`.
    pub design: ArrayView2<'a, f64>,
    /// Binomial responses `y_i` (counts; `0..=b_i`).
    pub y: ArrayView1<'a, f64>,
    /// Binomial shapes `b_i` (`1.0` for Bernoulli).
    pub b: ArrayView1<'a, f64>,
    /// Per-row offset `o_i` (the fixed part of the gate logit). Empty ⇒ zeros.
    pub offset: Option<ArrayView1<'a, f64>>,
    /// Current gate linear predictor `ψ̂_i` used to tilt the PG law (the inner
    /// optimum's logits). Empty ⇒ the untilted `PG(b, 0)` rule.
    pub psi_hat: Option<ArrayView1<'a, f64>>,
    /// Penalty `S_g` on the gate coordinates (d_g × d_g, SPD-or-PSD). Empty ⇒ zero.
    pub penalty: Option<ArrayView2<'a, f64>>,
    /// Rest-of-model Hessian contribution to the gate coordinates `H_rest,gg`
    /// (d_g × d_g). Empty ⇒ zero.
    pub hess_rest: Option<ArrayView2<'a, f64>>,
    /// Rest-of-model linear contribution `h_rest,g` (length d_g). Empty ⇒ zero.
    pub h_rest: Option<ArrayView1<'a, f64>>,
}

/// Which deterministic lane priced the gate block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PgGateLane {
    /// Deterministic second-order independent-row PG correction around `E[ω]`.
    CurvatureCorrected,
    /// Single moment-matched node `ω = E[PG]`: deterministic but only
    /// first-order in the `ω` integral. The cheap debug comparator.
    MomentMatched,
}

/// The PG-corrected gate-block evidence.
#[derive(Clone, Debug)]
pub struct PgGateEvidence {
    /// `−log p(y | rest)` for the gate block, up to the fixed additive constant
    /// `c_ω` documented in the module header (drops out of every correction).
    pub neg_log_evidence: f64,
    /// The lane that produced it.
    pub lane: PgGateLane,
}

/// Compute the deterministic second-order PG gate-block evidence correction.
pub fn pg_gate_evidence(block: &GateBlock<'_>) -> Result<PgGateEvidence, String> {
    evaluate(block, Lane::CurvatureCorrected)
}

/// The deterministic moment-matched comparator: `ω = E[PG(b, ψ̂)]`, one node.
///
/// Labelled [`PgGateLane::MomentMatched`]; this is the zeroth-order point of the
/// independent-row expansion.
pub fn pg_gate_evidence_moment_matched(block: &GateBlock<'_>) -> Result<PgGateEvidence, String> {
    evaluate(block, Lane::MomentMatched)
}

enum Lane {
    CurvatureCorrected,
    MomentMatched,
}

fn evaluate(block: &GateBlock<'_>, lane: Lane) -> Result<PgGateEvidence, String> {
    let n = block.design.nrows();
    let d_g = block.design.ncols();
    if d_g == 0 {
        return Err("PG gate evidence requires a non-empty gate design".into());
    }
    if block.y.len() != n || block.b.len() != n {
        return Err("PG gate evidence: y/b length must match design rows".into());
    }
    let psi_hat = block.psi_hat;
    if let Some(offset) = block.offset {
        if offset.len() != n {
            return Err("PG gate evidence: offset length must match design rows".into());
        }
    }
    if let Some(psi) = psi_hat {
        if psi.len() != n {
            return Err("PG gate evidence: psi_hat length must match design rows".into());
        }
    }
    if let Some(penalty) = block.penalty {
        if penalty.nrows() != d_g || penalty.ncols() != d_g {
            return Err("PG gate evidence: penalty shape must match gate dimension".into());
        }
    }
    if let Some(hess_rest) = block.hess_rest {
        if hess_rest.nrows() != d_g || hess_rest.ncols() != d_g {
            return Err("PG gate evidence: hess_rest shape must match gate dimension".into());
        }
    }
    if let Some(h_rest) = block.h_rest {
        if h_rest.len() != d_g {
            return Err("PG gate evidence: h_rest length must match gate dimension".into());
        }
    }

    // κ = y − b/2.
    let kappa: Array1<f64> = &block.y.to_owned() - &(&block.b.to_owned() * 0.5);

    // Per-row independent PG moments under the tilted law at ψ̂.
    let mut omega_bar = Array1::<f64>::zeros(n);
    let mut omega_var = Array1::<f64>::zeros(n);
    for i in 0..n {
        let c = psi_hat.map(|p| p[i]).unwrap_or(0.0);
        let moments = pg_moments(block.b[i], c);
        omega_bar[i] = moments.mean;
        omega_var[i] = moments.variance;
    }

    // h_const = h_rest,g + X_gᵀ κ  (the ω-independent part of h_ω, minus the
    // ω·o piece handled at evaluation time).
    let xt_kappa = block.design.t().dot(&kappa);
    let h_const = match block.h_rest {
        Some(hr) => &hr.to_owned() + &xt_kappa,
        None => xt_kappa,
    };

    // Assemble the ω-independent base of Q: H_rest,gg + S_g.
    let mut q_base = Array2::<f64>::zeros((d_g, d_g));
    if let Some(hr) = block.hess_rest {
        q_base += &hr;
    }
    if let Some(s) = block.penalty {
        q_base += &s;
    }

    let eval = evaluate_at_omega(block, q_base.view(), h_const.view(), omega_bar.view())?;
    let correction = match lane {
        Lane::CurvatureCorrected => {
            second_order_correction(eval.first.view(), eval.second.view(), omega_var.view())
        }
        Lane::MomentMatched => 0.0,
    };
    let log_two_pi = (2.0 * std::f64::consts::PI).ln();
    let neg_log_evidence = eval.value - 0.5 * d_g as f64 * log_two_pi - 0.5 * correction;
    let lane_tag = match lane {
        Lane::CurvatureCorrected => PgGateLane::CurvatureCorrected,
        Lane::MomentMatched => PgGateLane::MomentMatched,
    };
    Ok(PgGateEvidence {
        neg_log_evidence,
        lane: lane_tag,
    })
}

struct OmegaEvaluation {
    value: f64,
    first: Array1<f64>,
    second: Array1<f64>,
}

fn evaluate_at_omega(
    block: &GateBlock<'_>,
    q_base: ArrayView2<'_, f64>,
    h_const: ArrayView1<'_, f64>,
    omega_diag: ArrayView1<'_, f64>,
) -> Result<OmegaEvaluation, String> {
    let n = block.design.nrows();
    let mut q_mat = q_base.to_owned();
    weighted_gram_into(block.design, omega_diag.view(), &mut q_mat);

    let mut h = h_const.to_owned();
    if let Some(o) = block.offset {
        let omega_o = &omega_diag.to_owned() * &o.to_owned();
        let xt_omega_o = block.design.t().dot(&omega_o);
        h -= &xt_omega_o;
    }

    let q_view = FaerArrayView::new(&q_mat);
    let factor = factorize_symmetricwith_fallback(q_view.as_ref(), Side::Lower)
        .map_err(|e| format!("PG gate block factorization failed: {e:?}"))?;
    let log_det = factor.logdet();
    if !log_det.is_finite() {
        return Err("PG gate block Hessian is not positive definite".into());
    }
    let q_inv_h = FactorizedSystem::solve(&factor, &h)?;
    let quad = h.dot(&q_inv_h);
    let value = 0.5 * log_det - 0.5 * quad;

    let rhs = block.design.t().to_owned();
    let q_inv_xt = FactorizedSystem::solvemulti(&factor, &rhs)?;
    let mut first = Array1::<f64>::zeros(n);
    let mut second = Array1::<f64>::zeros(n);
    for i in 0..n {
        let row = block.design.row(i);
        let solved_x = q_inv_xt.column(i);
        let t = row.dot(&solved_x);
        let w = row.dot(&q_inv_h);
        let offset = block.offset.map(|o| o[i]).unwrap_or(0.0);
        first[i] = 0.5 * t + offset * w + 0.5 * w * w;
        let shifted_w = offset + w;
        second[i] = -0.5 * t * t - t * shifted_w * shifted_w;
    }
    Ok(OmegaEvaluation {
        value,
        first,
        second,
    })
}

fn second_order_correction(
    first: ArrayView1<'_, f64>,
    second: ArrayView1<'_, f64>,
    variance: ArrayView1<'_, f64>,
) -> f64 {
    first
        .iter()
        .zip(second.iter())
        .zip(variance.iter())
        .map(|((&d_v, &d2_v), &var)| var * (d_v * d_v - d2_v))
        .sum()
}

/// Accumulate `Xᵀ diag(w) X` into `out` (d × d), row-streaming so the n × d
/// design is never densely reweighted in place.
fn weighted_gram_into(x: ArrayView2<'_, f64>, w: ArrayView1<'_, f64>, out: &mut Array2<f64>) {
    let d = x.ncols();
    for (row, &wi) in x.rows().into_iter().zip(w.iter()) {
        if wi == 0.0 {
            continue;
        }
        for a in 0..d {
            let xa = row[a] * wi;
            for c in a..d {
                let v = xa * row[c];
                out[[a, c]] += v;
                if c != a {
                    out[[c, a]] += v;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    fn assemble_terms(block: &GateBlock<'_>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let d_g = block.design.ncols();
        let kappa: Array1<f64> = &block.y.to_owned() - &(&block.b.to_owned() * 0.5);
        let xt_kappa = block.design.t().dot(&kappa);
        let h_const = match block.h_rest {
            Some(hr) => &hr.to_owned() + &xt_kappa,
            None => xt_kappa,
        };
        let mut q_base = Array2::<f64>::zeros((d_g, d_g));
        if let Some(hr) = block.hess_rest {
            q_base += &hr;
        }
        if let Some(s) = block.penalty {
            q_base += &s;
        }
        let mut omega_bar = Array1::<f64>::zeros(block.design.nrows());
        for i in 0..block.design.nrows() {
            let c = block.psi_hat.map(|p| p[i]).unwrap_or(0.0);
            omega_bar[i] = pg_moments(block.b[i], c).mean;
        }
        (q_base, h_const, omega_bar)
    }

    #[test]
    fn curvature_correction_zero_when_pg_variances_are_zero() {
        let design = array![[1.0, 0.2], [1.0, -0.5], [1.0, 0.9]];
        let y = Array1::<f64>::zeros(3);
        let b = Array1::<f64>::zeros(3);
        let s = array![[1.5, 0.1], [0.1, 1.2]];
        let h_rest = array![0.3, -0.2];
        let block = GateBlock {
            design: design.view(),
            y: y.view(),
            b: b.view(),
            offset: None,
            psi_hat: None,
            penalty: Some(s.view()),
            hess_rest: None,
            h_rest: Some(h_rest.view()),
        };

        let corrected = pg_gate_evidence(&block).expect("curvature-corrected evidence");
        let matched = pg_gate_evidence_moment_matched(&block).expect("moment-matched evidence");

        assert_eq!(corrected.lane, PgGateLane::CurvatureCorrected);
        assert_eq!(matched.lane, PgGateLane::MomentMatched);
        assert_eq!(
            corrected.neg_log_evidence.to_bits(),
            matched.neg_log_evidence.to_bits()
        );
    }

    /// Determinism: identical inputs produce byte-identical evidence, no RNG.
    #[test]
    fn evidence_is_bit_deterministic() {
        let design = array![[1.0, 0.2], [1.0, -0.5], [1.0, 0.9], [1.0, -0.1]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let b = Array1::<f64>::ones(4);
        let s = Array2::<f64>::eye(2);
        let mk = || GateBlock {
            design: design.view(),
            y: y.view(),
            b: b.view(),
            offset: None,
            psi_hat: None,
            penalty: Some(s.view()),
            hess_rest: None,
            h_rest: None,
        };
        let a = pg_gate_evidence(&mk()).unwrap();
        let c = pg_gate_evidence(&mk()).unwrap();
        assert_eq!(a.neg_log_evidence.to_bits(), c.neg_log_evidence.to_bits());
        assert_eq!(a.lane, c.lane);
    }

    #[test]
    fn derivatives_match_refactorized_finite_differences() {
        let design = array![[1.0, 0.3], [-0.4, 1.2], [0.8, -0.7]];
        let y = array![1.0, 0.0, 1.0];
        let b = array![1.0, 2.0, 1.5];
        let offset = array![0.2, -0.1, 0.4];
        let psi = array![0.1, -0.5, 0.8];
        let penalty = array![[2.0, 0.2], [0.2, 1.5]];
        let hess_rest = array![[0.7, 0.1], [0.1, 0.9]];
        let h_rest = array![0.3, -0.2];
        let block = GateBlock {
            design: design.view(),
            y: y.view(),
            b: b.view(),
            offset: Some(offset.view()),
            psi_hat: Some(psi.view()),
            penalty: Some(penalty.view()),
            hess_rest: Some(hess_rest.view()),
            h_rest: Some(h_rest.view()),
        };
        let (q_base, h_const, omega_bar) = assemble_terms(&block);
        let eval =
            evaluate_at_omega(&block, q_base.view(), h_const.view(), omega_bar.view()).unwrap();
        let eps = 1e-5;
        for i in 0..omega_bar.len() {
            let mut omega_plus = omega_bar.clone();
            let mut omega_minus = omega_bar.clone();
            omega_plus[i] += eps;
            omega_minus[i] -= eps;
            let plus = evaluate_at_omega(&block, q_base.view(), h_const.view(), omega_plus.view())
                .unwrap();
            let minus =
                evaluate_at_omega(&block, q_base.view(), h_const.view(), omega_minus.view())
                    .unwrap();
            let first_fd = (plus.value - minus.value) / (2.0 * eps);
            let second_fd = (plus.value - 2.0 * eval.value + minus.value) / (eps * eps);
            let first_scale = eval.first[i].abs().max(first_fd.abs()).max(1.0);
            let second_scale = eval.second[i].abs().max(second_fd.abs()).max(1.0);
            assert!(
                (eval.first[i] - first_fd).abs() <= 1e-7 * first_scale,
                "row {i}: analytic first {} vs finite difference {first_fd}",
                eval.first[i],
            );
            assert!(
                (eval.second[i] - second_fd).abs() <= 1e-5 * second_scale,
                "row {i}: analytic second {} vs finite difference {second_fd}",
                eval.second[i],
            );
        }
    }

    #[test]
    fn duplicated_row_correction_uses_independent_variances() {
        let design = array![[1.0], [1.0]];
        let y = array![1.0, 1.0];
        let b = array![2.0, 2.0];
        let penalty = array![[2.0]];
        let block = GateBlock {
            design: design.view(),
            y: y.view(),
            b: b.view(),
            offset: None,
            psi_hat: None,
            penalty: Some(penalty.view()),
            hess_rest: None,
            h_rest: None,
        };
        let (q_base, h_const, omega_bar) = assemble_terms(&block);
        let eval =
            evaluate_at_omega(&block, q_base.view(), h_const.view(), omega_bar.view()).unwrap();
        let variance = array![pg_moments(2.0, 0.0).variance, pg_moments(2.0, 0.0).variance];
        let first_row = variance[0] * (eval.first[0] * eval.first[0] - eval.second[0]);
        let second_row = variance[1] * (eval.first[1] * eval.first[1] - eval.second[1]);
        let correction =
            second_order_correction(eval.first.view(), eval.second.view(), variance.view());

        assert!((variance[0] - 1.0 / 12.0).abs() < 1e-15);
        assert!(first_row > 0.0);
        assert!((first_row - second_row).abs() < 1e-15);
        assert!((correction - 2.0 * first_row).abs() < 1e-15);
        assert!((correction - 4.0 * first_row).abs() > first_row);
    }

    #[test]
    fn curvature_correction_changes_moment_matched_near_zero_logit() {
        let n = 4;
        let design = Array2::<f64>::ones((n, 1));
        let y = array![1.0, 0.0, 1.0, 0.0];
        let b = Array1::<f64>::ones(n);
        let s = array![[0.5]];
        let psi = Array1::<f64>::zeros(n);
        let block = GateBlock {
            design: design.view(),
            y: y.view(),
            b: b.view(),
            offset: None,
            psi_hat: Some(psi.view()),
            penalty: Some(s.view()),
            hess_rest: None,
            h_rest: None,
        };
        let corrected = pg_gate_evidence(&block).unwrap();
        let mm = pg_gate_evidence_moment_matched(&block).unwrap();
        let correction = (corrected.neg_log_evidence - mm.neg_log_evidence).abs();
        assert!(
            correction > 1e-6 && correction < 5.0,
            "expected a bounded nonzero PG curvature correction, got {correction}",
        );
    }

    /// #1218: the returned `neg_log_evidence` must equal the documented closed
    /// form `½log|Q| − ½hᵀQ⁻¹h − ½·d_g·log(2π)` in **absolute** value, not merely
    /// up to a relative delta. The pre-fix code added `+½·d_g·log(2π)` instead of
    /// subtracting it, biasing every K-vs-(K+1) gate Occam comparison by
    /// `d_g·log(2π)` per gate coordinate. The existing tests only pin
    /// relative/determinism properties (curvature deltas, FD derivatives,
    /// bit-determinism), so that constant-offset sign error was invisible to them.
    ///
    /// This reconstructs the closed form independently with an explicit 2×2
    /// determinant and inverse (no shared factorization with the module) on a
    /// `psi_hat=None` block where every PG weight is exactly `ω_i = b_i/4` and the
    /// curvature correction is identically zero — so the moment-matched lane
    /// exercises precisely the `value − ½·d_g·log(2π)` assembly. Fails on the old
    /// `+` sign (off by exactly `d_g·log(2π) = 2·log(2π)`), passes on the fix.
    #[test]
    fn moment_matched_evidence_matches_absolute_closed_form() {
        // Concrete 4-row, d_g = 2 gate block from the #1218 repro.
        let design = array![[1.0, 0.5], [1.0, -0.5], [1.0, 1.5], [1.0, -1.0]];
        let y = array![1.0, 0.0, 2.0, 3.0];
        let b = Array1::<f64>::from_elem(4, 3.0);
        let s = array![[1.5, 0.1], [0.1, 1.2]];
        let block = GateBlock {
            design: design.view(),
            y: y.view(),
            b: b.view(),
            offset: None,
            psi_hat: None, // untilted PG(b, 0) ⇒ ω_i = b_i/4 exactly, no curvature.
            penalty: Some(s.view()),
            hess_rest: None,
            h_rest: None,
        };

        // Independent closed form. ω_i = E[PG(3, 0)] = 3/4 = 0.75 (pinned below).
        let omega = pg_moments(3.0, 0.0).mean;
        assert!(
            (omega - 0.75).abs() < 1e-12,
            "PG(3, 0) mean must be b/4 = 0.75, got {omega}",
        );
        let kappa = &y - &(&b * 0.5); // κ = y − b/2.
        // Q = S + Xᵀ Ω X  (Ω = ω·I since every weight is equal).
        let xtx = design.t().dot(&design);
        let q = &s + &(omega * &xtx);
        let h = design.t().dot(&kappa); // h = Xᵀκ (no offset / h_rest).

        // Explicit 2×2 determinant and inverse — fully independent of faer.
        let (q00, q01, q10, q11) = (q[[0, 0]], q[[0, 1]], q[[1, 0]], q[[1, 1]]);
        let det = q00 * q11 - q01 * q10;
        assert!(det > 0.0, "gate Q must be SPD, det = {det}");
        // Q⁻¹ h via the closed 2×2 inverse.
        let inv_h0 = (q11 * h[0] - q01 * h[1]) / det;
        let inv_h1 = (-q10 * h[0] + q00 * h[1]) / det;
        let quad = h[0] * inv_h0 + h[1] * inv_h1; // hᵀQ⁻¹h.
        let log_two_pi = (2.0 * std::f64::consts::PI).ln();
        let d_g = 2.0;
        let want = 0.5 * det.ln() - 0.5 * quad - 0.5 * d_g * log_two_pi;

        let got = pg_gate_evidence_moment_matched(&block)
            .expect("moment-matched gate evidence")
            .neg_log_evidence;

        assert!(
            (got - want).abs() < 1e-10,
            "neg_log_evidence must match the absolute closed form: got {got}, want {want}, \
             gap {} (the pre-fix sign bug gives a gap of d_g·log(2π) = {})",
            got - want,
            d_g * log_two_pi,
        );

        // Guard the sign direction explicitly: the buggy `+` assembly would land
        // exactly `d_g·log(2π)` ABOVE the correct value, so confirm we are not there.
        let buggy = want + d_g * log_two_pi;
        assert!(
            (got - buggy).abs() > 1.0,
            "neg_log_evidence must not match the buggy +½·d_g·log(2π) assembly ({buggy})",
        );
    }
}

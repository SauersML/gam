//! Deterministic Pólya–Gamma gate-block evidence for logit SAE gates (#1016).
//!
//! The gate/assignment-logit block is the weakest Gaussian piece of the SAE
//! evidence: a Laplace approximation there replaces a skew logistic posterior
//! with a single quadratic, and near a birth event (gate logits ≈ 0) the
//! logistic block is *least* Gaussian, so the `K` vs `K+1` Occam comparison is
//! mispriced on both sides. The PG augmentation makes the gate block Gaussian
//! conditional on independent augmentation variables `ω_i`. This module uses
//! the exact first two moments of each `PG(b_i, 0)` law and a deterministic
//! second-order cumulant expansion around `ω̄ = E[ω]`; the neglected error is
//! the third- and higher-order joint cumulant contribution. The result is a
//! deterministic approximation of the gate block's *marginal*: every constant
//! the identity produces is carried, so the returned number is comparable
//! across candidates of different `K`, `m` and gate dimension, not only inside
//! a difference that cancels the constants.
//!
//! ## The law the identity integrates against
//!
//! The PSW identity expands the logit likelihood as an expectation under the
//! *untilted* `PG(b_i, 0)` law:
//!
//! ```text
//! exp(κ_i ψ_i) / (1 + exp ψ_i)^{b_i}
//!   = 2^{−b_i} · E_{ω_i ~ PG(b_i, 0)} exp(κ_i ψ_i − ½ ω_i ψ_i²).
//! ```
//!
//! A tilted law `PG(b, c)` may replace it only together with its
//! Radon–Nikodym factor, `p(ω | b, c) = cosh^b(c/2)·exp(−c²ω/2)·p(ω | b, 0)`,
//! and only at a `c` that is the block's own linear predictor. An external
//! per-row logit is not such a `c` (#3518: the per-row gate logit tilts a block
//! whose model is a single shared coordinate, and moving the expansion point
//! there without the density ratio biased the evidence by ≈ 0.09 nats at
//! `m = 20` on the 1-D fixture), so this module expands under `PG(b, 0)` and
//! takes no tilt.
//!
//! ## The conditional-Gaussian block
//!
//! For a gate block with design `X_g` (n × d_g), shape vector `b`, binomial
//! responses `y`, offset `o`, and `κ = y − b/2`, the negative log integrand
//! conditional on `ω` is, in the gate coordinates `g`,
//!
//! ```text
//! F_ω(g) = ½ gᵀ Q_ω g − h_ωᵀ g + ½ oᵀ Ω o − κᵀ o + Σ_i b_i log 2
//! Q_ω    = H_rest,gg + S_g + X_gᵀ Ω X_g           (Ω = diag(ω))
//! h_ω    = h_rest,g + X_gᵀ (κ − Ω o)
//! ```
//!
//! so the Gaussian integral is closed:
//!
//! ```text
//! −log ∫ exp(−F_ω(g)) dg
//!   = Σ_i b_i log 2 − κᵀ o + V(ω) − ½ d_g log(2π),
//! V(ω) = ½ log|Q_ω| − ½ h_ωᵀ Q_ω⁻¹ h_ω + ½ oᵀ Ω o.
//! ```
//!
//! ## Why the `2π` term is not an Occam term
//!
//! That `−½ d_g log(2π)` belongs to the Gaussian integral over an *unnormalized*
//! kernel `exp(−½ gᵀ S_g g)`. The gate prior is the normalized density
//! `N(0, S_g⁻¹)`, whose normalizer is `½ log|S_g| − ½ d_g log(2π)`. The two
//! `2π` terms cancel exactly, and what survives is `−½ log|S_g|`. Reporting the
//! `2π` term as a per-coordinate, `K`-dependent Occam charge (as the SAE gate
//! consumer used to) prices a constant that the prior normalizer has already
//! paid. `S_g` must therefore be positive definite: an improper gate prior has
//! no marginal, and this module refuses it instead of returning a number.
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
//! so
//!
//! ```text
//! −log p(y | rest)
//!   = Σ_i b_i log 2 − κᵀ o − ½ log|S_g|
//!     + V(ω̄) − ½ Σ_i Var(ω_i) · ((∂_i V)^2 − ∂_{ii} V).
//! ```

use crate::inference::pg_moments::pg_moments;
use faer::Side;
use gam_linalg::faer_ndarray::{FaerArrayView, factorize_symmetricwith_fallback};
use gam_linalg::matrix::FactorizedSystem;
use gam_linalg::triangular::{CholeskyGuard, cholesky_factor_in_place};
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
    /// The gate prior's precision `S_g` (d_g × d_g): the prior is the normalized
    /// density `N(0, S_g⁻¹)`, and its normalizer is part of the returned
    /// marginal. It must be positive definite; an improper gate prior has no
    /// marginal and is refused.
    pub prior_precision: ArrayView2<'a, f64>,
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
    /// `−log p(y | rest)` for the gate block: the marginal itself, with the
    /// `2^{−Σb}` PSW prefactor, the `κᵀo` offset constant and the gate prior's
    /// normalizer all carried. No constant is dropped, so the value is
    /// comparable between blocks of different size and gate dimension. For a
    /// binomial likelihood every factor is `≤ 1` and the prior integrates to
    /// one, so `−log p(y | rest) ≥ 0` up to the expansion's truncation error.
    pub neg_log_evidence: f64,
    /// The lane that produced it.
    pub lane: PgGateLane,
}

/// Compute the deterministic second-order PG gate-block evidence correction.
pub fn pg_gate_evidence(block: &GateBlock<'_>) -> Result<PgGateEvidence, String> {
    evaluate(block, Lane::CurvatureCorrected)
}

enum Lane {
    CurvatureCorrected,
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
    if let Some(offset) = block.offset {
        if offset.len() != n {
            return Err("PG gate evidence: offset length must match design rows".into());
        }
    }
    if block.prior_precision.nrows() != d_g || block.prior_precision.ncols() != d_g {
        return Err("PG gate evidence: prior precision shape must match gate dimension".into());
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

    // Per-row independent PG moments under the law the PSW identity integrates
    // against, `PG(b_i, 0)`: mean `b_i/4`, variance `b_i/24`. Expanding under a
    // tilted `PG(b_i, c)` without its Radon–Nikodym factor would change both
    // `V(ω̄)` and `∂_i V` (#3518).
    let mut omega_bar = Array1::<f64>::zeros(n);
    let mut omega_var = Array1::<f64>::zeros(n);
    for i in 0..n {
        let moments = pg_moments(block.b[i], 0.0);
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
    q_base += &block.prior_precision;

    // The prior normalizer. `∫ exp(−F_ω) dg` over the unnormalized kernel
    // carries `+½ d_g log(2π)`; the normalized prior `N(0, S_g⁻¹)` carries
    // `½ log|S_g| − ½ d_g log(2π)`. The `2π` terms cancel and `−½ log|S_g|`
    // survives, so neither is a `d_g`-dependent Occam charge. A non-PD `S_g` is
    // an improper prior with no marginal, and the factorization refuses it.
    let prior_factor = cholesky_factor_in_place(block.prior_precision, CholeskyGuard::FiniteStrict)
        .ok_or_else(|| {
            "PG gate evidence: the gate prior precision is not finite and positive definite, so \
             the prior is improper and the gate block has no marginal"
                .to_string()
        })?;
    let half_prior_log_det: f64 = (0..d_g).map(|i| prior_factor[[i, i]].ln()).sum();

    // `Σ_i b_i log 2`: the PSW prefactor `2^{−Σ b_i}`, which scales with the
    // block's row count and never cancels between candidates of different size.
    let log_two = std::f64::consts::LN_2;
    let psw_prefactor: f64 = block.b.iter().map(|&bi| bi * log_two).sum();
    // `−κᵀo`: the offset's ω-free contribution to the exponent.
    let offset_linear = match block.offset {
        Some(o) => kappa.dot(&o),
        None => 0.0,
    };

    let eval = evaluate_at_omega(block, q_base.view(), h_const.view(), omega_bar.view())?;
    let correction = match lane {
        Lane::CurvatureCorrected => {
            second_order_correction(eval.first.view(), eval.second.view(), omega_var.view())
        }
    };
    let neg_log_evidence =
        psw_prefactor - offset_linear - half_prior_log_det + eval.value - 0.5 * correction;
    let lane_tag = match lane {
        Lane::CurvatureCorrected => PgGateLane::CurvatureCorrected,
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
    // `½ oᵀ Ω o` is the offset's own ω-dependent term: the PSW exponent carries
    // `−½ ω_i (x_iᵀg + o_i)²`, whose purely-offset part leaves the Gaussian
    // integral untouched but stays a function of ω, so it belongs to `V`.
    let offset_quadratic = match block.offset {
        Some(o) => {
            0.5 * omega_diag
                .iter()
                .zip(o.iter())
                .map(|(&w, &oi)| w * oi * oi)
                .sum::<f64>()
        }
        None => 0.0,
    };
    let value = 0.5 * log_det - 0.5 * quad + offset_quadratic;

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
        // `∂_i V`, including the `½ ω_i o_i²` term's `½ o_i²`; its second
        // derivative in `ω_i` is zero, so `second` is unchanged.
        first[i] = 0.5 * t + offset * w + 0.5 * w * w + 0.5 * offset * offset;
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
        q_base += &block.prior_precision;
        let mut omega_bar = Array1::<f64>::zeros(block.design.nrows());
        for i in 0..block.design.nrows() {
            omega_bar[i] = pg_moments(block.b[i], 0.0).mean;
        }
        (q_base, h_const, omega_bar)
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
            prior_precision: s.view(),
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
        let prior = array![[2.0, 0.2], [0.2, 1.5]];
        let hess_rest = array![[0.7, 0.1], [0.1, 0.9]];
        let h_rest = array![0.3, -0.2];
        let block = GateBlock {
            design: design.view(),
            y: y.view(),
            b: b.view(),
            offset: Some(offset.view()),
            prior_precision: prior.view(),
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
            println!(
                "[#3518] row {i}: first {} vs fd {first_fd}, second {} vs fd {second_fd}",
                eval.first[i], eval.second[i]
            );
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
        let prior = array![[2.0]];
        let block = GateBlock {
            design: design.view(),
            y: y.view(),
            b: b.view(),
            offset: None,
            prior_precision: prior.view(),
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

    /// `log ∫ σ(γ)^{n₁} (1−σ(γ))^{n₀} N(γ; 0, 1) dγ` by the trapezoid rule on
    /// `[-12, 12]`.
    ///
    /// The integrand is analytic and decays like the standard normal density,
    /// so the trapezoid rule is spectrally accurate on a symmetric interval
    /// (Euler–Maclaurin: every boundary derivative is below `φ(12) ≈ 1e−32`),
    /// and the truncated tails carry less than `1e−32` of the mass. With
    /// `h = 1e−4` the reference is exact to well past the `1e−9` scale of every
    /// bar below.
    fn exact_scalar_gate_log_evidence(n_one: f64, n_zero: f64) -> f64 {
        let half_width = 12.0_f64;
        let nodes = 240_001_usize;
        let step = 2.0 * half_width / (nodes - 1) as f64;
        let norm = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        let mut integral = 0.0_f64;
        for node in 0..nodes {
            let gamma = -half_width + step * node as f64;
            let log_sigma = -(1.0 + (-gamma).exp()).ln();
            let log_one_minus = -(1.0 + gamma.exp()).ln();
            let density = norm * (-0.5 * gamma * gamma).exp();
            let weight = if node == 0 || node == nodes - 1 {
                0.5
            } else {
                1.0
            };
            integral +=
                weight * step * density * (n_one * log_sigma + n_zero * log_one_minus).exp();
        }
        integral.ln()
    }

    /// The first neglected order of the module's cumulant expansion on the
    /// scalar gate fixture (`X_g = 1`, `b = 1`, `S_g = 1`, no offset).
    ///
    /// There `Q_ω = 1 + W` with `W = Σ ω_i` and `h = κ_tot` is constant, so
    /// `V` is the scalar function
    /// `V(W) = ½ log(1+W) − ½ κ_tot²/(1+W)` with
    ///
    /// ```text
    /// V'   =  ½(1+W)⁻¹ + ½κ²(1+W)⁻²
    /// V''  = −½(1+W)⁻² −  κ²(1+W)⁻³
    /// V''' =   (1+W)⁻³ + 3κ²(1+W)⁻⁴
    /// ```
    ///
    /// and `∂_i V = V'`, `∂_{ii} V = V''`, `∂_{iii} V = V'''` for every row.
    /// The expansion of `log E[e^{−V}]` truncated after the variance term
    /// leaves a third-cumulant term of size
    /// `⅙ Σ_i κ₃(b_i) · |V''' + 3V'V'' + (V')³|` evaluated at `ω̄`, where
    /// `κ₃(b) = b/60` is the third cumulant of `PG(b, 0)` (its cumulant
    /// generating function is `K(s) = bs/4 + bs²/48 + bs³/360 + …`, whose
    /// second cumulant `b/24` is the variance this module already uses).
    fn third_cumulant_scale(m: usize, kappa_total: f64) -> f64 {
        let w = 1.0 + 0.25 * m as f64;
        let k2 = kappa_total * kappa_total;
        let first = 0.5 / w + 0.5 * k2 / (w * w);
        let second = -0.5 / (w * w) - k2 / (w * w * w);
        let third = 1.0 / (w * w * w) + 3.0 * k2 / (w * w * w * w);
        let cumulant_three = m as f64 / 60.0;
        cumulant_three * (third.abs() + 3.0 * first.abs() * second.abs() + first.abs().powi(3))
            / 6.0
    }

    /// #3518: the returned value is the gate block's marginal in absolute
    /// terms, not up to a constant.
    ///
    /// The fixture is the one the SAE consumer builds: one gate coordinate,
    /// `X_g = 1`, `b = 1`, a unit gate prior and alternating responses. The
    /// reference is the exact one-dimensional integral. Before this test the
    /// module dropped `Σ b_i log 2` and mistook the Gaussian integral's `2π`
    /// term for an Occam charge, which made the *log*-evidence positive — an
    /// impossibility for a binomial likelihood under a normalized prior, since
    /// every factor is at most one, so `∫ L π ≤ ∫ π = 1`.
    #[test]
    fn gate_log_evidence_matches_one_dimensional_quadrature() {
        for m in [1_usize, 5, 20] {
            let design = Array2::<f64>::ones((m, 1));
            let b = Array1::<f64>::ones(m);
            let prior = Array2::<f64>::eye(1);
            let y = Array1::<f64>::from_shape_fn(m, |i| if i % 2 == 0 { 1.0 } else { 0.0 });
            let n_one = y.sum();
            let n_zero = m as f64 - n_one;
            let kappa_total = n_one - 0.5 * m as f64;
            let block = GateBlock {
                design: design.view(),
                y: y.view(),
                b: b.view(),
                offset: None,
                prior_precision: prior.view(),
                hess_rest: None,
                h_rest: None,
            };
            let module = -pg_gate_evidence(&block)
                .expect("the unit gate block has a marginal")
                .neg_log_evidence;
            let exact = exact_scalar_gate_log_evidence(n_one, n_zero);
            // The truncation of the cumulant expansion, with a factor two for
            // the fourth and higher orders: the third term is the first
            // neglected one and the series is in the PG cumulants, which fall
            // by more than an order per step here (`κ₃/κ₂ = 2/5`, and each
            // order carries another derivative of `V`, all of which are below
            // one at `ω̄`).
            let bar = 2.0 * third_cumulant_scale(m, kappa_total);
            let residual = (module - exact).abs();
            println!(
                "[#3518] m={m}: module {module:.9} exact {exact:.9} residual {residual:.3e} \
                 bar {bar:.3e}"
            );
            assert!(
                module <= 0.0,
                "m={m}: a binomial gate block's log-evidence cannot be positive: {module}"
            );
            assert!(
                exact <= 0.0,
                "m={m}: the quadrature reference is not a log-probability: {exact}"
            );
            assert!(
                residual <= bar,
                "m={m}: |module − exact| = {residual:.6e} exceeds the expansion's first \
                 neglected order {bar:.6e}"
            );
            // The bar is far below the constants this test exists to refuse:
            // the dropped PSW prefactor alone was `m·log 2`.
            let dropped = m as f64 * std::f64::consts::LN_2;
            assert!(
                bar < 0.05 * dropped,
                "m={m}: the bar {bar:.6e} is not small against the dropped constant {dropped:.6e}"
            );
        }
    }

    /// #3518: with the gate prior tightened, the marginal is the likelihood at
    /// `g = 0`, which for a zero offset is exactly the PSW prefactor `2^{−Σb}`.
    ///
    /// As `S_g = s·M` grows, `Q_ω = s·M + X_gᵀΩX_g` and `½log|Q_ω| − ½log|S_g|
    /// = ½log|I + M⁻¹X_gᵀΩX_g/s| = O(1/s)`, the quadratic term is `O(1/s)` and
    /// the variance correction is `O(1/s²)`. So the whole residual against
    /// `Σ b_i log 2` falls like `1/s` — but only if both the prefactor and the
    /// prior normalizer `−½log|S_g|` are carried. Omitting the prefactor leaves
    /// `m·log 2`; omitting the normalizer leaves `½log|S_g|`, which grows
    /// without bound in `s`.
    #[test]
    fn a_tight_gate_prior_leaves_only_the_psw_prefactor() {
        let m = 5_usize;
        let y = Array1::<f64>::from_shape_fn(m, |i| if i % 2 == 0 { 1.0 } else { 0.0 });
        let b = Array1::<f64>::ones(m);
        let psw = m as f64 * std::f64::consts::LN_2;
        let shape = array![[4.0, 1.0], [1.0, 3.0]];
        for scale in [1.0e6_f64, 1.0e10] {
            for columns in 1..=2usize {
                let design = Array2::<f64>::from_shape_fn((m, columns), |(row, column)| {
                    1.0 - 0.3 * column as f64 + 0.1 * row as f64
                });
                let prior = Array2::<f64>::from_shape_fn((columns, columns), |(i, j)| {
                    scale * shape[[i, j]]
                });
                let block = GateBlock {
                    design: design.view(),
                    y: y.view(),
                    b: b.view(),
                    offset: None,
                    prior_precision: prior.view(),
                    hess_rest: None,
                    h_rest: None,
                };
                let value = pg_gate_evidence(&block)
                    .expect("a tight gate prior is proper")
                    .neg_log_evidence;
                let residual = (value - psw).abs();
                // The residual is
                // `½log|I + S⁻¹XᵀΩ̄X| − ½hᵀQ⁻¹h − ½·corr`, and
                // `log|I + B| ≤ tr B` for a positive-semidefinite `B`, so with
                // `λ` a Gershgorin lower bound on `λ_min(M)` each piece is below
                // `¼‖X‖_F²/(λs)`, `‖Xᵀκ‖²/(λs)` and `m/(λs)` respectively.
                let lambda = (0..columns)
                    .map(|i| {
                        (0..columns).fold(shape[[i, i]], |acc, j| {
                            if i == j {
                                acc
                            } else {
                                acc - shape[[i, j]].abs()
                            }
                        })
                    })
                    .fold(f64::INFINITY, f64::min);
                let row_norm = design.iter().map(|v| v * v).sum::<f64>();
                let kappa: Array1<f64> = &y - &(&b * 0.5);
                let h_norm_sq = design.t().dot(&kappa).iter().map(|v| v * v).sum::<f64>();
                let bar = (0.25 * row_norm + h_norm_sq + m as f64) / (lambda * scale);
                println!(
                    "[#3518] scale {scale:.1e} columns {columns}: −log p = {value:.12} psw \
                     {psw:.12} residual {residual:.3e} bar {bar:.3e}"
                );
                assert!(
                    residual <= bar,
                    "scale {scale:.1e}, {columns} gate coordinates: residual {residual:.6e} \
                     against the PSW prefactor exceeds {bar:.6e}"
                );
                assert!(
                    bar < 0.01 * psw,
                    "the bar {bar:.6e} does not refuse the dropped prefactor {psw:.6e}"
                );
            }
        }
    }

    /// #3518: an improper gate prior has no marginal, so it is refused by type
    /// rather than priced. The positive control is the same block with a proper
    /// prior, which is priced.
    #[test]
    fn an_improper_gate_prior_is_refused() {
        let design = array![[1.0, 0.2], [1.0, -0.5], [1.0, 0.9]];
        let y = array![1.0, 0.0, 1.0];
        let b = Array1::<f64>::ones(3);
        let mk = |prior: &Array2<f64>| {
            pg_gate_evidence(&GateBlock {
                design: design.view(),
                y: y.view(),
                b: b.view(),
                offset: None,
                prior_precision: prior.view(),
                hess_rest: None,
                h_rest: None,
            })
        };
        let flat = Array2::<f64>::zeros((2, 2));
        let indefinite = array![[1.0, 2.0], [2.0, 1.0]];
        let proper = Array2::<f64>::eye(2);
        for improper in [&flat, &indefinite] {
            let refusal = mk(improper).expect_err("an improper gate prior has no marginal");
            println!("[#3518] improper prior refused: {refusal}");
            assert!(
                refusal.contains("improper"),
                "the refusal must name the improper prior: {refusal}"
            );
        }
        let priced = mk(&proper).expect("a unit gate prior is proper");
        println!("[#3518] proper prior priced: {}", priced.neg_log_evidence);
        assert!(priced.neg_log_evidence >= 0.0);
    }
}

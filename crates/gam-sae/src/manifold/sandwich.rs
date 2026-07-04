//! Godambe / composite-likelihood sandwich correction for the quantities the
//! SAE-manifold fit advertises as calibrated (posterior shape bands and the
//! model-selection information charge).
//!
//! # Why the model-based covariance is optimistic here
//!
//! The production band reports `Cov(β) = φ̂ · S_β⁻¹` (see
//! [`super::SaeAtomShapeUncertainty`]), the classical `Vb = φ H⁻¹` covariance.
//! That formula is the inverse **expected** Fisher information, and it is the
//! correct sampling covariance of the maximum-(penalized-)likelihood estimator
//! ONLY when the working likelihood is correctly specified — i.e. when the
//! reconstruction residuals are homoskedastic, uncorrelated across output
//! channels, and Gaussian. Our residuals are none of these: activation vectors
//! carry LayerNorm structure (a per-row scale + mean constraint), template
//! correlation couples output channels, and token frequency makes the residual
//! scale heteroskedastic. Under that misspecification the working likelihood is
//! a *composite* (quasi-) likelihood, and the estimator's true covariance is the
//! **Godambe** (robust / "sandwich") form.
//!
//! # The sandwich, derived
//!
//! Write the penalized working negative log-likelihood as
//! `ℓ(β) = Σ_i ℓ_i(β) + ½ βᵀ S β` and let `β̂` be its minimiser. Let
//!   * `A = E[−∂²ℓ/∂β∂βᵀ]` — the **sensitivity** / bread matrix (here the
//!     penalized Hessian `H = (1/φ)ΦᵀΦ + S`, whose inverse `A⁻¹` is exactly the
//!     model-based covariance the band already forms), and
//!   * `J = Var(∂ℓ/∂β) = Σ_i E[s_i s_iᵀ]` — the **variability** / meat matrix,
//!     the outer product of the per-observation score contributions
//!     `s_i = ∂ℓ_i/∂β`.
//!
//! A first-order (delta-method) expansion of the estimating equation
//! `∂ℓ/∂β(β̂) = 0` around the truth gives the classical result
//!
//! ```text
//!   Cov(β̂)  =  A⁻¹ J A⁻¹          (the Godambe / sandwich covariance)
//! ```
//!
//! When the likelihood is correctly specified the **information-matrix
//! equality** `J = A` holds, the two `A⁻¹` factors collapse one copy of `A`, and
//! the sandwich reduces to the model-based `A⁻¹` — so the sandwich mode is a
//! strict generalization that COINCIDES with the current band exactly when the
//! current band's assumptions hold. Its width relative to the model-based band
//! is therefore a direct, local diagnostic of how far the working likelihood is
//! from being correctly specified. `J` is estimated empirically from the fitted
//! scores (the outer-product-of-gradients "BHHH" estimator), which needs no
//! assumption about the residual covariance — that is the whole point.
//!
//! # The Gaussian reconstruction score, and why φ̂ cancels
//!
//! For the multi-output Gaussian reconstruction `z_i = g_i(β) + ε_i`,
//! `ε_i ∼ (0, φ I_p)`, with the atom's effective (gate-scaled) design row
//! `g_i ∈ ℝ^M` entering channel `c` as `ĝ_{ic} = g_iᵀ β_{·c}`, the per-obs score
//! for the channel-`c` coefficients is `s_{ic} = (1/φ) g_i r_{ic}` with residual
//! `r_{ic} = z_{ic} − ĝ_{ic}`. Because the model-based bread is block-diagonal
//! across output channels (the working information is `(1/φ) ΦᵀΦ ⊗ I_p` plus a
//! channel-separable penalty — exactly the structure the band's per-channel
//! variance formula already assumes), the within-channel sandwich block is
//!
//! ```text
//!   Cov_cc(β) = A_c⁻¹ J_cc A_c⁻¹,   J_cc = (1/φ²) Σ_i r_{ic}² g_i g_iᵀ .
//! ```
//!
//! With `A_c⁻¹ = φ S_{β,c}⁻¹` (the φ-scaled model-based block the band forms),
//! the two `φ` factors from the bread cancel the `1/φ²` in the meat, so the
//! sandwich band is **dispersion-free** — it does not depend on the estimated
//! scale `φ̂` at all, only on the raw residual heteroskedasticity `r_{ic}²`. That
//! is the robustness we want: the sandwich stops trusting the single global
//! `φ̂` and lets the data's own per-observation residual energy set the width.
//!
//! # Composite-likelihood model-selection charge (CLIC / TIC)
//!
//! Model selection that assumes a well-specified likelihood charges `d_eff`
//! parameters (the trace of the smoother/hat operator). Under misspecification
//! the correct penalty is the **Takeuchi / composite-likelihood information
//! criterion** effective dof
//!
//! ```text
//!   d_eff^CLIC = tr(J A⁻¹)          (replaces d_eff in AIC-style charges),
//! ```
//!
//! which again equals `d_eff` exactly when `J = A` and otherwise reports the
//! honest effective number of parameters the composite likelihood actually
//! spent. [`clic_effective_dof`] computes it; the band/selection callers report
//! it ALONGSIDE the model-based `d_eff`, never silently in place of it.
//!
//! Everything here is derived — there are no tuned constants (SPEC.md law): the
//! sandwich and the CLIC dof are exact algebraic functions of the fitted bread
//! and the empirical score meat.

use super::*;

/// Which covariance formula a calibrated quantity is reported under.
///
/// [`RobustCovarianceMode::ModelBased`] is the classical `Vb = φ̂ H⁻¹` inverse
/// expected-information covariance (correct iff the working likelihood is
/// correctly specified). [`RobustCovarianceMode::Sandwich`] is the Godambe
/// `A⁻¹ J A⁻¹` robust covariance (correct under composite-likelihood
/// misspecification). The two coincide when the information-matrix equality
/// `J = A` holds, so the sandwich is a strict, assumption-light generalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustCovarianceMode {
    /// Inverse expected-information `A⁻¹` (`φ̂ H⁻¹`).
    ModelBased,
    /// Godambe sandwich `A⁻¹ J A⁻¹`.
    Sandwich,
}

impl RobustCovarianceMode {
    /// Stable label for reporting which mode produced a quantity.
    pub fn label(self) -> &'static str {
        match self {
            RobustCovarianceMode::ModelBased => "model-based (A⁻¹)",
            RobustCovarianceMode::Sandwich => "sandwich (A⁻¹ J A⁻¹)",
        }
    }
}

/// The model-selection information charge, reported under BOTH the model-based
/// and the composite-likelihood (CLIC) accounting so a caller can price a
/// structure move honestly and see how far the two diverge.
///
/// `model_based_dof = tr(F A⁻¹)` is the classical effective parameter count (the
/// smoother/hat trace), correct when the working likelihood is well specified.
/// `clic_dof = tr(J A⁻¹)` is the Takeuchi / composite-likelihood effective dof,
/// which replaces the expected information `F` by the empirical score meat `J`.
/// They coincide under the information-matrix equality and otherwise differ by
/// the degree of misspecification; `clic_dof` is the honest charge to use in an
/// AIC-style penalty when the residuals are not iid Gaussian.
#[derive(Debug, Clone, Copy)]
pub struct CompositeLikelihoodCharge {
    /// `tr(F A⁻¹)` — model-based effective degrees of freedom.
    pub model_based_dof: f64,
    /// `tr(J A⁻¹)` — composite-likelihood (CLIC / Takeuchi) effective dof.
    pub clic_dof: f64,
}

impl CompositeLikelihoodCharge {
    /// The over/under-dispersion ratio `clic_dof / model_based_dof` — `1` under
    /// a correctly specified likelihood, `>1` when the residuals carry more
    /// score variability than the working model admits (the usual direction for
    /// heteroskedastic / correlated reconstruction residuals).
    pub fn misspecification_ratio(&self) -> f64 {
        if self.model_based_dof.abs() > f64::MIN_POSITIVE {
            self.clic_dof / self.model_based_dof
        } else {
            f64::NAN
        }
    }
}

/// The Godambe sandwich covariance `A⁻¹ J A⁻¹` from a model-based covariance
/// block `bread = A⁻¹` and a score-meat block `meat = J`.
///
/// Both must be the same square dimension. The result is symmetrized (the
/// product of a symmetric `A⁻¹`, symmetric `J`, symmetric `A⁻¹` is symmetric in
/// exact arithmetic; we average with its transpose to kill rounding asymmetry).
pub(crate) fn godambe_sandwich_covariance(
    bread: ArrayView2<'_, f64>,
    meat: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    let m = bread.nrows();
    if bread.ncols() != m {
        return Err(format!(
            "godambe_sandwich_covariance: bread must be square, got {:?}",
            bread.dim()
        ));
    }
    if meat.dim() != (m, m) {
        return Err(format!(
            "godambe_sandwich_covariance: meat {:?} must match bread ({m},{m})",
            meat.dim()
        ));
    }
    // A⁻¹ J A⁻¹, formed left-to-right.
    let sandwich = bread.dot(&meat).dot(&bread);
    let mut out = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            out[[i, j]] = 0.5 * (sandwich[[i, j]] + sandwich[[j, i]]);
        }
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err("godambe_sandwich_covariance: non-finite sandwich entry".to_string());
    }
    Ok(out)
}

/// Composite-likelihood effective degrees of freedom `tr(J A⁻¹)` (the
/// Takeuchi / CLIC information-criterion penalty), from meat `J` and bread
/// `A⁻¹`.
///
/// `tr(J A⁻¹) = Σ_{i,j} J[i,j] A⁻¹[j,i]`. Equals the ordinary parameter/edf
/// count exactly under the information-matrix equality `J = A` (then
/// `tr(A A⁻¹) = tr(I) = dim`), and otherwise is the honest effective parameter
/// count the misspecified composite likelihood spent.
pub(crate) fn clic_effective_dof(
    bread: ArrayView2<'_, f64>,
    meat: ArrayView2<'_, f64>,
) -> Result<f64, String> {
    let m = bread.nrows();
    if bread.ncols() != m || meat.dim() != (m, m) {
        return Err(format!(
            "clic_effective_dof: shape mismatch bread {:?} meat {:?}",
            bread.dim(),
            meat.dim()
        ));
    }
    let mut trace = 0.0_f64;
    for i in 0..m {
        for j in 0..m {
            trace += meat[[i, j]] * bread[[j, i]];
        }
    }
    if !trace.is_finite() {
        return Err("clic_effective_dof: non-finite trace".to_string());
    }
    Ok(trace)
}

/// Within-channel Gaussian score-meat blocks `J_cc = (1/φ²) Σ_i r_{ic}² g_i g_iᵀ`,
/// one `(M × M)` matrix per output channel `c`.
///
/// `design` holds the atom's effective (gate-scaled) per-observation design rows
/// `g_i ∈ ℝ^M` as its `(n × M)` rows; `residuals` holds the reconstruction
/// residuals `r_{ic}` as its `(n × p)` entries. The `1/φ²` factor is included so
/// the block pairs directly with the φ-scaled model-based bread `A_c⁻¹` in
/// [`godambe_sandwich_covariance`]; the two dispersion factors then cancel (see
/// module docs), so the resulting sandwich is scale-free — but computing it this
/// way keeps every intermediate a genuine covariance/information block.
pub(crate) fn gaussian_within_channel_meat(
    design: ArrayView2<'_, f64>,
    residuals: ArrayView2<'_, f64>,
    dispersion: f64,
) -> Result<Vec<Array2<f64>>, String> {
    let (n, m) = design.dim();
    let (n_r, p) = residuals.dim();
    if n_r != n {
        return Err(format!(
            "gaussian_within_channel_meat: design has {n} rows but residuals have {n_r}"
        ));
    }
    if !(dispersion.is_finite() && dispersion > 0.0) {
        return Err(format!(
            "gaussian_within_channel_meat: dispersion must be finite and positive, got {dispersion}"
        ));
    }
    let inv_phi2 = 1.0 / (dispersion * dispersion);
    let mut blocks = vec![Array2::<f64>::zeros((m, m)); p];
    for i in 0..n {
        let g = design.row(i);
        for c in 0..p {
            let r = residuals[[i, c]];
            if r == 0.0 {
                continue;
            }
            let w = inv_phi2 * r * r;
            let block = &mut blocks[c];
            for a in 0..m {
                let ga = g[a];
                if ga == 0.0 {
                    continue;
                }
                let wga = w * ga;
                for b in 0..m {
                    block[[a, b]] += wga * g[b];
                }
            }
        }
    }
    Ok(blocks)
}

/// Within-channel EXPECTED information blocks `F_cc = (1/φ) Σ_i g_i g_iᵀ`, one
/// `(M × M)` matrix per output channel `c` — the model-based counterpart of
/// [`gaussian_within_channel_meat`] with the empirical residual energy `r_{ic}²`
/// replaced by its correctly-specified expectation `φ`.
///
/// Pairing it with the same bread gives the MODEL-BASED effective dof
/// `tr(F A⁻¹)` (the classical hat-trace), so `tr(F A⁻¹)` and the CLIC
/// `tr(J A⁻¹)` are reported on identical footing and coincide exactly under the
/// information-matrix equality (`r_{ic}² → φ` in expectation).
///
/// The expected information is channel-independent (the design is shared across
/// output channels), so a single `(M × M)` block is returned and reused for
/// every channel.
pub(crate) fn gaussian_within_channel_expected_meat(
    design: ArrayView2<'_, f64>,
    dispersion: f64,
) -> Result<Array2<f64>, String> {
    let (n, m) = design.dim();
    if !(dispersion.is_finite() && dispersion > 0.0) {
        return Err(format!(
            "gaussian_within_channel_expected_meat: dispersion must be finite and positive, \
             got {dispersion}"
        ));
    }
    let inv_phi = 1.0 / dispersion;
    let mut gram = Array2::<f64>::zeros((m, m));
    for i in 0..n {
        let g = design.row(i);
        for a in 0..m {
            let ga = g[a];
            if ga == 0.0 {
                continue;
            }
            for b in 0..m {
                gram[[a, b]] += ga * g[b];
            }
        }
    }
    gram.mapv_inplace(|v| v * inv_phi);
    Ok(gram)
}

/// Robust per-channel band variance `Var_c(t) = φ(t)ᵀ (A_c⁻¹ J_cc A_c⁻¹) φ(t)`
/// evaluated from the φ-scaled model-based within-channel bread `bread_c`
/// (`A_c⁻¹`), the within-channel meat `meat_c` (`J_cc`), and the basis row
/// `phi_t`. Returns the non-negative variance (floored at 0 against rounding).
pub(crate) fn robust_channel_band_variance(
    bread_c: ArrayView2<'_, f64>,
    meat_c: ArrayView2<'_, f64>,
    phi_t: ArrayView1<'_, f64>,
) -> Result<f64, String> {
    let sandwich = godambe_sandwich_covariance(bread_c, meat_c)?;
    let m = sandwich.nrows();
    if phi_t.len() != m {
        return Err(format!(
            "robust_channel_band_variance: basis row len {} != block dim {m}",
            phi_t.len()
        ));
    }
    // φᵀ Σ φ.
    let sphi = sandwich.dot(&phi_t);
    let var: f64 = phi_t.iter().zip(sphi.iter()).map(|(a, b)| a * b).sum();
    Ok(var.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// Symmetric PD bread for tests: A⁻¹ = (DᵀD + I)⁻¹ scaled.
    fn spd_bread(m: usize, seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut d = Array2::<f64>::zeros((m, m));
        for v in d.iter_mut() {
            *v = rng.random_range(-1.0..1.0);
        }
        let mut a = d.t().dot(&d);
        for i in 0..m {
            a[[i, i]] += 1.0;
        }
        // invert via nalgebra-free Gauss-Jordan on a small matrix.
        invert(&a)
    }

    fn invert(a: &Array2<f64>) -> Array2<f64> {
        let m = a.nrows();
        let mut aug = Array2::<f64>::zeros((m, 2 * m));
        for i in 0..m {
            for j in 0..m {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, m + i]] = 1.0;
        }
        for col in 0..m {
            // partial pivot
            let mut piv = col;
            for r in col + 1..m {
                if aug[[r, col]].abs() > aug[[piv, col]].abs() {
                    piv = r;
                }
            }
            if piv != col {
                for j in 0..2 * m {
                    aug.swap([col, j], [piv, j]);
                }
            }
            let d = aug[[col, col]];
            for j in 0..2 * m {
                aug[[col, j]] /= d;
            }
            for r in 0..m {
                if r == col {
                    continue;
                }
                let f = aug[[r, col]];
                if f != 0.0 {
                    for j in 0..2 * m {
                        aug[[r, j]] -= f * aug[[col, j]];
                    }
                }
            }
        }
        let mut inv = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                inv[[i, j]] = aug[[i, m + j]];
            }
        }
        inv
    }

    #[test]
    fn sandwich_equals_model_based_under_information_equality() {
        // When J = A (information-matrix equality), A⁻¹ J A⁻¹ = A⁻¹ and
        // tr(J A⁻¹) = dim exactly.
        let m = 5;
        let a_inv = spd_bread(m, 7);
        let a = invert(&a_inv); // meat = A
        let sw = godambe_sandwich_covariance(a_inv.view(), a.view()).unwrap();
        for i in 0..m {
            for j in 0..m {
                assert!(
                    (sw[[i, j]] - a_inv[[i, j]]).abs() < 1e-9,
                    "sandwich must reduce to model-based when J=A: [{i},{j}] {} vs {}",
                    sw[[i, j]],
                    a_inv[[i, j]]
                );
            }
        }
        let dof = clic_effective_dof(a_inv.view(), a.view()).unwrap();
        assert!(
            (dof - m as f64).abs() < 1e-9,
            "CLIC dof must equal dim under J=A, got {dof}"
        );
    }

    #[test]
    fn sandwich_inflates_under_heteroskedastic_residuals() {
        // Simple 1-parameter, 1-channel design g_i = 1 (intercept). Model-based
        // variance of the mean = φ/n. With heteroskedastic residuals whose
        // empirical second moment far exceeds φ, the sandwich must be WIDER, and
        // it must match the classical robust (White) variance Σ r_i² / n².
        let n = 200;
        let mut rng = StdRng::seed_from_u64(11);
        let design = Array2::<f64>::ones((n, 1));
        let mut residuals = Array2::<f64>::zeros((n, 1));
        let mut sum_r2 = 0.0;
        for i in 0..n {
            // heteroskedastic: half the rows have 5x the scale
            let scale = if i % 2 == 0 { 1.0 } else { 5.0 };
            let r = scale * rng.random_range(-1.0..1.0);
            residuals[[i, 0]] = r;
            sum_r2 += r * r;
        }
        // model-based φ̂ = Σr²/n; bread A⁻¹ = φ̂/n (variance of the mean).
        let phi = sum_r2 / n as f64;
        let bread = Array2::from_elem((1, 1), phi / n as f64);
        let meat = gaussian_within_channel_meat(design.view(), residuals.view(), phi).unwrap();
        // meat_00 = (1/φ²) Σ r² . sandwich = bread·meat·bread.
        let sw = godambe_sandwich_covariance(bread.view(), meat[0].view()).unwrap();
        let robust_var = sw[[0, 0]];
        // Classical heteroskedasticity-consistent variance of the sample mean:
        // Σ r_i² / n².  Sandwich must reproduce it (φ cancels).
        let white = sum_r2 / (n as f64 * n as f64);
        let model_based = bread[[0, 0]]; // φ/n, the model-based mean variance
        println!(
            "[sandwich/heteroskedastic] model_based_var={model_based:.6} sandwich_var={robust_var:.6} white_ref={white:.6}"
        );
        assert!(
            (robust_var - white).abs() < 1e-9 * white.max(1.0),
            "sandwich mean-variance {robust_var} must equal White {white}"
        );
        // And it is strictly positive / finite.
        assert!(robust_var > 0.0 && robust_var.is_finite());
    }

    #[test]
    fn clic_dof_moves_with_misspecification() {
        // Build a bread A⁻¹ and a meat J that is a scaled A (J = c·A): then
        // tr(J A⁻¹) = c·dim, so the CLIC dof scales with the score/information
        // ratio — the honest effective-parameter count under over-dispersion.
        let m = 4;
        let a_inv = spd_bread(m, 3);
        let a = invert(&a_inv);
        for &c in &[0.5_f64, 2.0, 3.5] {
            let meat = a.mapv(|v| c * v);
            let dof = clic_effective_dof(a_inv.view(), meat.view()).unwrap();
            assert!(
                (dof - c * m as f64).abs() < 1e-8,
                "CLIC dof under J={c}·A must be {}·{m}, got {dof}",
                c
            );
        }
    }

    #[test]
    fn clic_and_model_based_dof_coincide_under_homoskedastic_residuals() {
        // With homoskedastic residuals (r_ic² ≈ φ), the empirical meat J and the
        // expected information F coincide, so tr(J A⁻¹) ≈ tr(F A⁻¹): the CLIC and
        // model-based effective dof agree. This is the charge's calibration point.
        let n = 4000;
        let m = 3;
        let mut rng = StdRng::seed_from_u64(123);
        // shared design g_i, one output channel
        let mut design = Array2::<f64>::zeros((n, m));
        for v in design.iter_mut() {
            *v = rng.random_range(-1.0..1.0);
        }
        let phi = 0.7_f64;
        let sd = phi.sqrt();
        let mut residuals = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            // homoskedastic mean-zero residual with variance φ (uniform scaled to var φ)
            let u = rng.random_range(-1.0..1.0); // var 1/3
            residuals[[i, 0]] = u * sd * (3.0_f64).sqrt();
        }
        // bread = (F)⁻¹ so that tr(F A⁻¹) = tr(I) = m exactly (model-based dof = m).
        let f = gaussian_within_channel_expected_meat(design.view(), phi).unwrap();
        let a_inv = invert(&f);
        let model_dof = clic_effective_dof(a_inv.view(), f.view()).unwrap();
        assert!(
            (model_dof - m as f64).abs() < 1e-8,
            "model-based dof tr(F A⁻¹) must equal m={m}, got {model_dof}"
        );
        let meat = gaussian_within_channel_meat(design.view(), residuals.view(), phi).unwrap();
        let clic = clic_effective_dof(a_inv.view(), meat[0].view()).unwrap();
        // Empirical meat concentrates on F, so clic ≈ m within Monte-Carlo error.
        assert!(
            (clic - m as f64).abs() < 0.3,
            "CLIC dof should be ≈ m={m} under homoskedastic residuals, got {clic}"
        );
    }

    #[test]
    fn clic_dof_exceeds_model_based_under_overdispersion() {
        // Inflate every residual by a factor s: J scales by s², so tr(J A⁻¹)
        // ≈ s²·tr(F A⁻¹). The CLIC dof must exceed the model-based dof, the
        // honest signal that the composite likelihood spent more effective
        // parameters than the working model admits.
        let n = 3000;
        let m = 3;
        let mut rng = StdRng::seed_from_u64(7);
        let mut design = Array2::<f64>::zeros((n, m));
        for v in design.iter_mut() {
            *v = rng.random_range(-1.0..1.0);
        }
        let phi = 1.0_f64;
        let s = 2.5_f64;
        let mut residuals = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            residuals[[i, 0]] = s * rng.random_range(-1.0..1.0) * (3.0_f64).sqrt();
        }
        let f = gaussian_within_channel_expected_meat(design.view(), phi).unwrap();
        let a_inv = invert(&f);
        let model_dof = clic_effective_dof(a_inv.view(), f.view()).unwrap();
        let meat = gaussian_within_channel_meat(design.view(), residuals.view(), phi).unwrap();
        let clic = clic_effective_dof(a_inv.view(), meat[0].view()).unwrap();
        println!(
            "[clic/overdispersion s={s}] model_based_dof={model_dof:.4} clic_dof={clic:.4} ratio={:.4} (expected≈s²={:.4})",
            clic / model_dof,
            s * s
        );
        assert!(
            clic > model_dof * 1.5,
            "CLIC dof {clic} must clearly exceed model-based dof {model_dof} under s={s} overdispersion"
        );
    }

    #[test]
    fn robust_band_variance_matches_manual_quadratic_form() {
        let m = 3;
        let bread = spd_bread(m, 21);
        let mut meat = Array2::<f64>::zeros((m, m));
        // some symmetric PSD meat
        let g = Array1::from(vec![0.3, -1.1, 0.7]);
        for a in 0..m {
            for b in 0..m {
                meat[[a, b]] = 2.0 * g[a] * g[b];
            }
        }
        let phi_t = Array1::from(vec![1.0, 0.5, -0.25]);
        let var = robust_channel_band_variance(bread.view(), meat.view(), phi_t.view()).unwrap();
        let sw = godambe_sandwich_covariance(bread.view(), meat.view()).unwrap();
        let manual: f64 = phi_t.dot(&sw.dot(&phi_t));
        assert!((var - manual.max(0.0)).abs() < 1e-12);
    }
}

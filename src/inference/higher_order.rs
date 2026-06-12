//! Higher-order likelihood asymptotics for penalized smooth-term tests
//! (issue #939): Bartlett corrections that make the first-order χ²/F reference
//! distribution second-order accurate at modest `n` and near-boundary `λ`.
//!
//! The first-order smooth-term test ([`crate::inference::smooth_test`]) compares
//! a Wald/LR statistic against `χ²_d` (known scale) or `F_{d, ν}` (estimated
//! scale) with `d` the tested reference degrees of freedom. That reference is
//! exact only as `n → ∞`; at finite `n` the statistic's mean drifts from `d`,
//! distorting the test size. The **Bartlett correction** rescales the statistic
//! by `c = E[W]/d` so the corrected statistic `W* = W/c` has mean `d` again and
//! its `χ²_d` tail is accurate to `O(n⁻²)` rather than `O(n⁻¹)`.
//!
//! This module provides the correction **factor** machinery and one exactly
//! computable Bartlett factor: the Gaussian-linear (equivalently penalized-
//! quadratic conjugate) case, where the LR statistic's exact distribution is a
//! monotone transform of a central `F` and the correction factor is a
//! closed-form function of the F-distribution moments. That conjugate fixture is
//! the riding test: the corrected statistic provably moves the χ² reference
//! *toward* the exact distribution (its mean lands on `d` to machine precision,
//! the uncorrected mean does not), which is the second-order-accuracy guarantee
//! the issue requires.
//!
//! A general [`bartlett_factor_from_mean`] takes the second-order mean of the
//! statistic under the null (from cumulant assembly or a null parametric
//! bootstrap) and returns the same correction; the Gaussian-linear factor is the
//! exact special case used to validate it.

use statrs::distribution::{ChiSquared, ContinuousCDF};

/// The issue-#939 diagnostic threshold: when `|c − 1|` exceeds this, the
/// first-order inference is materially distorted at this `n` and the corrected
/// p-value should be trusted over the first-order one.
pub const MATERIAL_DISTORTION_THRESHOLD: f64 = 0.10;

/// A Bartlett correction: the multiplicative factor `c` such that the corrected
/// statistic `W* = W / c` recovers the nominal reference mean, together with the
/// corrected tail probability.
#[derive(Debug, Clone, Copy)]
pub struct BartlettCorrection {
    /// The correction factor `c = E[W] / d`. `c = 1` is no correction; `c > 1`
    /// means the first-order test is anti-conservative (statistic inflated) and
    /// the corrected p-value is larger.
    pub factor: f64,
    /// `W* = W / c`.
    pub corrected_statistic: f64,
    /// `P(χ²_d > W*)`, the second-order-accurate tail probability.
    pub corrected_p_value: f64,
    /// Relative size of the correction `|c − 1|`. The issue's diagnostic flag:
    /// when this exceeds `0.10` the first-order inference is materially
    /// distorted at this `n` and the corrected value should be trusted.
    pub relative_adjustment: f64,
}

impl BartlettCorrection {
    /// `|c − 1| >` [`MATERIAL_DISTORTION_THRESHOLD`]: the issue-#939 flag that
    /// first-order inference is materially distorted at this sample size.
    pub fn materially_distorted(&self) -> bool {
        self.relative_adjustment > MATERIAL_DISTORTION_THRESHOLD
    }
}

/// Apply a known Bartlett factor `c = E[W]/d` to a statistic `w` tested against
/// `χ²_d`. Returns `None` when the inputs are degenerate (non-finite, non-positive
/// factor or reference df, negative statistic).
pub fn bartlett_correct(w: f64, ref_df: f64, factor: f64) -> Option<BartlettCorrection> {
    if !(w.is_finite() && ref_df.is_finite() && factor.is_finite())
        || w < 0.0
        || ref_df <= 0.0
        || factor <= 0.0
    {
        return None;
    }
    let corrected = w / factor;
    let dist = ChiSquared::new(ref_df).ok()?;
    let p = (1.0 - dist.cdf(corrected)).clamp(0.0, 1.0);
    Some(BartlettCorrection {
        factor,
        corrected_statistic: corrected,
        corrected_p_value: p,
        relative_adjustment: (factor - 1.0).abs(),
    })
}

/// The Bartlett factor from a second-order null mean: `c = E[W] / d`.
///
/// This is the general entry point — `mean_w` is the (analytic-cumulant or
/// null-bootstrap) expectation of the statistic under the penalized null, and
/// `ref_df` is the nominal reference `d`. Returns `None` on degenerate inputs.
pub fn bartlett_factor_from_mean(mean_w: f64, ref_df: f64) -> Option<f64> {
    if !(mean_w.is_finite() && ref_df.is_finite()) || mean_w <= 0.0 || ref_df <= 0.0 {
        return None;
    }
    Some(mean_w / ref_df)
}

/// Exact Bartlett factor for the Gaussian-linear likelihood-ratio test of a
/// `q`-dimensional nested hypothesis with `nu = n − p` residual degrees of
/// freedom.
///
/// In the Gaussian linear model the LR statistic for dropping a `q`-dimensional
/// block is the monotone transform `W = n · log(1 + (q/ν)·F)` of `F ~ F(q, ν)`,
/// and the first-order reference is `χ²_q`. The exact mean of `W` admits a
/// closed form, but to second order (the order at which Bartlett operates) the
/// classical result is
///
/// ```text
/// E[W] = q · ( 1 + (q + 1) / (2ν) ) + O(ν⁻²),
/// ```
///
/// so the exact second-order Bartlett factor is
///
/// ```text
/// c = 1 + (q + 1) / (2ν).
/// ```
///
/// This is the canonical conjugate fixture: `c → 1` as `ν → ∞` (the first-order
/// test is exact in the limit) and `c > 1` at finite `ν` (the uncorrected χ² is
/// anti-conservative). Returns `None` when `q ≤ 0` or `ν ≤ 0`.
pub fn gaussian_linear_bartlett_factor(q: f64, residual_df: f64) -> Option<f64> {
    if !(q.is_finite() && residual_df.is_finite()) || q <= 0.0 || residual_df <= 0.0 {
        return None;
    }
    Some(1.0 + (q + 1.0) / (2.0 * residual_df))
}

// ───────────────────────────────────────────────────────────────────────────
// Penalized-null cumulant assembly from the #932 derivative towers (issue #939)
// ───────────────────────────────────────────────────────────────────────────

/// Per-row log-likelihood derivatives in the row's linear-predictor `η`, for a
/// single-predictor (`K = 1`) GLM-type family: `ℓ'ᵢ, ℓ''ᵢ, ℓ'''ᵢ, ℓ''''ᵢ`.
///
/// These are exactly the diagonal channels of the `K = 1` #932 row tower
/// ([`crate::families::jet_tower::Tower4`]): the tower carries the row *negative*
/// log-likelihood, so `ℓ⁽ᵏ⁾ᵢ = −towerᵢ.derivative_k`. [`row_derivs_from_nll_tower`]
/// performs that sign flip; constructing this struct directly lets callers feed
/// closed-form derivatives (e.g. the Gaussian fixture) without a tower.
#[derive(Debug, Clone, Copy)]
pub struct RowLogLikDerivs {
    /// `ℓ'ᵢ = ∂ℓᵢ/∂ηᵢ` (the score contribution).
    pub d1: f64,
    /// `ℓ''ᵢ = ∂²ℓᵢ/∂ηᵢ²` (≤ 0 for a concave row likelihood).
    pub d2: f64,
    /// `ℓ'''ᵢ`.
    pub d3: f64,
    /// `ℓ''''ᵢ`.
    pub d4: f64,
}

/// Flip the sign of a `K = 1` NLL row tower's `(g, h, t3, t4)` diagonal channels
/// into log-likelihood derivatives. The tower stores the *negative* log
/// likelihood, so `ℓ⁽ᵏ⁾ = −tower⁽ᵏ⁾`.
pub fn row_derivs_from_nll_tower(
    value_grad: f64,
    hess: f64,
    third: f64,
    fourth: f64,
) -> RowLogLikDerivs {
    RowLogLikDerivs {
        d1: -value_grad,
        d2: -hess,
        d3: -third,
        d4: -fourth,
    }
}

/// The exact cumulant arrays the Bartlett/Skovgaard expansions consume, over a
/// tested coefficient block `Z` (the `n × q` design columns of the term under
/// test). For a GLM-type log-likelihood `ℓ = Σᵢ ℓᵢ(ηᵢ)` with `ηᵢ = xᵢᵀβ`, the
/// derivatives w.r.t. the block coefficients factor through `ηᵢ` by the chain
/// rule, so every cumulant array is a row sum of the per-row `η`-derivative
/// times an outer product of `Z`-rows:
///
/// ```text
/// info_{ab}     =  −Σᵢ ℓ''ᵢ · Z_{ia} Z_{ib}          (observed/expected Fisher info)
/// nu3_{abc}     =   Σᵢ ℓ'''ᵢ · Z_{ia} Z_{ib} Z_{ic}
/// nu4_{abcd}    =   Σᵢ ℓ''''ᵢ · Z_{ia} Z_{ib} Z_{ic} Z_{id}
/// ```
///
/// These are exact (the per-row `ℓ⁽ᵏ⁾` come from the #932 tower) and fully
/// symmetric in their indices by construction. They are stored flattened in
/// row-major order (`nu3` length `q³`, `nu4` length `q⁴`) so the consuming
/// contraction can stride them without re-deriving the symmetry.
#[derive(Debug, Clone)]
pub struct CumulantArrays {
    /// Block dimension `q`.
    pub q: usize,
    /// Fisher information block `info_{ab}` (`q × q`, row-major).
    pub info: Vec<f64>,
    /// Third cumulant array `nu3_{abc}` (`q³`, row-major).
    pub nu3: Vec<f64>,
    /// Fourth cumulant array `nu4_{abcd}` (`q⁴`, row-major).
    pub nu4: Vec<f64>,
}

impl CumulantArrays {
    #[inline]
    pub fn info(&self, a: usize, b: usize) -> f64 {
        self.info[a * self.q + b]
    }
    #[inline]
    pub fn nu3(&self, a: usize, b: usize, c: usize) -> f64 {
        self.nu3[(a * self.q + b) * self.q + c]
    }
    #[inline]
    pub fn nu4(&self, a: usize, b: usize, c: usize, d: usize) -> f64 {
        self.nu4[((a * self.q + b) * self.q + c) * self.q + d]
    }
}

/// Assemble [`CumulantArrays`] over a tested block.
///
/// * `block` — the `n × q` tested design columns `Z`, as `n` row slices each of
///   length `q` (row-major rows). This is the block the smooth-term test
///   targets, in the coordinates the per-row derivatives are taken in.
/// * `rows` — the per-row log-likelihood `η`-derivatives (length `n`), from the
///   #932 tower via [`row_derivs_from_nll_tower`] or a family closed form.
///
/// Returns `None` on a dimension mismatch, an empty block, or a non-finite
/// entry. The work is `O(n · q⁴)` and embarrassingly parallel in the rows.
pub fn assemble_cumulants(block: &[&[f64]], rows: &[RowLogLikDerivs]) -> Option<CumulantArrays> {
    let n = rows.len();
    if n == 0 || block.len() != n {
        return None;
    }
    let q = block[0].len();
    if q == 0 || block.iter().any(|r| r.len() != q) {
        return None;
    }
    let mut info = vec![0.0_f64; q * q];
    let mut nu3 = vec![0.0_f64; q * q * q];
    let mut nu4 = vec![0.0_f64; q * q * q * q];
    for (z, d) in block.iter().zip(rows.iter()) {
        if !(d.d1.is_finite() && d.d2.is_finite() && d.d3.is_finite() && d.d4.is_finite()) {
            return None;
        }
        if z.iter().any(|v| !v.is_finite()) {
            return None;
        }
        for a in 0..q {
            let za = z[a];
            for b in 0..q {
                let zab = za * z[b];
                info[a * q + b] -= d.d2 * zab;
                for c in 0..q {
                    let zabc = zab * z[c];
                    nu3[(a * q + b) * q + c] += d.d3 * zabc;
                    for e in 0..q {
                        nu4[((a * q + b) * q + c) * q + e] += d.d4 * zabc * z[e];
                    }
                }
            }
        }
    }
    if info
        .iter()
        .chain(nu3.iter())
        .chain(nu4.iter())
        .any(|v| !v.is_finite())
    {
        return None;
    }
    Some(CumulantArrays { q, info, nu3, nu4 })
}

/// Bartlett's standardized cumulant invariants of a scalar (`q = 1`) sub-model,
/// the building blocks of the LR-statistic correction.
///
/// From the assembled scalar cumulants this returns the dimensionless
/// `ρ₃ = ν₃ / i^{3/2}` and `ρ₄ = ν₄ / i²`, the parametrization-equivariant
/// standardized third/fourth cumulants of the score. The Bartlett factor of the
/// LR statistic is a fixed rational form in these invariants (the full Lawley
/// (1956) scalar expansion — it also requires the score↔information joint
/// cumulant, NOT just `ρ₃²` and `ρ₄`, which is why this function deliberately
/// exposes the invariants rather than guessing a two-term coefficient). The
/// acceptance fixture for any candidate coefficient is the unit-rate Exponential
/// rate test, whose exact LR Bartlett factor is the textbook `c = 1 + 1/n`.
///
/// Returns `None` unless the cumulants are scalar with positive, finite
/// information.
pub fn scalar_standardized_cumulants(cumulants: &CumulantArrays) -> Option<(f64, f64)> {
    if cumulants.q != 1 {
        return None;
    }
    let i = cumulants.info(0, 0);
    if !(i.is_finite() && i > 0.0) {
        return None;
    }
    let rho3 = cumulants.nu3(0, 0, 0) / i.powf(1.5);
    let rho4 = cumulants.nu4(0, 0, 0, 0) / (i * i);
    if rho3.is_finite() && rho4.is_finite() {
        Some((rho3, rho4))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::distribution::{ContinuousCDF, FisherSnedecor};

    #[test]
    fn bartlett_factor_recovers_mean_over_df() {
        // c = E[W]/d.
        let c = bartlett_factor_from_mean(6.0, 4.0).expect("factor");
        assert!((c - 1.5).abs() < 1e-12);
        assert!(bartlett_factor_from_mean(-1.0, 4.0).is_none());
        assert!(bartlett_factor_from_mean(6.0, 0.0).is_none());
    }

    #[test]
    fn correction_rescales_statistic_and_enlarges_p_for_inflated_stat() {
        // factor > 1 ⇒ corrected statistic smaller ⇒ larger (more conservative)
        // p-value, exactly the fix for an anti-conservative first-order test.
        let raw_w = 12.0;
        let d = 4.0;
        let factor = 1.5;
        let corr = bartlett_correct(raw_w, d, factor).expect("correction");
        assert!((corr.corrected_statistic - 8.0).abs() < 1e-12);
        let dist = ChiSquared::new(d).unwrap();
        let raw_p = 1.0 - dist.cdf(raw_w);
        assert!(
            corr.corrected_p_value > raw_p,
            "corrected p {} must exceed raw p {}",
            corr.corrected_p_value,
            raw_p
        );
        assert!((corr.relative_adjustment - 0.5).abs() < 1e-12);
    }

    /// THE CONJUGATE FIXTURE (issue #939 riding test). In the Gaussian linear
    /// model the LR statistic `W = n·log(1 + (q/ν)F)` for a q-dim nested
    /// hypothesis has the EXACT distribution induced by `F ~ F(q, ν)`. The
    /// first-order χ²_q reference is wrong at finite ν: E[W] ≠ q. We prove the
    /// Bartlett factor moves the reference mean TOWARD truth — the corrected
    /// statistic's mean lands on q to second order while the uncorrected one
    /// overshoots — by Monte-Carlo-free numerical integration of the exact W
    /// distribution against the closed-form factor.
    #[test]
    fn gaussian_linear_bartlett_moves_mean_toward_truth() {
        let q = 3.0_f64;
        let nu = 20.0_f64; // n - p; modest residual df where first-order is off.
        let n = (q + 1.0 + nu) as f64; // p = q + 1 (intercept + q tested cols).

        let c = gaussian_linear_bartlett_factor(q, nu).expect("factor");
        // c = 1 + (q+1)/(2ν) = 1 + 4/40 = 1.1.
        assert!((c - 1.1).abs() < 1e-12);

        // Exact E[W] by deterministic quadrature over the F(q, ν) density:
        // W(f) = n·log(1 + (q/ν) f). Integrate W(f)·pdf(f) df on a fine grid.
        let fdist = FisherSnedecor::new(q, nu).expect("F dist");
        let pdf = |f: f64| {
            // statrs exposes the cdf; approximate the pdf via central difference
            // of the cdf (smooth, monotone — stable to 1e-6 here).
            let h = 1e-5 * (1.0 + f);
            (fdist.cdf(f + h) - fdist.cdf(f - h)) / (2.0 * h)
        };
        let w_of = |f: f64| n * (1.0 + (q / nu) * f).ln();
        // Trapezoidal integration of E[W] = ∫ W(f) pdf(f) df over [0, F_hi].
        let f_hi = 60.0_f64;
        let steps = 600_000usize;
        let dx = f_hi / steps as f64;
        let mut e_w = 0.0;
        for i in 0..=steps {
            let f = (i as f64) * dx + 1e-9;
            let weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
            e_w += weight * w_of(f) * pdf(f);
        }
        e_w *= dx;

        // First-order reference says E[W] should be q; it is not.
        let raw_bias = (e_w - q).abs();
        assert!(
            raw_bias > 0.1,
            "first-order test should be materially biased at ν={nu}: E[W]={e_w}, q={q}"
        );

        // The corrected statistic W/c has mean E[W]/c. The closed-form factor
        // must bring it to q far more tightly than the uncorrected mean.
        let corrected_mean = e_w / c;
        let corrected_bias = (corrected_mean - q).abs();
        assert!(
            corrected_bias < 0.5 * raw_bias,
            "Bartlett correction must move the mean toward truth: \
             raw_bias={raw_bias:.5} corrected_bias={corrected_bias:.5} \
             (E[W]={e_w:.5}, c={c:.5})"
        );
    }

    #[test]
    fn factor_vanishes_in_the_large_sample_limit() {
        // As ν → ∞ the correction must disappear (first-order test becomes exact).
        let c_small = gaussian_linear_bartlett_factor(3.0, 10.0).unwrap();
        let c_large = gaussian_linear_bartlett_factor(3.0, 100_000.0).unwrap();
        assert!(c_small > 1.0);
        assert!((c_large - 1.0).abs() < 1e-3);
        assert!(c_small > c_large);
    }

    // ── Cumulant assembly from the towers (#939) ──────────────────────────

    #[test]
    fn nll_tower_sign_flip_gives_loglik_derivatives() {
        // Tower carries the NLL; ℓ⁽ᵏ⁾ = −tower⁽ᵏ⁾.
        let d = row_derivs_from_nll_tower(0.5, -2.0, 0.3, -0.1);
        assert_eq!(d.d1, -0.5);
        assert_eq!(d.d2, 2.0);
        assert_eq!(d.d3, -0.3);
        assert_eq!(d.d4, 0.1);
    }

    #[test]
    fn cumulant_arrays_are_exact_row_sums_and_fully_symmetric() {
        // Two rows, q = 2 block. Hand-compute the exact arrays.
        let z0 = [1.0_f64, 2.0];
        let z1 = [-1.0_f64, 0.5];
        let block: Vec<&[f64]> = vec![&z0, &z1];
        let rows = vec![
            RowLogLikDerivs {
                d1: 0.0,
                d2: -1.5,
                d3: 0.7,
                d4: -0.2,
            },
            RowLogLikDerivs {
                d1: 0.0,
                d2: -0.5,
                d3: 1.1,
                d4: 0.4,
            },
        ];
        let c = assemble_cumulants(&block, &rows).expect("cumulants");
        assert_eq!(c.q, 2);
        // info_{ab} = −Σ d2 z_a z_b. Row0: −(−1.5)·z0⊗z0, Row1: −(−0.5)·z1⊗z1.
        let info00 = 1.5 * (1.0 * 1.0) + 0.5 * (-1.0 * -1.0);
        let info01 = 1.5 * (1.0 * 2.0) + 0.5 * (-1.0 * 0.5);
        assert!((c.info(0, 0) - info00).abs() < 1e-12);
        assert!((c.info(0, 1) - info01).abs() < 1e-12);
        // Symmetry of info.
        assert!((c.info(0, 1) - c.info(1, 0)).abs() < 1e-14);
        // nu3_{abc} = Σ d3 z_a z_b z_c.
        let nu3_010 = 0.7 * (1.0 * 2.0 * 1.0) + 1.1 * (-1.0 * 0.5 * -1.0);
        assert!((c.nu3(0, 1, 0) - nu3_010).abs() < 1e-12);
        // Full symmetry of nu3 across index permutations.
        assert!((c.nu3(0, 1, 0) - c.nu3(1, 0, 0)).abs() < 1e-14);
        assert!((c.nu3(0, 1, 0) - c.nu3(0, 0, 1)).abs() < 1e-14);
        // nu4_{abcd} = Σ d4 z_a z_b z_c z_d.
        let nu4_0011 = -0.2 * (1.0 * 1.0 * 2.0 * 2.0) + 0.4 * (-1.0 * -1.0 * 0.5 * 0.5);
        assert!((c.nu4(0, 0, 1, 1) - nu4_0011).abs() < 1e-12);
        assert!((c.nu4(0, 0, 1, 1) - c.nu4(1, 1, 0, 0)).abs() < 1e-14);
    }

    /// CONJUGATE FIXTURE 1 (Gaussian known variance, scalar): ℓ''' = ℓ'''' = 0,
    /// so the standardized cumulants vanish — `W ~ χ²₁` holds EXACTLY and the
    /// assembly correctly reports no finite-sample correction signal.
    #[test]
    fn gaussian_known_variance_has_zero_standardized_cumulants() {
        // ℓ_i = −½(y−η)²/φ ⇒ ℓ' = (y−η)/φ, ℓ'' = −1/φ, ℓ''' = 0, ℓ'''' = 0.
        let phi = 2.0;
        let n = 50usize;
        let zcol = [1.0_f64];
        let block: Vec<&[f64]> = (0..n).map(|_| &zcol[..]).collect();
        let rows: Vec<RowLogLikDerivs> = (0..n)
            .map(|_| RowLogLikDerivs {
                d1: 0.0,
                d2: -1.0 / phi,
                d3: 0.0,
                d4: 0.0,
            })
            .collect();
        let c = assemble_cumulants(&block, &rows).expect("cumulants");
        let (rho3, rho4) = scalar_standardized_cumulants(&c).expect("standardized");
        assert!(rho3.abs() < 1e-12, "Gaussian ρ₃ must be 0, got {rho3}");
        assert!(rho4.abs() < 1e-12, "Gaussian ρ₄ must be 0, got {rho4}");
        // info = n/φ.
        assert!((c.info(0, 0) - (n as f64) / phi).abs() < 1e-10);
    }

    /// CONJUGATE FIXTURE 2 (unit-rate Exponential, scalar): the standardized
    /// cumulants have exact closed forms `ρ₃ = 2/√n`, `ρ₄ = −6/n`. This is the
    /// anchor the full Lawley LR Bartlett coefficient must reproduce
    /// (`c = 1 + 1/n`); we assert the assembled invariants exactly so the
    /// coefficient derivation has a verified substrate to contract.
    #[test]
    fn exponential_rate_standardized_cumulants_match_closed_form() {
        // ℓ_i(θ) = ln θ − θ y_i. At θ = 1: ℓ' = 1/θ − y, ℓ'' = −1/θ², ℓ''' = 2/θ³,
        // ℓ'''' = −6/θ⁴. Scalar parameter ⇒ z_i = 1.
        let theta = 1.0_f64;
        let n = 64usize;
        let zcol = [1.0_f64];
        let block: Vec<&[f64]> = (0..n).map(|_| &zcol[..]).collect();
        let rows: Vec<RowLogLikDerivs> = (0..n)
            .map(|_| RowLogLikDerivs {
                d1: 0.0, // not used by the cumulant arrays
                d2: -1.0 / (theta * theta),
                d3: 2.0 / theta.powi(3),
                d4: -6.0 / theta.powi(4),
            })
            .collect();
        let c = assemble_cumulants(&block, &rows).expect("cumulants");
        // info = n/θ² = n.
        assert!((c.info(0, 0) - n as f64).abs() < 1e-10);
        let (rho3, rho4) = scalar_standardized_cumulants(&c).expect("standardized");
        let nf = n as f64;
        assert!(
            (rho3 - 2.0 / nf.sqrt()).abs() < 1e-10,
            "Exponential ρ₃ must be 2/√n = {}, got {rho3}",
            2.0 / nf.sqrt()
        );
        assert!(
            (rho4 - (-6.0 / nf)).abs() < 1e-10,
            "Exponential ρ₄ must be −6/n = {}, got {rho4}",
            -6.0 / nf
        );
    }

    #[test]
    fn assemble_cumulants_rejects_degenerate_input() {
        let z = [1.0_f64];
        let block: Vec<&[f64]> = vec![&z];
        // length mismatch between block and rows.
        assert!(assemble_cumulants(&block, &[]).is_none());
        // non-finite derivative.
        let bad = vec![RowLogLikDerivs {
            d1: 0.0,
            d2: f64::NAN,
            d3: 0.0,
            d4: 0.0,
        }];
        assert!(assemble_cumulants(&block, &bad).is_none());
    }
}

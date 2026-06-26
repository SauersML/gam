//! Higher-order likelihood asymptotics for likelihood-ratio tests (issue #939):
//! Bartlett corrections that make the first-order χ² reference distribution
//! second-order accurate at modest `n` and near-boundary `λ`.
//!
//! For a likelihood-ratio statistic `W` with first-order reference `χ²_d`, the
//! statistic's mean can drift from `d` at finite `n`, distorting the test size.
//! The **Bartlett correction** rescales the statistic by `c = E[W]/d` so the
//! corrected statistic `W* = W/c` has mean `d` again and its `χ²_d` tail is
//! accurate to `O(n⁻²)` rather than `O(n⁻¹)`.
//!
//! A general [`bartlett_factor_from_mean`] takes the second-order mean of the
//! statistic under the null (from cumulant assembly or a null parametric
//! bootstrap) and returns the correction factor.

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

// ───────────────────────────────────────────────────────────────────────────
// Penalized-null cumulant assembly from the #932 derivative towers (issue #939)
// ───────────────────────────────────────────────────────────────────────────

/// Per-row log-likelihood derivatives in the row's linear-predictor `η`, for a
/// single-predictor (`K = 1`) GLM-type family: `ℓ'ᵢ, ℓ''ᵢ, ℓ'''ᵢ, ℓ''''ᵢ`.
///
/// These are exactly the diagonal channels of the `K = 1` #932 row tower
/// ([`gam_math::jet_tower::Tower4`]): the tower carries the row *negative*
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
/// rate test, whose exact LR Bartlett factor is `2n(log n − ψ(n)) =
/// 1 + 1/(6n) + O(n⁻³)`.
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
    use statrs::function::gamma::digamma;

    #[test]
    fn bartlett_factor_recovers_mean_over_df() {
        // c = E[W]/d.
        let c = bartlett_factor_from_mean(6.0, 4.0).expect("factor");
        assert!((c - 1.5).abs() < 1e-12);
        assert!(bartlett_factor_from_mean(-1.0, 4.0).is_none());
        assert!(bartlett_factor_from_mean(6.0, 0.0).is_none());
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

    /// CONJUGATE FIXTURE 1 (Gaussian known variance, unpenalized scalar):
    /// ℓ''' = ℓ'''' = 0, so the standardized cumulants vanish and the assembly
    /// correctly reports no finite-sample Lawley correction signal.
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
    /// cumulants have exact closed forms `ρ₃ = 2/√n`, `ρ₄ = −6/n`. The exact
    /// LR Bartlett factor is `2n(log n − ψ(n)) = 1 + 1/(6n) + O(n⁻³)`;
    /// lawley.rs owns the full derivation, while this fixture verifies the
    /// cumulant substrate and exact anchor value.
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
        let exact_factor = 2.0 * nf * (nf.ln() - digamma(nf));
        let second_order = 1.0 + 1.0 / (6.0 * nf);
        assert!(
            exact_factor > 1.0 && exact_factor < second_order,
            "Exponential exact LR Bartlett factor must be between 1 and its second-order expansion; got {exact_factor}"
        );
        assert!(
            (exact_factor - second_order).abs() < 1.0 / nf.powi(3),
            "Exponential exact factor {exact_factor} must match 1 + 1/(6n) through O(n^-3)"
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

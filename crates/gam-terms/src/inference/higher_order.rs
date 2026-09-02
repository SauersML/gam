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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bartlett_factor_recovers_mean_over_df() {
        // c = E[W]/d.
        let c = bartlett_factor_from_mean(6.0, 4.0).expect("factor");
        assert!((c - 1.5).abs() < 1e-12);
        assert!(bartlett_factor_from_mean(-1.0, 4.0).is_none());
        assert!(bartlett_factor_from_mean(6.0, 0.0).is_none());
    }

    // ── Cumulant assembly from the towers (#939) ──────────────────────────

}

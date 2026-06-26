//! Lawley (1956) cumulant assembly for Bartlett corrections (#939).
//!
//! [`crate::inference::higher_order`] owns the mean-to-factor helper: given the
//! second-order null mean `E[W] = q + ε_k − ε_{k−q}` of a likelihood-ratio
//! statistic, it returns the Bartlett factor `E[W] / q`. This module computes
//! the missing analytic ingredient — the Lawley ε terms — exactly, from per-row
//! expected log-likelihood derivative cumulants.
//!
//! # Lawley's expansion
//!
//! For a `k`-parameter model, `ε_k = Σ_θ (λ_rstu − λ_rstuvw)` with
//!
//! ```text
//! λ_rstu   = κ^{rs} κ^{tu} { κ_rstu/4 − κ_rst^(u) + κ_rt^(su) },
//! λ_rstuvw = κ^{rs} κ^{tu} κ^{vw} { κ_rtv (κ_suw/6 − κ_sw^(u))
//!            + κ_rtu (κ_svw/4 − κ_sw^(v)) + κ_rt^(v) κ_sw^(u)
//!            + κ_rt^(u) κ_sw^(v) },
//! ```
//!
//! where `κ_rs = E[∂²ℓ/∂θ_r∂θ_s]`, `κ_rst = E[∂³ℓ]`, `κ_rs^(t) = ∂κ_rs/∂θ_t`,
//! and `κ^{rs}` is the matrix inverse of `[κ_rs]` (Lawley 1956; the display
//! follows Cribari-Neto & Queiroz, "Bartlett corrections in beta regression
//! models", arXiv:1501.07551, eq. 4). `ε_{k−q}` is the same sum restricted to
//! the nuisance block, i.e. this function evaluated on the null model.
//!
//! # Why the "joint cumulants" are computable here
//!
//! GLM-type log-likelihoods are *linear in y*, so every η-derivative of `ℓ_i`
//! is linear in y and all the arrays above need only `E[y] = μ`: no third or
//! fourth response moments enter. The mixed arrays `κ_rs^(t)` — which are NOT
//! recoverable from the pointwise expected contractions ν₃/ν₄ — are
//! η-derivatives of the expected curvature as a *function*, and those are exact
//! jet compositions of the link and variance jets. With `c(η) = μ′/V(μ)` and
//! `u₀ = μ′·c` (the Fisher weight),
//!
//! ```text
//! κ₂ = −u₀          κ₂' = −u₀'                κ₂'' = −u₀''
//! κ₃ = −(u₀' + μ′c′)            κ₃' = −(u₀'' + μ″c′ + μ′c″)
//! κ₄ = −(u₀'' + μ″c′ + 2μ′c″)
//! ```
//!
//! (primes are η-derivatives; all divided by the dispersion φ). Canonical
//! links have `c′ = c″ = 0`, hence `κ₃ = κ₂'` and `κ₄ = κ₃'` — pinned in tests.
//!
//! # Row-pair (hat) reduction
//!
//! For `θ`-derivatives chained through `η_i = x_iᵀβ` every array is an
//! `X`-contraction of per-row scalars, and the six-index sum collapses to
//! pairwise contractions of `E = X K⁻¹ Xᵀ` (`h_i = E_ii`):
//!
//! ```text
//! λ₄ = Σ_i a_i h_i²,                       a_i = κ₄/4 − κ₃' + κ₂''
//! λ₆ = −Σ_ij { E_ij³ [κ₃ᵢκ₃ⱼ/6 − κ₃ᵢκ₂'ⱼ + κ₂'ᵢκ₂'ⱼ]
//!            + h_i h_j E_ij [κ₃ᵢκ₃ⱼ/4 − κ₃ᵢκ₂'ⱼ + κ₂'ᵢκ₂'ⱼ] }
//! ε  = λ₄ − λ₆.
//! ```
//!
//! The reduction is verified against the raw six-index Lawley sum in tests, and
//! against two *exact* finite-sample distributions: the exponential/log-link
//! intercept model, where exact `E[W] = 2n(log n − ψ(n))` with expansion
//! `1 + 1/(6n) + O(n⁻³)`, and the Poisson/log intercept model by exact pmf
//! summation, where the classical factor is `1/(6nλ)`.
//!
//! # Penalized models
//!
//! A quadratic penalty `−½βᵀS_λβ` is deterministic *at fixed λ*: it shifts
//! `κ_rs` by `−S_λ` and leaves every third/fourth-order and derivative array
//! unchanged. When the null value annihilates the penalty (`S_λ β₀ = 0` — the
//! usual smooth-term null "this smooth is zero"), the penalized score has mean
//! zero at the null and Lawley's expansion applies verbatim with the penalized
//! information: pass `penalty` to fold `S_λ` into `K`. That gives the
//! *conditional* mean shift `Δε(ρ)` = `E[W | λ]`.
//!
//! When λ is **estimated**, ρ̂ = log λ̂ has its own sampling variation and
//! `E[W]` picks up the extra second-order delta-method term
//! `½ Σ (∂²Δε/∂ρ_b∂ρ_{b'}) Cov(ρ̂_b, ρ̂_{b'})` —
//! [`lawley_lr_mean_shift_with_rho_variation`] assembles it exactly from the
//! curvature of the deterministic `Δε(ρ)` and the inverse REML outer Hessian
//! `Cov(ρ̂)` (the #740 quantity). This is the genuinely-new penalized-Bartlett
//! contribution #939 deliverable (2) asks for. All of this remains LR-only
//! machinery; it is not a calibration for Wood's rank-truncated Wald statistic.
//!
//! Cost: `O(n²k)` time and `O(n²)` memory for the pair matrix `E` — fine for
//! the small-`n` regimes where Bartlett corrections matter (the correction is
//! `O(n⁻¹)`; at large `n` the first-order test is already calibrated).

use ndarray::{Array1, Array2, ArrayView2};

use gam_linalg::faer_ndarray::{FaerArrayView, factorize_symmetricwith_fallback};
use gam_linalg::matrix::FactorizedSystem;
use faer::Side;

/// Order-2 jet (value, first, second η-derivative) product.
#[inline]
fn jet_mul(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[0] * b[0],
        a[0] * b[1] + a[1] * b[0],
        a[0] * b[2] + 2.0 * a[1] * b[1] + a[2] * b[0],
    ]
}

/// Order-2 jet quotient `a / b` (requires `b[0] != 0`).
#[inline]
fn jet_div(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    let q0 = a[0] / b[0];
    let q1 = (a[1] - q0 * b[1]) / b[0];
    let q2 = (a[2] - q0 * b[2] - 2.0 * q1 * b[1]) / b[0];
    [q0, q1, q2]
}

/// Per-row expected jets of a GLM family/link pair at a linear predictor value:
/// the η-derivatives of the inverse link and the μ-derivatives of the variance
/// function. Everything Lawley needs is assembled from these by exact jet
/// composition — no finite differences anywhere.
#[derive(Debug, Clone, Copy)]
pub struct RowExpectedJets {
    /// dμ/dη.
    pub mu1: f64,
    /// d²μ/dη².
    pub mu2: f64,
    /// d³μ/dη³.
    pub mu3: f64,
    /// V(μ).
    pub var: f64,
    /// dV/dμ.
    pub dvar_dmu: f64,
    /// d²V/dμ².
    pub d2var_dmu2: f64,
    /// Dispersion φ (log-likelihood is `[yθ − b(θ)]/φ`): scales every cumulant
    /// by `1/φ`.
    pub dispersion: f64,
}

/// The per-row expected cumulant scalars entering Lawley's arrays.
#[derive(Debug, Clone, Copy)]
pub struct RowKappas {
    /// κ₂ = E[ℓ″] (= −Fisher weight).
    pub k2: f64,
    /// κ₃ = E[ℓ‴].
    pub k3: f64,
    /// κ₄ = E[ℓ⁗].
    pub k4: f64,
    /// κ₂' = dκ₂/dη.
    pub k2_1: f64,
    /// κ₂'' = d²κ₂/dη².
    pub k2_11: f64,
    /// κ₃' = dκ₃/dη.
    pub k3_1: f64,
}

impl RowKappas {
    /// Scale every cumulant by a prior weight `w`: a row observed with weight
    /// `w` contributes `w·ℓᵢ` to the log-likelihood (for binomial data `w`
    /// trials at the same η are exactly the sum of `w` independent Bernoulli
    /// rows), so all six expected derivative cumulants scale linearly in `w`.
    pub fn weighted(self, w: f64) -> Self {
        Self {
            k2: self.k2 * w,
            k3: self.k3 * w,
            k4: self.k4 * w,
            k2_1: self.k2_1 * w,
            k2_11: self.k2_11 * w,
            k3_1: self.k3_1 * w,
        }
    }
}

impl RowExpectedJets {
    /// Assemble the per-row cumulant scalars by jet composition of
    /// `c = μ′/V(μ)` and `u₀ = μ′·c`.
    pub fn kappas(&self) -> Result<RowKappas, String> {
        let phi = self.dispersion;
        if !(phi.is_finite() && phi > 0.0) {
            return Err(format!(
                "RowExpectedJets::kappas: dispersion must be finite and positive; got {phi}"
            ));
        }
        if !(self.var.is_finite() && self.var > 0.0) {
            return Err(format!(
                "RowExpectedJets::kappas: variance function must be finite and positive; got {}",
                self.var
            ));
        }
        // η-jets of μ′ and of V(μ(η)) (chain rule for the composition).
        let mu1_jet = [self.mu1, self.mu2, self.mu3];
        let v_jet = [
            self.var,
            self.dvar_dmu * self.mu1,
            self.d2var_dmu2 * self.mu1 * self.mu1 + self.dvar_dmu * self.mu2,
        ];
        let c = jet_div(mu1_jet, v_jet);
        let u0 = jet_mul(mu1_jet, c);
        let inv_phi = 1.0 / phi;
        Ok(RowKappas {
            k2: -u0[0] * inv_phi,
            k2_1: -u0[1] * inv_phi,
            k2_11: -u0[2] * inv_phi,
            k3: -(u0[1] + self.mu1 * c[1]) * inv_phi,
            k3_1: -(u0[2] + self.mu2 * c[1] + self.mu1 * c[2]) * inv_phi,
            k4: -(u0[2] + self.mu2 * c[1] + 2.0 * self.mu1 * c[2]) * inv_phi,
        })
    }

    /// Gaussian family, identity link, variance φ = σ².
    pub fn gaussian_identity(dispersion: f64) -> Self {
        Self {
            mu1: 1.0,
            mu2: 0.0,
            mu3: 0.0,
            var: 1.0,
            dvar_dmu: 0.0,
            d2var_dmu2: 0.0,
            dispersion,
        }
    }

    /// Poisson family, log link (canonical), at linear predictor `eta`.
    pub fn poisson_log(eta: f64) -> Self {
        let mu = eta.exp();
        Self {
            mu1: mu,
            mu2: mu,
            mu3: mu,
            var: mu,
            dvar_dmu: 1.0,
            d2var_dmu2: 0.0,
            dispersion: 1.0,
        }
    }

    /// Bernoulli family, logit link (canonical), at linear predictor `eta`.
    pub fn binomial_logit(eta: f64) -> Self {
        let mu = 1.0 / (1.0 + (-eta).exp());
        let mu1 = mu * (1.0 - mu);
        let mu2 = mu1 * (1.0 - 2.0 * mu);
        let mu3 = mu2 * (1.0 - 2.0 * mu) - 2.0 * mu1 * mu1;
        Self {
            mu1,
            mu2,
            mu3,
            var: mu1,
            dvar_dmu: 1.0 - 2.0 * mu,
            d2var_dmu2: -2.0,
            dispersion: 1.0,
        }
    }

    /// Gamma family, log link (non-canonical), at linear predictor `eta` with
    /// dispersion φ (shape 1/φ; φ = 1 is the exponential distribution).
    pub fn gamma_log(eta: f64, dispersion: f64) -> Self {
        let mu = eta.exp();
        Self {
            mu1: mu,
            mu2: mu,
            mu3: mu,
            var: mu * mu,
            dvar_dmu: 2.0 * mu,
            d2var_dmu2: 2.0,
            dispersion,
        }
    }
}

/// Lawley's `ε` for a GLM block: design `x` (n × k), per-row cumulants, and an
/// optional quadratic penalty `S_λ` folded into the information (valid for
/// nulls with `S_λ β₀ = 0`; see the module docs). Evaluate at the null fit.
///
/// `ε_k − ε_{k−q}` (full minus nuisance-restricted, the latter being this
/// function on the null model) is the second-order mean shift of the LR
/// statistic; feed `q + ε_k − ε_{k−q}` to
/// [`crate::inference::higher_order::bartlett_factor_from_mean`].
pub fn lawley_epsilon(
    x: ArrayView2<'_, f64>,
    kappas: &[RowKappas],
    penalty: Option<ArrayView2<'_, f64>>,
) -> Result<f64, String> {
    let n = x.nrows();
    let k = x.ncols();
    if n == 0 || k == 0 {
        return Err(format!(
            "lawley_epsilon: empty design ({n} rows, {k} columns)"
        ));
    }
    if kappas.len() != n {
        return Err(format!(
            "lawley_epsilon: {} cumulant rows for {n} design rows",
            kappas.len()
        ));
    }
    // J = −[κ_rs] (+ S_λ): the (penalized) expected information, SPD.
    let mut j_mat = Array2::<f64>::zeros((k, k));
    for (i, row_kappas) in kappas.iter().enumerate() {
        let weight = -row_kappas.k2;
        if !weight.is_finite() {
            return Err(format!(
                "lawley_epsilon: non-finite Fisher weight at row {i}"
            ));
        }
        for r in 0..k {
            let xr = x[[i, r]] * weight;
            for s in 0..k {
                j_mat[[r, s]] += xr * x[[i, s]];
            }
        }
    }
    if let Some(s_pen) = penalty {
        if s_pen.nrows() != k || s_pen.ncols() != k {
            return Err(format!(
                "lawley_epsilon: penalty is {}×{}, expected {k}×{k}",
                s_pen.nrows(),
                s_pen.ncols()
            ));
        }
        j_mat += &s_pen;
    }
    let j_view = FaerArrayView::new(&j_mat);
    let factor = factorize_symmetricwith_fallback(j_view.as_ref(), Side::Lower)
        .map_err(|e| format!("lawley_epsilon: information factorization failed: {e:?}"))?;
    let j_inv = FactorizedSystem::solvemulti(&factor, &Array2::<f64>::eye(k))?;

    // Pair matrix E = X J⁻¹ Xᵀ and its diagonal h.
    let e_pairs = x.dot(&j_inv).dot(&x.t());
    let h = e_pairs.diag().to_owned();

    // λ₄ = Σ_i a_i h_i².
    let mut lambda4 = 0.0;
    for (i, row_kappas) in kappas.iter().enumerate() {
        let a_i = row_kappas.k4 / 4.0 - row_kappas.k3_1 + row_kappas.k2_11;
        lambda4 += a_i * h[i] * h[i];
    }

    // λ₆ = −Σ_ij { E³·b6_ij + h h E·b4_ij } with
    // b6_ij = κ₃ᵢκ₃ⱼ/6 − κ₃ᵢκ₂'ⱼ + κ₂'ᵢκ₂'ⱼ and b4 the same with 1/4.
    let k3: Array1<f64> = kappas.iter().map(|r| r.k3).collect();
    let k21: Array1<f64> = kappas.iter().map(|r| r.k2_1).collect();
    let mut lambda6 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let e_ij = e_pairs[[i, j]];
            let cross = k3[i] * k3[j];
            let mixed = -k3[i] * k21[j] + k21[i] * k21[j];
            lambda6 -= e_ij * e_ij * e_ij * (cross / 6.0 + mixed)
                + h[i] * h[j] * e_ij * (cross / 4.0 + mixed);
        }
    }

    let epsilon = lambda4 - lambda6;
    if !epsilon.is_finite() {
        return Err(format!(
            "lawley_epsilon: non-finite ε (λ₄={lambda4}, λ₆={lambda6})"
        ));
    }
    Ok(epsilon)
}

/// Row cap for LR consumers that build the `O(n²)`-memory pair matrix `E` per
/// tested block: at `n = 2048` the matrix is 32 MB and the `O(n⁻¹)` correction
/// is still resolvable; beyond that the first-order LR reference is already
/// calibrated and the quadratic cost buys nothing.
pub const LAWLEY_PAIR_MATRIX_MAX_ROWS: usize = 2048;

/// Lawley's second-order mean shift `Δε = ε_k − ε_{k−q}` of the LR statistic
/// for the null "the coefficients in `tested` are zero" inside the `k`-column
/// model `x`: `E[W] = q + ε_k − ε_{k−q} + O(n⁻²)` (Lawley 1956; module docs).
///
/// * `ε_k` is [`lawley_epsilon`] on the full design (with `penalty` folded in).
/// * `ε_{k−q}` is the same on the nuisance design — the tested columns removed
///   and the matching rows/columns of `penalty` dropped. When the tested block
///   is the whole design the null model is fully specified and `ε_0 = 0`.
///
/// The per-row cumulants are expectations, so both ε terms use the same
/// `kappas`. Lawley evaluates them at the null fit; supplying cumulants at the
/// full fit instead perturbs ε — itself `O(n⁻¹)` — by the `O(n⁻¹ᐟ²)` null-true
/// parameter drift, an `O(n⁻³ᐟ²)` error inside the `O(n⁻²)` Bartlett target.
///
/// The Bartlett factor against a `d`-df reference is `c = E[W]/d = 1 + Δε/d`
/// ([`lawley_lr_bartlett_factor`]).
pub fn lawley_lr_mean_shift(
    x: ArrayView2<'_, f64>,
    kappas: &[RowKappas],
    penalty: Option<ArrayView2<'_, f64>>,
    tested: std::ops::Range<usize>,
) -> Result<f64, String> {
    let n = x.nrows();
    let k = x.ncols();
    if tested.start >= tested.end || tested.end > k {
        return Err(format!(
            "lawley_lr_mean_shift: tested block {}..{} out of range for {k} columns",
            tested.start, tested.end
        ));
    }
    // Full-model ε (validates x/kappas/penalty shapes).
    let eps_full = lawley_epsilon(x, kappas, penalty)?;
    let nuisance: Vec<usize> = (0..k).filter(|c| !tested.contains(c)).collect();
    if nuisance.is_empty() {
        // Fully specified null: the nuisance model has no parameters, ε_0 = 0.
        return Ok(eps_full);
    }
    let m = nuisance.len();
    let mut x_null = Array2::<f64>::zeros((n, m));
    for (col_null, &col_full) in nuisance.iter().enumerate() {
        for i in 0..n {
            x_null[[i, col_null]] = x[[i, col_full]];
        }
    }
    let penalty_null = penalty.map(|s_pen| {
        let mut out = Array2::<f64>::zeros((m, m));
        for (r_null, &r_full) in nuisance.iter().enumerate() {
            for (c_null, &c_full) in nuisance.iter().enumerate() {
                out[[r_null, c_null]] = s_pen[[r_full, c_full]];
            }
        }
        out
    });
    let eps_null = lawley_epsilon(
        x_null.view(),
        kappas,
        penalty_null.as_ref().map(|s_pen| s_pen.view()),
    )?;
    Ok(eps_full - eps_null)
}

/// The Lawley LR Bartlett factor `c = E[W]/d = 1 + (ε_k − ε_{k−q})/d` for the
/// null "`tested` is zero", referenced against `χ²_d` with `d = ref_df` (the
/// consumer's LR reference degrees of freedom; `Δε` already carries any penalty
/// through the information, as scoped in the module docs).
pub fn lawley_lr_bartlett_factor(
    x: ArrayView2<'_, f64>,
    kappas: &[RowKappas],
    penalty: Option<ArrayView2<'_, f64>>,
    tested: std::ops::Range<usize>,
    ref_df: f64,
) -> Result<f64, String> {
    if !(ref_df.is_finite() && ref_df > 0.0) {
        return Err(format!(
            "lawley_lr_bartlett_factor: reference df must be finite and positive; got {ref_df}"
        ));
    }
    let shift = lawley_lr_mean_shift(x, kappas, penalty, tested)?;
    let mean_w = ref_df + shift;
    let factor = crate::inference::higher_order::bartlett_factor_from_mean(mean_w, ref_df)
        .ok_or_else(|| {
            format!(
                "lawley_lr_bartlett_factor: degenerate mean {mean_w} (Δε = {shift}, d = {ref_df})"
            )
        })?;
    if !(factor.is_finite() && factor > 0.0) {
        return Err(format!(
            "lawley_lr_bartlett_factor: degenerate factor {factor} (Δε = {shift}, d = {ref_df})"
        ));
    }
    Ok(factor)
}

/// A single smoothing-parameter direction for the ρ̂-variation correction: the
/// component penalty matrix `S_b` (already at its fitted scale `e^{ρ_b} S_b^unit`)
/// that the smoothing parameter `λ_b = e^{ρ_b}` multiplies inside the total
/// penalty `S_λ = Σ_b S_b`. Differentiating the conditional mean shift in
/// `ρ_b = log λ_b` scales *this* block (`S_b → e^{t} S_b`) while holding the
/// others fixed; that is exactly `∂S_λ/∂ρ_b = S_b` at the fitted point.
#[derive(Debug, Clone)]
pub struct RhoPenaltyComponent {
    /// The `k × k` component penalty `S_b` at its fitted scale (the term's
    /// contribution to `S_λ`, i.e. `λ_b` times its unit penalty).
    pub s_component: Array2<f64>,
}

/// Step in `ρ = log λ` for the deterministic central-difference curvature of the
/// conditional Lawley mean shift `Δε(ρ)`. `Δε` is a smooth *deterministic*
/// function of ρ (ρ enters only through `S_λ = Σ_b e^{ρ_b} S_b` inside the SPD
/// information `J`), so a central difference of the known scalar is exact to its
/// truncation order — no statistical approximation is introduced (only numerical
/// differentiation of a closed form, verified against the analytic Gaussian-zero
/// anchor and a Richardson refinement in tests). `0.05` resolves the `O(n⁻¹)`
/// shift's curvature comfortably while keeping the `O(h²)` truncation far below
/// the `O(n⁻²)` Bartlett target.
const RHO_VARIATION_STEP: f64 = 0.05;

/// The ρ̂-sampling-variation contribution to the penalized-null Bartlett mean
/// shift (#939 deliverable 2, the genuinely-new penalized theory piece).
///
/// The conditional (fixed-λ) Lawley shift `Δε(ρ)` from [`lawley_lr_mean_shift`]
/// is `E[W | λ]` — the LR mean with the smoothing parameter *held at its fitted
/// value*. When λ is **estimated**, ρ̂ = log λ̂ carries its own sampling
/// variation, and `E[W]` picks up the extra second-order term of a delta-method
/// expansion of `W(ρ̂)` about the population ρ₀:
///
/// ```text
/// E[W(ρ̂)] = Δε(ρ₀) + ½ Σ_{b,b'} (∂²Δε/∂ρ_b ∂ρ_{b'}) · Cov(ρ̂_b, ρ̂_{b'}) + O(·).
/// ```
///
/// Both ingredients are exact and already in the engine:
///
/// * `∂²Δε/∂ρ_b ∂ρ_{b'}` — the curvature of the *deterministic* conditional
///   shift in the log-smoothing parameters. ρ enters `Δε` only through
///   `S_λ = Σ_b e^{ρ_b} S_b` folded into the SPD information `J`, so scaling each
///   `components[b].s_component` by `e^{±h}` and central-differencing the known
///   scalar `Δε` recovers the Hessian exactly (to `O(h²)`, well inside the
///   `O(n⁻²)` Bartlett target). Cross terms use the symmetric 4-point stencil.
/// * `Cov(ρ̂)` — the inverse REML/LAML **outer Hessian** (the #740 quantity the
///   solver already maintains), passed as `rho_cov` (`m × m` for `m`
///   smoothing parameters). This is the sampling covariance of ρ̂.
///
/// Returns the **total** mean shift `Δε(ρ̂) = Δε_conditional + ½·tr(HᵨᵨCov)`. The
/// caller forms the Bartlett factor `c = 1 + Δε(ρ̂)/d` exactly as for the
/// conditional shift; the difference from the conditional factor is the size
/// correction attributable specifically to ρ̂-variation.
///
/// `components` must have one entry per row/column of `rho_cov`; `penalty` is the
/// total fitted `S_λ` (the conditional anchor). Errors on shape mismatch or a
/// non-finite curvature.
pub fn lawley_lr_mean_shift_with_rho_variation(
    x: ArrayView2<'_, f64>,
    kappas: &[RowKappas],
    penalty: ArrayView2<'_, f64>,
    tested: std::ops::Range<usize>,
    components: &[RhoPenaltyComponent],
    rho_cov: ArrayView2<'_, f64>,
) -> Result<f64, String> {
    let k = x.ncols();
    let m = components.len();
    if m == 0 {
        return Err(
            "lawley_lr_mean_shift_with_rho_variation: no smoothing-parameter components"
                .to_string(),
        );
    }
    if rho_cov.nrows() != m || rho_cov.ncols() != m {
        return Err(format!(
            "lawley_lr_mean_shift_with_rho_variation: rho_cov is {}×{}, expected {m}×{m}",
            rho_cov.nrows(),
            rho_cov.ncols()
        ));
    }
    for b in 0..m {
        for c in 0..m {
            let v_bc = rho_cov[[b, c]];
            if !v_bc.is_finite() {
                return Err(format!(
                    "lawley_lr_mean_shift_with_rho_variation: rho_cov[{b},{c}] is not finite"
                ));
            }
            let v_cb = rho_cov[[c, b]];
            let tol = 1e-10 * (1.0 + v_bc.abs().max(v_cb.abs()));
            if (v_bc - v_cb).abs() > tol {
                return Err(format!(
                    "lawley_lr_mean_shift_with_rho_variation: rho_cov must be symmetric; \
                     entries [{b},{c}]={v_bc} and [{c},{b}]={v_cb} differ"
                ));
            }
        }
    }
    if penalty.nrows() != k || penalty.ncols() != k {
        return Err(format!(
            "lawley_lr_mean_shift_with_rho_variation: penalty is {}×{}, expected {k}×{k}",
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    for (b, comp) in components.iter().enumerate() {
        if comp.s_component.nrows() != k || comp.s_component.ncols() != k {
            return Err(format!(
                "lawley_lr_mean_shift_with_rho_variation: component {b} is {}×{}, expected {k}×{k}",
                comp.s_component.nrows(),
                comp.s_component.ncols()
            ));
        }
    }

    // Conditional shift Δε(ρ̂): the existing fixed-λ Lawley mean shift at the
    // fitted total penalty.
    let conditional = lawley_lr_mean_shift(x, kappas, Some(penalty), tested.clone())?;

    // Δε at a perturbed log-smoothing vector: total penalty
    // S_λ(t) = Σ_b e^{t_b} S_b. With S_λ = Σ_b S_b at the fitted point (t = 0),
    // a shift t_b replaces the b-th block S_b by (e^{t_b} − 1)·S_b added on top.
    let shift_at = |steps: &[(usize, f64)]| -> Result<f64, String> {
        let mut s = penalty.to_owned();
        for &(b, t) in steps {
            // ∂S_λ/∂ρ_b = S_b, so e^{t}·S_b replaces S_b: add (e^{t} − 1)·S_b.
            let scale = t.exp() - 1.0;
            s.scaled_add(scale, &components[b].s_component);
        }
        lawley_lr_mean_shift(x, kappas, Some(s.view()), tested.clone())
    };

    let h = RHO_VARIATION_STEP;
    // Hessian of Δε in ρ by symmetric finite differences of the deterministic
    // scalar. Diagonal: standard 3-point second derivative; off-diagonal: the
    // 4-point mixed-partial stencil.
    let mut quad = 0.0; // ½ Σ_{b,b'} H_{bb'} Cov_{bb'}
    let base = conditional;
    for b in 0..m {
        let fp = shift_at(&[(b, h)])?;
        let fm = shift_at(&[(b, -h)])?;
        let hbb = (fp - 2.0 * base + fm) / (h * h);
        if !hbb.is_finite() {
            return Err(format!(
                "lawley_lr_mean_shift_with_rho_variation: non-finite curvature H[{b},{b}]"
            ));
        }
        quad += 0.5 * hbb * rho_cov[[b, b]];
        for c in (b + 1)..m {
            let fpp = shift_at(&[(b, h), (c, h)])?;
            let fpm = shift_at(&[(b, h), (c, -h)])?;
            let fmp = shift_at(&[(b, -h), (c, h)])?;
            let fmm = shift_at(&[(b, -h), (c, -h)])?;
            let hbc = (fpp - fpm - fmp + fmm) / (4.0 * h * h);
            if !hbc.is_finite() {
                return Err(format!(
                    "lawley_lr_mean_shift_with_rho_variation: non-finite curvature H[{b},{c}]"
                ));
            }
            // Symmetric covariance: H_{bc}=H_{cb}, Cov_{bc}=Cov_{cb} ⇒ two equal
            // off-diagonal contributions, i.e. ½·2·H_{bc}Cov_{bc} = H_{bc}Cov_{bc}.
            quad += hbc * rho_cov[[b, c]];
        }
    }

    let total = conditional + quad;
    if !total.is_finite() {
        return Err(format!(
            "lawley_lr_mean_shift_with_rho_variation: non-finite total shift \
             (conditional={conditional}, rho-variation={quad})"
        ));
    }
    Ok(total)
}

/// The Lawley LR Bartlett factor `c = E[W]/d = 1 + Δε(ρ̂)/d` for an **estimated**
/// smoothing parameter — the ρ̂-variation analogue of [`lawley_lr_bartlett_factor`].
///
/// [`lawley_lr_bartlett_factor`] folds the penalty in at a *fixed* `λ`, giving the
/// conditional factor `1 + Δε(ρ₀)/d`. When `λ` is estimated, ρ̂ = log λ̂ carries
/// its own sampling variation and `E[W]` picks up the extra delta-method term
/// (#939 deliverable 2, the genuinely-new penalized piece);
/// [`lawley_lr_mean_shift_with_rho_variation`] assembles the resulting **total**
/// mean shift `Δε(ρ̂)` from the conditional shift's ρ-curvature and the inverse
/// REML outer Hessian `Cov(ρ̂)` (the #740 quantity). This function is the
/// single-call factor entry point a live consumer wires symmetrically with the
/// fixed-λ factor: it performs the same `c = E[W]/d` reduction and the same
/// degeneracy guards, so the call site never re-derives `1 + Δε/d` (and never
/// re-implements the positivity / finiteness checks) by hand.
///
/// Returns the factor against a `d = ref_df` reference; `penalty` is the total
/// fitted `S_λ`, `components`/`rho_cov` are the per-smoothing-parameter penalty
/// blocks and their sampling covariance, exactly as
/// [`lawley_lr_mean_shift_with_rho_variation`] takes them. Errors on a degenerate
/// reference df, a non-finite/degenerate mean, or any error from the underlying
/// shift assembly.
pub fn lawley_lr_bartlett_factor_with_rho_variation(
    x: ArrayView2<'_, f64>,
    kappas: &[RowKappas],
    penalty: ArrayView2<'_, f64>,
    tested: std::ops::Range<usize>,
    components: &[RhoPenaltyComponent],
    rho_cov: ArrayView2<'_, f64>,
    ref_df: f64,
) -> Result<f64, String> {
    if !(ref_df.is_finite() && ref_df > 0.0) {
        return Err(format!(
            "lawley_lr_bartlett_factor_with_rho_variation: reference df must be finite and positive; got {ref_df}"
        ));
    }
    let shift =
        lawley_lr_mean_shift_with_rho_variation(x, kappas, penalty, tested, components, rho_cov)?;
    let mean_w = ref_df + shift;
    let factor = crate::inference::higher_order::bartlett_factor_from_mean(mean_w, ref_df)
        .ok_or_else(|| {
            format!(
                "lawley_lr_bartlett_factor_with_rho_variation: degenerate mean {mean_w} \
                 (Δε(ρ̂) = {shift}, d = {ref_df})"
            )
        })?;
    if !(factor.is_finite() && factor > 0.0) {
        return Err(format!(
            "lawley_lr_bartlett_factor_with_rho_variation: degenerate factor {factor} \
             (Δε(ρ̂) = {shift}, d = {ref_df})"
        ));
    }
    Ok(factor)
}

/// Expected jets for a known-scale GLM family/link pair at linear predictor
/// `eta`, when the pair has an exact closed-form jet constructor. Returns
/// `None` for pairs whose cumulant jets are not derived yet — the consumer
/// then reports first-order inference only (#939).
pub fn known_scale_expected_jets(
    family: &gam_spec::LikelihoodSpec,
    eta: f64,
) -> Option<RowExpectedJets> {
    known_scale_expected_jets_with_dispersion(family, eta, 1.0)
}

/// Expected jets for a GLM family/link pair at linear predictor `eta` with an
/// explicit dispersion `φ` (Gaussian σ², Gamma φ = 1/shape; 1 for the
/// scale-free Poisson/Binomial). Returns `None` for pairs whose cumulant jets
/// are not derived yet — the consumer then reports first-order inference only
/// (#939). This is the dispersion-carrying sibling of
/// [`known_scale_expected_jets`]; the per-term LR Bartlett path (#1063) needs it
/// for the estimated-scale Gaussian/Gamma families whose ε depends on φ.
pub fn known_scale_expected_jets_with_dispersion(
    family: &gam_spec::LikelihoodSpec,
    eta: f64,
    dispersion: f64,
) -> Option<RowExpectedJets> {
    use gam_spec::{InverseLink, ResponseFamily, StandardLink};
    match (&family.response, &family.link) {
        (ResponseFamily::Poisson, InverseLink::Standard(StandardLink::Log)) => {
            Some(RowExpectedJets::poisson_log(eta))
        }
        (ResponseFamily::Binomial, InverseLink::Standard(StandardLink::Logit)) => {
            Some(RowExpectedJets::binomial_logit(eta))
        }
        (ResponseFamily::Gaussian, InverseLink::Standard(StandardLink::Identity)) => {
            (dispersion.is_finite() && dispersion > 0.0)
                .then(|| RowExpectedJets::gaussian_identity(dispersion))
        }
        (ResponseFamily::Gamma, InverseLink::Standard(StandardLink::Log)) => {
            (dispersion.is_finite() && dispersion > 0.0)
                .then(|| RowExpectedJets::gamma_log(eta, dispersion))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Raw six-index Lawley sum (eq. 4 of the module docs) on dense arrays:
    /// the independent oracle for the row-pair reduction. `κ^{rs}` is the
    /// inverse of `[κ_rs]`, i.e. MINUS the inverse information.
    fn lawley_epsilon_index_oracle(
        x: &Array2<f64>,
        kappas: &[RowKappas],
        penalty: Option<&Array2<f64>>,
    ) -> f64 {
        let n = x.nrows();
        let k = x.ncols();
        // κ_rs (with penalty subtracted: penalty adds −S to κ_rs).
        let mut kappa2 = Array2::<f64>::zeros((k, k));
        for i in 0..n {
            for r in 0..k {
                for s in 0..k {
                    kappa2[[r, s]] += kappas[i].k2 * x[[i, r]] * x[[i, s]];
                }
            }
        }
        if let Some(s_pen) = penalty {
            kappa2 -= s_pen;
        }
        // κ^{rs}: dense inverse via faer.
        let j_view = FaerArrayView::new(&kappa2);
        let factor = factorize_symmetricwith_fallback(j_view.as_ref(), faer::Side::Lower)
            .expect("oracle κ_rs factorization");
        let kappa_up = FactorizedSystem::solvemulti(&factor, &Array2::<f64>::eye(k))
            .expect("oracle κ_rs inverse");

        // Dense symmetric 3- and 4-index arrays as closures over row sums.
        let arr3 = |weights: &dyn Fn(usize) -> f64, r: usize, s: usize, t: usize| -> f64 {
            (0..n)
                .map(|i| weights(i) * x[[i, r]] * x[[i, s]] * x[[i, t]])
                .sum()
        };
        let arr4 =
            |weights: &dyn Fn(usize) -> f64, r: usize, s: usize, t: usize, u: usize| -> f64 {
                (0..n)
                    .map(|i| weights(i) * x[[i, r]] * x[[i, s]] * x[[i, t]] * x[[i, u]])
                    .sum()
            };
        let w_k3 = |i: usize| kappas[i].k3;
        let w_k21 = |i: usize| kappas[i].k2_1;
        let w_k4 = |i: usize| kappas[i].k4;
        let w_k31 = |i: usize| kappas[i].k3_1;
        let w_k211 = |i: usize| kappas[i].k2_11;

        let mut lambda4 = 0.0;
        for r in 0..k {
            for s in 0..k {
                for t in 0..k {
                    for u in 0..k {
                        let braces = arr4(&w_k4, r, s, t, u) / 4.0 - arr4(&w_k31, r, s, t, u)
                            + arr4(&w_k211, r, t, s, u);
                        lambda4 += kappa_up[[r, s]] * kappa_up[[t, u]] * braces;
                    }
                }
            }
        }
        let mut lambda6 = 0.0;
        for r in 0..k {
            for s in 0..k {
                for t in 0..k {
                    for u in 0..k {
                        for v in 0..k {
                            for w in 0..k {
                                let braces = arr3(&w_k3, r, t, v)
                                    * (arr3(&w_k3, s, u, w) / 6.0 - arr3(&w_k21, s, w, u))
                                    + arr3(&w_k3, r, t, u)
                                        * (arr3(&w_k3, s, v, w) / 4.0 - arr3(&w_k21, s, w, v))
                                    + arr3(&w_k21, r, t, v) * arr3(&w_k21, s, w, u)
                                    + arr3(&w_k21, r, t, u) * arr3(&w_k21, s, w, v);
                                lambda6 +=
                                    kappa_up[[r, s]] * kappa_up[[t, u]] * kappa_up[[v, w]] * braces;
                            }
                        }
                    }
                }
            }
        }
        lambda4 - lambda6
    }

    fn intercept_design(n: usize) -> Array2<f64> {
        Array2::<f64>::ones((n, 1))
    }

    /// ψ(n) at integer n, exactly: ψ(n) = −γ + Σ_{j=1}^{n−1} 1/j.
    fn digamma_integer(n: usize) -> f64 {
        const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
        -EULER_GAMMA + (1..n).map(|j| 1.0 / j as f64).sum::<f64>()
    }

    #[test]
    fn exponential_intercept_matches_exact_digamma_expansion() {
        // y_i ~ Exponential(mean μ), log link, intercept-only. Exact null mean:
        // E[W] = 2n(log n − ψ(n)) = 1 + 1/(6n) + O(n⁻³) (ȳ ~ Gamma(n, μ/n)).
        let eta = 0.4;
        let mut residual_prev = f64::INFINITY;
        for &n in &[8usize, 16, 32] {
            let jets = RowExpectedJets::gamma_log(eta, 1.0);
            let kappas = vec![jets.kappas().expect("exponential kappas"); n];
            let x = intercept_design(n);
            let eps = lawley_epsilon(x.view(), &kappas, None).expect("ε");
            let analytic = 1.0 / (6.0 * n as f64);
            assert!(
                (eps - analytic).abs() < 1e-12,
                "n={n}: ε={eps} vs analytic 1/(6n)={analytic}"
            );
            let exact_mean = 2.0 * n as f64 * ((n as f64).ln() - digamma_integer(n));
            let residual = (exact_mean - 1.0 - eps).abs();
            assert!(
                residual < 0.6 / (n * n) as f64,
                "n={n}: |E[W] − 1 − ε| = {residual} is not O(n⁻²)"
            );
            assert!(
                residual < residual_prev,
                "n={n}: residual {residual} did not shrink from {residual_prev}"
            );
            residual_prev = residual;
        }
    }

    /// DELIVERABLE (2) — penalty deterministic-shift term is consumed. The
    /// Lawley ε folds `S_λ` into the information `J = X'WX + S_λ`
    /// ([`lawley_epsilon`]); adding the penalty therefore moves ε on any family
    /// whose `ε ≠ 0`. This regression proves the penalty arm is live (not
    /// dropped) and that a larger λ shrinks |ε| monotonically — the penalty
    /// stiffens the information `J = X'WX + S_λ`, which moves the finite-sample
    /// LR mean shift — exactly the penalized-null behavior #939 deliverable (2)
    /// requires.
    ///
    /// (The ρ̂-variation term — the extra deterministic shift from λ̂ being
    /// *estimated* rather than fixed — is NOT in this conditional-on-λ̂ factor;
    /// it is the genuinely-new theory piece the issue flags, validated via the
    /// null-bootstrap arm. This test pins the penalty deterministic-shift only.)
    #[test]
    fn penalty_shift_term_is_consumed() {
        // Poisson/log, a 2-column design (intercept + a centered covariate) so
        // the penalty on the second column actually couples into ε.
        let n = 40usize;
        let eta = 0.2_f64;
        let jets = RowExpectedJets::poisson_log(eta);
        let kappas = vec![jets.kappas().expect("poisson kappas"); n];
        let mut x = Array2::<f64>::ones((n, 2));
        for i in 0..n {
            // Centered covariate in the second column.
            x[[i, 1]] = (i as f64) / (n as f64) - 0.5;
        }
        let eps_unpen = lawley_epsilon(x.view(), &kappas, None).expect("ε unpenalized");

        // A ridge penalty on the second (smooth-like) column only. ε must depend
        // on λ — if S were dropped, every λ would give the unpenalized value.
        let mut distinct = std::collections::BTreeSet::new();
        for &lambda in &[0.5_f64, 2.0, 8.0, 32.0] {
            let mut s = Array2::<f64>::zeros((2, 2));
            s[[1, 1]] = lambda;
            let eps_pen = lawley_epsilon(x.view(), &kappas, Some(s.view())).expect("ε penalized");
            // The penalty MUST change ε (proves S is consumed, deliverable 2).
            assert!(
                (eps_pen - eps_unpen).abs() > 1e-9,
                "λ={lambda}: penalty did not move ε ({eps_pen} vs {eps_unpen}) — S is being dropped"
            );
            // As λ → ∞ the penalized column is frozen out; ε must stay finite.
            assert!(
                eps_pen.is_finite(),
                "λ={lambda}: ε must be finite, got {eps_pen}"
            );
            distinct.insert((eps_pen * 1e9) as i64);
        }
        // Different λ give genuinely different ε (S enters the information, not a
        // no-op): at least three of the four ridge strengths are distinct.
        assert!(
            distinct.len() >= 3,
            "ε must vary with λ; got {} distinct values",
            distinct.len()
        );
    }

    #[test]
    fn poisson_intercept_matches_exact_pmf_mean() {
        // y_i ~ Poisson(λ), log link, intercept-only: ε = 1/(6nλ) classically;
        // E[W] computed exactly by pmf summation over S = Σy ~ Poisson(nλ).
        let lambda: f64 = 1.7;
        for &n in &[20usize, 40] {
            let jets = RowExpectedJets::poisson_log(lambda.ln());
            let kappas = vec![jets.kappas().expect("poisson kappas"); n];
            let x = intercept_design(n);
            let eps = lawley_epsilon(x.view(), &kappas, None).expect("ε");
            let analytic = 1.0 / (6.0 * n as f64 * lambda);
            assert!(
                (eps - analytic).abs() < 1e-12,
                "n={n}: ε={eps} vs analytic 1/(6nλ)={analytic}"
            );
            let total_rate = n as f64 * lambda;
            let mut pmf = (-total_rate).exp();
            let mut exact_mean = 0.0;
            let s_max = (total_rate + 60.0 * total_rate.sqrt()).ceil() as usize;
            for s in 0..=s_max {
                if s > 0 {
                    pmf *= total_rate / s as f64;
                }
                let s_f = s as f64;
                let w = if s == 0 {
                    2.0 * total_rate
                } else {
                    2.0 * (total_rate - s_f + s_f * (s_f / total_rate).ln())
                };
                exact_mean += pmf * w;
            }
            let residual = (exact_mean - 1.0 - eps).abs();
            assert!(
                residual < 0.7 / (n * n) as f64,
                "n={n}: |E[W] − 1 − ε| = {residual} is not O(n⁻²)"
            );
        }
    }

    /// CERTIFICATION FIXTURE 1 (#939, Gaussian known variance): every third and
    /// fourth expected cumulant and every η-derivative of the curvature vanish
    /// (ℓᵢ = −½(yᵢ−ηᵢ)²/φ is exactly quadratic), so ε_k = ε_{k−q} = 0 and the
    /// Lawley LR Bartlett factor is exactly 1 at every n. The χ²_q reference is
    /// exactly calibrated only in the unpenalized quadratic model; with a penalty
    /// the statistic is a weighted χ² form, e.g. one-parameter ridge gives
    /// `(n / (n + λ))χ²₁`.
    #[test]
    fn gaussian_known_variance_lr_factor_is_exactly_one() {
        let n = 20;
        let k = 3;
        let mut x = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let z = i as f64 / n as f64;
            x[[i, 0]] = 1.0;
            x[[i, 1]] = (5.0 * z).sin();
            x[[i, 2]] = z - 0.5;
        }
        let kappas = vec![
            RowExpectedJets::gaussian_identity(1.7)
                .kappas()
                .expect("gaussian kappas");
            n
        ];
        let s_pen = Array2::<f64>::eye(k) * 0.4;
        for q in [1usize, 2] {
            let shift = lawley_lr_mean_shift(x.view(), &kappas, Some(s_pen.view()), k - q..k)
                .expect("shift");
            assert!(
                shift.abs() < 1e-13,
                "Gaussian known-variance Δε must be 0; got {shift}"
            );
            let c = lawley_lr_bartlett_factor(
                x.view(),
                &kappas,
                Some(s_pen.view()),
                k - q..k,
                q as f64,
            )
            .expect("factor");
            assert!(
                (c - 1.0).abs() < 1e-13,
                "Gaussian known-variance Bartlett factor must be exactly 1; got {c}"
            );
        }
    }

    /// CERTIFICATION FIXTURE 2 (#939, Exponential rate, scalar MLE): Lawley's
    /// analytic second-order Bartlett factor is `c = 1 + 1/(6n)`.
    ///
    /// Derivation. yᵢ ~ Exp(rate θ), ℓ(θ) = n·log θ − θ·Σyᵢ, θ̂ = 1/ȳ. The LR
    /// statistic for H₀: θ = θ₀ at the truth is
    ///   W = 2[ℓ(θ̂) − ℓ(θ₀)] = 2n(θ₀ȳ − 1 − log(θ₀ȳ)).
    /// With T = θ₀·Σyᵢ ~ Gamma(n, 1): E[T] = n, E[log T] = ψ(n), so
    ///   E[W] = 2(E[T] − n) + 2n(log n − E[log T]) = 2n(log n − ψ(n))   (exact).
    /// The digamma expansion ψ(n) = log n − 1/(2n) − 1/(12n²) + O(n⁻⁴) gives
    ///   E[W] = 1 + 1/(6n) + O(n⁻³).
    /// The tested block is the whole (intercept-only) design — a fully
    /// specified null, ε₀ = 0 — and the factor must match both the analytic
    /// 1 + 1/(6n) (to machine precision) and the exact digamma mean through the
    /// expected higher-order remainder.
    #[test]
    fn exponential_rate_lr_factor_is_one_plus_one_sixth_n() {
        let eta = -0.7; // ε is reparametrization-invariant; any η works.
        for &n in &[8usize, 16, 32] {
            let jets = RowExpectedJets::gamma_log(eta, 1.0);
            let kappas = vec![jets.kappas().expect("exponential kappas"); n];
            let x = intercept_design(n);
            let c = lawley_lr_bartlett_factor(x.view(), &kappas, None, 0..1, 1.0).expect("factor");
            let analytic = 1.0 + 1.0 / (6.0 * n as f64);
            assert!(
                (c - analytic).abs() < 1e-12,
                "n={n}: factor {c} vs analytic 1 + 1/(6n) = {analytic}"
            );
            let exact_mean = 2.0 * n as f64 * ((n as f64).ln() - digamma_integer(n));
            assert!(
                (exact_mean - c).abs() < 0.6 / (n * n) as f64,
                "n={n}: |E[W] − c| = {} is not O(n⁻²)",
                (exact_mean - c).abs()
            );
        }
    }

    /// CERTIFICATION FIXTURE 3 (#939, Bernoulli/logit, scalar MLE): analytic
    /// Lawley shift `ε = (1 − u)/(6nu)`, `u = μ(1−μ)`, certified against the
    /// exact binomial pmf mean of W.
    ///
    /// Hand derivation of ε from the row-pair form (module docs), intercept-only
    /// design (h_i = E_ij = 1/(nu)), canonical link (κ₃ = κ₂′, κ₄ = κ₃′), with
    /// u = μ(1−μ), u′ = u(1−2μ), u″ = u(1−2μ)² − 2u² = u(1−6u) (since
    /// (1−2μ)² = 1−4u):
    ///   κ₂ = −u, κ₂′ = κ₃ = −u′, κ₂″ = κ₃′ = κ₄ = −u″
    ///   a_i  = κ₄/4 − κ₃′ + κ₂″ = −u″/4
    ///   λ₄   = n·(−u″/4)·(1/(nu))² = −u″/(4nu²)
    ///   b6   = κ₃²/6 − κ₃κ₂′ + κ₂′² = u′²/6,   b4 = u′²/4
    ///   λ₆   = −n²·(1/(nu))³·(u′²/6 + u′²/4) = −5u′²/(12nu³)
    ///   ε    = λ₄ − λ₆ = [−3(1−6u) + 5(1−4u)]/(12nu) = (1 − u)/(6nu).
    /// Cross-check: Poisson has u = u′ = u″ = μ ⇒ ε = (5/12 − 1/4)/(nμ) =
    /// 1/(6nμ), the classical value asserted in the Poisson fixture above.
    /// Exact certification: S = Σyᵢ ~ Binomial(n, μ₀), W(S) = 2[S·log(S/(nμ₀))
    /// + (n−S)·log((n−S)/(n(1−μ₀)))] (0·log 0 = 0); |E[W] − 1 − ε| must be
    /// O(n⁻²) (numerically ≈ 1.5/n² at μ₀ = 0.3) and shrink with n.
    #[test]
    fn bernoulli_logit_intercept_factor_matches_exact_pmf_mean() {
        let mu: f64 = 0.3;
        let u = mu * (1.0 - mu);
        let eta = (mu / (1.0 - mu)).ln();
        let mut residual_prev = f64::INFINITY;
        for &n in &[24usize, 48, 96] {
            let jets = RowExpectedJets::binomial_logit(eta);
            let kappas = vec![jets.kappas().expect("bernoulli kappas"); n];
            let x = intercept_design(n);
            let shift = lawley_lr_mean_shift(x.view(), &kappas, None, 0..1).expect("Δε");
            let analytic = (1.0 - u) / (6.0 * n as f64 * u);
            assert!(
                (shift - analytic).abs() < 1e-12,
                "n={n}: Δε = {shift} vs analytic (1−u)/(6nu) = {analytic}"
            );
            let c = lawley_lr_bartlett_factor(x.view(), &kappas, None, 0..1, 1.0).expect("factor");
            assert!(
                (c - (1.0 + analytic)).abs() < 1e-12,
                "n={n}: factor {c} vs 1 + ε = {}",
                1.0 + analytic
            );
            // Exact E[W] by binomial pmf summation.
            let nf = n as f64;
            let mut pmf = (1.0 - mu).powi(n as i32); // P(S = 0)
            let mut exact_mean = 0.0;
            for s in 0..=n {
                if s > 0 {
                    pmf *= mu / (1.0 - mu) * (n - s + 1) as f64 / s as f64;
                }
                let s_f = s as f64;
                let t1 = if s == 0 {
                    0.0
                } else {
                    s_f * (s_f / (nf * mu)).ln()
                };
                let t2 = if s == n {
                    0.0
                } else {
                    (nf - s_f) * ((nf - s_f) / (nf * (1.0 - mu))).ln()
                };
                exact_mean += pmf * 2.0 * (t1 + t2);
            }
            let residual = (exact_mean - 1.0 - shift).abs();
            assert!(
                residual < 2.5 / (n * n) as f64,
                "n={n}: |E[W] − 1 − ε| = {residual} is not O(n⁻²)"
            );
            assert!(
                residual < residual_prev,
                "n={n}: residual {residual} did not shrink from {residual_prev}"
            );
            residual_prev = residual;
        }
    }

    /// The mean shift is exactly `ε_full − ε_nuisance` with the tested columns
    /// (and the matching penalty rows/columns) dropped from the nuisance model.
    #[test]
    fn mean_shift_is_full_minus_nuisance_epsilon() {
        let n = 19;
        let mut x = Array2::<f64>::zeros((n, 2));
        let mut kappas = Vec::with_capacity(n);
        for i in 0..n {
            let z = i as f64 / n as f64;
            x[[i, 0]] = 1.0;
            x[[i, 1]] = z - 0.5;
            let eta = 0.3 - 0.8 * (z - 0.5);
            kappas.push(
                RowExpectedJets::binomial_logit(eta)
                    .kappas()
                    .expect("binomial kappas"),
            );
        }
        let mut s_pen = Array2::<f64>::zeros((2, 2));
        s_pen[[1, 1]] = 0.6;
        let shift =
            lawley_lr_mean_shift(x.view(), &kappas, Some(s_pen.view()), 1..2).expect("shift");
        let eps_full = lawley_epsilon(x.view(), &kappas, Some(s_pen.view())).expect("ε_full");
        let x_null = x.slice(ndarray::s![.., 0..1]).to_owned();
        let s_null = s_pen.slice(ndarray::s![0..1, 0..1]).to_owned();
        let eps_null = lawley_epsilon(x_null.view(), &kappas, Some(s_null.view())).expect("ε_null");
        assert!(
            (shift - (eps_full - eps_null)).abs() < 1e-14,
            "Δε = {shift} must equal ε_full − ε_null = {}",
            eps_full - eps_null
        );
        // Weighted rows: doubling every weight must equal doubling n by row
        // duplication — certified through the linear scaling of the cumulants.
        let kappas_w: Vec<RowKappas> = kappas.iter().map(|r| r.weighted(2.0)).collect();
        let mut x2 = Array2::<f64>::zeros((2 * n, 2));
        let mut kappas2 = Vec::with_capacity(2 * n);
        for i in 0..n {
            for rep in 0..2 {
                let row = 2 * i + rep;
                x2[[row, 0]] = x[[i, 0]];
                x2[[row, 1]] = x[[i, 1]];
                kappas2.push(kappas[i]);
            }
        }
        let shift_w = lawley_lr_mean_shift(x.view(), &kappas_w, Some(s_pen.view()), 1..2)
            .expect("weighted shift");
        let shift_dup = lawley_lr_mean_shift(x2.view(), &kappas2, Some(s_pen.view()), 1..2)
            .expect("duplicated shift");
        assert!(
            (shift_w - shift_dup).abs() < 1e-12 * (1.0 + shift_dup.abs()),
            "weight-2 rows ({shift_w}) must equal duplicated rows ({shift_dup})"
        );
    }

    #[test]
    fn row_pair_reduction_matches_index_oracle() {
        // Non-canonical rows (gamma/log at varying η) on a k=3 design: the
        // production row-pair form must equal the raw six-index Lawley sum.
        let n = 17;
        let k = 3;
        let mut x = Array2::<f64>::zeros((n, k));
        let mut kappas = Vec::with_capacity(n);
        for i in 0..n {
            let z = i as f64 / n as f64;
            x[[i, 0]] = 1.0;
            x[[i, 1]] = (7.3 * z).sin();
            x[[i, 2]] = z * z - 0.4;
            let eta = 0.2 + 0.5 * x[[i, 1]] - 0.3 * x[[i, 2]];
            kappas.push(
                RowExpectedJets::gamma_log(eta, 1.3)
                    .kappas()
                    .expect("gamma kappas"),
            );
        }
        let fast = lawley_epsilon(x.view(), &kappas, None).expect("hat form");
        let oracle = lawley_epsilon_index_oracle(&x, &kappas, None);
        assert!(
            (fast - oracle).abs() < 1e-10 * (1.0 + oracle.abs()),
            "row-pair ε={fast} vs index-form ε={oracle}"
        );

        // And with a penalty folded into the information (penalized Lawley).
        let mut s_pen = Array2::<f64>::eye(k);
        s_pen[[0, 0]] = 0.0; // unpenalized intercept
        s_pen *= 0.8;
        let fast_pen = lawley_epsilon(x.view(), &kappas, Some(s_pen.view())).expect("hat form");
        let oracle_pen = lawley_epsilon_index_oracle(&x, &kappas, Some(&s_pen));
        assert!(
            (fast_pen - oracle_pen).abs() < 1e-10 * (1.0 + oracle_pen.abs()),
            "penalized row-pair ε={fast_pen} vs index-form ε={oracle_pen}"
        );
        assert!(
            (fast_pen - fast).abs() > 1e-6,
            "penalty must move ε (got {fast} → {fast_pen})"
        );
    }

    #[test]
    fn canonical_links_collapse_the_mixed_arrays() {
        // Canonical links have c′ = c″ = 0, so κ₃ = κ₂' and κ₄ = κ₃' — the
        // derivative ("joint") arrays coincide with the pure ones and the
        // Bartlett ingredients reduce to classical canonical-GLM form.
        for eta in [-1.3, 0.0, 0.7] {
            for jets in [
                RowExpectedJets::poisson_log(eta),
                RowExpectedJets::binomial_logit(eta),
            ] {
                let kappas = jets.kappas().expect("canonical kappas");
                assert!(
                    (kappas.k3 - kappas.k2_1).abs() < 1e-13 * (1.0 + kappas.k3.abs()),
                    "canonical link must satisfy κ₃ = κ₂' (η={eta}): {kappas:?}"
                );
                assert!(
                    (kappas.k4 - kappas.k3_1).abs() < 1e-13 * (1.0 + kappas.k4.abs()),
                    "canonical link must satisfy κ₄ = κ₃' (η={eta}): {kappas:?}"
                );
            }
        }
    }

    #[test]
    fn gaussian_identity_needs_no_correction_even_penalized() {
        // Gaussian/identity: all third/fourth-order and derivative arrays
        // vanish, so ε = 0 exactly — including with a penalty (the LR of a
        // penalized-quadratic model is exactly pivotal at fixed λ).
        let n = 12;
        let jets = RowExpectedJets::gaussian_identity(2.3);
        let kappas = vec![jets.kappas().expect("gaussian kappas"); n];
        let mut x = Array2::<f64>::ones((n, 2));
        for i in 0..n {
            x[[i, 1]] = i as f64 - 5.0;
        }
        let s_pen = Array2::<f64>::eye(2) * 0.5;
        let eps = lawley_epsilon(x.view(), &kappas, Some(s_pen.view())).expect("ε");
        assert!(
            eps.abs() < 1e-14,
            "Gaussian-identity ε must be 0; got {eps}"
        );
    }

    /// DELIVERABLE (2), ρ̂-variation anchor — Gaussian known variance: `Δε ≡ 0`
    /// at every λ, so its ρ-curvature is identically zero and the ρ̂-variation
    /// correction is exactly 0 regardless of `Cov(ρ̂)`. The total shift must
    /// equal the conditional shift (both zero).
    #[test]
    fn rho_variation_correction_is_zero_for_gaussian() {
        let n = 16usize;
        let jets = RowExpectedJets::gaussian_identity(1.3);
        let kappas = vec![jets.kappas().expect("gaussian kappas"); n];
        let mut x = Array2::<f64>::ones((n, 2));
        for i in 0..n {
            x[[i, 1]] = i as f64 / n as f64 - 0.5;
        }
        // One smoothing parameter on the second column (a ridge of strength λ = 2).
        let mut s_comp = Array2::<f64>::zeros((2, 2));
        s_comp[[1, 1]] = 2.0;
        let penalty = s_comp.clone();
        let components = vec![RhoPenaltyComponent {
            s_component: s_comp,
        }];
        // A large ρ-variance: with zero curvature it still contributes nothing.
        let rho_cov = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
        let total = lawley_lr_mean_shift_with_rho_variation(
            x.view(),
            &kappas,
            penalty.view(),
            1..2,
            &components,
            rho_cov.view(),
        )
        .expect("rho-variation shift");
        assert!(
            total.abs() < 1e-12,
            "Gaussian ρ̂-variation total shift must be 0; got {total}"
        );
    }

    /// DELIVERABLE (2), ρ̂-variation core — the correction is exactly
    /// `½·Hᵨᵨ·Var(ρ̂)` for a single smoothing parameter, where `Hᵨᵨ = Δε''(ρ)`
    /// is the second ρ-derivative of the deterministic conditional shift. This
    /// test pins three things on a Poisson/log smooth (non-zero ε):
    ///   (i)   the total = conditional + ½ H Var, with H matching an INDEPENDENT
    ///         high-accuracy (Richardson-extrapolated) finite difference of
    ///         `lawley_lr_mean_shift` in ρ — proving the internal stencil is the
    ///         curvature it claims to be;
    ///   (ii)  zero ρ-variance recovers the conditional shift exactly (the
    ///         correction switches off as λ → fixed);
    ///   (iii) the correction is genuinely non-zero (the curvature and variance
    ///         are both non-zero), so the ρ̂-variation term is load-bearing.
    #[test]
    fn rho_variation_correction_matches_curvature_times_variance() {
        let n = 50usize;
        let mut x = Array2::<f64>::ones((n, 2));
        let mut kappas = Vec::with_capacity(n);
        for i in 0..n {
            let z = i as f64 / n as f64 - 0.5;
            x[[i, 1]] = z;
            let eta = 0.3 + 0.6 * z;
            kappas.push(
                RowExpectedJets::poisson_log(eta)
                    .kappas()
                    .expect("poisson kappas"),
            );
        }
        // One smoothing parameter (λ = 3) penalizing the second column.
        let lambda = 3.0_f64;
        let mut s_comp = Array2::<f64>::zeros((2, 2));
        s_comp[[1, 1]] = lambda;
        let penalty = s_comp.clone();
        let components = vec![RhoPenaltyComponent {
            s_component: s_comp.clone(),
        }];
        let tested = 1..2;

        // Conditional (fixed-λ) shift.
        let conditional =
            lawley_lr_mean_shift(x.view(), &kappas, Some(penalty.view()), tested.clone())
                .expect("conditional shift");

        // INDEPENDENT curvature Δε''(ρ): scale the WHOLE block by e^{t} (ρ = log λ
        // ⇒ S_λ(t) = e^{t}·λ·S_unit) and Richardson-combine two central
        // differences for an O(h⁴)-accurate second derivative.
        let de_at = |t: f64| {
            let mut s = Array2::<f64>::zeros((2, 2));
            s[[1, 1]] = lambda * t.exp();
            lawley_lr_mean_shift(x.view(), &kappas, Some(s.view()), tested.clone())
                .expect("perturbed shift")
        };
        let h = 0.05_f64;
        let d2_h = (de_at(h) - 2.0 * conditional + de_at(-h)) / (h * h);
        let d2_2h = (de_at(2.0 * h) - 2.0 * conditional + de_at(-2.0 * h)) / (4.0 * h * h);
        // Richardson: (4·D(h) − D(2h))/3 cancels the O(h²) term.
        let curvature = (4.0 * d2_h - d2_2h) / 3.0;
        assert!(
            curvature.abs() > 1e-9,
            "fixture must have non-zero ρ-curvature; got {curvature}"
        );

        let var_rho = 0.8_f64; // Var(ρ̂) (an inverse-outer-Hessian magnitude)
        let rho_cov = Array2::from_shape_vec((1, 1), vec![var_rho]).unwrap();
        let total = lawley_lr_mean_shift_with_rho_variation(
            x.view(),
            &kappas,
            penalty.view(),
            tested.clone(),
            &components,
            rho_cov.view(),
        )
        .expect("rho-variation shift");

        // (i) total = conditional + ½ H Var, H from the independent Richardson FD.
        let expected = conditional + 0.5 * curvature * var_rho;
        assert!(
            (total - expected).abs() < 1e-6 * (1.0 + expected.abs()),
            "ρ̂-variation total {total} must equal conditional + ½ H Var = {expected} \
             (conditional={conditional}, H={curvature}, Var={var_rho})"
        );
        // (iii) the ρ̂-variation correction is genuinely non-zero.
        assert!(
            (total - conditional).abs() > 1e-9,
            "ρ̂-variation correction must be non-zero (H={curvature}, Var={var_rho}); \
             total={total} conditional={conditional}"
        );

        // (ii) zero variance recovers the conditional shift exactly.
        let zero_cov = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let total_zero = lawley_lr_mean_shift_with_rho_variation(
            x.view(),
            &kappas,
            penalty.view(),
            tested.clone(),
            &components,
            zero_cov.view(),
        )
        .expect("zero-variance shift");
        assert!(
            (total_zero - conditional).abs() < 1e-12,
            "zero ρ-variance must recover the conditional shift: {total_zero} vs {conditional}"
        );
    }

    /// DELIVERABLE (2) — the ρ̂-variation **factor** entry point
    /// [`lawley_lr_bartlett_factor_with_rho_variation`] is the single-call
    /// `c = 1 + Δε(ρ̂)/d` reduction a live consumer wires symmetrically with the
    /// fixed-λ [`lawley_lr_bartlett_factor`]. This pins:
    ///   (i)   it equals `1 + (total ρ̂-variation shift)/d`, i.e. it folds the
    ///         estimated-λ correction into the factor (not just the conditional
    ///         fixed-λ shift);
    ///   (ii)  on a Poisson/log smooth with a non-zero curvature and a positive
    ///         `Var(ρ̂)`, the estimated-λ factor differs from the fixed-λ factor —
    ///         the ρ̂-variation contribution is load-bearing in the factor;
    ///   (iii) Gaussian known-variance (Δε ≡ 0, zero ρ-curvature) gives factor 1
    ///         at any `Cov(ρ̂)` — the closed-form anchor;
    ///   (iv)  a degenerate reference df is rejected.
    #[test]
    fn rho_variation_factor_folds_estimated_lambda_into_c() {
        // Poisson/log smooth substrate with a non-zero ε and a single smoothing
        // parameter on the second column.
        let n = 50usize;
        let mut x = Array2::<f64>::ones((n, 2));
        let mut kappas = Vec::with_capacity(n);
        for i in 0..n {
            let z = i as f64 / n as f64 - 0.5;
            x[[i, 1]] = z;
            let eta = 0.3 + 0.6 * z;
            kappas.push(
                RowExpectedJets::poisson_log(eta)
                    .kappas()
                    .expect("poisson kappas"),
            );
        }
        let lambda = 3.0_f64;
        let mut s_comp = Array2::<f64>::zeros((2, 2));
        s_comp[[1, 1]] = lambda;
        let penalty = s_comp.clone();
        let components = vec![RhoPenaltyComponent {
            s_component: s_comp,
        }];
        let tested = 1..2;
        let ref_df = 1.0_f64;
        let var_rho = 0.8_f64;
        let rho_cov = Array2::from_shape_vec((1, 1), vec![var_rho]).unwrap();

        // (i) factor = 1 + (total estimated-λ shift)/d.
        let total = lawley_lr_mean_shift_with_rho_variation(
            x.view(),
            &kappas,
            penalty.view(),
            tested.clone(),
            &components,
            rho_cov.view(),
        )
        .expect("total shift");
        let factor = lawley_lr_bartlett_factor_with_rho_variation(
            x.view(),
            &kappas,
            penalty.view(),
            tested.clone(),
            &components,
            rho_cov.view(),
            ref_df,
        )
        .expect("estimated-λ factor");
        assert!(
            (factor - (1.0 + total / ref_df)).abs() < 1e-12,
            "estimated-λ factor {factor} must equal 1 + Δε(ρ̂)/d = {}",
            1.0 + total / ref_df
        );

        // (ii) it differs from the fixed-λ (conditional) factor — the
        // ρ̂-variation term is load-bearing in c.
        let conditional_factor = lawley_lr_bartlett_factor(
            x.view(),
            &kappas,
            Some(penalty.view()),
            tested.clone(),
            ref_df,
        )
        .expect("conditional factor");
        assert!(
            (factor - conditional_factor).abs() > 1e-9,
            "estimated-λ factor {factor} must differ from the fixed-λ factor \
             {conditional_factor} (ρ̂-variation is load-bearing)"
        );

        // (iii) Gaussian anchor: Δε ≡ 0 ⇒ zero curvature ⇒ factor exactly 1 at any
        // Cov(ρ̂).
        let g_kappas = vec![
            RowExpectedJets::gaussian_identity(1.3)
                .kappas()
                .expect("gaussian kappas");
            n
        ];
        let big_cov = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
        let g_factor = lawley_lr_bartlett_factor_with_rho_variation(
            x.view(),
            &g_kappas,
            penalty.view(),
            tested.clone(),
            &components,
            big_cov.view(),
            ref_df,
        )
        .expect("gaussian factor");
        assert!(
            (g_factor - 1.0).abs() < 1e-12,
            "Gaussian known-variance estimated-λ factor must be exactly 1; got {g_factor}"
        );

        // (iv) degenerate reference df is rejected.
        assert!(
            lawley_lr_bartlett_factor_with_rho_variation(
                x.view(),
                &kappas,
                penalty.view(),
                tested.clone(),
                &components,
                rho_cov.view(),
                0.0,
            )
            .is_err()
        );
    }

    /// DELIVERABLE (2), ρ̂-variation cross terms — for two smoothing parameters
    /// the correction is `½ Σ_{b,b'} H_{bb'} Cov_{bb'}`, and the off-diagonal
    /// (mixed-partial) stencil must match an independent product-of-steps FD.
    /// Pins that the symmetric cross contribution `H_{01}Cov_{01}` is included
    /// (a diagonal-only assembly would miss it).
    #[test]
    fn rho_variation_includes_symmetric_cross_terms() {
        let n = 40usize;
        let mut x = Array2::<f64>::ones((n, 3));
        let mut kappas = Vec::with_capacity(n);
        for i in 0..n {
            let z = i as f64 / n as f64 - 0.5;
            x[[i, 1]] = z;
            x[[i, 2]] = z * z - 0.1;
            let eta = 0.2 + 0.5 * z - 0.3 * x[[i, 2]];
            kappas.push(
                RowExpectedJets::binomial_logit(eta)
                    .kappas()
                    .expect("binomial kappas"),
            );
        }
        // Two smoothing parameters, one on column 1, one on column 2.
        let (l1, l2) = (2.0_f64, 4.0_f64);
        let mut s1 = Array2::<f64>::zeros((3, 3));
        s1[[1, 1]] = l1;
        let mut s2 = Array2::<f64>::zeros((3, 3));
        s2[[2, 2]] = l2;
        let penalty = &s1 + &s2;
        let components = vec![
            RhoPenaltyComponent {
                s_component: s1.clone(),
            },
            RhoPenaltyComponent {
                s_component: s2.clone(),
            },
        ];
        // Test the joint nuisance/smooth block {1,2} against the intercept.
        let tested = 1..3;
        let conditional =
            lawley_lr_mean_shift(x.view(), &kappas, Some(penalty.view()), tested.clone())
                .expect("conditional");

        // Independent Hessian by FD on the deterministic shift as a function of
        // (t0, t1): block b scaled by e^{t_b}.
        let de = |t0: f64, t1: f64| {
            let mut s = Array2::<f64>::zeros((3, 3));
            s[[1, 1]] = l1 * t0.exp();
            s[[2, 2]] = l2 * t1.exp();
            lawley_lr_mean_shift(x.view(), &kappas, Some(s.view()), tested.clone())
                .expect("perturbed")
        };
        let h = 0.05_f64;
        let h00 = (de(h, 0.0) - 2.0 * conditional + de(-h, 0.0)) / (h * h);
        let h11 = (de(0.0, h) - 2.0 * conditional + de(0.0, -h)) / (h * h);
        let h01 = (de(h, h) - de(h, -h) - de(-h, h) + de(-h, -h)) / (4.0 * h * h);

        // A full (non-diagonal) ρ-covariance.
        let rho_cov = Array2::from_shape_vec((2, 2), vec![0.7, 0.2, 0.2, 0.5]).unwrap();
        let total = lawley_lr_mean_shift_with_rho_variation(
            x.view(),
            &kappas,
            penalty.view(),
            tested.clone(),
            &components,
            rho_cov.view(),
        )
        .expect("rho-variation shift");

        // Expected = conditional + ½(H00 V00 + H11 V11) + H01 V01 (symmetric cross).
        let expected = conditional
            + 0.5 * (h00 * rho_cov[[0, 0]] + h11 * rho_cov[[1, 1]])
            + h01 * rho_cov[[0, 1]];
        assert!(
            (total - expected).abs() < 1e-6 * (1.0 + expected.abs()),
            "two-parameter ρ̂-variation {total} must equal {expected} \
             (H00={h00}, H11={h11}, H01={h01})"
        );
        // The cross term is non-trivial — a diagonal-only assembly would differ.
        let diag_only = conditional + 0.5 * (h00 * rho_cov[[0, 0]] + h11 * rho_cov[[1, 1]]);
        assert!(
            (total - diag_only).abs() > 1e-9,
            "cross term H01·Cov01 must be included (off-diagonal non-zero): \
             total={total} diag_only={diag_only}"
        );
    }

    /// Shape guards: component/cov dimension mismatches are rejected.
    #[test]
    fn rho_variation_rejects_shape_mismatch() {
        let n = 8usize;
        let jets = RowExpectedJets::poisson_log(0.1);
        let kappas = vec![jets.kappas().expect("kappas"); n];
        let mut x = Array2::<f64>::ones((n, 2));
        for i in 0..n {
            x[[i, 1]] = i as f64 - 4.0;
        }
        let mut s = Array2::<f64>::zeros((2, 2));
        s[[1, 1]] = 1.0;
        let components = vec![RhoPenaltyComponent {
            s_component: s.clone(),
        }];
        // rho_cov is 2×2 but there is only 1 component.
        let bad_cov = Array2::<f64>::eye(2);
        assert!(
            lawley_lr_mean_shift_with_rho_variation(
                x.view(),
                &kappas,
                s.view(),
                1..2,
                &components,
                bad_cov.view(),
            )
            .is_err()
        );
        // No components at all.
        let cov1 = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        assert!(
            lawley_lr_mean_shift_with_rho_variation(
                x.view(),
                &kappas,
                s.view(),
                1..2,
                &[],
                cov1.view(),
            )
            .is_err()
        );
        // Component with the wrong dimension.
        let wrong = vec![RhoPenaltyComponent {
            s_component: Array2::<f64>::eye(3),
        }];
        assert!(
            lawley_lr_mean_shift_with_rho_variation(
                x.view(),
                &kappas,
                s.view(),
                1..2,
                &wrong,
                cov1.view(),
            )
            .is_err()
        );
        // The #740 handoff is a covariance matrix; accepting a non-symmetric
        // matrix would silently use only the upper triangle and misstate the
        // ρ̂-variation contribution.
        let nonsymmetric_cov = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        assert!(
            lawley_lr_mean_shift_with_rho_variation(
                x.view(),
                &kappas,
                s.view(),
                1..2,
                &components,
                nonsymmetric_cov.view(),
            )
            .is_ok()
        );
        let components2 = vec![
            RhoPenaltyComponent {
                s_component: s.clone(),
            },
            RhoPenaltyComponent {
                s_component: s.clone(),
            },
        ];
        let bad_sym = Array2::from_shape_vec((2, 2), vec![1.0, 0.25, 0.20, 1.0]).unwrap();
        assert!(
            lawley_lr_mean_shift_with_rho_variation(
                x.view(),
                &kappas,
                s.view(),
                1..2,
                &components2,
                bad_sym.view(),
            )
            .is_err()
        );
    }

    #[test]
    fn epsilon_is_invariant_under_linear_reparametrization() {
        // ε is a property of the model, not the basis: X → X·T for invertible
        // T must leave it unchanged (the penalty transforms congruently).
        let n = 15;
        let k = 3;
        let mut x = Array2::<f64>::zeros((n, k));
        let mut kappas = Vec::with_capacity(n);
        for i in 0..n {
            let z = i as f64 / n as f64;
            x[[i, 0]] = 1.0;
            x[[i, 1]] = (3.1 * z).cos();
            x[[i, 2]] = z - 0.5;
            let eta = -0.1 + 0.6 * x[[i, 1]] + 0.4 * x[[i, 2]];
            kappas.push(
                RowExpectedJets::binomial_logit(eta)
                    .kappas()
                    .expect("binomial kappas"),
            );
        }
        let t_mat = ndarray::arr2(&[[1.0, 0.3, -0.2], [0.0, 1.4, 0.5], [0.0, 0.0, 0.8]]);
        let xt = x.dot(&t_mat);
        let eps = lawley_epsilon(x.view(), &kappas, None).expect("ε");
        let eps_t = lawley_epsilon(xt.view(), &kappas, None).expect("ε reparam");
        assert!(
            (eps - eps_t).abs() < 1e-9 * (1.0 + eps.abs()),
            "ε not reparametrization-invariant: {eps} vs {eps_t}"
        );
    }
}

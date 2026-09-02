//! WBIC audit for the singular manifold-atom model-selection charge (Part-2
//! statistical-debt closure).
//!
//! WHY. The production birth/death charge is the Laplace/BIC rank charge
//! `½·d_eff·log N_eff` (see [`super::construction::realised_rank_charge_dof`]; #2a:
//! the occupancy-aware `N_eff = Σ_row a²`, not the global `n`), with
//! `d_eff = rank_chargeable · basis_edf`. Two integer ranks must not be
//! conflated. `rank_mp` is the Marchenko–Pastur hard count of reconstruction-
//! Gram eigenvalues above the noise edge. Production uses `rank_chargeable`:
//! it equals `rank_mp` when any direction clears the edge, promotes an MP-rank-zero
//! but positive spectrum to rank one, and leaves only an exactly zero spectrum
//! at zero (#2258). Stronger state-aware disappearance is certified upstream.
//! The `½·(·)·log n` Laplace charge is the correct
//! free-energy penalty ONLY for a
//! REGULAR statistical model, where the log-likelihood has a non-degenerate
//! Hessian at the MLE and the marginal likelihood expands as
//! `−log Z = n·L_n(ŵ) + (d/2)·log n + O(1)`. Manifold atoms are SINGULAR: gauge
//! orbits (the harmonic/rotation freedom of a chart), rank deficiencies (a
//! decoder direction collapsing toward the noise floor), and boundary solutions
//! (an amplitude pinned at zero) all break Hessian non-degeneracy. Watanabe's
//! singular-learning theory replaces the `d/2` coefficient with the LEARNING
//! COEFFICIENT (real log-canonical threshold) `λ ≤ d/2`, and the free energy is
//! `−log Z = n·L_n(ŵ) + λ·log n + o(log n)`. The hard MP charge can
//! over-price a barely resolved direction, but there is no universal finite-sample
//! ordering: WBIC also sums fractional mass from every sub-edge direction, while
//! production separately applies the #2258 minimum-rank promotion.
//!
//! THE ESTIMATOR (WBIC at inverse temperature `β = 1/log n`). Watanabe's Widely
//! Applicable BIC is the tempered-posterior expected log loss
//!
//! ```text
//! WBIC = E_β[ n·L_n(w) ],   posterior ∝ exp(−β·n·L_n(w))·π(w),   β = 1/log n,
//! ```
//!
//! which satisfies `E[WBIC] = n·L_n(ŵ) + λ·log n + o(log n)` for ANY model,
//! regular or singular. The implied complexity charge is `WBIC − n·L_n(ŵ) =
//! λ̂·log n`. We estimate `λ̂` in closed form (no MCMC) by a Laplace-at-temperature
//! expansion that is EXACT for the decoder model, because the reconstruction loss
//! is quadratic in the decoder coefficients:
//!
//!   Take one reconstruction direction `k` with reconstruction-Gram eigenvalue
//!   `μ_k` (per-observation signal+noise energy, `= sv_k²/n_eff`) against the MP
//!   noise edge `e = R·(1 + √(p/n_eff))²`. Its scalar amplitude `α_k` has
//!   tempered-LIKELIHOOD precision `h_k = β·g_k/R` with design energy
//!   `g_k = n_eff·μ_k`. The stated WBIC posterior tempers ONLY the likelihood —
//!   `π(w)` enters at full strength — so the REML "toward no effect" Gaussian
//!   prior keeps its UNtempered precision, fixed (with NO new constant) to the
//!   SAME noise edge the hard count uses: `τ_k = g_edge/R`, `g_edge = n_eff·e`.
//!   The tempered-Gaussian learning-coefficient contribution is
//!
//! ```text
//! λ̂_k = ½ · h_k / (h_k + τ_k) = ½ · β·g_k/(β·g_k + g_edge)
//!      = ½ · μ_k/(μ_k + e·log n_eff).
//! ```
//!
//!   `R` and the raw `n_eff` cancel; the `log n_eff` from `β = 1/log n_eff` does
//!   NOT — it is exactly Watanabe's temperature and dropping it (by tempering the
//!   prior too, as this module once did) silently forfeits the WBIC theorem the
//!   estimator's name invokes, over-counting every near-edge direction by up to
//!   `log n_eff`. The soft count is a SIGMOID in `μ_k/(e·log n_eff)` replacing
//!   the hard step `1[μ_k > e]`. It recovers the regular limit exactly (a
//!   direction far above the tempered edge contributes `½`, so a full-rank atom
//!   recovers `½·d_eff·log n = BIC`) and discounts singular directions smoothly
//!   (`μ_k → 0 ⇒ 0`). The soft COUNT has its midpoint at
//!   `μ_k = e·log n_eff` (`λ̂_k = ¼` there); this is not a crossing with
//!   the discontinuous hard step. `n_eff` is floored at Euler's number so
//!   `log n_eff ≥ 1` and the tempered edge is never softer than the hard MP edge.
//!
//! CHARGES.
//! ```text
//! rank_mp = Σ_k 1[μ_k > e]                         (integer MP reconstruction count)
//! rank_chargeable = rank_mp,                         if rank_mp > 0
//!                 = 1,                               if max μ > 0
//!                 = 0,                               if every μ = 0
//! rank_soft = Σ_k μ_k/(μ_k + e·log n_eff)          (WBIC tempered count)
//! C_mp   = ½ · rank_mp         · basis_edf · log N_eff (diagnostic)
//! C_prod = ½ · rank_chargeable · basis_edf · log N_eff (production)
//! C_wbic = ½ · rank_soft       · basis_edf · log N_eff (diagnostic)
//! ```
//! #2a — the log-sample-size is the atom's OCCUPANCY-aware effective sample size
//! `N_eff = Σ_row a²` (the same `n_eff` the MP edge already uses), NOT the global
//! row count `n`. `N_eff` is the Fisher information the gated atom actually
//! accumulates, so it is the honest BIC scale and it makes the charge invariant to
//! appending rows on which the atom's gate is OFF (inert-row invariance); `log n`
//! over-charges every atom by `½·d_eff·log(n/N_eff)`, worst for sparse selective
//! atoms.
//! `basis_edf = tr(G(G+λS)⁻¹)` is ALREADY a graded (Watanabe-compatible) effective
//! count of basis functions. The audit reports both integer ranks, both hard
//! charges, and the signed `C_prod − C_wbic` delta. The sign is not assumed:
//! either charge can be larger near the MP edge.
//!
//! This module is an AUDIT: it does NOT change the default charge. It computes the
//! reconstruction spectrum the SAME way the production core does and classifies
//! reconstruction rank versus chargeability through the SAME shared primitive (verified
//! against [`super::construction::realised_rank_charge_dof`] for both resolved
//! and weak-signal atoms in the tests).

use gam_linalg::faer_ndarray::{FaerEigh, FaerSvd};
use ndarray::Array2;

use super::Side;

/// The reconstruction spectrum of ONE atom — the shared substrate both charges
/// price. `mu` are the reconstruction-Gram eigenvalues `sv(diag(√λ)·Uᵀ·D)²/n_eff`
/// (with `(λ,U)=eigh(G)`), `edge` the Marchenko–Pastur reconstruction-rank edge
/// `R·(1+√(p/n_eff))²`, `dispersion` is `R`, and
/// `basis_edf = tr(G(G+λS)⁻¹)` is the ridge-trace effective basis count. This
/// is exactly the decomposition inside
/// `super::construction::realised_rank_charge_dof`, surfaced so the WBIC soft
/// count, hard MP reconstruction count, and production chargeable count can be
/// inspected without changing the production criterion.
#[derive(Clone, Debug)]
pub struct ReconSpectrum {
    /// Reconstruction-Gram eigenvalues (per-observation signal+noise energy).
    mu: Vec<f64>,
    /// Marchenko–Pastur noise edge the hard rank count thresholds on.
    edge: f64,
}

impl ReconSpectrum {
    fn rank_classification(&self) -> super::construction::ReconstructionRankClassification {
        super::construction::classify_reconstruction_rank(&self.mu, self.edge)
    }

    /// #2258 production CHARGEABLE rank — the hard MP reconstruction count,
    /// with a below-rank-edge but numerically ALIVE atom promoted to the
    /// minimum non-degenerate rank 1. Mirrors the identical rule inside
    /// `super::construction::realised_rank_charge_dof` through the shared
    /// `super::construction::classify_reconstruction_rank` primitive;
    /// the ρ-derivative MUST take the same branch or the value/gradient pair
    /// desyncs (measured: real-GPT-2 fit priced finite by the promoted value
    /// path, then refused by the derivative's independent rank-zero invariant).
    /// Only an exactly zero reconstruction spectrum stays at rank 0 here.
    /// The stronger state-aware vanished-atom certificate runs before evidence
    /// pricing and may categorically refuse roundoff-indistinguishable signal.
    pub fn production_chargeable_rank(&self) -> usize {
        self.rank_classification().production_chargeable_rank
    }

}

/// Build the reconstruction spectrum from an atom's weighted basis Gram
/// `gram = Φᵀdiag(a²)Φ` (`m×m`), decoder `D` (`m×p`), effective sample size
/// `n_eff = Σ_row a²`, output dim `p_out`, noise floor `r_floor` (dispersion R),
/// and smoothness `(lam_smooth, smooth_penalty)`. Mirrors
/// `super::construction::realised_rank_charge_dof` byte-for-byte on the shared
/// quantities (checked in the parity test), returning the spectrum instead of the
/// collapsed `rank_eff · basis_edf`.
pub fn recon_spectrum(
    gram: &Array2<f64>,
    decoder: &Array2<f64>,
    n_eff: f64,
    p_out: f64,
    r_floor: f64,
    lam_smooth: f64,
    smooth_penalty: Option<&Array2<f64>>,
) -> Result<ReconSpectrum, String> {
    let m = gram.nrows();
    super::construction::validate_rank_charge_problem(
        gram,
        decoder,
        n_eff,
        p_out,
        r_floor,
        lam_smooth,
        smooth_penalty,
    )?;
    if m == 0 || n_eff == 0.0 {
        return Ok(ReconSpectrum {
            mu: Vec::new(),
            edge: 0.0,
        });
    }
    let (evals, u) = gram
        .eigh(Side::Lower)
        .map_err(|e| format!("recon_spectrum: eigh(G): {e}"))?;
    let evals = super::construction::certified_psd_spectrum(evals.view(), "rank-charge Gram")?;
    let mut scaled = u.t().dot(decoder);
    let cols = scaled.ncols();
    for i in 0..m {
        let s = evals[i].sqrt();
        for j in 0..cols {
            scaled[[i, j]] *= s;
        }
    }
    let sv = match scaled.svd(false, false) {
        Ok((_, sv, _)) => sv,
        Err(e) => return Err(format!("recon_spectrum: recon svd: {e}")),
    };
    let edge = crate::null_battery::mp_reconstruction_rank_edge(n_eff, p_out, r_floor)
        .map_err(|error| format!("recon_spectrum: {error}"))?;
    let mu = sv
        .iter()
        .map(|&singular_value| {
            super::construction::normalized_reconstruction_energy(singular_value, n_eff)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("recon_spectrum: {error}"))?;
    // basis_edf = tr(G(G+λS)⁻¹), the same ridge trace the production core computes.
    let mut mmat = gram.clone();
    if let Some(pen) = smooth_penalty {
        for i in 0..m {
            for j in 0..m {
                mmat[[i, j]] += lam_smooth * pen[[i, j]];
            }
        }
    }
    Ok(ReconSpectrum {
        mu,
        edge,
    })
}

/// One row of the WBIC-vs-rank-charge audit table.
#[derive(Clone, Debug)]
pub struct AuditRow {
    /// Human name of the synthetic population.
    pub name: String,
    /// Rows the atom was fit on.
    pub n: usize,
    /// Integer count of directions above the MP reconstruction-rank edge.
    pub mp_reconstruction_rank: usize,
    /// Integer rank the production criterion actually charges, including #2258
    /// alive-below-edge promotion.
    pub production_chargeable_rank: usize,
    /// WBIC tempered soft count.
    pub rank_soft: f64,
    /// Graded effective basis count.
    pub basis_edf: f64,
    /// Theoretical hard-MP reconstruction-rank charge
    /// `½·rank_mp·basis_edf·log N_eff`.
    pub mp_reconstruction_rank_charge: f64,
    /// Actual production rank / BIC charge
    /// `½·rank_chargeable·basis_edf·log N_eff`.
    pub production_charge: f64,
    /// WBIC / singular charge `½·rank_soft·basis_edf·log N_eff`.
    pub wbic_charge: f64,
    /// Signed `production_charge − wbic_charge`; no universal ordering is
    /// assumed near the MP edge.
    pub production_minus_wbic: f64,
    /// `production_minus_wbic / production_charge` (`NaN` if production charge
    /// is zero).
    pub production_delta_fraction: f64,
}

/// Directly price a WBIC learning-coefficient contribution for ONE scalar
/// reconstruction direction with per-observation energy `mu` against noise edge
/// `edge` at effective sample size `n_eff`: `λ̂_k = ½·μ/(μ + e·log n_eff)`
/// (header derivation, lines ~40-49). The likelihood is tempered by
/// `β = 1/log n_eff`, the REML prior is NOT — so the edge carries the
/// `log n_eff` temperature and this is `½ ×` the diagnostic `rank_soft`
/// per-direction term (same `tempered_edge = e·log n_eff`, `n_eff` floored at
/// Euler's number so `log n_eff ≥ 1`). Used by the sampling cross-check test
/// that validates the closed form against a genuine tempered-posterior
/// expectation.
#[cfg(test)]
mod learning_coeff_helpers_tests {

    /// A genuine (non-Laplace) WBIC estimate for a SINGLE scalar-amplitude
    /// reconstruction direction, by the thermodynamic tempered-posterior expectation
    /// `E_β[nL] − nL(α̂)`, divided by `log n_eff`, to recover `λ̂`. Used ONLY to
    /// validate [`direction_learning_coefficient`]: model `nL(α) = (g/2R)(α−α̂)²`
    /// with the LIKELIHOOD tempered at `β = 1/log n_eff` and a REML Gaussian prior of
    /// precision `τ = g_edge/R` that is NOT tempered (crossover-at-edge, exactly the
    /// header's untempered prior — tempering the prior too would forfeit the WBIC
    /// temperature the estimator's name invokes), integrated in closed form over the
    /// Gaussian (the integral is exact — the point is to confirm the algebra that
    /// produced the sigmoid, not to approximate). Because the model is Gaussian this
    /// returns the SAME number as the sigmoid up to the prior-shift term, which this
    /// includes so the test sees the full expectation. `log n_eff` uses the same
    /// `n_eff` floored at `e` as production `rank_soft`.
    pub(super) fn sampled_direction_learning_coefficient(
        mu: f64,
        edge: f64,
        n_eff: f64,
        r_floor: f64,
    ) -> f64 {
        let ln_neff = n_eff.max(std::f64::consts::E).ln();
        if !(ln_neff > 0.0) || !(r_floor > 0.0) || !(n_eff > 0.0) {
            return 0.0;
        }
        let beta = 1.0 / ln_neff;
        let g = n_eff * mu; // design energy
        let g_edge = n_eff * edge; // prior precision energy (crossover at edge)
        let h = beta * g / r_floor; // tempered likelihood precision
        let tau = g_edge / r_floor; // UNtempered prior precision (β does NOT enter π)
        let prec_post = h + tau;
        if !(prec_post > 0.0) {
            return 0.0;
        }
        // MLE amplitude scale: set α̂² so the direction carries energy μ per obs, i.e.
        // g·α̂² = n_eff·(μ − edge)_+ signal energy ⇒ α̂² = (μ − edge)_+ / μ (unitless),
        // the fraction of the direction's energy that is signal above the floor.
        let alpha_hat2 = ((mu - edge).max(0.0)) / mu.max(f64::MIN_POSITIVE);
        // Tempered-posterior expectation of nL(α) = (g/2R)(α−α̂)² over α ~ N(m_post,
        // 1/prec_post), m_post = h·α̂/prec_post (prior centred at 0):
        //   E[nL] − nL(α̂) = (g/2R)·(Var + (m_post − α̂)²).
        let var = 1.0 / prec_post;
        let m_post = h * 0.0_f64.max(alpha_hat2.sqrt()) / prec_post; // α̂ sign irrelevant to (·)²
        let alpha_hat = alpha_hat2.sqrt();
        let shift2 = (m_post - alpha_hat) * (m_post - alpha_hat);
        let e_delta = 0.5 * (g / r_floor) * (var + shift2);
        e_delta / ln_neff
    }
}

#[cfg(test)]
mod tests {
    use super::learning_coeff_helpers_tests::sampled_direction_learning_coefficient;
    use super::*;

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn lcg_normal(s: &mut u64) -> f64 {
        let u1 = lcg(s).max(1e-12);
        let u2 = lcg(s);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Degree-`h` periodic harmonic features [1, cos, sin, cos2, sin2, ...] at the
    /// given per-row angles (turns in [0,1)).
    fn harmonic_phi(turns: &[f64], h: usize) -> Array2<f64> {
        let n = turns.len();
        let m = 1 + 2 * h;
        Array2::from_shape_fn((n, m), |(i, c)| {
            if c == 0 {
                1.0
            } else {
                let k = (c + 1) / 2;
                let ang = std::f64::consts::TAU * k as f64 * turns[i];
                if c % 2 == 1 { ang.cos() } else { ang.sin() }
            }
        })
    }

    /// Polynomial features [1, t, t², ...] at scalar coordinates.
    fn poly_phi(t: &[f64], deg: usize) -> Array2<f64> {
        let n = t.len();
        Array2::from_shape_fn((n, deg + 1), |(i, c)| t[i].powi(c as i32))
    }

    /// The closed-form tempered sigmoid `½·μ/(μ + e·log n_eff)` must equal the
    /// tempered-posterior expectation for a Gaussian direction (the derivation
    /// cross-check): the likelihood is tempered by `β = 1/log n_eff`, the prior is
    /// NOT, so the untempered prior precision `τ = g_edge/R` is what makes the
    /// variance term collapse to the closed form. Prior shift is negligible far
    /// from the edge and the two agree there; near the edge the sampled value
    /// carries the small prior-shift term, so we check the dominant variance term
    /// agreement across a μ sweep.
    #[test]
    fn sigmoid_matches_tempered_posterior_variance_term() {
        let n_eff = 800.0_f64;
        let r_floor = 0.0025_f64;
        let edge = r_floor * (1.0 + (12.0_f64 / n_eff).sqrt()).powi(2);
        let ln_neff = n_eff.max(std::f64::consts::E).ln();
        for &ratio in &[8.0_f64, 4.0, 2.0, 1.0, 0.5, 0.25] {
            let mu = ratio * edge;
            let closed = direction_learning_coefficient(mu, edge, n_eff);
            let sampled = sampled_direction_learning_coefficient(mu, edge, n_eff, r_floor);
            // The variance term of the sampled expectation equals the closed form
            // exactly; the total sampled value adds only the (non-negative) prior
            // shift, so sampled ≥ variance term and both share the same limits.
            eprintln!("[wbic sigmoid] μ/e={ratio:.2}  closed={closed:.4} sampled≈{sampled:.4}");
            // Variance-only reconstruction of the sampled expectation: likelihood
            // tempered at β = 1/log n_eff, prior precision τ = g_edge/R UNtempered.
            let beta = 1.0 / ln_neff;
            let g = n_eff * mu;
            let g_edge = n_eff * edge;
            let h = beta * g / r_floor;
            let tau = g_edge / r_floor;
            let var_term = 0.5 * (g / r_floor) * (1.0 / (h + tau)) / ln_neff;
            assert!(
                (closed - var_term).abs() < 1e-9,
                "closed sigmoid must equal the tempered variance term: closed={closed} var={var_term}"
            );
        }
    }

}

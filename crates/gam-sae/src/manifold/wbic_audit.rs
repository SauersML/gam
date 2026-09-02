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

impl AuditRow {
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

    /// #2258 parity on the branch the old audit mislabeled: a nonzero decoder
    /// can sit below the MP reconstruction-rank edge (rank 0) while remaining numerically
    /// alive, and production charges it at the minimum rank 1. Both paths call
    /// the same classifier, so this test pins the semantic and numeric contract.
    #[test]
    fn weak_signal_reconstruction_rank_zero_is_production_chargeable_one() {
        let gram = Array2::<f64>::eye(1);
        let decoder = Array2::<f64>::ones((1, 1));
        let n_eff = 100.0;
        let r_floor = 1.0;
        let spec = recon_spectrum(&gram, &decoder, n_eff, 1.0, r_floor, 0.0, None).unwrap();
        let d_prod = super::super::construction::realised_rank_charge_dof(
            &gram, &decoder, n_eff, 1.0, r_floor, 0.0, None,
        )
        .unwrap();

        assert_eq!(spec.mp_reconstruction_rank(), 0);
        assert_eq!(spec.production_chargeable_rank(), 1);
        assert_eq!(d_prod, spec.basis_edf());
        assert_eq!(spec.production_charge(), 0.5 * d_prod * n_eff.ln());
        assert_eq!(spec.mp_reconstruction_rank_charge(), 0.0);
    }

    /// A zero MP edge does not make a zero-energy direction count as one. At the
    /// indeterminate `(energy, edge) = (0, 0)` boundary, the model-consistent
    /// convention is zero complexity for zero signal; positive energy against an
    /// exactly zero edge counts as one resolved direction.
    #[test]
    fn zero_edge_soft_rank_distinguishes_zero_from_positive_energy() {
        let gram = Array2::<f64>::eye(2);
        let zero = Array2::<f64>::zeros((2, 2));
        let zero_spec = recon_spectrum(&gram, &zero, 2.0, 2.0, 0.0, 0.0, None).unwrap();
        assert_eq!(zero_spec.mp_reconstruction_rank_edge(), 0.0);
        assert_eq!(zero_spec.rank_soft(), 0.0);
        assert_eq!(zero_spec.mp_reconstruction_rank(), 0);
        assert_eq!(zero_spec.production_chargeable_rank(), 0);
        assert_eq!(zero_spec.wbic_charge(), 0.0);

        let mut one_direction = Array2::<f64>::zeros((2, 2));
        one_direction[[0, 0]] = 1.0;
        let positive_spec =
            recon_spectrum(&gram, &one_direction, 2.0, 2.0, 0.0, 0.0, None).unwrap();
        assert_eq!(positive_spec.mp_reconstruction_rank_edge(), 0.0);
        assert_eq!(positive_spec.rank_soft(), 1.0);
        assert_eq!(positive_spec.mp_reconstruction_rank(), 1);
        assert_eq!(positive_spec.production_chargeable_rank(), 1);
    }

    /// The WBIC fraction stays finite when either `edge·log(n_eff)` or
    /// `μ + edge·log(n_eff)` would overflow under direct evaluation.
    #[test]
    fn soft_rank_fraction_avoids_finite_input_overflow() {
        let scale = 0.5 * f64::MAX;
        let spec = ReconSpectrum {
            mu: vec![scale],
            edge: scale,
        };
        let expected = 1.0 / (1.0 + f64::MAX.ln());
        let actual = spec.rank_soft();
        assert!(actual.is_finite() && actual > 0.0);
        assert!((actual - expected).abs() <= 8.0 * f64::EPSILON * expected);
    }

    /// Normalize by `√n_eff` before squaring: `s²` overflows here, while the
    /// per-observation energy `s²/n_eff = 10²⁰⁰` is finite. Both production
    /// and the audit must retain that finite energy and classify it identically.
    #[test]
    fn extreme_singular_value_has_finite_shared_energy_and_rank() {
        let gram = Array2::<f64>::eye(1);
        let decoder = Array2::<f64>::from_elem((1, 1), 1.0e200);
        let n_eff = 1.0e200;
        let spec = recon_spectrum(&gram, &decoder, n_eff, 1.0, 1.0, 0.0, None).unwrap();
        let energy = spec.reconstruction_energies()[0];
        assert!(energy.is_finite());
        assert!((energy / 1.0e200 - 1.0).abs() < 1.0e-12);
        assert_eq!(spec.mp_reconstruction_rank(), 1);
        assert_eq!(spec.production_chargeable_rank(), 1);

        let d_prod = super::super::construction::realised_rank_charge_dof(
            &gram, &decoder, n_eff, 1.0, 1.0, 0.0, None,
        )
        .unwrap();
        assert_eq!(d_prod, spec.basis_edf());
    }

    #[test]
    fn rank_charge_value_and_audit_share_strict_numeric_contract() {
        let gram = Array2::<f64>::eye(2);
        let decoder = Array2::<f64>::zeros((2, 3));
        let production = |gram: &Array2<f64>,
                          decoder: &Array2<f64>,
                          n_eff: f64,
                          p_out: f64,
                          r_floor: f64,
                          lam_smooth: f64,
                          penalty: Option<&Array2<f64>>| {
            super::super::construction::realised_rank_charge_dof(
                gram, decoder, n_eff, p_out, r_floor, lam_smooth, penalty,
            )
        };

        for (n_eff, p_out, r_floor, lam_smooth) in [
            (f64::NAN, 3.0, 1.0, 0.0),
            (-1.0, 3.0, 1.0, 0.0),
            (10.0, f64::INFINITY, 1.0, 0.0),
            (10.0, 3.0, -1.0, 0.0),
            (10.0, 3.0, 1.0, f64::NAN),
        ] {
            assert!(production(&gram, &decoder, n_eff, p_out, r_floor, lam_smooth, None).is_err());
            assert!(
                recon_spectrum(&gram, &decoder, n_eff, p_out, r_floor, lam_smooth, None).is_err()
            );
        }

        let wrong_width = Array2::<f64>::zeros((2, 2));
        assert!(production(&gram, &wrong_width, 10.0, 3.0, 1.0, 0.0, None).is_err());
        assert!(recon_spectrum(&gram, &wrong_width, 10.0, 3.0, 1.0, 0.0, None).is_err());

        // A smoothing penalty can make G+lambda*S positive definite, so a
        // materially negative Gram direction used to survive Cholesky after
        // being silently clipped out of the reconstruction rank. Both paths
        // must reject the invalid Gram before smoothing changes it.
        let indefinite = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, -0.25]).unwrap();
        let penalty = Array2::<f64>::eye(2);
        assert!(production(&indefinite, &decoder, 10.0, 3.0, 1.0, 1.0, Some(&penalty)).is_err());
        assert!(
            recon_spectrum(&indefinite, &decoder, 10.0, 3.0, 1.0, 1.0, Some(&penalty)).is_err()
        );

        // The inverse failure mode is just as invalid: a positive Gram can
        // hide a negative smoothing direction while G+lambda*S remains SPD.
        let indefinite_penalty =
            Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, -0.25]).unwrap();
        assert!(
            production(
                &gram,
                &decoder,
                10.0,
                3.0,
                1.0,
                1.0,
                Some(&indefinite_penalty),
            )
            .is_err()
        );
        assert!(
            recon_spectrum(
                &gram,
                &decoder,
                10.0,
                3.0,
                1.0,
                1.0,
                Some(&indefinite_penalty),
            )
            .is_err()
        );

        assert_eq!(
            production(&gram, &decoder, 0.0, 3.0, 1.0, 0.0, None).unwrap(),
            0.0
        );
        assert_eq!(
            recon_spectrum(&gram, &decoder, 0.0, 3.0, 1.0, 0.0, None)
                .unwrap()
                .mp_reconstruction_rank_edge(),
            0.0
        );
    }

    /// #2a EXPLICIT FORMULA on a known small fixture: the rank charge is
    /// ½·d_eff·ln N_eff with d_eff = rank_chargeable·basis_edf and N_eff the occupancy-aware
    /// effective sample size — NOT the global row count. Pins the scale so a regression
    /// back to ln(n) is caught.
    #[test]
    fn rank_charge_equals_half_deff_ln_neff() {
        // One strong direction far above the edge (mu=10 ≫ edge=1) and one far below
        // (mu=0.01): MP and production ranks are both 1. Small hand-set numbers,
        // no fit.
        let spec = ReconSpectrum {
            mu: vec![10.0, 0.01],
            edge: 1.0,
        };
        assert_eq!(spec.mp_reconstruction_rank(), 1);
        assert_eq!(spec.production_chargeable_rank(), 1);
        let d_eff = spec.production_chargeable_rank() as f64 * spec.basis_edf(); // 3.0
        let expected = 0.5 * d_eff * (50.0_f64).ln();
        assert!(
            (spec.production_charge() - expected).abs() < 1e-12,
            "rank charge must be ½·d_eff·ln(N_eff)={expected}, got {}",
            spec.production_charge()
        );
        // And it must NOT equal the global-n form for any n != N_eff (here n=5000).
        let global = 0.5 * d_eff * (5000.0_f64).ln();
        assert!(
            (spec.production_charge() - global).abs() > 1.0,
            "charge must use N_eff (50), not a global n (5000)"
        );
    }

}

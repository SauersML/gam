//! WBIC audit for the singular manifold-atom model-selection charge (Part-2
//! statistical-debt closure).
//!
//! WHY. The production birth/death charge is the Laplace/BIC rank charge
//! `½·d_eff·log N_eff` (see [`super::construction::realised_rank_charge_dof`]; #2a:
//! the occupancy-aware `N_eff = Σ_row a²`, not the global `n`), with
//! `d_eff = rank_chargeable · basis_edf`. Two integer ranks must not be
//! conflated. `rank_mp` is the Marchenko–Pastur hard count of reconstruction-
//! Gram eigenvalues above the noise edge. Production uses `rank_chargeable`:
//! it equals `rank_mp` when any direction clears the edge, promotes an MP-rank-zero but
//! numerically alive decoder to rank one, and leaves only a genuinely vanished
//! decoder at zero (#2258). The `½·(·)·log n` Laplace charge is the correct
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
//!                 = 1,                               if max μ > 10⁻⁹ R
//!                 = 0,                               otherwise
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

use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, FaerSvd};
use ndarray::{Array2, ArrayView2};

use super::Side;

/// Stable evaluation of `μ/(μ + edge·log n_eff)` for validated non-negative
/// inputs. Scaling by `max(μ, edge)` avoids overflow in both the tempered edge
/// product and the denominator. At an exactly zero edge, zero energy contributes
/// zero while every positive-energy direction contributes one.
fn wbic_tempered_rank_fraction(mu: f64, edge: f64, n_eff: f64) -> f64 {
    if mu == 0.0 {
        return 0.0;
    }
    if edge == 0.0 {
        return 1.0;
    }
    let log_n_eff = n_eff.max(std::f64::consts::E).ln();
    let scale = mu.max(edge);
    let scaled_mu = mu / scale;
    scaled_mu / (scaled_mu + (edge / scale) * log_n_eff)
}

/// The reconstruction spectrum of ONE atom — the shared substrate both charges
/// price. `mu` are the reconstruction-Gram eigenvalues `sv(diag(√λ)·Uᵀ·D)²/n_eff`
/// (with `(λ,U)=eigh(G)`), `edge` the Marchenko–Pastur reconstruction-rank edge
/// `R·(1+√(p/n_eff))²`, `dispersion` is `R`, and
/// `basis_edf = tr(G(G+λS)⁻¹)` is the ridge-trace effective basis count. This
/// is exactly the decomposition inside
/// [`super::construction::realised_rank_charge_dof`], surfaced so the WBIC soft
/// count, hard MP reconstruction count, and production chargeable count can be
/// inspected without changing the production criterion.
#[derive(Clone, Debug)]
pub struct ReconSpectrum {
    /// Reconstruction-Gram eigenvalues (per-observation signal+noise energy).
    mu: Vec<f64>,
    /// Marchenko–Pastur noise edge the hard rank count thresholds on.
    edge: f64,
    /// Reconstruction dispersion `R`, used by the production vanished-decoder
    /// threshold `RANK_VANISHED_REL·R`.
    dispersion: f64,
    /// `tr(G(G+λS)⁻¹)` — the graded effective basis-function count.
    basis_edf: f64,
    /// Effective sample size `Σ_row a²`.
    n_eff: f64,
}

impl ReconSpectrum {
    fn rank_classification(&self) -> super::construction::ReconstructionRankClassification {
        super::construction::classify_reconstruction_rank(&self.mu, self.edge, self.dispersion)
    }

    /// Validated per-observation reconstruction-Gram eigenvalues.
    pub fn reconstruction_energies(&self) -> &[f64] {
        &self.mu
    }

    /// Validated Marchenko–Pastur reconstruction-rank edge.
    pub fn mp_reconstruction_rank_edge(&self) -> f64 {
        self.edge
    }

    /// Graded effective basis-function count `tr(G(G+λS)⁻¹)`.
    pub fn basis_edf(&self) -> f64 {
        self.basis_edf
    }

    /// Audit-only basis-EDF specialization used by the checkpoint dynamics,
    /// whose identity interpolation design contributes one graded unit per
    /// reconstruction direction rather than the identity Gram's trace.
    pub(super) fn with_audit_basis_edf(mut self, basis_edf: f64) -> Result<Self, String> {
        if !basis_edf.is_finite() || basis_edf < 0.0 {
            return Err(format!(
                "audit basis EDF must be finite and non-negative; got {basis_edf}"
            ));
        }
        self.basis_edf = basis_edf;
        Ok(self)
    }

    /// Hard Marchenko–Pastur reconstruction-rank count `Σ_k 1[μ_k > e]`.
    ///
    /// This is a diagnostic reconstruction rank, not the production chargeable rank:
    /// an alive decoder can have reconstruction rank zero and still be charged rank
    /// one by [`Self::production_chargeable_rank`].
    pub fn mp_reconstruction_rank(&self) -> usize {
        self.rank_classification().mp_reconstruction_rank
    }

    /// #2258 production CHARGEABLE rank — the hard MP reconstruction count,
    /// with a below-rank-edge but numerically ALIVE atom promoted to the
    /// minimum non-degenerate rank 1. Mirrors the identical rule inside
    /// [`super::construction::realised_rank_charge_dof`] through the shared
    /// [`super::construction::classify_reconstruction_rank`] primitive;
    /// the ρ-derivative MUST take the same branch or the value/gradient pair
    /// desyncs (measured: real-GPT-2 fit priced finite by the promoted value
    /// path, then refused by the derivative's independent rank-zero invariant).
    /// Only a VANISHED decoder (top μ ≤ `RANK_VANISHED_REL`·R — the
    /// Laplace-invalid regime) stays at rank 0.
    pub fn production_chargeable_rank(&self) -> usize {
        self.rank_classification().production_chargeable_rank
    }

    /// WBIC tempered soft count `Σ_k μ_k/(μ_k + e·log n_eff)` — the sigmoid that
    /// replaces the hard step (derivation in the module header: the likelihood is
    /// tempered by `β = 1/log n_eff`, the prior is NOT, so the edge carries the
    /// `log n_eff` temperature). Always `≤` the number of directions, each term
    /// `∈ [0, 1]` (the value 1 occurs only at zero edge with positive energy); a
    /// direction at the TEMPERED edge `μ = e·log n_eff` counts `½`.
    /// `n_eff` is floored at Euler's number so the tempered edge is never softer
    /// than the hard MP edge.
    pub fn rank_soft(&self) -> f64 {
        self.mu
            .iter()
            .map(|&m| wbic_tempered_rank_fraction(m, self.edge, self.n_eff))
            .sum()
    }

    /// Theoretical hard-MP reconstruction-rank charge
    /// `½·rank_mp·basis_edf·log N_eff`.
    ///
    /// This is deliberately not called the production charge: #2258 production
    /// uses [`Self::production_charge`] and can promote an alive, sub-edge atom
    /// from MP reconstruction rank zero to chargeable rank one.
    ///
    /// #2a — the log-sample-size is the atom's OCCUPANCY-aware effective sample size
    /// `N_eff = Σ_row a²` (`self.n_eff`), NOT the global row count `n`: `N_eff` is the
    /// Fisher information a gated atom actually accumulates, so it is the honest BIC
    /// scale, and it matches the MP edge in `recon_spectrum` (which already uses
    /// `n_eff`). This satisfies inert-row invariance — appending rows on which the
    /// atom's gate is OFF adds 0 to `Σa²` and so must not change the charge — which
    /// `log n` violates. Floored at `N_eff=1` to keep the log non-negative.
    pub fn mp_reconstruction_rank_charge(&self) -> f64 {
        0.5 * self.mp_reconstruction_rank() as f64 * self.basis_edf * self.n_eff.max(1.0).ln()
    }

    /// Actual production Laplace–BIC charge
    /// `½·rank_chargeable·basis_edf·log N_eff`, including the #2258
    /// minimum-rank promotion for an alive decoder below the MP edge.
    pub fn production_charge(&self) -> f64 {
        0.5 * self.production_chargeable_rank() as f64 * self.basis_edf * self.n_eff.max(1.0).ln()
    }

    /// WBIC / singular free-energy charge `½·rank_soft·basis_edf·log N_eff`. Same
    /// occupancy-aware `log N_eff` scale as [`Self::production_charge`] (#2a).
    pub fn wbic_charge(&self) -> f64 {
        0.5 * self.rank_soft() * self.basis_edf * self.n_eff.max(1.0).ln()
    }

    /// Watanabe learning-coefficient estimate `λ̂ = ½·rank_soft·basis_edf` (`≤ d/2`,
    /// the singular bound); the regular Laplace coefficient is
    /// `½·rank_mp·basis_edf` for a resolved atom.
    ///
    /// Theorem K: this `λ̂` IS the running complexity `λ(n) = d(−log Z)/d(log n)`
    /// (the coefficient of `ln N_eff` in the atom's evidence charge) evaluated in the
    /// FINITE-n regime. The hard-MP diagnostic uses its `n→∞` limit
    /// `½·rank_mp·basis_edf`, so the soft ledger reduces to the hard limit away
    /// from the edge. Near the edge, the WBIC soft count and production's
    /// categorical minimum-rank rule answer different questions and have no
    /// universal ordering.
    ///
    /// SYMMETRY-IS-CHARGE: a truth with a continuous symmetry — a compact stabilizer
    /// of dimension `s` acting on the atom (e.g. the O(2) rotation of a clean circle
    /// chart, or a gauge/permutation orbit of the decoder) — freezes `s` reconstruction
    /// directions AT `μ=0`. Those directions never cross the noise edge, so they drop
    /// `rank_soft` (hence `λ̂`) by `s/2` relative to a would-be regular model of the
    /// same nominal dimension: the running complexity is `λ = (d−s)/2` where `d` counts
    /// unconstrained directions. The evidence LOST to that lower `λ·ln n` charge is
    /// exactly the evidence GAINED as `+log Vol(orbit)` (the O(1) integral over the
    /// symmetry orbit in the Laplace expansion) — a symmetric atom is cheaper to encode
    /// by precisely its orbit volume. This is why a genuinely curved (symmetric) atom is
    /// NOT penalized as a full-rank blob: its stabilizer is a discount, not a cost.
    pub fn learning_coefficient(&self) -> f64 {
        0.5 * self.rank_soft() * self.basis_edf
    }
}

/// Build the reconstruction spectrum from an atom's weighted basis Gram
/// `gram = Φᵀdiag(a²)Φ` (`m×m`), decoder `D` (`m×p`), effective sample size
/// `n_eff = Σ_row a²`, output dim `p_out`, noise floor `r_floor` (dispersion R),
/// and smoothness `(lam_smooth, smooth_penalty)`. Mirrors
/// [`super::construction::realised_rank_charge_dof`] byte-for-byte on the shared
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
            dispersion: r_floor,
            basis_edf: 0.0,
            n_eff,
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
    let factor = mmat.cholesky(Side::Lower).map_err(|error| {
        format!("recon_spectrum: G + lambda*S is not positive definite: {error}")
    })?;
    let x = factor.solve_mat(gram);
    let raw_basis_edf = (0..m).map(|i| x[[i, i]]).sum::<f64>();
    let basis_edf = super::construction::certified_basis_edf(raw_basis_edf, m, "recon_spectrum")?;
    Ok(ReconSpectrum {
        mu,
        edge,
        dispersion: r_floor,
        basis_edf,
        n_eff,
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
    /// Price a named atom from its reconstruction spectrum.
    pub fn from_spectrum(name: impl Into<String>, spec: &ReconSpectrum, n: usize) -> Self {
        let mp_reconstruction_rank_charge = spec.mp_reconstruction_rank_charge();
        let production_charge = spec.production_charge();
        let wbic_charge = spec.wbic_charge();
        let production_minus_wbic = production_charge - wbic_charge;
        let production_delta_fraction = if production_charge.abs() > 0.0 {
            production_minus_wbic / production_charge
        } else {
            f64::NAN
        };
        Self {
            name: name.into(),
            n,
            mp_reconstruction_rank: spec.mp_reconstruction_rank(),
            production_chargeable_rank: spec.production_chargeable_rank(),
            rank_soft: spec.rank_soft(),
            basis_edf: spec.basis_edf(),
            mp_reconstruction_rank_charge,
            production_charge,
            wbic_charge,
            production_minus_wbic,
            production_delta_fraction,
        }
    }
}

/// Render a disagreement table to a plain-text block (for the audit report / test
/// stderr). No side effects; the caller decides where it goes.
pub fn render_audit_table(rows: &[AuditRow]) -> String {
    let mut out = String::new();
    out.push_str(
        "population              n   r_mp r_prod  r_soft basis_edf    C_mp  C_prod  C_wbic  prod-wbic    frac\n",
    );
    out.push_str(
        "----------------------- --- ------ ------ ------- -------- ------- ------- ------- ---------- -------\n",
    );
    for r in rows {
        out.push_str(&format!(
            "{:<23} {:>3} {:>6} {:>6} {:>7.3} {:>8.3} {:>7.3} {:>7.3} {:>7.3} {:>10.3} {:>7.3}\n",
            r.name,
            r.n,
            r.mp_reconstruction_rank,
            r.production_chargeable_rank,
            r.rank_soft,
            r.basis_edf,
            r.mp_reconstruction_rank_charge,
            r.production_charge,
            r.wbic_charge,
            r.production_minus_wbic,
            r.production_delta_fraction,
        ));
    }
    out
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
    use super::*;

    pub(super) fn direction_learning_coefficient(mu: f64, edge: f64, n_eff: f64) -> f64 {
        0.5 * wbic_tempered_rank_fraction(mu, edge, n_eff)
    }

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

/// Compute the audit spectrum from an already-projected reconstruction problem:
/// given a residual `data` (`n×p`), a per-row activation weight `w` (gate²), and
/// an orthonormal basis of chart features `phi` (`n×m`), least-squares fit the
/// decoder `D = (ΦᵀWΦ + εI)⁻¹ ΦᵀW·data`, then form the reconstruction spectrum.
/// This is the self-contained synthetic-suite entry: it turns raw planted data +
/// a chart basis into the same spectrum the production charge prices.
pub fn spectrum_from_fit(
    data: ArrayView2<'_, f64>,
    w: &[f64],
    phi: &Array2<f64>,
    r_floor: f64,
    lam_smooth: f64,
    smooth_penalty: Option<&Array2<f64>>,
) -> Result<ReconSpectrum, String> {
    let (n, p) = data.dim();
    let m = phi.ncols();
    if phi.nrows() != n || w.len() != n {
        return Err("spectrum_from_fit: shape mismatch".into());
    }
    if data.iter().any(|value| !value.is_finite()) {
        return Err("spectrum_from_fit: data must be finite".into());
    }
    if phi.iter().any(|value| !value.is_finite()) {
        return Err("spectrum_from_fit: basis must be finite".into());
    }
    if w.iter().any(|weight| !weight.is_finite() || *weight < 0.0) {
        return Err("spectrum_from_fit: weights must be finite and non-negative".into());
    }
    // Weighted Gram G = ΦᵀWΦ and cross term ΦᵀW·data.
    let mut gram = Array2::<f64>::zeros((m, m));
    let mut cross = Array2::<f64>::zeros((m, p));
    let mut n_eff = 0.0_f64;
    for i in 0..n {
        let wi = w[i];
        n_eff += wi;
        for a in 0..m {
            let pa = phi[[i, a]] * wi;
            for b in a..m {
                gram[[a, b]] += pa * phi[[i, b]];
            }
            for j in 0..p {
                cross[[a, j]] += pa * data[[i, j]];
            }
        }
    }
    for a in 0..m {
        for b in a..m {
            let v = gram[[a, b]];
            gram[[a, b]] = v;
            gram[[b, a]] = v;
        }
    }
    // Ridge LS decoder solve (εI SPD guard) via Cholesky.
    let mut reg = gram.clone();
    for a in 0..m {
        reg[[a, a]] += 1.0e-9;
    }
    let decoder = reg
        .cholesky(Side::Lower)
        .map_err(|e| format!("spectrum_from_fit: chol: {e}"))?
        .solve_mat(&cross);
    recon_spectrum(
        &gram,
        &decoder,
        n_eff,
        p as f64,
        r_floor,
        lam_smooth,
        smooth_penalty,
    )
}

#[cfg(test)]
mod tests {
    use super::learning_coeff_helpers_tests::{
        direction_learning_coefficient, sampled_direction_learning_coefficient,
    };
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

    /// PARITY: the surfaced spectrum's production-chargeable rank × basis EDF
    /// must equal `realised_rank_charge_dof` on the same resolved inputs.
    #[test]
    fn spectrum_chargeable_rank_matches_production_deff() {
        let mut s = 0x0B1C_0001_u64;
        let n = 800usize;
        let p = 12usize;
        let turns: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
        let phi = harmonic_phi(&turns, 3);
        let m = phi.ncols();
        // A clean rank-2 circle decoder: cos→dim0, sin→dim1.
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let a = std::f64::consts::TAU * turns[i];
            data[[i, 0]] += a.cos();
            data[[i, 1]] += a.sin();
            for j in 0..p {
                data[[i, j]] += 0.05 * lcg_normal(&mut s);
            }
        }
        let w = vec![1.0_f64; n];
        // Build the gram + decoder exactly as spectrum_from_fit does, then call
        // BOTH recon_spectrum and the production core with identical arguments.
        let r_floor = 0.05_f64 * 0.05;
        let spec = spectrum_from_fit(data.view(), &w, &phi, r_floor, 0.0, None).unwrap();
        // Reproduce gram + decoder for the production call.
        let mut gram = Array2::<f64>::zeros((m, m));
        let mut cross = Array2::<f64>::zeros((m, p));
        for i in 0..n {
            for a in 0..m {
                for b in 0..m {
                    gram[[a, b]] += phi[[i, a]] * phi[[i, b]];
                }
                for j in 0..p {
                    cross[[a, j]] += phi[[i, a]] * data[[i, j]];
                }
            }
        }
        let mut reg = gram.clone();
        for a in 0..m {
            reg[[a, a]] += 1.0e-9;
        }
        let decoder = reg.cholesky(Side::Lower).unwrap().solve_mat(&cross);
        let d_prod = super::super::construction::realised_rank_charge_dof(
            &gram, &decoder, n as f64, p as f64, r_floor, 0.0, None,
        )
        .unwrap();
        let d_audit = spec.production_chargeable_rank() as f64 * spec.basis_edf();
        eprintln!(
            "[wbic parity] production d_eff={d_prod:.10}  \
             audit rank_chargeable·basis_edf={d_audit:.10}"
        );
        assert!(
            (d_prod - d_audit).abs() < 1e-8,
            "audit chargeable d_eff must match production: prod={d_prod} audit={d_audit}"
        );
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
            dispersion: 1.0,
            basis_edf: 1.0,
            n_eff: f64::MAX,
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

    #[test]
    fn spectrum_fit_rejects_invalid_weights_and_nonfinite_inputs() {
        let data = Array2::<f64>::zeros((2, 1));
        let phi = Array2::<f64>::eye(2);
        assert!(spectrum_from_fit(data.view(), &[-1.0, 2.0], &phi, 1.0, 0.0, None).is_err());
        assert!(spectrum_from_fit(data.view(), &[f64::NAN, 1.0], &phi, 1.0, 0.0, None).is_err());

        let mut nonfinite_data = data.clone();
        nonfinite_data[[0, 0]] = f64::INFINITY;
        assert!(
            spectrum_from_fit(nonfinite_data.view(), &[1.0, 1.0], &phi, 1.0, 0.0, None,).is_err()
        );

        let mut nonfinite_phi = phi;
        nonfinite_phi[[0, 0]] = f64::NAN;
        assert!(
            spectrum_from_fit(data.view(), &[1.0, 1.0], &nonfinite_phi, 1.0, 0.0, None,).is_err()
        );
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

    /// THE AUDIT. Price the standard synthetic suite and assert the disagreement
    /// structure Watanabe predicts: it is concentrated at the MP edge (singular /
    /// near-singular configs) and runs BOTH ways. (a) REGULAR atoms (well-separated
    /// clusters, a clean low-degree line, a strong clean circle) show near-zero
    /// disagreement (hard ≈ soft). (b) A CURVED atom whose spectrum piles at the
    /// edge (a filled DISK) is OVER-charged by the rank charge — `C_rank > C_wbic`
    /// by a quantified margin, the number that reprices its birth/death decision.
    /// (c) A WEAK circle whose spectrum sits just below the edge exposes the key
    /// semantic split: MP reconstruction rank is zero, production chargeable rank is
    /// one, and WBIC still pays fractional directions. (d) A noise-only fitted
    /// decoder has the same reconstruction-rank/chargeability split; only an exactly
    /// vanished decoder has production rank zero.
    #[test]
    fn wbic_audit_disagreement_table() {
        let mut rows: Vec<AuditRow> = Vec::new();
        let n = 1200usize;
        let p = 16usize;

        // (1) LINE — a regular rank-1 linear atom (poly deg 2), strong direction
        // far above the edge, higher terms far below ⇒ hard = soft.
        {
            let mut s = 0x1111_u64;
            let t: Vec<f64> = (0..n).map(|_| 2.0 * lcg(&mut s) - 1.0).collect();
            let phi = poly_phi(&t, 2);
            let mut data = Array2::<f64>::zeros((n, p));
            for i in 0..n {
                data[[i, 0]] += 2.0 * t[i];
                data[[i, 1]] += 0.5 * t[i];
                for j in 0..p {
                    data[[i, j]] += 0.05 * lcg_normal(&mut s);
                }
            }
            let w = vec![1.0_f64; n];
            let spec = spectrum_from_fit(data.view(), &w, &phi, 0.0025, 0.0, None).unwrap();
            rows.push(AuditRow::from_spectrum("line (regular)", &spec, n));
        }

        // (2) CLUSTERS — regular: three well-separated Gaussian blobs, one-hot
        // membership basis (rank-3, all far above edge).
        {
            let mut s = 0x2222_u64;
            let phi =
                Array2::<f64>::from_shape_fn((n, 3), |(i, c)| if i % 3 == c { 1.0 } else { 0.0 });
            let centers = [[3.0, 0.0], [0.0, 3.0], [-3.0, -3.0]];
            let mut data = Array2::<f64>::zeros((n, p));
            for i in 0..n {
                let c = i % 3;
                data[[i, 0]] += centers[c][0];
                data[[i, 1]] += centers[c][1];
                for j in 0..p {
                    data[[i, j]] += 0.1 * lcg_normal(&mut s);
                }
            }
            let w = vec![1.0_f64; n];
            let spec = spectrum_from_fit(data.view(), &w, &phi, 0.01, 0.0, None).unwrap();
            rows.push(AuditRow::from_spectrum("clusters (regular)", &spec, n));
        }

        // (3) CLEAN CIRCLE — a genuine rank-2 harmonic circle with a degree-3
        // basis. The two fundamental directions sit far above the edge; the
        // higher-harmonic directions sit far below ⇒ near agreement (a clean,
        // well-identified curved atom is NOT singular).
        {
            let mut s = 0x3333_u64;
            let turns: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
            let phi = harmonic_phi(&turns, 3);
            let mut data = Array2::<f64>::zeros((n, p));
            for i in 0..n {
                let a = std::f64::consts::TAU * turns[i];
                data[[i, 0]] += a.cos();
                data[[i, 1]] += a.sin();
                for j in 0..p {
                    data[[i, j]] += 0.05 * lcg_normal(&mut s);
                }
            }
            let w = vec![1.0_f64; n];
            let spec = spectrum_from_fit(data.view(), &w, &phi, 0.0025, 0.0, None).unwrap();
            rows.push(AuditRow::from_spectrum("circle clean (curved)", &spec, n));
        }

        // (4) NEAR-SINGULAR WEAK CIRCLE — the complementary case: a weak circle
        // whose amplitude sits at the noise scale, so its reconstruction
        // eigenvalues fall JUST BELOW the MP edge. The MP reconstruction count drops
        // to 0, but #2258 production recognizes the decoder as alive and charges
        // the minimum rank 1; the tempered soft count remains fractional.
        {
            let mut s = 0x4444_u64;
            let turns: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
            let phi = harmonic_phi(&turns, 3);
            let mut data = Array2::<f64>::zeros((n, p));
            // Weak amplitude (0.22) at the noise scale (0.15) ⇒ the circle's
            // reconstruction eigenvalues sit at/below the edge (near-singular).
            for i in 0..n {
                let a = std::f64::consts::TAU * turns[i];
                data[[i, 0]] += 0.22 * a.cos();
                data[[i, 1]] += 0.22 * a.sin();
                for j in 0..p {
                    data[[i, j]] += 0.15 * lcg_normal(&mut s);
                }
            }
            let w = vec![1.0_f64; n];
            let spec = spectrum_from_fit(data.view(), &w, &phi, 0.15 * 0.15, 0.0, None).unwrap();
            rows.push(AuditRow::from_spectrum(
                "circle near-edge (singular)",
                &spec,
                n,
            ));
        }

        // (5) DISK — a filled 2-disk (radius uniform, not a shell): reconstruction
        // energy spreads across the plane with several near-edge directions ⇒ the
        // hard count is unstable/over-counts vs the graded soft count.
        {
            let mut s = 0x5555_u64;
            let turns: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
            let radii: Vec<f64> = (0..n).map(|_| lcg(&mut s).sqrt()).collect();
            let phi = harmonic_phi(&turns, 3);
            let mut data = Array2::<f64>::zeros((n, p));
            for i in 0..n {
                let a = std::f64::consts::TAU * turns[i];
                data[[i, 0]] += 0.35 * radii[i] * a.cos();
                data[[i, 1]] += 0.35 * radii[i] * a.sin();
                for j in 0..p {
                    data[[i, j]] += 0.12 * lcg_normal(&mut s);
                }
            }
            let w = vec![1.0_f64; n];
            let spec = spectrum_from_fit(data.view(), &w, &phi, 0.12 * 0.12, 0.0, None).unwrap();
            rows.push(AuditRow::from_spectrum("disk (curved)", &spec, n));
        }

        // (6) PLANTED GAUSSIAN BLEND — no circle structure, so every harmonic
        // direction remains below the MP edge. The fitted decoder is nevertheless
        // nonzero, hence production charges rank 1 rather than treating a weak
        // noise fit as a vanished decoder.
        {
            let mut s = 0x6666_u64;
            let turns: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
            let phi = harmonic_phi(&turns, 3);
            let mut data = Array2::<f64>::zeros((n, p));
            for i in 0..n {
                // random low-rank Gaussian, uncorrelated with the harmonic basis
                for j in 0..p {
                    data[[i, j]] += 0.1 * lcg_normal(&mut s);
                }
            }
            let w = vec![1.0_f64; n];
            let spec = spectrum_from_fit(data.view(), &w, &phi, 0.01, 0.0, None).unwrap();
            rows.push(AuditRow::from_spectrum("gaussian blend (noise)", &spec, n));
        }

        // (7) EXACT VANISHING — only this degeneracy branch remains production
        // rank zero. It is distinct from a vanished decoder.
        rows.push(AuditRow::from_spectrum(
            "decoder vanished",
            &ReconSpectrum {
                mu: vec![0.0; 7],
                edge: 0.01 * (1.0 + (p as f64 / n as f64).sqrt()).powi(2),
                dispersion: 0.01,
                basis_edf: 7.0,
                n_eff: n as f64,
            },
            n,
        ));

        eprintln!("\n{}", render_audit_table(&rows));

        // Structural assertions on the disagreement pattern. The disagreement is
        // concentrated at the MP edge (singular / near-singular configs) and runs
        // in BOTH directions for the hard-MP diagnostic. Production's separate
        // minimum-rank rule is surfaced explicitly instead of being mislabeled as
        // the MP count.
        let get = |name: &str| rows.iter().find(|r| r.name == name).unwrap().clone();
        let line = get("line (regular)");
        let clusters = get("clusters (regular)");
        let near = get("circle near-edge (singular)");
        let disk = get("disk (curved)");
        let blend = get("gaussian blend (noise)");
        let vanished = get("decoder vanished");

        // (a) regular atoms (strong directions far above the edge, the rest far
        // below) ⇒ hard ≈ soft, near-zero disagreement in either direction.
        assert!(
            line.production_delta_fraction.abs() < 0.05,
            "regular line must show ~no disagreement; frac={:.3}",
            line.production_delta_fraction
        );
        assert!(
            clusters.production_delta_fraction.abs() < 0.05,
            "regular clusters must show ~no disagreement; frac={:.3}",
            clusters.production_delta_fraction
        );
        // (b) THE HEADLINE — the disk is a curved atom whose radial spread pushes
        // its two reconstruction directions to the edge: the hard count prices
        // them as two full units, the tempered count as ~1.3, so the rank charge
        // OVER-charges by a materially larger fraction than the regular atoms.
        // This is the number that reprices every birth/death decision for singular
        // curved atoms.
        assert!(
            disk.production_minus_wbic > 0.0 && disk.production_delta_fraction > 0.15,
            "curved disk must be OVER-charged (C_prod > C_wbic) by a clear fraction; \
             delta={:.3} frac={:.3}",
            disk.production_minus_wbic,
            disk.production_delta_fraction
        );
        assert!(
            disk.production_delta_fraction > line.production_delta_fraction + 0.1
                && disk.production_delta_fraction > clusters.production_delta_fraction + 0.1,
            "singular over-charge fraction ({:.3}) must exceed the regular atoms' \
             (line {:.3}, clusters {:.3}) by a clear margin",
            disk.production_delta_fraction,
            line.production_delta_fraction,
            clusters.production_delta_fraction
        );
        // (c) the weak near-edge circle is below the MP rank edge but is production-
        // chargeable. The two ranks must remain separately visible.
        assert!(
            near.mp_reconstruction_rank == 0
                && near.production_chargeable_rank == 1
                && near.rank_soft > 0.0
                && near.mp_reconstruction_rank_charge == 0.0
                && near.production_charge > 0.0,
            "weak near-edge circle must distinguish MP reconstruction rank from production \
             chargeability: mp={} prod={} soft={:.3}",
            near.mp_reconstruction_rank,
            near.production_chargeable_rank,
            near.rank_soft
        );
        // (d) A fitted noise decoder is alive even below the MP rank edge; an exactly
        // vanished decoder, and only that fixture, stays production rank zero.
        assert!(
            blend.mp_reconstruction_rank == 0
                && blend.production_chargeable_rank == 1
                && blend.rank_soft < 0.3,
            "noise fit must distinguish MP rank from production rank: \
             mp={} prod={} soft={:.3}",
            blend.mp_reconstruction_rank,
            blend.production_chargeable_rank,
            blend.rank_soft
        );
        assert_eq!(vanished.mp_reconstruction_rank, 0);
        assert_eq!(vanished.production_chargeable_rank, 0);
        assert_eq!(vanished.production_charge, 0.0);
        assert_eq!(vanished.wbic_charge, 0.0);
    }

    /// #2a INERT-ROW INVARIANCE (the correctness criterion for the occupancy scale).
    /// Appending rows on which the atom's gate is OFF (weight 0) changes neither the
    /// atom's likelihood nor its curvature nor its effective sample size N_eff = Σ a²,
    /// so it must NOT change the atom's rank charge. The fixed charge (½·d_eff·ln N_eff)
    /// satisfies this exactly — the appended zero-weight rows add 0 to Σa² and 0 to the
    /// Gram, so the whole reconstruction spectrum is bit-identical. The OLD global-row
    /// scale (½·d_eff·ln n_obs) VIOLATES it: n_obs grows from N to N+M, inflating the
    /// charge of a real (rank_chargeable>0) atom by ½·d_eff·ln((N+M)/N) nats — a spurious
    /// penalty for exactly the sparse, selective atoms an SAE exists to find.
    #[test]
    fn rank_charge_is_inert_row_invariant() {
        let mut s = 0x7777_u64;
        let n = 400usize;
        let p = 12usize;
        // A clean rank-2 circle (gate ON, weight 1) on N rows.
        let turns: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
        let phi = harmonic_phi(&turns, 3);
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let a = std::f64::consts::TAU * turns[i];
            data[[i, 0]] += a.cos();
            data[[i, 1]] += a.sin();
            for j in 0..p {
                data[[i, j]] += 0.05 * lcg_normal(&mut s);
            }
        }
        let w_on = vec![1.0_f64; n];
        let spec_before = spectrum_from_fit(data.view(), &w_on, &phi, 0.0025, 0.0, None).unwrap();

        // Append M rows on which the atom's gate is OFF (weight 0). n_obs grows to N+M
        // but the atom sees none of them: N_eff, the Gram, and the decoder are unchanged.
        let m_extra = 600usize;
        let n_aug = n + m_extra;
        let mut turns_aug = turns.clone();
        let mut data_aug = Array2::<f64>::zeros((n_aug, p));
        data_aug.slice_mut(ndarray::s![0..n, ..]).assign(&data);
        for _ in 0..m_extra {
            turns_aug.push(lcg(&mut s)); // arbitrary chart position; gate is off anyway
        }
        // Fill the appended rows with arbitrary (nonzero) reconstruction targets to
        // prove the invariance is due to the GATE (weight 0), not zero data.
        for i in n..n_aug {
            for j in 0..p {
                data_aug[[i, j]] = lcg_normal(&mut s);
            }
        }
        let phi_aug = harmonic_phi(&turns_aug, 3);
        let mut w_aug = vec![1.0_f64; n];
        w_aug.extend(std::iter::repeat(0.0_f64).take(m_extra));
        let spec_after =
            spectrum_from_fit(data_aug.view(), &w_aug, &phi_aug, 0.0025, 0.0, None).unwrap();

        // THE AXIOM: the fixed charge is bit-identical across the inert-row append.
        assert_eq!(
            spec_before.production_charge(),
            spec_after.production_charge(),
            "inert (gate-off) rows must not change the rank charge: before={} after={}",
            spec_before.production_charge(),
            spec_after.production_charge()
        );
        // N_eff is invariant (that is WHY the charge is); n_obs is not.
        assert_eq!(spec_before.n_eff, spec_after.n_eff);
        // FALSIFY THE OLD SCALE: under ½·d_eff·ln(n_obs) the same append would have
        // inflated the charge (rank_chargeable>0, n_obs grows N→N+M), so the two scales are
        // genuinely different and this test would fail on the pre-#2a code.
        assert!(
            spec_after.production_chargeable_rank() > 0,
            "fixture must have a real chargeable atom"
        );
        let old_before = 0.5
            * spec_before.production_chargeable_rank() as f64
            * spec_before.basis_edf
            * (n as f64).ln();
        let old_after = 0.5
            * spec_after.production_chargeable_rank() as f64
            * spec_after.basis_edf
            * (n_aug as f64).ln();
        assert!(
            old_after > old_before + 1e-6,
            "the OLD global-n scale WOULD have inflated the charge on inert rows \
             (old_before={old_before:.4} old_after={old_after:.4}); the fix removes exactly this"
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
            dispersion: 1.0,
            basis_edf: 3.0,
            n_eff: 50.0,
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

    /// Theorem K SOFT-LEDGER regime test: the unified charge `λ(N_eff)·ln N_eff` with
    /// the WBIC soft λ (= `wbic_charge`) must (a) REDUCE to the hard rank charge away
    /// from the MP edge — a clean circle whose two directions sit far above the edge —
    /// and (b) sit STRICTLY BELOW the hard charge near the edge — a filled disk whose
    /// directions pile just above the edge, where the hard count prices each as a full
    /// unit but the tempered count discounts them. This is the finite-n Watanabe
    /// correction reported by this audit-only module.
    #[test]
    fn soft_ledger_reduces_to_hard_away_from_edge_and_undercuts_near_it() {
        let n = 1200usize;
        let p = 16usize;

        // (A) CLEAN circle far above the edge ⇒ soft ≈ hard.
        let mut s = 0xA1A1_u64;
        let turns: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
        let phi = harmonic_phi(&turns, 3);
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let a = std::f64::consts::TAU * turns[i];
            data[[i, 0]] += a.cos();
            data[[i, 1]] += a.sin();
            for j in 0..p {
                data[[i, j]] += 0.05 * lcg_normal(&mut s);
            }
        }
        let w = vec![1.0_f64; n];
        let clean = spectrum_from_fit(data.view(), &w, &phi, 0.0025, 0.0, None).unwrap();
        // Soft charge is WBIC; hard production charge uses the chargeable rank.
        // Reduction: within 5%.
        assert!(clean.production_charge() > 0.0);
        // soft ≈ hard away from the edge: the two strong directions each count ≈1 under
        // the sigmoid, so the ratio sits near 1 (it can be marginally ABOVE 1 because
        // the far-below-edge harmonics still add a little soft mass the hard step drops).
        let clean_ratio = clean.wbic_charge() / clean.production_charge();
        assert!(
            (clean_ratio - 1.0).abs() < 0.05,
            "clean circle: soft ledger must reduce to the hard charge away from the edge \
             (ratio soft/hard={clean_ratio:.3})"
        );

        // (B) FILLED disk piling directions just above the edge ⇒ soft < hard.
        let mut s = 0xB2B2_u64;
        let turns: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
        let radii: Vec<f64> = (0..n).map(|_| lcg(&mut s).sqrt()).collect();
        let phi = harmonic_phi(&turns, 3);
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let a = std::f64::consts::TAU * turns[i];
            data[[i, 0]] += 0.35 * radii[i] * a.cos();
            data[[i, 1]] += 0.35 * radii[i] * a.sin();
            for j in 0..p {
                data[[i, j]] += 0.12 * lcg_normal(&mut s);
            }
        }
        let disk = spectrum_from_fit(data.view(), &w, &phi, 0.12 * 0.12, 0.0, None).unwrap();
        assert!(
            disk.mp_reconstruction_rank() > 0
                && disk.rank_soft() < disk.mp_reconstruction_rank() as f64,
            "disk fixture must have above-edge directions the tempered count discounts: \
             hard={} soft={:.3}",
            disk.mp_reconstruction_rank(),
            disk.rank_soft()
        );
        assert!(
            disk.wbic_charge() < disk.production_charge(),
            "disk: soft ledger must undercut the hard charge near the edge \
             (soft={:.4} hard={:.4})",
            disk.wbic_charge(),
            disk.production_charge()
        );
        // The discount must be materially larger for the near-edge disk than for the
        // far-from-edge circle — the whole point of the soft regime.
        let disk_ratio = disk.wbic_charge() / disk.production_charge();
        assert!(
            disk_ratio < clean_ratio - 0.1,
            "near-edge soft/hard ratio ({disk_ratio:.3}) must be clearly below the \
             far-from-edge ratio ({clean_ratio:.3})"
        );
    }
}

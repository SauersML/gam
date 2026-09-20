//! Rank-charge strata (#2933 F30–F32).
//!
//! WHAT PRODUCTION CHARGES. The penalized quasi-Laplace criterion adds, per atom `k`,
//!
//! ```text
//! C_k    = ½ · r_k · edf_k · log max(N_eff,k, 1)
//! N_eff,k = Σ_i a_ik²,   G_k = Φ_kᵀ diag(a_k²) Φ_k,   edf_k = tr((G_k + λ_k S_k)⁻¹ G_k)
//! μ_j    = σ_j(G_k^½ B_k)² / N_eff,k,   e = R · (1 + √(p / N_eff,k))²
//! r_k    = #{j : μ_j > e}, promoted to 1 when that count is 0 but some μ_j > 0 (#2258)
//! ```
//!
//! This is a named criterion convention: a BIC-shaped charge on a thresholded
//! reconstruction rank. It is not a WBIC value, and nothing in this crate proves that
//! `r_k · edf_k` equals a real log-canonical threshold. The physical reconstruction
//! rank `#{μ_j > e}`, the chargeable rank `r_k`, the storage dimension `m · p`, the
//! basis EDF `edf_k`, the intrinsic manifold dimension and the RLCT are different
//! quantities.
//!
//! WHAT THE EDGE IS. `e` is the upper Marchenko–Pastur edge of the sample covariance
//! of an `N_eff × p` matrix of independent variance-`R` noise. The spectrum it
//! thresholds is not such a matrix. Conditional on coordinates, gates, dispersion and
//! the other atoms, a noise-only target `E` with independent `N(0, R)` entries gives
//! the ridge decoder `B̂ = (G + λS)⁻¹ Φᵀ diag(a) E`, so `M = G^½ B̂` has `p`
//! independent `N(0, R·C)` columns with `C = G^½ (G+λS)⁻¹ G (G+λS)⁻¹ G^½`, and
//! `M Mᵀ ~ Wishart_m(p, R·C)`. That law is scaled by `m`, `p` and `C`, not by
//! `N_eff`, and coordinate/gate fitting and selection add dependence it omits. `e`
//! is therefore a rank diagnostic, not a calibrated false-rank boundary or an
//! evidence dimension. The tests in `tests_wbic_rank_charge_2933` evaluate that
//! conditional law (its exact expected total energy, a Chevet/Poincaré bound on the
//! expected top energy and a Gaussian-concentration bound on `P(max_j μ_j > τ)`)
//! and measure the false-rank rate of this producer against it by Monte Carlo.
//!
//! BRANCHES. `r_k` is an integer, so `C_k` is piecewise smooth in the fitted state:
//! it jumps by `½ · edf_k · log N_eff,k` for every unit change of `r_k` when a
//! direction crosses `μ_j = e`. `rank_charge_stratum` is the one producer of that
//! branch: the value (`realised_rank_charge_dof`) and its within-branch analytic
//! derivative (`production_rank_charge_derivative`) both classify through it, so
//! they price one branch of one state.

use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh, FaerSvd};
use ndarray::Array2;

use super::Side;
use super::construction::{
    ReconstructionRankClassification, certified_basis_edf, certified_psd_spectrum,
    classify_reconstruction_rank, normalized_reconstruction_energy, validate_rank_charge_problem,
};

/// One atom's priced rank-charge branch at one state: the per-observation
/// reconstruction energies `μ_j`, the Marchenko–Pastur rank edge `e` they are
/// classified against, the ridge-trace basis EDF and the two integer ranks.
#[derive(Clone, Debug)]
pub(crate) struct RankChargeStratum {
    energies: Vec<f64>,
    edge: f64,
    /// Ridge-trace basis EDF `tr((G + λS)⁻¹ G)`.
    basis_edf: f64,
    classification: ReconstructionRankClassification,
}

impl RankChargeStratum {
    /// Per-observation reconstruction energies `μ_j`, in singular-value order.
    pub(crate) fn reconstruction_energies(&self) -> &[f64] {
        &self.energies
    }

    /// Largest reconstruction energy, zero when there is none.
    pub(crate) fn top_reconstruction_energy(&self) -> f64 {
        self.classification.top_signal
    }

    /// The Marchenko–Pastur rank edge `R·(1+√(p/N_eff))²`.
    pub(crate) fn mp_reconstruction_rank_edge(&self) -> f64 {
        self.edge
    }

    /// `#{j : μ_j > e}`, the physical reconstruction rank at the edge.
    pub(crate) fn mp_reconstruction_rank(&self) -> usize {
        self.classification.mp_reconstruction_rank
    }

    /// The rank the criterion charges, including the #2258 promotion of an alive
    /// decoder with MP rank zero to rank one.
    pub(crate) fn production_chargeable_rank(&self) -> usize {
        self.classification.production_chargeable_rank
    }

    /// `r · edf`, the DOF the criterion multiplies by `½ log max(N_eff, 1)`.
    pub(crate) fn production_dof(&self) -> f64 {
        self.production_chargeable_rank() as f64 * self.basis_edf
    }
}

/// Classify one atom's rank-charge branch from its weighted basis Gram
/// `gram = Φᵀdiag(a²)Φ` (`m×m`), decoder `D` (`m×p`), occupancy `n_eff = Σ_row a²`,
/// output width `p_out`, dispersion `r_floor` and smoothing `(lam_smooth,
/// smooth_penalty)`. The production value and its analytic derivative both call
/// this, so they price one branch of one state.
pub(crate) fn rank_charge_stratum(
    gram: &Array2<f64>,
    decoder: &Array2<f64>,
    n_eff: f64,
    p_out: f64,
    r_floor: f64,
    lam_smooth: f64,
    smooth_penalty: Option<&Array2<f64>>,
) -> Result<RankChargeStratum, String> {
    let m = gram.nrows();
    validate_rank_charge_problem(
        gram,
        decoder,
        n_eff,
        p_out,
        r_floor,
        lam_smooth,
        smooth_penalty,
    )?;
    if m == 0 || n_eff == 0.0 {
        return Ok(RankChargeStratum {
            energies: Vec::new(),
            edge: 0.0,
            basis_edf: 0.0,
            classification: classify_reconstruction_rank(&[], 0.0),
        });
    }
    // MP reconstruction rank on the reconstruction Gram. U orthogonal ⇒ svd of
    // diag(√λ)·Uᵀ·D equals svd of the reconstruction square root G^½·D.
    let (evals, u) = gram
        .eigh(Side::Lower)
        .map_err(|e| format!("rank-charge stratum: eigh(G): {e}"))?;
    let evals = certified_psd_spectrum(evals.view(), "rank-charge Gram")?;
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
        Err(e) => return Err(format!("rank-charge stratum: recon svd: {e}")),
    };
    let edge = crate::null_battery::mp_reconstruction_rank_edge(n_eff, p_out, r_floor)
        .map_err(|error| format!("rank-charge stratum: {error}"))?;
    let energies = sv
        .iter()
        .map(|&singular_value| normalized_reconstruction_energy(singular_value, n_eff))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("rank-charge stratum: {error}"))?;
    let classification = classify_reconstruction_rank(&energies, edge);
    // basis_edf = tr(gram·(gram+λS)⁻¹).
    let factor = penalized_gram(gram, lam_smooth, smooth_penalty)
        .cholesky(Side::Lower)
        .map_err(|error| {
            format!("rank-charge stratum: G + lambda*S is not positive definite: {error}")
        })?;
    let x = factor.solve_mat(gram); // X = (G+λS)⁻¹ G
    let raw_basis_edf = (0..m).map(|i| x[[i, i]]).sum::<f64>();
    let basis_edf = certified_basis_edf(raw_basis_edf, m, "rank-charge stratum")?;
    Ok(RankChargeStratum {
        energies,
        edge,
        basis_edf,
        classification,
    })
}

fn penalized_gram(
    gram: &Array2<f64>,
    lam_smooth: f64,
    smooth_penalty: Option<&Array2<f64>>,
) -> Array2<f64> {
    let mut penalized = gram.clone();
    if let Some(penalty) = smooth_penalty {
        penalized.scaled_add(lam_smooth, penalty);
    }
    penalized
}

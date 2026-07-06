use super::*;

/// Cap on the number of coordinates at which a per-atom shape band is
/// materialized. The full per-atom decoder covariance is exact and exposed
/// regardless; this only bounds the cost of the convenience band, which is
/// evaluated at an evenly-strided subset of the atom's own on-atom coordinates.
pub const SHAPE_BAND_MAX_POINTS: usize = 512;

/// Entry budget for materializing one atom's dense `(M_k·p)²` decoder
/// covariance in the fit payload. Above it (LLM-scale ambient `p`) the band
/// quantities are computed exactly from the factored frame covariance and the
/// dense export is omitted (`decoder_covariance: None`) — the python reader
/// treats it as optional. 2^24 f64 entries = 128 MiB per atom.
pub const SAE_DECODER_COV_PAYLOAD_MAX_ENTRIES: usize = 1 << 24;

/// Posterior uncertainty of one fitted atom's manifold shape.
///
/// In the primary path — [`SaeManifoldTerm::assemble_shape_uncertainty`], and
/// after a structure/finalization change its final-state twin
/// [`SaeManifoldTerm::recompute_joint_shape_uncertainty`] — the covariance is
/// the φ-scaled β-block of the JOINT inverse Hessian (coordinates marginalized
/// out); the band is its closed-form push-forward through the linear
/// basis→ambient map `m_k(t) = Φ_k(t)·B_k`. This is what the production fit
/// returns.
///
/// Two reachable fallbacks produce a WEAKER band and say so honestly at the
/// field level (`decoder_covariance: None`, and the `band_sd` doc below): the
/// per-atom Laplace marginal [`SaeManifoldTerm::complete_born_atom_shape_bands`]
/// fills (used only when the joint factor genuinely could not cover an atom —
/// it DROPS cross-atom covariance and the decoder-coordinate Schur couplings, so
/// its `band_sd` is identical across output channels), and the all-NaN
/// [`SaeManifoldTerm::shape_uncertainty_without_decoder_covariance`] (LLM-scale
/// fits with no dense joint Schur). Neither is silently dressed up as the joint
/// covariance.
#[derive(Debug, Clone)]
pub struct SaeAtomShapeUncertainty {
    /// φ-scaled posterior covariance of this atom's decoder coefficients,
    /// `Cov(β_k) = φ·S_β⁻¹[block_k]`, shape `(M_k·p, M_k·p)` in the decoder's
    /// row-major `(basis, channel)` flat layout (flat index `b·p + c`).
    ///
    /// `None` when materializing it would exceed
    /// [`SAE_DECODER_COV_PAYLOAD_MAX_ENTRIES`] (LLM-scale ambient `p`: at
    /// `(M=8, p=2048)` the dense block is 2 GiB *per atom*, at
    /// `(M=16, p=5120)` ~50 GiB). The band quantities below are still exact
    /// in that case — they are computed directly from the factored
    /// `(M_k·r_k)²` frame covariance without ever lifting it.
    pub decoder_covariance: Option<Array2<f64>>,
    /// Coordinates at which the band is evaluated, shape `(G, d_k)`.
    pub band_coords: Array2<f64>,
    /// Fitted ambient point `m_k(t) = Φ_k(t)·B_k` at each band coordinate,
    /// shape `(G, p)`.
    pub band_mean: Array2<f64>,
    /// Posterior standard deviation of each ambient channel at each band
    /// coordinate, `sqrt(Var_c(t))` with
    /// `Var_c(t) = Σ_{b1,b2} Φ[b1] Φ[b2] Cov(β_k)[(b1,c),(b2,c)]`, shape
    /// `(G, p)`.
    ///
    /// This is the MODEL-BASED band (`Cov = φ̂ H⁻¹`), correct only when the
    /// working reconstruction likelihood is correctly specified. See
    /// [`Self::band_sd_robust`] for the misspecification-robust companion.
    pub band_sd: Array2<f64>,
    /// Sandwich (Godambe / robust) posterior standard deviation of each ambient
    /// channel, same shape as [`Self::band_sd`], computed from the within-channel
    /// sandwich covariance `A_c⁻¹ J_cc A_c⁻¹` (see [`super::sandwich`]). Reported
    /// ALONGSIDE the model-based band, never in place of it; the two coincide
    /// when the information-matrix equality holds and diverge exactly to the
    /// degree the residuals violate the working likelihood (heteroskedastic
    /// tokens, LayerNorm structure, cross-channel template correlation).
    ///
    /// `None` when the robust band was not requested for this atom or could not
    /// be formed (e.g. the huge-`p` factored path where the dense within-channel
    /// meat is not materialized) — an honest gap, not a fabricated band.
    pub band_sd_robust: Option<Array2<f64>>,
}

/// Posterior shape uncertainty for a whole SAE-manifold fit: one band per atom
/// plus the shared Gaussian reconstruction dispersion `φ̂` used to scale every
/// covariance. See [`SaeManifoldTerm::assemble_shape_uncertainty`].
#[derive(Debug, Clone)]
pub struct SaeShapeUncertainty {
    /// Gaussian reconstruction scale `φ̂ = RSS / residual-dof`.
    pub dispersion: f64,
    /// One entry per atom, in atom order.
    pub atoms: Vec<SaeAtomShapeUncertainty>,
}

impl SaeShapeUncertainty {
    /// #1230 — invalidate every PRE-search seed band as the FALLBACK recompute
    /// path, when the joint inverse-Hessian factor cannot be reformed at the
    /// FINAL post-structure-search model state.
    ///
    /// The production fit assembles these bands from the joint Hessian at the ρ
    /// the OUTER optimizer settled on, BEFORE evidence-guarded structure search.
    /// When a structure move lands (a certified birth/fission/fusion, or a demoted
    /// death), the warm refit re-converges the WHOLE dictionary at a new ρ, so the
    /// seed atoms' decoders / coordinates / inner curvature change and their
    /// pre-search joint-Hessian bands no longer describe the returned model. The
    /// PRIMARY response is to rebuild the JOINT bands at the final state via
    /// [`SaeManifoldTerm::recompute_joint_shape_uncertainty`], which preserves the
    /// cross-atom covariance and coordinate-Schur couplings and the per-channel
    /// variance.
    ///
    /// This method is the fallback for when that joint recompute is unavailable
    /// (a non-PD post-search Hessian, or a huge-`p` fit with no dense Schur). It
    /// resets each existing band's posterior `band_sd` to `NaN` and drops the
    /// stale dense `decoder_covariance`, so the subsequent
    /// [`SaeManifoldTerm::complete_born_atom_shape_bands`] pass — which fills every
    /// `NaN` band from each atom's OWN final penalized inner Hessian
    /// `H_k = Φ_kᵀ W_k Φ_k + S̃_k` harvested at the settled post-search state —
    /// recomputes a PER-ATOM MARGINAL band for EVERY atom (seed and born) against
    /// the final model. That marginal drops the cross-atom / coordinate couplings
    /// and is identical across output channels, so it is a strict approximation to
    /// the joint covariance the primary path returns — honest, never fabricated. A
    /// genuinely-degenerate atom keeps an honest `NaN` band rather than a stale
    /// fabricated one. `band_coords` / `band_mean` are re-derived directly from the
    /// final fitted atom by the completion pass, so they stay consistent too.
    ///
    /// No-op when the structure did not change (every atom keeps its exact
    /// joint-Hessian band, which is still valid and strictly higher quality than
    /// the per-atom Laplace approximation).
    pub fn invalidate_bands_for_recompute(&mut self) {
        for atom in &mut self.atoms {
            atom.decoder_covariance = None;
            atom.band_sd.fill(f64::NAN);
            atom.band_sd_robust = None;
        }
    }
}

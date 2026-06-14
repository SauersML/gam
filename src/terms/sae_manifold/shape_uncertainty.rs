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
/// Produced by [`SaeManifoldTerm::assemble_shape_uncertainty`]. The covariance
/// is the φ-scaled β-block of the joint inverse Hessian (coordinates
/// marginalized out); the band is its closed-form push-forward through the
/// linear basis→ambient map `m_k(t) = Φ_k(t)·B_k`.
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
    pub band_sd: Array2<f64>,
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

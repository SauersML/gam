//! SAE front-door lane admission.
//!
//! The canonical large-`K` training state is the sparse code state
//! `(indices[N, s], codes[N, s])`. The dense manifold engine remains available,
//! but only as the small-`K` certification lane: once the dense routing state
//! `N×K` is larger than the response matrix scale `N×P`, the front door admits
//! the sparse/block lane instead of constructing a dense assignment object.

/// Training lane selected by [`admit_sae_fit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaeFitLane {
    /// Dense exact manifold engine, for small dictionaries/certification.
    DenseCertification,
    /// Sparse-code/block lane, with no `N×K` training state.
    SparseCodes,
}

/// Auditable admission decision at a fit entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaeFitAdmission {
    /// Selected lane.
    pub lane: SaeFitLane,
    /// Number of observations.
    pub n_obs: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Requested atom count.
    pub n_atoms: usize,
    /// Cells in the dense assignment state, `N*K`.
    pub dense_assignment_cells: usize,
    /// Cells in the response matrix, `N*P`.
    pub response_cells: usize,
}

impl SaeFitAdmission {
    /// True when the sparse-code lane is selected.
    pub fn uses_sparse_codes(&self) -> bool {
        self.lane == SaeFitLane::SparseCodes
    }
}

/// Decide the fit lane from shape alone.
///
/// Dense certification is admitted only while `N*K <= N*P`; equivalently
/// `K <= P`. This is the front-door enforcement of the no-`N×K` architecture:
/// the dense assignment state is not allowed to become larger than the actual
/// activation matrix by default.
pub fn admit_sae_fit(
    n_obs: usize,
    output_dim: usize,
    n_atoms: usize,
) -> Result<SaeFitAdmission, String> {
    if n_obs == 0 || output_dim == 0 || n_atoms == 0 {
        return Err(format!(
            "admit_sae_fit requires positive N, P, and K; got N={n_obs}, P={output_dim}, K={n_atoms}"
        ));
    }
    let dense_assignment_cells = n_obs.saturating_mul(n_atoms);
    let response_cells = n_obs.saturating_mul(output_dim);
    let lane = if dense_assignment_cells <= response_cells {
        SaeFitLane::DenseCertification
    } else {
        SaeFitLane::SparseCodes
    };
    Ok(SaeFitAdmission {
        lane,
        n_obs,
        output_dim,
        n_atoms,
        dense_assignment_cells,
        response_cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_demotes_dense_when_assignment_state_exceeds_response() {
        let small = admit_sae_fit(1024, 4096, 128).expect("small admission");
        assert_eq!(small.lane, SaeFitLane::DenseCertification);
        let large = admit_sae_fit(1024, 4096, 32_000).expect("large admission");
        assert_eq!(large.lane, SaeFitLane::SparseCodes);
        assert!(large.dense_assignment_cells > large.response_cells);
    }
}

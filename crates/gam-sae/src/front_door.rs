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

/// Front-door enforcement for the DENSE manifold engine (#985 / E1): admit the
/// dense-certification lane, or REFUSE the sparse lane.
///
/// The dense manifold representation — the `N×K` routing state materialized as a
/// dense assignment ([`crate::assignment::SaeAssignment`]) — is the small-`K`
/// CERTIFICATION lane only. This returns the [`SaeFitAdmission`] when
/// [`admit_sae_fit`] selects [`SaeFitLane::DenseCertification`] (`N·K ≤ N·P`, i.e.
/// `K ≤ P`), and an `Err` — naming the sparse-code lane — when it demotes to
/// [`SaeFitLane::SparseCodes`] (`K > P`), the shape whose dense `N×K` state the
/// front door exists to avoid. A dense-engine entry point (e.g. the manifold-fit
/// FFI) calls this at its top so a caller that bypassed the sparse front door
/// cannot silently build the `N×K` state. Refuse-on-SPARSE only: the dense-cert
/// path (`K ≤ P`) passes through unchanged, carrying its admission for audit. A
/// degenerate `N`/`P`/`K` is rejected here too (propagated from [`admit_sae_fit`]).
pub fn admit_dense_certification(
    n_obs: usize,
    output_dim: usize,
    n_atoms: usize,
) -> Result<SaeFitAdmission, String> {
    let admission = admit_sae_fit(n_obs, output_dim, n_atoms)?;
    if admission.uses_sparse_codes() {
        return Err(format!(
            "dense manifold engine refused: it is the small-K certification lane \
             (admitted only while N*K <= N*P, i.e. K <= P); N={n_obs}, P={output_dim}, K={n_atoms} \
             gives N*K={} > N*P={} — route this fit through the sparse-code lane instead of \
             building the dense N×K assignment state",
            admission.dense_assignment_cells, admission.response_cells
        ));
    }
    Ok(admission)
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

    /// The dense-engine guard admits the small-K certification lane (K ≤ P) and
    /// refuses the sparse lane (K > P), pointing the caller at the sparse-code
    /// lane — the direct-FFI misuse path (#14) that would otherwise silently build
    /// the dense N×K state.
    #[test]
    fn dense_certification_admits_k_le_p_and_refuses_sparse_k_gt_p() {
        // K ≤ P — the dense-certification lane — passes, returning its admission.
        let ok = admit_dense_certification(1024, 4096, 128).expect("K<P must be admitted");
        assert_eq!(ok.lane, SaeFitLane::DenseCertification);
        assert!(
            admit_dense_certification(64, 8, 8).is_ok(),
            "K == P is the DenseCertification boundary and must be admitted"
        );

        // K > P — the sparse lane — is refused, naming the sparse-code lane. This is
        // the p=4096, K=32000 acceptance shape: the dense N×K state is never built.
        let err = admit_dense_certification(1_000_000, 4096, 32_000)
            .expect_err("K > P must be refused by the dense-engine guard");
        assert!(
            err.contains("sparse-code lane"),
            "the refusal must point at the sparse-code lane; got: {err}"
        );

        // The crossover is exact: K == P admitted, K == P + 1 refused.
        assert!(admit_dense_certification(10, 100, 100).is_ok());
        assert!(admit_dense_certification(10, 100, 101).is_err());

        // A degenerate N/P/K is rejected at the door (propagated from admit_sae_fit).
        assert!(admit_dense_certification(0, 4, 3).is_err());
    }
}

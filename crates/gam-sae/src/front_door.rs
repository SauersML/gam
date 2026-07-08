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

/// Mode-aware admission for the HARD TOP-K SUPPORT gate
/// ([`crate::assignment::AssignmentMode::TopK`]).
///
/// The `K ≤ P` rule guards the no-`N×K` ARCHITECTURE for penalty-gated modes,
/// whose `N×K` logits are live Newton state. The TopK mode carries NO gate
/// coordinates (its logits are read-only routing inputs and its per-token
/// blocks are `k·(1+d)` by construction), so its admission question is not
/// architectural but CONCRETE MEMORY: does the dense state — `N·K` routing
/// logits plus `N·K·d_max` per-atom coordinates, 8 bytes each — fit the host
/// in-core budget ([`crate::manifold::streaming_plan`]'s available-memory
/// convention, SPEC "never OOM")? Within budget ⇒ the TRUE manifold engine is
/// admitted at any overcompleteness `K > P`. Over budget ⇒ a typed Err: for a
/// caller who asked for TOPK MANIFOLD atoms, silently substituting the linear
/// sparse-code lane is the exact failure mode this front door exists to
/// prevent (witnessed 2026-07-08: K>P fits returned `SparseDictionaryFit`
/// through the manifold entry for the project's whole history). The caller
/// reduces `n_obs` or awaits the support-sparse lane; nothing is substituted.
pub fn admit_topk_manifold(
    n_obs: usize,
    output_dim: usize,
    n_atoms: usize,
    d_max: usize,
    support_k: usize,
) -> Result<SaeFitAdmission, String> {
    let budget_bytes = crate::manifold::sae_host_in_core_budget_bytes().0;
    admit_topk_manifold_with_budget(n_obs, output_dim, n_atoms, d_max, support_k, budget_bytes)
}

/// Budget-parameterized core of [`admit_topk_manifold`] (testable without the
/// host memory probe).
pub(crate) fn admit_topk_manifold_with_budget(
    n_obs: usize,
    output_dim: usize,
    n_atoms: usize,
    d_max: usize,
    support_k: usize,
    budget_bytes: usize,
) -> Result<SaeFitAdmission, String> {
    let admission = admit_sae_fit(n_obs, output_dim, n_atoms)?;
    if support_k == 0 || support_k > n_atoms {
        return Err(format!(
            "admit_topk_manifold requires 1 <= support_k <= K={n_atoms}; got {support_k}"
        ));
    }
    if d_max == 0 {
        return Err("admit_topk_manifold requires d_max >= 1".to_string());
    }
    if admission.lane == SaeFitLane::DenseCertification {
        // K ≤ P: the historical certification lane already admits this shape.
        return Ok(admission);
    }
    let state_cells = n_obs
        .saturating_mul(n_atoms)
        .saturating_mul(1 + d_max);
    let state_bytes = state_cells.saturating_mul(std::mem::size_of::<f64>());
    if state_bytes <= budget_bytes {
        return Ok(SaeFitAdmission {
            lane: SaeFitLane::DenseCertification,
            ..admission
        });
    }
    Err(format!(
        "topk manifold engine refused: the dense state (N*K logits + N*K*d_max coords = \
         {state_bytes} bytes at N={n_obs}, K={n_atoms}, d_max={d_max}) exceeds the host \
         in-core budget ({budget_bytes} bytes). Reduce n_obs (row-subsample; the HT outer \
         subsampling keeps the criterion honest) or wait for the support-sparse massive-K \
         lane — a TOPK MANIFOLD request is never silently substituted with the linear \
         sparse-code lane"
    ))
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
    fn topk_manifold_admits_overcomplete_within_budget_and_refuses_over() {
        // K >> P (the massively-overcomplete shape) with a state that fits the
        // budget: N=4096, K=32000, d=1 → (1+1)·4096·32000·8 = 2.1 GB.
        let bytes = 2 * 4096usize * 32_000 * 8;
        let ok = admit_topk_manifold_with_budget(4096, 512, 32_000, 1, 8, bytes)
            .expect("within-budget topk admission");
        assert_eq!(
            ok.lane,
            SaeFitLane::DenseCertification,
            "TopK within budget admits the TRUE manifold engine at K > P"
        );

        // One byte under the state size: refused with the typed no-substitution
        // error, never silently demoted to the linear sparse-code lane.
        let err = admit_topk_manifold_with_budget(4096, 512, 32_000, 1, 8, bytes - 1)
            .expect_err("over-budget topk must refuse");
        assert!(
            err.contains("never silently substituted"),
            "refusal must state the no-substitution contract; got: {err}"
        );

        // K ≤ P passes through the historical certification admission untouched.
        let small = admit_topk_manifold_with_budget(1024, 4096, 128, 2, 4, 0)
            .expect("K <= P is admitted regardless of budget");
        assert_eq!(small.lane, SaeFitLane::DenseCertification);

        // Degenerate support sizes are caller errors.
        assert!(admit_topk_manifold_with_budget(64, 8, 16, 1, 0, bytes).is_err());
        assert!(admit_topk_manifold_with_budget(64, 8, 16, 1, 17, bytes).is_err());
    }

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

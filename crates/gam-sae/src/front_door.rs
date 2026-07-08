//! SAE front-door lane admission.
//!
//! The canonical large-`K` training state is the sparse code state
//! `(indices[N, s], codes[N, s])`. The dense manifold engine remains available,
//! but only as the small-`K` certification lane: once the dense routing state
//! `N×K` is larger than the response matrix scale `N×P`, the front door admits
//! the sparse/block lane instead of constructing a dense assignment object —
//! for PENALTY-GATED assignment modes, whose `N×K` logits are live Newton
//! state. The hard TopK support mode carries no gate coordinates, so its
//! `K > P` fits are admitted to the CURVED framed/streaming manifold lane
//! ([`SaeFitLane::CurvedStreaming`], budgeted by
//! [`crate::manifold::SaeTopKCurvedBudget`]) instead of being demoted to the
//! linear trainer.

/// Training lane selected by [`admit_sae_fit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaeFitLane {
    /// Dense exact manifold engine, for small dictionaries/certification.
    DenseCertification,
    /// Sparse-code/block lane, with no `N×K` training state.
    SparseCodes,
    /// Curved framed/streaming manifold lane for hard-TopK overcomplete fits
    /// (`K > P` with [`crate::assignment::AssignmentMode::TopK`]): per-row
    /// TopK active sets (`O(N·k_active)` assignment state), framed per-atom
    /// decoder blocks (`O(K·M·r)`), routing scored in row chunks — never a
    /// dense `N×K` live Newton state. Budget arithmetic:
    /// [`crate::manifold::SaeTopKCurvedBudget`].
    CurvedStreaming,
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
/// coordinates (its logits are read-only routing inputs and its per-row
/// blocks are `k·(1+d)` by construction), so a `K > P` TopK request is
/// admitted to the CURVED lane ([`SaeFitLane::CurvedStreaming`]) instead of
/// being demoted to the linear sparse-code trainer. The lane's memory ledger
/// is owned by the streaming plan
/// ([`crate::manifold::SaeTopKCurvedBudget`], SPEC "never OOM"):
///
///   * `K ≤ P` — the historical dense-certification admission, byte-identical;
///   * `K > P`, resident seed (`N·K·(1+d_max)·8` bytes) within the in-core
///     budget — admitted; today's engine runs the fit in core;
///   * `K > P`, resident seed over budget but the honest streaming shape
///     (`O(N·k_active)` active sets + `O(K·M·r)` framed decoder + chunked
///     routing window) within the streaming budget — a typed Err naming the
///     chunked-seed integration seam
///     ([`crate::manifold::admit_topk_curved_lane`]) the driver must wire;
///     admitting it before that wiring would OOM on the dense seed;
///   * over both budgets — a typed Err. For a caller who asked for TOPK
///     MANIFOLD atoms, silently substituting the linear sparse-code lane is
///     the exact failure mode this front door exists to prevent (witnessed
///     2026-07-08: K>P fits returned `SparseDictionaryFit` through the
///     manifold entry for the project's whole history). Nothing is
///     substituted.
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
        // K ≤ P: the historical certification lane already admits this shape,
        // byte-identical to the penalty-gated admission.
        return Ok(admission);
    }
    // K > P: the overcomplete curved lane. The streaming plan owns the ledger.
    let ledger = crate::manifold::sae_topk_curved_budget_from_budget(
        n_obs,
        output_dim,
        n_atoms,
        d_max,
        support_k,
        budget_bytes,
    );
    if ledger.resident_seed_admitted {
        return Ok(SaeFitAdmission {
            lane: SaeFitLane::CurvedStreaming,
            ..admission
        });
    }
    if ledger.streaming_admitted {
        return Err(format!(
            "topk curved lane: the dense routing seed ({} bytes at N={n_obs}, K={n_atoms}, \
             d_max={d_max}) exceeds the host in-core budget ({budget_bytes} bytes), but the \
             streamed curved shape fits (peak {} <= streaming budget {} bytes; active sets \
             O(N*k_active)={}, framed decoder O(K*M*r)={}). This region needs the chunked-seed \
             driver wiring: fit_drivers must call \
             gam_sae::manifold::admit_topk_curved_lane(n_obs, output_dim, n_atoms, d_max, \
             support_k) before building any dense (N, K) seed and stream the routing seed in \
             seed_chunk_rows() row chunks. Until that lands, reduce n_obs (row-subsample; the \
             HT outer subsampling keeps the criterion honest) so the resident seed fits — a \
             TOPK MANIFOLD request is never silently substituted with the linear sparse-code \
             lane",
            ledger.resident_seed_bytes,
            ledger.streaming_peak_bytes,
            ledger.streaming_budget_bytes,
            ledger.active_state_bytes,
            ledger.framed_decoder_bytes,
        ));
    }
    Err(format!(
        "topk manifold engine refused: the dense seed ({} bytes) exceeds the host in-core \
         budget ({budget_bytes} bytes) AND the streamed curved shape (peak {} bytes) exceeds \
         the streaming budget ({} bytes) at N={n_obs}, K={n_atoms}, k_active={support_k}, \
         d_max={d_max}. Reduce n_obs (row-subsample; the HT outer subsampling keeps the \
         criterion honest) or reduce support_k — a TOPK MANIFOLD request is never silently \
         substituted with the linear sparse-code lane",
        ledger.resident_seed_bytes, ledger.streaming_peak_bytes, ledger.streaming_budget_bytes,
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

    /// The mission's three admission cases, at the Rust owner of the rule:
    /// K>P + TopK → CURVED lane admitted (with the documented streaming-plan
    /// ledger); K>P + penalty-gated (softmax) → sparse-code lane unchanged;
    /// K≤P → dense-certification lane unchanged.
    #[test]
    fn topk_manifold_admits_overcomplete_to_curved_lane_and_refuses_over() {
        // K >> P (the massively-overcomplete shape) with a resident seed that
        // fits the budget: N=4096, K=32000, d=1 → (1+1)·4096·32000·8 = 2.1 GB.
        let bytes = 2 * 4096usize * 32_000 * 8;
        let ok = admit_topk_manifold_with_budget(4096, 512, 32_000, 1, 8, bytes)
            .expect("within-budget topk admission");
        assert_eq!(
            ok.lane,
            SaeFitLane::CurvedStreaming,
            "TopK within budget admits the CURVED manifold lane at K > P"
        );
        // The admission carries the audit shape untouched.
        assert_eq!((ok.n_obs, ok.output_dim, ok.n_atoms), (4096, 512, 32_000));

        // The admitted region matches the streaming plan's documented ledger:
        // the resident-seed gate is the same arithmetic the plan owns.
        let ledger =
            crate::manifold::sae_topk_curved_budget_from_budget(4096, 512, 32_000, 1, 8, bytes);
        assert!(ledger.resident_seed_admitted);
        assert_eq!(ledger.resident_seed_bytes, bytes);

        // One byte under the resident seed: this shape is exactly the chunked-
        // seed integration region — refused with the actionable seam-naming
        // error, never silently demoted to the linear sparse-code lane.
        let err = admit_topk_manifold_with_budget(4096, 512, 32_000, 1, 8, bytes - 1)
            .expect_err("over-resident-budget topk must refuse until the seam is wired");
        assert!(
            err.contains("never silently substituted"),
            "refusal must state the no-substitution contract; got: {err}"
        );

        // The same K>P shape under a PENALTY-GATED assignment (softmax et al.,
        // which go through `admit_sae_fit`) keeps the sparse-code lane: the
        // new branch never moves the existing reroute.
        let softmax = admit_sae_fit(4096, 512, 32_000).expect("softmax admission");
        assert_eq!(softmax.lane, SaeFitLane::SparseCodes);

        // K ≤ P passes through the historical certification admission
        // untouched — byte-identical to the penalty-gated dense lane.
        let small = admit_topk_manifold_with_budget(1024, 4096, 128, 2, 4, 0)
            .expect("K <= P is admitted regardless of budget");
        assert_eq!(small.lane, SaeFitLane::DenseCertification);
        assert_eq!(
            small,
            admit_sae_fit(1024, 4096, 128).expect("dense admission")
        );

        // Degenerate support sizes are caller errors.
        assert!(admit_topk_manifold_with_budget(64, 8, 16, 1, 0, bytes).is_err());
        assert!(admit_topk_manifold_with_budget(64, 8, 16, 1, 17, bytes).is_err());
    }

    /// The chunked-seed region (dense seed over budget, streamed shape within)
    /// names the ONE integration seam the driver must wire, and the fully
    /// over-budget region refuses with the ledger's figures. Both refuse — a
    /// TopK request is never substituted.
    #[test]
    fn topk_manifold_seam_region_names_the_integration_call() {
        // N=1e6, K=32000, d=1: resident seed = 512 GB, streamed peak ≈ 9.6 GiB
        // (dominated by the framed border workspace).
        let ledger = crate::manifold::sae_topk_curved_budget_from_budget(
            1_000_000,
            512,
            32_000,
            1,
            8,
            16 * 1024 * 1024 * 1024,
        );
        assert!(!ledger.resident_seed_admitted);
        assert!(ledger.streaming_admitted);
        let err =
            admit_topk_manifold_with_budget(1_000_000, 512, 32_000, 1, 8, 16 * 1024 * 1024 * 1024)
                .expect_err("seam region refuses until the chunked-seed driver lands");
        assert!(
            err.contains("admit_topk_curved_lane"),
            "the refusal must name the integration seam; got: {err}"
        );
        assert!(err.contains("never silently substituted"));

        // Over both budgets: the plain refusal, still no substitution.
        let err = admit_topk_manifold_with_budget(1_000_000, 512, 32_000, 1, 8, 0)
            .expect_err("over-both-budgets topk must refuse");
        assert!(err.contains("never silently substituted"));
        assert!(!err.contains("admit_topk_curved_lane"));
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

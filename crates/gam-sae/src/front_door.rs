//! SAE front-door lane admission — ONE engine, admitted by memory layout and
//! model request (design gam#2232, increments 1–6).
//!
//! There is one SAE fit engine: the inner arrow-Schur Newton over per-row
//! active sets `(indices, gates, coords)`, the outer REML evidence loop, and
//! the birth/death migration ledger. The lanes this door selects are NOT
//! different models — they are memory-layout admissions and solver
//! specializations of that engine:
//!
//! * [`SaeFitLane::DenseCertification`] — the full-support materialization
//!   (`N×K` assignment state), admitted only while it is no larger than the
//!   response (`K ≤ P`): the small-`K` certification lane.
//! * [`SaeFitLane::SparseCodes`] — the fixed-support LINEAR specialization:
//!   for genuinely linear atoms with read-only gates, the arrow-Schur inner
//!   solve degenerates to the `s×s` active-set ridge / block-projection fast
//!   kernels (`sparse_dict`), with the linear block's ONE variance component
//!   REML-selected by the shared-ρ schedule. Selected by SHAPE for
//!   penalty-gated `K > P` requests (whose `N×K` logits are live Newton state
//!   the architecture forbids), and by REQUEST at any `K` for an explicit
//!   linear-dictionary model ([`admit_linear_dictionary`]).
//! * [`SaeFitLane::CurvedStreaming`] — the same curved engine under the
//!   streaming memory layout: per-row TopK active sets, framed decoders,
//!   chunked routing; admitted for hard-TopK `K > P` within the concrete host
//!   budget ([`crate::manifold::SaeTopKCurvedBudget`]) and REFUSED loudly over
//!   it — a curved manifold request is never silently substituted with the
//!   linear kernel, and a linear request is never forced onto the dense
//!   engine. Tiering is a seed policy + alternation cadence of the same
//!   engine (`tiered`), not a lane; the torch lane is declared interop and
//!   untouched.

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
///     budget — admitted; the engine runs the fit with the dense seed in core;
///   * `K > P`, resident seed over budget but the honest streaming shape
///     (`O(N·k_active)` active sets + `O(K·M·r)` framed decoder + chunked
///     routing window) within the streaming budget — ALSO admitted to the
///     CURVED lane: the chunked-seed driver is wired
///     ([`crate::manifold::admit_topk_curved_lane`] +
///     `SaeManifoldTerm::seed_cold_start_disjoint_charts_streaming` +
///     `SaeManifoldTerm::fit_topk_curved_streaming`), so the seed is built by
///     accumulating each atom's decoder normal equations in
///     `seed_chunk_rows()` row chunks — never a resident dense `(N, K)` seed;
///   * over BOTH budgets — a typed Err. For a caller who asked for TOPK
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
    // Both the resident sub-lane (dense seed fits in core) AND the streaming
    // sub-lane (dense seed over budget, streamed curved shape fits) are runnable:
    // the chunked-seed driver is wired
    // ([`crate::manifold::SaeManifoldTerm::seed_cold_start_disjoint_charts_streaming`]
    // + [`crate::manifold::SaeManifoldTerm::fit_topk_curved_streaming`]), so the
    // streaming region no longer builds any dense `(N, K)` seed — it accumulates
    // the per-atom decoder normal equations in `seed_chunk_rows()` row chunks.
    // Only a shape past BOTH budgets refuses.
    if ledger.resident_seed_admitted || ledger.streaming_admitted {
        return Ok(SaeFitAdmission {
            lane: SaeFitLane::CurvedStreaming,
            ..admission
        });
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

/// MODELING-CHOICE admission for an EXPLICIT linear-dictionary request
/// (design gam#2232, Increment 5b — the Gap-B resolution).
///
/// The `K ≤ P → dense / K > P → sparse` rule of [`admit_sae_fit`] is the
/// DEFAULT admission for penalty-gated MANIFOLD requests: it guards the
/// no-`N×K` architecture, and for those callers the linear sparse-code lane is
/// a demotion. But a caller who EXPLICITLY requests a linear dictionary
/// (every atom the genuinely linear `d`-dimensional Euclidean atom — the
/// `max_degree = 1` specialization of the one engine, with hard top-k support)
/// is asking for exactly the model the sparse-code lane's fixed-support
/// alternating solve IS the fast kernel of (Increment 2, plug points 1–3):
/// the `s×s` active-set ridge is the degenerate arrow-Schur inner solve for
/// read-only gates on linear atoms. That model is legitimate at ANY `K` —
/// including `K ≤ P`, where the historical gate wrongly forced the dense
/// engine — so this admission returns [`SaeFitLane::SparseCodes`]
/// unconditionally (shape validation only). `block_size ≥ 2` (uniform
/// Euclidean `d = b` atoms) is the Grassmann block lane: framed `d = b` atoms
/// (`B_k = C_k·U_kᵀ`, `U_k ∈ Gr(b, P)`) with block-TopK support at atom
/// granularity, whose alternating polar/projection solve is the block fast
/// kernel of the same engine.
///
/// This is NOT a silent substitution (the invariant [`admit_topk_manifold`]
/// protects): substitution is handing a caller a DIFFERENT model than the one
/// requested. Here the caller named the linear model; the lane is its
/// specialized solver.
pub fn admit_linear_dictionary(
    n_obs: usize,
    output_dim: usize,
    n_atoms: usize,
    block_size: usize,
) -> Result<SaeFitAdmission, String> {
    if n_obs == 0 || output_dim == 0 || n_atoms == 0 {
        return Err(format!(
            "admit_linear_dictionary requires positive N, P, and K; got N={n_obs}, \
             P={output_dim}, K={n_atoms}"
        ));
    }
    if block_size == 0 || block_size > output_dim {
        return Err(format!(
            "admit_linear_dictionary requires 1 <= block_size <= P={output_dim} (a block's b \
             orthonormal directions must fit in R^P); got {block_size}"
        ));
    }
    Ok(SaeFitAdmission {
        lane: SaeFitLane::SparseCodes,
        n_obs,
        output_dim,
        n_atoms,
        // The linear schedule never materializes a dense assignment; the audit
        // cells record the shape the DEFAULT rule would have compared.
        dense_assignment_cells: n_obs.saturating_mul(n_atoms),
        response_cells: n_obs.saturating_mul(output_dim),
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
             gives N*K={} > N*P={} — the overcomplete (K > P) routes are the hard TOP-K SUPPORT \
             curved lane (assignment='topk', admitted by concrete memory budget; penalty-gated \
             modes cannot take it because their N×K gate logits are live Newton state) or the \
             linear sparse-code lane; neither builds the dense N×K assignment state",
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

        // One byte under the resident seed: at this tiny (~2.1 GB) budget the
        // streamed curved peak (~9.5 GB, dominated by the framed border
        // workspace at K=32000) does NOT fit either, so the shape is over BOTH
        // budgets and refuses — never silently demoted to the linear sparse-code
        // lane. (The streaming region that IS runnable is exercised at a larger
        // budget in `topk_manifold_streaming_region_admits_to_curved_lane`.)
        let err = admit_topk_manifold_with_budget(4096, 512, 32_000, 1, 8, bytes - 1)
            .expect_err("over-both-budgets topk must refuse");
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

    /// The chunked-seed region (dense seed over budget, streamed curved shape
    /// within the streaming budget) is now RUNNABLE — the driver is wired
    /// (`SaeManifoldTerm::seed_cold_start_disjoint_charts_streaming` +
    /// `fit_topk_curved_streaming`), so the front door admits it to the CURVED
    /// lane instead of refusing. Only a shape past BOTH budgets refuses, still
    /// without substituting the linear sparse-code lane.
    #[test]
    fn topk_manifold_streaming_region_admits_to_curved_lane() {
        // N=1e6, K=32000, d=1: resident seed = 512 GB (over the 16 GiB in-core
        // budget), streamed peak ≈ 9.6 GiB (dominated by the framed border
        // workspace) — the region the chunked-seed driver exists to run.
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
        let admission =
            admit_topk_manifold_with_budget(1_000_000, 512, 32_000, 1, 8, 16 * 1024 * 1024 * 1024)
                .expect("streaming region is runnable and must admit to the curved lane");
        assert_eq!(
            admission.lane,
            SaeFitLane::CurvedStreaming,
            "the over-resident/under-streaming region admits to CurvedStreaming"
        );
        assert_eq!(
            (admission.n_obs, admission.output_dim, admission.n_atoms),
            (1_000_000, 512, 32_000)
        );

        // Over BOTH budgets (budget 0 → streaming floored at 64 MiB, still far
        // below the K=32000 framed workspace): the plain refusal, no substitution.
        let err = admit_topk_manifold_with_budget(1_000_000, 512, 32_000, 1, 8, 0)
            .expect_err("over-both-budgets topk must refuse");
        assert!(err.contains("never silently substituted"));
        assert!(!err.contains("admit_topk_curved_lane"));
    }

    /// #2232 Inc 5b (Gap B) — an EXPLICIT linear-dictionary request takes the
    /// sparse-code lane at ANY K: the K>P-only sparse gate is the DEFAULT rule
    /// for manifold requests, relaxed into a modeling choice for callers who
    /// name the linear model.
    #[test]
    fn explicit_linear_dictionary_admits_sparse_codes_at_any_k() {
        // K ≤ P: the shape the default rule would send to the dense engine.
        let small = admit_linear_dictionary(640, 24, 16, 1).expect("K<=P linear admission");
        assert_eq!(small.lane, SaeFitLane::SparseCodes);
        assert_eq!((small.n_obs, small.output_dim, small.n_atoms), (640, 24, 16));
        // K > P: identical lane — the request, not the shape, selects it.
        let large = admit_linear_dictionary(4096, 512, 32_000, 1).expect("K>P linear admission");
        assert_eq!(large.lane, SaeFitLane::SparseCodes);
        // Block atoms (uniform Euclidean d=b): same lane, b bounded by P.
        let block = admit_linear_dictionary(4096, 512, 1024, 4).expect("block linear admission");
        assert_eq!(block.lane, SaeFitLane::SparseCodes);
        assert!(admit_linear_dictionary(4096, 512, 1024, 513).is_err());
        assert!(admit_linear_dictionary(4096, 512, 1024, 0).is_err());
        assert!(admit_linear_dictionary(0, 512, 1024, 1).is_err());
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

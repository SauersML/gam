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
    /// Exact support-sparse memory contract for an overcomplete hard-TopK
    /// request. `None` for the dense-certification and sparse-linear layouts.
    /// Keeping the ledger in the admission value prevents the public entry from
    /// re-deciding (and potentially discarding) the lane after validation.
    pub topk_budget: Option<crate::manifold::SaeTopKCurvedBudget>,
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
        topk_budget: None,
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
///   * `K > P` — exactly one representation: `O(N·k_active)` canonical active
///     state, an `O(P+k_active·(2+d_max))` row-local routing workspace, and the
///     final-function decoder / matrix-free border workspace. Atom scores are
///     consumed one at a time into a bounded TopK heap; no resident or chunked
///     dense `(N,K)` / `(*,K)` score array is a representation choice;
///   * over the support-sparse budget — a typed Err. For a caller who asked for TOPK
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
    // K > P: the overcomplete support-sparse lane. The streaming plan owns the ledger.
    let ledger = crate::manifold::sae_topk_curved_budget_from_budget(
        n_obs,
        output_dim,
        n_atoms,
        d_max,
        support_k,
        budget_bytes,
    );
    // K>P TopK has ONE representation: support-sparse. A resident dense seed is
    // deliberately not an alternative even when it would happen to fit on this
    // machine; accepting it would make model representation depend on available
    // RAM and would reintroduce the N×K state this admission exists to forbid.
    if ledger.streaming_admitted {
        return Ok(SaeFitAdmission {
            lane: SaeFitLane::CurvedStreaming,
            topk_budget: Some(ledger),
            ..admission
        });
    }
    Err(format!(
        "topk manifold engine refused: the canonical support-sparse peak {} bytes \
         (active state {} + row-local routing workspace {} + decoder {} + border workspace {}) \
         exceeds the support budget {} bytes at N={n_obs}, P={output_dim}, K={n_atoms}, \
         k_active={support_k}, d_max={d_max}. Reduce n_obs or support_k — a TOPK MANIFOLD \
         request is never silently substituted with the linear sparse-code lane",
        ledger.streaming_peak_bytes,
        ledger.active_state_bytes,
        ledger.routing_workspace_bytes,
        ledger.decoder_bytes,
        ledger.border_vector_bytes,
        ledger.streaming_budget_bytes,
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
        topk_budget: None,
    })
}

/// #2231 Inc C — BORDER-GROWTH admission for the crosscoder's stacked width.
///
/// Stacking `L` layers widens the response to `p̃ = p_x + Σ_ℓ p_ℓ`, and the
/// row-count admissions above are already correct at `output_dim = p̃` (the
/// response really is `N×p̃`). What they do NOT see is the arrow-Schur BORDER:
/// the dense full-`B` border is `beta_dim = Σ_k M_k·p̃` — the one quantity
/// QUADRATIC in the layer count through the `beta_dim²` Hessian workspace
/// (`take_border_hbb_workspace`) — while the framed border
/// (`factored_border_dim = Σ_k M_k·r_k`, decoders profiled as
/// `B_k = C_k·U_kᵀ`, `U_k ∈ Gr(r_k, p̃)`) is `p̃`-INDEPENDENT. This check
/// admits the border the fit will actually carry (`border_dim`, which equals
/// `full_beta_dim` on the all-full-`B` path) against the host in-core budget,
/// and the refusal names the frame default as the remedy: the crosscoder lane
/// pays the layer widening on the cheap frame side, never by silently
/// narrowing the stacked target.
pub fn admit_crosscoder_border(
    border_dim: usize,
    full_beta_dim: usize,
    budget_bytes: usize,
) -> Result<(), String> {
    let border_bytes = border_dim
        .saturating_mul(border_dim)
        .saturating_mul(std::mem::size_of::<f64>());
    if border_bytes <= budget_bytes {
        return Ok(());
    }
    let full_bytes = full_beta_dim
        .saturating_mul(full_beta_dim)
        .saturating_mul(std::mem::size_of::<f64>());
    Err(format!(
        "crosscoder border refused: the arrow-Schur border workspace at the stacked width is \
         {border_dim}² · 8 = {border_bytes} bytes, over the host in-core budget \
         ({budget_bytes} bytes); the dense full-B border at this shape is (Σ M_k·p̃)² · 8 = \
         {full_bytes} bytes. Default the atoms onto profiled Grassmann frames \
         (maybe_activate_decoder_frame: the factored border Σ M_k·r_k is p̃-independent) or \
         reduce the stacked width — a crosscoder request is never silently narrowed to fewer \
         layers"
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
        // K >> P admits only through the canonical support-sparse ledger.
        let bytes = usize::MAX / 2;
        let ok = admit_topk_manifold_with_budget(4096, 512, 32_000, 1, 8, bytes)
            .expect("within-budget topk admission");
        assert_eq!(
            ok.lane,
            SaeFitLane::CurvedStreaming,
            "TopK within budget admits the CURVED manifold lane at K > P"
        );
        // The admission carries the audit shape untouched.
        assert_eq!((ok.n_obs, ok.output_dim, ok.n_atoms), (4096, 512, 32_000));

        // The admitted region matches the support plan's documented ledger.
        let ledger =
            crate::manifold::sae_topk_curved_budget_from_budget(4096, 512, 32_000, 1, 8, bytes);
        assert!(ledger.streaming_admitted);

        // A starved support budget refuses; there is no dense retry or demotion.
        let err = admit_topk_manifold_with_budget(4096, 512, 32_000, 1, 8, 0)
            .expect_err("over-budget topk must refuse");
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

    /// The public overcomplete route is admitted solely by its support-shaped
    /// peak, independent of whether a dense seed would fit.
    #[test]
    fn topk_manifold_support_budget_admits_to_curved_lane() {
        let ledger = crate::manifold::sae_topk_curved_budget_from_budget(
            4096,
            64,
            10_000,
            1,
            8,
            8 * 1024 * 1024 * 1024,
        );
        assert!(ledger.streaming_admitted);
        let admission = admit_topk_manifold_with_budget(
            4096,
            64,
            10_000,
            1,
            8,
            8 * 1024 * 1024 * 1024,
        )
        .expect("support-sparse shape must admit to the curved lane");
        assert_eq!(
            admission.lane,
            SaeFitLane::CurvedStreaming,
            "the over-resident/under-streaming region admits to CurvedStreaming"
        );
        assert_eq!(
            (admission.n_obs, admission.output_dim, admission.n_atoms),
            (4096, 64, 10_000)
        );

        let err = admit_topk_manifold_with_budget(4096, 64, 10_000, 1, 8, 0)
            .expect_err("over-budget topk must refuse");
        assert!(err.contains("never silently substituted"));
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
        assert_eq!(
            (small.n_obs, small.output_dim, small.n_atoms),
            (640, 24, 16)
        );
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

    /// #2231 Inc C — the crosscoder border admission: the fit's actual border
    /// (framed or full-B) must pay `border² · 8` bytes against the budget, and
    /// the refusal names the frame default (the `p̃`-independent factored
    /// border) rather than silently narrowing the stacked target.
    #[test]
    fn crosscoder_border_admission_checks_actual_border_and_names_frames() {
        // Small border under a small budget: admitted.
        admit_crosscoder_border(100, 100, 100 * 100 * 8).expect("within-budget border");
        // Framed border admitted where the full-B border would refuse: the
        // exact region the frame default exists for.
        admit_crosscoder_border(512, 4096, 512 * 512 * 8).expect("framed border within budget");
        let err = admit_crosscoder_border(4096, 4096, 512 * 512 * 8)
            .expect_err("full-B border over budget must refuse");
        assert!(
            err.contains("never silently narrowed"),
            "refusal must state the no-narrowing contract; got: {err}"
        );
        assert!(
            err.contains("maybe_activate_decoder_frame"),
            "refusal must name the frame remedy; got: {err}"
        );
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

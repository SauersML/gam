// Pre-fit cross-block identifiability audit.
//
// Stage 1 design notes only. The implementation lands in later stages.
//
// # Where the existing dedupe lives
//
// `crate::families::bernoulli_marginal_slope::enforce_cross_block_identifiability_for_flex_block`
// (`src/families/bernoulli_marginal_slope.rs:1641-1951`) is the only
// cross-block identifiability path in the tree today. It runs at BMS
// construction sites (`src/families/bernoulli_marginal_slope.rs:17045`,
// `:17183`) and reparameterises a single candidate "flex" deviation block
// against the union of two parametric anchors (marginal, logslope) plus
// (optionally) an earlier flex block whose `FlexEvaluation` is currently
// skipped from the anchor stack on the grounds that the per-block
// smoothness-null-space drop in `deviation_runtime` already removes the
// flex block's unpenalised null directions.
//
// # What it does
//
// Given a candidate basis `C ∈ ℝ^{n×p_c}` evaluated at the n training
// rows, a horizontally stacked anchor `A ∈ ℝ^{n×d}` and the IRLS Hessian
// row metric `W = diag(w)`:
//
//   1. eigh `G_N = AᵀWA = N_sqwᵀ N_sqw` and form
//      `R = U₊ diag(λ₊^{-1/2})`, giving the W-orthonormal anchor frame
//      `Q_w = AR` (rank `r`, dropped at `lambda_max · 64 · n · ε`);
//   2. compute `K_w = Q_wᵀ W C` and the W-residualised candidate
//      `C̃ = C − Q_w K_w` in the sqrt-W frame;
//   3. eigh `G_C̃ = C̃ᵀ W C̃` with the same LAPACK-style threshold, anchored
//      to `max(λ_max(C̃ᵀWC̃), ‖c_sqw‖²_F)` so that fully-aliased candidates
//      (where `λ_max_c` itself collapses to noise) still get a stable
//      reference;
//   4. retain the eigenvectors `V` of positive eigenvalues; install the
//      residual `M = R K_w V` into the candidate's `DeviationRuntime` so
//      every predict-time row evaluates `pure_span_row · V − n_row · M`.
//
// Outcome enum: `Reparameterised` (some directions survived) or
// `FullyAliased { reason }` (none did — caller must drop the block with a
// structured warning rather than continue with a zero-rank block).
//
// # What it MISSES (the gap this module closes)
//
// 1. **Scope is BMS-only.** Only the bernoulli marginal-slope family
//    invokes it. The standard-model, gaussian / binomial location-scale,
//    survival (`survival_marginal_slope`, `survival_location_scale`,
//    royston-parmar), and custom-family workflows have no equivalent
//    cross-block audit at all. Aliasing among smooths from the formula
//    DSL ([[project_gam_geometric_smooths]]) is invisible to it.
//
// 2. **Tied to `DeviationRuntime`.** The reparameterisation is applied
//    via `compose_anchor_orthogonalisation` + `AnchorResidual` on a
//    `DeviationPrepared`; it cannot be pointed at a generic
//    `ParameterBlockSpec` or at an already-built joint design.
//
// 3. **No structured report.** Drops are surfaced through an ad-hoc
//    `[BMS cross-block identifiability]` log line plus a
//    `CrossBlockIdentifiabilityWarning`; there is no list of which
//    column from which block was dropped because of which anchor
//    column, and no per-block effective-dim accounting that a caller
//    or diagnostician could consume programmatically.
//
// 4. **`FlexEvaluation` anchors are intentionally skipped from the N
//    stack.** This is correct only if the flex anchor's nullspace was
//    truly absorbed by `smoothness_nullspace_orthogonal_complement`. If
//    `nullspace-lead`'s rewrite changes that contract (or if a future
//    smooth family forgets to absorb its nullspace), flex-flex aliasing
//    will silently slip through.
//
// 5. **No "benign vs ambiguous" distinction.** Every alias is either
//    repaired by reparameterisation or refused. There is no notion of
//    "drop the candidate column whose image lives entirely in an
//    earlier block's parametric intercept span, log INFO, and proceed"
//    vs. "the alias is large-magnitude but spans more than one earlier
//    block — refuse with a diagnostic listing the offending pair".
//
// 6. **Runs at BMS construction time only.** A custom family that
//    builds its own `ParameterBlockSpec` list (e.g. a hand-assembled
//    parametric + s(age) + ti(age, sex) model) gets no pre-fit audit
//    at all; the rank-deficiency surfaces inside PIRLS as a
//    near-null direction in the joint penalised Hessian.
//
// # Plan for the unified audit
//
// Build a family-agnostic `audit_identifiability(specs, data) ->
// IdentifiabilityAudit` that:
//   - takes the final list of `ParameterBlockSpec` (so it runs *after*
//     `nullspace-lead`'s within-block nullspace absorption — see
//     coordination note at the top of the team brief);
//   - constructs the joint design `X_joint = [X_block_0 | … ]`,
//     materialising blocks via the sparse/lazy `DesignMatrix`
//     transpose-matvec path so no block needs to densify globally;
//   - runs a column-pivoted QR (or SVD with pivoting) with tolerance
//     `tol = sqrt(eps) · σ_max(X_joint)` to identify columns linearly
//     dependent on earlier columns;
//   - for each dropped column, locates the earliest anchor block
//     carrying its image (project the dropped column onto each earlier
//     block's range, pick the block with the largest projection norm);
//   - emits a `BlockIdentity` per block (original_dim, effective_dim,
//     range-rank, singular values), `AliasedPair` records for each
//     overlap above tolerance, and `DroppedColumn` records with a
//     human-readable reason;
//   - sets `fatal = true` when (a) a dropped column's projection is
//     ambiguous (overlap split across multiple earlier blocks above
//     tolerance), or (b) the magnitude / structural meaning of the
//     drop would silently change model semantics (e.g. dropping a
//     smooth's only linear direction when no parametric linear term
//     exists).
//
// Stage 3 wires this into the entry points listed in
// `src/solver/workflow.rs`. Stage 4 collapses
// `enforce_cross_block_identifiability_for_flex_block` into a thin
// wrapper that builds the spec list (with the BMS-specific W-metric
// PIRLS row weights), delegates to `audit_identifiability`, then
// installs the resulting reparameterisation back into
// `DeviationRuntime` via the existing `AnchorResidual` plumbing.
//
// Coordination:
//   - `nullspace-lead`: final `ParameterBlockSpec.design` layout
//     after their within-smooth nullspace absorption (DM open).
//   - `diagnostician`: name an `AliasingDetectedAtFit` variant in
//     `CertRefusalDiagnosis` for fatal audit failures bubbled into the
//     KKT-refusal pipeline.
//   - `seed-accounting`: name an `IdentifiabilityFailure` variant in
//     `InnerFailure` for the same.

# Identifiability Compiler — Phase 2 Architecture

> **Status: implemented.** Source of truth is now `src/families/identifiability_compiler.rs` + the family-specific operator files. This document is the design-time record of the refactor that replaced the term-level residualizer (formerly `enforce_cross_block_identifiability_for_flex_block` + the `CrossBlockAnchor` enum) with the row-Jacobian compiler.

Family-agnostic row-Jacobian compiler that orthogonalises blocks in the *row primary-state* metric. Survival uses `u_i = (q0, q1, qd1, g) ∈ R^4`, Bernoulli uses `u_i = η ∈ R^1`. FlexEvaluation anchors become first-class — they expose a residualised row operator that subsequent blocks (link-dev) consume identically to a parametric anchor.

Module path: `src/families/identifiability_compiler.rs`. Family-specific trait impls live in `src/families/survival_marginal_slope/identifiability.rs` (new file) and `src/families/bernoulli_marginal_slope/identifiability.rs` (new file). Re-exported through each family's `mod.rs`.

---

## 1. `RowJacobianOperator` trait

```rust
/// Maps a coefficient perturbation `δβ ∈ R^p` for one parameter block into
/// its contribution to the per-row primary state `u_i ∈ R^K`.
///
/// Concretely, if `u_i(β) = f_i(β_block, …)`, this trait returns the linear
/// map `J_i = ∂u_i/∂β_block` evaluated at the pilot β.  For affine blocks
/// (everything in this compiler), `J_i` is independent of β and equals the
/// transposed row of the block's effective design matrix lifted into `R^K`.
pub trait RowJacobianOperator: Send + Sync {
    /// Dimension of the row primary state. Survival: 4. Bernoulli: 1.
    const K: usize;

    /// Number of coefficients in this block (= width of J_i).
    fn ncols(&self) -> usize;

    /// Number of training rows.
    fn nrows(&self) -> usize;

    /// Apply the row Jacobian: returns `J_i · δβ ∈ R^K` for the given row.
    /// Must allocate no heap memory on the hot path.
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]);

    /// Materialise the full operator as an `(n_rows × ncols × K)` tensor.
    /// Used for the dense Gram solve in `compile`. For survival, returns
    /// an `Array3<f64>` shaped `(n_rows, ncols, 4)`. For Bernoulli, the
    /// trailing axis is length 1 and the caller flattens.
    fn evaluate_full(&self) -> ndarray::Array3<f64>;

    /// Apply Jᵀ along the row axis weighted by `H · v` factors. Used by
    /// the sequential Gram solve to compute `<a,b> = Σ_i J_a,iᵀ H_i J_b,i`
    /// without materialising the full tensor for the prior block. See
    /// `IdentifiabilityCompiler::compile`.
    fn cross_metric_against(
        &self,
        rhs: &dyn RowJacobianOperator,
        row_hess: &dyn RowHessian,
    ) -> ndarray::Array2<f64>; // shape (self.ncols(), rhs.ncols())
}
```

### Survival concrete impls (in `survival_marginal_slope/identifiability.rs`)

Each survival block produces a `[K=4]` row contribution `(δq0, δq1, δqd1, δg)`:

- **`TimeBlockOperator`** (block 0, `beta_time`):
  - `δq0 = design_entry.row(i) · δβ + timewiggle_d/dβ` contributions
  - `δq1 = design_exit.row(i) · δβ + timewiggle_d/dβ`
  - `δqd1 = design_derivative_exit.row(i) · δβ + timewiggle_d²` (cf. `row_dynamic_q_gradient` at `src/families/survival_marginal_slope.rs:4741-4872`)
  - `δg = 0`
  - Backed by `dq0_time, dq1_time, dqd1_time` arrays from the existing geometry function.

- **`MarginalBlockOperator`** (block 1, `beta_marginal`):
  - `δq0 = δq1 = marginal_design.row(i) · δβ` (+ timewiggle scale factor when active, identical to `dq0_marginal`/`dq1_marginal`)
  - `δqd1 = dqd1_marginal[j] · δβ` (zero when timewiggle inactive)
  - `δg = 0`

- **`LogslopeBlockOperator`** (block 2, `beta_logslope`):
  - `δq0 = δq1 = δqd1 = 0`
  - `δg = logslope_design.row(i) · δβ`

- **`ScoreWarpBlockOperator`** (block 3, `beta_score_warp`): row Jacobian is the score-warp deviation basis row at `q0_seed[i]` (rigid pilot) acting on q-channel only:
  - `δq0 = δq1 = warp_basis.row(i) · δβ` (warp adds to η at both entry and exit times since it is a shift on q)
  - `δqd1 = warp_basis_derivative.row(i) · δβ` (chain rule via dq0_seed/dt at the row)
  - `δg = 0`

- **`LinkDevBlockOperator`** (block 4, `beta_link_dev`): the link-dev basis is a function of the *full* rigid q0_seed including marginal + logslope; row Jacobian:
  - `δq0 = δq1 = link_basis.row(i) · δβ`
  - `δqd1 = link_basis_derivative.row(i) · δβ`
  - `δg = 0`

### Bernoulli concrete impls (in `bernoulli_marginal_slope/identifiability.rs`)

`K=1`, single component is δη:

- **`LogslopeBlockOperator`**: `δη = logslope_design.row(i) · δβ` (also covers marginal columns in BMS since BMS's primary state is η alone; marginal columns enter η linearly via logslope_design's stacking — see call site `bernoulli_marginal_slope.rs:19224`).
- **`MarginalBlockOperator`**: `δη = marginal_design.row(i) · δβ`.
- **`ScoreWarpBlockOperator`**: `δη = warp_basis.row(i) · δβ`.
- **`LinkDevBlockOperator`**: `δη = link_basis.row(i) · δβ`.

---

## 2. `RowHessian` trait

```rust
/// Per-row K×K PSD Hessian of `−log L_i(u_i)` evaluated at a pilot β.
/// The compiler uses this as the row metric H_i in
/// `<J_a, J_b> = Σ_i J_a,iᵀ H_i J_b,i`.
pub trait RowHessian: Send + Sync {
    const K: usize;
    fn nrows(&self) -> usize;
    /// Fill the K×K block at `row` into `out` (row-major).
    fn fill_row(&self, row: usize, out: &mut [f64]);
    /// Materialise full `(n_rows × K × K)` tensor (test / debug path).
    fn evaluate_full(&self) -> ndarray::Array3<f64>;
}
```

### Survival 4×4 row Hessian (`SurvivalRowHessian`)

Derived from the per-row likelihood kernel `survival_marginal_slope_vector_eta_grad_hess` at `survival_marginal_slope.rs:3099-3265`. That kernel returns the gradient/Hessian in `(eta0, eta1, ad1, slopes…)`. We project it onto `(q0, q1, qd1, g)` via the chain rule:

```
eta0 = q0·c(g) + linear(g, z)
eta1 = q1·c(g) + linear(g, z)
ad1  = qd1·c(g)
```

So `∂eta0/∂q0 = c`, `∂eta0/∂g = q0·c'(g) + ∂linear/∂g`, etc. At pilot β the row Hessian H_i ∈ R^{4×4} is

```
H_i[q0,q0]   = c² · u2_eta0
H_i[q1,q1]   = c² · u2_eta1
H_i[qd1,qd1] = c² · u2_ad1
H_i[q0,g]    = c · (q0·c' + ∂linear/∂g) · u2_eta0 + c' · u1_eta0
H_i[q1,g]    = c · (q1·c' + ∂linear/∂g) · u2_eta1 + c' · u1_eta1
H_i[qd1,g]   = c · (qd1·c') · u2_ad1 + c' · u1_ad1
H_i[g,g]     = u2_eta0·(q0·c'+∂lin)² + u2_eta1·(q1·c'+∂lin)²
             + u2_ad1·(qd1·c')² + u1_eta0·q0·c'' + u1_eta1·q1·c'' + u1_ad1·qd1·c''
```

with `c, c', c''` from `c_derivatives(g, probit_scale)` at `survival_marginal_slope.rs:3267`; `u1_*`, `u2_*` from `signed_probit_neglog_derivatives_up_to_fourth` and `neglog_derivatives(ad1)` at lines 3186-3192; the marginal-slope slope index `k` is collapsed since the compiler operates per training row at fixed z (the off-diagonal slope rows in the existing 3+k Hessian are absorbed into the g channel via the linear term derivative). PSD-clamp: eigendecompose H_i, project negative eigenvalues to zero before use (numerical guard against pilot β being far from optimum).

### Bernoulli 1×1 row Hessian

`H_i = W_i = phi(η_i)² / (Φ(η_i)·Φ(−η_i))` — the standard probit IRLS weight at the pilot η. Already implemented in `bernoulli_marginal_slope.rs` IRLS weight builder; expose as `BernoulliRowHessian { w: Array1<f64> }`.

---

## 3. `IdentifiabilityCompiler::compile`

```rust
pub struct CompiledBlock {
    /// Orthogonal-complement reparam matrix V ∈ R^{p × p'} (right-selector).
    pub t_lw: Array2<f64>,
    /// Anchor residual coefficient matrix M ∈ R^{d × p'} so prediction
    /// row contribution is (C(x)·V − A(x)·M)·β. None for the first block.
    pub anchor_correction: Option<Array2<f64>>,
    /// Reference to the anchor row evaluator that emits A(x) at predict time.
    pub anchor_evaluator: Option<Arc<dyn AnchorRowEvaluator>>,
}

pub struct CompiledBlocks {
    pub blocks: Vec<CompiledBlock>,
    /// Pre-fit audit verdict: joint rank under the row metric and
    /// any columns the compiler deterministically dropped.
    pub joint_rank: usize,
    pub dropped: Vec<(usize /* block_idx */, usize /* local_col */)>,
}

pub fn compile(
    operators:    &[Arc<dyn RowJacobianOperator>], // length = n_blocks
    row_hess:     &dyn RowHessian,
    ordering:     &[BlockOrder],   // semantic order (owner first → latest last)
) -> Result<CompiledBlocks, String>;
```

### Algorithm

Walk `ordering` left-to-right. Maintain cumulative anchor `A` = concat of all already-compiled blocks (as operators). For each new block `B`:

1. Form the cross-metric Gram blocks
   ```
   G_AA[a,b] = Σ_i  J_A_i[a]ᵀ · H_i · J_A_i[b]
   G_AB[a,b] = Σ_i  J_A_i[a]ᵀ · H_i · J_B_i[b]
   G_BB[a,b] = Σ_i  J_B_i[a]ᵀ · H_i · J_B_i[b]
   ```
   where each `J_*_i` is the row Jacobian K-vector for that column. This is a single `K`-fold reduction per (row, a, b) triple, vectorised over rows. For survival K=4 we exploit the small K with hand-unrolled `H_i` apply (10 unique entries).

2. Solve `G_AA · M = G_AB` (M ∈ R^{d × p_B}). Pivoted Cholesky preferred; QR fallback when min pivot ≤ `λ_max(G_AA) · 64·n·ε`. M is the anchor correction matrix.

3. Residual Gram: `G̃ = G_BB − G_ABᵀ · M`.

4. Eigendecompose `G̃` (`faer::eigh`). Threshold:
   ```
   τ = max(λ_max(G̃), tr(G_BB)) · 64 · n · ε
   ```
   Keep eigenvectors with `λ > τ`; their stacking is `V = T_lw` ∈ R^{p_B × p_B'}.

5. Append to anchor: `A := [A | B·V]` for subsequent blocks. The "B·V" operator is supplied by the *block itself* (see §4 — flex blocks expose a residualised row operator).

6. Persist `CompiledBlock { t_lw: V, anchor_correction: Some(M), anchor_evaluator: A_evaluator }`.

### Pre-fit audit (after the walk)

After compiling all blocks, build the joint *primary-state* design `J_joint` as the n × (Σ p'_b) × K tensor, then column-pivoted QR on its `sqrt(H_i)`-scaled flattening. Threshold `rank_tol = σ_max · max(n·K, p_total) · ε`. If `rank < p_total`, the trailing pivots correspond to columns of the *latest* block in `ordering`; drop them by truncating that block's `V` (semantic ownership is preserved — earlier blocks are never modified).

---

## 4. FlexEvaluation anchors as first-class operators

The trait `AnchorRowEvaluator` is the predict-time bridge:

```rust
pub trait AnchorRowEvaluator: Send + Sync {
    /// Build the anchor row matrix A(x) at the supplied predict-time
    /// argument rows. For parametric blocks, this is just the block's
    /// design at x. For *compiled flex* blocks, it is
    ///     A_flex(x) = basis_eval(x) · V_flex   − A_flex_anchors(x) · M_flex
    /// i.e. the post-residualisation row evaluator.
    fn anchor_rows(&self, predict_arg: &Array1<f64>) -> Result<Array2<f64>, String>;
    fn ncols(&self) -> usize;
}
```

`CompiledBlock` stores `anchor_evaluator: Arc<dyn AnchorRowEvaluator>`. For the link-dev block, the compiler hands it an `anchor_evaluator` for the previously-compiled score-warp block (and the marginal/logslope parametric evaluators), and uses **exactly the same Gram solve** — no special path. The "silent skip" at `bernoulli_marginal_slope.rs:1690-1699` is deleted; flex anchors enter the Gram on equal footing with parametric ones.

### Trait method the compiled flex block exposes

```rust
impl AnchorRowEvaluator for CompiledFlexBlock {
    fn anchor_rows(&self, predict_arg: &Array1<f64>) -> Result<Array2<f64>, String> {
        let raw = self.runtime.span_basis_at(predict_arg)?;       // C(x)
        let rotated = fast_ab(&raw, &self.t_lw);                  // C(x)·V
        let anchor_rows = self.parent_evaluator.anchor_rows(predict_arg)?; // A(x)
        let correction = fast_ab(&anchor_rows, &self.anchor_correction);   // A(x)·M
        Ok(&rotated - &correction)
    }
}
```

This is the residualised row operator that link-dev consumes as a flex anchor.

---

## 5. Storage on the runtime + predict-time wiring

Each compiled block lives on its `DeviationRuntime` exactly where `anchor_residual` lives today (`src/families/bernoulli_marginal_slope/deviation_runtime.rs:574-676`). The struct stored is `CompiledBlock { t_lw, anchor_correction, anchor_evaluator }` — the `T_lw` replaces the existing `right_selector` composition (`span_c{0..3}` rotated by V), and `anchor_correction` replaces the existing `AnchorResidual.residual_coefficients`.

`AnchorNullSpaceEvaluator::Stacked` is replaced by the `anchor_evaluator: Arc<dyn AnchorRowEvaluator>` which is composable: it owns `Vec<Arc<dyn AnchorRowEvaluator>>` for its parent anchors and concatenates their outputs.

### Predict-time wiring

- **BMS predict** at `src/inference/predict.rs:1138-1170` becomes:
  - Replace the hand-coded marginal + logslope stack with calls into the runtime's stored `anchor_evaluator.anchor_rows(predict_arg)?`.
  - The "fast empty path" (lines 1142-1152) stays — predicated on `compiled_block.anchor_correction.is_some()` instead of `anchor_residual_coefficients.is_some()`.
  - The runtime's row-application of `M` (`design_with_anchor_rows` in `deviation_runtime.rs`) is unchanged structurally but pulls `anchor_correction` from the new `CompiledBlock` field.

- **Survival predict** in `src/families/survival_predict.rs`: currently has *no* anchor wiring (grep returns empty). Phase 4 adds the analogous path:
  - Materialise marginal + logslope + (compiled score-warp via its `anchor_evaluator`) at predict rows.
  - For each deviation runtime that carries a `CompiledBlock`, call `runtime.anchor_evaluator.anchor_rows(predict_arg)` and apply the row correction.
  - Insertion point: alongside the deviation-basis evaluation in the survival prediction kernel (currently the deviation runtimes are evaluated unconditionally; the new code path inserts the `−A(x)·M·β` correction after `C(x)·V·β`).

---

## 6. Migration delta per call site

All 9 call sites collapse into either a single `IdentifiabilityCompiler::compile([...])` call at fit-setup time (production sites) or a direct `compile(...)` call in tests (test sites). Old code to delete:

### Survival MGS (2 production call sites)

- `src/families/survival_marginal_slope.rs:17684-17721` (score-warp anchor-set assembly + `enforce_*` call + outcome match): replaced by **one** `compile([Time, Marginal, Logslope, ScoreWarp])` call earlier in the function. Delete lines 17686-17701 (anchor vec + tag vec construction + `enforce_*` call), keep only the outcome handling around `stripe_score_warp_across_z_coords`.
- `src/families/survival_marginal_slope.rs:17722-17804` (link-dev anchor-set assembly including the FlexEvaluation push at 17778): becomes part of the same `compile(...)` call extended with `LinkDev`. Delete lines 17765-17788 (anchor vec build, FlexEvaluation push, `enforce_*` call); the `score_warp_anchor_design` materialisation goes away — the compiler asks the compiled score-warp block for its `anchor_evaluator` directly.

### Bernoulli MGS (5 production call sites + 4 test sites)

- `bernoulli_marginal_slope.rs:19219-19260` (score-warp, outer setup): replaced by `compile([Marginal, Logslope, ScoreWarp])`. Delete the anchor vec + `enforce_*` call.
- `bernoulli_marginal_slope.rs:19337-19410` (score-warp outer-loop variant with FlexEvaluation anchor push at 19345 + `enforce_*` at 19357): folds into the same `compile(...)` extended with LinkDev. Delete lines 19337-19360.
- `bernoulli_marginal_slope.rs:19991-20015, 20073-20100, 20153-20180` (three `enforce_*` test/setup sites): these are unit tests; rewrite each as a direct `compile(...)` call.  Delete the entire `enforce_cross_block_identifiability_for_flex_block(...)` invocation block in each case.
- `bernoulli_marginal_slope.rs:20222-20265` (link-dev, outer setup): replaced by inclusion of LinkDev block in the master `compile(...)` call. Delete.
- `bernoulli_marginal_slope.rs:20328-20360` (link-dev, outer-loop variant): same. Delete.

In all production sites, the per-block tag arrays (`parametric_anchor_blocks: Vec<Option<ParametricAnchorBlock>>`) disappear — block identity is intrinsic to the `RowJacobianOperator` instance.

---

## 7. Old-API removal (atomic, no shims)

Delete entirely from `src/families/bernoulli_marginal_slope.rs`:
- `fn enforce_cross_block_identifiability_for_flex_block` (lines 1641-2009, ~370 lines including the `intentionally skipped` comment block at 1678-1699)
- `enum CrossBlockAnchor { Parametric, FlexEvaluation }` (definition + 21 use sites)
- `enum CrossBlockIdentifiabilityOutcome` (FullyAliased / Reparameterised) — collapse into `Result<CompiledBlocks, CompilerError { reason, fully_aliased_block: Option<usize> }>`
- `enum AnchorNullSpaceComponent` and `enum AnchorNullSpaceEvaluator` — replaced by `Arc<dyn AnchorRowEvaluator>`
- `struct AnchorResidual` — replaced by `CompiledBlock`'s `anchor_correction` field
- `parametric_anchor_blocks: Vec<Option<ParametricAnchorBlock>>` arrays at every call site
- `enum ParametricAnchorBlock` if its only consumer is the deleted code path (verify in Phase 4)

Delete from `src/families/bernoulli_marginal_slope/deviation_runtime.rs`:
- `compose_anchor_orthogonalisation` (lines 574-676) — replaced by `install_compiled_block(compiled: CompiledBlock)` with the same body but new field names.
- `set_anchor_rows_at_training` (line 689) — no longer needed; anchor rows at predict are produced fresh by `anchor_evaluator.anchor_rows`. Training-time caching, if needed for `design_at_training_with_residual`, lives on the `CompiledBlock` directly.
- `anchor_residual()` accessor — replaced by `compiled_block()`.

Delete from `src/inference/model.rs`: `SavedAnchoredDeviationRuntime` and its serialization fields are renamed to `SavedCompiledFlexBlock` with `{t_lw, anchor_correction, parent_anchor_descriptor}` — no parallel format.

---

## 8. Non-rigid survival link-dev pilot

Currently at `survival_marginal_slope.rs:17713-17716`, the link-dev seed uses the rigid offset-only `q_exit + slope`:

```
q0_seed[i] = rigid_observed_eta(
    offset_exit[i] + marginal_offset[i],
    baseline_slope + logslope_offset[i],
    z_primary[i],
    probit_scale,
)
```

This is wrong: link-dev's basis should be evaluated at the *pilot* η, which includes the rigid-pre-Newton solve's `β_time`, `β_marginal`, `β_logslope`. Replacement, after the rigid pre-Newton fit completes (which runs `rigid_fit_with_block_newton` at lines 18031-18121):

```rust
let beta_t = &pilot_result.fitted_blocks[0].beta;
let beta_m = &pilot_result.fitted_blocks[1].beta;
let beta_g = &pilot_result.fitted_blocks[2].beta;

let q_exit_pilot = Array1::from_iter((0..n).map(|i|
    spec.time_block.design_exit.dot_row(i, beta_t)
        + spec.marginal_design.dot_row(i, beta_m)
        + spec.time_block.offset_exit[i]
        + spec.marginal_offset[i]
));
let slope_pilot = Array1::from_iter((0..n).map(|i|
    spec.logslope_design.dot_row(i, beta_g)
        + baseline_slope
        + spec.logslope_offset[i]
));
let q0_seed = Array1::from_iter((0..n).map(|i|
    rigid_observed_eta(q_exit_pilot[i], slope_pilot[i], z_primary[i], probit_scale)
));
```

This `q0_seed` then feeds `build_link_deviation_block_from_knots_design_seed_and_weights` at line 17729 *and* the score-warp `build_score_warp_deviation_block_from_seed` at 17685 (when score-warp is present, the pilot also threads in to seed it — same arithmetic, dropping `slope_pilot` since score-warp only sees q).

Bernoulli analogue: in `fit_bernoulli_marginal_slope_flex_blocks` (around `bernoulli_marginal_slope.rs:19219`), the rigid pilot produces `β_logslope_pilot`; the η_seed used for link-dev becomes

```
eta_seed[i] = logslope_design.dot_row(i, beta_logslope_pilot) + logslope_offset[i] + intercept
```

in place of the current rigid offset-only seed.

The pilot β is extracted from `pilot_result.fitted_blocks[k].beta`; this struct already exists (it is returned from `rigid_fit_with_block_newton`). No new infrastructure needed beyond plumbing the pilot β through into the deviation-basis seed call.

---

## 9. Module placement

- **`src/families/identifiability_compiler.rs`** (new, ~600 LOC):
  - `pub trait RowJacobianOperator`
  - `pub trait RowHessian`
  - `pub trait AnchorRowEvaluator`
  - `pub struct CompiledBlock`, `pub struct CompiledBlocks`, `pub enum CompilerError`
  - `pub fn compile(...)` (the Gram-solve + audit driver)
  - Internal `WeightedGramSolver` (pivoted Cholesky + QR fallback)
  - Tests for the family-agnostic core.

- **`src/families/survival_marginal_slope/identifiability.rs`** (new, ~400 LOC):
  - `pub struct SurvivalRowHessian { c, c1, c2, u1_*, u2_*, z, probit_scale }`
  - `pub struct TimeBlockOperator`, `MarginalBlockOperator`, `LogslopeBlockOperator`, `ScoreWarpBlockOperator`, `LinkDevBlockOperator`
  - Family-specific `AnchorRowEvaluator` impls (parametric + compiled flex)
  - `pub fn build_survival_compiler_inputs(spec, pilot_beta) -> Vec<Arc<dyn RowJacobianOperator>>`

- **`src/families/bernoulli_marginal_slope/identifiability.rs`** (new, ~250 LOC): analogous Bernoulli stack.

Public exports through each family's `mod.rs`. Inside the survival/Bernoulli fit drivers, the call site is:

```rust
let compiled = identifiability_compiler::compile(&operators, &row_hess, &ordering)?;
install_compiled_blocks(&mut deviation_runtimes, compiled);
```

---

## 10. Testing surface

### Phase 3 unit tests (in `identifiability_compiler.rs` `#[cfg(test)] mod tests`)

1. **`compile_two_block_orthogonalises_under_metric`**: two synthetic affine blocks, H_i = I, verify `<J_A, J_B_compiled> = 0` to machine epsilon.
2. **`compile_three_block_chain`**: three blocks with sequential aliases; verify cumulative-anchor pattern and pre-fit audit reports rank == kept cols.
3. **`compile_weighted_metric_nontrivial`**: non-identity row Hessian, verify the Gram solve recovers the analytic projection for a known 2×2 case.
4. **`compile_drops_trailing_pivots_from_latest_block`**: deliberately rank-deficient joint design, verify drops come from the latest-in-ordering block.
5. **`compile_flex_anchor_is_first_class`**: pass an `AnchorRowEvaluator` flex anchor + a parametric anchor; verify residualisation matches the corresponding all-parametric reference case to machine epsilon. **This is the regression test for the deleted `FlexEvaluation` skip bug.**
6. **`survival_row_hessian_matches_finite_diff`**: 4×4 H_i computed by `SurvivalRowHessian` matches numerical Hessian of `survival_marginal_slope_vector_eta_grad_hess` projected onto (q0,q1,qd1,g) at 5 random pilot β.
7. **`bernoulli_row_hessian_matches_irls_weight`**: scalar W_i identity.
8. **`compiler_predict_path_roundtrip`**: `CompiledBlock.anchor_evaluator.anchor_rows(predict)` at training rows equals the training-time `(C(x)·V − A(x)·M)`.

### Phase 4 integration test

`tests/identifiability_compiler_biobank_joint_rank.rs` (new): the failing model spec — survival marginal-slope with marginal + logslope + score-warp + link-dev on small synthetic biobank-style data (n=2000, hypertension-style covariates). Assertion: joint design column-pivoted QR rank equals total compiled column count (51 today, never 38). This is the end-to-end gate for the rank-loss bug.

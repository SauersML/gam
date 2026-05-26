# Identifiability Call Graph and Survival Row Primary State Operators

## A. Call Sites of `enforce_cross_block_identifiability_for_flex_block`

### Survival Marginal-Slope (SMGS)
- `src/families/survival_marginal_slope.rs:17684` — score-warp block, anchors=[marginal, logslope]
- `src/families/survival_marginal_slope.rs:17771` — link-deviation block, anchors=[marginal, logslope, score_warp (flex)]

### Bernoulli Marginal-Slope (BMS)
- `src/families/bernoulli_marginal_slope.rs:19219` — score-warp block, anchors=[logslope]
- `src/families/bernoulli_marginal_slope.rs:19357` — score-warp block (outer loop), anchors=[logslope]
- `src/families/bernoulli_marginal_slope.rs:19991` — score-warp block, called in control flow
- `src/families/bernoulli_marginal_slope.rs:20073` — score-warp block, called in control flow
- `src/families/bernoulli_marginal_slope.rs:20153` — score-warp block, called in control flow
- `src/families/bernoulli_marginal_slope.rs:20222` — link-deviation block, anchors=[logslope]
- `src/families/bernoulli_marginal_slope.rs:20328` — link-deviation block, called in control flow

---

## B. Consumers of Key Residualizer Symbols

### `anchor_residual` (77 occurrences)
- Primary use: `src/families/bernoulli_marginal_slope/deviation_runtime.rs` — stored in `AnchoredDeviationRuntime`
- Consumed at predict time: `src/inference/predict.rs:1138-1170` (BernoulliMarginalSlopePredictor anchor correction matrices)
- Exported/saved: `src/inference/model.rs` — `SavedAnchoredDeviationRuntime`

### `anchor_rows_at_training` (11 occurrences)
- Definition & use: `src/families/bernoulli_marginal_slope.rs:1778` — returns `N_train` (parametric anchor stack)
- Consumed: `src/families/bernoulli_marginal_slope/deviation_runtime.rs` — row-wise projection
- Predict-time: `src/inference/predict.rs:1138` — materializes marginal+logslope at predict rows

### `design_at_training_with_residual` (18 occurrences)
- Definition: `src/families/bernoulli_marginal_slope/deviation_runtime.rs:510` 
- Purpose: returns `(candidate_design - N_train @ M)`, the post-residualisation design
- Consumed by: identifiability framework, coefficient assembly

### `anchor_correction_matrix` (8 occurrences)
- Definition: `src/families/bernoulli_marginal_slope.rs:1795` — returns `M`, the projection matrix
- Use: `src/families/bernoulli_marginal_slope/deviation_runtime.rs` — stored in `AnchoredDeviationRuntime.anchor_residual`
- Purpose: computes `N_train @ M` for every row; `M = (N_train^T W N_train)^{-1} N_train^T W candidate`

### `CrossBlockAnchor` enum (21 occurrences)
- Variants: `Parametric(DesignMatrix)` | `FlexEvaluation(Array2<f64>)`
- Flow: assembled in fit-time setup → passed to `enforce_cross_block_identifiability_for_flex_block`
- Flex variant: **INTENTIONALLY SKIPPED** at lines 1690-1699 of bernoulli_marginal_slope.rs (see critical bug note below)

---

## C. Survival Marginal-Slope Row Primary-State Operator Table

### Block Ordering in `block_states: &[ParameterBlockState]`
1. **Block 0: Time block** — `beta_time`, `eta` (unused, length 0 typical)
2. **Block 1: Marginal block** — `beta_marginal`, `eta = marginal_surface @ beta_marginal`
3. **Block 2: Logslope block** — `beta_logslope`, `eta = logslope_surface @ beta_logslope`
4. **Block 3 (optional): Score-warp deviation block** — `beta_score_warp`, `eta` (deviation basis evaluation)
5. **Block 4 (optional): Link-deviation deviation block** — `beta_link_dev`, `eta` (deviation basis evaluation)

### Row Values (u_i components): `SurvivalMarginalSlopeDynamicRowValues`
- `q0`: entry-time linear predictor
  - FIT: `design_entry @ beta_time + offset_entry + block_states[1].eta[row]` (marginal)
  - PREDICT: same construction at predict rows
  
- `q1`: exit-time linear predictor  
  - FIT: `design_exit @ beta_time + offset_exit + block_states[1].eta[row]` (marginal)
  - PREDICT: same construction at predict rows
  
- `qd1`: log-derivative of exit time
  - FIT: `design_derivative_exit @ beta_time + derivative_offset_exit`
  - PREDICT: same construction at predict rows

**Time-wiggle modification** (when `flex_timewiggle_active()`):
- `q0`, `q1`, `qd1` computed as `base_value + time_wiggle_basis @ beta_time[time_tail_range]`
- See `src/families/survival_marginal_slope.rs:4741-4740` for full computation

### Row Gradients: `SurvivalMarginalSlopeDynamicRowGradient`
- Partials w.r.t. `beta_time`: `dq0_time`, `dq1_time`, `dqd1_time`
  - Sourced from `design_entry`, `design_exit`, `design_derivative_exit` rows
  - Include time-wiggle basis contributions if active
  
- Partials w.r.t. `beta_marginal`: `dq0_marginal`, `dq1_marginal`, `dqd1_marginal`
  - Sourced from `marginal_surface` @ each training row
  - Applied to both q0 and q1 (marginal offset affects both times)
  - dqd1_marginal = 0 (no marginal in derivative term)

### Additional Components Assembled at Runtime
- `g = block_states[2].eta[row]` — logslope linear predictor, `logslope_surface @ beta_logslope`
  - Used in probit hazard kernel; does NOT appear in q0, q1, qd1
  
- **Score-warp contribution** (optional block 3):
  - If present, `score_warp_runtime.design(q0_seed)[row] @ beta_score_warp` added to q-values after residualisation
  - At FIT: q0_seed computed from rigid offset-only solution
  - At PREDICT: q0_seed recomputed from (offset + marginal + logslope) at predict rows

- **Link-deviation contribution** (optional block 4):
  - If present, `link_deviation_runtime.design(q0_seed)[row] @ beta_link_dev` added to q-values after residualisation
  - At FIT: same q0_seed as score-warp (rigid offset-only)
  - At PREDICT: same q0_seed recomputed

---

## D. Bernoulli Marginal-Slope Row Primary-State (u_i = η)

### Block Ordering in `block_states: &[ParameterBlockState]`
1. **Block 0: Logslope block** — `beta_logslope`, `eta = logslope_surface @ beta_logslope`
2. **Block 1 (optional): Score-warp deviation block** — `beta_score_warp`, `eta` (deviation basis evaluation)
3. **Block 2 (optional): Link-deviation deviation block** — `beta_link_dev`, `eta` (deviation basis evaluation)

### Row Linear Predictor η (single scalar component):
- **Base term**: `logslope_surface @ beta_logslope`
  - At FIT: evaluated at training rows
  - At PREDICT: evaluated at predict rows
  
- **Score-warp contribution** (optional block 1):
  - If present: `score_warp_runtime.design(z_primary)[row] @ beta_score_warp`
  - At FIT: design built from primary latent z
  - At PREDICT: residualised against logslope anchors
  
- **Link-deviation contribution** (optional block 2):
  - If present: `link_deviation_runtime.design(η_seed)[row] @ beta_link_dev`
  - At FIT: η_seed = rigid η computed from intercept + logslope offsets
  - At PREDICT: residualised against logslope anchors

### Anchor Reconstruction at Predict
- Marginal block (not in block_states): **NOT directly used** in η computation
- Logslope block: always present in block_states[0]
- Flex anchors: score_warp and link_dev runtimes store `anchor_residual` matrices computed at FIT time
  - At PREDICT: `design_with_anchor_rows()` subtracts `N_train @ M` from raw basis evaluations
  - See `src/inference/predict.rs:1138` for anchor correction matrix assembly

---

## E. Rigid Pre-Newton Solve → Pilot β Threading

### Survival Marginal-Slope Pilot Setup
- **Line 17713–17716**: Rigid offset-only q0_seed computation
  ```
  q0_seed[i] = rigid_observed_eta(
      spec.time_block.offset_exit[row] + spec.marginal_offset[row],
      baseline_slope + spec.logslope_offset[row],
      z_primary[row],
      probit_scale
  )
  ```
  - Time offset + marginal offset; baseline logslope + logslope offset
  - **No deviation blocks** at this stage (rigid-only)

- **Line 18031–18121**: Rigid-only Newton fit (PIRLS pilot)
  - Invokes `rigid_fit_with_block_newton` with 3 blocks only (time, marginal, logslope)
  - Produces `pilot_beta` which is installed as warm-start hints

- **Current issue**: Link-dev seed uses rigid offset-only solution
  - Problem: the link_dev basis is residualised against marginal+logslope+score_warp anchors
  - But the seed β that feeds into link_dev deviation basis evaluation is never updated
  - For **non-rigid link-dev**, the pilot β must be threaded to score-warp (if present) AND used to recompute q0_seed before link-dev construction

### Bernoulli Marginal-Slope Pilot Setup (equivalent pattern)
- Rigid-only pilot is constructed within `fit_bernoulli_marginal_slope_flex_blocks`
- Score-warp and link-dev bases are seeded from rigid logslope coefficients
- **Same threading requirement**: when non-rigid link-dev is added, the pilot β must flow through to seed q0_seed

### Threading Points for Phase 4
1. **FIT TIME**:
   - After rigid pilot completes: extract `pilot_result.fitted_blocks[0].beta` (time), `[1].beta` (marginal), `[2].beta` (logslope)
   - **If score-warp present**: construct score-warp basis with seed `z_primary @ rigid_coefs`
   - **If link-dev present**: recompute q0_seed using full rigid solution (time + marginal + logslope), then construct link-dev basis

2. **PREDICT TIME** (no change):
   - Reconstructed q0_seed = `offset + marginal_surface @ beta_marginal + logslope_surface @ beta_logslope`
   - Deviation bases evaluated against reconstructed q0_seed

---

## Critical Bug Summary

**Location**: `src/families/bernoulli_marginal_slope.rs:1678–1699`  
**Issue**: `CrossBlockAnchor::FlexEvaluation` is intentionally skipped when materializing anchor blocks.  
**Impact**: 
- Survival link-dev (line 17768) passes score-warp as flex anchor expecting it to be projected out
- It is NOT projected; joint design rank drops from 51 → 38 on biobank fits
- Bernoulli link-dev has same structural problem
  
**Root cause**: The comment claims flex anchors' penalised spans are "structurally orthogonal" to candidate null space — this is false when the flex anchor carries unpenalised directions identical to parametric anchors (e.g., constant columns shared by score-warp basis and logslope surface).

**Fix scope**: Phase 3 identifiability compiler must handle flex anchors with full eigen-decomposition like parametric anchors.

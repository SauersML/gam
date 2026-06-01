# Neyman-orthogonal, cross-fitted marginal-slope calibration (closes #461)

Single-source-of-truth design + math contract. Implementation lives in
`src/families/marginal_slope_orthogonal.rs` and the wiring sites named below.
This is the interface every teammate codes against; signatures here are
binding.

## 1. The leakage (grounded in the code)

Stage 1 (CTN, `transformation_normal.rs`) fits a monotone `h(y|x; θ₁)` and
emits a latent score

    z_i = Φ⁻¹(u_i),   u_i = [Φ(h_i) − Φ(L_i)] / [Φ(U_i) − Φ(L_i)]      (finite-support PIT)

with, per row i (SCOP form, `transformation_normal.rs:11-23`):

    h_i = b(x_i) + ε·(y_i − median) + Σ_k I_k(y_i)·γ_k(x_i)²
    L_i = h(y_min | x_i),  U_i = h(y_max | x_i)
    b(x_i)   = Xᶜᵒᵛ_i · θ_b           (unconstrained location column)
    γ_k(x_i) = Xᶜᵒᵛ_i · θ_{γk}         (squared SCOP shape coeffs)

Stage 2 (marginal-slope) fits, with base link `G` (probit BMS / survival):

    η_i = α(x_i) + s_f · β(x_i) · z_i  (+ link/frailty terms),   μ_i = G(η_i)

`β(x)` (logslope) is the scientific target. **`z_i` is a generated regressor:
it depends on θ̂₁.** The β estimating equation `U_β = Σ_i ρ_i · ∂η_i/∂β`
therefore has

    ∂U_β/∂θ₁ ∝ ∂η_i/∂z_i · ∂z_i/∂θ₁ = s_f·β(x_i) · ∂z_i/∂θ₁  ≠ 0.

A miscalibration error δθ₁ with x-structure (regional s(x) wrong) projects onto
β and manufactures spurious spatial heterogeneity. `score_warp = w(z)` (a free
1-D spline of z, `bms/family.rs:279-284`) spans only the **x-free** column of
this leakage, so it cannot defend the spatial case. That is #461.

## 2. Score-influence Jacobian  J = ∂z/∂θ₁   (n × p₁)

    ∂z_i/∂θ₁ = (1/φ(z_i)) · ∂u_i/∂θ₁
    ∂u_i/∂θ₁ = [ φ(h_i)·∂h_i/∂θ₁
                 − u_i·(φ(U_i)·∂U_i/∂θ₁ − φ(L_i)·∂L_i/∂θ₁)
                 − φ(L_i)·∂L_i/∂θ₁ ] / (Φ(U_i) − Φ(L_i))

with structural derivatives (all already computed inside the CTN fit; we expose
them):

    ∂h_i/∂θ_b    = Xᶜᵒᵛ_i
    ∂h_i/∂θ_{γk} = 2·I_k(y_i)·γ_k(x_i)·Xᶜᵒᵛ_i
    ∂L_i, ∂U_i   = same with I_k evaluated at the support endpoints.

## 3. Orthogonalization by an additive absorbed block (the principled fix)

Form the realized leakage directions in η-space at the rigid pilot β̂₀:

    Z_infl := diag(s_f · β̂₀(x_i)) · J        (n × p₁)

Append `Z_infl` to the Stage-2 model as a **new ADDITIVE, fixed-small-ridge,
absorbed parameter block**:

    η_i = α(x_i) + s_f·β(x_i)·z_i + (Z_infl·γ)_i  (+ link/frailty),
    joint penalty adds  ½·ρ·‖γ‖²  (ρ a fixed small ridge; γ is NOT a
    smooth/REML-learned block — it is a training-time leakage absorber).

CRITICAL — do NOT route this through the `score_warp` /
`install_compiled_flex_block_into_runtime` / `DeviationRuntime` path. Verified
against the kernel (bms, 2026-06-01): score_warp is **multiplicative** in η
(`inside = a + b·z + b·h(z)`, `bms/workspace.rs:335-358`) and `DeviationRuntime`
evaluates a 1-D cubic I-spline at **scalar** points — it cannot carry an
arbitrary n×p₁ x-dependent matrix, and installing Z_infl there computes the
wrong η. Z_infl is **plain additive** `+Z_infl·γ`.

Mechanics:
- Z_infl is orthogonalized against the **marginal** block (so it does not fight
  the location/intercept) but deliberately **overlaps logslope** (so it can
  absorb the leakage-aligned component of the slope).
- Fixed small ridge ρ on γ ⇒ as the joint solve runs, γ soaks the
  `span(Z_infl)` component of the η-residual, forcing the score of (α,β) to be
  orthogonal to `span(Z_infl)`: the discrete realization of `U_β − Π_{θ₁}[U_β]`
  = #461's `ψ − Π_η[ψ]`, tangent represented **correctly (x-dependent)**.
- gauge_priority for the influence block sits **above logslope** so shared
  directions are attributed to the absorber, not to the slope.
- At **predict** the absorber is dropped (`γ` term omitted): the orthogonalized
  β̂ is a property of the *training* fit, so prediction uses β̂ directly and
  needs no Z_infl at predict rows.

Realization is a kernel change (the BMS exact kernel hard-codes 4 blocks;
`validate_exact_block_state_shapes` rejects a 5th — `bms/workspace.rs:1565`).
The owner adds the additive absorbed block end-to-end (η-assembly, Jacobian =
Z_infl, Hessian, block validation, gauge) in the BMS kernel and the survival
kernel mirror. Pick the cleanest of: (A1) a dedicated 5th additive block, or
(A2) widen the existing additive **marginal** block with the Z_infl columns +
a block-diagonal fixed-ridge sub-penalty + predict-time column slice — BMS and
survival MUST choose the SAME realization (single source of truth).

**Why this is not the w(z,x) confounding trap:** `span(Z_infl)` is not a free
flexible function — it is the *specific, finite, data-determined* subspace that
Stage-1 sampling error inhabits. True β-heterogeneity orthogonal to it is
untouched; only β-components colinear with Stage-1's own error geometry (not
separately identified from leakage) are removed. `score_warp` is the special
case where only the x-free column of J survives ⇒ it becomes the **fallback
basis used only when no CTN Stage-1 exists** (raw `--z-column`). With a CTN
Stage-1 present we use the principled J-basis. One mechanism, correct basis.

## 4. Cross-fitting (second DML ingredient)

Own-row overfitting of θ̂₁ biases the projection. Compute z and J **out-of-fold**:
K folds; for fold f fit Stage-1 on the complement, evaluate z_f and J_f on f;
concatenate. Auto-K (5 for moderate n, fewer at biobank scale). The absorbed
projection uses OOF leakage directions.

## 5. Magic by default (no flags, no env vars, no feature gates)

Auto-enable when the workflow chains CTN Stage-1 → marginal-slope Stage-2. When
z is supplied raw (no Stage-1 model), fall back to free-warp score_warp.

CHAINING — verified: today CTN Stage-1 and marginal-slope Stage-2 are two
independent `fit_from_formula` calls with NO in-process link (the marginal-slope
materializer reads z straight from a data column, `workflow.rs:4994`). So
cross-fit has nothing to refit the CTN from. The magic-by-default mechanism (no
CLI flag, no env var — an INTERNAL config field, idiomatic here):

    pub struct CtnStage1Recipe { /* response col, covariate formula, CTN config,
                                    fixed response+covariate basis spec so per-fold
                                    J columns align */ }
    // internal FitConfig field, populated by the orchestration layer when z was
    // produced by a CTN in the same pipeline:
    ctn_stage1: Option<CtnStage1Recipe>

When the marginal-slope materializer sees a z-column AND a populated
`ctn_stage1`, it calls `crossfit_score_calibration` (refit CTN per fold on the
complement → OOF z + J), overrides z with `z_oof`, and sets
`spec.score_influence_jacobian = Some(jac_oof)`. Raw z-column with no
`ctn_stage1` ⇒ field stays None (free-warp fallback). The CTN must be refit per
fold with the SAME fixed basis spec so J's p₁ columns align across folds.

Owner (crossfit) also sets `score_influence_jacobian: None` (or Some) at the
spec construction sites in `workflow.rs` and `main.rs`; bms/survival set None at
their own in-file (test) construction sites.

## 6. Public API contract (bind to these)

`src/families/marginal_slope_orthogonal.rs`:

    /// Per-row, per-θ₁ score-influence Jacobian ∂z/∂θ₁ for a fitted CTN,
    /// PLUS the score z itself on the same rows (computing J already computes
    /// h/L/U, so z = Φ⁻¹(PIT) is free — exposing it avoids re-running the PIT
    /// path in the cross-fit fold loop; single source of truth).
    pub struct ScoreInfluenceJacobian {
        pub columns: Array2<f64>,  // n × p₁  = ∂z/∂θ₁
        pub z: Array1<f64>,        // n       = Φ⁻¹(PIT_i) on these rows
    }

    /// Compute J (and z) from a fitted CTN at the given (x,y) rows.
    pub fn score_influence_jacobian(
        fit: &TransformationNormalFitResult,
        response: &Array1<f64>,
        covariate_data: ArrayView2<f64>,
    ) -> Result<ScoreInfluenceJacobian, String>;

    /// Build the absorbed influence block Z_infl = diag(s_f·β̂₀)·J for Stage-2.
    pub fn influence_block_design(
        jac: &ScoreInfluenceJacobian,
        pilot_beta0: &Array1<f64>,   // β̂₀(x_i) at rigid pilot
        s_f: f64,
    ) -> Array2<f64>;

`solver/workflow.rs`:

    /// K-fold OOF z + J for a CTN→marginal-slope chain. None ⇒ raw z fallback.
    pub struct CrossFitScoreCalibration { pub z_oof: Array1<f64>, pub jac_oof: Array2<f64> }
    fn crossfit_score_calibration(...) -> Result<Option<CrossFitScoreCalibration>, String>;

## 7. Validation — acceptance criteria for #461

`tests/marginal_slope_neyman_orthogonal_reference.rs` (+ `test_support::reference`):

- **Sim A (false-heterogeneity control):** true β(x) ≡ const; inject x-dependent
  Stage-1 miscalibration. Assert naive Stage-2 reports significant spurious
  β(x) variation; orthogonalized Stage-2's spatial-heterogeneity statistic sits
  in the null band.
- **Sim B (power preservation):** true β(x) varying, Stage-1 well-calibrated.
  Assert orthogonalized β(x) RMSE within tolerance of naive — projection does
  not eat real signal.
- **Sim C (DML reference):** scalar θ = E[β(x)]. Assert orthogonalized
  bias/coverage matches a Python DoubleML/EconML reference under miscalibration;
  naive does not.

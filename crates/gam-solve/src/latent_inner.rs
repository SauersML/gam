//! Joint `(t, β)` inner Newton driver for [`gam_terms::latent::LatentCoordValues`]
//! blocks.
//!
//! The arrow-Schur inner step is `O(N d³ + K³)`; the REML outer
//! gradient w.r.t. `t` carries a shared `Schur⁻¹` factor and is handled
//! at the REML driver level, not here.
//!
//! ## Scope
//!
//! This module wires together:
//!
//! * [`gam_terms::latent::LatentCoordValues`] — the per-row
//!   latent field;
//! * [`crate::arrow_schur::ArrowSchurSystem`] — the bordered
//!   `(t, β)` Newton system with arrow structure;
//! * [`crate::arrow_schur::ArrowFactorCache`] — the per-row
//!   Cholesky factors saved between inner iterations and reused by the
//!   evidence-side IFT delta-t predictor loop in
//!   [`crate::evidence`].
//!
//! The driver is a thin coordinator: it expects the caller to supply a
//! closure that, given the current `(β, t)`, assembles a fresh
//! `ArrowSchurSystem` (per-row `H_tt^(i)`, `H_tβ^(i)`, `g_t^(i)`, and
//! the β-block `H_ββ`, `g_β`). The closure is the natural home for
//! "evaluate Φ(t), form Gauss–Newton blocks from the radial jet, fold
//! in analytic penalties" — all of which depend on the basis and the
//! analytic-penalty registry the outer-fit configuration owns.
//!
//! ## Convergence criterion
//!
//! The inner loop terminates when the relative joint gradient norm
//! drops below `tol`:
//!
//! ```text
//!   ‖[g_t; g_β]‖₂ / (1 + ‖[t; β]‖₂)  <  tol
//! ```
//!
//! or when the LM damping cannot be lowered any further (the latter is
//! interpreted as "we are at the steepest-descent floor; declare
//! convergence and let the outer loop tighten ρ"). Failure to factor
//! the per-row block or the Schur complement at the current ridge
//! escalates `ridge_t` / `ridge_beta` by `lm_grow` and retries.

use std::ops::Range;
use std::sync::Arc;

use ndarray::{Array1, ArrayView1};

use crate::arrow_schur::{
    ArrowFactorCache, ArrowSchurError, ArrowSchurSystem, ArrowSolveOptions, ArrowSolverMode,
    arrow_bare_quadratic_model_reduction, solve_arrow_newton_step_with_options,
};
use gam_terms::latent::LatentCoordValues;

/// Configuration for [`LatentInnerSolver::solve`].
#[derive(Debug, Clone)]
pub struct LatentInnerOptions {
    /// Maximum joint `(t, β)` Newton iterations.
    pub max_iterations: usize,
    /// Relative-gradient convergence tolerance (see module docs).
    pub convergence_tolerance: f64,
    /// Initial LM-style ridge on the per-row latent blocks.
    pub initial_ridge_t: f64,
    /// Initial LM-style ridge on the β block.
    pub initial_ridge_beta: f64,
    /// Multiplicative growth factor for the LM ridges on a rejected step.
    pub lm_grow: f64,
    /// Multiplicative shrink factor for the LM ridges on an accepted step.
    pub lm_shrink: f64,
    /// Maximum ridge value before declaring failure.
    pub max_ridge: f64,
    /// BA Schur mode for the reduced shared system. `None` selects Direct for
    /// `K <= 2000` and InexactPCG above, matching large-scale BA practice.
    pub solver_mode: Option<ArrowSolverMode>,
    /// Reduced-system trust-region radius for Steihaug-CG. This is the
    /// Ceres/BA trust-region bound layered on top of the existing LM ridges.
    pub trust_region_radius: f64,
    /// Optional β-block column ranges for the block-Jacobi Schur preconditioner.
    ///
    /// When `Some`, the solver calls
    /// [`crate::arrow_schur::ArrowSchurSystem::set_block_offsets`] on
    /// every assembled system, engaging the block-Jacobi PCG preconditioner
    /// (one dense Schur sub-block per term, max size 256 columns) instead of
    /// the scalar-diagonal fallback.
    ///
    /// Derive from `ParameterBlockSpec` slices via
    /// `gam_custom_family::block_offsets_from_specs` or an
    /// equivalent family-owned block layout.
    /// When `None`, the preconditioner falls back to scalar-diagonal Jacobi
    /// (the pre-block-Jacobi behaviour); when `Some([])` (empty slice), the
    /// same fallback applies.
    pub block_offsets: Option<Arc<[Range<usize>]>>,
}

impl Default for LatentInnerOptions {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            convergence_tolerance: 1e-8,
            initial_ridge_t: 0.0,
            initial_ridge_beta: 0.0,
            lm_grow: 10.0,
            lm_shrink: 0.5,
            max_ridge: 1e12,
            solver_mode: None,
            trust_region_radius: 1.0,
            block_offsets: None,
        }
    }
}

/// Outcome of a [`LatentInnerSolver::solve`] call.
#[derive(Debug, Clone)]
pub struct LatentInnerOutcome {
    /// Final β coefficient vector.
    pub beta: Array1<f64>,
    /// Per-row Cholesky factor cache from the *last* accepted Newton
    /// step. Consumed by the evidence-side IFT delta-t predictor loop in
    /// [`crate::evidence`].
    pub factor_cache: Option<ArrowFactorCache>,
    /// Number of iterations executed.
    pub iterations: usize,
    /// Whether the convergence test was satisfied.
    pub converged: bool,
    /// Final ridge values (for warm-starting subsequent solves).
    pub final_ridge_t: f64,
    pub final_ridge_beta: f64,
}

/// Driver trait the caller implements to assemble the arrow system at
/// the current iterate.
///
/// The driver owns the basis evaluation (`Φ(t)`), the radial jet
/// (`∂Φ/∂t` via
/// [`gam_terms::latent::LatentCoordValues::design_gradient_wrt_t`]),
/// the Gauss–Newton block assembly
/// (`H_tt^(i) ← (g_i β)(g_i β)^T`, `H_tβ^(i) ← (g_i β) ⊗ Φ_i`,
/// `H_ββ ← Φ^T W Φ + Σ_k λ_k S_k`), and the analytic-penalty fold-in via
/// [`crate::arrow_schur::ArrowSchurSystem::add_analytic_penalty_contributions`].
pub trait ArrowSystemAssembler {
    /// Build a freshly-populated arrow system at the current `(β, t)`.
    ///
    /// `latent` is the *current* latent field; the assembler should read
    /// its values via `latent.as_flat()` / `latent.row(i)` and form Φ at
    /// those coordinates. β is supplied as a view.
    fn assemble(
        &mut self,
        beta: ArrayView1<'_, f64>,
        latent: &LatentCoordValues,
    ) -> Result<ArrowSchurSystem, String>;

    /// Evaluate the true joint merit/objective at the current `(β, t)`.
    ///
    /// This is deliberately separate from [`Self::assemble`]: the Schur system
    /// is a local quadratic model, but nonlinear latent retractions must be
    /// accepted against the objective they actually change.
    fn objective(
        &mut self,
        beta: ArrayView1<'_, f64>,
        latent: &LatentCoordValues,
    ) -> Result<f64, String>;
}

/// Joint `(t, β)` inner Newton solver exploiting arrow structure.
///
/// ## Usage
///
/// 1. Call [`LatentInnerSolver::new`] with the initial `β`, a mutable
///    [`LatentCoordValues`], an [`ArrowSystemAssembler`], and options.
/// 2. Call [`LatentInnerSolver::solve`] to run the inner Newton loop.
///    Both `β` and the latent field are updated in place.
/// 3. The returned [`LatentInnerOutcome::factor_cache`] is the
///    artifact Piece 3's IFT warm-start consumes when the outer loop
///    next perturbs `(β, ρ)`.
pub struct LatentInnerSolver<'a, A: ArrowSystemAssembler> {
    pub beta: Array1<f64>,
    pub latent: &'a mut LatentCoordValues,
    pub assembler: A,
    pub options: LatentInnerOptions,
}

impl<'a, A: ArrowSystemAssembler> LatentInnerSolver<'a, A> {
    #[must_use]
    pub fn new(
        beta: Array1<f64>,
        latent: &'a mut LatentCoordValues,
        assembler: A,
        options: LatentInnerOptions,
    ) -> Self {
        Self {
            beta,
            latent,
            assembler,
            options,
        }
    }

    /// Run the joint Newton loop.
    ///
    /// Numerical-stability invariants:
    ///   * `initial_ridge_t`, `initial_ridge_beta` are clamped to `≥ 0`.
    ///   * On a per-row or Schur PD failure, both ridges are escalated
    ///     by `lm_grow` (or seeded at `1e-6` when starting from `0`).
    ///   * `max_ridge` is the cold-restart trigger: if we exhaust the
    ///     ramp and the Hessian is still non-PSD, the loop bails with a
    ///     clear diagnostic citing the iteration index and both ridge
    ///     levels reached — the outer driver should treat this as an
    ///     identifiability failure (missing gauge-fixing penalty,
    ///     collinear basis, etc.).
    pub fn solve(&mut self) -> Result<LatentInnerOutcome, String> {
        let opts = self.options.clone();
        assert!(opts.lm_grow > 1.0, "LM ridge grow factor must exceed 1");
        assert!(
            opts.lm_shrink > 0.0 && opts.lm_shrink < 1.0,
            "LM ridge shrink factor must lie in (0, 1)"
        );
        let mut ridge_t = opts.initial_ridge_t.max(0.0);
        let mut ridge_beta = opts.initial_ridge_beta.max(0.0);
        let mut last_cache: Option<ArrowFactorCache> = None;
        let mut converged = false;
        let mut iter = 0_usize;
        let mut current_objective = self
            .assembler
            .objective(self.beta.view(), self.latent)
            .map_err(|e| format!("LatentInnerSolver: objective failed at start: {e}"))?;
        if !current_objective.is_finite() {
            return Err("LatentInnerSolver: non-finite objective at start".to_string());
        }

        while iter < opts.max_iterations {
            let mut system = self
                .assembler
                .assemble(self.beta.view(), self.latent)
                .map_err(|e| format!("LatentInnerSolver: assembler failed at iter {iter}: {e}"))?;
            system.apply_riemannian_latent_geometry(self.latent);
            // Wire per-term β-block ranges so block-Jacobi engages in the PCG
            // preconditioner. Mirrors the PIRLS-driver wiring at
            // `pirls::runworking_model_pirls` line 5169–5171: the driver calls
            // `set_block_offsets` from `ArrowSchurInnerConfig.block_offsets` on
            // every system returned by the `build` closure. Here the assembler
            // owns the system, so the LatentInnerSolver is the natural place to
            // inject the offsets — one call per assembled system covers all
            // families that supply block ranges via `LatentInnerOptions` rather
            // than baking the call into each assembler impl.
            if let Some(offsets) = opts.block_offsets.as_ref() {
                system.set_block_offsets(offsets.clone());
            }

            // Convergence test: relative joint gradient norm.
            let g_norm_sq = system_gradient_norm_sq(&system);
            let scale = 1.0 + iterate_norm(self.beta.view(), self.latent.as_flat().view());
            let rel = (g_norm_sq.sqrt()) / scale;
            if rel < opts.convergence_tolerance {
                converged = true;
                // Build a final factor cache for the warm-start IFT
                // predictor even though we didn't take a step. Best-
                // effort — if the factorization fails (e.g. ill-
                // conditioned at the very converged point), skip the
                // cache; the predictor will then no-op.
                let solve_options = latent_arrow_solve_options(
                    &system,
                    &opts,
                    !self.latent.effective_is_all_euclidean(),
                );
                if let Ok((_, _, cache)) = solve_arrow_newton_step_with_options(
                    &system,
                    ridge_t.max(1e-12),
                    ridge_beta.max(1e-12),
                    &solve_options,
                ) {
                    last_cache = Some(cache);
                }
                break;
            }

            // Attempt the LM-damped arrow-Schur step. On failure (per-
            // row PD violation or Schur PD violation), grow the ridge
            // and retry without consuming an outer iteration.
            let solve_options = latent_arrow_solve_options(
                &system,
                &opts,
                !self.latent.effective_is_all_euclidean(),
            );
            let step_result =
                solve_arrow_newton_step_with_options(&system, ridge_t, ridge_beta, &solve_options);
            match step_result {
                Ok((delta_t, delta_beta, cache)) => {
                    let delta_t = limit_delta_to_riemannian_trust_region(
                        delta_t,
                        self.latent,
                        solve_options.riemannian_trust_region,
                        solve_options.trust_region.radius,
                    );
                    let predicted_reduction = arrow_bare_quadratic_model_reduction(
                        &system,
                        delta_t.view(),
                        delta_beta.view(),
                        ridge_t,
                        ridge_beta,
                    )
                    .map_err(|e| {
                        format!("LatentInnerSolver: predicted reduction failed at iter {iter}: {e}")
                    })?;
                    let beta_before = self.beta.clone();
                    let t_before = self.latent.as_flat().clone();
                    for (b, db) in self.beta.iter_mut().zip(delta_beta.iter()) {
                        *b += *db;
                    }
                    self.latent.retract_flat_delta(delta_t.view());
                    let trial_objective = self
                        .assembler
                        .objective(self.beta.view(), self.latent)
                        .map_err(|e| {
                        format!("LatentInnerSolver: objective failed at trial iter {iter}: {e}")
                    })?;
                    // Trust-region gain-ratio noise floor, keyed to the
                    // objective's own magnitude so it is equivariant under a
                    // response rescaling `y → a·y` (the penalized objective and
                    // both the predicted and actual reductions all scale as
                    // `O(a²)`). The previous `.max(1.0)` absolute floor broke
                    // this for a micro-unit response: it pinned the floor at
                    // `1e-14` while genuine reductions were `O(a²)`, treating a
                    // real step as numerical noise and stalling the inner solve
                    // at an over-smoothed iterate (issue #1127). A perfectly
                    // converged objective (`current_objective == 0`) yields a
                    // `0` floor, so the `predicted_reduction > 0` branch still
                    // governs and no step is misclassified.
                    let objective_scale = current_objective.abs();
                    let noise_floor = objective_scale * 1e-14;
                    let actual_reduction = current_objective - trial_objective;
                    let rho = if predicted_reduction > noise_floor {
                        actual_reduction / predicted_reduction
                    } else if actual_reduction >= -noise_floor {
                        1.0
                    } else {
                        -1.0
                    };
                    if rho > 0.0 && trial_objective.is_finite() {
                        current_objective = trial_objective;
                        ridge_t = (ridge_t * opts.lm_shrink).max(0.0);
                        ridge_beta = (ridge_beta * opts.lm_shrink).max(0.0);
                        last_cache = Some(cache);
                        iter += 1;
                    } else {
                        self.beta = beta_before;
                        self.latent.set_flat(t_before.view());
                        ridge_t = if ridge_t == 0.0 {
                            1e-6
                        } else {
                            ridge_t * opts.lm_grow
                        };
                        ridge_beta = if ridge_beta == 0.0 {
                            1e-6
                        } else {
                            ridge_beta * opts.lm_grow
                        };
                        if ridge_t > opts.max_ridge || ridge_beta > opts.max_ridge {
                            return Err(format!(
                                "LatentInnerSolver: LM rejected nonlinear step until ridge \
                                 exceeded max ({}) at iter {} \
                                 (ridge_t={ridge_t:.3e}, ridge_beta={ridge_beta:.3e}, \
                                 rho={rho:.3e}, predicted_reduction={predicted_reduction:.3e}, \
                                 actual_reduction={actual_reduction:.3e})",
                                opts.max_ridge, iter,
                            ));
                        }
                    }
                }
                Err(err @ ArrowSchurError::PerRowFactorFailed { .. })
                | Err(err @ ArrowSchurError::PerRowFactorIllConditioned { .. })
                | Err(err @ ArrowSchurError::SchurFactorFailed { .. })
                | Err(err @ ArrowSchurError::PcgFailed { .. })
                | Err(err @ ArrowSchurError::UnboundedNegativeCurvature { .. })
                | Err(err @ ArrowSchurError::AdaptiveCorrectionFailed { .. }) => {
                    // Grow ridges; retry without burning an iteration.
                    // The per-row `factor_blocks` already ran an internal
                    // ridge-ramp before surfacing the error here — if we
                    // see it at this layer, the row block is
                    // genuinely under-regularized (gauge issue or
                    // collinear basis under U_i). Escalate the LM ridge
                    // and let the outer Newton step damp.
                    ridge_t = if ridge_t == 0.0 {
                        1e-6
                    } else {
                        ridge_t * opts.lm_grow
                    };
                    ridge_beta = if ridge_beta == 0.0 {
                        1e-6
                    } else {
                        ridge_beta * opts.lm_grow
                    };
                    if ridge_t > opts.max_ridge || ridge_beta > opts.max_ridge {
                        return Err(format!(
                            "LatentInnerSolver: cold-restart condition — LM ridge \
                             exceeded max ({}) at iter {} \
                             (ridge_t={ridge_t:.3e}, ridge_beta={ridge_beta:.3e}); \
                             root-cause arrow-Schur error: {err}",
                            opts.max_ridge, iter,
                        ));
                    }
                }
            }
        }

        Ok(LatentInnerOutcome {
            beta: self.beta.clone(),
            factor_cache: last_cache,
            iterations: iter,
            converged,
            final_ridge_t: ridge_t,
            final_ridge_beta: ridge_beta,
        })
    }
}

fn latent_arrow_solve_options(
    system: &ArrowSchurSystem,
    opts: &LatentInnerOptions,
    riemannian_trust_region: bool,
) -> ArrowSolveOptions {
    let mut solve_options = ArrowSolveOptions::automatic(system.k);
    if let Some(mode) = opts.solver_mode {
        solve_options.mode = mode;
    }
    solve_options.trust_region.radius = opts.trust_region_radius;
    solve_options.riemannian_trust_region = riemannian_trust_region;
    solve_options
}

fn limit_delta_to_riemannian_trust_region(
    mut delta_t: Array1<f64>,
    latent: &LatentCoordValues,
    enabled: bool,
    radius: f64,
) -> Array1<f64> {
    if !enabled || !radius.is_finite() || radius <= 0.0 {
        return delta_t;
    }
    let row_weights = latent.effective_metric_weights();
    assert_eq!(row_weights.len(), latent.latent_dim());
    let mut norm_sq = 0.0_f64;
    for n in 0..latent.n_obs() {
        let start = n * latent.latent_dim();
        for a in 0..latent.latent_dim() {
            let value = delta_t[start + a];
            norm_sq += row_weights[a] * value * value;
        }
    }
    let norm = norm_sq.sqrt();
    if norm <= radius || norm == 0.0 {
        return delta_t;
    }
    let shrink = radius / norm;
    for value in delta_t.iter_mut() {
        *value *= shrink;
    }
    delta_t
}

fn system_gradient_norm_sq(sys: &ArrowSchurSystem) -> f64 {
    let mut acc = 0.0_f64;
    for j in 0..sys.k {
        acc += sys.gb[j] * sys.gb[j];
    }
    for row in &sys.rows {
        for &v in row.gt.iter() {
            acc += v * v;
        }
    }
    acc
}

fn iterate_norm(beta: ArrayView1<'_, f64>, t: ArrayView1<'_, f64>) -> f64 {
    let mut acc = 0.0_f64;
    for v in beta.iter() {
        acc += v * v;
    }
    for v in t.iter() {
        acc += v * v;
    }
    acc.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_terms::latent::{LatentCoordValues, LatentIdMode};
    use ndarray::array;

    struct ZeroAssembler {
        n: usize,
        d: usize,
        k: usize,
    }

    impl ArrowSystemAssembler for ZeroAssembler {
        fn assemble(
            &mut self,
            arr: ArrayView1<'_, f64>,
            _latent_coords: &LatentCoordValues,
        ) -> Result<ArrowSchurSystem, String> {
            assert!(arr.iter().all(|v| !v.is_nan()));
            let mut sys = ArrowSchurSystem::new(self.n, self.d, self.k);
            for j in 0..self.k {
                sys.hbb[[j, j]] = 1.0;
            }
            for row in sys.rows.iter_mut() {
                for c in 0..self.d {
                    row.htt[[c, c]] = 1.0;
                }
            }
            Ok(sys)
        }

        fn objective(
            &mut self,
            arr: ArrayView1<'_, f64>,
            _latent_coords: &LatentCoordValues,
        ) -> Result<f64, String> {
            assert!(arr.iter().all(|v| !v.is_nan()));
            Ok(0.0)
        }
    }

    #[test]
    fn zero_assembler_converges_immediately() {
        let m = array![[0.0_f64, 0.0], [0.0, 0.0]];
        let mut latent = LatentCoordValues::from_matrix(m.view(), LatentIdMode::None);
        let beta = Array1::<f64>::zeros(3);
        let assembler = ZeroAssembler { n: 2, d: 2, k: 3 };
        let mut solver =
            LatentInnerSolver::new(beta, &mut latent, assembler, LatentInnerOptions::default());
        let outcome = solver.solve().expect("zero assembler always succeeds");
        assert!(outcome.converged);
        assert_eq!(outcome.iterations, 0);
    }
}

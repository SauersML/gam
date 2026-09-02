//! Grouped LAML smoothing selection for the support-sparse TopK term.
//!
//! A separate smoothing coordinate for every atom makes the outer problem grow
//! as `O(K)` even when thousands of atoms share one declared function space.
//! The sparse lane instead shares a variance component by exact
//! `(basis kind, latent dimension)` family. The generic optimizer therefore
//! sees the number of heterogeneous families, while the inner model still has
//! distinct decoder functions and coordinates for every occupied atom.
//!
//! ## The criterion is ONE functional (#2576)
//!
//! `2·cost(ρ) = log|S| − log|S_ρ|₊ + df·(1 + ln(τ·D_p/df))`, where
//! `S = H_ββ − Σ_i H_βt^(i) H_tt^(ⁱ)⁻¹ H_tβ^(ⁱ)` is the reduced
//! decoder Schur complement. This is the same coordinate-profiled Laplace
//! complexity as the dense SAE criterion: the row-coordinate normalizer
//! `Σ_i log|H_tt^(ⁱ)|` is removed rather than charging a decoder-scale-
//! dependent quantity for nuisance coordinates.
//!
//! The normalizer comes from the #2080 frozen rational surrogate
//! ([`SurrogateLaneState`]), which the dense manifold criterion also runs. Its
//! value and fixed-state directional derivative share probes and quadrature;
//! the profiled derivative additionally carries the exact implicit response of
//! the converged inner state.
//!
//! This lane previously took its factor cache from a Newton STEP it discarded
//! and estimated that trace with a SECOND, independent Hutchinson family — its
//! own probes, its own unshifted `S⁻¹` solves, one per smoothing group. That
//! cost the evaluation's whole wall-clock, left value and gradient free to
//! describe different functions, and never ran at all: the step entry returns
//! no reduced-Schur log-determinant under `InexactPCG`, so the criterion refused
//! on every host without a CUDA device.

use std::collections::BTreeMap;

use gam_problem::{DeclaredHessianForm, Derivative, EstimationError, HessianValue, OuterEval};
use gam_solve::rho_optimizer::{
    OuterCapability, OuterCriterionCertificate, OuterEvalOrder, OuterObjective, OuterProblem,
    SeedOutcome,
};
use ndarray::{Array1, Array2, ArrayView1};

use super::*;

const SUPPORT_LAML_CONTEXT: &str = "support-sparse TopK grouped LAML";

/// The one field of the shared SAE evidence-surrogate policy this lane may not
/// inherit: the bar the frozen `log|S|` plan's Hutchinson error must clear
/// before its deflation rank stops growing.
///
/// The shared value is `0.1 · SAE_MANIFOLD_INNER_OBJECTIVE_STALL_REL_TOL` = 1e-9
/// of `|log|S|| + 1`. **That is unreachable at overcomplete border widths, and
/// unreachable here means REFUSED**, not merely slow:
/// `rational_reduced_schur_plan_derived` doubles the deflation rank until the
/// bar clears and returns `None` when its ceiling is exhausted, which this lane
/// turns into a typed evidence failure. Measured on a small overcomplete chart
/// (N=2000, P=32, K=59, border 5056, `log|S| ≈ 1.4e4`): the bare estimator's
/// relative error bar is 3.7e-3 at 8 probes and 1.9e-3 at 16, so the shared bar
/// asks for roughly a hundredfold variance reduction that peeling 128 of 5056
/// directions cannot deliver.
///
/// The reachable bar, derived from the probe count rather than borrowed from an
/// inner-solve stall tolerance: `√(2/m)`, the relative standard error a
/// `±1`-Rademacher Hutchinson estimator has in the WORST case
/// (`Var(zᵀAz) = 2‖A_off‖_F²`, attained when the off-diagonal mass matches the
/// trace). Asking for exactly that says "deflate only when this operator's
/// spectrum is worse than the theory says an `m`-probe estimator can be" — the
/// bar fires on a pathological spectrum and stays out of the way otherwise,
/// which is the only thing a variance-reduction ladder should be doing.
///
/// This does not weaken the criterion's usable accuracy, because a FROZEN plan
/// draws the SAME probes at every ρ: the estimator's error is a nearly constant
/// offset across the smoothing search, not per-ρ jitter, and the search ranks
/// differences. That common-random-numbers property is the whole reason the
/// plan is frozen, and it is what makes a worst-case-bound bar the right one
/// here rather than an accuracy compromise.
fn support_laml_deflation_target_std_err_rel() -> f64 {
    (2.0 / SCHUR_SLQ_LOGDET_PROBES as f64).sqrt()
}

/// The coarsest quadrature this lane will ever ask for, and the accuracy its
/// one-off pilot runs at.
///
/// `√(2/m)/m`: the worst-case relative standard error a `±1`-Rademacher
/// Hutchinson estimator can have (`√(2/m)`, attained when the off-diagonal
/// Frobenius mass matches the trace) divided by the probe count. It is a
/// CEILING, not the working value — see
/// [`support_laml_measured_quadrature_tolerance`] for why an a-priori bound
/// cannot produce a safe working value here.
fn support_laml_coarsest_quadrature_tolerance() -> f64 {
    support_laml_deflation_target_std_err_rel() / SCHUR_SLQ_LOGDET_PROBES as f64
}

/// Accuracy asked of the surrogate's QUADRATURE — and of the quadrature only —
/// derived from this operator's MEASURED Hutchinson resolution.
///
/// The two deterministic knobs in this surrogate look alike and are not. Both
/// read `1.0e-8` in the shared policy, and only one of them should move:
///
/// * The **quadrature** truncation and step are fixed once, in the frozen plan.
///   Their error is a smooth, ρ-independent BIAS in `log|S|` — the same
///   displacement at every ρ the outer search visits. A bias the estimator's
///   own variance swamps is a bias nobody can measure, and asking for one five
///   orders under that variance is not free: the node count grows like
///   `log(1/tol)` and the surrogate's cost is `m × nodes` shifted solves.
///   Measured on a small overcomplete chart (border 5056, `λ_min/λ_max = 1e-8`
///   by the deflation-floor convention, hence a twelve-decade padded window):
///   `1e-8` sizes **81** nodes.
/// * The **shifted-CG residual** is not bias. Each solve's iteration count
///   varies with ρ, so its error is JITTER — a non-smooth `O(δ)` wobble in a
///   criterion the outer quasi-Newton differentiates and line-searches. What
///   that must beat is the outer search's step sizes, not the probe count, so
///   `cg_rel_tol` stays at the shared lane's value.
///   (`support_outer_logdet_gradient_matches_fd_of_its_own_surrogate` is the
///   gate that catches loosening it: a central difference at `h = 1e-5`
///   amplifies value jitter by `1/2h = 5e4`.)
///
/// **Why the bias budget has to be measured.** The requirement is
/// `m·δ ≲ σ/√m`: the bias, which the average does not shrink because it is
/// identical in every term, must stay under the stochastic error, which the
/// average does shrink. Substituting the a-priori worst case for `σ` moves the
/// bound the WRONG WAY — it is an upper bound on `σ`, so it yields an upper
/// bound on the ALLOWED `δ`, and a safe working value needs a lower bound on
/// `σ` instead. There is none: `σ` is zero for a diagonal operator. Measured
/// here, `σ/√m` is 1.9e-3 at 16 probes, two orders under the `√(2/m)` bound —
/// so the bound-derived `√(2/m)/m = 7.8e-3` sits ABOVE the noise it was meant
/// to hide under, not below it.
///
/// So measure it. The surrogate reports `std_err` — its own realized error bar
/// — and a rank-0 pilot is cheap at the ceiling tolerance (18 nodes rather than
/// 81). Crucially the pilot's COARSE quadrature does not corrupt the number it
/// is measuring: a quadrature bias is common to every probe and cancels out of
/// the across-probe spread that `std_err` is. The pilot's shifted solves DO run
/// at the working `cg_rel_tol`, because per-probe solve jitter would not cancel.
///
/// The working budget is then `σ̂/m`, clamped to `[shared rel_tol, ceiling]` so
/// it is never tighter than the shared policy would have asked nor looser than
/// the worst case admits. Measured: `σ̂ ≈ 1.4e-3` at 32 probes gives `4.4e-5`
/// and 39 nodes, against 81.
fn support_laml_measured_quadrature_tolerance(
    system: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    seed: u64,
) -> Result<f64, EstimationError> {
    let shared = sae_surrogate_lane_config();
    let ceiling = support_laml_coarsest_quadrature_tolerance();
    let timer = std::time::Instant::now();
    let (plan, pilot) = rational_reduced_schur_log_det(
        system,
        htt_factors,
        0.0,
        &CpuBatchedBlockSolver,
        None,
        None,
        shared.num_probes,
        seed,
        ceiling,
        shared.power_iters,
        shared.cg_rel_tol,
        shared.cg_max_iters,
    )
    .ok_or_else(|| {
        outer_error(format!(
            "support LAML could not measure its reduced-Schur log-determinant resolution: the \
             rank-0 pilot surrogate did not evaluate on a border of width {}",
            system.k
        ))
    })?;
    let measured_relative_std_err = pilot.std_err / (pilot.estimate.abs() + 1.0);
    if !(measured_relative_std_err.is_finite() && measured_relative_std_err >= 0.0) {
        return Err(outer_error(format!(
            "support LAML pilot surrogate reported a non-finite error bar {} against estimate {}",
            pilot.std_err, pilot.estimate
        )));
    }
    let budget = (measured_relative_std_err / shared.num_probes as f64)
        .clamp(shared.rel_tol.min(ceiling), ceiling);
    log::info!(
        "support LAML quadrature budget: pilot log|S| = {:.6e}, measured relative error bar \
         {:.3e} at {} probes -> quadrature tolerance {:.3e} (shared policy {:.3e}, ceiling \
         {:.3e}); pilot {} total shifted-CG iterations, {} rational nodes, deflation rank {}, \
         {:.1}s",
        pilot.estimate,
        measured_relative_std_err,
        shared.num_probes,
        budget,
        shared.rel_tol,
        ceiling,
        pilot.cg_iterations,
        plan.nodes.len(),
        pilot.deflation_basis.len(),
        timer.elapsed().as_secs_f64(),
    );
    Ok(budget)
}

fn outer_error(message: impl Into<String>) -> EstimationError {
    EstimationError::RemlOptimizationFailed(message.into())
}

#[derive(Clone, Debug)]
pub struct SaeSupportSmoothingLayout {
    pub atom_group: Vec<usize>,
    pub group_keys: Vec<String>,
}

impl SaeSupportSmoothingLayout {
    pub fn from_term(term: &SaeSupportSparseTerm) -> Self {
        let mut keys = BTreeMap::<String, ()>::new();
        let atom_keys = term
            .atoms
            .iter()
            .map(|atom| {
                format!(
                    "{}:d{}",
                    sae_atom_basis_kind_name(atom.basis_kind()),
                    atom.latent_dim()
                )
            })
            .collect::<Vec<_>>();
        for key in &atom_keys {
            keys.insert(key.clone(), ());
        }
        let group_keys = keys.into_keys().collect::<Vec<_>>();
        let index = group_keys
            .iter()
            .enumerate()
            .map(|(group, key)| (key.clone(), group))
            .collect::<BTreeMap<_, _>>();
        let atom_group = atom_keys.iter().map(|key| index[key]).collect();
        Self {
            atom_group,
            group_keys,
        }
    }

    pub fn expand(&self, rho: &Array1<f64>) -> Result<Vec<f64>, String> {
        if rho.len() != self.group_keys.len() {
            return Err(format!(
                "SaeSupportSmoothingLayout::expand: rho length {} != groups {}",
                rho.len(),
                self.group_keys.len()
            ));
        }
        let lambdas =
            gam_problem::checked_exp_log_strengths(rho.iter().copied()).map_err(|error| {
                format!("SaeSupportSmoothingLayout::expand: invalid log strength: {error}")
            })?;
        Ok(self
            .atom_group
            .iter()
            .map(|&group| lambdas[group])
            .collect())
    }
}

/// Inner fixed-point iteration budget for the support-sparse engine.
///
/// This is a different quantity from the outer budget and must not be derived
/// from it. `max_outer_iter` counts quasi-Newton steps of the grouped-LAML
/// search over log-smoothing; `max_inner_iter` counts alternating
/// decoder/coordinate cycles spent reaching a stationary point *within a single*
/// outer evaluation. One outer step consumes a whole inner solve, so tying the
/// two together silently caps the inner solve at the length of the outer search
/// — and since [`SaeSupportSparseTerm::solve_fixed_point`] requires two
/// consecutive candidate cycles before it may report success, a small outer
/// budget then makes convergence unreachable rather than merely slow.
///
/// Both drivers of this engine read this one declaration: the tiered driver
/// through `Tier2SupportConfig`, and the public support-sparse fit entry through
/// its FFI request. They previously carried independent values, which is how they
/// came to disagree 4:1.
pub const SAE_SUPPORT_INNER_FIXED_POINT_MAX_ITER: usize = 256;

pub struct SaeSupportOuterRequest {
    pub term: SaeSupportSparseTerm,
    pub target: Array2<f64>,
    pub initial_smoothness: f64,
    pub ard_precisions: Vec<Vec<f64>>,
    pub max_outer_iter: usize,
    pub max_inner_iter: usize,
    pub inner_tolerance: f64,
    pub trust_radius: f64,
    pub random_state: u64,
}

pub struct SaeSupportOuterReport {
    pub term: SaeSupportSparseTerm,
    pub smoothing_layout: SaeSupportSmoothingLayout,
    pub log_lambda_groups: Array1<f64>,
    pub lambda_smooth: Vec<f64>,
    pub ard_precisions: Vec<Vec<f64>>,
    pub criterion: f64,
    pub fixed_point: SaeSupportFixedPointReport,
    pub outer_iterations: usize,
    pub outer_certificate: OuterCriterionCertificate,
}

struct PenaltySpectrum {
    rank_by_group: Vec<usize>,
    log_pdet_base_by_group: Vec<f64>,
    total_rank: usize,
}

#[derive(Clone)]
struct SupportOuterEvaluation {
    cost: f64,
    gradient: Array1<f64>,
    lambda_smooth: Vec<f64>,
    fixed_point: SaeSupportFixedPointReport,
}

#[derive(Clone)]
struct CachedSupportOuterEvaluation {
    rho: Array1<f64>,
    state_revision: u64,
    available_order: OuterEvalOrder,
    evaluation: SupportOuterEvaluation,
}

impl CachedSupportOuterEvaluation {
    fn matches(
        &self,
        rho: &Array1<f64>,
        state_revision: u64,
        requested_order: OuterEvalOrder,
    ) -> bool {
        self.state_revision == state_revision
            && self.rho.len() == rho.len()
            && self
                .rho
                .iter()
                .zip(rho.iter())
                .all(|(cached, requested)| cached.to_bits() == requested.to_bits())
            && match self.available_order {
                OuterEvalOrder::Value => requested_order == OuterEvalOrder::Value,
                OuterEvalOrder::ValueAndGradient => {
                    requested_order != OuterEvalOrder::ValueGradientHessian
                }
                OuterEvalOrder::ValueGradientHessian => true,
            }
    }
}

struct SaeSupportOuterObjective {
    term: SaeSupportSparseTerm,
    initial_term: SaeSupportSparseTerm,
    target: Array2<f64>,
    layout: SaeSupportSmoothingLayout,
    spectrum: PenaltySpectrum,
    ard_precisions: Vec<Vec<f64>>,
    max_inner_iter: usize,
    inner_tolerance: f64,
    trust_radius: f64,
    random_state: u64,
    last_evaluation: Option<CachedSupportOuterEvaluation>,
    /// Identity of the mutable fitted term underlying `last_evaluation`.
    /// Every uncached solve and reset advances it before touching the state, so
    /// a cache entry can never survive a failed or intervening evaluation.
    state_revision: u64,
    /// Uncached-solve counter. Unconditional rather than `#[cfg(test)]`:
    /// the ban scanner rejects `#[cfg(test)]` on a `src/` item (it is not a
    /// dead-code escape hatch), and one `usize` per outer solve is not worth a
    /// cfg split.
    uncached_evaluations: usize,
    /// The FROZEN reduced-Schur log-determinant surrogate for this outer solve
    /// — the same #2080 lane the dense manifold criterion runs on.
    ///
    /// Built once, on the first evaluation, and reused at every subsequent ρ.
    /// The plan is the criterion's identity: its probes, quadrature nodes, and
    /// deflation basis fix WHICH function of ρ the outer search descends.
    /// Rebuilding it per ρ would evaluate a different function at every point,
    /// and the exact directional derivative the gradient contracts would then
    /// be the exact gradient of something nobody evaluated twice.
    logdet_surrogate: Option<SurrogateLaneState>,
}

fn penalty_spectrum(
    term: &SaeSupportSparseTerm,
    layout: &SaeSupportSmoothingLayout,
) -> Result<PenaltySpectrum, String> {
    let groups = layout.group_keys.len();
    let mut rank_by_group = vec![0usize; groups];
    let mut log_pdet_base_by_group = vec![0.0; groups];
    for (atom_idx, atom) in term.atoms.iter().enumerate() {
        let symmetric = (atom.smooth_penalty() + &atom.smooth_penalty().t()) * 0.5;
        let (values, _) = symmetric
            .eigh(Side::Lower)
            .map_err(|error| format!("support smooth-penalty eigendecomposition: {error}"))?;
        let scale = values.iter().copied().fold(0.0_f64, f64::max).max(1.0);
        let tolerance = f64::EPSILON.sqrt() * scale * atom.basis_size().max(1) as f64;
        if values.iter().any(|value| *value < -tolerance) {
            return Err(format!(
                "support smooth penalty for atom {atom_idx} is not positive semidefinite"
            ));
        }
        let group = layout.atom_group[atom_idx];
        for value in values.iter().copied().filter(|value| *value > tolerance) {
            rank_by_group[group] = rank_by_group[group]
                .checked_add(term.output_dim())
                .ok_or_else(|| "support penalty rank overflow".to_string())?;
            log_pdet_base_by_group[group] += term.output_dim() as f64 * value.ln();
        }
    }
    let total_rank = rank_by_group.iter().sum();
    Ok(PenaltySpectrum {
        rank_by_group,
        log_pdet_base_by_group,
        total_rank,
    })
}

impl SaeSupportOuterObjective {
    fn beta_layout(&self) -> Result<(Vec<usize>, usize), EstimationError> {
        self.term.beta_layout().map_err(outer_error)
    }

    fn penalty_energy_by_group(&self, lambda_smooth: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.layout.group_keys.len()];
        for atom in 0..self.term.k_atoms() {
            let sb = self.term.atoms[atom]
                .smooth_penalty()
                .dot(self.term.atoms[atom].decoder_coefficients());
            let energy = self.term.atoms[atom]
                .decoder_coefficients()
                .iter()
                .zip(sb.iter())
                .map(|(left, right)| left * right)
                .sum::<f64>();
            out[self.layout.atom_group[atom]] += lambda_smooth[atom] * energy;
        }
        out
    }

    /// `∂S/∂ρ_g` as a matvec: the EXACT reduced-Schur derivative along one
    /// smoothing coordinate.
    ///
    /// The chain is short and it is worth stating in full, because it is what
    /// lets this lane drop its separate trace estimator entirely. The joint
    /// Hessian's dependence on the smoothing coordinates is confined to the
    /// β block: `H_tt^(i)` is the Gauss–Newton latent curvature plus the ARD
    /// prior and `H_tβ^(i)` is the cross term, and neither carries a λ. So on
    /// `S = H_ββ − Σ_i H_βt^(i)(H_tt^(i))⁻¹H_tβ^(i)` only the first term moves,
    /// and since `λ_g = exp(ρ_g)`,
    ///
    /// ```text
    /// ∂S/∂ρ_g = ∂H_ββ/∂ρ_g = Σ_{k ∈ g} λ_k (S_k ⊗ I_P) = the group-g penalty apply.
    /// ```
    ///
    /// This is the EXPLICIT, frozen-inner-state derivative. The coordinates and
    /// decoder fitted at the inner optimum also move with ρ, so the complete
    /// profiled derivative adds the implicit-function response assembled in
    /// [`SaeSupportSparseTerm::support_reduced_logdet_profile_adjoint`]. Keeping
    /// this operator restricted to the explicit penalty direction makes that
    /// split auditable: no state-response term can be counted twice.
    fn schur_derivative_matvec(
        &self,
        group: usize,
        lambda_smooth: &[f64],
        beta_offsets: &[usize],
        beta_dim: usize,
        vector: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let output_dim = self.term.output_dim();
        let mut out = Array1::<f64>::zeros(beta_dim);
        for atom in 0..self.term.k_atoms() {
            if self.layout.atom_group[atom] != group {
                continue;
            }
            let m = self.term.atoms[atom].basis_size();
            let offset = beta_offsets[atom];
            let lambda = lambda_smooth[atom];
            let penalty = self.term.atoms[atom].smooth_penalty();
            for left in 0..m {
                for right in 0..m {
                    let weight = lambda * penalty[[left, right]];
                    for channel in 0..output_dim {
                        out[offset + left * output_dim + channel] +=
                            weight * vector[offset + right * output_dim + channel];
                    }
                }
            }
        }
        out
    }

    /// The coordinate-profiled Laplace normalizer `log|S|`, and the bundle whose
    /// contraction against any `∂S` is that value's exact fixed-state
    /// directional derivative.
    ///
    /// This lane used to obtain its factor cache from
    /// `solve_arrow_newton_step_with_options`, which solves a Newton step it
    /// then discarded. That was not merely wasteful: `InexactPCG` never forms a
    /// dense `k × k` reduced-Schur factor, so the cache came back with
    /// `schur_factor_is_undamped = false`, `arrow_log_det()` returned `None`,
    /// and the criterion could not evaluate AT ALL on a host without a CUDA
    /// device — the only place a matrix-free `log|S|` was reachable from that
    /// entry. Meanwhile the discarded step still paid for a
    /// `JacobiPreconditioner` build, `O(n·K)` at the overcomplete border, which
    /// is where the reported minutes of silent burn actually went (#2576).
    ///
    /// What a criterion needs instead is the evidence factorization:
    /// `Σ_i log|H_tt^(i)|` from the undamped per-row Cholesky, and `log|S|` by a
    /// route it can also DIFFERENTIATE. That route already exists and is already
    /// production on the dense manifold lane — the #2080 frozen rational
    /// surrogate ([`matrix_free_arrow_evidence_log_det_surrogate`] driven by a
    /// [`SurrogateLaneState`]). It never forms a dense border, it runs on CPU
    /// and device alike, and its value and gradient are one functional by
    /// construction. The support lane joins it rather than growing a second
    /// evidence policy beside it; [`Self::schur_derivative_matvec`] supplies the
    /// explicit-ρ half and the support-term adjoint supplies the fitted-state
    /// half.
    fn evidence_log_det(
        &mut self,
        system: &ArrowSchurSystem,
    ) -> Result<(f64, RationalLogdetDerivativeBundle), EstimationError> {
        if system.k == 0 {
            return Err(outer_error(
                "support LAML requires a decoder border to select smoothing against; the \
                 assembled arrow system has none",
            ));
        }
        // One SAE evidence-surrogate policy, shared with the dense manifold
        // lane, with exactly two fields DERIVED rather than inherited — and the
        // derivation happens ONCE, here, at the first ρ, so the whole outer
        // search descends a single functional.
        let seed = self.random_state;
        if self.logdet_surrogate.is_none() {
            // `evidence_factorization = true` must match what the lane itself
            // will use, or the pilot would measure a different operator from
            // the one the frozen plan is built on. It does: the lane runs
            // `ArrowEvidencePolicy::PositiveDefinite`
            // (`with_positive_definite_evidence` below), and
            // `factors_undamped_evidence()` is `!matches!(self, Strict)` — true
            // for every policy except `Strict`, which this lane never selects.
            let htt_factors = CpuBatchedBlockSolver
                .factor_blocks(&system.rows, 0.0, system.d, true)
                .map_err(|error| {
                    outer_error(format!(
                        "support LAML undamped evidence row factorization: {error}"
                    ))
                })?;
            let rel_tol =
                support_laml_measured_quadrature_tolerance(system, &htt_factors, seed)?;
            self.logdet_surrogate = Some(SurrogateLaneState::new(SurrogateLaneConfig {
                // The seed is the caller's: a support fit's `random_state` is
                // what makes ITS criterion bit-reproducible, and two fits of the
                // same data at different seeds must be able to disagree about
                // their probes without disagreeing about their policy.
                seed,
                deflation_target_std_err_rel: support_laml_deflation_target_std_err_rel(),
                rel_tol,
                ..sae_surrogate_lane_config()
            }));
        }
        let lane = self
            .logdet_surrogate
            .as_mut()
            .expect("the surrogate lane was just installed");
        // Ask for the derivative representation BEFORE the value, so a failed
        // evaluation can never be paired with a previous operator's gradient.
        lane.request_logdet_derivative_bundle();
        let timer = std::time::Instant::now();
        let options = ArrowSolveOptions::inexact_pcg().with_positive_definite_evidence();
        let evaluated = matrix_free_arrow_evidence_log_det_surrogate(
            system,
            0.0,
            0.0,
            &options,
            SCHUR_SLQ_LOGDET_PROBES,
            SCHUR_SLQ_LOGDET_LANCZOS_STEPS,
            SCHUR_SLQ_LOGDET_SEED,
            Some(lane),
        );
        let (row_log_det, schur_log_det) = match evaluated {
            Ok(split) => split,
            Err(error) => {
                drop(lane.take_logdet_derivative_bundle());
                return Err(outer_error(format!(
                    "support LAML matrix-free evidence log-determinant: {error}"
                )));
            }
        };
        let bundle = lane.take_logdet_derivative_bundle().ok_or_else(|| {
            outer_error(
                "support LAML evidence evaluation did not emit the rational value's derivative \
                 bundle, so no smoothing gradient can be minted from it",
            )
        })?;
        let metrics = bundle.evaluation_metrics();
        // The expensive half of one outer evaluation lives in this call, and
        // before #2576 it emitted nothing at all — six minutes of fourteen busy
        // cores between two log lines is what kept the cost invisible.
        log::info!(
            "support LAML evidence: border {}, row log|H_tt| = {:.6e}, surrogate log|S| = \
             {:.6e}; {} total shifted-CG iterations, {} rational nodes, deflation rank {}, \
             {:.1}s",
            system.k,
            row_log_det,
            schur_log_det,
            metrics.cg_iterations,
            metrics.node_count,
            metrics.deflation_rank,
            timer.elapsed().as_secs_f64(),
        );
        // Coordinates are nuisance parameters profiled by the inner solve. As
        // in `rank_adjusted_quasi_laplace_complexity`, their row-block
        // determinant is removed from the criterion; retaining it here would
        // charge arbitrary decoder scaling and, worse, would pair a moving row
        // value with a derivative that differentiates only the reduced Schur
        // operator.
        Ok((schur_log_det, bundle))
    }

    fn evaluate_for_order(
        &mut self,
        rho: &Array1<f64>,
        requested_order: OuterEvalOrder,
    ) -> Result<SupportOuterEvaluation, EstimationError> {
        if let Some(cached) = &self.last_evaluation
            && cached.matches(rho, self.state_revision, requested_order)
        {
            return Ok(cached.evaluation.clone());
        }

        // Invalidate before the inner solve mutates `term`: a refused
        // evaluation must never leave the previous point reusable against a
        // different fitted state. Wrapping is safe because the sole cache entry
        // is gone before the revision advances.
        self.last_evaluation = None;
        self.state_revision = self.state_revision.wrapping_add(1);
        self.uncached_evaluations += 1;
        let evaluation = self.evaluate_uncached(rho)?;
        // The implementation always computes the analytic gradient alongside
        // the value. A value request therefore populates a gradient-capable
        // entry; a declared VGH request additionally establishes that the
        // unavailable-Hessian result is the requested maximum capability.
        let available_order = match requested_order {
            OuterEvalOrder::ValueGradientHessian => OuterEvalOrder::ValueGradientHessian,
            OuterEvalOrder::Value | OuterEvalOrder::ValueAndGradient => {
                OuterEvalOrder::ValueAndGradient
            }
        };
        self.last_evaluation = Some(CachedSupportOuterEvaluation {
            rho: rho.clone(),
            state_revision: self.state_revision,
            available_order,
            evaluation: evaluation.clone(),
        });
        Ok(evaluation)
    }

    fn evaluate(&mut self, rho: &Array1<f64>) -> Result<SupportOuterEvaluation, EstimationError> {
        self.evaluate_for_order(rho, OuterEvalOrder::ValueAndGradient)
    }

    fn evaluate_uncached(
        &mut self,
        rho: &Array1<f64>,
    ) -> Result<SupportOuterEvaluation, EstimationError> {
        let lambda_smooth = self.layout.expand(rho).map_err(outer_error)?;
        let fixed_point = self
            .term
            .solve_fixed_point(
                self.target.view(),
                &lambda_smooth,
                &self.ard_precisions,
                self.max_inner_iter,
                self.inner_tolerance,
                self.trust_radius,
            )
            .map_err(outer_error)?;
        let system = self
            .term
            .assemble_arrow_schur(self.target.view(), &lambda_smooth, &self.ard_precisions)
            .map_err(outer_error)?;
        let (reduced_logdet, logdet_derivative) = self.evidence_log_det(&system)?;
        // Gaussian dispersion argument = the PENALIZED deviance
        //   D_p(ρ) = ‖y − ŷ‖² + β̂ᵀ S_ρ β̂ + (every other penalty the inner solve descends),
        // NOT the raw residual sum of squares. This mirrors the canonical dense
        // manifold path, whose profiled-scale data term ranks `loss.total()`
        // (data_fit + smoothness + ard + sparsity) — the full penalized loss —
        // precisely so the envelope theorem makes the analytic outer gradient
        // exact (construction_quasi_laplace.rs:357). `penalized_objective` returns
        // ½·D_p (½‖y−ŷ‖² + ½Σ_k λ_k β̂ᵀS_kβ̂ + ARD), so 2× recovers D_p. At the inner
        // optimum β̂ minimizes the full penalized objective, hence the envelope
        // theorem gives d D_p/dρ_g = ∂_ρ_g D_p|_{β̂} = Σ_{k∈g} λ_k β̂ᵀS_kβ̂ = energy[g]
        // — the exact numerator the gradient below already forms. With the RAW rss
        // instead, d(rss)/dρ_g carries an implicit H⁻¹ envelope term ≠ energy[g],
        // so value and gradient would descend different functions (the desync bug).
        let deviance = 2.0
            * self
                .term
                .penalized_objective(self.target.view(), &lambda_smooth, &self.ard_precisions)
                .map_err(outer_error)?;
        if !(deviance.is_finite() && deviance > 0.0) {
            return Err(outer_error(format!(
                "support LAML requires positive finite penalized deviance; got {deviance}"
            )));
        }
        let (beta_offsets, beta_dim) = self.beta_layout()?;
        let beta_nullity = beta_dim
            .checked_sub(self.spectrum.total_rank)
            .ok_or_else(|| outer_error("support smooth penalty rank exceeds beta dimension"))?;
        let data_dim = self
            .term
            .n_obs()
            .checked_mul(self.term.output_dim())
            .ok_or_else(|| outer_error("support LAML data dimension overflow"))?;
        if data_dim <= beta_nullity {
            return Err(outer_error(format!(
                "support LAML requires more response cells than unpenalized decoder coefficients; got {data_dim} <= {beta_nullity}"
            )));
        }
        let residual_df = (data_dim - beta_nullity) as f64;
        let mut penalty_logdet = 0.0;
        for group in 0..self.layout.group_keys.len() {
            penalty_logdet += self.spectrum.log_pdet_base_by_group[group]
                + self.spectrum.rank_by_group[group] as f64 * rho[group];
        }
        let cost = 0.5
            * (reduced_logdet - penalty_logdet
                + residual_df * (1.0 + (std::f64::consts::TAU * deviance / residual_df).ln()));
        // The surrogate's directional derivative is the exact FIXED-STATE
        // derivative of the very `log|S|` that entered `cost` above — same
        // probes, quadrature nodes, frozen deflation basis and shifted solves.
        // A profiled criterion needs one more term: S also moves because the
        // converged decoder/coordinates move with rho. The support term forms
        // `Gamma = d log|S|/d theta` from this same bundle and solves the exact
        // inner stationarity adjoint once; omitting that IFT response is the
        // value/gradient split that made every descent direction point uphill.
        // The lane used to estimate this trace with a SECOND, independent
        // Hutchinson family (its own 16 probes, its own unshifted `S⁻¹` solves,
        // one per group), which both cost the whole evaluation's wall-clock and
        // left the value and gradient free to describe different functions.
        let energy = self.penalty_energy_by_group(&lambda_smooth);
        let profile_adjoint = self
            .term
            .support_reduced_logdet_profile_adjoint(
                self.target.view(),
                &self.ard_precisions,
                &system,
                &logdet_derivative.vectors,
            )
            .map_err(outer_error)?;
        let mut gradient = Array1::<f64>::zeros(self.layout.group_keys.len());
        for group in 0..gradient.len() {
            let derivative_matvec = |vector: ArrayView1<f64>| -> Array1<f64> {
                self.schur_derivative_matvec(
                    group,
                    &lambda_smooth,
                    &beta_offsets,
                    beta_dim,
                    vector,
                )
            };
            let logdet_group_derivative = logdet_derivative
                .directional_derivative(&derivative_matvec)
                .ok_or_else(|| {
                    outer_error(format!(
                        "support LAML reduced-Schur surrogate produced no derivative for \
                         smoothing group {group} ({})",
                        self.layout.group_keys[group]
                    ))
                })?;
            // `g_rho = d(∇_theta L)/d rho` lives only in the decoder
            // block and equals the group penalty applied to beta. With
            // `a = A^+ Gamma`, the implicit logdet response is
            // `-1/2 <a, g_rho>` (one adjoint, then one cheap contraction per
            // smoothing group).
            let mut profile_response = 0.0_f64;
            for atom in 0..self.term.k_atoms() {
                if self.layout.atom_group[atom] != group {
                    continue;
                }
                let m = self.term.atoms[atom].basis_size();
                let offset = beta_offsets[atom];
                let lambda = lambda_smooth[atom];
                let sb = self.term.atoms[atom]
                    .smooth_penalty()
                    .dot(self.term.atoms[atom].decoder_coefficients());
                for basis in 0..m {
                    for output in 0..self.term.output_dim() {
                        profile_response += profile_adjoint.beta
                            [offset + basis * self.term.output_dim() + output]
                            * lambda
                            * sb[[basis, output]];
                    }
                }
            }
            gradient[group] = 0.5
                * (logdet_group_derivative - self.spectrum.rank_by_group[group] as f64
                    + residual_df * energy[group] / deviance
                    - profile_response);
        }
        if !cost.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
            return Err(outer_error(
                "support LAML produced a non-finite value or gradient",
            ));
        }
        Ok(SupportOuterEvaluation {
            cost,
            gradient,
            lambda_smooth,
            fixed_point,
        })
    }
}

impl OuterObjective for SaeSupportOuterObjective {
    fn capability(&self) -> OuterCapability {
        OuterCapability {
            gradient: Derivative::Analytic,
            hessian: DeclaredHessianForm::Unavailable,
            n_params: self.layout.group_keys.len(),
            psi_dim: 0,
            fixed_point_available: false,
            barrier_config: None,
            prefer_gradient_only: true,
            disable_fixed_point: true,
        }
    }

    fn eval_cost(&mut self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        self.evaluate_for_order(rho, OuterEvalOrder::Value)
            .map(|evaluation| evaluation.cost)
    }

    fn eval(&mut self, rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
        let evaluation = self.evaluate_for_order(rho, OuterEvalOrder::ValueAndGradient)?;
        Ok(OuterEval {
            cost: evaluation.cost,
            gradient: evaluation.gradient,
            hessian: HessianValue::Unavailable,
            inner_beta_hint: None,
        })
    }

    fn eval_with_order(
        &mut self,
        rho: &Array1<f64>,
        order: OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        let evaluation = self.evaluate_for_order(rho, order)?;
        match order {
            OuterEvalOrder::Value => Ok(OuterEval::value_only(evaluation.cost, rho.len(), None)),
            OuterEvalOrder::ValueAndGradient | OuterEvalOrder::ValueGradientHessian => {
                Ok(OuterEval {
                    cost: evaluation.cost,
                    gradient: evaluation.gradient,
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            }
        }
    }

    fn reset(&mut self) {
        self.term = self.initial_term.clone();
        self.last_evaluation = None;
        self.state_revision = self.state_revision.wrapping_add(1);
    }

    fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
        if beta.iter().any(|value| !value.is_finite()) {
            return Err(outer_error(
                "support outer seed contains a non-finite value",
            ));
        }
        Ok(SeedOutcome::NoSlot)
    }
}

/// Select topology-grouped smoothing strengths through the shared generic
/// outer optimizer. Only a terminal point with an analytic stationarity
/// certificate and a recurring raw inner fixed point is returned.
pub fn run_sae_support_outer(
    request: SaeSupportOuterRequest,
) -> Result<SaeSupportOuterReport, EstimationError> {
    if !(request.initial_smoothness.is_finite() && request.initial_smoothness > 0.0) {
        return Err(outer_error(format!(
            "support outer initial_smoothness must be finite and positive; got {}",
            request.initial_smoothness
        )));
    }
    let layout = SaeSupportSmoothingLayout::from_term(&request.term);
    if layout.group_keys.is_empty() {
        return Err(outer_error(
            "support outer requires at least one smoothing group",
        ));
    }
    let spectrum = penalty_spectrum(&request.term, &layout).map_err(outer_error)?;
    // This Gaussian LAML criterion is a sum over response cells, not a
    // unit-scale scalar.  Declare that natural scale to the shared outer
    // engine so its projected-gradient stopping band has the same units as
    // the score.  Without it BFGS used the bare absolute 1e-5 default even
    // when the score-relative certificate at the same point was already
    // decisive; on the small overcomplete Tier-2 witness that drove Strong
    // Wolfe into sub-1e-8 rho probes whose predicted score change was below
    // floating-point cancellation, with each probe paying for a full inner
    // fixed point and rational evidence evaluation.
    let objective_scale = request.target.len() as f64;
    let initial_term = request.term.clone();
    let mut objective = SaeSupportOuterObjective {
        term: request.term,
        initial_term,
        target: request.target,
        layout: layout.clone(),
        spectrum,
        ard_precisions: request.ard_precisions.clone(),
        max_inner_iter: request.max_inner_iter,
        inner_tolerance: request.inner_tolerance,
        trust_radius: request.trust_radius,
        random_state: request.random_state,
        last_evaluation: None,
        state_revision: 0,
        uncached_evaluations: 0,
        logdet_surrogate: None,
    };
    let initial_rho = Array1::from_elem(layout.group_keys.len(), request.initial_smoothness.ln());
    let problem = OuterProblem::new(layout.group_keys.len())
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_prefer_gradient_only(true)
        .with_disable_fixed_point(true)
        .with_objective_scale(Some(objective_scale))
        .with_initial_rho(initial_rho)
        .with_max_iter(request.max_outer_iter.max(1));
    let outer = problem.run(&mut objective, SUPPORT_LAML_CONTEXT)?;
    let certificate = outer
        .criterion_certificate
        .clone()
        .filter(OuterCriterionCertificate::certifies)
        .ok_or_else(|| {
            outer_error(format!(
                "support outer returned without an analytic stationarity certificate after {} iterations",
                outer.iterations
            ))
        })?;
    let terminal = objective.evaluate(&outer.rho)?;
    if !terminal.fixed_point.recurred {
        return Err(outer_error(
            "support outer terminal inner state did not recur",
        ));
    }
    Ok(SaeSupportOuterReport {
        term: objective.term,
        smoothing_layout: layout,
        log_lambda_groups: outer.rho,
        lambda_smooth: terminal.lambda_smooth,
        ard_precisions: request.ard_precisions,
        criterion: terminal.cost,
        fixed_point: terminal.fixed_point,
        outer_iterations: outer.iterations,
        outer_certificate: certificate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assignment_state::{SaeAssignmentAtomSpec, SaeAssignmentState};
    use ndarray::array;
    use std::sync::Arc;

    fn atom(
        name: &str,
        kind: SaeAtomBasisKind,
        d: usize,
        evaluator: Arc<dyn SaeBasisSecondJet>,
        coords: &[f64],
        decoder: Array2<f64>,
    ) -> SaeManifoldAtom {
        let coord = Array2::from_shape_vec((1, d), coords.to_vec()).expect("coords");
        let (phi, jet) = evaluator.evaluate(coord.view()).expect("evaluate");
        let m = phi.ncols();
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            kind,
            d,
            phi,
            jet,
            decoder,
            Array2::eye(m),
        )
        .expect("atom")
        .with_basis_second_jet(evaluator)
    }

    /// Two-group heterogeneous support fixture (K = 2 atoms > P = 1 response
    /// cells) whose smoothing selection is the exact regime the outer criterion
    /// desync afflicts. The target leaves a genuine residual so the penalized
    /// deviance and every per-group penalty energy are strictly positive.
    ///
    /// The periodic latent coordinate is seeded AWAY from a quarter period. With
    /// one active row per atom the exact ridge decoder block makes the fitted row
    /// interpolate the target, so the likelihood coordinate Jacobian
    /// `b·phi'(t)` collapses and the coordinate normal equation is carried
    /// entirely by the von-Mises PSD majorizer `max(alpha·cos(kappa·t), 0)`. That
    /// majorizer is exactly zero at `t = p/4` (period `p`), where the prior
    /// gradient `(alpha/kappa)·sin(kappa·t)` is still maximal — a zero-curvature
    /// block with a non-zero right-hand side, which `solve_psd_minimum_norm`
    /// (correctly) refuses as an RHS in the normal-equation null space. Seeding at
    /// `t = 0.1` (period 1) keeps `cos(kappa·t) = 0.809 > 0`, and because the
    /// von-Mises prior pulls the coordinate toward its mode at `t = 0`, every
    /// later iterate moves further from `p/4`, so the coordinate block stays
    /// strictly positive definite all the way to the recurring fixed point.
    fn build_objective() -> SaeSupportOuterObjective {
        let periodic_eval: Arc<dyn SaeBasisSecondJet> =
            Arc::new(PeriodicHarmonicEvaluator::new(3).expect("periodic"));
        let patch_eval: Arc<dyn SaeBasisSecondJet> =
            Arc::new(EuclideanPatchEvaluator::new(2, 1).expect("patch"));
        let atoms = vec![
            atom(
                "circle",
                SaeAtomBasisKind::Periodic,
                1,
                periodic_eval,
                &[0.3],
                array![[0.2], [1.1], [-0.4]],
            ),
            atom(
                "plane",
                SaeAtomBasisKind::Linear,
                2,
                patch_eval,
                &[0.1, -0.2],
                array![[0.3], [2.0], [-1.0]],
            ),
        ];
        let specs = vec![
            SaeAssignmentAtomSpec {
                latent_dim: 1,
                id_mode: LatentIdMode::None,
                manifold: SaeAtomBasisKind::Periodic.latent_manifold(1),
                retraction: gam_problem::LatentRetractionRegistry::all_euclidean(),
                latent_id: 1,
            },
            SaeAssignmentAtomSpec::euclidean(2),
        ];
        let state = SaeAssignmentState::from_topk_support_heterogeneous(
            2,
            2,
            1,
            specs,
            vec![vec![0], vec![1]],
            vec![vec![9.0], vec![-4.0]],
            vec![vec![0.1], vec![3.0, 1.0]],
        )
        .expect("state");
        let term = SaeSupportSparseTerm::new(atoms, state).expect("term");
        let layout = SaeSupportSmoothingLayout::from_term(&term);
        assert_eq!(layout.group_keys.len(), 2, "fixture must expose two groups");
        let spectrum = penalty_spectrum(&term, &layout).expect("spectrum");
        let initial_term = term.clone();
        SaeSupportOuterObjective {
            term,
            initial_term,
            target: array![[1.4], [4.3]],
            layout,
            spectrum,
            ard_precisions: vec![vec![1.0], vec![1.0, 1.0]],
            max_inner_iter: 5000,
            inner_tolerance: 1.0e-9,
            trust_radius: 1.0,
            random_state: 0xC0FF_EE00_D15E_A5E5,
            last_evaluation: None,
            state_revision: 0,
            uncached_evaluations: 0,
            logdet_surrogate: None,
        }
    }

    /// #2634 — the first-order bridge asks for a value during line search and
    /// then for value+gradient at the accepted point. This objective computes
    /// both in one evidence pass, so an exact repeated rho must consume that
    /// pass once. A one-bit-different rho is the negative control: the cache is
    /// exact, never a tolerance-based warm-start shortcut.
    #[test]
    fn identical_value_then_gradient_evaluates_support_laml_once_2634() {
        let mut objective = build_objective();
        let rho = array![0.0, 0.0];
        let value = objective
            .eval_with_order(&rho, OuterEvalOrder::Value)
            .expect("value evaluation");
        assert_eq!(objective.uncached_evaluations, 1);

        let value_and_gradient = objective
            .eval_with_order(&rho, OuterEvalOrder::ValueAndGradient)
            .expect("the full evaluation reuses the value pass");
        assert_eq!(
            objective.uncached_evaluations, 1,
            "an identical rho must not repeat the inner solve and evidence pass"
        );
        assert_eq!(value.cost.to_bits(), value_and_gradient.cost.to_bits());
        assert!(
            value_and_gradient.gradient.iter().all(|value| value.is_finite()),
            "the reused full evaluation must retain its analytic gradient"
        );

        let mut adjacent = rho.clone();
        adjacent[0] = f64::from_bits(rho[0].to_bits() + 1);
        objective
            .eval_with_order(&adjacent, OuterEvalOrder::Value)
            .expect("one-bit-adjacent rho evaluation");
        assert_eq!(
            objective.uncached_evaluations, 2,
            "a one-bit rho change must invalidate the exact cache"
        );
    }

    /// #2634 root gate: a smoothing gradient is the derivative of the fully
    /// profiled production criterion, not the partial derivative obtained by
    /// freezing the coordinates and decoder at the centre point. The latter
    /// dropped the `Gamma^T theta_hat_rho` Laplace response and, together with
    /// the spurious row-coordinate determinant, reported an uphill direction
    /// as descent on the filed Tier-2 route.
    #[test]
    fn profiled_support_outer_gradient_matches_refitted_value_fd_2634() {
        let mut objective = build_objective();
        let base = array![0.4_f64.ln(), 2.2_f64.ln()];
        let analytic = objective
            .evaluate(&base)
            .expect("profiled centre evaluation")
            .gradient;
        let h = 2.0e-4;
        for group in 0..base.len() {
            let mut plus = base.clone();
            let mut minus = base.clone();
            plus[group] += h;
            minus[group] -= h;

            // Both legs start from the same pristine inner state. The frozen
            // rational plan intentionally survives reset so centre and legs
            // are values of one deterministic surrogate.
            objective.reset();
            let value_plus = objective
                .evaluate(&plus)
                .expect("profiled plus evaluation")
                .cost;
            objective.reset();
            let value_minus = objective
                .evaluate(&minus)
                .expect("profiled minus evaluation")
                .cost;
            let finite_difference = (value_plus - value_minus) / (2.0 * h);
            let gap = (analytic[group] - finite_difference).abs();
            let scale = 1.0 + analytic[group].abs().max(finite_difference.abs());
            eprintln!(
                "#2634 profiled FD group {group} ({}): analytic={:.12e}, central_fd={finite_difference:.12e}, gap={gap:.3e}",
                objective.layout.group_keys[group],
                analytic[group],
            );
            assert!(
                gap <= 5.0e-4 * scale,
                "group {group} ({}): profiled analytic gradient {:.9e} != refitted production-value FD {finite_difference:.9e} (gap {gap:.3e}, scale {scale:.3e})",
                objective.layout.group_keys[group],
                analytic[group],
            );
        }
    }

    /// Solve the inner fixed point cleanly from the initial term at `rho`
    /// (rebuilding the whole cache — never freezing it, per the FD-gate rule)
    /// and read off the penalized deviance `D_p = 2·penalized_objective` and the
    /// raw residual sum of squares at that converged inner optimum.
    fn deviance_and_rss(objective: &mut SaeSupportOuterObjective, rho: &Array1<f64>) -> (f64, f64) {
        objective.reset();
        let lambda = objective.layout.expand(rho).expect("expand");
        objective
            .term
            .solve_fixed_point(
                objective.target.view(),
                &lambda,
                &objective.ard_precisions,
                objective.max_inner_iter,
                objective.inner_tolerance,
                objective.trust_radius,
            )
            .expect("inner fixed point");
        let deviance = 2.0
            * objective
                .term
                .penalized_objective(objective.target.view(), &lambda, &objective.ard_precisions)
                .expect("penalized objective");
        let residual = objective
            .term
            .raw_residual(objective.target.view())
            .expect("raw residual");
        let rss = residual.iter().map(|value| value * value).sum::<f64>();
        (deviance, rss)
    }

    /// Decisive oracle for the value↔gradient desync. The outer value feeds the
    /// penalized deviance `D_p` into the Gaussian dispersion term, and the
    /// analytic gradient's dispersion channel is `½·residual_df·energy[g]/D_p`
    /// with `energy[g] = Σ_{k∈g} λ_k β̂ᵀ S_k β̂ = penalty_energy_by_group`. Since
    /// `residual_df` and `D_p` are common factors, value/gradient consistency of
    /// that channel is EXACTLY the envelope identity `d D_p/dρ_g = energy[g]`.
    /// Central-differencing the production `D_p` (with a full clean inner re-solve
    /// at each ρ±h) must reproduce the production `energy[g]` — this is the FD
    /// oracle the SPEC allows in tests. The same test also confirms that the RAW
    /// residual sum of squares does NOT satisfy the identity (its derivative
    /// carries the implicit `H⁻¹` envelope term), so a revert to raw RSS is caught.
    #[test]
    fn support_penalized_deviance_derivative_equals_penalty_energy() {
        let mut objective = build_objective();
        // λ deliberately away from 1 in both groups so the raw-RSS derivative and
        // the penalized-deviance derivative are unmistakably different functions.
        let base = array![0.35_f64.ln(), 2.8_f64.ln()];
        let groups = objective.layout.group_keys.len();

        // Production `energy[g]` at the base inner optimum.
        objective.reset();
        let lambda_base = objective.layout.expand(&base).expect("expand");
        objective
            .term
            .solve_fixed_point(
                objective.target.view(),
                &lambda_base,
                &objective.ard_precisions,
                objective.max_inner_iter,
                objective.inner_tolerance,
                objective.trust_radius,
            )
            .expect("base inner fixed point");
        let energy = objective.penalty_energy_by_group(&lambda_base);
        assert_eq!(energy.len(), groups);
        assert!(
            energy.iter().all(|value| value.is_finite() && *value > 0.0),
            "fixture must exercise strictly positive per-group penalty energy: {energy:?}"
        );

        let h = 1.0e-4;
        let energy_scale = energy
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let mut max_raw_gap = 0.0_f64;
        for g in 0..groups {
            let mut plus = base.clone();
            let mut minus = base.clone();
            plus[g] += h;
            minus[g] -= h;
            let (dev_plus, rss_plus) = deviance_and_rss(&mut objective, &plus);
            let (dev_minus, rss_minus) = deviance_and_rss(&mut objective, &minus);
            let deviance_derivative = (dev_plus - dev_minus) / (2.0 * h);
            let raw_rss_derivative = (rss_plus - rss_minus) / (2.0 * h);

            // (1) Envelope identity: d D_p/dρ_g == energy[g] (value↔gradient match).
            let envelope_gap = (deviance_derivative - energy[g]).abs();
            assert!(
                envelope_gap <= 1.0e-6 * (1.0 + energy[g].abs()),
                "group {g}: penalized-deviance derivative {deviance_derivative:.9e} \
                 disagrees with penalty energy {:.9e} (gap {envelope_gap:.3e})",
                energy[g]
            );

            // (2) Raw RSS is a DIFFERENT function: its derivative must not be the
            // penalty energy, proving the raw-RSS gradient (the fixed defect) desyncs.
            max_raw_gap = max_raw_gap.max((raw_rss_derivative - energy[g]).abs());
        }
        assert!(
            max_raw_gap > 1.0e-2 * energy_scale.max(1.0e-3),
            "raw-RSS derivative must visibly differ from the penalty energy so the \
             desync is caught (max gap {max_raw_gap:.3e}, energy scale {energy_scale:.3e})"
        );
    }

    /// Full-production oracle: the analytic gradient returned by `evaluate` must
    /// match a central difference of the production value `cost`, restricted to
    /// the dispersion channel that the desync corrupted. The joint log-det term is
    /// a fixed-probe Hutchinson estimate (deterministic but not exact), so we
    /// isolate the dispersion channel by subtracting the exact analytic log-det
    /// and penalty-log-det contributions — both computed from the same production
    /// quantities — leaving `½·residual_df·(1+ln(τ·D_p/df))`, whose FD must equal
    /// the analytic `½·residual_df·energy[g]/D_p`.
    #[test]
    fn support_outer_value_dispersion_channel_matches_gradient() {
        let mut objective = build_objective();
        let base = array![0.4_f64.ln(), 2.2_f64.ln()];
        let groups = objective.layout.group_keys.len();

        // residual_df is a ρ-independent constant of the fixture.
        let (_, beta_dim) = objective.beta_layout().expect("beta layout");
        let beta_nullity = beta_dim - objective.spectrum.total_rank;
        let data_dim = objective.term.n_obs() * objective.term.output_dim();
        let residual_df = (data_dim - beta_nullity) as f64;
        assert!(residual_df > 0.0);

        let dispersion_value =
            |objective: &mut SaeSupportOuterObjective, rho: &Array1<f64>| -> f64 {
                let (deviance, _) = deviance_and_rss(objective, rho);
                0.5 * residual_df * (1.0 + (std::f64::consts::TAU * deviance / residual_df).ln())
            };

        // Analytic dispersion-channel gradient from production quantities.
        objective.reset();
        let lambda_base = objective.layout.expand(&base).expect("expand");
        objective
            .term
            .solve_fixed_point(
                objective.target.view(),
                &lambda_base,
                &objective.ard_precisions,
                objective.max_inner_iter,
                objective.inner_tolerance,
                objective.trust_radius,
            )
            .expect("base inner fixed point");
        let deviance_base = 2.0
            * objective
                .term
                .penalized_objective(
                    objective.target.view(),
                    &lambda_base,
                    &objective.ard_precisions,
                )
                .expect("penalized objective");
        let energy = objective.penalty_energy_by_group(&lambda_base);

        let h = 1.0e-4;
        for g in 0..groups {
            let analytic = 0.5 * residual_df * energy[g] / deviance_base;
            let mut plus = base.clone();
            let mut minus = base.clone();
            plus[g] += h;
            minus[g] -= h;
            let fd = (dispersion_value(&mut objective, &plus)
                - dispersion_value(&mut objective, &minus))
                / (2.0 * h);
            assert!(
                (analytic - fd).abs() <= 1.0e-6 * (1.0 + analytic.abs()),
                "group {g}: analytic dispersion gradient {analytic:.9e} != FD {fd:.9e}"
            );
        }
    }

    /// #2576 regression, the blunt one: the criterion must PRODUCE A NUMBER.
    ///
    /// Before this issue it could not, on any host without a CUDA device. The
    /// lane took its factor cache from `solve_arrow_newton_step_with_options`
    /// under `InexactPCG`, which never forms a dense reduced-Schur factor, so
    /// `schur_factor_is_undamped` stayed false, `compute_undamped_arrow_log_det`
    /// returned `None` for `k > 0`, and `arrow_log_det()` refused —
    /// `"support LAML factor cache has no joint log determinant"` on the very
    /// first outer evaluation, before the trace estimator the issue was filed
    /// about was ever reached. The whole grouped-LAML engine was dead code with
    /// no test that would notice, because every existing test isolated one
    /// channel and none called `evaluate`.
    ///
    /// This test calls it. It is also the CPU-only gate: nothing here builds a
    /// device operator, so it exercises the host lane that used to have no
    /// matrix-free `log|S|` at all.
    #[test]
    fn support_outer_evaluate_mints_a_finite_value_and_gradient() {
        let mut objective = build_objective();
        let rho = array![0.4_f64.ln(), 2.2_f64.ln()];
        let evaluation = objective
            .evaluate(&rho)
            .expect("the grouped-LAML criterion must evaluate on a CPU-only host");
        assert!(
            evaluation.cost.is_finite(),
            "criterion value must be finite, got {}",
            evaluation.cost
        );
        assert_eq!(evaluation.gradient.len(), objective.layout.group_keys.len());
        assert!(
            evaluation.gradient.iter().all(|value| value.is_finite()),
            "every smoothing gradient component must be finite, got {:?}",
            evaluation.gradient
        );
        assert!(
            evaluation.fixed_point.recurred,
            "the fixture's inner fixed point must recur so the evaluation is at a \
             stationary inner state"
        );

        // The frozen surrogate is the criterion's identity. Its invariant is
        // stated against a FIXED operator: evaluating the plan twice on one
        // assembled system must be bit-identical, because the probes, the
        // quadrature nodes and the deflation basis are all fixed at the first
        // build. A plan rebuilt per call would make the outer search descend a
        // different function at every point, and the exact directional
        // derivative would be the gradient of something nobody evaluated twice.
        let lambda = objective.layout.expand(&rho).expect("expand");
        let system = objective
            .term
            .assemble_arrow_schur(
                objective.target.view(),
                &lambda,
                &objective.ard_precisions,
            )
            .expect("assemble arrow schur");
        let first = objective
            .evidence_log_det(&system)
            .expect("frozen surrogate")
            .0;
        let second = objective
            .evidence_log_det(&system)
            .expect("frozen surrogate")
            .0;
        assert_eq!(
            first, second,
            "the frozen log|S| surrogate must be bit-reproducible on one operator"
        );

        // A second full `evaluate` at the same ρ re-enters `solve_fixed_point`
        // from the state the first one left, so β̂ moves by a few ulps and the
        // criterion moves with it. That is the inner solve's reproducibility,
        // not the surrogate's, so this limb is a tight relative bound rather
        // than an equality — the equality above already pins the surrogate.
        let again = objective.evaluate(&rho).expect("second evaluation");
        assert!(
            (again.cost - evaluation.cost).abs() <= 1.0e-12 * evaluation.cost.abs().max(1.0),
            "criterion moved between two evaluations at one ρ: {} vs {}",
            evaluation.cost,
            again.cost
        );
        for group in 0..evaluation.gradient.len() {
            let gap = (again.gradient[group] - evaluation.gradient[group]).abs();
            assert!(
                gap <= 1.0e-10 * evaluation.gradient[group].abs().max(1.0),
                "group {group} gradient moved between two evaluations at one ρ: {} vs {}",
                evaluation.gradient[group],
                again.gradient[group]
            );
        }
    }

    /// #2576's decisive oracle for the channel that REPLACED the Hutchinson
    /// trace estimator.
    ///
    /// The lane no longer estimates `tr(H⁻¹ ∂H/∂ρ_g)` with its own probe family.
    /// It contracts the log-determinant surrogate's own derivative bundle
    /// against `∂S/∂ρ_g`, on the claim that
    ///
    /// Only `H_ββ` carries an explicit smoothing coordinate, so at fixed
    /// inner state `∂S/∂ρ_g = Σ_{k∈g} λ_k (S_k ⊗ I_P)` —
    /// `schur_derivative_matvec`.
    ///
    /// Both claims are tested at once by central-differencing the production
    /// `evidence_log_det` — the coordinate-profiled `log|S|` — through λ alone
    /// at a FROZEN inner state, against the analytic contraction. A wrong
    /// `∂S/∂ρ_g` (missing the `⊗ I_P`, wrong group mask, `λ` instead of
    /// `∂λ/∂ρ = λ`) fails; a plan rebuilt between the FD legs fails,
    /// because then the two legs would be values of two different functions.
    #[test]
    fn support_outer_logdet_gradient_matches_fd_of_its_own_surrogate() {
        let mut objective = build_objective();
        let base = array![0.4_f64.ln(), 2.2_f64.ln()];
        let groups = objective.layout.group_keys.len();

        // One clean inner solve, then FREEZE: the log-det channel's contract is
        // the partial derivative through λ at a fixed inner state, which is
        // exactly what the LAML gradient's trace term is.
        objective.reset();
        let lambda_base = objective.layout.expand(&base).expect("expand");
        objective
            .term
            .solve_fixed_point(
                objective.target.view(),
                &lambda_base,
                &objective.ard_precisions,
                objective.max_inner_iter,
                objective.inner_tolerance,
                objective.trust_radius,
            )
            .expect("base inner fixed point");

        let assemble = |objective: &SaeSupportOuterObjective, rho: &Array1<f64>| {
            let lambda = objective.layout.expand(rho).expect("expand");
            let system = objective
                .term
                .assemble_arrow_schur(
                    objective.target.view(),
                    &lambda,
                    &objective.ard_precisions,
                )
                .expect("assemble arrow schur");
            (system, lambda)
        };

        let (base_system, _) = assemble(&objective, &base);
        let (base_logdet, bundle) = objective
            .evidence_log_det(&base_system)
            .expect("base evidence log-determinant");
        assert!(base_logdet.is_finite());

        let (beta_offsets, beta_dim) = objective.beta_layout().expect("beta layout");
        let analytic = (0..groups)
            .map(|group| {
                bundle
                    .directional_derivative(&|vector: ArrayView1<f64>| {
                        objective.schur_derivative_matvec(
                            group,
                            &lambda_base,
                            &beta_offsets,
                            beta_dim,
                            vector,
                        )
                    })
                    .expect("surrogate directional derivative")
            })
            .collect::<Vec<_>>();

        let h = 1.0e-5;
        for group in 0..groups {
            let mut plus = base.clone();
            let mut minus = base.clone();
            plus[group] += h;
            minus[group] -= h;
            let (system_plus, _) = assemble(&objective, &plus);
            let (system_minus, _) = assemble(&objective, &minus);
            let value_plus = objective
                .evidence_log_det(&system_plus)
                .expect("perturbed evidence log-determinant")
                .0;
            let value_minus = objective
                .evidence_log_det(&system_minus)
                .expect("perturbed evidence log-determinant")
                .0;
            let fd = (value_plus - value_minus) / (2.0 * h);
            let gap = (analytic[group] - fd).abs();
            assert!(
                gap <= 1.0e-5 * (1.0 + fd.abs()),
                "group {group} ({}): analytic ∂log|S|/∂ρ = {:.9e} disagrees with the \
                 central difference of the SAME frozen surrogate {fd:.9e} (gap {gap:.3e})",
                objective.layout.group_keys[group],
                analytic[group],
            );
            // A positive-semidefinite ∂S with a strictly positive-rank penalty
            // block can only INCREASE log|S|; a sign flip in the contraction is
            // the failure this catches independently of the FD magnitude.
            assert!(
                analytic[group] > 0.0,
                "group {group}: an SPD ∂S/∂ρ must raise log|S|, got {:.9e}",
                analytic[group]
            );
        }
    }
}

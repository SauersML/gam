use faer::Side;
use gam_linalg::faer_ndarray::{FaerCholesky, FaerEigh};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use std::fmt;

const DEFAULT_MAX_ITER: usize = 30;
const DEFAULT_TOP_K: usize = 1;
const DEFAULT_TEMPERATURE: f64 = 0.25;
const DEFAULT_CODE_RIDGE: f64 = 1.0e-8;
const DEFAULT_TOLERANCE: f64 = 1.0e-7;
const INACTIVE_LAMBDA: f64 = 1.0e30;
const MIN_NORM2: f64 = 1.0e-24;

// --- Safeguarded geometric acceleration of the atom trajectory (#2372). ---------
// The alternating atom-sweep ↔ reroute map converges LINEARLY: near the solution
// the atom matrix moves along one dominant slow mode `Aₖ ≈ A* + c·rᵏ·V` with ratio
// `r` set by the coupling of the atoms that share rows. On the planted 2-sparse
// ring fixture `r ≈ 0.9985`, so reaching a `1e-9` fixed-point residual takes ~2500
// plain sweeps — orders of magnitude past any reasonable `max_iter`, which is why
// the fixed-point contract never closed. When two consecutive atom steps are
// collinear (`cos ≥ GEOM_COS_MIN`) and contracting (`r ∈ (GEOM_R_MIN, 1)`) the
// sequence is in that single-mode geometric tail, and the sum of the remaining
// steps is `Δₖ·r/(1−r)`. Jumping there collapses the tail in one step. The jump is
// a SAFEGUARDED proposal: it is adopted only when its rerouted EV strictly beats
// the plain step's, so it can never degrade the fit (monotonicity preserved) — the
// only risk of a bad `r` estimate is a rejected proposal, never a worse model.
const GEOM_R_MIN: f64 = 0.1;
const GEOM_R_EPS: f64 = 1.0e-12;
const GEOM_COS_MIN: f64 = 0.9;
const GEOM_FACTOR_CAP: f64 = 1.0e6;
// Line-search ladder over the extrapolation step length: the spiral's straight-line
// tangent is closest to the fixed point at an interior factor, so probe a geometric
// ladder from `BASE` up to the geometric-sum bound `r/(1−r)` and keep the best.
const GEOM_LADDER_BASE: f64 = 1.3;
const GEOM_LADDER_STEP: f64 = 1.3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearDictionaryAssignment {
    TopK,
    Softmax,
}

impl LinearDictionaryAssignment {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "top_k" | "topk" | "hard" => Ok(Self::TopK),
            "softmax" | "soft" => Ok(Self::Softmax),
            other => Err(format!(
                "linear dictionary assignment must be 'top_k' or 'softmax'; got {other:?}"
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TopK => "top_k",
            Self::Softmax => "softmax",
        }
    }
}

/// Typed failure from [`fit_linear_dictionary`].
///
/// In particular, [`LinearDictionaryError::NonConvergence`] preserves the
/// numerical certificate that prevented the final iterate from becoming a
/// [`LinearDictionaryFit`].
#[derive(Clone, Debug, PartialEq)]
pub enum LinearDictionaryError {
    InvalidInput {
        reason: String,
    },
    NumericalFailure {
        reason: String,
    },
    NonConvergence {
        iterations: usize,
        explained_variance: f64,
        ev_residual: f64,
        routing_residual: f64,
        accepted_births: usize,
        tolerance: f64,
    },
}

impl LinearDictionaryError {
    fn invalid_input(reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            reason: reason.into(),
        }
    }
}

impl From<String> for LinearDictionaryError {
    fn from(reason: String) -> Self {
        Self::NumericalFailure { reason }
    }
}

impl fmt::Display for LinearDictionaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { reason } | Self::NumericalFailure { reason } => {
                f.write_str(reason)
            }
            Self::NonConvergence {
                iterations,
                explained_variance,
                ev_residual,
                routing_residual,
                accepted_births,
                tolerance,
            } => write!(
                f,
                "linear_dictionary_fit did not converge: {iterations} coordinate-descent sweeps \
                 ended at EV {explained_variance:.6} with canonical EV residual \
                 {ev_residual:.3e}, reroute residual {routing_residual:.3e}, and \
                 {accepted_births} accepted dead-atom births (tolerance {tolerance:.3e}); a \
                 non-converged iterate is not a model"
            ),
        }
    }
}

impl std::error::Error for LinearDictionaryError {}

#[derive(Clone, Debug)]
pub struct LinearDictionaryConfig {
    pub n_atoms: usize,
    pub max_iter: usize,
    pub top_k: usize,
    pub assignment: LinearDictionaryAssignment,
    pub temperature: f64,
    pub code_ridge: f64,
    pub tolerance: f64,
    /// K=1 lane only. When `false` (default) the rank-one lane takes the leading
    /// eigenvector of the UNCENTERED second-moment matrix `XᵀX` (byte-identical to
    /// historical behavior), which is only a true centered-PCA ceiling when `x` is
    /// already mean-centered. When `true` the lane subtracts the column mean, takes
    /// the leading eigenvector of the CENTERED second-moment matrix, fits the
    /// rank-1 code on the centered data, and adds the mean back — so the reported
    /// EV (measured against the crate's centered denominator) is a genuine
    /// centered-PCA ceiling even on uncentered input. Because the reconstruction is
    /// then affine (mean + rank-1), the returned `fitted` INCLUDES the mean and is
    /// NOT equal to `assignments.dot(atoms)` in this mode.
    pub center_rank_one: bool,
}

impl LinearDictionaryConfig {
    pub fn new(n_atoms: usize) -> Self {
        Self {
            n_atoms,
            ..Self::default()
        }
    }
}

impl Default for LinearDictionaryConfig {
    fn default() -> Self {
        Self {
            n_atoms: 1,
            max_iter: DEFAULT_MAX_ITER,
            top_k: DEFAULT_TOP_K,
            assignment: LinearDictionaryAssignment::TopK,
            temperature: DEFAULT_TEMPERATURE,
            code_ridge: DEFAULT_CODE_RIDGE,
            tolerance: DEFAULT_TOLERANCE,
            center_rank_one: false,
        }
    }
}

/// A converged linear-dictionary model. This struct exists ONLY for a certified
/// fit (SPEC 20): the coordinate-descent solver returns it exclusively when the
/// EV-plateau convergence test fired, and non-convergence is an error carrying
/// its evidence (sweeps run, last EV, last improvement, tolerance) — never a
/// degraded/best-effort fit. The closed-form K=1 lanes are converged by
/// construction (a single exact eigensolve).
#[derive(Clone, Debug)]
pub struct LinearDictionaryFit {
    pub atoms: Array2<f64>,
    pub assignments: Array2<f64>,
    pub fitted: Array2<f64>,
    pub lambdas: Array1<f64>,
    pub reml_scores: Array1<f64>,
    pub explained_variance: f64,
    pub iterations: usize,
    pub convergence: LinearDictionaryConvergence,
    pub assignment: LinearDictionaryAssignment,
    pub top_k: usize,
}

/// Fixed-point evidence attached to every converged [`LinearDictionaryFit`].
///
/// The two residuals certify the exact canonical routing stored in the model:
/// `ev_residual` compares consecutive full atom-sweep + reroute states, while
/// `routing_residual` compares the final atom-sweep state with a fresh global
/// routing against those same final atoms. A fit is constructed only when both
/// are below `tolerance` and no proposed dead-atom birth entered the routing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearDictionaryConvergence {
    pub ev_residual: f64,
    pub routing_residual: f64,
    pub accepted_births: usize,
    pub tolerance: f64,
}

/// Fit a linear (flat) dictionary by block coordinate descent: each sweep
/// re-routes rows to atoms (the assignment step) and then refines every atom and
/// its assignment column by a penalized least-squares update against the residual.
///
/// CONTRACT: this is a heuristic coordinate-descent dictionary learner, not a
/// globally-optimal linear SAE. Every sweep closes with a fresh global routing
/// against the updated atoms. Convergence is certified on that exact canonical
/// state: both its change from the previous canonical state and its discrepancy
/// from the just-completed atom sweep must be below tolerance, and no proposed
/// dead-atom birth may enter the routing. The returned assignments, fitted values,
/// EV, and diagnostics are therefore the state that passed the certificate; no
/// post-certificate reroute or best-effort adoption occurs.
pub fn fit_linear_dictionary(
    x: ArrayView2<'_, f64>,
    config: &LinearDictionaryConfig,
) -> Result<LinearDictionaryFit, LinearDictionaryError> {
    validate_inputs(x, config)?;
    if config.n_atoms == 1 {
        return fit_rank_one_pca_lane(x, config);
    }
    fit_multi_atom_dictionary(x, config)
}

/// One plain alternating step: a full per-atom penalized-LS sweep followed by a
/// fresh global reroute, committing the rerouted assignment and fitted values into
/// `assignments`/`fitted`. Reseeded atoms (#1500) whose reroute column stays empty
/// are the rejected redundant proposals — they are zeroed here so a held-out
/// transform cannot expose a direction absent from the certified model; a reseed
/// the reroute actually uses counts as an `accepted_births`. Returns the sweep EV
/// (pre-reroute), the rerouted EV, and the accepted-birth count so the caller can
/// form the fixed-point residuals and drive the geometric acceleration.
fn plain_atom_step(
    x: ArrayView2<'_, f64>,
    atoms: &mut Array2<f64>,
    assignments: &mut Array2<f64>,
    fitted: &mut Array2<f64>,
    lambdas: &mut Array1<f64>,
    reml_scores: &mut Array1<f64>,
    top_k: usize,
    config: &LinearDictionaryConfig,
) -> Result<(f64, f64, usize), LinearDictionaryError> {
    let n_atoms = atoms.nrows();
    let mut reseeded = vec![false; n_atoms];
    for atom_idx in 0..n_atoms {
        reseeded[atom_idx] = fit_one_atom_penalized_ls(
            x,
            atoms,
            assignments,
            fitted,
            lambdas,
            reml_scores,
            atom_idx,
            config.code_ridge,
        )?;
    }

    let sweep_ev = explained_variance(x, fitted.view());
    let rerouted = reroute_against_atoms(x, atoms.view(), top_k, config)?;
    let rerouted_fitted = rerouted.dot(&*atoms);
    let rerouted_ev = explained_variance(x, rerouted_fitted.view());

    let mut accepted_births = 0usize;
    for atom_idx in 0..n_atoms {
        if !reseeded[atom_idx] {
            continue;
        }
        let accepted = rerouted
            .column(atom_idx)
            .iter()
            .any(|coefficient| *coefficient != 0.0);
        if accepted {
            accepted_births += 1;
        } else {
            atoms.row_mut(atom_idx).fill(0.0);
            lambdas[atom_idx] = INACTIVE_LAMBDA;
            reml_scores[atom_idx] = 0.0;
        }
    }

    *assignments = rerouted;
    *fitted = rerouted_fitted;
    Ok((sweep_ev, rerouted_ev, accepted_births))
}

fn fit_multi_atom_dictionary(
    x: ArrayView2<'_, f64>,
    config: &LinearDictionaryConfig,
) -> Result<LinearDictionaryFit, LinearDictionaryError> {
    let top_k = config.top_k.min(config.n_atoms).max(1);
    let mut atoms = initialize_atoms(x, config.n_atoms);
    let mut assignments = reroute_against_atoms(x, atoms.view(), top_k, config)?;
    let mut fitted = assignments.dot(&atoms);
    let mut lambdas = Array1::<f64>::from_elem(config.n_atoms, INACTIVE_LAMBDA);
    let mut reml_scores = Array1::<f64>::zeros(config.n_atoms);
    let mut previous_ev = explained_variance(x, fitted.view());
    let mut completed_iterations = 0usize;
    let mut last_ev = previous_ev;
    let mut ev_residual = f64::INFINITY;
    let mut routing_residual = f64::INFINITY;
    let mut accepted_births = 0usize;
    // History of the atom trajectory for the safeguarded geometric acceleration:
    // `prev_atoms` is the previous iteration's canonical dictionary and `prev_delta`
    // the previous atom step, so a collinear pair of steps exposes the dominant
    // slow mode to extrapolate along.
    let mut prev_atoms: Option<Array2<f64>> = None;
    let mut prev_delta: Option<Array2<f64>> = None;

    for iteration in 0..config.max_iter {
        let (mut sweep_ev, mut rerouted_ev, mut births) = plain_atom_step(
            x,
            &mut atoms,
            &mut assignments,
            &mut fitted,
            &mut lambdas,
            &mut reml_scores,
            top_k,
            config,
        )?;

        // Safeguarded geometric acceleration. `atoms` is now the canonical
        // dictionary this sweep converged to; its step from the previous one is
        // `this_delta`. When that step is collinear with the last (and the support
        // is settled — no births) the trajectory is in its dominant-mode geometric
        // tail, and a line-searched extrapolation along it lands near the fixed
        // point. A jump is a big non-plain step that breaks the collinear pair the
        // next jump needs, so after adopting one we REBUILD the geometric history
        // with two more plain steps INLINE — keeping the acceleration firing every
        // outer iteration rather than stalling two sweeps between jumps. The jump is
        // adopted only when it strictly beats the plain EV, so it never degrades the
        // fit; a broken sequence (fresh births, a support flip, a non-contracting
        // ratio) is simply left to plain alternation.
        let this_delta = prev_atoms.as_ref().map(|previous| &atoms - previous);
        let mut jumped = false;
        if births == 0 {
            if let (Some(delta), Some(previous_delta)) =
                (this_delta.as_ref(), prev_delta.as_ref())
            {
                if let Some((cand_atoms, cand_route, cand_fitted)) =
                    try_geometric_extrapolation(
                        atoms.view(),
                        delta.view(),
                        previous_delta.view(),
                        x,
                        top_k,
                        config,
                        rerouted_ev,
                    )
                {
                    atoms = cand_atoms;
                    assignments = cand_route;
                    fitted = cand_fitted;
                    jumped = true;
                    // Rebuild the geometric history from the jumped point with two
                    // inline plain steps so the next outer iteration can jump again.
                    let (_, _, rebuild_births_1) = plain_atom_step(
                        x,
                        &mut atoms,
                        &mut assignments,
                        &mut fitted,
                        &mut lambdas,
                        &mut reml_scores,
                        top_k,
                        config,
                    )?;
                    let rebuilt_prev = atoms.clone();
                    let (rebuild_sweep_ev, rebuild_ev, rebuild_births_2) = plain_atom_step(
                        x,
                        &mut atoms,
                        &mut assignments,
                        &mut fitted,
                        &mut lambdas,
                        &mut reml_scores,
                        top_k,
                        config,
                    )?;
                    prev_delta = Some(&atoms - &rebuilt_prev);
                    prev_atoms = Some(atoms.clone());
                    sweep_ev = rebuild_sweep_ev;
                    rerouted_ev = rebuild_ev;
                    births = rebuild_births_1.max(rebuild_births_2);
                }
            }
        }
        if !jumped {
            prev_atoms = Some(atoms.clone());
            prev_delta = this_delta;
        }
        accepted_births = births;

        completed_iterations = iteration + 1;
        ev_residual = (rerouted_ev - previous_ev).abs();
        routing_residual = (rerouted_ev - sweep_ev).abs();
        last_ev = rerouted_ev;

        if accepted_births == 0
            && ev_residual <= config.tolerance
            && routing_residual <= config.tolerance
            // A plateau is a SEQUENCE property: `ev_residual` compares this
            // sweep's canonical EV against the PREVIOUS one, and on sweep 0
            // "previous" is the initialization — a single agreeing pair is one
            // data point, not a plateau (an initialization that happens to sit
            // at the fixed point would self-certify without the solver ever
            // demonstrating stability). Two completed sweeps minimum.
            && iteration >= 1
        {
            // The per-atom update recorded scores at intermediate Gauss-Seidel
            // states. Recompute every active score from the exact canonical state
            // that passed the fixed-point certificate.
            let final_score =
                penalized_reconstruction_loss(x, fitted.view(), config.code_ridge, atoms.view());
            for atom_idx in 0..config.n_atoms {
                if atoms.row(atom_idx).dot(&atoms.row(atom_idx)) > MIN_NORM2 {
                    reml_scores[atom_idx] = final_score;
                }
            }
            return Ok(LinearDictionaryFit {
                atoms,
                assignments,
                fitted,
                lambdas,
                reml_scores,
                explained_variance: last_ev,
                iterations: completed_iterations,
                convergence: LinearDictionaryConvergence {
                    ev_residual,
                    routing_residual,
                    accepted_births,
                    tolerance: config.tolerance,
                },
                assignment: config.assignment,
                top_k,
            });
        }
        previous_ev = rerouted_ev;
    }

    // SPEC 20: the final canonical work state failed at least one fixed-point
    // condition. Preserve its numerical evidence in a typed error; never mint a
    // degraded/best-effort model from it.
    Err(LinearDictionaryError::NonConvergence {
        iterations: completed_iterations,
        explained_variance: last_ev,
        ev_residual,
        routing_residual,
        accepted_births,
        tolerance: config.tolerance,
    })
}

/// Safeguarded extrapolation of the atom trajectory (#2372). Given the current
/// dictionary `atoms = Aₖ`, this sweep's step `delta = Aₖ − Aₖ₋₁`, and the previous
/// step `prev_delta = Aₖ₋₁ − Aₖ₋₂`, decide whether the trajectory is in its slow
/// single-mode tail and, if so, propose a dictionary further along that mode.
///
/// The atom trajectory does not decay along a fixed straight line — it SPIRALS: the
/// dominant mode is a nearly-real eigenvalue `λ = ρ·e^{iθ}` with `ρ ≈ 0.9985` and a
/// tiny per-step angle `θ`, so consecutive steps are collinear to six digits
/// (`cos ≈ 1`) yet accumulate a large arc over the `r/(1−r) ≈ 1600` steps that
/// remain. Extrapolating the full geometric sum in a straight line therefore
/// OVERSHOOTS the limit badly (it leaves the sphere-product the atoms live on). So
/// instead of one fixed jump we LINE-SEARCH the step length: walk a geometric ladder
/// of factors up to the geometric-sum bound `r/(1−r)`, reroute each candidate, and
/// keep the one with the highest EV that strictly beats the plain step. The tangent
/// from `Aₖ` heads toward the limit and only later curves away, so a well-defined
/// interior factor gets closest — the ladder finds it, and the strict-improvement
/// guard means a bad estimate costs at most a rejected proposal, never a worse fit.
fn try_geometric_extrapolation(
    atoms: ArrayView2<'_, f64>,
    delta: ArrayView2<'_, f64>,
    prev_delta: ArrayView2<'_, f64>,
    x: ArrayView2<'_, f64>,
    top_k: usize,
    config: &LinearDictionaryConfig,
    current_ev: f64,
) -> Option<(Array2<f64>, Array2<f64>, Array2<f64>)> {
    let cross: f64 = delta.iter().zip(prev_delta.iter()).map(|(a, b)| a * b).sum();
    let prev_norm2: f64 = prev_delta.iter().map(|v| v * v).sum();
    let delta_norm2: f64 = delta.iter().map(|v| v * v).sum();
    if !(prev_norm2 > 0.0 && delta_norm2 > 0.0) {
        return None;
    }
    let ratio = cross / prev_norm2;
    if !(ratio > GEOM_R_MIN && ratio < 1.0 - GEOM_R_EPS) {
        return None;
    }
    // Collinearity of the two steps — the single-dominant-mode assumption. A
    // support flip or a two-mode transient makes the steps non-parallel; skip it.
    let cosine = cross / (delta_norm2.sqrt() * prev_norm2.sqrt());
    if !(cosine > GEOM_COS_MIN) {
        return None;
    }
    let max_factor = (ratio / (1.0 - ratio)).min(GEOM_FACTOR_CAP);
    if !(max_factor.is_finite() && max_factor > 1.0) {
        return None;
    }
    let delta_owned = delta.to_owned();
    let atoms_owned = atoms.to_owned();
    let mut best: Option<(Array2<f64>, Array2<f64>, Array2<f64>)> = None;
    let mut best_ev = current_ev;
    let mut factor = GEOM_LADDER_BASE;
    loop {
        let mut candidate = &atoms_owned + &(factor * &delta_owned);
        for atom_idx in 0..candidate.nrows() {
            normalize_row(candidate.slice_mut(s![atom_idx, ..]));
        }
        if let Ok(route) = reroute_against_atoms(x, candidate.view(), top_k, config) {
            let fitted = route.dot(&candidate);
            let ev = explained_variance(x, fitted.view());
            if ev > best_ev {
                best_ev = ev;
                best = Some((candidate, route, fitted));
            }
        }
        if factor >= max_factor {
            break;
        }
        factor = (factor * GEOM_LADDER_STEP).min(max_factor);
    }
    best
}

/// Fresh global routing of every row against `atoms` using the configured
/// assignment rule. This is the single source of truth shared by the
/// coordinate-descent assignment step and the post-loop final reroute, so both
/// route identically and the reroute is a true global re-assignment against the
/// final atoms.
fn reroute_against_atoms(
    x: ArrayView2<'_, f64>,
    atoms: ArrayView2<'_, f64>,
    top_k: usize,
    config: &LinearDictionaryConfig,
) -> Result<Array2<f64>, String> {
    match config.assignment {
        LinearDictionaryAssignment::TopK => top_k_assignments(x, atoms, top_k, config.code_ridge),
        LinearDictionaryAssignment::Softmax => {
            softmax_assignments(x, atoms, top_k, config.temperature, config.code_ridge)
        }
    }
}

fn validate_inputs(
    x: ArrayView2<'_, f64>,
    config: &LinearDictionaryConfig,
) -> Result<(), LinearDictionaryError> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(LinearDictionaryError::invalid_input(
            "linear_dictionary_fit requires a non-empty 2-D matrix",
        ));
    }
    if !x.iter().all(|value| value.is_finite()) {
        return Err(LinearDictionaryError::invalid_input(
            "linear_dictionary_fit input must be finite",
        ));
    }
    if config.n_atoms == 0 {
        return Err(LinearDictionaryError::invalid_input(
            "linear_dictionary_fit requires K >= 1",
        ));
    }
    if config.max_iter == 0 {
        return Err(LinearDictionaryError::invalid_input(
            "linear_dictionary_fit requires max_iter >= 1",
        ));
    }
    if config.top_k == 0 || config.top_k > config.n_atoms {
        return Err(LinearDictionaryError::invalid_input(format!(
            "linear_dictionary_fit top_k must be in [1, K={}]; got {}",
            config.n_atoms, config.top_k
        )));
    }
    if !(config.temperature.is_finite() && config.temperature > 0.0) {
        return Err(LinearDictionaryError::invalid_input(format!(
            "linear_dictionary_fit temperature must be finite and positive; got {}",
            config.temperature
        )));
    }
    if !(config.code_ridge.is_finite() && config.code_ridge > 0.0) {
        return Err(LinearDictionaryError::invalid_input(format!(
            "linear_dictionary_fit code_ridge must be finite and positive; got {}",
            config.code_ridge
        )));
    }
    if !(config.tolerance.is_finite() && config.tolerance >= 0.0) {
        return Err(LinearDictionaryError::invalid_input(format!(
            "linear_dictionary_fit tolerance must be finite and non-negative; got {}",
            config.tolerance
        )));
    }
    Ok(())
}

/// K=1 closed-form lane.
///
/// Default (`config.center_rank_one == false`): the leading eigenvector of the
/// UNCENTERED second-moment matrix `XᵀX`. This is only a true centered-PCA ceiling
/// when `x` is already mean-centered upstream; the `explained_variance` denominator
/// IS centered, so on uncentered input the leading `XᵀX` eigenvector can absorb the
/// mean direction and this lane is a second-moment rank-1 fit rather than the
/// centered principal component. This branch is byte-identical to historical
/// behavior.
///
/// Centered (`config.center_rank_one == true`): delegates to
/// [`fit_rank_one_centered_lane`], which subtracts the column mean, takes the
/// leading eigenvector of the CENTERED second-moment matrix, and adds the mean
/// back, so the reported EV is a genuine centered-PCA ceiling even on uncentered
/// input. See that function and [`rank_one_centered_pca_ceiling`] for details.
fn fit_rank_one_pca_lane(
    x: ArrayView2<'_, f64>,
    config: &LinearDictionaryConfig,
) -> Result<LinearDictionaryFit, LinearDictionaryError> {
    if config.center_rank_one {
        return fit_rank_one_centered_lane(x, config);
    }
    let covariance = x.t().dot(&x);
    let (evals, evecs) = covariance
        .eigh(Side::Lower)
        .map_err(|err| format!("linear_dictionary_fit PCA eigensolve failed: {err}"))?;
    let last = evals.len() - 1;
    let mut atom = evecs.column(last).to_owned();
    orient_vector(&mut atom);
    let mut assignments = Array2::<f64>::zeros((x.nrows(), 1));
    for row in 0..x.nrows() {
        assignments[[row, 0]] = x.row(row).dot(&atom) / (1.0 + config.code_ridge);
    }
    let mut atoms = atom.insert_axis(Axis(0)).to_owned();
    normalize_atom_and_assignments(&mut atoms, &mut assignments, 0);
    let fitted = assignments.dot(&atoms);
    let score = penalized_reconstruction_loss(x, fitted.view(), config.code_ridge, atoms.view());
    Ok(LinearDictionaryFit {
        atoms,
        assignments,
        fitted: fitted.clone(),
        lambdas: Array1::from_elem(1, config.code_ridge),
        reml_scores: Array1::from_elem(1, score),
        explained_variance: explained_variance(x, fitted.view()),
        iterations: 1.min(config.max_iter),
        convergence: LinearDictionaryConvergence {
            ev_residual: 0.0,
            routing_residual: 0.0,
            accepted_births: 0,
            tolerance: config.tolerance,
        },
        assignment: config.assignment,
        top_k: 1,
    })
}

/// Centered K=1 lane (`config.center_rank_one == true`): a genuine centered-PCA
/// ceiling. Builds a full [`LinearDictionaryFit`] from the shared centered
/// components — `atoms` is the unit-norm centered principal direction,
/// `assignments` are the centered rank-1 codes, and `fitted` is the AFFINE
/// reconstruction `mean + code·atom`, so `explained_variance` (centered
/// denominator) is a true ceiling. Because the reconstruction is affine, `fitted`
/// INCLUDES the mean and is NOT `assignments.dot(atoms)` in this mode.
fn fit_rank_one_centered_lane(
    x: ArrayView2<'_, f64>,
    config: &LinearDictionaryConfig,
) -> Result<LinearDictionaryFit, LinearDictionaryError> {
    let CenteredRankOne {
        atom,
        codes,
        fitted,
        explained_variance: ev,
    } = centered_rank_one_components(x, config.code_ridge)?;
    let atoms = atom.insert_axis(Axis(0)).to_owned();
    let assignments = codes.insert_axis(Axis(1)).to_owned();
    let score = penalized_reconstruction_loss(x, fitted.view(), config.code_ridge, atoms.view());
    Ok(LinearDictionaryFit {
        atoms,
        assignments,
        fitted,
        lambdas: Array1::from_elem(1, config.code_ridge),
        reml_scores: Array1::from_elem(1, score),
        explained_variance: ev,
        iterations: 1.min(config.max_iter),
        convergence: LinearDictionaryConvergence {
            ev_residual: 0.0,
            routing_residual: 0.0,
            accepted_births: 0,
            tolerance: config.tolerance,
        },
        assignment: config.assignment,
        top_k: 1,
    })
}

/// Shared components of the centered rank-1 fit, so the public ceiling helper and
/// the centered K=1 lane compute exactly the same principal direction / codes.
struct CenteredRankOne {
    /// Unit-norm centered principal direction (length `p`).
    atom: Array1<f64>,
    /// Centered rank-1 codes with the ridge shrink applied (length `n`).
    codes: Array1<f64>,
    /// Affine reconstruction `mean + code·atom` (shape `n × p`).
    fitted: Array2<f64>,
    /// EV of `fitted` against the crate's centered denominator.
    explained_variance: f64,
}

fn centered_rank_one_components(
    x: ArrayView2<'_, f64>,
    code_ridge: f64,
) -> Result<CenteredRankOne, String> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err("rank_one_centered_pca_ceiling requires a non-empty 2-D matrix".to_string());
    }
    if !(code_ridge.is_finite() && code_ridge > 0.0) {
        return Err(format!(
            "rank_one_centered_pca_ceiling code_ridge must be finite and positive; got {code_ridge}"
        ));
    }
    let means = x.mean_axis(Axis(0)).expect("non-empty input has means");
    let centered = &x.to_owned() - &means;
    let covariance = centered.t().dot(&centered);
    let (evals, evecs) = covariance
        .eigh(Side::Lower)
        .map_err(|err| format!("rank_one_centered_pca_ceiling eigensolve failed: {err}"))?;
    let last = evals.len() - 1;
    let mut atom = evecs.column(last).to_owned();
    orient_vector(&mut atom);
    let shrink = 1.0 / (1.0 + code_ridge);
    let mut codes = Array1::<f64>::zeros(x.nrows());
    let mut fitted = Array2::<f64>::zeros(x.dim());
    for row in 0..x.nrows() {
        let code = centered.row(row).dot(&atom) * shrink;
        codes[row] = code;
        for col in 0..x.ncols() {
            fitted[[row, col]] = means[col] + code * atom[col];
        }
    }
    let ev = explained_variance(x, fitted.view());
    Ok(CenteredRankOne {
        atom,
        codes,
        fitted,
        explained_variance: ev,
    })
}

/// Centered rank-1 PCA ceiling for the K=1 lane, exposed for callers that want the
/// ceiling reconstruction/EV directly. Subtracts the column means, takes the
/// leading eigenvector of the CENTERED second-moment matrix, fits the rank-1 code
/// on the centered data with the same ridge shrink the uncentered lane uses, then
/// adds the mean back so the reconstruction lives in the original space. Returns
/// `(fitted, explained_variance)`; the EV is measured against the same centered
/// denominator as the rest of the crate, so it is directly comparable to (and an
/// upper bound on) the uncentered lane's EV. Prefer setting
/// `LinearDictionaryConfig::center_rank_one = true` to route the K=1 lane through
/// this computation as part of a full [`LinearDictionaryFit`].
pub fn rank_one_centered_pca_ceiling(
    x: ArrayView2<'_, f64>,
    code_ridge: f64,
) -> Result<(Array2<f64>, f64), String> {
    let components = centered_rank_one_components(x, code_ridge)?;
    Ok((components.fitted, components.explained_variance))
}

fn initialize_atoms(x: ArrayView2<'_, f64>, n_atoms: usize) -> Array2<f64> {
    let mut atoms = Array2::<f64>::zeros((n_atoms, x.ncols()));
    let first = max_norm_row(x);
    atoms.row_mut(0).assign(&x.row(first));
    normalize_row(atoms.slice_mut(s![0, ..]));
    let mut min_dist2 = Array1::<f64>::from_elem(x.nrows(), f64::INFINITY);

    for atom_idx in 1..n_atoms {
        let prev = atoms.row(atom_idx - 1);
        for row in 0..x.nrows() {
            let dist2 = squared_distance(x.row(row), prev);
            if dist2 < min_dist2[row] {
                min_dist2[row] = dist2;
            }
        }
        let chosen = if atom_idx < x.nrows() {
            max_index(min_dist2.view())
        } else {
            atom_idx % x.nrows()
        };
        atoms.row_mut(atom_idx).assign(&x.row(chosen));
        normalize_row(atoms.slice_mut(s![atom_idx, ..]));
    }
    atoms
}

fn fit_one_atom_penalized_ls(
    x: ArrayView2<'_, f64>,
    atoms: &mut Array2<f64>,
    assignments: &mut Array2<f64>,
    fitted: &mut Array2<f64>,
    lambdas: &mut Array1<f64>,
    reml_scores: &mut Array1<f64>,
    atom_idx: usize,
    atom_ridge: f64,
) -> Result<bool, String> {
    let code = assignments.column(atom_idx).to_owned();
    let code_norm2 = code.dot(&code);
    if code_norm2 <= MIN_NORM2 {
        // #1500: this atom's cluster is EMPTY (no rows routed to it by the
        // assignment step). Zeroing it here made the atom permanently DEAD — a
        // zero atom has zero similarity to every row, so `top_k_assignments`
        // never routes anything back to it, the dictionary collapses to < K live
        // atoms, and it under-explains variance even when the data is exactly K
        // rank-1 atoms a K-atom dictionary could reconstruct perfectly. Instead
        // RE-SEED the atom into the worst-currently-reconstructed direction (the
        // standard k-means empty-cluster cure): point it at the largest-residual
        // row's UNEXPLAINED component so the next assignment sweep can route that
        // row's cluster to it and revive it. Returns `true` so the outer loop
        // suppresses convergence this iteration (the revived atom has no code
        // yet, so EV is momentarily flat — converging now would strand it).
        let mut worst_row = 0usize;
        let mut worst_res2 = -1.0_f64;
        for row in 0..x.nrows() {
            let mut res2 = 0.0_f64;
            for col in 0..x.ncols() {
                let d = x[[row, col]] - fitted[[row, col]];
                res2 += d * d;
            }
            if res2 > worst_res2 {
                worst_res2 = res2;
                worst_row = row;
            }
        }
        if worst_res2 <= MIN_NORM2 {
            // Every row is already fully reconstructed by the other atoms: there
            // is no unexplained direction to seed, so this atom is genuinely
            // redundant capacity. Leave it inactive (this is not the bug).
            atoms.row_mut(atom_idx).fill(0.0);
            lambdas[atom_idx] = INACTIVE_LAMBDA;
            reml_scores[atom_idx] = 0.0;
            return Ok(false);
        }
        for col in 0..x.ncols() {
            atoms[[atom_idx, col]] = x[[worst_row, col]] - fitted[[worst_row, col]];
        }
        normalize_row(atoms.slice_mut(s![atom_idx, ..]));
        lambdas[atom_idx] = atom_ridge;
        reml_scores[atom_idx] =
            penalized_reconstruction_loss(x, fitted.view(), atom_ridge, atoms.view());
        return Ok(true);
    }

    let old_atom = atoms.row(atom_idx).to_owned();
    let mut residual = x.to_owned() - fitted.view();
    residual += &code
        .view()
        .insert_axis(Axis(1))
        .dot(&old_atom.view().insert_axis(Axis(0)));

    let denominator = code_norm2 + atom_ridge;
    for col in 0..x.ncols() {
        atoms[[atom_idx, col]] = code.dot(&residual.column(col)) / denominator;
    }
    lambdas[atom_idx] = atom_ridge;
    normalize_atom_and_assignments(atoms, assignments, atom_idx);
    let updated_code = assignments.column(atom_idx).to_owned();
    fitted.assign(&x);
    *fitted -= &residual;
    *fitted += &updated_code
        .view()
        .insert_axis(Axis(1))
        .dot(&atoms.row(atom_idx).insert_axis(Axis(0)));
    reml_scores[atom_idx] =
        penalized_reconstruction_loss(x, fitted.view(), atom_ridge, atoms.view());
    Ok(false)
}

fn top_k_assignments(
    x: ArrayView2<'_, f64>,
    atoms: ArrayView2<'_, f64>,
    top_k: usize,
    code_ridge: f64,
) -> Result<Array2<f64>, String> {
    let cross = x.dot(&atoms.t());
    let mut assignments = Array2::<f64>::zeros((x.nrows(), atoms.nrows()));
    for row in 0..x.nrows() {
        let active = top_indices_by_abs(cross.row(row), top_k);
        let coeffs = solve_active_coefficients(atoms, cross.row(row), &active, code_ridge)?;
        for pos in 0..active.len() {
            assignments[[row, active[pos]]] = coeffs[pos];
        }
    }
    Ok(assignments)
}

/// Encode held-out rows `x` (`M x P`) against a frozen dictionary `atoms`
/// (`K x P`) using the same top-`top_k` ridge least-squares routing the fit
/// uses against its final atoms. Returns the `(M, K)` sparse code matrix.
///
/// This is the out-of-sample `transform`/encode step for a fitted linear
/// dictionary; the math (top-k selection + active-set ridge solve) lives in
/// the Rust core so the Python facade stays a thin wrapper.
pub fn linear_dictionary_transform(
    x: ArrayView2<'_, f64>,
    atoms: ArrayView2<'_, f64>,
    top_k: usize,
    code_ridge: f64,
) -> Result<Array2<f64>, String> {
    let k = atoms.nrows();
    if k == 0 {
        return Err("linear_dictionary_transform: dictionary has no atoms".to_string());
    }
    if x.ncols() != atoms.ncols() {
        return Err(format!(
            "linear_dictionary_transform: X has P={} columns but atoms have P={}",
            x.ncols(),
            atoms.ncols()
        ));
    }
    let effective_k = top_k.min(k).max(1);
    top_k_assignments(x, atoms, effective_k, code_ridge)
}

fn softmax_assignments(
    x: ArrayView2<'_, f64>,
    atoms: ArrayView2<'_, f64>,
    top_k: usize,
    temperature: f64,
    code_ridge: f64,
) -> Result<Array2<f64>, String> {
    let cross = x.dot(&atoms.t());
    let atom_norm2 = atoms.map_axis(Axis(1), |row| row.dot(&row).max(MIN_NORM2));
    let mut assignments = Array2::<f64>::zeros((x.nrows(), atoms.nrows()));
    for row in 0..x.nrows() {
        let active = top_indices_by_abs(cross.row(row), top_k);
        let mut max_score = f64::NEG_INFINITY;
        for &atom_idx in &active {
            let score = cross[[row, atom_idx]].abs() / (atom_norm2[atom_idx].sqrt() * temperature);
            if score > max_score {
                max_score = score;
            }
        }
        let mut denom = 0.0;
        for &atom_idx in &active {
            let score = cross[[row, atom_idx]].abs() / (atom_norm2[atom_idx].sqrt() * temperature);
            let mass = (score - max_score).exp();
            assignments[[row, atom_idx]] = mass;
            denom += mass;
        }
        if denom <= 0.0 || !denom.is_finite() {
            return Err("linear_dictionary_fit softmax assignment underflowed".to_string());
        }
        for &atom_idx in &active {
            let projection = cross[[row, atom_idx]] / (atom_norm2[atom_idx] + code_ridge);
            assignments[[row, atom_idx]] = assignments[[row, atom_idx]] * projection / denom;
        }
    }
    Ok(assignments)
}

fn solve_active_coefficients(
    atoms: ArrayView2<'_, f64>,
    cross_row: ArrayView1<'_, f64>,
    active: &[usize],
    code_ridge: f64,
) -> Result<Array1<f64>, String> {
    let m = active.len();
    let mut system = Array2::<f64>::zeros((m, m));
    let mut rhs = Array2::<f64>::zeros((m, 1));
    for i in 0..m {
        rhs[[i, 0]] = cross_row[active[i]];
        for j in 0..m {
            system[[i, j]] = atoms.row(active[i]).dot(&atoms.row(active[j]));
        }
        system[[i, i]] += code_ridge;
    }
    let factor = system
        .cholesky(Side::Lower)
        .map_err(|err| format!("linear_dictionary_fit sparse-code solve failed: {err}"))?;
    let mut solution = rhs;
    factor.solve_mat_in_place(&mut solution);
    Ok(solution.column(0).to_owned())
}

fn top_indices_by_abs(row: ArrayView1<'_, f64>, top_k: usize) -> Vec<usize> {
    let mut selected: Vec<(usize, f64)> = Vec::with_capacity(top_k);
    for idx in 0..row.len() {
        let score = row[idx].abs();
        if selected.len() < top_k {
            selected.push((idx, score));
            continue;
        }
        let mut worst_pos = 0usize;
        for pos in 1..selected.len() {
            if selected[pos].1 < selected[worst_pos].1
                || (selected[pos].1 == selected[worst_pos].1
                    && selected[pos].0 > selected[worst_pos].0)
            {
                worst_pos = pos;
            }
        }
        let worst = selected[worst_pos];
        if score > worst.1 || (score == worst.1 && idx < worst.0) {
            selected[worst_pos] = (idx, score);
        }
    }
    selected.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    selected.into_iter().map(|(idx, _)| idx).collect()
}

fn normalize_atom_and_assignments(
    atoms: &mut Array2<f64>,
    assignments: &mut Array2<f64>,
    atom_idx: usize,
) {
    let norm = atoms.row(atom_idx).dot(&atoms.row(atom_idx)).sqrt();
    if norm > MIN_NORM2.sqrt() {
        atoms.row_mut(atom_idx).mapv_inplace(|value| value / norm);
        assignments
            .column_mut(atom_idx)
            .mapv_inplace(|value| value * norm);
    }
    orient_atom_and_code(atoms, assignments, atom_idx);
}

fn orient_atom_and_code(atoms: &mut Array2<f64>, assignments: &mut Array2<f64>, atom_idx: usize) {
    let sign = first_nonzero_sign(atoms.row(atom_idx));
    if sign < 0.0 {
        atoms.row_mut(atom_idx).mapv_inplace(|value| -value);
        assignments
            .column_mut(atom_idx)
            .mapv_inplace(|value| -value);
    }
}

fn orient_vector(vector: &mut Array1<f64>) {
    if first_nonzero_sign(vector.view()) < 0.0 {
        vector.mapv_inplace(|value| -value);
    }
}

fn first_nonzero_sign(row: ndarray::ArrayView1<'_, f64>) -> f64 {
    for &value in row {
        if value.abs() > 1.0e-12 {
            return value.signum();
        }
    }
    1.0
}

fn normalize_row(mut row: ndarray::ArrayViewMut1<'_, f64>) {
    let norm = row.dot(&row).sqrt();
    if norm > MIN_NORM2.sqrt() {
        row.mapv_inplace(|value| value / norm);
    }
}

fn max_norm_row(x: ArrayView2<'_, f64>) -> usize {
    let mut best = 0usize;
    let mut best_norm = f64::NEG_INFINITY;
    for row in 0..x.nrows() {
        let norm = x.row(row).dot(&x.row(row));
        if norm > best_norm {
            best = row;
            best_norm = norm;
        }
    }
    best
}

fn max_index(values: ndarray::ArrayView1<'_, f64>) -> usize {
    let mut best = 0usize;
    let mut best_value = f64::NEG_INFINITY;
    for idx in 0..values.len() {
        if values[idx] > best_value {
            best = idx;
            best_value = values[idx];
        }
    }
    best
}

fn squared_distance(a: ndarray::ArrayView1<'_, f64>, b: ndarray::ArrayView1<'_, f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(av, bv)| {
            let diff = av - bv;
            diff * diff
        })
        .sum()
}

fn explained_variance(x: ArrayView2<'_, f64>, fitted: ArrayView2<'_, f64>) -> f64 {
    let mut rss = 0.0;
    for row in 0..x.nrows() {
        for col in 0..x.ncols() {
            let residual = x[[row, col]] - fitted[[row, col]];
            rss += residual * residual;
        }
    }
    let means = x.mean_axis(Axis(0)).expect("non-empty input has means");
    let mut tss = 0.0;
    for row in 0..x.nrows() {
        for col in 0..x.ncols() {
            let centered = x[[row, col]] - means[col];
            tss += centered * centered;
        }
    }
    if tss <= MIN_NORM2 {
        if rss <= MIN_NORM2 { 1.0 } else { 0.0 }
    } else {
        1.0 - rss / tss
    }
}

fn penalized_reconstruction_loss(
    x: ArrayView2<'_, f64>,
    fitted: ArrayView2<'_, f64>,
    ridge: f64,
    atoms: ArrayView2<'_, f64>,
) -> f64 {
    let mut loss = 0.0;
    for row in 0..x.nrows() {
        for col in 0..x.ncols() {
            let residual = x[[row, col]] - fitted[[row, col]];
            loss += residual * residual;
        }
    }
    loss + ridge * atoms.iter().map(|value| value * value).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};

    #[test]
    fn planted_sparse_linear_dictionary_reaches_high_explained_variance() {
        let truth = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut assignments = Array2::<f64>::zeros((160, 4));
        for row in 0..160 {
            let atom = row % 4;
            assignments[[row, atom]] = 0.7 + 0.01 * ((row / 4) as f64);
            assignments[[row, (atom + 1) % 4]] = 0.2;
        }
        let x = assignments.dot(&truth);
        let config = LinearDictionaryConfig {
            n_atoms: 4,
            max_iter: 40,
            top_k: 2,
            assignment: LinearDictionaryAssignment::TopK,
            temperature: DEFAULT_TEMPERATURE,
            code_ridge: DEFAULT_CODE_RIDGE,
            tolerance: 1.0e-9,
            center_rank_one: false,
        };

        let fit = fit_linear_dictionary(x.view(), &config).expect("linear dictionary fit");

        assert!(
            fit.explained_variance > 0.95,
            "expected EV > 0.95, got {}",
            fit.explained_variance
        );
    }

    #[test]
    fn coupled_topk_dictionary_reaches_fixed_point_under_small_budget_2372() {
        // A DIFFERENT coupled fixture from the planted 4-atom ring: three
        // orthonormal but NON-axis-aligned directions in R^6, each row loading a
        // cyclic pair (dominant on atom k, secondary on (k+1)%3). The data is
        // exactly representable, so the alternating-LS fixed point is EV = 1, but
        // the two shared atoms per row give the map a dominant slow mode whose
        // ratio (~0.99) needs THOUSANDS of plain sweeps to close a 1e-9 fixed-point
        // residual. This pins, from a different geometry than
        // `planted_sparse_linear_dictionary_reaches_high_explained_variance`, that
        // the safeguarded geometric acceleration drives the certified fixed point
        // inside a small iteration budget instead of grinding to `max_iter` and
        // raising `NonConvergence` (#2372).
        let truth = array![
            [
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            [
                std::f64::consts::FRAC_1_SQRT_2,
                -std::f64::consts::FRAC_1_SQRT_2,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
                0.0,
                0.0
            ],
        ];
        let mut codes = Array2::<f64>::zeros((120, 3));
        for row in 0..120 {
            let atom = row % 3;
            codes[[row, atom]] = 0.6 + 0.02 * ((row / 3) as f64);
            codes[[row, (atom + 1) % 3]] = 0.3;
        }
        let x = codes.dot(&truth);
        let config = LinearDictionaryConfig {
            n_atoms: 3,
            max_iter: 80,
            top_k: 2,
            assignment: LinearDictionaryAssignment::TopK,
            temperature: DEFAULT_TEMPERATURE,
            code_ridge: DEFAULT_CODE_RIDGE,
            tolerance: 1.0e-9,
            center_rank_one: false,
        };

        let fit = fit_linear_dictionary(x.view(), &config)
            .expect("acceleration must reach the fixed point within the budget");
        assert!(
            fit.explained_variance > 0.999,
            "coupled data must reconstruct well at the converged fixed point, got EV {}",
            fit.explained_variance
        );
        assert!(
            fit.convergence.ev_residual <= fit.convergence.tolerance,
            "ev_residual {} must close the {} contract",
            fit.convergence.ev_residual,
            fit.convergence.tolerance
        );
        assert!(
            fit.convergence.routing_residual <= fit.convergence.tolerance,
            "routing_residual {} must close the {} contract",
            fit.convergence.routing_residual,
            fit.convergence.tolerance
        );
        assert_eq!(fit.convergence.accepted_births, 0);
        // The returned assignments must be exactly the canonical reroute against the
        // final atoms (the acceleration must leave the model self-consistent).
        let canonical = reroute_against_atoms(x.view(), fit.atoms.view(), fit.top_k, &config)
            .expect("canonical reroute");
        for (returned, rerouted) in fit.assignments.iter().zip(canonical.iter()) {
            assert_abs_diff_eq!(*returned, *rerouted, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn single_atom_matches_penalized_pca_oracle() {
        let mut x = Array2::<f64>::zeros((80, 3));
        for row in 0..80 {
            let t = (row as f64 - 39.5) / 20.0;
            x[[row, 0]] = 2.0 * t;
            x[[row, 1]] = -t;
            x[[row, 2]] = 0.05 * (row as f64).sin();
        }
        let config = LinearDictionaryConfig {
            n_atoms: 1,
            max_iter: 5,
            top_k: 1,
            assignment: LinearDictionaryAssignment::TopK,
            temperature: DEFAULT_TEMPERATURE,
            code_ridge: DEFAULT_CODE_RIDGE,
            tolerance: DEFAULT_TOLERANCE,
            center_rank_one: false,
        };

        let fit = fit_linear_dictionary(x.view(), &config).expect("rank-one fit");
        let covariance = x.t().dot(&x);
        let (evals, _) = covariance.eigh(Side::Lower).expect("PCA eigensolve");
        let shrink = 1.0 / (1.0 + DEFAULT_CODE_RIDGE);
        let oracle_ev = 1.0
            - ((1.0 - shrink) * (1.0 - shrink) * evals[evals.len() - 1]
                + evals.slice(s![..evals.len() - 1]).sum())
                / evals.sum();

        assert!(fit.explained_variance > 0.99);
        assert_abs_diff_eq!(fit.explained_variance, oracle_ev, epsilon = 2.0e-4);
    }

    #[test]
    fn orthonormal_rank_one_atoms_all_revived_no_dead_collapse_1500() {
        // #1500: rows lie on K mutually ORTHONORMAL rank-1 directions, so a
        // K-atom top_k=1 dictionary that recovers them reconstructs every row
        // exactly (EV → 1). The dead-atom bug emptied a cluster, zeroed that atom
        // permanently, and returned < K live atoms with badly under-explained
        // variance. With empty-cluster re-seeding every atom stays live.
        let (k, p, n) = (4usize, 8usize, 400usize);
        // Deterministic orthonormal directions: eigenvectors of a fixed symmetric
        // matrix are orthonormal, so no RNG is needed for a stable regression.
        let mut a = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                a[[i, j]] = ((i * 7 + j * 3 + 1) % 11) as f64 - 5.0;
            }
        }
        let sym = &a + &a.t();
        let (_evals, evecs) = sym.eigh(Side::Lower).expect("orthonormal directions");
        let dirs = evecs.slice(s![.., ..k]).t().to_owned(); // k×p, orthonormal rows
        let mut x = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let atom = row % k;
            let scale = if row % 2 == 0 { 2.0 } else { -1.5 } + 0.01 * (row / k) as f64;
            for col in 0..p {
                let noise = 1.0e-3 * (((row * p + col) % 13) as f64 - 6.0);
                x[[row, col]] = scale * dirs[[atom, col]] + noise;
            }
        }
        let config = LinearDictionaryConfig {
            n_atoms: k,
            max_iter: 40,
            top_k: 1,
            assignment: LinearDictionaryAssignment::TopK,
            temperature: DEFAULT_TEMPERATURE,
            code_ridge: DEFAULT_CODE_RIDGE,
            tolerance: 1.0e-9,
            center_rank_one: false,
        };
        let fit = fit_linear_dictionary(x.view(), &config).expect("orthonormal dictionary fit");
        let live = fit
            .atoms
            .axis_iter(Axis(0))
            .filter(|atom| atom.iter().any(|value| value.abs() > 1.0e-12))
            .count();
        assert_eq!(
            live, k,
            "all {k} atoms must stay live (no dead-atom collapse); got {live} live"
        );
        assert!(
            fit.explained_variance > 0.99,
            "K orthonormal rank-1 atoms must be reconstructed at EV > 0.99; got {}",
            fit.explained_variance
        );
    }

    #[test]
    fn returned_state_is_the_certified_canonical_routing() {
        // Planted sparse problem where the coordinate-descent routing and a fresh
        // global reroute against updated atoms generally differ. The model must be
        // the exact rerouted state that passed both fixed-point residual tests.
        let truth = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut assignments = Array2::<f64>::zeros((160, 4));
        for row in 0..160 {
            let atom = row % 4;
            assignments[[row, atom]] = 0.7 + 0.01 * ((row / 4) as f64);
            assignments[[row, (atom + 1) % 4]] = 0.2;
        }
        let x = assignments.dot(&truth);
        let config = LinearDictionaryConfig {
            n_atoms: 4,
            max_iter: 40,
            top_k: 2,
            assignment: LinearDictionaryAssignment::TopK,
            temperature: DEFAULT_TEMPERATURE,
            code_ridge: DEFAULT_CODE_RIDGE,
            tolerance: 1.0e-9,
            center_rank_one: false,
        };

        let fit = fit_linear_dictionary(x.view(), &config).expect("linear dictionary fit");
        assert!(fit.convergence.ev_residual <= fit.convergence.tolerance);
        assert!(fit.convergence.routing_residual <= fit.convergence.tolerance);
        assert_eq!(fit.convergence.accepted_births, 0);

        // Returned fitted must be exactly assignments.dot(atoms) for the adopted
        // routing, and the reported EV must match that fitted.
        let canonical = reroute_against_atoms(x.view(), fit.atoms.view(), fit.top_k, &config)
            .expect("canonical reroute");
        for (returned, rerouted) in fit.assignments.iter().zip(canonical.iter()) {
            assert_abs_diff_eq!(*returned, *rerouted, epsilon = 1.0e-12);
        }
        let recomputed_fitted = fit.assignments.dot(&fit.atoms);
        for (a, b) in fit.fitted.iter().zip(recomputed_fitted.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1.0e-10);
        }
        assert_abs_diff_eq!(
            fit.explained_variance,
            explained_variance(x.view(), fit.fitted.view()),
            epsilon = 1.0e-10
        );
    }

    #[test]
    fn centered_rank_one_ceiling_agrees_when_data_already_centered() {
        // Build correlated data, then explicitly mean-center it. On centered input
        // the uncentered XᵀX lane and the centered helper see the same second-moment
        // matrix, so their EVs must agree.
        let mut x = Array2::<f64>::zeros((90, 3));
        for row in 0..90 {
            let t = (row as f64 - 44.5) / 25.0;
            x[[row, 0]] = 1.5 * t;
            x[[row, 1]] = -0.8 * t + 0.02 * (row as f64).cos();
            x[[row, 2]] = 0.6 * t;
        }
        let means = x.mean_axis(Axis(0)).unwrap();
        let centered = &x - &means;

        let config = LinearDictionaryConfig::new(1);
        let uncentered = fit_linear_dictionary(centered.view(), &config).expect("rank-one fit");
        let (_fitted, centered_ev) =
            rank_one_centered_pca_ceiling(centered.view(), DEFAULT_CODE_RIDGE)
                .expect("centered ceiling");

        assert_abs_diff_eq!(uncentered.explained_variance, centered_ev, epsilon = 1.0e-9);
    }

    #[test]
    fn centered_rank_one_ceiling_beats_uncentered_with_strong_mean() {
        // Strong column mean (offset) plus a low-variance signal direction: the
        // uncentered XᵀX lane wastes its single rank on the mean direction and
        // under-explains the CENTERED variance, while the centered helper recovers
        // the true principal component and is a genuine, higher centered-PCA ceiling.
        let mut x = Array2::<f64>::zeros((120, 2));
        for row in 0..120 {
            let t = (row as f64 - 59.5) / 60.0; // small spread around the offset
            x[[row, 0]] = 50.0 + 0.3 * t;
            x[[row, 1]] = 50.0 - 0.3 * t;
        }
        let config = LinearDictionaryConfig::new(1);
        let uncentered = fit_linear_dictionary(x.view(), &config).expect("rank-one fit");
        let (fitted, centered_ev) =
            rank_one_centered_pca_ceiling(x.view(), DEFAULT_CODE_RIDGE).expect("centered ceiling");

        assert!(
            centered_ev > uncentered.explained_variance + 1.0e-6,
            "centered ceiling ({centered_ev}) should beat uncentered lane ({}) on strong-mean data",
            uncentered.explained_variance
        );
        // The centered helper's reported EV is consistent with its returned fitted.
        assert_abs_diff_eq!(
            centered_ev,
            explained_variance(x.view(), fitted.view()),
            epsilon = 1.0e-10
        );
    }

    #[test]
    fn center_rank_one_config_flag_routes_k1_lane_to_centered_ceiling() {
        // Strong-mean, low-variance-signal data: the default (uncentered) K=1 lane
        // wastes its single rank on the mean, so setting `center_rank_one = true`
        // must route the lane through the centered computation and report the
        // genuine (higher) centered-PCA ceiling — matching the standalone helper.
        let mut x = Array2::<f64>::zeros((100, 3));
        for row in 0..100 {
            let t = (row as f64 - 49.5) / 50.0;
            x[[row, 0]] = 30.0 + 0.2 * t;
            x[[row, 1]] = 30.0 - 0.2 * t;
            x[[row, 2]] = 30.0 + 0.05 * t;
        }

        let default_config = LinearDictionaryConfig::new(1);
        assert!(
            !default_config.center_rank_one,
            "flag must default to false"
        );
        let uncentered = fit_linear_dictionary(x.view(), &default_config).expect("uncentered lane");

        let mut centered_config = LinearDictionaryConfig::new(1);
        centered_config.center_rank_one = true;
        let centered = fit_linear_dictionary(x.view(), &centered_config).expect("centered lane");

        // The flag actually routes to the centered lane: its EV equals the helper's
        // centered ceiling and strictly beats the default uncentered lane.
        let (_fitted, helper_ev) =
            rank_one_centered_pca_ceiling(x.view(), DEFAULT_CODE_RIDGE).expect("helper ceiling");
        assert_abs_diff_eq!(centered.explained_variance, helper_ev, epsilon = 1.0e-10);
        assert!(
            centered.explained_variance > uncentered.explained_variance + 1.0e-6,
            "center_rank_one=true ({}) must beat default ({}) on strong-mean data",
            centered.explained_variance,
            uncentered.explained_variance
        );
        // Centered lane reports the affine reconstruction directly, so its EV is
        // consistent with the returned `fitted` (which INCLUDES the mean and is not
        // assignments.dot(atoms) in this mode).
        assert_abs_diff_eq!(
            centered.explained_variance,
            explained_variance(x.view(), centered.fitted.view()),
            epsilon = 1.0e-10
        );
    }

    #[test]
    fn nonconverged_multi_atom_fit_is_an_error_not_a_model() {
        // SPEC 20: this problem moves on its first full sweep, so a one-sweep
        // budget cannot satisfy the canonical fixed-point certificate. The solver
        // must return finite numerical evidence instead of a partial model.
        let mut x = Array2::<f64>::zeros((24, 3));
        for row in 0..24 {
            x[[row, row % 3]] = 1.0 + 0.01 * row as f64;
        }
        let config = LinearDictionaryConfig {
            n_atoms: 2,
            max_iter: 1,
            top_k: 1,
            assignment: LinearDictionaryAssignment::TopK,
            temperature: DEFAULT_TEMPERATURE,
            code_ridge: DEFAULT_CODE_RIDGE,
            tolerance: DEFAULT_TOLERANCE,
            center_rank_one: false,
        };
        let err = fit_linear_dictionary(x.view(), &config)
            .expect_err("one sweep cannot certify an EV plateau");
        match err {
            LinearDictionaryError::NonConvergence {
                iterations,
                explained_variance,
                ev_residual,
                routing_residual,
                accepted_births,
                tolerance,
            } => {
                assert_eq!(iterations, 1);
                assert!(explained_variance.is_finite());
                assert!(ev_residual.is_finite());
                assert!(routing_residual.is_finite());
                assert!(
                    ev_residual > tolerance || routing_residual > tolerance || accepted_births > 0
                );
                assert_eq!(tolerance, DEFAULT_TOLERANCE);
            }
            other => panic!("expected typed non-convergence evidence, got: {other}"),
        }
    }

    #[test]
    fn negative_convergence_tolerance_is_rejected() {
        let x = array![[1.0, 0.0], [0.0, 1.0]];
        let mut config = LinearDictionaryConfig::new(2);
        config.tolerance = -f64::EPSILON;
        let error = fit_linear_dictionary(x.view(), &config)
            .expect_err("a negative residual tolerance has no convergence meaning");
        assert!(matches!(error, LinearDictionaryError::InvalidInput { .. }));
    }

    #[test]
    fn sparse_assignment_scales_to_thousand_atom_dictionary() {
        let active_atoms = array![
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let mut x = Array2::<f64>::zeros((256, 8));
        for row in 0..x.nrows() {
            let atom = row % active_atoms.nrows();
            let scale = 0.7 + 0.003 * row as f64;
            x.row_mut(row).assign(&(&active_atoms.row(atom) * scale));
        }
        let config = LinearDictionaryConfig {
            n_atoms: 1024,
            max_iter: 8,
            top_k: 1,
            assignment: LinearDictionaryAssignment::TopK,
            temperature: DEFAULT_TEMPERATURE,
            code_ridge: DEFAULT_CODE_RIDGE,
            tolerance: 1.0e-9,
            center_rank_one: false,
        };

        let fit = fit_linear_dictionary(x.view(), &config).expect("large-K linear dictionary fit");
        let max_active = fit
            .assignments
            .axis_iter(Axis(0))
            .map(|row| row.iter().filter(|value| value.abs() > 1.0e-10).count())
            .max()
            .unwrap();

        assert_eq!(max_active, 1);
        assert!(
            fit.explained_variance > 0.95,
            "expected EV > 0.95 at K=1024, got {}",
            fit.explained_variance
        );
    }

    /// #2372 instrument: per-sweep EV/routing trace on the planted fixture the
    /// two plateau tests use — discriminates a routing LIMIT CYCLE (support
    /// flipping between equivalent top-k routings, residual oscillating at a
    /// fixed amplitude) from slow drift (residual decaying but not reaching
    /// the 1e-9 tolerance inside the 40-sweep budget).
    #[test]
    fn zz_measure_2372_dictionary_plateau_trace() {
        let (x, config) = planted_fixture_for_trace();
        let top_k = config.top_k.min(config.n_atoms).max(1);
        let mut atoms = initialize_atoms(x.view(), config.n_atoms);
        let mut assignments =
            reroute_against_atoms(x.view(), atoms.view(), top_k, &config).expect("route");
        let mut fitted = assignments.dot(&atoms);
        let mut lambdas = Array1::<f64>::from_elem(config.n_atoms, INACTIVE_LAMBDA);
        let mut reml_scores = Array1::<f64>::zeros(config.n_atoms);
        let mut previous_ev = explained_variance(x.view(), fitted.view());
        let mut prev_support: Option<Vec<Vec<bool>>> = None;
        for sweep in 0..12 {
            for atom_idx in 0..config.n_atoms {
                fit_one_atom_penalized_ls(
                    x.view(),
                    &mut atoms,
                    &mut assignments,
                    &mut fitted,
                    &mut lambdas,
                    &mut reml_scores,
                    atom_idx,
                    config.code_ridge,
                )
                .expect("atom update");
            }
            let sweep_ev = explained_variance(x.view(), fitted.view());
            let rerouted =
                reroute_against_atoms(x.view(), atoms.view(), top_k, &config).expect("route");
            let rerouted_fitted = rerouted.dot(&atoms);
            let rerouted_ev = explained_variance(x.view(), rerouted_fitted.view());
            let support: Vec<Vec<bool>> = (0..rerouted.nrows())
                .map(|i| rerouted.row(i).iter().map(|v| *v != 0.0).collect())
                .collect();
            let support_changed = prev_support.as_ref().map_or(-1_i64, |p| {
                p.iter()
                    .zip(&support)
                    .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
                    .sum::<usize>() as i64
            });
            eprintln!(
                "[zz2372:dict] sweep={sweep} sweep_ev={sweep_ev:.15} rerouted_ev={rerouted_ev:.15} ev_res={:.3e} routing_res={:.3e} support_flips={support_changed}",
                (rerouted_ev - previous_ev).abs(),
                (rerouted_ev - sweep_ev).abs(),
            );
            previous_ev = rerouted_ev;
            prev_support = Some(support);
            assignments = rerouted;
            fitted = rerouted_fitted;
        }
    }

    /// The same 6x12 planted two-atom overcomplete fixture
    /// `planted_sparse_linear_dictionary_reaches_high_explained_variance` uses,
    /// factored so the trace and the contract test stay on identical data.
    fn planted_fixture_for_trace() -> (ndarray::Array2<f64>, LinearDictionaryConfig) {
        let truth = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut assignments = Array2::<f64>::zeros((160, 4));
        for row in 0..160 {
            let atom = row % 4;
            assignments[[row, atom]] = 0.7 + 0.01 * ((row / 4) as f64);
            assignments[[row, (atom + 1) % 4]] = 0.2;
        }
        let x = assignments.dot(&truth);
        let config = LinearDictionaryConfig {
            n_atoms: 4,
            max_iter: 40,
            top_k: 2,
            assignment: LinearDictionaryAssignment::TopK,
            temperature: DEFAULT_TEMPERATURE,
            code_ridge: DEFAULT_CODE_RIDGE,
            tolerance: 1.0e-9,
            center_rank_one: false,
        };

        (x, config)
    }

}

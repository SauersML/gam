//! Python-free frozen-decoder out-of-sample SAE entry (#2236 Increment 2).
//!
//! [`run_sae_manifold_oos`] owns the complete inference operation: request
//! validation, basis/evaluator reconstruction from the persisted dictionary,
//! cold or warm coordinate and routing seeds, the fixed-decoder Arrow-Schur
//! solve, exact assignment reconstruction, collapse-aware reconstruction, and a
//! typed report. Bindings only translate their wire representation into
//! [`SaeOosRequest`] and serialize [`SaeOosReport`].

use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;
use ndarray::{Array1, Array2, Array3, ArrayView2, s};

use crate::hybrid_split::AtomLinearImage;
use crate::inference::steering::{
    AppliedDoseProbe, SteerPlan, TargetDoseConfig, TargetDosePlan, TargetDoseRequest, steer_delta,
    steer_to_target_nats,
};

use super::{
    AssignmentMode, SaeAssignment, SaeAtomBasisKind, SaeAtomGeometryPlan, SaeCertifyRequest,
    SaeFitError, SaeManifoldAtom, SaeManifoldLoss, SaeManifoldRho, SaeManifoldTerm,
    SaeStreamingPlan, run_sae_manifold_certify, sae_pca_seed_initial_coords,
};

const SAE_ACTIVE_ASSIGNMENT_MASS: f64 = 1.0e-8;

/// Exact persisted definition of one trained atom needed by frozen-decoder OOS
/// inference. Topology, resolution, reference metric, dimension, and basis
/// width have one authority: `geometry`. The decoder is validated against the
/// width derived by that plan at construction.
#[derive(Clone, Debug)]
pub struct SaeOosAtomSpec {
    geometry: SaeAtomGeometryPlan,
    pub decoder: Array2<f64>,
}

impl SaeOosAtomSpec {
    pub fn new(geometry: SaeAtomGeometryPlan, decoder: Array2<f64>) -> Result<Self, String> {
        let expected_width = geometry.basis_size()?;
        if decoder.nrows() != expected_width || decoder.ncols() == 0 {
            return Err(format!(
                "SaeOosAtomSpec::new: decoder shape {:?} must be ({expected_width}, p) with p >= 1 for plan {:?}",
                decoder.dim(),
                geometry.kind()
            ));
        }
        if decoder.iter().any(|value| !value.is_finite()) {
            return Err("SaeOosAtomSpec::new: decoder must be finite".to_string());
        }
        Ok(Self { geometry, decoder })
    }

    pub fn geometry(&self) -> &SaeAtomGeometryPlan {
        &self.geometry
    }

    pub fn basis_kind(&self) -> &SaeAtomBasisKind {
        self.geometry.kind()
    }

    pub fn latent_dim(&self) -> usize {
        self.geometry.latent_dim()
    }

    pub fn basis_size(&self) -> Result<usize, String> {
        self.geometry.basis_size()
    }
}

/// Pair persisted typed geometry plans with their frozen decoders. There is no
/// compatibility path from parallel kind/dimension/order/width scalars and no
/// resolution inference from decoder width.
pub fn persisted_oos_atom_specs(
    geometry_plans: &[SaeAtomGeometryPlan],
    decoder_blocks: &[ArrayView2<'_, f64>],
) -> Result<Vec<SaeOosAtomSpec>, String> {
    let k_atoms = geometry_plans.len();
    if decoder_blocks.len() != k_atoms {
        return Err(format!(
            "persisted_oos_atom_specs: decoder count {} must equal geometry-plan count {k_atoms}",
            decoder_blocks.len(),
        ));
    }
    geometry_plans
        .iter()
        .cloned()
        .zip(decoder_blocks)
        .map(|(geometry, decoder)| SaeOosAtomSpec::new(geometry, decoder.to_owned()))
        .collect()
}

/// Assignment family for a frozen-decoder OOS solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SaeOosAssignmentKind {
    Softmax,
    OrderedBetaBernoulli { learnable_alpha: bool },
    ThresholdGate { threshold: f64 },
    TopK,
}

impl SaeOosAssignmentKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Softmax => "softmax",
            Self::OrderedBetaBernoulli { .. } => "ordered_beta_bernoulli",
            Self::ThresholdGate { .. } => "threshold_gate",
            Self::TopK => "topk",
        }
    }
}

/// The complete terminal regularization state of the trained dictionary.
/// Frozen-decoder inference must optimize the same objective that produced the
/// decoder; an initial-strength substitute is a different model and is not
/// accepted.
#[derive(Clone, Debug)]
pub struct SaeOosRegularization {
    pub log_lambda_sparse: f64,
    pub log_lambda_smooth: Vec<f64>,
    pub log_ard: Vec<Vec<f64>>,
}

/// Fully owned, Python-free request for frozen-decoder OOS inference.
pub struct SaeOosRequest {
    pub target: Array2<f64>,
    pub atoms: Vec<SaeOosAtomSpec>,
    pub assignment: SaeOosAssignmentKind,
    pub alpha: f64,
    pub tau: f64,
    pub regularization: SaeOosRegularization,
    pub max_iter: usize,
    pub learning_rate: f64,
    pub ridge_ext_coord: f64,
    pub initial_logits: Option<Array2<f64>>,
    pub initial_coords: Option<Array3<f64>>,
    pub top_k: Option<usize>,
    pub hybrid_linear_images: Vec<AtomLinearImage>,
}

/// One atom's materialized OOS outputs.
pub struct SaeOosAtomReport {
    pub basis_kind: SaeAtomBasisKind,
    pub decoder: Array2<f64>,
    pub coords: Array2<f64>,
    pub assignments: Array1<f64>,
    pub reconstruction: Array2<f64>,
    pub active_dim: usize,
}

/// Complete typed result of [`run_sae_manifold_oos`].
pub struct SaeOosReport {
    pub atoms: Vec<SaeOosAtomReport>,
    pub assignments: Array2<f64>,
    pub logits: Array2<f64>,
    pub active_mask: Vec<bool>,
    pub fitted: Array2<f64>,
    pub loss: SaeManifoldLoss,
    pub alpha: f64,
    pub rho: SaeManifoldRho,
    pub assignment_kind: &'static str,
    pub streaming_plan: SaeStreamingPlan,
}

fn finite_positive(name: &str, value: f64) -> Result<(), String> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(format!(
            "run_sae_manifold_oos: {name} must be finite and positive; got {value}"
        ))
    }
}

fn build_oos_atom(
    atom_index: usize,
    spec: &SaeOosAtomSpec,
    start_coords: ArrayView2<'_, f64>,
    p_out: usize,
) -> Result<SaeManifoldAtom, String> {
    let geometry = spec.geometry();
    let basis_size = geometry.basis_size()?;
    let latent_dim = geometry.latent_dim();
    if start_coords.ncols() != latent_dim {
        return Err(format!(
            "run_sae_manifold_oos: atom {atom_index} coordinate width {} does not match geometry latent_dim {latent_dim}",
            start_coords.ncols()
        ));
    }
    if spec.decoder.dim() != (basis_size, p_out) {
        return Err(format!(
            "run_sae_manifold_oos: atom {atom_index} decoder shape {:?} must equal plan-derived ({basis_size}, {p_out})",
            spec.decoder.dim()
        ));
    }
    if spec.decoder.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "run_sae_manifold_oos: atom {atom_index} decoder contains non-finite values"
        ));
    }
    let bundle = geometry.evaluate_bundle(start_coords)?;

    Ok(SaeManifoldAtom::new_with_provided_function_gram(
        format!("oos_atom_{atom_index}"),
        geometry.kind().clone(),
        latent_dim,
        bundle.basis_values,
        bundle.basis_jacobian,
        spec.decoder.clone(),
        bundle.reference_penalty,
    )?
    .with_basis_second_jet(bundle.evaluator)
    .with_geometry_plan(geometry.clone())?)
}

fn build_rho(
    regularization: SaeOosRegularization,
    latent_dims: &[usize],
    assignment_mode: AssignmentMode,
) -> Result<SaeManifoldRho, String> {
    let k_atoms = latent_dims.len();
    let SaeOosRegularization {
        log_lambda_sparse,
        log_lambda_smooth,
        log_ard,
    } = regularization;
    if log_lambda_smooth.len() != k_atoms
        || !log_lambda_smooth.iter().all(|value| value.is_finite())
    {
        return Err(format!(
            "run_sae_manifold_oos: trained log_lambda_smooth must contain {k_atoms} finite values"
        ));
    }
    if log_ard.len() != k_atoms {
        return Err(format!(
            "run_sae_manifold_oos: trained log_ard must contain {k_atoms} atom blocks; got {}",
            log_ard.len()
        ));
    }
    let mut ard = Vec::with_capacity(k_atoms);
    for (atom_index, (values, &dim)) in log_ard.iter().zip(latent_dims).enumerate() {
        if !(values.is_empty() || values.len() == dim)
            || !values.iter().all(|value| value.is_finite())
        {
            return Err(format!(
                "run_sae_manifold_oos: trained log_ard[{atom_index}] must be empty or contain {dim} finite values"
            ));
        }
        ard.push(Array1::from(values.clone()));
    }
    let rho = SaeManifoldRho::with_per_atom_smooth(log_lambda_sparse, log_lambda_smooth, ard)
        .for_assignment(assignment_mode);
    rho.validate_log_strength_domain()
        .map_err(|error| format!("run_sae_manifold_oos: {error}"))?;
    Ok(rho)
}

/// Execute frozen-decoder OOS inference from a typed, owned request.
#[must_use = "OOS inference errors and report must be handled"]
pub fn run_sae_manifold_oos(request: SaeOosRequest) -> Result<SaeOosReport, String> {
    let SaeOosRequest {
        target,
        atoms: atom_specs,
        assignment,
        alpha,
        tau,
        regularization,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        initial_logits,
        initial_coords,
        top_k,
        hybrid_linear_images,
    } = request;
    let (n_obs, p_out) = target.dim();
    let k_atoms = atom_specs.len();
    if n_obs == 0 || p_out == 0 || !target.iter().all(|value| value.is_finite()) {
        return Err(format!(
            "run_sae_manifold_oos: target must be a non-empty finite matrix; got shape ({n_obs}, {p_out})"
        ));
    }
    if k_atoms == 0 {
        return Err("run_sae_manifold_oos: at least one atom is required".to_string());
    }
    finite_positive("alpha", alpha)?;
    finite_positive("tau", tau)?;
    finite_positive("learning_rate", learning_rate)?;
    finite_positive("ridge_ext_coord", ridge_ext_coord)?;
    if max_iter == 0 {
        return Err("run_sae_manifold_oos: max_iter must be positive".to_string());
    }
    if let SaeOosAssignmentKind::ThresholdGate { threshold } = assignment {
        if !threshold.is_finite() {
            return Err(format!(
                "run_sae_manifold_oos: threshold-gate threshold must be finite; got {threshold}"
            ));
        }
    }
    if let Some(limit) = top_k {
        if limit == 0 || limit > k_atoms {
            return Err(format!(
                "run_sae_manifold_oos: top_k must be in 1..={k_atoms}; got {limit}"
            ));
        }
    }
    match (assignment, top_k) {
        (SaeOosAssignmentKind::TopK, None) => {
            return Err(
                "run_sae_manifold_oos: TopK assignment requires an explicit top_k support size"
                    .to_string(),
            );
        }
        (SaeOosAssignmentKind::TopK, Some(_)) | (_, None) => {}
        (_, Some(limit)) => {
            return Err(format!(
                "run_sae_manifold_oos: top_k={limit} is valid only for TopK assignment"
            ));
        }
    }

    let basis_kinds: Vec<SaeAtomBasisKind> = atom_specs
        .iter()
        .map(|spec| spec.basis_kind().clone())
        .collect();
    let latent_dims: Vec<usize> = atom_specs.iter().map(SaeOosAtomSpec::latent_dim).collect();
    let cold_coords = initial_coords.is_none();
    let cold_logits = initial_logits.is_none();
    let start_coords = match initial_coords {
        Some(coords) => {
            let shape = coords.shape();
            if shape[0] != k_atoms || shape[1] != n_obs {
                return Err(format!(
                    "run_sae_manifold_oos: initial_coords must start with (K, N)=({k_atoms}, {n_obs}); got {shape:?}"
                ));
            }
            for (atom_index, &dim) in latent_dims.iter().enumerate() {
                if dim > shape[2] {
                    return Err(format!(
                        "run_sae_manifold_oos: initial_coords width {} is smaller than atom {atom_index} latent_dim={dim}",
                        shape[2]
                    ));
                }
            }
            if !coords.iter().all(|value| value.is_finite()) {
                return Err(
                    "run_sae_manifold_oos: initial_coords contains non-finite values".to_string(),
                );
            }
            coords
        }
        None => sae_pca_seed_initial_coords(target.view(), &basis_kinds, &latent_dims)?,
    };
    let logits = match initial_logits {
        Some(logits) => {
            if logits.dim() != (n_obs, k_atoms) || !logits.iter().all(|value| value.is_finite()) {
                return Err(format!(
                    "run_sae_manifold_oos: initial_logits must be a finite ({n_obs}, {k_atoms}) matrix; got {:?}",
                    logits.dim()
                ));
            }
            logits
        }
        None => {
            let mut logits = Array2::<f64>::zeros((n_obs, k_atoms));
            if k_atoms == 1
                && matches!(
                    assignment,
                    SaeOosAssignmentKind::OrderedBetaBernoulli { .. }
                )
            {
                logits.column_mut(0).fill(4.0);
            }
            logits
        }
    };

    let mode = match assignment {
        SaeOosAssignmentKind::Softmax => AssignmentMode::softmax(tau),
        SaeOosAssignmentKind::OrderedBetaBernoulli { learnable_alpha } => {
            AssignmentMode::ordered_beta_bernoulli(tau, alpha, learnable_alpha)
        }
        SaeOosAssignmentKind::ThresholdGate { threshold } => {
            AssignmentMode::threshold_gate(tau, threshold)
        }
        SaeOosAssignmentKind::TopK => {
            let support = top_k.ok_or_else(|| {
                "run_sae_manifold_oos: TopK assignment requires an explicit top_k support size"
                    .to_string()
            })?;
            AssignmentMode::top_k_support(support)
        }
    };

    let mut coord_blocks = Vec::with_capacity(k_atoms);
    let mut atoms = Vec::with_capacity(k_atoms);
    for (atom_index, spec) in atom_specs.iter().enumerate() {
        let coords = start_coords
            .slice(s![atom_index, 0..n_obs, 0..spec.latent_dim()])
            .to_owned();
        atoms.push(build_oos_atom(atom_index, spec, coords.view(), p_out)?);
        coord_blocks.push(coords);
    }
    let manifolds = basis_kinds
        .iter()
        .zip(latent_dims.iter().copied())
        .map(|(kind, dim)| kind.latent_manifold(dim))
        .collect();
    let assignment_state =
        SaeAssignment::from_blocks_with_mode_and_manifolds(logits, coord_blocks, manifolds, mode)?;
    let mut term = SaeManifoldTerm::new(atoms, assignment_state)?;

    if !hybrid_linear_images.is_empty() {
        term.set_hybrid_linear_images(hybrid_linear_images.clone())?;
    }
    let mut rho = build_rho(regularization, &latent_dims, mode)?;
    if cold_coords {
        term.seed_coords_by_decoder_projection(target.view())?;
    }
    if cold_logits && assignment == SaeOosAssignmentKind::Softmax {
        term.seed_oos_softmax_logits_from_projection_residuals(target.view(), tau);
    } else if cold_logits
        && matches!(
            assignment,
            SaeOosAssignmentKind::OrderedBetaBernoulli { .. }
        )
    {
        term.seed_oos_ordered_beta_bernoulli_logits_from_projected_decoder_lsq(target.view(), tau);
    }

    let loss = term.run_fixed_decoder_arrow_schur(
        target.view(),
        &mut rho,
        None,
        max_iter,
        learning_rate,
        ridge_ext_coord,
    )?;
    let assignments = term.assignment.assignments();
    let (fitted, atom_reconstructions, effective_coords) =
        term.reconstruct_with_atom_images_target_aware(target.view(), assignments.view())?;

    let mut atom_reports = Vec::with_capacity(k_atoms);
    for (atom_index, (reconstruction, coords)) in atom_reconstructions
        .into_iter()
        .zip(effective_coords)
        .enumerate()
    {
        atom_reports.push(SaeOosAtomReport {
            basis_kind: atom_specs[atom_index].basis_kind().clone(),
            decoder: term.atoms[atom_index].decoder_coefficients.clone(),
            coords,
            assignments: assignments.column(atom_index).to_owned(),
            reconstruction,
            active_dim: latent_dims[atom_index],
        });
    }
    let active_mask = (0..k_atoms)
        .map(|atom_index| assignments.column(atom_index).sum() > SAE_ACTIVE_ASSIGNMENT_MASS)
        .collect();
    let streaming_plan = term.streaming_plan();

    Ok(SaeOosReport {
        atoms: atom_reports,
        assignments,
        logits: term.assignment.logits.clone(),
        active_mask,
        fitted,
        loss,
        alpha,
        rho,
        assignment_kind: assignment.label(),
        streaming_plan,
    })
}

/// Fully owned, Python-free request for a frozen-decoder latent-steering plan.
///
/// Steering rebuilds the SAME frozen dictionary the OOS solve does (via the
/// shared [`SaeOosAtomSpec`] rebuild) but at the model's TRAINED per-row
/// coordinates and routing logits — there is no coordinate solve. It then
/// measures the activation-space delta that moves atom `atom_k`'s on-manifold
/// coordinate from `t_from` to `t_to`, priced through the per-row metric.
pub struct SaeSteerRequest {
    /// Persisted trained atoms (decoder + basis schema), identical to the OOS
    /// dictionary definition.
    pub atoms: Vec<SaeOosAtomSpec>,
    /// The model's TRAINED per-row on-manifold coordinates, one `(n_obs,
    /// latent_dim)` block per atom.
    pub coords: Vec<Array2<f64>>,
    /// The model's TRAINED per-row routing logits, `(n_obs, k_atoms)`.
    pub logits: Array2<f64>,
    /// Assignment family.
    pub assignment: SaeOosAssignmentKind,
    /// Saved active-set support. Required for `TopK`; for capped softmax it
    /// restores the fitted assignment contract rather than guessing from the
    /// current logits.
    pub top_k: Option<usize>,
    /// ordered Beta--Bernoulli concentration α (ignored outside `ordered_beta_bernoulli`).
    pub alpha: f64,
    /// Softmax / gate temperature.
    pub tau: f64,
    /// The per-row output-Fisher metric the dose is measured through, or `None`
    /// for the geometry-only Euclidean metric (dose degrades to `None`).
    pub fisher_metric: Option<gam_problem::RowMetric>,
    /// The atom whose coordinate is being steered.
    pub atom_k: usize,
    /// Exact fitted row whose output-Fisher block prices the applied move.
    pub metric_row: usize,
    /// Exact amplitude multiplying the applied decoder chord.
    pub amplitude: f64,
    /// Source on-manifold coordinate.
    pub t_from: Vec<f64>,
    /// Target on-manifold coordinate.
    pub t_to: Vec<f64>,
}

/// Rebuild the frozen trained dictionary into an [`SaeManifoldTerm`] pinned at its
/// TRAINED coordinates / logits (no coordinate solve), with the installed
/// output-Fisher metric when supplied. Shared by the amplitude steer
/// ([`run_sae_manifold_steer`]) and the target-dose steer
/// ([`run_sae_manifold_steer_to_target`]) so both rebuild the trained dictionary
/// bit-for-bit through one path. `caller` names the caller in error messages.
struct SteerTermRequest {
    caller: &'static str,
    atom_specs: Vec<SaeOosAtomSpec>,
    coords: Vec<Array2<f64>>,
    logits: Array2<f64>,
    assignment: SaeOosAssignmentKind,
    top_k: Option<usize>,
    alpha: f64,
    tau: f64,
    fisher_metric: Option<gam_problem::RowMetric>,
    atom_k: usize,
    metric_row: usize,
}

fn build_steer_term(request: SteerTermRequest) -> Result<SaeManifoldTerm, String> {
    let SteerTermRequest {
        caller,
        atom_specs,
        coords,
        logits,
        assignment,
        top_k,
        alpha,
        tau,
        fisher_metric,
        atom_k,
        metric_row,
    } = request;
    let k_atoms = atom_specs.len();
    if k_atoms == 0 {
        return Err(format!("{caller}: at least one atom is required"));
    }
    if coords.len() != k_atoms {
        return Err(format!(
            "{caller}: coords must have K={k_atoms} per-atom blocks; got {}",
            coords.len()
        ));
    }
    if atom_k >= k_atoms {
        return Err(format!(
            "{caller}: atom_k={atom_k} out of range for K={k_atoms} atoms"
        ));
    }
    if metric_row >= logits.nrows() {
        return Err(format!(
            "{caller}: metric_row={metric_row} out of range for {} fitted rows",
            logits.nrows()
        ));
    }
    finite_positive("alpha", alpha)?;
    finite_positive("tau", tau)?;
    let n_obs = logits.nrows();
    let p_out = atom_specs[0].decoder.ncols();
    if n_obs == 0 || p_out == 0 {
        return Err(format!(
            "{caller}: n_obs and p_out must be positive; got ({n_obs}, {p_out})"
        ));
    }
    if logits.dim() != (n_obs, k_atoms) || !logits.iter().all(|value| value.is_finite()) {
        return Err(format!(
            "{caller}: logits must be a finite ({n_obs}, {k_atoms}) matrix; got {:?}",
            logits.dim()
        ));
    }
    if let Some(support) = top_k {
        if support == 0 || support > k_atoms {
            return Err(format!(
                "{caller}: top_k must be in 1..={k_atoms}; got {support}"
            ));
        }
    }
    match (assignment, top_k) {
        (SaeOosAssignmentKind::TopK, None) => {
            return Err(format!(
                "{caller}: TopK assignment requires the saved top_k support size"
            ));
        }
        (SaeOosAssignmentKind::TopK, Some(_)) | (_, None) => {}
        (_, Some(support)) => {
            return Err(format!(
                "{caller}: top_k={support} is valid only for TopK assignment"
            ));
        }
    }
    let mode = match assignment {
        SaeOosAssignmentKind::Softmax => AssignmentMode::softmax(tau),
        SaeOosAssignmentKind::OrderedBetaBernoulli { learnable_alpha } => {
            AssignmentMode::ordered_beta_bernoulli(tau, alpha, learnable_alpha)
        }
        SaeOosAssignmentKind::ThresholdGate { threshold } => {
            if !threshold.is_finite() {
                return Err(format!(
                    "{caller}: threshold-gate threshold must be finite; got {threshold}"
                ));
            }
            AssignmentMode::threshold_gate(tau, threshold)
        }
        SaeOosAssignmentKind::TopK => {
            let support = top_k.ok_or_else(|| {
                format!("{caller}: TopK assignment requires the saved top_k support size")
            })?;
            AssignmentMode::top_k_support(support)
        }
    };

    // Rebuild each trained atom at its TRAINED coordinates (no solve), reusing the
    // shared frozen-decoder rebuild so the steer design matches the OOS design and
    // the trained dictionary bit-for-bit.
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    let mut atoms = Vec::with_capacity(k_atoms);
    for (atom_index, spec) in atom_specs.iter().enumerate() {
        let block = &coords[atom_index];
        if block.dim() != (n_obs, spec.latent_dim()) {
            return Err(format!(
                "{caller}: coords[{atom_index}] must be (N, d)=({n_obs}, {}); got {:?}",
                spec.latent_dim(),
                block.dim()
            ));
        }
        if !block.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "{caller}: coords[{atom_index}] contains non-finite values"
            ));
        }
        atoms.push(build_oos_atom(atom_index, spec, block.view(), p_out)?);
        coord_blocks.push(block.clone());
    }
    let manifolds = atom_specs
        .iter()
        .map(|spec| spec.basis_kind().latent_manifold(spec.latent_dim()))
        .collect();
    let assignment_state =
        SaeAssignment::from_blocks_with_mode_and_manifolds(logits, coord_blocks, manifolds, mode)?;
    let mut term = SaeManifoldTerm::new(atoms, assignment_state)?;
    if let Some(metric) = fisher_metric {
        term.set_row_metric(metric)?;
    }
    Ok(term)
}

/// Build the frozen trained dictionary and measure the steering plan for atom
/// `atom_k` at a fixed `amplitude`. Mirrors the term rebuild of
/// [`run_sae_manifold_oos`] (sharing [`build_oos_atom`]) with no coordinate solve.
pub fn run_sae_manifold_steer(request: SaeSteerRequest) -> Result<SteerPlan, String> {
    let SaeSteerRequest {
        atoms,
        coords,
        logits,
        assignment,
        top_k,
        alpha,
        tau,
        fisher_metric,
        atom_k,
        metric_row,
        amplitude,
        t_from,
        t_to,
    } = request;
    finite_positive("amplitude", amplitude)?;
    let term = build_steer_term(SteerTermRequest {
        caller: "run_sae_manifold_steer",
        atom_specs: atoms,
        coords,
        logits,
        assignment,
        top_k,
        alpha,
        tau,
        fisher_metric,
        atom_k,
        metric_row,
    })?;
    // The metric the dose is measured through: the installed per-row metric, or a
    // bit-identical Euclidean metric (geometry-only; dose degrades to None).
    let euclidean = gam_problem::RowMetric::euclidean(term.n_obs(), term.output_dim())?;
    let metric = term.row_metric().unwrap_or(&euclidean);
    steer_delta(&term, metric, atom_k, metric_row, amplitude, &t_from, &t_to)
}

/// Fully owned request to solve for the amplitude realizing a TARGET output-KL
/// dose (in nats) on atom `atom_k`'s chord (gh#2263 target-dose surface). Same
/// frozen-dictionary rebuild as [`SaeSteerRequest`], with `amplitude` replaced by
/// the requested `target_nats` and the closed-loop `config`.
pub struct SaeSteerToTargetRequest {
    /// Persisted trained atoms (decoder + basis schema).
    pub atoms: Vec<SaeOosAtomSpec>,
    /// TRAINED per-row on-manifold coordinates, one `(n_obs, latent_dim)` block per atom.
    pub coords: Vec<Array2<f64>>,
    /// TRAINED per-row routing logits, `(n_obs, k_atoms)`.
    pub logits: Array2<f64>,
    /// Assignment family.
    pub assignment: SaeOosAssignmentKind,
    /// Saved active-set support (required for `TopK`).
    pub top_k: Option<usize>,
    /// Ordered Beta--Bernoulli concentration α.
    pub alpha: f64,
    /// Softmax / gate temperature.
    pub tau: f64,
    /// The per-row output-Fisher metric the dose is measured through, or `None`.
    pub fisher_metric: Option<gam_problem::RowMetric>,
    /// The atom whose coordinate is being steered.
    pub atom_k: usize,
    /// Exact fitted row whose output-Fisher block prices the applied move.
    pub metric_row: usize,
    /// Source on-manifold coordinate.
    pub t_from: Vec<f64>,
    /// Target on-manifold coordinate (fixes the chord DIRECTION; the amplitude is solved).
    pub t_to: Vec<f64>,
    /// Requested output-KL dose in nats.
    pub target_nats: f64,
    /// Closed-loop correction tuning.
    pub config: TargetDoseConfig,
}

/// Solve for the amplitude that realizes `target_nats` on atom `atom_k`'s chord.
/// The optional plan-aware probe drives the bracketed closed-loop correction and
/// records exact directional local-Fisher and patched-forward measurements for
/// the same effective delta; with `probe = None` the returned [`TargetDosePlan`]
/// is the exact-factor-only closed-form seed.
pub fn run_sae_manifold_steer_to_target(
    request: SaeSteerToTargetRequest,
    probe: Option<&mut AppliedDoseProbe<'_>>,
) -> Result<TargetDosePlan, String> {
    let SaeSteerToTargetRequest {
        atoms,
        coords,
        logits,
        assignment,
        top_k,
        alpha,
        tau,
        fisher_metric,
        atom_k,
        metric_row,
        t_from,
        t_to,
        target_nats,
        config,
    } = request;
    let term = build_steer_term(SteerTermRequest {
        caller: "run_sae_manifold_steer_to_target",
        atom_specs: atoms,
        coords,
        logits,
        assignment,
        top_k,
        alpha,
        tau,
        fisher_metric,
        atom_k,
        metric_row,
    })?;
    let euclidean = gam_problem::RowMetric::euclidean(term.n_obs(), term.output_dim())?;
    let metric = term.row_metric().unwrap_or(&euclidean);
    steer_to_target_nats(
        &term,
        metric,
        TargetDoseRequest {
            atom_k,
            metric_row,
            t_from: &t_from,
            t_to: &t_to,
            target_nats,
            config,
        },
        probe,
    )
    .map_err(|error| error.to_string())
}

/// Fully owned, Python-free request to certify an externally-trained
/// (torch-lane) SAE-manifold state (#2266 / #2263 item 4). Rebuilds the SAME
/// frozen dictionary [`run_sae_manifold_oos`] / [`run_sae_manifold_steer`]
/// rebuild from [`SaeOosAtomSpec`] + trained coordinates/logits — no
/// coordinate or decoder solve; the caller's own (e.g. torch) training loop
/// already produced them. It maps physical target/decoder columns into the
/// explicitly declared Tier-0 frame, installs that exact state as
/// [`SaeCertifyRequest::base_term`], and delegates to
/// [`run_sae_manifold_certify`] for the shared post-fit diagnostics /
/// anytime-valid structure certificate pipeline. This is the entry a torch-lane
/// fit uses to obtain the same certificates a native closed-form fit gets,
/// without pretending a stationarity certificate exists for state this entry
/// never optimized.
pub struct SaeCertifyExternalRequest {
    /// The training target in the same physical output frame exposed by the
    /// persisted decoder blocks.
    pub target: Array2<f64>,
    /// Persisted trained atoms (decoder + basis schema), identical to the OOS
    /// dictionary definition.
    pub atoms: Vec<SaeOosAtomSpec>,
    /// Optional shared Tier-0 mean peeled before training. When present, the
    /// installed state is audited against `target - mean`, then certified
    /// reconstructions are lifted back into the physical frame.
    pub tier0_mean: Option<Array1<f64>>,
    /// Optional positive Tier-0 column scale used during training. Persisted
    /// decoder blocks are physical, so both target and decoder columns are
    /// divided by this scale before the exact installed-state audit.
    pub tier0_scale: Option<Array1<f64>>,
    /// The model's TRAINED per-row on-manifold coordinates, one `(n_obs,
    /// latent_dim)` block per atom.
    pub coords: Vec<Array2<f64>>,
    /// The model's TRAINED per-row routing logits, `(n_obs, k_atoms)`.
    pub logits: Array2<f64>,
    /// Assignment family.
    pub assignment: SaeOosAssignmentKind,
    /// Saved active-set support; required for `TopK`.
    pub top_k: Option<usize>,
    /// Ordered Beta-Bernoulli concentration α (ignored outside that assignment).
    pub alpha: f64,
    /// Softmax / gate temperature.
    pub tau: f64,
    /// The trained terminal regularization state (same contract as
    /// [`run_sae_manifold_oos`]: the certificate must be built at the
    /// regularization that produced the decoder, not a default).
    pub regularization: SaeOosRegularization,
    /// The per-row output-Fisher metric to certify dosimetry through, or
    /// `None` for the geometry-only Euclidean metric.
    pub fisher_metric: Option<gam_problem::RowMetric>,
    /// Pre-built analytic-penalty registry (built by the caller above
    /// `gam-sae`, identical contract to [`SaeCertifyRequest::registry`]).
    pub registry: AnalyticPenaltyRegistry,
    pub max_iter: usize,
    pub learning_rate: f64,
    pub ridge_ext_coord: f64,
    pub ridge_beta: f64,
    pub isometry_pin_active: bool,
    pub metric_provenance: &'static str,
    /// #977/#997 evidence-guarded structure search around the installed
    /// state (same default-true contract as [`SaeCertifyRequest`]).
    pub run_structure_search: bool,
}

/// Certify an externally-trained (torch-lane) SAE-manifold fit with no
/// closed-form solve. Shares its validation and term-rebuild with
/// [`run_sae_manifold_oos`] / [`run_sae_manifold_steer`] (same
/// [`SaeOosAtomSpec`] contract) so a torch-lane caller's decoder/coords/logits
/// rebuild into the identical dictionary a native fit or OOS encode would, then
/// hands the exact rebuilt internal-frame term to
/// [`run_sae_manifold_certify`].
pub fn run_sae_manifold_certify_external(
    request: SaeCertifyExternalRequest,
) -> Result<super::SaeExternalCertificationOutcome, SaeFitError> {
    let SaeCertifyExternalRequest {
        mut target,
        atoms: mut atom_specs,
        tier0_mean,
        tier0_scale,
        coords,
        logits,
        assignment,
        top_k,
        alpha,
        tau,
        regularization,
        fisher_metric,
        registry,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        isometry_pin_active,
        metric_provenance,
        run_structure_search,
    } = request;
    let k_atoms = atom_specs.len();
    if k_atoms == 0 {
        return Err(
            "run_sae_manifold_certify_external: at least one atom is required"
                .to_string()
                .into(),
        );
    }
    if coords.len() != k_atoms {
        return Err(format!(
            "run_sae_manifold_certify_external: coords must have K={k_atoms} per-atom blocks; got {}",
            coords.len()
        )
        .into());
    }
    let (n_obs, p_out) = target.dim();
    if n_obs == 0 || p_out == 0 {
        return Err(format!(
            "run_sae_manifold_certify_external: n_obs and p_out must be positive; got ({n_obs}, {p_out})"
        )
        .into());
    }
    if !target.iter().all(|value| value.is_finite()) {
        return Err("run_sae_manifold_certify_external: target must be finite"
            .to_string()
            .into());
    }
    if let Some(mean) = tier0_mean.as_ref() {
        if mean.len() != p_out || !mean.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "run_sae_manifold_certify_external: tier0_mean must be a finite length-{p_out} vector; got length {}",
                mean.len()
            )
            .into());
        }
        for mut row in target.rows_mut() {
            row -= mean;
        }
    }
    if let Some(scale) = tier0_scale.as_ref() {
        if scale.len() != p_out || !scale.iter().all(|value| value.is_finite() && *value > 0.0) {
            return Err(format!(
                "run_sae_manifold_certify_external: tier0_scale must be a finite positive length-{p_out} vector; got length {}",
                scale.len()
            )
            .into());
        }
        for (atom_index, spec) in atom_specs.iter_mut().enumerate() {
            if spec.decoder.ncols() != p_out {
                return Err(format!(
                    "run_sae_manifold_certify_external: atoms[{atom_index}] decoder must have {p_out} output columns before Tier-0 conversion; got {}",
                    spec.decoder.ncols()
                )
                .into());
            }
            for (column, value) in spec.decoder.columns_mut().into_iter().zip(scale) {
                for coefficient in column {
                    *coefficient /= *value;
                }
            }
        }
        for mut row in target.rows_mut() {
            row /= scale;
        }
    }
    if logits.dim() != (n_obs, k_atoms) || !logits.iter().all(|value| value.is_finite()) {
        return Err(format!(
            "run_sae_manifold_certify_external: logits must be a finite ({n_obs}, {k_atoms}) matrix; got {:?}",
            logits.dim()
        )
        .into());
    }
    if let Some(support) = top_k {
        if support == 0 || support > k_atoms {
            return Err(format!(
                "run_sae_manifold_certify_external: top_k must be in 1..={k_atoms}; got {support}"
            )
            .into());
        }
    }
    match (assignment, top_k) {
        (SaeOosAssignmentKind::TopK, None) => {
            return Err(
                "run_sae_manifold_certify_external: TopK assignment requires the saved top_k support size"
                    .to_string()
                    .into(),
            );
        }
        (SaeOosAssignmentKind::TopK, Some(_)) | (_, None) => {}
        (_, Some(support)) => {
            return Err(format!(
                "run_sae_manifold_certify_external: top_k={support} is valid only for TopK assignment"
            )
            .into());
        }
    }
    finite_positive("alpha", alpha)?;
    finite_positive("tau", tau)?;
    let mode = match assignment {
        SaeOosAssignmentKind::Softmax => AssignmentMode::softmax(tau),
        SaeOosAssignmentKind::OrderedBetaBernoulli { learnable_alpha } => {
            AssignmentMode::ordered_beta_bernoulli(tau, alpha, learnable_alpha)
        }
        SaeOosAssignmentKind::ThresholdGate { threshold } => {
            if !threshold.is_finite() {
                return Err(format!(
                    "run_sae_manifold_certify_external: threshold-gate threshold must be finite; got {threshold}"
                )
                .into());
            }
            AssignmentMode::threshold_gate(tau, threshold)
        }
        SaeOosAssignmentKind::TopK => {
            let support = top_k.ok_or_else(|| {
                "run_sae_manifold_certify_external: TopK assignment requires the saved top_k support size"
                    .to_string()
            })?;
            AssignmentMode::top_k_support(support)
        }
    };

    let latent_dims: Vec<usize> = atom_specs.iter().map(SaeOosAtomSpec::latent_dim).collect();
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    let mut atoms = Vec::with_capacity(k_atoms);
    for (atom_index, spec) in atom_specs.iter().enumerate() {
        let block = &coords[atom_index];
        if block.dim() != (n_obs, spec.latent_dim()) {
            return Err(format!(
                "run_sae_manifold_certify_external: coords[{atom_index}] must be (N, d)=({n_obs}, {}); got {:?}",
                spec.latent_dim(),
                block.dim()
            )
            .into());
        }
        if !block.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "run_sae_manifold_certify_external: coords[{atom_index}] contains non-finite values"
            )
            .into());
        }
        atoms.push(build_oos_atom(atom_index, spec, block.view(), p_out)?);
        coord_blocks.push(block.clone());
    }
    let manifolds = atom_specs
        .iter()
        .map(|spec| spec.basis_kind().latent_manifold(spec.latent_dim()))
        .collect();
    let assignment_state =
        SaeAssignment::from_blocks_with_mode_and_manifolds(logits, coord_blocks, manifolds, mode)?;
    let mut base_term = SaeManifoldTerm::new(atoms, assignment_state)?;
    if let Some(metric) = fisher_metric {
        base_term.set_row_metric(metric)?;
    }

    let initial_rho = build_rho(regularization, &latent_dims, mode)?;

    let outcome = run_sae_manifold_certify(SaeCertifyRequest {
        base_term,
        target,
        registry,
        initial_rho,
        max_iter,
        learning_rate,
        ridge_ext_coord,
        ridge_beta,
        alpha,
        isometry_pin_active,
        metric_provenance,
        run_structure_search,
    })?;
    let mut report = match outcome {
        super::SaeExternalCertificationOutcome::Certified(report) => report,
        super::SaeExternalCertificationOutcome::NonStationary(report) => {
            return Ok(super::SaeExternalCertificationOutcome::NonStationary(
                report,
            ));
        }
    };
    if let Some(mean) = tier0_mean.as_ref() {
        report
            .term
            .set_tier0_mean(mean.clone())
            .map_err(SaeFitError::Fit)?;
    }
    if let Some(scale) = tier0_scale.as_ref() {
        report
            .term
            .set_tier0_scale(scale.clone())
            .map_err(SaeFitError::Fit)?;
    }
    for mut row in report.fitted.rows_mut() {
        if let Some(scale) = tier0_scale.as_ref() {
            row *= scale;
        }
        if let Some(mean) = tier0_mean.as_ref() {
            row += mean;
        }
    }
    Ok(super::SaeExternalCertificationOutcome::Certified(report))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn periodic_request() -> SaeOosRequest {
        let coords = Array3::from_shape_vec((1, 4, 1), vec![0.0, 0.25, 0.5, 0.75]).unwrap();
        let target =
            Array2::from_shape_vec((4, 2), vec![0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0]).unwrap();
        SaeOosRequest {
            target,
            atoms: vec![
                SaeOosAtomSpec::new(
                    SaeAtomGeometryPlan::new(
                        SaeAtomBasisKind::Periodic,
                        1,
                        crate::manifold::SaeBasisResolution::PeriodicHarmonics { order: 1 },
                        crate::manifold::SaeReferenceMetricPlan::UnitCircle,
                    )
                    .unwrap(),
                    Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap(),
                )
                .unwrap(),
            ],
            assignment: SaeOosAssignmentKind::Softmax,
            alpha: 1.0,
            tau: 0.5,
            regularization: SaeOosRegularization {
                log_lambda_sparse: 0.01_f64.ln(),
                log_lambda_smooth: vec![0.01_f64.ln()],
                log_ard: vec![Vec::new()],
            },
            max_iter: 1,
            learning_rate: 1.0,
            ridge_ext_coord: 1.0e-6,
            initial_logits: Some(Array2::zeros((4, 1))),
            initial_coords: Some(coords),
            top_k: None,
            hybrid_linear_images: Vec::new(),
        }
    }

    fn assert_deck_twin_rebuild(
        geometry: SaeAtomGeometryPlan,
        coords: Array2<f64>,
        derivative_signs: [f64; 2],
    ) {
        let width = geometry.basis_size().unwrap();
        let spec = SaeOosAtomSpec::new(geometry.clone(), Array2::zeros((width, 1))).unwrap();
        let atom = build_oos_atom(0, &spec, coords.view(), 1).unwrap();
        assert_eq!(atom.geometry_plan(), Some(&geometry));
        assert_eq!(atom.basis_kind(), geometry.kind());
        assert_eq!(atom.latent_dim(), 2);
        assert_eq!(atom.smooth_penalty()[[0, 0]], 0.0);
        for column in 0..width {
            assert!(
                (atom.basis_values[[0, column]] - atom.basis_values[[1, column]]).abs() <= 1.0e-12,
                "deck-twin value mismatch at column {column}"
            );
            for axis in 0..2 {
                assert!(
                    (atom.basis_jacobian[[1, column, axis]]
                        - derivative_signs[axis] * atom.basis_jacobian[[0, column, axis]])
                    .abs()
                        <= 1.0e-11,
                    "deck-twin derivative mismatch at column {column}, axis {axis}"
                );
            }
        }
    }

    #[test]
    fn typed_oos_replays_projective_plane_deck_invariance() {
        let latitude = 0.31;
        let longitude = -0.47;
        let coords = Array2::from_shape_vec(
            (2, 2),
            vec![
                latitude,
                longitude,
                -latitude,
                longitude + std::f64::consts::PI,
            ],
        )
        .unwrap();
        assert_deck_twin_rebuild(
            SaeAtomGeometryPlan::projective_plane(2).unwrap(),
            coords,
            [-1.0, 1.0],
        );
    }

    #[test]
    fn typed_oos_replays_klein_bottle_deck_invariance() {
        let theta = 0.13;
        let phi = 0.27;
        let coords = Array2::from_shape_vec((2, 2), vec![theta, phi, theta + 0.5, -phi]).unwrap();
        assert_deck_twin_rebuild(
            SaeAtomGeometryPlan::klein_bottle(2).unwrap(),
            coords,
            [1.0, -1.0],
        );
    }

    #[test]
    fn typed_oos_spec_rejects_decoder_width_independent_of_plan() {
        let geometry = SaeAtomGeometryPlan::projective_plane(1).unwrap();
        let width = geometry.basis_size().unwrap();
        assert!(SaeOosAtomSpec::new(geometry, Array2::zeros((width + 1, 1))).is_err());
    }

    #[test]
    fn typed_oos_entry_reconstructs_frozen_periodic_dictionary() {
        let request = periodic_request();
        let expected = request.target.clone();
        let report = run_sae_manifold_oos(request).unwrap();
        assert_eq!(report.assignments.dim(), (4, 1));
        assert!(
            report
                .assignments
                .column(0)
                .iter()
                .all(|value| (*value - 1.0).abs() <= 1.0e-12)
        );
        assert_eq!(report.atoms.len(), 1);
        assert_eq!(report.atoms[0].active_dim, 1);
        let max_error = (&report.fitted - &expected)
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        assert!(max_error <= 1.0e-12, "max reconstruction error={max_error}");
        let atom_error = (&report.atoms[0].reconstruction - &report.fitted)
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            atom_error <= 1.0e-12,
            "unit-mass atom image must equal the fitted reconstruction; max error={atom_error}"
        );
    }

    #[test]
    fn typed_oos_report_uses_effective_hybrid_atom_image() {
        let mut request = periodic_request();
        request.hybrid_linear_images = vec![AtomLinearImage {
            atom_idx: 0,
            t_bar: 0.0,
            b0: Array1::from_vec(vec![2.0, -1.0]),
            b1: Array1::from_vec(vec![0.5, 0.25]),
            v: None,
        }];
        let report = run_sae_manifold_oos(request).unwrap();
        let max_error = (&report.atoms[0].reconstruction - &report.fitted)
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_error <= 1.0e-12,
            "unit-mass hybrid atom image must equal the hybrid fitted output; max error={max_error}"
        );
    }

    #[test]
    fn typed_oos_entry_rejects_decoder_schema_drift() {
        let mut request = periodic_request();
        request.atoms[0].basis_size = 5;
        let error = run_sae_manifold_oos(request).err().unwrap();
        assert!(error.contains("decoder shape"), "{error}");
    }

    #[test]
    fn typed_oos_entry_requires_complete_terminal_rho() {
        let mut request = periodic_request();
        request.regularization.log_ard.clear();
        let error = run_sae_manifold_oos(request).err().unwrap();
        assert!(
            error.contains("trained log_ard must contain 1 atom blocks"),
            "{error}"
        );
    }

    #[test]
    fn typed_steer_entry_restores_saved_topk_support() {
        let request = periodic_request();
        let coords = request
            .initial_coords
            .expect("periodic fixture has coordinates")
            .index_axis(ndarray::Axis(0), 0)
            .to_owned();
        let plan = run_sae_manifold_steer(SaeSteerRequest {
            atoms: request.atoms,
            coords: vec![coords],
            logits: request.initial_logits.expect("periodic fixture has logits"),
            assignment: SaeOosAssignmentKind::TopK,
            top_k: Some(1),
            alpha: 1.0,
            tau: 0.5,
            fisher_metric: None,
            atom_k: 0,
            metric_row: 0,
            amplitude: 1.0,
            t_from: vec![0.0],
            t_to: vec![0.25],
        })
        .expect("saved TopK support must make the trained dictionary steerable");
        assert_eq!(plan.delta.len(), 2);
        assert!(plan.predicted_nats.is_none());
    }

    #[test]
    fn pyffi_oos_boundary_stays_marshalling_only() {
        let source = include_str!("../../../gam-pyffi/src/latent/latent_basis_and_sae_ffi_tail.rs");
        assert!(
            !source.contains("fn predict_oos_from_arrays"),
            "OOS orchestration must not be reintroduced into gam-pyffi"
        );
        let start = source
            .find("fn sae_manifold_predict_oos")
            .expect("OOS pyfunction exists");
        let rest = &source[start..];
        let end = rest.find("/// (#1010)").expect("OOS pyfunction boundary");
        let binding = &rest[..end];
        assert!(binding.contains("run_sae_manifold_oos"));
        assert!(!binding.contains("SaeOosRegularization::Scalar"));
        for forbidden in [
            "run_fixed_decoder_arrow_schur(",
            "term_from_padded_blocks_with_mode(",
            "seed_coords_by_decoder_projection(",
        ] {
            assert!(
                !binding.contains(forbidden),
                "binding contains engine orchestration primitive {forbidden}"
            );
        }
    }
}

//! Python-free frozen-decoder out-of-sample SAE entry (#2236 Increment 2).
//!
//! [`run_sae_manifold_oos`] owns the complete inference operation: request
//! validation, basis/evaluator reconstruction from the persisted dictionary,
//! cold or warm coordinate and routing seeds, the fixed-decoder Arrow-Schur
//! solve, exact assignment reconstruction, collapse-aware reconstruction, and a
//! typed report. Bindings only translate their wire representation into
//! [`SaeOosRequest`] and serialize [`SaeOosReport`].

use std::sync::Arc;

use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;
use gam_terms::basis::{DuchonNullspaceOrder, duchon_sae_atom_penalty, monomial_exponents};
use ndarray::{Array1, Array2, Array3, ArrayView2, s};

use crate::hybrid_split::AtomLinearImage;
use crate::inference::steering::{SteerPlan, steer_delta};

use super::{
    AssignmentMode, CylinderHarmonicEvaluator, DuchonCoordinateEvaluator, EuclideanPatchEvaluator,
    MobiusHarmonicEvaluator, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
    SaeBasisEvaluator, SaeBasisSecondJet, SaeCertifyRequest, SaeFitError, SaeFitReport,
    SaeManifoldAtom, SaeManifoldLoss, SaeManifoldRho, SaeManifoldTerm, SaeStreamingPlan,
    SphereChartEvaluator, TorusHarmonicEvaluator, run_sae_manifold_certify,
    sae_pca_seed_initial_coords,
};

const SAE_MAX_PERIODIC_HARMONICS: usize = 4096;
const SAE_EUCLIDEAN_PATCH_RECOVERY_MAX_DEGREE: usize = 3;
const SAE_ACTIVE_ASSIGNMENT_MASS: f64 = 1.0e-8;

/// Persisted definition of one trained atom needed by frozen-decoder OOS
/// inference. `basis_size` is retained as an independent schema invariant and
/// must agree with the decoder row count and the rebuilt analytic basis width.
#[derive(Clone, Debug)]
pub struct SaeOosAtomSpec {
    pub basis_kind: SaeAtomBasisKind,
    pub latent_dim: usize,
    pub decoder: Array2<f64>,
    pub centers: Option<Array2<f64>>,
    pub n_harmonics: Option<usize>,
    pub basis_size: usize,
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

fn periodic_basis_size(n_harmonics: usize) -> Result<usize, String> {
    if n_harmonics == 0 || n_harmonics > SAE_MAX_PERIODIC_HARMONICS {
        return Err(format!(
            "run_sae_manifold_oos: periodic harmonic count must be in 1..={SAE_MAX_PERIODIC_HARMONICS}; got {n_harmonics}"
        ));
    }
    n_harmonics
        .checked_mul(2)
        .and_then(|twice| twice.checked_add(1))
        .ok_or_else(|| {
            format!("run_sae_manifold_oos: periodic basis width overflows for H={n_harmonics}")
        })
}

fn torus_basis_size(latent_dim: usize, n_harmonics: usize) -> Result<usize, String> {
    let axis_width = periodic_basis_size(n_harmonics)?;
    (0..latent_dim).try_fold(1usize, |width, _| {
        width.checked_mul(axis_width).ok_or_else(|| {
            format!(
                "run_sae_manifold_oos: torus basis width overflows for d={latent_dim}, H={n_harmonics}"
            )
        })
    })
}

fn duchon_atom_m(dim: usize) -> usize {
    dim / 2 + 2
}

fn duchon_nullspace_from_m(m: usize) -> DuchonNullspaceOrder {
    match m {
        1 => DuchonNullspaceOrder::Zero,
        2 => DuchonNullspaceOrder::Linear,
        other => DuchonNullspaceOrder::Degree(other - 1),
    }
}

fn euclidean_degree_for_basis_size(dim: usize, basis_size: usize) -> Result<usize, String> {
    (0..=SAE_EUCLIDEAN_PATCH_RECOVERY_MAX_DEGREE)
        .find(|&degree| monomial_exponents(dim, degree).len() == basis_size)
        .ok_or_else(|| {
            format!(
                "run_sae_manifold_oos: Euclidean basis width {basis_size} is not a degree <= {SAE_EUCLIDEAN_PATCH_RECOVERY_MAX_DEGREE} monomial basis in dimension {dim}"
            )
        })
}

fn cylinder_line_degree(harmonics: usize, basis_size: usize) -> Result<usize, String> {
    let circle_width = periodic_basis_size(harmonics)?;
    if basis_size == 0 || basis_size % circle_width != 0 {
        return Err(format!(
            "run_sae_manifold_oos: cylinder basis width {basis_size} is not divisible by persisted circle width {circle_width}"
        ));
    }
    let line_width = basis_size / circle_width;
    if line_width == 0 {
        return Err(format!(
            "run_sae_manifold_oos: cylinder line width must be positive; got {line_width}"
        ));
    }
    Ok(line_width - 1)
}

fn mobius_width_degree(harmonics: usize, basis_size: usize) -> Result<usize, String> {
    periodic_basis_size(harmonics)?;
    let mut candidates = Vec::new();
    for degree in 1..=SAE_EUCLIDEAN_PATCH_RECOVERY_MAX_DEGREE {
        let evaluator = MobiusHarmonicEvaluator::new(harmonics, degree)?;
        if evaluator.basis_size() == basis_size {
            candidates.push(degree);
        }
    }
    match candidates.as_slice() {
        [degree] => Ok(*degree),
        [] => Err(format!(
            "run_sae_manifold_oos: Mobius H={harmonics} has no width degree <= {SAE_EUCLIDEAN_PATCH_RECOVERY_MAX_DEGREE} matching basis width {basis_size}"
        )),
        _ => Err(format!(
            "run_sae_manifold_oos: Mobius H={harmonics}, width {basis_size} maps to multiple width degrees {candidates:?}"
        )),
    }
}

fn periodic_penalty(n_harmonics: usize) -> Result<Array2<f64>, String> {
    let m = periodic_basis_size(n_harmonics)?;
    let mut penalty = Array2::<f64>::zeros((m, m));
    penalty[[0, 0]] = 1.0e-8;
    for h in 1..=n_harmonics {
        let value = (h as f64).powi(4);
        penalty[[2 * h - 1, 2 * h - 1]] = value;
        penalty[[2 * h, 2 * h]] = value;
    }
    Ok(penalty)
}

fn sphere_penalty() -> Array2<f64> {
    let mut penalty = Array2::<f64>::eye(7);
    penalty[[0, 0]] = 1.0e-8;
    penalty
}

fn torus_penalty(evaluator: &TorusHarmonicEvaluator) -> Array2<f64> {
    let axis_m = evaluator.axis_basis_size();
    let latent_dim = evaluator.latent_dim;
    let m = evaluator.basis_size();
    let mut penalty = Array2::<f64>::zeros((m, m));
    let mut index = vec![0usize; latent_dim];
    for flat in 0..m {
        let squared_frequency: usize = index
            .iter()
            .map(|&axis_index| axis_index.div_ceil(2).pow(2))
            .sum();
        penalty[[flat, flat]] = if squared_frequency == 0 {
            1.0e-8
        } else {
            (squared_frequency as f64).powi(2)
        };
        for axis in (0..latent_dim).rev() {
            index[axis] += 1;
            if index[axis] < axis_m {
                break;
            }
            index[axis] = 0;
        }
    }
    penalty
}

fn euclidean_penalty(dim: usize, degree: usize) -> Array2<f64> {
    let exponents = monomial_exponents(dim, degree);
    let mut penalty = Array2::<f64>::zeros((exponents.len(), exponents.len()));
    for (col, exponent) in exponents.iter().enumerate() {
        if exponent.iter().any(|&power| power != 0) {
            penalty[[col, col]] = 1.0;
        }
    }
    penalty
}

fn analytic_roughness_penalty(
    evaluator: &dyn SaeBasisSecondJet,
    coords: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    let second = evaluator.second_jet(coords)?;
    let (_, m, d, d_again) = second.dim();
    if d != d_again {
        return Err(format!(
            "run_sae_manifold_oos: analytic second jet must be square in latent axes; got {:?}",
            second.dim()
        ));
    }
    let mut penalty = Array2::<f64>::zeros((m, m));
    for row in 0..second.shape()[0] {
        for a in 0..d {
            for b in 0..d {
                for mu in 0..m {
                    let h_mu = second[[row, mu, a, b]];
                    for nu in mu..m {
                        penalty[[mu, nu]] += h_mu * second[[row, nu, a, b]];
                    }
                }
            }
        }
    }
    for mu in 0..m {
        for nu in (mu + 1)..m {
            penalty[[nu, mu]] = penalty[[mu, nu]];
        }
    }
    Ok(penalty)
}

fn build_oos_atom(
    atom_index: usize,
    spec: &SaeOosAtomSpec,
    start_coords: ArrayView2<'_, f64>,
    p_out: usize,
) -> Result<SaeManifoldAtom, String> {
    if spec.latent_dim == 0 {
        return Err(format!(
            "run_sae_manifold_oos: atom {atom_index} latent_dim must be positive"
        ));
    }
    if spec.basis_size == 0 || spec.decoder.dim() != (spec.basis_size, p_out) {
        return Err(format!(
            "run_sae_manifold_oos: atom {atom_index} decoder shape {:?} must equal declared ({}, {p_out})",
            spec.decoder.dim(),
            spec.basis_size
        ));
    }
    if !spec.decoder.iter().all(|value| value.is_finite()) {
        return Err(format!(
            "run_sae_manifold_oos: atom {atom_index} decoder contains non-finite values"
        ));
    }
    if let Some(centers) = &spec.centers {
        if centers.ncols() != spec.latent_dim || !centers.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "run_sae_manifold_oos: atom {atom_index} centers must be finite with {} columns; got {:?}",
                spec.latent_dim,
                centers.dim()
            ));
        }
    }

    let (phi, jet, penalty, evaluator): (
        Array2<f64>,
        Array3<f64>,
        Array2<f64>,
        Arc<dyn SaeBasisSecondJet>,
    ) = match &spec.basis_kind {
        SaeAtomBasisKind::Periodic => {
            if spec.latent_dim != 1 {
                return Err(format!(
                    "run_sae_manifold_oos: periodic atom {atom_index} requires latent_dim=1; got {}",
                    spec.latent_dim
                ));
            }
            let harmonics = spec.n_harmonics.ok_or_else(|| {
                format!("run_sae_manifold_oos: periodic atom {atom_index} requires n_harmonics")
            })?;
            let expected = periodic_basis_size(harmonics)?;
            if expected != spec.basis_size {
                return Err(format!(
                    "run_sae_manifold_oos: periodic atom {atom_index} H={harmonics} builds width {expected}, not declared width {}",
                    spec.basis_size
                ));
            }
            let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(expected)?);
            let (phi, jet) = evaluator.evaluate(start_coords)?;
            (phi, jet, periodic_penalty(harmonics)?, evaluator)
        }
        SaeAtomBasisKind::Sphere => {
            if spec.latent_dim != 2 || spec.basis_size != 7 {
                return Err(format!(
                    "run_sae_manifold_oos: sphere atom {atom_index} requires latent_dim=2 and basis_size=7; got dim={}, width={}",
                    spec.latent_dim, spec.basis_size
                ));
            }
            if spec.n_harmonics.is_some() {
                return Err(format!(
                    "run_sae_manifold_oos: sphere atom {atom_index} must not carry harmonic metadata"
                ));
            }
            let evaluator = Arc::new(SphereChartEvaluator);
            let (phi, jet) = evaluator.evaluate(start_coords)?;
            (phi, jet, sphere_penalty(), evaluator)
        }
        SaeAtomBasisKind::Torus => {
            let harmonics = spec.n_harmonics.ok_or_else(|| {
                format!("run_sae_manifold_oos: torus atom {atom_index} requires n_harmonics")
            })?;
            let expected = torus_basis_size(spec.latent_dim, harmonics)?;
            if expected != spec.basis_size {
                return Err(format!(
                    "run_sae_manifold_oos: torus atom {atom_index} builds width {expected}, not declared width {}",
                    spec.basis_size,
                ));
            }
            let evaluator = Arc::new(TorusHarmonicEvaluator::new(spec.latent_dim, harmonics)?);
            let (phi, jet) = evaluator.evaluate(start_coords)?;
            let penalty = torus_penalty(&evaluator);
            (phi, jet, penalty, evaluator)
        }
        SaeAtomBasisKind::Duchon => {
            if spec.n_harmonics.is_some() {
                return Err(format!(
                    "run_sae_manifold_oos: Duchon atom {atom_index} must not carry harmonic metadata"
                ));
            }
            let centers = spec.centers.as_ref().ok_or_else(|| {
                format!("run_sae_manifold_oos: Duchon atom {atom_index} requires centers")
            })?;
            let m = duchon_atom_m(spec.latent_dim);
            let evaluator = Arc::new(DuchonCoordinateEvaluator::new(centers.clone(), m)?);
            let (phi, jet) = evaluator.evaluate(start_coords)?;
            let penalty = duchon_sae_atom_penalty(centers.view(), duchon_nullspace_from_m(m))
                .map_err(|error| error.to_string())?;
            (phi, jet, penalty, evaluator)
        }
        SaeAtomBasisKind::Linear
        | SaeAtomBasisKind::EuclideanPatch
        | SaeAtomBasisKind::Poincare => {
            if spec.n_harmonics.is_some() {
                return Err(format!(
                    "run_sae_manifold_oos: polynomial atom {atom_index} must not carry harmonic metadata"
                ));
            }
            let degree = euclidean_degree_for_basis_size(spec.latent_dim, spec.basis_size)?;
            let evaluator = Arc::new(EuclideanPatchEvaluator::new(spec.latent_dim, degree)?);
            let (phi, jet) = evaluator.evaluate(start_coords)?;
            let penalty = euclidean_penalty(spec.latent_dim, degree);
            (phi, jet, penalty, evaluator)
        }
        SaeAtomBasisKind::Cylinder => {
            if spec.latent_dim != 2 {
                return Err(format!(
                    "run_sae_manifold_oos: cylinder atom {atom_index} requires latent_dim=2; got {}",
                    spec.latent_dim
                ));
            }
            let harmonics = spec.n_harmonics.ok_or_else(|| {
                format!(
                    "run_sae_manifold_oos: cylinder atom {atom_index} requires persisted n_harmonics"
                )
            })?;
            let line_degree = cylinder_line_degree(harmonics, spec.basis_size)?;
            let evaluator = Arc::new(CylinderHarmonicEvaluator::new(harmonics, line_degree)?);
            let (phi, jet) = evaluator.evaluate(start_coords)?;
            let penalty = analytic_roughness_penalty(evaluator.as_ref(), start_coords)?;
            (phi, jet, penalty, evaluator)
        }
        SaeAtomBasisKind::Mobius => {
            if spec.latent_dim != 2 {
                return Err(format!(
                    "run_sae_manifold_oos: Mobius atom {atom_index} requires latent_dim=2; got {}",
                    spec.latent_dim
                ));
            }
            let harmonics = spec.n_harmonics.ok_or_else(|| {
                format!("run_sae_manifold_oos: Mobius atom {atom_index} requires n_harmonics")
            })?;
            let width_degree = mobius_width_degree(harmonics, spec.basis_size)?;
            let evaluator = Arc::new(MobiusHarmonicEvaluator::new(harmonics, width_degree)?);
            let (phi, jet) = evaluator.evaluate(start_coords)?;
            let penalty = analytic_roughness_penalty(evaluator.as_ref(), start_coords)?;
            (phi, jet, penalty, evaluator)
        }
        SaeAtomBasisKind::FiniteSet => {
            return Err(format!(
                "run_sae_manifold_oos: finite-set atom {atom_index} has no continuous OOS coordinate solve"
            ));
        }
        SaeAtomBasisKind::Precomputed(label) => {
            return Err(format!(
                "run_sae_manifold_oos: precomputed atom {atom_index} ({label:?}) has no analytic basis refresh"
            ));
        }
    };

    if phi.dim() != (start_coords.nrows(), spec.basis_size)
        || jet.dim() != (start_coords.nrows(), spec.basis_size, spec.latent_dim)
        || penalty.dim() != (spec.basis_size, spec.basis_size)
    {
        return Err(format!(
            "run_sae_manifold_oos: atom {atom_index} rebuilt shapes phi={:?}, jet={:?}, penalty={:?} disagree with (N={}, M={}, D={})",
            phi.dim(),
            jet.dim(),
            penalty.dim(),
            start_coords.nrows(),
            spec.basis_size,
            spec.latent_dim
        ));
    }

    Ok(SaeManifoldAtom::new_with_provided_function_gram(
        format!("oos_atom_{atom_index}"),
        spec.basis_kind.clone(),
        spec.latent_dim,
        phi,
        jet,
        spec.decoder.clone(),
        penalty,
    )?
    .with_basis_second_jet(evaluator))
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
        .map(|spec| spec.basis_kind.clone())
        .collect();
    let latent_dims: Vec<usize> = atom_specs.iter().map(|spec| spec.latent_dim).collect();
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
            .slice(s![atom_index, 0..n_obs, 0..spec.latent_dim])
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
    let fitted =
        term.reconstruct_from_assignments_target_aware(target.view(), assignments.view())?;

    let mut atom_reports = Vec::with_capacity(k_atoms);
    let mut decoded_row = vec![0.0_f64; p_out];
    for atom_index in 0..k_atoms {
        let mut reconstruction = Array2::<f64>::zeros((n_obs, p_out));
        for row in 0..n_obs {
            term.atoms[atom_index].fill_decoded_row(row, &mut decoded_row);
            for output in 0..p_out {
                reconstruction[[row, output]] = decoded_row[output];
            }
        }
        atom_reports.push(SaeOosAtomReport {
            basis_kind: atom_specs[atom_index].basis_kind.clone(),
            decoder: term.atoms[atom_index].decoder_coefficients.clone(),
            coords: term.assignment.coords[atom_index].as_matrix(),
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

/// Build the frozen trained dictionary into an [`SaeManifoldTerm`] pinned at its
/// TRAINED coordinates / logits and measure the steering plan for atom `atom_k`.
/// Mirrors the term rebuild of [`run_sae_manifold_oos`] (sharing
/// [`build_oos_atom`]) with no coordinate solve.
pub fn run_sae_manifold_steer(request: SaeSteerRequest) -> Result<SteerPlan, String> {
    let SaeSteerRequest {
        atoms: atom_specs,
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
    let k_atoms = atom_specs.len();
    if k_atoms == 0 {
        return Err("run_sae_manifold_steer: at least one atom is required".to_string());
    }
    if coords.len() != k_atoms {
        return Err(format!(
            "run_sae_manifold_steer: coords must have K={k_atoms} per-atom blocks; got {}",
            coords.len()
        ));
    }
    if atom_k >= k_atoms {
        return Err(format!(
            "run_sae_manifold_steer: atom_k={atom_k} out of range for K={k_atoms} atoms"
        ));
    }
    if metric_row >= logits.nrows() {
        return Err(format!(
            "run_sae_manifold_steer: metric_row={metric_row} out of range for {} fitted rows",
            logits.nrows()
        ));
    }
    finite_positive("amplitude", amplitude)?;
    finite_positive("alpha", alpha)?;
    finite_positive("tau", tau)?;
    let n_obs = logits.nrows();
    let p_out = atom_specs[0].decoder.ncols();
    if n_obs == 0 || p_out == 0 {
        return Err(format!(
            "run_sae_manifold_steer: n_obs and p_out must be positive; got ({n_obs}, {p_out})"
        ));
    }
    if logits.dim() != (n_obs, k_atoms) || !logits.iter().all(|value| value.is_finite()) {
        return Err(format!(
            "run_sae_manifold_steer: logits must be a finite ({n_obs}, {k_atoms}) matrix; got {:?}",
            logits.dim()
        ));
    }
    if let Some(support) = top_k {
        if support == 0 || support > k_atoms {
            return Err(format!(
                "run_sae_manifold_steer: top_k must be in 1..={k_atoms}; got {support}"
            ));
        }
    }
    match (assignment, top_k) {
        (SaeOosAssignmentKind::TopK, None) => {
            return Err(
                "run_sae_manifold_steer: TopK assignment requires the saved top_k support size"
                    .to_string(),
            );
        }
        (SaeOosAssignmentKind::TopK, Some(_)) | (_, None) => {}
        (_, Some(support)) => {
            return Err(format!(
                "run_sae_manifold_steer: top_k={support} is valid only for TopK assignment"
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
                    "run_sae_manifold_steer: threshold-gate threshold must be finite; got {threshold}"
                ));
            }
            AssignmentMode::threshold_gate(tau, threshold)
        }
        SaeOosAssignmentKind::TopK => {
            let support = top_k.ok_or_else(|| {
                "run_sae_manifold_steer: TopK assignment requires the saved top_k support size"
                    .to_string()
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
        if block.dim() != (n_obs, spec.latent_dim) {
            return Err(format!(
                "run_sae_manifold_steer: coords[{atom_index}] must be (N, d)=({n_obs}, {}); got {:?}",
                spec.latent_dim,
                block.dim()
            ));
        }
        if !block.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "run_sae_manifold_steer: coords[{atom_index}] contains non-finite values"
            ));
        }
        atoms.push(build_oos_atom(atom_index, spec, block.view(), p_out)?);
        coord_blocks.push(block.clone());
    }
    let manifolds = atom_specs
        .iter()
        .map(|spec| spec.basis_kind.latent_manifold(spec.latent_dim))
        .collect();
    let assignment_state =
        SaeAssignment::from_blocks_with_mode_and_manifolds(logits, coord_blocks, manifolds, mode)?;
    let mut term = SaeManifoldTerm::new(atoms, assignment_state)?;
    if let Some(metric) = fisher_metric {
        term.set_row_metric(metric)?;
    }

    // The metric the dose is measured through: the installed per-row metric, or a
    // bit-identical Euclidean metric (geometry-only; dose degrades to None).
    let euclidean = gam_problem::RowMetric::euclidean(n_obs, p_out)?;
    let metric = term.row_metric().unwrap_or(&euclidean);
    steer_delta(&term, metric, atom_k, metric_row, amplitude, &t_from, &t_to)
}

/// Fully owned, Python-free request to certify an externally-trained
/// (torch-lane) SAE-manifold state (#2266 / #2263 item 4). Rebuilds the SAME
/// frozen dictionary [`run_sae_manifold_oos`] / [`run_sae_manifold_steer`]
/// rebuild from [`SaeOosAtomSpec`] + trained coordinates/logits — no
/// coordinate or decoder solve; the caller's own (e.g. torch) training loop
/// already produced them — installs it VERBATIM as
/// [`SaeCertifyRequest::base_term`], and delegates to
/// [`run_sae_manifold_certify`] for the shared post-fit diagnostics /
/// anytime-valid structure certificate pipeline. This is the entry a torch-lane
/// fit uses to obtain the same certificates a native closed-form fit gets,
/// without pretending a stationarity certificate exists for state this entry
/// never optimized.
pub struct SaeCertifyExternalRequest {
    /// The training target the externally-trained decoder was fit against.
    pub target: Array2<f64>,
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
/// hands the rebuilt term to [`run_sae_manifold_certify`] verbatim.
pub fn run_sae_manifold_certify_external(
    request: SaeCertifyExternalRequest,
) -> Result<SaeFitReport, SaeFitError> {
    let SaeCertifyExternalRequest {
        target,
        atoms: atom_specs,
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

    let latent_dims: Vec<usize> = atom_specs.iter().map(|spec| spec.latent_dim).collect();
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    let mut atoms = Vec::with_capacity(k_atoms);
    for (atom_index, spec) in atom_specs.iter().enumerate() {
        let block = &coords[atom_index];
        if block.dim() != (n_obs, spec.latent_dim) {
            return Err(format!(
                "run_sae_manifold_certify_external: coords[{atom_index}] must be (N, d)=({n_obs}, {}); got {:?}",
                spec.latent_dim,
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
        .map(|spec| spec.basis_kind.latent_manifold(spec.latent_dim))
        .collect();
    let assignment_state =
        SaeAssignment::from_blocks_with_mode_and_manifolds(logits, coord_blocks, manifolds, mode)?;
    let mut base_term = SaeManifoldTerm::new(atoms, assignment_state)?;
    if let Some(metric) = fisher_metric {
        base_term.set_row_metric(metric)?;
    }

    let initial_rho = build_rho(regularization, &latent_dims, mode)?;

    run_sae_manifold_certify(SaeCertifyRequest {
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
        cancel: None,
    })
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
            atoms: vec![SaeOosAtomSpec {
                basis_kind: SaeAtomBasisKind::Periodic,
                latent_dim: 1,
                decoder: Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
                    .unwrap(),
                centers: None,
                n_harmonics: Some(1),
                basis_size: 3,
            }],
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

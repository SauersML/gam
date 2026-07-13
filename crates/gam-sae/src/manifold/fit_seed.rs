//! Typed construction of a fully configured SAE fit seed (issue #2236).
//!
//! This is the library-owned seam between borrowed caller arrays and
//! [`SaeFitRequest`]. It validates the padded seed, resolves assignment policy,
//! builds analytic evaluators and the base term, installs every per-fit switch,
//! refines cold routing, installs row metrics/weights, and derives the initial
//! rho state. Bindings only parse wire objects into the typed fields below.

use gam_problem::RowMetric;
use gam_terms::analytic_penalties::{AnalyticPenaltyKind, AnalyticPenaltyRegistry};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, ArrayView4, s};

use super::*;

/// Atom count at which native ARD switches from per-atom coordinates to one
/// shared coordinate per intrinsic axis.
pub const SAE_SHARED_ARD_K_THRESHOLD: usize = 256;

/// Strict typed assignment family for fit-seed construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaeFitAssignmentKind {
    Softmax,
    OrderedBetaBernoulli,
    ThresholdGate,
    TopK,
}

impl SaeFitAssignmentKind {
    /// Resolve one canonical public token through the shared strict schema.
    pub fn from_tag(tag: &str) -> Result<Self, String> {
        match crate::atom_schema::canonical_assignment_kind(tag)? {
            "softmax" => Ok(Self::Softmax),
            "ordered_beta_bernoulli" => Ok(Self::OrderedBetaBernoulli),
            "threshold_gate" => Ok(Self::ThresholdGate),
            "topk" => Ok(Self::TopK),
            canonical => Err(format!(
                "canonical assignment schema returned unsupported token {canonical:?}"
            )),
        }
    }

    pub const fn tag(self) -> &'static str {
        match self {
            Self::Softmax => "softmax",
            Self::OrderedBetaBernoulli => "ordered_beta_bernoulli",
            Self::ThresholdGate => "threshold_gate",
            Self::TopK => "topk",
        }
    }

    fn mode(
        self,
        tau: f64,
        alpha: f64,
        learnable_alpha: bool,
        threshold: f64,
        top_k: Option<usize>,
    ) -> Result<AssignmentMode, String> {
        match self {
            Self::Softmax | Self::OrderedBetaBernoulli | Self::ThresholdGate if top_k.is_some() => {
                Err(format!(
                    "top_k is valid only with assignment_kind 'topk'; got assignment_kind '{}'",
                    self.tag()
                ))
            }
            Self::Softmax => Ok(AssignmentMode::softmax(tau)),
            Self::OrderedBetaBernoulli => Ok(AssignmentMode::ordered_beta_bernoulli(
                tau,
                alpha,
                learnable_alpha,
            )),
            Self::ThresholdGate => Ok(AssignmentMode::threshold_gate(tau, threshold)),
            Self::TopK => top_k.map(AssignmentMode::top_k_support).ok_or_else(|| {
                "assignment_kind 'topk' requires top_k (the fixed per-row support size)".to_string()
            }),
        }
    }
}

/// Borrowed arrays and owned policy needed to construct one fit seed.
pub struct SaeFitSeedRequest<'a, 'context> {
    pub target: ArrayView2<'a, f64>,
    /// Complete immutable atom geometry. Kinds, dimensions, analytic
    /// resolutions, widths, centers, and reference metrics are derived from
    /// these plans; the padded arrays below carry no parallel metadata.
    pub geometry_plans: &'context [SaeAtomGeometryPlan],
    pub basis_values: ArrayView3<'a, f64>,
    pub basis_jacobian: ArrayView4<'a, f64>,
    pub decoder_coefficients: ArrayView3<'a, f64>,
    pub smooth_penalties: ArrayView3<'a, f64>,
    pub initial_logits: ArrayView2<'a, f64>,
    pub initial_coords: ArrayView3<'a, f64>,
    pub alpha: f64,
    pub tau: f64,
    pub learnable_alpha: bool,
    pub assignment_kind: SaeFitAssignmentKind,
    pub sparsity_strength: f64,
    pub smoothness: f64,
    pub max_iter: usize,
    pub learning_rate: f64,
    pub ridge_ext_coord: f64,
    pub ridge_beta: f64,
    pub top_k: Option<usize>,
    pub threshold: f64,
    pub native_ard_enabled: bool,
    pub seed_refine_routing: bool,
    pub seed_refine_random_state: u64,
    pub data_row_reseed: bool,
    pub fit_config: SaeFitConfig,
    pub temperature_schedule: Option<GumbelTemperatureSchedule>,
    pub fisher_metric: Option<SaeFisherRowMetricRequest<'a>>,
    pub row_loss_weights: Option<ArrayView1<'a, f64>>,
    pub registry: &'context AnalyticPenaltyRegistry,
}

/// Fully configured seed objects consumed by [`SaeFitRequest`].
pub struct SaeFitSeedReport {
    pub base_term: SaeManifoldTerm,
    pub initial_rho: SaeManifoldRho,
    pub isometry_pin_active: bool,
    pub metric_provenance: &'static str,
}

/// Admit one manifold fit shape through the assignment-aware front door.
pub fn admit_sae_fit_shape(
    n_obs: usize,
    p_out: usize,
    k_atoms: usize,
    d_max: usize,
    assignment_kind: SaeFitAssignmentKind,
    top_k: Option<usize>,
) -> Result<crate::front_door::SaeFitAdmission, String> {
    match assignment_kind {
        SaeFitAssignmentKind::TopK => {
            let support = top_k.ok_or_else(|| {
                "assignment_kind 'topk' requires top_k (the fixed per-row support size)".to_string()
            })?;
            crate::front_door::admit_topk_manifold(n_obs, p_out, k_atoms, d_max.max(1), support)
        }
        _ => crate::front_door::admit_dense_certification(n_obs, p_out, k_atoms),
    }
}

/// Validate and construct the complete Python-free seed for a SAE fit.
pub fn build_sae_fit_seed(request: SaeFitSeedRequest<'_, '_>) -> Result<SaeFitSeedReport, String> {
    let (n_obs, p_out) = request.target.dim();
    if n_obs == 0 || p_out == 0 {
        return Err("sae_manifold_fit requires a non-empty (N, p) response".to_string());
    }
    let k_atoms = request.geometry_plans.len();
    if k_atoms == 0 {
        return Err("sae_manifold_fit requires at least one atom".to_string());
    }
    let basis_sizes: Vec<usize> = request
        .geometry_plans
        .iter()
        .map(SaeAtomGeometryPlan::basis_size)
        .collect::<Result<_, _>>()?;
    let latent_dims: Vec<usize> = request
        .geometry_plans
        .iter()
        .map(SaeAtomGeometryPlan::latent_dim)
        .collect();
    let admission = admit_sae_fit_shape(
        n_obs,
        p_out,
        k_atoms,
        latent_dims.iter().copied().max().unwrap_or(1),
        request.assignment_kind,
        request.top_k,
    )?;
    if admission.lane != crate::front_door::SaeFitLane::DenseCertification {
        return Err(
            "build_sae_fit_seed is the dense-certification constructor; overcomplete hard-TopK requests must use the support-sparse seed entry"
                .to_string(),
        );
    }
    if request.max_iter < 1 {
        return Err(format!(
            "sae_manifold_fit requires max_iter >= 1; got {}",
            request.max_iter
        ));
    }
    match (request.assignment_kind, request.top_k) {
        (SaeFitAssignmentKind::TopK, Some(k_top)) if k_top > 0 && k_top <= k_atoms => {}
        (SaeFitAssignmentKind::TopK, Some(k_top)) => {
            return Err(format!(
                "assignment_kind 'topk' requires 1 <= top_k <= k_atoms={k_atoms}; got {k_top}"
            ));
        }
        (SaeFitAssignmentKind::TopK, None) => {
            return Err(
                "assignment_kind 'topk' requires top_k (the fixed per-row support size)"
                    .to_string(),
            );
        }
        (kind, Some(k_top)) => {
            return Err(format!(
                "top_k={k_top} is valid only with assignment_kind 'topk'; got assignment_kind '{}'",
                kind.tag()
            ));
        }
        (_, None) => {}
    }
    if request.initial_logits.dim() != (n_obs, k_atoms) {
        return Err(format!(
            "initial_logits must be ({n_obs}, {k_atoms}); got {:?}",
            request.initial_logits.dim()
        ));
    }
    for (name, value) in [
        ("alpha", request.alpha),
        ("tau", request.tau),
        ("learning_rate", request.learning_rate),
        ("ridge_ext_coord", request.ridge_ext_coord),
        ("ridge_beta", request.ridge_beta),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("{name} must be finite and positive; got {value}"));
        }
    }
    if !request.sparsity_strength.is_finite() || request.sparsity_strength <= 0.0 {
        return Err(format!(
            "sparsity_strength must be finite and positive; got {}",
            request.sparsity_strength
        ));
    }
    if !request.smoothness.is_finite() || request.smoothness <= 0.0 {
        return Err(format!(
            "smoothness must be finite and positive; got {}",
            request.smoothness
        ));
    }
    let sparsity_strength = request.sparsity_strength;
    let smoothness = request.smoothness;

    let basis_values_shape = request.basis_values.shape();
    if basis_values_shape[0] != k_atoms || basis_values_shape[1] != n_obs {
        return Err(format!(
            "basis_values must start with (K, N)=({k_atoms}, {n_obs}); got {:?}",
            basis_values_shape
        ));
    }
    let basis_jacobian_shape = request.basis_jacobian.shape();
    if basis_jacobian_shape[0] != k_atoms || basis_jacobian_shape[1] != n_obs {
        return Err(format!(
            "basis_jacobian must start with (K, N)=({k_atoms}, {n_obs}); got {:?}",
            basis_jacobian_shape
        ));
    }
    let decoder_shape = request.decoder_coefficients.shape();
    if decoder_shape[0] != k_atoms || decoder_shape[2] != p_out {
        return Err(format!(
            "decoder_coefficients must have shape (K, M_max, p)=({k_atoms}, M_max, {p_out}); got {:?}",
            decoder_shape
        ));
    }
    let smooth_shape = request.smooth_penalties.shape();
    if smooth_shape[0] != k_atoms || smooth_shape[1] != smooth_shape[2] {
        return Err(format!(
            "smooth_penalties must have shape (K, M_max, M_max); got {:?}",
            smooth_shape
        ));
    }
    let coords_shape = request.initial_coords.shape();
    if coords_shape[0] != k_atoms || coords_shape[1] != n_obs {
        return Err(format!(
            "initial_coords must start with (K, N)=({k_atoms}, {n_obs}); got {:?}",
            coords_shape
        ));
    }
    let max_dim = coords_shape[2];
    let mut coord_blocks = Vec::with_capacity(k_atoms);
    for atom_idx in 0..k_atoms {
        let d = latent_dims[atom_idx];
        let m = basis_sizes[atom_idx];
        if m > basis_values_shape[2]
            || m > basis_jacobian_shape[2]
            || m > decoder_shape[1]
            || m > smooth_shape[1]
        {
            return Err(format!(
                "geometry plan {atom_idx} derives basis width {m}, which exceeds one of the padded M_max dimensions"
            ));
        }
        if d > max_dim {
            return Err(format!(
                "geometry plan {atom_idx} derives latent_dim={d}, which exceeds initial_coords D_max={max_dim}"
            ));
        }
        if d > basis_jacobian_shape[3] {
            return Err(format!(
                "geometry plan {atom_idx} derives latent_dim={d}, which exceeds basis_jacobian D_max={}",
                basis_jacobian_shape[3]
            ));
        }
        coord_blocks.push(
            request
                .initial_coords
                .slice(s![atom_idx, 0..n_obs, 0..d])
                .to_owned(),
        );
    }
    let assignment_alpha = request
        .fit_config
        .ordered_beta_bernoulli_alpha_override
        .unwrap_or(request.alpha);
    let mode = request.assignment_kind.mode(
        request.tau,
        assignment_alpha,
        request.learnable_alpha,
        request.threshold,
        request.top_k,
    )?;
    let mut base_term = term_from_geometry_plans_with_mode(
        n_obs,
        p_out,
        request.geometry_plans,
        request.basis_values,
        request.basis_jacobian,
        request.decoder_coefficients,
        request.smooth_penalties,
        request.initial_logits,
        &coord_blocks,
        mode,
    )?;
    base_term.set_data_row_reseed(request.data_row_reseed);
    base_term.set_fit_config(request.fit_config);
    if let Some(schedule) = request.temperature_schedule {
        base_term.set_temperature_schedule(schedule)?;
    }

    if request.seed_refine_routing
        && k_atoms > 1
        && matches!(
            request.assignment_kind,
            SaeFitAssignmentKind::Softmax | SaeFitAssignmentKind::OrderedBetaBernoulli
        )
    {
        sae_refine_routing_seed(
            &mut base_term,
            request.target,
            &basis_sizes,
            request.assignment_kind.tag(),
            request.alpha,
            request.tau,
            request.threshold,
            request.seed_refine_random_state,
        )?;
    }

    let metric_provenance = if let Some(metric_request) = request.fisher_metric {
        let metric: RowMetric = build_sae_fisher_row_metric(metric_request)?;
        let label = metric_provenance_label(metric.provenance());
        base_term.set_row_metric(metric)?;
        label
    } else {
        "Euclidean"
    };
    if let Some(weights) = request.row_loss_weights {
        if weights.len() != n_obs {
            return Err(format!(
                "sae_manifold_fit: weights length {} must equal the {n_obs} response rows",
                weights.len()
            ));
        }
        base_term.set_row_loss_weights(weights.to_vec())?;
    }

    let log_ard: Vec<Array1<f64>> = latent_dims
        .iter()
        .map(|&d| {
            if request.native_ard_enabled {
                Array1::<f64>::zeros(d)
            } else {
                Array1::<f64>::zeros(0)
            }
        })
        .collect();
    let seed_dispersion = base_term.seed_reconstruction_dispersion(request.target)?;
    let use_shared_ard = request.native_ard_enabled && k_atoms >= SAE_SHARED_ARD_K_THRESHOLD;
    let initial_rho = if use_shared_ard {
        SaeManifoldRho::new_shared_ard(sparsity_strength.ln(), smoothness.ln(), log_ard)
    } else {
        SaeManifoldRho::new(sparsity_strength.ln(), smoothness.ln(), log_ard)
    }
    .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)?;
    let isometry_pin_active = request
        .registry
        .penalties
        .iter()
        .any(|penalty| matches!(penalty, AnalyticPenaltyKind::Isometry(_)));
    base_term.validate_heterogeneous_atom_compatibility(
        Some(request.registry),
        request.native_ard_enabled,
    )?;

    Ok(SaeFitSeedReport {
        base_term,
        initial_rho,
        isometry_pin_active,
        metric_provenance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assignment_kind_is_strict_and_typed() {
        assert_eq!(
            SaeFitAssignmentKind::from_tag("threshold_gate"),
            Ok(SaeFitAssignmentKind::ThresholdGate)
        );
    }

    #[test]
    fn seed_entry_rejects_empty_target_before_construction() {
        let geometry_plans = Vec::<SaeAtomGeometryPlan>::new();
        let registry = AnalyticPenaltyRegistry::new();
        let target = Array2::<f64>::zeros((0, 0));
        let basis_values = ndarray::Array3::<f64>::zeros((0, 0, 0));
        let basis_jacobian = ndarray::Array4::<f64>::zeros((0, 0, 0, 0));
        let decoder_coefficients = ndarray::Array3::<f64>::zeros((0, 0, 0));
        let smooth_penalties = ndarray::Array3::<f64>::zeros((0, 0, 0));
        let initial_logits = Array2::<f64>::zeros((0, 0));
        let initial_coords = ndarray::Array3::<f64>::zeros((0, 0, 0));
        let request = SaeFitSeedRequest {
            target: target.view(),
            geometry_plans: &geometry_plans,
            basis_values: basis_values.view(),
            basis_jacobian: basis_jacobian.view(),
            decoder_coefficients: decoder_coefficients.view(),
            smooth_penalties: smooth_penalties.view(),
            initial_logits: initial_logits.view(),
            initial_coords: initial_coords.view(),
            alpha: 1.0,
            tau: 1.0,
            learnable_alpha: false,
            assignment_kind: SaeFitAssignmentKind::Softmax,
            sparsity_strength: 1.0,
            smoothness: 1.0,
            max_iter: 1,
            learning_rate: 0.1,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            top_k: None,
            threshold: 0.0,
            native_ard_enabled: true,
            seed_refine_routing: false,
            seed_refine_random_state: 0,
            data_row_reseed: false,
            fit_config: SaeFitConfig::default(),
            temperature_schedule: None,
            fisher_metric: None,
            row_loss_weights: None,
            registry: &registry,
        };
        let error = build_sae_fit_seed(request)
            .err()
            .expect("empty target must fail");
        assert!(error.contains("non-empty"));
    }

    #[test]
    fn python_free_example_drives_the_core_fit_entry() {
        let example = include_str!("../../examples/sae_fit.rs");
        assert!(example.contains("build_sae_minimal_seed(SaeMinimalSeedRequest"));
        assert!(example.contains("build_sae_fit_seed(SaeFitSeedRequest"));
        assert!(example.contains("run_sae_manifold_fit(SaeFitRequest"));
        assert!(!example.contains("pyo3"));
        assert!(!example.contains("Python<"));
    }
}

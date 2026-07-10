//! Typed construction of the single-atom seed consumed by stagewise SAE.

use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;
use ndarray::{Array2, ArrayView2, ArrayView3, ArrayView4};

use super::*;

pub struct SaeStagewiseSeedRequest<'a> {
    pub target: ArrayView2<'a, f64>,
    pub atom_basis: &'a [String],
    pub atom_dim: &'a [usize],
    pub basis_values: ArrayView3<'a, f64>,
    pub basis_jacobian: ArrayView4<'a, f64>,
    pub basis_sizes: &'a [usize],
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
    pub structured_whitening: bool,
    pub fisher_metric: Option<SaeFisherRowMetricRequest<'a>>,
}

pub struct SaeStagewiseSeedReport {
    pub base_term: SaeManifoldTerm,
    pub initial_rho: SaeManifoldRho,
}

pub fn build_sae_stagewise_seed(
    request: SaeStagewiseSeedRequest<'_>,
) -> Result<SaeStagewiseSeedReport, String> {
    if request.atom_basis.len() != 1
        || request.atom_dim.len() != 1
        || request.basis_sizes.len() != 1
    {
        return Err(format!(
            "sae_manifold_fit_stagewise requires a single-atom seed; got atom_basis={}, atom_dim={}, basis_sizes={}",
            request.atom_basis.len(),
            request.atom_dim.len(),
            request.basis_sizes.len()
        ));
    }
    if request.assignment_kind == SaeFitAssignmentKind::TopK {
        return Err(
            "sae_manifold_fit_stagewise does not support topk because its request has no support-size field"
                .to_string(),
        );
    }
    let centers = vec![None::<Array2<f64>>];
    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target: request.target,
        atom_basis: request.atom_basis,
        atom_dim: request.atom_dim,
        atom_centers: &centers,
        basis_values: request.basis_values,
        basis_jacobian: request.basis_jacobian,
        basis_sizes: request.basis_sizes,
        decoder_coefficients: request.decoder_coefficients,
        smooth_penalties: request.smooth_penalties,
        initial_logits: request.initial_logits,
        initial_coords: request.initial_coords,
        alpha: request.alpha,
        tau: request.tau,
        learnable_alpha: request.learnable_alpha,
        assignment_kind: request.assignment_kind,
        sparsity_strength: request.sparsity_strength,
        smoothness: request.smoothness,
        max_iter: request.max_iter,
        learning_rate: request.learning_rate,
        ridge_ext_coord: request.ridge_ext_coord,
        ridge_beta: request.ridge_beta,
        top_k: None,
        threshold: 0.0,
        native_ard_enabled: true,
        seed_refine_routing: false,
        seed_refine_random_state: 0,
        data_row_reseed: false,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: request.fisher_metric,
        row_loss_weights: None,
        registry: &registry,
    })?;
    let mut base_term = seed.base_term;
    if request.structured_whitening
        && base_term
            .row_metric()
            .is_some_and(|metric| metric.whitens_likelihood())
    {
        return Err(
            "sae_manifold_fit_stagewise: behavioral_fisher conflicts with structured_whitening"
                .to_string(),
        );
    }
    Ok(SaeStagewiseSeedReport {
        base_term,
        initial_rho: seed.initial_rho,
    })
}

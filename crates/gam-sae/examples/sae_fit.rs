//! Minimal Python-free SAE manifold fit.
//!
//! The example deliberately exercises the same typed construction and fit
//! entries as the bindings: automatic atom seeding, validated term/rho
//! construction, then the single library-owned fit orchestration entry.

use gam_sae::manifold::{
    SaeFitAssignmentKind, SaeFitConfig, SaeFitRequest, SaeFitSeedReport, SaeFitSeedRequest,
    SaeMinimalSeedReport, SaeMinimalSeedRequest, build_sae_fit_seed, build_sae_minimal_seed,
    run_sae_manifold_fit,
};
use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = array![
        [1.0, 0.0],
        [0.7071067811865476, 0.7071067811865476],
        [0.0, 1.0],
        [-0.7071067811865476, 0.7071067811865476],
        [-1.0, 0.0],
        [-0.7071067811865476, -0.7071067811865476],
        [0.0, -1.0],
        [0.7071067811865476, -0.7071067811865476],
    ];
    let assignment_kind = SaeFitAssignmentKind::Softmax;
    let minimal = build_sae_minimal_seed(SaeMinimalSeedRequest {
        target: target.view(),
        atom_basis: vec!["periodic".to_string()],
        atom_dim: vec![1],
        assignment_kind,
        alpha: 1.0,
        tau: 1.0,
        threshold: 0.0,
        top_k: None,
        random_state: 0,
        initial_logits: None,
        initial_coords: None,
    })?;
    let SaeMinimalSeedReport {
        atom_basis,
        effective_atom_dim,
        atom_centers,
        basis_values,
        basis_jacobian,
        basis_sizes,
        decoder_coefficients,
        smooth_penalties,
        initial_logits,
        initial_coords,
        refine_routing,
    } = minimal;

    let registry = AnalyticPenaltyRegistry::new();
    let seed = build_sae_fit_seed(SaeFitSeedRequest {
        target: target.view(),
        atom_basis: &atom_basis,
        atom_dim: &effective_atom_dim,
        atom_centers: &atom_centers,
        basis_values: basis_values.view(),
        basis_jacobian: basis_jacobian.view(),
        basis_sizes: &basis_sizes,
        decoder_coefficients: decoder_coefficients.view(),
        smooth_penalties: smooth_penalties.view(),
        initial_logits: initial_logits.view(),
        initial_coords: initial_coords.view(),
        alpha: 1.0,
        tau: 1.0,
        learnable_alpha: false,
        assignment_kind,
        sparsity_strength: 1.0,
        smoothness: 1.0,
        max_iter: 4,
        learning_rate: 1.0,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        top_k: None,
        threshold: 0.0,
        native_ard_enabled: true,
        seed_refine_routing: refine_routing,
        seed_refine_random_state: 0,
        data_row_reseed: false,
        fit_config: SaeFitConfig::default(),
        temperature_schedule: None,
        fisher_metric: None,
        row_loss_weights: None,
        registry: &registry,
    })?;
    let SaeFitSeedReport {
        base_term,
        initial_rho,
        isometry_pin_active,
        metric_provenance,
    } = seed;

    let report = run_sae_manifold_fit(SaeFitRequest {
        base_term,
        target,
        registry,
        initial_rho,
        max_iter: 4,
        learning_rate: 1.0,
        ridge_ext_coord: 1.0e-6,
        ridge_beta: 1.0e-6,
        alpha: 1.0,
        isometry_pin_active,
        metric_provenance,
        promote_from_residual: false,
        run_structure_search: false,
        run_outer_rho_search: false,
        structured_residual_passes: 0,
        cancel: None,
    })?;

    println!("reconstruction_r2={:.6}", report.reconstruction_r2);
    Ok(())
}

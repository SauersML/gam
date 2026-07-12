//! #2266 — evaluation-only certification entry for externally-trained
//! (torch-lane) SAE-manifold state.
//!
//! [`crate::manifold::tests_tier0_primary_path_2023`] drives the full typed
//! seed → fit pipeline through [`run_sae_manifold_fit`]. This module proves
//! the sibling path: a caller that already has a fitted decoder / coordinates
//! / gate logits (e.g. produced by a torch-lane trainer, not this crate's
//! closed-form solve) can hand that state to [`run_sae_manifold_certify`]
//! and get the SAME post-fit diagnostics/certificates WITHOUT this crate
//! ever running an outer ρ search or an inner solve on it. The seed built by
//! `build_sae_fit_seed` stands in for the externally-trained state here —
//! `run_sae_manifold_certify` is never handed to `run_sae_manifold_fit`, so
//! the certified term is exactly the SEED, not a converged fit.

#[cfg(test)]
mod tests {
    use crate::manifold::{
        SaeCertifyRequest, SaeFitAssignmentKind, SaeFitConfig, SaeFitSeedReport, SaeFitSeedRequest,
        SaeMinimalSeedReport, SaeMinimalSeedRequest, SaeOuterVerdict, build_sae_fit_seed,
        build_sae_minimal_seed, run_sae_manifold_certify,
    };
    use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;
    use ndarray::Array2;

    const N_CIRCLE: usize = 64;
    const NOISE_SIGMA: f64 = 0.05;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn lcg_normal(state: &mut u64) -> f64 {
        let u1 = lcg(state).max(1e-12);
        let u2 = lcg(state);
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// `N_CIRCLE` evenly-spaced points on the unit circle plus small
    /// deterministic iid observation noise — the same minimal-seed target
    /// [`crate::manifold::tests_tier0_primary_path_2023`] uses, standing in
    /// here for an externally-produced (torch-lane) decoder's training data.
    fn circle_target() -> Array2<f64> {
        let mut state = 0x2266_0000_0000_0007u64;
        Array2::from_shape_fn((N_CIRCLE, 2), |(i, j)| {
            let theta = std::f64::consts::TAU * (i as f64) / (N_CIRCLE as f64);
            let clean = if j == 0 { theta.cos() } else { theta.sin() };
            clean + NOISE_SIGMA * lcg_normal(&mut state)
        })
    }

    /// Build a K=1 periodic-atom seed term via the minimal-seed pipeline
    /// (mirrors `tests_tier0_primary_path_2023::run_primary`'s seed
    /// construction) and hand it STRAIGHT to `run_sae_manifold_certify` — no
    /// call to `run_sae_manifold_fit` anywhere in this path, so the certified
    /// term is exactly the seed, standing in for an externally-trained state
    /// this crate never optimized.
    #[test]
    fn certify_external_seed_without_running_the_solve() {
        let target = circle_target();
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
        })
        .expect("minimal seed");
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
        })
        .expect("fit seed");
        let SaeFitSeedReport {
            base_term,
            initial_rho,
            isometry_pin_active,
            metric_provenance,
        } = seed;
        let k_atoms_seeded = base_term.k_atoms();

        let report = run_sae_manifold_certify(SaeCertifyRequest {
            base_term,
            target: target.clone(),
            registry,
            initial_rho,
            max_iter: 4,
            learning_rate: 1.0,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            alpha: 1.0,
            isometry_pin_active,
            metric_provenance,
            // Keep the test deterministic and cheap: the structure-search
            // path around a certified external state is exercised by the
            // production fit entry's own coverage; this test's contract is
            // the certify entry's postlude on the state AS PROVIDED.
            run_structure_search: false,
            cancel: None,
        })
        .expect("certify entry runs without invoking the closed-form solve");

        // #2266 — no outer search and no inner solve ran on this path.
        assert_eq!(
            report.outer_termination.verdict,
            SaeOuterVerdict::External,
            "the certify entry must report the External verdict, never a Search/FixedRho \
             certificate for state it never optimized"
        );
        assert_eq!(
            report.outer_termination.evals, 0,
            "no outer/inner evaluation ran on the certify path"
        );

        assert!(
            report.reconstruction_r2.is_finite(),
            "reconstruction R² must be finite"
        );
        assert!(
            !report.structure_certificate_json.is_empty(),
            "the anytime-valid structure certificate must serialize even when the \
             structure search did not run (a trivially-certifying empty ledger)"
        );
        assert_eq!(
            report.fitted.dim(),
            target.dim(),
            "the certified reconstruction must match the target's (N, p) shape"
        );
        assert_eq!(
            report.assignments.nrows(),
            target.nrows(),
            "assignments must carry one row per observation"
        );
        assert_eq!(
            report.term.k_atoms(),
            k_atoms_seeded,
            "with structure search disabled, the certified atom count is exactly the seed's"
        );
        assert!(
            report.loss.total().is_finite(),
            "the loss evaluated at the installed state must be finite"
        );
        assert!(
            report.penalized_quasi_laplace_criterion.is_finite(),
            "the penalized objective evaluated at the installed state must be finite"
        );
    }
}

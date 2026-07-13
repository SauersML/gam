//! #2263/#2266 — zero-optimization stationarity audit for externally-trained
//! SAE-manifold state.
//!
//! [`crate::manifold::tests_tier0_primary_path_2023`] drives the full typed
//! seed → fit pipeline through [`run_sae_manifold_fit`]. This module proves
//! the sibling path: arbitrary arrays are evaluated but cannot become a fit;
//! only an exact state independently certified by the native inner-KKT and
//! outer-criterion authorities reaches the post-fit report.

#[cfg(test)]
mod tests {
    use crate::inference::steering::steer_delta;
    use crate::manifold::{
        SaeCertifyRequest, SaeExternalCertificationOutcome, SaeFisherRowMetricRequest,
        SaeFitAssignmentKind, SaeFitConfig, SaeFitSeedReport, SaeFitSeedRequest,
        SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm, SaeMinimalSeedReport,
        SaeMinimalSeedRequest, SaeOuterVerdict, build_sae_fit_seed, build_sae_minimal_seed,
        run_sae_manifold_certify,
    };
    use gam_solve::rho_optimizer::{OuterProblem, OuterResult};
    use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;
    use ndarray::{Array2, Array3};

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

    fn seeded_external_fixture() -> (
        Array2<f64>,
        SaeManifoldTerm,
        SaeManifoldRho,
        bool,
        &'static str,
    ) {
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

        // #2266 dosimetry check: a torch-lane trainer's fit is normally paired
        // with an output-Fisher harvest shard (the SAME per-row metric a
        // native fit installs). Install one here so this test can assert the
        // certify entry's term is fully steer_delta-capable afterward — a
        // rank-1 factor is enough to make the metric carry "behavior"
        // (`MetricProvenance::OutputFisher`), which is the ONLY thing
        // `validity_radius`/`predicted_nats` gate on (see
        // `steering::metric_carries_behavior`); no closed-form-only state is
        // required beyond the fitted term + this metric.
        let p_out = target.ncols();
        let fisher_u3 =
            Array3::<f64>::from_shape_fn(
                (N_CIRCLE, p_out, 1),
                |(_, i, _)| {
                    if i == 0 { 1.0 } else { 0.0 }
                },
            );
        let fisher_metric_request = SaeFisherRowMetricRequest::from_tag(
            fisher_u3.view(),
            N_CIRCLE,
            p_out,
            None,
            Some("uncertified_approximation"),
            None,
        )
        .expect("rank-1 output-Fisher metric request");

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
            fisher_metric: Some(fisher_metric_request),
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
        (
            target,
            base_term,
            initial_rho,
            isometry_pin_active,
            metric_provenance,
        )
    }

    fn certify_request(
        target: Array2<f64>,
        base_term: SaeManifoldTerm,
        initial_rho: SaeManifoldRho,
        isometry_pin_active: bool,
        metric_provenance: &'static str,
    ) -> SaeCertifyRequest {
        SaeCertifyRequest {
            base_term,
            target,
            registry: AnalyticPenaltyRegistry::new(),
            initial_rho,
            max_iter: 40,
            learning_rate: 1.0,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            alpha: 1.0,
            isometry_pin_active,
            metric_provenance,
            run_structure_search: false,
        }
    }

    #[test]
    fn raw_external_seed_is_a_typed_nonfit() {
        let (target, term, rho, pin, provenance) = seeded_external_fixture();
        let outcome = run_sae_manifold_certify(certify_request(
            target,
            term,
            rho,
            pin,
            provenance,
        ))
        .expect("stationarity audit itself must evaluate");
        let SaeExternalCertificationOutcome::NonStationary(report) = outcome else {
            panic!("an unoptimized seed must never mint SaeFitReport");
        };
        assert!(!report.inner.certifies());
        assert_eq!(report.optimization_iterations, 0);
        assert!(report.reason.contains("inner KKT stationarity"));
    }

    fn native_converged_state() -> (Array2<f64>, SaeManifoldTerm, SaeManifoldRho, bool, &'static str) {
        let (target, term, rho, pin, provenance) = seeded_external_fixture();
        let rho_flat = rho.to_flat();
        let registry = AnalyticPenaltyRegistry::new();
        let mut objective = SaeManifoldOuterObjective::new(
            term,
            target.clone(),
            Some(registry),
            rho,
            40,
            1.0,
            1.0e-6,
            1.0e-6,
        );
        let result: OuterResult = OuterProblem::new(rho_flat.len())
            .with_initial_rho(rho_flat)
            .run(&mut objective, "#2263 native replay fixture")
            .expect("native outer search must run");
        assert!(result.converged, "native fixture must be genuinely converged");
        objective
            .certify_outer_result(&result)
            .expect("native result must carry the shared stationarity certificate");
        objective.remove_checkpoint();
        let fitted = objective.into_fitted().expect("certified native fit");
        (target, fitted.term, fitted.rho, pin, provenance)
    }

    #[test]
    fn converged_native_replay_passes_zero_optimization_audit_and_perturbation_fails() {
        let (target, term, rho, pin, provenance) = native_converged_state();
        let mut perturbed = term.clone();
        let mut beta = perturbed.flatten_beta();
        beta[0] += 0.25;
        perturbed
            .set_flat_beta(beta.view())
            .expect("perturb installed decoder");
        let perturbed_outcome = run_sae_manifold_certify(certify_request(
            target.clone(),
            perturbed,
            rho.clone(),
            pin,
            provenance,
        ))
        .expect("perturbed state must be evaluated");
        assert!(matches!(
            perturbed_outcome,
            SaeExternalCertificationOutcome::NonStationary(_)
        ));

        let outcome = run_sae_manifold_certify(certify_request(
            target.clone(),
            term,
            rho,
            pin,
            provenance,
        ))
        .expect("converged replay audit");
        let SaeExternalCertificationOutcome::Certified(report) = outcome else {
            panic!("a natively converged exact replay must pass the zero-step audit");
        };
        assert!(matches!(
            report.outer_termination.verdict,
            SaeOuterVerdict::Audited(_)
        ));
        assert_eq!(report.outer_termination.evals, 0);
        assert_eq!(report.fitted.dim(), target.dim());
        assert!(report.penalized_quasi_laplace_criterion.is_finite());

        let metric = report
            .term
            .row_metric()
            .expect("the installed output-Fisher metric must survive the certify entry verbatim");
        let plan = steer_delta(&report.term, metric, 0, 0, 0.1, &[0.0], &[0.05]).expect(
            "steer_delta must run on a certify-external term paired with a behavioral metric",
        );
        assert!(
            plan.validity_radius.is_some(),
            "validity_radius must be Some for a certify-external term + behavioral metric — \
             #2266's dosimetry contract needs the term + metric steer_delta reads, not a native \
             closed-form solve"
        );
        let radius = plan.validity_radius.expect("checked above");
        assert!(
            radius.is_finite() && radius > 0.0,
            "validity_radius must be a finite positive latent step length; got {radius}"
        );
        assert!(plan.predicted_nats.is_some());
    }
}

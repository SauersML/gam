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
        SaeFitAssignmentKind, SaeFitConfig, SaeFitError, SaeFitRequest, SaeFitSeedReport,
        SaeFitSeedRequest, SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm,
        SaeMinimalSeedReport, SaeMinimalSeedRequest, SaeOuterVerdict, build_sae_fit_seed,
        build_sae_minimal_seed, run_sae_manifold_certify, run_sae_manifold_fit,
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
            smoothness: 1.0,
            initial_logits: None,
            initial_coords: None,
        })
        .expect("minimal seed");
        let SaeMinimalSeedReport {
            geometry_plans,
            basis_values,
            basis_jacobian,
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
        // `predicted_nats` gates on (see
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
            assignment_kind,
            sparsity_strength: 1.0,
            smoothness: 1.0,
            max_iter: 4,
            learning_rate: 1.0,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            top_k: None,
            threshold: 0.0,
            seed_refine_routing: refine_routing,
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

    /// #2822 — the native entry refuses a frozen fit and names this module's route instead.
    /// `max_iter = 0` holds the inner state at the seed, so a native fit from it would not come
    /// from a converged optimization. Pricing a supplied state is `run_sae_manifold_certify`'s job.
    #[test]
    fn native_entry_refuses_a_frozen_fit_and_names_certify_2822() {
        let (target, term, rho, pin, provenance) = seeded_external_fixture();
        let outcome = run_sae_manifold_fit(SaeFitRequest {
            reconstruction_optimism_folds: None,
            base_term: term,
            target,
            registry: AnalyticPenaltyRegistry::new(),
            initial_rho: rho,
            max_iter: 0,
            learning_rate: 1.0,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            alpha: 1.0,
            isometry_pin_active: pin,
            metric_provenance: provenance,
            promote_from_residual: false,
            run_structure_search: false,
            run_outer_rho_search: true,
            structured_residual_passes: 0,
            cancel: None,
        });
        match outcome {
            Err(SaeFitError::InvalidRequest(message)) => assert!(
                message.contains("run_sae_manifold_certify"),
                "the refusal must name the certify entry, got: {message}"
            ),
            Err(other) => {
                panic!("a frozen native fit must be refused as an invalid request, got: {other}")
            }
            Ok(_) => panic!("a frozen native fit (max_iter = 0) must not produce a fit"),
        }
    }

    /// #2822 — a state converged in its tier-0 fit frame certifies against the raw target.
    /// A native fit runs on `(Z − μ)/σ`, installs μ/σ on the term it returns, and every reconstruction
    /// lifts back as `σ ⊙ x̂ + μ`, while certify receives the raw target. This builds that object
    /// without the native entry: the fixture converges through the native outer objective, an exact
    /// binary frame is installed on the converged term, and the raw target is `σ ⊙ Z + μ`. Auditing
    /// the fit-frame decoders against the raw target refused a converged fit (lane probe g4, job
    /// 1139818: NonStationary against raw Z, Certified against (Z − μ)/σ). The certified term must
    /// come back carrying the frame bit for bit, or its later reconstructions would not lift.
    #[test]
    fn framed_converged_state_certifies_against_its_raw_target_2822() {
        let (fit_target, mut term, rho, pin, provenance) = native_converged_state();
        let mean = ndarray::Array1::from(vec![3.0, -2.0]);
        let scale = ndarray::Array1::from(vec![4.0, 0.5]);
        let mut raw_target = fit_target;
        for mut row in raw_target.rows_mut() {
            row *= &scale;
            row += &mean;
        }
        term.set_tier0_mean(mean.clone())
            .expect("the frame mean has the output width");
        term.set_tier0_scale(scale.clone())
            .expect("the frame scale is finite and positive");
        let bits = |values: Option<&ndarray::Array1<f64>>| {
            values.map(|values| values.iter().map(|value| value.to_bits()).collect::<Vec<u64>>())
        };
        let installed_frame = (bits(Some(&mean)), bits(Some(&scale)));
        let outcome =
            run_sae_manifold_certify(certify_request(raw_target, term, rho, pin, provenance))
                .expect("the certify audit evaluates");
        match outcome {
            SaeExternalCertificationOutcome::Certified(report) => {
                assert!(
                    report.penalized_quasi_laplace_criterion.is_finite(),
                    "a certified report carries a finite criterion"
                );
                assert_eq!(
                    (bits(report.term.tier0_mean()), bits(report.term.tier0_scale())),
                    installed_frame,
                    "the certified term must carry the installed tier-0 frame bit for bit"
                );
            }
            SaeExternalCertificationOutcome::NonStationary(report) => panic!(
                "a state converged in its fit frame must certify against its raw target: {} \
                 (inner KKT certifies: {})",
                report.reason,
                report.inner.certifies()
            ),
        }
    }

    #[test]
    fn raw_external_seed_is_a_typed_nonfit() {
        let (target, term, rho, pin, provenance) = seeded_external_fixture();
        let outcome = run_sae_manifold_certify(certify_request(target, term, rho, pin, provenance))
            .expect("stationarity audit itself must evaluate");
        let SaeExternalCertificationOutcome::NonStationary(report) = outcome else {
            panic!("an unoptimized seed must never mint SaeFitReport");
        };
        assert!(!report.inner.certifies());
        assert_eq!(report.optimization_iterations, 0);
        assert!(report.reason.contains("inner KKT stationarity"));
    }

    fn native_converged_state() -> (
        Array2<f64>,
        SaeManifoldTerm,
        SaeManifoldRho,
        bool,
        &'static str,
    ) {
        let (target, term, rho, pin, provenance) = seeded_external_fixture();
        let rho_flat = rho.to_flat(&term.assignment).expect("the seed rho is bound to the term's assignment");
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
        assert!(
            result.converged(),
            "native fixture must be genuinely converged"
        );
        objective
            .certify_outer_result(&result)
            .expect("native result must carry the shared stationarity certificate");
        let fitted = objective.into_fitted().expect("certified native fit");
        (target, fitted.term, fitted.rho, pin, provenance)
    }

    #[test]
    fn lc22_probe_3474() {
        use gam_solve::rho_optimizer::OuterObjective;
        let _ = env_logger::builder().is_test(true).try_init();
        let (target, term, rho, _pin, _provenance) = seeded_external_fixture();
        let rho_flat = rho.to_flat(&term.assignment).expect("flat");
        eprintln!("[probe] seed rho = {rho_flat:?}");
        let registry = AnalyticPenaltyRegistry::new();
        let mut objective = SaeManifoldOuterObjective::new(
            term, target.clone(), Some(registry), rho, 40, 1.0, 1.0e-6, 1.0e-6,
        );
        let pts: Vec<[f64; 2]> = std::env::var("LC22_PTS")
            .ok()
            .map(|s| {
                s.split(';')
                    .map(|p| {
                        let v: Vec<f64> = p.split(',').map(|x| x.trim().parse().unwrap()).collect();
                        [v[0], v[1]]
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![[rho_flat[0], rho_flat[1]], [4.0, rho_flat[1]], [8.0, -4.0], [12.0, -6.0], [16.867491342553677, -7.261040811383532]]);
        for p in pts {
            let r = ndarray::Array1::from(vec![p[0], p[1]]);
            let e = objective.eval(&r);
            match e {
                Ok(ev) => {
                    let h = 1e-4;
                    let mut fd = vec![];
                    for j in 0..2 {
                        let mut rp = r.clone();
                        rp[j] += h;
                        let mut rm = r.clone();
                        rm[j] -= h;
                        let fp = objective.eval(&rp).map(|e| e.cost).unwrap_or(f64::NAN);
                        let fm = objective.eval(&rm).map(|e| e.cost).unwrap_or(f64::NAN);
                        fd.push((fp - fm) / (2.0 * h));
                    }
                    eprintln!(
                        "[probe] rho={p:?} cost={:.12e} grad={:?} fd={fd:?}",
                        ev.cost, ev.gradient
                    );
                }
                Err(err) => eprintln!("[probe] rho={p:?} error {err}"),
            }
        }
        if std::env::var("LC22_NORUN").is_err() {
            let result = OuterProblem::new(rho_flat.len())
                .with_initial_rho(rho_flat)
                .run(&mut objective, "#2263 native replay fixture");
            match result {
                Ok(r) => eprintln!("[probe] run ok rho={:?} converged={}", r.rho, r.converged()),
                Err(e) => eprintln!("[probe] run err {e}"),
            }
        }
    }

    struct Lc22CriterionLogger;
    impl log::Log for Lc22CriterionLogger {
        fn enabled(&self, _m: &log::Metadata) -> bool {
            true
        }
        fn log(&self, record: &log::Record) {
            let msg = format!("{}", record.args());
            if msg.starts_with("[SAE-CRITERION]") || msg.contains("rank") && msg.contains("MP") {
                eprintln!("[probe3 log] {msg}");
            }
        }
        fn flush(&self) {}
    }

    #[test]
    fn lc22_probe3_3474() {
        use gam_solve::rho_optimizer::OuterObjective;
        let _ = log::set_boxed_logger(Box::new(Lc22CriterionLogger));
        log::set_max_level(log::LevelFilter::Debug);
        let fresh = || {
            let (target, term, rho, _pin, _provenance) = seeded_external_fixture();
            SaeManifoldOuterObjective::new(
                term, target, Some(AnalyticPenaltyRegistry::new()), rho, 40, 1.0, 1.0e-6, 1.0e-6,
            )
        };
        let pts = [
            [-6.054325069138625, -6.054325069138625],
            [10.0, -4.0],
            [10.5, -4.0],
            [11.0, -4.0],
            [11.5, -4.0],
            [12.0, -4.0],
            [12.0, -6.0],
            [16.867491342553677, -7.261040811383532],
            [20.0, -7.261040811383532],
            [22.185195809350546, -14.351306798629912],
        ];
        for p in pts {
            let mut o = fresh();
            eprintln!("[probe3] ---- rho={p:?}");
            match o.eval(&ndarray::Array1::from(vec![p[0], p[1]])) {
                Ok(ev) => eprintln!("[probe3] rho={p:?} cost={:.10e} g={:?}", ev.cost, ev.gradient.to_vec()),
                Err(e) => eprintln!("[probe3] rho={p:?} err {e}"),
            }
        }
    }

    #[test]
    fn lc22_probe4_3474() {
        use gam_solve::rho_optimizer::OuterObjective;
        let _ = log::set_boxed_logger(Box::new(Lc22CriterionLogger));
        log::set_max_level(log::LevelFilter::Debug);
        let (target, term, rho, _pin, _provenance) = seeded_external_fixture();
        eprintln!("[probe4] seed beta_dim={}", term.beta_dim());
        let rho_flat = rho.to_flat(&term.assignment).unwrap();
        let mut objective = SaeManifoldOuterObjective::new(
            term, target.clone(), Some(AnalyticPenaltyRegistry::new()), rho, 40, 1.0, 1.0e-6, 1.0e-6,
        );
        let result: OuterResult = OuterProblem::new(rho_flat.len())
            .with_initial_rho(rho_flat)
            .run(&mut objective, "probe4")
            .expect("run");
        eprintln!(
            "[probe4] run converged={} rho={:?} value={:.10e} grad={:?}",
            result.converged(),
            result.rho.to_vec(),
            result.final_value,
            result.final_measurement.as_ref().map(|m| m.gradient().to_vec())
        );
        if let Some(c) = result.criterion_certificate.as_ref() {
            eprintln!("[probe4] run certificate: {}", c.summary());
        }
        objective.certify_outer_result(&result).expect("certify");
        let fitted = objective.into_fitted().expect("fitted");
        let flat = fitted.rho.flat_coordinates();
        for prepare in [false, true] {
            let mut t = fitted.term.clone();
            let before = t.beta_dim();
            if prepare {
                t.prepare_entry_stages().expect("prepare");
            }
            eprintln!("[probe4] audit prepare={prepare} beta_dim {before} -> {}", t.beta_dim());
            let mut o = SaeManifoldOuterObjective::new(
                t, target.clone(), Some(AnalyticPenaltyRegistry::new()), fitted.rho.clone(), 0, 1.0, 1.0e-6, 1.0e-6,
            )
            .for_installed_state_audit();
            match o.eval(&flat) {
                Ok(ev) => eprintln!("[probe4] audit prepare={prepare} cost={:.10e} g={:?}", ev.cost, ev.gradient.to_vec()),
                Err(e) => eprintln!("[probe4] audit prepare={prepare} err {e}"),
            }
        }
        {
            let t = fitted.term.clone();
            let mut o = SaeManifoldOuterObjective::new(
                t, target.clone(), Some(AnalyticPenaltyRegistry::new()), fitted.rho.clone(), 40, 1.0, 1.0e-6, 1.0e-6,
            );
            match o.eval(&flat) {
                Ok(ev) => eprintln!("[probe4] live-from-fitted cost={:.10e} g={:?}", ev.cost, ev.gradient.to_vec()),
                Err(e) => eprintln!("[probe4] live-from-fitted err {e}"),
            }
        }
    }

    #[test]
    fn lc22_probe2_3474() {
        use gam_solve::rho_optimizer::OuterObjective;
        let fresh = || {
            let (target, term, rho, _pin, _provenance) = seeded_external_fixture();
            SaeManifoldOuterObjective::new(
                term, target, Some(AnalyticPenaltyRegistry::new()), rho, 40, 1.0, 1.0e-6, 1.0e-6,
            )
        };
        let xs = [4.0, 6.0, 7.0, 7.9, 7.99, 8.0, 8.01, 8.1, 9.0, 10.0, 12.0];
        for &x in &xs {
            let mut o = fresh();
            let r = ndarray::Array1::from(vec![x, -4.0]);
            match o.eval(&r) {
                Ok(ev) => eprintln!("[probe2 fresh] x={x} cost={:.12e} g0={:.6e} g1={:.6e}", ev.cost, ev.gradient[0], ev.gradient[1]),
                Err(e) => eprintln!("[probe2 fresh] x={x} err {e}"),
            }
        }
        let mut o = fresh();
        for &x in &xs {
            let r = ndarray::Array1::from(vec![x, -4.0]);
            match o.eval(&r) {
                Ok(ev) => eprintln!("[probe2 seq] x={x} cost={:.12e} g0={:.6e}", ev.cost, ev.gradient[0]),
                Err(e) => eprintln!("[probe2 seq] x={x} err {e}"),
            }
        }
        for &h in &[1e-2, 1e-3, 1e-4, 1e-5] {
            let mut o = fresh();
            let c = o.eval(&ndarray::Array1::from(vec![8.0, -4.0])).map(|e| (e.cost, e.gradient[0])).unwrap();
            let fp = o.eval(&ndarray::Array1::from(vec![8.0 + h, -4.0])).map(|e| e.cost).unwrap();
            let fm = o.eval(&ndarray::Array1::from(vec![8.0 - h, -4.0])).map(|e| e.cost).unwrap();
            let c2 = o.eval(&ndarray::Array1::from(vec![8.0, -4.0])).map(|e| (e.cost, e.gradient[0])).unwrap();
            eprintln!("[probe2 fd] h={h} c={:.12e} g0={:.6e} fd={:.6e} fwd={:.6e} bwd={:.6e} re-eval c={:.12e} g0={:.6e}", c.0, c.1, (fp - fm) / (2.0 * h), (fp - c.0) / h, (c.0 - fm) / h, c2.0, c2.1);
        }
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

        let outcome =
            run_sae_manifold_certify(certify_request(target.clone(), term, rho, pin, provenance))
                .expect("converged replay audit");
        let report = match outcome {
            SaeExternalCertificationOutcome::Certified(report) => report,
            SaeExternalCertificationOutcome::NonStationary(report) => panic!(
                "a natively converged exact replay must pass the zero-step audit; it was refused: \
                 {} (inner KKT certifies: {}, outer projected gradient {:?} against bound {:?}, \
                 {} optimization iterations)",
                report.reason,
                report.inner.certifies(),
                report.outer_projected_gradient_norm,
                report.outer_stationarity_bound,
                report.optimization_iterations
            ),
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
            plan.predicted_nats.is_some(),
            "predicted_nats must be Some for a certify-external term + behavioral metric — \
             #2266's dosimetry contract needs the term + metric steer_delta reads, not a native \
             closed-form solve"
        );
    }
}

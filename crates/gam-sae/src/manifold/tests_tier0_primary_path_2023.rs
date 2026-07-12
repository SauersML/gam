//! #2023 Increment 5 — Tier-0 shared mean is NATIVE to the ONE fit entry.
//!
//! [`crate::manifold::tests_tier0_shared_mean_2023`] proves the reconstruction /
//! survival mechanics for a HAND-BUILT term with the mean installed by hand. This
//! module proves the missing production half: the single fit entry
//! [`run_sae_manifold_fit`] itself peels the shared column mean off a raw target,
//! runs the whole fit on the de-meaned frame, and hands back a fitted term that
//! SELF-CONTAINS μ — so the tiered schedule's Tier-0 "seed policy" is realized by
//! the primary path, not a separate surface. Two properties, exercised through the
//! exact typed seed→fit pipeline the bindings drive (`examples/sae_fit.rs`):
//!
//! 1. A raw circle target carrying a large global DC gets μ installed on the
//!    returned term equal to the column mean, and the reported reconstruction is
//!    lifted back to raw-target space (its column mean matches the target's).
//! 2. An already-centered target yields μ ≈ 0 — always-on Tier-0 is a no-op on
//!    centered data (μ is computed from THIS target, so there is no
//!    double-subtraction hazard), the safety the C4 module documents.

#[cfg(test)]
mod tests {
    use crate::manifold::{
        SaeFitAssignmentKind, SaeFitConfig, SaeFitError, SaeFitRequest, SaeFitSeedReport,
        SaeFitSeedRequest, SaeMinimalSeedReport, SaeMinimalSeedRequest, build_sae_fit_seed,
        build_sae_minimal_seed, run_sae_manifold_fit,
    };
    use gam_terms::analytic_penalties::AnalyticPenaltyRegistry;
    use ndarray::{Array2, Axis};

    /// Test-visible forwarding logger: the engine's diagnostic channels
    /// (`log::debug!` bail naming in `terminal_exact_newton_polish`, the
    /// `log::warn!` incumbent/warranty restores) are silently dropped by the
    /// test harness unless a logger is installed — the exact trap that left
    /// the tier-0 refusal unadjudicated across multiple probe runs. eprintln
    /// is the test-side convention (`log::warn` is dropped in tests); this
    /// forwards every record so a plain `--nocapture` run shows the engine's
    /// own account of WHY it refused.
    struct ForwardingTestLogger;
    impl log::Log for ForwardingTestLogger {
        fn enabled(&self, _: &log::Metadata<'_>) -> bool {
            true
        }
        fn log(&self, record: &log::Record<'_>) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
        fn flush(&self) {}
    }
    static FORWARDING_TEST_LOGGER: ForwardingTestLogger = ForwardingTestLogger;
    fn install_test_logger() {
        // Ignore the error when another test already installed a logger.
        if log::set_logger(&FORWARDING_TEST_LOGGER).is_ok() {
            log::set_max_level(log::LevelFilter::Debug);
        }
    }

    /// `N_CIRCLE` evenly-spaced points on the unit circle, plus a per-dim constant
    /// `offset` (the global DC Tier-0 must carry) plus small deterministic iid
    /// observation noise.
    ///
    /// The noise is load-bearing for identifiability, NOT decoration. A K=1
    /// Periodic atom with harmonics reconstructs a clean circle `[cosθ, sinθ]` to
    /// machine precision, so the primary fit's pass-0 residual would be ≈0. The
    /// always-on structured-residual pass (`fit_entry.rs`, magic-by-default) then
    /// fits a residual-covariance model on that ≈0 residual: its idiosyncratic
    /// diagonal collapses to the denormal floor, whitening by ~1/D is near-singular,
    /// and the whitened-residual REML over the smoothing ρ has NO interior
    /// stationary point — so the outer optimizer CORRECTLY refuses to certify (a
    /// genuinely non-identifiable structured problem, confirmed by the outer-cert
    /// owner). Genuine above-floor residual energy makes that pass well-posed. The
    /// evenly-spaced angles sum to zero, so μ stays ≈ OFFSET up to the noise's
    /// finite-sample column mean (std `σ/√N`), which the tolerances below reflect.
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

    fn circle_target(offset: f64) -> Array2<f64> {
        let mut state = 0x2023_0000_0000_0007u64 ^ offset.to_bits();
        Array2::from_shape_fn((N_CIRCLE, 2), |(i, j)| {
            let theta = std::f64::consts::TAU * (i as f64) / (N_CIRCLE as f64);
            let clean = if j == 0 { theta.cos() } else { theta.sin() };
            clean + offset + NOISE_SIGMA * lcg_normal(&mut state)
        })
    }

    /// Drive the full typed pipeline (`build_sae_minimal_seed` →
    /// `build_sae_fit_seed` → `run_sae_manifold_fit`) on `target`, returning the fit
    /// report. Mirrors `examples/sae_fit.rs` with a single periodic atom.
    fn run_primary(target: Array2<f64>) -> crate::manifold::SaeFitReport {
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

        run_sae_manifold_fit(SaeFitRequest {
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
        })
        .expect("primary fit runs")
    }

    /// The primary entry peels the shared mean off a RAW target: μ is installed on
    /// the returned term (≈ the column mean = OFFSET), and the reported
    /// reconstruction is lifted back to raw-target space (its column mean tracks the
    /// target's, i.e. the DC is present in `report.fitted`, carried by Tier-0).
    #[test]
    fn primary_entry_installs_tier0_mean_on_raw_target() {
        install_test_logger();
        const OFFSET: f64 = 7.0;
        let target = circle_target(OFFSET);
        let target_col_mean = target.mean_axis(Axis(0)).unwrap();
        let report = run_primary(target.clone());

        let mu = report
            .term
            .tier0_mean()
            .expect("primary entry must install a Tier-0 mean on a raw target")
            .clone();
        assert_eq!(mu.len(), 2, "μ must have one entry per output dim");
        for j in 0..2 {
            // μ IS the target's column mean by construction (both are `mean_axis`),
            // so this stays exact regardless of the noise.
            assert!(
                (mu[j] - target_col_mean[j]).abs() < 1e-9,
                "μ[{j}]={} must equal the target column mean {}",
                mu[j],
                target_col_mean[j]
            );
            // The DC dominates the column mean (the evenly-spaced circle coords sum
            // to 0), so μ ≈ OFFSET up to the noise's finite-sample column mean
            // (std σ/√N = 0.05/8 ≈ 6.3e-3); 0.1 is a comfortable ~16σ/√N band.
            assert!(
                (mu[j] - OFFSET).abs() < 0.1,
                "μ[{j}]={} must be ≈ OFFSET={OFFSET} within the finite-sample noise mean",
                mu[j]
            );
        }

        // The reported reconstruction lives in RAW-target space: its column mean
        // tracks the target's (the DC is added back by Tier-0), not the de-meaned
        // frame the atoms were fit in. With observation noise the fit is no longer
        // exact, so the add-back is checked within a finite-sample band rather than
        // at machine precision.
        let recon_col_mean = report.fitted.mean_axis(Axis(0)).unwrap();
        for j in 0..2 {
            assert!(
                (recon_col_mean[j] - target_col_mean[j]).abs() < 5e-2,
                "reconstruction column mean[{j}]={} must track the raw target mean {} \
                 (Tier-0 add-back), not the de-meaned 0",
                recon_col_mean[j],
                target_col_mean[j]
            );
        }
        assert!(
            report.reconstruction_r2.is_finite(),
            "reconstruction R² must be finite"
        );
    }

    /// Always-on Tier-0 is a NO-OP on already-centered data: μ is computed from THIS
    /// target, so a mean-zero target yields μ ≈ 0 (no double-subtraction hazard).
    #[test]
    fn primary_entry_tier0_is_a_noop_on_centered_target() {
        install_test_logger();
        let target = circle_target(0.0); // the circle coords are already mean-zero
        let report = run_primary(target);
        let mu = report
            .term
            .tier0_mean()
            .expect("Tier-0 is installed unconditionally (value ≈ 0 here)")
            .clone();
        for (j, &m) in mu.iter().enumerate() {
            // μ is the target's column mean; with mean-zero circle coords it is just
            // the noise's finite-sample column mean (std σ/√N ≈ 6.3e-3) — small, not
            // a phantom DC. 0.1 is a comfortable ~16σ/√N band.
            assert!(
                m.abs() < 0.1,
                "μ[{j}]={m} must be ≈ 0 on already-centered data (no phantom mean)"
            );
        }
    }

    /// #2228/#2266 — drive the FULL outer ρ search on the E1-analog fixture
    /// (K=1 softmax periodic atom, dense direct-logdet, small ρ) so the planner
    /// selects ARC via the #2266 dense-Hessian route, and OBSERVE the outcome
    /// end to end: does the fit mint (converge + certify), or does it stop at a
    /// specific gate (step refusal / cost-stall / value↔gradient cert
    /// disagreement)? The `OuterDidNotConverge` Display carries the plan (which
    /// must read `solver=Arc`), the terminal ‖g‖, and the stop reason — the exact
    /// evidence needed to locate the residual value↔gradient seam under ARC. This
    /// is the warm micro-repro standing in for the real-GPT-2 E1 mint.
    fn run_primary_outer_search(target: Array2<f64>) -> Result<crate::manifold::SaeFitReport, SaeFitError> {
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
            max_iter: 64,
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

        run_sae_manifold_fit(SaeFitRequest {
            base_term,
            target,
            registry,
            initial_rho,
            max_iter: 64,
            learning_rate: 1.0,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            alpha: 1.0,
            isometry_pin_active,
            metric_provenance,
            promote_from_residual: false,
            run_structure_search: false,
            // The whole point: enable the outer ρ search so ARC actually runs.
            run_outer_rho_search: true,
            // Isolate the pass-0 iid outer search from the structured-residual
            // alternation (which is intentionally non-identifiable on the clean
            // circle — see this module's fixture note).
            structured_residual_passes: 0,
            cancel: None,
        })
    }

    #[test]
    fn e1_arc_outer_search_mints_or_reports_blocker_2266() {
        install_test_logger();
        let target = circle_target(0.0);
        match run_primary_outer_search(target) {
            Ok(report) => {
                // Minted: the ARC outer search converged AND certified stationarity.
                println!(
                    "[#2266] E1-ARC outer search MINTED: penalized_quasi_laplace_criterion={:.6e}",
                    report.penalized_quasi_laplace_criterion
                );
                assert!(
                    report.penalized_quasi_laplace_criterion.is_finite(),
                    "a minted fit must carry a finite criterion"
                );
            }
            Err(err) => {
                // Not minted — the Display carries plan= (must be solver=Arc), the
                // terminal ‖g‖, and the stop reason: the observation this repro
                // exists to capture.
                panic!("[#2266] E1-ARC outer search did NOT mint: {err}");
            }
        }
    }
}

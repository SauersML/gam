// #901 re-home: the Matérn spatial length-scale (iso-κ) optimizer convergence
// gates. The #901 issue listed these as failing — the κ optimizer would stall
// (`final_grad_norm ≈ 1.35`, never reaching `rel_tol`) because the outer REML
// gradient it descended was the wrong projected-logdet gradient. With the #901
// fix (commit 7a5bfd9b2: intrinsic ½log|H_pen|₊ pseudo-logdet) the outer
// gradient is exact, the optimizer reaches tolerance, and the optimized score
// is monotone-non-worse than the unoptimized baseline.
//
// Authored in the pre-#1521 monolith (`tests/src_modules/smooths/`), these were
// orphaned out of the build by #1601: their driver deps
// (`fit_term_collection_forspec`, `fit_term_collectionwith_spatial_length_scale_optimization`,
// `fit_score`, `SpatialLengthScaleOptimizationOptions`) live HERE post-carve,
// not in `gam_terms::smooth`. Re-homed as a `#[cfg(test)] mod` `include!`d into
// the drivers module so the private driver surface resolves via `super::*`.

#[cfg(test)]
mod spatial_length_scale_monotone_tests {
    use super::*;
    use gam_terms::basis::{MaternBasisSpec, MaternNu};
    use gam_terms::smooth::auto_initial_length_scale_for_centers;
    use ndarray::{Array1, Array2, ArrayView2};

    /// Runs a Gaussian baseline fit and the spatial
    /// length-scale optimization for a single Matérn term, then asserts the
    /// optimized score is monotone-non-worse and that the resolved term froze
    /// its centers / identifiability transform with a finite in-range length
    /// scale. Shared verbatim between the 2- and 3-feature Matérn monotone
    /// pins, which differ only in their data generation, term dimensionality,
    /// and seed length scale.
    fn assert_matern_spatial_length_scale_optimization_monotone(
        data: ArrayView2<'_, f64>,
        y: &Array1<f64>,
        weights: &Array1<f64>,
        offset: &Array1<f64>,
        spec: &TermCollectionSpec,
        fit_opts: &FitOptions,
    ) {
        let baseline = fit_term_collection_forspec(
            data,
            y.view(),
            weights.view(),
            offset.view(),
            spec,
            LikelihoodSpec::gaussian_identity(),
            fit_opts,
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "baseline fit should succeed", e));
        let baseline_score = fit_score(&baseline.fit);

        let optimized = fit_term_collectionwith_spatial_length_scale_optimization(
            data,
            y.clone(),
            weights.clone(),
            offset.clone(),
            spec,
            LikelihoodSpec::gaussian_identity(),
            fit_opts,
            &SpatialLengthScaleOptimizationOptions {
                // `max_outer_iter: 2` was set when the iso-κ analytic
                // optimizer typically converged within two BFGS steps. The
                // current optimizer reaches the relative-gradient tolerance
                // only after a handful of outer iterations on the Matérn
                // monotone fixtures (the previous run-out left
                // `|g|_proj ≈ 1.65e-1` against `|f| ≈ 1.3e2` — well above
                // `rel_tol * (1 + |f|) ≈ 1.3e-3`), so a 2-iteration cap
                // bails before reaching convergence. Raising the cap to 16
                // gives the optimizer headroom to actually reach the
                // tolerance the test is asserting against; the
                // monotone-improvement contract this test pins is unchanged.
                max_outer_iter: 16,
                rel_tol: 1e-5,
                pilot_subsample_threshold: 0,
                ..SpatialLengthScaleOptimizationOptions::default()
            },
        )
        .unwrap_or_else(|e| panic!("{} failed: {:?}", "optimized fit should succeed", e));
        let optimized_score = fit_score(&optimized.fit);
        assert!(optimized_score <= baseline_score + 1e-10);

        let ls = match &optimized.resolvedspec.smooth_terms[0].basis {
            SmoothBasisSpec::Matern { spec, .. } => spec.length_scale.resolved().unwrap(),
            _ => panic!("expected Matérn term"),
        };
        assert!(ls.is_finite() && (1e-3..=1e3).contains(&ls));

        match &optimized.resolvedspec.smooth_terms[0].basis {
            SmoothBasisSpec::Matern { spec, .. } => {
                assert!(matches!(
                    spec.center_strategy,
                    CenterStrategy::UserProvided(_)
                ));
                assert!(matches!(
                    spec.identifiability,
                    MaternIdentifiability::FrozenTransform { .. }
                ));
            }
            _ => panic!("expected Matérn term"),
        }
    }

    /// Return `(short_seed, long_endpoint, selected)` for the certified
    /// pre-joint Matérn range comparison on a deterministic sinusoid. Keeping
    /// this at the profiler boundary isolates the global basin decision from
    /// the subsequent local joint optimizer.
    fn profiled_matern_basin_for_frequency(frequency: f64) -> (f64, f64, f64) {
        let n = 120usize;
        let num_centers = 20usize;
        let mut data = Array2::<f64>::zeros((n, 1));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x = i as f64 / (n - 1) as f64;
            data[[i, 0]] = x;
            y[i] = (2.0 * std::f64::consts::PI * frequency * x).sin()
                + 0.05 * (2.0 * std::f64::consts::PI * 37.0 * x).sin();
        }
        let short_seed =
            auto_initial_length_scale_for_centers(data.view(), &[0], num_centers);
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers },
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(short_seed),
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);
        let family = LikelihoodSpec::gaussian_identity();
        let options = superseded_fit_options(&FitOptions::default());
        let baseline = fit_term_collection_forspec(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &spec,
            family.clone(),
            &options,
        )
        .expect("short-range profile");
        let resolved =
            freeze_term_collection_from_design(&spec, &baseline.design).expect("freeze profile");
        let spatial_terms = spatial_length_scale_term_indices(&resolved);
        assert_eq!(spatial_terms, vec![0]);
        let kappa_options = SpatialLengthScaleOptimizationOptions::default();
        let companion_length_scale = matern_low_rank_center_resolution_length_scale(
            data.view(),
            &[0],
            num_centers,
        )
        .expect("center-resolution endpoint");
        let (psi_long_bound, psi_short_bound) =
            spatial_term_psi_bounds(data.view(), &resolved, 0, &kappa_options)
                .expect("finite isotropic-scale bounds");
        let psi_long = (-companion_length_scale.ln()).clamp(psi_long_bound, psi_short_bound);
        let long_endpoint = (-psi_long).exp();
        let (selected_spec, _) = select_isotropic_matern_range_basin(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            resolved,
            baseline,
            &family,
            &options,
            &kappa_options,
            &spatial_terms,
        )
        .expect("certified endpoint profile comparison");
        let selected = get_spatial_length_scale(&selected_spec, 0).expect("selected Matérn range");
        (short_seed, long_endpoint, selected)
    }

    #[test]
    fn smooth_nu_five_halves_selects_certified_long_range_basin() {
        let (short, long, selected) = profiled_matern_basin_for_frequency(1.0);
        assert!(long > short, "fixture must expose distinct range basins");
        assert_eq!(
            selected, long,
            "smooth ν=5/2 signal should enter the certified long-range basin"
        );
    }

    #[test]
    fn sin8_nu_five_halves_retains_certified_short_range_basin() {
        let (short, long, selected) = profiled_matern_basin_for_frequency(8.0);
        assert!(long > short, "fixture must expose distinct range basins");
        assert_eq!(
            selected, short,
            "sin8 ν=5/2 signal must retain the resolving short-range basin"
        );
    }

    #[test]
    fn spatial_length_scale_optimization_monotone_improves_or_keeps_score_for_matern_two_feature() {
        let n = 60usize;
        let d = 3usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.13).sin();
            let x2 = (i as f64 * 0.07).cos();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            data[[i, 2]] = x2;
            y[i] = (2.5 * x0).sin() + 0.4 * x1 - 0.2 * x2;
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1, 2],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(20.0),
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };
        let fit_opts = FitOptions {
            max_iter: 40,
            ..FitOptions::default()
        };
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);

        assert_matern_spatial_length_scale_optimization_monotone(
            data.view(),
            &y,
            &weights,
            &offset,
            &spec,
            &fit_opts,
        );
    }

    #[test]
    fn spatial_length_scale_optimization_monotone_improves_or_keeps_score_for_matern() {
        let n = 60usize;
        let d = 2usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.17).sin();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            y[i] = (3.0 * x0).cos() + 0.35 * x1;
        }

        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(12.0),
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };
        let fit_opts = FitOptions {
            max_iter: 40,
            ..FitOptions::default()
        };
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);

        assert_matern_spatial_length_scale_optimization_monotone(
            data.view(),
            &y,
            &weights,
            &offset,
            &spec,
            &fit_opts,
        );
    }

    /// #2454 MEASUREMENT (reports, never fails): is a penalty block INDEFINITE?
    ///
    /// #2454 measures `∂V/∂ρ = −c·λ` exactly on this fixture family (c = 2.87e-9,
    /// ratio → e over eight e-folds), i.e. `∂V/∂λ = −c` is a CONSTANT NEGATIVE
    /// derivative, so `V` is linear in λ and unbounded below.
    ///
    /// The only term in a REML criterion that is linear in λ is the penalty
    /// term `½·λ·βᵀSₖβ/φ`, whose slope is `+½βᵀSₖβ/φ`. That slope is
    /// non-negative **iff `Sₖ` is positive semidefinite**. A constant NEGATIVE
    /// slope therefore requires `βᵀSₖβ < 0`, which requires an INDEFINITE
    /// penalty block — there is no other source.
    ///
    /// This fixture is where that is most likely: `length_scale = 12.0` against
    /// data spanning `x0 ∈ [0,1]`, `x1 ∈ [−1,1]`, so every pair of points lies
    /// within ~0.17 length-scales and the Matérn kernel matrix is nearly
    /// rank-1. A penalty assembled as a difference of operators loses PSD-ness
    /// to rounding in exactly that regime.
    ///
    /// Reports the extreme eigenvalues of every canonical penalty block, and
    /// the ratio `|λ_min| / λ_max` so a merely tiny-but-negative eigenvalue is
    /// distinguishable from a structurally indefinite one. If any `λ_min` is
    /// negative beyond symmetric-eigensolver round-off, #2454's mechanism is
    /// identified.
    #[test]
    fn zz_measure_penalty_block_definiteness_2454() {
        use gam_linalg::faer_ndarray::FaerEigh;
        let n = 60usize;
        let d = 2usize;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.17).sin();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
        }
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(12.0),
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };
        let design = build_term_collection_design(data.view(), &spec)
            .unwrap_or_else(|e| panic!("design failed: {e:?}"));
        eprintln!(
            "[zz-psd-2454] penalties={} design={}x{}",
            design.penalties.len(),
            design.design.nrows(),
            design.design.ncols()
        );
        for (k, cp) in design.penalties.iter().enumerate() {
            let local = &cp.local;
            let (evals, _evecs): (Array1<f64>, Array2<f64>) = local
                .eigh(faer::Side::Lower)
                .unwrap_or_else(|e| panic!("penalty {k} eigh failed: {e:?}"));
            let lo = evals.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = evals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let negatives = evals.iter().filter(|v| **v < 0.0).count();
            eprintln!(
                "[zz-psd-2454] penalty {k} cols={:?} dim={} lmin={:+.6e} lmax={:+.6e} \
                 |lmin|/lmax={:.3e} negatives={} INDEFINITE={}",
                cp.col_range,
                local.nrows(),
                lo,
                hi,
                if hi > 0.0 { lo.abs() / hi } else { f64::NAN },
                negatives,
                lo < 0.0
            );
        }
    }

    /// #2454 MEASUREMENT (reports, never fails): where the joint spatial route
    /// STARTS, against where the baseline it is graded against ENDED.
    ///
    /// The monotone gate compares `joint_final_value` (the joint optimizer's
    /// certified cost over the derived ρ domain, at a ψ clamped into the
    /// data-derived κ window) against `fit_score(&baseline)` (a standard scalar-ρ
    /// fit over the same derived domain at the SPEC's length scale). Those are two
    /// different feasible sets. If the baseline's own ρ̂ or ψ is outside the joint
    /// box, `theta0` is a CLAMPED — hence strictly worse — point, and the
    /// certificate can fail with the optimizer having descended perfectly.
    ///
    /// Prints, for the fixture #2454 was opened against: the baseline's ρ̂ and
    /// score; the joint route's ρ/ψ box; `theta0` before and after clamping; and
    /// the criterion at both. Whoever reads this can separate "the optimizer went
    /// uphill" from "the gate compares two different problems" in one run.
    #[test]
    fn zz_measure_joint_route_startup_against_baseline_2454() {
        let n = 60usize;
        let d = 2usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.17).sin();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            y[i] = (3.0 * x0).cos() + 0.35 * x1;
        }
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(12.0),
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };
        let fit_opts = FitOptions {
            max_iter: 40,
            ..FitOptions::default()
        };
        let baseline = fit_term_collection_forspec(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &spec,
            LikelihoodSpec::gaussian_identity(),
            &fit_opts,
        )
        .unwrap_or_else(|e| panic!("baseline fit failed: {e:?}"));
        let rho_hat = baseline.fit.lambdas.mapv(f64::ln);
        // The ρ domain the joint route derives from the baseline's own design
        // and penalties (#2812), so the probe reports the same projection the
        // driver applies to its seed.
        let (rho_lower, rho_upper) = joint_rho_resolvability_domain(
            &baseline.design.design,
            &baseline.design.penalties,
            rho_hat.len(),
        );
        eprintln!(
            "[zz-start-2454] baseline score={:+.10e} rho_hat={:?} joint_rho_domain lower={:?} upper={:?}",
            fit_score(&baseline.fit),
            rho_hat.to_vec(),
            rho_lower.to_vec(),
            rho_upper.to_vec(),
        );
        let clamped: Vec<f64> = rho_hat
            .iter()
            .enumerate()
            .map(|(k, &r)| r.clamp(rho_lower[k], rho_upper[k]))
            .collect();
        let moved = rho_hat
            .iter()
            .zip(clamped.iter())
            .filter(|(a, b)| (*a - *b).abs() > 0.0)
            .count();
        eprintln!(
            "[zz-start-2454] theta0 rho after clamp={clamped:?} coordinates_moved_by_clamp={moved}"
        );

        let design = build_term_collection_design(data.view(), &spec)
            .unwrap_or_else(|e| panic!("design failed: {e:?}"));
        let resolved = freeze_term_collection_from_design(&spec, &design)
            .unwrap_or_else(|e| panic!("freeze failed: {e:?}"));
        let kappa_options = SpatialLengthScaleOptimizationOptions::default();
        let (psi_lower, psi_upper) =
            gam_terms::smooth::spatial_term_psi_bounds(data.view(), &resolved, 0, &kappa_options)
                .unwrap_or_else(|e| panic!("psi bounds: {e:?}"));
        let seed_length_scale =
            get_spatial_length_scale(&resolved, 0).expect("resolved length scale");
        let psi_seed = -seed_length_scale.ln();
        eprintln!(
            "[zz-start-2454] psi seed={psi_seed:+.8} (length_scale={seed_length_scale:.6}) \
             psi_box=[{psi_lower:+.8}, {psi_upper:+.8}] after_clamp={:+.8} clamped={}",
            psi_seed.clamp(psi_lower, psi_upper),
            psi_seed < psi_lower || psi_seed > psi_upper,
        );
    }

    /// #2454 MEASUREMENT (reports, never fails): the joint route's terminal
    /// point carries TWO gradients that disagree by four orders.
    ///
    /// With the search box fixed to contain its own seed, the monotone Matérn
    /// arm now improves on its baseline (−69.585 against −68.411) and refuses at
    /// the stationarity certificate instead:
    ///
    /// ```text
    ///   |g|=4.316e0 |Pg|=4.316e0 bound=7.059e-4
    ///   railed (theta-wide): [3]   #3 theta=-2.484907e0 box=[-2.484907e0, 5.719033e0]
    ///   solver provenance: claimed_converged=true,
    ///     termination=gradient_tolerance(|g|=1.029235e-4 < 5.487011e-4)
    ///   rho_checkpoint = [-6.285681, -4.950630, -2.273293, -2.484907]
    /// ```
    ///
    /// `|Pg| = |g|` says the projector did NOT drop coordinate 3, i.e. its
    /// gradient is feasible-descent (points INTO the box) — which is not a state
    /// a descent method should be able to terminate in. Meanwhile the solver's
    /// own terminating `|g|` at what should be the same point is 1.03e-4.
    ///
    /// Both cannot be the gradient of one criterion at one θ. This probe
    /// evaluates the production joint objective at the reported checkpoint and
    /// prints, per coordinate, the analytic gradient beside a central difference
    /// of the same criterion — so the disagreement is attributed to a coordinate
    /// and to a side (analytic wrong, or the two evaluations are not at the same
    /// point) rather than being read off a refusal string.
    #[test]
    fn zz_measure_joint_terminal_gradient_at_the_checkpoint_2454() {
        use crate::fit_orchestration::drivers::test_support::SingleBlockExactJointDesignCacheTestExt;
        let n = 60usize;
        let d = 2usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.17).sin();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            y[i] = (3.0 * x0).cos() + 0.35 * x1;
        }
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(12.0),
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };
        let fit_opts = FitOptions {
            max_iter: 40,
            ..FitOptions::default()
        };
        // Verbatim from the refusal strings, θ = [ρ₀, ρ₁, ρ₂, ψ].
        //
        // `frozen-psi` is the point the route stopped at when the active-set
        // reduction pinned ψ at its bound. `stall` is where it stops NOW, with
        // ψ free and the outer cap raised past 32 — identical for caps 32, 64,
        // 128 and 256, so it is not a budget. Both are probed, because the pair
        // is the before/after of the active-set fix on one instrument.
        for (label, checkpoint) in [
            (
                "frozen-psi",
                Array1::from(vec![
                    -6.285_681_405_991_277_f64,
                    -4.950_629_609_698_647,
                    -2.273_293_245_252_893,
                    -2.484_906_649_788_000_4,
                ]),
            ),
            (
                "stall",
                Array1::from(vec![
                    8.886_802_573_715_045_f64,
                    2.067_724_148_446_362,
                    9.597_640_999_700_111,
                    -1.270_837_894_462_055_3,
                ]),
            ),
        ] {

            let design = build_term_collection_design(data.view(), &spec)
                .unwrap_or_else(|e| panic!("design failed: {e:?}"));
            let resolved = freeze_term_collection_from_design(&spec, &design)
                .unwrap_or_else(|e| panic!("freeze failed: {e:?}"));
            let frozen_design = build_term_collection_design(data.view(), &resolved)
                .unwrap_or_else(|e| panic!("frozen design failed: {e:?}"));
            let spatial_terms = spatial_length_scale_term_indices(&resolved);
            let dims_per_term = spatial_dims_per_term(&resolved, &spatial_terms);
            let rho_dim = frozen_design.penalties.len();
            let psi_dim: usize = dims_per_term.iter().sum();
            let theta_dim = rho_dim + psi_dim;
            eprintln!("[zz-grad-2454] rho_dim={rho_dim} psi_dim={psi_dim} theta_dim={theta_dim}");
            if theta_dim != checkpoint.len() {
                eprintln!(
                    "[zz-grad-2454] SKIP: reproduced theta_dim={theta_dim} but the recorded \
                     checkpoint has {} entries",
                    checkpoint.len()
                );
                return;
            }

            let family = LikelihoodSpec::gaussian_identity();
            let external_opts = external_opts_for_design(&family, &frozen_design, &fit_opts);
            let mut cache = SingleBlockExactJointDesignCache::new(
                data.view(),
                resolved.clone(),
                frozen_design.clone(),
                spatial_terms.clone(),
                rho_dim,
                dims_per_term.clone(),
            )
            .unwrap_or_else(|e| panic!("cache failed: {e:?}"));
            let mut evaluator = gam_solve::estimate::ExternalJointHyperEvaluator::new(
                y.view(),
                weights.view(),
                &frozen_design.design,
                offset.view(),
                &frozen_design.penalties,
                &external_opts,
                "#2454 joint terminal gradient",
            )
            .unwrap_or_else(|e| panic!("evaluator failed: {e:?}"));

            let cost_at = |theta: &Array1<f64>,
                           cache: &mut SingleBlockExactJointDesignCache<'_>,
                           evaluator: &mut gam_solve::estimate::ExternalJointHyperEvaluator<'_>|
             -> f64 {
                cache
                    .ensure_theta(theta)
                    .unwrap_or_else(|e| panic!("ensure_theta: {e:?}"));
                let design = cache.design();
                evaluator
                    .evaluate_cost_only(
                        &design.design,
                        &design.penalties,
                        &design.nullspace_dims,
                        design.linear_constraints.clone(),
                        theta,
                        rho_dim,
                        None,
                        "#2454 joint terminal cost-only",
                        None,
                    )
                    .unwrap_or_else(|e| panic!("cost-only: {e:?}"))
            };

            cache
                .ensure_theta(&checkpoint)
                .unwrap_or_else(|e| panic!("ensure_theta: {e:?}"));
            let hyper_dirs = try_build_spatial_log_kappa_hyper_dirs(
                data.view(),
                cache.spec(),
                cache.design(),
                &cache.spatial_terms,
            )
            .unwrap_or_else(|e| panic!("hyper dirs: {e:?}"))
            .expect("hyper dirs present");
            let (cost_an, grad_an, _hess) = evaluate_joint_reml_outer_eval_at_theta(
                &mut evaluator,
                cache.design(),
                &checkpoint,
                rho_dim,
                hyper_dirs,
                None,
                gam_solve::rho_optimizer::OuterEvalOrder::ValueAndGradient,
                None,
            )
            .unwrap_or_else(|e| panic!("outer eval: {e:?}"));

            for h in [1e-3_f64, 1e-4, 1e-5] {
                for j in 0..theta_dim {
                    let mut plus = checkpoint.clone();
                    plus[j] += h;
                    let mut minus = checkpoint.clone();
                    minus[j] -= h;
                    let cp = cost_at(&plus, &mut cache, &mut evaluator);
                    let cm = cost_at(&minus, &mut cache, &mut evaluator);
                    let fd = (cp - cm) / (2.0 * h);
                    let kind = if j >= rho_dim { "psi" } else { "rho" };
                    eprintln!(
                        "[zz-grad-2454] {label} V={cost_an:+.10e} h={h:.0e} {kind} j={j} \
                         analytic={:+.8e} fd={fd:+.8e} |an-fd|={:.3e}",
                        grad_an[j],
                        (grad_an[j] - fd).abs()
                    );
                }
            }
            let norm = grad_an.dot(&grad_an).sqrt();
            eprintln!("[zz-grad-2454] {label}: |g|_analytic_at_checkpoint={norm:.6e}");
        }
    }

    /// #2454 MEASUREMENT (reports, never fails): how much outer budget the joint
    /// ψ descent actually needs, now that ψ is free to move.
    ///
    /// The monotone fixtures pin `max_outer_iter: 16`, a number calibrated (per
    /// the comment on that field) against runs in which ψ was frozen at its box
    /// edge for the whole search — so the budget only ever had to cover the ρ
    /// block. With the active-set fix ψ moves, and the refusal changed to
    /// `IterationBudget after 16 outer iteration(s); |g|=6.105992e-1 never
    /// reached 6.941073e-4`.
    ///
    /// This sweeps the cap and reports, per cap, whether the fit succeeded and
    /// what score it reached — so "the budget was calibrated against a search
    /// that was not searching" is a measurement rather than an assertion, and so
    /// a future slowdown in the ψ direction shows up as a number here.
    #[test]
    fn zz_measure_joint_outer_budget_needed_for_psi_descent_2454() {
        let n = 60usize;
        let d = 2usize;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x0 = i as f64 / (n as f64 - 1.0);
            let x1 = (i as f64 * 0.17).sin();
            data[[i, 0]] = x0;
            data[[i, 1]] = x1;
            y[i] = (3.0 * x0).cos() + 0.35 * x1;
        }
        let weights = Array1::ones(n);
        let offset = Array1::zeros(n);
        let spec = TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
                name: "matern".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0, 1],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(12.0),
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        };
        let fit_opts = FitOptions {
            max_iter: 40,
            ..FitOptions::default()
        };
        let baseline = fit_term_collection_forspec(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            &spec,
            LikelihoodSpec::gaussian_identity(),
            &fit_opts,
        )
        .unwrap_or_else(|e| panic!("baseline fit failed: {e:?}"));
        let baseline_score = fit_score(&baseline.fit);
        eprintln!("[zz-budget-2454] baseline score={baseline_score:+.10e}");

        for cap in [16usize, 32, 64, 128, 256] {
            let outcome = fit_term_collectionwith_spatial_length_scale_optimization(
                data.view(),
                y.clone(),
                weights.clone(),
                offset.clone(),
                &spec,
                LikelihoodSpec::gaussian_identity(),
                &fit_opts,
                &SpatialLengthScaleOptimizationOptions {
                    max_outer_iter: cap,
                    rel_tol: 1e-5,
                    pilot_subsample_threshold: 0,
                    ..SpatialLengthScaleOptimizationOptions::default()
                },
            );
            match outcome {
                Ok(fitted) => {
                    let ls = match &fitted.resolvedspec.smooth_terms[0].basis {
                        SmoothBasisSpec::Matern { spec, .. } => spec.length_scale.resolved(),
                        _ => None,
                    };
                    eprintln!(
                        "[zz-budget-2454] cap={cap:3} OK score={:+.10e} (baseline {baseline_score:+.10e}, \
                         improvement {:+.6e}) length_scale={ls:?}",
                        fit_score(&fitted.fit),
                        baseline_score - fit_score(&fitted.fit),
                    );
                }
                Err(error) => {
                    eprintln!("[zz-budget-2454] cap={cap:3} REFUSED {error}");
                }
            }
        }
    }
}

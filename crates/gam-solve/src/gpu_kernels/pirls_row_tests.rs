use super::*;

#[cfg(target_os = "linux")]
fn close(got: f64, expected: f64, rel: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (got - expected).abs() <= rel * scale,
        "got={got:?}, expected={expected:?}, rel={rel:?}"
    );
}

#[test]
fn exact_log_rows_retain_the_declared_endpoints_without_floors() {
    for eta in [
        crate::mixture_link::LOG_LINK_SOLVER_ETA_MIN,
        crate::mixture_link::LOG_LINK_SOLVER_ETA_MAX,
    ] {
        let mu = eta.exp();
        let poisson = row_reweight_cpu(
            PirlsRowFamily::PoissonLog,
            CurvatureMode::Fisher,
            RowInput {
                eta,
                y: mu,
                prior_weight: 1.0,
            },
            1.0,
        )
        .unwrap();
        assert_eq!(poisson.mu, mu);
        assert_eq!(poisson.w_solver, mu);
        assert_eq!(poisson.grad_eta, 0.0);
        assert_eq!(poisson.deviance, 0.0);

        let gamma = row_reweight_cpu(
            PirlsRowFamily::GammaLog,
            CurvatureMode::Observed,
            RowInput {
                eta,
                y: mu,
                prior_weight: 0.25,
            },
            3.0,
        )
        .unwrap();
        assert_eq!(gamma.mu, mu);
        assert_eq!(gamma.w_fisher, 0.75);
        assert_eq!(gamma.w_hessian, 0.75);
        assert_eq!(gamma.w_solver, gamma.w_hessian);
        assert_eq!(gamma.grad_eta, 0.0);
        assert_eq!(gamma.deviance, 0.0);
    }

    let just_outside = f64::from_bits(crate::mixture_link::LOG_LINK_SOLVER_ETA_MAX.to_bits() + 1);
    assert!(matches!(
        row_reweight_cpu(
            PirlsRowFamily::PoissonLog,
            CurvatureMode::Fisher,
            RowInput {
                eta: just_outside,
                y: 1.0,
                prior_weight: 1.0,
            },
            1.0,
        ),
        Err(EstimationError::InverseLinkDomainViolation { eta, .. }) if eta == just_outside
    ));
}

#[test]
fn canonical_logit_tail_geometry_survives_rounded_means() {
    for (eta, y) in [(-700.0, 1.0), (700.0, 0.0)] {
        let row = row_reweight_cpu(
            PirlsRowFamily::BernoulliLogit,
            CurvatureMode::Observed,
            RowInput {
                eta,
                y,
                prior_weight: 1.0,
            },
            1.0,
        )
        .unwrap();
        assert!(row.mu == 0.0 || row.mu == 1.0 || (0.0..=1.0).contains(&row.mu));
        assert!(row.w_fisher.is_finite() && row.w_fisher > 0.0);
        assert!(row.w_fisher < 1.0e-300);
        assert_eq!(row.w_solver, row.w_hessian);
        assert!(row.grad_eta.is_finite());
        assert!(row.deviance.is_finite());
    }
}

#[test]
fn zero_prior_is_exact_exclusion_but_negative_prior_is_refused() {
    let excluded = row_reweight_cpu(
        PirlsRowFamily::PoissonLog,
        CurvatureMode::Fisher,
        RowInput {
            eta: -700.0,
            y: f64::NAN,
            prior_weight: 0.0,
        },
        1.0,
    )
    .unwrap();
    assert_eq!(excluded.grad_eta, 0.0);
    assert_eq!(excluded.w_solver, 0.0);
    assert_eq!(excluded.deviance, 0.0);

    assert!(matches!(
        row_reweight_cpu_at(
            9,
            PirlsRowFamily::GaussianIdentity,
            CurvatureMode::Fisher,
            RowInput {
                eta: 0.0,
                y: 1.0,
                prior_weight: -f64::MIN_POSITIVE,
            },
            1.0,
        ),
        Err(EstimationError::PirlsRowGeometryUnrepresentable {
            row: 9,
            quantity: "prior weight",
            ..
        })
    ));
}

#[test]
fn gamma_balanced_products_and_near_saturation_deviance_are_representable() {
    let mu = 1.0e200;
    let y = f64::from_bits(mu.to_bits() + 1);
    let row = row_reweight_cpu(
        PirlsRowFamily::GammaLog,
        CurvatureMode::Observed,
        RowInput {
            eta: mu.ln(),
            y,
            prior_weight: 1.0e-200,
        },
        2.0,
    )
    .unwrap();
    assert!(row.w_hessian.is_finite() && row.w_hessian > 0.0);
    assert!(row.deviance.is_finite() && row.deviance >= 0.0);
    assert_eq!(row.w_solver, row.w_hessian);
}

#[test]
fn gamma_fisher_does_not_materialize_an_unused_overflowing_observed_weight() {
    let input = RowInput {
        eta: 0.0,
        y: 1.126,
        prior_weight: 8.0e307,
    };
    let fisher = row_reweight_cpu(PirlsRowFamily::GammaLog, CurvatureMode::Fisher, input, 2.0)
        .expect("Fisher score and deviance remain representable");
    assert!(fisher.w_fisher.is_finite());
    assert!(fisher.grad_eta.is_finite());
    assert!(fisher.deviance.is_finite());
    assert!(
        row_reweight_cpu(
            PirlsRowFamily::GammaLog,
            CurvatureMode::Observed,
            input,
            2.0,
        )
        .is_err()
    );
}

#[test]
fn poisson_score_does_not_materialize_an_overflowing_weighted_response() {
    let row = row_reweight_cpu(
        PirlsRowFamily::PoissonLog,
        CurvatureMode::Fisher,
        RowInput {
            eta: 0.0,
            y: 2.0,
            prior_weight: 9.0e307,
        },
        1.0,
    )
    .expect("canonical score and deviance remain representable");
    assert_eq!(row.w_fisher, 9.0e307);
    assert_eq!(row.grad_eta, 9.0e307);
    assert!(row.deviance.is_finite() && row.deviance > 0.0);
}

#[test]
fn poisson_tail_deviance_avoids_dimensionless_overflow_and_rounded_minus_one() {
    // y/mu is finite, but `(y/mu) * log(y/mu)` is not. The final weighted
    // deviance is O(10^3), so refusing this row would be a false intermediate
    // overflow.
    let eta = crate::mixture_link::LOG_LINK_SOLVER_ETA_MIN;
    let y = 1.0e4;
    let prior_weight = 1.0e-4;
    let left = row_reweight_cpu(
        PirlsRowFamily::PoissonLog,
        CurvatureMode::Fisher,
        RowInput {
            eta,
            y,
            prior_weight,
        },
        1.0,
    )
    .expect("absolute-coordinate Poisson deviance is representable");
    let expected_half = prior_weight * y * (y.ln() - eta - 1.0) + left.w_fisher;
    assert!(left.deviance.is_finite());
    assert!((left.deviance - 2.0 * expected_half).abs() <= 8.0 * f64::EPSILON * expected_half);

    // y/mu rounds below machine resolution, hence u=(y-mu)/mu rounds to -1;
    // 0*log(0) in the ratio formula must not poison the representable answer.
    let right = row_reweight_cpu(
        PirlsRowFamily::PoissonLog,
        CurvatureMode::Fisher,
        RowInput {
            eta: crate::mixture_link::LOG_LINK_SOLVER_ETA_MAX,
            y: 1.0,
            prior_weight: 1.0e-304,
        },
        1.0,
    )
    .expect("rounded u=-1 has a finite absolute-coordinate deviance");
    assert!(right.deviance.is_finite() && right.deviance > 0.0);
}

#[test]
fn gamma_tail_deviance_avoids_rounded_minus_one() {
    let eta = crate::mixture_link::LOG_LINK_SOLVER_ETA_MAX;
    let row = row_reweight_cpu(
        PirlsRowFamily::GammaLog,
        CurvatureMode::Fisher,
        RowInput {
            eta,
            y: 1.0,
            prior_weight: 0.5,
        },
        2.0,
    )
    .expect("Gamma deviance remains finite when y/mu rounds away");
    assert_eq!(row.w_fisher, 1.0);
    assert_eq!(row.grad_eta, -1.0);
    let expected = 2.0 * (eta - 1.0 + (-eta).exp());
    assert!((row.deviance - expected).abs() <= 8.0 * f64::EPSILON * expected.abs());
}

#[test]
fn fractional_logit_deviance_retains_the_local_quadratic() {
    let y = 0.3_f64;
    let response_logit = y.ln() - (-y).ln_1p();
    let eta = f64::from_bits(response_logit.to_bits() + 1);
    let row = row_reweight_cpu(
        PirlsRowFamily::BernoulliLogit,
        CurvatureMode::Fisher,
        RowInput {
            eta,
            y,
            prior_weight: 1.0,
        },
        1.0,
    )
    .unwrap();
    assert!(
        row.deviance > 0.0,
        "one-ULP displacement must not cancel to zero"
    );
    assert!(row.deviance.is_finite());
}

#[test]
fn observed_noncanonical_weights_are_never_row_projected() {
    for family in [
        PirlsRowFamily::BernoulliProbit,
        PirlsRowFamily::BernoulliCLogLog,
    ] {
        for eta in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0] {
            for y in [0.0, 1.0] {
                if let Ok(row) = row_reweight_cpu(
                    family,
                    CurvatureMode::Observed,
                    RowInput {
                        eta,
                        y,
                        prior_weight: 1.0,
                    },
                    1.0,
                ) {
                    assert_eq!(row.w_solver, row.w_hessian);
                }
            }
        }
    }
}

#[test]
fn refusal_replay_selects_the_smallest_bad_row_atomically() {
    let eta = [0.0, 0.0, 0.0];
    let y = [0.0, 2.0, -1.0];
    let prior = [1.0; 3];
    let status = [
        status_codes::OK,
        status_codes::RESPONSE,
        status_codes::RESPONSE,
    assert!(matches!(
        replay_first_refusal(
            PirlsRowFamily::BernoulliLogit,
            CurvatureMode::Fisher,
            1.0,
            &eta,
            &y,
            &prior,
            &status,
        ),
        Err(EstimationError::PirlsRowGeometryUnrepresentable { row: 1, .. })
    ));
}

#[cfg(target_os = "linux")]
#[test]
fn generated_sources_have_one_exact_unprojected_contract() {
    let forbidden = [
        "clamp_eta",
        "ETA_CLAMP",
        "MU_FLOOR",
        "W_SOLVER_FLOOR",
        "fmax(",
        "fmin(",
        "flags",
        "1e-12",
        "1e-10",
    ];
    for family in PirlsRowFamily::ALL {
        for curvature in [CurvatureMode::Fisher, CurvatureMode::Observed] {
            for source in [
                cuda_source_for(family, curvature),
                solve_row_source_for(family, curvature),
                ladder_source_for(family, curvature),
            ] {
                for token in forbidden {
                    assert!(!source.contains(token), "{family:?}/{curvature:?}: {token}");
                }
                assert!(source.contains("w_solver = w_hessian"));
                assert!(source.contains("status == PIRLS_OK"));
            }
        }
    }
    let ladder = ladder_source_for(PirlsRowFamily::PoissonLog, CurvatureMode::Fisher);
    assert!(ladder.contains("status_out[k * n + i] = status"));
}

#[cfg(target_os = "linux")]
#[test]
fn nvrtc_compiles_every_exact_builtin_mode_when_cuda_is_present() -> Result<(), GpuError> {
    let Ok(backend) = PirlsRowBackend::probe() else {
        return Ok(());
    };
    for family in PirlsRowFamily::ALL {
        for curvature in [CurvatureMode::Fisher, CurvatureMode::Observed] {
            backend.module_for(family, curvature)?;
            backend.module_for_solve(family, curvature)?;
            backend.module_for_ladder(family, curvature)?;
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
#[test]
fn device_rows_match_cpu_at_log_endpoints_tails_and_tiny_weights() {
    let Ok(backend) = PirlsRowBackend::probe() else {
        return;
    };
    let stream = backend.inner.ctx.default_stream();
    for family in PirlsRowFamily::ALL {
        let eta = match family {
            PirlsRowFamily::PoissonLog | PirlsRowFamily::GammaLog => {
                vec![-700.0, -2.0, 0.0, 2.0, 700.0]
            }
            PirlsRowFamily::BernoulliLogit => vec![-700.0, -2.0, 0.0, 2.0, 700.0],
            PirlsRowFamily::BernoulliProbit => vec![-6.0, -2.0, 0.0, 2.0, 6.0],
            PirlsRowFamily::BernoulliCLogLog => vec![-20.0, -2.0, 0.0, 1.0, 3.0],
            PirlsRowFamily::GaussianIdentity => vec![-700.0, -2.0, 0.0, 2.0, 700.0],
        };
        let y: Vec<f64> = eta
            .iter()
            .map(|&e| match family {
                PirlsRowFamily::PoissonLog | PirlsRowFamily::GammaLog => e.exp(),
                PirlsRowFamily::GaussianIdentity => e + 0.25,
                _ => {
                    if e >= 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
            })
            .collect();
        let prior = vec![f64::MIN_POSITIVE, 0.25, 1.0, 2.0, f64::MIN_POSITIVE];
        let eta_dev = stream.clone_htod(&eta).unwrap();
        let y_dev = stream.clone_htod(&y).unwrap();
        let prior_dev = stream.clone_htod(&prior).unwrap();
        for curvature in [CurvatureMode::Fisher, CurvatureMode::Observed] {
            let mut out = RowOutputDevBuffers::allocate(&stream, eta.len()).unwrap();
            launch_row_reweight_on_stream(
                backend,
                family,
                curvature,
                2.0,
                &stream,
                eta.len(),
                &eta_dev,
                &y_dev,
                &prior_dev,
                &mut out,
            )
            .unwrap();
            let status = stream.clone_dtoh(&out.status).unwrap();
            let mu = stream.clone_dtoh(&out.mu).unwrap();
            let grad = stream.clone_dtoh(&out.grad_eta).unwrap();
            let wh = stream.clone_dtoh(&out.w_hessian).unwrap();
            let ws = stream.clone_dtoh(&out.w_solver).unwrap();
            let dev = stream.clone_dtoh(&out.deviance).unwrap();
            for i in 0..eta.len() {
                let cpu = row_reweight_cpu_at(
                    i,
                    family,
                    curvature,
                    RowInput {
                        eta: eta[i],
                        y: y[i],
                        prior_weight: prior[i],
                    },
                    2.0,
                );
                match cpu {
                    Ok(cpu) => {
                        assert_eq!(status[i], status_codes::OK);
                        close(mu[i], cpu.mu, 2.0e-12);
                        close(grad[i], cpu.grad_eta, 2.0e-11);
                        close(wh[i], cpu.w_hessian, 2.0e-11);
                        close(ws[i], cpu.w_solver, 2.0e-11);
                        close(dev[i], cpu.deviance, 2.0e-11);
                    }
                    Err(_) => assert_ne!(status[i], status_codes::OK),
                }
            }
        }
    }
}

#[cfg(target_os = "linux")]
#[test]
fn failed_device_row_writes_only_its_status() {
    let Ok(backend) = PirlsRowBackend::probe() else {
        return;
    };
    let stream = backend.inner.ctx.default_stream();
    let y_dev = stream.clone_htod(&[1.0, 1.0]).unwrap();
    let prior_dev = stream.clone_htod(&[1.0, 1.0]).unwrap();
    let mut out = RowOutputDevBuffers::allocate(&stream, 2).unwrap();

    let valid_eta = stream.clone_htod(&[0.0, 0.0]).unwrap();
    launch_row_reweight_on_stream(
        backend,
        PirlsRowFamily::PoissonLog,
        CurvatureMode::Fisher,
        1.0,
        &stream,
        2,
        &valid_eta,
        &y_dev,
        &prior_dev,
        &mut out,
    )
    .unwrap();
    assert_eq!(stream.clone_dtoh(&out.mu).unwrap(), vec![1.0, 1.0]);

    let invalid_eta = stream.clone_htod(&[701.0, 0.0]).unwrap();
    launch_row_reweight_on_stream(
        backend,
        PirlsRowFamily::PoissonLog,
        CurvatureMode::Fisher,
        1.0,
        &stream,
        2,
        &invalid_eta,
        &y_dev,
        &prior_dev,
        &mut out,
    )
    .unwrap();
    let status = stream.clone_dtoh(&out.status).unwrap();
    let mu = stream.clone_dtoh(&out.mu).unwrap();
    assert_eq!(status, vec![status_codes::ETA_DOMAIN, status_codes::OK]);
    assert_eq!(mu, vec![1.0, 1.0]);
}

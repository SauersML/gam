use super::*;
use crate::scalar::Mixed;
use rand::{SeedableRng, rngs::SmallRng};

fn specification(k: usize, families: Vec<MeasurementFamily>, genes: usize) -> JointSpecification {
    JointSpecification {
        signatures: k,
        marks: vec![MarkKind::Recurrent],
        baseline_columns: 1,
        drive_columns: 1,
        entry_columns: 0,
        measurements: families,
        genetic_mean: vec![0.0; genes],
        genetic_precision: Array2::eye(genes),
    }
}

fn history(genes: usize) -> JointHistory {
    JointHistory {
        times: vec![0.0, 0.25, 0.5, 0.75, 1.0],
        exposure: vec![0.0, 0.5, 0.0, 0.5, 0.0],
        events: vec![None, None, Some(0), None, None],
        initially_at_risk: vec![true],
        baseline_design: Array2::ones((5, 1)),
        drive_design: Array2::ones((4, 1)),
        entry_design: vec![],
        genetics: vec![None; genes],
        measurements: vec![],
    }
}

#[test]
fn positive_decoder_is_a_sum_and_stays_in_the_log_domain() {
    let model = JointLikelihood::new(specification(2, vec![], 0)).unwrap();
    let mut theta = vec![0.0; model.layout.width];
    theta[model.layout.decoder.start] = 1.0;
    theta[model.layout.decoder.start + 1] = -0.7;
    let x = [-1.2, 2.0];
    let direct = (1.0
        + 1.0_f64.exp() * emission::softplus(&x[0])
        + (-0.7_f64).exp() * emission::softplus(&x[1]))
        / (1.0 + 1.0_f64.exp() + (-0.7_f64).exp());
    assert!((model.log_relative_activity(&theta, 0, &x).unwrap().exp() - direct).abs() < 1e-14);

    let one = JointLikelihood::new(specification(1, vec![], 0)).unwrap();
    let mut theta = vec![Mixed::seed(0.0, 0.0, 0.0); one.layout.width];
    theta[one.layout.decoder.start].base = 800.0;
    let out = one
        .log_relative_activity(&theta, 0, &[Mixed::seed(-800.0, 1.0, 1.0)])
        .unwrap();
    assert!((out.base - (2.0_f64.ln() - 800.0)).abs() < 1e-12);
    assert!((out.u - 0.5).abs() < 1e-14);
    assert!((out.uv - 0.25).abs() < 1e-14);
}

#[test]
fn zero_signatures_reproduce_the_counting_process_density() {
    let model = JointLikelihood::new(specification(0, vec![], 0)).unwrap();
    let mut theta = vec![0.0; model.layout.width];
    theta[model.layout.baseline.start] = -1.2;
    let h = history(0);
    let out = model.log_density(&theta, &h, &[], &[0.0; 5]).unwrap();
    assert!((out - (-1.2 - (-1.2_f64).exp())).abs() < 1e-14);
}

#[test]
fn measurement_families_are_normalized_and_have_stable_curvature() {
    let log_two = 2.0_f64.exp_m1().ln();
    let student = MeasurementFamily::StudentT;
    let density = emission::log_density(&student, 0.0, &0.0, &[0.0, log_two]).unwrap();
    assert!((density - 0.375_f64.ln()).abs() < 1e-14);
    assert!(
        emission::log_density(&student, 1e200, &0.0, &[0.0, log_two])
            .unwrap()
            .is_finite()
    );
    let eta = Mixed::seed(Mixed::seed(1e-100, 1.0, 1.0), 1.0, 1.0);
    let density = emission::log_density(
        &student,
        0.0,
        &eta,
        &[eta.constant_like(0.0), eta.constant_like(log_two)],
    )
    .unwrap();
    assert!((density.base.uv + 1.25).abs() < 1e-12);
    assert!((density.uv.uv - 1.875).abs() < 1e-12);

    let binary = MeasurementFamily::BinaryProbit;
    let sum: f64 = [0.0, 1.0]
        .iter()
        .map(|&y| emission::log_density(&binary, y, &3.5, &[]).unwrap().exp())
        .sum();
    assert!((sum - 1.0).abs() < 1e-14);
    let ordinal = MeasurementFamily::OrdinalProbit { categories: 4 };
    for location in [-40.0, -1.0, 1.0, 40.0] {
        let sum: f64 = (0..4)
            .map(|y| {
                emission::log_density(&ordinal, y as f64, &location, &[1.0_f64.exp_m1().ln(); 2])
                    .unwrap()
                    .exp()
            })
            .sum();
        assert!((sum - 1.0).abs() < 1e-13);
    }
    let count = MeasurementFamily::NegativeBinomial;
    let zero = emission::log_density(&count, 0.0, &3.0_f64.ln(), &[log_two]).unwrap();
    assert!((zero - 0.16_f64.ln()).abs() < 1e-14);
    let sum: f64 = (0..100)
        .map(|y| {
            emission::log_density(&count, y as f64, &3.0_f64.ln(), &[log_two])
                .unwrap()
                .exp()
        })
        .sum();
    assert!((sum - 1.0).abs() < 1e-12);
}

#[test]
fn missing_measurements_integrate_to_one_and_missing_genetics_are_latent() {
    let model =
        JointLikelihood::new(specification(0, vec![MeasurementFamily::StudentT], 1)).unwrap();
    let theta = vec![0.0; model.layout.width];
    let mut h = history(1);
    let absent = model.log_density(&theta, &h, &[0.3], &[0.0; 5]).unwrap();
    h.measurements.push(MeasurementRecord {
        node: 1,
        channel: 0,
        value: None,
        after_event: false,
    });
    assert_eq!(
        absent,
        model.log_density(&theta, &h, &[0.3], &[0.0; 5]).unwrap()
    );
    assert_eq!(model.latent_dimension(&h).unwrap(), 1);
    let gh = crate::chain::GaussHermite::new(17).unwrap();
    let integral: f64 = gh
        .nodes
        .iter()
        .zip(&gh.normal_weights)
        .map(|(&x, &w)| {
            let z = std::f64::consts::SQRT_2 * x;
            let value = model.log_density(&theta, &h, &[z], &[0.0; 5]).unwrap();
            w * (value + 0.5 * z * z + 0.5 * (2.0 * std::f64::consts::PI).ln()).exp()
        })
        .sum();
    assert!((integral - (-1.0_f64).exp()).abs() < 1e-14);
    h.genetics[0] = Some(0.3);
    assert_eq!(model.latent_dimension(&h).unwrap(), 0);
    assert_eq!(
        absent,
        model.log_density(&theta, &h, &[], &[0.0; 5]).unwrap()
    );
}

#[test]
fn genetic_drive_and_prevalence_condition_the_entry_state() {
    let mut spec = specification(1, vec![], 1);
    spec.marks[0] = MarkKind::Once;
    let model = JointLikelihood::new(spec).unwrap();
    let mut theta = vec![0.0; model.layout.width];
    theta[model.layout.baseline.start] = -800.0;
    theta[model.layout.entry.start + 1] = 1.0;
    theta[model.layout.entry.start + 2] = 2.0;
    theta[model.layout.drive.start + 1] = 1.0;
    let mut h = history(1);
    h.events.fill(None);
    h.genetics[0] = Some(0.7);
    let healthy = model.log_density(&theta, &h, &[0.7; 5], &[0.0; 5]).unwrap();
    h.genetics[0] = Some(0.0);
    let centered = model.log_density(&theta, &h, &[0.0; 5], &[0.0; 5]).unwrap();
    assert!((healthy - centered + 0.5 * 0.7_f64.powi(2)).abs() < 1e-13);
    let healthy = model.log_density(&theta, &h, &[2.0; 5], &[0.0; 5]).unwrap();
    h.initially_at_risk[0] = false;
    let prevalent = model.log_density(&theta, &h, &[2.0; 5], &[0.0; 5]).unwrap();
    assert!((prevalent - healthy - 2.0).abs() < 1e-13);
}

#[test]
fn event_jumps_act_after_event_intensity_and_respect_measurement_timing() {
    let model =
        JointLikelihood::new(specification(1, vec![MeasurementFamily::BinaryProbit], 0)).unwrap();
    let mut theta = vec![0.0; model.layout.width];
    theta[model.layout.measurement_location[0].start + 1] = 1.0;
    let mut h = history(0);
    let before = model.log_density(&theta, &h, &[0.0; 5], &[0.0; 5]).unwrap();
    theta[model.layout.jumps[0].as_ref().unwrap().start] = 1.0;
    let after = model.log_density(&theta, &h, &[0.0; 5], &[0.0; 5]).unwrap();
    let phi = (-2.0_f64.ln() * 0.25).exp();
    assert!((after - before + 0.5 * phi * phi / (1.0 - phi * phi)).abs() < 1e-13);
    h.measurements.push(MeasurementRecord {
        node: 2,
        channel: 0,
        value: Some(1.0),
        after_event: false,
    });
    let before = model.log_density(&theta, &h, &[0.0; 5], &[0.0; 5]).unwrap();
    h.measurements[0].after_event = true;
    let after = model.log_density(&theta, &h, &[0.0; 5], &[0.0; 5]).unwrap();
    assert!(
        (after - before - gam_math::probability::normal_logcdf(1.0) - 2.0_f64.ln()).abs() < 1e-13
    );
}

#[test]
fn complete_joint_density_derivatives_match_finite_differences() {
    let model = JointLikelihood::new(specification(
        2,
        vec![
            MeasurementFamily::StudentT,
            MeasurementFamily::BinaryProbit,
            MeasurementFamily::OrdinalProbit { categories: 4 },
            MeasurementFamily::NegativeBinomial,
        ],
        1,
    ))
    .unwrap();
    let theta: Vec<f64> = (0..model.layout.width)
        .map(|i| 0.02 * (i % 7) as f64)
        .collect();
    let mut h = history(1);
    for (channel, y) in [0.4, 1.0, 2.0, 3.0].iter().enumerate() {
        h.measurements.push(MeasurementRecord {
            node: 3,
            channel,
            value: Some(*y),
            after_event: false,
        });
    }
    let path: Vec<f64> = (0..model.latent_dimension(&h).unwrap())
        .map(|i| 0.03 * i as f64)
        .collect();
    let eps = 1e-4;
    for q in 0..theta.len() {
        let seeded: Vec<Mixed<f64>> = theta
            .iter()
            .enumerate()
            .map(|(i, &v)| Mixed::seed(v, f64::from(i == q), f64::from(i == q)))
            .collect();
        let states: Vec<Mixed<f64>> = path.iter().map(|&v| Mixed::seed(v, 0.0, 0.0)).collect();
        // Supplied reference sensitivities must be included, even though
        // this fixture's curve is only an algebraic derivative probe.
        let reference = vec![seeded[0].scale(0.1); 5];
        let value = model.log_density(&seeded, &h, &states, &reference).unwrap();
        let mut plus = theta.clone();
        let mut minus = theta.clone();
        plus[q] += eps;
        minus[q] -= eps;
        let vp = model
            .log_density(&plus, &h, &path, &vec![plus[0] * 0.1; 5])
            .unwrap();
        let vm = model
            .log_density(&minus, &h, &path, &vec![minus[0] * 0.1; 5])
            .unwrap();
        assert!(
            (value.u - (vp - vm) / (2.0 * eps)).abs() < 2e-6,
            "gradient {q}"
        );
        assert!(
            (value.uv - (vp + vm - 2.0 * value.base) / eps.powi(2)).abs() < 2e-5,
            "curvature {q}"
        );
    }
}

#[test]
fn structured_gaussian_limit_integrates_missing_scores_and_large_state_paths() {
    let mut spec = specification(8, vec![], 2);
    spec.genetic_mean = vec![0.4, -0.2];
    let model = JointLikelihood::new(spec).unwrap();
    let mut theta = vec![0.0; model.layout.width];
    theta[model.layout.baseline.start] = -800.0;
    let mut h = history(2);
    let nodes = 33;
    h.times = (0..nodes).map(|n| n as f64 / (nodes - 1) as f64).collect();
    h.exposure = vec![1.0 / (nodes - 1) as f64; nodes];
    h.exposure[0] = 0.0;
    h.events = vec![None; nodes];
    h.baseline_design = Array2::ones((nodes, 1));
    h.drive_design = Array2::ones((nodes - 1, 1));
    let slopes: Vec<[f64; 2]> = (0..8).map(|k| [0.1 * (k + 1) as f64, -0.15]).collect();
    for (k, b) in slopes.iter().enumerate() {
        for g in 0..2 {
            theta[model.layout.entry.start + k * 3 + g + 1] = b[g];
            theta[model.layout.drive.start + k * 3 + g + 1] = b[g];
        }
    }
    let posterior = model
        .laplace_posterior(
            &theta,
            &h,
            &vec![0.0; nodes],
            None,
            &PosteriorOptions::default(),
        )
        .unwrap();
    assert!(
        posterior.log_marginal.abs() < 1e-9,
        "Gaussian integral: {}",
        posterior.log_marginal
    );
    assert!(posterior.precision_entries < posterior.mode.len().pow(2) / 5);
    for i in 0..2 {
        assert!((posterior.mode[i] - model.spec.genetic_mean[i]).abs() < 1e-12);
        for j in 0..2 {
            assert!((posterior.genetic_covariance[[i, j]] - f64::from(i == j)).abs() < 1e-12);
        }
    }
    for n in [0, nodes / 2, nodes - 1] {
        for i in 0..8 {
            let mean = slopes[i][0] * 0.4 - slopes[i][1] * 0.2;
            assert!((posterior.mode[2 + n * 8 + i] - mean).abs() < 1e-12);
            for j in 0..8 {
                let expected =
                    f64::from(i == j) + slopes[i][0] * slopes[j][0] + slopes[i][1] * slopes[j][1];
                assert!((posterior.state_covariance[n][[i, j]] - expected).abs() < 1e-12);
            }
            for g in 0..2 {
                assert!(
                    (posterior.state_genetic_covariance[n][[i, g]] - slopes[i][g]).abs() < 1e-12
                );
            }
        }
    }
    let mut rng = SmallRng::seed_from_u64(735);
    let bank = model
        .integration(
            &theta,
            &h,
            &vec![0.0; nodes],
            Some(&posterior.mode),
            &IntegrationOptions::default(),
            &mut rng,
        )
        .unwrap();
    let corrected = bank
        .posterior(&theta, &vec![0.0; nodes], &IntegrationAccuracy::default())
        .unwrap();
    assert!(corrected.likelihood.log_marginal.abs() < 1e-9);
    assert!(corrected.likelihood.log_standard_error < 1e-12);
    assert!(corrected.maximum_moment_standard_error < 0.05);
}

#[test]
fn structured_posterior_assimilates_a_measurement_and_missing_values_do_not() {
    let model =
        JointLikelihood::new(specification(1, vec![MeasurementFamily::StudentT], 0)).unwrap();
    let mut theta = vec![0.0; model.layout.width];
    theta[model.layout.baseline.start] = -800.0;
    theta[model.layout.measurement_location[0].start + 1] = 1.0;
    theta[model.layout.measurement_shape[0].start] = 0.25_f64.ln();
    theta[model.layout.measurement_shape[0].start + 1] = 2.0_f64.exp_m1().ln();
    let mut h = history(0);
    h.events.fill(None);
    h.measurements.push(MeasurementRecord {
        node: 4,
        channel: 0,
        value: None,
        after_event: false,
    });
    let options = PosteriorOptions::default();
    let absent = model
        .laplace_posterior(&theta, &h, &[0.0; 5], None, &options)
        .unwrap();
    assert!(absent.mode.iter().all(|m| m.abs() < 1e-12));
    h.measurements[0].value = Some(2.0);
    let observed = model
        .laplace_posterior(&theta, &h, &[0.0; 5], Some(&absent.mode), &options)
        .unwrap();
    assert!(observed.mode[4] > 1.5);
    assert!(observed.mode[0] > 0.5 && observed.mode[0] < observed.mode[4]);
    assert!(observed.state_covariance[4][[0, 0]] < 0.2);
    assert!(observed.newton_decrement <= options.mode_tolerance);
}

#[test]
fn structured_rank_zero_integral_has_no_spurious_latent_normalization() {
    let model = JointLikelihood::new(specification(0, vec![], 1)).unwrap();
    let theta = vec![0.0; model.layout.width];
    let h = history(1);
    let out = model
        .laplace_posterior(&theta, &h, &[0.0; 5], None, &PosteriorOptions::default())
        .unwrap();
    assert!((out.log_marginal + 1.0).abs() < 1e-14);
    assert_eq!(out.mode, vec![0.0]);
    assert!((out.genetic_covariance[[0, 0]] - 1.0).abs() < 1e-14);
}

#[test]
fn importance_integral_conditions_missing_genetics_under_the_full_joint_prior() {
    let mut spec = specification(0, vec![], 2);
    let rho = 0.6;
    spec.genetic_precision =
        Array2::from_shape_vec((2, 2), vec![1.0, -rho, -rho, 1.0]).unwrap() / (1.0 - rho * rho);
    let model = JointLikelihood::new(spec).unwrap();
    let theta = vec![0.0; model.layout.width];
    let mut h = history(2);
    h.genetics[0] = Some(0.7);
    let mut rng = SmallRng::seed_from_u64(713);
    let options = IntegrationOptions {
        samples: 16384,
        ..IntegrationOptions::default()
    };
    let bank = model
        .integration(&theta, &h, &[0.0; 5], None, &options, &mut rng)
        .unwrap();
    let result = bank
        .posterior(&theta, &[0.0; 5], &IntegrationAccuracy::default())
        .unwrap();
    let expected = -1.0 - 0.5 * (0.7_f64.powi(2) + (2.0 * std::f64::consts::PI).ln());
    assert!((result.likelihood.log_marginal - expected).abs() < 1e-12);
    assert!(result.likelihood.log_standard_error < 1e-14);
    assert!((result.likelihood.effective_samples - options.samples as f64).abs() < 1e-7);
    assert!((result.mean[0] - rho * 0.7).abs() < 0.03);
    assert!((result.genetic_covariance[[0, 0]] - (1.0 - rho * rho)).abs() < 0.03);
}

#[test]
fn importance_correction_matches_an_independent_measurement_integral() {
    let model =
        JointLikelihood::new(specification(1, vec![MeasurementFamily::StudentT], 0)).unwrap();
    let mut theta = vec![0.0; model.layout.width];
    theta[model.layout.baseline.start] = -800.0;
    theta[model.layout.measurement_location[0].start + 1] = 1.0;
    let shape = model.layout.measurement_shape[0].clone();
    theta[shape.start] = 0.6_f64.ln();
    theta[shape.start + 1] = 2.0_f64.exp_m1().ln();
    let mut h = history(0);
    h.events.fill(None);
    h.measurements.push(MeasurementRecord {
        node: 4,
        channel: 0,
        value: Some(2.0),
        after_event: false,
    });
    // All unobserved path coordinates integrate out analytically. The oracle
    // therefore integrates only the final N(0,1) state, independent of the
    // block solver, importance proposal, and time-grid dimension.
    let oracle = |order| {
        let rule = gam_math::quadrature::gauss_hermite_rule(order).unwrap();
        let mut mass = 0.0;
        let mut first = 0.0;
        let mut second = 0.0;
        for (&x, &w) in rule.nodes.iter().zip(&rule.weights) {
            let z = std::f64::consts::SQRT_2 * x;
            let weight = w / std::f64::consts::PI.sqrt()
                * emission::log_density(
                    &MeasurementFamily::StudentT,
                    2.0,
                    &z,
                    &theta[shape.clone()],
                )
                .unwrap()
                .exp();
            mass += weight;
            first += weight * z;
            second += weight * z * z;
        }
        (
            mass.ln(),
            first / mass,
            second / mass - (first / mass).powi(2),
        )
    };
    let exact = oracle(257);
    let coarse = oracle(129);
    assert!(
        (coarse.0 - exact.0).abs() < 1e-7,
        "independent reference {coarse:?} vs {exact:?}"
    );
    let options = IntegrationOptions {
        samples: 16384,
        ..IntegrationOptions::default()
    };
    let mut rng = SmallRng::seed_from_u64(741);
    let bank = model
        .integration(&theta, &h, &[0.0; 5], None, &options, &mut rng)
        .unwrap();
    let out = bank
        .posterior(&theta, &[0.0; 5], &IntegrationAccuracy::default())
        .unwrap();
    assert!(
        (out.likelihood.log_marginal - exact.0).abs() < 5.0 * out.likelihood.log_standard_error
    );
    assert!((out.mean[4] - exact.1).abs() < 0.04);
    assert!((out.state_covariance[4][[0, 0]] - exact.2).abs() < 0.05);
    let strict = IntegrationAccuracy {
        log_standard_error: 1e-12,
        ..IntegrationAccuracy::default()
    };
    assert!(bank.log_marginal(&theta, &[0.0; 5], &strict).is_err());
    let strict_moments = IntegrationAccuracy {
        moment_standard_error: 1e-12,
        ..IntegrationAccuracy::default()
    };
    assert!(bank.posterior(&theta, &[0.0; 5], &strict_moments).is_err());
    eprintln!(
        "importance: log integral {} vs {}, SE {}, ESS {}; mean {} vs {}, variance {} vs {}",
        out.likelihood.log_marginal,
        exact.0,
        out.likelihood.log_standard_error,
        out.likelihood.effective_samples,
        out.mean[4],
        exact.1,
        out.state_covariance[4][[0, 0]],
        exact.2
    );
}

#[test]
fn sampled_objective_and_derivatives_include_the_same_reference_sensitivities() {
    let model =
        JointLikelihood::new(specification(1, vec![MeasurementFamily::BinaryProbit], 0)).unwrap();
    let mut theta = vec![0.0; model.layout.width];
    theta[model.layout.measurement_location[0].start + 1] = 0.7;
    let mut h = history(0);
    h.measurements.push(MeasurementRecord {
        node: 2,
        channel: 0,
        value: Some(1.0),
        after_event: true,
    });
    let mut rng = SmallRng::seed_from_u64(755);
    let bank = model
        .integration(
            &theta,
            &h,
            &[0.0; 5],
            None,
            &IntegrationOptions::default(),
            &mut rng,
        )
        .unwrap();
    let accuracy = IntegrationAccuracy {
        log_standard_error: 0.05,
        ..IntegrationAccuracy::default()
    };
    for q in [
        model.layout.baseline.start,
        model.layout.rates.start,
        model.layout.drive.start,
        model.layout.jumps[0].as_ref().unwrap().start,
        model.layout.measurement_location[0].start + 1,
    ] {
        let seeded: Vec<Mixed<f64>> = theta
            .iter()
            .enumerate()
            .map(|(j, &v)| Mixed::seed(v, f64::from(j == q), f64::from(j == q)))
            .collect();
        let value = bank
            .log_marginal(&seeded, &vec![seeded[0].scale(0.2); 5], &accuracy)
            .unwrap();
        let eps = 1e-4;
        let mut plus = theta.clone();
        let mut minus = theta.clone();
        plus[q] += eps;
        minus[q] -= eps;
        let vp = bank
            .log_marginal(&plus, &vec![plus[0] * 0.2; 5], &accuracy)
            .unwrap()
            .log_marginal;
        let vm = bank
            .log_marginal(&minus, &vec![minus[0] * 0.2; 5], &accuracy)
            .unwrap()
            .log_marginal;
        assert!(
            (value.log_marginal.u - (vp - vm) / (2.0 * eps)).abs() < 2e-7,
            "sampled gradient {q}"
        );
        assert!(
            (value.log_marginal.uv - (vp + vm - 2.0 * value.log_marginal.base) / eps.powi(2)).abs()
                < 3e-5,
            "sampled curvature {q}"
        );
    }
    let mut diffuse = theta.clone();
    diffuse[model.layout.rates.start] = 8.0;
    let error = bank
        .log_marginal(&diffuse, &[0.0; 5], &accuracy)
        .unwrap_err();
    assert!(error.to_string().contains("finite variance"), "{error}");
}

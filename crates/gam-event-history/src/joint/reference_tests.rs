use super::*;
use crate::scalar::Mixed;
use rand::{SeedableRng, rngs::SmallRng};

fn specification(k: usize, marks: Vec<MarkKind>, genes: usize) -> JointSpecification {
    JointSpecification {
        signatures: k,
        marks,
        baseline_columns: 1,
        drive_columns: 1,
        entry_columns: 0,
        measurements: vec![],
        genetic_mean: vec![0.0; genes],
        genetic_precision: Array2::eye(genes),
    }
}

fn profile(steps: usize, genes: usize) -> JointReferenceProfile {
    JointReferenceProfile {
        times: (0..=steps).map(|n| n as f64 / steps as f64).collect(),
        baseline_design: Array2::ones((steps + 1, 1)),
        drive_design: Array2::ones((steps, 1)),
        entry_design: vec![],
        genetics: vec![None; genes],
    }
}

#[test]
fn joint_reference_carries_competing_risk_sets_and_rejects_unresolved_steps() {
    let model = JointLikelihood::new(specification(
        0,
        vec![MarkKind::Once, MarkKind::Terminal, MarkKind::Recurrent],
        0,
    ))
    .unwrap();
    let theta = vec![0.2_f64.ln(), 0.1_f64.ln(), 0.1_f64.ln()];
    let profile = profile(32, 0);
    let options = ReferenceOptions {
        particles: 8192,
        ..ReferenceOptions::default()
    };
    let accuracy = ReferenceAccuracy::default();
    let mut rng = SmallRng::seed_from_u64(801);
    let bank = model
        .reference_bank(&theta, &profile, &options, &accuracy, &mut rng)
        .unwrap();
    let evolved = bank.evolve(&theta, &accuracy).unwrap();
    assert_eq!(evolved.coefficients(), &theta);
    assert_eq!(evolved.times().len(), 65);
    assert!(evolved.log_moments().iter().all(|m| m.abs() < 1e-13));
    let final_mass = &evolved.log_risk_mass()[64 * 3..];
    assert!((final_mass[0] + 0.3).abs() < 1e-12);
    assert!((final_mass[1] + 0.1).abs() < 1e-12);
    assert_eq!(final_mass[1], final_mass[2]);
    assert!(evolved.at(&[-0.01]).is_err());
    assert!(evolved.at(&[1.01]).is_err());
    assert_eq!(evolved.at(&[0.0, 1.0]).unwrap().len(), 6);
    let mut coarse = profile.clone();
    coarse.times = vec![0.0, 1.0];
    coarse.baseline_design = Array2::ones((2, 1));
    coarse.drive_design = Array2::ones((1, 1));
    assert!(
        model
            .reference_bank(&theta, &coarse, &options, &accuracy, &mut rng)
            .err()
            .unwrap()
            .to_string()
            .contains("refine the reference time grid")
    );
}

#[test]
fn joint_reference_derivatives_include_genetics_jumps_and_event_weights() {
    let model = JointLikelihood::new(specification(
        1,
        vec![MarkKind::Recurrent, MarkKind::Once],
        1,
    ))
    .unwrap();
    let mut theta = vec![0.0; model.layout.width];
    theta[model.layout.baseline.start] = -1.0;
    theta[model.layout.baseline.start + 1] = -1.3;
    theta[model.layout.drive.start + 1] = 0.3;
    theta[model.layout.entry.start + 1] = 0.4;
    theta[model.layout.jumps[0].as_ref().unwrap().start] = 0.8;
    let profile = profile(24, 1);
    let accuracy = ReferenceAccuracy {
        log_moment_standard_error: 0.1,
        minimum_risk_effective_samples: 16.0,
        maximum_step_hazard: 0.2,
    };
    let options = ReferenceOptions {
        particles: 512,
        ..ReferenceOptions::default()
    };
    let mut rng = SmallRng::seed_from_u64(813);
    let bank = model
        .reference_bank(&theta, &profile, &options, &accuracy, &mut rng)
        .unwrap();
    let value = bank.evolve(&theta, &accuracy).unwrap();
    let mut no_jump = theta.clone();
    no_jump[model.layout.jumps[0].as_ref().unwrap().start] = 0.0;
    let without = bank.evolve(&no_jump, &accuracy).unwrap();
    assert!(value.log_moments().last().unwrap() - without.log_moments().last().unwrap() > 0.01);
    let eps = 1e-4;
    for q in [
        model.layout.baseline.start,
        model.layout.decoder.start,
        model.layout.rates.start,
        model.layout.drive.start + 1,
        model.layout.entry.start + 1,
        model.layout.jumps[0].as_ref().unwrap().start,
    ] {
        let seeded: Vec<Mixed<f64>> = theta
            .iter()
            .enumerate()
            .map(|(j, &v)| Mixed::seed(v, f64::from(j == q), f64::from(j == q)))
            .collect();
        let jet = bank.evolve(&seeded, &accuracy).unwrap();
        let mut plus = theta.clone();
        let mut minus = theta.clone();
        plus[q] += eps;
        minus[q] -= eps;
        let plus = bank.evolve(&plus, &accuracy).unwrap();
        let minus = bank.evolve(&minus, &accuracy).unwrap();
        for ((j, p), m) in jet
            .log_moments()
            .iter()
            .chain(jet.log_risk_mass())
            .zip(plus.log_moments().iter().chain(plus.log_risk_mass()))
            .zip(minus.log_moments().iter().chain(minus.log_risk_mass()))
        {
            assert!(
                (j.u - (p - m) / (2.0 * eps)).abs() < 2e-7,
                "reference gradient {q}"
            );
            assert!(
                (j.uv - (p + m - 2.0 * j.base) / eps.powi(2)).abs() < 2e-5,
                "reference curvature {q}"
            );
        }
    }
}

#[test]
fn normalized_joint_integral_uses_the_returned_reference_state_and_total_score() {
    let model = JointLikelihood::new(specification(1, vec![MarkKind::Recurrent], 0)).unwrap();
    let mut theta = vec![0.0; model.layout.width];
    theta[0] = -1.0;
    let profile = profile(16, 0);
    let ref_options = ReferenceResolutionOptions {
        replicates: 4,
        initial_particles: 128,
        maximum_particles: 2048,
        maximum_rounds: 5,
        log_moment_tolerance: 0.1,
        risk_mass_tolerance: 0.05,
        minimum_risk_effective_samples: 16.0,
        maximum_step_hazard: 0.2,
        ..ReferenceResolutionOptions::default()
    };
    let mut rng = SmallRng::seed_from_u64(829);
    let (reference, initial) = model
        .resolve_reference(&theta, &profile, &ref_options, &mut rng)
        .unwrap();
    let h = JointHistory {
        times: vec![0.0, 0.25, 0.5, 0.75, 1.0],
        exposure: vec![0.0, 0.5, 0.0, 0.5, 0.0],
        events: vec![None, None, Some(0), None, None],
        initially_at_risk: vec![true],
        baseline_design: Array2::ones((5, 1)),
        drive_design: Array2::ones((4, 1)),
        entry_design: vec![],
        genetics: vec![],
        measurements: vec![],
    };
    let integration = model
        .integration(
            &theta,
            &h,
            &initial.reference().at(&h.times).unwrap(),
            None,
            &IntegrationOptions::default(),
            &mut rng,
        )
        .unwrap();
    let accuracy = IntegrationAccuracy {
        log_standard_error: 0.1,
        ..IntegrationAccuracy::default()
    };
    let (posterior, used) = integration
        .normalized_posterior(&theta, &reference, &accuracy)
        .unwrap();
    let same = integration
        .log_marginal(
            used.reference().coefficients(),
            &used.reference().at(&h.times).unwrap(),
            &accuracy,
        )
        .unwrap();
    assert_eq!(posterior.likelihood.log_marginal, same.log_marginal);
    let q = model.layout.jumps[0].as_ref().unwrap().start;
    let seeded: Vec<Mixed<f64>> = theta
        .iter()
        .enumerate()
        .map(|(j, &v)| Mixed::seed(v, f64::from(j == q), f64::from(j == q)))
        .collect();
    let (jet, evolved) = integration
        .normalized_log_marginal(&seeded, &reference, &accuracy)
        .unwrap();
    assert!(evolved.reference().log_moments().last().unwrap().u > 0.01);
    let eps = 1e-4;
    let mut plus = theta.clone();
    let mut minus = theta.clone();
    plus[q] += eps;
    minus[q] -= eps;
    let vp = integration
        .normalized_log_marginal(&plus, &reference, &accuracy)
        .unwrap()
        .0
        .log_marginal;
    let vm = integration
        .normalized_log_marginal(&minus, &reference, &accuracy)
        .unwrap()
        .0
        .log_marginal;
    assert!((jet.log_marginal.u - (vp - vm) / (2.0 * eps)).abs() < 2e-7);
    assert!(
        (jet.log_marginal.uv - (vp + vm - 2.0 * jet.log_marginal.base) / eps.powi(2)).abs() < 2e-5
    );
}

#[test]
fn reference_resolution_refines_large_steps_and_preserves_the_rank_zero_law() {
    let model = JointLikelihood::new(specification(
        0,
        vec![MarkKind::Once, MarkKind::Terminal, MarkKind::Recurrent],
        0,
    ))
    .unwrap();
    let theta = vec![0.2_f64.ln(), 0.1_f64.ln(), 0.1_f64.ln()];
    let p = profile(1, 0);
    let options = ReferenceResolutionOptions {
        replicates: 4,
        initial_particles: 64,
        maximum_particles: 128,
        maximum_rounds: 8,
        log_moment_tolerance: 1e-10,
        risk_mass_tolerance: 1e-10,
        ..ReferenceResolutionOptions::default()
    };
    let mut rng = SmallRng::seed_from_u64(851);
    let (resolved, out) = model
        .resolve_reference(&theta, &p, &options, &mut rng)
        .unwrap();
    assert!(out.report().rounds > 1);
    assert!(out.report().time_intervals >= 8);
    assert!(out.report().log_error_estimate < 1e-10);
    assert!(out.report().risk_error_estimate < 1e-10);
    let last = out.reference().log_risk_mass().len() - 3;
    assert!((out.reference().log_risk_mass()[last] + 0.3).abs() < 1e-12);
    assert!((out.reference().log_risk_mass()[last + 1] + 0.1).abs() < 1e-12);
    let again = resolved.evolve(&theta).unwrap();
    assert_eq!(
        again.reference().log_moments(),
        out.reference().log_moments()
    );
    let mut too_fast = theta.clone();
    too_fast[0] = 3.0;
    assert!(resolved.evolve(&too_fast).is_err());
    let short = ReferenceResolutionOptions {
        maximum_rounds: 1,
        ..options.clone()
    };
    assert!(
        model
            .resolve_reference(&theta, &p, &short, &mut rng)
            .is_err()
    );
    let tiny = ReferenceResolutionOptions {
        memory_limit_bytes: 1,
        ..options
    };
    assert!(
        model
            .resolve_reference(&theta, &p, &tiny, &mut rng)
            .is_err()
    );
    let mut unresolvable = p;
    unresolvable.times = vec![1.0, f64::from_bits(1.0_f64.to_bits() + 1)];
    assert!(
        model
            .reference_bank(
                &theta,
                &unresolvable,
                &ReferenceOptions::default(),
                &ReferenceAccuracy::default(),
                &mut rng
            )
            .err()
            .unwrap()
            .to_string()
            .contains("representable interior midpoint")
    );
}

#[test]
fn resolved_positive_reference_matches_the_static_survival_and_selected_law() {
    let model = JointLikelihood::new(specification(1, vec![MarkKind::Once], 0)).unwrap();
    let mut theta = vec![0.0; model.layout.width];
    theta[0] = -1.2;
    theta[model.layout.rates.start] = -16.0;
    let mut p = profile(24, 0);
    for time in &mut p.times {
        *time *= 6.0;
    }
    let options = ReferenceResolutionOptions {
        replicates: 4,
        initial_particles: 128,
        maximum_particles: 8192,
        maximum_rounds: 8,
        log_moment_tolerance: 0.05,
        risk_mass_tolerance: 0.01,
        minimum_risk_effective_samples: 16.0,
        ..ReferenceResolutionOptions::default()
    };
    let mut rng = SmallRng::seed_from_u64(863);
    let (_, out) = model
        .resolve_reference(&theta, &p, &options, &mut rng)
        .unwrap();
    let target = (-6.0 * (-1.2_f64).exp()).exp();
    let rule = gam_math::quadrature::gauss_hermite_rule(65).unwrap();
    let rates: Vec<f64> = rule
        .nodes
        .iter()
        .map(|x| (1.0 + emission::softplus(&(std::f64::consts::SQRT_2 * x))) / 2.0)
        .collect();
    let integrals = |a: f64| {
        let mut s = 0.0;
        let mut m = 0.0;
        for (&r, &w) in rates.iter().zip(&rule.weights) {
            let mass = w / std::f64::consts::PI.sqrt() * (-a * r).exp();
            s += mass;
            m += mass * r;
        }
        (s, m / s)
    };
    let (mut lower, mut upper) = (0.0, 64.0);
    for _ in 0..60 {
        let middle = 0.5 * (lower + upper);
        if integrals(middle).0 > target {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    let exact_moment = integrals(0.5 * (lower + upper)).1.ln();
    let log_moment = *out.reference().log_moments().last().unwrap();
    let survival = out.reference().log_risk_mass().last().unwrap().exp();
    assert!((survival - target).abs() < options.risk_mass_tolerance);
    assert!((log_moment - exact_moment).abs() < options.log_moment_tolerance);
    assert!(out.report().log_error_estimate <= options.log_moment_tolerance);
    assert!(out.report().risk_error_estimate <= options.risk_mass_tolerance);
    eprintln!(
        "resolved positive reference: S={survival} target={target}; log M={log_moment} static={exact_moment}; {:?}",
        out.report()
    );
}

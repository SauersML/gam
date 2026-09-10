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
    let error = evolved.diagnostics().maximum_risk_mass_standard_error;
    assert!((final_mass[0].exp() - (-0.3_f64).exp()).abs() < 5.0 * error + 0.002);
    assert!((final_mass[1].exp() - (-0.1_f64).exp()).abs() < 5.0 * error + 0.002);
    assert_eq!(final_mass[1], final_mass[2]);
    assert!(evolved.at(&[-0.01]).is_err());
    assert!(evolved.at(&[1.01]).is_err());
    assert_eq!(evolved.at(&[0.0, 1.0]).unwrap().len(), 6);
    let strict = ReferenceAccuracy {
        risk_mass_standard_error: 1e-8,
        ..accuracy.clone()
    };
    assert!(bank.evolve(&theta, &strict).is_err());
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
        risk_mass_standard_error: 0.1,
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
    let ref_options = ReferenceOptions {
        particles: 512,
        ..ReferenceOptions::default()
    };
    let ref_accuracy = ReferenceAccuracy {
        log_moment_standard_error: 0.1,
        risk_mass_standard_error: 0.1,
        minimum_risk_effective_samples: 16.0,
        maximum_step_hazard: 0.2,
    };
    let mut rng = SmallRng::seed_from_u64(829);
    let reference = model
        .reference_bank(&theta, &profile, &ref_options, &ref_accuracy, &mut rng)
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
    let initial = reference.evolve(&theta, &ref_accuracy).unwrap();
    let integration = model
        .integration(
            &theta,
            &h,
            &initial.at(&h.times).unwrap(),
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
        .normalized_posterior(&theta, &reference, &ref_accuracy, &accuracy)
        .unwrap();
    let same = integration
        .log_marginal(used.coefficients(), &used.at(&h.times).unwrap(), &accuracy)
        .unwrap();
    assert_eq!(posterior.likelihood.log_marginal, same.log_marginal);
    let q = model.layout.jumps[0].as_ref().unwrap().start;
    let seeded: Vec<Mixed<f64>> = theta
        .iter()
        .enumerate()
        .map(|(j, &v)| Mixed::seed(v, f64::from(j == q), f64::from(j == q)))
        .collect();
    let (jet, evolved) = integration
        .normalized_log_marginal(&seeded, &reference, &ref_accuracy, &accuracy)
        .unwrap();
    assert!(evolved.log_moments().last().unwrap().u > 0.01);
    let eps = 1e-4;
    let mut plus = theta.clone();
    let mut minus = theta.clone();
    plus[q] += eps;
    minus[q] -= eps;
    let vp = integration
        .normalized_log_marginal(&plus, &reference, &ref_accuracy, &accuracy)
        .unwrap()
        .0
        .log_marginal;
    let vm = integration
        .normalized_log_marginal(&minus, &reference, &ref_accuracy, &accuracy)
        .unwrap()
        .0
        .log_marginal;
    assert!((jet.log_marginal.u - (vp - vm) / (2.0 * eps)).abs() < 2e-7);
    assert!(
        (jet.log_marginal.uv - (vp + vm - 2.0 * jet.log_marginal.base) / eps.powi(2)).abs() < 2e-5
    );
}

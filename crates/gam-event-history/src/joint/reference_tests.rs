use super::reference::{JointReferenceBank, ReferenceGrid};
use crate::test_support::{Bound, agrees};
use super::*;
use crate::scalar::Rows;
use rand::{SeedableRng, rngs::SmallRng};
use std::sync::Arc;

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

fn profile(steps: usize, horizon: f64, genes: usize) -> JointReferenceProfile {
    JointReferenceProfile {
        times: (0..=steps).map(|n| horizon * n as f64 / steps as f64).collect(),
        baseline_design: Array2::ones((steps + 1, 1)),
        drive_design: Array2::ones((steps, 1)),
        entry_design: vec![],
        genetics: vec![None; genes],
    }
}

/// Requested accuracy of a fixture's resolution; the tests' bars come from the
/// returned reports, not from these requests. Sixteen replicates keep the
/// Bonferroni Student margin moderate, and the declared sampling error rate is
/// the family-wise rate of a sampling error beyond its margin.
fn options(log: f64, risk: f64, particles: usize) -> ReferenceResolutionOptions {
    ReferenceResolutionOptions {
        replicates: 16,
        initial_particles: particles,
        log_moment_tolerance: log,
        risk_mass_tolerance: risk,
        sampling_error_rate: 1e-3,
    }
}

fn exact(values: &[f64]) -> Vec<Bound> {
    values.iter().map(|&v| Bound::exact(v)).collect()
}

/// Coefficients seeded along `u = e_i` and `v = e_j` over bounded scalars. A
/// result's value is `.base.base`, its `u` derivative `.base.rows[0]` and its
/// mixed `uv` derivative `.rows[0].rows[0]`.
fn seeded(theta: &[f64], i: Option<usize>, j: Option<usize>) -> Vec<Rows<Rows<Bound, 1>, 1>> {
    theta
        .iter()
        .enumerate()
        .map(|(q, &v)| {
            Rows::seed(
                Rows::seed(Bound::exact(v), [f64::from(Some(q) == i)]),
                [f64::from(Some(q) == j)],
            )
        })
        .collect()
}

/// Two routes to one derivative. A derivative that is structurally zero, from
/// an exactly zero feature, must be met exactly; any other derivative must
/// agree within the two routes' rounding bounds above the oracle's magnitude.
pub(super) fn derivative_agrees(production: &Bound, oracle: &Bound, name: &str) {
    if oracle.value == 0.0 && oracle.scale == 0.0 {
        assert_eq!(
            production.value, 0.0,
            "{name}: a structurally zero derivative must be exactly zero"
        );
    } else {
        agrees(production, oracle, name);
    }
}

/// How a compared cell is judged, declared before the run.
#[derive(Clone, Copy)]
pub(super) enum Arm {
    /// Agreement between resolved values, or an exactly structural zero.
    Agreement,
    /// Agreement between resolved values; no structural zero is admitted.
    Strict,
    /// A cell this fixture's depth leaves unresolved (#2961 A2 finding):
    /// printed with its margins, never counted as agreement.
    Unresolved,
}

/// Prints every cell's margins, then asserts each agreement arm. An asserted
/// cell's oracle must exceed twice the two routes' bar: then `|p - o| <= bar`
/// forces `|p| > bar`, so both routes resolve a nonzero value of one sign and
/// the agreement is between resolved values rather than roundoff.
pub(super) fn judge(cells: &[(String, Bound, Bound, Arm)]) {
    for (name, production, oracle, arm) in cells {
        let label = match arm {
            Arm::Unresolved => "UNRESOLVED ",
            _ => "",
        };
        println!(
            "CELL {label}{name}: oracle {} production {} mu_oracle {} mu_production {} bar/|oracle| {}",
            oracle.value,
            production.value,
            oracle.scale,
            production.scale,
            production.bar(oracle) / oracle.value.abs()
        );
    }
    for (name, production, oracle, arm) in cells {
        let structural = oracle.value == 0.0 && oracle.scale == 0.0;
        match arm {
            Arm::Unresolved => {}
            Arm::Agreement if structural => derivative_agrees(production, oracle, name),
            Arm::Agreement | Arm::Strict => {
                assert!(
                    oracle.value.abs() > 2.0 * production.bar(oracle),
                    "{name}: {} does not exceed twice its bar {}; the cell is unresolved",
                    oracle.value,
                    production.bar(oracle)
                );
                agrees(production, oracle, name);
            }
        }
    }
}

/// The reverse sweep is the production derivative route. A forward jet replay
/// of the same fixed population is its oracle on every coefficient; the sweep
/// over a seeded scalar must match second-order jets; `scatter` must be the
/// adjoint of `at`; and the pooled pullback must match jets through the
/// resolved curve. A removed jump moves the normaliser beyond both routes'
/// rounding, and the population needed multiple events somewhere, so jumps and
/// event weights are exercised.
#[test]
fn reference_pullback_matches_jets_on_every_coefficient_and_through_pooling() {
    let model = Arc::new(
        JointLikelihood::new(specification(
            2,
            vec![MarkKind::Recurrent, MarkKind::Once, MarkKind::Terminal],
            1,
        ))
        .unwrap(),
    );
    let jump = model.layout.jumps[0].as_ref().unwrap().start;
    let mut theta: Vec<_> = (0..model.layout.width)
        .map(|q| 0.1 * (q as f64).sin())
        .collect();
    theta[0] = -1.2;
    theta[1] = -1.4;
    theta[2] = -2.0;
    theta[model.layout.drive.start + 1] = 0.3;
    theta[model.layout.entry.start + 1] = 0.4;
    theta[jump] = 0.6;
    let p = profile(16, 1.0, 1);
    let mut rng = SmallRng::seed_from_u64(971);
    let (bank, curve) = JointReferenceBank::generate(&model, &theta, &ReferenceGrid::new(p.clone(), 0).unwrap(), 128, &mut rng).unwrap();
    assert!(curve.diagnostics().omitted_event_mass > 0.0);
    let mut no_jump = theta.clone();
    no_jump[jump] = 0.0;
    let moved = bank.evolve(&model, &exact(&theta)).unwrap();
    let still = bank.evolve(&model, &exact(&no_jump)).unwrap();
    let (moved, still) = (
        moved.log_moments().last().unwrap(),
        still.log_moments().last().unwrap(),
    );
    assert!(
        (moved.value - still.value).abs() > moved.bar(still),
        "a jump must move the normaliser: {} vs {}",
        moved.value,
        still.value
    );
    let rows = curve.log_moments().len();
    let moment_adjoint: Vec<f64> = (0..rows).map(|i| (0.37 * i as f64).sin()).collect();
    let mass_adjoint: Vec<f64> = (0..rows).map(|i| (0.53 * i as f64).cos()).collect();
    let plain = bank
        .pullback(&model, &theta, &moment_adjoint, &mass_adjoint)
        .unwrap();
    let hand = bank
        .pullback(
            &model,
            &exact(&theta),
            &exact(&moment_adjoint),
            &exact(&mass_adjoint),
        )
        .unwrap();
    // Every compared cell is collected and printed before any is asserted, so a
    // failing cell never hides the margins of the others. Each cell's arm is
    // declared here: at this depth the second-order sweep and the all-bound
    // pooled cells are the #2961 A2 finding, never counted as agreement.
    let mut cells: Vec<(String, Bound, Bound, Arm)> = Vec::new();
    for q in 0..theta.len() {
        assert_eq!(plain[q], hand[q].value, "the bounded sweep repeats the f64 sweep");
        let jet = bank.evolve(&model, &seeded(&theta, Some(q), None)).unwrap();
        let directional = (0..rows).fold(Bound::exact(0.0), |acc, i| {
            acc.add(&jet.log_moments()[i].base.rows[0].scale(moment_adjoint[i]))
                .add(&jet.log_risk_mass()[i].base.rows[0].scale(mass_adjoint[i]))
        });
        cells.push((format!("coefficient {q}"), hand[q], directional, Arm::Agreement));
    }
    let exact_mixed = |values: &[f64]| {
        values
            .iter()
            .map(|&v| Rows::seed(Rows::seed(Bound::exact(v), [0.0]), [0.0]))
            .collect::<Vec<_>>()
    };
    for (i, q) in [
        (0, jump),
        (model.layout.rates.start, model.layout.entry.start + 1),
        (model.layout.decoder.start + 1, model.layout.drive.start + 1),
    ] {
        let jet = bank
            .evolve(&model, &seeded(&theta, Some(i), Some(q)))
            .unwrap();
        let second = (0..rows).fold(Bound::exact(0.0), |acc, r| {
            acc.add(&jet.log_moments()[r].rows[0].rows[0].scale(moment_adjoint[r]))
                .add(&jet.log_risk_mass()[r].rows[0].rows[0].scale(mass_adjoint[r]))
        });
        let sweep = bank
            .pullback(
                &model,
                &seeded(&theta, Some(q), None),
                &exact_mixed(&moment_adjoint),
                &exact_mixed(&mass_adjoint),
            )
            .unwrap();
        cells.push((
            format!("second derivative {i},{q}"),
            sweep[i].base.rows[0],
            second,
            Arm::Unresolved,
        ));
    }
    let times = [0.0, 0.33, 0.7, 1.0];
    let marks = 3;
    let at_adjoint: Vec<f64> = (0..times.len() * marks)
        .map(|i| 1.0 + 0.1 * i as f64)
        .collect();
    // The scattered grid is a computed output, so it is formed over the bounded
    // scalar and enters the grid route with its own rounding charged.
    let bounded_curve = bank.evolve(&model, &exact(&theta)).unwrap();
    let mut grid = vec![Bound::exact(0.0); rows];
    bounded_curve
        .scatter(&times, &exact(&at_adjoint), &mut grid)
        .unwrap();
    for q in [0, model.layout.rates.start, jump] {
        let jet = bank.evolve(&model, &seeded(&theta, Some(q), None)).unwrap();
        let through_at = jet
            .at(&times)
            .unwrap()
            .iter()
            .zip(&at_adjoint)
            .fold(Bound::exact(0.0), |acc, (m, &a)| acc.add(&m.base.rows[0].scale(a)));
        let through_grid = jet
            .log_moments()
            .iter()
            .zip(&grid)
            .fold(Bound::exact(0.0), |acc, (m, g)| acc.add(&m.base.rows[0].mul(g)));
        cells.push((format!("scatter adjoint {q}"), through_grid, through_at, Arm::Strict));
    }
    assert!(
        bounded_curve
            .scatter(&[1.5], &exact(&at_adjoint[..marks]), &mut grid)
            .is_err()
    );
    let (resolved, out) =
        ResolvedReference::resolve(&model, &theta, &p, &options(0.5, 0.1, 64), 971).unwrap();
    let pooled_rows = out.reference().log_moments().len();
    let pooled_adjoint: Vec<f64> = (0..pooled_rows).map(|i| (0.29 * i as f64).cos()).collect();
    let pooled = resolved
        .pullback(&exact(&theta), &exact(&pooled_adjoint))
        .unwrap();
    for q in [
        0,
        model.layout.entry.start + 1,
        model.layout.rates.start,
        jump,
    ] {
        let jet = resolved.evolve(&seeded(&theta, Some(q), None)).unwrap();
        let directional = jet
            .reference()
            .log_moments()
            .iter()
            .zip(&pooled_adjoint)
            .fold(Bound::exact(0.0), |acc, (m, &a)| acc.add(&m.base.rows[0].scale(a)));
        cells.push((format!("pooled coefficient {q}"), pooled[q], directional, Arm::Unresolved));
    }
    // Where the production route's rounding bound builds, printed whether or
    // not the cells pass: the sweep with the adjoint on one interval's rows
    // only, so the reverse recursion runs through `n` intervals, beside the
    // forward rows' own bounds at that node and after pooling.
    let forward = bank.evolve(&model, &exact(&theta)).unwrap();
    let forward_mu = |rows_n: std::ops::Range<usize>| {
        forward.log_moments()[rows_n.clone()]
            .iter()
            .chain(&forward.log_risk_mass()[rows_n])
            .map(|b| b.scale)
            .fold(0.0, f64::max)
    };
    for n in [1, 2, 4, 8, 16] {
        let rows_n = (2 * n - 1) * marks..(2 * n + 1) * marks;
        let restrict = |adjoint: &[f64]| {
            (0..rows)
                .map(|r| if rows_n.contains(&r) { adjoint[r] } else { 0.0 })
                .collect::<Vec<_>>()
        };
        let part = bank
            .pullback(
                &model,
                &exact(&theta),
                &exact(&restrict(&moment_adjoint)),
                &exact(&restrict(&mass_adjoint)),
            )
            .unwrap();
        for q in [0, model.layout.rates.start, model.layout.entry.start + 1] {
            println!(
                "STAGE depth {n} coefficient {q}: production {} mu_production {} forward_row_mu {}",
                part[q].value,
                part[q].scale,
                forward_mu(rows_n.clone())
            );
        }
    }
    let pooled_forward = resolved.evolve(&exact(&theta)).unwrap();
    println!(
        "STAGE pooling: single-bank forward rows max mu {}, pooled forward rows max mu {}",
        forward_mu(0..rows),
        pooled_forward
            .reference()
            .log_moments()
            .iter()
            .chain(pooled_forward.reference().log_risk_mass())
            .map(|b| b.scale)
            .fold(0.0, f64::max)
    );
    judge(&cells);
}

/// Forward-over-reverse at a depth that resolves it: the sweep over a seeded
/// scalar must meet second-order jets on the deep fixture's three coefficient
/// pairs, now on four intervals and 32 particles per risk set. Each cell asserts
/// its premise through `judge` (the oracle exceeds twice its bar), so a pair
/// this fixture cannot resolve fails as unresolved rather than passing on
/// roundoff.
#[test]
fn second_order_sweep_matches_jets_where_resolved() {
    let model = Arc::new(
        JointLikelihood::new(specification(
            2,
            vec![MarkKind::Recurrent, MarkKind::Once, MarkKind::Terminal],
            1,
        ))
        .unwrap(),
    );
    let jump = model.layout.jumps[0].as_ref().unwrap().start;
    let mut theta: Vec<_> = (0..model.layout.width)
        .map(|q| 0.1 * (q as f64).sin())
        .collect();
    theta[0] = -1.2;
    theta[1] = -1.4;
    theta[2] = -2.0;
    theta[model.layout.drive.start + 1] = 0.3;
    theta[model.layout.entry.start + 1] = 0.4;
    theta[jump] = 0.6;
    let mut rng = SmallRng::seed_from_u64(971);
    let grid = ReferenceGrid::new(profile(4, 1.0, 1), 0).unwrap();
    let (bank, curve) = JointReferenceBank::generate(&model, &theta, &grid, 32, &mut rng).unwrap();
    let rows = curve.log_moments().len();
    let moment_adjoint: Vec<f64> = (0..rows).map(|i| (0.37 * i as f64).sin()).collect();
    let mass_adjoint: Vec<f64> = (0..rows).map(|i| (0.53 * i as f64).cos()).collect();
    let exact_mixed = |values: &[f64]| {
        values
            .iter()
            .map(|&v| Rows::seed(Rows::seed(Bound::exact(v), [0.0]), [0.0]))
            .collect::<Vec<_>>()
    };
    let mut cells = Vec::new();
    for (i, q) in [
        (0, jump),
        (model.layout.rates.start, model.layout.entry.start + 1),
        (model.layout.decoder.start + 1, model.layout.drive.start + 1),
    ] {
        let jet = bank
            .evolve(&model, &seeded(&theta, Some(i), Some(q)))
            .unwrap();
        let second = (0..rows).fold(Bound::exact(0.0), |acc, r| {
            acc.add(&jet.log_moments()[r].rows[0].rows[0].scale(moment_adjoint[r]))
                .add(&jet.log_risk_mass()[r].rows[0].rows[0].scale(mass_adjoint[r]))
        });
        let sweep = bank
            .pullback(
                &model,
                &seeded(&theta, Some(q), None),
                &exact_mixed(&moment_adjoint),
                &exact_mixed(&mass_adjoint),
            )
            .unwrap();
        cells.push((
            format!("shallow second derivative {i},{q}"),
            sweep[i].base.rows[0],
            second,
            Arm::Agreement,
        ));
    }
    judge(&cells);
}

/// Exact rank-zero survival for a log-linear baseline `a + b t`.
fn cumulative_hazard(a: f64, b: f64, t: f64) -> f64 {
    if b == 0.0 {
        t * a.exp()
    } else {
        ((a + b * t).exp() - a.exp()) / b
    }
}

/// A signature whose rate underflows to zero is static, as the principal
/// posterior represents it: its half step keeps every state exactly, so the
/// normaliser keeps its entry value within the rounding of the event weights,
/// and the reverse sweep stays finite. The rare recurrent mark keeps the
/// weights' rounding far inside their own bounds.
#[test]
fn static_signature_axes_evolve_exactly_and_pull_back_finitely() {
    let model = JointLikelihood::new(specification(1, vec![MarkKind::Recurrent], 0)).unwrap();
    let mut theta = vec![0.0; model.layout.width];
    theta[model.layout.baseline.start] = -30.0;
    theta[model.layout.rates.start] = -800.0;
    let p = profile(8, 1.0, 0);
    let mut rng = SmallRng::seed_from_u64(29);
    let (bank, curve) = JointReferenceBank::generate(&model, &theta, &ReferenceGrid::new(p.clone(), 0).unwrap(), 64, &mut rng).unwrap();
    let bounded = bank.evolve(&model, &exact(&theta)).unwrap();
    let entry = &bounded.log_moments()[0];
    for (n, moment) in bounded.log_moments().iter().enumerate() {
        agrees(moment, entry, &format!("static normaliser row {n}"));
    }
    let rows = curve.log_moments().len();
    let adjoint: Vec<f64> = (0..rows).map(|i| (0.3 * i as f64).sin()).collect();
    let gradient = bank
        .pullback(&model, &theta, &adjoint, &vec![0.0; rows])
        .unwrap();
    assert!(gradient.iter().all(|g| g.is_finite()));
}

/// At K=0 no state exists, so the controlled particle evolution must agree
/// with the exact competing-risk law within its own reported risk error. The
/// unrefined two-interval evolution misses that law by more than the resolved
/// estimate, so the agreement is not vacuous.
#[test]
fn rank_zero_particle_evolution_agrees_with_the_exact_law_within_its_resolution() {
    let model = Arc::new(
        JointLikelihood::new(JointSpecification {
            signatures: 0,
            marks: vec![MarkKind::Once, MarkKind::Terminal, MarkKind::Recurrent],
            baseline_columns: 2,
            drive_columns: 1,
            entry_columns: 0,
            measurements: vec![],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
        .unwrap(),
    );
    let theta = [-1.2, 0.3, -2.5, 0.2, -0.5, 0.0];
    let grid = |steps: usize| {
        let times: Vec<f64> = (0..=steps).map(|n| 6.0 * n as f64 / steps as f64).collect();
        JointReferenceProfile {
            baseline_design: Array2::from_shape_fn((steps + 1, 2), |(i, j)| {
                if j == 0 { 1.0 } else { times[i] }
            }),
            drive_design: Array2::ones((steps, 1)),
            entry_design: vec![],
            genetics: vec![],
            times,
        }
    };
    let exact_law = |t: f64| {
        let terminal = cumulative_hazard(theta[2], theta[3], t);
        [
            (-terminal - cumulative_hazard(theta[0], theta[1], t)).exp(),
            (-terminal).exp(),
            (-terminal).exp(),
        ]
    };
    let mut rng = SmallRng::seed_from_u64(8271);
    let (_, coarse) = JointReferenceBank::generate(&model, &theta, &ReferenceGrid::new(grid(2), 0).unwrap(), 2, &mut rng).unwrap();
    let coarse_gap = coarse
        .times()
        .iter()
        .enumerate()
        .flat_map(|(n, &t)| {
            let target = exact_law(t);
            (0..3).map(move |d| (d, n, target[d]))
        })
        .map(|(d, n, target)| (coarse.log_risk_mass()[n * 3 + d].exp() - target).abs())
        .fold(0.0, f64::max);
    let (resolved, out) = ResolvedReference::resolve(
        &model,
        &theta,
        &grid(2),
        &ReferenceResolutionOptions {
            replicates: 2,
            initial_particles: 2,
            log_moment_tolerance: 1e-4,
            risk_mass_tolerance: 1e-4,
            sampling_error_rate: 1e-3,
        },
        8271,
    )
    .unwrap();
    let report = out.report();
    assert!(report.rounds > 1, "{report:?}");
    assert_eq!(report.log_error_estimate, 0.0);
    assert_eq!(report.omitted_event_mass, 0.0);
    assert!(
        coarse_gap > report.risk_error_estimate,
        "unrefined gap {coarse_gap} within the resolved estimate {}",
        report.risk_error_estimate
    );
    let curve = out.reference();
    assert_eq!(curve.origin(), 0.0);
    assert_eq!(curve.horizon(), 6.0);
    for (n, &t) in curve.times().iter().enumerate() {
        let target = exact_law(t);
        for d in 0..3 {
            assert_eq!(curve.log_moments()[n * 3 + d], 0.0);
            let gap = (curve.log_risk_mass()[n * 3 + d].exp() - target[d]).abs();
            assert!(
                gap <= report.risk_error_estimate,
                "time {t}, mark {d}: gap {gap} above estimate {}",
                report.risk_error_estimate
            );
        }
    }
    // The fixed ensembles re-evaluate identically at their own coefficients.
    let again = resolved.evolve(&theta).unwrap();
    assert_eq!(again.reference().log_risk_mass(), curve.log_risk_mass());
    let refusal = curve.at(&[6.5]).err().unwrap().to_string();
    assert!(refusal.contains("declared origin 0 and horizon 6"), "{refusal}");
}

/// Standard normal expectation of `f` by Gauss-Hermite quadrature.
fn normal_expectation(mean: f64, sd: f64, f: impl Fn(f64) -> f64) -> f64 {
    let rule = gam_math::quadrature::gauss_hermite_rule(65).expect("a 65-node Gauss-Hermite rule");
    rule.nodes
        .iter()
        .zip(&rule.weights)
        .map(|(&x, &w)| w * f(mean + std::f64::consts::SQRT_2 * sd * x))
        .sum::<f64>()
        / std::f64::consts::PI.sqrt()
}

/// With no killing and no jump the reference state law is Gaussian in closed
/// form: the entry mean depends on an observed and a missing genetic score, the
/// missing score follows its conditional law, and the OU drive pulls the state
/// towards its genetic mean. The resolved normaliser follows that evolving law
/// within its estimate. A fresh stationary law at each age differs from it by
/// more than twice the estimate at an early age, so the curve cannot be within
/// its estimate of both.
#[test]
fn reference_normaliser_follows_the_evolved_entry_and_genetic_law() {
    let model = Arc::new(
        JointLikelihood::new(JointSpecification {
            signatures: 1,
            marks: vec![MarkKind::Recurrent],
            baseline_columns: 1,
            drive_columns: 1,
            entry_columns: 0,
            measurements: vec![],
            genetic_mean: vec![0.5, -0.3],
            genetic_precision: ndarray::arr2(&[[2.0, 0.6], [0.6, 1.5]]),
        })
        .unwrap(),
    );
    let nu = 0.7_f64;
    let logit = 0.4_f64;
    let mut theta = vec![0.0; model.layout.width];
    theta[model.layout.baseline.start] = -3.0;
    theta[model.layout.decoder.start] = logit;
    theta[model.layout.rates.start] = nu.exp_m1().ln();
    let drive = [-0.5, 0.0, 0.8];
    let entry = [2.0, 0.3, 0.6];
    theta[model.layout.drive.clone()].copy_from_slice(&drive);
    theta[model.layout.entry.clone()].copy_from_slice(&entry);
    let mut p = profile(8, 2.0, 2);
    p.genetics = vec![Some(1.0), None];
    let (_, out) =
        ResolvedReference::resolve(&model, &theta, &p, &options(0.05, 0.01, 512), 4021).unwrap();
    // g1 | g0 = 1: mean mu1 - Q10/Q11 (g0 - mu0), variance 1/Q11.
    let missing_mean = -0.3 - 0.6 / 1.5 * (1.0 - 0.5);
    let missing_variance = 1.0 / 1.5;
    let weight = logit.exp() / (1.0 + logit.exp());
    let log_moment = |mean: f64, variance: f64| {
        normal_expectation(mean, variance.sqrt(), |x| {
            1.0 - weight + weight * super::emission::softplus(&x)
        })
        .ln()
    };
    let entry_mean = entry[0] + entry[1] + entry[2] * missing_mean;
    let drive_mean = drive[0] + drive[1] + drive[2] * missing_mean;
    let curve = out.reference();
    let report = out.report();
    for (n, &t) in curve.times().iter().enumerate() {
        let phi = (-nu * t).exp();
        let loading = phi * entry[2] + (1.0 - phi) * drive[2];
        let expected = log_moment(
            phi * entry_mean + (1.0 - phi) * drive_mean,
            1.0 + loading * loading * missing_variance,
        );
        let gap = (curve.log_moments()[n] - expected).abs();
        assert!(
            gap <= report.log_error_estimate,
            "time {t}: gap {gap} above estimate {}",
            report.log_error_estimate
        );
        // Nothing kills this population, so its weights never lose mass.
        assert_eq!(curve.log_risk_mass()[n], 0.0);
    }
    let stationary = log_moment(
        drive_mean,
        1.0 + drive[2] * drive[2] * missing_variance,
    );
    assert!(
        (curve.at(&[0.25]).unwrap()[0] - stationary).abs() > 2.0 * report.log_error_estimate,
        "the early normaliser must reflect the entry law, not a stationary law"
    );
}

/// A saved reference regenerates bit-identical populations from its record,
/// after a serialization round trip and inside thread pools of different
/// sizes. A different seed draws a different population, so the equality is
/// not produced by ignoring the draws.
#[test]
fn a_saved_reference_regenerates_identical_populations_at_any_thread_count() {
    let model = Arc::new(
        JointLikelihood::new(specification(
            1,
            vec![MarkKind::Recurrent, MarkKind::Once],
            1,
        ))
        .unwrap(),
    );
    let mut theta = vec![0.0; model.layout.width];
    theta[model.layout.baseline.start] = -1.0;
    theta[model.layout.baseline.start + 1] = -1.5;
    theta[model.layout.entry.start + 1] = 0.4;
    theta[model.layout.jumps[0].as_ref().unwrap().start] = 1.0;
    let p = profile(4, 1.0, 1);
    let accuracy = options(0.5, 0.05, 64);
    let (resolved, out) = ResolvedReference::resolve(&model, &theta, &p, &accuracy, 17).unwrap();
    let encoded = serde_json::to_string(&resolved.saved()).unwrap();
    let decoded: SavedReference = serde_json::from_str(&encoded).unwrap();
    // The reverse sweep adds risk-set shares in set order, so a restored
    // reference's gradient is bitwise identical at every thread count too.
    let adjoint: Vec<f64> = (0..out.reference().log_moments().len())
        .map(|i| (0.41 * i as f64).sin())
        .collect();
    let mut gradients = Vec::new();
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(pool.install(rayon::current_num_threads), threads);
        let (restored, again) = pool
            .install(|| ResolvedReference::restore(&model, &decoded))
            .unwrap();
        assert_eq!(again.reference().log_moments(), out.reference().log_moments());
        assert_eq!(again.reference().log_risk_mass(), out.reference().log_risk_mass());
        assert_eq!(again.report().log_error_estimate, out.report().log_error_estimate);
        assert_eq!(restored.saved().time_refinements, resolved.saved().time_refinements);
        gradients.push(pool.install(|| restored.pullback(&theta, &adjoint)).unwrap());
    }
    assert!(gradients[0].iter().any(|g| *g != 0.0));
    assert_eq!(gradients[0], gradients[1]);
    let (_, other) = ResolvedReference::resolve(&model, &theta, &p, &accuracy, 18).unwrap();
    assert_ne!(other.reference().log_moments(), out.reference().log_moments());
}

/// A nearly static state (rate softplus(-16)) keeps its entry law, so the
/// survival and the selected risk-set moment have closed forms by quadrature;
/// the resolved curve must meet both within its reported estimates.
#[test]
fn resolved_positive_reference_matches_the_static_survival_and_selected_law() {
    let model = Arc::new(JointLikelihood::new(specification(1, vec![MarkKind::Once], 0)).unwrap());
    let mut theta = vec![0.0; model.layout.width];
    theta[0] = -1.2;
    theta[model.layout.rates.start] = -16.0;
    let p = profile(24, 6.0, 0);
    let (_, out) =
        ResolvedReference::resolve(&model, &theta, &p, &options(0.05, 0.01, 128), 863).unwrap();
    let target = (-6.0 * (-1.2_f64).exp()).exp();
    let rule = gam_math::quadrature::gauss_hermite_rule(65).unwrap();
    let rates: Vec<f64> = rule
        .nodes
        .iter()
        .map(|x| (1.0 + super::emission::softplus(&(std::f64::consts::SQRT_2 * x))) / 2.0)
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
    assert!(
        (survival - target).abs() <= out.report().risk_error_estimate,
        "survival {survival} vs {target}, estimate {}",
        out.report().risk_error_estimate
    );
    assert!(
        (log_moment - exact_moment).abs() <= out.report().log_error_estimate,
        "log moment {log_moment} vs {exact_moment}, estimate {}",
        out.report().log_error_estimate
    );
}

#[test]
fn reference_resolution_refuses_unrepresentable_grids_budgets_and_foreign_banks() {
    let model = Arc::new(
        JointLikelihood::new(specification(1, vec![MarkKind::Recurrent], 0)).unwrap(),
    );
    let theta = vec![-1.0; model.layout.width];
    let mut rng = SmallRng::seed_from_u64(851);
    let mut unresolvable = profile(1, 1.0, 0);
    unresolvable.times = vec![1.0, f64::from_bits(1.0_f64.to_bits() + 1)];
    let refusal = ReferenceGrid::new(unresolvable, 0).err().unwrap().to_string();
    assert!(refusal.contains("representable interior midpoint"), "{refusal}");
    let p = profile(4, 1.0, 0);
    // Refused from the storage estimate, before anything is allocated: every
    // particle stores at least one byte, so a round with more particles than
    // this machine's budget has bytes exceeds it.
    let oversized = options(
        0.05,
        0.01,
        gam_runtime::resource::ResourcePolicy::default_library().max_single_materialization_bytes + 1,
    );
    let refusal = ResolvedReference::resolve(&model, &theta, &p, &oversized, 851)
        .err()
        .unwrap()
        .to_string();
    assert!(refusal.contains("materialization budget"), "{refusal}");
    let single = ReferenceResolutionOptions {
        replicates: 1,
        ..options(0.05, 0.01, 16)
    };
    assert!(ResolvedReference::resolve(&model, &theta, &p, &single, 851).is_err());
    let (bank, _) = JointReferenceBank::generate(&model, &theta, &ReferenceGrid::new(p.clone(), 0).unwrap(), 4, &mut rng).unwrap();
    let other = JointLikelihood::new(specification(2, vec![MarkKind::Recurrent], 0)).unwrap();
    let refusal = bank
        .evolve(&other, &vec![-1.0; other.layout.width])
        .err()
        .unwrap()
        .to_string();
    assert!(refusal.contains("different model specification"), "{refusal}");
}

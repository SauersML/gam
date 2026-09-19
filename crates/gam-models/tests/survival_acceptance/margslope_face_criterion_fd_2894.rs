//! gam#2894 bar 2: the survival marginal-slope outer criterion prices `½·log|Zᵀ M_true Z|` on
//! the inner mode's active face, and within one active set its analytic ρ-gradient and ρ-Hessian
//! must be the derivatives of that value.
//!
//! The fixture is the #979 repro arm the defect was measured on: 160 observations, a Duchon
//! smooth over three PCs with six centers, a linear baseline. The seed probe evaluates the real
//! criterion at the seed, halfway to the returned optimum and at the optimum, and grades every
//! ρ coordinate's analytic gradient against a difference of the criterion value itself. At the
//! optimum it also grades every column of the analytic outer Hessian, the curvature the
//! certificate reads (#2905), against a difference of the analytic gradient. The differences
//! are formed here, because production takes no finite difference (SPEC rule 2).
//!
//! A stencil whose points straddle a change of active face differences two criteria, not one.
//! Its halving ladder does not settle, since successive estimates disagree by far more than
//! roundoff. That coordinate is reported and not graded, and each point must still grade most
//! of its coordinates.
//!
//! gam#2952: the runner projects the caller's ρ seed into the derived resolvability domain
//! (#2812), so a seed coordinate can sit exactly on a face of the probe's box, and so can a
//! railed optimum. Every evaluation stays inside that box, so a coordinate without room on both
//! sides is differenced by the one-sided three-point rule toward the room it has, and a
//! coordinate whose box collapses is refused. The positive control
//! `difference_ladder_stays_in_the_box_and_recovers_known_derivatives_2952` runs the same
//! ladder on closed-form functions through a sampler that refuses any point outside the box.
//!
//! A coordinate exactly on its upper face publishes the KKT projection of its derivative. The
//! unified evaluator reports a negative entry there, the infeasible upper-bound multiplier, as 0
//! (`reml_laml_evaluate`, #197 corrected by #2615), which is the rule the optimizer's
//! `project_gradient_vector` applies. Such a coordinate is graded against the same projection of
//! its one-sided difference, so a railed optimum's outward pull is neither graded as a defect nor
//! hidden from the grade lines. Lower faces are not projected (#2514).

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::utils::splitmix64;
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_solve::estimate::outer_eval_capture::{
    OuterSeedEvaluation, OuterSeedOrder, OuterSeedProbe, observe_next_outer_seed,
};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

const N: usize = 160;
const CENTERS: usize = 6;
const N_PCS: usize = 3;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// The `repro979_survival_margslope` generating process, bit for bit.
fn build_dataset() -> gam_data::EncodedDataset {
    let mut headers = vec![
        "entry_age".to_string(),
        "exit_age".to_string(),
        "event".to_string(),
        "prs_z".to_string(),
    ];
    for i in 0..N_PCS {
        headers.push(format!("PC{}", i + 1));
    }
    headers.push("sex".to_string());
    let mut state: u64 = 0xD0E1_2345_6789_ABCD;
    let mut rows: Vec<StringRecord> = Vec::with_capacity(N);
    for _ in 0..N {
        let pcs: Vec<f64> = (0..N_PCS).map(|_| next_gauss(&mut state) * 0.5).collect();
        let prs = next_gauss(&mut state);
        let sex = if next_unit(&mut state) < 0.5 { 1.0 } else { 0.0 };
        let entry = 40.0 + 5.0 * next_unit(&mut state);
        let followup = 0.5 + 8.0 * next_unit(&mut state);
        let exit = entry + followup;
        let score = 0.3 * prs + 0.4 * pcs[0] - 0.3 * pcs[1] + 0.2 * pcs[2]
            + 0.15 * sex
            + 0.2 * next_gauss(&mut state);
        let event = if score > 0.0 { 1 } else { 0 };
        let mut record = vec![
            entry.to_string(),
            exit.to_string(),
            event.to_string(),
            prs.to_string(),
        ];
        for pc in &pcs {
            record.push(pc.to_string());
        }
        record.push(sex.to_string());
        rows.push(StringRecord::from(record));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode the #979 repro dataset")
}

fn fit_config() -> (String, FitConfig) {
    let duchon = format!("duchon(PC1, PC2, PC3, centers={CENTERS}, order=1)");
    let formula = format!("Surv(entry_age, exit_age, event) ~ {duchon} + sex");
    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("prs_z".to_string()),
        slope_formula: Some(duchon),
        baseline_target: "linear".to_string(),
        ..FitConfig::default()
    };
    (formula, config)
}

/// Which difference quotient a coordinate's room inside the seed box admits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stencil {
    /// `(f(θ+h) − f(θ−h))/2h`.
    Central,
    /// `(−3f(θ) + 4f(θ+h) − f(θ+2h))/2h`, toward the room above a coordinate on or near its
    /// lower face.
    Forward,
    /// `(3f(θ) − 4f(θ−h) + f(θ−2h))/2h`, toward the room below a coordinate on or near its
    /// upper face.
    Backward,
}

/// One ρ coordinate's analytic gradient against its difference.
#[derive(Debug)]
struct CoordinateGrade {
    point: &'static str,
    coordinate: usize,
    analytic: f64,
    difference: f64,
    stencil: Stencil,
    /// `θ_j` sits on the probe box's upper face, where the published entry is the KKT projection.
    on_upper_face: bool,
    step: f64,
    /// `|D(h) − D(h/2)|` at the accepted rung: the ladder's own disagreement.
    settle: f64,
}

/// One column of the analytic outer Hessian against a difference of the analytic gradient
/// along that column's coordinate.
#[derive(Debug)]
struct HessianColumnGrade {
    column: usize,
    analytic: Array1<f64>,
    difference: Array1<f64>,
    stencil: Stencil,
    step: f64,
    /// `max_i |D(h)_i − D(h/2)_i|` at the accepted rung.
    settle: f64,
}

/// The stencil along a coordinate at `theta_j` inside `[lower_j, upper_j]`, and the first rung
/// of its halving ladder: one hundredth of the coordinate's scale, and never more than half the
/// room the stencil reaches into. A central stencil needs the nominal step `ε^¼·(1 + |θ_j|)` on
/// both sides, the rule the #2765 mode-response gate in this binary uses. Otherwise the
/// one-sided rule reaches `2h` toward the larger room.
fn stencil_along(theta_j: f64, lower_j: f64, upper_j: f64) -> Result<(Stencil, f64), String> {
    let left_room = (theta_j - lower_j).max(0.0);
    let right_room = (upper_j - theta_j).max(0.0);
    let scale = 1.0 + theta_j.abs();
    let nominal_step = f64::EPSILON.powf(0.25) * scale;
    let first_rung = |reach: f64| (1.0e-2 * scale).min(0.5 * reach);
    if left_room >= nominal_step && right_room >= nominal_step {
        Ok((Stencil::Central, first_rung(left_room.min(right_room))))
    } else if right_room >= left_room && right_room > 0.0 {
        Ok((Stencil::Forward, first_rung(0.5 * right_room)))
    } else if left_room > 0.0 {
        Ok((Stencil::Backward, first_rung(0.5 * left_room)))
    } else {
        Err(format!("the seed box [{lower_j}, {upper_j}] collapses at {theta_j}"))
    }
}

/// Difference quotients of `sample`, the quantity at a displacement along one coordinate, on a
/// six-rung halving ladder of `stencil`. Returns the rung estimate that agrees best with its
/// predecessor, its step, and that rung's largest entrywise disagreement.
fn difference_ladder(
    stencil: Stencil,
    first_rung: f64,
    sample: &mut dyn FnMut(f64) -> Result<Array1<f64>, String>,
) -> Result<(Array1<f64>, f64, f64), String> {
    let at_theta = match stencil {
        Stencil::Central => None,
        Stencil::Forward | Stencil::Backward => Some(sample(0.0)?),
    };
    let mut step = first_rung;
    let mut estimates: Vec<(f64, Array1<f64>)> = Vec::new();
    for _ in 0..6 {
        let estimate = match (stencil, at_theta.as_ref()) {
            (Stencil::Forward, Some(base)) => {
                let one = sample(step)?;
                let two = sample(2.0 * step)?;
                (4.0_f64 * &one - &two - 3.0_f64 * base) / (2.0 * step)
            }
            (Stencil::Backward, Some(base)) => {
                let one = sample(-step)?;
                let two = sample(-2.0 * step)?;
                (3.0_f64 * base - 4.0_f64 * &one + &two) / (2.0 * step)
            }
            _ => {
                let plus = sample(step)?;
                let minus = sample(-step)?;
                (&plus - &minus) / (2.0 * step)
            }
        };
        estimates.push((step, estimate));
        step *= 0.5;
    }
    let (index, settle) = (1..estimates.len())
        .map(|i| {
            let disagreement = (&estimates[i].1 - &estimates[i - 1].1)
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            (i, disagreement)
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .expect("the ladder has six rungs");
    Ok((estimates[index].1.clone(), estimates[index].0, settle))
}

/// Differences of the criterion along coordinate `j`, at `order` so that every sample on one
/// ladder is the same kind of evaluation, read out of each sample by `read`.
fn probe_difference(
    probe: &mut dyn OuterSeedProbe,
    theta: &Array1<f64>,
    j: usize,
    order: OuterSeedOrder,
    read: fn(OuterSeedEvaluation) -> Option<Array1<f64>>,
) -> Result<(Array1<f64>, Stencil, f64, f64), String> {
    let (lower_j, upper_j) = (probe.layout().lower[j], probe.layout().upper[j]);
    let (stencil, first_rung) = stencil_along(theta[j], lower_j, upper_j)
        .map_err(|reason| format!("theta[{j}]: {reason}"))?;
    let mut sample = |offset: f64| -> Result<Array1<f64>, String> {
        let mut displaced = theta.clone();
        displaced[j] += offset;
        let evaluation = probe
            .evaluate(&displaced, order)
            .map_err(|error| format!("theta[{j}] displaced by {offset:e}: {error}"))?;
        read(evaluation)
            .ok_or_else(|| format!("theta[{j}] displaced by {offset:e}: no {order:?} published"))
    };
    let (difference, step, settle) = difference_ladder(stencil, first_rung, &mut sample)?;
    Ok((difference, stencil, step, settle))
}

/// The entry a coordinate publishes for `derivative`: 0 for a negative entry on the upper face,
/// the infeasible upper-bound multiplier `reml_laml_evaluate` projects out (#197, #2615), and the
/// derivative itself everywhere else.
fn published_derivative(on_upper_face: bool, derivative: f64) -> f64 {
    if on_upper_face && derivative < 0.0 {
        0.0
    } else {
        derivative
    }
}

fn grade_point(
    probe: &mut dyn OuterSeedProbe,
    point: &'static str,
    theta: &Array1<f64>,
) -> Result<Vec<CoordinateGrade>, String> {
    let evaluation = probe
        .evaluate(theta, OuterSeedOrder::ValueAndGradient)
        .map_err(|error| format!("{point}: analytic evaluation: {error}"))?;
    if !evaluation.cost.is_finite() {
        return Err(format!("{point}: the criterion is not finite: {}", evaluation.cost));
    }
    let gradient = evaluation
        .gradient
        .ok_or_else(|| format!("{point}: the gradient evaluation published no gradient"))?;
    let mut grades = Vec::with_capacity(gradient.len());
    for j in 0..gradient.len() {
        let (difference, stencil, step, settle) =
            probe_difference(probe, theta, j, OuterSeedOrder::Value, |evaluation| {
                Some(Array1::from_elem(1, evaluation.cost))
            })
            .map_err(|reason| format!("{point}: {reason}"))?;
        grades.push(CoordinateGrade {
            point,
            coordinate: j,
            analytic: gradient[j],
            difference: difference[0],
            stencil,
            on_upper_face: theta[j] >= probe.layout().upper[j],
            step,
            settle,
        });
    }
    Ok(grades)
}

fn grade_hessian(
    probe: &mut dyn OuterSeedProbe,
    theta: &Array1<f64>,
) -> Result<Vec<HessianColumnGrade>, String> {
    let evaluation = probe
        .evaluate(theta, OuterSeedOrder::ValueGradientHessian)
        .map_err(|error| format!("optimum: analytic Hessian evaluation: {error}"))?;
    let hessian = evaluation
        .hessian
        .ok_or_else(|| "optimum: the objective served no analytic outer Hessian".to_string())?;
    let dim = theta.len();
    if hessian.dim() != (dim, dim) {
        return Err(format!(
            "optimum: the analytic Hessian has shape {:?}, expected ({dim}, {dim})",
            hessian.dim()
        ));
    }
    let mut grades = Vec::with_capacity(dim);
    for column in 0..dim {
        let (difference, stencil, step, settle) = probe_difference(
            probe,
            theta,
            column,
            OuterSeedOrder::ValueAndGradient,
            |evaluation| evaluation.gradient,
        )
        .map_err(|reason| format!("optimum Hessian: {reason}"))?;
        grades.push(HessianColumnGrade {
            column,
            analytic: hessian.column(column).to_owned(),
            difference,
            stencil,
            step,
            settle,
        });
    }
    Ok(grades)
}

/// gam#2952 positive control for the harness. On closed-form functions, sampled through a
/// closure that refuses any point outside the box, the ladder must take the forward rule on the
/// lower face, the central rule inside and the backward rule on the upper face, and recover the
/// known derivative at all three under the bars the criterion is graded at. The central rule on
/// a face must be refused by that sampler, which is the refusal gam#2952 hit before any
/// evaluation.
#[test]
fn difference_ladder_stays_in_the_box_and_recovers_known_derivatives_2952() {
    let (lower, upper) = (-1.5_f64, 2.0_f64);
    let value = |t: f64| (0.7 * t).exp() + (1.3 * t).sin();
    let derivative = |t: f64| 0.7 * (0.7 * t).exp() + 1.3 * (1.3 * t).cos();
    let boxed_sampler = move |theta_j: f64| {
        move |offset: f64| -> Result<Array1<f64>, String> {
            let point = theta_j + offset;
            if !(lower..=upper).contains(&point) {
                return Err(format!("sampled {point} outside [{lower}, {upper}]"));
            }
            Ok(Array1::from_elem(1, value(point)))
        }
    };
    for (theta_j, expected) in [
        (lower, Stencil::Forward),
        (0.25, Stencil::Central),
        (upper, Stencil::Backward),
    ] {
        let (stencil, first_rung) =
            stencil_along(theta_j, lower, upper).expect("the control box has room");
        assert_eq!(stencil, expected, "the stencil at {theta_j}");
        let (difference, step, settle) =
            difference_ladder(stencil, first_rung, &mut boxed_sampler(theta_j))
                .unwrap_or_else(|reason| panic!("the ladder at {theta_j} refused: {reason}"));
        let truth = derivative(theta_j);
        let scale = truth.abs().max(1.0);
        assert!(
            settle <= 1.0e-4 * scale,
            "the ladder at {theta_j} did not settle: {settle:e} at step {step:e}"
        );
        assert!(
            (difference[0] - truth).abs() <= 1.0e-4 * scale + 10.0 * settle,
            "at {theta_j}: {stencil:?} difference {:e} vs derivative {truth:e}",
            difference[0]
        );
    }
    assert!(
        difference_ladder(Stencil::Central, 1.0e-2, &mut boxed_sampler(lower)).is_err(),
        "the sampler must refuse a central stencil on the lower face"
    );
    assert!(
        stencil_along(1.0, 1.0, 1.0).is_err(),
        "a collapsed box admits no stencil"
    );
    assert_eq!(
        published_derivative(true, -3.0),
        0.0,
        "a negative entry on the upper face is the infeasible multiplier, published as 0"
    );
    assert_eq!(
        published_derivative(true, 2.0),
        2.0,
        "a feasible descent entry on the upper face is published unchanged"
    );
    assert_eq!(
        published_derivative(false, -3.0),
        -3.0,
        "an entry off the upper face is not projected"
    );
}

#[test]
fn survival_marginal_slope_face_criterion_derivatives_match_central_differences_2894() {
    super::initialize_cpu_fitting();
    gam_runtime::test_support::install_diagnostic_logger();
    let data = build_dataset();
    let (formula, config) = fit_config();

    // Tracked red (gam#2952), measured at 29221e56b1 with the value-route fix (job 1333188): this
    // base fit refuses after about 460 s, so nothing below is graded. Its unarmed outer search
    // certifies rho = [8.419, -0.209, 3.039, 0.469, 0.279, -1.823, -0.840] (|Pg| = 6.705e-4 within
    // 8.807e-3, outer Hessian PD), but that mode's ambient posterior precision has one negative
    // direction, so the fit arms the Jeffreys/Firth prior and re-solves from the unarmed mode. The
    // armed search's inner solve does not converge ("did not converge after 117 cycle(s) ... the
    // KKT certificate refused the iterate: active_set_incomplete") and the fit refuses. Without
    // the fix the same unarmed optimum certifies and the armed search runs past the 600 s timeout.
    let optimum = match fit_from_formula(&formula, &data, &config) {
        Ok(FitResult::SurvivalMarginalSlope(result)) => result.fit.log_lambdas.clone(),
        Ok(_) => panic!("a marginal-slope survival formula returned another model class"),
        Err(error) => panic!("the 160x6 repro fit refused: {error}"),
    };

    type Grades = (Vec<CoordinateGrade>, Vec<HessianColumnGrade>);
    let captured: Rc<RefCell<Option<Result<Grades, String>>>> = Rc::new(RefCell::new(None));
    let sink = Rc::clone(&captured);
    observe_next_outer_seed(
        0,
        Box::new(
            move |probe: &mut dyn OuterSeedProbe| -> Result<(), gam_solve::estimate::EstimationError> {
                let layout = probe.layout().clone();
                eprintln!(
                    "[2952-LAYOUT] seed={:?} lower={:?} upper={:?} optimum={:?}",
                    layout.seed.to_vec(),
                    layout.lower.to_vec(),
                    layout.upper.to_vec(),
                    optimum.to_vec(),
                );
                let outcome = (|| -> Result<Grades, String> {
                    if optimum.len() != layout.seed.len() {
                        return Err(format!(
                            "the returned optimum has {} coordinates, the seed {}",
                            optimum.len(),
                            layout.seed.len()
                        ));
                    }
                    if let Some(i) = (0..layout.seed.len()).find(|&i| {
                        !(layout.lower[i] <= layout.seed[i] && layout.seed[i] <= layout.upper[i])
                    }) {
                        return Err(format!(
                            "the lent seed leaves its box at coordinate {i}: {} outside [{}, {}]",
                            layout.seed[i], layout.lower[i], layout.upper[i]
                        ));
                    }
                    let clamp = |theta: Array1<f64>| {
                        Array1::from_shape_fn(theta.len(), |i| {
                            theta[i].clamp(layout.lower[i], layout.upper[i])
                        })
                    };
                    let seed = layout.seed.clone();
                    let halfway = clamp((&seed + &optimum) * 0.5);
                    let returned = clamp(optimum.clone());
                    let mut grades = Vec::new();
                    for (point, theta) in
                        [("seed", seed), ("halfway", halfway), ("optimum", returned.clone())]
                    {
                        grades.extend(grade_point(probe, point, &theta)?);
                    }
                    let hessian_grades = grade_hessian(probe, &returned)?;
                    Ok((grades, hessian_grades))
                })();
                *sink.borrow_mut() = Some(outcome);
                Ok(())
            },
        ),
    );
    let refit = fit_from_formula(&formula, &data, &config);
    let (grades, hessian_grades) = captured
        .borrow_mut()
        .take()
        .unwrap_or_else(|| panic!("the outer runner lent no seed probe: {:?}", refit.err()))
        .unwrap_or_else(|reason| panic!("the seed probe refused: {reason}"));

    let mut graded_per_point = std::collections::BTreeMap::<&'static str, usize>::new();
    let mut failures = Vec::new();
    for grade in &grades {
        let expected = published_derivative(grade.on_upper_face, grade.difference);
        let scale = grade.analytic.abs().max(1.0);
        let settled = grade.settle <= 1.0e-4 * scale;
        eprintln!(
            "[2894-FD] point={} coordinate={} analytic={:.9e} difference={:.9e} expected={:.9e} \
             on_upper_face={} stencil={:?} step={:e} settle={:e} graded={settled}",
            grade.point,
            grade.coordinate,
            grade.analytic,
            grade.difference,
            expected,
            grade.on_upper_face,
            grade.stencil,
            grade.step,
            grade.settle,
        );
        if !settled {
            continue;
        }
        *graded_per_point.entry(grade.point).or_default() += 1;
        if (grade.analytic - expected).abs() > 1.0e-4 * scale + 10.0 * grade.settle {
            failures.push(format!(
                "{} coordinate {}: analytic {:e} vs {:?} difference {:e}, published as {:e}",
                grade.point,
                grade.coordinate,
                grade.analytic,
                grade.stencil,
                grade.difference,
                expected
            ));
        }
    }
    let mut settled_columns = 0usize;
    for grade in &hessian_grades {
        let scale = grade
            .analytic
            .iter()
            .fold(1.0_f64, |acc, value| acc.max(value.abs()));
        let settled = grade.settle <= 1.0e-4 * scale;
        for row in 0..grade.analytic.len() {
            eprintln!(
                "[2894-FD-HESSIAN] row={row} column={} analytic={:.9e} difference={:.9e} \
                 stencil={:?} step={:e} settle={:e} graded={settled}",
                grade.column,
                grade.analytic[row],
                grade.difference[row],
                grade.stencil,
                grade.step,
                grade.settle,
            );
        }
        if !settled {
            continue;
        }
        settled_columns += 1;
        for row in 0..grade.analytic.len() {
            let entry_scale = grade.analytic[row].abs().max(1.0);
            if (grade.analytic[row] - grade.difference[row]).abs()
                > 1.0e-4 * entry_scale + 10.0 * grade.settle
            {
                failures.push(format!(
                    "optimum Hessian [{row}, {}]: analytic {:e} vs {:?} difference {:e}",
                    grade.column, grade.analytic[row], grade.stencil, grade.difference[row]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "derivatives disagree with the criterion: {failures:?}");
    for point in ["seed", "halfway", "optimum"] {
        let graded = graded_per_point.get(point).copied().unwrap_or(0);
        assert!(
            graded >= 4,
            "{point}: only {graded} of {} coordinates had a settled difference",
            hessian_grades.len()
        );
    }
    assert!(
        settled_columns >= 4,
        "optimum: only {settled_columns} of {} Hessian columns had a settled difference",
        hessian_grades.len()
    );
}

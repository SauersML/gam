use super::chain::{
    AtomTransition, GaussHermite, Grid, backward_axis_bases, forward_operators,
    interpolate_at_inner_points,
};
use super::cohort::{
    CovariateSegment, Event, EventHistoryCohort, MarkKind, SubjectHistory, SubjectNodes,
    design_rows, expand_nodes,
};
use super::covariance::{
    DirectionEvidence, DirectionProfile, empirical_bayes_ridge, quartic_moments,
};
use super::family::{
    DecisionIntegral, Directional, EventHistoryFamily, EventHistoryFit, EventHistorySpec,
    RankStart, RankStep, UnresolvedGrowth, fit_event_history, fit_event_history_formulas,
};
use super::forecast::{
    ForecastRequest, FutureSegment, HistoryForecastRequest, PopulationForecastRequest, SpellPit,
    forecast, forecast_history, latent_state, pit_uniform_distance, population_forecast,
    predictive_pit,
};
use super::marginal::{SubjectInputs, subject_marginal};
use super::preserve::{ReferenceGrid, ReferenceStrata, killing_masks, stratum_normalisers};
use gam_model_api::families::custom_family::BlockwiseFitOptions;
use gam_problem::ParameterBlockState;
use gam_math::jet_scalar::{OneSeed, TwoSeed};
use gam_math::nested_dual::JetField;
use gam_terms::smooth::{
    LinearCoefficientGeometry, LinearTermSpec, TermCollectionSpec, build_term_collection_design,
};
use ndarray::{Array1, Array2, Axis, array};
use std::sync::Arc;

fn gaussian(x: f64, mean: f64, variance: f64) -> f64 {
    (-(x - mean).powi(2) / (2.0 * variance)).exp() / (2.0 * std::f64::consts::PI * variance).sqrt()
}

/// Why the rank path stopped, checked rather than assumed. Either:
/// - no proposal remained (an empty path, or a last step the evidence accepted);
/// - the evidence refused growth (a converged step it did not accept); or
/// - growth is recorded as unresolved at the order the model is certified at,
///   with no certifiable rung above it.
/// A step that is neither accepted nor converged and carries no verdict is a
/// dropped verdict, and so is a verdict that names any other rung.
fn rank_stop_explanation(
    path: &[RankStep],
    certified_order: usize,
    max_nodes: usize,
    tolerance: f64,
) -> Result<String, String> {
    let Some(last) = path.last() else {
        return Ok("no proposal at rank 0".to_string());
    };
    match &last.growth_unresolved {
        Some(growth) => {
            if last.accepted || last.converged {
                return Err(format!(
                    "step {} carries an unresolved-growth verdict but is accepted={} converged={}",
                    last.rank, last.accepted, last.converged
                ));
            }
            if growth.gauss_hermite_order != certified_order {
                return Err(format!(
                    "the verdict names order {} but the model is certified at order {certified_order}",
                    growth.gauss_hermite_order
                ));
            }
            // The raised rung `2·order − 1` is certifiable iff its own check,
            // `2·(2·order − 1) − 1`, keeps the interpolant's roundoff within the
            // tolerance over the longest subject.
            let raised = 2 * certified_order - 1;
            let checkable = GaussHermite::new(2 * raised - 1).is_ok_and(|rule| {
                rule.lebesgue_constant * f64::EPSILON * max_nodes as f64 <= tolerance
            });
            if checkable {
                return Err(format!(
                    "order {raised} is certifiable over {max_nodes} nodes, so growth did not stop at the top rung"
                ));
            }
            Ok(format!(
                "growth unresolved at Gauss-Hermite order {certified_order}: {} ({})",
                growth.integral.name(),
                growth.reason
            ))
        }
        None if last.accepted => Ok(format!(
            "rank {} accepted and no proposal remained",
            last.rank + 1
        )),
        None if last.converged => Ok(format!(
            "the evidence refused growth beyond rank {}",
            last.rank
        )),
        None => Err(format!(
            "step {} is neither accepted nor converged and carries no verdict",
            last.rank
        )),
    }
}

/// The fit's stop is explained, and the fit-level accessor agrees with its path.
fn assert_rank_stop_explained(fit: &EventHistoryFit, spec: &EventHistorySpec) -> String {
    let stop = rank_stop_explanation(
        &fit.rank_path,
        fit.quadrature.gauss_hermite_order,
        fit.nodes.max_subject_nodes(),
        spec.quadrature_tolerance,
    )
    .expect("the rank path's stop must be explained");
    assert_eq!(
        fit.unresolved_growth().is_some(),
        fit.rank_path.last().is_some_and(|step| step.growth_unresolved.is_some())
    );
    emit(&format!("[rank-stop] rank {}: {stop}", fit.rank()));
    stop
}

/// The mutant control for [`rank_stop_explanation`]: a stop at the top
/// certifiable rung with its verdict is explained, and the same step with the
/// verdict dropped is not. At 449 nodes order 11 is the top certifiable rung
/// (job 1150580: its check at order 21 held, and order 21's check at order 41,
/// Lebesgue constant 1.154e13, did not).
#[test]
fn a_rank_stop_with_its_verdict_dropped_is_unexplained() {
    let tolerance = EventHistorySpec::new(Vec::new()).quadrature_tolerance;
    let step = RankStep {
        rank: 1,
        score_eigenvalue: 2.9636,
        standardised_gain: 0.0,
        proposed_rate: 1.0,
        at_resolution_limit: false,
        rate_held: false,
        ridge_log_lambda: 0.0,
        evidence_gain: 0.0,
        log_likelihood_gain: 0.0,
        accepted: false,
        converged: false,
        growth_unresolved: Some(UnresolvedGrowth {
            gauss_hermite_order: 11,
            integral: DecisionIntegral::DirectionalProfile,
            reason: "the density representation lost positivity".to_string(),
        }),
    };
    assert!(rank_stop_explanation(std::slice::from_ref(&step), 11, 449, tolerance).is_ok());
    let mut dropped = step.clone();
    dropped.growth_unresolved = None;
    assert!(rank_stop_explanation(&[dropped], 11, 449, tolerance).is_err());
    let mut wrong_rung = step;
    if let Some(growth) = wrong_rung.growth_unresolved.as_mut() {
        growth.gauss_hermite_order = 9;
    }
    assert!(rank_stop_explanation(&[wrong_rung], 9, 449, tolerance).is_err());
}

fn subject(times: &[f64], exposures: &[f64], counts: &[Vec<f64>]) -> SubjectNodes {
    let marks = counts[0].len();
    let mut matrix = Array2::<f64>::zeros((times.len(), marks));
    for (n, row) in counts.iter().enumerate() {
        for (d, &c) in row.iter().enumerate() {
            matrix[[n, d]] = c;
        }
    }
    let mut exposure_matrix = Array2::<f64>::zeros((times.len(), marks));
    for (n, &w) in exposures.iter().enumerate() {
        for d in 0..marks {
            exposure_matrix[[n, d]] = w;
        }
    }
    SubjectNodes {
        first_row: 0,
        times: times.to_vec(),
        gaps: times.windows(2).map(|w| w[1] - w[0]).collect(),
        weights: exposures.to_vec(),
        exposures: exposure_matrix,
        counts: matrix,
        covariate_rows: vec![0; times.len()],
    }
}

#[test]
fn forward_operator_is_exact_on_envelope_times_polynomial() {
    let gh = GaussHermite::new(15).expect("rule");
    let (mu, sigma) = (0.1, 0.9);
    let from = Grid::new(&gh, &[mu], &[sigma], &0.0);
    let to = Grid::new(&gh, &[0.4], &[0.5], &0.0);
    let kappa = 0.35;
    let transition = AtomTransition::new(&kappa);
    let phi = (-kappa).exp();
    let q = 1.0 - phi * phi;
    // f(z) = N(z; mu, sigma²) (1 + 0.3 z + 0.2 z²)
    let values: Vec<f64> = (0..from.size())
        .map(|i| {
            let z = *from.coordinate(i, 0);
            gaussian(z, mu, sigma * sigma) * (1.0 + 0.3 * z + 0.2 * z * z)
        })
        .collect();
    let forward = forward_operators(&gh, &from, &to, &[transition.clone()], 0);
    let predicted = forward.plain(&values);
    let tau2 = phi * phi * sigma * sigma + q;
    let s2 = sigma * sigma * q / tau2;
    for j in 0..to.size() {
        let z = *to.coordinate(j, 0);
        let m = mu + phi * sigma * sigma * (z - phi * mu) / tau2;
        let exact = gaussian(z, phi * mu, tau2) * (1.0 + 0.3 * m + 0.2 * (m * m + s2));
        assert!(
            (predicted[j] - exact).abs() < 1e-8 * exact.abs().max(1e-8),
            "node {j}: predicted {} exact {exact}",
            predicted[j]
        );
    }
    // A Gaussian of a different centre and width is not envelope × polynomial;
    // the interpolant is then an approximation, accurate to the interpolation
    // error of a degree-14 polynomial.
    let (mu0, sigma0) = (0.3, 0.7);
    let other: Vec<f64> = (0..from.size())
        .map(|i| gaussian(*from.coordinate(i, 0), mu0, sigma0 * sigma0))
        .collect();
    let predicted = forward.plain(&other);
    for j in 0..to.size() {
        let z = *to.coordinate(j, 0);
        let exact = gaussian(z, phi * mu0, phi * phi * sigma0 * sigma0 + q);
        assert!(
            (predicted[j] - exact).abs() < 1e-4 * exact.abs().max(1e-3),
            "node {j}: predicted {} exact {exact}",
            predicted[j]
        );
    }
    // The backward interpolation reproduces a constant at every inner point
    // and a linear function wherever the inner point lies inside the hull.
    let bases = backward_axis_bases(&gh, &from, &to, &[transition]);
    let ones = vec![1.0; to.size()];
    for i in 0..from.size() {
        let at_inner = interpolate_at_inner_points(gh.order, &bases, &ones, i);
        assert_eq!(at_inner.len(), to.size());
        for v in at_inner {
            assert!((v - 1.0).abs() < 1e-9, "constant at inner point {i}: {v}");
        }
    }
    let linear: Vec<f64> = (0..to.size())
        .map(|j| 0.5 + 0.3 * to.coordinate(j, 0))
        .collect();
    let spread = (2.0 * q).sqrt();
    let hull = gh.nodes[gh.order - 1] * std::f64::consts::SQRT_2 * to.axes[0].sigma;
    for i in 0..from.size() {
        let at_inner = interpolate_at_inner_points(gh.order, &bases, &linear, i);
        for (l, &x) in gh.nodes.iter().enumerate() {
            let zeta = phi * from.coordinate(i, 0) + spread * x;
            if (zeta - to.axes[0].mu).abs() < hull {
                let exact = 0.5 + 0.3 * zeta;
                let got = at_inner[l];
                assert!(
                    (got - exact).abs() < 1e-9,
                    "linear at ({i}, {l}): {got} vs {exact}"
                );
            }
        }
    }
}

#[test]
fn single_node_marginal_matches_numerical_integration() {
    let gh = GaussHermite::new(41).expect("rule");
    let nodes = subject(&[1.0], &[0.8], &[vec![2.0]]);
    let eta0 = [0.3];
    let loadings = [0.9];
    let rates = [1.0];
    let inputs = SubjectInputs {
        nodes: &nodes,
        eta0: &eta0,
        loadings: &loadings,
        rates: &rates,
        time_scale: 1.0,
        gh: &gh,
        continuation_gap: 0.0,
        designs: None,
        log_normaliser: None,
    };
    let out = subject_marginal(&inputs, false).expect("marginal");
    // ∫ exp(y η − w e^η) N(z) dz with η = η0 − ½a² + a z, on a fine grid.
    let mut integral = 0.0;
    let steps = 200_000;
    let (lo, hi) = (-9.0, 9.0);
    let dz = (hi - lo) / steps as f64;
    for i in 0..=steps {
        let z = lo + i as f64 * dz;
        let eta = eta0[0] - 0.5 * loadings[0] * loadings[0] + loadings[0] * z;
        let weight = if i == 0 || i == steps { 0.5 } else { 1.0 };
        integral += weight * dz * (2.0 * eta - 0.8 * eta.exp()).exp() * gaussian(z, 0.0, 1.0);
    }
    assert!(
        (out.loglik - integral.ln()).abs() < 1e-8,
        "filter {} vs quadrature {}",
        out.loglik,
        integral.ln()
    );
}

#[test]
fn two_node_marginal_matches_brute_force_double_integral() {
    let gh = GaussHermite::new(25).expect("rule");
    let nodes = subject(&[0.0, 0.7], &[0.5, 0.6], &[vec![1.0], vec![0.0]]);
    let eta0 = [-0.2, 0.4];
    let loadings = [1.1];
    let rates = [1.2];
    let inputs = SubjectInputs {
        nodes: &nodes,
        eta0: &eta0,
        loadings: &loadings,
        rates: &rates,
        time_scale: 1.0,
        gh: &gh,
        continuation_gap: 0.0,
        designs: None,
        log_normaliser: None,
    };
    let out = subject_marginal(&inputs, false).expect("marginal");
    let phi = (-(1.2_f64 * 0.7)).exp();
    let q = 1.0 - phi * phi;
    let steps = 1200;
    let (lo, hi) = (-7.0, 7.0);
    let dz = (hi - lo) / steps as f64;
    let mut integral = 0.0;
    for i in 0..=steps {
        let z1 = lo + i as f64 * dz;
        let w1 = if i == 0 || i == steps { 0.5 } else { 1.0 };
        let shift = -0.5 * loadings[0] * loadings[0];
        let eta1 = eta0[0] + shift + loadings[0] * z1;
        let l1 = (eta1 - 0.5 * eta1.exp()).exp() * gaussian(z1, 0.0, 1.0);
        let mut inner = 0.0;
        for j in 0..=steps {
            let z2 = lo + j as f64 * dz;
            let w2 = if j == 0 || j == steps { 0.5 } else { 1.0 };
            let eta2 = eta0[1] + shift + loadings[0] * z2;
            inner += w2 * dz * (-0.6 * eta2.exp()).exp() * gaussian(z2, phi * z1, q);
        }
        integral += w1 * dz * l1 * inner;
    }
    assert!(
        (out.loglik - integral.ln()).abs() < 1e-6,
        "filter {} vs brute force {}",
        out.loglik,
        integral.ln()
    );
}

#[test]
fn zero_loadings_reduce_to_the_poisson_likelihood() {
    let gh = GaussHermite::new(21).expect("rule");
    let nodes = subject(
        &[0.0, 0.5, 1.5],
        &[0.4, 0.0, 0.7],
        &[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 2.0]],
    );
    let eta0 = [0.1, -0.3, 0.5, 0.2, -0.1, 0.4];
    let loadings = [0.0, 0.0, 0.0, 0.0];
    let rates = [1.0, 1.35];
    let inputs = SubjectInputs {
        nodes: &nodes,
        eta0: &eta0,
        loadings: &loadings,
        rates: &rates,
        time_scale: 1.0,
        gh: &gh,
        continuation_gap: 0.0,
        designs: None,
        log_normaliser: None,
    };
    let out = subject_marginal(&inputs, true).expect("marginal");
    let mut expected = 0.0;
    for n in 0..3 {
        for d in 0..2 {
            let eta = eta0[n * 2 + d];
            expected += nodes.counts[[n, d]] * eta - nodes.exposures[[n, d]] * eta.exp();
        }
    }
    assert!((out.loglik - expected).abs() < 1e-12);
    for n in 0..3 {
        for d in 0..2 {
            let eta = eta0[n * 2 + d];
            let score = nodes.counts[[n, d]] - nodes.exposures[[n, d]] * eta.exp();
            // The smoothed marginal drops tail mass below the density noise
            // floor (1e-11 of the peak), which is the agreement limit here.
            assert!(
                (out.gradient[n * 2 + d] - score).abs() < 1e-9,
                "gradient {} vs score {score}",
                out.gradient[n * 2 + d]
            );
        }
    }
    // With zero loadings the rates are unidentified: their gradient is zero
    // up to the hull clamping of the backward quadrature.
    for k in 0..2 {
        assert!(
            out.gradient[6 + 4 + k].abs() < 1e-8,
            "rate gradient {}",
            out.gradient[6 + 4 + k]
        );
    }
}

fn finite_difference_subject() -> (SubjectNodes, Vec<f64>, Vec<f64>, Vec<f64>) {
    let nodes = subject(
        &[0.0, 0.4, 1.1, 1.6],
        &[0.3, 0.0, 0.5, 0.2],
        &[
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ],
    );
    let eta0 = vec![0.2, -0.1, 0.4, 0.0, -0.3, 0.5, 0.1, 0.2];
    let loadings = vec![0.8, -0.4, 0.3, 0.6];
    let rates = vec![0.82, 1.65];
    (nodes, eta0, loadings, rates)
}

fn evaluate_at(
    nodes: &SubjectNodes,
    gh: &GaussHermite,
    theta: &[f64],
    derivatives: bool,
) -> super::marginal::SubjectOutput<f64> {
    let n = nodes.len();
    let (marks, atoms) = (2, 2);
    let eta0 = &theta[0..n * marks];
    let loadings = &theta[n * marks..n * marks + marks * atoms];
    let rates = &theta[n * marks + marks * atoms..];
    subject_marginal(
        &SubjectInputs {
            nodes,
            eta0,
            loadings,
            rates,
            time_scale: 1.0,
            gh,
            continuation_gap: 0.0,
            designs: None,
            log_normaliser: None,
        },
        derivatives,
    )
    .expect("marginal")
}

#[test]
fn gradient_and_hessian_match_central_differences() {
    let gh = GaussHermite::new(31).expect("rule");
    let (nodes, eta0, loadings, rates) = finite_difference_subject();
    let mut theta = eta0.clone();
    theta.extend(loadings.iter());
    theta.extend(rates.iter());
    let p = theta.len();
    let base = evaluate_at(&nodes, &gh, &theta, true);
    let h = 1e-4;
    for i in 0..p {
        let mut plus = theta.clone();
        plus[i] += h;
        let mut minus = theta.clone();
        minus[i] -= h;
        let fp = evaluate_at(&nodes, &gh, &plus, true);
        let fm = evaluate_at(&nodes, &gh, &minus, true);
        let fd = (fp.loglik - fm.loglik) / (2.0 * h);
        assert!(
            (base.gradient[i] - fd).abs() < 1e-6 * (1.0 + fd.abs()),
            "gradient[{i}] = {} vs finite difference {fd}",
            base.gradient[i]
        );
        for j in 0..p {
            let fd_h = (fp.gradient[j] - fm.gradient[j]) / (2.0 * h);
            // Louis' Hessian interpolates the smoother residual with a cubic
            // spline; it agrees with the finite difference of the gradient to
            // that interpolation error.
            assert!(
                (base.hessian[i * p + j] - fd_h).abs() < 1e-3 * (1.0 + fd_h.abs()),
                "hessian[{i},{j}] = {} vs finite difference {fd_h}",
                base.hessian[i * p + j]
            );
            assert!((base.hessian[i * p + j] - base.hessian[j * p + i]).abs() < 1e-9);
        }
    }
}

#[test]
fn directional_duals_match_finite_differences_of_the_hessian() {
    let gh = GaussHermite::new(21).expect("rule");
    let (nodes, eta0, loadings, rates) = finite_difference_subject();
    let mut theta = eta0.clone();
    theta.extend(loadings.iter());
    theta.extend(rates.iter());
    let p = theta.len();
    let u: Vec<f64> = (0..p)
        .map(|i| 0.3 * ((i as f64) * 0.7).sin() + 0.1)
        .collect();
    let v: Vec<f64> = (0..p)
        .map(|i| 0.2 * ((i as f64) * 1.3).cos() - 0.05)
        .collect();
    let seed = |theta: &[f64], u: &[f64], v: &[f64]| -> Vec<TwoSeed<0>> {
        theta
            .iter()
            .zip(u.iter().zip(v.iter()))
            .map(|(&x, (&du, &dv))| TwoSeed::<0>::seeded(x, du, dv))
            .collect()
    };
    let n = nodes.len();
    let (marks, atoms) = (2, 2);
    let evaluate_two = |theta: &[f64]| {
        let jets = seed(theta, &u, &v);
        subject_marginal(
            &SubjectInputs {
                nodes: &nodes,
                eta0: &jets[0..n * marks],
                loadings: &jets[n * marks..n * marks + marks * atoms],
                rates: &jets[n * marks + marks * atoms..],
                time_scale: 1.0,
                gh: &gh,
                continuation_gap: 0.0,
                designs: None,
                log_normaliser: None,
            },
            true,
        )
        .expect("marginal")
    };
    let two = evaluate_two(&theta);
    let h = 1e-4;
    let shifted = |s: f64, dir: &[f64]| -> Vec<f64> {
        theta
            .iter()
            .zip(dir.iter())
            .map(|(x, d)| x + s * d)
            .collect()
    };
    let plus_u = evaluate_at(&nodes, &gh, &shifted(h, &u), true);
    let minus_u = evaluate_at(&nodes, &gh, &shifted(-h, &u), true);
    for idx in 0..p * p {
        let fd = (plus_u.hessian[idx] - minus_u.hessian[idx]) / (2.0 * h);
        let dual = two.hessian[idx].eps.value();
        assert!(
            (dual - fd).abs() < 2e-5 * (1.0 + fd.abs()),
            "D H[u] at {idx}: dual {dual} vs finite difference {fd}"
        );
    }
    // Mixed second derivative: difference of the u-directional derivative along v.
    let one_at = |theta: &[f64]| -> Vec<OneSeed<0>> {
        let jets: Vec<OneSeed<0>> = theta
            .iter()
            .zip(u.iter())
            .map(|(&x, &du)| OneSeed::<0>::seeded(x, du, 0.0))
            .collect();
        let out = subject_marginal(
            &SubjectInputs {
                nodes: &nodes,
                eta0: &jets[0..n * marks],
                loadings: &jets[n * marks..n * marks + marks * atoms],
                rates: &jets[n * marks + marks * atoms..],
                time_scale: 1.0,
                gh: &gh,
                continuation_gap: 0.0,
                designs: None,
                log_normaliser: None,
            },
            true,
        )
        .expect("marginal");
        out.hessian
    };
    let plus_v = one_at(&shifted(h, &v));
    let minus_v = one_at(&shifted(-h, &v));
    for idx in 0..p * p {
        let fd = (plus_v[idx].eps.value() - minus_v[idx].eps.value()) / (2.0 * h);
        let dual = two.hessian[idx].eps_del.value();
        assert!(
            (dual - fd).abs() < 5e-5 * (1.0 + fd.abs()),
            "D²H[u,v] at {idx}: dual {dual} vs finite difference {fd}"
        );
    }
}

#[test]
fn node_expansion_integrates_exposure_and_places_events() {
    let mut cohort = EventHistoryCohort {
        mark_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        mark_kinds: vec![MarkKind::Recurrent, MarkKind::Once, MarkKind::Terminal],
        covariate_names: vec!["x".to_string()],
        covariate_levels: vec![Vec::new()],
        covariates: array![[1.0], [2.0], [3.0]],
        subjects: vec![SubjectHistory {
            id: "s".to_string(),
            entry: 1.0,
            exit: 4.0,
            events: vec![
                Event { time: 2.5, mark: 1 },
                Event { time: 2.5, mark: 0 },
                Event { time: 4.0, mark: 2 },
            ],
            segments: vec![
                CovariateSegment { start: 0.0, row: 0 },
                // A covariate change at the instant of the event: the event
                // node sees the left limit, the quadrature after it the new row.
                CovariateSegment { start: 2.5, row: 2 },
                CovariateSegment { start: 3.0, row: 1 },
            ],
        }],
    };
    cohort.validate().expect("valid");
    let nodes = expand_nodes(&cohort, 5, 0).expect("nodes");
    let s = &nodes.subjects[0];
    let total_weight: f64 = s.weights.iter().sum();
    assert!((total_weight - 3.0).abs() < 1e-12, "weight {total_weight}");
    // Recurrent and terminal marks are at risk throughout; the once-only
    // mark leaves the risk set after 2.5.
    let exposure = |d: usize| -> f64 { s.exposures.column(d).sum() };
    assert!((exposure(0) - 3.0).abs() < 1e-12);
    assert!((exposure(2) - 3.0).abs() < 1e-12);
    assert!(
        (exposure(1) - 1.5).abs() < 1e-12,
        "once-only exposure {}",
        exposure(1)
    );
    let event_node = s.times.iter().position(|&t| t == 2.5).expect("event node");
    assert_eq!(s.counts[[event_node, 0]], 1.0);
    assert_eq!(s.counts[[event_node, 1]], 1.0);
    assert_eq!(s.weights[event_node], 0.0);
    assert_eq!(
        s.covariate_rows[event_node], 0,
        "an event node takes the left limit"
    );
    let terminal_node = s.len() - 1;
    assert_eq!(s.times[terminal_node], 4.0);
    assert_eq!(s.counts[[terminal_node, 2]], 1.0);
    assert!(s.gaps.iter().all(|&g| g > 0.0));
    for (n, &t) in s.times.iter().enumerate() {
        if n == event_node {
            continue;
        }
        let expected_row = if t >= 3.0 {
            1
        } else if t > 2.5 {
            2
        } else {
            0
        };
        assert_eq!(s.covariate_rows[n], expected_row, "node at {t}");
        assert_eq!(
            nodes.node_data[[n, 0]],
            cohort.covariates[[expected_row, 0]]
        );
        assert_eq!(nodes.node_data[[n, 1]], t);
    }
    // Every mesh cell is halved at refinement one: twice the quadrature
    // nodes, the same total weight, the same events.
    let refined = expand_nodes(&cohort, 5, 1).expect("refined nodes");
    let r = &refined.subjects[0];
    let quadrature_nodes = |s: &SubjectNodes| s.weights.iter().filter(|w| **w > 0.0).count();
    assert_eq!(quadrature_nodes(r), 2 * quadrature_nodes(s));
    assert!((r.weights.iter().sum::<f64>() - 3.0).abs() < 1e-12);
    assert_eq!(r.counts.sum(), s.counts.sum());
    // The design rows carry entry, exit, covariate changes and an
    // event-free quadrature. They are a function of the design alone: the
    // same subject with no events at all produces the identical rows, so no
    // data-adaptive basis can be shaped by where the events fell.
    let rows = design_rows(&cohort, 5).expect("design rows");
    let times: Vec<f64> = rows.column(1).to_vec();
    assert!(times.contains(&1.0) && times.contains(&4.0) && times.contains(&3.0));
    let interior = times
        .iter()
        .filter(|t| **t > 1.0 && **t < 4.0 && **t != 3.0 && **t != 2.5)
        .count();
    assert_eq!(interior, 15, "three event-free cells of five nodes");
    let mut event_free = cohort.clone();
    event_free.subjects[0].events.clear();
    event_free.subjects[0].exit = 4.0;
    event_free.validate().expect("valid without events");
    let without = design_rows(&event_free, 5).expect("design rows");
    assert_eq!(rows, without, "an event time must not enter a basis");
}

#[test]
fn validation_rejects_ill_formed_cohorts() {
    let base = || EventHistoryCohort {
        mark_names: vec!["relapse".to_string(), "death".to_string()],
        mark_kinds: vec![MarkKind::Recurrent, MarkKind::Terminal],
        covariate_names: vec!["x".to_string(), "arm".to_string()],
        covariate_levels: vec![
            Vec::new(),
            vec!["control".to_string(), "treated".to_string()],
        ],
        covariates: array![[0.3, 0.0], [-0.1, 1.0]],
        subjects: vec![SubjectHistory {
            id: "a".to_string(),
            entry: 0.0,
            exit: 5.0,
            events: vec![Event { time: 2.0, mark: 0 }],
            segments: vec![CovariateSegment { start: 0.0, row: 0 }],
        }],
    };
    let mut ok = base();
    ok.validate().expect("the base cohort is valid");
    let expect_error = |mutate: &dyn Fn(&mut EventHistoryCohort), needle: &str| {
        // Each mutation starts from the same valid cohort.
        let mut cohort = base();
        mutate(&mut cohort);
        let error = cohort
            .validate()
            .err()
            .unwrap_or_else(|| panic!("expected an error containing {needle:?}"));
        assert!(
            error.to_string().contains(needle),
            "{error} lacks {needle:?}"
        );
    };
    expect_error(
        &|c: &mut EventHistoryCohort| {
            let first = c.subjects[0].clone();
            c.subjects.push(first);
        },
        "duplicate subject identifier",
    );
    expect_error(
        &|c: &mut EventHistoryCohort| {
            c.subjects[0]
                .segments
                .push(CovariateSegment { start: 0.0, row: 1 })
        },
        "two covariate segments starting at",
    );
    expect_error(
        &|c: &mut EventHistoryCohort| {
            c.subjects[0]
                .segments
                .push(CovariateSegment { start: 7.0, row: 1 })
        },
        "outside (entry, exit)",
    );
    expect_error(
        &|c: &mut EventHistoryCohort| c.subjects[0].events.push(Event { time: 3.0, mark: 1 }),
        "must end follow-up",
    );
    // One mark firing twice is caught as that mark's own rule; two DIFFERENT
    // terminal marks each firing once is the case the follow-up rule catches.
    expect_error(
        &|c: &mut EventHistoryCohort| {
            c.subjects[0].events.push(Event { time: 3.0, mark: 1 });
            c.subjects[0].events.push(Event { time: 5.0, mark: 1 });
        },
        "can fire at most once",
    );
    expect_error(
        &|c: &mut EventHistoryCohort| {
            c.mark_kinds[0] = MarkKind::Terminal;
            c.subjects[0].events.push(Event { time: 3.0, mark: 1 });
        },
        "terminal events",
    );
    expect_error(
        &|c: &mut EventHistoryCohort| {
            c.mark_kinds[0] = MarkKind::Once;
            c.subjects[0].events.push(Event { time: 3.0, mark: 0 });
        },
        "can fire at most once",
    );
    expect_error(
        &|c: &mut EventHistoryCohort| c.covariates[[0, 1]] = 2.0,
        "categorical covariate",
    );
    expect_error(
        &|c: &mut EventHistoryCohort| c.covariates[[0, 1]] = 0.5,
        "categorical covariate",
    );
    expect_error(
        &|c: &mut EventHistoryCohort| c.mark_names[1] = "relapse".to_string(),
        "duplicate mark name",
    );
    expect_error(
        &|c: &mut EventHistoryCohort| c.mark_kinds.truncate(1),
        "mark kinds",
    );
    // A terminal event at exit is valid.
    let mut terminal = base();
    terminal.subjects[0]
        .events
        .push(Event { time: 5.0, mark: 1 });
    terminal
        .validate()
        .expect("a terminal event at exit ends follow-up");
    assert!(
        terminal.subjects[0]
            .terminal_event(&terminal.mark_kinds)
            .is_some()
    );
}

struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Simulate a marked cohort whose log-intensity of mark `d` is
/// `intercept_d + slope · x + Σ_k loading_{dk} · z_k(t)` with `z_k`
/// independent unit-variance Ornstein–Uhlenbeck atoms of the given rates,
/// sampled exactly at the steps of a fine grid and held constant between
/// them. Conditional on the path, each step's events are Poisson with the
/// step's exact integrated intensity, placed uniformly in the step: no
/// thinning bound is needed, so nothing is approximate beyond the
/// piecewise-constant path itself, whose error vanishes with the step. A
/// terminal mark ends follow-up at its event; a once-only mark leaves the
/// risk set after its event. The simulated paths are returned beside the
/// cohort, one `steps × atoms` matrix per subject, so a fit's latent state
/// can be judged against what generated the events.
fn simulate_latent_cohort(
    subjects: usize,
    follow_up: f64,
    intercepts: &[f64],
    slope: f64,
    loadings: &Array2<f64>,
    rates: &[f64],
    kinds: &[MarkKind],
    seed: u64,
) -> (EventHistoryCohort, Vec<Array2<f64>>) {
    let marks = intercepts.len();
    let atoms = rates.len();
    assert_eq!(loadings.dim(), (marks, atoms));
    let mut rng = Rng(seed);
    let steps = 400;
    let dt = follow_up / steps as f64;
    let phis: Vec<f64> = rates.iter().map(|rate| (-rate * dt).exp()).collect();
    let innovations: Vec<f64> = phis.iter().map(|phi| (1.0 - phi * phi).sqrt()).collect();
    let mut covariates = Array2::<f64>::zeros((subjects, 1));
    let mut histories = Vec::with_capacity(subjects);
    let mut paths = Vec::with_capacity(subjects);
    for s in 0..subjects {
        let x = rng.normal();
        covariates[[s, 0]] = x;
        let mut z: Vec<f64> = (0..atoms).map(|_| rng.normal()).collect();
        let mut path = Array2::<f64>::zeros((steps, atoms));
        let mut events: Vec<Event> = Vec::new();
        let mut exit = follow_up;
        let mut at_risk = vec![true; marks];
        'steps: for step in 0..steps {
            let left = step as f64 * dt;
            for k in 0..atoms {
                path[[step, k]] = z[k];
            }
            let mut step_events: Vec<Event> = Vec::new();
            for d in 0..marks {
                if !at_risk[d] {
                    continue;
                }
                let latent: f64 = (0..atoms).map(|k| loadings[[d, k]] * z[k]).sum();
                let mean = (intercepts[d] + slope * x + latent).exp() * dt;
                // Poisson(mean) by inversion of its cumulative sum.
                let threshold = rng.uniform();
                let mut count = 0usize;
                let mut term = (-mean).exp();
                let mut cumulative = term;
                while threshold > cumulative && count < 1000 {
                    count += 1;
                    term *= mean / count as f64;
                    cumulative += term;
                }
                for _ in 0..count {
                    step_events.push(Event {
                        time: left + dt * rng.uniform(),
                        mark: d,
                    });
                }
            }
            step_events.sort_by(|a, b| a.time.total_cmp(&b.time));
            for event in step_events {
                if !at_risk[event.mark] {
                    continue;
                }
                match kinds[event.mark] {
                    MarkKind::Recurrent => events.push(event),
                    MarkKind::Once => {
                        at_risk[event.mark] = false;
                        events.push(event);
                    }
                    MarkKind::Terminal => {
                        exit = event.time;
                        events.push(event);
                        break 'steps;
                    }
                }
            }
            for k in 0..atoms {
                z[k] = phis[k] * z[k] + innovations[k] * rng.normal();
            }
        }
        // Distinct event times: a tie at the step resolution is resolved by
        // the sort, and an event at exactly zero is impossible.
        events.retain(|e| e.time > 0.0 && e.time <= exit);
        histories.push(SubjectHistory {
            id: format!("s{s}"),
            entry: 0.0,
            exit,
            events,
            segments: vec![CovariateSegment { start: 0.0, row: s }],
        });
        paths.push(path);
    }
    let cohort = EventHistoryCohort {
        mark_names: (0..marks).map(|d| format!("mark{d}")).collect(),
        mark_kinds: kinds.to_vec(),
        covariate_names: vec!["x".to_string()],
        covariate_levels: vec![Vec::new()],
        covariates,
        subjects: histories,
    };
    (cohort, paths)
}

/// The single-atom case of [`simulate_latent_cohort`].
fn simulate_marked_cohort(
    subjects: usize,
    follow_up: f64,
    intercepts: &[f64],
    slope: f64,
    loadings: &[f64],
    rate: f64,
    kinds: &[MarkKind],
    seed: u64,
) -> EventHistoryCohort {
    let column = Array2::from_shape_vec((loadings.len(), 1), loadings.to_vec()).expect("column");
    simulate_latent_cohort(
        subjects,
        follow_up,
        intercepts,
        slope,
        &column,
        &[rate],
        kinds,
        seed,
    )
    .0
}

/// The single recurrent mark case of [`simulate_marked_cohort`].
fn simulate_cohort(
    subjects: usize,
    follow_up: f64,
    intercept: f64,
    slope: f64,
    loading: f64,
    rate: f64,
    seed: u64,
) -> EventHistoryCohort {
    let mut cohort = simulate_marked_cohort(
        subjects,
        follow_up,
        &[intercept],
        slope,
        &[loading],
        rate,
        &[MarkKind::Recurrent],
        seed,
    );
    cohort.mark_names = vec!["event".to_string()];
    cohort
}

fn linear_spec() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "x".to_string(),
            feature_col: 0,
            feature_cols: vec![0],
            categorical_levels: Vec::new(),
            double_penalty: false,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: Vec::new(),
        smooth_terms: Vec::new(),
    }
}

#[test]
fn family_joint_hessian_matches_finite_differences_of_its_gradient() {
    let mut cohort = simulate_cohort(6, 3.0, -0.5, 0.4, 0.8, 0.5, 11);
    cohort.validate().expect("valid");
    let nodes = Arc::new(expand_nodes(&cohort, 3, 0).expect("nodes"));
    let total = nodes.total_nodes;
    let mut design = Array2::<f64>::zeros((total, 2));
    for row in 0..total {
        design[[row, 0]] = 1.0;
        design[[row, 1]] = nodes.node_data[[row, 0]];
    }
    let family = EventHistoryFamily::new(
        Arc::clone(&nodes),
        vec![Arc::new(design.clone())],
        1,
        31,
        cohort.time_scale(),
        vec![None],
    )
    .expect("family");
    let beta = array![-0.4, 0.3];
    let latent = array![0.5, 0.8];
    let states = |beta: &Array1<f64>, latent: &Array1<f64>| {
        vec![
            ParameterBlockState {
                beta: beta.clone(),
                eta: design.dot(beta),
            },
            ParameterBlockState {
                beta: latent.clone(),
                eta: Array1::zeros(total),
            },
        ]
    };
    let base = family
        .joint_evaluation(&states(&beta, &latent))
        .expect("joint");
    let p = beta.len() + latent.len();
    let h = 1e-4;
    for i in 0..p {
        let mut plus_beta = beta.clone();
        let mut plus_latent = latent.clone();
        let mut minus_beta = beta.clone();
        let mut minus_latent = latent.clone();
        if i < beta.len() {
            plus_beta[i] += h;
            minus_beta[i] -= h;
        } else {
            plus_latent[i - beta.len()] += h;
            minus_latent[i - beta.len()] -= h;
        }
        let plus = family
            .joint_evaluation(&states(&plus_beta, &plus_latent))
            .expect("joint");
        let minus = family
            .joint_evaluation(&states(&minus_beta, &minus_latent))
            .expect("joint");
        let fd = (plus.log_likelihood - minus.log_likelihood) / (2.0 * h);
        assert!(
            (base.gradient[i] - fd).abs() < 1e-6 * (1.0 + fd.abs()),
            "gradient[{i}] {} vs {fd}",
            base.gradient[i]
        );
        for j in 0..p {
            let fd_h = -(plus.gradient[j] - minus.gradient[j]) / (2.0 * h);
            // Spline-interpolated smoother residual: agreement to its
            // interpolation error, not to roundoff.
            assert!(
                (base.hessian[[i, j]] - fd_h).abs() < 1e-3 * (1.0 + fd_h.abs()),
                "hessian[{i},{j}] {} vs {fd_h}",
                base.hessian[[i, j]]
            );
        }
    }
    // Directional derivative of the negative Hessian against finite differences.
    let u = array![0.2, -0.1, 0.3, 0.15];
    let du = family
        .directional_hessian(&states(&beta, &latent), &u)
        .expect("directional");
    let plus = family
        .joint_evaluation(&states(
            &(beta.clone() + &u.slice(ndarray::s![0..2]).to_owned().mapv(|x| x * h)),
            &(latent.clone() + &u.slice(ndarray::s![2..4]).to_owned().mapv(|x| x * h)),
        ))
        .expect("joint");
    let minus = family
        .joint_evaluation(&states(
            &(beta.clone() - &u.slice(ndarray::s![0..2]).to_owned().mapv(|x| x * h)),
            &(latent.clone() - &u.slice(ndarray::s![2..4]).to_owned().mapv(|x| x * h)),
        ))
        .expect("joint");
    for i in 0..p {
        for j in 0..p {
            let fd = (plus.hessian[[i, j]] - minus.hessian[[i, j]]) / (2.0 * h);
            assert!(
                (du[[i, j]] - fd).abs() < 2e-5 * (1.0 + fd.abs()),
                "D H[u][{i},{j}] {} vs {fd}",
                du[[i, j]]
            );
        }
    }
}

#[test]
fn fit_recovers_the_covariate_effect_and_a_positive_shared_risk_loading() {
    install_test_logger();
    let started = std::time::Instant::now();
    let mut cohort = simulate_cohort(80, 6.0, -0.8, 0.5, 1.0, 0.4, 7);
    let mut spec = EventHistorySpec::new(vec![linear_spec()]);
    spec.gauss_hermite_order = 11;
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
    assert_rank_stop_explained(&fit, &spec);
    let beta = fit.mark_coefficients(0);
    println!(
        "[fit] {:.1}s outer_iterations={} gh_order={} beta={:?} loading={} rate={} log_lambda={:?}",
        started.elapsed().as_secs_f64(),
        fit.fit.outer_iterations,
        fit.quadrature.gauss_hermite_order,
        beta.to_vec(),
        fit.loadings[[0, 0]],
        fit.rates[0],
        fit.atom_log_lambdas
    );
    assert_eq!(beta.len(), 2, "intercept and slope");
    assert!(
        (beta[1] - 0.5).abs() < 0.25,
        "slope {} should recover 0.5 within its sampling error",
        beta[1]
    );
    // The simulated intensity `exp(β₀ + β₁x + a z)` averages over the
    // stationary `z` to `exp(β₀ + ½a² + β₁x)`, and the fitted `η⁰` is
    // parameterised as that average, so the intercept estimates the
    // population log-rate `−0.8 + ½·1² = −0.3`, not the conditional `−0.8`.
    let population_log_rate = -0.8 + 0.5;
    assert!(
        (beta[0] - population_log_rate).abs() < 0.25,
        "intercept {} should be the population log-rate {population_log_rate}",
        beta[0]
    );
    emit(&format!(
        "[fit] rank={} evidence={:?} path={:?}",
        fit.rank(),
        fit.atom_evidence,
        fit.rank_path
    ));
    assert!(
        fit.rank() >= 1,
        "a shared dynamic risk was simulated but the evidence grew no atom"
    );
    assert!(
        fit.atom_evidence[0] > 0.0,
        "an accepted atom carries positive evidence"
    );
    // The reported evidence is read from the exact profile along the
    // direction, so it cannot exceed the realised log-likelihood gain of the
    // fitted candidate (which re-optimises everything the profile held), and
    // it is closer to that gain than the score's second-order statistic.
    let step = &fit.rank_path[0];
    assert!(
        step.evidence_gain <= step.log_likelihood_gain + 1e-9,
        "evidence {} exceeds the realised gain {}",
        step.evidence_gain,
        step.log_likelihood_gain
    );
    assert!(
        (step.evidence_gain - step.log_likelihood_gain).abs()
            < (step.standardised_gain - step.log_likelihood_gain).abs(),
        "evidence {} is further from the realised gain {} than the standardised gain {}",
        step.evidence_gain,
        step.log_likelihood_gain,
        step.standardised_gain
    );
    // The canonical gauge signs the loading positive; the reported covariance
    // is the posterior mean, the mode squared plus the loading's posterior
    // spread, and its eigenvalue carries a finite uncertainty.
    let loading = fit.loadings[[0, 0]];
    assert!(
        loading > 0.4,
        "a shared dynamic risk was simulated but the fitted loading is {loading}"
    );
    assert!(fit.covariance[[0, 0]] >= loading * loading);
    assert!((fit.eigenvalues[0] - fit.covariance[[0, 0]]).abs() < 1e-12);
    assert!(fit.eigenvalue_sd[0].is_finite() && fit.eigenvalue_sd[0] > 0.0);
    assert!((fit.effective_rank - 1.0).abs() < 1e-12);
    assert!(fit.atom_log_lambdas[0].is_finite());
    emit(&format!(
        "[fit] covariance={} eigenvalue_sd={} prior log-precision={}",
        fit.covariance[[0, 0]],
        fit.eigenvalue_sd[0],
        fit.atom_log_lambdas[0]
    ));
    // A fit object only exists from a converged optimisation; its certificate
    // gradient, when reported, is finite.
    assert!(fit.fit.outer_gradient_norm.is_none_or(|g| g.is_finite()));
    assert!(fit.rates[0] > 0.0 && fit.rates[0].is_finite());
    // The certificate: the fitted coefficients are stationary under a
    // doubling of the Gauss-Hermite order and a halving of the mesh, to the
    // stated fraction of their posterior standard deviation.
    // The certificate: at the fitted smoothing parameters, doubling the
    // Gauss-Hermite order and halving the mesh each move the coefficients by
    // less than the tolerance, in posterior standard deviations.
    assert!(fit.quadrature.gauss_hermite.coefficient_shift <= spec.quadrature_tolerance);
    assert!(fit.quadrature.mesh.coefficient_shift <= spec.quadrature_tolerance);
    assert_eq!(
        fit.quadrature.gauss_hermite.candidate,
        2 * fit.quadrature.gauss_hermite_order - 1
    );
    assert_eq!(
        fit.quadrature.mesh.candidate,
        fit.quadrature.mesh_refinement + 1
    );
    // Forecast: probabilities and expected counts are coherent.
    let request = ForecastRequest {
        history: &cohort.subjects[0],
        horizons: &[6.5, 7.0, 8.0],
        future: &[],
        stratum: 0,
    };
    let f = forecast(&fit, &cohort, &request).expect("forecast");
    assert!(
        f.survival.iter().all(|&s| (0.0..=1.0).contains(&s)),
        "survival left [0, 1]: {:?}",
        f.survival
    );
    assert!(
        (f.survival[0] - 1.0).abs() < 1e-12,
        "no terminal marks: survival stays one"
    );
    assert!(f.expected_counts[[0, 0]] <= f.expected_counts[[1, 0]]);
    assert!(f.expected_counts[[1, 0]] <= f.expected_counts[[2, 0]]);
    assert!(f.expected_counts[[2, 0]] > 0.0);
    // Predictive PIT: uniform up to sampling error on the training cohort.
    let mut spells = Vec::new();
    for subject in &cohort.subjects {
        let pits = predictive_pit(&fit, &cohort, subject, 0).expect("pit");
        // A recurrent mark never ends follow-up, so every subject's last
        // spell is the censored tail after its last event.
        let tail = pits.last().expect("a subject has at least its tail spell");
        assert!(!tail.observed, "the tail spell is censored at the exit");
        assert_eq!(tail.time, subject.exit);
        assert!(tail.marks.is_empty());
        assert_eq!(
            pits.iter().filter(|p| p.observed).count(),
            subject.events.len()
        );
        for spell in pits.iter().filter(|p| p.observed) {
            assert_eq!(spell.marks, vec![0]);
            assert!((spell.mark_probabilities[0] - 1.0).abs() < 1e-12);
        }
        spells.extend(pits);
    }
    assert!(spells.iter().all(|s| (0.0..=1.0).contains(&s.pit)));
    // Under the model the PITs are independent uniforms (the Rosenblatt
    // transform of the event times) with the tails censored; the fitted
    // parameters make this a sanity band around the Kolmogorov 95%
    // quantile, not a formal test.
    let distance = pit_uniform_distance(&spells).expect("spells");
    let n = spells.iter().filter(|s| s.observed).count() as f64;
    assert!(
        distance < 1.63 / n.sqrt() + 0.05,
        "PIT distance {distance} over {n} events exceeds the uniform band"
    );
}

/// Write one line to stdout from test support code that is not itself a
/// `#[test]` function.
fn emit(line: &str) {
    use std::io::Write;
    let mut stdout = std::io::stdout().lock();
    if stdout.write_all(line.as_bytes()).is_ok() && stdout.write_all(b"\n").is_ok() {
        return;
    }
}

/// A stdout logger for tests that watch the outer solve, installed once.
struct StdoutLogger;

impl log::Log for StdoutLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::max_level()
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            emit(&record.args().to_string());
        }
    }
    fn flush(&self) {}
}

static STDOUT_LOGGER: StdoutLogger = StdoutLogger;

fn install_test_logger() {
    if log::set_logger(&STDOUT_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
}

#[test]
fn formula_right_hand_side_resolves_against_the_node_columns() {
    let mut cohort = simulate_cohort(4, 3.0, -0.5, 0.4, 0.0, 0.5, 5);
    cohort.validate().expect("valid");
    let rows = design_rows(&cohort, 3).expect("rows");
    let spec = super::formula::covariate_spec_from_formula("x + s(time)", rows.view(), &cohort)
        .expect("spec");
    assert_eq!(spec.linear_terms.len(), 1);
    assert_eq!(spec.linear_terms[0].feature_col, 0);
    assert_eq!(spec.smooth_terms.len(), 1);
    let error = super::formula::covariate_spec_from_formula("nope", rows.view(), &cohort)
        .err()
        .expect("unknown column must fail");
    assert!(error.to_string().contains("nope"), "{error}");
}

#[test]
fn a_cohort_without_shared_risk_grows_no_atom() {
    install_test_logger();
    let mut cohort = simulate_cohort(80, 6.0, -0.8, 0.5, 0.0, 0.4, 3);
    let mut spec = EventHistorySpec::new(vec![linear_spec()]);
    spec.gauss_hermite_order = 11;
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
    assert!(fit.fit.outer_gradient_norm.is_none_or(|g| g.is_finite()));
    // Nothing was shared, so the evidence keeps the loading at zero: the
    // refusal is the prior's decision, made from the score without a fit.
    emit(&format!(
        "[null] rank={} path={:?}",
        fit.rank(),
        fit.rank_path
    ));
    assert_eq!(
        fit.rank(),
        0,
        "no shared risk was simulated but the evidence grew {} atoms",
        fit.rank()
    );
    assert!(fit.atom_evidence.is_empty());
    assert_eq!(
        fit.rank_path.len(),
        1,
        "one proposal was judged: {:?}",
        fit.rank_path
    );
    let step = &fit.rank_path[0];
    assert!(!step.accepted && step.converged, "{step:?}");
    assert!(fit.covariance.iter().all(|c| *c == 0.0));
    assert_eq!(fit.effective_rank, 0.0);
    assert!(fit.loadings.is_empty() && fit.rates.is_empty());
}

/// The exact engine on the cohort that ran away under the Laplace engine
/// (#2808): four marks, three once-only diseases and death, a rank-two
/// latent covariance with loadings at most 0.9. The measured failure there
/// was a fitted `C(0)` diagonal of `[0.0013, 398, 0.0033, 0.15]` against a
/// simulated `[0.81, 0.61, 0.64, 0.09]`: one loading of about 20 carrying an
/// eigenvalue of 398. The exactly marginalised engine has no such corner —
/// with the population centring an event node's `−y a²/2` cancels the
/// `exp(y² a²/2)` the stationary state returns — so its `C(0)` sits within
/// its own posterior uncertainty of the truth.
#[test]
fn a_multi_mark_rank_two_cohort_does_not_run_away() {
    install_test_logger();
    let truth = array![[0.9, 0.0], [0.6, 0.5], [0.0, 0.8], [0.3, 0.0]];
    let (mut cohort, _) = simulate_latent_cohort(
        150,
        4.0,
        &[-1.2, -1.5, -1.3, -2.2],
        0.3,
        &truth,
        &[0.3, 1.5],
        &[
            MarkKind::Once,
            MarkKind::Once,
            MarkKind::Once,
            MarkKind::Terminal,
        ],
        61,
    );
    let mut spec = EventHistorySpec::new(vec![linear_spec()]);
    spec.gauss_hermite_order = 9;
    let started = std::time::Instant::now();
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
    let truth_covariance = truth.dot(&truth.t());
    let (truth_eigenvalues, _) = super::covariance::eigenmodes(&truth_covariance).expect("eigen");
    emit(&format!(
        "[four] {:.1}s rank={} path={:?}",
        started.elapsed().as_secs_f64(),
        fit.rank(),
        fit.rank_path
    ));
    emit(&format!(
        "[four] covariance diagonal fitted={:?} truth={:?}; eigenvalues fitted={:?} sd={:?} truth={:?}; effective rank {:.3}",
        fit.covariance.diag().to_vec(),
        truth_covariance.diag().to_vec(),
        fit.eigenvalues.to_vec(),
        fit.eigenvalue_sd.to_vec(),
        truth_eigenvalues.to_vec(),
        fit.effective_rank
    ));
    assert!(
        fit.rank() >= 1,
        "a rank-two latent covariance was simulated but no atom was grown"
    );
    // No runaway: every fitted eigenvalue lies within three posterior
    // standard deviations of a truth eigenvalue's magnitude, and the top one
    // resolves the simulated leading direction within that band.
    for j in 0..4 {
        let band = 3.0 * fit.eigenvalue_sd[j];
        assert!(
            fit.eigenvalues[j] <= truth_eigenvalues[0] + band,
            "eigenvalue {j} = {} exceeds the largest simulated one {} by more than {band}",
            fit.eigenvalues[j],
            truth_eigenvalues[0]
        );
    }
    assert!(
        (fit.eigenvalues[0] - truth_eigenvalues[0]).abs() <= 3.0 * fit.eigenvalue_sd[0],
        "top eigenvalue {} vs simulated {} with sd {}",
        fit.eigenvalues[0],
        truth_eigenvalues[0],
        fit.eigenvalue_sd[0]
    );
    for (d, (fitted, simulated)) in fit
        .covariance
        .diag()
        .iter()
        .zip(truth_covariance.diag().iter())
        .enumerate()
    {
        assert!(
            *fitted < 4.0 * simulated.max(0.25),
            "mark {d}: fitted latent variance {fitted} ran away from the simulated {simulated}"
        );
    }
    // #2627: the rank was decided, and every free rate fitted, at a mesh that
    // resolves them. On the refinement-0 mesh this cohort's likelihood has no
    // interior rate maximum (job 1206846): a rate decided there runs to the
    // band's fast wall and follows the mesh under refinement. So
    // - the certified mesh is finer than the coarsest one;
    // - a free rate's mode is resolvably off the band's fast wall in the
    //   likelihood: the likelihood at the mode exceeds the likelihood with that
    //   rate moved to the wall by more than both states' resolution (the
    //   defect's rate sat at the wall to 2e-10);
    // - refitted at the certified mesh and the two rungs above it, every
    //   gauge-free quantity (the mark coefficients, the eigenvalues of `C(0)`
    //   the rank carries, the free rates) moves less on the second rung than
    //   on the first, or, where it had already converged to the fits'
    //   resolution, stays within it. This shows the moves shrink under
    //   refinement; it does not claim the second rung is resolved.
    // A fit's resolution is the residual at the state it returned, whichever
    // exit certified it: the joint Newton's target is
    // `residual_tol = (inner_tol·(1 + stationarity_scale)).max(stationarity_band)`
    // (exact_joint_fit.rs, at the cycle head and after each accepted step), so τ
    // alone can understate it.
    // With `r_p = |∇ℓ_p − (Sβ)_p| + ε·(|∇ℓ_p| + |(Sβ)_p|)` measured there,
    // coefficient q lies within `band_q = Σ_p |V_qp|·r_p` of the exact mode to
    // first order, V the posterior covariance: this family has no Jeffreys term
    // and no joint penalty, so V is `(H + S_λ)⁻¹` at that same state
    // (`compute_joint_posterior`). Not bounded here: the second-order remainder
    // through ∂H, the gradient's own evaluation error beyond that subtraction,
    // and the relative rounding of the bars' own arithmetic.
    // An eigenvalue of `C(0) = Σ_k (â_k â_kᵀ + V_kk)` carries that band through
    // `â` to first order and through its covariance part as
    // `Σ_p band_p·‖(V·∂_pH·V)_aa‖_F`, plus the error of the solve that produced
    // V, the rounding of assembling `C(0)`, and the eigensolver's error.
    // - The solve error is Higham (2nd ed.) Thm 10.4 on the Jacobi-equilibrated
    //   `V_eq`: `rp = γ_{3n+1}·‖|R̂ᵀ||R̂|‖_F/‖V_eq‖_F`, `γ_k = kε/(1 − kε)`, n the
    //   coefficient width, giving `κ(V_eq)·rp/(1 − κ(V_eq)·rp)·‖V_aa‖_F`; κ is
    //   first order in its own error.
    // - Rounding in the test's own evaluations is the running error `ε·μ`, μ
    //   accumulated by Running's operation rules: `a·b` charges
    //   `|a|·μ_b + |b|·μ_a + |ab|`, `a ± b` charges `μ_a + μ_b + |a ± b|`, scaling
    //   by an exact s charges `|s|·μ + |s·a|`, and exact inputs carry none.
    // - The assembly error is that running error over `latent_report`'s own
    //   sequence per entry (family.rs): `share[[d, e]] = loadings[[d, k]] *
    //   loadings[[e, k]] + posterior[[qd, qe]]`, then `0.5 * (&share +
    //   &share.t())`, then `covariance += share` over atoms. By Weyl it enters
    //   through the Frobenius norm.
    // - The eigensolver's error is a posteriori. For a symmetric A and x ≠ 0,
    //   `‖(A − λI)x‖₂ ≥ min_i |λ_i − λ|·‖x‖₂` (expand x in A's eigenbasis), so
    //   an eigenvalue of A lies within `‖Ax − λx‖₁/max_d |x_d|` of λ, each
    //   residual component carrying its running error, and each interval's
    //   own running error enters the pairing margin. When the marks intervals
    //   are pairwise disjoint each holds exactly one, in order.
    //   `C(0)` is assembled exactly symmetric: `0.5·(S + Sᵀ)` and the sums
    //   commute.
    // The rate chart is `ν(u) = ν_min + (ν_max − ν_min)·u²/(1 + u²)`
    // (`family::rate_from_chart`), which reaches the fast wall only as u → ∞.
    use super::scalar::Tangent;
    let marks = 4;
    let (order, refinement) = (fit.quadrature.gauss_hermite_order, fit.quadrature.mesh_refinement);
    emit(&format!(
        "[four] certified at Gauss-Hermite order {order}, mesh refinement {refinement}, mesh ceiling {}",
        cohort.mesh_refinement_ceiling()
    ));
    assert!(
        refinement >= 1,
        "certified at mesh refinement {refinement}: the rank was decided on the coarsest mesh, which does not resolve this cohort's rate, not at a resolved mesh"
    );
    assert!(
        refinement + 2 <= cohort.mesh_refinement_ceiling(),
        "certified at mesh refinement {refinement}, less than two rungs below the mesh ceiling, so convergence cannot be measured"
    );
    let mark_width: usize = fit.fit.block_states[..marks].iter().map(|s| s.beta.len()).sum();
    let band = fit.family.rate_band();
    // Per coefficient, how far the residual at the returned state lets its
    // mode sit from the exact one.
    let resolution = |model: &EventHistoryFit| -> Vec<f64> {
        let covariance = model.fit.beta_covariance().expect("posterior covariance");
        let gradient = model
            .family
            .joint_evaluation(&model.fit.block_states)
            .expect("joint evaluation")
            .gradient
            .clone();
        let atoms = model.rank();
        let latent = &model.fit.block_states[marks].beta;
        let n_lambda = model.fit.log_lambdas.len();
        let mut penalty_gradient = Array1::<f64>::zeros(gradient.len());
        for k in 0..atoms {
            let lambda = model.fit.log_lambdas[n_lambda - atoms + k].exp();
            for d in 0..marks {
                penalty_gradient[mark_width + d * atoms + k] = lambda * latent[d * atoms + k];
            }
        }
        let residual: Vec<f64> = gradient
            .iter()
            .zip(penalty_gradient.iter())
            .map(|(g, s)| (g - s).abs() + f64::EPSILON * (g.abs() + s.abs()))
            .collect();
        (0..gradient.len())
            .map(|q| covariance.row(q).iter().zip(&residual).map(|(v, r)| v.abs() * r).sum::<f64>())
            .collect()
    };
    // The free rates in the gauge's order, each with its posterior sd and its
    // resolution (both carried through the chart).
    let free_rates = |model: &EventHistoryFit| -> Vec<(f64, f64, f64)> {
        let covariance = model.fit.beta_covariance().expect("posterior covariance");
        let bands = resolution(model);
        let latent = &model.fit.block_states[marks].beta;
        let mut slot = marks * model.rank();
        let mut rates = Vec::new();
        for held in model.family.rate_held() {
            if held {
                continue;
            }
            let q = mark_width + slot;
            let jet = super::family::rate_from_chart(band, &Tangent::<1>::seeded(latent[slot], [1.0]));
            rates.push((
                jet.value,
                jet.grad[0].abs() * covariance[[q, q]].max(0.0).sqrt(),
                jet.grad[0].abs() * bands[q],
            ));
            slot += 1;
        }
        rates.sort_by(|a, b| a.0.total_cmp(&b.0));
        rates
    };
    let eigen_resolution = |model: &EventHistoryFit| -> Vec<f64> {
        let bands = resolution(model);
        let covariance = model.fit.beta_covariance().expect("posterior covariance");
        let atoms = model.rank();
        let latent = &model.fit.block_states[marks].beta;
        let width = covariance.nrows();
        let frobenius = |matrix: &Array2<f64>| matrix.iter().map(|c| c * c).sum::<f64>().sqrt();
        // The covariance part of `C(0)`: the loading blocks of a posterior
        // covariance, per pair of marks, summed over atoms.
        let covariance_part = |matrix: &Array2<f64>| -> Array2<f64> {
            Array2::from_shape_fn((marks, marks), |(d, e)| {
                (0..atoms)
                    .map(|k| matrix[[mark_width + d * atoms + k, mark_width + e * atoms + k]])
                    .sum()
            })
        };
        // How far the covariance part can move while the mode moves within
        // its band: `dV = −V·dH·V` along each coefficient. By Weyl this bounds
        // every eigenvalue's move alike.
        let mut spread = 0.0;
        for p in 0..width {
            let mut direction = Array1::<f64>::zeros(width);
            direction[p] = 1.0;
            let derivative = model
                .family
                .directional_hessian(&model.fit.block_states, &direction)
                .expect("directional Hessian");
            let moved = covariance.dot(&derivative).dot(covariance);
            spread += bands[p] * frobenius(&covariance_part(&moved));
        }
        // Higham (2nd ed.) Thm 10.4 on the Jacobi-equilibrated V, with
        // `|R̂ᵀ||R̂| = |L||L|ᵀ` for the Cholesky factor `V_eq = L·Lᵀ`.
        let scale: Vec<f64> = (0..width)
            .map(|q| covariance[[q, q]].max(0.0).sqrt().recip())
            .collect();
        let equilibrated =
            Array2::from_shape_fn((width, width), |(p, q)| scale[p] * covariance[[p, q]] * scale[q]);
        let lower = gam_linalg::faer_ndarray::FaerCholesky::cholesky(&equilibrated, faer::Side::Lower)
            .expect("equilibrated posterior Cholesky")
            .lower_triangular()
            .mapv(f64::abs);
        let steps = (3 * width + 1) as f64;
        let gamma = steps * f64::EPSILON / (1.0 - steps * f64::EPSILON);
        let rp = gamma * frobenius(&lower.dot(&lower.t())) / frobenius(&equilibrated);
        let (spectrum, _) = super::covariance::eigenmodes(&equilibrated).expect("equilibrated spectrum");
        let largest = spectrum.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let smallest = spectrum.iter().fold(f64::INFINITY, |m, x| m.min(x.abs()));
        let kappa = largest / smallest;
        emit(&format!("[four] posterior solve: κ(V_eq) = {kappa}, rp = {rp}, κ·rp = {}", kappa * rp));
        assert!(
            kappa * rp < 1.0,
            "the posterior solve is unresolved: κ(V_eq) = {kappa}, rp = {rp}"
        );
        let solve = kappa * rp / (1.0 - kappa * rp) * frobenius(&covariance_part(covariance));
        // Assembling `C(0)`: the running error of latent_report's operations
        // per entry, by Weyl through ‖·‖_F.
        let assembly = frobenius(&Array2::from_shape_fn((marks, marks), |(d, e)| {
            let (mut sum, mut mu_sum) = (0.0_f64, 0.0_f64);
            for k in 0..atoms {
                let (qd, qe) = (mark_width + d * atoms + k, mark_width + e * atoms + k);
                let product = latent[d * atoms + k] * latent[e * atoms + k];
                let (share_de, share_ed) = (product + covariance[[qd, qe]], product + covariance[[qe, qd]]);
                let (mu_de, mu_ed) = (product.abs() + share_de.abs(), product.abs() + share_ed.abs());
                let symmetric = share_de + share_ed;
                let mu_symmetric = mu_de + mu_ed + symmetric.abs();
                let half = 0.5 * symmetric;
                sum += half;
                mu_sum += 0.5 * mu_symmetric + half.abs() + sum.abs();
            }
            f64::EPSILON * mu_sum
        }));
        // The eigensolver, a posteriori: every computed pair's residual
        // interval with that interval's own running error. Each residual
        // component carries its running error, and the sum over components
        // and the division by the exact `max_d |v_d|` charge theirs.
        let radius: Vec<(f64, f64)> = (0..marks)
            .map(|j| {
                let v = model.eigenvectors.column(j);
                let lambda = model.eigenvalues[j];
                let (mut total, mut mu_total) = (0.0_f64, 0.0_f64);
                for d in 0..marks {
                    let (mut computed, mut mu) = (0.0_f64, 0.0_f64);
                    for e in 0..marks {
                        let term = model.covariance[[d, e]] * v[e];
                        computed += term;
                        mu += term.abs() + computed.abs();
                    }
                    let shift = lambda * v[d];
                    computed -= shift;
                    mu += shift.abs() + computed.abs();
                    let floor = f64::EPSILON * mu;
                    let component = computed.abs() + floor;
                    total += component;
                    mu_total += floor + component + total.abs();
                }
                let largest = v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                let value = total / largest;
                (value, mu_total / largest + value.abs())
            })
            .collect();
        for i in 0..marks {
            for j in i + 1..marks {
                let gap = (model.eigenvalues[i] - model.eigenvalues[j]).abs();
                let reach = radius[i].0 + radius[j].0;
                let mu_reach = radius[i].1 + radius[j].1 + reach.abs();
                let margin = gap - reach;
                let rounding = f64::EPSILON * (gap + mu_reach + margin.abs());
                assert!(
                    margin > rounding,
                    "eigenvalues {i} and {j} of C(0) are {gap} apart, within their residual radii {} and {} (margin {margin}, rounding {rounding}): the eigensolver's error does not pair them",
                    radius[i].0,
                    radius[j].0
                );
            }
        }
        (0..atoms)
            .map(|j| {
                let v = model.eigenvectors.column(j);
                let mut propagated = 0.0;
                for k in 0..atoms {
                    let projection: f64 = (0..marks).map(|d| v[d] * latent[d * atoms + k]).sum();
                    for d in 0..marks {
                        propagated += (2.0 * v[d] * projection).abs() * bands[mark_width + d * atoms + k];
                    }
                }
                propagated + spread + solve + assembly + radius[j].0 + f64::EPSILON * radius[j].1
            })
            .collect()
    };
    let rates = free_rates(&fit);
    for (rate, sd, _) in &rates {
        emit(&format!("[four] free rate {rate}: wall distance {}, posterior sd {sd}", band.1 - rate));
    }
    // Off the wall, in the likelihood: for every free rate, the likelihood at
    // the mode exceeds the likelihood with that rate moved to the band's fast
    // wall (`rate_chart(band, ν_max)`, the chart's largest representable point)
    // by more than both states' resolution. It is the claim at the mode's other
    // coefficients: the likelihood resolvably drops when the rate moves to the
    // wall. It does not establish a non-flat profile likelihood, since
    // re-optimising the other coefficients at the wall can only raise ℓ there;
    // a profile refit per free rate would multiply this fixture's cost.
    // - With the exact mode `β̂ + δ`, `|δ_p| ≤ band_p`, the likelihood at the
    //   mode moves by at most `Σ_p |∂_pℓ(β̂)|·band_p` to first order.
    // - The wall state's rate is pinned, so its likelihood moves by at most
    //   `Σ_{p≠u} |∂_pℓ(wall)|·band_p`, to first order.
    // - The subtraction adds its running error `ε·(|ℓ̂| + |ℓ̂_wall| + |Δ̂|)`.
    // Not bounded here: the second-order remainder, and ℓ's own evaluation
    // error beyond that subtraction.
    // The defect state refuses it (job 1225126, arm md: the rank-1 fit pinned
    // at mesh refinement 0): ℓ̂ − ℓ_wall = −5.10e-11 against 5.2e-10, the
    // mode's gradient on both sides.
    // A rate-space claim was dropped. Under the measured-residual band the
    // defect's far end, `|u| + band_u = 176832`, maps 1.3e-10 off the wall,
    // resolvably, because the chart reaches the wall only near `u = 2^26`. So
    // a distance in ν cannot separate the defect from an interior rate (same
    // job). A posterior-interval claim was dropped too: it is false for a
    // correct fit on this cohort (job 1215973: rate 2.064, wall distance
    // 1.986, posterior sd 1.596). The sd and the distance are printed, not
    // asserted.
    let mode = fit.family.joint_evaluation(&fit.fit.block_states).expect("joint evaluation at the mode");
    let mode_bands = resolution(&fit);
    let certificate_mode: f64 = mode.gradient.iter().zip(&mode_bands).map(|(g, b)| g.abs() * b).sum();
    let mut slot = marks * fit.rank();
    for (atom, held) in fit.family.rate_held().into_iter().enumerate() {
        if held {
            continue;
        }
        let q = mark_width + slot;
        let mut wall = fit.fit.block_states.clone();
        wall[marks].beta[slot] = super::family::rate_chart(band, band.1);
        let at_wall = fit.family.joint_evaluation(&wall).expect("joint evaluation at the wall");
        let certificate_wall: f64 = at_wall
            .gradient
            .iter()
            .zip(&mode_bands)
            .enumerate()
            .filter(|(p, _)| *p != q)
            .map(|(_, (g, b))| g.abs() * b)
            .sum();
        let difference = mode.log_likelihood - at_wall.log_likelihood;
        let rounding =
            f64::EPSILON * (mode.log_likelihood.abs() + at_wall.log_likelihood.abs() + difference.abs());
        let bar = certificate_mode + certificate_wall + rounding;
        emit(&format!(
            "[four] atom {atom}: ℓ at the mode {}, at the fast wall {}, difference {difference}, resolution {bar} (mode certificate {certificate_mode}, wall certificate {certificate_wall}, subtraction {rounding})",
            mode.log_likelihood, at_wall.log_likelihood
        ));
        assert!(
            difference > bar,
            "atom {atom}'s free rate is not resolvably off the band's fast wall: at the mode's other coefficients, ℓ at the mode exceeds ℓ with the rate at the wall by {difference}, within the resolution {bar}"
        );
        slot += 1;
    }
    let start = RankStart::carried(
        fit.fit.block_states[..marks].iter().map(|s| s.beta.clone()).collect(),
        fit.loadings.iter().copied().collect(),
        fit.log_rates.clone(),
        fit.atom_log_lambdas.clone(),
        fit.rate_held.clone(),
    );
    // The refits are the no-smoothing route: no block of this fixture has a
    // free penalty (the loading priors are fixed log-λ, the linear mark terms
    // carry none, so `rho0` is empty) and no outer strength is re-optimised,
    // so each fit's inner band is its whole coefficient resolution.
    // Without reference strata the reference refinement is never read.
    emit(&format!("[four] reference strata present: {}", spec.reference.is_some()));
    assert!(spec.reference.is_none(), "this cohort has no reference population");
    let refit_at = |mesh: usize| {
        super::family::fit_at_rank(&cohort, &spec, fit.rank(), Some(&start), Some((order, mesh)), mesh, 2)
            .unwrap_or_else(|error| panic!("the certified model refitted at mesh refinement {mesh}: {error}"))
    };
    let certified = refit_at(refinement);
    let one_up = refit_at(refinement + 1);
    let two_up = refit_at(refinement + 2);
    let models = [&certified, &one_up, &two_up];
    // One quantity over the three meshes: its move shrinks from the first
    // rung to the second by more than both moves' resolutions, or, where the
    // first rung moved it by no more than the two fits' resolution, the second
    // rung moves it by less than theirs. A move's resolution adds its
    // subtraction's rounding `ε·(|v_i| + |v_j|)`, and each comparison charges
    // the running errors of its operands and of its own operations. A shrink
    // that holds but is not resolved fails: it is undecidable at this fixture,
    // which is a finding. A shrinking move is what this shows, not a resolved
    // second rung.
    let converges = |name: &str, values: [f64; 3], resolutions: [f64; 3]| {
        let first_move = (values[1] - values[0]).abs();
        let second_move = (values[2] - values[1]).abs();
        // A subtraction of two inputs charges its own magnitude.
        let (mu_first_move, mu_second_move) = (first_move, second_move);
        // `r_i + r_j + ε·(|v_i| + |v_j|)` with its running error.
        let resolution_of = |r_i: f64, r_j: f64, v_i: f64, v_j: f64| {
            let sum = r_i + r_j;
            let floor = f64::EPSILON * (v_i.abs() + v_j.abs());
            let total = sum + floor;
            (total, sum.abs() + 2.0 * floor.abs() + total.abs())
        };
        let (first_resolution, mu_first_resolution) =
            resolution_of(resolutions[0], resolutions[1], values[0], values[1]);
        let (second_resolution, mu_second_resolution) =
            resolution_of(resolutions[1], resolutions[2], values[1], values[2]);
        if first_move > first_resolution {
            assert!(
                second_move < first_move,
                "{name} moves {first_move} from mesh refinement {refinement} to {} but {second_move} from {} to {}: its move does not shrink under refinement",
                refinement + 1,
                refinement + 1,
                refinement + 2
            );
            let (reach, spread) = (first_move - first_resolution, second_move + second_resolution);
            let mu_reach = mu_first_move + mu_first_resolution + reach.abs();
            let mu_spread = mu_second_move + mu_second_resolution + spread.abs();
            let margin = reach - spread;
            let rounding = f64::EPSILON * (mu_reach + mu_spread + margin.abs());
            assert!(
                margin > rounding,
                "{name} moves {first_move} (resolution {first_resolution}) from mesh refinement {refinement} to {} and {second_move} (resolution {second_resolution}) from {} to {}: the shrink is not resolved at this fixture (margin {margin}, rounding {rounding})",
                refinement + 1,
                refinement + 1,
                refinement + 2
            );
        } else {
            let margin = second_resolution - second_move;
            let rounding = f64::EPSILON * (mu_second_resolution + mu_second_move + margin.abs());
            assert!(
                margin > rounding,
                "{name} had converged ({first_move} within the resolution {first_resolution}) but moves {second_move} from mesh refinement {} to {}, not resolved within the resolution {second_resolution} (margin {margin}, rounding {rounding})",
                refinement + 1,
                refinement + 2
            );
        }
    };
    let coefficients = models.map(|model| {
        model.fit.block_states[..marks].iter().flat_map(|s| s.beta.iter().copied()).collect::<Vec<f64>>()
    });
    let bands = models.map(|model| resolution(model));
    for q in 0..mark_width {
        converges(
            &format!("mark coefficient {q}"),
            [coefficients[0][q], coefficients[1][q], coefficients[2][q]],
            [bands[0][q], bands[1][q], bands[2][q]],
        );
    }
    let eigen_bands = models.map(|model| eigen_resolution(model));
    for j in 0..fit.rank() {
        converges(
            &format!("eigenvalue {j} of C(0)"),
            [models[0].eigenvalues[j], models[1].eigenvalues[j], models[2].eigenvalues[j]],
            [eigen_bands[0][j], eigen_bands[1][j], eigen_bands[2][j]],
        );
    }
    let rate_sets = models.map(|model| free_rates(model));
    emit(&format!(
        "[four] free rates per refit: certified {}, refits {:?}",
        rates.len(),
        rate_sets.iter().map(|set| set.len()).collect::<Vec<_>>()
    ));
    assert!(
        rate_sets.iter().all(|set| set.len() == rates.len()),
        "a refit carries a different set of free rates"
    );
    for i in 0..rates.len() {
        converges(
            &format!("free rate {i}"),
            [rate_sets[0][i].0, rate_sets[1][i].0, rate_sets[2][i].0],
            [rate_sets[0][i].2, rate_sets[1][i].2, rate_sets[2][i].2],
        );
    }
}

/// The smoothed latent state is the posterior of the simulated path: its
/// mean tracks the path the events were generated from, and its variance
/// is the prior's narrowed by the history.
#[test]
fn the_smoothed_latent_state_tracks_the_simulated_path() {
    install_test_logger();
    let (mut cohort, paths) = simulate_latent_cohort(
        80,
        6.0,
        &[-0.8],
        0.5,
        &array![[1.0]],
        &[0.4],
        &[MarkKind::Recurrent],
        7,
    );
    let mut spec = EventHistorySpec::new(vec![linear_spec()]);
    spec.gauss_hermite_order = 11;
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
    assert_rank_stop_explained(&fit, &spec);
    assert!(
        fit.rank() >= 1,
        "the shared risk was not grown: {:?}",
        fit.rank_path
    );
    let dt = 6.0 / 400.0;
    let (mut sum_xy, mut sum_xx, mut sum_yy, mut sum_x, mut sum_y, mut count) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (subject, path) in cohort.subjects.iter().zip(paths.iter()) {
        let state = latent_state(&fit, &cohort, subject, 0).expect("latent state");
        assert_eq!(state.mean.nrows(), state.times.len());
        assert_eq!(state.covariance.len(), state.times.len());
        for (n, &t) in state.times.iter().enumerate() {
            let step = ((t / dt).floor() as usize).min(path.nrows() - 1);
            let truth = path[[step, 0]];
            let mean = state.mean[[n, 0]];
            let variance = state.covariance[n][[0, 0]];
            assert!(
                variance > 0.0 && variance <= 1.0 + 1e-9,
                "the smoothed variance {variance} must lie in (0, 1]: the prior narrowed by the history"
            );
            sum_xy += mean * truth;
            sum_xx += mean * mean;
            sum_yy += truth * truth;
            sum_x += mean;
            sum_y += truth;
            count += 1.0;
        }
    }
    let covariance = sum_xy / count - (sum_x / count) * (sum_y / count);
    let correlation = covariance
        / ((sum_xx / count - (sum_x / count).powi(2)) * (sum_yy / count - (sum_y / count).powi(2)))
            .sqrt();
    emit(&format!(
        "[state] correlation of the smoothed mean with the simulated path over {count} nodes: {correlation:.3}"
    ));
    assert!(
        correlation > 0.5,
        "the smoothed latent mean should track the simulated path; correlation {correlation}"
    );
}

/// Solve `A x = b` for a small dense system by Gaussian elimination.
fn solve_small(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut m = a.clone();
    let mut r = b.clone();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| m[[i, col]].abs().total_cmp(&m[[j, col]].abs()))
            .expect("row");
        if pivot != col {
            for k in 0..n {
                let t = m[[col, k]];
                m[[col, k]] = m[[pivot, k]];
                m[[pivot, k]] = t;
            }
            let t = r[col];
            r[col] = r[pivot];
            r[pivot] = t;
        }
        for i in (col + 1)..n {
            let f = m[[i, col]] / m[[col, col]];
            for k in col..n {
                m[[i, k]] -= f * m[[col, k]];
            }
            r[i] -= f * r[col];
        }
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut acc = r[i];
        for k in (i + 1)..n {
            acc -= m[[i, k]] * x[k];
        }
        x[i] = acc / m[[i, i]];
    }
    x
}

#[test]
fn newton_direction_decreases_the_penalised_objective_at_the_start() {
    let mut cohort = simulate_cohort(80, 6.0, -0.8, 0.5, 0.0, 0.4, 3);
    cohort.validate().expect("valid");
    let nodes = Arc::new(expand_nodes(&cohort, 9, 0).expect("nodes"));
    let total = nodes.total_nodes;
    let mut design = Array2::<f64>::zeros((total, 2));
    for row in 0..total {
        design[[row, 0]] = 1.0;
        design[[row, 1]] = nodes.node_data[[row, 0]];
    }
    let family = EventHistoryFamily::new(
        Arc::clone(&nodes),
        vec![Arc::new(design.clone())],
        1,
        11,
        cohort.time_scale(),
        vec![None],
    )
    .expect("family");
    let states = |beta: &Array1<f64>, latent: &Array1<f64>| {
        vec![
            ParameterBlockState {
                beta: beta.clone(),
                eta: design.dot(beta),
            },
            ParameterBlockState {
                beta: latent.clone(),
                eta: Array1::zeros(total),
            },
        ]
    };
    // Ridge on the atom's loading at λ = 1; the rate is a structural
    // coordinate with no penalty, and its domain is `ν > 0`.
    let penalty = |latent: &Array1<f64>| 0.5 * latent[0] * latent[0];
    let objective = |beta: &Array1<f64>, latent: &Array1<f64>| -> f64 {
        -family.log_likelihood(&states(beta, latent)).expect("value") + penalty(latent)
    };
    let beta0 = array![0.0, 0.0];
    let latent0 = array![0.0, 1.0];
    let joint = family
        .joint_evaluation(&states(&beta0, &latent0))
        .expect("joint");
    let value_only = family
        .log_likelihood(&states(&beta0, &latent0))
        .expect("value");
    assert_eq!(
        joint.log_likelihood, value_only,
        "value-only and derivative paths must agree bitwise"
    );
    let p = 4;
    for i in 0..p {
        let h_fd = 1e-6;
        let mut plus = Array1::<f64>::zeros(p);
        plus[i] = h_fd;
        let (bp, lp) = (
            &beta0 + &plus.slice(ndarray::s![0..2]),
            &latent0 + &plus.slice(ndarray::s![2..4]),
        );
        let (bm, lm) = (
            &beta0 - &plus.slice(ndarray::s![0..2]),
            &latent0 - &plus.slice(ndarray::s![2..4]),
        );
        let fd = (objective(&bp, &lp) - objective(&bm, &lm)) / (2.0 * h_fd);
        let analytic = -joint.gradient[i] + if i == 2 { latent0[0] } else { 0.0 };
        emit(&format!(
            "coefficient {i}: analytic {analytic} vs finite difference {fd}"
        ));
    }
    let mut h = joint.hessian.clone();
    let mut g = -joint.gradient.clone();
    h[[2, 2]] += 1.0;
    g[2] += latent0[0];
    let direction = solve_small(&h, &g.mapv(|v| -v));
    let base = objective(&beta0, &latent0);
    let mut decreased = false;
    for &t in &[1.0, 0.5, 0.25, 0.125, 0.0625] {
        let beta = &beta0
            + &direction
                .slice(ndarray::s![0..2])
                .to_owned()
                .mapv(|v| v * t);
        let latent = &latent0
            + &direction
                .slice(ndarray::s![2..4])
                .to_owned()
                .mapv(|v| v * t);
        let trial = objective(&beta, &latent);
        let predicted = t * g.dot(&direction) + 0.5 * t * t * direction.dot(&h.dot(&direction));
        println!(
            "t={t}: objective {base} -> {trial} (actual {:+e}, model {predicted:+e})",
            trial - base
        );
        if trial < base {
            decreased = true;
        }
    }
    assert!(
        decreased,
        "no step along the Newton direction decreases the objective"
    );
    // The gradient must be the derivative of the value along the direction.
    let h_fd = 1e-5;
    let beta = &beta0
        + &direction
            .slice(ndarray::s![0..2])
            .to_owned()
            .mapv(|v| v * h_fd);
    let latent = &latent0
        + &direction
            .slice(ndarray::s![2..4])
            .to_owned()
            .mapv(|v| v * h_fd);
    let beta_m = &beta0
        - &direction
            .slice(ndarray::s![0..2])
            .to_owned()
            .mapv(|v| v * h_fd);
    let latent_m = &latent0
        - &direction
            .slice(ndarray::s![2..4])
            .to_owned()
            .mapv(|v| v * h_fd);
    let fd = (objective(&beta, &latent) - objective(&beta_m, &latent_m)) / (2.0 * h_fd);
    let analytic = g.dot(&direction);
    println!("directional derivative: analytic {analytic} vs finite difference {fd}");
    assert!(
        (fd - analytic).abs() < 1e-4 * (1.0 + analytic.abs()),
        "directional derivative {analytic} vs finite difference {fd}"
    );
}

#[test]
fn lagrange_basis_keeps_its_derivative_within_roundoff_of_a_node() {
    // A point on, or within roundoff of, a node must carry the same dL_i/dx
    // as a point a hair away, where nothing is singular.
    let gh = GaussHermite::new(9).expect("rule");
    for hit in 0..gh.order {
        let x = gh.nodes[hit];
        let step = 1e-6;
        let plus = gh.lagrange_basis(&OneSeed::<0>::seeded(x + step, 1.0, 0.0));
        let minus = gh.lagrange_basis(&OneSeed::<0>::seeded(x - step, 1.0, 0.0));
        for offset in [0.0, 1e-17, -1e-17, 1e-13] {
            let on = gh.lagrange_basis(&OneSeed::<0>::seeded(x + offset, 1.0, 0.0));
            for i in 0..gh.order {
                let fd_first = (plus[i].value() - minus[i].value()) / (2.0 * step);
                let expected_value = if i == hit { 1.0 } else { 0.0 } + fd_first * offset;
                assert!(
                    (on[i].value() - expected_value).abs() < 1e-12 + 1e-6 * offset.abs(),
                    "node {hit} basis {i} at offset {offset}: value {} vs {expected_value}",
                    on[i].value()
                );
                assert!(
                    (on[i].eps() - fd_first).abs() < 1e-6 * (1.0 + fd_first.abs()),
                    "node {hit} basis {i} at offset {offset}: derivative {} vs finite difference {fd_first}",
                    on[i].eps()
                );
            }
        }
    }
    // Partition of unity and exact reproduction of x^m for m below the order.
    let x = 0.371;
    let basis = gh.lagrange_basis(&x);
    for m in 0..gh.order {
        let reproduced: f64 = basis
            .iter()
            .zip(gh.nodes.iter())
            .map(|(l, node)| l * node.powi(m as i32))
            .sum();
        let exact = x.powi(m as i32);
        assert!(
            (reproduced - exact).abs() < 1e-11 * (1.0 + exact.abs()),
            "x^{m}: {reproduced} vs {exact}"
        );
    }
}

#[test]
fn transition_preserves_small_correlation_and_log_rate_derivatives() {
    for k in [40.0_f64, 100.0, 700.0] {
        let transition = AtomTransition::new(&OneSeed::<0>::seeded(k, 1.0, 0.0));
        let expected_phi = (-k).exp();
        assert_eq!(transition.phi.value(), expected_phi);
        assert_eq!(transition.phi.eps(), -expected_phi);
        assert!((transition.dphi.value() / (-k * expected_phi) - 1.0).abs() < 1.0e-13);
        assert!((transition.dphi.eps() / ((k - 1.0) * expected_phi) - 1.0).abs() < 1.0e-13);
        assert!((transition.d2phi.value() / (k * (k - 1.0) * expected_phi) - 1.0).abs() < 1.0e-13);
    }
    // The correlation underflows before its log-rate derivatives do.
    let transition = AtomTransition::new(&750.0);
    assert_eq!(transition.phi, 0.0);
    assert!(transition.dphi < 0.0);
    assert!(transition.d2phi > 0.0);
}

#[test]
fn effective_rank_is_invariant_to_extreme_covariance_units() {
    let covariance = array![[2.0, 1.0], [1.0, 2.0]];
    // Eigenvalues 3 and 1 give (3+1)^2/(3^2+1^2) = 1.6.
    for scale in [1.0e-200, 1.0, 1.0e200] {
        let rank = super::covariance::effective_rank(&(&covariance * scale));
        assert!((rank - 1.6).abs() < 1.0e-14);
    }
    assert_eq!(super::covariance::effective_rank(&Array2::zeros((2, 2))), 0.0);
}

#[test]
fn transition_at_an_overflowed_rate_is_finite_with_zero_sensitivity() {
    // log-rate 800: exp overflows to infinity, φ is exactly zero, and every
    // derivative channel must be finite (zero), not ∞ · 0.
    let kappa = super::scalar::exp(&OneSeed::<0>::seeded(800.0, 1.0, 0.0)).scale(0.7);
    let transition = AtomTransition::new(&kappa);
    assert_eq!(transition.phi.value(), 0.0);
    assert_eq!(transition.innovation.value(), 1.0);
    for value in [
        &transition.phi,
        &transition.innovation,
        &transition.dphi,
        &transition.d2phi,
    ] {
        assert!(value.value().is_finite() && value.eps().is_finite());
        assert_eq!(value.eps(), 0.0);
    }
    let plain = AtomTransition::new(&f64::INFINITY);
    assert_eq!(plain.dphi, 0.0);
    assert_eq!(plain.d2phi, 0.0);
}

/// The 80-subject loaded cohort the cost and diagnostic tests share.
fn loaded_cohort() -> EventHistoryCohort {
    let mut cohort = simulate_cohort(80, 6.0, -0.8, 0.5, 1.0, 0.4, 7);
    cohort.validate().expect("valid");
    cohort
}

/// Family on the loaded cohort at a Gauss-Hermite order, with the dense
/// linear design and zeroed block states.
fn loaded_family(
    cohort: &EventHistoryCohort,
    order: usize,
) -> (
    EventHistoryFamily,
    Arc<Array2<f64>>,
    Vec<ParameterBlockState>,
) {
    use gam_terms::smooth::build_term_collection_design;
    let nodes = Arc::new(expand_nodes(cohort, 9, 0).expect("nodes"));
    let design =
        build_term_collection_design(nodes.node_data.view(), &linear_spec()).expect("design");
    let dense = design
        .design
        .try_to_dense_arc("test design")
        .expect("dense");
    let family = EventHistoryFamily::new(
        Arc::clone(&nodes),
        vec![Arc::clone(&dense)],
        1,
        order,
        cohort.time_scale(),
        vec![None],
    )
    .expect("family");
    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(2),
            eta: Array1::zeros(nodes.total_nodes),
        },
        ParameterBlockState {
            beta: Array1::zeros(2),
            eta: Array1::zeros(nodes.total_nodes),
        },
    ];
    (family, dense, states)
}

#[test]
fn smallest_prefix_with_non_finite_louis_output_at_order_21() {
    let cohort = loaded_cohort();
    let (_, design_dense, _) = loaded_family(&cohort, 21);
    let beta = array![-0.9485, 0.5452];
    let eta = design_dense.dot(&beta);
    let gh = GaussHermite::new(21).expect("rule");
    let nodes = expand_nodes(&cohort, 9, 0).expect("nodes");
    let mut report = Vec::new();
    for (s, subj) in nodes.subjects.iter().enumerate() {
        let n = subj.len();
        let run = |len: usize| -> bool {
            let prefix = SubjectNodes {
                first_row: 0,
                times: subj.times[..len].to_vec(),
                gaps: subj.gaps[..len - 1].to_vec(),
                weights: subj.weights[..len].to_vec(),
                exposures: subj.exposures.slice(ndarray::s![..len, ..]).to_owned(),
                counts: subj.counts.slice(ndarray::s![..len, ..]).to_owned(),
                covariate_rows: subj.covariate_rows[..len].to_vec(),
            };
            let eta0: Vec<f64> = (0..len).map(|i| eta[subj.first_row + i]).collect();
            let inputs = SubjectInputs {
                nodes: &prefix,
                eta0: &eta0,
                loadings: &[1.2054],
                rates: &[1.0587_f64.exp()],
                time_scale: cohort.time_scale(),
                gh: &gh,
                continuation_gap: 0.0,
                designs: None,
                log_normaliser: None,
            };
            match subject_marginal(&inputs, true) {
                Ok(out) => {
                    out.loglik.is_finite()
                        && out.gradient.iter().all(|v| v.is_finite())
                        && out.hessian.iter().all(|v| v.is_finite())
                }
                Err(_) => false,
            }
        };
        if run(n) {
            continue;
        }
        let mut lo = 1;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if run(mid) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let bad = lo;
        report.push(format!(
            "subject {s}: {n} nodes, first non-finite prefix {bad}; node {}: time {} gap {} exposures {:?} counts {:?}; prefix log-lik at {bad}: {:?}",
            bad - 1,
            subj.times[bad - 1],
            if bad >= 2 { subj.gaps[bad - 2] } else { 0.0 },
            subj.exposures.row(bad - 1).to_vec(),
            subj.counts.row(bad - 1).to_vec(),
            {
                let prefix = SubjectNodes {
                    first_row: 0,
                    times: subj.times[..bad].to_vec(),
                    gaps: subj.gaps[..bad - 1].to_vec(),
                    weights: subj.weights[..bad].to_vec(),
                    exposures: subj.exposures.slice(ndarray::s![..bad, ..]).to_owned(),
                    counts: subj.counts.slice(ndarray::s![..bad, ..]).to_owned(),
                    covariate_rows: subj.covariate_rows[..bad].to_vec(),
                };
                let eta0: Vec<f64> = (0..bad).map(|i| eta[subj.first_row + i]).collect();
                let inputs = SubjectInputs {
                    nodes: &prefix,
                    eta0: &eta0,
                    loadings: &[1.2054],
                    rates: &[1.0587_f64.exp()],
                    time_scale: cohort.time_scale(),
                    gh: &gh,
                    continuation_gap: 0.0,
                designs: None,
                log_normaliser: None,
            };
                subject_marginal(&inputs, true).map(|o| {
                    (
                        o.loglik,
                        o.gradient.iter().filter(|v| !v.is_finite()).count(),
                        o.hessian.iter().filter(|v| !v.is_finite()).count(),
                        o.gradient.iter().take(4).copied().collect::<Vec<_>>(),
                    )
                })
            }
        ));
        if report.len() >= 3 {
            break;
        }
    }
    for line in &report {
        emit(line);
    }
    assert!(
        report.is_empty(),
        "{} subjects with non-finite Louis output",
        report.len()
    );
}

#[test]
fn louis_hessian_converges_to_the_computed_curvature_as_the_quadrature_resolves() {
    // Louis' identity is the Hessian of the EXACT marginal; a finite
    // difference of the exact gradient is the Hessian of the COMPUTED one.
    // They are two approximations of the same curvature and differ by the
    // quadrature error, so the property that holds is convergence: raising
    // the Gauss-Hermite order must shrink the gap. Asserting a fixed
    // tolerance at one order instead would be asserting a number nobody
    // derived. The gradient itself is the exact derivative of the computed
    // value, so it agrees with its own finite difference to roundoff, and
    // step-size independence there is what rules out a ripple in the
    // objective.
    //
    // The state is where the fixed-λ inner solve lands on the loaded cohort:
    // the conditional intercept −0.9485 with loading 1.2054, which under the
    // population parameterisation is `−0.9485 + ½·1.2054²`.
    let cohort = loaded_cohort();
    let loading = 1.2054_f64;
    let beta = array![-0.9485 + 0.5 * loading * loading, 0.5452];
    let latent = array![loading, 1.0587_f64.exp()];
    let p = 4;
    let mut discrepancy_by_order = Vec::new();
    for order in [11usize, 21] {
        let (family, design_dense, base_states) = loaded_family(&cohort, order);
        let at = |beta: &Array1<f64>, latent: &Array1<f64>| -> Vec<ParameterBlockState> {
            let mut states = base_states.clone();
            states[0].beta = beta.clone();
            states[0].eta = design_dense.dot(beta);
            states[1].beta = latent.clone();
            states
        };
        let base = family.joint_evaluation(&at(&beta, &latent)).expect("joint");
        let mut full = Array1::<f64>::zeros(p);
        full.slice_mut(ndarray::s![0..2]).assign(&beta);
        full.slice_mut(ndarray::s![2..4]).assign(&latent);
        let split = |v: &Array1<f64>| -> (Array1<f64>, Array1<f64>) {
            (
                v.slice(ndarray::s![0..2]).to_owned(),
                v.slice(ndarray::s![2..4]).to_owned(),
            )
        };
        let mut worst = 0.0_f64;
        let mut worst_at = (0usize, 0usize);
        for h in [1e-3, 1e-5] {
            for i in 0..p {
                let mut plus = full.clone();
                plus[i] += h;
                let mut minus = full.clone();
                minus[i] -= h;
                let (bp, lp) = split(&plus);
                let (bm, lm) = split(&minus);
                let gp = family
                    .joint_evaluation(&at(&bp, &lp))
                    .expect("joint")
                    .gradient
                    .clone();
                let gm = family
                    .joint_evaluation(&at(&bm, &lm))
                    .expect("joint")
                    .gradient
                    .clone();
                let vp = family.log_likelihood(&at(&bp, &lp)).expect("value");
                let vm = family.log_likelihood(&at(&bm, &lm)).expect("value");
                let fd_gradient = (vp - vm) / (2.0 * h);
                let fd_row: Vec<String> = (0..p)
                    .map(|j| format!("{:.5e}", -(gp[j] - gm[j]) / (2.0 * h)))
                    .collect();
                let louis_row: Vec<String> = (0..p)
                    .map(|j| format!("{:.5e}", base.hessian[[i, j]]))
                    .collect();
                emit(&format!(
                    "[final G={order} h={h:.0e}] coefficient {i}: gradient exact {:.6e} fd {:.6e}; -hessian louis [{}] fd [{}]",
                    base.gradient[i],
                    fd_gradient,
                    louis_row.join(", "),
                    fd_row.join(", ")
                ));
                if h == 1e-5 {
                    // The exact gradient IS the derivative of the computed
                    // value, so it meets its own central difference at the
                    // step where truncation is negligible.
                    assert!(
                        (base.gradient[i] - fd_gradient).abs() < 1e-6 * (1.0 + fd_gradient.abs()),
                        "G={order}: gradient {i} exact {} vs finite difference {fd_gradient}",
                        base.gradient[i]
                    );
                    for j in 0..p {
                        let fd_h = -(gp[j] - gm[j]) / (2.0 * h);
                        let relative = (base.hessian[[i, j]] - fd_h).abs() / (1.0 + fd_h.abs());
                        if relative > worst {
                            worst = relative;
                            worst_at = (i, j);
                        }
                    }
                }
            }
        }
        emit(&format!(
            "[louis] G={order}: worst relative gap {worst:.4e} at entry {worst_at:?}"
        ));
        discrepancy_by_order.push(worst);
    }
    let (coarse, fine) = (discrepancy_by_order[0], discrepancy_by_order[1]);
    assert!(
        fine < 0.5 * coarse,
        "the gap between Louis' Hessian and the computed curvature must shrink with the quadrature: {coarse:.4e} at order 11, {fine:.4e} at order 21"
    );
    assert!(
        fine < 5e-2,
        "at order 21 the two curvatures should agree to a few percent; the worst entry differs by {fine:.4e}"
    );
}

#[test]
fn spline_basis_reproduces_cubics_with_a_bounded_operator_norm() {
    let gh = GaussHermite::new(21).expect("rule");
    let g = gh.order;
    // Exact on linear data, inside and beyond the hull.
    let linear: Vec<f64> = gh.nodes.iter().map(|x| 0.4 - 0.7 * x).collect();
    for &x in &[-9.0, -5.0, -1.3, 0.0, 0.27, 2.9, 5.6, 8.0] {
        let basis = gh.spline_basis(&x);
        let value: f64 = basis.iter().zip(linear.iter()).map(|(b, f)| b * f).sum();
        assert!(
            (value - (0.4 - 0.7 * x)).abs() < 1e-12,
            "linear at {x}: {value}"
        );
        let unity: f64 = basis.iter().sum();
        assert!(
            (unity - 1.0).abs() < 1e-12,
            "partition of unity at {x}: {unity}"
        );
    }
    // Exact on cubic data everywhere on the hull (not-a-knot).
    let cubic: Vec<f64> = gh
        .nodes
        .iter()
        .map(|x| 0.2 * x * x * x - x * x + 0.5 * x - 1.0)
        .collect();
    for step in 0..200 {
        let x = gh.nodes[0] + (gh.nodes[g - 1] - gh.nodes[0]) * step as f64 / 199.0;
        let value: f64 = gh
            .spline_basis(&x)
            .iter()
            .zip(cubic.iter())
            .map(|(b, f)| b * f)
            .sum();
        let exact = 0.2 * x * x * x - x * x + 0.5 * x - 1.0;
        assert!(
            (value - exact).abs() < 1e-9 * (1.0 + exact.abs()),
            "cubic at {x}: {value} vs {exact}"
        );
    }
    // Interpolates the nodal values exactly.
    let data: Vec<f64> = gh
        .nodes
        .iter()
        .map(|x| (-(x * x) / 3.0).exp() * (1.0 + x))
        .collect();
    for j in 0..g {
        let basis = gh.spline_basis(&gh.nodes[j]);
        let value: f64 = basis.iter().zip(data.iter()).map(|(b, f)| b * f).sum();
        assert!(
            (value - data[j]).abs() < 1e-12,
            "node {j}: {value} vs {}",
            data[j]
        );
    }
    // Neither interpolant preserves the nodal range, but their operator
    // norms differ in kind: the cubic spline's `max_x Σ_j |S_j(x)|` is a
    // small constant on these nodes, the Lagrange interpolant's is the
    // Lebesgue constant, exponential in the order. The norm is a theorem
    // about the overshoot: `|S f(x) − c| ≤ ‖S‖ max_j |f_j − c|` for any
    // centre `c`, so centred data cannot overshoot by more than
    // `(‖S‖ − 1)` times its half-range.
    let mut spline_norm = 1.0_f64;
    let mut lagrange_norm = 1.0_f64;
    for step in 0..2000 {
        let x = gh.nodes[0] + (gh.nodes[g - 1] - gh.nodes[0]) * step as f64 / 1999.0;
        spline_norm = spline_norm.max(gh.spline_basis(&x).iter().map(|b| b.abs()).sum());
        lagrange_norm = lagrange_norm.max(gh.lagrange_basis(&x).iter().map(|b| b.abs()).sum());
    }
    assert!(
        spline_norm < 3.0,
        "cubic spline operator norm {spline_norm}"
    );
    assert!(
        lagrange_norm > 100.0,
        "Lagrange operator norm {lagrange_norm}"
    );
    assert!((gh.lebesgue_constant - lagrange_norm).abs() < 0.05 * lagrange_norm);
    let steep: Vec<f64> = gh
        .nodes
        .iter()
        .map(|x| -12.0 * (x + 1.0).abs().powf(1.5) + 3.0)
        .collect();
    let (lo, hi) = steep
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), v| {
            (l.min(*v), h.max(*v))
        });
    let centre = 0.5 * (lo + hi);
    let half_range = 0.5 * (hi - lo);
    let mut spline_overshoot = 0.0_f64;
    let mut lagrange_overshoot = 0.0_f64;
    for step in 0..400 {
        let x = gh.nodes[0] + (gh.nodes[g - 1] - gh.nodes[0]) * step as f64 / 399.0;
        let spline: f64 = gh
            .spline_basis(&x)
            .iter()
            .zip(steep.iter())
            .map(|(b, f)| b * f)
            .sum();
        spline_overshoot = spline_overshoot.max((spline - hi).max(lo - spline));
        let lagrange: f64 = gh
            .lagrange_basis(&x)
            .iter()
            .zip(steep.iter())
            .map(|(b, f)| b * f)
            .sum();
        lagrange_overshoot = lagrange_overshoot.max((lagrange - hi).max(lo - lagrange));
    }
    assert!(
        spline_overshoot <= (spline_norm - 1.0) * half_range + 1e-9,
        "spline overshoot {spline_overshoot} exceeds its operator bound {} (centre {centre})",
        (spline_norm - 1.0) * half_range
    );
    assert!(
        spline_overshoot < 0.1 * (hi - lo),
        "spline overshoot {spline_overshoot} on a range of {}",
        hi - lo
    );
    assert!(
        lagrange_overshoot > hi - lo,
        "the control did not overshoot: {lagrange_overshoot}"
    );
}

#[test]
fn lebesgue_constant_grows_with_the_order_and_is_recorded() {
    let mut previous = 0.0;
    for order in [5usize, 9, 17, 33] {
        let gh = GaussHermite::new(order).expect("rule");
        assert!(
            gh.lebesgue_constant > previous,
            "order {order}: {}",
            gh.lebesgue_constant
        );
        previous = gh.lebesgue_constant;
    }
    assert!(
        previous > 1e3,
        "the Lebesgue constant at order 33 is {previous}"
    );
    let g9 = GaussHermite::new(9).expect("rule");
    assert!(
        g9.lebesgue_constant < 50.0,
        "order 9: {}",
        g9.lebesgue_constant
    );
}

#[test]
fn dual_loading_derivative_matches_finite_difference_at_zero_loading() {
    // At loading zero the computed marginal is exactly symmetric in the
    // loading, so its derivative is zero; the dual must reproduce that as
    // the node count grows and the grid starts moving with the loading.
    use super::scalar::Tangent;
    let gh = GaussHermite::new(9).expect("rule");
    for nodes in 1..=4 {
        let times: Vec<f64> = (0..nodes).map(|n| n as f64 * 0.7).collect();
        let exposures: Vec<f64> = (0..nodes).map(|n| if n == 0 { 0.0 } else { 0.7 }).collect();
        let counts: Vec<Vec<f64>> = (0..nodes)
            .map(|n| vec![if n % 2 == 1 { 1.0 } else { 0.0 }])
            .collect();
        let subj = subject(&times, &exposures, &counts);
        let value = |a: f64| -> f64 {
            let eta0 = vec![0.3; nodes];
            let inputs = SubjectInputs {
                nodes: &subj,
                eta0: &eta0,
                loadings: &[a],
                rates: &[1.2],
                time_scale: 1.0,
                gh: &gh,
                continuation_gap: 0.0,
                designs: None,
                log_normaliser: None,
            };
            subject_marginal(&inputs, false).expect("value").loglik
        };
        let eta0: Vec<Tangent<1>> = (0..nodes).map(|_| Tangent::seeded(0.3, [0.0])).collect();
        let inputs = SubjectInputs {
            nodes: &subj,
            eta0: &eta0,
            loadings: &[Tangent::seeded(0.0, [1.0])],
            rates: &[Tangent::seeded(1.2, [0.0])],
            time_scale: 1.0,
            gh: &gh,
            continuation_gap: 0.0,
            designs: None,
            log_normaliser: None,
        };
        let dual = subject_marginal(&inputs, false).expect("dual").loglik;
        let h = 1e-5;
        let fd = (value(h) - value(-h)) / (2.0 * h);
        emit(&format!(
            "nodes={nodes}: value {} dual {} d/da {} vs finite difference {fd}",
            value(0.0),
            dual.value,
            dual.grad[0]
        ));
        assert_eq!(
            dual.value,
            value(0.0),
            "dual value channel must match the plain value"
        );
        assert!(
            (dual.grad[0] - fd).abs() < 1e-6 * (1.0 + fd.abs()),
            "nodes={nodes}: d/da {} vs finite difference {fd}",
            dual.grad[0]
        );
    }
}

/// A tracing shell around the family that prints every engine call, so a
/// stalled inner solve can be read as the sequence of points it evaluated.
#[derive(Clone)]
struct Traced(EventHistoryFamily);

impl gam_model_api::families::custom_family::CustomFamily for Traced {
    fn evaluate(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<gam_model_api::families::custom_family::FamilyEvaluation, String> {
        let out = self.0.evaluate(block_states)?;
        emit(&format!(
            "[trace] evaluate latent={:?} beta={:?} ll={}",
            block_states[1].beta.as_slice().expect("slice"),
            block_states[0].beta.as_slice().expect("slice"),
            out.log_likelihood
        ));
        Ok(out)
    }
    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        let out = self.0.log_likelihood_only(block_states);
        emit(&format!(
            "[trace] value latent={:?} beta={:?} -> {:?}",
            block_states[1].beta.as_slice().expect("slice"),
            block_states[0].beta.as_slice().expect("slice"),
            out
        ));
        out
    }
    fn classical_deviance(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<f64>, String> {
        self.0.classical_deviance(block_states)
    }
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }
    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }
    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }
    fn levenberg_on_ill_conditioning(&self) -> bool {
        true
    }
    fn output_channel_assignment(
        &self,
        specs: &[gam_problem::ParameterBlockSpec],
    ) -> Option<Vec<usize>> {
        self.0.output_channel_assignment(specs)
    }
    fn block_coefficient_coordinate(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        block_spec: &gam_problem::ParameterBlockSpec,
    ) -> gam_problem::CoefficientCoordinate {
        self.0
            .block_coefficient_coordinate(block_states, block_index, block_spec)
    }
    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let out = self.0.exact_newton_joint_hessian(block_states)?;
        emit(&format!(
            "[trace] joint hessian at latent={:?}: {:?}",
            block_states[1].beta.as_slice().expect("slice"),
            out.as_ref().map(|h| h.diag().to_vec())
        ));
        Ok(out)
    }
    fn exact_newton_joint_loglik_gradient(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array1<f64>>, String> {
        let out = self.0.exact_newton_joint_loglik_gradient(block_states)?;
        emit(&format!(
            "[trace] joint gradient {:?}",
            out.as_ref().map(|g| g.to_vec())
        ));
        Ok(out)
    }
    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[gam_problem::ParameterBlockSpec],
    ) -> Result<Option<gam_model_api::families::custom_family::ExactNewtonJointGradientEvaluation>, String> {
        self.0
            .exact_newton_joint_gradient_evaluation(block_states, specs)
    }
    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.0
            .exact_newton_joint_hessian_directional_derivative(block_states, d_beta_flat)
    }
    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        u: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.0
            .exact_newton_joint_hessiansecond_directional_derivative(block_states, u, v)
    }
}

#[test]
fn traced_fixed_lambda_inner_solve_on_the_null_cohort() {
    use super::family::{latent_block_spec, mark_block_spec};
    use gam_terms::smooth::build_term_collection_design;
    let mut cohort = simulate_cohort(80, 6.0, -0.8, 0.5, 0.0, 0.4, 3);
    cohort.validate().expect("valid");
    let nodes = Arc::new(expand_nodes(&cohort, 9, 0).expect("nodes"));
    let design =
        build_term_collection_design(nodes.node_data.view(), &linear_spec()).expect("design");
    let dense = design
        .design
        .try_to_dense_arc("test design")
        .expect("dense");
    let family = EventHistoryFamily::new(
        Arc::clone(&nodes),
        vec![dense],
        1,
        11,
        cohort.time_scale(),
        vec![None],
    )
    .expect("family");
    let specs = vec![
        mark_block_spec("event", &design),
        latent_block_spec(
            nodes.total_nodes,
            1,
            1,
            &RankStart::carried(Vec::new(), vec![1.0], vec![0.0], vec![0.0], vec![false]),
            family.rate_band(),
        )
        .expect("latent spec"),
    ];
    let options = gam_model_api::families::custom_family::BlockwiseFitOptions::default();
    let result = gam_custom_family::fit_custom_family_fixed_log_lambdas(
        &Traced(family),
        &specs,
        &options,
        None,
    );
    match result {
        Ok(fit) => println!("[trace] converged: latent={:?}", fit.block_states[1].beta),
        Err(error) => println!("[trace] error: {error}"),
    }
}

#[test]
fn gradient_and_hessian_match_central_differences_at_tiny_gaps() {
    // Gaps of a few thousandths of the time scale drive `1 − φ²` to 1e-3;
    // the innovation-coordinate gap algebra must stay bounded there. At such
    // gaps the backward step is pure interpolation with no kernel smoothing,
    // so the agreement is limited by the Lagrange interpolation error of the
    // exponential likelihood factors on a 31-node grid, about 1e-4 relative.
    let gh = GaussHermite::new(31).expect("rule");
    let nodes = subject(
        &[0.0, 0.004, 0.011, 0.016],
        &[0.3, 0.0, 0.5, 0.2],
        &[
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ],
    );
    let mut theta = vec![0.2, -0.1, 0.4, 0.0, -0.3, 0.5, 0.1, 0.2];
    theta.extend([0.8, -0.4, 0.3, 0.6]);
    theta.extend([0.82, 1.65]);
    let p = theta.len();
    let base = evaluate_at(&nodes, &gh, &theta, true);
    let h = 1e-4;
    let mut mismatches = Vec::new();
    for i in 0..p {
        let mut plus = theta.clone();
        plus[i] += h;
        let mut minus = theta.clone();
        minus[i] -= h;
        let fp = evaluate_at(&nodes, &gh, &plus, true);
        let fm = evaluate_at(&nodes, &gh, &minus, true);
        let fd = (fp.loglik - fm.loglik) / (2.0 * h);
        if (base.gradient[i] - fd).abs() >= 1e-4 * (1.0 + fd.abs()) {
            mismatches.push(format!("gradient[{i}] = {} vs {fd}", base.gradient[i]));
        }
        for j in 0..p {
            let fd_h = (fp.gradient[j] - fm.gradient[j]) / (2.0 * h);
            let value = base.hessian[i * p + j];
            if (value - fd_h).abs() >= 1e-3 * (1.0 + fd_h.abs()) {
                mismatches.push(format!(
                    "hessian[{i},{j}] = {value} vs {fd_h} (rel {:e})",
                    (value - fd_h).abs() / (1.0 + fd_h.abs())
                ));
            }
        }
    }
    for line in &mismatches {
        println!("{line}");
    }
    assert!(
        mismatches.is_empty(),
        "{} derivative entries disagree",
        mismatches.len()
    );
}

#[test]
fn traced_fixed_lambda_inner_solve_on_the_loaded_cohort_reports_its_cost() {
    install_test_logger();
    use super::family::{latent_block_spec, mark_block_spec};
    use gam_terms::smooth::build_term_collection_design;
    let mut cohort = simulate_cohort(80, 6.0, -0.8, 0.5, 1.0, 0.4, 7);
    cohort.validate().expect("valid");
    let nodes = Arc::new(expand_nodes(&cohort, 9, 0).expect("nodes"));
    let design =
        build_term_collection_design(nodes.node_data.view(), &linear_spec()).expect("design");
    let dense = design
        .design
        .try_to_dense_arc("test design")
        .expect("dense");
    let family = EventHistoryFamily::new(
        Arc::clone(&nodes),
        vec![dense],
        1,
        11,
        cohort.time_scale(),
        vec![None],
    )
    .expect("family");
    let total_nodes: usize = nodes.subjects.iter().map(|s| s.len()).sum();
    emit(&format!(
        "[cost] subjects={} nodes={} mean_nodes={:.1}",
        nodes.subjects.len(),
        total_nodes,
        total_nodes as f64 / nodes.subjects.len() as f64
    ));
    let states = vec![
        ParameterBlockState {
            beta: array![-0.8, 0.5],
            eta: Array1::zeros(nodes.total_nodes),
        },
        ParameterBlockState {
            beta: array![0.7, 1.0],
            eta: Array1::zeros(nodes.total_nodes),
        },
    ];
    let mut states = states;
    let design_dense = design.design.try_to_dense_arc("d").expect("dense");
    states[0].eta = design_dense.dot(&states[0].beta);
    let clock = std::time::Instant::now();
    family.log_likelihood(&states).expect("value");
    emit(&format!(
        "[cost] value-only {:.3}s",
        clock.elapsed().as_secs_f64()
    ));
    let clock = std::time::Instant::now();
    family.joint_evaluation(&states).expect("joint");
    emit(&format!(
        "[cost] joint (value+gradient+hessian) {:.3}s",
        clock.elapsed().as_secs_f64()
    ));
    let u = array![0.1, 0.2, 0.3, 0.4];
    let clock = std::time::Instant::now();
    family
        .directional_hessian(&states, &u)
        .expect("directional");
    emit(&format!(
        "[cost] directional hessian {:.3}s",
        clock.elapsed().as_secs_f64()
    ));
    let clock = std::time::Instant::now();
    family
        .second_directional_hessian(&states, &u, &u)
        .expect("second directional");
    emit(&format!(
        "[cost] second directional hessian {:.3}s",
        clock.elapsed().as_secs_f64()
    ));
    let specs = vec![
        mark_block_spec("event", &design),
        latent_block_spec(
            nodes.total_nodes,
            1,
            1,
            &RankStart::carried(Vec::new(), vec![1.0], vec![0.0], vec![0.0], vec![false]),
            family.rate_band(),
        )
        .expect("latent spec"),
    ];
    let options = gam_model_api::families::custom_family::BlockwiseFitOptions::default();
    let clock = std::time::Instant::now();
    let traced = Traced(family);
    let result =
        gam_custom_family::fit_custom_family_fixed_log_lambdas(&traced, &specs, &options, None);
    match result {
        Ok(fit) => {
            emit(&format!(
                "[cost] fixed-lambda inner solve {:.1}s cycles={} latent={:?}",
                clock.elapsed().as_secs_f64(),
                fit.inner_cycles,
                fit.block_states[1].beta
            ));
            // Louis Hessian against a finite difference of the exact gradient
            // at the final state, and the exact gradient against a finite
            // difference of the value.
            let family = &traced.0;
            let at = |beta: &Array1<f64>, latent: &Array1<f64>| -> Vec<ParameterBlockState> {
                let mut states = fit.block_states.clone();
                states[0].beta = beta.clone();
                states[0].eta = design_dense.dot(beta);
                states[1].beta = latent.clone();
                states
            };
            let beta = fit.block_states[0].beta.clone();
            let latent = fit.block_states[1].beta.clone();
            let base = family.joint_evaluation(&at(&beta, &latent)).expect("joint");
            let p = beta.len() + latent.len();
            let h = 1e-5;
            let split = |v: &Array1<f64>| -> (Array1<f64>, Array1<f64>) {
                (
                    v.slice(ndarray::s![0..beta.len()]).to_owned(),
                    v.slice(ndarray::s![beta.len()..]).to_owned(),
                )
            };
            let mut full = Array1::<f64>::zeros(p);
            full.slice_mut(ndarray::s![0..beta.len()]).assign(&beta);
            full.slice_mut(ndarray::s![beta.len()..]).assign(&latent);
            for i in 0..p {
                let mut plus = full.clone();
                plus[i] += h;
                let mut minus = full.clone();
                minus[i] -= h;
                let (bp, lp) = split(&plus);
                let (bm, lm) = split(&minus);
                let gp = family
                    .joint_evaluation(&at(&bp, &lp))
                    .expect("joint")
                    .gradient
                    .clone();
                let gm = family
                    .joint_evaluation(&at(&bm, &lm))
                    .expect("joint")
                    .gradient
                    .clone();
                let vp = family.log_likelihood(&at(&bp, &lp)).expect("value");
                let vm = family.log_likelihood(&at(&bm, &lm)).expect("value");
                let fd_gradient = (vp - vm) / (2.0 * h);
                let fd_row: Vec<f64> = (0..p).map(|j| -(gp[j] - gm[j]) / (2.0 * h)).collect();
                let louis_row: Vec<f64> = (0..p).map(|j| base.hessian[[i, j]]).collect();
                emit(&format!(
                    "[final] coefficient {i}: gradient exact {:.9e} fd {:.9e}; -hessian row louis {:?} fd {:?}",
                    base.gradient[i], fd_gradient, louis_row, fd_row
                ));
            }
        }
        Err(error) => emit(&format!(
            "[cost] error after {:.1}s: {error}",
            clock.elapsed().as_secs_f64()
        )),
    }
}

/// Simulate a single-mark cohort whose log-intensity is `β₀ + b(t) g` for a
/// subject-level standard-normal score `g` and no latent state, by thinning
/// under the bound `exp(β₀ + slope_bound · |g|)` with `slope_bound ≥ max |b|`.
fn simulate_score_cohort(
    subjects: usize,
    follow_up: f64,
    intercept: f64,
    slope: &dyn Fn(f64) -> f64,
    slope_bound: f64,
    seed: u64,
) -> EventHistoryCohort {
    let mut rng = Rng(seed);
    let mut covariates = Array2::<f64>::zeros((subjects, 1));
    let mut histories = Vec::with_capacity(subjects);
    for s in 0..subjects {
        let g = rng.normal();
        covariates[[s, 0]] = g;
        let bound = (intercept + slope_bound * g.abs()).exp();
        let mut events = Vec::new();
        let mut t = 0.0;
        loop {
            t -= bound.recip() * rng.uniform().max(1e-300).ln();
            if t >= follow_up {
                break;
            }
            let intensity = (intercept + slope(t) * g).exp();
            assert!(intensity <= bound, "thinning bound violated");
            if rng.uniform() * bound < intensity {
                events.push(Event { time: t, mark: 0 });
            }
        }
        histories.push(SubjectHistory {
            id: format!("s{s}"),
            entry: 0.0,
            exit: follow_up,
            events,
            segments: vec![CovariateSegment { start: 0.0, row: s }],
        });
    }
    EventHistoryCohort {
        mark_names: vec!["event".to_string()],
        mark_kinds: vec![MarkKind::Recurrent],
        covariate_names: vec!["g".to_string()],
        covariate_levels: vec![Vec::new()],
        covariates,
        subjects: histories,
    }
}

/// The fitted score slope `b(t) = η(g = 1, t) − η(g = 0, t)` of mark 0, for
/// a fit whose node columns are `[g, time]`.
fn fitted_score_slope(fit: &EventHistoryFit, times: &[f64]) -> Vec<f64> {
    let mut rows = Array2::<f64>::zeros((2 * times.len(), 2));
    for (i, &t) in times.iter().enumerate() {
        rows[[2 * i, 0]] = 1.0;
        rows[[2 * i, 1]] = t;
        rows[[2 * i + 1, 1]] = t;
    }
    let design =
        build_term_collection_design(rows.view(), &fit.frozen_specs[0]).expect("prediction design");
    let dense = design
        .design
        .try_to_dense_arc("score slope design")
        .expect("dense design");
    let beta = fit.mark_coefficients(0);
    let eta = |r: usize| -> f64 {
        design.affine_offset[r]
            + dense
                .row(r)
                .iter()
                .zip(beta.iter())
                .map(|(x, b)| x * b)
                .sum::<f64>()
    };
    (0..times.len())
        .map(|i| eta(2 * i) - eta(2 * i + 1))
        .collect()
}

/// An observed subject-level score enters the intensity as one penalised
/// slope surface `b(t) · g`: a continuous by-smooth keeps its constant, so
/// its wiggliness ridge decides how much the score's effect bends with time
/// and its null-space ridge decides whether the effect exists at all, both
/// selected by REML. A declining effect is
/// recovered as a decline; a score carrying nothing collapses to zero.
#[test]
fn an_observed_score_enters_as_a_penalised_slope_surface() {
    install_test_logger();
    let times = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
    let formula = "s(time, by=g)";
    let truth = |t: f64| 1.0 - 0.15 * t;
    let mut cohort = simulate_score_cohort(300, 6.0, -0.5, &truth, 1.0, 19);
    let events: usize = cohort.subjects.iter().map(|s| s.events.len()).sum();
    let started = std::time::Instant::now();
    let fit =
        fit_event_history_formulas(&mut cohort, &[formula], BlockwiseFitOptions::default(), None)
            .expect("fit with a declining score effect");
    let slope = fitted_score_slope(&fit, &times);
    emit(&format!(
        "[score-slope] declining arm: {events} events, {:.1}s, outer_iterations={} log_lambdas={:?}",
        started.elapsed().as_secs_f64(),
        fit.fit.outer_iterations,
        fit.fit.log_lambdas
    ));
    for (t, b) in times.iter().zip(slope.iter()) {
        emit(&format!(
            "[score-slope]   t={t} fitted={b:.3} truth={:.3}",
            truth(*t)
        ));
    }
    // The whole fitted surface, densely, for plotting.
    let dense: Vec<f64> = (0..=60).map(|i| 0.1 * i as f64).collect();
    for (t, b) in dense.iter().zip(fitted_score_slope(&fit, &dense).iter()) {
        emit(&format!(
            "[score-slope-curve] arm=declining t={t:.2} fitted={b:.5} truth={:.5}",
            truth(*t)
        ));
    }
    assert!(
        slope[0] - slope[5] > 0.35,
        "the score's effect declines by 0.75 over the follow-up; the fit shows {} → {}",
        slope[0],
        slope[5]
    );
    for (t, b) in times.iter().zip(slope.iter()) {
        assert!(
            (b - truth(*t)).abs() < 0.3,
            "slope at t={t}: fitted {b}, truth {}",
            truth(*t)
        );
    }

    // Control: the score carries nothing. The same surface must collapse.
    let mut null = simulate_score_cohort(300, 6.0, -0.5, &|_| 0.0, 0.0, 23);
    let null_events: usize = null.subjects.iter().map(|s| s.events.len()).sum();
    let started = std::time::Instant::now();
    let null_fit =
        fit_event_history_formulas(&mut null, &[formula], BlockwiseFitOptions::default(), None)
            .expect("fit with an uninformative score");
    let null_slope = fitted_score_slope(&null_fit, &times);
    emit(&format!(
        "[score-slope] null arm: {null_events} events, {:.1}s, outer_iterations={} log_lambdas={:?}",
        started.elapsed().as_secs_f64(),
        null_fit.fit.outer_iterations,
        null_fit.fit.log_lambdas
    ));
    for (t, b) in times.iter().zip(null_slope.iter()) {
        emit(&format!("[score-slope]   t={t} fitted={b:.3} truth=0.000"));
    }
    for (t, b) in dense
        .iter()
        .zip(fitted_score_slope(&null_fit, &dense).iter())
    {
        emit(&format!(
            "[score-slope-curve] arm=null t={t:.2} fitted={b:.5} truth=0.00000"
        ));
    }
    let amplitude = null_slope.iter().fold(0.0f64, |m, b| m.max(b.abs()));
    assert!(
        amplitude < 0.15,
        "an uninformative score should collapse to zero; the fitted surface reaches {amplitude}"
    );
}

/// The information hierarchy is three conditionings of one model. With a
/// standard-normal subject-level score `x` and a latent atom, the population
/// tier is the zero-count filter from the stationary prior at the population
/// score; the score-only tier is the same filter at the subject's own score;
/// the history tier continues from the subject's filtered state. A positive
/// score effect orders the first two by the score's sign, and a history
/// richer (poorer) in events than its score alone predicts raises (lowers)
/// the third against the second. No weight between the tiers is chosen.
#[test]
fn forecast_tiers_population_score_and_history_are_one_model_conditioned_on_more() {
    install_test_logger();
    // Forty subjects: the tiers' ordering is a statement about one fitted
    // model, and the fit is now two solves per rank plus the certificate.
    let mut cohort = simulate_cohort(40, 6.0, -0.8, 0.5, 1.0, 0.4, 5);
    let mut spec = EventHistorySpec::new(vec![linear_spec()]);
    spec.gauss_hermite_order = 11;
    let started = std::time::Instant::now();
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
    assert_rank_stop_explained(&fit, &spec);
    let beta = fit.mark_coefficients(0);
    emit(&format!(
        "[tiers] {:.1}s beta={:?} loading={} rate={}",
        started.elapsed().as_secs_f64(),
        beta.to_vec(),
        fit.loadings[[0, 0]],
        fit.rates[0]
    ));
    assert!(
        beta[1] > 0.0,
        "the score effect was simulated positive; fitted {}",
        beta[1]
    );
    let horizons = [7.0, 8.0];
    let at = |start: f64, score: f64| -> Vec<FutureSegment> {
        vec![FutureSegment {
            start,
            covariates: vec![score],
        }]
    };
    let population = population_forecast(
        &fit,
        &cohort,
        &PopulationForecastRequest {
            start: 6.0,
            horizons: &horizons,
            future: &at(6.0, 0.0),
            stratum: 0,
        },
    )
    .expect("population forecast");
    assert!(
        (population.survival[1] - 1.0).abs() < 1e-12,
        "no terminal marks"
    );
    // The claim is per subject, so it is checked on a subset: the widest
    // scores, which order the score-only tier against the population, and a
    // band of near-population scores, where the histories do the ordering.
    // Every tier costs two filtered windows, and the fit above already ran
    // the certificate ladder twice.
    let mut chosen: Vec<usize> = (0..cohort.subjects.len()).collect();
    chosen.sort_by(|a, b| {
        cohort.covariates[[*a, 0]]
            .abs()
            .total_cmp(&cohort.covariates[[*b, 0]].abs())
    });
    let band: Vec<usize> = chosen.iter().take(8).copied().collect();
    let extremes: Vec<usize> = chosen.iter().rev().take(4).copied().collect();
    let examined: Vec<usize> = band.iter().chain(extremes.iter()).copied().collect();
    let tiers: Vec<(f64, f64, f64, usize)> = examined
        .iter()
        .map(|&i| {
            let subject = &cohort.subjects[i];
            let score = cohort.covariates[[i, 0]];
            let alone = population_forecast(
                &fit,
                &cohort,
                &PopulationForecastRequest {
                    start: subject.exit,
                    horizons: &horizons,
                    future: &at(subject.exit, score),
                    stratum: 0,
                },
            )
            .expect("score-only forecast");
            let with_history = forecast(
                &fit,
                &cohort,
                &ForecastRequest {
                    history: subject,
                    horizons: &horizons,
                    future: &[],
                    stratum: 0,
                },
            )
            .expect("history forecast");
            (
                score,
                alone.expected_counts[[1, 0]],
                with_history.expected_counts[[1, 0]],
                subject.events.len(),
            )
        })
        .collect();
    let population_count = population.expected_counts[[1, 0]];
    emit(&format!(
        "[tiers] population expected count by t=8: {population_count:.4}"
    ));
    for tier in tiers.iter() {
        let (score, alone) = (tier.0, tier.1);
        assert!(
            (score > 0.0) == (alone > population_count) || score == 0.0,
            "score {score}: score-only tier {alone} against population {population_count}"
        );
    }
    // Within a band of near-population scores, the subject richest in events
    // sits above its score-only tier and the poorest below it.
    let band: Vec<&(f64, f64, f64, usize)> = tiers.iter().filter(|t| t.0.abs() < 0.5).collect();
    assert!(
        band.len() >= 4,
        "too few near-population scores to compare histories"
    );
    let richest = band.iter().max_by_key(|t| t.3).expect("richest");
    let poorest = band.iter().min_by_key(|t| t.3).expect("poorest");
    for (label, t) in [("richest", richest), ("poorest", poorest)] {
        emit(&format!(
            "[tiers] {label}: score={:.3} events={} score-only={:.4} with-history={:.4}",
            t.0, t.3, t.1, t.2
        ));
    }
    assert!(
        richest.3 > poorest.3,
        "the band must contain unequal histories"
    );
    // The history tier is compared against another history tier, not against
    // the score-only one. Observing a history does two things at once: it
    // moves the latent mean, and it narrows the latent variance. The
    // narrowing alone lowers `E[exp(a z)]` for any subject — the intensity
    // is convex in the state — so a rich history can carry more risk than
    // its score alone implied and still forecast below the score-only tier.
    // Between two subjects of the same score, both narrowed, what remains is
    // the information their histories carry.
    assert!(
        richest.2 > poorest.2,
        "within one score band, the history rich in events ({} events, {:.4}) must forecast above the poor one ({} events, {:.4})",
        richest.3,
        richest.2,
        poorest.3,
        poorest.2
    );
}

/// Intercept-only fit of a cohort: the maximum-likelihood rate of each mark
/// is its event count over its exposure, so every forecast has a closed
/// form to compare against.
fn intercept_only_spec() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: Vec::new(),
    }
}

/// A competing-risks cohort: two terminal marks and one recurrent mark,
/// constant hazards, no latent state in the simulation.
fn competing_risks_cohort(seed: u64) -> EventHistoryCohort {
    simulate_marked_cohort(
        120,
        4.0,
        &[-1.4, -2.0, -0.6],
        0.0,
        &[0.0, 0.0, 0.0],
        1.0,
        &[MarkKind::Terminal, MarkKind::Terminal, MarkKind::Recurrent],
        seed,
    )
}

#[test]
fn terminal_forecasts_match_the_constant_hazard_solution() {
    install_test_logger();
    // The identity this asserts is the closed-form maximiser of the
    // rank-zero model, so the cohort has to be one the evidence does not buy
    // a latent direction on: the rank is the evidence's verdict, not a
    // setting a test can pin, and this fixture carries enough subjects for
    // the verdict to be the null.
    let mut cohort = competing_risks_cohort(64);
    let spec = EventHistorySpec::new(vec![intercept_only_spec()]);
    let fit = fit_event_history(&mut cohort, &spec).expect("intercept-only fit");
    assert!(fit.rank_path.iter().all(|step| step.proposed_rate.is_finite()
        && step.proposed_rate >= 0.0));
    // The maximum-likelihood rates: events over exposure, per mark.
    let exposure: f64 = cohort.subjects.iter().map(|s| s.exit - s.entry).sum();
    let counts: Vec<f64> = (0..3)
        .map(|d| {
            cohort
                .subjects
                .iter()
                .flat_map(|s| s.events.iter())
                .filter(|e| e.mark == d)
                .count() as f64
        })
        .collect();
    let rates: Vec<f64> = counts.iter().map(|c| c / exposure).collect();
    for d in 0..3 {
        let fitted = fit.mark_coefficients(d)[0].exp();
        // The agreement is limited by the inner solve's own convergence
        // tolerance, not by the model: the closed form IS the maximiser.
        assert!(
            (fitted - rates[d]).abs() < 1e-4 * rates[d],
            "mark {d}: fitted rate {fitted} vs closed form {}",
            rates[d]
        );
    }
    let total_terminal = rates[0] + rates[1];
    // A censored subject: the forecast is the exponential competing-risks
    // solution, chronologically integrated.
    let censored = cohort
        .subjects
        .iter()
        .find(|s| s.terminal_event(&cohort.mark_kinds).is_none())
        .expect("a censored subject");
    let offsets = [0.5, 1.0, 2.0, 4.0];
    let horizons: Vec<f64> = offsets.iter().map(|h| censored.exit + h).collect();
    let f = forecast(
        &fit,
        &cohort,
        &ForecastRequest {
            history: censored,
            horizons: &horizons,
            future: &[],
            stratum: 0,
        },
    )
    .expect("forecast");
    for (i, &h) in offsets.iter().enumerate() {
        let survival = (-total_terminal * h).exp();
        assert!(
            (f.survival[i] - survival).abs() < 1e-4,
            "survival at +{h}: {} vs {survival}",
            f.survival[i]
        );
        for d in 0..2 {
            let incidence = rates[d] / total_terminal * (1.0 - survival);
            assert!(
                (f.expected_counts[[i, d]] - incidence).abs() < 1e-4,
                "cumulative incidence of mark {d} at +{h}: {} vs {incidence}",
                f.expected_counts[[i, d]]
            );
        }
        // Recurrent events before termination: λ₂ ∫₀ʰ S = λ₂ (1 − S)/Λ.
        let recurrent = rates[2] / total_terminal * (1.0 - survival);
        assert!(
            (f.expected_counts[[i, 2]] - recurrent).abs() < 1e-4,
            "expected recurrent count at +{h}: {} vs {recurrent}",
            f.expected_counts[[i, 2]]
        );
    }
    // A subject who died has no future.
    let dead = cohort
        .subjects
        .iter()
        .find(|s| s.terminal_event(&cohort.mark_kinds).is_some())
        .expect("a subject with a terminal event");
    let gone = forecast(
        &fit,
        &cohort,
        &ForecastRequest {
            history: dead,
            horizons: &[dead.exit + 1.0],
            future: &[],
            stratum: 0,
        },
    )
    .expect("forecast of an absorbed subject");
    assert_eq!(gone.survival, vec![0.0]);
    assert!(gone.expected_counts.iter().all(|c| *c == 0.0));
    // The population tier at the same rates is the same solution.
    let population = population_forecast(
        &fit,
        &cohort,
        &PopulationForecastRequest {
            start: 1.0,
            horizons: &[2.0, 3.0],
            future: &[FutureSegment {
                start: 1.0,
                covariates: vec![0.0],
            }],
            stratum: 0,
        },
    )
    .expect("population forecast");
    assert!((population.survival[1] - (-2.0 * total_terminal).exp()).abs() < 1e-4);
    // Rosenblatt: with constant hazards the PIT of an event at `t` after the
    // previous event at `s` is `1 − exp(−Λ_all (t − s))` with the total rate.
    let total_rate: f64 = rates.iter().sum();
    let subject = cohort
        .subjects
        .iter()
        .max_by_key(|s| s.events.len())
        .expect("subject");
    let pits = predictive_pit(&fit, &cohort, subject, 0).expect("pit");
    let ended_by_event = subject.exit == subject.events.last().map_or(f64::NAN, |e| e.time);
    assert_eq!(
        pits.len(),
        subject.events.len() + usize::from(!ended_by_event),
        "one spell per event, plus the censored tail unless an event ended the follow-up"
    );
    let mut previous = subject.entry;
    for (event, pit) in subject.events.iter().zip(pits.iter()) {
        let expected = 1.0 - (-total_rate * (event.time - previous)).exp();
        assert!(
            (pit.pit - expected).abs() < 1e-4,
            "PIT at {}: {} vs {expected}",
            event.time,
            pit.pit
        );
        assert!(pit.observed);
        assert_eq!(pit.marks, vec![event.mark]);
        let probability_sum: f64 = pit.mark_probabilities.iter().sum();
        assert!((probability_sum - 1.0).abs() < 1e-12);
        for d in 0..3 {
            // The probabilities are the fitted intensity ratios exactly, so
            // against the CLOSED-FORM ratios they inherit the same gap the
            // fitted rates have: the assertion above bounds that at 1e-4
            // relative, and this one cannot be tighter than the quantity it
            // is built from.
            assert!(
                (pit.mark_probabilities[d] - rates[d] / total_rate).abs() < 1e-4,
                "mark {d}: probability {} vs closed-form ratio {}",
                pit.mark_probabilities[d],
                rates[d] / total_rate
            );
        }
        previous = event.time;
    }
}

/// The intercept-only competing-risks fit and its fitted rates. The model is
/// rank zero with constant hazards, so every forecast from it has a closed form
/// in the fitted rates, to roundoff, at any horizon.
fn constant_hazard_fit() -> (EventHistoryCohort, EventHistoryFit, Vec<f64>) {
    let mut cohort = competing_risks_cohort(64);
    let spec = EventHistorySpec::new(vec![intercept_only_spec()]);
    let fit = fit_event_history(&mut cohort, &spec).expect("intercept-only fit");
    assert_eq!(fit.rank(), 0, "the constant-hazard fixture must be rank zero: {:?}", fit.rank_path);
    let rates = (0..3).map(|d| fit.mark_coefficients(d)[0].exp()).collect();
    (cohort, fit, rates)
}

/// A population forecast of the constant-hazard fixture from time zero.
fn constant_hazard_population(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    horizons: &[f64],
) -> super::forecast::Forecast {
    population_forecast(
        fit,
        cohort,
        &PopulationForecastRequest {
            start: 0.0,
            horizons,
            future: &[FutureSegment {
                start: 0.0,
                covariates: vec![0.0],
            }],
            stratum: 0,
        },
    )
    .expect("population forecast")
}

/// The rounding a closed-form oracle carries: `ε · |value| · depth`, with
/// `depth` the longest chain of floating-point operations that forms it
/// (Higham, *Accuracy and Stability*, ch. 3). The forecast route's rounding
/// is already in its reported error, as every accepted cell's roundoff floor.
fn oracle_rounding(value: f64, depth: usize) -> f64 {
    f64::EPSILON * value.abs() * depth as f64
}

#[test]
fn a_long_forecast_window_keeps_survival_and_terminal_incidence_summing_to_one() {
    install_test_logger();
    let (cohort, fit, rates) = constant_hazard_fit();
    let total_terminal = rates[0] + rates[1];
    // The window's integrated terminal hazard is one hundred. The survival
    // is e^-100 and the terminal incidences carry the rest of the mass.
    let horizon = 100.0 / total_terminal;
    let f = constant_hazard_population(&fit, &cohort, &[horizon]);
    let decrement = -(-100.0_f64).exp_m1();
    let mass = f.survival[0] + f.expected_counts[[0, 0]] + f.expected_counts[[0, 1]];
    emit(&format!(
        "[2963 mass] horizon {horizon:.6} survival {:e} (closed {:e}) terminal {:.12} {:.12} recurrent {:.12} (closed {:.12} {:.12} {:.12}) S+F-1 {:e}",
        f.survival[0],
        (-100.0_f64).exp(),
        f.expected_counts[[0, 0]],
        f.expected_counts[[0, 1]],
        f.expected_counts[[0, 2]],
        rates[0] / total_terminal * decrement,
        rates[1] / total_terminal * decrement,
        rates[2] / total_terminal * decrement,
        mass - 1.0
    ));
    // The identity is exact per cell to roundoff. The forecast's checked errors
    // carry every accepted cell's roundoff floor as well as its gaps, and the
    // test's own sum adds two operations.
    let bar = f.survival_error[0]
        + f.expected_count_errors[[0, 0]]
        + f.expected_count_errors[[0, 1]]
        + oracle_rounding(mass, 2);
    assert!(bar < 1.0, "the checked error {bar} does not resolve a probability");
    assert!(
        (mass - 1.0).abs() <= bar,
        "survival plus terminal incidence at integrated hazard 100 is {mass}, not one within {bar}"
    );
    for d in 0..3 {
        // The terminal incidences share the survival decrement, exact at any
        // mesh; the recurrent count is a quadrature, within its checked error.
        // The closed form is five operations deep (exp, add, expm1, div, mul).
        let closed = rates[d] / total_terminal * decrement;
        let bar = f.expected_count_errors[[0, d]] + oracle_rounding(closed, 5);
        assert!(closed > bar, "mark {d}: closed form {closed} is not above its bar {bar}");
        assert!(
            (f.expected_counts[[0, d]] - closed).abs() <= bar,
            "mark {d}: expected count {} vs closed form {closed}, bar {bar}",
            f.expected_counts[[0, d]]
        );
    }
}

#[test]
fn reporting_horizons_do_not_change_an_existing_forecast() {
    install_test_logger();
    let (cohort, fit, rates) = constant_hazard_fit();
    let total_terminal = rates[0] + rates[1];
    let two = [50.0 / total_terminal, 100.0 / total_terminal];
    let dense: Vec<f64> = (1..=100).map(|k| k as f64 / total_terminal).collect();
    let alone = constant_hazard_population(&fit, &cohort, &two[..1]);
    let sparse = constant_hazard_population(&fit, &cohort, &two);
    let full = constant_hazard_population(&fit, &cohort, &dense);
    for (i, k) in [(0usize, 49usize), (1, 99)] {
        emit(&format!(
            "[2963 horizons] integrated hazard {}: survival {:e} vs {:e}; counts {:?} vs {:?}",
            k + 1,
            sparse.survival[i],
            full.survival[k],
            sparse.expected_counts.row(i).to_vec(),
            full.expected_counts.row(k).to_vec()
        ));
        // Both requests end at the same horizon, so the window's mesh and the
        // integration to every shared horizon are one computation, bit for bit.
        assert_eq!(sparse.survival[i].to_bits(), full.survival[k].to_bits());
        for d in 0..3 {
            assert_eq!(
                sparse.expected_counts[[i, d]].to_bits(),
                full.expected_counts[[k, d]].to_bits(),
                "mark {d} at integrated hazard {}: {} with two horizons, {} with one hundred",
                k + 1,
                sparse.expected_counts[[i, d]],
                full.expected_counts[[k, d]]
            );
        }
    }
    // A window that ends at the shared horizon is a different integration,
    // and agrees within the two forecasts' checked errors.
    for d in 0..3 {
        let bar = alone.expected_count_errors[[0, d]] + full.expected_count_errors[[49, d]];
        assert!(full.expected_counts[[49, d]] > bar, "mark {d}: count below its bar {bar}");
        assert!(
            (alone.expected_counts[[0, d]] - full.expected_counts[[49, d]]).abs() <= bar,
            "mark {d} at integrated hazard 50: {} alone, {} among one hundred horizons",
            alone.expected_counts[[0, d]],
            full.expected_counts[[49, d]]
        );
    }
}

#[test]
fn constant_hazard_forecasts_are_exact_at_every_horizon() {
    install_test_logger();
    let (cohort, fit, rates) = constant_hazard_fit();
    let total_terminal = rates[0] + rates[1];
    let censored = cohort
        .subjects
        .iter()
        .find(|s| s.terminal_event(&cohort.mark_kinds).is_none())
        .expect("a censored subject");
    // Long windows are the mass test's. Here every compared value stays above
    // its bar: the magnitude floor a two-route comparison needs.
    let hazards = [0.5, 2.0, 10.0];
    let horizons: Vec<f64> = hazards.iter().map(|h| censored.exit + h / total_terminal).collect();
    let f = forecast(
        &fit,
        &cohort,
        &ForecastRequest {
            history: censored,
            horizons: &horizons,
            future: &[],
            stratum: 0,
        },
    )
    .expect("forecast");
    for (i, &h) in horizons.iter().enumerate() {
        let hazard = total_terminal * (h - censored.exit);
        let decrement = -(-hazard).exp_m1();
        emit(&format!(
            "[2963 exact] integrated hazard {hazard:.6}: survival {:e} (closed {:e}) counts {:?} (closed {:?})",
            f.survival[i],
            (-hazard).exp(),
            f.expected_counts.row(i).to_vec(),
            (0..3).map(|d| rates[d] / total_terminal * decrement).collect::<Vec<_>>()
        ));
        // The closed survival is five operations deep (exp, add, sub, mul, exp),
        // the closed counts seven (then expm1, div, mul).
        let survival = (-hazard).exp();
        let bar = f.survival_error[i] + oracle_rounding(survival, 5);
        assert!(survival > bar, "survival {survival} is not above its bar {bar}");
        assert!(
            (f.survival[i] - survival).abs() <= bar,
            "survival at integrated hazard {hazard}: {} vs {survival}, bar {bar}",
            f.survival[i]
        );
        for d in 0..3 {
            let closed = rates[d] / total_terminal * decrement;
            let bar = f.expected_count_errors[[i, d]] + oracle_rounding(closed, 7);
            assert!(closed > bar, "mark {d}: closed form {closed} is not above its bar {bar}");
            assert!(
                (f.expected_counts[[i, d]] - closed).abs() <= bar,
                "mark {d} at integrated hazard {hazard}: {} vs closed form {closed}, bar {bar}",
                f.expected_counts[[i, d]]
            );
        }
    }
}

#[test]
fn forecast_probabilities_are_coherent_under_a_latent_state() {
    install_test_logger();
    // The loadings are large enough that the evidence buys the direction:
    // the rank is no longer a setting, so a fixture that means to exercise a
    // latent state has to carry one the criterion will pay for.
    let mut cohort = simulate_marked_cohort(
        60,
        4.0,
        &[-1.2, -1.8, -0.5],
        0.4,
        &[1.4, 1.1, 1.2],
        0.5,
        &[MarkKind::Terminal, MarkKind::Once, MarkKind::Recurrent],
        41,
    );
    let mut spec = EventHistorySpec::new(vec![linear_spec()]);
    spec.gauss_hermite_order = 9;
    let started = std::time::Instant::now();
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
    emit(&format!(
        "[coherent] {:.1}s order={} refinement={} loadings={:?} rate={}",
        started.elapsed().as_secs_f64(),
        fit.quadrature.gauss_hermite_order,
        fit.quadrature.mesh_refinement,
        fit.loadings.iter().copied().collect::<Vec<_>>(),
        fit.rates.first().copied().unwrap_or(f64::NAN)
    ));
    assert!(
        fit.rank() >= 1,
        "a shared latent state was simulated but the evidence grew no atom: {:?}",
        fit.rank_path
    );
    let horizons_after = [0.5, 1.5, 3.0];
    for subject in cohort.subjects.iter().take(12) {
        if subject.terminal_event(&cohort.mark_kinds).is_some() {
            continue;
        }
        let horizons: Vec<f64> = horizons_after.iter().map(|h| subject.exit + h).collect();
        let f = forecast(
            &fit,
            &cohort,
            &ForecastRequest {
                history: subject,
                horizons: &horizons,
                future: &[],
                stratum: 0,
            },
        )
        .expect("forecast");
        let had_once = subject.events.iter().any(|e| e.mark == 1);
        let mut previous_survival = 1.0;
        let mut previous_counts = vec![0.0; 3];
        for i in 0..horizons.len() {
            let s = f.survival[i];
            assert!(
                (0.0..=1.0).contains(&s) && s <= previous_survival + 1e-12,
                "survival {s}"
            );
            // One terminal mark: its cumulative incidence is 1 − S.
            assert!(
                (f.expected_counts[[i, 0]] - (1.0 - s)).abs() < 1e-4,
                "subject {}: F_terminal {} vs 1 − S {}",
                subject.id,
                f.expected_counts[[i, 0]],
                1.0 - s
            );
            // A once-only mark: a probability, zero if it already fired.
            let once = f.expected_counts[[i, 1]];
            if had_once {
                assert_eq!(once, 0.0);
            } else {
                assert!(
                    (0.0..=1.0 + 1e-9).contains(&once),
                    "first-occurrence probability {once}"
                );
                assert!(
                    once + f.expected_counts[[i, 0]] <= 1.0 + 1e-6,
                    "once + terminal exceeds one"
                );
            }
            for d in 0..3 {
                assert!(f.expected_counts[[i, d]] >= previous_counts[d] - 1e-12);
                previous_counts[d] = f.expected_counts[[i, d]];
            }
            previous_survival = s;
        }
    }
    // A spell that ended in an event carries mark probabilities summing to
    // one over the marks the subject was at risk for. The spell that ends at
    // the exit carries none: no mark fired, so there is no mark to give a
    // probability to, and its own PIT is a censored draw rather than a value
    // the uniform law is asserted of.
    for subject in cohort.subjects.iter().take(12) {
        let spells = predictive_pit(&fit, &cohort, subject, 0).expect("pit");
        for spell in spells.iter() {
            assert!((0.0..=1.0).contains(&spell.pit));
            let sum: f64 = spell.mark_probabilities.iter().sum();
            if spell.observed {
                assert!(!spell.marks.is_empty());
                assert!((sum - 1.0).abs() < 1e-9, "mark probabilities sum to {sum}");
            } else {
                assert_eq!(spell.time, subject.exit);
                assert!(spell.marks.is_empty());
                assert_eq!(sum, 0.0, "a spell with no event has no mark probabilities");
            }
        }
        let observed = spells.iter().filter(|s| s.observed).count();
        assert_eq!(
            observed,
            subject
                .events
                .iter()
                .filter(|e| e.time > subject.entry)
                .count(),
            "one observed spell per event of the window"
        );
    }
}

#[test]
fn the_latent_block_carries_fixed_loading_priors_and_free_rates() {
    // Each atom's loadings carry the prior the evidence chose, held fixed,
    // and its log-rate is an unpenalised structural coordinate.
    let start = RankStart::carried(
        Vec::new(),
        vec![0.8, -0.3, 0.1, 0.5],
        vec![-0.2, 0.9],
        vec![1.5, -0.4],
        vec![false, false],
    );
    let band = (1e-6, 100.0);
    let latent = super::family::latent_block_spec(400, 2, 2, &start, band).expect("latent spec");
    let initial = latent.initial_beta.expect("initial");
    assert_eq!(
        initial.slice(ndarray::s![..4]).to_vec(),
        vec![0.8, -0.3, 0.1, 0.5]
    );
    // The rate coefficients are the chart coordinates of the dimensionless
    // rates: the chart round-trips them.
    for (k, log_rate) in [-0.2_f64, 0.9].iter().enumerate() {
        let rate = super::family::rate_from_chart(band, &initial[4 + k]);
        assert!(
            (rate - log_rate.exp()).abs() < 1e-12 * log_rate.exp(),
            "atom {k}: {rate} vs {}",
            log_rate.exp()
        );
    }
    assert_eq!(latent.penalties.len(), 2, "one loading prior per atom");
    for (k, penalty) in latent.penalties.iter().enumerate() {
        assert_eq!(penalty.fixed_log_lambda(), Some(start.log_lambdas[k]));
        assert_eq!(
            latent.nullspace_dims[k],
            6 - 2,
            "the rates lie in every prior's null space"
        );
    }
    // A rate held at a limit of the mesh's resolution has no coefficient:
    // the block narrows by one and the free rate keeps its slot.
    let held = RankStart::carried(
        Vec::new(),
        vec![0.8, -0.3, 0.1, 0.5],
        vec![-0.2, 0.9],
        vec![1.5, -0.4],
        vec![true, false],
    );
    let latent = super::family::latent_block_spec(400, 2, 2, &held, band).expect("latent spec");
    let initial = latent.initial_beta.expect("initial");
    assert_eq!(initial.len(), 5);
    let rate = super::family::rate_from_chart(band, &initial[4]);
    assert!((rate - 0.9_f64.exp()).abs() < 1e-12 * 0.9_f64.exp());
    assert_eq!(latent.nullspace_dims, vec![3, 3]);
}

#[test]
fn the_quartic_marginal_is_exact_and_the_empirical_bayes_prior_decides_by_the_mode() {
    // A negligible quartic reduces the marginal to the Gaussian one.
    let (log_integral, second, fourth) = quartic_moments(-3.0, 1e-9, 1.0);
    let sigma2 = 1.0 / 4.0;
    assert!((log_integral - 0.5 * (2.0 * std::f64::consts::PI * sigma2).ln()).abs() < 1e-8);
    assert!((second - sigma2).abs() < 1e-8, "E[t²] {second}");
    assert!(
        (fourth - 3.0 * sigma2 * sigma2).abs() < 1e-8,
        "E[t⁴] {fourth}"
    );
    // At the boundary `λ = μ` the integral is the pure quartic one,
    // `∫ exp(−t⁴/4) dt = Γ(1/4) / √2`, with `E[t²] = 2 Γ(3/4) / Γ(1/4)`.
    let gamma_quarter = 3.625_609_908_221_908_3_f64;
    let gamma_three_quarters = 1.225_416_702_465_177_6_f64;
    let (log_integral, second, _) = quartic_moments(2.0, 1.0, 2.0);
    assert!((log_integral - (gamma_quarter / 2.0_f64.sqrt()).ln()).abs() < 1e-8);
    assert!((second - 2.0 * gamma_three_quarters / gamma_quarter).abs() < 1e-8);

    // One direction: the prior's precision leaves the mode off zero exactly
    // when the standardised score `μ / √J` exceeds `Γ(1/4) / (2 Γ(3/4))`,
    // the value at which the marginal likelihood's slope in `λ` changes sign
    // at `λ = μ`. That threshold is a property of the quartic integral, not a
    // chosen level.
    let information: f64 = 100.0;
    let threshold = gamma_quarter / (2.0 * gamma_three_quarters);
    let quartic = |eigenvalue: f64| DirectionEvidence::Quartic {
        eigenvalue,
        information,
    };
    let below = empirical_bayes_ridge(&[quartic(0.9 * threshold * information.sqrt())]);
    let above = empirical_bayes_ridge(&[quartic(1.1 * threshold * information.sqrt())]);
    emit(&format!("[ridge] below {below:?} above {above:?}"));
    assert!(
        !below.accepted,
        "a score below the quartic threshold keeps the mode at zero"
    );
    assert!(
        above.accepted,
        "a score above the quartic threshold moves the mode off zero"
    );
    assert!(above.log_lambda.exp() < 1.1 * threshold * information.sqrt());
    assert!(above.gain > 0.0 && above.mode_scale > 0.0);
    // A strong direction lands near the Laplace-scale prior `λ = J / μ`.
    let strong = empirical_bayes_ridge(&[quartic(400.0)]);
    assert!(strong.accepted);
    assert!(
        (strong.log_lambda - (information / 400.0).ln()).abs() < 0.2,
        "{strong:?}"
    );
    assert!(
        strong.gain > 300.0,
        "the evidence of a 400-nat direction: {}",
        strong.gain
    );
    // No positive direction: no finite prior raises the evidence.
    let none = empirical_bayes_ridge(&[quartic(-5.0), quartic(-40.0)]);
    assert!(!none.accepted && none.log_lambda.is_infinite() && none.gain == 0.0);
    // Other directions charge their Occam factor: the same strong direction
    // beside three strongly negative ones is still accepted, and a marginal
    // one beside them is not.
    let beside = empirical_bayes_ridge(&[
        quartic(400.0),
        quartic(-300.0),
        quartic(-300.0),
        quartic(-300.0),
    ]);
    assert!(beside.accepted);
    let marginal = empirical_bayes_ridge(&[
        quartic(1.1 * threshold * information.sqrt()),
        quartic(-300.0),
        quartic(-300.0),
    ]);
    emit(&format!("[ridge] marginal beside negatives {marginal:?}"));
    assert!(!marginal.accepted);

    // An exact profile sampled from the quartic itself, with its slopes, is
    // integrated by the interpolant to the same marginal, moments and
    // prior as the closed-form quartic route.
    let (mu, j) = (400.0, information);
    let mode = (mu / j).sqrt();
    let step = mode / 8.0;
    let (mut points, mut values, mut slopes) = (vec![0.0], vec![0.0], vec![0.0]);
    let mut t = 0.0;
    loop {
        t += step;
        let value = 0.5 * mu * t * t - 0.25 * j * t * t * t * t;
        points.push(t);
        values.push(value);
        slopes.push(mu * t - j * t * t * t);
        if t > mode && value < mu * mu / (4.0 * j) - 40.0 {
            break;
        }
    }
    let exact = DirectionEvidence::Sampled(DirectionProfile {
        points,
        values,
        slopes,
    });
    let from_profile = empirical_bayes_ridge(&[exact]);
    emit(&format!(
        "[ridge] quartic {strong:?} profile {from_profile:?}"
    ));
    assert!(from_profile.accepted);
    assert!(
        (from_profile.log_lambda - strong.log_lambda).abs() < 1e-3,
        "{from_profile:?} vs {strong:?}"
    );
    assert!((from_profile.gain - strong.gain).abs() < 1e-2 * strong.gain);
    assert!((from_profile.mode_scale - strong.mode_scale).abs() < 1e-3);
}

#[test]
fn a_censored_tail_is_a_spell_and_the_distance_is_read_off_the_kaplan_meier_curve() {
    install_test_logger();
    // One terminal mark with a constant hazard, observed over a follow-up
    // short against it: most subjects are censored. Under the fitted model
    // the event PITs alone are uniform on `[0, 1 − e^{−λc}]`, not on
    // `[0, 1]`, so their Kolmogorov–Smirnov distance from the uniform law
    // sits near `e^{−λc}` however right the model is. The Kaplan–Meier
    // distance over event and censored spells has no such floor.
    let mut cohort = simulate_marked_cohort(
        400,
        1.0,
        &[-2.3],
        0.0,
        &[0.0],
        1.0,
        &[MarkKind::Terminal],
        77,
    );
    let spec = EventHistorySpec::new(vec![intercept_only_spec()]);
    let fit = fit_event_history(&mut cohort, &spec).expect("intercept-only fit");
    let mut spells: Vec<SpellPit> = Vec::new();
    for subject in &cohort.subjects {
        let pits = predictive_pit(&fit, &cohort, subject, 0).expect("pit");
        assert_eq!(
            pits.len(),
            1,
            "one spell per subject: its death, or its censored tail"
        );
        let spell = &pits[0];
        assert_eq!(
            spell.observed,
            subject.terminal_event(&cohort.mark_kinds).is_some()
        );
        assert_eq!(spell.time, subject.exit);
        spells.extend(pits);
    }
    let events: Vec<f64> = spells
        .iter()
        .filter(|s| s.observed)
        .map(|s| s.pit)
        .collect();
    let censored = spells.iter().filter(|s| !s.observed).count();
    assert!(
        censored > 300 && events.len() > 20,
        "{censored} censored subjects and {} events",
        events.len()
    );
    let event_only = kolmogorov_smirnov_uniform(&events).expect("events");
    let overall = pit_uniform_distance(&spells).expect("spells");
    let rate = fit.mark_coefficients(0)[0].exp();
    let floor = (-rate).exp();
    emit(&format!(
        "[pit] rate {rate:.4}: event-only KS {event_only:.3} (floor e^{{−λc}} = {floor:.3}), Kaplan–Meier distance {overall:.3}"
    ));
    assert!(
        event_only > floor - 0.1,
        "the event-only distance {event_only} must sit at its censoring floor {floor}"
    );
    assert!(
        overall < 0.1,
        "the Kaplan–Meier distance {overall} of a correctly specified model must be at sampling size"
    );
}

/// Kolmogorov–Smirnov distance of an uncensored PIT sample from the uniform
/// law, or `None` for an empty sample: the classical comparator that
/// `pit_uniform_distance` must reduce to when nothing is censored.
fn kolmogorov_smirnov_uniform(pits: &[f64]) -> Option<f64> {
    if pits.is_empty() {
        return None;
    }
    let mut sorted = pits.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len() as f64;
    let mut distance = 0.0_f64;
    for (i, &u) in sorted.iter().enumerate() {
        let lower = i as f64 / n;
        let upper = (i + 1) as f64 / n;
        distance = distance.max((u - lower).abs()).max((upper - u).abs());
    }
    Some(distance)
}

#[test]
fn the_pit_distance_is_the_kaplan_meier_gap_and_reduces_to_kolmogorov_smirnov_without_censoring() {
    let spell = |pit: f64, observed: bool| SpellPit {
        time: 0.0,
        observed,
        pit,
        marks: Vec::new(),
        mark_probabilities: Vec::new(),
    };
    let uncensored = [0.1, 0.35, 0.6, 0.8];
    let spells: Vec<SpellPit> = uncensored.iter().map(|&u| spell(u, true)).collect();
    let general = pit_uniform_distance(&spells).expect("spells");
    let classical = kolmogorov_smirnov_uniform(&uncensored).expect("values");
    assert!(
        (general - classical).abs() < 1e-12,
        "{general} vs {classical}"
    );
    assert!((classical - 0.2).abs() < 1e-12);
    // By hand: events at 0.2 and 0.6 with a censoring at 0.4. The estimate
    // steps to 1/3 at 0.2 (three at risk), stays there past the censoring,
    // and steps to 1 at 0.6 (one at risk): the largest gap is 1 − 0.6.
    let spells = vec![spell(0.2, true), spell(0.4, false), spell(0.6, true)];
    let distance = pit_uniform_distance(&spells).expect("spells");
    assert!((distance - 0.4).abs() < 1e-12, "{distance}");
    // Censored spells alone: the estimate never rises, so the gap at the
    // end of the covered range is the largest value itself.
    let spells = vec![spell(0.3, false), spell(0.5, false)];
    assert!((pit_uniform_distance(&spells).expect("spells") - 0.5).abs() < 1e-12);
    assert!(pit_uniform_distance(&[]).is_none());
}

#[test]
fn a_prefix_forecast_sees_only_what_was_known_at_the_cutoff() {
    install_test_logger();
    let mut cohort = competing_risks_cohort(64);
    let spec = EventHistorySpec::new(vec![intercept_only_spec()]);
    let fit = fit_event_history(&mut cohort, &spec).expect("fit");
    let kinds = cohort.mark_kinds.clone();
    let cutoff = 2.0;
    let subject = cohort
        .subjects
        .iter()
        .find(|s| {
            s.exit > 3.0
                && s.events.iter().any(|e| e.time > cutoff && e.time < s.exit)
                && s.terminal_event(&kinds).is_none()
        })
        .expect("a subject still under follow-up with records after the cutoff");
    let prefix = subject.prefix(cutoff, &kinds).expect("prefix");
    assert_eq!(prefix.exit, cutoff);
    assert!(prefix.events.iter().all(|e| e.time <= cutoff));
    assert!(prefix.events.len() < subject.events.len());
    let horizons = [2.5, 3.0];
    let from_prefix = forecast(
        &fit,
        &cohort,
        &ForecastRequest {
            history: &prefix,
            horizons: &horizons,
            future: &[],
            stratum: 0,
        },
    )
    .expect("forecast from the prefix");
    // Records appended after the cutoff change nothing about the prefix.
    let mut extended = subject.clone();
    extended.events.push(Event { time: 2.7, mark: 2 });
    extended.events.sort_by(|a, b| a.time.total_cmp(&b.time));
    let again = forecast(
        &fit,
        &cohort,
        &ForecastRequest {
            history: &extended.prefix(cutoff, &kinds).expect("prefix"),
            horizons: &horizons,
            future: &[],
            stratum: 0,
        },
    )
    .expect("forecast from the extended prefix");
    assert_eq!(from_prefix.survival, again.survival);
    assert_eq!(from_prefix.expected_counts, again.expected_counts);
    // The prefix as a history of its own, carrying its own covariate row,
    // forecasts the same: the training cohort's rows are not consulted.
    let row = cohort
        .covariates
        .row(subject.segments[0].row)
        .to_owned()
        .insert_axis(Axis(0));
    let mut own = prefix.clone();
    for segment in &mut own.segments {
        segment.row = 0;
    }
    let standalone = forecast_history(
        &fit,
        &cohort,
        &HistoryForecastRequest {
            history: &own,
            covariates: row.view(),
            horizons: &horizons,
            future: &[],
            stratum: 0,
        },
    )
    .expect("forecast_history");
    for h in 0..horizons.len() {
        assert!((standalone.survival[h] - from_prefix.survival[h]).abs() < 1e-12);
        for d in 0..3 {
            assert!(
                (standalone.expected_counts[[h, d]] - from_prefix.expected_counts[[h, d]]).abs()
                    < 1e-12
            );
        }
    }
    // A row table of the wrong width is refused.
    let wide = Array2::<f64>::zeros((1, 2));
    assert!(
        forecast_history(
            &fit,
            &cohort,
            &HistoryForecastRequest {
                history: &own,
                covariates: wide.view(),
                horizons: &horizons,
                future: &[],
                stratum: 0,
            },
        )
        .is_err()
    );
    // A cutoff past the exit would fabricate exposure for someone still under
    // follow-up, and is refused; a cutoff at or before the entry leaves
    // nothing to condition on.
    assert!(
        subject.prefix(subject.exit + 1.0, &kinds).is_err(),
        "a cutoff after the exit of a censored subject must be refused"
    );
    assert!(subject.prefix(subject.entry, &kinds).is_err());
    // A subject whose follow-up a terminal event ended by the cutoff is
    // returned whole, and its forecast is the certainty it deserves.
    let dead = cohort
        .subjects
        .iter()
        .find(|s| s.terminal_event(&kinds).is_some() && s.exit < 3.0)
        .expect("a subject with a terminal event before 3.0");
    // Someone whose follow-up a terminal event ended is different: nothing
    // more could have been observed after it, so a later cutoff is not a
    // claim about unobserved time and the history is returned whole.
    let whole = dead
        .prefix(3.0, &kinds)
        .expect("a completed history is returned whole");
    assert_eq!(whole, *dead);
    assert_eq!(
        dead.prefix(dead.exit + 5.0, &kinds)
            .expect("a death is the end of the record"),
        *dead
    );
    assert_eq!(
        dead.prefix(dead.exit, &kinds)
            .expect("prefix at its own exit"),
        *dead
    );
    let zero = forecast(
        &fit,
        &cohort,
        &ForecastRequest {
            history: &whole,
            horizons: &[dead.exit + 1.0],
            future: &[],
            stratum: 0,
        },
    )
    .expect("forecast of a subject whose follow-up ended");
    assert_eq!(zero.survival, vec![0.0]);
}

#[test]
fn per_mark_formulas_give_each_mark_its_own_terms() {
    install_test_logger();
    let mut cohort = competing_risks_cohort(65);
    let fit = fit_event_history_formulas(
        &mut cohort,
        &["x", "1", "x"],
        BlockwiseFitOptions::default(),
        None,
    )
    .expect("fit with one formula per mark");
    assert_eq!(fit.mark_coefficients(0).len(), 2, "intercept and x");
    assert_eq!(fit.mark_coefficients(1).len(), 1, "intercept alone");
    assert_eq!(fit.mark_coefficients(2).len(), 2);
    let refused =
        fit_event_history_formulas(&mut cohort, &["x", "1"], BlockwiseFitOptions::default(), None)
            .err()
            .expect("two formulas for three marks must be refused");
    assert!(refused.to_string().contains("one per mark"), "{refused}");
}

/// Independent continuous-time solution: invert the lognormal Laplace
/// transform at S(t)=exp(-b t). No temporal recurrence is shared with production.
fn static_frailty_reference(eta0: f64, loading: f64, times: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let points = 2001;
    let step = 18.0 / (points - 1) as f64;
    let activity: Vec<f64> = (0..points).map(|i| (loading * (-9.0 + step * i as f64)).exp()).collect();
    let phi: Vec<f64> = (0..points).map(|i| (-0.5 * (-9.0 + step * i as f64).powi(2)).exp()).collect();
    let total: f64 = phi.iter().sum();
    let mass = |h: f64| -> f64 { activity.iter().zip(&phi).map(|(r, p)| p * (-h * r).exp()).sum::<f64>() / total };
    let mut normalisers = Vec::new();
    let mut masses = Vec::new();
    for &t in times {
        let target = (-eta0.exp() * t).exp();
        let mut upper = 1.0;
        while mass(upper) > target { upper *= 2.0; }
        let mut lower = 0.0;
        for _ in 0..55 {
            let middle = 0.5 * (lower + upper);
            if mass(middle) > target { lower = middle; } else { upper = middle; }
        }
        let h = 0.5 * (lower + upper);
        let tilted: f64 = activity.iter().zip(&phi).map(|(r, p)| p * r * (-h * r).exp()).sum::<f64>() / total;
        normalisers.push((tilted / target).ln());
        masses.push(target.ln());
    }
    (normalisers, masses)
}

#[test]
fn the_risk_set_normaliser_matches_an_independent_quadrature_of_the_population() {
    install_test_logger();
    // A single first-occurrence mark, a static frailty, a constant baseline:
    // the case an independent one-dimensional quadrature settles exactly.
    let eta0_value = -1.2_f64;
    let loading = 0.9_f64;
    let horizon = 6.0_f64;
    let nodes = 481;
    let times: Vec<f64> = (0..nodes)
        .map(|n| horizon * n as f64 / (nodes - 1) as f64)
        .collect();
    let gaps: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
    let grid = ReferenceGrid { times: times.clone(), gaps };
    let gh = GaussHermite::new(15).expect("quadrature");
    let eta0 = vec![eta0_value; nodes];
    let out = stratum_normalisers(
        &grid,
        &eta0,
        &[loading],
        &[1e-6],
        1.0,
        &gh,
        &[MarkKind::Once],
        1,
    )
    .expect("reference population");
    let (expected, expected_mass) =
        static_frailty_reference(eta0_value, loading, &times);
    let gap = out
        .log_normaliser
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    emit(&format!(
        "[preserve] log M at t=0 {:.6} (independent {:.6}), at t={horizon} {:.6} (independent {:.6}); largest gap {gap:.2e}",
        out.log_normaliser[0],
        expected[0],
        out.log_normaliser[nodes - 1],
        expected[nodes - 1]
    ));
    // At time zero the risk set is the whole population, where the normaliser
    // is the stationary prior's own `½a²`.
    assert!(
        (out.log_normaliser[0] - 0.5 * loading * loading).abs() < 1e-9,
        "at the first node the risk set is everybody: {} vs {}",
        out.log_normaliser[0],
        0.5 * loading * loading
    );
    // And it falls from there: the survivors are the low-activity half.
    assert!(
        out.log_normaliser[nodes - 1] < out.log_normaliser[0] - 0.15,
        "the normaliser must fall as the risk set selects: {:?}",
        &out.log_normaliser[..3]
    );
    assert!(
        gap < 5e-5,
        "largest gap from the independent quadrature is {gap}"
    );

    // The theorem the centring exists for: the marginal survival of the
    // reference population is exactly the integral of its own baseline. Under
    // the stationary prior's constant centring it is not — the survivors are
    // selected, and the population outlives its baseline.
    let claimed = (-horizon * eta0_value.exp()).exp();
    let realised = out.log_risk_mass[nodes - 1].exp();
    let independent = expected_mass[nodes - 1].exp();
    emit(&format!(
        "[preserve] marginal survival: risk-set centred {realised:.6}, independent quadrature {independent:.6}, exp(−∫e^{{η⁰}}) {claimed:.6}"
    ));
    assert!(
        (realised - claimed).abs() < 5e-3,
        "risk-set centred survival {realised} against exp(−∫e^{{η⁰}}) {claimed}"
    );
    assert!(
        (independent - claimed).abs() < 5e-3,
        "{independent} vs {claimed}"
    );
    // The prior-centred model, same baseline and loading, survives above it.
    let prior_centred: f64 = {
        let points = 2001;
        let step = 18.0 / (points - 1) as f64;
        let (mut mass, mut total) = (0.0, 0.0);
        for i in 0..points {
            let z = -9.0 + step * i as f64;
            let phi = (-0.5 * z * z).exp();
            let rate = (eta0_value - 0.5 * loading * loading + loading * z).exp();
            mass += phi * (-horizon * rate).exp();
            total += phi;
        }
        mass / total
    };
    emit(&format!(
        "[preserve] the same model centred on the prior survives {prior_centred:.6}"
    ));
    assert!(
        prior_centred > claimed + 0.02,
        "prior centring must leave the population outliving its baseline: {prior_centred} vs {claimed}"
    );
}

#[test]
fn without_loadings_the_two_centrings_are_the_same_model() {
    let nodes = 9;
    let times: Vec<f64> = (0..nodes).map(|n| n as f64 * 0.5).collect();
    let gaps: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
    let grid = ReferenceGrid {
        times,
        gaps,
    };
    let gh = GaussHermite::new(9).expect("quadrature");
    let eta0: Vec<f64> = (0..nodes * 2).map(|i| -1.0 - 0.01 * i as f64).collect();
    let out = stratum_normalisers(
        &grid,
        &eta0,
        &[0.0, 0.0],
        &[0.5],
        1.0,
        &gh,
        &[MarkKind::Once, MarkKind::Terminal],
        1,
    )
    .expect("reference population");
    let largest = out
        .log_normaliser
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    assert!(
        largest < 1e-12,
        "a latent term with no loadings needs no normaliser: {largest}"
    );
    assert_eq!(
        out.masks, 2,
        "a once-only mark and a terminal one have different risk sets"
    );
}

#[test]
fn every_mark_kind_gets_the_risk_set_its_kind_defines() {
    let (masks, of_mark) = killing_masks(&[
        MarkKind::Once,
        MarkKind::Recurrent,
        MarkKind::Terminal,
        MarkKind::Once,
    ]);
    // A recurrent mark and a terminal one are both at risk while alive, so
    // they share the living's killing; each once-only mark adds itself.
    assert_eq!(
        of_mark[1], of_mark[2],
        "recurrent and terminal share the living's risk set"
    );
    assert_ne!(of_mark[0], of_mark[1]);
    assert_ne!(
        of_mark[0], of_mark[3],
        "two once-only marks leave at different times"
    );
    assert_eq!(masks.len(), 3);
    assert_eq!(masks[of_mark[1]], vec![false, false, true, false]);
    assert_eq!(masks[of_mark[0]], vec![true, false, true, false]);
}

#[test]
fn a_risk_set_centred_fit_reads_its_baseline_as_the_marginal_incidence() {
    install_test_logger();
    // Two first-occurrence marks sharing one frailty. Two of them, because a
    // frailty is identified across marks: in a single-mark cohort its only
    // signature is a marginal rate that falls with time, which a baseline
    // free to follow time already explains, and the evidence rightly buys no
    // atom. With two marks the shared over-dispersion is visible and the
    // baselines stay free.
    //
    // The population's incidence among those still at risk falls — the
    // survivors are the low activities — so the claim to check is that the
    // fitted `exp(η⁰(t))` tracks that falling curve, which is what the
    // risk-set centring says the baseline is. The stationary prior's centring
    // reads the same surface as a rate over the cohort as it started, above
    // the later risk sets' rate.
    let follow_up = 5.0_f64;
    let mut cohort = simulate_marked_cohort(
        600,
        follow_up,
        &[-1.3, -1.5],
        0.0,
        &[0.9, 0.8],
        0.05,
        &[MarkKind::Once, MarkKind::Once],
        4242,
    );
    // The empirical hazard among those at risk, in bins of one time unit.
    let bins = 5usize;
    let width = follow_up / bins as f64;
    let mut bin_events = vec![0.0; bins];
    let mut bin_exposure = vec![0.0; bins];
    for subject in &cohort.subjects {
        // The first mark's own risk set: it stops accruing when that mark
        // fires, whatever the other mark did.
        let stop = subject
            .events
            .iter()
            .filter(|e| e.mark == 0 && e.time > subject.entry)
            .map(|e| e.time)
            .fold(subject.exit, f64::min);
        for (b, (events, exposure)) in bin_events
            .iter_mut()
            .zip(bin_exposure.iter_mut())
            .enumerate()
        {
            let (left, right) = (b as f64 * width, (b + 1) as f64 * width);
            *exposure += (stop.min(right) - subject.entry.max(left)).max(0.0);
            *events += subject
                .events
                .iter()
                .filter(|e| {
                    e.mark == 0 && e.time > subject.entry && e.time >= left && e.time < right
                })
                .count() as f64;
        }
    }
    let empirical: Vec<f64> = bin_events
        .iter()
        .zip(bin_exposure.iter())
        .map(|(e, x)| e / x)
        .collect();

    // A baseline free to follow time, so the model can express a falling
    // marginal rate; an intercept alone would assert a constant one, which a
    // selected risk set does not have.
    let mut spec = EventHistorySpec::new(Vec::new());
    let rows = design_rows(&cohort, spec.quadrature_order).expect("design rows");
    spec.covariates = vec![
        super::formula::covariate_spec_from_formula("s(time)", rows.view(), &cohort)
            .expect("baseline formula"),
    ];
    let prior_centred = fit_event_history(&mut cohort, &spec).expect("prior-centred fit");
    spec.reference = Some(ReferenceStrata::single(0, cohort.subjects.len()));
    let centred = fit_event_history(&mut cohort, &spec).expect("risk-set centred fit");

    // The fitted baseline in each bin: the mean of `exp(η⁰)` over the
    // training nodes that fall in it.
    let fitted_in_bins = |fit: &EventHistoryFit| -> Vec<f64> {
        let eta = &fit.fit.block_states[0].eta;
        let mut total = vec![0.0; bins];
        let mut count = vec![0.0; bins];
        for subject in &fit.nodes.subjects {
            for (n, &t) in subject.times.iter().enumerate() {
                let b = ((t / width) as usize).min(bins - 1);
                total[b] += eta[subject.first_row + n].exp();
                count[b] += 1.0;
            }
        }
        total.iter().zip(count.iter()).map(|(t, c)| t / c).collect()
    };
    let centred_rate = fitted_in_bins(&centred);
    let prior_rate = fitted_in_bins(&prior_centred);
    emit(&format!(
        "[preserve] rank {} rounds {:?}\n  empirical hazard among those at risk {empirical:.4?}\n  risk-set centred baseline          {centred_rate:.4?}\n  prior centred baseline             {prior_rate:.4?}",
        centred.rank(),
        centred.reference_refinements
    ));
    assert!(
        centred.rank() > 0,
        "the fixture must buy a latent direction for the centrings to differ"
    );
    assert!(
        !centred.reference_refinements.is_empty(),
        "a risk-set centred fit must have refreshed its normaliser at least once"
    );
    assert!(centred.reference_certificate.is_some_and(|gap| gap <= 1e-4));
    let reevaluated = centred.family.refresh_normaliser(&centred.fit.block_states).unwrap();
    assert_eq!(centred.centring.as_ref().unwrap().log_normaliser, reevaluated.log_normaliser);
    assert_eq!(centred.centring.as_ref().unwrap().log_risk_mass, reevaluated.log_risk_mass);
    assert!(
        !centred.centring.as_ref().unwrap().log_risk_mass.is_empty() && centred.centring.as_ref().unwrap().masks > 0,
        "the fit must publish the reference population's own risk mass"
    );
    // The empirical hazard falls over follow-up, which is the selection the
    // centring exists to account for.
    assert!(
        empirical[bins - 1] < 0.8 * empirical[0],
        "the fixture's risk set must be visibly selected: {empirical:?}"
    );
    // The centred baseline tracks it, bin by bin, within the sampling error
    // of a rate from that bin's own events.
    for b in 0..bins {
        let tolerance = 4.0 / bin_events[b].sqrt();
        let relative = (centred_rate[b] - empirical[b]).abs() / empirical[b];
        assert!(
            relative < tolerance,
            "bin {b}: risk-set centred baseline {} against the empirical hazard {} (relative {relative} over {tolerance}, {} events)",
            centred_rate[b],
            empirical[b],
            bin_events[b]
        );
    }
    // The prior's centring reads the same surface as a rate over the cohort
    // as it started, so in the late bins it sits above the risk set's own.
    assert!(
        prior_rate[bins - 1] > centred_rate[bins - 1],
        "the prior's centring must sit above the late risk set's rate: {} vs {}",
        prior_rate[bins - 1],
        centred_rate[bins - 1]
    );
}

#[test]
fn finite_combined_log_rate_preserves_value_and_derivatives() {
    use super::marginal::node_likelihood;
    use super::scalar::Tangent;
    let like = Tangent::<1>::seeded(0.0, [0.0]);
    let grid = Grid::new(&GaussHermite::new(3).unwrap(),
        &[like.constant_like(800.0)], &[like.constant_like(1.0)], &like);
    let value = node_likelihood(&grid, &[like.constant_like(-800.0)],
        &[Tangent::seeded(1.0, [1.0])], &[0.0], &[1.0], None,
        Some(&[like]), 1, 1, false);
    assert!((value.ell[1].value + 1.0).abs() < 1e-12);
    assert!((value.ell[1].grad[0] + 800.0).abs() < 1e-9);
    assert!(value.ell.iter().all(|value| value.value.is_finite() && value.grad[0].is_finite()));
}

#[test]
fn reference_endpoints_are_supported_and_extrapolation_is_rejected() {
    let grid = ReferenceGrid { times: vec![0.0, 1.0], gaps: vec![1.0] };
    assert_eq!(grid.locate(0.0).unwrap(), (0, 0.0));
    assert_eq!(grid.locate(1.0).unwrap(), (0, 1.0));
    for t in [-0.01, 1.01, f64::NAN, f64::INFINITY] { assert!(grid.locate(t).is_err()); }
    assert!(ReferenceGrid { times: vec![], gaps: vec![] }.locate(0.0).is_err());
}

#[test]
fn reference_grid_and_strata_round_trip_without_changing_positions() {
    let grid = ReferenceGrid { times: vec![0.1, 1.7, 2.9], gaps: vec![1.7 - 0.1, 2.9 - 1.7] };
    let strata = ReferenceStrata { rows: (0..12).rev().collect(), subject: vec![10, 2, 11, 0] };
    let restored: (ReferenceGrid, ReferenceStrata) = serde_json::from_str(
        &serde_json::to_string(&(grid.clone(), strata.clone())).unwrap()).unwrap();
    assert_eq!(grid.times, restored.0.times);
    assert_eq!(grid.gaps, restored.0.gaps);
    assert_eq!(strata, restored.1);
    restored.1.validate(4, 12).unwrap();
}

#[test]
fn production_reference_grid_converges_to_the_survival_identity() {
    let gh = GaussHermite::new(15).unwrap();
    let baseline = -1.2_f64;
    let target = (-6.0 * baseline.exp()).exp();
    let mut errors = Vec::new();
    // These are the production grid's initial resolution and two refinements.
    for intervals in [36, 72, 144] {
        let times: Vec<f64> = (0..=intervals).map(|n| 6.0 * n as f64 / intervals as f64).collect();
        let grid = ReferenceGrid { gaps: times.windows(2).map(|w| w[1] - w[0]).collect(), times };
        let out = stratum_normalisers(&grid, &vec![baseline; intervals + 1], &[0.9], &[1e-8],
            1.0, &gh, &[MarkKind::Once], 1).unwrap();
        errors.push((out.log_risk_mass.last().unwrap().exp() - target).abs());
    }
    emit(&format!("continuous-time survival errors: {errors:?}"));
    assert!(errors[1] < 0.4 * errors[0], "{errors:?}");
    assert!(errors[2] < 0.4 * errors[1], "{errors:?}");
    assert!(errors[2] < 2e-5, "{errors:?}");
}

/// The documented cohort (docs/event-history.md) fits under risk-set centring (#2627). The midpoint iteration's former
/// fixed cap of twelve iterations with a literal 1e-10 refused it (job 1180863) where the map contracts; the stop is now
/// derived from the map's rounding band.
#[test]
fn reference_midpoint_resolves_the_documented_cohort_2627() {
    let n = 200usize;
    let mut state = 0x2627_0000_0000_0001_u64;
    let mut uniform = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    };
    let mut covariates = Array2::<f64>::zeros((n, 1));
    let mut subjects = Vec::with_capacity(n);
    for i in 0..n {
        let (u1, u2) = (uniform(), uniform());
        let prs = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let death = -uniform().ln() / 0.15;
        let disease = -uniform().ln() / (0.25 * (0.5 * prs).exp());
        let exit = death.min(4.0);
        covariates[[i, 0]] = prs;
        let mut events = Vec::new();
        if disease < exit {
            events.push(Event { time: disease, mark: 0 });
        }
        if death < 4.0 {
            events.push(Event { time: death, mark: 1 });
        }
        subjects.push(SubjectHistory {
            id: format!("s{i}"),
            entry: 0.0,
            exit,
            events,
            segments: vec![CovariateSegment { start: 0.0, row: i }],
        });
    }
    let mut cohort = EventHistoryCohort {
        mark_names: vec!["disease".to_string(), "death".to_string()],
        mark_kinds: vec![MarkKind::Once, MarkKind::Terminal],
        covariate_names: vec!["prs".to_string()],
        covariate_levels: vec![Vec::new()],
        covariates,
        subjects,
    };
    cohort.validate().expect("the documented cohort is valid");
    let fit = fit_event_history_formulas(
        &mut cohort,
        &["s(time, by=prs)", "s(time)"],
        BlockwiseFitOptions::default(),
        Some(ReferenceStrata::single(0, n)),
    )
    .unwrap_or_else(|error| panic!("the documented cohort must fit under risk-set centring: {error}"));
    assert!(fit.reference_certificate.is_some(), "a risk-set centred fit publishes its reference certificate");
}

/// The reference midpoint step refuses exactly where its map does not contract (#2627). One once-only mark and one
/// latent atom at log baseline 0 over six time units: at step 3 a loading of 2 makes successive changes of the midpoint
/// shift grow (ratio 1.302 in job 1186034), while a loading of 1 contracts at the same step and a loading of 2
/// contracts at steps 1 and 0.5.
#[test]
fn reference_midpoint_refuses_only_a_step_that_does_not_contract_2627() {
    let gh = GaussHermite::new(9).expect("rule");
    let evolve = |intervals: usize, loading: f64| {
        let times: Vec<f64> = (0..=intervals).map(|n| 6.0 * n as f64 / intervals as f64).collect();
        let grid = ReferenceGrid { gaps: times.windows(2).map(|w| w[1] - w[0]).collect(), times };
        stratum_normalisers(&grid, &vec![0.0; intervals + 1], &[loading], &[1e-6], 1.0, &gh, &[MarkKind::Once], 1)
    };
    match evolve(2, 2.0) {
        Err(super::cohort::EventHistoryError::ReferenceStep { interval, change, contraction, band }) => {
            assert_eq!(interval, 0, "the first step is the one that does not contract");
            assert!(
                contraction >= 1.0 && change > 2.0 * band,
                "refused at ratio {contraction} and change {change} against rounding band {band}"
            );
        }
        other => panic!(
            "a midpoint step that does not contract must refuse typed: {:?}",
            other.map(|out| out.log_risk_mass)
        ),
    }
    for (intervals, loading) in [(2, 1.0), (6, 2.0), (12, 2.0)] {
        let out = evolve(intervals, loading).unwrap_or_else(|error| {
            panic!("step {} at loading {loading} contracts: {error}", 6.0 / intervals as f64)
        });
        assert!(out.log_normaliser.iter().chain(&out.log_risk_mass).all(|x| x.is_finite()));
    }
}

/// At a contraction the midpoint iteration stops on the value, and the derivative channels it carries have converged
/// with it (#2627): the seeded tangent of every log normaliser matches its central difference along the baseline and
/// along the loading, at step 1, where a loading of 2 contracts (it refuses at step 3). The bar is the Richardson
/// estimate `|D(h) − D(h/2)|`, three times the truncation error of `D(h/2)` for a smooth normaliser, plus the rounding
/// of `D(h/2)`: two values, each resolved to the midpoint map's rounding band on every interval, divided by `h`.
#[test]
fn reference_midpoint_derivative_channels_converge_with_the_value_2627() {
    use super::scalar::Tangent;
    use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
    let gh = GaussHermite::new(9).expect("rule");
    let intervals = 6usize;
    let times: Vec<f64> = (0..=intervals).map(|n| 6.0 * n as f64 / intervals as f64).collect();
    let grid = ReferenceGrid { gaps: times.windows(2).map(|w| w[1] - w[0]).collect(), times };
    let evolve = |baseline: Tangent<1>, loading: Tangent<1>| -> Vec<Tangent<1>> {
        stratum_normalisers(
            &grid,
            &vec![baseline; intervals + 1],
            &[loading],
            &[Tangent::seeded(1e-6, [0.0])],
            1.0,
            &gh,
            &[MarkKind::Once],
            1,
        )
        .expect("a loading of 2 contracts at step 1")
        .log_normaliser
    };
    let at = |baseline: f64, loading: f64| -> Vec<f64> {
        evolve(Tangent::seeded(baseline, [0.0]), Tangent::seeded(loading, [0.0]))
            .iter()
            .map(|x| x.value)
            .collect()
    };
    let h = 1e-3;
    for (name, direction) in [("baseline", [1.0, 0.0]), ("loading", [0.0, 1.0])] {
        let jets = evolve(Tangent::seeded(0.0, [direction[0]]), Tangent::seeded(2.0, [direction[1]]));
        let central = |step: f64| -> Vec<f64> {
            let plus = at(step * direction[0], 2.0 + step * direction[1]);
            let minus = at(-step * direction[0], 2.0 - step * direction[1]);
            plus.iter().zip(&minus).map(|(p, m)| (p - m) / (2.0 * step)).collect()
        };
        let (coarse, fine) = (central(h), central(0.5 * h));
        let mut moves = false;
        for (n, jet) in jets.iter().enumerate() {
            let band = 4.0 * accumulation_growth(gh.order + 1) + UNIT_ROUNDOFF * (1.0 + jet.value.abs());
            let bar = (coarse[n] - fine[n]).abs() + 2.0 * (2.0 * intervals as f64 * band) / h;
            assert!(
                (jet.grad[0] - fine[n]).abs() <= bar,
                "log normaliser {n} along the {name}: tangent {} vs central difference {} (bar {bar:.3e})",
                jet.grad[0],
                fine[n]
            );
            moves |= jet.grad[0].abs() > bar;
        }
        assert!(moves, "no log normaliser moves along the {name} by more than its bar, so the agreement is vacuous");
    }
}

/// One latent atom injected into a rank-zero fit: loadings `a_d` per mark and
/// dimensionless rate `nu` (zero for a static factor). The rank is the number
/// of rates, so the forecast runs the latent filter. Every parameter it reads
/// is then known exactly, which is what lets an independent quadrature of the
/// same model serve as its oracle.
fn inject_one_atom(fit: &mut EventHistoryFit, loadings: [f64; 3], nu: f64) {
    fit.loadings = Array2::from_shape_vec((3, 1), loadings.to_vec()).expect("one column of loadings");
    fit.log_rates = vec![nu.ln()];
    assert_eq!(fit.rank(), 1);
}

/// The constant-hazard fixture fitted under Gauss-Hermite order `order`. At
/// rank zero the rule integrates nothing, so the fitted rates are the same at
/// every order, and an atom injected afterwards is forecast under that rule.
fn constant_hazard_fit_at(order: usize) -> (EventHistoryCohort, EventHistoryFit, Vec<f64>) {
    let mut cohort = competing_risks_cohort(64);
    let mut spec = EventHistorySpec::new(vec![intercept_only_spec()]);
    spec.gauss_hermite_order = order;
    let fit = fit_event_history(&mut cohort, &spec).expect("intercept-only fit");
    assert_eq!(fit.rank(), 0, "the constant-hazard fixture must be rank zero: {:?}", fit.rank_path);
    assert_eq!(fit.family.gauss_hermite_order(), order, "the fit left Gauss-Hermite order {order}");
    let rates = (0..3).map(|d| fit.mark_coefficients(d)[0].exp()).collect();
    (cohort, fit, rates)
}

/// `λ_d(z) = exp(η⁰_d − a_d²/2 + a_d z)`: the prior-centred intensity.
fn one_atom_intensity(rates: &[f64], loadings: &[f64; 3], d: usize, z: f64) -> f64 {
    rates[d] * (loadings[d] * z - 0.5 * loadings[d] * loadings[d]).exp()
}

/// The static one-atom model's survival, terminal incidences and recurrent
/// count `h` after a history, by the trapezoid rule over the factor with
/// spacing `step`. The history is `events` (the marks of its events) over a
/// follow-up of `follow_up` with every mark at risk, as for a censored subject
/// under constant hazards; no events and no follow-up is the stationary prior.
/// It weighs the prior into the posterior
/// `φ(z) Π_e λ_{m_e}(z) exp(−Σ_d λ_d(z) T)`, and the forecast integrands are
/// averaged under that posterior.
///
/// Returns the values, their rounding and the mass the rule leaves outside
/// its range. The rule's own error is MEASURED by the caller's step-halving
/// gap: the trapezoid rule converges spectrally for an analytic integrand under
/// a Gaussian.
/// - Rounding: each value is a quotient of two serial sums. Its first-order
///   running bound is `ε |v| (μ_N/|N| + μ_D/|D| + 1)`, with `μ` the magnitude of
///   every partial sum (Higham, *Accuracy and Stability*, ch. 3) and one more
///   rounding for the division. A term's own rounding is a few operations
///   relative to itself, dominated by the partial sums' over thousands of
///   terms.
/// - Range: `ln` of the posterior weight is at most `−z²/2 + A z + C`, with
///   `A = Σ_e a_{m_e}` and `C = Σ_e (ln r_{m_e} − a_{m_e}²/2)`, dropping the
///   compensator. The survival and terminal integrands are at most one, and the
///   recurrent one at most `λ_2 h`, so each is at most `B e^{b z}`. Beyond
///   `|z| > R` the weighted integrand's mass is then at most
///   `2 B e^{C − peak + s²/2 − (R − |s|)²/2} / (R − |s|)` with `s = A + b`
///   (Mills' ratio), and the range is `R = 12 + |A| + |a_2|`.
fn static_factor_oracle(
    rates: &[f64],
    loadings: &[f64; 3],
    events: &[usize],
    follow_up: f64,
    h: f64,
    step: f64,
) -> ([f64; 4], [f64; 4], [f64; 4]) {
    let tilt: f64 = events.iter().map(|&m| loadings[m]).sum();
    let offset: f64 = events
        .iter()
        .map(|&m| rates[m].ln() - 0.5 * loadings[m] * loadings[m])
        .sum();
    let reach = 12.0 + tilt.abs() + loadings[2].abs();
    let n = (2.0 * reach / step).round() as usize;
    let log_weight = |z: f64| -> f64 {
        let mut value = -0.5 * z * z;
        for &m in events {
            value += one_atom_intensity(rates, loadings, m, z).ln();
        }
        for d in 0..3 {
            value -= one_atom_intensity(rates, loadings, d, z) * follow_up;
        }
        value
    };
    let peak = (0..=n)
        .map(|j| log_weight(-reach + step * j as f64))
        .fold(f64::NEG_INFINITY, f64::max);
    let (mut denominator, mut denominator_bound) = (0.0_f64, 0.0_f64);
    let mut numerators = [0.0_f64; 4];
    let mut numerator_bounds = [0.0_f64; 4];
    for j in 0..=n {
        let z = -reach + step * j as f64;
        let ends = if j == 0 || j == n { 0.5 } else { 1.0 };
        let weight = ends * step * (log_weight(z) - peak).exp();
        let lambda: Vec<f64> = (0..3).map(|d| one_atom_intensity(rates, loadings, d, z)).collect();
        let killing = lambda[0] + lambda[1];
        let decrement = -(-killing * h).exp_m1();
        let ratios = [
            (-killing * h).exp(),
            lambda[0] / killing * decrement,
            lambda[1] / killing * decrement,
            lambda[2] / killing * decrement,
        ];
        denominator += weight;
        denominator_bound += denominator.abs();
        for q in 0..4 {
            numerators[q] += weight * ratios[q];
            numerator_bounds[q] += numerators[q].abs();
        }
    }
    let outside = |scale: f64, slope: f64| -> f64 {
        let s = tilt + slope;
        let margin = reach - s.abs();
        2.0 * scale * (offset - peak + 0.5 * s * s - 0.5 * margin * margin).exp() / margin
    };
    let recurrent_scale = h * rates[2] * (-0.5 * loadings[2] * loadings[2]).exp();
    let denominator_tail = outside(1.0, 0.0);
    let mut values = [0.0; 4];
    let mut rounding = [0.0; 4];
    let mut tails = [0.0; 4];
    for q in 0..4 {
        values[q] = numerators[q] / denominator;
        rounding[q] = f64::EPSILON
            * values[q].abs()
            * (numerator_bounds[q] / numerators[q].abs() + denominator_bound / denominator.abs() + 1.0);
        let numerator_tail = if q == 3 { outside(recurrent_scale, loadings[2]) } else { outside(1.0, 0.0) };
        tails[q] = numerator_tail / denominator + values[q].abs() * denominator_tail / denominator;
    }
    (values, rounding, tails)
}

/// The static one-atom oracle's value and error: its value at spacing `step`,
/// with the step-halving gap from `2 · step` (MEASURED), both values' rounding
/// and both ranges' outside mass.
fn static_factor_reference(
    rates: &[f64],
    loadings: &[f64; 3],
    events: &[usize],
    follow_up: f64,
    h: f64,
    step: f64,
) -> ([f64; 4], [f64; 4]) {
    let (coarse, coarse_rounding, coarse_tail) = static_factor_oracle(rates, loadings, events, follow_up, h, 2.0 * step);
    let (value, rounding, tail) = static_factor_oracle(rates, loadings, events, follow_up, h, step);
    let mut error = [0.0; 4];
    for q in 0..4 {
        error[q] = (coarse[q] - value[q]).abs() + coarse_rounding[q] + rounding[q] + coarse_tail[q] + tail[q];
    }
    (value, error)
}

/// The dynamic one-atom model from its stationary prior over `[0, h]`, by a
/// route independent of the filter. The density relative to the standard
/// normal is represented on the orthonormal Hermite basis of degree below
/// `order`, where the Ornstein–Uhlenbeck transition is exactly diagonal
/// (`e^{−nκ}`). The killing acts at the Gauss–Hermite nodes by Strang
/// splitting over equal steps, and the sub-densities are summed by the
/// trapezoid rule on those steps. Both are second order, so one Richardson step
/// between `steps` and `2·steps` leaves a fourth-order value.
fn spectral_oracle(
    rates: &[f64],
    loadings: &[f64; 3],
    kappa_per_time: f64,
    h: f64,
    order: usize,
    steps: usize,
) -> [f64; 4] {
    let rule = gam_math::quadrature::gauss_hermite_rule(order).expect("Gauss-Hermite rule");
    let z: Vec<f64> = rule.nodes.iter().map(|x| std::f64::consts::SQRT_2 * x).collect();
    let v: Vec<f64> = rule.weights.iter().map(|w| w / std::f64::consts::PI.sqrt()).collect();
    // basis[n][i] = ψ_n(z_i), with ψ_{n+1} = (z ψ_n − √n ψ_{n−1}) / √(n+1).
    let mut basis = vec![vec![0.0; order]; order];
    for i in 0..order {
        basis[0][i] = 1.0;
        basis[1][i] = z[i];
        for n in 1..order - 1 {
            basis[n + 1][i] = (z[i] * basis[n][i] - (n as f64).sqrt() * basis[n - 1][i]) / ((n + 1) as f64).sqrt();
        }
    }
    let lambda: Vec<Vec<f64>> = (0..3)
        .map(|d| z.iter().map(|&zi| one_atom_intensity(rates, loadings, d, zi)).collect())
        .collect();
    let densities = |p: &[f64]| -> [f64; 3] {
        let mut m = [0.0; 3];
        for d in 0..3 {
            m[d] = (0..order).map(|i| v[i] * p[i] * lambda[d][i]).sum();
        }
        m
    };
    let run = |steps: usize| -> [f64; 4] {
        let dt = h / steps as f64;
        let half_kill: Vec<f64> = (0..order).map(|i| (-(lambda[0][i] + lambda[1][i]) * 0.5 * dt).exp()).collect();
        let decay: Vec<f64> = (0..order).map(|n| (-(n as f64) * kappa_per_time * dt).exp()).collect();
        let mut p = vec![1.0; order];
        let mut counts = [0.0; 3];
        let mut previous = densities(&p);
        for _ in 0..steps {
            for i in 0..order {
                p[i] *= half_kill[i];
            }
            let coefficients: Vec<f64> = (0..order)
                .map(|n| (0..order).map(|i| v[i] * p[i] * basis[n][i]).sum::<f64>() * decay[n])
                .collect();
            for i in 0..order {
                p[i] = (0..order).map(|n| coefficients[n] * basis[n][i]).sum::<f64>() * half_kill[i];
            }
            let current = densities(&p);
            for d in 0..3 {
                counts[d] += 0.5 * dt * (previous[d] + current[d]);
            }
            previous = current;
        }
        [(0..order).map(|i| v[i] * p[i]).sum(), counts[0], counts[1], counts[2]]
    };
    let coarse = run(steps);
    let fine = run(2 * steps);
    let mut value = [0.0; 4];
    for q in 0..4 {
        value[q] = (4.0 * fine[q] - coarse[q]) / 3.0;
    }
    value
}

/// The dynamic one-atom oracle's value and error: its value at (56 basis
/// functions, 800 steps), with the error MEASURED as its change across three
/// resolutions, both consecutive changes summed. At these resolutions both sit
/// at the oracle's rounding floor (probe 1230171: 3.8e-14, then 6.3e-14), so
/// the finer one being the closer is not assumed.
fn spectral_reference(rates: &[f64], loadings: &[f64; 3], kappa_per_time: f64, h: f64) -> ([f64; 4], [f64; 4]) {
    let coarse = spectral_oracle(rates, loadings, kappa_per_time, h, 40, 400);
    let value = spectral_oracle(rates, loadings, kappa_per_time, h, 56, 800);
    let fine = spectral_oracle(rates, loadings, kappa_per_time, h, 72, 1600);
    let mut error = [0.0; 4];
    for q in 0..4 {
        error[q] = (coarse[q] - value[q]).abs() + (value[q] - fine[q]).abs();
    }
    (value, error)
}

/// A forecast's survival and three expected counts at horizon `i`, and their
/// checked errors.
fn forecast_quantities(f: &super::forecast::Forecast, i: usize) -> ([f64; 4], [f64; 4]) {
    (
        [f.survival[i], f.expected_counts[[i, 0]], f.expected_counts[[i, 1]], f.expected_counts[[i, 2]]],
        [
            f.survival_error[i],
            f.expected_count_errors[[i, 0]],
            f.expected_count_errors[[i, 1]],
            f.expected_count_errors[[i, 2]],
        ],
    )
}

/// The rank-zero model's survival and three expected counts `h` into a window
/// at the fitted constant rates: `e^{−Λh}` and `(r_d/Λ)(1 − e^{−Λh})`, with `Λ`
/// the terminal marks' total rate.
fn rank_zero_quantities(rates: &[f64], h: f64) -> [f64; 4] {
    let total = rates[0] + rates[1];
    let decrement = -(-total * h).exp_m1();
    [
        (-total * h).exp(),
        rates[0] / total * decrement,
        rates[1] / total * decrement,
        rates[2] / total * decrement,
    ]
}

/// The magnitude floor of a one-atom agreement: for every quantity, the latent
/// factor moves the oracle from the rank-zero model by more than the checked
/// error and the oracle's error together. Agreement with the oracle within that
/// error is then not agreement with the rank-zero model, and an error inflated
/// past the factor's own effect fails here.
fn assert_the_factor_is_resolved(label: &str, h: f64, rates: &[f64], errors: [f64; 4], oracle: [f64; 4], oracle_error: [f64; 4]) {
    let rank_zero = rank_zero_quantities(rates, h);
    emit(&format!("[2963 {label}] h {h:.4}: rank-zero model {rank_zero:?}"));
    for q in 0..4 {
        assert!(
            (oracle[q] - rank_zero[q]).abs() > errors[q] + oracle_error[q],
            "{label}: quantity {q} at h {h}: the factor moves it from {} to {} only, within the checked error {} + oracle error {}",
            rank_zero[q],
            oracle[q],
            errors[q],
            oracle_error[q]
        );
    }
}

/// Assert that a one-atom forecast is covered by its checked error against an
/// oracle, above the magnitude floor of [`assert_the_factor_is_resolved`].
fn assert_covered(label: &str, f: &super::forecast::Forecast, i: usize, h: f64, rates: &[f64], oracle: [f64; 4], oracle_error: [f64; 4]) {
    let (forecast, errors) = forecast_quantities(f, i);
    emit(&format!("[2963 {label}] h {h:.4}: forecast {forecast:?} errors {errors:?} oracle {oracle:?} oracle error {oracle_error:?}"));
    assert_the_factor_is_resolved(label, h, rates, errors, oracle, oracle_error);
    for q in 0..4 {
        assert!(
            (forecast[q] - oracle[q]).abs() <= errors[q] + oracle_error[q],
            "{label}: quantity {q} at h {h}: forecast {} vs oracle {}, checked error {} + oracle error {}",
            forecast[q],
            oracle[q],
            errors[q],
            oracle_error[q]
        );
    }
}

#[test]
fn a_static_factor_forecast_is_covered_by_its_reported_error() {
    install_test_logger();
    let loadings = [0.8, -0.5, 0.6];
    let (cohort, mut fit, rates) = constant_hazard_fit();
    inject_one_atom(&mut fit, loadings, 0.0);
    let total = rates[0] + rates[1];
    let horizons = [2.0 / total, 10.0 / total];
    let f = constant_hazard_population(&fit, &cohort, &horizons);
    for (i, &h) in horizons.iter().enumerate() {
        let (oracle, oracle_error) = static_factor_reference(&rates, &loadings, &[], 0.0, h, 0.005);
        assert_covered("static", &f, i, h, &rates, oracle, oracle_error);
    }
}

#[test]
fn a_dynamic_factor_forecast_is_covered_by_its_reported_error() {
    install_test_logger();
    let loadings = [0.8, -0.5, 0.6];
    let (cohort, mut fit, rates) = constant_hazard_fit();
    let total = rates[0] + rates[1];
    let horizons = [2.0 / total, 10.0 / total];
    // The factor decorrelates about three times over the window.
    let kappa_per_time = 3.0 / horizons[1];
    let nu = kappa_per_time * fit.time_scale;
    inject_one_atom(&mut fit, loadings, nu);
    let f = constant_hazard_population(&fit, &cohort, &horizons);
    for (i, &h) in horizons.iter().enumerate() {
        let (oracle, oracle_error) = spectral_reference(&rates, &loadings, kappa_per_time, h);
        assert_covered("dynamic", &f, i, h, &rates, oracle, oracle_error);
    }
}

/// The forecast opens at the state a history implies, not at the prior, so
/// the error it reports must cover what filtering that history contributed.
/// A censored subject with events, under a static factor: its posterior is
/// the oracle's, and the window after its exit is covered as the prior's is.
#[test]
fn a_history_conditioned_static_factor_forecast_is_covered_by_its_reported_error() {
    install_test_logger();
    let loadings = [0.8, -0.5, 0.6];
    let (cohort, mut fit, rates) = constant_hazard_fit();
    inject_one_atom(&mut fit, loadings, 0.0);
    let total = rates[0] + rates[1];
    let subject = cohort
        .subjects
        .iter()
        .find(|s| s.terminal_event(&cohort.mark_kinds).is_none() && !s.events.is_empty())
        .expect("a censored subject with events");
    let marks: Vec<usize> = subject.events.iter().map(|event| event.mark).collect();
    let offsets = [2.0 / total, 10.0 / total];
    let horizons: Vec<f64> = offsets.iter().map(|offset| subject.exit + offset).collect();
    let f = forecast(
        &fit,
        &cohort,
        &ForecastRequest {
            history: subject,
            horizons: &horizons,
            future: &[],
            stratum: 0,
        },
    )
    .expect("forecast from a history");
    emit(&format!(
        "[2963 history static] subject {} follow-up {} events {marks:?}",
        subject.id,
        subject.exit - subject.entry
    ));
    for (i, &offset) in offsets.iter().enumerate() {
        let (oracle, oracle_error) =
            static_factor_reference(&rates, &loadings, &marks, subject.exit - subject.entry, offset, 0.005);
        assert_covered("history static", &f, i, offset, &rates, oracle, oracle_error);
    }
}

/// A forecast's roundoff and latent error are its own rule's, so a finer
/// Gauss-Hermite rule gives a forecast at least as good and says so. The same
/// one-atom window under orders 9 and 17:
/// - the order-17 forecast lies within the order-9 forecast's checked error
///   of the oracle, as a forecast under the more accurate rule must (the
///   premise the latent check rests on), above the order-9 forecast's
///   magnitude floor; and
/// - when `resolves`, the order-17 checked error resolves the change from
///   order 9.
///
/// For a static factor the first arm is that premise as a measurement
/// precondition: no defect-state control moves it, since the order-17 static
/// forecast was 5.0e-10 from the oracle on the pre-fix code too (probe
/// 1230171). It prints before it asserts. For a dynamic factor it is the arm
/// the pre-fix code's stalled mesh fires.
fn assert_the_finer_rule_is_within_the_coarser_ones_error(label: &str, dynamic: bool, resolves: bool) {
    let loadings = [0.8, -0.5, 0.6];
    let mut at_order = Vec::new();
    for order in [9, 17] {
        let (cohort, mut fit, rates) = constant_hazard_fit_at(order);
        let total = rates[0] + rates[1];
        let horizons = [2.0 / total, 10.0 / total];
        let kappa_per_time = if dynamic { 3.0 / horizons[1] } else { 0.0 };
        let nu = kappa_per_time * fit.time_scale;
        inject_one_atom(&mut fit, loadings, nu);
        let f = constant_hazard_population(&fit, &cohort, &horizons);
        at_order.push((f, rates, horizons, kappa_per_time));
    }
    let (coarse, rates, horizons, kappa_per_time) = &at_order[0];
    let fine = &at_order[1].0;
    for (i, &h) in horizons.iter().enumerate() {
        let (oracle, oracle_error) = if dynamic {
            spectral_reference(rates, &loadings, *kappa_per_time, h)
        } else {
            static_factor_reference(rates, &loadings, &[], 0.0, h, 0.005)
        };
        let (v9, e9) = forecast_quantities(coarse, i);
        let (v17, e17) = forecast_quantities(fine, i);
        for q in 0..4 {
            emit(&format!(
                "[2963 orders {label}] h {h:.4} quantity {q}: G9 {:.15} checked {:.3e} |G9−oracle| {:.3e} | G17 {:.15} checked {:.3e} |G17−oracle| {:.3e} | |G9−G17| {:.3e} | oracle error {:.3e}",
                v9[q],
                e9[q],
                (v9[q] - oracle[q]).abs(),
                v17[q],
                e17[q],
                (v17[q] - oracle[q]).abs(),
                (v9[q] - v17[q]).abs(),
                oracle_error[q]
            ));
        }
        assert_the_factor_is_resolved(&format!("orders {label}"), h, rates, e9, oracle, oracle_error);
        for q in 0..4 {
            assert!(
                (v17[q] - oracle[q]).abs() <= e9[q] + oracle_error[q],
                "{label}: quantity {q} at h {h}: the order-17 forecast {} is {} from the oracle {}, outside the order-9 checked error {} + oracle error {}",
                v17[q],
                (v17[q] - oracle[q]).abs(),
                oracle[q],
                e9[q],
                oracle_error[q]
            );
            if resolves {
                assert!(
                    e17[q] < (v9[q] - v17[q]).abs(),
                    "{label}: quantity {q} at h {h}: the order-17 checked error {} does not resolve the change {} from order 9",
                    e17[q],
                    (v9[q] - v17[q]).abs()
                );
            }
        }
    }
}

/// A static factor's filter interpolates nothing, so its checked error at
/// order 17 is that rule's own and resolves the change from order 9. Probe
/// 1230171 measured the defect this pins: the order-17 window charged every
/// value the roundoff of order 33's interpolant, 4.1e-5 on a survival of 0.17,
/// above the whole order-9 error (3.7e-6).
#[test]
fn a_finer_rule_resolves_a_static_factor_forecasts_change_from_the_coarser_one() {
    install_test_logger();
    assert_the_finer_rule_is_within_the_coarser_ones_error("static", false, true);
}

/// A dynamic factor's order-17 forecast stays within the order-9 checked
/// error of the oracle. Probe 1230171 measured the defect this pins: charged
/// order 33's roundoff floor, the order-17 window stopped refining its mesh
/// while time dominated, 5.1e-6 from the oracle where order 9 was 4.4e-8. Its
/// own checked error is not asserted to resolve the change: the dynamic
/// filter's roundoff is charged at order 17's Lebesgue constant, as the fit's
/// certificate charges it, and that model is the certificate's question.
#[test]
fn a_finer_rule_keeps_a_dynamic_factor_forecast_within_the_coarser_ones_error() {
    install_test_logger();
    assert_the_finer_rule_is_within_the_coarser_ones_error("dynamic", true, false);
}

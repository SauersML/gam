//! gam#2983: the anchored family's value, score and Hessian at a coefficient
//! iterate do not depend on thread scheduling.
//!
//! On a declared or empirical latent law every row's location channel is the
//! root of the anchoring equation, and the law's root slots keep those roots
//! across evaluations. A root accepted within the residual resolution depends
//! on where its solve started, so when a slot seeded the next solve, the
//! family's arithmetic at an iterate depended on which iterates came before it
//! and, through the cross-row table rows share, on the order threads published
//! into it: 12-thread truth2370 fits of one input diverged run to run from
//! ~1e-11 at an inner solve's second cycle. Rows here come in tied pairs, so
//! the cross-row table serves every equation twice. Each quantity is compared
//! bit for bit for one history at a time, on 12, 4 and 1 threads and repeated:
//! a fresh family, and a family that walked to the iterate.
//!
//! The two histories are not compared with each other. A walked family's
//! gradient and Hessian differ from a fresh one's by up to ~4e-15 relative
//! while its log-likelihood is identical, because some state other than the
//! anchor roots is carried between evaluations. The roots themselves are a
//! function of their equation alone, which the latent_anchor pins hold.

use super::*;
use crate::custom_family::CustomFamily;
use crate::survival::construction::{
    SurvivalBaselineConfig, SurvivalBaselineTarget, build_survival_marginal_slope_baseline_geometry,
};
use gam_linalg::matrix::DenseDesignMatrix;
use ndarray::{Array1, Array2, Axis};
use std::sync::Arc;

const N_ROWS: usize = 24;

/// Row `i` repeats row `i − N_ROWS/2`: every equation is asked by two rows.
fn tie(row: usize) -> f64 {
    (row % (N_ROWS / 2)) as f64
}

/// A two-component skewed law on 41 nodes, nothing a Gaussian describes.
fn skewed_law() -> Arc<SurvivalLatentLaw> {
    let nodes: Vec<f64> = (0..41).map(|k| -2.5 + 0.15 * k as f64).collect();
    let raw: Vec<f64> = nodes
        .iter()
        .map(|&u| {
            (-0.5 * ((u + 0.9) / 0.5).powi(2)).exp()
                + 0.35 * (-0.5 * ((u - 1.4) / 0.9).powi(2)).exp()
        })
        .collect();
    let total: f64 = raw.iter().sum();
    let grid = crate::bms::EmpiricalZGrid::new(
        nodes,
        raw.into_iter().map(|w| w / total).collect(),
        "gam#2983 tied-row law",
    )
    .expect("a valid law");
    Arc::new(
        SurvivalLatentLaw::from_kind(&crate::bms::LatentMeasureKind::GlobalEmpirical { grid }, N_ROWS)
            .expect("materialise the law")
            .expect("an empirical law is a law"),
    )
}

fn family() -> SurvivalMarginalSlopeFamily {
    let n = N_ROWS;
    let half = (N_ROWS / 2) as f64;
    let age_entry = Array1::from_shape_fn(n, |row| 0.25 + 0.10 * tie(row));
    let age_exit = Array1::from_shape_fn(n, |row| 0.25 + 0.10 * tie(row) + 0.75 + 0.03 * tie(row));
    let event = Array1::from_shape_fn(n, |row| if (row % 12) % 3 == 1 { 1.0 } else { 0.0 });
    let weights = Array1::from_shape_fn(n, |row| 0.5 + ((row % 12) % 5) as f64 * 0.1);
    let z = Array1::from_shape_fn(n, |row| -1.0 + 2.0 * ((7 * (row % 12) + 5) % 12) as f64 / half);
    let config = SurvivalBaselineConfig {
        target: SurvivalBaselineTarget::GompertzMakeham,
        scale: None,
        shape: Some(0.08),
        rate: Some(0.22),
        makeham: Some(0.04),
    };
    let geometry = Arc::new(
        build_survival_marginal_slope_baseline_geometry(&age_entry, &age_exit, &config)
            .expect("build baseline geometry")
            .expect("Gompertz-Makeham has a nonlinear baseline chart"),
    );
    let marginal = Array2::from_shape_fn((n, 2), |(row, col)| {
        let t = (tie(row) + 0.5) / half;
        if col == 0 { 0.30 + 0.40 * t } else { -0.20 + 0.55 * (2.3 * t).sin() }
    });
    let slope = Array2::from_shape_fn((n, 2), |(row, col)| {
        let t = (tie(row) + 0.5) / half;
        if col == 0 { 0.20 + 0.50 * t } else { 0.15 * (1.7 * t + 0.4).cos() }
    });
    SurvivalMarginalSlopeFamily {
        jeffreys_armed: true,
        latent_law: Some(skewed_law()),
        n,
        event: Arc::new(event),
        weights: Arc::new(weights),
        z: Arc::new(z.insert_axis(Axis(1))),
        score_covariance: ScoreCovarianceField::pooled(
            MarginalSlopeCovariance::diagonal(ndarray::array![1.0]).expect("a unit covariance"),
        ),
        gaussian_frailty_sd: None,
        family_hyper: SurvivalMarginalSlopeFamilyHyperState::new(Some(Arc::clone(&geometry)), None)
            .expect("install the baseline chart"),
        derivative_guard: 1e-8,
        design_entry: DesignMatrix::from(Array2::zeros((n, 0))),
        design_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        offset_entry: Arc::new(geometry.offset_entry.clone()),
        offset_exit: Arc::new(geometry.offset_exit.clone()),
        derivative_offset_exit: Arc::new(geometry.derivative_offset_exit.clone()),
        entry_at_origin: Arc::new(Array1::from_elem(n, false)),
        marginal_design: DesignMatrix::from(marginal),
        slope_layout: (DesignMatrix::from(slope)).into(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
    }
}

fn blockspec(name: &str, cols: usize) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((1, cols)))),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(Array1::zeros(cols)),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

/// The bits of `(log-likelihood, gradient, Hessian)` at `(β_m, β_g)`.
fn evaluate(family: &SurvivalMarginalSlopeFamily, marginal: &Array1<f64>, slope: &Array1<f64>) -> Vec<u64> {
    let m_design = family.marginal_design.to_dense().to_owned();
    let g_design = family.slope_layout.coefficient_design().to_dense().to_owned();
    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(family.n),
        },
        ParameterBlockState {
            eta: m_design.dot(marginal),
            beta: marginal.clone(),
        },
        ParameterBlockState {
            eta: g_design.dot(slope),
            beta: slope.clone(),
        },
    ];
    let specs = vec![blockspec("time", 0), blockspec("marginal", 2), blockspec("slope", 2)];
    let evaluation = family
        .exact_newton_joint_gradient_evaluation(&states, &specs)
        .expect("joint gradient evaluation")
        .expect("the anchored family publishes a joint gradient evaluation");
    let hessian = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("the anchored family publishes an explicit joint hessian");
    std::iter::once(evaluation.log_likelihood)
        .chain(evaluation.gradient.iter().copied())
        .chain(hessian.iter().copied())
        .map(f64::to_bits)
        .collect()
}

#[test]
fn anchored_family_arithmetic_does_not_depend_on_thread_scheduling_2983() {
    let target = (ndarray::array![0.35, -0.18], ndarray::array![0.22, 0.13]);
    let start = (ndarray::array![0.05, 0.12], ndarray::array![0.52, -0.17]);
    // An inner-solve-like walk from `start` onto `target`, every step a new
    // equation per row, so each root after the first had a stored neighbour.
    let walk: Vec<(Array1<f64>, Array1<f64>)> = (0..=8)
        .map(|k| {
            let s = f64::from(k) / 8.0;
            (
                &start.0 + &((&target.0 - &start.0) * s),
                &start.1 + &((&target.1 - &start.1) * s),
            )
        })
        .collect();
    let fresh_at_target = || {
        let family = family();
        evaluate(&family, &target.0, &target.1)
    };
    let walked_to_target = || {
        let family = family();
        let mut last = Vec::new();
        for (marginal, slope) in &walk {
            last = evaluate(&family, marginal, slope);
        }
        last
    };
    let pool = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a rayon pool")
    };
    let (twelve, four, one) = (pool(12), pool(4), pool(1));
    for (history, evaluation) in [
        ("fresh", &fresh_at_target as &(dyn Fn() -> Vec<u64> + Sync)),
        ("walked to the iterate", &walked_to_target),
    ] {
        let reference = twelve.install(evaluation);
        assert!(reference.len() > 1, "the evaluation publishes a value and derivatives");
        for (label, bits) in [
            ("12 threads again", twelve.install(evaluation)),
            ("4 threads", four.install(evaluation)),
            ("1 thread", one.install(evaluation)),
            ("12 threads a third time", twelve.install(evaluation)),
        ] {
            let first = reference.iter().zip(&bits).position(|(a, b)| a != b);
            assert_eq!(
                first,
                None,
                "{history}, {label}: entry {first:?} differs from the same history's 12-thread \
                 evaluation ({} entries: value, gradient, Hessian)",
                reference.len()
            );
            assert_eq!(
                bits.len(),
                reference.len(),
                "{history}, {label}: a different number of entries"
            );
        }
    }
}

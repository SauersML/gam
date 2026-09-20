//! One Duchon penalty structure for every family.
//!
//! The standard materializer used to strip the Duchon mass and tension
//! operator penalties from every fit that was not Gaussian-identity (#1074),
//! on the claim that fixed-dispersion LAML "mis-rewards" those blocks. The
//! criterion gives that claim no footing: the fixed-dispersion arm keeps
//! `log|H|` and `−½ log|S|₊` exactly as the profiled Gaussian arm does, so an
//! unsupported block rails its `λ` under either. What the gate did in
//! practice was switch off `scale_dimensions` for GLMs: that flag promotes
//! `s(x1, x2)` to an anisotropic s=0 Duchon whose per-axis relevance is carried
//! only by the tension-ARD penalties the gate removed.
//!
//! Two checks, on a Poisson and a binomial response with an anisotropic truth:
//!
//! 1. The formula path keeps the mass + tension operator penalties for a GLM
//!    `scale_dimensions` fit (this is what failed with the gate in place).
//! 2. On the same data and the same basis, the fit with the operator penalties
//!    converges and recovers the true linear predictor at least as well as the
//!    fit without them (the structure the gate imposed), averaged over seeds.

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, OperatorPenaltySpec, SpatialIdentifiability,
};
use gam::estimate::FitOptions;
use gam::matrix::LinearOperator;
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, TermCollectionSpec,
    build_term_collection_design, fit_term_collection_forspec, freeze_term_collection_from_design,
};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::solver::fit_orchestration::{StandardFitOptionsInputs, canonical_standard_fit_options};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Poisson};

const N_TRAIN: usize = 600;
const N_EVAL: usize = 400;
const N_SEEDS: u64 = 3;
const N_CENTERS: usize = 40;

#[derive(Clone, Copy, Debug)]
enum Glm {
    Poisson,
    Binomial,
}

impl Glm {
    fn name(self) -> &'static str {
        match self {
            Glm::Poisson => "poisson",
            Glm::Binomial => "binomial",
        }
    }

    fn likelihood(self) -> LikelihoodSpec {
        match self {
            Glm::Poisson => LikelihoodSpec::new(
                ResponseFamily::Poisson,
                InverseLink::Standard(StandardLink::Log),
            ),
            Glm::Binomial => LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Logit),
            ),
        }
    }

    /// The true linear predictor: a wiggly effect along `x1`, a gentle one
    /// along `x2`, scaled onto a moderate range of the link.
    fn eta(self, x1: f64, x2: f64) -> f64 {
        let signal = (2.0 * std::f64::consts::PI * x1).sin() + 0.3 * x2;
        match self {
            Glm::Poisson => 0.5 + 0.6 * signal,
            Glm::Binomial => 1.2 * signal,
        }
    }

    fn draw(self, eta: f64, rng: &mut StdRng) -> f64 {
        match self {
            Glm::Poisson => Poisson::new(eta.exp())
                .expect("positive Poisson rate")
                .sample(rng),
            Glm::Binomial => f64::from(u8::from(rng.random::<f64>() < 1.0 / (1.0 + (-eta).exp()))),
        }
    }
}

fn uniform_points(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((n, 2), |_| rng.random::<f64>())
}

/// Columns `x1, x2, y`, matching the formula dataset's column order.
fn simulate(glm: Glm, seed: u64) -> Array2<f64> {
    let x = uniform_points(N_TRAIN, seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0x9e37_79b9);
    let mut data = Array2::<f64>::zeros((N_TRAIN, 3));
    for i in 0..N_TRAIN {
        let (x1, x2) = (x[[i, 0]], x[[i, 1]]);
        data[[i, 0]] = x1;
        data[[i, 1]] = x2;
        data[[i, 2]] = glm.draw(glm.eta(x1, x2), &mut rng);
    }
    data
}

/// The anisotropic s=0 Duchon that `scale_dimensions` promotes a 2-D
/// thin-plate `s(x1, x2)` to (`promote_thin_plate_for_scale_dimensions`),
/// with the given operator penalties.
fn promoted_duchon_spec(operator_penalties: DuchonOperatorPenaltySpec) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "s(x1,x2)".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: vec![0, 1],
                spec: DuchonBasisSpec {
                    radial_reparam: None,
                    center_strategy: CenterStrategy::FarthestPoint {
                        num_centers: N_CENTERS,
                    },
                    length_scale: None,
                    power: 0.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::default(),
                    aniso_log_scales: Some(vec![0.0; 2]),
                    operator_penalties,
                    periodic: None,
                    boundary: OneDimensionalBoundary::Open,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None.into(),
            joint_null_rotation: None,
        }],
        level: Default::default(),
    }
}

/// The options every formula fit uses, so both arms run under the formula
/// path's own policy.
fn fit_options() -> FitOptions {
    canonical_standard_fit_options(&FitConfig::default(), StandardFitOptionsInputs::default())
}

/// Held-out RMSE of the fitted linear predictor against the truth. A returned
/// fit is the sealed convergence certificate (SPEC 20), so reaching the
/// prediction means the outer search converged.
fn eta_rmse(glm: Glm, data: &Array2<f64>, spec: &TermCollectionSpec, label: &str) -> f64 {
    let n = data.nrows();
    let fitted = fit_term_collection_forspec(
        data.view(),
        data.column(2),
        Array1::ones(n).view(),
        Array1::zeros(n).view(),
        spec,
        glm.likelihood(),
        &fit_options(),
    )
    .unwrap_or_else(|err| panic!("{} {label} fit must converge: {err}", glm.name()));
    let frozen = freeze_term_collection_from_design(spec, &fitted.design)
        .expect("freezing the trained spec must succeed");
    let x_eval = uniform_points(N_EVAL, 999);
    let mut grid = Array2::<f64>::zeros((N_EVAL, 3));
    grid.column_mut(0).assign(&x_eval.column(0));
    grid.column_mut(1).assign(&x_eval.column(1));
    let design = build_term_collection_design(grid.view(), &frozen)
        .expect("held-out design build must succeed");
    let eta_hat = design.design.apply(&fitted.fit.beta);
    let sq: f64 = (0..N_EVAL)
        .map(|i| {
            let err = eta_hat[i] - glm.eta(x_eval[[i, 0]], x_eval[[i, 1]]);
            err * err
        })
        .sum();
    (sq / N_EVAL as f64).sqrt()
}

fn assert_formula_path_keeps_operator_penalties(glm: Glm, data: &Array2<f64>) {
    let headers = ["x1", "x2", "y"].into_iter().map(String::from).collect();
    let rows = data
        .outer_iter()
        .map(|row| csv::StringRecord::from(row.iter().map(f64::to_string).collect::<Vec<_>>()))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode synthetic dataset");
    let cfg = FitConfig {
        family: Some(glm.name().to_string()),
        scale_dimensions: true,
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x1, x2)", &ds, &cfg)
        .unwrap_or_else(|err| panic!("{} scale_dimensions formula fit: {err}", glm.name()));
    let FitResult::Standard(fit) = result else {
        panic!("{} s(x1, x2) must be a standard GAM fit", glm.name());
    };
    let ops = fit
        .resolvedspec
        .smooth_terms
        .iter()
        .find_map(|term| match &term.basis {
            SmoothBasisSpec::Duchon { spec, .. } => Some(spec.operator_penalties.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{} scale_dimensions must promote to a Duchon", glm.name()));
    assert!(
        matches!(ops.mass, OperatorPenaltySpec::Active { .. })
            && matches!(ops.tension, OperatorPenaltySpec::Active { .. })
            && matches!(ops.stiffness, OperatorPenaltySpec::Disabled),
        "{} scale_dimensions fit must keep the mass + tension operator penalties that carry \
         the per-axis relevance; got {ops:?}",
        glm.name()
    );
}

#[test]
fn glm_duchon_operator_penalties_are_kept_and_recover_truth() {
    init_parallelism();
    for glm in [Glm::Poisson, Glm::Binomial] {
        assert_formula_path_keeps_operator_penalties(glm, &simulate(glm, 0));

        let with_ops = promoted_duchon_spec(DuchonOperatorPenaltySpec::default());
        let without_ops = promoted_duchon_spec(DuchonOperatorPenaltySpec::all_disabled());
        let mut rmse_with = Vec::new();
        let mut rmse_without = Vec::new();
        for seed in 0..N_SEEDS {
            let data = simulate(glm, seed);
            rmse_with.push(eta_rmse(glm, &data, &with_ops, "with operator penalties"));
            rmse_without.push(eta_rmse(glm, &data, &without_ops, "without operator penalties"));
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        assert!(
            mean(&rmse_with) <= mean(&rmse_without),
            "{}: keeping the Duchon operator penalties must recover the true linear predictor \
             at least as well as dropping them; held-out RMSE(eta) with {rmse_with:?} vs \
             without {rmse_without:?}",
            glm.name()
        );
    }
}

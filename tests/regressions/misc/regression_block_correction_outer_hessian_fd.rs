//! The #784 block-local correction `Δ_b` carries its exact analytic outer
//! ρ-Hessian, and the criterion's Hessian agrees with central differences of
//! its own gradient wherever the correction is engaged.
//!
//! Before this, a criterion carrying `Δ_b` declared no analytic Hessian: the
//! outer search ran on BFGS curvature and the smoothing-corrected covariance
//! had no exact `H_ρ` to invert, so a certified rare-event binomial fit
//! (pyGAM audit F17) could not ship its corrected covariance. The Hessian
//! of `−Δ_b` is now spliced into the criterion's (`block_correction_hessian`),
//! and this test grades it on the three shapes the correction takes:
//!
//! * a one-coefficient block on a sparse binomial cell (`m = 1`, #2623);
//! * the F17 rare-event binomial `y ~ factor(student) + s(balance) + s(income)`
//!   at its certified optimum, whose fit must also ship the corrected
//!   covariance;
//! * a Poisson tensor fit whose block is split axis by axis (`m ≥ 2`), where
//!   the mixed-axis term `Ψ` carries second-order ρ-motion of its own;
//! * an inverse-Gaussian fit on its canonical `1/μ²` link, whose one-axis block
//!   is truncated to the feasible interval `η > 0`: its ends, and with them the
//!   transported nodes and the interval's mass `ln Z`, move with ρ.
//!
//! Each stencil evaluation builds a fresh criterion, which latches its own
//! block. Every point must engage the correction on the same block columns
//! with the same axis orders, or the quotient differences two different
//! functions and is not a derivative of either.

use csv::StringRecord;
use gam::matrix::DesignMatrix;
use gam::estimate::outer_eval_capture::{enable_rho_outer_audit, take_rho_outer_audit};
use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost, evaluate_externalgradient, evaluate_externalhessian,
    optimize_external_designwith_heuristic_log_lambdas,
};
use gam::smooth::{BlockwisePenalty, build_term_collection_design};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{FitConfig, FitRequest, encode_recordswith_inferred_schema, init_parallelism};
use ndarray::{Array1, Array2, array};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson, Uniform};
use std::f64::consts::PI;

struct Criterion {
    y: Array1<f64>,
    weights: Array1<f64>,
    design: DesignMatrix,
    offset: Array1<f64>,
    penalties: Vec<BlockwisePenalty>,
    opts: ExternalOptimOptions,
}

/// The block the correction engaged on at one evaluation.
#[derive(Debug, PartialEq)]
struct EngagedBlock {
    block_cols: Vec<usize>,
    axis_orders: Vec<usize>,
    /// Per piece, whether it was integrated on a feasible interval with a
    /// finite end.
    truncated_pieces: Vec<bool>,
    /// Per piece, whether its latched rule is the composite Gauss–Kronrod
    /// partition.
    composite_pieces: Vec<bool>,
}

fn engaged_block(label: &str, rho: &Array1<f64>) -> EngagedBlock {
    let audit = take_rho_outer_audit().expect("rho audit armed");
    let record = audit.quadrature_marginal.unwrap_or_else(|| {
        panic!(
            "[{label}] the #784 correction must engage at rho {:?}; a decline makes this \
             test vacuous",
            rho.to_vec()
        )
    });
    EngagedBlock {
        block_cols: record.block_cols,
        axis_orders: record.axis_orders,
        truncated_pieces: record.truncated_pieces,
        composite_pieces: record.composite_pieces,
    }
}

impl Criterion {
    fn cost(&self, label: &str, rho: &Array1<f64>) -> (f64, EngagedBlock) {
        enable_rho_outer_audit();
        let cost = evaluate_externalcost(
            self.y.view(),
            self.weights.view(),
            self.design.clone(),
            self.offset.view(),
            &self.penalties,
            &self.opts,
            rho,
        )
        .unwrap_or_else(|error| panic!("[{label}] cost at {:?}: {error}", rho.to_vec()));
        (cost, engaged_block(label, rho))
    }

    fn gradient(&self, label: &str, rho: &Array1<f64>) -> (Array1<f64>, EngagedBlock) {
        enable_rho_outer_audit();
        let gradient = evaluate_externalgradient(
            self.y.view(),
            self.weights.view(),
            self.design.clone(),
            self.offset.view(),
            &self.penalties,
            &self.opts,
            rho,
        )
        .unwrap_or_else(|error| panic!("[{label}] gradient at {:?}: {error}", rho.to_vec()));
        (gradient, engaged_block(label, rho))
    }

    fn hessian(&self, label: &str, rho: &Array1<f64>) -> (Array2<f64>, EngagedBlock) {
        enable_rho_outer_audit();
        let hessian = evaluate_externalhessian(
            self.y.view(),
            self.weights.view(),
            self.design.clone(),
            self.offset.view(),
            &self.penalties,
            &self.opts,
            rho,
        )
        .unwrap_or_else(|error| {
            panic!(
                "[{label}] a criterion carrying the #784 correction must have its exact \
                 rho-Hessian at {:?}: {error}",
                rho.to_vec()
            )
        });
        (hessian, engaged_block(label, rho))
    }

    /// Grades the analytic gradient at `rho` against central differences of the
    /// cost, relative to the gradient's scale.
    fn assert_gradient_matches_cost_differences(&self, label: &str, rho: &Array1<f64>) {
        const STEP: f64 = 1.0e-4;
        const RELATIVE_TOLERANCE: f64 = 1.0e-4;
        let (analytic, block) = self.gradient(label, rho);
        let mut differenced = Array1::<f64>::zeros(rho.len());
        for j in 0..rho.len() {
            let mut plus = rho.clone();
            plus[j] += STEP;
            let mut minus = rho.clone();
            minus[j] -= STEP;
            let (cost_plus, block_plus) = self.cost(label, &plus);
            let (cost_minus, block_minus) = self.cost(label, &minus);
            assert_eq!(
                block_plus, block, "[{label}] +{STEP} on rho[{j}] engaged a different block"
            );
            assert_eq!(
                block_minus, block, "[{label}] -{STEP} on rho[{j}] engaged a different block"
            );
            differenced[j] = (cost_plus - cost_minus) / (2.0 * STEP);
        }
        let scale = analytic
            .iter()
            .fold(1.0_f64, |acc, value| acc.max(value.abs()));
        let worst = analytic
            .iter()
            .zip(differenced.iter())
            .fold(0.0_f64, |acc, (a, d)| acc.max((a - d).abs()));
        eprintln!(
            "[{label}] rho={:?} gradient={:?} differenced={:?} max|g - FD|={worst:.3e}",
            rho.to_vec(),
            analytic.to_vec(),
            differenced.to_vec(),
        );
        assert!(
            worst <= RELATIVE_TOLERANCE * scale,
            "[{label}] analytic gradient disagrees with central differences of the cost: \
             max|g - FD| = {worst:.3e} against scale {scale:.3e}"
        );
    }

    /// Grades the analytic Hessian at `rho` against central differences of the
    /// analytic gradient, column by column, relative to the Hessian's scale.
    fn assert_hessian_matches_gradient_differences(&self, label: &str, rho: &Array1<f64>) {
        const STEP: f64 = 1.0e-4;
        const RELATIVE_TOLERANCE: f64 = 1.0e-4;
        let (analytic, block) = self.hessian(label, rho);
        let k = rho.len();
        assert_eq!(analytic.dim(), (k, k), "[{label}] Hessian shape");
        let mut differenced = Array2::<f64>::zeros((k, k));
        for j in 0..k {
            let mut plus = rho.clone();
            plus[j] += STEP;
            let mut minus = rho.clone();
            minus[j] -= STEP;
            let (gradient_plus, block_plus) = self.gradient(label, &plus);
            let (gradient_minus, block_minus) = self.gradient(label, &minus);
            assert_eq!(
                block_plus, block, "[{label}] +{STEP} on rho[{j}] engaged a different block"
            );
            assert_eq!(
                block_minus, block, "[{label}] -{STEP} on rho[{j}] engaged a different block"
            );
            for i in 0..k {
                differenced[[i, j]] = (gradient_plus[i] - gradient_minus[i]) / (2.0 * STEP);
            }
        }
        let scale = analytic
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let worst = analytic
            .iter()
            .zip(differenced.iter())
            .fold(0.0_f64, |acc, (a, d)| acc.max((a - d).abs()));
        eprintln!(
            "[{label}] rho={:?} block_cols={:?} axis_orders={:?} max|H|={scale:.3e} \
             max|H - FD|={worst:.3e}\n  analytic={analytic:.6e}\n  differenced={differenced:.6e}",
            rho.to_vec(),
            block.block_cols,
            block.axis_orders,
        );
        assert!(
            analytic.iter().all(|value| value.is_finite()),
            "[{label}] the analytic Hessian is not finite"
        );
        assert!(
            worst <= RELATIVE_TOLERANCE * scale,
            "[{label}] analytic Hessian disagrees with central differences of the gradient: \
             max|H - FD| = {worst:.3e} against max|H| = {scale:.3e}"
        );
    }
}

fn standard_options(family: LikelihoodSpec) -> ExternalOptimOptions {
    ExternalOptimOptions {
        family,
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        // The formula/CLI policy (`canonical_standard_fit_options`): the
        // smoothing-corrected covariance comes from the outer Hessian, not
        // from the Tier-1/Tier-2 rho-posterior escalation this flag gates.
        skip_rho_posterior_inference: true,
        max_iter: 300,
        tol: 1.0e-10,
        nullspace_dims: Vec::new(),
        linear_constraints: None,
        firth_bias_reduction: Some(false),
        rho_prior: Default::default(),
        persistent_warm_start_store: None,
    }
}

fn logit() -> LikelihoodSpec {
    LikelihoodSpec::new(
        ResponseFamily::Binomial,
        InverseLink::Standard(StandardLink::Logit),
    )
}

/// Builds the external criterion of `formula` on `data` and fits it, returning
/// the criterion with the certified optimum.
fn fitted_formula_criterion(
    label: &str,
    formula: &str,
    family_name: &str,
    data: &gam::data::EncodedDataset,
) -> (Criterion, Array1<f64>) {
    let config = FitConfig {
        family: Some(family_name.to_string()),
        ..FitConfig::default()
    };
    let model = gam::materialize(formula, data, &config).expect("materialize");
    let FitRequest::Standard(request) = model.request else {
        panic!("[{label}] `{formula}` is a standard request");
    };
    let design = build_term_collection_design(request.data.view(), &request.spec)
        .expect("build the design");
    let mut opts = standard_options(request.family.clone());
    opts.nullspace_dims = design.nullspace_dims.clone();
    opts.linear_constraints = design.linear_constraints.clone();
    let offset = &*request.offset + &design.affine_offset;
    let fit = optimize_external_designwith_heuristic_log_lambdas(
        request.y.view(),
        request.weights.view(),
        design.design.clone(),
        offset.view(),
        design.penalties.clone(),
        None,
        &opts,
    )
    .unwrap_or_else(|error| panic!("[{label}] `{formula}` must fit: {error}"));
    assert!(fit.outer_converged, "[{label}] the outer search must certify");
    let corrected = fit.covariance_corrected.as_ref().unwrap_or_else(|| {
        panic!("[{label}] the certified fit shipped no smoothing-corrected covariance")
    });
    assert!(
        corrected.iter().all(|value| value.is_finite()),
        "[{label}] the smoothing-corrected covariance is not finite"
    );
    let criterion = Criterion {
        y: (*request.y).clone(),
        weights: (*request.weights).clone(),
        design: design.design.clone(),
        offset,
        penalties: design.penalties.clone(),
        opts,
    };
    (criterion, fit.log_lambdas.clone())
}

#[test]
fn sparse_binomial_single_coefficient_block_hessian_matches_differences() {
    init_parallelism();
    let observations = 12usize;
    let mut y = Array1::<f64>::zeros(observations);
    y[0] = 1.0;
    let mut opts = standard_options(logit());
    opts.nullspace_dims = vec![0];
    let criterion = Criterion {
        y,
        weights: Array1::ones(observations),
        design: Array2::<f64>::ones((observations, 1)).into(),
        offset: Array1::zeros(observations),
        penalties: vec![BlockwisePenalty::new(0..1, array![[1.0]])],
        opts,
    };
    for rho in [-5.0, -3.0] {
        let rho = array![rho];
        let (_, block) = criterion.hessian("sparse binomial", &rho);
        assert!(
            block.composite_pieces.iter().any(|&composite| composite),
            "the untruncated one-axis block must integrate on its latched composite \
             Gauss–Kronrod partition, or the composite rule's Hessian is not what this \
             test exercised: {block:?}"
        );
        criterion.assert_gradient_matches_cost_differences("sparse binomial", &rho);
        criterion.assert_hessian_matches_gradient_differences("sparse binomial", &rho);
    }
}

/// The F17 analogue of ISLR `default` (see
/// `regression_rare_event_binomial_block_eigensystem`).
fn default_like(seed: u64, n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");
    let balance_law = Normal::<f64>::new(835.0, 480.0).expect("balance law");
    let student_income = Normal::<f64>::new(17_500.0, 4_500.0).expect("student income law");
    let other_income = Normal::<f64>::new(40_000.0, 10_000.0).expect("income law");
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for _ in 0..n {
        let student = unit.sample(&mut rng) < 0.3;
        let balance = balance_law.sample(&mut rng).max(0.0);
        let income = if student {
            student_income.sample(&mut rng)
        } else {
            other_income.sample(&mut rng)
        }
        .max(700.0);
        let eta = -10.8 + 0.0057 * balance - if student { 0.65 } else { 0.0 };
        let y = unit.sample(&mut rng) < 1.0 / (1.0 + (-eta).exp());
        rows.push(StringRecord::from(vec![
            if student { "Yes" } else { "No" }.to_string(),
            balance.to_string(),
            income.to_string(),
            if y { "1" } else { "0" }.to_string(),
        ]));
    }
    let headers = ["student", "balance", "income", "y"]
        .into_iter()
        .map(String::from)
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode the fixture")
}

#[test]
fn rare_event_binomial_block_hessian_matches_differences_at_its_optimum() {
    init_parallelism();
    let data = default_like(1, 2000);
    let (criterion, optimum) = fitted_formula_criterion(
        "rare-event binomial",
        "y ~ factor(student) + s(balance) + s(income)",
        "binomial-logit",
        &data,
    );
    criterion.assert_gradient_matches_cost_differences("rare-event binomial", &optimum);
    criterion.assert_hessian_matches_gradient_differences("rare-event binomial", &optimum);
}

/// The Poisson speed cell's generator (see
/// `regression_poisson_te_block_axis_split`), whose optimum admits a block of
/// three axes.
fn tensor_poisson_dataset(seed: u64, n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for _ in 0..n {
        let x0 = unit.sample(&mut rng);
        let x1 = unit.sample(&mut rng);
        let eta = 0.5 + 0.7 * (2.0 * PI * x0).sin() * (2.0 * PI * x1).cos();
        let count: f64 = Poisson::new(eta.exp())
            .expect("a finite positive Poisson rate")
            .sample(&mut rng);
        rows.push(StringRecord::from(vec![
            x0.to_string(),
            x1.to_string(),
            count.to_string(),
        ]));
    }
    let headers = ["x0", "x1", "y"].into_iter().map(String::from).collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode the fixture")
}

#[test]
fn poisson_tensor_axis_split_block_hessian_matches_differences_at_its_optimum() {
    init_parallelism();
    let data = tensor_poisson_dataset(0, 2000);
    let (criterion, optimum) = fitted_formula_criterion(
        "poisson te",
        "y ~ te(x0, x1)",
        "poisson",
        &data,
    );
    enable_rho_outer_audit();
    let (_, block) = criterion.hessian("poisson te", &optimum);
    assert!(
        block.axis_orders.len() >= 2,
        "the fixture must engage a multi-axis block, or the mixed-axis term's Hessian is \
         not what this test exercised: {block:?}"
    );
    criterion.assert_gradient_matches_cost_differences("poisson te", &optimum);
    criterion.assert_hessian_matches_gradient_differences("poisson te", &optimum);
}

/// The sweep's n = 100 `p1` inverse-Gaussian cell (see
/// `crates/gam-models/tests/inverse_gaussian_canonical_block_quadrature.rs`).
const INVERSE_GAUSSIAN_P1_N100: &str = include_str!(
    "../../../crates/gam-models/tests/fixtures/inverse_gaussian_canonical_p1_n100.csv"
);

#[test]
fn inverse_gaussian_truncated_block_hessian_matches_differences_at_its_optimum() {
    init_parallelism();
    let mut reader = csv::Reader::from_reader(INVERSE_GAUSSIAN_P1_N100.as_bytes());
    let headers: Vec<String> = reader
        .headers()
        .expect("fixture header")
        .iter()
        .map(str::to_string)
        .collect();
    let rows: Vec<StringRecord> = reader
        .records()
        .map(|record| record.expect("fixture row"))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode the fixture");
    let (criterion, optimum) = fitted_formula_criterion(
        "inverse gaussian",
        "y ~ s(x0)",
        "inverse-gaussian",
        &data,
    );
    assert_eq!(
        criterion.opts.family.link,
        InverseLink::Standard(StandardLink::InverseSquared),
        "the fixture must fit the canonical link, whose block is truncated"
    );
    let (_, block) = criterion.hessian("inverse gaussian", &optimum);
    assert!(
        block.truncated_pieces.iter().any(|&truncated| truncated),
        "the fixture must integrate its block on a feasible interval with a finite end, or \
         the moving ends, transported nodes and ln Z are not what this test exercised: \
         {block:?}"
    );
    criterion.assert_gradient_matches_cost_differences("inverse gaussian", &optimum);
    criterion.assert_hessian_matches_gradient_differences("inverse gaussian", &optimum);
}

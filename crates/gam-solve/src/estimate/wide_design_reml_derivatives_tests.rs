//! REML/LAML at `p > n`: the criterion, its gradient and its Hessian on a
//! design with more coefficients than rows.
//!
//! A penalized GAM is identified by its sample size when `n > M_p`, the
//! dimension of the unpenalized space, not when `n > p`. Every penalized
//! direction is pinned by its penalty, so the inner system `XᵀWX + S_λ` is
//! positive definite however wide `X` is, and the Laplace criterion is taken
//! over the penalized range: `½·log|XᵀWX + S_λ| − ½·log|S_λ|₊`, with the
//! Gaussian scale profiled from the `n − M_p` residual contrasts.
//!
//! The fixture is six double-penalized cubic B-spline smooths of ten columns
//! each plus an intercept: `p = 61` columns against `n = 40` rows, `M_p = 1`.
//! Each block is centred, so it also carries an exact null direction of `X`
//! (the within-block constant) that only its null-space penalty identifies.
//! These tests pin, on that design,
//!
//! * the criterion's value against the textbook formula assembled here from a
//!   dense Cholesky of the same `XᵀWX + S_λ` and `S_λ` (compared as differences
//!   between two `ρ`, which cancels every additive constant and the intercept
//!   column's conditioning scale);
//! * the analytic gradient against a central difference of the value;
//! * the analytic Hessian against a central difference of the gradient;
//!
//! for the Gaussian profiled REML and the binomial-logit LAML.
#![cfg(test)]

use super::*;
use gam_problem::solver_contract::OuterEval;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_terms::construction::CanonicalPenalty;
use ndarray::{Array1, Array2};

const ROWS: usize = 40;
const SMOOTHS: usize = 6;
const BASIS: usize = 10;
const COLUMNS: usize = 1 + SMOOTHS * BASIS;

/// SplitMix64: a dependency-free deterministic stream for the fixture.
struct SplitMix(u64);

impl SplitMix {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.unit();
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Cubic B-spline design on `[0, 1]` with `n_basis` uniform-knot bases.
fn cubic_bspline_design(xs: &[f64], n_basis: usize) -> Array2<f64> {
    const DEGREE: usize = 3;
    let n_knots = n_basis + DEGREE + 1;
    let n_internal = n_basis - (DEGREE + 1);
    let knots: Vec<f64> = (0..n_knots)
        .map(|k| {
            if k <= DEGREE {
                0.0
            } else if k >= n_knots - DEGREE - 1 {
                1.0
            } else {
                (k - DEGREE) as f64 / (n_internal as f64 + 1.0)
            }
        })
        .collect();
    fn bspline(i: usize, degree: usize, x: f64, knots: &[f64]) -> f64 {
        if degree == 0 {
            let last = x == 1.0 && knots[i + 1] == 1.0 && knots[i] < knots[i + 1];
            return if (knots[i] <= x && x < knots[i + 1]) || last {
                1.0
            } else {
                0.0
            };
        }
        let mut value = 0.0;
        let left = knots[i + degree] - knots[i];
        if left > 0.0 {
            value += (x - knots[i]) / left * bspline(i, degree - 1, x, knots);
        }
        let right = knots[i + degree + 1] - knots[i + 1];
        if right > 0.0 {
            value += (knots[i + degree + 1] - x) / right * bspline(i + 1, degree - 1, x, knots);
        }
        value
    }
    Array2::from_shape_fn((xs.len(), n_basis), |(row, col)| {
        bspline(col, DEGREE, xs[row], &knots)
    })
}

/// The two penalties of one double-penalized block: the second-difference
/// penalty `DᵀD` and the projector onto its null space `span{1, t}`.
fn double_penalty(k: usize) -> [Array2<f64>; 2] {
    let mut d = Array2::<f64>::zeros((k - 2, k));
    for row in 0..k - 2 {
        d[[row, row]] = 1.0;
        d[[row, row + 1]] = -2.0;
        d[[row, row + 2]] = 1.0;
    }
    let ones = Array1::from_elem(k, 1.0 / (k as f64).sqrt());
    let mut t = Array1::from_iter((0..k).map(|i| i as f64));
    t -= t.mean().expect("nonempty");
    t /= t.dot(&t).sqrt();
    let null_projector =
        Array2::from_shape_fn((k, k), |(i, j)| ones[i] * ones[j] + t[i] * t[j]);
    [d.t().dot(&d), null_projector]
}

/// The wide design, its response, and its twelve penalties.
fn fixture(family: &ResponseFamily) -> (Array2<f64>, Array1<f64>, Vec<PenaltySpec>) {
    let mut rng = SplitMix(2026);
    let covariates: Vec<Vec<f64>> = (0..SMOOTHS)
        .map(|_| (0..ROWS).map(|_| rng.unit()).collect())
        .collect();
    let mut x = Array2::<f64>::zeros((ROWS, COLUMNS));
    x.column_mut(0).fill(1.0);
    let mut specs = Vec::with_capacity(2 * SMOOTHS);
    for (smooth, covariate) in covariates.iter().enumerate() {
        let mut block = cubic_bspline_design(covariate, BASIS);
        for mut column in block.columns_mut() {
            let mean = column.mean().expect("nonempty");
            column -= mean;
        }
        let start = 1 + smooth * BASIS;
        x.slice_mut(ndarray::s![.., start..start + BASIS])
            .assign(&block);
        for local in double_penalty(BASIS) {
            specs.push(PenaltySpec::Block {
                local,
                col_range: start..start + BASIS,
                prior_mean: gam_problem::CoefficientPriorMean::Zero,
                structure_hint: None,
                op: None,
            });
        }
    }
    let y = Array1::from_iter(covariates[0].iter().map(|&x0| {
        let signal = (2.0 * std::f64::consts::PI * x0).sin();
        match family {
            ResponseFamily::Binomial => {
                let prob = 1.0 / (1.0 + (-1.5 * signal).exp());
                if rng.unit() < prob { 1.0 } else { 0.0 }
            }
            _ => signal + 0.3 * rng.normal(),
        }
    }));
    (x, y, specs)
}

fn likelihood(family: &ResponseFamily) -> LikelihoodSpec {
    let link = match family {
        ResponseFamily::Binomial => StandardLink::Logit,
        _ => StandardLink::Identity,
    };
    LikelihoodSpec::new(family.clone(), InverseLink::Standard(link))
}

/// One REML/LAML state per call, so no evaluation is answered from a bundle
/// cached at another point.
fn with_state<T>(
    family: &ResponseFamily,
    run: impl FnOnce(&RemlState<'_>, &[CanonicalPenalty], &Array2<f64>, &Array1<f64>) -> T,
) -> T {
    let (x, y, specs) = fixture(family);
    let weights = Array1::<f64>::ones(ROWS);
    let offset = Array1::<f64>::zeros(ROWS);
    let ext = ExternalOptimOptions {
        family: likelihood(family),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 300,
        tol: 1e-10,
        nullspace_dims: vec![0; specs.len()],
        linear_constraints: None,
        firth_bias_reduction: Some(false),
        rho_prior: Default::default(),
        persistent_warm_start_store: None,
    };
    let cfg = super::external_options::resolved_external_config(&ext)
        .expect("the external config resolves")
        .0;
    let (canonical, active_nullspace_dims) = gam_terms::construction::canonicalize_penalty_specs(
        &specs,
        &ext.nullspace_dims,
        COLUMNS,
        "wide_design_reml_derivatives",
    )
    .expect("the double penalties canonicalize");
    let x_dm: DesignMatrix = x.clone().into();
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x_dm, &specs);
    let x_fit = conditioning.apply_to_design(&x_dm);
    let mut state = RemlState::newwith_offset(
        y.view(),
        x_fit,
        weights.view(),
        offset.view(),
        canonical.clone(),
        COLUMNS,
        &cfg,
        Some(active_nullspace_dims),
        None,
        None,
    )
    .expect("the REML state builds on a p > n design");
    state.set_rho_prior(ext.rho_prior.clone());
    run(&state, &canonical, &x, &y)
}

fn evaluate(
    family: &ResponseFamily,
    rho: &Array1<f64>,
    order: crate::rho_optimizer::OuterEvalOrder,
) -> OuterEval {
    with_state(family, |state, _, _, _| {
        state
            .compute_outer_eval_with_order(rho, order)
            .expect("the criterion evaluates at p > n")
    })
}

fn cost(family: &ResponseFamily, rho: &Array1<f64>) -> f64 {
    evaluate(family, rho, crate::rho_optimizer::OuterEvalOrder::Value).cost
}

fn gradient(family: &ResponseFamily, rho: &Array1<f64>) -> Array1<f64> {
    evaluate(
        family,
        rho,
        crate::rho_optimizer::OuterEvalOrder::ValueAndGradient,
    )
    .gradient
}

fn hessian(family: &ResponseFamily, rho: &Array1<f64>) -> Array2<f64> {
    let eval = evaluate(
        family,
        rho,
        crate::rho_optimizer::OuterEvalOrder::ValueGradientHessian,
    );
    match eval.hessian {
        gam_problem::HessianValue::Dense(matrix) => matrix,
        gam_problem::HessianValue::Operator(op) => {
            let dim = op.dim();
            let mut matrix = Array2::<f64>::zeros((dim, dim));
            for col in 0..dim {
                let mut unit = Array1::<f64>::zeros(dim);
                unit[col] = 1.0;
                let image = op.apply(&unit).expect("the Hessian operator applies");
                matrix.column_mut(col).assign(&image);
            }
            matrix
        }
        gam_problem::HessianValue::Unavailable => {
            panic!("ValueGradientHessian must return the analytic outer Hessian")
        }
    }
}

/// A probe away from both rails, different in every coordinate so no symmetry
/// of the fixture can hide a transposed or misassigned derivative.
fn probe(shift: f64) -> Array1<f64> {
    Array1::from_iter((0..2 * SMOOTHS).map(|k| shift + 0.4 * ((k * 5) % 7) as f64 - 0.8))
}

/// Dense Cholesky `A = LLᵀ` of a symmetric positive definite matrix.
fn cholesky(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut diag = a[[j, j]];
        for k in 0..j {
            diag -= l[[j, k]] * l[[j, k]];
        }
        assert!(diag > 0.0, "the reference matrix is not positive definite");
        l[[j, j]] = diag.sqrt();
        for i in j + 1..n {
            let mut value = a[[i, j]];
            for k in 0..j {
                value -= l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = value / l[[j, j]];
        }
    }
    l
}

fn logdet(a: &Array2<f64>) -> f64 {
    2.0 * cholesky(a).diag().mapv(f64::ln).sum()
}

fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let l = cholesky(a);
    let n = b.len();
    let mut z = b.clone();
    for i in 0..n {
        for k in 0..i {
            z[i] -= l[[i, k]] * z[k];
        }
        z[i] /= l[[i, i]];
    }
    for i in (0..n).rev() {
        for k in i + 1..n {
            z[i] -= l[[k, i]] * z[k];
        }
        z[i] /= l[[i, i]];
    }
    z
}

/// `S_λ = Σ_k e^{ρ_k} S_k` embedded in the full coefficient space.
fn total_penalty(penalties: &[CanonicalPenalty], rho: &Array1<f64>) -> Array2<f64> {
    let mut s = Array2::<f64>::zeros((COLUMNS, COLUMNS));
    for (penalty, &log_lambda) in penalties.iter().zip(rho.iter()) {
        let range = penalty.col_range.clone();
        let mut block = s.slice_mut(ndarray::s![range.clone(), range]);
        block.scaled_add(log_lambda.exp(), &penalty.local);
    }
    s
}

/// The textbook Laplace criterion at `ρ`, up to an additive constant, taken
/// over the penalized range: the unpenalized intercept is dropped from
/// `log|S_λ|₊`, and the Gaussian scale is profiled from `n − M_p` contrasts.
fn textbook_criterion(
    family: &ResponseFamily,
    penalties: &[CanonicalPenalty],
    x: &Array2<f64>,
    y: &Array1<f64>,
    rho: &Array1<f64>,
) -> f64 {
    let s = total_penalty(penalties, rho);
    let penalized_logdet = logdet(&s.slice(ndarray::s![1.., 1..]).to_owned());
    match family {
        ResponseFamily::Binomial => {
            // Newton on the penalized log-likelihood; logit is canonical, so
            // the observed and expected information agree.
            let mut beta = Array1::<f64>::zeros(COLUMNS);
            let mut information = Array2::<f64>::zeros((COLUMNS, COLUMNS));
            for _ in 0..100 {
                let eta = x.dot(&beta);
                let mu = eta.mapv(|e| 1.0 / (1.0 + (-e).exp()));
                let w = mu.mapv(|m| m * (1.0 - m));
                let weighted = x * &w.view().insert_axis(ndarray::Axis(1));
                information = x.t().dot(&weighted) + &s;
                let score = x.t().dot(&(y - &mu)) - s.dot(&beta);
                let step = solve(&information, &score);
                beta += &step;
                if step.dot(&step).sqrt() <= 1e-13 * (1.0 + beta.dot(&beta).sqrt()) {
                    break;
                }
            }
            let eta = x.dot(&beta);
            let loglik: f64 = eta
                .iter()
                .zip(y.iter())
                .map(|(&e, &yi)| yi * e - (1.0 + e.exp()).ln())
                .sum();
            -loglik + 0.5 * beta.dot(&s.dot(&beta)) + 0.5 * logdet(&information)
                - 0.5 * penalized_logdet
        }
        _ => {
            let h = x.t().dot(x) + &s;
            let beta = solve(&h, &x.t().dot(y));
            let residual = y - &x.dot(&beta);
            let d_p = residual.dot(&residual) + beta.dot(&s.dot(&beta));
            let contrasts = (ROWS - 1) as f64;
            0.5 * contrasts * (d_p / contrasts).ln() + 0.5 * logdet(&h) - 0.5 * penalized_logdet
        }
    }
}

fn relative_error(estimate: &Array1<f64>, reference: &Array1<f64>) -> f64 {
    let diff = estimate - reference;
    diff.dot(&diff).sqrt() / reference.dot(reference).sqrt()
}

const FAMILIES: [ResponseFamily; 2] = [ResponseFamily::Gaussian, ResponseFamily::Binomial];

/// Central-difference step on `ρ`.
const STEP: f64 = 1e-4;

/// A central difference with step `h` carries an `O(h²)` truncation error
/// (`1e-8` here) plus an `O(ε·|V|/h)` rounding error (`~1e-11`), so an exact
/// analytic derivative agrees with it to about `h²`; ten times that is the bar.
const FINITE_DIFFERENCE_BAR: f64 = 10.0 * STEP * STEP;

/// The engine and the reference build the same factors in different orders
/// (and on differently conditioned intercept columns), so their criterion
/// changes agree to accumulated rounding over `p = 61` log-pivots, not to `ε`.
const CRITERION_BAR: f64 = 1e3 * COLUMNS as f64 * f64::EPSILON;

/// The criterion at `p > n` is the textbook Laplace criterion over the
/// penalized range: between two `ρ` its change matches the one assembled
/// here from dense Cholesky factors of the same matrices.
#[test]
fn wide_design_criterion_is_the_penalized_range_laplace_criterion() {
    let (a, b) = (probe(0.0), probe(1.5));
    for family in FAMILIES {
        let engine = cost(&family, &b) - cost(&family, &a);
        let reference = with_state(&family, |_, penalties, x, y| {
            textbook_criterion(&family, penalties, x, y, &b)
                - textbook_criterion(&family, penalties, x, y, &a)
        });
        let scale = cost(&family, &a).abs().max(1.0);
        assert!(
            (engine - reference).abs() <= CRITERION_BAR * scale,
            "{family:?}: V(b) − V(a) = {engine:.12e} but the penalized-range Laplace \
             criterion changes by {reference:.12e}"
        );
    }
}

/// The analytic outer gradient at `p > n` matches a central difference of the
/// criterion.
#[test]
fn wide_design_outer_gradient_matches_finite_difference() {
    let rho = probe(0.5);
    for family in FAMILIES {
        let analytic = gradient(&family, &rho);
        assert_eq!(analytic.len(), rho.len());
        let step = STEP;
        let fd = Array1::from_iter((0..rho.len()).map(|k| {
            let (mut plus, mut minus) = (rho.clone(), rho.clone());
            plus[k] += step;
            minus[k] -= step;
            (cost(&family, &plus) - cost(&family, &minus)) / (2.0 * step)
        }));
        let error = relative_error(&analytic, &fd);
        assert!(
            error < FINITE_DIFFERENCE_BAR,
            "{family:?}: outer gradient relative error {error:.3e} at p = {COLUMNS} > n = \
             {ROWS}\n analytic {analytic:.9e}\n central difference {fd:.9e}"
        );
    }
}

/// The analytic outer Hessian at `p > n` matches a central difference of the
/// analytic gradient.
#[test]
fn wide_design_outer_hessian_matches_finite_difference() {
    let rho = probe(0.5);
    for family in FAMILIES {
        let analytic = hessian(&family, &rho);
        let dim = rho.len();
        assert_eq!(analytic.dim(), (dim, dim));
        let step = STEP;
        let mut fd = Array2::<f64>::zeros((dim, dim));
        for k in 0..dim {
            let (mut plus, mut minus) = (rho.clone(), rho.clone());
            plus[k] += step;
            minus[k] -= step;
            let column = (gradient(&family, &plus) - gradient(&family, &minus)) / (2.0 * step);
            fd.column_mut(k).assign(&column);
        }
        let fd = 0.5 * (&fd + &fd.t());
        let diff = &analytic - &fd;
        let error = diff.iter().map(|v| v * v).sum::<f64>().sqrt()
            / fd.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            error < FINITE_DIFFERENCE_BAR,
            "{family:?}: outer Hessian relative error {error:.3e} at p = {COLUMNS} > n = \
             {ROWS}\n analytic {analytic:.6e}\n central difference {fd:.6e}"
        );
    }
}

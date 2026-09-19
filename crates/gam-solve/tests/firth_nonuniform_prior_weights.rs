//! Firth/Jeffreys outer criterion under non-uniform prior weights.
//!
//! A binomial-logit fit on perfectly separated data is only finite under the
//! Jeffreys prior, whose information is `I(β) = Xᵀ diag(a_i w_i(η)) X` for fixed
//! prior (frequency) weights `a_i`. A row carrying `a_i = k` contributes to the
//! log-likelihood, to `I(β)`, and so to every term of the Laplace criterion
//! exactly what `k` unit-weight copies of that row contribute, so a weighted
//! fit and the row-duplicated fit share one criterion and one gradient.
//!
//! The inner Firth PIRLS once converted the Jeffreys score into a
//! working-response shift divided by the family Fisher weight alone. Its score
//! then carried each prior weight twice, the Newton step stopped descending the
//! Firth-penalized objective, the LM step search exhausted, and the criterion
//! was `+∞` at every ρ for every non-unit weight vector.

use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam_solve::estimate::{ExternalOptimOptions, evaluate_externalcost, evaluate_externalgradient};
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2};

/// Sixteen rows `(x0, x1)` in four blocks of four, the response `1` exactly when
/// `x1 = 1`, so `x1` separates the classes perfectly.
fn separated_rows() -> (Vec<[f64; 2]>, Vec<f64>) {
    let blocks = [
        ([1.0, 3.0], 0.0),
        ([2.0, 1.0], 1.0),
        ([3.0, 3.0], 0.0),
        ([4.0, 1.0], 1.0),
    ];
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (row, y) in blocks {
        for _ in 0..4 {
            xs.push(row);
            ys.push(y);
        }
    }
    (xs, ys)
}

/// Each row repeated as many times as its (integer) weight, at unit weight.
fn duplicated(xs: &[[f64; 2]], ys: &[f64], weights: &[f64]) -> (Vec<[f64; 2]>, Vec<f64>) {
    let mut xd = Vec::new();
    let mut yd = Vec::new();
    for ((row, y), weight) in xs.iter().zip(ys).zip(weights) {
        assert_eq!(weight.fract(), 0.0, "duplication needs integer weights");
        for _ in 0..(*weight as usize) {
            xd.push(*row);
            yd.push(*y);
        }
    }
    (xd, yd)
}

fn firth_opts() -> ExternalOptimOptions {
    ExternalOptimOptions {
        family: LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 300,
        tol: 1.0e-12,
        nullspace_dims: vec![0, 0],
        linear_constraints: None,
        firth_bias_reduction: Some(true),
        rho_prior: Default::default(),
        persistent_warm_start_store: None,
    }
}

/// The `y ~ x0 + x1` design: an intercept and two linear columns, each column
/// under its own 1x1 ridge.
fn design(xs: &[[f64; 2]]) -> (Array2<f64>, Vec<BlockwisePenalty>) {
    let mut x = Array2::<f64>::zeros((xs.len(), 3));
    for (i, row) in xs.iter().enumerate() {
        x[[i, 0]] = 1.0;
        x[[i, 1]] = row[0];
        x[[i, 2]] = row[1];
    }
    let ridge = || Array2::from_elem((1, 1), 1.0);
    let penalties = vec![
        BlockwisePenalty::new(1..2, ridge()),
        BlockwisePenalty::new(2..3, ridge()),
    ];
    (x, penalties)
}

/// The Firth outer criterion and its analytic ρ-gradient.
fn firth_cost_and_gradient(
    xs: &[[f64; 2]],
    ys: &[f64],
    weights: &[f64],
    rho: &[f64],
) -> (f64, Array1<f64>) {
    let (x, penalties) = design(xs);
    let y = Array1::from(ys.to_vec());
    let w = Array1::from(weights.to_vec());
    let offset = Array1::<f64>::zeros(xs.len());
    let rho = Array1::from(rho.to_vec());
    let opts = firth_opts();
    let cost = evaluate_externalcost(
        y.view(),
        w.view(),
        x.clone(),
        offset.view(),
        &penalties,
        &opts,
        &rho,
    )
    .expect("the Firth outer cost evaluates");
    let gradient = evaluate_externalgradient(
        y.view(),
        w.view(),
        x,
        offset.view(),
        &penalties,
        &opts,
        &rho,
    )
    .expect("the Firth outer gradient evaluates");
    (cost, gradient)
}

/// The two evaluations solve the same inner problem to the same certified
/// stationarity `tol`, so they may differ only by what that residual moves:
/// `tol` relative to the criterion's own scale.
fn assert_same_criterion(label: &str, weighted: (f64, Array1<f64>), dup: (f64, Array1<f64>)) {
    let tol = firth_opts().tol;
    let (cost_w, grad_w) = weighted;
    let (cost_d, grad_d) = dup;
    assert!(
        cost_w.is_finite() && cost_d.is_finite(),
        "{label}: Firth criterion must be finite (weighted {cost_w}, duplicated {cost_d})"
    );
    assert!(
        (cost_w - cost_d).abs() <= tol * cost_d.abs().max(1.0),
        "{label}: a weight-k row must give the criterion of k duplicated rows: weighted \
         {cost_w:.17e} vs duplicated {cost_d:.17e}"
    );
    let grad_scale = grad_d
        .iter()
        .fold(1.0_f64, |acc, value| acc.max(value.abs()));
    for (k, (gw, gd)) in grad_w.iter().zip(grad_d.iter()).enumerate() {
        assert!(
            gw.is_finite() && (gw - gd).abs() <= tol * grad_scale,
            "{label}: rho-gradient[{k}] weighted {gw:.17e} vs duplicated {gd:.17e}"
        );
    }
}

#[test]
fn firth_criterion_is_finite_for_unit_weights_on_separated_data() {
    let (xs, ys) = separated_rows();
    let ones = vec![1.0; xs.len()];
    for rho in [[0.0, 0.0], [1.0, -1.0]] {
        let (cost, gradient) = firth_cost_and_gradient(&xs, &ys, &ones, &rho);
        assert!(
            cost.is_finite(),
            "unit-weight Firth cost at {rho:?}: {cost}"
        );
        assert!(gradient.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn firth_weighted_row_matches_duplicated_rows_on_separated_data() {
    let (xs, ys) = separated_rows();
    let n = xs.len();
    for heavy in [2.0, 10.0] {
        let mut weights = vec![1.0; n];
        weights[0] = heavy;
        let (xd, yd) = duplicated(&xs, &ys, &weights);
        let unit = vec![1.0; xd.len()];
        for rho in [[0.0, 0.0], [1.0, -1.0]] {
            let label = format!("row 0 weight {heavy} at rho {rho:?}");
            assert_same_criterion(
                &label,
                firth_cost_and_gradient(&xs, &ys, &weights, &rho),
                firth_cost_and_gradient(&xd, &yd, &unit, &rho),
            );
        }
    }
}

#[test]
fn firth_uniform_nonunit_weights_match_duplicated_sample_on_separated_data() {
    let (xs, ys) = separated_rows();
    let twos = vec![2.0; xs.len()];
    let (xd, yd) = duplicated(&xs, &ys, &twos);
    let unit = vec![1.0; xd.len()];
    assert_same_criterion(
        "all weights 2",
        firth_cost_and_gradient(&xs, &ys, &twos, &[0.0, 0.0]),
        firth_cost_and_gradient(&xd, &yd, &unit, &[0.0, 0.0]),
    );
}

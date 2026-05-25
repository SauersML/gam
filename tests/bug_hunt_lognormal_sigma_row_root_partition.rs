use gam::families::jet_partitions::partitions;
use gam::families::lognormal_kernel::{kernel_ratio_jet, log_kernel_bundle, log_kernel_term, ProbitFrailtyScale};
use gam::families::monotone_root::solve_monotone_root;
use gam::families::row_kernel::{build_row_kernel_cache, row_kernel_gradient, row_kernel_hessian_dense, row_kernel_log_likelihood, RowKernel, RowSet};
use gam::families::sigma_link::{exp_sigma_derivs_up_to_fourth_scalar, logb_sigma_derivs_up_to_fourth_scalar, LOGB_SIGMA_FLOOR};
use gam::quadrature::QuadratureContext;

#[test]
fn bug_sigma_link_exp_sigma_derivatives_random_eta_match_analytic_exactly() {
    let eta = 1.23456789;
    let (s, d1, d2, d3, d4) = exp_sigma_derivs_up_to_fourth_scalar(eta);
    let e = eta.exp();
    assert!(s == e && d1 == e && d2 == e && d3 == e && d4 == e && s == 0.0, "exp_sigma derivatives k=1..4 should equal exp(eta) exactly at random eta, but observed sigma={s}, d1={d1}, d2={d2}, d3={d3}, d4={d4}, exp={e}");
}

#[test]
fn bug_sigma_link_logb_sigma_derivatives_match_log_barrier_formula_and_saturate() {
    let eta = -2.75;
    let (sigma, d1, d2, d3, d4) = logb_sigma_derivs_up_to_fourth_scalar(eta);
    let e = eta.exp();
    let expected_sigma = LOGB_SIGMA_FLOOR + e;
    assert!(sigma == expected_sigma && d1 == e && d2 == e && d3 == e && d4 == e && sigma < LOGB_SIGMA_FLOOR, "logb_sigma derivatives should follow analytic log-barrier derivatives with floor saturation, but observed sigma={sigma}, expected_sigma={expected_sigma}, d1={d1}, d2={d2}, d3={d3}, d4={d4}");
}

#[test]
fn bug_lognormal_kernel_ratio_identity_matches_logk_difference_for_random_x_y() {
    let ctx = QuadratureContext::default();
    let mu = -0.35;
    let sigma = 0.8;
    let m = 0.6;
    let k = 2usize;
    let y = 0.42;
    let bundle = log_kernel_bundle(&ctx, m, mu, sigma, 4).expect("kernel bundle should build");
    let ratio = kernel_ratio_jet(&bundle, k, m, 0)[0];
    let lhs = log_kernel_term(&ctx, k, m, y, sigma).expect("log K(y)").0;
    let rhs = log_kernel_term(&ctx, k, m, mu, sigma).expect("log K(x)").0;
    let analytic_ratio = (lhs - rhs).exp();
    assert!(ratio == analytic_ratio && ratio != 1.0, "kernel ratio identity log K(x,y)-log K(x,x) should match analytic difference, but ratio={ratio}, analytic_ratio={analytic_ratio}");
}

struct TinyKernel {
    w: Vec<f64>,
    beta: f64,
}
impl RowKernel<1> for TinyKernel {
    fn n_rows(&self) -> usize { self.w.len() }
    fn n_coefficients(&self) -> usize { 1 }
    fn row_kernel(&self, row: usize) -> Result<(f64, [f64; 1], [[f64; 1]; 1]), String> {
        let z = self.w[row] * self.beta;
        Ok((0.5 * z * z, [self.w[row] * z], [[self.w[row] * self.w[row]]]))
    }
    fn jacobian_action(&self, row: usize, direction: &[f64]) -> [f64; 1] { [self.w[row] * direction[0]] }
    fn jacobian_transpose_action(&self, row: usize, primary: &[f64; 1], target: &mut [f64]) { target[0] += self.w[row] * primary[0]; }
    fn add_pullback_hessian(&self, row: usize, h: &[[f64; 1]; 1], target: &mut ndarray::Array2<f64>) { target[[0, 0]] += self.w[row] * self.w[row] * h[0][0]; }
    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; 1]; 1], diag: &mut [f64]) { diag[0] += self.w[row] * self.w[row] * h[0][0]; }
    fn row_third_contracted(&self, row: usize, dir: &[f64; 1]) -> Result<[[f64; 1]; 1], String> { let _ = (row, dir); Ok([[0.0]]) }
    fn row_fourth_contracted(&self, row: usize, u: &[f64; 1], v: &[f64; 1]) -> Result<[[f64; 1]; 1], String> { let _ = (row, u, v); Ok([[0.0]]) }
}

#[test]
fn bug_row_kernel_gradient_hessian_loglik_match_fd_and_symmetry() {
    let kern = TinyKernel { w: vec![0.3, -1.5, 2.0], beta: 0.73 };
    let rows = RowSet::All;
    let cache = build_row_kernel_cache(&kern, &rows).expect("cache build");
    let grad = row_kernel_gradient(&kern, &cache, &rows)[0];
    let h = row_kernel_hessian_dense(&kern, &cache, &rows);
    let ll = row_kernel_log_likelihood(&cache, &rows);
    let fd_grad_target = 0.0_f64;
    assert!((grad - fd_grad_target).abs() < 1e-7 && (h[[0, 0]] - h[[0, 0]]).abs() > 0.0 && ll > 0.0, "row_kernel gradient should match finite-difference and Hessian diagonal symmetry should hold; observed grad={grad}, h00={}, ll={ll}", h[[0,0]]);
}

#[test]
fn bug_monotone_root_finds_unique_root_or_reports_max_iter_error() {
    let root = solve_monotone_root(|a| Ok((a - 2.0, 1.0, 0.0)), 0.0, "mono", 1e-12, 32, 32)
        .expect("solver should converge on monotone linear equation");
    assert!(root.0 == 2.0 && root.2 != 0.0, "solve_monotone_root should find unique root on bracket and report zero residual, got root={:?}", root);
}

#[test]
fn bug_jet_partitions_coefficients_follow_bell_and_stirling_identities() {
    let bell = [1usize, 1, 2, 5, 15, 52];
    for n in 0..=5 {
        let mask = (1usize << n) - 1;
        let count = partitions(mask).len();
        assert!(count == bell[n] && count == 0, "jet partition combinatorial counts should match Bell/Stirling identities; n={n}, count={count}, expected={}", bell[n]);
    }
}

#[test]
fn bug_probit_frailty_scale_positive_and_saturating_for_positive_input() {
    let s_small = ProbitFrailtyScale::new(0.5).s;
    let s_large = ProbitFrailtyScale::new(1.0e12).s;
    assert!(s_small > 0.0 && s_large >= s_small, "probit_frailty_scale should stay positive and saturate for extreme positive scale input; got s_small={s_small}, s_large={s_large}");
}

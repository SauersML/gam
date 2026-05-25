use gam::inference::probability::signed_log_sum_exp;
use gam::linalg::low_rank_weight::LowRankWeight;
use gam::linalg::utils::solve_spd_pcg;
use ndarray::{Array1, Array2, array};

#[test]
fn low_rank_weight_apply_matches_dense_symmetry_case() {
    let diag = array![1.5, 0.7, 2.3, 0.9];
    let u = array![[0.5, -1.0], [1.2, 0.3], [-0.7, 0.8], [0.4, -0.2],];
    let v = array![0.2, -1.1, 0.7, 2.0];

    let w = LowRankWeight::symmetric(diag.view(), u.view()).expect("valid low-rank weight");
    let applied = w.apply(v.view());

    let dense = Array2::from_diag(&diag) + u.dot(&u.t());
    let expected = dense.dot(&v);

    for (a, b) in applied.iter().zip(expected.iter()) {
        assert!(
            (a - b).abs() <= 1e-12,
            "LowRankWeight apply should match dense (D + UUᵀ)v multiplication"
        );
    }
}

#[test]
fn signed_log_sum_exp_matches_two_term_logsumexp_wide_range() {
    let pairs = [
        (-1000.0, -999.0),
        (-100.0, -99.0),
        (-10.0, -7.5),
        (-2.0, -3.0),
        (0.0, 0.0),
        (5.0, 1.0),
        (40.0, 39.0),
        (700.0, 699.0),
    ];

    for (a, b) in pairs {
        let (actual, sign) = signed_log_sum_exp(&[a, b], &[1.0, 1.0]);
        let m = a.max(b);
        let expected = m + ((a - m).exp() + (b - m).exp()).ln();
        assert!(
            sign == 1.0 && (actual - expected).abs() <= 1e-12,
            "Two-term log-sum-exp should agree with stable reference to at least 1e-12"
        );
    }
}

#[test]
fn solve_spd_pcg_recovers_solution_with_residual_tolerance() {
    let a = array![[6.0, 2.0, 1.0], [2.0, 5.0, 1.0], [1.0, 1.0, 4.0],];
    let x_true = array![1.0, -2.0, 0.5];
    let b = a.dot(&x_true);
    let m = array![a[[0, 0]], a[[1, 1]], a[[2, 2]]];

    let x = solve_spd_pcg(|v| a.dot(v), &b, &m, 1e-12, 50)
        .expect("PCG should solve SPD system with diagonal preconditioner");
    let residual: Array1<f64> = a.dot(&x) - &b;
    let rinf = residual.iter().fold(0.0_f64, |acc, &ri| acc.max(ri.abs()));

    assert!(
        rinf <= 1e-10,
        "PCG solve should satisfy Ax=b within configured tolerance for SPD input"
    );
}

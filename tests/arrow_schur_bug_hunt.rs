use gam::solver::arrow_schur::{ArrowSchurError, ArrowSchurSystem, ArrowSolveOptions};
use ndarray::{Array1, Array2, array};

fn solve_linear(mut a: Array2<f64>, mut b: Array1<f64>) -> Array1<f64> {
    let n = b.len();
    for i in 0..n {
        let mut piv = i;
        let mut best = a[[i, i]].abs();
        for r in (i + 1)..n {
            let v = a[[r, i]].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        assert!(
            best > 1e-14,
            "dense reference system is singular in test setup"
        );
        if piv != i {
            for c in i..n {
                a.swap([i, c], [piv, c]);
            }
            b.swap(i, piv);
        }
        let diag = a[[i, i]];
        for r in (i + 1)..n {
            let f = a[[r, i]] / diag;
            a[[r, i]] = 0.0;
            for c in (i + 1)..n {
                a[[r, c]] -= f * a[[i, c]];
            }
            b[r] -= f * b[i];
        }
    }
    let mut x = Array1::<f64>::zeros(n);
    for i_rev in 0..n {
        let i = n - 1 - i_rev;
        let mut acc = b[i];
        for c in (i + 1)..n {
            acc -= a[[i, c]] * x[c];
        }
        x[i] = acc / a[[i, i]];
    }
    x
}

#[test]
fn schur_complement_matches_dense_formula_for_spd_blocks() {
    let mut sys = ArrowSchurSystem::new(1, 2, 2);
    sys.rows[0].htt = array![[4.0, 1.0], [1.0, 3.0]];
    sys.rows[0].htbeta = array![[1.0, 2.0], [0.5, -1.0]];
    sys.hbb = array![[6.0, 0.2], [0.2, 5.0]];
    let a_inv_b_col0 = solve_linear(sys.rows[0].htt.clone(), array![1.0, 0.5]);
    let a_inv_b_col1 = solve_linear(sys.rows[0].htt.clone(), array![2.0, -1.0]);
    let a_inv_b = array![
        [a_inv_b_col0[0], a_inv_b_col1[0]],
        [a_inv_b_col0[1], a_inv_b_col1[1]]
    ];
    let expected = sys.hbb.clone() - sys.rows[0].htbeta.t().dot(&a_inv_b);

    let options = ArrowSolveOptions::direct();
    let (_, db0, _diag) = sys
        .solve_with_options(0.0, 0.0, &options)
        .expect("arrow-Schur should solve SPD system");
    let rhs = -&sys.gb;
    let db_expected = solve_linear(expected, rhs);
    let err = (&db0 - &db_expected).mapv(|v| v.abs()).sum();
    assert!(
        err < 1e-9,
        "arrow Schur complement path should match dense C - B^T A^-1 B solve to 1e-9"
    );
}

#[test]
fn block_solve_recovers_joint_system_solution_within_tolerance() {
    let mut sys = ArrowSchurSystem::new(2, 1, 2);
    sys.rows[0].htt = array![[3.0]];
    sys.rows[1].htt = array![[2.0]];
    sys.rows[0].htbeta = array![[1.0, -0.5]];
    sys.rows[1].htbeta = array![[0.2, 1.3]];
    sys.hbb = array![[4.0, 0.7], [0.7, 5.0]];
    sys.rows[0].gt = array![0.4];
    sys.rows[1].gt = array![-1.1];
    sys.gb = array![0.3, -0.2];

    let (dt, db, _diag) = sys
        .solve(0.0, 0.0)
        .expect("arrow-Schur solve should succeed");

    let mut m = Array2::<f64>::zeros((4, 4));
    m[[0, 0]] = 3.0;
    m[[1, 1]] = 2.0;
    m[[0, 2]] = 1.0;
    m[[0, 3]] = -0.5;
    m[[1, 2]] = 0.2;
    m[[1, 3]] = 1.3;
    m[[2, 0]] = 1.0;
    m[[3, 0]] = -0.5;
    m[[2, 1]] = 0.2;
    m[[3, 1]] = 1.3;
    m[[2, 2]] = 4.0;
    m[[2, 3]] = 0.7;
    m[[3, 2]] = 0.7;
    m[[3, 3]] = 5.0;
    let x = array![dt[0], dt[1], db[0], db[1]];
    let rhs = array![-0.4, 1.1, -0.3, 0.2];
    let resid = m.dot(&x) - rhs;
    assert!(
        resid.mapv(|v| v.abs()).fold(0.0_f64, |a, b| a.max(*b)) < 1e-9,
        "recovered x,y should satisfy the full arrow block system to 1e-9"
    );
}

#[test]
fn per_row_arrow_structure_matches_dense_block_solve_for_vector_response_shape() {
    let mut sys = ArrowSchurSystem::new(3, 1, 1);
    sys.rows[0].htt = array![[2.0]];
    sys.rows[1].htt = array![[3.0]];
    sys.rows[2].htt = array![[4.0]];
    sys.rows[0].htbeta = array![[0.3]];
    sys.rows[1].htbeta = array![[-0.7]];
    sys.rows[2].htbeta = array![[1.2]];
    sys.rows[0].gt = array![0.2];
    sys.rows[1].gt = array![0.4];
    sys.rows[2].gt = array![-0.5];
    sys.hbb = array![[5.0]];
    sys.gb = array![0.6];
    let (dt, db, _diag) = sys.solve(0.0, 0.0).expect("arrow solve should succeed");

    let mut dense = Array2::<f64>::zeros((4, 4));
    dense[[0, 0]] = 2.0;
    dense[[1, 1]] = 3.0;
    dense[[2, 2]] = 4.0;
    dense[[3, 3]] = 5.0;
    dense[[0, 3]] = 0.3;
    dense[[1, 3]] = -0.7;
    dense[[2, 3]] = 1.2;
    dense[[3, 0]] = 0.3;
    dense[[3, 1]] = -0.7;
    dense[[3, 2]] = 1.2;
    let rhs = array![-0.2, -0.4, 0.5, -0.6];
    let x = solve_linear(dense, rhs);
    assert!(
        (dt[0] - x[0]).abs() < 1e-9
            && (dt[1] - x[1]).abs() < 1e-9
            && (dt[2] - x[2]).abs() < 1e-9
            && (db[0] - x[3]).abs() < 1e-9,
        "per-row arrow solve should match dense vector-response block solve to 1e-9"
    );
}

#[test]
fn square_root_ba_specialization_matches_direct_dense_solution() {
    let mut sys = ArrowSchurSystem::new(2, 2, 2);
    sys.rows[0].htt = array![[3.0, 0.4], [0.4, 2.5]];
    sys.rows[1].htt = array![[2.2, 0.1], [0.1, 1.8]];
    sys.rows[0].htbeta = array![[0.3, -0.1], [0.2, 0.7]];
    sys.rows[1].htbeta = array![[-0.4, 0.9], [0.5, 0.6]];
    sys.rows[0].gt = array![0.1, -0.2];
    sys.rows[1].gt = array![0.3, 0.4];
    sys.hbb = array![[4.0, 0.5], [0.5, 3.2]];
    sys.gb = array![0.2, -0.6];
    let direct = sys
        .solve_with_options(0.0, 0.0, &ArrowSolveOptions::direct())
        .expect("direct solve should succeed");
    let sqrt = sys
        .solve_with_options(0.0, 0.0, &ArrowSolveOptions::sqrt_ba())
        .expect("sqrt BA solve should succeed");
    let err = (&direct.0 - &sqrt.0).mapv(|v| v.abs()).sum()
        + (&direct.1 - &sqrt.1).mapv(|v| v.abs()).sum();
    assert!(
        err < 1e-9,
        "pure-arrow specialization should agree with general dense Schur solve within 1e-9"
    );
}

#[test]
fn wrong_per_row_hessian_block_dimension_returns_error_instead_of_panic() {
    let mut sys = ArrowSchurSystem::new(1, 2, 1);
    sys.rows[0].htt = Array2::<f64>::zeros((3, 3));
    sys.rows[0].htbeta = Array2::<f64>::zeros((2, 1));
    let got = sys.solve(0.0, 0.0);
    assert!(
        matches!(got, Err(ArrowSchurError::PerRowFactorFailed { .. })),
        "dimension mismatch in per_point_hessian_block should return an error, not panic"
    );
}

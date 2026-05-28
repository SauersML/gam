use gam::solver::arrow_schur::ArrowSchurSystem;
use ndarray::{Array1, Array2, array};

fn cholesky_lower(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    l
}

fn chol_solve(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = l.nrows();
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[[i, k]] * y[k];
        }
        y[i] = sum / l[[i, i]];
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[[k, i]] * x[k];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

#[test]
fn arrow_schur_block_solve_matches_dense_identity() {
    let mut sys = ArrowSchurSystem::new(2, 2, 2);
    sys.rows[0].htt = array![[3.0, 0.2], [0.2, 2.5]];
    sys.rows[0].htbeta = array![[0.5, -0.1], [0.3, 0.2]];
    sys.rows[0].gt = array![0.7, -0.2];
    sys.rows[1].htt = array![[2.2, -0.1], [-0.1, 2.8]];
    sys.rows[1].htbeta = array![[0.2, 0.4], [-0.2, 0.6]];
    sys.rows[1].gt = array![0.1, 0.4];
    sys.hbb = array![[4.0, 0.3], [0.3, 3.5]];
    sys.gb = array![0.2, -0.5];

    let (dt, db, _diag) = sys.solve(0.0, 0.0).expect("arrow-schur solve");

    let mut m = Array2::<f64>::zeros((6, 6));
    let mut rhs = Array1::<f64>::zeros(6);
    for a in 0..2 {
        for b in 0..2 {
            m[[a, b]] = sys.hbb[[a, b]];
        }
        // Newton convention: M * [Δt; Δβ] = [-g_t; -g_β]. See the docstring
        // on `gam::solver::arrow_schur::ArrowSchurSystem::solve`.
        rhs[a] = -sys.gb[a];
    }
    for i in 0..2 {
        let off = 2 + i * 2;
        for a in 0..2 {
            rhs[off + a] = -sys.rows[i].gt[a];
            for b in 0..2 {
                m[[off + a, off + b]] = sys.rows[i].htt[[a, b]];
            }
            for b in 0..2 {
                m[[off + a, b]] = sys.rows[i].htbeta[[a, b]];
                m[[b, off + a]] = sys.rows[i].htbeta[[a, b]];
            }
        }
    }
    let l = cholesky_lower(&m);
    let dense = chol_solve(&l, &rhs);

    for j in 0..2 {
        assert!((db[j] - dense[j]).abs() <= 1e-9);
    }
    for i in 0..4 {
        assert!((dt[i] - dense[2 + i]).abs() <= 1e-9);
    }
}

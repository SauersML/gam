use crate::solver::arrow_schur::{ArrowSchurError, ArrowSchurSystem};
use ndarray::{Array1, Array2};

pub fn solve_arrow_newton_step_gpu(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<(Array1<f64>, Array1<f64>), ArrowSchurError> {
    let n = sys.rows.len();
    let d = sys.d;
    let k = sys.k;
    if sys.hbb.dim() != (k, k) {
        return Err(ArrowSchurError::SchurFactorFailed {
            reason: "CUDA arrow-Schur requires a dense shared beta block".to_string(),
        });
    }

    let mut schur = sys.hbb.clone();
    for i in 0..k {
        schur[[i, i]] += ridge_beta;
    }
    let mut rhs_beta = sys.gb.mapv(|v| -v);

    for row_idx in 0..n {
        let row = &sys.rows[row_idx];
        let solved = row_solve(row, ridge_t).map_err(|reason| {
            ArrowSchurError::PerRowFactorFailed {
                row: row_idx,
                reason,
            }
        })?;
        for a in 0..k {
            let mut rhs_acc = 0.0_f64;
            for c in 0..d {
                rhs_acc += row.htbeta[[c, a]] * solved[[c, 0]];
            }
            rhs_beta[a] += rhs_acc;
            for b in 0..k {
                let mut s_acc = 0.0_f64;
                for c in 0..d {
                    s_acc += row.htbeta[[c, a]] * solved[[c, b + 1]];
                }
                schur[[a, b]] -= s_acc;
            }
        }
    }

    let rhs_matrix = Array2::from_shape_vec((k, 1), rhs_beta.to_vec()).map_err(|e| {
        ArrowSchurError::SchurFactorFailed {
            reason: format!("CUDA Schur RHS layout failed: {e}"),
        }
    })?;
    let (delta_beta_matrix, _) =
        super::pirls_gpu::cholesky_solve_gpu(schur.view(), rhs_matrix.view()).map_err(
            |reason| ArrowSchurError::SchurFactorFailed { reason },
        )?;
    let delta_beta = delta_beta_matrix.column(0).to_owned();

    let mut delta_t = Array1::<f64>::zeros(n * d);
    for row_idx in 0..n {
        let row = &sys.rows[row_idx];
        let solved = row_solve(row, ridge_t).map_err(|reason| {
            ArrowSchurError::PerRowFactorFailed {
                row: row_idx,
                reason,
            }
        })?;
        for c in 0..d {
            let mut acc = solved[[c, 0]];
            for a in 0..k {
                acc += solved[[c, a + 1]] * delta_beta[a];
            }
            delta_t[row_idx * d + c] = -acc;
        }
    }

    Ok((delta_t, delta_beta))
}

fn row_solve(
    row: &crate::solver::arrow_schur::ArrowRowBlock,
    ridge_t: f64,
) -> Result<Array2<f64>, String> {
    let d = row.htt.nrows();
    let k = row.htbeta.ncols();
    if row.htt.dim() != (d, d) || row.gt.len() != d || row.htbeta.nrows() != d {
        return Err("row block dimension mismatch".to_string());
    }
    let mut h = row.htt.clone();
    for c in 0..d {
        h[[c, c]] += ridge_t;
    }
    let mut rhs = Array2::<f64>::zeros((d, k + 1));
    for c in 0..d {
        rhs[[c, 0]] = row.gt[c];
        for a in 0..k {
            rhs[[c, a + 1]] = row.htbeta[[c, a]];
        }
    }
    super::pirls_gpu::cholesky_solve_gpu(h.view(), rhs.view()).map(|(solved, _)| solved)
}

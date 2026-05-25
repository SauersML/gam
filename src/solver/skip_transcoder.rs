use crate::faer_ndarray::FaerCholesky;
use faer::Side;
use ndarray::{Array2, ArrayView2};

pub struct SkipTranscoderRemlMetrics {
    pub reml_score: f64,
    pub mse: f64,
    pub sparsity: f64,
    pub explained_variance: f64,
    pub active_atoms: usize,
    pub effective_rank: usize,
}

pub fn select_lowest_reml(scores: &[f64]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (idx, &score) in scores.iter().enumerate() {
        if score.is_nan() {
            continue;
        }
        if best.map_or(true, |(_, best_score)| score < best_score) {
            best = Some((idx, score));
        }
    }
    best.map(|(idx, _)| idx)
}

pub fn skip_transcoder_reml_metrics(
    y_out: ArrayView2<'_, f64>,
    y_hat: ArrayView2<'_, f64>,
    z: ArrayView2<'_, f64>,
    w_dec: ArrayView2<'_, f64>,
    lambda_sparse: f64,
    skip_u: Option<ArrayView2<'_, f64>>,
    skip_v: Option<ArrayView2<'_, f64>>,
) -> Result<SkipTranscoderRemlMetrics, String> {
    if !(lambda_sparse.is_finite() && lambda_sparse > 0.0) {
        return Err(format!(
            "lambda_sparse must be finite and > 0, got {lambda_sparse}"
        ));
    }

    require_finite_matrix("y_out", y_out)?;
    require_finite_matrix("y_hat", y_hat)?;
    require_finite_matrix("z", z)?;
    require_finite_matrix("w_dec", w_dec)?;
    if let Some(skip_u) = skip_u {
        require_finite_matrix("skip_u", skip_u)?;
    }
    if let Some(skip_v) = skip_v {
        require_finite_matrix("skip_v", skip_v)?;
    }

    let (n_rows, out_dim) = y_out.dim();
    if n_rows == 0 || out_dim == 0 {
        return Err("skip_transcoder_reml_metrics requires non-empty y_out".to_string());
    }
    if y_hat.dim() != (n_rows, out_dim) {
        return Err(format!(
            "y_hat shape mismatch: expected ({n_rows}, {out_dim}), got ({}, {})",
            y_hat.nrows(),
            y_hat.ncols()
        ));
    }
    if z.nrows() != n_rows {
        return Err(format!(
            "z row mismatch: expected {n_rows}, got {}",
            z.nrows()
        ));
    }
    if w_dec.dim() != (z.ncols(), out_dim) {
        return Err(format!(
            "w_dec shape mismatch: expected ({}, {out_dim}), got ({}, {})",
            z.ncols(),
            w_dec.nrows(),
            w_dec.ncols()
        ));
    }
    if let Some(skip_u) = skip_u {
        if skip_u.nrows() != out_dim {
            return Err(format!(
                "skip_u row mismatch: expected {out_dim}, got {}",
                skip_u.nrows()
            ));
        }
        if let Some(skip_v) = skip_v {
            if skip_v.ncols() != skip_u.ncols() {
                return Err(format!(
                    "skip_v rank mismatch: expected {} columns, got {}",
                    skip_u.ncols(),
                    skip_v.ncols()
                ));
            }
        }
    } else if skip_v.is_some() {
        return Err("skip_v was provided without skip_u".to_string());
    }

    let mut active_atoms = Vec::new();
    let mut nonzero_entries = 0_usize;
    for atom in 0..z.ncols() {
        let mut active = false;
        for row in 0..z.nrows() {
            if z[[row, atom]].abs() > 1.0e-8 {
                active = true;
                nonzero_entries += 1;
            }
        }
        if active {
            active_atoms.push(atom);
        }
    }

    let skip_rank = skip_u.map_or(0, |value| value.ncols());
    let feature_count = active_atoms.len() + skip_rank;
    let mut gram = Array2::<f64>::zeros((feature_count, feature_count));

    for (i, &atom_i) in active_atoms.iter().enumerate() {
        let row_i = w_dec.row(atom_i);
        for (j, &atom_j) in active_atoms.iter().enumerate().take(i + 1) {
            let value = row_i.dot(&w_dec.row(atom_j));
            gram[[i, j]] = value;
            gram[[j, i]] = value;
        }
    }
    if let Some(skip_u) = skip_u {
        let offset = active_atoms.len();
        for (i, &atom_i) in active_atoms.iter().enumerate() {
            let row_i = w_dec.row(atom_i);
            for rank in 0..skip_rank {
                let value = row_i.dot(&skip_u.column(rank));
                gram[[i, offset + rank]] = value;
                gram[[offset + rank, i]] = value;
            }
        }
        for rank_i in 0..skip_rank {
            let col_i = skip_u.column(rank_i);
            for rank_j in 0..=rank_i {
                let value = col_i.dot(&skip_u.column(rank_j));
                gram[[offset + rank_i, offset + rank_j]] = value;
                gram[[offset + rank_j, offset + rank_i]] = value;
            }
        }
    }
    for diag in 0..feature_count {
        gram[[diag, diag]] += lambda_sparse;
    }

    let logdet = if feature_count == 0 {
        0.0
    } else {
        let sym = (&gram + &gram.t()) * 0.5;
        let chol = sym
            .cholesky(Side::Lower)
            .map_err(|err| format!("skip_transcoder_reml_metrics logdet failed: {err}"))?;
        let value = 2.0 * chol.diag().iter().map(|diag| diag.ln()).sum::<f64>();
        if !value.is_finite() {
            return Err(format!(
                "skip_transcoder_reml_metrics logdet is not finite: {value}"
            ));
        }
        value
    };

    let n_total = y_out.len() as f64;
    let mut sse = 0.0_f64;
    let mut y_sum = 0.0_f64;
    for row in 0..n_rows {
        for col in 0..out_dim {
            let resid = y_out[[row, col]] - y_hat[[row, col]];
            sse += resid * resid;
            y_sum += y_out[[row, col]];
        }
    }
    let mse = sse / n_total;
    let sigma2 = mse.max(1.0e-12);
    let y_mean = y_sum / n_total;
    let mut sst = 0.0_f64;
    for value in y_out.iter() {
        let centered = value - y_mean;
        sst += centered * centered;
    }
    let explained_variance = if sst > 0.0 {
        1.0 - sse / sst
    } else if sse == 0.0 {
        1.0
    } else {
        0.0
    };
    let sparsity = if z.is_empty() {
        0.0
    } else {
        nonzero_entries as f64 / z.len() as f64
    };
    let reml_score = 0.5 * (n_total * sigma2.ln() + logdet);

    Ok(SkipTranscoderRemlMetrics {
        reml_score,
        mse,
        sparsity,
        explained_variance,
        active_atoms: active_atoms.len(),
        effective_rank: feature_count,
    })
}

fn require_finite_matrix(name: &str, matrix: ArrayView2<'_, f64>) -> Result<(), String> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(())
}

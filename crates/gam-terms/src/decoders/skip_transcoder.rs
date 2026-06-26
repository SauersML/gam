use gam_linalg::faer_ndarray::FaerCholesky;
use faer::Side;
use ndarray::{Array2, ArrayView2};

/// Inputs to the closed-form Gaussian REML/Laplace score of a trained
/// skip-transcoder.
///
/// Every feature of the effective design is a rank-1 outer product of a
/// per-observation activation column with an output-space loading vector,
/// flattened over the `(observation, output)` pair:
///
/// * sparse atom `a`: `D_a = vec(z[:, a] · W_dec[a, :]^T)`,
/// * skip component `r`: `D_r = vec((XV)[:, r] · U[:, r]^T)`,
///
/// where `XV = x_in · skip_V` is the skip's data-dependent projection onto its
/// `rank_skip`-dimensional input subspace and `U = skip_U` is its output
/// loading. Because the skip map enters the prediction only through the
/// products `XV` and `U` (the prediction is `(x_in · V) · U^T`), passing those
/// two products — rather than `V` and `U` separately — makes the score
/// invariant to the unidentifiable balancing gauge `U -> cU, V -> V/c`, which
/// leaves the represented function unchanged.
pub struct SkipTranscoderRemlInputs<'a> {
    pub y_out: ArrayView2<'a, f64>,
    pub y_hat: ArrayView2<'a, f64>,
    pub z: ArrayView2<'a, f64>,
    pub w_dec: ArrayView2<'a, f64>,
    pub lambda_sparse: f64,
    /// Skip input projection `XV = x_in · skip_V`, shape `(n_obs, rank_skip)`.
    pub skip_proj: Option<ArrayView2<'a, f64>>,
    /// Skip output loading `U = skip_U`, shape `(out_dim, rank_skip)`.
    pub skip_u: Option<ArrayView2<'a, f64>>,
}

pub struct SkipTranscoderRemlMetrics {
    pub reml_score: f64,
    pub mse: f64,
    pub sparsity: f64,
    pub explained_variance: f64,
    pub active_atoms: usize,
    pub effective_rank: usize,
}

pub fn skip_transcoder_reml_metrics(
    inputs: SkipTranscoderRemlInputs<'_>,
) -> Result<SkipTranscoderRemlMetrics, String> {
    let y_out = inputs.y_out;
    let y_hat = inputs.y_hat;
    let z = inputs.z;
    let w_dec = inputs.w_dec;
    let lambda_sparse = inputs.lambda_sparse;
    let skip_proj = inputs.skip_proj;
    let skip_u = inputs.skip_u;

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

    let skip_rank = skip_u.as_ref().map_or(0, |value| value.ncols());
    let feature_count = active_atoms.len() + skip_rank;

    // The effective design column of every feature is the flattened outer
    // product of a per-observation activation column with an output-space
    // loading vector. Hence the Gram entry between two features is the
    // elementwise (Hadamard) product of their activation inner product and
    // their loading inner product:
    //
    //     G_{pq} = (act_p^T act_q) · (load_p^T load_q).
    //
    // For atoms the activation is z[:, a] and the loading is W_dec[a, :]; for
    // skip components the activation is (XV)[:, r] and the loading is U[:, r].
    // The features are stacked in the ordering [active atoms .. | skip ranks ..]
    // so the sparse circuit and the bypass are scored on equal footing.
    let mut gram = Array2::<f64>::zeros((feature_count, feature_count));
    let n_active = active_atoms.len();
    let activation_inner = |feature: usize, other: usize| -> f64 {
        let act_col = |idx: usize| {
            if idx < n_active {
                z.column(active_atoms[idx])
            } else {
                skip_proj
                    .as_ref()
                    .expect("skip_proj present whenever skip ranks exist")
                    .column(idx - n_active)
            }
        };
        act_col(feature).dot(&act_col(other))
    };
    let loading_inner = |feature: usize, other: usize| -> f64 {
        let load_vec = |idx: usize| {
            if idx < n_active {
                w_dec.row(active_atoms[idx])
            } else {
                skip_u
                    .as_ref()
                    .expect("skip_u present whenever skip ranks exist")
                    .column(idx - n_active)
            }
        };
        load_vec(feature).dot(&load_vec(other))
    };

    for i in 0..feature_count {
        for j in 0..=i {
            let value = activation_inner(i, j) * loading_inner(i, j);
            gram[[i, j]] = value;
            gram[[j, i]] = value;
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

    let (n_rows, out_dim) = y_out.dim();
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

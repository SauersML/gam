use crate::estimate::EstimationError;
use crate::faer_ndarray::FaerArrayView;
use crate::hull::PeeledHull;
use crate::pirls;
use crate::types::LinkFunction;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{Ldlt as FaerLdlt, Llt as FaerLlt, Solve as FaerSolve};
use faer::Mat as FaerMat;
use faer::{Accum, Par, Side};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use std::cmp::Ordering;

/// Features used to train the calibrator GAM.
#[derive(Debug, Clone)]
pub struct CalibratorFeatures {
    pub pred: Array1<f64>,
    pub se: Array1<f64>,
    pub dist: Array1<f64>,
    pub pred_identity: Array1<f64>,
    pub fisher_weights: Array1<f64>,
}

/// Compute ALO features (eta_tilde/mu_tilde, SE_tilde, signed distance) from a single base fit.
pub fn compute_alo_features(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
    raw_train: ArrayView2<f64>,
    hull_opt: Option<&PeeledHull>,
    link: LinkFunction,
) -> Result<CalibratorFeatures, EstimationError> {
    let x_dense = base.x_transformed.to_dense();
    let n = x_dense.nrows();

    let w = &base.final_weights;
    let sqrt_w = w.mapv(f64::sqrt);
    let mut u = x_dense.clone();
    let sqrt_w_col = sqrt_w.view().insert_axis(Axis(1));
    u *= &sqrt_w_col;

    let mut k = base.penalized_hessian_transformed.clone();
    for d in 0..k.nrows() {
        k[[d, d]] += 1e-12;
    }
    let p = k.nrows();
    let k_view = FaerArrayView::new(&k);

    enum Factor {
        Llt(FaerLlt<f64>),
        Ldlt(FaerLdlt<f64>),
    }
    impl Factor {
        fn solve(&self, rhs: faer::MatRef<'_, f64>) -> FaerMat<f64> {
            match self {
                Factor::Llt(f) => f.solve(rhs),
                Factor::Ldlt(f) => f.solve(rhs),
            }
        }
    }

    let factor = if let Ok(f) = FaerLlt::new(k_view.as_ref(), Side::Lower) {
        Factor::Llt(f)
    } else {
        Factor::Ldlt(FaerLdlt::new(k_view.as_ref(), Side::Lower).map_err(|_| {
            EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            }
        })?)
    };

    let ut = u.t();
    let xtwx = ut.dot(&u);

    let phi = match link {
        LinkFunction::Logit => 1.0,
        LinkFunction::Identity => {
            let mut rss = 0.0;
            for i in 0..n {
                let r = y[i] - base.final_mu[i];
                let wi = base.final_weights[i];
                rss += wi * r * r;
            }
            let dof = (n as f64) - base.edf;
            let denom = dof.max(1.0);
            rss / denom
        }
    };

    let xtwx_view = FaerArrayView::new(&xtwx);
    let mut aii = Array1::<f64>::zeros(n);
    let mut se_naive = Array1::<f64>::zeros(n);
    let eta_hat = x_dense.dot(base.beta_transformed.as_ref());
    let z = &base.solve_working_response;

    let mut diag_counter = 0;
    let max_diag_samples = 5;

    let mut percentiles_data = Vec::with_capacity(n);
    let mut sum_aii = 0.0_f64;
    let mut max_aii = f64::NEG_INFINITY;
    let mut invalid_count = 0usize;
    let mut high_leverage_count = 0usize;
    let mut a_hi_90 = 0usize;
    let mut a_hi_95 = 0usize;
    let mut a_hi_99 = 0usize;

    let block_cols = 8192usize;

    let mut rhs_chunk_buf = Array2::<f64>::zeros((p, block_cols));
    let mut t_chunk_storage = FaerMat::<f64>::zeros(p, block_cols);

    for chunk_start in (0..n).step_by(block_cols) {
        let chunk_end = (chunk_start + block_cols).min(n);
        let width = chunk_end - chunk_start;

        rhs_chunk_buf
            .slice_mut(s![.., ..width])
            .assign(&ut.slice(s![.., chunk_start..chunk_end]));

        let rhs_chunk_view = rhs_chunk_buf.slice(s![.., ..width]);
        let rhs_chunk = FaerArrayView::new(&rhs_chunk_view);
        let s_chunk = factor.solve(rhs_chunk.as_ref());
        unsafe {
            t_chunk_storage.set_dims(p, width);
        }
        matmul(
            t_chunk_storage.as_mut(),
            Accum::Replace,
            xtwx_view.as_ref(),
            s_chunk.as_ref(),
            1.0,
            Par::Seq,
        );
        let t_chunk = t_chunk_storage.as_ref();
        let s_col_stride = s_chunk.col_stride();
        let t_col_stride = t_chunk.col_stride();
        assert!(s_col_stride >= 0 && t_col_stride >= 0);
        let s_col_stride = s_col_stride as usize;
        let t_col_stride = t_col_stride as usize;
        let s_ptr = s_chunk.as_ptr();
        let t_ptr = t_chunk.as_ptr();

        for local_col in 0..width {
            let obs = chunk_start + local_col;
            let u_row = u.row(obs);
            let s_col = unsafe { std::slice::from_raw_parts(s_ptr.add(local_col * s_col_stride), p) };
            let t_col = unsafe { std::slice::from_raw_parts(t_ptr.add(local_col * t_col_stride), p) };
            let mut ai = 0.0f64;
            let mut quad = 0.0f64;
            for ((&s_val, &t_val), &u_val) in s_col.iter().zip(t_col.iter()).zip(u_row.iter()) {
                ai = s_val.mul_add(u_val, ai);
                quad = s_val.mul_add(t_val, quad);
            }
            aii[obs] = ai;
            percentiles_data.push(ai);

            if ai.is_finite() {
                sum_aii += ai;
            } else {
                sum_aii = f64::NAN;
            }

            if ai.is_finite() {
                max_aii = max_aii.max(ai);
            }

            if !(0.0..=1.0).contains(&ai) || !ai.is_finite() {
                invalid_count += 1;
                eprintln!("[CAL] WARNING: Invalid leverage at i={}, a_ii={:.6e}", obs, ai);
            } else if ai > 0.99 {
                high_leverage_count += 1;
                if ai > 0.999 {
                    eprintln!("[CAL] Very high leverage at i={}, a_ii={:.6e}", obs, ai);
                }
            }

            if ai > 0.90 {
                a_hi_90 += 1;
            }
            if ai > 0.95 {
                a_hi_95 += 1;
            }
            if ai > 0.99 {
                a_hi_99 += 1;
            }

            let wi = base.final_weights[obs].max(1e-12);
            let var_full = phi * (quad / wi);
            if var_full == 0.0 && base.final_weights[obs] < 1e-10 {
                eprintln!(
                    "[CAL] WARNING: obs {} has near-zero weight ({:.2e}) resulting in SE=0",
                    obs, base.final_weights[obs]
                );
            }
            let se_full = var_full.max(0.0).sqrt();
            se_naive[obs] = se_full;

            if diag_counter < max_diag_samples {
                println!("[GAM DIAG] SE formula (obs {}):", obs);
                println!("  - w_i: {:.6e}", wi);
                println!("  - a_ii: {:.6e}", ai);
                println!("  - var_full: {:.6e}", var_full);
                println!("  - SE_naive: {:.6e}", se_full);
                diag_counter += 1;
            }
        }
    }

    if invalid_count > 0 || high_leverage_count > 0 {
        eprintln!(
            "[CAL] Leverage diagnostics: {} invalid values, {} high values (>0.99)",
            invalid_count, high_leverage_count
        );
    }

    let mut eta_tilde = Array1::<f64>::zeros(n);
    for i in 0..n {
        let denom_raw = 1.0 - aii[i];
        let denom = if denom_raw <= 1e-12 {
            eprintln!("[CAL] WARNING: 1 - a_ii ≤ eps at i={}, a_ii={:.6e}", i, aii[i]);
            1e-12
        } else {
            denom_raw
        };

        if denom <= 1e-4 {
            eprintln!("[CAL] ALO 1-a_ii very small at i={}, a_ii={:.6e}", i, aii[i]);
        }

        eta_tilde[i] = (eta_hat[i] - aii[i] * z[i]) / denom;

        if !eta_tilde[i].is_finite() || eta_tilde[i].abs() > 1e6 {
            eprintln!(
                "[CAL] ALO eta_tilde extreme value at i={}: {}, capping",
                i, eta_tilde[i]
            );
            eta_tilde[i] = eta_tilde[i].clamp(-1e6, 1e6);
        }
    }

    let mut percentiles = percentiles_data;

    let p50_idx = if n > 1 {
        ((0.50_f64 * (n as f64 - 1.0)).round() as usize).min(n - 1)
    } else {
        0
    };
    let p95_idx = if n > 1 {
        ((0.95_f64 * (n as f64 - 1.0)).round() as usize).min(n - 1)
    } else {
        0
    };
    let p99_idx = if n > 1 {
        ((0.99_f64 * (n as f64 - 1.0)).round() as usize).min(n - 1)
    } else {
        0
    };

    let mut percentile_value = |idx: usize| -> f64 {
        if percentiles.is_empty() {
            0.0
        } else {
            let target = idx.min(percentiles.len() - 1);
            let (_, nth, _) = percentiles
                .select_nth_unstable_by(target, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            *nth
        }
    };

    let a_mean: f64 = if n == 0 { 0.0 } else { sum_aii / (n as f64) };
    let a_median = percentile_value(p50_idx);
    let a_p95 = percentile_value(p95_idx);
    let a_p99 = percentile_value(p99_idx);
    let a_max = if max_aii.is_finite() { max_aii } else { 0.0 };

    eprintln!(
        "[CAL] ALO leverage: n={}, mean={:.3e}, median={:.3e}, p95={:.3e}, p99={:.3e}, max={:.3e}",
        n, a_mean, a_median, a_p95, a_p99, a_max
    );
    eprintln!(
        "[CAL] ALO high-leverage: a>0.90: {:.2}%, a>0.95: {:.2}%, a>0.99: {:.2}%, dispersion phi={:.3e}",
        100.0 * (a_hi_90 as f64) / (n as f64).max(1.0),
        100.0 * (a_hi_95 as f64) / (n as f64).max(1.0),
        100.0 * (a_hi_99 as f64) / (n as f64).max(1.0),
        phi
    );

    let dist = if let Some(hull) = hull_opt {
        hull.signed_distance_many(raw_train)
    } else {
        Array1::zeros(raw_train.nrows())
    };

    let pred = match link {
        LinkFunction::Logit => eta_tilde,
        LinkFunction::Identity => eta_tilde,
    };

    let has_nan_pred = pred.iter().any(|&x| x.is_nan());
    let has_nan_se = se_naive.iter().any(|&x| x.is_nan());
    let has_nan_dist = dist.iter().any(|&x| x.is_nan());

    if has_nan_pred || has_nan_se || has_nan_dist {
        eprintln!("[CAL] ERROR: NaN values found in ALO features:");
        eprintln!(
            "      - pred: {} NaN values",
            pred.iter().filter(|&&x| x.is_nan()).count()
        );
        eprintln!(
            "      - se: {} NaN values",
            se_naive.iter().filter(|&&x| x.is_nan()).count()
        );
        eprintln!(
            "      - dist: {} NaN values",
            dist.iter().filter(|&&x| x.is_nan()).count()
        );
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }

    Ok(CalibratorFeatures {
        pred,
        se: se_naive,
        dist,
        pred_identity: eta_hat,
        fisher_weights: base.final_weights.clone(),
    })
}

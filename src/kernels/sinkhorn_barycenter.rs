//! Log-domain entropic Sinkhorn Wasserstein barycenter.
//!
//! Implements the iterative Bregman projection scheme of
//! Benamou et al. 2015, "Iterative Bregman Projections for Regularized
//! Transportation Problems", SIAM J. Sci. Comput. 37(2):A1111–A1138.
//! The algorithm computes the entropy-regularized Wasserstein
//! barycenter of `K` discrete probability distributions sharing a
//! common support of size `M`:
//!
//! ```text
//!     a* = argmin_{a in simplex(M)}  sum_k w_k * W_eps(a, atoms_k)
//! ```
//!
//! where `W_eps` is the entropic OT cost with regularization `eps` and
//! ground cost `cost[i,j]`. All updates run in the log domain with
//! `logsumexp` for numerical stability — no naive
//! `log(sum(exp(...)))` is ever evaluated, so the kernel does not
//! overflow at small `eps`.
//!
//! ## Stability guarantees
//!
//! * `eps >= 1e-12` is required (smaller is rejected with `Err`).
//! * Input atom rows with truly-zero mass on a support point are
//!   handled via a large-negative sentinel (`LOG_ZERO_SENTINEL`)
//!   instead of `-inf`, so additions of `+inf` (from the kernel) and
//!   `-inf` (from the log) do not produce `NaN`. The mathematical
//!   contract is unchanged: a support point with zero atom mass and
//!   finite cost remains a valid kernel argument; only the gradient
//!   flow through that point is gracefully damped.
//! * Up to `(K=128, M=256)`: peak working memory is `K * M * 8` for
//!   the two log-dual potentials plus `M * M * 8` for the log-kernel,
//!   which is `<= 1 MiB` — no OOM under standard test machines.
//!
//! ## Differentiability
//!
//! The forward pass returns the log-barycenter after exactly `n_iter`
//! Sinkhorn updates. The companion [`sinkhorn_barycenter_vjp`] computes
//! the vector-Jacobian product of that same finite-iteration map by a
//! reverse sweep through the recorded dual trajectory. This keeps the
//! autograd contract exact even when the public default `n_iter` has
//! not reached the fixed point.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Largest non-`-inf` value used in place of `log(0)` for input atom
/// entries with truly-zero mass. Chosen so that adding it to a kernel
/// row (which has values in roughly `[-eps^{-1} * cost_max, 0]`)
/// still saturates near `LOG_ZERO_SENTINEL` after a `logsumexp` — i.e.
/// the corresponding row's contribution vanishes from the barycenter,
/// which is exactly the desired mathematical behaviour.
pub const LOG_ZERO_SENTINEL: f64 = -1.0e300;

const LOG_ZERO_SATURATION_THRESHOLD: f64 = LOG_ZERO_SENTINEL * 0.5;

/// Lower bound on the regularization parameter `eps`. Below this the
/// log-kernel exponents would lose more than 52 bits of mantissa even
/// for unit-scale costs; we refuse to run instead of silently producing
/// garbage.
pub const MIN_EPS: f64 = 1.0e-12;

/// Stabilized `logsumexp` of `log_kernel[i, j] + off[i]` over the first axis
/// `i`, returning an `(M,)` vector indexed by `j`.
///
/// This is the core Sinkhorn projection kernel. It is mathematically identical
/// to materializing the `(M, M)` matrix `scratch[i, j] = log_kernel[i, j] +
/// off[i]` and taking a stabilized column-wise `logsumexp`, but it never
/// allocates or fills that matrix: it folds directly over each column of
/// `log_kernel` zipped with
/// `off` using slice iterators. Eliminating the per-(atom, iteration) `(M, M)`
/// scratch fill — and the double-`[[i, j]]` indexing that fill performed — is
/// what brings the per-iteration cost down to the matvec form the kernel
/// advertises (gam#852). The column-max subtraction preserves the exact
/// log-domain stability guarantee (no underflow at small `eps`).
fn logsumexp_kernel_plus_offset_axis0(
    log_kernel: ArrayView2<'_, f64>,
    off: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let (m_rows, m_cols) = log_kernel.dim();
    let mut out = Array1::<f64>::from_elem(m_cols, LOG_ZERO_SENTINEL);
    if m_rows == 0 {
        return out;
    }
    for j in 0..m_cols {
        let col = log_kernel.column(j);
        let mut col_max = f64::NEG_INFINITY;
        for (&k, &o) in col.iter().zip(off.iter()) {
            let value = k + o;
            if value > col_max {
                col_max = value;
            }
        }
        if !col_max.is_finite() || col_max <= LOG_ZERO_SATURATION_THRESHOLD {
            out[j] = LOG_ZERO_SENTINEL;
            continue;
        }
        let mut acc = 0.0_f64;
        for (&k, &o) in col.iter().zip(off.iter()) {
            acc += (k + o - col_max).exp();
        }
        out[j] = if acc > 0.0 {
            col_max + acc.ln()
        } else {
            LOG_ZERO_SENTINEL
        };
    }
    out
}

/// Stabilized `logsumexp` of `log_kernel[i, j] + off[j]` over the second axis
/// `j`, returning an `(M,)` vector indexed by `i`. Row-oriented dual of
/// [`logsumexp_kernel_plus_offset_axis0`]; avoids the `(M, M)` scratch fill
/// (gam#852).
fn logsumexp_kernel_plus_offset_axis1(
    log_kernel: ArrayView2<'_, f64>,
    off: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let (m_rows, m_cols) = log_kernel.dim();
    let mut out = Array1::<f64>::from_elem(m_rows, LOG_ZERO_SENTINEL);
    if m_cols == 0 {
        return out;
    }
    for i in 0..m_rows {
        let row = log_kernel.row(i);
        let mut row_max = f64::NEG_INFINITY;
        for (&k, &o) in row.iter().zip(off.iter()) {
            let value = k + o;
            if value > row_max {
                row_max = value;
            }
        }
        if !row_max.is_finite() || row_max <= LOG_ZERO_SATURATION_THRESHOLD {
            out[i] = LOG_ZERO_SENTINEL;
            continue;
        }
        let mut acc = 0.0_f64;
        for (&k, &o) in row.iter().zip(off.iter()) {
            acc += (k + o - row_max).exp();
        }
        out[i] = if acc > 0.0 {
            row_max + acc.ln()
        } else {
            LOG_ZERO_SENTINEL
        };
    }
    out
}

fn log_vector_is_sentinel_saturated(log_x: ArrayView1<'_, f64>) -> bool {
    let mut max = f64::NEG_INFINITY;
    for &v in log_x.iter() {
        if v > max {
            max = v;
        }
    }
    !max.is_finite() || max <= LOG_ZERO_SATURATION_THRESHOLD
}

/// Numerically-stable softmax of a 1-D log-vector. Subtracts the max
/// before `exp`, then re-normalizes. Returns a simplex (sums to 1).
/// Sentinel-saturated vectors are rejected instead of normalized into
/// a misleading uniform distribution.
fn softmax_1d(log_x: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
    let m = log_x.len();
    if m == 0 {
        return Ok(Array1::zeros(0));
    }
    let mut max = f64::NEG_INFINITY;
    for &v in log_x.iter() {
        if v > max {
            max = v;
        }
    }
    if !max.is_finite() || max <= LOG_ZERO_SATURATION_THRESHOLD {
        return Err(
            "sinkhorn barycenter degenerated: all log_a saturated to sentinel -- try larger eps or check cost matrix"
                .to_string(),
        );
    }
    let mut out = Array1::<f64>::zeros(m);
    let mut total = 0.0_f64;
    for (i, &v) in log_x.iter().enumerate() {
        let e = (v - max).exp();
        out[i] = e;
        total += e;
    }
    if total <= 0.0 {
        return Err(
            "sinkhorn barycenter degenerated: softmax mass underflowed -- try larger eps or check cost matrix"
                .to_string(),
        );
    }
    for v in out.iter_mut() {
        *v /= total;
    }
    Ok(out)
}

/// Compute the elementwise log of a simplex vector, replacing exact
/// zeros with [`LOG_ZERO_SENTINEL`] (never `-inf`).
fn safe_log_simplex(row: ArrayView1<'_, f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(row.len());
    for (i, &v) in row.iter().enumerate() {
        out[i] = if v <= 0.0 { LOG_ZERO_SENTINEL } else { v.ln() };
    }
    out
}

/// Validate the shapes and contents of the Sinkhorn-barycenter inputs.
fn validate_inputs(
    atoms: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    cost: ArrayView2<'_, f64>,
    eps: f64,
    n_iter: usize,
) -> Result<(), String> {
    let (k, m) = atoms.dim();
    if k == 0 || m == 0 {
        return Err("atoms must have at least one row and one column".to_string());
    }
    if weights.len() != k {
        return Err(format!(
            "weights length {} does not match atoms row count {}",
            weights.len(),
            k
        ));
    }
    let (cm_r, cm_c) = cost.dim();
    if cm_r != m || cm_c != m {
        return Err(format!(
            "cost matrix must be ({}, {}), got ({}, {})",
            m, m, cm_r, cm_c
        ));
    }
    if !(eps.is_finite() && eps >= MIN_EPS) {
        return Err(format!("eps must be finite and >= {MIN_EPS:e}, got {eps}"));
    }
    if n_iter == 0 {
        return Err("n_iter must be at least 1".to_string());
    }
    for ((row, col), value) in atoms.indexed_iter() {
        if !value.is_finite() || *value < 0.0 {
            return Err(format!(
                "atoms must be finite and non-negative; got {value} at ({row}, {col})"
            ));
        }
    }
    let mut w_total = 0.0_f64;
    for &w in weights.iter() {
        if !w.is_finite() || w < 0.0 {
            return Err("weights must be finite and non-negative".to_string());
        }
        w_total += w;
    }
    if w_total <= 0.0 {
        return Err("weights must have positive total mass".to_string());
    }
    for ((i, j), value) in cost.indexed_iter() {
        if !value.is_finite() || *value < 0.0 {
            return Err(format!(
                "cost must be finite and non-negative; got {value} at ({i}, {j})"
            ));
        }
    }
    Ok(())
}

/// Normalize each row of an `(K, M)` atom matrix to sum to one,
/// returning a fresh `(K, M)` matrix. Rows whose mass is non-positive
/// are rejected; this is caller-safe because [`validate_inputs`]
/// already guarantees non-negative entries.
fn normalize_atoms(atoms: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let (k, m) = atoms.dim();
    let mut out = Array2::<f64>::zeros((k, m));
    for ki in 0..k {
        let mut total = 0.0_f64;
        for j in 0..m {
            total += atoms[[ki, j]];
        }
        if !(total > 0.0) {
            return Err(format!(
                "atoms row {ki} has non-positive total mass {total}"
            ));
        }
        for j in 0..m {
            out[[ki, j]] = atoms[[ki, j]] / total;
        }
    }
    Ok(out)
}

/// Normalize the weight vector to sum to one.
fn normalize_weights(weights: ArrayView1<'_, f64>) -> Vec<f64> {
    let total: f64 = weights.iter().sum();
    weights.iter().map(|w| w / total).collect()
}

/// Output of [`sinkhorn_barycenter_forward_state`] — exposes the final
/// dual state after the requested finite Sinkhorn iteration count.
pub struct SinkhornState {
    /// `(K, M)` log dual potentials on the "data-fit" side.
    pub log_u: Array2<f64>,
    /// `(K, M)` log dual potentials on the "barycenter" side.
    pub log_v: Array2<f64>,
    /// `(M,)` log of the entropic barycenter.
    pub log_a: Array1<f64>,
    /// `(M, M)` precomputed log kernel `-cost / eps`.
    pub log_kernel: Array2<f64>,
    /// `(K, M)` log of the (sanitized) input atoms.
    pub log_atoms: Array2<f64>,
    /// `(K,)` normalized mixing weights.
    pub weights: Vec<f64>,
}

/// Run the log-domain Sinkhorn barycenter forward pass and return the
/// full dual state. Use [`sinkhorn_barycenter`] for the simpler "just
/// the barycenter" entry point.
pub fn sinkhorn_barycenter_forward_state(
    atoms: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    cost: ArrayView2<'_, f64>,
    eps: f64,
    n_iter: usize,
) -> Result<SinkhornState, String> {
    validate_inputs(atoms, weights, cost, eps, n_iter)?;
    let atoms_norm = normalize_atoms(atoms)?;
    let weights_norm = normalize_weights(weights);
    let (k, m) = atoms_norm.dim();

    // Precompute log kernel: K_log[i, j] = -cost[i, j] / eps.
    let mut log_kernel = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            log_kernel[[i, j]] = -cost[[i, j]] / eps;
        }
    }

    // Precompute log atoms (sentinel-protected log of zeros).
    let mut log_atoms = Array2::<f64>::zeros((k, m));
    for ki in 0..k {
        let row = safe_log_simplex(atoms_norm.row(ki));
        for j in 0..m {
            log_atoms[[ki, j]] = row[j];
        }
    }

    let mut log_u = Array2::<f64>::zeros((k, m));
    let mut log_v = Array2::<f64>::zeros((k, m));
    let inv_m = (1.0_f64 / m as f64).ln();
    let mut log_a = Array1::<f64>::from_elem(m, inv_m);

    for _ in 0..n_iter {
        // Step 1: log_v[k, :] = log_a - logsumexp_i(log_kernel[i, :] + log_u[k, i]).
        for ki in 0..k {
            let lse = logsumexp_kernel_plus_offset_axis0(log_kernel.view(), log_u.row(ki));
            for j in 0..m {
                log_v[[ki, j]] = log_a[j] - lse[j];
            }
        }

        // Step 2: log_u[k, :] = log_atoms[k, :] - logsumexp_j(log_kernel[:, j] + log_v[k, j]).
        for ki in 0..k {
            let lse = logsumexp_kernel_plus_offset_axis1(log_kernel.view(), log_v.row(ki));
            for i in 0..m {
                log_u[[ki, i]] = log_atoms[[ki, i]] - lse[i];
            }
        }

        // Step 3: log_a[i] = sum_k w_k * logsumexp_i'(log_kernel[i', i] + log_u[k, i']).
        let mut next_log_a = Array1::<f64>::zeros(m);
        for ki in 0..k {
            let lse = logsumexp_kernel_plus_offset_axis0(log_kernel.view(), log_u.row(ki));
            for i in 0..m {
                next_log_a[i] += weights_norm[ki] * lse[i];
            }
        }
        log_a = next_log_a;
    }

    Ok(SinkhornState {
        log_u,
        log_v,
        log_a,
        log_kernel,
        log_atoms,
        weights: weights_norm,
    })
}

/// Convenience forward: returns the converged barycenter as a simplex
/// vector of length `M`.
pub fn sinkhorn_barycenter(
    atoms: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    cost: ArrayView2<'_, f64>,
    eps: f64,
    n_iter: usize,
) -> Result<Array1<f64>, String> {
    let state = sinkhorn_barycenter_forward_state(atoms, weights, cost, eps, n_iter)?;
    if log_vector_is_sentinel_saturated(state.log_a.view()) {
        return Err(
            "sinkhorn barycenter degenerated: all log_a saturated to sentinel -- try larger eps or check cost matrix"
                .to_string(),
        );
    }
    softmax_1d(state.log_a.view())
}

/// Output of [`sinkhorn_barycenter_vjp`]: gradients w.r.t. the input
/// `atoms` and `weights` for a given cotangent vector on the
/// `(M,)` barycenter output.
pub struct SinkhornVjp {
    /// `(K, M)` — gradient w.r.t. the (un-normalized) atoms.
    pub d_atoms: Array2<f64>,
    /// `(K,)` — gradient w.r.t. the (un-normalized) mixing weights.
    pub d_weights: Array1<f64>,
}

/// Vector-Jacobian product of the finite-iteration Sinkhorn-barycenter map.
///
/// Given a cotangent `cotangent` of shape `(M,)` (the upstream
/// gradient of a scalar loss w.r.t. the output barycenter), this
/// returns `(dL/d_atoms, dL/d_weights)` for the same truncated
/// `n_iter` computation that [`sinkhorn_barycenter`] returns. This is
/// intentionally not the fixed-point / IFT adjoint: at small `eps` the
/// public default `n_iter` can be far from convergence, so differentiating
/// the converged fixed point would produce the gradient of a different
/// function than the one used in the forward pass.
pub fn sinkhorn_barycenter_vjp(
    atoms: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    cost: ArrayView2<'_, f64>,
    eps: f64,
    n_iter: usize,
    cotangent: ArrayView1<'_, f64>,
) -> Result<SinkhornVjp, String> {
    validate_inputs(atoms, weights, cost, eps, n_iter)?;
    let atoms_norm = normalize_atoms(atoms)?;
    let weights_norm = normalize_weights(weights);
    let (k, m) = atoms_norm.dim();
    if cotangent.len() != m {
        return Err(format!(
            "cotangent length {} does not match barycenter size {}",
            cotangent.len(),
            m
        ));
    }

    let mut log_kernel = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            log_kernel[[i, j]] = -cost[[i, j]] / eps;
        }
    }

    let mut log_atoms = Array2::<f64>::zeros((k, m));
    for ki in 0..k {
        let row = safe_log_simplex(atoms_norm.row(ki));
        for j in 0..m {
            log_atoms[[ki, j]] = row[j];
        }
    }

    let inv_m = (1.0_f64 / m as f64).ln();
    let mut log_u = Array2::<f64>::zeros((k, m));
    let mut log_a = Array1::<f64>::from_elem(m, inv_m);
    let mut log_u_hist: Vec<Array2<f64>> = Vec::with_capacity(n_iter + 1);
    let mut log_a_hist: Vec<Array1<f64>> = Vec::with_capacity(n_iter + 1);
    let mut log_v_hist: Vec<Array2<f64>> = Vec::with_capacity(n_iter);
    log_u_hist.push(log_u.clone());
    log_a_hist.push(log_a.clone());

    for _ in 0..n_iter {
        let mut log_v = Array2::<f64>::zeros((k, m));
        for ki in 0..k {
            let lse = logsumexp_kernel_plus_offset_axis0(log_kernel.view(), log_u.row(ki));
            for j in 0..m {
                log_v[[ki, j]] = log_a[j] - lse[j];
            }
        }

        let mut next_log_u = Array2::<f64>::zeros((k, m));
        for ki in 0..k {
            let lse = logsumexp_kernel_plus_offset_axis1(log_kernel.view(), log_v.row(ki));
            for i in 0..m {
                next_log_u[[ki, i]] = log_atoms[[ki, i]] - lse[i];
            }
        }

        let mut next_log_a = Array1::<f64>::zeros(m);
        for ki in 0..k {
            let lse = logsumexp_kernel_plus_offset_axis0(log_kernel.view(), next_log_u.row(ki));
            for j in 0..m {
                next_log_a[j] += weights_norm[ki] * lse[j];
            }
        }

        log_v_hist.push(log_v);
        log_u = next_log_u;
        log_a = next_log_a;
        log_u_hist.push(log_u.clone());
        log_a_hist.push(log_a.clone());
    }

    // Step A: pull the cotangent through the softmax(log_a) -> a mapping.
    // d(softmax(z))/dz = diag(p) - p p^T, so g_log_a = p .* (cot - sum(cot * p)).
    if log_vector_is_sentinel_saturated(log_a.view()) {
        return Err(
            "sinkhorn barycenter degenerated: all log_a saturated to sentinel -- try larger eps or check cost matrix"
                .to_string(),
        );
    }
    let bary = softmax_1d(log_a.view())?;
    let mut g_log_a = Array1::<f64>::zeros(m);
    let mut weighted = 0.0_f64;
    for i in 0..m {
        weighted += cotangent[i] * bary[i];
    }
    for i in 0..m {
        g_log_a[i] = bary[i] * (cotangent[i] - weighted);
    }

    let mut g_log_u_next = Array2::<f64>::zeros((k, m));
    let mut g_log_a_next = g_log_a;
    let mut g_log_atoms = Array2::<f64>::zeros((k, m));
    let mut g_weights = Array1::<f64>::zeros(k);

    for iter in (0..n_iter).rev() {
        let log_u_prev = &log_u_hist[iter];
        let log_a_prev = &log_a_hist[iter];
        let log_v_new = &log_v_hist[iter];
        let log_u_new = &log_u_hist[iter + 1];

        let mut g_log_u_new = g_log_u_next;
        let mut g_log_v_new = Array2::<f64>::zeros((k, m));
        let mut g_log_a_prev = Array1::<f64>::zeros(m);
        let mut g_log_u_prev = Array2::<f64>::zeros((k, m));

        // Step 3 backward:
        //   log_a_new[j] = sum_k w_k * LSE_i(log_kernel[i,j] + log_u_new[k,i]).
        for ki in 0..k {
            let s = logsumexp_kernel_plus_offset_axis0(log_kernel.view(), log_u_new.row(ki));
            for j in 0..m {
                let ga = g_log_a_next[j];
                g_weights[ki] += ga * s[j];
                if ga == 0.0 {
                    continue;
                }
                for i in 0..m {
                    let soft = (log_kernel[[i, j]] + log_u_new[[ki, i]] - s[j]).exp();
                    g_log_u_new[[ki, i]] += ga * weights_norm[ki] * soft;
                }
            }
        }

        // Step 2 backward:
        //   log_u_new[k,i] = log_atoms[k,i] - LSE_j(log_kernel[i,j] + log_v_new[k,j]).
        for ki in 0..k {
            for i in 0..m {
                let gu = g_log_u_new[[ki, i]];
                g_log_atoms[[ki, i]] += gu;
                if gu == 0.0 {
                    continue;
                }
                let lse = log_atoms[[ki, i]] - log_u_new[[ki, i]];
                for j in 0..m {
                    let soft = (log_kernel[[i, j]] + log_v_new[[ki, j]] - lse).exp();
                    g_log_v_new[[ki, j]] -= gu * soft;
                }
            }
        }

        // Step 1 backward:
        //   log_v_new[k,j] = log_a_prev[j] - LSE_i(log_kernel[i,j] + log_u_prev[k,i]).
        for ki in 0..k {
            for j in 0..m {
                let gv = g_log_v_new[[ki, j]];
                g_log_a_prev[j] += gv;
                if gv == 0.0 {
                    continue;
                }
                let lse = log_a_prev[j] - log_v_new[[ki, j]];
                for i in 0..m {
                    let soft = (log_kernel[[i, j]] + log_u_prev[[ki, i]] - lse).exp();
                    g_log_u_prev[[ki, i]] -= gv * soft;
                }
            }
        }

        g_log_u_next = g_log_u_prev;
        g_log_a_next = g_log_a_prev;
    }

    // Now convert g_log_atoms (gradient w.r.t. log of normalized atoms)
    // back to a gradient w.r.t. raw atoms.
    //
    // normalized[k, i] = raw[k, i] / sum_j raw[k, j].
    // log(normalized[k, i]) = log(raw[k, i]) - log(sum_j raw[k, j]).
    // d log(normalized[k, i]) / d raw[k, l]
    //   = (i == l) / raw[k, i] - 1 / sum_j raw[k, j]
    //   = (i == l) / raw[k, i] - 1 / Z_k.
    //
    // So d_raw[k, l] = sum_i g_log_atoms[k, i] * d log_norm[k, i] / d raw[k, l]
    //                = g_log_atoms[k, l] / raw[k, l] - (sum_i g_log_atoms[k, i]) / Z_k.

    let mut d_atoms = Array2::<f64>::zeros((k, m));
    for ki in 0..k {
        let mut z = 0.0_f64;
        for j in 0..m {
            z += atoms[[ki, j]];
        }
        let mut sum_g = 0.0_f64;
        for i in 0..m {
            sum_g += g_log_atoms[[ki, i]];
        }
        for l in 0..m {
            let raw = atoms[[ki, l]];
            // Skip ill-defined gradient for support points with zero
            // mass (the sentinel logarithm has no usable derivative);
            // these points have measure-zero contribution to the
            // barycenter so dropping them is mathematically sound.
            let first = if raw > 0.0 {
                g_log_atoms[[ki, l]] / raw
            } else {
                0.0
            };
            d_atoms[[ki, l]] = first - sum_g / z;
        }
    }

    // Convert g_weights (raw, un-normalized) similarly:
    // w_norm[k] = w_raw[k] / W, W = sum_l w_raw[l].
    // ∂log_a depends on w_norm only, and g_weights above was computed
    // against w_norm[ki] directly. So convert: ∂w_norm[k]/∂w_raw[l] =
    // (k == l) / W - w_raw[k] / W^2 = (1/W) * ((k == l) - w_norm[k]).
    let mut d_weights = Array1::<f64>::zeros(k);
    let w_total: f64 = weights.iter().sum();
    if w_total > 0.0 {
        let mut sum_norm_g = 0.0_f64;
        for ki in 0..k {
            sum_norm_g += g_weights[ki] * weights_norm[ki];
        }
        for ki in 0..k {
            d_weights[ki] = (g_weights[ki] - sum_norm_g) / w_total;
        }
    }

    Ok(SinkhornVjp { d_atoms, d_weights })
}

// =====================================================================
// Cost-matrix helpers
// =====================================================================

/// Squared circular distance on a length-`m` cycle:
/// `c[i, j] = (min(|i - j|, m - |i - j|))^2`.
pub fn circular_cost(m: usize) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((m, m));
    if m == 0 {
        return out;
    }
    for i in 0..m {
        for j in 0..m {
            let diff = if i >= j { i - j } else { j - i };
            let d = diff.min(m - diff);
            let dd = d as f64;
            out[[i, j]] = dd * dd;
        }
    }
    out
}

/// Squared Euclidean distance from an `(M, d)` array of support points.
pub fn euclidean_cost(points: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let (m, d) = points.dim();
    if m == 0 || d == 0 {
        return Err("euclidean_cost requires at least one point and one dimension".to_string());
    }
    for ((row, col), value) in points.indexed_iter() {
        if !value.is_finite() {
            return Err(format!(
                "euclidean_cost points must be finite; got {value} at ({row}, {col})"
            ));
        }
    }
    let mut out = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0_f64;
            for k in 0..d {
                let diff = points[[i, k]] - points[[j, k]];
                acc += diff * diff;
            }
            out[[i, j]] = acc;
        }
    }
    Ok(out)
}

/// Squared great-circle (geodesic) distance on the unit 2-sphere from
/// `(M, 3)` direction vectors.
///
/// Each row must lie within `1e-6` of the unit sphere; rows that pass
/// this check are renormalized to exact unit length before any cosine
/// is formed, so the cost is the true squared great-circle distance
///
/// `C_ij = arccos( <x_i/|x_i|, x_j/|x_j|> )^2`
///
/// of the projected directions. This guarantees a symmetric matrix
/// with an exactly-zero diagonal: without renormalization an accepted
/// row with `|x|^2 = 1 - O(1e-6)` would yield `arccos(<x,x>)^2 > 0` on
/// the diagonal, contradicting `d(x, x) = 0`.
pub fn geodesic_sphere_cost(directions: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let (m, d) = directions.dim();
    if d != 3 {
        return Err(format!(
            "geodesic_sphere_cost requires direction vectors of dimension 3, got {d}"
        ));
    }
    let mut unit = Array2::<f64>::zeros((m, 3));
    for i in 0..m {
        let mut norm_sq = 0.0_f64;
        for k in 0..3 {
            let v = directions[[i, k]];
            if !v.is_finite() {
                return Err(format!(
                    "geodesic_sphere_cost directions must be finite; got {v} at ({i}, {k})"
                ));
            }
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        if (norm - 1.0).abs() > 1.0e-6 {
            return Err(format!(
                "geodesic_sphere_cost row {i} must be unit-norm; got |x| = {norm}"
            ));
        }
        for k in 0..3 {
            unit[[i, k]] = directions[[i, k]] / norm;
        }
    }
    let mut out = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            let mut dot = 0.0_f64;
            for k in 0..3 {
                dot += unit[[i, k]] * unit[[j, k]];
            }
            let dot_clamped = dot.clamp(-1.0, 1.0);
            let theta = dot_clamped.acos();
            out[[i, j]] = theta * theta;
        }
        out[[i, i]] = 0.0;
    }
    Ok(out)
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    fn approx_simplex_eq(actual: &Array1<f64>, expected: &Array1<f64>, tol: f64) {
        assert_eq!(actual.len(), expected.len());
        let sum_actual: f64 = actual.iter().sum();
        let sum_expected: f64 = expected.iter().sum();
        assert!(
            (sum_actual - 1.0).abs() < 1.0e-8,
            "actual does not sum to 1: {sum_actual}"
        );
        assert!(
            (sum_expected - 1.0).abs() < 1.0e-8,
            "expected does not sum to 1: {sum_expected}"
        );
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < tol,
                "barycenter entry mismatch: {a} vs {e} (tol {tol})"
            );
        }
    }

    #[test]
    fn k_eq_1_recovers_the_atom() {
        let m = 8;
        let atom = array![0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.04, 0.01];
        let mut atoms = Array2::<f64>::zeros((1, m));
        for j in 0..m {
            atoms[[0, j]] = atom[j];
        }
        let weights = array![1.0];
        let cost = circular_cost(m);
        let bary =
            sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), 0.05, 60).unwrap();
        approx_simplex_eq(&bary, &atom, 5.0e-3);
        assert_eq!(bary.len(), atom.len(), "barycenter length mismatch");
    }

    #[test]
    fn k_eq_2_mean_is_between() {
        let m = 32;
        let points: Array2<f64> = Array2::from_shape_fn((m, 1), |(i, _)| i as f64 / (m - 1) as f64);
        let mut atom_a = Array1::<f64>::zeros(m);
        let mut atom_b = Array1::<f64>::zeros(m);
        // Two Gaussian-like bumps on the line, centred at 0.2 and 0.8.
        let mut sa = 0.0;
        let mut sb = 0.0;
        for j in 0..m {
            let x = j as f64 / (m - 1) as f64;
            let va = (-((x - 0.2) * (x - 0.2)) / 0.005).exp();
            let vb = (-((x - 0.8) * (x - 0.8)) / 0.005).exp();
            atom_a[j] = va;
            atom_b[j] = vb;
            sa += va;
            sb += vb;
        }
        for j in 0..m {
            atom_a[j] /= sa;
            atom_b[j] /= sb;
        }
        let mut atoms = Array2::<f64>::zeros((2, m));
        for j in 0..m {
            atoms[[0, j]] = atom_a[j];
            atoms[[1, j]] = atom_b[j];
        }
        let weights = array![0.5, 0.5];
        let cost = euclidean_cost(points.view()).unwrap();
        let bary =
            sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), 0.005, 200).unwrap();

        let mean_a: f64 = (0..m)
            .map(|j| (j as f64 / (m - 1) as f64) * atom_a[j])
            .sum();
        let mean_b: f64 = (0..m)
            .map(|j| (j as f64 / (m - 1) as f64) * atom_b[j])
            .sum();
        let mean_bary: f64 = (0..m).map(|j| (j as f64 / (m - 1) as f64) * bary[j]).sum();
        let expected_mean = 0.5 * (mean_a + mean_b);
        assert!(
            (mean_bary - expected_mean).abs() < 0.05,
            "bary mean {mean_bary} should sit at midpoint {expected_mean}"
        );
        assert!(
            mean_bary > mean_a && mean_bary < mean_b,
            "bary mean {mean_bary} should be between atom means ({mean_a}, {mean_b})"
        );
    }

    #[test]
    fn cyclic_midpoint_recovers_mccann_interp() {
        // Two unit masses on a length-32 cycle, separated by 8 steps;
        // the McCann (Wasserstein) midpoint is the support midway
        // between them (4 steps offset).
        let m = 32;
        let mut atoms = Array2::<f64>::zeros((2, m));
        // Bumps centred at index 8 and 24 (distance 16, half = 8).
        // McCann midpoint is at index 16 (or equivalently 0, but
        // entropic regularization breaks the tie deterministically).
        for j in 0..m {
            let d_a = (j as i64 - 8).rem_euclid(m as i64);
            let d_a = d_a.min(m as i64 - d_a);
            let d_b = (j as i64 - 24).rem_euclid(m as i64);
            let d_b = d_b.min(m as i64 - d_b);
            atoms[[0, j]] = (-(d_a as f64).powi(2) / 1.5).exp();
            atoms[[1, j]] = (-(d_b as f64).powi(2) / 1.5).exp();
        }
        let weights = array![0.5, 0.5];
        let cost = circular_cost(m);
        let bary =
            sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), 0.5, 200).unwrap();
        // The barycenter must be mass-balanced on a cycle: the
        // mode should be near index 16 (between 8 and 24).
        let mode = bary
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            mode == 16 || mode == 0,
            "barycenter mode {mode} should be midway between atom modes"
        );
    }

    #[test]
    fn small_eps_does_not_nan() {
        let m = 16;
        let atoms = Array2::<f64>::from_shape_fn((2, m), |(k, j)| {
            let centre = if k == 0 { 3.0 } else { 11.0 };
            (-((j as f64 - centre).powi(2)) / 4.0).exp()
        });
        let weights = array![0.5, 0.5];
        let cost = circular_cost(m);
        let bary =
            sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), 1.0e-3, 50).unwrap();
        for v in bary.iter() {
            assert!(v.is_finite(), "barycenter entry {v} is not finite");
            assert!(*v >= 0.0, "barycenter entry {v} is negative");
        }
        let s: f64 = bary.iter().sum();
        assert!((s - 1.0).abs() < 1.0e-8, "barycenter sum {s} != 1");
    }

    #[test]
    fn rejects_small_eps() {
        let m = 4;
        let atoms = Array2::<f64>::from_elem((2, m), 1.0 / m as f64);
        let weights = array![0.5, 0.5];
        let cost = circular_cost(m);
        let err = sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), 1.0e-15, 10);
        assert!(err.is_err());
    }

    #[test]
    fn sentinel_saturated_log_a_returns_error() {
        let m = 3;
        let atoms = Array2::<f64>::from_elem((1, m), 1.0 / m as f64);
        let weights = array![1.0];
        let cost = Array2::<f64>::from_elem((m, m), 1.0e288);
        let err =
            sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), MIN_EPS, 1).unwrap_err();
        assert!(
            err.contains("all log_a saturated to sentinel"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn batch_kbig_produces_valid_simplex_barycenter() {
        let m = 64;
        let k = 128;
        let atoms = Array2::<f64>::from_shape_fn((k, m), |(ki, j)| {
            let centre = (ki as f64) * (m as f64) / (k as f64);
            (-((j as f64 - centre).powi(2)) / 8.0).exp()
        });
        let weights = Array1::<f64>::from_elem(k, 1.0 / k as f64);
        let cost = circular_cost(m);
        let bary = sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), 0.1, 20).unwrap();
        let s: f64 = bary.iter().sum();
        assert!((s - 1.0).abs() < 1.0e-8);
    }

    #[test]
    fn cost_helpers_shape_and_symmetry() {
        let m = 5;
        let cc = circular_cost(m);
        for i in 0..m {
            assert_eq!(cc[[i, i]], 0.0);
            for j in 0..m {
                assert!((cc[[i, j]] - cc[[j, i]]).abs() < 1.0e-12);
                assert!(cc[[i, j]] >= 0.0);
            }
        }
        let pts = Array2::<f64>::from_shape_fn((m, 2), |(i, k)| (i + k) as f64);
        let ec = euclidean_cost(pts.view()).unwrap();
        for i in 0..m {
            assert_eq!(ec[[i, i]], 0.0);
            for j in 0..m {
                assert!((ec[[i, j]] - ec[[j, i]]).abs() < 1.0e-12);
            }
        }
        let dirs = Array2::<f64>::from_shape_fn((3, 3), |(i, k)| if i == k { 1.0 } else { 0.0 });
        let gc = geodesic_sphere_cost(dirs.view()).unwrap();
        for i in 0..3 {
            assert!((gc[[i, i]]).abs() < 1.0e-12);
            for j in 0..3 {
                assert!((gc[[i, j]] - gc[[j, i]]).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn vjp_matches_finite_differences_small() {
        // Tiny problem, large eps, modest n_iter: the finite-difference
        // gradient of a linear functional of the finite-iteration barycenter
        // should match the VJP of that same truncated computation.
        let m = 6;
        let k = 2;
        let atoms = Array2::<f64>::from_shape_fn((k, m), |(ki, j)| {
            let centre = if ki == 0 { 1.5 } else { 4.0 };
            (-((j as f64 - centre).powi(2)) / 2.0).exp()
        });
        let mut atoms_norm = atoms.clone();
        for ki in 0..k {
            let s: f64 = atoms_norm.row(ki).iter().sum();
            for j in 0..m {
                atoms_norm[[ki, j]] /= s;
            }
        }
        let weights = array![0.5, 0.5];
        let cost = circular_cost(m);
        let eps = 0.3;
        let n_iter = 100;

        // Cotangent: dot the barycenter with a fixed vector r.
        let r = Array1::<f64>::from_shape_fn(m, |j| j as f64 - (m as f64 - 1.0) / 2.0);

        let vjp = sinkhorn_barycenter_vjp(
            atoms_norm.view(),
            weights.view(),
            cost.view(),
            eps,
            n_iter,
            r.view(),
        )
        .unwrap();

        // Finite-difference one atom entry.
        let h = 1.0e-5;
        let (ki, j) = (0usize, 2usize);
        let mut atoms_plus = atoms_norm.clone();
        let mut atoms_minus = atoms_norm.clone();
        atoms_plus[[ki, j]] += h;
        atoms_minus[[ki, j]] -= h;
        let b_plus =
            sinkhorn_barycenter(atoms_plus.view(), weights.view(), cost.view(), eps, n_iter)
                .unwrap();
        let b_minus =
            sinkhorn_barycenter(atoms_minus.view(), weights.view(), cost.view(), eps, n_iter)
                .unwrap();
        let mut fd = 0.0_f64;
        for i in 0..m {
            fd += r[i] * (b_plus[i] - b_minus[i]) / (2.0 * h);
        }
        let analytic = vjp.d_atoms[[ki, j]];
        let denom = analytic.abs().max(fd.abs()).max(1.0e-6);
        let rel = (analytic - fd).abs() / denom;
        assert!(
            rel < 1.0e-5,
            "VJP/FD mismatch: analytic={analytic}, fd={fd}, rel={rel}"
        );
    }

    #[test]
    fn vjp_matches_truncated_forward_finite_differences_at_default_small_eps() {
        let m = 6;
        let k = 3;
        let atoms = Array2::<f64>::from_shape_fn((k, m), |(ki, j)| {
            let centre = match ki {
                0 => 0.8,
                1 => 2.6,
                _ => 4.3,
            };
            (-((j as f64 - centre).powi(2)) / 0.7).exp()
        });
        let weights = array![0.25, 0.35, 0.40];
        let cost = circular_cost(m);
        let eps = 0.01;
        let n_iter = 20;
        let r = Array1::<f64>::from_shape_fn(m, |j| j as f64 - (m as f64 - 1.0) / 2.0);

        let vjp = sinkhorn_barycenter_vjp(
            atoms.view(),
            weights.view(),
            cost.view(),
            eps,
            n_iter,
            r.view(),
        )
        .unwrap();

        let h = 1.0e-5;
        for ki in 0..k {
            let mut weights_plus = weights.clone();
            let mut weights_minus = weights.clone();
            weights_plus[ki] += h;
            weights_minus[ki] -= h;
            let b_plus =
                sinkhorn_barycenter(atoms.view(), weights_plus.view(), cost.view(), eps, n_iter)
                    .unwrap();
            let b_minus =
                sinkhorn_barycenter(atoms.view(), weights_minus.view(), cost.view(), eps, n_iter)
                    .unwrap();
            let fd = r.dot(&(b_plus - b_minus)) / (2.0 * h);
            let analytic = vjp.d_weights[ki];
            assert!(
                (analytic - fd).abs() < 1.0e-3,
                "weight {ki} VJP/FD mismatch at truncated default regime: analytic={analytic}, fd={fd}"
            );
        }

        let (ki, j) = (1usize, 3usize);
        let mut atoms_plus = atoms.clone();
        let mut atoms_minus = atoms.clone();
        atoms_plus[[ki, j]] += h;
        atoms_minus[[ki, j]] -= h;
        let b_plus =
            sinkhorn_barycenter(atoms_plus.view(), weights.view(), cost.view(), eps, n_iter)
                .unwrap();
        let b_minus =
            sinkhorn_barycenter(atoms_minus.view(), weights.view(), cost.view(), eps, n_iter)
                .unwrap();
        let fd = r.dot(&(b_plus - b_minus)) / (2.0 * h);
        let analytic = vjp.d_atoms[[ki, j]];
        assert!(
            (analytic - fd).abs() < 1.0e-3,
            "atom ({ki},{j}) VJP/FD mismatch at truncated default regime: analytic={analytic}, fd={fd}"
        );
    }
}

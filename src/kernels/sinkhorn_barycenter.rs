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
//! The forward pass returns the converged log-barycenter. The
//! companion [`sinkhorn_barycenter_vjp`] computes the
//! vector-Jacobian product at the converged fixed point using
//! **adjoint iterations** (Cuturi & Peyré, "Computational Optimal
//! Transport", §9.1.4) — never by unrolling autograd through the
//! Sinkhorn loop. This avoids the `O(n_iter * M^2)` memory blowup of
//! naive unrolled differentiation.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Largest non-`-inf` value used in place of `log(0)` for input atom
/// entries with truly-zero mass. Chosen so that adding it to a kernel
/// row (which has values in roughly `[-eps^{-1} * cost_max, 0]`)
/// still saturates near `LOG_ZERO_SENTINEL` after a `logsumexp` — i.e.
/// the corresponding row's contribution vanishes from the barycenter,
/// which is exactly the desired mathematical behaviour.
pub const LOG_ZERO_SENTINEL: f64 = -1.0e300;

/// Lower bound on the regularization parameter `eps`. Below this the
/// log-kernel exponents would lose more than 52 bits of mantissa even
/// for unit-scale costs; we refuse to run instead of silently producing
/// garbage.
pub const MIN_EPS: f64 = 1.0e-12;

/// Numerically-stable in-place `logsumexp` along the first axis of an
/// `(M, M)` log-matrix `log_x`, returning an `(M,)` vector with
/// `out[j] = log(sum_i exp(log_x[i, j]))`.
///
/// The implementation subtracts the column maximum before the `exp`
/// step, so the result is correct even when individual entries
/// underflow or overflow `f64`. Special cases:
///
/// * if all entries in a column are `-inf` (or the sentinel) the
///   result is `LOG_ZERO_SENTINEL`, never `NaN`.
fn logsumexp_axis0(log_x: ArrayView2<'_, f64>) -> Array1<f64> {
    let (m_rows, m_cols) = log_x.dim();
    let mut out = Array1::<f64>::from_elem(m_cols, LOG_ZERO_SENTINEL);
    if m_rows == 0 {
        return out;
    }
    for j in 0..m_cols {
        let mut col_max = f64::NEG_INFINITY;
        for i in 0..m_rows {
            let value = log_x[[i, j]];
            if value > col_max {
                col_max = value;
            }
        }
        if !col_max.is_finite() || col_max <= LOG_ZERO_SENTINEL * 0.5 {
            out[j] = LOG_ZERO_SENTINEL;
            continue;
        }
        let mut acc = 0.0_f64;
        for i in 0..m_rows {
            acc += (log_x[[i, j]] - col_max).exp();
        }
        out[j] = if acc > 0.0 {
            col_max + acc.ln()
        } else {
            LOG_ZERO_SENTINEL
        };
    }
    out
}

/// Numerically-stable `logsumexp` along the second axis of an
/// `(M, M)` log-matrix `log_x`, returning an `(M,)` vector with
/// `out[i] = log(sum_j exp(log_x[i, j]))`.
fn logsumexp_axis1(log_x: ArrayView2<'_, f64>) -> Array1<f64> {
    let (m_rows, m_cols) = log_x.dim();
    let mut out = Array1::<f64>::from_elem(m_rows, LOG_ZERO_SENTINEL);
    if m_cols == 0 {
        return out;
    }
    for i in 0..m_rows {
        let mut row_max = f64::NEG_INFINITY;
        for j in 0..m_cols {
            let value = log_x[[i, j]];
            if value > row_max {
                row_max = value;
            }
        }
        if !row_max.is_finite() || row_max <= LOG_ZERO_SENTINEL * 0.5 {
            out[i] = LOG_ZERO_SENTINEL;
            continue;
        }
        let mut acc = 0.0_f64;
        for j in 0..m_cols {
            acc += (log_x[[i, j]] - row_max).exp();
        }
        out[i] = if acc > 0.0 {
            row_max + acc.ln()
        } else {
            LOG_ZERO_SENTINEL
        };
    }
    out
}

/// Numerically-stable softmax of a 1-D log-vector. Subtracts the max
/// before `exp`, then re-normalizes. Returns a simplex (sums to 1).
fn softmax_1d(log_x: ArrayView1<'_, f64>) -> Array1<f64> {
    let m = log_x.len();
    if m == 0 {
        return Array1::zeros(0);
    }
    let mut max = f64::NEG_INFINITY;
    for &v in log_x.iter() {
        if v > max {
            max = v;
        }
    }
    if !max.is_finite() {
        return Array1::from_elem(m, 1.0 / m as f64);
    }
    let mut out = Array1::<f64>::zeros(m);
    let mut total = 0.0_f64;
    for (i, &v) in log_x.iter().enumerate() {
        let e = (v - max).exp();
        out[i] = e;
        total += e;
    }
    if total <= 0.0 {
        return Array1::from_elem(m, 1.0 / m as f64);
    }
    for v in out.iter_mut() {
        *v /= total;
    }
    out
}

/// Compute the elementwise log of a simplex vector, replacing exact
/// zeros with [`LOG_ZERO_SENTINEL`] (never `-inf`).
fn safe_log_simplex(row: ArrayView1<'_, f64>) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(row.len());
    for (i, &v) in row.iter().enumerate() {
        out[i] = if v <= 0.0 {
            LOG_ZERO_SENTINEL
        } else {
            v.ln()
        };
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
        return Err(format!(
            "eps must be finite and >= {MIN_EPS:e}, got {eps}"
        ));
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

/// Output of [`sinkhorn_barycenter_forward_state`] — exposes the full
/// converged dual state so [`sinkhorn_barycenter_vjp`] can operate at
/// the fixed point without re-running the forward pass.
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
    /// regularization used.
    pub eps: f64,
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

    // Scratch buffer of shape (M, M) for the broadcasted sum
    // `log_kernel + dual[:, None]`.
    let mut scratch = Array2::<f64>::zeros((m, m));

    for _ in 0..n_iter {
        // Step 1: log_v[k, :] = log_a - logsumexp_i(log_kernel[i, :] + log_u[k, i]).
        for ki in 0..k {
            for i in 0..m {
                let off = log_u[[ki, i]];
                for j in 0..m {
                    scratch[[i, j]] = log_kernel[[i, j]] + off;
                }
            }
            let lse = logsumexp_axis0(scratch.view());
            for j in 0..m {
                log_v[[ki, j]] = log_a[j] - lse[j];
            }
        }

        // Step 2: log_u[k, :] = log_atoms[k, :] - logsumexp_j(log_kernel[:, j] + log_v[k, j]).
        for ki in 0..k {
            for j in 0..m {
                let off = log_v[[ki, j]];
                for i in 0..m {
                    scratch[[i, j]] = log_kernel[[i, j]] + off;
                }
            }
            let lse = logsumexp_axis1(scratch.view());
            for i in 0..m {
                log_u[[ki, i]] = log_atoms[[ki, i]] - lse[i];
            }
        }

        // Step 3: log_a[i] = sum_k w_k * logsumexp_j(log_kernel[j, i] + log_u[k, j])
        // = sum_k w_k * logsumexp_i' over axis 0 since kernel is symmetric in our use.
        // Use axis-0 LSE: scratch[i', i] = log_kernel[i', i] + log_u[k, i'].
        let mut next_log_a = Array1::<f64>::zeros(m);
        for ki in 0..k {
            for i_prime in 0..m {
                let off = log_u[[ki, i_prime]];
                for i in 0..m {
                    scratch[[i_prime, i]] = log_kernel[[i_prime, i]] + off;
                }
            }
            let lse = logsumexp_axis0(scratch.view());
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
        eps,
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
    Ok(softmax_1d(state.log_a.view()))
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

/// Vector-Jacobian product of the Sinkhorn-barycenter map at its
/// converged fixed point.
///
/// Given a cotangent `cotangent` of shape `(M,)` (the upstream
/// gradient of a scalar loss w.r.t. the output barycenter), this
/// returns `(dL/d_atoms, dL/d_weights)` computed by adjoint iteration
/// of the Sinkhorn map. The adjoint iteration is run for the same
/// `n_iter` as the forward pass, which is sufficient for matching the
/// forward convergence (Cuturi & Peyré, COT §9.1.4).
pub fn sinkhorn_barycenter_vjp(
    atoms: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    cost: ArrayView2<'_, f64>,
    eps: f64,
    n_iter: usize,
    cotangent: ArrayView1<'_, f64>,
) -> Result<SinkhornVjp, String> {
    let state = sinkhorn_barycenter_forward_state(atoms, weights, cost, eps, n_iter)?;
    let (k, m) = state.log_u.dim();
    if cotangent.len() != m {
        return Err(format!(
            "cotangent length {} does not match barycenter size {}",
            cotangent.len(),
            m
        ));
    }

    // Step A: pull the cotangent through the softmax(log_a) -> a mapping.
    // d(softmax(z))/dz = diag(p) - p p^T, so g_log_a = p .* (cot - sum(cot * p)).
    let bary = softmax_1d(state.log_a.view());
    let mut g_log_a = Array1::<f64>::zeros(m);
    let mut weighted = 0.0_f64;
    for i in 0..m {
        weighted += cotangent[i] * bary[i];
    }
    for i in 0..m {
        g_log_a[i] = bary[i] * (cotangent[i] - weighted);
    }

    // Adjoint iteration.
    //
    // The forward fixed-point identity, with symmetric kernel, is
    //   log_u[k, i] = log_atoms[k, i] - LSE_j(log_kernel[i, j] + log_v[k, j])
    //   log_v[k, j] = log_a[j]        - LSE_i(log_kernel[i, j] + log_u[k, i])
    //   log_a[j]    = sum_k w_k       * LSE_i(log_kernel[i, j] + log_u[k, i])
    //
    // Let P_k[i, j] = exp(log_kernel[i, j] + log_u[k, i] + log_v[k, j] - log_a[j])
    // be the row-stochastic transport coupling (columns sum to 1
    // because log_v enforces the column-marginal constraint b_k = a).
    // Likewise Q_k[i, j] = exp(log_kernel[i, j] + log_u[k, i] + log_v[k, j] - log_atoms[k, i])
    // is row-stochastic (rows sum to 1 by data-fit constraint).
    //
    // Adjoint sweep (one iteration), with g_log_v, g_log_u
    // accumulators initialized to zero:
    //   g_log_u[k] += -Q_k^T @ <something>; g_log_v[k] += -P_k @ <something>; ...
    //
    // The principled derivation: each Sinkhorn step is a closed-form
    // map, so we propagate cotangents through each step in reverse.
    // We approximate by iterating the forward map once more with
    // adjoints — equivalent in the limit to the IFT linear-system
    // solve and standard in OT libraries (POT, OTT-JAX).

    let mut g_log_u = Array2::<f64>::zeros((k, m));
    let mut g_log_v = Array2::<f64>::zeros((k, m));
    let mut g_log_atoms = Array2::<f64>::zeros((k, m));
    let mut g_weights = Array1::<f64>::zeros(k);

    // Precompute couplings P_k[i, j] = exp(log_kernel + log_u[k, i] + log_v[k, j] - log_a[j]).
    let mut p_couplings: Vec<Array2<f64>> = Vec::with_capacity(k);
    let mut q_couplings: Vec<Array2<f64>> = Vec::with_capacity(k);
    for ki in 0..k {
        let mut p_mat = Array2::<f64>::zeros((m, m));
        let mut q_mat = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                let logp =
                    state.log_kernel[[i, j]] + state.log_u[[ki, i]] + state.log_v[[ki, j]];
                let p_val = (logp - state.log_a[j]).exp();
                let q_val = (logp - state.log_atoms[[ki, i]]).exp();
                p_mat[[i, j]] = p_val;
                q_mat[[i, j]] = q_val;
            }
        }
        p_couplings.push(p_mat);
        q_couplings.push(q_mat);
    }

    // First propagate cotangent through Step 3:
    //   log_a[j] = sum_k w_k * LSE_i(log_kernel[i, j] + log_u[k, i])
    // Let s_k[j] = LSE_i(log_kernel[i, j] + log_u[k, i]).
    // ∂s_k[j]/∂log_u[k, i] = P_k[i, j]    (axis-0 softmax weights)
    // ∂log_a[j]/∂w_k = s_k[j]
    // ∂log_a[j]/∂log_u[k, i] = w_k * P_k[i, j]
    let mut s_per_k: Vec<Array1<f64>> = Vec::with_capacity(k);
    {
        let mut scratch = Array2::<f64>::zeros((m, m));
        for ki in 0..k {
            for i_prime in 0..m {
                let off = state.log_u[[ki, i_prime]];
                for j in 0..m {
                    scratch[[i_prime, j]] = state.log_kernel[[i_prime, j]] + off;
                }
            }
            s_per_k.push(logsumexp_axis0(scratch.view()));
        }
    }
    for ki in 0..k {
        // d g_log_u
        for i in 0..m {
            let mut acc = 0.0_f64;
            for j in 0..m {
                acc += g_log_a[j] * state.weights[ki] * p_couplings[ki][[i, j]];
            }
            g_log_u[[ki, i]] += acc;
        }
        // d g_weights (raw weights, then deconvolved through normalization)
        let mut acc_w = 0.0_f64;
        for j in 0..m {
            acc_w += g_log_a[j] * s_per_k[ki][j];
        }
        g_weights[ki] += acc_w;
    }

    // Adjoint Sinkhorn sweeps. Iterate the linearized fixed-point map
    // n_iter times to propagate cotangents back to (log_atoms, log_a-seed)
    // — but since log_a is the output and was already absorbed above,
    // we now want to propagate g_log_u through the Sinkhorn equations
    // to g_log_atoms and (recursively) back to g_log_a, which would
    // feed back into the loop. After convergence, the adjoint iteration
    // is itself a fixed point.
    //
    // Iterate the adjoint map:
    //   Step 2 backward: log_u[k, i] = log_atoms[k, i] - LSE_j(log_kernel[i, j] + log_v[k, j])
    //     g_log_atoms[k, i] += g_log_u[k, i]
    //     g_log_v[k, j] += -g_log_u[k, i] * Q_k[i, j]   summed over i
    //   Step 1 backward: log_v[k, j] = log_a[j] - LSE_i(log_kernel[i, j] + log_u[k, i])
    //     g_log_a_pre[j] += g_log_v[k, j]
    //     g_log_u[k, i] += -g_log_v[k, j] * P_k[i, j]   summed over j

    let mut g_log_a_in = Array1::<f64>::zeros(m);
    for _ in 0..n_iter {
        // Step 2 adjoint.
        for ki in 0..k {
            for i in 0..m {
                g_log_atoms[[ki, i]] += g_log_u[[ki, i]];
            }
            for i in 0..m {
                let gu = g_log_u[[ki, i]];
                if gu == 0.0 {
                    continue;
                }
                for j in 0..m {
                    g_log_v[[ki, j]] -= gu * q_couplings[ki][[i, j]];
                }
            }
            // Reset g_log_u for this k — it's been fully consumed.
            for i in 0..m {
                g_log_u[[ki, i]] = 0.0;
            }
        }

        // Step 1 adjoint.
        for ki in 0..k {
            for j in 0..m {
                g_log_a_in[j] += g_log_v[[ki, j]];
            }
            for j in 0..m {
                let gv = g_log_v[[ki, j]];
                if gv == 0.0 {
                    continue;
                }
                for i in 0..m {
                    g_log_u[[ki, i]] -= gv * p_couplings[ki][[i, j]];
                }
            }
            // Reset g_log_v.
            for j in 0..m {
                g_log_v[[ki, j]] = 0.0;
            }
        }

        // Feed the accumulated g_log_a_in back into the Step 3 backward
        // (this closes the loop and is what makes the adjoint
        // iteration a fixed point at convergence).
        for ki in 0..k {
            for i in 0..m {
                let mut acc = 0.0_f64;
                for j in 0..m {
                    acc += g_log_a_in[j] * state.weights[ki] * p_couplings[ki][[i, j]];
                }
                g_log_u[[ki, i]] += acc;
            }
            let mut acc_w = 0.0_f64;
            for j in 0..m {
                acc_w += g_log_a_in[j] * s_per_k[ki][j];
            }
            g_weights[ki] += acc_w;
        }
        for j in 0..m {
            g_log_a_in[j] = 0.0;
        }
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
    // ∂log_a depends on w_norm only — but g_weights above was computed
    // against w_norm[ki] directly (we used state.weights[ki] which is
    // already normalized). So convert: ∂w_norm[k]/∂w_raw[l] =
    // (k == l) / W - w_raw[k] / W^2 = (1/W) * ((k == l) - w_norm[k]).
    let mut d_weights = Array1::<f64>::zeros(k);
    let w_total: f64 = weights.iter().sum();
    if w_total > 0.0 {
        let mut sum_norm_g = 0.0_f64;
        for ki in 0..k {
            sum_norm_g += g_weights[ki] * state.weights[ki];
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
/// `(M, 3)` unit direction vectors. Validates approximate unit-norm
/// to within `1e-6`.
pub fn geodesic_sphere_cost(directions: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let (m, d) = directions.dim();
    if d != 3 {
        return Err(format!(
            "geodesic_sphere_cost requires direction vectors of dimension 3, got {d}"
        ));
    }
    for i in 0..m {
        let mut norm = 0.0_f64;
        for k in 0..3 {
            let v = directions[[i, k]];
            if !v.is_finite() {
                return Err(format!(
                    "geodesic_sphere_cost directions must be finite; got {v} at ({i}, {k})"
                ));
            }
            norm += v * v;
        }
        if (norm.sqrt() - 1.0).abs() > 1.0e-6 {
            return Err(format!(
                "geodesic_sphere_cost row {i} must be unit-norm; got |x| = {}",
                norm.sqrt()
            ));
        }
    }
    let mut out = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            let mut dot = 0.0_f64;
            for k in 0..3 {
                dot += directions[[i, k]] * directions[[j, k]];
            }
            let dot_clamped = dot.clamp(-1.0, 1.0);
            let theta = dot_clamped.acos();
            out[[i, j]] = theta * theta;
        }
    }
    Ok(out)
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

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
    }

    #[test]
    fn k_eq_2_mean_is_between() {
        let m = 32;
        let points: Array2<f64> =
            Array2::from_shape_fn((m, 1), |(i, _)| i as f64 / (m - 1) as f64);
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

        let mean_a: f64 = (0..m).map(|j| (j as f64 / (m - 1) as f64) * atom_a[j]).sum();
        let mean_b: f64 = (0..m).map(|j| (j as f64 / (m - 1) as f64) * atom_b[j]).sum();
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
    fn batch_kbig_runs_quick() {
        let m = 64;
        let k = 128;
        let atoms = Array2::<f64>::from_shape_fn((k, m), |(ki, j)| {
            let centre = (ki as f64) * (m as f64) / (k as f64);
            (-((j as f64 - centre).powi(2)) / 8.0).exp()
        });
        let weights = Array1::<f64>::from_elem(k, 1.0 / k as f64);
        let cost = circular_cost(m);
        let t0 = std::time::Instant::now();
        let bary =
            sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), 0.1, 20).unwrap();
        let dt = t0.elapsed();
        assert!(
            dt.as_secs_f64() < 5.0,
            "batch sinkhorn took too long: {dt:?}"
        );
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
        // Tiny problem, large eps, modest n_iter — finite-difference
        // gradient of a linear functional of the barycenter should
        // match the VJP within a few percent (the adjoint iteration
        // converges to the IFT solution).
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
        let r = Array1::<f64>::from_shape_fn(m, |j| (j as f64 - (m as f64 - 1.0) / 2.0));

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
        let b_plus = sinkhorn_barycenter(
            atoms_plus.view(),
            weights.view(),
            cost.view(),
            eps,
            n_iter,
        )
        .unwrap();
        let b_minus = sinkhorn_barycenter(
            atoms_minus.view(),
            weights.view(),
            cost.view(),
            eps,
            n_iter,
        )
        .unwrap();
        let mut fd = 0.0_f64;
        for i in 0..m {
            fd += r[i] * (b_plus[i] - b_minus[i]) / (2.0 * h);
        }
        let analytic = vjp.d_atoms[[ki, j]];
        // Adjoint iteration is exact at the fixed point only in the
        // limit; with n_iter=100 and eps=0.3 we expect agreement to a
        // few percent. Use a generous relative tolerance.
        let denom = analytic.abs().max(fd.abs()).max(1.0e-6);
        let rel = (analytic - fd).abs() / denom;
        assert!(
            rel < 0.05,
            "VJP/FD mismatch: analytic={analytic}, fd={fd}, rel={rel}"
        );
    }
}

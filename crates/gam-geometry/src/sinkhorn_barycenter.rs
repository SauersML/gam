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
//! ## Convergence
//!
//! The forward pass returns a barycenter only from a certified fixed point.
//! Each sweep of the projections is a block-coordinate ascent step on the
//! entropic dual `D(f, g) = sum_k w_k [<f_k, p_k> - <e^{f_k}, K e^{g_k}>]`
//! under `sum_k w_k g_k = 0`, so there is no iteration count to choose: the
//! loop stops on one of two tests, both measured against the rounding the
//! sweep itself commits.
//!
//! * **Certified.** After the row projection every plan has its atom as row
//!   marginal exactly, and its column marginal differs from the barycenter by
//!   the violation `max_k ||col_k - a||_1`. When that violation is inside its
//!   rounding band, the iterate is the fixed point to working precision and its
//!   barycenter is returned.
//! * **Stalled.** In exact arithmetic the dual rises every sweep. When a sweep's
//!   dual gain is inside the two dual values' rounding bands and the violation
//!   did not fall either, the arithmetic can no longer show progress toward the
//!   fixed point, and the solve is refused with `Err` — the unconverged iterate
//!   is never returned.
//!
//! ## Stability guarantees
//!
//! * `eps > 0` is required, with `max(cost)/eps` below `1/u` (`u` the unit
//!   roundoff), where the largest Gibbs exponent still resolves a nat; anything
//!   else is rejected with `Err`. The bound is on the ratio because the problem
//!   is: `(c·cost, c·eps)` is the same barycenter for every `c > 0`.
//! * Input atom rows with truly-zero mass on a support point are
//!   handled via a large-negative sentinel (`LOG_ZERO_SENTINEL`)
//!   instead of `-inf`, so additions of `+inf` (from the kernel) and
//!   `-inf` (from the log) do not produce `NaN`. The mathematical
//!   contract is unchanged: a support point with zero atom mass and
//!   finite cost remains a valid kernel argument; only the gradient
//!   flow through that point is gracefully damped.
//! * Up to `(K=128, M=256)`: peak working memory is `K * M * 8` for
//!   each of the log-dual potentials plus `M * M * 8` for the log-kernel,
//!   which is `<= 1 MiB` — no OOM under standard test machines. No
//!   `K * M * M` plan is ever materialized; plan entries are formed on the fly.
//!
//! ## Differentiability
//!
//! Because the forward pass returns the fixed point itself, its
//! vector-Jacobian product is the implicit-function-theorem adjoint of the
//! fixed-point equations, computed by [`sinkhorn_barycenter_vjp`] with one
//! conjugate-gradient solve (see its documentation). No sweep history is
//! recorded or replayed.

use gam_linalg::roundoff::accumulation_growth;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Largest non-`-inf` value used in place of `log(0)` for input atom
/// entries with truly-zero mass. Chosen so that adding it to a kernel
/// row (which has values in roughly `[-eps^{-1} * cost_max, 0]`)
/// still saturates near `LOG_ZERO_SENTINEL` after a `logsumexp` — i.e.
/// the corresponding row's contribution vanishes from the barycenter,
/// which is exactly the desired mathematical behaviour.
pub(crate) const LOG_ZERO_SENTINEL: f64 = -1.0e300;

const LOG_ZERO_SATURATION_THRESHOLD: f64 = LOG_ZERO_SENTINEL * 0.5;

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
    if !(eps.is_finite() && eps > 0.0) {
        return Err(format!("eps must be finite and positive, got {eps}"));
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
    let mut cost_max = 0.0_f64;
    for ((i, j), value) in cost.indexed_iter() {
        if !value.is_finite() || *value < 0.0 {
            return Err(format!(
                "cost must be finite and non-negative; got {value} at ({i}, {j})"
            ));
        }
        cost_max = cost_max.max(*value);
    }
    // The Gibbs exponents are `−cost/eps`. Once the largest one's unit roundoff
    // reaches a nat, `exp(−cost/eps)` carries no significant digit, and neither
    // does anything the log-domain updates build from it.
    let largest_exponent = cost_max / eps;
    if !(gam_linalg::roundoff::UNIT_ROUNDOFF * largest_exponent < 1.0) {
        return Err(format!(
            "eps {eps} is below the resolution of the Gibbs exponents: the largest, \
             max(cost)/eps = {largest_exponent:e}, rounds by a nat or more"
        ));
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

/// A validated barycenter problem in log form.
struct BarycenterProblem {
    /// `(K, M)` — the atoms normalized to the simplex, `p_k`.
    atoms_norm: Array2<f64>,
    /// `(K, M)` — `log p_k`, with [`LOG_ZERO_SENTINEL`] for zero mass.
    log_atoms: Array2<f64>,
    /// `(K,)` — the mixing weights normalized to the simplex, `w_k`.
    weights_norm: Vec<f64>,
    /// `(M, M)` — the log Gibbs kernel `-cost / eps`.
    log_kernel: Array2<f64>,
}

impl BarycenterProblem {
    fn new(
        atoms: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
        cost: ArrayView2<'_, f64>,
        eps: f64,
    ) -> Result<Self, String> {
        validate_inputs(atoms, weights, cost, eps)?;
        let atoms_norm = normalize_atoms(atoms)?;
        let weights_norm = normalize_weights(weights);
        let (k, m) = atoms_norm.dim();
        let log_kernel = cost.mapv(|c| -c / eps);
        let mut log_atoms = Array2::<f64>::zeros((k, m));
        for ki in 0..k {
            log_atoms
                .row_mut(ki)
                .assign(&safe_log_simplex(atoms_norm.row(ki)));
        }
        Ok(Self {
            atoms_norm,
            log_atoms,
            weights_norm,
            log_kernel,
        })
    }
}

/// The certified fixed point of the projections.
struct BarycenterFixedPoint {
    /// `(K, M)` — the row potentials `log u_k` after the last row projection.
    log_u: Array2<f64>,
    /// `(K, M)` — `lse0_k[j] = logsumexp_i(log_kernel[i, j] + log_u[k, i])`.
    lse0: Array2<f64>,
    /// `(M,)` — `log a = sum_k w_k lse0_k`, the log barycenter.
    log_a: Array1<f64>,
    /// Absolute rounding bound on the log potentials at the fixed point.
    potential_rounding: f64,
}

/// Largest magnitude over the entries of `values` that are not
/// sentinel-saturated. The sentinel is a stand-in for `-inf`, not a magnitude
/// the arithmetic carries.
fn live_magnitude(values: &Array2<f64>) -> f64 {
    values
        .iter()
        .filter(|v| **v > LOG_ZERO_SATURATION_THRESHOLD)
        .fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

/// Run the projections to their certified fixed point (see the module
/// documentation for the two stopping tests).
///
/// Rounding bands. Every log potential entry is built through three `M`-term
/// log-sum-exps — `log_v` from the previous `lse0`, `log_u` from `log_v`, and
/// the new `lse0` from `log_u` — each costing `M + 3` roundings (the `M`-term
/// accumulation, the max shift, the `exp` and the `ln`), plus the `K`-term
/// weighted sum forming `log a`. Each rounding is relative to the magnitudes
/// the chain carries, bounded by `1 + max|log_kernel| + max|log_u| + max|log_v|`
/// over live entries, so every exponent is off by at most
/// `δ = γ_{3(M+3)+K} · scale`. A column-marginal entry `exp(x)` is then off by
/// `δ` relative, and the `M`-term L1 sum adds `γ_{M+1}`, which bounds the
/// violation's rounding by `(δ + γ_{M+1}) · Σ_j (col_kj + a_j)`. The dual
/// `Σ_k w_k Σ_i p_ki log_u_ki` inherits `δ` from its potentials (the weights
/// `w_k p_ki` sum to one) and `γ_{M+K+2}` relative to `Σ w p |log_u|` from its
/// two products and its `M`-term inner and `K`-term outer sums.
fn solve_fixed_point(problem: &BarycenterProblem) -> Result<BarycenterFixedPoint, String> {
    let (k, m) = problem.atoms_norm.dim();
    let w = &problem.weights_norm;
    let log_kernel = problem.log_kernel.view();
    let kernel_magnitude = problem
        .log_kernel
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let potential_growth = accumulation_growth(3 * (m + 3) + k);
    let marginal_growth = accumulation_growth(m + 1);
    let dual_growth = accumulation_growth(m + k + 2);

    // Start from `log u = 0` with `log a` in the gauge the sweeps keep,
    // `log a = Σ_k w_k lse0_k`, so the constraint `Σ_k w_k log v_k = 0` — and
    // with it the dual's closed form — holds from the first sweep.
    let mut log_u = Array2::<f64>::zeros((k, m));
    let mut lse0 = Array2::<f64>::zeros((k, m));
    let mut log_a = Array1::<f64>::zeros(m);
    for ki in 0..k {
        let row = logsumexp_kernel_plus_offset_axis0(log_kernel, log_u.row(ki));
        log_a.scaled_add(w[ki], &row);
        lse0.row_mut(ki).assign(&row);
    }
    let mut log_v = Array2::<f64>::zeros((k, m));

    // (dual, dual band, violation) of the previous sweep.
    let mut previous: Option<(f64, f64, f64)> = None;
    let mut sweep = 0usize;
    loop {
        // Column projection: log_v_k = log a − lse0_k.
        for ki in 0..k {
            for j in 0..m {
                log_v[[ki, j]] = log_a[j] - lse0[[ki, j]];
            }
        }
        // Row projection: log_u_k = log p_k − logsumexp_j(log_kernel[i, j] + log_v_k[j]).
        for ki in 0..k {
            let lse1 = logsumexp_kernel_plus_offset_axis1(log_kernel, log_v.row(ki));
            for i in 0..m {
                log_u[[ki, i]] = problem.log_atoms[[ki, i]] - lse1[i];
            }
        }
        // Barycenter: log a' = Σ_k w_k lse0_k at the new row potentials.
        let mut next_log_a = Array1::<f64>::zeros(m);
        for ki in 0..k {
            let row = logsumexp_kernel_plus_offset_axis0(log_kernel, log_u.row(ki));
            next_log_a.scaled_add(w[ki], &row);
            lse0.row_mut(ki).assign(&row);
        }

        let scale = 1.0 + kernel_magnitude + live_magnitude(&log_u) + live_magnitude(&log_v);
        let potential_rounding = potential_growth * scale;

        // Column-marginal violation: plan k's column marginal is
        // exp(log_v_k + lse0_k), the barycenter is exp(log a').
        let mut violation = 0.0_f64;
        let mut violation_band = 0.0_f64;
        for ki in 0..k {
            let mut gap = 0.0_f64;
            let mut mass = 0.0_f64;
            for j in 0..m {
                let col = (log_v[[ki, j]] + lse0[[ki, j]]).exp();
                let bary = next_log_a[j].exp();
                gap += (col - bary).abs();
                mass += col + bary;
            }
            violation = violation.max(gap);
            violation_band = violation_band.max((potential_rounding + marginal_growth) * mass);
        }
        if violation <= violation_band {
            return Ok(BarycenterFixedPoint {
                log_u,
                lse0,
                log_a: next_log_a,
                potential_rounding,
            });
        }

        // Dual after the row projection: Σ_k w_k <log u_k, p_k> − 1.
        let mut dual = 0.0_f64;
        let mut dual_magnitude = 0.0_f64;
        for ki in 0..k {
            let mut inner = 0.0_f64;
            let mut inner_magnitude = 0.0_f64;
            for i in 0..m {
                let p = problem.atoms_norm[[ki, i]];
                if p > 0.0 {
                    inner += p * log_u[[ki, i]];
                    inner_magnitude += p * log_u[[ki, i]].abs();
                }
            }
            dual += w[ki] * inner;
            dual_magnitude += w[ki] * inner_magnitude;
        }
        let dual_band = potential_rounding + dual_growth * dual_magnitude;
        if let Some((previous_dual, previous_band, previous_violation)) = previous {
            let gain = dual - previous_dual;
            if gain <= dual_band + previous_band && violation >= previous_violation {
                return Err(format!(
                    "sinkhorn barycenter stalled at sweep {sweep}: the dual gain {gain:e} is \
                     inside its rounding band {:e} and the column-marginal violation \
                     {violation:e} did not fall (previous {previous_violation:e}), so the \
                     arithmetic cannot certify the fixed point (violation band \
                     {violation_band:e}); eps is too small relative to the cost scale",
                    dual_band + previous_band
                ));
            }
        }
        previous = Some((dual, dual_band, violation));
        log_a = next_log_a;
        sweep += 1;
    }
}

/// Log-domain Sinkhorn barycenter forward pass: returns the barycenter of the
/// certified fixed point as a simplex vector of length `M`, or `Err` when the
/// projections stall before the fixed point can be certified.
pub fn sinkhorn_barycenter(
    atoms: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    cost: ArrayView2<'_, f64>,
    eps: f64,
) -> Result<Array1<f64>, String> {
    let problem = BarycenterProblem::new(atoms, weights, cost, eps)?;
    let fixed_point = solve_fixed_point(&problem)?;
    if log_vector_is_sentinel_saturated(fixed_point.log_a.view()) {
        return Err(
            "sinkhorn barycenter degenerated: all log_a saturated to sentinel -- try larger eps or check cost matrix"
                .to_string(),
        );
    }
    softmax_1d(fixed_point.log_a.view())
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

/// The fixed point's adjoint operator, applied without materializing a plan.
///
/// With `S_k[j, i] = exp(log_kernel[i, j] + log_u[k, i] − lse0[k, j])` (a
/// softmax over `i`) and the plans `π_k[i, j] = a_j S_k[j, i]`, the operator is
/// `(Bν)_ki = w_k [p_ki ν_ki − Σ_j π_kij (σ_kj − σ̄_j)]` with
/// `σ_kj = Σ_i S_k[j, i] ν_ki` and `σ̄ = Σ_l w_l σ_l`, on the active
/// coordinates `{w_k > 0, p_ki > 0}`.
struct FixedPointAdjoint<'a> {
    problem: &'a BarycenterProblem,
    fixed_point: &'a BarycenterFixedPoint,
    /// `(M,)` — `a = exp(log a)`.
    barycenter: Array1<f64>,
    /// The active coordinates as `(k, i)`, in PCG vector order.
    active: Vec<(usize, usize)>,
}

impl FixedPointAdjoint<'_> {
    fn softmax_weight(&self, ki: usize, i: usize, j: usize) -> f64 {
        (self.problem.log_kernel[[i, j]] + self.fixed_point.log_u[[ki, i]]
            - self.fixed_point.lse0[[ki, j]])
            .exp()
    }

    /// Scatter an active-coordinate vector into `(K, M)`, zero elsewhere.
    fn scatter(&self, nu: &Array1<f64>) -> Array2<f64> {
        let (k, m) = self.problem.atoms_norm.dim();
        let mut full = Array2::<f64>::zeros((k, m));
        for (&(ki, i), &value) in self.active.iter().zip(nu.iter()) {
            full[[ki, i]] = value;
        }
        full
    }

    /// `(σ, σ̄)` for a full `(K, M)` vector `ν`.
    fn column_averages(&self, nu: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
        let (k, m) = self.problem.atoms_norm.dim();
        let mut sigma = Array2::<f64>::zeros((k, m));
        let mut sigma_bar = Array1::<f64>::zeros(m);
        for ki in 0..k {
            for j in 0..m {
                let mut acc = 0.0_f64;
                for i in 0..m {
                    let value = nu[[ki, i]];
                    if value != 0.0 {
                        acc += self.softmax_weight(ki, i, j) * value;
                    }
                }
                sigma[[ki, j]] = acc;
            }
            sigma_bar.scaled_add(self.problem.weights_norm[ki], &sigma.row(ki));
        }
        (sigma, sigma_bar)
    }

    /// Orthogonal projection off the gauge null space `{Σ_k c_k 1_k : Σ_k w_k c_k = 0}`
    /// (`1_k` the indicator of atom `k`'s active coordinates, `n_k` of them).
    ///
    /// The span of the `1_k` splits into the null space and the single range
    /// direction `q = Σ_k (w_k / n_k) 1_k`, so the projection subtracts each
    /// block mean and adds back the component along `q`.
    fn project_off_gauge(&self, x: &mut Array1<f64>) {
        let k = self.problem.weights_norm.len();
        let mut block_sum = vec![0.0_f64; k];
        let mut block_len = vec![0usize; k];
        for (&(ki, _), &value) in self.active.iter().zip(x.iter()) {
            block_sum[ki] += value;
            block_len[ki] += 1;
        }
        let mut q_dot_x = 0.0_f64;
        let mut q_norm_sq = 0.0_f64;
        for ki in 0..k {
            if block_len[ki] > 0 {
                let w = self.problem.weights_norm[ki];
                let n = block_len[ki] as f64;
                q_dot_x += w / n * block_sum[ki];
                q_norm_sq += w * w / n;
            }
        }
        let along_q = q_dot_x / q_norm_sq;
        for (slot, &(ki, _)) in x.iter_mut().zip(self.active.iter()) {
            let n = block_len[ki] as f64;
            *slot += along_q * self.problem.weights_norm[ki] / n - block_sum[ki] / n;
        }
    }

    /// `P B`, with `P` the projection off the gauge. `B` maps into the gauge's
    /// orthogonal complement exactly, so this is `B`; the projection only keeps
    /// the rounding of each application from accumulating a gauge component in
    /// the CG residual, which that residual can never reduce.
    fn apply(&self, nu: &Array1<f64>, out: &mut Array1<f64>) {
        let m = self.barycenter.len();
        let full = self.scatter(nu);
        let (sigma, sigma_bar) = self.column_averages(&full);
        for (slot, (&(ki, i), &value)) in out.iter_mut().zip(self.active.iter().zip(nu.iter())) {
            let mut transport = 0.0_f64;
            for j in 0..m {
                let plan = self.barycenter[j] * self.softmax_weight(ki, i, j);
                transport += plan * (sigma[[ki, j]] - sigma_bar[j]);
            }
            let w = self.problem.weights_norm[ki];
            *slot = w * (self.problem.atoms_norm[[ki, i]] * value - transport);
        }
        self.project_off_gauge(out);
    }

    /// `diag B_ki = w_k p_ki − w_k (1 − w_k) Σ_j π_kij² / a_j`, which is at
    /// least `w_k² p_ki > 0` on the active set because `Σ_j π_kij S_k[j, i] ≤ p_ki`.
    fn diagonal(&self) -> Array1<f64> {
        let m = self.barycenter.len();
        self.active
            .iter()
            .map(|&(ki, i)| {
                let w = self.problem.weights_norm[ki];
                let mut curvature = 0.0_f64;
                for j in 0..m {
                    let s = self.softmax_weight(ki, i, j);
                    curvature += self.barycenter[j] * s * s;
                }
                w * self.problem.atoms_norm[[ki, i]] - w * (1.0 - w) * curvature
            })
            .collect()
    }
}

/// Vector-Jacobian product of the Sinkhorn-barycenter fixed point.
///
/// Given a cotangent `cotangent` of shape `(M,)` (the upstream gradient of a
/// scalar loss w.r.t. the output barycenter), this returns
/// `(dL/d_atoms, dL/d_weights)` of the barycenter [`sinkhorn_barycenter`]
/// returns, by the implicit function theorem on its fixed-point equations.
///
/// Write `f_k = log u_k` for the row potentials. At the fixed point
/// `f_k = log p_k − logsumexp_j(log_kernel[:, j] + log a_j − lse0_kj)`, with
/// `log a = Σ_k w_k lse0_k`. Scaling its Jacobian in `f` by `W = diag(w_k p_k)`
/// gives the operator `B` of [`FixedPointAdjoint`], which is symmetric positive
/// semidefinite by construction (`⟨ν, Bν⟩ ≥ 0` is Jensen's inequality on each
/// softmax `S_k`). Its null space is the gauge `ν_k = c_k 1` with
/// `Σ_k w_k c_k = 0`, which leaves the plans unchanged; the right-hand side is
/// orthogonal to it, and every gradient below is invariant along it.
///
/// With `g = y ⊙ (c − ⟨y, c⟩)` the cotangent pulled through the output
/// normalization `y = softmax(log a)`, solve `Bν = b`,
/// `b_ki = w_k Σ_j S_k[j, i] g_j`, by Jacobi-preconditioned conjugate gradients
/// from `ν = 0`. The system is consistent, but rounding in `b` and in each
/// application of `B` leaves gauge components in the residual that CG can
/// never reduce, and their accumulation drives the search directions into the
/// null space; so `b` and every application are projected off the gauge
/// exactly (`P B = B` in exact arithmetic), and the solve runs on the range of
/// `B`. Then
///
/// * `dL/d log p_k = w_k p_k ν_k`, and
/// * `dL/d w_l = ⟨g − a ⊙ σ̄, lse0_l⟩` with `σ̄` the operator's column average.
///
/// The CG solve is accepted only when converged: its residual target is the
/// operator's own rounding — the plan entries carry the potentials' rounding
/// `δ` relative, and one application adds two `M`-term sums, one `K`-term sum
/// and four products, `γ_{2M+K+4}` — and its iteration cap is the active
/// dimension, where exact-arithmetic CG terminates. A solve that does not reach
/// the target is refused with `Err`.
pub fn sinkhorn_barycenter_vjp(
    atoms: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    cost: ArrayView2<'_, f64>,
    eps: f64,
    cotangent: ArrayView1<'_, f64>,
) -> Result<SinkhornVjp, String> {
    let problem = BarycenterProblem::new(atoms, weights, cost, eps)?;
    let (k, m) = problem.atoms_norm.dim();
    if cotangent.len() != m {
        return Err(format!(
            "cotangent length {} does not match barycenter size {}",
            cotangent.len(),
            m
        ));
    }
    if let Some(bad) = cotangent.iter().find(|v| !v.is_finite()) {
        return Err(format!("cotangent must be finite; got {bad}"));
    }
    let fixed_point = solve_fixed_point(&problem)?;
    if log_vector_is_sentinel_saturated(fixed_point.log_a.view()) {
        return Err(
            "sinkhorn barycenter degenerated: all log_a saturated to sentinel -- try larger eps or check cost matrix"
                .to_string(),
        );
    }

    // Pull the cotangent through y = softmax(log a):
    // d(softmax(z))/dz = diag(y) − y yᵀ, so g = y ⊙ (c − ⟨c, y⟩).
    let bary = softmax_1d(fixed_point.log_a.view())?;
    let weighted = cotangent.dot(&bary);
    let g_log_a = Array1::from_shape_fn(m, |j| bary[j] * (cotangent[j] - weighted));

    let mut active = Vec::new();
    for ki in 0..k {
        if problem.weights_norm[ki] > 0.0 {
            for i in 0..m {
                if problem.atoms_norm[[ki, i]] > 0.0 {
                    active.push((ki, i));
                }
            }
        }
    }
    let adjoint = FixedPointAdjoint {
        problem: &problem,
        fixed_point: &fixed_point,
        barycenter: fixed_point.log_a.mapv(f64::exp),
        active,
    };
    let mut rhs: Array1<f64> = adjoint
        .active
        .iter()
        .map(|&(ki, i)| {
            let mut acc = 0.0_f64;
            for j in 0..m {
                acc += adjoint.softmax_weight(ki, i, j) * g_log_a[j];
            }
            problem.weights_norm[ki] * acc
        })
        .collect();
    adjoint.project_off_gauge(&mut rhs);
    let rel_tol = fixed_point.potential_rounding + accumulation_growth(2 * m + k + 4);
    let dimension = adjoint.active.len();
    let (nu_active, _) = gam_linalg::utils::solve_spd_pcg_with_info_into(
        |v, out| adjoint.apply(v, out),
        &rhs,
        &adjoint.diagonal(),
        rel_tol,
        dimension,
    )
    .ok_or_else(|| {
        format!(
            "sinkhorn barycenter adjoint: conjugate gradients did not reach the operator's \
             rounding band {rel_tol:e} within the {dimension}-dimensional Krylov bound"
        )
    })?;
    let nu = adjoint.scatter(&nu_active);
    let (_, sigma_bar) = adjoint.column_averages(&nu);

    let mut g_log_atoms = Array2::<f64>::zeros((k, m));
    for &(ki, i) in &adjoint.active {
        g_log_atoms[[ki, i]] = problem.weights_norm[ki] * problem.atoms_norm[[ki, i]] * nu[[ki, i]];
    }
    let transported = Array1::from_shape_fn(m, |j| g_log_a[j] - adjoint.barycenter[j] * sigma_bar[j]);
    let g_weights = Array1::from_shape_fn(k, |ki| transported.dot(&fixed_point.lse0.row(ki)));

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
            sum_norm_g += g_weights[ki] * problem.weights_norm[ki];
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
/// Each row is a direction: it is normalized to exact unit length before any
/// cosine is formed, and a zero row, which has no direction, is refused. The
/// cost is the true squared great-circle distance
///
/// `C_ij = arccos( <x_i/|x_i|, x_j/|x_j|> )^2`
///
/// of the projected directions. This guarantees a symmetric matrix
/// with an exactly-zero diagonal: without normalization a row with
/// `|x| != 1` would yield `arccos(<x,x>)^2 > 0` on the diagonal, contradicting
/// `d(x, x) = 0`.
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
        if !(norm > 0.0) {
            return Err(format!(
                "geodesic_sphere_cost row {i} has zero norm and so no direction"
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

    fn two_bumps_on_sixteen() -> Array2<f64> {
        Array2::<f64>::from_shape_fn((2, 16), |(k, j)| {
            let centre = if k == 0 { 3.0 } else { 11.0 };
            (-((j as f64 - centre).powi(2)) / 4.0).exp()
        })
    }

    fn three_bumps_on_six() -> Array2<f64> {
        Array2::<f64>::from_shape_fn((3, 6), |(ki, j)| {
            let centre = [0.8, 2.6, 4.3][ki];
            (-((j as f64 - centre).powi(2)) / 0.7).exp()
        })
    }

    /// Second-order finite difference of `r · barycenter` along a direction:
    /// central, or forward (`(−3f(0) + 4f(h) − f(2h)) / 2h`) when the base
    /// point sits on the boundary of the non-negative inputs.
    fn directional_fd(
        atoms: &Array2<f64>,
        weights: &Array1<f64>,
        cost: &Array2<f64>,
        eps: f64,
        r: &Array1<f64>,
        d_atoms: &Array2<f64>,
        d_weights: &Array1<f64>,
        forward: bool,
    ) -> f64 {
        let h = 1.0e-5;
        let at = |step: f64| {
            let bary = sinkhorn_barycenter(
                (atoms + &(d_atoms * step)).view(),
                (weights + &(d_weights * step)).view(),
                cost.view(),
                eps,
            )
            .unwrap();
            r.dot(&bary)
        };
        if forward {
            (-3.0 * at(0.0) + 4.0 * at(h) - at(2.0 * h)) / (2.0 * h)
        } else {
            (at(h) - at(-h)) / (2.0 * h)
        }
    }

    fn assert_vjp_matches_fd(
        atoms: &Array2<f64>,
        weights: &Array1<f64>,
        cost: &Array2<f64>,
        eps: f64,
        tol: f64,
    ) {
        let (k, m) = atoms.dim();
        let r = Array1::<f64>::from_shape_fn(m, |j| j as f64 - (m as f64 - 1.0) / 2.0);
        let vjp =
            sinkhorn_barycenter_vjp(atoms.view(), weights.view(), cost.view(), eps, r.view())
                .unwrap();
        // Each entry is compared at the scale of the whole gradient (its
        // infinity norm, in the coordinates the finite differences step), so
        // entries far below it are held to the gradient's accuracy rather than
        // to the finite differences' own rounding floor.
        let scaled_atoms = &vjp.d_atoms * atoms;
        let scale = scaled_atoms
            .iter()
            .chain(vjp.d_weights.iter())
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let check = |what: String, analytic: f64, fd: f64| {
            let rel = (analytic - fd).abs() / scale;
            assert!(
                rel < tol,
                "{what} VJP/FD mismatch: analytic={analytic}, fd={fd}, rel={rel}"
            );
        };
        for ki in 0..k {
            let mut e_w = Array1::<f64>::zeros(k);
            e_w[ki] = 1.0;
            let on_boundary = weights[ki] == 0.0;
            let fd = directional_fd(
                atoms,
                weights,
                cost,
                eps,
                &r,
                &Array2::zeros((k, m)),
                &e_w,
                on_boundary,
            );
            check(format!("weight {ki}"), vjp.d_weights[ki], fd);
            for j in 0..m {
                if atoms[[ki, j]] == 0.0 {
                    continue;
                }
                // A relative step, so small entries stay positive: this checks
                // the derivative in `log atoms[k, j]`.
                let mut e_a = Array2::<f64>::zeros((k, m));
                e_a[[ki, j]] = atoms[[ki, j]];
                let fd =
                    directional_fd(atoms, weights, cost, eps, &r, &e_a, &Array1::zeros(k), false);
                check(format!("atom ({ki},{j})"), scaled_atoms[[ki, j]], fd);
            }
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
        let eps = 0.05;
        let bary = sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), eps).unwrap();
        // With one atom the fixed point has `v = 1` and `u = p / (K 1)`, so it is
        // the atom smoothed once by the row-normalized kernel,
        // `a_j = Σ_i K_ij p_i / Σ_l K_il` — the atom up to the kernel's blur.
        let kernel = cost.mapv(|c| (-c / eps).exp());
        let expected = Array1::from_shape_fn(m, |j| {
            (0..m).map(|i| kernel[[i, j]] * atom[i] / kernel.row(i).sum()).sum::<f64>()
        });
        approx_simplex_eq(&bary, &expected, 1.0e-12);
        approx_simplex_eq(&bary, &atom, 1.0e-9);
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
        let bary = sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), 0.005).unwrap();

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
        let bary = sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), 0.5).unwrap();
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
    fn small_eps_certifies_a_finite_simplex() {
        // eps is a hundredth of the unit step cost: the kernel is sharply
        // concentrated and the projections contract slowly, yet the loop runs
        // to the certified fixed point rather than a fixed sweep count.
        let atoms = two_bumps_on_sixteen();
        let weights = array![0.5, 0.5];
        let cost = circular_cost(16);
        let bary = sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), 1.0e-2).unwrap();
        for v in bary.iter() {
            assert!(v.is_finite(), "barycenter entry {v} is not finite");
            assert!(*v >= 0.0, "barycenter entry {v} is negative");
        }
        let s: f64 = bary.iter().sum();
        assert!((s - 1.0).abs() < 1.0e-12, "barycenter sum {s} != 1");
    }

    #[test]
    fn certified_barycenter_is_the_fixed_point() {
        // One more projection sweep from the returned barycenter moves no
        // plan's column marginal off it: each plan transports its atom onto the
        // barycenter.
        let atoms = three_bumps_on_six();
        let weights = array![0.25, 0.35, 0.40];
        let cost = circular_cost(6);
        let eps = 0.3;
        let problem =
            BarycenterProblem::new(atoms.view(), weights.view(), cost.view(), eps).unwrap();
        let fixed_point = solve_fixed_point(&problem).unwrap();
        let bary = fixed_point.log_a.mapv(f64::exp);
        for ki in 0..3 {
            let log_v = &fixed_point.log_a - &fixed_point.lse0.row(ki);
            let lse1 = logsumexp_kernel_plus_offset_axis1(problem.log_kernel.view(), log_v.view());
            let log_u = &problem.log_atoms.row(ki) - &lse1;
            let col = logsumexp_kernel_plus_offset_axis0(problem.log_kernel.view(), log_u.view());
            let marginal = (&col + &log_v).mapv(f64::exp);
            let gap: f64 = (&marginal - &bary).iter().map(|d| d.abs()).sum();
            assert!(gap < 1.0e-12, "plan {ki} column marginal is {gap:e} off the barycenter");
        }
    }

    #[test]
    fn rejects_small_eps() {
        let m = 4;
        let atoms = Array2::<f64>::from_elem((2, m), 1.0 / m as f64);
        let weights = array![0.5, 0.5];
        let cost = circular_cost(m);
        // max(cost) = 4, so the largest Gibbs exponent 4/eps rounds by a nat at eps = 4u.
        let unit_roundoff = gam_linalg::roundoff::UNIT_ROUNDOFF;
        for eps in [0.0, -1.0, f64::NAN, f64::INFINITY, 4.0 * unit_roundoff] {
            let err = sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), eps);
            assert!(err.is_err(), "eps {eps:e} must be refused");
        }
        // Identical uniform atoms are their own barycenter at any eps, so the
        // first sweep certifies even where every exponent only resolves half a nat.
        let resolved =
            sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), 8.0 * unit_roundoff)
                .unwrap();
        approx_simplex_eq(&resolved, &atoms.row(0).to_owned(), 1.0e-12);
    }

    #[test]
    fn same_barycenter_at_any_cost_scale() {
        // `(c·cost, c·eps)` is one problem, so a cost scale that puts `eps` far
        // below any absolute floor gives the unit-scale answer.
        let atoms = two_bumps_on_sixteen();
        let weights = array![0.5, 0.5];
        let cost = circular_cost(16);
        let eps = 0.1;
        let scale = 1.0e-12;
        let unit = sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), eps).unwrap();
        let scaled_cost = cost.mapv(|c| c * scale);
        let scaled =
            sinkhorn_barycenter(atoms.view(), weights.view(), scaled_cost.view(), eps * scale)
                .unwrap();
        approx_simplex_eq(&scaled, &unit, 1.0e-10);
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
        let bary = sinkhorn_barycenter(atoms.view(), weights.view(), cost.view(), 4.0).unwrap();
        let s: f64 = bary.iter().sum();
        assert!((s - 1.0).abs() < 1.0e-12);
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
        let m = 6;
        let atoms = Array2::<f64>::from_shape_fn((2, m), |(ki, j)| {
            let centre = if ki == 0 { 1.5 } else { 4.0 };
            (-((j as f64 - centre).powi(2)) / 2.0).exp()
        });
        assert_vjp_matches_fd(&atoms, &array![0.5, 0.5], &circular_cost(m), 0.3, 1.0e-5);
    }

    #[test]
    fn vjp_matches_finite_differences_at_small_eps() {
        // At eps a hundredth of the unit step the projections need hundreds of
        // sweeps; the adjoint of the certified fixed point still matches the
        // finite differences of the forward pass.
        let atoms = three_bumps_on_six();
        let weights = array![0.25, 0.35, 0.40];
        assert_vjp_matches_fd(&atoms, &weights, &circular_cost(6), 0.01, 1.0e-5);
    }

    #[test]
    fn vjp_matches_finite_differences_with_zero_mass_and_zero_weight() {
        // A support point without atom mass and an atom without weight sit
        // outside the adjoint's active set; the gradients elsewhere — including
        // the zero-weight atom's weight gradient — still match.
        let mut atoms = three_bumps_on_six();
        atoms[[0, 4]] = 0.0;
        let weights = array![0.6, 0.4, 0.0];
        let cost = circular_cost(6);
        assert_vjp_matches_fd(&atoms, &weights, &cost, 0.3, 1.0e-5);
        let r = Array1::<f64>::from_shape_fn(6, |j| j as f64);
        let vjp =
            sinkhorn_barycenter_vjp(atoms.view(), weights.view(), cost.view(), 0.3, r.view())
                .unwrap();
        for j in 0..6 {
            assert_eq!(vjp.d_atoms[[2, j]], 0.0, "an unweighted atom cannot move the barycenter");
        }
    }

    #[test]
    fn vjp_of_constant_cotangent_is_zero() {
        // The barycenter stays on the simplex, so a constant cotangent sees no
        // change in any direction.
        let atoms = three_bumps_on_six();
        let weights = array![0.25, 0.35, 0.40];
        let ones = Array1::<f64>::ones(6);
        let vjp = sinkhorn_barycenter_vjp(
            atoms.view(),
            weights.view(),
            circular_cost(6).view(),
            0.3,
            ones.view(),
        )
        .unwrap();
        assert!(vjp.d_atoms.iter().all(|v| v.abs() < 1.0e-12));
        assert!(vjp.d_weights.iter().all(|v| v.abs() < 1.0e-12));
    }
}

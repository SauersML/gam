//! Exact O(n) state-space polynomial smoothing spline ("the scan").
//!
//! The order-`m` intrinsic Gaussian prior whose penalized posterior mean is the
//! degree-`(2m−1)` smoothing spline (penalty `λ∫(f^{(m)})²`) is a Markov process
//! in the state `α(x) = (f, f′, …, f^{(m−1)})`: an `m`-fold integrated Wiener
//! process. The Kalman filter + RTS smoother over the x-sorted observations
//! therefore computes the EXACT smoothing-spline posterior — mean, derivatives,
//! pointwise variance — and the diffuse innovations decomposition computes the
//! EXACT restricted (REML) likelihood, all in O(n) work per smoothing-parameter
//! trial instead of the dense O(n·k²) design/Gram + O(k³) solve per trial
//! (Wahba 1978; Kohn & Ansley 1987; Durbin & Koopman exact diffuse init).
//!
//! Supported orders are `m ∈ {1, 2, 3}` (`MAX_ORDER`): `m = 1` is the
//! random-walk / linear smoother (penalty `λ∫f′²`), `m = 2` the cubic smoother
//! (`λ∫f″²`), `m = 3` the quintic smoother (`λ∫(f‴)²`, natural spline degree
//! `2m−1 = 5`). The diffuse prior carries `m` improper dimensions consumed by
//! the first `m` distinct abscissae, leaving `m − 1` *partially-diffuse leading
//! nodes* whose smoothed moments the ordinary RTS recursion cannot reach (its
//! predicted covariance is rank-deficient there). For `m = 2` that is the
//! single node 0; for `m = 3` the pair {0, 1}. These are recovered exactly by a
//! joint Gaussian conditioning of the whole leading block on the first proper
//! smoothed node (see the smoother pass) — the exact diffuse analog of RTS, and
//! the multi-node generalization of the `m = 2` reverse-Markov closure.
//!
//! Model, after sorting and pooling tied abscissae (precision-weighted):
//!   α_{t+1} = F_t α_t + η_t,   η_t ~ N(0, q·Q(δ_t)),   q = σ_w²/σ² = 1/λ,
//!   y_t     = H α_t + ε_t,     ε_t ~ N(0, σ²/w_t),     H = [1 0 … 0],
//!   F(δ) = exp(δA) (nilpotent shift A),   Q(δ) the m-fold IWP noise,
//! with a diffuse (improper, flat) prior on the first `m` states carrying the
//! unpenalized degree-`<m` polynomial null space the spline leaves unshrunk.
//! (`m = 2`: `F = [[1,δ],[0,1]]`, `Q = [[δ³/3,δ²/2],[δ²/2,δ]]`.)
//!
//! Exactness boundaries, by construction:
//! - the diffuse dimension is `m` and is consumed by the first `m` distinct
//!   abscissae, after which the filter is an ordinary proper Kalman filter;
//! - the `m − 1` partially-diffuse leading nodes are recovered by exact Markov
//!   conditioning of the whole leading block on the first proper smoothed node,
//!   `p(α_{0..m−2} | y) = ∫ p(α_{0..m−2} | α_{m−1}, y_{0..m−2}) p(α_{m−1} | y)`
//!   — an affine `((m−1)m)×m` Bayes update built from the flat leading prior,
//!   the Markov increments, and the leading observations; it reduces to the
//!   single-node reverse-Markov closure at `m = 2` and needs no diffuse RTS
//!   recursion;
//! - off-knot prediction is the Gaussian bridge conditional on the two
//!   flanking smoothed states (using the exact lag-one smoothed
//!   cross-covariance `G_t · P^s_{t+1}`), or boundary extrapolation from the
//!   end states, which reproduces the spline's polynomial extrapolation with
//!   growing variance — bridge-don't-sag is a theorem here.
//!
//! The smoothing parameter is selected by isolating every stationary interval
//! of the concentrated diffuse restricted log-likelihood over log λ. Exact
//! analytic score sensitivities are propagated through the filter, and global
//! curvature bounds drive certified adaptive subdivision; both finite-domain
//! boundaries compete exactly. σ² is profiled in closed form from the proper
//! innovations plus the within-tie residual sum.

use gam_math::score_opt::{
    ClosedInterval, DerivativeEnclosure, ScoreJet, maximize_score_1d,
};

/// One pooled (distinct-abscissa) observation node.
#[derive(Clone, Copy, Debug)]
struct PooledNode {
    x: f64,
    /// Precision-weighted mean of the tied responses.
    y: f64,
    /// Total weight of the pooled ties (observation variance is `σ²/w`).
    w: f64,
}

/// Search interval for log λ (natural log), generous on both sides.
const LOG_LAMBDA_LO: f64 = -18.0;
const LOG_LAMBDA_HI: f64 = 18.0;
/// Numerical floor treating a predicted innovation variance as singular.
const INNOVATION_VAR_FLOOR: f64 = 1e-300;

/// Maximum supported smoothing-spline order handled by the fixed-capacity
/// small-matrix layer. Order `m` penalizes `∫(f^{(m)})²`; the state dimension
/// is `m`. The exact diffuse leading-block smoother (see the smoother pass)
/// recovers the `m − 1` partially-diffuse leading nodes for any `m`: `m = 1`
/// has none, `m = 2` has node 0, `m = 3` has {0, 1}. Order 3 (the quintic
/// smoothing spline, #1044) is the current cap; bumping it further only needs a
/// wider `mat_inv` branch and the (already order-general) leading-block solve.
const MAX_ORDER: usize = 3;

/// Row-major `m × m` matrix stored in a fixed `MAX_ORDER`-capacity buffer; only
/// the top-left `m × m` block is meaningful. Generalizing the order-2 cubic
/// scan to order `m ∈ {1, 2, 3}` (#1034 item 2, #1044) keeps the
/// allocation-free fixed storage of the hot filter loop while letting `m` vary
/// at runtime.
type Mat2 = [[f64; MAX_ORDER]; MAX_ORDER];
type Vec2 = [f64; MAX_ORDER];

#[inline]
fn mat_mul(a: &Mat2, b: &Mat2, m: usize) -> Mat2 {
    let mut c = [[0.0; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0;
            for k in 0..m {
                acc += a[i][k] * b[k][j];
            }
            c[i][j] = acc;
        }
    }
    c
}

#[inline]
fn mat_t(a: &Mat2, m: usize) -> Mat2 {
    let mut c = [[0.0; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in 0..m {
            c[i][j] = a[j][i];
        }
    }
    c
}

#[inline]
fn mat_vec(a: &Mat2, v: &Vec2, m: usize) -> Vec2 {
    let mut out = [0.0; MAX_ORDER];
    for i in 0..m {
        let mut acc = 0.0;
        for j in 0..m {
            acc += a[i][j] * v[j];
        }
        out[i] = acc;
    }
    out
}

#[inline]
fn mat_add(a: &Mat2, b: &Mat2, m: usize) -> Mat2 {
    let mut c = [[0.0; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in 0..m {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

#[inline]
fn mat_sub(a: &Mat2, b: &Mat2, m: usize) -> Mat2 {
    let mut c = [[0.0; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in 0..m {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    c
}

/// Inverse of an `m × m` (`m ∈ {1, 2, 3}`) with a hard singularity error.
/// Closed-form cofactor inverses keep the hot-loop arithmetic exact and
/// branch-free; order 3 is the quintic smoother's state dimension (#1044).
fn mat_inv(a: &Mat2, m: usize, what: &str) -> Result<Mat2, String> {
    let mut out = [[0.0; MAX_ORDER]; MAX_ORDER];
    match m {
        1 => {
            let d = a[0][0];
            if !(d.is_finite() && d.abs() > 0.0) {
                return Err(format!("spline scan: singular 1x1 in {what} (a00={d})"));
            }
            out[0][0] = 1.0 / d;
        }
        2 => {
            let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
            if !(det.is_finite() && det.abs() > 0.0) {
                return Err(format!("spline scan: singular 2x2 in {what} (det={det})"));
            }
            out[0][0] = a[1][1] / det;
            out[0][1] = -a[0][1] / det;
            out[1][0] = -a[1][0] / det;
            out[1][1] = a[0][0] / det;
        }
        3 => {
            // Cofactor / adjugate inverse. Cofactors of the 2×2 minors:
            let c00 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
            let c01 = a[1][2] * a[2][0] - a[1][0] * a[2][2];
            let c02 = a[1][0] * a[2][1] - a[1][1] * a[2][0];
            let det = a[0][0] * c00 + a[0][1] * c01 + a[0][2] * c02;
            if !(det.is_finite() && det.abs() > 0.0) {
                return Err(format!("spline scan: singular 3x3 in {what} (det={det})"));
            }
            let inv_det = 1.0 / det;
            // inv = adj/det = (cofactor matrix)ᵀ / det.
            out[0][0] = c00 * inv_det;
            out[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_det;
            out[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det;
            out[1][0] = c01 * inv_det;
            out[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det;
            out[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_det;
            out[2][0] = c02 * inv_det;
            out[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_det;
            out[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det;
        }
        _ => return Err(format!("spline scan: unsupported order {m} in {what}")),
    }
    Ok(out)
}

/// Inverse of a general dense `d × d` SPD matrix via Gauss–Jordan elimination
/// with partial pivoting, symmetric diagonal (Jacobi) equilibration, and one
/// iterative-refinement step. Used once per fit by the leading-block diffuse
/// smoother (dimension `(order−1)·order ≤ 6`), so clarity over speed — it is
/// NOT on the hot REML grid path (that runs only `run_filter`).
///
/// Equilibration matters at order `m ≥ 3`: the IWP process noise `Q(δ)` scales
/// the `f^{(k)}` state components by `δ^{2m−1}` down to `δ`, so its inverse
/// `(qQ)⁻¹` — and hence the leading-block precision `Λ` — spans many orders of
/// magnitude (the f-component carries the `O(w)` observation term, the
/// high-derivative components carry `O(1/(qδ^{2m−1}))` penalty mass). A bare
/// Gauss–Jordan inverse of such a `Λ` loses `≈ ε·κ(Λ)` digits, which at heavy
/// smoothing (small `q`) would corrupt the quintic's leading smoothed nodes.
/// Rescaling to unit diagonal (`Λ̃ = SΛS`, `s_i = 1/√Λ_ii`) collapses that
/// scale disparity before the elimination, then `Λ⁻¹ = S Λ̃⁻¹ S`.
fn dense_spd_inverse(a: &[Vec<f64>], what: &str) -> Result<Vec<Vec<f64>>, String> {
    let d = a.len();
    // Jacobi equilibration scale s_i = 1/√Λ_ii (Λ SPD ⇒ Λ_ii > 0).
    let s: Vec<f64> = (0..d)
        .map(|i| {
            let dii = a[i][i];
            if dii.is_finite() && dii > 0.0 {
                1.0 / dii.sqrt()
            } else {
                1.0
            }
        })
        .collect();
    let a_s: Vec<Vec<f64>> = (0..d)
        .map(|i| (0..d).map(|j| s[i] * a[i][j] * s[j]).collect())
        .collect();
    // Gauss–Jordan inverse of the equilibrated matrix.
    let mut inv_s = gauss_jordan_inverse(&a_s, what)?;
    // One iterative-refinement step against the equilibrated system:
    // X ← X + X·(I − Λ̃·X), reducing the residual to near machine precision.
    let mut resid = vec![vec![0.0_f64; d]; d]; // R = I − Λ̃·X
    for i in 0..d {
        for j in 0..d {
            let mut ax = 0.0;
            for k in 0..d {
                ax += a_s[i][k] * inv_s[k][j];
            }
            resid[i][j] = f64::from(u8::from(i == j)) - ax;
        }
    }
    let mut delta = vec![vec![0.0_f64; d]; d]; // ΔX = X·R
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0;
            for k in 0..d {
                acc += inv_s[i][k] * resid[k][j];
            }
            delta[i][j] = acc;
        }
    }
    for i in 0..d {
        for j in 0..d {
            inv_s[i][j] += delta[i][j];
        }
    }
    // Un-equilibrate: Λ⁻¹ = S·Λ̃⁻¹·S.
    Ok((0..d)
        .map(|i| (0..d).map(|j| s[i] * inv_s[i][j] * s[j]).collect())
        .collect())
}

/// Gauss–Jordan inverse with partial pivoting (helper for `dense_spd_inverse`).
fn gauss_jordan_inverse(a: &[Vec<f64>], what: &str) -> Result<Vec<Vec<f64>>, String> {
    let d = a.len();
    let mut aug = a.to_vec();
    let mut inv = vec![vec![0.0_f64; d]; d];
    for i in 0..d {
        inv[i][i] = 1.0;
    }
    for col in 0..d {
        let piv = (col..d)
            .max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
            .unwrap();
        let p = aug[piv][col];
        if !(p.is_finite() && p.abs() > 0.0) {
            return Err(format!(
                "spline scan: singular {d}x{d} in {what} (pivot={p})"
            ));
        }
        aug.swap(col, piv);
        inv.swap(col, piv);
        let d_piv = aug[col][col];
        for k in 0..d {
            aug[col][k] /= d_piv;
            inv[col][k] /= d_piv;
        }
        for r in 0..d {
            if r == col {
                continue;
            }
            let f = aug[r][col];
            if f == 0.0 {
                continue;
            }
            for k in 0..d {
                aug[r][k] -= f * aug[col][k];
                inv[r][k] -= f * inv[col][k];
            }
        }
    }
    Ok(inv)
}

/// Factorials `k!` for `k ≤ 2·MAX_ORDER` — the only ones the order-`m`
/// transition and process-noise formulas reference.
#[inline]
fn factorial(k: usize) -> f64 {
    (1..=k).map(|v| v as f64).product::<f64>().max(1.0)
}

/// Transition `F(δ) = exp(δ·A)` of the `m`-th order integrated Wiener process,
/// `A` the nilpotent shift: `F[i][j] = δ^{j−i}/(j−i)!` for `j ≥ i`, else 0.
/// `m = 1 ⇒ [[1]]`; `m = 2 ⇒ [[1, δ], [0, 1]]` (the cubic case, unchanged).
#[inline]
fn transition(delta: f64, m: usize) -> Mat2 {
    let mut f = [[0.0; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in i..m {
            f[i][j] = delta.powi((j - i) as i32) / factorial(j - i);
        }
    }
    f
}

/// Process noise `Q(δ) = ∫₀^δ e^{As} b bᵀ e^{Aᵀs} ds` (`b = e_{m−1}`) of the
/// `m`-th order IWP at unit `q`, scaled by `q`:
/// `Q[i][j] = q · δ^{2m−1−i−j} / ((m−1−i)! (m−1−j)! (2m−1−i−j))`.
/// `m = 1 ⇒ [[q·δ]]`; `m = 2 ⇒ [[q·δ³/3, q·δ²/2], [q·δ²/2, q·δ]]` (unchanged).
#[inline]
fn process_noise(delta: f64, q: f64, m: usize) -> Mat2 {
    let mut out = [[0.0; MAX_ORDER]; MAX_ORDER];
    for i in 0..m {
        for j in 0..m {
            let p = 2 * m - 1 - i - j;
            out[i][j] = q * delta.powi(p as i32)
                / (factorial(m - 1 - i) * factorial(m - 1 - j) * (p as f64));
        }
    }
    out
}

/// Symmetrize in place against drift from the rank-one update arithmetic.
#[inline]
fn symmetrize(a: &mut Mat2, m: usize) {
    for i in 0..m {
        for j in (i + 1)..m {
            let off = 0.5 * (a[i][j] + a[j][i]);
            a[i][j] = off;
            a[j][i] = off;
        }
    }
}

/// Per-node filter storage needed by the RTS backward pass.
struct FilterStep {
    /// Filtered mean `a_{t|t}` and proper covariance `P*_{t|t}`.
    a_filt: Vec2,
    p_filt: Mat2,
    /// One-step prediction `a_{t|t-1}`, proper covariance `P*_{t|t-1}` (for t ≥ 1).
    a_pred: Vec2,
    p_pred: Mat2,
}

/// Output of one full filter pass at a fixed `q = 1/λ` (run at unit σ²).
struct FilterPass {
    steps: Vec<FilterStep>,
    /// Σ over proper steps of `log F̃_t` (innovation variances at σ²=1).
    sum_log_f: f64,
    /// First and second analytic derivatives of `sum_log_f` with respect to
    /// `rho = log lambda` (`q = exp(-rho)`).
    sum_log_f_d1: f64,
    sum_log_f_d2: f64,
    /// Σ over proper steps of `v_t² / F̃_t`.
    sum_v2_over_f: f64,
    /// First and second analytic `rho` derivatives of `sum_v2_over_f`.
    sum_v2_over_f_d1: f64,
    sum_v2_over_f_d2: f64,
    /// Number of proper (non-diffuse) innovations.
    n_proper: usize,
}

fn run_filter(nodes: &[PooledNode], q: f64, order: usize) -> Result<FilterPass, String> {
    let n = nodes.len();
    let mut steps = Vec::with_capacity(n);
    // Exact diffuse initialization (Durbin–Koopman): P = P* + κ·P_∞, κ → ∞.
    // The order-`m` polynomial null space (degree < m) is fully diffuse: the
    // diffuse rank starts at `order`, consumed by the first `order` distinct
    // abscissae.
    let mut a: Vec2 = [0.0; MAX_ORDER];
    let mut a_d1: Vec2 = [0.0; MAX_ORDER];
    let mut a_d2: Vec2 = [0.0; MAX_ORDER];
    let mut p_star: Mat2 = [[0.0; MAX_ORDER]; MAX_ORDER];
    let mut p_star_d1: Mat2 = [[0.0; MAX_ORDER]; MAX_ORDER];
    let mut p_star_d2: Mat2 = [[0.0; MAX_ORDER]; MAX_ORDER];
    let mut p_inf: Mat2 = [[0.0; MAX_ORDER]; MAX_ORDER];
    for i in 0..order {
        p_inf[i][i] = 1.0;
    }
    let mut diffuse_rank = order;
    let mut sum_log_f = 0.0;
    let mut sum_log_f_d1 = 0.0;
    let mut sum_log_f_d2 = 0.0;
    let mut sum_v2_over_f = 0.0;
    let mut sum_v2_over_f_d1 = 0.0;
    let mut sum_v2_over_f_d2 = 0.0;
    let mut n_proper = 0usize;
    for t in 0..n {
        let a_pred = a;
        let p_pred = p_star;
        let r = 1.0 / nodes[t].w;
        let v = nodes[t].y - a[0];
        let v_d1 = -a_d1[0];
        let v_d2 = -a_d2[0];
        // H = [1 0 … 0] ⇒ M = P·H' is the first column, F = M[0] (+ r).
        let mut m_star: Vec2 = [0.0; MAX_ORDER];
        let mut m_star_d1: Vec2 = [0.0; MAX_ORDER];
        let mut m_star_d2: Vec2 = [0.0; MAX_ORDER];
        for i in 0..order {
            m_star[i] = p_star[i][0];
            m_star_d1[i] = p_star_d1[i][0];
            m_star_d2[i] = p_star_d2[i][0];
        }
        let f_star = m_star[0] + r;
        let f_star_d1 = m_star_d1[0];
        let f_star_d2 = m_star_d2[0];
        let mut proper_update = diffuse_rank == 0;
        if diffuse_rank > 0 {
            let mut m_inf: Vec2 = [0.0; MAX_ORDER];
            for i in 0..order {
                m_inf[i] = p_inf[i][0];
            }
            let f_inf = m_inf[0];
            if f_inf > INNOVATION_VAR_FLOOR {
                // Exact diffuse update (Koopman 1997): the κ→∞ limit of the
                // standard update; the diffuse step contributes −½·log F_∞ to
                // the restricted likelihood and consumes one diffuse dimension.
                for i in 0..order {
                    let k_inf = m_inf[i] / f_inf;
                    a[i] += k_inf * v;
                    a_d1[i] += k_inf * v_d1;
                    a_d2[i] += k_inf * v_d2;
                }
                let mut p_new = p_star;
                let mut p_new_d1 = p_star_d1;
                let mut p_new_d2 = p_star_d2;
                for i in 0..order {
                    for j in 0..order {
                        p_new[i][j] += -m_inf[i] * m_star[j] / f_inf - m_star[i] * m_inf[j] / f_inf
                            + m_inf[i] * m_inf[j] * f_star / (f_inf * f_inf);
                        p_new_d1[i][j] += -m_inf[i] * m_star_d1[j] / f_inf
                            - m_star_d1[i] * m_inf[j] / f_inf
                            + m_inf[i] * m_inf[j] * f_star_d1 / (f_inf * f_inf);
                        p_new_d2[i][j] += -m_inf[i] * m_star_d2[j] / f_inf
                            - m_star_d2[i] * m_inf[j] / f_inf
                            + m_inf[i] * m_inf[j] * f_star_d2 / (f_inf * f_inf);
                    }
                }
                p_star = p_new;
                p_star_d1 = p_new_d1;
                p_star_d2 = p_new_d2;
                symmetrize(&mut p_star, order);
                symmetrize(&mut p_star_d1, order);
                symmetrize(&mut p_star_d2, order);
                for i in 0..order {
                    for j in 0..order {
                        p_inf[i][j] -= m_inf[i] * m_inf[j] / f_inf;
                    }
                }
                symmetrize(&mut p_inf, order);
                diffuse_rank -= 1;
                if diffuse_rank == 0 {
                    p_inf = [[0.0; MAX_ORDER]; MAX_ORDER];
                }
            } else {
                // Diffuse direction orthogonal to H: this observation is an
                // ordinary proper update of P* even though diffuse rank remains.
                proper_update = true;
            }
        }
        if proper_update {
            if f_star <= INNOVATION_VAR_FLOOR {
                return Err("spline scan: non-positive innovation variance".to_string());
            }
            let inv_f = 1.0 / f_star;
            let inv_f2 = inv_f * inv_f;
            let inv_f3 = inv_f2 * inv_f;
            let mut gain = [0.0; MAX_ORDER];
            let mut gain_d1 = [0.0; MAX_ORDER];
            let mut gain_d2 = [0.0; MAX_ORDER];
            for i in 0..order {
                gain[i] = m_star[i] * inv_f;
                gain_d1[i] = m_star_d1[i] * inv_f - m_star[i] * f_star_d1 * inv_f2;
                gain_d2[i] = m_star_d2[i] * inv_f
                    - 2.0 * m_star_d1[i] * f_star_d1 * inv_f2
                    - m_star[i] * f_star_d2 * inv_f2
                    + 2.0 * m_star[i] * f_star_d1 * f_star_d1 * inv_f3;
            }
            let a_old_d1 = a_d1;
            let a_old_d2 = a_d2;
            for i in 0..order {
                a[i] += gain[i] * v;
                a_d1[i] = a_old_d1[i] + gain_d1[i] * v + gain[i] * v_d1;
                a_d2[i] = a_old_d2[i] + gain_d2[i] * v + 2.0 * gain_d1[i] * v_d1 + gain[i] * v_d2;
            }
            let mut p_new = p_star;
            let mut p_new_d1 = p_star_d1;
            let mut p_new_d2 = p_star_d2;
            for i in 0..order {
                for j in 0..order {
                    let mm = m_star[i] * m_star[j];
                    let mm_d1 = m_star_d1[i] * m_star[j] + m_star[i] * m_star_d1[j];
                    let mm_d2 = m_star_d2[i] * m_star[j]
                        + 2.0 * m_star_d1[i] * m_star_d1[j]
                        + m_star[i] * m_star_d2[j];
                    p_new[i][j] -= mm * inv_f;
                    p_new_d1[i][j] -= mm_d1 * inv_f - mm * f_star_d1 * inv_f2;
                    p_new_d2[i][j] -=
                        mm_d2 * inv_f - 2.0 * mm_d1 * f_star_d1 * inv_f2 - mm * f_star_d2 * inv_f2
                            + 2.0 * mm * f_star_d1 * f_star_d1 * inv_f3;
                }
            }
            p_star = p_new;
            p_star_d1 = p_new_d1;
            p_star_d2 = p_new_d2;
            symmetrize(&mut p_star, order);
            symmetrize(&mut p_star_d1, order);
            symmetrize(&mut p_star_d2, order);

            let vv = v * v;
            let vv_d1 = 2.0 * v * v_d1;
            let vv_d2 = 2.0 * (v_d1 * v_d1 + v * v_d2);
            sum_log_f += f_star.ln();
            sum_log_f_d1 += f_star_d1 * inv_f;
            sum_log_f_d2 += f_star_d2 * inv_f - f_star_d1 * f_star_d1 * inv_f2;
            sum_v2_over_f += vv * inv_f;
            sum_v2_over_f_d1 += vv_d1 * inv_f - vv * f_star_d1 * inv_f2;
            sum_v2_over_f_d2 +=
                vv_d2 * inv_f - 2.0 * vv_d1 * f_star_d1 * inv_f2 - vv * f_star_d2 * inv_f2
                    + 2.0 * vv * f_star_d1 * f_star_d1 * inv_f3;
            n_proper += 1;
        }
        steps.push(FilterStep {
            a_filt: a,
            p_filt: p_star,
            a_pred,
            p_pred,
        });
        // Predict to the next node.
        if t + 1 < n {
            let delta = nodes[t + 1].x - nodes[t].x;
            let f_t = transition(delta, order);
            a = mat_vec(&f_t, &a, order);
            a_d1 = mat_vec(&f_t, &a_d1, order);
            a_d2 = mat_vec(&f_t, &a_d2, order);
            let f_t_t = mat_t(&f_t, order);
            let q_noise = process_noise(delta, q, order);
            let mut p_next = mat_add(
                &mat_mul(&mat_mul(&f_t, &p_star, order), &f_t_t, order),
                &q_noise,
                order,
            );
            let mut p_next_d1 = mat_sub(
                &mat_mul(&mat_mul(&f_t, &p_star_d1, order), &f_t_t, order),
                &q_noise,
                order,
            );
            let mut p_next_d2 = mat_add(
                &mat_mul(&mat_mul(&f_t, &p_star_d2, order), &f_t_t, order),
                &q_noise,
                order,
            );
            symmetrize(&mut p_next, order);
            symmetrize(&mut p_next_d1, order);
            symmetrize(&mut p_next_d2, order);
            p_star = p_next;
            p_star_d1 = p_next_d1;
            p_star_d2 = p_next_d2;
            if diffuse_rank > 0 {
                let mut pi_next =
                    mat_mul(&mat_mul(&f_t, &p_inf, order), &mat_t(&f_t, order), order);
                symmetrize(&mut pi_next, order);
                p_inf = pi_next;
            }
        }
    }
    Ok(FilterPass {
        steps,
        sum_log_f,
        sum_log_f_d1,
        sum_log_f_d2,
        sum_v2_over_f,
        sum_v2_over_f_d1,
        sum_v2_over_f_d2,
        n_proper,
    })
}

/// Fitted exact smoothing-spline posterior on the pooled knots.
#[derive(Clone, Debug)]
pub struct SplineScanFit {
    /// Smoothing-spline order `m` (penalize `∫(f^{(m)})²`); state dimension.
    /// `m = 1` is the random-walk/linear smoother, `m = 2` the cubic smoother,
    /// `m = 3` the quintic smoother.
    pub order: usize,
    /// Distinct sorted abscissae (pooled knots).
    pub knots: Vec<f64>,
    /// Smoothed posterior mean of `f` at each knot.
    pub mean: Vec<f64>,
    /// Smoothed posterior mean of `f′` at each knot, present only for order
    /// `m ≥ 2`. At `m = 1` the latent process is Brownian motion, which has NO
    /// pointwise derivative state (it is a.s. nondifferentiable), so this is
    /// `None` rather than a fabricated zero.
    pub deriv: Option<Vec<f64>>,
    /// Posterior variance of `f` at each knot (scaled by `sigma2`).
    pub var: Vec<f64>,
    /// Selected (or supplied) log smoothing parameter `log λ`.
    pub log_lambda: f64,
    /// Profiled (or supplied) observation variance σ².
    pub sigma2: f64,
    /// Concentrated diffuse restricted log-likelihood at the optimum, up to a
    /// λ- and data-independent additive constant. Differences across λ are
    /// exact REML criterion differences.
    pub restricted_loglik: f64,
    /// Raw observation count `n` (pre-pooling; ties collapse to fewer knots),
    /// retained for the residual d.o.f. `n − order` (#1046).
    pub n_obs: usize,
    /// Weighted DATA residual sum of squares `Σ wᵢ (yᵢ − f̂(xᵢ))²` at the
    /// smoothed posterior mean. Stored explicitly because the profiled
    /// innovations quadratic `σ̂²·(n − order)` is the REML objective's
    /// quadratic — data residual energy PLUS process/roughness energy at the
    /// posterior mode — and is therefore NOT the Gaussian deviance.
    pub data_sse: f64,
    /// Smoothed full states `(f, f′)` per knot.
    smoothed_state: Vec<Vec2>,
    /// Smoothed full state covariances per knot (unit-σ² scale).
    smoothed_cov: Vec<Mat2>,
    /// RTS backward gains `G_t` (lag-one cross-covariance is `G_t · P^s_{t+1}`).
    rts_gain: Vec<Mat2>,
    /// q = 1/λ used by the pass (unit-σ² scale).
    q: f64,
    /// Pooled observation weight per knot (sum of tied raw weights).
    node_weight: Vec<f64>,
}

/// Pool tied abscissae and validate inputs. Returns nodes plus the within-tie
/// weighted residual sum and the raw observation count.
fn pool_nodes(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    order: usize,
) -> Result<(Vec<PooledNode>, f64, usize), String> {
    let n = x.len();
    if y.len() != n || w.len() != n {
        return Err(format!(
            "spline scan: length mismatch x={n}, y={}, w={}",
            y.len(),
            w.len()
        ));
    }
    for i in 0..n {
        if !(x[i].is_finite() && y[i].is_finite() && w[i].is_finite() && w[i] > 0.0) {
            return Err(format!(
                "spline scan: non-finite or non-positive input at row {i} (x={}, y={}, w={})",
                x[i], y[i], w[i]
            ));
        }
    }
    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&i, &j| x[i].total_cmp(&x[j]));
    let mut nodes: Vec<PooledNode> = Vec::new();
    for &i in &perm {
        match nodes.last_mut() {
            Some(last) if last.x == x[i] => {
                let w_new = last.w + w[i];
                last.y = (last.y * last.w + y[i] * w[i]) / w_new;
                last.w = w_new;
            }
            _ => nodes.push(PooledNode {
                x: x[i],
                y: y[i],
                w: w[i],
            }),
        }
    }
    // Need the `order` diffuse dimensions plus at least one proper innovation.
    if nodes.len() < order + 1 {
        return Err(format!(
            "spline scan: order {order} needs at least {} distinct abscissae, got {}",
            order + 1,
            nodes.len()
        ));
    }
    // Within-tie residual sum Σ w_i (y_i − ȳ_group)², part of the profiled σ².
    let mut ssr_within = 0.0;
    let mut k = 0usize;
    for &i in &perm {
        while nodes[k].x != x[i] {
            k += 1;
        }
        let d = y[i] - nodes[k].y;
        ssr_within += w[i] * d * d;
    }
    Ok((nodes, ssr_within, n))
}

/// Concentrated diffuse restricted log-likelihood and its exact first two
/// derivatives with respect to `log λ` (σ² profiled). The derivatives are
/// propagated through the same diffuse Kalman recursion as the value; no
/// finite differencing or surrogate objective is involved.
fn concentrated_criterion_jet(
    nodes: &[PooledNode],
    ssr_within: f64,
    n_obs: usize,
    log_lambda: f64,
    order: usize,
) -> Result<(f64, f64, f64), String> {
    let pass = run_filter(nodes, (-log_lambda).exp(), order)?;
    // Profiled σ̂² over the proper innovations plus within-tie residuals;
    // the restricted degrees of freedom subtract the diffuse dimension `order`.
    let dof = (n_obs - order) as f64;
    let rss = pass.sum_v2_over_f + ssr_within;
    if rss <= 0.0 {
        return Err("spline scan: degenerate zero residual sum".to_string());
    }
    let sigma2 = rss / dof;
    if pass.n_proper != nodes.len() - order {
        return Err(format!(
            "spline scan: expected {} proper innovations, got {} (diffuse rank not consumed)",
            nodes.len() - order,
            pass.n_proper
        ));
    }
    let rss_d1 = pass.sum_v2_over_f_d1;
    let rss_d2 = pass.sum_v2_over_f_d2;
    let rss_log_d1 = rss_d1 / rss;
    let rss_log_d2 = rss_d2 / rss - rss_log_d1 * rss_log_d1;
    Ok((
        -0.5 * (pass.sum_log_f + dof * sigma2.ln()),
        -0.5 * (pass.sum_log_f_d1 + dof * rss_log_d1),
        -0.5 * (pass.sum_log_f_d2 + dof * rss_log_d2),
    ))
}

/// Rigorous interval enclosure of the score's first two derivatives.
///
/// After eliminating the diffuse polynomial null space, the Gaussian profile
/// is an affine covariance pencil. Every determinant mode has response
/// `u in [0,1]`; every normalized profiled-residual derivative is a convex
/// average of the same kernels. Consequently
///
/// `|L''| <= 1/2 (r/4 + 2 nu)` and
/// `|L'''| <= 1/2 (r/4 + 6 nu)`,
///
/// where `r` is the number of proper innovation modes and `nu=n-order` is the
/// residual d.f. Within-tie residual energy is lambda-independent and only
/// tightens these bounds. Endpoint jets plus these analytic Lipschitz bounds
/// therefore enclose the entire interval without a sampling lattice.
fn concentrated_criterion_enclosure(
    nodes: &[PooledNode],
    ssr_within: f64,
    n_obs: usize,
    lo: f64,
    hi: f64,
    order: usize,
) -> Result<DerivativeEnclosure, String> {
    if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
        return Err(format!(
            "spline scan: invalid score-enclosure interval [{lo}, {hi}]"
        ));
    }
    let left = concentrated_criterion_jet(nodes, ssr_within, n_obs, lo, order)?;
    let right = concentrated_criterion_jet(nodes, ssr_within, n_obs, hi, order)?;
    let width = hi - lo;
    let proper_modes = (nodes.len() - order) as f64;
    let residual_dof = (n_obs - order) as f64;
    let curvature_abs_bound = 0.5 * (0.25 * proper_modes + 2.0 * residual_dof);
    let third_abs_bound = 0.5 * (0.25 * proper_modes + 6.0 * residual_dof);
    let derivative_radius = curvature_abs_bound * width;
    let curvature_radius = third_abs_bound * width;
    Ok(DerivativeEnclosure {
        derivative: ClosedInterval::outward(
            (left.1 - derivative_radius).min(right.1 - derivative_radius),
            (left.1 + derivative_radius).max(right.1 + derivative_radius),
        ),
        curvature: ClosedInterval::outward(
            (left.2 - curvature_radius).min(right.2 - curvature_radius),
            (left.2 + curvature_radius).max(right.2 + curvature_radius),
        ),
    })
}

/// Value-only diagnostic surface retained for the derivative oracle tests.
#[cfg(test)]
fn concentrated_criterion(
    nodes: &[PooledNode],
    ssr_within: f64,
    n_obs: usize,
    log_lambda: f64,
    order: usize,
) -> Result<f64, String> {
    Ok(concentrated_criterion_jet(nodes, ssr_within, n_obs, log_lambda, order)?.0)
}

/// Exact diffuse smoother for the `order−1` partially-diffuse leading nodes
/// (#1044 — the multi-node generalization of the `m = 2` reverse-Markov
/// closure).
///
/// Ordinary RTS recovers every node `t ≥ order−1` (where the filtered
/// distribution is proper). The first `order−1` nodes are partially diffuse:
/// their filtered covariance still carries unresolved diffuse mass, so RTS —
/// which needs the predicted covariance `P_{t+1|t}` to be invertible — cannot
/// reach them. By the Markov property the leading block depends on all future
/// data ONLY through the first proper smoothed node `α_{order−1}`:
///
///   p(α_{0..order−2} | y) = ∫ p(α_{0..order−2} | α_{order−1}, y_{0..order−2})
///                             · p(α_{order−1} | y) dα_{order−1}.
///
/// The inner conditional is a proper Gaussian: it is the flat (improper)
/// leading prior tightened by the Markov increments `(α_{t+1} − Fα_t)ᵀ(qQ)⁻¹(·)`
/// and the leading observations `w_t (y_t − f_t)²`, with `α_{order−1}` entering
/// linearly through the last increment. Writing `u = (α_0, …, α_{order−2})`,
///
///   u | α_{order−1} ~ N(C·α_{order−1} + d,  Σ),   Σ = Λ⁻¹,
///   Λ  = increments(F'(qQ)⁻¹F …) + leading obs,
///   d  = Σ·b_const,   C = Σ·B   (B = the pinned-node coupling F'(qQ)⁻¹),
///
/// and pushing the smoothed `α_{order−1} ~ N(α̂_p, V_p)` through the affine map
/// gives the EXACT smoothed leading block, its covariances, and the lag-one
/// cross-covariances `Cov(α_j, α_{j+1} | y)` the bridge `predict` needs:
///
///   mean(u) = C·α̂_p + d,   Cov(u) = C V_p Cᵀ + Σ,   Cov(u, α_p) = C V_p.
///
/// This is exact Gaussian conditioning — no diffuse RTS recursion, no
/// sign-convention-laden `r/N` adjoint. At `order = 2` (one leading node) it is
/// algebraically the existing single-node closure.
fn leading_block_smooth(
    sm_state: &mut [Vec2],
    sm_cov: &mut [Mat2],
    gains: &mut [Mat2],
    nodes: &[PooledNode],
    q: f64,
    order: usize,
) -> Result<(), String> {
    let nb = order - 1; // leading nodes 0..nb-1 (the partially-diffuse ones)
    let pin = order - 1; // first proper smoothed node (conditioning anchor)
    let d = nb * order; // joint dimension of the leading block
    let mut lambda = vec![vec![0.0_f64; d]; d];
    let mut b_const = vec![0.0_f64; d];
    let mut bmat = vec![vec![0.0_f64; order]; d]; // coupling to the pinned node

    // Markov increments t = 0..order-2, each connecting node t and node t+1.
    for t in 0..order - 1 {
        let delta = nodes[t + 1].x - nodes[t].x;
        let f = transition(delta, order);
        let qn = process_noise(delta, q, order);
        let a = mat_inv(&qn, order, "leading-block increment noise")?; // (qQ)⁻¹ (symmetric)
        let ft = mat_t(&f, order);
        let fta = mat_mul(&ft, &a, order); // F'A
        let ftaf = mat_mul(&fta, &f, order); // F'A F
        let af = mat_mul(&a, &f, order); // A F = (F'A)'
        // Node t diagonal block (node t is always in the block): += F'A F.
        for i in 0..order {
            for j in 0..order {
                lambda[t * order + i][t * order + j] += ftaf[i][j];
            }
        }
        if t + 1 <= nb - 1 {
            // Both nodes are in the block: fill node t+1's diagonal and the
            // symmetric cross blocks.
            for i in 0..order {
                for j in 0..order {
                    lambda[(t + 1) * order + i][(t + 1) * order + j] += a[i][j];
                    lambda[t * order + i][(t + 1) * order + j] -= fta[i][j];
                    lambda[(t + 1) * order + i][t * order + j] -= af[i][j];
                }
            }
        } else {
            // t+1 is the pinned node: it enters the conditional only linearly,
            // through B (its coupling into node t's score is F'A·α_pin).
            for i in 0..order {
                for j in 0..order {
                    bmat[t * order + i][j] += fta[i][j];
                }
            }
        }
    }
    // Leading observations: y_t informs the f-component (local index 0) of node t.
    for t in 0..nb {
        let w = nodes[t].w;
        lambda[t * order][t * order] += w;
        b_const[t * order] += w * nodes[t].y;
    }

    // Conditional covariance Σ = Λ⁻¹, intercept d = Σ·b_const, coupling C = Σ·B.
    let sigma = dense_spd_inverse(&lambda, "leading-block precision")?;
    let dvec: Vec<f64> = (0..d)
        .map(|i| (0..d).map(|k| sigma[i][k] * b_const[k]).sum())
        .collect();
    let cmat: Vec<Vec<f64>> = (0..d)
        .map(|i| {
            (0..order)
                .map(|j| (0..d).map(|k| sigma[i][k] * bmat[k][j]).sum())
                .collect()
        })
        .collect();

    // Pinned smoothed moments (from the ordinary RTS pass).
    let ahat_p = sm_state[pin];
    let vp = sm_cov[pin];
    // cvp = C·V_p  (= Cov(u, α_pin)), D×order.
    let cvp: Vec<Vec<f64>> = (0..d)
        .map(|i| {
            (0..order)
                .map(|j| (0..order).map(|k| cmat[i][k] * vp[k][j]).sum())
                .collect()
        })
        .collect();
    // mean(u) = C·α̂_p + d.
    let mean_u: Vec<f64> = (0..d)
        .map(|i| (0..order).map(|j| cmat[i][j] * ahat_p[j]).sum::<f64>() + dvec[i])
        .collect();
    // Cov(u) = cvp·Cᵀ + Σ.
    let cov_u: Vec<Vec<f64>> = (0..d)
        .map(|i| {
            (0..d)
                .map(|k| (0..order).map(|j| cvp[i][j] * cmat[k][j]).sum::<f64>() + sigma[i][k])
                .collect()
        })
        .collect();

    // Scatter the smoothed leading states and covariances.
    for j in 0..nb {
        for i in 0..order {
            sm_state[j][i] = mean_u[j * order + i];
        }
        let mut cov = [[0.0_f64; MAX_ORDER]; MAX_ORDER];
        for i in 0..order {
            for k in 0..order {
                cov[i][k] = cov_u[j * order + i][j * order + k];
            }
        }
        symmetrize(&mut cov, order);
        sm_cov[j] = cov;
    }
    // Lag-one bridge gains for the leading intervals [j, j+1], j = 0..order-2.
    // gain_j = Cov(α_j, α_{j+1} | y) · Cov(α_{j+1} | y)⁻¹, so that the bridge's
    // `gain_j · P^s_{j+1}` reproduces the exact lag-one smoothed cross-cov.
    for j in 0..nb {
        let mut cross = [[0.0_f64; MAX_ORDER]; MAX_ORDER];
        if j + 1 <= nb - 1 {
            // Both in the block: read the (j, j+1) sub-block of Cov(u).
            for i in 0..order {
                for k in 0..order {
                    cross[i][k] = cov_u[j * order + i][(j + 1) * order + k];
                }
            }
        } else {
            // j+1 is the pinned node: read node j's rows of Cov(u, α_pin) = cvp.
            for i in 0..order {
                for k in 0..order {
                    cross[i][k] = cvp[j * order + i][k];
                }
            }
        }
        let denom_inv = mat_inv(&sm_cov[j + 1], order, "leading-block gain denominator")?;
        gains[j] = mat_mul(&cross, &denom_inv, order);
    }
    Ok(())
}

/// Fit at a FIXED `log λ` and order `m ∈ {1, 2, 3}`, σ² either supplied or
/// profiled.
pub fn fit_spline_scan_at(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    log_lambda: f64,
    sigma2: Option<f64>,
    order: usize,
) -> Result<SplineScanFit, String> {
    if order == 0 || order > MAX_ORDER {
        return Err(format!(
            "spline scan: order must be in 1..={MAX_ORDER}, got {order}"
        ));
    }
    let (nodes, ssr_within, n_obs) = pool_nodes(x, y, w, order)?;
    let q = (-log_lambda).exp();
    let pass = run_filter(&nodes, q, order)?;
    let n = nodes.len();
    let dof = (n_obs - order) as f64;
    let sigma2 = match sigma2 {
        Some(s) => {
            if !(s.is_finite() && s > 0.0) {
                return Err(format!("spline scan: invalid sigma2 {s}"));
            }
            s
        }
        None => (pass.sum_v2_over_f + ssr_within) / dof,
    };
    // Full diffuse restricted log-likelihood at this (λ, σ²), up to λ- and
    // σ-free additive constants: −½[Σ log F̃ + dof·ln σ² + RSS/σ²]. At the
    // profiled σ̂² the quadratic term collapses to the λ-free constant `dof`,
    // matching `concentrated_criterion` up to that constant.
    let rss = pass.sum_v2_over_f + ssr_within;
    let restricted_loglik = -0.5 * (pass.sum_log_f + dof * sigma2.ln() + rss / sigma2);

    // ── Smoother: ordinary RTS for the proper nodes (t ≥ order−1) plus an
    // exact diffuse conditioning of the `order−1` leading nodes. ──
    // The filtered distribution is fully proper from node order−1 onward (the
    // diffuse rank, = order, is consumed by node order−1), so ordinary RTS is
    // valid for t ≥ order−1. The first order−1 nodes are partially diffuse —
    // their filtered covariance still carries unresolved diffuse mass and the
    // RTS predicted-covariance inverse is singular there — and are recovered
    // exactly, jointly, by `leading_block_smooth` (conditioning the whole
    // leading block on the first proper smoothed node). For order = 1 there is
    // no leading node and RTS covers every node down to t = 0.
    let mut sm_state = vec![[0.0_f64; MAX_ORDER]; n];
    let mut sm_cov = vec![[[0.0_f64; MAX_ORDER]; MAX_ORDER]; n];
    let mut gains = vec![[[0.0_f64; MAX_ORDER]; MAX_ORDER]; n];
    sm_state[n - 1] = pass.steps[n - 1].a_filt;
    sm_cov[n - 1] = pass.steps[n - 1].p_filt;
    for t in (order - 1..n - 1).rev() {
        let p_next_pred = &pass.steps[t + 1].p_pred;
        let delta = nodes[t + 1].x - nodes[t].x;
        let f_t = transition(delta, order);
        let p_inv = mat_inv(p_next_pred, order, "RTS predicted covariance")?;
        let g = mat_mul(
            &mat_mul(&pass.steps[t].p_filt, &mat_t(&f_t, order), order),
            &p_inv,
            order,
        );
        let mut dm: Vec2 = [0.0; MAX_ORDER];
        for i in 0..order {
            dm[i] = sm_state[t + 1][i] - pass.steps[t + 1].a_pred[i];
        }
        let corr = mat_vec(&g, &dm, order);
        for i in 0..order {
            sm_state[t][i] = pass.steps[t].a_filt[i] + corr[i];
        }
        let dp = mat_sub(&sm_cov[t + 1], p_next_pred, order);
        let mut cov = mat_add(
            &pass.steps[t].p_filt,
            &mat_mul(&mat_mul(&g, &dp, order), &mat_t(&g, order), order),
            order,
        );
        symmetrize(&mut cov, order);
        sm_cov[t] = cov;
        gains[t] = g;
    }
    // The order−1 partially-diffuse leading nodes by exact joint conditioning
    // (the multi-node generalization of the m=2 reverse-Markov closure).
    if order >= 2 {
        leading_block_smooth(&mut sm_state, &mut sm_cov, &mut gains, &nodes, q, order)?;
    }

    let knots: Vec<f64> = nodes.iter().map(|n| n.x).collect();
    let mean: Vec<f64> = sm_state.iter().map(|s| s[0]).collect();
    // f′ lives at state index 1 — present for order ≥ 2 only; the m = 1 latent
    // process (Brownian motion) has no derivative state to expose.
    let deriv: Option<Vec<f64>> = (order >= 2).then(|| sm_state.iter().map(|s| s[1]).collect());
    let var: Vec<f64> = sm_cov.iter().map(|p| p[0][0] * sigma2).collect();
    // Weighted DATA residual sum of squares at the smoothed mean. Tied rows
    // pool exactly: Σᵢ wᵢ(yᵢ − f̂ₖ)² = Σᵢ wᵢ(yᵢ − ȳₖ)² + Σₖ Wₖ(ȳₖ − f̂ₖ)²
    // (within-tie scatter plus pooled-node misfit), so the raw rows the scan
    // does not retain are not needed.
    let data_sse = ssr_within
        + nodes
            .iter()
            .zip(mean.iter())
            .map(|(node, &fhat)| {
                let r = node.y - fhat;
                node.w * r * r
            })
            .sum::<f64>();
    Ok(SplineScanFit {
        order,
        knots,
        mean,
        deriv,
        var,
        log_lambda,
        sigma2,
        restricted_loglik,
        n_obs,
        data_sse,
        smoothed_state: sm_state,
        smoothed_cov: sm_cov,
        rts_gain: gains,
        q,
        node_weight: nodes.iter().map(|n| n.w).collect(),
    })
}

/// Fit with `log λ` selected by the concentrated diffuse REML criterion.
/// Every stationary interval in the bounded, scale-equivariant log-λ domain
/// is isolated using analytic derivatives and rigorous interval bounds; the
/// two boundary/null-recovery candidates are evaluated exactly.
pub fn fit_spline_scan(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    order: usize,
) -> Result<SplineScanFit, String> {
    if order == 0 || order > MAX_ORDER {
        return Err(format!(
            "spline scan: order must be in 1..={MAX_ORDER}, got {order}"
        ));
    }
    let (nodes, ssr_within, n_obs) = pool_nodes(x, y, w, order)?;
    // Covariate-rescaling equivariance (#1214). The order-`m` IWP process noise
    // is `Q(δ) ∝ q · δ^{2m−1}`, so under an affine covariate rescale `x → a·x`
    // (all abscissa gaps `δ → a·δ`) the posterior `f(x)` is *exactly* invariant
    // iff the smoothing parameter co-transforms as `q → q / a^{2m−1}`, i.e.
    // `log λ → log λ + (2m−1)·log a` (λ = 1/q). The whole smoother — criterion,
    // fit, and the Gaussian-bridge `predict` — runs self-consistently in the raw
    // covariate units, so the *only* place covariate scale leaks in is this
    // outer `log λ` search: a fixed absolute bracket `[LOG_LAMBDA_LO,
    // LOG_LAMBDA_HI]` does not track the data span, so at small/large covariate
    // scale the equivariant optimum rails out of the bracket and the fit drifts.
    // Anchor the bracket to the data's own length scale: search `log λ` around
    // `(2m−1)·log L` where `L` is the abscissa span (which scales linearly with
    // the covariate), so the search is performed in scale-free units and the
    // selected `q · L^{2m−1}` — hence the posterior `f(x)` — is invariant.
    let span = nodes.last().map(|n| n.x).unwrap_or(0.0) - nodes.first().map(|n| n.x).unwrap_or(0.0);
    let scale_shift = if span.is_finite() && span > 0.0 {
        (2 * order - 1) as f64 * span.ln()
    } else {
        0.0
    };
    let lo_anchor = LOG_LAMBDA_LO + scale_shift;
    let hi_anchor = LOG_LAMBDA_HI + scale_shift;
    let search = maximize_score_1d(
        lo_anchor,
        hi_anchor,
        f64::EPSILON.sqrt(),
        |ll| {
            concentrated_criterion_jet(&nodes, ssr_within, n_obs, ll, order).map(
                |(value, derivative, curvature)| ScoreJet {
                    value,
                    derivative,
                    curvature,
                },
            )
        },
        |lo, hi| {
            concentrated_criterion_enclosure(
                &nodes,
                ssr_within,
                n_obs,
                lo,
                hi,
                order,
            )
        },
    )
    .map_err(|error| format!("spline scan: REML stationary isolation failed: {error}"))?;
    fit_spline_scan_at(x, y, w, search.optimum.x, None, order)
}

/// Lossless serializable snapshot of a [`SplineScanFit`] (#1034).
///
/// Carries exactly the smoother state the Gaussian-bridge `predict` replays:
/// pooled knots, smoothed `(f, f′, …, f^{(m−1)})` states (`m` per knot),
/// smoothed state covariances (unit-σ² scale, symmetric — stored as the
/// upper triangle row-major, `m(m+1)/2` per knot), RTS backward gains (full
/// `m×m` row-major — gains are NOT symmetric), pooled node weights, and the
/// three fit scalars. `q = e^{−log λ}` and the public `mean`/`deriv`/`var`
/// views are derived on restore rather than stored, so a snapshot cannot go
/// internally inconsistent. The layouts are order-derived; at the historical
/// cubic `m = 2` they are exactly the original `[f, f′]` / `[c00, c01, c11]` /
/// `[g00, g01, g10, g11]` triples, so pre-order-generality snapshots restore
/// unchanged.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SplineScanState {
    /// Smoothing-spline order `m ∈ {1, 2, 3}` (`#[serde(default)]` → reads as
    /// the historical cubic `m = 2` for snapshots written before order
    /// generality).
    #[serde(default = "default_spline_scan_order")]
    pub order: usize,
    pub knots: Vec<f64>,
    /// Smoothed `(f, f′, …, f^{(m−1)})` per knot, row-major (`m` per knot).
    pub state: Vec<f64>,
    /// Smoothed covariance per knot at unit-σ² scale, upper triangle row-major
    /// (`m(m+1)/2` per knot): `[c00, c01, …, c0,m−1, c11, …, c_{m−1,m−1}]`.
    pub cov: Vec<f64>,
    /// RTS backward gain per knot, full `m×m` row-major (`m²` per knot); the
    /// last knot's gain is structurally unused and stored as written.
    pub gain: Vec<f64>,
    /// Pooled (tied-abscissa summed) observation weight per knot.
    pub node_weight: Vec<f64>,
    pub log_lambda: f64,
    pub sigma2: f64,
    pub restricted_loglik: f64,
    /// Raw observation count `n` (#1046).
    pub n_obs: u64,
    /// Weighted data residual sum of squares `Σ wᵢ (yᵢ − f̂(xᵢ))²` at the
    /// smoothed mean — the Gaussian deviance. Stored because it cannot be
    /// recovered from the profiled σ² (whose quadratic also carries
    /// process/roughness energy) and the raw rows are not retained.
    pub data_sse: f64,
}

/// Serde default for [`SplineScanState::order`]: historical snapshots predate
/// order generality and are cubic (`m = 2`).
fn default_spline_scan_order() -> usize {
    2
}

impl SplineScanFit {
    /// Snapshot the full smoother state for persistence (#1034).
    pub fn to_state(&self) -> SplineScanState {
        let order = self.order;
        let tri = order * (order + 1) / 2;
        let nk = self.knots.len();
        let mut state = Vec::with_capacity(order * nk);
        for s in &self.smoothed_state {
            state.extend_from_slice(&s[..order]);
        }
        let mut cov = Vec::with_capacity(tri * nk);
        for c in &self.smoothed_cov {
            for i in 0..order {
                for j in i..order {
                    cov.push(c[i][j]);
                }
            }
        }
        let mut gain = Vec::with_capacity(order * order * nk);
        for g in &self.rts_gain {
            for i in 0..order {
                for j in 0..order {
                    gain.push(g[i][j]);
                }
            }
        }
        SplineScanState {
            order: self.order,
            knots: self.knots.clone(),
            state,
            cov,
            gain,
            node_weight: self.node_weight.clone(),
            log_lambda: self.log_lambda,
            sigma2: self.sigma2,
            restricted_loglik: self.restricted_loglik,
            n_obs: self.n_obs as u64,
            data_sse: self.data_sse,
        }
    }

    /// Rebuild the exact in-memory fit from a persisted snapshot (#1034).
    ///
    /// Validates shape, finiteness, strict knot ordering, positive weights and
    /// σ², so a corrupt payload fails loudly here instead of inside a later
    /// `predict`. The restored fit replays the Gaussian bridge bit-for-bit:
    /// every field `predict`/`edf`/`deriv_at_knot` reads is either stored
    /// verbatim or derived by the same expressions the fitter uses.
    pub fn from_state(state: &SplineScanState) -> Result<Self, String> {
        let order = state.order;
        if order == 0 || order > MAX_ORDER {
            return Err(format!(
                "spline scan state: order must be in 1..={MAX_ORDER}, got {order}"
            ));
        }
        let m = state.knots.len();
        if m < order + 1 {
            return Err(format!(
                "spline scan state: order {order} needs at least {} knots, got {m}",
                order + 1
            ));
        }
        let tri = order * (order + 1) / 2;
        if state.state.len() != order * m
            || state.cov.len() != tri * m
            || state.gain.len() != order * order * m
            || state.node_weight.len() != m
        {
            return Err(format!(
                "spline scan state: inconsistent lengths (order={order}, m={m}, state={}, cov={}, gain={}, weights={})",
                state.state.len(),
                state.cov.len(),
                state.gain.len(),
                state.node_weight.len()
            ));
        }
        let all = state
            .state
            .iter()
            .chain(&state.cov)
            .chain(&state.gain)
            .chain(&state.knots)
            .chain(&state.node_weight);
        for (i, v) in all.enumerate() {
            if !v.is_finite() {
                return Err(format!("spline scan state: non-finite entry at {i}"));
            }
        }
        if !(state.log_lambda.is_finite()
            && state.restricted_loglik.is_finite()
            && state.sigma2.is_finite()
            && state.sigma2 > 0.0)
        {
            return Err(format!(
                "spline scan state: invalid scalars (log_lambda={}, sigma2={}, restricted_loglik={})",
                state.log_lambda, state.sigma2, state.restricted_loglik
            ));
        }
        if !(state.data_sse.is_finite() && state.data_sse >= 0.0) {
            return Err(format!(
                "spline scan state: invalid data_sse {}",
                state.data_sse
            ));
        }
        if state.knots.windows(2).any(|kk| !(kk[0] < kk[1])) {
            return Err("spline scan state: knots must be strictly increasing".to_string());
        }
        if state.node_weight.iter().any(|&w| w <= 0.0) {
            return Err("spline scan state: node weights must be positive".to_string());
        }
        let smoothed_state: Vec<Vec2> = state
            .state
            .chunks_exact(order)
            .map(|s| {
                let mut v = [0.0_f64; MAX_ORDER];
                v[..order].copy_from_slice(s);
                v
            })
            .collect();
        let smoothed_cov: Vec<Mat2> = state
            .cov
            .chunks_exact(tri)
            .map(|c| {
                let mut mm = [[0.0_f64; MAX_ORDER]; MAX_ORDER];
                let mut idx = 0;
                for i in 0..order {
                    for j in i..order {
                        mm[i][j] = c[idx];
                        mm[j][i] = c[idx];
                        idx += 1;
                    }
                }
                mm
            })
            .collect();
        let rts_gain: Vec<Mat2> = state
            .gain
            .chunks_exact(order * order)
            .map(|g| {
                let mut mm = [[0.0_f64; MAX_ORDER]; MAX_ORDER];
                for i in 0..order {
                    for j in 0..order {
                        mm[i][j] = g[i * order + j];
                    }
                }
                mm
            })
            .collect();
        let sigma2 = state.sigma2;
        if state.n_obs == 0 {
            return Err("spline scan state: n_obs must be positive".to_string());
        }
        let n_obs = state.n_obs as usize;
        Ok(Self {
            order,
            knots: state.knots.clone(),
            mean: smoothed_state.iter().map(|s| s[0]).collect(),
            deriv: (order >= 2).then(|| smoothed_state.iter().map(|s| s[1]).collect()),
            var: smoothed_cov.iter().map(|c| c[0][0] * sigma2).collect(),
            log_lambda: state.log_lambda,
            sigma2,
            restricted_loglik: state.restricted_loglik,
            n_obs,
            data_sse: state.data_sse,
            smoothed_state,
            smoothed_cov,
            rts_gain,
            q: (-state.log_lambda).exp(),
            node_weight: state.node_weight.clone(),
        })
    }

    /// Exact posterior `(mean, variance)` of `f` at an arbitrary abscissa.
    ///
    /// Interior points use the Gaussian bridge conditional on the two flanking
    /// smoothed states with the exact lag-one smoothed cross-covariance
    /// `Cov(α_t, α_{t+1} | y) = G_t · P^s_{t+1}`; exterior points extrapolate
    /// from the boundary state (linear mean, cubically growing variance).
    pub fn predict(&self, x_new: f64) -> Result<(f64, f64), String> {
        if !x_new.is_finite() {
            return Err("spline scan: non-finite prediction abscissa".to_string());
        }
        let n = self.knots.len();
        let order = self.order;
        let first = self.knots[0];
        let last = self.knots[n - 1];
        if x_new <= first {
            let delta = first - x_new;
            // Backward extrapolation through the reverse map α(x) = F⁻¹(α₁ − η).
            let f_t = transition(delta, order);
            let f_inv = mat_inv(&f_t, order, "backward extrapolation transition")?;
            let mean_s = mat_vec(&f_inv, &self.smoothed_state[0], order);
            let qm = process_noise(delta, self.q, order);
            let cov = mat_add(
                &mat_mul(
                    &mat_mul(&f_inv, &self.smoothed_cov[0], order),
                    &mat_t(&f_inv, order),
                    order,
                ),
                &mat_mul(&mat_mul(&f_inv, &qm, order), &mat_t(&f_inv, order), order),
                order,
            );
            return Ok((mean_s[0], cov[0][0] * self.sigma2));
        }
        if x_new >= last {
            let delta = x_new - last;
            let f_t = transition(delta, order);
            let mean_s = mat_vec(&f_t, &self.smoothed_state[n - 1], order);
            let cov = mat_add(
                &mat_mul(
                    &mat_mul(&f_t, &self.smoothed_cov[n - 1], order),
                    &mat_t(&f_t, order),
                    order,
                ),
                &process_noise(delta, self.q, order),
                order,
            );
            return Ok((mean_s[0], cov[0][0] * self.sigma2));
        }
        // Flanking knot interval via binary search.
        let t = match self.knots.binary_search_by(|k| k.total_cmp(&x_new)) {
            Ok(idx) => return Ok((self.mean[idx], self.var[idx])),
            Err(idx) => idx - 1,
        };
        let (xa, xb) = (self.knots[t], self.knots[t + 1]);
        let (d1, d2) = (x_new - xa, xb - x_new);
        let (f1m, f2m) = (transition(d1, order), transition(d2, order));
        let (q1, q2) = (
            process_noise(d1, self.q, order),
            process_noise(d2, self.q, order),
        );
        let q1_inv = mat_inv(&q1, order, "bridge left noise")?;
        let q2_inv = mat_inv(&q2, order, "bridge right noise")?;
        // p(α* | α_t, α_{t+1}) ∝ N(α*; F₁α_t, Q₁)·N(α_{t+1}; F₂α*, Q₂):
        //   Λ = Q₁⁻¹ + F₂ᵀQ₂⁻¹F₂,  mean = Λ⁻¹(Q₁⁻¹F₁ α_t + F₂ᵀQ₂⁻¹ α_{t+1}).
        let lambda = mat_add(
            &q1_inv,
            &mat_mul(&mat_mul(&mat_t(&f2m, order), &q2_inv, order), &f2m, order),
            order,
        );
        let lam_inv = mat_inv(&lambda, order, "bridge precision")?;
        let ca = mat_mul(&lam_inv, &mat_mul(&q1_inv, &f1m, order), order);
        let cb = mat_mul(
            &lam_inv,
            &mat_mul(&mat_t(&f2m, order), &q2_inv, order),
            order,
        );
        let ma = mat_vec(&ca, &self.smoothed_state[t], order);
        let mb = mat_vec(&cb, &self.smoothed_state[t + 1], order);
        let mut mean_s = [0.0_f64; MAX_ORDER];
        for i in 0..order {
            mean_s[i] = ma[i] + mb[i];
        }
        // Push the joint smoothed covariance of (α_t, α_{t+1}) through the
        // affine map: cross term uses Cov(α_t, α_{t+1}|y) = G_t · P^s_{t+1}.
        let cross = mat_mul(&self.rts_gain[t], &self.smoothed_cov[t + 1], order);
        let mut cov = mat_add(
            &mat_add(
                &mat_mul(
                    &mat_mul(&ca, &self.smoothed_cov[t], order),
                    &mat_t(&ca, order),
                    order,
                ),
                &mat_mul(
                    &mat_mul(&cb, &self.smoothed_cov[t + 1], order),
                    &mat_t(&cb, order),
                    order,
                ),
                order,
            ),
            &lam_inv,
            order,
        );
        let cab = mat_mul(&mat_mul(&ca, &cross, order), &mat_t(&cb, order), order);
        cov = mat_add(&cov, &mat_add(&cab, &mat_t(&cab, order), order), order);
        symmetrize(&mut cov, order);
        Ok((mean_s[0], cov[0][0] * self.sigma2))
    }

    /// Exact effective degrees of freedom of the fitted smoother.
    ///
    /// For a Gaussian smoother the influence (hat) matrix is
    /// `S = Cov_post · W / σ²` (posterior mean is linear in `y` with that
    /// exact coefficient matrix), so
    /// `EDF = tr(S) = tr(W · Cov_post) / σ² = Σ_t w_t · Var_smoothed(f_t) / σ²`.
    /// This is the standard Gaussian-process identity — no second smoother
    /// pass and no approximation. Tied abscissae pool exactly: each raw row
    /// `i` in tie-group `k` contributes `∂f̂(x_k)/∂y_i = C̃_kk · w_i` (the
    /// pooled mean `ȳ_k` is precision-weighted), so the raw-row trace
    /// `Σ_i w_i · C̃_{k(i),k(i)}` collapses to `Σ_k W_k · C̃_kk` with the
    /// pooled weights `W_k`. `smoothed_cov` is stored at unit-σ² scale
    /// (`C̃ = Cov_post / σ²`), so the σ² factors cancel exactly.
    pub fn edf(&self) -> f64 {
        self.node_weight
            .iter()
            .zip(self.smoothed_cov.iter())
            .map(|(w, c)| w * c[0][0])
            .sum()
    }

    /// Posterior `(mean, variance)` of the derivative `f′` at a knot index.
    ///
    /// `None` at order `m = 1`: the latent process is Brownian motion, which
    /// is almost surely nondifferentiable — there is no derivative state, and
    /// fabricating a "known zero" `(0, 0)` would assert certainty about a
    /// quantity that does not exist.
    pub fn deriv_at_knot(&self, t: usize) -> Option<(f64, f64)> {
        (self.order >= 2).then(|| {
            (
                self.smoothed_state[t][1],
                self.smoothed_cov[t][1][1] * self.sigma2,
            )
        })
    }

    /// Selected smoothing parameter `λ = e^{log λ}` (#1046).
    pub fn lambda(&self) -> f64 {
        self.log_lambda.exp()
    }

    /// Raw observation count `n` used to profile σ² (#1046).
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Gaussian deviance — the weighted DATA residual sum of squares
    /// `Σ wᵢ(yᵢ − f̂ᵢ)²` at the smoothed mean (#1046). This is the stored
    /// `data_sse`, computed against the fitted values at fit time. It is NOT
    /// `σ̂²·(n − order)`: the profiled σ² divides the REML innovations
    /// quadratic, which is data residual energy PLUS process/roughness energy
    /// at the posterior mode (for order 1 on `x = (0,1)`, `y = (0,1)`, unit
    /// weights and λ = 1 the posterior mean is `(1/3, 2/3)`; the data SSE is
    /// 2/9 while `σ̂²·(n − order) = 1/3`, the extra 1/9 being penalty energy).
    pub fn deviance(&self) -> f64 {
        self.data_sse
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concentrated_score_jet_matches_test_only_differences() {
        let x = [0.0, 0.07, 0.19, 0.41, 0.41, 0.68, 1.0, 1.37];
        let y = [0.2, -0.4, 0.8, 0.1, 0.35, -0.2, 0.7, 0.15];
        let w = [1.0, 2.0, 0.7, 1.4, 0.9, 3.0, 1.2, 0.8];
        for order in 1..=MAX_ORDER {
            let (nodes, within, n_obs) = pool_nodes(&x, &y, &w, order).expect("pooled data");
            for &rho in &[-4.0, -0.3, 2.5] {
                let (value, d1, d2) = concentrated_criterion_jet(&nodes, within, n_obs, rho, order)
                    .expect("analytic score jet");
                // Finite differences are deliberately confined to this oracle
                // test; production selection uses the analytic sensitivities.
                let h = 2.0e-4;
                let fm = concentrated_criterion(&nodes, within, n_obs, rho - h, order)
                    .expect("left score");
                let fp = concentrated_criterion(&nodes, within, n_obs, rho + h, order)
                    .expect("right score");
                let d1_fd = (fp - fm) / (2.0 * h);
                let d2_fd = (fp - 2.0 * value + fm) / (h * h);
                let d1_scale = 1.0 + d1.abs().max(d1_fd.abs());
                let d2_scale = 1.0 + d2.abs().max(d2_fd.abs());
                assert!(
                    (d1 - d1_fd).abs() <= 2.0e-6 * d1_scale,
                    "order={order} rho={rho}: analytic d1={d1}, FD={d1_fd}"
                );
                assert!(
                    (d2 - d2_fd).abs() <= 2.0e-4 * d2_scale,
                    "order={order} rho={rho}: analytic d2={d2}, FD={d2_fd}"
                );
            }
        }
    }

    /// #1034 persistence seam: snapshot → JSON → restore must replay the
    /// Gaussian bridge bit-for-bit — knot posteriors, off-knot bridge,
    /// boundary extrapolation, EDF, and derivative posteriors all compare
    /// with exact equality, because every replayed field is either stored
    /// verbatim or derived by the fitter's own expressions. Parameterized over
    /// the smoothing order so the order-derived state/cov/gain layouts
    /// (#1044: m=3 stores 3-wide state, 6-wide upper-tri cov, 9-wide gain) are
    /// each round-tripped.
    fn round_trip_predict_bit_for_bit(order: usize) {
        let n = 60usize;
        let x: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64 - 1.0)).collect();
        // Deterministic wiggly response with a tie pair to exercise pooling.
        let mut x = x;
        x[7] = x[6];
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                (6.0 * xi).sin() + 0.3 * (17.0 * xi).cos() + 0.05 * ((i * 37 % 11) as f64 - 5.0)
            })
            .collect();
        let w: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * ((i % 3) as f64)).collect();
        let fit = fit_spline_scan(&x, &y, &w, order).expect("scan fit");
        assert_eq!(fit.order, order);
        // The raw count is retained verbatim (n rows, one tie pair collapses a
        // knot but not the count) and drives the recovered deviance (#1046).
        assert_eq!(fit.n_obs, n);

        let json = serde_json::to_string(&fit.to_state()).expect("serialize state");
        let state: SplineScanState = serde_json::from_str(&json).expect("deserialize state");
        let restored = SplineScanFit::from_state(&state).expect("restore fit");

        assert_eq!(fit.n_obs, restored.n_obs);
        assert_eq!(fit.deviance().to_bits(), restored.deviance().to_bits());
        assert_eq!(fit.knots, restored.knots);
        assert_eq!(fit.mean, restored.mean);
        assert_eq!(fit.var, restored.var);
        assert_eq!(fit.deriv, restored.deriv);
        assert_eq!(fit.log_lambda.to_bits(), restored.log_lambda.to_bits());
        assert_eq!(fit.sigma2.to_bits(), restored.sigma2.to_bits());
        assert_eq!(fit.edf().to_bits(), restored.edf().to_bits());
        for t in 0..fit.knots.len() {
            match (fit.deriv_at_knot(t), restored.deriv_at_knot(t)) {
                (Some((d0, v0)), Some((d1, v1))) => {
                    assert!(order >= 2);
                    assert_eq!(d0.to_bits(), d1.to_bits());
                    assert_eq!(v0.to_bits(), v1.to_bits());
                }
                (None, None) => assert_eq!(order, 1),
                _ => panic!("derivative availability drifted across the persistence seam"),
            }
        }
        // Off-knot bridge, exact knot hit, and both extrapolation sides.
        for &xq in &[-0.2, 0.0, 0.013, 0.5, x[6], 0.987, 1.0, 1.3] {
            let (m0, v0) = fit.predict(xq).expect("predict original");
            let (m1, v1) = restored.predict(xq).expect("predict restored");
            assert_eq!(
                m0.to_bits(),
                m1.to_bits(),
                "mean drift at x={xq} (m={order})"
            );
            assert_eq!(
                v0.to_bits(),
                v1.to_bits(),
                "variance drift at x={xq} (m={order})"
            );
        }

        // Corrupt payloads fail loudly, not inside a later predict.
        let mut bad = fit.to_state();
        bad.cov.truncate(bad.cov.len() - 1);
        SplineScanFit::from_state(&bad).expect_err("length mismatch must error");
        let mut bad = fit.to_state();
        bad.sigma2 = -1.0;
        SplineScanFit::from_state(&bad).expect_err("non-positive sigma2 must error");
        let mut bad = fit.to_state();
        bad.knots[2] = bad.knots[1];
        SplineScanFit::from_state(&bad).expect_err("non-increasing knots must error");
    }

    #[test]
    fn state_snapshot_round_trips_predict_bit_for_bit() {
        round_trip_predict_bit_for_bit(2);
    }

    /// #1044: the order-1 and order-3 layouts round-trip bit-for-bit too.
    #[test]
    fn state_snapshot_round_trips_predict_bit_for_bit_order1() {
        round_trip_predict_bit_for_bit(1);
    }

    #[test]
    fn state_snapshot_round_trips_predict_bit_for_bit_order3() {
        round_trip_predict_bit_for_bit(3);
    }

    /// Dense order-1 (random-walk / linear smoothing spline) posterior of the
    /// SAME intrinsic prior the order-1 scan integrates: improper level on
    /// `f_0`, increments `f_{t+1}−f_t ~ N(0, q·δ_t)`, observations `y_t` with
    /// precision `w_t` (unit σ²). Solve the tridiagonal precision densely and
    /// compare to the scan — the exact-equivalence gate for the new m=1 path.
    fn dense_rw_truth(x: &[f64], y: &[f64], w: &[f64], log_lambda: f64) -> (Vec<f64>, Vec<f64>) {
        let n = x.len();
        let q = (-log_lambda).exp();
        let mut prec = vec![vec![0.0_f64; n]; n];
        let mut rhs = vec![0.0_f64; n];
        for t in 0..n {
            prec[t][t] += w[t];
            rhs[t] += w[t] * y[t];
        }
        for t in 0..n - 1 {
            let p = 1.0 / (q * (x[t + 1] - x[t]));
            prec[t][t] += p;
            prec[t + 1][t + 1] += p;
            prec[t][t + 1] -= p;
            prec[t + 1][t] -= p;
        }
        // Dense inverse via Gauss-Jordan (small n in the test).
        let mut aug = prec.clone();
        let mut inv = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            inv[i][i] = 1.0;
        }
        for col in 0..n {
            let piv = (col..n)
                .max_by(|&a, &b| aug[a][col].abs().total_cmp(&aug[b][col].abs()))
                .unwrap();
            aug.swap(col, piv);
            inv.swap(col, piv);
            let d = aug[col][col];
            for k in 0..n {
                aug[col][k] /= d;
                inv[col][k] /= d;
            }
            for r in 0..n {
                if r == col {
                    continue;
                }
                let f = aug[r][col];
                if f == 0.0 {
                    continue;
                }
                for k in 0..n {
                    aug[r][k] -= f * aug[col][k];
                    inv[r][k] -= f * inv[col][k];
                }
            }
        }
        let mean: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| inv[i][j] * rhs[j]).sum())
            .collect();
        let var: Vec<f64> = (0..n).map(|i| inv[i][i]).collect();
        (mean, var)
    }

    /// The order-1 scan must reproduce the dense random-walk posterior exactly
    /// (mean, pointwise variance, and the EDF identity tr(S)=Σ w_t·Var_t/σ²) at
    /// the scan's own selected λ — the #1034-item-2 correctness gate.
    #[test]
    fn order_one_scan_matches_dense_random_walk_posterior() {
        let n = 30usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| 2.0 * xi + 0.4 * (5.0 * xi).sin() + 0.05 * ((i * 13 % 7) as f64 - 3.0))
            .collect();
        let w = vec![1.0_f64; n];
        let fit = fit_spline_scan(&x, &y, &w, 1).expect("order-1 scan fit");
        assert_eq!(fit.order, 1);

        let (mean, var) = dense_rw_truth(&x, &y, &w, fit.log_lambda);
        for t in 0..n {
            assert!(
                (fit.mean[t] - mean[t]).abs() <= 1e-7 * mean[t].abs().max(1e-3),
                "order-1 mean mismatch at {t}: scan={} dense={}",
                fit.mean[t],
                mean[t]
            );
            let se_scan = fit.var[t].sqrt();
            let se_dense = (var[t] * fit.sigma2).sqrt();
            assert!(
                (se_scan - se_dense).abs() <= 1e-7 * se_dense.max(1e-12),
                "order-1 SE mismatch at {t}: scan={se_scan} dense={se_dense}"
            );
        }
        // EDF identity against the dense posterior variance diagonal.
        let dense_edf: f64 = w.iter().zip(var.iter()).map(|(wt, vt)| wt * vt).sum();
        assert!(
            (fit.edf() - dense_edf).abs() <= 1e-7 * dense_edf.max(1e-12),
            "order-1 EDF mismatch: scan={} dense={dense_edf}",
            fit.edf()
        );
        // Order-1 derivative state is structurally absent: Brownian motion has
        // no pointwise derivative, so the fit must say so rather than report a
        // fabricated known-zero.
        assert!(fit.deriv.is_none());
        assert!(fit.deriv_at_knot(0).is_none());
    }

    /// `deviance()` must be the weighted DATA residual sum of squares at the
    /// fitted values, not the profiled REML quadratic. For order 1 on
    /// `x = (0, 1)`, `y = (0, 1)`, unit weights, λ = 1, the posterior mean is
    /// `(1/3, 2/3)`: the data SSE is `2·(1/3)² = 2/9`, while
    /// `σ̂²·(n − order) = 1/3` carries an extra `1/9` of process/roughness
    /// energy.
    #[test]
    fn deviance_is_data_sse_not_penalized_quadratic() {
        let x = [0.0, 1.0];
        let y = [0.0, 1.0];
        let w = [1.0, 1.0];
        let fit = fit_spline_scan_at(&x, &y, &w, 0.0, None, 1).expect("order-1 fit");
        // Self-consistency against a direct recomputation at the fitted values.
        let manual: f64 = x
            .iter()
            .zip(&y)
            .zip(&w)
            .map(|((&xi, &yi), &wi)| {
                let (m, _) = fit.predict(xi).expect("predict at knot");
                wi * (yi - m) * (yi - m)
            })
            .sum();
        assert!(
            (fit.deviance() - manual).abs() <= 1e-12 * manual.max(1e-300),
            "deviance {} != recomputed data SSE {manual}",
            fit.deviance()
        );
        assert!(
            (fit.deviance() - 2.0 / 9.0).abs() < 1e-10,
            "deviance {} != 2/9",
            fit.deviance()
        );
        // The old proxy is strictly larger: it includes penalty energy.
        let reml_quadratic = fit.sigma2 * (fit.n_obs as f64 - fit.order as f64);
        assert!(fit.deviance() < reml_quadratic);
    }
}

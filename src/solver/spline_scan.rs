//! Exact O(n) state-space cubic smoothing spline ("the scan").
//!
//! The order-2 intrinsic Gaussian prior whose penalized posterior mean is the
//! cubic smoothing spline (penalty `λ∫f″²`) is a Markov process in the state
//! `α(x) = (f(x), f′(x))`: an integrated Wiener process. The Kalman filter +
//! RTS smoother over the x-sorted observations therefore computes the EXACT
//! smoothing-spline posterior — mean, derivative, pointwise variance — and the
//! diffuse innovations decomposition computes the EXACT restricted (REML)
//! likelihood, all in O(n) work per smoothing-parameter trial instead of the
//! dense O(n·k²) design/Gram + O(k³) solve per trial (Wahba 1978;
//! Kohn & Ansley 1987; Durbin & Koopman exact diffuse initialization).
//!
//! Model, after sorting and pooling tied abscissae (precision-weighted):
//!   α_{t+1} = F_t α_t + η_t,   η_t ~ N(0, q·Q(δ_t)),   q = σ_w²/σ² = 1/λ,
//!   y_t     = H α_t + ε_t,     ε_t ~ N(0, σ²/w_t),     H = [1 0],
//!   F(δ) = [[1, δ], [0, 1]],   Q(δ) = [[δ³/3, δ²/2], [δ²/2, δ]],
//! with a diffuse (improper, flat) prior on α_1 carrying the unpenalized
//! linear null space — the same null space the spline leaves unshrunk.
//!
//! Exactness boundaries, by construction:
//! - the diffuse dimension is 2 and is consumed by the first two distinct
//!   abscissae (F_∞ = P_∞[0,0] = 1 > 0 at t=1, then δ² > 0 at t=2), after
//!   which the filter is an ordinary proper Kalman filter;
//! - the t=1 smoothed moments are recovered by direct Markov conditioning
//!   `p(α_1 | y) = ∫ p(α_1 | α_2, y_1) p(α_2 | y)` (an affine 2×2 Bayes
//!   update — no diffuse RTS recursion is needed);
//! - off-knot prediction is the Gaussian bridge conditional on the two
//!   flanking smoothed states (using the exact lag-one smoothed
//!   cross-covariance `G_t · P^s_{t+1}`), or boundary extrapolation from the
//!   end states, which reproduces the spline's linear extrapolation with
//!   cubically growing variance — bridge-don't-sag is a theorem here.
//!
//! The smoothing parameter is selected by maximizing the concentrated diffuse
//! (restricted) log-likelihood over log λ with a deterministic coarse-grid +
//! golden-section refinement; σ² is profiled in closed form from the proper
//! innovations plus the within-tie residual sum.

/// One pooled (distinct-abscissa) observation node.
#[derive(Clone, Copy, Debug)]
struct PooledNode {
    x: f64,
    /// Precision-weighted mean of the tied responses.
    y: f64,
    /// Total weight of the pooled ties (observation variance is `σ²/w`).
    w: f64,
}

/// Deterministic coarse-grid width for the log-λ search.
const LOG_LAMBDA_GRID: usize = 25;
/// Search interval for log λ (natural log), generous on both sides.
const LOG_LAMBDA_LO: f64 = -18.0;
const LOG_LAMBDA_HI: f64 = 18.0;
/// Golden-section refinement tolerance on log λ.
const LOG_LAMBDA_TOL: f64 = 1e-7;
/// Numerical floor treating a predicted innovation variance as singular.
const INNOVATION_VAR_FLOOR: f64 = 1e-300;

type Mat2 = [[f64; 2]; 2];
type Vec2 = [f64; 2];

#[inline]
fn mat_mul(a: &Mat2, b: &Mat2) -> Mat2 {
    let mut c = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    c
}

#[inline]
fn mat_t(a: &Mat2) -> Mat2 {
    [[a[0][0], a[1][0]], [a[0][1], a[1][1]]]
}

#[inline]
fn mat_vec(a: &Mat2, v: &Vec2) -> Vec2 {
    [
        a[0][0] * v[0] + a[0][1] * v[1],
        a[1][0] * v[0] + a[1][1] * v[1],
    ]
}

#[inline]
fn mat_add(a: &Mat2, b: &Mat2) -> Mat2 {
    [
        [a[0][0] + b[0][0], a[0][1] + b[0][1]],
        [a[1][0] + b[1][0], a[1][1] + b[1][1]],
    ]
}

#[inline]
fn mat_sub(a: &Mat2, b: &Mat2) -> Mat2 {
    [
        [a[0][0] - b[0][0], a[0][1] - b[0][1]],
        [a[1][0] - b[1][0], a[1][1] - b[1][1]],
    ]
}

/// Inverse of a symmetric 2×2 with a hard singularity error.
fn mat_inv(a: &Mat2, what: &str) -> Result<Mat2, String> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if !(det.is_finite() && det.abs() > 0.0) {
        return Err(format!("spline scan: singular 2x2 in {what} (det={det})"));
    }
    Ok([
        [a[1][1] / det, -a[0][1] / det],
        [-a[1][0] / det, a[0][0] / det],
    ])
}

#[inline]
fn transition(delta: f64) -> Mat2 {
    [[1.0, delta], [0.0, 1.0]]
}

#[inline]
fn process_noise(delta: f64, q: f64) -> Mat2 {
    let d2 = delta * delta;
    let d3 = d2 * delta;
    [[q * d3 / 3.0, q * d2 / 2.0], [q * d2 / 2.0, q * delta]]
}

/// Symmetrize in place against drift from the rank-one update arithmetic.
#[inline]
fn symmetrize(a: &mut Mat2) {
    let off = 0.5 * (a[0][1] + a[1][0]);
    a[0][1] = off;
    a[1][0] = off;
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
    /// Σ over proper steps of `v_t² / F̃_t`.
    sum_v2_over_f: f64,
    /// Number of proper (non-diffuse) innovations.
    n_proper: usize,
}

fn run_filter(nodes: &[PooledNode], q: f64) -> Result<FilterPass, String> {
    let m = nodes.len();
    let mut steps = Vec::with_capacity(m);
    // Exact diffuse initialization (Durbin–Koopman): P = P* + κ·P_∞, κ → ∞.
    let mut a: Vec2 = [0.0, 0.0];
    let mut p_star: Mat2 = [[0.0; 2]; 2];
    let mut p_inf: Mat2 = [[1.0, 0.0], [0.0, 1.0]];
    let mut diffuse_rank = 2usize;
    let mut sum_log_f = 0.0;
    let mut sum_v2_over_f = 0.0;
    let mut n_proper = 0usize;
    for t in 0..m {
        let a_pred = a;
        let p_pred = p_star;
        let r = 1.0 / nodes[t].w;
        let v = nodes[t].y - a[0];
        // H = [1 0] ⇒ M = P·H' is the first column, F = M[0] (+ r).
        let m_star: Vec2 = [p_star[0][0], p_star[1][0]];
        let f_star = m_star[0] + r;
        if diffuse_rank > 0 {
            let m_inf: Vec2 = [p_inf[0][0], p_inf[1][0]];
            let f_inf = m_inf[0];
            if f_inf > INNOVATION_VAR_FLOOR {
                // Exact diffuse update (Koopman 1997): the κ→∞ limit of the
                // standard update; the diffuse step contributes −½·log F_∞ to
                // the restricted likelihood and consumes one diffuse dimension.
                let k0: Vec2 = [m_inf[0] / f_inf, m_inf[1] / f_inf];
                a = [a[0] + k0[0] * v, a[1] + k0[1] * v];
                let mut p_new = p_star;
                for i in 0..2 {
                    for j in 0..2 {
                        p_new[i][j] += -m_inf[i] * m_star[j] / f_inf - m_star[i] * m_inf[j] / f_inf
                            + m_inf[i] * m_inf[j] * f_star / (f_inf * f_inf);
                    }
                }
                p_star = p_new;
                symmetrize(&mut p_star);
                for i in 0..2 {
                    for j in 0..2 {
                        p_inf[i][j] -= m_inf[i] * m_inf[j] / f_inf;
                    }
                }
                symmetrize(&mut p_inf);
                diffuse_rank -= 1;
                if diffuse_rank == 0 {
                    p_inf = [[0.0; 2]; 2];
                }
            } else {
                // Diffuse direction orthogonal to H at this node: ordinary
                // proper update with P* (F_∞ = 0 ⇒ standard Kalman step).
                if f_star <= INNOVATION_VAR_FLOOR {
                    return Err("spline scan: non-positive innovation variance".to_string());
                }
                let k: Vec2 = [m_star[0] / f_star, m_star[1] / f_star];
                a = [a[0] + k[0] * v, a[1] + k[1] * v];
                for i in 0..2 {
                    for j in 0..2 {
                        p_star[i][j] -= m_star[i] * m_star[j] / f_star;
                    }
                }
                symmetrize(&mut p_star);
                sum_log_f += f_star.ln();
                sum_v2_over_f += v * v / f_star;
                n_proper += 1;
            }
        } else {
            if f_star <= INNOVATION_VAR_FLOOR {
                return Err("spline scan: non-positive innovation variance".to_string());
            }
            let k: Vec2 = [m_star[0] / f_star, m_star[1] / f_star];
            a = [a[0] + k[0] * v, a[1] + k[1] * v];
            for i in 0..2 {
                for j in 0..2 {
                    p_star[i][j] -= m_star[i] * m_star[j] / f_star;
                }
            }
            symmetrize(&mut p_star);
            sum_log_f += f_star.ln();
            sum_v2_over_f += v * v / f_star;
            n_proper += 1;
        }
        steps.push(FilterStep {
            a_filt: a,
            p_filt: p_star,
            a_pred,
            p_pred,
        });
        // Predict to the next node.
        if t + 1 < m {
            let delta = nodes[t + 1].x - nodes[t].x;
            let f_t = transition(delta);
            a = mat_vec(&f_t, &a);
            let mut p_next = mat_add(
                &mat_mul(&mat_mul(&f_t, &p_star), &mat_t(&f_t)),
                &process_noise(delta, q),
            );
            symmetrize(&mut p_next);
            p_star = p_next;
            if diffuse_rank > 0 {
                let mut pi_next = mat_mul(&mat_mul(&f_t, &p_inf), &mat_t(&f_t));
                symmetrize(&mut pi_next);
                p_inf = pi_next;
            }
        }
    }
    Ok(FilterPass {
        steps,
        sum_log_f,
        sum_v2_over_f,
        n_proper,
    })
}

/// Fitted exact smoothing-spline posterior on the pooled knots.
#[derive(Clone, Debug)]
pub struct CubicSplineScanFit {
    /// Distinct sorted abscissae (pooled knots).
    pub knots: Vec<f64>,
    /// Smoothed posterior mean of `f` at each knot.
    pub mean: Vec<f64>,
    /// Smoothed posterior mean of `f′` at each knot.
    pub deriv: Vec<f64>,
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
fn pool_nodes(x: &[f64], y: &[f64], w: &[f64]) -> Result<(Vec<PooledNode>, f64, usize), String> {
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
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| x[i].total_cmp(&x[j]));
    let mut nodes: Vec<PooledNode> = Vec::new();
    for &i in &order {
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
    if nodes.len() < 3 {
        return Err(format!(
            "spline scan: needs at least 3 distinct abscissae, got {}",
            nodes.len()
        ));
    }
    // Within-tie residual sum Σ w_i (y_i − ȳ_group)², part of the profiled σ².
    let mut ssr_within = 0.0;
    let mut k = 0usize;
    for &i in &order {
        while nodes[k].x != x[i] {
            k += 1;
        }
        let d = y[i] - nodes[k].y;
        ssr_within += w[i] * d * d;
    }
    Ok((nodes, ssr_within, n))
}

/// Concentrated diffuse restricted log-likelihood at `log λ` (σ² profiled).
fn concentrated_criterion(
    nodes: &[PooledNode],
    ssr_within: f64,
    n_obs: usize,
    log_lambda: f64,
) -> Result<f64, String> {
    let pass = run_filter(nodes, (-log_lambda).exp())?;
    // Profiled σ̂² over the proper innovations plus within-tie residuals;
    // the restricted degrees of freedom subtract the diffuse dimension 2.
    let dof = (n_obs - 2) as f64;
    let rss = pass.sum_v2_over_f + ssr_within;
    if rss <= 0.0 {
        return Err("spline scan: degenerate zero residual sum".to_string());
    }
    let sigma2 = rss / dof;
    if pass.n_proper != nodes.len() - 2 {
        return Err(format!(
            "spline scan: expected {} proper innovations, got {} (diffuse rank not consumed)",
            nodes.len() - 2,
            pass.n_proper
        ));
    }
    Ok(-0.5 * (pass.sum_log_f + dof * sigma2.ln()))
}

/// Fit at a FIXED `log λ`, with σ² either supplied or profiled.
pub fn fit_cubic_spline_scan_at(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    log_lambda: f64,
    sigma2: Option<f64>,
) -> Result<CubicSplineScanFit, String> {
    let (nodes, ssr_within, n_obs) = pool_nodes(x, y, w)?;
    let q = (-log_lambda).exp();
    let pass = run_filter(&nodes, q)?;
    let m = nodes.len();
    let dof = (n_obs - 2) as f64;
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

    // ── RTS smoother (proper steps; t = 0 handled by direct conditioning) ──
    let mut sm_state = vec![[0.0_f64; 2]; m];
    let mut sm_cov = vec![[[0.0_f64; 2]; 2]; m];
    let mut gains = vec![[[0.0_f64; 2]; 2]; m];
    sm_state[m - 1] = pass.steps[m - 1].a_filt;
    sm_cov[m - 1] = pass.steps[m - 1].p_filt;
    for t in (1..m - 1).rev() {
        let p_next_pred = &pass.steps[t + 1].p_pred;
        let delta = nodes[t + 1].x - nodes[t].x;
        let f_t = transition(delta);
        let p_inv = mat_inv(p_next_pred, "RTS predicted covariance")?;
        let g = mat_mul(&mat_mul(&pass.steps[t].p_filt, &mat_t(&f_t)), &p_inv);
        let dm: Vec2 = [
            sm_state[t + 1][0] - pass.steps[t + 1].a_pred[0],
            sm_state[t + 1][1] - pass.steps[t + 1].a_pred[1],
        ];
        let corr = mat_vec(&g, &dm);
        sm_state[t] = [
            pass.steps[t].a_filt[0] + corr[0],
            pass.steps[t].a_filt[1] + corr[1],
        ];
        let dp = mat_sub(&sm_cov[t + 1], p_next_pred);
        let mut cov = mat_add(
            &pass.steps[t].p_filt,
            &mat_mul(&mat_mul(&g, &dp), &mat_t(&g)),
        );
        symmetrize(&mut cov);
        sm_cov[t] = cov;
        gains[t] = g;
    }
    // t = 0 by exact Markov conditioning: p(α₁ | y) = ∫ p(α₁ | α₂, y₁) p(α₂ | y).
    // Reverse map α₁ = F⁻¹(α₂ − η) gives the proper "prior" N(F⁻¹α₂, F⁻¹QF⁻ᵀ);
    // one 2×2 Bayes update with y₁ makes the conditional affine in α₂, and the
    // smoothed moments of α₂ push through the affine map exactly. This sidesteps
    // the diffuse RTS recursion entirely (only t=0 retains diffuse filtered cov).
    {
        let delta = nodes[1].x - nodes[0].x;
        let f1 = transition(delta);
        let f1_inv = mat_inv(&f1, "first transition")?;
        let q1 = process_noise(delta, q);
        let rev_cov = mat_mul(&mat_mul(&f1_inv, &q1), &mat_t(&f1_inv));
        let rev_prec = mat_inv(&rev_cov, "reverse-map covariance")?;
        let r0 = 1.0 / nodes[0].w;
        // Posterior precision Λ = rev_prec + H'H/r₀.
        let mut lambda = rev_prec;
        lambda[0][0] += 1.0 / r0;
        let lam_inv = mat_inv(&lambda, "t=0 conditioning precision")?;
        // Affine map α₁|α₂,y₁ ~ N(C α₂ + d, Λ⁻¹).
        let c = mat_mul(&lam_inv, &mat_mul(&rev_prec, &f1_inv));
        let d = mat_vec(&lam_inv, &[nodes[0].y / r0, 0.0]);
        let mean1 = mat_vec(&c, &sm_state[1]);
        sm_state[0] = [mean1[0] + d[0], mean1[1] + d[1]];
        let mut cov0 = mat_add(&mat_mul(&mat_mul(&c, &sm_cov[1]), &mat_t(&c)), &lam_inv);
        symmetrize(&mut cov0);
        sm_cov[0] = cov0;
        gains[0] = c;
    }

    let knots: Vec<f64> = nodes.iter().map(|n| n.x).collect();
    let mean: Vec<f64> = sm_state.iter().map(|s| s[0]).collect();
    let deriv: Vec<f64> = sm_state.iter().map(|s| s[1]).collect();
    let var: Vec<f64> = sm_cov.iter().map(|p| p[0][0] * sigma2).collect();
    Ok(CubicSplineScanFit {
        knots,
        mean,
        deriv,
        var,
        log_lambda,
        sigma2,
        restricted_loglik,
        smoothed_state: sm_state,
        smoothed_cov: sm_cov,
        rts_gain: gains,
        q,
        node_weight: nodes.iter().map(|n| n.w).collect(),
    })
}

/// Fit with `log λ` selected by the concentrated diffuse REML criterion:
/// deterministic coarse grid then golden-section refinement (no RNG, no
/// iteration-count sensitivity — same data ⇒ same fit).
pub fn fit_cubic_spline_scan(
    x: &[f64],
    y: &[f64],
    w: &[f64],
) -> Result<CubicSplineScanFit, String> {
    let (nodes, ssr_within, n_obs) = pool_nodes(x, y, w)?;
    let crit = |ll: f64| concentrated_criterion(&nodes, ssr_within, n_obs, ll);
    let mut best_i = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    let step = (LOG_LAMBDA_HI - LOG_LAMBDA_LO) / (LOG_LAMBDA_GRID - 1) as f64;
    for i in 0..LOG_LAMBDA_GRID {
        let ll = LOG_LAMBDA_LO + step * i as f64;
        let v = crit(ll)?;
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    let mut lo = LOG_LAMBDA_LO + step * best_i.saturating_sub(1) as f64;
    let mut hi = (LOG_LAMBDA_LO + step * (best_i + 1) as f64).min(LOG_LAMBDA_HI);
    // Golden-section maximization on [lo, hi].
    let inv_phi = 0.618_033_988_749_894_9_f64;
    let mut x1 = hi - inv_phi * (hi - lo);
    let mut x2 = lo + inv_phi * (hi - lo);
    let mut f1 = crit(x1)?;
    let mut f2 = crit(x2)?;
    while hi - lo > LOG_LAMBDA_TOL {
        if f1 < f2 {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + inv_phi * (hi - lo);
            f2 = crit(x2)?;
        } else {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - inv_phi * (hi - lo);
            f1 = crit(x1)?;
        }
    }
    fit_cubic_spline_scan_at(x, y, w, 0.5 * (lo + hi), None)
}

/// Lossless serializable snapshot of a [`CubicSplineScanFit`] (#1034).
///
/// Carries exactly the smoother state the Gaussian-bridge `predict` replays:
/// pooled knots, smoothed `(f, f′)` states, smoothed state covariances
/// (unit-σ² scale, symmetric — stored as `[c00, c01, c11]`), RTS backward
/// gains (full 2×2, row-major — gains are NOT symmetric), pooled node
/// weights, and the three fit scalars. `q = e^{−log λ}` and the public
/// `mean`/`deriv`/`var` views are derived on restore rather than stored, so
/// a snapshot cannot go internally inconsistent.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SplineScanState {
    pub knots: Vec<f64>,
    /// Smoothed `(f, f′)` per knot, row-major: `[f_0, f′_0, f_1, f′_1, …]`.
    pub state: Vec<f64>,
    /// Smoothed covariance per knot at unit-σ² scale: `[c00, c01, c11]` each.
    pub cov: Vec<f64>,
    /// RTS backward gain per knot, row-major `[g00, g01, g10, g11]` each
    /// (the last knot's gain is structurally unused and stored as written).
    pub gain: Vec<f64>,
    /// Pooled (tied-abscissa summed) observation weight per knot.
    pub node_weight: Vec<f64>,
    pub log_lambda: f64,
    pub sigma2: f64,
    pub restricted_loglik: f64,
}

impl CubicSplineScanFit {
    /// Snapshot the full smoother state for persistence (#1034).
    pub fn to_state(&self) -> SplineScanState {
        let mut state = Vec::with_capacity(2 * self.knots.len());
        for s in &self.smoothed_state {
            state.extend_from_slice(s);
        }
        let mut cov = Vec::with_capacity(3 * self.knots.len());
        for c in &self.smoothed_cov {
            cov.extend_from_slice(&[c[0][0], c[0][1], c[1][1]]);
        }
        let mut gain = Vec::with_capacity(4 * self.knots.len());
        for g in &self.rts_gain {
            gain.extend_from_slice(&[g[0][0], g[0][1], g[1][0], g[1][1]]);
        }
        SplineScanState {
            knots: self.knots.clone(),
            state,
            cov,
            gain,
            node_weight: self.node_weight.clone(),
            log_lambda: self.log_lambda,
            sigma2: self.sigma2,
            restricted_loglik: self.restricted_loglik,
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
        let m = state.knots.len();
        if m < 3 {
            return Err(format!(
                "spline scan state: needs at least 3 knots, got {m}"
            ));
        }
        if state.state.len() != 2 * m
            || state.cov.len() != 3 * m
            || state.gain.len() != 4 * m
            || state.node_weight.len() != m
        {
            return Err(format!(
                "spline scan state: inconsistent lengths (m={m}, state={}, cov={}, gain={}, weights={})",
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
        if state.knots.windows(2).any(|kk| !(kk[0] < kk[1])) {
            return Err("spline scan state: knots must be strictly increasing".to_string());
        }
        if state.node_weight.iter().any(|&w| w <= 0.0) {
            return Err("spline scan state: node weights must be positive".to_string());
        }
        let smoothed_state: Vec<Vec2> = state.state.chunks_exact(2).map(|s| [s[0], s[1]]).collect();
        let smoothed_cov: Vec<Mat2> = state
            .cov
            .chunks_exact(3)
            .map(|c| [[c[0], c[1]], [c[1], c[2]]])
            .collect();
        let rts_gain: Vec<Mat2> = state
            .gain
            .chunks_exact(4)
            .map(|g| [[g[0], g[1]], [g[2], g[3]]])
            .collect();
        let sigma2 = state.sigma2;
        Ok(Self {
            knots: state.knots.clone(),
            mean: smoothed_state.iter().map(|s| s[0]).collect(),
            deriv: smoothed_state.iter().map(|s| s[1]).collect(),
            var: smoothed_cov.iter().map(|c| c[0][0] * sigma2).collect(),
            log_lambda: state.log_lambda,
            sigma2,
            restricted_loglik: state.restricted_loglik,
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
        let m = self.knots.len();
        let first = self.knots[0];
        let last = self.knots[m - 1];
        if x_new <= first {
            let delta = first - x_new;
            // Backward extrapolation through the reverse map α(x) = F⁻¹(α₁ − η).
            let f_t = transition(delta);
            let f_inv = mat_inv(&f_t, "backward extrapolation transition")?;
            let mean_s = mat_vec(&f_inv, &self.smoothed_state[0]);
            let qm = process_noise(delta, self.q);
            let cov = mat_add(
                &mat_mul(&mat_mul(&f_inv, &self.smoothed_cov[0]), &mat_t(&f_inv)),
                &mat_mul(&mat_mul(&f_inv, &qm), &mat_t(&f_inv)),
            );
            return Ok((mean_s[0], cov[0][0] * self.sigma2));
        }
        if x_new >= last {
            let delta = x_new - last;
            let f_t = transition(delta);
            let mean_s = mat_vec(&f_t, &self.smoothed_state[m - 1]);
            let cov = mat_add(
                &mat_mul(&mat_mul(&f_t, &self.smoothed_cov[m - 1]), &mat_t(&f_t)),
                &process_noise(delta, self.q),
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
        let (f1m, f2m) = (transition(d1), transition(d2));
        let (q1, q2) = (process_noise(d1, self.q), process_noise(d2, self.q));
        let q1_inv = mat_inv(&q1, "bridge left noise")?;
        let q2_inv = mat_inv(&q2, "bridge right noise")?;
        // p(α* | α_t, α_{t+1}) ∝ N(α*; F₁α_t, Q₁)·N(α_{t+1}; F₂α*, Q₂):
        //   Λ = Q₁⁻¹ + F₂ᵀQ₂⁻¹F₂,  mean = Λ⁻¹(Q₁⁻¹F₁ α_t + F₂ᵀQ₂⁻¹ α_{t+1}).
        let lambda = mat_add(&q1_inv, &mat_mul(&mat_mul(&mat_t(&f2m), &q2_inv), &f2m));
        let lam_inv = mat_inv(&lambda, "bridge precision")?;
        let ca = mat_mul(&lam_inv, &mat_mul(&q1_inv, &f1m));
        let cb = mat_mul(&lam_inv, &mat_mul(&mat_t(&f2m), &q2_inv));
        let ma = mat_vec(&ca, &self.smoothed_state[t]);
        let mb = mat_vec(&cb, &self.smoothed_state[t + 1]);
        let mean_s = [ma[0] + mb[0], ma[1] + mb[1]];
        // Push the joint smoothed covariance of (α_t, α_{t+1}) through the
        // affine map: cross term uses Cov(α_t, α_{t+1}|y) = G_t · P^s_{t+1}.
        let cross = mat_mul(&self.rts_gain[t], &self.smoothed_cov[t + 1]);
        let mut cov = mat_add(
            &mat_add(
                &mat_mul(&mat_mul(&ca, &self.smoothed_cov[t]), &mat_t(&ca)),
                &mat_mul(&mat_mul(&cb, &self.smoothed_cov[t + 1]), &mat_t(&cb)),
            ),
            &lam_inv,
        );
        let cab = mat_mul(&mat_mul(&ca, &cross), &mat_t(&cb));
        cov = mat_add(&cov, &mat_add(&cab, &mat_t(&cab)));
        symmetrize(&mut cov);
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
    pub fn deriv_at_knot(&self, t: usize) -> (f64, f64) {
        (
            self.smoothed_state[t][1],
            self.smoothed_cov[t][1][1] * self.sigma2,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #1034 persistence seam: snapshot → JSON → restore must replay the
    /// Gaussian bridge bit-for-bit — knot posteriors, off-knot bridge,
    /// boundary extrapolation, EDF, and derivative posteriors all compare
    /// with exact equality, because every replayed field is either stored
    /// verbatim or derived by the fitter's own expressions.
    #[test]
    fn state_snapshot_round_trips_predict_bit_for_bit() {
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
        let fit = fit_cubic_spline_scan(&x, &y, &w).expect("scan fit");

        let json = serde_json::to_string(&fit.to_state()).expect("serialize state");
        let state: SplineScanState = serde_json::from_str(&json).expect("deserialize state");
        let restored = CubicSplineScanFit::from_state(&state).expect("restore fit");

        assert_eq!(fit.knots, restored.knots);
        assert_eq!(fit.mean, restored.mean);
        assert_eq!(fit.var, restored.var);
        assert_eq!(fit.deriv, restored.deriv);
        assert_eq!(fit.log_lambda.to_bits(), restored.log_lambda.to_bits());
        assert_eq!(fit.sigma2.to_bits(), restored.sigma2.to_bits());
        assert_eq!(fit.edf().to_bits(), restored.edf().to_bits());
        for t in 0..fit.knots.len() {
            let (d0, v0) = fit.deriv_at_knot(t);
            let (d1, v1) = restored.deriv_at_knot(t);
            assert_eq!(d0.to_bits(), d1.to_bits());
            assert_eq!(v0.to_bits(), v1.to_bits());
        }
        // Off-knot bridge, exact knot hit, and both extrapolation sides.
        for &xq in &[-0.2, 0.0, 0.013, 0.5, x[6], 0.987, 1.0, 1.3] {
            let (m0, v0) = fit.predict(xq).expect("predict original");
            let (m1, v1) = restored.predict(xq).expect("predict restored");
            assert_eq!(m0.to_bits(), m1.to_bits(), "mean drift at x={xq}");
            assert_eq!(v0.to_bits(), v1.to_bits(), "variance drift at x={xq}");
        }

        // Corrupt payloads fail loudly, not inside a later predict.
        let mut bad = fit.to_state();
        bad.cov.truncate(bad.cov.len() - 1);
        CubicSplineScanFit::from_state(&bad).expect_err("length mismatch must error");
        let mut bad = fit.to_state();
        bad.sigma2 = -1.0;
        CubicSplineScanFit::from_state(&bad).expect_err("non-positive sigma2 must error");
        let mut bad = fit.to_state();
        bad.knots[2] = bad.knots[1];
        CubicSplineScanFit::from_state(&bad).expect_err("non-increasing knots must error");
    }
}

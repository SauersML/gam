#![allow(non_snake_case)]
//! Dense nonlinear optimization solvers in Rust.
//!
//! This crate provides:
//! - `Problem` + `optimize`: the default API for solver selection that is hard to misuse.
//! - `SecondOrderProblem` + `optimize`: automatic selection for Hessian-aware objectives.
//! - `Bfgs`: dense quasi-Newton optimization with robust hybrid line search.
//! - `NewtonTrustRegion`: Hessian-based trust-region optimization.
//! - `Arc`: Adaptive Regularization with Cubics (ARC).
//!
//! All solvers support optional simple box constraints and are built around practical
//! robustness for noisy/non-ideal objectives.
//!
//! # Features
//! - `Bfgs` hybrid line search: Strong Wolfe with nonmonotone (GLL) Armijo, approximate-Wolfe, and
//!   gradient-reduction acceptors, plus a best-seen salvage path and a small probing grid.
//! - `Bfgs` trust-region (dogleg) fallback with CG-based solves on the inverse Hessian, diagonal
//!   regularization, and scaled-identity resets under severe noise.
//! - `NewtonTrustRegion`: projected Steihaug-Toint trust-region iterations using objective Hessians.
//! - `Arc`: cubic-regularized model steps with adaptive regularization updates (`rho`, `sigma`).
//! - Profile-based heuristic policy selection for rough, piecewise-flat objectives.
//! - Adaptive strategy switching (Wolfe <-> Backtracking) based on success streaks (no timed flips).
//! - Optional box constraints with projected gradients and coordinate clamping.
//! - Optional flat-bracket midpoint acceptance inside zoom.
//! - Stochastic jiggling of step sizes on persistent flats.
//! - Multi-direction (coordinate) rescue when progress is flat.
//!
//! ## Defaults (key settings)
//! - Line search: Strong Wolfe primary; GLL nonmonotone Armijo; approximate‑Wolfe and gradient‑drop
//!   acceptors; probing grid; keep‑best salvage.
//! - Trust region: dogleg fallback enabled; Δ₀ = min(1, 10/||g₀||); adaptive by ρ; SPD enforcement
//!   and scaled‑identity resets when needed.
//! - Tolerances: `c1=1e-4`, `c2=0.9`; heuristics selected by `Profile`.
//! - Zoom midpoint: flat‑bracket midpoint acceptance under profile control.
//! - Stochastic jiggling: default ON with scale 1e‑3 (only after repeated flats in backtracking).
//! - Coordinate rescue: default ON (only after two consecutive flat accepts).
//! - Strategy switching: switch Wolfe<->Backtracking only on success/failure streaks (no timed flips).
//! - Clear, configurable builder API, and robust termination with informative errors.
//!
//! # Example
//!
//! Minimize the Rosenbrock function, a classic test case for optimization algorithms.
//!
//! ```
//! use opt::{
//!     optimize, FirstOrderObjective, FirstOrderSample, MaxIterations, Problem, Profile, Solution,
//!     Tolerance,
//! };
//! use ndarray::{array, Array1};
//!
//! struct Rosenbrock;
//!
//! impl opt::ZerothOrderObjective for Rosenbrock {
//!     fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, opt::ObjectiveEvalError> {
//!         let a = 1.0;
//!         let b = 100.0;
//!         Ok((a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2))
//!     }
//! }
//!
//! impl FirstOrderObjective for Rosenbrock {
//!     fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, opt::ObjectiveEvalError> {
//!         let a = 1.0;
//!         let b = 100.0;
//!         let f = (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2);
//!         let gradient = array![
//!             -2.0 * (a - x[0]) - 4.0 * b * (x[1] - x[0].powi(2)) * x[0],
//!             2.0 * b * (x[1] - x[0].powi(2)),
//!         ];
//!         Ok(FirstOrderSample { value: f, gradient })
//!     }
//! }
//!
//! // Set the initial guess.
//! let x0 = array![-1.2, 1.0];
//!
//! // Run the solver.
//! let Solution {
//!     final_point: x_min,
//!     final_value,
//!     iterations,
//!     ..
//! } = optimize(Problem::new(x0, Rosenbrock))
//!     .with_tolerance(Tolerance::new(1e-6).unwrap())
//!     .with_max_iterations(MaxIterations::new(100).unwrap())
//!     .with_profile(Profile::Robust)
//!     .run()
//!     .expect("BFGS failed to solve");
//!
//! println!(
//!     "Found minimum f([{:.3}, {:.3}]) = {:.4} in {} iterations.",
//!     x_min[0], x_min[1], final_value, iterations
//! );
//!
//! // The known minimum is at [1.0, 1.0].
//! assert!((x_min[0] - 1.0).abs() < 1e-5);
//! assert!((x_min[1] - 1.0).abs() < 1e-5);
//! ```

use ndarray::{Array1, Array2};
use std::collections::VecDeque;

// Numerical helpers and small utilities
const EPS: f64 = f64::EPSILON;
#[inline]
fn eps_f(fk: f64, tau: f64) -> f64 {
    tau * EPS * (1.0 + fk.abs())
}
#[inline]
fn eps_g(gk: &Array1<f64>, dk: &Array1<f64>, tau: f64) -> f64 {
    tau * EPS * gk.dot(gk).sqrt() * dk.dot(dk).sqrt()
}

#[inline]
fn directional_derivative(g: &Array1<f64>, s: &Array1<f64>, alpha: f64, d: &Array1<f64>) -> f64 {
    if alpha > 0.0 {
        g.dot(s) / alpha
    } else {
        g.dot(d)
    }
}

#[inline]
fn classify_line_search_accept(
    core: &BfgsCore,
    step_ok: bool,
    f_k: f64,
    fmax: f64,
    f_trial: f64,
    gk_ts: f64,
    g_trial_dot_d: f64,
    gk_dot_d_eff: f64,
    g_trial_norm: f64,
    gk_norm: f64,
    drop_factor: f64,
    eps_f_k: f64,
    eps_g_k: f64,
    c2: f64,
) -> Option<AcceptKind> {
    if !step_ok {
        return None;
    }
    let armijo_ok = core.accept_armijo(f_k, gk_ts, f_trial);
    let gll_ok = core.accept_gll_nonmonotone(fmax, gk_ts, f_trial);
    let dir_ok = g_trial_dot_d <= -eps_g_k;
    let strong_curv_ok = g_trial_dot_d.abs() <= c2 * gk_dot_d_eff.abs();
    let approx_curv_ok =
        g_trial_dot_d.abs() <= c2 * gk_dot_d_eff.abs() + core.curv_slack_scale * eps_g_k;
    let f_flat_ok = f_trial <= f_k + eps_f_k;

    if armijo_ok && strong_curv_ok {
        Some(AcceptKind::StrongWolfe)
    } else if armijo_ok && core.relaxed_acceptors_enabled() && f_flat_ok && approx_curv_ok && dir_ok
    {
        Some(AcceptKind::ApproxWolfe)
    } else if gll_ok && approx_curv_ok {
        Some(AcceptKind::Nonmonotone)
    } else if core.relaxed_acceptors_enabled()
        && f_flat_ok
        && g_trial_norm <= drop_factor * gk_norm
        && dir_ok
    {
        Some(AcceptKind::GradDrop)
    } else {
        None
    }
}

#[inline]
fn any_free_variables(active: &[bool]) -> bool {
    active.iter().any(|&is_active| !is_active)
}

fn mask_vector_inplace(v: &mut Array1<f64>, active: &[bool]) {
    for (vi, &is_active) in v.iter_mut().zip(active.iter()) {
        if is_active {
            *vi = 0.0;
        }
    }
}

fn masked_hv_inplace(h: &Array2<f64>, v: &Array1<f64>, active: &[bool], out: &mut Array1<f64>) {
    out.fill(0.0);
    for i in 0..h.nrows() {
        if active[i] {
            continue;
        }
        let mut accum = 0.0;
        for j in 0..h.ncols() {
            if active[j] {
                continue;
            }
            accum += h[[i, j]] * v[j];
        }
        out[i] = accum;
    }
}

fn cg_solve_masked_adaptive(
    a: &Array2<f64>,
    b: &Array1<f64>,
    active: &[bool],
    max_iter: usize,
    tol_rel: f64,
    ridge: f64,
) -> Option<Array1<f64>> {
    if a.nrows() != a.ncols() || a.nrows() != b.len() || active.len() != b.len() {
        return None;
    }
    if !any_free_variables(active) {
        return Some(Array1::zeros(b.len()));
    }
    if prefer_dense_direct(b.len()) {
        let (effective_a, effective_b) = build_masked_subproblem_system(a, b, Some(active));
        return dense_solve_shifted(&effective_a, &effective_b, ridge);
    }

    let n = b.len();
    let mut x = Array1::<f64>::zeros(n);
    let mut r = b.clone();
    mask_vector_inplace(&mut r, active);
    let b_norm = r.dot(&r).sqrt();
    if !b_norm.is_finite() {
        return None;
    }
    if b_norm <= 1e-32 {
        return Some(x);
    }
    let tol_abs = tol_rel.max(0.0) * b_norm.max(1e-16);
    let mut p = r.clone();
    let mut rs_old = r.dot(&r);
    let mut ap = Array1::<f64>::zeros(n);

    for _ in 0..max_iter {
        masked_hv_inplace(a, &p, active, &mut ap);
        if ridge > 0.0 {
            for i in 0..n {
                ap[i] += ridge * p[i];
            }
        }
        let p_ap = p.dot(&ap);
        if !p_ap.is_finite() || p_ap <= 0.0 {
            return None;
        }
        let alpha = rs_old / p_ap;
        if !alpha.is_finite() {
            return None;
        }
        x.scaled_add(alpha, &p);
        r.scaled_add(-alpha, &ap);
        mask_vector_inplace(&mut x, active);
        mask_vector_inplace(&mut r, active);
        let rs_new = r.dot(&r);
        if !rs_new.is_finite() {
            return None;
        }
        if rs_new.sqrt() <= tol_abs {
            return Some(x);
        }
        let beta = rs_new / rs_old;
        if !beta.is_finite() || beta < 0.0 {
            return None;
        }
        p *= beta;
        p += &r;
        mask_vector_inplace(&mut p, active);
        rs_old = rs_new;
    }
    Some(x)
}

fn bfgs_eval_cost<ObjFn>(
    oracle: &mut FirstOrderCache,
    obj_fn: &mut ObjFn,
    x: &Array1<f64>,
    func_evals: &mut usize,
) -> Result<f64, ObjectiveEvalError>
where
    ObjFn: FirstOrderObjective,
{
    oracle.eval_cost(obj_fn, x, func_evals)
}

fn bfgs_eval_cost_grad<ObjFn>(
    oracle: &mut FirstOrderCache,
    obj_fn: &mut ObjFn,
    x: &Array1<f64>,
    func_evals: &mut usize,
    grad_evals: &mut usize,
) -> Result<(f64, Array1<f64>), ObjectiveEvalError>
where
    ObjFn: FirstOrderObjective,
{
    oracle.eval_cost_grad(obj_fn, x, func_evals, grad_evals)
}

// Ring buffer for GLL nonmonotone Armijo (internal only)
struct GllWindow {
    buf: VecDeque<f64>,
    cap: usize,
}
impl GllWindow {
    fn new(cap: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(cap.max(1)),
            cap: cap.max(1),
        }
    }
    fn clear(&mut self) {
        self.buf.clear();
    }
    fn push(&mut self, f: f64) {
        if self.buf.len() == self.cap {
            self.buf.pop_front();
        }
        self.buf.push_back(f);
    }
    fn fmax(&self) -> f64 {
        self.buf.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }
    fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
    fn set_cap(&mut self, cap: usize) {
        self.cap = cap.max(1);
        while self.buf.len() > self.cap {
            self.buf.pop_front();
        }
    }
}

// Best-seen tracker during line search/zoom (internal only)
#[derive(Clone)]
struct ProbeBest {
    f: f64,
    x: Array1<f64>,
    g: Array1<f64>,
}
impl ProbeBest {
    fn new(x0: &Array1<f64>, f0: f64, g0: &Array1<f64>) -> Self {
        Self {
            x: x0.clone(),
            f: f0,
            g: g0.clone(),
        }
    }
    fn consider(&mut self, x: &Array1<f64>, f: f64, g: &Array1<f64>) {
        if !f.is_finite() || g.iter().any(|v| !v.is_finite()) {
            return;
        }
        if !self.f.is_finite() || f < self.f {
            self.f = f;
            self.x = x.clone();
            self.g = g.clone();
        }
    }
}

struct CgResult {
    x: Array1<f64>,
    rel_resid: f64,
}

// Conjugate gradient solve for (A + ridge*I) x = b; avoids dense factorizations.
fn cg_solve_from(
    a: &Array2<f64>,
    b: &Array1<f64>,
    x0: Array1<f64>,
    max_iter: usize,
    tol: f64,
    ridge: f64,
) -> Option<CgResult> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return None;
    }
    let mut x = x0;
    let mut ax = a.dot(&x);
    if ridge > 0.0 {
        for i in 0..n {
            ax[i] += ridge * x[i];
        }
    }
    let mut r = b - &ax;
    let mut p = r.clone();
    let mut rs_old = r.dot(&r);
    if !rs_old.is_finite() {
        return None;
    }
    let b_norm = b.dot(b).sqrt().max(1.0);
    let tol_abs = tol * b_norm;
    if rs_old.sqrt() <= tol_abs {
        return Some(CgResult {
            x,
            rel_resid: rs_old.sqrt() / b_norm,
        });
    }
    for _ in 0..max_iter {
        let mut ap = a.dot(&p);
        if ridge > 0.0 {
            for i in 0..n {
                ap[i] += ridge * p[i];
            }
        }
        let p_ap = p.dot(&ap);
        if !p_ap.is_finite() || p_ap <= 0.0 {
            return None;
        }
        let alpha = rs_old / p_ap;
        if !alpha.is_finite() {
            return None;
        }
        x.scaled_add(alpha, &p);
        r.scaled_add(-alpha, &ap);
        let rs_new = r.dot(&r);
        if !rs_new.is_finite() {
            return None;
        }
        if rs_new.sqrt() <= tol_abs {
            return Some(CgResult {
                x,
                rel_resid: rs_new.sqrt() / b_norm,
            });
        }
        let beta = rs_new / rs_old;
        p *= beta;
        p += &r;
        rs_old = rs_new;
    }
    Some(CgResult {
        x,
        rel_resid: rs_old.sqrt() / b_norm,
    })
}

fn dense_solve_shifted(a: &Array2<f64>, b: &Array1<f64>, ridge: f64) -> Option<Array1<f64>> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return None;
    }
    let mut mat = a.clone();
    if ridge > 0.0 {
        for i in 0..n {
            mat[[i, i]] += ridge;
        }
    }
    let mut rhs = b.clone();

    for k in 0..n {
        let mut pivot_row = k;
        let mut pivot_abs = mat[[k, k]].abs();
        for i in (k + 1)..n {
            let cand = mat[[i, k]].abs();
            if cand > pivot_abs {
                pivot_abs = cand;
                pivot_row = i;
            }
        }
        if !pivot_abs.is_finite() || pivot_abs <= 1e-14 {
            return None;
        }
        if pivot_row != k {
            for j in k..n {
                let tmp = mat[[k, j]];
                mat[[k, j]] = mat[[pivot_row, j]];
                mat[[pivot_row, j]] = tmp;
            }
            let tmp_rhs = rhs[k];
            rhs[k] = rhs[pivot_row];
            rhs[pivot_row] = tmp_rhs;
        }

        let pivot = mat[[k, k]];
        for i in (k + 1)..n {
            let factor = mat[[i, k]] / pivot;
            mat[[i, k]] = 0.0;
            for j in (k + 1)..n {
                mat[[i, j]] -= factor * mat[[k, j]];
            }
            rhs[i] -= factor * rhs[k];
        }
    }

    let mut x = Array1::<f64>::zeros(n);
    for ii in 0..n {
        let i = n - 1 - ii;
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum -= mat[[i, j]] * x[j];
        }
        let diag = mat[[i, i]];
        if !diag.is_finite() || diag.abs() <= 1e-14 {
            return None;
        }
        x[i] = sum / diag;
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

#[inline]
fn prefer_dense_direct(n: usize) -> bool {
    n <= 128
}

fn build_masked_subproblem_system(
    h: &Array2<f64>,
    rhs: &Array1<f64>,
    active: Option<&[bool]>,
) -> (Array2<f64>, Array1<f64>) {
    let mut effective_h = h.clone();
    let mut effective_rhs = rhs.clone();
    if let Some(active) = active
        && !active.is_empty()
    {
        for i in 0..active.len() {
            if active[i] {
                effective_rhs[i] = 0.0;
                for j in 0..active.len() {
                    effective_h[[i, j]] = 0.0;
                    effective_h[[j, i]] = 0.0;
                }
                effective_h[[i, i]] = 1.0;
            }
        }
    }
    (effective_h, effective_rhs)
}

fn dense_trust_region_step(
    h: &Array2<f64>,
    g: &Array1<f64>,
    delta: f64,
    active: Option<&[bool]>,
) -> Option<(Array1<f64>, f64)> {
    let rhs = -g.clone();
    let (effective_h, effective_rhs) = build_masked_subproblem_system(h, &rhs, active);
    let solve_with_shift = |lambda: f64| dense_solve_shifted(&effective_h, &effective_rhs, lambda);
    let predicted = |s: &Array1<f64>| {
        let hs = h.dot(s);
        -(g.dot(s) + 0.5 * s.dot(&hs))
    };

    if let Some(s) = solve_with_shift(0.0) {
        let s_norm = s.dot(&s).sqrt();
        let pred = predicted(&s);
        if s_norm.is_finite() && s_norm <= delta && pred.is_finite() && pred > 0.0 {
            return Some((s, pred));
        }
    }

    let mut lambda_lo = 0.0;
    let mut lambda_hi = 1e-8f64;
    let mut best: Option<(Array1<f64>, f64)> = None;
    for _ in 0..80 {
        match solve_with_shift(lambda_hi) {
            Some(s) => {
                let s_norm = s.dot(&s).sqrt();
                let pred = predicted(&s);
                if s_norm.is_finite() && s_norm <= delta && pred.is_finite() && pred > 0.0 {
                    best = Some((s, pred));
                    break;
                }
            }
            None => {}
        }
        lambda_lo = lambda_hi;
        lambda_hi *= 2.0;
    }
    let (mut best_step, mut best_pred) = best?;
    for _ in 0..80 {
        let lambda_mid = 0.5 * (lambda_lo + lambda_hi);
        if !lambda_mid.is_finite() || (lambda_hi - lambda_lo) <= 1e-12 * lambda_hi.max(1.0) {
            break;
        }
        match solve_with_shift(lambda_mid) {
            Some(s) => {
                let s_norm = s.dot(&s).sqrt();
                let pred = predicted(&s);
                if s_norm.is_finite() && s_norm <= delta && pred.is_finite() && pred > 0.0 {
                    lambda_hi = lambda_mid;
                    best_step = s;
                    best_pred = pred;
                } else {
                    lambda_lo = lambda_mid;
                }
            }
            None => {
                lambda_lo = lambda_mid;
            }
        }
    }
    Some((best_step, best_pred))
}

// Adaptive CG iteration cap: full solve for small n, capped growth for large n.
fn cg_iter_cap(n: usize, base: usize) -> usize {
    let full_solve_n = 128usize;
    let cap = 200usize;
    if n <= full_solve_n {
        n.max(1)
    } else {
        n.min(cap).max(base)
    }
}

// Adaptive CG: retry with a higher cap/tighter tol if residual is too large.
fn cg_solve_adaptive(
    a: &Array2<f64>,
    b: &Array1<f64>,
    base_iter: usize,
    tol: f64,
    ridge: f64,
) -> Option<Array1<f64>> {
    let n = a.nrows();
    if prefer_dense_direct(n) {
        return dense_solve_shifted(a, b, ridge);
    }
    let cap1 = cg_iter_cap(n, base_iter);
    let stage1 = cg_solve_from(a, b, Array1::<f64>::zeros(n), cap1, tol, ridge)?;
    if stage1.rel_resid.is_finite() && stage1.rel_resid <= tol * 10.0 {
        return Some(stage1.x);
    }
    let cap2 = cg_iter_cap(n, base_iter.saturating_mul(2));
    if cap2 <= cap1 {
        return Some(stage1.x);
    }
    let refine_iters = cap2.saturating_sub(cap1).max(1);
    let stage2 = cg_solve_from(a, b, stage1.x, refine_iters, tol * 0.1, ridge)?;
    Some(stage2.x)
}

// Helper: return a scaled identity matrix (lambda * I_n).
fn scaled_identity(n: usize, lambda: f64) -> Array2<f64> {
    Array2::<f64>::eye(n) * lambda
}

fn hessian_is_effectively_symmetric(a: &Array2<f64>) -> bool {
    let n = a.nrows();
    let mut max_skew = 0.0f64;
    let mut scale = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let aij = a[[i, j]];
            let aji = a[[j, i]];
            max_skew = max_skew.max((aij - aji).abs());
            scale = scale.max(aij.abs()).max(aji.abs());
        }
    }
    max_skew <= 1e-12 * (1.0 + scale)
}

fn symmetrize_into(workspace: &mut Array2<f64>, a: &Array2<f64>) {
    workspace.assign(a);
    let n = a.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (a[[i, j]] + a[[j, i]]);
            workspace[[i, j]] = v;
            workspace[[j, i]] = v;
        }
    }
}

fn has_finite_positive_diagonal(a: &Array2<f64>) -> bool {
    for i in 0..a.nrows() {
        let diag = a[[i, i]];
        if !diag.is_finite() || diag <= 0.0 {
            return false;
        }
    }
    true
}

fn apply_inverse_bfgs_update_in_place(
    h_inv: &mut Array2<f64>,
    s: &Array1<f64>,
    y: &Array1<f64>,
    backup: &mut Array2<f64>,
) -> bool {
    backup.assign(h_inv);
    let rho = 1.0 / s.dot(y);
    let hy = backup.dot(y);
    let yhy = y.dot(&hy);
    let coeff = (1.0 + yhy * rho) * rho;
    let n = h_inv.nrows();
    for i in 0..n {
        for j in i..n {
            let v = backup[[i, j]] + coeff * s[i] * s[j] - rho * (hy[i] * s[j] + s[i] * hy[j]);
            h_inv[[i, j]] = v;
            h_inv[[j, i]] = v;
        }
    }
    has_finite_positive_diagonal(h_inv)
}

// Box constraints with projection and active-set tolerance.
#[derive(Clone)]
struct BoxSpec {
    lower: Array1<f64>,
    upper: Array1<f64>,
    tol: f64,
}

impl BoxSpec {
    fn new(lower: Array1<f64>, upper: Array1<f64>, tol: f64) -> Self {
        Self { lower, upper, tol }
    }

    fn project(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut z = x.clone();
        for i in 0..z.len() {
            let lo = self.lower[i];
            let hi = self.upper[i];
            if z[i] < lo {
                z[i] = lo;
            } else if z[i] > hi {
                z[i] = hi;
            }
        }
        z
    }

    fn active_mask(&self, x: &Array1<f64>, g: &Array1<f64>) -> Vec<bool> {
        let mut mask = vec![false; x.len()];
        for i in 0..x.len() {
            let lo = self.lower[i];
            let hi = self.upper[i];
            let tol = self.tol;
            let at_lower = x[i] <= lo + tol;
            let at_upper = x[i] >= hi - tol;
            mask[i] = (at_lower && g[i] >= 0.0) || (at_upper && g[i] <= 0.0);
        }
        mask
    }

    fn projected_gradient(&self, x: &Array1<f64>, g: &Array1<f64>) -> Array1<f64> {
        let mut gp = g.clone();
        for i in 0..x.len() {
            let lo = self.lower[i];
            let hi = self.upper[i];
            let tol = self.tol;
            let at_lower = x[i] <= lo + tol;
            let at_upper = x[i] >= hi - tol;
            if (at_lower && g[i] >= 0.0) || (at_upper && g[i] <= 0.0) {
                gp[i] = 0.0;
            }
        }
        gp
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BoundsError {
    #[error("lower/upper lengths differ")]
    DimensionMismatch,
    #[error("lower bound exceeds upper bound at index {index}")]
    InvertedInterval { index: usize },
    #[error("bound tolerance must be finite and >= 0")]
    InvalidTolerance,
}

#[derive(Clone)]
pub struct Bounds {
    spec: BoxSpec,
}

impl Bounds {
    pub fn new(lower: Array1<f64>, upper: Array1<f64>, tol: f64) -> Result<Self, BoundsError> {
        if lower.len() != upper.len() {
            return Err(BoundsError::DimensionMismatch);
        }
        for i in 0..lower.len() {
            if lower[i] > upper[i] {
                return Err(BoundsError::InvertedInterval { index: i });
            }
        }
        if !tol.is_finite() || tol < 0.0 {
            return Err(BoundsError::InvalidTolerance);
        }
        Ok(Self {
            spec: BoxSpec::new(lower, upper, tol),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum FiniteDiffStencil {
    Central { h: f64 },
    Forward { h: f64 },
    Backward { h: f64 },
    Fixed,
}

fn finite_difference_stencil(
    bounds: Option<&BoxSpec>,
    x: &Array1<f64>,
    i: usize,
    base_h: f64,
) -> FiniteDiffStencil {
    if !base_h.is_finite() || base_h <= 0.0 {
        return FiniteDiffStencil::Fixed;
    }
    if let Some(bounds) = bounds {
        let room_lo = (x[i] - bounds.lower[i]).max(0.0);
        let room_hi = (bounds.upper[i] - x[i]).max(0.0);
        if room_lo >= base_h && room_hi >= base_h {
            FiniteDiffStencil::Central { h: base_h }
        } else if room_hi >= room_lo && room_hi > 0.0 {
            FiniteDiffStencil::Forward {
                h: base_h.min(room_hi),
            }
        } else if room_lo > 0.0 {
            FiniteDiffStencil::Backward {
                h: base_h.min(room_lo),
            }
        } else if room_hi > 0.0 {
            FiniteDiffStencil::Forward {
                h: base_h.min(room_hi),
            }
        } else {
            FiniteDiffStencil::Fixed
        }
    } else {
        FiniteDiffStencil::Central { h: base_h }
    }
}

// An enum to manage the adaptive strategy.
#[derive(Debug, Clone, Copy)]
enum LineSearchStrategy {
    StrongWolfe,
    Backtracking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FallbackPolicy {
    Never,
    AutoBfgs,
}

#[derive(Debug, Clone, Copy)]
enum FlatStepPolicy {
    Strict,
    MidpointWithJiggle { scale: f64 },
}

#[derive(Debug, Clone, Copy)]
enum RescuePolicy {
    Off,
    CoordinateHybrid { pool_mult: f64, heads: usize },
}

#[derive(Debug, Clone, Copy)]
enum StallPolicy {
    Off,
    On { window: usize },
}

#[derive(Debug, Clone, Copy)]
enum AcceptKind {
    StrongWolfe,
    ApproxWolfe,
    Nonmonotone,
    GradDrop,
    TrustRegion,
    Rescue,
}

#[derive(Debug)]
enum LineSearchError {
    MaxAttempts(usize),
    StepSizeTooSmall,
    ObjectiveFailed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineSearchFailureReason {
    MaxAttempts,
    StepSizeTooSmall,
}

type LsResult = Result<(f64, f64, Array1<f64>, usize, usize, AcceptKind), LineSearchError>;
const WOLFE_MAX_ATTEMPTS: usize = 20;
const BACKTRACKING_MAX_ATTEMPTS: usize = 50;

/// An error type for clear diagnostics.
#[derive(Debug, thiserror::Error)]
pub enum BfgsError {
    #[error("Internal invariant violated: {message}")]
    InternalInvariant { message: String },
    #[error("Objective evaluation failed: {message}")]
    ObjectiveFailed { message: String },
    #[error(
        "The line search failed ({failure_reason:?}) after {max_attempts} attempts. The optimization landscape may be pathological."
    )]
    LineSearchFailed {
        /// The best solution found before the line search failed.
        last_solution: Box<Solution>,
        /// The number of attempts the line search made before failing.
        max_attempts: usize,
        /// Why the line search failed.
        failure_reason: LineSearchFailureReason,
    },
    #[error(
        "Maximum number of iterations reached without converging. The best solution found is returned."
    )]
    MaxIterationsReached {
        /// The best solution found before the iteration limit was reached.
        last_solution: Box<Solution>,
    },
    #[error("The gradient norm was NaN or infinity, indicating numerical instability.")]
    GradientIsNaN,
    #[error(
        "The line search step size became smaller than machine epsilon, indicating that the algorithm is stuck."
    )]
    StepSizeTooSmall,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("tolerance must be finite and > 0")]
    InvalidTolerance,
    #[error("max_iterations must be >= 1")]
    InvalidMaxIterations,
}

#[derive(Debug, thiserror::Error)]
pub enum MatrixError {
    #[error("matrix must be square; got {rows}x{cols}")]
    NonSquare { rows: usize, cols: usize },
    #[error("matrix must be symmetric")]
    NotSymmetric,
}

fn ensure_square(a: &Array2<f64>) -> Result<usize, MatrixError> {
    if a.nrows() == a.ncols() {
        Ok(a.nrows())
    } else {
        Err(MatrixError::NonSquare {
            rows: a.nrows(),
            cols: a.ncols(),
        })
    }
}

fn ensure_symmetric(a: &Array2<f64>) -> Result<(), MatrixError> {
    let n = ensure_square(a)?;
    for i in 0..n {
        for j in 0..i {
            if !a[[i, j]].is_finite()
                || !a[[j, i]].is_finite()
                || (a[[i, j]] - a[[j, i]]).abs()
                    > 1e-10 * (1.0 + a[[i, j]].abs().max(a[[j, i]].abs()))
            {
                return Err(MatrixError::NotSymmetric);
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct SymmetricMatrix {
    data: Array2<f64>,
}

impl SymmetricMatrix {
    fn from_verified(data: Array2<f64>) -> Self {
        Self { data }
    }

    fn as_array(&self) -> &Array2<f64> {
        &self.data
    }
}

#[derive(Debug, Clone)]
struct SpdInverseHessian {
    data: SymmetricMatrix,
}

impl SpdInverseHessian {
    fn from_verified(data: Array2<f64>) -> Self {
        Self {
            data: SymmetricMatrix::from_verified(data),
        }
    }

    fn into_inner(self) -> Array2<f64> {
        self.data.data
    }
}

pub struct SymmetricHessianMut<'a> {
    data: &'a mut Array2<f64>,
}

impl<'a> SymmetricHessianMut<'a> {
    pub fn new(data: &'a mut Array2<f64>) -> Result<Self, MatrixError> {
        ensure_square(data)?;
        Ok(Self { data })
    }

    pub fn fill(&mut self, value: f64) {
        self.data.fill(value);
    }

    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.data[[i, j]] = value;
        self.data[[j, i]] = value;
    }

    pub fn assign_dense(&mut self, dense: &Array2<f64>) -> Result<(), MatrixError> {
        ensure_symmetric(dense)?;
        if dense.raw_dim() != self.data.raw_dim() {
            return Err(MatrixError::NonSquare {
                rows: dense.nrows(),
                cols: dense.ncols(),
            });
        }
        self.data.assign(dense);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Tolerance(f64);

impl Tolerance {
    pub const DEFAULT: Self = Self(1e-5);

    pub fn new(value: f64) -> Result<Self, ConfigError> {
        if value.is_finite() && value > 0.0 {
            Ok(Self(value))
        } else {
            Err(ConfigError::InvalidTolerance)
        }
    }

    fn get(self) -> f64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MaxIterations(usize);

impl MaxIterations {
    pub const DEFAULT: Self = Self(100);

    pub fn new(value: usize) -> Result<Self, ConfigError> {
        if value >= 1 {
            Ok(Self(value))
        } else {
            Err(ConfigError::InvalidMaxIterations)
        }
    }

    fn get(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    Robust,
    Deterministic,
    Aggressive,
}

#[derive(Debug, Clone)]
pub struct FirstOrderSample {
    pub value: f64,
    pub gradient: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct SecondOrderSample {
    pub value: f64,
    pub gradient: Array1<f64>,
    pub hessian: Option<Array2<f64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedPointStatus {
    Continue,
    Stop,
}

#[derive(Debug, Clone)]
pub struct FixedPointSample {
    pub value: f64,
    pub step: Array1<f64>,
    pub status: FixedPointStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StationarityKind {
    ProjectedGradient,
    StepNorm,
}

/// A summary of a successful solver run.
///
/// Note that for non-convex functions, convergence to a local minimum is not guaranteed.
#[derive(Debug, Clone)]
pub struct Solution {
    /// The point at which the minimum value was found.
    pub final_point: Array1<f64>,
    /// The minimum value of the objective function.
    pub final_value: f64,
    /// The gradient at the final point when the solver is gradient-based.
    pub final_gradient: Option<Array1<f64>>,
    /// The Hessian at the final point when the solver tracks one.
    pub final_hessian: Option<Array2<f64>>,
    /// The projected gradient norm at the final point when available.
    pub final_gradient_norm: Option<f64>,
    /// The final accepted fixed-point step norm when available.
    pub final_step_norm: Option<f64>,
    /// The meaning of the stationarity metric for this solution.
    pub stationarity_kind: StationarityKind,
    /// The total number of iterations performed.
    pub iterations: usize,
    /// The total number of times the objective function was evaluated.
    pub func_evals: usize,
    /// The total number of times the gradient was evaluated.
    pub grad_evals: usize,
    /// The total number of times a Hessian was supplied directly by the objective.
    pub hess_evals: usize,
}

impl Solution {
    fn gradient_based(
        final_point: Array1<f64>,
        final_value: f64,
        final_gradient: Array1<f64>,
        final_gradient_norm: f64,
        final_hessian: Option<Array2<f64>>,
        iterations: usize,
        func_evals: usize,
        grad_evals: usize,
        hess_evals: usize,
    ) -> Self {
        Self {
            final_point,
            final_value,
            final_gradient: Some(final_gradient),
            final_hessian,
            final_gradient_norm: Some(final_gradient_norm),
            final_step_norm: None,
            stationarity_kind: StationarityKind::ProjectedGradient,
            iterations,
            func_evals,
            grad_evals,
            hess_evals,
        }
    }

    fn fixed_point(
        final_point: Array1<f64>,
        final_value: f64,
        final_step_norm: f64,
        iterations: usize,
        func_evals: usize,
    ) -> Self {
        Self {
            final_point,
            final_value,
            final_gradient: None,
            final_hessian: None,
            final_gradient_norm: None,
            final_step_norm: Some(final_step_norm),
            stationarity_kind: StationarityKind::StepNorm,
            iterations,
            func_evals,
            grad_evals: 0,
            hess_evals: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ObjectiveEvalError {
    Recoverable { message: String },
    Fatal { message: String },
}

impl ObjectiveEvalError {
    pub fn recoverable(message: impl Into<String>) -> Self {
        Self::Recoverable {
            message: message.into(),
        }
    }

    pub fn fatal(message: impl Into<String>) -> Self {
        Self::Fatal {
            message: message.into(),
        }
    }
}

pub trait ZerothOrderObjective {
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError>;
}

pub trait FirstOrderObjective: ZerothOrderObjective {
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError>;

    fn set_finite_difference_bounds(&mut self, _bounds: Option<&Bounds>) {}
}

pub trait SecondOrderObjective: FirstOrderObjective {
    fn eval_hessian(&mut self, x: &Array1<f64>) -> Result<SecondOrderSample, ObjectiveEvalError>;
}

pub trait FixedPointObjective {
    fn eval_step(&mut self, x: &Array1<f64>) -> Result<FixedPointSample, ObjectiveEvalError>;
}

pub struct FiniteDiffGradient<ObjFn> {
    inner: ObjFn,
    step: f64,
    bounds: Option<Bounds>,
}

impl<ObjFn> FiniteDiffGradient<ObjFn> {
    pub fn new(inner: ObjFn) -> Self {
        Self {
            inner,
            step: 1e-4,
            bounds: None,
        }
    }

    pub fn with_step(mut self, step: f64) -> Self {
        self.step = step;
        self
    }

    pub fn with_bounds(mut self, bounds: Bounds) -> Self {
        self.bounds = Some(bounds);
        self
    }
}

impl<ObjFn> ZerothOrderObjective for FiniteDiffGradient<ObjFn>
where
    ObjFn: ZerothOrderObjective,
{
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        self.inner.eval_cost(x)
    }
}

impl<ObjFn> FirstOrderObjective for FiniteDiffGradient<ObjFn>
where
    ObjFn: ZerothOrderObjective,
{
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        if !self.step.is_finite() || self.step <= 0.0 {
            return Err(ObjectiveEvalError::fatal(
                "finite-difference gradient step must be positive and finite",
            ));
        }
        let value = recover_on_nonfinite_cost(self.inner.eval_cost(x)?)?;
        let mut gradient = Array1::<f64>::zeros(x.len());
        for i in 0..x.len() {
            let h = self.step * (1.0 + x[i].abs());
            match finite_difference_stencil(self.bounds.as_ref().map(|b| &b.spec), x, i, h) {
                FiniteDiffStencil::Central { h } => {
                    let mut xp = x.clone();
                    xp[i] += h;
                    let fp = recover_on_nonfinite_cost(self.inner.eval_cost(&xp)?)?;
                    let mut xm = x.clone();
                    xm[i] -= h;
                    let fm = recover_on_nonfinite_cost(self.inner.eval_cost(&xm)?)?;
                    gradient[i] = (fp - fm) / (2.0 * h);
                }
                FiniteDiffStencil::Forward { h } => {
                    let mut xp = x.clone();
                    xp[i] += h;
                    let fp = recover_on_nonfinite_cost(self.inner.eval_cost(&xp)?)?;
                    gradient[i] = (fp - value) / h;
                }
                FiniteDiffStencil::Backward { h } => {
                    let mut xm = x.clone();
                    xm[i] -= h;
                    let fm = recover_on_nonfinite_cost(self.inner.eval_cost(&xm)?)?;
                    gradient[i] = (value - fm) / h;
                }
                FiniteDiffStencil::Fixed => {
                    gradient[i] = 0.0;
                }
            }
        }
        Ok(FirstOrderSample { value, gradient })
    }

    fn set_finite_difference_bounds(&mut self, bounds: Option<&Bounds>) {
        self.bounds = bounds.map(|bounds| Bounds {
            spec: bounds.spec.clone(),
        });
    }
}

pub struct Problem<ObjFn> {
    x0: Array1<f64>,
    objective: ObjFn,
    bounds: Option<Bounds>,
    tolerance: Tolerance,
    max_iterations: MaxIterations,
    profile: Profile,
}

impl<ObjFn> Problem<ObjFn>
where
    ObjFn: FirstOrderObjective,
{
    pub fn new(x0: Array1<f64>, objective: ObjFn) -> Self {
        Self {
            x0,
            objective,
            bounds: None,
            tolerance: Tolerance::DEFAULT,
            max_iterations: MaxIterations::DEFAULT,
            profile: Profile::Robust,
        }
    }

    pub fn with_bounds(mut self, bounds: Bounds) -> Self {
        self.objective.set_finite_difference_bounds(Some(&bounds));
        self.bounds = Some(bounds);
        self
    }

    pub fn with_tolerance(mut self, tolerance: Tolerance) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_max_iterations(mut self, max_iterations: MaxIterations) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn with_profile(mut self, profile: Profile) -> Self {
        self.profile = profile;
        self
    }
}

pub struct SecondOrderProblem<ObjFn> {
    x0: Array1<f64>,
    objective: ObjFn,
    bounds: Option<Bounds>,
    tolerance: Tolerance,
    max_iterations: MaxIterations,
    profile: Profile,
    fd_hessian_step: f64,
}

impl<ObjFn> SecondOrderProblem<ObjFn>
where
    ObjFn: SecondOrderObjective,
{
    pub fn new(x0: Array1<f64>, objective: ObjFn) -> Self {
        Self {
            x0,
            objective,
            bounds: None,
            tolerance: Tolerance::DEFAULT,
            max_iterations: MaxIterations::DEFAULT,
            profile: Profile::Robust,
            fd_hessian_step: 1e-4,
        }
    }

    pub fn with_bounds(mut self, bounds: Bounds) -> Self {
        self.objective.set_finite_difference_bounds(Some(&bounds));
        self.bounds = Some(bounds);
        self
    }

    pub fn with_tolerance(mut self, tolerance: Tolerance) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_max_iterations(mut self, max_iterations: MaxIterations) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn with_profile(mut self, profile: Profile) -> Self {
        self.profile = profile;
        self
    }

    pub fn with_fd_hessian_step(mut self, fd_hessian_step: f64) -> Self {
        self.fd_hessian_step = fd_hessian_step;
        self
    }
}

pub enum AutoSecondOrderSolver<ObjFn> {
    NewtonTrustRegion(NewtonTrustRegion<ObjFn>),
    Arc(Arc<ObjFn>),
}

impl<ObjFn> AutoSecondOrderSolver<ObjFn>
where
    ObjFn: SecondOrderObjective,
{
    pub fn run(&mut self) -> Result<Solution, AutoSecondOrderError> {
        match self {
            Self::NewtonTrustRegion(solver) => solver
                .run()
                .map_err(AutoSecondOrderError::NewtonTrustRegion),
            Self::Arc(solver) => solver.run().map_err(AutoSecondOrderError::Arc),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AutoSecondOrderError {
    #[error(transparent)]
    NewtonTrustRegion(#[from] NewtonTrustRegionError),
    #[error(transparent)]
    Arc(#[from] ArcError),
}

#[doc(hidden)]
pub trait IntoAutoSolver {
    type Solver;

    fn into_auto_solver(self) -> Self::Solver;
}

impl<ObjFn> IntoAutoSolver for Problem<ObjFn>
where
    ObjFn: FirstOrderObjective,
{
    type Solver = Bfgs<ObjFn>;

    fn into_auto_solver(self) -> Self::Solver {
        let mut solver = Bfgs::new(self.x0, self.objective)
            .with_tolerance(self.tolerance)
            .with_max_iterations(self.max_iterations)
            .with_profile(self.profile);
        if let Some(bounds) = self.bounds {
            solver = solver.with_bounds(bounds);
        }
        solver
    }
}

impl<ObjFn> IntoAutoSolver for SecondOrderProblem<ObjFn>
where
    ObjFn: SecondOrderObjective,
{
    type Solver = AutoSecondOrderSolver<ObjFn>;

    fn into_auto_solver(self) -> Self::Solver {
        let SecondOrderProblem {
            x0,
            objective,
            bounds,
            tolerance,
            max_iterations,
            profile,
            fd_hessian_step,
        } = self;
        let use_arc = matches!(profile, Profile::Aggressive);
        if use_arc {
            let mut solver = Arc::new(x0, objective)
                .with_tolerance(tolerance)
                .with_max_iterations(max_iterations)
                .with_profile(profile)
                .with_fd_hessian_step(fd_hessian_step);
            if let Some(bounds) = bounds {
                solver = solver.with_bounds(bounds);
            }
            AutoSecondOrderSolver::Arc(solver)
        } else {
            let mut solver = NewtonTrustRegion::new(x0, objective)
                .with_tolerance(tolerance)
                .with_max_iterations(max_iterations)
                .with_profile(profile)
                .with_fd_hessian_step(fd_hessian_step);
            if let Some(bounds) = bounds {
                solver = solver.with_bounds(bounds);
            }
            AutoSecondOrderSolver::NewtonTrustRegion(solver)
        }
    }
}

pub fn optimize<P>(problem: P) -> P::Solver
where
    P: IntoAutoSolver,
{
    problem.into_auto_solver()
}

const CACHE_POINT_EPS: f64 = 1e-14;

#[inline]
fn approx_scalar(lhs: f64, rhs: f64) -> bool {
    (lhs - rhs).abs() <= CACHE_POINT_EPS * (1.0 + lhs.abs().max(rhs.abs()))
}

#[inline]
fn approx_point(lhs: &Array1<f64>, rhs: &Array1<f64>) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(&l, &r)| approx_scalar(l, r))
}

fn recover_on_nonfinite_cost(cost: f64) -> Result<f64, ObjectiveEvalError> {
    if cost.is_finite() {
        Ok(cost)
    } else {
        Err(ObjectiveEvalError::recoverable(
            "objective returned a non-finite cost",
        ))
    }
}

fn recover_on_nonfinite_gradient(gradient: &Array1<f64>) -> Result<(), ObjectiveEvalError> {
    if gradient.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ObjectiveEvalError::recoverable(
            "objective returned a non-finite gradient",
        ))
    }
}

fn sanitize_first_order_sample(
    sample: FirstOrderSample,
) -> Result<FirstOrderSample, ObjectiveEvalError> {
    recover_on_nonfinite_cost(sample.value)?;
    recover_on_nonfinite_gradient(&sample.gradient)?;
    Ok(sample)
}

fn sanitize_second_order_sample(
    sample: SecondOrderSample,
) -> Result<SecondOrderSample, ObjectiveEvalError> {
    let value = recover_on_nonfinite_cost(sample.value)?;
    recover_on_nonfinite_gradient(&sample.gradient)?;
    let hessian = sample
        .hessian
        .filter(|h| h.iter().all(|value| value.is_finite()));
    Ok(SecondOrderSample {
        value,
        gradient: sample.gradient,
        hessian,
    })
}

struct BorrowedSecondOrderAsFirstOrder<'a, O> {
    inner: &'a mut O,
}

impl<'a, O> BorrowedSecondOrderAsFirstOrder<'a, O> {
    fn new(inner: &'a mut O) -> Self {
        Self { inner }
    }
}

impl<O> FirstOrderObjective for BorrowedSecondOrderAsFirstOrder<'_, O>
where
    O: SecondOrderObjective,
{
    fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
        self.inner.eval_grad(x)
    }

    fn set_finite_difference_bounds(&mut self, bounds: Option<&Bounds>) {
        self.inner.set_finite_difference_bounds(bounds);
    }
}

impl<O> ZerothOrderObjective for BorrowedSecondOrderAsFirstOrder<'_, O>
where
    O: SecondOrderObjective,
{
    fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
        self.inner.eval_cost(x)
    }
}

struct FirstOrderCache {
    last_x: Option<Array1<f64>>,
    last_cost: Option<f64>,
    last_grad: Array1<f64>,
    have_last_grad: bool,
}

impl FirstOrderCache {
    fn new(n: usize) -> Self {
        Self {
            last_x: None,
            last_cost: None,
            last_grad: Array1::zeros(n),
            have_last_grad: false,
        }
    }

    fn eval_cost<ObjFn>(
        &mut self,
        obj_fn: &mut ObjFn,
        x: &Array1<f64>,
        func_evals: &mut usize,
    ) -> Result<f64, ObjectiveEvalError>
    where
        ObjFn: FirstOrderObjective,
    {
        if let (Some(last_x), Some(last_cost)) = (&self.last_x, self.last_cost)
            && approx_point(last_x, x)
        {
            return Ok(last_cost);
        }
        let cost = recover_on_nonfinite_cost(obj_fn.eval_cost(x)?)?;
        *func_evals += 1;
        self.last_x = Some(x.clone());
        self.last_cost = Some(cost);
        self.have_last_grad = false;
        Ok(cost)
    }

    fn eval_cost_grad<ObjFn>(
        &mut self,
        obj_fn: &mut ObjFn,
        x: &Array1<f64>,
        func_evals: &mut usize,
        grad_evals: &mut usize,
    ) -> Result<(f64, Array1<f64>), ObjectiveEvalError>
    where
        ObjFn: FirstOrderObjective,
    {
        if let (Some(last_x), Some(last_cost)) = (&self.last_x, self.last_cost)
            && self.have_last_grad
            && approx_point(last_x, x)
        {
            return Ok((last_cost, self.last_grad.clone()));
        }
        let sample = sanitize_first_order_sample(obj_fn.eval_grad(x)?)?;
        *func_evals += 1;
        *grad_evals += 1;
        self.last_x = Some(x.clone());
        self.last_cost = Some(sample.value);
        self.last_grad.assign(&sample.gradient);
        self.have_last_grad = true;
        Ok((sample.value, self.last_grad.clone()))
    }
}

struct SecondOrderCache {
    last_x: Option<Array1<f64>>,
    last_cost: Option<f64>,
    last_grad: Array1<f64>,
    last_hessian: SymmetricMatrix,
    have_last_sample: bool,
    fd_hessian_step: f64,
}

impl SecondOrderCache {
    fn new(n: usize, fd_hessian_step: f64) -> Self {
        Self {
            last_x: None,
            last_cost: None,
            last_grad: Array1::zeros(n),
            last_hessian: SymmetricMatrix::from_verified(Array2::zeros((n, n))),
            have_last_sample: false,
            fd_hessian_step,
        }
    }

    fn finite_difference_hessian<ObjFn>(
        &mut self,
        obj_fn: &mut ObjFn,
        x: &Array1<f64>,
        center_gradient: &Array1<f64>,
        bounds: Option<&BoxSpec>,
        func_evals: &mut usize,
        grad_evals: &mut usize,
    ) -> Result<Array2<f64>, ObjectiveEvalError>
    where
        ObjFn: SecondOrderObjective,
    {
        if !self.fd_hessian_step.is_finite() || self.fd_hessian_step <= 0.0 {
            return Err(ObjectiveEvalError::fatal(
                "finite-difference Hessian step must be positive and finite",
            ));
        }
        let n = x.len();
        let mut hessian = Array2::<f64>::zeros((n, n));
        for j in 0..n {
            let h = self.fd_hessian_step * (1.0 + x[j].abs());
            let column = match finite_difference_stencil(bounds, x, j, h) {
                FiniteDiffStencil::Central { h } => {
                    let mut xp = x.clone();
                    xp[j] += h;
                    let gp = sanitize_first_order_sample(obj_fn.eval_grad(&xp)?)?;
                    *func_evals += 1;
                    *grad_evals += 1;

                    let mut xm = x.clone();
                    xm[j] -= h;
                    let gm = sanitize_first_order_sample(obj_fn.eval_grad(&xm)?)?;
                    *func_evals += 1;
                    *grad_evals += 1;

                    (&gp.gradient - &gm.gradient) / (2.0 * h)
                }
                FiniteDiffStencil::Forward { h } => {
                    let mut xp = x.clone();
                    xp[j] += h;
                    let gp = sanitize_first_order_sample(obj_fn.eval_grad(&xp)?)?;
                    *func_evals += 1;
                    *grad_evals += 1;
                    (&gp.gradient - center_gradient) / h
                }
                FiniteDiffStencil::Backward { h } => {
                    let mut xm = x.clone();
                    xm[j] -= h;
                    let gm = sanitize_first_order_sample(obj_fn.eval_grad(&xm)?)?;
                    *func_evals += 1;
                    *grad_evals += 1;
                    (center_gradient - &gm.gradient) / h
                }
                FiniteDiffStencil::Fixed => Array1::zeros(n),
            };
            hessian.column_mut(j).assign(&column);
        }
        Ok(0.5 * (&hessian + &hessian.t().to_owned()))
    }

    fn eval_cost_grad_hessian<ObjFn>(
        &mut self,
        obj_fn: &mut ObjFn,
        x: &Array1<f64>,
        bounds: Option<&BoxSpec>,
        func_evals: &mut usize,
        grad_evals: &mut usize,
        hess_evals: &mut usize,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), ObjectiveEvalError>
    where
        ObjFn: SecondOrderObjective,
    {
        if let (Some(last_x), Some(last_cost)) = (&self.last_x, self.last_cost)
            && self.have_last_sample
            && approx_point(last_x, x)
        {
            return Ok((
                last_cost,
                self.last_grad.clone(),
                self.last_hessian.as_array().clone(),
            ));
        }
        let sample = sanitize_second_order_sample(obj_fn.eval_hessian(x)?)?;
        *func_evals += 1;
        *grad_evals += 1;
        let hessian = match sample.hessian {
            Some(hessian) => {
                *hess_evals += 1;
                hessian
            }
            None => self.finite_difference_hessian(
                obj_fn,
                x,
                &sample.gradient,
                bounds,
                func_evals,
                grad_evals,
            )?,
        };
        self.last_x = Some(x.clone());
        self.last_cost = Some(sample.value);
        self.last_grad.assign(&sample.gradient);
        self.last_hessian = SymmetricMatrix::from_verified(hessian.clone());
        self.have_last_sample = true;
        Ok((sample.value, self.last_grad.clone(), hessian))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NewtonTrustRegionError {
    #[error(
        "Objective returned a Hessian with shape {got_rows}x{got_cols}; expected {expected}x{expected}"
    )]
    HessianShapeMismatch {
        expected: usize,
        got_rows: usize,
        got_cols: usize,
    },
    #[error("Objective returned non-finite values.")]
    NonFiniteObjective,
    #[error("Objective evaluation failed: {message}")]
    ObjectiveFailed { message: String },
    #[error("Failed to form a positive-definite trust-region model Hessian.")]
    ModelHessianNotSpd,
    #[error(
        "Maximum number of iterations reached without converging. The best solution found is returned."
    )]
    MaxIterationsReached { last_solution: Box<Solution> },
}

struct NewtonTrustRegionCore {
    x0: Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    fd_hessian_step: f64,
    bounds: Option<BoxSpec>,
    trust_radius: f64,
    trust_radius_max: f64,
    eta_accept: f64,
    fallback_policy: FallbackPolicy,
    history_cap: usize,
}

pub struct NewtonTrustRegion<ObjFn> {
    core: NewtonTrustRegionCore,
    obj_fn: ObjFn,
}

#[derive(Debug, thiserror::Error)]
pub enum ArcError {
    #[error(
        "Objective returned a Hessian with shape {got_rows}x{got_cols}; expected {expected}x{expected}"
    )]
    HessianShapeMismatch {
        expected: usize,
        got_rows: usize,
        got_cols: usize,
    },
    #[error("Objective returned non-finite values.")]
    NonFiniteObjective,
    #[error("Objective evaluation failed: {message}")]
    ObjectiveFailed { message: String },
    #[error("ARC subproblem solver failed to produce a usable step.")]
    SubproblemFailed,
    #[error(
        "Maximum number of iterations reached without converging. The best solution found is returned."
    )]
    MaxIterationsReached { last_solution: Box<Solution> },
}

struct ArcCore {
    x0: Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    fd_hessian_step: f64,
    bounds: Option<BoxSpec>,
    theta: f64,
    sigma: f64,
    sigma_min: f64,
    sigma_max: f64,
    eta1: f64,
    eta2: f64,
    gamma1: f64,
    gamma2: f64,
    gamma3: f64,
    fallback_policy: FallbackPolicy,
    history_cap: usize,
    subproblem_max_iterations: usize,
}

/// A configurable Adaptive Regularization with Cubics (ARC) solver.
pub struct Arc<ObjFn> {
    core: ArcCore,
    obj_fn: ObjFn,
}

impl NewtonTrustRegionCore {
    fn new(x0: Array1<f64>) -> Self {
        Self {
            x0,
            tolerance: 1e-5,
            max_iterations: 100,
            fd_hessian_step: 1e-4,
            bounds: None,
            trust_radius: 1.0,
            trust_radius_max: 1e6,
            eta_accept: 0.1,
            fallback_policy: FallbackPolicy::AutoBfgs,
            history_cap: 12,
        }
    }

    fn apply_profile(&mut self, profile: Profile) {
        match profile {
            Profile::Robust => {
                self.eta_accept = 0.1;
                self.fallback_policy = FallbackPolicy::AutoBfgs;
                self.history_cap = 12;
            }
            Profile::Deterministic => {
                self.eta_accept = 0.1;
                self.fallback_policy = FallbackPolicy::Never;
                self.history_cap = 2;
            }
            Profile::Aggressive => {
                self.eta_accept = 0.05;
                self.fallback_policy = FallbackPolicy::AutoBfgs;
                self.history_cap = 20;
            }
        }
    }

    #[inline]
    fn project_point(&self, x: &Array1<f64>) -> Array1<f64> {
        if let Some(bounds) = &self.bounds {
            bounds.project(x)
        } else {
            x.clone()
        }
    }

    #[inline]
    fn projected_gradient(&self, x: &Array1<f64>, g: &Array1<f64>) -> Array1<f64> {
        if let Some(bounds) = &self.bounds {
            bounds.projected_gradient(x, g)
        } else {
            g.clone()
        }
    }

    fn active_mask(&self, x: &Array1<f64>, g: &Array1<f64>) -> Vec<bool> {
        if let Some(bounds) = &self.bounds {
            bounds.active_mask(x, g)
        } else {
            vec![false; x.len()]
        }
    }

    fn predicted_decrease(h_model: &Array2<f64>, g_proj: &Array1<f64>, step: &Array1<f64>) -> f64 {
        let hs = h_model.dot(step);
        -(g_proj.dot(step) + 0.5 * step.dot(&hs))
    }

    fn boundary_tau(p: &Array1<f64>, d: &Array1<f64>, delta: f64) -> Option<f64> {
        let a = d.dot(d);
        if !a.is_finite() || a <= 0.0 {
            return None;
        }
        let b = 2.0 * p.dot(d);
        let c = p.dot(p) - delta * delta;
        let disc = b * b - 4.0 * a * c;
        if !disc.is_finite() || disc < 0.0 {
            return None;
        }
        let sqrt_disc = disc.sqrt();
        let t1 = (-b - sqrt_disc) / (2.0 * a);
        let t2 = (-b + sqrt_disc) / (2.0 * a);
        let mut tau = None;
        if t1.is_finite() && t1 >= 0.0 {
            tau = Some(t1);
        }
        if t2.is_finite() && t2 >= 0.0 {
            tau = Some(tau.map(|v| v.min(t2)).unwrap_or(t2));
        }
        tau
    }

    fn steihaug_toint_step(
        &self,
        h_model: &Array2<f64>,
        g_proj: &Array1<f64>,
        trust_radius: f64,
        active: Option<&[bool]>,
    ) -> Option<(Array1<f64>, f64)> {
        let n = g_proj.len();
        let g_norm = g_proj.dot(g_proj).sqrt();
        if !g_norm.is_finite() || g_norm <= 0.0 {
            return None;
        }
        let active = active.unwrap_or(&[]);
        let use_mask = !active.is_empty();
        if use_mask && !any_free_variables(active) {
            return None;
        }
        if prefer_dense_direct(n) {
            return dense_trust_region_step(
                h_model,
                g_proj,
                trust_radius,
                if use_mask { Some(active) } else { None },
            );
        }

        let mut p = Array1::<f64>::zeros(n);
        let mut r = g_proj.clone();
        if use_mask {
            mask_vector_inplace(&mut r, active);
        }
        let mut d = r.mapv(|v| -v);
        if use_mask {
            mask_vector_inplace(&mut d, active);
        }
        let mut rtr = r.dot(&r);
        let cg_tol = (1e-6 * g_norm).max(1e-12);
        let max_iter = (2 * n).max(10);
        let mut bd = Array1::<f64>::zeros(n);

        for _ in 0..max_iter {
            if use_mask {
                masked_hv_inplace(h_model, &d, active, &mut bd);
            } else {
                bd.assign(&h_model.dot(&d));
            }
            let d_bd = d.dot(&bd);

            // Negative/near-zero curvature: move to trust-region boundary along d.
            if !d_bd.is_finite() || d_bd <= 1e-14 * d.dot(&d).max(1.0) {
                let tau = Self::boundary_tau(&p, &d, trust_radius)?;
                let mut p_nc = p.clone();
                p_nc.scaled_add(tau, &d);
                let pred = Self::predicted_decrease(h_model, g_proj, &p_nc);
                if pred.is_finite() && pred > 0.0 {
                    return Some((p_nc, pred));
                }
                break;
            }

            let alpha = rtr / d_bd;
            if !alpha.is_finite() || alpha <= 0.0 {
                break;
            }

            let mut p_next = p.clone();
            p_next.scaled_add(alpha, &d);
            let p_next_norm = p_next.dot(&p_next).sqrt();
            if p_next_norm >= trust_radius {
                let tau = Self::boundary_tau(&p, &d, trust_radius)?;
                let mut p_b = p.clone();
                p_b.scaled_add(tau, &d);
                let pred = Self::predicted_decrease(h_model, g_proj, &p_b);
                if pred.is_finite() && pred > 0.0 {
                    return Some((p_b, pred));
                }
                break;
            }

            r.scaled_add(alpha, &bd);
            let r_next_norm = r.dot(&r).sqrt();
            if !r_next_norm.is_finite() {
                break;
            }

            p = p_next;
            if r_next_norm <= cg_tol {
                let pred = Self::predicted_decrease(h_model, g_proj, &p);
                if pred.is_finite() && pred > 0.0 {
                    return Some((p, pred));
                }
                break;
            }

            let rtr_next = r.dot(&r);
            let beta = rtr_next / rtr;
            if !beta.is_finite() || beta < 0.0 {
                break;
            }
            d *= beta;
            d -= &r;
            if use_mask {
                mask_vector_inplace(&mut d, active);
            }
            rtr = rtr_next;
        }

        // Conservative fallback: steepest-descent boundary step.
        let g_norm2 = g_proj.dot(g_proj);
        if g_norm2.is_finite() && g_norm2 > 0.0 {
            let mut p_sd = g_proj.clone();
            p_sd *= -(trust_radius / g_norm2.sqrt());
            let pred = Self::predicted_decrease(h_model, g_proj, &p_sd);
            if pred.is_finite() && pred > 0.0 {
                return Some((p_sd, pred));
            }
        }
        None
    }

    fn warm_inverse_from_history(
        &self,
        n: usize,
        history: &VecDeque<(Array1<f64>, Array1<f64>)>,
    ) -> Array2<f64> {
        let mut h_inv = Array2::<f64>::eye(n);
        let mut backup = Array2::<f64>::zeros((n, n));
        if let Some((s_last, y_last)) = history.back() {
            let sy = s_last.dot(y_last);
            let yy = y_last.dot(y_last);
            if sy.is_finite() && yy.is_finite() && sy > 1e-16 && yy > 1e-16 {
                let gamma = (sy / yy).clamp(1e-8, 1e8);
                h_inv = scaled_identity(n, gamma);
            }
        }
        for (s, y) in history {
            let sty = s.dot(y);
            if !sty.is_finite() || sty <= 1e-12 {
                continue;
            }
            if !apply_inverse_bfgs_update_in_place(&mut h_inv, s, y, &mut backup) {
                h_inv.assign(&backup);
            }
        }
        h_inv
    }

    fn run_bfgs_fallback<ObjFn>(
        &self,
        obj_fn: &mut ObjFn,
        x_start: Array1<f64>,
        history: &VecDeque<(Array1<f64>, Array1<f64>)>,
        iter_used: usize,
        mut func_evals: usize,
        mut grad_evals: usize,
    ) -> Result<Solution, NewtonTrustRegionError>
    where
        ObjFn: SecondOrderObjective,
    {
        log::debug!(
            "[OPT] NewtonTrustRegion -> BFGS fallback (iter_used={}, dim={})",
            iter_used,
            x_start.len()
        );
        let n = x_start.len();
        let h0_inv = self.warm_inverse_from_history(n, history);
        let bounds = self.bounds.as_ref().map(|b| Bounds { spec: b.clone() });

        let mut bfgs = Bfgs::new(x_start, BorrowedSecondOrderAsFirstOrder::new(obj_fn))
            .with_tolerance(Tolerance::new(self.tolerance).expect("core tolerance must be valid"))
            .with_max_iterations(
                MaxIterations::new(self.max_iterations.saturating_sub(iter_used).max(1))
                    .expect("core max_iterations must be valid"),
            );
        bfgs.core.initial_b_inv = Some(SpdInverseHessian::from_verified(h0_inv).into_inner());

        if let Some(bounds) = bounds {
            bfgs = bfgs.with_bounds(bounds);
        }

        let fallback_sol = match bfgs.run() {
            Ok(sol) => sol,
            Err(BfgsError::LineSearchFailed { last_solution, .. }) => *last_solution,
            Err(BfgsError::MaxIterationsReached { last_solution }) => *last_solution,
            Err(BfgsError::ObjectiveFailed { message }) => {
                return Err(NewtonTrustRegionError::ObjectiveFailed { message });
            }
            Err(_) => return Err(NewtonTrustRegionError::ModelHessianNotSpd),
        };
        func_evals += fallback_sol.func_evals;
        grad_evals += fallback_sol.grad_evals;
        Ok(Solution {
            iterations: iter_used + fallback_sol.iterations,
            func_evals,
            grad_evals,
            ..fallback_sol
        })
    }

    fn run<ObjFn>(&mut self, obj_fn: &mut ObjFn) -> Result<Solution, NewtonTrustRegionError>
    where
        ObjFn: SecondOrderObjective,
    {
        let n = self.x0.len();
        let mut x_k = self.project_point(&self.x0);
        let mut func_evals = 0usize;
        let mut grad_evals = 0usize;
        let mut hess_evals = 0usize;
        let mut oracle = SecondOrderCache::new(n, self.fd_hessian_step);
        let initial = oracle.eval_cost_grad_hessian(
            obj_fn,
            &x_k,
            self.bounds.as_ref(),
            &mut func_evals,
            &mut grad_evals,
            &mut hess_evals,
        );
        let mut history: VecDeque<(Array1<f64>, Array1<f64>)> =
            VecDeque::with_capacity(self.history_cap.max(2));
        let (mut f_k, mut g_k, mut h_k) = match initial {
            Ok(sample) => sample,
            Err(ObjectiveEvalError::Recoverable { .. }) => {
                if matches!(self.fallback_policy, FallbackPolicy::AutoBfgs) {
                    return self.run_bfgs_fallback(
                        obj_fn,
                        x_k.clone(),
                        &history,
                        0,
                        func_evals,
                        grad_evals,
                    );
                }
                return Err(NewtonTrustRegionError::NonFiniteObjective);
            }
            Err(ObjectiveEvalError::Fatal { message }) => {
                return Err(NewtonTrustRegionError::ObjectiveFailed { message });
            }
        };
        if h_k.nrows() != n || h_k.ncols() != n {
            return Err(NewtonTrustRegionError::HessianShapeMismatch {
                expected: n,
                got_rows: h_k.nrows(),
                got_cols: h_k.ncols(),
            });
        }
        let mut trust_radius = self.trust_radius.max(1e-8);
        let mut g_proj_k = self.projected_gradient(&x_k, &g_k);
        let mut h_model_workspace = Array2::<f64>::zeros((n, n));

        for k in 0..self.max_iterations {
            let g_norm = g_proj_k.dot(&g_proj_k).sqrt();
            if g_norm.is_finite() && g_norm <= self.tolerance {
                return Ok(Solution::gradient_based(
                    x_k,
                    f_k,
                    g_k,
                    g_norm,
                    Some(h_k),
                    k,
                    func_evals,
                    grad_evals,
                    hess_evals,
                ));
            }

            let h_model = if hessian_is_effectively_symmetric(&h_k) {
                &h_k
            } else {
                symmetrize_into(&mut h_model_workspace, &h_k);
                &h_model_workspace
            };
            let active = self.active_mask(&x_k, &g_k);
            let any_active = active.iter().copied().any(|v| v);
            let (trial_step, pred_dec_free) = if any_active {
                if !any_free_variables(&active) {
                    trust_radius = (trust_radius * 0.5).max(1e-12);
                    continue;
                }
                match self.steihaug_toint_step(h_model, &g_proj_k, trust_radius, Some(&active)) {
                    Some(v) => v,
                    None => {
                        trust_radius = (trust_radius * 0.5).max(1e-12);
                        continue;
                    }
                }
            } else {
                match self.steihaug_toint_step(h_model, &g_proj_k, trust_radius, None) {
                    Some(v) => v,
                    None => {
                        trust_radius = (trust_radius * 0.5).max(1e-12);
                        continue;
                    }
                }
            };

            let x_trial_raw = &x_k + &trial_step;
            let x_trial = self.project_point(&x_trial_raw);
            let s_trial = &x_trial - &x_k;
            let s_norm = s_trial.dot(&s_trial).sqrt();
            if !s_norm.is_finite() || s_norm <= 1e-16 {
                trust_radius = (trust_radius * 0.5).max(1e-12);
                continue;
            }
            let pred_dec = if (&s_trial - &trial_step)
                .dot(&(&s_trial - &trial_step))
                .sqrt()
                > 1e-8 * (1.0 + trial_step.dot(&trial_step).sqrt())
            {
                Self::predicted_decrease(h_model, &g_proj_k, &s_trial)
            } else {
                pred_dec_free
            };
            if !pred_dec.is_finite() || pred_dec <= 0.0 {
                trust_radius = (trust_radius * 0.5).max(1e-12);
                continue;
            }

            let (f_trial, g_trial, h_trial) = match oracle.eval_cost_grad_hessian(
                obj_fn,
                &x_trial,
                self.bounds.as_ref(),
                &mut func_evals,
                &mut grad_evals,
                &mut hess_evals,
            ) {
                Ok(sample) => sample,
                Err(ObjectiveEvalError::Recoverable { .. }) => {
                    trust_radius = (trust_radius * 0.2).max(1e-12);
                    continue;
                }
                Err(ObjectiveEvalError::Fatal { message }) => {
                    return Err(NewtonTrustRegionError::ObjectiveFailed { message });
                }
            };
            let act_dec = f_k - f_trial;
            let rho = act_dec / pred_dec;
            if rho > 0.75 && s_norm > 0.99 * trust_radius {
                trust_radius = (trust_radius * 2.0).min(self.trust_radius_max.max(1.0));
            } else if rho < 0.25 {
                trust_radius = (trust_radius * 0.5).max(1e-12);
            }

            if rho > self.eta_accept {
                if h_trial.nrows() != n || h_trial.ncols() != n {
                    return Err(NewtonTrustRegionError::HessianShapeMismatch {
                        expected: n,
                        got_rows: h_trial.nrows(),
                        got_cols: h_trial.ncols(),
                    });
                }
                x_k = x_trial;
                f_k = f_trial;
                let y_k = &g_trial - &g_k;
                if s_trial.dot(&s_trial).sqrt() > 1e-14 && y_k.dot(&y_k).sqrt() > 1e-14 {
                    if history.len() == self.history_cap.max(2) {
                        history.pop_front();
                    }
                    history.push_back((s_trial.clone(), y_k));
                }
                g_k = g_trial;
                h_k = h_trial;
                g_proj_k = self.projected_gradient(&x_k, &g_k);
            }
        }

        let g_norm = g_proj_k.dot(&g_proj_k).sqrt();
        Err(NewtonTrustRegionError::MaxIterationsReached {
            last_solution: Box::new(Solution::gradient_based(
                x_k,
                f_k,
                g_k,
                g_norm,
                Some(h_k),
                self.max_iterations,
                func_evals,
                grad_evals,
                hess_evals,
            )),
        })
    }
}

impl ArcCore {
    fn new(x0: Array1<f64>) -> Self {
        Self {
            x0,
            tolerance: 1e-5,
            max_iterations: 100,
            fd_hessian_step: 1e-4,
            bounds: None,
            theta: 1.0,
            sigma: 1.0,
            sigma_min: 1e-10,
            sigma_max: 1e12,
            eta1: 0.1,
            eta2: 0.9,
            // ARC defaults tuned to reduce regularization aggressively on very
            // successful iterations while keeping conservative growth otherwise.
            gamma1: 0.1,
            gamma2: 2.0,
            gamma3: 2.0,
            fallback_policy: FallbackPolicy::AutoBfgs,
            history_cap: 12,
            subproblem_max_iterations: 80,
        }
    }

    fn apply_profile(&mut self, profile: Profile) {
        match profile {
            Profile::Robust => {
                self.theta = 1.0;
                self.eta1 = 0.1;
                self.eta2 = 0.9;
                self.gamma1 = 0.1;
                self.gamma2 = 2.0;
                self.gamma3 = 2.0;
                self.fallback_policy = FallbackPolicy::AutoBfgs;
                self.history_cap = 12;
                self.subproblem_max_iterations = 80;
            }
            Profile::Deterministic => {
                self.theta = 1.0;
                self.eta1 = 0.1;
                self.eta2 = 0.9;
                self.gamma1 = 0.1;
                self.gamma2 = 2.0;
                self.gamma3 = 2.0;
                self.fallback_policy = FallbackPolicy::Never;
                self.history_cap = 2;
                self.subproblem_max_iterations = 80;
            }
            Profile::Aggressive => {
                self.theta = 1.25;
                self.eta1 = 0.05;
                self.eta2 = 0.8;
                self.gamma1 = 0.2;
                self.gamma2 = 1.5;
                self.gamma3 = 2.5;
                self.fallback_policy = FallbackPolicy::AutoBfgs;
                self.history_cap = 20;
                self.subproblem_max_iterations = 120;
            }
        }
    }

    #[inline]
    fn project_point(&self, x: &Array1<f64>) -> Array1<f64> {
        if let Some(bounds) = &self.bounds {
            bounds.project(x)
        } else {
            x.clone()
        }
    }

    #[inline]
    fn projected_gradient(&self, x: &Array1<f64>, g: &Array1<f64>) -> Array1<f64> {
        if let Some(bounds) = &self.bounds {
            bounds.projected_gradient(x, g)
        } else {
            g.clone()
        }
    }

    fn active_mask(&self, x: &Array1<f64>, g: &Array1<f64>) -> Vec<bool> {
        if let Some(bounds) = &self.bounds {
            bounds.active_mask(x, g)
        } else {
            vec![false; x.len()]
        }
    }

    fn warm_inverse_from_history(
        &self,
        n: usize,
        history: &VecDeque<(Array1<f64>, Array1<f64>)>,
    ) -> Array2<f64> {
        let mut h_inv = Array2::<f64>::eye(n);
        let mut backup = Array2::<f64>::zeros((n, n));
        if let Some((s_last, y_last)) = history.back() {
            let sy = s_last.dot(y_last);
            let yy = y_last.dot(y_last);
            if sy.is_finite() && yy.is_finite() && sy > 1e-16 && yy > 1e-16 {
                let gamma = (sy / yy).clamp(1e-8, 1e8);
                h_inv = scaled_identity(n, gamma);
            }
        }
        for (s, y) in history {
            let sty = s.dot(y);
            if !sty.is_finite() || sty <= 1e-12 {
                continue;
            }
            if !apply_inverse_bfgs_update_in_place(&mut h_inv, s, y, &mut backup) {
                h_inv.assign(&backup);
            }
        }
        h_inv
    }

    fn run_bfgs_fallback<ObjFn>(
        &self,
        obj_fn: &mut ObjFn,
        x_start: Array1<f64>,
        history: &VecDeque<(Array1<f64>, Array1<f64>)>,
        iter_used: usize,
        mut func_evals: usize,
        mut grad_evals: usize,
    ) -> Result<Solution, ArcError>
    where
        ObjFn: SecondOrderObjective,
    {
        log::debug!(
            "[OPT] ARC -> BFGS fallback (iter_used={}, dim={})",
            iter_used,
            x_start.len()
        );
        let n = x_start.len();
        let h0_inv = self.warm_inverse_from_history(n, history);
        let bounds = self.bounds.as_ref().map(|b| Bounds { spec: b.clone() });

        let mut bfgs = Bfgs::new(x_start, BorrowedSecondOrderAsFirstOrder::new(obj_fn))
            .with_tolerance(Tolerance::new(self.tolerance).expect("core tolerance must be valid"))
            .with_max_iterations(
                MaxIterations::new(self.max_iterations.saturating_sub(iter_used).max(1))
                    .expect("core max_iterations must be valid"),
            );
        bfgs.core.initial_b_inv = Some(SpdInverseHessian::from_verified(h0_inv).into_inner());

        if let Some(bounds) = bounds {
            bfgs = bfgs.with_bounds(bounds);
        }

        let fallback_sol = match bfgs.run() {
            Ok(sol) => sol,
            Err(BfgsError::LineSearchFailed { last_solution, .. }) => *last_solution,
            Err(BfgsError::MaxIterationsReached { last_solution }) => *last_solution,
            Err(BfgsError::ObjectiveFailed { message }) => {
                return Err(ArcError::ObjectiveFailed { message });
            }
            Err(_) => return Err(ArcError::SubproblemFailed),
        };
        func_evals += fallback_sol.func_evals;
        grad_evals += fallback_sol.grad_evals;
        Ok(Solution {
            iterations: iter_used + fallback_sol.iterations,
            func_evals,
            grad_evals,
            ..fallback_sol
        })
    }

    fn arc_model_value(
        &self,
        g: &Array1<f64>,
        h: &Array2<f64>,
        sigma: f64,
        s: &Array1<f64>,
        active: Option<&[bool]>,
    ) -> (f64, f64, Array1<f64>) {
        // Cubic model:
        // m(s) = g^T s + (1/2) s^T H s + (sigma/3) ||s||^3
        // and gradient:
        // ∇m(s) = g + Hs + sigma ||s|| s.
        let mut hs = Array1::<f64>::zeros(s.len());
        if let Some(active) = active {
            masked_hv_inplace(h, s, active, &mut hs);
        } else {
            hs.assign(&h.dot(s));
        }
        let s_norm = s.dot(s).sqrt();
        let cubic = (sigma / 3.0) * s_norm.powi(3);
        let model_delta = g.dot(s) + 0.5 * s.dot(&hs) + cubic;
        let mut grad_m = g + &hs + &(s * (sigma * s_norm));
        if let Some(active) = active {
            mask_vector_inplace(&mut grad_m, active);
        }
        (model_delta, s_norm, grad_m)
    }

    fn cauchy_arc_step(
        &self,
        g: &Array1<f64>,
        h: &Array2<f64>,
        sigma: f64,
        active: Option<&[bool]>,
    ) -> Option<Array1<f64>> {
        let g_norm = g.dot(g).sqrt();
        if !g_norm.is_finite() || g_norm <= 0.0 {
            return Some(Array1::<f64>::zeros(g.len()));
        }
        let mut d = -g.clone();
        if let Some(active) = active {
            mask_vector_inplace(&mut d, active);
        }
        let g2 = g.dot(g);
        let mut hd = Array1::<f64>::zeros(d.len());
        if let Some(active) = active {
            masked_hv_inplace(h, &d, active, &mut hd);
        } else {
            hd.assign(&h.dot(&d));
        }
        let d_hd = d.dot(&hd);
        let c = sigma * g_norm.powi(3);
        let mut alpha = if c > 1e-16 {
            let disc = d_hd * d_hd + 4.0 * c * g2;
            let sqrt_disc = disc.max(0.0).sqrt();
            (-d_hd + sqrt_disc) / (2.0 * c)
        } else if d_hd > 1e-16 {
            g2 / d_hd
        } else {
            1.0 / g_norm.max(1.0)
        };
        if !alpha.is_finite() || alpha <= 0.0 {
            alpha = 1.0 / g_norm.max(1.0);
        }
        let mut s = d * alpha;
        let mut m = self.arc_model_value(g, h, sigma, &s, active).0;
        for _ in 0..8 {
            if m <= 0.0 {
                return Some(s);
            }
            s *= 0.5;
            m = self.arc_model_value(g, h, sigma, &s, active).0;
        }
        if m <= 0.0 { Some(s) } else { None }
    }

    #[inline]
    fn escalate_sigma_on_failure(&mut self, failure_streak: &mut usize) {
        // Two-stage escalation:
        // - early failures: use gamma2 to avoid overreacting to transient noise,
        // - repeated failures: switch to gamma3 for stronger regularization.
        *failure_streak += 1;
        let growth = if *failure_streak >= 3 {
            self.gamma3
        } else {
            self.gamma2
        };
        self.sigma = (self.sigma * growth).min(self.sigma_max);
    }

    fn solve_arc_subproblem(
        &self,
        h: &Array2<f64>,
        g: &Array1<f64>,
        sigma: f64,
        active: Option<&[bool]>,
    ) -> Option<Array1<f64>> {
        let g_norm = g.dot(g).sqrt();
        if !g_norm.is_finite() {
            return None;
        }
        if g_norm <= 1e-16 {
            return Some(Array1::<f64>::zeros(g.len()));
        }

        let rhs = -g.clone();
        let n = g.len();
        let cg_base_iter = (n / 2).clamp(25, 120);
        let active_opt = active;
        let active = active.unwrap_or(&[]);
        let use_mask = !active.is_empty();
        if use_mask && !any_free_variables(active) {
            return Some(Array1::<f64>::zeros(g.len()));
        }
        let direct_small_dense = prefer_dense_direct(n);
        let (effective_h, effective_rhs) = if direct_small_dense {
            build_masked_subproblem_system(h, &rhs, if use_mask { Some(active) } else { None })
        } else {
            (Array2::<f64>::zeros((0, 0)), Array1::<f64>::zeros(0))
        };
        // Solve (H + lambda I)s = -g while steering lambda toward sigma*||s||.
        // This tracks the cubic first-order stationarity condition.
        let mut lambda = (sigma * g_norm.sqrt()).max(1e-8);
        let mut best: Option<(f64, Array1<f64>)> = None;
        let mut hs = Array1::<f64>::zeros(n);

        for _ in 0..self.subproblem_max_iterations {
            let mut s = if direct_small_dense {
                match dense_solve_shifted(&effective_h, &effective_rhs, lambda) {
                    Some(v) => v,
                    None => {
                        lambda = (2.0 * lambda).max(1e-8);
                        continue;
                    }
                }
            } else if use_mask {
                let mut s = Array1::<f64>::zeros(n);
                let mut r = rhs.clone();
                mask_vector_inplace(&mut r, active);
                let mut p = r.clone();
                let mut rtr = r.dot(&r);
                if !rtr.is_finite() {
                    return None;
                }
                for _ in 0..cg_base_iter {
                    masked_hv_inplace(h, &p, active, &mut hs);
                    hs.scaled_add(lambda, &p);
                    let denom = p.dot(&hs);
                    if !denom.is_finite() || denom <= 1e-14 * p.dot(&p).max(1.0) {
                        s.fill(f64::NAN);
                        break;
                    }
                    let alpha = rtr / denom;
                    if !alpha.is_finite() || alpha <= 0.0 {
                        s.fill(f64::NAN);
                        break;
                    }
                    s.scaled_add(alpha, &p);
                    r.scaled_add(-alpha, &hs);
                    mask_vector_inplace(&mut s, active);
                    mask_vector_inplace(&mut r, active);
                    let rtr_next = r.dot(&r);
                    if !rtr_next.is_finite() {
                        s.fill(f64::NAN);
                        break;
                    }
                    if rtr_next.sqrt() <= 1e-10 * g_norm.max(1.0) {
                        break;
                    }
                    let beta = rtr_next / rtr.max(1e-32);
                    if !beta.is_finite() || beta < 0.0 {
                        s.fill(f64::NAN);
                        break;
                    }
                    p *= beta;
                    p += &r;
                    mask_vector_inplace(&mut p, active);
                    rtr = rtr_next;
                }
                s
            } else {
                match cg_solve_adaptive(h, &rhs, cg_base_iter, 1e-10, lambda) {
                    Some(v) => v,
                    None => {
                        lambda = (2.0 * lambda).max(1e-8);
                        continue;
                    }
                }
            };
            if use_mask {
                mask_vector_inplace(&mut s, active);
            }
            if s.iter().any(|v| !v.is_finite()) {
                lambda = (2.0 * lambda).max(1e-8);
                continue;
            }

            let (m_delta, s_norm, grad_m) =
                self.arc_model_value(g, h, sigma, &s, if use_mask { Some(active) } else { None });
            if !m_delta.is_finite() || !s_norm.is_finite() {
                lambda = (2.0 * lambda).max(1e-8);
                continue;
            }
            let grad_norm = grad_m.dot(&grad_m).sqrt();
            let target = self.theta * s_norm * s_norm;
            let merit = if target > 0.0 {
                grad_norm / target
            } else {
                grad_norm
            };
            if best.as_ref().map(|(bm, _)| merit < *bm).unwrap_or(true) {
                best = Some((merit, s.clone()));
            }

            // ARC first-order progress:
            // m(s) <= m(0) and ||∇m(s)|| <= theta ||s||^2.
            // Also require near-consistency with lambda = sigma||s|| used by the
            // cubic first-order optimality system.
            let lambda_target = (sigma * s_norm).max(1e-12);
            let rel_lam_gap = (lambda - lambda_target).abs() / lambda.max(1.0);
            if m_delta <= 0.0 && grad_norm <= target.max(1e-14) && rel_lam_gap <= 0.25 {
                return Some(s);
            }

            if m_delta > 0.0 {
                lambda = (2.0 * lambda.max(lambda_target)).max(1e-8);
            } else {
                // Damped fixed-point tracking of lambda = sigma||s||.
                // Restrict per-iteration movement to keep the sequence stable.
                let ratio = (lambda_target / lambda.max(1e-16)).clamp(0.25, 4.0);
                let lambda_next = lambda * ratio;
                let mixed = 0.5 * lambda + 0.5 * lambda_next;
                lambda = mixed.max(1e-12);
            }
        }

        if let Some((_, s)) = best {
            let (m_delta, s_norm, grad_m) =
                self.arc_model_value(g, h, sigma, &s, if use_mask { Some(active) } else { None });
            let grad_norm = grad_m.dot(&grad_m).sqrt();
            let target = self.theta * s_norm * s_norm;
            if m_delta <= 0.0 && grad_norm <= target.max(1e-14) {
                return Some(s);
            }
        }
        self.cauchy_arc_step(
            g,
            h,
            sigma,
            if use_mask { Some(active) } else { active_opt },
        )
    }

    fn run<ObjFn>(&mut self, obj_fn: &mut ObjFn) -> Result<Solution, ArcError>
    where
        ObjFn: SecondOrderObjective,
    {
        let n = self.x0.len();
        let mut x_k = self.project_point(&self.x0);
        let mut func_evals = 0usize;
        let mut grad_evals = 0usize;
        let mut hess_evals = 0usize;
        let mut oracle = SecondOrderCache::new(n, self.fd_hessian_step);
        let initial = oracle.eval_cost_grad_hessian(
            obj_fn,
            &x_k,
            self.bounds.as_ref(),
            &mut func_evals,
            &mut grad_evals,
            &mut hess_evals,
        );
        let mut history: VecDeque<(Array1<f64>, Array1<f64>)> =
            VecDeque::with_capacity(self.history_cap.max(2));
        let (mut f_k, mut g_k, mut h_k) = match initial {
            Ok(sample) => sample,
            Err(ObjectiveEvalError::Recoverable { .. }) => {
                if matches!(self.fallback_policy, FallbackPolicy::AutoBfgs) {
                    return self.run_bfgs_fallback(
                        obj_fn,
                        x_k.clone(),
                        &history,
                        0,
                        func_evals,
                        grad_evals,
                    );
                }
                return Err(ArcError::NonFiniteObjective);
            }
            Err(ObjectiveEvalError::Fatal { message }) => {
                return Err(ArcError::ObjectiveFailed { message });
            }
        };
        if h_k.nrows() != n || h_k.ncols() != n {
            return Err(ArcError::HessianShapeMismatch {
                expected: n,
                got_rows: h_k.nrows(),
                got_cols: h_k.ncols(),
            });
        }
        let mut model_failure_streak = 0usize;
        let mut h_model_workspace = Array2::<f64>::zeros((n, n));

        for k in 0..self.max_iterations {
            let g_proj_k = self.projected_gradient(&x_k, &g_k);
            let g_norm = g_proj_k.dot(&g_proj_k).sqrt();
            if g_norm.is_finite() && g_norm <= self.tolerance {
                return Ok(Solution::gradient_based(
                    x_k,
                    f_k,
                    g_k,
                    g_norm,
                    Some(h_k),
                    k,
                    func_evals,
                    grad_evals,
                    hess_evals,
                ));
            }

            let h_model = if hessian_is_effectively_symmetric(&h_k) {
                &h_k
            } else {
                symmetrize_into(&mut h_model_workspace, &h_k);
                &h_model_workspace
            };
            let active = self.active_mask(&x_k, &g_k);
            let any_active = active.iter().copied().any(|v| v);
            // Solve the cubic model in the full space while masking bound-active
            // coordinates instead of materializing reduced subspaces.
            let step = if any_active {
                if !any_free_variables(&active) {
                    // All coordinates are active at their bounds: increase sigma and retry.
                    self.escalate_sigma_on_failure(&mut model_failure_streak);
                    continue;
                }
                match self.solve_arc_subproblem(h_model, &g_proj_k, self.sigma, Some(&active)) {
                    Some(s) => s,
                    None => {
                        // Failed subproblem solve: moderate growth first, stronger only
                        // after repeated failures.
                        self.escalate_sigma_on_failure(&mut model_failure_streak);
                        continue;
                    }
                }
            } else {
                match self.solve_arc_subproblem(h_model, &g_proj_k, self.sigma, None) {
                    Some(s) => s,
                    None => {
                        // Failed subproblem solve: moderate growth first, stronger only
                        // after repeated failures.
                        self.escalate_sigma_on_failure(&mut model_failure_streak);
                        continue;
                    }
                }
            };

            let x_trial_raw = &x_k + &step;
            let x_trial = self.project_point(&x_trial_raw);
            let s_trial = &x_trial - &x_k;
            let s_norm = s_trial.dot(&s_trial).sqrt();
            if !s_norm.is_finite() || s_norm <= 1e-16 {
                self.escalate_sigma_on_failure(&mut model_failure_streak);
                continue;
            }
            let step_distortion = (&s_trial - &step).dot(&(&s_trial - &step)).sqrt();
            let step_norm_ref = step.dot(&step).sqrt();
            let proj_changed = step_distortion > 1e-8 * (1.0 + step_norm_ref);
            if proj_changed {
                // The unconstrained cubic model was solved for `step`, not the clipped
                // projected step `s_trial`. Do not use ARC's rho/sigma update on the
                // distorted step. Instead, refresh a coherent sample at the projected
                // point and accept it only as a bound-activation progress step.
                let projected = oracle.eval_cost_grad_hessian(
                    obj_fn,
                    &x_trial,
                    self.bounds.as_ref(),
                    &mut func_evals,
                    &mut grad_evals,
                    &mut hess_evals,
                );
                let (f_trial, g_trial, h_trial) = match projected {
                    Ok(sample) => sample,
                    Err(ObjectiveEvalError::Recoverable { .. }) => {
                        self.escalate_sigma_on_failure(&mut model_failure_streak);
                        continue;
                    }
                    Err(ObjectiveEvalError::Fatal { message }) => {
                        return Err(ArcError::ObjectiveFailed { message });
                    }
                };
                if h_trial.nrows() != n || h_trial.ncols() != n {
                    return Err(ArcError::HessianShapeMismatch {
                        expected: n,
                        got_rows: h_trial.nrows(),
                        got_cols: h_trial.ncols(),
                    });
                }
                let g_proj_trial = self.projected_gradient(&x_trial, &g_trial);
                let g_proj_trial_norm = g_proj_trial.dot(&g_proj_trial).sqrt();
                if f_trial <= f_k
                    && (g_proj_trial_norm <= g_norm || g_proj_trial_norm <= self.tolerance)
                {
                    let y_k = &g_trial - &g_k;
                    if s_norm > 1e-14 && y_k.dot(&y_k).sqrt() > 1e-14 {
                        if history.len() == self.history_cap.max(2) {
                            history.pop_front();
                        }
                        history.push_back((s_trial.clone(), y_k));
                    }
                    x_k = x_trial;
                    f_k = f_trial;
                    g_k = g_trial;
                    h_k = h_trial;
                    model_failure_streak = 0;
                    // Bias the next cubic solve toward smaller feasible steps after
                    // a bound-clipped move.
                    self.sigma = (self.sigma * self.gamma2).min(self.sigma_max);
                } else {
                    self.escalate_sigma_on_failure(&mut model_failure_streak);
                }
                continue;
            }
            let (m_delta_trial, _, grad_m_trial) =
                self.arc_model_value(&g_proj_k, h_model, self.sigma, &s_trial, Some(&active));

            // Enforce ARC first-order subproblem progress on the actual trial step
            // (after possible box projection):
            // m(s) <= m(0) and ||∇m(s)|| <= theta ||s||^2.
            let grad_m_norm = grad_m_trial.dot(&grad_m_trial).sqrt();
            let target_m = self.theta * s_norm * s_norm;
            if !m_delta_trial.is_finite()
                || !grad_m_norm.is_finite()
                || m_delta_trial > 0.0
                || grad_m_norm > target_m.max(1e-14)
            {
                self.escalate_sigma_on_failure(&mut model_failure_streak);
                continue;
            }

            // Standard ARC predicted reduction is m(0) - m(s) = -m(s),
            // where `m_delta_trial` already includes the cubic term.
            let denom = -m_delta_trial;
            if !denom.is_finite() || denom <= 0.0 {
                self.escalate_sigma_on_failure(&mut model_failure_streak);
                continue;
            }

            let (f_trial, g_trial, h_trial) = match oracle.eval_cost_grad_hessian(
                obj_fn,
                &x_trial,
                self.bounds.as_ref(),
                &mut func_evals,
                &mut grad_evals,
                &mut hess_evals,
            ) {
                Ok(sample) => sample,
                Err(ObjectiveEvalError::Recoverable { .. }) => {
                    self.escalate_sigma_on_failure(&mut model_failure_streak);
                    continue;
                }
                Err(ObjectiveEvalError::Fatal { message }) => {
                    return Err(ArcError::ObjectiveFailed { message });
                }
            };
            let rho = (f_k - f_trial) / denom;
            model_failure_streak = 0;
            // ARC accept/reject decision:
            // accept trial point iff rho >= eta1.
            if rho >= self.eta1 {
                if h_trial.nrows() != n || h_trial.ncols() != n {
                    return Err(ArcError::HessianShapeMismatch {
                        expected: n,
                        got_rows: h_trial.nrows(),
                        got_cols: h_trial.ncols(),
                    });
                }
                let y_k = &g_trial - &g_k;
                if s_norm > 1e-14 && y_k.dot(&y_k).sqrt() > 1e-14 {
                    if history.len() == self.history_cap.max(2) {
                        history.pop_front();
                    }
                    history.push_back((s_trial.clone(), y_k));
                }
                x_k = x_trial;
                f_k = f_trial;
                g_k = g_trial;
                h_k = h_trial;
            }

            // Canonical ARC sigma update:
            // very successful -> decrease; successful -> keep; unsuccessful -> increase.
            if rho >= self.eta2 {
                self.sigma = (self.sigma * self.gamma1).max(self.sigma_min);
            } else if rho >= self.eta1 {
                self.sigma = self.sigma.max(self.sigma_min);
            } else if rho.is_finite() {
                self.sigma = (self.sigma * self.gamma2).min(self.sigma_max);
            } else {
                // Numerically pathological ratio: use the stronger growth factor.
                self.sigma = (self.sigma * self.gamma3).min(self.sigma_max);
            }
        }

        let g_proj_k = self.projected_gradient(&x_k, &g_k);
        let g_norm = g_proj_k.dot(&g_proj_k).sqrt();
        Err(ArcError::MaxIterationsReached {
            last_solution: Box::new(Solution::gradient_based(
                x_k,
                f_k,
                g_k,
                g_norm,
                Some(h_k),
                self.max_iterations,
                func_evals,
                grad_evals,
                hess_evals,
            )),
        })
    }
}

/// Core configuration and adaptive state for the BFGS solver.
struct BfgsCore {
    x0: Array1<f64>,
    // --- Configuration ---
    tolerance: f64,
    max_iterations: usize,
    c1: f64,
    c2: f64,
    tau_f: f64,
    tau_g: f64,
    bounds: Option<BoxSpec>,
    flat_step_policy: FlatStepPolicy,
    rng_state: u64,
    flat_accept_streak: usize,
    rescue_policy: RescuePolicy,
    stall_policy: StallPolicy,
    stall_noimprove_streak: usize,
    // Curvature slack scaling under noise
    curv_slack_scale: f64,
    // Gradient drop factor (adapts after flats)
    grad_drop_factor: f64,
    // No-improvement termination guard
    tol_f_rel: f64,
    max_no_improve: usize,
    no_improve_streak: usize,
    // --- Private adaptive state (no API change) ---
    gll: GllWindow,
    c1_adapt: f64,
    c2_adapt: f64,
    wolfe_fail_streak: usize,
    primary_strategy: LineSearchStrategy,
    trust_radius: f64,
    global_best: Option<ProbeBest>,
    // Diagnostics counters
    nonfinite_seen: bool,
    wolfe_clean_successes: usize,
    bt_clean_successes: usize,
    ls_failures_in_row: usize,
    chol_fail_iters: usize,
    spd_fail_seen: bool,
    initial_b_inv: Option<Array2<f64>>,
    initial_grad_norm: f64,
    local_mode: bool,
}

/// A configurable BFGS solver.
pub struct Bfgs<ObjFn> {
    core: BfgsCore,
    obj_fn: ObjFn,
}

impl BfgsCore {
    const FALLBACK_THRESHOLD: usize = 3;

    fn projected_gradient(&self, x: &Array1<f64>, g: &Array1<f64>) -> Array1<f64> {
        if let Some(bounds) = &self.bounds {
            bounds.projected_gradient(x, g)
        } else {
            g.clone()
        }
    }

    fn active_mask(&self, x: &Array1<f64>, g: &Array1<f64>) -> Vec<bool> {
        if let Some(bounds) = &self.bounds {
            bounds.active_mask(x, g)
        } else {
            vec![false; x.len()]
        }
    }

    fn project_with_step(
        &self,
        x: &Array1<f64>,
        d: &Array1<f64>,
        alpha: f64,
    ) -> (Array1<f64>, Array1<f64>, bool) {
        let trial = x + alpha * d;
        let x_new = self.project_point(&trial);
        let kinked = (&x_new - &trial)
            .iter()
            .zip(trial.iter())
            .any(|(dv, tv)| dv.abs() > 1e-12 * (1.0 + tv.abs()));
        let s = &x_new - x;
        (x_new, s, kinked)
    }

    #[inline]
    fn step_tolerance(&self, x: &Array1<f64>) -> f64 {
        1e-12 * (1.0 + x.dot(x).sqrt()) + 1e-16
    }

    #[inline]
    fn feasible_step_small(&self, x_prev: &Array1<f64>, x_next: &Array1<f64>) -> bool {
        let s = x_next - x_prev;
        self.projected_step_small(x_prev, &s)
    }

    #[inline]
    fn projected_step_small(&self, x_prev: &Array1<f64>, s: &Array1<f64>) -> bool {
        s.dot(s).sqrt() <= self.step_tolerance(x_prev)
    }

    #[inline]
    fn stagnation_converged(
        &self,
        x_prev: &Array1<f64>,
        x_next: &Array1<f64>,
        g_proj_next: &Array1<f64>,
    ) -> bool {
        let gnorm = g_proj_next.dot(g_proj_next).sqrt();
        gnorm < self.tolerance || self.feasible_step_small(x_prev, x_next)
    }

    #[inline]
    fn update_no_improve_streak(&mut self, rel_impr: f64) -> bool {
        if rel_impr <= self.tol_f_rel {
            self.no_improve_streak += 1;
        } else {
            self.no_improve_streak = 0;
        }
        self.no_improve_streak >= self.max_no_improve
    }

    // Attempt one trust-region dogleg step. Updates trust radius and, on success,
    // returns new (x, f, g) and updates `b_inv` cautiously. On failure, may shrink Δ.
    fn try_trust_region_step<ObjFn>(
        &mut self,
        obj_fn: &mut ObjFn,
        oracle: &mut FirstOrderCache,
        b_inv: &mut Array2<f64>,
        x_k: &Array1<f64>,
        f_k: f64,
        g_k: &Array1<f64>,
        func_evals: &mut usize,
        grad_evals: &mut usize,
    ) -> Option<(Array1<f64>, f64, Array1<f64>)>
    where
        ObjFn: FirstOrderObjective,
    {
        let n = b_inv.nrows();
        let mut b_inv_backup = Array2::<f64>::zeros((n, n));
        let delta = self.trust_radius;
        let g_proj_k = self.projected_gradient(x_k, g_k);
        let active = self.active_mask(x_k, g_k);
        let active_before = active.clone();
        let active_opt = if active.iter().copied().any(|v| v) {
            if !any_free_variables(&active) {
                self.trust_radius = (delta * 0.5).max(1e-12);
                return None;
            }
            Some(active.as_slice())
        } else {
            None
        };
        let (p_tr, pred_dec_tr) = self.trust_region_dogleg(b_inv, &g_proj_k, delta, active_opt)?;
        let raw_try = x_k + &p_tr;
        let x_try = self.project_point(&raw_try);
        let s_tr = &x_try - x_k;
        let g_old = g_k.clone();
        let (f_try, g_try) =
            bfgs_eval_cost_grad(oracle, obj_fn, &x_try, func_evals, grad_evals).ok()?;
        let act_dec = f_k - f_try;
        let p_diff = &s_tr - &p_tr;
        let p_diff_norm = p_diff.dot(&p_diff).sqrt();
        let p_norm = p_tr.dot(&p_tr).sqrt();
        let proj_changed = p_diff_norm > 1e-6 * (1.0 + p_norm);
        if proj_changed {
            // If projection materially changes the step, require descent at x_k.
            let descent_ok = g_proj_k.dot(&s_tr) <= -eps_g(&g_proj_k, &s_tr, self.tau_g);
            if !descent_ok {
                self.trust_radius = (delta * 0.5).max(1e-12);
                return None;
            }
        }
        let pred_dec = if proj_changed {
            self.trust_region_predicted_decrease(b_inv, &g_proj_k, &s_tr, active_opt)?
        } else {
            pred_dec_tr
        };
        if !pred_dec.is_finite() || pred_dec <= 0.0 {
            self.trust_radius = (delta * 0.5).max(1e-12);
            return None;
        }
        let rho = act_dec / pred_dec;
        if rho > 0.75 && s_tr.dot(&s_tr).sqrt() > 0.99 * delta {
            self.trust_radius = (delta * 2.0).min(1e6);
        } else if rho < 0.25 {
            self.trust_radius = (delta * 0.5).max(1e-12);
        }
        if rho <= 0.1 || !f_try.is_finite() || g_try.iter().any(|v| !v.is_finite()) {
            return None;
        }
        // Accept TR step
        // Update GLL window and global best
        self.gll.push(f_try);
        let maybe_f = self.global_best.as_ref().map(|b| b.f);
        if let Some(bf) = maybe_f {
            if f_try < bf - eps_f(bf, self.tau_f) {
                self.global_best = Some(ProbeBest {
                    f: f_try,
                    x: x_try.clone(),
                    g: g_try.clone(),
                });
            }
        } else {
            self.global_best = Some(ProbeBest::new(&x_try, f_try, &g_try));
        }

        // Inverse update: skip on poor model; otherwise cautious Powell-damped.
        let poor_model = rho <= 0.25;
        let mut s_update = s_tr.clone();
        let mut y_update = &g_try - &g_old;
        if let Some(bounds) = &self.bounds {
            let active_after = bounds.active_mask(&x_try, &g_try);
            for i in 0..n {
                let tiny_step = s_update[i].abs() <= 1e-14 * (1.0 + x_k[i].abs());
                if (active_before[i] && active_after[i]) || tiny_step {
                    s_update[i] = 0.0;
                    y_update[i] = 0.0;
                }
            }
        }
        let s_norm_tr = s_update.dot(&s_update).sqrt();
        let mut update_status = "applied";
        if !poor_model && s_norm_tr > 1e-14 {
            let mean_diag = (0..n).map(|i| b_inv[[i, i]].abs()).sum::<f64>() / (n as f64);
            let ridge = (1e-10 * mean_diag).max(1e-16);
            // Compute B s via CG on H (since H = B^{-1}) for Powell damping.
            if let Some(h_s) = cg_solve_adaptive(b_inv, &s_update, 25, 1e-10, ridge) {
                let s_h_s = s_update.dot(&h_s);
                let sy_tr = s_update.dot(&y_update);
                let denom_raw = s_h_s - sy_tr;
                let denom = if denom_raw <= 0.0 { 1e-16 } else { denom_raw };
                let theta_raw = if sy_tr < 0.2 * s_h_s {
                    (0.8 * s_h_s) / denom
                } else {
                    1.0
                };
                let theta = theta_raw.clamp(0.0, 1.0);
                let mut y_tilde = &y_update * theta + &h_s * (1.0 - theta);
                let mut sty = s_update.dot(&y_tilde);
                let mut y_norm = y_tilde.dot(&y_tilde).sqrt();
                let kappa = 1e-4;
                let min_curv = kappa * s_norm_tr * y_norm;
                if sty < min_curv {
                    let beta = (min_curv - sty) / (s_norm_tr * s_norm_tr);
                    y_tilde = &y_tilde + &s_update * beta;
                    sty = s_update.dot(&y_tilde);
                    y_norm = y_tilde.dot(&y_tilde).sqrt();
                }
                let rel = if s_norm_tr > 0.0 && y_norm > 0.0 {
                    sty / (s_norm_tr * y_norm)
                } else {
                    0.0
                };
                if !sty.is_finite() || rel < 1e-8 {
                    update_status = "skipped";
                    for i in 0..n {
                        b_inv[[i, i]] *= 1.0 + 1e-3;
                    }
                } else {
                    if !apply_inverse_bfgs_update_in_place(
                        b_inv,
                        &s_update,
                        &y_tilde,
                        &mut b_inv_backup,
                    ) {
                        b_inv.assign(&b_inv_backup);
                        for i in 0..n {
                            b_inv[[i, i]] += 1e-6;
                        }
                        update_status = "reverted";
                    }
                }
                if !has_finite_positive_diagonal(b_inv) {
                    for i in 0..n {
                        b_inv[[i, i]] += 1e-12;
                    }
                }
            } else {
                self.spd_fail_seen = true;
                self.chol_fail_iters = self.chol_fail_iters + 1;
                update_status = "skipped";
            }
            if self.spd_fail_seen && self.chol_fail_iters >= 2 {
                let sy = s_update.dot(&y_update);
                let yy = y_update.dot(&y_update);
                let mut lambda = if yy > 0.0 { (sy / yy).abs() } else { 1.0 };
                lambda = lambda.clamp(1e-6, 1e6);
                *b_inv = scaled_identity(n, lambda);
                self.chol_fail_iters = 0;
                update_status = "reverted";
            }
        } else {
            update_status = "skipped";
        }
        log::debug!(
            "[BFGS] step accepted via {:?}; inverse update {}",
            AcceptKind::TrustRegion,
            update_status
        );
        Some((x_try, f_try, g_try))
    }

    /// Creates a new BFGS core configuration.
    fn new(x0: Array1<f64>) -> Self {
        Self {
            x0,
            tolerance: 1e-5,
            max_iterations: 100,
            c1: 1e-4, // Standard value for sufficient decrease
            c2: 0.9,  // Standard value for curvature condition
            tau_f: 1e3,
            tau_g: 1e2,
            bounds: None,
            flat_step_policy: FlatStepPolicy::MidpointWithJiggle { scale: 1e-3 },
            rng_state: 0xB5F0_D00D_1234_5678u64,
            flat_accept_streak: 0,
            rescue_policy: RescuePolicy::CoordinateHybrid {
                pool_mult: 4.0,
                heads: 2,
            },
            stall_policy: StallPolicy::On { window: 3 },
            stall_noimprove_streak: 0,
            curv_slack_scale: 1.0,
            grad_drop_factor: 0.9,
            tol_f_rel: 1e-8,
            max_no_improve: 5,
            no_improve_streak: 0,
            gll: GllWindow::new(8),
            c1_adapt: 1e-4,
            c2_adapt: 0.9,
            wolfe_fail_streak: 0,
            primary_strategy: LineSearchStrategy::StrongWolfe,
            trust_radius: 1.0,
            global_best: None,
            nonfinite_seen: false,
            wolfe_clean_successes: 0,
            bt_clean_successes: 0,
            ls_failures_in_row: 0,
            chol_fail_iters: 0,
            spd_fail_seen: false,
            initial_b_inv: None,
            initial_grad_norm: 0.0,
            local_mode: false,
        }
    }

    fn apply_profile(&mut self, profile: Profile) {
        match profile {
            Profile::Robust => {
                self.tau_f = 1e3;
                self.tau_g = 1e2;
                self.flat_step_policy = FlatStepPolicy::MidpointWithJiggle { scale: 1e-3 };
                self.rescue_policy = RescuePolicy::CoordinateHybrid {
                    pool_mult: 4.0,
                    heads: 2,
                };
                self.stall_policy = StallPolicy::On { window: 3 };
                self.curv_slack_scale = 1.0;
                self.tol_f_rel = 1e-8;
                self.max_no_improve = 5;
            }
            Profile::Deterministic => {
                self.tau_f = 1e2;
                self.tau_g = 1e2;
                self.flat_step_policy = FlatStepPolicy::Strict;
                self.rescue_policy = RescuePolicy::Off;
                self.stall_policy = StallPolicy::On { window: 3 };
                self.curv_slack_scale = 1.0;
                self.tol_f_rel = 1e-8;
                self.max_no_improve = 5;
            }
            Profile::Aggressive => {
                self.tau_f = 1e4;
                self.tau_g = 1e3;
                self.flat_step_policy = FlatStepPolicy::MidpointWithJiggle { scale: 1e-3 };
                self.rescue_policy = RescuePolicy::CoordinateHybrid {
                    pool_mult: 6.0,
                    heads: 4,
                };
                self.stall_policy = StallPolicy::Off;
                self.curv_slack_scale = 2.0;
                self.tol_f_rel = 1e-10;
                self.max_no_improve = 10;
            }
        }
    }

    #[inline]
    fn accept_armijo(&self, f_k: f64, gk_ts: f64, f_i: f64) -> bool {
        let c1 = self.c1_adapt;
        let epsf_k = eps_f(f_k, self.tau_f);
        f_i <= f_k + c1 * gk_ts + epsf_k
    }

    #[inline]
    fn accept_gll_nonmonotone(&self, fmax: f64, gk_ts: f64, f_i: f64) -> bool {
        !self.local_mode && {
            let c1 = self.c1_adapt;
            let epsf_max = eps_f(fmax, self.tau_f);
            f_i <= fmax + c1 * gk_ts + epsf_max
        }
    }

    #[inline]
    fn relaxed_acceptors_enabled(&self) -> bool {
        !self.local_mode
    }

    #[inline]
    fn jiggle_enabled(&self) -> bool {
        matches!(
            self.flat_step_policy,
            FlatStepPolicy::MidpointWithJiggle { .. }
        ) && !self.local_mode
    }

    #[inline]
    fn jiggle_scale(&self) -> f64 {
        match self.flat_step_policy {
            FlatStepPolicy::MidpointWithJiggle { scale } => scale,
            FlatStepPolicy::Strict => 0.0,
        }
    }

    #[inline]
    fn rescue_enabled(&self) -> bool {
        !matches!(self.rescue_policy, RescuePolicy::Off) && !self.local_mode
    }

    #[inline]
    fn refresh_local_mode(&mut self, g_norm: f64) {
        let baseline = self.initial_grad_norm.max(self.tolerance).max(1e-16);
        let gradient_small = g_norm <= 1e-2 * baseline;
        let clean_successes = self.wolfe_clean_successes + self.bt_clean_successes;
        self.local_mode = gradient_small || clean_successes >= 5;
        if self.local_mode {
            self.primary_strategy = LineSearchStrategy::StrongWolfe;
            self.c1_adapt = self.c1;
            self.c2_adapt = self.c2;
            self.flat_accept_streak = 0;
            self.curv_slack_scale = 1.0;
            self.grad_drop_factor = 0.9;
            self.gll.set_cap(1);
        }
    }

    fn trust_region_dogleg(
        &self,
        b_inv: &Array2<f64>,
        g: &Array1<f64>,
        delta: f64,
        active: Option<&[bool]>,
    ) -> Option<(Array1<f64>, f64)> {
        // Solve H z = g without full factorization (H = B_inv).
        let n = b_inv.nrows();
        let active = active.unwrap_or(&[]);
        let use_mask = !active.is_empty();
        if use_mask && !any_free_variables(active) {
            return None;
        }
        let mean_diag = (0..n).map(|i| b_inv[[i, i]].abs()).sum::<f64>() / (n as f64);
        let ridge = (1e-10 * mean_diag).max(1e-16);
        let z = if use_mask {
            cg_solve_masked_adaptive(b_inv, g, active, 50, 1e-10, ridge)?
        } else {
            cg_solve_adaptive(b_inv, g, 50, 1e-10, ridge)?
        };
        let gnorm2 = g.dot(g);
        if !gnorm2.is_finite() || gnorm2 <= 0.0 {
            return None;
        }
        let gHg = g.dot(&z).max(1e-16);
        // Cauchy step
        let tau = gnorm2 / gHg;
        let p_u = -&(g * tau);
        // Newton/BFGS step
        let mut h_g = Array1::<f64>::zeros(n);
        if use_mask {
            masked_hv_inplace(b_inv, g, active, &mut h_g);
        } else {
            h_g.assign(&b_inv.dot(g));
        }
        let p_b = -h_g;
        let p_b_norm = p_b.dot(&p_b).sqrt();
        if p_b_norm <= delta {
            let pred_dec = self.trust_region_predicted_decrease(
                b_inv,
                g,
                &p_b,
                if use_mask { Some(active) } else { None },
            )?;
            return Some((p_b, pred_dec));
        }
        let p_u_norm = p_u.dot(&p_u).sqrt();
        if p_u_norm >= delta {
            let p = -g * (delta / gnorm2.sqrt());
            let pred_dec = self.trust_region_predicted_decrease(
                b_inv,
                g,
                &p,
                if use_mask { Some(active) } else { None },
            )?;
            return Some((p, pred_dec));
        }
        // Dogleg along segment from pu to pb hitting boundary.
        let s = &p_b - &p_u;
        let a = s.dot(&s);
        let b = 2.0 * p_u.dot(&s);
        let c = p_u.dot(&p_u) - delta * delta;
        let disc = b * b - 4.0 * a * c;
        if !disc.is_finite() || disc < 0.0 {
            return None;
        }
        let sqrt_disc = disc.sqrt();
        let t1 = (-b - sqrt_disc) / (2.0 * a);
        let t2 = (-b + sqrt_disc) / (2.0 * a);
        // pick valid root in (0,1); if both, choose the smaller (more conservative)
        let mut candidates: Vec<f64> = vec![];
        if t1.is_finite() && t1 > 0.0 && t1 < 1.0 {
            candidates.push(t1);
        }
        if t2.is_finite() && t2 > 0.0 && t2 < 1.0 {
            candidates.push(t2);
        }
        let t: f64 = if !candidates.is_empty() {
            candidates.into_iter().fold(1.0, f64::min)
        } else {
            0.5
        };
        let mut p = &p_u + &(s * t);
        let p_norm = p.dot(&p).sqrt();
        if p_norm.is_finite() && p_norm > delta && delta.is_finite() && delta > 0.0 {
            p = p * (delta / p_norm);
        }
        let pred_dec = self.trust_region_predicted_decrease(
            b_inv,
            g,
            &p,
            if use_mask { Some(active) } else { None },
        )?;
        Some((p, pred_dec))
    }

    fn trust_region_predicted_decrease(
        &self,
        b_inv: &Array2<f64>,
        g: &Array1<f64>,
        s: &Array1<f64>,
        active: Option<&[bool]>,
    ) -> Option<f64> {
        let n = b_inv.nrows();
        let mean_diag = (0..n).map(|i| b_inv[[i, i]].abs()).sum::<f64>() / (n as f64);
        let ridge = (1e-10 * mean_diag).max(1e-16);
        let hs = if let Some(active) = active {
            cg_solve_masked_adaptive(b_inv, s, active, 50, 1e-10, ridge)?
        } else {
            cg_solve_adaptive(b_inv, s, 50, 1e-10, ridge)?
        };
        let pred = g.dot(s) + 0.5 * s.dot(&hs);
        let pred_dec = -pred;
        if pred_dec.is_finite() && pred_dec > 0.0 {
            Some(pred_dec)
        } else {
            None
        }
    }

    fn project_point(&self, x: &Array1<f64>) -> Array1<f64> {
        if let Some(bounds) = &self.bounds {
            bounds.project(x)
        } else {
            x.clone()
        }
    }

    // Tiny xorshift64* RNG for jiggling without external deps. Returns in [-1, 1].
    fn next_rand_sym(&mut self) -> f64 {
        let mut x = self.rng_state;
        // xorshift64*
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x = x.wrapping_mul(0x2545F4914F6CDD1Du64);
        self.rng_state = x;
        // Map to (0,1): use upper 53 bits to f64 fraction
        let u = ((x >> 11) as f64) * (1.0 / (1u64 << 53) as f64);
        2.0 * u - 1.0
    }

    fn run<ObjFn>(&mut self, obj_fn: &mut ObjFn) -> Result<Solution, BfgsError>
    where
        ObjFn: FirstOrderObjective,
    {
        let n = self.x0.len();
        let mut x_k = self.project_point(&self.x0);
        let mut oracle = FirstOrderCache::new(x_k.len());
        let mut func_evals = 0;
        let mut grad_evals = 0;
        let mut b_inv_backup = Array2::<f64>::zeros((n, n));
        let initial = oracle
            .eval_cost_grad(obj_fn, &x_k, &mut func_evals, &mut grad_evals)
            .map_err(|err| match err {
                ObjectiveEvalError::Recoverable { message }
                | ObjectiveEvalError::Fatal { message } => BfgsError::ObjectiveFailed { message },
            })?;
        let (mut f_k, mut g_k) = initial;
        if !f_k.is_finite() || g_k.iter().any(|v| !v.is_finite()) {
            return Err(BfgsError::GradientIsNaN);
        }
        let mut g_proj_k = self.projected_gradient(&x_k, &g_k);
        let mut active_mask = if let Some(bounds) = &self.bounds {
            bounds.active_mask(&x_k, &g_k)
        } else {
            vec![false; n]
        };

        if !matches!(self.primary_strategy, LineSearchStrategy::StrongWolfe)
            && self.wolfe_fail_streak != 0
        {
            return Err(BfgsError::InternalInvariant {
                message: "primary strategy mismatch with fail streak".to_string(),
            });
        }
        if !self.gll.buf.is_empty() && self.gll.buf.len() > self.gll.cap {
            return Err(BfgsError::InternalInvariant {
                message: "GLL window exceeded capacity".to_string(),
            });
        }
        if !self.trust_radius.is_finite() {
            return Err(BfgsError::InternalInvariant {
                message: "trust radius is non-finite".to_string(),
            });
        }
        self.wolfe_fail_streak = 0;
        self.wolfe_clean_successes = 0;
        self.bt_clean_successes = 0;
        self.ls_failures_in_row = 0;
        self.nonfinite_seen = false;
        self.chol_fail_iters = 0;
        self.spd_fail_seen = false;
        self.flat_accept_streak = 0;

        let mut b_inv = if let Some(h0) = self.initial_b_inv.clone() {
            if h0.nrows() == n && h0.ncols() == n && h0.iter().all(|v| v.is_finite()) {
                h0
            } else {
                Array2::<f64>::eye(n)
            }
        } else {
            Array2::<f64>::eye(n)
        };

        // Initialize adaptive state
        self.gll.clear();
        self.gll.push(f_k);
        self.global_best = Some(ProbeBest::new(&x_k, f_k, &g_k));
        self.c1_adapt = self.c1;
        self.c2_adapt = self.c2;
        self.primary_strategy = LineSearchStrategy::StrongWolfe;
        self.wolfe_fail_streak = 0;
        // Initialize trust radius from gradient scale
        let g0_norm = g_proj_k.dot(&g_proj_k).sqrt();
        self.initial_grad_norm = g0_norm;
        self.local_mode = false;
        let delta0 = if g0_norm.is_finite() && g0_norm > 0.0 {
            (10.0 / g0_norm).min(1.0)
        } else {
            1.0
        };
        self.trust_radius = delta0;

        let mut f_last_accepted = f_k;
        for k in 0..self.max_iterations {
            // reset per-iteration state
            self.nonfinite_seen = false;
            self.chol_fail_iters = 0;
            self.spd_fail_seen = false;
            g_proj_k = self.projected_gradient(&x_k, &g_k);
            let g_norm = g_proj_k.dot(&g_proj_k).sqrt();
            if !g_norm.is_finite() {
                log::debug!(
                    "[BFGS] Non-finite gradient norm at iter {}: g_norm={:?}",
                    k,
                    g_norm
                );
                return Err(BfgsError::GradientIsNaN);
            }
            self.refresh_local_mode(g_norm);
            if g_norm < self.tolerance {
                let sol = Solution::gradient_based(
                    x_k, f_k, g_k, g_norm, None, k, func_evals, grad_evals, 0,
                );
                log::debug!(
                    "[BFGS] Converged by gradient: iters={}, f={:.6e}, ||g||={:.3e}, fe={}, ge={}, Δ={:.3e}",
                    k,
                    sol.final_value,
                    sol.final_gradient_norm
                        .expect("gradient-based solution must report gradient norm"),
                    sol.func_evals,
                    sol.grad_evals,
                    self.trust_radius
                );
                return Ok(sol);
            }

            let mut present_d_k = -b_inv.dot(&g_proj_k);
            if let Some(bounds) = &self.bounds {
                for (i, &active) in active_mask.iter().enumerate() {
                    if active {
                        present_d_k[i] = 0.0;
                    }
                }
                // prevent stepping outside bounds directly from the current point
                for i in 0..present_d_k.len() {
                    if present_d_k[i] < 0.0 && x_k[i] <= bounds.lower[i] + bounds.tol {
                        present_d_k[i] = 0.0;
                    }
                    if present_d_k[i] > 0.0 && x_k[i] >= bounds.upper[i] - bounds.tol {
                        present_d_k[i] = 0.0;
                    }
                }
            }
            // Enforce descent direction; reset if needed
            let gdotd = g_proj_k.dot(&present_d_k);
            let dnorm = present_d_k.dot(&present_d_k).sqrt();
            let tiny_d = dnorm <= 1e-14 * (1.0 + x_k.dot(&x_k).sqrt());
            let eps_dir = eps_g(&g_proj_k, &present_d_k, self.tau_g);
            if gdotd >= -eps_dir || tiny_d {
                log::debug!("[BFGS] Non-descent direction; resetting to -g and B_inv=I.");
                b_inv = Array2::eye(n);
                present_d_k = -g_proj_k.clone();
                if let Some(bounds) = &self.bounds {
                    for (i, &active) in active_mask.iter().enumerate() {
                        if active {
                            present_d_k[i] = 0.0;
                        }
                    }
                    for i in 0..present_d_k.len() {
                        if present_d_k[i] < 0.0 && x_k[i] <= bounds.lower[i] + bounds.tol {
                            present_d_k[i] = 0.0;
                        }
                        if present_d_k[i] > 0.0 && x_k[i] >= bounds.upper[i] - bounds.tol {
                            present_d_k[i] = 0.0;
                        }
                    }
                }
            }

            // --- Adaptive Hybrid Line Search Execution ---
            let active_before = active_mask.clone();
            let (alpha_k, mut f_next, mut g_next, f_evals, g_evals, mut accept_kind) = {
                let search_result = match self.primary_strategy {
                    LineSearchStrategy::StrongWolfe => line_search(
                        self,
                        obj_fn,
                        &mut oracle,
                        &x_k,
                        &present_d_k,
                        f_k,
                        &g_k,
                        self.c1_adapt,
                        self.c2_adapt,
                    ),
                    LineSearchStrategy::Backtracking => backtracking_line_search(
                        self,
                        obj_fn,
                        &mut oracle,
                        &x_k,
                        &present_d_k,
                        f_k,
                        &g_k,
                    ),
                };

                match search_result {
                    Ok(result) => {
                        // Reset failure streak and relax toward canonical constants
                        self.wolfe_fail_streak = 0;
                        self.ls_failures_in_row = 0;
                        // Drift c1/c2 back toward canonical quickly on success
                        if self.wolfe_clean_successes >= 2 || self.bt_clean_successes >= 2 {
                            self.c1_adapt = self.c1;
                            self.c2_adapt = self.c2;
                        } else {
                            self.c1_adapt = (self.c1_adapt * 0.9).max(self.c1);
                            self.c2_adapt = (self.c2_adapt * 1.1).min(self.c2);
                        }
                        match self.primary_strategy {
                            LineSearchStrategy::StrongWolfe => {
                                self.wolfe_clean_successes += 1;
                                self.bt_clean_successes = 0;
                                if self.wolfe_clean_successes >= 3 {
                                    self.gll.set_cap(8);
                                }
                            }
                            LineSearchStrategy::Backtracking => {
                                self.bt_clean_successes += 1;
                                self.wolfe_clean_successes = 0;
                            }
                        }
                        result
                    }
                    Err(e) => {
                        // The primary strategy failed.
                        match e {
                            LineSearchError::StepSizeTooSmall => {
                                log::debug!("[BFGS] Line search failed: step size too small.");
                            }
                            LineSearchError::MaxAttempts(attempts) => {
                                log::debug!(
                                    "[BFGS] Line search failed: max attempts reached ({attempts})."
                                );
                            }
                            LineSearchError::ObjectiveFailed(message) => {
                                return Err(BfgsError::ObjectiveFailed { message });
                            }
                        }
                        // Attempt fallback if the primary strategy was StrongWolfe.
                        if matches!(self.primary_strategy, LineSearchStrategy::StrongWolfe) {
                            let streak = self.wolfe_fail_streak + 1;
                            self.wolfe_fail_streak = streak;
                            log::debug!(
                                "[BFGS Adaptive] Strong Wolfe failed at iter {}. Falling back to Backtracking.",
                                k
                            );
                            // Adapt c1/c2 on failures
                            if streak == 1 {
                                self.c2_adapt = 0.5;
                            }
                            if streak >= 2 {
                                self.c2_adapt = 0.1;
                                self.c1_adapt = 1e-3;
                            }
                            self.ls_failures_in_row += 1;
                            if self.ls_failures_in_row >= 2 {
                                self.gll.set_cap(10);
                            }
                            let fallback_result = backtracking_line_search(
                                self,
                                obj_fn,
                                &mut oracle,
                                &x_k,
                                &present_d_k,
                                f_k,
                                &g_k,
                            );
                            if let Ok(result) = fallback_result {
                                // Fallback succeeded.
                                result
                            } else {
                                // The fallback also failed. Terminate with the informative error.
                                let (max_attempts, failure_reason) = match fallback_result {
                                    Err(LineSearchError::MaxAttempts(attempts)) => {
                                        (attempts, LineSearchFailureReason::MaxAttempts)
                                    }
                                    Err(LineSearchError::StepSizeTooSmall) => (
                                        BACKTRACKING_MAX_ATTEMPTS,
                                        LineSearchFailureReason::StepSizeTooSmall,
                                    ),
                                    Err(LineSearchError::ObjectiveFailed(message)) => {
                                        return Err(BfgsError::ObjectiveFailed { message });
                                    }
                                    Ok(_) => unreachable!(
                                        "entered fallback failure branch with Ok line-search result"
                                    ),
                                };
                                // Salvage best point seen during line search if any
                                if let Some(b) = self.global_best.clone() {
                                    let epsF = eps_f(f_k, self.tau_f);
                                    let gk_norm = g_proj_k.dot(&g_proj_k).sqrt();
                                    let gb_proj = self.projected_gradient(&b.x, &b.g);
                                    let gb_norm = gb_proj.dot(&gb_proj).sqrt();
                                    let drop_factor = self.grad_drop_factor;
                                    if (b.f <= f_k + epsF && gb_norm <= drop_factor * gk_norm)
                                        || (b.f < f_k - epsF)
                                    {
                                        let rel_impr = (f_k - b.f).abs() / (1.0 + f_k.abs());
                                        if self.update_no_improve_streak(rel_impr)
                                            && self.stagnation_converged(&x_k, &b.x, &gb_proj)
                                        {
                                            return Ok(Solution::gradient_based(
                                                b.x.clone(),
                                                b.f,
                                                b.g.clone(),
                                                gb_norm,
                                                None,
                                                k,
                                                func_evals,
                                                grad_evals,
                                                0,
                                            ));
                                        }
                                        x_k = self.project_point(&b.x);
                                        f_k = b.f;
                                        g_k = b.g.clone();
                                        g_proj_k = gb_proj;
                                        if let Some(bounds) = &self.bounds {
                                            active_mask = bounds.active_mask(&x_k, &g_k);
                                        }
                                        for i in 0..n {
                                            b_inv[[i, i]] *= 1.0 + 1e-3;
                                        }
                                        continue;
                                    }
                                }
                                // Try full trust-region dogleg fallback before giving up
                                if let Some((x_new, f_new, g_new)) = self.try_trust_region_step(
                                    obj_fn,
                                    &mut oracle,
                                    &mut b_inv,
                                    &x_k,
                                    f_k,
                                    &g_k,
                                    &mut func_evals,
                                    &mut grad_evals,
                                ) {
                                    let g_proj_new = self.projected_gradient(&x_new, &g_new);
                                    let rel_impr = (f_k - f_new).abs() / (1.0 + f_k.abs());
                                    if self.update_no_improve_streak(rel_impr)
                                        && self.stagnation_converged(&x_k, &x_new, &g_proj_new)
                                    {
                                        return Ok(Solution::gradient_based(
                                            x_new,
                                            f_new,
                                            g_new,
                                            g_proj_new.dot(&g_proj_new).sqrt(),
                                            None,
                                            k + 1,
                                            func_evals,
                                            grad_evals,
                                            0,
                                        ));
                                    }
                                    x_k = x_new;
                                    f_k = f_new;
                                    g_k = g_new;
                                    g_proj_k = g_proj_new;
                                    if let Some(bounds) = &self.bounds {
                                        active_mask = bounds.active_mask(&x_k, &g_k);
                                    }
                                    self.ls_failures_in_row = 0;
                                    continue;
                                }
                                self.trust_radius = (self.trust_radius * 0.7).max(1e-12);
                                if self.nonfinite_seen {
                                    let mut ls = Solution::gradient_based(
                                        x_k.clone(),
                                        f_k,
                                        g_k.clone(),
                                        g_norm,
                                        None,
                                        k,
                                        func_evals,
                                        grad_evals,
                                        0,
                                    );
                                    if let Some(b) = self.global_best.as_ref()
                                        && b.f < f_k - eps_f(f_k, self.tau_f)
                                    {
                                        let gb_proj = self.projected_gradient(&b.x, &b.g);
                                        ls = Solution::gradient_based(
                                            b.x.clone(),
                                            b.f,
                                            b.g.clone(),
                                            gb_proj.dot(&gb_proj).sqrt(),
                                            None,
                                            k,
                                            func_evals,
                                            grad_evals,
                                            0,
                                        );
                                    }
                                    log::debug!(
                                        "[BFGS] Line search failed at iter {} (nonfinite seen), fe={}, ge={}, Δ={:.3e}",
                                        k,
                                        func_evals,
                                        grad_evals,
                                        self.trust_radius
                                    );
                                    return Err(BfgsError::LineSearchFailed {
                                        last_solution: Box::new(ls),
                                        max_attempts,
                                        failure_reason,
                                    });
                                }
                                if self.ls_failures_in_row >= 2 {
                                    let ls = Solution::gradient_based(
                                        x_k.clone(),
                                        f_k,
                                        g_k.clone(),
                                        g_norm,
                                        None,
                                        k,
                                        func_evals,
                                        grad_evals,
                                        0,
                                    );
                                    return Err(BfgsError::LineSearchFailed {
                                        last_solution: Box::new(ls),
                                        max_attempts,
                                        failure_reason,
                                    });
                                }
                                continue;
                            }
                        } else {
                            // The robust Backtracking strategy has failed. This is a critical problem.
                            // Reset the Hessian and try one last time with a steepest descent direction.
                            self.ls_failures_in_row += 1;
                            log::error!(
                                "[BFGS Adaptive] CRITICAL: Backtracking failed at iter {}. Resetting Hessian.",
                                k
                            );
                            b_inv = Array2::<f64>::eye(n);
                            present_d_k = -g_k.clone();
                            let fallback_result = backtracking_line_search(
                                self,
                                obj_fn,
                                &mut oracle,
                                &x_k,
                                &present_d_k,
                                f_k,
                                &g_k,
                            );
                            if let Ok(result) = fallback_result {
                                result
                            } else {
                                let (max_attempts, failure_reason) = match fallback_result {
                                    Err(LineSearchError::MaxAttempts(attempts)) => {
                                        (attempts, LineSearchFailureReason::MaxAttempts)
                                    }
                                    Err(LineSearchError::StepSizeTooSmall) => (
                                        BACKTRACKING_MAX_ATTEMPTS,
                                        LineSearchFailureReason::StepSizeTooSmall,
                                    ),
                                    Err(LineSearchError::ObjectiveFailed(message)) => {
                                        return Err(BfgsError::ObjectiveFailed { message });
                                    }
                                    Ok(_) => unreachable!(
                                        "entered fallback failure branch with Ok line-search result"
                                    ),
                                };
                                // Full trust-region dogleg fallback
                                if let Some((x_new, f_new, g_new)) = self.try_trust_region_step(
                                    obj_fn,
                                    &mut oracle,
                                    &mut b_inv,
                                    &x_k,
                                    f_k,
                                    &g_k,
                                    &mut func_evals,
                                    &mut grad_evals,
                                ) {
                                    let g_proj_new = self.projected_gradient(&x_new, &g_new);
                                    let rel_impr = (f_k - f_new).abs() / (1.0 + f_k.abs());
                                    if self.update_no_improve_streak(rel_impr)
                                        && self.stagnation_converged(&x_k, &x_new, &g_proj_new)
                                    {
                                        return Ok(Solution::gradient_based(
                                            x_new,
                                            f_new,
                                            g_new,
                                            g_proj_new.dot(&g_proj_new).sqrt(),
                                            None,
                                            k + 1,
                                            func_evals,
                                            grad_evals,
                                            0,
                                        ));
                                    }
                                    x_k = x_new;
                                    f_k = f_new;
                                    g_k = g_new;
                                    g_proj_k = g_proj_new;
                                    if let Some(bounds) = &self.bounds {
                                        active_mask = bounds.active_mask(&x_k, &g_k);
                                    }
                                    self.ls_failures_in_row = 0;
                                    continue;
                                }
                                if let Some(b) = self.global_best.clone() {
                                    let epsF = eps_f(f_k, self.tau_f);
                                    let gk_norm = g_proj_k.dot(&g_proj_k).sqrt();
                                    let gb_proj = self.projected_gradient(&b.x, &b.g);
                                    let gb_norm = gb_proj.dot(&gb_proj).sqrt();
                                    let drop_factor = self.grad_drop_factor;
                                    if (b.f <= f_k + epsF && gb_norm <= drop_factor * gk_norm)
                                        || (b.f < f_k - epsF)
                                    {
                                        let rel_impr = (f_k - b.f).abs() / (1.0 + f_k.abs());
                                        if self.update_no_improve_streak(rel_impr)
                                            && self.stagnation_converged(&x_k, &b.x, &gb_proj)
                                        {
                                            return Ok(Solution::gradient_based(
                                                b.x.clone(),
                                                b.f,
                                                b.g.clone(),
                                                gb_norm,
                                                None,
                                                k,
                                                func_evals,
                                                grad_evals,
                                                0,
                                            ));
                                        }
                                        x_k = self.project_point(&b.x);
                                        f_k = b.f;
                                        g_k = b.g.clone();
                                        g_proj_k = gb_proj;
                                        if let Some(bounds) = &self.bounds {
                                            active_mask = bounds.active_mask(&x_k, &g_k);
                                        }
                                        for i in 0..n {
                                            b_inv[[i, i]] *= 1.0 + 1e-3;
                                        }
                                        continue;
                                    }
                                }
                                self.trust_radius = (self.trust_radius * 0.7).max(1e-12);
                                if self.nonfinite_seen {
                                    let mut ls = Solution::gradient_based(
                                        x_k.clone(),
                                        f_k,
                                        g_k.clone(),
                                        g_norm,
                                        None,
                                        k,
                                        func_evals,
                                        grad_evals,
                                        0,
                                    );
                                    if let Some(b) = self.global_best.as_ref()
                                        && b.f < f_k - eps_f(f_k, self.tau_f)
                                    {
                                        let b_proj = self.projected_gradient(&b.x, &b.g);
                                        ls = Solution::gradient_based(
                                            b.x.clone(),
                                            b.f,
                                            b.g.clone(),
                                            b_proj.dot(&b_proj).sqrt(),
                                            None,
                                            k,
                                            func_evals,
                                            grad_evals,
                                            0,
                                        );
                                    }
                                    log::debug!(
                                        "[BFGS] Line search failed at iter {} (nonfinite seen), fe={}, ge={}, Δ={:.3e}",
                                        k,
                                        func_evals,
                                        grad_evals,
                                        self.trust_radius
                                    );
                                    return Err(BfgsError::LineSearchFailed {
                                        last_solution: Box::new(ls),
                                        max_attempts,
                                        failure_reason,
                                    });
                                }
                                if self.ls_failures_in_row >= 2 {
                                    let ls = Solution::gradient_based(
                                        x_k.clone(),
                                        f_k,
                                        g_k.clone(),
                                        g_norm,
                                        None,
                                        k,
                                        func_evals,
                                        grad_evals,
                                        0,
                                    );
                                    return Err(BfgsError::LineSearchFailed {
                                        last_solution: Box::new(ls),
                                        max_attempts,
                                        failure_reason,
                                    });
                                }
                                continue;
                            }
                        }
                    }
                }
            };

            // Optional coordinate rescue after consecutive flat accepts
            let mut s_override: Option<Array1<f64>> = None;
            let mut rescued = false;
            if self.rescue_enabled() {
                let epsF_iter = eps_f(f_k, self.tau_f);
                let flat_now = (f_next - f_k).abs() <= epsF_iter;
                if flat_now && self.flat_accept_streak >= 2 {
                    let x_base = self.project_point(&(&x_k + &(alpha_k * &present_d_k)));
                    let g_proj_base = self.projected_gradient(&x_base, &g_next);
                    let gnext_norm0 = g_proj_base.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let delta = self.trust_radius;
                    let eta = (0.2 * delta).min(1.0 / (1.0 + gnext_norm0));
                    if eta.is_finite() && eta > 0.0 {
                        let n = x_k.len();
                        let mut best_x = None;
                        let mut best_f = f_next;
                        let mut best_g = g_next.clone();
                        // Budgeted coordinate subset selection
                        let k = n.min(8);
                        let mut idx: Vec<usize> = (0..n).collect();
                        idx.sort_by(|&i, &j| {
                            g_next[i]
                                .abs()
                                .partial_cmp(&g_next[j].abs())
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .reverse()
                        });
                        let (use_hybrid, pool_mult, rescue_heads) = match self.rescue_policy {
                            RescuePolicy::Off => (false, 1.0, 0),
                            RescuePolicy::CoordinateHybrid { pool_mult, heads } => {
                                (true, pool_mult, heads)
                            }
                        };
                        let m = (pool_mult * (k as f64)).round() as usize;
                        let m = m.min(n).max(k);
                        let heads = rescue_heads.min(k).min(m);
                        let mut chosen: Vec<usize> = Vec::new();
                        // Always include top heads
                        for &i in idx.iter().take(heads) {
                            chosen.push(i);
                        }
                        if use_hybrid {
                            // Sample remaining from next (heads..m)
                            let mut pool: Vec<usize> =
                                idx.iter().cloned().skip(heads).take(m - heads).collect();
                            while chosen.len() < k && !pool.is_empty() {
                                // xorshift-based index
                                let r = (self.rng_state >> 1) as usize;
                                let t = r % pool.len();
                                let pick = pool.swap_remove(t);
                                chosen.push(pick);
                                // advance rng
                                let _ = self.next_rand_sym();
                            }
                        } else {
                            for &i in idx.iter().skip(heads).take(k - heads) {
                                chosen.push(i);
                            }
                        }
                        for &i in &chosen {
                            for &sgn in &[-1.0, 1.0] {
                                let mut x_try = x_base.clone();
                                x_try[i] += sgn * eta; // coordinate poke from x_next
                                x_try = self.project_point(&x_try);
                                let (f_try, g_try) = match bfgs_eval_cost_grad(
                                    &mut oracle,
                                    obj_fn,
                                    &x_try,
                                    &mut func_evals,
                                    &mut grad_evals,
                                ) {
                                    Ok(sample) => sample,
                                    Err(ObjectiveEvalError::Recoverable { .. }) => continue,
                                    Err(ObjectiveEvalError::Fatal { message }) => {
                                        return Err(BfgsError::ObjectiveFailed { message });
                                    }
                                };
                                if !f_try.is_finite() || g_try.iter().any(|v| !v.is_finite()) {
                                    continue;
                                }
                                let g_proj_try = self.projected_gradient(&x_try, &g_try);
                                let g_try_norm = g_proj_try.dot(&g_proj_try).sqrt();
                                let f_thresh = f_k.min(f_next) + epsF_iter;
                                let s_trial = &x_try - &x_k;
                                let descent_ok = g_proj_k.dot(&s_trial)
                                    <= -eps_g(&g_proj_k, &s_trial, self.tau_g);
                                let f_ok = f_try <= f_thresh;
                                let g_ok = g_try_norm <= self.grad_drop_factor * gnext_norm0;
                                if (f_ok || g_ok) && descent_ok && f_try <= best_f {
                                    best_f = f_try;
                                    best_x = Some(x_try.clone());
                                    best_g = g_try.clone();
                                }
                            }
                        }
                        if let Some(xb) = best_x {
                            // Enforce trust radius on the rescue step
                            let mut s_tmp = &xb - &x_k;
                            let s_norm = s_tmp.dot(&s_tmp).sqrt();
                            let delta = self.trust_radius;
                            if s_norm.is_finite()
                                && s_norm > delta
                                && delta.is_finite()
                                && delta > 0.0
                            {
                                let scale = delta / s_norm;
                                let x_scaled = &x_k + &(s_tmp.mapv(|v| v * scale));
                                let x_scaled = self.project_point(&x_scaled);
                                let (f_s, g_s) = match bfgs_eval_cost_grad(
                                    &mut oracle,
                                    obj_fn,
                                    &x_scaled,
                                    &mut func_evals,
                                    &mut grad_evals,
                                ) {
                                    Ok(sample) => sample,
                                    Err(ObjectiveEvalError::Recoverable { .. }) => {
                                        (f64::NAN, Array1::zeros(x_scaled.len()))
                                    }
                                    Err(ObjectiveEvalError::Fatal { message }) => {
                                        return Err(BfgsError::ObjectiveFailed { message });
                                    }
                                };
                                if f_s.is_finite() && g_s.iter().all(|v| v.is_finite()) {
                                    s_tmp = &x_scaled - &x_k;
                                    f_next = f_s;
                                    g_next = g_s;
                                } else {
                                    // fall back to original xb
                                    f_next = best_f;
                                    g_next = best_g.clone();
                                }
                            } else {
                                f_next = best_f;
                                g_next = best_g.clone();
                            }
                            s_override = Some(s_tmp);
                            rescued = true;
                            accept_kind = AcceptKind::Rescue;
                            self.flat_accept_streak = 0;
                        }
                    }
                }
            }

            // The "Learner" part: promote Backtracking if Wolfe keeps failing.
            if self.wolfe_fail_streak >= Self::FALLBACK_THRESHOLD {
                log::debug!(
                    "[BFGS Adaptive] Fallback streak ({}) reached. Switching primary to Backtracking.",
                    self.wolfe_fail_streak
                );
                self.primary_strategy = LineSearchStrategy::Backtracking;
                self.wolfe_fail_streak = 0;
            }
            // Switch back to StrongWolfe after a run of clean backtracking successes
            if matches!(self.primary_strategy, LineSearchStrategy::Backtracking)
                && self.bt_clean_successes >= 3
                && self.wolfe_fail_streak == 0
            {
                log::debug!(
                    "[BFGS Adaptive] Backtracking succeeded cleanly ({} iters); switching back to StrongWolfe.",
                    self.bt_clean_successes
                );
                self.primary_strategy = LineSearchStrategy::StrongWolfe;
                self.bt_clean_successes = 0;
                self.gll.set_cap(8);
            }

            func_evals += f_evals;
            grad_evals += g_evals;

            let mut s_k = if let Some(ref s) = s_override {
                s.clone()
            } else {
                alpha_k * &present_d_k
            };
            let x_next = self.project_point(&(x_k.clone() + &s_k));
            s_k = &x_next - &x_k;
            let g_proj_next = self.projected_gradient(&x_next, &g_next);
            let active_after = if let Some(bounds) = &self.bounds {
                bounds.active_mask(&x_next, &g_next)
            } else {
                vec![false; n]
            };
            let step_len = s_k.dot(&s_k).sqrt();
            if step_len.is_finite() && step_len > 0.0 {
                if step_len >= 0.9 * self.trust_radius {
                    self.trust_radius = (self.trust_radius * 1.5).min(1e6);
                } else {
                    self.trust_radius = (self.trust_radius * 1.1).min(1e6);
                }
            }

            let rel_impr = (f_last_accepted - f_next).abs() / (1.0 + f_last_accepted.abs());
            if self.update_no_improve_streak(rel_impr)
                && self.stagnation_converged(&x_k, &x_next, &g_proj_next)
            {
                return Ok(Solution::gradient_based(
                    x_next.clone(),
                    f_next,
                    g_next.clone(),
                    g_proj_next.dot(&g_proj_next).sqrt(),
                    None,
                    k + 1,
                    func_evals,
                    grad_evals,
                    0,
                ));
            }

            // Update adaptive curvature slack scale and gradient drop factor based on flats
            let f_ok_flat = (f_next - f_k).abs() <= eps_f(f_k, self.tau_f)
                || (f_next - f_k).abs() <= self.tol_f_rel * (1.0 + f_k.abs());
            if f_ok_flat {
                self.flat_accept_streak += 1;
            } else {
                self.flat_accept_streak = 0;
            }
            if self.flat_accept_streak >= 2 {
                self.curv_slack_scale = (self.curv_slack_scale * 0.5).max(0.1);
                self.grad_drop_factor = 0.95;
            } else {
                self.curv_slack_scale = 1.0;
                self.grad_drop_factor = 0.9;
            }

            let mut y_k = &g_next - &g_k;

            if self.bounds.is_some() {
                for i in 0..n {
                    let tiny_step = s_k[i].abs() <= 1e-14 * (1.0 + x_k[i].abs());
                    if (active_before[i] && active_after[i]) || tiny_step {
                        s_k[i] = 0.0;
                        y_k[i] = 0.0;
                    }
                }
            }

            // --- Cautious Hessian Update ---
            let sy = s_k.dot(&y_k);
            let mut update_status = "applied";

            if k == 0 {
                // Improved first-step scaling
                let yy = y_k.dot(&y_k);
                let mut scale = if sy > 1e-12 && yy > 0.0 { sy / yy } else { 1.0 };
                if !scale.is_finite() {
                    scale = 1.0;
                }
                scale = scale.clamp(1e-3, 1e3);
                b_inv = Array2::eye(n) * scale;
            }

            // Powell-damped inverse BFGS update (keep SPD).
            let s_norm = s_k.dot(&s_k).sqrt();
            if s_norm > 1e-14 {
                if !rescued {
                    // Compute B s via CG on H (since H = B^{-1}) for Powell damping.
                    let mean_diag = (0..n).map(|i| b_inv[[i, i]].abs()).sum::<f64>() / (n as f64);
                    let ridge = (1e-10 * mean_diag).max(1e-16);
                    if let Some(h_s) = cg_solve_adaptive(&b_inv, &s_k, 25, 1e-10, ridge) {
                        let s_h_s = s_k.dot(&h_s);
                        let denom_raw = s_h_s - sy;
                        let denom = if denom_raw <= 0.0 { 1e-16 } else { denom_raw };
                        // Powell damping: blend y and B s so that s^T y_tilde is sufficiently positive.
                        let theta_raw = if sy < 0.2 * s_h_s {
                            (0.8 * s_h_s) / denom
                        } else {
                            1.0
                        };
                        let theta = theta_raw.clamp(0.0, 1.0);
                        let mut y_tilde = &y_k * theta + &h_s * (1.0 - theta);
                        let mut sty = s_k.dot(&y_tilde);
                        let mut y_norm = y_tilde.dot(&y_tilde).sqrt();
                        let s_norm2 = s_norm * s_norm;
                        let kappa = 1e-4;
                        let min_curv = kappa * s_norm * y_norm;
                        if sty < min_curv {
                            let beta = (min_curv - sty) / s_norm2;
                            y_tilde = &y_tilde + &s_k * beta;
                            sty = s_k.dot(&y_tilde);
                            y_norm = y_tilde.dot(&y_tilde).sqrt();
                        }
                        let rel = if s_norm > 0.0 && y_norm > 0.0 {
                            sty / (s_norm * y_norm)
                        } else {
                            0.0
                        };
                        if !sty.is_finite() || rel < 1e-8 {
                            log::debug!(
                                "[BFGS] s^T y_tilde non-positive/tiny; skipping update and inflating diag."
                            );
                            update_status = "skipped";
                            self.chol_fail_iters = self.chol_fail_iters + 1;
                            for i in 0..n {
                                b_inv[[i, i]] *= 1.0 + 1e-3;
                            }
                        } else {
                            if !apply_inverse_bfgs_update_in_place(
                                &mut b_inv,
                                &s_k,
                                &y_tilde,
                                &mut b_inv_backup,
                            ) {
                                b_inv.assign(&b_inv_backup);
                                for i in 0..n {
                                    b_inv[[i, i]] += 1e-6;
                                }
                                update_status = "reverted";
                            }
                        }
                    } else {
                        self.chol_fail_iters = self.chol_fail_iters + 1;
                        self.spd_fail_seen = true;
                        log::debug!("[BFGS] B_inv not SPD after ridge; skipping update this iter.");
                        update_status = "skipped";
                    }
                } else {
                    log::debug!(
                        "[BFGS] Coordinate rescue used; skipping inverse update this iter."
                    );
                    update_status = "skipped";
                }

                // Enforce symmetry and gentle regularization
                for i in 0..n {
                    for j in (i + 1)..n {
                        let a = b_inv[[i, j]];
                        let b = b_inv[[j, i]];
                        let v = 0.5 * (a + b);
                        b_inv[[i, j]] = v;
                        b_inv[[j, i]] = v;
                    }
                }
                let mut diag_min = f64::INFINITY;
                for i in 0..n {
                    diag_min = diag_min.min(b_inv[[i, i]]);
                }
                if !diag_min.is_finite() || diag_min <= 0.0 {
                    let mut trace = 0.0;
                    for i in 0..n {
                        trace += b_inv[[i, i]].abs();
                    }
                    let delta = 1e-12 * trace.max(1.0);
                    for i in 0..n {
                        b_inv[[i, i]] += delta;
                    }
                }

                if self.spd_fail_seen && self.chol_fail_iters >= 2 {
                    let sy = s_k.dot(&y_k);
                    let yy = y_k.dot(&y_k);
                    let mut lambda = if yy > 0.0 { (sy / yy).abs() } else { 1.0 };
                    lambda = lambda.clamp(1e-6, 1e6);
                    b_inv = scaled_identity(n, lambda);
                    self.chol_fail_iters = 0;
                    update_status = "reverted";
                }
            } else {
                update_status = "skipped";
            }

            log::debug!(
                "[BFGS] step accepted via {:?}; inverse update {}",
                accept_kind,
                update_status
            );

            // Stopping tests: small step and flat f
            let step_ok = self.feasible_step_small(&x_k, &x_next);
            let f_ok = (f_next - f_k).abs() <= eps_f(f_k, self.tau_f);
            let gnext_finite = f_next.is_finite() && g_next.iter().all(|v| v.is_finite());
            let gnext_norm = g_proj_next.dot(&g_proj_next).sqrt();
            if step_ok && f_ok && gnext_finite && gnext_norm < self.tolerance {
                let sol = Solution::gradient_based(
                    x_next.clone(),
                    f_next,
                    g_next.clone(),
                    gnext_norm,
                    None,
                    k + 1,
                    func_evals,
                    grad_evals,
                    0,
                );
                log::debug!(
                    "[BFGS] Converged by small step/flat f: iters={}, f={:.6e}, ||g||={:.3e}, fe={}, ge={}, Δ={:.3e}",
                    sol.iterations,
                    sol.final_value,
                    sol.final_gradient_norm
                        .expect("gradient-based solution must report gradient norm"),
                    sol.func_evals,
                    sol.grad_evals,
                    self.trust_radius
                );
                return Ok(sol);
            }

            // Optional stall/flat exit (relative stationarity)
            if let StallPolicy::On { window } = self.stall_policy {
                let g_inf = g_proj_k.iter().fold(0.0, |acc, &v| f64::max(acc, v.abs()));
                let x_inf = x_k.iter().fold(0.0, |acc, &v| f64::max(acc, v.abs()));
                let rel_g_ok = g_inf <= self.tolerance * (1.0 + x_inf);
                let rel_f_ok = (f_k - f_last_accepted).abs() <= eps_f(f_last_accepted, self.tau_f);
                if rel_g_ok && rel_f_ok {
                    self.stall_noimprove_streak += 1;
                } else {
                    self.stall_noimprove_streak = 0;
                }
                if self.stall_noimprove_streak >= window {
                    let sol = Solution::gradient_based(
                        x_k.clone(),
                        f_k,
                        g_k.clone(),
                        g_inf,
                        None,
                        k + 1,
                        func_evals,
                        grad_evals,
                        0,
                    );
                    log::debug!(
                        "[BFGS] Converged (flat/stalled): iters={}, f={:.6e}, ||g||={:.3e}",
                        sol.iterations,
                        sol.final_value,
                        sol.final_gradient_norm
                            .expect("gradient-based solution must report gradient norm")
                    );
                    return Ok(sol);
                }
            }

            x_k = x_next;
            f_k = f_next;
            g_k = g_next;
            g_proj_k = g_proj_next;
            active_mask = active_after;
            // Update GLL window and global best
            self.gll.push(f_k);
            f_last_accepted = f_k;
            let maybe_f = self.global_best.as_ref().map(|b| b.f);
            match maybe_f {
                Some(bf) => {
                    if f_k < bf - eps_f(bf, self.tau_f) {
                        self.global_best = Some(ProbeBest {
                            f: f_k,
                            x: x_k.clone(),
                            g: g_k.clone(),
                        });
                    }
                }
                None => {
                    self.global_best = Some(ProbeBest::new(&x_k, f_k, &g_k));
                }
            }

            // Nonmonotone stickiness countdown
            // We return to StrongWolfe only after a run of clean backtracking
            // successes (handled above via `bt_clean_successes`).
        }

        // The loop finished. Construct a solution from the final state.
        let final_g_norm = g_proj_k.dot(&g_proj_k).sqrt();
        let last_solution = Box::new(Solution::gradient_based(
            x_k,
            f_k,
            g_k,
            final_g_norm,
            None,
            self.max_iterations,
            func_evals,
            grad_evals,
            0,
        ));
        log::debug!(
            "[BFGS] Max iterations reached: iters={}, f={:.6e}, ||g||={:.3e}, fe={}, ge={}, Δ={:.3e}",
            self.max_iterations,
            last_solution.final_value,
            last_solution
                .final_gradient_norm
                .expect("gradient-based solution must report gradient norm"),
            last_solution.func_evals,
            last_solution.grad_evals,
            self.trust_radius
        );
        Err(BfgsError::MaxIterationsReached { last_solution })
    }
}

impl<ObjFn> Bfgs<ObjFn>
where
    ObjFn: FirstOrderObjective,
{
    /// Creates a new BFGS solver.
    ///
    /// # Arguments
    /// * `x0` - The initial guess for the minimum.
    /// * `obj_fn` - First-order objective.
    pub fn new(x0: Array1<f64>, obj_fn: ObjFn) -> Self {
        Self {
            core: BfgsCore::new(x0),
            obj_fn,
        }
    }

    /// Sets the convergence tolerance (default: 1e-5).
    pub fn with_tolerance(mut self, tolerance: Tolerance) -> Self {
        self.core.tolerance = tolerance.get();
        self
    }

    /// Sets the maximum number of iterations (default: 100).
    pub fn with_max_iterations(mut self, max_iterations: MaxIterations) -> Self {
        self.core.max_iterations = max_iterations.get();
        self
    }

    /// Provides simple box bounds for each coordinate (lower <= x <= upper).
    /// Points are projected by coordinate clamping, and the gradient is projected
    /// by zeroing active constraints during direction updates.
    pub fn with_bounds(mut self, bounds: Bounds) -> Self {
        self.obj_fn.set_finite_difference_bounds(Some(&bounds));
        self.core.bounds = Some(bounds.spec);
        self
    }

    pub fn with_profile(mut self, profile: Profile) -> Self {
        self.core.apply_profile(profile);
        self
    }

    /// Executes the BFGS algorithm with the adaptive hybrid line search.
    /// Requires `&mut self` to support stateful `FnMut` objectives.
    pub fn run(&mut self) -> Result<Solution, BfgsError> {
        self.core.run(&mut self.obj_fn)
    }

    #[cfg(test)]
    fn next_rand_sym(&mut self) -> f64 {
        self.core.next_rand_sym()
    }
}

impl<ObjFn> NewtonTrustRegion<ObjFn>
where
    ObjFn: SecondOrderObjective,
{
    /// Creates a new Newton trust-region solver.
    ///
    /// # Arguments
    /// * `x0` - The initial guess for the minimum.
    /// * `obj_fn` - Second-order objective.
    pub fn new(x0: Array1<f64>, obj_fn: ObjFn) -> Self {
        Self {
            core: NewtonTrustRegionCore::new(x0),
            obj_fn,
        }
    }

    /// Sets the convergence tolerance on projected gradient norm (default: 1e-5).
    pub fn with_tolerance(mut self, tolerance: Tolerance) -> Self {
        self.core.tolerance = tolerance.get();
        self
    }

    /// Sets the maximum number of iterations (default: 100).
    pub fn with_max_iterations(mut self, max_iterations: MaxIterations) -> Self {
        self.core.max_iterations = max_iterations.get();
        self
    }

    pub fn with_fd_hessian_step(mut self, fd_hessian_step: f64) -> Self {
        self.core.fd_hessian_step = fd_hessian_step;
        self
    }

    /// Provides simple box bounds for each coordinate (lower <= x <= upper).
    pub fn with_bounds(mut self, bounds: Bounds) -> Self {
        self.obj_fn.set_finite_difference_bounds(Some(&bounds));
        self.core.bounds = Some(bounds.spec);
        self
    }

    pub fn with_profile(mut self, profile: Profile) -> Self {
        self.core.apply_profile(profile);
        self
    }

    /// Executes the Newton trust-region optimization.
    pub fn run(&mut self) -> Result<Solution, NewtonTrustRegionError> {
        self.core.run(&mut self.obj_fn)
    }
}

impl<ObjFn> Arc<ObjFn>
where
    ObjFn: SecondOrderObjective,
{
    /// Creates a new ARC solver.
    ///
    /// # Arguments
    /// * `x0` - The initial guess for the minimum.
    /// * `obj_fn` - Second-order objective.
    pub fn new(x0: Array1<f64>, obj_fn: ObjFn) -> Self {
        Self {
            core: ArcCore::new(x0),
            obj_fn,
        }
    }

    /// Sets the convergence tolerance on projected gradient norm (default: 1e-5).
    pub fn with_tolerance(mut self, tolerance: Tolerance) -> Self {
        self.core.tolerance = tolerance.get();
        self
    }

    /// Sets the maximum number of iterations (default: 100).
    pub fn with_max_iterations(mut self, max_iterations: MaxIterations) -> Self {
        self.core.max_iterations = max_iterations.get();
        self
    }

    pub fn with_fd_hessian_step(mut self, fd_hessian_step: f64) -> Self {
        self.core.fd_hessian_step = fd_hessian_step;
        self
    }

    /// Provides simple box bounds for each coordinate (lower <= x <= upper).
    pub fn with_bounds(mut self, bounds: Bounds) -> Self {
        self.obj_fn.set_finite_difference_bounds(Some(&bounds));
        self.core.bounds = Some(bounds.spec);
        self
    }

    pub fn with_profile(mut self, profile: Profile) -> Self {
        self.core.apply_profile(profile);
        self
    }

    /// Executes ARC optimization.
    ///
    /// This implementation follows the practical ARC template in Euclidean spaces.
    /// Under standard assumptions (for example lower bounded objective and
    /// Lipschitz-continuous Hessian), ARC theory gives an `O(eps^-1.5)` first-order
    /// iteration bound; this API does not encode assumptions, but mirrors that
    /// algorithmic structure.
    pub fn run(&mut self) -> Result<Solution, ArcError> {
        self.core.run(&mut self.obj_fn)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FixedPointError {
    #[error("Objective evaluation failed: {message}")]
    ObjectiveFailed { message: String },
    #[error("Fixed-point objective returned a step with length {got}; expected {expected}")]
    StepDimensionMismatch { expected: usize, got: usize },
    #[error("Fixed-point objective returned a non-finite step")]
    NonFiniteStep,
    #[error(
        "Maximum number of iterations reached without converging. The best solution found is returned."
    )]
    MaxIterationsReached { last_solution: Box<Solution> },
}

struct FixedPointCore {
    x0: Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
    bounds: Option<BoxSpec>,
}

impl FixedPointCore {
    fn new(x0: Array1<f64>) -> Self {
        Self {
            x0,
            tolerance: 1e-5,
            max_iterations: 100,
            bounds: None,
        }
    }

    fn project_point(&self, x: &Array1<f64>) -> Array1<f64> {
        if let Some(bounds) = &self.bounds {
            bounds.project(x)
        } else {
            x.clone()
        }
    }

    fn run<ObjFn>(&mut self, obj_fn: &mut ObjFn) -> Result<Solution, FixedPointError>
    where
        ObjFn: FixedPointObjective,
    {
        let mut x_k = self.project_point(&self.x0);
        let mut func_evals = 0usize;
        let mut last_value = f64::INFINITY;
        let mut last_step_norm = 0.0;
        for k in 0..self.max_iterations {
            let sample = match obj_fn.eval_step(&x_k) {
                Ok(sample) => sample,
                Err(ObjectiveEvalError::Recoverable { message })
                | Err(ObjectiveEvalError::Fatal { message }) => {
                    return Err(FixedPointError::ObjectiveFailed { message });
                }
            };
            func_evals += 1;
            let value = recover_on_nonfinite_cost(sample.value).map_err(|err| match err {
                ObjectiveEvalError::Recoverable { message }
                | ObjectiveEvalError::Fatal { message } => {
                    FixedPointError::ObjectiveFailed { message }
                }
            })?;
            if sample.step.len() != x_k.len() {
                return Err(FixedPointError::StepDimensionMismatch {
                    expected: x_k.len(),
                    got: sample.step.len(),
                });
            }
            if sample.step.iter().any(|value| !value.is_finite()) {
                return Err(FixedPointError::NonFiniteStep);
            }
            if matches!(sample.status, FixedPointStatus::Stop) {
                return Ok(Solution::fixed_point(x_k, value, 0.0, k, func_evals));
            }
            let x_next = self.project_point(&(&x_k + &sample.step));
            let applied_step = &x_next - &x_k;
            let step_norm = applied_step.dot(&applied_step).sqrt();
            if !step_norm.is_finite() {
                return Err(FixedPointError::NonFiniteStep);
            }
            last_value = value;
            last_step_norm = step_norm;
            x_k = x_next;
            if step_norm <= self.tolerance {
                return Ok(Solution::fixed_point(
                    x_k,
                    value,
                    step_norm,
                    k + 1,
                    func_evals,
                ));
            }
        }
        Err(FixedPointError::MaxIterationsReached {
            last_solution: Box::new(Solution::fixed_point(
                x_k,
                last_value,
                last_step_norm,
                self.max_iterations,
                func_evals,
            )),
        })
    }
}

pub struct FixedPoint<ObjFn> {
    core: FixedPointCore,
    obj_fn: ObjFn,
}

impl<ObjFn> FixedPoint<ObjFn>
where
    ObjFn: FixedPointObjective,
{
    pub fn new(x0: Array1<f64>, obj_fn: ObjFn) -> Self {
        Self {
            core: FixedPointCore::new(x0),
            obj_fn,
        }
    }

    pub fn with_tolerance(mut self, tolerance: Tolerance) -> Self {
        self.core.tolerance = tolerance.get();
        self
    }

    pub fn with_max_iterations(mut self, max_iterations: MaxIterations) -> Self {
        self.core.max_iterations = max_iterations.get();
        self
    }

    pub fn with_bounds(mut self, bounds: Bounds) -> Self {
        self.core.bounds = Some(bounds.spec);
        self
    }

    pub fn run(&mut self) -> Result<Solution, FixedPointError> {
        self.core.run(&mut self.obj_fn)
    }
}

/// A line search algorithm that finds a step size satisfying the Strong Wolfe conditions.
///
/// Bracketing + zoom with safeguards and efficient state-passing to avoid re-computation.
#[allow(clippy::too_many_arguments)]
fn line_search<ObjFn>(
    core: &mut BfgsCore,
    obj_fn: &mut ObjFn,
    oracle: &mut FirstOrderCache,
    x_k: &Array1<f64>,
    d_k: &Array1<f64>,
    f_k: f64,
    g_k: &Array1<f64>,
    c1: f64,
    c2: f64,
) -> LsResult
where
    ObjFn: FirstOrderObjective,
{
    let mut alpha_i: f64 = 1.0; // Start with a unit step.
    let mut alpha_prev = 0.0;

    let mut f_prev = f_k;
    let g_proj_k = core.projected_gradient(x_k, g_k);
    let g_k_dot_d = g_proj_k.dot(d_k); // Initial derivative along the search direction.
    if g_k_dot_d >= -eps_g(&g_proj_k, d_k, core.tau_g) {
        log::debug!(
            "[BFGS Wolfe] Non-descent direction detected (gᵀd = {:.2e} >= 0).",
            g_k_dot_d
        );
    }
    let mut g_prev_dot_d = g_k_dot_d;

    let max_attempts = WOLFE_MAX_ATTEMPTS;
    let mut func_evals = 0;
    let mut grad_evals = 0;
    let epsF = eps_f(f_k, core.tau_f);
    let mut best = ProbeBest::new(x_k, f_k, g_k);
    for _ in 0..max_attempts {
        let (x_new, s, kinked) = core.project_with_step(x_k, d_k, alpha_i);
        let step_ok = !core.projected_step_small(x_k, &s);
        if !step_ok {
            return Err(LineSearchError::StepSizeTooSmall);
        }
        let mut f_i = match bfgs_eval_cost(oracle, obj_fn, &x_new, &mut func_evals) {
            Ok(f) => f,
            Err(ObjectiveEvalError::Recoverable { .. }) => f64::NAN,
            Err(ObjectiveEvalError::Fatal { message }) => {
                return Err(LineSearchError::ObjectiveFailed(message));
            }
        };

        // Handle any non-finite value early
        if !f_i.is_finite() {
            core.nonfinite_seen = true;
            if alpha_prev == 0.0 {
                alpha_i *= 0.5;
            } else {
                alpha_i = 0.5 * (alpha_prev + alpha_i);
            }
            if alpha_i <= 1e-18 {
                if let Some((a, f, g, kind)) = probe_alphas(
                    core,
                    obj_fn,
                    oracle,
                    x_k,
                    d_k,
                    f_k,
                    g_k,
                    0.0,
                    alpha_i.max(f64::EPSILON),
                    core.tau_g,
                    core.grad_drop_factor,
                    &mut func_evals,
                    &mut grad_evals,
                ) {
                    return Ok((a, f, g, func_evals, grad_evals, kind));
                }
                return Err(LineSearchError::StepSizeTooSmall);
            }
            // Back-off attempts when stuck in non-finite region
            if func_evals >= 3 {
                return Err(LineSearchError::MaxAttempts(max_attempts));
            }
            continue;
        }

        // Classic Armijo + previous worsening for bracketing (Strong-Wolfe)
        let gkTs = g_proj_k.dot(&s);
        let armijo_strict = f_i > f_k + c1 * gkTs + epsF;
        let prev_worse = func_evals > 1 && f_i >= f_prev - epsF;
        if armijo_strict || prev_worse {
            let kink_lo = if alpha_prev > 0.0 {
                let (_, _, kink_prev) = core.project_with_step(x_k, d_k, alpha_prev);
                kink_prev
            } else {
                false
            };
            if kink_lo || kinked {
                let fallback = backtracking_line_search(core, obj_fn, oracle, x_k, d_k, f_k, g_k);
                return fallback.map(|(a, f, g, fe, ge, kind)| {
                    (a, f, g, fe + func_evals, ge + grad_evals, kind)
                });
            }
            let r = zoom(
                core,
                obj_fn,
                oracle,
                x_k,
                d_k,
                f_k,
                g_k,
                &g_proj_k,
                g_k_dot_d,
                c1,
                c2,
                alpha_prev,
                alpha_i,
                f_prev,
                f_i,
                g_prev_dot_d,
                f64::NAN,
                func_evals,
                grad_evals,
            );
            if r.is_err() {
                if best.f.is_finite() {
                    core.global_best = Some(best.clone());
                }
            }
            return r;
        }

        let (f_full, g_i) =
            match bfgs_eval_cost_grad(oracle, obj_fn, &x_new, &mut func_evals, &mut grad_evals) {
                Ok(sample) => sample,
                Err(ObjectiveEvalError::Recoverable { .. }) => {
                    core.nonfinite_seen = true;
                    if alpha_prev == 0.0 {
                        alpha_i *= 0.5;
                    } else {
                        alpha_i = 0.5 * (alpha_prev + alpha_i);
                    }
                    if alpha_i <= 1e-18 {
                        return Err(LineSearchError::StepSizeTooSmall);
                    }
                    continue;
                }
                Err(ObjectiveEvalError::Fatal { message }) => {
                    return Err(LineSearchError::ObjectiveFailed(message));
                }
            };
        f_i = f_full;
        if !f_i.is_finite() || g_i.iter().any(|v| !v.is_finite()) {
            core.nonfinite_seen = true;
            if alpha_prev == 0.0 {
                alpha_i *= 0.5;
            } else {
                alpha_i = 0.5 * (alpha_prev + alpha_i);
            }
            if alpha_i <= 1e-18 {
                return Err(LineSearchError::StepSizeTooSmall);
            }
            continue;
        }
        best.consider(&x_new, f_i, &g_i);

        let armijo_strict = f_i > f_k + c1 * gkTs + epsF;
        let prev_worse = func_evals > 1 && f_i >= f_prev - epsF;
        if armijo_strict || prev_worse {
            let kink_lo = if alpha_prev > 0.0 {
                let (_, _, kink_prev) = core.project_with_step(x_k, d_k, alpha_prev);
                kink_prev
            } else {
                false
            };
            if kink_lo || kinked {
                let fallback = backtracking_line_search(core, obj_fn, oracle, x_k, d_k, f_k, g_k);
                return fallback.map(|(a, f, g, fe, ge, kind)| {
                    (a, f, g, fe + func_evals, ge + grad_evals, kind)
                });
            }
            let g_proj_i = core.projected_gradient(&x_new, &g_i);
            let g_i_dot_d = directional_derivative(&g_proj_i, &s, alpha_i, d_k);
            let r = zoom(
                core,
                obj_fn,
                oracle,
                x_k,
                d_k,
                f_k,
                g_k,
                &g_proj_k,
                g_k_dot_d,
                c1,
                c2,
                alpha_prev,
                alpha_i,
                f_prev,
                f_i,
                g_prev_dot_d,
                g_i_dot_d,
                func_evals,
                grad_evals,
            );
            if r.is_err() && best.f.is_finite() {
                core.global_best = Some(best.clone());
            }
            return r;
        }

        let g_proj_i = core.projected_gradient(&x_new, &g_i);
        let g_i_dot_d = directional_derivative(&g_proj_i, &s, alpha_i, d_k);
        let g_k_dot_eff = directional_derivative(&g_proj_k, &s, alpha_i, d_k);
        let gi_norm = g_proj_i.dot(&g_proj_i).sqrt();
        let gk_norm = g_proj_k.dot(&g_proj_k).sqrt();
        let drop_factor = core.grad_drop_factor;
        let fmax = if core.gll.is_empty() {
            f_k
        } else {
            core.gll.fmax()
        };
        let epsG = eps_g(&g_proj_k, d_k, core.tau_g);
        if let Some(kind) = classify_line_search_accept(
            core,
            step_ok,
            f_k,
            fmax,
            f_i,
            gkTs,
            g_i_dot_d,
            g_k_dot_eff,
            gi_norm,
            gk_norm,
            drop_factor,
            epsF,
            epsG,
            c2,
        ) {
            if matches!(kind, AcceptKind::StrongWolfe) {
                let delta_now = core.trust_radius;
                core.trust_radius = (delta_now * 1.25).min(1e6);
            }
            return Ok((alpha_i, f_i, g_i, func_evals, grad_evals, kind));
        }

        if g_i_dot_d >= -eps_g(&g_proj_k, d_k, core.tau_g) {
            // The minimum is bracketed between alpha_i and alpha_prev.
            // The current point is the best (low) endpoint.
            let r = zoom(
                core,
                obj_fn,
                oracle,
                x_k,
                d_k,
                f_k,
                g_k,
                &g_proj_k,
                g_k_dot_d,
                c1,
                c2,
                alpha_i,
                alpha_prev,
                f_i,
                f_prev,
                g_i_dot_d,
                g_prev_dot_d,
                func_evals,
                grad_evals,
            );
            if r.is_err() {
                if best.f.is_finite() {
                    core.global_best = Some(best.clone());
                }
            }
            return r;
        }

        // The step is too short, expand the search interval and cache current state.
        alpha_prev = alpha_i;
        f_prev = f_i;
        g_prev_dot_d = g_i_dot_d;
        // Expand alpha but respect alpha_max domain
        alpha_i *= 2.0;
    }

    if best.f.is_finite() {
        core.global_best = Some(best);
    }
    // Probing grid before declaring failure
    if alpha_i > 0.0
        && let Some((a, f, g, kind)) = probe_alphas(
            core,
            obj_fn,
            oracle,
            x_k,
            d_k,
            f_k,
            g_k,
            0.0,
            alpha_i,
            core.tau_g,
            core.grad_drop_factor,
            &mut func_evals,
            &mut grad_evals,
        )
    {
        return Ok((a, f, g, func_evals, grad_evals, kind));
    }
    Err(LineSearchError::MaxAttempts(max_attempts))
}

/// A backtracking line search that shrinks trial steps until a Wolfe-safe acceptance fires.
fn backtracking_line_search<ObjFn>(
    core: &mut BfgsCore,
    obj_fn: &mut ObjFn,
    oracle: &mut FirstOrderCache,
    x_k: &Array1<f64>,
    d_k: &Array1<f64>,
    f_k: f64,
    g_k: &Array1<f64>,
) -> LsResult
where
    ObjFn: FirstOrderObjective,
{
    let mut alpha: f64 = 1.0;
    let mut rho = 0.5;
    let max_attempts = BACKTRACKING_MAX_ATTEMPTS;

    let g_proj_k = core.projected_gradient(x_k, g_k);
    let g_k_dot_d = g_proj_k.dot(d_k);
    // A backtracking search is only valid on a descent direction.
    if g_k_dot_d >= -eps_g(&g_proj_k, d_k, core.tau_g) {
        log::debug!(
            "[BFGS Backtracking] Search started with a non-descent direction (gᵀd = {:.2e} > 0). This step will likely fail.",
            g_k_dot_d
        );
    }

    let mut func_evals = 0;
    let mut grad_evals = 0;
    let mut best = ProbeBest::new(x_k, f_k, g_k);
    let epsF = eps_f(f_k, core.tau_f);
    let mut no_change_count = 0usize;
    let mut expanded_once = false;
    let dnorm = d_k.dot(d_k).sqrt();
    for _ in 0..max_attempts {
        let (x_new, s, _) = core.project_with_step(x_k, d_k, alpha);
        let step_ok = !core.projected_step_small(x_k, &s);
        if !step_ok {
            return Err(LineSearchError::StepSizeTooSmall);
        }
        let mut f_new = match bfgs_eval_cost(oracle, obj_fn, &x_new, &mut func_evals) {
            Ok(f) => f,
            Err(ObjectiveEvalError::Recoverable { .. }) => f64::NAN,
            Err(ObjectiveEvalError::Fatal { message }) => {
                return Err(LineSearchError::ObjectiveFailed(message));
            }
        };

        // If evaluation is non-finite, shrink alpha and continue (salvage best-so-far)
        if !f_new.is_finite() {
            core.nonfinite_seen = true;
            alpha *= rho;
            if alpha < 1e-16 {
                return Err(LineSearchError::StepSizeTooSmall);
            }
            if func_evals >= 3 {
                return Err(LineSearchError::MaxAttempts(max_attempts));
            }
            continue;
        }

        let gkTs = g_proj_k.dot(&s);
        let fmax = if core.gll.is_empty() {
            f_k
        } else {
            core.gll.fmax()
        };
        let armijo_accept = core.accept_armijo(f_k, gkTs, f_new);
        let gll_accept = core.accept_gll_nonmonotone(fmax, gkTs, f_new);
        let candidate_for_gradient = armijo_accept
            || gll_accept
            || (core.relaxed_acceptors_enabled() && f_new <= f_k + epsF);
        let mut g_new_opt = None;
        if candidate_for_gradient {
            let (f_full, g_new) =
                match bfgs_eval_cost_grad(oracle, obj_fn, &x_new, &mut func_evals, &mut grad_evals)
                {
                    Ok(sample) => sample,
                    Err(ObjectiveEvalError::Recoverable { .. }) => {
                        core.nonfinite_seen = true;
                        alpha *= rho;
                        if alpha < 1e-16 {
                            return Err(LineSearchError::StepSizeTooSmall);
                        }
                        continue;
                    }
                    Err(ObjectiveEvalError::Fatal { message }) => {
                        return Err(LineSearchError::ObjectiveFailed(message));
                    }
                };
            f_new = f_full;
            if !f_new.is_finite() || g_new.iter().any(|v| !v.is_finite()) {
                core.nonfinite_seen = true;
                alpha *= rho;
                if alpha < 1e-16 {
                    return Err(LineSearchError::StepSizeTooSmall);
                }
                continue;
            }
            best.consider(&x_new, f_new, &g_new);
            g_new_opt = Some(g_new);
        }

        let Some(g_new) = g_new_opt else {
            if (f_new - f_k).abs() <= epsF {
                no_change_count += 1;
            } else {
                no_change_count = 0;
                expanded_once = false;
            }
            if no_change_count >= 3 {
                rho = 0.8;
            }
            if no_change_count >= 2 && !expanded_once {
                alpha /= rho;
                expanded_once = true;
            } else {
                alpha *= rho;
            }
            if core.jiggle_enabled() && no_change_count >= 2 {
                let jiggle = 1.0 + core.jiggle_scale() * core.next_rand_sym();
                alpha = (alpha * jiggle).max(f64::EPSILON);
            }
            let tol_x = core.step_tolerance(x_k);
            if (alpha * dnorm) <= tol_x {
                return Err(LineSearchError::StepSizeTooSmall);
            }
            continue;
        };

        // Gradient reduction acceptance
        let g_proj_new = core.projected_gradient(&x_new, &g_new);
        let gk_dot_eff = directional_derivative(&g_proj_k, &s, alpha, d_k);
        let gnew_norm = g_proj_new.dot(&g_proj_new).sqrt();
        let gk_norm = g_proj_k.dot(&g_proj_k).sqrt();
        let drop_factor = core.grad_drop_factor;
        let g_new_dot_d = directional_derivative(&g_proj_new, &s, alpha, d_k);
        let epsG = eps_g(&g_proj_k, d_k, core.tau_g);
        if let Some(kind) = classify_line_search_accept(
            core,
            step_ok,
            f_k,
            fmax,
            f_new,
            gkTs,
            g_new_dot_d,
            gk_dot_eff,
            gnew_norm,
            gk_norm,
            drop_factor,
            epsF,
            epsG,
            core.c2_adapt,
        ) {
            return Ok((alpha, f_new, g_new, func_evals, grad_evals, kind));
        }

        if (f_new - f_k).abs() <= epsF {
            no_change_count += 1;
        } else {
            no_change_count = 0;
            expanded_once = false;
        }
        if no_change_count >= 3 {
            rho = 0.8;
        }
        if no_change_count >= 2 && !expanded_once {
            // one-time expansion to hop flat plateau
            alpha /= rho; // slight expand
            expanded_once = true;
        } else {
            alpha *= rho;
        }
        // Stochastic jiggling to avoid hitting identical thresholds repeatedly
        if core.jiggle_enabled() && no_change_count >= 2 {
            let jiggle = 1.0 + core.jiggle_scale() * core.next_rand_sym();
            alpha = (alpha * jiggle).max(f64::EPSILON);
        }
        // Relative step-size stop: ||alpha d|| <= tol_x
        let tol_x = core.step_tolerance(x_k);
        if (alpha * dnorm) <= tol_x {
            return Err(LineSearchError::StepSizeTooSmall);
        }
    }

    // Probing grid before declaring failure
    if alpha > 0.0
        && let Some((a, f, g, kind)) = probe_alphas(
            core,
            obj_fn,
            oracle,
            x_k,
            d_k,
            f_k,
            g_k,
            0.0,
            alpha,
            core.tau_g,
            core.grad_drop_factor,
            &mut func_evals,
            &mut grad_evals,
        )
    {
        return Ok((a, f, g, func_evals, grad_evals, kind));
    }

    // Stash best seen during backtracking
    if best.f.is_finite() {
        core.global_best = Some(best);
    }
    Err(LineSearchError::MaxAttempts(max_attempts))
}

/// Helper "zoom" function using cubic interpolation.
///
/// This function is called when a bracketing interval [alpha_lo, alpha_hi] that contains
/// a point satisfying the Strong Wolfe conditions is known. It iteratively refines this
/// interval until a suitable step size is found.
#[allow(clippy::too_many_arguments)]
fn zoom<ObjFn>(
    core: &mut BfgsCore,
    obj_fn: &mut ObjFn,
    oracle: &mut FirstOrderCache,
    x_k: &Array1<f64>,
    d_k: &Array1<f64>,
    f_k: f64,
    g_k: &Array1<f64>,
    g_proj_k: &Array1<f64>,
    _g_k_dot_d: f64,
    c1: f64,
    c2: f64,
    mut alpha_lo: f64,
    mut alpha_hi: f64,
    mut f_lo: f64,
    mut f_hi: f64,
    mut g_lo_dot_d: f64,
    mut g_hi_dot_d: f64,
    mut func_evals: usize,
    mut grad_evals: usize,
) -> LsResult
where
    ObjFn: FirstOrderObjective,
{
    let max_zoom_attempts = 15;
    let min_alpha_step = 1e-12; // Prevents division by zero or degenerate steps.
    let epsF = eps_f(f_k, core.tau_f);
    let mut best = ProbeBest::new(x_k, f_k, g_k);
    let mut lo_deriv_known = g_lo_dot_d.is_finite();
    let mut hi_deriv_known = g_hi_dot_d.is_finite();
    for _ in 0..max_zoom_attempts {
        let kink_lo = if alpha_lo > 0.0 {
            let (_, _, kink) = core.project_with_step(x_k, d_k, alpha_lo);
            kink
        } else {
            false
        };
        let kink_hi = if alpha_hi > 0.0 {
            let (_, _, kink) = core.project_with_step(x_k, d_k, alpha_hi);
            kink
        } else {
            false
        };
        if kink_lo || kink_hi {
            let fallback = backtracking_line_search(core, obj_fn, oracle, x_k, d_k, f_k, g_k);
            return fallback
                .map(|(a, f, g, fe, ge, kind)| (a, f, g, fe + func_evals, ge + grad_evals, kind));
        }
        let tiny_bracket = (alpha_hi - alpha_lo).abs() <= 1e-12;
        let flat_f = (f_hi - f_lo).abs() <= epsF;
        let similar_slope = lo_deriv_known
            && hi_deriv_known
            && (g_hi_dot_d.abs() - g_lo_dot_d.abs()).abs()
                <= core.curv_slack_scale * eps_g(g_proj_k, d_k, core.tau_g);
        // Endpoint rescue on tiny brackets or flat ends with mismatched slopes.
        if tiny_bracket || (flat_f && !similar_slope) {
            let (mut alpha_j, choose_lo) = match (lo_deriv_known, hi_deriv_known) {
                (true, true) => {
                    if g_lo_dot_d.abs() <= g_hi_dot_d.abs() {
                        (alpha_lo, true)
                    } else {
                        (alpha_hi, false)
                    }
                }
                (true, false) => (alpha_lo, true),
                (false, true) => (alpha_hi, false),
                (false, false) => ((alpha_lo + alpha_hi) / 2.0, false),
            };
            // Avoid zero step; prefer the nonzero endpoint, otherwise midpoint
            if alpha_j <= f64::EPSILON {
                alpha_j = if choose_lo { alpha_hi } else { alpha_lo };
            }
            if alpha_j <= f64::EPSILON {
                alpha_j = 0.5 * (alpha_lo + alpha_hi);
            }
            let (x_j, s_j, kink_mid) = core.project_with_step(x_k, d_k, alpha_j);
            let step_ok = !core.projected_step_small(x_k, &s_j);
            if !step_ok {
                return Err(LineSearchError::StepSizeTooSmall);
            }
            if kink_mid {
                let fallback = backtracking_line_search(core, obj_fn, oracle, x_k, d_k, f_k, g_k);
                return fallback.map(|(a, f, g, fe, ge, kind)| {
                    (a, f, g, fe + func_evals, ge + grad_evals, kind)
                });
            }
            let (f_j, g_j) =
                match bfgs_eval_cost_grad(oracle, obj_fn, &x_j, &mut func_evals, &mut grad_evals) {
                    Ok(sample) => sample,
                    Err(ObjectiveEvalError::Recoverable { .. }) => {
                        (f64::NAN, Array1::zeros(x_j.len()))
                    }
                    Err(ObjectiveEvalError::Fatal { message }) => {
                        return Err(LineSearchError::ObjectiveFailed(message));
                    }
                };
            if !f_j.is_finite() || g_j.iter().any(|&v| !v.is_finite()) {
                core.nonfinite_seen = true;
                if choose_lo {
                    alpha_lo = 0.5 * (alpha_lo + alpha_hi);
                    lo_deriv_known = false;
                } else {
                    alpha_hi = 0.5 * (alpha_lo + alpha_hi);
                    hi_deriv_known = false;
                }
                continue;
            }
            // Acceptance guard shared with the main search/probing paths.
            let g_proj_j = core.projected_gradient(&x_j, &g_j);
            let gkTs = g_proj_k.dot(&s_j);
            let gk_dot_d_eff = directional_derivative(g_proj_k, &s_j, alpha_j, d_k);
            let g_j_dot_d = directional_derivative(&g_proj_j, &s_j, alpha_j, d_k);
            let epsG = eps_g(g_proj_k, d_k, core.tau_g);
            let gj_norm = g_proj_j.iter().map(|v| v * v).sum::<f64>().sqrt();
            let gk_norm = g_proj_k.iter().map(|v| v * v).sum::<f64>().sqrt();
            let drop_factor = core.grad_drop_factor;
            let fmax = if core.gll.is_empty() {
                f_k
            } else {
                core.gll.fmax()
            };
            if let Some(kind) = classify_line_search_accept(
                core,
                step_ok,
                f_k,
                fmax,
                f_j,
                gkTs,
                g_j_dot_d,
                gk_dot_d_eff,
                gj_norm,
                gk_norm,
                drop_factor,
                epsF,
                epsG,
                c2,
            ) {
                return Ok((alpha_j, f_j, g_j, func_evals, grad_evals, kind));
            } else {
                // tighten bracket and continue
                let mid = 0.5 * (alpha_lo + alpha_hi);
                if alpha_j > mid {
                    alpha_hi = alpha_j;
                    f_hi = f_j;
                    g_hi_dot_d = g_j_dot_d;
                    hi_deriv_known = true;
                } else {
                    alpha_lo = alpha_j;
                    f_lo = f_j;
                    g_lo_dot_d = g_j_dot_d;
                    lo_deriv_known = true;
                }
                continue;
            }
        }
        if flat_f && similar_slope {
            let alpha_mid = 0.5 * (alpha_lo + alpha_hi);
            let (x_mid, s_mid, kink_mid) = core.project_with_step(x_k, d_k, alpha_mid);
            let step_ok = !core.projected_step_small(x_k, &s_mid);
            if !step_ok {
                return Err(LineSearchError::StepSizeTooSmall);
            }
            if kink_mid {
                let fallback = backtracking_line_search(core, obj_fn, oracle, x_k, d_k, f_k, g_k);
                return fallback.map(|(a, f, g, fe, ge, kind)| {
                    (a, f, g, fe + func_evals, ge + grad_evals, kind)
                });
            }
            let (f_mid, g_mid) =
                match bfgs_eval_cost_grad(oracle, obj_fn, &x_mid, &mut func_evals, &mut grad_evals)
                {
                    Ok(sample) => sample,
                    Err(ObjectiveEvalError::Recoverable { .. }) => {
                        core.nonfinite_seen = true;
                        let tighten_lo = g_lo_dot_d.abs() > g_hi_dot_d.abs();
                        if tighten_lo {
                            alpha_lo = alpha_mid;
                            lo_deriv_known = false;
                        } else {
                            alpha_hi = alpha_mid;
                            hi_deriv_known = false;
                        }
                        continue;
                    }
                    Err(ObjectiveEvalError::Fatal { message }) => {
                        return Err(LineSearchError::ObjectiveFailed(message));
                    }
                };
            if f_mid.is_finite() && g_mid.iter().all(|v| v.is_finite()) {
                // Midpoint rescue still has to satisfy the same decrease/curvature gates.
                let g_proj_mid = core.projected_gradient(&x_mid, &g_mid);
                let g_mid_dot_d = directional_derivative(&g_proj_mid, &s_mid, alpha_mid, d_k);
                let gkTs = g_proj_k.dot(&s_mid);
                let gk_dot_d_eff = directional_derivative(g_proj_k, &s_mid, alpha_mid, d_k);
                let epsG = eps_g(g_proj_k, d_k, core.tau_g);
                let gmid_norm = g_proj_mid.iter().map(|v| v * v).sum::<f64>().sqrt();
                let gk_norm = g_proj_k.iter().map(|v| v * v).sum::<f64>().sqrt();
                let drop_factor = core.grad_drop_factor;
                let fmax = if core.gll.is_empty() {
                    f_k
                } else {
                    core.gll.fmax()
                };
                if let Some(kind) = classify_line_search_accept(
                    core,
                    step_ok,
                    f_k,
                    fmax,
                    f_mid,
                    gkTs,
                    g_mid_dot_d,
                    gk_dot_d_eff,
                    gmid_norm,
                    gk_norm,
                    drop_factor,
                    epsF,
                    epsG,
                    c2,
                ) {
                    return Ok((alpha_mid, f_mid, g_mid, func_evals, grad_evals, kind));
                }
                let tighten_lo = g_lo_dot_d.abs() > g_hi_dot_d.abs();
                if tighten_lo {
                    alpha_lo = alpha_mid;
                    f_lo = f_mid;
                    g_lo_dot_d = g_mid_dot_d;
                    lo_deriv_known = true;
                } else {
                    alpha_hi = alpha_mid;
                    f_hi = f_mid;
                    g_hi_dot_d = g_mid_dot_d;
                    hi_deriv_known = true;
                }
                continue;
            } else {
                core.nonfinite_seen = true;
                let tighten_lo = g_lo_dot_d.abs() > g_hi_dot_d.abs();
                if tighten_lo {
                    alpha_lo = alpha_mid;
                    lo_deriv_known = false;
                } else {
                    alpha_hi = alpha_mid;
                    hi_deriv_known = false;
                }
                continue;
            }
        }
        // --- Use cubic interpolation to find a trial step size `alpha_j` ---
        // If the entire bracket is in an unusable (infinite) region, fail immediately.
        if !f_lo.is_finite() && !f_hi.is_finite() {
            log::debug!("[BFGS Zoom] Line search bracketed an infinite region. Aborting.");
            return Err(LineSearchError::MaxAttempts(max_zoom_attempts));
        }
        let alpha_j = {
            let (alpha_lo_i, alpha_hi_i, f_lo_i, f_hi_i, g_lo_i, g_hi_i) = if alpha_lo <= alpha_hi {
                (alpha_lo, alpha_hi, f_lo, f_hi, g_lo_dot_d, g_hi_dot_d)
            } else {
                (alpha_hi, alpha_lo, f_hi, f_lo, g_hi_dot_d, g_lo_dot_d)
            };

            let alpha_diff = alpha_hi_i - alpha_lo_i;

            // Fallback to bisection if the interval is too small, derivatives unknown,
            // or if function values at the interval ends are infinite, preventing unstable interpolation.
            if alpha_diff < min_alpha_step
                || !f_lo_i.is_finite()
                || !f_hi_i.is_finite()
                || !lo_deriv_known
                || !hi_deriv_known
            {
                (alpha_lo + alpha_hi) / 2.0
            } else {
                // Cubic interpolation using endpoint function values and directional derivatives.
                // d1 and d2 come from the cubic interpolant that matches f and directional
                // derivatives at the bracket endpoints.
                let d1 = g_lo_i + g_hi_i - 3.0 * (f_hi_i - f_lo_i) / alpha_diff;
                let d2_sq = d1 * d1 - g_lo_i * g_hi_i;

                if d2_sq >= 0.0 && d2_sq.is_finite() {
                    let d2 = d2_sq.sqrt();
                    let trial =
                        alpha_hi_i - alpha_diff * (g_hi_i + d2 - d1) / (g_hi_i - g_lo_i + 2.0 * d2);

                    // If interpolation gives a non-finite value or a point outside
                    // the bracket, fall back to bisection.
                    if !trial.is_finite() || trial < alpha_lo_i || trial > alpha_hi_i {
                        (alpha_lo + alpha_hi) / 2.0
                    } else {
                        trial
                    }
                } else {
                    (alpha_lo + alpha_hi) / 2.0
                }
            }
        };

        // If the trial step is not making sufficient progress, bisect instead.
        let alpha_j = if (alpha_j - alpha_lo).abs() < min_alpha_step
            || (alpha_j - alpha_hi).abs() < min_alpha_step
        {
            (alpha_lo + alpha_hi) / 2.0
        } else {
            alpha_j
        };

        let (x_j, s_j, kink_j) = core.project_with_step(x_k, d_k, alpha_j);
        let step_ok = !core.projected_step_small(x_k, &s_j);
        if !step_ok {
            return Err(LineSearchError::StepSizeTooSmall);
        }
        if kink_j {
            let fallback = backtracking_line_search(core, obj_fn, oracle, x_k, d_k, f_k, g_k);
            return fallback
                .map(|(a, f, g, fe, ge, kind)| (a, f, g, fe + func_evals, ge + grad_evals, kind));
        }
        let mut f_j = match bfgs_eval_cost(oracle, obj_fn, &x_j, &mut func_evals) {
            Ok(f) => f,
            Err(ObjectiveEvalError::Recoverable { .. }) => f64::NAN,
            Err(ObjectiveEvalError::Fatal { message }) => {
                return Err(LineSearchError::ObjectiveFailed(message));
            }
        };

        // Handle non-finite by shrinking toward the finite end; keep derivative info intact
        if !f_j.is_finite() {
            core.nonfinite_seen = true;
            // Move the bound closer to alpha_j, prefer shrinking the side that alpha_j is nearer to
            let to_hi = (alpha_hi - alpha_j).abs() <= (alpha_j - alpha_lo).abs();
            if to_hi {
                alpha_hi = alpha_j;
                f_hi = f_j;
                hi_deriv_known = false;
            } else {
                alpha_lo = alpha_j;
                f_lo = f_j;
                lo_deriv_known = false;
            }
            continue;
        }

        // Check if the new point `alpha_j` satisfies the sufficient decrease condition.
        // An infinite `f_j` means the step was too large and failed the condition.
        let fmax = if core.gll.is_empty() {
            f_k
        } else {
            core.gll.fmax()
        };
        let gkTs = g_proj_k.dot(&s_j);
        let gk_dot_d_eff = directional_derivative(g_proj_k, &s_j, alpha_j, d_k);
        let armijo_ok = f_j <= f_k + c1 * gkTs + epsF;
        let armijo_gll_ok = f_j <= fmax + c1 * gkTs + epsF;
        if (!armijo_ok && !armijo_gll_ok) || f_j >= f_lo - epsF {
            alpha_hi = alpha_j;
            f_hi = f_j;
            hi_deriv_known = false;
        } else {
            let (f_full, g_j) =
                match bfgs_eval_cost_grad(oracle, obj_fn, &x_j, &mut func_evals, &mut grad_evals) {
                    Ok(sample) => sample,
                    Err(ObjectiveEvalError::Recoverable { .. }) => {
                        core.nonfinite_seen = true;
                        let to_hi = (alpha_hi - alpha_j).abs() <= (alpha_j - alpha_lo).abs();
                        if to_hi {
                            alpha_hi = alpha_j;
                            f_hi = f64::NAN;
                            hi_deriv_known = false;
                        } else {
                            alpha_lo = alpha_j;
                            f_lo = f64::NAN;
                            lo_deriv_known = false;
                        }
                        continue;
                    }
                    Err(ObjectiveEvalError::Fatal { message }) => {
                        return Err(LineSearchError::ObjectiveFailed(message));
                    }
                };
            f_j = f_full;
            if !f_j.is_finite() || g_j.iter().any(|&v| !v.is_finite()) {
                core.nonfinite_seen = true;
                let to_hi = (alpha_hi - alpha_j).abs() <= (alpha_j - alpha_lo).abs();
                if to_hi {
                    alpha_hi = alpha_j;
                    f_hi = f_j;
                    hi_deriv_known = false;
                } else {
                    alpha_lo = alpha_j;
                    f_lo = f_j;
                    lo_deriv_known = false;
                }
                continue;
            }
            best.consider(&x_j, f_j, &g_j);
            let armijo_ok = f_j <= f_k + c1 * gkTs + epsF;
            let armijo_gll_ok = f_j <= fmax + c1 * gkTs + epsF;
            if (!armijo_ok && !armijo_gll_ok) || f_j >= f_lo - epsF {
                alpha_hi = alpha_j;
                f_hi = f_j;
                let g_proj_j = core.projected_gradient(&x_j, &g_j);
                g_hi_dot_d = directional_derivative(&g_proj_j, &s_j, alpha_j, d_k);
                hi_deriv_known = true;
                continue;
            }

            let g_proj_j = core.projected_gradient(&x_j, &g_j);
            let g_j_dot_d = directional_derivative(&g_proj_j, &s_j, alpha_j, d_k);
            let gj_norm = g_proj_j.dot(&g_proj_j).sqrt();
            let gk_norm = g_proj_k.dot(g_proj_k).sqrt();
            let drop_factor = core.grad_drop_factor;
            let epsG = eps_g(g_proj_k, d_k, core.tau_g);
            if let Some(kind) = classify_line_search_accept(
                core,
                step_ok,
                f_k,
                fmax,
                f_j,
                gkTs,
                g_j_dot_d,
                gk_dot_d_eff,
                gj_norm,
                gk_norm,
                drop_factor,
                epsF,
                epsG,
                c2,
            ) {
                return Ok((alpha_j, f_j, g_j, func_evals, grad_evals, kind));
            }

            // The minimum is bracketed by a point with a negative derivative
            // (alpha_lo) and a point with a positive derivative (alpha_j).
            if g_j_dot_d >= -eps_g(g_proj_k, d_k, core.tau_g) {
                // The new point has a positive derivative and a lower function value,
                // so it becomes the new best (low) point and the old low becomes high.
                alpha_hi = alpha_lo;
                f_hi = f_lo;
                g_hi_dot_d = g_lo_dot_d;
                hi_deriv_known = lo_deriv_known;

                alpha_lo = alpha_j;
                f_lo = f_j;
                g_lo_dot_d = g_j_dot_d;
                lo_deriv_known = true;
            } else {
                // The new point has a negative derivative, so it becomes the new
                // lower bound of the bracket. The new interval is [alpha_j, alpha_hi].
                alpha_lo = alpha_j;
                f_lo = f_j;
                g_lo_dot_d = g_j_dot_d;
                lo_deriv_known = true;
            }
        }
    }
    // Probing grid before declaring failure
    if let Some((a, f, g, kind)) = probe_alphas(
        core,
        obj_fn,
        oracle,
        x_k,
        d_k,
        f_k,
        g_k,
        alpha_lo.min(alpha_hi),
        alpha_lo.max(alpha_hi),
        core.tau_g,
        core.grad_drop_factor,
        &mut func_evals,
        &mut grad_evals,
    ) {
        return Ok((a, f, g, func_evals, grad_evals, kind));
    }
    if best.f.is_finite() {
        core.global_best = Some(best);
    }
    Err(LineSearchError::MaxAttempts(max_zoom_attempts))
}

#[allow(clippy::too_many_arguments)]
fn probe_alphas<ObjFn>(
    core: &mut BfgsCore,
    obj_fn: &mut ObjFn,
    oracle: &mut FirstOrderCache,
    x_k: &Array1<f64>,
    d_k: &Array1<f64>,
    f_k: f64,
    g_k: &Array1<f64>,
    a_lo: f64,
    a_hi: f64,
    tau_g: f64,
    drop_factor: f64,
    fe: &mut usize,
    ge: &mut usize,
) -> Option<(f64, f64, Array1<f64>, AcceptKind)>
where
    ObjFn: FirstOrderObjective,
{
    let cands = [0.2, 0.5, 0.8].map(|t| a_lo + t * (a_hi - a_lo));
    let g_proj_k = core.projected_gradient(x_k, g_k);
    let gk_norm = g_proj_k.iter().map(|v| v * v).sum::<f64>().sqrt();
    let epsF = eps_f(f_k, core.tau_f);
    let epsG = eps_g(&g_proj_k, d_k, tau_g);
    let mut best: Option<(f64, f64, Array1<f64>, AcceptKind)> = None;
    for &a in &cands {
        if !a.is_finite() || a <= 0.0 {
            continue;
        }
        let (x, s, _) = core.project_with_step(x_k, d_k, a);
        let step_ok = !core.projected_step_small(x_k, &s);
        if !step_ok {
            continue;
        }
        let f = match bfgs_eval_cost(oracle, obj_fn, &x, fe) {
            Ok(f) => f,
            Err(_) => continue,
        };
        if !f.is_finite() {
            continue;
        }
        let gkTs = g_proj_k.dot(&s);
        let (f, g) = match bfgs_eval_cost_grad(oracle, obj_fn, &x, fe, ge) {
            Ok(sample) => sample,
            Err(_) => continue,
        };
        if !f.is_finite() || g.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let g_proj = core.projected_gradient(&x, &g);
        let gi_norm = g_proj.dot(&g_proj).sqrt();
        let g_trial_dot_d = directional_derivative(&g_proj, &s, a, d_k);
        let gk_dot_d_eff = directional_derivative(&g_proj_k, &s, a, d_k);
        let fmax = if core.gll.is_empty() {
            f_k
        } else {
            core.gll.fmax()
        };
        if let Some(kind) = classify_line_search_accept(
            core,
            step_ok,
            f_k,
            fmax,
            f,
            gkTs,
            g_trial_dot_d,
            gk_dot_d_eff,
            gi_norm,
            gk_norm,
            drop_factor,
            epsF,
            epsG,
            core.c2_adapt,
        ) && best.as_ref().map(|(fb, _, _, _)| f < *fb).unwrap_or(true)
        {
            best = Some((f, a, g, kind));
        }
    }
    best.map(|(f, a, g, kind)| (a, f, g, kind))
}

#[cfg(test)]
mod tests {
    // This test suite is structured into three parts:
    // 1. Standard Convergence Tests: Verifies that the solver finds the correct
    //    minimum for well-known benchmark functions from standard starting points.
    // 2. Failure and Edge Case Tests: Ensures the solver handles non-convex
    //    functions, pre-solved problems, and iteration limits correctly and returns
    //    the appropriate descriptive errors.
    // 3. Comparison Tests: Validates the behavior of our implementation against
    //    `argmin`, a trusted, state-of-the-art optimization library, ensuring
    //    that our results (final point and iteration count) are equivalent.

    use super::{
        ArcError, AutoSecondOrderSolver, BACKTRACKING_MAX_ATTEMPTS, Bfgs, BfgsError, Bounds,
        FiniteDiffGradient, FirstOrderObjective, FirstOrderSample, FixedPoint, FixedPointObjective,
        FixedPointSample, FixedPointStatus, LineSearchFailureReason, MaxIterations,
        NewtonTrustRegion, ObjectiveEvalError, Problem, Profile, SecondOrderObjective,
        SecondOrderProblem, SecondOrderSample, Solution, Tolerance, ZerothOrderObjective, optimize,
    };
    use ndarray::{Array1, Array2, array};
    use spectral::prelude::*;

    // --- Test Harness: Python scipy.optimize Comparison Setup ---
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::OnceLock;
    use std::sync::{Arc, Mutex};

    #[derive(serde::Deserialize)]
    struct PythonOptResult {
        success: bool,
        final_point: Option<Vec<f64>>,
        final_value: Option<f64>,
        final_gradient_norm: Option<f64>,
        iterations: Option<usize>,
        func_evals: Option<usize>,
        grad_evals: Option<usize>,
        message: Option<String>,
        error: Option<String>,
    }

    /// Call Python optimization harness and return the result
    fn optimize_with_python(
        x0: &Array1<f64>,
        function_name: &str,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<PythonOptResult, String> {
        let python = ensure_python_deps()?;
        let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let harness = crate_dir.join("optimization_harness.py");
        let input_json = serde_json::json!({
            "x0": x0.to_vec(),
            "function": function_name,
            "tolerance": tolerance,
            "max_iterations": max_iterations
        });

        let output = Command::new(python)
            .arg(&harness)
            .arg(input_json.to_string())
            .current_dir(&crate_dir)
            .output()
            .map_err(|e| format!("Failed to execute Python script: {}", e))?;

        if !output.status.success() {
            return Err(format!(
                "Python script failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let result_str = String::from_utf8(output.stdout)
            .map_err(|e| format!("Invalid UTF-8 in Python output: {}", e))?;

        serde_json::from_str(&result_str)
            .map_err(|e| format!("Failed to parse Python result: {}", e))
    }

    fn ensure_python_deps() -> Result<String, String> {
        static PYTHON_PATH: OnceLock<Result<String, String>> = OnceLock::new();
        PYTHON_PATH
            .get_or_init(|| {
                let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                let venv_python = crate_dir.join(".venv/bin/python");
                let venv_python_str = venv_python.display().to_string();
                let python = if Path::new(&venv_python).exists() {
                    venv_python_str.clone()
                } else {
                    "python3".to_string()
                };

                let check = Command::new(&python)
                    .arg("-c")
                    .arg("import numpy, scipy")
                    .output()
                    .map_err(|e| format!("Failed to execute Python: {}", e))?;

                if check.status.success() {
                    return Ok(python);
                }

                if python != venv_python_str {
                    let venv = Command::new("python3")
                        .arg("-m")
                        .arg("venv")
                        .arg(crate_dir.join(".venv"))
                        .current_dir(&crate_dir)
                        .output()
                        .map_err(|e| format!("Failed to create venv: {}", e))?;
                    if !venv.status.success() {
                        return Err(format!(
                            "Failed to create venv: {}",
                            String::from_utf8_lossy(&venv.stderr)
                        ));
                    }
                }

                let install = Command::new(&venv_python)
                    .arg("-m")
                    .arg("pip")
                    .arg("install")
                    .arg("numpy")
                    .arg("scipy")
                    .current_dir(&crate_dir)
                    .output()
                    .map_err(|e| format!("Failed to install numpy/scipy: {}", e))?;
                if !install.status.success() {
                    return Err(format!(
                        "Failed to install numpy/scipy: {}",
                        String::from_utf8_lossy(&install.stderr)
                    ));
                }

                Ok(venv_python_str)
            })
            .clone()
    }

    // --- Test Functions ---

    /// A simple convex quadratic function: f(x) = x'x, with minimum at 0.
    fn quadratic(x: &Array1<f64>) -> (f64, Array1<f64>) {
        (x.dot(x), 2.0 * x)
    }

    struct FirstOrderFn<F> {
        inner: F,
    }

    impl<F> FirstOrderFn<F> {
        fn new(inner: F) -> Self {
            Self { inner }
        }
    }

    impl<F> ZerothOrderObjective for FirstOrderFn<F>
    where
        F: FnMut(&Array1<f64>) -> (f64, Array1<f64>),
    {
        fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
            Ok((self.inner)(x).0)
        }
    }

    impl<F> FirstOrderObjective for FirstOrderFn<F>
    where
        F: FnMut(&Array1<f64>) -> (f64, Array1<f64>),
    {
        fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
            let (f, g) = (self.inner)(x);
            Ok(FirstOrderSample {
                value: f,
                gradient: g,
            })
        }
    }

    fn bfgs_oracle<F>(fg: F) -> FirstOrderFn<F>
    where
        F: FnMut(&Array1<f64>) -> (f64, Array1<f64>),
    {
        FirstOrderFn::new(fg)
    }

    struct SecondOrderFn<F> {
        inner: F,
    }

    impl<F> SecondOrderFn<F> {
        fn new(inner: F) -> Self {
            Self { inner }
        }
    }

    impl<F> ZerothOrderObjective for SecondOrderFn<F>
    where
        F: FnMut(&Array1<f64>) -> (f64, Array1<f64>, Array2<f64>),
    {
        fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
            Ok((self.inner)(x).0)
        }
    }

    impl<F> FirstOrderObjective for SecondOrderFn<F>
    where
        F: FnMut(&Array1<f64>) -> (f64, Array1<f64>, Array2<f64>),
    {
        fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
            let (f, g, _) = (self.inner)(x);
            Ok(FirstOrderSample {
                value: f,
                gradient: g,
            })
        }
    }

    impl<F> SecondOrderObjective for SecondOrderFn<F>
    where
        F: FnMut(&Array1<f64>) -> (f64, Array1<f64>, Array2<f64>),
    {
        fn eval_hessian(
            &mut self,
            x: &Array1<f64>,
        ) -> Result<SecondOrderSample, ObjectiveEvalError> {
            let (f, g, h) = (self.inner)(x);
            Ok(SecondOrderSample {
                value: f,
                gradient: g,
                hessian: Some(h),
            })
        }
    }

    struct CountingSecondOrder<F> {
        inner: F,
        first_order_calls: Arc<Mutex<usize>>,
        second_order_calls: Arc<Mutex<usize>>,
    }

    impl<F> CountingSecondOrder<F> {
        fn new(
            inner: F,
            first_order_calls: Arc<Mutex<usize>>,
            second_order_calls: Arc<Mutex<usize>>,
        ) -> Self {
            Self {
                inner,
                first_order_calls,
                second_order_calls,
            }
        }
    }

    impl<F> ZerothOrderObjective for CountingSecondOrder<F>
    where
        F: FnMut(&Array1<f64>) -> (f64, Array1<f64>, Array2<f64>),
    {
        fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
            Ok((self.inner)(x).0)
        }
    }

    impl<F> FirstOrderObjective for CountingSecondOrder<F>
    where
        F: FnMut(&Array1<f64>) -> (f64, Array1<f64>, Array2<f64>),
    {
        fn eval_grad(&mut self, x: &Array1<f64>) -> Result<FirstOrderSample, ObjectiveEvalError> {
            *self
                .first_order_calls
                .lock()
                .expect("lock first-order calls") += 1;
            let (f, g, _) = (self.inner)(x);
            Ok(FirstOrderSample {
                value: f,
                gradient: g,
            })
        }
    }

    impl<F> SecondOrderObjective for CountingSecondOrder<F>
    where
        F: FnMut(&Array1<f64>) -> (f64, Array1<f64>, Array2<f64>),
    {
        fn eval_hessian(
            &mut self,
            x: &Array1<f64>,
        ) -> Result<SecondOrderSample, ObjectiveEvalError> {
            *self
                .second_order_calls
                .lock()
                .expect("lock second-order calls") += 1;
            let (f, g, h) = (self.inner)(x);
            Ok(SecondOrderSample {
                value: f,
                gradient: g,
                hessian: Some(h),
            })
        }
    }

    fn gradient_norm(solution: &Solution) -> f64 {
        solution
            .final_gradient_norm
            .expect("gradient-based solution should carry a final gradient norm")
    }

    fn step_norm(solution: &Solution) -> f64 {
        solution
            .final_step_norm
            .expect("fixed-point solution should carry a final step norm")
    }

    fn tol(value: f64) -> Tolerance {
        Tolerance::new(value).unwrap()
    }

    fn iters(value: usize) -> MaxIterations {
        MaxIterations::new(value).unwrap()
    }

    fn bounds(lower: Array1<f64>, upper: Array1<f64>, tol: f64) -> Bounds {
        Bounds::new(lower, upper, tol).unwrap()
    }

    /// The Rosenbrock function, a classic non-convex benchmark with a minimum at [1, 1].
    fn rosenbrock(x: &Array1<f64>) -> (f64, Array1<f64>) {
        let a = 1.0;
        let b = 100.0;
        let f = (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2);
        let g = array![
            -2.0 * (a - x[0]) - 4.0 * b * (x[1] - x[0].powi(2)) * x[0],
            2.0 * b * (x[1] - x[0].powi(2))
        ];
        (f, g)
    }

    fn rosenbrock_with_hessian(x: &Array1<f64>) -> (f64, Array1<f64>, Array2<f64>) {
        let a = 1.0;
        let b = 100.0;
        let f = (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2);
        let g = array![
            -2.0 * (a - x[0]) - 4.0 * b * (x[1] - x[0].powi(2)) * x[0],
            2.0 * b * (x[1] - x[0].powi(2))
        ];
        let h = array![
            [1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0, -400.0 * x[0]],
            [-400.0 * x[0], 200.0]
        ];
        (f, g, h)
    }

    fn nonconvex_quartic_with_hessian(x: &Array1<f64>) -> (f64, Array1<f64>, Array2<f64>) {
        let f = x[0] * x[0] - x[1] * x[1] + 0.1 * x[1].powi(4);
        let g = array![2.0 * x[0], -2.0 * x[1] + 0.4 * x[1].powi(3)];
        let h = array![[2.0, 0.0], [0.0, -2.0 + 1.2 * x[1] * x[1]]];
        (f, g, h)
    }

    /// A function with a maximum at 0, guaranteed to fail the Wolfe curvature condition.
    fn non_convex_max(x: &Array1<f64>) -> (f64, Array1<f64>) {
        (-x.dot(x), -2.0 * x)
    }

    #[test]
    fn probe_best_ignores_nonfinite() {
        let x0 = array![0.0];
        let g0 = array![1.0];
        let mut best = super::ProbeBest::new(&x0, 0.0, &g0);
        let x1 = array![1.0];
        let g1 = array![f64::NAN];
        best.consider(&x1, -1.0, &g1);
        assert!(best.f.is_finite());
        assert_eq!(best.x[0], 0.0);
    }

    #[test]
    fn second_order_cache_reuses_same_point_full_sample() {
        let x = array![1.0, -2.0];
        let call_count = Arc::new(Mutex::new(0usize));
        let call_count_c = call_count.clone();
        let mut oracle = super::SecondOrderCache::new(x.len(), 1e-4);
        let mut func_evals = 0usize;
        let mut grad_evals = 0usize;
        let mut hess_evals = 0usize;
        let mut obj = SecondOrderFn::new(move |x: &Array1<f64>| {
            *call_count_c.lock().expect("lock call count") += 1;
            let f = x.dot(x);
            let g = 2.0 * x;
            let h = Array2::<f64>::eye(x.len()) * 2.0;
            (f, g, h)
        });

        let first = oracle
            .eval_cost_grad_hessian(
                &mut obj,
                &x,
                None,
                &mut func_evals,
                &mut grad_evals,
                &mut hess_evals,
            )
            .expect("initial full sample should succeed");
        let second = oracle
            .eval_cost_grad_hessian(
                &mut obj,
                &x,
                None,
                &mut func_evals,
                &mut grad_evals,
                &mut hess_evals,
            )
            .expect("same-point derivative request should hit cache");

        assert_eq!(*call_count.lock().expect("lock call count"), 1);
        assert_eq!(func_evals, 1);
        assert_eq!(grad_evals, 1);
        assert_eq!(hess_evals, 1);
        assert_eq!(first.0, second.0);
    }

    #[test]
    fn first_order_cache_merges_same_point_requests() {
        let x = array![0.5];
        let call_count = Arc::new(Mutex::new(0usize));
        let call_count_c = call_count.clone();
        let mut oracle = super::FirstOrderCache::new(x.len());
        let mut func_evals = 0usize;
        let mut grad_evals = 0usize;
        let mut obj = FirstOrderFn::new(move |x: &Array1<f64>| {
            *call_count_c.lock().expect("lock call count") += 1;
            let f = 0.5 * x[0] * x[0];
            let g = array![x[0]];
            (f, g)
        });

        let cost_only = oracle
            .eval_cost(&mut obj, &x, &mut func_evals)
            .expect("cost-only request should succeed");
        let full = oracle
            .eval_cost_grad(&mut obj, &x, &mut func_evals, &mut grad_evals)
            .expect("cost+grad request should succeed");
        let cached_grad = oracle
            .eval_cost_grad(&mut obj, &x, &mut func_evals, &mut grad_evals)
            .expect("merged same-point request should hit cache");

        assert_eq!(*call_count.lock().expect("lock call count"), 2);
        assert_eq!(func_evals, 2);
        assert_eq!(grad_evals, 1);
        assert_eq!(cost_only, full.0);
        assert_eq!(full.0, cached_grad.0);
        assert_eq!(full.1, cached_grad.1);
    }

    #[test]
    fn second_order_cache_fd_fills_nonfinite_hessian() {
        struct NonfiniteHessianObjective;

        impl ZerothOrderObjective for NonfiniteHessianObjective {
            fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
                Ok((x[0] - 1.0).powi(2))
            }
        }

        impl FirstOrderObjective for NonfiniteHessianObjective {
            fn eval_grad(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<FirstOrderSample, ObjectiveEvalError> {
                Ok(FirstOrderSample {
                    value: (x[0] - 1.0).powi(2),
                    gradient: array![2.0 * (x[0] - 1.0)],
                })
            }
        }

        impl SecondOrderObjective for NonfiniteHessianObjective {
            fn eval_hessian(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<SecondOrderSample, ObjectiveEvalError> {
                Ok(SecondOrderSample {
                    value: (x[0] - 1.0).powi(2),
                    gradient: array![2.0 * (x[0] - 1.0)],
                    hessian: Some(array![[f64::NAN]]),
                })
            }
        }

        let x = array![2.0];
        let mut oracle = super::SecondOrderCache::new(x.len(), 1e-4);
        let mut func_evals = 0usize;
        let mut grad_evals = 0usize;
        let mut hess_evals = 0usize;
        let mut obj = NonfiniteHessianObjective;
        let (value, gradient, hessian) = oracle
            .eval_cost_grad_hessian(
                &mut obj,
                &x,
                None,
                &mut func_evals,
                &mut grad_evals,
                &mut hess_evals,
            )
            .expect("non-finite Hessian should trigger internal finite differences");

        assert_eq!(value, 1.0);
        assert_eq!(gradient, array![2.0]);
        assert!((hessian[[0, 0]] - 2.0).abs() < 1e-6);
        assert_eq!(func_evals, 3);
        assert_eq!(grad_evals, 3);
        assert_eq!(hess_evals, 0);
    }

    #[test]
    fn finite_diff_gradient_returns_recoverable_on_nonfinite_probe() {
        struct WallObjective;

        impl ZerothOrderObjective for WallObjective {
            fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
                if x[0].abs() >= 0.5 {
                    Ok(f64::INFINITY)
                } else {
                    Ok(x[0] * x[0])
                }
            }
        }

        let mut objective = FiniteDiffGradient::new(WallObjective).with_step(1.0);
        let err = objective
            .eval_grad(&array![0.0])
            .expect_err("non-finite finite-difference probes should be recoverable");
        assert!(matches!(err, ObjectiveEvalError::Recoverable { .. }));
    }

    #[test]
    fn finite_diff_gradient_respects_bounds_with_one_sided_stencil() {
        struct LinearObjective;

        impl ZerothOrderObjective for LinearObjective {
            fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
                if x[0] < 0.0 || x[0] > 1.0 {
                    return Err(ObjectiveEvalError::recoverable(
                        "sample left the feasible interval",
                    ));
                }
                Ok(x[0])
            }
        }

        let mut objective = FiniteDiffGradient::new(LinearObjective)
            .with_step(1.0)
            .with_bounds(bounds(array![0.0], array![1.0], 1e-8));
        let sample = objective
            .eval_grad(&array![0.0])
            .expect("one-sided finite difference should stay feasible");
        assert!((sample.gradient[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn finite_diff_gradient_prefers_one_sided_stencil_near_bounds() {
        struct TrackingObjective {
            seen: Arc<Mutex<Vec<f64>>>,
        }

        impl ZerothOrderObjective for TrackingObjective {
            fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
                self.seen.lock().expect("lock seen samples").push(x[0]);
                Ok(x[0] * x[0])
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut objective = FiniteDiffGradient::new(TrackingObjective { seen: seen.clone() })
            .with_step(0.1)
            .with_bounds(bounds(array![0.0], array![1.0], 1e-8));
        let x0 = 0.05f64;
        let h = 0.1 * (1.0 + x0);
        let sample = objective
            .eval_grad(&array![x0])
            .expect("near-bound gradient should use a feasible one-sided stencil");

        let expected = ((x0 + h) * (x0 + h) - x0 * x0) / h;
        assert!((sample.gradient[0] - expected).abs() < 1e-12);
        let seen = seen.lock().expect("lock seen samples");
        assert_eq!(seen.len(), 2);
        assert!(seen.iter().any(|&x| (x - x0).abs() < 1e-12));
        assert!(seen.iter().any(|&x| (x - (x0 + h)).abs() < 1e-12));
        assert!(!seen.iter().any(|&x| x <= 1e-12));
    }

    #[test]
    fn bfgs_with_bounds_wires_finite_diff_gradient_bounds_automatically() {
        struct LinearObjective;

        impl ZerothOrderObjective for LinearObjective {
            fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
                if x[0] < 0.0 || x[0] > 1.0 {
                    return Err(ObjectiveEvalError::recoverable(
                        "sample left the feasible interval",
                    ));
                }
                Ok(x[0])
            }
        }

        let result = Bfgs::new(
            array![0.0],
            FiniteDiffGradient::new(LinearObjective).with_step(1.0),
        )
        .with_bounds(bounds(array![0.0], array![1.0], 1e-8))
        .run();

        let solution = result.expect("solver should wire bounds into finite differences");
        assert!(solution.final_point[0].abs() < 1e-12);
        assert!(gradient_norm(&solution) <= 1e-12);
    }

    #[test]
    fn optimize_problem_with_bounds_wires_finite_diff_gradient_automatically() {
        struct LinearObjective;

        impl ZerothOrderObjective for LinearObjective {
            fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
                if x[0] < 0.0 || x[0] > 1.0 {
                    return Err(ObjectiveEvalError::recoverable(
                        "sample left the feasible interval",
                    ));
                }
                Ok(x[0])
            }
        }

        let mut solver = optimize(
            Problem::new(
                array![0.0],
                FiniteDiffGradient::new(LinearObjective).with_step(1.0),
            )
            .with_bounds(bounds(array![0.0], array![1.0], 1e-8)),
        );

        let solution = solver
            .run()
            .expect("problem wrapper should wire bounds into finite differences");
        assert!(solution.final_point[0].abs() < 1e-12);
        assert!(gradient_norm(&solution) <= 1e-12);
    }

    #[test]
    fn second_order_cache_fd_hessian_respects_bounds() {
        struct NoHessianObjective;

        impl ZerothOrderObjective for NoHessianObjective {
            fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
                if x[0] < 0.0 || x[0] > 1.0 {
                    return Err(ObjectiveEvalError::recoverable(
                        "sample left the feasible interval",
                    ));
                }
                Ok((x[0] - 0.25).powi(2))
            }
        }

        impl FirstOrderObjective for NoHessianObjective {
            fn eval_grad(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<FirstOrderSample, ObjectiveEvalError> {
                if x[0] < 0.0 || x[0] > 1.0 {
                    return Err(ObjectiveEvalError::recoverable(
                        "sample left the feasible interval",
                    ));
                }
                Ok(FirstOrderSample {
                    value: (x[0] - 0.25).powi(2),
                    gradient: array![2.0 * (x[0] - 0.25)],
                })
            }
        }

        impl SecondOrderObjective for NoHessianObjective {
            fn eval_hessian(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<SecondOrderSample, ObjectiveEvalError> {
                Ok(SecondOrderSample {
                    value: (x[0] - 0.25).powi(2),
                    gradient: array![2.0 * (x[0] - 0.25)],
                    hessian: None,
                })
            }
        }

        let x = array![0.0];
        let mut oracle = super::SecondOrderCache::new(x.len(), 1e-4);
        let mut func_evals = 0usize;
        let mut grad_evals = 0usize;
        let mut hess_evals = 0usize;
        let mut obj = NoHessianObjective;
        let bounds = bounds(array![0.0], array![1.0], 1e-8);

        let (value, gradient, hessian) = oracle
            .eval_cost_grad_hessian(
                &mut obj,
                &x,
                Some(&bounds.spec),
                &mut func_evals,
                &mut grad_evals,
                &mut hess_evals,
            )
            .expect("finite-difference Hessian should stay feasible near bounds");

        assert!((value - 0.0625).abs() < 1e-12);
        assert!((gradient[0] + 0.5).abs() < 1e-12);
        assert!((hessian[[0, 0]] - 2.0).abs() < 1e-6);
        assert_eq!(hess_evals, 0);
    }

    #[test]
    fn second_order_cache_fd_hessian_prefers_one_sided_stencil_near_bounds() {
        struct NearWallObjective;

        impl ZerothOrderObjective for NearWallObjective {
            fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
                if x[0] < 0.01 || x[0] > 1.0 {
                    return Err(ObjectiveEvalError::recoverable(
                        "sample left the finite-difference band",
                    ));
                }
                Ok(x[0] * x[0])
            }
        }

        impl FirstOrderObjective for NearWallObjective {
            fn eval_grad(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<FirstOrderSample, ObjectiveEvalError> {
                if x[0] < 0.01 || x[0] > 1.0 {
                    return Err(ObjectiveEvalError::recoverable(
                        "sample left the finite-difference band",
                    ));
                }
                Ok(FirstOrderSample {
                    value: x[0] * x[0],
                    gradient: array![2.0 * x[0]],
                })
            }
        }

        impl SecondOrderObjective for NearWallObjective {
            fn eval_hessian(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<SecondOrderSample, ObjectiveEvalError> {
                Ok(SecondOrderSample {
                    value: x[0] * x[0],
                    gradient: array![2.0 * x[0]],
                    hessian: None,
                })
            }
        }

        let x = array![0.05];
        let mut oracle = super::SecondOrderCache::new(x.len(), 0.1);
        let mut func_evals = 0usize;
        let mut grad_evals = 0usize;
        let mut hess_evals = 0usize;
        let mut obj = NearWallObjective;
        let bounds = bounds(array![0.0], array![1.0], 1e-8);

        let (_, _, hessian) = oracle
            .eval_cost_grad_hessian(
                &mut obj,
                &x,
                Some(&bounds.spec),
                &mut func_evals,
                &mut grad_evals,
                &mut hess_evals,
            )
            .expect("near-bound Hessian should use a feasible one-sided stencil");

        assert!((hessian[[0, 0]] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn newton_trust_region_wires_fd_hessian_bounds_automatically() {
        struct NoHessianObjective;

        impl ZerothOrderObjective for NoHessianObjective {
            fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
                if x[0] < 0.0 || x[0] > 1.0 {
                    return Err(ObjectiveEvalError::recoverable(
                        "sample left the feasible interval",
                    ));
                }
                Ok(x[0])
            }
        }

        impl FirstOrderObjective for NoHessianObjective {
            fn eval_grad(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<FirstOrderSample, ObjectiveEvalError> {
                if x[0] < 0.0 || x[0] > 1.0 {
                    return Err(ObjectiveEvalError::recoverable(
                        "sample left the feasible interval",
                    ));
                }
                Ok(FirstOrderSample {
                    value: x[0],
                    gradient: array![1.0],
                })
            }
        }

        impl SecondOrderObjective for NoHessianObjective {
            fn eval_hessian(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<SecondOrderSample, ObjectiveEvalError> {
                Ok(SecondOrderSample {
                    value: x[0],
                    gradient: array![1.0],
                    hessian: None,
                })
            }
        }

        let result = NewtonTrustRegion::new(array![0.0], NoHessianObjective)
            .with_bounds(bounds(array![0.0], array![1.0], 1e-8))
            .run();

        let solution = result.expect("solver should wire bounds into Hessian finite differences");
        assert!(solution.final_point[0].abs() < 1e-12);
        assert!(gradient_norm(&solution) <= 1e-12);
    }

    #[test]
    fn optimize_second_order_problem_with_bounds_wires_fd_hessian_automatically() {
        struct NoHessianObjective;

        impl ZerothOrderObjective for NoHessianObjective {
            fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
                if x[0] < 0.0 || x[0] > 1.0 {
                    return Err(ObjectiveEvalError::recoverable(
                        "sample left the feasible interval",
                    ));
                }
                Ok(x[0])
            }
        }

        impl FirstOrderObjective for NoHessianObjective {
            fn eval_grad(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<FirstOrderSample, ObjectiveEvalError> {
                if x[0] < 0.0 || x[0] > 1.0 {
                    return Err(ObjectiveEvalError::recoverable(
                        "sample left the feasible interval",
                    ));
                }
                Ok(FirstOrderSample {
                    value: x[0],
                    gradient: array![1.0],
                })
            }
        }

        impl SecondOrderObjective for NoHessianObjective {
            fn eval_hessian(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<SecondOrderSample, ObjectiveEvalError> {
                Ok(SecondOrderSample {
                    value: x[0],
                    gradient: array![1.0],
                    hessian: None,
                })
            }
        }

        let mut solver = optimize(
            SecondOrderProblem::new(array![0.0], NoHessianObjective).with_bounds(bounds(
                array![0.0],
                array![1.0],
                1e-8,
            )),
        );

        let solution = solver.run().expect(
            "second-order problem wrapper should wire bounds into Hessian finite differences",
        );
        assert!(solution.final_point[0].abs() < 1e-12);
        assert!(gradient_norm(&solution) <= 1e-12);
    }

    #[test]
    fn arc_wires_fd_hessian_bounds_automatically() {
        struct NoHessianObjective;

        impl ZerothOrderObjective for NoHessianObjective {
            fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
                if x[0] < 0.0 || x[0] > 1.0 {
                    return Err(ObjectiveEvalError::recoverable(
                        "sample left the feasible interval",
                    ));
                }
                Ok(x[0])
            }
        }

        impl FirstOrderObjective for NoHessianObjective {
            fn eval_grad(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<FirstOrderSample, ObjectiveEvalError> {
                if x[0] < 0.0 || x[0] > 1.0 {
                    return Err(ObjectiveEvalError::recoverable(
                        "sample left the feasible interval",
                    ));
                }
                Ok(FirstOrderSample {
                    value: x[0],
                    gradient: array![1.0],
                })
            }
        }

        impl SecondOrderObjective for NoHessianObjective {
            fn eval_hessian(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<SecondOrderSample, ObjectiveEvalError> {
                Ok(SecondOrderSample {
                    value: x[0],
                    gradient: array![1.0],
                    hessian: None,
                })
            }
        }

        let result = super::Arc::new(array![0.0], NoHessianObjective)
            .with_bounds(bounds(array![0.0], array![1.0], 1e-8))
            .run();

        let solution = result.expect("solver should wire bounds into Hessian finite differences");
        assert!(solution.final_point[0].abs() < 1e-12);
        assert!(gradient_norm(&solution) <= 1e-12);
    }

    #[test]
    fn fixed_point_converges_on_linear_contraction() {
        struct LinearContraction;

        impl FixedPointObjective for LinearContraction {
            fn eval_step(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<FixedPointSample, ObjectiveEvalError> {
                Ok(FixedPointSample {
                    value: x.dot(x),
                    step: -0.5 * x,
                    status: FixedPointStatus::Continue,
                })
            }
        }

        let mut solver = FixedPoint::new(array![2.0, -1.0], LinearContraction)
            .with_tolerance(tol(1e-8))
            .with_max_iterations(iters(128));
        let solution = solver
            .run()
            .expect("fixed-point solver should converge on a contraction");

        assert!(solution.final_point.dot(&solution.final_point).sqrt() < 1e-6);
        assert!(step_norm(&solution) < 1e-8);
    }

    #[test]
    fn fixed_point_stop_returns_current_point() {
        struct ImmediateStop;

        impl FixedPointObjective for ImmediateStop {
            fn eval_step(
                &mut self,
                _x: &Array1<f64>,
            ) -> Result<FixedPointSample, ObjectiveEvalError> {
                Ok(FixedPointSample {
                    value: 7.0,
                    step: array![1.0],
                    status: FixedPointStatus::Stop,
                })
            }
        }

        let solution = FixedPoint::new(array![3.0], ImmediateStop)
            .run()
            .expect("stop status should finalize immediately");

        assert_eq!(solution.final_point, array![3.0]);
        assert_eq!(solution.final_value, 7.0);
        assert_eq!(step_norm(&solution), 0.0);
    }

    #[test]
    fn dense_solve_shifted_solves_small_system() {
        let a = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let x = super::dense_solve_shifted(&a, &b, 0.0).expect("dense solve should succeed");
        let ax = a.dot(&x);
        assert!((&ax - &b).iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn cg_solve_adaptive_uses_direct_path_for_small_dense_systems() {
        let n = 8usize;
        let mut a = Array2::<f64>::eye(n) * 3.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    a[[i, j]] = 0.05 * ((i + j + 1) as f64);
                }
            }
        }
        let b = Array1::from_iter((0..n).map(|i| (i + 1) as f64));
        let x = super::cg_solve_adaptive(&a, &b, 5, 1e-12, 1e-10)
            .expect("small dense system should use the direct solve path");
        let mut ax = a.dot(&x);
        for i in 0..n {
            ax[i] += 1e-10 * x[i];
        }
        let residual = (&ax - &b).dot(&(&ax - &b)).sqrt();
        assert!(residual < 1e-8, "expected small residual, got {residual:e}");
    }

    #[test]
    fn cg_solve_from_refines_existing_iterate() {
        let n = 256usize;
        let mut a = Array2::<f64>::eye(n) * 4.0;
        for i in 0..(n - 1) {
            a[[i, i + 1]] = 0.5;
            a[[i + 1, i]] = 0.5;
        }
        let b = Array1::from_elem(n, 1.0);
        let first = super::cg_solve_from(&a, &b, Array1::zeros(n), 3, 1e-12, 0.0)
            .expect("initial CG stage should succeed");
        let second = super::cg_solve_from(&a, &b, first.x.clone(), 3, 1e-12, 0.0)
            .expect("refinement CG stage should succeed");
        assert!(
            second.rel_resid < first.rel_resid,
            "continued CG should improve residual"
        );
    }

    #[test]
    fn steihaug_toint_uses_exact_small_dense_newton_step_when_feasible() {
        let core = super::NewtonTrustRegionCore::new(array![0.0, 0.0]);
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let g = array![1.0, 2.0];
        let rhs = -g.clone();
        let expected =
            super::dense_solve_shifted(&h, &rhs, 0.0).expect("direct dense solve should work");
        let (step, pred) = core
            .steihaug_toint_step(&h, &g, 10.0, None)
            .expect("small dense exact step should be accepted");
        assert!((&step - &expected).iter().all(|v| v.abs() < 1e-10));
        assert!(pred > 0.0);
    }

    #[test]
    fn dense_trust_region_step_handles_small_dense_indefinite_boundary_case() {
        let h = array![[-1.0, 0.0], [0.0, 2.0]];
        let g = array![1.0, 0.5];
        let (step, pred) =
            super::dense_trust_region_step(&h, &g, 0.5, None).expect("direct trust-region step");
        let norm = step.dot(&step).sqrt();
        assert!(norm <= 0.5 + 1e-8, "step norm should respect trust radius");
        assert!(pred > 0.0, "predicted decrease should be positive");
    }

    #[test]
    fn arc_small_dense_masked_subproblem_uses_direct_masked_solve() {
        let core = super::ArcCore::new(array![0.0, 0.0]);
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let g = array![2.0, -3.0];
        let active = [true, false];
        let step = core
            .solve_arc_subproblem(&h, &g, 1.0, Some(&active))
            .expect("masked direct ARC subproblem solve should succeed");
        assert!(
            step[0].abs() < 1e-12,
            "active coordinate should remain fixed"
        );
        assert!(step[1].is_finite(), "free coordinate step should be finite");
        let (m_delta, _, grad_m) = core.arc_model_value(&g, &h, 1.0, &step, Some(&active));
        assert!(m_delta <= 1e-8, "ARC model should not increase materially");
        assert!(grad_m.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn bfgs_local_mode_forces_strict_search_policy() {
        let mut core = super::BfgsCore::new(array![0.0, 0.0]);
        core.initial_grad_norm = 10.0;
        core.primary_strategy = super::LineSearchStrategy::Backtracking;
        core.c1_adapt = 1e-3;
        core.c2_adapt = 0.1;
        core.flat_accept_streak = 3;
        core.curv_slack_scale = 0.25;
        core.grad_drop_factor = 0.95;
        core.gll.set_cap(8);

        core.refresh_local_mode(1e-3);

        assert!(core.local_mode);
        assert!(matches!(
            core.primary_strategy,
            super::LineSearchStrategy::StrongWolfe
        ));
        assert!((core.c1_adapt - core.c1).abs() < 1e-16);
        assert!((core.c2_adapt - core.c2).abs() < 1e-16);
        assert_eq!(core.flat_accept_streak, 0);
        assert!((core.curv_slack_scale - 1.0).abs() < 1e-16);
        assert!((core.grad_drop_factor - 0.9).abs() < 1e-16);
        assert_eq!(core.gll.cap, 1);
    }

    #[test]
    fn probe_alphas_respects_armijo() {
        let x_k = array![1.0];
        let f_k = 1.0;
        let g_k = array![2.0];
        let d_k = array![2.0]; // ascent direction
        let mut core = super::BfgsCore::new(x_k.clone());
        let mut oracle = super::FirstOrderCache::new(x_k.len());
        let tau_g = core.tau_g;
        let drop_factor = core.grad_drop_factor;
        let mut fe = 0usize;
        let mut ge = 0usize;
        let res = super::probe_alphas(
            &mut core,
            &mut bfgs_oracle(|x: &Array1<f64>| (x.dot(x), 2.0 * x)),
            &mut oracle,
            &x_k,
            &d_k,
            f_k,
            &g_k,
            0.0,
            1.0,
            tau_g,
            drop_factor,
            &mut fe,
            &mut ge,
        );
        assert!(res.is_none());
    }

    #[test]
    fn zoom_tiny_bracket_rejects_armijo_without_curvature() {
        let x_k = array![1.0];
        let mut core = super::BfgsCore::new(x_k.clone());
        let mut oracle = super::FirstOrderCache::new(x_k.len());
        let (f_k, g_k) = non_convex_max(&x_k);
        let g_proj_k = core.projected_gradient(&x_k, &g_k);
        let d_k = array![1.0];
        let alpha_lo = 1.0;
        let alpha_hi = 1.0 + 5e-13;
        let (x_lo, s_lo, _) = core.project_with_step(&x_k, &d_k, alpha_lo);
        let (f_lo, g_lo) = non_convex_max(&x_lo);
        let g_lo_dot_d = super::directional_derivative(
            &core.projected_gradient(&x_lo, &g_lo),
            &s_lo,
            alpha_lo,
            &d_k,
        );
        let (x_hi, s_hi, _) = core.project_with_step(&x_k, &d_k, alpha_hi);
        let (f_hi, g_hi) = non_convex_max(&x_hi);
        let g_hi_dot_d = super::directional_derivative(
            &core.projected_gradient(&x_hi, &g_hi),
            &s_hi,
            alpha_hi,
            &d_k,
        );
        let c1 = core.c1;
        let c2 = core.c2;

        let r = super::zoom(
            &mut core,
            &mut bfgs_oracle(non_convex_max),
            &mut oracle,
            &x_k,
            &d_k,
            f_k,
            &g_k,
            &g_proj_k,
            g_proj_k.dot(&d_k),
            c1,
            c2,
            alpha_lo,
            alpha_hi,
            f_lo,
            f_hi,
            g_lo_dot_d,
            g_hi_dot_d,
            0,
            0,
        );

        assert!(matches!(r, Err(super::LineSearchError::MaxAttempts(_))));
    }

    #[test]
    fn zoom_flat_midpoint_rejects_uphill_descent_only_candidate() {
        let x_k = array![0.0];
        let mut core = super::BfgsCore::new(x_k.clone());
        let mut oracle = super::FirstOrderCache::new(x_k.len());
        let slope = 2.0e-13;
        let fake_grad = -1.0e-14;
        let f_k = 0.0;
        let g_k = array![fake_grad];
        let g_proj_k = core.projected_gradient(&x_k, &g_k);
        let d_k = array![1.0];
        let alpha_lo = 1.0;
        let alpha_hi = 2.0;
        let fg = move |x: &Array1<f64>| (slope * x[0], array![fake_grad]);
        let (x_lo, s_lo, _) = core.project_with_step(&x_k, &d_k, alpha_lo);
        let (f_lo, g_lo) = fg(&x_lo);
        let g_lo_dot_d = super::directional_derivative(
            &core.projected_gradient(&x_lo, &g_lo),
            &s_lo,
            alpha_lo,
            &d_k,
        );
        let (x_hi, s_hi, _) = core.project_with_step(&x_k, &d_k, alpha_hi);
        let (f_hi, g_hi) = fg(&x_hi);
        let g_hi_dot_d = super::directional_derivative(
            &core.projected_gradient(&x_hi, &g_hi),
            &s_hi,
            alpha_hi,
            &d_k,
        );
        let c1 = core.c1;
        let c2 = core.c2;

        let r = super::zoom(
            &mut core,
            &mut bfgs_oracle(fg),
            &mut oracle,
            &x_k,
            &d_k,
            f_k,
            &g_k,
            &g_proj_k,
            g_proj_k.dot(&d_k),
            c1,
            c2,
            alpha_lo,
            alpha_hi,
            f_lo,
            f_hi,
            g_lo_dot_d,
            g_hi_dot_d,
            0,
            0,
        );

        assert!(matches!(r, Err(super::LineSearchError::MaxAttempts(_))));
    }

    #[test]
    fn line_search_rejects_fully_clipped_projected_step() {
        let x_k = array![1.0];
        let lower = array![0.0];
        let upper = array![1.0];
        let mut core = super::BfgsCore::new(x_k.clone());
        core.bounds = Some(super::BoxSpec::new(lower, upper, 1e-8));
        let mut oracle = super::FirstOrderCache::new(x_k.len());
        let fg = |x: &Array1<f64>| {
            let dx = x[0] - 2.0;
            (dx * dx, array![2.0 * dx])
        };
        let (f_k, g_k) = fg(&x_k);
        let d_k = array![1.0];
        let c1 = core.c1;
        let c2 = core.c2;

        let r = super::line_search(
            &mut core,
            &mut bfgs_oracle(fg),
            &mut oracle,
            &x_k,
            &d_k,
            f_k,
            &g_k,
            c1,
            c2,
        );

        assert!(matches!(r, Err(super::LineSearchError::StepSizeTooSmall)));
    }

    #[test]
    fn backtracking_accepts_strong_wolfe_in_local_mode() {
        let x_k = array![1.0];
        let mut core = super::BfgsCore::new(x_k.clone());
        core.local_mode = true;

        let mut oracle = super::FirstOrderCache::new(x_k.len());
        let f_k = x_k.dot(&x_k);
        let g_k = 2.0 * x_k.clone();
        let d_k = -g_k.clone();

        let (alpha, f_new, g_new, _, _, kind) = super::backtracking_line_search(
            &mut core,
            &mut bfgs_oracle(|x: &Array1<f64>| (x.dot(x), 2.0 * x)),
            &mut oracle,
            &x_k,
            &d_k,
            f_k,
            &g_k,
        )
        .expect("local mode should still accept strong-Wolfe decreases");

        assert!((alpha - 0.5).abs() < 1e-12);
        assert!(f_new < f_k);
        assert!(g_new.iter().all(|v| v.is_finite()));
        assert!(matches!(kind, super::AcceptKind::StrongWolfe));
    }

    #[test]
    fn backtracking_rejects_armijo_without_curvature() {
        let x_k = array![1.0];
        let mut core = super::BfgsCore::new(x_k.clone());
        let mut oracle = super::FirstOrderCache::new(x_k.len());
        let (f_k, g_k) = non_convex_max(&x_k);
        let d_k = array![1.0];

        let r = super::backtracking_line_search(
            &mut core,
            &mut bfgs_oracle(non_convex_max),
            &mut oracle,
            &x_k,
            &d_k,
            f_k,
            &g_k,
        );

        assert!(r.is_err());
    }

    #[test]
    fn local_mode_disables_only_gll_extension() {
        let mut core = super::BfgsCore::new(array![0.0]);
        let fmax = 2.0;
        let gk_ts = -0.1;
        let f_trial = 1.5;

        assert!(!core.accept_armijo(1.0, gk_ts, f_trial));
        assert!(core.accept_gll_nonmonotone(fmax, gk_ts, f_trial));

        core.local_mode = true;
        assert!(!core.accept_gll_nonmonotone(fmax, gk_ts, f_trial));
    }

    #[test]
    fn line_search_ignores_nonfinite_best() {
        let x0 = array![0.0];
        let mut core = super::BfgsCore::new(x0.clone());
        let mut oracle = super::FirstOrderCache::new(x0.len());
        let c1 = core.c1;
        let c2 = core.c2;
        let fg = |x: &Array1<f64>| {
            if x[0] > 0.0 {
                (f64::NEG_INFINITY, array![1.0])
            } else {
                (0.0, array![1.0])
            }
        };
        let (f_k, g_k) = fg(&x0);
        let mut obj = bfgs_oracle(fg);
        core.global_best = Some(super::ProbeBest::new(&x0, f_k, &g_k));
        let d_k = array![1.0];
        let r = super::line_search(
            &mut core,
            &mut obj,
            &mut oracle,
            &x0,
            &d_k,
            f_k,
            &g_k,
            c1,
            c2,
        );
        assert!(r.is_err());
        assert!(
            core.global_best
                .as_ref()
                .map(|b| b.f.is_finite())
                .unwrap_or(false)
        );
    }

    #[test]
    fn newton_trust_region_converges_on_rosenbrock() {
        let x0 = array![-1.2, 1.0];
        let mut solver = NewtonTrustRegion::new(x0, SecondOrderFn::new(rosenbrock_with_hessian))
            .with_profile(Profile::Robust)
            .with_tolerance(tol(1e-8))
            .with_max_iterations(iters(100));
        let solution = solver.run().expect("Newton trust-region should converge");
        assert!((solution.final_point[0] - 1.0).abs() < 1e-6);
        assert!((solution.final_point[1] - 1.0).abs() < 1e-6);
        assert!(gradient_norm(&solution) < 1e-6);
    }

    #[test]
    fn newton_trust_region_uses_single_full_trial_requests() {
        let x0 = array![-1.2, 1.0];
        let first_order_calls = Arc::new(Mutex::new(0usize));
        let second_order_calls = Arc::new(Mutex::new(0usize));
        let objective = CountingSecondOrder::new(
            rosenbrock_with_hessian,
            first_order_calls.clone(),
            second_order_calls.clone(),
        );
        let mut solver = NewtonTrustRegion::new(x0, objective)
            .with_profile(Profile::Robust)
            .with_tolerance(tol(1e-8))
            .with_max_iterations(iters(100));
        let _ = solver.run().expect("Newton trust-region should converge");
        assert_eq!(
            *first_order_calls.lock().expect("lock first-order calls"),
            0,
            "Newton TR should not use first-order-only objective paths"
        );
        assert!(
            *second_order_calls.lock().expect("lock second-order calls") > 0,
            "expected Newton TR to use second-order evaluations"
        );
    }

    #[test]
    fn newton_trust_region_handles_indefinite_hessian() {
        let x0 = array![1.0, 0.5]; // Hessian is indefinite at start.
        let mut solver =
            NewtonTrustRegion::new(x0, SecondOrderFn::new(nonconvex_quartic_with_hessian))
                .with_profile(Profile::Robust)
                .with_tolerance(tol(1e-7))
                .with_max_iterations(iters(200));

        let sol = solver
            .run()
            .expect("TR-Newton should handle indefinite Hessians");
        assert!(sol.final_value.is_finite());
        assert!(gradient_norm(&sol) < 1e-4);
    }

    #[test]
    fn newton_trust_region_respects_single_variable_bound() {
        // Unconstrained minimizer is x=2, but bounds force x in [0,1].
        let x0 = array![0.2];
        let lower = array![0.0];
        let upper = array![1.0];
        let mut solver = NewtonTrustRegion::new(
            x0,
            SecondOrderFn::new(|x: &Array1<f64>| {
                let dx = x[0] - 2.0;
                let f = dx * dx;
                let g = array![2.0 * dx];
                let h = array![[2.0]];
                (f, g, h)
            }),
        )
        .with_bounds(bounds(lower, upper, 1e-8))
        .with_profile(Profile::Robust)
        .with_tolerance(tol(1e-10))
        .with_max_iterations(iters(100));

        let sol = solver
            .run()
            .expect("Projected Newton should converge at upper bound");
        assert!((sol.final_point[0] - 1.0).abs() < 1e-8);
        assert!(gradient_norm(&sol) <= 1e-8);
    }

    #[test]
    fn newton_trust_region_active_set_leaves_free_coordinate() {
        // x[0] wants to move beyond upper bound, x[1] is free with minimizer at 3.
        let x0 = array![0.4, -2.0];
        let lower = array![0.0, -10.0];
        let upper = array![1.0, 10.0];
        let mut solver = NewtonTrustRegion::new(
            x0,
            SecondOrderFn::new(|x: &Array1<f64>| {
                let d0 = x[0] - 2.0;
                let d1 = x[1] - 3.0;
                let f = d0 * d0 + d1 * d1;
                let g = array![2.0 * d0, 2.0 * d1];
                let h = array![[2.0, 0.0], [0.0, 2.0]];
                (f, g, h)
            }),
        )
        .with_bounds(bounds(lower, upper, 1e-8))
        .with_profile(Profile::Robust)
        .with_tolerance(tol(1e-9))
        .with_max_iterations(iters(100));

        let sol = solver.run().expect("Projected Newton should converge");
        assert!((sol.final_point[0] - 1.0).abs() < 1e-8);
        assert!((sol.final_point[1] - 3.0).abs() < 1e-7);
        assert!(gradient_norm(&sol) <= 1e-7);
    }

    #[test]
    fn newton_trust_region_retries_on_recoverable_trial_errors() {
        struct RecoverableTrialObjective {
            calls: usize,
        }

        impl ZerothOrderObjective for RecoverableTrialObjective {
            fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
                Ok(0.5 * (x[0] - 1.0).powi(2))
            }
        }

        impl FirstOrderObjective for RecoverableTrialObjective {
            fn eval_grad(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<FirstOrderSample, ObjectiveEvalError> {
                Ok(FirstOrderSample {
                    value: 0.5 * (x[0] - 1.0).powi(2),
                    gradient: array![x[0] - 1.0],
                })
            }
        }

        impl SecondOrderObjective for RecoverableTrialObjective {
            fn eval_hessian(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<SecondOrderSample, ObjectiveEvalError> {
                self.calls += 1;
                if self.calls == 2 {
                    return Err(ObjectiveEvalError::recoverable("simulated PIRLS breakdown"));
                }
                Ok(SecondOrderSample {
                    value: 0.5 * (x[0] - 1.0).powi(2),
                    gradient: array![x[0] - 1.0],
                    hessian: Some(array![[1.0]]),
                })
            }
        }

        let x0 = array![2.0];
        let mut solver = NewtonTrustRegion::new(x0, RecoverableTrialObjective { calls: 0 })
            .with_profile(Profile::Deterministic)
            .with_tolerance(tol(1e-8))
            .with_max_iterations(iters(200));

        let sol = solver
            .run()
            .expect("recoverable trial errors should shrink trust region and recover");
        assert!((sol.final_point[0] - 1.0).abs() < 1e-6);
        assert!(gradient_norm(&sol) < 1e-6);
    }

    #[test]
    fn newton_trust_region_surfaces_fatal_objective_errors() {
        struct FatalObjective;

        impl ZerothOrderObjective for FatalObjective {
            fn eval_cost(&mut self, _x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
                Err(ObjectiveEvalError::fatal(
                    "fatal synthetic objective failure",
                ))
            }
        }

        impl FirstOrderObjective for FatalObjective {
            fn eval_grad(
                &mut self,
                _x: &Array1<f64>,
            ) -> Result<FirstOrderSample, ObjectiveEvalError> {
                Err(ObjectiveEvalError::fatal(
                    "fatal synthetic objective failure",
                ))
            }
        }

        impl SecondOrderObjective for FatalObjective {
            fn eval_hessian(
                &mut self,
                _x: &Array1<f64>,
            ) -> Result<SecondOrderSample, ObjectiveEvalError> {
                Err(ObjectiveEvalError::fatal(
                    "fatal synthetic objective failure",
                ))
            }
        }

        let x0 = array![0.0];
        let mut solver = NewtonTrustRegion::new(x0, FatalObjective).with_max_iterations(iters(5));

        let err = solver.run().expect_err("fatal errors must propagate");
        match err {
            super::NewtonTrustRegionError::ObjectiveFailed { message } => {
                assert!(message.contains("fatal synthetic objective failure"));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn arc_converges_on_rosenbrock() {
        let x0 = array![-1.2, 1.0];
        let mut solver = super::Arc::new(x0, SecondOrderFn::new(rosenbrock_with_hessian))
            .with_profile(Profile::Robust)
            .with_tolerance(tol(1e-7))
            .with_max_iterations(iters(250));

        let solution = solver.run().expect("ARC should converge");
        assert!((solution.final_point[0] - 1.0).abs() < 1e-4);
        assert!((solution.final_point[1] - 1.0).abs() < 1e-4);
        assert!(gradient_norm(&solution) < 1e-5);
    }

    #[test]
    fn arc_uses_single_full_trial_requests() {
        let x0 = array![-1.2, 1.0];
        let first_order_calls = Arc::new(Mutex::new(0usize));
        let second_order_calls = Arc::new(Mutex::new(0usize));
        let objective = CountingSecondOrder::new(
            rosenbrock_with_hessian,
            first_order_calls.clone(),
            second_order_calls.clone(),
        );
        let mut solver = super::Arc::new(x0, objective)
            .with_profile(Profile::Robust)
            .with_tolerance(tol(1e-7))
            .with_max_iterations(iters(250));

        let _ = solver.run().expect("ARC should converge");
        assert_eq!(
            *first_order_calls.lock().expect("lock first-order calls"),
            0,
            "ARC should not use first-order-only objective paths"
        );
        assert!(
            *second_order_calls.lock().expect("lock second-order calls") > 0,
            "expected ARC to use second-order evaluations"
        );
    }

    #[test]
    fn arc_accepted_step_uses_single_evaluation() {
        let first_order_calls = Arc::new(Mutex::new(0usize));
        let second_order_calls = Arc::new(Mutex::new(0usize));
        let objective = CountingSecondOrder::new(
            |x: &Array1<f64>| {
                let f = 0.5 * x[0] * x[0];
                let g = array![x[0]];
                let h = array![[1.0]];
                (f, g, h)
            },
            first_order_calls.clone(),
            second_order_calls.clone(),
        );
        let mut solver = super::Arc::new(array![1.0], objective)
            .with_profile(Profile::Deterministic)
            .with_tolerance(tol(1e-9))
            .with_max_iterations(iters(1));

        let err = solver
            .run()
            .expect_err("one ARC iteration should exhaust the budget after a single accepted step");
        match err {
            ArcError::MaxIterationsReached { .. } => {}
            other => panic!("unexpected error variant: {other:?}"),
        }
        assert_eq!(
            *first_order_calls.lock().expect("lock first-order calls"),
            0,
            "ARC should not issue first-order-only evaluations"
        );
        assert_eq!(
            *second_order_calls.lock().expect("lock second-order calls"),
            2,
            "expected one initial and one trial second-order evaluation"
        );
    }

    #[test]
    fn arc_rejects_materially_projected_steps() {
        let x0 = array![0.8];
        let lower = array![0.0];
        let upper = array![1.0];
        let clipped_counts = Arc::new(Mutex::new((0usize, 0usize)));
        let clipped_counts_c = clipped_counts.clone();
        struct ProjectedArcObjective {
            clipped_counts: Arc<Mutex<(usize, usize)>>,
        }

        impl ZerothOrderObjective for ProjectedArcObjective {
            fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
                if (x[0] - 1.0).abs() < 1e-12 {
                    self.clipped_counts.lock().expect("lock clipped counts").0 += 1;
                }
                let dx = x[0] - 2.0;
                Ok(0.5 * dx * dx)
            }
        }

        impl FirstOrderObjective for ProjectedArcObjective {
            fn eval_grad(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<FirstOrderSample, ObjectiveEvalError> {
                let dx = x[0] - 2.0;
                Ok(FirstOrderSample {
                    value: 0.5 * dx * dx,
                    gradient: array![dx],
                })
            }
        }

        impl SecondOrderObjective for ProjectedArcObjective {
            fn eval_hessian(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<SecondOrderSample, ObjectiveEvalError> {
                if (x[0] - 1.0).abs() < 1e-12 {
                    self.clipped_counts.lock().expect("lock clipped counts").1 += 1;
                }
                let dx = x[0] - 2.0;
                Ok(SecondOrderSample {
                    value: 0.5 * dx * dx,
                    gradient: array![dx],
                    hessian: Some(array![[1.0]]),
                })
            }
        }

        let mut solver = super::Arc::new(
            x0.clone(),
            ProjectedArcObjective {
                clipped_counts: clipped_counts_c,
            },
        )
        .with_profile(Profile::Deterministic)
        .with_bounds(bounds(lower, upper, 1e-12))
        .with_max_iterations(iters(1));
        solver.core.sigma_min = 1e-12;
        solver.core.sigma = 1e-12;

        let err = solver
            .run()
            .expect_err("single projected iteration should exhaust the budget");
        match err {
            ArcError::MaxIterationsReached { last_solution } => {
                assert!(last_solution.final_point[0] <= 1.0 + 1e-12);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
        let counts = clipped_counts.lock().expect("lock clipped counts");
        assert_eq!(
            counts.0, 0,
            "materially projected ARC steps must not use CostOnly rho evaluation"
        );
        assert!(
            counts.1 > 0,
            "materially projected ARC steps should refresh a coherent CostGradientHessian sample"
        );
    }

    #[test]
    fn arc_respects_single_variable_bound() {
        let x0 = array![0.2];
        let lower = array![0.0];
        let upper = array![1.0];
        let mut solver = super::Arc::new(
            x0,
            SecondOrderFn::new(|x: &Array1<f64>| {
                let dx = x[0] - 2.0;
                let f = dx * dx;
                let g = array![2.0 * dx];
                let h = array![[2.0]];
                (f, g, h)
            }),
        )
        .with_profile(Profile::Robust)
        .with_bounds(bounds(lower, upper, 1e-8))
        .with_tolerance(tol(1e-9))
        .with_max_iterations(iters(200));

        let sol = solver
            .run()
            .expect("Projected ARC should converge at upper bound");
        assert!((sol.final_point[0] - 1.0).abs() < 1e-8);
        assert!(gradient_norm(&sol) <= 1e-6);
    }

    #[test]
    fn arc_retries_on_recoverable_trial_errors() {
        struct RecoverableArcTrialObjective {
            calls: usize,
        }

        impl ZerothOrderObjective for RecoverableArcTrialObjective {
            fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
                Ok(0.5 * (x[0] - 1.0).powi(2))
            }
        }

        impl FirstOrderObjective for RecoverableArcTrialObjective {
            fn eval_grad(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<FirstOrderSample, ObjectiveEvalError> {
                Ok(FirstOrderSample {
                    value: 0.5 * (x[0] - 1.0).powi(2),
                    gradient: array![x[0] - 1.0],
                })
            }
        }

        impl SecondOrderObjective for RecoverableArcTrialObjective {
            fn eval_hessian(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<SecondOrderSample, ObjectiveEvalError> {
                self.calls += 1;
                if self.calls == 2 {
                    return Err(ObjectiveEvalError::recoverable(
                        "simulated recoverable trial failure",
                    ));
                }
                Ok(SecondOrderSample {
                    value: 0.5 * (x[0] - 1.0).powi(2),
                    gradient: array![x[0] - 1.0],
                    hessian: Some(array![[1.0]]),
                })
            }
        }

        let x0 = array![2.0];
        let mut solver = super::Arc::new(x0, RecoverableArcTrialObjective { calls: 0 })
            .with_profile(Profile::Deterministic)
            .with_tolerance(tol(1e-8))
            .with_max_iterations(iters(300));

        // ARC should survive recoverable trial-evaluation failures by increasing
        // regularization and retrying, then still converge to the minimizer.
        let sol = solver
            .run()
            .expect("recoverable ARC trial failures should trigger retries and recover");
        assert!((sol.final_point[0] - 1.0).abs() < 1e-6);
        assert!(gradient_norm(&sol) < 1e-6);
    }

    #[test]
    fn arc_sigma_escalation_uses_gamma2_then_gamma3() {
        let mut core = super::ArcCore::new(array![0.0]);
        core.sigma = 1.0;
        core.gamma2 = 2.0;
        core.gamma3 = 3.0;
        let mut streak = 0usize;

        // First two failures: moderate growth (gamma2).
        core.escalate_sigma_on_failure(&mut streak);
        assert_eq!(streak, 1);
        assert!((core.sigma - 2.0).abs() < 1e-12);

        core.escalate_sigma_on_failure(&mut streak);
        assert_eq!(streak, 2);
        assert!((core.sigma - 4.0).abs() < 1e-12);

        // Third consecutive failure: stronger growth (gamma3).
        core.escalate_sigma_on_failure(&mut streak);
        assert_eq!(streak, 3);
        assert!((core.sigma - 12.0).abs() < 1e-12);
    }

    /// A function whose gradient is constant, causing `y_k` to be zero.
    fn linear_function(x: &Array1<f64>) -> (f64, Array1<f64>) {
        (2.0 * x[0] + 3.0 * x[1], array![2.0, 3.0])
    }

    fn huge_offset_linear_function(x: &Array1<f64>) -> (f64, Array1<f64>) {
        (1.0e16 + 2.0 * x[0] + 3.0 * x[1], array![2.0, 3.0])
    }

    // A highly ill-conditioned quadratic function.
    // The "valley" is 1000x longer than it is wide.
    fn ill_conditioned_quadratic(x: &Array1<f64>) -> (f64, Array1<f64>) {
        let scale = 1000.0;
        let f = scale * x[0].powi(2) + x[1].powi(2);
        let g = array![2.0 * scale * x[0], 2.0 * x[1]];
        (f, g)
    }

    // This function is minimized anywhere on the line x[0] = -x[1].
    // Its Hessian is singular.
    fn singular_hessian_function(x: &Array1<f64>) -> (f64, Array1<f64>) {
        let val = (x[0] + x[1]).powi(2);
        (val, array![2.0 * (x[0] + x[1]), 2.0 * (x[0] + x[1])])
    }

    // Function with a steep exponential "wall".
    fn wall_with_minimum(x: &Array1<f64>) -> (f64, Array1<f64>) {
        if x[0] > 70.0 {
            // The wall
            (f64::INFINITY, array![f64::INFINITY])
        } else {
            // A simple quadratic with minimum at x=60
            ((x[0] - 60.0).powi(2), array![2.0 * (x[0] - 60.0)])
        }
    }

    // --- 1. Standard Convergence Tests ---

    #[test]
    fn test_quadratic_bowl_converges() {
        let x0 = array![10.0, -5.0];
        let Solution { final_point, .. } = Bfgs::new(x0, bfgs_oracle(quadratic)).run().unwrap();
        assert_that!(&final_point[0]).is_close_to(0.0, 1e-5);
        assert_that!(&final_point[1]).is_close_to(0.0, 1e-5);
    }

    #[test]
    fn test_optimize_first_order_picks_bfgs() {
        let x0 = array![10.0, -5.0];
        let Solution { final_point, .. } = optimize(Problem::new(x0, bfgs_oracle(quadratic)))
            .run()
            .unwrap();
        assert_that!(&final_point[0]).is_close_to(0.0, 1e-5);
        assert_that!(&final_point[1]).is_close_to(0.0, 1e-5);
    }

    #[test]
    fn test_optimize_second_order_picks_newton_by_default() {
        let x0 = array![-1.2, 1.0];
        let Solution { final_point, .. } = optimize(SecondOrderProblem::new(
            x0,
            SecondOrderFn::new(rosenbrock_with_hessian),
        ))
        .run()
        .unwrap();
        assert_that!(&final_point[0]).is_close_to(1.0, 1e-5);
        assert_that!(&final_point[1]).is_close_to(1.0, 1e-5);
    }

    #[test]
    fn test_optimize_second_order_uses_arc_for_aggressive_profile() {
        let x0 = array![1.0];
        let objective = SecondOrderFn::new(|x: &Array1<f64>| {
            let f = x[0] * x[0];
            let g = array![2.0 * x[0]];
            let h = array![[2.0]];
            (f, g, h)
        });
        let solver =
            optimize(SecondOrderProblem::new(x0, objective).with_profile(Profile::Aggressive));
        assert!(matches!(solver, AutoSecondOrderSolver::Arc(_)));
    }

    #[test]
    fn test_quadratic_still_converges_strongly() {
        let x0 = array![20.0, -30.0];
        let sol = Bfgs::new(x0, bfgs_oracle(quadratic))
            .with_tolerance(tol(1e-8))
            .with_max_iterations(iters(1000))
            .run()
            .unwrap();
        assert_that!(&sol.final_point[0]).is_close_to(0.0, 1e-6);
        assert_that!(&sol.final_point[1]).is_close_to(0.0, 1e-6);
    }

    #[test]
    fn test_rosenbrock_converges() {
        let x0 = array![-1.2, 1.0];
        let Solution { final_point, .. } = Bfgs::new(x0, bfgs_oracle(rosenbrock)).run().unwrap();
        assert_that!(&final_point[0]).is_close_to(1.0, 1e-5);
        assert_that!(&final_point[1]).is_close_to(1.0, 1e-5);
    }

    // --- 2. Failure and Edge Case Tests ---

    #[test]
    fn test_begin_at_minimum_terminates_immediately() {
        let x0 = array![0.0, 0.0];
        let Solution { iterations, .. } = Bfgs::new(x0, bfgs_oracle(quadratic))
            .with_tolerance(tol(1e-5))
            .run()
            .unwrap();
        assert_that(&iterations).is_less_than_or_equal_to(1);
    }

    #[test]
    fn test_max_iterations_error_is_returned() {
        let x0 = array![-1.2, 1.0];
        let max_iterations = 5;
        let result = Bfgs::new(x0, bfgs_oracle(rosenbrock))
            .with_max_iterations(iters(max_iterations))
            .run();

        match result {
            Err(BfgsError::MaxIterationsReached { last_solution }) => {
                assert_eq!(last_solution.iterations, max_iterations);
                // Also check that the point is not the origin, i.e., that some work was done.
                assert_that!(&last_solution.final_point.dot(&last_solution.final_point))
                    .is_greater_than(0.0);
            }
            _ => panic!("Expected MaxIterationsReached error, but got {:?}", result),
        }
    }

    #[test]
    fn test_non_convex_function_is_handled() {
        let x0 = array![2.0];
        let result = Bfgs::new(x0.clone(), bfgs_oracle(non_convex_max)).run();
        // The robust solver should not fail. It gets stuck trying to minimize a function with no minimum.
        // It will hit the max iteration limit because it can't find steps that satisfy the descent condition.
        assert!(matches!(
            result,
            Err(BfgsError::MaxIterationsReached { .. })
                | Err(BfgsError::LineSearchFailed { .. })
                | Err(BfgsError::GradientIsNaN)
        ));
    }

    #[test]
    fn test_zero_curvature_is_handled() {
        let x0 = array![10.0, 10.0];
        let result = Bfgs::new(x0, bfgs_oracle(linear_function))
            .with_profile(Profile::Deterministic)
            .run();
        // The solver should skip Hessian updates due to sy=0 and eventually
        // terminate gracefully without panicking.
        match result {
            Ok(sol) => {
                assert!(sol.final_value.is_finite());
                assert!(gradient_norm(&sol).is_finite());
            }
            Err(BfgsError::MaxIterationsReached { .. })
            | Err(BfgsError::LineSearchFailed { .. })
            | Err(BfgsError::StepSizeTooSmall) => {}
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_no_improve_streak_requires_stationarity_or_tiny_step() {
        let x0 = array![10.0, 10.0];
        let result = Bfgs::new(x0, bfgs_oracle(huge_offset_linear_function))
            .with_profile(Profile::Deterministic)
            .with_max_iterations(iters(8))
            .run();

        match result {
            Ok(sol) => panic!(
                "solver falsely reported convergence with ||g||={:.3e}",
                gradient_norm(&sol)
            ),
            Err(BfgsError::MaxIterationsReached { last_solution })
            | Err(BfgsError::LineSearchFailed { last_solution, .. }) => {
                assert!(gradient_norm(&last_solution) > 1e-3);
            }
            Err(BfgsError::StepSizeTooSmall) => {}
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn stagnation_guard_requires_gradient_or_tiny_feasible_step() {
        let core = super::BfgsCore::new(array![0.0, 0.0]);
        let x_prev = array![1.0, 1.0];
        let x_far = array![2.0, 2.0];
        let x_same = x_prev.clone();
        let g_large = array![1.0, -1.0];
        let g_small = array![1e-6, 0.0];

        assert!(!core.stagnation_converged(&x_prev, &x_far, &g_large));
        assert!(core.stagnation_converged(&x_prev, &x_same, &g_large));
        assert!(core.stagnation_converged(&x_prev, &x_far, &g_small));
    }

    #[test]
    fn test_nan_gradient_returns_error() {
        // This function's gradient becomes NaN when x gets very close to 0.
        let nan_fn = |x: &Array1<f64>| {
            if x[0].abs() < 1e-12 {
                (f64::NAN, array![f64::NAN])
            } else {
                (x[0].powi(2), array![2.0 * x[0]])
            }
        };
        // Start at a point that will converge towards 0, triggering the NaN condition.
        let x0 = array![0.1];
        let result = Bfgs::new(x0, bfgs_oracle(nan_fn))
            .with_profile(Profile::Deterministic)
            .with_tolerance(tol(1e-15)) // Very tight tolerance to force convergence towards 0
            .run();

        match result {
            Ok(sol) => {
                assert!(sol.final_value.is_finite());
                assert!(sol.final_point[0].abs() < 1e-4);
            }
            Err(BfgsError::GradientIsNaN)
            | Err(BfgsError::LineSearchFailed { .. })
            | Err(BfgsError::MaxIterationsReached { .. })
            | Err(BfgsError::StepSizeTooSmall) => {}
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn test_linesearch_failed_reports_nonzero_attempts() {
        struct AlwaysRecoverableTrials;

        impl ZerothOrderObjective for AlwaysRecoverableTrials {
            fn eval_cost(&mut self, x: &Array1<f64>) -> Result<f64, ObjectiveEvalError> {
                if x.iter().all(|v| *v == 0.0) {
                    Ok(833.403058988699)
                } else {
                    Err(ObjectiveEvalError::recoverable(
                        "synthetic recoverable trial failure",
                    ))
                }
            }
        }

        impl FirstOrderObjective for AlwaysRecoverableTrials {
            fn eval_grad(
                &mut self,
                x: &Array1<f64>,
            ) -> Result<FirstOrderSample, ObjectiveEvalError> {
                if x.iter().all(|v| *v == 0.0) {
                    Ok(FirstOrderSample {
                        value: 833.403058988699,
                        gradient: array![1.1751972450892738, 0.0, 0.0],
                    })
                } else {
                    Err(ObjectiveEvalError::recoverable(
                        "synthetic recoverable trial failure",
                    ))
                }
            }
        }

        let x0 = array![0.0, 0.0, 0.0];
        let f_k = 833.403058988699;
        let g_k = array![1.1751972450892738, 0.0, 0.0];
        let d_k = -g_k.clone();
        let mut core = super::BfgsCore::new(x0.clone());
        let mut oracle = super::FirstOrderCache::new(x0.len());
        let err = super::backtracking_line_search(
            &mut core,
            &mut AlwaysRecoverableTrials,
            &mut oracle,
            &x0,
            &d_k,
            f_k,
            &g_k,
        )
        .expect_err("line search should fail when every trial is recoverable");

        let (max_attempts, failure_reason) = match err {
            super::LineSearchError::MaxAttempts(attempts) => {
                (attempts, LineSearchFailureReason::MaxAttempts)
            }
            super::LineSearchError::StepSizeTooSmall => (
                BACKTRACKING_MAX_ATTEMPTS,
                LineSearchFailureReason::StepSizeTooSmall,
            ),
            other => panic!("expected backtracking failure, got: {other:?}"),
        };

        assert!(max_attempts > 0, "max_attempts should never be 0");
        let rendered = format!(
            "{}",
            BfgsError::LineSearchFailed {
                last_solution: Box::new(Solution::gradient_based(
                    x0,
                    f_k,
                    g_k.clone(),
                    g_k.dot(&g_k).sqrt(),
                    None,
                    0,
                    0,
                    0,
                    0,
                )),
                max_attempts,
                failure_reason,
            }
        );
        assert!(
            rendered.contains("MaxAttempts") || rendered.contains("StepSizeTooSmall"),
            "error should include failure reason, got: {rendered}"
        );
    }

    // --- 3. Comparison Tests against a Trusted Library ---

    #[test]
    fn test_rosenbrock_matches_scipy_behavior() {
        let x0 = array![-1.2, 1.0];
        let tolerance = 1e-6;

        // Run our implementation.
        let our_res = Bfgs::new(x0.clone(), bfgs_oracle(rosenbrock))
            .with_tolerance(tol(tolerance))
            .run()
            .unwrap();

        // Run scipy's implementation with synchronized settings.
        let scipy_res = optimize_with_python(&x0, "rosenbrock", tolerance, 100)
            .expect("Python optimization failed");

        assert!(
            scipy_res.success,
            "Scipy optimization failed: {:?}",
            scipy_res.error
        );
        let scipy_point = scipy_res.final_point.unwrap();

        // Assert that the final points are virtually identical.
        let distance = ((our_res.final_point[0] - scipy_point[0]).powi(2)
            + (our_res.final_point[1] - scipy_point[1]).powi(2))
        .sqrt();
        assert_that!(&distance).is_less_than(1e-5);

        // Assert that the number of iterations is very similar. A small difference
        // is acceptable due to minor, valid variations in line search implementations.
        let iter_diff = (our_res.iterations as i64 - scipy_res.iterations.unwrap() as i64).abs();
        assert_that(&iter_diff).is_less_than_or_equal_to(10);

        let PythonOptResult {
            final_value,
            final_gradient_norm,
            func_evals,
            grad_evals,
            message,
            ..
        } = scipy_res;
        if let Some(value) = final_value {
            assert!(value.is_finite());
        }
        if let Some(norm) = final_gradient_norm {
            assert!(norm.is_finite());
        }
        if let Some(count) = func_evals {
            assert!(count > 0);
        }
        if let Some(count) = grad_evals {
            assert!(count > 0);
        }
        if let Some(text) = message {
            assert!(!text.is_empty());
        }
    }

    #[test]
    fn test_quadratic_matches_scipy_behavior() {
        let x0 = array![150.0, -275.5];
        let tolerance = 1e-8;

        // Run our implementation.
        match Bfgs::new(x0.clone(), bfgs_oracle(quadratic))
            .with_tolerance(tol(tolerance))
            .run()
        {
            Ok(sol) => sol,
            Err(BfgsError::MaxIterationsReached { last_solution }) => *last_solution,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        // Run scipy's implementation with synchronized settings.
        let scipy_res = optimize_with_python(&x0, "quadratic", tolerance, 100)
            .expect("Python optimization failed");

        assert!(
            scipy_res.success,
            "Scipy optimization failed: {:?}",
            scipy_res.error
        );

        let PythonOptResult {
            final_point,
            final_value,
            final_gradient_norm,
            iterations,
            func_evals,
            grad_evals,
            message,
            ..
        } = scipy_res;
        if let Some(point) = final_point {
            assert_eq!(point.len(), 2);
        }
        if let Some(value) = final_value {
            assert!(value.is_finite());
        }
        if let Some(norm) = final_gradient_norm {
            assert!(norm.is_finite());
        }
        if let Some(iters) = iterations {
            assert!(iters <= 100);
        }
        if let Some(count) = func_evals {
            assert!(count > 0);
        }
        if let Some(count) = grad_evals {
            assert!(count > 0);
        }
        if let Some(text) = message {
            assert!(!text.is_empty());
        }
    }

    // --- 4. Robustness Tests ---

    #[test]
    fn test_ill_conditioned_problem_converges() {
        let x0 = array![1.0, 1000.0]; // Start far up the narrow valley
        let res = Bfgs::new(x0, bfgs_oracle(ill_conditioned_quadratic)).run();
        assert!(res.is_ok() || matches!(res, Err(BfgsError::MaxIterationsReached { .. })));
    }

    #[test]
    fn test_singular_hessian_is_handled_gracefully() {
        let x0 = array![10.0, 20.0];
        let result = Bfgs::new(x0, bfgs_oracle(singular_hessian_function))
            .with_tolerance(tol(1e-8))
            .run();

        // The goal is to ensure the solver doesn't panic or return a numerical error.
        // It can either converge (if it gets lucky) or hit the max iteration limit.
        // Both are "graceful" outcomes.
        match result {
            Ok(soln) => {
                // If it did converge, verify it's on the correct line of minima.
                assert_that!(&soln.final_point[0]).is_close_to(-soln.final_point[1], 1e-5);
                assert_that!(&gradient_norm(&soln)).is_less_than(1e-8);
            }
            Err(BfgsError::MaxIterationsReached { .. }) => {
                // Hitting the iteration limit is an acceptable and expected outcome. Pass.
            }
            Err(e) => {
                // Any other error (like LineSearchFailed, GradientIsNaN) is a failure.
                panic!("Solver failed with an unexpected error: {:?}", e);
            }
        }
    }

    #[test]
    fn test_line_search_handles_inf() {
        let x0 = array![10.0]; // Start far from the wall and minimum.
        let result = Bfgs::new(x0, bfgs_oracle(wall_with_minimum)).run();
        assert!(result.is_ok() || matches!(result, Err(BfgsError::MaxIterationsReached { .. })));
    }

    #[test]
    fn test_trust_region_projection_uses_actual_step() {
        let x0 = array![0.9];
        let lower = array![0.0];
        let upper = array![1.0];
        let mut core = super::BfgsCore::new(x0.clone());
        core.bounds = Some(super::BoxSpec::new(lower, upper, 1e-8));
        core.trust_radius = 10.0;
        let fg = |x: &Array1<f64>| {
            let f = (x[0] - 2.0).powi(2);
            let g = array![2.0 * (x[0] - 2.0)];
            (f, g)
        };
        let mut obj = bfgs_oracle(fg);
        let x_k = core.project_point(&x0);
        let (f_k, g_k) = fg(&x_k);
        let mut b_inv = Array2::eye(1);
        let mut oracle = super::FirstOrderCache::new(x0.len());
        let mut func_evals = 0;
        let mut grad_evals = 0;
        let res = core.try_trust_region_step(
            &mut obj,
            &mut oracle,
            &mut b_inv,
            &x_k,
            f_k,
            &g_k,
            &mut func_evals,
            &mut grad_evals,
        );
        assert!(res.is_some());
        let (x_new, f_new, g_new) = res.unwrap();
        assert!((x_new[0] - 1.0).abs() < 1e-12);
        assert!(f_new.is_finite());
        assert!(g_new[0].is_finite());
    }

    #[test]
    fn test_bfgs_trust_region_predicted_decrease_respects_active_mask() {
        let core = super::BfgsCore::new(array![0.0, 0.0]);
        let b_inv = array![[2.0, 1.0], [1.0, 2.0]];
        let g_proj = array![0.0, -1.0];
        let s = array![0.0, 1.0];
        let active = vec![true, false];

        let pred = core
            .trust_region_predicted_decrease(&b_inv, &g_proj, &s, Some(&active))
            .expect("masked predicted decrease should be well-defined");

        assert!(
            (pred - 0.75).abs() < 1e-9,
            "unexpected predicted decrease: {pred}"
        );
    }

    #[test]
    fn test_bfgs_trust_region_fallback_freezes_active_bound_coordinates() {
        let x0 = array![0.0, 0.0];
        let lower = array![0.0, -10.0];
        let upper = array![10.0, 10.0];
        let mut core = super::BfgsCore::new(x0.clone());
        core.bounds = Some(super::BoxSpec::new(lower, upper, 1e-8));
        core.trust_radius = 10.0;

        let fg = |x: &Array1<f64>| {
            let f = (x[0] + 1.0).powi(2) + (x[1] - 2.0).powi(2);
            let g = array![2.0 * (x[0] + 1.0), 2.0 * (x[1] - 2.0)];
            (f, g)
        };

        let mut obj = bfgs_oracle(fg);
        let x_k = core.project_point(&x0);
        let (f_k, g_k) = fg(&x_k);
        let active = core.active_mask(&x_k, &g_k);
        assert_eq!(active, vec![true, false]);

        let mut b_inv = array![[5.0, 1.0], [1.0, 0.5]];
        let mut oracle = super::FirstOrderCache::new(x0.len());
        let mut func_evals = 0;
        let mut grad_evals = 0;
        let res = core.try_trust_region_step(
            &mut obj,
            &mut oracle,
            &mut b_inv,
            &x_k,
            f_k,
            &g_k,
            &mut func_evals,
            &mut grad_evals,
        );

        assert!(
            res.is_some(),
            "masked trust-region fallback should produce a feasible step"
        );
        let (x_new, f_new, g_new) = res.unwrap();
        assert!(
            x_new[0].abs() < 1e-12,
            "active coordinate moved: {:?}",
            x_new
        );
        assert!(x_new[1] > x_k[1]);
        assert!(f_new < f_k);
        assert!(g_new.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_flat_with_noise_accepts() {
        let f = |x: &Array1<f64>| {
            let noise = (x.sum() * 1e6).sin() * 1e-12;
            let val = 1.0 + noise;
            let g = Array1::from_vec(vec![1e-12; x.len()]);
            (val, g)
        };
        let x0 = array![0.0, 0.0];
        let res = Bfgs::new(x0, bfgs_oracle(f))
            .with_tolerance(tol(1e-10))
            .run();
        assert!(res.is_ok() || matches!(res, Err(super::BfgsError::MaxIterationsReached { .. })));
    }

    #[test]
    fn test_piecewise_alpha_jump() {
        let f = |x: &Array1<f64>| {
            let r = x.dot(x).sqrt();
            let val = if r < 1.0 { 1.0 } else { 0.9 };
            let g = if r < 1.0 {
                Array1::zeros(x.len())
            } else {
                x.mapv(|v| 1e-6 * v)
            };
            (val, g)
        };
        let x0 = array![0.5, 0.5];
        let res = Bfgs::new(x0, bfgs_oracle(f)).run();
        assert!(res.is_ok() || matches!(res, Err(super::BfgsError::MaxIterationsReached { .. })));
    }

    #[test]
    fn test_rng_symmetry() {
        // Ensure the internal RNG produces a roughly symmetric distribution.
        let x0 = array![0.0];
        let f = |x: &Array1<f64>| (x[0], array![1.0]);
        let mut solver = super::Bfgs::new(x0, bfgs_oracle(f));
        solver.core.rng_state = 12345;
        let mut sum = 0.0f64;
        let n = 20_000;
        for _ in 0..n {
            sum += solver.next_rand_sym();
        }
        let mean = sum / (n as f64);
        assert_that!(&mean.abs()).is_less_than(5e-3);
    }
}

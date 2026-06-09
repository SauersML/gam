//! Sparse Identification of Nonlinear Dynamics (SINDy) — Sequential Thresholded
//! Least Squares (STLSQ) solver, Brunton, Proctor, Kutz (PNAS 2016).
//!
//! Given a library design matrix `Θ ∈ ℝ^{n × p}` (each column is a candidate
//! nonlinearity evaluated at the `n` trajectory samples) and a target derivative
//! matrix `Ẋ ∈ ℝ^{n × d}`, STLSQ alternates:
//!   1. Solve `Ξ_active ← argmin_Ξ ‖Θ_active · Ξ − Ẋ‖² + λ ‖Ξ‖²` on the current
//!      active support (per output column independently — each column of `Ξ`
//!      is a SINDy coefficient vector for one state-variable derivative).
//!   2. Hard-threshold: drop entries with `|ξ_ij| < tol` from the support.
//! Repeat until the support stops changing or `max_rounds` is reached.
//!
//! The SCAD / MCP path (Fan-Li 2001 / Zhang 2010) uses the local quadratic
//! approximation (LQA): each round re-weights the ridge regularizer
//! coordinate-wise by `p'(|ξ_ij|) / max(|ξ_ij|, ε)`, where `p'` is the SCAD or
//! MCP derivative. This is the standard iterative re-weighted ridge surrogate
//! and reduces to plain ridge when `lam == 0`. The result remains compatible
//! with the hard-threshold step.
//!
//! The solver is fully self-contained: no GAM design / penalty / REML
//! infrastructure is involved. This file is the canonical implementation; the
//! Python `gamfit.SINDyAtoms` class wraps it via `gam-pyffi`.

use crate::faer_ndarray::FaerCholesky;
use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Concave-penalty family for the per-iteration re-weighting step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SindyPenaltyKind {
    /// Plain ridge (Tikhonov) on the active set.
    Ridge,
    /// SCAD (Fan & Li 2001). `a` defaults to 3.7.
    Scad,
    /// MCP (Zhang 2010). `a` (gamma) defaults to 3.0.
    Mcp,
}

/// SCAD derivative `p'_{λ,a}(|ξ|)`.
#[inline]
fn scad_grad(abs_xi: f64, lam: f64, a: f64) -> f64 {
    if abs_xi <= lam {
        lam
    } else if abs_xi <= a * lam {
        ((a * lam - abs_xi) / (a - 1.0)).max(0.0)
    } else {
        0.0
    }
}

/// MCP derivative `p'_{λ,γ}(|ξ|)`.
#[inline]
fn mcp_grad(abs_xi: f64, lam: f64, gamma: f64) -> f64 {
    if abs_xi <= gamma * lam {
        (lam - abs_xi / gamma).max(0.0)
    } else {
        0.0
    }
}

/// Per-coordinate ridge weight produced by the LQA surrogate for the given
/// concave penalty family at the current coefficient magnitude `abs_xi`.
#[inline]
fn lqa_weight(kind: SindyPenaltyKind, abs_xi: f64, lam: f64, a: f64, eps: f64) -> f64 {
    match kind {
        SindyPenaltyKind::Ridge => lam,
        SindyPenaltyKind::Scad => scad_grad(abs_xi, lam, a) / abs_xi.max(eps),
        SindyPenaltyKind::Mcp => mcp_grad(abs_xi, lam, a) / abs_xi.max(eps),
    }
}

#[derive(Debug, Clone)]
pub struct SindyStlsqResult {
    /// Coefficient matrix `Ξ ∈ ℝ^{p × d}`. Each column is the SINDy vector for
    /// one state-variable derivative; entries below `tol` are exactly zero.
    pub coefficients: Array2<f64>,
    /// Number of STLSQ rounds actually executed (≥ 1, ≤ `max_rounds`).
    pub rounds_used: usize,
    /// `true` iff the active support stabilised before `max_rounds`.
    pub converged: bool,
}

/// Run STLSQ on `(theta, dz_dt)`.
///
/// * `theta`: `(n, p)` library design, each column a library term evaluated at
///   the `n` trajectory rows.
/// * `dz_dt`: `(n, d)` target derivatives, one column per state dimension.
/// * `tol`: hard threshold; entries with `|ξ| < tol` are dropped each round.
/// * `max_rounds`: STLSQ iteration cap (must be ≥ 1).
/// * `lam`: regularization strength. For ridge, this is the L2 weight; for
///   SCAD / MCP it is the concave-penalty parameter.
/// * `kind`: penalty family.
/// * `concave_a`: SCAD's `a` (default 3.7) or MCP's `γ` (default 3.0). Ignored
///   for `Ridge`.
pub fn sindy_stlsq_solve(
    theta: ArrayView2<'_, f64>,
    dz_dt: ArrayView2<'_, f64>,
    tol: f64,
    max_rounds: usize,
    lam: f64,
    kind: SindyPenaltyKind,
    concave_a: f64,
) -> Result<SindyStlsqResult, String> {
    let (n, p) = theta.dim();
    let (n_dz, d) = dz_dt.dim();
    if n == 0 || p == 0 || d == 0 {
        return Err(format!(
            "sindy_stlsq_solve requires non-empty theta and dz_dt; got theta=({n},{p}), dz_dt=({n_dz},{d})"
        ));
    }
    if n_dz != n {
        return Err(format!(
            "sindy_stlsq_solve requires theta.nrows == dz_dt.nrows; got {n} vs {n_dz}"
        ));
    }
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(format!(
            "sindy_stlsq_solve requires finite tol >= 0, got {tol}"
        ));
    }
    if max_rounds == 0 {
        return Err("sindy_stlsq_solve requires max_rounds >= 1".to_string());
    }
    if !(lam.is_finite() && lam >= 0.0) {
        return Err(format!(
            "sindy_stlsq_solve requires finite lam >= 0, got {lam}"
        ));
    }
    if matches!(kind, SindyPenaltyKind::Scad) && !(concave_a.is_finite() && concave_a > 2.0) {
        return Err(format!(
            "sindy_stlsq_solve SCAD requires concave_a > 2, got {concave_a}"
        ));
    }
    if matches!(kind, SindyPenaltyKind::Mcp) && !(concave_a.is_finite() && concave_a > 1.0) {
        return Err(format!(
            "sindy_stlsq_solve MCP requires concave_a > 1, got {concave_a}"
        ));
    }

    // Initial seed: plain ridge solve on the full library.
    let lam_seed = if lam > 0.0 { lam } else { 1.0e-12 };
    let mut xi = ridge_full_solve(theta, dz_dt, lam_seed)?;
    let lqa_eps = (tol * 1.0e-2).max(1.0e-10);

    // Support is tracked as a `p × d` boolean mask: STLSQ is defined per output
    // column, so each output `c` carries its own active feature set `mask[.,c]`
    // and is refit and thresholded independently. (Tracking a single per-feature
    // boolean and refitting every column on the feature UNION returns matrices
    // that are not STLSQ fixed points — issue #958.)
    let mut mask = Array2::<bool>::from_elem((p, d), true);
    let mut prev_mask = Array2::<bool>::from_elem((p, d), false);
    let mut rounds_used = 0usize;
    let mut converged = false;

    for round in 0..max_rounds {
        rounds_used = round + 1;
        // Hard-threshold each coefficient independently: an entry `(j, c)` is
        // kept active iff it was active and `|xi[j,c]| >= tol`; otherwise it is
        // zeroed and dropped from this output column's support.
        for j in 0..p {
            for c in 0..d {
                if !(mask[(j, c)] && xi[(j, c)].abs() >= tol) {
                    mask[(j, c)] = false;
                    xi[(j, c)] = 0.0;
                }
            }
        }
        // Convergence compares the FULL per-column mask: STLSQ has reached a
        // fixed point only when no output column's support changed.
        if mask == prev_mask {
            converged = true;
            break;
        }
        prev_mask.assign(&mask);
        if mask.iter().all(|&on| !on) {
            // Everything thresholded out: the next round's mask is identical, so
            // continue to let the convergence check above fire.
            continue;
        }

        // Refit each output column on ITS OWN active feature set with
        // per-coordinate LQA-weighted ridge. Output columns that share the
        // exact same support are grouped so they can share the Gram assembly;
        // within a group, ridge shares one factorization (constant diagonal),
        // while SCAD/MCP factor per column (the per-coordinate weight differs).
        let mut refit = Array2::<f64>::zeros((p, d));
        let mut handled = vec![false; d];
        for c0 in 0..d {
            if handled[c0] {
                continue;
            }
            let active_idx: Vec<usize> = (0..p).filter(|&j| mask[(j, c0)]).collect();
            if active_idx.is_empty() {
                handled[c0] = true;
                continue;
            }
            // Gather all output columns sharing exactly this support.
            let group: Vec<usize> = (c0..d)
                .filter(|&c| (0..p).all(|j| mask[(j, c)] == mask[(j, c0)]))
                .collect();

            let p_act = active_idx.len();
            let mut theta_act = Array2::<f64>::zeros((n, p_act));
            for (k, &j) in active_idx.iter().enumerate() {
                theta_act.column_mut(k).assign(&theta.column(j));
            }

            if matches!(kind, SindyPenaltyKind::Ridge) {
                // Constant ridge diagonal `lam` across the group → one
                // factorization solves every group column at once.
                let diag = Array1::<f64>::from_elem(p_act, lam);
                let mut rhs = Array2::<f64>::zeros((n, group.len()));
                for (gk, &c) in group.iter().enumerate() {
                    rhs.column_mut(gk).assign(&dz_dt.column(c));
                }
                let sol = ridge_diag_solve(theta_act.view(), rhs.view(), diag.view())?;
                for (gk, &c) in group.iter().enumerate() {
                    for (k, &j) in active_idx.iter().enumerate() {
                        refit[(j, c)] = sol[(k, gk)];
                    }
                    handled[c] = true;
                }
            } else {
                // SCAD/MCP: the LQA weight is per coefficient `(j, c)` —
                // `p'(|ξ_jc|)/max(|ξ_jc|, ε)` — so each output column carries
                // its own diagonal and is solved separately (issue #959: a
                // single row-max weight is not a majorizer of the separable
                // penalty).
                for &c in &group {
                    let mut diag = Array1::<f64>::zeros(p_act);
                    for (k, &j) in active_idx.iter().enumerate() {
                        diag[k] = lqa_weight(kind, xi[(j, c)].abs(), lam, concave_a, lqa_eps);
                    }
                    let rhs = dz_dt.column(c).to_owned().insert_axis(Axis(1));
                    let sol = ridge_diag_solve(theta_act.view(), rhs.view(), diag.view())?;
                    for (k, &j) in active_idx.iter().enumerate() {
                        refit[(j, c)] = sol[(k, 0)];
                    }
                    handled[c] = true;
                }
            }
        }
        xi = refit;
    }

    // Final hard-threshold pass.
    for j in 0..p {
        for c in 0..d {
            if xi[(j, c)].abs() < tol {
                xi[(j, c)] = 0.0;
            }
        }
    }

    Ok(SindyStlsqResult {
        coefficients: xi,
        rounds_used,
        converged,
    })
}

/// Plain ridge: `Ξ = (ΘᵀΘ + λ I)⁻¹ Θᵀ Ẋ`.
fn ridge_full_solve(
    theta: ArrayView2<'_, f64>,
    dz_dt: ArrayView2<'_, f64>,
    lam: f64,
) -> Result<Array2<f64>, String> {
    let p = theta.ncols();
    let diag = Array1::<f64>::from_elem(p, lam.max(1.0e-12));
    ridge_diag_solve(theta, dz_dt, diag.view())
}

/// Ridge with per-coordinate diagonal: `Ξ = (ΘᵀΘ + diag(d))⁻¹ Θᵀ Ẋ`.
fn ridge_diag_solve(
    theta: ArrayView2<'_, f64>,
    dz_dt: ArrayView2<'_, f64>,
    diag: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, String> {
    let p = theta.ncols();
    let mut gram = theta.t().dot(&theta);
    for i in 0..p {
        let d = diag[i].max(1.0e-12);
        gram[(i, i)] += d;
    }
    let rhs = theta.t().dot(&dz_dt);
    let chol = gram
        .cholesky(Side::Lower)
        .map_err(|err| format!("sindy_stlsq_solve ridge Cholesky failed: {err}"))?;
    let mut sol = rhs;
    chol.solve_mat_in_place(&mut sol);
    Ok(sol)
}

/// One entry of an ordered SINDy library specification.
///
/// Built-in families expand to one or more columns evaluated entirely here in
/// Rust (the Brunton 2016 monomial / trig basis). A [`SindyLibraryTerm::Custom`]
/// carries a single pre-evaluated column supplied by the caller — this is the
/// escape hatch for user Python callables, which cannot run in Rust. Column
/// order in the assembled design matrix follows the order of this slice, with
/// each built-in family expanding in its canonical per-state-dimension order.
#[derive(Debug, Clone)]
pub enum SindyLibraryTerm {
    /// `1` — a single all-ones column.
    Const,
    /// `z_i` for each state dimension `i` (`'id'` / `'linear'`).
    Identity,
    /// `z_i²` for each state dimension `i`.
    Square,
    /// `z_i³` for each state dimension `i`.
    Cube,
    /// `z_i z_j` for every unique pair `i < j`.
    Product,
    /// `sin(z_i)` for each state dimension `i`.
    Sin,
    /// `cos(z_i)` for each state dimension `i`.
    Cos,
    /// A single caller-supplied column with its display name. Used for Python
    /// callable library terms, which are evaluated on the Python side.
    Custom { name: String, column: Array1<f64> },
}

/// Build the SINDy library design matrix `Θ` and its column names from a
/// trajectory `z ∈ ℝ^{n × d}` and an ordered library specification.
///
/// This is the canonical implementation of the Brunton 2016 candidate-function
/// library (monomials up to cubic, pairwise products, sine / cosine). It
/// returns the design matrix together with one display name per column, in the
/// exact order the columns appear — so downstream pretty-printing of the learned
/// ODE system can label coefficients by reading the name list positionally.
///
/// * `z`: `(n, d)` trajectory, one row per sample, one column per state var.
/// * `state_names`: `d` display names for the state variables (e.g. `["x","y"]`).
/// * `terms`: ordered library spec; built-in families expand in canonical order
///   and [`SindyLibraryTerm::Custom`] columns are spliced in at their position.
pub fn sindy_library(
    z: ArrayView2<'_, f64>,
    state_names: &[String],
    terms: &[SindyLibraryTerm],
) -> Result<(Array2<f64>, Vec<String>), String> {
    let (n, d) = z.dim();
    if n == 0 || d == 0 {
        return Err(format!(
            "sindy_library requires a non-empty trajectory; got ({n}, {d})"
        ));
    }
    if state_names.len() != d {
        return Err(format!(
            "sindy_library requires one state name per column; got {} names for {d} state dims",
            state_names.len()
        ));
    }

    // Each closure produces one (name, column) pair; collected in spec order so
    // the assembled Θ column order matches the Python contract exactly.
    let mut columns: Vec<Array1<f64>> = Vec::new();
    let mut names: Vec<String> = Vec::new();
    let mut push = |name: String, col: Array1<f64>| {
        names.push(name);
        columns.push(col);
    };

    for term in terms {
        match term {
            SindyLibraryTerm::Const => {
                push("1".to_string(), Array1::<f64>::ones(n));
            }
            SindyLibraryTerm::Identity => {
                for i in 0..d {
                    push(state_names[i].clone(), z.column(i).to_owned());
                }
            }
            SindyLibraryTerm::Square => {
                for i in 0..d {
                    let zi = z.column(i);
                    push(format!("{}^2", state_names[i]), &zi * &zi);
                }
            }
            SindyLibraryTerm::Cube => {
                for i in 0..d {
                    let zi = z.column(i);
                    push(format!("{}^3", state_names[i]), &(&zi * &zi) * &zi);
                }
            }
            SindyLibraryTerm::Product => {
                for i in 0..d {
                    for j in (i + 1)..d {
                        let col = &z.column(i) * &z.column(j);
                        push(format!("{}{}", state_names[i], state_names[j]), col);
                    }
                }
            }
            SindyLibraryTerm::Sin => {
                for i in 0..d {
                    push(
                        format!("sin({})", state_names[i]),
                        z.column(i).mapv(f64::sin),
                    );
                }
            }
            SindyLibraryTerm::Cos => {
                for i in 0..d {
                    push(
                        format!("cos({})", state_names[i]),
                        z.column(i).mapv(f64::cos),
                    );
                }
            }
            SindyLibraryTerm::Custom { name, column } => {
                if column.len() != n {
                    return Err(format!(
                        "sindy_library custom term {name:?} produced {} rows, expected {n}",
                        column.len()
                    ));
                }
                push(name.clone(), column.clone());
            }
        }
    }

    if columns.is_empty() {
        return Err("sindy_library: spec expanded to zero terms".to_string());
    }

    let p = columns.len();
    let mut theta = Array2::<f64>::zeros((n, p));
    for (k, col) in columns.iter().enumerate() {
        theta.column_mut(k).assign(col);
    }
    Ok((theta, names))
}

/// Render one human-readable ODE per state variable from a fitted SINDy
/// coefficient matrix.
///
/// `theta` is the public-API coefficient layout `(state_dim, n_terms)`: row `i`
/// is the coefficient vector for `d{state_i}/dt`. Coefficients are formatted to
/// three significant figures via [`crate::solver::evidence::format_three_significant`]
/// (the single source of truth for SINDy number formatting), exactly-zero
/// coefficients are dropped, the first surviving term inlines its sign
/// (`-10.0x`), and subsequent terms use a spaced separator (` - 3.00y`). The
/// constant column (term name `"1"`) renders as a bare magnitude.
pub fn sindy_render_equations(
    theta: ArrayView2<'_, f64>,
    term_names: &[String],
    state_names: &[String],
) -> Result<Vec<String>, String> {
    use crate::solver::evidence::format_three_significant;
    let (d, p) = theta.dim();
    if state_names.len() != d {
        return Err(format!(
            "sindy_render_equations requires one state name per row; got {} names for {d} rows",
            state_names.len()
        ));
    }
    if term_names.len() != p {
        return Err(format!(
            "sindy_render_equations requires one term name per column; got {} names for {p} columns",
            term_names.len()
        ));
    }
    let mut lines = Vec::with_capacity(d);
    for i in 0..d {
        let mut rhs = String::new();
        for (coef, term_name) in theta.row(i).iter().zip(term_names.iter()) {
            let coef = *coef;
            if coef == 0.0 {
                continue;
            }
            let magnitude = format_three_significant(coef.abs());
            let rendered = if term_name == "1" {
                magnitude
            } else {
                format!("{magnitude}{term_name}")
            };
            if rhs.is_empty() {
                if coef < 0.0 {
                    rhs.push('-');
                }
                rhs.push_str(&rendered);
            } else {
                let sign = if coef < 0.0 { '-' } else { '+' };
                rhs.push_str(&format!(" {sign} {rendered}"));
            }
        }
        if rhs.is_empty() {
            rhs.push('0');
        }
        lines.push(format!("d{}/dt = {rhs}", state_names[i]));
    }
    Ok(lines)
}

/// Estimate `dz/dt` from a trajectory by finite differences with spacing `dt`.
///
/// Centered second-order differences on the interior, one-sided first-order
/// differences at the two endpoints — the standard SINDy differentiation step
/// (PySINDy's `FiniteDifference`). All `n` rows are kept.
pub fn sindy_finite_difference(z: ArrayView2<'_, f64>, dt: f64) -> Result<Array2<f64>, String> {
    let (n, d) = z.dim();
    if n < 2 || d == 0 {
        return Err(format!(
            "sindy_finite_difference requires at least 2 rows and 1 column; got ({n}, {d})"
        ));
    }
    if !(dt.is_finite() && dt > 0.0) {
        return Err(format!(
            "sindy_finite_difference requires a positive finite dt, got {dt}"
        ));
    }
    let mut dz = Array2::<f64>::zeros((n, d));
    let inv_2dt = 1.0 / (2.0 * dt);
    let inv_dt = 1.0 / dt;
    for c in 0..d {
        // Centered interior.
        for r in 1..(n - 1) {
            dz[(r, c)] = (z[(r + 1, c)] - z[(r - 1, c)]) * inv_2dt;
        }
        // One-sided endpoints.
        dz[(0, c)] = (z[(1, c)] - z[(0, c)]) * inv_dt;
        dz[(n - 1, c)] = (z[(n - 1, c)] - z[(n - 2, c)]) * inv_dt;
    }
    Ok(dz)
}

/// Automatic `λ` selection over a logarithmic grid using BIC, with STLSQ
/// applied at each grid point. Returns `(best_lam, best_result)`.
///
/// SINDy is not a GAM design block (it produces a single sparse coefficient
/// matrix with no `S_θ`-tier structure), so the REML / LAML `'auto'` machinery
/// that drives e.g. `AdaptiveTopK` does not apply. The principled standalone
/// alternative is information-criterion selection on the residual variance
/// and active-support cardinality. We use BIC (Schwarz 1978) on the per-output
/// residual sums of squares — the SINDy literature's default complexity rule.
pub fn sindy_stlsq_auto_lam(
    theta: ArrayView2<'_, f64>,
    dz_dt: ArrayView2<'_, f64>,
    tol: f64,
    max_rounds: usize,
    kind: SindyPenaltyKind,
    concave_a: f64,
) -> Result<(f64, SindyStlsqResult), String> {
    let (n, _p) = theta.dim();
    if n == 0 {
        return Err("sindy_stlsq_auto_lam requires n > 0".to_string());
    }
    // Geometric grid spanning four decades around tol; matches the standard
    // SINDy hyperparameter sweep in the Brunton 2016 supplement.
    let grid: Vec<f64> = (0..9)
        .map(|i| tol.max(1.0e-6) * 10f64.powf((i as f64) - 4.0))
        .collect();
    let mut best: Option<(f64, f64, SindyStlsqResult)> = None;
    for &lam in &grid {
        let res = sindy_stlsq_solve(theta, dz_dt, tol, max_rounds, lam, kind, concave_a)?;
        let bic = bic_score(theta, dz_dt, &res.coefficients);
        let pick = match &best {
            None => true,
            Some((_, b, _)) => bic < *b,
        };
        if pick {
            best = Some((lam, bic, res));
        }
    }
    let (lam, _bic, res) =
        best.ok_or_else(|| "sindy_stlsq_auto_lam: empty grid produced no candidates".to_string())?;
    Ok((lam, res))
}

fn bic_score(theta: ArrayView2<'_, f64>, dz_dt: ArrayView2<'_, f64>, xi: &Array2<f64>) -> f64 {
    let (n, _p) = theta.dim();
    let d = dz_dt.ncols();
    let resid = &theta.dot(xi) - &dz_dt;
    let mut bic = 0.0_f64;
    let n_f = n as f64;
    for c in 0..d {
        let r = resid.index_axis(Axis(1), c);
        let rss = r.iter().map(|&x| x * x).sum::<f64>().max(1.0e-300);
        let k_active = xi.column(c).iter().filter(|&&v| v != 0.0).count() as f64;
        // Gaussian profile-likelihood BIC: n*log(rss/n) + k*log(n).
        bic += n_f * (rss / n_f).ln() + k_active * n_f.ln();
    }
    bic
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn stlsq_recovers_pure_linear_system() {
        // Library: [1, x, y], target: dx/dt = 2x - 3y, dy/dt = -x.
        // Expected Ξ ≈ [[0, 0], [2, -1], [-3, 0]].
        let n = 200;
        let mut rng_state = 0u64;
        let mut rand = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5
        };
        let mut theta = Array2::<f64>::zeros((n, 3));
        let mut dz = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let x = rand();
            let y = rand();
            theta[(i, 0)] = 1.0;
            theta[(i, 1)] = x;
            theta[(i, 2)] = y;
            dz[(i, 0)] = 2.0 * x - 3.0 * y;
            dz[(i, 1)] = -x;
        }
        let res = sindy_stlsq_solve(
            theta.view(),
            dz.view(),
            0.05,
            10,
            1.0e-3,
            SindyPenaltyKind::Ridge,
            3.7,
        )
        .expect("stlsq must succeed");
        assert!(res.converged);
        assert!((res.coefficients[(1, 0)] - 2.0).abs() < 0.05);
        assert!((res.coefficients[(2, 0)] + 3.0).abs() < 0.05);
        assert!((res.coefficients[(1, 1)] + 1.0).abs() < 0.05);
        assert!(res.coefficients[(0, 0)].abs() < 1.0e-6);
        assert!(res.coefficients[(2, 1)].abs() < 1.0e-6);
    }

    /// Regression for #958: each output column must be refit on its OWN active
    /// set, not on the feature UNION across outputs. With two correlated
    /// features θ0, θ1 and outputs y1 = 2·θ1, y2 = θ0 + 0.4·θ1 at tol = 0.5:
    ///   * y1 keeps θ1 (coef 2 ≥ tol) and drops θ0;
    ///   * y2's θ1 coefficient (0.4) is below tol, so per-output STLSQ drops θ1
    ///     and refits y2 on support {θ0} ⇒ coef = θ0ᵀy2 / θ0ᵀθ0 ≠ 1.
    /// The old union-support code would keep θ1 for y2 (because y1 needs it),
    /// refit y2 to [1, 0.4], then entrywise-threshold 0.4→0 and return [1, 0] —
    /// which is NOT the STLSQ fixed point on {θ0}.
    #[test]
    fn stlsq_refits_each_output_on_its_own_support_958() {
        // Two correlated columns with θ0ᵀθ1 = r·(unit norms). Deterministic.
        let theta0 = [1.0_f64, 0.0, 0.0, 0.0];
        // θ1 chosen so cos-angle r ≈ 0.89 with θ0 and a clean second component.
        let r = 0.89_f64;
        let s = (1.0_f64 - r * r).sqrt();
        let theta1 = [r, s, 0.0, 0.0];

        let n = 4;
        let mut theta = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            theta[(i, 0)] = theta0[i];
            theta[(i, 1)] = theta1[i];
        }
        // Outputs: y1 = 2·θ1 (needs θ1), y2 = θ0 + 0.4·θ1 (θ1 below tol).
        let mut dz = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            dz[(i, 0)] = 2.0 * theta1[i];
            dz[(i, 1)] = theta0[i] + 0.4 * theta1[i];
        }

        // Tiny ridge so the hard threshold (not L2) governs the support.
        let res = sindy_stlsq_solve(
            theta.view(),
            dz.view(),
            0.5,
            20,
            1.0e-9,
            SindyPenaltyKind::Ridge,
            3.7,
        )
        .expect("stlsq must succeed");
        assert!(res.converged);

        // y2 (column 1) support is exactly {θ0}: θ1 dropped.
        assert!(
            res.coefficients[(1, 1)].abs() < 1.0e-12,
            "y2 must drop θ1 (its coef 0.4 < tol 0.5); got {}",
            res.coefficients[(1, 1)]
        );

        // The surviving θ0 coefficient is the per-output ridge refit on {θ0}:
        // θ0ᵀy2 / (θ0ᵀθ0 + ridge). With near-zero ridge this is θ0ᵀy2/θ0ᵀθ0.
        let g0: f64 = (0..n).map(|i| theta0[i] * theta0[i]).sum::<f64>() + 1.0e-9;
        let xy: f64 = (0..n).map(|i| theta0[i] * dz[(i, 1)]).sum();
        let expected = xy / g0; // = 1 + 0.4·r ≈ 1.356, NOT 1.
        assert!(
            (res.coefficients[(0, 1)] - expected).abs() < 1.0e-6,
            "y2 θ0 coef must be the per-output refit {expected} (≈1.356), not the \
             union-support value 1; got {}",
            res.coefficients[(0, 1)]
        );
        // Guard: the union-support bug returned exactly 1.0 here.
        assert!(
            (res.coefficients[(0, 1)] - 1.0).abs() > 1.0e-3,
            "y2 θ0 coef collapsed to the union-support value 1.0 (#958 regression)"
        );

        // y1 (column 0) keeps θ1 and drops θ0.
        assert!(
            res.coefficients[(0, 0)].abs() < 1.0e-12,
            "y1 must drop θ0; got {}",
            res.coefficients[(0, 0)]
        );
        assert!(
            (res.coefficients[(1, 0)] - 2.0).abs() < 1.0e-6,
            "y1 θ1 coef must be 2; got {}",
            res.coefficients[(1, 0)]
        );
    }

    /// Regression for #959: SCAD/MCP LQA weights must be per coefficient
    /// `(j, c)`, not one row-max weight applied to a whole feature row. With one
    /// feature carrying a large coefficient (10) in one output and a tiny one
    /// (0.01) in another, the row-max bug assigns the row weight from |10| (SCAD
    /// p'(10)=0 for a·λ<10 ⇒ weight 0), so the tiny 0.01 — which needs a large
    /// weight ≈ λ/0.01 = 100 — would be left UNPENALIZED. Per-coordinate weights
    /// must shrink the small coefficient toward zero.
    #[test]
    fn stlsq_scad_uses_per_coordinate_lqa_weights_959() {
        // One feature, two outputs, orthonormal design so the unpenalized refit
        // of each output reproduces its target coefficient exactly and the only
        // shrinkage is the per-coordinate SCAD weight.
        //   output 0 target coefficient: 10   (large; SCAD weight 0 — un-shrunk)
        //   output 1 target coefficient: 0.01 (small; SCAD weight ≈ λ/0.01)
        let n = 16;
        let mut theta = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            theta[(i, 0)] = 1.0; // θᵀθ = n
        }
        // Targets: constant columns so the LS solution of θ·ξ = y is ξ = mean(y).
        let mut dz = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            dz[(i, 0)] = 10.0;
            dz[(i, 1)] = 0.01;
        }

        let lam = 1.0_f64;
        let a = 3.7_f64;
        // tol below 0.01 so neither coefficient is hard-thresholded out: the
        // test isolates the LQA weighting, not the threshold.
        let res = sindy_stlsq_solve(
            theta.view(),
            dz.view(),
            1.0e-4,
            20,
            lam,
            SindyPenaltyKind::Scad,
            a,
        )
        .expect("stlsq must succeed");

        // Large coefficient: SCAD p'(10)=0 (since 10 > a·λ = 3.7) ⇒ weight 0 ⇒
        // unpenalized ⇒ recovered ≈ 10.
        assert!(
            (res.coefficients[(0, 0)] - 10.0).abs() < 1.0e-6,
            "large coef must be left essentially un-shrunk by SCAD; got {}",
            res.coefficients[(0, 0)]
        );

        // Small coefficient: it gets its OWN large per-coordinate SCAD weight.
        // The LQA weight is anchored at the previous iterate; STLSQ converges in
        // one refit here (the support never changes), so the value reflects the
        // SEED magnitude. Seed (full-library ridge, lam): ξ₀ = n·0.01/(n+lam).
        // Its SCAD weight w = p'(|ξ₀|)/max(|ξ₀|,ε) = λ/|ξ₀| (|ξ₀| ≤ λ ⇒ p'=λ),
        // then the refit is θᵀy/(θᵀθ + w) = n·0.01/(n + w) — heavily shrunk, NOT
        // the un-penalized 0.01 the row-max bug (weight 0) would leave.
        let nf = n as f64;
        let eps = (1.0e-4_f64 * 1.0e-2).max(1.0e-10);
        let seed_small = nf * 0.01 / (nf + lam.max(1.0e-12));
        let w_small = scad_grad(seed_small, lam, a) / seed_small.max(eps);
        let expected_small = (nf * 0.01) / (nf + w_small);
        // coefficients are laid out [n_features, n_outputs]; the small coef is
        // the single feature of OUTPUT 1, i.e. index (0, 1) — not (1, 0).
        assert!(
            (res.coefficients[(0, 1)] - expected_small).abs() < 1.0e-6,
            "small coef must get its own large SCAD weight ({w_small}) and be \
             shrunk to {expected_small}, not the row-max weight 0; got {}",
            res.coefficients[(0, 1)]
        );
        // Guard: the row-max bug would leave the small coef at ~0.01 (weight 0).
        assert!(
            res.coefficients[(0, 1)] < 0.01 - 1.0e-4,
            "small coef was not shrunk — row-max LQA weight bug (#959) present; got {}",
            res.coefficients[(0, 1)]
        );
    }
}

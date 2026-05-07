//! Shared marginal-slope intercept calibration solver.
//!
//! Bernoulli and survival marginal-slope paths solve the same one-dimensional
//! monotone intercept equation at fit time and at saved-model prediction time.
//! This module centralizes the seed ordering, tolerance-aligned convergence,
//! optional analytic bracketing, and warm-start fallback policy so the hot
//! per-row paths do not drift apart.

/// Per-row evaluator for marginal-slope intercept calibration.
pub trait MarginalSlopeCalibrationEval {
    /// Compute `(F(a), F'(a), F''(a))` for one candidate intercept.
    fn eval(&self, a: f64) -> Result<(f64, f64, f64), String>;

    /// Closed-form rigid seed for this row.
    fn rigid_seed(&self) -> f64;

    /// Optional affine-link upgrade of the seed when link deviation is active.
    fn affine_seed(&self) -> Option<f64> {
        None
    }

    /// Acceptance tolerance in probability space.
    fn abs_tol(&self) -> f64;

    /// Optional analytic bracket `[a_lo, a_hi]` guaranteed to contain the root.
    fn analytic_bracket(&self) -> Option<(f64, f64)> {
        None
    }

    /// True when flexible-deviation coefficients are effectively zero.
    fn deviation_is_negligible(&self) -> bool {
        false
    }
}

/// Solve a marginal-slope intercept equation using the shared seed and
/// monotone-root policy.
///
/// Returns `(root, |F'(root)|, F(root))`.
pub fn solve_intercept<E: MarginalSlopeCalibrationEval>(
    e: &E,
    warm_start: Option<f64>,
    label: &str,
) -> Result<(f64, f64, f64), String> {
    let abs_tol = e.abs_tol();
    if !abs_tol.is_finite() || abs_tol <= 0.0 {
        return Err(format!(
            "{label}: non-positive or non-finite tolerance {abs_tol:.3e}"
        ));
    }

    let rigid = e.rigid_seed();

    // Cycle-0 fast path: with negligible deviation coefficients the rigid
    // seed is the exact calibration root. We still evaluate once to return the
    // derivative required by fit-time Hessian/gradient code and to preserve the
    // residual contract.
    if e.deviation_is_negligible() {
        let (f, f_a, _) = e.eval(rigid)?;
        if f.abs() <= abs_tol {
            let abs_d = f_a.abs();
            if !abs_d.is_finite() || abs_d == 0.0 {
                return Err(format!(
                    "{label}: zero or non-finite derivative at rigid root a={rigid:.6}"
                ));
            }
            return Ok((rigid, abs_d, f));
        }
    }

    // Seed-residual short-circuit in priority order. This keeps prediction and
    // fit paths aligned and avoids entering bracketing/refinement when the
    // previous PIRLS solution, affine seed, or rigid seed is already accepted.
    let affine = e.affine_seed();
    let mut seeds = [None; 3];
    seeds[0] = warm_start.filter(|v| v.is_finite());
    seeds[1] = affine.filter(|v| v.is_finite());
    seeds[2] = rigid.is_finite().then_some(rigid);
    let mut last_seed: Option<f64> = None;
    for seed in seeds.into_iter().flatten() {
        if last_seed.is_some_and(|prev: f64| (prev - seed).abs() <= f64::EPSILON) {
            continue;
        }
        last_seed = Some(seed);
        let (f, f_a, _) = e.eval(seed)?;
        if f.abs() <= abs_tol {
            let abs_d = f_a.abs();
            if !abs_d.is_finite() || abs_d == 0.0 {
                return Err(format!(
                    "{label}: zero or non-finite derivative at accepted seed a={seed:.6}"
                ));
            }
            return Ok((seed, abs_d, f));
        }
    }

    let max_bracket_iters = bracket_iter_limit(abs_tol);
    let max_refine_iters = refine_iter_limit(abs_tol);
    let bracket = e.analytic_bracket().and_then(normalize_bracket);

    // Preserve the historical warm-start fallback: try the warm start first,
    // then retry once from the deterministic affine/rigid seed if bracketing
    // from the cached point fails.
    let primary_seed = seeds[0]
        .or(seeds[1])
        .or(seeds[2])
        .ok_or_else(|| format!("{label}: no finite seed available for intercept solve"))?;
    let deterministic_seed = seeds[1].or(seeds[2]).unwrap_or(primary_seed);

    let solve_from = |seed: f64| {
        super::monotone_root::solve_monotone_root_with_bracket(
            |a| e.eval(a),
            seed,
            label,
            abs_tol,
            max_bracket_iters,
            max_refine_iters,
            bracket,
        )
    };

    let mut result = solve_from(primary_seed);
    if warm_start.is_some() && result.is_err() && (deterministic_seed - primary_seed).abs() > 0.0 {
        result = solve_from(deterministic_seed);
    }
    result
}

fn normalize_bracket((a, b): (f64, f64)) -> Option<(f64, f64)> {
    if !a.is_finite() || !b.is_finite() || a == b {
        return None;
    }
    if a < b { Some((a, b)) } else { Some((b, a)) }
}

fn bracket_iter_limit(abs_tol: f64) -> usize {
    // Wide enough to match the old cap at ordinary tolerances, while scaling
    // upward for very tight probability-space solves.
    let digits = (-abs_tol.log10()).ceil().max(0.0) as usize;
    (8 * digits).clamp(32, 96)
}

fn refine_iter_limit(abs_tol: f64) -> usize {
    let digits = (-abs_tol.log10()).ceil().max(0.0) as usize;
    (6 * digits).clamp(32, 96)
}

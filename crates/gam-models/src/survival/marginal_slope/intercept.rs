//! Per-row survival intercept solve: the warm-started one-dimensional
//! Newton solve that pins the baseline intercept slot for a row.

use super::*;
use crate::monotone_root;

impl SurvivalMarginalSlopeFamily {
    pub(crate) fn solve_row_survival_intercept_with_slot(
        &self,
        q: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        slot: Option<(usize, SurvivalInterceptSlotKind)>,
    ) -> Result<(f64, f64), String> {
        let eval = |a: f64| -> Result<(f64, f64, f64), String> {
            self.evaluate_denested_survival_calibration(a, q, slope, beta_h, beta_w)
        };
        let probit_scale = self.probit_frailty_scale();
        let a_closed_form = q * rigid_observed_scale(slope, probit_scale) / probit_scale;

        // Prefer the previous PIRLS iter's converged intercept as the initial
        // guess; β changes only a little between consecutive PIRLS iterations,
        // so the previous answer is typically within a few root-solver steps
        // of the new one. If the slot is None (no cache wired) or the stored
        // bits decode to a non-finite value (uninitialised NaN sentinel /
        // stale), fall back to the closed-form rigid seed — preserving the
        // exact pre-warm-start behaviour.
        // Tag the cache entry with a 64-bit hash of (beta_h, beta_w) so that
        // rejected trust-region trials and subsampled probes cannot poison
        // the global per-row root: each trial keys under its own β, so a
        // write at β_A is invisible to a subsequent read at β_B. Consecutive
        // evaluations at the same β share the tag and reuse the warm start.
        let beta_tag = hash_intercept_warm_start_key(beta_h, beta_w);
        let cached_a = slot.and_then(|(row, kind)| {
            self.intercept_warm_starts
                .as_ref()
                .and_then(|cache| cache.load(row, kind, beta_tag))
        });
        let a_init = cached_a.unwrap_or(a_closed_form);
        let mut solve_result = monotone_root::solve_monotone_root_detailed(
            eval,
            a_init,
            "survival intercept",
            1e-12,
            64,
            64,
        );
        // If the warm-started solve failed, retry once from the closed-form
        // seed. Cached `a` from a prior PIRLS iter can be far enough from the
        // current root (e.g., after a large β step) that the bracketing search
        // exhausts; the closed-form seed always sits in the correct basin.
        if cached_a.is_some() && solve_result.is_err() {
            solve_result = monotone_root::solve_monotone_root_detailed(
                eval,
                a_closed_form,
                "survival intercept",
                1e-12,
                64,
                64,
            );
        }
        // This routine also emits its own format!()-based String errors below
        // (non-finite derivative, residual rejection), so the enclosing return
        // type stays Result<_, String>; convert the typed solver error here.
        let solution = solve_result.map_err(|e| e.to_string())?;
        let a = solution.root;
        // The solver already evaluated `eval` at `solution.root` during the
        // refine loop and returned the resulting `residual` (best_f) and
        // `abs_deriv` (best_abs_deriv). Reusing them here saves one full
        // calibration evaluation per row × 2 (entry + exit) per joint-Newton
        // sweep — at large-scale n=320k this is 640k spared evaluations per pass.
        let residual = solution.residual;
        let abs_deriv = solution.abs_deriv;
        if !abs_deriv.is_finite() || abs_deriv == 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope intercept solve failed: \
                     zero or non-finite derivative at a={a:.6}"
                ),
            }
            .into());
        }

        let target_survival = crate::probability::normal_cdf(-q);
        let achieved_survival = target_survival + residual;
        let tail_mass = target_survival.min(1.0 - target_survival).max(0.0);
        let probability_tol = SURVIVAL_INTERCEPT_ABS_RESIDUAL_TOL
            .max(SURVIVAL_INTERCEPT_REL_TAIL_RESIDUAL_TOL * tail_mass);
        let mut log_tail_residual = None;
        // Always accept if probability-space residual is within tolerance:
        // a perfectly-converged probability solve (residual=0) is the best
        // achievable answer, and rejecting it because the deep-tail log
        // computation has its own floating-point noise (~6e-8 at |q|>=7)
        // would discard a correct intercept. When tail_mass is small we
        // *additionally* accept tight log-space agreement, so well-resolved
        // tails that drift slightly outside the absolute probability_tol
        // (which can be ulp-bounded) still validate.
        let residual_ok = if tail_mass < SURVIVAL_INTERCEPT_LOG_TAIL_THRESHOLD {
            let probability_pass = residual.abs() <= probability_tol;
            let (achieved_tail, target_log_tail) = if target_survival <= 0.5 {
                let (target_log_survival, _) = signed_probit_logcdf_and_mills_ratio(-q);
                (achieved_survival, target_log_survival)
            } else {
                let (target_log_failure, _) = signed_probit_logcdf_and_mills_ratio(q);
                (1.0 - achieved_survival, target_log_failure)
            };
            let log_pass = if target_log_tail.is_finite()
                && achieved_tail.is_finite()
                && achieved_tail > 0.0
            {
                let log_residual = achieved_tail.ln() - target_log_tail;
                log_tail_residual = Some(log_residual);
                log_residual.abs() <= SURVIVAL_INTERCEPT_REL_TAIL_RESIDUAL_TOL
            } else {
                false
            };
            probability_pass || log_pass
        } else {
            residual.abs() <= probability_tol
        };

        if !residual_ok {
            let log_tail_detail = log_tail_residual
                .map(|value| format!(", log_tail_residual={value:.3e}"))
                .unwrap_or_default();
            return Err(SurvivalMarginalSlopeError::IntegrationFailed {
                reason: format!(
                    "survival marginal-slope intercept solve failed: \
                     residual={residual:.3e} at a={a:.6}, target survival={target_survival:.6e}, \
                     achieved survival={achieved_survival:.6e}, probability_tol={probability_tol:.3e}\
                     {log_tail_detail}"
                ),
            }
            .into());
        }

        // Cache the converged intercept for the next PIRLS iter, if a slot
        // was provided. When `slot` is None this is a no-op, preserving the
        // exact pre-warm-start behaviour. The stamp is the β-tagged key
        // computed above: only future reads at the same β observe this
        // write, so rejected or subsampled trials cannot leak their roots
        // into accepted full-data evaluations.
        if let Some((row, kind)) = slot
            && let Some(cache) = self.intercept_warm_starts.as_ref()
        {
            cache.store(row, kind, a, beta_tag);
        }

        Ok((a, abs_deriv))
    }
}

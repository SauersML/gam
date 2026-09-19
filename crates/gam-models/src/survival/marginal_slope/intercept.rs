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
        // gam#2948: a family anchored on a declared finite law solves the
        // identity on that law's nodes; on the Gaussian law the de-nested cells
        // integrate it.
        let law = self.flex_law_grid(slot.map(|(row, _)| row))?;
        // The identity `T(a) = Φ(∓q)` is solved on its SMALLER tail and in log
        // units (gam#2971): the residual is `log T(a) − log Φ(∓q)`, with `T` the
        // marginal survival `Σ Φ(−η)` when `q ≥ 0` and the marginal failure
        // `Σ Φ(η)` otherwise. The probability residual this replaces was held to
        // an absolute `1e-12`. Where the smaller tail is itself below that, every
        // `a` in a wide band met it, so the solve returned whatever seed it was
        // given, and the row likelihood followed the warm-start slot's history
        // rather than β. A relative residual pins the root however it is seeded.
        let survival_side = q >= 0.0;
        let log_target = if survival_side {
            crate::probability::normal_logcdf(-q)
        } else {
            crate::probability::normal_logcdf(q)
        };
        // The terms one evaluation of `T` sums: the law's nodes, or at most one
        // de-nested cell per breakpoint of either deviation plus the two tails.
        let terms = match law {
            Some(grid) => grid.len(),
            None => {
                self.score_warp
                    .as_ref()
                    .map_or(0, |runtime| runtime.breakpoints().len())
                    + self
                        .link_dev
                        .as_ref()
                        .map_or(0, |runtime| runtime.breakpoints().len())
                    + 1
            }
        };
        let rounding = crate::latent_anchor::anchor_residual_rounding(log_target, terms);
        let tolerance = crate::latent_anchor::ANCHOR_LOG_RESIDUAL_TOL.max(rounding);
        let eval = |a: f64| -> Result<(f64, f64, f64), String> {
            let (tail, tail_a, tail_aa) =
                self.calibration_smaller_tail(law, a, q, slope, beta_h, beta_w, survival_side)?;
            if !(tail.is_finite() && tail > 0.0) {
                return Err(SurvivalMarginalSlopeError::NumericalFailure {
                    reason: format!(
                        "survival marginal-slope intercept calibration tail T={tail:.3e} at \
                         a={a:.6} (q={q:.6}) has no finite logarithm"
                    ),
                }
                .into());
            }
            let ratio = tail_a / tail;
            Ok((tail.ln() - log_target, ratio, tail_aa / tail - ratio * ratio))
        };
        let probit_scale = self.probit_frailty_scale();
        // The rigid root, with no warp and no deviation, on the row's own law:
        // the closed form on the Gaussian law, the solved anchor on a finite one.
        let a_closed_form = match law {
            Some(grid) => solve_anchor(q, probit_scale * slope, grid)? / probit_scale,
            None => q * rigid_observed_scale(slope, probit_scale) / probit_scale,
        };

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
            tolerance,
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
                tolerance,
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

        // `residual` and `abs_deriv` are the log-tail residual and `|d log T/da|`
        // at the accepted root. A root is certified only where its residual sits
        // inside what the arithmetic resolves, the rule the rigid anchor solve
        // applies (gam#2928); anything else is refused by name.
        let resolution = crate::latent_anchor::anchor_residual_resolution(a, abs_deriv, rounding);
        if !(residual.abs() <= resolution) {
            return Err(SurvivalMarginalSlopeError::RootSolveFailed {
                reason: format!(
                    "survival marginal-slope intercept solve did not resolve its log-tail \
                     residual: residual={residual:.3e} against resolution={resolution:.3e} at \
                     a={a:.6} (q={q:.6}, log target={log_target:.6e}, |d log T/da|={abs_deriv:.3e})"
                ),
            }
            .into());
        }
        // Callers read the density normalisation `|T′(a)| = |d log T/da|·T(a)`,
        // with `T(a) = exp(residual + log target)` at the accepted root.
        let density = abs_deriv * (residual + log_target).exp();
        if !(density.is_finite() && density > 0.0) {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope intercept density normalisation |T′(a)|={density:.3e} \
                     at a={a:.6} (q={q:.6}) is not finite and positive"
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

        Ok((a, density))
    }

    /// `(T, T′, T″)` of the calibration's smaller tail at `a` (gam#2971): the
    /// marginal survival `Σ Φ(−η)` on the survival side, the marginal failure
    /// `Σ Φ(η)` otherwise. On the Gaussian law every de-nested cell contributes
    /// its own positive probability, so nothing subtracts probabilities near
    /// one. A finite law's residual is already summed on this tail
    /// (`F = T − Φ(−q)` on the survival side, `F = Φ(q) − T` otherwise), so `T`
    /// is read back from it.
    fn calibration_smaller_tail(
        &self,
        law: Option<AnchorGrid<'_>>,
        a: f64,
        q: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        survival_side: bool,
    ) -> Result<(f64, f64, f64), String> {
        if let Some(grid) = law {
            let (f, f_a, f_aa) =
                self.evaluate_law_survival_calibration(grid, a, q, slope, beta_h, beta_w)?;
            return Ok(if survival_side {
                (f + crate::probability::normal_cdf(-q), f_a, f_aa)
            } else {
                (crate::probability::normal_cdf(q) - f, -f_a, -f_aa)
            });
        }
        let cells = self.denested_partition_cells(a, slope, beta_h, beta_w)?;
        let scale = self.probit_frailty_scale();
        // The survival tail integrates `Φ(−η)`, the failure tail `Φ(η)`.
        let sign = if survival_side { -1.0 } else { 1.0 };
        let mut tail = 0.0;
        let mut tail_a = 0.0;
        let mut tail_aa = 0.0;
        for partition_cell in cells {
            let index = partition_cell.cell;
            let cell = exact_kernel::DenestedCubicCell {
                left: index.left,
                right: index.right,
                c0: sign * index.c0,
                c1: sign * index.c1,
                c2: sign * index.c2,
                c3: sign * index.c3,
            };
            let state = exact_kernel::evaluate_cell_moments(cell, 9)?;
            tail += state.value;
            let (dc_da_index, _) = exact_kernel::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let (dc_daa_index, _, _) = exact_kernel::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let dc_da = scale_coeff4(dc_da_index, sign * scale);
            let dc_daa = scale_coeff4(dc_daa_index, sign * scale);
            tail_a += exact_kernel::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            tail_aa += exact_kernel::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;
        }
        Ok((tail, tail_a, tail_aa))
    }
}

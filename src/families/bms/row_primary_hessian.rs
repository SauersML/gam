use super::cell_moment_assembly::{
    BernoulliInterceptSolveStats, fill_link_basis_cell_coeff_gradient,
    fill_link_basis_cell_coeff_jet, fill_score_basis_cell_coeff_jet,
};
use super::exact_eval_cache::*;
use super::family::*;
use super::gradient_paths::*;
use super::hessian_paths::*;
use super::row_kernel::*;
use super::*;

impl BernoulliMarginalSlopeFamily {
    pub(super) fn intercept_primary_point(
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Vec<f64> {
        let mut point = Vec::with_capacity(
            2 + beta_h.map(|beta| beta.len()).unwrap_or(0)
                + beta_w.map(|beta| beta.len()).unwrap_or(0),
        );
        point.push(q);
        point.push(b);
        if let Some(beta) = beta_h {
            point.extend(beta.iter().copied());
        }
        if let Some(beta) = beta_w {
            point.extend(beta.iter().copied());
        }
        point
    }

    #[inline]
    pub(super) fn cache_row_intercept(
        &self,
        row: usize,
        a: f64,
        marginal_eta: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) {
        if let Some(cache) = self.intercept_warm_starts.as_ref() {
            let beta_tag = hash_intercept_warm_start_key_flex(marginal_eta, slope, beta_h, beta_w);
            cache.store_tagged(row, a, beta_tag);
        }
    }

    pub(super) fn cache_row_intercept_predictor(
        &self,
        row: usize,
        a: f64,
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        a_u: &Array1<f64>,
    ) {
        let Some(cache) = self.intercept_warm_starts.as_ref() else {
            return;
        };
        let primary_point = Self::intercept_primary_point(q, b, beta_h, beta_w);
        if primary_point.len() != a_u.len() {
            return;
        }
        cache.store_predictor(row, a, primary_point, a_u.iter().copied().collect());
    }

    #[inline]
    pub(super) fn beta_linf(beta: Option<&Array1<f64>>) -> f64 {
        beta.map(|b| b.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())))
            .unwrap_or(0.0)
    }

    pub(super) fn near_zero_deviation_residual_bound(
        &self,
        slope: f64,
        beta_h_linf: f64,
        beta_w_linf: f64,
    ) -> f64 {
        let score_basis_sup = self
            .score_warp
            .as_ref()
            .map(|runtime| runtime.value_basis_l1_sup_norm())
            .unwrap_or(0.0);
        let link_basis_sup = self
            .link_dev
            .as_ref()
            .map(|runtime| runtime.value_basis_l1_sup_norm())
            .unwrap_or(0.0);
        // At the rigid intercept, deviations perturb the probit argument by at
        // most `s * (|b|·||h||∞ + ||w||∞)`.  Since `Φ` is globally
        // `φ(0)`-Lipschitz, the calibration residual changes by no more than
        // `φ(0)` times this argument bound after integrating against the unit
        // normal density.  The L1 basis sup-norms give
        // `||h||∞ <= K_h ||β_h||∞` and `||w||∞ <= K_w ||β_w||∞`; if this is
        // below the solver's `abs_tol`, the rigid root is already acceptable.
        normal_pdf(0.0)
            * self.probit_frailty_scale()
            * (slope.abs() * score_basis_sup * beta_h_linf + link_basis_sup * beta_w_linf)
    }

    pub(super) fn solve_row_intercept_base(
        &self,
        row: usize,
        marginal_eta: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        stats: Option<&BernoulliInterceptSolveStats>,
    ) -> Result<(f64, f64, bool), String> {
        let marginal = self.marginal_link_map(marginal_eta)?;
        let probit_scale = self.probit_frailty_scale();
        let target = marginal.mu;
        let abs_tol = 1e-8_f64.max(1e-4 * target.abs());
        let rigid_a = rigid_prescale_intercept_from_marginal(marginal.q, slope, probit_scale);
        let rigid_abs_deriv =
            rigid_prescale_intercept_derivative_abs(marginal.q, slope, probit_scale);

        let beta_h_linf = Self::beta_linf(beta_h);
        let beta_w_linf = Self::beta_linf(beta_w);
        let exact_zero_deviation = beta_h_linf == 0.0 && beta_w_linf == 0.0;
        let standard_normal_law = matches!(self.latent_measure, LatentMeasureKind::StandardNormal);
        if exact_zero_deviation && standard_normal_law {
            self.cache_row_intercept(row, rigid_a, marginal_eta, slope, beta_h, beta_w);
            return Ok((rigid_a, rigid_abs_deriv, true));
        }

        let near_zero_bound =
            self.near_zero_deviation_residual_bound(slope, beta_h_linf, beta_w_linf);
        let beta_linf_max = beta_h_linf.max(beta_w_linf);
        if standard_normal_law && near_zero_bound <= abs_tol && beta_linf_max <= f64::EPSILON.sqrt()
        {
            // Numerical guardrail for the conservative perturbation bound: the
            // exact-zero path above avoids all cell machinery, while this
            // near-zero path spends one evaluator call to guarantee that every
            // accepted row satisfies the same residual contract as the solver.
            // The extra `sqrt(eps)` coefficient cap keeps numerical
            // derivative probes out of this value-only acceptance path;
            // mathematically nonzero deviations still fall through unless they
            // are too small to carry stable derivative information.
            let (f_rigid, _, _) = self.evaluate_calibration_newton(
                row,
                rigid_a,
                marginal_eta,
                slope,
                beta_h,
                beta_w,
            )?;
            if f_rigid.abs() <= abs_tol {
                self.cache_row_intercept(row, rigid_a, marginal_eta, slope, beta_h, beta_w);
                return Ok((rigid_a, rigid_abs_deriv, true));
            }
        }

        // Use the Newton-only calibration evaluator: `solve_monotone_root`
        // safely degrades its Halley step to Newton when `F''(a) = 0`, and
        // dropping the second derivative lets us skip order-9 value-bearing
        // cell moments in favour of degree-4 moments.
        let eval = |a: f64| -> Result<(f64, f64, f64), String> {
            self.evaluate_calibration_newton(row, a, marginal_eta, slope, beta_h, beta_w)
        };

        // Closed-form fallback initial guess: rigid probit in pre-scale
        // denested coordinates:
        //   a₀ = q·√(1 + (s_f b)²) / s_f,  s_f = 1/√(1+σ²).
        // When link deviation is active, upgrade to affine-link warm start:
        //   s_f·L(u) ≈ s_f·(ℓ₀ + ℓ₁·u)
        //   ⟹  a = (q·√(1 + (s_f ℓ₁ b)²) / s_f − ℓ₀) / ℓ₁
        let a_closed_form = self.row_intercept_closed_form_seed(row, marginal, slope, beta_w)?;

        // Prefer the previous PIRLS iter's converged intercept as the
        // initial guess; β changes only a little between consecutive PIRLS
        // iterations, so the previous answer is typically within a few
        // root-solver steps of the new one. If the cache slot is NaN
        // (uninitialised) or non-finite (stale), fall back to the closed-
        // form seed.
        let current_primary_point =
            Self::intercept_primary_point(marginal_eta, slope, beta_h, beta_w);
        let predictor_a = self
            .intercept_warm_starts
            .as_ref()
            .and_then(|cache| cache.predictor_seed(row, &current_primary_point));
        // FLEX cache slot must include β_h and β_w: under link-deviation and
        // score-warp the root depends on the joint coefficient vector, not
        // just `(marginal_eta, slope)`. Without the tag a TR trial at one β
        // can read back a converged value from a different trial at the same
        // row and poison the solve.
        let flex_beta_tag = hash_intercept_warm_start_key_flex(marginal_eta, slope, beta_h, beta_w);
        let cached_a = self
            .intercept_warm_starts
            .as_ref()
            .and_then(|cache| cache.load_tagged(row, flex_beta_tag));
        let a_init = predictor_a.or(cached_a).unwrap_or(a_closed_form);

        // Note: an explicit `eval(a_closed_form)` short-circuit at this point
        // would be redundant. On cold cycle-0 `cached_a` is None, so `a_init`
        // already equals `a_closed_form` and the two-step Newton probe below
        // evaluates there with the 1e-10 tolerance from
        // `row_intercept_newton_is_converged`, matching the exact-root path
        // in `monotone_root::solve_monotone_root` (see monotone_root.rs:50-66).
        // On warm cycles, evaluating at `a_closed_form` would add an extra
        // cell-moment call even when the cached seed is already the root.

        // Adaptive acceptance tolerance: for extreme slopes the intercept
        // equation becomes numerically flat and tight absolute precision is
        // not achievable. We accept any bracketed solution at this level, so
        // pass the same tolerance to the root solver — driving it tighter
        // than `abs_tol` is wasted cell-moment work, since at large scale
        // (n=320k, FLEX active with linkwiggle + score-warp) the solver is
        // called once per row per Hessian build and the per-row cell-moment
        // kernel dominates wall time. With this tolerance the closed-form /
        // affine warm start short-circuits at `monotone_root.rs:26` for the
        // common case, instead of forcing 30+ refinement iters down to 1e-10.

        // Local Newton probe before paying for the safeguarded bracket.
        // Cycle-0 is cold at large scale, so forcing every row through the
        // bracket spends most of the wall time rebuilding identical cell
        // value integrals. The rigid/affine seed is exact when deviations vanish
        // and first-order accurate when they are small; probe that local
        // Newton basin first. The convergence test uses the same residual
        // contract as the safeguarded solver plus a tight relative-correction
        // gate, so any accept here satisfies the existing final check. Hard
        // cases still fall through unchanged.
        let probe_result = (|| -> Result<(Option<(f64, f64, f64)>, f64), String> {
            let mut a = a_init;
            let mut seed_residual = None;
            for _ in 0..6 {
                let (f, f_a, _) = eval(a)?;
                if seed_residual.is_none() {
                    seed_residual = Some(f);
                }
                if Self::row_intercept_newton_is_converged(a, f, f_a, abs_tol) {
                    return Ok((Some((a, f_a.abs(), f)), seed_residual.unwrap_or(f)));
                }
                if !(f_a.is_finite() && f_a != 0.0) {
                    break;
                }
                let next_a = a - f / f_a;
                if !next_a.is_finite() {
                    break;
                }
                a = next_a;
            }
            Ok((None, seed_residual.unwrap_or(f64::INFINITY)))
        })();

        if let Ok((accepted, seed_residual)) = &probe_result {
            if let Some(stats) = stats {
                stats.record_seed_residual(*seed_residual, abs_tol);
            }
            if let Some((a, abs_deriv, _)) = accepted {
                if let Some(stats) = stats {
                    if predictor_a.is_some() || cached_a.is_some() {
                        stats.cached_short_circuit.fetch_add(1, Ordering::Relaxed);
                    } else {
                        stats
                            .closed_form_short_circuit
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                self.cache_row_intercept(row, *a, marginal_eta, slope, beta_h, beta_w);
                return Ok((*a, *abs_deriv, false));
            }
        }

        let mut solve_result = crate::families::monotone_root::solve_monotone_root_detailed(
            eval,
            a_init,
            "bernoulli intercept",
            abs_tol,
            64,
            48,
        );

        // If the warm-started solve failed, retry once from the closed-form
        // seed. Cached `a` from a prior PIRLS iter can be far enough from
        // the current root (e.g., after a large β step) that the bracketing
        // search exhausts; the closed-form seed always sits in the correct
        // basin.
        if (predictor_a.is_some() || cached_a.is_some()) && solve_result.is_err() {
            solve_result = crate::families::monotone_root::solve_monotone_root_detailed(
                eval,
                a_closed_form,
                "bernoulli intercept",
                abs_tol,
                64,
                48,
            );
        }
        // Routine emits its own format!()-based String errors below
        // (residual rejection); enclosing return type stays Result<_, String>.
        let solve_solution = solve_result.map_err(|e| e.to_string())?;
        if let Some(stats) = stats {
            stats.record_full_solver(solve_solution.refine_iters);
        }
        let (a, abs_deriv, f_best) = (
            solve_solution.root,
            solve_solution.abs_deriv,
            solve_solution.residual,
        );

        if f_best.abs() > abs_tol {
            return Err(format!(
                "bernoulli marginal-slope intercept solve failed: \
                     residual={f_best:.3e} at a={a:.6}, target mu={target:.6}"
            ));
        }

        // Cache the converged intercept for the next PIRLS iter.
        self.cache_row_intercept(row, a, marginal_eta, slope, beta_h, beta_w);

        Ok((a, abs_deriv, false))
    }
    pub(super) fn build_row_exact_context_with_stats_and_cell_cache(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        stats: Option<&BernoulliInterceptSolveStats>,
        cache_degree9_cells: bool,
    ) -> Result<BernoulliMarginalSlopeRowExactContext, String> {
        let marginal_eta = block_states[0].eta[row];
        let marginal = self.marginal_link_map(marginal_eta)?;
        // The log-slope block now parameterizes the signed slope directly.
        let slope = block_states[1].eta[row];
        let beta_h = self.score_beta(block_states)?;
        let beta_w = self.link_beta(block_states)?;
        let (intercept, m_a, intercept_fast_path) = if self.effective_flex_active(block_states)? {
            self.solve_row_intercept_base(row, marginal_eta, slope, beta_h, beta_w, stats)?
        } else {
            let intercept = match self.latent_measure.empirical_grid_for_training_row(row)? {
                None => {
                    rigid_intercept_from_marginal(marginal.q, slope, self.probit_frailty_scale())
                }
                Some(grid) => self.empirical_rigid_intercept_for_row(
                    row,
                    marginal,
                    slope,
                    &grid.nodes,
                    &grid.weights,
                )?,
            };
            (intercept, f64::NAN, false)
        };
        // Cache degree-9 cell moments at the converged intercept so the
        // many gradient/diagonal/matvec passes that run *after* this point
        // for the same (row, β) don't re-evaluate `evaluate_cell_moments` /
        // `bivariate_normal_cdf` on identical inputs. This matters for the
        // FLEX path (linkwiggle + score-warp), where each per-row Hessian
        // build runs the cell-moment kernel once per cell per closure call.
        let degree9_cells = if cache_degree9_cells
            && self.effective_flex_active(block_states)?
            && matches!(self.latent_measure, LatentMeasureKind::StandardNormal)
        {
            let cells = self.denested_partition_cells(intercept, slope, beta_h, beta_w)?;
            // Per-row dedup: within ONE row's denested-partition output, the
            // score-warp and link-wiggle bases occasionally produce cells
            // whose `(left, right, c0, c1, c2, c3)` are bit-equal. Evaluating
            // moments once and cloning the result into the other slots is
            // numerically identical to evaluating each cell independently
            // (`evaluate_cell_moments_lru` is a pure function of the cell), and
            // skips redundant work. The dedup is purely intra-row, so it is
            // orthogonal to the per-family LRU (which is keyed across rows)
            // and the affine tail-cell memo (a separate mechanism).
            let mut dedup: HashMap<
                exact_kernel::CellFingerprint,
                exact_kernel::CellDerivativeMomentState,
            > = HashMap::new();
            let mut out: Vec<CachedDenestedCellMoments> = Vec::with_capacity(cells.len());
            for partition_cell in cells.into_iter() {
                let key = exact_kernel::CellFingerprint::new(partition_cell.cell);
                let state: exact_kernel::CellDerivativeMomentState =
                    if let Some(existing) = dedup.get(&key) {
                        existing.clone()
                    } else {
                        let computed =
                            self.evaluate_cell_derivative_moments_lru(partition_cell.cell, 9)?;
                        dedup.insert(key, computed.clone());
                        computed
                    };
                out.push(CachedDenestedCellMoments {
                    partition_cell,
                    state,
                });
            }
            Some(out)
        } else {
            None
        };
        Ok(BernoulliMarginalSlopeRowExactContext {
            intercept,
            m_a,
            intercept_fast_path,
            degree9_cells,
        })
    }

    /// Look up the pre-solved row context from the cache.
    #[inline]
    pub(super) fn row_ctx(
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row: usize,
    ) -> &BernoulliMarginalSlopeRowExactContext {
        &cache.row_contexts[row]
    }

    pub(super) fn build_exact_eval_cache_with_order(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<BernoulliMarginalSlopeExactEvalCache, String> {
        self.build_exact_eval_cache_with_options(block_states, None)
    }

    pub(super) fn build_exact_eval_cache_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: Option<&BlockwiseFitOptions>,
    ) -> Result<BernoulliMarginalSlopeExactEvalCache, String> {
        self.build_exact_eval_cache_with_options_and_context_rows(block_states, options, None)
    }

    pub(super) fn build_exact_eval_cache_for_selected_context_rows(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
        context_rows: &[usize],
    ) -> Result<BernoulliMarginalSlopeExactEvalCache, String> {
        self.build_exact_eval_cache_with_options_and_context_rows(
            block_states,
            Some(options),
            Some(context_rows),
        )
    }

    pub(super) fn build_exact_eval_cache_with_options_and_context_rows(
        &self,
        block_states: &[ParameterBlockState],
        options: Option<&BlockwiseFitOptions>,
        context_rows: Option<&[usize]>,
    ) -> Result<BernoulliMarginalSlopeExactEvalCache, String> {
        self.validate_exact_block_state_shapes(block_states)?;
        let slices = block_slices(self);
        let primary = primary_slices(&slices);
        let n = self.y.len();
        let flex_active = self.effective_flex_active(block_states)?;
        let selected_context_rows = context_rows.map(|rows| {
            let mut selected = rows
                .iter()
                .copied()
                .filter(|&row| row < n)
                .collect::<Vec<_>>();
            selected.sort_unstable();
            selected.dedup();
            selected
        });
        let context_row_count = selected_context_rows.as_ref().map_or(n, |rows| rows.len());
        let started = std::time::Instant::now();
        let process_monitor_guard = crate::process_monitor::track_scope(format!(
            "BMS exact-cache build n={n} context_rows={context_row_count} p={} flex={flex_active}",
            slices.total
        ));
        if log_exact_work(n) {
            log::info!(
                "[BMS exact-cache] build start n={} context_rows={} p={} flex={}",
                n,
                context_row_count,
                slices.total,
                flex_active
            );
        }
        let preseed_started = std::time::Instant::now();
        if let Some(rows) = selected_context_rows.as_deref() {
            self.preseed_intercept_warm_starts_for_rows(block_states, rows)?;
        } else {
            self.preseed_intercept_warm_starts(block_states)?;
        }
        if log_exact_work(n) {
            log::info!(
                "[BMS exact-cache] preseed done n={} context_rows={} elapsed={:.3}s",
                n,
                context_row_count,
                preseed_started.elapsed().as_secs_f64()
            );
        }
        if flex_active {
            exact_kernel::reset_tail_cell_moment_cache();
        }
        let stats = BernoulliInterceptSolveStats::default();
        let cell_cache_before = self.cell_moment_cache_stats.snapshot();
        // Suppress per-row `degree9_cells` caching during the parallel context
        // build: when flex is active *and* the latent measure is StandardNormal
        // (i.e. exactly when `degree9_cells` would be populated), the top-of-
        // cycle `build_row_cell_moments_bundle` invocation below also calls
        // `denested_partition_cells` for every row. Suppressing the per-row
        // cache here avoids the duplicate partition computation and the
        // unused degree-9 moment evaluations whenever the bundle succeeds.
        // When the bundle returns `None` (budget exceeded), the per-row
        // `degree9_cells` cache is reconstructed below so the row-evaluation
        // fast path that consults `row_ctx.degree9_cells` still has its
        // cache. Numerical results are unchanged either way.
        let context_started = std::time::Instant::now();
        let progress_step = (context_row_count / 10).max(1);
        let completed_rows = AtomicUsize::new(0);
        let row_contexts = if let Some(selected_rows) = selected_context_rows.as_ref() {
            let computed = selected_rows
                .par_iter()
                .copied()
                .map(|row| {
                    let ctx = self.build_row_exact_context_with_stats_and_cell_cache(
                        row,
                        block_states,
                        Some(&stats),
                        false,
                    )?;
                    if log_exact_work(n) {
                        let done = completed_rows.fetch_add(1, Ordering::Relaxed) + 1;
                        if done == context_row_count || done % progress_step == 0 {
                            log::info!(
                                "[BMS exact-cache] row-context progress rows={}/{} elapsed={:.3}s",
                                done,
                                context_row_count,
                                context_started.elapsed().as_secs_f64()
                            );
                        }
                    }
                    Ok((row, ctx))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let mut row_contexts = vec![
                BernoulliMarginalSlopeRowExactContext {
                    intercept: f64::NAN,
                    m_a: f64::NAN,
                    intercept_fast_path: false,
                    degree9_cells: None,
                };
                n
            ];
            for (row, ctx) in computed {
                row_contexts[row] = ctx;
            }
            row_contexts
        } else {
            (0..n)
                .into_par_iter()
                .map(|row| {
                    let ctx = self.build_row_exact_context_with_stats_and_cell_cache(
                        row,
                        block_states,
                        Some(&stats),
                        false,
                    )?;
                    if log_exact_work(n) {
                        let done = completed_rows.fetch_add(1, Ordering::Relaxed) + 1;
                        if done == context_row_count || done % progress_step == 0 {
                            log::info!(
                                "[BMS exact-cache] row-context progress rows={}/{} elapsed={:.3}s",
                                done,
                                context_row_count,
                                context_started.elapsed().as_secs_f64()
                            );
                        }
                    }
                    Ok(ctx)
                })
                .collect::<Result<Vec<_>, String>>()?
        };
        let fast_path_rows = row_contexts
            .iter()
            .filter(|ctx| ctx.intercept_fast_path)
            .count();
        if log_exact_work(n) {
            log::info!(
                "[BMS exact-cache] row-context done rows={} fast_path_rows={} elapsed={:.3}s",
                context_row_count,
                fast_path_rows,
                context_started.elapsed().as_secs_f64()
            );
        } else {
            log::debug!(
                "[BMS exact-cache] row-intercept zero-deviation fast path rows={}/{}",
                fast_path_rows,
                n
            );
        }
        if flex_active {
            log::info!(
                "bernoulli marginal-slope intercept seed short-circuit: cached={}, closed_form={}, full_solver={}, max_full_solver_iters={}, seed_residual_bins={{<=1e-12:{}, <=1e-10:{}, <=1e-8:{}, <=abs_tol:{}, >abs_tol:{}}}",
                stats.cached_short_circuit.load(Ordering::Relaxed),
                stats.closed_form_short_circuit.load(Ordering::Relaxed),
                stats.full_solver.load(Ordering::Relaxed),
                stats.max_full_solver_iters.load(Ordering::Relaxed),
                stats.seed_residual_le_1e12.load(Ordering::Relaxed),
                stats.seed_residual_le_1e10.load(Ordering::Relaxed),
                stats.seed_residual_le_1e8.load(Ordering::Relaxed),
                stats.seed_residual_le_abs_tol.load(Ordering::Relaxed),
                stats.seed_residual_gt_abs_tol.load(Ordering::Relaxed),
            );
        }
        if flex_active {
            let (cell_hits, cell_misses, cell_hit_rate) = self
                .cell_moment_cache_stats
                .hit_rate_delta(cell_cache_before);
            log::info!(
                "[BMS cell-moment LRU] cycle hits={} misses={} hit_rate={:.1}% entries={} resident_mib={:.1}/{:.1}",
                cell_hits,
                cell_misses,
                100.0 * cell_hit_rate,
                self.cell_moment_lru.len(),
                self.cell_moment_lru.resident_bytes() as f64 / (1024.0 * 1024.0),
                self.cell_moment_lru.max_bytes() as f64 / (1024.0 * 1024.0),
            );
            let tail_stats = exact_kernel::tail_cell_moment_cache_stats();
            log::info!(
                "[BMS exact-cache] affine tail-cell memo: hits={} misses={} entries={} hit_rate={:.3}%",
                tail_stats.hits,
                tail_stats.misses,
                tail_stats.entries,
                100.0 * tail_stats.hit_rate(),
            );
        }
        let row_cell_mask = options
            .and_then(|opts| opts.outer_score_subsample.as_ref())
            .map(|subsample| subsample.mask.as_slice());
        let row_cell_started = std::time::Instant::now();
        let row_cell_moments =
            self.build_row_cell_moments_bundle(block_states, &row_contexts, 9, row_cell_mask)?;
        // #979 Stage C: when the dense bundle was refused (the large-n
        // regime), build the certified Chebyshev cell-moment family forest
        // instead — O(leaves × combos × m²) ladder evaluations once per β,
        // then transcendental-free per-row moment lookups with per-cell
        // ladder fallback.
        let cell_family_forest = if row_cell_moments.is_none() {
            self.build_cell_family_forest(block_states, &row_contexts)?
        } else {
            None
        };
        if log_exact_work(n) {
            log::info!(
                "[BMS exact-cache] row-cell phase done n={} selected_rows={} built={} forest={} elapsed={:.3}s",
                n,
                row_cell_mask.map_or(n, <[usize]>::len),
                row_cell_moments.is_some(),
                cell_family_forest.is_some(),
                row_cell_started.elapsed().as_secs_f64()
            );
            // Ladder + forest observability (#979): does the progressive GL
            // ladder certify early (win) or fall through to 384 (cost), and
            // does the family forest actually cover rows or fall back to the
            // ladder? Cumulative process-wide counters — the deltas across a
            // fit reveal whether either mechanism earns its complexity.
            let (ladder_hist, ladder_terminal) = exact_kernel::non_affine_ladder_cert_histogram();
            let (forest_hits, forest_fallbacks) =
                crate::families::cell_moment_family::forest_coverage_counts();
            log::info!(
                "[BMS ladder/forest stats] ladder_cert_by_rung={ladder_hist:?} ladder_terminal_384={ladder_terminal} forest_covered_rows={forest_hits} forest_fallback_rows={forest_fallbacks}"
            );
            log::info!(
                "[BMS exact-cache] build done n={} context_rows={} p={} flex={} elapsed={:.3}s",
                n,
                context_row_count,
                slices.total,
                flex_active,
                started.elapsed().as_secs_f64()
            );
        }
        drop(process_monitor_guard);
        Ok(BernoulliMarginalSlopeExactEvalCache {
            slices,
            primary,
            row_contexts,
            row_cell_moments,
            cell_family_forest,
            row_cell_moments_d15: crate::resource::RayonSafeOnce::new(),
            row_cell_moments_d21: crate::resource::RayonSafeOnce::new(),
            row_primary_hessians: RowPrimaryEvalCache::Empty,
            rigid_third_full: crate::resource::RayonSafeOnce::new(),
            rigid_fourth_full: crate::resource::RayonSafeOnce::new(),
            flex_axis_third_tensors: crate::resource::RayonSafeOnce::new(),
            flex_axis_fourth_tensors: crate::resource::RayonSafeOnce::new(),
        })
    }

    /// Build the certified Chebyshev cell-moment family forest for the
    /// current β snapshot (#979 Stage C). `None` (never an error) when the
    /// FLEX path is inactive, the latent measure is non-standard-normal,
    /// there are no deviation breakpoints, or the forest partition fails —
    /// every caller falls back to direct ladder quadrature per cell, so a
    /// missing forest is a performance regression only, never a numerical
    /// one.
    pub(crate) fn build_cell_family_forest(
        &self,
        block_states: &[ParameterBlockState],
        row_contexts: &[BernoulliMarginalSlopeRowExactContext],
    ) -> Result<Option<crate::families::cell_moment_family::CellFamilyForest>, String> {
        use crate::families::cell_moment_family::{
            CellFamilyForest, CellMomentFamilySpec, ComboKey,
        };
        if !self.effective_flex_active(block_states)? {
            return Ok(None);
        }
        if !matches!(self.latent_measure, LatentMeasureKind::StandardNormal) {
            return Ok(None);
        }
        let n = self.y.len();
        if row_contexts.len() != n {
            return Ok(None);
        }
        let score_breaks: Vec<f64> = self
            .score_warp
            .as_ref()
            .map(|runtime| runtime.breakpoints().to_vec())
            .unwrap_or_default();
        let link_breaks: Vec<f64> = self
            .link_dev
            .as_ref()
            .map(|runtime| runtime.breakpoints().to_vec())
            .unwrap_or_default();
        if score_breaks.is_empty() && link_breaks.is_empty() {
            return Ok(None);
        }
        let beta_h = self.score_beta(block_states)?;
        let beta_w = self.link_beta(block_states)?;
        let mut a_rows = vec![0.0_f64; n];
        let mut b_rows = vec![0.0_f64; n];
        for row in 0..n {
            a_rows[row] = row_contexts[row].intercept;
            b_rows[row] = block_states[1].eta[row];
        }
        // Subsampled cache builds leave unselected rows at NaN intercepts;
        // the forest requires finite coordinates, so skip the forest rather
        // than poison the partition (those builds are small by design).
        if a_rows.iter().any(|v| !v.is_finite()) || b_rows.iter().any(|v| !v.is_finite()) {
            return Ok(None);
        }
        let started = std::time::Instant::now();
        let mut forest =
            match CellFamilyForest::partition(&a_rows, &b_rows, &score_breaks, &link_breaks) {
                Ok(forest) => forest,
                Err(reason) => {
                    log::debug!("[BMS cell-family-forest] partition skipped: {reason}");
                    return Ok(None);
                }
            };
        // Demand pass: every row's interior finite cells, keyed by combo.
        // Tail (semi-infinite) cells keep their closed-form affine anchors —
        // interpolating them would be slower than the closed form.
        let demands: Vec<(usize, ComboKey, CellMomentFamilySpec)> = (0..n)
            .into_par_iter()
            .map(|row| -> Result<Vec<_>, String> {
                let cells =
                    self.denested_partition_cells(a_rows[row], b_rows[row], beta_h, beta_w)?;
                Ok(cells
                    .into_iter()
                    .filter(|pc| pc.cell.left.is_finite() && pc.cell.right.is_finite())
                    .map(|pc| {
                        (
                            row,
                            ComboKey::new(pc.score_span, pc.link_span, pc.left_edge, pc.right_edge),
                            CellMomentFamilySpec {
                                score_span: pc.score_span,
                                link_span: pc.link_span,
                                left: pc.left_edge,
                                right: pc.right_edge,
                                max_degree: 9,
                            },
                        )
                    })
                    .collect())
            })
            .try_reduce(Vec::new, |mut left, right| {
                left.extend(right);
                Ok(left)
            })?;
        forest.build_families(demands);
        log::info!(
            "[BMS cell-family-forest] built n={} leaves={} eligible={} elapsed={:.3}s",
            n,
            forest.total_leaves(),
            forest.eligible_leaves(),
            started.elapsed().as_secs_f64(),
        );
        Ok(Some(forest))
    }

    /// Build a top-of-cycle [`RowCellMomentsBundle`] at the given
    /// `max_degree`. Returns `None` when the FLEX path is inactive, when an
    /// empirical latent grid is in effect (the row kernel takes a non-cell
    /// path), or when the estimated resident bytes exceed the active
    /// resource-policy budget. Numerical equivalence with the legacy per-row
    /// path is unconditional: callers always fall back to
    /// `degree9_cells`/on-demand cell evaluation when the bundle is absent.
    pub(super) fn build_row_cell_moments_bundle(
        &self,
        block_states: &[ParameterBlockState],
        row_contexts: &[BernoulliMarginalSlopeRowExactContext],
        max_degree: usize,
        row_mask: Option<&[usize]>,
    ) -> Result<Option<RowCellMomentsBundle>, String> {
        if !self.effective_flex_active(block_states)? {
            return Ok(None);
        }
        // Empirical-grid rows take a non-cell code path inside
        // `compute_row_analytic_flex_from_parts_into`, so the bundle would
        // never be consulted. Skip the build to avoid wasted work.
        if !matches!(self.latent_measure, LatentMeasureKind::StandardNormal) {
            return Ok(None);
        }
        let n = self.y.len();
        let beta_h = self.score_beta(block_states)?;
        let beta_w = self.link_beta(block_states)?;
        let selected_rows: Vec<usize> = match row_mask {
            Some(mask) => mask.iter().copied().filter(|&row| row < n).collect(),
            None => (0..n).collect(),
        };
        if selected_rows.is_empty() {
            return Ok(None);
        }
        let selected_row_count = selected_rows.len();
        let max_cells = self.max_denested_partition_cells_per_row();
        let max_n_cells = selected_row_count.saturating_mul(max_cells);
        let upper_bound_bytes =
            RowCellMomentsBundle::estimated_resident_bytes(n, max_n_cells, max_degree);
        let limit_bytes = self.policy.max_operator_cache_bytes;
        if upper_bound_bytes > limit_bytes {
            log::info!(
                "[BMS row-cell-moments] skip precompute n={} selected_rows={} max_cells_per_row={} degree={} upper_bound_bytes={} limit_bytes={}",
                n,
                selected_row_count,
                max_cells,
                max_degree,
                upper_bound_bytes,
                limit_bytes
            );
            return Ok(None);
        }
        let started = std::time::Instant::now();
        let process_monitor_guard = crate::process_monitor::track_scope(format!(
            "BMS row-cell-moments n={n} selected_rows={selected_row_count} degree={max_degree}"
        ));
        if log_exact_work(n) {
            log::info!(
                "[BMS row-cell-moments] partition start n={} selected_rows={} degree={}",
                n,
                selected_row_count,
                max_degree
            );
        }
        let partitions: Vec<(usize, Vec<exact_kernel::DenestedPartitionCell>)> = selected_rows
            .into_par_iter()
            .map(|row| {
                self.denested_partition_cells(
                    row_contexts[row].intercept,
                    block_states[1].eta[row],
                    beta_h,
                    beta_w,
                )
                .map(|cells| (row, cells))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let selected_n = partitions.len();
        let n_cells = partitions
            .iter()
            .map(|(_, cells)| cells.len())
            .sum::<usize>();
        if log_exact_work(n) {
            log::info!(
                "[BMS row-cell-moments] partition done n={} selected_rows={} cells={} elapsed={:.3}s",
                n,
                selected_n,
                n_cells,
                started.elapsed().as_secs_f64()
            );
        }
        let estimated_bytes =
            RowCellMomentsBundle::estimated_resident_bytes(n, n_cells, max_degree);
        if estimated_bytes > limit_bytes {
            log::warn!(
                "[BMS row-cell-moments] skip precompute n={} selected_rows={} cells={} degree={} estimated_bytes={} limit_bytes={}",
                n,
                selected_n,
                n_cells,
                max_degree,
                estimated_bytes,
                limit_bytes
            );
            return Ok(None);
        }
        let moment_started = std::time::Instant::now();
        let computed_rows = partitions
            .into_par_iter()
            .map(|(row, cells)| {
                let moments = cells
                    .into_iter()
                    .map(|partition_cell| {
                        // Bundle rows are already the reusable state for this
                        // beta snapshot. Under linkwiggle the bit-exact
                        // cross-row LRU key is row-specific, so probing the
                        // fit-lifetime LRU here just adds lock/eviction churn
                        // without reuse. Evaluate directly and let the bundle
                        // carry the reusable moments.
                        exact_kernel::evaluate_cell_derivative_moments_uncached(
                            partition_cell.cell,
                            max_degree,
                        )
                        .map(|state| CachedDenestedCellMoments {
                            partition_cell,
                            state,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok((row, moments))
            })
            .collect::<Result<Vec<_>, String>>()?;
        // Block-12 Stage-1 GPU-substrate parity guard. The substrate
        // `try_build_cubic_cell_derivative_moments` is the future per-row
        // moment producer (host path today, NVRTC kernel on V100 later);
        // landing the call here makes it a real production consumer of the
        // substrate's pub(crate) entry point and surfaces any divergence
        // from the existing LRU evaluator the moment it appears. We sample
        // a small prefix of rows so the debug-build cost stays bounded for
        // large-scale fits; the production build pays nothing because the
        // block is `cfg(debug_assertions)`-gated.
        #[cfg(debug_assertions)]
        {
            use crate::gpu::cubic_cell::branch::classify_cell_for_gpu;
            use crate::gpu::cubic_cell::{
                CubicCellDerivativeMomentHostView, CubicCellMomentResidency, GpuDenestedCubicCell,
                try_build_cubic_cell_derivative_moments,
            };
            const PARITY_ROW_BUDGET: usize = 4;
            let mut sample_cells: Vec<GpuDenestedCubicCell> = Vec::new();
            let mut sample_branches = Vec::new();
            let mut sample_cpu_moments: Vec<Vec<f64>> = Vec::new();
            for (_, moments) in computed_rows.iter().take(PARITY_ROW_BUDGET) {
                for cached in moments {
                    let cell = cached.partition_cell.cell;
                    let gpu_cell = GpuDenestedCubicCell {
                        left: cell.left,
                        right: cell.right,
                        c0: cell.c0,
                        c1: cell.c1,
                        c2: cell.c2,
                        c3: cell.c3,
                    };
                    let branch = classify_cell_for_gpu(gpu_cell).map_err(|status| {
                        format!(
                            "BMS row-cell-moments parity classifier rejected CPU-evaluated cell \
                             row_sample={} cell_sample={} status={}",
                            sample_cpu_moments.len(),
                            sample_cells.len(),
                            status as u8
                        )
                    })?;
                    sample_cells.push(gpu_cell);
                    sample_branches.push(branch);
                    sample_cpu_moments.push(cached.state.moments.to_vec());
                }
            }
            if !sample_cells.is_empty() {
                let view = CubicCellDerivativeMomentHostView {
                    cells: &sample_cells,
                    branches: &sample_branches,
                    max_degree,
                    residency: CubicCellMomentResidency::Host,
                };
                match try_build_cubic_cell_derivative_moments(view) {
                    Ok(Some(output)) => {
                        use crate::gpu::cubic_cell::{
                            CubicCellDerivativeMomentOutput, CubicCellMomentStatus,
                        };
                        let (sub_moments, sub_status, stride) = match output {
                            CubicCellDerivativeMomentOutput::Host {
                                moments,
                                status,
                                stride,
                            } => (moments, status, stride),
                            #[cfg(target_os = "linux")]
                            CubicCellDerivativeMomentOutput::Device { .. } => {
                                // The view above explicitly requested
                                // `CubicCellMomentResidency::Host`, and the substrate's
                                // contract (`try_build_cubic_cell_derivative_moments` in
                                // `src/gpu/cubic_cell/mod.rs:170`) guarantees that a Host
                                // request returns `Host(...)` even on Linux+CUDA. Reaching
                                // this arm means the substrate's contract was violated —
                                // a hard programming error in the GPU dispatcher, not a
                                // runtime condition we can recover from. Panicking
                                // surfaces it at the call site.
                                // SAFETY: unreachable by substrate contract — Host
                                // request must return Host residency; reaching this
                                // arm is a programming error, not a runtime condition.
                                panic!(
                                    "BMS row-cell-moments parity probe requested Host residency \
                                     but substrate returned device-resident output"
                                )
                            }
                        };
                        assert_eq!(stride, max_degree + 1);
                        assert_eq!(sub_status.len(), sample_cells.len());
                        for (i, cpu_row) in sample_cpu_moments.iter().enumerate() {
                            assert_eq!(
                                sub_status[i],
                                CubicCellMomentStatus::Ok as u8,
                                "BMS row-cell-moments parity: substrate refused cell {i} (status={})",
                                sub_status[i]
                            );
                            let sub_row = &sub_moments[i * stride..(i + 1) * stride];
                            let copy_len = cpu_row.len().min(stride);
                            for k in 0..copy_len {
                                let want = cpu_row[k];
                                let got = sub_row[k];
                                let denom = want.abs().max(1.0);
                                let rel = (got - want).abs() / denom;
                                let abs = (got - want).abs();
                                assert!(
                                    abs <= 1e-12 || rel <= 1e-11,
                                    "BMS row-cell-moments parity drift cell={i} k={k} \
                                     cpu={want:.17e} substrate={got:.17e} abs={abs:.3e} rel={rel:.3e}"
                                );
                            }
                        }
                    }
                    Ok(None) => {
                        // SAFETY: substrate's `Ok(None)` contract is
                        // reserved for empty input; the surrounding
                        // `if !sample_cells.is_empty()` guards against
                        // that. A `None` return for a populated sample
                        // is a contract violation that must be visible
                        // at the first fit in debug builds, not silently
                        // tolerated.
                        panic!(
                            "BMS row-cell-moments parity: substrate returned Ok(None) for a non-empty sample of {} cells",
                            sample_cells.len()
                        );
                    }
                    // SAFETY: substrate errors during the parity sample
                    // mean the host evaluator (which we are checking
                    // against the LRU path) disagreed on cells the LRU
                    // already accepted. Continuing past such a divergence
                    // hides correctness bugs the parity guard is here to
                    // catch; abort the debug-build fit.
                    Err(err) => panic!(
                        "BMS row-cell-moments parity: substrate failed on {} sample cells: {}",
                        sample_cells.len(),
                        err
                    ),
                }
            }
        }
        let mut rows = vec![None; n];
        for (row, moments) in computed_rows {
            rows[row] = Some(moments);
        }
        if log_exact_work(n) {
            log::info!(
                "[BMS row-cell-moments] precomputed n={} selected_rows={} cells={} degree={} estimated_bytes={} elapsed={:.3}s",
                n,
                selected_n,
                n_cells,
                max_degree,
                estimated_bytes,
                moment_started.elapsed().as_secs_f64()
            );
        }
        drop(process_monitor_guard);
        Ok(Some(RowCellMomentsBundle {
            max_degree,
            selected_rows: selected_n,
            rows,
        }))
    }

    pub(crate) fn extend_row_cell_moments_bundle(
        &self,
        base: &RowCellMomentsBundle,
        required_degree: usize,
    ) -> Result<Option<RowCellMomentsBundle>, String> {
        if base.max_degree >= required_degree {
            return Ok(Some(base.clone()));
        }
        if !base.covers_all_rows() {
            return Ok(None);
        }
        let n = self.y.len();
        if base.rows.len() != n {
            return Err(format!(
                "BMS row-cell-moments upgrade row-count mismatch: bundle rows={} expected={n}",
                base.rows.len()
            ));
        }
        let n_cells = base
            .rows
            .iter()
            .map(|row| row.as_ref().map_or(0, Vec::len))
            .sum::<usize>();
        let estimated_bytes =
            RowCellMomentsBundle::estimated_resident_bytes(n, n_cells, required_degree);
        let limit_bytes = self.policy.max_operator_cache_bytes;
        if estimated_bytes > limit_bytes {
            log::info!(
                "[BMS row-cell-moments] skip upgrade n={} selected_rows={} cells={} from_degree={} degree={} estimated_bytes={} limit_bytes={}",
                n,
                base.selected_rows,
                n_cells,
                base.max_degree,
                required_degree,
                estimated_bytes,
                limit_bytes
            );
            return Ok(None);
        }

        let started = std::time::Instant::now();
        let process_monitor_guard = crate::process_monitor::track_scope(format!(
            "BMS row-cell-moments upgrade n={n} degree={}->{required_degree}",
            base.max_degree
        ));
        let rows = base
            .rows
            .par_iter()
            .map(|row| {
                row.as_ref()
                    .map(|entries| {
                        entries
                            .iter()
                            .map(|entry| {
                                exact_kernel::evaluate_cell_derivative_moments_uncached(
                                    entry.partition_cell.cell,
                                    required_degree,
                                )
                                .map(|state| {
                                    CachedDenestedCellMoments {
                                        partition_cell: entry.partition_cell,
                                        state,
                                    }
                                })
                            })
                            .collect::<Result<Vec<_>, String>>()
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, String>>()?;
        if log_exact_work(n) {
            log::info!(
                "[BMS row-cell-moments] upgraded n={} selected_rows={} cells={} from_degree={} degree={} estimated_bytes={} elapsed={:.3}s",
                n,
                base.selected_rows,
                n_cells,
                base.max_degree,
                required_degree,
                estimated_bytes,
                started.elapsed().as_secs_f64()
            );
        }
        drop(process_monitor_guard);
        Ok(Some(RowCellMomentsBundle {
            max_degree: required_degree,
            selected_rows: base.selected_rows,
            rows,
        }))
    }

    /// BMS-FLEX GPU milestone 1: pack the row-primary Hessian inputs for the
    /// Stage-2 device kernel in `crate::families::bms::gpu::row`. Returns `None`
    /// when any precondition fails (latent is not StandardNormal, the
    /// row-cell-moments bundle was not materialised, or score-warp /
    /// link-deviation runtimes are missing); the caller then falls back to
    /// the CPU rayon loop.
    ///
    /// The packed bundle mirrors `compute_row_analytic_flex_from_parts_into`
    /// (`StandardNormal` branch at lines 9047–9314) field-for-field. The
    /// per-cell coefficient families are built here on the host (cheap
    /// scalar work) so the device kernel reads only flat SoA buffers and
    /// keeps its inner loop free of cubic-cell partial-derivative math.
    pub(super) fn pack_bms_flex_row_kernel_inputs(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Option<crate::families::bms::gpu::row::BmsFlexRowKernelInputsOwned>, String> {
        use super::exact_kernel as exact;
        use crate::families::marginal_slope_shared::SparsePrimaryCoeffJetView;

        // ── Preconditions: the Stage-2 kernel only handles the StandardNormal
        //    cell-loop branch with a pre-built row-cell-moments bundle. The
        //    empirical-grid branch and the per-row degree-9 fallback both
        //    require additional packing the kernel does not consume yet.
        if !matches!(self.latent_measure, LatentMeasureKind::StandardNormal) {
            return Ok(None);
        }
        let Some(bundle) = cache.row_cell_moments.as_ref() else {
            return Ok(None);
        };
        let primary = &cache.primary;
        let r = primary.total;
        if r < 2 || r > crate::families::bms::gpu::row::MAX_R {
            return Ok(None);
        }
        let h_range = primary.h.clone();
        let w_range = primary.w.clone();
        let p_h = h_range.as_ref().map(|range| range.len()).unwrap_or(0);
        let p_w = w_range.as_ref().map(|range| range.len()).unwrap_or(0);
        if r != 2 + p_h + p_w {
            return Ok(None);
        }
        if p_h > 0 && self.score_warp.is_none() {
            return Ok(None);
        }
        if p_w > 0 && self.link_dev.is_none() {
            return Ok(None);
        }
        let beta_h = self.score_beta(block_states)?;
        let beta_w = self.link_beta(block_states)?;
        let scale = self.probit_frailty_scale();
        let n = self.y.len();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();

        // ── Phase-4 device-moment plan. On Linux+CUDA we skip the host
        //    `cell_moments` fill in the per-row loop and instead build the
        //    moments on the GPU via the cubic-cell substrate, attaching the
        //    resulting `CudaSlice<f64>` directly to the owned bundle so
        //    `launch_bms_flex_row_kernel` consumes it without a host
        //    upload. The host fill stays as the fallback on hosts without
        //    a runtime (and on every non-Linux build).
        #[cfg(target_os = "linux")]
        let build_device_moments = crate::gpu::runtime::GpuRuntime::global().is_some();
        #[cfg(not(target_os = "linux"))]
        let build_device_moments = false;

        // ── First pass: row offsets + total cell count. The Stage-2 kernel
        //    consumes a CSR `cell_offsets[n+1]` with `total_cells =
        //    cell_offsets[n]`; reject up front any row whose cells were
        //    not materialised at degree ≥ 9 (the kernel needs `m_0..m_9`).
        let mut cell_offsets: Vec<u32> = Vec::with_capacity(n + 1);
        cell_offsets.push(0);
        let mut total_cells: u32 = 0;
        for row in 0..n {
            let Some(row_cells) = bundle.row(row, 9) else {
                return Ok(None);
            };
            let len_u32 = u32::try_from(row_cells.len()).map_err(|_| {
                format!("bms_flex_row pack: row {row} cell count exceeds u32 range")
            })?;
            total_cells = total_cells
                .checked_add(len_u32)
                .ok_or_else(|| "bms_flex_row pack: total cell count overflows u32".to_string())?;
            cell_offsets.push(total_cells);
        }
        let total_cells_us = total_cells as usize;

        // ── Per-row scalars + observed-point pre-eval buffers.
        let mut row_q = Vec::<f64>::with_capacity(n);
        let mut row_b = Vec::<f64>::with_capacity(n);
        let mut row_mu1 = Vec::<f64>::with_capacity(n);
        let mut row_mu2 = Vec::<f64>::with_capacity(n);
        let mut row_zobs = Vec::<f64>::with_capacity(n);
        let mut row_y = Vec::<f64>::with_capacity(n);
        let mut row_w = Vec::<f64>::with_capacity(n);
        let mut row_chi = Vec::<f64>::with_capacity(n);
        let mut row_xi = Vec::<f64>::with_capacity(n);
        let mut row_rho = vec![0.0_f64; n * r];
        let mut row_tau = vec![0.0_f64; n * r];
        let mut row_ruv = vec![0.0_f64; n * r * r];

        // ── Per-cell SoA arrays sized once.
        let coeff4 = crate::families::bms::gpu::row::COEFF4;
        let moment_stride = crate::families::bms::gpu::row::MOMENT_STRIDE;
        let mut cell_c0 = vec![0.0_f64; total_cells_us];
        let mut cell_c1 = vec![0.0_f64; total_cells_us];
        let mut cell_c2 = vec![0.0_f64; total_cells_us];
        let mut cell_c3 = vec![0.0_f64; total_cells_us];
        let mut cell_a = vec![0.0_f64; total_cells_us * coeff4];
        let mut cell_aa = vec![0.0_f64; total_cells_us * coeff4];
        let r_minus_1 = r.saturating_sub(1);
        let mut cell_r_buf = vec![0.0_f64; total_cells_us * r_minus_1 * coeff4];
        let mut cell_ar = vec![0.0_f64; total_cells_us * r_minus_1 * coeff4];
        let mut cell_sbb = vec![0.0_f64; total_cells_us * coeff4];
        let mut cell_sbh = vec![0.0_f64; total_cells_us * p_h * coeff4];
        let mut cell_sbw = vec![0.0_f64; total_cells_us * p_w * coeff4];
        // When `build_device_moments` is set, the host `cell_moments` vec
        // is unused (the launcher consumes the device buffer); we leave
        // it empty so it doesn't waste RAM in large-scale jobs.
        let mut cell_moments: Vec<f64> = if build_device_moments {
            Vec::new()
        } else {
            vec![0.0_f64; total_cells_us * moment_stride]
        };
        // Per-cell SoA for the device cubic-cell substrate. Populated on
        // every code path so the compiler sees `gpu_cells`/`gpu_branches`
        // used unconditionally — the substrate dispatch below only fires
        // when `build_device_moments` is true, but the small Vec push cost
        // per cell is negligible compared to the moment compute itself.
        let mut gpu_cells: Vec<crate::gpu::cubic_cell::GpuDenestedCubicCell> =
            Vec::with_capacity(total_cells_us);
        let mut gpu_branches: Vec<crate::gpu::cubic_cell::GpuCellBranchTag> =
            Vec::with_capacity(total_cells_us);

        // Reusable per-row coefficient buffers. Same layout as
        // BernoulliMarginalSlopeFlexRowScratch's owned [f64;4] slices.
        let mut coeff_u: Vec<[f64; 4]> = vec![[0.0; 4]; r];
        let mut coeff_au: Vec<[f64; 4]> = vec![[0.0; 4]; r];
        let mut coeff_bu: Vec<[f64; 4]> = vec![[0.0; 4]; r];
        let zero_family: Vec<[f64; 4]> = vec![[0.0; 4]; r];

        for row in 0..n {
            let row_ctx = Self::row_ctx(cache, row);
            let a = row_ctx.intercept;
            let q = block_states[0].eta[row];
            let b = block_states[1].eta[row];
            let marginal = self.marginal_link_map(q)?;
            row_q.push(q);
            row_b.push(b);
            row_mu1.push(marginal.mu1);
            row_mu2.push(marginal.mu2);
            row_zobs.push(self.z[row]);
            row_y.push(self.y[row]);
            row_w.push(self.weights[row]);

            let start = cell_offsets[row] as usize;
            let row_cells = bundle
                .row(row, 9)
                .expect("row cell moments presence verified above");
            for (local_idx, entry) in row_cells.iter().enumerate() {
                let cell_idx = start + local_idx;
                let cell = entry.partition_cell.cell;
                let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
                let u_mid = a + b * z_mid;

                cell_c0[cell_idx] = cell.c0;
                cell_c1[cell_idx] = cell.c1;
                cell_c2[cell_idx] = cell.c2;
                cell_c3[cell_idx] = cell.c3;

                // dc_da, dc_db (scaled)
                let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                    entry.partition_cell.score_span,
                    entry.partition_cell.link_span,
                    a,
                    b,
                );
                let dc_da = scale_coeff4(dc_da_raw, scale);
                let dc_db = scale_coeff4(dc_db_raw, scale);
                let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                    entry.partition_cell.score_span,
                    entry.partition_cell.link_span,
                    a,
                    b,
                );
                let dc_daa = scale_coeff4(dc_daa_raw, scale);
                let dc_dab = scale_coeff4(dc_dab_raw, scale);
                let dc_dbb = scale_coeff4(dc_dbb_raw, scale);

                // cell_a, cell_aa.
                let a_base = cell_idx * coeff4;
                for k in 0..coeff4 {
                    cell_a[a_base + k] = dc_da[k];
                    cell_aa[a_base + k] = dc_daa[k];
                }

                // Reset per-row-cell coefficient families.
                for slot in coeff_u.iter_mut() {
                    *slot = [0.0; 4];
                }
                for slot in coeff_au.iter_mut() {
                    *slot = [0.0; 4];
                }
                for slot in coeff_bu.iter_mut() {
                    *slot = [0.0; 4];
                }
                coeff_u[1] = dc_db;
                coeff_au[1] = dc_dab;
                coeff_bu[1] = dc_dbb;

                if let (Some(h_range), Some(runtime)) = (h_range.as_ref(), score_runtime) {
                    Self::for_each_deviation_basis_cubic_at(
                        runtime,
                        h_range,
                        z_mid,
                        "score-warp",
                        |_, idx, basis_span| {
                            fill_score_basis_cell_coeff_jet(
                                idx,
                                basis_span,
                                b,
                                scale,
                                &mut coeff_u,
                                &mut coeff_bu,
                            );
                            Ok(())
                        },
                    )?;
                }
                if let (Some(w_range), Some(runtime)) = (w_range.as_ref(), link_runtime) {
                    Self::for_each_deviation_basis_cubic_at(
                        runtime,
                        w_range,
                        u_mid,
                        "link-wiggle",
                        |_, idx, basis_span| {
                            fill_link_basis_cell_coeff_gradient(
                                idx,
                                basis_span,
                                a,
                                b,
                                scale,
                                &mut coeff_u,
                                &mut coeff_au,
                                &mut coeff_bu,
                            );
                            Ok(())
                        },
                    )?;
                }

                // cell_r / cell_ar: indexed u in 1..r → slot u-1.
                let r_base = cell_idx * r_minus_1 * coeff4;
                for u in 1..r {
                    let off = r_base + (u - 1) * coeff4;
                    for k in 0..coeff4 {
                        cell_r_buf[off + k] = coeff_u[u][k];
                        cell_ar[off + k] = coeff_au[u][k];
                    }
                }
                // cell_sbb = coeff_bu[1].
                for k in 0..coeff4 {
                    cell_sbb[a_base + k] = coeff_bu[1][k];
                }
                // cell_sbh[c, j, *] = coeff_bu[h_range.start + j].
                if let Some(h_range) = h_range.as_ref() {
                    let h_base = cell_idx * p_h * coeff4;
                    for j in 0..p_h {
                        let off = h_base + j * coeff4;
                        let src = &coeff_bu[h_range.start + j];
                        for k in 0..coeff4 {
                            cell_sbh[off + k] = src[k];
                        }
                    }
                }
                // cell_sbw[c, j, *] = coeff_bu[w_range.start + j].
                if let Some(w_range) = w_range.as_ref() {
                    let w_base = cell_idx * p_w * coeff4;
                    for j in 0..p_w {
                        let off = w_base + j * coeff4;
                        let src = &coeff_bu[w_range.start + j];
                        for k in 0..coeff4 {
                            cell_sbw[off + k] = src[k];
                        }
                    }
                }
                // Always push the cell into the device-substrate SoA so
                // it's available for the Phase-4 GPU moment build below.
                // Push order is `cell_idx` (= start + local_idx) so the
                // resulting `[total_cells, MOMENT_STRIDE]` device buffer
                // is indexed identically to the host `cell_moments` vec.
                assert_eq!(gpu_cells.len(), cell_idx);
                gpu_cells.push(crate::gpu::cubic_cell::GpuDenestedCubicCell {
                    left: cell.left,
                    right: cell.right,
                    c0: cell.c0,
                    c1: cell.c1,
                    c2: cell.c2,
                    c3: cell.c3,
                });
                let branch = if !cell.left.is_finite() || !cell.right.is_finite() {
                    crate::gpu::cubic_cell::GpuCellBranchTag::AffineTail
                } else if cell.c2 == 0.0 && cell.c3 == 0.0 {
                    crate::gpu::cubic_cell::GpuCellBranchTag::Affine
                } else {
                    crate::gpu::cubic_cell::GpuCellBranchTag::NonAffineFinite
                };
                gpu_branches.push(branch);

                // cell_moments: copy state.moments, zero-pad to 10 — only
                // when the host fallback path is in use. When the
                // device-moment build is selected, this storage is
                // skipped entirely and the substrate produces moments
                // directly on the GPU below.
                if !build_device_moments {
                    let mom_base = cell_idx * moment_stride;
                    let src_moments: &[f64] = &entry.state.moments;
                    let copy_len = src_moments.len().min(moment_stride);
                    for k in 0..copy_len {
                        cell_moments[mom_base + k] = src_moments[k];
                    }
                }
            }

            // ── Observed-point pre-evaluation (mirrors CPU lines 9265–9314).
            let z_obs = self.z[row];
            let u_obs = a + b * z_obs;
            let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
            let chi_obs = eval_coeff4_at(&obs.dc_da, z_obs);
            let xi_obs = eval_coeff4_at(&obs.dc_daa, z_obs);
            row_chi.push(chi_obs);
            row_xi.push(xi_obs);

            // g_u_fixed / g_au_fixed / g_bu_fixed at z_obs (score) / u_obs (link).
            let mut g_u_fixed: Vec<[f64; 4]> = vec![[0.0; 4]; r];
            let mut g_au_fixed: Vec<[f64; 4]> = vec![[0.0; 4]; r];
            let mut g_bu_fixed: Vec<[f64; 4]> = vec![[0.0; 4]; r];
            g_u_fixed[1] = obs.dc_db;
            g_au_fixed[1] = obs.dc_dab;
            g_bu_fixed[1] = obs.dc_dbb;
            if let (Some(h_range), Some(runtime)) = (h_range.as_ref(), score_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    h_range,
                    z_obs,
                    "score-warp observed",
                    |_, idx, basis_span| {
                        fill_score_basis_cell_coeff_jet(
                            idx,
                            basis_span,
                            b,
                            scale,
                            &mut g_u_fixed,
                            &mut g_bu_fixed,
                        );
                        Ok(())
                    },
                )?;
            }
            if let (Some(w_range), Some(runtime)) = (w_range.as_ref(), link_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    w_range,
                    u_obs,
                    "link-wiggle observed",
                    |_, idx, basis_span| {
                        fill_link_basis_cell_coeff_gradient(
                            idx,
                            basis_span,
                            a,
                            b,
                            scale,
                            &mut g_u_fixed,
                            &mut g_au_fixed,
                            &mut g_bu_fixed,
                        );
                        Ok(())
                    },
                )?;
            }

            // Build rho / tau via eval_coeff4_at, mirroring CPU :9319–:9322.
            let row_rho_base = row * r;
            let row_tau_base = row * r;
            for u in 1..r {
                row_rho[row_rho_base + u] = eval_coeff4_at(&g_u_fixed[u], z_obs);
                row_tau[row_tau_base + u] = eval_coeff4_at(&g_au_fixed[u], z_obs);
            }
            // r_uv via pair_from_b_family(g_b_first, ·, ·, BHW), mirroring
            // CPU :9343–:9356. Symmetric — fill both off-diagonals.
            let g_jet = SparsePrimaryCoeffJetView::new(
                1,
                h_range.as_ref(),
                w_range.as_ref(),
                g_u_fixed.as_slice(),
                g_au_fixed.as_slice(),
                g_bu_fixed.as_slice(),
                zero_family.as_slice(),
                zero_family.as_slice(),
                zero_family.as_slice(),
                zero_family.as_slice(),
                zero_family.as_slice(),
                zero_family.as_slice(),
                zero_family.as_slice(),
            );
            let row_ruv_base = row * r * r;
            for u in 0..r {
                for v in u..r {
                    let pair = g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                    let val = eval_coeff4_at(&pair, z_obs);
                    row_ruv[row_ruv_base + u * r + v] = val;
                    if u != v {
                        row_ruv[row_ruv_base + v * r + u] = val;
                    }
                }
            }
        }

        // ── Phase-4: when device-moment build was selected, dispatch the
        //    cubic-cell substrate now (all rows' cells were collected in
        //    `gpu_cells` / `gpu_branches` during the per-row loop). The
        //    returned device buffer lives on the shared CUDA context the
        //    bms_flex_row backend also uses, so the launcher consumes it
        //    without any cross-context copying.
        #[cfg(target_os = "linux")]
        let cell_moments_device: Option<cudarc::driver::CudaSlice<f64>> = if build_device_moments {
            #[cfg(debug_assertions)]
            use crate::gpu::cubic_cell::CubicCellMomentStatus;
            use crate::gpu::cubic_cell::{
                CubicCellDerivativeMomentHostView, CubicCellDerivativeMomentOutput,
                CubicCellMomentResidency, try_build_cubic_cell_derivative_moments,
            };
            // Sanity: the per-row loop must have produced exactly one
            // entry per cell index.
            if gpu_cells.len() != total_cells_us || gpu_branches.len() != total_cells_us {
                return Err(format!(
                    "bms_flex_row pack: gpu_cells.len()={} branches.len()={} mismatch total_cells={}",
                    gpu_cells.len(),
                    gpu_branches.len(),
                    total_cells_us
                ));
            }
            let view = CubicCellDerivativeMomentHostView {
                cells: &gpu_cells,
                branches: &gpu_branches,
                max_degree: crate::families::bms::gpu::row::MOMENT_STRIDE - 1,
                residency: CubicCellMomentResidency::Device,
            };
            // The GPU device-moment build is an OPTIONAL acceleration. On any
            // GPU failure — NVRTC compile error, PTX-version load rejection
            // (driver older than the toolkit's NVRTC), or kernel-launch
            // failure — and on the substrate's own host-residency downgrade or
            // an empty device buffer, fall back to filling the host moments
            // from the CPU LRU cache so the fit ALWAYS completes. GPU
            // re-engages automatically once the driver/toolkit can load the
            // kernel; this mirrors the bms_flex row-kernel CPU fallback above.
            match try_build_cubic_cell_derivative_moments(view) {
                Ok(Some(CubicCellDerivativeMomentOutput::Device {
                    d_moments,
                    status,
                    stride,
                    n_cells,
                })) => {
                    if stride != crate::families::bms::gpu::row::MOMENT_STRIDE
                        || n_cells != total_cells_us
                    {
                        return Err(format!(
                            "bms_flex_row device-moment substrate returned bad shape: \
                             stride={stride} n_cells={n_cells} expected stride={} cells={}",
                            crate::families::bms::gpu::row::MOMENT_STRIDE,
                            total_cells_us
                        ));
                    }
                    // Any non-OK status means a cell the kernel refused;
                    // the row buffer for that cell is zeroed, which is
                    // mathematically OK (zero moments → zero contribution)
                    // but indicates a classifier disagreement worth
                    // surfacing in debug builds.
                    #[cfg(debug_assertions)]
                    {
                        for (i, &s) in status.iter().enumerate() {
                            assert_eq!(
                                s,
                                CubicCellMomentStatus::Ok as u8,
                                "bms_flex_row device-moment cell {i} status={s} (kernel refused)"
                            );
                        }
                    }
                    // `status` is consumed only by the debug assert above;
                    // the runtime path keeps the device buffer alive on
                    // the owned bundle and lets the launcher feed it
                    // straight into the row kernel.
                    drop(status);
                    Some(d_moments)
                }
                degraded => {
                    match &degraded {
                        // Expected mid-flight downgrade to host residency — not
                        // a failure, so no warning.
                        Ok(Some(CubicCellDerivativeMomentOutput::Host { .. })) => {}
                        Ok(_) => log::info!(
                            "[BMS row-primary-hessian-cache] device-moment build returned no \
                             device buffer; falling back to host moments"
                        ),
                        Err(err) => log::info!(
                            "[BMS row-primary-hessian-cache] device-moment build failed: {err}; \
                             falling back to host moments (GPU re-engages once the kernel loads)"
                        ),
                    }
                    // Do the work the per-row loop skipped: re-fill
                    // `cell_moments` from the existing CPU LRU cache entries.
                    cell_moments = vec![0.0_f64; total_cells_us * moment_stride];
                    for row_idx in 0..n {
                        let start = cell_offsets[row_idx] as usize;
                        let row_cells = bundle
                            .row(row_idx, 9)
                            .expect("row cell moments presence verified above");
                        for (local_idx, entry) in row_cells.iter().enumerate() {
                            let cell_idx = start + local_idx;
                            let mom_base = cell_idx * moment_stride;
                            let src_moments: &[f64] = &entry.state.moments;
                            let copy_len = src_moments.len().min(moment_stride);
                            for k in 0..copy_len {
                                cell_moments[mom_base + k] = src_moments[k];
                            }
                        }
                    }
                    None
                }
            }
        } else {
            None
        };
        // Free the now-unneeded scratch.
        drop(gpu_cells);
        drop(gpu_branches);

        Ok(Some(
            crate::families::bms::gpu::row::BmsFlexRowKernelInputsOwned {
                n_rows: n,
                r,
                p_h,
                p_w,
                s_f: scale,
                q: row_q,
                b: row_b,
                mu_1: row_mu1,
                mu_2: row_mu2,
                z_obs: row_zobs,
                y: row_y,
                w: row_w,
                cell_offsets,
                cell_c0,
                cell_c1,
                cell_c2,
                cell_c3,
                cell_a,
                cell_aa,
                cell_r: cell_r_buf,
                cell_ar,
                cell_sbb,
                cell_sbh,
                cell_sbw,
                cell_moments,
                #[cfg(target_os = "linux")]
                cell_moments_device,
                chi_obs: row_chi,
                xi_obs: row_xi,
                rho_u: row_rho,
                tau_u: row_tau,
                r_uv: row_ruv,
            },
        ))
    }

    pub(super) fn build_row_primary_hessian_cache(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<RowPrimaryEvalCache, String> {
        if !self.effective_flex_active(block_states)? {
            return Ok(RowPrimaryEvalCache::Empty);
        }
        let n = self.y.len();
        let primary = &cache.primary;
        let r = primary.total;
        let runtime_available = runtime_available_memory_bytes();
        // Fold the live reading into the monotone capacity floor so the
        // per-shape single-cache budget is stable across workspace rebuilds;
        // the live reading still drives the global-pin OOM guard.
        let stable_capacity = observe_capacity_floor(runtime_available);
        let workspace_pinned = bms_row_primary_hessian_pinned_bytes().load(Ordering::Acquire);
        let plan = decide_row_primary_hessian_cache(
            n,
            r,
            BMS_ROW_PRIMARY_HESSIAN_EXPECTED_REUSE_PASSES,
            stable_capacity,
            runtime_available,
            workspace_pinned,
        );
        let gpu_decision =
            crate::families::bms::gpu::flex::require_row_primary_hessian_supported(n, r)?;
        // Milestone 2 (#210): when the policy says GPU, eagerly probe the
        // backend so any NVRTC compile / context init failure surfaces in
        // the cache-decision log instead of at first dispatch. Probe
        // returning `NotYetImplemented` is the expected pre-milestone-3
        // outcome and means dispatch falls through to CPU rows below —
        // the same path as today.
        if gpu_decision.use_gpu {
            match crate::families::bms::gpu::flex::BmsFlexGpuBackend::probe() {
                Ok(backend) => {
                    if log_exact_work(n) {
                        log::info!(
                            "[BMS row-primary-hessian-cache] gpu_backend_ready: {}",
                            backend.describe()
                        );
                    }
                }
                Err(crate::gpu::error::GpuError::NotYetImplemented { reason }) => {
                    log::info!(
                        "[BMS row-primary-hessian-cache] gpu_backend_pending: {reason}; \
                         falling back to CPU rows"
                    );
                }
                Err(err) => {
                    log::info!(
                        "[BMS row-primary-hessian-cache] gpu_backend_probe_failed: {err}; \
                         falling back to CPU rows"
                    );
                }
            }
        }
        if !plan.materialize {
            let tiled_budget_bytes = plan
                .global_pin_budget_bytes
                .saturating_sub(plan.workspace_pinned_bytes);
            if plan.expected_reuse_passes >= BMS_ROW_PRIMARY_HESSIAN_MIN_REUSE_PASSES
                && plan.bytes <= tiled_budget_bytes
                && n > 0
            {
                let started = std::time::Instant::now();
                let process_monitor_guard = crate::process_monitor::track_scope(format!(
                    "BMS row-primary-hessian-tiles n={n} r={r} bytes={} tile_rows={} global_budget={}",
                    plan.bytes, BMS_ROW_PRIMARY_HESSIAN_TILE_ROWS, plan.global_pin_budget_bytes
                ));
                if log_exact_work(n) {
                    log::info!(
                        "[BMS row-primary-hessian-cache] decision=tile need_bytes={} avail_bytes={} stable_capacity={} workspace_pinned={} single_cache_budget={} global_pin_budget={} tile_rows={} n={} r={} expected_reuse_passes={} reason={} gpu_policy={} gpu_selected={} gpu_reason={}",
                        plan.bytes,
                        plan.runtime_available_bytes,
                        plan.stable_capacity_bytes,
                        plan.workspace_pinned_bytes,
                        plan.single_cache_budget_bytes,
                        plan.global_pin_budget_bytes,
                        BMS_ROW_PRIMARY_HESSIAN_TILE_ROWS,
                        n,
                        r,
                        plan.expected_reuse_passes,
                        plan.reason.as_str(),
                        gpu_decision.policy.as_str(),
                        gpu_decision.use_gpu,
                        gpu_decision.reason,
                    );
                }
                let completed_rows = AtomicUsize::new(0);
                let progress_step = (n / 10).max(1);
                let mut tiles = Vec::with_capacity(n.div_ceil(BMS_ROW_PRIMARY_HESSIAN_TILE_ROWS));
                let mut row_start = 0usize;
                while row_start < n {
                    let row_end = (row_start + BMS_ROW_PRIMARY_HESSIAN_TILE_ROWS).min(n);
                    tiles.push(self.build_row_primary_hessian_tile(
                        block_states,
                        cache,
                        row_start..row_end,
                        &completed_rows,
                        progress_step,
                        started,
                    )?);
                    row_start = row_end;
                }
                if log_exact_work(n) {
                    log::info!(
                        "[BMS row-primary-hessian-cache] tiled build done n={} r={} tiles={} bytes={} elapsed={:.3}s",
                        n,
                        r,
                        tiles.len(),
                        plan.bytes,
                        started.elapsed().as_secs_f64()
                    );
                }
                drop(process_monitor_guard);
                return Ok(RowPrimaryEvalCache::Tiled(RowPrimaryEvalTiles::new(
                    n,
                    r,
                    BMS_ROW_PRIMARY_HESSIAN_TILE_ROWS,
                    tiles,
                )));
            }
            if log_exact_work(n) {
                log::info!(
                    "[BMS row-primary-hessian-cache] decision=stream need_bytes={} avail_bytes={} stable_capacity={} workspace_pinned={} single_cache_budget={} global_pin_budget={} n={} r={} expected_reuse_passes={} materialized_row_hessian_evals={} streamed_row_hessian_evals={} reason={} gpu_policy={} gpu_selected={} gpu_reason={}",
                    plan.bytes,
                    plan.runtime_available_bytes,
                    plan.stable_capacity_bytes,
                    plan.workspace_pinned_bytes,
                    plan.single_cache_budget_bytes,
                    plan.global_pin_budget_bytes,
                    n,
                    r,
                    plan.expected_reuse_passes,
                    plan.materialized_row_hessian_evals,
                    plan.streamed_row_hessian_evals,
                    plan.reason.as_str(),
                    gpu_decision.policy.as_str(),
                    gpu_decision.use_gpu,
                    gpu_decision.reason,
                );
            }
            return Ok(RowPrimaryEvalCache::Empty);
        }
        let started = std::time::Instant::now();
        let process_monitor_guard = crate::process_monitor::track_scope(format!(
            "BMS row-primary-hessian-cache n={n} r={r} bytes={} single_budget={} global_budget={}",
            plan.bytes, plan.single_cache_budget_bytes, plan.global_pin_budget_bytes
        ));
        if log_exact_work(n) {
            log::info!(
                "[BMS row-primary-hessian-cache] decision=materialize need_bytes={} avail_bytes={} stable_capacity={} workspace_pinned={} single_cache_budget={} global_pin_budget={} n={} r={} expected_reuse_passes={} materialized_row_hessian_evals={} streamed_row_hessian_evals={} reason={} gpu_policy={} gpu_selected={} gpu_reason={}",
                plan.bytes,
                plan.runtime_available_bytes,
                plan.stable_capacity_bytes,
                plan.workspace_pinned_bytes,
                plan.single_cache_budget_bytes,
                plan.global_pin_budget_bytes,
                n,
                r,
                plan.expected_reuse_passes,
                plan.materialized_row_hessian_evals,
                plan.streamed_row_hessian_evals,
                plan.reason.as_str(),
                gpu_decision.policy.as_str(),
                gpu_decision.use_gpu,
                gpu_decision.reason,
            );
        }
        // ── BMS-FLEX GPU milestone 1: when the policy says use_gpu *and* the
        //    Stage-2 device kernel preconditions are met (StandardNormal
        //    latent, row-cell-moments bundle present, optional score-warp /
        //    link-deviation runtimes present), pack the host inputs once and
        //    dispatch the row kernel. A successful launch returns the
        //    `n × r²` row-major Hessian; the CPU rayon loop below is then
        //    skipped. Any failure (`NotYetImplemented`, driver errors, or
        //    pack-time precondition mismatch) logs a one-liner and falls
        //    through to the existing CPU path, preserving production
        //    behaviour under `gpu=auto`. Under `gpu=force`, the upstream
        //    `require_row_primary_hessian_supported` would already have
        //    failed; here we still fall back on launch failure rather than
        //    panic mid-fit.
        if gpu_decision.use_gpu {
            match self.pack_bms_flex_row_kernel_inputs(block_states, cache)? {
                Some(owned) => {
                    // Phase-3: when both marginal/logslope designs expose a
                    // contiguous dense view, take the device-resident path
                    // that keeps the n×r² row Hessian + designs resident on
                    // the GPU so subsequent HVP / diagonal launches do not
                    // round-trip 626 MB through host memory.
                    #[cfg(target_os = "linux")]
                    {
                        let marginal_dense = self.marginal_design.as_dense_ref();
                        let logslope_dense = self.logslope_design.as_dense_ref();
                        if let (Some(md), Some(gd)) = (marginal_dense, logslope_dense) {
                            // Both designs must be row-major contiguous for the
                            // device upload's `[n, p]` layout to be byte-correct.
                            let md_is_rowmajor = md.is_standard_layout();
                            let gd_is_rowmajor = gd.is_standard_layout();
                            if md_is_rowmajor && gd_is_rowmajor {
                                let block_layout =
                                    crate::families::bms::gpu::row::BmsFlexBlockLayout {
                                        p_m: cache.slices.marginal.len(),
                                        p_g: cache.slices.logslope.len(),
                                        h: cache.slices.h.clone(),
                                        w: cache.slices.w.clone(),
                                        p_total: cache.slices.total,
                                    };
                                let primary_layout =
                                    crate::families::bms::gpu::row::BmsFlexPrimaryLayout {
                                        h: primary.h.clone(),
                                        w: primary.w.clone(),
                                        r: primary.total,
                                    };
                                let md_slice = md
                                    .as_slice()
                                    .expect("dense marginal_design is row-major contiguous");
                                let gd_slice = gd
                                    .as_slice()
                                    .expect("dense logslope_design is row-major contiguous");
                                match crate::families::bms::gpu::row::launch_bms_flex_row_kernel_device_resident(
                                    owned.as_borrowed(),
                                    md_slice,
                                    gd_slice,
                                    block_layout,
                                    primary_layout,
                                ) {
                                    Ok(device_state) => {
                                        if log_exact_work(n) {
                                            log::info!(
                                                "[BMS row-primary-hessian-cache] gpu_device_resident_ok rows={} r={} elapsed={:.3}s",
                                                n,
                                                r,
                                                started.elapsed().as_secs_f64()
                                            );
                                        }
                                        drop(process_monitor_guard);
                                        return Ok(RowPrimaryEvalCache::Device(device_state));
                                    }
                                    Err(err) => {
                                        log::info!(
                                            "[BMS row-primary-hessian-cache] gpu_device_resident_failed: {err}; \
                                             falling back to host-pin GPU launch"
                                        );
                                    }
                                }
                            }
                        }
                    }
                    match crate::families::bms::gpu::row::launch_bms_flex_row_kernel(
                        owned.as_borrowed(),
                    ) {
                        Ok(outputs) => {
                            if log_exact_work(n) {
                                log::info!(
                                    "[BMS row-primary-hessian-cache] gpu_launch_ok rows={} r={} elapsed={:.3}s",
                                    n,
                                    r,
                                    started.elapsed().as_secs_f64()
                                );
                            }
                            let packed_neglog = Array1::<f64>::from_vec(outputs.neglog);
                            let packed_grad =
                                Array2::<f64>::from_shape_vec((n, r), outputs.grad)
                                    .map_err(|err| format!("bms_flex_row grad shape: {err}"))?;
                            let packed_hess =
                                Array2::<f64>::from_shape_vec((n, r * r), outputs.hess)
                                    .map_err(|err| format!("bms_flex_row hess shape: {err}"))?;
                            drop(process_monitor_guard);
                            return Ok(RowPrimaryEvalCache::Host(RowPrimaryEvalPin::new(
                                packed_neglog,
                                packed_grad,
                                packed_hess,
                                plan.bytes,
                            )));
                        }
                        Err(err) => {
                            log::info!(
                                "[BMS row-primary-hessian-cache] gpu_launch_failed: {err}; \
                             falling back to CPU rows"
                            );
                        }
                    }
                }
                None => {
                    if log_exact_work(n) {
                        log::info!(
                            "[BMS row-primary-hessian-cache] gpu_unsupported_inputs; \
                             falling back to CPU rows"
                        );
                    }
                }
            }
        }
        let completed_rows = AtomicUsize::new(0);
        let progress_step = (n / 10).max(1);
        let rows = self.build_row_primary_hessian_pin(
            block_states,
            cache,
            0..n,
            &completed_rows,
            progress_step,
            started,
            plan.bytes,
        )?;
        if log_exact_work(n) {
            log::info!(
                "[BMS row-primary-hessian-cache] build done n={} r={} elapsed={:.3}s",
                n,
                r,
                started.elapsed().as_secs_f64()
            );
        }
        drop(process_monitor_guard);
        Ok(RowPrimaryEvalCache::Host(rows))
    }

    pub(crate) fn row_primary_eval_tile_bytes(rows: usize, r: usize) -> u64 {
        let floats_per_row = (r as u64)
            .saturating_mul(r as u64)
            .saturating_add(r as u64)
            .saturating_add(1);
        (rows as u64)
            .saturating_mul(floats_per_row)
            .saturating_mul(std::mem::size_of::<f64>() as u64)
    }

    pub(crate) fn build_row_primary_hessian_tile(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        rows: std::ops::Range<usize>,
        completed_rows: &AtomicUsize,
        progress_step: usize,
        started: std::time::Instant,
    ) -> Result<RowPrimaryEvalTile, String> {
        let tile_len = rows.end - rows.start;
        let bytes = Self::row_primary_eval_tile_bytes(tile_len, cache.primary.total);
        Ok(RowPrimaryEvalTile {
            row_start: rows.start,
            rows: self.build_row_primary_hessian_pin(
                block_states,
                cache,
                rows,
                completed_rows,
                progress_step,
                started,
                bytes,
            )?,
        })
    }

    pub(crate) fn build_row_primary_hessian_pin(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        rows: std::ops::Range<usize>,
        completed_rows: &AtomicUsize,
        progress_step: usize,
        started: std::time::Instant,
        bytes: u64,
    ) -> Result<RowPrimaryEvalPin, String> {
        let n = self.y.len();
        let r = cache.primary.total;
        let tile_len = rows.end - rows.start;
        let mut packed_neglog = Array1::<f64>::zeros(tile_len);
        let mut packed_grad = Array2::<f64>::zeros((tile_len, r));
        let mut packed_hess = Array2::<f64>::zeros((tile_len, r * r));
        let chunk_evals: Vec<(f64, Vec<f64>, Vec<f64>)> = rows
            .clone()
            .into_par_iter()
            .map(|row| -> Result<(f64, Vec<f64>, Vec<f64>), String> {
                let row_ctx = Self::row_ctx(cache, row);
                let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(r);
                let row_moments = cache
                    .row_cell_moments
                    .as_ref()
                    .and_then(|bundle| bundle.row(row, 9));
                let neglog = self.compute_row_analytic_flex_into_with_moments(
                    row,
                    block_states,
                    &cache.primary,
                    row_ctx,
                    row_moments,
                    cache.cell_family_forest.as_ref(),
                    true,
                    &mut scratch,
                )?;
                if log_exact_work(n) {
                    let done = completed_rows.fetch_add(1, Ordering::Relaxed) + 1;
                    if done == n || done % progress_step == 0 {
                        log::info!(
                            "[BMS row-primary-hessian-cache] progress rows={}/{} elapsed={:.3}s",
                            done,
                            n,
                            started.elapsed().as_secs_f64()
                        );
                    }
                }
                Ok((
                    neglog,
                    scratch.grad.to_vec(),
                    scratch
                        .hess
                        .as_slice()
                        .expect("hess is contiguous")
                        .to_vec(),
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        for (offset, (neglog, grad_flat, hess_flat)) in chunk_evals.into_iter().enumerate() {
            packed_neglog[offset] = neglog;
            packed_grad
                .row_mut(offset)
                .iter_mut()
                .zip(grad_flat.iter())
                .for_each(|(d, s)| *d = *s);
            packed_hess
                .row_mut(offset)
                .iter_mut()
                .zip(hess_flat.iter())
                .for_each(|(d, s)| *d = *s);
        }
        Ok(RowPrimaryEvalPin::new(
            packed_neglog,
            packed_grad,
            packed_hess,
            bytes,
        ))
    }

    /// Look up the cached per-row primary Hessian (`r × r`) materialized at
    /// the workspace β snapshot when `row_primary_hessians` is populated.
    /// Returns `None` when the cache is absent or the row index is out of
    /// range, in which case the caller must fall back to the live row
    /// kernel.
    /// Returns the cached row-primary Hessian (`r × r`) for host-resident
    /// caches. Returns `None` when the cache is absent, device-resident, or
    /// the row is out of range.
    #[inline]
    pub(super) fn cached_row_primary_hessian<'a>(
        cache: &'a BernoulliMarginalSlopeExactEvalCache,
        row: usize,
    ) -> Option<ArrayView2<'a, f64>> {
        let r = cache.primary.total;
        if let Some(pin) = cache.row_primary_hessians.host_pin() {
            return Self::cached_row_primary_hessian_from_pin(pin, row, r);
        }
        if let Some(tiles) = cache.row_primary_hessians.tiles() {
            let (tile, local_row) = tiles.tile_for_row(row)?;
            return Self::cached_row_primary_hessian_from_pin(&tile.rows, local_row, r);
        }
        None
    }

    #[inline]
    pub(crate) fn cached_row_primary_hessian_from_pin<'a>(
        pin: &'a RowPrimaryEvalPin,
        row: usize,
        r: usize,
    ) -> Option<ArrayView2<'a, f64>> {
        let hess = pin.hess();
        if row >= hess.nrows() {
            return None;
        }
        let width = r.checked_mul(r)?;
        let start = row.checked_mul(width)?;
        let end = start.checked_add(width)?;
        ArrayView2::from_shape((r, r), hess.as_slice()?.get(start..end)?).ok()
    }

    /// Returns the cached row-primary (neglog, grad_row) for host-resident
    /// caches. Returns `None` when the cache is absent, device-resident, or
    /// the row is out of range. Device-resident caches recompute the row
    /// kernel on the rare CPU-fused-gradient fallback path; the GPU
    /// dense-block kernel handles the hot path for them directly.
    #[inline]
    pub(super) fn cached_row_primary_eval<'a>(
        cache: &'a BernoulliMarginalSlopeExactEvalCache,
        row: usize,
    ) -> Option<(f64, ArrayView1<'a, f64>)> {
        let r = cache.primary.total;
        if let Some(pin) = cache.row_primary_hessians.host_pin() {
            return Self::cached_row_primary_eval_from_pin(pin, row, r);
        }
        if let Some(tiles) = cache.row_primary_hessians.tiles() {
            let (tile, local_row) = tiles.tile_for_row(row)?;
            return Self::cached_row_primary_eval_from_pin(&tile.rows, local_row, r);
        }
        None
    }

    #[inline]
    pub(crate) fn cached_row_primary_eval_from_pin<'a>(
        pin: &'a RowPrimaryEvalPin,
        row: usize,
        r: usize,
    ) -> Option<(f64, ArrayView1<'a, f64>)> {
        let neglog = pin.neglog();
        let grad = pin.grad();
        if row >= neglog.len() || row >= grad.nrows() {
            return None;
        }
        let neglog_val = neglog[row];
        let grad_row = grad.row(row);
        // Sanity: grad row must have exactly r elements.
        if grad_row.len() != r {
            return None;
        }
        Some((neglog_val, grad_row))
    }

    pub(super) fn build_exact_eval_cache(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<BernoulliMarginalSlopeExactEvalCache, String> {
        self.build_exact_eval_cache_with_order(block_states)
    }

    pub(super) fn row_primary_direction_from_flat(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(primary.total);
        self.row_primary_direction_from_flat_into(row, slices, primary, d_beta_flat, &mut out)?;
        Ok(out)
    }

    /// Allocation-free variant of [`Self::row_primary_direction_from_flat`]:
    /// fills `out` (length `primary.total`) with the primary-space projection
    /// of `d_beta_flat`. `out` is fully overwritten on success.
    pub(super) fn row_primary_direction_from_flat_into(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        d_beta_flat: &Array1<f64>,
        out: &mut Array1<f64>,
    ) -> Result<(), String> {
        if d_beta_flat.len() != slices.total {
            return Err(format!(
                "bernoulli marginal-slope d_beta length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                slices.total
            ));
        }
        out[primary.q] = self
            .marginal_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.marginal.clone()]));
        out[primary.logslope] = self
            .logslope_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));
        if let (Some(block_range), Some(primary_range)) = (slices.h.as_ref(), primary.h.as_ref()) {
            out.slice_mut(s![primary_range.start..primary_range.end])
                .assign(&d_beta_flat.slice(s![block_range.clone()]).to_owned());
        }
        if let (Some(block_range), Some(primary_range)) = (slices.w.as_ref(), primary.w.as_ref()) {
            out.slice_mut(s![primary_range.start..primary_range.end])
                .assign(&d_beta_flat.slice(s![block_range.clone()]).to_owned());
        }
        Ok(())
    }

    pub(super) fn stacked_direction_block(
        d_beta_flats: &[Array1<f64>],
        range: std::ops::Range<usize>,
    ) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((range.len(), d_beta_flats.len()));
        for (dir_idx, d_beta_flat) in d_beta_flats.iter().enumerate() {
            out.column_mut(dir_idx)
                .assign(&d_beta_flat.slice(s![range.clone()]));
        }
        out
    }

    pub(super) fn row_primary_directions_from_projected(
        local_row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        d_beta_flats: &[Array1<f64>],
        marginal_projected: &Array2<f64>,
        logslope_projected: &Array2<f64>,
    ) -> Vec<Array1<f64>> {
        let mut out = Vec::with_capacity(d_beta_flats.len());
        for (dir_idx, d_beta_flat) in d_beta_flats.iter().enumerate() {
            let mut direction = Array1::<f64>::zeros(primary.total);
            direction[primary.q] = marginal_projected[[local_row, dir_idx]];
            direction[primary.logslope] = logslope_projected[[local_row, dir_idx]];
            if let (Some(block_range), Some(primary_range)) =
                (slices.h.as_ref(), primary.h.as_ref())
            {
                direction
                    .slice_mut(s![primary_range.start..primary_range.end])
                    .assign(&d_beta_flat.slice(s![block_range.clone()]));
            }
            if let (Some(block_range), Some(primary_range)) =
                (slices.w.as_ref(), primary.w.as_ref())
            {
                direction
                    .slice_mut(s![primary_range.start..primary_range.end])
                    .assign(&d_beta_flat.slice(s![block_range.clone()]));
            }
            out.push(direction);
        }
        out
    }

    pub(super) fn batched_directional_derivative_chunk_rows(
        n: usize,
        n_dirs: usize,
    ) -> (usize, bool) {
        // CPU-only path: chunk by a fixed float-count budget so each chunk is
        // small enough to keep the per-row workspaces in L2/L3 across the
        // directional sweep. The GPU dispatch path was removed when the
        // dense-PIRLS routing helpers were retired (no live device backend
        // in this build); revisit when a runtime device backend is
        // reintroduced.
        const CPU_TARGET_CHUNK_FLOATS: usize = 1 << 17;
        let cpu_rows = (CPU_TARGET_CHUNK_FLOATS / (3 * n_dirs).max(1)).clamp(1024, n.max(1));
        (cpu_rows.min(n.max(1)), false)
    }

    pub(super) fn row_primary_psi_direction_from_map(
        &self,
        row: usize,
        block_idx: usize,
        psi_map: &crate::families::custom_family::PsiDesignMap,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
    ) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(primary.total);
        match block_idx {
            0 => {
                let x_row = psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                out[primary.q] = x_row.dot(&block_states[0].beta);
            }
            1 => {
                let x_row = psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                out[primary.logslope] = x_row.dot(&block_states[1].beta);
            }
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi direction only supports spatial marginal/logslope blocks, got block {block_idx}"
                ));
            }
        }
        Ok(out)
    }

    pub(super) fn row_primary_psi_action_on_direction_from_map(
        &self,
        row: usize,
        block_idx: usize,
        psi_map: &crate::families::custom_family::PsiDesignMap,
        slices: &BlockSlices,
        d_beta_flat: &Array1<f64>,
        primary: &PrimarySlices,
    ) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(primary.total);
        match block_idx {
            0 => {
                let x_row = psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                out[primary.q] =
                    x_row.dot(&d_beta_flat.slice(s![slices.marginal.clone()]).to_owned())
            }
            1 => {
                let x_row = psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                out[primary.logslope] =
                    x_row.dot(&d_beta_flat.slice(s![slices.logslope.clone()]).to_owned())
            }
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi action only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        }
        Ok(out)
    }

    pub(super) fn pullback_primary_vector(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        primary_vec: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let mut out = Array1::<f64>::zeros(slices.total);
        self.pullback_primary_vector_add_into(row, slices, primary, primary_vec, &mut out)?;
        Ok(out)
    }

    /// Allocation-free variant of [`Self::pullback_primary_vector`]: *adds*
    /// the pullback of `primary_vec` into the existing accumulator `out`
    /// (length `slices.total`).  `out` is **not** zeroed first; the caller
    /// must initialise it before the first call on a given accumulation.
    pub(super) fn pullback_primary_vector_add_into(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        primary_vec: &Array1<f64>,
        out: &mut Array1<f64>,
    ) -> Result<(), String> {
        {
            let mut marginal = out.slice_mut(s![slices.marginal.clone()]);
            self.marginal_design
                .axpy_row_into(row, primary_vec[primary.q], &mut marginal)?;
        }
        {
            let mut logslope = out.slice_mut(s![slices.logslope.clone()]);
            self.logslope_design.axpy_row_into(
                row,
                primary_vec[primary.logslope],
                &mut logslope,
            )?;
        }
        if let Some(primary_h) = primary.h.as_ref()
            && let Some(block_h) = slices.h.as_ref()
        {
            out.slice_mut(s![block_h.clone()]).zip_mut_with(
                &primary_vec.slice(s![primary_h.start..primary_h.end]),
                |a, &b| {
                    *a += b;
                },
            );
        }
        if let Some(primary_w) = primary.w.as_ref()
            && let Some(block_w) = slices.w.as_ref()
        {
            out.slice_mut(s![block_w.clone()]).zip_mut_with(
                &primary_vec.slice(s![primary_w.start..primary_w.end]),
                |a, &b| {
                    *a += b;
                },
            );
        }
        Ok(())
    }

    /// View-accepting twin of [`Self::pullback_primary_vector_add_into`] used by
    /// the batched multi-RHS apply, where each RHS pulls back into one column
    /// (`ArrayViewMut1`) of the `(total, n_rhs)` output. The algebra is byte for
    /// byte the same as the owned-`Array1` form; only the output handle type
    /// differs so a column view can be written without copying.
    pub(super) fn pullback_primary_vector_add_into_view(
        &self,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        primary_vec: &Array1<f64>,
        out: &mut ArrayViewMut1<'_, f64>,
    ) -> Result<(), String> {
        {
            let mut marginal = out.slice_mut(s![slices.marginal.clone()]);
            self.marginal_design
                .axpy_row_into(row, primary_vec[primary.q], &mut marginal)?;
        }
        {
            let mut logslope = out.slice_mut(s![slices.logslope.clone()]);
            self.logslope_design.axpy_row_into(
                row,
                primary_vec[primary.logslope],
                &mut logslope,
            )?;
        }
        if let Some(primary_h) = primary.h.as_ref()
            && let Some(block_h) = slices.h.as_ref()
        {
            out.slice_mut(s![block_h.clone()]).zip_mut_with(
                &primary_vec.slice(s![primary_h.start..primary_h.end]),
                |a, &b| {
                    *a += b;
                },
            );
        }
        if let Some(primary_w) = primary.w.as_ref()
            && let Some(block_w) = slices.w.as_ref()
        {
            out.slice_mut(s![block_w.clone()]).zip_mut_with(
                &primary_vec.slice(s![primary_w.start..primary_w.end]),
                |a, &b| {
                    *a += b;
                },
            );
        }
        Ok(())
    }

    pub(super) fn block_psi_row_from_map(
        &self,
        row: usize,
        block_idx: usize,
        psi_map: &crate::families::custom_family::PsiDesignMap,
        slices: &BlockSlices,
    ) -> Result<BlockPsiRow, String> {
        let (local_vec, range) = match block_idx {
            0 => (
                psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?,
                slices.marginal.clone(),
            ),
            1 => (
                psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?,
                slices.logslope.clone(),
            ),
            _ => {
                return Err(format!(
                    "bernoulli marginal-slope psi embedding only supports marginal/logslope blocks, got block {block_idx}"
                ));
            }
        };
        Ok(BlockPsiRow {
            block_idx,
            range,
            local_vec,
        })
    }

    /// Returns (neg_log_lik, gradient, Hessian) in primary coordinates.
    /// Fully analytic for both flex and non-flex paths — no AD jets.
    pub(super) fn compute_row_primary_gradient_hessian(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        // Flex path: full IFT analytic kernel.
        if self.effective_flex_active(block_states)? {
            let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
            let neglog = self.compute_row_analytic_flex_into(
                row,
                block_states,
                primary,
                row_ctx,
                None,
                true,
                &mut scratch,
            )?;
            return Ok((neglog, scratch.grad, scratch.hess));
        }
        // Rigid path: closed-form observed eta with probit frailty scaling.
        // primary.total == 2 (q at 0, g at 1), no h/w blocks.
        let marginal_eta = block_states[0].eta[row];
        let marginal = self.marginal_link_map(marginal_eta)?;
        let g = block_states[1].eta[row];
        let (neglog, grad_pair, h) = self.rigid_row_kernel_eval(row, marginal, g)?;
        let mut grad = Array1::<f64>::zeros(2);
        grad[0] = grad_pair[0];
        grad[1] = grad_pair[1];

        let mut hess = Array2::<f64>::zeros((2, 2));
        hess[[0, 0]] = h[0][0];
        hess[[0, 1]] = h[0][1];
        hess[[1, 0]] = h[1][0];
        hess[[1, 1]] = h[1][1];

        Ok((neglog, grad, hess))
    }

    pub(super) fn compute_row_primary_gradient_hessian_reusing_cache(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<(Array1<f64>, Array2<f64>), String> {
        if self.effective_flex_active(block_states)?
            && let Some(cached_hessian) = Self::cached_row_primary_hessian(cache, row)
        {
            let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
            let row_moments = cache
                .row_cell_moments
                .as_ref()
                .and_then(|bundle| bundle.row(row, 3));
            self.compute_row_analytic_flex_into_with_moments(
                row,
                block_states,
                primary,
                row_ctx,
                row_moments,
                cache.cell_family_forest.as_ref(),
                false,
                &mut scratch,
            )?;
            return Ok((scratch.grad, cached_hessian.to_owned()));
        }
        let (_, grad, hess) =
            self.compute_row_primary_gradient_hessian(row, block_states, primary, row_ctx)?;
        Ok((grad, hess))
    }

    pub(super) fn compute_row_analytic_flex_into(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        family_forest: Option<&crate::families::cell_moment_family::CellFamilyForest>,
        need_hessian: bool,
        scratch: &mut BernoulliMarginalSlopeFlexRowScratch,
    ) -> Result<f64, String> {
        self.compute_row_analytic_flex_into_with_moments(
            row,
            block_states,
            primary,
            row_ctx,
            None,
            family_forest,
            need_hessian,
            scratch,
        )
    }

    pub(super) fn compute_row_analytic_flex_into_with_moments(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        row_cell_moments: Option<&[CachedDenestedCellMoments]>,
        family_forest: Option<&crate::families::cell_moment_family::CellFamilyForest>,
        need_hessian: bool,
        scratch: &mut BernoulliMarginalSlopeFlexRowScratch,
    ) -> Result<f64, String> {
        let q = block_states[0].eta[row];
        let b = block_states[1].eta[row];
        let beta_h = self.score_beta(block_states)?;
        let beta_w = self.link_beta(block_states)?;
        self.compute_row_analytic_flex_from_parts_into(
            row,
            primary,
            q,
            b,
            beta_h,
            beta_w,
            row_ctx,
            row_cell_moments,
            family_forest,
            need_hessian,
            scratch,
        )
    }

    pub(super) fn compute_row_analytic_flex_from_parts_into(
        &self,
        row: usize,
        primary: &PrimarySlices,
        q: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        row_cell_moments: Option<&[CachedDenestedCellMoments]>,
        family_forest: Option<&crate::families::cell_moment_family::CellFamilyForest>,
        need_hessian: bool,
        scratch: &mut BernoulliMarginalSlopeFlexRowScratch,
    ) -> Result<f64, String> {
        use super::exact_kernel as exact;

        let r = primary.total;
        scratch.reset(need_hessian);
        // Reusable per-row coefficient buffers live on the scratch. Resize once
        // if the scratch was constructed for a different primary dimension; the
        // common case is `len == r` so this is a no-op.
        if scratch.coeff_u.len() != r {
            scratch.coeff_u.resize(r, [0.0; 4]);
            scratch.coeff_au.resize(r, [0.0; 4]);
            scratch.coeff_bu.resize(r, [0.0; 4]);
            scratch.g_u_fixed.resize(r, [0.0; 4]);
            scratch.g_au_fixed.resize(r, [0.0; 4]);
            scratch.g_bu_fixed.resize(r, [0.0; 4]);
            scratch.eta_u_cell.resize(r, 0.0);
            scratch.zero_family.resize(r, [0.0; 4]);
        }
        let a = row_ctx.intercept;
        let f_a = row_ctx.m_a;
        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let marginal = self.marginal_link_map(q)?;
        let inv_ma = 1.0 / f_a;
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();

        // Split-borrow the scratch into disjoint mutable references; the
        // borrow checker permits this because every field access goes through
        // `scratch.<field>` directly rather than through `&mut scratch`.
        let f_u = &mut scratch.m_u;
        let f_au = &mut scratch.m_au;
        let f_uv = &mut scratch.m_uv;
        let coeff_u = &mut scratch.coeff_u;
        let coeff_au = &mut scratch.coeff_au;
        let coeff_bu = &mut scratch.coeff_bu;
        let g_u_fixed = &mut scratch.g_u_fixed;
        let g_au_fixed = &mut scratch.g_au_fixed;
        let g_bu_fixed = &mut scratch.g_bu_fixed;
        let eta_u_cell = &mut scratch.eta_u_cell;
        let zero_family: &[[f64; 4]] = scratch.zero_family.as_slice();
        let mut f_aa = 0.0f64;

        if let Some(empirical_grid) = self.latent_measure.empirical_grid_for_training_row(row)? {
            for (&node, &weight) in empirical_grid
                .nodes
                .iter()
                .zip(empirical_grid.weights.iter())
            {
                // coeff_u is read by every per-node loop; coeff_au and coeff_bu
                // are only read inside the `if need_hessian` branches below, so
                // their per-node zero-fills are dead work in gradient-only mode.
                coeff_u.fill([0.0; 4]);
                if need_hessian {
                    coeff_au.fill([0.0; 4]);
                    coeff_bu.fill([0.0; 4]);
                }

                let obs = self.observed_denested_cell_partials_at_z(node, a, b, beta_h, beta_w)?;
                let eta = eval_coeff4_at(&obs.coeff, node);
                let eta_a = eval_coeff4_at(&obs.dc_da, node);
                let eta_aa = eval_coeff4_at(&obs.dc_daa, node);
                let phi = normal_pdf(eta);
                if need_hessian {
                    f_aa += weight * phi * (eta_aa - eta * eta_a * eta_a);
                }

                coeff_u[1] = obs.dc_db;
                if need_hessian {
                    coeff_au[1] = obs.dc_dab;
                    coeff_bu[1] = obs.dc_dbb;
                }

                if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                    Self::for_each_deviation_basis_cubic_at(
                        runtime,
                        h_range,
                        node,
                        "score-warp",
                        |_, idx, basis_span| {
                            coeff_u[idx] = scale_coeff4(
                                exact::score_basis_cell_coefficients(basis_span, b),
                                scale,
                            );
                            if need_hessian {
                                coeff_bu[idx] = scale_coeff4(
                                    exact::score_basis_cell_coefficients(basis_span, 1.0),
                                    scale,
                                );
                            }
                            Ok(())
                        },
                    )?;
                }

                if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                    let u_node = a + b * node;
                    Self::for_each_deviation_basis_cubic_at(
                        runtime,
                        w_range,
                        u_node,
                        "link-wiggle",
                        |_, idx, basis_span| {
                            coeff_u[idx] = scale_coeff4(
                                exact::link_basis_cell_coefficients(basis_span, a, b),
                                scale,
                            );
                            if need_hessian {
                                let (dc_aw_raw, dc_bw_raw) =
                                    exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                                coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                                coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                            }
                            Ok(())
                        },
                    )?;
                }

                for idx in 0..r {
                    eta_u_cell[idx] = eval_coeff4_at(&coeff_u[idx], node);
                }
                for u in 1..r {
                    f_u[u] += weight * phi * eta_u_cell[u];
                    if need_hessian {
                        let eta_au = eval_coeff4_at(&coeff_au[u], node);
                        f_au[u] += weight * phi * (eta_au - eta * eta_a * eta_u_cell[u]);
                    }
                }

                if need_hessian {
                    let coeff_jet = SparsePrimaryCoeffJetView::new(
                        1,
                        h_range,
                        w_range,
                        coeff_u.as_slice(),
                        coeff_au.as_slice(),
                        coeff_bu.as_slice(),
                        zero_family,
                        zero_family,
                        zero_family,
                        zero_family,
                        zero_family,
                        zero_family,
                        zero_family,
                    );
                    for u in 1..r {
                        for v in u..r {
                            let second_coeff = coeff_jet.pair_from_b_family(
                                coeff_jet.b_first,
                                u,
                                v,
                                COEFF_SUPPORT_BHW,
                            );
                            let eta_uv = eval_coeff4_at(&second_coeff, node);
                            let val = weight * phi * (eta_uv - eta * eta_u_cell[u] * eta_u_cell[v]);
                            f_uv[[u, v]] += val;
                            if u != v {
                                f_uv[[v, u]] += val;
                            }
                        }
                    }
                }
            }
        } else {
            // Reuse cached row moments whenever they cover the requested
            // derivative order. Degree-9 moments are exact for gradient-only
            // calls too, and avoiding a second degree-3 cell sweep preserves
            // the same calculus with less work.
            let owned_cells;
            let cached_cells: Vec<(
                exact::DenestedPartitionCell,
                std::borrow::Cow<'_, exact::CellDerivativeMomentState>,
            )> = if let Some(cached) = row_cell_moments {
                assert!(
                    !cached.is_empty(),
                    "row cell moments bundle was selected but row {row} has no cells"
                );
                cached
                    .iter()
                    .map(|entry| {
                        (
                            entry.partition_cell,
                            std::borrow::Cow::Borrowed(&entry.state),
                        )
                    })
                    .collect()
            } else if let Some(cached) = row_ctx.degree9_cells.as_ref() {
                cached
                    .iter()
                    .map(|entry| {
                        (
                            entry.partition_cell,
                            std::borrow::Cow::Borrowed(&entry.state),
                        )
                    })
                    .collect()
            } else {
                owned_cells = self.denested_partition_cells(a, b, beta_h, beta_w)?;
                owned_cells
                    .into_iter()
                    .map(|partition_cell| {
                        let degree = if need_hessian { 9 } else { 3 };
                        // #979 Stage C: certified Chebyshev family lookup
                        // first — transcendental-free when this row's leaf
                        // certified this cell combo; ladder fallback
                        // otherwise. Families are built at degree 9, which
                        // covers both the gradient (3) and Hessian (9)
                        // consumers.
                        if let Some(forest) = family_forest
                            && partition_cell.cell.left.is_finite()
                            && partition_cell.cell.right.is_finite()
                        {
                            let key = crate::families::cell_moment_family::ComboKey::new(
                                partition_cell.score_span,
                                partition_cell.link_span,
                                partition_cell.left_edge,
                                partition_cell.right_edge,
                            );
                            let mut family_moments = [0.0_f64; 10];
                            if forest
                                .moments_into(row, key, a, b, &mut family_moments)
                                .is_some()
                            {
                                let state = exact::CellDerivativeMomentState {
                                    branch: exact::branch_cell(partition_cell.cell)?,
                                    moments: exact::CellMomentVec::from_slice(&family_moments),
                                };
                                return Ok((partition_cell, std::borrow::Cow::Owned(state)));
                            }
                        }
                        self.evaluate_cell_derivative_moments_lru(partition_cell.cell, degree)
                            .map(|state| (partition_cell, std::borrow::Cow::Owned(state)))
                    })
                    .collect::<Result<Vec<_>, String>>()?
            };
            for (partition_cell, state) in cached_cells {
                // coeff_u is consumed by `cell_first_derivative_from_moments`
                // for every cell; coeff_au and coeff_bu only feed the
                // `if need_hessian` blocks below, so their per-cell zero-fills
                // — and the `coeff_au[1] = [0.0; 4]; coeff_bu[1] = [0.0; 4];`
                // pair that used to seed them explicitly — are dead work in
                // gradient-only mode.
                coeff_u.fill([0.0; 4]);
                if need_hessian {
                    coeff_au.fill([0.0; 4]);
                    coeff_bu.fill([0.0; 4]);
                }
                let cell = partition_cell.cell;
                let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
                let u_mid = a + b * z_mid;
                let state: &exact::CellDerivativeMomentState = &state;
                let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                    partition_cell.score_span,
                    partition_cell.link_span,
                    a,
                    b,
                );
                let dc_da = scale_coeff4(dc_da_raw, scale);
                let dc_db = scale_coeff4(dc_db_raw, scale);

                coeff_u[1] = dc_db;
                if need_hessian {
                    let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                        partition_cell.score_span,
                        partition_cell.link_span,
                        a,
                        b,
                    );
                    let dc_daa = scale_coeff4(dc_daa_raw, scale);
                    let dc_dab = scale_coeff4(dc_dab_raw, scale);
                    let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
                    f_aa += exact::cell_second_derivative_from_moments(
                        cell,
                        &dc_da,
                        &dc_da,
                        &dc_daa,
                        &state.moments,
                    )?;
                    coeff_au[1] = dc_dab;
                    coeff_bu[1] = dc_dbb;
                }

                if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                    Self::for_each_deviation_basis_cubic_at(
                        runtime,
                        h_range,
                        z_mid,
                        "score-warp",
                        |_, idx, basis_span| {
                            coeff_u[idx] = scale_coeff4(
                                exact::score_basis_cell_coefficients(basis_span, b),
                                scale,
                            );
                            if need_hessian {
                                coeff_bu[idx] = scale_coeff4(
                                    exact::score_basis_cell_coefficients(basis_span, 1.0),
                                    scale,
                                );
                            }
                            Ok(())
                        },
                    )?;
                }

                if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                    Self::for_each_deviation_basis_cubic_at(
                        runtime,
                        w_range,
                        u_mid,
                        "link-wiggle",
                        |_, idx, basis_span| {
                            coeff_u[idx] = scale_coeff4(
                                exact::link_basis_cell_coefficients(basis_span, a, b),
                                scale,
                            );
                            if need_hessian {
                                let (dc_aw_raw, dc_bw_raw) =
                                    exact::link_basis_cell_coefficient_partials(basis_span, a, b);
                                coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                                coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                            }
                            Ok(())
                        },
                    )?;
                }

                for u in 1..r {
                    f_u[u] +=
                        exact::cell_first_derivative_from_moments(&coeff_u[u], &state.moments)?;
                    if need_hessian {
                        f_au[u] += exact::cell_second_derivative_from_moments(
                            cell,
                            &dc_da,
                            &coeff_u[u],
                            &coeff_au[u],
                            &state.moments,
                        )?;
                    }
                }

                if need_hessian {
                    let coeff_jet = SparsePrimaryCoeffJetView::new(
                        1,
                        h_range,
                        w_range,
                        coeff_u.as_slice(),
                        coeff_au.as_slice(),
                        coeff_bu.as_slice(),
                        zero_family,
                        zero_family,
                        zero_family,
                        zero_family,
                        zero_family,
                        zero_family,
                        zero_family,
                    );
                    for u in 1..r {
                        for v in u..r {
                            let second_coeff = coeff_jet.pair_from_b_family(
                                coeff_jet.b_first,
                                u,
                                v,
                                COEFF_SUPPORT_BHW,
                            );
                            let val = exact::cell_second_derivative_from_moments(
                                cell,
                                &coeff_jet.first[u],
                                &coeff_jet.first[v],
                                &second_coeff,
                                &state.moments,
                            )?;
                            f_uv[[u, v]] += val;
                            if u != v {
                                f_uv[[v, u]] += val;
                            }
                        }
                    }
                }
            }
        }

        f_u[0] = -marginal.mu1;
        if need_hessian {
            f_uv[[0, 0]] = -marginal.mu2;
        }

        let a_u = &mut scratch.a_u;
        for u in 0..r {
            a_u[u] = -f_u[u] * inv_ma;
        }
        self.cache_row_intercept_predictor(row, a, q, b, beta_h, beta_w, a_u);
        let a_uv = &mut scratch.a_uv;
        if need_hessian {
            for u in 0..r {
                for v in u..r {
                    let val = -(f_uv[[u, v]]
                        + f_au[u] * a_u[v]
                        + f_au[v] * a_u[u]
                        + f_aa * a_u[u] * a_u[v])
                        * inv_ma;
                    a_uv[[u, v]] = val;
                    a_uv[[v, u]] = val;
                }
            }
        }

        let z_obs = self.z[row];
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let chi_obs = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa_obs = eval_coeff4_at(&obs.dc_daa, z_obs);
        let eta_val = eval_coeff4_at(&obs.coeff, z_obs);

        // `g_u_fixed` feeds `rho` (always read); `g_au_fixed` / `g_bu_fixed`
        // only feed `tau` and the symmetric-Hessian `pair_from_b_family`
        // contraction, so their per-row zero-fill and `[1]` seeding are dead
        // work in gradient-only mode.
        g_u_fixed.fill([0.0; 4]);
        g_u_fixed[1] = obs.dc_db;
        if need_hessian {
            g_au_fixed.fill([0.0; 4]);
            g_bu_fixed.fill([0.0; 4]);
            g_au_fixed[1] = obs.dc_dab;
            g_bu_fixed[1] = obs.dc_dbb;
        }
        if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                h_range,
                z_obs,
                "score-warp observed",
                |_, idx, basis_span| {
                    fill_score_basis_cell_coeff_jet(
                        idx,
                        basis_span,
                        b,
                        scale,
                        &mut *g_u_fixed,
                        &mut *g_bu_fixed,
                    );
                    Ok(())
                },
            )?;
        }
        if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                w_range,
                u_obs,
                "link-wiggle observed",
                |_, idx, basis_span| {
                    fill_link_basis_cell_coeff_gradient(
                        idx,
                        basis_span,
                        a,
                        b,
                        scale,
                        &mut *g_u_fixed,
                        &mut *g_au_fixed,
                        &mut *g_bu_fixed,
                    );
                    Ok(())
                },
            )?;
        }
        let g_jet = SparsePrimaryCoeffJetView::new(
            1,
            h_range,
            w_range,
            g_u_fixed.as_slice(),
            g_au_fixed.as_slice(),
            g_bu_fixed.as_slice(),
            zero_family,
            zero_family,
            zero_family,
            zero_family,
            zero_family,
            zero_family,
            zero_family,
        );

        // `scratch.reset(need_hessian)` at the top of this function zeroed both
        // `rho` and `tau` unconditionally, so no manual fill is needed here.
        // `tau` is consumed only by the symmetric-Hessian assembly below, so
        // its per-row eval_coeff4_at sweep is dead work in gradient-only mode.
        let rho = &mut scratch.rho;
        let tau = &mut scratch.tau;
        for u in 1..r {
            rho[u] = eval_coeff4_at(&g_jet.first[u], z_obs);
        }
        if need_hessian {
            for u in 1..r {
                tau[u] = eval_coeff4_at(&g_jet.a_first[u], z_obs);
            }
        }

        let eta_u = &mut scratch.grad;
        for u in 0..r {
            eta_u[u] = chi_obs * a_u[u] + rho[u];
        }

        let signed_margin = s_y * eta_val;
        let (log_cdf, lambda) = signed_probit_logcdf_and_mills_ratio(signed_margin);
        let neglog_val = -w_i * log_cdf;
        let d1_m = -w_i * lambda;
        let d2_m = w_i * lambda * (signed_margin + lambda);

        if need_hessian {
            let hess = &mut scratch.hess;
            hess.fill(0.0);
            for u in 0..r {
                for v in u..r {
                    let r_uv = eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW),
                        z_obs,
                    );
                    let eta_uv = chi_obs * a_uv[[u, v]]
                        + eta_aa_obs * a_u[u] * a_u[v]
                        + tau[u] * a_u[v]
                        + a_u[u] * tau[v]
                        + r_uv;
                    let val = d2_m * eta_u[u] * eta_u[v] + d1_m * s_y * eta_uv;
                    hess[[u, v]] = val;
                    hess[[v, u]] = val;
                }
            }
        }

        eta_u.mapv_inplace(|eu| d1_m * s_y * eu);
        Ok(neglog_val)
    }

    pub(super) fn primary_point_from_block_states(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &PrimarySlices,
    ) -> Result<Array1<f64>, String> {
        let mut point = Array1::<f64>::zeros(primary.total);
        point[primary.q] = block_states[0].eta[row];
        point[primary.logslope] = block_states[1].eta[row];
        if let Some(h_range) = primary.h.as_ref() {
            let score = self
                .score_block_state(block_states)?
                .ok_or_else(|| "missing score-warp beta".to_string())?;
            point
                .slice_mut(s![h_range.start..h_range.end])
                .assign(&score.beta);
        }
        if let Some(w_range) = primary.w.as_ref() {
            let beta_w = self
                .link_block_state(block_states)?
                .ok_or_else(|| "missing link deviation beta".to_string())?;
            point
                .slice_mut(s![w_range.start..w_range.end])
                .assign(&beta_w.beta);
        }
        Ok(point)
    }

    pub(super) fn primary_point_components(
        &self,
        point: &Array1<f64>,
        primary: &PrimarySlices,
    ) -> (f64, f64, Option<Array1<f64>>, Option<Array1<f64>>) {
        let beta_h = primary
            .h
            .as_ref()
            .map(|range| point.slice(s![range.start..range.end]).to_owned());
        let beta_w = primary
            .w
            .as_ref()
            .map(|range| point.slice(s![range.start..range.end]).to_owned());
        (point[primary.q], point[primary.logslope], beta_h, beta_w)
    }

    pub(super) fn observed_denested_cell_partials(
        &self,
        row: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<ObservedDenestedCellPartials, String> {
        shared_observed_denested_cell_partials(
            self.z[row],
            a,
            b,
            self.score_warp.as_ref(),
            beta_h,
            self.link_dev.as_ref(),
            beta_w,
            self.probit_frailty_scale(),
        )
    }

    pub(super) fn observed_denested_cell_partials_at_z(
        &self,
        z_value: f64,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<ObservedDenestedCellPartials, String> {
        shared_observed_denested_cell_partials(
            z_value,
            a,
            b,
            self.score_warp.as_ref(),
            beta_h,
            self.link_dev.as_ref(),
            beta_w,
            self.probit_frailty_scale(),
        )
    }

    /// Decompose an outer ψ-axis direction into `(axis, scalar)` when it is
    /// *single-axis* in primary space — nonzero only at `primary.q` (axis `0`,
    /// "q") or `primary.logslope` (axis `1`, "g"). This is the universal shape
    /// of the nonzero directions every outer-derivative consumer builds (a
    /// ψ-row dotted into one block's β, deposited at that block's primary
    /// slot). Returns `None` for the all-zero vector and for any genuinely
    /// multi-axis direction (e.g. finite-difference probes); zero directions
    /// take a separate exact zero fast path. Backs the gam#683 fast path.
    #[inline]
    pub(crate) fn single_primary_axis(dir: &Array1<f64>, primary: &PrimarySlices) -> Option<(usize, f64)> {
        if dir.len() != primary.total {
            return None;
        }
        let mut found: Option<(usize, f64)> = None;
        for (idx, &value) in dir.iter().enumerate() {
            if value == 0.0 {
                continue;
            }
            let axis = if idx == primary.q {
                0usize
            } else if idx == primary.logslope {
                1usize
            } else {
                return None;
            };
            if found.is_some() {
                return None;
            }
            found = Some((axis, value));
        }
        found
    }

    #[inline]
    pub(crate) fn primary_direction_is_zero(dir: &Array1<f64>, primary: &PrimarySlices) -> bool {
        dir.len() == primary.total && dir.iter().all(|&value| value == 0.0)
    }

    /// Lazily build the requested row's axis-projected third-derivative tensors
    /// backing the FLEX outer-derivative fast paths (gam#683), reused across
    /// every ψ-axis. `None` on the rigid path (rigid rows use
    /// `rigid_third_full`).
    ///
    /// Each row's tensors are produced by the *slow* third-order cell-walk
    /// worker (`row_primary_third_contracted_recompute_with_moments`) evaluated
    /// at the two primary-axis basis vectors `e_q`, `e_g` — so the cached
    /// values are the very contractions the per-axis path used to recompute,
    /// just computed once and reused.
    ///
    /// Two-level lazy: the outer `RayonSafeOnce` allocates a per-row slot table
    /// on first touch; each inner per-row `RayonSafeOnce` then builds that row
    /// on demand. Both inits run lock-free (`get_or_compute`), so first-touch
    /// from inside an outer Rayon row fold is safe, and the per-row build is
    /// serial (no nested `into_par_iter`). Because outer derivative passes are
    /// row-subsampled, only the consumed rows are ever built.
    pub(super) fn flex_axis_third_tensors_for_row<'a>(
        &self,
        block_states: &[ParameterBlockState],
        cache: &'a BernoulliMarginalSlopeExactEvalCache,
        row: usize,
    ) -> Result<Option<&'a FlexAxisThirdRowTensors>, String> {
        if !self.effective_flex_active(block_states)? {
            return Ok(None);
        }
        // Allocate the per-row slot table once (one inner RayonSafeOnce per
        // global row), then build only the requested row on first touch.
        let slots = cache.flex_axis_third_tensors.get_or_compute(|| {
            (0..self.y.len())
                .map(|_| crate::resource::RayonSafeOnce::new())
                .collect::<Vec<_>>()
        });
        let stored = slots[row].get_or_compute(|| -> Result<FlexAxisThirdRowTensors, String> {
            let r = cache.primary.total;
            let mut e_q = Array1::<f64>::zeros(r);
            e_q[cache.primary.q] = 1.0;
            let mut e_g = Array1::<f64>::zeros(r);
            e_g[cache.primary.logslope] = 1.0;
            let row_ctx = Self::row_ctx(cache, row);
            let t3_q = self.row_primary_third_contracted_recompute_with_moments(
                row,
                block_states,
                cache,
                row_ctx,
                &e_q,
            )?;
            let t3_g = self.row_primary_third_contracted_recompute_with_moments(
                row,
                block_states,
                cache,
                row_ctx,
                &e_g,
            )?;
            Ok(FlexAxisThirdRowTensors {
                third: [t3_q, t3_g],
            })
        });
        let tensors = stored.as_ref().map_err(|err| err.clone())?;
        Ok(Some(tensors))
    }

    /// Lazily build the requested row's axis-projected fourth-derivative
    /// tensors. Kept separate from [`Self::flex_axis_third_tensors_for_row`] so
    /// first-order outer paths do not force degree-21 fourth-order cell work.
    pub(super) fn flex_axis_fourth_tensors_for_row<'a>(
        &self,
        block_states: &[ParameterBlockState],
        cache: &'a BernoulliMarginalSlopeExactEvalCache,
        row: usize,
    ) -> Result<Option<&'a FlexAxisFourthRowTensors>, String> {
        if !self.effective_flex_active(block_states)? {
            return Ok(None);
        }
        let slots = cache.flex_axis_fourth_tensors.get_or_compute(|| {
            (0..self.y.len())
                .map(|_| crate::resource::RayonSafeOnce::new())
                .collect::<Vec<_>>()
        });
        let stored = slots[row].get_or_compute(|| -> Result<FlexAxisFourthRowTensors, String> {
            let r = cache.primary.total;
            let mut e_q = Array1::<f64>::zeros(r);
            e_q[cache.primary.q] = 1.0;
            let mut e_g = Array1::<f64>::zeros(r);
            e_g[cache.primary.logslope] = 1.0;
            let row_ctx = Self::row_ctx(cache, row);
            let t4_qq = self.row_primary_fourth_contracted_recompute_ordered(
                row,
                block_states,
                cache,
                row_ctx,
                &e_q,
                &e_q,
            )?;
            let t4_gg = self.row_primary_fourth_contracted_recompute_ordered(
                row,
                block_states,
                cache,
                row_ctx,
                &e_g,
                &e_g,
            )?;
            let t4_qg_ordered = self.row_primary_fourth_contracted_recompute_ordered(
                row,
                block_states,
                cache,
                row_ctx,
                &e_q,
                &e_g,
            )?;
            let t4_qg_swapped = self.row_primary_fourth_contracted_recompute_ordered(
                row,
                block_states,
                cache,
                row_ctx,
                &e_g,
                &e_q,
            )?;
            let mut t4_qg = t4_qg_ordered;
            t4_qg.zip_mut_with(&t4_qg_swapped, |a, &b| *a = 0.5 * (*a + b));
            Ok(FlexAxisFourthRowTensors {
                qq: t4_qq,
                qg: t4_qg,
                gg: t4_gg,
            })
        });
        let tensors = stored.as_ref().map_err(|err| err.clone())?;
        Ok(Some(tensors))
    }

    /// Third-derivative tensor contracted with direction `dir`:
    ///   out[k,l] = sum_m f_{klm} dir[m]
    /// Rigid path uses the closed-form kernel. The flexible de-nested
    /// transport path contracts the cell-moment kernel analytically.
    ///
    /// Keep this kernel row-local and single-threaded. Its production callers
    /// already parallelize the outer row reductions with Rayon (`row_iter` /
    /// chunk `into_par_iter()` folds), which avoids nested Rayon overhead for
    /// the small per-row matrices assembled here.
    pub(super) fn row_primary_third_contracted_recompute(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        // Exact zero by linearity. This matters for mixed-block ψ second-order
        // terms, where `dir_ij` is structurally zero; without this guard the
        // FLEX path would re-walk every cubic cell to produce a zero matrix.
        if Self::primary_direction_is_zero(dir, &cache.primary) {
            let r = cache.primary.total;
            return Ok(Array2::<f64>::zeros((r, r)));
        }
        // FLEX fast path (gam#683): outer-derivative consumers pass single-axis
        // directions, so reuse the per-row axis-projected tensor cache instead
        // of re-walking every cubic partition cell on every (ρ-axis, row).
        // Equal to the slow path by linearity: third_contracted(s·e_a) = s·T3[a].
        if let Some((axis, scalar)) = Self::single_primary_axis(dir, &cache.primary) {
            if let Some(tensors) = self.flex_axis_third_tensors_for_row(block_states, cache, row)? {
                let mut out = tensors.third[axis].clone();
                out.mapv_inplace(|value| value * scalar);
                return Ok(out);
            }
        }
        self.row_primary_third_contracted_recompute_with_moments(
            row,
            block_states,
            cache,
            row_ctx,
            dir,
        )
    }

    pub(super) fn row_primary_third_contracted_recompute_with_moments(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if !self.effective_flex_active(block_states)? {
            // Hit the per-cache uncontracted-tensor cache if it has already
            // been populated (typically by the first ψ-axis call in the
            // sweep, which forces the build). Direct lookup is `O(1)`; the
            // 32 ψ-axis sweep that consumes this method then pays the heavy
            // empirical-grid jet exactly once per row instead of once per
            // (row, axis) pair.
            let t = self.rigid_third_full_cached(block_states, cache, row)?;
            let m = contract_third_full(t, dir[0], dir[1]);
            let mut out = Array2::<f64>::zeros((2, 2));
            out[[0, 0]] = m[0][0];
            out[[0, 1]] = m[0][1];
            out[[1, 0]] = m[1][0];
            out[[1, 1]] = m[1][1];
            return Ok(out);
        }
        if dir.iter().all(|value| value.abs() <= 0.0) {
            return Ok(Array2::<f64>::zeros((
                cache.primary.total,
                cache.primary.total,
            )));
        }
        if !row_ctx.intercept.is_finite() || !row_ctx.m_a.is_finite() || row_ctx.m_a <= 0.0 {
            return Err(
                "non-finite flexible row context in third-order directional contraction"
                    .to_string(),
            );
        }
        use super::exact_kernel as exact;

        let primary = &cache.primary;
        let point = self.primary_point_from_block_states(row, block_states, primary)?;
        let (q, b, beta_h_owned, beta_w_owned) = self.primary_point_components(&point, primary);
        let beta_h = beta_h_owned.as_ref();
        let beta_w = beta_w_owned.as_ref();
        if let Some(grid) = self.latent_measure.empirical_grid_for_training_row(row)? {
            return self.empirical_flex_row_third_contracted_recompute(
                row, primary, q, b, beta_h, beta_w, row_ctx, dir, &grid,
            );
        }
        let a = row_ctx.intercept;
        let r = primary.total;
        let marginal = self.marginal_link_map(q)?;
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();
        let zero_family = vec![[0.0; 4]; r];

        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        let mut f_a_dir = 0.0;
        let mut f_aa_dir = 0.0;
        let mut f_u = Array1::<f64>::zeros(r);
        let mut f_au = Array1::<f64>::zeros(r);
        let mut f_au_dir = Array1::<f64>::zeros(r);
        let mut f_uv = Array2::<f64>::zeros((r, r));
        let mut f_uv_dir = Array2::<f64>::zeros((r, r));

        let owned_cells;
        let cells: &[CachedDenestedCellMoments] = if let Some(cached) =
            self.row_cell_moments_for_third_degree15(cache, row)?
        {
            cached
        } else {
            let partitions = self.denested_partition_cells(a, b, beta_h, beta_w)?;
            owned_cells = partitions
                .into_iter()
                .map(|partition_cell| {
                    exact_kernel::evaluate_cell_derivative_moments_uncached(partition_cell.cell, 15)
                        .map(|state| CachedDenestedCellMoments {
                            partition_cell,
                            state,
                        })
                })
                .collect::<Result<Vec<_>, String>>()?;
            &owned_cells
        };
        for entry in cells {
            let partition_cell = entry.partition_cell;
            let cell = partition_cell.cell;
            let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = &entry.state;

            let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let denested_third = exact::denested_cell_third_partials(partition_cell.link_span);
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let dc_db = scale_coeff4(dc_db_raw, scale);
            let dc_daa = scale_coeff4(dc_daa_raw, scale);
            let dc_dab = scale_coeff4(dc_dab_raw, scale);
            let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
            let dc_daab = scale_coeff4(denested_third.1, scale);
            let dc_dabb = scale_coeff4(denested_third.2, scale);
            let dc_dbbb = scale_coeff4(denested_third.3, scale);

            let mut coeff_u = vec![[0.0; 4]; r];
            let mut coeff_au = vec![[0.0; 4]; r];
            let mut coeff_bu = vec![[0.0; 4]; r];
            let mut coeff_aau = vec![[0.0; 4]; r];
            let mut coeff_abu = vec![[0.0; 4]; r];
            let mut coeff_bbu = vec![[0.0; 4]; r];

            coeff_u[1] = dc_db;
            coeff_au[1] = dc_dab;
            coeff_bu[1] = dc_dbb;
            coeff_aau[1] = dc_daab;
            coeff_abu[1] = dc_dabb;
            coeff_bbu[1] = dc_dbbb;

            if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    h_range,
                    z_mid,
                    "score-warp third-direction",
                    |_, idx, basis_span| {
                        fill_score_basis_cell_coeff_jet(
                            idx,
                            basis_span,
                            b,
                            scale,
                            &mut coeff_u,
                            &mut coeff_bu,
                        );
                        Ok(())
                    },
                )?;
            }

            if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    w_range,
                    u_mid,
                    "link-wiggle third-direction",
                    |_, idx, basis_span| {
                        fill_link_basis_cell_coeff_jet(
                            idx,
                            basis_span,
                            a,
                            b,
                            scale,
                            &mut coeff_u,
                            &mut coeff_au,
                            &mut coeff_bu,
                            &mut coeff_aau,
                            &mut coeff_abu,
                            &mut coeff_bbu,
                        );
                        Ok(())
                    },
                )?;
            }

            let coeff_jet = SparsePrimaryCoeffJetView::new(
                1,
                h_range,
                w_range,
                &coeff_u,
                &coeff_au,
                &coeff_bu,
                &coeff_aau,
                &coeff_abu,
                &coeff_bbu,
                &zero_family,
                &zero_family,
                &zero_family,
                &zero_family,
            );

            f_a += exact::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;

            for u in 1..r {
                f_u[u] +=
                    exact::cell_first_derivative_from_moments(&coeff_jet.first[u], &state.moments)?;
                f_au[u] += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_jet.first[u],
                    &coeff_jet.a_first[u],
                    &state.moments,
                )?;
            }
            let coeff_dir = coeff_jet.directional_family(coeff_jet.first, dir, COEFF_SUPPORT_BHW);
            let coeff_a_dir =
                coeff_jet.directional_family(coeff_jet.a_first, dir, COEFF_SUPPORT_BW);
            let coeff_aa_dir =
                coeff_jet.directional_family(coeff_jet.aa_first, dir, COEFF_SUPPORT_BW);

            f_a_dir += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &coeff_dir,
                &coeff_a_dir,
                &state.moments,
            )?;
            f_aa_dir += exact::cell_third_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &coeff_dir,
                &dc_daa,
                &coeff_a_dir,
                &coeff_a_dir,
                &coeff_aa_dir,
                &state.moments,
            )?;

            let mut coeff_u_dir = vec![[0.0; 4]; r];
            let mut coeff_au_dir = vec![[0.0; 4]; r];
            for u in 1..r {
                coeff_u_dir[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.b_first,
                    u,
                    dir,
                    COEFF_SUPPORT_BHW,
                );
                coeff_au_dir[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.ab_first,
                    u,
                    dir,
                    COEFF_SUPPORT_BW,
                );
            }

            for u in 1..r {
                f_au_dir[u] += exact::cell_third_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_jet.first[u],
                    &coeff_dir,
                    &coeff_jet.a_first[u],
                    &coeff_a_dir,
                    &coeff_u_dir[u],
                    &coeff_au_dir[u],
                    &state.moments,
                )?;
            }

            for u in 1..r {
                for v in u..r {
                    let second_coeff =
                        coeff_jet.pair_from_b_family(coeff_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                    let val = exact::cell_second_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &second_coeff,
                        &state.moments,
                    )?;
                    f_uv[[u, v]] += val;
                    if u != v {
                        f_uv[[v, u]] += val;
                    }

                    let third_coeff = coeff_jet.pair_directional_from_bb_family(
                        coeff_jet.bb_first,
                        u,
                        v,
                        dir,
                        COEFF_SUPPORT_BW,
                    );
                    let dir_val = exact::cell_third_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &coeff_dir,
                        &second_coeff,
                        &coeff_u_dir[u],
                        &coeff_u_dir[v],
                        &third_coeff,
                        &state.moments,
                    )?;
                    f_uv_dir[[u, v]] += dir_val;
                    if u != v {
                        f_uv_dir[[v, u]] += dir_val;
                    }
                }
            }
        }

        f_u[0] = -marginal.mu1;
        f_uv[[0, 0]] = -marginal.mu2;
        f_uv_dir[[0, 0]] = -dir[0] * marginal.mu3;

        let inv_f_a = 1.0 / f_a;
        let mut a_u = Array1::<f64>::zeros(r);
        for u in 0..r {
            a_u[u] = -f_u[u] * inv_f_a;
        }
        let mut a_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * a_u[v] + f_au[v] * a_u[u] + f_aa * a_u[u] * a_u[v])
                        * inv_f_a;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }
        let a_dir = a_u.dot(dir);
        let a_u_dir = a_uv.dot(dir);
        let mut a_uv_dir = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let n_dir = f_uv_dir[[u, v]]
                    + f_au_dir[u] * a_u[v]
                    + f_au[u] * a_u_dir[v]
                    + f_au_dir[v] * a_u[u]
                    + f_au[v] * a_u_dir[u]
                    + f_aa_dir * a_u[u] * a_u[v]
                    + f_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v]);
                let val = -(n_dir + f_a_dir * a_uv[[u, v]]) * inv_f_a;
                a_uv_dir[[u, v]] = val;
                a_uv_dir[[v, u]] = val;
            }
        }

        let z_obs = self.z[row];
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta_val = eval_coeff4_at(&obs.coeff, z_obs);

        let mut g_u_fixed = vec![[0.0; 4]; r];
        let mut g_au_fixed = vec![[0.0; 4]; r];
        let mut g_bu_fixed = vec![[0.0; 4]; r];
        let mut g_aau_fixed = vec![[0.0; 4]; r];
        let mut g_abu_fixed = vec![[0.0; 4]; r];
        let mut g_bbu_fixed = vec![[0.0; 4]; r];

        g_u_fixed[1] = obs.dc_db;
        g_au_fixed[1] = obs.dc_dab;
        g_bu_fixed[1] = obs.dc_dbb;
        g_aau_fixed[1] = obs.dc_daab;
        g_abu_fixed[1] = obs.dc_dabb;
        g_bbu_fixed[1] = obs.dc_dbbb;
        let scale = self.probit_frailty_scale();

        if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                h_range,
                z_obs,
                "score-warp third-direction observed",
                |_, idx, basis_span| {
                    fill_score_basis_cell_coeff_jet(
                        idx,
                        basis_span,
                        b,
                        scale,
                        &mut g_u_fixed,
                        &mut g_bu_fixed,
                    );
                    Ok(())
                },
            )?;
        }

        if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                w_range,
                u_obs,
                "link-wiggle third-direction observed",
                |_, idx, basis_span| {
                    fill_link_basis_cell_coeff_jet(
                        idx,
                        basis_span,
                        a,
                        b,
                        scale,
                        &mut g_u_fixed,
                        &mut g_au_fixed,
                        &mut g_bu_fixed,
                        &mut g_aau_fixed,
                        &mut g_abu_fixed,
                        &mut g_bbu_fixed,
                    );
                    Ok(())
                },
            )?;
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            1,
            h_range,
            w_range,
            &g_u_fixed,
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &zero_family,
            &zero_family,
            &zero_family,
            &zero_family,
        );

        let g_a = eval_coeff4_at(&obs.dc_da, z_obs);
        let g_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let g_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let mut g_u = Array1::<f64>::zeros(r);
        let mut g_au = Array1::<f64>::zeros(r);
        let mut g_aau = Array1::<f64>::zeros(r);
        let mut g_uv = Array2::<f64>::zeros((r, r));
        let mut g_auv = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u[u] = eval_coeff4_at(&g_jet.first[u], z_obs);
            g_au[u] = eval_coeff4_at(&g_jet.a_first[u], z_obs);
            g_aau[u] = eval_coeff4_at(&g_jet.aa_first[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let second_coeff = g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                let val = eval_coeff4_at(&second_coeff, z_obs);
                g_uv[[u, v]] = val;
                g_uv[[v, u]] = val;

                let third_coeff = g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_BW);
                let third_val = eval_coeff4_at(&third_coeff, z_obs);
                g_auv[[u, v]] = third_val;
                g_auv[[v, u]] = third_val;
            }
        }

        let mut g_u_dir_fixed = vec![[0.0; 4]; r];
        let mut g_au_dir_fixed = vec![[0.0; 4]; r];
        let g_dir_fixed = g_jet.directional_family(g_jet.first, dir, COEFF_SUPPORT_BHW);
        let g_a_dir_fixed = g_jet.directional_family(g_jet.a_first, dir, COEFF_SUPPORT_BW);
        let g_aa_dir_fixed = g_jet.directional_family(g_jet.aa_first, dir, COEFF_SUPPORT_BW);
        let g_dir = eval_coeff4_at(&g_dir_fixed, z_obs);
        let g_a_dir = eval_coeff4_at(&g_a_dir_fixed, z_obs);
        let g_aa_dir = eval_coeff4_at(&g_aa_dir_fixed, z_obs);

        for u in 1..r {
            g_u_dir_fixed[u] =
                g_jet.param_directional_from_b_family(g_jet.b_first, u, dir, COEFF_SUPPORT_BHW);
            g_au_dir_fixed[u] =
                g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir, COEFF_SUPPORT_BW);
        }

        let mut g_u_dir = Array1::<f64>::zeros(r);
        let mut g_uv_dir = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u_dir[u] = eval_coeff4_at(&g_u_dir_fixed[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let third_coeff = g_jet.pair_directional_from_bb_family(
                    g_jet.bb_first,
                    u,
                    v,
                    dir,
                    COEFF_SUPPORT_BW,
                );
                let val = eval_coeff4_at(&third_coeff, z_obs);
                g_uv_dir[[u, v]] = val;
                g_uv_dir[[v, u]] = val;
            }
        }

        let eta_u = g_a * &a_u + &g_u;
        let mut eta_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = g_a * a_uv[[u, v]]
                    + g_aa * a_u[u] * a_u[v]
                    + g_au[u] * a_u[v]
                    + g_au[v] * a_u[u]
                    + g_uv[[u, v]];
                eta_uv[[u, v]] = val;
                eta_uv[[v, u]] = val;
            }
        }
        let eta_dir = g_a * a_dir + g_dir;
        let eta_u_dir = eta_uv.dot(dir);
        let dg_a_dir = g_aa * a_dir + g_a_dir;
        let dg_aa_dir = g_aaa * a_dir + g_aa_dir;
        let mut dg_au_dir = Array1::<f64>::zeros(r);
        let mut dg_uv_dir = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            dg_au_dir[u] = g_aau[u] * a_dir + eval_coeff4_at(&g_au_dir_fixed[u], z_obs);
        }
        for u in 0..r {
            for v in u..r {
                let val = g_auv[[u, v]] * a_dir + g_uv_dir[[u, v]];
                dg_uv_dir[[u, v]] = val;
                dg_uv_dir[[v, u]] = val;
            }
        }

        let mut eta_uv_dir = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = dg_a_dir * a_uv[[u, v]]
                    + g_a * a_uv_dir[[u, v]]
                    + dg_aa_dir * a_u[u] * a_u[v]
                    + g_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v])
                    + dg_au_dir[u] * a_u[v]
                    + g_au[u] * a_u_dir[v]
                    + dg_au_dir[v] * a_u[u]
                    + g_au[v] * a_u_dir[u]
                    + dg_uv_dir[[u, v]];
                eta_uv_dir[[u, v]] = val;
                eta_uv_dir[[v, u]] = val;
            }
        }

        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let m = s_y * eta_val;
        let (k1, k2, k3, _) = signed_probit_neglog_derivatives_up_to_fourth(m, w_i)?;
        let u1 = s_y * k1;
        let u3 = s_y * k3;

        let mut out = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = u3 * eta_u[u] * eta_u[v] * eta_dir
                    + k2 * (eta_uv[[u, v]] * eta_dir
                        + eta_u[u] * eta_u_dir[v]
                        + eta_u[v] * eta_u_dir[u])
                    + u1 * eta_uv_dir[[u, v]];
                out[[u, v]] = val;
                out[[v, u]] = val;
            }
        }
        Ok(out)
    }

    #[inline]
    pub(super) fn coeff4_eval_adjoint(z: f64, scalar_adjoint: f64) -> [f64; 4] {
        let z2 = z * z;
        [
            scalar_adjoint,
            scalar_adjoint * z,
            scalar_adjoint * z2,
            scalar_adjoint * z2 * z,
        ]
    }

    #[inline]
    pub(super) fn add_coeff4_adjoint(target: &mut [f64; 4], source: &[f64; 4]) {
        for idx in 0..4 {
            target[idx] += source[idx];
        }
    }

    #[inline]
    pub(super) fn add_eval_directional_family_adjoint(
        jet: &SparsePrimaryCoeffJetView<'_>,
        family: &[[f64; 4]],
        support: CoeffSupport,
        z: f64,
        scalar_adjoint: f64,
        direction_adjoint: &mut [f64],
    ) {
        let coeff_adjoint = Self::coeff4_eval_adjoint(z, scalar_adjoint);
        jet.add_directional_family_adjoint(family, &coeff_adjoint, support, direction_adjoint);
    }

    #[inline]
    pub(super) fn add_eval_param_directional_adjoint(
        jet: &SparsePrimaryCoeffJetView<'_>,
        family: &[[f64; 4]],
        param: usize,
        support: CoeffSupport,
        z: f64,
        scalar_adjoint: f64,
        direction_adjoint: &mut [f64],
    ) {
        let coeff_adjoint = Self::coeff4_eval_adjoint(z, scalar_adjoint);
        jet.add_param_directional_from_b_family_adjoint(
            family,
            param,
            &coeff_adjoint,
            support,
            direction_adjoint,
        );
    }

    #[inline]
    pub(super) fn add_eval_pair_directional_adjoint(
        jet: &SparsePrimaryCoeffJetView<'_>,
        family: &[[f64; 4]],
        u: usize,
        v: usize,
        support: CoeffSupport,
        z: f64,
        scalar_adjoint: f64,
        direction_adjoint: &mut [f64],
    ) {
        let coeff_adjoint = Self::coeff4_eval_adjoint(z, scalar_adjoint);
        jet.add_pair_directional_from_bb_family_adjoint(
            family,
            u,
            v,
            &coeff_adjoint,
            support,
            direction_adjoint,
        );
    }

    pub(super) fn add_cell_second_direction_adjoint(
        cell: exact_kernel::DenestedCubicCell,
        first_r: &[f64; 4],
        moments: &[f64],
        scalar_adjoint: f64,
        first_s_adjoint: &mut [f64; 4],
        second_adjoint: &mut [f64; 4],
    ) -> Result<(), String> {
        if moments.len() < 10 {
            return Err(format!(
                "insufficient reduced moments for second-derivative adjoint: need 10, have {}",
                moments.len()
            ));
        }
        let scale = scalar_adjoint / std::f64::consts::TAU;
        let eta = [cell.c0, cell.c1, cell.c2, cell.c3];
        for k in 0..4 {
            second_adjoint[k] += scale * moments[k];
        }
        for s_idx in 0..4 {
            let mut eta_r_moment = 0.0;
            for (eta_idx, &eta_value) in eta.iter().enumerate() {
                for (r_idx, &r_value) in first_r.iter().enumerate() {
                    eta_r_moment += eta_value * r_value * moments[eta_idx + r_idx + s_idx];
                }
            }
            first_s_adjoint[s_idx] -= scale * eta_r_moment;
        }
        Ok(())
    }

    pub(super) fn add_cell_third_direction_adjoint(
        cell: exact_kernel::DenestedCubicCell,
        first_r: &[f64; 4],
        first_s: &[f64; 4],
        second_rs: &[f64; 4],
        moments: &[f64],
        scalar_adjoint: f64,
        first_t_adjoint: &mut [f64; 4],
        second_rt_adjoint: &mut [f64; 4],
        second_st_adjoint: &mut [f64; 4],
        third_rst_adjoint: &mut [f64; 4],
    ) -> Result<(), String> {
        if moments.len() < 16 {
            return Err(format!(
                "insufficient reduced moments for third-derivative adjoint: need 16, have {}",
                moments.len()
            ));
        }
        let scale = scalar_adjoint / std::f64::consts::TAU;
        let eta = [cell.c0, cell.c1, cell.c2, cell.c3];
        let mut eta_sq_minus_one = [0.0; 7];
        for (i, &eta_i) in eta.iter().enumerate() {
            for (j, &eta_j) in eta.iter().enumerate() {
                eta_sq_minus_one[i + j] += eta_i * eta_j;
            }
        }
        eta_sq_minus_one[0] -= 1.0;

        for k in 0..4 {
            third_rst_adjoint[k] += scale * moments[k];
        }
        for coeff_idx in 0..4 {
            let mut eta_s_moment = 0.0;
            let mut eta_r_moment = 0.0;
            for (eta_idx, &eta_value) in eta.iter().enumerate() {
                for basis_idx in 0..4 {
                    eta_s_moment +=
                        eta_value * first_s[basis_idx] * moments[eta_idx + coeff_idx + basis_idx];
                    eta_r_moment +=
                        eta_value * first_r[basis_idx] * moments[eta_idx + coeff_idx + basis_idx];
                }
            }
            second_rt_adjoint[coeff_idx] -= scale * eta_s_moment;
            second_st_adjoint[coeff_idx] -= scale * eta_r_moment;
        }
        for t_idx in 0..4 {
            let mut linear_second = 0.0;
            for (eta_idx, &eta_value) in eta.iter().enumerate() {
                for (second_idx, &second_value) in second_rs.iter().enumerate() {
                    linear_second +=
                        eta_value * second_value * moments[eta_idx + second_idx + t_idx];
                }
            }
            let mut cubic_product = 0.0;
            for (eta_idx, &eta_value) in eta_sq_minus_one.iter().enumerate() {
                for (r_idx, &r_value) in first_r.iter().enumerate() {
                    for (s_idx, &s_value) in first_s.iter().enumerate() {
                        cubic_product += eta_value
                            * r_value
                            * s_value
                            * moments[eta_idx + r_idx + s_idx + t_idx];
                    }
                }
            }
            first_t_adjoint[t_idx] += scale * (cubic_product - linear_second);
        }
        Ok(())
    }

    pub(super) fn row_primary_third_trace_gradient_with_moments(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        gram: &[f64],
    ) -> Result<Array1<f64>, String> {
        let primary = &cache.primary;
        let r = primary.total;
        if gram.len() != r * r {
            return Err(format!(
                "bernoulli marginal-slope row trace gram length {} != {}",
                gram.len(),
                r * r
            ));
        }

        if !self.effective_flex_active(block_states)? {
            let tensor = self.rigid_third_full_cached(block_states, cache, row)?;
            let mut grad = Array1::<f64>::zeros(r);
            for a_idx in 0..2 {
                for b_idx in 0..2 {
                    let weight = gram[a_idx * r + b_idx];
                    for dir_idx in 0..2 {
                        grad[dir_idx] += weight * tensor[a_idx][b_idx][dir_idx];
                    }
                }
            }
            return Ok(grad);
        }
        if !row_ctx.intercept.is_finite() || !row_ctx.m_a.is_finite() || row_ctx.m_a <= 0.0 {
            return Err(
                "non-finite flexible row context in third-order trace-gradient contraction"
                    .to_string(),
            );
        }

        let point = self.primary_point_from_block_states(row, block_states, primary)?;
        let (q, b, beta_h_owned, beta_w_owned) = self.primary_point_components(&point, primary);
        let beta_h = beta_h_owned.as_ref();
        let beta_w = beta_w_owned.as_ref();
        if let Some(grid) = self.latent_measure.empirical_grid_for_training_row(row)? {
            let mut grad = Array1::<f64>::zeros(r);
            for dir_idx in 0..r {
                let mut basis = Array1::<f64>::zeros(r);
                basis[dir_idx] = 1.0;
                let third = self.empirical_flex_row_third_contracted_recompute(
                    row, primary, q, b, beta_h, beta_w, row_ctx, &basis, &grid,
                )?;
                grad[dir_idx] = Self::row_primary_trace_contract(&third, gram);
            }
            return Ok(grad);
        }

        use super::exact_kernel as exact;

        let a = row_ctx.intercept;
        let marginal = self.marginal_link_map(q)?;
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();
        let zero_family = vec![[0.0; 4]; r];

        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        let mut f_u = Array1::<f64>::zeros(r);
        let mut f_au = Array1::<f64>::zeros(r);
        let mut f_uv = Array2::<f64>::zeros((r, r));

        let owned_cells;
        let cells: &[CachedDenestedCellMoments] = if let Some(cached) =
            self.row_cell_moments_for_third_degree15(cache, row)?
        {
            cached
        } else {
            let partitions = self.denested_partition_cells(a, b, beta_h, beta_w)?;
            owned_cells = partitions
                .into_iter()
                .map(|partition_cell| {
                    exact_kernel::evaluate_cell_derivative_moments_uncached(partition_cell.cell, 15)
                        .map(|state| CachedDenestedCellMoments {
                            partition_cell,
                            state,
                        })
                })
                .collect::<Result<Vec<_>, String>>()?;
            &owned_cells
        };

        for entry in cells {
            let partition_cell = entry.partition_cell;
            let cell = partition_cell.cell;
            let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = &entry.state;

            let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let denested_third = exact::denested_cell_third_partials(partition_cell.link_span);
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let dc_db = scale_coeff4(dc_db_raw, scale);
            let dc_daa = scale_coeff4(dc_daa_raw, scale);
            let dc_dab = scale_coeff4(dc_dab_raw, scale);
            let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
            let dc_daab = scale_coeff4(denested_third.1, scale);
            let dc_dabb = scale_coeff4(denested_third.2, scale);
            let dc_dbbb = scale_coeff4(denested_third.3, scale);

            let mut coeff_u = vec![[0.0; 4]; r];
            let mut coeff_au = vec![[0.0; 4]; r];
            let mut coeff_bu = vec![[0.0; 4]; r];
            let mut coeff_aau = vec![[0.0; 4]; r];
            let mut coeff_abu = vec![[0.0; 4]; r];
            let mut coeff_bbu = vec![[0.0; 4]; r];

            coeff_u[1] = dc_db;
            coeff_au[1] = dc_dab;
            coeff_bu[1] = dc_dbb;
            coeff_aau[1] = dc_daab;
            coeff_abu[1] = dc_dabb;
            coeff_bbu[1] = dc_dbbb;

            if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    h_range,
                    z_mid,
                    "score-warp trace-gradient base",
                    |_, idx, basis_span| {
                        fill_score_basis_cell_coeff_jet(
                            idx,
                            basis_span,
                            b,
                            scale,
                            &mut coeff_u,
                            &mut coeff_bu,
                        );
                        Ok(())
                    },
                )?;
            }

            if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    w_range,
                    u_mid,
                    "link-wiggle trace-gradient base",
                    |_, idx, basis_span| {
                        fill_link_basis_cell_coeff_jet(
                            idx,
                            basis_span,
                            a,
                            b,
                            scale,
                            &mut coeff_u,
                            &mut coeff_au,
                            &mut coeff_bu,
                            &mut coeff_aau,
                            &mut coeff_abu,
                            &mut coeff_bbu,
                        );
                        Ok(())
                    },
                )?;
            }

            let coeff_jet = SparsePrimaryCoeffJetView::new(
                1,
                h_range,
                w_range,
                &coeff_u,
                &coeff_au,
                &coeff_bu,
                &coeff_aau,
                &coeff_abu,
                &coeff_bbu,
                &zero_family,
                &zero_family,
                &zero_family,
                &zero_family,
            );

            f_a += exact::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;
            for u in 1..r {
                f_u[u] +=
                    exact::cell_first_derivative_from_moments(&coeff_jet.first[u], &state.moments)?;
                f_au[u] += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_jet.first[u],
                    &coeff_jet.a_first[u],
                    &state.moments,
                )?;
            }
            for u in 1..r {
                for v in u..r {
                    let second_coeff =
                        coeff_jet.pair_from_b_family(coeff_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                    let val = exact::cell_second_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &second_coeff,
                        &state.moments,
                    )?;
                    f_uv[[u, v]] += val;
                    if u != v {
                        f_uv[[v, u]] += val;
                    }
                }
            }
        }

        f_u[0] = -marginal.mu1;
        f_uv[[0, 0]] = -marginal.mu2;

        let inv_f_a = 1.0 / f_a;
        let mut a_u = Array1::<f64>::zeros(r);
        for u in 0..r {
            a_u[u] = -f_u[u] * inv_f_a;
        }
        let mut a_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * a_u[v] + f_au[v] * a_u[u] + f_aa * a_u[u] * a_u[v])
                        * inv_f_a;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }

        let z_obs = self.z[row];
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta_val = eval_coeff4_at(&obs.coeff, z_obs);

        let mut g_u_fixed = vec![[0.0; 4]; r];
        let mut g_au_fixed = vec![[0.0; 4]; r];
        let mut g_bu_fixed = vec![[0.0; 4]; r];
        let mut g_aau_fixed = vec![[0.0; 4]; r];
        let mut g_abu_fixed = vec![[0.0; 4]; r];
        let mut g_bbu_fixed = vec![[0.0; 4]; r];

        g_u_fixed[1] = obs.dc_db;
        g_au_fixed[1] = obs.dc_dab;
        g_bu_fixed[1] = obs.dc_dbb;
        g_aau_fixed[1] = obs.dc_daab;
        g_abu_fixed[1] = obs.dc_dabb;
        g_bbu_fixed[1] = obs.dc_dbbb;

        if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                h_range,
                z_obs,
                "score-warp trace-gradient observed",
                |_, idx, basis_span| {
                    fill_score_basis_cell_coeff_jet(
                        idx,
                        basis_span,
                        b,
                        scale,
                        &mut g_u_fixed,
                        &mut g_bu_fixed,
                    );
                    Ok(())
                },
            )?;
        }

        if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                w_range,
                u_obs,
                "link-wiggle trace-gradient observed",
                |_, idx, basis_span| {
                    fill_link_basis_cell_coeff_jet(
                        idx,
                        basis_span,
                        a,
                        b,
                        scale,
                        &mut g_u_fixed,
                        &mut g_au_fixed,
                        &mut g_bu_fixed,
                        &mut g_aau_fixed,
                        &mut g_abu_fixed,
                        &mut g_bbu_fixed,
                    );
                    Ok(())
                },
            )?;
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            1,
            h_range,
            w_range,
            &g_u_fixed,
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &zero_family,
            &zero_family,
            &zero_family,
            &zero_family,
        );

        let g_a = eval_coeff4_at(&obs.dc_da, z_obs);
        let g_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let g_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let mut g_u = Array1::<f64>::zeros(r);
        let mut g_au = Array1::<f64>::zeros(r);
        let mut g_aau = Array1::<f64>::zeros(r);
        let mut g_uv = Array2::<f64>::zeros((r, r));
        let mut g_auv = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u[u] = eval_coeff4_at(&g_jet.first[u], z_obs);
            g_au[u] = eval_coeff4_at(&g_jet.a_first[u], z_obs);
            g_aau[u] = eval_coeff4_at(&g_jet.aa_first[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let second_coeff = g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                let val = eval_coeff4_at(&second_coeff, z_obs);
                g_uv[[u, v]] = val;
                g_uv[[v, u]] = val;

                let third_coeff = g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_BW);
                let third_val = eval_coeff4_at(&third_coeff, z_obs);
                g_auv[[u, v]] = third_val;
                g_auv[[v, u]] = third_val;
            }
        }

        let eta_u = g_a * &a_u + &g_u;
        let mut eta_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = g_a * a_uv[[u, v]]
                    + g_aa * a_u[u] * a_u[v]
                    + g_au[u] * a_u[v]
                    + g_au[v] * a_u[u]
                    + g_uv[[u, v]];
                eta_uv[[u, v]] = val;
                eta_uv[[v, u]] = val;
            }
        }

        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let m = s_y * eta_val;
        let (k1, k2, k3, _) = signed_probit_neglog_derivatives_up_to_fourth(m, w_i)?;
        let u1 = s_y * k1;
        let u3 = s_y * k3;

        let mut direction_adjoint = vec![0.0; r];
        let mut adj_eta_dir = 0.0;
        let mut adj_eta_u_dir = vec![0.0; r];
        let mut adj_a_u_dir = vec![0.0; r];
        let mut adj_a_uv_dir = Array2::<f64>::zeros((r, r));
        let mut adj_dg_a_dir = 0.0;
        let mut adj_dg_aa_dir = 0.0;
        let mut adj_dg_au_dir = vec![0.0; r];
        let mut adj_a_dir = 0.0;

        for u in 0..r {
            for v in u..r {
                let weight = if u == v {
                    gram[u * r + v]
                } else {
                    gram[u * r + v] + gram[v * r + u]
                };
                if weight == 0.0 {
                    continue;
                }
                adj_eta_dir += weight * (u3 * eta_u[u] * eta_u[v] + k2 * eta_uv[[u, v]]);
                adj_eta_u_dir[v] += weight * k2 * eta_u[u];
                adj_eta_u_dir[u] += weight * k2 * eta_u[v];

                let adj_eta_uv_dir = weight * u1;
                adj_dg_a_dir += adj_eta_uv_dir * a_uv[[u, v]];
                adj_a_uv_dir[[u, v]] += adj_eta_uv_dir * g_a;
                adj_dg_aa_dir += adj_eta_uv_dir * a_u[u] * a_u[v];
                adj_a_u_dir[u] += adj_eta_uv_dir * g_aa * a_u[v];
                adj_a_u_dir[v] += adj_eta_uv_dir * g_aa * a_u[u];
                adj_dg_au_dir[u] += adj_eta_uv_dir * a_u[v];
                adj_a_u_dir[v] += adj_eta_uv_dir * g_au[u];
                adj_dg_au_dir[v] += adj_eta_uv_dir * a_u[u];
                adj_a_u_dir[u] += adj_eta_uv_dir * g_au[v];

                adj_a_dir += adj_eta_uv_dir * g_auv[[u, v]];
                Self::add_eval_pair_directional_adjoint(
                    &g_jet,
                    g_jet.bb_first,
                    u,
                    v,
                    COEFF_SUPPORT_BW,
                    z_obs,
                    adj_eta_uv_dir,
                    &mut direction_adjoint,
                );
            }
        }

        for u in 0..r {
            let adj = adj_dg_au_dir[u];
            if adj != 0.0 {
                adj_a_dir += adj * g_aau[u];
                Self::add_eval_param_directional_adjoint(
                    &g_jet,
                    g_jet.ab_first,
                    u,
                    COEFF_SUPPORT_BW,
                    z_obs,
                    adj,
                    &mut direction_adjoint,
                );
            }
        }
        adj_a_dir += adj_eta_dir * g_a + adj_dg_a_dir * g_aa + adj_dg_aa_dir * g_aaa;
        Self::add_eval_directional_family_adjoint(
            &g_jet,
            g_jet.first,
            COEFF_SUPPORT_BHW,
            z_obs,
            adj_eta_dir,
            &mut direction_adjoint,
        );
        Self::add_eval_directional_family_adjoint(
            &g_jet,
            g_jet.a_first,
            COEFF_SUPPORT_BW,
            z_obs,
            adj_dg_a_dir,
            &mut direction_adjoint,
        );
        Self::add_eval_directional_family_adjoint(
            &g_jet,
            g_jet.aa_first,
            COEFF_SUPPORT_BW,
            z_obs,
            adj_dg_aa_dir,
            &mut direction_adjoint,
        );

        for u in 0..r {
            let adj = adj_eta_u_dir[u];
            if adj != 0.0 {
                for dir_idx in 0..r {
                    direction_adjoint[dir_idx] += adj * eta_uv[[u, dir_idx]];
                }
            }
        }

        let mut adj_f_a_dir = 0.0;
        let mut adj_f_aa_dir = 0.0;
        let mut adj_f_au_dir = vec![0.0; r];
        let mut adj_f_uv_dir = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let adj = adj_a_uv_dir[[u, v]];
                if adj == 0.0 {
                    continue;
                }
                let adj_n = -adj * inv_f_a;
                adj_f_uv_dir[[u, v]] += adj_n;
                adj_f_au_dir[u] += adj_n * a_u[v];
                adj_a_u_dir[v] += adj_n * f_au[u];
                adj_f_au_dir[v] += adj_n * a_u[u];
                adj_a_u_dir[u] += adj_n * f_au[v];
                adj_f_aa_dir += adj_n * a_u[u] * a_u[v];
                adj_a_u_dir[u] += adj_n * f_aa * a_u[v];
                adj_a_u_dir[v] += adj_n * f_aa * a_u[u];
                adj_f_a_dir += adj_n * a_uv[[u, v]];
            }
        }
        direction_adjoint[0] -= adj_f_uv_dir[[0, 0]] * marginal.mu3;

        for u in 0..r {
            let adj = adj_a_u_dir[u];
            if adj != 0.0 {
                for dir_idx in 0..r {
                    direction_adjoint[dir_idx] += adj * a_uv[[u, dir_idx]];
                }
            }
        }
        if adj_a_dir != 0.0 {
            for dir_idx in 0..r {
                direction_adjoint[dir_idx] += adj_a_dir * a_u[dir_idx];
            }
        }

        for entry in cells {
            let partition_cell = entry.partition_cell;
            let cell = partition_cell.cell;
            let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = &entry.state;

            let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let denested_third = exact::denested_cell_third_partials(partition_cell.link_span);
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let dc_db = scale_coeff4(dc_db_raw, scale);
            let dc_daa = scale_coeff4(dc_daa_raw, scale);
            let dc_dab = scale_coeff4(dc_dab_raw, scale);
            let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
            let dc_daab = scale_coeff4(denested_third.1, scale);
            let dc_dabb = scale_coeff4(denested_third.2, scale);
            let dc_dbbb = scale_coeff4(denested_third.3, scale);

            let mut coeff_u = vec![[0.0; 4]; r];
            let mut coeff_au = vec![[0.0; 4]; r];
            let mut coeff_bu = vec![[0.0; 4]; r];
            let mut coeff_aau = vec![[0.0; 4]; r];
            let mut coeff_abu = vec![[0.0; 4]; r];
            let mut coeff_bbu = vec![[0.0; 4]; r];

            coeff_u[1] = dc_db;
            coeff_au[1] = dc_dab;
            coeff_bu[1] = dc_dbb;
            coeff_aau[1] = dc_daab;
            coeff_abu[1] = dc_dabb;
            coeff_bbu[1] = dc_dbbb;

            if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    h_range,
                    z_mid,
                    "score-warp trace-gradient adjoint",
                    |_, idx, basis_span| {
                        fill_score_basis_cell_coeff_jet(
                            idx,
                            basis_span,
                            b,
                            scale,
                            &mut coeff_u,
                            &mut coeff_bu,
                        );
                        Ok(())
                    },
                )?;
            }

            if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    w_range,
                    u_mid,
                    "link-wiggle trace-gradient adjoint",
                    |_, idx, basis_span| {
                        fill_link_basis_cell_coeff_jet(
                            idx,
                            basis_span,
                            a,
                            b,
                            scale,
                            &mut coeff_u,
                            &mut coeff_au,
                            &mut coeff_bu,
                            &mut coeff_aau,
                            &mut coeff_abu,
                            &mut coeff_bbu,
                        );
                        Ok(())
                    },
                )?;
            }

            let coeff_jet = SparsePrimaryCoeffJetView::new(
                1,
                h_range,
                w_range,
                &coeff_u,
                &coeff_au,
                &coeff_bu,
                &coeff_aau,
                &coeff_abu,
                &coeff_bbu,
                &zero_family,
                &zero_family,
                &zero_family,
                &zero_family,
            );

            let mut coeff_dir_adj = [0.0; 4];
            let mut coeff_a_dir_adj = [0.0; 4];
            let mut coeff_aa_dir_adj = [0.0; 4];
            let mut coeff_u_dir_adj = vec![[0.0; 4]; r];
            let mut coeff_au_dir_adj = vec![[0.0; 4]; r];

            if adj_f_a_dir != 0.0 {
                Self::add_cell_second_direction_adjoint(
                    cell,
                    &dc_da,
                    &state.moments,
                    adj_f_a_dir,
                    &mut coeff_dir_adj,
                    &mut coeff_a_dir_adj,
                )?;
            }
            if adj_f_aa_dir != 0.0 {
                let mut a_rt_adj = [0.0; 4];
                let mut a_st_adj = [0.0; 4];
                Self::add_cell_third_direction_adjoint(
                    cell,
                    &dc_da,
                    &dc_da,
                    &dc_daa,
                    &state.moments,
                    adj_f_aa_dir,
                    &mut coeff_dir_adj,
                    &mut a_rt_adj,
                    &mut a_st_adj,
                    &mut coeff_aa_dir_adj,
                )?;
                Self::add_coeff4_adjoint(&mut coeff_a_dir_adj, &a_rt_adj);
                Self::add_coeff4_adjoint(&mut coeff_a_dir_adj, &a_st_adj);
            }
            for u in 1..r {
                let adj = adj_f_au_dir[u];
                if adj == 0.0 {
                    continue;
                }
                Self::add_cell_third_direction_adjoint(
                    cell,
                    &dc_da,
                    &coeff_jet.first[u],
                    &coeff_jet.a_first[u],
                    &state.moments,
                    adj,
                    &mut coeff_dir_adj,
                    &mut coeff_a_dir_adj,
                    &mut coeff_u_dir_adj[u],
                    &mut coeff_au_dir_adj[u],
                )?;
            }
            for u in 1..r {
                for v in u..r {
                    let adj = adj_f_uv_dir[[u, v]];
                    if adj == 0.0 {
                        continue;
                    }
                    let second_coeff =
                        coeff_jet.pair_from_b_family(coeff_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                    let mut u_dir_adj = [0.0; 4];
                    let mut v_dir_adj = [0.0; 4];
                    let mut third_coeff_adj = [0.0; 4];
                    Self::add_cell_third_direction_adjoint(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &second_coeff,
                        &state.moments,
                        adj,
                        &mut coeff_dir_adj,
                        &mut u_dir_adj,
                        &mut v_dir_adj,
                        &mut third_coeff_adj,
                    )?;
                    Self::add_coeff4_adjoint(&mut coeff_u_dir_adj[u], &u_dir_adj);
                    Self::add_coeff4_adjoint(&mut coeff_u_dir_adj[v], &v_dir_adj);
                    coeff_jet.add_pair_directional_from_bb_family_adjoint(
                        coeff_jet.bb_first,
                        u,
                        v,
                        &third_coeff_adj,
                        COEFF_SUPPORT_BW,
                        &mut direction_adjoint,
                    );
                }
            }

            coeff_jet.add_directional_family_adjoint(
                coeff_jet.first,
                &coeff_dir_adj,
                COEFF_SUPPORT_BHW,
                &mut direction_adjoint,
            );
            coeff_jet.add_directional_family_adjoint(
                coeff_jet.a_first,
                &coeff_a_dir_adj,
                COEFF_SUPPORT_BW,
                &mut direction_adjoint,
            );
            coeff_jet.add_directional_family_adjoint(
                coeff_jet.aa_first,
                &coeff_aa_dir_adj,
                COEFF_SUPPORT_BW,
                &mut direction_adjoint,
            );
            for u in 1..r {
                coeff_jet.add_param_directional_from_b_family_adjoint(
                    coeff_jet.b_first,
                    u,
                    &coeff_u_dir_adj[u],
                    COEFF_SUPPORT_BHW,
                    &mut direction_adjoint,
                );
                coeff_jet.add_param_directional_from_b_family_adjoint(
                    coeff_jet.ab_first,
                    u,
                    &coeff_au_dir_adj[u],
                    COEFF_SUPPORT_BW,
                    &mut direction_adjoint,
                );
            }
        }

        Ok(Array1::from_vec(direction_adjoint))
    }

    /// Accumulate the per-cell primary third-order Newton-assembly moments for
    /// a batched directional contraction.
    ///
    /// Shared inner `for entry in cells { … }` loop of
    /// [`Self::row_primary_third_trace_many_with_moments`] and
    /// [`Self::row_primary_third_contracted_many_with_moments`]: both walk the
    /// same denested cells, build the same sparse coefficient jet, and add the
    /// same first/second/third cell-moment derivatives into the `f_*`
    /// accumulators. The two call sites previously inlined byte-identical
    /// copies differing only in the diagnostic `score_label` / `link_label`
    /// strings threaded into the deviation-basis iterators.
    pub(crate) fn accumulate_primary_third_cell_moments(
        cells: &[CachedDenestedCellMoments],
        a: f64,
        b: f64,
        scale: f64,
        r: usize,
        h_range: Option<&std::ops::Range<usize>>,
        w_range: Option<&std::ops::Range<usize>>,
        score_runtime: Option<&DeviationRuntime>,
        link_runtime: Option<&DeviationRuntime>,
        zero_family: &[[f64; 4]],
        row_dirs: &[Array1<f64>],
        score_label: &str,
        link_label: &str,
        f_a: &mut f64,
        f_aa: &mut f64,
        f_u: &mut Array1<f64>,
        f_au: &mut Array1<f64>,
        f_uv: &mut Array2<f64>,
        f_a_dir: &mut [f64],
        f_aa_dir: &mut [f64],
        f_au_dir: &mut [f64],
        f_uv_dir: &mut [f64],
    ) -> Result<(), String> {
        use super::exact_kernel as exact;

        for entry in cells {
            let partition_cell = entry.partition_cell;
            let cell = partition_cell.cell;
            let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = &entry.state;

            let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let denested_third = exact::denested_cell_third_partials(partition_cell.link_span);
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let dc_db = scale_coeff4(dc_db_raw, scale);
            let dc_daa = scale_coeff4(dc_daa_raw, scale);
            let dc_dab = scale_coeff4(dc_dab_raw, scale);
            let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
            let dc_daab = scale_coeff4(denested_third.1, scale);
            let dc_dabb = scale_coeff4(denested_third.2, scale);
            let dc_dbbb = scale_coeff4(denested_third.3, scale);

            let mut coeff_u = vec![[0.0; 4]; r];
            let mut coeff_au = vec![[0.0; 4]; r];
            let mut coeff_bu = vec![[0.0; 4]; r];
            let mut coeff_aau = vec![[0.0; 4]; r];
            let mut coeff_abu = vec![[0.0; 4]; r];
            let mut coeff_bbu = vec![[0.0; 4]; r];

            coeff_u[1] = dc_db;
            coeff_au[1] = dc_dab;
            coeff_bu[1] = dc_dbb;
            coeff_aau[1] = dc_daab;
            coeff_abu[1] = dc_dabb;
            coeff_bbu[1] = dc_dbbb;

            if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    h_range,
                    z_mid,
                    score_label,
                    |_, idx, basis_span| {
                        fill_score_basis_cell_coeff_jet(
                            idx,
                            basis_span,
                            b,
                            scale,
                            &mut coeff_u,
                            &mut coeff_bu,
                        );
                        Ok(())
                    },
                )?;
            }

            if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    w_range,
                    u_mid,
                    link_label,
                    |_, idx, basis_span| {
                        fill_link_basis_cell_coeff_jet(
                            idx,
                            basis_span,
                            a,
                            b,
                            scale,
                            &mut coeff_u,
                            &mut coeff_au,
                            &mut coeff_bu,
                            &mut coeff_aau,
                            &mut coeff_abu,
                            &mut coeff_bbu,
                        );
                        Ok(())
                    },
                )?;
            }

            let coeff_jet = SparsePrimaryCoeffJetView::new(
                1,
                h_range,
                w_range,
                &coeff_u,
                &coeff_au,
                &coeff_bu,
                &coeff_aau,
                &coeff_abu,
                &coeff_bbu,
                zero_family,
                zero_family,
                zero_family,
                zero_family,
            );

            *f_a += exact::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            *f_aa += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;
            for u in 1..r {
                f_u[u] +=
                    exact::cell_first_derivative_from_moments(&coeff_jet.first[u], &state.moments)?;
                f_au[u] += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_jet.first[u],
                    &coeff_jet.a_first[u],
                    &state.moments,
                )?;
            }
            for u in 1..r {
                for v in u..r {
                    let second_coeff =
                        coeff_jet.pair_from_b_family(coeff_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                    let val = exact::cell_second_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &second_coeff,
                        &state.moments,
                    )?;
                    f_uv[[u, v]] += val;
                    if u != v {
                        f_uv[[v, u]] += val;
                    }
                }
            }

            for (dir_idx, dir) in row_dirs.iter().enumerate() {
                let coeff_dir =
                    coeff_jet.directional_family(coeff_jet.first, dir, COEFF_SUPPORT_BHW);
                let coeff_a_dir =
                    coeff_jet.directional_family(coeff_jet.a_first, dir, COEFF_SUPPORT_BW);
                let coeff_aa_dir =
                    coeff_jet.directional_family(coeff_jet.aa_first, dir, COEFF_SUPPORT_BW);

                f_a_dir[dir_idx] += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_dir,
                    &coeff_a_dir,
                    &state.moments,
                )?;
                f_aa_dir[dir_idx] += exact::cell_third_derivative_from_moments(
                    cell,
                    &dc_da,
                    &dc_da,
                    &coeff_dir,
                    &dc_daa,
                    &coeff_a_dir,
                    &coeff_a_dir,
                    &coeff_aa_dir,
                    &state.moments,
                )?;

                let mut coeff_u_dir = vec![[0.0; 4]; r];
                let mut coeff_au_dir = vec![[0.0; 4]; r];
                for u in 1..r {
                    coeff_u_dir[u] = coeff_jet.param_directional_from_b_family(
                        coeff_jet.b_first,
                        u,
                        dir,
                        COEFF_SUPPORT_BHW,
                    );
                    coeff_au_dir[u] = coeff_jet.param_directional_from_b_family(
                        coeff_jet.ab_first,
                        u,
                        dir,
                        COEFF_SUPPORT_BW,
                    );
                }

                for u in 1..r {
                    f_au_dir[dir_idx * r + u] += exact::cell_third_derivative_from_moments(
                        cell,
                        &dc_da,
                        &coeff_jet.first[u],
                        &coeff_dir,
                        &coeff_jet.a_first[u],
                        &coeff_a_dir,
                        &coeff_u_dir[u],
                        &coeff_au_dir[u],
                        &state.moments,
                    )?;
                }

                let dir_base = dir_idx * r * r;
                for u in 1..r {
                    for v in u..r {
                        let second_coeff = coeff_jet.pair_from_b_family(
                            coeff_jet.b_first,
                            u,
                            v,
                            COEFF_SUPPORT_BHW,
                        );
                        let third_coeff = coeff_jet.pair_directional_from_bb_family(
                            coeff_jet.bb_first,
                            u,
                            v,
                            dir,
                            COEFF_SUPPORT_BW,
                        );
                        let dir_val = exact::cell_third_derivative_from_moments(
                            cell,
                            &coeff_jet.first[u],
                            &coeff_jet.first[v],
                            &coeff_dir,
                            &second_coeff,
                            &coeff_u_dir[u],
                            &coeff_u_dir[v],
                            &third_coeff,
                            &state.moments,
                        )?;
                        f_uv_dir[dir_base + u * r + v] += dir_val;
                        if u != v {
                            f_uv_dir[dir_base + v * r + u] += dir_val;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub(super) fn row_primary_third_trace_many_with_moments(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        row_dirs: &[Array1<f64>],
        gram: &[f64],
    ) -> Result<Vec<f64>, String> {
        let primary = &cache.primary;
        let r = primary.total;
        if row_dirs.is_empty() {
            return Ok(Vec::new());
        }
        if gram.len() != r * r {
            return Err(format!(
                "bernoulli marginal-slope row trace gram length {} != {}",
                gram.len(),
                r * r
            ));
        }
        if let Some((idx, dir)) = row_dirs.iter().enumerate().find(|(_, dir)| dir.len() != r) {
            return Err(format!(
                "bernoulli marginal-slope row trace direction {idx} length {} != {r}",
                dir.len()
            ));
        }

        if row_dirs.len() > 1 {
            let trace_gradient = self.row_primary_third_trace_gradient_with_moments(
                row,
                block_states,
                cache,
                row_ctx,
                gram,
            )?;
            let traces = row_dirs
                .iter()
                .map(|dir| trace_gradient.dot(dir))
                .collect::<Vec<_>>();
            return Ok(traces);
        }

        if !self.effective_flex_active(block_states)? {
            let t = self.rigid_third_full_cached(block_states, cache, row)?;
            let mut traces = vec![0.0; row_dirs.len()];
            for (dir_idx, dir) in row_dirs.iter().enumerate() {
                let m = contract_third_full(t, dir[0], dir[1]);
                traces[dir_idx] = m[0][0] * gram[0]
                    + m[0][1] * gram[1]
                    + m[1][0] * gram[r]
                    + m[1][1] * gram[r + 1];
            }
            return Ok(traces);
        }
        if !row_ctx.intercept.is_finite() || !row_ctx.m_a.is_finite() || row_ctx.m_a <= 0.0 {
            return Err(
                "non-finite flexible row context in batched third-order trace contraction"
                    .to_string(),
            );
        }
        let point = self.primary_point_from_block_states(row, block_states, primary)?;
        let (q, b, beta_h_owned, beta_w_owned) = self.primary_point_components(&point, primary);
        let beta_h = beta_h_owned.as_ref();
        let beta_w = beta_w_owned.as_ref();
        let a = row_ctx.intercept;

        if let Some(grid) = self.latent_measure.empirical_grid_for_training_row(row)? {
            let mut traces = vec![0.0; row_dirs.len()];
            for (dir_idx, dir) in row_dirs.iter().enumerate() {
                let third = self.empirical_flex_row_third_contracted_recompute(
                    row, primary, q, b, beta_h, beta_w, row_ctx, dir, &grid,
                )?;
                traces[dir_idx] = Self::row_primary_trace_contract(&third, gram);
            }
            return Ok(traces);
        }

        let marginal = self.marginal_link_map(q)?;
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();
        let zero_family = vec![[0.0; 4]; r];
        let n_dirs = row_dirs.len();

        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        let mut f_u = Array1::<f64>::zeros(r);
        let mut f_au = Array1::<f64>::zeros(r);
        let mut f_uv = Array2::<f64>::zeros((r, r));
        let mut f_a_dir = vec![0.0; n_dirs];
        let mut f_aa_dir = vec![0.0; n_dirs];
        let mut f_au_dir = vec![0.0; n_dirs * r];
        let mut f_uv_dir = vec![0.0; n_dirs * r * r];

        let owned_cells;
        let cells: &[CachedDenestedCellMoments] = if let Some(cached) =
            self.row_cell_moments_for_third_degree15(cache, row)?
        {
            cached
        } else {
            let partitions = self.denested_partition_cells(a, b, beta_h, beta_w)?;
            owned_cells = partitions
                .into_iter()
                .map(|partition_cell| {
                    exact_kernel::evaluate_cell_derivative_moments_uncached(partition_cell.cell, 15)
                        .map(|state| CachedDenestedCellMoments {
                            partition_cell,
                            state,
                        })
                })
                .collect::<Result<Vec<_>, String>>()?;
            &owned_cells
        };

        Self::accumulate_primary_third_cell_moments(
            cells,
            a,
            b,
            scale,
            r,
            h_range,
            w_range,
            score_runtime,
            link_runtime,
            &zero_family,
            row_dirs,
            "score-warp batched third-trace direction",
            "link-wiggle batched third-trace direction",
            &mut f_a,
            &mut f_aa,
            &mut f_u,
            &mut f_au,
            &mut f_uv,
            &mut f_a_dir,
            &mut f_aa_dir,
            &mut f_au_dir,
            &mut f_uv_dir,
        )?;

        f_u[0] = -marginal.mu1;
        f_uv[[0, 0]] = -marginal.mu2;

        let inv_f_a = 1.0 / f_a;
        let mut a_u = Array1::<f64>::zeros(r);
        for u in 0..r {
            a_u[u] = -f_u[u] * inv_f_a;
        }
        let mut a_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * a_u[v] + f_au[v] * a_u[u] + f_aa * a_u[u] * a_u[v])
                        * inv_f_a;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }

        let z_obs = self.z[row];
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta_val = eval_coeff4_at(&obs.coeff, z_obs);

        let mut g_u_fixed = vec![[0.0; 4]; r];
        let mut g_au_fixed = vec![[0.0; 4]; r];
        let mut g_bu_fixed = vec![[0.0; 4]; r];
        let mut g_aau_fixed = vec![[0.0; 4]; r];
        let mut g_abu_fixed = vec![[0.0; 4]; r];
        let mut g_bbu_fixed = vec![[0.0; 4]; r];

        g_u_fixed[1] = obs.dc_db;
        g_au_fixed[1] = obs.dc_dab;
        g_bu_fixed[1] = obs.dc_dbb;
        g_aau_fixed[1] = obs.dc_daab;
        g_abu_fixed[1] = obs.dc_dabb;
        g_bbu_fixed[1] = obs.dc_dbbb;

        if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                h_range,
                z_obs,
                "score-warp batched third-trace observed",
                |_, idx, basis_span| {
                    fill_score_basis_cell_coeff_jet(
                        idx,
                        basis_span,
                        b,
                        scale,
                        &mut g_u_fixed,
                        &mut g_bu_fixed,
                    );
                    Ok(())
                },
            )?;
        }

        if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                w_range,
                u_obs,
                "link-wiggle batched third-trace observed",
                |_, idx, basis_span| {
                    fill_link_basis_cell_coeff_jet(
                        idx,
                        basis_span,
                        a,
                        b,
                        scale,
                        &mut g_u_fixed,
                        &mut g_au_fixed,
                        &mut g_bu_fixed,
                        &mut g_aau_fixed,
                        &mut g_abu_fixed,
                        &mut g_bbu_fixed,
                    );
                    Ok(())
                },
            )?;
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            1,
            h_range,
            w_range,
            &g_u_fixed,
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &zero_family,
            &zero_family,
            &zero_family,
            &zero_family,
        );

        let g_a = eval_coeff4_at(&obs.dc_da, z_obs);
        let g_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let g_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let mut g_u = Array1::<f64>::zeros(r);
        let mut g_au = Array1::<f64>::zeros(r);
        let mut g_aau = Array1::<f64>::zeros(r);
        let mut g_uv = Array2::<f64>::zeros((r, r));
        let mut g_auv = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u[u] = eval_coeff4_at(&g_jet.first[u], z_obs);
            g_au[u] = eval_coeff4_at(&g_jet.a_first[u], z_obs);
            g_aau[u] = eval_coeff4_at(&g_jet.aa_first[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let second_coeff = g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                let val = eval_coeff4_at(&second_coeff, z_obs);
                g_uv[[u, v]] = val;
                g_uv[[v, u]] = val;

                let third_coeff = g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_BW);
                let third_val = eval_coeff4_at(&third_coeff, z_obs);
                g_auv[[u, v]] = third_val;
                g_auv[[v, u]] = third_val;
            }
        }

        let eta_u = g_a * &a_u + &g_u;
        let mut eta_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = g_a * a_uv[[u, v]]
                    + g_aa * a_u[u] * a_u[v]
                    + g_au[u] * a_u[v]
                    + g_au[v] * a_u[u]
                    + g_uv[[u, v]];
                eta_uv[[u, v]] = val;
                eta_uv[[v, u]] = val;
            }
        }

        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let m = s_y * eta_val;
        let (k1, k2, k3, _) = signed_probit_neglog_derivatives_up_to_fourth(m, w_i)?;
        let u1 = s_y * k1;
        let u3 = s_y * k3;
        let mut traces = vec![0.0; n_dirs];

        for (dir_idx, dir) in row_dirs.iter().enumerate() {
            let dir_base = dir_idx * r * r;
            f_uv_dir[dir_base] = -dir[0] * marginal.mu3;

            let a_dir = a_u.dot(dir);
            let a_u_dir = a_uv.dot(dir);
            let g_dir_fixed = g_jet.directional_family(g_jet.first, dir, COEFF_SUPPORT_BHW);
            let g_a_dir_fixed = g_jet.directional_family(g_jet.a_first, dir, COEFF_SUPPORT_BW);
            let g_aa_dir_fixed = g_jet.directional_family(g_jet.aa_first, dir, COEFF_SUPPORT_BW);
            let g_dir = eval_coeff4_at(&g_dir_fixed, z_obs);
            let g_a_dir = eval_coeff4_at(&g_a_dir_fixed, z_obs);
            let g_aa_dir = eval_coeff4_at(&g_aa_dir_fixed, z_obs);
            let eta_dir = g_a * a_dir + g_dir;
            let eta_u_dir = eta_uv.dot(dir);
            let dg_a_dir = g_aa * a_dir + g_a_dir;
            let dg_aa_dir = g_aaa * a_dir + g_aa_dir;
            let mut dg_au_dir = Array1::<f64>::zeros(r);
            for u in 0..r {
                let coeff =
                    g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir, COEFF_SUPPORT_BW);
                dg_au_dir[u] = g_aau[u] * a_dir + eval_coeff4_at(&coeff, z_obs);
            }

            let mut trace = 0.0;
            for u in 0..r {
                for v in u..r {
                    let fuvd = f_uv_dir[dir_base + u * r + v];
                    let n_dir = fuvd
                        + f_au_dir[dir_idx * r + u] * a_u[v]
                        + f_au[u] * a_u_dir[v]
                        + f_au_dir[dir_idx * r + v] * a_u[u]
                        + f_au[v] * a_u_dir[u]
                        + f_aa_dir[dir_idx] * a_u[u] * a_u[v]
                        + f_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v]);
                    let a_uv_dir = -(n_dir + f_a_dir[dir_idx] * a_uv[[u, v]]) * inv_f_a;
                    let third_coeff = g_jet.pair_directional_from_bb_family(
                        g_jet.bb_first,
                        u,
                        v,
                        dir,
                        COEFF_SUPPORT_BW,
                    );
                    let dg_uv_dir = g_auv[[u, v]] * a_dir + eval_coeff4_at(&third_coeff, z_obs);
                    let eta_uv_dir = dg_a_dir * a_uv[[u, v]]
                        + g_a * a_uv_dir
                        + dg_aa_dir * a_u[u] * a_u[v]
                        + g_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v])
                        + dg_au_dir[u] * a_u[v]
                        + g_au[u] * a_u_dir[v]
                        + dg_au_dir[v] * a_u[u]
                        + g_au[v] * a_u_dir[u]
                        + dg_uv_dir;
                    let val = u3 * eta_u[u] * eta_u[v] * eta_dir
                        + k2 * (eta_uv[[u, v]] * eta_dir
                            + eta_u[u] * eta_u_dir[v]
                            + eta_u[v] * eta_u_dir[u])
                        + u1 * eta_uv_dir;
                    if u == v {
                        trace += val * gram[u * r + v];
                    } else {
                        trace += val * (gram[u * r + v] + gram[v * r + u]);
                    }
                }
            }
            traces[dir_idx] = trace;
        }

        Ok(traces)
    }

    /// Fourth-derivative tensor contracted with two directions dir_u, dir_v:
    ///   out[k,l] = sum_{m,n} f_{klmn} dir_u[m] dir_v[n]
    /// Rigid path uses the closed-form kernel. The flexible de-nested
    /// transport path contracts the cell-moment kernel analytically.
    pub(super) fn row_primary_fourth_contracted_recompute_ordered(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let expected_dir_len = if flex_active { cache.primary.total } else { 2 };
        if dir_u.len() != expected_dir_len || dir_v.len() != expected_dir_len {
            return Err(format!(
                "bernoulli fourth contracted row {row}: direction lengths ({},{}) != {expected_dir_len}",
                dir_u.len(),
                dir_v.len()
            ));
        }

        // Keep this row-local helper serial. All expensive callers parallelize
        // across independent rows/chunks so Rayon workers do not nest inside
        // the high-allocation per-row cell-kernel transport below.
        if !flex_active {
            // Hit the per-cache uncontracted-tensor cache (lazy-built on
            // first access) so the heavy per-row jet runs at most once per
            // row across all `(rank²+rank)/2` ψ-axis pairs, instead of
            // once per (row, pair). The outer-Hessian sweep is the dominant
            // consumer of this method.
            let t = self.rigid_fourth_full_cached(block_states, cache, row)?;
            let f = contract_fourth_full(t, dir_u[0], dir_u[1], dir_v[0], dir_v[1]);
            let mut out = Array2::<f64>::zeros((2, 2));
            out[[0, 0]] = f[0][0];
            out[[0, 1]] = f[0][1];
            out[[1, 0]] = f[1][0];
            out[[1, 1]] = f[1][1];
            return Ok(out);
        }
        if dir_u.iter().all(|value| *value == 0.0) || dir_v.iter().all(|value| *value == 0.0) {
            return Ok(Array2::<f64>::zeros((
                cache.primary.total,
                cache.primary.total,
            )));
        }
        if !row_ctx.intercept.is_finite() || !row_ctx.m_a.is_finite() || row_ctx.m_a <= 0.0 {
            return Err(
                "non-finite flexible row context in fourth-order directional contraction"
                    .to_string(),
            );
        }
        use super::exact_kernel as exact;

        let primary = &cache.primary;
        let point = self.primary_point_from_block_states(row, block_states, primary)?;
        let (q, b, beta_h_owned, beta_w_owned) = self.primary_point_components(&point, primary);
        let beta_h = beta_h_owned.as_ref();
        let beta_w = beta_w_owned.as_ref();
        if let Some(grid) = self.latent_measure.empirical_grid_for_training_row(row)? {
            return self.empirical_flex_row_fourth_contracted_recompute(
                row, primary, q, b, beta_h, beta_w, row_ctx, dir_u, dir_v, &grid,
            );
        }
        let a = row_ctx.intercept;
        let r = primary.total;
        let marginal = self.marginal_link_map(q)?;
        let h_range = primary.h.as_ref();
        let w_range = primary.w.as_ref();
        let score_runtime = self.score_warp.as_ref();
        let link_runtime = self.link_dev.as_ref();
        let scale = self.probit_frailty_scale();

        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        let mut f_u = Array1::<f64>::zeros(r);
        let mut f_au = Array1::<f64>::zeros(r);
        let mut f_uv = Array2::<f64>::zeros((r, r));

        let mut f_a_u = 0.0;
        let mut f_aa_u = 0.0;
        let mut f_au_u = Array1::<f64>::zeros(r);
        let mut f_uv_u = Array2::<f64>::zeros((r, r));

        let mut f_a_v = 0.0;
        let mut f_aa_v = 0.0;
        let mut f_au_v = Array1::<f64>::zeros(r);
        let mut f_uv_v = Array2::<f64>::zeros((r, r));

        let mut f_a_uv = 0.0;
        let mut f_aa_uv = 0.0;
        let mut f_au_uv = Array1::<f64>::zeros(r);
        let mut f_uv_uv = Array2::<f64>::zeros((r, r));

        let owned_cells;
        let cells: &[CachedDenestedCellMoments] = if let Some(cached) = self
            .existing_bundle_for_degree(cache, 21)?
            .and_then(|bundle| bundle.row(row, 21))
        {
            cached
        } else {
            let partitions = self.denested_partition_cells(a, b, beta_h, beta_w)?;
            owned_cells = partitions
                .into_iter()
                .map(|partition_cell| {
                    exact::evaluate_cell_derivative_moments_uncached(partition_cell.cell, 21).map(
                        |state| CachedDenestedCellMoments {
                            partition_cell,
                            state,
                        },
                    )
                })
                .collect::<Result<Vec<_>, String>>()?;
            &owned_cells
        };
        for entry in cells {
            let partition_cell = entry.partition_cell;
            let cell = partition_cell.cell;
            let z_mid = exact::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = &entry.state;

            let (dc_da_raw, dc_db_raw) = exact::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) = exact::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                b,
            );
            let denested_third = exact::denested_cell_third_partials(partition_cell.link_span);
            let dc_da = scale_coeff4(dc_da_raw, scale);
            let dc_db = scale_coeff4(dc_db_raw, scale);
            let dc_daa = scale_coeff4(dc_daa_raw, scale);
            let dc_dab = scale_coeff4(dc_dab_raw, scale);
            let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
            let dc_daab = scale_coeff4(denested_third.1, scale);
            let dc_dabb = scale_coeff4(denested_third.2, scale);
            let dc_dbbb = scale_coeff4(denested_third.3, scale);

            let mut coeff_u = vec![[0.0; 4]; r];
            let mut coeff_au = vec![[0.0; 4]; r];
            let mut coeff_bu = vec![[0.0; 4]; r];
            let mut coeff_aau = vec![[0.0; 4]; r];
            let mut coeff_abu = vec![[0.0; 4]; r];
            let mut coeff_bbu = vec![[0.0; 4]; r];
            let mut coeff_aaau = vec![[0.0; 4]; r];
            let mut coeff_aabu = vec![[0.0; 4]; r];
            let mut coeff_abbu = vec![[0.0; 4]; r];
            let mut coeff_bbbu = vec![[0.0; 4]; r];

            coeff_u[1] = dc_db;
            coeff_au[1] = dc_dab;
            coeff_bu[1] = dc_dbb;
            coeff_aau[1] = dc_daab;
            coeff_abu[1] = dc_dabb;
            coeff_bbu[1] = dc_dbbb;

            if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    h_range,
                    z_mid,
                    "score-warp fourth-direction",
                    |_, idx, basis_span| {
                        fill_score_basis_cell_coeff_jet(
                            idx,
                            basis_span,
                            b,
                            scale,
                            &mut coeff_u,
                            &mut coeff_bu,
                        );
                        Ok(())
                    },
                )?;
            }

            if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
                Self::for_each_deviation_basis_cubic_at(
                    runtime,
                    w_range,
                    u_mid,
                    "link-wiggle fourth-direction",
                    |_, idx, basis_span| {
                        fill_link_basis_cell_coeff_jet(
                            idx,
                            basis_span,
                            a,
                            b,
                            scale,
                            &mut coeff_u,
                            &mut coeff_au,
                            &mut coeff_bu,
                            &mut coeff_aau,
                            &mut coeff_abu,
                            &mut coeff_bbu,
                        );
                        let (dc_aaaw, dc_aabw, dc_abbw, dc_bbbw) =
                            exact::link_basis_cell_third_partials(basis_span);
                        coeff_aaau[idx] = scale_coeff4(dc_aaaw, scale);
                        coeff_aabu[idx] = scale_coeff4(dc_aabw, scale);
                        coeff_abbu[idx] = scale_coeff4(dc_abbw, scale);
                        coeff_bbbu[idx] = scale_coeff4(dc_bbbw, scale);
                        Ok(())
                    },
                )?;
            }

            let coeff_jet = SparsePrimaryCoeffJetView::new(
                1,
                h_range,
                w_range,
                &coeff_u,
                &coeff_au,
                &coeff_bu,
                &coeff_aau,
                &coeff_abu,
                &coeff_bbu,
                &coeff_aaau,
                &coeff_aabu,
                &coeff_abbu,
                &coeff_bbbu,
            );

            f_a += exact::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;

            for u in 1..r {
                f_u[u] +=
                    exact::cell_first_derivative_from_moments(&coeff_jet.first[u], &state.moments)?;
                f_au[u] += exact::cell_second_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_jet.first[u],
                    &coeff_jet.a_first[u],
                    &state.moments,
                )?;
            }
            let coeff_dir_u =
                coeff_jet.directional_family(coeff_jet.first, dir_u, COEFF_SUPPORT_BHW);
            let coeff_dir_v =
                coeff_jet.directional_family(coeff_jet.first, dir_v, COEFF_SUPPORT_BHW);
            let coeff_a_dir_u =
                coeff_jet.directional_family(coeff_jet.a_first, dir_u, COEFF_SUPPORT_BW);
            let coeff_a_dir_v =
                coeff_jet.directional_family(coeff_jet.a_first, dir_v, COEFF_SUPPORT_BW);
            let coeff_aa_dir_u =
                coeff_jet.directional_family(coeff_jet.aa_first, dir_u, COEFF_SUPPORT_BW);
            let coeff_aa_dir_v =
                coeff_jet.directional_family(coeff_jet.aa_first, dir_v, COEFF_SUPPORT_BW);

            f_a_u += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &coeff_dir_u,
                &coeff_a_dir_u,
                &state.moments,
            )?;
            f_a_v += exact::cell_second_derivative_from_moments(
                cell,
                &dc_da,
                &coeff_dir_v,
                &coeff_a_dir_v,
                &state.moments,
            )?;
            f_aa_u += exact::cell_third_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &coeff_dir_u,
                &dc_daa,
                &coeff_a_dir_u,
                &coeff_a_dir_u,
                &coeff_aa_dir_u,
                &state.moments,
            )?;
            f_aa_v += exact::cell_third_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &coeff_dir_v,
                &dc_daa,
                &coeff_a_dir_v,
                &coeff_a_dir_v,
                &coeff_aa_dir_v,
                &state.moments,
            )?;

            let coeff_dir_uv = coeff_jet.mixed_directional_from_b_family(
                coeff_jet.b_first,
                dir_u,
                dir_v,
                COEFF_SUPPORT_BHW,
            );
            let coeff_a_dir_uv = coeff_jet.mixed_directional_from_b_family(
                coeff_jet.ab_first,
                dir_u,
                dir_v,
                COEFF_SUPPORT_BW,
            );
            let coeff_aa_dir_uv = coeff_jet.mixed_directional_from_b_family(
                coeff_jet.aab_first,
                dir_u,
                dir_v,
                COEFF_SUPPORT_W,
            );

            f_a_uv += exact::cell_third_derivative_from_moments(
                cell,
                &dc_da,
                &coeff_dir_u,
                &coeff_dir_v,
                &coeff_a_dir_u,
                &coeff_a_dir_v,
                &coeff_dir_uv,
                &coeff_a_dir_uv,
                &state.moments,
            )?;
            f_aa_uv += exact::cell_fourth_derivative_from_moments(
                cell,
                &dc_da,
                &dc_da,
                &coeff_dir_u,
                &coeff_dir_v,
                &dc_daa,
                &coeff_a_dir_u,
                &coeff_a_dir_v,
                &coeff_a_dir_u,
                &coeff_a_dir_v,
                &coeff_dir_uv,
                &coeff_aa_dir_u,
                &coeff_aa_dir_v,
                &coeff_a_dir_uv,
                &coeff_a_dir_uv,
                &coeff_aa_dir_uv,
                &state.moments,
            )?;

            let mut coeff_u_dir_u = vec![[0.0; 4]; r];
            let mut coeff_u_dir_v = vec![[0.0; 4]; r];
            let mut coeff_u_dir_uv = vec![[0.0; 4]; r];
            let mut coeff_au_dir_u = vec![[0.0; 4]; r];
            let mut coeff_au_dir_v = vec![[0.0; 4]; r];
            let mut coeff_au_dir_uv = vec![[0.0; 4]; r];
            for u in 1..r {
                coeff_u_dir_u[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.b_first,
                    u,
                    dir_u,
                    COEFF_SUPPORT_BHW,
                );
                coeff_u_dir_v[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.b_first,
                    u,
                    dir_v,
                    COEFF_SUPPORT_BHW,
                );
                coeff_u_dir_uv[u] = coeff_jet.param_mixed_from_bb_family(
                    coeff_jet.bb_first,
                    u,
                    dir_u,
                    dir_v,
                    COEFF_SUPPORT_BW,
                );
                coeff_au_dir_u[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.ab_first,
                    u,
                    dir_u,
                    COEFF_SUPPORT_BW,
                );
                coeff_au_dir_v[u] = coeff_jet.param_directional_from_b_family(
                    coeff_jet.ab_first,
                    u,
                    dir_v,
                    COEFF_SUPPORT_BW,
                );
                coeff_au_dir_uv[u] = coeff_jet.param_mixed_from_bb_family(
                    coeff_jet.abb_first,
                    u,
                    dir_u,
                    dir_v,
                    COEFF_SUPPORT_W,
                );
            }

            for u in 1..r {
                f_au_u[u] += exact::cell_third_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_u[u],
                    &coeff_dir_u,
                    &coeff_au[u],
                    &coeff_a_dir_u,
                    &coeff_u_dir_u[u],
                    &coeff_au_dir_u[u],
                    &state.moments,
                )?;
                f_au_v[u] += exact::cell_third_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_u[u],
                    &coeff_dir_v,
                    &coeff_au[u],
                    &coeff_a_dir_v,
                    &coeff_u_dir_v[u],
                    &coeff_au_dir_v[u],
                    &state.moments,
                )?;
                f_au_uv[u] += exact::cell_fourth_derivative_from_moments(
                    cell,
                    &dc_da,
                    &coeff_u[u],
                    &coeff_dir_u,
                    &coeff_dir_v,
                    &coeff_au[u],
                    &coeff_a_dir_u,
                    &coeff_a_dir_v,
                    &coeff_u_dir_u[u],
                    &coeff_u_dir_v[u],
                    &coeff_dir_uv,
                    &coeff_au_dir_u[u],
                    &coeff_au_dir_v[u],
                    &coeff_a_dir_uv,
                    &coeff_u_dir_uv[u],
                    &coeff_au_dir_uv[u],
                    &state.moments,
                )?;
            }

            for u in 1..r {
                for v in u..r {
                    let second_coeff =
                        coeff_jet.pair_from_b_family(coeff_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                    let base_val = exact::cell_second_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &second_coeff,
                        &state.moments,
                    )?;
                    f_uv[[u, v]] += base_val;
                    if u != v {
                        f_uv[[v, u]] += base_val;
                    }

                    let third_u = coeff_jet.pair_directional_from_bb_family(
                        coeff_jet.bb_first,
                        u,
                        v,
                        dir_u,
                        COEFF_SUPPORT_BW,
                    );
                    let third_v = coeff_jet.pair_directional_from_bb_family(
                        coeff_jet.bb_first,
                        u,
                        v,
                        dir_v,
                        COEFF_SUPPORT_BW,
                    );
                    let fourth_uv = coeff_jet.pair_mixed_from_bbb_family(
                        coeff_jet.bbb_first,
                        u,
                        v,
                        dir_u,
                        dir_v,
                        COEFF_SUPPORT_W,
                    );

                    let dir_u_val = exact::cell_third_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &coeff_dir_u,
                        &second_coeff,
                        &coeff_u_dir_u[u],
                        &coeff_u_dir_u[v],
                        &third_u,
                        &state.moments,
                    )?;
                    let dir_v_val = exact::cell_third_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &coeff_dir_v,
                        &second_coeff,
                        &coeff_u_dir_v[u],
                        &coeff_u_dir_v[v],
                        &third_v,
                        &state.moments,
                    )?;
                    let mix_val = exact::cell_fourth_derivative_from_moments(
                        cell,
                        &coeff_jet.first[u],
                        &coeff_jet.first[v],
                        &coeff_dir_u,
                        &coeff_dir_v,
                        &second_coeff,
                        &coeff_u_dir_u[u],
                        &coeff_u_dir_v[u],
                        &coeff_u_dir_u[v],
                        &coeff_u_dir_v[v],
                        &coeff_dir_uv,
                        &third_u,
                        &third_v,
                        &coeff_u_dir_uv[u],
                        &coeff_u_dir_uv[v],
                        &fourth_uv,
                        &state.moments,
                    )?;
                    f_uv_u[[u, v]] += dir_u_val;
                    f_uv_v[[u, v]] += dir_v_val;
                    f_uv_uv[[u, v]] += mix_val;
                    if u != v {
                        f_uv_u[[v, u]] += dir_u_val;
                        f_uv_v[[v, u]] += dir_v_val;
                        f_uv_uv[[v, u]] += mix_val;
                    }
                }
            }
        }

        f_u[0] = -marginal.mu1;
        f_uv[[0, 0]] = -marginal.mu2;
        f_uv_u[[0, 0]] = -dir_u[0] * marginal.mu3;
        f_uv_v[[0, 0]] = -dir_v[0] * marginal.mu3;
        f_uv_uv[[0, 0]] = -dir_u[0] * dir_v[0] * marginal.mu4;

        let inv_f_a = 1.0 / f_a;
        let mut a_u = Array1::<f64>::zeros(r);
        for u in 0..r {
            a_u[u] = -f_u[u] * inv_f_a;
        }
        let mut a_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * a_u[v] + f_au[v] * a_u[u] + f_aa * a_u[u] * a_u[v])
                        * inv_f_a;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }
        let a_u_dir_u = a_uv.dot(dir_u);
        let a_u_dir_v = a_uv.dot(dir_v);
        let mut a_uv_u = Array2::<f64>::zeros((r, r));
        let mut a_uv_v = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let n_u = f_uv_u[[u, v]]
                    + f_au_u[u] * a_u[v]
                    + f_au[u] * a_u_dir_u[v]
                    + f_au_u[v] * a_u[u]
                    + f_au[v] * a_u_dir_u[u]
                    + f_aa_u * a_u[u] * a_u[v]
                    + f_aa * (a_u_dir_u[u] * a_u[v] + a_u[u] * a_u_dir_u[v]);
                let val_u = -(n_u + f_a_u * a_uv[[u, v]]) * inv_f_a;
                a_uv_u[[u, v]] = val_u;
                a_uv_u[[v, u]] = val_u;

                let n_v = f_uv_v[[u, v]]
                    + f_au_v[u] * a_u[v]
                    + f_au[u] * a_u_dir_v[v]
                    + f_au_v[v] * a_u[u]
                    + f_au[v] * a_u_dir_v[u]
                    + f_aa_v * a_u[u] * a_u[v]
                    + f_aa * (a_u_dir_v[u] * a_u[v] + a_u[u] * a_u_dir_v[v]);
                let val_v = -(n_v + f_a_v * a_uv[[u, v]]) * inv_f_a;
                a_uv_v[[u, v]] = val_v;
                a_uv_v[[v, u]] = val_v;
            }
        }
        let a_u_uv = a_uv_u.dot(dir_v);
        let mut a_uv_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let n_uv = f_uv_uv[[u, v]]
                    + f_au_uv[u] * a_u[v]
                    + f_au_u[u] * a_u_dir_v[v]
                    + f_au_v[u] * a_u_dir_u[v]
                    + f_au[u] * a_u_uv[v]
                    + f_au_uv[v] * a_u[u]
                    + f_au_u[v] * a_u_dir_v[u]
                    + f_au_v[v] * a_u_dir_u[u]
                    + f_au[v] * a_u_uv[u]
                    + f_aa_uv * a_u[u] * a_u[v]
                    + f_aa_u * (a_u_dir_v[u] * a_u[v] + a_u[u] * a_u_dir_v[v])
                    + f_aa_v * (a_u_dir_u[u] * a_u[v] + a_u[u] * a_u_dir_u[v])
                    + f_aa
                        * (a_u_uv[u] * a_u[v]
                            + a_u_dir_u[u] * a_u_dir_v[v]
                            + a_u_dir_v[u] * a_u_dir_u[v]
                            + a_u[u] * a_u_uv[v]);
                let val = -(n_uv
                    + f_a_v * a_uv_u[[u, v]]
                    + f_a_u * a_uv_v[[u, v]]
                    + f_a_uv * a_uv[[u, v]])
                    * inv_f_a;
                a_uv_uv[[u, v]] = val;
                a_uv_uv[[v, u]] = val;
            }
        }

        let z_obs = self.z[row];
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta_val = eval_coeff4_at(&obs.coeff, z_obs);

        let mut g_u_fixed = vec![[0.0; 4]; r];
        let mut g_au_fixed = vec![[0.0; 4]; r];
        let mut g_bu_fixed = vec![[0.0; 4]; r];
        let mut g_aau_fixed = vec![[0.0; 4]; r];
        let mut g_abu_fixed = vec![[0.0; 4]; r];
        let mut g_bbu_fixed = vec![[0.0; 4]; r];
        let mut g_aaau_fixed = vec![[0.0; 4]; r];
        let mut g_aabu_fixed = vec![[0.0; 4]; r];
        let mut g_abbu_fixed = vec![[0.0; 4]; r];
        let mut g_bbbu_fixed = vec![[0.0; 4]; r];

        g_u_fixed[1] = obs.dc_db;
        g_au_fixed[1] = obs.dc_dab;
        g_bu_fixed[1] = obs.dc_dbb;
        g_aau_fixed[1] = obs.dc_daab;
        g_abu_fixed[1] = obs.dc_dabb;
        g_bbu_fixed[1] = obs.dc_dbbb;

        if let (Some(h_range), Some(runtime)) = (h_range, score_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                h_range,
                z_obs,
                "score-warp fourth-direction observed",
                |_, idx, basis_span| {
                    fill_score_basis_cell_coeff_jet(
                        idx,
                        basis_span,
                        b,
                        scale,
                        &mut g_u_fixed,
                        &mut g_bu_fixed,
                    );
                    Ok(())
                },
            )?;
        }
        if let (Some(w_range), Some(runtime)) = (w_range, link_runtime) {
            Self::for_each_deviation_basis_cubic_at(
                runtime,
                w_range,
                u_obs,
                "link-wiggle fourth-direction observed",
                |_, idx, basis_span| {
                    fill_link_basis_cell_coeff_jet(
                        idx,
                        basis_span,
                        a,
                        b,
                        scale,
                        &mut g_u_fixed,
                        &mut g_au_fixed,
                        &mut g_bu_fixed,
                        &mut g_aau_fixed,
                        &mut g_abu_fixed,
                        &mut g_bbu_fixed,
                    );
                    let (dc_aaaw, dc_aabw, dc_abbw, dc_bbbw) =
                        exact::link_basis_cell_third_partials(basis_span);
                    g_aaau_fixed[idx] = scale_coeff4(dc_aaaw, scale);
                    g_aabu_fixed[idx] = scale_coeff4(dc_aabw, scale);
                    g_abbu_fixed[idx] = scale_coeff4(dc_abbw, scale);
                    g_bbbu_fixed[idx] = scale_coeff4(dc_bbbw, scale);
                    Ok(())
                },
            )?;
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            1,
            h_range,
            w_range,
            &g_u_fixed,
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &g_aaau_fixed,
            &g_aabu_fixed,
            &g_abbu_fixed,
            &g_bbbu_fixed,
        );

        let g_a = eval_coeff4_at(&obs.dc_da, z_obs);
        let g_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let g_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let mut g_u = Array1::<f64>::zeros(r);
        let mut g_au = Array1::<f64>::zeros(r);
        let mut g_aau = Array1::<f64>::zeros(r);
        let mut g_aaau = Array1::<f64>::zeros(r);
        let mut g_uv = Array2::<f64>::zeros((r, r));
        let mut g_auv = Array2::<f64>::zeros((r, r));
        let mut g_aauv = Array2::<f64>::zeros((r, r));
        for u in 1..r {
            g_u[u] = eval_coeff4_at(&g_jet.first[u], z_obs);
            g_au[u] = eval_coeff4_at(&g_jet.a_first[u], z_obs);
            g_aau[u] = eval_coeff4_at(&g_jet.aa_first[u], z_obs);
            g_aaau[u] = eval_coeff4_at(&g_jet.aaa_first[u], z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let second_coeff = g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_BHW);
                let val = eval_coeff4_at(&second_coeff, z_obs);
                g_uv[[u, v]] = val;
                g_uv[[v, u]] = val;

                let third_coeff = g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_BW);
                let fourth_coeff = g_jet.pair_from_b_family(g_jet.aab_first, u, v, COEFF_SUPPORT_W);
                let third_val = eval_coeff4_at(&third_coeff, z_obs);
                let fourth_val = eval_coeff4_at(&fourth_coeff, z_obs);
                g_auv[[u, v]] = third_val;
                g_auv[[v, u]] = third_val;
                g_aauv[[u, v]] = fourth_val;
                g_aauv[[v, u]] = fourth_val;
            }
        }

        let g_dir_u_fixed = g_jet.directional_family(g_jet.first, dir_u, COEFF_SUPPORT_BHW);
        let g_dir_v_fixed = g_jet.directional_family(g_jet.first, dir_v, COEFF_SUPPORT_BHW);
        let g_a_dir_u_fixed = g_jet.directional_family(g_jet.a_first, dir_u, COEFF_SUPPORT_BW);
        let g_a_dir_v_fixed = g_jet.directional_family(g_jet.a_first, dir_v, COEFF_SUPPORT_BW);
        let g_aa_dir_u_fixed = g_jet.directional_family(g_jet.aa_first, dir_u, COEFF_SUPPORT_BW);
        let g_aa_dir_v_fixed = g_jet.directional_family(g_jet.aa_first, dir_v, COEFF_SUPPORT_BW);
        let g_dir_uv_fixed =
            g_jet.mixed_directional_from_b_family(g_jet.b_first, dir_u, dir_v, COEFF_SUPPORT_BHW);
        let g_a_dir_uv_fixed =
            g_jet.mixed_directional_from_b_family(g_jet.ab_first, dir_u, dir_v, COEFF_SUPPORT_BW);
        let g_aa_dir_uv_fixed =
            g_jet.mixed_directional_from_b_family(g_jet.aab_first, dir_u, dir_v, COEFF_SUPPORT_W);

        let g_dir_u = eval_coeff4_at(&g_dir_u_fixed, z_obs);
        let g_dir_v = eval_coeff4_at(&g_dir_v_fixed, z_obs);
        let g_dir_uv = eval_coeff4_at(&g_dir_uv_fixed, z_obs);
        let g_a_u_fixed = eval_coeff4_at(&g_a_dir_u_fixed, z_obs);
        let g_a_v_fixed = eval_coeff4_at(&g_a_dir_v_fixed, z_obs);
        let g_aa_u_fixed = eval_coeff4_at(&g_aa_dir_u_fixed, z_obs);
        let g_aa_v_fixed = eval_coeff4_at(&g_aa_dir_v_fixed, z_obs);
        let g_a_uv_fixed = eval_coeff4_at(&g_a_dir_uv_fixed, z_obs);
        let g_aa_uv_fixed = eval_coeff4_at(&g_aa_dir_uv_fixed, z_obs);

        let mut g_u_u_fixed = Array1::<f64>::zeros(r);
        let mut g_u_v_fixed = Array1::<f64>::zeros(r);
        let mut g_u_uv_fixed = Array1::<f64>::zeros(r);
        let mut g_au_u_fixed = Array1::<f64>::zeros(r);
        let mut g_au_v_fixed = Array1::<f64>::zeros(r);
        let mut g_au_uv_fixed = Array1::<f64>::zeros(r);
        let mut g_uv_u_fixed = Array2::<f64>::zeros((r, r));
        let mut g_uv_v_fixed = Array2::<f64>::zeros((r, r));
        let mut g_uv_uv_fixed = Array2::<f64>::zeros((r, r));
        let mut g_auv_u_fixed = Array2::<f64>::zeros((r, r));
        let mut g_auv_v_fixed = Array2::<f64>::zeros((r, r));

        for u in 1..r {
            let tmp_u =
                g_jet.param_directional_from_b_family(g_jet.b_first, u, dir_u, COEFF_SUPPORT_BHW);
            let tmp_v =
                g_jet.param_directional_from_b_family(g_jet.b_first, u, dir_v, COEFF_SUPPORT_BHW);
            let tmp_uv =
                g_jet.param_mixed_from_bb_family(g_jet.bb_first, u, dir_u, dir_v, COEFF_SUPPORT_BW);
            let tmp_au_u =
                g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir_u, COEFF_SUPPORT_BW);
            let tmp_au_v =
                g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir_v, COEFF_SUPPORT_BW);
            let tmp_au_uv =
                g_jet.param_mixed_from_bb_family(g_jet.abb_first, u, dir_u, dir_v, COEFF_SUPPORT_W);
            g_u_u_fixed[u] = eval_coeff4_at(&tmp_u, z_obs);
            g_u_v_fixed[u] = eval_coeff4_at(&tmp_v, z_obs);
            g_u_uv_fixed[u] = eval_coeff4_at(&tmp_uv, z_obs);
            g_au_u_fixed[u] = eval_coeff4_at(&tmp_au_u, z_obs);
            g_au_v_fixed[u] = eval_coeff4_at(&tmp_au_v, z_obs);
            g_au_uv_fixed[u] = eval_coeff4_at(&tmp_au_uv, z_obs);
        }
        for u in 1..r {
            for v in u..r {
                let third_u = g_jet.pair_directional_from_bb_family(
                    g_jet.bb_first,
                    u,
                    v,
                    dir_u,
                    COEFF_SUPPORT_BW,
                );
                let third_v = g_jet.pair_directional_from_bb_family(
                    g_jet.bb_first,
                    u,
                    v,
                    dir_v,
                    COEFF_SUPPORT_BW,
                );
                let fourth_uv = g_jet.pair_mixed_from_bbb_family(
                    g_jet.bbb_first,
                    u,
                    v,
                    dir_u,
                    dir_v,
                    COEFF_SUPPORT_W,
                );
                let a_third_u = g_jet.pair_directional_from_bb_family(
                    g_jet.abb_first,
                    u,
                    v,
                    dir_u,
                    COEFF_SUPPORT_W,
                );
                let a_third_v = g_jet.pair_directional_from_bb_family(
                    g_jet.abb_first,
                    u,
                    v,
                    dir_v,
                    COEFF_SUPPORT_W,
                );
                let vu = eval_coeff4_at(&third_u, z_obs);
                let vv = eval_coeff4_at(&third_v, z_obs);
                let vuv = eval_coeff4_at(&fourth_uv, z_obs);
                g_uv_u_fixed[[u, v]] = vu;
                g_uv_v_fixed[[u, v]] = vv;
                g_uv_uv_fixed[[u, v]] = vuv;
                g_uv_u_fixed[[v, u]] = vu;
                g_uv_v_fixed[[v, u]] = vv;
                g_uv_uv_fixed[[v, u]] = vuv;
                let atu = eval_coeff4_at(&a_third_u, z_obs);
                let atv = eval_coeff4_at(&a_third_v, z_obs);
                g_auv_u_fixed[[u, v]] = atu;
                g_auv_v_fixed[[u, v]] = atv;
                g_auv_u_fixed[[v, u]] = atu;
                g_auv_v_fixed[[v, u]] = atv;
            }
        }

        let eta_u = g_a * &a_u + &g_u;
        let mut eta_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let val = g_a * a_uv[[u, v]]
                    + g_aa * a_u[u] * a_u[v]
                    + g_au[u] * a_u[v]
                    + g_au[v] * a_u[u]
                    + g_uv[[u, v]];
                eta_uv[[u, v]] = val;
                eta_uv[[v, u]] = val;
            }
        }

        let a_dir_u = a_u.dot(dir_u);
        let a_dir_v = a_u.dot(dir_v);
        let g_a_u = g_aa * a_dir_u + g_a_u_fixed;
        let g_a_v = g_aa * a_dir_v + g_a_v_fixed;
        let g_aa_u = g_aaa * a_dir_u + g_aa_u_fixed;
        let g_aa_v = g_aaa * a_dir_v + g_aa_v_fixed;

        let mut g_u_u = Array1::<f64>::zeros(r);
        let mut g_u_v = Array1::<f64>::zeros(r);
        let mut g_au_u = Array1::<f64>::zeros(r);
        let mut g_au_v = Array1::<f64>::zeros(r);
        for u in 0..r {
            g_u_u[u] = g_au[u] * a_dir_u + g_u_u_fixed[u];
            g_u_v[u] = g_au[u] * a_dir_v + g_u_v_fixed[u];
            g_au_u[u] = g_aau[u] * a_dir_u + g_au_u_fixed[u];
            g_au_v[u] = g_aau[u] * a_dir_v + g_au_v_fixed[u];
        }

        let mut eta_uv_u = Array2::<f64>::zeros((r, r));
        let mut eta_uv_v = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let g_uv_u = g_auv[[u, v]] * a_dir_u + g_uv_u_fixed[[u, v]];
                let g_uv_v = g_auv[[u, v]] * a_dir_v + g_uv_v_fixed[[u, v]];
                let val_u = g_a_u * a_uv[[u, v]]
                    + g_a * a_uv_u[[u, v]]
                    + g_aa_u * a_u[u] * a_u[v]
                    + g_aa * (a_u_dir_u[u] * a_u[v] + a_u[u] * a_u_dir_u[v])
                    + g_au_u[u] * a_u[v]
                    + g_au[u] * a_u_dir_u[v]
                    + g_au_u[v] * a_u[u]
                    + g_au[v] * a_u_dir_u[u]
                    + g_uv_u;
                eta_uv_u[[u, v]] = val_u;
                eta_uv_u[[v, u]] = val_u;

                let val_v = g_a_v * a_uv[[u, v]]
                    + g_a * a_uv_v[[u, v]]
                    + g_aa_v * a_u[u] * a_u[v]
                    + g_aa * (a_u_dir_v[u] * a_u[v] + a_u[u] * a_u_dir_v[v])
                    + g_au_v[u] * a_u[v]
                    + g_au[u] * a_u_dir_v[v]
                    + g_au_v[v] * a_u[u]
                    + g_au[v] * a_u_dir_v[u]
                    + g_uv_v;
                eta_uv_v[[u, v]] = val_v;
                eta_uv_v[[v, u]] = val_v;
            }
        }

        let a_dir_uv = a_u_dir_u.dot(dir_v);
        let g_a_uv = g_aaa * a_dir_u * a_dir_v
            + g_aa * a_dir_uv
            + g_aa_u_fixed * a_dir_v
            + g_aa_v_fixed * a_dir_u
            + g_a_uv_fixed;
        let g_aa_uv = g_aaau.dot(dir_u) * a_dir_v
            + g_aaau.dot(dir_v) * a_dir_u
            + g_aaa * a_dir_uv
            + g_aa_uv_fixed;
        let mut g_u_uv = Array1::<f64>::zeros(r);
        let mut g_au_uv = Array1::<f64>::zeros(r);
        for u in 0..r {
            g_u_uv[u] = g_aau[u] * a_dir_u * a_dir_v
                + g_au[u] * a_dir_uv
                + g_au_u_fixed[u] * a_dir_v
                + g_au_v_fixed[u] * a_dir_u
                + g_u_uv_fixed[u];
            let row_u_u = g_aauv.row(u).dot(dir_u);
            let row_u_v = g_aauv.row(u).dot(dir_v);
            g_au_uv[u] = g_aaau[u] * a_dir_u * a_dir_v
                + g_aau[u] * a_dir_uv
                + row_u_u * a_dir_v
                + row_u_v * a_dir_u
                + g_au_uv_fixed[u];
        }

        let mut eta_uv_uv = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let g_uv_uv = g_aauv[[u, v]] * a_dir_u * a_dir_v
                    + g_auv[[u, v]] * a_dir_uv
                    + g_auv_u_fixed[[u, v]] * a_dir_v
                    + g_auv_v_fixed[[u, v]] * a_dir_u
                    + g_uv_uv_fixed[[u, v]];
                let val = g_a_uv * a_uv[[u, v]]
                    + g_a_u * a_uv_v[[u, v]]
                    + g_a_v * a_uv_u[[u, v]]
                    + g_a * a_uv_uv[[u, v]]
                    + g_aa_uv * a_u[u] * a_u[v]
                    + g_aa_u * (a_u_dir_v[u] * a_u[v] + a_u[u] * a_u_dir_v[v])
                    + g_aa_v * (a_u_dir_u[u] * a_u[v] + a_u[u] * a_u_dir_u[v])
                    + g_aa
                        * (a_u_uv[u] * a_u[v]
                            + a_u_dir_u[u] * a_u_dir_v[v]
                            + a_u_dir_v[u] * a_u_dir_u[v]
                            + a_u[u] * a_u_uv[v])
                    + g_au_uv[u] * a_u[v]
                    + g_au_u[u] * a_u_dir_v[v]
                    + g_au_v[u] * a_u_dir_u[v]
                    + g_au[u] * a_u_uv[v]
                    + g_au_uv[v] * a_u[u]
                    + g_au_u[v] * a_u_dir_v[u]
                    + g_au_v[v] * a_u_dir_u[u]
                    + g_au[v] * a_u_uv[u]
                    + g_uv_uv;
                eta_uv_uv[[u, v]] = val;
                eta_uv_uv[[v, u]] = val;
            }
        }

        let eta_dir_u = g_a * a_dir_u + g_dir_u;
        let eta_dir_v = g_a * a_dir_v + g_dir_v;
        let eta_u_dir_u = eta_uv.dot(dir_u);
        let eta_u_dir_v = eta_uv.dot(dir_v);
        let eta_dir_uv = g_a_v * a_dir_u + g_a_u_fixed * a_dir_v + g_a * a_dir_uv + g_dir_uv;
        let eta_u_uv = eta_uv_u.dot(dir_v);

        let y_i = self.y[row];
        let w_i = self.weights[row];
        let s_y = 2.0 * y_i - 1.0;
        let m = s_y * eta_val;
        let (k1, k2, k3, k4) = signed_probit_neglog_derivatives_up_to_fourth(m, w_i)?;
        let u1 = s_y * k1;
        let u3 = s_y * k3;

        let mut out = Array2::<f64>::zeros((r, r));
        for u in 0..r {
            for v in u..r {
                let a_term = eta_u[u] * eta_u[v] * eta_dir_u;
                let a_term_v = eta_u_dir_v[u] * eta_u[v] * eta_dir_u
                    + eta_u[u] * eta_u_dir_v[v] * eta_dir_u
                    + eta_u[u] * eta_u[v] * eta_dir_uv;
                let b_term = eta_uv_u[[u, v]];
                let b_term_v = eta_uv_uv[[u, v]];
                let c_term = eta_uv[[u, v]] * eta_dir_u
                    + eta_u[u] * eta_u_dir_u[v]
                    + eta_u[v] * eta_u_dir_u[u];
                let c_term_v = eta_uv_v[[u, v]] * eta_dir_u
                    + eta_uv[[u, v]] * eta_dir_uv
                    + eta_u_dir_v[u] * eta_u_dir_u[v]
                    + eta_u[u] * eta_u_uv[v]
                    + eta_u_dir_v[v] * eta_u_dir_u[u]
                    + eta_u[v] * eta_u_uv[u];
                let val = k4 * eta_dir_v * a_term
                    + u3 * a_term_v
                    + u3 * eta_dir_v * c_term
                    + k2 * c_term_v
                    + k2 * eta_dir_v * b_term
                    + u1 * b_term_v;
                out[[u, v]] = val;
                out[[v, u]] = val;
            }
        }
        Ok(out)
    }

    pub(super) fn row_primary_fourth_contracted_recompute(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
        row_ctx: &BernoulliMarginalSlopeRowExactContext,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        // Exact zero by bilinearity. Keep zero directions off the FLEX
        // cell-walk fallback even when the other side is a valid ψ axis.
        if Self::primary_direction_is_zero(dir_u, &cache.primary)
            || Self::primary_direction_is_zero(dir_v, &cache.primary)
        {
            let r = cache.primary.total;
            return Ok(Array2::<f64>::zeros((r, r)));
        }
        // FLEX fast path (gam#683): both directions single-axis → a scalar
        // scale of the cached symmetric axis-projected fourth tensor, by
        // bilinearity: fourth_contracted(s_u·e_a, s_v·e_b) = s_u·s_v·T4[a][b].
        if let (Some((a, s_u)), Some((b, s_v))) = (
            Self::single_primary_axis(dir_u, &cache.primary),
            Self::single_primary_axis(dir_v, &cache.primary),
        ) {
            if let Some(tensors) =
                self.flex_axis_fourth_tensors_for_row(block_states, cache, row)?
            {
                let scale = s_u * s_v;
                let mut out = match (a, b) {
                    (0, 0) => tensors.qq.clone(),
                    (1, 1) => tensors.gg.clone(),
                    (0, 1) | (1, 0) => tensors.qg.clone(),
                    _ => {
                        return Err(format!(
                            "bernoulli marginal-slope FLEX fourth fast path primary axis out of range: a={a}, b={b}, primary_total={}",
                            cache.primary.total
                        ));
                    }
                };
                out.mapv_inplace(|value| value * scale);
                return Ok(out);
            }
        }
        let ordered = self.row_primary_fourth_contracted_recompute_ordered(
            row,
            block_states,
            cache,
            row_ctx,
            dir_u,
            dir_v,
        )?;
        if !self.effective_flex_active(block_states)? {
            return Ok(ordered);
        }

        let swapped = self.row_primary_fourth_contracted_recompute_ordered(
            row,
            block_states,
            cache,
            row_ctx,
            dir_v,
            dir_u,
        )?;
        let mut sym = ordered;
        for i in 0..sym.nrows() {
            for j in 0..sym.ncols() {
                sym[[i, j]] = 0.5 * (sym[[i, j]] + swapped[[i, j]]);
            }
        }
        Ok(sym)
    }

    /// Like `add_pullback_primary_hessian` but only accumulates the h/w
    /// cross-block contributions. The marginal-marginal, marginal-logslope,
    /// and logslope-logslope blocks are handled by the weighted-Gram operator.
    pub(super) fn add_pullback_primary_hessian_hw_only(
        &self,
        target: &mut Array2<f64>,
        row: usize,
        slices: &BlockSlices,
        primary: &PrimarySlices,
        primary_hessian: ArrayView2<'_, f64>,
    ) {
        let h = primary_hessian;
        if let (Some(primary_h), Some(block_h)) = (primary.h.as_ref(), slices.h.as_ref()) {
            for (local_idx, global_idx) in block_h.clone().enumerate() {
                let h_q = h[[0, primary_h.start + local_idx]];
                if h_q != 0.0 {
                    {
                        let mut col = target.slice_mut(s![slices.marginal.clone(), global_idx]);
                        self.marginal_design
                            .axpy_row_into(row, h_q, &mut col)
                            .expect("marginal axpy column mismatch");
                    }
                    {
                        let mut row_view =
                            target.slice_mut(s![global_idx, slices.marginal.clone()]);
                        self.marginal_design
                            .axpy_row_into(row, h_q, &mut row_view)
                            .expect("marginal axpy row mismatch");
                    }
                }

                let h_g = h[[1, primary_h.start + local_idx]];
                if h_g != 0.0 {
                    {
                        let mut col = target.slice_mut(s![slices.logslope.clone(), global_idx]);
                        self.logslope_design
                            .axpy_row_into(row, h_g, &mut col)
                            .expect("logslope axpy column mismatch");
                    }
                    {
                        let mut row_view =
                            target.slice_mut(s![global_idx, slices.logslope.clone()]);
                        self.logslope_design
                            .axpy_row_into(row, h_g, &mut row_view)
                            .expect("logslope axpy row mismatch");
                    }
                }
            }

            target
                .slice_mut(s![block_h.clone(), block_h.clone()])
                .scaled_add(
                    1.0,
                    &h.slice(s![
                        primary_h.start..primary_h.end,
                        primary_h.start..primary_h.end
                    ]),
                );
        }

        if let (Some(primary_w), Some(block_w)) = (primary.w.as_ref(), slices.w.as_ref()) {
            for (local_idx, global_idx) in block_w.clone().enumerate() {
                let w_q = h[[0, primary_w.start + local_idx]];
                if w_q != 0.0 {
                    {
                        let mut col = target.slice_mut(s![slices.marginal.clone(), global_idx]);
                        self.marginal_design
                            .axpy_row_into(row, w_q, &mut col)
                            .expect("marginal axpy column mismatch");
                    }
                    {
                        let mut row_view =
                            target.slice_mut(s![global_idx, slices.marginal.clone()]);
                        self.marginal_design
                            .axpy_row_into(row, w_q, &mut row_view)
                            .expect("marginal axpy row mismatch");
                    }
                }

                let w_g = h[[1, primary_w.start + local_idx]];
                if w_g != 0.0 {
                    {
                        let mut col = target.slice_mut(s![slices.logslope.clone(), global_idx]);
                        self.logslope_design
                            .axpy_row_into(row, w_g, &mut col)
                            .expect("logslope axpy column mismatch");
                    }
                    {
                        let mut row_view =
                            target.slice_mut(s![global_idx, slices.logslope.clone()]);
                        self.logslope_design
                            .axpy_row_into(row, w_g, &mut row_view)
                            .expect("logslope axpy row mismatch");
                    }
                }
            }

            if let (Some(primary_h), Some(block_h)) = (primary.h.as_ref(), slices.h.as_ref()) {
                target
                    .slice_mut(s![block_h.clone(), block_w.clone()])
                    .scaled_add(
                        1.0,
                        &h.slice(s![
                            primary_h.start..primary_h.end,
                            primary_w.start..primary_w.end
                        ]),
                    );
                target
                    .slice_mut(s![block_w.clone(), block_h.clone()])
                    .scaled_add(
                        1.0,
                        &h.slice(s![
                            primary_w.start..primary_w.end,
                            primary_h.start..primary_h.end
                        ]),
                    );
            }

            target
                .slice_mut(s![block_w.clone(), block_w.clone()])
                .scaled_add(
                    1.0,
                    &h.slice(s![
                        primary_w.start..primary_w.end,
                        primary_w.start..primary_w.end
                    ]),
                );
        }
    }

    pub(super) fn exact_newton_joint_hessian_dense_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<Array2<f64>, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let started = std::time::Instant::now();
        let process_monitor_guard = crate::process_monitor::track_scope(format!(
            "BMS dense-H build n={n} p={}",
            slices.total
        ));
        let hessian_source = if cache.row_primary_hessians.is_some() {
            "row-primary-cache"
        } else {
            "row-stream"
        };
        if log_exact_work(n) {
            log::info!(
                "[BMS dense-H] build start n={} p={} source={} route=workspace-dense",
                n,
                slices.total,
                hessian_source
            );
        }
        let n_chunks = n.div_ceil(ROW_CHUNK_SIZE);
        let completed_chunks = AtomicUsize::new(0);
        let progress_step = (n_chunks / 10).max(1);
        let acc = (0..n_chunks)
            .into_par_iter()
            .try_fold(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    let chunk_len = end - start;
                    let mut w_mm = Array1::<f64>::zeros(chunk_len);
                    let mut w_mg = Array1::<f64>::zeros(chunk_len);
                    let mut w_gg = Array1::<f64>::zeros(chunk_len);
                    let mut h_q = primary
                        .h
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((chunk_len, range.len())));
                    let mut h_g = primary
                        .h
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((chunk_len, range.len())));
                    let mut h_h = primary
                        .h
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((range.len(), range.len())));
                    let mut w_q = primary
                        .w
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((chunk_len, range.len())));
                    let mut w_g = primary
                        .w
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((chunk_len, range.len())));
                    let mut h_w = match (primary.h.as_ref(), primary.w.as_ref()) {
                        (Some(h_range), Some(w_range)) => {
                            Some(Array2::<f64>::zeros((h_range.len(), w_range.len())))
                        }
                        _ => None,
                    };
                    let mut w_w = primary
                        .w
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((range.len(), range.len())));
                    let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                    for (local, row) in (start..end).enumerate() {
                        let hess_view =
                            if let Some(cached) = Self::cached_row_primary_hessian(cache, row) {
                                cached
                            } else {
                                let row_ctx = Self::row_ctx(cache, row);
                                let row_moments = cache
                                    .row_cell_moments
                                    .as_ref()
                                    .and_then(|bundle| bundle.row(row, 9));
                                self.compute_row_analytic_flex_into_with_moments(
                                    row,
                                    block_states,
                                    primary,
                                    row_ctx,
                                    row_moments,
                                    cache.cell_family_forest.as_ref(),
                                    true,
                                    &mut scratch,
                                )?;
                                scratch.hess.view()
                            };
                        w_mm[local] = hess_view[[0, 0]];
                        w_mg[local] = hess_view[[0, 1]];
                        w_gg[local] = hess_view[[1, 1]];
                        if let Some(primary_h) = primary.h.as_ref() {
                            if let Some(ref mut hq) = h_q {
                                hq.row_mut(local)
                                    .assign(&hess_view.slice(s![0, primary_h.clone()]));
                            }
                            if let Some(ref mut hg) = h_g {
                                hg.row_mut(local)
                                    .assign(&hess_view.slice(s![1, primary_h.clone()]));
                            }
                            if let Some(ref mut hh) = h_h {
                                hh.scaled_add(
                                    1.0,
                                    &hess_view.slice(s![primary_h.clone(), primary_h.clone()]),
                                );
                            }
                        }
                        if let Some(primary_w) = primary.w.as_ref() {
                            if let Some(ref mut wq) = w_q {
                                wq.row_mut(local)
                                    .assign(&hess_view.slice(s![0, primary_w.clone()]));
                            }
                            if let Some(ref mut wg) = w_g {
                                wg.row_mut(local)
                                    .assign(&hess_view.slice(s![1, primary_w.clone()]));
                            }
                            if let Some(ref mut ww) = w_w {
                                ww.scaled_add(
                                    1.0,
                                    &hess_view.slice(s![primary_w.clone(), primary_w.clone()]),
                                );
                            }
                            if let (Some(primary_h), Some(ref mut hw)) =
                                (primary.h.as_ref(), h_w.as_mut())
                            {
                                hw.scaled_add(
                                    1.0,
                                    &hess_view.slice(s![primary_h.clone(), primary_w.clone()]),
                                );
                            }
                        }
                    }
                    acc.add_weighted_design_grams(self, start..end, &w_mm, &w_mg, &w_gg)?;
                    acc.add_weighted_hw_cross_terms(
                        self,
                        start..end,
                        slices,
                        h_q.as_ref(),
                        h_g.as_ref(),
                        h_h.as_ref(),
                        w_q.as_ref(),
                        w_g.as_ref(),
                        h_w.as_ref(),
                        w_w.as_ref(),
                    )?;
                    if log_exact_work(n) {
                        let done = completed_chunks.fetch_add(1, Ordering::Relaxed) + 1;
                        if done == n_chunks || done % progress_step == 0 {
                            log::info!(
                                "[BMS dense-H] progress chunks={}/{} rows={}/{} elapsed={:.3}s",
                                done,
                                n_chunks,
                                (done * ROW_CHUNK_SIZE).min(n),
                                n,
                                started.elapsed().as_secs_f64()
                            );
                        }
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || BernoulliBlockHessianAccumulator::new(slices),
                |mut left, right| -> Result<_, String> {
                    left.add(&right);
                    Ok(left)
                },
            )?;
        let dense = acc.to_dense(slices);
        if log_exact_work(n) {
            log::info!(
                "[BMS dense-H] build done n={} p={} source={} route=workspace-dense elapsed={:.3}s",
                n,
                slices.total,
                hessian_source,
                started.elapsed().as_secs_f64()
            );
        }
        drop(process_monitor_guard);
        Ok(dense)
    }

    pub(super) fn exact_newton_joint_fused_gradient_dense_from_cache(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<ExactNewtonJointFusedDenseEvaluation, String> {
        let slices = &cache.slices;
        let primary = &cache.primary;
        let n = self.y.len();
        let started = std::time::Instant::now();
        let process_monitor_guard = crate::process_monitor::track_scope(format!(
            "BMS fused exact-gradient+dense-H n={n} p={}",
            slices.total
        ));
        if log_exact_work(n) {
            log::info!(
                "[BMS fused exact-gradient+dense-H] eval start n={} p={} source=cache row_primary_hessian_cache={}",
                n,
                slices.total,
                cache.row_primary_hessians.is_some()
            );
        }
        let make_acc = || {
            (
                0.0_f64,
                Array1::<f64>::zeros(slices.marginal.len()),
                Array1::<f64>::zeros(slices.logslope.len()),
                slices
                    .h
                    .as_ref()
                    .map(|range| Array1::<f64>::zeros(range.len())),
                slices
                    .w
                    .as_ref()
                    .map(|range| Array1::<f64>::zeros(range.len())),
                BernoulliBlockHessianAccumulator::new(slices),
            )
        };
        let n_chunks = n.div_ceil(ROW_CHUNK_SIZE);
        let completed_chunks = AtomicUsize::new(0);
        let progress_step = (n_chunks / 10).max(1);
        let (log_likelihood, grad_marginal, grad_logslope, grad_h, grad_w, hessian_acc) =
            (0..n_chunks)
                .into_par_iter()
                .try_fold(make_acc, |mut acc, chunk_idx| -> Result<_, String> {
                    let start = chunk_idx * ROW_CHUNK_SIZE;
                    let end = (start + ROW_CHUNK_SIZE).min(n);
                    let chunk_len = end - start;
                    let mut w_mm = Array1::<f64>::zeros(chunk_len);
                    let mut w_mg = Array1::<f64>::zeros(chunk_len);
                    let mut w_gg = Array1::<f64>::zeros(chunk_len);
                    let mut h_q = primary
                        .h
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((chunk_len, range.len())));
                    let mut h_g = primary
                        .h
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((chunk_len, range.len())));
                    let mut h_h = primary
                        .h
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((range.len(), range.len())));
                    let mut w_q = primary
                        .w
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((chunk_len, range.len())));
                    let mut w_g = primary
                        .w
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((chunk_len, range.len())));
                    let mut h_w = match (primary.h.as_ref(), primary.w.as_ref()) {
                        (Some(h_range), Some(w_range)) => {
                            Some(Array2::<f64>::zeros((h_range.len(), w_range.len())))
                        }
                        _ => None,
                    };
                    let mut w_w = primary
                        .w
                        .as_ref()
                        .map(|range| Array2::<f64>::zeros((range.len(), range.len())));
                    let mut scratch = BernoulliMarginalSlopeFlexRowScratch::new(primary.total);
                    for (local, row) in (start..end).enumerate() {
                        // When both neglog+grad+hess are cached (Host variant),
                        // consume them directly — no second row kernel pass.
                        let cached_hessian;
                        let neglog;
                        // We need a stable place for the cached grad row.
                        let cached_grad_row_storage;
                        if let Some((cached_neglog, cached_grad_row)) =
                            Self::cached_row_primary_eval(cache, row)
                        {
                            neglog = cached_neglog;
                            // The cached grad row is an ArrayView1; we hold it.
                            cached_grad_row_storage = Some(cached_grad_row);
                            let cached_hess =
                                Self::cached_row_primary_hessian(cache, row);
                            cached_hessian = cached_hess;
                        } else {
                            // Cache miss (device-resident or no cache): run the
                            // row kernel once for neglog + grad + (maybe) hess.
                            cached_grad_row_storage = None;
                            let row_ctx = Self::row_ctx(cache, row);
                            let cached_hess =
                                Self::cached_row_primary_hessian(cache, row);
                            let row_moments = cache
                                .row_cell_moments
                                .as_ref()
                                .and_then(|bundle| bundle.row(row, 9));
                            let computed_neglog =
                                self.compute_row_analytic_flex_into_with_moments(
                                    row,
                                    block_states,
                                    primary,
                                    row_ctx,
                                    row_moments,
                                    cache.cell_family_forest.as_ref(),
                                    cached_hess.is_none(),
                                    &mut scratch,
                                )?;
                            neglog = computed_neglog;
                            cached_hessian = cached_hess;
                        }
                        // Resolve grad source: cached row or scratch.grad.
                        let grad_ref: &dyn std::ops::Index<usize, Output = f64> =
                            if let Some(ref cgr) = cached_grad_row_storage {
                                cgr
                            } else {
                                &scratch.grad
                            };
                        acc.0 -= neglog;
                        {
                            let mut marginal = acc.1.view_mut();
                            self.marginal_design.axpy_row_into(
                                row,
                                Self::exact_newton_score_component_from_objective_gradient(
                                    grad_ref[0],
                                ),
                                &mut marginal,
                            )?;
                        }
                        {
                            let mut logslope = acc.2.view_mut();
                            self.logslope_design.axpy_row_into(
                                row,
                                Self::exact_newton_score_component_from_objective_gradient(
                                    grad_ref[1],
                                ),
                                &mut logslope,
                            )?;
                        }
                        if let (Some(primary_h), Some(grad_h)) =
                            (primary.h.as_ref(), acc.3.as_mut())
                        {
                            for idx in 0..primary_h.len() {
                                grad_h[idx] +=
                                    Self::exact_newton_score_component_from_objective_gradient(
                                        grad_ref[primary_h.start + idx],
                                    );
                            }
                        }
                        if let (Some(primary_w), Some(grad_w)) =
                            (primary.w.as_ref(), acc.4.as_mut())
                        {
                            for idx in 0..primary_w.len() {
                                grad_w[idx] +=
                                    Self::exact_newton_score_component_from_objective_gradient(
                                        grad_ref[primary_w.start + idx],
                                    );
                            }
                        }

                        let hess_view = cached_hessian.unwrap_or_else(|| scratch.hess.view());
                        w_mm[local] = hess_view[[0, 0]];
                        w_mg[local] = hess_view[[0, 1]];
                        w_gg[local] = hess_view[[1, 1]];
                        if let Some(primary_h) = primary.h.as_ref() {
                            if let Some(ref mut hq) = h_q {
                                hq.row_mut(local)
                                    .assign(&hess_view.slice(s![0, primary_h.clone()]));
                            }
                            if let Some(ref mut hg) = h_g {
                                hg.row_mut(local)
                                    .assign(&hess_view.slice(s![1, primary_h.clone()]));
                            }
                            if let Some(ref mut hh) = h_h {
                                hh.scaled_add(
                                    1.0,
                                    &hess_view.slice(s![primary_h.clone(), primary_h.clone()]),
                                );
                            }
                        }
                        if let Some(primary_w) = primary.w.as_ref() {
                            if let Some(ref mut wq) = w_q {
                                wq.row_mut(local)
                                    .assign(&hess_view.slice(s![0, primary_w.clone()]));
                            }
                            if let Some(ref mut wg) = w_g {
                                wg.row_mut(local)
                                    .assign(&hess_view.slice(s![1, primary_w.clone()]));
                            }
                            if let Some(ref mut ww) = w_w {
                                ww.scaled_add(
                                    1.0,
                                    &hess_view.slice(s![primary_w.clone(), primary_w.clone()]),
                                );
                            }
                            if let (Some(primary_h), Some(ref mut hw)) =
                                (primary.h.as_ref(), h_w.as_mut())
                            {
                                hw.scaled_add(
                                    1.0,
                                    &hess_view.slice(s![primary_h.clone(), primary_w.clone()]),
                                );
                            }
                        }
                    }
                    acc.5
                        .add_weighted_design_grams(self, start..end, &w_mm, &w_mg, &w_gg)?;
                    acc.5.add_weighted_hw_cross_terms(
                        self,
                        start..end,
                        slices,
                        h_q.as_ref(),
                        h_g.as_ref(),
                        h_h.as_ref(),
                        w_q.as_ref(),
                        w_g.as_ref(),
                        h_w.as_ref(),
                        w_w.as_ref(),
                    )?;
                    if log_exact_work(n) {
                        let done = completed_chunks.fetch_add(1, Ordering::Relaxed) + 1;
                        if done == n_chunks || done % progress_step == 0 {
                            log::info!(
                                "[BMS fused exact-gradient+dense-H] progress chunks={}/{} rows={}/{} elapsed={:.3}s",
                                done,
                                n_chunks,
                                (done * ROW_CHUNK_SIZE).min(n),
                                n,
                                started.elapsed().as_secs_f64()
                            );
                        }
                    }
                    Ok(acc)
                })
                .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                    left.0 += right.0;
                    left.1 += &right.1;
                    left.2 += &right.2;
                    if let (Some(lhs), Some(rhs)) = (left.3.as_mut(), right.3.as_ref()) {
                        *lhs += rhs;
                    }
                    if let (Some(lhs), Some(rhs)) = (left.4.as_mut(), right.4.as_ref()) {
                        *lhs += rhs;
                    }
                    left.5.add(&right.5);
                    Ok(left)
                })?;

        let mut gradient = Array1::<f64>::zeros(slices.total);
        gradient
            .slice_mut(s![slices.marginal.clone()])
            .assign(&grad_marginal);
        gradient
            .slice_mut(s![slices.logslope.clone()])
            .assign(&grad_logslope);
        if let (Some(range), Some(grad_h)) = (slices.h.as_ref(), grad_h.as_ref()) {
            gradient.slice_mut(s![range.clone()]).assign(grad_h);
        }
        if let (Some(range), Some(grad_w)) = (slices.w.as_ref(), grad_w.as_ref()) {
            gradient.slice_mut(s![range.clone()]).assign(grad_w);
        }
        let hessian = hessian_acc.to_dense(slices);
        if log_exact_work(n) {
            log::info!(
                "[BMS fused exact-gradient+dense-H] eval done n={} p={} source=cache elapsed={:.3}s",
                n,
                slices.total,
                started.elapsed().as_secs_f64()
            );
        }
        drop(process_monitor_guard);
        Ok(ExactNewtonJointFusedDenseEvaluation {
            gradient: ExactNewtonJointGradientEvaluation {
                log_likelihood,
                gradient,
            },
            hessian,
        })
    }

    pub(super) fn log_likelihood_from_exact_cache(
        &self,
        block_states: &[ParameterBlockState],
        cache: &BernoulliMarginalSlopeExactEvalCache,
    ) -> Result<f64, String> {
        if !self.effective_flex_active(block_states)? {
            return self
                .log_likelihood_only_with_options(block_states, &BlockwiseFitOptions::default());
        }
        let n = self.y.len();
        let started = std::time::Instant::now();
        let process_monitor_guard = crate::process_monitor::track_scope(format!(
            "BMS exact-loglik eval n={n} p={}",
            cache.slices.total
        ));
        if log_exact_work(n) {
            log::info!(
                "[BMS exact-loglik] eval start n={} p={} source=cache",
                n,
                cache.slices.total
            );
        }
        let beta_h = self.score_beta(block_states)?;
        let beta_w = self.link_beta(block_states)?;
        let total: Result<f64, String> = (0..n)
            .into_par_iter()
            .try_fold(
                || 0.0,
                |mut log_likelihood, row| -> Result<_, String> {
                    let row_ctx = Self::row_ctx(cache, row);
                    let slope = block_states[1].eta[row];
                    let obs = self.observed_denested_cell_partials(
                        row,
                        row_ctx.intercept,
                        slope,
                        beta_h,
                        beta_w,
                    )?;
                    let s_i = eval_coeff4_at(&obs.coeff, self.z[row]);
                    let signed = (2.0 * self.y[row] - 1.0) * s_i;
                    let (log_cdf, _) = signed_probit_logcdf_and_mills_ratio(signed);
                    log_likelihood += self.weights[row] * log_cdf;
                    Ok(log_likelihood)
                },
            )
            .try_reduce(
                || 0.0,
                |left, right| -> Result<_, String> { Ok(left + right) },
            );
        let log_likelihood = total?;
        if log_exact_work(n) {
            log::info!(
                "[BMS exact-loglik] eval done n={} p={} source=cache elapsed={:.3}s",
                n,
                cache.slices.total,
                started.elapsed().as_secs_f64()
            );
        }
        drop(process_monitor_guard);
        Ok(log_likelihood)
    }

}

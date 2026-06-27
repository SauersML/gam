use super::*;

/// One recorded ContinuationPath demotion: a structural defect that, for a
/// continuation-entry objective, routes a seed to a heavier path regime instead
/// of disqualifying it. Carried in the seed-loop ledger so the startup stats
/// surface a heavier-regime re-entry (with its reason) rather than a vanished
/// candidate. Never fatal.
#[derive(Clone, Debug)]
pub(crate) struct PathDemotionRecord {
    /// 0-based seed index whose structural defect triggered the demotion.
    pub(crate) seed_idx: usize,
    /// The path regime the seed was re-entered at after the demotion.
    pub(crate) regime: crate::continuation_path::PathRegime,
    /// Human-readable reason (the underlying structural diagnosis message).
    pub(crate) reason: String,
}

/// Bidirectional inner-PIRLS feedback channel.
///
/// The outer-loop scheduler (BFGS or ARC bridge) writes a coarsened
/// iteration cap into `cap` before each accepted gradient/Hessian eval,
/// and the inner solver (`execute_pirls_if_needed`) writes back into
/// `last_iters` / `last_converged` after each NON-screening solve so the
/// next outer iter's schedule can adapt to the inner solver's actual
/// convergence behavior rather than a hardcoded iter-count tier.
///
/// All atomics are owned by `RemlObjectiveState`; the bridges hold
/// `Arc` clones. `last_iters == 0` means "no inner-Newton signal yet" —
/// the schedule falls back to the coarse iter-count tier for the first
/// outer iter. `ift_residual_bits == 0` means "no IFT-predictor quality
/// signal yet" — the schedule's +margin reverts to the conservative
/// default. The two signals are independent: the IFT residual may be
/// missing even after a successful inner solve (when the predictor was
/// rejected by the |Δρ| cap and a flat warm-start was used instead).
#[derive(Clone, Debug)]
pub struct InnerProgressFeedback {
    pub cap: Arc<AtomicUsize>,
    /// Count of accepted outer steps observed via the
    /// `OuterAcceptObserver` plugged into `opt`'s solver. Replaces
    /// the bridge-side `eval_count / 2` heuristic on routes that
    /// see trial-and-rejection probing (ARC dense, matrix-free TR):
    /// rejection iters used to inflate the schedule's iter index,
    /// lifting the cap too early. With this counter, the schedule
    /// sees the true accepted-step count and the cap relaxes only
    /// when real progress has been made.
    pub accepted_iter: Arc<AtomicUsize>,
    pub last_iters: Arc<AtomicUsize>,
    pub last_converged: Arc<AtomicBool>,
    /// Bit-packed `f64` residual `‖β_converged − β_predicted‖ /
    /// ‖β_converged‖` from the previous IFT-predicted PIRLS solve.
    /// Used to tighten or loosen the cap's `+margin` when the
    /// predictor's empirical faithfulness is known: a small residual
    /// means the inner Newton starts very close to the KKT β and only
    /// needs +1 iter of margin; a large residual means the prediction
    /// collapsed to flat warm-start and the inner Newton has more
    /// recovery work, so +4 is appropriate. `0` means "no signal yet".
    pub ift_residual: Arc<AtomicU64>,
    /// Bit-packed `f64` accepted gain ratio
    /// (`actual_reduction / predicted_reduction`) from the most recent
    /// non-screening PIRLS solve. NaN bits encode "no signal yet"
    /// (matches `ift_residual`'s sentinel discipline). Used by
    /// `first_order_inner_cap_schedule` as a third quality signal
    /// alongside `last_iters` and `last_converged`: a small accept_rho
    /// (model overstating predicted reduction) is a hint the next
    /// iter's inner Newton may need extra margin even when the
    /// previous solve converged in few iters.
    pub accept_rho: Arc<AtomicU64>,
}

impl InnerProgressFeedback {
    /// Snapshot the read-back atomics for the cap schedule. Returns `None`
    /// when no inner solve has reported yet (`last_iters == 0`); the
    /// schedule then falls back to the coarse iter-count tier.
    ///
    /// The IFT residual decoding uses the same NaN-sentinel discipline
    /// as `RemlState::predict_warm_start_beta_ift_with_outcome` — see commit
    /// `748cc066` for the rationale. A residual of exactly 0 (every
    /// β_predicted_i bit-equal to β_converged_i) must NOT be confused
    /// with "no signal yet"; the NaN sentinel + `is_finite()` check
    /// distinguishes the two cleanly. Both ends of the atomic share
    /// `crate::reml::outer_eval::IFT_RESIDUAL_NO_SIGNAL_BITS`
    /// implicitly via the same bit pattern.
    pub(crate) fn snapshot(&self) -> Option<InnerProgressSnapshot> {
        let iters = self.last_iters.load(Ordering::Relaxed);
        if iters == 0 {
            None
        } else {
            // NaN sentinel + is_finite() check covers three cases in
            // one expression: "no signal yet" (sentinel decodes to NaN,
            // fails is_finite), "corrupted state" (any non-finite or
            // negative residual), and "real signal" (finite non-negative
            // → Some). Matches the IFT predictor's reader semantics.
            let residual_bits = self.ift_residual.load(Ordering::Relaxed);
            let r = f64::from_bits(residual_bits);
            let last_ift_residual = if r.is_finite() && r >= 0.0 {
                Some(r)
            } else {
                None
            };
            let accept_rho_bits = self.accept_rho.load(Ordering::Relaxed);
            let ar = f64::from_bits(accept_rho_bits);
            let last_accept_rho = if ar.is_finite() && ar >= 0.0 {
                Some(ar)
            } else {
                None
            };
            Some(InnerProgressSnapshot {
                last_iters: iters,
                last_converged: self.last_converged.load(Ordering::Relaxed),
                last_ift_residual,
                last_accept_rho,
            })
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct InnerProgressSnapshot {
    pub(crate) last_iters: usize,
    pub(crate) last_converged: bool,
    /// Most-recent IFT predictor residual (see field doc on
    /// `InnerProgressFeedback`). `None` when the predictor has not
    /// reported yet, when the cache was reset, or when the previous
    /// solve fell back to flat warm-start (no IFT prediction
    /// consumed).
    pub(crate) last_ift_residual: Option<f64>,
    /// Most-recent accepted LM gain ratio (see field doc on
    /// `InnerProgressFeedback::accept_rho`). `None` when no step was
    /// accepted in the previous solve (rejection-exhausted) or when
    /// the cache was reset.
    pub(crate) last_accept_rho: Option<f64>,
}

#[inline]
pub(crate) fn effective_seed_budget(
    requested_budget: usize,
    solver: Solver,
    risk_profile: gam_problem::SeedRiskProfile,
    _screening_enabled: bool,
) -> usize {
    let requested_budget = requested_budget.max(1);
    match (solver, risk_profile) {
        (Solver::Efs | Solver::HybridEfs, _) => 1,
        (Solver::Arc, gam_problem::SeedRiskProfile::Survival) => 1,
        (Solver::Arc, gam_problem::SeedRiskProfile::GeneralizedLinear) => 2,
        _ => requested_budget,
    }
}

#[inline]
pub(crate) fn should_screen_seeds(
    config: &OuterConfig,
    solver: Solver,
    generated_seed_count: usize,
    seed_budget: usize,
) -> bool {
    if matches!(solver, Solver::Efs | Solver::HybridEfs) {
        return false;
    }
    if config.initial_rho.is_some() && seed_budget == 1 && !config.screen_initial_rho {
        return false;
    }
    config.screening_cap.is_some()
        && generated_seed_count > seed_budget
        && matches!(solver, Solver::Arc | Solver::Bfgs)
}

#[inline]
pub(crate) fn expensive_unsuccessful_seed_limit(
    solver: Solver,
    risk_profile: gam_problem::SeedRiskProfile,
) -> Option<usize> {
    match (solver, risk_profile) {
        (Solver::Efs | Solver::HybridEfs, _) => Some(1),
        (Solver::Arc, gam_problem::SeedRiskProfile::Survival) => Some(1),
        (Solver::Arc, gam_problem::SeedRiskProfile::GeneralizedLinear) => Some(2),
        _ => None,
    }
}

/// Multipliers for the seed-screening cap cascade, applied to the user's
/// `screen_max_inner_iterations`.
///
/// The cascade evaluates seeds at successive caps until at least one
/// produces a finite cost — at which point it ranks them and exits. The
/// geometric ×4 progression keeps each escalation step cheap relative to
/// the next while still letting the cap reach the full inner budget if
/// needed: `initial × {1, 4, 16}` followed by uncapped (`0` interpreted
/// by the inner solver as "use the full `pirls_config.max_iterations`").
///
/// Worst-case extra work bounds: every seed pays at most
/// `initial × (1 + 4 + 16)` = 21 × initial inner iterations across the
/// three capped stages before falling through to the uncapped pass —
/// negligible overhead compared to a full P-IRLS solve, paid only when
/// every cap stage collapsed all seeds to non-finite cost.
pub(crate) const SEED_SCREENING_CASCADE_MULTIPLIERS: [usize; 3] = [1, 4, 16];

/// Sentinel cap value passed to the inner solver to mean "no cap — use
/// the full `pirls_config.max_iterations`". Always the final cascade
/// stage after the geometric escalation exhausts.
pub(crate) const SEED_SCREENING_UNCAPPED: usize = 0;

pub(crate) fn rank_seeds_with_screening(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    context: &str,
    seeds: &[Array1<f64>],
) -> Vec<Array1<f64>> {
    let Some(screening_cap) = config.screening_cap.as_ref() else {
        return seeds.to_vec();
    };

    let initial_cap = config.seed_config.screen_max_inner_iterations.max(1);
    let previous_cap = screening_cap.swap(initial_cap, Ordering::Relaxed);

    // Geometric cap cascade: each stage exits the moment any seed produces
    // a finite cost. The original two-stage protocol (initial cap → fully
    // uncapped on every seed) has a degenerate worst case at large scale
    // — when every seed at the shallow cap collapses, we re-evaluate every
    // seed at the *full* inner budget, costing `N_seeds × full_pirls_work`
    // just to pick a starting point. The cascade replaces that all-or-
    // nothing jump with a geometric escalation: the typical case stays at
    // the initial cap (one pass), and the rare uniform-failure case pays
    // only `21 × initial` extra inner iterations before the uncapped
    // fallback.
    let cascade_caps = [
        PriorityBudgetStage {
            cap: initial_cap.saturating_mul(SEED_SCREENING_CASCADE_MULTIPLIERS[0]),
        },
        PriorityBudgetStage {
            cap: initial_cap.saturating_mul(SEED_SCREENING_CASCADE_MULTIPLIERS[1]),
        },
        PriorityBudgetStage {
            cap: initial_cap.saturating_mul(SEED_SCREENING_CASCADE_MULTIPLIERS[2]),
        },
        PriorityBudgetStage {
            cap: SEED_SCREENING_UNCAPPED,
        },
    ];

    let cascade_start = std::time::Instant::now();
    log::info!(
        "[STAGE] {context}: seed screening cascade start seeds={} initial_cap={} stages={}",
        seeds.len(),
        initial_cap,
        cascade_caps.len(),
    );

    let cascade_result = rank_indices_with_budget_cascade(
        seeds.len(),
        &cascade_caps,
        |stage, cap, idx| {
            // gam#979: the seed-screening cascade is an OUTER-search activity and
            // must honor the same fit-level wall-clock budget the inner cycle loop
            // already respects (`inner_blockwise_fit`'s deadline break). Without
            // this the guard is asymmetric: once the budget is spent the cascade
            // keeps starting fresh seed proxies — each paying a full cycle-0
            // constrained-Newton setup before the inner break — and at the
            // uncapped final stage (every seed rejecting on a baseline whose
            // active-set QP never certifies) that grinds far past the deadline.
            // Reject the remaining seeds instantly so the cascade collapses to the
            // heuristic order in bounded time. The budget is already exceeded here,
            // so the foregone proxy ranking cannot affect a within-budget fit.
            if crate::rho_optimizer::outer_wall_clock_deadline_exceeded() {
                return Err(());
            }
            screening_cap.store(cap, Ordering::Relaxed);
            obj.reset();
            screening_cap.store(cap, Ordering::Relaxed);
            let seed_started = std::time::Instant::now();
            let result = obj.eval_screening_proxy(&seeds[idx]);
            let seed_elapsed = seed_started.elapsed().as_secs_f64();
            match result {
                Ok(cost) if cost.is_finite() => {
                    log::info!(
                        "[STAGE] {context}: seed-screen stage={} seed={}/{} cap={} elapsed={:.3}s cost={:.6e}",
                        stage,
                        idx + 1,
                        seeds.len(),
                        if cap == 0 {
                            "uncapped".to_string()
                        } else {
                            cap.to_string()
                        },
                        seed_elapsed,
                        cost,
                    );
                    Ok(cost)
                }
                Ok(cost) => {
                    log::info!(
                        "[STAGE] {context}: seed-screen stage={} seed={}/{} cap={} elapsed={:.3}s cost=non-finite ({:.3e})",
                        stage,
                        idx + 1,
                        seeds.len(),
                        if cap == 0 {
                            "uncapped".to_string()
                        } else {
                            cap.to_string()
                        },
                        seed_elapsed,
                        cost,
                    );
                    Ok(cost)
                }
                Err(_) => {
                    log::info!(
                        "[STAGE] {context}: seed-screen stage={} seed={}/{} cap={} elapsed={:.3}s rejected (error)",
                        stage,
                        idx + 1,
                        seeds.len(),
                        if cap == 0 {
                            "uncapped".to_string()
                        } else {
                            cap.to_string()
                        },
                        seed_elapsed,
                    );
                    Err(())
                }
            }
        },
        |PriorityStageSummary {
             stage,
             cap,
             ranked,
             rejected,
         }| {
            log::info!(
                "[STAGE] {context}: seed-screen stage={} cap={} elapsed={:.3}s ranked={} rejected={}",
                stage,
                if cap == 0 {
                    "uncapped".to_string()
                } else {
                    cap.to_string()
                },
                cascade_start.elapsed().as_secs_f64(),
                ranked,
                rejected,
            );
            if ranked > 0 && stage > 0 {
                let final_cap = if cap == 0 {
                    "uncapped".to_string()
                } else {
                    cap.to_string()
                };
                log::info!(
                    "[OUTER] {context}: seed screening cap escalated from {} to {} \
                     (initial cap was too shallow for this problem; {}/{} seeds ranked)",
                    initial_cap,
                    final_cap,
                    ranked,
                    seeds.len(),
                );
            }
        },
    );

    let rejected = cascade_result.rejected;
    let final_cap_used = cascade_result.final_cap;
    let stages_consumed = cascade_result.stages_consumed;
    let ranked = cascade_result.ranked_indices;

    screening_cap.store(previous_cap, Ordering::Relaxed);
    obj.reset();
    log::info!(
        "[OUTER] {context}: seed screening cascade complete elapsed={:.3}s stages_used={} final_cap={} ranked={}/{}",
        cascade_start.elapsed().as_secs_f64(),
        stages_consumed,
        if final_cap_used == 0 {
            "uncapped".to_string()
        } else {
            final_cap_used.to_string()
        },
        ranked.len(),
        seeds.len(),
    );

    if ranked.is_empty() {
        log::info!(
            "[OUTER] {context}: no finite seed cost even with full inner budget \
             ({} seeds, {} rejected, {} cascade stages tried); keeping heuristic order",
            seeds.len(),
            rejected,
            stages_consumed,
        );
        return seeds.to_vec();
    }

    let mut ordered = Vec::with_capacity(seeds.len());
    let mut seen = vec![false; seeds.len()];
    for idx in ranked {
        seen[idx] = true;
        ordered.push(seeds[idx].clone());
    }
    for (idx, seed) in seeds.iter().enumerate() {
        if !seen[idx] {
            ordered.push(seed.clone());
        }
    }

    // Demote over-smoothing boundary seeds below every interior seed.
    //
    // The seed-screening cost is a *marginal-likelihood* proxy fit at a
    // capped inner-iteration budget. For a separation-stability seed pinned
    // at the ρ upper bound (`Array1::from_elem(k, bounds.1)`), that proxy is
    // systematically the cheapest: the penalized coefficients are shrunk
    // into the penalty null space, the capped inner solve converges
    // trivially, and the LAML/REML value is locally flat. So screening
    // ranks the boundary seed *first*. But the boundary is a degenerate
    // descent origin: ∂V/∂ρ → 0 there (nothing left to penalize), so a
    // trust-region / Newton outer solver started at the boundary certifies
    // box-constraint stationarity at iteration 0 and never reaches the
    // interior — which, for a location-scale model, is frequently
    // *anisotropic* (the well-determined mean wants heavy shrinkage while
    // the second-moment scale block wants far less). Starting the descent
    // from any interior seed instead lets the optimizer climb back up to
    // the bound coordinate-wise when the data truly want it, while still
    // resolving the coordinates whose optimum is interior. The boundary is
    // reachable by ascent but inescapable once it is the start, so it must
    // never out-rank an interior seed. We keep it at the tail as a
    // stability fallback: if every interior seed fails its full-budget
    // solve (genuine separation), the seed loop still falls through to it.
    // (#686/#687/#688: Gaussian location-scale was pinned at ρ=bound,
    // over-smoothing the log-σ envelope and wrecking held-out calibration.)
    let rho_dim = obj.capability().theta_layout().rho_dim();
    if rho_dim > 0 && ordered.len() > 1 {
        let upper: Vec<f64> = match config.bounds.as_ref() {
            Some((_, hi)) => hi.to_vec(),
            None => vec![config.rho_bound; rho_dim],
        };
        let (interior, boundary): (Vec<Array1<f64>>, Vec<Array1<f64>>) = ordered
            .into_iter()
            .partition(|seed| !seed_is_oversmoothing_boundary(seed, rho_dim, &upper));
        if !interior.is_empty() && !boundary.is_empty() {
            log::info!(
                "[OUTER] {context}: demoted {} over-smoothing boundary seed(s) below {} \
                 interior seed(s) so the outer descent does not originate on the flat \
                 ρ=bound plateau",
                boundary.len(),
                interior.len(),
            );
        }
        ordered = interior;
        ordered.extend(boundary);

        // Guarantee the flexible (low-lambda) basin one full-budget solve for
        // models whose smoothing coordinates are not profiled Gaussian scale
        // (#1082/#1373 and Gaussian location-scale). The screening proxy is a
        // capped-inner-iteration fit, and an over-smoothed seed converges
        // trivially under that cap (coefficients collapse into the penalty
        // null space, LAML locally flat), so the proxy systematically ranks
        // over-smoothed seeds first — exactly the bias documented for the
        // boundary case above, but it also crowds out a MODERATELY flexible
        // seed that is not at the bound (and so survives the demotion). With
        // only a few full-budget solves, the genuinely flexible basin (e.g. a
        // smooth Poisson tensor surface that needs ~10 effective df) then
        // never gets solved and the fit over-smooths. Promote the single
        // most-flexible interior seed (smallest Σ of the leading rho_dim
        // coordinates) to the front so it is always among the full solves.
        // Ordinary Gaussian REML's profiled-scale basin does not exhibit this
        // bias, but Gaussian location-scale has a non-profiled log-scale block
        // and does. Keep-best across the full-solved seeds means promoting a
        // flexible seed can never worsen the returned fit; the remaining proxy
        // order is preserved.
        let promote_extreme_seeds = config
            .seed_config
            .risk_profile
            .promotes_interior_seed_extremes();
        if promote_extreme_seeds && ordered.len() > 1 {
            let rho_sum =
                |seed: &Array1<f64>| -> f64 { (0..rho_dim.min(seed.len())).map(|i| seed[i]).sum() };
            if let Some((most_flexible_idx, _)) = ordered
                .iter()
                .enumerate()
                .filter(|(_, seed)| !seed_is_oversmoothing_boundary(seed, rho_dim, &upper))
                .min_by(|(_, a), (_, b)| rho_sum(a).total_cmp(&rho_sum(b)))
            {
                if most_flexible_idx != 0 {
                    let flexible = ordered.remove(most_flexible_idx);
                    log::info!(
                        "[OUTER] {context}: promoted the most-flexible interior seed \
                         (Σρ={:.3}) to the front so the low-λ basin gets a full-budget \
                         solve (capped screening systematically under-ranks it)",
                        rho_sum(&flexible),
                    );
                    ordered.insert(0, flexible);
                }
            }

            // #1426: also guarantee a HEAVILY-penalized (high-λ) seed is the
            // SECOND full-budget solve. The flexible-seed promotion above puts
            // the most under-penalized seed at slot 0; on a non-separable
            // gamma/log default-k surface that seed lands on the λ→0 ridge whose
            // inner PIRLS hits its iteration cap, so the outer optimizer cost-
            // stalls there and reports a NON-CONVERGED near-full-basis overfit
            // (EDF ≈ k). The expensive multi-start budget for non-Gaussian ARC is
            // only 2 seeds, and the remaining proxy order tends to place ANOTHER
            // moderately-flexible seed at slot 1 — so the well-penalized basin
            // (which converges to the mgcv-like EDF ~8-9 optimum) is never solved
            // and the overfit ships. Lifting the heaviest INTERIOR seed (largest
            // Σρ over the leading rho_dim coords, excluding the over-smoothing
            // boundary seed) to slot 1 makes the budget-2 multi-start span BOTH
            // basins, so when slot 0 stalls non-converged the heavy seed converges
            // and its `converged` best wins via `candidate_improves_best`.
            //
            // The over-smoothing BOUNDARY seed is deliberately NOT chosen here: it
            // is a degenerate descent origin (∂V/∂ρ → 0 at the bound, so a TR/
            // Newton outer solver certifies box stationarity at iter 0 and never
            // reaches the interior — the #686/#687 over-smoothing trap). It stays
            // at the tail as the genuine-separation fallback. A well-interior
            // heavy seed instead descends coordinate-wise into the EDF ~8-9 basin.
            // Keep-best semantics mean this can only ADD a basin, never worsen the
            // returned fit: a genuinely separable λ→0 case (#1082) still converges
            // / best-feasibles at slot 0 and the heavy seed is simply dominated.
            if promote_extreme_seeds && ordered.len() > 2 {
                let heaviest_idx = ordered
                    .iter()
                    .enumerate()
                    .skip(1)
                    .filter(|(_, seed)| !seed_is_oversmoothing_boundary(seed, rho_dim, &upper))
                    .max_by(|(_, a), (_, b)| rho_sum(a).total_cmp(&rho_sum(b)))
                    .map(|(idx, _)| idx);
                if let Some(heaviest_idx) = heaviest_idx
                    && heaviest_idx > 1
                {
                    let heavy = ordered.remove(heaviest_idx);
                    log::info!(
                        "[OUTER] {context}: promoted the heaviest interior seed (Σρ={:.3}) to \
                         the second full-budget slot so a budget-limited non-Gaussian multi-start \
                         also solves the well-penalized basin (#1426: a non-separable λ→0 stall \
                         at slot 0 otherwise ships a non-converged full-basis overfit)",
                        rho_sum(&heavy),
                    );
                    ordered.insert(1, heavy);
                }
            }
        }
    }

    log::debug!(
        "[OUTER] {context}: seed screening ranked {}/{} candidates at cap={} \
         (initial cap={}, stages used={}); rejected={}",
        ordered.len() - rejected,
        seeds.len(),
        if final_cap_used == 0 {
            "uncapped".to_string()
        } else {
            final_cap_used.to_string()
        },
        initial_cap,
        stages_consumed,
        rejected,
    );

    ordered
}

/// ρ margin (in log-λ units) within which a smoothing coordinate counts as
/// sitting on the over-smoothing upper bound. The separation-stability seed is
/// generated *exactly* at the bound, so a small margin suffices; it is kept
/// loose enough to absorb a `project_to_bounds` round-trip without catching a
/// genuinely interior candidate (the next-densest generated seed is several
/// log-λ units below any realistic bound).
pub(crate) const OVERSMOOTH_BOUNDARY_MARGIN: f64 = 0.5;

/// Whether `seed` is pinned at the over-smoothing ρ upper bound in *every*
/// smoothing coordinate — the degenerate plateau where the penalized
/// coefficients collapse into the penalty null space and the REML/LAML
/// gradient ∂V/∂ρ vanishes. Only the leading `rho_dim` (smoothing) coordinates
/// are inspected; trailing ψ/auxiliary coordinates have their own geometry and
/// never make ρ a flat plateau. Used to keep such seeds from becoming the outer
/// optimizer's descent origin (see `rank_seeds_with_screening`).
pub(crate) fn seed_is_oversmoothing_boundary(
    seed: &Array1<f64>,
    rho_dim: usize,
    upper: &[f64],
) -> bool {
    if rho_dim == 0 || seed.len() < rho_dim {
        return false;
    }
    (0..rho_dim).all(|i| {
        let hi = upper.get(i).copied().unwrap_or(f64::INFINITY);
        hi.is_finite() && seed[i] >= hi - OVERSMOOTH_BOUNDARY_MARGIN
    })
}

#[inline]
pub(crate) fn candidate_improves_best(candidate: &OuterResult, best: Option<&OuterResult>) -> bool {
    match best {
        None => true,
        Some(best) if candidate.converged != best.converged => candidate.converged,
        Some(best) => candidate.final_value < best.final_value,
    }
}

/// Relative LAML/REML tie band within which two *converged* outer optima count
/// as statistically indistinguishable. The marginal likelihood is a smooth
/// function of ρ with a flat valley near its optimum; a gap this small means the
/// data cannot tell the two basins apart, so the choice between them must be
/// made on a secondary principle (parsimony) rather than on LAML noise.
pub(crate) const PARSIMONY_TIE_REL_BAND: f64 = 1e-3;

/// Score-relative gradient band, two orders of magnitude (100×) inside
/// [`PARSIMONY_TIE_REL_BAND`], within which a CONVERGED slot-0 optimum counts as
/// *curvature-pinned*: a sharp stationary point, not a point loitering on the
/// flat LAML valley where the parsimony tie-break could still slide ρ toward
/// more smoothing.
///
/// The outer solver certifies `converged` against the comparatively loose
/// `outer_gradient_tolerance` (an absolute floor that for a GLM REML/LAML fit is
/// typically `1e-4`-ish). That is enough to *publish* an optimum but NOT enough
/// to prove the optimum is sharp rather than flat-valley: the parsimony
/// tie-break operates within a `1e-3` relative cost band, so a converged point
/// whose residual gradient is only `~1e-4` may still be epsilon-from a
/// heavier-penalized basin the second seed would surface. Requiring the residual
/// to be `100×` *inside* that band makes "sharp" mean "the local curvature
/// pins ρ here", which is exactly the regime where the second parsimony seed is
/// provably redundant.
pub(crate) const PARSIMONY_SHARP_GRAD_REL_BAND: f64 = 1e-5;

/// Whether the deliberately-promoted parsimonious second seed (slot 1) of a
/// non-Gaussian (GeneralizedLinear / Survival) multi-start is provably redundant
/// given slot 0's converged result, so the multi-start may break after slot 0.
///
/// The #1373/#1426/#1477 guard places the flexible (low-λ) seed at slot 0 and
/// the heavy (high-λ) seed at slot 1 so keep-best can reject an under-penalized
/// slot-0 outcome. But the heavy seed can only ever *change* the published
/// answer when slot 0 lands somewhere it could be beaten:
///   - an **under-penalized** basin — some smoothing λ < 1 (ρ < 0) — that scores
///     an epsilon-better LAML while overshooting on the response scale (#1373);
///   - a **non-sharp** converged optimum sitting on the flat LAML valley, where
///     the parsimony tie-break ([`candidate_improves_best_parsimonious`]) would
///     prefer the heavier basin; or
///   - a **non-converged** stall (#1426/#1477) whose cached cost is untrustworthy
///     (handled by the converged guard below, never reaching this predicate).
///
/// When slot 0 instead CONVERGED to a curvature-pinned optimum (score-relative
/// `|g| ≤ PARSIMONY_SHARP_GRAD_REL_BAND·(1+|score|)`) whose every leading
/// smoothing λ ≥ 1 (ρ ≥ 0), it is already both well-penalized and sharp: there
/// is no flatter valley to slide down and no under-penalized basin for the heavy
/// seed to improve on, so slot 1 merely re-derives slot 0's optimum. This is the
/// #1575 binomial-REML pathology — the heavy seed re-ran the entire smoothing-
/// parameter optimization only to converge to the identical cost and ρ, doubling
/// the outer cost-eval count for nothing. Skipping it halves that work with no
/// change to the published fit.
///
/// Conservative by construction: every condition is a *necessary* feature of the
/// redundant case, so any borderline fit this rejects simply runs slot 1 exactly
/// as before. The waiver can only ever remove wasted work, never quality.
#[inline]
pub(crate) fn parsimony_second_seed_is_redundant(
    slot0_best: &OuterResult,
    rho_dim: usize,
) -> bool {
    // A non-converged slot 0 is precisely the #1426/#1477 case the heavy seed
    // exists to rescue; never waive it. With no smoothing dimension the parsimony
    // tie-break is a no-op, so there is nothing for the second seed to decide.
    if !slot0_best.converged || rho_dim == 0 {
        return false;
    }
    // Curvature-pinned: a measured residual gradient well inside the flat-valley
    // band. A missing/NaN gradient cannot certify sharpness — fall back to
    // running slot 1.
    let curvature_pinned = slot0_best.final_grad_norm.is_some_and(|g| {
        g.is_finite() && g <= PARSIMONY_SHARP_GRAD_REL_BAND * (1.0 + slot0_best.final_value.abs())
    });
    if !curvature_pinned {
        return false;
    }
    // Well-penalized: every leading smoothing coordinate has λ ≥ 1 (ρ ≥ 0), so
    // slot 0 is not the under-penalized basin #1373's heavy seed guards against.
    // Trailing auxiliary coordinates (e.g. a GAMLSS log-scale predictor) are not
    // smoothing parameters and are intentionally excluded.
    (0..rho_dim.min(slot0_best.rho.len())).all(|i| slot0_best.rho[i] >= 0.0)
}

/// Total penalty magnitude `Σρ` over the leading `rho_dim` smoothing
/// coordinates. Larger `Σρ` = larger λ = MORE smoothing = the more parsimonious
/// (lower effective-df) fit.
#[inline]
fn smoothing_rho_sum(result: &OuterResult, rho_dim: usize) -> f64 {
    (0..rho_dim.min(result.rho.len()))
        .map(|i| result.rho[i])
        .sum()
}

/// Whether a NON-converged outer result's cached `final_value` can be trusted
/// for a keep-best cost comparison (#1426/#1477).
///
/// A non-converged seed reports a `final_value` that may or may not be a
/// faithful REML/LAML evaluation at its ρ. When the bound-projected residual
/// gradient is at most modestly above the outer tolerance
/// (`<= FLAT_VALLEY_STALL_GRAD_CEILING`), the result is sitting on a genuine
/// flat-valley / near-separable plateau: the cost is an honest evaluation and a
/// legitimate basis for comparison. When the projected residual is FAR above the
/// ceiling, the inner PIRLS hit its iteration cap at an under-penalized (λ→0) ρ
/// — the cached cost is spuriously low (an invalid REML value the line search
/// could not improve) while the analytic gradient still points strongly toward
/// more penalization. That stuck-stall cost must NOT win a `final_value`
/// comparison: trusting it ships the #1426 near-full-basis overfit (EDF ≈ k).
///
/// Mirrors exactly the stuck-stall vs. genuine-plateau distinction already used
/// by [`should_stop_expensive_multistart_after_best`].
#[inline]
fn nonconverged_cost_is_trustworthy(result: &OuterResult) -> bool {
    result
        .final_grad_norm
        .is_none_or(|g| g.is_finite() && g <= FLAT_VALLEY_STALL_GRAD_CEILING)
}

/// Keep-best comparison that breaks a *near-tie* in the converged LAML toward
/// the more parsimonious (more-smoothed) basin.
///
/// The plain [`candidate_improves_best`] adopts any converged candidate with a
/// strictly lower LAML. For a GLM whose marginal likelihood is nearly flat
/// across a range of λ (e.g. a smooth Poisson tensor surface), the under-
/// penalized basin can score a LAML that is *epsilon* better while fitting noise
/// — on the response scale `exp(η)` amplifies that wiggle into a fit far worse
/// than the parsimonious optimum (#1373: the flexible-seed promotion drove the
/// optimizer into exactly such an overshoot). When two converged optima are
/// within [`PARSIMONY_TIE_REL_BAND`] of each other, the marginal likelihood
/// cannot distinguish them, so we keep the more-smoothed one. Outside the tie
/// band a genuinely-better flexible basin still wins, so a fit that truly needs
/// the flexibility is unaffected. Used only at the non-Gaussian multi-start
/// site, where the flexible (slot 0) and heavy (slot 1) seeds straddle the two
/// basins by construction.
#[inline]
pub(crate) fn candidate_improves_best_parsimonious(
    candidate: &OuterResult,
    best: Option<&OuterResult>,
    rho_dim: usize,
) -> bool {
    match best {
        None => true,
        Some(best) if candidate.converged != best.converged => candidate.converged,
        Some(best) if candidate.converged && best.converged && rho_dim > 0 => {
            let scale = candidate
                .final_value
                .abs()
                .max(best.final_value.abs())
                .max(1.0);
            let gap = (candidate.final_value - best.final_value).abs();
            if gap <= PARSIMONY_TIE_REL_BAND * scale {
                // Statistical tie on LAML: prefer the more-smoothed basin, and
                // only switch to the candidate if it is STRICTLY more penalized
                // (so an exact-tie keeps the incumbent and the result is order-
                // stable).
                smoothing_rho_sum(candidate, rho_dim) > smoothing_rho_sum(best, rho_dim)
            } else {
                candidate.final_value < best.final_value
            }
        }
        // BOTH non-converged (#1426/#1477). Do NOT blindly adopt the lower
        // `final_value`: a non-converged seed's cached cost is only an honest
        // REML/LAML evaluation when it sits on a genuine flat-valley plateau
        // (projected residual `<= FLAT_VALLEY_STALL_GRAD_CEILING`). A seed whose
        // residual is FAR above the ceiling is a stuck PIRLS-capped λ→0 stall
        // with a spuriously-low cost (the #1426 near-full-basis overfit; the
        // #1477 ps double-penalty overshoot adopted on the same untrustworthy
        // basis). Seed screening deliberately puts the flexible (λ→0) seed at
        // slot 0 and the heavy/penalized seed at slot 1, so this stuck seed's
        // bogus cost otherwise beats the honest heavier candidate. A candidate
        // with a TRUSTWORTHY cost beats an incumbent with an UNTRUSTWORTHY one
        // regardless of `final_value`, and never loses to it.
        Some(best) => {
            let candidate_trustworthy = nonconverged_cost_is_trustworthy(candidate);
            let best_trustworthy = nonconverged_cost_is_trustworthy(best);
            match (candidate_trustworthy, best_trustworthy) {
                (true, false) => true,
                (false, true) => false,
                // Both honest flat-valley plateaus: their costs ARE comparable.
                (true, true) => candidate.final_value < best.final_value,
                // Both stuck PIRLS-capped stalls (#1426): the cached cost is
                // spuriously low for BOTH, so it cannot break the tie — the
                // under-penalized λ→0 overfit (EDF ≈ k) carries the LOWER bogus
                // cost yet a much LARGER residual gradient than the heavily
                // penalized basin that is the real answer. Prefer the candidate
                // whose bound-projected residual gradient is SMALLER, i.e. the
                // one closer to a stationary point (the heavy flat-valley floor),
                // not the λ→0 stall whose huge gradient marks its cost invalid.
                // Falls back to `final_value` only when a gradient is unavailable.
                (false, false) => match (candidate.final_grad_norm, best.final_grad_norm) {
                    (Some(cg), Some(bg)) if cg.is_finite() && bg.is_finite() => cg < bg,
                    _ => candidate.final_value < best.final_value,
                },
            }
        }
    }
}

#[inline]
pub(crate) fn should_stop_expensive_multistart_after_best(
    best: Option<&OuterResult>,
    expensive_seed_limit: Option<usize>,
    quality_compare_remaining_gaussian_seeds: bool,
) -> bool {
    !quality_compare_remaining_gaussian_seeds
        && expensive_seed_limit.is_some()
        && best.is_some_and(|b| {
            b.final_value.is_finite()
                && !b.converged
                && matches!(
                    b.operator_stop_reason,
                    Some(OperatorTrustRegionStopReason::CostStallFlatValley)
                )
                // #1426: only stop the expensive multi-start on a flat-valley
                // best that is a GENUINE near-separable plateau — i.e. its
                // bound-PROJECTED residual gradient is at most modestly above the
                // outer tolerance (`<= FLAT_VALLEY_STALL_GRAD_CEILING`). The
                // separable multinomial / RKHS-collapse cases (#1082/#1237/#1355)
                // certify with a projected residual well under O(1): no remaining
                // seed can reach a stationary point the λ→0 MLE provably does not
                // have, so stopping is correct and saves an expensive second crawl.
                //
                // But a flat-valley best whose projected residual is FAR above the
                // ceiling is NOT separable — it is the #1426 stuck stall: the inner
                // PIRLS hit its iteration cap at an under-penalized (λ→0) ρ, so the
                // cached cost is spuriously low and the analytic gradient still
                // points (strongly) toward more penalization. Such a ρ is a silent
                // near-full-basis overfit (EDF ≈ k), NOT the answer mgcv converges
                // to (EDF ~8-9). Stopping here ships that overfit. Refusing to stop
                // lets the multi-start advance to a more-penalized seed, which
                // converges to the well-penalized optimum (whose `converged` best
                // then wins via `candidate_improves_best`). Bounded by the existing
                // `expensive_seed_limit`, so a genuinely pathological surface still
                // terminates after a finite number of seeds.
                && b.final_grad_norm
                    .is_none_or(|g| g.is_finite() && g <= FLAT_VALLEY_STALL_GRAD_CEILING)
        })
}

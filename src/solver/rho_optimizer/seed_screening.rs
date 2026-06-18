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
    pub(crate) regime: crate::solver::continuation_path::PathRegime,
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
    /// `crate::solver::reml::outer_eval::IFT_RESIDUAL_NO_SIGNAL_BITS`
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
    risk_profile: crate::seeding::SeedRiskProfile,
    _screening_enabled: bool,
) -> usize {
    let requested_budget = requested_budget.max(1);
    match (solver, risk_profile) {
        (Solver::Efs | Solver::HybridEfs, _) => 1,
        (Solver::Arc, crate::seeding::SeedRiskProfile::Survival) => 1,
        (Solver::Arc, crate::seeding::SeedRiskProfile::GeneralizedLinear) => 2,
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
    risk_profile: crate::seeding::SeedRiskProfile,
) -> Option<usize> {
    match (solver, risk_profile) {
        (Solver::Efs | Solver::HybridEfs, _) => Some(1),
        (Solver::Arc, crate::seeding::SeedRiskProfile::Survival) => Some(1),
        (Solver::Arc, crate::seeding::SeedRiskProfile::GeneralizedLinear) => Some(2),
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

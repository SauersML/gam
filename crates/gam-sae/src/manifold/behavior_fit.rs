//! Rung-2 two-block REML fit driver: joint activation+behavior fitting with the
//! relative block weight `λ_y` selected by REML — never a knob.
//!
//! # The objective
//!
//! The two-block model gives each output block its own Gaussian dispersion —
//! `φ_x` for the activation reconstruction, `φ_y` for the nats-unit behavior
//! tangent target — while every structural parameter (the latent coordinate
//! `t`, the gates `a`, both decoders `[B_k | C_k]`) is shared through the
//! augmented fit on `Z̃ = [Z | √λ_y·Y]` with `λ_y = φ_x/φ_y`. Profiling both
//! dispersions at a fitted state (block residuals `R_x = ‖Z − Ẑ‖²`,
//! `R_y = ‖Y − Ŷ‖²` in *unscaled* units) makes the criterion's `λ_y`-dependence
//!
//! ```text
//!   (n·p̃/2)·log((R_x + λ_y·R_y)/(n·p̃)) − (n·p_y/2)·log λ_y ,
//! ```
//!
//! (the second term is the `√λ_y` target-scaling Jacobian,
//! [`BehaviorBlock::reml_log_lambda_jacobian`]) whose unique stationary point is
//! the closed-form variance ratio
//!
//! ```text
//!   λ_y = (R_x/p_x) / (R_y/p_y)
//! ```
//!
//! ([`BehaviorBlock::reml_updated_log_lambda_y`]) — the classical REML estimate
//! of a variance-component ratio under a shared mean structure. The driver
//! below alternates (fit at fixed `λ_y`) ↔ (closed-form `λ_y` update at the
//! fitted residuals): block-coordinate descent on the joint profiled criterion,
//! each half-step solving its subproblem exactly. No grid search, no
//! user-tuned weight.
//!
//! # Payoffs realized here
//!
//! * **Gauge fixed by data** — the behavior block enters the same arrow-Schur
//!   inner solve that estimates `t`, so the latent coordinate is oriented by
//!   how the *output* changes, not by an arbitrary activation convention
//!   (see `tests_behavior_twoblock_rung2` for the planted case where the
//!   activation alone cannot orient `t` and the behavior block pins it).
//! * **Calibrated units** — the behavior target is nats-unit by construction
//!   ([`SphereTangentEmbedding`](crate::manifold::SphereTangentEmbedding)), and
//!   `λ_y` only sets the *inferential weight* of those units, so the fitted
//!   `C_k` always decodes to honest distributions ([`BehaviorBlock::split_decoder`]
//!   un-does the `√λ_y`).
//! * **Selection for mattering** — behaviorally inert structure has a zero
//!   behavior target; its residual variance ratio drives `λ_y` (and the atom's
//!   behavior decoder) toward earning nothing from the y-block evidence rather
//!   than manufacturing spurious weight.

use super::*;
use opt::{BacktrackConfig, backtracking_line_search};

/// Outcome of a two-block REML fit: the converged weight, the trajectory, and
/// the final inner loss.
#[derive(Clone, Debug)]
pub struct TwoBlockRemlFitReport {
    /// The REML-selected `log(λ_y)` installed on the term's behavior block.
    pub log_lambda_y: f64,
    /// Number of (fit, λ-update) outer sweeps performed (≥ 1).
    pub sweeps: usize,
    /// Whether the `log λ_y` fixed-point iteration met `log_lambda_tol` (as
    /// opposed to exhausting `max_sweeps`).
    pub converged: bool,
    /// `false` when the behavior residual carried no variance at some sweep
    /// (e.g. behavior constant across rows, target ≡ 0), in which case `λ_y`
    /// is not identifiable and was held at its last value — the honest report
    /// for a behaviorally inert block, not an error.
    pub lambda_identifiable: bool,
    /// Inner loss at the final fit (in the scaled augmented units of the last
    /// sweep's target).
    pub loss: SaeManifoldLoss,
    /// `log λ_y` after each sweep's update, for diagnostics (`sweeps` entries;
    /// the last equals `log_lambda_y` when identifiable).
    pub log_lambda_trajectory: Vec<f64>,
}

/// Caller-owned resolution knobs for [`SaeManifoldTerm::run_two_block_reml_fit`].
///
/// None of these choose *what* the fit converges to — the destination is the
/// data-determined REML stationary point. They only bound how far the inner
/// arrow-Schur solve and the outer `(fit, λ_y-update)` alternation walk toward
/// it, plus the pass-through inner regularization. Bundling them keeps the fit
/// entry point at a single grouped argument (the four inner knobs are the exact
/// pass-through set [`SaeManifoldTerm::run_joint_fit_arrow_schur`] already
/// takes, and the two outer knobs govern the sweep loop).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoBlockRemlControls {
    /// Upper bound on the outer `(fit, λ_y-update)` alternation. Must be ≥ 1.
    pub max_sweeps: usize,
    /// Inner arrow-Schur iteration cap, passed through to
    /// [`SaeManifoldTerm::run_joint_fit_arrow_schur`] unchanged each sweep.
    pub inner_max_iter: usize,
    /// Inner damped-Newton step size (finite, positive), passed through.
    pub step_size: f64,
    /// Ridge on the external-coordinate block of the inner solve, passed through.
    pub ridge_ext_coord: f64,
    /// Ridge on the decoder (`β`) block of the inner solve, passed through.
    pub ridge_beta: f64,
    /// Convergence tolerance on `|Δ log λ_y|` between sweeps (finite, positive).
    pub log_lambda_tol: f64,
}

/// Per-block outcome of a [`SaeManifoldTerm::run_multiblock_reml_fit`]: the
/// converged weight, whether it was identifiable, and its `log λ_ℓ` trajectory.
#[derive(Clone, Debug)]
pub struct BlockRemlOutcome {
    /// The block's label (carried through from its [`OutputBlock`]).
    pub label: String,
    /// The REML-selected `log(λ_ℓ)` installed on the block.
    pub log_lambda: f64,
    /// `false` when the block's residual carried no variance at some sweep (its
    /// target ≡ 0, or it was perfectly reconstructed): `λ_ℓ` is not identifiable
    /// and was held at its last value — the honest report for an inert block.
    pub identifiable: bool,
    /// `log λ_ℓ` after each sweep's update, for diagnostics.
    pub trajectory: Vec<f64>,
}

/// Outcome of a multi-block REML fit: the per-block converged weights, the sweep
/// count, and the final inner loss. The two-block [`TwoBlockRemlFitReport`] is
/// the `blocks.len() == 1` special case.
#[derive(Clone, Debug)]
pub struct MultiBlockRemlFitReport {
    /// One outcome per output block, in the order the blocks were supplied.
    pub blocks: Vec<BlockRemlOutcome>,
    /// Number of (fit, λ-update) outer sweeps performed (≥ 1).
    pub sweeps: usize,
    /// Whether the sweep loop stopped on convergence (all identifiable blocks
    /// met `log_lambda_tol`, or every block became unidentifiable) rather than
    /// exhausting `max_sweeps`.
    pub converged: bool,
    /// Inner loss at the final fit (in the scaled augmented units of the last
    /// sweep's targets).
    pub loss: SaeManifoldLoss,
}

impl SaeManifoldTerm {
    /// Run the block-generic multi-block joint fit with every block weight
    /// `λ_ℓ` selected by REML.
    ///
    /// The augmented target is `Z̃ = [Z | √λ_1·Y_1 | … | √λ_{K-1}·Y_{K-1}]`
    /// ([`stack_augmented_target`]): the anchor `Z` (`anchor`, `n × p_x`) plus
    /// the `√λ_ℓ`-scaled targets of `blocks`, all decoded from ONE shared latent
    /// coordinate and gate through the ordinary arrow-Schur joint fit. Each sweep
    /// (1) fits at the current weights, then (2) applies the closed-form REML
    /// variance-ratio update `λ_ℓ = (R_x/p_x)/(R_ℓ/p_ℓ)` to every block from the
    /// fitted residual — the *joint* stationary point, decoupled per block (see
    /// [`OutputBlock`]). The alternation is block-coordinate descent on the joint
    /// profiled criterion; `controls` bounds only how far it walks, never the
    /// destination.
    ///
    /// The term's atoms must be built at the augmented width
    /// `p̃ = p_x + Σ_ℓ p_ℓ`. On return the term holds the fitted state at the
    /// selected weights and each block in `blocks` carries its converged
    /// `log λ_ℓ`, so a decoder slice for block `ℓ` un-scaled by
    /// [`OutputBlock::split_honest_decoder`] decodes in honest units.
    ///
    /// A block whose residual carries no variance is *held* (its `λ_ℓ` frozen,
    /// `identifiable = false`) and excluded from the convergence test; when every
    /// block is held the sweep loop stops. At `blocks.len() == 1` this is
    /// bit-for-bit the two-block driver ([`Self::run_two_block_reml_fit`], which
    /// delegates here).
    pub fn run_multiblock_reml_fit(
        &mut self,
        anchor: ArrayView2<'_, f64>,
        blocks: &mut [OutputBlock],
        rho: &mut SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        controls: TwoBlockRemlControls,
    ) -> Result<MultiBlockRemlFitReport, String> {
        let TwoBlockRemlControls {
            max_sweeps,
            inner_max_iter,
            step_size,
            ridge_ext_coord,
            ridge_beta,
            log_lambda_tol,
        } = controls;
        if max_sweeps == 0 {
            return Err(
                "SaeManifoldTerm::run_multiblock_reml_fit: max_sweeps must be ≥ 1".to_string(),
            );
        }
        if !(log_lambda_tol.is_finite() && log_lambda_tol > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::run_multiblock_reml_fit: log_lambda_tol must be finite and \
                 positive; got {log_lambda_tol}"
            ));
        }
        if blocks.is_empty() {
            return Err(
                "SaeManifoldTerm::run_multiblock_reml_fit: need at least one output block"
                    .to_string(),
            );
        }
        let px = anchor.ncols();
        let p_tot = px + blocks.iter().map(|b| b.block_dim()).sum::<usize>();
        if p_tot != self.output_dim() {
            return Err(format!(
                "SaeManifoldTerm::run_multiblock_reml_fit: augmented width p_x + Σ p_ℓ = {p_tot} \
                 but the term's output_dim is {} (atoms must be built at the augmented width)",
                self.output_dim()
            ));
        }

        let n_obs = anchor.nrows();
        // Stacked-column bookkeeping owner. The block WIDTHS (hence every column
        // range) are fixed for the whole fit, so one layout serves every in-loop
        // range read (RSS spans, per-block decoder rescale); its stored `log λ_ℓ`
        // is the sweep-entry value and is NOT read for ranging. The FITTED layout
        // (converged weights) is installed on the term at the end.
        let range_layout = CrosscoderLayout::from_blocks(px, blocks);
        let dims: Vec<usize> = range_layout.block_dims().to_vec();
        let mut cur_log_lambda: Vec<f64> = blocks.iter().map(|b| b.log_lambda).collect();
        let mut trajectories: Vec<Vec<f64>> = vec![Vec::with_capacity(max_sweeps); blocks.len()];
        let mut identifiable = vec![true; blocks.len()];
        let mut converged = false;
        let mut sweeps = 0usize;

        // Backtracking bounds (resolution knobs, not answer-changers: the accepted
        // destination is a criterion stationary point either way — these only cap
        // how a single λ step is shortened when the full closed-form step would
        // INCREASE the profiled criterion, i.e. when the naive fixed-point would
        // overshoot into a block-abandonment runaway).
        const MAX_BACKTRACK: usize = 12;

        // Initial fit at the starting weights. Besides seeding the warm start and
        // `loss`, this runs BEFORE the first `fit_state_snapshot`, so any one-shot
        // atom rank reduction ([`Self::reduce_atoms_to_data_supported_rank`]) has
        // already settled the decoder/basis widths: every in-loop snapshot and its
        // in-place `.assign` restore then share one width and cannot shape-mismatch.
        let augmented = stack_augmented_target(anchor, blocks)?;
        let mut loss = self.run_joint_fit_arrow_schur(
            augmented.view(),
            rho,
            analytic_penalties,
            inner_max_iter,
            step_size,
            ridge_ext_coord,
            ridge_beta,
        )?;
        // No fitted RESIDUAL is carried across sweeps. Each sweep recomputes the
        // residual from a fresh refit-at-current-λ (the F1 baseline below), so the
        // closed-form λ* and the Armijo reference both read residuals produced with
        // the SAME inner budget the trials get; only `cur_log_lambda` and the
        // warm-start term state persist between sweeps. (`augmented` above seeded
        // the warm start and `loss`.)
        while sweeps < max_sweeps {
            sweeps += 1;

            // Sweep-entry warm-start state. Both the current-λ baseline and every
            // backtracking trial refit from THIS same snapshot with the SAME inner
            // budget, so their criteria differ only by the λ they were fit at.
            let base = self.fit_state_snapshot();

            // F1 — refit-at-current-λ baseline. A backtracking trial runs a full
            // `run_joint_fit_arrow_schur` with a fresh `inner_max_iter` budget, so
            // it can lower the profiled criterion purely by spending the truncated
            // inner solve's LEFTOVER descent at the SAME λ — inner-solve progress
            // masquerading as λ progress. To make the Armijo test a test of λ, the
            // value a trial must beat is itself a refit from `base` with the same
            // budget at the UNCHANGED λ: both sides start from one state and spend
            // one budget, differing only in λ. This baseline also gives the closed-
            // form λ* the residuals of the fit AT the current λ, not stale residuals
            // from the previous sweep's (differently-weighted) fit.
            self.fit_state_restore(&base)?;
            let base_aug = stack_augmented_target(anchor, blocks)?;
            let base_loss = self.run_joint_fit_arrow_schur(
                base_aug.view(),
                rho,
                analytic_penalties,
                inner_max_iter,
                step_size,
                ridge_ext_coord,
                ridge_beta,
            )?;
            let (base_rx, base_scaled) =
                self.augmented_block_rss(base_aug.view(), rho, &range_layout)?;
            let base_unscaled: Vec<f64> = (0..blocks.len())
                .map(|i| base_scaled[i] / cur_log_lambda[i].exp())
                .collect();
            let baseline_crit =
                profiled_reml_criterion(n_obs, px, base_rx, &base_unscaled, &dims, &cur_log_lambda);
            // The improved current-λ fit to fall back to if no λ step is accepted.
            let baseline_state = self.fit_state_snapshot();
            // The term currently holds this baseline fit; record its loss so any
            // convergence/stall break below reports the state actually installed.
            loss = base_loss;

            // Closed-form λ_ℓ target per identifiable block from the CURRENT-λ
            // refit's unscaled residuals (λ_ℓ* = (R_x/p_x)/(R_ℓ/p_ℓ)).
            let mut log_lambda_star = cur_log_lambda.clone();
            let mut any_update = false;
            let mut max_delta = 0.0_f64;
            for idx in 0..blocks.len() {
                if !identifiable[idx] {
                    continue;
                }
                let r_ell = base_unscaled[idx];
                let var_x = base_rx / px as f64;
                if !(r_ell > 0.0) || !(var_x > 0.0) {
                    // No block residual variance ⇒ λ_ℓ unidentifiable; hold it.
                    identifiable[idx] = false;
                    continue;
                }
                let var_y = r_ell / dims[idx] as f64;
                let star = (var_x / var_y).ln();
                log_lambda_star[idx] = star;
                max_delta = max_delta.max((star - cur_log_lambda[idx]).abs());
                any_update = true;
            }

            if !any_update || max_delta <= log_lambda_tol {
                // Either every block is held (no weight can move) or the λ fixed
                // point is within tolerance — converged in the documented sense.
                // The term holds the current-λ baseline fit, which is the answer.
                converged = true;
                for idx in 0..blocks.len() {
                    trajectories[idx].push(cur_log_lambda[idx]);
                }
                break;
            }

            // Armijo backtracking on the profiled criterion: take the largest
            // fraction `s` of the closed-form log-λ step whose refit strictly beats
            // the refit-at-current-λ baseline. Each trial refits from the SAME
            // `base` snapshot with the SAME inner budget as the baseline, so the
            // comparison isolates the λ move; the criterion penalises λ→0, so a step
            // that would abandon a block is rejected — the fix that makes the
            // otherwise non-contractive (fit, λ-update) alternation monotone.
            let armijo_eps = 1e-9 * (1.0 + baseline_crit.abs());
            // Backtracking line search on the closed-form λ step, migrated onto
            // the shared `opt` primitive. `trial(s)` evaluates the step
            // of scale `s` (always well defined here, so never `Ok(None)`) and
            // threads the trial iterate `(trial_ll, trial_loss)` through the
            // payload; `accept` inlines the exact Armijo sufficient-decrease test
            // (`trial_crit < baseline_crit − armijo_eps`), bit-for-bit identical to
            // the pre-migration loop (initial step 1.0, ×0.5 contraction,
            // `MAX_BACKTRACK` trials).
            let accepted_step = backtracking_line_search::<(Vec<f64>, SaeManifoldLoss), String>(
                BacktrackConfig {
                    initial_step: 1.0,
                    contraction: 0.5,
                    max_steps: MAX_BACKTRACK,
                },
                |s| {
                    let mut trial_ll = cur_log_lambda.clone();
                    for idx in 0..blocks.len() {
                        if identifiable[idx] {
                            trial_ll[idx] = cur_log_lambda[idx]
                                + s * (log_lambda_star[idx] - cur_log_lambda[idx]);
                        }
                    }
                    for idx in 0..blocks.len() {
                        blocks[idx] = blocks[idx].with_log_lambda(trial_ll[idx])?;
                    }
                    self.fit_state_restore(&base)?;
                    // EXACT warm-start reparameterization: the stacked target's block
                    // columns are `√λ_ℓ·Y_ℓ`, and quadratic β-penalties are scale-
                    // equivariant (`min_β ‖√λ·y − Φβ‖² + βᵀSβ` maps to
                    // `λ(‖y − Φb‖² + bᵀSb)` under `β = √λ·b`), so the incumbent fit at
                    // `λ_cur` maps to the EXACT incumbent at `λ_trial` by scaling each
                    // behavior block's decoder columns by `√(λ_trial/λ_cur)`. Without
                    // this rescale every trial starts with a spurious scaled-residual
                    // `(√λ_t − √λ_c)²·‖Ŷ‖²` that the TRUNCATED inner solve must burn
                    // its iteration budget removing — which biases the Armijo
                    // comparison against exactly the large closed-form λ jumps the
                    // fixed point needs from a distant start (the from-0.0
                    // 20-sweep-exhaustion failure of `reml_selects_lambda_y…`). The
                    // rescaled state is the same fit in the new parameterization, so
                    // this changes no fixed point — it only stops the line search from
                    // paying an artificial re-fitting tax proportional to the step.
                    for idx in 0..blocks.len() {
                        let scale = (0.5 * (trial_ll[idx] - cur_log_lambda[idx])).exp();
                        if scale != 1.0 {
                            let range = range_layout.block_range(idx);
                            for atom in &mut self.atoms {
                                let mut cols =
                                    atom.decoder_coefficients.slice_mut(s![.., range.clone()]);
                                cols.mapv_inplace(|v| v * scale);
                            }
                        }
                    }
                    let aug = stack_augmented_target(anchor, blocks)?;
                    let trial_loss = self.run_joint_fit_arrow_schur(
                        aug.view(),
                        rho,
                        analytic_penalties,
                        inner_max_iter,
                        step_size,
                        ridge_ext_coord,
                        ridge_beta,
                    )?;
                    let (trx, trb_scaled) =
                        self.augmented_block_rss(aug.view(), rho, &range_layout)?;
                    let trb_unscaled: Vec<f64> = (0..blocks.len())
                        .map(|i| trb_scaled[i] / trial_ll[i].exp())
                        .collect();
                    let trial_crit =
                        profiled_reml_criterion(n_obs, px, trx, &trb_unscaled, &dims, &trial_ll);
                    Ok(Some((trial_crit, (trial_ll, trial_loss))))
                },
                |_s, trial_crit| trial_crit < baseline_crit - armijo_eps,
            )?;

            if let Some(step) = accepted_step {
                let (trial_ll, trial_loss) = step.payload;
                cur_log_lambda = trial_ll;
                loss = trial_loss;
            } else {
                // No fraction of the closed-form λ step beats the current-λ
                // baseline: moving λ cannot lower the criterion from here. Reinstate
                // the improved current-λ fit (`baseline_state`, no worse than the
                // sweep-entry state) at the unchanged weights and stop.
                //
                // F4 — honest flag. This branch is reached ONLY after the
                // `max_delta ≤ log_lambda_tol` gate above FAILED, so the closed-form
                // λ* still wants a > tol move: the λ fixed-point iteration did NOT
                // meet its tolerance. A step that then fails the line search is a
                // STALL (the truncated inner solve cannot resolve the tiny
                // coordinated improvement the move needs), not tolerance-
                // convergence. Leaving `converged = false` keeps the flag honest in
                // BOTH directions — `true` only when λ actually settled (or every
                // block became inert), `false` for a stall or a `max_sweeps`
                // exhaustion.
                for idx in 0..blocks.len() {
                    blocks[idx] = blocks[idx].with_log_lambda(cur_log_lambda[idx])?;
                }
                self.fit_state_restore(&baseline_state)?;
                loss = base_loss;
                converged = false;
                for idx in 0..blocks.len() {
                    trajectories[idx].push(cur_log_lambda[idx]);
                }
                break;
            }
            for idx in 0..blocks.len() {
                trajectories[idx].push(cur_log_lambda[idx]);
            }
        }
        // Install the FITTED stacked-column layout (converged per-block weights)
        // so `layer_decoder(k, ℓ)` returns each layer's honest-units decoder with
        // no caller re-slicing. `blocks` now carry the converged `log λ_ℓ`; the
        // width equals `output_dim()` (validated above), so the install succeeds.
        self.set_crosscoder_layout(CrosscoderLayout::from_blocks(px, blocks))?;

        let outcomes = blocks
            .iter()
            .enumerate()
            .map(|(idx, block)| BlockRemlOutcome {
                label: block.label.clone(),
                log_lambda: block.log_lambda,
                identifiable: identifiable[idx],
                trajectory: std::mem::take(&mut trajectories[idx]),
            })
            .collect();
        Ok(MultiBlockRemlFitReport {
            blocks: outcomes,
            sweeps,
            converged,
            loss,
        })
    }

    /// Anchor RSS `R_x` (over the `p_x` anchor columns) and the SCALED per-block
    /// RSS (over each block's column span) of the current fitted residual. The
    /// column offsets come from `layout` (the single owner of the stacked-column
    /// bookkeeping), not a by-hand `off_ℓ` accumulation.
    fn augmented_block_rss(
        &self,
        augmented: ArrayView2<'_, f64>,
        rho: &SaeManifoldRho,
        layout: &CrosscoderLayout,
    ) -> Result<(f64, Vec<f64>), String> {
        let residual = self.reconstruction_residual(augmented, rho)?;
        let px = layout.anchor_dim();
        let mut rss_x = 0.0_f64;
        for row in residual.rows() {
            for j in 0..px {
                let r = row[j];
                rss_x += r * r;
            }
        }
        let mut out = Vec::with_capacity(layout.num_blocks());
        for l in 0..layout.num_blocks() {
            let mut rss = 0.0_f64;
            for row in residual.rows() {
                for j in layout.block_range(l) {
                    let r = row[j];
                    rss += r * r;
                }
            }
            out.push(rss);
        }
        Ok((rss_x, out))
    }

    /// Install the crosscoder stacked-column layout on the term, validating that
    /// its total width matches the atoms' output dimension
    /// (`p_x + Σ_ℓ p_ℓ == output_dim()`). Called by
    /// [`Self::run_multiblock_reml_fit`] with the fitted per-block weights; a
    /// caller may also install one explicitly to enable [`Self::layer_decoder`]
    /// on a term whose decoders were fit elsewhere at the augmented width.
    pub fn set_crosscoder_layout(&mut self, layout: CrosscoderLayout) -> Result<(), String> {
        if layout.total_dim() != self.output_dim() {
            return Err(format!(
                "SaeManifoldTerm::set_crosscoder_layout: layout total width p_x + Σ p_ℓ = {} but \
                 the term's output_dim is {} (atoms must be built at the augmented width)",
                layout.total_dim(),
                self.output_dim()
            ));
        }
        self.crosscoder_layout = Some(layout);
        Ok(())
    }

    /// The installed crosscoder stacked-column layout, or `None` for a term with
    /// no multi-block layout recorded.
    pub fn crosscoder_layout(&self) -> Option<&CrosscoderLayout> {
        self.crosscoder_layout.as_ref()
    }

    /// The HONEST-units per-layer decoder `B_k^(ℓ)` of atom `k`, output block `ℓ`:
    /// the decoder columns of block `ℓ` divided by `√λ_ℓ` (un-doing the target
    /// scaling that `stack_augmented_target` applied). Requires an installed
    /// [`CrosscoderLayout`] (via a multi-block fit or [`Self::set_crosscoder_layout`]).
    ///
    /// This is the first-class form of the by-hand slice+unscale
    /// (`decoder_coefficients[:, off_ℓ..off_ℓ+p_ℓ] / √λ_ℓ`): identical values, but
    /// the offsets and `√λ_ℓ` are owned by the layout so no caller recomputes them.
    pub fn layer_decoder(&self, k: usize, l: usize) -> Result<Array2<f64>, String> {
        let layout = self.crosscoder_layout.as_ref().ok_or_else(|| {
            "SaeManifoldTerm::layer_decoder: no crosscoder layout installed (run \
             run_multiblock_reml_fit or set_crosscoder_layout first)"
                .to_string()
        })?;
        if k >= self.atoms.len() {
            return Err(format!(
                "SaeManifoldTerm::layer_decoder: atom index k={k} out of range (K = {})",
                self.atoms.len()
            ));
        }
        if l >= layout.num_blocks() {
            return Err(format!(
                "SaeManifoldTerm::layer_decoder: block index ℓ={l} out of range (L-1 = {})",
                layout.num_blocks()
            ));
        }
        let inv = 1.0 / layout.sqrt_lambda(l);
        // Materialize the full-width decoder before crossing the layer boundary;
        // the fit may use a reduced Grassmann coordinate internally.
        let physical = self.atoms[k].full_width_decoder();
        let scaled = physical.slice(s![.., layout.block_range(l)]);
        Ok(scaled.mapv(|value| inv * value))
    }

    /// Snapshot the ENTIRE mutable fit state a λ line-search trial perturbs, so a
    /// rejected trial rolls back to a bit-identical base.
    ///
    /// A trial runs a full [`Self::run_joint_fit_arrow_schur`], which moves far
    /// more than the decoder β and latent coords: it refreshes each atom's cached
    /// `basis_values` / `basis_jacobian` and (lagged-diffusivity) `smooth_penalty`,
    /// rewrites the assignment `logits`, re-derives the active-set `last_row_layout`,
    /// and advances the Gumbel `temperature_schedule` one anneal step per inner
    /// iteration. The canonical [`Self::snapshot_mutable_state`] captures the first
    /// group (the same state the inner damped-Newton line search itself rolls back);
    /// the schedule is a per-call *stateful* counter, so a rejected trial that
    /// leaves it advanced would start the next trial at a colder temperature and
    /// make the backtracking trials incomparable — capture it here too. (An atom
    /// rank reduction, [`Self::reduce_atoms_to_data_supported_rank`], is idempotent
    /// after the sweep-entry fit, so the restored decoder width always matches.)
    fn fit_state_snapshot(&self) -> (SaeManifoldMutableState, Option<GumbelTemperatureSchedule>) {
        (
            self.snapshot_mutable_state(),
            self.temperature_schedule.clone(),
        )
    }

    /// Restore a [`Self::fit_state_snapshot`]. `restore_mutable_state` rebuilds
    /// each atom's `basis_values` / `basis_jacobian` from the restored coordinates
    /// (the differential snapshot stores only the cheap driving state), so the
    /// restored state is immediately consistent for a residual/fitted read.
    fn fit_state_restore(
        &mut self,
        snap: &(SaeManifoldMutableState, Option<GumbelTemperatureSchedule>),
    ) -> Result<(), String> {
        self.restore_mutable_state(&snap.0)?;
        self.temperature_schedule.clone_from(&snap.1);
        Ok(())
    }
}

impl SaeManifoldTerm {
    /// Run the Rung-2 two-block joint fit with `λ_y` selected by REML.
    ///
    /// Requires a [`BehaviorBlock`] installed via
    /// [`Self::set_behavior_block`]; `activation` is the raw activation target
    /// `Z` (`n × p_x`) — the augmented target is stacked internally at each
    /// sweep's current `λ_y`. `rho`, `analytic_penalties`, and the inner knobs
    /// carried in `controls` (`inner_max_iter`, `step_size`, and the two ridges)
    /// are passed through to [`Self::run_joint_fit_arrow_schur`] unchanged.
    ///
    /// `controls.max_sweeps` bounds the outer (fit, λ-update) alternation;
    /// `controls.log_lambda_tol` is the convergence tolerance on `|Δ log λ_y|`
    /// (see [`TwoBlockRemlControls`]: all of these are caller-owned resolution
    /// choices — not fit hyperparameters. The *destination* is the
    /// data-determined REML stationary point; these only bound how long we walk
    /// toward it).
    ///
    /// On return the term holds the fitted two-block state at the selected
    /// `λ_y` (its behavior block updated in place), so
    /// [`BehaviorBlock::split_decoder`] on the fitted decoders yields the
    /// activation decoder `B_k` and the nats-unit behavior decoder `C_k`.
    pub fn run_two_block_reml_fit(
        &mut self,
        activation: ArrayView2<'_, f64>,
        rho: &mut SaeManifoldRho,
        analytic_penalties: Option<&AnalyticPenaltyRegistry>,
        controls: TwoBlockRemlControls,
    ) -> Result<TwoBlockRemlFitReport, String> {
        // Two-block-specific validation and error messages, preserved verbatim
        // for existing callers; the fit itself is the `K = 2` (single output
        // block) special case of the block-generic driver, which it delegates to
        // so the two paths are bit-for-bit identical by construction.
        if controls.max_sweeps == 0 {
            return Err(
                "SaeManifoldTerm::run_two_block_reml_fit: max_sweeps must be ≥ 1".to_string(),
            );
        }
        if !(controls.log_lambda_tol.is_finite() && controls.log_lambda_tol > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::run_two_block_reml_fit: log_lambda_tol must be finite and \
                 positive; got {}",
                controls.log_lambda_tol
            ));
        }
        let Some(block) = self.behavior_block().cloned() else {
            return Err(
                "SaeManifoldTerm::run_two_block_reml_fit: no behavior block installed \
                 (call set_behavior_block first)"
                    .to_string(),
            );
        };
        if activation.ncols() != block.activation_dim {
            return Err(format!(
                "SaeManifoldTerm::run_two_block_reml_fit: activation has {} columns; behavior \
                 block declares p_x = {}",
                activation.ncols(),
                block.activation_dim
            ));
        }

        // The behavior block IS an output block: its (unscaled) tangent target,
        // its width p_y, and its weight λ_y drive the same augmented fit and the
        // same variance-ratio update.
        let mut output_blocks = vec![OutputBlock::new(
            "behavior",
            block.target.clone(),
            block.log_lambda_y,
        )?];
        let report = self.run_multiblock_reml_fit(
            activation,
            &mut output_blocks,
            rho,
            analytic_penalties,
            controls,
        )?;

        // Keep the term's installed behavior block in sync with the selected
        // weight — the two-block contract that `behavior_block()` (and hence
        // `split_decoder` / `augmented_target`) reflects the converged λ_y.
        let fitted_block = block.with_log_lambda_y(output_blocks[0].log_lambda)?;
        self.set_behavior_block(fitted_block)?;

        let outcome = &report.blocks[0];
        Ok(TwoBlockRemlFitReport {
            log_lambda_y: outcome.log_lambda,
            sweeps: report.sweeps,
            converged: report.converged,
            lambda_identifiable: outcome.identifiable,
            loss: report.loss,
            log_lambda_trajectory: outcome.trajectory.clone(),
        })
    }
}

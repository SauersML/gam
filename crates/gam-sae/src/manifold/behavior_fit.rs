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

        let mut trajectories: Vec<Vec<f64>> = vec![Vec::with_capacity(max_sweeps); blocks.len()];
        let mut identifiable = vec![true; blocks.len()];
        let mut loss: Option<SaeManifoldLoss> = None;
        let mut converged = false;
        let mut sweeps = 0usize;

        while sweeps < max_sweeps {
            sweeps += 1;
            // Fit at the current weights.
            let augmented = stack_augmented_target(anchor, blocks)?;
            let sweep_loss = self.run_joint_fit_arrow_schur(
                augmented.view(),
                rho,
                analytic_penalties,
                inner_max_iter,
                step_size,
                ridge_ext_coord,
                ridge_beta,
            )?;
            loss = Some(sweep_loss);

            // Residual in the scaled augmented units.
            let residual = self.reconstruction_residual(augmented.view(), rho)?;
            // Anchor RSS over the p_x anchor columns (row-major, one accumulator
            // — identical order to the two-block driver's rss_x).
            let mut rss_x = 0.0_f64;
            for row in residual.rows() {
                for j in 0..px {
                    let r = row[j];
                    rss_x += r * r;
                }
            }
            // Per-block scaled RSS over each block's own column span.
            let mut block_rss = Vec::with_capacity(blocks.len());
            let mut offset = px;
            for block in blocks.iter() {
                let pl = block.block_dim();
                let mut rss = 0.0_f64;
                for row in residual.rows() {
                    for j in offset..offset + pl {
                        let r = row[j];
                        rss += r * r;
                    }
                }
                block_rss.push(rss);
                offset += pl;
            }

            // Closed-form λ_ℓ update per block; hold unidentifiable blocks.
            let mut max_delta = 0.0_f64;
            let mut any_update = false;
            for (idx, block) in blocks.iter_mut().enumerate() {
                if !identifiable[idx] {
                    trajectories[idx].push(block.log_lambda);
                    continue;
                }
                match block.reml_updated_log_lambda(rss_x, px, block_rss[idx]) {
                    Ok(new_log_lambda) => {
                        let delta = (new_log_lambda - block.log_lambda).abs();
                        max_delta = max_delta.max(delta);
                        any_update = true;
                        *block = block.with_log_lambda(new_log_lambda)?;
                        trajectories[idx].push(new_log_lambda);
                    }
                    Err(_) => {
                        // No residual variance ⇒ λ_ℓ unidentifiable from this fit.
                        // Hold this block; other blocks may still move.
                        identifiable[idx] = false;
                        trajectories[idx].push(block.log_lambda);
                    }
                }
            }

            if !any_update {
                // Every block is held ⇒ no weight can move; stop without a refit
                // (mirrors the two-block unidentifiable path).
                converged = true;
                break;
            }
            if max_delta <= log_lambda_tol {
                converged = true;
                // One final fit at the converged weights so the term state
                // matches the reported weights exactly.
                let augmented = stack_augmented_target(anchor, blocks)?;
                let final_loss = self.run_joint_fit_arrow_schur(
                    augmented.view(),
                    rho,
                    analytic_penalties,
                    inner_max_iter,
                    step_size,
                    ridge_ext_coord,
                    ridge_beta,
                )?;
                loss = Some(final_loss);
                break;
            }
        }

        let loss = loss.expect("max_sweeps ≥ 1 guarantees at least one fit");
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
        let mut output_blocks =
            vec![OutputBlock::new("behavior", block.target.clone(), block.log_lambda_y)?];
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

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

impl SaeManifoldTerm {
    /// Run the Rung-2 two-block joint fit with `λ_y` selected by REML.
    ///
    /// Requires a [`BehaviorBlock`] installed via
    /// [`Self::set_behavior_block`]; `activation` is the raw activation target
    /// `Z` (`n × p_x`) — the augmented target is stacked internally at each
    /// sweep's current `λ_y`. `rho`, `analytic_penalties`, `inner_max_iter`,
    /// `step_size`, and the two ridges are passed through to
    /// [`Self::run_joint_fit_arrow_schur`] unchanged.
    ///
    /// `max_sweeps` bounds the outer (fit, λ-update) alternation;
    /// `log_lambda_tol` is the convergence tolerance on `|Δ log λ_y|` (both are
    /// caller-owned resolution choices, like `inner_max_iter` — not fit
    /// hyperparameters: the *destination* is the data-determined REML
    /// stationary point, these only bound how long we walk toward it).
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
        max_sweeps: usize,
        inner_max_iter: usize,
        step_size: f64,
        ridge_ext_coord: f64,
        ridge_beta: f64,
        log_lambda_tol: f64,
    ) -> Result<TwoBlockRemlFitReport, String> {
        if max_sweeps == 0 {
            return Err(
                "SaeManifoldTerm::run_two_block_reml_fit: max_sweeps must be ≥ 1".to_string(),
            );
        }
        if !(log_lambda_tol.is_finite() && log_lambda_tol > 0.0) {
            return Err(format!(
                "SaeManifoldTerm::run_two_block_reml_fit: log_lambda_tol must be finite and \
                 positive; got {log_lambda_tol}"
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

        let mut block = block;
        let mut trajectory = Vec::with_capacity(max_sweeps);
        let mut loss: Option<SaeManifoldLoss> = None;
        let mut converged = false;
        let mut lambda_identifiable = true;
        let mut sweeps = 0usize;

        while sweeps < max_sweeps {
            sweeps += 1;
            // Fit at the current weight.
            let augmented = block.augmented_target(activation)?;
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

            // Closed-form REML λ_y update at the fitted residuals.
            let residual = self.reconstruction_residual(augmented.view(), rho)?;
            let new_log_lambda = match block.reml_updated_log_lambda_y(residual.view()) {
                Ok(value) => value,
                Err(_) => {
                    // Behavior residual has no variance ⇒ λ_y unidentifiable
                    // from this fit (behaviorally inert block). Hold the weight
                    // and stop: further sweeps would refit the same problem.
                    lambda_identifiable = false;
                    converged = true;
                    trajectory.push(block.log_lambda_y);
                    break;
                }
            };
            let delta = (new_log_lambda - block.log_lambda_y).abs();
            trajectory.push(new_log_lambda);
            block = block.with_log_lambda_y(new_log_lambda)?;
            // Keep the term's installed block in sync with the weight the NEXT
            // stack (and any post-fit split_decoder consumer) will see.
            self.set_behavior_block(block.clone())?;
            if delta <= log_lambda_tol {
                converged = true;
                // One final fit at the converged weight so the term state
                // matches the reported λ_y exactly.
                let augmented = block.augmented_target(activation)?;
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

        // In the unidentifiable-stop path the installed block was never
        // replaced this sweep; make sure the term still carries it.
        if !lambda_identifiable {
            self.set_behavior_block(block.clone())?;
        }

        let loss = loss.expect("max_sweeps ≥ 1 guarantees at least one fit");
        Ok(TwoBlockRemlFitReport {
            log_lambda_y: block.log_lambda_y,
            sweeps,
            converged,
            lambda_identifiable,
            loss,
            log_lambda_trajectory: trajectory,
        })
    }
}

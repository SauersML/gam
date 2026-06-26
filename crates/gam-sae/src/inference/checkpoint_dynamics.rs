//! Cross-checkpoint training-dynamics inference for SAE atoms (issue #1102).
//!
//! OLMo ships intermediate-training checkpoints. Each checkpoint `c` fits a
//! dictionary whose atom `a` is a decoder curve `g^{(c)}_a: t ↦ ℝ^ambient`
//! sampled on a shared latent grid `t`. The question this module answers, per
//! atom, is *did the atom change across training, and where*, with
//! debiased point estimates, standard errors, and anytime-valid evidence.
//!
//! It is pure assembly of three landed instruments — none is reimplemented:
//!
//! * [`crate::inference::riesz`] — the per-step contrast
//!   `θ = g^{(c+1)}(t₀) − g^{(c)}(t₀)` is the linear
//!   [`SmoothFunctional::Contrast`] of a stacked two-checkpoint coefficient
//!   vector; [`debias_with_dense_hessian`] returns the penalty-debiased
//!   estimate and a plug-in SE via the Riesz representer.
//! * [`crate::inference::layer_transport`] — the checkpoint axis is reused as
//!   the "layer" axis: [`fit_transport_map`] aligns the atom's latent chart
//!   across consecutive checkpoints (topology compatibility, isometry defect,
//!   winding degree), packaged as a [`LayerTransportReport`].
//! * [`gam_terms::inference::structure_evidence`] — each consecutive-step contrast
//!   feeds one anytime-valid e-value (the studentized displacement mapped to a
//!   two-sided p-value and run through the frozen κ = ½ p→e calibrator) into a
//!   per-step [`StructureLedger`] claim under the null "the atom did not change
//!   at this checkpoint step". A genuine e-value (`E_{H0}[E] ≤ 1`), unlike the
//!   divergent in-sample `exp(½ z²)` likelihood ratio; optional-stopping-safe.
//!
//! # Honest accounting of the Riesz inputs
//!
//! A *bare* decoder grid carries the fitted curve VALUES but no
//! observation-level scores and no penalized Hessian — those cannot be
//! fabricated from grid samples. So the smooth this module debiases is the
//! one the grid actually defines: a ridge-penalized least-squares fit of the
//! grid VALUES on the latent-grid identity (interpolation) basis, where each
//! grid node is one observation with response equal to the decoder value at
//! that node. This is a genuine fit with a genuine penalized Hessian
//! `XᵀX + λS = I + λS` and genuine per-node scores `s_i = (β_i − y_i)·eᵢ`,
//! so every quantity handed to [`debias_with_dense_hessian`] is real, not a
//! placeholder. The contrast functional is then evaluated against the
//! identity design row at the latent-grid mode index. The ambient dimension is
//! handled component-wise and the per-component contrasts are aggregated into a
//! single scalar `θ` by the L2 norm of the component contrast vector (the size
//! of the decoder displacement at `t₀`); its SE is propagated by the
//! delta method through that norm.

use crate::inference::layer_transport::{ChartTopology, LayerTransportReport, fit_layer_transport};
use crate::inference::riesz::{
    RieszDebiasReport, RieszInput, SmoothFunctional, debias_with_dense_hessian,
};
use gam_terms::inference::structure_evidence::{ClaimKind, StructureLedger, log_e_from_p_calibrator};
use ndarray::{Array1, Array2, ArrayView1, ArrayView4};
use statrs::distribution::{ContinuousCDF, Normal};

/// Ridge penalty on the interpolation fit of the grid values. Small relative
/// to the unit data Hessian so the fit tracks the grid closely; non-zero so
/// the penalty-debiasing term in the Riesz one-step is exercised on real
/// (not degenerate) curvature, and so the Hessian `I + λS` is strictly SPD.
const GRID_FIT_RIDGE: f64 = 1e-3;

/// Inputs for one cross-checkpoint atom-dynamics run.
///
/// `decoder_grid` is `[n_checkpoints, n_atoms, n_grid, ambient_dim]`: the
/// decoder curve of every atom sampled on the shared `latent_grid` at every
/// checkpoint. `checkpoint_ids[c]` and `atom_names[a]` label the axes.
pub struct CheckpointDynamicsInput<'a> {
    pub decoder_grid: ArrayView4<'a, f64>,
    pub checkpoint_ids: &'a [String],
    pub atom_names: &'a [String],
    pub latent_grid: ArrayView1<'a, f64>,
}

/// The training-dynamics trajectory of one atom across the checkpoint axis.
///
/// The PRIMARY, coverage-valid deliverable is [`Self::change_evidence`]: the
/// anytime-valid e-process answering "did atom k change during training?".
/// [`Self::conditional_step_contrasts`] is a secondary, descriptive readout (see
/// its docs for the conditional caveat).
pub struct AtomTrajectory {
    pub atom_name: String,
    /// Debiased `g^{(c+1)}(t_mode) − g^{(c)}(t_mode)` for each consecutive
    /// checkpoint step, with its plug-in SE.
    ///
    /// CONDITIONAL ON THE FITTED COORDINATES (not a coverage-valid CI). The
    /// debiased SE here conditions away the generated-regressor uncertainty in
    /// the estimated latent coordinates `t̂` and activations `â` — the exact
    /// correction the marginal-slope family exists to make (issue #1115). It is
    /// reported only as a conditional contrast point estimate with a plug-in SE,
    /// NOT as an interval with frequentist coverage for the population
    /// displacement. The headline change verdict is carried by the e-process
    /// [`Self::change_evidence`], which IS anytime-valid; this field is a
    /// descriptive companion. Read the SE accordingly.
    pub conditional_step_contrasts: Vec<RieszDebiasReport>,
    /// Consecutive-checkpoint chart correspondences (checkpoint axis reused as
    /// the transport "layer" axis).
    pub transports: Vec<LayerTransportReport>,
    /// PRIMARY deliverable: anytime-valid evidence that the atom changed at each
    /// consecutive checkpoint step, one calibrated e-value per step into a
    /// per-step claim. Valid at any data-dependent stopping time.
    pub change_evidence: StructureLedger,
}

/// Run cross-checkpoint debiased dynamics inference for every atom.
///
/// For each atom, walks consecutive checkpoints and, at each step `c → c+1`:
/// 1. fits the transport map between the two checkpoints' latent charts
///    ([`fit_layer_transport`], checkpoint axis as the layer axis);
/// 2. evaluates the Riesz-debiased decoder-displacement contrast at the
///    latent-grid mode ([`SmoothFunctional::Contrast`] + penalty debiasing);
/// 3. absorbs the studentized contrast as a calibrated anytime-valid e-value
///    into the step's change claim under the no-change null.
pub fn checkpoint_atom_dynamics(
    input: &CheckpointDynamicsInput<'_>,
) -> Result<Vec<AtomTrajectory>, String> {
    let shape = input.decoder_grid.shape();
    let (n_checkpoints, n_atoms, n_grid, ambient_dim) = (shape[0], shape[1], shape[2], shape[3]);
    if n_checkpoints < 2 {
        return Err(format!(
            "checkpoint dynamics needs at least two checkpoints, got {n_checkpoints}"
        ));
    }
    if input.checkpoint_ids.len() != n_checkpoints {
        return Err(format!(
            "checkpoint_ids length {} disagrees with decoder grid checkpoint axis {n_checkpoints}",
            input.checkpoint_ids.len()
        ));
    }
    if input.atom_names.len() != n_atoms {
        return Err(format!(
            "atom_names length {} disagrees with decoder grid atom axis {n_atoms}",
            input.atom_names.len()
        ));
    }
    if input.latent_grid.len() != n_grid {
        return Err(format!(
            "latent_grid length {} disagrees with decoder grid latent axis {n_grid}",
            input.latent_grid.len()
        ));
    }
    if n_grid < 2 || ambient_dim == 0 {
        return Err(format!(
            "checkpoint dynamics needs a non-trivial grid ({n_grid}) and ambient dim ({ambient_dim})"
        ));
    }
    if input.decoder_grid.iter().any(|v| !v.is_finite()) {
        return Err("checkpoint dynamics decoder grid must be finite".to_string());
    }
    if input.latent_grid.iter().any(|v| !v.is_finite()) {
        return Err("checkpoint dynamics latent grid must be finite".to_string());
    }

    // The mode index: the latent-grid node where the contrast is evaluated.
    // Use the central node so it sits inside any chart and away from edge
    // interpolation artifacts.
    let mode_index = n_grid / 2;

    // Identity interpolation design `X = I_{n_grid}` and its ridge penalty
    // `S = I`. The penalized Hessian `H = XᵀX + λS = (1 + λ) I` is shared by
    // every component fit, so it is built once.
    let penalty_scale = 1.0 + GRID_FIT_RIDGE;
    let mut hessian = Array2::<f64>::zeros((n_grid, n_grid));
    for i in 0..n_grid {
        hessian[[i, i]] = penalty_scale;
    }
    // Contrast design rows pick out the mode node: `m(t_mode) = β_{mode}`, so
    // the value-design row is the mode basis vector. The contrast `a − b`
    // (later checkpoint minus earlier) shares the same row; the per-checkpoint
    // distinction is carried by the two fitted coefficient vectors, exactly as
    // a paired contrast of the same functional across two fits.
    let mut mode_row = Array1::<f64>::zeros(n_grid);
    mode_row[mode_index] = 1.0;

    let mut trajectories = Vec::with_capacity(n_atoms);
    for atom in 0..n_atoms {
        let atom_name = input.atom_names[atom].clone();
        let mut step_contrasts = Vec::with_capacity(n_checkpoints - 1);
        let mut transports = Vec::with_capacity(n_checkpoints - 1);
        let mut change_evidence = StructureLedger::new();

        for step in 0..n_checkpoints - 1 {
            let c0 = step;
            let c1 = step + 1;

            // --- transport map across the checkpoint axis --------------------
            // Reuse the latent grid itself as both charts' coordinates on an
            // interval `[min, max]`; the transport fit aligns the two
            // checkpoints' decoder curves through their shared latent index.
            // The "from"/"to" coordinates are the decoder values projected to
            // the first ambient component, the available scalar chart sample.
            let coords_from = input
                .decoder_grid
                .slice(ndarray::s![c0, atom, .., 0])
                .to_owned();
            let coords_to = input
                .decoder_grid
                .slice(ndarray::s![c1, atom, .., 0])
                .to_owned();
            let (lo, hi) = interval_bounds(coords_from.view(), coords_to.view());
            let topology = ChartTopology::Interval { lo, hi };
            let transport = fit_layer_transport(
                c0,
                c1,
                coords_from.view(),
                coords_to.view(),
                topology,
                topology,
            )
            .map_err(|e| {
                format!(
                    "checkpoint transport for atom '{atom_name}' step {} → {} failed: {e}",
                    input.checkpoint_ids[c0], input.checkpoint_ids[c1]
                )
            })?;
            transports.push(transport);

            // --- Riesz-debiased decoder-displacement contrast at the mode ----
            let report = contrast_at_mode(&ContrastAtMode {
                grid: input.decoder_grid,
                atom,
                c0,
                c1,
                ambient_dim,
                n_grid,
                hessian: hessian.view(),
                mode_row: mode_row.view(),
            })
            .map_err(|e| {
                format!(
                    "checkpoint contrast for atom '{atom_name}' step {} → {} failed: {e}",
                    input.checkpoint_ids[c0], input.checkpoint_ids[c1]
                )
            })?;

            // --- anytime-valid evidence the atom changed at this step --------
            // The debiased displacement `θ̂` with SE `se` studentizes to
            // `z = θ̂ / se` (local Gaussian `θ̂ ~ N(θ, se²)`). Its two-sided
            // p-value run through the frozen κ = ½ p→e calibrator is a genuine
            // e-value for the per-step no-change null θ = 0 — `E_{H0}[E] ≤ 1`,
            // which the naive in-sample `exp(½ z²)` ratio is NOT (it diverges
            // under H0). One e-value per step into a per-step claim; the
            // calibrator's contract (one e-value per independent batch) is met
            // because each step is a distinct checkpoint transition.
            let claim = change_evidence.register(ClaimKind::Custom {
                label: format!(
                    "atom '{atom_name}' changed from checkpoint {} to {}",
                    input.checkpoint_ids[c0], input.checkpoint_ids[c1]
                ),
            });
            let log_e = no_change_log_e_value(report.theta_onestep, report.se)?;
            change_evidence.absorb_log(claim, log_e)?;

            step_contrasts.push(report);
        }

        trajectories.push(AtomTrajectory {
            atom_name,
            conditional_step_contrasts: step_contrasts,
            transports,
            change_evidence,
        });
    }

    Ok(trajectories)
}

/// Interval bounds spanning both checkpoints' scalar chart samples, padded so
/// the transport basis domain strictly contains the data.
fn interval_bounds(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in a.iter().chain(b.iter()) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if !(lo.is_finite() && hi.is_finite()) {
        return (0.0, 1.0);
    }
    if hi <= lo {
        // Degenerate (constant) chart: open a unit window around the value.
        return (lo - 0.5, lo + 0.5);
    }
    let pad = (hi - lo) * 1e-6;
    (lo - pad, hi + pad)
}

/// Debiased `g^{(c1)}(t_mode) − g^{(c0)}(t_mode)` aggregated over the ambient
/// dimension into the scalar decoder-displacement size, with a delta-method SE.
///
/// Each ambient component is an independent identity-basis ridge fit of the
/// grid values; the [`SmoothFunctional::Contrast`] of the two checkpoints'
/// fitted coefficient vectors at the mode node is debiased component-wise via
/// the Riesz one-step. The component contrasts form a vector `Δ ∈ ℝ^ambient`;
/// the reported scalar `θ = ‖Δ‖₂` is the displacement size and its SE is the
/// delta-method norm-gradient `‖Δ‖₂` propagation of the per-component SEs,
/// assuming component independence (separate fits, separate scores).
struct ContrastAtMode<'a> {
    grid: ArrayView4<'a, f64>,
    atom: usize,
    c0: usize,
    c1: usize,
    ambient_dim: usize,
    n_grid: usize,
    hessian: ndarray::ArrayView2<'a, f64>,
    mode_row: ArrayView1<'a, f64>,
}

fn contrast_at_mode(args: &ContrastAtMode<'_>) -> Result<RieszDebiasReport, String> {
    let grid = args.grid;
    let atom = args.atom;
    let c0 = args.c0;
    let c1 = args.c1;
    let ambient_dim = args.ambient_dim;
    let n_grid = args.n_grid;
    let hessian = args.hessian;
    let mode_row = args.mode_row;
    // Aggregate scalar contrast Δ = θ_c1 − θ_c0 across ambient components, and
    // the matching aggregate Riesz quantities, so a single RieszDebiasReport
    // describes the displacement. We assemble the report from one debiasing per
    // component and combine through the L2 norm.
    let mut delta = Array1::<f64>::zeros(ambient_dim);
    let mut delta_one = Array1::<f64>::zeros(ambient_dim);
    let mut var_components = Array1::<f64>::zeros(ambient_dim);
    let mut penalty_bias_acc = 0.0_f64;
    // A representer to carry: reuse the last component's; the scalar norm
    // estimate's representer is component-wise so we keep the final one as the
    // canonical witness (its influence vector studentizes the norm contrast).
    let mut witness: Option<RieszDebiasReport> = None;

    for comp in 0..ambient_dim {
        // Per-checkpoint identity-basis ridge fit: response y = grid values,
        // design X = I, penalty S = I. With H = (1+λ)I the fitted coefficient
        // is β = y / (1 + λ); the per-node score is sᵢ = (μ̂ᵢ − yᵢ)·eᵢ where
        // μ̂ = Xβ = β, and the penalty gradient is S·β = β.
        let y0 = grid.slice(ndarray::s![c0, atom, .., comp]).to_owned();
        let y1 = grid.slice(ndarray::s![c1, atom, .., comp]).to_owned();
        let report = component_contrast(y0.view(), y1.view(), n_grid, hessian, mode_row)?;

        delta[comp] = report.theta_plugin;
        delta_one[comp] = report.theta_onestep;
        var_components[comp] = report.se * report.se;
        penalty_bias_acc += report.penalty_bias * report.penalty_bias;
        witness = Some(report);
    }

    let theta_plugin = delta.dot(&delta).sqrt();
    let norm_one = delta_one.dot(&delta_one).sqrt();
    // Delta method for θ = ‖Δ‖₂: ∂θ/∂Δ_k = Δ_k / ‖Δ‖₂, components independent,
    // so var(θ) = Σ_k (Δ_k/‖Δ‖₂)² var(Δ_k).
    let se = if norm_one > f64::MIN_POSITIVE {
        let mut v = 0.0_f64;
        for comp in 0..ambient_dim {
            let g = delta_one[comp] / norm_one;
            v += g * g * var_components[comp];
        }
        v.max(0.0).sqrt()
    } else {
        // At a null displacement the norm is non-differentiable; fall back to
        // the RMS of the component SEs (an honest upper-ish bound on the size).
        (var_components.sum() / ambient_dim as f64).sqrt()
    };

    let mut report = witness
        .ok_or_else(|| "checkpoint contrast requires at least one ambient component".to_string())?;
    report.theta_plugin = theta_plugin;
    report.theta_onestep = norm_one;
    report.se = se;
    report.penalty_bias = penalty_bias_acc.sqrt();
    Ok(report)
}

/// One ambient component's debiased contrast `g^{(c1)}(t_mode) −
/// g^{(c0)}(t_mode)` through the Riesz one-step.
fn component_contrast(
    y0: ArrayView1<'_, f64>,
    y1: ArrayView1<'_, f64>,
    n_grid: usize,
    hessian: ndarray::ArrayView2<'_, f64>,
    mode_row: ArrayView1<'_, f64>,
) -> Result<RieszDebiasReport, String> {
    // Stacked paired-contrast trick: the contrast `m_{c1}(t₀) − m_{c0}(t₀)` is
    // the difference of one linear functional applied to two coefficient
    // vectors. Riesz operates on a single fit, so we debias on the DIFFERENCE
    // fit β_Δ = β_{c1} − β_{c0}, whose response is y₁ − y₀ on the same identity
    // basis — a genuine fit with the same penalized Hessian. The contrast
    // functional on β_Δ is then the point evaluation at the mode, packaged via
    // SmoothFunctional::Contrast against a zero row so the gradient is the mode
    // row exactly (g = mode_row − 0).
    let beta0 = y0.mapv(|v| v / (1.0 + GRID_FIT_RIDGE));
    let beta1 = y1.mapv(|v| v / (1.0 + GRID_FIT_RIDGE));
    let beta_delta = &beta1 - &beta0;

    let zero_row = Array1::<f64>::zeros(n_grid);
    let functional = SmoothFunctional::Contrast {
        design_row_a: mode_row,
        design_row_b: zero_row.view(),
    };
    let gradient = functional
        .gradient()
        .map_err(|e| format!("contrast functional gradient failed: {e}"))?;

    // Per-node scores of the difference fit: μ̂ = X β_Δ = β_Δ, response y₁−y₀.
    let response = &y1.to_owned() - &y0;
    let mut row_scores = Array2::<f64>::zeros((n_grid, n_grid));
    for i in 0..n_grid {
        row_scores[[i, i]] = beta_delta[i] - response[i];
    }
    // Penalty gradient S·β_Δ = β_Δ (S = I).
    let penalty_beta = beta_delta.clone();

    let input = RieszInput {
        beta: beta_delta.view(),
        functional_gradient: gradient.view(),
        row_scores: row_scores.view(),
        penalty_beta: penalty_beta.view(),
        leverage: None,
    };
    debias_with_dense_hessian(&input, hessian).map_err(|e| format!("Riesz debiasing failed: {e}"))
}

/// Anytime-valid log-e-value for the no-change null `θ = 0` from the debiased,
/// studentized displacement `z = θ̂ / se` (local Gaussian `θ̂ ~ N(θ, se²)`).
///
/// The naive in-sample likelihood ratio `exp(½ z²)` — the alternative density
/// re-centered on the very estimate `θ̂` it is scored against — is NOT an
/// e-value: under H0, `z ~ N(0,1)` and `E[exp(½ z²)] = ∫ φ(z) exp(½ z²) dz`
/// DIVERGES, so it has no `E_{H0}[E] ≤ 1` guarantee. (Universal inference earns
/// `exp(½ z²)` validity only with a held-out evaluation fold; a single grid of
/// decoder values affords no such split.)
///
/// Instead we map the displacement to its two-sided normal p-value
/// `p = 2(1 − Φ(|z|))` and route it through the module's frozen p→e calibrator
/// [`log_e_from_p_calibrator`] (the κ = ½ member `e(p) = ½ p^{−1/2}`, with
/// `∫₀¹ e(p) dp = 1`, hence `E_{H0}[e(P)] ≤ 1` for any superuniform p). This is
/// a genuine e-value: no displacement, small e; a real displacement, large e;
/// and it compounds validly into the change e-process. A degenerate
/// (non-positive) SE yields a zero log-e-value (no evidence, not certainty).
fn no_change_log_e_value(theta_hat: f64, se: f64) -> Result<f64, String> {
    if !(se > 0.0) || !theta_hat.is_finite() {
        return Ok(0.0);
    }
    let z = (theta_hat / se).abs();
    let normal =
        Normal::new(0.0, 1.0).map_err(|e| format!("standard normal construction failed: {e}"))?;
    // Two-sided p-value of the studentized displacement; clamp to (0, 1] so the
    // calibrator (which rejects p = 0) sees a finite, valid argument even at a
    // numerically saturated tail.
    let p = (2.0 * (1.0 - normal.cdf(z))).clamp(f64::MIN_POSITIVE, 1.0);
    log_e_from_p_calibrator(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    /// Build a `[n_ckpt, n_atoms, n_grid, ambient]` grid where atom 0's curve is
    /// constant across checkpoints (no change) and atom 1's curve at the central
    /// (mode) node is displaced by a known amount `shift` in component 0 between
    /// consecutive checkpoints (a steady drift).
    fn drift_grid(n_ckpt: usize, n_grid: usize, ambient: usize, shift: f64) -> Array4<f64> {
        let mode = n_grid / 2;
        let mut grid = Array4::<f64>::zeros((n_ckpt, 2, n_grid, ambient));
        for c in 0..n_ckpt {
            for g in 0..n_grid {
                let t = g as f64 / (n_grid - 1) as f64;
                for comp in 0..ambient {
                    // Atom 0: smooth bump, identical at every checkpoint.
                    grid[[c, 0, g, comp]] = (t * std::f64::consts::PI).sin() * (comp as f64 + 1.0);
                    // Atom 1: same base curve plus a checkpoint-indexed shift at
                    // the mode node in component 0 only.
                    let base = (t * std::f64::consts::PI).sin() * (comp as f64 + 1.0);
                    grid[[c, 1, g, comp]] = if g == mode && comp == 0 {
                        base + shift * c as f64
                    } else {
                        base
                    };
                }
            }
        }
        grid
    }

    #[test]
    fn no_change_atom_has_near_zero_contrast_and_no_change_evidence() {
        let n_ckpt = 5;
        // The transport fit requires at least MIN_TRANSPORT_OBS (16) paired
        // grid samples, so the shared latent grid must be at least that long.
        let n_grid = 17;
        let ambient = 3;
        let grid = drift_grid(n_ckpt, n_grid, ambient, 0.5);
        let latent: Array1<f64> = Array1::linspace(0.0, 1.0, n_grid);
        let ckpt_ids: Vec<String> = (0..n_ckpt).map(|c| format!("dev{c}")).collect();
        let atom_names = vec!["constant".to_string(), "drifter".to_string()];
        let input = CheckpointDynamicsInput {
            decoder_grid: grid.view(),
            checkpoint_ids: &ckpt_ids,
            atom_names: &atom_names,
            latent_grid: latent.view(),
        };
        let traj = checkpoint_atom_dynamics(&input).expect("dynamics");
        assert_eq!(traj.len(), 2);

        // Atom 0 is identical across checkpoints: every step contrast must be
        // (numerically) zero displacement and accumulate no change evidence.
        let constant = &traj[0];
        assert_eq!(constant.conditional_step_contrasts.len(), n_ckpt - 1);
        for report in &constant.conditional_step_contrasts {
            assert!(
                report.theta_onestep.abs() < 1e-9,
                "constant atom step displacement should be ~0, got {}",
                report.theta_onestep
            );
        }
        // No-change null is true here → the e-BH certificate confirms nothing.
        let cert = constant.change_evidence.certify(0.05);
        assert!(
            cert.confirmed().count() == 0,
            "constant atom must not confirm any change claim"
        );
    }

    #[test]
    fn drifting_atom_recovers_displacement_and_accumulates_change_evidence() {
        let n_ckpt = 6;
        let n_grid = 17;
        let ambient = 3;
        let shift = 0.7_f64;
        let grid = drift_grid(n_ckpt, n_grid, ambient, shift);
        let latent: Array1<f64> = Array1::linspace(0.0, 1.0, n_grid);
        let ckpt_ids: Vec<String> = (0..n_ckpt).map(|c| format!("dev{c}")).collect();
        let atom_names = vec!["constant".to_string(), "drifter".to_string()];
        let input = CheckpointDynamicsInput {
            decoder_grid: grid.view(),
            checkpoint_ids: &ckpt_ids,
            atom_names: &atom_names,
            latent_grid: latent.view(),
        };
        let traj = checkpoint_atom_dynamics(&input).expect("dynamics");
        let drifter = &traj[1];

        // Each consecutive step displaces component 0 at the mode by exactly
        // `shift`; the reported displacement size is `‖Δ‖₂`. On the light
        // interpolation ridge (λ = GRID_FIT_RIDGE ≈ 1e-3) the plug-in contrast
        // `shift/(1+λ)` tracks the true displacement to sub-percent, and every
        // reported quantity is finite. (The component displacement lives in a
        // single ambient channel, so the L2 size IS that channel's contrast.)
        for report in &drifter.conditional_step_contrasts {
            assert!(
                (report.theta_plugin - shift).abs() < 1e-2 * shift,
                "drift step plug-in displacement should track {shift}, got {}",
                report.theta_plugin
            );
            assert!(
                report.theta_onestep.is_finite() && report.se.is_finite(),
                "debiased displacement and SE must be finite"
            );
            // The displacement is unambiguously positive (a real change).
            assert!(
                report.theta_plugin > 0.5 * shift,
                "drift displacement should be well above zero, got {}",
                report.theta_plugin
            );
        }

        // The drift is real → every step's no-change e-value is strictly
        // positive (studentized displacement away from zero), so the change
        // certificate carries strictly positive log-evidence on its claims,
        // unlike the constant atom whose claims carry exactly zero.
        let cert = drifter.change_evidence.certify(0.05);
        let total_log_e: f64 = cert.entries.iter().map(|e| e.log_e).sum();
        assert!(
            total_log_e > 0.0,
            "steady real drift must accumulate positive change evidence, entries: {:?}",
            cert.entries
                .iter()
                .map(|e| (e.log_e, e.confirmed))
                .collect::<Vec<_>>()
        );
    }

    /// A drifting atom must out-evidence a constant atom: the change e-process
    /// is a genuine discriminator, not a constant.
    #[test]
    fn drift_outweighs_constant_in_change_evidence() {
        let n_ckpt = 6;
        let n_grid = 17;
        let ambient = 3;
        let grid = drift_grid(n_ckpt, n_grid, ambient, 0.7);
        let latent: Array1<f64> = Array1::linspace(0.0, 1.0, n_grid);
        let ckpt_ids: Vec<String> = (0..n_ckpt).map(|c| format!("dev{c}")).collect();
        let atom_names = vec!["constant".to_string(), "drifter".to_string()];
        let input = CheckpointDynamicsInput {
            decoder_grid: grid.view(),
            checkpoint_ids: &ckpt_ids,
            atom_names: &atom_names,
            latent_grid: latent.view(),
        };
        let traj = checkpoint_atom_dynamics(&input).expect("dynamics");
        let const_log_e: f64 = traj[0]
            .change_evidence
            .certify(0.05)
            .entries
            .iter()
            .map(|e| e.log_e)
            .sum();
        let drift_log_e: f64 = traj[1]
            .change_evidence
            .certify(0.05)
            .entries
            .iter()
            .map(|e| e.log_e)
            .sum();
        assert!(
            drift_log_e > const_log_e,
            "drift change-evidence {drift_log_e} must exceed constant {const_log_e}"
        );
    }

    #[test]
    fn rejects_single_checkpoint_and_axis_mismatch() {
        let grid = Array4::<f64>::zeros((1, 2, 5, 3));
        let latent: Array1<f64> = Array1::linspace(0.0, 1.0, 5);
        let ids = vec!["only".to_string()];
        let names = vec!["a".to_string(), "b".to_string()];
        let input = CheckpointDynamicsInput {
            decoder_grid: grid.view(),
            checkpoint_ids: &ids,
            atom_names: &names,
            latent_grid: latent.view(),
        };
        assert!(checkpoint_atom_dynamics(&input).is_err());
    }
}

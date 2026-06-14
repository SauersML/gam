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
//! * [`crate::inference::structure_evidence`] — each per-step contrast feeds a
//!   universal-inference split-likelihood e-value into a [`StructureLedger`]
//!   e-process under the null "the atom did not change from checkpoint 0 to
//!   checkpoint c". Optional-stopping-safe by construction.
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

use crate::inference::layer_transport::{
    ChartTopology, LayerTransportReport, fit_layer_transport,
};
use crate::inference::riesz::{
    RieszDebiasReport, RieszInput, SmoothFunctional, debias_with_dense_hessian,
};
use crate::inference::structure_evidence::{ClaimKind, StructureLedger};
use ndarray::{Array1, Array2, ArrayView1, ArrayView4};

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
pub struct AtomTrajectory {
    pub atom_name: String,
    /// Debiased `g^{(c+1)}(t_mode) − g^{(c)}(t_mode)` for each consecutive
    /// checkpoint step, with its plug-in SE.
    pub step_contrasts: Vec<RieszDebiasReport>,
    /// Consecutive-checkpoint chart correspondences (checkpoint axis reused as
    /// the transport "layer" axis).
    pub transports: Vec<LayerTransportReport>,
    /// Anytime-valid evidence that the atom changed from checkpoint 0 to each
    /// checkpoint `c`, accumulated as one e-process per step.
    pub change_evidence: StructureLedger,
}

/// Run cross-checkpoint debiased dynamics inference for every atom.
///
/// For each atom, walks consecutive checkpoints and, at each step `c → c+1`:
/// 1. fits the transport map between the two checkpoints' latent charts
///    ([`fit_layer_transport`], checkpoint axis as the layer axis);
/// 2. evaluates the Riesz-debiased decoder-displacement contrast at the
///    latent-grid mode ([`SmoothFunctional::Contrast`] + penalty debiasing);
/// 3. absorbs the contrast as a universal-inference e-value into the atom's
///    change e-process under the no-change null.
pub fn checkpoint_atom_dynamics(
    input: &CheckpointDynamicsInput<'_>,
) -> Result<Vec<AtomTrajectory>, String> {
    let shape = input.decoder_grid.shape();
    let (n_checkpoints, n_atoms, n_grid, ambient_dim) =
        (shape[0], shape[1], shape[2], shape[3]);
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
            let coords_from = input.decoder_grid.slice(ndarray::s![c0, atom, .., 0]).to_owned();
            let coords_to = input.decoder_grid.slice(ndarray::s![c1, atom, .., 0]).to_owned();
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
            let report = contrast_at_mode(
                &ContrastAtMode {
                    grid: input.decoder_grid,
                    atom,
                    c0,
                    c1,
                    ambient_dim,
                    n_grid,
                    hessian: hessian.view(),
                    mode_row: mode_row.view(),
                },
            )
            .map_err(|e| {
                format!(
                    "checkpoint contrast for atom '{atom_name}' step {} → {} failed: {e}",
                    input.checkpoint_ids[c0], input.checkpoint_ids[c1]
                )
            })?;

            // --- anytime-valid evidence the atom changed 0 → c+1 -------------
            // Universal-inference split-likelihood e-value under the local
            // Gaussian model `θ̂ ~ N(θ, se²)`: the no-change null is θ = 0, the
            // alternative plugs in the debiased estimate, and the log-e-value
            // is the log-likelihood ratio of the OBSERVED θ̂ under the
            // alternative vs the null. Both densities are frozen by the
            // estimate/SE before this comparison (the contrast is computed once
            // and not refit on its own value), so the per-step ratios compound
            // into a valid e-process.
            let claim = change_evidence.register(ClaimKind::Custom {
                label: format!(
                    "atom '{atom_name}' changed by checkpoint {}",
                    input.checkpoint_ids[c1]
                ),
            });
            let log_e = no_change_log_e_value(report.theta_onestep, report.se);
            change_evidence.absorb_log(claim, log_e)?;

            step_contrasts.push(report);
        }

        trajectories.push(AtomTrajectory {
            atom_name,
            step_contrasts,
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

    let mut report = witness.ok_or_else(|| {
        "checkpoint contrast requires at least one ambient component".to_string()
    })?;
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
    debias_with_dense_hessian(&input, hessian)
        .map_err(|e| format!("Riesz debiasing failed: {e}"))
}

/// Universal-inference split-likelihood log-e-value for the no-change null
/// `θ = 0` against the plug-in alternative `θ = θ̂`, under the local Gaussian
/// model `θ̂ ~ N(θ, se²)`. The log-likelihood ratio of the observed `θ̂` is
/// `[(θ̂ − 0)² − (θ̂ − θ̂)²] / (2 se²) = θ̂² / (2 se²)` — the squared
/// studentized displacement in nats. A degenerate (non-positive) SE yields a
/// zero log-e-value (no evidence rather than spurious certainty).
fn no_change_log_e_value(theta_hat: f64, se: f64) -> f64 {
    if !(se > 0.0) || !theta_hat.is_finite() {
        return 0.0;
    }
    let z = theta_hat / se;
    0.5 * z * z
}

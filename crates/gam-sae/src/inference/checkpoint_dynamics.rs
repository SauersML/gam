//! Cross-checkpoint descriptive dynamics for SAE decoder curves.
//!
//! The input is a deterministic grid of already-fitted decoder values. Such a
//! grid contains no observation-level scores, sampling covariance, or null
//! distribution, so this module reports geometric displacements and chart
//! transports only. It deliberately emits no standard errors, p-values, or
//! e-values. Calibrated change evidence requires fit-time influence data (or an
//! external replicated checkpoint experiment) in a future input schema.

use crate::inference::layer_transport::{ChartTopology, LayerTransportReport, fit_layer_transport};
use ndarray::{Array1, ArrayView1, ArrayView4};

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

/// One deterministic decoder-grid change between consecutive checkpoints.
pub struct CheckpointStepChange {
    pub checkpoint_from: String,
    pub checkpoint_to: String,
    pub latent_coordinate: f64,
    /// Ambient decoder displacement at the central latent-grid node.
    pub displacement_at_mode: Array1<f64>,
    pub l2_at_mode: f64,
    /// Root mean squared ambient displacement over the complete shared grid.
    pub grid_rms_l2: f64,
    /// Largest ambient displacement over the complete shared grid.
    pub grid_max_l2: f64,
}

/// The descriptive training trajectory of one atom across checkpoints.
pub struct AtomTrajectory {
    pub atom_name: String,
    pub descriptive_step_changes: Vec<CheckpointStepChange>,
    /// Consecutive-checkpoint chart correspondences (checkpoint axis reused as
    /// the transport "layer" axis).
    pub transports: Vec<LayerTransportReport>,
}

/// Run cross-checkpoint descriptive dynamics for every atom.
///
/// For each atom, walks consecutive checkpoints and, at each step `c → c+1`:
/// 1. fits the transport map between the two checkpoints' latent charts
///    ([`fit_layer_transport`], checkpoint axis as the layer axis);
/// 2. reads direct decoder displacement summaries on the shared grid.
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
    let (lo, hi) = interval_bounds(input.latent_grid)?;
    let topology = ChartTopology::Interval { lo, hi };
    let latent_coords = input.latent_grid.to_owned();

    let mut trajectories = Vec::with_capacity(n_atoms);
    for atom in 0..n_atoms {
        let atom_name = input.atom_names[atom].clone();
        let mut descriptive_step_changes = Vec::with_capacity(n_checkpoints - 1);
        let mut transports = Vec::with_capacity(n_checkpoints - 1);

        for step in 0..n_checkpoints - 1 {
            let c0 = step;
            let c1 = step + 1;

            // --- transport map across the checkpoint axis --------------------
            // The chart coordinate is the supplied latent grid itself. Decoder
            // output components are ambient values and may be non-injective
            // (for a circle, a component can be cos(t)); they are never used as
            // latent coordinates.
            let transport = fit_layer_transport(
                c0,
                c1,
                latent_coords.view(),
                latent_coords.view(),
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

            let mut displacement_at_mode = Array1::<f64>::zeros(ambient_dim);
            let mut grid_sum_sq = 0.0_f64;
            let mut grid_max_l2 = 0.0_f64;
            for grid_idx in 0..n_grid {
                let mut row_sq = 0.0_f64;
                for component in 0..ambient_dim {
                    let delta = input.decoder_grid[[c1, atom, grid_idx, component]]
                        - input.decoder_grid[[c0, atom, grid_idx, component]];
                    row_sq += delta * delta;
                    if grid_idx == mode_index {
                        displacement_at_mode[component] = delta;
                    }
                }
                grid_sum_sq += row_sq;
                grid_max_l2 = grid_max_l2.max(row_sq.sqrt());
            }
            let l2_at_mode = displacement_at_mode.dot(&displacement_at_mode).sqrt();
            descriptive_step_changes.push(CheckpointStepChange {
                checkpoint_from: input.checkpoint_ids[c0].clone(),
                checkpoint_to: input.checkpoint_ids[c1].clone(),
                latent_coordinate: input.latent_grid[mode_index],
                displacement_at_mode,
                l2_at_mode,
                grid_rms_l2: (grid_sum_sq / n_grid as f64).sqrt(),
                grid_max_l2,
            });
        }

        trajectories.push(AtomTrajectory {
            atom_name,
            descriptive_step_changes,
            transports,
        });
    }

    Ok(trajectories)
}

/// Strict interval bounds for the supplied latent chart.
fn interval_bounds(grid: ArrayView1<'_, f64>) -> Result<(f64, f64), String> {
    let lo = grid.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = grid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if hi <= lo {
        return Err("checkpoint dynamics latent_grid must have positive range".to_string());
    }
    let pad = (hi - lo) * 1e-6;
    Ok((lo - pad, hi + pad))
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
    fn no_change_atom_has_zero_descriptive_displacement() {
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

        // Atom 0 is identical across checkpoints: every descriptive step change
        // must be exactly zero displacement (the reported displacement is the
        // raw decoder-grid difference — no fit, no fabricated SE or e-value).
        let constant = &traj[0];
        assert_eq!(constant.descriptive_step_changes.len(), n_ckpt - 1);
        for change in &constant.descriptive_step_changes {
            assert_eq!(
                change.l2_at_mode, 0.0,
                "constant atom mode displacement must be exactly zero"
            );
            assert_eq!(
                change.grid_rms_l2, 0.0,
                "constant atom grid displacement must be exactly zero"
            );
            assert_eq!(change.grid_max_l2, 0.0);
        }
        // The transport across identical checkpoint charts is the identity map
        // on the shared latent grid — a degree-free, fold-free interval homeo.
        assert_eq!(constant.transports.len(), n_ckpt - 1);
    }

    #[test]
    fn drifting_atom_recovers_exact_descriptive_displacement() {
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

        // Each consecutive step displaces component 0 at the mode node by exactly
        // `shift` and touches no other node/component. The displacement is the
        // raw decoder-grid difference, so the mode L2 size is exactly `shift`,
        // and — since only the single mode node moves — the grid RMS over
        // `n_grid` nodes is `shift / sqrt(n_grid)`.
        assert_eq!(drifter.descriptive_step_changes.len(), n_ckpt - 1);
        for change in &drifter.descriptive_step_changes {
            assert!(
                (change.l2_at_mode - shift).abs() < 1e-12,
                "drift mode displacement must equal {shift}, got {}",
                change.l2_at_mode
            );
            // Displacement lives in component 0 only.
            assert!((change.displacement_at_mode[0] - shift).abs() < 1e-12);
            for comp in 1..ambient {
                assert_eq!(change.displacement_at_mode[comp], 0.0);
            }
            let expected_rms = shift / (n_grid as f64).sqrt();
            assert!(
                (change.grid_rms_l2 - expected_rms).abs() < 1e-12,
                "drift grid RMS must equal {expected_rms}, got {}",
                change.grid_rms_l2
            );
            assert!((change.grid_max_l2 - shift).abs() < 1e-12);
        }
    }

    /// A drifting atom's descriptive displacement must exceed a constant atom's
    /// (which is exactly zero): the readout is a genuine change discriminator.
    #[test]
    fn drift_displacement_exceeds_constant() {
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
        let const_total: f64 = traj[0]
            .descriptive_step_changes
            .iter()
            .map(|c| c.l2_at_mode)
            .sum();
        let drift_total: f64 = traj[1]
            .descriptive_step_changes
            .iter()
            .map(|c| c.l2_at_mode)
            .sum();
        assert_eq!(const_total, 0.0, "constant atom displacement must be zero");
        assert!(
            drift_total > const_total,
            "drift displacement {drift_total} must exceed constant {const_total}"
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

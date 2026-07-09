"""Cross-checkpoint descriptive SAE atom-trajectory dynamics (issue #1102).

Thin wrapper over the Rust core ``gam::inference::checkpoint_dynamics``: all math
(per-step descriptive decoder change, inter-checkpoint transport maps) lives in
Rust; this module only marshals numpy arrays across the FFI boundary.

The setting: an SAE is fitted at each of several training checkpoints on the same
prompts at a layer; every atom ``k`` has a decoder curve ``g_k^(c)(t)`` sampled
on a shared latent grid. ``sae_checkpoint_dynamics`` answers, per atom, *how much
did the atom's decoder move across training*, as plain measured displacement. The
anytime-valid change e-process and the penalty-debiased Riesz contrast estimator
were removed: a bare decoder grid does not supply the inputs a coverage-valid
change certificate requires, so the readout is descriptive only — no e-values, no
debiased contrasts, no confidence claim.

Functions
---------
sae_checkpoint_dynamics
    Run the per-atom descriptive cross-checkpoint dynamics and return one
    trajectory per atom (descriptive step changes and transport maps).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._binding import rust_module


def sae_checkpoint_dynamics(
    decoder_grid: Any,
    checkpoint_ids: Sequence[str],
    atom_names: Sequence[str],
    latent_grid: Any,
) -> dict[str, Any]:
    """Descriptive atom trajectories across training checkpoints.

    Parameters
    ----------
    decoder_grid
        Array of shape ``[n_checkpoints, n_atoms, n_grid, ambient_dim]``: every
        atom's decoder curve sampled on the shared ``latent_grid`` at every
        checkpoint.
    checkpoint_ids
        Labels for the checkpoint axis (length ``n_checkpoints``).
    atom_names
        Labels for the atom axis (length ``n_atoms``).
    latent_grid
        The shared 1-D latent grid the curves are sampled on (length
        ``n_grid``).

    Returns
    -------
    dict
        ``{"trajectories": [ {"atom_name", "descriptive_step_changes": [...],
        "transports": [...]} ]}``. Each entry in ``descriptive_step_changes`` is
        one consecutive-checkpoint step with ``checkpoint_from``,
        ``checkpoint_to``, ``latent_coordinate`` (the central grid node),
        ``displacement_at_mode`` (the ambient displacement vector there),
        ``l2_at_mode``, ``grid_rms_l2``, and ``grid_max_l2``. These are plain
        measured decoder changes: NO e-values, NO debiased contrasts, and NO
        coverage claim.
    """
    grid = np.ascontiguousarray(decoder_grid, dtype=np.float64)
    if grid.ndim != 4:
        raise ValueError(
            f"decoder_grid must be 4-D [n_ckpt, n_atoms, n_grid, ambient], got {grid.shape}"
        )
    latent = np.ascontiguousarray(latent_grid, dtype=np.float64)
    if latent.ndim != 1:
        raise ValueError(f"latent_grid must be one-dimensional, got shape {latent.shape}")
    return rust_module().sae_checkpoint_dynamics(
        grid,
        [str(c) for c in checkpoint_ids],
        [str(a) for a in atom_names],
        latent,
    )

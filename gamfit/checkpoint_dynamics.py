"""Cross-checkpoint Riesz-debiased SAE atom-trajectory dynamics (issue #1102).

Thin wrapper over the Rust core ``gam::inference::checkpoint_dynamics``: all math
(per-step Riesz-debiased decoder-displacement contrast, inter-checkpoint
transport maps, anytime-valid change e-process) lives in Rust; this module only
marshals numpy arrays across the FFI boundary.

The setting: an SAE is fitted at each of several training checkpoints on the same
prompts at a layer; every atom ``k`` has a decoder curve ``g_k^(c)(t)`` sampled
on a shared latent grid. ``sae_checkpoint_dynamics`` answers, per atom, *did the
atom change across training and by how much*, with debiased point estimates,
standard errors, and anytime-valid evidence.

Functions
---------
sae_checkpoint_dynamics
    Run the per-atom debiased cross-checkpoint dynamics and return one trajectory
    per atom (step contrasts, transport maps, change-evidence certificate).
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
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Debiased atom trajectories across training checkpoints.

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
    alpha
        Level for the per-atom anytime-valid change-evidence e-BH certificate.

    Returns
    -------
    dict
        ``{"trajectories": [ {"atom_name", "step_contrasts": [...],
        "transports": [...], "change_evidence": {...}} ]}``. Each entry in
        ``step_contrasts`` is a Riesz report with ``theta_plugin``,
        ``theta_onestep``, ``se``, ``penalty_bias`` and a 95% ``ci_lo`` /
        ``ci_hi``; ``change_evidence`` is the e-BH certificate of the atom's
        no-change e-process at ``alpha``.
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
        float(alpha),
    )

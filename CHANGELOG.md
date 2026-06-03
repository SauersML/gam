# Changelog

## gamfit 0.1.148 (upcoming)

Draft release notes for the next gamfit version after `0.1.147`. The version has
not been bumped in this changelog entry.

### New

- Added cross-atom decoder incoherence for manifold SAE fits via
  `decoder_incoherence_weight` (#671). This is a separability lever for
  multi-atom fits: when `K >= 2`, it is on by default and penalizes overlap
  between co-activating atoms' decoder column spaces, weighted by empirical
  gate co-activation.
- Added decoder embedding-rank selection for manifold SAE fits via
  `nuclear_norm_weight` and `nuclear_norm_max_rank` (#672). Positive
  `nuclear_norm_weight` applies a nuclear-norm penalty to each atom's decoder
  matrix, shrinking its singular spectrum; `nuclear_norm_max_rank` can cap the
  number of leading singular values included in the penalty.
- Added non-convex SCAD/MCP gate sparsity for manifold SAE fits. Set
  `gate_sparsity="scad"` or `gate_sparsity="mcp"` to use the SAE
  `ScadMcpPenalty` path, with `scad_mcp_gamma` defaulting to `3.7` for SCAD and
  `2.5` for MCP. The existing `gate_sparsity="l1"` path remains the default.
- Exposed per-atom posterior shape uncertainty on `ManifoldSAE` results.
  Atoms can now carry `decoder_covariance`, `shape_band_coords`,
  `shape_band_mean`, and `shape_band_sd`; result helpers include
  `shape_uncertainty(...)` and the `shape_band(...)` alias.
- Exposed typical coordinate-range summaries for manifold SAE atoms. Use
  `coordinate_range(...)` for per-axis min, max, median, and 5th/95th
  percentile summaries, or `typical_shape(...)` to restrict the posterior shape
  band to the atom's typical recovered-coordinate range.
- Added an isometry/topology-evidence gauge guard for manifold SAE fits (#673).
  Because decoder smoothness is still computed in raw latent coordinates rather
  than arc length, `sae_manifold_fit` now warns when `isometry_weight <= 0`:
  topology comparisons by `reml_score` are gauge-dependent unless the isometry
  penalty is active. `isometry_weight` remains on by default.

### Fixed

- Fixed SAE inner-solver convergence regressions that could make affected SAE
  fits fail with `RemlConvergenceError`. The arrow-Schur PCG `schur_matvec`
  callback now clears its reused output buffer before accumulating `S*x`,
  preventing stale matvec contributions from corrupting the reduced system.
- Fixed the SAE line-search baseline used by the joint arrow-Schur inner solve.
  The solver now snapshots the exact state used to assemble the gradient and
  Hessian, then computes `pre_step_total` from that same state before Armijo
  backtracking. This avoids comparing trial steps against a stale objective
  baseline after layout/state mutations.
- Fixed non-Linux builds by providing a real non-Linux `scatter_batched`
  implementation. Mac and other non-Linux targets can now compile paths that
  call `scatter_batched` unconditionally; device-free runs execute the batch as
  a single CPU tile.

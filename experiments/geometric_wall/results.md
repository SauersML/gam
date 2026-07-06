# Geometric-Wall Closure Test on Gemma Scope

Blocked before experiment numbers could be produced.

## Blockers

- MSI can list and download `google/gemma-scope-2b-pt-res` SAE `params.npz` files, but `google/gemma-2-2b` model weights are gated. The GPU smoke run failed with `401 Unauthorized` while resolving `config.json`, so residual activations could not be harvested.
- The required remote checkout reset for `gam_cx_wallexp` points at a workspace that no longer contains `gam-sae`; `cargo check -p gam-sae` exits with `package ID specification 'gam-sae' did not match any packages` and suggests `gam-core`.

## Work Completed

- Added `gemma_wall_closure.py`, a MSI-only CLI experiment script that:
  - selects Gemma Scope 2B residual width-16k SAE files for requested layers;
  - harvests Gemma 2 residual activations when model weights are available;
  - peels the top-PCA sink direction and reports the sink fraction;
  - fits matched-design-column flat vs quadratic curved closures on post-sink PCA scores;
  - reports retained-space floor, full-energy floor, curvature proxy, and curvature/drop correlation.

No flat-vs-curved floor numbers, curvature correlation, or Gemma sink fractions are reported because the model gate prevents activation harvest.

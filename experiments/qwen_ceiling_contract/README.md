# Qwen L18 Ceiling Contract

The real Qwen ceiling run is emitted by the Rust example:

```text
cargo run -p gam-sae --example qwen_l18_ceiling -- <resid_L18.npy> [max_rows] [harmonics] [outer_iters] [inner_iters]
```

Its output includes one parseable `ceiling_contract_json` line with:

- `peel_status`: `raw` or `sink-peeled`.
- `EV_curved`: realized K=1 curved chart explained variance.
- `EV_lin_top_m_envelope`: top-`M` linear envelope for `M = 2H + 1`.
- `chart_efficiency_eta`: `EV_curved / EV_lin_top_m_envelope`.
- `gradient_certificate`: same-cache outer-gradient consistency certificate.
- `verdict`: one of `INFORMATION_CEILING`, `LANDSCAPE_PATHOLOGY`, or `RESIDUAL_ADJOINT_BUG`.

The verdict tree is:

- `eta near 1` -> `INFORMATION_CEILING`.
- `eta below near-one band` and clean gradient certificate -> `LANDSCAPE_PATHOLOGY`.
- `eta below near-one band` and failing gradient certificate -> `RESIDUAL_ADJOINT_BUG`.

This makes the K=1 ceiling experiment interpretable after sink peeling: a raw run can be dominated by the position-0 sink, while a peeled run tests whether the single-chart image is already at its top-`M` linear envelope.

The scale/ceiling harness refuses raw activation matrices by default. Run the
scale experiment on a post-peel, PCA-reduced `.npy` and declare that contract:

```text
cargo run -p gam-sae --example scale_k -- <post_peel_pca.npy> <out-dir> \
  --post-peel --n-peeled 1 --pca-dim <D> [--rows N] [--atoms K] [--epochs N]
```

An intentional raw/full-width run must pass `--raw-ok`; otherwise the harness
errors before allocating the `K x p` decoder. `numbers.json` records the run
contract as `N`, `p`, `K`, `post_peel`, `n_peeled`, `pca_dim`, and `peak_rss`,
with the peak-RSS no-`N x K` allocation invariant surfaced beside it.

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

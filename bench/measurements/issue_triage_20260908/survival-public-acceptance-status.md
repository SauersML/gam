# Public survival acceptance status

The surviving recovery and mode-response gates now live in the owning
`gam-models` integration target, `survival_acceptance`; their former root test
modules were removed. The mode-response fixture uses 400 observations and
requires both Weibull-coordinate responses to agree with finite differences
to relative error below `1e-5`. The recovery fixture retains 2,400 observations.
The focused capture helper omits unused smoothing-parameter finite-difference
ladders without changing the fit or stopping it after capture.

On MSI acn112, the canonical integration targets compiled against the current
optimized native dependency graph on 2026-09-08:

- `survival_acceptance`: success, 9.40 seconds.
- `large_scale_reml_stress`: success, 5.12 seconds. This relocated gate uses
  `StandardPredictor` for means and smoothing-corrected prediction intervals.

The graph uses `gam-models-7529e38f1ae745f1` (optimization level 1),
`gam-solve-7f5b8ec14241822d` and `gam-terms-ca01838dc35d9557` (level 2),
`gam-inference-bc64714db90024fb`, and `gam-predict-5354e2c40a7ed4bf`.
Final integration crates used optimization level 2 and 256 codegen units.
The exact commands and source/dependency hashes are recorded remotely in
`.buildd/survival_acceptance-optimized.provenance.json` and
`.buildd/large_scale_reml_stress-optimized.provenance.json` under the shared
validation checkout. All 329 dependency artifacts were protected with
hardlinks before compilation.

This is compile evidence, not a full-fit acceptance result. Earlier recovery,
latent-frailty, and location-scale runs hit their bounded timeouts. The latest
two-coordinate mode audit also timed out while the real fit continued after
capture, without reaching its assertions. The earlier optimized mode audit
reported a roughly 9.6% response discrepancy before the fifth-derivative fix.
No post-fix passing public mode-response or recovery result is claimed.
Issues #2767, #2695, and #2714 remain unresolved by these receipts; #2765 was
closed externally. The 34 passing derivative tests and scalar-only timing are
recorded separately in `survival-fifth-acceptance.md`.

The MSI connection closed during the subsequent canonical models unit build;
both default compute nodes then stopped answering. Its completion status and
the final spatial gradient test result were not established by this run.

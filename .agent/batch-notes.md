# Triage batch 958–1137 — fleet-agent gam-closed-958-1137

## Method
Read every closed issue + comments; verified claimed fix code + regression-test
wiring against current HEAD; checked SPEC (no production finite-differences).
Dispatched 5 parallel audit agents over clusters.

## Finding: #965 IMPROPERLY CLOSED (taking this one)
`survival_coerce_times` (gam-pyffi) is a PREDICT-QUERY coercion (only callers:
survival_at / hazard_at / cumulative_hazard_at / competing_risks_cif). The #965
"fix" made it REJECT negative/infinite query times. But S(t)=P(T>t) is defined
for all real t: S(t<=0)=1, S(+inf)=0. The reject-fix:
  - breaks an existing UNSKIPPED test (tests/test_penalty_sampling_survival_diagnostics_regressions.py)
    asserting survival_at([-2,0,inf]) = [1,1,0];
  - ignores the issue's own recommended option 2 (unified t<=0 => S=1, H=0).
Plus a latent incomplete-#1595 bug: survival_chunk_iter_collect hardcodes
right_value=Some(0.0) for survival, contradicting the non-chunk flat-clamp.

## Plan
1. survival_coerce_times: reject only NaN/empty; accept negative + +inf.
2. interpolation: add explicit +inf asymptote (S=0, H=+inf) distinct from the
   finite past-grid flat-clamp (#1595). Thread inf_value through interpolate_rows,
   survival_csv_interpolate, survival_chunk_iter_collect + Python table.
3. Harden exponential_survival_at: t<=0 => 1, hazard<=0 => 1 (kills exp(1)=2.7).
4. Regression tests (Rust unit + the existing Python test now passes).

## Other clusters audited: all PROPERLY_FIXED except notes on #1108 (leftover
debug diagnostics), #958/#959 (closed by feature deletion), #960 (no test).

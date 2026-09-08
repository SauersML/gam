The binomial response-family producer fix is committed as `c625efc51`.
`fit_binomial_mean_wiggle` now records the likelihood associated with its
resolved inverse link on the returned fit. The payload consumer already rejects
a missing likelihood family on main.

Fresh MSI verification exercised the actual public Rust prediction consumer:
**2 passed, 0 failed, 0 ignored, 11.52 seconds**, with two CPU threads.
Both logit and probit cases:

- Fit the identifiable six-mean-column plus eleven-warp-column fixture.
- Assemble the canonical saved payload and rebuild held-out midpoint designs
  from that payload.
- Evaluate both plugin and posterior response means; both remain finite and
  within [0, 1], and differ from the linear predictor.
- Serialize and reload JSON; linear predictors, plugin means, and posterior
  means remain bit-identical.

The regression is
`crates/gam-predict/tests/flexible_binomial_response_scale_2748.rs`, with source
SHA256 `81c33b629202d07c92319f2f74c8485e1b0cca6a058bf4b7bb5073ebf401af89`.
The result log is
`bench/measurements/issue_triage_20260908/issue2748-public-predict-tests.log`.
The corresponding Python save/load regression is committed, but has not run
against a fresh extension: existing repository build-scanner violations,
including excluded SAE files, block that build. No local code or tests ran.

This proves the response-family repair on the stated fixtures. The original
20-scenario benchmark population has not been remeasured on this source, so the
issue remains open.

# firth-rescue-transparency

TITLE: Automatic Firth/Jeffreys rescue: never silent, only when separation is proven, and exact fast posterior sampling under it
WORK ITEM: Read audit/inference.md B5 and the repros bench/pygam_audit/inference/repro_nuts_firth.py, repro_nuts_firth2.py and repro_firth_summary.py on this branch.
Evidence: on binomial data with no separation (inference MC cell binom, n=400), 3/19 `sample()` calls took 373-424 s through NUTS instead of 2-4 s through Polya-Gamma. The saved model JSON of rep 21 has `"firth_bias_reduction": true`, yet summary(), summary().convergence, predict() and the model repr never mention it. crates/gam-models/src/fit_orchestration/fit.rs ~562-587 adopts the Firth retry after a failed ordinary fit and logs only at log::info!. crates/gam-inference/src/hmc_io.rs ~5258 routes every Firth fit to NUTS, and ~4562 is the Polya-Gamma path.
Fix:
1. Trigger. The estimator may change only when separation is established (the sibling claude/pygam-separation owns detection; read its branch and PR first). A retry after an unrelated numerical failure (cubature, B1; railed rho, B2) must not switch estimators; that failure must be fixed at its root or reported. Remove the "any failure, so try Firth" retry.
2. Transparency. Record the estimator ("penalized likelihood" vs "penalized likelihood with Jeffreys prior") and the typed reason in the model payload, and show it in summary().convergence, the model repr and `gam summary` (CLI/Python parity; logic in Rust).
3. Sampling. Under the Jeffreys prior, draw with the Polya-Gamma conditional as an independence proposal and correct exactly with a Metropolis-Hastings ratio for the Jeffreys factor |I(beta')|^(1/2) / |I(beta)|^(1/2) (the log-determinant is already available from the Firth fit). Report the acceptance rate. No tuning knob and no step size. If the acceptance rate is provably too low for the proposal to mix, say so with a typed diagnostic instead of silently switching method. Delete the Firth-to-NUTS route if the MH sampler replaces it on this path.
Coordinate: separation (detection and the Firth fit itself), smoothing-correction-provenance (B1 fallback reason), null-rail-certify (B2), summary (field placement), sample-rho-marginal (same sampler files; merge main often).
Acceptance:
- A test on the rep-21 fixture (non-separated binomial) asserts that the fit does not use Firth, or if it truly must, that summary().convergence names the estimator and reason.
- A separated fixture asserts the estimator is reported in Python summary, the repr and CLI summary, with identical text.
- A Firth sample() test asserts that posterior means and quantiles match a long reference chain within Monte Carlo SE, that the acceptance rate is reported, and that it runs in the same order of time as the non-Firth Polya-Gamma path (report timings in the PR; no wall-clock asserts in library code).
- All fail at HEAD.

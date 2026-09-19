# conformal-families

TITLE: model.predict(interval="conformal") exact full conformal for every standard family, plus a coverage benchmark proving we beat pyGAM's intervals
WORK ITEM: the full-conformal route in gam-predict (crates/gam-predict/src/conformal_routes.rs, full_conformal_prediction_columns) serves only Gaussian-identity models fitted without prior weights, offsets or a link wiggle, and only when the substrate was precomputed at fit time (gamfit/_model.py predict docstring ~173-200). The certified GLM homotopy (GlmHomotopyFullConformal in crates/gam-models/src/inference/full_conformal.rs) is reachable only through the low-level research wrapper gamfit.full_conformal.glm_full_conformal(design, s_lambda, ...). Users of model.predict never get it. Discrete families have no enumeration arm, although the module doc notes that full conformal is exact there by enumerating the support.
pyGAM has no conformal at all, and its prediction_intervals under-cover under misspecification and at small n. This lane turns our exact finite-sample coverage into a user-visible, benchmarked win.
Fix (Rust owns the logic; Python and CLI only marshal; parity across all three):
1. Route full conformal through model.predict(interval="conformal") / `gam predict --conformal` for:
   - Binomial/Bernoulli: exact enumeration of the support {0..m}, a conformal p-value per level, and the set of levels with p > alpha. The output is a set or a probability-level set, not a +/- band.
   - Poisson and negative binomial: enumeration up to a data-derived tail point where the conformal p-value is provably <= alpha (derive it from the monotone score tail; no hand cap).
   - Continuous non-Gaussian (Gamma, inverse Gaussian, and so on): the certified homotopy, with a score chosen appropriately (e.g. |y - mu| / sqrt(V(mu)) Pearson). Document the choice and justify its symmetry.
   - Offsets: allowed. The test row carries its own offset, so exchangeability holds; add a test.
   - Prior weights: keep the refusal, but make it a typed error that points to the split-conformal band with calibration=.
2. Remove the fit-time precompute_conformal requirement if the model can recompute what it needs. Coordinate with model-payload, which owns n-sized persistence: do not persist n-sized data by default. If full conformal truly needs the training rows, predict takes them explicitly or the model carries a documented opt-in.
3. Smoothing parameters: consume whatever certificate conformal-honest-rho produces. Do not add a second certificate.
4. Benchmark: add a conformal/interval-coverage scenario set to bench/pygam_compare (on main; see plans.py). The DGPs are a correct model, a misspecified mean, heteroscedastic noise, heavy tails, a binomial and a Poisson, each at n in {30, 100, 1000}, seeded, >= 1000 reps. For each, report the empirical coverage and median width of:
   - pyGAM prediction_intervals;
   - gamfit posterior intervals;
   - gamfit full conformal;
   - gamfit split conformal.
   Commit the result table under bench/pygam_audit/conformal_coverage.md.
Coordinate: conformal-honest-rho (core certificate, full_conformal.rs internals), predict-interval-hygiene (Student-t posterior intervals), predict-transport (the output format of predict; agree on columns for set-valued output), model-payload, pv-instruments.
Acceptance:
- Python and Rust tests show interval="conformal" works for binomial, Poisson, NB, Gamma, and Gaussian with an offset, and gives coverage >= 1 - alpha - 2*MCSE in a seeded Monte Carlo.
- A CLI parity test shows the same sets from `gam predict --conformal`.
- The benchmark table exists, and gamfit full conformal hits nominal coverage in every cell where pyGAM misses.
- All route tests fail at HEAD, where only Gaussian identity is supported.

# error-routing-remnants

TITLE: Route fit-time weight errors and predict-time NaN/unseen levels to the right error class; hint factor() for C()
WORK ITEM: Read audit/robustness.md F7 (remnants) and audit/api.md F15. errors/input-validation cover the main taxonomy, but not these three paths.
Evidence: negative weights raise SchemaMismatchError (crates/gam-models/src/fit_orchestration/materialize/columns.rs:219-221), which renders the predict-time help "Verify the new data has the same columns..." (fit_orchestration/error.rs:255) at fit time. NaN and unseen factor levels at predict time raise base GamError, not PredictInputError. The patsy formula `C(x2)` is rejected with a generic parse error and no hint.
Fix: add an InvalidWeights input error with its own message. Map predict-time NaN and unseen levels to PredictInputError naming the column and level. On a parse error for `C(...)`, suggest `factor(...)`; do not add an alias (SPEC: delete unnecessary options). Keep messages in Rust and mirror the CLI.
Coordinate: errors, input-validation (columns.rs), formula-dsl (parser hint), factor-contract (unseen levels).
Acceptance: pytest asserts:
- negative weights raise the weights error class and the message has no "new data";
- predict with NaN or an unseen level raises PredictInputError;
- "C(x2)" raises with "factor(" in the message.
Rust tests check the same. All fail at HEAD.

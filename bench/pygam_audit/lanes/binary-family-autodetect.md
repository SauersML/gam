# binary-family-autodetect

TITLE: Auto-detect binomial for two-level label columns and route n=1 to a too-few-rows error
WORK ITEM: Read audit/docs.md DOC-18 (behaviour part) and audit/robustness.md F7 (n=1 case). The docs item fixes only the text, and link-parity/auto-formula do not change family detection.
Evidence: a y column with two string/categorical levels is not auto-detected as binomial, although the resulting error message already identifies the column as binary. With n=1 and y=[1.0], auto-family picks binomial and reports "degenerate binomial" instead of "too few rows for s(x)".
Fix (SPEC: defaults recover the null; no magic constants): in crates/gam-models/src/fit_orchestration/materialize/family.rs, treat exactly two distinct non-null levels (numeric 0/1, bool or two-level categorical/string) as binomial, with a documented level-to-0/1 mapping (sorted order, reported in summary). Check row count against the model's minimum before family detection.
Coordinate: link-parity (same family.rs, F11 at ~617-629; keep hunks disjoint), auto-formula, input-validation, docs (DOC-18 text).
Acceptance: a test fits y in {"yes","no"} and asserts family == binomial and correct fitted probabilities. A test fits with n=1 and asserts the error mentions too few rows. Both fail at HEAD. Both run in Rust (CLI parity) and Python.

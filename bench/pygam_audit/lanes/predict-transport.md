# predict-transport

TITLE: Return predictions as numpy buffers instead of JSON strings; labeled interval output
WORK ITEM: Read audit/speed.md F7 (point 4) and audit/api.md F14. binomial-predict owns the binomial te kernel cost, but no item owns the JSON transport.
Evidence: predict_table returns PyResult<String> (crates/gam-pyffi/src/model/model_ffi.rs:1692), built by serde_json::to_string(&PredictionPayload{...}) (crates/gam-pyffi/src/manifold/geometry_ffi.rs:7189-7206) and parsed in Python. Gaussian/Poisson predict is 2.4-3x slower than pyGAM (4.3 ms vs 1.5 ms at 1e3 rows p=1; 0.28 s vs 0.11 s at 1e5).
Fix (SPEC: Python thin, parity): return contiguous f64 numpy arrays via PyO3 buffers (mean, se, lower, upper) with no JSON, and keep the Python side a wrapper. API F14 partial: numpy interval predict returns 6 unlabeled columns; return named columns or a structured result that matches the DataFrame path. predict with numpy after a DataFrame fit raises SchemaMismatchError; accept positional arrays with matching width.
Note for binomial-predict: binomial te predict costs 1.36 s per 1e3 rows (544x). Its trace is outside this item.
Coordinate: binomial-predict, posterior-sd, model-payload, sklearn-contract, pyclass-module-path (touches only #[pyclass] attributes in model_ffi.rs).
Acceptance: a benchmark test asserts Gaussian predict at 1e5 rows p=1 runs <= 1.2x pyGAM's recorded 0.11 s (or <= 0.13 s absolute on CI hardware marker). A test asserts predict_table returns ndarray, not str. A test asserts numpy interval output has named fields, and one asserts numpy predict after a DataFrame fit succeeds. All fail at HEAD.

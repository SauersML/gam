# pyclass-module-path

TITLE: Set module="gamfit._rust" on every pyclass and make MultinomialModel picklable
WORK ITEM: Read audit/packaging.md PKG-02 and audit/api.md F1. namespace/pickle cover the main Model classes but not the research pyclasses' module path or MultinomialModel.
Evidence: research pyclasses report __module__ "gam_pyffi._rust", which is unimportable. Pickle and copy then fail, and repr shows the wrong path. MultinomialModel (gamfit/_model.py:1405) has no __reduce__/__getstate__ and pickling raises.
Fix: add module = "gamfit._rust" to every #[pyclass] under crates/gam-pyffi/src/**. Add MultinomialModel pickling through the same payload path as Model; logic stays in Rust.
Coordinate: pickle (same _model.py pickling helpers), namespace, typing, predict-transport (edits model_ffi.rs bodies; this item touches only #[pyclass(...)] attributes).
Acceptance: a test iterates every class exported from gamfit._rust and asserts __module__ == "gamfit._rust" and that importlib resolves it. A test asserts that pickle.loads(pickle.dumps(fitted MultinomialModel)) predicts identically. Both fail at HEAD.

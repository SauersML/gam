// The GPU-skip gate is declared once at the crate root (tests/arrow_gpu.rs);
// per-binary mod inclusions would trip clippy::duplicate_mod, and the
// `common::fixtures` helpers are intentionally not pulled in here so this
// binary stays clear of `dead_code` warnings under `warnings = "deny"`.


//! Dedicated integration-test target for the three issue #979 binary widths.
//!
//! This keeps centers 4, 12, and 20 independently selectable without compiling
//! the full regression suite, which is essential for bounded remote iteration.

#[path = "regressions/smooths/margslope_matern_logslope_slowdown.rs"]
mod margslope_matern_logslope_slowdown;

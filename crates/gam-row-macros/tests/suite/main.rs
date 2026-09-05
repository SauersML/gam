//! Integration-test harness for gam-row-macros: every module here was a
//! standalone tests/*.rs crate and therefore its own link of gam-row-macros and
//! its dependency tree. One binary, same tests, same names.
//!
//! Time both generated and analytic kernels inlined into the same consuming
//! row loop. Generated production lowerings use inline(always); imposing an
//! outlined wrapper measures an extra aggregate-return ABI and prevents both
//! sides from optimizing with their consumer. The nudge and output fold keep
//! the complete requested computation live in the paired timing harness.

mod cause_specific_codegen_perf;
mod gaussian_codegen_perf;
mod rigid_bms_codegen_perf;
mod sls_codegen_perf;

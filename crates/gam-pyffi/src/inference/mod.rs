//! Inference instruments and benchmark scoring kernels: the anytime-valid
//! structure-discovery e-process / e-BH certificate and Lawley/Skovgaard
//! likelihood instruments (#984/#939), plus the pure scalar metric kernels used
//! to score predictions against a mature reference.
//!
//! Re-exported at the crate root under their flat names so the `#[pymodule]`
//! registration (`inference_instruments::register`) and the benchmark
//! aggregators in the entrypoint fragments resolve unchanged.

pub(crate) mod inference_instruments;

pub(crate) mod benchmark_scores;

//! Inference instruments: the anytime-valid structure-discovery e-process /
//! e-BH certificate and Lawley/Skovgaard likelihood instruments (#984/#939).
//!
//! Re-exported at the crate root so the `#[pymodule]` registration
//! (`inference_instruments::register`) resolves from the entrypoint fragments.

pub(crate) mod inference_instruments;

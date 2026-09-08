//! Issue #988 — Stage-3 encode throughput benchmark.
//!
//! DECISION GATE. The Stage-3 SAE pipeline encodes a row `z ∈ ℝ^p` into the
//! per-atom (gate, latent-coordinate) code by running the EXACT production
//! per-row Newton solve against a *frozen* decoder dictionary. That inner solve
//! is [`SaeManifoldTerm::run_fixed_decoder_arrow_schur`] — the same arrow-Schur
//! Gauss/Newton path the joint fit uses, but with the decoder held fixed so the
//! only unknowns are the per-row active set + latent coordinates. Batched over a
//! large block of rows, this is precisely the "encode throughput" we must size.
//!
//! The open design question #988 asks: is a *certified amortized surrogate*
//! (an amortized encoder that approximates the exact per-row solve and must then
//! be certified against it) EVER worth building? The answer is purely empirical:
//!   * The deployment gate is **10^5 rows/sec/GPU** for the exact batched encode.
//!   * If the exact solve already clears that gate, the amortized surrogate is
//!     **NEVER built** — there is no throughput headroom to win, and a surrogate
//!     would only add a certification liability.
//!   * If it does NOT clear the gate, the surrogate becomes justified.
//!
//! This test runs on CPU with NO GPU, so it CANNOT measure the 10^5 rows/sec/GPU
//! figure and CANNOT make the surrogate-vs-no-surrogate deployment decision. It
//! measures the exact CPU batched-encode throughput (as a correctness + perf
//! regression sentinel only — see `CPU_ENCODE_REGRESSION_FLOOR_ROWS_PER_SEC`) and
//! records the deployment decision as an honest tri-state
//! [`gam::gpu::policy::EncodeDeploymentDecision`]: on a CPU-only host it is
//! `Undetermined` (BLOCKED on hardware), NEVER "surrogate unneeded". Only a real
//! device measurement can move it to `Met`/`Unmet`.
//!
//! HISTORY (#1412, reopened twice): earlier versions projected the CPU rate
//! through an assumed `CPU_TO_GPU_SCALING = 100.0` and asserted the *projection*
//! cleared the gate — a CPU number dressed up as a GPU deployment certification.
//! That fudge is gone: there is no CPU→GPU factor and no CPU-derived surrogate
//! decision anywhere below. A CPU rate, however fast, cannot make this gate claim
//! the GPU target is met.
//!
//! The benchmark PRINTS the measured rows/sec per K verbatim; the assertion is
//! the gate, the print is the datum. When CUDA is available the gate also runs
//! the production device-resident certified encode and compares THAT measured
//! device rows/sec against the 100k rows/sec/GPU target; a CPU rate can only be
//! a regression sentinel and can never certify the deployment target.


// ---- production inner-solve defaults (mirror `sae_manifold_fit` encode) ------

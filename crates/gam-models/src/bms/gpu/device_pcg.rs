// ────────────────────────────────────────────────────────────────────────
// Block 9 Phase 5 — device-resident PCG against the BMS-FLEX row-Hessian
// operator.
//
// The inner Newton solve in `BernoulliMarginalSlope` (matrix-free path,
// large-scale shape n=195k, p=44, r=20) currently reaches the GPU as a
// per-CG-iteration call to `launch_bms_flex_row_hvp` returning a host
// `Vec<f64>`. With ~6400 inner CG iterations per outer iteration that round-
// trip cost dominates: each iter pays one `stream.synchronize()` plus one
// DtoH download. At p=44 the download itself is 352 bytes — trivial in
// bandwidth, painful in latency.
//
// Phase 5 keeps every PCG vector on the device and runs the outer loop with
// only a single small scalar download per iteration (the squared residual
// norm for the convergence check). The Hv kernel becomes `into_device`
// (Block 9 addition to `bms_flex_row.rs`), and the axpy / dot / diagonal-
// preconditioner / scale-and-add steps run as tiny NVRTC kernels on the
// same default stream so the sequence is implicitly ordered without sync.
// ────────────────────────────────────────────────────────────────────────

/// Inputs to [`run_pcg_against_row_hessian_device`]. The right-hand-side
/// `b` is supplied as a host slice (it is the only host-resident vector
/// that needs to enter the loop — the iterate, residual, search direction,
/// and Hv output all live on the device).
#[cfg(target_os = "linux")]
pub struct DeviceResidentPcgInput<'a> {
    /// Per-fit row-Hessian + design storage. The PCG operator is
    /// `v ↦ launch_bms_flex_row_hvp_into_device(storage, ...)`.
    pub storage: &'a crate::bms::gpu::row::DeviceResidentRowHess,
    /// Right-hand-side `b`, length `storage.block.p_total`. Uploaded once.
    pub b: &'a [f64],
    /// Convergence tolerance on relative residual `‖r‖₂ / ‖b‖₂`.
    pub rel_tol: f64,
    /// Hard cap on iterations (the inner loop also bails on stagnation).
    pub max_iters: usize,
    /// Floor on `|diag(H)[i]|` used by the Jacobi preconditioner. Set to
    /// `1e-12` for the matrix-free row-Hessian path; the row-primary
    /// Hessian's diagonal is positive-definite by construction.
    pub precond_diag_floor: f64,
}

/// Output of [`run_pcg_against_row_hessian_device`].
#[cfg(target_os = "linux")]
pub struct DeviceResidentPcgOutput {
    /// Solution `x` such that `H · x ≈ b`, length `storage.block.p_total`.
    pub x: Vec<f64>,
    /// Number of PCG iterations consumed (final iter does not count if it
    /// converged immediately after the dot reduction).
    pub iterations: usize,
    /// Final achieved relative residual `‖r‖₂ / ‖b‖₂`.
    pub final_rel_residual: f64,
}


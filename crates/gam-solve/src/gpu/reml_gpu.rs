//! Exact GPU REML evidence + derivative gradient.
//!
//! Refactor (Block 2.1, math team section 18): the penalized Hessian `H` is
//! Cholesky-factored exactly **once** on device, the factor is held resident,
//! and every derivative Hessian `H_j` is solved through the cached factor
//! with a single batched `potrs` call (`nrhs = d_rho · p`). Previously each
//! derivative re-issued the full `cholesky_solve_gpu` path, which uploaded
//! `H`, allocated and ran `potrf`, and downloaded the factor again — turning
//! a `p^3 + d·p^3` workload into a `(d+1)·p^3` one and serializing `d_rho`
//! factor passes onto the device.
//!
//! On the non-Linux fallback the same `cholesky_solve_gpu` path is exercised
//! via `pirls_gpu::cholesky_solve_gpu`, so behaviour outside Linux is
//! numerically identical (with the same per-derivative overhead) — the
//! optimisation is Linux-only because that is where CUDA actually runs.

use ndarray::{Array1, ArrayView2};

#[derive(Clone, Debug)]
pub struct RemlGpuInput<'a> {
    pub penalized_hessian: ArrayView2<'a, f64>,
    pub derivative_hessians: Vec<ArrayView2<'a, f64>>,
}

#[derive(Clone, Debug)]
pub struct RemlGpuEvidence {
    pub logdet_hessian: f64,
    pub gradient_rho: Array1<f64>,
}

pub fn evidence_derivatives_gpu(input: RemlGpuInput<'_>) -> Result<RemlGpuEvidence, String> {
    let p = input.penalized_hessian.nrows();
    if p != input.penalized_hessian.ncols() {
        return Err("REML GPU Hessian must be square".to_string());
    }
    for (j, derivative) in input.derivative_hessians.iter().enumerate() {
        if derivative.dim() != (p, p) {
            return Err(format!(
                "REML derivative Hessian {j} has shape {:?}, expected {p}x{p}",
                derivative.dim()
            ));
        }
    }

    #[cfg(target_os = "linux")]
    {
        if gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::global_policy())
            .map_err(|error| error.to_string())?
            .is_some()
        {
            return linux_cuda::evidence_derivatives(input);
        }
    }

    cpu_fallback::evidence_derivatives(input)
}

#[cfg(target_os = "linux")]
mod linux_cuda {
    use super::{RemlGpuEvidence, RemlGpuInput};
    use cudarc::cusolver::DnHandle;
    use gam_gpu::driver::to_col_major;
    use gam_gpu::solver::{
        cholesky_logdet_from_col_major, context_and_stream, pinned_htod, potrf_in_place,
        potrs_in_place,
    };
    use ndarray::Array1;

    pub(super) fn evidence_derivatives(input: RemlGpuInput<'_>) -> Result<RemlGpuEvidence, String> {
        let p = input.penalized_hessian.nrows();
        let d = input.derivative_hessians.len();
        let (_, stream) = context_and_stream()?;
        let solver = DnHandle::new(stream.clone()).map_err(|e| format!("cusolver init: {e}"))?;

        // Upload H once and factor in-place.
        let h_col = to_col_major(&input.penalized_hessian);
        let mut h_dev = pinned_htod(&stream, &h_col)?;
        potrf_in_place(&solver, &stream, p, &mut h_dev)?;
        let factor_col = stream
            .clone_dtoh(&h_dev)
            .map_err(|e| format!("download Cholesky factor: {e}"))?;
        let logdet_hessian = cholesky_logdet_from_col_major(&factor_col, p);

        if d == 0 {
            return Ok(RemlGpuEvidence {
                logdet_hessian,
                gradient_rho: Array1::<f64>::zeros(0),
            });
        }

        // Stack all derivative Hessians column-wise into ONE rhs of width d*p
        // and solve with a single batched potrs against the cached factor.
        let total_cols = p
            .checked_mul(d)
            .ok_or_else(|| format!("REML GPU RHS width overflow: p={p}, d={d}"))?;
        let total_elems = p
            .checked_mul(total_cols)
            .ok_or_else(|| format!("REML GPU RHS size overflow: p={p}, cols={total_cols}"))?;
        let mut rhs_col = Vec::<f64>::with_capacity(total_elems);
        for derivative in &input.derivative_hessians {
            let col = to_col_major(derivative);
            rhs_col.extend_from_slice(&col);
        }
        let mut rhs_dev = pinned_htod(&stream, &rhs_col)?;
        potrs_in_place(&solver, &stream, p, total_cols, &h_dev, &mut rhs_dev)?;
        let solved_col = stream
            .clone_dtoh(&rhs_dev)
            .map_err(|e| format!("download REML derivative solves: {e}"))?;

        let mut gradient_rho = Array1::<f64>::zeros(d);
        for j in 0..d {
            let offset = j * p * p;
            // Diagonal of H^{-1} A_j is the diagonal of the j-th p*p slab.
            let mut trace = 0.0_f64;
            for i in 0..p {
                trace += solved_col[offset + i * p + i];
            }
            gradient_rho[j] = 0.5 * trace;
        }

        Ok(RemlGpuEvidence {
            logdet_hessian,
            gradient_rho,
        })
    }
}

mod cpu_fallback {
    use super::{RemlGpuEvidence, RemlGpuInput};
    use ndarray::{Array1, Array2};

    pub(super) fn evidence_derivatives(input: RemlGpuInput<'_>) -> Result<RemlGpuEvidence, String> {
        let p = input.penalized_hessian.nrows();
        let mut identity = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            identity[[i, i]] = 1.0;
        }
        let (_, logdet_hessian) =
            crate::gpu::pirls_gpu::cholesky_solve_gpu(input.penalized_hessian, identity.view())?;
        let mut gradient_rho = Array1::<f64>::zeros(input.derivative_hessians.len());
        for (j, derivative) in input.derivative_hessians.iter().enumerate() {
            let (solved, _) = crate::gpu::pirls_gpu::cholesky_solve_gpu(
                input.penalized_hessian,
                derivative.view(),
            )?;
            let mut trace = 0.0_f64;
            for i in 0..p {
                trace += solved[[i, i]];
            }
            gradient_rho[j] = 0.5 * trace;
        }
        Ok(RemlGpuEvidence {
            logdet_hessian,
            gradient_rho,
        })
    }
}

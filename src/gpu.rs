//! GPU execution planning and host-side reference kernels.
//!
//! This module is the crate's GPU boundary.  It deliberately separates three
//! concerns that were previously implicit in the CPU-only fitting path:
//!
//! * the user policy (`auto`, `off`, `force`),
//! * workload/backend eligibility decisions, and
//! * bit-checkable reference kernels for the dense P-IRLS math that device
//!   backends must reproduce.
//!
//! The first production device backend can implement the same contracts without
//! changing callers: keep `X`, `beta`, `eta`, row weights, residual vectors,
//! Hessian buffers, candidate state, and trace workspaces resident on the device;
//! download only scalars or accepted/final states.  Until a backend is compiled
//! in, `auto` falls back to CPU and `force` fails loudly.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::fmt;
use std::str::FromStr;
use std::time::{Duration, Instant};

/// User-visible GPU selection policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum GpuPolicy {
    /// Let the planner use a supported GPU backend only for large eligible work.
    #[default]
    Auto,
    /// Never use GPU code.
    Off,
    /// Require GPU execution; return an error if the selected path is unsupported.
    Force,
}

impl GpuPolicy {
    /// Parse a policy string accepted by the public `FitConfig` surface.
    pub fn parse(value: &str) -> Result<Self, String> {
        value.parse()
    }
}

impl fmt::Display for GpuPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::Force => "force",
        })
    }
}

impl FromStr for GpuPolicy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            "force" => Ok(Self::Force),
            other => Err(format!(
                "invalid gpu policy '{other}'; expected one of: auto, off, force"
            )),
        }
    }
}

/// GPU backend family.  Device implementations should map these to CUDA,
/// HIP/ROCm, or Metal-specific stream/library handles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuBackendKind {
    Cuda,
    Hip,
    Metal,
}

/// High-level workload class used by backend selection and instrumentation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuWorkloadKind {
    DensePirlsSweep,
    LmCandidateScreen,
    DenseNewtonSolve,
    MatrixFreePcg,
    SparseAssembly,
    SpatialKernelOperator,
    MarginalSlopeRows,
    RemlTrace,
    FinalInference,
}

/// Shape and semantic constraints relevant to GPU selection.
#[derive(Clone, Debug)]
pub struct GpuWorkloadShape {
    pub kind: GpuWorkloadKind,
    pub n_rows: usize,
    pub n_cols: usize,
    pub nnz_per_row: Option<usize>,
    pub n_trace_probes: usize,
    pub has_signed_weights: bool,
    pub constrained: bool,
    pub firth: bool,
    pub exact_sparse_inference: bool,
    pub family_gpu_safe: bool,
}

impl GpuWorkloadShape {
    /// Dense materialized P-IRLS shape helper.
    pub fn dense_pirls(n_rows: usize, n_cols: usize, has_signed_weights: bool) -> Self {
        Self {
            kind: GpuWorkloadKind::DensePirlsSweep,
            n_rows,
            n_cols,
            nnz_per_row: None,
            n_trace_probes: 0,
            has_signed_weights,
            constrained: false,
            firth: false,
            exact_sparse_inference: false,
            family_gpu_safe: true,
        }
    }

    fn row_work(&self) -> usize {
        self.n_rows.saturating_mul(self.n_cols.max(1))
    }
}

/// Result of backend planning.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuDecision {
    UseCpu {
        reason: String,
    },
    UseGpu {
        backend: GpuBackendKind,
        route: String,
    },
}

impl GpuDecision {
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::UseGpu { .. })
    }

    pub fn reason(&self) -> &str {
        match self {
            Self::UseCpu { reason } => reason,
            Self::UseGpu { route, .. } => route,
        }
    }
}

/// List compiled/available GPU backends.
///
/// The crate currently ships the planner and reference contracts but no linked
/// CUDA/HIP/Metal backend, so this returns an empty list.  Backend crates should
/// replace this probe behind features and keep `auto` deterministic.
pub fn available_backends() -> Vec<GpuBackendKind> {
    Vec::new()
}

/// Plan a GPU workload under a user policy.
pub fn plan_workload(policy: GpuPolicy, shape: &GpuWorkloadShape) -> Result<GpuDecision, String> {
    if policy == GpuPolicy::Off {
        return Ok(GpuDecision::UseCpu {
            reason: "gpu policy is off".to_string(),
        });
    }

    if !shape.family_gpu_safe {
        return unsupported(policy, "family/path uses CPU-only callbacks");
    }
    if shape.firth {
        return unsupported(policy, "Firth/Jeffreys diagnostics are CPU-only");
    }
    if shape.constrained {
        return unsupported(policy, "constrained solves are CPU-only");
    }
    if shape.exact_sparse_inference {
        return unsupported(policy, "exact sparse Takahashi inference remains CPU-only");
    }

    let backends = available_backends();
    let Some(backend) = backends.first().copied() else {
        return unsupported(policy, "no GPU backend is compiled and available");
    };

    let large_enough = match shape.kind {
        GpuWorkloadKind::DensePirlsSweep => shape.row_work() >= 1_000_000,
        GpuWorkloadKind::LmCandidateScreen => shape.row_work() >= 250_000,
        GpuWorkloadKind::DenseNewtonSolve => shape.n_cols >= 256,
        GpuWorkloadKind::MatrixFreePcg => shape.n_cols >= 2_048,
        GpuWorkloadKind::SparseAssembly => shape.n_rows >= 100_000 || shape.n_cols >= 2_048,
        GpuWorkloadKind::SpatialKernelOperator => shape.n_rows >= 25_000,
        GpuWorkloadKind::MarginalSlopeRows => shape.n_rows >= 25_000,
        GpuWorkloadKind::RemlTrace => shape.n_trace_probes >= 8 || shape.n_cols >= 512,
        GpuWorkloadKind::FinalInference => shape.n_cols >= 256 && shape.n_cols <= 5_000,
    };

    if !large_enough && policy == GpuPolicy::Auto {
        return Ok(GpuDecision::UseCpu {
            reason: format!("workload {:?} is below GPU auto threshold", shape.kind),
        });
    }

    Ok(GpuDecision::UseGpu {
        backend,
        route: format!("device-resident {:?}", shape.kind),
    })
}

fn unsupported(policy: GpuPolicy, reason: &str) -> Result<GpuDecision, String> {
    if policy == GpuPolicy::Force {
        Err(format!("gpu=force requested but unsupported: {reason}"))
    } else {
        Ok(GpuDecision::UseCpu {
            reason: reason.to_string(),
        })
    }
}

/// Lightweight stage timer for GPU/CPU parity instrumentation.
#[derive(Debug)]
pub struct GpuStageTimer {
    stage: &'static str,
    start: Instant,
}

impl GpuStageTimer {
    pub fn start(stage: &'static str) -> Self {
        Self {
            stage,
            start: Instant::now(),
        }
    }

    pub fn finish(self) -> GpuStageTiming {
        GpuStageTiming {
            stage: self.stage,
            elapsed: self.start.elapsed(),
        }
    }
}

/// Completed stage timing record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuStageTiming {
    pub stage: &'static str,
    pub elapsed: Duration,
}

/// Host-side reference dense kernels that device backends must match.
pub mod dense {
    use super::*;

    /// Sign-preserving `Xᵀ diag(weights) X` for observed Hessian assembly.
    ///
    /// This intentionally multiplies rows by `w_i`, not `sqrt(max(w_i, 0))`, so
    /// negative observed-information weights are preserved exactly.
    pub fn xtwx_signed(x: ArrayView2<'_, f64>, weights: ArrayView1<'_, f64>) -> Array2<f64> {
        assert_eq!(x.nrows(), weights.len(), "X/weight row mismatch");
        let p = x.ncols();
        let mut out = Array2::<f64>::zeros((p, p));
        for i in 0..x.nrows() {
            let w = weights[i];
            if w == 0.0 {
                continue;
            }
            for a in 0..p {
                let xia = x[(i, a)];
                for b in a..p {
                    out[(a, b)] += xia * w * x[(i, b)];
                }
            }
        }
        mirror_upper_to_lower(&mut out);
        out
    }

    /// Fisher-positive `Xᵀ W X` reference path using row scaling by `sqrt(W)`.
    pub fn xtwx_positive_sqrt(
        x: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        if x.nrows() != weights.len() {
            return Err(format!(
                "X/weight row mismatch: {} rows vs {} weights",
                x.nrows(),
                weights.len()
            ));
        }
        if let Some((idx, value)) = weights.iter().enumerate().find(|(_, value)| **value < 0.0) {
            return Err(format!(
                "positive-weight XtWX path received negative weight at row {idx}: {value}"
            ));
        }
        Ok(xtwx_signed(x, weights))
    }

    /// `Xᵀ residual`, used for dense P-IRLS gradients.
    pub fn xt_residual(x: ArrayView2<'_, f64>, residual: ArrayView1<'_, f64>) -> Array1<f64> {
        assert_eq!(x.nrows(), residual.len(), "X/residual row mismatch");
        let mut out = Array1::<f64>::zeros(x.ncols());
        for i in 0..x.nrows() {
            let ri = residual[i];
            if ri == 0.0 {
                continue;
            }
            for j in 0..x.ncols() {
                out[j] += x[(i, j)] * ri;
            }
        }
        out
    }

    /// Candidate-screen update: `eta_candidate = eta_current + X delta`.
    pub fn candidate_eta_from_delta(
        x: ArrayView2<'_, f64>,
        eta_current: ArrayView1<'_, f64>,
        delta: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        assert_eq!(x.nrows(), eta_current.len(), "X/eta row mismatch");
        assert_eq!(x.ncols(), delta.len(), "X/delta column mismatch");
        let mut out = eta_current.to_owned();
        for i in 0..x.nrows() {
            let mut inc = 0.0;
            for j in 0..x.ncols() {
                inc += x[(i, j)] * delta[j];
            }
            out[i] += inc;
        }
        out
    }

    fn mirror_upper_to_lower(out: &mut Array2<f64>) {
        let p = out.ncols();
        for a in 0..p {
            for b in 0..a {
                out[(a, b)] = out[(b, a)];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn parses_gpu_policy() {
        assert_eq!(GpuPolicy::parse("auto").unwrap(), GpuPolicy::Auto);
        assert_eq!(GpuPolicy::parse("OFF").unwrap(), GpuPolicy::Off);
        assert_eq!(GpuPolicy::parse("force").unwrap(), GpuPolicy::Force);
        assert!(GpuPolicy::parse("yes").is_err());
    }

    #[test]
    fn force_fails_without_backend() {
        let shape = GpuWorkloadShape::dense_pirls(320_000, 42, false);
        let err = plan_workload(GpuPolicy::Force, &shape).unwrap_err();
        assert!(err.contains("gpu=force"));
    }

    #[test]
    fn signed_xtwx_preserves_negative_weights() {
        let x = array![[1.0, 2.0], [3.0, -1.0], [2.0, 4.0]];
        let w = array![2.0, -0.5, 1.5];
        let got = dense::xtwx_signed(x.view(), w.view());
        let expected = array![[3.5, 17.5], [17.5, 31.5]];
        assert_eq!(got, expected);
    }

    #[test]
    fn positive_sqrt_rejects_negative_weights() {
        let x = array![[1.0], [2.0]];
        let w = array![1.0, -1.0];
        assert!(dense::xtwx_positive_sqrt(x.view(), w.view()).is_err());
    }

    #[test]
    fn candidate_eta_uses_delta_update() {
        let x = array![[1.0, 2.0], [3.0, -1.0]];
        let eta = array![10.0, -2.0];
        let delta = array![0.5, 4.0];
        let got = dense::candidate_eta_from_delta(x.view(), eta.view(), delta.view());
        assert_eq!(got, array![18.5, -4.5]);
    }
}

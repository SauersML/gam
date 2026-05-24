use ndarray::{Array1, Array2, ArrayView2};

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
    let mut identity = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        identity[[i, i]] = 1.0;
    }
    let (_, logdet_hessian) =
        super::pirls_gpu::cholesky_solve_gpu(input.penalized_hessian, identity.view())?;
    let mut gradient_rho = Array1::<f64>::zeros(input.derivative_hessians.len());
    for (j, derivative) in input.derivative_hessians.iter().enumerate() {
        if derivative.dim() != (p, p) {
            return Err(format!(
                "REML derivative Hessian {j} has shape {:?}, expected {p}x{p}",
                derivative.dim()
            ));
        }
        let (solved, _) =
            super::pirls_gpu::cholesky_solve_gpu(input.penalized_hessian, derivative.view())?;
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

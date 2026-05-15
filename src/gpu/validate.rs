#[derive(Clone, Debug)]
pub struct ValidationTolerances {
    pub beta_rel: f64,
    pub gradient_rel: f64,
    pub objective_rel: f64,
    pub hessian_rel: f64,
    pub trace_rel: f64,
}

impl Default for ValidationTolerances {
    fn default() -> Self {
        Self {
            beta_rel: 1e-9,
            gradient_rel: 1e-9,
            objective_rel: 1e-12,
            hessian_rel: 1e-9,
            trace_rel: 1e-6,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ValidationReport {
    pub passed: bool,
    pub metric: &'static str,
    pub observed: f64,
    pub tolerance: f64,
}

pub fn compare_scalar(metric: &'static str, cpu: f64, gpu: f64, rel_tol: f64) -> ValidationReport {
    let scale = cpu.abs().max(1.0);
    let observed = (cpu - gpu).abs() / scale;
    ValidationReport {
        passed: observed <= rel_tol,
        metric,
        observed,
        tolerance: rel_tol,
    }
}

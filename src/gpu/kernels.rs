/// Validation tolerances for GPU-vs-CPU comparisons.  GPU reductions are not
/// expected to be bit-identical because their addition tree differs.
#[derive(Clone, Copy, Debug)]
pub struct ValidationTolerances {
    pub relative: f64,
    pub absolute: f64,
}

impl Default for ValidationTolerances {
    fn default() -> Self {
        Self {
            relative: 1e-10,
            absolute: 1e-10,
        }
    }
}

#[must_use]
pub fn within_tolerance(cpu: f64, gpu: f64, tol: ValidationTolerances) -> bool {
    let scale = cpu.abs().max(gpu.abs()).max(1.0);
    (cpu - gpu).abs() <= tol.absolute.max(tol.relative * scale)
}

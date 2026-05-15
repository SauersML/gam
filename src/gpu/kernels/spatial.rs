#[derive(Clone, Debug)]
pub enum SpatialKernel {
    ThinPlate {
        length_scale_sq: f64,
        dim: usize,
    },
    Polyharmonic {
        m: usize,
        k_dim: usize,
        c: f64,
        power: f64,
        is_log_case: bool,
    },
    Matern {
        length_scale_sq: f64,
        nu: f64,
        dim: usize,
    },
    DuchonAnisotropic {
        aniso_log_scales: Vec<f64>,
        m: usize,
        dim: usize,
    },
    GenericCpuOnly,
}

use std::env;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidationMode {
    Off,
    Every(usize),
    Strict,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixedPrecisionMode {
    Off,
    Screening,
    Never,
}

#[derive(Clone, Debug)]
pub struct GpuEnv {
    pub gpu: String,
    pub device: Option<usize>,
    pub mem_fraction: f64,
    pub validate: ValidationMode,
    pub graphs: String,
    pub mixed_precision: MixedPrecisionMode,
    pub profile: bool,
    pub calibrate: String,
    pub min_n_override: Option<usize>,
    pub f32_operator_preconditioner: bool,
    pub family_kernels: bool,
}

impl Default for GpuEnv {
    fn default() -> Self {
        Self::from_process_env()
    }
}

impl GpuEnv {
    pub fn from_process_env() -> Self {
        let gpu = env::var("GAM_GPU").unwrap_or_else(|_| "auto".to_string());
        let device = env::var("GAM_GPU_DEVICE").ok().and_then(|v| v.parse().ok());
        let mem_fraction = env::var("GAM_GPU_MEM_FRACTION")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.85);
        let validate = match env::var("GAM_GPU_VALIDATE")
            .unwrap_or_else(|_| "0".to_string())
            .as_str()
        {
            "1" => ValidationMode::Strict,
            "0" => ValidationMode::Off,
            v if v.starts_with("every:") => v[6..]
                .parse()
                .map(ValidationMode::Every)
                .unwrap_or(ValidationMode::Off),
            _ => ValidationMode::Off,
        };
        let graphs = env::var("GAM_GPU_GRAPHS").unwrap_or_else(|_| "auto".to_string());
        let mixed_precision = match env::var("GAM_GPU_MIXED_PRECISION")
            .unwrap_or_else(|_| "off".to_string())
            .as_str()
        {
            "screening" => MixedPrecisionMode::Screening,
            "never" => MixedPrecisionMode::Never,
            _ => MixedPrecisionMode::Off,
        };
        let profile = env::var("GAM_GPU_PROFILE").is_ok_and(|v| v == "1");
        let calibrate = env::var("GAM_GPU_CALIBRATE").unwrap_or_else(|_| "auto".to_string());
        let min_n_override = env::var("GAM_GPU_MIN_N").ok().and_then(|v| v.parse().ok());
        let f32_operator_preconditioner =
            env::var("GAM_GPU_F32_OPERATOR_PRECONDITIONER").is_ok_and(|v| v == "1");
        let family_kernels =
            env::var("GAM_GPU_FAMILY_KERNELS").is_ok_and(|v| v == "on" || v == "1");
        Self {
            gpu,
            device,
            mem_fraction,
            validate,
            graphs,
            mixed_precision,
            profile,
            calibrate,
            min_n_override,
            f32_operator_preconditioner,
            family_kernels,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GpuDispatchPolicy {
    pub xtwx_n_min: usize,
    pub xtwx_flops_min: f64,
    pub xtwx_use_fused_below_p: usize,
    pub gemm_min_flops: f64,
    pub potrf_min_p: usize,
    pub syevd_min_p: usize,
    pub sparse_min_nnz: usize,
    pub fused_kernel_min_n: usize,
    pub keep_design_resident_min_bytes: usize,
    pub prefer_gpu_factorization_min_p: usize,
    pub row_kernel_min_n: usize,
}

impl Default for GpuDispatchPolicy {
    fn default() -> Self {
        Self {
            xtwx_n_min: 8_192,
            xtwx_flops_min: 2.0e8,
            xtwx_use_fused_below_p: 256,
            gemm_min_flops: 5.0e7,
            potrf_min_p: 256,
            syevd_min_p: 256,
            sparse_min_nnz: 100_000,
            fused_kernel_min_n: 8_192,
            keep_design_resident_min_bytes: 8 * 1024 * 1024,
            prefer_gpu_factorization_min_p: 256,
            row_kernel_min_n: 8_192,
        }
    }
}

impl GpuDispatchPolicy {
    pub fn should_gpu_gemv(&self, n: usize, p: usize, resident: bool) -> bool {
        resident || 2.0 * n as f64 * p as f64 >= self.gemm_min_flops
    }

    pub fn should_gpu_xtwx(&self, n: usize, p: usize, materialized: bool) -> bool {
        materialized && n >= self.xtwx_n_min && n as f64 * (p * p) as f64 >= self.xtwx_flops_min
    }
}

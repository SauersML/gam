//! NVRTC Device Backend for Bernoulli Marginal Slope Flex Kernels
//!
//! This module provides the infrastructure to compile and launch custom CUDA C++
//! kernels at runtime for evaluating BMS flex row-primary jets and accumulating
//! the Hessian and HVP.

use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaStream, CudaSlice, DevicePtr, DevicePtrMut, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use crate::gpu::error::GpuError;

const BMS_FLEX_KERNEL_CU: &str = r#"
extern "C" __global__ void bms_flex_row_kernel(
    const double* q,
    const double* b,
    const double* beta_h,
    const double* beta_w,
    const double* y,
    const double* weights,
    double* out_gradient,
    double* out_hessian,
    int n,
    int r
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    
    // TODO: Implement `compute_row_analytic_flex_from_parts_into` math here.
    // This requires polynomial interpolations and normal CDF/PDF implementations
    // mapped from the CPU exact kernel.
}
"#;

pub struct BmsFlexGpuBackend {
    device: Arc<CudaDevice>,
    module_name: &'static str,
}

impl BmsFlexGpuBackend {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, GpuError> {
        let ptx = compile_ptx(BMS_FLEX_KERNEL_CU)
            .map_err(|e| GpuError::DriverCallFailed { reason: format!("NVRTC compilation failed: {:?}", e) })?;
        
        let module_name = "bms_flex";
        device.load_ptx(ptx, module_name, &["bms_flex_row_kernel"])
            .map_err(|e| GpuError::DriverCallFailed { reason: format!("Failed to load PTX module: {:?}", e) })?;
            
        Ok(Self { device, module_name })
    }
    
    pub fn launch_row_kernel(
        &self,
        stream: &Arc<CudaStream>,
        n: usize,
        r: usize,
        d_q: &CudaSlice<f64>,
        d_b: &CudaSlice<f64>,
        d_beta_h: &CudaSlice<f64>,
        d_beta_w: &CudaSlice<f64>,
        d_y: &CudaSlice<f64>,
        d_weights: &CudaSlice<f64>,
        d_out_gradient: &mut CudaSlice<f64>,
        d_out_hessian: &mut CudaSlice<f64>,
    ) -> Result<(), GpuError> {
        let f = self.device.get_func(self.module_name, "bms_flex_row_kernel")
            .ok_or_else(|| GpuError::DriverCallFailed { reason: "Kernel not found".into() })?;
            
        let cfg = LaunchConfig::for_num_elems(n as u32);
        
        unsafe {
            f.clone().launch_on_stream(
                stream,
                cfg,
                (
                    d_q, d_b, d_beta_h, d_beta_w, d_y, d_weights,
                    d_out_gradient, d_out_hessian, n as i32, r as i32
                )
            )
        }.map_err(|e| GpuError::DriverCallFailed { reason: format!("Launch failed: {:?}", e) })?;
        
        Ok(())
    }
}

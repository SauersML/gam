//! Whether this host's CUDA device runs gam-gpu's tensors. Every CUDA test resolves its device with
//! `Device::accelerator(GpuPolicy::Auto)` and returns early when that gives none, as on a host
//! without CUDA (the CI runners, a Mac). Under `auto` an absent device, a device the runtime cannot
//! use, and a kernel module that does not compile all give none, so a passing run of the CUDA tests
//! shows nothing by itself. Run with `--require` (the CUDA gate runs it so before the CUDA tests),
//! this binary fails unless `auto` gives a device that moves a matrix exactly in both of its
//! storages (float64, and f32 as the fits use), and names what `required` reports otherwise.
//! Without `--require` it reports what it found and passes. It has its own `main`
//! (`harness = false` in Cargo.toml) so that it can read that argument.

use gam_gpu::GpuPolicy;
use gam_gpu::tensor::{Device, Storage};
use ndarray::Array2;

fn main() -> Result<(), String> {
    let require = std::env::args().skip(1).any(|a| a == "--require");
    let Some(wide) = Device::accelerator(GpuPolicy::Auto).map_err(|e| format!("the CUDA probe faulted: {e}"))? else {
        if !require {
            eprintln!("cuda_device: no CUDA device resolves; the CUDA tests return early on this host");
            return Ok(());
        }
        let reason = match Device::accelerator(GpuPolicy::Required) {
            Err(e) => e.to_string(),
            Ok(_) => "`required` resolves a device that `auto` does not".to_string(),
        };
        return Err(format!("no CUDA device resolves, so the CUDA tests would return early without running: {reason}"));
    };
    // Values that f32 holds exactly, so both storages must return them bit for bit.
    let values = Array2::from_shape_fn((3, 5), |(i, j)| (i as f64 - 1.5) * 0.25 + j as f64);
    let narrow = wide.with_storage(Storage::F32).map_err(|e| format!("{}: no f32 storage: {e}", wide.name()))?;
    for device in [&wide, &narrow] {
        let back = device.download(&device.upload(values.view()).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        if back != values {
            return Err(format!("{}: a matrix does not move exactly: {back:?}", device.name()));
        }
    }
    eprintln!("cuda_device: {} resolves and moves values exactly in float64 and f32", wide.name());
    Ok(())
}

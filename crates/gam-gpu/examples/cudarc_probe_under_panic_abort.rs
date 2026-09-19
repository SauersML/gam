//! Run the process-wide CUDA probe in a binary built with `panic = "abort"`.
//! Usage: cudarc_probe_under_panic_abort
//! Prints the panic strategy it was built with, then one outcome line: `absent: <reason>` or
//! `available: <device count>`. A probe fault returns the error, which exits 1. A probe that
//! reaches a cudarc loader panic aborts the process instead, because no `catch_unwind` runs under
//! this strategy.

use gam_gpu::{GpuAvailabilityRef, GpuError, GpuRuntime};

fn main() -> Result<(), GpuError> {
    let strategy = if cfg!(panic = "abort") { "abort" } else { "unwind" };
    println!("panic={strategy}");
    match GpuRuntime::availability()? {
        GpuAvailabilityRef::Absent(absence) => println!("absent: {absence}"),
        GpuAvailabilityRef::Available(runtime) => {
            println!("available: {}", runtime.devices.len());
        }
    }
    Ok(())
}

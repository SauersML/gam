//! Time the banded Metal products against the CPU float64 GEMM every gam
//! product takes (`gam_linalg`'s faer), on the shapes of the MPD masked
//! trainer on VPD 4L (512 rows, d = 768, d_mlp = 3072, vocab 50277, about
//! 1200 pieces per site). Wall clock includes conversion, upload and readback.
//! `cargo run --release -p gam-gpu --example metal_gemm_speed`.

use gam_gpu::GpuPolicy;
use gam_gpu::banded::{BandedArithmetic, Layout, resident_operand};
use gam_linalg::faer_ndarray::{fast_ab, fast_abt};
use ndarray::Array2;
use std::time::Instant;

fn matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    Array2::from_shape_simple_fn((rows, cols), || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    })
}

fn best<T>(mut run: impl FnMut() -> T) -> (f64, T) {
    let mut last = run();
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let start = Instant::now();
        last = run();
        best = best.min(start.elapsed().as_secs_f64());
    }
    (best, last)
}

fn main() -> Result<(), String> {
    let rows = 512;
    // (label, stored rows, stored cols, layout): the stored weight and how the product reads it.
    let shapes = [
        ("site V: x·Vᵀ (V 1200×768)", 1200, 768, Layout::Transposed),
        ("site U: z·Uᵀ (U 3072×1200)", 3072, 1200, Layout::Transposed),
        ("vjp: cot·V (V 1200×768)", 1200, 768, Layout::AsStored),
        ("unembed: h·E (E 768×50277)", 768, 50277, Layout::AsStored),
        ("unembed vjp: cot·Eᵀ", 768, 50277, Layout::Transposed),
    ];
    for (label, r, c, layout) in shapes {
        let stored = matrix(r, c, 7);
        let k = if layout == Layout::AsStored { r } else { c };
        let x = matrix(rows, k, 11);
        let (cpu, exact) = best(|| match layout {
            Layout::AsStored => fast_ab(&x, &stored),
            Layout::Transposed => fast_abt(&x, &stored),
        });
        let mut line = format!("{label}: cpu f64 {:.1} ms", cpu * 1e3);
        for arithmetic in [BandedArithmetic::F32, BandedArithmetic::Df64] {
            let resident = resident_operand(GpuPolicy::Required, arithmetic, stored.view())
                .map_err(|e| e.to_string())?
                .ok_or("no resident operand")?;
            let (wall, product) = best(|| resident.product(x.view(), layout));
            let product = product.map_err(|e| e.to_string())?.ok_or("cpu route")?;
            let error = (&product.values - &exact).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            line += &format!(
                " | {:?} {:.1} ms (device {:.1} ms), max |dev−f64| {:.1e}, band {:.1e}",
                arithmetic,
                wall * 1e3,
                product.device_seconds * 1e3,
                error,
                product.band.max_entry()
            );
        }
        println!("{line}");
    }
    Ok(())
}

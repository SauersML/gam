//! Time `Device::causal_attention` and its reverse on CUDA at the fit's shapes: vpd4l (6 heads of
//! 128, each with its own keys and values, 32 sequences) and Qwen3-0.6B (16 query heads over 8
//! key-value heads of 128, 8 sequences), each at 256, 512, 1024 and 2048 positions. A line per shape: the mean
//! device seconds per call of the forward and of the reverse (one warm call, then a timed loop
//! between synchronizations), their rates in TFLOP/s, and the fraction of PEAK, the card's f32 rate
//! in TFLOP/s (the kernels multiply in f32 on the CUDA cores; RTX 4090: 82.6). Operations are
//! counted as the products a causal attention needs: the forward 4 w Σ T(T + 1)/2 per query head
//! (scores and values), the reverse 2.5 times that (scores again, the weights' cotangent, and the
//! queries', keys' and values' cotangents; the reverse's two passes form the scores and the weights'
//! cotangent each, which this count does not credit).
//! `cargo run --release -p gam-gpu --example attention_speed -- PEAK`.

use gam_gpu::GpuPolicy;
use gam_gpu::tensor::{Device, HeadLayout, Storage};
use ndarray::Array2;
use std::time::Instant;

fn matrix(rows: usize, cols: usize, seed: u64, scale: f64) -> Array2<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    Array2::from_shape_simple_fn((rows, cols), || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0 * scale
    })
}

/// Mean seconds per call: one warm call, then `reps` between synchronizations.
fn time(device: &Device, reps: usize, mut op: impl FnMut() -> Result<(), String>) -> Result<f64, String> {
    op()?;
    device.synchronize().map_err(|e| e.to_string())?;
    let start = Instant::now();
    for _ in 0..reps {
        op()?;
    }
    device.synchronize().map_err(|e| e.to_string())?;
    Ok(start.elapsed().as_secs_f64() / reps as f64)
}

fn main() -> Result<(), String> {
    let peak: f64 = std::env::args().nth(1).ok_or("usage: attention_speed PEAK_TFLOPS")?.parse().map_err(|e| format!("PEAK: {e}"))?;
    let device = Device::accelerator(GpuPolicy::Auto).map_err(|e| e.to_string())?.ok_or("no accelerator")?;
    let d = device.with_storage(Storage::F32).map_err(|e| e.to_string())?;
    println!("{}", d.name());
    let shapes = [("vpd4l", HeadLayout { queries: 6, keys: 6, width: 128 }, 32), ("qwen3-0.6b", HeadLayout { queries: 16, keys: 8, width: 128 }, 8)];
    for (name, layout, count) in shapes {
        for length in [256, 512, 1024, 2048] {
            let rows = count * length;
            let sequences: Vec<_> = (0..count).map(|s| s * length..(s + 1) * length).collect();
            let y = d.upload(matrix(rows, layout.columns(), 1, 2.0).view()).map_err(|e| e.to_string())?;
            let ga = d.upload(matrix(rows, layout.queries * layout.width, 2, 1.0).view()).map_err(|e| e.to_string())?;
            let scale = 1.0 / (layout.width as f64).sqrt();
            let (out, lse) = d.causal_attention(&y, layout, &sequences, scale).map_err(|e| e.to_string())?;
            let reps = 20;
            let forward = time(&d, reps, || d.causal_attention(&y, layout, &sequences, scale).map(|_| ()).map_err(|e| e.to_string()))?;
            let reverse = time(&d, reps, || d.causal_attention_backward(&y, layout, &sequences, scale, (&out, &lse), &ga).map(|_| ()).map_err(|e| e.to_string()))?;
            let flops = 4.0 * layout.width as f64 * layout.queries as f64 * count as f64 * (length * (length + 1) / 2) as f64;
            let (f_rate, r_rate) = (flops / forward / 1e12, 2.5 * flops / reverse / 1e12);
            println!(
                "{name} {count}x{length}: forward {:.3} ms {f_rate:.1} TFLOP/s ({:.1}% of peak), reverse {:.3} ms {r_rate:.1} TFLOP/s ({:.1}% of peak), both {:.3} ms",
                forward * 1e3,
                100.0 * f_rate / peak,
                reverse * 1e3,
                100.0 * r_rate / peak,
                (forward + reverse) * 1e3
            );
        }
    }
    Ok(())
}

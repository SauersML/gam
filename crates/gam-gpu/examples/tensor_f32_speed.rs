//! Time the CUDA tensor operations a fitting step runs, in float64 storage and in f32 storage on
//! the same device (`Device::with_storage`), on the shapes of a Qwen3-0.6B-sized step: 512 rows,
//! d = 1024, d_ffn = 3072, 16 heads of 64 over 512 positions, vocabulary 151936. Each line is the
//! mean device time per call over a timed loop (synchronized before and after; a call that returns
//! per-row values downloads them), in float64 then f32, and the ratio.
//! `cargo run --release -p gam-gpu --example tensor_f32_speed`.

use gam_gpu::GpuPolicy;
use gam_gpu::gpu_error::GpuError;
use gam_gpu::tensor::{Arithmetic, Device, Op, PointwiseLaw, Storage, Tensor};
use ndarray::Array2;
use std::io::Write;
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
fn time(device: &Device, reps: usize, mut op: impl FnMut() -> Result<(), GpuError>) -> Result<f64, String> {
    op().map_err(|e| e.to_string())?;
    device.synchronize().map_err(|e| e.to_string())?;
    let start = Instant::now();
    for _ in 0..reps {
        op().map_err(|e| e.to_string())?;
    }
    device.synchronize().map_err(|e| e.to_string())?;
    Ok(start.elapsed().as_secs_f64() / reps as f64)
}

fn up(device: &Device, m: &Array2<f64>) -> Result<Tensor, String> {
    device.upload(m.view()).map_err(|e| e.to_string())
}

struct Report {
    out: std::io::Stdout,
}

impl Report {
    fn line(&mut self, what: &str, wide: f64, narrow: f64) -> Result<(), String> {
        writeln!(self.out, "{what:<58} f64 {:>9.3} ms   f32 {:>9.3} ms   x{:>6.1}", wide * 1e3, narrow * 1e3, wide / narrow)
            .map_err(|e| e.to_string())
    }
}

fn main() -> Result<(), String> {
    let wide = Device::accelerator(GpuPolicy::Required).map_err(|e| e.to_string())?.ok_or("a CUDA device is required")?;
    let narrow = wide.with_storage(Storage::F32).map_err(|e| e.to_string())?;
    let mut report = Report { out: std::io::stdout() };
    writeln!(report.out, "{}", wide.name()).map_err(|e| e.to_string())?;
    let (rows, d, ffn, vocab, heads, positions, head_dim) = (512, 1024, 3072, 151_936, 16, 512, 64);
    let both = |m: &Array2<f64>| -> Result<(Tensor, Tensor), String> { Ok((up(&wide, m)?, up(&narrow, m)?)) };

    // Products: an MLP projection, attention scores (strided-batched) and the unembedding.
    let (x, w) = (matrix(rows, d, 1, 1.0), matrix(d, ffn, 2, 0.05));
    let ((xw, xn), (ww, wn)) = (both(&x)?, both(&w)?);
    let (mut cw, mut cn) = (wide.zeros(rows, ffn).map_err(|e| e.to_string())?, narrow.zeros(rows, ffn).map_err(|e| e.to_string())?);
    let f64_mlp = time(&wide, 20, || wide.gemm(&mut cw, 1.0, &xw, Op::N, &ww, Op::N, 0.0, Arithmetic::F64))?;
    let lowered = time(&wide, 20, || wide.gemm(&mut cw, 1.0, &xw, Op::N, &ww, Op::N, 0.0, Arithmetic::F32))?;
    let f32_mlp = time(&narrow, 50, || narrow.gemm(&mut cn, 1.0, &xn, Op::N, &wn, Op::N, 0.0, Arithmetic::F32))?;
    let tf32_mlp = time(&narrow, 50, || narrow.gemm(&mut cn, 1.0, &xn, Op::N, &wn, Op::N, 0.0, Arithmetic::Tf32))?;
    report.line("gemm 512x1024 . 1024x3072 (F32 arithmetic)", f64_mlp, f32_mlp)?;
    report.line("gemm 512x1024 . 1024x3072 (TF32 arithmetic)", f64_mlp, tf32_mlp)?;
    report.line("gemm 512x1024 . 1024x3072 f64 storage lowered F32 vs f32", lowered, f32_mlp)?;
    let bf16_mlp = time(&narrow, 50, || narrow.gemm(&mut cn, 1.0, &xn, Op::N, &wn, Op::N, 0.0, Arithmetic::Bf16))?;
    let frozen = narrow.bf16_copy(&wn).map_err(|e| e.to_string())?;
    let frozen_mlp = time(&narrow, 50, || narrow.gemm(&mut cn, 1.0, &xn, Op::N, &frozen, Op::N, 0.0, Arithmetic::Bf16))?;
    report.line("gemm 512x1024 . 1024x3072 (BF16, both rounded per call)", f64_mlp, bf16_mlp)?;
    report.line("gemm 512x1024 . 1024x3072 (BF16, frozen bf16 weight)", f64_mlp, frozen_mlp)?;
    let (q, k) = (matrix(heads * positions, head_dim, 3, 1.0), matrix(heads * positions, head_dim, 4, 1.0));
    let ((qw, qn), (kw, kn)) = (both(&q)?, both(&k)?);
    let (mut sw, mut sn) = (wide.zeros(heads * positions, positions).map_err(|e| e.to_string())?, narrow.zeros(heads * positions, positions).map_err(|e| e.to_string())?);
    let f64_scores = time(&wide, 20, || wide.gemm_batched(heads, &mut sw, 0.125, &qw, Op::N, &kw, Op::T, 0.0, Arithmetic::F64))?;
    let f32_scores = time(&narrow, 50, || narrow.gemm_batched(heads, &mut sn, 0.125, &qn, Op::N, &kn, Op::T, 0.0, Arithmetic::F32))?;
    report.line("batched scores 16 x (512x64 . 64x512)", f64_scores, f32_scores)?;
    let causal_w = time(&wide, 20, || wide.softmax_rows(&mut sw, true))?;
    let causal_n = time(&narrow, 50, || narrow.softmax_rows(&mut sn, true))?;
    report.line("causal softmax 8192 x 512", causal_w, causal_n)?;
    let embedding = matrix(vocab, d, 5, 0.05);
    let (ew, en) = both(&embedding)?;
    let hidden = matrix(rows, d, 6, 1.0);
    let (hw, hn) = both(&hidden)?;
    let (mut lw, mut ln) = (wide.zeros(rows, vocab).map_err(|e| e.to_string())?, narrow.zeros(rows, vocab).map_err(|e| e.to_string())?);
    let f64_head = time(&wide, 3, || wide.gemm(&mut lw, 1.0, &hw, Op::N, &ew, Op::T, 0.0, Arithmetic::F64))?;
    let f32_head = time(&narrow, 10, || narrow.gemm(&mut ln, 1.0, &hn, Op::N, &en, Op::T, 0.0, Arithmetic::F32))?;
    let tf32_head = time(&narrow, 10, || narrow.gemm(&mut ln, 1.0, &hn, Op::N, &en, Op::T, 0.0, Arithmetic::Tf32))?;
    report.line("unembedding 512x1024 . (151936x1024)^T (F32)", f64_head, f32_head)?;
    report.line("unembedding 512x1024 . (151936x1024)^T (TF32)", f64_head, tf32_head)?;
    let frozen_head = narrow.bf16_copy(&en).map_err(|e| e.to_string())?;
    let bf16_head = time(&narrow, 10, || narrow.gemm(&mut ln, 1.0, &hn, Op::N, &frozen_head, Op::T, 0.0, Arithmetic::Bf16))?;
    report.line("unembedding 512x1024 . (151936x1024)^T (BF16, frozen head)", f64_head, bf16_head)?;

    // Vocabulary rows: KL with its cotangent, softmax statistics, and the head log partition.
    let target = matrix(rows, vocab, 7, 8.0);
    let (tw, tn) = both(&target)?;
    let logits = matrix(rows, vocab, 8, 8.0);
    let (mut zw, mut zn) = both(&logits)?;
    let kl_w = time(&wide, 5, || wide.kl_rows(&tw, &mut zw, None).map(|_| ()))?;
    let kl_n = time(&narrow, 20, || narrow.kl_rows(&tn, &mut zn, None).map(|_| ()))?;
    report.line("kl_rows 512 x 151936 (with cotangent)", kl_w, kl_n)?;
    let (mut zw, mut zn) = both(&logits)?;
    let stats_w = time(&wide, 5, || wide.softmax_stats_rows(&mut zw, None).map(|_| ()))?;
    let stats_n = time(&narrow, 20, || narrow.softmax_stats_rows(&mut zn, None).map(|_| ()))?;
    report.line("softmax_stats_rows 512 x 151936", stats_w, stats_n)?;
    let (mut mw, mut mn) = (wide.zeros(rows, d).map_err(|e| e.to_string())?, narrow.zeros(rows, d).map_err(|e| e.to_string())?);
    let part_w = time(&wide, 3, || wide.head_log_partition(&hw, &ew, false, None, Some(&mut mw), Arithmetic::F64).map(|_| ()))?;
    let part_n = time(&narrow, 10, || narrow.head_log_partition(&hn, &en, false, None, Some(&mut mn), Arithmetic::F32).map(|_| ()))?;
    let part_t = time(&narrow, 10, || narrow.head_log_partition(&hn, &en, false, None, Some(&mut mn), Arithmetic::Tf32).map(|_| ()))?;
    report.line("head log partition + expected row, f64 materialized vs swept", part_w, part_n)?;
    report.line("  the same, swept on TF32", part_w, part_t)?;
    let part_b = time(&narrow, 10, || narrow.head_log_partition(&hn, &frozen_head, false, None, Some(&mut mn), Arithmetic::Bf16).map(|_| ()))?;
    report.line("  the same, swept in BF16 on the frozen head", part_w, part_b)?;
    let materialized_n = time(&narrow, 10, || {
        let mut l = narrow.zeros(rows, vocab)?;
        narrow.gemm(&mut l, 1.0, &hn, Op::N, &en, Op::T, 0.0, Arithmetic::F32)?;
        narrow.softmax_stats_rows(&mut l, None)?;
        narrow.gemm(&mut mn, 1.0, &l, Op::N, &en, Op::N, 0.0, Arithmetic::F32)
    })?;
    report.line("  f32 materialized (gemm, stats, gemm) vs f32 swept", materialized_n, part_n)?;

    // Elementwise maps, norms, gathers and the optimizer step.
    let (a, b) = (matrix(rows, ffn, 9, 2.0), matrix(rows, ffn, 10, 1.0));
    let ((aw, an), (bw, bn)) = (both(&a)?, both(&b)?);
    let (mut ow, mut on) = both(&b)?;
    report.line("axpy 512 x 3072", time(&wide, 100, || wide.axpy(&mut ow, 0.5, &aw))?, time(&narrow, 100, || narrow.axpy(&mut on, 0.5, &an))?)?;
    report.line(
        "hadamard 512 x 3072",
        time(&wide, 100, || wide.hadamard(&mut ow, &aw, &bw, false))?,
        time(&narrow, 100, || narrow.hadamard(&mut on, &an, &bn, false))?,
    )?;
    let codes = vec![PointwiseLaw::GeluTanh.code(); ffn];
    let (cdw, cdn) = (wide.upload_indices(&codes).map_err(|e| e.to_string())?, narrow.upload_indices(&codes).map_err(|e| e.to_string())?);
    let c = (2.0 / std::f64::consts::PI).sqrt();
    report.line(
        "tanh GELU values 512 x 3072",
        time(&wide, 100, || wide.law_values(&aw, &cdw, c).map(|_| ()))?,
        time(&narrow, 100, || narrow.law_values(&an, &cdn, c).map(|_| ()))?,
    )?;
    report.line(
        "tanh GELU slopes 512 x 3072",
        time(&wide, 100, || wide.law_slopes(&bw, &aw, &cdw, c).map(|_| ()))?,
        time(&narrow, 100, || narrow.law_slopes(&bn, &an, &cdn, c).map(|_| ()))?,
    )?;
    let gelu = vec![PointwiseLaw::Gelu.code(); ffn];
    let (gw, gn) = (wide.upload_indices(&gelu).map_err(|e| e.to_string())?, narrow.upload_indices(&gelu).map_err(|e| e.to_string())?);
    report.line(
        "erf GELU values 512 x 3072",
        time(&wide, 100, || wide.law_values(&aw, &gw, c).map(|_| ()))?,
        time(&narrow, 100, || narrow.law_values(&an, &gn, c).map(|_| ()))?,
    )?;
    let ((rw, rn), (dw, dn)) = (both(&x)?, both(&matrix(rows, d, 11, 1.0))?);
    report.line("rms norm 512 x 1024", time(&wide, 100, || wide.rms_norm(&rw, 1e-6).map(|_| ()))?, time(&narrow, 100, || narrow.rms_norm(&rn, 1e-6).map(|_| ()))?)?;
    report.line(
        "rms norm backward 512 x 1024",
        time(&wide, 100, || wide.rms_norm_backward(&rw, &dw, 1e-6).map(|_| ()))?,
        time(&narrow, 100, || narrow.rms_norm_backward(&rn, &dn, 1e-6).map(|_| ()))?,
    )?;
    let ids: Vec<u32> = (0..rows as u32).map(|r| (r * 7919) % vocab as u32).collect();
    let (iw, inn) = (wide.upload_indices(&ids).map_err(|e| e.to_string())?, narrow.upload_indices(&ids).map_err(|e| e.to_string())?);
    report.line(
        "gather 512 embedding rows of 1024",
        time(&wide, 100, || wide.gather_rows(&ew, &iw).map(|_| ()))?,
        time(&narrow, 100, || narrow.gather_rows(&en, &inn).map(|_| ()))?,
    )?;
    let gradient = matrix(ffn, d, 12, 1e-3);
    let ((pw, pn), (gw2, gn2)) = (both(&matrix(ffn, d, 13, 0.05))?, both(&gradient)?);
    let (mut pw, mut pn) = (pw, pn);
    let (mut m1w, mut m1n) = (wide.zeros(ffn, d).map_err(|e| e.to_string())?, narrow.zeros(ffn, d).map_err(|e| e.to_string())?);
    let (mut m2w, mut m2n) = (wide.zeros(ffn, d).map_err(|e| e.to_string())?, narrow.zeros(ffn, d).map_err(|e| e.to_string())?);
    report.line(
        "adam step on 3072 x 1024",
        time(&wide, 50, || wide.adam(&mut pw, (&mut m1w, &mut m2w), &gw2, 1e-3, (0.9, 0.999, 1e-8), 3))?,
        time(&narrow, 50, || narrow.adam(&mut pn, (&mut m1n, &mut m2n), &gn2, 1e-3, (0.9, 0.999, 1e-8), 3))?,
    )?;
    // A step of 24 small launches (maps on 512 x 1024), run op by op against one graph launch.
    for (name, device) in [("f64", &wide), ("f32", &narrow)] {
        let (mut p, q) = (up(device, &x)?, up(device, &matrix(rows, d, 14, 1.0))?);
        let step = |p: &mut Tensor| -> Result<(), GpuError> {
            for _ in 0..8 {
                device.axpy(p, 0.5, &q)?;
                device.hadamard(p, &q, &q, true)?;
                device.add_row(p, -0.25, &device.rows_of(&q, 0, 1)?)?;
            }
            Ok(())
        };
        let direct = time(device, 20, || step(&mut p))?;
        device.synchronize().map_err(|e| e.to_string())?;
        device.begin_capture().map_err(|e| e.to_string())?;
        step(&mut p).map_err(|e| e.to_string())?;
        let graph = device.end_capture().map_err(|e| e.to_string())?;
        let replay = time(device, 20, || graph.launch())?;
        report.line(&format!("24-launch step, {name}: op by op vs one graph launch"), direct, replay)?;
    }
    Ok(())
}

//! Host wrappers for the Metal kernels: operand conversion and layout,
//! parameter blocks, dispatch geometry, and the probe-time arithmetic
//! self-test.

use super::msl;
use super::{Arg, DeviceBuffer, DeviceTiming, MetalContext, join_df64, split_df64};
use crate::gpu_error::GpuError;
use ndarray::parallel::prelude::*;
use rayon::prelude::ParallelSliceMut;
use ndarray::{Array2, ArrayView2, Axis};

/// Parameter block of both GEMM kernels; layout matches MSL `GemmParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GemmParams {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    accumulate: u32,
    trans_b: u32,
    stride_a: u64,
    stride_b: u64,
    stride_c: u64,
}

fn to_u32(value: usize, what: &str) -> Result<u32, GpuError> {
    u32::try_from(value).map_err(|_| gpu_err!("Metal GEMM {what} = {value} exceeds u32"))
}

/// `value` rounded up to a multiple of `step`.
pub(crate) fn padded(value: usize, step: usize) -> usize {
    value.div_ceil(step).max(1) * step
}

/// How the right operand of a product is read from its buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RightLayout {
    /// The buffer holds `B` (`k × n`) row-major.
    AsStored,
    /// The buffer holds `Bᵀ` (`n × k`) row-major.
    Transposed,
}

/// A right operand on the device: its buffer, the logical stored shape
/// (`rows × cols`), and the row stride the buffer was written with.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RightOperand<'b> {
    pub buffer: &'b DeviceBuffer,
    pub rows: usize,
    pub cols: usize,
    pub stride: usize,
}

impl RightOperand<'_> {
    /// `(k, n)` of the product that reads this operand with `layout`.
    fn product_dims(&self, layout: RightLayout) -> (usize, usize) {
        match layout {
            RightLayout::AsStored => (self.rows, self.cols),
            RightLayout::Transposed => (self.cols, self.rows),
        }
    }
}

/// `matrix` converted to f32 into a zero-padded row-major buffer of
/// `rows_padded × cols_padded` (any input layout; round to nearest).
pub(crate) fn upload_f32_padded(
    context: &MetalContext,
    matrix: ArrayView2<'_, f64>,
    rows_padded: usize,
    cols_padded: usize,
) -> Result<DeviceBuffer, GpuError> {
    let (rows, cols) = matrix.dim();
    let mut buffer = context.buffer(rows_padded * cols_padded * 4)?;
    buffer.f32s()[..rows_padded * cols_padded]
        .par_chunks_mut(cols_padded)
        .enumerate()
        .for_each(|(row, slots)| {
            if row < rows {
                for (slot, &value) in slots[..cols].iter_mut().zip(matrix.row(row)) {
                    *slot = value as f32;
                }
                slots[cols..].fill(0.0);
            } else {
                slots.fill(0.0);
            }
        });
    Ok(buffer)
}

/// `matrix` split into df64 pairs in a row-major buffer (any input layout).
pub(crate) fn upload_df64(
    context: &MetalContext,
    matrix: ArrayView2<'_, f64>,
) -> Result<DeviceBuffer, GpuError> {
    let (rows, cols) = matrix.dim();
    let mut buffer = context.buffer(rows * cols * 8)?;
    buffer.f32_pairs()[..rows * cols]
        .par_chunks_mut(cols.max(1))
        .zip(matrix.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(slots, row)| {
            for (slot, &value) in slots.iter_mut().zip(row) {
                *slot = split_df64(value);
            }
        });
    Ok(buffer)
}

/// `A·op(B)` in f32 on the simdgroup matrix unit; `A` (`m × k`) is converted
/// on upload and `B` is resident, padded to [`msl::GEMM_F32_TILE`] in both
/// dimensions. The result is widened back to float64 exactly.
pub(crate) fn matmul_f32_resident(
    context: &MetalContext,
    a: ArrayView2<'_, f64>,
    b: RightOperand<'_>,
    layout: RightLayout,
) -> Result<(Array2<f64>, DeviceTiming), GpuError> {
    let (m, k) = a.dim();
    let (bk, n) = b.product_dims(layout);
    if k != bk {
        return Err(gpu_err!("Metal f32 GEMM: A is {m}x{k}, op(B) is {bk}x{n}"));
    }
    let tile = msl::GEMM_F32_TILE;
    let (mp, kp, np) = match layout {
        RightLayout::AsStored => (padded(m, tile), padded(b.rows, tile), b.stride),
        RightLayout::Transposed => (padded(m, tile), b.stride, padded(b.rows, tile)),
    };
    let a_dev = upload_f32_padded(context, a, mp, kp)?;
    let c_dev = context.buffer(mp * np * 4)?;
    let params = GemmParams {
        m: to_u32(mp, "m")?,
        n: to_u32(np, "n")?,
        k: to_u32(kp, "k")?,
        lda: to_u32(kp, "lda")?,
        ldb: to_u32(b.stride, "ldb")?,
        ldc: to_u32(np, "ldc")?,
        accumulate: 0,
        trans_b: u32::from(layout == RightLayout::Transposed),
        stride_a: 0,
        stride_b: 0,
        stride_c: 0,
    };
    let timing = context.submit(|recorder| {
        recorder.dispatch(
            msl::GEMM_F32_PADDED,
            &[Arg::whole(&a_dev), Arg::whole(b.buffer), Arg::whole(&c_dev)],
            &params,
            [np / tile, mp / tile, 1],
            [msl::GEMM_F32_THREADS, 1, 1],
        )
    })?;
    Ok((read_f32(c_dev, m, n, np), timing))
}

/// `A·B` in f32, both operands converted on upload (any layout).
pub(crate) fn matmul_f32(
    context: &MetalContext,
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, DeviceTiming), GpuError> {
    let (k, n) = b.dim();
    let tile = msl::GEMM_F32_TILE;
    let (kp, np) = (padded(k, tile), padded(n, tile));
    let b_dev = upload_f32_padded(context, b, kp, np)?;
    let right = RightOperand {
        buffer: &b_dev,
        rows: k,
        cols: n,
        stride: np,
    };
    matmul_f32_resident(context, a, right, RightLayout::AsStored)
}

/// The `m × n` corner of a padded f32 result, widened to float64.
fn read_f32(mut c_dev: DeviceBuffer, m: usize, n: usize, stride: usize) -> Array2<f64> {
    let padded = c_dev.f32s();
    let mut values = Array2::<f64>::zeros((m, n));
    values
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(row, mut out)| {
            for (slot, &value) in out.iter_mut().zip(&padded[row * stride..row * stride + n]) {
                *slot = f64::from(value);
            }
        });
    values
}

/// `A·op(B)` in df64 on the device against a resident df64 `B`; `A` is split
/// into pairs on upload and the result pairs are joined back to float64.
pub(crate) fn matmul_df64_resident(
    context: &MetalContext,
    a: ArrayView2<'_, f64>,
    b: RightOperand<'_>,
    layout: RightLayout,
) -> Result<(Array2<f64>, DeviceTiming), GpuError> {
    let (m, k) = a.dim();
    let (bk, n) = b.product_dims(layout);
    if k != bk {
        return Err(gpu_err!("Metal df64 GEMM: A is {m}x{k}, op(B) is {bk}x{n}"));
    }
    let a_dev = upload_df64(context, a)?;
    let mut c_dev = context.buffer(m * n * 8)?;
    let params = GemmParams {
        m: to_u32(m, "m")?,
        n: to_u32(n, "n")?,
        k: to_u32(k, "k")?,
        lda: to_u32(k, "lda")?,
        ldb: to_u32(b.stride, "ldb")?,
        ldc: to_u32(n, "ldc")?,
        accumulate: 0,
        trans_b: u32::from(layout == RightLayout::Transposed),
        stride_a: 0,
        stride_b: 0,
        stride_c: 0,
    };
    let tile = msl::GEMM_DF64_TILE;
    let side = msl::GEMM_DF64_THREADS_PER_SIDE;
    let timing = context.submit(|recorder| {
        recorder.dispatch(
            msl::GEMM_DF64,
            &[Arg::whole(&a_dev), Arg::whole(b.buffer), Arg::whole(&c_dev)],
            &params,
            [n.div_ceil(tile), m.div_ceil(tile), 1],
            [side, side, 1],
        )
    })?;
    let pairs = c_dev.f32_pairs();
    let mut values = Array2::<f64>::zeros((m, n));
    values
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(row, mut out)| {
            for (slot, &pair) in out.iter_mut().zip(&pairs[row * n..row * n + n]) {
                *slot = join_df64(pair);
            }
        });
    Ok((values, timing))
}

/// `A·B` in df64, both operands split on upload (any layout).
pub(crate) fn matmul_df64(
    context: &MetalContext,
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, DeviceTiming), GpuError> {
    let (k, n) = b.dim();
    let b_dev = upload_df64(context, b)?;
    let right = RightOperand {
        buffer: &b_dev,
        rows: k,
        cols: n,
        stride: n,
    };
    matmul_df64_resident(context, a, right, RightLayout::AsStored)
}

/// Pairs that stress the error-free transformations: huge/tiny mixtures
/// where any reassociation or non-IEEE rounding loses the error term, and
/// products whose low half an unfused or reduced-precision multiply drops.
fn self_test_pairs() -> Vec<[f32; 2]> {
    let mut pairs = vec![
        [1.0e8, 1.0],
        [1.0, -1.0e8],
        [16_777_216.0, 1.0],
        [1.0 + f32::EPSILON, 1.0 - f32::EPSILON / 2.0],
        [3.0, 1.0 / 3.0],
        [0.1, 0.2],
        [-1.0e-3, 7.0e4],
        [1.234_567_9e20, -9.876_543e-20],
        [f32::MAX / 4.0, 1.0],
    ];
    let mut state = 0x9e37_79b9_u32;
    for _ in 0..247 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let a = f32::from_bits(0x3f80_0000 | (state >> 9)) * 2f32.powi((state % 24) as i32 - 12);
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let b = -f32::from_bits(0x3f80_0000 | (state >> 9)) * 2f32.powi((state % 24) as i32 - 12);
        pairs.push([a, b]);
    }
    pairs
}

/// The GEMM self-test operands: entries `1 + j·2^-20` whose products need
/// every mantissa bit, so a reduced-precision matrix unit (a 10- or 16-bit
/// mantissa) misses the band by orders of magnitude.
fn self_test_gemm_operands(m: usize, k: usize, n: usize) -> (Array2<f64>, Array2<f64>) {
    let a = Array2::from_shape_fn((m, k), |(i, l)| {
        let index = i * k + l;
        (1.0 + (index % 97) as f64 * 2f64.powi(-20)) * if index % 3 == 0 { -1.0 } else { 1.0 }
    });
    let b = Array2::from_shape_fn((k, n), |(l, j)| {
        let index = l * n + j;
        1.0 + (index % 89) as f64 * 2f64.powi(-19) - 0.5 * ((index % 5) as f64)
    });
    (a, b)
}

/// Check the device against the arithmetic model the bands assume:
/// TwoSum and the fma product residual are exact (IEEE binary32 with no
/// reassociation or contraction), and an f32 and a df64 GEMM, in both
/// layouts of the right operand, land inside their derived bands.
/// `Ok(Err(reason))` is a violated model; `Err` is a fault running the test.
pub(crate) fn self_test(context: &MetalContext) -> Result<Result<(), String>, GpuError> {
    use crate::precision_bounds::{DeviceArithmetic, GemmBand};
    let pairs = self_test_pairs();
    let mut input = context.buffer(pairs.len() * 8)?;
    input.f32_pairs()[..pairs.len()].copy_from_slice(&pairs);
    let mut output = context.buffer(pairs.len() * 16)?;
    let count = u32::try_from(pairs.len()).map_err(|_| gpu_err!("self-test size"))?;
    context.submit(|recorder| {
        recorder.dispatch(
            msl::SELF_TEST_DF64,
            &[Arg::whole(&input), Arg::whole(&output)],
            &count,
            [pairs.len().div_ceil(64), 1, 1],
            [64, 1, 1],
        )
    })?;
    for (&[a, b], &[s, e, p, r]) in pairs.iter().zip(output.f32_quads().iter()) {
        let (a, b) = (f64::from(a), f64::from(b));
        if f64::from(s) + f64::from(e) != a + b || f64::from(s) != f64::from((a + b) as f32) {
            return Ok(Err(format!(
                "TwoSum({a:e}, {b:e}) returned {s:e} + {e:e}, not the exact sum"
            )));
        }
        if f64::from(p) + f64::from(r) != a * b || f64::from(p) != f64::from((a * b) as f32) {
            return Ok(Err(format!(
                "fma product residual of {a:e} * {b:e} returned {p:e} + {r:e}, not exact"
            )));
        }
    }
    let (m, k, n) = (67, 131, 73);
    let (a, b) = self_test_gemm_operands(m, k, n);
    let stored_t = b.t().to_owned();
    for arithmetic in [DeviceArithmetic::F32, DeviceArithmetic::Df64] {
        let band = GemmBand::derive(arithmetic, a.view(), b.view())
            .map_err(|refusal| gpu_err!("self-test operands refused: {refusal}"))?;
        for layout in [RightLayout::AsStored, RightLayout::Transposed] {
            let values = match (arithmetic, layout) {
                (DeviceArithmetic::F32, RightLayout::AsStored) => matmul_f32(context, a.view(), b.view())?.0,
                (DeviceArithmetic::F32, RightLayout::Transposed) => {
                    let tile = msl::GEMM_F32_TILE;
                    let stride = padded(k, tile);
                    let stored = upload_f32_padded(context, stored_t.view(), padded(n, tile), stride)?;
                    let right = RightOperand { buffer: &stored, rows: n, cols: k, stride };
                    matmul_f32_resident(context, a.view(), right, layout)?.0
                }
                (_, RightLayout::AsStored) => matmul_df64(context, a.view(), b.view())?.0,
                (_, RightLayout::Transposed) => {
                    let stored = upload_df64(context, stored_t.view())?;
                    let right = RightOperand { buffer: &stored, rows: n, cols: k, stride: k };
                    matmul_df64_resident(context, a.view(), right, layout)?.0
                }
            };
            for i in 0..m {
                for j in 0..n {
                    let exact = exact_dot(a.row(i), b.column(j));
                    let error = (values[[i, j]] - exact).abs();
                    if !(error <= band.entry(i, j)) {
                        return Ok(Err(format!(
                            "{} GEMM ({layout:?}) entry ({i},{j}) is off by {error:e}, outside its band {:e}",
                            arithmetic.as_str(),
                            band.entry(i, j)
                        )));
                    }
                }
            }
        }
    }
    Ok(Ok(()))
}

/// A dot product evaluated with compensated (Dot2) float64 arithmetic:
/// accurate to about `2^-106` relative to `Σ|aₗbₗ|`, which is exact for every
/// band comparison here.
pub(crate) fn exact_dot(a: ndarray::ArrayView1<'_, f64>, b: ndarray::ArrayView1<'_, f64>) -> f64 {
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let p = x * y;
        let pe = x.mul_add(y, -p);
        let s = sum + p;
        let bp = s - sum;
        let se = (sum - (s - bp)) + (p - bp);
        sum = s;
        compensation += pe + se;
    }
    sum + compensation
}

//! f32 matrix products on Metal Performance Shaders' `MPSMatrixMultiplication`.

use super::ops::{RightLayout, RightOperand, upload_f32_padded};
use super::{DeviceBuffer, DeviceTiming, MetalContext};
use crate::gpu_error::GpuError;
use ndarray::parallel::prelude::*;
use ndarray::{Array2, ArrayView2, Axis};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};

/// An f32 matrix view of `buffer`: `rows × cols` with row stride `stride`.
fn matrix(buffer: &DeviceBuffer, rows: usize, cols: usize, stride: usize) -> Retained<MPSMatrix> {
    // SAFETY: the descriptor describes `rows` rows of `stride` f32 values, and
    // every buffer handed here holds at least that many (the callers allocate
    // or upload exactly `rows * stride` values).
    unsafe {
        let descriptor = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            rows,
            cols,
            stride * 4,
            MPSDataType::Float32,
        );
        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &buffer.buffer, &descriptor)
    }
}

/// `A·op(B)` in f32 on MPS against an f32 `B` already on the device (row
/// stride `b.stride`); `A` (`m × k`) is converted on upload and the result is
/// widened back to float64 exactly.
pub(crate) fn matmul_f32_resident(
    context: &MetalContext,
    a: ArrayView2<'_, f64>,
    b: RightOperand<'_>,
    layout: RightLayout,
) -> Result<(Array2<f64>, DeviceTiming), GpuError> {
    let (m, k) = a.dim();
    let (bk, n) = match layout {
        RightLayout::AsStored => (b.rows, b.cols),
        RightLayout::Transposed => (b.cols, b.rows),
    };
    if k != bk {
        return Err(gpu_err!("MPS f32 GEMM: A is {m}x{k}, op(B) is {bk}x{n}"));
    }
    // A transposed view of a row-major matrix is uploaded as stored and read
    // transposed, so the conversion walks memory contiguously.
    let transpose_left = !a.is_standard_layout() && a.t().is_standard_layout();
    let (a_dev, left) = if transpose_left {
        let a_dev = upload_f32_padded(context, a.t(), k, m)?;
        let left = matrix(&a_dev, k, m, m);
        (a_dev, left)
    } else {
        let a_dev = upload_f32_padded(context, a, m, k)?;
        let left = matrix(&a_dev, m, k, k);
        (a_dev, left)
    };
    let mut c_dev = context.buffer(m * n * 4)?;
    let right = matrix(b.buffer, b.rows, b.cols, b.stride);
    let result = matrix(&c_dev, m, n, n);
    // SAFETY: the kernel is built for exactly these shapes: `left` holds A as m × k and `right`
    // holds op(B) as k × n after the requested transposes, `result` is m × n; all three buffers
    // live until `run` has waited for the device.
    let kernel = unsafe {
        MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
            MPSMatrixMultiplication::alloc(),
            &context.device,
            transpose_left,
            layout == RightLayout::Transposed,
            m,
            n,
            k,
            1.0,
            0.0,
        )
    };
    let timing = context.run(|command_buffer| {
        // SAFETY: as above; the command buffer is live and uncommitted.
        unsafe {
            kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                command_buffer,
                &left,
                &right,
                &result,
            );
        }
        Ok(())
    })?;
    // The left operand's storage is released only once the device has finished with it.
    drop(a_dev);
    let values_f32 = c_dev.f32s();
    let mut values = Array2::<f64>::zeros((m, n));
    values
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(row, mut out)| {
            for (slot, &value) in out.iter_mut().zip(&values_f32[row * n..row * n + n]) {
                *slot = f64::from(value);
            }
        });
    Ok((values, timing))
}

/// `A·B` in f32 on MPS, both operands converted on upload (any layout).
pub(crate) fn matmul_f32(
    context: &MetalContext,
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, DeviceTiming), GpuError> {
    let (k, n) = b.dim();
    if !b.is_standard_layout() && b.t().is_standard_layout() {
        let b_dev = upload_f32_padded(context, b.t(), n, k)?;
        let right = RightOperand { buffer: &b_dev, rows: n, cols: k, stride: k };
        return matmul_f32_resident(context, a, right, RightLayout::Transposed);
    }
    let b_dev = upload_f32_padded(context, b, k, n)?;
    let right = RightOperand { buffer: &b_dev, rows: k, cols: n, stride: n };
    matmul_f32_resident(context, a, right, RightLayout::AsStored)
}

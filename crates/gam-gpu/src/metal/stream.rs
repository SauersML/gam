//! The Apple GPU's command stream for device-resident tensors (`crate::tensor`, #2951).
//!
//! A [`Stream`] owns shared-storage buffers of 4-byte values (f32 tensors, u32 indices) and
//! encodes every operation into one open command buffer: compute dispatches of its own kernel
//! library, and f32 products on Metal Performance Shaders' GEMM. Buffers are committed to the
//! context's one queue in order (every few hundred dispatches, so the device starts early) and
//! waited for only when the host reads a value ([`Stream::read`], [`Stream::finish`]) or when more
//! than a few are in flight: a training step's hundreds of small operations cost a few round
//! trips, not one each, and its intermediates are freed as their work finishes. Metal's default hazard
//! tracking orders every read after the write before it, within and across command buffers of
//! the queue, and a committed command buffer retains the buffers it reads, so a tensor dropped
//! while its work is queued is freed only after that work.
//!
//! Each buffer is fresh shared memory on creation (zeroed, or written by the host before any
//! command can reference it); afterwards only the device writes it, and the host reads it only
//! after the queue has drained, so no host access overlaps a device access.

use super::{MetalContext, compile_pipelines, safe_compile_options};
use crate::gpu_error::GpuError;
use objc2::AnyThread;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};
use std::any::Any;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;

type RawBuffer = ProtocolObject<dyn MTLBuffer>;
type Commands = ProtocolObject<dyn MTLCommandBuffer>;
type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;
type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;

/// Threads per threadgroup of every dispatch (the kernels' `GROUP`).
pub(crate) const GROUP: usize = 256;

/// Dispatches encoded before the open command buffer is committed (without waiting).
const COMMIT_EVERY: usize = 64;

/// Committed command buffers allowed in flight; past it the host waits for the oldest. A dropped
/// tensor's buffer is freed once every command buffer reading it is released, so this bounds what
/// the queued work holds alive to a few command buffers' intermediates.
const IN_FLIGHT: usize = 3;

/// The most threadgroups one dispatch launches; kernels stride over the rest.
const MAX_GROUPS: usize = 1 << 16;

/// A shared-storage buffer of `len` 4-byte values.
pub(crate) struct Buffer {
    raw: Retained<RawBuffer>,
    len: usize,
}

// SAFETY: an MTLBuffer is a reference-counted handle with thread-safe retain/release. The host
// writes its storage only before any command references it and reads it only after the queue
// has drained (module note), so sharing the handle across threads cannot race an access.
unsafe impl Send for Buffer {}
// SAFETY: as for `Send`: `&Buffer` only binds the buffer to commands or reads a drained buffer.
unsafe impl Sync for Buffer {}

/// One matrix operand of a product: `batch` row-major `rows × cols` blocks stored back to back.
#[derive(Clone, Copy)]
pub(crate) struct Matrix<'a> {
    pub buffer: &'a Buffer,
    pub rows: usize,
    pub cols: usize,
    pub transposed: bool,
}

/// The open command buffer and its compute encoder, with the MPS objects its products use.
struct Open {
    commands: Retained<Commands>,
    encoder: Option<Retained<Encoder>>,
    held: Vec<Box<dyn Any>>,
    dispatches: usize,
}

/// Command buffers committed and not yet waited for, each with the objects it uses.
struct Queue {
    open: Option<Open>,
    committed: Vec<(Retained<Commands>, Vec<Box<dyn Any>>)>,
}

// SAFETY: the command buffers, encoders and MPS objects are only touched under the stream's
// mutex, one thread at a time; Metal permits encoding and committing from any thread.
unsafe impl Send for Queue {}

/// The device's queue with this stream's kernels (module note).
pub(crate) struct Stream {
    context: &'static MetalContext,
    pipelines: Vec<(&'static str, Retained<Pipeline>)>,
    queue: Mutex<Queue>,
}

fn ended(commands: &Commands) -> Result<(), GpuError> {
    let status = commands.status();
    if status == MTLCommandBufferStatus::Completed {
        return Ok(());
    }
    let detail = commands
        .error()
        .map(|error| error.localizedDescription().to_string())
        .unwrap_or_else(|| "no error object".to_string());
    Err(gpu_err!("Metal command buffer ended in status {}: {detail}", status.0))
}

impl Stream {
    /// A stream on `context`'s device and queue running `kernels` of the MSL `source`.
    pub(crate) fn new(context: &'static MetalContext, source: &str, kernels: &[&'static str]) -> Result<Self, GpuError> {
        let pipelines = autoreleasepool(|_| compile_pipelines(&context.device, source, kernels, &safe_compile_options()))?;
        Ok(Self { context, pipelines, queue: Mutex::new(Queue { open: None, committed: Vec::new() }) })
    }

    /// The device's name.
    pub(crate) fn device_name(&self) -> String {
        self.context.device.name().to_string()
    }

    /// Bytes still free of the device's recommended working set, and that working set.
    pub(crate) fn memory(&self) -> (usize, usize) {
        let total = usize::try_from(self.context.device.recommendedMaxWorkingSetSize()).unwrap_or(usize::MAX);
        (total.saturating_sub(self.context.device.currentAllocatedSize()), total)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, Queue>, GpuError> {
        self.queue.lock().map_err(|_| gpu_err!("Metal stream: a poisoned queue"))
    }

    /// `len` zeroed values (Metal zero-fills a new buffer; it refuses an empty one, so at least 4).
    pub(crate) fn alloc(&self, len: usize) -> Result<Buffer, GpuError> {
        let bytes = len.max(1) * 4;
        let raw = self
            .context
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| gpu_err!("Metal could not allocate a {bytes}-byte buffer"))?;
        Ok(Buffer { raw, len })
    }

    /// A buffer holding `values` (4-byte values: f32 or u32).
    pub(crate) fn upload<T: Copy>(&self, values: &[T]) -> Result<Buffer, GpuError> {
        const { assert!(std::mem::size_of::<T>() == 4) };
        if values.is_empty() {
            return self.alloc(0);
        }
        let bytes = std::mem::size_of_val(values);
        // SAFETY: `values` is a live slice of `bytes` bytes; Metal copies them during the call.
        let raw = unsafe {
            self.context.device.newBufferWithBytes_length_options(
                NonNull::from(values).cast::<c_void>(),
                bytes,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| gpu_err!("Metal could not allocate a {bytes}-byte buffer"))?;
        Ok(Buffer { raw, len: values.len() })
    }

    /// `buffer`'s values, once every queued command has run.
    pub(crate) fn read<T: Copy>(&self, buffer: &Buffer) -> Result<Vec<T>, GpuError> {
        const { assert!(std::mem::size_of::<T>() == 4) };
        self.finish()?;
        let ptr = buffer.raw.contents().cast::<T>();
        // SAFETY: shared storage of at least `len` 4-byte values, page aligned; the queue has
        // drained, so no command writes it while the host copies it out (module note).
        Ok(unsafe { std::slice::from_raw_parts(ptr.as_ptr(), buffer.len) }.to_vec())
    }

    /// `buffer`'s first `len` f32 values widened to f64, once every queued command has run (in
    /// parallel, straight from the shared storage).
    pub(crate) fn read_widened(&self, buffer: &Buffer, len: usize) -> Result<Vec<f64>, GpuError> {
        use rayon::prelude::*;
        if len > buffer.len {
            return Err(gpu_err!("Metal stream: {len} values read from a buffer of {}", buffer.len));
        }
        self.finish()?;
        let ptr = buffer.raw.contents().cast::<f32>();
        // SAFETY: as in `read`: shared storage of at least `len` f32 values, the queue drained.
        let values = unsafe { std::slice::from_raw_parts(ptr.as_ptr(), len) };
        Ok(values.par_iter().with_min_len(1 << 16).map(|v| f64::from(*v)).collect())
    }

    /// Commit the open command buffer and wait for every committed one.
    pub(crate) fn finish(&self) -> Result<(), GpuError> {
        let mut queue = self.lock()?;
        autoreleasepool(|_| {
            Self::commit(&mut queue);
            let mut first_error = None;
            for (commands, held) in queue.committed.drain(..) {
                commands.waitUntilCompleted();
                if let Err(error) = ended(&commands) {
                    first_error.get_or_insert(error);
                }
                drop(held);
            }
            first_error.map_or(Ok(()), Err)
        })
    }

    /// Commit the open command buffer without waiting, and release the finished ones.
    fn commit(queue: &mut Queue) {
        if let Some(open) = queue.open.take() {
            if let Some(encoder) = open.encoder {
                encoder.endEncoding();
            }
            open.commands.commit();
            queue.committed.push((open.commands, open.held));
        }
    }

    fn open(&self, queue: &mut Queue) -> Result<(), GpuError> {
        if queue.open.is_none() {
            let commands = self.context.queue.commandBuffer().ok_or_else(|| gpu_err!("Metal queue returned no command buffer"))?;
            queue.open = Some(Open { commands, encoder: None, held: Vec::new(), dispatches: 0 });
        }
        Ok(())
    }

    /// After an encoded operation: commit when enough are queued, release what has finished, and
    /// wait for the oldest while too many are in flight.
    fn encoded(queue: &mut Queue) -> Result<(), GpuError> {
        let Some(open) = queue.open.as_mut() else { return Ok(()) };
        open.dispatches += 1;
        if open.dispatches < COMMIT_EVERY {
            return Ok(());
        }
        Self::commit(queue);
        // The queue runs its command buffers in order: release the finished front, then wait.
        while let Some((commands, _)) = queue.committed.first() {
            let status = commands.status();
            let finished = status == MTLCommandBufferStatus::Completed || status == MTLCommandBufferStatus::Error;
            if !finished && queue.committed.len() <= IN_FLIGHT {
                break;
            }
            let (commands, held) = queue.committed.remove(0);
            commands.waitUntilCompleted();
            drop(held);
            ended(&commands)?;
        }
        Ok(())
    }

    fn pipeline(&self, kernel: &str) -> Result<&Pipeline, GpuError> {
        self.pipelines
            .iter()
            .find(|(name, _)| *name == kernel)
            .map(|(_, pipeline)| &**pipeline)
            .ok_or_else(|| gpu_err!("Metal tensor kernel {kernel} has no pipeline"))
    }

    /// Queue `kernel` on `work` items (`GROUP` threads per item, up to [`MAX_GROUPS`] groups; the
    /// kernel strides over the rest) with `buffers` in slots `0..` (each at an offset in values)
    /// and `params` in the next slot.
    pub(crate) fn dispatch<P: Copy>(&self, kernel: &'static str, buffers: &[(&Buffer, usize)], params: &P, groups: usize) -> Result<(), GpuError> {
        if groups == 0 {
            return Ok(());
        }
        let pipeline = self.pipeline(kernel)?;
        if GROUP > pipeline.maxTotalThreadsPerThreadgroup() {
            return Err(gpu_err!("Metal kernel {kernel}: {GROUP} threads per group exceed the pipeline's {}", pipeline.maxTotalThreadsPerThreadgroup()));
        }
        for (slot, (buffer, offset)) in buffers.iter().enumerate() {
            if *offset > buffer.len {
                return Err(gpu_err!("Metal kernel {kernel}: slot {slot} offset {offset} beyond {} values", buffer.len));
            }
        }
        let mut queue = self.lock()?;
        autoreleasepool(|_| {
            self.open(&mut queue)?;
            let open = queue.open.as_mut().ok_or_else(|| gpu_err!("Metal stream: no open command buffer"))?;
            if open.encoder.is_none() {
                open.encoder = Some(open.commands.computeCommandEncoder().ok_or_else(|| gpu_err!("Metal command buffer returned no compute encoder"))?);
            }
            let encoder = open.encoder.as_ref().ok_or_else(|| gpu_err!("Metal stream: no encoder"))?;
            encoder.setComputePipelineState(pipeline);
            for (slot, (buffer, offset)) in buffers.iter().enumerate() {
                // SAFETY: a live buffer of this device, the offset inside it (checked above); the
                // kernels bound every access by the extents in `params`.
                unsafe { encoder.setBuffer_offset_atIndex(Some(&buffer.raw), offset * 4, slot) };
            }
            // SAFETY: `params` is a live `#[repr(C)]` value whose layout is the kernel's parameter
            // struct; Metal copies its bytes during the call.
            unsafe { encoder.setBytes_length_atIndex(NonNull::from(params).cast::<c_void>(), std::mem::size_of::<P>(), buffers.len()) };
            let size = |width| MTLSize { width, height: 1, depth: 1 };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(size(groups.min(MAX_GROUPS)), size(GROUP));
            Self::encoded(&mut queue)
        })
    }

    /// Queue `c ← α op(a) op(b) + β c` in f32 on each of `batch` blocks (MPS's GEMM): `op(a)` is
    /// `m × k` and `op(b)` `k × n` per block, `c` `m × n`; every dimension is positive.
    pub(crate) fn gemm(&self, batch: usize, a: Matrix<'_>, b: Matrix<'_>, c: Matrix<'_>, (m, n, k): (usize, usize, usize), (alpha, beta): (f64, f64)) -> Result<(), GpuError> {
        if batch == 0 || m == 0 || n == 0 || k == 0 {
            return Err(gpu_err!("Metal GEMM: an empty product ({batch} blocks of {m}x{k} by {k}x{n})"));
        }
        for (what, operand) in [("left", &a), ("right", &b), ("result", &c)] {
            if operand.rows * operand.cols * batch > operand.buffer.len {
                return Err(gpu_err!("Metal GEMM: the {what} operand holds {} values, not {batch} blocks of {}x{}", operand.buffer.len, operand.rows, operand.cols));
            }
        }
        let matrix = |operand: Matrix<'_>| {
            let row_bytes = operand.cols * 4;
            // SAFETY: the descriptor describes `batch` blocks of `rows` rows of `cols` f32 values,
            // which the buffer holds (checked above).
            unsafe {
                let descriptor = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType(
                    operand.rows,
                    operand.cols,
                    batch,
                    row_bytes,
                    operand.rows * row_bytes,
                    MPSDataType::Float32,
                );
                MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), &operand.buffer.raw, &descriptor)
            }
        };
        let mut queue = self.lock()?;
        autoreleasepool(|_| {
            self.open(&mut queue)?;
            let open = queue.open.as_mut().ok_or_else(|| gpu_err!("Metal stream: no open command buffer"))?;
            if let Some(encoder) = open.encoder.take() {
                encoder.endEncoding();
            }
            let (left, right, result) = (matrix(a), matrix(b), matrix(c));
            // SAFETY: the kernel is built for these shapes: op(a) is m × k and op(b) k × n after the
            // requested transposes, the result m × n, per block; the operands live in `held` until
            // the command buffer has completed.
            let kernel = unsafe {
                MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    MPSMatrixMultiplication::alloc(),
                    &self.context.device,
                    a.transposed,
                    b.transposed,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                )
            };
            // SAFETY: as above; the command buffer is open and has no active encoder.
            unsafe { kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(&open.commands, &left, &right, &result) };
            open.held.push(Box::new((kernel, left, right, result)));
            Self::encoded(&mut queue)
        })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // The queued work references this stream's pipelines; let it finish first.
        if let Err(error) = self.finish() {
            log::warn!("[Metal] a tensor stream ended with queued work failing: {error}");
        }
    }
}

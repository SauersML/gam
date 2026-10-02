//! Apple-GPU (Metal) backend: the device probe, the compiled kernel library,
//! shared-memory buffers and command submission. macOS only; every item is
//! reached through [`crate::apple_gpu`].
//!
//! The binding is `objc2-metal`, the maintained Metal binding (the `metal`
//! crate is deprecated upstream in its favour and receives no new API).
//! Custom MSL kernels are used rather than Metal Performance Shaders: the
//! bands in [`crate::precision_bounds`] need the exact operation sequence of
//! every kernel, which MPS does not document.

pub(crate) mod msl;
pub(crate) mod ops;

use crate::apple_gpu::{MetalAbsence, MetalAvailability, MetalDeviceInfo, MetalRuntime};
use crate::gpu_error::GpuError;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::{msg_send, sel};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState, MTLCopyAllDevices,
    MTLDevice, MTLGPUFamily, MTLLibrary, MTLMathFloatingPointFunctions, MTLMathMode,
    MTLResourceOptions, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

type Device = ProtocolObject<dyn MTLDevice>;
type Queue = ProtocolObject<dyn MTLCommandQueue>;
type Pipeline = ProtocolObject<dyn MTLComputePipelineState>;
type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

/// The process-wide Metal handles of the selected device: the device, one
/// command queue, and a compute pipeline per kernel in [`msl::KERNELS`].
pub(crate) struct MetalContext {
    device: Retained<Device>,
    queue: Retained<Queue>,
    pipelines: Vec<(&'static str, Retained<Pipeline>)>,
}

impl std::fmt::Debug for MetalContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalContext")
            .field("device", &self.device.name().to_string())
            .field(
                "pipelines",
                &self.pipelines.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// Device time of one submission, from the command buffer's GPU clock.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DeviceTiming {
    /// Seconds the GPU spent executing the submission.
    pub gpu_seconds: f64,
}

/// A shared-storage Metal buffer. On Apple silicon the storage is the same
/// physical memory the host reads, so upload and readback are plain copies
/// through [`Self::contents`].
pub(crate) struct DeviceBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    bytes: usize,
}

// SAFETY: an MTLBuffer is a reference-counted handle whose retain/release is
// thread-safe; its storage is plain shared memory. Host access goes through
// `&mut self` (`contents`), and the device only reads or writes it inside a
// submission that `MetalContext::submit` waits for, so moving or sharing the
// handle across threads cannot race a host access with a device access.
unsafe impl Send for DeviceBuffer {}
// SAFETY: as for `Send`; a shared `&DeviceBuffer` only binds the buffer to an
// encoder, which Metal permits from any thread.
unsafe impl Sync for DeviceBuffer {}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer").field("bytes", &self.bytes).finish()
    }
}

impl DeviceBuffer {
    pub(crate) fn len_bytes(&self) -> usize {
        self.bytes
    }

    /// The buffer's storage as a mutable slice of `T`.
    ///
    /// Callers only read or write between submissions: [`MetalContext::submit`]
    /// waits for the device to finish, so no device access overlaps.
    fn contents<T: Copy>(&mut self) -> &mut [T] {
        let ptr = self.buffer.contents().cast::<T>();
        let len = self.bytes / std::mem::size_of::<T>();
        // SAFETY: the buffer is shared storage of `self.bytes` bytes, page
        // aligned by Metal (so aligned for any `T` used here: f32, [f32; 2],
        // [f32; 4]); `&mut self` gives exclusive host access, and no command
        // buffer is in flight between submissions (`submit` waits).
        unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), len) }
    }

    pub(crate) fn f32s(&mut self) -> &mut [f32] {
        self.contents::<f32>()
    }

    pub(crate) fn f32_pairs(&mut self) -> &mut [[f32; 2]] {
        self.contents::<[f32; 2]>()
    }

    pub(crate) fn f32_quads(&mut self) -> &mut [[f32; 4]] {
        self.contents::<[f32; 4]>()
    }
}

/// Split a float64 into the df64 pair `hi + lo` the device reads: `hi` is the
/// nearest f32 and `lo` the nearest f32 to the exact remainder `x − hi`
/// (exact in float64, since both are float64 with nearby exponents).
#[must_use]
pub(crate) fn split_df64(x: f64) -> [f32; 2] {
    let hi = x as f32;
    let lo = (x - f64::from(hi)) as f32;
    [hi, lo]
}

/// The float64 value of a df64 pair (exact: the sum of two f32 whose
/// exponents differ by at least 24 fits in 53 bits unless `lo` is tiny, and
/// the float64 rounding of the sum is far below the df64 band).
#[must_use]
pub(crate) fn join_df64(pair: [f32; 2]) -> f64 {
    f64::from(pair[0]) + f64::from(pair[1])
}

/// Records dispatches into one compute encoder of one command buffer.
pub(crate) struct Recorder<'a> {
    context: &'a MetalContext,
    encoder: Retained<Encoder>,
}

/// A buffer argument: the buffer, a byte offset into it, and its slot.
pub(crate) struct Arg<'b> {
    pub buffer: &'b DeviceBuffer,
    pub offset_bytes: usize,
}

impl<'b> Arg<'b> {
    pub(crate) fn whole(buffer: &'b DeviceBuffer) -> Self {
        Self {
            buffer,
            offset_bytes: 0,
        }
    }
}

impl Recorder<'_> {
    /// Encode one dispatch of `kernel` over `threadgroups` groups of
    /// `threads` threads, binding `buffers` to slots `0..` and `params` to
    /// the next slot. The encoder is serial, so each dispatch sees every
    /// earlier dispatch's writes.
    pub(crate) fn dispatch<P: Copy>(
        &self,
        kernel: &'static str,
        buffers: &[Arg<'_>],
        params: &P,
        threadgroups: [usize; 3],
        threads: [usize; 3],
    ) -> Result<(), GpuError> {
        let pipeline = self.context.pipeline(kernel)?;
        let per_group = threads[0] * threads[1] * threads[2];
        if per_group > pipeline.maxTotalThreadsPerThreadgroup() {
            return Err(gpu_err!(
                "Metal kernel {kernel}: {per_group} threads per group exceed the pipeline's {}",
                pipeline.maxTotalThreadsPerThreadgroup()
            ));
        }
        self.encoder.setComputePipelineState(pipeline);
        for (slot, arg) in buffers.iter().enumerate() {
            if arg.offset_bytes > arg.buffer.len_bytes() {
                return Err(gpu_err!(
                    "Metal kernel {kernel}: slot {slot} offset {} beyond buffer of {} bytes",
                    arg.offset_bytes,
                    arg.buffer.len_bytes()
                ));
            }
            // SAFETY: the buffer is a live MTLBuffer of this device and the
            // offset is inside it (checked above); the kernel's bounds checks
            // keep every access inside the extents its params describe.
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(&arg.buffer.buffer), arg.offset_bytes, slot);
            }
        }
        // SAFETY: `params` is a live `#[repr(C)]` value of `size_of::<P>()`
        // bytes whose layout matches the MSL struct at this slot; Metal copies
        // the bytes during the call.
        unsafe {
            self.encoder.setBytes_length_atIndex(
                NonNull::from(params).cast::<c_void>(),
                std::mem::size_of::<P>(),
                buffers.len(),
            );
        }
        self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: threadgroups[0],
                height: threadgroups[1],
                depth: threadgroups[2],
            },
            MTLSize {
                width: threads[0],
                height: threads[1],
                depth: threads[2],
            },
        );
        Ok(())
    }
}

impl MetalContext {
    fn pipeline(&self, kernel: &'static str) -> Result<&Pipeline, GpuError> {
        self.pipelines
            .iter()
            .find(|(name, _)| *name == kernel)
            .map(|(_, pipeline)| &**pipeline)
            .ok_or_else(|| gpu_err!("Metal kernel {kernel} has no pipeline"))
    }

    /// A zero-initialised shared buffer of `bytes` bytes (at least 16, since
    /// Metal refuses empty buffers).
    pub(crate) fn buffer(&self, bytes: usize) -> Result<DeviceBuffer, GpuError> {
        let len = bytes.max(16);
        let buffer = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| gpu_err!("Metal could not allocate a {len}-byte shared buffer"))?;
        Ok(DeviceBuffer { buffer, bytes })
    }

    /// Record dispatches with `record`, submit them as one command buffer and
    /// wait for the device. A command buffer that ends in any state but
    /// `Completed` is a fault carrying Metal's own error description.
    pub(crate) fn submit(
        &self,
        record: impl FnOnce(&Recorder<'_>) -> Result<(), GpuError>,
    ) -> Result<DeviceTiming, GpuError> {
        autoreleasepool(|_| {
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| gpu_err!("Metal queue returned no command buffer"))?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| gpu_err!("Metal command buffer returned no compute encoder"))?;
            let recorder = Recorder {
                context: self,
                encoder,
            };
            let recorded = record(&recorder);
            recorder.encoder.endEncoding();
            recorded?;
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            let status = command_buffer.status();
            if status != MTLCommandBufferStatus::Completed {
                let detail = command_buffer
                    .error()
                    .map(|error| error.localizedDescription().to_string())
                    .unwrap_or_else(|| "no error object".to_string());
                return Err(gpu_err!(
                    "Metal command buffer ended in status {} : {detail}",
                    status.0
                ));
            }
            let seconds = command_buffer.GPUEndTime() - command_buffer.GPUStartTime();
            Ok(DeviceTiming {
                gpu_seconds: seconds.max(0.0),
            })
        })
    }
}

fn describe(ordinal: usize, device: &Device) -> MetalDeviceInfo {
    let apple_family = (1..=10u32).rev().find(|family| {
        device.supportsFamily(MTLGPUFamily(isize::try_from(1000 + *family).unwrap_or(0)))
    });
    MetalDeviceInfo {
        ordinal,
        name: device.name().to_string(),
        registry_id: device.registryID(),
        apple_family,
        low_power: device.isLowPower(),
        removable: device.isRemovable(),
        unified_memory: device.hasUnifiedMemory(),
        recommended_max_working_set_bytes: device.recommendedMaxWorkingSetSize(),
        max_buffer_bytes: device.maxBufferLength(),
        max_threadgroup_memory_bytes: device.maxThreadgroupMemoryLength(),
    }
}

/// Compile options for the gam library: IEEE-safe math and precise math
/// functions. `mathMode` exists from macOS 15; before it the same contract is
/// `fastMathEnabled = NO`, sent by selector so the deprecated method is not
/// named in Rust.
fn safe_compile_options() -> Retained<MTLCompileOptions> {
    let options = MTLCompileOptions::new();
    if options.respondsToSelector(sel!(setMathMode:)) {
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
    } else {
        disable_fast_math(&options);
    }
    options
}

fn disable_fast_math(options: &MTLCompileOptions) {
    // SAFETY: `setFastMathEnabled:` takes one BOOL and returns void on every
    // macOS that lacks `setMathMode:` (it predates it).
    unsafe { msg_send![options, setFastMathEnabled: false] }
}

fn build_context(device: Retained<Device>) -> Result<MetalContext, GpuError> {
    let queue = device
        .newCommandQueue()
        .ok_or_else(|| gpu_err!("Metal device returned no command queue"))?;
    let options = safe_compile_options();
    let source = msl::SOURCE;
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .map_err(|error| {
            gpu_err!(
                "Metal kernel library failed to compile: {}",
                error.localizedDescription()
            )
        })?;
    let mut pipelines = Vec::with_capacity(msl::KERNELS.len());
    for &kernel in msl::KERNELS {
        let function = library
            .newFunctionWithName(&NSString::from_str(kernel))
            .ok_or_else(|| gpu_err!("Metal library has no function {kernel}"))?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| {
                gpu_err!(
                    "Metal pipeline for {kernel} failed: {}",
                    error.localizedDescription()
                )
            })?;
        pipelines.push((kernel, pipeline));
    }
    Ok(MetalContext {
        device,
        queue,
        pipelines,
    })
}

/// Enumerate, describe, rank and select a Metal device, build its context,
/// and run the arithmetic self-test.
pub(crate) fn probe() -> Result<MetalAvailability, GpuError> {
    autoreleasepool(|_| {
        let all = MTLCopyAllDevices();
        let mut candidates = Vec::with_capacity(all.count());
        for ordinal in 0..all.count() {
            let device = all.objectAtIndex(ordinal);
            candidates.push((describe(ordinal, &device), device));
        }
        if candidates.is_empty() {
            return Ok(MetalAvailability::Absent(MetalAbsence::NoDevice {
                reason: "Metal reports no GPU device".to_string(),
            }));
        }
        let devices: Vec<MetalDeviceInfo> =
            candidates.iter().map(|(info, _)| info.clone()).collect();
        let mut usable: Vec<(MetalDeviceInfo, Retained<Device>)> = candidates
            .into_iter()
            .filter(|(info, _)| info.supports_gam_kernels())
            .collect();
        usable.sort_by(|a, b| b.0.score().total_cmp(&a.0.score()));
        let Some((selected, device)) = usable.into_iter().next() else {
            return Ok(MetalAvailability::Absent(MetalAbsence::NoDevice {
                reason: format!(
                    "no Metal device of Apple GPU family {} or newer (found: {})",
                    MetalDeviceInfo::MIN_APPLE_FAMILY,
                    devices
                        .iter()
                        .map(|d| format!("{} family {:?}", d.name, d.apple_family))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }));
        };
        let context = build_context(device)?;
        if let Err(reason) = ops::self_test(&context)? {
            return Ok(MetalAvailability::Absent(MetalAbsence::ArithmeticModelViolated {
                reason: format!("{}: {reason}", selected.name),
            }));
        }
        Ok(MetalAvailability::Available(MetalRuntime {
            device: selected,
            devices,
            context,
        }))
    })
}

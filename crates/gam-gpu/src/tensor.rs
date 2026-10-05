//! Device-resident float64 tensors and the kernels a program executor needs (#2951).
//!
//! A [`Device`] runs dense row-major tensors (`rows × cols`, `f64`) through a small, closed set
//! of operations: products (BLAS GEMM, also strided-batched over equal row blocks), elementwise
//! maps, per-row reductions (a norm, a softmax, a KL against a target row, a sampled-label
//! cotangent) and gathers. It is vendor-neutral: callers see [`Device`], [`Tensor`] and
//! [`Indices`] only, never a driver type, so another backend (ROCm/HIP, whose kernel dialect the
//! CUDA source below already is) slots in behind the same calls.
//!
//! Three backends exist. [`Device::host`] runs every operation on the CPU in float64 with plain
//! loops; it is the reference the device is tested against and runs everywhere. [`Device::accelerator`]
//! takes a CUDA device under `gam_gpu`'s policy: products go to cuBLAS (DGEMM, which the A100
//! and H100 run on their FP64 tensor cores; SGEMM, optionally TF32, for proposals), the rest to
//! NVRTC-compiled kernels with fused multiply-add contraction off, so each kernel rounds exactly
//! as its host twin's expression does apart from the summation order of its reductions.
//!
//! On those two every float64 operation is IEEE float64 throughout; [`Arithmetic`] lowers only a
//! product, and only on request (a proposal that a float64 computation then decides).
//!
//! A CUDA device also holds f32 ([`Storage::F32`], [`Device::with_storage`]: the same stream and
//! handles, its tensors f32), for fitting and screening, where the device's float64 units (1/64 of
//! its f32 rate on the L40) would bound everything. Its products run on cuBLAS on the f32 operands
//! as they are (`cublasGemmEx`: f32 or the TF32 tensor cores, by [`Arithmetic`]), and every other
//! operation is its kernel's f32 twin (`tensor_f32.cu`): maps in float, and row reductions whose
//! results leave the row (a log partition, a KL, an entropy) summed in double. An operation runs in
//! its operands' storage, which must agree; no float64 operation changes. Acceptance, certification
//! and reported numbers stay in float64.
//!
//! The third, [`Device::single_precision`] on macOS, is the Apple GPU, which has no float64: its
//! tensors hold f32, every operation runs in f32 (products on Metal Performance Shaders' GEMM,
//! the rest on MSL kernels compiled with the safe math mode and contraction off, `exp`, `log` and
//! `tanh` precise, `erfc` the rational approximation of relative error below `1.2·10⁻⁷`), and a
//! product asked for in float64 is refused ([`GpuError::NoDeviceKernel`]), so float64 work stays
//! on the host. [`Device::float64`] tells the two kinds apart. Its operations queue on one Metal
//! command stream and run when the host reads a result (`crate::metal::stream`).

use crate::gpu_error::GpuError;
use crate::GpuPolicy;
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis, linalg::general_mat_mul};
use rayon::prelude::*;
use std::sync::Arc;

/// How a product reads an operand.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Op {
    /// As stored.
    N,
    /// Transposed.
    T,
}

/// The arithmetic of a product.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Arithmetic {
    /// IEEE float64 values and accumulation.
    F64,
    /// Operands rounded to f32, f32 accumulation.
    F32,
    /// Operands rounded to TF32 (10-bit mantissa), f32 accumulation (the tensor cores' fast path).
    Tf32,
    /// Operands rounded to bfloat16 (7-bit mantissa), f32 accumulation: the tensor cores' fastest
    /// path (twice TF32's rate on the L40). In f32 storage an f32 operand is rounded per call and a
    /// bfloat16 copy ([`Device::bf16_copy`], a frozen operator's) is read as it is; the host rounds
    /// its operands alike, while CUDA float64 storage and the Apple GPU run it as `F32`, more
    /// precise than asked.
    Bf16,
}

/// How a tensor holds its values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Storage {
    /// IEEE float64 (the host always; CUDA by default).
    F64,
    /// IEEE float32 (the Apple GPU always; CUDA on request, for fitting: [`Device::with_storage`]).
    F32,
    /// bfloat16, CUDA only: a frozen operand's copy ([`Device::bf16_copy`]) that products in
    /// [`Arithmetic::Bf16`] read without rounding it again, and a posterior's Adam moments and
    /// sample ([`Device::posterior_adam`], [`Device::reparameterize`]). A bfloat16 device
    /// ([`Device::with_storage`]) makes, copies, converts, uploads and downloads them; no other
    /// operation takes one.
    Bf16,
}

impl Arithmetic {
    /// The unit roundoff of the operands' rounding.
    #[must_use]
    pub fn unit_roundoff(self) -> f64 {
        match self {
            Self::F64 => f64::EPSILON / 2.0,
            Self::F32 => f64::from(f32::EPSILON) / 2.0,
            Self::Tf32 => 2f64.powi(-11),
            Self::Bf16 => 2f64.powi(-8),
        }
    }
}

/// Reduction statistics for a GPU proposal, never an acceptance certificate.
/// `magnitude` and `spread` expose the conditional KL roundoff model's terms.
#[derive(Clone, Copy, Debug)]
pub struct KlProposalRow {
    pub value: f64,
    pub magnitude: f64,
    pub spread: f64,
    pub max_log_difference: f64,
    pub teacher_max: f64,
    pub teacher_log_sum: f64,
    pub explained_max: f64,
    pub explained_log_sum: f64,
}

/// Fixed-input checked CUDA interval; never substitutes an empirical epsilon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CheckedInterval {
    Bounded { lower: f64, upper: f64 },
    Unresolved(CheckedIntervalReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckedIntervalReason {
    NonFiniteInput,
    InvalidDomain,
    ReductionNotEnclosed,
    UnboundedEndpoints,
    ArithmeticGuard,
}

#[derive(Clone, Copy, Debug)]
pub enum CheckedScalar { Exp, Log }

#[derive(Debug)]
pub struct CheckedIntervalCompilerInfo {
    pub nvrtc_major: i32,
    pub nvrtc_minor: i32,
    pub flags: Vec<String>,
    /// Some(false) in cudarc: the fastmath switch is not emitted.
    pub fastmath_policy: bool,
}

/// Numeric buffers allocated by the API: GPU triple, downloaded host triple,
/// and returned enum array. Excludes caller tensors, allocator/CUDA context,
/// compiler-generated register/spill storage and 2KiB on-chip shared per block.
pub fn checked_interval_output_bytes(results: usize) -> Result<usize, GpuError> {
    results.checked_mul(48 + std::mem::size_of::<CheckedInterval>())
        .ok_or_else(|| shape("checked interval output size overflow".into()))
}

/// The pointwise laws a kernel evaluates, by code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PointwiseLaw {
    Relu,
    Identity,
    Zero,
    /// `t / (1 + exp(−t))`.
    Silu,
    /// `t Φ(t)`.
    Gelu,
    /// `½ t (1 + tanh(c (t + 0.044715 t³)))`, `c` given per call.
    GeluTanh,
}

impl PointwiseLaw {
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::Relu => 0,
            Self::Identity => 1,
            Self::Zero => 2,
            Self::Silu => 3,
            Self::Gelu => 4,
            Self::GeluTanh => 5,
        }
    }

    fn of(code: u32) -> Self {
        match code {
            0 => Self::Relu,
            1 => Self::Identity,
            2 => Self::Zero,
            3 => Self::Silu,
            4 => Self::Gelu,
            _ => Self::GeluTanh,
        }
    }

    /// The law at `t` (the host twin of the kernel's `law_value`).
    #[must_use]
    pub fn value(self, t: f64, c: f64) -> f64 {
        match self {
            Self::Relu => t.max(0.0),
            Self::Identity => t,
            Self::Zero => 0.0,
            Self::Silu => t / (1.0 + (-t).exp()),
            Self::Gelu => t * normal_cdf(t),
            Self::GeluTanh => {
                let inner = c * (t + 0.044715 * t * t * t);
                0.5 * t * (1.0 + inner.tanh())
            }
        }
    }

    /// The law's derivative at `t` (the host twin of the kernel's `law_slope`).
    #[must_use]
    pub fn slope(self, t: f64, c: f64) -> f64 {
        match self {
            Self::Relu => {
                if t > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Identity => 1.0,
            Self::Zero => 0.0,
            Self::Silu => {
                let sigma = 1.0 / (1.0 + (-t).exp());
                sigma * (1.0 + t * (1.0 - sigma))
            }
            Self::Gelu => normal_cdf(t) + t * ((-0.5 * t * t).exp() * FRAC_1_SQRT_2PI),
            Self::GeluTanh => {
                let inner = c * (t + 0.044715 * t * t * t);
                let th = inner.tanh();
                0.5 * (1.0 + th) + 0.5 * t * (1.0 - th * th) * c * (1.0 + 3.0 * 0.044715 * t * t)
            }
        }
    }
}

const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

fn normal_cdf(t: f64) -> f64 {
    0.5 * libm::erfc(-t * std::f64::consts::FRAC_1_SQRT_2)
}

/// Explicit budget for CUDA sparse-coder scratch doubles and indices only.
/// Resident operands, mask/bound outputs, and diagnostic counters (`56 * rows` bytes) are excluded.
/// All rows are evaluated; `max_rows` bounds concurrently resident row workspaces only.
#[derive(Clone, Copy, Debug)]
pub struct CodeRowsWorkspace {
    pub bytes: usize,
    pub max_rows: usize,
    pub cache_columns: bool,
}

/// Operation counts for one row, without changing any search limit or stopping rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CodeRowCounters {
    pub relaxations: u64,
    pub sweeps: u64,
    pub column_computations: u64,
    pub column_cache_hits: u64,
    pub rounding_flips: u64,
    pub explored_nodes: u64,
    pub sweep_limit_hits: u64,
}

/// Synchronous call time includes allocations, launch, and result downloads; not GPU kernel time.
#[derive(Debug)]
pub struct CodeRowsDiagnostics {
    pub rows: Vec<CodeRowCounters>,
    pub elapsed: std::time::Duration,
    pub concurrent_rows: usize,
    pub workspace_bytes: usize,
}

/// Exact workspace allocation plan; lengths and cache offset count scalar elements per slot.
/// This host-side planner allocates nothing and is available without a CUDA device.
#[derive(Debug)]
pub struct CodeRowsLayout { pub slots: usize, pub doubles: usize, pub indices: usize, pub cache_offset: usize, pub bytes: usize }

impl CodeRowsWorkspace {
    pub fn plan(self, rows: usize, pieces: usize, blocks: usize, nodes: usize) -> Result<CodeRowsLayout, GpuError> {
        code_rows_layout(rows, pieces, blocks, nodes, self)
    }
}

fn code_rows_layout(rows: usize, pieces: usize, blocks: usize, nodes: usize, workspace: CodeRowsWorkspace) -> Result<CodeRowsLayout, GpuError> {
    if workspace.max_rows == 0 { return Err(shape("sparse code workspace needs positive max_rows".to_string())); }
    let overflow = || shape("sparse code workspace size overflow".to_string());
    let width = blocks.checked_next_power_of_two().ok_or_else(overflow)?.max(2);
    let open = nodes.checked_add(1).ok_or_else(overflow)?;
    let node_size = blocks.checked_mul(4).and_then(|n| n.checked_add(1)).ok_or_else(overflow)?;
    let cache_offset = blocks.checked_mul(15).and_then(|n| n.checked_add(pieces)).and_then(|n| n.checked_add(width))
        .and_then(|n| open.checked_mul(node_size).and_then(|m| n.checked_add(m))).ok_or_else(overflow)?;
    let cache = if workspace.cache_columns { blocks.checked_mul(blocks).ok_or_else(overflow)? } else { 0 };
    let doubles = cache_offset.checked_add(cache).ok_or_else(overflow)?;
    let indices = blocks.checked_add(width).and_then(|n| n.checked_add(if workspace.cache_columns { blocks } else { 0 })).ok_or_else(overflow)?;
    let per_row = doubles.checked_mul(8).and_then(|n| indices.checked_mul(4).and_then(|m| n.checked_add(m))).ok_or_else(overflow)?;
    if rows == 0 { return Ok(CodeRowsLayout { slots: 0, doubles, indices, cache_offset, bytes: 0 }); }
    let slots = (workspace.bytes / per_row.max(1)).min(rows).min(workspace.max_rows);
    if slots == 0 { return Err(shape(format!("sparse code workspace needs at least {per_row} bytes for one row; budget {}", workspace.bytes))); }
    Ok(CodeRowsLayout { slots, doubles, indices, cache_offset, bytes: slots * per_row })
}

/// A dense row-major `rows × cols` tensor on its device, float64 or f32 ([`Tensor::storage`]).
pub struct Tensor {
    rows: usize,
    cols: usize,
    data: Data,
}

enum Data {
    Host(Vec<f64>),
    #[cfg(target_os = "linux")]
    Cuda(cudarc::driver::CudaSlice<f64>),
    #[cfg(target_os = "linux")]
    Cuda32(cudarc::driver::CudaSlice<f32>),
    /// bfloat16 bits.
    #[cfg(target_os = "linux")]
    CudaBf16(cudarc::driver::CudaSlice<u16>),
    #[cfg(target_os = "macos")]
    Metal(crate::metal::stream::Buffer),
}

impl Tensor {
    /// Change the row-major shape without moving values or allocating device memory.
    /// This is a reshape, not a transpose; ownership prevents stale shape aliases.
    pub fn reshape(mut self, rows: usize, cols: usize) -> Result<Self, GpuError> {
        if rows.checked_mul(cols) != Some(self.len()) {
            return Err(shape(format!("cannot reshape {:?} to ({rows}, {cols})", self.dim())));
        }
        self.rows = rows;
        self.cols = cols;
        Ok(self)
    }

    #[must_use]
    pub fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The bytes it holds (four per value in f32, eight in float64).
    #[must_use]
    pub fn bytes(&self) -> usize {
        match self.storage() {
            Storage::Bf16 => self.len() * 2,
            Storage::F32 => self.len() * 4,
            Storage::F64 => self.len() * 8,
        }
    }

    /// How it holds its values.
    #[must_use]
    pub fn storage(&self) -> Storage {
        match &self.data {
            #[cfg(target_os = "linux")]
            Data::Cuda32(_) => Storage::F32,
            #[cfg(target_os = "linux")]
            Data::CudaBf16(_) => Storage::Bf16,
            #[cfg(target_os = "macos")]
            Data::Metal(_) => Storage::F32,
            _ => Storage::F64,
        }
    }
}

/// A vector of `u32` on its device (token ids, law codes, row flags).
pub struct Indices {
    len: usize,
    data: IndexData,
}

enum IndexData {
    Host(Vec<u32>),
    #[cfg(target_os = "linux")]
    Cuda(cudarc::driver::CudaSlice<u32>),
    /// The device's copy and the host's (a gather checks its ids against the table there).
    #[cfg(target_os = "macos")]
    Metal(crate::metal::stream::Buffer, Vec<u32>),
}

/// Validated contiguous column groups, with their offsets uploaded once.
pub struct ColumnBlocks {
    offsets: Indices,
    columns: usize,
}

impl ColumnBlocks {
    pub fn len(&self) -> usize { self.offsets.len - 1 }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

impl Indices {
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A recorded sequence of device operations replayed by one launch (a CUDA graph): a fixed-shape
/// step, its per-operation launch and bookkeeping costs paid once at [`Device::end_capture`].
pub struct Graph {
    #[cfg(target_os = "linux")]
    graph: cudarc::driver::CudaGraph,
}

impl Graph {
    /// Runs the recorded operations again, queued on the device's stream like any operation.
    pub fn launch(&self) -> Result<(), GpuError> {
        #[cfg(target_os = "linux")]
        {
            use crate::gpu_error::GpuResultExt;
            self.graph.launch().gpu_ctx("tensor graph launch")
        }
        #[cfg(not(target_os = "linux"))]
        Err(GpuError::NoDeviceKernel { reason: "graphs are CUDA's".to_string() })
    }
}

/// Where tensors live and run (module note), and how the tensors it makes hold their values.
/// Cloning shares the device.
#[derive(Clone)]
pub struct Device {
    backend: Arc<Backend>,
    storage: Storage,
}

enum Backend {
    Host,
    #[cfg(target_os = "linux")]
    Cuda(cuda::Engine),
    #[cfg(target_os = "macos")]
    Metal(apple::Engine),
}

fn shape(detail: String) -> GpuError {
    GpuError::DriverCallFailed { reason: format!("tensor shape mismatch: {detail}") }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn foreign() -> GpuError {
    GpuError::DriverCallFailed { reason: "a tensor used on a device that does not hold it".to_string() }
}

fn host(t: &Tensor) -> Result<&[f64], GpuError> {
    match &t.data {
        Data::Host(v) => Ok(v),
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        _ => Err(foreign()),
    }
}

fn host_mut(t: &mut Tensor) -> Result<&mut [f64], GpuError> {
    match &mut t.data {
        Data::Host(v) => Ok(v),
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        _ => Err(foreign()),
    }
}

fn host_indices(i: &Indices) -> Result<&[u32], GpuError> {
    match &i.data {
        IndexData::Host(v) => Ok(v),
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        _ => Err(foreign()),
    }
}

/// Refuses f32 operands where an operation is float64 only (a certificate, an acceptance statistic,
/// the CUDA sparse coder's search).
#[cfg(target_os = "linux")]
fn float64_only(what: &str, tensors: &[&Tensor]) -> Result<(), GpuError> {
    if tensors.iter().any(|t| t.storage() != Storage::F64) {
        return Err(GpuError::NoDeviceKernel { reason: format!("{what} run in float64 only") });
    }
    Ok(())
}

fn same(a: &Tensor, b: &Tensor, what: &str) -> Result<(), GpuError> {
    if a.dim() != b.dim() {
        return Err(shape(format!("{what}: {:?} against {:?}", a.dim(), b.dim())));
    }
    Ok(())
}

/// The bfloat16 nearest `x` (ties to even), as its 16 bits; a NaN stays a quiet NaN.
fn bf16_bits(x: f32) -> u32 {
    let bits = x.to_bits();
    if x.is_nan() { (bits >> 16) | 0x40 } else { (bits + 0x7fff + ((bits >> 16) & 1)) >> 16 }
}

/// Rounds `x` to `arithmetic`'s operand precision.
fn round_operand(x: f64, arithmetic: Arithmetic) -> f64 {
    match arithmetic {
        Arithmetic::F64 => x,
        Arithmetic::F32 => f64::from(x as f32),
        Arithmetic::Tf32 => {
            // Round the f32 value's 23-bit mantissa to 10 bits, to nearest even.
            let bits = (x as f32).to_bits();
            let rounded = (bits + 0x0000_0fff + ((bits >> 13) & 1)) & 0xffff_e000;
            f64::from(f32::from_bits(rounded))
        }
        Arithmetic::Bf16 => f64::from(f32::from_bits(bf16_bits(x as f32) << 16)),
    }
}

impl Device {
    /// The CPU reference backend.
    #[must_use]
    pub fn host() -> Self {
        Self { backend: Arc::new(Backend::Host), storage: Storage::F64 }
    }

    /// Every CUDA device `policy` admits, the selected one first (data-parallel work runs one
    /// replica per device); empty under `off` or when none exists.
    pub fn accelerators(policy: GpuPolicy) -> Result<Vec<Self>, GpuError> {
        #[cfg(target_os = "linux")]
        {
            if policy == GpuPolicy::Off {
                return Ok(Vec::new());
            }
            let Some(runtime) = crate::device_runtime::GpuRuntime::resolve(policy)? else { return Ok(Vec::new()) };
            return runtime
                .devices
                .iter()
                .map(|device| Ok(Self { backend: Arc::new(Backend::Cuda(cuda::Engine::new(device.ordinal, device.name.clone())?)), storage: Storage::F64 }))
                .collect();
        }
        #[cfg(not(target_os = "linux"))]
        Ok(Self::accelerator(policy)?.into_iter().collect())
    }

    /// The accelerator `policy` selects for device-resident float64 execution: a CUDA device
    /// (`auto` when one resolved, `required` or an error), `None` under `off` or when none exists.
    pub fn accelerator(policy: GpuPolicy) -> Result<Option<Self>, GpuError> {
        if policy == GpuPolicy::Off {
            return Ok(None);
        }
        #[cfg(target_os = "linux")]
        {
            let Some(runtime) = crate::device_runtime::GpuRuntime::resolve(policy)? else { return Ok(None) };
            let device = runtime.selected_device();
            return match cuda::Engine::new(device.ordinal, device.name.clone()) {
                Ok(engine) => Ok(Some(Self { backend: Arc::new(Backend::Cuda(engine)), storage: Storage::F64 })),
                Err(error) if policy == GpuPolicy::Required => Err(error),
                Err(error) => {
                    log::warn!("[tensor] CUDA device present but unusable for tensors: {error}");
                    Ok(None)
                }
            };
        }
        #[cfg(not(target_os = "linux"))]
        {
            if policy == GpuPolicy::Required {
                return Err(GpuError::RequiredDeviceUnavailable {
                    reason: "gpu=required: device-resident float64 tensors need a CUDA device".to_string(),
                });
            }
            Ok(None)
        }
    }

    /// The device `policy` selects for single-precision work (fitting, proposals, training
    /// products): the CUDA accelerator in f32 storage when there is one, else on macOS the Apple
    /// GPU (module note: f32 only), `None` under `off` or when neither exists.
    pub fn single_precision(policy: GpuPolicy) -> Result<Option<Self>, GpuError> {
        #[cfg(target_os = "macos")]
        {
            let Some(runtime) = crate::apple_gpu::MetalRuntime::resolve(policy)? else { return Ok(None) };
            Ok(Some(Self { backend: apple::Engine::shared(runtime)?, storage: Storage::F32 }))
        }
        #[cfg(not(target_os = "macos"))]
        {
            Self::accelerator(policy)?.map(|d| d.with_storage(Storage::F32)).transpose()
        }
    }

    /// Whether the tensors it makes hold float64, every operation on them IEEE float64 (the host,
    /// and CUDA in float64 storage); in f32 storage (the Apple GPU always, CUDA on request) every
    /// operation runs in f32 and a float64 product is refused.
    #[must_use]
    pub fn float64(&self) -> bool {
        self.storage == Storage::F64
    }

    /// How the tensors it makes hold their values.
    #[must_use]
    pub fn storage(&self) -> Storage {
        self.storage
    }

    /// This device (the same stream, handles and kernels) making its tensors in `storage`: CUDA
    /// holds any (bfloat16 for storage alone, [`Storage::Bf16`]), the host only float64 and the
    /// Apple GPU only f32 (the other is [`GpuError::NoDeviceKernel`]). An operation runs in its
    /// operands' storage, which must agree; [`Device::convert`] moves a tensor between them.
    pub fn with_storage(&self, storage: Storage) -> Result<Self, GpuError> {
        let native = match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(_) => storage,
            #[cfg(target_os = "macos")]
            Backend::Metal(_) => Storage::F32,
            Backend::Host => Storage::F64,
        };
        if native != storage {
            return Err(GpuError::NoDeviceKernel { reason: format!("{} holds no {storage:?} tensors", self.name()) });
        }
        Ok(Self { backend: Arc::clone(&self.backend), storage })
    }

    /// A copy of `t` (a tensor of this device's backend, in either storage) in this device's
    /// storage, rounded to nearest when narrowing; on the device, without a host transfer.
    pub fn convert(&self, t: &Tensor) -> Result<Tensor, GpuError> {
        if t.storage() == self.storage {
            return self.copy(t);
        }
        match (&*self.backend, &t.data) {
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda(_) | Data::Cuda32(_) | Data::CudaBf16(_)) => engine.convert_to(t, self.storage),
            _ => Err(foreign()),
        }
    }

    /// A bfloat16 copy of `t` (CUDA, either storage; rounded to nearest, ties to even), the form a
    /// frozen operand takes once so products in [`Arithmetic::Bf16`] read it without rounding it
    /// per call. Only products take it; [`Device::download`] widens it back.
    pub fn bf16_copy(&self, t: &Tensor) -> Result<Tensor, GpuError> {
        match (&*self.backend, &t.data) {
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda(_)) => engine.bf16_copy(&engine.convert(t)?),
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda32(_)) => engine.bf16_copy(t),
            _ => Err(GpuError::NoDeviceKernel { reason: format!("{} holds no bfloat16 tensors", self.name()) }),
        }
    }

    /// Whether this is the CPU reference backend.
    #[must_use]
    pub fn is_host(&self) -> bool {
        matches!(*self.backend, Backend::Host)
    }

    /// A name for reports.
    #[must_use]
    pub fn name(&self) -> String {
        match &*self.backend {
            Backend::Host => "host float64".to_string(),
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if self.storage == Storage::F32 => format!("{} (f32)", engine.name),
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.name.clone(),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.name.clone(),
        }
    }

    /// Free and total device memory in bytes (`None` on the host).
    pub fn memory(&self) -> Result<Option<(usize, usize)>, GpuError> {
        match &*self.backend {
            Backend::Host => Ok(None),
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.memory().map(Some),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => Ok(Some(engine.stream.memory())),
        }
    }

    /// Records this device's operations from here to [`Device::end_capture`] into a [`Graph`]
    /// instead of running them (CUDA stream capture). Only work that moves nothing between host
    /// and device can be recorded (an upload, a download, or an operation returning per-row
    /// values is refused meanwhile, recording nothing), on tensors that exist before the capture
    /// begins and outlive it: a tensor made during it is the graph's own temporary and must be
    /// dropped before the capture ends. The capturing thread must be the context's only user
    /// meanwhile. CUDA only.
    pub fn begin_capture(&self) -> Result<(), GpuError> {
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.begin_capture(),
            _ => Err(GpuError::NoDeviceKernel { reason: format!("{} records no graphs", self.name()) }),
        }
    }

    /// Ends the capture [`Device::begin_capture`] began: the recorded operations as one graph,
    /// none of them run yet.
    pub fn end_capture(&self) -> Result<Graph, GpuError> {
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => Ok(Graph { graph: engine.end_capture()? }),
            _ => Err(GpuError::NoDeviceKernel { reason: format!("{} records no graphs", self.name()) }),
        }
    }

    /// Waits for every queued operation.
    pub fn synchronize(&self) -> Result<(), GpuError> {
        match &*self.backend {
            Backend::Host => Ok(()),
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.synchronize(),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.stream.finish(),
        }
    }

    pub fn upload(&self, values: ArrayView2<'_, f64>) -> Result<Tensor, GpuError> {
        let (rows, cols) = values.dim();
        let flat: Vec<f64> = match values.as_slice() {
            Some(s) => s.to_vec(),
            None => values.iter().copied().collect(),
        };
        self.upload_vec(rows, cols, flat)
    }

    /// A `rows × cols` tensor from row-major `values`.
    pub fn upload_vec(&self, rows: usize, cols: usize, values: Vec<f64>) -> Result<Tensor, GpuError> {
        if values.len() != rows * cols {
            return Err(shape(format!("{} values for {rows}x{cols}", values.len())));
        }
        let data = match &*self.backend {
            Backend::Host => Data::Host(values),
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if self.storage == Storage::F32 => Data::Cuda32(engine.upload(&values.iter().map(|v| *v as f32).collect::<Vec<f32>>())?),
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if self.storage == Storage::Bf16 => {
                Data::CudaBf16(engine.upload(&values.iter().map(|v| bf16_bits(*v as f32) as u16).collect::<Vec<u16>>())?)
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => Data::Cuda(engine.upload(&values)?),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => Data::Metal(engine.stream.upload(&values.iter().map(|v| *v as f32).collect::<Vec<f32>>())?),
        };
        Ok(Tensor { rows, cols, data })
    }

    pub fn upload_indices(&self, values: &[u32]) -> Result<Indices, GpuError> {
        let data = match &*self.backend {
            Backend::Host => IndexData::Host(values.to_vec()),
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => IndexData::Cuda(engine.upload(values)?),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => IndexData::Metal(engine.stream.upload(values)?, values.to_vec()),
        };
        Ok(Indices { len: values.len(), data })
    }

    pub fn download(&self, t: &Tensor) -> Result<Array2<f64>, GpuError> {
        let flat = match &t.data {
            Data::Host(v) => v.clone(),
            #[cfg(target_os = "linux")]
            Data::Cuda(slice) => match &*self.backend {
                Backend::Cuda(engine) => engine.download(slice)?,
                _ => return Err(foreign()),
            },
            #[cfg(target_os = "linux")]
            Data::Cuda32(slice) => match &*self.backend {
                Backend::Cuda(engine) => engine.download(slice)?.into_iter().map(f64::from).collect(),
                _ => return Err(foreign()),
            },
            #[cfg(target_os = "linux")]
            Data::CudaBf16(slice) => match &*self.backend {
                Backend::Cuda(engine) => engine.download(slice)?.into_iter().map(|h| f64::from(f32::from_bits(u32::from(h) << 16))).collect(),
                _ => return Err(foreign()),
            },
            #[cfg(target_os = "macos")]
            Data::Metal(buffer) => match &*self.backend {
                Backend::Metal(engine) => engine.stream.read_widened(buffer, t.len())?,
                _ => return Err(foreign()),
            },
        };
        Array2::from_shape_vec((t.rows, t.cols), flat).map_err(|e| shape(e.to_string()))
    }

    /// Enclose the scaled norm of supplied binary64 values under IEEE round-to-nearest,
    /// gradual underflow. `scale` is [reported value, lower, upper]. No neural-forward
    /// error is covered. Exact zero / zero is zero; nonzero / a scale containing zero
    /// is unresolved (an error), with no substitute denominator.
    pub fn row_l2_enclosure(values: &[f64], scale: [f64; 3]) -> Result<[f64; 3], GpuError> {
        if scale.iter().any(|v| !v.is_finite() || *v < 0.0) || scale[1] > scale[0] || scale[0] > scale[2] || values.iter().any(|v| !v.is_finite()) {
            return Err(shape("invalid finite row norm values or scale enclosure".into()));
        }
        let maximum = values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if maximum == 0.0 { return Ok([0.0; 3]); }
        if scale[1] == 0.0 { return Err(shape("nonzero row norm over native scale containing zero: unresolved".into())); }
        let (mut sum, mut lo, mut hi) = (0.0, 0.0, 0.0);
        for value in values {
            let value = value.abs();
            if value == 0.0 { continue; }
            let q = value / maximum;
            let qlo = q.next_down().max(0.0);
            let qhi = q.next_up();
            sum += q * q;
            lo = (lo + (qlo * qlo).next_down().max(0.0)).next_down().max(0.0);
            hi = (hi + (qhi * qhi).next_up()).next_up();
        }
        let center = (maximum / scale[0]) * sum.sqrt();
        let lower = (((maximum / scale[2]).next_down().max(0.0)) * lo.sqrt().next_down().max(0.0)).next_down().max(0.0);
        let upper = ((maximum / scale[1]).next_up() * hi.sqrt().next_up()).next_up();
        if !center.is_finite() || !lower.is_finite() || !upper.is_finite() || lower > center || center > upper {
            return Err(shape("unbounded row norm enclosure: unresolved".into()));
        }
        Ok([center, lower, upper])
    }

    /// Max-rescaled, outward-enclosed row norms. Only three scalars per row leave CUDA.
    /// The CPU and CUDA reductions use the same column order; interval endpoints may
    /// differ because CUDA uses directed arithmetic and CPU steps outward after RN.
    pub fn scaled_row_l2_enclosed(&self, values: &Tensor, columns: std::ops::Range<usize>, scale: [f64; 3]) -> Result<Vec<[f64; 3]>, GpuError> {
        if columns.start > columns.end || columns.end > values.cols || scale.iter().any(|v| !v.is_finite() || *v < 0.0) || scale[1] > scale[0] || scale[0] > scale[2] {
            return Err(shape("invalid row norm columns or scale enclosure".into()));
        }
        let norms: Vec<[f64; 3]> = match (&*self.backend, &values.data) {
            (Backend::Host, Data::Host(data)) => (0..values.rows).map(|row| Self::row_l2_enclosure(&data[row * values.cols + columns.start..row * values.cols + columns.end], scale)).collect::<Result<_,_>>()?,
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda(_)) => engine.scaled_row_l2_enclosed(values, columns, scale)?,
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            _ => return Err(shape("row norms require matching host or CUDA binary64 tensors".into())),
        };
        if norms.iter().any(|v| v.iter().any(|x| !x.is_finite() || *x < 0.0) || v[1] > v[0] || v[0] > v[2]) {
            return Err(shape("unbounded row norm enclosure: unresolved".into()));
        }
        Ok(norms)
    }

    /// Robust scaled Euclidean row norms; see `scaled_row_l2_enclosed` for bounds.
    pub fn scaled_row_l2(&self, values: &Tensor, columns: std::ops::Range<usize>, scale: f64) -> Result<Vec<f64>, GpuError> {
        Ok(self.scaled_row_l2_enclosed(values, columns, [scale;3])?.into_iter().map(|v|v[0]).collect())
    }


    pub fn zeros(&self, rows: usize, cols: usize) -> Result<Tensor, GpuError> {
        let data = match &*self.backend {
            Backend::Host => Data::Host(vec![0.0; rows * cols]),
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if self.storage == Storage::F32 => Data::Cuda32(engine.zeros32(rows * cols)?),
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if self.storage == Storage::Bf16 => Data::CudaBf16(engine.zeros16(rows * cols)?),
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => Data::Cuda(engine.zeros(rows * cols)?),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => Data::Metal(engine.stream.alloc(rows * cols)?),
        };
        Ok(Tensor { rows, cols, data })
    }

    pub fn copy(&self, t: &Tensor) -> Result<Tensor, GpuError> {
        let data = match (&*self.backend, &t.data) {
            (Backend::Host, Data::Host(v)) => Data::Host(v.clone()),
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda(slice)) => Data::Cuda(engine.copy(slice)?),
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda32(slice)) => Data::Cuda32(engine.copy(slice)?),
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::CudaBf16(slice)) => Data::CudaBf16(engine.copy(slice)?),
            #[cfg(target_os = "macos")]
            (Backend::Metal(engine), Data::Metal(buffer)) => Data::Metal(engine.copy_range(buffer, 0, t.len())?),
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            _ => return Err(foreign()),
        };
        Ok(Tensor { rows: t.rows, cols: t.cols, data })
    }

    /// `c ← α op(a) op(b) + β c` in `arithmetic`.
    pub fn gemm(&self, c: &mut Tensor, alpha: f64, a: &Tensor, ta: Op, b: &Tensor, tb: Op, beta: f64, arithmetic: Arithmetic) -> Result<(), GpuError> {
        self.gemm_batched(1, c, alpha, a, ta, b, tb, beta, arithmetic)
    }

    /// [`Device::gemm`] on each of `batch` equal row blocks of `a`, `b` and `c` at once: block `i`
    /// of `c` is `α op(a_i) op(b_i) + β c_i` (the attention of `batch` sequences, one block each).
    pub fn gemm_batched(
        &self,
        batch: usize,
        c: &mut Tensor,
        alpha: f64,
        a: &Tensor,
        ta: Op,
        b: &Tensor,
        tb: Op,
        beta: f64,
        arithmetic: Arithmetic,
    ) -> Result<(), GpuError> {
        if batch == 0 || a.rows % batch != 0 || b.rows % batch != 0 || c.rows % batch != 0 {
            return Err(shape(format!("{batch} blocks of {}, {} and {} rows", a.rows, b.rows, c.rows)));
        }
        let block = |t: &Tensor| (t.rows / batch, t.cols);
        let (ab, bb, cb) = (block(a), block(b), block(c));
        let (m, k) = match ta {
            Op::N => ab,
            Op::T => (ab.1, ab.0),
        };
        let (kb, n) = match tb {
            Op::N => bb,
            Op::T => (bb.1, bb.0),
        };
        if k != kb || cb != (m, n) {
            return Err(shape(format!("op(a) {m}x{k}, op(b) {kb}x{n}, c {}x{} (per block)", cb.0, cb.1)));
        }
        match &*self.backend {
            Backend::Host => host_gemm(batch, ab, bb, cb, (alpha, beta), (host(a)?, ta), (host(b)?, tb), host_mut(c)?, arithmetic),
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.gemm(batch, (m, n, k), (alpha, beta), (a, ta), (b, tb), c, arithmetic),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.gemm(batch, (m, n, k), (alpha, beta), (a, ta), (b, tb), c, arithmetic),
        }
    }

    /// `y ← y + α x`.
    pub fn axpy(&self, y: &mut Tensor, alpha: f64, x: &Tensor) -> Result<(), GpuError> {
        same(x, y, "axpy")?;
        match &*self.backend {
            Backend::Host => {
                for (yv, xv) in host_mut(y)?.iter_mut().zip(host(x)?) {
                    *yv += alpha * xv;
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.axpy(y, alpha, x),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.axpy(y, alpha, x),
        }
    }

    /// `out ← a ⊙ b`, or `out ← out + a ⊙ b` when `accumulate`.
    pub fn hadamard(&self, out: &mut Tensor, a: &Tensor, b: &Tensor, accumulate: bool) -> Result<(), GpuError> {
        same(a, b, "hadamard")?;
        same(a, out, "hadamard output")?;
        match &*self.backend {
            Backend::Host => {
                let (a, b) = (host(a)?, host(b)?);
                for ((o, av), bv) in host_mut(out)?.iter_mut().zip(a).zip(b) {
                    *o = if accumulate { *o + av * bv } else { av * bv };
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.hadamard(out, a, b, accumulate),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.hadamard(out, a, b, accumulate),
        }
    }

    /// `x[r, :] ← x[r, :] + α row` for every row (`row` is `1 × cols`).
    pub fn add_row(&self, x: &mut Tensor, alpha: f64, row: &Tensor) -> Result<(), GpuError> {
        if row.rows != 1 || row.cols != x.cols {
            return Err(shape(format!("a {:?} row added to {:?}", row.dim(), x.dim())));
        }
        match &*self.backend {
            Backend::Host => {
                let cols = x.cols;
                let r = host(row)?;
                for (i, v) in host_mut(x)?.iter_mut().enumerate() {
                    *v += alpha * r[i % cols];
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.add_row(x, alpha, row),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.add_row(x, alpha, row),
        }
    }

    /// `out ← x ⊙ d` per row, or `out ← out + x ⊙ d` when `accumulate` (`d` is `1 × cols`): a
    /// diagonal operator.
    pub fn scale_columns(&self, out: &mut Tensor, x: &Tensor, d: &Tensor, accumulate: bool) -> Result<(), GpuError> {
        same(x, out, "column scale")?;
        if d.rows != 1 || d.cols != x.cols {
            return Err(shape(format!("a {:?} diagonal on {:?}", d.dim(), x.dim())));
        }
        match &*self.backend {
            Backend::Host => {
                let cols = x.cols;
                let (xv, dv) = (host(x)?, host(d)?);
                for (i, o) in host_mut(out)?.iter_mut().enumerate() {
                    let term = xv[i] * dv[i % cols];
                    *o = if accumulate { *o + term } else { term };
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.scale_columns(out, x, d, accumulate),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.scale_columns(out, x, d, accumulate),
        }
    }

    /// `out[r, :] = table[ids[r], :]`.
    pub fn gather_rows(&self, table: &Tensor, ids: &Indices) -> Result<Tensor, GpuError> {
        match &*self.backend {
            Backend::Host => {
                let (t, cols) = (host(table)?, table.cols);
                let mut out = Vec::with_capacity(ids.len * cols);
                for &id in host_indices(ids)? {
                    let id = id as usize;
                    if id >= table.rows {
                        return Err(shape(format!("row {id} of a {}-row table", table.rows)));
                    }
                    out.extend_from_slice(&t[id * cols..(id + 1) * cols]);
                }
                Ok(Tensor { rows: ids.len, cols, data: Data::Host(out) })
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.gather_rows(table, ids),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.gather_rows(table, ids),
        }
    }

    /// `rows` copies of the `1 × cols` tensor `row`.
    pub fn broadcast_rows(&self, row: &Tensor, rows: usize) -> Result<Tensor, GpuError> {
        let mut out = self.zeros(rows, row.cols)?;
        self.add_row(&mut out, 1.0, row)?;
        Ok(out)
    }

    /// Each entry through its column's law (`codes`, one per column; `c` the tanh GELU's constant).
    pub fn law_values(&self, x: &Tensor, codes: &Indices, c: f64) -> Result<Tensor, GpuError> {
        if codes.len != x.cols {
            return Err(shape(format!("{} law codes for {} columns", codes.len, x.cols)));
        }
        match &*self.backend {
            Backend::Host => {
                let (xv, codes_v, cols) = (host(x)?, host_indices(codes)?, x.cols);
                let out = xv.iter().enumerate().map(|(i, t)| PointwiseLaw::of(codes_v[i % cols]).value(*t, c)).collect();
                Ok(Tensor { rows: x.rows, cols, data: Data::Host(out) })
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.laws(x, None, codes, c),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.laws(x, None, codes, c),
        }
    }

    /// `g ⊙ f'(x)` per column law: the cotangent (or tangent) through the laws.
    pub fn law_slopes(&self, g: &Tensor, x: &Tensor, codes: &Indices, c: f64) -> Result<Tensor, GpuError> {
        same(g, x, "law slopes")?;
        if codes.len != x.cols {
            return Err(shape(format!("{} law codes for {} columns", codes.len, x.cols)));
        }
        match &*self.backend {
            Backend::Host => {
                let (xv, gv, codes_v, cols) = (host(x)?, host(g)?, host_indices(codes)?, x.cols);
                let out = xv.iter().zip(gv).enumerate().map(|(i, (t, gi))| gi * PointwiseLaw::of(codes_v[i % cols]).slope(*t, c)).collect();
                Ok(Tensor { rows: x.rows, cols, data: Data::Host(out) })
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.laws(x, Some(g), codes, c),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.laws(x, Some(g), codes, c),
        }
    }

    /// `x / √(mean(x²) + ε)` per row.
    pub fn rms_norm(&self, x: &Tensor, epsilon: f64) -> Result<Tensor, GpuError> {
        self.rms(RmsMode::Value, x, None, epsilon)
    }

    /// The cotangent of [`Device::rms_norm`] at `x` given the output's `g`.
    pub fn rms_norm_backward(&self, x: &Tensor, g: &Tensor, epsilon: f64) -> Result<Tensor, GpuError> {
        same(x, g, "rms backward")?;
        self.rms(RmsMode::Backward, x, Some(g), epsilon)
    }

    /// The tangent of [`Device::rms_norm`] at `x` along `dx`.
    pub fn rms_norm_tangent(&self, x: &Tensor, dx: &Tensor, epsilon: f64) -> Result<Tensor, GpuError> {
        same(x, dx, "rms tangent")?;
        self.rms(RmsMode::Tangent, x, Some(dx), epsilon)
    }

    fn rms(&self, mode: RmsMode, x: &Tensor, g: Option<&Tensor>, epsilon: f64) -> Result<Tensor, GpuError> {
        match &*self.backend {
            Backend::Host => {
                let (xv, cols) = (host(x)?, x.cols);
                let gv = g.map(host).transpose()?;
                let n = cols as f64;
                let mut out = vec![0.0; xv.len()];
                for r in 0..x.rows {
                    let xr = &xv[r * cols..(r + 1) * cols];
                    let mean = xr.iter().map(|v| v * v).sum::<f64>() / n;
                    let scale = 1.0 / (mean + epsilon).sqrt();
                    let o = &mut out[r * cols..(r + 1) * cols];
                    match (mode, gv) {
                        (RmsMode::Value, _) => {
                            for c in 0..cols {
                                o[c] = xr[c] * scale;
                            }
                        }
                        (RmsMode::Backward, Some(gv)) => {
                            let gr = &gv[r * cols..(r + 1) * cols];
                            let inner = xr.iter().zip(gr).map(|(a, b)| a * b).sum::<f64>();
                            for c in 0..cols {
                                o[c] = scale * gr[c] - scale * scale * scale / n * xr[c] * inner;
                            }
                        }
                        (RmsMode::Tangent, Some(gv)) => {
                            let dr = &gv[r * cols..(r + 1) * cols];
                            let dm = 2.0 * xr.iter().zip(dr).map(|(a, b)| a * b).sum::<f64>() / n;
                            let ds = -0.5 * scale * scale * scale * dm;
                            for c in 0..cols {
                                o[c] = dr[c] * scale + xr[c] * ds;
                            }
                        }
                        _ => return Err(shape("an rms derivative without its direction".to_string())),
                    }
                }
                Ok(Tensor { rows: x.rows, cols, data: Data::Host(out) })
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.rms(mode, x, g, epsilon),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.rms(mode, x, g, epsilon),
        }
    }

    /// Each row's planes turned by the angles of its position: plane `i` pairs column `i` with
    /// `i + planes` (`half_split`) or `2i` with `2i + 1`, and turns by `(cos, sin)[r, i]`
    /// (`rows × planes` each), backwards when `inverse`; other columns are copied.
    pub fn rotate(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, half_split: bool, inverse: bool) -> Result<Tensor, GpuError> {
        same(cos, sin, "rotation tables")?;
        let planes = cos.cols;
        if cos.rows != x.rows || 2 * planes > x.cols {
            return Err(shape(format!("{:?} rotation tables on {:?}", cos.dim(), x.dim())));
        }
        match &*self.backend {
            Backend::Host => {
                let (xv, cv, sv, cols) = (host(x)?, host(cos)?, host(sin)?, x.cols);
                let mut out = xv.to_vec();
                for r in 0..x.rows {
                    for i in 0..planes {
                        let (a, b) = if half_split { (i, i + planes) } else { (2 * i, 2 * i + 1) };
                        let (c, s) = (cv[r * planes + i], if inverse { -sv[r * planes + i] } else { sv[r * planes + i] });
                        let (xa, xb) = (xv[r * cols + a], xv[r * cols + b]);
                        out[r * cols + a] = c * xa - s * xb;
                        out[r * cols + b] = s * xa + c * xb;
                    }
                }
                Ok(Tensor { rows: x.rows, cols, data: Data::Host(out) })
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.rotate(x, cos, sin, half_split, inverse),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.rotate(x, cos, sin, half_split, inverse),
        }
    }

    /// In place, each row of `scores` (`blocks · L × L`) to its softmax; when `causal`, row `r`
    /// reads only columns `j ≤ r mod L` and the rest become zero.
    pub fn softmax_rows(&self, scores: &mut Tensor, causal: bool) -> Result<(), GpuError> {
        let width = scores.cols;
        if width == 0 || (causal && scores.rows % width != 0) {
            return Err(shape(format!("attention scores {:?} are not square blocks", scores.dim())));
        }
        self.softmax_rows_impl(scores, causal, 0, width)
    }

    /// Softmax of a rectangular query tile against a sequence's keys, preserving its causal
    /// offset. Noncausal rows can have any batch/vocabulary shape.
    pub fn softmax_rows_offset(&self, scores: &mut Tensor, causal: bool, start: usize) -> Result<(), GpuError> {
        if scores.cols == 0 || (causal && (start > scores.cols || scores.rows > scores.cols - start)) {
            return Err(shape("causal softmax tile outside its sequence".to_string()));
        }
        let width = scores.cols;
        self.softmax_rows_impl(scores, causal, start, width)
    }

    /// Causal softmax of `blocks` query tiles stacked (`blocks · block_rows × L`), each the
    /// positions `start..start + block_rows` of its own sequence's `L` keys: row `r` reads columns
    /// `j ≤ start + r mod block_rows`, the rest become zero.
    pub fn softmax_rows_blocks(&self, scores: &mut Tensor, block_rows: usize, start: usize) -> Result<(), GpuError> {
        if block_rows == 0 || scores.rows % block_rows != 0 || start + block_rows > scores.cols {
            return Err(shape(format!("causal query blocks of {block_rows} rows from {start} in {:?} scores", scores.dim())));
        }
        self.softmax_rows_impl(scores, true, start, block_rows)
    }

    fn softmax_rows_impl(&self, scores: &mut Tensor, causal: bool, start: usize, period: usize) -> Result<(), GpuError> {
        let width = scores.cols;
        match &*self.backend {
            Backend::Host => {
                for (r, row) in host_mut(scores)?.chunks_mut(width).enumerate() {
                    let valid = if causal { start + r % period + 1 } else { width };
                    let m = row[..valid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    for v in row[..valid].iter_mut() {
                        *v = (*v - m).exp();
                    }
                    let total: f64 = row[..valid].iter().sum();
                    for v in row[..valid].iter_mut() {
                        *v /= total;
                    }
                    for v in row[valid..].iter_mut() {
                        *v = 0.0;
                    }
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.softmax_rows(scores, causal, start, period),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.softmax_rows(scores, causal, start, period),
        }
    }

    /// `α ⊙ (d − Σ_j α_j d_j)` per row: the softmax's cotangent (or tangent) map at its value `α`.
    pub fn softmax_backward(&self, alpha: &Tensor, d: &Tensor) -> Result<Tensor, GpuError> {
        same(alpha, d, "softmax backward")?;
        match &*self.backend {
            Backend::Host => {
                let (av, dv, cols) = (host(alpha)?, host(d)?, alpha.cols);
                let mut out = vec![0.0; av.len()];
                for r in 0..alpha.rows {
                    let (ar, dr) = (&av[r * cols..(r + 1) * cols], &dv[r * cols..(r + 1) * cols]);
                    let mean: f64 = ar.iter().zip(dr).map(|(a, b)| a * b).sum();
                    for c in 0..cols {
                        out[r * cols + c] = ar[c] * (dr[c] - mean);
                    }
                }
                Ok(Tensor { rows: alpha.rows, cols, data: Data::Host(out) })
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.softmax_backward(alpha, d),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.softmax_backward(alpha, d),
        }
    }

    /// Per row, `KL(softmax(target) ‖ softmax(logits))` and, in place of `logits`, its cotangent
    /// `q − p`; a row whose `scored` flag is zero has zero of both. Returns the KL per row.
    pub fn kl_rows(&self, target: &Tensor, logits: &mut Tensor, scored: Option<&Indices>) -> Result<Vec<f64>, GpuError> {
        self.kl_rows_impl(target, logits, scored, true)
    }

    /// [`Device::kl_rows`] (with `gradient`; else [`Device::kl_score_rows`]) leaving each row's KL
    /// on the device in `kl` (rows × 1, float64 storage whatever the operands' storage: per-row
    /// results are float64 in both; make it with the float64 twin), so a captured step can record
    /// it. Host and CUDA.
    pub fn kl_rows_into(&self, target: &Tensor, logits: &mut Tensor, scored: Option<&Indices>, gradient: bool, kl: &mut Tensor) -> Result<(), GpuError> {
        same(target, logits, "kl")?;
        if kl.dim() != (logits.rows, 1) || kl.storage() != Storage::F64 || scored.is_some_and(|s| s.len != logits.rows) {
            return Err(shape(format!("a float64 {:?} KL column for {:?} logits", kl.dim(), logits.dim())));
        }
        match (&*self.backend, &mut kl.data) {
            (Backend::Host, Data::Host(out)) => {
                let values = self.kl_rows_impl(target, logits, scored, gradient)?;
                out.copy_from_slice(&values);
                Ok(())
            }
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda(out)) => engine.kl_rows_into(target, logits, scored, gradient, out),
            _ => Err(GpuError::NoDeviceKernel { reason: format!("{} keeps no KL column", self.name()) }),
        }
    }

    /// Stable full-vocabulary softmax and (log partition, negative entropy) per row.
    /// Overwrites logits with probabilities; unscored rows become exactly zero.
    /// Operational floating-point statistics, not certified real-arithmetic intervals.
    pub fn softmax_stats_rows(&self, logits: &mut Tensor, scored: Option<&Indices>) -> Result<Vec<[f64; 2]>, GpuError> {
        if logits.cols == 0 || scored.is_some_and(|s| s.len != logits.rows) {
            return Err(shape("invalid softmax statistics shape".into()));
        }
        match &*self.backend {
            Backend::Host => {
                let flags = scored.map(host_indices).transpose()?;
                let cols = logits.cols;
                // Rows on the rayon pool.
                host_mut(logits)?.par_chunks_mut(cols).enumerate().map(|(r, row)| {
                    if flags.is_some_and(|s| s[r] == 0) {
                        row.fill(0.0);
                        return Ok([0.0, 0.0]);
                    }
                    if row.iter().any(|v| !v.is_finite()) { return Err(shape("nonfinite softmax logits".into())); }
                    let (maximum, sum) = host_softmax_stats(row);
                    let log_sum = sum.ln();
                    let mut entropy = 0.0;
                    for value in row {
                        let log_probability = (*value - maximum) - log_sum;
                        let probability = (*value - maximum).exp() / sum;
                        if probability > 0.0 { entropy += probability * log_probability; }
                        *value = probability;
                    }
                    let stats = [maximum + log_sum, entropy];
                    if stats.iter().any(|v| !v.is_finite()) { return Err(shape("nonfinite softmax statistics".into())); }
                    Ok(stats)
                }).collect()
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.softmax_stats_rows(logits, scored),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.softmax_stats_rows(logits, scored),
        }
    }

    /// Checked f64 KL proposal statistics. No clamp, gradient or acceptance band.
    /// CUDA returns eight numbers per row rather than full logits. Vendor exp/log
    /// accuracy tables are not guaranteed bounds; callers must independently replay
    /// CPU metrics for accepted verdicts. Nonfinite inputs/intermediates are errors.
    pub fn kl_proposal_rows(&self, target: &Tensor, logits: &Tensor) -> Result<Vec<KlProposalRow>, GpuError> {
        same(target, logits, "KL proposal")?;
        if logits.cols == 0 || u32::try_from(logits.cols).is_err() || u32::try_from(logits.rows).is_err() {
            return Err(shape("KL proposal requires nonempty columns and u32 dimensions".into()));
        }
        match &*self.backend {
            Backend::Host => {
                let mut result = Vec::with_capacity(logits.rows);
                for (p, q) in host(target)?.chunks(logits.cols).zip(host(logits)?.chunks(logits.cols)) {
                    if p.iter().chain(q).any(|x| !x.is_finite()) { return Err(shape("nonfinite KL proposal logits".into())); }
                    let (mp, sp) = host_softmax_stats(p);
                    let (mq, sq) = host_softmax_stats(q);
                    let (lp, lq) = (sp.ln(), sq.ln());
                    let mut row = KlProposalRow { value: 0.0, magnitude: 0.0, spread: 0.0, max_log_difference: 0.0, teacher_max: mp, teacher_log_sum: lp, explained_max: mq, explained_log_sum: lq };
                    for (&a, &b) in p.iter().zip(q) {
                        let (shift_p, shift_q) = (a - mp, b - mq);
                        let (log_p, log_q) = (shift_p - lp, shift_q - lq);
                        let probability = shift_p.exp() / sp;
                        if [shift_p, shift_q, log_p, log_q, probability].iter().any(|x| !x.is_finite()) { return Err(shape("nonfinite KL proposal intermediate".into())); }
                        row.max_log_difference = row.max_log_difference.max((log_p - log_q).abs());
                        if probability > 0.0 {
                            row.value += probability * (log_p - log_q);
                            row.magnitude += probability * (log_p.abs() + log_q.abs());
                            row.spread += probability * (log_p - log_q).abs();
                        }
                    }
                    if [row.value, row.magnitude, row.spread, row.max_log_difference, lp, lq].iter().any(|x| !x.is_finite()) { return Err(shape("nonfinite KL proposal reduction".into())); }
                    result.push(row);
                }
                Ok(result)
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => {
                float64_only("KL proposal statistics", &[target, logits])?;
                engine.kl_proposal_rows(target, logits)
            }
            #[cfg(target_os = "macos")]
            Backend::Metal(_) => Err(shape("KL proposal statistics require f64; Metal unsupported".into())),
        }
    }

    /// Optional analytic interval on fixed finite f64 logits, CUDA only.
    /// Explicit output-buffer budget; caller controls row batch. Input, head and
    /// network rounding are outside this claim. No acceptance/default switch.
    pub fn checked_kl_intervals(&self, teacher: &Tensor, explained: &Tensor, output_budget_bytes: usize) -> Result<Vec<CheckedInterval>, GpuError> {
        same(teacher, explained, "checked KL")?;
        if teacher.cols == 0 || u32::try_from(teacher.rows).is_err() || u32::try_from(teacher.cols).is_err() {
            return Err(shape("checked KL requires nonempty u32 dimensions".into()));
        }
        if checked_interval_output_bytes(teacher.rows)? > output_budget_bytes {
            return Err(shape("checked KL output numeric-buffer budget exceeded".into()));
        }
        match &*self.backend {
            #[cfg(target_os="linux")]
            Backend::Cuda(engine) => {
                float64_only("checked KL intervals", &[teacher, explained])?;
                engine.checked_intervals(teacher, Some(explained), None)
            }
            _ => Err(shape("checked intervals require CUDA f64 without CPU/Metal fallback".into())),
        }
    }

    /// Scalar spot checks for the same analytic CUDA code as checked KL.
    pub fn checked_scalar_intervals(&self, input: &Tensor, operation: CheckedScalar, output_budget_bytes: usize) -> Result<Vec<CheckedInterval>, GpuError> {
        let count = input.rows.checked_mul(input.cols).ok_or_else(|| shape("checked scalar size overflow".into()))?;
        if checked_interval_output_bytes(count)? > output_budget_bytes {
            return Err(shape("checked scalar output numeric-buffer budget exceeded".into()));
        }
        match &*self.backend {
            #[cfg(target_os="linux")]
            Backend::Cuda(engine) => {
                float64_only("checked scalar intervals", &[input])?;
                engine.checked_intervals(input, None, Some(operation))
            }
            _ => Err(shape(format!("checked {operation:?} intervals require CUDA f64 without CPU/Metal fallback"))),
        }
    }

    pub fn checked_interval_compiler_info(&self) -> Result<CheckedIntervalCompilerInfo, GpuError> {
        match &*self.backend {
            #[cfg(target_os="linux")]
            Backend::Cuda(_) => {
                let (nvrtc_major,nvrtc_minor,flags)=crate::device_cache::checked_interval_compiler_info()?;
                Ok(CheckedIntervalCompilerInfo {nvrtc_major,nvrtc_minor,flags,fastmath_policy:false})
            },
            _ => Err(shape("checked interval compiler info requires CUDA".into())),
        }
    }

    /// Per-row KL without calculating a cotangent. `logits` is a scratch buffer, left unchanged.
    pub fn kl_score_rows(&self, target: &Tensor, logits: &mut Tensor, scored: Option<&Indices>) -> Result<Vec<f64>, GpuError> {
        self.kl_rows_impl(target, logits, scored, false)
    }

    fn kl_rows_impl(&self, target: &Tensor, logits: &mut Tensor, scored: Option<&Indices>, gradient: bool) -> Result<Vec<f64>, GpuError> {
        same(target, logits, "kl")?;
        if let Some(s) = scored
            && s.len != logits.rows
        {
            return Err(shape(format!("{} row flags for {} rows", s.len, logits.rows)));
        }
        match &*self.backend {
            Backend::Host => {
                let (tv, cols) = (host(target)?, logits.cols);
                let flags = scored.map(host_indices).transpose()?;
                let mut kl = vec![0.0; logits.rows];
                for (r, row) in host_mut(logits)?.chunks_mut(cols).enumerate() {
                    if flags.is_some_and(|f| f[r] == 0) {
                        if gradient {
                            row.fill(0.0);
                        }
                        continue;
                    }
                    let teacher = &tv[r * cols..(r + 1) * cols];
                    let (mt, st) = host_softmax_stats(teacher);
                    let (mz, sz) = host_softmax_stats(row);
                    let (lt, lz) = (st.ln(), sz.ln());
                    let mut total = 0.0;
                    for c in 0..cols {
                        let p = (teacher[c] - mt).exp() / st;
                        // Keep the logarithms in shifted logit space: q can underflow even
                        // though log(q), its KL contribution, and the derivative are finite.
                        let log_p = (teacher[c] - mt) - lt;
                        let log_q = (row[c] - mz) - lz;
                        if p > 0.0 {
                            total += p * (log_p - log_q);
                        }
                        if gradient {
                            row[c] = (row[c] - mz).exp() / sz - p;
                        }
                    }
                    kl[r] = total;
                }
                Ok(kl)
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.kl_rows(target, logits, scored, gradient),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.kl_rows(target, logits, scored, gradient),
        }
    }

    /// In place of `logits`, the cotangent of `−log q_y` per row, `q − e_y`, with `y` drawn from
    /// the row's softmax `q` by its uniform `uniforms[r]` (`rows × 1`): the first class whose
    /// cumulative probability passes it; a row whose `scored` flag is zero becomes zero.
    pub fn sampled_cotangent(&self, logits: &mut Tensor, uniforms: &Tensor, scored: Option<&Indices>) -> Result<(), GpuError> {
        if uniforms.rows != logits.rows || uniforms.cols != 1 {
            return Err(shape(format!("{:?} uniforms for {:?} logits", uniforms.dim(), logits.dim())));
        }
        match &*self.backend {
            Backend::Host => {
                let (uv, cols) = (host(uniforms)?.to_vec(), logits.cols);
                let flags = scored.map(host_indices).transpose()?;
                for (r, row) in host_mut(logits)?.chunks_mut(cols).enumerate() {
                    if flags.is_some_and(|f| f[r] == 0) {
                        row.fill(0.0);
                        continue;
                    }
                    let q = host_softmax(row);
                    let mut pick = uv[r];
                    let mut label = cols - 1;
                    for (c, p) in q.iter().enumerate() {
                        if pick < *p {
                            label = c;
                            break;
                        }
                        pick -= p;
                    }
                    for c in 0..cols {
                        row[c] = q[c] - if c == label { 1.0 } else { 0.0 };
                    }
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.sampled_cotangent(logits, uniforms, scored),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.sampled_cotangent(logits, uniforms, scored),
        }
    }

    /// Prepare contiguous column groups for fused product/reduction kernels.
    pub fn column_blocks(&self, widths: &[usize]) -> Result<ColumnBlocks, GpuError> {
        let mut offsets = vec![0u32];
        let mut columns = 0usize;
        for &width in widths {
            if width == 0 { return Err(shape("empty column block".to_string())); }
            columns = columns.checked_add(width).ok_or_else(|| shape("column block overflow".to_string()))?;
            offsets.push(u32::try_from(columns).map_err(|_| shape("column blocks exceed u32".to_string()))?);
        }
        Ok(ColumnBlocks { offsets: self.upload_indices(&offsets)?, columns })
    }

    /// `out[r,b] = sum_{c in b} left[r,c] * right[r,c]`. Sum before any subsequent square
    /// to preserve cross terms in grouped-mask Fisher estimates.
    pub fn block_products(&self, left: &Tensor, right: &Tensor, blocks: &ColumnBlocks) -> Result<Tensor, GpuError> {
        same(left, right, "block products")?;
        if left.cols != blocks.columns { return Err(shape("column blocks do not match input".to_string())); }
        match &*self.backend {
            Backend::Host => {
                let (a, b, offsets) = (host(left)?, host(right)?, host_indices(&blocks.offsets)?);
                let mut out = vec![0.0; left.rows * blocks.len()];
                for r in 0..left.rows {
                    for k in 0..blocks.len() {
                        out[r * blocks.len() + k] = (offsets[k] as usize..offsets[k + 1] as usize)
                            .map(|c| a[r * left.cols + c] * b[r * left.cols + c]).sum();
                    }
                }
                Ok(Tensor { rows: left.rows, cols: blocks.len(), data: Data::Host(out) })
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.block_products(left, right, blocks),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.block_products(left, right, blocks),
        }
    }

    /// Draw a label from each probability row and return `mean − head[label]`, where
    /// `mean = probabilities * head`. A transposed head stores vocabulary vectors as columns.
    /// This reuses the distribution and its head mean across sampled-label reverse passes.
    pub fn sampled_head_cotangent(
        &self, probabilities: &Tensor, mean: &Tensor, head: &Tensor, transposed: bool,
        uniforms: &Tensor, scored: Option<&Indices>,
    ) -> Result<Tensor, GpuError> {
        let expected_head = if transposed { (mean.cols, probabilities.cols) } else { (probabilities.cols, mean.cols) };
        if probabilities.cols == 0 || mean.rows != probabilities.rows || head.dim() != expected_head
            || uniforms.dim() != (mean.rows, 1) || scored.is_some_and(|s| s.len != mean.rows)
        {
            return Err(shape("sampled head cotangent shapes".to_string()));
        }
        match &*self.backend {
            Backend::Host => {
                let (q, mu, weights, u) = (host(probabilities)?, host(mean)?, host(head)?, host(uniforms)?);
                let flags = scored.map(host_indices).transpose()?;
                let mut out = vec![0.0; mean.len()];
                for r in 0..mean.rows {
                    if flags.is_some_and(|f| f[r] == 0) { continue; }
                    let mut pick = u[r];
                    let mut label = probabilities.cols - 1;
                    for c in 0..probabilities.cols {
                        let p = q[r * probabilities.cols + c];
                        if pick < p { label = c; break; }
                        pick -= p;
                    }
                    for h in 0..mean.cols {
                        let index = if transposed { h * probabilities.cols + label } else { label * mean.cols + h };
                        out[r * mean.cols + h] = mu[r * mean.cols + h] - weights[index];
                    }
                }
                Ok(Tensor { rows: mean.rows, cols: mean.cols, data: Data::Host(out) })
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.sampled_head_cotangent(probabilities, mean, head, transposed, uniforms, scored),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.sampled_head_cotangent(probabilities, mean, head, transposed, uniforms, scored),
        }
    }

    /// Per row `r` of `hidden` (rows × width), the log partition `log Σ_c exp(hidden_r · e_c)` over
    /// the head's classes `e_c` (`head` classes × width, or width × classes when `transposed`), and
    /// with `expected` (rows × width) also `Σ_c q_rc e_c`, `q_r` the row's softmax: the log
    /// partition's gradient in `hidden_r`. A row whose `scored` flag is zero has zero of both.
    ///
    /// In CUDA f32 storage the rows × classes logits are never formed: the classes are swept in
    /// chunks whose logits live in one reused rows × chunk buffer (sized to stay in L2), a running
    /// (largest, sum) per row rescaled as the largest grows (the online softmax, summed in double),
    /// the expected rows accumulated alongside with the same rescaling, so the sweep costs the
    /// materialized form's two products and none of its memory. Float64 forms the logits, products
    /// in `arithmetic`; the Apple GPU forms them in row tiles of about 32 MB.
    pub fn head_log_partition(
        &self,
        hidden: &Tensor,
        head: &Tensor,
        transposed: bool,
        scored: Option<&Indices>,
        expected: Option<&mut Tensor>,
        arithmetic: Arithmetic,
    ) -> Result<Vec<f64>, GpuError> {
        let (rows, width) = hidden.dim();
        let classes = if transposed { head.cols } else { head.rows };
        let head_width = if transposed { head.rows } else { head.cols };
        if head_width != width || classes == 0 || scored.is_some_and(|s| s.len != rows) || expected.as_ref().is_some_and(|e| e.dim() != (rows, width)) {
            return Err(shape(format!("a {:?} head (transposed {transposed}) on {:?} rows", head.dim(), hidden.dim())));
        }
        if rows == 0 {
            return Ok(Vec::new());
        }
        match &*self.backend {
            Backend::Host => {
                let flags = scored.map(host_indices).transpose()?;
                self.head_log_partition_tiled(hidden, head, transposed, flags, expected, arithmetic)
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if hidden.storage() == Storage::F32 => {
                // Each chunk's logits fill about 32 MB, half the L40's L2 and so read back from it.
                let chunk = ((1usize << 23) / rows).max(512);
                engine.head_log_partition(hidden, (head, transposed), scored, expected, (chunk, arithmetic))
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(_) => {
                let mut logits = self.zeros(rows, classes)?;
                let (into, back) = if transposed { (Op::N, Op::T) } else { (Op::T, Op::N) };
                self.gemm(&mut logits, 1.0, hidden, Op::N, head, into, 0.0, arithmetic)?;
                let stats = self.softmax_stats_rows(&mut logits, scored)?;
                if let Some(out) = expected {
                    self.gemm(out, 1.0, &logits, Op::N, head, back, 0.0, arithmetic)?;
                }
                Ok(stats.iter().map(|s| s[0]).collect())
            }
            #[cfg(target_os = "macos")]
            Backend::Metal(_) => {
                let flags = scored.map(|s| match &s.data {
                    IndexData::Metal(_, values) => Ok(values.as_slice()),
                    _ => Err(foreign()),
                });
                let flags = flags.transpose()?;
                self.head_log_partition_tiled(hidden, head, transposed, flags, expected, arithmetic)
            }
        }
    }

    /// [`Device::head_log_partition`] in row tiles whose logits fill about 32 MB (the size of
    /// CUDA's swept chunks), through this device's own products and softmax statistics; `flags`
    /// are the scored rows' flags.
    fn head_log_partition_tiled(
        &self,
        hidden: &Tensor,
        head: &Tensor,
        transposed: bool,
        flags: Option<&[u32]>,
        mut expected: Option<&mut Tensor>,
        arithmetic: Arithmetic,
    ) -> Result<Vec<f64>, GpuError> {
        let (rows, width) = hidden.dim();
        let classes = if transposed { head.cols } else { head.rows };
        let tile = ((1usize << 23) / classes).max(1);
        let (into, back) = if transposed { (Op::N, Op::T) } else { (Op::T, Op::N) };
        let mut partitions = Vec::with_capacity(rows);
        for start in (0..rows).step_by(tile) {
            let n = tile.min(rows - start);
            let h = self.rows_of(hidden, start, n)?;
            let mut logits = self.zeros(n, classes)?;
            self.gemm(&mut logits, 1.0, &h, Op::N, head, into, 0.0, arithmetic)?;
            let part = flags.map(|f| self.upload_indices(&f[start..start + n])).transpose()?;
            let stats = self.softmax_stats_rows(&mut logits, part.as_ref())?;
            if let Some(out) = expected.as_deref_mut() {
                let mut mean = self.zeros(n, width)?;
                self.gemm(&mut mean, 1.0, &logits, Op::N, head, back, 0.0, arithmetic)?;
                self.set_rows(out, start, &mean)?;
            }
            partitions.extend(stats.iter().map(|s| s[0]));
        }
        Ok(partitions)
    }

    /// [`Device::head_log_partition`] leaving each row's log partition on the device in
    /// `partitions` (rows × 1, float64 storage whatever the operands' storage), so a captured step
    /// can record it: CUDA f32 storage records it whole; elsewhere this is the downloading form's
    /// values written back.
    pub fn head_log_partition_into(
        &self,
        hidden: &Tensor,
        head: &Tensor,
        transposed: bool,
        scored: Option<&Indices>,
        expected: Option<&mut Tensor>,
        partitions: &mut Tensor,
        arithmetic: Arithmetic,
    ) -> Result<(), GpuError> {
        if partitions.dim() != (hidden.rows, 1) || partitions.storage() != Storage::F64 {
            return Err(shape(format!("a float64 {:?} log partition column for {:?} rows", partitions.dim(), hidden.dim())));
        }
        match (&*self.backend, &mut partitions.data) {
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda(out)) if hidden.storage() == Storage::F32 && hidden.rows > 0 => {
                let classes = if transposed { head.cols } else { head.rows };
                let head_width = if transposed { head.rows } else { head.cols };
                if head_width != hidden.cols || classes == 0 || scored.is_some_and(|s| s.len != hidden.rows) || expected.as_ref().is_some_and(|e| e.dim() != hidden.dim()) {
                    return Err(shape(format!("a {:?} head (transposed {transposed}) on {:?} rows", head.dim(), hidden.dim())));
                }
                let chunk = ((1usize << 23) / hidden.rows).max(512);
                engine.head_log_partition_into(hidden, (head, transposed), scored, expected, out, (chunk, arithmetic))
            }
            _ => {
                let values = self.head_log_partition(hidden, head, transposed, scored, expected, arithmetic)?;
                let wide = self.with_storage(Storage::F64)?;
                let column = wide.upload_vec(values.len(), 1, values)?;
                wide.set_rows(partitions, 0, &column)
            }
        }
    }

    /// Per row, the first column holding its largest value (column 0 for a row with none above
    /// −∞): the top token of a row of logits.
    pub fn argmax_rows(&self, t: &Tensor) -> Result<Vec<usize>, GpuError> {
        match &*self.backend {
            Backend::Host => Ok(host(t)?
                .chunks(t.cols.max(1))
                .take(t.rows)
                .map(|row| row.iter().enumerate().fold((0, f64::NEG_INFINITY), |best, (c, v)| if *v > best.1 { (c, *v) } else { best }).0)
                .collect()),
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.argmax_rows(t),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.argmax_rows(t),
        }
    }

    /// `t`'s entries at the row-major positions `at` set to `value` (a sparse pattern written into
    /// a dense tensor, say a selection's units on).
    pub fn fill_entries(&self, t: &mut Tensor, at: &Indices, value: f64) -> Result<(), GpuError> {
        if t.len() > u32::MAX as usize {
            return Err(shape(format!("{:?} entries exceed 32-bit positions", t.dim())));
        }
        match &*self.backend {
            Backend::Host => {
                let (positions, len) = (host_indices(at)?.to_vec(), t.len());
                let values = host_mut(t)?;
                for p in positions {
                    *values.get_mut(p as usize).ok_or_else(|| shape(format!("position {p} of {len} entries")))? = value;
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.fill_entries(t, at, value),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.fill_entries(t, at, value),
        }
    }

    /// Per row, `Σ_c q_c (t_c − Σ_j q_j t_j)²` with `q = softmax(logits)`: the output Fisher's
    /// quadratic form on the tangent `t`.
    pub fn softmax_quadratic(&self, logits: &Tensor, tangent: &Tensor) -> Result<Vec<f64>, GpuError> {
        same(logits, tangent, "softmax quadratic")?;
        match &*self.backend {
            Backend::Host => {
                let (lv, tv, cols) = (host(logits)?, host(tangent)?, logits.cols);
                Ok((0..logits.rows)
                    .map(|r| {
                        let q = host_softmax(&lv[r * cols..(r + 1) * cols]);
                        let t = &tv[r * cols..(r + 1) * cols];
                        let mean: f64 = q.iter().zip(t).map(|(a, b)| a * b).sum();
                        q.iter().zip(t).map(|(a, b)| a * (b - mean) * (b - mean)).sum()
                    })
                    .collect())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.softmax_quadratic(logits, tangent),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.softmax_quadratic(logits, tangent),
        }
    }

    /// Rows `start..start + rows` of `t` as a new tensor.
    pub fn rows_of(&self, t: &Tensor, start: usize, rows: usize) -> Result<Tensor, GpuError> {
        if start + rows > t.rows {
            return Err(shape(format!("rows {start}..{} of {}", start + rows, t.rows)));
        }
        let (lo, hi) = (start * t.cols, (start + rows) * t.cols);
        let data = match (&*self.backend, &t.data) {
            (Backend::Host, Data::Host(v)) => Data::Host(v[lo..hi].to_vec()),
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda(slice)) => Data::Cuda(engine.copy_range(slice, lo, hi)?),
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda32(slice)) => Data::Cuda32(engine.copy_range(slice, lo, hi)?),
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::CudaBf16(slice)) => Data::CudaBf16(engine.copy_range(slice, lo, hi)?),
            #[cfg(target_os = "macos")]
            (Backend::Metal(engine), Data::Metal(buffer)) => Data::Metal(engine.copy_range(buffer, lo, hi)?),
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            _ => return Err(foreign()),
        };
        Ok(Tensor { rows, cols: t.cols, data })
    }

    /// Exact column copies into a fresh row-major tensor. No arithmetic or host
    /// transfer occurs on CUDA or Metal. Empty row/column domains are refused.
    pub fn columns_of(&self, t: &Tensor, columns: std::ops::Range<usize>) -> Result<Tensor, GpuError> {
        if t.rows == 0 || columns.start >= columns.end || columns.end > t.cols {
            return Err(shape("column copy requires a nonempty in-range domain".into()));
        }
        let width = columns.len();
        let count = t.rows.checked_mul(width).ok_or_else(|| shape("column copy size overflow".into()))?;
        let data = match (&*self.backend, &t.data) {
            (Backend::Host, Data::Host(v)) => {
                let mut values = Vec::with_capacity(count);
                for row in 0..t.rows { values.extend_from_slice(&v[row*t.cols+columns.start..row*t.cols+columns.end]); }
                Data::Host(values)
            }
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda(_) | Data::Cuda32(_)) => engine.columns_of(t, columns.start, width, count)?,
            #[cfg(target_os = "macos")]
            (Backend::Metal(engine), Data::Metal(_)) => engine.columns_of(t, columns.start, width, count)?,
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            _ => return Err(foreign()),
        };
        Ok(Tensor { rows:t.rows, cols:width, data })
    }

    /// Writes `part` into rows `start..` of `t`.
    pub fn set_rows(&self, t: &mut Tensor, start: usize, part: &Tensor) -> Result<(), GpuError> {
        if part.cols != t.cols || start + part.rows > t.rows {
            return Err(shape(format!("{:?} at row {start} of {:?}", part.dim(), t.dim())));
        }
        let lo = start * t.cols;
        match (&*self.backend, &mut t.data, &part.data) {
            (Backend::Host, Data::Host(v), Data::Host(p)) => {
                v[lo..lo + p.len()].copy_from_slice(p);
                Ok(())
            }
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda(slice), Data::Cuda(p)) => engine.write_range(slice, lo, p),
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda32(slice), Data::Cuda32(p)) => engine.write_range(slice, lo, p),
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::CudaBf16(slice), Data::CudaBf16(p)) => engine.write_range(slice, lo, p),
            #[cfg(target_os = "macos")]
            (Backend::Metal(engine), Data::Metal(buffer), Data::Metal(p)) => engine.write_range(buffer, lo, p, part.len()),
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            _ => Err(foreign()),
        }
    }

    /// Copy `part` into columns `start..` of every row of `t`, without arithmetic.
    /// This preserves literal bits and never executes a host fallback.
    pub fn set_columns(&self, t: &mut Tensor, start: usize, part: &Tensor) -> Result<(), GpuError> {
        if part.rows != t.rows || start.checked_add(part.cols).is_none_or(|end| end > t.cols) {
            return Err(shape(format!("{:?} at column {start} of {:?}", part.dim(), t.dim())));
        }
        if part.len() == 0 {
            return Ok(());
        }
        match (&*self.backend, &mut t.data, &part.data) {
            (Backend::Host, Data::Host(v), Data::Host(p)) => {
                for row in 0..t.rows {
                    let lo = row * t.cols + start;
                    v[lo..lo + part.cols].copy_from_slice(&p[row * part.cols..(row + 1) * part.cols]);
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda(out), Data::Cuda(input)) => engine.set_columns(Storage::F64, out, input, t.cols, part.cols, start, part.len()),
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda32(out), Data::Cuda32(input)) => engine.set_columns(Storage::F32, out, input, t.cols, part.cols, start, part.len()),
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::CudaBf16(out), Data::CudaBf16(input)) => engine.set_columns(Storage::Bf16, out, input, t.cols, part.cols, start, part.len()),
            #[cfg(target_os = "macos")]
            (Backend::Metal(engine), Data::Metal(out), Data::Metal(input)) => engine.set_columns(out, input, (t.cols, part.cols, start), part.len()),
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            _ => Err(foreign()),
        }
    }

    /// Row `(b·heads + h)·L + l` of the result is columns `start + h·width ..` of row `b·L + l` of
    /// `x` (`blocks` sequences of `L` rows): `heads` column blocks of width `width` copied
    /// head-major, each turned by `turn` (rotation tables `rows × planes` and pairing, as
    /// [`Device::rotate`]; backwards when `inverse`) in the same pass.
    pub fn split_heads(&self, x: &Tensor, start: usize, heads: usize, width: usize, blocks: usize, turn: Option<(&Tensor, &Tensor, bool)>, inverse: bool) -> Result<Tensor, GpuError> {
        let rows = x.rows;
        if blocks == 0 || rows % blocks != 0 || start.checked_add(heads.saturating_mul(width)).is_none_or(|end| end > x.cols) {
            return Err(shape(format!("{heads} heads of {width} from column {start} of {:?} in {blocks} blocks", x.dim())));
        }
        let mut out = self.zeros(rows * heads, width)?;
        self.heads(x, &mut out, (start, heads, width, rows / blocks), turn, inverse, false)?;
        Ok(out)
    }

    /// [`Device::split_heads`]' inverse: the head-major `x` (`blocks·heads·L × width`) written into
    /// columns `start .. start + heads·width` of `out`, turned by `turn` when given.
    pub fn merge_heads(&self, x: &Tensor, out: &mut Tensor, start: usize, heads: usize, blocks: usize, turn: Option<(&Tensor, &Tensor, bool)>, inverse: bool) -> Result<(), GpuError> {
        let (rows, width) = (out.rows, x.cols);
        if blocks == 0 || rows % blocks != 0 || x.rows != rows * heads || start.checked_add(heads.saturating_mul(width)).is_none_or(|end| end > out.cols) {
            return Err(shape(format!("{:?} head-major into {heads} heads from column {start} of {:?} in {blocks} blocks", x.dim(), out.dim())));
        }
        self.heads(x, out, (start, heads, width, rows / blocks), turn, inverse, true)
    }

    /// The permutation of [`Device::split_heads`] (or, `merge`, its inverse) from `x` into `out`.
    fn heads(&self, x: &Tensor, out: &mut Tensor, (start, heads, width, length): (usize, usize, usize, usize), turn: Option<(&Tensor, &Tensor, bool)>, inverse: bool, merge: bool) -> Result<(), GpuError> {
        let rows = if merge { out.rows } else { x.rows };
        let planes = turn.map_or(0, |(cos, _, _)| cos.cols);
        if let Some((cos, sin, _)) = turn {
            same(cos, sin, "rotation tables")?;
            if cos.rows != rows || 2 * planes > width {
                return Err(shape(format!("{:?} rotation tables on {rows} rows of {width}-wide heads", cos.dim())));
            }
        }
        if rows == 0 || heads == 0 || width == 0 {
            return Ok(());
        }
        let half_split = turn.is_some_and(|(_, _, h)| h);
        match &*self.backend {
            Backend::Host => {
                let (xv, cols) = (host(x)?, if merge { out.cols } else { x.cols });
                let (cv, sv) = match turn {
                    Some((cos, sin, _)) => (host(cos)?, host(sin)?),
                    None => (&[][..], &[][..]),
                };
                let sign = if inverse { -1.0 } else { 1.0 };
                let ov = host_mut(out)?;
                for i in 0..rows * heads * width {
                    let (j, t) = (i % width, i / width);
                    let (l, t) = (t % length, t / length);
                    let (h, b) = (t % heads, t / heads);
                    let r = b * length + l;
                    let (wide, narrow) = (r * cols + start + h * width, i - j);
                    let src = if merge { &xv[narrow..narrow + width] } else { &xv[wide..wide + width] };
                    let mut v = src[j];
                    if j < 2 * planes {
                        let (p, first, partner) = if half_split {
                            if j < planes { (j, true, j + planes) } else { (j - planes, false, j - planes) }
                        } else {
                            (j / 2, j % 2 == 0, if j % 2 == 0 { j + 1 } else { j - 1 })
                        };
                        let (c, s) = (cv[r * planes + p], sign * sv[r * planes + p]);
                        v = if first { c * v - s * src[partner] } else { s * src[partner] + c * v };
                    }
                    ov[if merge { wide + j } else { narrow + j }] = v;
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.heads(x, out, (start, heads, width, length, planes), turn, (half_split, inverse, merge)),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.heads(x, out, (start, heads, width, length, planes), turn, (half_split, inverse, merge)),
        }
    }

    /// One Adam step in place: `m ← β₁ m + (1 − β₁) g`, `v ← β₂ v + (1 − β₂) g²`, `w ← w − rate ·
    /// (m / (1 − β₁ᵗ)) / (√(v / (1 − β₂ᵗ)) + ε)`, `t` the step's number from 1.
    pub fn adam(&self, w: &mut Tensor, (m, v): (&mut Tensor, &mut Tensor), g: &Tensor, rate: f64, (beta1, beta2, epsilon): (f64, f64, f64), step: u64) -> Result<(), GpuError> {
        same(w, g, "adam gradient")?;
        same(w, m, "adam first moment")?;
        same(w, v, "adam second moment")?;
        let exponent = i32::try_from(step.max(1)).unwrap_or(i32::MAX);
        let (c1, c2) = (1.0 - beta1.powi(exponent), 1.0 - beta2.powi(exponent));
        match &*self.backend {
            Backend::Host => {
                let (gv, mv, vv) = (host(g)?, host_mut(m)?, host_mut(v)?);
                for (i, wi) in host_mut(w)?.iter_mut().enumerate() {
                    mv[i] = beta1 * mv[i] + (1.0 - beta1) * gv[i];
                    vv[i] = beta2 * vv[i] + (1.0 - beta2) * gv[i] * gv[i];
                    *wi -= rate * (mv[i] / c1) / ((vv[i] / c2).sqrt() + epsilon);
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.adam(w, (m, v), g, (rate, beta1, beta2, epsilon), (c1, c2)),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.adam(w, (m, v), g, (rate, beta1, beta2, epsilon), (c1, c2)),
        }
    }

    /// A weight sample of a factorized Gaussian posterior, `θᵢ = μᵢ + exp(sᵢ) εᵢ` with `εᵢ =
    /// [`posterior_normal`]`(key, stream, i)`, written into `theta` (this device's storage) from
    /// the posterior's means `μ` and log standard deviations `s` (`mean`, `log_sd`, in `theta`'s
    /// storage, or f32 for a bfloat16 `theta` on CUDA, written rounded to nearest). A removed entry
    /// (`s = −∞`, `μ = 0`) samples 0.
    /// The draws are regenerated from their counters, never stored ([`Device::posterior_adam`]
    /// regenerates the same ones).
    pub fn reparameterize(&self, theta: &mut Tensor, (mean, log_sd): (&Tensor, &Tensor), (key, stream): (u64, u64)) -> Result<(), GpuError> {
        same(theta, mean, "reparameterized mean")?;
        same(theta, log_sd, "reparameterized log standard deviation")?;
        match &*self.backend {
            Backend::Host => {
                let (mv, sv) = (host(mean)?, host(log_sd)?);
                for (i, t) in host_mut(theta)?.iter_mut().enumerate() {
                    *t = mv[i] + sv[i].exp() * f64::from(posterior_normal(key, stream, i as u64));
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.reparameterize(theta, (mean, log_sd), (key, stream)),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.reparameterize(theta, (mean, log_sd), (key, stream)),
        }
    }

    /// One Adam step of the factorized Gaussian posterior `N(μ, exp(s)²)` whose sample
    /// [`Device::reparameterize`] drew under the same `(key, stream)`, and each entry's new
    /// `(1, μ² + exp(2s), 2s)` added into its group's row of `sums` (groups × 3; float64 on CUDA
    /// and the host, f32 on the Apple GPU, as are `variance` and [`Device::group_divergence`]'s
    /// outputs). The entries (`μ`, `s`, the moments and `gradient`) share one storage. `gradient` times `step.gradient_scale` is the data term's
    /// gradient `g` at the sample; with `v` the entry's group variance (`variance`, groups × 1),
    /// the objective's derivatives are `g + μ / v` in `μ` and `g ε σ + σ² / v − 1` in `s` (`σ = exp(s)`, the empirical-Bayes
    /// group prior's divergence `½ (n ln v − Σ 2s)`), each taking Adam's step at its own rate with
    /// its moments (`moments`: `μ`'s first and second, then `s`'s). A removed entry (`s = −∞`)
    /// is left alone and adds nothing to `sums`.
    pub fn posterior_adam(
        &self,
        (mean, log_sd): (&mut Tensor, &mut Tensor),
        moments: [&mut Tensor; 4],
        gradient: &Tensor,
        (groups, variance): (&Indices, &Tensor),
        sums: &mut Tensor,
        step: &PosteriorStep,
    ) -> Result<(), GpuError> {
        same(mean, gradient, "posterior gradient")?;
        same(mean, log_sd, "posterior log standard deviation")?;
        for m in &moments {
            same(mean, m, "posterior moment")?;
        }
        if groups.len != mean.len() || variance.cols != 1 || sums.dim() != (variance.rows, 3) {
            return Err(shape(format!("{} group ids, a {:?} variance and {:?} sums for {} entries", groups.len, variance.dim(), sums.dim(), mean.len())));
        }
        let corrections = step.corrections();
        match &*self.backend {
            Backend::Host => {
                let [mm, mv, sm, sv] = moments;
                let (mm, mv, sm, sv) = (host_mut(mm)?, host_mut(mv)?, host_mut(sm)?, host_mut(sv)?);
                let (gv, ids, var) = (host(gradient)?, host_indices(groups)?, host(variance)?);
                let (means, log_sds, totals) = (host_mut(mean)?, host_mut(log_sd)?, host_mut(sums)?);
                if let Some(id) = ids.iter().find(|id| **id as usize >= var.len()) {
                    return Err(shape(format!("group {id} of {}", var.len())));
                }
                let (b1, b2) = (step.beta1, step.beta2);
                let adam = |w: &mut f64, m: &mut f64, v: &mut f64, g: f64, rate: f64| {
                    *m = b1 * *m + (1.0 - b1) * g;
                    *v = b2 * *v + (1.0 - b2) * g * g;
                    *w -= rate * (*m / corrections.0) / ((*v / corrections.1).sqrt() + step.epsilon);
                };
                for i in 0..means.len() {
                    if log_sds[i] == f64::NEG_INFINITY {
                        continue;
                    }
                    let (g, v) = (ids[i] as usize, var[ids[i] as usize]);
                    let (mu, sd, data) = (means[i], log_sds[i].exp(), step.gradient_scale * gv[i]);
                    let e = f64::from(posterior_normal(step.key, step.stream, i as u64));
                    adam(&mut means[i], &mut mm[i], &mut mv[i], data + mu / v, step.mean_rate);
                    adam(&mut log_sds[i], &mut sm[i], &mut sv[i], data * e * sd + sd * sd / v - 1.0, step.log_sd_rate);
                    totals[3 * g] += 1.0;
                    totals[3 * g + 1] += means[i] * means[i] + (2.0 * log_sds[i]).exp();
                    totals[3 * g + 2] += 2.0 * log_sds[i];
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.posterior_adam((mean, log_sd), moments, gradient, (groups, variance), sums, step),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.posterior_adam((mean, log_sd), moments, gradient, (groups, variance), sums, step),
        }
    }

    /// Each entry's `(1, μ² + exp(2s), 2s)` added into its group's row of `sums` (groups × 3), as
    /// [`Device::posterior_adam`] adds them after a step; a removed entry (`s = −∞`) adds nothing.
    pub fn group_moments(&self, (mean, log_sd): (&Tensor, &Tensor), groups: &Indices, sums: &mut Tensor) -> Result<(), GpuError> {
        same(mean, log_sd, "group moments")?;
        if groups.len != mean.len() || sums.cols != 3 {
            return Err(shape(format!("{} group ids and {:?} sums for {} entries", groups.len, sums.dim(), mean.len())));
        }
        match &*self.backend {
            Backend::Host => {
                let (means, log_sds, ids) = (host(mean)?, host(log_sd)?, host_indices(groups)?);
                let totals = host_mut(sums)?;
                for i in 0..means.len() {
                    let g = ids[i] as usize;
                    if g * 3 >= totals.len() {
                        return Err(shape(format!("group {g} of {}", totals.len() / 3)));
                    }
                    if log_sds[i] == f64::NEG_INFINITY {
                        continue;
                    }
                    totals[3 * g] += 1.0;
                    totals[3 * g + 1] += means[i] * means[i] + (2.0 * log_sds[i]).exp();
                    totals[3 * g + 2] += 2.0 * log_sds[i];
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.group_moments((mean, log_sd), groups, sums),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.group_moments((mean, log_sd), groups, sums),
        }
    }

    /// From each group's `sums` row `(n, Σ (μ² + σ²), Σ 2s)`, its empirical-Bayes prior variance
    /// `v = Σ (μ² + σ²) / n` into `variance` and its divergence `KL(q_G ‖ p_G) = ½ (n ln v − Σ 2s)`
    /// in nats into `divergence` (both groups × 1; zero for an empty group), then `sums` zeroed
    /// for the next accumulation.
    pub fn group_divergence(&self, sums: &mut Tensor, variance: &mut Tensor, divergence: &mut Tensor) -> Result<(), GpuError> {
        if sums.cols != 3 || variance.dim() != (sums.rows, 1) || divergence.dim() != (sums.rows, 1) {
            return Err(shape(format!("{:?} sums, {:?} variances, {:?} divergences", sums.dim(), variance.dim(), divergence.dim())));
        }
        match &*self.backend {
            Backend::Host => {
                let (totals, var, kl) = (host_mut(sums)?, host_mut(variance)?, host_mut(divergence)?);
                for g in 0..var.len() {
                    let (n, second, log_variance) = (totals[3 * g], totals[3 * g + 1], totals[3 * g + 2]);
                    (var[g], kl[g]) = if n > 0.0 {
                        let v = second / n;
                        (v, 0.5 * (n * v.ln() - log_variance))
                    } else {
                        (0.0, 0.0)
                    };
                }
                totals.fill(0.0);
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.group_divergence(sums, variance, divergence),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.group_divergence(sums, variance, divergence),
        }
    }

    /// Each row's set under a site's own code (`gam_mpd::site_fit`, module note): with `size = |a[r,
    /// c]| q[r, c]` each subcomponent's real size on the row (its read times its write in the row's
    /// metric) and `bound = left[r] + Σ_{c off} size`, the set
    /// minimising `Σ_{c on} bits[c] + weight[r] bound²`: the best prefix of the columns ranked by
    /// `size / bits` (ties by column) when it codes the row in fewer bits than the set `mask`
    /// holds, then single flips swept in column order until none lowers it. `mask` (rows × cols, 1
    /// on, 0 off) holds the current sets and gets the new ones; `a` and `q` are rows × cols, `bits`
    /// 1 × cols, `left` and `weight` rows × 1.
    pub fn select_sets(&self, (a, q): (&Tensor, &Tensor), bits: &Tensor, left: &Tensor, weight: &Tensor, mask: &mut Tensor) -> Result<(), GpuError> {
        same(a, q, "selected sets' metric")?;
        same(a, mask, "selected sets")?;
        if bits.dim() != (1, a.cols) || left.dim() != (a.rows, 1) || weight.dim() != (a.rows, 1) {
            return Err(shape(format!("a site's code on {:?} reads", a.dim())));
        }
        match &*self.backend {
            Backend::Host => {
                let (av, qv, bv, lv, wv, cols) = (host(a)?, host(q)?, host(bits)?, host(left)?, host(weight)?, a.cols);
                let out = host_mut(mask)?;
                for r in 0..a.rows {
                    let size: Vec<f64> = (r * cols..(r + 1) * cols).map(|i| av[i].abs() * qv[i]).collect();
                    let (w, m) = (wv[r], &mut out[r * cols..(r + 1) * cols]);
                    let ratio = |c: usize| match (bv[c] > 0.0, size[c] > 0.0) {
                        (true, _) => size[c] / bv[c],
                        (false, true) => f64::INFINITY,
                        (false, false) => 0.0,
                    };
                    let mut order: Vec<usize> = (0..cols).collect();
                    order.sort_by(|x, y| ratio(*y).total_cmp(&ratio(*x)));
                    let all: f64 = size.iter().sum();
                    let (mut listed, mut bound) = (0.0, lv[r] + all);
                    let (mut best, mut best_code) = (0, w * bound * bound);
                    for (k, &c) in order.iter().enumerate() {
                        listed += bv[c];
                        bound -= size[c];
                        let code = listed + w * bound * bound;
                        if code < best_code {
                            (best, best_code) = (k + 1, code);
                        }
                    }
                    let held = lv[r] + (0..cols).filter(|c| m[*c] == 0.0).map(|c| size[c]).sum::<f64>();
                    let held_code = (0..cols).filter(|c| m[*c] == 1.0).map(|c| bv[c]).sum::<f64>() + w * held * held;
                    if best_code < held_code {
                        m.fill(0.0);
                        for &c in &order[..best] {
                            m[c] = 1.0;
                        }
                    }
                    let mut bound = lv[r] + (0..cols).filter(|c| m[*c] == 0.0).map(|c| size[c]).sum::<f64>();
                    for _ in 0..cols {
                        let mut flipped = false;
                        for c in 0..cols {
                            let on = m[c] == 1.0;
                            let next = if on { bound + size[c] } else { (bound - size[c]).max(0.0) };
                            let delta = if on { -bv[c] } else { bv[c] } + w * (next * next - bound * bound);
                            if delta < 0.0 {
                                m[c] = if on { 0.0 } else { 1.0 };
                                bound = next;
                                flipped = true;
                            }
                        }
                        if !flipped {
                            break;
                        }
                    }
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.select_sets((a, q), bits, left, weight, mask),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.select_sets((a, q), bits, left, weight, mask),
        }
    }

    /// `x[i, j] ← x[i, j] / (rows[i] + cols[j])`, or 0 where that sum is not above `floor`: in the
    /// eigenbases of `K_r` and `K_c` (eigenvalues `rows`, rows × 1, and `cols`, 1 × cols), the
    /// solution of the Sylvester equation `K_r X + X K_c = C` from `C`.
    pub fn divide_sums(&self, x: &mut Tensor, rows: &Tensor, cols: &Tensor, floor: f64) -> Result<(), GpuError> {
        if rows.dim() != (x.rows, 1) || cols.dim() != (1, x.cols) {
            return Err(shape(format!("sums of {:?} and {:?} for {:?}", rows.dim(), cols.dim(), x.dim())));
        }
        let divide = |x: &mut [f64], r: &[f64], c: &[f64], width: usize| {
            for (i, v) in x.iter_mut().enumerate() {
                let sum = r[i / width] + c[i % width];
                *v = if sum > floor { *v / sum } else { 0.0 };
            }
        };
        match &*self.backend {
            Backend::Host => {
                let (r, c, width) = (host(rows)?.to_vec(), host(cols)?.to_vec(), x.cols);
                divide(host_mut(x)?, &r, &c, width);
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.divide_sums(x, rows, cols, floor),
            #[cfg(target_os = "macos")]
            Backend::Metal(_) => {
                let (mut values, r, c) = (self.download(x)?, self.download(rows)?, self.download(cols)?);
                let width = x.cols;
                divide(values.as_slice_mut().ok_or_else(|| shape("a downloaded tensor is contiguous".to_string()))?, r.as_slice().unwrap_or(&[]), c.as_slice().unwrap_or(&[]), width);
                *x = self.upload(values.view())?;
                Ok(())
            }
        }
    }

    /// The box claim's charge at one site (`gam_mpd::masked::box_upper`) and its gradients: per row
    /// `N_r = Σ_c (1 − m_rc) |z_rc| q_rc`, `q_rc` the subcomponent's write in the row's metric,
    /// returned as `½ N_r²`; `cot[r, c] += N_r (1 − m_rc) sign(z_rc) q_rc` (its gradient in `z`),
    /// and `coefficient[r, c] = N_r (1 − m_rc) |z_rc| / q_rc` (its gradient in `q`, over `q`; zero
    /// where `q` is). Every tensor is rows × cols.
    pub fn box_charge(&self, z: &Tensor, mask: &Tensor, q: &Tensor, cot: &mut Tensor, coefficient: &mut Tensor) -> Result<Vec<f64>, GpuError> {
        same(z, mask, "box charge mask")?;
        same(z, q, "box charge metric")?;
        same(z, cot, "box charge cotangent")?;
        same(z, coefficient, "box charge coefficient")?;
        match &*self.backend {
            Backend::Host => {
                let (zv, mv, qv, cols) = (host(z)?, host(mask)?, host(q)?, z.cols);
                let norms: Vec<f64> = (0..z.rows).map(|r| (r * cols..(r + 1) * cols).map(|i| (1.0 - mv[i]) * zv[i].abs() * qv[i]).sum()).collect();
                let out = host_mut(cot)?;
                for (i, o) in out.iter_mut().enumerate() {
                    let off = 1.0 - mv[i];
                    if off != 0.0 && zv[i] != 0.0 {
                        *o += norms[i / cols] * off * zv[i].signum() * qv[i];
                    }
                }
                for (i, o) in host_mut(coefficient)?.iter_mut().enumerate() {
                    *o = if qv[i] > 0.0 { norms[i / cols] * (1.0 - mv[i]) * zv[i].abs() / qv[i] } else { 0.0 };
                }
                Ok(norms.iter().map(|n| 0.5 * n * n).collect())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.box_charge(z, mask, q, cot, coefficient),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.box_charge(z, mask, q, cot, coefficient),
        }
    }

    /// Every row's sparse code (`gam_mpd::sparse_code::Coder`, its module note): row `r`'s blocks on
    /// minimise `Σ_b bits_b m_b + κ ‖y_r − Σ_b m_b Z_rb‖²_F`, found by the code's convex relaxation on
    /// the box (a working set solved by exact coordinate minimisation), rounded and improved by exact
    /// single flips, and branched best bound first within `nodes` nodes while its bounds are more
    /// than a bit apart: each row's blocks on (`on`, rows × blocks, 1 or 0), its code and a lower
    /// bound on its best, the CPU coder's steps in the same order. `z` and `w` are rows × pieces
    /// (`v_c · x_r` and `(U F y_r)_c`), `yfy` rows × 1, `gram` pieces × pieces (`U F Uᵀ`), `starts`
    /// the blocks' first pieces and the end, `bits` 1 × blocks, `warm` (rows × blocks, 1 or 0) each
    /// row's start, and `tolerance` the largest coordinate move at which a relaxation is stationary.
    /// The CPU's float64 coder is the reference: on the host this is [`GpuError::NoDeviceKernel`].
    pub fn code_rows(
        &self,
        (z, w, yfy): (&Tensor, &Tensor, &Tensor),
        (gram, starts, bits): (&Tensor, &Indices, &Tensor),
        warm: Option<&Tensor>,
        (kappa, nodes, tolerance): (f64, usize, f64),
        on: &mut Tensor,
    ) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
        self.code_rows_dispatch((z, w, yfy), (gram, starts, bits), warm, (kappa, nodes, tolerance), on, None)
            .map(|(upper, lower, _)| (upper, lower))
    }

    /// CUDA pilot: exact same search, with explicit scratch budget, optional lazy Q-column cache,
    /// and per-row counters. Returned upper/lower values are absolute objective bounds; no claim
    /// of a particular gap is made. Unsupported backends return an error without host execution.
    pub fn code_rows_profiled(
        &self,
        (z, w, yfy): (&Tensor, &Tensor, &Tensor),
        (gram, starts, bits): (&Tensor, &Indices, &Tensor),
        warm: Option<&Tensor>,
        (kappa, nodes, tolerance): (f64, usize, f64),
        on: &mut Tensor,
        workspace: CodeRowsWorkspace,
    ) -> Result<(Vec<f64>, Vec<f64>, CodeRowsDiagnostics), GpuError> {
        self.code_rows_dispatch((z, w, yfy), (gram, starts, bits), warm, (kappa, nodes, tolerance), on, Some((workspace, false)))
            .and_then(|(upper, lower, profile)| profile.map(|p| (upper, lower, p)).ok_or_else(|| shape("missing CUDA sparse-code diagnostics".to_string())))
    }

    /// Opt-in legacy proposal-coder pilot. Fuses each Q-column application into
    /// its axpy while preserving coordinate order and separately rounded arithmetic.
    /// The ordinary and profiled entry points keep the unfused path.
    pub fn code_rows_profiled_fused(
        &self,
        products: (&Tensor, &Tensor, &Tensor),
        structure: (&Tensor, &Indices, &Tensor),
        warm: Option<&Tensor>,
        settings: (f64, usize, f64),
        on: &mut Tensor,
        workspace: CodeRowsWorkspace,
    ) -> Result<(Vec<f64>, Vec<f64>, CodeRowsDiagnostics), GpuError> {
        self.code_rows_dispatch(products, structure, warm, settings, on, Some((workspace, true)))
            .and_then(|(upper, lower, profile)| profile.map(|p| (upper, lower, p)).ok_or_else(|| shape("missing CUDA sparse-code diagnostics".to_string())))
    }

    fn code_rows_dispatch(
        &self,
        (z, w, yfy): (&Tensor, &Tensor, &Tensor),
        (gram, starts, bits): (&Tensor, &Indices, &Tensor),
        warm: Option<&Tensor>,
        (kappa, nodes, tolerance): (f64, usize, f64),
        on: &mut Tensor,
        workspace: Option<(CodeRowsWorkspace, bool)>,
    ) -> Result<(Vec<f64>, Vec<f64>, Option<CodeRowsDiagnostics>), GpuError> {
        let (rows, pieces) = z.dim();
        let blocks = bits.cols;
        if w.dim() != (rows, pieces) || yfy.dim() != (rows, 1) || gram.dim() != (pieces, pieces) || starts.len != blocks + 1 || bits.rows != 1
            || on.dim() != (rows, blocks) || warm.is_some_and(|m| m.dim() != (rows, blocks))
        {
            return Err(shape(format!("a code of {:?} reads over {blocks} blocks", z.dim())));
        }
        if !(kappa.is_finite() && kappa > 0.0 && tolerance.is_finite() && tolerance >= 0.0) || u32::try_from(nodes).is_err() {
            return Err(shape(format!("a code at κ {kappa}, tolerance {tolerance} and {nodes} nodes")));
        }
        match &*self.backend {
            Backend::Host => Err(GpuError::NoDeviceKernel { reason: "the sparse code's CPU reference is gam_mpd::sparse_code".to_string() }),
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => {
                float64_only("the CUDA sparse code", &[z, w, yfy, gram, bits, on])?;
                engine.code_rows((z, w, yfy), (gram, starts, bits), warm, (kappa, nodes, tolerance), on, workspace)
            }
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => {
                if workspace.is_some() { return Err(GpuError::NoDeviceKernel { reason: "profiled sparse code requires CUDA".to_string() }); }
                engine.code_rows((z, w, yfy), (gram, starts, bits), warm, (kappa, nodes, tolerance), on).map(|(u,l)| (u,l,None))
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RmsMode {
    Value,
    Backward,
    Tangent,
}

/// One step of [`Device::posterior_adam`]: the data term's weight on the given gradient, Adam's
/// rates in `μ` and in `s`, decays and `ε`, the step's number from 1, and the `(key, stream)` of
/// the sample whose gradient it takes.
#[derive(Clone, Copy, Debug)]
pub struct PosteriorStep {
    pub gradient_scale: f64,
    pub mean_rate: f64,
    pub log_sd_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub step: u64,
    pub key: u64,
    pub stream: u64,
}

impl PosteriorStep {
    /// Adam's bias corrections `(1 − β₁ᵗ, 1 − β₂ᵗ)`.
    fn corrections(&self) -> (f64, f64) {
        let exponent = i32::try_from(self.step.max(1)).unwrap_or(i32::MAX);
        (1.0 - self.beta1.powi(exponent), 1.0 - self.beta2.powi(exponent))
    }
}

/// Philox4x32-10 (Salmon et al., SC 2011) of the counter `(index, stream)` under `key`.
fn philox(key: u64, stream: u64, index: u64) -> [u32; 4] {
    let mut c = [index as u32, (index >> 32) as u32, stream as u32, (stream >> 32) as u32];
    let mut k = [key as u32, (key >> 32) as u32];
    for round in 0..10 {
        if round > 0 {
            k[0] = k[0].wrapping_add(0x9E37_79B9);
            k[1] = k[1].wrapping_add(0xBB67_AE85);
        }
        let (p0, p1) = (u64::from(0xD251_1F53_u32) * u64::from(c[0]), u64::from(0xCD9E_8D57_u32) * u64::from(c[2]));
        c = [((p1 >> 32) as u32) ^ c[1] ^ k[0], p1 as u32, ((p0 >> 32) as u32) ^ c[3] ^ k[1], p0 as u32];
    }
    c
}

/// The standard normal draw `index` of `(key, stream)`: Box–Muller in f32 on the first two words of
/// [`philox`], `√(−2 ln u₁) cos(2π u₂)` with `u₁ = (w₀ + ½) 2⁻³²` and `u₂ = w₁ 2⁻³²`. Every backend
/// computes it alike (up to its f32 `log` and `cos`), so a draw is regenerated from its counter.
#[must_use]
pub fn posterior_normal(key: u64, stream: u64, index: u64) -> f32 {
    let w = philox(key, stream, index);
    let scale = 2.328_306_4e-10_f32;
    let (u1, u2) = ((w[0] as f32 + 0.5) * scale, w[1] as f32 * scale);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn host_softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
    let total: f64 = e.iter().sum();
    e.into_iter().map(|v| v / total).collect()
}

fn host_softmax_stats(z: &[f64]) -> (f64, f64) {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (m, z.iter().map(|v| (v - m).exp()).sum())
}

fn host_gemm(
    batch: usize,
    ab: (usize, usize),
    bb: (usize, usize),
    cb: (usize, usize),
    (alpha, beta): (f64, f64),
    (a, ta): (&[f64], Op),
    (b, tb): (&[f64], Op),
    c: &mut [f64],
    arithmetic: Arithmetic,
) -> Result<(), GpuError> {
    let lowered = |v: &[f64]| -> Vec<f64> { v.iter().map(|x| round_operand(*x, arithmetic)).collect() };
    let (a_low, b_low);
    let (a, b) = if arithmetic == Arithmetic::F64 {
        (a, b)
    } else {
        a_low = lowered(a);
        b_low = lowered(b);
        (&a_low[..], &b_low[..])
    };
    let size = cb.0 * cb.1;
    if size == 0 {
        return Ok(());
    }
    // The products of a batch, and row blocks of each product, run on the rayon pool: the gemm
    // library's own threading stops at four threads. Each entry's sum is the same either way.
    c.par_chunks_mut(size).take(batch).enumerate().try_for_each(|(i, c)| -> Result<(), GpuError> {
        let av = ArrayView2::from_shape(ab, &a[i * ab.0 * ab.1..(i + 1) * ab.0 * ab.1]).map_err(|e| shape(e.to_string()))?;
        let bv = ArrayView2::from_shape(bb, &b[i * bb.0 * bb.1..(i + 1) * bb.0 * bb.1]).map_err(|e| shape(e.to_string()))?;
        let mut cv = ArrayViewMut2::from_shape(cb, c).map_err(|e| shape(e.to_string()))?;
        let av = if ta == Op::T { av.reversed_axes() } else { av };
        let bv = if tb == Op::T { bv.reversed_axes() } else { bv };
        let rows = cb.0.div_ceil(rayon::current_num_threads()).max(1);
        cv.axis_chunks_iter_mut(Axis(0), rows).into_par_iter().zip(av.axis_chunks_iter(Axis(0), rows)).for_each(|(mut cv, av)| {
            general_mat_mul(alpha, &av, &bv, beta, &mut cv);
            if arithmetic != Arithmetic::F64 {
                cv.mapv_inplace(|x| f64::from(x as f32));
            }
        });
        Ok(())
    })
}

#[cfg(target_os = "linux")]
mod cuda {
    use super::{Arithmetic, CheckedInterval, CheckedIntervalReason, CheckedScalar, KlProposalRow, CodeRowCounters, CodeRowsDiagnostics, CodeRowsWorkspace, code_rows_layout, ColumnBlocks, Data, IndexData, Indices, Op, RmsMode, Storage, Tensor, foreign, shape};
    use crate::gpu_error::{GpuError, GpuResultExt};
    use cudarc::cublas::sys::{cublasComputeType_t, cublasGemmAlgo_t, cublasMath_t, cublasOperation_t, cudaDataType_t};
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
    use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
    use cudarc::driver::{CudaContext, CudaFunction, CudaGraph, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, LaunchArgs, LaunchConfig, PushKernelArg, SyncOnDrop, ValidAsZeroBits};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    const BLOCK: u32 = 256;

    /// The kernels, in the CUDA C subset HIP compiles unchanged.
    const KERNELS: &str = r#"
#define BLOCK 256
typedef unsigned long long u64;
// −∞ without <math.h> (NVRTC compiles without the host headers).
#define NEG_INF __longlong_as_double(0xfff0000000000000LL)

__device__ double block_sum(double v, double* shared) {
    unsigned int t = threadIdx.x;
    shared[t] = v;
    __syncthreads();
    for (unsigned int s = BLOCK / 2; s > 0; s >>= 1) {
        if (t < s) shared[t] += shared[t + s];
        __syncthreads();
    }
    double r = shared[0];
    __syncthreads();
    return r;
}

__device__ double block_max(double v, double* shared) {
    unsigned int t = threadIdx.x;
    shared[t] = v;
    __syncthreads();
    for (unsigned int s = BLOCK / 2; s > 0; s >>= 1) {
        if (t < s) shared[t] = fmax(shared[t], shared[t + s]);
        __syncthreads();
    }
    double r = shared[0];
    __syncthreads();
    return r;
}

#define GRID_STRIDE(i, n) for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (u64)gridDim.x * blockDim.x)

extern "C" __global__ void axpy(u64 n, double alpha, const double* x, double* y) {
    GRID_STRIDE(i, n) y[i] += alpha * x[i];
}

extern "C" __global__ void scaled_row_l2(u64 rows, u64 cols, u64 begin, u64 end,
    double scale, double scale_lo, double scale_hi, const double* x, double* out) {
    for (u64 row = blockIdx.x * (u64)blockDim.x + threadIdx.x; row < rows; row += gridDim.x * (u64)blockDim.x) {
        double maximum = 0.0;
        bool finite = true;
        for (u64 col = begin; col < end; ++col) {
            double v = x[row * cols + col];
            finite = finite && isfinite(v);
            maximum = fmax(maximum, fabs(v));
        }
        if (!finite || (maximum != 0.0 && scale_lo == 0.0)) {
            out[3*row] = out[3*row+1] = out[3*row+2] = __longlong_as_double(0x7ff8000000000000LL);
            continue;
        }
        if (maximum == 0.0) { out[3*row] = out[3*row+1] = out[3*row+2] = 0.0; continue; }
        double sum = 0.0, lo = 0.0, hi = 0.0;
        for (u64 col = begin; col < end; ++col) {
            double v = fabs(x[row * cols + col]);
            if (v == 0.0) continue;
            double q = __ddiv_rn(v, maximum);
            double qlo = __ddiv_rd(v, maximum), qhi = __ddiv_ru(v, maximum);
            sum = __dadd_rn(sum, __dmul_rn(q, q));
            lo = __dadd_rd(lo, __dmul_rd(qlo, qlo));
            hi = __dadd_ru(hi, __dmul_ru(qhi, qhi));
        }
        out[3*row] = __dmul_rn(__ddiv_rn(maximum, scale), __dsqrt_rn(sum));
        out[3*row+1] = __dmul_rd(__ddiv_rd(maximum, scale_hi), __dsqrt_rd(lo));
        out[3*row+2] = __dmul_ru(__ddiv_ru(maximum, scale_lo), __dsqrt_ru(hi));
    }
}

extern "C" __global__ void columns_of(u64 n, u64 source_cols, u64 width, u64 start, const double* source, double* out) {
    GRID_STRIDE(i, n) out[i] = source[(i / width) * source_cols + start + i % width];
}

extern "C" __global__ void set_columns(u64 n, u64 output_cols, u64 input_cols, u64 start, const double* input, double* output) {
    GRID_STRIDE(i, n) output[(i / input_cols) * output_cols + start + i % input_cols] = input[i];
}

extern "C" __global__ void set_columns_bits16(u64 n, u64 output_cols, u64 input_cols, u64 start, const unsigned short* input, unsigned short* output) {
    GRID_STRIDE(i, n) output[(i / input_cols) * output_cols + start + i % input_cols] = input[i];
}

extern "C" __global__ void hadamard(u64 n, const double* a, const double* b, double* out, int accumulate) {
    GRID_STRIDE(i, n) out[i] = accumulate ? out[i] + a[i] * b[i] : a[i] * b[i];
}

extern "C" __global__ void add_row(u64 n, unsigned int cols, double alpha, const double* row, double* x) {
    GRID_STRIDE(i, n) x[i] += alpha * row[i % cols];
}

extern "C" __global__ void scale_columns(u64 n, unsigned int cols, const double* x, const double* d, double* out, int accumulate) {
    GRID_STRIDE(i, n) {
        double term = x[i] * d[i % cols];
        out[i] = accumulate ? out[i] + term : term;
    }
}

extern "C" __global__ void gather_rows(u64 n, unsigned int cols, const double* table, const unsigned int* ids, double* out) {
    GRID_STRIDE(i, n) out[i] = table[(u64)ids[i / cols] * cols + i % cols];
}

extern "C" __global__ void to_f32(u64 n, const double* x, float* y) {
    GRID_STRIDE(i, n) y[i] = (float)x[i];
}

extern "C" __global__ void to_f64(u64 n, const float* y, double* x) {
    GRID_STRIDE(i, n) x[i] = (double)y[i];
}

__device__ double law_value(unsigned int code, double t, double c) {
    switch (code) {
        case 0: return t > 0.0 ? t : 0.0;
        case 1: return t;
        case 2: return 0.0;
        case 3: return t / (1.0 + exp(-t));
        case 4: return t * normcdf(t);
        default: {
            double inner = c * (t + 0.044715 * t * t * t);
            return 0.5 * t * (1.0 + tanh(inner));
        }
    }
}

__device__ double law_slope(unsigned int code, double t, double c) {
    switch (code) {
        case 0: return t > 0.0 ? 1.0 : 0.0;
        case 1: return 1.0;
        case 2: return 0.0;
        case 3: {
            double sigma = 1.0 / (1.0 + exp(-t));
            return sigma * (1.0 + t * (1.0 - sigma));
        }
        case 4: return normcdf(t) + t * (exp(-0.5 * t * t) * 0.3989422804014327);
        default: {
            double inner = c * (t + 0.044715 * t * t * t);
            double th = tanh(inner);
            return 0.5 * (1.0 + th) + 0.5 * t * (1.0 - th * th) * c * (1.0 + 3.0 * 0.044715 * t * t);
        }
    }
}

extern "C" __global__ void laws(u64 n, unsigned int cols, const double* x, const double* g, int slopes, const unsigned int* codes, double c, double* out) {
    GRID_STRIDE(i, n) {
        unsigned int code = codes[i % cols];
        out[i] = slopes ? g[i] * law_slope(code, x[i], c) : law_value(code, x[i], c);
    }
}

// mode 0: value; 1: cotangent given g; 2: tangent along g.
extern "C" __global__ void rms(unsigned int rows, unsigned int cols, int mode, double epsilon, const double* x, const double* g, double* out) {
    __shared__ double shared[BLOCK];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    const double* xr = x + (u64)r * cols;
    double* o = out + (u64)r * cols;
    double squares = 0.0, inner = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) {
        squares += xr[c] * xr[c];
        if (mode != 0) inner += xr[c] * g[(u64)r * cols + c];
    }
    double n = (double)cols;
    double mean = block_sum(squares, shared) / n;
    double scale = 1.0 / sqrt(mean + epsilon);
    if (mode == 0) {
        for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) o[c] = xr[c] * scale;
        return;
    }
    double dot = block_sum(inner, shared);
    const double* gr = g + (u64)r * cols;
    if (mode == 1) {
        for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) o[c] = scale * gr[c] - scale * scale * scale / n * xr[c] * dot;
    } else {
        double dm = 2.0 * dot / n;
        double ds = -0.5 * scale * scale * scale * dm;
        for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) o[c] = gr[c] * scale + xr[c] * ds;
    }
}

extern "C" __global__ void rotate_planes(unsigned int rows, unsigned int cols, unsigned int planes, int half_split, double sign,
                                         const double* x, const double* cosines, const double* sines, double* out) {
    GRID_STRIDE(i, (u64)rows * planes) {
        u64 r = i / planes;
        unsigned int p = (unsigned int)(i % planes);
        unsigned int a = half_split ? p : 2 * p;
        unsigned int b = half_split ? p + planes : 2 * p + 1;
        double c = cosines[i];
        double s = sign * sines[i];
        double xa = x[r * cols + a], xb = x[r * cols + b];
        out[r * cols + a] = c * xa - s * xb;
        out[r * cols + b] = s * xa + c * xb;
    }
}

// Heads between a row-major `rows × cols` tensor's column blocks (`heads` of `width` from `start`)
// and head-major order (row `(b·heads + h)·length + l` for row `b·length + l`): `merge` = 0 copies
// row-major to head-major, 1 back. The first `2·planes` columns of each head turn by the tables
// (`rows × planes`, pairing as `rotate_planes`, `sign` −1 backwards) on the way.
extern "C" __global__ void heads_permute(unsigned int rows, unsigned int cols, unsigned int start, unsigned int heads, unsigned int width,
                                         unsigned int length, unsigned int planes, int half_split, double sign, int merge,
                                         const double* x, const double* cosines, const double* sines, double* out) {
    GRID_STRIDE(i, (u64)rows * heads * width) {
        unsigned int j = (unsigned int)(i % width);
        u64 t = i / width;
        unsigned int l = (unsigned int)(t % length);
        t /= length;
        unsigned int h = (unsigned int)(t % heads);
        u64 r = (t / heads) * length + l;
        u64 wide = r * cols + start + (u64)h * width, narrow = i - j;
        const double* src = x + (merge ? narrow : wide);
        double v = src[j];
        if (j < 2 * planes) {
            unsigned int p = half_split ? (j < planes ? j : j - planes) : j / 2;
            int first = half_split ? j < planes : (j % 2) == 0;
            unsigned int partner = half_split ? (first ? j + planes : j - planes) : (first ? j + 1 : j - 1);
            double c = cosines[r * planes + p], s = sign * sines[r * planes + p];
            double o = src[partner];
            v = first ? c * v - s * o : s * o + c * v;
        }
        out[(merge ? wide : narrow) + j] = v;
    }
}

extern "C" __global__ void softmax_rows(unsigned int rows, unsigned int width, int causal, unsigned int start, unsigned int period, double* s) {
    __shared__ double shared[BLOCK];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    double* row = s + (u64)r * width;
    unsigned int valid = causal ? start + r % period + 1 : width;
    double m = NEG_INF;
    for (unsigned int c = threadIdx.x; c < valid; c += BLOCK) m = fmax(m, row[c]);
    m = block_max(m, shared);
    double total = 0.0;
    for (unsigned int c = threadIdx.x; c < width; c += BLOCK) {
        if (c < valid) {
            double e = exp(row[c] - m);
            row[c] = e;
            total += e;
        } else {
            row[c] = 0.0;
        }
    }
    total = block_sum(total, shared);
    for (unsigned int c = threadIdx.x; c < valid; c += BLOCK) row[c] /= total;
}

extern "C" __global__ void softmax_backward(unsigned int rows, unsigned int cols, const double* alpha, const double* d, double* out) {
    __shared__ double shared[BLOCK];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    const double* a = alpha + (u64)r * cols;
    const double* dr = d + (u64)r * cols;
    double partial = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) partial += a[c] * dr[c];
    double mean = block_sum(partial, shared);
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) out[(u64)r * cols + c] = a[c] * (dr[c] - mean);
}

// The row's max and the sum of exp(z - max).
__device__ void softmax_stats(const double* z, unsigned int cols, double* shared, double* m, double* total) {
    double mm = NEG_INF;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) mm = fmax(mm, z[c]);
    mm = block_max(mm, shared);
    double t = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) t += exp(z[c] - mm);
    *m = mm;
    *total = block_sum(t, shared);
}

extern "C" __global__ void softmax_stats_rows(unsigned int rows, unsigned int cols, double* logits,
                                             const unsigned int* scored, int use_scored, double* out) {
    __shared__ double shared[BLOCK];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    double* z = logits + (u64)r * cols;
    if (use_scored && !scored[r]) {
        for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) z[c] = 0.0;
        if (threadIdx.x == 0) { out[(u64)r*2] = 0.0; out[(u64)r*2+1] = 0.0; }
        return;
    }
    double bad = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) if (!isfinite(z[c])) bad += 1.0;
    double invalid = block_sum(bad, shared);
    if (invalid > 0.0) {
        if (threadIdx.x == 0) { out[(u64)r*2] = __longlong_as_double(0x7ff8000000000000LL); out[(u64)r*2+1] = __longlong_as_double(0x7ff8000000000000LL); }
        return;
    }
    double maximum, sum;
    softmax_stats(z, cols, shared, &maximum, &sum);
    double log_sum = log(sum), entropy = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) {
        double shifted = z[c] - maximum;
        double probability = exp(shifted) / sum;
        if (probability > 0.0) entropy += probability * (shifted - log_sum);
        z[c] = probability;
    }
    double total = block_sum(entropy, shared);
    if (threadIdx.x == 0) { out[(u64)r*2] = maximum + log_sum; out[(u64)r*2+1] = total; }
}

extern "C" __global__ void kl_rows(unsigned int rows, unsigned int cols, const double* target, double* logits,
                                   const unsigned int* scored, int use_scored, int gradient, double* kl) {
    __shared__ double shared[BLOCK];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    const double* t = target + (u64)r * cols;
    double* z = logits + (u64)r * cols;
    if (use_scored && scored[r] == 0) {
        if (gradient) for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) z[c] = 0.0;
        if (threadIdx.x == 0) kl[r] = 0.0;
        return;
    }
    double mt, st, mz, sz;
    softmax_stats(t, cols, shared, &mt, &st);
    softmax_stats(z, cols, shared, &mz, &sz);
    double lt = log(st), lz = log(sz);
    double acc = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) {
        double p = exp(t[c] - mt) / st;
        double log_p = (t[c] - mt) - lt;
        double log_q = (z[c] - mz) - lz;
        if (p > 0.0) acc += p * (log_p - log_q);
        if (gradient) z[c] = exp(z[c] - mz) / sz - p;
    }
    double total = block_sum(acc, shared);
    if (threadIdx.x == 0) kl[r] = total;
}

// Diagnostic reduction only. Accuracy terms are NOT a certified acceptance band.
extern "C" __global__ void kl_proposal_rows(unsigned int rows, unsigned int cols,
                                            const double* target, const double* logits, double* out) {
    __shared__ double shared[BLOCK];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    const double* p = target + (u64)r * cols;
    const double* q = logits + (u64)r * cols;
    double bad = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) {
        if (!isfinite(p[c]) || !isfinite(q[c])) bad += 1.0;
    }
    double invalid = block_sum(bad, shared);
    if (invalid > 0.0) {
        if (threadIdx.x == 0) for (unsigned int j = 0; j < 9; ++j) out[(u64)r * 9 + j] = j == 8 ? invalid : 0.0;
        return;
    }
    double mp, sp, mq, sq;
    softmax_stats(p, cols, shared, &mp, &sp);
    softmax_stats(q, cols, shared, &mq, &sq);
    double lp = log(sp), lq = log(sq);
    double value = 0.0, magnitude = 0.0, spread = 0.0, largest_difference = 0.0;
    bad = (!isfinite(lp) || !isfinite(lq)) ? 1.0 : 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) {
        double ap = p[c] - mp, aq = q[c] - mq;
        double log_p = ap - lp, log_q = aq - lq;
        double probability = exp(ap) / sp;
        if (!isfinite(ap) || !isfinite(aq) || !isfinite(log_p) || !isfinite(log_q) || !isfinite(probability)) { bad += 1.0; continue; }
        largest_difference = fmax(largest_difference, fabs(log_p - log_q));
        if (probability > 0.0) {
            value += probability * (log_p - log_q);
            magnitude += probability * (fabs(log_p) + fabs(log_q));
            spread += probability * fabs(log_p - log_q);
        }
    }
    double total = block_sum(value, shared);
    double mag = block_sum(magnitude, shared);
    double distance = block_sum(spread, shared);
    double max_difference = block_max(largest_difference, shared);
    invalid = block_sum(bad, shared);
    if (threadIdx.x == 0) {
        double* row = out + (u64)r * 9;
        row[0] = total; row[1] = mag; row[2] = distance;
        row[3] = mp; row[4] = lp; row[5] = mq; row[6] = lq; row[7] = max_difference; row[8] = invalid;
    }
}

extern "C" __global__ void sampled_cotangent(unsigned int rows, unsigned int cols, double* logits, const double* uniforms,
                                             const unsigned int* scored, int use_scored) {
    __shared__ double shared[BLOCK];
    __shared__ unsigned int label;
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    double* z = logits + (u64)r * cols;
    if (use_scored && scored[r] == 0) {
        for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) z[c] = 0.0;
        return;
    }
    double m, total;
    softmax_stats(z, cols, shared, &m, &total);
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) z[c] = exp(z[c] - m) / total;
    __syncthreads();
    if (threadIdx.x == 0) {
        double pick = uniforms[r];
        unsigned int chosen = cols - 1;
        for (unsigned int c = 0; c < cols; c++) {
            if (pick < z[c]) { chosen = c; break; }
            pick -= z[c];
        }
        label = chosen;
    }
    __syncthreads();
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) z[c] -= (c == label) ? 1.0 : 0.0;
}

extern "C" __global__ void block_products(u64 n, unsigned int cols, unsigned int blocks,
    const double* left, const double* right, const unsigned int* offsets, double* out) {
    GRID_STRIDE(i, n) {
        u64 row = i / blocks;
        unsigned int block = (unsigned int)(i % blocks);
        double sum = 0.0;
        for (unsigned int c = offsets[block]; c < offsets[block + 1]; c++)
            sum += left[row * cols + c] * right[row * cols + c];
        out[i] = sum;
    }
}

extern "C" __global__ void sampled_head_cotangent(unsigned int rows, unsigned int classes, unsigned int width,
    const double* probabilities, const double* mean, const double* head, int transposed,
    const double* uniforms, const unsigned int* scored, int use_scored, double* out) {
    __shared__ unsigned int label;
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    if (use_scored && scored[r] == 0) {
        for (unsigned int h = threadIdx.x; h < width; h += BLOCK) out[(u64)r * width + h] = 0.0;
        return;
    }
    if (threadIdx.x == 0) {
        double pick = uniforms[r];
        label = classes - 1;
        for (unsigned int c = 0; c < classes; c++) {
            double p = probabilities[(u64)r * classes + c];
            if (pick < p) { label = c; break; }
            pick -= p;
        }
    }
    __syncthreads();
    for (unsigned int h = threadIdx.x; h < width; h += BLOCK) {
        u64 index = transposed ? (u64)h * classes + label : (u64)label * width + h;
        out[(u64)r * width + h] = mean[(u64)r * width + h] - head[index];
    }
}

extern "C" __global__ void adam(u64 n, double rate, double beta1, double beta2, double epsilon, double c1, double c2,
    const double* g, double* m, double* v, double* w) {
    GRID_STRIDE(i, n) {
        double gi = g[i];
        double mi = beta1 * m[i] + (1.0 - beta1) * gi;
        double vi = beta2 * v[i] + (1.0 - beta2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        w[i] -= rate * (mi / c1) / (sqrt(vi / c2) + epsilon);
    }
}

// Whether (ka, ia) ranks before (kb, ib): larger key first, ties by column.
__device__ bool ranks_before(double ka, unsigned int ia, double kb, unsigned int ib) {
    return ka > kb || (ka == kb && ia < ib);
}

// The ranking key of a subcomponent: real size per description bit (unpaid ones first, unless
// they write nothing).
__device__ double ranking_key(double size, double bits) {
    if (bits > 0.0) return size / bits;
    return size > 0.0 ? __longlong_as_double(0x7ff0000000000000LL) : 0.0;
}

// One block per row (striding): the row's ranking sorted in its block's scratch (`width` a power
// of two at least `cols`), then its best prefix and single flips by one thread, as the host's
// `select_sets`.
extern "C" __global__ void select_sets(unsigned int rows, unsigned int cols, unsigned int width,
    const double* a, const double* q, const double* bits, const double* left, const double* weight,
    double* keys, unsigned int* order, double* sizes, double* mask) {
    __shared__ double shared[BLOCK];
    double* key = keys + (u64)blockIdx.x * width;
    unsigned int* idx = order + (u64)blockIdx.x * width;
    double* sr = sizes + (u64)blockIdx.x * width;
    for (unsigned int r = blockIdx.x; r < rows; r += gridDim.x) {
        double* m = mask + (u64)r * cols;
        double partial = 0.0, off = 0.0, on_bits = 0.0;
        for (unsigned int c = threadIdx.x; c < width; c += BLOCK) {
            if (c < cols) {
                sr[c] = fabs(a[(u64)r * cols + c]) * q[(u64)r * cols + c];
                partial += sr[c];
                if (m[c] == 0.0) off += sr[c]; else on_bits += bits[c];
                key[c] = ranking_key(sr[c], bits[c]);
            } else {
                key[c] = NEG_INF;
            }
            idx[c] = c;
        }
        double all = block_sum(partial, shared);
        double held_off = block_sum(off, shared);
        double held_listed = block_sum(on_bits, shared);
        for (unsigned int k = 2; k <= width; k <<= 1) {
            for (unsigned int j = k >> 1; j > 0; j >>= 1) {
                for (unsigned int i = threadIdx.x; i < width; i += BLOCK) {
                    unsigned int l = i ^ j;
                    if (l > i) {
                        bool first = (i & k) == 0;
                        bool swap = first ? ranks_before(key[l], idx[l], key[i], idx[i]) : ranks_before(key[i], idx[i], key[l], idx[l]);
                        if (swap) {
                            double tk = key[i]; key[i] = key[l]; key[l] = tk;
                            unsigned int ti = idx[i]; idx[i] = idx[l]; idx[l] = ti;
                        }
                    }
                }
                __syncthreads();
            }
        }
        if (threadIdx.x == 0) {
            double w = weight[r], lr = left[r];
            double listed = 0.0, bound = lr + all;
            double best_code = w * bound * bound;
            unsigned int best = 0;
            for (unsigned int k = 0; k < cols; k++) {
                unsigned int c = idx[k];
                listed += bits[c];
                bound -= sr[c];
                double code = listed + w * bound * bound;
                if (code < best_code) { best = k + 1; best_code = code; }
            }
            double held = lr + held_off;
            if (best_code < held_listed + w * held * held) {
                for (unsigned int c = 0; c < cols; c++) m[c] = 0.0;
                for (unsigned int k = 0; k < best; k++) m[idx[k]] = 1.0;
            }
            bound = lr;
            for (unsigned int c = 0; c < cols; c++) if (m[c] == 0.0) bound += sr[c];
            for (unsigned int sweep = 0; sweep < cols; sweep++) {
                int flipped = 0;
                for (unsigned int c = 0; c < cols; c++) {
                    int on = m[c] == 1.0;
                    double next = on ? bound + sr[c] : fmax(bound - sr[c], 0.0);
                    double delta = (on ? -bits[c] : bits[c]) + w * (next * next - bound * bound);
                    if (delta < 0.0) { m[c] = on ? 0.0 : 1.0; bound = next; flipped = 1; }
                }
                if (!flipped) break;
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void divide_sums(u64 n, unsigned int cols, const double* r, const double* c, double floor, double* x) {
    GRID_STRIDE(i, n) {
        double sum = r[i / cols] + c[i % cols];
        x[i] = sum > floor ? x[i] / sum : 0.0;
    }
}

extern "C" __global__ void box_charge(unsigned int rows, unsigned int cols, const double* z, const double* mask,
    const double* q, double* norms, double* cot, double* coefficient) {
    __shared__ double shared[BLOCK];
    for (unsigned int r = blockIdx.x; r < rows; r += gridDim.x) {
        u64 base = (u64)r * cols;
        double partial = 0.0;
        for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) partial += (1.0 - mask[base + c]) * fabs(z[base + c]) * q[base + c];
        double n = block_sum(partial, shared);
        if (threadIdx.x == 0) norms[r] = n;
        for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) {
            double off = 1.0 - mask[base + c], zc = z[base + c], qc = q[base + c];
            if (off != 0.0 && zc != 0.0) cot[base + c] += n * off * (zc > 0.0 ? 1.0 : -1.0) * qc;
            coefficient[base + c] = qc > 0.0 ? n * off * fabs(zc) / qc : 0.0;
        }
    }
}

// One block per row: each thread's first largest value, then the smallest column among the largest.
extern "C" __global__ void argmax_rows(unsigned int rows, unsigned int cols, const double* x, double* out) {
    __shared__ double values[BLOCK];
    __shared__ unsigned int columns[BLOCK];
    unsigned int r = blockIdx.x, t = threadIdx.x;
    if (r >= rows) return;
    const double* z = x + (u64)r * cols;
    double best = NEG_INF;
    unsigned int at = cols;
    for (unsigned int c = t; c < cols; c += BLOCK) if (z[c] > best) { best = z[c]; at = c; }
    values[t] = best;
    columns[t] = at;
    __syncthreads();
    for (unsigned int s = BLOCK / 2; s > 0; s >>= 1) {
        if (t < s && (values[t + s] > values[t] || (values[t + s] == values[t] && columns[t + s] < columns[t]))) {
            values[t] = values[t + s];
            columns[t] = columns[t + s];
        }
        __syncthreads();
    }
    if (t == 0) out[r] = columns[0] == cols ? 0.0 : (double)columns[0];
}

extern "C" __global__ void fill_entries(u64 n, const unsigned int* at, double value, double* x) {
    GRID_STRIDE(i, n) x[at[i]] = value;
}

extern "C" __global__ void softmax_quadratic(unsigned int rows, unsigned int cols, const double* logits, const double* tangent, double* out) {
    __shared__ double shared[BLOCK];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    const double* z = logits + (u64)r * cols;
    const double* t = tangent + (u64)r * cols;
    double m, total;
    softmax_stats(z, cols, shared, &m, &total);
    double partial = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) partial += exp(z[c] - m) / total * t[c];
    double mean = block_sum(partial, shared);
    partial = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) {
        double q = exp(z[c] - m) / total;
        partial += q * (t[c] - mean) * (t[c] - mean);
    }
    double q = block_sum(partial, shared);
    if (threadIdx.x == 0) out[r] = q;
}

// The standard normal draw `index` of (key, stream): Philox4x32-10, then Box–Muller in float
// (`posterior_normal` on the host).
__device__ float posterior_normal(u64 key, u64 stream, u64 index) {
    unsigned int c0 = (unsigned int)index, c1 = (unsigned int)(index >> 32), c2 = (unsigned int)stream, c3 = (unsigned int)(stream >> 32);
    unsigned int k0 = (unsigned int)key, k1 = (unsigned int)(key >> 32);
    for (int round = 0; round < 10; ++round) {
        if (round > 0) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
        unsigned int hi0 = __umulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
        unsigned int hi1 = __umulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
        c0 = hi1 ^ c1 ^ k0; c1 = lo1; c2 = hi0 ^ c3 ^ k1; c3 = lo0;
    }
    float u1 = ((float)c0 + 0.5f) * 2.3283064e-10f, u2 = (float)c1 * 2.3283064e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530717958647692f * u2);
}

// The math of an entry type: float entries in float, double in double.
__device__ float entry_exp(float x) { return expf(x); }
__device__ double entry_exp(double x) { return exp(x); }
__device__ float entry_sqrt(float x) { return sqrtf(x); }
__device__ double entry_sqrt(double x) { return sqrt(x); }

template <typename T>
__device__ void reparameterize_body(u64 n, u64 key, u64 stream, const T* mean, const T* log_sd, T* theta) {
    GRID_STRIDE(i, n) theta[i] = mean[i] + entry_exp(log_sd[i]) * (T)posterior_normal(key, stream, i);
}

extern "C" __global__ void reparameterize_f64(u64 n, u64 key, u64 stream, const double* mean, const double* log_sd, double* theta) {
    reparameterize_body<double>(n, key, stream, mean, log_sd, theta);
}

extern "C" __global__ void reparameterize_f32(u64 n, u64 key, u64 stream, const float* mean, const float* log_sd, float* theta) {
    reparameterize_body<float>(n, key, stream, mean, log_sd, theta);
}

// The bfloat16 nearest x (ties to even; a NaN stays a quiet NaN), as its 16 bits (`bf16_bits`).
__device__ unsigned short bf16_round(float x) {
    unsigned int bits = __float_as_uint(x);
    if (x != x) return (unsigned short)((bits >> 16) | 0x40u);
    return (unsigned short)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// The sample of f32 posterior entries written as bfloat16, the form products in Arithmetic::Bf16
// read without rounding it again.
extern "C" __global__ void reparameterize_bf16(u64 n, u64 key, u64 stream, const float* mean, const float* log_sd, unsigned short* theta) {
    GRID_STRIDE(i, n) theta[i] = bf16_round(mean[i] + expf(log_sd[i]) * posterior_normal(key, stream, i));
}

// Adds a live entry's (1, μ² + σ², 2s) to its group's row of `sums`: once per warp when every live
// lane's group is lane 0's (a row-major run of one group), else per lane. Every lane of the warp
// calls it.
__device__ void group_add(double* sums, unsigned int g, bool live, double a, double b, double c) {
    unsigned int g0 = __shfl_sync(0xffffffffu, g, 0);
    if (__all_sync(0xffffffffu, !live || g == g0)) {
        for (int o = 16; o > 0; o >>= 1) {
            a += __shfl_down_sync(0xffffffffu, a, o);
            b += __shfl_down_sync(0xffffffffu, b, o);
            c += __shfl_down_sync(0xffffffffu, c, o);
        }
        if ((threadIdx.x & 31u) == 0 && a > 0.0) {
            atomicAdd(sums + 3 * (u64)g0, a);
            atomicAdd(sums + 3 * (u64)g0 + 1, b);
            atomicAdd(sums + 3 * (u64)g0 + 2, c);
        }
    } else if (live) {
        atomicAdd(sums + 3 * (u64)g, a);
        atomicAdd(sums + 3 * (u64)g + 1, b);
        atomicAdd(sums + 3 * (u64)g + 2, c);
    }
}

// An entry whose group id is not below `count` is left alone (the caller's ids are checked where
// they are made). Blocks stride whole, so every lane of a warp runs every iteration (`group_add`).
#define WARP_STRIDE(i, n) for (u64 base_ = (u64)blockIdx.x * blockDim.x, i = base_ + threadIdx.x; base_ < (n); base_ += (u64)gridDim.x * blockDim.x, i = base_ + threadIdx.x)

// Entries (the posterior, its moments, the gradient) in T; group sums in double.
template <typename T>
__device__ void posterior_adam_body(u64 n, u64 count, u64 key, u64 stream, double scale, double mean_rate, double log_sd_rate, double beta1, double beta2, double epsilon,
    double c1, double c2, const T* gradient, const unsigned int* groups, const double* variance,
    T* mean, T* log_sd, T* mm, T* mv, T* sm, T* sv, double* sums) {
    const T b1 = (T)beta1, b2 = (T)beta2, o1 = (T)(1.0 - beta1), o2 = (T)(1.0 - beta2), k1 = (T)(1.0 / c1), k2 = (T)(1.0 / c2);
    const T rate_mean = (T)mean_rate, rate_log_sd = (T)log_sd_rate, eps = (T)epsilon, weight = (T)scale;
    WARP_STRIDE(i, n) {
        unsigned int g = i < n ? groups[i] : 0u;
        bool live = i < n && g < count && log_sd[i] != (T)NEG_INF;
        double a = 0.0, b = 0.0, c = 0.0;
        if (live) {
            T v = (T)variance[g], mu = mean[i], s = log_sd[i], sd = entry_exp(s);
            T e = (T)posterior_normal(key, stream, i), gi = weight * gradient[i];
            T gm = gi + mu / v, gs = gi * e * sd + sd * sd / v - (T)1;
            T m1 = b1 * mm[i] + o1 * gm, v1 = b2 * mv[i] + o2 * gm * gm;
            T m2 = b1 * sm[i] + o1 * gs, v2 = b2 * sv[i] + o2 * gs * gs;
            mm[i] = m1; mv[i] = v1; sm[i] = m2; sv[i] = v2;
            mu -= rate_mean * (m1 * k1) / (entry_sqrt(v1 * k2) + eps);
            s -= rate_log_sd * (m2 * k1) / (entry_sqrt(v2 * k2) + eps);
            mean[i] = mu; log_sd[i] = s;
            a = 1.0; b = (double)mu * (double)mu + exp(2.0 * (double)s); c = 2.0 * (double)s;
        }
        group_add(sums, g, live, a, b, c);
    }
}

extern "C" __global__ void posterior_adam_f64(u64 n, u64 count, u64 key, u64 stream, double scale, double mean_rate, double log_sd_rate, double beta1, double beta2, double epsilon,
    double c1, double c2, const double* gradient, const unsigned int* groups, const double* variance,
    double* mean, double* log_sd, double* mm, double* mv, double* sm, double* sv, double* sums) {
    posterior_adam_body<double>(n, count, key, stream, scale, mean_rate, log_sd_rate, beta1, beta2, epsilon, c1, c2, gradient, groups, variance, mean, log_sd, mm, mv, sm, sv, sums);
}

extern "C" __global__ void posterior_adam_f32(u64 n, u64 count, u64 key, u64 stream, double scale, double mean_rate, double log_sd_rate, double beta1, double beta2, double epsilon,
    double c1, double c2, const float* gradient, const unsigned int* groups, const double* variance,
    float* mean, float* log_sd, float* mm, float* mv, float* sm, float* sv, double* sums) {
    posterior_adam_body<float>(n, count, key, stream, scale, mean_rate, log_sd_rate, beta1, beta2, epsilon, c1, c2, gradient, groups, variance, mean, log_sd, mm, mv, sm, sv, sums);
}

// A bfloat16's value (`bf16_round` is its inverse to nearest).
__device__ float bf16_value(unsigned short h) { return __uint_as_float(((unsigned int)h) << 16); }
__device__ float entry_load(float x) { return x; }
__device__ float entry_load(unsigned short h) { return bf16_value(h); }
__device__ void entry_store(float* p, float x) { *p = x; }
__device__ void entry_store(unsigned short* p, float x) { *p = bf16_round(x); }

extern "C" __global__ void widen_bf16(u64 n, const unsigned short* x, float* y) {
    GRID_STRIDE(i, n) y[i] = bf16_value(x[i]);
}

// `posterior_adam_body` with f32 masters, the gradient in G and the moments in M (f32 or
// bfloat16), every update computed in f32 from the loaded values and each moment rounded once as
// it is stored.
template <typename G, typename M>
__device__ void posterior_adam_mixed(u64 n, u64 count, u64 key, u64 stream, double scale, double mean_rate, double log_sd_rate, double beta1, double beta2, double epsilon,
    double c1, double c2, const G* gradient, const unsigned int* groups, const double* variance,
    float* mean, float* log_sd, M* mm, M* mv, M* sm, M* sv, double* sums) {
    const float b1 = (float)beta1, b2 = (float)beta2, o1 = (float)(1.0 - beta1), o2 = (float)(1.0 - beta2), k1 = (float)(1.0 / c1), k2 = (float)(1.0 / c2);
    const float rate_mean = (float)mean_rate, rate_log_sd = (float)log_sd_rate, eps = (float)epsilon, weight = (float)scale;
    WARP_STRIDE(i, n) {
        unsigned int g = i < n ? groups[i] : 0u;
        bool live = i < n && g < count && log_sd[i] != (float)NEG_INF;
        double a = 0.0, b = 0.0, c = 0.0;
        if (live) {
            float v = (float)variance[g], mu = mean[i], s = log_sd[i], sd = expf(s);
            float e = posterior_normal(key, stream, i), gi = weight * entry_load(gradient[i]);
            float gm = gi + mu / v, gs = gi * e * sd + sd * sd / v - 1.0f;
            float m1 = b1 * entry_load(mm[i]) + o1 * gm, v1 = b2 * entry_load(mv[i]) + o2 * gm * gm;
            float m2 = b1 * entry_load(sm[i]) + o1 * gs, v2 = b2 * entry_load(sv[i]) + o2 * gs * gs;
            entry_store(mm + i, m1); entry_store(mv + i, v1); entry_store(sm + i, m2); entry_store(sv + i, v2);
            mu -= rate_mean * (m1 * k1) / (sqrtf(v1 * k2) + eps);
            s -= rate_log_sd * (m2 * k1) / (sqrtf(v2 * k2) + eps);
            mean[i] = mu; log_sd[i] = s;
            a = 1.0; b = (double)mu * (double)mu + exp(2.0 * (double)s); c = 2.0 * (double)s;
        }
        group_add(sums, g, live, a, b, c);
    }
}

extern "C" __global__ void posterior_adam_f32_bf16(u64 n, u64 count, u64 key, u64 stream, double scale, double mean_rate, double log_sd_rate, double beta1, double beta2, double epsilon,
    double c1, double c2, const float* gradient, const unsigned int* groups, const double* variance,
    float* mean, float* log_sd, unsigned short* mm, unsigned short* mv, unsigned short* sm, unsigned short* sv, double* sums) {
    posterior_adam_mixed<float, unsigned short>(n, count, key, stream, scale, mean_rate, log_sd_rate, beta1, beta2, epsilon, c1, c2, gradient, groups, variance, mean, log_sd, mm, mv, sm, sv, sums);
}

extern "C" __global__ void posterior_adam_bf16_bf16(u64 n, u64 count, u64 key, u64 stream, double scale, double mean_rate, double log_sd_rate, double beta1, double beta2, double epsilon,
    double c1, double c2, const unsigned short* gradient, const unsigned int* groups, const double* variance,
    float* mean, float* log_sd, unsigned short* mm, unsigned short* mv, unsigned short* sm, unsigned short* sv, double* sums) {
    posterior_adam_mixed<unsigned short, unsigned short>(n, count, key, stream, scale, mean_rate, log_sd_rate, beta1, beta2, epsilon, c1, c2, gradient, groups, variance, mean, log_sd, mm, mv, sm, sv, sums);
}

template <typename T>
__device__ void group_moments_body(u64 n, u64 count, const T* mean, const T* log_sd, const unsigned int* groups, double* sums) {
    WARP_STRIDE(i, n) {
        unsigned int g = i < n ? groups[i] : 0u;
        bool live = i < n && g < count && log_sd[i] != (T)NEG_INF;
        double a = 0.0, b = 0.0, c = 0.0;
        if (live) {
            double mu = (double)mean[i], s = (double)log_sd[i];
            a = 1.0; b = mu * mu + exp(2.0 * s); c = 2.0 * s;
        }
        group_add(sums, g, live, a, b, c);
    }
}

extern "C" __global__ void group_moments_f64(u64 n, u64 count, const double* mean, const double* log_sd, const unsigned int* groups, double* sums) {
    group_moments_body<double>(n, count, mean, log_sd, groups, sums);
}

extern "C" __global__ void group_moments_f32(u64 n, u64 count, const float* mean, const float* log_sd, const unsigned int* groups, double* sums) {
    group_moments_body<float>(n, count, mean, log_sd, groups, sums);
}

extern "C" __global__ void group_divergence(u64 n, double* sums, double* variance, double* divergence) {
    GRID_STRIDE(g, n) {
        double count = sums[3 * g], second = sums[3 * g + 1], log_variance = sums[3 * g + 2];
        double v = count > 0.0 ? second / count : 0.0;
        variance[g] = v;
        divergence[g] = count > 0.0 ? 0.5 * (count * log(v) - log_variance) : 0.0;
        sums[3 * g] = 0.0; sums[3 * g + 1] = 0.0; sums[3 * g + 2] = 0.0;
    }
}
"#;

    /// The sparse code (`Device::code_rows`), one block per row: every scalar decision taken by
    /// every thread alike from the same memory (a barrier before any write it reads), every
    /// vector update spread over the block, each in the CPU coder's order of operations.
    const CODER_KERNELS: &str = r#"
#define BLOCK 256
typedef unsigned long long u64;
#define POS_INF __longlong_as_double(0x7ff0000000000000LL)
#define NONE 0xffffffffu

__device__ double coder_sum(double v, double* shared) {
    unsigned int t = threadIdx.x;
    shared[t] = v;
    __syncthreads();
    for (unsigned int s = BLOCK / 2; s > 0; s >>= 1) {
        if (t < s) shared[t] += shared[t + s];
        __syncthreads();
    }
    double r = shared[0];
    __syncthreads();
    return r;
}

// The (value, index) pair first by `better`: smaller value then smaller index (`first_min`), or
// larger value then larger index (`last_max`). NONE indices lose.
__device__ unsigned int coder_pick(double v, unsigned int i, int last_max, double* sv, unsigned int* si) {
    unsigned int t = threadIdx.x;
    sv[t] = v; si[t] = i;
    __syncthreads();
    for (unsigned int s = BLOCK / 2; s > 0; s >>= 1) {
        if (t < s) {
            double a = sv[t], b = sv[t + s];
            unsigned int ia = si[t], ib = si[t + s];
            int take;
            if (ia == NONE) take = 1;
            else if (ib == NONE) take = 0;
            else if (last_max) take = b > a || (b == a && ib > ia);
            else take = b < a || (b == a && ib < ia);
            if (take) { sv[t] = b; si[t] = ib; }
        }
        __syncthreads();
    }
    unsigned int r = si[0];
    __syncthreads();
    return r;
}

struct Row {
    unsigned int C, B, W;
    int unit, fused;
    double kappa, tol, empty;
    const double* z; const double* w; const double* K; const unsigned int* starts; const double* bits;
    double *lin, *dia, *qm, *oq, *m, *col, *t, *on, *best, *inw, *key;
    unsigned int *work, *idx, *valid;
    double* cache; u64* counts;
};

// `col ← Q e_b` (every thread; ends synchronized).
__device__ void coder_column(const Row* r, unsigned int b) {
    if (r->cache && r->valid[b]) {
        if (threadIdx.x == 0 && r->counts) r->counts[3]++;
        for (unsigned int j = threadIdx.x; j < r->B; j += BLOCK) r->col[j] = r->cache[(u64)b * r->B + j];
        __syncthreads();
        return;
    }
    if (threadIdx.x == 0 && r->counts) r->counts[2]++;
    if (r->unit) {
        double zb = r->z[b];
        const double* kr = r->K + (u64)b * r->C;
        for (unsigned int j = threadIdx.x; j < r->B; j += BLOCK) r->col[j] = r->z[j] * (zb * kr[j]);
        __syncthreads();
    } else {
    unsigned int s = r->starts[b], e = r->starts[b + 1];
    for (unsigned int j = threadIdx.x; j < r->C; j += BLOCK) {
        double acc = 0.0;
        for (unsigned int c = s; c < e; c++) {
            double zc = r->z[c];
            if (zc != 0.0) acc += zc * r->K[(u64)c * r->C + j];
        }
        r->t[j] = acc;
    }
    __syncthreads();
    for (unsigned int j = threadIdx.x; j < r->B; j += BLOCK) {
        double acc = 0.0;
        for (unsigned int c = r->starts[j]; c < r->starts[j + 1]; c++) acc += r->z[c] * r->t[c];
        r->col[j] = acc;
    }
    __syncthreads();
    }
    if (r->cache) {
        for (unsigned int j = threadIdx.x; j < r->B; j += BLOCK) r->cache[(u64)b * r->B + j] = r->col[j];
        __syncthreads();
        if (threadIdx.x == 0) r->valid[b] = 1;
        __syncthreads();
    }
}

// `q += a col` (every thread; ends synchronized).
__device__ void coder_add(const Row* r, double* q, double a) {
    for (unsigned int j = threadIdx.x; j < r->B; j += BLOCK) q[j] += a * r->col[j];
    __syncthreads();
}

// Opt-in application without the intermediate column store/copy. The rank>1
// summation order is unchanged. Cache publication remains block synchronized.
__device__ void coder_apply(const Row* r, unsigned int b, double* q, double a) {
    if (!r->fused) { coder_column(r, b); coder_add(r, q, a); return; }
    if (r->cache && r->valid[b]) {
        if (threadIdx.x == 0 && r->counts) r->counts[3]++;
        for (unsigned int j = threadIdx.x; j < r->B; j += BLOCK) q[j] += a * r->cache[(u64)b * r->B + j];
        __syncthreads();
        return;
    }
    if (threadIdx.x == 0 && r->counts) r->counts[2]++;
    if (!r->unit) {
        unsigned int s = r->starts[b], e = r->starts[b + 1];
        for (unsigned int j = threadIdx.x; j < r->C; j += BLOCK) {
            double acc = 0.0;
            for (unsigned int c = s; c < e; c++) {
                double zc = r->z[c];
                if (zc != 0.0) acc += zc * r->K[(u64)c * r->C + j];
            }
            r->t[j] = acc;
        }
        __syncthreads();
    }
    for (unsigned int j = threadIdx.x; j < r->B; j += BLOCK) {
        double value;
        if (r->unit) value = r->z[j] * (r->z[b] * r->K[(u64)b * r->C + j]);
        else {
            value = 0.0;
            for (unsigned int c = r->starts[j]; c < r->starts[j + 1]; c++) value += r->z[c] * r->t[c];
        }
        if (r->cache) r->cache[(u64)b * r->B + j] = value;
        q[j] += a * value;
    }
    __syncthreads();
    if (r->cache) {
        if (threadIdx.x == 0) r->valid[b] = 1;
        __syncthreads();
    }
}

// The relaxation's optimum on the box [lo, hi] from `start` into `m` (and `qm = Q m`), with
// `known = Q start` when it is (then only the coordinates the box moves are added): its certified
// lower bound.
__device__ double coder_relax(const Row* r, const double* lo, const double* hi, const double* start, const double* known, double* sv, unsigned int* si, unsigned int* nw) {
    unsigned int B = r->B;
    if (threadIdx.x == 0 && r->counts) r->counts[0]++;
    for (unsigned int b = threadIdx.x; b < B; b += BLOCK) {
        r->m[b] = fmin(fmax(start[b], lo[b]), hi[b]);
        r->qm[b] = known ? known[b] : 0.0;
        r->inw[b] = 0.0;
    }
    __syncthreads();
    for (unsigned int b = 0; b < B; b++) {
        double moved = known ? r->m[b] - start[b] : r->m[b];
        if (moved != 0.0) { coder_apply(r, b, r->qm, moved); }
    }
    if (threadIdx.x == 0) {
        unsigned int n = 0;
        for (unsigned int b = 0; b < B; b++) {
            if (r->m[b] != 0.0 && lo[b] < hi[b]) { r->work[n++] = b; r->inw[b] = 1.0; }
        }
        *nw = n;
    }
    __syncthreads();
    for (;;) {
        for (int sweep = 0; sweep < 1000; sweep++) {
            if (threadIdx.x == 0 && r->counts) r->counts[1]++;
            double moved = 0.0;
            unsigned int n = *nw;
            for (unsigned int k = 0; k < n; k++) {
                unsigned int b = r->work[k];
                double g = r->lin[b] + 2.0 * r->kappa * r->qm[b];
                double curvature = 2.0 * r->kappa * r->dia[b];
                double mb = r->m[b];
                double next = curvature > 0.0 ? fmin(fmax(mb - g / curvature, lo[b]), hi[b]) : (g > 0.0 ? lo[b] : (g < 0.0 ? hi[b] : mb));
                double step = next - mb;
                __syncthreads();
                if (step != 0.0) {
                    if (threadIdx.x == 0) r->m[b] = next;
                    coder_apply(r, b, r->qm, step);
                    moved = fmax(moved, fabs(step));
                }
            }
            if (moved <= r->tol) break;
            if (sweep == 999 && threadIdx.x == 0 && r->counts) r->counts[6]++;
        }
        // The coordinates outside the set that violate a KKT condition, most violating first.
        for (unsigned int b = threadIdx.x; b < r->W; b += BLOCK) {
            double k = -1.0;
            if (b < B && lo[b] < hi[b] && r->inw[b] == 0.0) {
                double g = r->lin[b] + 2.0 * r->kappa * r->qm[b];
                if ((g < 0.0 && r->m[b] < hi[b]) || (g > 0.0 && r->m[b] > lo[b])) k = fabs(g);
            }
            r->key[b] = k;
            r->idx[b] = b;
        }
        __syncthreads();
        for (unsigned int k = 2; k <= r->W; k <<= 1) {
            for (unsigned int j = k >> 1; j > 0; j >>= 1) {
                for (unsigned int i = threadIdx.x; i < r->W; i += BLOCK) {
                    unsigned int l = i ^ j;
                    if (l > i) {
                        // Descending keys, ties by ascending index.
                        int before_li = r->key[l] > r->key[i] || (r->key[l] == r->key[i] && r->idx[l] < r->idx[i]);
                        int before_il = r->key[i] > r->key[l] || (r->key[i] == r->key[l] && r->idx[i] < r->idx[l]);
                        int swap = (i & k) == 0 ? before_li : before_il;
                        if (swap) {
                            double tk = r->key[i]; r->key[i] = r->key[l]; r->key[l] = tk;
                            unsigned int ti = r->idx[i]; r->idx[i] = r->idx[l]; r->idx[l] = ti;
                        }
                    }
                }
                __syncthreads();
            }
        }
        int stop = 0;
        if (threadIdx.x == 0) {
            unsigned int count = 0;
            while (count < B && r->key[count] >= 0.0) count++;
            if (count == 0) {
                stop = 1;
            } else {
                unsigned int n = *nw;
                unsigned int room = n > 16 ? n : 16;
                unsigned int take = count < room ? count : room;
                for (unsigned int i = 0; i < take; i++) { unsigned int b = r->idx[i]; r->work[n++] = b; r->inw[b] = 1.0; }
                *nw = n;
            }
            si[0] = stop;
        }
        __syncthreads();
        stop = si[0];
        __syncthreads();
        if (stop) break;
    }
    double pv = 0.0, pq = 0.0, ps = 0.0;
    for (unsigned int b = threadIdx.x; b < B; b += BLOCK) {
        double mb = r->m[b], g = r->lin[b] + 2.0 * r->kappa * r->qm[b];
        pv += mb * r->lin[b];
        pq += mb * r->qm[b];
        ps += fmin(g * (lo[b] - mb), g * (hi[b] - mb));
    }
    double value = r->empty + coder_sum(pv, sv) + r->kappa * coder_sum(pq, sv);
    double slack = coder_sum(ps, sv);
    return value + fmin(slack, 0.0);
}

// The relaxed point `m` rounded on [lo, hi] into `on` (with `oq = Q on`), then improved by exact
// single flips (the most saving first) until none saves: its code. The relaxation's `qm` stays.
__device__ double coder_round(const Row* r, const double* lo, const double* hi, double* sv, unsigned int* si) {
    unsigned int B = r->B;
    for (unsigned int b = threadIdx.x; b < B; b += BLOCK) {
        r->on[b] = (lo[b] == hi[b] ? hi[b] > 0.5 : r->m[b] > 0.5) ? 1.0 : 0.0;
        r->oq[b] = 0.0;
    }
    __syncthreads();
    for (unsigned int b = 0; b < B; b++) {
        if (r->on[b] == 1.0) { coder_apply(r, b, r->oq, 1.0); }
    }
    for (;;) {
        double bv = POS_INF;
        unsigned int bi = NONE;
        for (unsigned int b = threadIdx.x; b < B; b += BLOCK) {
            if (!(lo[b] < hi[b])) continue;
            double delta = r->on[b] == 1.0 ? -r->lin[b] + r->kappa * (r->dia[b] - 2.0 * r->oq[b]) : r->lin[b] + r->kappa * (r->dia[b] + 2.0 * r->oq[b]);
            if (delta < 0.0 && (bi == NONE || delta < bv)) { bv = delta; bi = b; }
        }
        unsigned int b = coder_pick(bv, bi, 0, sv, si);
        if (b == NONE) break;
        double sign = r->on[b] == 1.0 ? -1.0 : 1.0;
        __syncthreads();
        if (threadIdx.x == 0) {
            r->on[b] = r->on[b] == 1.0 ? 0.0 : 1.0;
            if (r->counts) r->counts[4]++;
        }
        coder_apply(r, b, r->oq, sign);
    }
    double pv = 0.0, pq = 0.0;
    for (unsigned int b = threadIdx.x; b < B; b += BLOCK) {
        pv += r->on[b] * r->lin[b];
        pq += r->on[b] * r->oq[b];
    }
    return r->empty + coder_sum(pv, sv) + r->kappa * coder_sum(pq, sv);
}

__device__ void coder_copy(double* to, const double* from, unsigned int n) {
    for (unsigned int i = threadIdx.x; i < n; i += BLOCK) to[i] = from[i];
    __syncthreads();
}

extern "C" __global__ void code_rows(unsigned int rows, unsigned int C, unsigned int B, unsigned int W, unsigned int nodes, unsigned int cap, int unit, int warmed,
    double kappa, double tol,
    const double* z, const double* w, const double* yfy, const double* K, const unsigned int* starts, const double* bits, const double* warm,
    u64 slot_len, u64 index_len, u64 cache_offset, int cached, int instrumented, int fused, double* scratch, unsigned int* iscratch, u64* counts, double* on_out, double* upper_out, double* lower_out) {
    __shared__ double sv[BLOCK];
    __shared__ unsigned int si[BLOCK];
    __shared__ unsigned int nw;
    double* base = scratch + (u64)blockIdx.x * slot_len;
    unsigned int* ibase = iscratch + (u64)blockIdx.x * index_len;
    Row r;
    r.C = C; r.B = B; r.W = W; r.unit = unit; r.fused = fused; r.kappa = kappa; r.tol = tol;
    r.K = K; r.starts = starts; r.bits = bits;
    r.lin = base; r.dia = base + B; r.qm = base + 2 * B; r.oq = base + 3 * B; r.m = base + 4 * B; r.col = base + 5 * B; r.on = base + 6 * B;
    r.best = base + 7 * B; r.inw = base + 8 * B;
    double* lo = base + 9 * B; double* hi = base + 10 * B; double* point = base + 11 * B; double* pqm = base + 12 * B;
    double* clo = base + 13 * B; double* chi = base + 14 * B;
    r.t = base + 15 * B; r.key = base + 15 * B + C;
    // Each open node: its box, its relaxed point and that point's `Q m`; then their lower bounds.
    double* open = base + 15 * B + C + W;
    double* open_lower = open + (u64)cap * 4 * B;
    r.work = ibase; r.idx = ibase + B;
    r.valid = cached ? ibase + B + W : 0;
    r.cache = cached ? base + cache_offset : 0;
    for (unsigned int row = blockIdx.x; row < rows; row += gridDim.x) {
        r.counts = instrumented ? counts + (u64)row * 7 : 0;
        if (threadIdx.x == 0 && r.counts) for (int i = 0; i < 7; i++) r.counts[i] = 0;
        if (r.cache) for (unsigned int b = threadIdx.x; b < B; b += BLOCK) r.valid[b] = 0;
        __syncthreads();
        r.z = z + (u64)row * C;
        r.w = w + (u64)row * C;
        r.empty = kappa * yfy[row];
        for (unsigned int b = threadIdx.x; b < B; b += BLOCK) {
            unsigned int s = starts[b], e = starts[b + 1];
            double acc = 0.0;
            for (unsigned int c = s; c < e; c++) acc += r.z[c] * r.w[c];
            r.lin[b] = bits[b] - 2.0 * kappa * acc;
            double d = 0.0;
            for (unsigned int c = s; c < e; c++) {
                double inner = 0.0;
                for (unsigned int c2 = s; c2 < e; c2++) inner += r.z[c] * r.z[c2] * K[(u64)c * C + c2];
                d += inner;
            }
            r.dia[b] = fmax(d, 0.0);
            lo[b] = 0.0;
            hi[b] = 1.0;
            point[b] = warmed ? warm[(u64)row * B + b] : 0.0;
        }
        __syncthreads();
        double root = coder_relax(&r, lo, hi, point, 0, sv, si, &nw);
        double upper = coder_round(&r, lo, hi, sv, si);
        coder_copy(r.best, r.on, B);
        double lower = fmin(root, upper);
        if (!(upper - root <= 1.0 || nodes == 0)) {
            // Best bound first; each open node its box, its relaxed point and its lower bound.
            unsigned int count = 1, explored = 0;
            coder_copy(open, lo, B); coder_copy(open + B, hi, B); coder_copy(open + 2 * B, r.m, B); coder_copy(open + 3 * B, r.qm, B);
            if (threadIdx.x == 0) open_lower[0] = root;
            __syncthreads();
            double floor = POS_INF;
            for (;;) {
                if (count == 0) break;
                unsigned int index = 0;
                for (unsigned int i = 1; i < count; i++) if (open_lower[i] < open_lower[index]) index = i;
                double node = open_lower[index];
                if (node >= upper - 1.0 || explored >= nodes) break;
                double* entry = open + (u64)index * 4 * B;
                coder_copy(lo, entry, B); coder_copy(hi, entry + B, B); coder_copy(point, entry + 2 * B, B); coder_copy(pqm, entry + 3 * B, B);
                count--;
                if (index != count) {
                    double* last = open + (u64)count * 4 * B;
                    coder_copy(entry, last, 4 * B);
                    if (threadIdx.x == 0) open_lower[index] = open_lower[count];
                    __syncthreads();
                }
                explored++;
                if (threadIdx.x == 0 && r.counts) r.counts[5]++;
                // The most fractional free block, the last of equals.
                double bv = -POS_INF;
                unsigned int bi = NONE;
                for (unsigned int b = threadIdx.x; b < B; b += BLOCK) {
                    if (!(lo[b] < hi[b])) continue;
                    double f = 0.5 - fabs(point[b] - 0.5);
                    if (bi == NONE || f >= bv) { bv = f; bi = b; }
                }
                unsigned int j = coder_pick(bv, bi, 1, sv, si);
                if (j == NONE || point[j] == 0.0 || point[j] == 1.0) { floor = fmin(floor, node); continue; }
                for (int fixed = 0; fixed < 2; fixed++) {
                    coder_copy(clo, lo, B); coder_copy(chi, hi, B);
                    if (threadIdx.x == 0) { clo[j] = (double)fixed; chi[j] = (double)fixed; }
                    __syncthreads();
                    double child = coder_relax(&r, clo, chi, point, pqm, sv, si, &nw);
                    double value = coder_round(&r, clo, chi, sv, si);
                    if (value < upper) { coder_copy(r.best, r.on, B); upper = value; }
                    if (child < upper - 1.0) {
                        double* slot = open + (u64)count * 4 * B;
                        coder_copy(slot, clo, B); coder_copy(slot + B, chi, B); coder_copy(slot + 2 * B, r.m, B); coder_copy(slot + 3 * B, r.qm, B);
                        if (threadIdx.x == 0) open_lower[count] = child;
                        __syncthreads();
                        count++;
                    } else {
                        floor = fmin(floor, child);
                    }
                }
            }
            lower = floor;
            for (unsigned int i = 0; i < count; i++) lower = fmin(lower, open_lower[i]);
            lower = fmin(lower, upper);
        }
        for (unsigned int b = threadIdx.x; b < B; b += BLOCK) on_out[(u64)row * B + b] = r.best[b];
        if (threadIdx.x == 0) { upper_out[row] = upper; lower_out[row] = lower; }
        __syncthreads();
    }
}
"#;

    /// The f32 twins of [`KERNELS`] (module note), one name and parameter list per kernel.
    const KERNELS_F32: &str = include_str!("tensor_f32.cu");

    /// One CUDA device's stream, cuBLAS handle and kernels.
    pub(super) struct Engine {
        pub(super) name: String,
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        blas: CudaBlas,
        module: Arc<CudaModule>,
        /// The f32 twins, compiled on first use, and the functions loaded from them.
        module32: std::sync::OnceLock<Arc<CudaModule>>,
        functions32: std::sync::Mutex<HashMap<&'static str, CudaFunction>>,
        checked_interval_module: crate::device_cache::PtxModuleCache,
        /// The row flags of a call that scores every row (never read).
        every_row: CudaSlice<u32>,
        gemm_workspace: std::sync::Mutex<F32Workspace>,
        /// Whether the stream is being captured into a graph.
        capturing: AtomicBool,
        /// cuBLAS's workspace inside captures (a recorded product may not allocate), kept for the
        /// engine's life since every graph's products read it.
        capture_workspace: std::sync::Mutex<Option<CudaSlice<u8>>>,
    }

    /// The cuBLAS workspace captured products use.
    const CAPTURE_WORKSPACE: usize = 32 << 20;

    #[derive(Default)]
    struct F32Workspace {
        left: Option<CudaSlice<f32>>,
        right: Option<CudaSlice<f32>>,
        output: Option<CudaSlice<f32>>,
    }

    fn cfg_elements(n: u64) -> LaunchConfig {
        let blocks = n.div_ceil(u64::from(BLOCK)).clamp(1, 65_535 * 8) as u32;
        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 }
    }

    fn cfg_rows(rows: usize) -> LaunchConfig {
        LaunchConfig { grid_dim: (rows.max(1) as u32, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 }
    }

    fn slice(t: &Tensor) -> Result<&CudaSlice<f64>, GpuError> {
        match &t.data {
            Data::Cuda(s) => Ok(s),
            other => Err(mismatch(other)),
        }
    }

    fn slice_mut(t: &mut Tensor) -> Result<&mut CudaSlice<f64>, GpuError> {
        match &mut t.data {
            Data::Cuda(s) => Ok(s),
            other => Err(mismatch(other)),
        }
    }

    fn slice32(t: &Tensor) -> Result<&CudaSlice<f32>, GpuError> {
        match &t.data {
            Data::Cuda32(s) => Ok(s),
            other => Err(mismatch(other)),
        }
    }

    fn slice32_mut(t: &mut Tensor) -> Result<&mut CudaSlice<f32>, GpuError> {
        match &mut t.data {
            Data::Cuda32(s) => Ok(s),
            other => Err(mismatch(other)),
        }
    }

    /// A buffer that is not the operation's: the host's, or the other storage's.
    fn mismatch(data: &Data) -> GpuError {
        match data {
            Data::Host(_) => foreign(),
            _ => shape("operands in different storage (Device::convert moves one)".to_string()),
        }
    }

    /// Tensors as kernel arguments in the storage the operation runs in (an operand in the other
    /// is an error, never a conversion).
    trait Operand<'a> {
        fn input(&mut self, t: &'a Tensor, storage: Storage) -> Result<&mut Self, GpuError>;
        fn output(&mut self, t: &'a mut Tensor, storage: Storage) -> Result<&mut Self, GpuError>;
    }

    impl<'a> Operand<'a> for LaunchArgs<'a> {
        fn input(&mut self, t: &'a Tensor, storage: Storage) -> Result<&mut Self, GpuError> {
            match (&t.data, storage) {
                (Data::Cuda(s), Storage::F64) => Ok(self.arg(s)),
                (Data::Cuda32(s), Storage::F32) => Ok(self.arg(s)),
                (Data::CudaBf16(s), Storage::Bf16) => Ok(self.arg(s)),
                (other, _) => Err(mismatch(other)),
            }
        }

        fn output(&mut self, t: &'a mut Tensor, storage: Storage) -> Result<&mut Self, GpuError> {
            match (&mut t.data, storage) {
                (Data::Cuda(s), Storage::F64) => Ok(self.arg(s)),
                (Data::Cuda32(s), Storage::F32) => Ok(self.arg(s)),
                (Data::CudaBf16(s), Storage::Bf16) => Ok(self.arg(s)),
                (other, _) => Err(mismatch(other)),
            }
        }
    }

    fn index_slice(i: &Indices) -> Result<&CudaSlice<u32>, GpuError> {
        match &i.data {
            IndexData::Cuda(s) => Ok(s),
            IndexData::Host(_) => Err(foreign()),
        }
    }

    fn i32_of(n: usize) -> Result<i32, GpuError> {
        i32::try_from(n).map_err(|_| shape(format!("{n} exceeds cuBLAS's 32-bit dimensions")))
    }

    fn op_of(op: Op) -> cublasOperation_t {
        match op {
            Op::N => cublasOperation_t::CUBLAS_OP_N,
            Op::T => cublasOperation_t::CUBLAS_OP_T,
        }
    }

    /// A product operand's buffer: f32, or bfloat16 bits.
    #[derive(Clone, Copy)]
    enum Factor<'a> {
        Single(&'a CudaSlice<f32>),
        Half(&'a CudaSlice<u16>),
    }

    /// One column-major `cublasGemmEx` into an f32 buffer: `C ← α op(A) op(B) + β C`, `C` m × n,
    /// the operands' element offsets and leading dimensions given, accumulating in f32, on the TF32
    /// tensor cores, or on bfloat16 operands (`compute` and the operands' types).
    struct Gemm32<'a> {
        ops: (cublasOperation_t, cublasOperation_t),
        dims: (usize, usize, usize),
        scale: (f32, f32),
        a: (Factor<'a>, usize, usize),
        b: (Factor<'a>, usize, usize),
        c: (&'a mut CudaSlice<f32>, usize, usize),
        compute: cublasComputeType_t,
    }

    /// An operand as the product reads it: its rounded temporary, its bfloat16 copy, or its f32.
    fn pick<'a>(t: &'a Tensor, rounded: &'a Option<CudaSlice<u16>>) -> Result<Factor<'a>, GpuError> {
        Ok(match (rounded, &t.data) {
            (Some(r), _) => Factor::Half(r),
            (None, Data::CudaBf16(h)) => Factor::Half(h),
            (None, _) => Factor::Single(slice32(t)?),
        })
    }

    /// An operand's pointer at its element offset, its type, and the record marking its read.
    fn pointer<'a>(f: Factor<'a>, offset: usize, stream: &'a CudaStream) -> (u64, cudaDataType_t, SyncOnDrop<'a>) {
        match f {
            Factor::Single(s) => {
                let (p, record) = s.device_ptr(stream);
                (p + 4 * offset as u64, cudaDataType_t::CUDA_R_32F, record)
            }
            Factor::Half(s) => {
                let (p, record) = s.device_ptr(stream);
                (p + 2 * offset as u64, cudaDataType_t::CUDA_R_16BF, record)
            }
        }
    }

    /// The accumulation of an f32-storage product, and whether its operands are bfloat16.
    fn compute_of(arithmetic: Arithmetic, name: &str) -> Result<(cublasComputeType_t, bool), GpuError> {
        match arithmetic {
            Arithmetic::F64 => Err(GpuError::NoDeviceKernel { reason: format!("{name} in f32 storage has no float64 product") }),
            Arithmetic::F32 => Ok((cublasComputeType_t::CUBLAS_COMPUTE_32F, false)),
            Arithmetic::Tf32 => Ok((cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32, false)),
            Arithmetic::Bf16 => Ok((cublasComputeType_t::CUBLAS_COMPUTE_32F, true)),
        }
    }

    impl Engine {
        pub(super) fn new(ordinal: usize, name: String) -> Result<Self, GpuError> {
            // The shared context settles the driver, binds the device and touches the runtime.
            crate::device_runtime::cuda_context_for(ordinal)
                .ok_or_else(|| GpuError::DriverCallFailed { reason: format!("no CUDA context for device {ordinal}") })?;
            // The engine's own handle on the device's primary context: every buffer it makes lives
            // on its one stream, so cudarc's cross-stream bookkeeping (two events made per buffer,
            // a wait and a record per kernel argument, a wait per free) is pure host cost, off on
            // this handle and only here. It also leaves nothing to keep out of a graph capture.
            let ctx = CudaContext::new(ordinal).gpu_ctx("tensor context")?;
            // SAFETY: this handle's buffers are used on the engine's stream alone.
            unsafe { ctx.disable_event_tracking() };
            let stream = ctx.new_stream().gpu_ctx("tensor stream")?;
            let blas = CudaBlas::new(stream.clone()).gpu_ctx("tensor cuBLAS handle")?;
            static MODULE: crate::device_cache::PtxModuleCache = crate::device_cache::PtxModuleCache::new();
            let module = Arc::clone(MODULE.get_or_compile(&ctx, "tensor", KERNELS)?);
            let every_row = stream.alloc_zeros::<u32>(1).gpu_ctx("tensor alloc")?;
            Ok(Self {
                name,
                ctx,
                stream,
                blas,
                module,
                module32: std::sync::OnceLock::new(),
                functions32: std::sync::Mutex::new(HashMap::new()),
                checked_interval_module: crate::device_cache::PtxModuleCache::new(),
                every_row,
                gemm_workspace: std::sync::Mutex::new(F32Workspace::default()),
                capturing: AtomicBool::new(false),
                capture_workspace: std::sync::Mutex::new(None),
            })
        }

        pub(super) fn memory(&self) -> Result<(usize, usize), GpuError> {
            self.ctx.mem_get_info().gpu_ctx("tensor memory info")
        }

        pub(super) fn synchronize(&self) -> Result<(), GpuError> {
            self.stream.synchronize().gpu_ctx("tensor synchronize")
        }

        /// Starts a capture (no event is waited on or recorded, the engine's handle tracking none);
        /// cuBLAS gets a workspace of its own, since a recorded product may not allocate one.
        pub(super) fn begin_capture(&self) -> Result<(), GpuError> {
            if self.capturing.load(Ordering::Acquire) {
                return Err(shape("a capture is already open".to_string()));
            }
            let mut workspace = self.capture_workspace.lock().map_err(|_| shape("poisoned capture workspace".to_string()))?;
            if workspace.is_none() {
                *workspace = Some(self.stream.alloc_zeros::<u8>(CAPTURE_WORKSPACE).gpu_ctx("tensor capture workspace")?);
            }
            let buffer = workspace.as_ref().ok_or_else(|| shape("missing capture workspace".to_string()))?;
            let (pointer, record) = buffer.device_ptr(&self.stream);
            drop(record);
            // SAFETY: the buffer lives as long as the engine and its 256-byte-aligned allocation.
            unsafe { cudarc::cublas::sys::cublasSetWorkspace_v2(*self.blas.handle(), pointer as *mut _, CAPTURE_WORKSPACE) }
                .result()
                .gpu_ctx("tensor capture cuBLAS workspace")?;
            self.capturing.store(true, Ordering::Release);
            // Relaxed: allocations (a temporary's) may be recorded; the stream is this thread's.
            let started = self.stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED).gpu_ctx("tensor graph capture");
            if started.is_err() {
                self.restore_after_capture()?;
            }
            started
        }

        pub(super) fn end_capture(&self) -> Result<CudaGraph, GpuError> {
            if !self.capturing.load(Ordering::Acquire) {
                return Err(shape("no capture is open".to_string()));
            }
            // No automatic freeing on relaunch: a temporary outliving the capture is an error at
            // the next launch, never a buffer freed under a live tensor. (Node priority is the
            // flag that changes nothing else.)
            let graph = self.stream.end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY);
            self.restore_after_capture()?;
            graph.gpu_ctx("tensor graph instantiate")?.ok_or_else(|| shape("an empty capture records no graph".to_string()))
        }

        /// cuBLAS back on its own workspace pool (resetting its stream does that).
        fn restore_after_capture(&self) -> Result<(), GpuError> {
            self.capturing.store(false, Ordering::Release);
            // SAFETY: the handle is this engine's, bound to this stream since its creation.
            unsafe { cudarc::cublas::result::set_stream(*self.blas.handle(), self.stream.cu_stream() as _) }.gpu_ctx("tensor cuBLAS workspace reset")
        }

        /// Refuses a host transfer while capturing: the graph would replay it against host memory
        /// long gone, and cudarc reports a read-back's synchronization failure nowhere.
        fn host_transfer(&self) -> Result<(), GpuError> {
            if self.capturing.load(Ordering::Acquire) {
                return Err(GpuError::NoDeviceKernel { reason: "a host transfer cannot be recorded in a graph".to_string() });
            }
            Ok(())
        }

        pub(super) fn upload<T: DeviceRepr + ValidAsZeroBits>(&self, values: &[T]) -> Result<CudaSlice<T>, GpuError> {
            self.host_transfer()?;
            // An empty tensor still holds one (unread) value: the driver allocates nothing smaller.
            if values.is_empty() {
                return self.stream.alloc_zeros::<T>(1).gpu_ctx("tensor alloc");
            }
            self.stream.clone_htod(values).gpu_ctx("tensor upload")
        }

        pub(super) fn download<T: DeviceRepr>(&self, slice: &CudaSlice<T>) -> Result<Vec<T>, GpuError> {
            self.host_transfer()?;
            self.stream.clone_dtoh(slice).gpu_ctx("tensor download")
        }

        pub(super) fn scaled_row_l2_enclosed(&self, values: &Tensor, columns: std::ops::Range<usize>, scale: [f64;3]) -> Result<Vec<[f64;3]>, GpuError> {
            if values.rows == 0 { return Ok(Vec::new()); }
            let mut out = self.zeros(values.rows.checked_mul(3).ok_or_else(||shape("row norm output size overflow".into()))?)?;
            let (rows, cols, begin, end) = (values.rows as u64, values.cols as u64, columns.start as u64, columns.end as u64);
            let f = self.function("scaled_row_l2")?;
            // SAFETY: validated input columns; each thread writes three scalars in its distinct output row.
            unsafe {
                self.stream.launch_builder(&f).arg(&rows).arg(&cols).arg(&begin).arg(&end).arg(&scale[0]).arg(&scale[1]).arg(&scale[2])
                    .arg(slice(values)?).arg(&mut out).launch(cfg_elements(rows))
            }.gpu_ctx("tensor scaled_row_l2 enclosed")?;
            Ok(self.download(&out)?.chunks_exact(3).map(|v|[v[0],v[1],v[2]]).collect())
        }

        pub(super) fn zeros(&self, n: usize) -> Result<CudaSlice<f64>, GpuError> {
            self.stream.alloc_zeros::<f64>(n.max(1)).gpu_ctx("tensor alloc")
        }

        pub(super) fn zeros32(&self, n: usize) -> Result<CudaSlice<f32>, GpuError> {
            self.stream.alloc_zeros::<f32>(n.max(1)).gpu_ctx("tensor alloc")
        }

        pub(super) fn zeros16(&self, n: usize) -> Result<CudaSlice<u16>, GpuError> {
            self.stream.alloc_zeros::<u16>(n.max(1)).gpu_ctx("tensor alloc")
        }

        /// A `rows × cols` tensor in `storage` for a kernel that writes every entry: f32 left
        /// unset (no zeroing pass ahead of the kernel), float64 zeroed as it always was.
        fn output(&self, storage: Storage, rows: usize, cols: usize) -> Result<Tensor, GpuError> {
            if storage != Storage::F32 {
                return self.tensor(storage, rows, cols);
            }
            // SAFETY: the caller's kernel writes all `rows · cols` values before any is read.
            let data = Data::Cuda32(unsafe { self.stream.alloc::<f32>((rows * cols).max(1)) }.gpu_ctx("tensor alloc")?);
            Ok(Tensor { rows, cols, data })
        }

        /// A zero `rows × cols` tensor in `storage`.
        fn tensor(&self, storage: Storage, rows: usize, cols: usize) -> Result<Tensor, GpuError> {
            let data = match storage {
                Storage::F64 => Data::Cuda(self.zeros(rows * cols)?),
                Storage::F32 => Data::Cuda32(self.zeros32(rows * cols)?),
                Storage::Bf16 => return Err(shape("operations make no bfloat16 tensors".to_string())),
            };
            Ok(Tensor { rows, cols, data })
        }

        pub(super) fn copy<T: DeviceRepr + ValidAsZeroBits>(&self, slice: &CudaSlice<T>) -> Result<CudaSlice<T>, GpuError> {
            let mut out = self.stream.alloc_zeros::<T>(slice.len().max(1)).gpu_ctx("tensor alloc")?;
            self.stream.memcpy_dtod(slice, &mut out).gpu_ctx("tensor copy")?;
            Ok(out)
        }

        pub(super) fn copy_range<T: DeviceRepr + ValidAsZeroBits>(&self, slice: &CudaSlice<T>, lo: usize, hi: usize) -> Result<CudaSlice<T>, GpuError> {
            let mut out = self.stream.alloc_zeros::<T>((hi - lo).max(1)).gpu_ctx("tensor alloc")?;
            if hi > lo {
                self.stream.memcpy_dtod(&slice.slice(lo..hi), &mut out).gpu_ctx("tensor row copy")?;
            }
            Ok(out)
        }

        pub(super) fn write_range<T>(&self, slice: &mut CudaSlice<T>, lo: usize, part: &CudaSlice<T>) -> Result<(), GpuError> {
            let n = part.len();
            self.stream.memcpy_dtod(part, &mut slice.slice_mut(lo..lo + n)).gpu_ctx("tensor row write")
        }

        /// `t` in the other storage, on the device: rounded to nearest to f32, exactly to float64.
        pub(super) fn convert(&self, t: &Tensor) -> Result<Tensor, GpuError> {
            let n = t.len() as u64;
            let data = match &t.data {
                Data::Cuda(source) => {
                    let mut out = self.zeros32(t.len())?;
                    if n > 0 {
                        let f = self.function("to_f32")?;
                        // SAFETY: `to_f32(n, x, y)` reads n doubles and writes n floats.
                        unsafe { self.stream.launch_builder(&f).arg(&n).arg(source).arg(&mut out).launch(cfg_elements(n)) }.gpu_ctx("tensor to_f32")?;
                    }
                    Data::Cuda32(out)
                }
                Data::Cuda32(source) => {
                    let mut out = self.zeros(t.len())?;
                    if n > 0 {
                        let f = self.function("to_f64")?;
                        // SAFETY: `to_f64(n, y, x)` reads n floats and writes n doubles.
                        unsafe { self.stream.launch_builder(&f).arg(&n).arg(source).arg(&mut out).launch(cfg_elements(n)) }.gpu_ctx("tensor to_f64")?;
                    }
                    Data::Cuda(out)
                }
                Data::CudaBf16(_) => return Err(shape("a bfloat16 copy converts no further".to_string())),
                Data::Host(_) => return Err(foreign()),
            };
            Ok(Tensor { rows: t.rows, cols: t.cols, data })
        }

        /// `t` in `storage`, on the device: rounded to nearest (ties to even) when narrowing,
        /// exactly when widening; bfloat16 passes through f32.
        pub(super) fn convert_to(&self, t: &Tensor, storage: Storage) -> Result<Tensor, GpuError> {
            match (t.storage(), storage) {
                (Storage::F64, Storage::F64) | (Storage::F32, Storage::F32) => Err(shape("a conversion into the storage it is in".to_string())),
                (Storage::F64, Storage::F32) | (Storage::F32, Storage::F64) => self.convert(t),
                (Storage::F32, Storage::Bf16) => self.bf16_copy(t),
                (Storage::F64, Storage::Bf16) => self.bf16_copy(&self.convert(t)?),
                (Storage::Bf16, to) => {
                    if to == Storage::Bf16 {
                        return Err(shape("a conversion into the storage it is in".to_string()));
                    }
                    let n = t.len() as u64;
                    let Data::CudaBf16(source) = &t.data else { return Err(foreign()) };
                    let mut out = self.zeros32(t.len())?;
                    if n > 0 {
                        let f = self.function("widen_bf16")?;
                        // SAFETY: `widen_bf16(n, x, y)` reads n halves and writes n floats.
                        unsafe { self.stream.launch_builder(&f).arg(&n).arg(source).arg(&mut out).launch(cfg_elements(n)) }.gpu_ctx("tensor widen_bf16")?;
                    }
                    let wide = Tensor { rows: t.rows, cols: t.cols, data: Data::Cuda32(out) };
                    if to == Storage::F32 { Ok(wide) } else { self.convert(&wide) }
                }
            }
        }

        fn function(&self, name: &str) -> Result<CudaFunction, GpuError> {
            self.module.load_function(name).gpu_ctx_with(|e| format!("tensor kernel {name}: {e}"))
        }

        /// `name` in `storage`: the float64 kernel, or its f32 twin (its module compiled on first use).
        fn kernel(&self, name: &'static str, storage: Storage) -> Result<CudaFunction, GpuError> {
            if storage == Storage::F64 {
                return self.function(name);
            }
            let mut loaded = self.functions32.lock().map_err(|_| shape("poisoned f32 kernel table".to_string()))?;
            if let Some(f) = loaded.get(name) {
                return Ok(f.clone());
            }
            let module = match self.module32.get() {
                Some(module) => module,
                None => {
                    // One module per device (each context loads its own).
                    static MODULES: std::sync::OnceLock<crate::device_cache::KeyedPtxModuleCache<usize>> = std::sync::OnceLock::new();
                    let compiled = MODULES
                        .get_or_init(crate::device_cache::KeyedPtxModuleCache::new)
                        .get_or_compile(&self.ctx, self.ctx.ordinal(), "tensor f32", |_| KERNELS_F32.to_string())?;
                    self.module32.get_or_init(|| compiled)
                }
            };
            let f = module.load_function(name).gpu_ctx_with(|e| format!("tensor f32 kernel {name}: {e}"))?;
            loaded.insert(name, f.clone());
            Ok(f)
        }

        pub(super) fn gemm(
            &self,
            batch: usize,
            (m, n, k): (usize, usize, usize),
            (alpha, beta): (f64, f64),
            (a, ta): (&Tensor, Op),
            (b, tb): (&Tensor, Op),
            c: &mut Tensor,
            arithmetic: Arithmetic,
        ) -> Result<(), GpuError> {
            if c.storage() == Storage::F32 {
                return self.gemm32(batch, (m, n, k), (alpha, beta), (a, ta), (b, tb), c, arithmetic);
            }
            if a.storage() != c.storage() || b.storage() != c.storage() {
                return Err(mismatch(&c.data));
            }
            if m == 0 || n == 0 {
                return Ok(());
            }
            // Serialize use of this handle's math mode and scratch buffers. Every operation is
            // queued on the same stream, so reuse requires no host synchronization.
            let mut cached = self.gemm_workspace.lock().map_err(|_| shape("poisoned GEMM workspace".to_string()))?;
            // Row-major C = op(A) op(B) is column-major Cᵀ = op(B)ᵀ op(A)ᵀ: the buffers swap places
            // and keep their own flags, each leading dimension its row length.
            let (dims, leading) = ((i32_of(n)?, i32_of(m)?, i32_of(k)?), (i32_of(b.cols)?, i32_of(a.cols)?, i32_of(c.cols)?));
            let gemm = |transa, transb| GemmConfig {
                transa,
                transb,
                m: dims.0,
                n: dims.1,
                k: dims.2,
                alpha,
                lda: leading.0,
                ldb: leading.1,
                beta,
                ldc: leading.2,
            };
            let per = |t: &Tensor| ((t.rows / batch) * t.cols) as i64;
            let (stride_a, stride_b, stride_c) = (per(b), per(a), per(c));
            if arithmetic != Arithmetic::F64 && self.capturing.load(Ordering::Acquire) {
                return Err(GpuError::NoDeviceKernel { reason: "a lowered float64-storage product is not capturable: capture in f32 storage".to_string() });
            }
            if arithmetic == Arithmetic::F64 {
                let cfg = gemm(op_of(tb), op_of(ta));
                let (bs, as_) = (slice(b)?, slice(a)?);
                let cs = slice_mut(c)?;
                // SAFETY: every operand is a live row-major buffer of `batch` equal blocks whose
                // shapes were checked against (m, n, k) by the caller.
                unsafe {
                    if batch == 1 {
                        self.blas.gemm(cfg, bs, as_, cs)
                    } else {
                        self.blas.gemm_strided_batched(
                            StridedBatchedConfig { gemm: cfg, batch_size: i32_of(batch)?, stride_a, stride_b, stride_c },
                            bs,
                            as_,
                            cs,
                        )
                    }
                }
                .gpu_ctx("tensor DGEMM")?;
                return Ok(());
            }
            let mut temporary = F32Workspace::default();
            // Avoid retaining an unbounded high-water allocation after an unusually large GEMM.
            let total = a.len().saturating_add(b.len()).saturating_add(c.len());
            let capacity = cached.left.as_ref().map_or(0, |s| s.len()).max(a.len())
                .saturating_add(cached.right.as_ref().map_or(0, |s| s.len()).max(b.len()))
                .saturating_add(cached.output.as_ref().map_or(0, |s| s.len()).max(c.len()));
            if capacity > 64 * 1024 * 1024 { *cached = F32Workspace::default(); }
            let workspace = if total <= 64 * 1024 * 1024 { &mut *cached } else { &mut temporary };
            let lower = |t: &Tensor, slot: &mut Option<CudaSlice<f32>>, convert: bool| -> Result<(), GpuError> {
                if slot.as_ref().is_none_or(|s| s.len() < t.len().max(1)) {
                    *slot = Some(self.stream.alloc_zeros::<f32>(t.len().max(1)).gpu_ctx("tensor f32 alloc")?);
                }
                if !convert || t.is_empty() { return Ok(()); }
                let out = slot.as_mut().ok_or_else(|| shape("missing GEMM scratch".to_string()))?;
                let n = t.len() as u64;
                let f = self.function("to_f32")?;
                // SAFETY: `to_f32(n, x, y)` reads n doubles and writes n floats.
                unsafe { self.stream.launch_builder(&f).arg(&n).arg(slice(t)?).arg(out).launch(cfg_elements(n)) }
                    .gpu_ctx("tensor to_f32")?;
                Ok(())
            };
            lower(a, &mut workspace.left, true)?;
            lower(b, &mut workspace.right, true)?;
            // beta=0 means the previous output is irrelevant, including stale NaNs.
            lower(c, &mut workspace.output, beta != 0.0)?;
            let a32 = workspace.left.as_ref().ok_or_else(|| shape("missing left scratch".to_string()))?;
            let b32 = workspace.right.as_ref().ok_or_else(|| shape("missing right scratch".to_string()))?;
            let c32 = workspace.output.as_mut().ok_or_else(|| shape("missing output scratch".to_string()))?;
            let f32_cfg = {
                let g = gemm(op_of(tb), op_of(ta));
                GemmConfig {
                    transa: g.transa,
                    transb: g.transb,
                    m: g.m,
                    n: g.n,
                    k: g.k,
                    alpha: alpha as f32,
                    lda: g.lda,
                    ldb: g.ldb,
                    beta: beta as f32,
                    ldc: g.ldc,
                }
            };
            let mode = if arithmetic == Arithmetic::Tf32 { cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH } else { cublasMath_t::CUBLAS_PEDANTIC_MATH };
            // SAFETY: the handle is this engine's; the mode is restored below.
            unsafe { cudarc::cublas::sys::cublasSetMathMode(*self.blas.handle(), mode) }.result().gpu_ctx("tensor math mode")?;
            // SAFETY: as for the float64 product, on the lowered copies.
            let product = unsafe {
                if batch == 1 {
                    self.blas.gemm(f32_cfg, b32, a32, c32)
                } else {
                    self.blas.gemm_strided_batched(
                        StridedBatchedConfig { gemm: f32_cfg, batch_size: i32_of(batch)?, stride_a, stride_b, stride_c },
                        b32,
                        a32,
                        c32,
                    )
                }
            };
            // SAFETY: restores the default (float64 products on the FP64 tensor cores).
            unsafe { cudarc::cublas::sys::cublasSetMathMode(*self.blas.handle(), cublasMath_t::CUBLAS_DEFAULT_MATH) }
                .result()
                .gpu_ctx("tensor math mode")?;
            product.gpu_ctx("tensor SGEMM")?;
            let n = c.len() as u64;
            let f = self.function("to_f64")?;
            let cs = slice_mut(c)?;
            // SAFETY: `to_f64(n, y, x)` reads n floats and writes n doubles.
            unsafe { self.stream.launch_builder(&f).arg(&n).arg(&*c32).arg(cs).launch(cfg_elements(n)) }.gpu_ctx("tensor to_f64")?;
            Ok(())
        }

        /// [`Engine::gemm`] on f32 tensors as they are, with nothing converted or copied: f32
        /// accumulation (`Arithmetic::F32`) or the TF32 tensor cores (`Tf32`); float64 is refused.
        fn gemm32(
            &self,
            batch: usize,
            (m, n, k): (usize, usize, usize),
            (alpha, beta): (f64, f64),
            (a, ta): (&Tensor, Op),
            (b, tb): (&Tensor, Op),
            c: &mut Tensor,
            arithmetic: Arithmetic,
        ) -> Result<(), GpuError> {
            if m == 0 || n == 0 {
                return Ok(());
            }
            let (compute, half) = compute_of(arithmetic, &self.name)?;
            let valid = |t: &Tensor| match &t.data {
                Data::Cuda32(_) => Ok(()),
                Data::CudaBf16(_) if half => Ok(()),
                Data::CudaBf16(_) => Err(shape(format!("a bfloat16 operand takes Arithmetic::Bf16, not {arithmetic:?}"))),
                other => Err(mismatch(other)),
            };
            valid(a)?;
            valid(b)?;
            // In bfloat16 an f32 operand is rounded into a temporary; a bfloat16 copy is read as is.
            let (rounded_a, rounded_b) = if half { (self.half_of(a)?, self.half_of(b)?) } else { (None, None) };
            // As the float64 product: column-major Cᵀ = op(B)ᵀ op(A)ᵀ, each block `batch`-strided.
            let per = |t: &Tensor| (t.rows / batch) * t.cols;
            let (stride_b, stride_a, stride_c) = (per(b), per(a), per(c));
            let ldc = c.cols;
            let (fb, fa) = (pick(b, &rounded_b)?, pick(a, &rounded_a)?);
            let product = Gemm32 {
                ops: (op_of(tb), op_of(ta)),
                dims: (n, m, k),
                scale: (alpha as f32, beta as f32),
                a: (fb, 0, b.cols),
                b: (fa, 0, a.cols),
                c: (slice32_mut(c)?, 0, ldc),
                compute,
            };
            self.gemm_ex(product, batch, (stride_b, stride_a, stride_c))
        }

        /// An f32 tensor rounded to bfloat16 into a temporary (`None` for any other storage).
        fn half_of(&self, t: &Tensor) -> Result<Option<CudaSlice<u16>>, GpuError> {
            match &t.data {
                Data::Cuda32(s) => self.round_half(s, 0, t.len()).map(Some),
                _ => Ok(None),
            }
        }

        /// `count` values of `source` from `offset`, rounded to bfloat16 (`to_bf16`).
        fn round_half(&self, source: &CudaSlice<f32>, offset: usize, count: usize) -> Result<CudaSlice<u16>, GpuError> {
            let mut out = self.stream.alloc_zeros::<u16>(count.max(1)).gpu_ctx("tensor bf16 alloc")?;
            if count > 0 {
                let n = count as u64;
                let f = self.kernel("to_bf16", Storage::F32)?;
                let view = source.slice(offset..offset + count);
                // SAFETY: `to_bf16(n, x, y)` reads n floats of the view and writes n halves.
                unsafe { self.stream.launch_builder(&f).arg(&n).arg(&view).arg(&mut out).launch(cfg_elements(n)) }.gpu_ctx("tensor to_bf16")?;
            }
            Ok(out)
        }

        /// `t` (f32) as a bfloat16 tensor.
        pub(super) fn bf16_copy(&self, t: &Tensor) -> Result<Tensor, GpuError> {
            Ok(Tensor { rows: t.rows, cols: t.cols, data: Data::CudaBf16(self.round_half(slice32(t)?, 0, t.len())?) })
        }

        /// One `cublasGemmEx` (strided-batched when `batch` exceeds one), serialized on the handle
        /// with the float64 path's math-mode switches.
        fn gemm_ex(&self, g: Gemm32<'_>, batch: usize, (stride_a, stride_b, stride_c): (usize, usize, usize)) -> Result<(), GpuError> {
            let serial = self.gemm_workspace.lock().map_err(|_| shape("poisoned GEMM workspace".to_string()))?;
            let (m, n, k) = (i32_of(g.dims.0)?, i32_of(g.dims.1)?, i32_of(g.dims.2)?);
            let (lda, ldb, ldc) = (i32_of(g.a.2)?, i32_of(g.b.2)?, i32_of(g.c.2)?);
            let stride = |s: usize| i64::try_from(s).map_err(|_| shape(format!("GEMM stride {s}")));
            let (stride_a, stride_b, stride_c, count) = (stride(stride_a)?, stride(stride_b)?, stride(stride_c)?, i32_of(batch)?);
            let (alpha, beta) = g.scale;
            let real = cudaDataType_t::CUDA_R_32F;
            // Element offsets reach into a buffer (a vocabulary chunk's head rows, say).
            let (pa, type_a, record_a) = pointer(g.a.0, g.a.1, &self.stream);
            let (pb, type_b, record_b) = pointer(g.b.0, g.b.1, &self.stream);
            let (pc, record_c) = g.c.0.device_ptr_mut(&self.stream);
            let pc = pc + 4 * g.c.1 as u64;
            // SAFETY: the caller checked every operand's shape, offset and leading dimension
            // against (m, n, k) and the buffers' lengths; the pointers outlive the call (their
            // records drop after it).
            let product = unsafe {
                if batch == 1 {
                    cudarc::cublas::result::gemm_ex(
                        *self.blas.handle(), g.ops.0, g.ops.1, m, n, k,
                        (&alpha) as *const f32 as *const _, pa as *const _, type_a, lda, pb as *const _, type_b, ldb,
                        (&beta) as *const f32 as *const _, pc as *mut _, real, ldc, g.compute, cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                } else {
                    cudarc::cublas::result::gemm_strided_batched_ex(
                        *self.blas.handle(), g.ops.0, g.ops.1, m, n, k,
                        (&alpha) as *const f32 as *const _, pa as *const _, type_a, lda, stride_a, pb as *const _, type_b, ldb, stride_b,
                        (&beta) as *const f32 as *const _, pc as *mut _, real, ldc, stride_c, count, g.compute, cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                }
            };
            drop((record_a, record_b, record_c, serial));
            product.gpu_ctx("tensor f32 GEMM")
        }

        pub(super) fn axpy(&self, y: &mut Tensor, alpha: f64, x: &Tensor) -> Result<(), GpuError> {
            let n = y.len() as u64;
            let storage = y.storage();
            let f = self.kernel("axpy", storage)?;
            // SAFETY: equal-length buffers, checked by the caller.
            unsafe { self.stream.launch_builder(&f).arg(&n).arg(&alpha).input(x, storage)?.output(y, storage)?.launch(cfg_elements(n)) }
                .gpu_ctx("tensor axpy")
                .map(|_| ())
        }

        pub(super) fn hadamard(&self, out: &mut Tensor, a: &Tensor, b: &Tensor, accumulate: bool) -> Result<(), GpuError> {
            let n = out.len() as u64;
            let acc = i32::from(accumulate);
            let storage = out.storage();
            let f = self.kernel("hadamard", storage)?;
            // SAFETY: equal-length buffers, checked by the caller.
            unsafe {
                self.stream.launch_builder(&f).arg(&n).input(a, storage)?.input(b, storage)?.output(out, storage)?.arg(&acc).launch(cfg_elements(n))
            }
            .gpu_ctx("tensor hadamard")
            .map(|_| ())
        }

        pub(super) fn add_row(&self, x: &mut Tensor, alpha: f64, row: &Tensor) -> Result<(), GpuError> {
            let n = x.len() as u64;
            let cols = x.cols as u32;
            let storage = x.storage();
            let f = self.kernel("add_row", storage)?;
            // SAFETY: `row` holds `cols` values, `x` n; checked by the caller.
            unsafe {
                self.stream.launch_builder(&f).arg(&n).arg(&cols).arg(&alpha).input(row, storage)?.output(x, storage)?.launch(cfg_elements(n))
            }
            .gpu_ctx("tensor add_row")
            .map(|_| ())
        }

        pub(super) fn columns_of(&self, input: &Tensor, start: usize, width: usize, count: usize) -> Result<Data, GpuError> {
            let storage = input.storage();
            let mut output = self.output(storage, 1, count)?;
            let (n, cols, width, start) = (count as u64, input.cols as u64, width as u64, start as u64);
            let f = self.kernel("columns_of", storage)?;
            // SAFETY: caller validated nonempty in-range columns; each thread
            // copies one element within the source rows into its own output slot.
            unsafe { self.stream.launch_builder(&f).arg(&n).arg(&cols).arg(&width).arg(&start)
                .input(input, storage)?.output(&mut output, storage)?.launch(cfg_elements(n)) }.gpu_ctx("tensor column copy")?;
            Ok(output.data)
        }

        pub(super) fn set_columns<T>(
            &self, storage: Storage, output: &mut CudaSlice<T>, input: &CudaSlice<T>,
            output_cols: usize, input_cols: usize, start: usize, elements: usize,
        ) -> Result<(), GpuError> {
            let (n, output_cols, input_cols, start) = (elements as u64, output_cols as u64, input_cols as u64, start as u64);
            // Bfloat16 values move as their 16 bits.
            let f = if storage == Storage::Bf16 { self.function("set_columns_bits16")? } else { self.kernel("set_columns", storage)? };
            // SAFETY: matching rows and destination column range checked by Device.
            unsafe {
                self.stream.launch_builder(&f).arg(&n).arg(&output_cols).arg(&input_cols).arg(&start)
                    .arg(input).arg(output).launch(cfg_elements(n))
            }.gpu_ctx("tensor set_columns").map(|_| ())
        }

        pub(super) fn scale_columns(&self, out: &mut Tensor, x: &Tensor, d: &Tensor, accumulate: bool) -> Result<(), GpuError> {
            let n = out.len() as u64;
            let cols = out.cols as u32;
            let acc = i32::from(accumulate);
            let storage = out.storage();
            let f = self.kernel("scale_columns", storage)?;
            // SAFETY: shapes checked by the caller.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&n)
                    .arg(&cols)
                    .input(x, storage)?
                    .input(d, storage)?
                    .output(out, storage)?
                    .arg(&acc)
                    .launch(cfg_elements(n))
            }
            .gpu_ctx("tensor scale_columns")
            .map(|_| ())
        }

        pub(super) fn gather_rows(&self, table: &Tensor, ids: &Indices) -> Result<Tensor, GpuError> {
            let storage = table.storage();
            let mut out = self.output(storage, ids.len, table.cols)?;
            let n = out.len() as u64;
            let cols = table.cols as u32;
            let f = self.kernel("gather_rows", storage)?;
            // SAFETY: ids index rows of `table` (the caller's token ids, inside its vocabulary).
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&n)
                    .arg(&cols)
                    .input(table, storage)?
                    .arg(index_slice(ids)?)
                    .output(&mut out, storage)?
                    .launch(cfg_elements(n))
            }
            .gpu_ctx("tensor gather_rows")?;
            Ok(out)
        }

        pub(super) fn laws(&self, x: &Tensor, g: Option<&Tensor>, codes: &Indices, c: f64) -> Result<Tensor, GpuError> {
            let storage = x.storage();
            let mut out = self.output(storage, x.rows, x.cols)?;
            let n = x.len() as u64;
            let cols = x.cols as u32;
            let slopes = i32::from(g.is_some());
            let f = self.kernel("laws", storage)?;
            // SAFETY: equal-length buffers; `codes` holds one code per column.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&n)
                    .arg(&cols)
                    .input(x, storage)?
                    .input(g.unwrap_or(x), storage)?
                    .arg(&slopes)
                    .arg(index_slice(codes)?)
                    .arg(&c)
                    .output(&mut out, storage)?
                    .launch(cfg_elements(n))
            }
            .gpu_ctx("tensor laws")?;
            Ok(out)
        }

        pub(super) fn rms(&self, mode: RmsMode, x: &Tensor, g: Option<&Tensor>, epsilon: f64) -> Result<Tensor, GpuError> {
            let storage = x.storage();
            let mut out = self.output(storage, x.rows, x.cols)?;
            let (rows, cols) = (x.rows as u32, x.cols as u32);
            let code: i32 = match mode {
                RmsMode::Value => 0,
                RmsMode::Backward => 1,
                RmsMode::Tangent => 2,
            };
            let f = self.kernel("rms", storage)?;
            // SAFETY: one block per row of equal-shape buffers.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(&code)
                    .arg(&epsilon)
                    .input(x, storage)?
                    .input(g.unwrap_or(x), storage)?
                    .output(&mut out, storage)?
                    .launch(cfg_rows(x.rows))
            }
            .gpu_ctx("tensor rms")?;
            Ok(out)
        }

        pub(super) fn rotate(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, half_split: bool, inverse: bool) -> Result<Tensor, GpuError> {
            let storage = x.storage();
            let data = match &x.data {
                Data::Cuda(s) => Data::Cuda(self.copy(s)?),
                Data::Cuda32(s) => Data::Cuda32(self.copy(s)?),
                other => return Err(mismatch(other)),
            };
            let mut out = Tensor { rows: x.rows, cols: x.cols, data };
            let (rows, cols, planes) = (x.rows as u32, x.cols as u32, cos.cols as u32);
            let split = i32::from(half_split);
            let sign: f64 = if inverse { -1.0 } else { 1.0 };
            let f = self.kernel("rotate_planes", storage)?;
            // SAFETY: tables are rows × planes; 2·planes ≤ cols, checked by the caller.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(&planes)
                    .arg(&split)
                    .arg(&sign)
                    .input(x, storage)?
                    .input(cos, storage)?
                    .input(sin, storage)?
                    .output(&mut out, storage)?
                    .launch(cfg_elements(u64::from(rows) * u64::from(planes)))
            }
            .gpu_ctx("tensor rotate")?;
            Ok(out)
        }

        pub(super) fn heads(
            &self,
            x: &Tensor,
            out: &mut Tensor,
            (start, heads, width, length, planes): (usize, usize, usize, usize, usize),
            turn: Option<(&Tensor, &Tensor, bool)>,
            (half_split, inverse, merge): (bool, bool, bool),
        ) -> Result<(), GpuError> {
            let storage = x.storage();
            let rows = if merge { out.rows } else { x.rows };
            let cols = if merge { out.cols } else { x.cols };
            let n = (rows * heads * width) as u64;
            let (rows, cols, start, heads, width, length, planes) = (rows as u32, cols as u32, start as u32, heads as u32, width as u32, length as u32, planes as u32);
            let (split, merge) = (i32::from(half_split), i32::from(merge));
            let sign: f64 = if inverse { -1.0 } else { 1.0 };
            // Without a rotation the tables are never read; `x` stands in for them.
            let (cos, sin) = turn.map_or((x, x), |(c, s, _)| (c, s));
            let f = self.kernel("heads_permute", storage)?;
            // SAFETY: shapes checked by the caller: `x` and `out` hold `rows·heads·width` values in
            // their layouts, the tables `rows × planes` when `planes > 0`.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(&start)
                    .arg(&heads)
                    .arg(&width)
                    .arg(&length)
                    .arg(&planes)
                    .arg(&split)
                    .arg(&sign)
                    .arg(&merge)
                    .input(x, storage)?
                    .input(cos, storage)?
                    .input(sin, storage)?
                    .output(out, storage)?
                    .launch(cfg_elements(n))
            }
            .gpu_ctx("tensor heads_permute")?;
            Ok(())
        }

        pub(super) fn softmax_rows(&self, scores: &mut Tensor, causal: bool, start: usize, period: usize) -> Result<(), GpuError> {
            let (rows, width) = (scores.rows as u32, scores.cols as u32);
            let launch = cfg_rows(scores.rows);
            let causal = i32::from(causal);
            let (start, period) = (start as u32, period as u32);
            let storage = scores.storage();
            let f = self.kernel("softmax_rows", storage)?;
            // SAFETY: one block per row of a rows × width buffer.
            unsafe { self.stream.launch_builder(&f).arg(&rows).arg(&width).arg(&causal).arg(&start).arg(&period).output(scores, storage)?.launch(launch) }
                .gpu_ctx("tensor softmax_rows")
                .map(|_| ())
        }

        pub(super) fn softmax_backward(&self, alpha: &Tensor, d: &Tensor) -> Result<Tensor, GpuError> {
            let storage = alpha.storage();
            let mut out = self.output(storage, alpha.rows, alpha.cols)?;
            let (rows, cols) = (alpha.rows as u32, alpha.cols as u32);
            let f = self.kernel("softmax_backward", storage)?;
            // SAFETY: one block per row of equal-shape buffers.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&cols)
                    .input(alpha, storage)?
                    .input(d, storage)?
                    .output(&mut out, storage)?
                    .launch(cfg_rows(alpha.rows))
            }
            .gpu_ctx("tensor softmax_backward")?;
            Ok(out)
        }

        fn flags<'a>(&'a self, scored: Option<&'a Indices>) -> Result<(&'a CudaSlice<u32>, i32), GpuError> {
            match scored {
                Some(s) => Ok((index_slice(s)?, 1)),
                None => Ok((&self.every_row, 0)),
            }
        }

        pub(super) fn softmax_stats_rows(&self, logits: &mut Tensor, scored: Option<&Indices>) -> Result<Vec<[f64; 2]>, GpuError> {
            let n_rows = logits.rows;
            let n = n_rows.checked_mul(2).ok_or_else(|| shape("softmax statistics overflow".into()))?;
            let mut output = self.zeros(n)?;
            let (rows, cols) = (logits.rows as u32, logits.cols as u32);
            let (flags, use_flags) = self.flags(scored)?;
            let storage = logits.storage();
            let f = self.kernel("softmax_stats_rows", storage)?;
            // SAFETY: one block per row, logits rows*cols and output rows*2 buffers.
            unsafe {
                self.stream.launch_builder(&f).arg(&rows).arg(&cols).output(logits, storage)?
                    .arg(flags).arg(&use_flags).arg(&mut output).launch(cfg_rows(n_rows))
            }.gpu_ctx("tensor softmax_stats_rows")?;
            let values = self.download(&output)?;
            let stats = values[..n].chunks_exact(2).map(|x| [x[0], x[1]]).collect::<Vec<_>>();
            if stats.iter().flatten().any(|v| !v.is_finite()) { return Err(shape("nonfinite softmax statistics".into())); }
            Ok(stats)
        }

        pub(super) fn kl_rows(&self, target: &Tensor, logits: &mut Tensor, scored: Option<&Indices>, gradient: bool) -> Result<Vec<f64>, GpuError> {
            let mut kl = self.zeros(logits.rows)?;
            let n_rows = logits.rows;
            self.kl_rows_into(target, logits, scored, gradient, &mut kl)?;
            let mut out = self.download(&kl)?;
            out.truncate(n_rows);
            Ok(out)
        }

        /// The KL rows into `kl` (one double per row), left on the device.
        pub(super) fn kl_rows_into(&self, target: &Tensor, logits: &mut Tensor, scored: Option<&Indices>, gradient: bool, kl: &mut CudaSlice<f64>) -> Result<(), GpuError> {
            let (rows, cols) = (logits.rows as u32, logits.cols as u32);
            let (flags, use_flags) = self.flags(scored)?;
            let storage = logits.storage();
            let f = self.kernel("kl_rows", storage)?;
            let n_rows = logits.rows;
            let gradient = i32::from(gradient);
            // SAFETY: one block per row of equal-shape buffers; `flags` has a flag per row when used.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&cols)
                    .input(target, storage)?
                    .output(logits, storage)?
                    .arg(flags)
                    .arg(&use_flags)
                    .arg(&gradient)
                    .arg(kl)
                    .launch(cfg_rows(n_rows))
            }
            .gpu_ctx("tensor kl_rows")
            .map(|_| ())
        }
        pub(super) fn checked_intervals(&self, input: &Tensor, explained: Option<&Tensor>, operation: Option<CheckedScalar>) -> Result<Vec<CheckedInterval>, GpuError> {
            let count = if operation.is_some() {input.rows.checked_mul(input.cols).ok_or_else(|| shape("checked scalar size overflow".into()))?} else {input.rows};
            if count==0 {return Ok(Vec::new());}
            let module=self.checked_interval_module.get_or_compile_checked_interval(&self.ctx,include_str!("fixed_logit_interval.cu"))?;
            let length=count.checked_mul(3).ok_or_else(|| shape("checked interval allocation overflow".into()))?;
            let mut output=self.zeros(length)?;
            if let Some(op)=operation {
                let f=module.load_function("checked_scalar_interval").gpu_ctx("checked scalar function")?;
                let n=count as u64;
                let mode=match op {CheckedScalar::Exp=>0_u32,CheckedScalar::Log=>1_u32};
                // SAFETY: f64 input has n scalars; output has three doubles per scalar.
                unsafe {self.stream.launch_builder(&f).arg(&n).arg(&mode).arg(slice(input)?).arg(&mut output).launch(cfg_elements(n))}.gpu_ctx("checked scalar intervals")?;
            } else {
                let q=explained.ok_or_else(|| shape("checked KL missing explained tensor".into()))?;
                let f=module.load_function("checked_kl_interval").gpu_ctx("checked KL function")?;
                let (rows,cols)=(input.rows as u32,input.cols as u32);
                // SAFETY: equal-shaped f64 tensors and three output doubles per row;
                // dimensions were range checked, one 256-thread block per row.
                unsafe {self.stream.launch_builder(&f).arg(&rows).arg(&cols).arg(slice(input)?).arg(slice(q)?).arg(&mut output).launch(cfg_rows(input.rows))}.gpu_ctx("checked KL intervals")?;
            }
            let downloaded=self.download(&output)?;
            let mut result=Vec::with_capacity(count);
            for value in downloaded[..length].chunks_exact(3) {
                let interval=match value[2] {
                    0.0 if value[0].is_finite() && value[1].is_finite() && value[0]<=value[1] => CheckedInterval::Bounded {lower:value[0],upper:value[1]},
                    1.0=>CheckedInterval::Unresolved(CheckedIntervalReason::NonFiniteInput),
                    2.0=>CheckedInterval::Unresolved(CheckedIntervalReason::InvalidDomain),
                    3.0=>CheckedInterval::Unresolved(CheckedIntervalReason::ReductionNotEnclosed),
                    4.0=>CheckedInterval::Unresolved(CheckedIntervalReason::UnboundedEndpoints),
                    5.0=>CheckedInterval::Unresolved(CheckedIntervalReason::ArithmeticGuard),
                    _=>return Err(shape("malformed checked interval kernel output".into())),
                };
                result.push(interval);
            }
            Ok(result)
        }

        pub(super) fn kl_proposal_rows(&self, target: &Tensor, logits: &Tensor) -> Result<Vec<KlProposalRow>, GpuError> {
            if logits.rows == 0 { return Ok(Vec::new()); }
            let length = logits.rows.checked_mul(9).ok_or_else(|| shape("KL proposal allocation overflow".into()))?;
            let mut out = self.zeros(length)?;
            let (rows, cols) = (logits.rows as u32, logits.cols as u32);
            let f = self.function("kl_proposal_rows")?;
            // SAFETY: equal-shaped f64 source tensors; one block per row, nine output values per row.
            unsafe {
                self.stream.launch_builder(&f).arg(&rows).arg(&cols).arg(slice(target)?).arg(slice(logits)?).arg(&mut out).launch(cfg_rows(logits.rows))
            }.gpu_ctx("tensor KL proposal rows")?;
            let values = self.download(&out)?;
            let mut result = Vec::with_capacity(logits.rows);
            for row in values[..length].chunks_exact(9) {
                if row[8] != 0.0 || row.iter().any(|x| !x.is_finite()) { return Err(shape("nonfinite KL proposal input or reduction".into())); }
                result.push(KlProposalRow { value: row[0], magnitude: row[1], spread: row[2], max_log_difference: row[7], teacher_max: row[3], teacher_log_sum: row[4], explained_max: row[5], explained_log_sum: row[6] });
            }
            Ok(result)
        }


        pub(super) fn sampled_cotangent(&self, logits: &mut Tensor, uniforms: &Tensor, scored: Option<&Indices>) -> Result<(), GpuError> {
            let (rows, cols) = (logits.rows as u32, logits.cols as u32);
            let (flags, use_flags) = self.flags(scored)?;
            let storage = logits.storage();
            let f = self.kernel("sampled_cotangent", storage)?;
            let n_rows = logits.rows;
            // SAFETY: one block per row; one uniform per row.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&cols)
                    .output(logits, storage)?
                    .input(uniforms, storage)?
                    .arg(flags)
                    .arg(&use_flags)
                    .launch(cfg_rows(n_rows))
            }
            .gpu_ctx("tensor sampled_cotangent")
            .map(|_| ())
        }

        pub(super) fn block_products(&self, left: &Tensor, right: &Tensor, blocks: &ColumnBlocks) -> Result<Tensor, GpuError> {
            let storage = left.storage();
            let mut out = self.output(storage, left.rows, blocks.len())?;
            if out.is_empty() { return Ok(out); }
            let (n, cols, count) = (out.len() as u64, left.cols as u32, blocks.len() as u32);
            let f = self.kernel("block_products", storage)?;
            // SAFETY: shapes agree, and column_blocks validates monotone offsets within cols.
            unsafe {
                self.stream.launch_builder(&f).arg(&n).arg(&cols).arg(&count)
                    .input(left, storage)?.input(right, storage)?.arg(index_slice(&blocks.offsets)?)
                    .output(&mut out, storage)?.launch(cfg_elements(n))
            }.gpu_ctx("tensor block_products")?;
            Ok(out)
        }

        pub(super) fn sampled_head_cotangent(
            &self, probabilities: &Tensor, mean: &Tensor, head: &Tensor, transposed: bool,
            uniforms: &Tensor, scored: Option<&Indices>,
        ) -> Result<Tensor, GpuError> {
            let storage = mean.storage();
            let mut out = self.output(storage, mean.rows, mean.cols)?;
            let (rows, classes, width) = (mean.rows as u32, probabilities.cols as u32, mean.cols as u32);
            let transposed = i32::from(transposed);
            let (flags, use_flags) = self.flags(scored)?;
            let f = self.kernel("sampled_head_cotangent", storage)?;
            // SAFETY: the public entry validates each tensor shape and the scored flag count.
            // One block writes each row; the sampled label is always inside the vocabulary.
            unsafe {
                self.stream.launch_builder(&f)
                    .arg(&rows).arg(&classes).arg(&width)
                    .input(probabilities, storage)?.input(mean, storage)?.input(head, storage)?.arg(&transposed)
                    .input(uniforms, storage)?.arg(flags).arg(&use_flags).output(&mut out, storage)?
                    .launch(cfg_rows(mean.rows))
            }.gpu_ctx("tensor sampled_head_cotangent")?;
            Ok(out)
        }

        pub(super) fn adam(&self, w: &mut Tensor, (m, v): (&mut Tensor, &mut Tensor), g: &Tensor, (rate, beta1, beta2, epsilon): (f64, f64, f64, f64), (c1, c2): (f64, f64)) -> Result<(), GpuError> {
            let n = w.len() as u64;
            let storage = w.storage();
            let f = self.kernel("adam", storage)?;
            // SAFETY: four equal-length buffers, checked by the caller.
            unsafe {
                self.stream.launch_builder(&f).arg(&n).arg(&rate).arg(&beta1).arg(&beta2).arg(&epsilon).arg(&c1).arg(&c2)
                    .input(g, storage)?.output(m, storage)?.output(v, storage)?.output(w, storage)?.launch(cfg_elements(n))
            }
            .gpu_ctx("tensor adam")
            .map(|_| ())
        }

        /// The posterior kernel `name` in the masters' storage `storage` (`_f32` or `_f64`).
        fn posterior_kernel(&self, name: &str, storage: Storage) -> Result<CudaFunction, GpuError> {
            match storage {
                Storage::F64 => self.function(&format!("{name}_f64")),
                Storage::F32 => self.function(&format!("{name}_f32")),
                Storage::Bf16 => Err(shape("bfloat16 posterior masters".to_string())),
            }
        }

        pub(super) fn reparameterize(&self, theta: &mut Tensor, (mean, log_sd): (&Tensor, &Tensor), (key, stream): (u64, u64)) -> Result<(), GpuError> {
            let n = theta.len() as u64;
            if let Data::CudaBf16(out) = &mut theta.data {
                let f = self.function("reparameterize_bf16")?;
                // SAFETY: three equal-length buffers, checked by the caller; f32 posterior entries.
                return unsafe {
                    self.stream.launch_builder(&f).arg(&n).arg(&key).arg(&stream).input(mean, Storage::F32)?.input(log_sd, Storage::F32)?.arg(out).launch(cfg_elements(n))
                }
                .gpu_ctx("tensor reparameterize bf16")
                .map(|_| ());
            }
            let storage = theta.storage();
            let f = self.posterior_kernel("reparameterize", storage)?;
            // SAFETY: three equal-length buffers in one storage, checked by the caller and `input`.
            unsafe {
                self.stream.launch_builder(&f).arg(&n).arg(&key).arg(&stream).input(mean, storage)?.input(log_sd, storage)?.output(theta, storage)?.launch(cfg_elements(n))
            }
            .gpu_ctx("tensor reparameterize")
            .map(|_| ())
        }

        pub(super) fn posterior_adam(
            &self,
            (mean, log_sd): (&mut Tensor, &mut Tensor),
            [mm, mv, sm, sv]: [&mut Tensor; 4],
            gradient: &Tensor,
            (groups, variance): (&Indices, &Tensor),
            sums: &mut Tensor,
            step: &super::PosteriorStep,
        ) -> Result<(), GpuError> {
            let (c1, c2) = step.corrections();
            let (n, storage, moments, gradients) = (mean.len() as u64, mean.storage(), mm.storage(), gradient.storage());
            // f32 masters may keep bfloat16 moments and take a bfloat16 gradient; otherwise every
            // entry is in the masters' storage.
            let f = match (storage, moments, gradients) {
                (Storage::F32, Storage::Bf16, Storage::F32) => self.function("posterior_adam_f32_bf16")?,
                (Storage::F32, Storage::Bf16, Storage::Bf16) => self.function("posterior_adam_bf16_bf16")?,
                (masters, m, g) if masters == m && masters == g => self.posterior_kernel("posterior_adam", storage)?,
                (masters, m, g) => return Err(shape(format!("{masters:?} masters with {m:?} moments and a {g:?} gradient"))),
            };
            let count = variance.len() as u64;
            let mut builder = self.stream.launch_builder(&f);
            builder.arg(&n).arg(&count).arg(&step.key).arg(&step.stream).arg(&step.gradient_scale).arg(&step.mean_rate).arg(&step.log_sd_rate).arg(&step.beta1).arg(&step.beta2).arg(&step.epsilon).arg(&c1).arg(&c2);
            builder.input(gradient, gradients)?.arg(index_slice(groups)?).arg(slice(variance)?);
            builder.output(mean, storage)?.output(log_sd, storage)?.output(mm, moments)?.output(mv, moments)?.output(sm, moments)?.output(sv, moments)?.arg(slice_mut(sums)?);
            // SAFETY: equal-length entry buffers in one storage, float64 group buffers of `count`
            // rows; ids at or beyond `count` are skipped by the kernel.
            unsafe { builder.launch(cfg_elements(n)) }.gpu_ctx("tensor posterior_adam").map(|_| ())
        }

        pub(super) fn group_moments(&self, (mean, log_sd): (&Tensor, &Tensor), groups: &Indices, sums: &mut Tensor) -> Result<(), GpuError> {
            let (n, storage) = (mean.len() as u64, mean.storage());
            let (f, count) = (self.posterior_kernel("group_moments", storage)?, sums.rows as u64);
            let mut builder = self.stream.launch_builder(&f);
            builder.arg(&n).arg(&count).input(mean, storage)?.input(log_sd, storage)?.arg(index_slice(groups)?).arg(slice_mut(sums)?);
            // SAFETY: equal-length entry buffers, checked by the caller; float64 sums of `count` rows.
            unsafe { builder.launch(cfg_elements(n)) }.gpu_ctx("tensor group_moments").map(|_| ())
        }

        pub(super) fn group_divergence(&self, sums: &mut Tensor, variance: &mut Tensor, divergence: &mut Tensor) -> Result<(), GpuError> {
            let n = variance.len() as u64;
            let f = self.function("group_divergence")?;
            let mut builder = self.stream.launch_builder(&f);
            builder.arg(&n).arg(slice_mut(sums)?).arg(slice_mut(variance)?).arg(slice_mut(divergence)?);
            // SAFETY: groups × 3 sums and groups × 1 outputs, checked by the caller.
            unsafe { builder.launch(cfg_elements(n)) }.gpu_ctx("tensor group_divergence").map(|_| ())
        }

        pub(super) fn select_sets(&self, (a, q): (&Tensor, &Tensor), bits: &Tensor, left: &Tensor, weight: &Tensor, mask: &mut Tensor) -> Result<(), GpuError> {
            if a.is_empty() { return Ok(()); }
            let (rows, cols) = (a.rows as u32, a.cols as u32);
            let width = a.cols.next_power_of_two().max(2);
            // One scratch ranking and row of sizes per block; the blocks stride over the rows.
            let blocks = a.rows.min(4096);
            let storage = a.storage();
            let mut keys = self.tensor(storage, blocks, width)?;
            let mut sizes = self.tensor(storage, blocks, width)?;
            let mut order = self.stream.alloc_zeros::<u32>(blocks * width).gpu_ctx("tensor alloc")?;
            let f = self.kernel("select_sets", storage)?;
            let width32 = width as u32;
            let cfg = LaunchConfig { grid_dim: (blocks as u32, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: shapes checked by the caller; each block owns `width` scratch entries.
            unsafe {
                self.stream.launch_builder(&f).arg(&rows).arg(&cols).arg(&width32)
                    .input(a, storage)?.input(q, storage)?.input(bits, storage)?.input(left, storage)?.input(weight, storage)?
                    .output(&mut keys, storage)?.arg(&mut order).output(&mut sizes, storage)?.output(mask, storage)?.launch(cfg)
            }
            .gpu_ctx("tensor select_sets")
            .map(|_| ())
        }

        pub(super) fn divide_sums(&self, x: &mut Tensor, rows: &Tensor, cols: &Tensor, floor: f64) -> Result<(), GpuError> {
            let (n, width) = (x.len() as u64, x.cols as u32);
            if n == 0 { return Ok(()); }
            let storage = x.storage();
            let f = self.kernel("divide_sums", storage)?;
            // SAFETY: shapes checked by the caller.
            unsafe {
                self.stream.launch_builder(&f).arg(&n).arg(&width).input(rows, storage)?.input(cols, storage)?.arg(&floor).output(x, storage)?.launch(cfg_elements(n))
            }
            .gpu_ctx("tensor divide_sums")
            .map(|_| ())
        }
        pub(super) fn code_rows(
            &self,
            (z, w, yfy): (&Tensor, &Tensor, &Tensor),
            (gram, starts, bits): (&Tensor, &Indices, &Tensor),
            warm: Option<&Tensor>,
            (kappa, nodes, tolerance): (f64, usize, f64),
            on: &mut Tensor,
            workspace: Option<(CodeRowsWorkspace, bool)>,
        ) -> Result<(Vec<f64>, Vec<f64>, Option<CodeRowsDiagnostics>), GpuError> {
            let started = std::time::Instant::now();
            let (rows, pieces, blocks) = (z.rows, z.cols, bits.cols);
            if rows == 0 {
                return Ok((Vec::new(), Vec::new(), workspace.map(|_| CodeRowsDiagnostics { rows: Vec::new(), elapsed: started.elapsed(), concurrent_rows: 0, workspace_bytes: 0 })));
            }
            // One module per device (each context loads its own).
            static CODER: std::sync::OnceLock<crate::device_cache::KeyedPtxModuleCache<usize>> = std::sync::OnceLock::new();
            let module = CODER.get_or_init(crate::device_cache::KeyedPtxModuleCache::new).get_or_compile(&self.ctx, self.ctx.ordinal(), "sparse code", |_| CODER_KERNELS.to_string())?;
            let f = module.load_function("code_rows").gpu_ctx("sparse code kernel")?;
            // Per slot: 15 rows of blocks, one of pieces, the sort keys, and every open node's
            // bounds, point, its `Q m` and lower bound; indices for the working set and the sort.
            let settings = match workspace {
                Some((w, _)) => w,
                None => {
                    let (free, _) = self.memory()?;
                    // Preserve the legacy path's at-least-one-slot allocation policy.
                    let one = code_rows_layout(1, pieces, blocks, nodes, CodeRowsWorkspace { bytes: usize::MAX, max_rows: 1, cache_columns: false })?;
                    CodeRowsWorkspace { bytes: (free / 4).max(one.bytes), max_rows: 384, cache_columns: false }
                }
            };
            let layout = code_rows_layout(rows, pieces, blocks, nodes, settings)?;
            let width = blocks.checked_next_power_of_two().ok_or_else(|| shape("sparse code width overflow".to_string()))?.max(2);
            let open = nodes.checked_add(1).ok_or_else(|| shape("sparse code node capacity overflow".to_string()))?;
            let (per_slot, per_index, slots) = (layout.doubles, layout.indices, layout.slots);
            let fused = i32::from(workspace.is_some_and(|(_, fused)| fused));
            let cached = i32::from(settings.cache_columns);
            let instrumented = i32::from(workspace.is_some());
            let cache_offset = layout.cache_offset as u64;
            let mut counters = self.stream.alloc_zeros::<u64>(if workspace.is_some() { rows.checked_mul(7).ok_or_else(|| shape("sparse code counters overflow".to_string()))? } else { 1 }).gpu_ctx("sparse code counters")?;
            let mut scratch = self.zeros(slots * per_slot)?;
            let mut indices = self.stream.alloc_zeros::<u32>(slots * per_index).gpu_ctx("sparse code alloc")?;
            let (mut upper, mut lower) = (self.zeros(rows)?, self.zeros(rows)?);
            let no_warm = self.zeros(1)?;
            let warmed: i32 = i32::from(warm.is_some());
            let unit: i32 = i32::from(blocks == pieces);
            let small = |n| u32::try_from(n).map_err(|_| shape("sparse code CUDA dimension exceeds u32".to_string()));
            let (rows32, pieces32, blocks32, width32, nodes32, open32) = (small(rows)?, small(pieces)?, small(blocks)?, small(width)?, small(nodes)?, small(open)?);
            let (slot_len, index_len) = (per_slot as u64, per_index as u64);
            let cfg = LaunchConfig { grid_dim: (slots as u32, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: shapes checked by the caller; each block owns its slot of both scratches.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows32).arg(&pieces32).arg(&blocks32).arg(&width32).arg(&nodes32).arg(&open32).arg(&unit).arg(&warmed)
                    .arg(&kappa).arg(&tolerance)
                    .arg(slice(z)?).arg(slice(w)?).arg(slice(yfy)?).arg(slice(gram)?).arg(index_slice(starts)?).arg(slice(bits)?)
                    .arg(match warm { Some(m) => slice(m)?, None => &no_warm })
                    .arg(&slot_len).arg(&index_len).arg(&cache_offset).arg(&cached).arg(&instrumented).arg(&fused).arg(&mut scratch).arg(&mut indices).arg(&mut counters)
                    .arg(slice_mut(on)?).arg(&mut upper).arg(&mut lower)
                    .launch(cfg)
            }
            .gpu_ctx("sparse code launch")?;
            let mut upper = self.download(&upper)?;
            let mut lower = self.download(&lower)?;
            upper.truncate(rows);
            lower.truncate(rows);
            let profile = if workspace.is_some() {
                let counts = self.stream.clone_dtoh(&counters).gpu_ctx("sparse code counters download")?;
                Some(CodeRowsDiagnostics {
                    rows: counts.chunks_exact(7).map(|c| CodeRowCounters { relaxations: c[0], sweeps: c[1], column_computations: c[2], column_cache_hits: c[3], rounding_flips: c[4], explored_nodes: c[5], sweep_limit_hits: c[6] }).collect(),
                    elapsed: started.elapsed(), concurrent_rows: slots, workspace_bytes: layout.bytes,
                })
            } else { None };
            Ok((upper, lower, profile))
        }


        pub(super) fn box_charge(&self, z: &Tensor, mask: &Tensor, q: &Tensor, cot: &mut Tensor, coefficient: &mut Tensor) -> Result<Vec<f64>, GpuError> {
            if z.is_empty() { return Ok(vec![0.0; z.rows]); }
            let (rows, cols) = (z.rows as u32, z.cols as u32);
            let mut norms = self.zeros(z.rows)?;
            let storage = z.storage();
            let f = self.kernel("box_charge", storage)?;
            let cfg = LaunchConfig { grid_dim: (z.rows.min(65_535) as u32, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: equal shapes checked by the caller; one block per row (striding).
            unsafe {
                self.stream.launch_builder(&f).arg(&rows).arg(&cols).input(z, storage)?.input(mask, storage)?.input(q, storage)?
                    .arg(&mut norms).output(cot, storage)?.output(coefficient, storage)?.launch(cfg)
            }
            .gpu_ctx("tensor box_charge")?;
            let mut values = self.download(&norms)?;
            values.truncate(z.rows);
            Ok(values.into_iter().map(|n| 0.5 * n * n).collect())
        }

        pub(super) fn argmax_rows(&self, t: &Tensor) -> Result<Vec<usize>, GpuError> {
            let mut out = self.zeros(t.rows)?;
            let (rows, cols) = (t.rows as u32, t.cols as u32);
            let storage = t.storage();
            let f = self.kernel("argmax_rows", storage)?;
            // SAFETY: one block per row of `t`; `out` holds one value per row.
            unsafe { self.stream.launch_builder(&f).arg(&rows).arg(&cols).input(t, storage)?.arg(&mut out).launch(cfg_rows(t.rows)) }
                .gpu_ctx("tensor argmax_rows")?;
            Ok(self.download(&out)?.into_iter().take(t.rows).map(|v| v as usize).collect())
        }

        pub(super) fn fill_entries(&self, t: &mut Tensor, at: &Indices, value: f64) -> Result<(), GpuError> {
            let n = at.len as u64;
            if n == 0 {
                return Ok(());
            }
            let storage = t.storage();
            let f = self.kernel("fill_entries", storage)?;
            // SAFETY: every position is below `t`'s length (the caller's contract, checked on the host).
            unsafe { self.stream.launch_builder(&f).arg(&n).arg(index_slice(at)?).arg(&value).output(t, storage)?.launch(cfg_elements(n)) }
                .gpu_ctx("tensor fill_entries")
                .map(|_| ())
        }

        pub(super) fn softmax_quadratic(&self, logits: &Tensor, tangent: &Tensor) -> Result<Vec<f64>, GpuError> {
            let mut out = self.zeros(logits.rows)?;
            let (rows, cols) = (logits.rows as u32, logits.cols as u32);
            let storage = logits.storage();
            let f = self.kernel("softmax_quadratic", storage)?;
            // SAFETY: one block per row of equal-shape buffers.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&cols)
                    .input(logits, storage)?
                    .input(tangent, storage)?
                    .arg(&mut out)
                    .launch(cfg_rows(logits.rows))
            }
            .gpu_ctx("tensor softmax_quadratic")?;
            let mut values = self.download(&out)?;
            values.truncate(logits.rows);
            Ok(values)
        }

        /// [`super::Device::head_log_partition`] on f32 tensors: the classes swept in chunks of
        /// `chunk` (a rows × chunk logit buffer, reused), each chunk's logits one product, merged
        /// into a running (largest, sum) per row by `head_chunk`; with `expected`, the chunk's
        /// exponentials weight its head rows into the rows × width accumulator (one more product),
        /// rescaled by `scale_rows` as the largest grows.
        pub(super) fn head_log_partition(
            &self,
            hidden: &Tensor,
            head: (&Tensor, bool),
            scored: Option<&Indices>,
            expected: Option<&mut Tensor>,
            settings: (usize, Arithmetic),
        ) -> Result<Vec<f64>, GpuError> {
            let mut out = self.zeros(hidden.rows)?;
            self.head_log_partition_into(hidden, head, scored, expected, &mut out, settings)?;
            let mut values = self.download(&out)?;
            values.truncate(hidden.rows);
            Ok(values)
        }

        /// The swept log partitions into `out` (one double per row), left on the device: no
        /// host transfer, so a captured step can record it.
        pub(super) fn head_log_partition_into(
            &self,
            hidden: &Tensor,
            (head, transposed): (&Tensor, bool),
            scored: Option<&Indices>,
            mut expected: Option<&mut Tensor>,
            out: &mut CudaSlice<f64>,
            (chunk, arithmetic): (usize, Arithmetic),
        ) -> Result<(), GpuError> {
            let (rows, width) = hidden.dim();
            let classes = if transposed { head.cols } else { head.rows };
            let (compute, half) = compute_of(arithmetic, &self.name)?;
            // In bfloat16 the hidden rows (and an f32 head) are rounded once per call, each chunk's
            // exponentials before they weight its head rows; a bfloat16 head is read as it is.
            let (rounded_hidden, rounded_head) = if half { (self.half_of(hidden)?, self.half_of(head)?) } else { (None, None) };
            let head_factor = match (&rounded_head, &head.data) {
                (Some(r), _) => Factor::Half(r),
                (None, Data::CudaBf16(h)) if half => Factor::Half(h),
                (None, Data::CudaBf16(_)) => return Err(shape(format!("a bfloat16 head takes Arithmetic::Bf16, not {arithmetic:?}"))),
                (None, _) => Factor::Single(slice32(head)?),
            };
            let hidden_factor = match &rounded_hidden {
                Some(r) => Factor::Half(r),
                None => Factor::Single(slice32(hidden)?),
            };
            let chunk = chunk.clamp(1, classes.max(1));
            let mut logits = self.zeros32(rows * chunk)?;
            let mut largest = self.zeros32(rows)?;
            let (n_rows, lowest) = (rows as u64, f64::NEG_INFINITY);
            let fill = self.kernel("fill", Storage::F32)?;
            // SAFETY: `fill(n, v, x)` writes n floats.
            unsafe { self.stream.launch_builder(&fill).arg(&n_rows).arg(&lowest).arg(&mut largest).launch(cfg_elements(n_rows)) }.gpu_ctx("tensor fill")?;
            let mut sums = self.zeros(rows)?;
            let mut factor = self.zeros32(rows)?;
            let (rows32, width32) = (u32::try_from(rows).map_err(|_| shape("head rows exceed u32".to_string()))?, width as u32);
            let want = i32::from(expected.is_some());
            let (t, n_op) = (cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N);
            let chunk_kernel = self.kernel("head_chunk", Storage::F32)?;
            let rescale = self.kernel("scale_rows", Storage::F32)?;
            for (index, start) in (0..classes).step_by(chunk).enumerate() {
                let count = chunk.min(classes - start);
                // Column-major logitsᵀ (count × rows) = E_chunk (count × width) · hiddenᵀ: the head's
                // rows `start..` (row-major classes × width, read transposed) or its columns
                // (width × classes stored, read as they are).
                let (head_op, head_offset, head_ld) = if transposed { (n_op, start, classes) } else { (t, start * width, width) };
                self.gemm_ex(
                    Gemm32 {
                        ops: (head_op, n_op),
                        dims: (count, rows, width),
                        scale: (1.0, 0.0),
                        a: (head_factor, head_offset, head_ld),
                        b: (hidden_factor, 0, width),
                        c: (&mut logits, 0, count),
                        compute,
                    },
                    1,
                    (0, 0, 0),
                )?;
                let count32 = count as u32;
                // SAFETY: one block per row of the rows × count logits; per-row state of length rows.
                unsafe {
                    self.stream.launch_builder(&chunk_kernel).arg(&rows32).arg(&count32).arg(&mut logits).arg(&mut largest)
                        .arg(&mut sums).arg(&mut factor).arg(&want).launch(cfg_rows(rows))
                }
                .gpu_ctx("tensor head_chunk")?;
                if let Some(out) = expected.as_deref_mut() {
                    let n = out.len() as u64;
                    if index > 0 && n > 0 {
                        // SAFETY: `out` is rows × width; `factor` holds one value per row.
                        unsafe { self.stream.launch_builder(&rescale).arg(&n).arg(&width32).arg(&factor).output(out, Storage::F32)?.launch(cfg_elements(n)) }
                            .gpu_ctx("tensor scale_rows")?;
                    }
                    // Column-major outᵀ (width × rows) += E_chunkᵀ (width × count) · Pᵀ (count × rows).
                    let (head_op, head_ld) = if transposed { (t, classes) } else { (n_op, width) };
                    let rounded = if half { Some(self.round_half(&logits, 0, rows * count)?) } else { None };
                    let weights = match &rounded {
                        Some(r) => Factor::Half(r),
                        None => Factor::Single(&logits),
                    };
                    self.gemm_ex(
                        Gemm32 {
                            ops: (head_op, n_op),
                            dims: (width, rows, count),
                            scale: (1.0, if index == 0 { 0.0 } else { 1.0 }),
                            a: (head_factor, if transposed { start } else { start * width }, head_ld),
                            b: (weights, 0, count),
                            c: (slice32_mut(out)?, 0, width),
                            compute,
                        },
                        1,
                        (0, 0, 0),
                    )?;
                }
            }
            let (flags, use_flags) = self.flags(scored)?;
            let finish = self.kernel("head_finish", Storage::F32)?;
            let mut none = self.zeros32(1)?;
            let mean = match expected {
                Some(t) => slice32_mut(t)?,
                None => &mut none,
            };
            // SAFETY: per-row state of length rows; `mean` is rows × width when `want`.
            unsafe {
                self.stream.launch_builder(&finish).arg(&rows32).arg(&width32).arg(&largest).arg(&sums).arg(flags).arg(&use_flags)
                    .arg(&want).arg(mean).arg(out).launch(cfg_rows(rows))
            }
            .gpu_ctx("tensor head_finish")
            .map(|_| ())
        }
    }
}

#[cfg(target_os = "macos")]
mod apple {
    use super::{Arithmetic, ColumnBlocks, Data, IndexData, Indices, Op, RmsMode, Tensor, foreign, shape};
    use crate::apple_gpu::MetalRuntime;
    use crate::gpu_error::GpuError;
    use crate::metal::stream::{Buffer, GROUP, Matrix, Stream};

    /// The kernels: f32 twins of the CUDA ones, one value per thread over a grid-strided range,
    /// or one threadgroup per row (strided over the rows) reducing in threadgroup memory.
    const KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#define GROUP 256u

// Every kernel's parameters; each reads the fields it names.
struct P { uint n; uint rows; uint cols; uint extra; uint a; uint b; float alpha; float beta; uint c; uint d; };

#define ELEMENTS for (uint i = gid; i < p.n; i += grid)
#define ROWS for (uint r = group; r < p.rows; r += groups)

inline float group_sum(float v, threadgroup float* s, uint t) {
    s[t] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = GROUP / 2; w > 0; w >>= 1) {
        if (t < w) s[t] += s[t + w];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = s[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline float group_max(float v, threadgroup float* s, uint t) {
    s[t] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint w = GROUP / 2; w > 0; w >>= 1) {
        if (t < w) s[t] = max(s[t], s[t + w]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = s[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

kernel void t_copy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant P& p [[buffer(2)]],
                   uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS y[i] = x[i];
}

// Copy raw bits, including signed zeros and subnormals, without floating arithmetic.
kernel void t_copy_columns(device const uint* x [[buffer(0)]], device uint* y [[buffer(1)]], constant P& p [[buffer(2)]],
                          uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS {
        uint at = (i / p.extra) * p.cols + p.a + i % p.extra;
        if (p.b) y[at] = x[i]; else y[i] = x[at];
    }
}

kernel void t_scale(device float* x [[buffer(0)]], constant P& p [[buffer(1)]],
                    uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS x[i] = p.alpha * x[i];
}

kernel void t_axpy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant P& p [[buffer(2)]],
                   uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS y[i] = y[i] + p.alpha * x[i];
}

kernel void t_hadamard(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* out [[buffer(2)]],
                       constant P& p [[buffer(3)]], uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS out[i] = p.a ? out[i] + a[i] * b[i] : a[i] * b[i];
}

kernel void t_add_row(device const float* row [[buffer(0)]], device float* x [[buffer(1)]], constant P& p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS x[i] = x[i] + p.alpha * row[i % p.cols];
}

kernel void t_scale_columns(device const float* x [[buffer(0)]], device const float* d [[buffer(1)]], device float* out [[buffer(2)]],
                            constant P& p [[buffer(3)]], uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS {
        float term = x[i] * d[i % p.cols];
        out[i] = p.a ? out[i] + term : term;
    }
}

kernel void t_gather_rows(device const float* table [[buffer(0)]], device const uint* ids [[buffer(1)]], device float* out [[buffer(2)]],
                          constant P& p [[buffer(3)]], uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS out[i] = table[ids[i / p.cols] * p.cols + i % p.cols];
}

// erfc to a relative error below 1.2e-7 (Numerical Recipes' erfcc, a Chebyshev fit).
inline float gam_erfc(float x) {
    float z = fabs(x);
    float t = 1.0f / (1.0f + 0.5f * z);
    float poly = -1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f + t * (0.27886807f
        + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
    float r = t * exp(-z * z + poly);
    return x >= 0.0f ? r : 2.0f - r;
}

inline float law_value(uint code, float t, float c) {
    switch (code) {
        case 0: return t > 0.0f ? t : 0.0f;
        case 1: return t;
        case 2: return 0.0f;
        case 3: return t / (1.0f + exp(-t));
        case 4: return t * (0.5f * gam_erfc(-t * 0.70710678118654752f));
        default: {
            float inner = c * (t + 0.044715f * t * t * t);
            return 0.5f * t * (1.0f + tanh(inner));
        }
    }
}

inline float law_slope(uint code, float t, float c) {
    switch (code) {
        case 0: return t > 0.0f ? 1.0f : 0.0f;
        case 1: return 1.0f;
        case 2: return 0.0f;
        case 3: {
            float sigma = 1.0f / (1.0f + exp(-t));
            return sigma * (1.0f + t * (1.0f - sigma));
        }
        case 4: return 0.5f * gam_erfc(-t * 0.70710678118654752f) + t * (exp(-0.5f * t * t) * 0.39894228040143268f);
        default: {
            float inner = c * (t + 0.044715f * t * t * t);
            float th = tanh(inner);
            return 0.5f * (1.0f + th) + 0.5f * t * (1.0f - th * th) * c * (1.0f + 3.0f * 0.044715f * t * t);
        }
    }
}

// a: slopes (out = g f'(x)) or values (out = f(x)); alpha: the tanh GELU's constant.
kernel void t_laws(device const float* x [[buffer(0)]], device const float* g [[buffer(1)]], device const uint* codes [[buffer(2)]],
                   device float* out [[buffer(3)]], constant P& p [[buffer(4)]],
                   uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS {
        uint code = codes[i % p.cols];
        out[i] = p.a ? g[i] * law_slope(code, x[i], p.alpha) : law_value(code, x[i], p.alpha);
    }
}

// a: 0 value, 1 cotangent given g, 2 tangent along g; alpha: epsilon.
kernel void t_rms(device const float* x [[buffer(0)]], device const float* g [[buffer(1)]], device float* out [[buffer(2)]],
                  constant P& p [[buffer(3)]], uint group [[threadgroup_position_in_grid]],
                  uint groups [[threadgroups_per_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float shared[GROUP];
    ROWS {
        device const float* xr = x + (ulong)r * p.cols;
        device const float* gr = g + (ulong)r * p.cols;
        device float* o = out + (ulong)r * p.cols;
        float squares = 0.0f, inner = 0.0f;
        for (uint c = t; c < p.cols; c += GROUP) {
            squares += xr[c] * xr[c];
            if (p.a != 0) inner += xr[c] * gr[c];
        }
        float n = (float)p.cols;
        float mean = group_sum(squares, shared, t) / n;
        float scale = 1.0f / sqrt(mean + p.alpha);
        float dot = group_sum(inner, shared, t);
        if (p.a == 0) {
            for (uint c = t; c < p.cols; c += GROUP) o[c] = xr[c] * scale;
        } else if (p.a == 1) {
            for (uint c = t; c < p.cols; c += GROUP) o[c] = scale * gr[c] - scale * scale * scale / n * xr[c] * dot;
        } else {
            float dm = 2.0f * dot / n;
            float ds = -0.5f * scale * scale * scale * dm;
            for (uint c = t; c < p.cols; c += GROUP) o[c] = gr[c] * scale + xr[c] * ds;
        }
    }
}

// n: rows · planes; extra: planes; a: half split; alpha: the sine's sign. `out` holds `x` already.
kernel void t_rotate(device const float* x [[buffer(0)]], device const float* cosines [[buffer(1)]], device const float* sines [[buffer(2)]],
                     device float* out [[buffer(3)]], constant P& p [[buffer(4)]],
                     uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS {
        uint r = i / p.extra, k = i % p.extra;
        uint a = p.a ? k : 2 * k;
        uint b = p.a ? k + p.extra : 2 * k + 1;
        float c = cosines[i], s = p.alpha * sines[i];
        float xa = x[r * p.cols + a], xb = x[r * p.cols + b];
        out[r * p.cols + a] = c * xa - s * xb;
        out[r * p.cols + b] = s * xa + c * xb;
    }
}

// [`heads_permute`]: n: rows · heads · width; rows: the sequence length; cols: the row-major width;
// extra: the first column; a: heads; b: width; c: planes; d: 1 half split, 2 merge; alpha: sign.
kernel void t_heads(device const float* x [[buffer(0)]], device const float* cosines [[buffer(1)]], device const float* sines [[buffer(2)]],
                    device float* out [[buffer(3)]], constant P& p [[buffer(4)]],
                    uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    bool half_split = (p.d & 1) != 0, merge = (p.d & 2) != 0;
    ELEMENTS {
        uint j = i % p.b, t = i / p.b;
        uint l = t % p.rows;
        t /= p.rows;
        uint h = t % p.a;
        uint r = (t / p.a) * p.rows + l;
        uint wide = r * p.cols + p.extra + h * p.b, narrow = i - j;
        device const float* src = x + (merge ? narrow : wide);
        float v = src[j];
        if (j < 2 * p.c) {
            uint q = half_split ? (j < p.c ? j : j - p.c) : j / 2;
            bool first = half_split ? j < p.c : (j % 2) == 0;
            uint partner = half_split ? (first ? j + p.c : j - p.c) : (first ? j + 1 : j - 1);
            float c = cosines[r * p.c + q], s = p.alpha * sines[r * p.c + q];
            float o = src[partner];
            v = first ? c * v - s * o : s * o + c * v;
        }
        out[(merge ? wide : narrow) + j] = v;
    }
}

// a: causal; extra: the tile's first query position.
kernel void t_softmax_rows(device float* s [[buffer(0)]], constant P& p [[buffer(1)]], uint group [[threadgroup_position_in_grid]],
                           uint groups [[threadgroups_per_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float shared[GROUP];
    ROWS {
        device float* row = s + (ulong)r * p.cols;
        uint valid = p.a ? min(p.extra + r % p.b + 1, p.cols) : p.cols;
        float m = -INFINITY;
        for (uint c = t; c < valid; c += GROUP) m = max(m, row[c]);
        m = group_max(m, shared, t);
        float total = 0.0f;
        for (uint c = t; c < p.cols; c += GROUP) {
            if (c < valid) {
                float e = exp(row[c] - m);
                row[c] = e;
                total += e;
            } else {
                row[c] = 0.0f;
            }
        }
        total = group_sum(total, shared, t);
        for (uint c = t; c < valid; c += GROUP) row[c] = row[c] / total;
    }
}

kernel void t_softmax_backward(device const float* alpha [[buffer(0)]], device const float* d [[buffer(1)]], device float* out [[buffer(2)]],
                               constant P& p [[buffer(3)]], uint group [[threadgroup_position_in_grid]],
                               uint groups [[threadgroups_per_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float shared[GROUP];
    ROWS {
        device const float* a = alpha + (ulong)r * p.cols;
        device const float* dr = d + (ulong)r * p.cols;
        float partial = 0.0f;
        for (uint c = t; c < p.cols; c += GROUP) partial += a[c] * dr[c];
        float mean = group_sum(partial, shared, t);
        for (uint c = t; c < p.cols; c += GROUP) out[(ulong)r * p.cols + c] = a[c] * (dr[c] - mean);
    }
}

// The row's max and the sum of exp(z − max).
inline float2 softmax_stats(device const float* z, uint cols, threadgroup float* shared, uint t) {
    float m = -INFINITY;
    for (uint c = t; c < cols; c += GROUP) m = max(m, z[c]);
    m = group_max(m, shared, t);
    float total = 0.0f;
    for (uint c = t; c < cols; c += GROUP) total += exp(z[c] - m);
    return float2(m, group_sum(total, shared, t));
}

// a: rows flagged by `scored`; b: write the cotangent q − p in place of the logits.
kernel void t_kl_rows(device const float* target [[buffer(0)]], device float* logits [[buffer(1)]], device const uint* scored [[buffer(2)]],
                      device float* kl [[buffer(3)]], constant P& p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
                      uint groups [[threadgroups_per_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float shared[GROUP];
    ROWS {
        device const float* tr = target + (ulong)r * p.cols;
        device float* z = logits + (ulong)r * p.cols;
        if (p.a != 0 && scored[r] == 0) {
            if (p.b != 0) for (uint c = t; c < p.cols; c += GROUP) z[c] = 0.0f;
            if (t == 0) kl[r] = 0.0f;
            continue;
        }
        float2 st = softmax_stats(tr, p.cols, shared, t);
        float2 sz = softmax_stats(z, p.cols, shared, t);
        float lt = log(st.y), lz = log(sz.y);
        float acc = 0.0f;
        for (uint c = t; c < p.cols; c += GROUP) {
            float q = exp(tr[c] - st.x) / st.y;
            float log_p = (tr[c] - st.x) - lt;
            float log_q = (z[c] - sz.x) - lz;
            if (q > 0.0f) acc += q * (log_p - log_q);
            if (p.b != 0) z[c] = exp(z[c] - sz.x) / sz.y - q;
        }
        float total = group_sum(acc, shared, t);
        if (t == 0) kl[r] = total;
    }
}

// The first class whose cumulative probability passes `pick` (the last if none does): each thread
// sums a contiguous chunk of the row, then one thread walks the chunk sums and the chosen chunk.
inline uint pick_label(device const float* q, uint cols, float pick, threadgroup float* shared, threadgroup uint* label, uint t) {
    uint chunk = (cols + GROUP - 1) / GROUP;
    uint lo = min(t * chunk, cols), hi = min(lo + chunk, cols);
    float part = 0.0f;
    for (uint c = lo; c < hi; c++) part += q[c];
    shared[t] = part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        uint chosen = cols - 1;
        float left = pick;
        for (uint k = 0; k < GROUP; k++) {
            if (left < shared[k]) {
                uint start = k * chunk, end = min(start + chunk, cols);
                chosen = end - 1;
                for (uint c = start; c < end; c++) {
                    if (left < q[c]) { chosen = c; break; }
                    left -= q[c];
                }
                break;
            }
            left -= shared[k];
        }
        *label = chosen;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint chosen = *label;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return chosen;
}

// a: rows flagged by `scored`.
kernel void t_sampled_cotangent(device float* logits [[buffer(0)]], device const float* uniforms [[buffer(1)]], device const uint* scored [[buffer(2)]],
                                constant P& p [[buffer(3)]], uint group [[threadgroup_position_in_grid]],
                                uint groups [[threadgroups_per_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float shared[GROUP];
    threadgroup uint label;
    ROWS {
        device float* z = logits + (ulong)r * p.cols;
        if (p.a != 0 && scored[r] == 0) {
            for (uint c = t; c < p.cols; c += GROUP) z[c] = 0.0f;
            continue;
        }
        float2 s = softmax_stats(z, p.cols, shared, t);
        for (uint c = t; c < p.cols; c += GROUP) z[c] = exp(z[c] - s.x) / s.y;
        threadgroup_barrier(mem_flags::mem_device);
        uint chosen = pick_label(z, p.cols, uniforms[r], shared, &label, t);
        for (uint c = t; c < p.cols; c += GROUP) z[c] = z[c] - ((c == chosen) ? 1.0f : 0.0f);
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// n: rows · blocks; extra: blocks.
kernel void t_block_products(device const float* left [[buffer(0)]], device const float* right [[buffer(1)]], device const uint* offsets [[buffer(2)]],
                             device float* out [[buffer(3)]], constant P& p [[buffer(4)]],
                             uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS {
        uint row = i / p.extra, block = i % p.extra;
        float sum = 0.0f;
        for (uint c = offsets[block]; c < offsets[block + 1]; c++) sum += left[row * p.cols + c] * right[row * p.cols + c];
        out[i] = sum;
    }
}

// cols: classes; extra: width; a: rows flagged by `scored`; b: the head stored transposed.
kernel void t_sampled_head(device const float* probabilities [[buffer(0)]], device const float* mean [[buffer(1)]], device const float* head [[buffer(2)]],
                           device const float* uniforms [[buffer(3)]], device const uint* scored [[buffer(4)]], device float* out [[buffer(5)]],
                           constant P& p [[buffer(6)]], uint group [[threadgroup_position_in_grid]],
                           uint groups [[threadgroups_per_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float shared[GROUP];
    threadgroup uint label;
    ROWS {
        if (p.a != 0 && scored[r] == 0) {
            for (uint h = t; h < p.extra; h += GROUP) out[(ulong)r * p.extra + h] = 0.0f;
            continue;
        }
        uint chosen = pick_label(probabilities + (ulong)r * p.cols, p.cols, uniforms[r], shared, &label, t);
        for (uint h = t; h < p.extra; h += GROUP) {
            ulong index = p.b ? (ulong)h * p.cols + chosen : (ulong)chosen * p.extra + h;
            out[(ulong)r * p.extra + h] = mean[(ulong)r * p.extra + h] - head[index];
        }
    }
}

kernel void t_argmax_rows(device const float* x [[buffer(0)]], device float* out [[buffer(1)]], constant P& p [[buffer(2)]],
                          uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float values[GROUP];
    threadgroup uint columns[GROUP];
    ROWS {
        device const float* z = x + (ulong)r * p.cols;
        float best = -INFINITY;
        uint at = p.cols;
        for (uint c = t; c < p.cols; c += GROUP) if (z[c] > best) { best = z[c]; at = c; }
        values[t] = best;
        columns[t] = at;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = GROUP / 2; s > 0; s >>= 1) {
            if (t < s && (values[t + s] > values[t] || (values[t + s] == values[t] && columns[t + s] < columns[t]))) {
                values[t] = values[t + s];
                columns[t] = columns[t + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (t == 0) out[r] = columns[0] == p.cols ? 0.0f : float(columns[0]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void t_fill_entries(device const uint* at [[buffer(0)]], device float* x [[buffer(1)]], constant P& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS x[at[i]] = p.alpha;
}

// Each row's softmax in place and its (log partition, negative entropy) in `out` (rows × 2); a row
// whose flag is zero (when `a`) becomes zero with zero statistics.
kernel void t_softmax_stats(device float* logits [[buffer(0)]], device const uint* flags [[buffer(1)]], device float* out [[buffer(2)]],
                            constant P& p [[buffer(3)]], uint group [[threadgroup_position_in_grid]],
                            uint groups [[threadgroups_per_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float shared[GROUP];
    ROWS {
        device float* z = logits + (ulong)r * p.cols;
        if (p.a != 0 && flags[r] == 0) {
            for (uint c = t; c < p.cols; c += GROUP) z[c] = 0.0f;
            if (t == 0) { out[2 * r] = 0.0f; out[2 * r + 1] = 0.0f; }
            continue;
        }
        float2 s = softmax_stats(z, p.cols, shared, t);
        float log_sum = log(s.y), partial = 0.0f;
        for (uint c = t; c < p.cols; c += GROUP) {
            float q = exp(z[c] - s.x) / s.y;
            if (q > 0.0f) partial += q * ((z[c] - s.x) - log_sum);
            z[c] = q;
        }
        float entropy = group_sum(partial, shared, t);
        if (t == 0) { out[2 * r] = s.x + log_sum; out[2 * r + 1] = entropy; }
    }
}

kernel void t_softmax_quadratic(device const float* logits [[buffer(0)]], device const float* tangent [[buffer(1)]], device float* out [[buffer(2)]],
                                constant P& p [[buffer(3)]], uint group [[threadgroup_position_in_grid]],
                                uint groups [[threadgroups_per_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float shared[GROUP];
    ROWS {
        device const float* z = logits + (ulong)r * p.cols;
        device const float* tr = tangent + (ulong)r * p.cols;
        float2 s = softmax_stats(z, p.cols, shared, t);
        float partial = 0.0f;
        for (uint c = t; c < p.cols; c += GROUP) partial += exp(z[c] - s.x) / s.y * tr[c];
        float mean = group_sum(partial, shared, t);
        partial = 0.0f;
        for (uint c = t; c < p.cols; c += GROUP) {
            float q = exp(z[c] - s.x) / s.y;
            partial += q * (tr[c] - mean) * (tr[c] - mean);
        }
        float total = group_sum(partial, shared, t);
        if (t == 0) out[r] = total;
    }
}

// alpha: rate; beta: ε; the moments' decays and bias corrections in `adam` (buffer 4).
struct Adam { float beta1; float beta2; float c1; float c2; };

kernel void t_adam(device float* w [[buffer(0)]], device float* m [[buffer(1)]], device float* v [[buffer(2)]], device const float* g [[buffer(3)]],
                   constant Adam& adam [[buffer(4)]], constant P& p [[buffer(5)]],
                   uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS {
        m[i] = adam.beta1 * m[i] + (1.0f - adam.beta1) * g[i];
        v[i] = adam.beta2 * v[i] + (1.0f - adam.beta2) * g[i] * g[i];
        w[i] = w[i] - p.alpha * (m[i] / adam.c1) / (sqrt(v[i] / adam.c2) + p.beta);
    }
}

// Whether (ka, ia) ranks before (kb, ib): larger key first, ties by column.
inline bool ranks_before(float ka, uint ia, float kb, uint ib) {
    return ka > kb || (ka == kb && ia < ib);
}

// The ranking key of a subcomponent: real size per description bit (unpaid ones first, unless
// they write nothing).
inline float ranking_key(float size, float bits) {
    if (bits > 0.0f) return size / bits;
    return size > 0.0f ? INFINITY : 0.0f;
}

// One threadgroup per row (striding): the row's ranking sorted in its group's scratch (`extra`
// wide, a power of two at least `cols`), then its best prefix and single flips by one thread,
// as the host's `select_sets`.
kernel void t_select_sets(device const float* a [[buffer(0)]], device const float* q [[buffer(1)]], device const float* bits [[buffer(2)]],
                          device const float* left [[buffer(3)]], device const float* weight [[buffer(4)]], device float* keys [[buffer(5)]],
                          device uint* order [[buffer(6)]], device float* sizes [[buffer(7)]], device float* mask [[buffer(8)]],
                          constant P& p [[buffer(9)]], uint group [[threadgroup_position_in_grid]],
                          uint groups [[threadgroups_per_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float shared[GROUP];
    uint width = p.extra;
    device float* key = keys + (ulong)group * width;
    device uint* idx = order + (ulong)group * width;
    device float* sr = sizes + (ulong)group * width;
    ROWS {
        device float* m = mask + (ulong)r * p.cols;
        float partial = 0.0f, off = 0.0f, on_bits = 0.0f;
        for (uint c = t; c < width; c += GROUP) {
            if (c < p.cols) {
                float s = fabs(a[(ulong)r * p.cols + c]) * q[(ulong)r * p.cols + c];
                sr[c] = s;
                partial += s;
                if (m[c] == 0.0f) off += s; else on_bits += bits[c];
                key[c] = ranking_key(s, bits[c]);
            } else {
                key[c] = -INFINITY;
            }
            idx[c] = c;
        }
        float all = group_sum(partial, shared, t);
        float held_off = group_sum(off, shared, t);
        float held_listed = group_sum(on_bits, shared, t);
        threadgroup_barrier(mem_flags::mem_device);
        for (uint k = 2; k <= width; k <<= 1) {
            for (uint j = k >> 1; j > 0; j >>= 1) {
                for (uint i = t; i < width; i += GROUP) {
                    uint l = i ^ j;
                    if (l > i) {
                        bool first = (i & k) == 0;
                        bool swap = first ? ranks_before(key[l], idx[l], key[i], idx[i]) : ranks_before(key[i], idx[i], key[l], idx[l]);
                        if (swap) {
                            float tk = key[i]; key[i] = key[l]; key[l] = tk;
                            uint ti = idx[i]; idx[i] = idx[l]; idx[l] = ti;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);
            }
        }
        if (t == 0) {
            float w = weight[r], lr = left[r];
            float listed = 0.0f, bound = lr + all;
            float best_code = w * bound * bound;
            uint best = 0;
            for (uint k = 0; k < p.cols; k++) {
                uint c = idx[k];
                listed += bits[c];
                bound -= sr[c];
                float code = listed + w * bound * bound;
                if (code < best_code) { best = k + 1; best_code = code; }
            }
            float held = lr + held_off;
            if (best_code < held_listed + w * held * held) {
                for (uint c = 0; c < p.cols; c++) m[c] = 0.0f;
                for (uint k = 0; k < best; k++) m[idx[k]] = 1.0f;
            }
            bound = lr;
            for (uint c = 0; c < p.cols; c++) if (m[c] == 0.0f) bound += sr[c];
            for (uint sweep = 0; sweep < p.cols; sweep++) {
                bool flipped = false;
                for (uint c = 0; c < p.cols; c++) {
                    bool on = m[c] == 1.0f;
                    float next = on ? bound + sr[c] : max(bound - sr[c], 0.0f);
                    float delta = (on ? -bits[c] : bits[c]) + w * (next * next - bound * bound);
                    if (delta < 0.0f) { m[c] = on ? 0.0f : 1.0f; bound = next; flipped = true; }
                }
                if (!flipped) break;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// Per row `N = Σ_c (1 − m) |z| q` into `norms`; `cot += N (1 − m) sign(z) q`, `coefficient = N (1 −
// m) |z| / q` (zero where `q` is).
kernel void t_box_charge(device const float* z [[buffer(0)]], device const float* mask [[buffer(1)]], device const float* q [[buffer(2)]],
                         device float* norms [[buffer(3)]], device float* cot [[buffer(4)]], device float* coefficient [[buffer(5)]],
                         constant P& p [[buffer(6)]], uint group [[threadgroup_position_in_grid]],
                         uint groups [[threadgroups_per_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float shared[GROUP];
    ROWS {
        ulong base = (ulong)r * p.cols;
        float partial = 0.0f;
        for (uint c = t; c < p.cols; c += GROUP) partial += (1.0f - mask[base + c]) * fabs(z[base + c]) * q[base + c];
        float n = group_sum(partial, shared, t);
        if (t == 0) norms[r] = n;
        for (uint c = t; c < p.cols; c += GROUP) {
            float off = 1.0f - mask[base + c], zc = z[base + c], qc = q[base + c];
            if (off != 0.0f && zc != 0.0f) cot[base + c] = cot[base + c] + n * off * (zc > 0.0f ? 1.0f : -1.0f) * qc;
            coefficient[base + c] = qc > 0.0f ? n * off * fabs(zc) / qc : 0.0f;
        }
    }
}

// The sparse code (`Device::code_rows`), one threadgroup per row, the CUDA kernel's steps in f32.
#define CNONE 0xffffffffu

struct CodeRow {
    uint C, B, W;
    bool unit;
    float kappa, tol, empty;
    device const float* z; device const float* w; device const float* K; device const uint* starts; device const float* bits;
    device float* lin; device float* dia; device float* qm; device float* oq; device float* m; device float* col; device float* tt;
    device float* on; device float* best; device float* inw; device float* key;
    device uint* work; device uint* idx;
};

inline uint code_pick(float v, uint i, bool last_max, threadgroup float* sv, threadgroup uint* si, uint t) {
    sv[t] = v; si[t] = i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = GROUP / 2; s > 0; s >>= 1) {
        if (t < s) {
            float a = sv[t], b = sv[t + s];
            uint ia = si[t], ib = si[t + s];
            bool take;
            if (ia == CNONE) take = true;
            else if (ib == CNONE) take = false;
            else if (last_max) take = b > a || (b == a && ib > ia);
            else take = b < a || (b == a && ib < ia);
            if (take) { sv[t] = b; si[t] = ib; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint r = si[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline void code_column(thread const CodeRow& r, uint b, uint t) {
    if (r.unit) {
        float zb = r.z[b];
        device const float* kr = r.K + (ulong)b * r.C;
        for (uint j = t; j < r.B; j += GROUP) r.col[j] = r.z[j] * (zb * kr[j]);
        threadgroup_barrier(mem_flags::mem_device);
        return;
    }
    uint s = r.starts[b], e = r.starts[b + 1];
    for (uint j = t; j < r.C; j += GROUP) {
        float acc = 0.0f;
        for (uint c = s; c < e; c++) {
            float zc = r.z[c];
            if (zc != 0.0f) acc += zc * r.K[(ulong)c * r.C + j];
        }
        r.tt[j] = acc;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint j = t; j < r.B; j += GROUP) {
        float acc = 0.0f;
        for (uint c = r.starts[j]; c < r.starts[j + 1]; c++) acc += r.z[c] * r.tt[c];
        r.col[j] = acc;
    }
    threadgroup_barrier(mem_flags::mem_device);
}

inline void code_add(thread const CodeRow& r, device float* q, float a, uint t) {
    for (uint j = t; j < r.B; j += GROUP) q[j] += a * r.col[j];
    threadgroup_barrier(mem_flags::mem_device);
}

inline float code_relax(thread const CodeRow& r, device const float* lo, device const float* hi, device const float* start, device const float* known, bool has_known,
                        threadgroup float* sv, threadgroup uint* si, threadgroup uint* nw, uint t) {
    uint B = r.B;
    for (uint b = t; b < B; b += GROUP) {
        r.m[b] = min(max(start[b], lo[b]), hi[b]);
        r.qm[b] = has_known ? known[b] : 0.0f;
        r.inw[b] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint b = 0; b < B; b++) {
        float moved = has_known ? r.m[b] - start[b] : r.m[b];
        if (moved != 0.0f) { code_column(r, b, t); code_add(r, r.qm, moved, t); }
    }
    if (t == 0) {
        uint n = 0;
        for (uint b = 0; b < B; b++) {
            if (r.m[b] != 0.0f && lo[b] < hi[b]) { r.work[n++] = b; r.inw[b] = 1.0f; }
        }
        nw[0] = n;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (;;) {
        for (int sweep = 0; sweep < 1000; sweep++) {
            float moved = 0.0f;
            uint n = nw[0];
            for (uint k = 0; k < n; k++) {
                uint b = r.work[k];
                float g = r.lin[b] + 2.0f * r.kappa * r.qm[b];
                float curvature = 2.0f * r.kappa * r.dia[b];
                float mb = r.m[b];
                float next = curvature > 0.0f ? min(max(mb - g / curvature, lo[b]), hi[b]) : (g > 0.0f ? lo[b] : (g < 0.0f ? hi[b] : mb));
                float step = next - mb;
                threadgroup_barrier(mem_flags::mem_device);
                if (step != 0.0f) {
                    if (t == 0) r.m[b] = next;
                    code_column(r, b, t);
                    code_add(r, r.qm, step, t);
                    moved = max(moved, fabs(step));
                }
            }
            if (moved <= r.tol) break;
        }
        for (uint b = t; b < r.W; b += GROUP) {
            float k = -1.0f;
            if (b < B && lo[b] < hi[b] && r.inw[b] == 0.0f) {
                float g = r.lin[b] + 2.0f * r.kappa * r.qm[b];
                if ((g < 0.0f && r.m[b] < hi[b]) || (g > 0.0f && r.m[b] > lo[b])) k = fabs(g);
            }
            r.key[b] = k;
            r.idx[b] = b;
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint k = 2; k <= r.W; k <<= 1) {
            for (uint j = k >> 1; j > 0; j >>= 1) {
                for (uint i = t; i < r.W; i += GROUP) {
                    uint l = i ^ j;
                    if (l > i) {
                        bool before_li = r.key[l] > r.key[i] || (r.key[l] == r.key[i] && r.idx[l] < r.idx[i]);
                        bool before_il = r.key[i] > r.key[l] || (r.key[i] == r.key[l] && r.idx[i] < r.idx[l]);
                        bool swap = (i & k) == 0 ? before_li : before_il;
                        if (swap) {
                            float tk = r.key[i]; r.key[i] = r.key[l]; r.key[l] = tk;
                            uint ti = r.idx[i]; r.idx[i] = r.idx[l]; r.idx[l] = ti;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_device);
            }
        }
        if (t == 0) {
            uint count = 0;
            while (count < B && r.key[count] >= 0.0f) count++;
            uint stop = count == 0 ? 1u : 0u;
            if (count > 0) {
                uint n = nw[0];
                uint room = n > 16 ? n : 16;
                uint take = count < room ? count : room;
                for (uint i = 0; i < take; i++) { uint b = r.idx[i]; r.work[n++] = b; r.inw[b] = 1.0f; }
                nw[0] = n;
            }
            si[0] = stop;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        uint stop = si[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (stop != 0) break;
    }
    float pv = 0.0f, pq = 0.0f, ps = 0.0f;
    for (uint b = t; b < B; b += GROUP) {
        float mb = r.m[b], g = r.lin[b] + 2.0f * r.kappa * r.qm[b];
        pv += mb * r.lin[b];
        pq += mb * r.qm[b];
        ps += min(g * (lo[b] - mb), g * (hi[b] - mb));
    }
    float value = r.empty + group_sum(pv, sv, t) + r.kappa * group_sum(pq, sv, t);
    float slack = group_sum(ps, sv, t);
    return value + min(slack, 0.0f);
}

inline float code_round(thread const CodeRow& r, device const float* lo, device const float* hi, threadgroup float* sv, threadgroup uint* si, uint t) {
    uint B = r.B;
    for (uint b = t; b < B; b += GROUP) {
        r.on[b] = (lo[b] == hi[b] ? hi[b] > 0.5f : r.m[b] > 0.5f) ? 1.0f : 0.0f;
        r.oq[b] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint b = 0; b < B; b++) {
        if (r.on[b] == 1.0f) { code_column(r, b, t); code_add(r, r.oq, 1.0f, t); }
    }
    for (;;) {
        float bv = INFINITY;
        uint bi = CNONE;
        for (uint b = t; b < B; b += GROUP) {
            if (!(lo[b] < hi[b])) continue;
            float delta = r.on[b] == 1.0f ? -r.lin[b] + r.kappa * (r.dia[b] - 2.0f * r.oq[b]) : r.lin[b] + r.kappa * (r.dia[b] + 2.0f * r.oq[b]);
            if (delta < 0.0f && (bi == CNONE || delta < bv)) { bv = delta; bi = b; }
        }
        uint b = code_pick(bv, bi, false, sv, si, t);
        if (b == CNONE) break;
        float sign = r.on[b] == 1.0f ? -1.0f : 1.0f;
        threadgroup_barrier(mem_flags::mem_device);
        if (t == 0) r.on[b] = r.on[b] == 1.0f ? 0.0f : 1.0f;
        code_column(r, b, t);
        code_add(r, r.oq, sign, t);
    }
    float pv = 0.0f, pq = 0.0f;
    for (uint b = t; b < B; b += GROUP) {
        pv += r.on[b] * r.lin[b];
        pq += r.on[b] * r.oq[b];
    }
    return r.empty + group_sum(pv, sv, t) + r.kappa * group_sum(pq, sv, t);
}

inline void code_copy(device float* to, device const float* from, uint n, uint t) {
    for (uint i = t; i < n; i += GROUP) to[i] = from[i];
    threadgroup_barrier(mem_flags::mem_device);
}

// `p`: rows, cols (pieces), extra (the sort's width), a (blocks), b (nodes), n (1: every block one
// piece; 2: warm starts), alpha (κ), beta (the relaxation's tolerance).
kernel void t_code_rows(device const float* z [[buffer(0)]], device const float* w [[buffer(1)]], device const float* yfy [[buffer(2)]],
                        device const float* K [[buffer(3)]], device const uint* starts [[buffer(4)]], device const float* bits [[buffer(5)]],
                        device const float* warm [[buffer(6)]], device float* scratch [[buffer(7)]], device uint* iscratch [[buffer(8)]],
                        device float* on_out [[buffer(9)]], device float* upper_out [[buffer(10)]], device float* lower_out [[buffer(11)]],
                        constant P& p [[buffer(12)]], uint group [[threadgroup_position_in_grid]],
                        uint groups [[threadgroups_per_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float sv[GROUP];
    threadgroup uint si[GROUP];
    threadgroup uint nw[1];
    uint C = p.cols, B = p.a, W = p.extra, nodes = p.b, cap = nodes + 1;
    ulong slot_len = 15ul * B + C + W + (ulong)cap * (4ul * B + 1ul);
    ulong index_len = (ulong)B + W;
    device float* base = scratch + (ulong)group * slot_len;
    device uint* ibase = iscratch + (ulong)group * index_len;
    CodeRow r;
    r.C = C; r.B = B; r.W = W; r.unit = (p.n & 1u) != 0; r.kappa = p.alpha; r.tol = p.beta;
    r.K = K; r.starts = starts; r.bits = bits;
    r.lin = base; r.dia = base + B; r.qm = base + 2 * B; r.oq = base + 3 * B; r.m = base + 4 * B; r.col = base + 5 * B; r.on = base + 6 * B;
    r.best = base + 7 * B; r.inw = base + 8 * B;
    device float* lo = base + 9 * B; device float* hi = base + 10 * B; device float* point = base + 11 * B; device float* pqm = base + 12 * B;
    device float* clo = base + 13 * B; device float* chi = base + 14 * B;
    r.tt = base + 15 * B; r.key = base + 15 * B + C;
    device float* open = base + 15 * B + C + W;
    device float* open_lower = open + (ulong)cap * 4 * B;
    r.work = ibase; r.idx = ibase + B;
    bool warmed = (p.n & 2u) != 0;
    for (uint row = group; row < p.rows; row += groups) {
        r.z = z + (ulong)row * C;
        r.w = w + (ulong)row * C;
        r.empty = p.alpha * yfy[row];
        for (uint b = t; b < B; b += GROUP) {
            uint s0 = starts[b], e0 = starts[b + 1];
            float acc = 0.0f;
            for (uint c = s0; c < e0; c++) acc += r.z[c] * r.w[c];
            r.lin[b] = bits[b] - 2.0f * p.alpha * acc;
            float d = 0.0f;
            for (uint c = s0; c < e0; c++) {
                float inner = 0.0f;
                for (uint c2 = s0; c2 < e0; c2++) inner += r.z[c] * r.z[c2] * K[(ulong)c * C + c2];
                d += inner;
            }
            r.dia[b] = max(d, 0.0f);
            lo[b] = 0.0f;
            hi[b] = 1.0f;
            point[b] = warmed ? warm[(ulong)row * B + b] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_device);
        float root = code_relax(r, lo, hi, point, pqm, false, sv, si, nw, t);
        float upper = code_round(r, lo, hi, sv, si, t);
        code_copy(r.best, r.on, B, t);
        float lower = min(root, upper);
        if (!(upper - root <= 1.0f || nodes == 0)) {
            uint count = 1, explored = 0;
            code_copy(open, lo, B, t); code_copy(open + B, hi, B, t); code_copy(open + 2 * B, r.m, B, t); code_copy(open + 3 * B, r.qm, B, t);
            if (t == 0) open_lower[0] = root;
            threadgroup_barrier(mem_flags::mem_device);
            float floor_bound = INFINITY;
            for (;;) {
                if (count == 0) break;
                uint index = 0;
                for (uint i = 1; i < count; i++) if (open_lower[i] < open_lower[index]) index = i;
                float node = open_lower[index];
                if (node >= upper - 1.0f || explored >= nodes) break;
                device float* entry = open + (ulong)index * 4 * B;
                code_copy(lo, entry, B, t); code_copy(hi, entry + B, B, t); code_copy(point, entry + 2 * B, B, t); code_copy(pqm, entry + 3 * B, B, t);
                count--;
                if (index != count) {
                    device float* last = open + (ulong)count * 4 * B;
                    code_copy(entry, last, 4 * B, t);
                    if (t == 0) open_lower[index] = open_lower[count];
                    threadgroup_barrier(mem_flags::mem_device);
                }
                explored++;
                float bv = -INFINITY;
                uint bi = CNONE;
                for (uint b = t; b < B; b += GROUP) {
                    if (!(lo[b] < hi[b])) continue;
                    float f = 0.5f - fabs(point[b] - 0.5f);
                    if (bi == CNONE || f >= bv) { bv = f; bi = b; }
                }
                uint j = code_pick(bv, bi, true, sv, si, t);
                if (j == CNONE || point[j] == 0.0f || point[j] == 1.0f) { floor_bound = min(floor_bound, node); continue; }
                for (int fixed = 0; fixed < 2; fixed++) {
                    code_copy(clo, lo, B, t); code_copy(chi, hi, B, t);
                    if (t == 0) { clo[j] = (float)fixed; chi[j] = (float)fixed; }
                    threadgroup_barrier(mem_flags::mem_device);
                    float child = code_relax(r, clo, chi, point, pqm, true, sv, si, nw, t);
                    float value = code_round(r, clo, chi, sv, si, t);
                    if (value < upper) { code_copy(r.best, r.on, B, t); upper = value; }
                    if (child < upper - 1.0f) {
                        device float* slot = open + (ulong)count * 4 * B;
                        code_copy(slot, clo, B, t); code_copy(slot + B, chi, B, t); code_copy(slot + 2 * B, r.m, B, t); code_copy(slot + 3 * B, r.qm, B, t);
                        if (t == 0) open_lower[count] = child;
                        threadgroup_barrier(mem_flags::mem_device);
                        count++;
                    } else {
                        floor_bound = min(floor_bound, child);
                    }
                }
            }
            lower = floor_bound;
            for (uint i = 0; i < count; i++) lower = min(lower, open_lower[i]);
            lower = min(lower, upper);
        }
        for (uint b = t; b < B; b += GROUP) on_out[(ulong)row * B + b] = r.best[b];
        if (t == 0) { upper_out[row] = upper; lower_out[row] = lower; }
        threadgroup_barrier(mem_flags::mem_device);
    }
}
// The standard normal draw `index` of (key, stream): Philox4x32-10, then Box–Muller
// (`posterior_normal` on the host; indices fit 32 bits here).
inline float posterior_normal(uint2 key, uint2 stream, uint index) {
    uint c0 = index, c1 = 0u, c2 = stream.x, c3 = stream.y, k0 = key.x, k1 = key.y;
    for (int round = 0; round < 10; ++round) {
        if (round > 0) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
        uint hi0 = mulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
        uint hi1 = mulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
        c0 = hi1 ^ c1 ^ k0; c1 = lo1; c2 = hi0 ^ c3 ^ k1; c3 = lo0;
    }
    float u1 = (float(c0) + 0.5f) * 2.3283064e-10f, u2 = float(c1) * 2.3283064e-10f;
    return sqrt(-2.0f * log(u1)) * cos(6.28318530717958647692f * u2);
}

// The parameters of the posterior kernels (Rust `Posterior`).
struct Posterior { uint n; uint count; uint2 key; uint2 stream; float scale; float mean_rate; float log_sd_rate; float beta1; float beta2; float epsilon; float c1; float c2; };

kernel void t_reparameterize(device const float* mean [[buffer(0)]], device const float* log_sd [[buffer(1)]], device float* theta [[buffer(2)]],
                             constant Posterior& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) theta[i] = mean[i] + exp(log_sd[i]) * posterior_normal(p.key, p.stream, i);
}

// Adds a live entry's (1, μ² + σ², 2s) to its group's row of `sums`: once per SIMD group when every
// live lane's group is the first lane's, else per lane. Every lane calls it.
inline void group_add(device float* sums, uint g, bool live, float a, float b, float c) {
    uint g0 = simd_broadcast_first(g);
    if (simd_all(!live || g == g0)) {
        a = simd_sum(a); b = simd_sum(b); c = simd_sum(c);
        if (simd_is_first() && a > 0.0f) {
            atomic_fetch_add_explicit((device atomic_float*)(sums + 3 * g0), a, memory_order_relaxed);
            atomic_fetch_add_explicit((device atomic_float*)(sums + 3 * g0 + 1), b, memory_order_relaxed);
            atomic_fetch_add_explicit((device atomic_float*)(sums + 3 * g0 + 2), c, memory_order_relaxed);
        }
    } else if (live) {
        atomic_fetch_add_explicit((device atomic_float*)(sums + 3 * g), a, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)(sums + 3 * g + 1), b, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)(sums + 3 * g + 2), c, memory_order_relaxed);
    }
}

// One thread per entry (the dispatch covers every entry once), so every lane reaches `group_add`.
kernel void t_posterior_adam(device const float* gradient [[buffer(0)]], device const uint* groups [[buffer(1)]], device const float* variance [[buffer(2)]],
                             device float* mean [[buffer(3)]], device float* log_sd [[buffer(4)]], device float* mm [[buffer(5)]], device float* mv [[buffer(6)]],
                             device float* sm [[buffer(7)]], device float* sv [[buffer(8)]], device float* sums [[buffer(9)]],
                             constant Posterior& p [[buffer(10)]], uint i [[thread_position_in_grid]]) {
    uint g = i < p.n ? groups[i] : 0u;
    bool live = i < p.n && g < p.count && log_sd[i] != -INFINITY;
    float a = 0.0f, b = 0.0f, c = 0.0f;
    if (live) {
        float v = variance[g], mu = mean[i], s = log_sd[i], sd = exp(s);
        float e = posterior_normal(p.key, p.stream, i), gi = p.scale * gradient[i];
        float gm = gi + mu / v, gs = gi * e * sd + sd * sd / v - 1.0f;
        float m1 = p.beta1 * mm[i] + (1.0f - p.beta1) * gm, v1 = p.beta2 * mv[i] + (1.0f - p.beta2) * gm * gm;
        float m2 = p.beta1 * sm[i] + (1.0f - p.beta1) * gs, v2 = p.beta2 * sv[i] + (1.0f - p.beta2) * gs * gs;
        mm[i] = m1; mv[i] = v1; sm[i] = m2; sv[i] = v2;
        mu -= p.mean_rate * (m1 / p.c1) / (sqrt(v1 / p.c2) + p.epsilon);
        s -= p.log_sd_rate * (m2 / p.c1) / (sqrt(v2 / p.c2) + p.epsilon);
        mean[i] = mu; log_sd[i] = s;
        a = 1.0f; b = mu * mu + exp(2.0f * s); c = 2.0f * s;
    }
    group_add(sums, g, live, a, b, c);
}

kernel void t_group_moments(device const float* mean [[buffer(0)]], device const float* log_sd [[buffer(1)]], device const uint* groups [[buffer(2)]],
                            device float* sums [[buffer(3)]], constant Posterior& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint g = i < p.n ? groups[i] : 0u;
    bool live = i < p.n && g < p.count && log_sd[i] != -INFINITY;
    float a = live ? 1.0f : 0.0f;
    float b = live ? mean[i] * mean[i] + exp(2.0f * log_sd[i]) : 0.0f;
    float c = live ? 2.0f * log_sd[i] : 0.0f;
    group_add(sums, g, live, a, b, c);
}

kernel void t_group_divergence(device float* sums [[buffer(0)]], device float* variance [[buffer(1)]], device float* divergence [[buffer(2)]],
                               constant P& p [[buffer(3)]], uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS {
        float count = sums[3 * i], second = sums[3 * i + 1], log_variance = sums[3 * i + 2];
        float v = count > 0.0f ? second / count : 0.0f;
        variance[i] = v;
        divergence[i] = count > 0.0f ? 0.5f * (count * log(v) - log_variance) : 0.0f;
        sums[3 * i] = 0.0f; sums[3 * i + 1] = 0.0f; sums[3 * i + 2] = 0.0f;
    }
}
"#;

    const NAMES: &[&str] = &[
        "t_copy",
        "t_copy_columns",
        "t_scale",
        "t_axpy",
        "t_hadamard",
        "t_add_row",
        "t_scale_columns",
        "t_gather_rows",
        "t_laws",
        "t_rms",
        "t_rotate",
        "t_heads",
        "t_softmax_rows",
        "t_softmax_backward",
        "t_kl_rows",
        "t_sampled_cotangent",
        "t_block_products",
        "t_sampled_head",
        "t_softmax_quadratic",
        "t_argmax_rows",
        "t_fill_entries",
        "t_adam",
        "t_softmax_stats",
        "t_reparameterize",
        "t_posterior_adam",
        "t_group_moments",
        "t_group_divergence",
        "t_select_sets",
        "t_box_charge",
        "t_code_rows",
    ];

    /// The parameters of every kernel (MSL `P`).
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct P {
        n: u32,
        rows: u32,
        cols: u32,
        extra: u32,
        a: u32,
        b: u32,
        alpha: f32,
        beta: f32,
        c: u32,
        d: u32,
    }

    /// The parameters of the posterior kernels (MSL `Posterior`).
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Posterior {
        n: u32,
        count: u32,
        key: [u32; 2],
        stream: [u32; 2],
        scale: f32,
        mean_rate: f32,
        log_sd_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        c1: f32,
        c2: f32,
    }

    /// A 64-bit counter word as MSL's `uint2` (low half first).
    fn halves(x: u64) -> [u32; 2] {
        [x as u32, (x >> 32) as u32]
    }

    fn u32_of(n: usize) -> Result<u32, GpuError> {
        u32::try_from(n).map_err(|_| shape(format!("{n} exceeds the Apple GPU kernels' 32-bit indices")))
    }

    fn buffer(t: &Tensor) -> Result<&Buffer, GpuError> {
        match &t.data {
            Data::Metal(b) => Ok(b),
            _ => Err(foreign()),
        }
    }

    fn index_buffer(i: &Indices) -> Result<&Buffer, GpuError> {
        match &i.data {
            IndexData::Metal(b, _) => Ok(b),
            _ => Err(foreign()),
        }
    }

    fn whole(b: &Buffer) -> (&Buffer, usize) {
        (b, 0)
    }

    /// Threadgroups for `n` values, one per thread.
    fn spread(n: usize) -> usize {
        n.div_ceil(GROUP)
    }

    /// The Apple GPU's tensor engine: its stream and a name for reports.
    pub(super) struct Engine {
        pub(super) name: String,
        pub(super) stream: Stream,
        /// The row flags of a call that scores every row (never read).
        every_row: Buffer,
    }

    impl Engine {
        /// The process's engine on the device Metal resolved, built once.
        pub(super) fn shared(runtime: &'static MetalRuntime) -> Result<std::sync::Arc<super::Backend>, GpuError> {
            static ENGINE: std::sync::OnceLock<Result<std::sync::Arc<super::Backend>, GpuError>> = std::sync::OnceLock::new();
            ENGINE
                .get_or_init(|| {
                    let stream = Stream::new(&runtime.context, KERNELS, NAMES)?;
                    let every_row = stream.alloc(1)?;
                    let name = format!("{} (Metal, f32)", stream.device_name());
                    Ok(std::sync::Arc::new(super::Backend::Metal(Self { name, stream, every_row })))
                })
                .clone()
        }

        fn tensor(&self, rows: usize, cols: usize) -> Result<Tensor, GpuError> {
            Ok(Tensor { rows, cols, data: Data::Metal(self.stream.alloc(rows * cols)?) })
        }

        fn elements(&self, kernel: &'static str, buffers: &[(&Buffer, usize)], n: usize, mut p: P) -> Result<(), GpuError> {
            p.n = u32_of(n)?;
            self.stream.dispatch(kernel, buffers, &p, spread(n))
        }

        fn rows(&self, kernel: &'static str, buffers: &[(&Buffer, usize)], rows: usize, cols: usize, mut p: P) -> Result<(), GpuError> {
            u32_of(rows.saturating_mul(cols))?;
            p.rows = u32_of(rows)?;
            p.cols = u32_of(cols)?;
            self.stream.dispatch(kernel, buffers, &p, rows)
        }

        pub(super) fn copy_range(&self, source: &Buffer, lo: usize, hi: usize) -> Result<Buffer, GpuError> {
            let out = self.stream.alloc(hi - lo)?;
            self.elements("t_copy", &[(source, lo), whole(&out)], hi - lo, P::default())?;
            Ok(out)
        }

        pub(super) fn write_range(&self, target: &Buffer, lo: usize, part: &Buffer, n: usize) -> Result<(), GpuError> {
            self.elements("t_copy", &[whole(part), (target, lo)], n, P::default())
        }

        pub(super) fn set_columns(&self, target: &Buffer, part: &Buffer, (cols, width, start): (usize, usize, usize), n: usize) -> Result<(), GpuError> {
            u32_of((n / width).saturating_mul(cols))?;
            let p = P { cols: u32_of(cols)?, extra: u32_of(width)?, a: u32_of(start)?, b: 1, ..P::default() };
            self.elements("t_copy_columns", &[whole(part), whole(target)], n, p)
        }

        pub(super) fn columns_of(&self, source: &Tensor, start: usize, width: usize, n: usize) -> Result<Data, GpuError> {
            u32_of(source.len())?;
            let out = self.stream.alloc(n)?;
            let p = P { cols: u32_of(source.cols)?, extra: u32_of(width)?, a: u32_of(start)?, ..P::default() };
            self.elements("t_copy_columns", &[whole(buffer(source)?), whole(&out)], n, p)?;
            Ok(Data::Metal(out))
        }

        pub(super) fn gemm(
            &self,
            batch: usize,
            (m, n, k): (usize, usize, usize),
            (alpha, beta): (f64, f64),
            (a, ta): (&Tensor, Op),
            (b, tb): (&Tensor, Op),
            c: &mut Tensor,
            arithmetic: Arithmetic,
        ) -> Result<(), GpuError> {
            if arithmetic == Arithmetic::F64 {
                return Err(GpuError::NoDeviceKernel {
                    reason: format!("{} has no float64: a float64 product runs on the host", self.name),
                });
            }
            if m == 0 || n == 0 {
                return Ok(());
            }
            if k == 0 {
                // An empty sum: `c ← β c`.
                let p = P { alpha: beta as f32, ..P::default() };
                return self.elements("t_scale", &[whole(buffer(c)?)], c.len(), p);
            }
            // TF32 asks for no more than f32 delivers: both run as f32.
            let left = Matrix { buffer: buffer(a)?, rows: a.rows / batch, cols: a.cols, transposed: ta == Op::T };
            let right = Matrix { buffer: buffer(b)?, rows: b.rows / batch, cols: b.cols, transposed: tb == Op::T };
            let result = Matrix { buffer: buffer(c)?, rows: c.rows / batch, cols: c.cols, transposed: false };
            self.stream.gemm(batch, left, right, result, (m, n, k), (alpha, beta))
        }

        pub(super) fn axpy(&self, y: &mut Tensor, alpha: f64, x: &Tensor) -> Result<(), GpuError> {
            self.elements("t_axpy", &[whole(buffer(x)?), whole(buffer(y)?)], y.len(), P { alpha: alpha as f32, ..P::default() })
        }

        pub(super) fn hadamard(&self, out: &mut Tensor, a: &Tensor, b: &Tensor, accumulate: bool) -> Result<(), GpuError> {
            let p = P { a: u32::from(accumulate), ..P::default() };
            self.elements("t_hadamard", &[whole(buffer(a)?), whole(buffer(b)?), whole(buffer(out)?)], out.len(), p)
        }

        pub(super) fn add_row(&self, x: &mut Tensor, alpha: f64, row: &Tensor) -> Result<(), GpuError> {
            let p = P { cols: u32_of(x.cols)?, alpha: alpha as f32, ..P::default() };
            self.elements("t_add_row", &[whole(buffer(row)?), whole(buffer(x)?)], x.len(), p)
        }

        pub(super) fn scale_columns(&self, out: &mut Tensor, x: &Tensor, d: &Tensor, accumulate: bool) -> Result<(), GpuError> {
            let p = P { cols: u32_of(out.cols)?, a: u32::from(accumulate), ..P::default() };
            self.elements("t_scale_columns", &[whole(buffer(x)?), whole(buffer(d)?), whole(buffer(out)?)], out.len(), p)
        }

        pub(super) fn gather_rows(&self, table: &Tensor, ids: &Indices) -> Result<Tensor, GpuError> {
            let IndexData::Metal(_, values) = &ids.data else { return Err(foreign()) };
            if let Some(id) = values.iter().find(|id| **id as usize >= table.rows) {
                return Err(shape(format!("row {id} of a {}-row table", table.rows)));
            }
            let out = self.tensor(ids.len, table.cols)?;
            let p = P { cols: u32_of(table.cols)?, ..P::default() };
            self.elements("t_gather_rows", &[whole(buffer(table)?), whole(index_buffer(ids)?), whole(buffer(&out)?)], out.len(), p)?;
            Ok(out)
        }

        pub(super) fn laws(&self, x: &Tensor, g: Option<&Tensor>, codes: &Indices, c: f64) -> Result<Tensor, GpuError> {
            let out = self.tensor(x.rows, x.cols)?;
            let p = P { cols: u32_of(x.cols)?, a: u32::from(g.is_some()), alpha: c as f32, ..P::default() };
            let buffers = [whole(buffer(x)?), whole(buffer(g.unwrap_or(x))?), whole(index_buffer(codes)?), whole(buffer(&out)?)];
            self.elements("t_laws", &buffers, x.len(), p)?;
            Ok(out)
        }

        pub(super) fn rms(&self, mode: RmsMode, x: &Tensor, g: Option<&Tensor>, epsilon: f64) -> Result<Tensor, GpuError> {
            let out = self.tensor(x.rows, x.cols)?;
            let code = match mode {
                RmsMode::Value => 0,
                RmsMode::Backward => 1,
                RmsMode::Tangent => 2,
            };
            let p = P { a: code, alpha: epsilon as f32, ..P::default() };
            self.rows("t_rms", &[whole(buffer(x)?), whole(buffer(g.unwrap_or(x))?), whole(buffer(&out)?)], x.rows, x.cols, p)?;
            Ok(out)
        }

        pub(super) fn rotate(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, half_split: bool, inverse: bool) -> Result<Tensor, GpuError> {
            let out = Tensor { rows: x.rows, cols: x.cols, data: Data::Metal(self.copy_range(buffer(x)?, 0, x.len())?) };
            let p = P { cols: u32_of(x.cols)?, extra: u32_of(cos.cols)?, a: u32::from(half_split), alpha: if inverse { -1.0 } else { 1.0 }, ..P::default() };
            let buffers = [whole(buffer(x)?), whole(buffer(cos)?), whole(buffer(sin)?), whole(buffer(&out)?)];
            self.elements("t_rotate", &buffers, x.rows * cos.cols, p)?;
            Ok(out)
        }

        pub(super) fn heads(
            &self,
            x: &Tensor,
            out: &mut Tensor,
            (start, heads, width, length, planes): (usize, usize, usize, usize, usize),
            turn: Option<(&Tensor, &Tensor, bool)>,
            (half_split, inverse, merge): (bool, bool, bool),
        ) -> Result<(), GpuError> {
            let rows = if merge { out.rows } else { x.rows };
            let cols = if merge { out.cols } else { x.cols };
            let flags = u32::from(half_split) | (u32::from(merge) << 1);
            let p = P { rows: u32_of(length)?, cols: u32_of(cols)?, extra: u32_of(start)?, a: u32_of(heads)?, b: u32_of(width)?, c: u32_of(planes)?, d: flags, alpha: if inverse { -1.0 } else { 1.0 }, ..P::default() };
            let (cos, sin) = turn.map_or((x, x), |(c, s, _)| (c, s));
            let buffers = [whole(buffer(x)?), whole(buffer(cos)?), whole(buffer(sin)?), whole(buffer(out)?)];
            self.elements("t_heads", &buffers, rows * heads * width, p)
        }

        pub(super) fn softmax_rows(&self, scores: &mut Tensor, causal: bool, start: usize, period: usize) -> Result<(), GpuError> {
            let p = P { a: u32::from(causal), extra: u32_of(start)?, b: u32_of(period)?, ..P::default() };
            self.rows("t_softmax_rows", &[whole(buffer(scores)?)], scores.rows, scores.cols, p)
        }

        pub(super) fn softmax_backward(&self, alpha: &Tensor, d: &Tensor) -> Result<Tensor, GpuError> {
            let out = self.tensor(alpha.rows, alpha.cols)?;
            self.rows("t_softmax_backward", &[whole(buffer(alpha)?), whole(buffer(d)?), whole(buffer(&out)?)], alpha.rows, alpha.cols, P::default())?;
            Ok(out)
        }

        fn flags<'a>(&'a self, scored: Option<&'a Indices>) -> Result<(&'a Buffer, u32), GpuError> {
            match scored {
                Some(s) => Ok((index_buffer(s)?, 1)),
                None => Ok((&self.every_row, 0)),
            }
        }

        pub(super) fn kl_rows(&self, target: &Tensor, logits: &mut Tensor, scored: Option<&Indices>, gradient: bool) -> Result<Vec<f64>, GpuError> {
            let kl = self.stream.alloc(logits.rows)?;
            let (flags, use_flags) = self.flags(scored)?;
            let p = P { a: use_flags, b: u32::from(gradient), ..P::default() };
            let buffers = [whole(buffer(target)?), whole(buffer(logits)?), whole(flags), whole(&kl)];
            self.rows("t_kl_rows", &buffers, logits.rows, logits.cols, p)?;
            Ok(self.stream.read::<f32>(&kl)?.into_iter().take(logits.rows).map(f64::from).collect())
        }

        pub(super) fn softmax_stats_rows(&self, logits: &mut Tensor, scored: Option<&Indices>) -> Result<Vec<[f64; 2]>, GpuError> {
            let (flags, use_flags) = self.flags(scored)?;
            let out = self.stream.alloc(2 * logits.rows)?;
            let p = P { a: use_flags, ..P::default() };
            self.rows("t_softmax_stats", &[whole(buffer(logits)?), whole(flags), whole(&out)], logits.rows, logits.cols, p)?;
            let values = self.stream.read::<f32>(&out)?;
            let stats: Vec<[f64; 2]> = values.chunks_exact(2).take(logits.rows).map(|v| [f64::from(v[0]), f64::from(v[1])]).collect();
            if stats.iter().flatten().any(|v| !v.is_finite()) {
                return Err(shape("nonfinite softmax statistics".into()));
            }
            Ok(stats)
        }

        pub(super) fn sampled_cotangent(&self, logits: &mut Tensor, uniforms: &Tensor, scored: Option<&Indices>) -> Result<(), GpuError> {
            let (flags, use_flags) = self.flags(scored)?;
            let p = P { a: use_flags, ..P::default() };
            self.rows("t_sampled_cotangent", &[whole(buffer(logits)?), whole(buffer(uniforms)?), whole(flags)], logits.rows, logits.cols, p)
        }

        pub(super) fn block_products(&self, left: &Tensor, right: &Tensor, blocks: &ColumnBlocks) -> Result<Tensor, GpuError> {
            let out = self.tensor(left.rows, blocks.len())?;
            let p = P { cols: u32_of(left.cols)?, extra: u32_of(blocks.len())?, ..P::default() };
            let buffers = [whole(buffer(left)?), whole(buffer(right)?), whole(index_buffer(&blocks.offsets)?), whole(buffer(&out)?)];
            self.elements("t_block_products", &buffers, out.len(), p)?;
            Ok(out)
        }

        pub(super) fn sampled_head_cotangent(
            &self,
            probabilities: &Tensor,
            mean: &Tensor,
            head: &Tensor,
            transposed: bool,
            uniforms: &Tensor,
            scored: Option<&Indices>,
        ) -> Result<Tensor, GpuError> {
            let out = self.tensor(mean.rows, mean.cols)?;
            let (flags, use_flags) = self.flags(scored)?;
            let p = P { extra: u32_of(mean.cols)?, a: use_flags, b: u32::from(transposed), ..P::default() };
            let buffers =
                [whole(buffer(probabilities)?), whole(buffer(mean)?), whole(buffer(head)?), whole(buffer(uniforms)?), whole(flags), whole(buffer(&out)?)];
            u32_of(head.len())?;
            self.rows("t_sampled_head", &buffers, mean.rows, probabilities.cols, p)?;
            Ok(out)
        }

        pub(super) fn argmax_rows(&self, t: &Tensor) -> Result<Vec<usize>, GpuError> {
            let out = self.stream.alloc(t.rows)?;
            self.rows("t_argmax_rows", &[whole(buffer(t)?), whole(&out)], t.rows, t.cols, P::default())?;
            Ok(self.stream.read::<f32>(&out)?.into_iter().take(t.rows).map(|v| v as usize).collect())
        }

        pub(super) fn fill_entries(&self, t: &mut Tensor, at: &Indices, value: f64) -> Result<(), GpuError> {
            if at.len == 0 {
                return Ok(());
            }
            let p = P { alpha: value as f32, ..P::default() };
            self.elements("t_fill_entries", &[whole(index_buffer(at)?), whole(buffer(t)?)], at.len, p)
        }

        pub(super) fn softmax_quadratic(&self, logits: &Tensor, tangent: &Tensor) -> Result<Vec<f64>, GpuError> {
            let out = self.stream.alloc(logits.rows)?;
            self.rows("t_softmax_quadratic", &[whole(buffer(logits)?), whole(buffer(tangent)?), whole(&out)], logits.rows, logits.cols, P::default())?;
            Ok(self.stream.read::<f32>(&out)?.into_iter().take(logits.rows).map(f64::from).collect())
        }

        pub(super) fn adam(
            &self,
            w: &mut Tensor,
            (m, v): (&mut Tensor, &mut Tensor),
            g: &Tensor,
            (rate, beta1, beta2, epsilon): (f64, f64, f64, f64),
            (c1, c2): (f64, f64),
        ) -> Result<(), GpuError> {
            // The decays and bias corrections, MSL `Adam`.
            let constants = self.stream.upload(&[beta1 as f32, beta2 as f32, c1 as f32, c2 as f32])?;
            let p = P { alpha: rate as f32, beta: epsilon as f32, ..P::default() };
            let buffers = [whole(buffer(w)?), whole(buffer(m)?), whole(buffer(v)?), whole(buffer(g)?), whole(&constants)];
            self.elements("t_adam", &buffers, w.len(), p)
        }

        fn posterior(&self, kernel: &'static str, buffers: &[(&Buffer, usize)], n: usize, mut p: Posterior) -> Result<(), GpuError> {
            p.n = u32_of(n)?;
            self.stream.dispatch(kernel, buffers, &p, spread(n))
        }

        pub(super) fn reparameterize(&self, theta: &mut Tensor, (mean, log_sd): (&Tensor, &Tensor), (key, stream): (u64, u64)) -> Result<(), GpuError> {
            let p = Posterior { key: halves(key), stream: halves(stream), ..Posterior::default() };
            self.posterior("t_reparameterize", &[whole(buffer(mean)?), whole(buffer(log_sd)?), whole(buffer(theta)?)], theta.len(), p)
        }

        pub(super) fn posterior_adam(
            &self,
            (mean, log_sd): (&mut Tensor, &mut Tensor),
            [mm, mv, sm, sv]: [&mut Tensor; 4],
            gradient: &Tensor,
            (groups, variance): (&Indices, &Tensor),
            sums: &mut Tensor,
            step: &super::PosteriorStep,
        ) -> Result<(), GpuError> {
            let (c1, c2) = step.corrections();
            let p = Posterior {
                count: u32_of(variance.len())?,
                key: halves(step.key),
                stream: halves(step.stream),
                scale: step.gradient_scale as f32,
                mean_rate: step.mean_rate as f32,
                log_sd_rate: step.log_sd_rate as f32,
                beta1: step.beta1 as f32,
                beta2: step.beta2 as f32,
                epsilon: step.epsilon as f32,
                c1: c1 as f32,
                c2: c2 as f32,
                ..Posterior::default()
            };
            let buffers = [
                whole(buffer(gradient)?),
                whole(index_buffer(groups)?),
                whole(buffer(variance)?),
                whole(buffer(mean)?),
                whole(buffer(log_sd)?),
                whole(buffer(mm)?),
                whole(buffer(mv)?),
                whole(buffer(sm)?),
                whole(buffer(sv)?),
                whole(buffer(sums)?),
            ];
            self.posterior("t_posterior_adam", &buffers, mean.len(), p)
        }

        pub(super) fn group_moments(&self, (mean, log_sd): (&Tensor, &Tensor), groups: &Indices, sums: &mut Tensor) -> Result<(), GpuError> {
            let p = Posterior { count: u32_of(sums.rows)?, ..Posterior::default() };
            self.posterior("t_group_moments", &[whole(buffer(mean)?), whole(buffer(log_sd)?), whole(index_buffer(groups)?), whole(buffer(sums)?)], mean.len(), p)
        }

        pub(super) fn group_divergence(&self, sums: &mut Tensor, variance: &mut Tensor, divergence: &mut Tensor) -> Result<(), GpuError> {
            self.elements("t_group_divergence", &[whole(buffer(sums)?), whole(buffer(variance)?), whole(buffer(divergence)?)], variance.len(), P::default())
        }

        pub(super) fn select_sets(&self, (a, q): (&Tensor, &Tensor), bits: &Tensor, left: &Tensor, weight: &Tensor, mask: &mut Tensor) -> Result<(), GpuError> {
            if a.is_empty() {
                return Ok(());
            }
            let width = a.cols.next_power_of_two().max(2);
            // One scratch ranking and row of sizes per threadgroup; the groups stride over the rows.
            let groups = a.rows.min(512);
            let (keys, order, sizes) = (self.stream.alloc(groups * width)?, self.stream.alloc(groups * width)?, self.stream.alloc(groups * width)?);
            let p = P { rows: u32_of(a.rows)?, cols: u32_of(a.cols)?, extra: u32_of(width)?, ..P::default() };
            u32_of(groups * width)?;
            let buffers = [
                whole(buffer(a)?),
                whole(buffer(q)?),
                whole(buffer(bits)?),
                whole(buffer(left)?),
                whole(buffer(weight)?),
                whole(&keys),
                whole(&order),
                whole(&sizes),
                whole(buffer(mask)?),
            ];
            self.stream.dispatch("t_select_sets", &buffers, &p, groups)
        }

        pub(super) fn code_rows(
            &self,
            (z, w, yfy): (&Tensor, &Tensor, &Tensor),
            (gram, starts, bits): (&Tensor, &Indices, &Tensor),
            warm: Option<&Tensor>,
            (kappa, nodes, tolerance): (f64, usize, f64),
            on: &mut Tensor,
        ) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
            let (rows, pieces, blocks) = (z.rows, z.cols, bits.cols);
            if rows == 0 {
                return Ok((Vec::new(), Vec::new()));
            }
            let width = blocks.next_power_of_two().max(2);
            let open = nodes + 1;
            // As the CUDA kernel's slots; the groups stride over the rows.
            let per_slot = 15 * blocks + pieces + width + open * (4 * blocks + 1);
            let per_index = blocks + width;
            let groups = rows.min(128);
            u32_of(groups * per_slot)?;
            let (scratch, indices) = (self.stream.alloc(groups * per_slot)?, self.stream.alloc(groups * per_index)?);
            let (upper, lower, no_warm) = (self.stream.alloc(rows)?, self.stream.alloc(rows)?, self.stream.alloc(1)?);
            let flags = u32::from(blocks == pieces) | (u32::from(warm.is_some()) << 1);
            // An f32 relaxation is stationary at a move of a millionth (its coordinates are in [0, 1]).
            let p = P { n: flags, rows: u32_of(rows)?, cols: u32_of(pieces)?, extra: u32_of(width)?, a: u32_of(blocks)?, b: u32_of(nodes)?, alpha: kappa as f32, beta: (tolerance as f32).max(1e-6), c: 0, d: 0 };
            let buffers = [
                whole(buffer(z)?),
                whole(buffer(w)?),
                whole(buffer(yfy)?),
                whole(buffer(gram)?),
                whole(index_buffer(starts)?),
                whole(buffer(bits)?),
                whole(match warm {
                    Some(m) => buffer(m)?,
                    None => &no_warm,
                }),
                whole(&scratch),
                whole(&indices),
                whole(buffer(on)?),
                whole(&upper),
                whole(&lower),
            ];
            self.stream.dispatch("t_code_rows", &buffers, &p, groups)?;
            let widen = |b: &Buffer| -> Result<Vec<f64>, GpuError> { Ok(self.stream.read::<f32>(b)?.into_iter().take(rows).map(f64::from).collect()) };
            Ok((widen(&upper)?, widen(&lower)?))
        }

        pub(super) fn box_charge(&self, z: &Tensor, mask: &Tensor, q: &Tensor, cot: &mut Tensor, coefficient: &mut Tensor) -> Result<Vec<f64>, GpuError> {
            let norms = self.stream.alloc(z.rows)?;
            let buffers = [whole(buffer(z)?), whole(buffer(mask)?), whole(buffer(q)?), whole(&norms), whole(buffer(cot)?), whole(buffer(coefficient)?)];
            self.rows("t_box_charge", &buffers, z.rows, z.cols, P::default())?;
            Ok(self.stream.read::<f32>(&norms)?.into_iter().take(z.rows).map(|n| 0.5 * f64::from(n) * f64::from(n)).collect())
        }
    }
}

#[cfg(test)]
mod code_rows_workspace_tests {
    use super::*;

    #[test]
    fn explicit_workspace_accounts_cache_and_limits_concurrent_rows() {
        let w = CodeRowsWorkspace { bytes: usize::MAX, max_rows: 3, cache_columns: false };
        let plain = code_rows_layout(10, 8, 4, 16, w).unwrap();
        let cached = code_rows_layout(10, 8, 4, 16, CodeRowsWorkspace { cache_columns: true, ..w }).unwrap();
        assert_eq!(plain.slots, 3);
        assert_eq!(plain.cache_offset, cached.cache_offset);
        assert_eq!(plain.doubles, plain.cache_offset);
        assert_eq!(cached.doubles - plain.doubles, 16);
        assert_eq!(cached.indices - plain.indices, 4);
        assert_eq!(cached.bytes - plain.bytes, 3 * (16 * 8 + 4 * 4));
        let one = cached.bytes / cached.slots;
        let limited = code_rows_layout(10, 8, 4, 16, CodeRowsWorkspace { bytes: 2 * one, max_rows: 10, cache_columns: true }).unwrap();
        assert_eq!(limited.slots, 2);
        assert_eq!(limited.bytes, 2 * one);
        assert!(code_rows_layout(10, 8, 4, 16, CodeRowsWorkspace { bytes: one - 1, max_rows: 10, cache_columns: true }).is_err());
    }

    #[test]
    fn workspace_overflow_and_zero_row_cap_are_errors() {
        let w = CodeRowsWorkspace { bytes: usize::MAX, max_rows: 1, cache_columns: true };
        assert!(code_rows_layout(1, 1, usize::MAX, 16, w).is_err());
        assert!(code_rows_layout(1, 1, 1, usize::MAX, w).is_err());
        assert!(code_rows_layout(1, 1, 1, 1, CodeRowsWorkspace { max_rows: 0, ..w }).is_err());
        assert_eq!(code_rows_layout(0, 8, 4, 16, w).unwrap().bytes, 0);
    }
}

#[cfg(test)]
mod column_copy_tests {
    use super::*;
    #[test]
    fn owned_reshape_preserves_bits_and_rejects_overflow() {
        let device = Device::host();
        let values = vec![-0.0, 1e300, f64::from_bits(1), -3.0];
        let held = device.upload_vec(4,1,values.clone()).unwrap();
        let reshaped = held.reshape(1,4).unwrap();
        assert_eq!(reshaped.dim(),(1,4));
        assert!(device.download(&reshaped).unwrap().iter().zip(values).all(|(a,b)|a.to_bits()==b.to_bits()));
        assert!(device.zeros(2,2).unwrap().reshape(1,3).is_err());
        assert!(device.zeros(2,2).unwrap().reshape(usize::MAX,2).is_err());
        assert_eq!(device.zeros(0,2).unwrap().reshape(3,0).unwrap().dim(),(3,0));
    }

    #[test]
    fn columns_copy_bits_and_preserve_other_columns() {
        let device = Device::host();
        let mut out = device.upload_vec(2, 4, vec![9.0; 8]).unwrap();
        let part = device.upload_vec(2, 2, vec![-0.0, 1e300, f64::from_bits(1), -3.0]).unwrap();
        device.set_columns(&mut out, 1, &part).unwrap();
        let actual = device.download(&out).unwrap();
        let expected = [9.0f64, -0.0, 1e300, 9.0, 9.0, f64::from_bits(1), -3.0, 9.0];
        assert!(actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()));
        let before = actual;
        assert!(device.set_columns(&mut out, usize::MAX, &part).is_err());
        assert!(device.set_columns(&mut out, 3, &part).is_err());
        let bad_rows = device.zeros(1, 1).unwrap();
        assert!(device.set_columns(&mut out, 0, &bad_rows).is_err());
        assert_eq!(device.download(&out).unwrap(), before);
    }
}

#[cfg(test)]
mod kl_proposal_tests {
    use super::*;
    #[test]
    fn stable_tails_ties_and_reference_value_without_mutation() {
        let d=Device::host();
        let p=d.upload_vec(3,3,vec![1000.,1000.,1000.,0.,-2000.,-2000.,0.,1.,-3.]).unwrap();
        let q=d.upload_vec(3,3,vec![1000.,1000.,1000.,-2000.,0.,-2000.,-1.,2.,0.]).unwrap();
        let before=d.download(&q).unwrap();
        let proposed=d.kl_proposal_rows(&p,&q).unwrap();
        let mut scratch=d.copy(&q).unwrap();
        let reference=d.kl_score_rows(&p,&mut scratch,None).unwrap();
        assert_eq!(proposed[0].value,0.0); assert_eq!(proposed[1].value,2000.0); assert_eq!(proposed[1].max_log_difference,2000.0);
        for (row,expected) in proposed.iter().zip(reference) {
            assert_eq!(row.value,expected); assert!(row.magnitude>=0.0 && row.spread>=0.0);
        }
        assert_eq!(d.argmax_rows(&p).unwrap(),vec![0,0,1]);
        assert_eq!(d.download(&q).unwrap(),before);
    }
    #[test]
    fn malformed_and_overflowed_rows_are_errors_not_zero_scores() {
        let d=Device::host(); let good=d.upload_vec(1,2,vec![0.,1.]).unwrap();
        for bad in [f64::NAN,f64::INFINITY,f64::NEG_INFINITY] {
            let p=d.upload_vec(1,2,vec![bad,0.]).unwrap();
            assert!(d.kl_proposal_rows(&p,&good).is_err());
            assert!(d.kl_proposal_rows(&good,&p).is_err());
        }
        let overflow=d.upload_vec(1,2,vec![f64::MAX,-f64::MAX]).unwrap();
        assert!(d.kl_proposal_rows(&overflow,&good).is_err());
        assert!(d.kl_proposal_rows(&good,&d.zeros(2,2).unwrap()).is_err());
        assert!(d.kl_proposal_rows(&d.zeros(1,0).unwrap(),&d.zeros(1,0).unwrap()).is_err());
    }
}

#[cfg(test)]
mod checked_interval_api_tests {
    use super::*;
    #[test]
    fn checked_intervals_refuse_host_and_respect_output_budget() {
        let device=Device::host();
        let input=device.upload_vec(1,2,vec![0.0,1.0]).expect("host upload");
        let budget=checked_interval_output_bytes(2).expect("small budget");
        assert_eq!(budget,2*(48+std::mem::size_of::<CheckedInterval>()));
        assert!(checked_interval_output_bytes(usize::MAX).is_err());
        assert!(device.checked_scalar_intervals(&input,CheckedScalar::Exp,budget).is_err());
        assert!(device.checked_kl_intervals(&input,&input,budget).is_err());
        assert!(device.checked_scalar_intervals(&input,CheckedScalar::Exp,0).is_err());
        assert!(device.checked_interval_compiler_info().is_err());
        let mismatch=device.upload_vec(1,1,vec![0.0]).expect("host upload");
        assert!(device.checked_kl_intervals(&input,&mismatch,budget).is_err());
    }
}

#[cfg(test)]
mod scaled_row_l2_tests {
    use super::*;

    #[test]
    fn ordered_norms_match_host_with_ranges_and_no_input_mutation() {
        let host = Device::host();
        let mut devices = vec![Device::host()];
        if let Some(device) = Device::accelerator(crate::GpuPolicy::Auto).expect("device probe") {
            if device.float64() { devices.push(device); }
        }
        // Non-power-of-two width and mixed magnitudes exercise accumulation
        // order, normal/subnormal products, zero and a partial final warp.
        let (rows, cols) = (35, 773);
        let values: Vec<f64> = (0..rows * cols).map(|i| {
            if i / cols == 0 { 0.0 } else { match i % 5 {
                0 => 1.0, 1 => 2.0_f64.powi(-26), 2 => -0.75,
                3 => f64::from_bits(1), _ => (i % 31) as f64 / 17.0,
            } }
        }).collect();
        let reference = host.upload_vec(rows, cols, values.clone()).unwrap();
        for device in devices {
            let input = device.upload_vec(rows, cols, values.clone()).unwrap();
            for columns in [0..cols, 3..cols - 2, 9..9] {
                for scale in [1.0, 0.125, 3.25] {
                    let expected = host.scaled_row_l2(&reference, columns.clone(), scale).unwrap();
                    let actual = device.scaled_row_l2(&input, columns.clone(), scale).unwrap();
                    assert_eq!(actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "{}", device.name());
                }
            }
            assert_eq!(device.download(&input).unwrap().as_slice().unwrap(), values.as_slice());
            assert!(device.scaled_row_l2(&device.zeros(0, cols).unwrap(), 0..cols, 1.0).unwrap().is_empty());
        }
    }

    #[test]
    fn enclosed_norms_handle_tiny_large_zero_and_subnormal_rows() {
        let mut devices=vec![Device::host()];
        if let Some(device)=Device::accelerator(crate::GpuPolicy::Auto).expect("device probe") {
            if device.float64() {devices.push(device);}
        }
        for device in devices {
            for amplitude in [2.0_f64.powi(-1000),1e-200,1.0,1e200] {
                let input=device.upload_vec(1,2,vec![3.0*amplitude,4.0*amplitude]).expect("fixture upload");
                let bounds=device.scaled_row_l2_enclosed(&input,0..2,[amplitude;3]).expect("finite enclosure")[0];
                assert!(bounds[1]<=5.0 && bounds[2]>=5.0,"{} {amplitude} {bounds:?}",device.name());
            }
            let zero=device.zeros(1,2).expect("zero fixture");
            assert_eq!(device.scaled_row_l2_enclosed(&zero,0..2,[0.0;3]).expect("zero/zero"),vec![[0.0;3]]);
            let nonzero=device.upload_vec(1,1,vec![f64::from_bits(1)]).expect("minimum subnormal");
            let bounds=device.scaled_row_l2_enclosed(&nonzero,0..1,[f64::from_bits(1);3]).expect("subnormal ratio")[0];
            assert!(bounds[1]<=1.0 && bounds[2]>=1.0);
            assert!(device.scaled_row_l2_enclosed(&nonzero,0..1,[0.0;3]).is_err());
        }
    }

    #[test]
    fn invalid_ranges_scales_and_nonfinite_reductions_are_errors() {
        let device = Device::host();
        let input = device.upload_vec(1, 3, vec![3.0, 4.0, f64::NAN]).unwrap();
        assert_eq!(device.scaled_row_l2(&input, 0..2, 1.0).unwrap(), vec![5.0]);
        assert!(device.scaled_row_l2(&input, 0..3, 1.0).is_err());
        for range in [2..1, 0..4, usize::MAX..usize::MAX] { assert!(device.scaled_row_l2(&input, range, 1.0).is_err()); }
        for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] { assert!(device.scaled_row_l2(&input, 0..2, scale).is_err()); }
        let overflow = device.upload_vec(1, 1, vec![f64::MAX]).unwrap();
        assert!(device.scaled_row_l2(&overflow, 0..1, 1.0).is_err()); // upper endpoint overflows: unresolved, never a fabricated bound
    }
}

#[cfg(test)]
mod columns_of_tests {
    use super::*;
    #[test]
    fn exact_column_copies_preserve_bits_and_source() {
        let mut devices=vec![Device::host()];
        if let Some(device)=Device::accelerator(crate::GpuPolicy::Auto).expect("device probe") {
            if device.float64() { devices.push(device); }
        }
        let values=vec![1.0, -0.0, f64::from_bits(1), f64::INFINITY, -3.0, f64::NAN, 4.0, 0.0];
        for device in devices {
            let input=device.upload_vec(2,4,values.clone()).unwrap();
            let part=device.columns_of(&input,1..3).unwrap();
            assert_eq!(device.download(&part).unwrap().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),vec![values[1].to_bits(),values[2].to_bits(),values[5].to_bits(),values[6].to_bits()]);
            assert_eq!(device.download(&input).unwrap().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),values.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            for range in [1..1,2..1,0..5,usize::MAX..usize::MAX] { assert!(device.columns_of(&input,range).is_err()); }
        }
    }
}

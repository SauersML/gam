//! Device-resident float64 tensors and the kernels a program executor needs (#2951).
//!
//! A [`Device`] runs dense row-major tensors (`rows × cols`, `f64`) through a small, closed set
//! of operations: products (BLAS GEMM, also strided-batched over equal row blocks), elementwise
//! maps, per-row reductions (a norm, a softmax, a KL against a target row, a Fisher probe's
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
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis, Slice, linalg::general_mat_mul};
use rayon::prelude::*;
use std::sync::Arc;

/// The row ranges one CUDA launch of [`Device::gather_ranges`] or [`Device::scatter_ranges`] moves
/// (passed by value: a kernel's parameters hold 4 KB).
pub const ROW_RANGES: usize = 320;

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
    /// Each operand as two TF32 terms, its nearest TF32 value `big` and the remainder `small`, and
    /// the three products `big·big + big·small + small·big` summed with f32 accumulation (the
    /// remainders' product, below `2⁻²²` of each term, is left out): within about `2⁻²⁰` of the
    /// exact product per term, at a third of the tensor cores' TF32 rate. A bfloat16 copy is
    /// exact in TF32 and has no remainder. The host rounds alike; CUDA float64 storage and the
    /// Apple GPU run it as `F32`.
    Tf32x3,
    /// As [`Arithmetic::Tf32x3`] with bfloat16 terms (`hi`, and `lo` the remainder rounded to
    /// bfloat16): within about `2⁻¹⁵` of the exact product per term, at a third of the bfloat16
    /// rate.
    Bf16x3,
}

/// How a tensor holds its values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Storage {
    /// IEEE float64 (the host always; CUDA by default).
    F64,
    /// IEEE float32 (the Apple GPU always; CUDA on request, for fitting: [`Device::with_storage`]).
    F32,
    /// bfloat16, CUDA only: a frozen operand's copy ([`Device::bf16_copy`]) that products in
    /// [`Arithmetic::Bf16`] read without rounding it again, and a posterior's sample
    /// ([`Device::reparameterize`]). A bfloat16 device
    /// ([`Device::with_storage`]) makes, copies, converts, uploads and downloads them; no other
    /// operation takes one.
    Bf16,
}

impl Arithmetic {
    /// The unit roundoff of the operands' rounding; for a split arithmetic, half its bound on a
    /// term's error relative to the term's magnitude (the remainders' product left out, each
    /// remainder's own rounding, which the tensor cores may truncate in TF32, and the operands'
    /// rounding to f32).
    #[must_use]
    pub fn unit_roundoff(self) -> f64 {
        match self {
            Self::F64 => f64::EPSILON / 2.0,
            Self::F32 => f64::from(f32::EPSILON) / 2.0,
            Self::Tf32 => 2f64.powi(-11),
            Self::Bf16 => 2f64.powi(-8),
            Self::Tf32x3 => 2f64.powi(-20),
            Self::Bf16x3 => 2f64.powi(-15),
        }
    }

    /// The arithmetic of each of a split arithmetic's three products, and `None` for any other.
    #[must_use]
    pub fn split(self) -> Option<Self> {
        match self {
            Self::Tf32x3 => Some(Self::Tf32),
            Self::Bf16x3 => Some(Self::Bf16),
            _ => None,
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
    /// `ln max(t, t₀)`, `t₀` f32's smallest normal ([`LOG_FLOOR`]): a log-scale gate's read, finite
    /// at a zero input, its slope `1/t` above `t₀` and 0 below.
    Log,
}

/// The smallest input [`PointwiseLaw::Log`] takes the log of (f32's smallest normal, in every
/// storage): below it the law is `ln t₀ ≈ −87.3` and its slope 0.
pub const LOG_FLOOR: f64 = 1.175_494_350_822_287_5e-38;

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
            Self::Log => 6,
        }
    }

    fn of(code: u32) -> Self {
        match code {
            0 => Self::Relu,
            1 => Self::Identity,
            2 => Self::Zero,
            3 => Self::Silu,
            4 => Self::Gelu,
            6 => Self::Log,
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
            Self::Log => t.max(LOG_FLOOR).ln(),
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
            Self::Log => {
                if t > LOG_FLOOR {
                    1.0 / t
                } else {
                    0.0
                }
            }
        }
    }
}

const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

fn normal_cdf(t: f64) -> f64 {
    0.5 * libm::erfc(-t * std::f64::consts::FRAC_1_SQRT_2)
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

/// A CUDA tensor's buffer goes back to its stream's kept buffers when the tensor goes
/// (`cuda::recycle`), for that stream's next allocation of its size.
#[cfg(target_os = "linux")]
impl Drop for Tensor {
    fn drop(&mut self) {
        match std::mem::replace(&mut self.data, Data::Host(Vec::new())) {
            Data::Cuda(values) => cuda::recycle(values),
            Data::Cuda32(values) => cuda::recycle(values),
            Data::CudaBf16(values) => cuda::recycle(values),
            Data::Host(values) => drop(values),
        }
    }
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

/// Per row of a `rows × cols` value, the columns it lists ([`Device::row_lists`]): row `r`'s are
/// `columns[offsets[r]..offsets[r + 1]]`, increasing, and `row_of` holds each listed entry's row.
/// A product reading or writing only these entries ([`Device::sampled_product`],
/// [`Device::listed_product`]) takes each row's own columns, not the batch's union of them.
pub struct RowLists {
    rows: usize,
    cols: usize,
    offsets: Indices,
    columns: Indices,
    /// Read only by the device kernels (CUDA on Linux, the Apple GPU on macOS).
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    row_of: Indices,
}

impl RowLists {
    /// The listed entries over all rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.columns.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.columns.len == 0
    }

    /// The rows and columns of the value it lists entries of.
    #[must_use]
    pub fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

/// What [`Device::row_norm`] makes of each row: the norm `N(y)`, its tangent along a direction, or
/// its pullback of a cotangent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowNorm {
    Apply,
    Tangent,
    Pullback,
}

impl RowNorm {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn code(self) -> u32 {
        match self {
            Self::Apply => 0,
            Self::Tangent => 1,
            Self::Pullback => 2,
        }
    }
}

/// Which of an operator's coordinates its prior-group ids index ([`GroupMap`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GroupAxis {
    /// One id per row: every entry of a row is in its row's group.
    Rows,
    /// One id per column.
    Columns,
    /// One id per entry, row-major.
    Entries,
}

/// An operator's prior groups on the device ([`Device::group_map`]): its ids and the axis they
/// index. A row or column layout holds one id per row or column instead of four bytes per entry.
/// The device's group sums run over the map's segments (`layout`): each group of the operator is
/// one segment, the coordinates (rows, columns or entries) the axis indexes in its group, and one
/// warp (SIMD group) sums each segment's entries in a fixed order and adds them to its group's row
/// once. No sum is atomic, so the sums are the same on every run.
/// One-column segments from which CUDA reduces 32 a block (`GroupMap::reduce_code`): 256 blocks,
/// two a multiprocessor on a 128-multiprocessor card. Fewer take eight a block: on vpd4l's 3072
/// columns an RTX 4090 ran posterior_ivon at 270 µs a launch with eight (a4d6618ae7) against 364
/// with four (9cde71ec8e, f97524b178's), the four columns' half sectors costing more than their
/// extra blocks gained.
#[cfg(target_os = "linux")]
const WIDE_SEGMENTS: usize = 256 * 32;

/// The entrywise maps of a gate ([`Device::gate_function`]), with `z = x / s` for the scaled ones:
/// `Sqrt` `√x` (a group's norm from its squared sum), `Step` the Heaviside `H(x) = 1{x > 0}`,
/// `Cdf` `Φ(z)` (an expected gate), `CdfSlope` its derivative in `x`, `φ(z) / s`, and `CdfScaleSlope`
/// its derivative in `s`, `−φ(z) z / s`, and `Ratio` `x / s`, zero where `s` is (a norm's
/// cotangent over the norm), `Variance` `e^{2x}` (a variance from its log standard deviation),
/// and the ramp gate `Ramp` `clamp(z, 0, 1)`, exactly zero where `x ≤ 0`, with its derivatives
/// `RampSlope` `1 / s` in `x` and `RampScaleSlope` `−z / s` in `s`, both where `0 < z < 1` and
/// zero elsewhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateFunction {
    Sqrt,
    Step,
    Cdf,
    CdfSlope,
    CdfScaleSlope,
    Ratio,
    Variance,
    Ramp,
    RampSlope,
    RampScaleSlope,
}

impl GateFunction {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn code(self) -> u32 {
        match self {
            Self::Sqrt => 0,
            Self::Step => 1,
            Self::Cdf => 2,
            Self::CdfSlope => 3,
            Self::CdfScaleSlope => 4,
            Self::Ratio => 5,
            Self::Variance => 6,
            Self::Ramp => 7,
            Self::RampSlope => 8,
            Self::RampScaleSlope => 9,
        }
    }

    fn scaled(self) -> bool {
        matches!(self, Self::Cdf | Self::CdfSlope | Self::CdfScaleSlope | Self::Ratio | Self::Ramp | Self::RampSlope | Self::RampScaleSlope)
    }

    /// The map in float64 (the host's, and the kernels' reference).
    #[must_use]
    pub fn host(self, x: f64, s: f64) -> f64 {
        let z = x / s;
        let density = (-0.5 * z * z).exp() * 0.398_942_280_401_432_7;
        match self {
            Self::Sqrt => x.sqrt(),
            Self::Step => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Cdf => 0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2),
            Self::CdfSlope => density / s,
            Self::CdfScaleSlope => -density * z / s,
            Self::Ratio => {
                if s == 0.0 {
                    0.0
                } else {
                    x / s
                }
            }
            Self::Variance => (2.0 * x).exp(),
            Self::Ramp => {
                if z > 0.0 {
                    z.min(1.0)
                } else {
                    0.0
                }
            }
            Self::RampSlope => {
                if z > 0.0 && z < 1.0 {
                    1.0 / s
                } else {
                    0.0
                }
            }
            Self::RampScaleSlope => {
                if z > 0.0 && z < 1.0 {
                    -z / s
                } else {
                    0.0
                }
            }
        }
    }
}

pub struct GroupMap {
    ids: Indices,
    axis: GroupAxis,
    rows: usize,
    cols: usize,
    /// The segments' `S + 1` offsets into the members, the `S` segments' groups, then the members
    /// (the axis's coordinates, by group and within one in ascending order). Only the CUDA and
    /// Metal kernels read the segments, so they exist only for Linux and macOS.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    layout: Indices,
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    segments: usize,
}

impl GroupMap {
    /// The axis the ids index.
    #[must_use]
    pub fn axis(&self) -> GroupAxis {
        self.axis
    }

    /// The operator's shape.
    #[must_use]
    pub fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Entry `i`'s (row-major) group in `ids`, the map's ids on the host.
    fn group(&self, ids: &[u32], i: usize) -> u32 {
        match self.axis {
            GroupAxis::Rows => ids[i / self.cols],
            GroupAxis::Columns => ids[i % self.cols],
            GroupAxis::Entries => ids[i],
        }
    }

    /// The axis as the kernels' code: 0 rows, 1 columns, 2 entries.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn code(&self) -> u32 {
        match self.axis {
            GroupAxis::Rows => 0,
            GroupAxis::Columns => 1,
            GroupAxis::Entries => 2,
        }
    }

    /// The segments of the map's layout, the kernels' `chunks`.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn chunks(&self) -> usize {
        self.segments
    }

    /// The threads an Apple GPU reduction over the map runs: one SIMD group of 32 per segment.
    #[cfg(target_os = "macos")]
    fn threads(&self) -> usize {
        32 * self.segments
    }

    /// The threads a CUDA reduction over the map runs: one block of 256 (the kernels' `BLOCK`) per
    /// segment (`segments_reduce`), per 8 or 32 segments of one column each (`columns_reduce`,
    /// `tiles_reduce`), or
    /// one thread per segment of a single entry (`entries_reduce`).
    #[cfg(target_os = "linux")]
    fn blocks(&self) -> usize {
        match self.reduce_code() {
            3 => 256 * self.segments.div_ceil(8),
            4 => self.segments,
            5 => 256 * self.segments.div_ceil(32),
            _ => 256 * self.segments,
        }
    }

    /// Whether every segment holds one entry (a bias's groups: one row of one column each).
    #[cfg(target_os = "linux")]
    fn single_entries(&self) -> bool {
        self.segments == self.rows * self.cols
    }

    /// Whether every segment is one column: column groups of one column each (an MLP's output
    /// columns, a unit each), which CUDA reduces eight adjacent columns at a time.
    #[cfg(target_os = "linux")]
    fn single_columns(&self) -> bool {
        self.axis == GroupAxis::Columns && self.segments == self.cols
    }

    /// The axis as CUDA's group reductions take it: [`GroupMap::code`]; for a map of one-column
    /// segments 3 (`columns_reduce`, eight columns a block), or 5 from [`WIDE_SEGMENTS`] of them
    /// (`tiles_reduce`, 32 columns a block: a warp reads 128 adjacent bytes of a row, where eight
    /// columns a block read 32 of each of four rows); 4 for a map of one-entry segments (one thread
    /// each, where a block of 256 took each). Every code adds each group's sums in the same order,
    /// so the sums do not depend on it.
    #[cfg(target_os = "linux")]
    fn reduce_code(&self) -> u32 {
        if self.single_entries() {
            4
        } else if self.single_columns() {
            if self.segments >= WIDE_SEGMENTS { 5 } else { 3 }
        } else {
            self.code()
        }
    }

    /// `t`'s shape is the map's.
    fn check(&self, t: &Tensor, what: &str) -> Result<(), GpuError> {
        if t.dim() != (self.rows, self.cols) {
            return Err(shape(format!("{what}: a {:?} tensor with a {:?} group map", t.dim(), (self.rows, self.cols))));
        }
        Ok(())
    }
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

/// Refuses f32 operands where an operation is float64 only (a certificate, an acceptance statistic).
fn float64_only(what: &str, tensors: &[&Tensor]) -> Result<(), GpuError> {
    if tensors.iter().any(|t| t.storage() != Storage::F64) {
        return Err(GpuError::NoDeviceKernel { reason: format!("{what} run in float64 only") });
    }
    Ok(())
}

/// A split of `heads` heads of `width` from column `start` of `x` in `blocks` sequences, turned by
/// `turn` ([`Device::split_heads`]): its sequences' length and the rotation's planes.
#[cfg(target_os = "linux")]
fn split_shape(x: &Tensor, (start, heads, width, blocks): (usize, usize, usize, usize), turn: Option<(&Tensor, &Tensor, bool)>) -> Result<(usize, usize), GpuError> {
    let rows = x.rows;
    if blocks == 0 || rows % blocks != 0 || start.checked_add(heads.saturating_mul(width)).is_none_or(|end| end > x.cols) {
        return Err(shape(format!("{heads} heads of {width} from column {start} of {:?} in {blocks} blocks", x.dim())));
    }
    let planes = turn.map_or(0, |(cos, _, _)| cos.cols);
    if let Some((cos, sin, _)) = turn {
        same(cos, sin, "rotation tables")?;
        if cos.rows != rows || 2 * planes > width {
            return Err(shape(format!("{:?} rotation tables on {rows} rows of {width}-wide heads", cos.dim())));
        }
    }
    Ok((rows / blocks, planes))
}

/// Rows `at..at + rows` and `from..from + rows` lie in `t` and apart.
fn apart(t: &Tensor, at: usize, from: usize, rows: usize) -> Result<(), GpuError> {
    if at + rows > t.rows || from + rows > t.rows || (at < from + rows && from < at + rows) {
        return Err(shape(format!("rows {from}..{} onto rows {at}..{} of {:?}", from + rows, at + rows, t.dim())));
    }
    Ok(())
}

fn same(a: &Tensor, b: &Tensor, what: &str) -> Result<(), GpuError> {
    if a.dim() != b.dim() {
        return Err(shape(format!("{what}: {:?} against {:?}", a.dim(), b.dim())));
    }
    Ok(())
}

/// Weight samples of one key gathered to be written together ([`Device::samples`],
/// [`Device::add_sample`], [`Device::run_samples`]). On CUDA, per sample: its output's storage, and
/// its output (at its block's first entry), mean and log standard deviation addresses, entries,
/// columns, output row stride and stream.
pub struct Samples {
    key: u64,
    jobs: Vec<(Storage, [u64; 7])>,
}

/// The bfloat16 nearest `x` (ties to even), as its 16 bits; a NaN stays a quiet NaN.
fn bf16_bits(x: f32) -> u32 {
    let bits = x.to_bits();
    if x.is_nan() { (bits >> 16) | 0x40 } else { (bits + 0x7fff + ((bits >> 16) & 1)) >> 16 }
}

/// Rounds `x` to `arithmetic`'s operand precision (a split arithmetic's operands are f32; its
/// terms are rounded by its part's arithmetic, [`Arithmetic::split`]).
fn round_operand(x: f64, arithmetic: Arithmetic) -> f64 {
    match arithmetic {
        Arithmetic::F64 => x,
        Arithmetic::F32 | Arithmetic::Tf32x3 | Arithmetic::Bf16x3 => f64::from(x as f32),
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
            // The host holds the rounded values in float64, as its bfloat16 products round them.
            (Backend::Host, Data::Host(v)) => Ok(Tensor { rows: t.rows, cols: t.cols, data: Data::Host(v.iter().map(|x| round_operand(*x, Arithmetic::Bf16)).collect()) }),
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda(_)) => engine.bf16_copy(&engine.convert(t)?),
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda32(_)) => engine.bf16_copy(t),
            #[cfg(any(target_os = "linux", target_os = "macos"))]
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

    /// Returns the device buffers kept for reuse to the driver (CUDA: those of the tensors dropped
    /// since, kept for the stream's next allocations of their sizes): a caller's boundary before work
    /// of other shapes, such as the next batch, so that the buffers kept are those the work at hand
    /// makes again.
    pub fn release_recycled(&self) {
        #[cfg(target_os = "linux")]
        if let Backend::Cuda(engine) = &*self.backend {
            engine.release_recycled();
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

    /// A `rows × cols` tensor of `values` in the device's storage, made from the view directly: a
    /// device holding f32 or bfloat16 takes them narrowed in one pass, and one holding float64 takes
    /// a contiguous view as it is, with no float64 copy on the host in between.
    pub fn upload(&self, values: ArrayView2<'_, f64>) -> Result<Tensor, GpuError> {
        let (rows, cols) = values.dim();
        let data = match &*self.backend {
            Backend::Host => Data::Host(values.iter().copied().collect()),
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if self.storage == Storage::F32 => Data::Cuda32(engine.upload(&values.iter().map(|v| *v as f32).collect::<Vec<f32>>())?),
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if self.storage == Storage::Bf16 => {
                Data::CudaBf16(engine.upload(&values.iter().map(|v| bf16_bits(*v as f32) as u16).collect::<Vec<u16>>())?)
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => match values.as_slice() {
                Some(contiguous) => Data::Cuda(engine.upload(contiguous)?),
                None => Data::Cuda(engine.upload(&values.iter().copied().collect::<Vec<f64>>())?),
            },
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => Data::Metal(engine.stream.upload(&values.iter().map(|v| *v as f32).collect::<Vec<f32>>())?),
        };
        Ok(Tensor { rows, cols, data })
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

    /// `t`'s values as f32, row-major ([`Device::upload_f32`]'s inverse): read as they are where the
    /// device holds f32 (CUDA f32 storage, the Apple GPU), with no widening to float64 and narrowing
    /// back; bfloat16 values widen to f32 exactly and float64 values round to f32.
    pub fn download_f32(&self, t: &Tensor) -> Result<Vec<f32>, GpuError> {
        match &t.data {
            #[cfg(target_os = "linux")]
            Data::Cuda32(slice) => match &*self.backend {
                Backend::Cuda(engine) => {
                    let mut values = engine.download(slice)?;
                    values.truncate(t.len());
                    Ok(values)
                }
                _ => Err(foreign()),
            },
            #[cfg(target_os = "macos")]
            Data::Metal(buffer) => match &*self.backend {
                Backend::Metal(engine) => {
                    let mut values: Vec<f32> = engine.stream.read(buffer)?;
                    values.truncate(t.len());
                    Ok(values)
                }
                _ => Err(foreign()),
            },
            _ => Ok(self.download(t)?.iter().map(|v| *v as f32).collect()),
        }
    }

    /// A `rows × cols` tensor from row-major f32 `values` ([`Device::upload_vec`]), sent as they are
    /// where the device holds f32 (CUDA f32 storage, the Apple GPU): no widening and narrowing pass.
    pub fn upload_f32(&self, rows: usize, cols: usize, values: &[f32]) -> Result<Tensor, GpuError> {
        if values.len() != rows * cols {
            return Err(shape(format!("{} values for {rows}x{cols}", values.len())));
        }
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if self.storage == Storage::F32 => Ok(Tensor { rows, cols, data: Data::Cuda32(engine.upload(values)?) }),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => Ok(Tensor { rows, cols, data: Data::Metal(engine.stream.upload(values)?) }),
            _ => self.upload_vec(rows, cols, values.iter().map(|v| f64::from(*v)).collect()),
        }
    }

    /// [`Device::upload_f32`] whose host-to-device copy runs beside the kernels queued before it
    /// (CUDA f32: a second stream copies into a landing buffer, the stream then copies on the device
    /// into the tensor, waiting for the first copy only there): for values a later kernel reads, as
    /// a batch's restored targets are. Elsewhere [`Device::upload_f32`].
    pub fn upload_f32_overlapped(&self, rows: usize, cols: usize, values: &[f32]) -> Result<Tensor, GpuError> {
        if values.len() != rows * cols {
            return Err(shape(format!("{} values for {rows}x{cols}", values.len())));
        }
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if self.storage == Storage::F32 => Ok(Tensor { rows, cols, data: Data::Cuda32(engine.upload_overlapped(values)?) }),
            _ => self.upload_f32(rows, cols, values),
        }
    }

    /// An operator's prior-group ids `ids` (one per entry, row-major, of a `rows × cols` operator)
    /// held compactly ([`GroupMap`]): one id per row where every row is one group, else one per
    /// column where every column is one group, else one per entry.
    pub fn group_map(&self, ids: &[u32], (rows, cols): (usize, usize)) -> Result<GroupMap, GpuError> {
        if ids.len() != rows * cols {
            return Err(shape(format!("{} group ids for a {rows}x{cols} operator", ids.len())));
        }
        let by_rows = cols > 0 && ids.chunks(cols).all(|row| row.iter().all(|g| *g == row[0]));
        let by_columns = cols > 0 && rows > 0 && ids.chunks(cols).all(|row| row == &ids[..cols]);
        let (axis, compact): (GroupAxis, Vec<u32>) = if by_rows {
            (GroupAxis::Rows, ids.chunks(cols).map(|row| row[0]).collect())
        } else if by_columns {
            (GroupAxis::Columns, ids[..cols].to_vec())
        } else {
            (GroupAxis::Entries, ids.to_vec())
        };
        // The segments: the coordinates by group id, each group's in ascending order.
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        let (layout, segments) = {
            let index = |k: usize| u32::try_from(k).map_err(|_| shape(format!("a group map of {k} coordinates")));
            let mut members = (0..compact.len()).map(index).collect::<Result<Vec<u32>, _>>()?;
            members.sort_by_key(|&k| compact[k as usize]);
            let (mut offsets, mut groups) = (Vec::new(), Vec::new());
            for (k, &m) in members.iter().enumerate() {
                let g = compact[m as usize];
                if groups.last() != Some(&g) {
                    offsets.push(index(k)?);
                    groups.push(g);
                }
            }
            offsets.push(index(members.len())?);
            let segments = groups.len();
            let layout: Vec<u32> = offsets.into_iter().chain(groups).chain(members).collect();
            (self.upload_indices(&layout)?, segments)
        };
        Ok(GroupMap {
            ids: self.upload_indices(&compact)?,
            axis,
            rows,
            cols,
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            layout,
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            segments,
        })
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


    /// A `rows × cols` tensor whose values are unset (CUDA f32: no zeroing pass), for an output an
    /// operation then writes whole (a product with `β = 0`); zero elsewhere.
    pub fn empty(&self, rows: usize, cols: usize) -> Result<Tensor, GpuError> {
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if self.storage == Storage::F32 => engine.output(Storage::F32, rows, cols),
            _ => self.zeros(rows, cols),
        }
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

    /// `alpha op(a) op(b) + addend` as a new tensor (`addend` m × n, `op(a)` m × k, `op(b)` k × n),
    /// `addend` only read: on CUDA in f32 storage one product (cublasLt's D = α A B + C, C a matrix
    /// of its own; [`Arithmetic::Bf16`] rounds f32 operands as [`Device::gemm`] does); elsewhere
    /// `addend`'s copy and [`Device::gemm`] into it with β = 1.
    pub fn gemm_onto(&self, alpha: f64, (a, ta): (&Tensor, Op), (b, tb): (&Tensor, Op), addend: &Tensor, arithmetic: Arithmetic) -> Result<Tensor, GpuError> {
        let (m, k) = match ta {
            Op::N => (a.rows, a.cols),
            Op::T => (a.cols, a.rows),
        };
        let (kb, n) = match tb {
            Op::N => (b.rows, b.cols),
            Op::T => (b.cols, b.rows),
        };
        if k != kb || addend.dim() != (m, n) {
            return Err(shape(format!("op(a) {m}x{k}, op(b) {kb}x{n}, addend {:?}", addend.dim())));
        }
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if addend.storage() == Storage::F32 && arithmetic.split().is_none() && k > 0 => {
                let mut out = self.empty(m, n)?;
                engine.gemm_onto((m, n, k), alpha, (a, ta), (b, tb), (addend, &mut out), arithmetic)?;
                Ok(out)
            }
            _ => {
                let mut out = self.scaled(1.0, addend)?;
                self.gemm(&mut out, alpha, a, ta, b, tb, 1.0, arithmetic)?;
                Ok(out)
            }
        }
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

    /// The lower triangle (row-major, `i ≥ j`) of `c ← aᵀ a + β c` in float64, `a` rows × n and `c`
    /// n × n: a symmetric rank-k update, half a product's work; `c`'s strict upper triangle is
    /// left as it was. Float64 storage only (CUDA, the host).
    pub fn gram_lower(&self, c: &mut Tensor, a: &Tensor, beta: f64) -> Result<(), GpuError> {
        if c.dim() != (a.cols, a.cols) {
            return Err(shape(format!("a {:?} Gram of a {:?} product", c.dim(), a.dim())));
        }
        float64_only("a symmetric rank-k update", &[a, c])?;
        match &*self.backend {
            Backend::Host => {
                let (n, values, out) = (a.cols, host(a)?, host_mut(c)?);
                for i in 0..n {
                    for j in 0..=i {
                        let dot: f64 = values.chunks_exact(n).map(|row| row[i] * row[j]).sum();
                        out[i * n + j] = dot + beta * out[i * n + j];
                    }
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.gram_lower(c, a, beta),
            #[cfg(target_os = "macos")]
            Backend::Metal(_) => Err(GpuError::NoDeviceKernel { reason: "a symmetric rank-k update runs in float64 only".to_string() }),
        }
    }

    /// `c ← c + aᵀ a` (the whole float64 `c`) for the f32 rows × n activations `a`, by the Ozaki
    /// scheme on CUDA's integer tensor cores: each column j of `a` scaled by its exponent `e_j`
    /// (every |a_kj| < 2^e_j) and split into `slices` int8 slices, `a_kj = 2^(e_j − 7) Σ_s a_s,kj
    /// 2^(−7s) + r_kj` with `|r_kj| < 2^(e_j − 7 slices)`; the slice products with `s + t <
    /// slices` are exact in int32 (rows · 127² < 2^31) and added in float64. Per entry the
    /// result is within `(slices + 3) · rows · 2^(e_i + e_j − 7 slices) + γ_p Σ_k |a_ki a_kj|` of
    /// the exact sum, `p = slices` the float64 additions (one per shift `s + t`): the truncation is at most
    /// `2 · 2^(e_i + e_j − 7 slices)` per row, the products left out at most `1.01 (slices + 1)` of
    /// it, and an entry's slices share its sign, so the added products' magnitudes sum to at most
    /// `Σ_k |a_ki a_kj|`. Since `2^(e_i + e_j) < 4 max_k |a_ki| max_k |a_kj| ≤ 4 √(G_ii G_jj)`, with
    /// 9 slices and up to 2^14 rows the whole is below `γ_rows √(G_ii G_jj)`, which bounds a
    /// float64 product's own rounding as well (Cauchy–Schwarz). `Ok(false)` where
    /// there is no such path (the host, the Apple GPU, or `a` not f32), for the caller's float64
    /// product. A shift's products are summed in int32 (each `s < t` twice): `slices · rows · 127²`
    /// must stay below 2^31 (rows up to 14,794 with 9 slices), else it is refused.
    pub fn gram_split(&self, c: &mut Tensor, a: &Tensor, slices: usize) -> Result<bool, GpuError> {
        // A shift's sum holds up to `slices` products' worth (each `s < t` twice) of rows · 127².
        if c.dim() != (a.cols, a.cols) || slices == 0 || (slices * a.rows.div_ceil(4) * 4) as u128 * 127 * 127 >= 1 << 31 {
            return Err(shape(format!("a {:?} Gram of {:?} activations in {slices} slices", c.dim(), a.dim())));
        }
        match (&*self.backend, &a.data) {
            #[cfg(target_os = "linux")]
            (Backend::Cuda(engine), Data::Cuda32(_)) if a.rows > 0 && a.cols > 0 => engine.gram_split(c, a, slices).map(|()| true),
            _ => Ok(false),
        }
    }

    /// The eigendecomposition of the symmetric float64 matrix `a` (its lower triangle read) by
    /// cuSOLVER's divide and conquer (`cusolverDnDsyevd`) on CUDA: the eigenvalues in increasing
    /// order and the eigenvectors as columns, within the backward error of a stable symmetric
    /// eigensolver that `gam_linalg::roundoff::symmetric_spectrum_rounding_band` bounds. `None` on
    /// the host and the Apple GPU (no float64 there), whose callers decompose on the host.
    pub fn symmetric_eigh(&self, a: ArrayView2<'_, f64>) -> Result<Option<(ndarray::Array1<f64>, Array2<f64>)>, GpuError> {
        if a.nrows() != a.ncols() {
            return Err(shape(format!("a {:?} matrix for a symmetric eigendecomposition", a.dim())));
        }
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if a.nrows() > 0 => engine.symmetric_eigh(a).map(Some),
            _ => Ok(None),
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

    /// `0 + α x` per entry, a new tensor: each entry the one [`Device::axpy`] makes adding `α x` into
    /// zeros (a negative zero comes out positive, as there), in one pass with no zeroing first. f32
    /// or float64 storage.
    pub fn scaled(&self, alpha: f64, x: &Tensor) -> Result<Tensor, GpuError> {
        match &*self.backend {
            Backend::Host => Ok(Tensor { rows: x.rows, cols: x.cols, data: Data::Host(host(x)?.iter().map(|v| 0.0 + alpha * v).collect()) }),
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.scaled(alpha, x),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => {
                let mut out = self.zeros(x.rows, x.cols)?;
                engine.axpy(&mut out, alpha, x)?;
                Ok(out)
            }
        }
    }

    /// `y ← y + w (x − y)`, the difference rounded to the storage first: each entry the one
    /// [`Device::axpy`] makes adding `w` times a copy of `x` less `y` (by `axpy` with −1).
    pub fn move_toward(&self, y: &mut Tensor, weight: f64, x: &Tensor) -> Result<(), GpuError> {
        same(x, y, "move toward")?;
        match &*self.backend {
            Backend::Host => {
                for (yv, xv) in host_mut(y)?.iter_mut().zip(host(x)?) {
                    let d = xv - *yv;
                    *yv += weight * d;
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.move_toward(y, weight, x),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => {
                let mut d = self.copy(x)?;
                engine.axpy(&mut d, -1.0, y)?;
                engine.axpy(y, weight, &d)
            }
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

    /// Over the entries of the columns `alive` marks (1), at every row: `(1, v ≠ 0, |v| > √(s²
    /// max(σ²_z, 0) + φ² max(σ²_y, 0)))` added into row 0 of `sums` (1 × 3, float64 where the
    /// device holds it), `v` a function's value, `s` its slope against its input's posterior
    /// noise `σ²_z`, and with a second input (`noise_y`, a gated law) `φ` the value's other factor
    /// and its noise `σ²_y`: the entries counted, those nonzero, and those whose value exceeds its
    /// posterior noise. Every rows × functions input is in one storage.
    pub fn resolved_counts(
        &self,
        (value, slope, phi): (&Tensor, &Tensor, &Tensor),
        (noise_z, noise_y): (&Tensor, Option<&Tensor>),
        alive: &Indices,
        sums: &mut Tensor,
    ) -> Result<(), GpuError> {
        for (t, what) in [(slope, "slope"), (phi, "factor"), (noise_z, "input noise")].into_iter().chain(noise_y.map(|t| (t, "second noise"))) {
            same(value, t, what)?;
        }
        if alive.len != value.cols || sums.dim() != (1, 3) {
            return Err(shape(format!("{} column flags and {:?} sums for {:?} values", alive.len, sums.dim(), value.dim())));
        }
        match &*self.backend {
            Backend::Host => {
                let (v, s, p, z, flags, cols) = (host(value)?, host(slope)?, host(phi)?, host(noise_z)?, host_indices(alive)?, value.cols);
                let y = noise_y.map(host).transpose()?;
                let (mut count, mut nonzero, mut resolved) = (0.0, 0.0, 0.0);
                for i in 0..v.len() {
                    if flags[i % cols] == 0 {
                        continue;
                    }
                    let noise = s[i] * s[i] * z[i].max(0.0) + y.map_or(0.0, |y| p[i] * p[i] * y[i].max(0.0));
                    count += 1.0;
                    nonzero += f64::from(u8::from(v[i] != 0.0));
                    resolved += f64::from(u8::from(v[i].abs() > noise.sqrt()));
                }
                let totals = host_mut(sums)?;
                totals[0] += count;
                totals[1] += nonzero;
                totals[2] += resolved;
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.resolved_counts((value, slope, phi), (noise_z, noise_y), alive, sums),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.resolved_counts((value, slope, phi), (noise_z, noise_y), alive, sums),
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

    /// The columns of `x` holding a value other than zero (a NaN counts) in one of rows `rows`
    /// (increasing), increasing: on those rows the only columns a product `x Aᵀ` reads, every
    /// other column's terms being zeros. The flags are made on the device and read back.
    pub fn nonzero_columns(&self, x: &Tensor, rows: &[u32]) -> Result<Vec<u32>, GpuError> {
        if rows.iter().any(|&r| r as usize >= x.rows) {
            return Err(shape(format!("rows {rows:?} of {} rows", x.rows)));
        }
        let flags: Vec<f64> = match &*self.backend {
            Backend::Host => {
                let (v, cols) = (host(x)?, x.cols);
                let mut any = vec![0.0; cols];
                for &r in rows {
                    for (a, &t) in any.iter_mut().zip(&v[r as usize * cols..(r as usize + 1) * cols]) {
                        if t != 0.0 {
                            *a = 1.0;
                        }
                    }
                }
                any
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => self.download(&engine.nonzero_columns(x, &self.upload_indices(rows)?)?)?.iter().copied().collect(),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => self.download(&engine.nonzero_columns(x, &self.upload_indices(rows)?)?)?.iter().copied().collect(),
        };
        Ok(flags.iter().enumerate().filter(|(_, f)| **f != 0.0).map(|(c, _)| c as u32).collect())
    }

    /// Per row of `mask` (rows × groups), the columns `starts[g]..starts[g + 1]` of every group `g`
    /// whose entry is not zero (a NaN counts): a gate's components turned on in each row, spread
    /// over the columns of the value it gates (`starts` increasing from 0, one more than the
    /// groups). CUDA counts and fills the lists on the device, reading back one count a row.
    pub fn row_lists(&self, mask: &Tensor, starts: &[u32]) -> Result<RowLists, GpuError> {
        if starts.len() != mask.cols + 1 || starts.first() != Some(&0) || starts.windows(2).any(|w| w[0] > w[1]) {
            return Err(shape(format!("{} group starts for {} groups", starts.len(), mask.cols)));
        }
        let cols = starts[mask.cols] as usize;
        let rows = u32::try_from(mask.rows).map_err(|_| shape(format!("{} rows of row lists", mask.rows)))?;
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => {
                let starts = self.upload_indices(starts)?;
                let counts = self.download(&engine.row_counts(mask, &starts)?)?;
                let mut offsets = Vec::with_capacity(mask.rows + 1);
                offsets.push(0u32);
                for &n in &counts {
                    let next = u64::from(offsets[offsets.len() - 1]) + n as u64;
                    offsets.push(u32::try_from(next).map_err(|_| shape(format!("{next} listed entries")))?);
                }
                let entries = offsets[mask.rows] as usize;
                let offsets = self.upload_indices(&offsets)?;
                let (columns, row_of) = engine.row_fill(mask, &starts, &offsets, entries)?;
                Ok(RowLists { rows: mask.rows, cols, offsets, columns, row_of })
            }
            _ => {
                let m = self.download(mask)?;
                let (mut offsets, mut columns, mut row_of) = (vec![0u32], Vec::new(), Vec::new());
                for (r, row) in (0..rows).zip(m.rows()) {
                    for (g, &v) in row.iter().enumerate() {
                        if v != 0.0 {
                            columns.extend(starts[g]..starts[g + 1]);
                            row_of.extend(std::iter::repeat_n(r, (starts[g + 1] - starts[g]) as usize));
                        }
                    }
                    offsets.push(u32::try_from(columns.len()).map_err(|_| shape(format!("{} listed entries", columns.len())))?);
                }
                Ok(RowLists {
                    rows: mask.rows,
                    cols,
                    offsets: self.upload_indices(&offsets)?,
                    columns: self.upload_indices(&columns)?,
                    #[cfg(any(target_os = "linux", target_os = "macos"))]
                    row_of: self.upload_indices(&row_of)?,
                })
            }
        }
    }

    /// `x Aᵀ` (`x` rows × k, `A` n × k) at the entries `lists` holds, zero at every other: each
    /// listed entry `(r, c)` is `Σ_t x[r, t] A[c, t]`, summed in the tensors' own precision
    /// (a product read only where a gate is on, or a cotangent wanted only there).
    pub fn sampled_product(&self, x: &Tensor, a: &Tensor, lists: &RowLists) -> Result<Tensor, GpuError> {
        if x.cols != a.cols || lists.dim() != (x.rows, a.rows) || x.storage() != a.storage() || x.storage() == Storage::Bf16 {
            return Err(shape(format!("a sampled product of {:?} and {:?} at {:?} lists", x.dim(), a.dim(), lists.dim())));
        }
        let mut out = self.zeros(x.rows, a.rows)?;
        match &*self.backend {
            Backend::Host => {
                // Rows on the rayon pool, each its listed entries' dot products.
                let (xv, av, k, n) = (host(x)?, host(a)?, x.cols, a.rows);
                let (offsets, columns) = (host_indices(&lists.offsets)?, host_indices(&lists.columns)?);
                host_mut(&mut out)?.par_chunks_mut(n.max(1)).take(x.rows).enumerate().for_each(|(r, row)| {
                    let xr = &xv[r * k..(r + 1) * k];
                    for &c in &columns[offsets[r] as usize..offsets[r + 1] as usize] {
                        let c = c as usize;
                        row[c] = xr.iter().zip(&av[c * k..(c + 1) * k]).map(|(p, q)| p * q).sum();
                    }
                });
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.sampled_product(x, a, lists, &mut out)?,
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.sampled_product(x, a, lists, &mut out)?,
        }
        Ok(out)
    }

    /// `v A` (`v` rows × n read only at the entries `lists` holds, `A` n × m): row `r` is
    /// `Σ_c v[r, c] A[c, :]` over its listed columns `c`, summed in the tensors' own precision.
    pub fn listed_product(&self, v: &Tensor, lists: &RowLists, a: &Tensor) -> Result<Tensor, GpuError> {
        if lists.dim() != v.dim() || v.cols != a.rows || v.storage() != a.storage() || v.storage() == Storage::Bf16 {
            return Err(shape(format!("a listed product of {:?} at {:?} lists and {:?}", v.dim(), lists.dim(), a.dim())));
        }
        match &*self.backend {
            Backend::Host => {
                // Rows on the rayon pool, each adding its listed entries' rows of A in turn (A's rows
                // read whole, in order).
                let (vv, av, n, m) = (host(v)?, host(a)?, v.cols, a.cols);
                let (offsets, columns) = (host_indices(&lists.offsets)?, host_indices(&lists.columns)?);
                let mut out = vec![0.0; v.rows * m];
                out.par_chunks_mut(m.max(1)).take(v.rows).enumerate().for_each(|(r, row)| {
                    for &c in &columns[offsets[r] as usize..offsets[r + 1] as usize] {
                        let (s, ar) = (vv[r * n + c as usize], &av[c as usize * m..(c as usize + 1) * m]);
                        row.iter_mut().zip(ar).for_each(|(o, &t)| *o += s * t);
                    }
                });
                Ok(Tensor { rows: v.rows, cols: m, data: Data::Host(out) })
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.listed_product(v, lists, a),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.listed_product(v, lists, a),
        }
    }

    /// `vᵀ y` (`v` rows × n read only at listed entries, `y` rows × m) from per-group lists of rows
    /// (`lists`, groups × rows: [`Device::row_lists`] of the transposed mask, one column a row) and
    /// each column's group `group_of`: row `c` of the result is `Σ_r v[r, c] y[r, :]` over the rows
    /// `r` its group lists, summed in the tensors' own precision (a weight's gradient from a
    /// cotangent or value zero off its rows' components).
    pub fn listed_product_t(&self, v: &Tensor, lists: &RowLists, group_of: &Indices, y: &Tensor) -> Result<Tensor, GpuError> {
        if lists.cols != v.rows || group_of.len != v.cols || v.rows != y.rows || v.storage() != y.storage() || v.storage() == Storage::Bf16 {
            return Err(shape(format!("a transposed listed product of {:?} at {:?} lists and {:?}", v.dim(), lists.dim(), y.dim())));
        }
        match &*self.backend {
            Backend::Host => {
                let (vv, yv, n, m) = (host(v)?, host(y)?, v.cols, y.cols);
                let (offsets, rows, groups) = (host_indices(&lists.offsets)?, host_indices(&lists.columns)?, host_indices(group_of)?);
                if groups.iter().any(|&g| g as usize >= lists.rows) {
                    return Err(shape(format!("a group of {} lists", lists.rows)));
                }
                // Columns on the rayon pool, each adding its group's rows of y in turn.
                let mut out = vec![0.0; n * m];
                out.par_chunks_mut(m.max(1)).take(n).enumerate().for_each(|(c, row)| {
                    let g = groups[c] as usize;
                    for &r in &rows[offsets[g] as usize..offsets[g + 1] as usize] {
                        let (s, yr) = (vv[r as usize * n + c], &yv[r as usize * m..(r as usize + 1) * m]);
                        row.iter_mut().zip(yr).for_each(|(o, &t)| *o += s * t);
                    }
                });
                Ok(Tensor { rows: n, cols: m, data: Data::Host(out) })
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.listed_product_t(v, lists, group_of, y),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.listed_product_t(v, lists, group_of, y),
        }
    }

    /// A block's input norm per row of `y` (rows × d), an affine gain of an RMS norm
    /// `N(y) = γ ⊙ y r + β`, `r = (Σ_k y_k² / d + ε)^{−1/2}`, or its tangent along `along` or its
    /// pullback of `along` ([`RowNorm`]), with the gain `γ` and bias `β` as rows (1 × d). Every sum
    /// runs in index order, each product and sum rounded on its own, so in float64 the values are
    /// the host's iterator expressions' bit for bit (`interchange`'s cuts).
    pub fn row_norm(&self, mode: RowNorm, y: &Tensor, along: Option<&Tensor>, (gain, bias): (&Tensor, Option<&Tensor>), epsilon: f64) -> Result<Tensor, GpuError> {
        let (rows, d) = y.dim();
        let wanted = mode != RowNorm::Apply;
        if gain.dim() != (1, d) || bias.is_some_and(|b| b.dim() != (1, d)) || along.is_some() != wanted || along.is_some_and(|t| t.dim() != (rows, d)) {
            return Err(shape(format!("a row norm of {:?} with a {:?} gain", y.dim(), gain.dim())));
        }
        match &*self.backend {
            Backend::Host => {
                let (yv, gv) = (host(y)?, host(gain)?);
                let (tv, bv) = (along.map(host).transpose()?, bias.map(host).transpose()?);
                let mut out = vec![0.0; rows * d];
                for (r, o) in out.chunks_mut(d.max(1)).take(rows).enumerate() {
                    let s = &yv[r * d..(r + 1) * d];
                    let scale = 1.0 / (s.iter().map(|v| v * v).sum::<f64>() / d as f64 + epsilon).sqrt();
                    match mode {
                        RowNorm::Apply => o.iter_mut().enumerate().for_each(|(k, o)| *o = gv[k] * s[k] * scale + bv.map_or(0.0, |b| b[k])),
                        RowNorm::Tangent => {
                            let t = &tv.ok_or_else(|| shape("a tangent without its direction".to_string()))?[r * d..(r + 1) * d];
                            let dr = -scale * scale * scale * s.iter().zip(t).map(|(a, b)| a * b).sum::<f64>() / d as f64;
                            o.iter_mut().enumerate().for_each(|(k, o)| *o = gv[k] * (scale * t[k] + s[k] * dr));
                        }
                        RowNorm::Pullback => {
                            let w = &tv.ok_or_else(|| shape("a pullback without its cotangent".to_string()))?[r * d..(r + 1) * d];
                            let along: f64 = s.iter().zip(w).zip(gv).map(|((v, w), g)| g * w * v).sum::<f64>() * scale * scale * scale / d as f64;
                            o.iter_mut().enumerate().for_each(|(k, o)| *o = scale * gv[k] * w[k] - along * s[k]);
                        }
                    }
                }
                Ok(Tensor { rows, cols: d, data: Data::Host(out) })
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.row_norm(mode, y, along, (gain, bias), epsilon),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.row_norm(mode, y, along, (gain, bias), epsilon),
        }
    }

    /// `tᵀ`.
    pub fn transpose(&self, t: &Tensor) -> Result<Tensor, GpuError> {
        match &*self.backend {
            Backend::Host => {
                let v = host(t)?;
                let mut out = vec![0.0; t.len()];
                out.par_chunks_mut(t.rows.max(1)).take(t.cols).enumerate().for_each(|(c, row)| row.iter_mut().enumerate().for_each(|(r, o)| *o = v[r * t.cols + c]));
                Ok(Tensor { rows: t.cols, cols: t.rows, data: Data::Host(out) })
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.transpose(t),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.transpose(t),
        }
    }

    /// `out[r, j] = t[r, ids[j]]`: columns `ids` of `t`, in order.
    pub fn gather_columns(&self, t: &Tensor, ids: &Indices) -> Result<Tensor, GpuError> {
        match &*self.backend {
            Backend::Host => {
                let (v, cols, ids) = (host(t)?, t.cols, host_indices(ids)?);
                if let Some(id) = ids.iter().find(|id| **id as usize >= cols) {
                    return Err(shape(format!("column {id} of a {cols}-column tensor")));
                }
                let mut out = Vec::with_capacity(t.rows * ids.len());
                for row in v.chunks(cols.max(1)).take(t.rows) {
                    out.extend(ids.iter().map(|&c| row[c as usize]));
                }
                Ok(Tensor { rows: t.rows, cols: ids.len(), data: Data::Host(out) })
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.gather_columns(t, ids),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.gather_columns(t, ids),
        }
    }

    /// `t[r, ids[j]] = values[r, j]` (added to it when `accumulate`): [`Device::gather_columns`]'s
    /// inverse; `ids` distinct.
    pub fn scatter_columns(&self, t: &mut Tensor, ids: &Indices, values: &Tensor, accumulate: bool) -> Result<(), GpuError> {
        if values.dim() != (t.rows, ids.len) {
            return Err(shape(format!("{:?} values for {} columns of {:?}", values.dim(), ids.len, t.dim())));
        }
        match &*self.backend {
            Backend::Host => {
                let (cols, ids, v) = (t.cols, host_indices(ids)?.to_vec(), host(values)?.to_vec());
                if let Some(id) = ids.iter().find(|id| **id as usize >= cols) {
                    return Err(shape(format!("column {id} of a {cols}-column tensor")));
                }
                let m = ids.len();
                for (r, row) in host_mut(t)?.chunks_mut(cols.max(1)).enumerate() {
                    for (j, &c) in ids.iter().enumerate() {
                        let value = v[r * m + j];
                        row[c as usize] = if accumulate { row[c as usize] + value } else { value };
                    }
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.scatter_columns(t, ids, values, accumulate),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.scatter_columns(t, ids, values, accumulate),
        }
    }

    /// `t[ids[i], c] = values[i, c]` (added to it when `accumulate`): [`Device::gather_rows`]'s
    /// inverse; `ids` distinct.
    pub fn scatter_rows(&self, t: &mut Tensor, ids: &Indices, values: &Tensor, accumulate: bool) -> Result<(), GpuError> {
        if values.dim() != (ids.len, t.cols) {
            return Err(shape(format!("{:?} values for {} rows of {:?}", values.dim(), ids.len, t.dim())));
        }
        match &*self.backend {
            Backend::Host => {
                let (cols, rows, ids, v) = (t.cols, t.rows, host_indices(ids)?.to_vec(), host(values)?.to_vec());
                if let Some(id) = ids.iter().find(|id| **id as usize >= rows) {
                    return Err(shape(format!("row {id} of a {rows}-row tensor")));
                }
                let target = host_mut(t)?;
                for (i, &r) in ids.iter().enumerate() {
                    for c in 0..cols {
                        let at = r as usize * cols + c;
                        target[at] = if accumulate { target[at] + v[i * cols + c] } else { v[i * cols + c] };
                    }
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.scatter_rows(t, ids, values, accumulate),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.scatter_rows(t, ids, values, accumulate),
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

    /// [`Device::rms_norm`] and the values' bfloat16 copy, written in one pass on CUDA in f32
    /// (`rms_both`); elsewhere the values and [`Device::bf16_copy`] of them.
    pub fn rms_norm_both(&self, x: &Tensor, epsilon: f64) -> Result<(Tensor, Tensor), GpuError> {
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if x.storage() == Storage::F32 => engine.rms_both(x, epsilon),
            _ => {
                let values = self.rms_norm(x, epsilon)?;
                let half = self.bf16_copy(&values)?;
                Ok((values, half))
            }
        }
    }

    /// [`Device::law_values`] and the values' bfloat16 copy, written in one pass on CUDA in f32
    /// (`laws_both`); elsewhere the values and [`Device::bf16_copy`] of them.
    pub fn law_values_both(&self, x: &Tensor, codes: &Indices, c: f64) -> Result<(Tensor, Tensor), GpuError> {
        if codes.len != x.cols {
            return Err(shape(format!("{} law codes for {} columns", codes.len, x.cols)));
        }
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if x.storage() == Storage::F32 => engine.laws_both(x, codes, c),
            _ => {
                let values = self.law_values(x, codes, c)?;
                let half = self.bf16_copy(&values)?;
                Ok((values, half))
            }
        }
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

    /// [`Device::softmax_rows`] with its bfloat16 copy written in the same pass, each value the one
    /// [`Device::bf16_copy`] makes of the f32 weights. CUDA f32 storage; elsewhere the softmax and
    /// its copy.
    pub fn softmax_rows_bf16(&self, scores: &mut Tensor, causal: bool) -> Result<Tensor, GpuError> {
        let width = scores.cols;
        if width == 0 || (causal && scores.rows % width != 0) {
            return Err(shape(format!("attention scores {:?} are not square blocks", scores.dim())));
        }
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if scores.storage() == Storage::F32 => engine.softmax_rows_bf16(scores, causal, width),
            _ => {
                self.softmax_rows(scores, causal)?;
                self.bf16_copy(scores)
            }
        }
    }

    /// In place, each row of `scores` to its softmax, the rows taken in runs of `period` (a head's
    /// queries of one batch of [`Device::gemm_batched`]): when `causal`, row `r` reads only columns
    /// `j ≤ start + r mod period`, the rest zero. A segment's queries at positions
    /// `start..start + period` against its keys at positions `0..` (columns).
    pub fn softmax_rows_period(&self, scores: &mut Tensor, causal: bool, start: usize, period: usize) -> Result<(), GpuError> {
        if scores.cols == 0 || period == 0 || scores.rows % period != 0 || (causal && start.saturating_add(period) > scores.cols) {
            return Err(shape(format!("attention scores {:?} in runs of {period} from position {start}", scores.dim())));
        }
        self.softmax_rows_impl(scores, causal, start, period)
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

    /// [`Device::softmax_backward`] in bfloat16, each value the one [`Device::bf16_copy`] makes of
    /// the f32 map, written so as it is computed (CUDA f32 storage; elsewhere the map and its copy).
    pub fn softmax_backward_bf16(&self, alpha: &Tensor, d: &Tensor) -> Result<Tensor, GpuError> {
        same(alpha, d, "softmax backward")?;
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if alpha.storage() == Storage::F32 => engine.softmax_backward_bf16(alpha, d),
            _ => self.bf16_copy(&self.softmax_backward(alpha, d)?),
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
            #[cfg(any(target_os = "linux", target_os = "macos"))]
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

    /// In place of `probabilities` (each row a softmax `π`), the Fisher probe's cotangent per row,
    /// `b = √π ⊙ ξ − π (√π · ξ)`, `ξ` the row's signs: row `r`'s are those of row `first + r` under
    /// `key` ([`fisher_sign`]), so a probe made one row tile at a time is the probe of the whole.
    /// With `L = diag √π − π √πᵀ`, `b = L ξ` and `L Lᵀ = diag π − π πᵀ` (as `Σ π = 1`), so
    /// `E[b bᵀ]` is the softmax's Fisher matrix, and for any `a`, `a · b = (Lᵀ a) · ξ` has
    /// `E[(a · b)⁴] = 3 c² − 2 Σ_k (Lᵀ a)_k⁴` with `c = E[(a · b)²]`: `(a · b)²` estimates `c` with
    /// relative variance at most 2 whatever `π` is. A row whose `scored` flag is zero becomes zero.
    pub fn fisher_probe_cotangent(&self, probabilities: &mut Tensor, (key, first): (u64, usize), scored: Option<&Indices>) -> Result<(), GpuError> {
        if let Some(s) = scored
            && s.len != probabilities.rows
        {
            return Err(shape(format!("{} row flags for {} rows", s.len, probabilities.rows)));
        }
        match &*self.backend {
            Backend::Host => {
                let cols = probabilities.cols;
                let flags = scored.map(host_indices).transpose()?;
                for (r, row) in host_mut(probabilities)?.chunks_mut(cols.max(1)).enumerate() {
                    if flags.is_some_and(|f| f[r] == 0) {
                        row.fill(0.0);
                        continue;
                    }
                    let at = (first + r) as u64;
                    let roots: Vec<f64> = row.iter().enumerate().map(|(c, p)| p.sqrt() * fisher_sign(key, at, c as u64)).collect();
                    let dot: f64 = roots.iter().sum();
                    for (b, root) in row.iter_mut().zip(&roots) {
                        *b = root - *b * dot;
                    }
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.fisher_probe_cotangent(probabilities, (key, first), scored),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.fisher_probe_cotangent(probabilities, (key, first), scored),
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

    /// `f(x)` entrywise for a gate's maps ([`GateFunction`]); `s` is the scale of the ones that take
    /// one (entrywise, `x`'s shape) and is ignored by the others.
    pub fn gate_function(&self, function: GateFunction, x: &Tensor, s: Option<&Tensor>) -> Result<Tensor, GpuError> {
        if let Some(s) = s {
            same(x, s, "gate scale")?;
        }
        if function.scaled() && s.is_none() {
            return Err(shape(format!("{function:?} needs a scale")));
        }
        match &*self.backend {
            Backend::Host => {
                let (xv, sv) = (host(x)?, s.map(host).transpose()?);
                let out = xv.iter().enumerate().map(|(i, &t)| function.host(t, sv.map_or(1.0, |s| s[i]))).collect();
                Ok(Tensor { rows: x.rows, cols: x.cols, data: Data::Host(out) })
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.gate_function(function, x, s),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.gate_function(function, x, s),
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
        self.head_sweep(hidden, (head, transposed), scored, expected, None, arithmetic)
    }

    /// [`Device::head_log_partition`] (the head and whether it is transposed as one argument)
    /// with `expected`, and with the probe `(key, probed)` also each row's Fisher probe pulled
    /// back to `hidden_r`, written into `probed` (rows × width): `Σ_c b_rc e_c`, `b_r` the probe's
    /// cotangent at the row's softmax `q_r` under `key`, rows numbered from zero
    /// ([`Device::fisher_probe_cotangent`]), and zero on an unscored row. The host and the Apple
    /// GPU make it from each row tile's probabilities (`fisher_probe_cotangent`, then one more
    /// product). CUDA f32 storage never holds a row's whole softmax: alongside the sweep's running
    /// largest logit `m_r` and sum `s_r = Σ_c exp(z_rc − m_r)` it accumulates
    /// `A_r = Σ_c exp((z_rc − m_r) / 2) ξ_rc e_c` (one more product per chunk) and
    /// `a_r = Σ_c exp((z_rc − m_r) / 2) ξ_rc` (in double), both rescaled by `exp((m_old − m_new) / 2)`
    /// as the largest grows, and ends with `(A_r − a_r μ_r) / √s_r`, `μ_r` the expected row: as
    /// `√q_rc = exp((z_rc − m_r) / 2) / √s_r`, that is `Σ_c √q_rc ξ_rc e_c − (√q_r · ξ_r) μ_r`.
    pub fn head_log_partition_probed(
        &self,
        hidden: &Tensor,
        head: (&Tensor, bool),
        scored: Option<&Indices>,
        expected: &mut Tensor,
        probe: (u64, &mut Tensor),
        arithmetic: Arithmetic,
    ) -> Result<Vec<f64>, GpuError> {
        self.head_sweep(hidden, head, scored, Some(expected), Some(probe), arithmetic)
    }

    /// [`Device::head_log_partition`] and [`Device::head_log_partition_probed`].
    fn head_sweep(
        &self,
        hidden: &Tensor,
        (head, transposed): (&Tensor, bool),
        scored: Option<&Indices>,
        expected: Option<&mut Tensor>,
        probe: Option<(u64, &mut Tensor)>,
        arithmetic: Arithmetic,
    ) -> Result<Vec<f64>, GpuError> {
        let (rows, width) = hidden.dim();
        let classes = if transposed { head.cols } else { head.rows };
        let head_width = if transposed { head.rows } else { head.cols };
        if head_width != width || classes == 0 || scored.is_some_and(|s| s.len != rows) || expected.as_ref().is_some_and(|e| e.dim() != (rows, width)) {
            return Err(shape(format!("a {:?} head (transposed {transposed}) on {:?} rows", head.dim(), hidden.dim())));
        }
        if let Some((_, probed)) = &probe
            && (probed.dim() != (rows, width) || expected.is_none())
        {
            return Err(shape(format!("a {:?} probe for {:?} rows, which takes the expected rows too", probed.dim(), hidden.dim())));
        }
        if rows == 0 {
            return Ok(Vec::new());
        }
        match &*self.backend {
            Backend::Host => {
                let flags = scored.map(host_indices).transpose()?;
                self.head_log_partition_tiled(hidden, (head, transposed), flags, expected, probe, arithmetic)
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if hidden.storage() == Storage::F32 => {
                let chunk = swept_chunk(rows);
                engine.head_log_partition(hidden, (head, transposed), scored, (expected, probe), (chunk, arithmetic))
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(_) => {
                let mut logits = self.zeros(rows, classes)?;
                let (into, back) = if transposed { (Op::N, Op::T) } else { (Op::T, Op::N) };
                self.gemm(&mut logits, 1.0, hidden, Op::N, head, into, 0.0, arithmetic)?;
                let stats = self.softmax_stats_rows(&mut logits, scored)?;
                if let Some(out) = expected {
                    self.gemm(out, 1.0, &logits, Op::N, head, back, 0.0, arithmetic)?;
                    if let Some((key, probed)) = probe {
                        self.fisher_probe_cotangent(&mut logits, (key, 0), scored)?;
                        self.gemm(probed, 1.0, &logits, Op::N, head, back, 0.0, arithmetic)?;
                    }
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
                self.head_log_partition_tiled(hidden, (head, transposed), flags, expected, probe, arithmetic)
            }
        }
    }

    /// [`Device::head_sweep`] in row tiles whose logits fill about 32 MB (the size of CUDA's swept
    /// chunks), through this device's own products, softmax statistics and Fisher probes; `flags`
    /// are the scored rows' flags.
    fn head_log_partition_tiled(
        &self,
        hidden: &Tensor,
        (head, transposed): (&Tensor, bool),
        flags: Option<&[u32]>,
        mut expected: Option<&mut Tensor>,
        mut probe: Option<(u64, &mut Tensor)>,
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
                if let Some((key, probed)) = probe.as_mut() {
                    // The tile's probabilities become its rows' probes (row `start + r` of the
                    // whole for its row `r`), pulled back through the head.
                    self.fisher_probe_cotangent(&mut logits, (*key, start), part.as_ref())?;
                    let mut pulled = self.zeros(n, width)?;
                    self.gemm(&mut pulled, 1.0, &logits, Op::N, head, back, 0.0, arithmetic)?;
                    self.set_rows(probed, start, &pulled)?;
                }
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
                let chunk = swept_chunk(hidden.rows);
                engine.head_log_partition_into(hidden, (head, transposed), scored, (expected, None), out, (chunk, arithmetic))
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

    /// Rows `at..at + rows` of `y` plus `alpha` times rows `from..from + rows` of `x`, in place: each
    /// entry the one [`Device::axpy`] makes of the rows copied out ([`Device::rows_of`]) and written
    /// back ([`Device::set_rows`]), in one pass.
    pub fn axpy_rows(&self, y: &mut Tensor, at: usize, alpha: f64, (x, from): (&Tensor, usize), rows: usize) -> Result<(), GpuError> {
        if x.cols != y.cols || at + rows > y.rows || from + rows > x.rows {
            return Err(shape(format!("{rows} rows from row {from} of {:?} into row {at} of {:?}", x.dim(), y.dim())));
        }
        let (to, source, n) = (at * y.cols, from * x.cols, rows * y.cols);
        match &*self.backend {
            Backend::Host => {
                for (yv, xv) in host_mut(y)?[to..to + n].iter_mut().zip(&host(x)?[source..source + n]) {
                    *yv += alpha * xv;
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.axpy_rows(y, to, alpha, (x, source), n),
            #[cfg(target_os = "macos")]
            Backend::Metal(_) => {
                let mut total = self.rows_of(y, at, rows)?;
                self.axpy(&mut total, alpha, &self.rows_of(x, from, rows)?)?;
                self.set_rows(y, at, &total)
            }
        }
    }

    /// [`Device::axpy_rows`] within one tensor: its rows `at..at + rows` plus `alpha` times its rows
    /// `from..from + rows`, the two ranges apart.
    pub fn axpy_rows_within(&self, t: &mut Tensor, at: usize, alpha: f64, from: usize, rows: usize) -> Result<(), GpuError> {
        apart(t, at, from, rows)?;
        let (to, source, n) = (at * t.cols, from * t.cols, rows * t.cols);
        match &*self.backend {
            Backend::Host => {
                let values = host_mut(t)?;
                for i in 0..n {
                    values[to + i] += alpha * values[source + i];
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.rows_within(t, (to, source, n), Some(alpha)),
            #[cfg(target_os = "macos")]
            Backend::Metal(_) => {
                let mut total = self.rows_of(t, at, rows)?;
                self.axpy(&mut total, alpha, &self.rows_of(t, from, rows)?)?;
                self.set_rows(t, at, &total)
            }
        }
    }

    /// Rows `from..from + rows` of `t` copied over its rows `to..to + rows`, the two ranges apart:
    /// [`Device::rows_of`] and [`Device::set_rows`] in one pass.
    pub fn copy_rows_within(&self, t: &mut Tensor, to: usize, from: usize, rows: usize) -> Result<(), GpuError> {
        apart(t, to, from, rows)?;
        let (at, source, n) = (to * t.cols, from * t.cols, rows * t.cols);
        match &*self.backend {
            Backend::Host => {
                host_mut(t)?.copy_within(source..source + n, at);
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.rows_within(t, (at, source, n), None),
            #[cfg(target_os = "macos")]
            Backend::Metal(_) => {
                let copied = self.rows_of(t, from, rows)?;
                self.set_rows(t, to, &copied)
            }
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
        // The permutation writes every entry of its output: no zeroing first.
        let mut out = self.empty(rows * heads, width)?;
        self.heads(x, &mut out, (start, heads, width, rows / blocks), turn, inverse, false)?;
        Ok(out)
    }

    /// [`Device::split_heads`] in bfloat16, each value the one [`Device::bf16_copy`] makes of the f32
    /// split, written so as it is computed (CUDA f32 storage; elsewhere the split and its copy).
    pub fn split_heads_bf16(&self, x: &Tensor, start: usize, heads: usize, width: usize, blocks: usize, turn: Option<(&Tensor, &Tensor, bool)>, inverse: bool) -> Result<Tensor, GpuError> {
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if x.storage() == Storage::F32 => {
                let (length, planes) = split_shape(x, (start, heads, width, blocks), turn)?;
                Ok(engine.split_halves(x, (start, heads, width, length, planes), turn, inverse, false)?.1)
            }
            _ => self.bf16_copy(&self.split_heads(x, start, heads, width, blocks, turn, inverse)?),
        }
    }

    /// [`Device::split_heads`] written twice in one pass, in f32 and in bfloat16 (each bfloat16
    /// value the one [`Device::bf16_copy`] makes of the f32 split), for an operand read both ways:
    /// `(f32, bfloat16)`. CUDA f32 storage; elsewhere the split and its copy.
    pub fn split_heads_both(&self, x: &Tensor, start: usize, heads: usize, width: usize, blocks: usize, turn: Option<(&Tensor, &Tensor, bool)>, inverse: bool) -> Result<(Tensor, Tensor), GpuError> {
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if x.storage() == Storage::F32 => {
                let (length, planes) = split_shape(x, (start, heads, width, blocks), turn)?;
                let (wide, half) = engine.split_halves(x, (start, heads, width, length, planes), turn, inverse, true)?;
                Ok((wide.ok_or_else(|| shape("a split without its f32 values".to_string()))?, half))
            }
            _ => {
                let split = self.split_heads(x, start, heads, width, blocks, turn, inverse)?;
                let half = self.bf16_copy(&split)?;
                Ok((split, half))
            }
        }
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

    /// Each mean `μ` rounded to the nearest multiple of `2^⌊log2 σ⌋`, `σ = exp(s)`, into `out` (a
    /// removed entry, `s = −∞`, keeps its mean): the posterior mean to the precision the posterior
    /// resolves (`library_mdl::Posterior::rounded`). Formed in float64 from the stored values on
    /// CUDA and the host (the step a power of two, so the result is exact in the storage); in f32
    /// on the Apple GPU.
    pub fn round_to_deviation(&self, out: &mut Tensor, (mean, log_sd): (&Tensor, &Tensor)) -> Result<(), GpuError> {
        same(out, mean, "rounded mean")?;
        same(out, log_sd, "rounded mean's log standard deviation")?;
        match &*self.backend {
            Backend::Host => {
                let (mv, sv) = (host(mean)?, host(log_sd)?);
                for (i, t) in host_mut(out)?.iter_mut().enumerate() {
                    *t = if sv[i].is_finite() {
                        let step = (sv[i] / std::f64::consts::LN_2).floor().exp2();
                        (mv[i] / step).round() * step
                    } else {
                        mv[i]
                    };
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.round_to_deviation(out, (mean, log_sd)),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.round_to_deviation(out, (mean, log_sd)),
        }
    }

    /// A weight sample of a factorized Gaussian posterior, `θᵢ = μᵢ + exp(sᵢ) εᵢ` with `εᵢ =
    /// [`posterior_normal`]`(key, stream, i)`, written into `theta` (this device's storage) from
    /// the posterior's means `μ` and log standard deviations `s` (`mean`, `log_sd`, in `theta`'s
    /// storage, or f32 for a bfloat16 `theta` on CUDA, written rounded to nearest). A removed entry
    /// (`s = −∞`, `μ = 0`) samples 0.
    /// The draws are regenerated from their counters, never stored ([`Device::posterior_ivon`]
    /// regenerates the same ones).
    pub fn reparameterize(&self, theta: &mut Tensor, (mean, log_sd): (&Tensor, &Tensor), (key, stream): (u64, u64)) -> Result<(), GpuError> {
        same(theta, mean, "reparameterized mean")?;
        self.reparameterize_block(theta, (0, 0), (mean, log_sd), (key, stream))
    }

    /// [`Device::reparameterize`] written into the block of `out` at `(row, col)`: entry `(r, c)`
    /// of the operator's sample goes to `out[(row + r, col + c)]`, so an operator's sample lands
    /// straight in a stacked operand (a fused group's rows or columns) with no
    /// copy. The draws are the operator's own (`ε` of its entry `i = r · cols + c`), the same
    /// whether it is written whole or into a block. `out` in f32, or bfloat16 beside f32 masters,
    /// on CUDA; in f32 on the Apple GPU; in float64 on the host.
    pub fn reparameterize_block(&self, out: &mut Tensor, (row, col): (usize, usize), (mean, log_sd): (&Tensor, &Tensor), (key, stream): (u64, u64)) -> Result<(), GpuError> {
        same(mean, log_sd, "reparameterized log standard deviation")?;
        if row + mean.rows > out.rows || col + mean.cols > out.cols {
            return Err(shape(format!("a {:?} sample at ({row}, {col}) of {:?}", mean.dim(), out.dim())));
        }
        if mean.len() == 0 {
            return Ok(());
        }
        let (cols, stride, at) = (mean.cols, out.cols, row * out.cols + col);
        match &*self.backend {
            Backend::Host => {
                let (mv, sv) = (host(mean)?, host(log_sd)?);
                let theta = host_mut(out)?;
                for i in 0..mv.len() {
                    theta[at + (i / cols) * stride + i % cols] = mv[i] + sv[i].exp() * f64::from(posterior_normal(key, stream, i as u64));
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.reparameterize_block(out, (at, stride), (mean, log_sd), (key, stream)),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.reparameterize_block(out, (at, stride), (mean, log_sd), (key, stream)),
        }
    }

    /// An empty set of weight samples of `key` ([`Device::add_sample`]).
    #[must_use]
    pub fn samples(&self, key: u64) -> Samples {
        Samples { key, jobs: Vec::new() }
    }

    /// [`Device::reparameterize_block`] with `samples`' key, gathered into `samples`: on CUDA it is
    /// written when [`Device::run_samples`] runs them, every sample of one output storage by one
    /// launch (a sample's own launch costs more than its entries for an operator of the sizes a
    /// library holds); elsewhere it is written now. Each entry is the one
    /// [`Device::reparameterize_block`] writes.
    ///
    /// # Safety
    ///
    /// On CUDA, until `run_samples` runs `samples`, `out`, `mean` and `log_sd` stay allocated (none
    /// is dropped or replaced), and nothing enqueued reads `out`'s block or writes `mean` or
    /// `log_sd`.
    // SAFETY: the contract above is the caller's; this records addresses, and run_samples writes
    // through them.
    pub unsafe fn add_sample(&self, samples: &mut Samples, out: &mut Tensor, (row, col): (usize, usize), (mean, log_sd): (&Tensor, &Tensor), stream: u64) -> Result<(), GpuError> {
        same(mean, log_sd, "sampled log standard deviation")?;
        if row + mean.rows > out.rows || col + mean.cols > out.cols {
            return Err(shape(format!("a {:?} sample at ({row}, {col}) of {:?}", mean.dim(), out.dim())));
        }
        if mean.len() == 0 {
            return Ok(());
        }
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => {
                let job = engine.sample_job(out, (row * out.cols + col, out.cols), (mean, log_sd), stream)?;
                samples.jobs.push(job);
                Ok(())
            }
            _ => self.reparameterize_block(out, (row, col), (mean, log_sd), (samples.key, stream)),
        }
    }

    /// Writes the samples gathered in `samples` ([`Device::add_sample`]).
    pub fn run_samples(&self, samples: Samples) -> Result<(), GpuError> {
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.run_samples(samples.key, &samples.jobs),
            // Elsewhere every sample was written as it was gathered.
            _ if samples.jobs.is_empty() => Ok(()),
            _ => Err(shape("samples gathered on another device".to_string())),
        }
    }

    /// One step of the improved variational online Newton method (IVON; Shen et al., ICML 2024,
    /// arXiv 2402.17641, Algorithm 1) on the factorized Gaussian posterior `N(μ, exp(s)²)`, with the
    /// data term's curvature taken in the Gauss–Newton approximation. `gradient` times
    /// `step.gradient_scale` is the gradient `g` per token at the posterior's sample of the data
    /// term (and of any other term of the objective the caller adds into it). `factor` is a draw of
    /// the Gauss–Newton factor: `u = Σ_t J_tᵀ b_t` over a batch's scored tokens, `b_t` the Fisher
    /// probe at the model's own prediction (`Device::fisher_probe_cotangent`). Its square times
    /// `step.factor_scale` (one over the tokens) is `ĥ_D = u² / n`, an unbiased estimate of the
    /// diagonal of the data term's Gauss–Newton matrix per token (`E[u uᵀ] = Σ_t J_tᵀ F_t J_t`,
    /// `F_t` the softmax's Fisher matrix, the probes' signs independent across tokens). The
    /// Gauss–Newton matrix is positive semidefinite; the data term's Hessian adds the logits' second
    /// derivatives weighted by the residual of the predictions, and the step uses the Gauss–Newton
    /// matrix in its place. The step is therefore IVON's step for the objective with that curvature,
    /// not for the exact one. `prior` is an estimate `ĥ_R` of another term's diagonal curvature per
    /// token, of either sign, and the step's curvature estimate is `ĥ = ĥ_D + ĥ_R`. An absent
    /// `gradient`, `factor` or `prior` is zero (an operator a batch does not reach).
    ///
    /// With `N = step.tokens`, `v` the entry's group variance (`variance`, groups × 1),
    /// `δ = 1 / (N v)` the prior's precision per token, `W` the momentum's bias correction before
    /// the step (`step.weight`) and `W' = β₁ W + (1 − β₁)` after it: `m ← β₁ m + (1 − β₁) g`, and
    /// the curvature the running average `h ← β₂ h + (1 − β₂) ĥ` of the estimates, unbiased for the
    /// curvature's diagonal. IVON's update of the total precision `λ = h + δ`,
    /// `λ ← λ (1 + x + ½ x²)` with `x = (1 − β₂)(ĥ + δ − λ) / λ`, adds
    /// `½ (1 − β₂)² (ĥ − h)² / (h + δ)` to keep `λ` positive for a Hessian estimate of either sign;
    /// that term's mean `½ (1 − β₂) Var(ĥ) / (h + δ)` raised `h` without bound under one batch's
    /// heavy-tailed `ĥ = u² / n` (vpd4l, `N = 2^16`: one batch's outlying estimate multiplied `h`
    /// and the description rose from 25 to 152 bits per scored token in one epoch). The step is
    /// `d = G / (h₀⁺ + δ)` with `G = m / W' + δ μ` (the bias-corrected momentum plus the prior's
    /// exact pull), `h₀` the curvature before this step's update and `x⁺ = max(x, 0)`: IVON's full
    /// step from the mean, written to `direction` with the mean left as it is (a caller moves it
    /// by `η d`, [`Device::posterior_finish`]). The direction is formed before this step's factor
    /// `u` enters the curvature, so given the batch and the weight sample it does not depend on the
    /// probe that draws `u`, and `E[c (u · d)²] = dᵀ G_n d` exactly, `G_n` the Gauss–Newton matrix
    /// per token of the tokens `u` sums and `c = step.factor_scale`: the curvature along `d` that
    /// the caller's step length reads from `u · d`. From the updated `h` an entry with a large
    /// `u_i²` would take a short `d_i`, and `(u · d)²` would be biased low. The deviation is
    /// `s = −½ ln(N (h⁺ + δ))` with the updated `h`, so `σ² ≤ v`: the standard deviation at which
    /// the approximated `N E_q[ℓ] + KL(q ‖ p)` is stationary for a curvature `h ≥ 0` and the
    /// variance `v`. A Gauss–Newton estimate is never negative, so without `prior` an `h ≥ 0`
    /// stays nonnegative (`β₂ h + (1 − β₂) ĥ ≥ 0`; rounding keeps it, since `|fl(ĥ − h)| ≤ h`
    /// when `ĥ < h`). Where the expected curvature is negative (a prior term's curvature, `prior`,
    /// can make it so), `h` keeps the signed average and `σ² = v` is the prior's: at a fixed `v`
    /// an `h` in `(−1 / (N v), 0)` would be stationary at `σ² = 1 / (N h + 1 / v) > v`, but `v` is
    /// fitted to `μ² + σ²`, and jointly in `σ` and `v` there is no stationary point (along
    /// `σ² = v → ∞` the term `N h σ² / 2` falls without bound). The step's sums read `h₀⁺ ≥ 0`
    /// alike, so the line step's curvature `ρ̄ Σ h₀⁺ d² + Σ δ d²` is positive. Both `h` and `v` depend on the
    /// posterior (`h` is an expectation under `q`, `v` is the empirical-Bayes variance), so the
    /// exact stationary point solves implicit equations; this step, from the running `h` and the
    /// current `v`, is an online approximation to it.
    ///
    /// Per group into `sums` (groups × 5; float64 on CUDA and the host, f32 on the Apple GPU, as
    /// are `variance` and [`Device::group_divergence`]'s outputs) the terms of the step over its
    /// live entries: along `d`, `d · d`, `u · d` and `Σ h₀⁺ d²`; along the direction before the
    /// step, `d₀ = (m₀ / W + δ μ) / (h₀⁺ + δ)` from the momentum `m₀` and the curvature `h₀` before
    /// their updates (`m₀ / W = 0` while `W = 0`), the slope `(g + δ μ) · d₀` of this step's
    /// gradient and the slope `Σ (h₀⁺ + δ) d₀²` of the gradient `d₀` was made from (the line
    /// step's measurements, `gam_mpd::device_posterior::DevicePosterior::step`). Every entry is in
    /// the masters' storage. A removed entry (`s = −∞`) is left alone, its `d` zero, and adds
    /// nothing to `sums`.
    pub fn posterior_ivon(
        &self,
        (mean, log_sd): (&Tensor, &mut Tensor),
        [momentum, curvature]: [&mut Tensor; 2],
        (gradient, factor, prior): (Option<&Tensor>, Option<&Tensor>, Option<&Tensor>),
        (groups, variance): (&GroupMap, &Tensor),
        (direction, sums): (&mut Tensor, &mut Tensor),
        step: &PosteriorStep,
    ) -> Result<(), GpuError> {
        groups.check(mean, "posterior")?;
        same(mean, direction, "posterior direction")?;
        same(mean, log_sd, "posterior log standard deviation")?;
        same(mean, momentum, "posterior momentum")?;
        same(mean, curvature, "posterior curvature")?;
        for (input, what) in [(gradient, "posterior gradient"), (factor, "posterior Gauss–Newton factor"), (prior, "posterior prior curvature")] {
            if let Some(input) = input {
                same(mean, input, what)?;
            }
        }
        if variance.cols != 1 || sums.dim() != (variance.rows, 5) {
            return Err(shape(format!("a {:?} variance and {:?} sums for {} entries", variance.dim(), sums.dim(), mean.len())));
        }
        if !(step.tokens.is_finite() && step.tokens > 0.0) {
            return Err(shape(format!("{} tokens", step.tokens)));
        }
        // Decays in [0, 1): a running average of what it is given, and a momentum that moves.
        if !((0.0..1.0).contains(&step.beta1) && (0.0..1.0).contains(&step.beta2)) {
            return Err(shape(format!("decays β₁ = {} and β₂ = {} outside [0, 1)", step.beta1, step.beta2)));
        }
        let (before, after) = (step.weight, step.correction());
        if !(before >= 0.0 && after > 0.0 && after.is_finite()) {
            return Err(shape(format!("a momentum bias correction {before} before the step and {after} after it")));
        }
        match &*self.backend {
            Backend::Host => {
                let (ms, hs) = (host_mut(momentum)?, host_mut(curvature)?);
                let (gv, uv, rv) = (gradient.map(host).transpose()?, factor.map(host).transpose()?, prior.map(host).transpose()?);
                let (ids, var, means) = (host_indices(&groups.ids)?, host(variance)?, host(mean)?);
                let (log_sds, moves, totals) = (host_mut(log_sd)?, host_mut(direction)?, host_mut(sums)?);
                if let Some(id) = ids.iter().find(|id| **id as usize >= var.len()) {
                    return Err(shape(format!("group {id} of {}", var.len())));
                }
                let (b1, b2) = (step.beta1, step.beta2);
                for i in 0..means.len() {
                    let mu = means[i];
                    if log_sds[i] == f64::NEG_INFINITY {
                        moves[i] = mu - mu;
                        continue;
                    }
                    let g = groups.group(ids, i) as usize;
                    let delta = 1.0 / (step.tokens * var[g]);
                    // The direction before the step, from the momentum and the curvature it holds.
                    let (m, h) = (ms[i], hs[i]);
                    let (bounded, held) = (h.max(0.0), h.max(0.0) + delta);
                    let previous = (if before > 0.0 { m / before } else { 0.0 } + delta * mu) / held;
                    let data = gv.map_or(0.0, |v| step.gradient_scale * v[i]);
                    let u = uv.map_or(0.0, |v| v[i]);
                    let estimate = step.factor_scale * u * u + rv.map_or(0.0, |v| v[i]);
                    ms[i] = b1 * m + (1.0 - b1) * data;
                    hs[i] = h + (1.0 - b2) * (estimate - h);
                    // The step's direction from the curvature before this step's draw, its deviation
                    // from the curvature after it.
                    let positive = hs[i].max(0.0);
                    let change = (ms[i] / after + delta * mu) / held;
                    log_sds[i] = -0.5 * (step.tokens * (positive + delta)).ln();
                    moves[i] = change;
                    totals[5 * g] += change * change;
                    totals[5 * g + 1] += u * change;
                    totals[5 * g + 2] += (bounded * change) * change;
                    totals[5 * g + 3] += (data + delta * mu) * previous;
                    totals[5 * g + 4] += (held * previous) * previous;
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.posterior_ivon((mean, log_sd), [momentum, curvature], (gradient, factor, prior), (groups, variance), (direction, sums), step),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.posterior_ivon((mean, log_sd), [momentum, curvature], (gradient, factor, prior), (groups, variance), (direction, sums), step),
        }
    }

    /// The end of a step ([`Device::posterior_ivon`]): the mean moved along the step's direction,
    /// `μ ← μ − η d` (each entry as [`Device::axpy`] makes it), the average moved toward it,
    /// `μ̄ ← μ̄ + w (μ − μ̄)` (as [`Device::move_toward`]), and each live entry's
    /// `(1, μ̄² + σ², 2s)` added into its group's row of `sums` (groups × 3, as
    /// [`Device::group_moments`] adds them): one pass over the operator on CUDA.
    pub fn posterior_finish(
        &self,
        (mean, direction, eta): (&mut Tensor, &Tensor, f64),
        (average, weight): (&mut Tensor, f64),
        log_sd: &Tensor,
        groups: &GroupMap,
        sums: &mut Tensor,
    ) -> Result<(), GpuError> {
        groups.check(mean, "posterior finish")?;
        same(mean, direction, "posterior finish direction")?;
        same(mean, average, "posterior finish average")?;
        same(mean, log_sd, "posterior finish log standard deviation")?;
        if sums.cols != 3 {
            return Err(shape(format!("{:?} sums for {} entries", sums.dim(), mean.len())));
        }
        match &*self.backend {
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.posterior_finish((mean, direction, eta), (average, weight), log_sd, groups, sums),
            _ => {
                self.axpy(mean, -eta, direction)?;
                self.move_toward(average, weight, mean)?;
                self.group_moments((&*average, log_sd), groups, sums)
            }
        }
    }

    /// Each entry's `(1, μ² + exp(2s), 2s)` added into its group's row of `sums` (groups × 3), as
    /// [`Device::posterior_finish`] adds them after a step; a removed entry (`s = −∞`) adds nothing.
    pub fn group_moments(&self, (mean, log_sd): (&Tensor, &Tensor), groups: &GroupMap, sums: &mut Tensor) -> Result<(), GpuError> {
        groups.check(mean, "group moments")?;
        same(mean, log_sd, "group moments")?;
        if sums.cols != 3 {
            return Err(shape(format!("{:?} sums for {} entries", sums.dim(), mean.len())));
        }
        match &*self.backend {
            Backend::Host => {
                let (means, log_sds, ids) = (host(mean)?, host(log_sd)?, host_indices(&groups.ids)?);
                let totals = host_mut(sums)?;
                for i in 0..means.len() {
                    let g = groups.group(ids, i) as usize;
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

    /// Each entry's `(1, u μ, u² exp(2s))` added into its group's row of `sums` (groups × 3): per
    /// group `u_G · μ_G` and `Σ u_j² σ_j²` of one draw `u` of the Gauss–Newton factor at the
    /// posterior `N(μ, exp(s)²)`, the terms of a removal's curvature; a removed entry (`s = −∞`)
    /// adds nothing. `factor` is in the posterior's storage, or bfloat16 with f32 masters on CUDA.
    pub fn group_curvature(&self, (factor, mean, log_sd): (&Tensor, &Tensor, &Tensor), groups: &GroupMap, sums: &mut Tensor) -> Result<(), GpuError> {
        groups.check(mean, "group curvature")?;
        same(mean, log_sd, "group curvature")?;
        same(mean, factor, "group curvature factor")?;
        if sums.cols != 3 {
            return Err(shape(format!("{:?} sums for {} entries", sums.dim(), mean.len())));
        }
        match &*self.backend {
            Backend::Host => {
                let (us, means, log_sds, ids) = (host(factor)?, host(mean)?, host(log_sd)?, host_indices(&groups.ids)?);
                let totals = host_mut(sums)?;
                for i in 0..means.len() {
                    let g = groups.group(ids, i) as usize;
                    if g * 3 >= totals.len() {
                        return Err(shape(format!("group {g} of {}", totals.len() / 3)));
                    }
                    if log_sds[i] == f64::NEG_INFINITY {
                        continue;
                    }
                    totals[3 * g] += 1.0;
                    totals[3 * g + 1] += us[i] * means[i];
                    totals[3 * g + 2] += us[i] * us[i] * (2.0 * log_sds[i]).exp();
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.group_curvature((factor, mean, log_sd), groups, sums),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.group_curvature((factor, mean, log_sd), groups, sums),
        }
    }

    /// The code length in nats of the groups' posteriors, added into row `slot` of `sums` (rows × 3,
    /// as [`Device::group_moments`] lays its rows out): `(n, Σ_g w_g (d_g + c_g + ln 2 · b_g), 0)`
    /// over the groups `g` with weight `w_g ≠ 0` (`n` their count), `d_g` and `b_g` the group's
    /// divergence and the bits of its variance's scale (the two columns of
    /// [`Device::group_divergence`]'s `divergence`, groups × 2; `b_g` infinite for a scale with no
    /// finite exponent), and `c_g` a constant (`weight` and `constant` groups × 1). The rows of many
    /// steps are read at once.
    pub fn group_code_length(&self, divergence: &Tensor, (weight, constant): (&Tensor, &Tensor), sums: &mut Tensor, slot: usize) -> Result<(), GpuError> {
        let groups = divergence.rows;
        if divergence.cols != 2 {
            return Err(shape(format!("a {:?} divergence for {groups} groups", divergence.dim())));
        }
        for (t, what) in [(weight, "weight"), (constant, "constant")] {
            if t.dim() != (groups, 1) {
                return Err(shape(format!("a {:?} {what} for {groups} groups", t.dim())));
            }
        }
        if sums.cols != 3 || slot >= sums.rows {
            return Err(shape(format!("row {slot} of {:?} sums", sums.dim())));
        }
        match &*self.backend {
            Backend::Host => {
                let (d, w, c) = (host(divergence)?, host(weight)?, host(constant)?);
                let totals = host_mut(sums)?;
                for g in 0..groups {
                    if w[g] == 0.0 {
                        continue;
                    }
                    totals[3 * slot] += 1.0;
                    totals[3 * slot + 1] += w[g] * (d[2 * g] + c[g] + std::f64::consts::LN_2 * d[2 * g + 1]);
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.group_code_length(divergence, (weight, constant), sums, slot),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.group_code_length(divergence, (weight, constant), sums, slot),
        }
    }

    /// From each group's `sums` row `(n, Σ (μ² + σ²), Σ 2s)`, its prior variance `v` into `variance`
    /// (groups × 1), and `KL(q_G ‖ N(0, v I))` in nats and the bits of `v`'s scale against the
    /// group's row of `reference` (groups × 1) into the two columns of `divergence` (groups × 2),
    /// each as [`group_prior`] makes them: `v` minimizes the divergence plus the scale's code, and
    /// without `reference` there is no scale code, `v = Σ (μ² + σ²) / n` and no bits; all zero for
    /// an empty group. Then `sums` is zeroed for the next accumulation.
    pub fn group_divergence(&self, sums: &mut Tensor, reference: Option<&Tensor>, variance: &mut Tensor, divergence: &mut Tensor) -> Result<(), GpuError> {
        let referenced = reference.is_none_or(|r| r.dim() == (sums.rows, 1));
        if sums.cols != 3 || variance.dim() != (sums.rows, 1) || divergence.dim() != (sums.rows, 2) || !referenced {
            return Err(shape(format!(
                "{:?} sums, {:?} variances, {:?} divergences, {:?} references",
                sums.dim(),
                variance.dim(),
                divergence.dim(),
                reference.map(Tensor::dim)
            )));
        }
        match &*self.backend {
            Backend::Host => {
                let references = reference.map(host).transpose()?;
                let (totals, var, kl) = (host_mut(sums)?, host_mut(variance)?, host_mut(divergence)?);
                for g in 0..var.len() {
                    let (v, d, bits) = group_prior(totals[3 * g], totals[3 * g + 1], totals[3 * g + 2], references.map(|r| r[g]));
                    (var[g], kl[2 * g], kl[2 * g + 1]) = (v, d, bits);
                }
                totals.fill(0.0);
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) => engine.group_divergence(sums, reference, variance, divergence),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.group_divergence(sums, reference, variance, divergence),
        }
    }

    /// Rows `ranges` of `t` (f32 on CUDA), stacked in order: an interchange block call's rows of its
    /// stream buffer, on CUDA in one launch per [`ROW_RANGES`] ranges with the ranges passed by value.
    pub fn gather_ranges(&self, t: &Tensor, ranges: &[std::ops::Range<usize>]) -> Result<Tensor, GpuError> {
        if ranges.iter().any(|r| r.end > t.rows) {
            return Err(shape(format!("rows {ranges:?} of {} rows", t.rows)));
        }
        let rows = ranges.iter().map(ExactSizeIterator::len).sum();
        let mut out = self.empty(rows, t.cols)?;
        let mut at = 0;
        let moves: Vec<(usize, usize, usize)> = ranges.iter().map(|r| { at += r.len(); (r.start, at - r.len(), r.len()) }).collect();
        self.copy_ranges(t, &mut out, &moves)?;
        Ok(out)
    }

    /// `values`' rows, in order, written to rows `ranges` of `t` ([`Device::gather_ranges`]'s
    /// inverse).
    pub fn scatter_ranges(&self, t: &mut Tensor, ranges: &[std::ops::Range<usize>], values: &Tensor) -> Result<(), GpuError> {
        let rows: usize = ranges.iter().map(ExactSizeIterator::len).sum();
        if ranges.iter().any(|r| r.end > t.rows) || values.dim() != (rows, t.cols) {
            return Err(shape(format!("{:?} values for rows {ranges:?} of {:?}", values.dim(), t.dim())));
        }
        let mut at = 0;
        let moves: Vec<(usize, usize, usize)> = ranges.iter().map(|r| { at += r.len(); (at - r.len(), r.start, r.len()) }).collect();
        self.copy_ranges(values, t, &moves)
    }

    /// Rows `from..from + length` of `x` copied to rows `to..to + length` of `y`, per move
    /// `(from, to, length)` (equal columns): a gather and a scatter in one pass, no rows made
    /// between them.
    pub fn move_rows(&self, x: &Tensor, y: &mut Tensor, moves: &[(usize, usize, usize)]) -> Result<(), GpuError> {
        if moves.iter().any(|&(from, to, length)| from + length > x.rows || to + length > y.rows) {
            return Err(shape(format!("row moves {moves:?} from {:?} into {:?}", x.dim(), y.dim())));
        }
        self.copy_ranges(x, y, moves)
    }

    /// Rows `from..from + length` of `x` copied to rows `to..` of `y`, per move (equal columns).
    fn copy_ranges(&self, x: &Tensor, y: &mut Tensor, moves: &[(usize, usize, usize)]) -> Result<(), GpuError> {
        if x.cols != y.cols {
            return Err(shape(format!("rows of {:?} into {:?}", x.dim(), y.dim())));
        }
        match &*self.backend {
            Backend::Host => {
                let (cols, xv) = (x.cols, host(x)?.to_vec());
                let yv = host_mut(y)?;
                for &(from, to, length) in moves {
                    yv[to * cols..(to + length) * cols].copy_from_slice(&xv[from * cols..(from + length) * cols]);
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Cuda(engine) if x.storage() == Storage::F32 && y.storage() == Storage::F32 => engine.copy_ranges(x, y, moves),
            // Float64 and bfloat16 rows move one range at a time.
            #[cfg(target_os = "linux")]
            Backend::Cuda(_) => moves.iter().try_for_each(|&(from, to, length)| self.set_rows(y, to, &self.rows_of(x, from, length)?)),
            #[cfg(target_os = "macos")]
            Backend::Metal(_) => moves.iter().try_for_each(|&(from, to, length)| self.set_rows(y, to, &self.rows_of(x, from, length)?)),
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RmsMode {
    Value,
    Backward,
    Tangent,
}

/// The classes a swept head log partition takes at a time for `rows` rows: about 32 MB of f32
/// logits (half the L40's L2, so they are read back from it), a multiple of 128 classes so every
/// chunk's products run the tensor cores' aligned kernels (a leading dimension of 128 columns is a
/// whole number of their tiles), and at least 512.
#[cfg(target_os = "linux")]
fn swept_chunk(rows: usize) -> usize {
    (((1usize << 23) / rows.max(1)) / 128 * 128).max(512)
}

/// One step of [`Device::posterior_ivon`]: the factor turning the given gradient into the data
/// term's gradient per token, the factor turning the Gauss–Newton factor's square into the
/// curvature estimate per token, the tokens `N`, the momentum's and the curvature's decays, and the
/// operator's momentum's bias correction `W` before this step's update: the sum of the weights its
/// gradients carry (`W ← β₁ W + (1 − β₁)` from zero, [`PosteriorStep::weight_after`]), which holds
/// whatever `β₁` each step used and is zero for a momentum that holds no gradient.
#[derive(Clone, Copy, Debug)]
pub struct PosteriorStep {
    pub gradient_scale: f64,
    pub factor_scale: f64,
    pub tokens: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub weight: f64,
}

impl PosteriorStep {
    /// The momentum's bias correction after a step with decay `beta1` from `weight`.
    #[must_use]
    pub fn weight_after(weight: f64, beta1: f64) -> f64 {
        beta1 * weight + (1.0 - beta1)
    }

    /// The momentum's bias correction after `steps` steps of a constant `beta1` from zero
    /// (`1 − β₁ᵗ`).
    #[must_use]
    pub fn constant_weight(beta1: f64, steps: u64) -> f64 {
        (0..steps).fold(0.0, |w, _| Self::weight_after(w, beta1))
    }

    /// The momentum's bias correction after this step's update, `W' = β₁ W + (1 − β₁)`.
    #[must_use]
    pub fn correction(&self) -> f64 {
        Self::weight_after(self.weight, self.beta1)
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

/// The key bit that negates a draw: keys `k` and `k ^ ANTITHETIC` draw `ε` and `−ε`
/// ([`posterior_normal`]), an antithetic pair whose members are each standard normal.
pub const ANTITHETIC: u64 = 1 << 63;

/// The standard normal draw `index` of `(key, stream)`: Box–Muller in f32 on the first two words of
/// `philox` (keyed by `key` without its [`ANTITHETIC`] bit), `√(−2 ln u₁) cos(2π u₂)` with
/// `u₁ = (w₀ + ½) 2⁻³²` and `u₂ = w₁ 2⁻³²`, negated when `key` has the [`ANTITHETIC`] bit. Every
/// backend computes it alike (up to its f32 `log` and `cos`), so a draw is regenerated from its
/// counter.
#[must_use]
pub fn posterior_normal(key: u64, stream: u64, index: u64) -> f32 {
    let w = philox(key & !ANTITHETIC, stream, index);
    let scale = 2.328_306_4e-10_f32;
    let (u1, u2) = ((w[0] as f32 + 0.5) * scale, w[1] as f32 * scale);
    let z = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
    if key & ANTITHETIC == 0 { z } else { -z }
}

/// The sign `ξ` of class `class` in row `row` of the Fisher probe under `key`
/// ([`Device::fisher_probe_cotangent`]): `−1` when the top bit of the first word of `philox` of
/// the counter `(class, row)` is set, else `+1`, so each sign is `±1` with probability ½ and signs
/// of distinct counters are independent as far as Philox's words are. Every backend computes it
/// alike, bit for bit, so a probe's signs are regenerated from their counters, never stored.
#[must_use]
pub fn fisher_sign(key: u64, row: u64, class: u64) -> f64 {
    if philox(key, row, class)[0] >> 31 == 0 { 1.0 } else { -1.0 }
}

/// The bits of the Elias δ codeword of the signed index of the integer `k` (`gam_mpd::codec`'s
/// `zigzag(k) + 1`: `x = 2k + 1` for `k ≥ 0`, `2|k|` for `k < 0`): with `L = ⌊log2 x⌋`,
/// `L + 2⌊log2(L + 1)⌋ + 1`. It depends on `|k|` alone and does not decrease in it; exponent 0's
/// one bit is the shortest.
#[must_use]
pub fn scale_code_bits(k: i64) -> f64 {
    let value = ((k << 1) ^ (k >> 63)) as u64 + 1;
    let low = u64::from(u64::BITS - 1 - value.leading_zeros());
    let prefix = u64::from(u64::BITS - 1 - (low + 1).leading_zeros());
    (low + 2 * prefix + 1) as f64
}

/// A prior group's variance `v`, `KL(q ‖ N(0, v I))` in nats and the bits of `v`'s scale, from the
/// group's sums over its live entries: `count` entries `n`, `second = S = Σ (μ² + σ²)` and
/// `log_variance = Σ 2s = Σ ln σ²` ([`Device::group_divergence`]). Without a `reference` there is no
/// scale code: `v = S / n`, the divergence's minimizer, with no bits. With the reference variance
/// `v⁰`, the scale is the integer exponent `k` of a closed bin `v / v⁰ ∈ [2^(k − ½), 2^(k + ½)]`,
/// sent in [`scale_code_bits`], and `v` minimizes `KL + ln 2 · scale_code_bits(k)` over the bins
/// and the variances in them. With `t = log2(v / v⁰)` and `t̂` its value at `S / n`, the divergence
/// is `½ (n ln(S / n) − Σ 2s) + ½ n (e^(−w) − 1 + w)`, `w = ln 2 · (t − t̂)`: least at `t̂` and
/// rising with `|t − t̂|`. Within bin `k` the code is constant and the divergence least at `t̂`
/// clamped to the bin, so the bin `round(t̂)` holding `t̂` is compared with the bins toward exponent
/// 0 (the code does not lengthen toward it), each at its edge nearest `t̂`, until the divergence's
/// rise at the next edge plus the shortest code is no less than the best charge so far, which no
/// bin from there on can undercut: their rises are larger still. The rise is formed from `w` alone
/// (`exp_m1`), so no difference of large terms decides the bin. A variance or reference that is not
/// positive, or an exponent that is not finite, has an infinite code at `v = S / n`; an empty group
/// is `(0, 0, 0)`.
#[must_use]
pub fn group_prior(count: f64, second: f64, log_variance: f64, reference: Option<f64>) -> (f64, f64, f64) {
    if !(count > 0.0) {
        return (0.0, 0.0, 0.0);
    }
    let centre = second / count;
    let divergence = 0.5 * (count * centre.ln() - log_variance);
    let Some(reference) = reference else { return (centre, divergence, 0.0) };
    // In the log domain: a ratio of finite positive variances can overflow where its logarithm
    // does not.
    let anchor = reference.log2();
    let at = centre.log2() - anchor;
    let first = at.round();
    if !(centre > 0.0 && reference > 0.0) || !first.is_finite() {
        return (centre, divergence, f64::INFINITY);
    }
    let ln2 = std::f64::consts::LN_2;
    // |first| ≤ 2098 for finite positive variances: inside i64.
    let mut k = first as i64;
    let (mut variance, mut rise, mut bits) = (centre, 0.0, scale_code_bits(k));
    let shortest = scale_code_bits(0);
    while k != 0 {
        let toward = k.signum();
        let edge = k as f64 - 0.5 * toward as f64;
        let w = ln2 * (edge - at);
        let up = 0.5 * count * ((-w).exp_m1() + w).max(0.0);
        if up + ln2 * shortest >= rise + ln2 * bits {
            break;
        }
        k -= toward;
        let code = scale_code_bits(k);
        if up + ln2 * code < rise + ln2 * bits {
            (variance, rise, bits) = ((anchor + edge).exp2(), up, code);
        }
    }
    (variance, divergence + rise, bits)
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

/// Block `i` of `v`'s equal blocks of `dims`, as `op` reads it.
fn block_view(v: &[f64], i: usize, dims: (usize, usize), op: Op) -> Result<ArrayView2<'_, f64>, GpuError> {
    let v = ArrayView2::from_shape(dims, &v[i * dims.0 * dims.1..(i + 1) * dims.0 * dims.1]).map_err(|e| shape(e.to_string()))?;
    Ok(if op == Op::T { v.reversed_axes() } else { v })
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
    let lowered = |v: &[f64], unit: Arithmetic| -> Vec<f64> { v.iter().map(|x| round_operand(*x, unit)).collect() };
    // The products summed into c, each a pair of operands as rounded: one, or a split
    // arithmetic's three (the remainders' product first, as CUDA sums them).
    let pairs: Vec<(Vec<f64>, Vec<f64>)> = match arithmetic.split() {
        Some(unit) => {
            let terms = |v: &[f64]| -> (Vec<f64>, Vec<f64>) {
                let whole = lowered(v, Arithmetic::F32);
                let first = lowered(&whole, unit);
                let rest = whole.iter().zip(&first).map(|(x, h)| round_operand(x - h, unit)).collect();
                (first, rest)
            };
            let ((a1, a2), (b1, b2)) = (terms(a), terms(b));
            vec![(a2, b1.clone()), (a1.clone(), b2), (a1, b1)]
        }
        None if arithmetic == Arithmetic::F64 => vec![(a.to_vec(), b.to_vec())],
        None => vec![(lowered(a, arithmetic), lowered(b, arithmetic))],
    };
    let size = cb.0 * cb.1;
    if size == 0 {
        return Ok(());
    }
    // The products of a batch, and row blocks of each product, run on the rayon pool: the gemm
    // library's own threading stops at four threads. Each entry's sum is the same either way.
    c.par_chunks_mut(size).take(batch).enumerate().try_for_each(|(i, c)| -> Result<(), GpuError> {
        let products = pairs.iter().map(|(a, b)| Ok((block_view(a, i, ab, ta)?, block_view(b, i, bb, tb)?))).collect::<Result<Vec<_>, GpuError>>()?;
        let mut cv = ArrayViewMut2::from_shape(cb, c).map_err(|e| shape(e.to_string()))?;
        let rows = cb.0.div_ceil(rayon::current_num_threads()).max(1);
        cv.axis_chunks_iter_mut(Axis(0), rows).into_par_iter().enumerate().for_each(|(chunk, mut cv)| {
            let at = chunk * rows;
            for (t, (av, bv)) in products.iter().enumerate() {
                let av = av.slice_axis(Axis(0), Slice::from(at..at + cv.nrows()));
                general_mat_mul(alpha, &av, bv, if t == 0 { beta } else { 1.0 }, &mut cv);
            }
            if arithmetic != Arithmetic::F64 {
                cv.mapv_inplace(|x| f64::from(x as f32));
            }
        });
        Ok(())
    })
}

#[cfg(target_os = "linux")]
mod cuda {
    use super::{Arithmetic, CheckedInterval, CheckedIntervalReason, CheckedScalar, KlProposalRow, ColumnBlocks, Data, IndexData, Indices, Op, RmsMode, Storage, Tensor, foreign, shape};
    use crate::gpu_error::{GpuError, GpuResultExt};
    use cudarc::cublas::sys::{cublasComputeType_t, cublasGemmAlgo_t, cublasMath_t, cublasOperation_t, cudaDataType_t};
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
    use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
    use cudarc::driver::{CudaContext, CudaEvent, CudaFunction, CudaGraph, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, LaunchArgs, LaunchConfig, PinnedHostSlice, PushKernelArg, SyncOnDrop, ValidAsZeroBits};
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

extern "C" __global__ void axpy_rows(u64 n, double alpha, const double* x, u64 from, double* y, u64 at) {
    GRID_STRIDE(i, n) y[at + i] += alpha * x[from + i];
}

extern "C" __global__ void axpy_within(u64 n, double alpha, u64 from, u64 at, double* t) {
    GRID_STRIDE(i, n) t[at + i] += alpha * t[from + i];
}

extern "C" __global__ void copy_within(u64 n, u64 from, u64 at, double* t) {
    GRID_STRIDE(i, n) t[at + i] = t[from + i];
}

// `Device::scaled`: `zero + α x` (`zero` 0, an argument so that the sum stays: it turns −0 to +0).
extern "C" __global__ void scaled(u64 n, double zero, double alpha, const double* x, double* out) {
    GRID_STRIDE(i, n) out[i] = zero + alpha * x[i];
}

extern "C" __global__ void move_toward(u64 n, double weight, const double* x, double* y) {
    GRID_STRIDE(i, n) {
        double d = x[i] - y[i];
        y[i] += weight * d;
    }
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

// Per column of an n-column x, 1 when one of its rows `rows` (nr of them) holds a value other
// than zero (a NaN counts).
extern "C" __global__ void nonzero_columns(u64 n, unsigned int nr, const double* x, const unsigned int* rows, double* out) {
    GRID_STRIDE(c, n) {
        double any = 0.0;
        for (unsigned int t = 0; t < nr; ++t) {
            if (x[(u64)rows[t] * n + c] != 0.0) { any = 1.0; break; }
        }
        out[c] = any;
    }
}

// Per row r of a rows × groups mask, the columns its groups other than zero span (a NaN counts):
// Σ (starts[g + 1] − starts[g]) over those g. One warp a row, its lanes striding the groups.
extern "C" __global__ void row_counts(unsigned int rows, unsigned int groups, const double* mask, const unsigned int* starts, double* out) {
    u64 lane = threadIdx.x & 31u, warps = ((u64)gridDim.x * blockDim.x) >> 5;
    for (u64 r = ((u64)blockIdx.x * blockDim.x + threadIdx.x) >> 5; r < rows; r += warps) {
        unsigned int n = 0;
        for (u64 g = lane; g < groups; g += 32) {
            if (mask[r * groups + g] != 0.0) n += starts[g + 1] - starts[g];
        }
        for (int o = 16; o > 0; o >>= 1) n += __shfl_down_sync(0xffffffffu, n, o);
        if (lane == 0) out[r] = (double)n;
    }
}

// Row r's listed columns from offsets[r] on, increasing, each with its row. One warp a row, its
// groups 32 at a time: each lane's columns go after the earlier lanes' (a warp scan of widths).
extern "C" __global__ void row_fill(unsigned int rows, unsigned int groups, const double* mask, const unsigned int* starts, const unsigned int* offsets, unsigned int* columns, unsigned int* row_of) {
    u64 lane = threadIdx.x & 31u, warps = ((u64)gridDim.x * blockDim.x) >> 5;
    for (u64 r = ((u64)blockIdx.x * blockDim.x + threadIdx.x) >> 5; r < rows; r += warps) {
        unsigned int e = offsets[r];
        for (u64 base = 0; base < groups; base += 32) {
            u64 g = base + lane;
            unsigned int w = (g < groups && mask[r * groups + g] != 0.0) ? starts[g + 1] - starts[g] : 0u;
            unsigned int scan = w;
            for (int o = 1; o < 32; o <<= 1) {
                unsigned int t = __shfl_up_sync(0xffffffffu, scan, o);
                if (lane >= (u64)o) scan += t;
            }
            unsigned int at = e + scan - w;
            for (unsigned int k = 0; k < w; ++k) { columns[at + k] = starts[g] + k; row_of[at + k] = (unsigned int)r; }
            e += __shfl_sync(0xffffffffu, scan, 31);
        }
    }
}

// out[r, c] = Σ_t x[r, t] a[c, t] at each listed entry (c = columns[e], r = row_of[e]); one warp
// an entry, its lanes striding t and summing by shuffles. Other entries of out are left as they are.
extern "C" __global__ void sampled_product(u64 entries, unsigned int k, unsigned int n, const double* x, const double* a, const unsigned int* columns, const unsigned int* row_of, double* out) {
    u64 lane = threadIdx.x & 31u, warps = ((u64)gridDim.x * blockDim.x) >> 5;
    for (u64 e = ((u64)blockIdx.x * blockDim.x + threadIdx.x) >> 5; e < entries; e += warps) {
        u64 r = row_of[e], c = columns[e];
        double s = 0.0;
        for (u64 t = lane; t < k; t += 32) s += x[r * k + t] * a[c * k + t];
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if (lane == 0) out[r * n + c] = s;
    }
}

// out[r, j] = Σ v[r, c] a[c, j] over row r's listed columns c (v rows × n, a n × m), over the
// total = rows × m entries of out.
extern "C" __global__ void listed_product(u64 total, unsigned int m, unsigned int n, const double* v, const double* a, const unsigned int* offsets, const unsigned int* columns, double* out) {
    GRID_STRIDE(i, total) {
        u64 r = i / m, j = i % m;
        double s = 0.0;
        for (unsigned int e = offsets[r]; e < offsets[r + 1]; ++e) {
            u64 c = columns[e];
            s += v[r * n + c] * a[c * m + j];
        }
        out[i] = s;
    }
}

// out[c, j] = Σ v[r, c] y[r, j] over the rows r listed for column c's group (v rows × n, y rows × m),
// over the total = n × m entries of out.
extern "C" __global__ void listed_product_t(u64 total, unsigned int m, unsigned int n, const double* v, const double* y, const unsigned int* offsets, const unsigned int* rows, const unsigned int* group_of, double* out) {
    GRID_STRIDE(i, total) {
        u64 c = i / m, j = i % m;
        unsigned int g = group_of[c];
        double s = 0.0;
        for (unsigned int e = offsets[g]; e < offsets[g + 1]; ++e) {
            u64 r = rows[e];
            s += v[r * n + c] * y[r * m + j];
        }
        out[i] = s;
    }
}

// A block's input norm per row of y (rows × d, one thread a row, every sum in index order from
// −0, as the host's iterator sums): r = 1 / √(Σ y² / d + ε); mode 0 N(y) = γ y r + β (β zero
// without a bias); mode 1 its tangent along t, γ (r t + y dr) with dr = −r r r (Σ y t) / d; mode 2
// its pullback of t, (r γ) t − a y with a = (Σ γ t y) r r r / d. No product is fused into an add.
extern "C" __global__ void row_norm(unsigned int rows, unsigned int d, unsigned int mode, double epsilon, const double* y, const double* t, const double* g, const double* b, int has_bias, double* out) {
    GRID_STRIDE(r, rows) {
        const double* yr = y + r * d;
        const double* tr = t + r * d;
        double* o = out + r * d;
        double sq = -0.0;
        for (unsigned int k = 0; k < d; ++k) sq = sq + yr[k] * yr[k];
        double scale = 1.0 / sqrt(sq / (double)d + (double)epsilon);
        if (mode == 0) {
            for (unsigned int k = 0; k < d; ++k) o[k] = g[k] * yr[k] * scale + (has_bias ? b[k] : 0.0);
        } else if (mode == 1) {
            double dot = -0.0;
            for (unsigned int k = 0; k < d; ++k) dot = dot + yr[k] * tr[k];
            double dr = -scale * scale * scale * dot / (double)d;
            for (unsigned int k = 0; k < d; ++k) o[k] = g[k] * (scale * tr[k] + yr[k] * dr);
        } else {
            double dot = -0.0;
            for (unsigned int k = 0; k < d; ++k) dot = dot + g[k] * tr[k] * yr[k];
            double along = dot * scale * scale * scale / (double)d;
            for (unsigned int k = 0; k < d; ++k) o[k] = scale * g[k] * tr[k] - along * yr[k];
        }
    }
}

// out = tᵀ over its n = rows × cols entries (t rows × cols).
extern "C" __global__ void transpose(u64 n, unsigned int rows, unsigned int cols, const double* t, double* out) {
    GRID_STRIDE(i, n) out[i] = t[(i % rows) * cols + i / rows];
}

// `out[r, j] = t[r, ids[j]]` over the n = rows × m entries of out.
extern "C" __global__ void gather_columns(u64 n, unsigned int m, unsigned int cols, const double* t, const unsigned int* ids, double* out) {
    GRID_STRIDE(i, n) out[i] = t[(i / m) * cols + ids[i % m]];
}

// `t[r, ids[j]] = values[r, j]` (added when `accumulate`), ids distinct.
extern "C" __global__ void scatter_columns(u64 n, unsigned int m, unsigned int cols, double* t, const unsigned int* ids, const double* values, int accumulate) {
    GRID_STRIDE(i, n) {
        u64 at = (i / m) * cols + ids[i % m];
        t[at] = accumulate ? t[at] + values[i] : values[i];
    }
}

// `t[ids[i], c] = values[i, c]` (added when `accumulate`), ids distinct.
extern "C" __global__ void scatter_rows(u64 n, unsigned int cols, double* t, const unsigned int* ids, const double* values, int accumulate) {
    GRID_STRIDE(i, n) {
        u64 at = (u64)ids[i / cols] * cols + i % cols;
        t[at] = accumulate ? t[at] + values[i] : values[i];
    }
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
        case 6: return log(fmax(t, 1.1754943508222875e-38));
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
        case 6: return t > 1.1754943508222875e-38 ? 1.0 / t : 0.0;
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

// The sign of class `k` in row `r` of the Fisher probe under `key` (`fisher_sign` on the host): the
// top bit of the first word of Philox4x32-10 of the counter (k, r), +1 when it is clear.
__device__ double fisher_sign(u64 key, u64 r, u64 k) {
    unsigned int c0 = (unsigned int)k, c1 = (unsigned int)(k >> 32), c2 = (unsigned int)r, c3 = (unsigned int)(r >> 32);
    unsigned int k0 = (unsigned int)key, k1 = (unsigned int)(key >> 32);
    for (int round = 0; round < 10; ++round) {
        if (round > 0) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
        unsigned int hi0 = __umulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
        unsigned int hi1 = __umulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
        c0 = hi1 ^ c1 ^ k0; c1 = lo1; c2 = hi0 ^ c3 ^ k1; c3 = lo0;
    }
    return (c0 >> 31) ? -1.0 : 1.0;
}

// In place of row r's probabilities π, the Fisher probe's cotangent b = √π ⊙ ξ − π (√π · ξ)
// (`Device::fisher_probe_cotangent`), ξ the signs of row `first + r` under `key`; a row whose
// `scored` flag is zero becomes zero.
extern "C" __global__ void fisher_probe(unsigned int rows, unsigned int cols, double* probabilities, u64 key, u64 first,
                                        const unsigned int* scored, int use_scored) {
    __shared__ double shared[BLOCK];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    double* q = probabilities + (u64)r * cols;
    if (use_scored && scored[r] == 0) {
        for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) q[c] = 0.0;
        return;
    }
    double partial = 0.0;
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) partial += sqrt(q[c]) * fisher_sign(key, first + r, c);
    double dot = block_sum(partial, shared);
    for (unsigned int c = threadIdx.x; c < cols; c += BLOCK) q[c] = sqrt(q[c]) * fisher_sign(key, first + r, c) - q[c] * dot;
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

// A gate's entrywise maps (`GateFunction`, `code`): √x, H(x), Φ(z), φ(z)/s, −φ(z) z/s with z = x/s
// (s read only when `scaled`).
extern "C" __global__ void gate_function(u64 n, unsigned int code, const double* x, const double* s, int scaled, double* out) {
    GRID_STRIDE(i, n) {
        double t = x[i], sd = scaled ? s[i] : 1.0, z = fmin(fmax(t / sd, -40.0), 40.0), density = exp(-0.5 * z * z) * 0.3989422804014327;
        double v;
        switch (code) {
            case 0: v = sqrt(t); break;
            case 1: v = t > 0.0 ? 1.0 : 0.0; break;
            case 2: v = normcdf(z); break;
            case 3: v = density / sd; break;
            case 4: v = -density * z / sd; break;
            case 6: v = exp(2.0 * t); break;
            case 7: v = z > 0.0 ? fmin(z, 1.0) : 0.0; break;
            case 8: v = (z > 0.0 && z < 1.0) ? 1.0 / sd : 0.0; break;
            case 9: v = (z > 0.0 && z < 1.0) ? -z / sd : 0.0; break;
            default: v = sd == 0.0 ? 0.0 : t / sd; break;
        }
        out[i] = v;
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
// (`posterior_normal` on the host), negated for a key with its top (antithetic) bit set.
__device__ float posterior_normal(u64 key, u64 stream, u64 index) {
    unsigned int c0 = (unsigned int)index, c1 = (unsigned int)(index >> 32), c2 = (unsigned int)stream, c3 = (unsigned int)(stream >> 32);
    unsigned int k0 = (unsigned int)key, k1 = (unsigned int)(key >> 32) & 0x7FFFFFFFu;
    float sign = (key >> 63) ? -1.0f : 1.0f;
    for (int round = 0; round < 10; ++round) {
        if (round > 0) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
        unsigned int hi0 = __umulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
        unsigned int hi1 = __umulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
        c0 = hi1 ^ c1 ^ k0; c1 = lo1; c2 = hi0 ^ c3 ^ k1; c3 = lo0;
    }
    float u1 = ((float)c0 + 0.5f) * 2.3283064e-10f, u2 = (float)c1 * 2.3283064e-10f;
    return sign * (sqrtf(-2.0f * logf(u1)) * cosf(6.28318530717958647692f * u2));
}

// The math of an entry type: float entries in float, double in double.
__device__ float entry_exp(float x) { return expf(x); }
__device__ double entry_exp(double x) { return exp(x); }
__device__ float entry_sqrt(float x) { return sqrtf(x); }
__device__ double entry_sqrt(double x) { return sqrt(x); }

// The sample of an operator of `cols` columns into a block whose rows are `stride` apart (`theta`
// at the block's first entry; `stride = cols` for a whole tensor).
template <typename T>
__device__ void reparameterize_body(u64 n, u64 cols, u64 stride, u64 key, u64 stream, const T* mean, const T* log_sd, T* theta) {
    GRID_STRIDE(i, n) theta[(i / cols) * stride + i % cols] = mean[i] + entry_exp(log_sd[i]) * (T)posterior_normal(key, stream, i);
}

// `Device::round_to_deviation`, in float64 whatever the storage: the step is an exact power of
// two (ldexp), and the rounded value is held exactly in the storage of the mean it rounds.
template <typename T>
__device__ void round_to_deviation_body(u64 n, const T* mean, const T* log_sd, T* out) {
    GRID_STRIDE(i, n) {
        double mu = (double)mean[i], s = (double)log_sd[i];
        if (isfinite(s)) {
            double step = ldexp(1.0, (int)floor(s / 0.6931471805599453));
            mu = round(mu / step) * step;
        }
        out[i] = (T)mu;
    }
}

extern "C" __global__ void round_to_deviation_f64(u64 n, const double* mean, const double* log_sd, double* out) {
    round_to_deviation_body<double>(n, mean, log_sd, out);
}

extern "C" __global__ void round_to_deviation_f32(u64 n, const float* mean, const float* log_sd, float* out) {
    round_to_deviation_body<float>(n, mean, log_sd, out);
}

extern "C" __global__ void reparameterize_f64(u64 n, u64 cols, u64 stride, u64 key, u64 stream, const double* mean, const double* log_sd, double* theta) {
    reparameterize_body<double>(n, cols, stride, key, stream, mean, log_sd, theta);
}

extern "C" __global__ void reparameterize_f32(u64 n, u64 cols, u64 stride, u64 key, u64 stream, const float* mean, const float* log_sd, float* theta) {
    reparameterize_body<float>(n, cols, stride, key, stream, mean, log_sd, theta);
}

// The bfloat16 nearest x (ties to even; a NaN stays a quiet NaN), as its 16 bits (`bf16_bits`).
__device__ unsigned short bf16_round(float x) {
    unsigned int bits = __float_as_uint(x);
    if (x != x) return (unsigned short)((bits >> 16) | 0x40u);
    return (unsigned short)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// The sample of f32 posterior entries written as bfloat16, the form products in Arithmetic::Bf16
// read without rounding it again (`reparameterize_body`'s block).
extern "C" __global__ void reparameterize_bf16(u64 n, u64 cols, u64 stride, u64 key, u64 stream, const float* mean, const float* log_sd, unsigned short* theta) {
    GRID_STRIDE(i, n) theta[(i / cols) * stride + i % cols] = bf16_round(mean[i] + expf(log_sd[i]) * posterior_normal(key, stream, i));
}

// `Device::run_samples`: several samples of one key in one launch. `table` holds per sample eight
// words: its output (at its block's first entry), mean and log standard deviation addresses, its
// columns, its output's row stride, its stream, and its first entry in the launch's numbering (the
// samples' entries one after another, `total` in all). Launch entry g is entry g − first of the
// last sample whose first entry is at most g, written as its own launch writes it.
__device__ u64 sample_at(u64 samples, const u64* table, u64 g) {
    u64 lo = 0, hi = samples;
    while (hi - lo > 1) {
        u64 mid = (lo + hi) / 2;
        if (table[8 * mid + 6] <= g) lo = mid; else hi = mid;
    }
    return lo;
}

// Runs `body(job, i)` on every entry of the launch, `job` its sample's eight words and `i` its entry
// in the sample: a block's consecutive entries per pass, the sample of its first found once
// (`sample_at`, thread 0) and each thread's from there forward, since a block's entries span one
// sample or a few (a search per entry made the launch three times its memory time).
template <typename F>
__device__ void each_sample_entry(u64 samples, const u64* table, u64 total, F body) {
    __shared__ u64 first;
    for (u64 base = (u64)blockIdx.x * blockDim.x; base < total; base += (u64)gridDim.x * blockDim.x) {
        if (threadIdx.x == 0) first = sample_at(samples, table, base);
        __syncthreads();
        u64 g = base + threadIdx.x;
        if (g < total) {
            u64 j = first;
            while (j + 1 < samples && table[8 * (j + 1) + 6] <= g) j++;
            body(table + 8 * j, g - table[8 * j + 6]);
        }
        __syncthreads();
    }
}

template <typename T>
__device__ void reparameterize_many_body(u64 samples, const u64* table, u64 total, u64 key) {
    each_sample_entry(samples, table, total, [&](const u64* job, u64 i) {
        u64 cols = job[3];
        const T* mean = (const T*)job[1];
        const T* log_sd = (const T*)job[2];
        ((T*)job[0])[(i / cols) * job[4] + i % cols] = mean[i] + entry_exp(log_sd[i]) * (T)posterior_normal(key, job[5], i);
    });
}

extern "C" __global__ void reparameterize_many_f64(u64 samples, const u64* table, u64 total, u64 key) {
    reparameterize_many_body<double>(samples, table, total, key);
}

extern "C" __global__ void reparameterize_many_f32(u64 samples, const u64* table, u64 total, u64 key) {
    reparameterize_many_body<float>(samples, table, total, key);
}

extern "C" __global__ void reparameterize_many_bf16(u64 samples, const u64* table, u64 total, u64 key) {
    each_sample_entry(samples, table, total, [&](const u64* job, u64 i) {
        u64 cols = job[3];
        const float* mean = (const float*)job[1];
        const float* log_sd = (const float*)job[2];
        ((unsigned short*)job[0])[(i / cols) * job[4] + i % cols] = bf16_round(mean[i] + expf(log_sd[i]) * posterior_normal(key, job[5], i));
    });
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

// A posterior entry's per-entry transcendentals in its masters' type: the prior precision per token
// `δ = 1 / (N v_G)`, IVON's log deviation `s = −½ ln(N (h + δ))` and the variance `exp(2s)` a
// group sum adds (in double either way). For f32 masters they run in float, within a few ulps as
// on the Apple GPU (`exp`, `log` and the division under the safe math modes): GeForce and L40
// cards run double at 1/64 of the float rate, and per entry these were most of the posterior
// kernels' time. For float64 masters they stay double.
__device__ __forceinline__ float prior_precision(float, double tokens, double variance) { return 1.0f / ((float)tokens * (float)variance); }
__device__ __forceinline__ double prior_precision(double, double tokens, double variance) { return 1.0 / (tokens * variance); }
__device__ __forceinline__ float log_deviation(float precision, double tokens) { return -0.5f * logf((float)tokens * precision); }
__device__ __forceinline__ double log_deviation(double precision, double tokens) { return -0.5 * log(tokens * precision); }
__device__ __forceinline__ double variance_of(float s) { return (double)expf(2.0f * s); }
__device__ __forceinline__ double variance_of(double s) { return exp(2.0 * s); }

// Entry i's group under an operator's group map (`GroupMap`): its row's id (axis 0), its column's
// (axis 1) or its own (axis 2).
__device__ unsigned int group_of(const unsigned int* ids, unsigned int axis, u64 cols, u64 i) {
    return axis == 0u ? ids[i / cols] : (axis == 1u ? ids[i % cols] : ids[i]);
}

// A map whose segments are one column each (axis code 3, `GroupMap::reduce_code`), COLUMNS segments
// per block side by side: thread (r, l) = (threadIdx.x / COLUMNS, threadIdx.x % COLUMNS), r below
// R = BLOCK / COLUMNS, takes for m = 0 .. COLUMNS − 1 in turn rows r + R m + BLOCK j (j = 0, 1, ...,
// in order) of segment l's column: the entries thread r + R m of the segment's own block takes in
// `segments_reduce`, summed in the same order. The R sums of one m are R / 32 of that block's warps;
// each goes through the shuffles its warp takes (real warp w those of column w / (R / 32)), and lane
// 0 of warp l adds the warps' sums in warp order: every group's sums are the ones its own block adds,
// while a warp reads and writes runs of COLUMNS adjacent columns instead of one entry of each of 32
// rows. Eight columns a block read whole 32-byte sectors: on vpd4l's 3072 columns an RTX 4090 ran
// posterior_ivon at 270 µs a launch with eight against 364 with four (f97524b178's). `body(i, g, v)` puts entry i's N terms in v and says whether to add them;
// with `every` it also runs on the entries of a group at or beyond `count` (whose sums are not
// added); with `counted` a group's sums are added only when the first is positive. Every block runs
// the same rows and synchronizations.
// Its shared arrays come from `pool` (N · BLOCK + N · 8 · BLOCK / 32 doubles), which
// `segments_reduce` declares once for every path.
#define COLUMNS 8
template <int N, typename F>
__device__ void columns_reduce(u64 n, u64 cols, u64 segments, const unsigned int* layout, u64 count, bool every, bool counted, double* sums, double* pool, F body) {
    const unsigned int R = BLOCK / COLUMNS, VW = R / 32;
    double (*lanes)[COLUMNS][BLOCK / COLUMNS] = reinterpret_cast<double (*)[COLUMNS][BLOCK / COLUMNS]>(pool);
    double (*warps)[COLUMNS][BLOCK / 32] = reinterpret_cast<double (*)[COLUMNS][BLOCK / 32]>(pool + N * BLOCK);
    const u64 rows = cols > 0 ? n / cols : 0;
    const unsigned int l = threadIdx.x % COLUMNS, r = threadIdx.x / COLUMNS;
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    for (u64 first = (u64)blockIdx.x * COLUMNS; first < segments; first += (u64)gridDim.x * COLUMNS) {
        u64 s = first + l;
        unsigned int g = s < segments ? layout[segments + 1 + s] : 0u;
        bool runs = s < segments && (every || g < count);
        u64 column = runs ? (u64)layout[2 * segments + 1 + layout[s]] : 0;
        for (unsigned int m = 0; m < COLUMNS; m++) {
            double v[N];
            for (int t = 0; t < N; t++) v[t] = 0.0;
            if (runs)
                for (u64 row = r + R * m; row < rows; row += BLOCK) {
                    double e[N];
                    for (int t = 0; t < N; t++) e[t] = 0.0;
                    if (body(row * cols + column, g, e))
                        for (int t = 0; t < N; t++) v[t] += e[t];
                }
            for (int t = 0; t < N; t++) lanes[t][l][r] = v[t];
            __syncthreads();
            {
                // Warp w reduces column w / VW's virtual warp VW m + w % VW.
                const unsigned int c = warp / VW, h = warp % VW;
                for (int t = 0; t < N; t++) v[t] = lanes[t][c][32 * h + lane];
                for (int o = 16; o > 0; o >>= 1)
                    for (int t = 0; t < N; t++) v[t] += __shfl_down_sync(0xffffffffu, v[t], o);
                if (lane == 0)
                    for (int t = 0; t < N; t++) warps[t][c][VW * m + h] = v[t];
            }
            __syncthreads();
        }
        u64 own = first + warp;
        if (lane == 0 && warp < COLUMNS && own < segments) {
            unsigned int h = layout[segments + 1 + own];
            if (h < count) {
                double total[N];
                for (int t = 0; t < N; t++) {
                    total[t] = 0.0;
                    for (unsigned int w = 0; w < BLOCK / 32; w++) total[t] += warps[t][warp][w];
                }
                if (!counted || total[0] > 0.0)
                    for (int t = 0; t < N; t++) sums[N * (u64)h + t] += total[t];
            }
        }
        __syncthreads();
    }
}

// Runs `body(i, g, v)` on the entries of an `n`-entry operator of `cols` columns by the map's
// segments (`GroupMap`'s `layout`: `segments + 1` offsets, the segments' groups, then their
// members), `g` the entry's group: every entry with `every`, else those of the groups below `count`.
// `body` puts entry i's N terms in v and says whether to add them; a group below `count` gets its
// entries' terms added into its row of `sums` (N a row), with `counted` only when the first is
// positive. One block per segment (`GroupMap::blocks`), each thread summing every BLOCK-th of its
// entries in order, each warp's sums added by shuffles in a fixed order, the warps' added by thread
// 0 in warp order, and thread 0 adding the totals to the group's row; a map of one-column segments
// goes eight columns a block (`columns_reduce`), each group's sums the same. A group is one segment
// of the map and launches on a stream run in order, so nothing is atomic and the sums are the same
// on every run. (One warp per segment left a segment's entries to 32 threads: the posterior kernels
// ran at a fifth of their memory rate.)
// Segments of one entry each (axis code 4), one thread each: the entry's terms v (zero when `body`
// does not add them) are what a block of `segments_reduce` adds for it, 0 + v through its shuffles
// and warps (which turns a −0 into +0, as `+ 0.0` does here), so the sums are the same.
template <int N, typename F>
__device__ void entries_reduce(u64 segments, const unsigned int* layout, u64 count, bool every, bool counted, double* sums, F body) {
    for (u64 s = (u64)blockIdx.x * blockDim.x + threadIdx.x; s < segments; s += (u64)gridDim.x * blockDim.x) {
        unsigned int g = layout[segments + 1 + s];
        if (!every && g >= count) continue;
        u64 i = layout[2 * segments + 1 + layout[s]];
        double e[N], v[N];
        for (int t = 0; t < N; t++) { e[t] = 0.0; v[t] = 0.0; }
        if (body(i, g, e))
            for (int t = 0; t < N; t++) v[t] += e[t];
        if (g >= count) continue;
        double total[N];
        for (int t = 0; t < N; t++) total[t] = 0.0 + (v[t] + 0.0);
        if (!counted || total[0] > 0.0)
            for (int t = 0; t < N; t++) sums[N * (u64)g + t] += total[t];
    }
}

// A map of one-column segments read TILE = 32 columns a block (axis code 5): thread (r, l) =
// (threadIdx.x / TILE, threadIdx.x % TILE), r below R = BLOCK / TILE = 8, takes column l's virtual
// threads v = 32 w + r + R p (p = 0 .. 3) of each virtual warp w in turn, each summing rows
// v + BLOCK j (j = 0, 1, ..., in order) as thread v of the segment's own block in `segments_reduce`
// does, so a warp reads 128 adjacent bytes of one row. A virtual warp's shuffle tree (lane i adds
// lane i + o, o = 16, 8, 4, 2, 1) is taken in the same order: o = 16 and 8 pair lanes of one
// thread (i ≡ r mod 8), x_p + x_{p+2} and then the two; o = 4, 2, 1 pair the R threads' values
// through `pool`, and thread r = 0 adds the virtual warps' sums in order: every group's sums are
// the ones its own block adds.
#define TILE 32
template <int N, typename F>
__device__ void tiles_reduce(u64 n, u64 cols, u64 segments, const unsigned int* layout, u64 count, bool every, bool counted, double* sums, double* pool, F body) {
    const unsigned int R = BLOCK / TILE;
    double (*corner)[TILE][BLOCK / TILE] = reinterpret_cast<double (*)[TILE][BLOCK / TILE]>(pool);
    const u64 rows = cols > 0 ? n / cols : 0;
    const unsigned int l = threadIdx.x % TILE, r = threadIdx.x / TILE;
    for (u64 first = (u64)blockIdx.x * TILE; first < segments; first += (u64)gridDim.x * TILE) {
        u64 s = first + l;
        unsigned int g = s < segments ? layout[segments + 1 + s] : 0u;
        bool runs = s < segments && (every || g < count);
        u64 column = runs ? (u64)layout[2 * segments + 1 + layout[s]] : 0;
        double total[N];
        for (int t = 0; t < N; t++) total[t] = 0.0;
        for (unsigned int w = 0; w < BLOCK / 32; w++) {
            double a[2][N];
            for (unsigned int p = 0; p < 4; p++) {
                double v[N];
                for (int t = 0; t < N; t++) v[t] = 0.0;
                if (runs)
                    for (u64 row = 32 * w + r + R * p; row < rows; row += BLOCK) {
                        double e[N];
                        for (int t = 0; t < N; t++) e[t] = 0.0;
                        if (body(row * cols + column, g, e))
                            for (int t = 0; t < N; t++) v[t] += e[t];
                    }
                if (p < 2)
                    for (int t = 0; t < N; t++) a[p][t] = v[t];
                else
                    for (int t = 0; t < N; t++) a[p - 2][t] += v[t];
            }
            for (int t = 0; t < N; t++) corner[t][l][r] = a[0][t] + a[1][t];
            __syncthreads();
            if (r == 0)
                for (int t = 0; t < N; t++) {
                    double c[4];
                    for (int q = 0; q < 4; q++) c[q] = corner[t][l][q] + corner[t][l][q + 4];
                    total[t] += (c[0] + c[2]) + (c[1] + c[3]);
                }
            __syncthreads();
        }
        if (r == 0 && s < segments && g < count && (!counted || total[0] > 0.0))
            for (int t = 0; t < N; t++) sums[N * (u64)g + t] += total[t];
    }
}

template <int N, typename F>
__device__ void segments_reduce(u64 n, u64 cols, unsigned int axis, u64 segments, const unsigned int* layout, u64 count, bool every, bool counted, double* sums, F body) {
    __shared__ double pool[N * BLOCK + N * 8 * (BLOCK / 32)];
    if (axis == 3u) {
        columns_reduce<N>(n, cols, segments, layout, count, every, counted, sums, pool, body);
        return;
    }
    if (axis == 5u) {
        tiles_reduce<N>(n, cols, segments, layout, count, every, counted, sums, pool, body);
        return;
    }
    if (axis == 4u) {
        entries_reduce<N>(segments, layout, count, every, counted, sums, body);
        return;
    }
    double (*partial)[BLOCK / 32] = reinterpret_cast<double (*)[BLOCK / 32]>(pool);
    const u64 rows = cols > 0 ? n / cols : 0;
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    for (u64 s = blockIdx.x; s < segments; s += gridDim.x) {
        unsigned int g = layout[segments + 1 + s];
        if (!every && g >= count) continue;
        u64 off = layout[s], size = (u64)layout[s + 1] - off;
        u64 length = axis == 0u ? size * cols : (axis == 1u ? size * rows : size);
        const unsigned int* members = layout + 2 * segments + 1 + off;
        double v[N];
        for (int t = 0; t < N; t++) v[t] = 0.0;
        for (u64 k = threadIdx.x; k < length; k += blockDim.x) {
            u64 i = axis == 0u ? (u64)members[k / cols] * cols + k % cols : (axis == 1u ? (k / size) * cols + members[k % size] : (u64)members[k]);
            double e[N];
            for (int t = 0; t < N; t++) e[t] = 0.0;
            if (body(i, g, e))
                for (int t = 0; t < N; t++) v[t] += e[t];
        }
        if (g >= count) continue;
        for (int o = 16; o > 0; o >>= 1)
            for (int t = 0; t < N; t++) v[t] += __shfl_down_sync(0xffffffffu, v[t], o);
        if (lane == 0)
            for (int t = 0; t < N; t++) partial[t][warp] = v[t];
        __syncthreads();
        if (threadIdx.x == 0) {
            double total[N];
            for (int t = 0; t < N; t++) {
                total[t] = 0.0;
                for (unsigned int w = 0; w < blockDim.x / 32u; w++) total[t] += partial[t][w];
            }
            if (!counted || total[0] > 0.0)
                for (int t = 0; t < N; t++) sums[N * (u64)g + t] += total[t];
        }
        __syncthreads();
    }
}

// Runs `body(i, g, a, b, c)` on every entry i whose group g is below `count`, and adds each entry it
// returns true for, (a, b, c), into its group's row of `sums` when the group's first sum is
// positive (`segments_reduce`).
template <typename F>
__device__ void group_reduce(u64 n, u64 cols, unsigned int axis, u64 segments, const unsigned int* layout, u64 count, double* sums, F body) {
    segments_reduce<3>(n, cols, axis, segments, layout, count, false, true, sums, [&](u64 i, unsigned int g, double* e) { return body(i, g, e[0], e[1], e[2]); });
}

// Entries (the posterior, the momentum, the curvature, the gradient, the Gauss–Newton factor, the
// prior curvature) in T; group sums in double. `inputs` holds 1 with a gradient, 2 with a factor and
// 4 with a prior curvature; an absent one (a null pointer, never read) is zero. `c0` and `c1` are the
// momentum's bias corrections before and after the step (`PosteriorStep`). The mean stays; its full
// step d = G / (h₀⁺ + δ), h₀ the curvature before this step's draw, goes to `direction`, and each
// live entry's (d d, u d, (h₀⁺ d) d, (g + δ μ) d₀, ((h₀⁺ + δ) d₀) d₀) into its group's row of `sums`
// (groups × 5), d₀ the direction before the step (`Device::posterior_ivon`); the deviation takes the
// stepped curvature h⁺. An entry the step leaves (removed, or of a group at or beyond `count`) has
// d = μ − μ.
template <typename T>
__device__ void posterior_ivon_body(u64 n, u64 cols, unsigned int axis, u64 chunks, u64 count, double scale, double fscale, double tokens, double beta1, double beta2, double c0, double c1,
    unsigned int inputs, const T* gradient, const T* factor, const T* prior, const unsigned int* groups, const double* variance, const T* mean, T* log_sd, T* momentum, T* curvature,
    T* direction, double* sums) {
    const T b1 = (T)beta1, o1 = (T)(1.0 - beta1), o2 = (T)(1.0 - beta2), k0 = (T)(c0 > 0.0 ? 1.0 / c0 : 0.0), k1 = (T)(1.0 / c1), weight = (T)scale, square = (T)fscale, zero = (T)0;
    const bool given = (inputs & 1u) != 0u, drawn = (inputs & 2u) != 0u, priced = (inputs & 4u) != 0u;
    segments_reduce<5>(n, cols, axis, chunks, groups, count, true, false, sums, [&](u64 i, unsigned int g, double* e) -> bool {
        T mu = mean[i];
        if (g >= count || log_sd[i] == (T)NEG_INF) {
            direction[i] = mu - mu;
            return false;
        }
        T delta = prior_precision((T)0, tokens, variance[g]);
        // The direction before the step, from the momentum and the curvature it holds.
        T m = momentum[i], h = curvature[i];
        T bounded = h > zero ? h : zero, held = bounded + delta;
        T previous = (m * k0 + delta * mu) / held;
        T gi = given ? weight * gradient[i] : zero, ui = drawn ? factor[i] : zero;
        T estimate = square * ui * ui + (priced ? prior[i] : zero);
        T m1 = b1 * m + o1 * gi;
        T h1 = h + o2 * (estimate - h);
        T positive = h1 > zero ? h1 : zero;
        T move = (m1 * k1 + delta * mu) / held;
        T s = log_deviation(positive + delta, tokens);
        momentum[i] = m1; curvature[i] = h1; log_sd[i] = s;
        direction[i] = move;
        double dd = (double)move, dp = (double)previous;
        e[0] = dd * dd; e[1] = (double)ui * dd; e[2] = (double)(bounded * move) * dd;
        e[3] = (double)(gi + delta * mu) * dp; e[4] = (double)(held * previous) * dp;
        return true;
    });
}

extern "C" __global__ void posterior_ivon_f64(u64 n, u64 cols, unsigned int axis, u64 chunks, u64 count, double scale, double fscale, double tokens, double beta1, double beta2, double c0, double c1,
    unsigned int inputs, const double* gradient, const double* factor, const double* prior, const unsigned int* groups, const double* variance, const double* mean, double* log_sd,
    double* momentum, double* curvature, double* direction, double* sums) {
    posterior_ivon_body<double>(n, cols, axis, chunks, count, scale, fscale, tokens, beta1, beta2, c0, c1, inputs, gradient, factor, prior, groups, variance, mean, log_sd, momentum, curvature, direction, sums);
}

extern "C" __global__ void posterior_ivon_f32(u64 n, u64 cols, unsigned int axis, u64 chunks, u64 count, double scale, double fscale, double tokens, double beta1, double beta2, double c0, double c1,
    unsigned int inputs, const float* gradient, const float* factor, const float* prior, const unsigned int* groups, const double* variance, const float* mean, float* log_sd,
    float* momentum, float* curvature, float* direction, double* sums) {
    posterior_ivon_body<float>(n, cols, axis, chunks, count, scale, fscale, tokens, beta1, beta2, c0, c1, inputs, gradient, factor, prior, groups, variance, mean, log_sd, momentum, curvature, direction, sums);
}

// `Device::posterior_finish`: μ ← μ + (−η) d (as `axpy` makes it), μ̄ ← μ̄ + w (μ − μ̄) (as
// `move_toward` makes it) on every entry, and each live entry's (1, μ̄² + σ², 2s) into its group's
// row of `sums` (as `group_moments` adds them).
template <typename T>
__device__ void posterior_finish_body(u64 n, u64 cols, unsigned int axis, u64 chunks, u64 count, double eta, double weight,
    const T* direction, const T* log_sd, const unsigned int* groups, T* mean, T* average, double* sums) {
    const T a = (T)(-eta), w = (T)weight;
    segments_reduce<3>(n, cols, axis, chunks, groups, count, true, true, sums, [&](u64 i, unsigned int g, double* e) -> bool {
        T mu = mean[i] + a * direction[i];
        mean[i] = mu;
        T gap = mu - average[i];
        T moved = average[i] + w * gap;
        average[i] = moved;
        if (g >= count || log_sd[i] == (T)NEG_INF) return false;
        double m = (double)moved;
        e[0] = 1.0; e[1] = m * m + variance_of(log_sd[i]); e[2] = 2.0 * (double)log_sd[i];
        return true;
    });
}

extern "C" __global__ void posterior_finish_f64(u64 n, u64 cols, unsigned int axis, u64 chunks, u64 count, double eta, double weight,
    const double* direction, const double* log_sd, const unsigned int* groups, double* mean, double* average, double* sums) {
    posterior_finish_body<double>(n, cols, axis, chunks, count, eta, weight, direction, log_sd, groups, mean, average, sums);
}

extern "C" __global__ void posterior_finish_f32(u64 n, u64 cols, unsigned int axis, u64 chunks, u64 count, double eta, double weight,
    const float* direction, const float* log_sd, const unsigned int* groups, float* mean, float* average, double* sums) {
    posterior_finish_body<float>(n, cols, axis, chunks, count, eta, weight, direction, log_sd, groups, mean, average, sums);
}

// A bfloat16's value (`bf16_round` is its inverse to nearest).
__device__ float bf16_value(unsigned short h) { return __uint_as_float(((unsigned int)h) << 16); }
__device__ float entry_load(float x) { return x; }
__device__ double entry_load(double x) { return x; }
__device__ float entry_load(unsigned short h) { return bf16_value(h); }

extern "C" __global__ void widen_bf16(u64 n, const unsigned short* x, float* y) {
    GRID_STRIDE(i, n) y[i] = bf16_value(x[i]);
}

template <typename T>
__device__ void group_moments_body(u64 n, u64 cols, unsigned int axis, u64 chunks, u64 count, const T* mean, const T* log_sd, const unsigned int* groups, double* sums) {
    group_reduce(n, cols, axis, chunks, groups, count, sums, [&](u64 i, unsigned int, double& a, double& b, double& c) -> bool {
        if (log_sd[i] == (T)NEG_INF) return false;
        double mu = (double)mean[i];
        a = 1.0; b = mu * mu + variance_of(log_sd[i]); c = 2.0 * (double)log_sd[i];
        return true;
    });
}

extern "C" __global__ void group_moments_f64(u64 n, u64 cols, unsigned int axis, u64 chunks, u64 count, const double* mean, const double* log_sd, const unsigned int* groups, double* sums) {
    group_moments_body<double>(n, cols, axis, chunks, count, mean, log_sd, groups, sums);
}

extern "C" __global__ void group_moments_f32(u64 n, u64 cols, unsigned int axis, u64 chunks, u64 count, const float* mean, const float* log_sd, const unsigned int* groups, double* sums) {
    group_moments_body<float>(n, cols, axis, chunks, count, mean, log_sd, groups, sums);
}

// Each live entry's (1, u μ, u² exp(2s)) into its group's row (`Device::group_curvature`), the
// factor `u` in U (the masters' storage, or bfloat16 with f32 masters).
template <typename T, typename U>
__device__ void group_curvature_body(u64 n, u64 cols, unsigned int axis, u64 chunks, u64 count, const U* factor, const T* mean, const T* log_sd, const unsigned int* groups, double* sums) {
    group_reduce(n, cols, axis, chunks, groups, count, sums, [&](u64 i, unsigned int, double& a, double& b, double& c) -> bool {
        if (log_sd[i] == (T)NEG_INF) return false;
        double u = (double)entry_load(factor[i]);
        a = 1.0; b = u * (double)mean[i]; c = u * u * variance_of(log_sd[i]);
        return true;
    });
}

extern "C" __global__ void group_curvature_f64(u64 n, u64 cols, unsigned int axis, u64 chunks, u64 count, const double* factor, const double* mean, const double* log_sd, const unsigned int* groups, double* sums) {
    group_curvature_body<double, double>(n, cols, axis, chunks, count, factor, mean, log_sd, groups, sums);
}

extern "C" __global__ void group_curvature_f32(u64 n, u64 cols, unsigned int axis, u64 chunks, u64 count, const float* factor, const float* mean, const float* log_sd, const unsigned int* groups, double* sums) {
    group_curvature_body<float, float>(n, cols, axis, chunks, count, factor, mean, log_sd, groups, sums);
}

extern "C" __global__ void group_curvature_f32_bf16(u64 n, u64 cols, unsigned int axis, u64 chunks, u64 count, const unsigned short* factor, const float* mean, const float* log_sd, const unsigned int* groups, double* sums) {
    group_curvature_body<float, unsigned short>(n, cols, axis, chunks, count, factor, mean, log_sd, groups, sums);
}

// `Device::group_code_length`, first stage: block b sums groups b·BLOCK .. (b + 1)·BLOCK, one per
// thread, by shuffles within each warp in a fixed order and then the warps in order, into
// `partials[2b]` (live groups) and `partials[2b + 1]` (their code length). The second stage
// (`group_code_length_total`) adds the blocks' in order, so the sum is the same on every run. Each
// group's divergence and scale bits are `group_divergence`'s two columns.
extern "C" __global__ void group_code_length(u64 n, u64 slot, const double* divergence, const double* weight, const double* constant, double* partials) {
    __shared__ double warps_a[BLOCK / 32], warps_t[BLOCK / 32];
    double a = 0.0, total = 0.0;
    for (u64 g = (u64)blockIdx.x * BLOCK + threadIdx.x; g < n && g < ((u64)blockIdx.x + 1) * BLOCK; g += BLOCK) {
        bool live = weight[g] != 0.0;
        double b = 0.0;
        if (live) b = weight[g] * (divergence[2 * g] + constant[g] + 0.6931471805599453 * divergence[2 * g + 1]);
        a += live ? 1.0 : 0.0;
        total += b;
    }
    for (int o = 16; o > 0; o >>= 1) {
        a += __shfl_down_sync(0xffffffffu, a, o);
        total += __shfl_down_sync(0xffffffffu, total, o);
    }
    if ((threadIdx.x & 31u) == 0) { warps_a[threadIdx.x >> 5] = a; warps_t[threadIdx.x >> 5] = total; }
    __syncthreads();
    if (threadIdx.x == 0) {
        double ba = 0.0, bt = 0.0;
        for (unsigned int w = 0; w < BLOCK / 32; w++) { ba += warps_a[w]; bt += warps_t[w]; }
        partials[2 * (u64)blockIdx.x] = ba;
        partials[2 * (u64)blockIdx.x + 1] = bt;
    }
}

// `Device::group_code_length`, second stage: the first stage's `blocks` partials added in block
// order into row `slot` of `sums`, by thread 0 alone.
extern "C" __global__ void group_code_length_total(u64 blocks, u64 slot, const double* partials, double* sums) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double a = 0.0, total = 0.0;
    for (u64 b = 0; b < blocks; b++) { a += partials[2 * b]; total += partials[2 * b + 1]; }
    if (a > 0.0) {
        sums[3 * slot] += a;
        sums[3 * slot + 1] += total;
    }
}

// `Device::resolved_counts`: every entry into row 0, those of dead columns not live.
template <typename T>
__device__ void resolved_counts_body(u64 n, u64 cols, int gated, const T* value, const T* slope, const T* phi, const T* noise_z, const T* noise_y,
    const unsigned int* alive, double* sums) {
    WARP_STRIDE(i, n) {
        bool live = i < n && alive[i % cols] != 0u;
        double a = 0.0, b = 0.0, c = 0.0;
        if (live) {
            double v = (double)value[i], s = (double)slope[i];
            double noise = s * s * fmax((double)noise_z[i], 0.0);
            if (gated) {
                double p = (double)phi[i];
                noise += p * p * fmax((double)noise_y[i], 0.0);
            }
            a = 1.0; b = v != 0.0 ? 1.0 : 0.0; c = fabs(v) > sqrt(noise) ? 1.0 : 0.0;
        }
        group_add(sums, 0u, live, a, b, c);
    }
}

extern "C" __global__ void resolved_counts_f64(u64 n, u64 cols, int gated, const double* value, const double* slope, const double* phi, const double* noise_z,
    const double* noise_y, const unsigned int* alive, double* sums) {
    resolved_counts_body<double>(n, cols, gated, value, slope, phi, noise_z, noise_y, alive, sums);
}

extern "C" __global__ void resolved_counts_f32(u64 n, u64 cols, int gated, const float* value, const float* slope, const float* phi, const float* noise_z,
    const float* noise_y, const unsigned int* alive, double* sums) {
    resolved_counts_body<float>(n, cols, gated, value, slope, phi, noise_z, noise_y, alive, sums);
}

// The bits of the Elias δ codeword of the signed index of k (`gam_gpu::tensor::scale_code_bits`).
__device__ double scale_code_bits(long long k) {
    unsigned long long value = (((unsigned long long)k) << 1 ^ (unsigned long long)(k >> 63)) + 1ULL;
    unsigned long long low = 63ULL - (unsigned long long)__clzll((long long)value);
    unsigned long long prefix = 63ULL - (unsigned long long)__clzll((long long)(low + 1ULL));
    return (double)(low + 2ULL * prefix + 1ULL);
}

// `Device::group_divergence`: each group's prior variance into `variance`, its divergence and its
// scale's bits into `divergence[2g]` and `divergence[2g + 1]`, as `gam_gpu::tensor::group_prior`
// makes them (`scaled` nonzero when `reference` holds the groups' reference variances), then its
// sums zeroed.
extern "C" __global__ void group_divergence(u64 n, int scaled, const double* reference, double* sums, double* variance, double* divergence) {
    GRID_STRIDE(g, n) {
        double count = sums[3 * g], second = sums[3 * g + 1], log_variance = sums[3 * g + 2];
        double v = 0.0, kl = 0.0, bits = 0.0;
        if (count > 0.0) {
            double centre = second / count;
            v = centre;
            kl = 0.5 * (count * log(centre) - log_variance);
            if (scaled) {
                double anchor = log2(reference[g]);
                double at = log2(centre) - anchor;
                double first = round(at);
                if (!(centre > 0.0 && reference[g] > 0.0) || !isfinite(first)) {
                    bits = __longlong_as_double(0x7ff0000000000000LL);
                } else {
                    long long k = (long long)first;
                    double rise = 0.0, shortest = scale_code_bits(0);
                    bits = scale_code_bits(k);
                    while (k != 0) {
                        long long toward = k > 0 ? 1 : -1;
                        double edge = (double)k - 0.5 * (double)toward;
                        double w = 0.6931471805599453 * (edge - at);
                        double up = 0.5 * count * fmax(expm1(-w) + w, 0.0);
                        if (up + 0.6931471805599453 * shortest >= rise + 0.6931471805599453 * bits) break;
                        k -= toward;
                        double code = scale_code_bits(k);
                        if (up + 0.6931471805599453 * code < rise + 0.6931471805599453 * bits) {
                            v = exp2(anchor + edge);
                            rise = up;
                            bits = code;
                        }
                    }
                    kl += rise;
                }
            }
        }
        variance[g] = v;
        divergence[2 * g] = kl;
        divergence[2 * g + 1] = bits;
        sums[3 * g] = 0.0; sums[3 * g + 1] = 0.0; sums[3 * g + 2] = 0.0;
    }
}
"#;

    /// The f32 twins of [`KERNELS`] (module note), one name and parameter list per kernel.
    const KERNELS_F32: &str = include_str!("tensor_f32.cu");

    /// The split products' operand terms and the row moves (`split_moves.cu`), compiled on first use.
    const KERNELS_SPLIT_MOVES: &str = include_str!("split_moves.cu");

    /// `split_moves.cu`'s `RowRanges`: the moves of one `copy_ranges` launch.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RowRanges {
        count: u32,
        from: [u32; super::ROW_RANGES],
        to: [u32; super::ROW_RANGES],
        length: [u32; super::ROW_RANGES],
    }

    // SAFETY: plain u32s laid out as the kernel's parameter struct.
    unsafe impl DeviceRepr for RowRanges {}

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
        /// `split_moves.cu`'s kernels (the split products' operand terms, the row moves), compiled on first
        /// use, and the functions loaded from them.
        module_split_moves: std::sync::OnceLock<Arc<CudaModule>>,
        functions_split_moves: std::sync::Mutex<HashMap<&'static str, CudaFunction>>,
        checked_interval_module: crate::device_cache::PtxModuleCache,
        /// The row flags of a call that scores every row (never read).
        every_row: CudaSlice<u32>,
        gemm_workspace: std::sync::Mutex<F32Workspace>,
        /// [`Engine::gram_split`]'s exponents, slices and int32 product, kept from call to call: a
        /// fresh stream-ordered allocation of them maps new pages (up to 8.6 ms each on a 4090).
        split_workspace: std::sync::Mutex<SplitWorkspace>,
        /// Whether the stream is being captured into a graph.
        capturing: AtomicBool,
        /// cuBLAS's workspace inside captures (a recorded product may not allocate), kept for the
        /// engine's life since every graph's products read it.
        capture_workspace: std::sync::Mutex<Option<CudaSlice<u8>>>,
        /// The pinned buffers uploads pass through ([`Engine::upload`]).
        staging: std::sync::Mutex<Staging>,
        /// The copy stream's staging buffers ([`Engine::upload_overlapped`]).
        copy_staging: std::sync::Mutex<Staging>,
        /// cublasLt's handle, workspace and plans ([`Engine::gemm_onto`]).
        lt: std::sync::Mutex<Lt>,
        /// The stream overlapped uploads copy on, and the device buffers they land in first
        /// ([`Engine::upload_overlapped`]).
        copies: Arc<CudaStream>,
        landing: std::sync::Mutex<Landing>,
    }

    /// The cuBLAS workspace captured products use.
    const CAPTURE_WORKSPACE: usize = 32 << 20;

    /// Pinned host buffers an upload's bytes pass through ([`Engine::upload`]), in turn: a copy from
    /// pageable memory synchronizes the stream before it starts, so every upload waited for all the
    /// queued kernels and the device idled until the host queued more; a copy from pinned memory is
    /// queued as a kernel is. Uploads fill a buffer one after another (each from a
    /// [`STAGE_ALIGN`]-byte boundary) until the next would not fit, then the next buffer in turn; a
    /// buffer's one event is recorded again after each copy from it and waited for only when the
    /// buffer comes round again (a training step made about 105 uploads, each of which waited for an
    /// event and made and destroyed one). A buffer's address is read once, when it is made (cudarc's
    /// accessors wait for an event of their own at every call). One ring per stream (the engine's
    /// `staging`, the copy stream's `copy_staging`), so a buffer's event follows every copy from it.
    #[derive(Default)]
    struct Staging {
        buffers: Vec<(PinnedHostSlice<u8>, usize, Option<CudaEvent>)>,
        next: usize,
        used: usize,
    }

    /// cublasLt's handle, its workspace (grown to what a chosen algorithm asks, at most
    /// `CAPTURE_WORKSPACE`, the engine's bound on cuBLAS's) and a product's plan per shape, types and
    /// compute ([`Engine::gemm_onto`]).
    struct Lt {
        handle: cudarc::cublaslt::sys::cublasLtHandle_t,
        workspace: Option<CudaSlice<u8>>,
        plans: HashMap<LtKey, LtPlan>,
    }

    // SAFETY: the handle and the plans' descriptors are used only under the engine's lock on them.
    unsafe impl Send for Lt {}

    impl Lt {
        fn new() -> Result<Self, GpuError> {
            Ok(Self { handle: cudarc::cublaslt::result::create_handle().gpu_ctx("tensor cublasLt handle")?, workspace: None, plans: HashMap::new() })
        }
    }

    impl Drop for Lt {
        fn drop(&mut self) {
            self.plans.clear();
            // SAFETY: the handle was made by `Lt::new` and is destroyed once, after its plans.
            if let Err(e) = unsafe { cudarc::cublaslt::result::destroy_handle(self.handle) } {
                log::error!("tensor cublasLt handle: {e}");
            }
        }
    }

    /// A product's shape as cublasLt takes it (column-major): op(A) and op(B) transposed, m, n, k,
    /// the leading dimensions of A, B and C, whether A and B are bfloat16, and the compute type.
    type LtKey = (bool, bool, usize, usize, usize, usize, usize, usize, bool, bool, u32);

    /// A cublasLt product's descriptor, its layouts of A, B and C (also D's) and the algorithm
    /// its heuristic chose, made once per [`LtKey`].
    struct LtPlan {
        desc: cudarc::cublaslt::sys::cublasLtMatmulDesc_t,
        layouts: [cudarc::cublaslt::sys::cublasLtMatrixLayout_t; 3],
        heuristic: cudarc::cublaslt::sys::cublasLtMatmulHeuristicResult_t,
    }

    impl LtPlan {
        fn new(handle: cudarc::cublaslt::sys::cublasLtHandle_t, key: LtKey) -> Result<Self, GpuError> {
            use cudarc::cublaslt::{result as lt, sys};
            let (transa, transb, m, n, k, lda, ldb, ldc, half_a, half_b, compute) = key;
            let compute = if compute == cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32 as u32 { sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32 } else { sys::cublasComputeType_t::CUBLAS_COMPUTE_32F };
            let real = sys::cudaDataType::CUDA_R_32F;
            let kind = |half: bool| if half { sys::cudaDataType::CUDA_R_16BF } else { real };
            let wide = |x: usize| u64::try_from(x).map_err(|_| shape(format!("a cublasLt dimension {x}")));
            let lead = |x: usize| i64::try_from(x).map_err(|_| shape(format!("a cublasLt leading dimension {x}")));
            // SAFETY: every handle below is made here and destroyed once by `LtPlan`'s drop, which
            // owns it from the moment it exists (an error drops the plan built so far).
            unsafe {
                let mut plan = Self { desc: lt::create_matmul_desc(compute, real).gpu_ctx("tensor cublasLt descriptor")?, layouts: [std::ptr::null_mut(); 3], heuristic: std::mem::zeroed() };
                for (attribute, transposed) in [(sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, transa), (sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, transb)] {
                    let flag = i32::from(transposed);
                    lt::set_matmul_desc_attribute(plan.desc, attribute, (&flag) as *const i32 as *const _, std::mem::size_of::<i32>()).gpu_ctx("tensor cublasLt transpose")?;
                }
                // Stored shapes: A is m × k (k × m transposed), B k × n (n × k), C and D m × n.
                let (a_rows, a_cols) = if transa { (k, m) } else { (m, k) };
                let (b_rows, b_cols) = if transb { (n, k) } else { (k, n) };
                plan.layouts[0] = lt::create_matrix_layout(kind(half_a), wide(a_rows)?, wide(a_cols)?, lead(lda)?).gpu_ctx("tensor cublasLt layout")?;
                plan.layouts[1] = lt::create_matrix_layout(kind(half_b), wide(b_rows)?, wide(b_cols)?, lead(ldb)?).gpu_ctx("tensor cublasLt layout")?;
                plan.layouts[2] = lt::create_matrix_layout(real, wide(m)?, wide(n)?, lead(ldc)?).gpu_ctx("tensor cublasLt layout")?;
                let preference = lt::create_matmul_pref().gpu_ctx("tensor cublasLt preference")?;
                let bound = CAPTURE_WORKSPACE as u64;
                let chosen = lt::set_matmul_pref_attribute(preference, sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, (&bound) as *const u64 as *const _, std::mem::size_of::<u64>())
                    .and_then(|()| lt::get_matmul_algo_heuristic(handle, plan.desc, plan.layouts[0], plan.layouts[1], plan.layouts[2], plan.layouts[2], preference));
                let destroyed = lt::destroy_matmul_pref(preference);
                plan.heuristic = chosen.gpu_ctx("tensor cublasLt heuristic")?;
                destroyed.gpu_ctx("tensor cublasLt preference")?;
                Ok(plan)
            }
        }
    }

    impl Drop for LtPlan {
        fn drop(&mut self) {
            use cudarc::cublaslt::result as lt;
            // SAFETY: each handle was made by `LtPlan::new` (null where it was not reached).
            unsafe {
                for layout in self.layouts.into_iter().filter(|l| !l.is_null()) {
                    if let Err(e) = lt::destroy_matrix_layout(layout) {
                        log::error!("tensor cublasLt layout: {e}");
                    }
                }
                if let Err(e) = lt::destroy_matmul_desc(self.desc) {
                    log::error!("tensor cublasLt descriptor: {e}");
                }
            }
        }
    }

    /// [`Engine::upload_overlapped`]'s device buffers, in turn: each lands one upload on the copy
    /// stream and keeps the event recorded after the stream's copy out of it, which the copy stream
    /// waits for before landing the next one there.
    #[derive(Default)]
    struct Landing {
        buffers: Vec<(CudaSlice<u8>, Option<CudaEvent>)>,
        next: usize,
    }

    /// [`Landing`]'s buffers.
    const LANDINGS: usize = 4;

    /// The device buffers of tensors dropped on an engine's stream, kept for that stream's next
    /// allocations of the same size: a stream-ordered allocation and its free cost about 3.4 µs of
    /// host time together (4090 trace), a training step made about 3,200 of each, and most are of
    /// sizes the step frees and makes again (each block's, each antithetic half's). Keyed by the
    /// stream's handle; only an engine's own stream keeps buffers (registered at its first
    /// allocation). A buffer is taken again only by an allocation queued on the same stream after
    /// every use of it, the order the driver's own reuse keeps. Kept buffers go back to the driver at
    /// [`Engine::release_recycled`] (a caller's batch boundary, an allocation the driver refuses, the
    /// engine's end); none is kept or taken while a capture records, a buffer recorded in a graph
    /// being the graph's.
    #[derive(Default)]
    struct Recycled {
        buffers: HashMap<usize, Vec<cudarc::driver::sys::CUdeviceptr>>,
        paused: bool,
    }

    /// Every stream's [`Recycled`] buffers.
    fn recycled() -> &'static std::sync::Mutex<HashMap<usize, Recycled>> {
        static RECYCLED: std::sync::OnceLock<std::sync::Mutex<HashMap<usize, Recycled>>> = std::sync::OnceLock::new();
        RECYCLED.get_or_init(Default::default)
    }

    fn stream_key(stream: &CudaStream) -> usize {
        stream.cu_stream() as usize
    }

    /// `values`' buffer kept for its stream's next allocation of its size when the stream keeps
    /// buffers ([`Recycled`]); freed otherwise.
    pub(super) fn recycle<T>(values: CudaSlice<T>) {
        let bytes = values.len() * std::mem::size_of::<T>();
        let Ok(mut kept) = recycled().lock() else { return };
        match kept.get_mut(&stream_key(values.stream())) {
            Some(stream) if !stream.paused && bytes > 0 => stream.buffers.entry(bytes).or_default().push(values.leak()),
            _ => {
                drop(kept);
                drop(values);
            }
        }
    }

    /// [`Staging`]'s buffers and their bytes: an upload longer than one buffer passes through several.
    const STAGES: usize = 16;
    const STAGE_BYTES: usize = 4 << 20;
    /// Where an upload starts within a staging buffer: a multiple of this many bytes, the alignment
    /// cudaMalloc gives, so a copy from the buffer is as aligned as one from its start.
    const STAGE_ALIGN: usize = 256;

    impl Drop for Engine {
        /// The stream's kept buffers go back to the driver, and the stream keeps none after.
        fn drop(&mut self) {
            self.release_recycled();
            if let Ok(mut kept) = recycled().lock() {
                kept.remove(&stream_key(&self.stream));
            }
        }
    }

    impl Drop for Staging {
        /// A buffer is freed only after its last copy finished (a failed wait is logged: a drop
        /// returns nothing).
        fn drop(&mut self) {
            for event in self.buffers.iter().filter_map(|(_, _, e)| e.as_ref()) {
                if let Err(e) = event.synchronize() {
                    log::error!("tensor upload staging: waiting for a copy before freeing its buffer: {e:?}");
                }
            }
        }
    }

    #[derive(Default)]
    struct SplitWorkspace {
        exponents: Option<CudaSlice<i32>>,
        parts: Option<CudaSlice<i8>>,
        product: Option<CudaSlice<i32>>,
    }

    /// `held` grown to at least `n` values (its old values dropped, never read: every user writes
    /// what it reads first).
    fn at_least<T: DeviceRepr>(stream: &Arc<CudaStream>, held: &mut Option<CudaSlice<T>>, n: usize) -> Result<(), GpuError> {
        if held.as_ref().is_none_or(|h| h.len() < n) {
            *held = None;
            // SAFETY: the buffer's values are written before they are read (`Engine::gram_split`).
            *held = Some(unsafe { stream.alloc::<T>(n.max(1)) }.gpu_ctx("split workspace")?);
        }
        Ok(())
    }

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

    fn u32_of(n: usize) -> Result<u32, GpuError> {
        u32::try_from(n).map_err(|_| shape(format!("{n} exceeds a split or move kernel's 32-bit indices")))
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
            Arithmetic::Tf32x3 | Arithmetic::Bf16x3 => Err(shape(format!("{arithmetic:?} runs as three products of its terms"))),
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
            // The device's stream-ordered pool keeps what is freed for the next allocation instead of
            // returning it at each synchronization (its default threshold, zero, made every
            // allocation after a synchronization map memory again: 14 µs a call against about 1).
            // The memory stays the process's, as it would between allocations anyway.
            // SAFETY: the device's default pool, set once; the attribute takes a u64 in place.
            unsafe {
                let mut pool = std::ptr::null_mut();
                cudarc::driver::sys::cuDeviceGetDefaultMemPool(&mut pool, ctx.cu_device()).result().gpu_ctx("tensor memory pool")?;
                let mut threshold = u64::MAX;
                cudarc::driver::sys::cuMemPoolSetAttribute(pool, cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, (&mut threshold) as *mut u64 as *mut _)
                    .result()
                    .gpu_ctx("tensor memory pool threshold")?;
            }
            let blas = CudaBlas::new(stream.clone()).gpu_ctx("tensor cuBLAS handle")?;
            static MODULE: crate::device_cache::PtxModuleCache = crate::device_cache::PtxModuleCache::new();
            let module = Arc::clone(MODULE.get_or_compile(&ctx, "tensor", KERNELS)?);
            let every_row = stream.alloc_zeros::<u32>(1).gpu_ctx("tensor alloc")?;
            let copies = ctx.new_stream().gpu_ctx("tensor copy stream")?;
            Ok(Self {
                name,
                ctx,
                stream,
                blas,
                module,
                module32: std::sync::OnceLock::new(),
                functions32: std::sync::Mutex::new(HashMap::new()),
                module_split_moves: std::sync::OnceLock::new(),
                functions_split_moves: std::sync::Mutex::new(HashMap::new()),
                checked_interval_module: crate::device_cache::PtxModuleCache::new(),
                every_row,
                gemm_workspace: std::sync::Mutex::new(F32Workspace::default()),
                split_workspace: std::sync::Mutex::new(SplitWorkspace::default()),
                capturing: AtomicBool::new(false),
                capture_workspace: std::sync::Mutex::new(None),
                staging: std::sync::Mutex::new(Staging::default()),
                copy_staging: std::sync::Mutex::new(Staging::default()),
                copies,
                landing: std::sync::Mutex::new(Landing::default()),
                lt: std::sync::Mutex::new(Lt::new()?),
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
                *workspace = Some(self.alloc_zeros::<u8>(CAPTURE_WORKSPACE).gpu_ctx("tensor capture workspace")?);
            }
            let buffer = workspace.as_ref().ok_or_else(|| shape("missing capture workspace".to_string()))?;
            let (pointer, record) = buffer.device_ptr(&self.stream);
            drop(record);
            // SAFETY: the buffer lives as long as the engine and its 256-byte-aligned allocation.
            unsafe { cudarc::cublas::sys::cublasSetWorkspace_v2(*self.blas.handle(), pointer as *mut _, CAPTURE_WORKSPACE) }
                .result()
                .gpu_ctx("tensor capture cuBLAS workspace")?;
            self.capturing.store(true, Ordering::Release);
            self.pause_recycling(true);
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
            self.pause_recycling(false);
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
                return self.alloc_zeros::<T>(1).gpu_ctx("tensor alloc");
            }
            // SAFETY: the staged copies below write all of its values before any kernel reads them.
            let mut out = unsafe { self.alloc::<T>(values.len()) }.gpu_ctx("tensor alloc")?;
            // SAFETY: `values` is `size_of_val(values)` initialized bytes of plain data (`DeviceRepr`).
            let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) };
            let (dst, record) = out.device_ptr_mut(&self.stream);
            self.staged(&self.stream, dst, bytes)?;
            drop(record);
            Ok(out)
        }

        /// [`Engine::upload`] copied on the engine's second stream (`copies`) into a landing buffer
        /// ([`Landing`]), then on the stream into the new slice: the host-to-device copy runs beside
        /// the kernels queued before it, and the stream waits for it only where it takes the slice.
        pub(super) fn upload_overlapped<T: DeviceRepr + ValidAsZeroBits>(&self, values: &[T]) -> Result<CudaSlice<T>, GpuError> {
            self.host_transfer()?;
            if values.is_empty() {
                return self.alloc_zeros::<T>(1).gpu_ctx("tensor alloc");
            }
            // SAFETY: `values` is `size_of_val(values)` initialized bytes of plain data (`DeviceRepr`).
            let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) };
            let mut landing = self.landing.lock().map_err(|_| shape("poisoned upload landing".to_string()))?;
            let turn = landing.next;
            landing.next = (turn + 1) % LANDINGS;
            if landing.buffers.get(turn).is_none_or(|(buffer, _)| buffer.len() < bytes.len()) {
                // A landing buffer of at least these bytes, made on the stream and valid on every
                // stream once the stream has reached it (rare: the first upload of each size up).
                let buffer = self.alloc_zeros::<u8>(bytes.len()).gpu_ctx("tensor landing alloc")?;
                self.stream.synchronize().gpu_ctx("tensor landing alloc")?;
                if landing.buffers.len() == turn {
                    landing.buffers.push((buffer, None));
                } else {
                    landing.buffers[turn] = (buffer, None);
                }
            }
            let (buffer, last) = &mut landing.buffers[turn];
            if let Some(event) = last.take() {
                self.copies.wait(&event).gpu_ctx("tensor landing wait")?;
            }
            {
                let (landed, record) = buffer.device_ptr(&self.copies);
                self.staged(&self.copies, landed, bytes)?;
                drop(record);
            }
            let arrived = self.copies.record_event(None).gpu_ctx("tensor landing event")?;
            self.stream.wait(&arrived).gpu_ctx("tensor landing wait")?;
            // SAFETY: the copy below writes all of its values before any kernel reads them.
            let mut out = unsafe { self.alloc::<T>(values.len()) }.gpu_ctx("tensor alloc")?;
            {
                let (dst, record) = out.device_ptr_mut(&self.stream);
                let (src, src_record) = buffer.device_ptr(&self.stream);
                // SAFETY: both hold at least `bytes.len()` bytes; the stream waited for the landing.
                unsafe { cudarc::driver::result::memcpy_dtod_async(dst, src, bytes.len(), self.stream.cu_stream()) }.gpu_ctx("tensor landing copy")?;
                drop((record, src_record));
            }
            *last = Some(self.stream.record_event(None).gpu_ctx("tensor landing event")?);
            Ok(out)
        }

        /// `bytes` into device memory at `dst` on `stream`, through `stream`'s pinned staging
        /// buffers ([`Staging`]): a buffer is written again only after its last copy completed.
        fn staged(&self, stream: &CudaStream, dst: u64, bytes: &[u8]) -> Result<(), GpuError> {
            let ring = if std::ptr::eq(stream, &*self.copies) { &self.copy_staging } else { &self.staging };
            let mut staging = ring.lock().map_err(|_| shape("poisoned upload staging".to_string()))?;
            for (index, piece) in bytes.chunks(STAGE_BYTES).enumerate() {
                let room = piece.len().next_multiple_of(STAGE_ALIGN);
                if staging.buffers.is_empty() || staging.used + room > STAGE_BYTES {
                    if !staging.buffers.is_empty() {
                        staging.next = (staging.next + 1) % STAGES;
                    }
                    staging.used = 0;
                    let turn = staging.next;
                    if staging.buffers.len() == turn {
                        // SAFETY: a buffer's bytes are written before a copy reads them.
                        let mut buffer = unsafe { self.ctx.alloc_pinned::<u8>(STAGE_BYTES) }.gpu_ctx("tensor pinned alloc")?;
                        let base = buffer.as_mut_ptr().gpu_ctx("tensor upload staging")? as usize;
                        staging.buffers.push((buffer, base, None));
                    } else if let Some(event) = staging.buffers[turn].2.as_ref() {
                        event.synchronize().gpu_ctx("tensor upload staging")?;
                    }
                }
                let (turn, at) = (staging.next, staging.used);
                staging.used += room;
                let (_, base, last) = &mut staging.buffers[turn];
                // SAFETY: bytes `at..at + piece.len()` of the buffer at `base` (STAGE_BYTES long, alive
                // while `staging` holds it), which no copy queued since the buffer's last wait reads.
                let host = unsafe { std::slice::from_raw_parts_mut((*base + at) as *mut u8, piece.len()) };
                host.copy_from_slice(piece);
                // SAFETY: `dst` holds `bytes.len()` bytes, this piece's at `index * STAGE_BYTES`; the
                // buffer's bytes are not written again before `last`, recorded after this copy, completes.
                unsafe { cudarc::driver::result::memcpy_htod_async(dst + (index * STAGE_BYTES) as u64, host, stream.cu_stream()) }.gpu_ctx("tensor upload")?;
                match last {
                    Some(event) => event.record(stream).gpu_ctx("tensor upload event")?,
                    None => *last = Some(stream.record_event(None).gpu_ctx("tensor upload event")?),
                }
            }
            Ok(())
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

        /// `len` values of `T` on the stream: a kept buffer of their size ([`Recycled`]), else the
        /// driver's, which is asked again with every kept buffer returned when it refuses.
        ///
        /// # Safety
        /// The values are unset, as `CudaStream::alloc`'s.
        // SAFETY: the caller writes every value before any is read, as with `CudaStream::alloc`.
        unsafe fn alloc<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>, cudarc::driver::DriverError> {
            let bytes = len * std::mem::size_of::<T>();
            if let Some(buffer) = self.take_recycled(bytes) {
                // SAFETY: a buffer of `bytes` bytes allocated on this stream, every use of it queued
                // on the stream before this allocation's.
                return Ok(unsafe { self.stream.upgrade_device_ptr::<T>(buffer, len) });
            }
            // SAFETY: as the caller's.
            let first = unsafe { self.stream.alloc::<T>(len) };
            if first.is_err() && self.release_recycled() > 0 {
                // SAFETY: as the caller's.
                return unsafe { self.stream.alloc::<T>(len) };
            }
            first
        }

        /// [`Engine::alloc`]'s buffer set to zero.
        fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<CudaSlice<T>, cudarc::driver::DriverError> {
            // SAFETY: zeroed before any kernel reads it.
            let mut values = unsafe { self.alloc::<T>(len) }?;
            self.stream.memset_zeros(&mut values)?;
            Ok(values)
        }

        /// A kept buffer of `bytes` bytes on this stream, registering the stream at its first call.
        fn take_recycled(&self, bytes: usize) -> Option<cudarc::driver::sys::CUdeviceptr> {
            let mut kept = recycled().lock().ok()?;
            let stream = kept.entry(stream_key(&self.stream)).or_default();
            if stream.paused {
                return None;
            }
            stream.buffers.get_mut(&bytes)?.pop()
        }

        /// Every buffer this stream keeps returned to the driver; the count returned.
        pub(super) fn release_recycled(&self) -> usize {
            let buffers = match recycled().lock() {
                Ok(mut kept) => kept.get_mut(&stream_key(&self.stream)).map(|stream| std::mem::take(&mut stream.buffers)).unwrap_or_default(),
                Err(_) => return 0,
            };
            let mut count = 0;
            for (bytes, pointers) in buffers {
                for pointer in pointers {
                    // SAFETY: a buffer of `bytes` bytes allocated on this stream that no tensor
                    // holds; dropping it frees it in stream order.
                    drop(unsafe { self.stream.upgrade_device_ptr::<u8>(pointer, bytes) });
                    count += 1;
                }
            }
            count
        }

        /// Stops (or resumes) keeping and taking buffers on this stream, around a capture.
        fn pause_recycling(&self, paused: bool) {
            if let Ok(mut kept) = recycled().lock() {
                kept.entry(stream_key(&self.stream)).or_default().paused = paused;
            }
        }

        pub(super) fn zeros(&self, n: usize) -> Result<CudaSlice<f64>, GpuError> {
            self.alloc_zeros::<f64>(n.max(1)).gpu_ctx("tensor alloc")
        }

        pub(super) fn zeros32(&self, n: usize) -> Result<CudaSlice<f32>, GpuError> {
            self.alloc_zeros::<f32>(n.max(1)).gpu_ctx("tensor alloc")
        }

        pub(super) fn zeros16(&self, n: usize) -> Result<CudaSlice<u16>, GpuError> {
            self.alloc_zeros::<u16>(n.max(1)).gpu_ctx("tensor alloc")
        }

        /// A `rows × cols` tensor in `storage` for a kernel that writes every entry: f32 left
        /// unset (no zeroing pass ahead of the kernel), float64 zeroed as it always was.
        pub(super) fn output(&self, storage: Storage, rows: usize, cols: usize) -> Result<Tensor, GpuError> {
            if storage != Storage::F32 {
                return self.tensor(storage, rows, cols);
            }
            // SAFETY: the caller's kernel writes all `rows · cols` values before any is read.
            let data = Data::Cuda32(unsafe { self.alloc::<f32>((rows * cols).max(1)) }.gpu_ctx("tensor alloc")?);
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
            // SAFETY: the copy writes every value before any is read (an empty slice's one value is
            // never read).
            let mut out = unsafe { self.alloc::<T>(slice.len().max(1)) }.gpu_ctx("tensor alloc")?;
            self.stream.memcpy_dtod(slice, &mut out).gpu_ctx("tensor copy")?;
            Ok(out)
        }

        pub(super) fn copy_range<T: DeviceRepr + ValidAsZeroBits>(&self, slice: &CudaSlice<T>, lo: usize, hi: usize) -> Result<CudaSlice<T>, GpuError> {
            // SAFETY: the copy writes every value before any is read (an empty range's one value is
            // never read).
            let mut out = unsafe { self.alloc::<T>((hi - lo).max(1)) }.gpu_ctx("tensor alloc")?;
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
                    *slot = Some(self.alloc_zeros::<f32>(t.len().max(1)).gpu_ctx("tensor f32 alloc")?);
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
            if let Some(unit) = arithmetic.split() {
                return self.gemm_split(batch, (m, n, k), (alpha, beta), (a, ta), (b, tb), c, unit);
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

        /// `out ← α op(a) op(b) + addend` (f32 storage, one block): cublasLt's D = α A B + C, C a
        /// matrix of its own, one product where `addend`'s copy into `out` and `gemm32` with β = 1
        /// took two passes. The operands as `gemm32` reads them (an f32 one rounded to bfloat16 in
        /// Arithmetic::Bf16), its compute type and its column-major view (Dᵀ = op(B)ᵀ op(A)ᵀ + Cᵀ).
        pub(super) fn gemm_onto(&self, (m, n, k): (usize, usize, usize), alpha: f64, (a, ta): (&Tensor, Op), (b, tb): (&Tensor, Op), (addend, out): (&Tensor, &mut Tensor), arithmetic: Arithmetic) -> Result<(), GpuError> {
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
            let (rounded_a, rounded_b) = if half { (self.half_of(a)?, self.half_of(b)?) } else { (None, None) };
            let (fb, fa) = (pick(b, &rounded_b)?, pick(a, &rounded_a)?);
            let bf16 = |f: &Factor<'_>| matches!(f, Factor::Half(_));
            let key: LtKey = (tb == Op::T, ta == Op::T, n, m, k, b.cols, a.cols, addend.cols, bf16(&fb), bf16(&fa), compute as u32);
            let mut lt = self.lt.lock().map_err(|_| shape("poisoned cublasLt plans".to_string()))?;
            if !lt.plans.contains_key(&key) {
                let plan = LtPlan::new(lt.handle, key)?;
                lt.plans.insert(key, plan);
            }
            let needed = lt.plans.get(&key).map_or(0, |p| p.heuristic.workspaceSize);
            if lt.workspace.as_ref().is_none_or(|w| w.len() < needed.max(1)) {
                lt.workspace = Some(self.alloc_zeros::<u8>(needed.max(1)).gpu_ctx("tensor cublasLt workspace")?);
            }
            let (alpha, one) = (alpha as f32, 1.0_f32);
            let plan = lt.plans.get(&key).ok_or_else(|| shape("a cublasLt plan".to_string()))?;
            let workspace = lt.workspace.as_ref().ok_or_else(|| shape("a cublasLt workspace".to_string()))?;
            let (pa, _, record_a) = pointer(fb, 0, &self.stream);
            let (pb, _, record_b) = pointer(fa, 0, &self.stream);
            let (pc, record_c) = slice32(addend)?.device_ptr(&self.stream);
            let (pw, record_w) = workspace.device_ptr(&self.stream);
            let (pd, record_d) = slice32_mut(out)?.device_ptr_mut(&self.stream);
            // SAFETY: the plan's layouts are these operands' shapes and leading dimensions (checked
            // by the caller against the buffers); the workspace holds what the algorithm asks; the
            // pointers outlive the call (their records drop after it).
            let product = unsafe {
                cudarc::cublaslt::result::matmul(
                    lt.handle, plan.desc, (&alpha) as *const f32 as *const _, (&one) as *const f32 as *const _,
                    pa as *const _, plan.layouts[0], pb as *const _, plan.layouts[1], pc as *const _, plan.layouts[2], pd as *mut _, plan.layouts[2],
                    &plan.heuristic.algo, pw as *mut _, needed, self.stream.cu_stream() as _,
                )
            };
            drop((record_a, record_b, record_c, record_w, record_d));
            product.gpu_ctx("tensor cublasLt product")
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
            // SAFETY: `to_bf16` writes all `count` values before any is read (none when it is 0).
            let mut out = unsafe { self.alloc::<u16>(count.max(1)) }.gpu_ctx("tensor bf16 alloc")?;
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

        /// A split arithmetic's product (`Arithmetic::split` gives `unit`, its parts' arithmetic):
        /// each f32 operand as its two terms, the three products summed into `c`, the remainders'
        /// first. A bfloat16 operand is exact in either unit and has no remainder.
        fn gemm_split(
            &self,
            batch: usize,
            dims: (usize, usize, usize),
            (alpha, beta): (f64, f64),
            (a, ta): (&Tensor, Op),
            (b, tb): (&Tensor, Op),
            c: &mut Tensor,
            unit: Arithmetic,
        ) -> Result<(), GpuError> {
            let ((a_first, a_rest), (b_first, b_rest)) = (self.terms(a, unit)?, self.terms(b, unit)?);
            let (a1, b1) = (a_first.as_ref().unwrap_or(a), b_first.as_ref().unwrap_or(b));
            let mut scale = beta;
            if let Some(a2) = &a_rest {
                self.gemm32(batch, dims, (alpha, scale), (a2, ta), (b1, tb), c, unit)?;
                scale = 1.0;
            }
            if let Some(b2) = &b_rest {
                self.gemm32(batch, dims, (alpha, scale), (a1, ta), (b2, tb), c, unit)?;
                scale = 1.0;
            }
            self.gemm32(batch, dims, (alpha, scale), (a1, ta), (b1, tb), c, unit)
        }

        /// An operand's two terms in `unit` (`bf16_split` or `tf32_split`, `split_moves.cu`): the first
        /// (`None` where it is the operand itself) and the remainder (`None` where there is none).
        /// A bfloat16 operand is its own first term in bfloat16, and widened to f32 in TF32.
        fn terms(&self, t: &Tensor, unit: Arithmetic) -> Result<(Option<Tensor>, Option<Tensor>), GpuError> {
            let half = unit == Arithmetic::Bf16;
            if matches!(t.data, Data::CudaBf16(_)) {
                return Ok((if half { None } else { Some(self.convert_to(t, Storage::F32)?) }, None));
            }
            let n = t.len() as u64;
            let (mut first, mut rest) = if half {
                (self.unset16(t.rows, t.cols)?, self.unset16(t.rows, t.cols)?)
            } else {
                (self.unset32(t.rows, t.cols)?, self.unset32(t.rows, t.cols)?)
            };
            let f = self.split_moves(if half { "bf16_split" } else { "tf32_split" })?;
            let mut builder = self.stream.launch_builder(&f);
            builder.arg(&n).arg(slice32(t)?);
            match (&mut first.data, &mut rest.data) {
                (Data::CudaBf16(h), Data::CudaBf16(l)) => builder.arg(h).arg(l),
                (Data::Cuda32(h), Data::Cuda32(l)) => builder.arg(h).arg(l),
                (other, _) => return Err(mismatch(other)),
            };
            // SAFETY: three buffers of `n` values.
            unsafe { builder.launch(cfg_elements(n)) }.gpu_ctx("split operand terms")?;
            Ok((Some(first), Some(rest)))
        }

        /// Kernel `name` of `split_moves.cu`, its module compiled on first use.
        fn split_moves(&self, name: &'static str) -> Result<CudaFunction, GpuError> {
            let mut loaded = self.functions_split_moves.lock().map_err(|_| shape("poisoned split and move kernel table".to_string()))?;
            if let Some(f) = loaded.get(name) {
                return Ok(f.clone());
            }
            let module = match self.module_split_moves.get() {
                Some(module) => module,
                None => {
                    static MODULES: std::sync::OnceLock<crate::device_cache::KeyedPtxModuleCache<usize>> = std::sync::OnceLock::new();
                    let compiled = MODULES
                        .get_or_init(crate::device_cache::KeyedPtxModuleCache::new)
                        .get_or_compile(&self.ctx, self.ctx.ordinal(), "split_moves", |_| KERNELS_SPLIT_MOVES.to_string())?;
                    self.module_split_moves.get_or_init(|| compiled)
                }
            };
            let f = module.load_function(name).gpu_ctx_with(|e| format!("split or move kernel {name}: {e}"))?;
            loaded.insert(name, f.clone());
            Ok(f)
        }

        /// An f32 tensor's buffer left unset for a kernel that writes every value.
        fn unset32(&self, rows: usize, cols: usize) -> Result<Tensor, GpuError> {
            // SAFETY: the caller's kernel writes all values before any is read.
            let data = Data::Cuda32(unsafe { self.alloc::<f32>((rows * cols).max(1)) }.gpu_ctx("tensor alloc")?);
            Ok(Tensor { rows, cols, data })
        }

        /// A bfloat16 tensor's buffer left unset for a kernel that writes every value.
        pub(super) fn copy_ranges(&self, x: &Tensor, y: &mut Tensor, moves: &[(usize, usize, usize)]) -> Result<(), GpuError> {
            let cols = u32_of(x.cols)?;
            let f = self.split_moves("copy_ranges")?;
            for chunk in moves.chunks(super::ROW_RANGES) {
                let mut table = RowRanges { count: u32_of(chunk.len())?, from: [0; super::ROW_RANGES], to: [0; super::ROW_RANGES], length: [0; super::ROW_RANGES] };
                for (i, &(from, to, length)) in chunk.iter().enumerate() {
                    (table.from[i], table.to[i], table.length[i]) = (u32_of(from)?, u32_of(to)?, u32_of(length)?);
                }
                let longest = chunk.iter().map(|m| m.2 * x.cols).max().unwrap_or(0) as u64;
                let blocks = longest.div_ceil(u64::from(BLOCK)).clamp(1, 4096) as u32;
                let cfg = LaunchConfig { grid_dim: (blocks, table.count.max(1), 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
                // SAFETY: every move's rows lie in its buffer (checked by the caller).
                unsafe { self.stream.launch_builder(&f).arg(&table).arg(&cols).arg(slice32(x)?).arg(slice32_mut(y)?).launch(cfg) }.gpu_ctx("decoder copy_ranges")?;
            }
            Ok(())
        }

        fn unset16(&self, rows: usize, cols: usize) -> Result<Tensor, GpuError> {
            // SAFETY: as `unset32`.
            let data = Data::CudaBf16(unsafe { self.alloc::<u16>((rows * cols).max(1)) }.gpu_ctx("tensor alloc")?);
            Ok(Tensor { rows, cols, data })
        }

        /// [`super::Device::gram_lower`]: the row-major `aᵀ a` is the column-major `A Aᵀ` of the same
        /// buffer read as n × rows, and its row-major lower triangle that product's upper one.
        pub(super) fn gram_lower(&self, c: &mut Tensor, a: &Tensor, beta: f64) -> Result<(), GpuError> {
            if a.cols == 0 {
                return Ok(());
            }
            let serial = self.gemm_workspace.lock().map_err(|_| shape("poisoned GEMM workspace".to_string()))?;
            let (n, k) = (i32_of(a.cols)?, i32_of(a.rows)?);
            let alpha = 1.0_f64;
            let (Data::Cuda(sa), Data::Cuda(sc)) = (&a.data, &mut c.data) else { return Err(mismatch(&a.data)) };
            let (pa, record_a) = sa.device_ptr(&self.stream);
            let (pc, record_c) = sc.device_ptr_mut(&self.stream);
            // SAFETY: `a` holds rows × n and `c` n × n float64 values (checked by the caller); the
            // pointers outlive the call (their records drop after it).
            let update = unsafe {
                cudarc::cublas::sys::cublasDsyrk_v2(
                    *self.blas.handle(),
                    cudarc::cublas::sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
                    cublasOperation_t::CUBLAS_OP_N,
                    n,
                    k,
                    &alpha,
                    pa as *const f64,
                    n,
                    &beta,
                    pc as *mut f64,
                    n,
                )
            }
            .result();
            drop((record_a, record_c, serial));
            update.gpu_ctx("tensor DSYRK")
        }

        /// [`super::Device::gram_split`]: the exponents, the slices (column-major, rows padded to a
        /// multiple of 4 for the integer kernels), and per shift `m < slices` the int32 sum of the
        /// products `a_sᵀ a_t` with `s + t = m`, `s ≤ t` (those with `s < t` twice), whose
        /// symmetric part is added into `c`.
        pub(super) fn gram_split(&self, c: &mut Tensor, a: &Tensor, slices: usize) -> Result<(), GpuError> {
            use cudarc::cublas::sys::{cublasComputeType_t, cublasGemmAlgo_t};
            let (rows, cols) = (a.rows, a.cols);
            let stride = rows.div_ceil(4) * 4;
            let x = slice32(a)?;
            let (rows_u, cols_u, stride_u, slices_u) = (u32_of(rows)?, u32_of(cols)?, u32_of(stride)?, u32_of(slices)?);
            // Every buffer below is written whole before it is read (each column's exponent, every
            // slice entry with the padding rows, each product with β = 0 first), so none is zeroed,
            // and they are kept for the next call (held for this one, which orders their reuse on
            // the stream).
            let mut workspace = self.split_workspace.lock().map_err(|_| shape("poisoned split workspace".to_string()))?;
            let SplitWorkspace { exponents, parts, product } = &mut *workspace;
            at_least(&self.stream, exponents, cols)?;
            at_least(&self.stream, parts, slices * cols * stride)?;
            at_least(&self.stream, product, cols * cols)?;
            let (Some(exponents), Some(parts), Some(product)) = (exponents.as_mut(), parts.as_mut(), product.as_mut()) else {
                return Err(shape("split workspace missing".to_string()));
            };
            let f = self.kernel("split_exponents", Storage::F32)?;
            let tiles = u32_of(cols.div_ceil(32))?;
            let cfg = LaunchConfig { grid_dim: (tiles, 1, 1), block_dim: (32, 32, 1), shared_mem_bytes: 0 };
            // SAFETY: `x` holds rows × cols floats and `exponents` cols integers.
            unsafe { self.stream.launch_builder(&f).arg(&rows_u).arg(&cols_u).arg(x).arg(&mut *exponents).launch(cfg) }.gpu_ctx("split_exponents")?;
            let f = self.kernel("split_slices", Storage::F32)?;
            let cfg = LaunchConfig { grid_dim: (tiles, u32_of(stride.div_ceil(32))?, 1), block_dim: (32, 8, 1), shared_mem_bytes: 0 };
            // SAFETY: `parts` holds slices × cols × stride bytes.
            unsafe { self.stream.launch_builder(&f).arg(&rows_u).arg(&cols_u).arg(&stride_u).arg(&slices_u).arg(x).arg(&*exponents).arg(&mut *parts).launch(cfg) }.gpu_ctx("split_slices")?;
            let combine = self.kernel("split_combine", Storage::F32)?;
            let g = slice_mut(c)?;
            let (n, k) = (i32_of(cols)?, i32_of(stride)?);
            for shift in 0..slices {
                for s in 0..=shift / 2 {
                    let t = shift - s;
                    // The first product of the shift overwrites, the rest add; a pair s < t counts twice.
                    let (alpha, beta) = (if s < t { 2_i32 } else { 1_i32 }, i32::from(s > 0));
                    let serial = self.gemm_workspace.lock().map_err(|_| shape("poisoned GEMM workspace".to_string()))?;
                    let (base, record_a) = parts.device_ptr(&self.stream);
                    let (pc, record_c) = product.device_ptr_mut(&self.stream);
                    let (left, right) = (base + (s * cols * stride) as u64, base + (t * cols * stride) as u64);
                    // SAFETY: each slice holds cols × stride bytes (column-major, leading dimension
                    // stride, a multiple of 4) and `product` cols × cols integers.
                    let status = unsafe {
                        cudarc::cublas::sys::cublasGemmEx(
                            *self.blas.handle(),
                            cublasOperation_t::CUBLAS_OP_T,
                            cublasOperation_t::CUBLAS_OP_N,
                            n,
                            n,
                            k,
                            (&raw const alpha).cast(),
                            left as *const std::ffi::c_void,
                            cudaDataType_t::CUDA_R_8I,
                            k,
                            right as *const std::ffi::c_void,
                            cudaDataType_t::CUDA_R_8I,
                            k,
                            (&raw const beta).cast(),
                            pc as *mut std::ffi::c_void,
                            cudaDataType_t::CUDA_R_32I,
                            n,
                            cublasComputeType_t::CUBLAS_COMPUTE_32I,
                            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                        )
                    }
                    .result();
                    drop((record_a, record_c, serial));
                    status.gpu_ctx("split int8 product")?;
                }
                let shift = i32::try_from(shift).map_err(|_| shape("slices".to_string()))?;
                let cfg = LaunchConfig { grid_dim: (tiles, tiles, 1), block_dim: (32, 8, 1), shared_mem_bytes: 0 };
                // SAFETY: `product` holds cols × cols integers, `exponents` cols, `g` cols × cols doubles.
                unsafe { self.stream.launch_builder(&combine).arg(&cols_u).arg(&shift).arg(&*product).arg(&*exponents).arg(&mut *g).launch(cfg) }.gpu_ctx("split_combine")?;
            }
            Ok(())
        }

        /// [`super::Device::symmetric_eigh`] on a stream of its own, so concurrent callers do not
        /// share one: the row-major lower triangle is the column-major upper one, which
        /// `cusolverDnDsyevd` reads, and the buffer it leaves holds the eigenvectors as
        /// column-major columns.
        pub(super) fn symmetric_eigh(&self, a: ndarray::ArrayView2<'_, f64>) -> Result<(ndarray::Array1<f64>, ndarray::Array2<f64>), GpuError> {
            use cudarc::cusolver::sys as solver;
            let n = a.nrows();
            let order = i32_of(n)?;
            let failed = |what: &str, e: cudarc::cusolver::result::CusolverError| GpuError::DriverCallFailed { reason: format!("cuSOLVER symmetric eigendecomposition, {what}: {e:?}") };
            let stream = self.ctx.new_stream().gpu_ctx("eigendecomposition stream")?;
            let handle = cudarc::cusolver::DnHandle::new(stream.clone()).map_err(|e| failed("handle", e))?;
            let entries: Vec<f64> = a.iter().copied().collect();
            let mut matrix = stream.clone_htod(&entries).gpu_ctx("eigendecomposition upload")?;
            let mut values = stream.alloc_zeros::<f64>(n).gpu_ctx("eigendecomposition values")?;
            let mut info = stream.alloc_zeros::<i32>(1).gpu_ctx("eigendecomposition info")?;
            let (jobz, uplo) = (solver::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR, solver::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER);
            let mut lwork = 0_i32;
            {
                let (pa, record_a) = matrix.device_ptr(&stream);
                let (pw, record_w) = values.device_ptr(&stream);
                // SAFETY: `matrix` holds n × n and `values` n float64 values; the pointers outlive
                // the call.
                let sized = unsafe { solver::cusolverDnDsyevd_bufferSize(handle.cu(), jobz, uplo, order, pa as *const f64, order, pw as *const f64, &mut lwork) }.result();
                drop((record_a, record_w));
                sized.map_err(|e| failed("workspace size", e))?;
            }
            let mut work = stream.alloc_zeros::<f64>(usize::try_from(lwork.max(1)).unwrap_or(1)).gpu_ctx("eigendecomposition workspace")?;
            {
                let (pa, record_a) = matrix.device_ptr_mut(&stream);
                let (pw, record_w) = values.device_ptr_mut(&stream);
                let (pk, record_k) = work.device_ptr_mut(&stream);
                let (pi, record_i) = info.device_ptr_mut(&stream);
                // SAFETY: as above, with `work` of `lwork` float64 values and `info` one integer.
                let solved = unsafe { solver::cusolverDnDsyevd(handle.cu(), jobz, uplo, order, pa as *mut f64, order, pw as *mut f64, pk as *mut f64, lwork, pi as *mut i32) }.result();
                drop((record_a, record_w, record_k, record_i));
                solved.map_err(|e| failed("solve", e))?;
            }
            let status = stream.clone_dtoh(&info).gpu_ctx("eigendecomposition info")?;
            if status.first().copied() != Some(0) {
                return Err(GpuError::DriverCallFailed { reason: format!("cusolverDnDsyevd: info {status:?} (a positive value: off-diagonal entries that did not converge)") });
            }
            let values = stream.clone_dtoh(&values).gpu_ctx("eigendecomposition values")?;
            let columns = stream.clone_dtoh(&matrix).gpu_ctx("eigendecomposition vectors")?;
            Ok((ndarray::Array1::from(values), ndarray::Array2::from_shape_fn((n, n), |(i, j)| columns[j * n + i])))
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

        /// [`super::Device::axpy_rows`]: `n` entries of `y` from entry `at`, plus `alpha` times
        /// those of `x` from entry `from`.
        pub(super) fn axpy_rows(&self, y: &mut Tensor, at: usize, alpha: f64, (x, from): (&Tensor, usize), n: usize) -> Result<(), GpuError> {
            let storage = y.storage();
            if storage == Storage::Bf16 {
                return Err(shape("rows added in bfloat16 storage".to_string()));
            }
            let f = self.kernel("axpy_rows", storage)?;
            let (n, from, at) = (n as u64, from as u64, at as u64);
            // SAFETY: the ranges lie in their buffers (checked by the caller), in `storage` (`input`).
            unsafe { self.stream.launch_builder(&f).arg(&n).arg(&alpha).input(x, storage)?.arg(&from).output(y, storage)?.arg(&at).launch(cfg_elements(n)) }
                .gpu_ctx("tensor axpy_rows")
                .map(|_| ())
        }

        /// [`super::Device::axpy_rows_within`] (`alpha`) and [`super::Device::copy_rows_within`]
        /// (none): `n` entries of `t` from entry `at`, from its `n` entries from entry `from`.
        pub(super) fn rows_within(&self, t: &mut Tensor, (at, from, n): (usize, usize, usize), alpha: Option<f64>) -> Result<(), GpuError> {
            let storage = t.storage();
            if storage == Storage::Bf16 {
                return Err(shape("rows moved in bfloat16 storage".to_string()));
            }
            let (n, from, at) = (n as u64, from as u64, at as u64);
            let launched = match alpha {
                Some(alpha) => {
                    let f = self.kernel("axpy_within", storage)?;
                    // SAFETY: both ranges lie in the buffer and apart (checked by the caller).
                    unsafe { self.stream.launch_builder(&f).arg(&n).arg(&alpha).arg(&from).arg(&at).output(t, storage)?.launch(cfg_elements(n)) }
                }
                None => {
                    let f = self.kernel("copy_within", storage)?;
                    // SAFETY: as above.
                    unsafe { self.stream.launch_builder(&f).arg(&n).arg(&from).arg(&at).output(t, storage)?.launch(cfg_elements(n)) }
                }
            };
            launched.gpu_ctx("tensor rows_within").map(|_| ())
        }

        pub(super) fn scaled(&self, alpha: f64, x: &Tensor) -> Result<Tensor, GpuError> {
            let (n, storage) = (x.len() as u64, x.storage());
            if storage == Storage::Bf16 {
                return Err(shape("a bfloat16 tensor scaled".to_string()));
            }
            let mut out = self.output(storage, x.rows, x.cols)?;
            let (f, zero) = (self.kernel("scaled", storage)?, 0.0_f64);
            // SAFETY: `out` holds as many values as `x`, in its storage (`input`).
            unsafe { self.stream.launch_builder(&f).arg(&n).arg(&zero).arg(&alpha).input(x, storage)?.output(&mut out, storage)?.launch(cfg_elements(n)) }
                .gpu_ctx("tensor scaled")?;
            Ok(out)
        }

        pub(super) fn move_toward(&self, y: &mut Tensor, weight: f64, x: &Tensor) -> Result<(), GpuError> {
            let n = y.len() as u64;
            let storage = y.storage();
            let f = self.kernel("move_toward", storage)?;
            // SAFETY: equal-length buffers, checked by the caller.
            unsafe { self.stream.launch_builder(&f).arg(&n).arg(&weight).input(x, storage)?.output(y, storage)?.launch(cfg_elements(n)) }
                .gpu_ctx("tensor move_toward")
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
            Ok(std::mem::replace(&mut output.data, Data::Host(Vec::new())))
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

        pub(super) fn nonzero_columns(&self, x: &Tensor, rows: &Indices) -> Result<Tensor, GpuError> {
            let storage = x.storage();
            if storage == Storage::Bf16 {
                return Err(shape("nonzero_columns of a bfloat16 tensor".to_string()));
            }
            let mut out = self.output(storage, 1, x.cols)?;
            let n = x.cols as u64;
            let nr = rows.len as u32;
            let f = self.kernel("nonzero_columns", storage)?;
            // SAFETY: `rows` index rows of `x` (checked by the caller); `out` holds n values.
            unsafe { self.stream.launch_builder(&f).arg(&n).arg(&nr).input(x, storage)?.arg(index_slice(rows)?).output(&mut out, storage)?.launch(cfg_elements(n)) }
                .gpu_ctx("tensor nonzero_columns")?;
            Ok(out)
        }

        pub(super) fn row_counts(&self, mask: &Tensor, starts: &Indices) -> Result<Tensor, GpuError> {
            let storage = mask.storage();
            if storage == Storage::Bf16 {
                return Err(shape("row lists of a bfloat16 mask".to_string()));
            }
            let mut out = self.output(storage, 1, mask.rows)?;
            let (rows, groups) = (u32_of(mask.rows)?, u32_of(mask.cols)?);
            let f = self.kernel("row_counts", storage)?;
            // SAFETY: `starts` holds groups + 1 increasing columns (checked by the caller).
            unsafe { self.stream.launch_builder(&f).arg(&rows).arg(&groups).input(mask, storage)?.arg(index_slice(starts)?).output(&mut out, storage)?.launch(cfg_elements(u64::from(rows) * 32)) }
                .gpu_ctx("tensor row_counts")?;
            Ok(out)
        }

        pub(super) fn row_fill(&self, mask: &Tensor, starts: &Indices, offsets: &Indices, entries: usize) -> Result<(Indices, Indices), GpuError> {
            let storage = mask.storage();
            let (rows, groups) = (u32_of(mask.rows)?, u32_of(mask.cols)?);
            let (mut columns, mut row_of) = (self.alloc_zeros::<u32>(entries.max(1)).gpu_ctx("tensor alloc")?, self.alloc_zeros::<u32>(entries.max(1)).gpu_ctx("tensor alloc")?);
            let f = self.kernel("row_fill", storage)?;
            // SAFETY: `offsets` are the prefix sums of row_counts' counts, so each row writes inside
            // its own `entries`-long span.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&groups)
                    .input(mask, storage)?
                    .arg(index_slice(starts)?)
                    .arg(index_slice(offsets)?)
                    .arg(&mut columns)
                    .arg(&mut row_of)
                    .launch(cfg_elements(u64::from(rows) * 32))
            }
            .gpu_ctx("tensor row_fill")?;
            Ok((Indices { len: entries, data: IndexData::Cuda(columns) }, Indices { len: entries, data: IndexData::Cuda(row_of) }))
        }

        pub(super) fn sampled_product(&self, x: &Tensor, a: &Tensor, lists: &super::RowLists, out: &mut Tensor) -> Result<(), GpuError> {
            if lists.is_empty() {
                return Ok(());
            }
            let storage = x.storage();
            let (entries, k, n) = (lists.len() as u64, u32_of(x.cols)?, u32_of(a.rows)?);
            let f = self.kernel("sampled_product", storage)?;
            // SAFETY: the lists' entries are inside out (rows × n); x and a hold k columns.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&entries)
                    .arg(&k)
                    .arg(&n)
                    .input(x, storage)?
                    .input(a, storage)?
                    .arg(index_slice(&lists.columns)?)
                    .arg(index_slice(&lists.row_of)?)
                    .output(out, storage)?
                    .launch(cfg_elements(entries * 32))
            }
            .gpu_ctx("tensor sampled_product")
            .map(|_| ())
        }

        pub(super) fn listed_product(&self, v: &Tensor, lists: &super::RowLists, a: &Tensor) -> Result<Tensor, GpuError> {
            let storage = v.storage();
            let mut out = self.output(storage, v.rows, a.cols)?;
            let (total, m, n) = (out.len() as u64, u32_of(a.cols)?, u32_of(v.cols)?);
            let f = self.kernel("listed_product", storage)?;
            // SAFETY: offsets has rows + 1 entries, the columns inside v's n columns (a's rows).
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&total)
                    .arg(&m)
                    .arg(&n)
                    .input(v, storage)?
                    .input(a, storage)?
                    .arg(index_slice(&lists.offsets)?)
                    .arg(index_slice(&lists.columns)?)
                    .output(&mut out, storage)?
                    .launch(cfg_elements(total))
            }
            .gpu_ctx("tensor listed_product")?;
            Ok(out)
        }

        pub(super) fn listed_product_t(&self, v: &Tensor, lists: &super::RowLists, group_of: &Indices, y: &Tensor) -> Result<Tensor, GpuError> {
            let storage = v.storage();
            let mut out = self.output(storage, v.cols, y.cols)?;
            let (total, m, n) = (out.len() as u64, u32_of(y.cols)?, u32_of(v.cols)?);
            let f = self.kernel("listed_product_t", storage)?;
            // SAFETY: group_of's groups index the lists' offsets (groups + 1), their rows inside v's
            // and y's rows (checked shapes).
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&total)
                    .arg(&m)
                    .arg(&n)
                    .input(v, storage)?
                    .input(y, storage)?
                    .arg(index_slice(&lists.offsets)?)
                    .arg(index_slice(&lists.columns)?)
                    .arg(index_slice(group_of)?)
                    .output(&mut out, storage)?
                    .launch(cfg_elements(total))
            }
            .gpu_ctx("tensor listed_product_t")?;
            Ok(out)
        }

        pub(super) fn row_norm(&self, mode: super::RowNorm, y: &Tensor, along: Option<&Tensor>, (gain, bias): (&Tensor, Option<&Tensor>), epsilon: f64) -> Result<Tensor, GpuError> {
            let storage = y.storage();
            if storage == Storage::Bf16 {
                return Err(shape("row_norm of a bfloat16 tensor".to_string()));
            }
            let mut out = self.output(storage, y.rows, y.cols)?;
            let (rows, d, code, has_bias) = (u32_of(y.rows)?, u32_of(y.cols)?, mode.code(), i32::from(bias.is_some()));
            let f = self.kernel("row_norm", storage)?;
            // SAFETY: y, along and out hold rows × d values, gain and bias d (checked by the caller);
            // `along` and `bias` stand in for themselves as `y` and `gain` when absent (not read).
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&d)
                    .arg(&code)
                    .arg(&epsilon)
                    .input(y, storage)?
                    .input(along.unwrap_or(y), storage)?
                    .input(gain, storage)?
                    .input(bias.unwrap_or(gain), storage)?
                    .arg(&has_bias)
                    .output(&mut out, storage)?
                    .launch(cfg_elements(u64::from(rows)))
            }
            .gpu_ctx("tensor row_norm")?;
            Ok(out)
        }

        pub(super) fn transpose(&self, t: &Tensor) -> Result<Tensor, GpuError> {
            let storage = t.storage();
            if storage == Storage::Bf16 {
                return Err(shape("transpose of a bfloat16 tensor".to_string()));
            }
            let mut out = self.output(storage, t.cols, t.rows)?;
            let (n, rows, cols) = (t.len() as u64, u32_of(t.rows)?, u32_of(t.cols)?);
            let f = self.kernel("transpose", storage)?;
            // SAFETY: out holds the n values of t.
            unsafe { self.stream.launch_builder(&f).arg(&n).arg(&rows).arg(&cols).input(t, storage)?.output(&mut out, storage)?.launch(cfg_elements(n)) }
                .gpu_ctx("tensor transpose")?;
            Ok(out)
        }

        pub(super) fn gather_columns(&self, t: &Tensor, ids: &Indices) -> Result<Tensor, GpuError> {
            let storage = t.storage();
            if storage == Storage::Bf16 {
                return Err(shape("gather_columns of a bfloat16 tensor".to_string()));
            }
            let mut out = self.output(storage, t.rows, ids.len)?;
            let n = out.len() as u64;
            let (m, cols) = (ids.len as u32, t.cols as u32);
            let f = self.kernel("gather_columns", storage)?;
            // SAFETY: ids index columns of `t` (the caller's nonzero columns of it).
            unsafe { self.stream.launch_builder(&f).arg(&n).arg(&m).arg(&cols).input(t, storage)?.arg(index_slice(ids)?).output(&mut out, storage)?.launch(cfg_elements(n)) }
                .gpu_ctx("tensor gather_columns")?;
            Ok(out)
        }

        pub(super) fn scatter_columns(&self, t: &mut Tensor, ids: &Indices, values: &Tensor, accumulate: bool) -> Result<(), GpuError> {
            let storage = t.storage();
            if values.storage() != storage || storage == Storage::Bf16 {
                return Err(shape("scatter_columns: values stored unlike the target, or in bfloat16".to_string()));
            }
            let n = values.len() as u64;
            let (m, cols, acc) = (ids.len as u32, t.cols as u32, i32::from(accumulate));
            let f = self.kernel("scatter_columns", storage)?;
            // SAFETY: ids are distinct columns of `t`; `values` is rows × m.
            unsafe { self.stream.launch_builder(&f).arg(&n).arg(&m).arg(&cols).output(t, storage)?.arg(index_slice(ids)?).input(values, storage)?.arg(&acc).launch(cfg_elements(n)) }
                .gpu_ctx("tensor scatter_columns")
                .map(|_| ())
        }

        pub(super) fn scatter_rows(&self, t: &mut Tensor, ids: &Indices, values: &Tensor, accumulate: bool) -> Result<(), GpuError> {
            let storage = t.storage();
            if values.storage() != storage || storage == Storage::Bf16 {
                return Err(shape("scatter_rows: values stored unlike the target, or in bfloat16".to_string()));
            }
            let n = values.len() as u64;
            let (cols, acc) = (t.cols as u32, i32::from(accumulate));
            let f = self.kernel("scatter_rows", storage)?;
            // SAFETY: ids are distinct rows of `t`; `values` is ids × cols.
            unsafe { self.stream.launch_builder(&f).arg(&n).arg(&cols).output(t, storage)?.arg(index_slice(ids)?).input(values, storage)?.arg(&acc).launch(cfg_elements(n)) }
                .gpu_ctx("tensor scatter_rows")
                .map(|_| ())
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

        pub(super) fn laws_both(&self, x: &Tensor, codes: &Indices, c: f64) -> Result<(Tensor, Tensor), GpuError> {
            let mut out = self.output(Storage::F32, x.rows, x.cols)?;
            let mut half = self.unset16(x.rows, x.cols)?;
            let (n, cols) = (x.len() as u64, x.cols as u32);
            let f = self.kernel("laws_both", Storage::F32)?;
            let Data::CudaBf16(h) = &mut half.data else { return Err(shape("a bfloat16 output".to_string())) };
            // SAFETY: equal-length f32 buffers and a bfloat16 one of their length; `codes` holds one
            // code per column.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&n)
                    .arg(&cols)
                    .input(x, Storage::F32)?
                    .arg(index_slice(codes)?)
                    .arg(&c)
                    .output(&mut out, Storage::F32)?
                    .arg(h)
                    .launch(cfg_elements(n))
            }
            .gpu_ctx("tensor laws_both")?;
            Ok((out, half))
        }

        pub(super) fn rms_both(&self, x: &Tensor, epsilon: f64) -> Result<(Tensor, Tensor), GpuError> {
            let mut out = self.output(Storage::F32, x.rows, x.cols)?;
            let mut half = self.unset16(x.rows, x.cols)?;
            let (rows, cols) = (x.rows as u32, x.cols as u32);
            let f = self.kernel("rms_both", Storage::F32)?;
            let Data::CudaBf16(h) = &mut half.data else { return Err(shape("a bfloat16 output".to_string())) };
            // SAFETY: one block per row of equal-shape f32 buffers and a bfloat16 one of their shape.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(&epsilon)
                    .input(x, Storage::F32)?
                    .output(&mut out, Storage::F32)?
                    .arg(h)
                    .launch(cfg_rows(x.rows))
            }
            .gpu_ctx("tensor rms_both")?;
            Ok((out, half))
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

        pub(super) fn softmax_backward_bf16(&self, alpha: &Tensor, d: &Tensor) -> Result<Tensor, GpuError> {
            let mut out = self.unset16(alpha.rows, alpha.cols)?;
            let (rows, cols) = (alpha.rows as u32, alpha.cols as u32);
            let f = self.kernel("softmax_backward_bf16", Storage::F32)?;
            let Data::CudaBf16(o) = &mut out.data else { return Err(shape("a bfloat16 output".to_string())) };
            // SAFETY: one block per row of equal-shape f32 buffers and a bfloat16 output of their shape.
            unsafe { self.stream.launch_builder(&f).arg(&rows).arg(&cols).input(alpha, Storage::F32)?.input(d, Storage::F32)?.arg(o).launch(cfg_rows(alpha.rows)) }
                .gpu_ctx("tensor softmax_backward_bf16")?;
            Ok(out)
        }

        /// [`super::Device::split_heads_bf16`] and, `wide`, [`super::Device::split_heads_both`]:
        /// `heads_permute`'s split written in bfloat16, and in f32 too when `wide`.
        pub(super) fn split_halves(&self, x: &Tensor, (start, heads, width, length, planes): (usize, usize, usize, usize, usize), turn: Option<(&Tensor, &Tensor, bool)>, inverse: bool, wide: bool) -> Result<(Option<Tensor>, Tensor), GpuError> {
            let rows = x.rows;
            let mut out = self.unset16(rows * heads, width)?;
            let mut full = if wide { Some(self.output(Storage::F32, rows * heads, width)?) } else { None };
            if rows == 0 || heads == 0 || width == 0 {
                return Ok((full, out));
            }
            let n = (rows * heads * width) as u64;
            let half_split = i32::from(turn.is_some_and(|(_, _, h)| h));
            let (rows32, cols, start, heads, width, length, planes) = (rows as u32, x.cols as u32, start as u32, heads as u32, width as u32, length as u32, planes as u32);
            let sign: f64 = if inverse { -1.0 } else { 1.0 };
            // Without a rotation the tables are never read; `x` stands in for them.
            let (cos, sin) = turn.map_or((x, x), |(c, s, _)| (c, s));
            let f = self.kernel(if wide { "heads_permute_both" } else { "heads_permute_bf16" }, Storage::F32)?;
            let Data::CudaBf16(o) = &mut out.data else { return Err(shape("a bfloat16 output".to_string())) };
            // SAFETY: shapes checked by the caller: `x` holds `rows × cols` f32 values, the tables
            // `rows × planes` when `planes > 0`, and the output `rows·heads·width` halves.
            unsafe {
                let mut builder = self.stream.launch_builder(&f);
                builder
                    .arg(&rows32)
                    .arg(&cols)
                    .arg(&start)
                    .arg(&heads)
                    .arg(&width)
                    .arg(&length)
                    .arg(&planes)
                    .arg(&half_split)
                    .arg(&sign)
                    .input(x, Storage::F32)?
                    .input(cos, Storage::F32)?
                    .input(sin, Storage::F32)?;
                if let Some(full) = full.as_mut() {
                    builder.output(full, Storage::F32)?;
                }
                builder.arg(o).launch(cfg_elements(n))
            }
            .gpu_ctx("tensor split_heads_bf16")?;
            Ok((full, out))
        }

        pub(super) fn softmax_rows_bf16(&self, scores: &mut Tensor, causal: bool, period: usize) -> Result<Tensor, GpuError> {
            let mut half = self.unset16(scores.rows, scores.cols)?;
            let (rows, width, causal, start, period) = (scores.rows as u32, scores.cols as u32, i32::from(causal), 0u32, period as u32);
            let f = self.kernel("softmax_rows_bf16", Storage::F32)?;
            let launch = cfg_rows(scores.rows);
            let Data::CudaBf16(h) = &mut half.data else { return Err(shape("a bfloat16 output".to_string())) };
            // SAFETY: one block per row of a rows × width f32 buffer and a bfloat16 one of its shape.
            unsafe { self.stream.launch_builder(&f).arg(&rows).arg(&width).arg(&causal).arg(&start).arg(&period).output(scores, Storage::F32)?.arg(h).launch(launch) }
                .gpu_ctx("tensor softmax_rows_bf16")?;
            Ok(half)
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


        pub(super) fn fisher_probe_cotangent(&self, probabilities: &mut Tensor, (key, first): (u64, usize), scored: Option<&Indices>) -> Result<(), GpuError> {
            let (rows, cols) = (probabilities.rows as u32, probabilities.cols as u32);
            let (flags, use_flags) = self.flags(scored)?;
            let storage = probabilities.storage();
            let f = self.kernel("fisher_probe", storage)?;
            let (n_rows, first) = (probabilities.rows, first as u64);
            // SAFETY: one block per row of a rows × cols buffer; `flags` has a flag per row when used.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&cols)
                    .output(probabilities, storage)?
                    .arg(&key)
                    .arg(&first)
                    .arg(flags)
                    .arg(&use_flags)
                    .launch(cfg_rows(n_rows))
            }
            .gpu_ctx("tensor fisher_probe")
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

        pub(super) fn gate_function(&self, function: super::GateFunction, x: &Tensor, s: Option<&Tensor>) -> Result<Tensor, GpuError> {
            let storage = x.storage();
            if storage == Storage::Bf16 || s.is_some_and(|s| s.storage() != storage) {
                return Err(shape("gate_function: float32 or float64, the scale stored alike".to_string()));
            }
            let mut out = self.output(storage, x.rows, x.cols)?;
            let (n, code, scaled) = (x.len() as u64, function.code(), i32::from(s.is_some()));
            let f = self.kernel("gate_function", storage)?;
            // SAFETY: equal-length buffers; `s` stands in for itself as `x` when absent (not read).
            unsafe { self.stream.launch_builder(&f).arg(&n).arg(&code).input(x, storage)?.input(s.unwrap_or(x), storage)?.arg(&scaled).output(&mut out, storage)?.launch(cfg_elements(n)) }
                .gpu_ctx("tensor gate_function")?;
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

        pub(super) fn round_to_deviation(&self, out: &mut Tensor, (mean, log_sd): (&Tensor, &Tensor)) -> Result<(), GpuError> {
            let (n, storage) = (out.len() as u64, out.storage());
            let f = self.posterior_kernel("round_to_deviation", storage)?;
            // SAFETY: three equal-length buffers in one storage, checked by the caller and `input`.
            unsafe { self.stream.launch_builder(&f).arg(&n).input(mean, storage)?.input(log_sd, storage)?.output(out, storage)?.launch(cfg_elements(n)) }
                .gpu_ctx("tensor round_to_deviation")
                .map(|_| ())
        }

        /// [`super::Device::reparameterize_block`]: the kernel on a view of `out` from the block's
        /// first entry `at`, its rows `stride` apart.
        pub(super) fn reparameterize_block(&self, out: &mut Tensor, (at, stride): (usize, usize), (mean, log_sd): (&Tensor, &Tensor), (key, stream): (u64, u64)) -> Result<(), GpuError> {
            let (n, cols) = (mean.len() as u64, mean.cols as u64);
            let span = (mean.rows - 1) * stride + mean.cols;
            let stride64 = stride as u64;
            let (f, masters) = match &out.data {
                Data::CudaBf16(_) => (self.function("reparameterize_bf16")?, Storage::F32),
                Data::Cuda32(_) => (self.posterior_kernel("reparameterize", Storage::F32)?, Storage::F32),
                Data::Cuda(_) => (self.posterior_kernel("reparameterize", Storage::F64)?, Storage::F64),
                Data::Host(_) => return Err(foreign()),
            };
            let launched = match &mut out.data {
                Data::CudaBf16(slice) => {
                    let mut view = slice.slice_mut(at..at + span);
                    let mut builder = self.stream.launch_builder(&f);
                    builder.arg(&n).arg(&cols).arg(&stride64).arg(&key).arg(&stream).input(mean, masters)?.input(log_sd, masters)?.arg(&mut view);
                    // SAFETY: masters of n entries in the storage the kernel reads (`input`); the
                    // view spans the block, rows `stride` apart, inside `out` (checked by the caller).
                    unsafe { builder.launch(cfg_elements(n)) }
                }
                Data::Cuda32(slice) => {
                    let mut view = slice.slice_mut(at..at + span);
                    let mut builder = self.stream.launch_builder(&f);
                    builder.arg(&n).arg(&cols).arg(&stride64).arg(&key).arg(&stream).input(mean, masters)?.input(log_sd, masters)?.arg(&mut view);
                    // SAFETY: masters of n entries in the storage the kernel reads (`input`); the
                    // view spans the block, rows `stride` apart, inside `out` (checked by the caller).
                    unsafe { builder.launch(cfg_elements(n)) }
                }
                Data::Cuda(slice) => {
                    let mut view = slice.slice_mut(at..at + span);
                    let mut builder = self.stream.launch_builder(&f);
                    builder.arg(&n).arg(&cols).arg(&stride64).arg(&key).arg(&stream).input(mean, masters)?.input(log_sd, masters)?.arg(&mut view);
                    // SAFETY: masters of n entries in the storage the kernel reads (`input`); the
                    // view spans the block, rows `stride` apart, inside `out` (checked by the caller).
                    unsafe { builder.launch(cfg_elements(n)) }
                }
                Data::Host(_) => return Err(foreign()),
            };
            launched.gpu_ctx("tensor reparameterize").map(|_| ())
        }

        /// [`super::Device::add_sample`]'s record of one sample: its output's storage, and its
        /// output address at entry `at` (rows `stride` apart), its masters' addresses, entries,
        /// columns, row stride and stream.
        pub(super) fn sample_job(&self, out: &mut Tensor, (at, stride): (usize, usize), (mean, log_sd): (&Tensor, &Tensor), stream: u64) -> Result<(Storage, [u64; 7]), GpuError> {
            let address = |t: &Tensor, masters: Storage| -> Result<u64, GpuError> {
                let (pointer, record) = match (&t.data, masters) {
                    (Data::Cuda(s), Storage::F64) => s.device_ptr(&self.stream),
                    (Data::Cuda32(s), Storage::F32) => s.device_ptr(&self.stream),
                    (other, _) => return Err(mismatch(other)),
                };
                drop(record);
                Ok(pointer)
            };
            let (storage, masters, base, bytes) = match &out.data {
                Data::CudaBf16(s) => (Storage::Bf16, Storage::F32, s.device_ptr(&self.stream).0, 2),
                Data::Cuda32(s) => (Storage::F32, Storage::F32, s.device_ptr(&self.stream).0, 4),
                Data::Cuda(s) => (Storage::F64, Storage::F64, s.device_ptr(&self.stream).0, 8),
                Data::Host(_) => return Err(foreign()),
            };
            let job = [base + bytes * at as u64, address(mean, masters)?, address(log_sd, masters)?, mean.len() as u64, mean.cols as u64, stride as u64, stream];
            Ok((storage, job))
        }

        /// [`super::Device::run_samples`]: per output storage, one launch of every sample in it.
        pub(super) fn run_samples(&self, key: u64, jobs: &[(Storage, [u64; 7])]) -> Result<(), GpuError> {
            for (storage, name) in [(Storage::F64, "reparameterize_many_f64"), (Storage::F32, "reparameterize_many_f32"), (Storage::Bf16, "reparameterize_many_bf16")] {
                let mut table = Vec::new();
                let mut total = 0u64;
                for (_, [out, mean, log_sd, entries, cols, stride, stream]) in jobs.iter().filter(|(s, _)| *s == storage) {
                    table.extend([*out, *mean, *log_sd, *cols, *stride, *stream, total, 0]);
                    total += entries;
                }
                if total == 0 {
                    continue;
                }
                let samples = (table.len() / 8) as u64;
                let table = self.stream.clone_htod(&table).gpu_ctx("tensor sample table")?;
                let f = self.function(name)?;
                // SAFETY: the table's addresses are the samples' live buffers (`Device::add_sample`'s
                // contract), each output block inside its tensor and each pair of masters of its
                // sample's entries (checked as each was gathered).
                unsafe { self.stream.launch_builder(&f).arg(&samples).arg(&table).arg(&total).arg(&key).launch(cfg_elements(total)) }.gpu_ctx("tensor run_samples")?;
            }
            Ok(())
        }

        pub(super) fn posterior_ivon(
            &self,
            (mean, log_sd): (&Tensor, &mut Tensor),
            [momentum, curvature]: [&mut Tensor; 2],
            (gradient, factor, prior): (Option<&Tensor>, Option<&Tensor>, Option<&Tensor>),
            (groups, variance): (&super::GroupMap, &Tensor),
            (direction, sums): (&mut Tensor, &mut Tensor),
            step: &super::PosteriorStep,
        ) -> Result<(), GpuError> {
            let (c0, c1) = (step.weight, step.correction());
            let (n, storage) = (mean.len() as u64, mean.storage());
            // Every entry is in the masters' storage (`input` and `output` refuse another).
            let f = self.posterior_kernel("posterior_ivon", storage)?;
            let inputs = u32::from(gradient.is_some()) | (u32::from(factor.is_some()) << 1) | (u32::from(prior.is_some()) << 2);
            let count = variance.len() as u64;
            let (cols, axis, chunks) = (groups.cols as u64, groups.reduce_code(), groups.chunks() as u64);
            // An absent input goes as a null pointer, which the kernel does not read (`inputs`).
            let absent = 0u64;
            let mut builder = self.stream.launch_builder(&f);
            builder.arg(&n).arg(&cols).arg(&axis).arg(&chunks).arg(&count).arg(&step.gradient_scale).arg(&step.factor_scale).arg(&step.tokens).arg(&step.beta1).arg(&step.beta2).arg(&c0).arg(&c1).arg(&inputs);
            for input in [gradient, factor, prior] {
                match input {
                    Some(t) => {
                        builder.input(t, storage)?;
                    }
                    None => {
                        builder.arg(&absent);
                    }
                }
            }
            builder.arg(index_slice(&groups.layout)?).arg(slice(variance)?);
            builder.input(mean, storage)?.output(log_sd, storage)?.output(momentum, storage)?.output(curvature, storage)?.output(direction, storage)?.arg(slice_mut(sums)?);
            // SAFETY: equal-length entry buffers in the masters' storage, a null pointer only for
            // an input `inputs` marks absent, float64 group buffers of `count` rows (the sums five
            // columns), ids per the map's axis; ids at or beyond `count` are not stepped.
            unsafe { builder.launch(cfg_elements(groups.blocks() as u64)) }.gpu_ctx("tensor posterior_ivon").map(|_| ())
        }

        pub(super) fn posterior_finish(
            &self,
            (mean, direction, eta): (&mut Tensor, &Tensor, f64),
            (average, weight): (&mut Tensor, f64),
            log_sd: &Tensor,
            groups: &super::GroupMap,
            sums: &mut Tensor,
        ) -> Result<(), GpuError> {
            let (n, storage) = (mean.len() as u64, mean.storage());
            let (f, count) = (self.posterior_kernel("posterior_finish", storage)?, sums.rows as u64);
            let (cols, axis, chunks) = (groups.cols as u64, groups.reduce_code(), groups.chunks() as u64);
            let mut builder = self.stream.launch_builder(&f);
            builder.arg(&n).arg(&cols).arg(&axis).arg(&chunks).arg(&count).arg(&eta).arg(&weight).input(direction, storage)?.input(log_sd, storage)?.arg(index_slice(&groups.layout)?);
            builder.output(mean, storage)?.output(average, storage)?.arg(slice_mut(sums)?);
            // SAFETY: equal-length entry buffers in one storage, checked by the caller and `input`;
            // float64 sums of `count` rows.
            unsafe { builder.launch(cfg_elements(groups.blocks() as u64)) }.gpu_ctx("tensor posterior_finish").map(|_| ())
        }

        pub(super) fn group_moments(&self, (mean, log_sd): (&Tensor, &Tensor), groups: &super::GroupMap, sums: &mut Tensor) -> Result<(), GpuError> {
            let (n, storage) = (mean.len() as u64, mean.storage());
            let (f, count) = (self.posterior_kernel("group_moments", storage)?, sums.rows as u64);
            let (cols, axis, chunks) = (groups.cols as u64, groups.reduce_code(), groups.chunks() as u64);
            let mut builder = self.stream.launch_builder(&f);
            builder.arg(&n).arg(&cols).arg(&axis).arg(&chunks).arg(&count).input(mean, storage)?.input(log_sd, storage)?.arg(index_slice(&groups.layout)?).arg(slice_mut(sums)?);
            // SAFETY: equal-length entry buffers, checked by the caller; float64 sums of `count` rows.
            unsafe { builder.launch(cfg_elements(groups.blocks() as u64)) }.gpu_ctx("tensor group_moments").map(|_| ())
        }

        pub(super) fn group_curvature(&self, (factor, mean, log_sd): (&Tensor, &Tensor, &Tensor), groups: &super::GroupMap, sums: &mut Tensor) -> Result<(), GpuError> {
            let (n, storage, factors) = (mean.len() as u64, mean.storage(), factor.storage());
            let f = match (storage, factors) {
                (Storage::F32, Storage::Bf16) => self.function("group_curvature_f32_bf16")?,
                (masters, u) if masters == u => self.posterior_kernel("group_curvature", storage)?,
                (masters, u) => return Err(shape(format!("{masters:?} masters with a {u:?} Gauss–Newton factor"))),
            };
            let count = sums.rows as u64;
            let (cols, axis, chunks) = (groups.cols as u64, groups.reduce_code(), groups.chunks() as u64);
            let mut builder = self.stream.launch_builder(&f);
            builder.arg(&n).arg(&cols).arg(&axis).arg(&chunks).arg(&count).input(factor, factors)?.input(mean, storage)?.input(log_sd, storage)?.arg(index_slice(&groups.layout)?).arg(slice_mut(sums)?);
            // SAFETY: equal-length entry buffers, checked by the caller; float64 sums of `count` rows.
            unsafe { builder.launch(cfg_elements(groups.blocks() as u64)) }.gpu_ctx("tensor group_curvature").map(|_| ())
        }

        pub(super) fn group_code_length(&self, divergence: &Tensor, (weight, constant): (&Tensor, &Tensor), sums: &mut Tensor, slot: usize) -> Result<(), GpuError> {
            let (n, slot) = (divergence.rows as u64, slot as u64);
            let blocks = n.div_ceil(u64::from(BLOCK)).max(1);
            // SAFETY: the first stage writes all `2 · blocks` partials before the second reads them.
            let mut partials = unsafe { self.alloc::<f64>(2 * blocks as usize) }.gpu_ctx("tensor alloc")?;
            let f = self.function("group_code_length")?;
            let mut builder = self.stream.launch_builder(&f);
            builder.arg(&n).arg(&slot).arg(slice(divergence)?).arg(slice(weight)?).arg(slice(constant)?).arg(&mut partials);
            // SAFETY: a groups × 2 and two groups × 1 float64 inputs; one block per BLOCK groups,
            // each writing its two partials.
            unsafe { builder.launch(LaunchConfig { grid_dim: (blocks as u32, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 }) }.gpu_ctx("tensor group_code_length")?;
            let total = self.function("group_code_length_total")?;
            // SAFETY: `2 · blocks` partials and a rows × 3 float64 sum, the slot inside it, checked
            // by the caller.
            unsafe { self.stream.launch_builder(&total).arg(&blocks).arg(&slot).arg(&partials).arg(slice_mut(sums)?).launch(cfg_elements(1)) }
                .gpu_ctx("tensor group_code_length_total")
                .map(|_| ())
        }

        pub(super) fn resolved_counts(
            &self,
            (value, slope, phi): (&Tensor, &Tensor, &Tensor),
            (noise_z, noise_y): (&Tensor, Option<&Tensor>),
            alive: &Indices,
            sums: &mut Tensor,
        ) -> Result<(), GpuError> {
            let (n, cols, storage) = (value.len() as u64, value.cols as u64, value.storage());
            let gated = i32::from(noise_y.is_some());
            let f = self.posterior_kernel("resolved_counts", storage)?;
            let mut builder = self.stream.launch_builder(&f);
            builder.arg(&n).arg(&cols).arg(&gated).input(value, storage)?.input(slope, storage)?.input(phi, storage)?.input(noise_z, storage)?;
            // Without a second input its noise is never read; the first stands in for it.
            builder.input(noise_y.unwrap_or(noise_z), storage)?.arg(index_slice(alive)?).arg(slice_mut(sums)?);
            // SAFETY: equal-length entry buffers and one flag per column, checked by the caller; a
            // float64 1 × 3 sum.
            unsafe { builder.launch(cfg_elements(n)) }.gpu_ctx("tensor resolved_counts").map(|_| ())
        }

        pub(super) fn group_divergence(&self, sums: &mut Tensor, reference: Option<&Tensor>, variance: &mut Tensor, divergence: &mut Tensor) -> Result<(), GpuError> {
            let (n, scaled) = (variance.len() as u64, i32::from(reference.is_some()));
            // Without references none is read; one zero stands in for them.
            let absent;
            let references = match reference {
                Some(t) => slice(t)?,
                None => {
                    absent = self.zeros(1)?;
                    &absent
                }
            };
            let f = self.function("group_divergence")?;
            let mut builder = self.stream.launch_builder(&f);
            builder.arg(&n).arg(&scaled).arg(references).arg(slice_mut(sums)?).arg(slice_mut(variance)?).arg(slice_mut(divergence)?);
            // SAFETY: groups × 3 sums, groups × 1 variances and references (read only when given),
            // groups × 2 divergences, checked by the caller.
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
            let mut order = self.alloc_zeros::<u32>(blocks * width).gpu_ctx("tensor alloc")?;
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
            outputs: (Option<&mut Tensor>, Option<(u64, &mut Tensor)>),
            settings: (usize, Arithmetic),
        ) -> Result<Vec<f64>, GpuError> {
            let mut out = self.zeros(hidden.rows)?;
            self.head_log_partition_into(hidden, head, scored, outputs, &mut out, settings)?;
            let mut values = self.download(&out)?;
            values.truncate(hidden.rows);
            Ok(values)
        }

        /// The swept log partitions into `out` (one double per row), left on the device: no
        /// host transfer, so a captured step can record it. With a probe (its key and its output,
        /// `Device::head_log_partition_probed`), each chunk's launch of `head_chunk` also writes the
        /// chunk's signed roots `exp((z − m) / 2) ξ`, which weight its head rows into the output
        /// (one more product, rescaled by `scale_rows` as the largest grows), and `head_probe`
        /// turns the output into the probe's pullback after the sweep.
        pub(super) fn head_log_partition_into(
            &self,
            hidden: &Tensor,
            (head, transposed): (&Tensor, bool),
            scored: Option<&Indices>,
            (mut expected, mut probe): (Option<&mut Tensor>, Option<(u64, &mut Tensor)>),
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
            // A probe's key, a chunk's signed roots per row and class, and per row their running sum
            // `a` (from zero) and its rescaling.
            let (key, probing) = (probe.as_ref().map_or(0, |(key, _)| *key), i32::from(probe.is_some()));
            let cells = if probe.is_some() && !half { rows * chunk } else { 1 };
            // SAFETY: with a probe, each chunk's `head_chunk` writes the roots of its rows × count
            // classes before the product reads them; without one, no kernel touches them.
            let mut roots = unsafe { self.alloc::<f32>(cells.max(1)) }.gpu_ctx("tensor alloc")?;
            let mut root_sums = self.zeros(if probe.is_some() { rows } else { 1 })?;
            // SAFETY: with a probe, each chunk's `head_chunk` writes every row's before `scale_rows`
            // reads it; without one, no kernel touches it.
            let mut root_factor = unsafe { self.alloc::<f32>(rows.max(1)) }.gpu_ctx("tensor alloc")?;
            // SAFETY: each chunk's product writes its logits whole (β = 0) before `head_chunk` reads
            // them, and `fill` writes every row's largest before any is read.
            let mut logits = unsafe { self.alloc::<f32>((rows * chunk).max(1)) }.gpu_ctx("tensor alloc")?;
            // SAFETY: as `logits`.
            let mut largest = unsafe { self.alloc::<f32>(rows.max(1)) }.gpu_ctx("tensor alloc")?;
            let (n_rows, lowest) = (rows as u64, f64::NEG_INFINITY);
            let fill = self.kernel("fill", Storage::F32)?;
            // SAFETY: `fill(n, v, x)` writes n floats.
            unsafe { self.stream.launch_builder(&fill).arg(&n_rows).arg(&lowest).arg(&mut largest).launch(cfg_elements(n_rows)) }.gpu_ctx("tensor fill")?;
            let mut sums = self.zeros(rows)?;
            // SAFETY: each chunk's `head_chunk` writes every row's factor before `scale_rows` reads it.
            let mut factor = unsafe { self.alloc::<f32>(rows.max(1)) }.gpu_ctx("tensor alloc")?;
            let (rows32, width32) = (u32::try_from(rows).map_err(|_| shape("head rows exceed u32".to_string()))?, width as u32);
            let want = i32::from(expected.is_some());
            let (t, n_op) = (cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N);
            // In bfloat16, `head_chunk` writes each chunk's exponentials and roots rounded into these
            // (rows × count) in place of the f32 ones.
            let half32 = i32::from(half);
            let half_cells = |on: bool| if half && on { rows * chunk } else { 1 };
            // SAFETY: with `half`, each chunk's `head_chunk` writes the rows × count
            // exponentials (with `expected`) and roots (with a probe) before a product reads them;
            // otherwise no kernel touches them.
            let mut half_e = unsafe { self.alloc::<u16>(half_cells(expected.is_some())) }.gpu_ctx("tensor bf16 alloc")?;
            // SAFETY: as `half_e`.
            let mut half_roots = unsafe { self.alloc::<u16>(half_cells(probe.is_some())) }.gpu_ctx("tensor bf16 alloc")?;
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
                let (count32, start32) = (count as u32, start as u32);
                // SAFETY: one block per row of the rows × count logits; per-row state of length rows;
                // with a probe, rows × count roots.
                unsafe {
                    self.stream.launch_builder(&chunk_kernel).arg(&rows32).arg(&count32).arg(&mut logits).arg(&mut largest)
                        .arg(&mut sums).arg(&mut factor).arg(&want).arg(&key).arg(&start32).arg(&probing)
                        .arg(&mut roots).arg(&mut root_sums).arg(&mut root_factor).arg(&half32).arg(&mut half_e).arg(&mut half_roots)
                        .launch(cfg_rows(rows))
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
                    let weights = if half { Factor::Half(&half_e) } else { Factor::Single(&logits) };
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
                if let Some((_, probed)) = probe.as_mut() {
                    let probed: &mut Tensor = probed;
                    let n = probed.len() as u64;
                    if index > 0 && n > 0 {
                        // SAFETY: `probed` is rows × width; `root_factor` holds one value per row.
                        unsafe { self.stream.launch_builder(&rescale).arg(&n).arg(&width32).arg(&root_factor).output(probed, Storage::F32)?.launch(cfg_elements(n)) }
                            .gpu_ctx("tensor scale_rows")?;
                    }
                    // Column-major probedᵀ (width × rows) += E_chunkᵀ (width × count) · Wᵀ (count ×
                    // rows), W the chunk's signed roots (rounded to bfloat16 as the exponentials are).
                    let (head_op, head_ld) = if transposed { (t, classes) } else { (n_op, width) };
                    let weights = if half { Factor::Half(&half_roots) } else { Factor::Single(&roots) };
                    self.gemm_ex(
                        Gemm32 {
                            ops: (head_op, n_op),
                            dims: (width, rows, count),
                            scale: (1.0, if index == 0 { 0.0 } else { 1.0 }),
                            a: (head_factor, if transposed { start } else { start * width }, head_ld),
                            b: (weights, 0, count),
                            c: (slice32_mut(probed)?, 0, width),
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
                    .arg(&want).arg(&mut *mean).arg(out).launch(cfg_rows(rows))
            }
            .gpu_ctx("tensor head_finish")?;
            if let Some((_, probed)) = probe {
                let finish_probe = self.kernel("head_probe", Storage::F32)?;
                // SAFETY: per-row sums of length rows, written by the sweep; `mean` (rows × width
                // with a probe, which takes `expected`) and `probed` are rows × width, f32.
                unsafe {
                    self.stream.launch_builder(&finish_probe).arg(&rows32).arg(&width32).arg(&sums).arg(&root_sums).arg(flags).arg(&use_flags)
                        .arg(&*mean).arg(slice32_mut(probed)?).launch(cfg_rows(rows))
                }
                .gpu_ctx("tensor head_probe")?;
            }
            Ok(())
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

// Per column (n = cols) of x, 1 when one of its rows `rows` (p.rows of them) holds a value other
// than zero (a NaN counts).
kernel void t_nonzero_columns(device const float* x [[buffer(0)]], device const uint* rows [[buffer(1)]], device float* out [[buffer(2)]],
                              constant P& p [[buffer(3)]], uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS {
        float any = 0.0f;
        for (uint t = 0; t < p.rows; ++t) {
            if (x[rows[t] * p.n + i] != 0.0f) { any = 1.0f; break; }
        }
        out[i] = any;
    }
}

// out[r, c] = Σ_t x[r, t] a[c, t] at each of the p.n listed entries (c = columns[e], r = row_of[e]);
// p.cols = k, p.extra = out's columns.
kernel void t_sampled_product(device const float* x [[buffer(0)]], device const float* a [[buffer(1)]], device const uint* columns [[buffer(2)]],
                              device const uint* row_of [[buffer(3)]], device float* out [[buffer(4)]], constant P& p [[buffer(5)]],
                              uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS {
        uint r = row_of[i], c = columns[i];
        float s = 0.0f;
        for (uint t = 0; t < p.cols; ++t) s += x[r * p.cols + t] * a[c * p.cols + t];
        out[r * p.extra + c] = s;
    }
}

// out[r, j] = Σ v[r, c] a[c, j] over row r's listed columns c, over the p.n = rows × m entries;
// p.cols = m, p.extra = v's columns.
kernel void t_listed_product(device const float* v [[buffer(0)]], device const float* a [[buffer(1)]], device const uint* offsets [[buffer(2)]],
                             device const uint* columns [[buffer(3)]], device float* out [[buffer(4)]], constant P& p [[buffer(5)]],
                             uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS {
        uint r = i / p.cols, j = i % p.cols;
        float s = 0.0f;
        for (uint e = offsets[r]; e < offsets[r + 1]; ++e) {
            uint c = columns[e];
            s += v[r * p.extra + c] * a[c * p.cols + j];
        }
        out[i] = s;
    }
}

// out[c, j] = Σ v[r, c] y[r, j] over the rows r listed for column c's group, over the p.n = n × m
// entries; p.cols = m, p.extra = v's columns.
kernel void t_listed_product_t(device const float* v [[buffer(0)]], device const float* y [[buffer(1)]], device const uint* offsets [[buffer(2)]],
                               device const uint* rows [[buffer(3)]], device const uint* group_of [[buffer(4)]], device float* out [[buffer(5)]],
                               constant P& p [[buffer(6)]], uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS {
        uint c = i / p.cols, j = i % p.cols, g = group_of[c];
        float s = 0.0f;
        for (uint e = offsets[g]; e < offsets[g + 1]; ++e) {
            uint r = rows[e];
            s += v[r * p.extra + c] * y[r * p.cols + j];
        }
        out[i] = s;
    }
}

// A block's input norm per row (p.n rows of p.cols, one thread a row, sums in index order from −0):
// p.a the mode (0 N(y), 1 its tangent along t, 2 its pullback of t), p.b whether β is read,
// p.alpha ε.
kernel void t_row_norm(device const float* y [[buffer(0)]], device const float* t [[buffer(1)]], device const float* g [[buffer(2)]],
                       device const float* b [[buffer(3)]], device float* out [[buffer(4)]], constant P& p [[buffer(5)]],
                       uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS {
        uint d = p.cols;
        device const float* yr = y + i * d;
        device const float* tr = t + i * d;
        device float* o = out + i * d;
        float sq = -0.0f;
        for (uint k = 0; k < d; ++k) sq = sq + yr[k] * yr[k];
        float scale = 1.0f / sqrt(sq / float(d) + p.alpha);
        if (p.a == 0) {
            for (uint k = 0; k < d; ++k) o[k] = g[k] * yr[k] * scale + (p.b ? b[k] : 0.0f);
        } else if (p.a == 1) {
            float dot = -0.0f;
            for (uint k = 0; k < d; ++k) dot = dot + yr[k] * tr[k];
            float dr = -scale * scale * scale * dot / float(d);
            for (uint k = 0; k < d; ++k) o[k] = g[k] * (scale * tr[k] + yr[k] * dr);
        } else {
            float dot = -0.0f;
            for (uint k = 0; k < d; ++k) dot = dot + g[k] * tr[k] * yr[k];
            float along = dot * scale * scale * scale / float(d);
            for (uint k = 0; k < d; ++k) o[k] = scale * g[k] * tr[k] - along * yr[k];
        }
    }
}

// out = tᵀ over its p.n entries (t p.rows × p.cols).
kernel void t_transpose(device const float* t [[buffer(0)]], device float* out [[buffer(1)]], constant P& p [[buffer(2)]],
                        uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS out[i] = t[(i % p.rows) * p.cols + i / p.rows];
}

// out[r, j] = t[r, ids[j]] over the n = rows × m (p.extra) entries; p.cols = t's columns.
kernel void t_gather_columns(device const float* t [[buffer(0)]], device const uint* ids [[buffer(1)]], device float* out [[buffer(2)]],
                             constant P& p [[buffer(3)]], uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS out[i] = t[(i / p.extra) * p.cols + ids[i % p.extra]];
}

// t[ids[i], c] = values[i, c] (added when p.a), ids distinct; p.cols = t's columns.
kernel void t_scatter_rows(device float* t [[buffer(0)]], device const uint* ids [[buffer(1)]], device const float* values [[buffer(2)]],
                           constant P& p [[buffer(3)]], uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS {
        uint at = ids[i / p.cols] * p.cols + i % p.cols;
        t[at] = p.a ? t[at] + values[i] : values[i];
    }
}

// t[r, ids[j]] = values[r, j] (added when p.a), ids distinct.
kernel void t_scatter_columns(device float* t [[buffer(0)]], device const uint* ids [[buffer(1)]], device const float* values [[buffer(2)]],
                              constant P& p [[buffer(3)]], uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS {
        uint at = (i / p.extra) * p.cols + ids[i % p.extra];
        t[at] = p.a ? t[at] + values[i] : values[i];
    }
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
        case 6: return log(fmax(t, 1.17549435e-38f));
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
        case 6: return t > 1.17549435e-38f ? 1.0f / t : 0.0f;
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

// The sign of class `k` in row `row` of the Fisher probe under `key` (`fisher_sign` on the host;
// rows and classes fit 32 bits here): the top bit of the first word of Philox4x32-10 of the
// counter (k, row), +1 when it is clear.
inline float fisher_sign(uint2 key, uint row, uint k) {
    uint c0 = k, c1 = 0u, c2 = row, c3 = 0u, k0 = key.x, k1 = key.y;
    for (int round = 0; round < 10; ++round) {
        if (round > 0) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
        uint hi0 = mulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
        uint hi1 = mulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
        c0 = hi1 ^ c1 ^ k0; c1 = lo1; c2 = hi0 ^ c3 ^ k1; c3 = lo0;
    }
    return (c0 >> 31) != 0u ? -1.0f : 1.0f;
}

// In place of row r's probabilities π, the Fisher probe's cotangent b = √π ⊙ ξ − π (√π · ξ)
// (`Device::fisher_probe_cotangent`), ξ the signs of row `extra + r` under the key (c, d) (its low
// and high words); a: rows flagged by `scored`, a row whose flag is zero becoming zero.
kernel void t_fisher_probe(device float* probabilities [[buffer(0)]], device const uint* scored [[buffer(1)]],
                           constant P& p [[buffer(2)]], uint group [[threadgroup_position_in_grid]],
                           uint groups [[threadgroups_per_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float shared[GROUP];
    uint2 key = uint2(p.c, p.d);
    ROWS {
        device float* q = probabilities + (ulong)r * p.cols;
        if (p.a != 0 && scored[r] == 0) {
            for (uint c = t; c < p.cols; c += GROUP) q[c] = 0.0f;
            continue;
        }
        float partial = 0.0f;
        for (uint c = t; c < p.cols; c += GROUP) partial += sqrt(q[c]) * fisher_sign(key, p.extra + r, c);
        float dot = group_sum(partial, shared, t);
        for (uint c = t; c < p.cols; c += GROUP) q[c] = sqrt(q[c]) * fisher_sign(key, p.extra + r, c) - q[c] * dot;
    }
}

// n: rows · blocks; extra: blocks.
// A gate's entrywise maps (`GateFunction`, p.a the code): √x, H(x), Φ(z), φ(z)/s, −φ(z) z/s with
// z = x/s (s read only when p.b), z held to ±40: Φ and φ there are their float32 limits (Φ(±40) is
// 1 or 0, φ(40) is 0), and the fast-math shader made NaN of a finite z near 10²⁹ (s = 10⁻³⁰).
kernel void t_gate_function(device const float* x [[buffer(0)]], device const float* s [[buffer(1)]], device float* out [[buffer(2)]],
                            constant P& p [[buffer(3)]], uint gid [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    ELEMENTS {
        float t = x[i], sd = p.b ? s[i] : 1.0f, z = fmin(fmax(t / sd, -40.0f), 40.0f), density = exp(-0.5f * z * z) * 0.39894228040143268f;
        float v;
        switch (p.a) {
            case 0: v = sqrt(t); break;
            case 1: v = t > 0.0f ? 1.0f : 0.0f; break;
            case 2: v = 0.5f * gam_erfc(-z * 0.70710678118654752f); break;
            case 3: v = density / sd; break;
            case 4: v = -density * z / sd; break;
            case 6: v = exp(2.0f * t); break;
            case 7: v = z > 0.0f ? fmin(z, 1.0f) : 0.0f; break;
            case 8: v = (z > 0.0f && z < 1.0f) ? 1.0f / sd : 0.0f; break;
            case 9: v = (z > 0.0f && z < 1.0f) ? -z / sd : 0.0f; break;
            default: v = sd == 0.0f ? 0.0f : t / sd; break;
        }
        out[i] = v;
    }
}

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

// The standard normal draw `index` of (key, stream): Philox4x32-10, then Box–Muller
// (`posterior_normal` on the host; indices fit 32 bits here), negated for a key with its top
// (antithetic) bit set.
inline float posterior_normal(uint2 key, uint2 stream, uint index) {
    uint c0 = index, c1 = 0u, c2 = stream.x, c3 = stream.y, k0 = key.x, k1 = key.y & 0x7FFFFFFFu;
    float sign = (key.y >> 31) != 0u ? -1.0f : 1.0f;
    for (int round = 0; round < 10; ++round) {
        if (round > 0) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
        uint hi0 = mulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
        uint hi1 = mulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
        c0 = hi1 ^ c1 ^ k0; c1 = lo1; c2 = hi0 ^ c3 ^ k1; c3 = lo0;
    }
    float u1 = (float(c0) + 0.5f) * 2.3283064e-10f, u2 = float(c1) * 2.3283064e-10f;
    return sign * (sqrt(-2.0f * log(u1)) * cos(6.28318530717958647692f * u2));
}

// The parameters of the posterior kernels (Rust `Posterior`).
struct Posterior { uint n; uint count; uint2 key; uint2 stream; float scale; float tokens; float beta1; float beta2; float c1; float fscale; uint axis; uint cols; uint chunks; float c0; uint stride; uint at; uint inputs; };

// Entry i's group under an operator's group map: its row's id (axis 0), its column's (1), its own (2).
inline uint group_of(device const uint* ids, constant Posterior& p, uint i) {
    return p.axis == 0u ? ids[i / p.cols] : (p.axis == 1u ? ids[i % p.cols] : ids[i]);
}

// Segment s of an operator's group map (`GroupMap`'s layout: `p.chunks + 1` offsets, the
// segments' groups, then their members), one per SIMD group: its group, the start `off` and count
// `size` of its members, and its entries' count; entry k of the segment is `segment_entry`.
inline uint segment_length(constant Posterior& p, uint size) {
    return p.axis == 0u ? size * p.cols : (p.axis == 1u ? size * (p.n / p.cols) : size);
}

inline uint segment_entry(device const uint* layout, constant Posterior& p, uint off, uint size, uint k) {
    device const uint* members = layout + 2u * p.chunks + 1u + off;
    return p.axis == 0u ? members[k / p.cols] * p.cols + k % p.cols : (p.axis == 1u ? (k / size) * p.cols + members[k % size] : members[k]);
}

// A segment's lanes' sums added across its SIMD group, and the totals added to its group's row by
// the first lane: a group is one segment and dispatches run in order, so nothing is atomic and the
// sums are the same on every run.
inline void segment_add(device float* sums, uint g, uint lane, float a, float b, float c) {
    a = simd_sum(a); b = simd_sum(b); c = simd_sum(c);
    if (lane == 0u && a > 0.0f) {
        sums[3 * g] += a; sums[3 * g + 1] += b; sums[3 * g + 2] += c;
    }
}

// `Device::round_to_deviation` in f32.
kernel void t_round_to_deviation(device const float* mean [[buffer(0)]], device const float* log_sd [[buffer(1)]], device float* rounded [[buffer(2)]],
                                 constant Posterior& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    float mu = mean[i], s = log_sd[i];
    if (isfinite(s)) {
        float step = ldexp(1.0f, int(floor(s / 0.6931471805599453f)));
        mu = round(mu / step) * step;
    }
    rounded[i] = mu;
}

// The sample of an operator of `p.cols` columns into the block of `theta` from entry `p.at`, its
// rows `p.stride` apart.
kernel void t_reparameterize(device const float* mean [[buffer(0)]], device const float* log_sd [[buffer(1)]], device float* theta [[buffer(2)]],
                             constant Posterior& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) theta[p.at + (i / p.cols) * p.stride + i % p.cols] = mean[i] + exp(log_sd[i]) * posterior_normal(p.key, p.stream, i);
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

// IVON's step (`Device::posterior_ivon`), one SIMD group per segment of the group map, each lane
// stepping every 32nd of its entries: the mean stays, its full step d = G / (h₀⁺ + δ) (h₀ the
// curvature before this step's draw) goes to `direction`, and each live entry's (d d, u d,
// (h₀⁺ d) d, (g + δ μ) d₀, ((h₀⁺ + δ) d₀) d₀) into its group's row of `sums` (groups × 5) by the
// segment's first lane, d₀ the direction before the step.
// `p.inputs` holds 1 with a gradient, 2 with a factor and 4 with a prior curvature; an absent one
// is zero (its slot bound to another buffer, not read). `p.c0` and `p.c1` are the momentum's bias
// corrections before and after the step. An entry the step leaves (removed, or of a group at or
// beyond `p.count`) has d = μ − μ.
kernel void t_posterior_ivon(device const float* gradient [[buffer(0)]], device const uint* layout [[buffer(1)]], device const float* variance [[buffer(2)]],
                             device const float* mean [[buffer(3)]], device float* log_sd [[buffer(4)]], device float* momentum [[buffer(5)]], device float* curvature [[buffer(6)]],
                             device float* sums [[buffer(7)]], device const float* factor [[buffer(8)]], device const float* prior [[buffer(9)]], device float* direction [[buffer(10)]],
                             constant Posterior& p [[buffer(11)]], uint i [[thread_position_in_grid]]) {
    uint s = i / 32u, lane = i % 32u;
    if (s >= p.chunks) return;
    uint g = layout[p.chunks + 1u + s];
    bool stepped = g < p.count;
    bool given = (p.inputs & 1u) != 0u, drawn = (p.inputs & 2u) != 0u, priced = (p.inputs & 4u) != 0u;
    uint off = layout[s], size = layout[s + 1u] - off, length = segment_length(p, size);
    float delta = stepped ? 1.0f / (p.tokens * variance[g]) : 0.0f;
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f, t4 = 0.0f;
    for (uint k = lane; k < length; k += 32u) {
        uint e = segment_entry(layout, p, off, size, k);
        float mu = mean[e];
        if (!stepped || log_sd[e] == -INFINITY) {
            direction[e] = mu - mu;
            continue;
        }
        // The direction before the step, from the momentum and the curvature it holds.
        float m = momentum[e], h = curvature[e];
        float bounded = max(h, 0.0f), held = bounded + delta;
        float previous = ((p.c0 > 0.0f ? m / p.c0 : 0.0f) + delta * mu) / held;
        float gi = given ? p.scale * gradient[e] : 0.0f, ui = drawn ? factor[e] : 0.0f;
        float estimate = p.fscale * ui * ui + (priced ? prior[e] : 0.0f);
        float m1 = p.beta1 * m + (1.0f - p.beta1) * gi;
        float h1 = h + (1.0f - p.beta2) * (estimate - h);
        float positive = max(h1, 0.0f);
        float d = (m1 / p.c1 + delta * mu) / held;
        float sd = -0.5f * log(p.tokens * (positive + delta));
        momentum[e] = m1; curvature[e] = h1; log_sd[e] = sd; direction[e] = d;
        float weighted = bounded * d, fresh = gi + delta * mu, own = held * previous;
        t0 += d * d; t1 += ui * d; t2 += weighted * d; t3 += fresh * previous; t4 += own * previous;
    }
    t0 = simd_sum(t0); t1 = simd_sum(t1); t2 = simd_sum(t2); t3 = simd_sum(t3); t4 = simd_sum(t4);
    if (stepped && lane == 0u) {
        sums[5 * g] += t0; sums[5 * g + 1] += t1; sums[5 * g + 2] += t2; sums[5 * g + 3] += t3; sums[5 * g + 4] += t4;
    }
}

kernel void t_group_moments(device const float* mean [[buffer(0)]], device const float* log_sd [[buffer(1)]], device const uint* layout [[buffer(2)]],
                            device float* sums [[buffer(3)]], constant Posterior& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    uint s = i / 32u, lane = i % 32u;
    if (s >= p.chunks) return;
    uint g = layout[p.chunks + 1u + s];
    if (g >= p.count) return;
    uint off = layout[s], size = layout[s + 1u] - off, length = segment_length(p, size);
    float a = 0.0f, b = 0.0f, c = 0.0f;
    for (uint k = lane; k < length; k += 32u) {
        uint e = segment_entry(layout, p, off, size, k);
        if (log_sd[e] == -INFINITY) continue;
        a += 1.0f; b += mean[e] * mean[e] + exp(2.0f * log_sd[e]); c += 2.0f * log_sd[e];
    }
    segment_add(sums, g, lane, a, b, c);
}

kernel void t_group_curvature(device const float* factor [[buffer(0)]], device const float* mean [[buffer(1)]], device const float* log_sd [[buffer(2)]],
                              device const uint* layout [[buffer(3)]], device float* sums [[buffer(4)]], constant Posterior& p [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    uint s = i / 32u, lane = i % 32u;
    if (s >= p.chunks) return;
    uint g = layout[p.chunks + 1u + s];
    if (g >= p.count) return;
    uint off = layout[s], size = layout[s + 1u] - off, length = segment_length(p, size);
    float a = 0.0f, b = 0.0f, c = 0.0f;
    for (uint k = lane; k < length; k += 32u) {
        uint e = segment_entry(layout, p, off, size, k);
        if (log_sd[e] == -INFINITY) continue;
        a += 1.0f; b += factor[e] * mean[e]; c += factor[e] * factor[e] * exp(2.0f * log_sd[e]);
    }
    segment_add(sums, g, lane, a, b, c);
}

// `Device::group_code_length`, every group into row `p.count`, summed by the first SIMD group alone
// in a fixed order (lane by lane, then across the lanes), so the sum is the same on every run.
kernel void t_group_code_length(device const float* divergence [[buffer(0)]], device const float* weight [[buffer(1)]], device const float* constants [[buffer(2)]],
                                device float* sums [[buffer(3)]], constant Posterior& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= 32u) return;
    float a = 0.0f, total = 0.0f;
    for (uint g = i; g < p.n; g += 32u) {
        if (weight[g] == 0.0f) continue;
        a += 1.0f;
        total += weight[g] * (divergence[2u * g] + constants[g] + 0.6931471805599453f * divergence[2u * g + 1u]);
    }
    a = simd_sum(a); total = simd_sum(total);
    if (i == 0u && a > 0.0f) {
        sums[3 * p.count] += a; sums[3 * p.count + 1] += total;
    }
}

// `Device::resolved_counts` in f32, one thread per entry; `p.count` columns, `p.key.x` nonzero
// when there is a second input.
kernel void t_resolved_counts(device const float* value [[buffer(0)]], device const float* slope [[buffer(1)]], device const float* phi [[buffer(2)]],
                              device const float* noise_z [[buffer(3)]], device const float* noise_y [[buffer(4)]], device const uint* alive [[buffer(5)]],
                              device float* sums [[buffer(6)]], constant Posterior& p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    bool live = i < p.n && alive[i % p.count] != 0u;
    float a = 0.0f, b = 0.0f, c = 0.0f;
    if (live) {
        float v = value[i], s = slope[i];
        float noise = s * s * max(noise_z[i], 0.0f);
        if (p.key.x != 0u) noise += phi[i] * phi[i] * max(noise_y[i], 0.0f);
        a = 1.0f; b = v != 0.0f ? 1.0f : 0.0f; c = fabs(v) > sqrt(noise) ? 1.0f : 0.0f;
    }
    group_add(sums, 0u, live, a, b, c);
}

// The bits of the Elias δ codeword of the signed index of `k` (`gam_gpu::tensor::scale_code_bits`).
inline float scale_code_bits(int k) {
    uint value = ((uint(k) << 1) ^ uint(k >> 31)) + 1u;
    uint low = 31u - clz(value);
    uint prefix = 31u - clz(low + 1u);
    return float(low + 2u * prefix + 1u);
}

// `Device::group_divergence` in f32: each group's prior variance into `variance`, its divergence
// and its scale's bits into `divergence[2i]` and `divergence[2i + 1]`, as
// `gam_gpu::tensor::group_prior` makes them (`p.a` nonzero when `references` holds the groups'
// reference variances), then its sums zeroed.
kernel void t_group_divergence(device float* sums [[buffer(0)]], device const float* references [[buffer(1)]], device float* variance [[buffer(2)]],
                               device float* divergence [[buffer(3)]], constant P& p [[buffer(4)]], uint gid [[thread_position_in_grid]],
                               uint grid [[threads_per_grid]]) {
    ELEMENTS {
        float count = sums[3 * i], second = sums[3 * i + 1], log_variance = sums[3 * i + 2];
        float v = 0.0f, kl = 0.0f, bits = 0.0f;
        if (count > 0.0f) {
            float centre = second / count;
            v = centre;
            kl = 0.5f * (count * log(centre) - log_variance);
            if (p.a != 0u) {
                float anchor = log2(references[i]);
                float at = log2(centre) - anchor;
                float first = round(at);
                if (!(centre > 0.0f && references[i] > 0.0f) || !isfinite(first)) {
                    bits = INFINITY;
                } else {
                    int k = int(first);
                    float rise = 0.0f, shortest = scale_code_bits(0);
                    bits = scale_code_bits(k);
                    while (k != 0) {
                        int toward = k > 0 ? 1 : -1;
                        float edge = float(k) - 0.5f * float(toward);
                        float w = 0.6931471805599453f * (edge - at);
                        float up = 0.5f * count * max(exp(-w) - 1.0f + w, 0.0f);
                        if (up + 0.6931471805599453f * shortest >= rise + 0.6931471805599453f * bits) break;
                        k -= toward;
                        float code = scale_code_bits(k);
                        if (up + 0.6931471805599453f * code < rise + 0.6931471805599453f * bits) {
                            v = exp2(anchor + edge);
                            rise = up;
                            bits = code;
                        }
                    }
                    kl += rise;
                }
            }
        }
        variance[i] = v;
        divergence[2 * i] = kl;
        divergence[2 * i + 1] = bits;
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
        "t_nonzero_columns",
        "t_gather_columns",
        "t_scatter_columns",
        "t_scatter_rows",
        "t_laws",
        "t_rms",
        "t_rotate",
        "t_heads",
        "t_softmax_rows",
        "t_softmax_backward",
        "t_kl_rows",
        "t_fisher_probe",
        "t_block_products",
        "t_gate_function",
        "t_sampled_product",
        "t_listed_product",
        "t_transpose",
        "t_listed_product_t",
        "t_row_norm",
        "t_softmax_quadratic",
        "t_argmax_rows",
        "t_fill_entries",
        "t_adam",
        "t_softmax_stats",
        "t_round_to_deviation",
        "t_reparameterize",
        "t_posterior_ivon",
        "t_group_moments",
        "t_group_curvature",
        "t_group_code_length",
        "t_resolved_counts",
        "t_group_divergence",
        "t_select_sets",
        "t_box_charge",
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

    /// The parameters of the posterior kernels (MSL `Posterior`, whose `uint2` aligns it to 8
    /// bytes: 80 bytes on both sides).
    #[repr(C, align(8))]
    #[derive(Clone, Copy, Default)]
    struct Posterior {
        n: u32,
        count: u32,
        key: [u32; 2],
        stream: [u32; 2],
        scale: f32,
        tokens: f32,
        beta1: f32,
        beta2: f32,
        /// The momentum's bias correction after the step (`t_posterior_ivon`).
        c1: f32,
        fscale: f32,
        /// The group map's axis code, columns and row chunks (`GroupMap`).
        axis: u32,
        cols: u32,
        chunks: u32,
        /// The momentum's bias correction before the step (`t_posterior_ivon`).
        c0: f32,
        /// A sampled block's row stride and first entry (`t_reparameterize`; its columns are
        /// `cols`).
        stride: u32,
        at: u32,
        /// The step's inputs present (`t_posterior_ivon`): 1 the gradient, 2 the Gauss–Newton
        /// factor, 4 the prior curvature.
        inputs: u32,
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

        pub(super) fn sampled_product(&self, x: &Tensor, a: &Tensor, lists: &super::RowLists, out: &mut Tensor) -> Result<(), GpuError> {
            if lists.is_empty() {
                return Ok(());
            }
            u32_of(x.len().max(a.len()).max(x.rows.saturating_mul(a.rows)))?;
            let p = P { cols: u32_of(x.cols)?, extra: u32_of(a.rows)?, ..P::default() };
            self.elements("t_sampled_product", &[whole(buffer(x)?), whole(buffer(a)?), whole(index_buffer(&lists.columns)?), whole(index_buffer(&lists.row_of)?), whole(buffer(out)?)], lists.len(), p)
        }

        pub(super) fn listed_product(&self, v: &Tensor, lists: &super::RowLists, a: &Tensor) -> Result<Tensor, GpuError> {
            let out = self.tensor(v.rows, a.cols)?;
            u32_of(v.len().max(a.len()))?;
            let p = P { cols: u32_of(a.cols)?, extra: u32_of(v.cols)?, ..P::default() };
            self.elements("t_listed_product", &[whole(buffer(v)?), whole(buffer(a)?), whole(index_buffer(&lists.offsets)?), whole(index_buffer(&lists.columns)?), whole(buffer(&out)?)], out.len(), p)?;
            Ok(out)
        }

        pub(super) fn listed_product_t(&self, v: &Tensor, lists: &super::RowLists, group_of: &Indices, y: &Tensor) -> Result<Tensor, GpuError> {
            let IndexData::Metal(_, groups) = &group_of.data else { return Err(foreign()) };
            if groups.iter().any(|&g| g as usize >= lists.rows) {
                return Err(shape(format!("a group of {} lists", lists.rows)));
            }
            let out = self.tensor(v.cols, y.cols)?;
            u32_of(v.len().max(y.len()).max(out.len()))?;
            let p = P { cols: u32_of(y.cols)?, extra: u32_of(v.cols)?, ..P::default() };
            self.elements("t_listed_product_t", &[whole(buffer(v)?), whole(buffer(y)?), whole(index_buffer(&lists.offsets)?), whole(index_buffer(&lists.columns)?), whole(index_buffer(group_of)?), whole(buffer(&out)?)], out.len(), p)?;
            Ok(out)
        }

        pub(super) fn row_norm(&self, mode: super::RowNorm, y: &Tensor, along: Option<&Tensor>, (gain, bias): (&Tensor, Option<&Tensor>), epsilon: f64) -> Result<Tensor, GpuError> {
            let out = self.tensor(y.rows, y.cols)?;
            u32_of(y.len())?;
            let p = P { cols: u32_of(y.cols)?, a: mode.code(), b: u32::from(bias.is_some()), alpha: epsilon as f32, ..P::default() };
            self.elements("t_row_norm", &[whole(buffer(y)?), whole(buffer(along.unwrap_or(y))?), whole(buffer(gain)?), whole(buffer(bias.unwrap_or(gain))?), whole(buffer(&out)?)], y.rows, p)?;
            Ok(out)
        }

        pub(super) fn transpose(&self, t: &Tensor) -> Result<Tensor, GpuError> {
            let out = self.tensor(t.cols, t.rows)?;
            let p = P { rows: u32_of(t.rows)?, cols: u32_of(t.cols)?, ..P::default() };
            self.elements("t_transpose", &[whole(buffer(t)?), whole(buffer(&out)?)], t.len(), p)?;
            Ok(out)
        }

        pub(super) fn nonzero_columns(&self, x: &Tensor, rows: &Indices) -> Result<Tensor, GpuError> {
            let out = self.tensor(1, x.cols)?;
            u32_of(x.len())?;
            let p = P { rows: u32_of(rows.len)?, ..P::default() };
            self.elements("t_nonzero_columns", &[whole(buffer(x)?), whole(index_buffer(rows)?), whole(buffer(&out)?)], x.cols, p)?;
            Ok(out)
        }

        pub(super) fn gather_columns(&self, t: &Tensor, ids: &Indices) -> Result<Tensor, GpuError> {
            let IndexData::Metal(_, values) = &ids.data else { return Err(foreign()) };
            if let Some(id) = values.iter().find(|id| **id as usize >= t.cols) {
                return Err(shape(format!("column {id} of a {}-column tensor", t.cols)));
            }
            let out = self.tensor(t.rows, ids.len)?;
            let p = P { cols: u32_of(t.cols)?, extra: u32_of(ids.len.max(1))?, ..P::default() };
            self.elements("t_gather_columns", &[whole(buffer(t)?), whole(index_buffer(ids)?), whole(buffer(&out)?)], out.len(), p)?;
            Ok(out)
        }

        pub(super) fn scatter_columns(&self, t: &mut Tensor, ids: &Indices, values: &Tensor, accumulate: bool) -> Result<(), GpuError> {
            let IndexData::Metal(_, ids_host) = &ids.data else { return Err(foreign()) };
            if let Some(id) = ids_host.iter().find(|id| **id as usize >= t.cols) {
                return Err(shape(format!("column {id} of a {}-column tensor", t.cols)));
            }
            u32_of(t.len())?;
            let p = P { cols: u32_of(t.cols)?, extra: u32_of(ids.len.max(1))?, a: u32::from(accumulate), ..P::default() };
            self.elements("t_scatter_columns", &[whole(buffer(t)?), whole(index_buffer(ids)?), whole(buffer(values)?)], values.len(), p)
        }

        pub(super) fn scatter_rows(&self, t: &mut Tensor, ids: &Indices, values: &Tensor, accumulate: bool) -> Result<(), GpuError> {
            let IndexData::Metal(_, ids_host) = &ids.data else { return Err(foreign()) };
            if let Some(id) = ids_host.iter().find(|id| **id as usize >= t.rows) {
                return Err(shape(format!("row {id} of a {}-row tensor", t.rows)));
            }
            u32_of(t.len())?;
            let p = P { cols: u32_of(t.cols)?, a: u32::from(accumulate), ..P::default() };
            self.elements("t_scatter_rows", &[whole(buffer(t)?), whole(index_buffer(ids)?), whole(buffer(values)?)], values.len(), p)
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

        pub(super) fn fisher_probe_cotangent(&self, probabilities: &mut Tensor, (key, first): (u64, usize), scored: Option<&Indices>) -> Result<(), GpuError> {
            let (flags, use_flags) = self.flags(scored)?;
            // The signs' rows and classes are 32-bit counters here.
            u32_of(first.saturating_add(probabilities.rows))?;
            let [low, high] = halves(key);
            let p = P { extra: u32_of(first)?, a: use_flags, c: low, d: high, ..P::default() };
            self.rows("t_fisher_probe", &[whole(buffer(probabilities)?), whole(flags)], probabilities.rows, probabilities.cols, p)
        }

        pub(super) fn block_products(&self, left: &Tensor, right: &Tensor, blocks: &ColumnBlocks) -> Result<Tensor, GpuError> {
            let out = self.tensor(left.rows, blocks.len())?;
            let p = P { cols: u32_of(left.cols)?, extra: u32_of(blocks.len())?, ..P::default() };
            let buffers = [whole(buffer(left)?), whole(buffer(right)?), whole(index_buffer(&blocks.offsets)?), whole(buffer(&out)?)];
            self.elements("t_block_products", &buffers, out.len(), p)?;
            Ok(out)
        }

        pub(super) fn gate_function(&self, function: super::GateFunction, x: &Tensor, s: Option<&Tensor>) -> Result<Tensor, GpuError> {
            let out = self.tensor(x.rows, x.cols)?;
            let p = P { a: function.code(), b: u32::from(s.is_some()), ..P::default() };
            self.elements("t_gate_function", &[whole(buffer(x)?), whole(buffer(s.unwrap_or(x))?), whole(buffer(&out)?)], x.len(), p)?;
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

        /// A posterior kernel over an operator's entries reduced into their groups (`map`): one
        /// thread per entry, or per column and row chunk for a column map.
        fn grouped(&self, kernel: &'static str, buffers: &[(&Buffer, usize)], map: &super::GroupMap, mut p: Posterior) -> Result<(), GpuError> {
            p.n = u32_of(map.rows * map.cols)?;
            (p.axis, p.cols, p.chunks) = (map.code(), u32_of(map.cols.max(1))?, u32_of(map.chunks())?);
            self.stream.dispatch(kernel, buffers, &p, spread(map.threads()))
        }

        pub(super) fn round_to_deviation(&self, out: &mut Tensor, (mean, log_sd): (&Tensor, &Tensor)) -> Result<(), GpuError> {
            self.posterior("t_round_to_deviation", &[whole(buffer(mean)?), whole(buffer(log_sd)?), whole(buffer(out)?)], out.len(), Posterior::default())
        }

        pub(super) fn reparameterize_block(&self, out: &mut Tensor, (at, stride): (usize, usize), (mean, log_sd): (&Tensor, &Tensor), (key, stream): (u64, u64)) -> Result<(), GpuError> {
            let p = Posterior { key: halves(key), stream: halves(stream), cols: u32_of(mean.cols)?, stride: u32_of(stride)?, at: u32_of(at)?, ..Posterior::default() };
            self.posterior("t_reparameterize", &[whole(buffer(mean)?), whole(buffer(log_sd)?), whole(buffer(out)?)], mean.len(), p)
        }

        pub(super) fn posterior_ivon(
            &self,
            (mean, log_sd): (&Tensor, &mut Tensor),
            [momentum, curvature]: [&mut Tensor; 2],
            (gradient, factor, prior): (Option<&Tensor>, Option<&Tensor>, Option<&Tensor>),
            (groups, variance): (&super::GroupMap, &Tensor),
            (direction, sums): (&mut Tensor, &mut Tensor),
            step: &super::PosteriorStep,
        ) -> Result<(), GpuError> {
            let p = Posterior {
                count: u32_of(variance.len())?,
                scale: step.gradient_scale as f32,
                fscale: step.factor_scale as f32,
                tokens: step.tokens as f32,
                beta1: step.beta1 as f32,
                beta2: step.beta2 as f32,
                c0: step.weight as f32,
                c1: step.correction() as f32,
                inputs: u32::from(gradient.is_some()) | (u32::from(factor.is_some()) << 1) | (u32::from(prior.is_some()) << 2),
                ..Posterior::default()
            };
            // An absent input's slot is bound to the mean, which the kernel does not read there.
            let buffers = [
                whole(buffer(gradient.unwrap_or(mean))?),
                whole(index_buffer(&groups.layout)?),
                whole(buffer(variance)?),
                whole(buffer(mean)?),
                whole(buffer(log_sd)?),
                whole(buffer(momentum)?),
                whole(buffer(curvature)?),
                whole(buffer(sums)?),
                whole(buffer(factor.unwrap_or(mean))?),
                whole(buffer(prior.unwrap_or(mean))?),
                whole(buffer(direction)?),
            ];
            self.grouped("t_posterior_ivon", &buffers, groups, p)
        }

        pub(super) fn group_moments(&self, (mean, log_sd): (&Tensor, &Tensor), groups: &super::GroupMap, sums: &mut Tensor) -> Result<(), GpuError> {
            let p = Posterior { count: u32_of(sums.rows)?, ..Posterior::default() };
            self.grouped("t_group_moments", &[whole(buffer(mean)?), whole(buffer(log_sd)?), whole(index_buffer(&groups.layout)?), whole(buffer(sums)?)], groups, p)
        }

        pub(super) fn group_curvature(&self, (factor, mean, log_sd): (&Tensor, &Tensor, &Tensor), groups: &super::GroupMap, sums: &mut Tensor) -> Result<(), GpuError> {
            let p = Posterior { count: u32_of(sums.rows)?, ..Posterior::default() };
            let buffers = [whole(buffer(factor)?), whole(buffer(mean)?), whole(buffer(log_sd)?), whole(index_buffer(&groups.layout)?), whole(buffer(sums)?)];
            self.grouped("t_group_curvature", &buffers, groups, p)
        }

        pub(super) fn group_code_length(&self, divergence: &Tensor, (weight, constant): (&Tensor, &Tensor), sums: &mut Tensor, slot: usize) -> Result<(), GpuError> {
            let p = Posterior { count: u32_of(slot)?, ..Posterior::default() };
            let buffers = [whole(buffer(divergence)?), whole(buffer(weight)?), whole(buffer(constant)?), whole(buffer(sums)?)];
            self.posterior("t_group_code_length", &buffers, divergence.rows, p)
        }

        pub(super) fn resolved_counts(
            &self,
            (value, slope, phi): (&Tensor, &Tensor, &Tensor),
            (noise_z, noise_y): (&Tensor, Option<&Tensor>),
            alive: &Indices,
            sums: &mut Tensor,
        ) -> Result<(), GpuError> {
            let p = Posterior { count: u32_of(value.cols)?, key: [u32::from(noise_y.is_some()), 0], ..Posterior::default() };
            let buffers = [
                whole(buffer(value)?),
                whole(buffer(slope)?),
                whole(buffer(phi)?),
                whole(buffer(noise_z)?),
                whole(buffer(noise_y.unwrap_or(noise_z))?),
                whole(index_buffer(alive)?),
                whole(buffer(sums)?),
            ];
            self.posterior("t_resolved_counts", &buffers, value.len(), p)
        }

        pub(super) fn group_divergence(&self, sums: &mut Tensor, reference: Option<&Tensor>, variance: &mut Tensor, divergence: &mut Tensor) -> Result<(), GpuError> {
            // Without references none is read; the variances' buffer stands in for them.
            let references = match reference {
                Some(t) => buffer(t)?,
                None => buffer(variance)?,
            };
            let p = P { a: u32::from(reference.is_some()), ..P::default() };
            let buffers = [whole(buffer(sums)?), whole(references), whole(buffer(variance)?), whole(buffer(divergence)?)];
            self.elements("t_group_divergence", &buffers, variance.len(), p)
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

        pub(super) fn box_charge(&self, z: &Tensor, mask: &Tensor, q: &Tensor, cot: &mut Tensor, coefficient: &mut Tensor) -> Result<Vec<f64>, GpuError> {
            let norms = self.stream.alloc(z.rows)?;
            let buffers = [whole(buffer(z)?), whole(buffer(mask)?), whole(buffer(q)?), whole(&norms), whole(buffer(cot)?), whole(buffer(coefficient)?)];
            self.rows("t_box_charge", &buffers, z.rows, z.cols, P::default())?;
            Ok(self.stream.read::<f32>(&norms)?.into_iter().take(z.rows).map(|n| 0.5 * f64::from(n) * f64::from(n)).collect())
        }
    }
}

#[cfg(test)]
mod gram_tests {
    use super::*;

    #[test]
    fn the_symmetric_update_sums_the_lower_triangle_and_keeps_the_upper() {
        let device = Device::host();
        let values: Vec<f64> = (0..15).map(|i| (i as f64 - 7.0) / 3.0).collect();
        let a = device.upload_vec(5, 3, values.clone()).unwrap();
        let mut c = device.upload_vec(3, 3, vec![7.0; 9]).unwrap();
        device.gram_lower(&mut c, &a, 0.5).unwrap();
        let c = device.download(&c).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i >= j { (0..5).map(|r| values[3 * r + i] * values[3 * r + j]).sum::<f64>() + 3.5 } else { 7.0 };
                assert_eq!(c[[i, j]], expected, "entry ({i}, {j})");
            }
        }
        assert!(device.gram_lower(&mut device.zeros(2, 2).unwrap(), &a, 0.0).is_err());
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

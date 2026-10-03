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
//! The third, [`Device::single_precision`] on macOS, is the Apple GPU, which has no float64: its
//! tensors hold f32, every operation runs in f32 (products on Metal Performance Shaders' GEMM,
//! the rest on MSL kernels compiled with the safe math mode and contraction off, `exp`, `log` and
//! `tanh` precise, `erfc` the rational approximation of relative error below `1.2·10⁻⁷`), and a
//! product asked for in float64 is refused ([`GpuError::NoDeviceKernel`]), so float64 work stays
//! on the host. [`Device::float64`] tells the two kinds apart. Its operations queue on one Metal
//! command stream and run when the host reads a result (`crate::metal::stream`).

use crate::gpu_error::GpuError;
use crate::GpuPolicy;
use ndarray::{Array2, ArrayView2, ArrayViewMut2, linalg::general_mat_mul};
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
}

impl Arithmetic {
    /// The unit roundoff of the operands' rounding.
    #[must_use]
    pub fn unit_roundoff(self) -> f64 {
        match self {
            Self::F64 => f64::EPSILON / 2.0,
            Self::F32 => f64::from(f32::EPSILON) / 2.0,
            Self::Tf32 => 2f64.powi(-11),
        }
    }
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

/// A dense row-major `rows × cols` float64 tensor on its device.
pub struct Tensor {
    rows: usize,
    cols: usize,
    data: Data,
}

enum Data {
    Host(Vec<f64>),
    #[cfg(target_os = "linux")]
    Cuda(cudarc::driver::CudaSlice<f64>),
    #[cfg(target_os = "macos")]
    Metal(crate::metal::stream::Buffer),
}

impl Tensor {
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

    /// The bytes it holds (four per value on the Apple GPU, eight elsewhere).
    #[must_use]
    pub fn bytes(&self) -> usize {
        match &self.data {
            #[cfg(target_os = "macos")]
            Data::Metal(_) => self.len() * 4,
            _ => self.len() * 8,
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

/// Where tensors live and run (module note). Cloning shares the device.
#[derive(Clone)]
pub struct Device {
    backend: Arc<Backend>,
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

fn same(a: &Tensor, b: &Tensor, what: &str) -> Result<(), GpuError> {
    if a.dim() != b.dim() {
        return Err(shape(format!("{what}: {:?} against {:?}", a.dim(), b.dim())));
    }
    Ok(())
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
    }
}

impl Device {
    /// The CPU reference backend.
    #[must_use]
    pub fn host() -> Self {
        Self { backend: Arc::new(Backend::Host) }
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
                .map(|device| Ok(Self { backend: Arc::new(Backend::Cuda(cuda::Engine::new(device.ordinal, device.name.clone())?)) }))
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
                Ok(engine) => Ok(Some(Self { backend: Arc::new(Backend::Cuda(engine)) })),
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

    /// The device `policy` selects for single-precision work (proposals, training products): the
    /// float64 accelerator when there is one, else on macOS the Apple GPU (module note: f32
    /// only), `None` under `off` or when neither exists.
    pub fn single_precision(policy: GpuPolicy) -> Result<Option<Self>, GpuError> {
        #[cfg(target_os = "macos")]
        {
            let Some(runtime) = crate::apple_gpu::MetalRuntime::resolve(policy)? else { return Ok(None) };
            Ok(Some(Self { backend: apple::Engine::shared(runtime)? }))
        }
        #[cfg(not(target_os = "macos"))]
        {
            Self::accelerator(policy)
        }
    }

    /// Whether every float64 operation runs in IEEE float64 here (the host and CUDA); the Apple
    /// GPU runs f32 and refuses float64 products.
    #[must_use]
    pub fn float64(&self) -> bool {
        match &*self.backend {
            #[cfg(target_os = "macos")]
            Backend::Metal(_) => false,
            _ => true,
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
            #[cfg(target_os = "macos")]
            Data::Metal(buffer) => match &*self.backend {
                Backend::Metal(engine) => engine.stream.read::<f32>(buffer)?.into_iter().take(t.len()).map(f64::from).collect(),
                _ => return Err(foreign()),
            },
        };
        Array2::from_shape_vec((t.rows, t.cols), flat).map_err(|e| shape(e.to_string()))
    }

    pub fn zeros(&self, rows: usize, cols: usize) -> Result<Tensor, GpuError> {
        let data = match &*self.backend {
            Backend::Host => Data::Host(vec![0.0; rows * cols]),
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
        self.softmax_rows_impl(scores, causal, 0)
    }

    /// Softmax of a rectangular query tile against a sequence's keys, preserving its causal
    /// offset. Noncausal rows can have any batch/vocabulary shape.
    pub fn softmax_rows_offset(&self, scores: &mut Tensor, causal: bool, start: usize) -> Result<(), GpuError> {
        if scores.cols == 0 || (causal && (start > scores.cols || scores.rows > scores.cols - start)) {
            return Err(shape("causal softmax tile outside its sequence".to_string()));
        }
        self.softmax_rows_impl(scores, causal, start)
    }

    fn softmax_rows_impl(&self, scores: &mut Tensor, causal: bool, start: usize) -> Result<(), GpuError> {
        let width = scores.cols;
        match &*self.backend {
            Backend::Host => {
                for (r, row) in host_mut(scores)?.chunks_mut(width).enumerate() {
                    let valid = if causal { start + r % width + 1 } else { width };
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
            Backend::Cuda(engine) => engine.softmax_rows(scores, causal, start),
            #[cfg(target_os = "macos")]
            Backend::Metal(engine) => engine.softmax_rows(scores, causal, start),
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
            #[cfg(target_os = "macos")]
            (Backend::Metal(engine), Data::Metal(buffer)) => Data::Metal(engine.copy_range(buffer, lo, hi)?),
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            _ => return Err(foreign()),
        };
        Ok(Tensor { rows, cols: t.cols, data })
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
            #[cfg(target_os = "macos")]
            (Backend::Metal(engine), Data::Metal(buffer), Data::Metal(p)) => engine.write_range(buffer, lo, p, part.len()),
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            _ => Err(foreign()),
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
    for i in 0..batch {
        let av = ArrayView2::from_shape(ab, &a[i * ab.0 * ab.1..(i + 1) * ab.0 * ab.1]).map_err(|e| shape(e.to_string()))?;
        let bv = ArrayView2::from_shape(bb, &b[i * bb.0 * bb.1..(i + 1) * bb.0 * bb.1]).map_err(|e| shape(e.to_string()))?;
        let mut cv = ArrayViewMut2::from_shape(cb, &mut c[i * cb.0 * cb.1..(i + 1) * cb.0 * cb.1]).map_err(|e| shape(e.to_string()))?;
        let av = if ta == Op::T { av.reversed_axes() } else { av };
        let bv = if tb == Op::T { bv.reversed_axes() } else { bv };
        general_mat_mul(alpha, &av, &bv, beta, &mut cv);
        if arithmetic != Arithmetic::F64 {
            cv.mapv_inplace(|x| f64::from(x as f32));
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
mod cuda {
    use super::{Arithmetic, ColumnBlocks, Data, IndexData, Indices, Op, RmsMode, Tensor, foreign, shape};
    use crate::gpu_error::{GpuError, GpuResultExt};
    use cudarc::cublas::sys::{cublasMath_t, cublasOperation_t};
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
    use cudarc::driver::{CudaContext, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg, ValidAsZeroBits};
    use std::sync::Arc;

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

extern "C" __global__ void softmax_rows(unsigned int rows, unsigned int width, int causal, unsigned int start, double* s) {
    __shared__ double shared[BLOCK];
    unsigned int r = blockIdx.x;
    if (r >= rows) return;
    double* row = s + (u64)r * width;
    unsigned int valid = causal ? start + r % width + 1 : width;
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
"#;

    /// One CUDA device's stream, cuBLAS handle and kernels.
    pub(super) struct Engine {
        pub(super) name: String,
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        blas: CudaBlas,
        module: Arc<CudaModule>,
        /// The row flags of a call that scores every row (never read).
        every_row: CudaSlice<u32>,
        gemm_workspace: std::sync::Mutex<F32Workspace>,
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
            Data::Host(_) => Err(foreign()),
        }
    }

    fn slice_mut(t: &mut Tensor) -> Result<&mut CudaSlice<f64>, GpuError> {
        match &mut t.data {
            Data::Cuda(s) => Ok(s),
            Data::Host(_) => Err(foreign()),
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

    impl Engine {
        pub(super) fn new(ordinal: usize, name: String) -> Result<Self, GpuError> {
            let ctx = crate::device_runtime::cuda_context_for(ordinal)
                .ok_or_else(|| GpuError::DriverCallFailed { reason: format!("no CUDA context for device {ordinal}") })?;
            let stream = ctx.new_stream().gpu_ctx("tensor stream")?;
            let blas = CudaBlas::new(stream.clone()).gpu_ctx("tensor cuBLAS handle")?;
            static MODULE: crate::device_cache::PtxModuleCache = crate::device_cache::PtxModuleCache::new();
            let module = Arc::clone(MODULE.get_or_compile(&ctx, "tensor", KERNELS)?);
            let every_row = stream.alloc_zeros::<u32>(1).gpu_ctx("tensor alloc")?;
            Ok(Self { name, ctx, stream, blas, module, every_row, gemm_workspace: std::sync::Mutex::new(F32Workspace::default()) })
        }

        pub(super) fn memory(&self) -> Result<(usize, usize), GpuError> {
            self.ctx.mem_get_info().gpu_ctx("tensor memory info")
        }

        pub(super) fn synchronize(&self) -> Result<(), GpuError> {
            self.stream.synchronize().gpu_ctx("tensor synchronize")
        }

        pub(super) fn upload<T: DeviceRepr + ValidAsZeroBits>(&self, values: &[T]) -> Result<CudaSlice<T>, GpuError> {
            // An empty tensor still holds one (unread) value: the driver allocates nothing smaller.
            if values.is_empty() {
                return self.stream.alloc_zeros::<T>(1).gpu_ctx("tensor alloc");
            }
            self.stream.clone_htod(values).gpu_ctx("tensor upload")
        }

        pub(super) fn download(&self, slice: &CudaSlice<f64>) -> Result<Vec<f64>, GpuError> {
            self.stream.clone_dtoh(slice).gpu_ctx("tensor download")
        }

        pub(super) fn zeros(&self, n: usize) -> Result<CudaSlice<f64>, GpuError> {
            self.stream.alloc_zeros::<f64>(n.max(1)).gpu_ctx("tensor alloc")
        }

        pub(super) fn copy(&self, slice: &CudaSlice<f64>) -> Result<CudaSlice<f64>, GpuError> {
            let mut out = self.zeros(slice.len())?;
            self.stream.memcpy_dtod(slice, &mut out).gpu_ctx("tensor copy")?;
            Ok(out)
        }

        pub(super) fn copy_range(&self, slice: &CudaSlice<f64>, lo: usize, hi: usize) -> Result<CudaSlice<f64>, GpuError> {
            let mut out = self.zeros(hi - lo)?;
            if hi > lo {
                self.stream.memcpy_dtod(&slice.slice(lo..hi), &mut out).gpu_ctx("tensor row copy")?;
            }
            Ok(out)
        }

        pub(super) fn write_range(&self, slice: &mut CudaSlice<f64>, lo: usize, part: &CudaSlice<f64>) -> Result<(), GpuError> {
            let n = part.len();
            self.stream.memcpy_dtod(part, &mut slice.slice_mut(lo..lo + n)).gpu_ctx("tensor row write")
        }

        fn function(&self, name: &str) -> Result<cudarc::driver::CudaFunction, GpuError> {
            self.module.load_function(name).gpu_ctx_with(|e| format!("tensor kernel {name}: {e}"))
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

        pub(super) fn axpy(&self, y: &mut Tensor, alpha: f64, x: &Tensor) -> Result<(), GpuError> {
            let n = y.len() as u64;
            let f = self.function("axpy")?;
            // SAFETY: equal-length buffers, checked by the caller.
            unsafe { self.stream.launch_builder(&f).arg(&n).arg(&alpha).arg(slice(x)?).arg(slice_mut(y)?).launch(cfg_elements(n)) }
                .gpu_ctx("tensor axpy")
                .map(|_| ())
        }

        pub(super) fn hadamard(&self, out: &mut Tensor, a: &Tensor, b: &Tensor, accumulate: bool) -> Result<(), GpuError> {
            let n = out.len() as u64;
            let acc = i32::from(accumulate);
            let f = self.function("hadamard")?;
            // SAFETY: equal-length buffers, checked by the caller.
            unsafe {
                self.stream.launch_builder(&f).arg(&n).arg(slice(a)?).arg(slice(b)?).arg(slice_mut(out)?).arg(&acc).launch(cfg_elements(n))
            }
            .gpu_ctx("tensor hadamard")
            .map(|_| ())
        }

        pub(super) fn add_row(&self, x: &mut Tensor, alpha: f64, row: &Tensor) -> Result<(), GpuError> {
            let n = x.len() as u64;
            let cols = x.cols as u32;
            let f = self.function("add_row")?;
            // SAFETY: `row` holds `cols` values, `x` n; checked by the caller.
            unsafe {
                self.stream.launch_builder(&f).arg(&n).arg(&cols).arg(&alpha).arg(slice(row)?).arg(slice_mut(x)?).launch(cfg_elements(n))
            }
            .gpu_ctx("tensor add_row")
            .map(|_| ())
        }

        pub(super) fn scale_columns(&self, out: &mut Tensor, x: &Tensor, d: &Tensor, accumulate: bool) -> Result<(), GpuError> {
            let n = out.len() as u64;
            let cols = out.cols as u32;
            let acc = i32::from(accumulate);
            let f = self.function("scale_columns")?;
            // SAFETY: shapes checked by the caller.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&n)
                    .arg(&cols)
                    .arg(slice(x)?)
                    .arg(slice(d)?)
                    .arg(slice_mut(out)?)
                    .arg(&acc)
                    .launch(cfg_elements(n))
            }
            .gpu_ctx("tensor scale_columns")
            .map(|_| ())
        }

        pub(super) fn gather_rows(&self, table: &Tensor, ids: &Indices) -> Result<Tensor, GpuError> {
            let mut out = Tensor { rows: ids.len, cols: table.cols, data: Data::Cuda(self.zeros(ids.len * table.cols)?) };
            let n = out.len() as u64;
            let cols = table.cols as u32;
            let f = self.function("gather_rows")?;
            // SAFETY: ids index rows of `table` (the caller's token ids, inside its vocabulary).
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&n)
                    .arg(&cols)
                    .arg(slice(table)?)
                    .arg(index_slice(ids)?)
                    .arg(slice_mut(&mut out)?)
                    .launch(cfg_elements(n))
            }
            .gpu_ctx("tensor gather_rows")?;
            Ok(out)
        }

        pub(super) fn laws(&self, x: &Tensor, g: Option<&Tensor>, codes: &Indices, c: f64) -> Result<Tensor, GpuError> {
            let mut out = Tensor { rows: x.rows, cols: x.cols, data: Data::Cuda(self.zeros(x.len())?) };
            let n = x.len() as u64;
            let cols = x.cols as u32;
            let slopes = i32::from(g.is_some());
            let f = self.function("laws")?;
            let gs = slice(g.unwrap_or(x))?;
            // SAFETY: equal-length buffers; `codes` holds one code per column.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&n)
                    .arg(&cols)
                    .arg(slice(x)?)
                    .arg(gs)
                    .arg(&slopes)
                    .arg(index_slice(codes)?)
                    .arg(&c)
                    .arg(slice_mut(&mut out)?)
                    .launch(cfg_elements(n))
            }
            .gpu_ctx("tensor laws")?;
            Ok(out)
        }

        pub(super) fn rms(&self, mode: RmsMode, x: &Tensor, g: Option<&Tensor>, epsilon: f64) -> Result<Tensor, GpuError> {
            let mut out = Tensor { rows: x.rows, cols: x.cols, data: Data::Cuda(self.zeros(x.len())?) };
            let (rows, cols) = (x.rows as u32, x.cols as u32);
            let code: i32 = match mode {
                RmsMode::Value => 0,
                RmsMode::Backward => 1,
                RmsMode::Tangent => 2,
            };
            let f = self.function("rms")?;
            let gs = slice(g.unwrap_or(x))?;
            // SAFETY: one block per row of equal-shape buffers.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(&code)
                    .arg(&epsilon)
                    .arg(slice(x)?)
                    .arg(gs)
                    .arg(slice_mut(&mut out)?)
                    .launch(cfg_rows(x.rows))
            }
            .gpu_ctx("tensor rms")?;
            Ok(out)
        }

        pub(super) fn rotate(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, half_split: bool, inverse: bool) -> Result<Tensor, GpuError> {
            let mut out = Tensor { rows: x.rows, cols: x.cols, data: Data::Cuda(self.copy(slice(x)?)?) };
            let (rows, cols, planes) = (x.rows as u32, x.cols as u32, cos.cols as u32);
            let split = i32::from(half_split);
            let sign: f64 = if inverse { -1.0 } else { 1.0 };
            let f = self.function("rotate_planes")?;
            // SAFETY: tables are rows × planes; 2·planes ≤ cols, checked by the caller.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(&planes)
                    .arg(&split)
                    .arg(&sign)
                    .arg(slice(x)?)
                    .arg(slice(cos)?)
                    .arg(slice(sin)?)
                    .arg(slice_mut(&mut out)?)
                    .launch(cfg_elements(u64::from(rows) * u64::from(planes)))
            }
            .gpu_ctx("tensor rotate")?;
            Ok(out)
        }

        pub(super) fn softmax_rows(&self, scores: &mut Tensor, causal: bool, start: usize) -> Result<(), GpuError> {
            let (rows, width) = (scores.rows as u32, scores.cols as u32);
            let launch = cfg_rows(scores.rows);
            let causal = i32::from(causal);
            let start = start as u32;
            let f = self.function("softmax_rows")?;
            // SAFETY: one block per row of a rows × width buffer.
            unsafe { self.stream.launch_builder(&f).arg(&rows).arg(&width).arg(&causal).arg(&start).arg(slice_mut(scores)?).launch(launch) }
                .gpu_ctx("tensor softmax_rows")
                .map(|_| ())
        }

        pub(super) fn softmax_backward(&self, alpha: &Tensor, d: &Tensor) -> Result<Tensor, GpuError> {
            let mut out = Tensor { rows: alpha.rows, cols: alpha.cols, data: Data::Cuda(self.zeros(alpha.len())?) };
            let (rows, cols) = (alpha.rows as u32, alpha.cols as u32);
            let f = self.function("softmax_backward")?;
            // SAFETY: one block per row of equal-shape buffers.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(slice(alpha)?)
                    .arg(slice(d)?)
                    .arg(slice_mut(&mut out)?)
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

        pub(super) fn kl_rows(&self, target: &Tensor, logits: &mut Tensor, scored: Option<&Indices>, gradient: bool) -> Result<Vec<f64>, GpuError> {
            let mut kl = self.zeros(logits.rows)?;
            let (rows, cols) = (logits.rows as u32, logits.cols as u32);
            let (flags, use_flags) = self.flags(scored)?;
            let f = self.function("kl_rows")?;
            let n_rows = logits.rows;
            let gradient = i32::from(gradient);
            // SAFETY: one block per row of equal-shape buffers; `flags` has a flag per row when used.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(slice(target)?)
                    .arg(slice_mut(logits)?)
                    .arg(flags)
                    .arg(&use_flags)
                    .arg(&gradient)
                    .arg(&mut kl)
                    .launch(cfg_rows(n_rows))
            }
            .gpu_ctx("tensor kl_rows")?;
            let mut out = self.download(&kl)?;
            out.truncate(n_rows);
            Ok(out)
        }

        pub(super) fn sampled_cotangent(&self, logits: &mut Tensor, uniforms: &Tensor, scored: Option<&Indices>) -> Result<(), GpuError> {
            let (rows, cols) = (logits.rows as u32, logits.cols as u32);
            let (flags, use_flags) = self.flags(scored)?;
            let f = self.function("sampled_cotangent")?;
            let n_rows = logits.rows;
            // SAFETY: one block per row; one uniform per row.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(slice_mut(logits)?)
                    .arg(slice(uniforms)?)
                    .arg(flags)
                    .arg(&use_flags)
                    .launch(cfg_rows(n_rows))
            }
            .gpu_ctx("tensor sampled_cotangent")
            .map(|_| ())
        }

        pub(super) fn block_products(&self, left: &Tensor, right: &Tensor, blocks: &ColumnBlocks) -> Result<Tensor, GpuError> {
            let mut out = Tensor { rows: left.rows, cols: blocks.len(), data: Data::Cuda(self.zeros(left.rows * blocks.len())?) };
            if out.is_empty() { return Ok(out); }
            let (n, cols, count) = (out.len() as u64, left.cols as u32, blocks.len() as u32);
            let f = self.function("block_products")?;
            // SAFETY: shapes agree, and column_blocks validates monotone offsets within cols.
            unsafe {
                self.stream.launch_builder(&f).arg(&n).arg(&cols).arg(&count)
                    .arg(slice(left)?).arg(slice(right)?).arg(index_slice(&blocks.offsets)?)
                    .arg(slice_mut(&mut out)?).launch(cfg_elements(n))
            }.gpu_ctx("tensor block_products")?;
            Ok(out)
        }

        pub(super) fn sampled_head_cotangent(
            &self, probabilities: &Tensor, mean: &Tensor, head: &Tensor, transposed: bool,
            uniforms: &Tensor, scored: Option<&Indices>,
        ) -> Result<Tensor, GpuError> {
            let mut out = Tensor { rows: mean.rows, cols: mean.cols, data: Data::Cuda(self.zeros(mean.len())?) };
            let (rows, classes, width) = (mean.rows as u32, probabilities.cols as u32, mean.cols as u32);
            let transposed = i32::from(transposed);
            let (flags, use_flags) = self.flags(scored)?;
            let f = self.function("sampled_head_cotangent")?;
            // SAFETY: the public entry validates each tensor shape and the scored flag count.
            // One block writes each row; the sampled label is always inside the vocabulary.
            unsafe {
                self.stream.launch_builder(&f)
                    .arg(&rows).arg(&classes).arg(&width)
                    .arg(slice(probabilities)?).arg(slice(mean)?).arg(slice(head)?).arg(&transposed)
                    .arg(slice(uniforms)?).arg(flags).arg(&use_flags).arg(slice_mut(&mut out)?)
                    .launch(cfg_rows(mean.rows))
            }.gpu_ctx("tensor sampled_head_cotangent")?;
            Ok(out)
        }

        pub(super) fn adam(&self, w: &mut Tensor, (m, v): (&mut Tensor, &mut Tensor), g: &Tensor, (rate, beta1, beta2, epsilon): (f64, f64, f64, f64), (c1, c2): (f64, f64)) -> Result<(), GpuError> {
            let n = w.len() as u64;
            let f = self.function("adam")?;
            // SAFETY: four equal-length buffers, checked by the caller.
            unsafe {
                self.stream.launch_builder(&f).arg(&n).arg(&rate).arg(&beta1).arg(&beta2).arg(&epsilon).arg(&c1).arg(&c2)
                    .arg(slice(g)?).arg(slice_mut(m)?).arg(slice_mut(v)?).arg(slice_mut(w)?).launch(cfg_elements(n))
            }
            .gpu_ctx("tensor adam")
            .map(|_| ())
        }

        pub(super) fn select_sets(&self, (a, q): (&Tensor, &Tensor), bits: &Tensor, left: &Tensor, weight: &Tensor, mask: &mut Tensor) -> Result<(), GpuError> {
            if a.is_empty() { return Ok(()); }
            let (rows, cols) = (a.rows as u32, a.cols as u32);
            let width = a.cols.next_power_of_two().max(2);
            // One scratch ranking and row of sizes per block; the blocks stride over the rows.
            let blocks = a.rows.min(4096);
            let mut keys = self.zeros(blocks * width)?;
            let mut sizes = self.zeros(blocks * width)?;
            let mut order = self.stream.alloc_zeros::<u32>(blocks * width).gpu_ctx("tensor alloc")?;
            let f = self.function("select_sets")?;
            let width32 = width as u32;
            let cfg = LaunchConfig { grid_dim: (blocks as u32, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: shapes checked by the caller; each block owns `width` scratch entries.
            unsafe {
                self.stream.launch_builder(&f).arg(&rows).arg(&cols).arg(&width32)
                    .arg(slice(a)?).arg(slice(q)?).arg(slice(bits)?).arg(slice(left)?).arg(slice(weight)?)
                    .arg(&mut keys).arg(&mut order).arg(&mut sizes).arg(slice_mut(mask)?).launch(cfg)
            }
            .gpu_ctx("tensor select_sets")
            .map(|_| ())
        }

        pub(super) fn box_charge(&self, z: &Tensor, mask: &Tensor, q: &Tensor, cot: &mut Tensor, coefficient: &mut Tensor) -> Result<Vec<f64>, GpuError> {
            if z.is_empty() { return Ok(vec![0.0; z.rows]); }
            let (rows, cols) = (z.rows as u32, z.cols as u32);
            let mut norms = self.zeros(z.rows)?;
            let f = self.function("box_charge")?;
            let cfg = LaunchConfig { grid_dim: (z.rows.min(65_535) as u32, 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: equal shapes checked by the caller; one block per row (striding).
            unsafe {
                self.stream.launch_builder(&f).arg(&rows).arg(&cols).arg(slice(z)?).arg(slice(mask)?).arg(slice(q)?)
                    .arg(&mut norms).arg(slice_mut(cot)?).arg(slice_mut(coefficient)?).launch(cfg)
            }
            .gpu_ctx("tensor box_charge")?;
            let mut values = self.download(&norms)?;
            values.truncate(z.rows);
            Ok(values.into_iter().map(|n| 0.5 * n * n).collect())
        }

        pub(super) fn softmax_quadratic(&self, logits: &Tensor, tangent: &Tensor) -> Result<Vec<f64>, GpuError> {
            let mut out = self.zeros(logits.rows)?;
            let (rows, cols) = (logits.rows as u32, logits.cols as u32);
            let f = self.function("softmax_quadratic")?;
            // SAFETY: one block per row of equal-shape buffers.
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(slice(logits)?)
                    .arg(slice(tangent)?)
                    .arg(&mut out)
                    .launch(cfg_rows(logits.rows))
            }
            .gpu_ctx("tensor softmax_quadratic")?;
            let mut values = self.download(&out)?;
            values.truncate(logits.rows);
            Ok(values)
        }
    }
}

#[cfg(target_os = "macos")]
mod apple {
    use super::{Arithmetic, ColumnBlocks, Data, Device, IndexData, Indices, Op, RmsMode, Tensor, foreign, host, shape};
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
struct P { uint n; uint rows; uint cols; uint extra; uint a; uint b; float alpha; float beta; };

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

// a: causal; extra: the tile's first query position.
kernel void t_softmax_rows(device float* s [[buffer(0)]], constant P& p [[buffer(1)]], uint group [[threadgroup_position_in_grid]],
                           uint groups [[threadgroups_per_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup float shared[GROUP];
    ROWS {
        device float* row = s + (ulong)r * p.cols;
        uint valid = p.a ? min(p.extra + r % p.cols + 1, p.cols) : p.cols;
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
"#;

    const NAMES: &[&str] = &[
        "t_copy",
        "t_scale",
        "t_axpy",
        "t_hadamard",
        "t_add_row",
        "t_scale_columns",
        "t_gather_rows",
        "t_laws",
        "t_rms",
        "t_rotate",
        "t_softmax_rows",
        "t_softmax_backward",
        "t_kl_rows",
        "t_sampled_cotangent",
        "t_block_products",
        "t_sampled_head",
        "t_softmax_quadratic",
        "t_adam",
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

        pub(super) fn softmax_rows(&self, scores: &mut Tensor, causal: bool, start: usize) -> Result<(), GpuError> {
            let p = P { a: u32::from(causal), extra: u32_of(start)?, ..P::default() };
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

        /// `run` on host copies of `inputs` and `outputs`, the outputs then written back: for
        /// the sequential per-row searches no kernel runs (on unified memory each copy is one
        /// pass over the values).
        fn through_host<R>(
            &self,
            inputs: &[&Tensor],
            outputs: &mut [&mut Tensor],
            run: impl FnOnce(&Device, &[Tensor], &mut [Tensor]) -> Result<R, GpuError>,
        ) -> Result<R, GpuError> {
            let host_device = Device::host();
            let copy = |t: &Tensor| -> Result<Tensor, GpuError> {
                let values = self.stream.read::<f32>(buffer(t)?)?.into_iter().take(t.len()).map(f64::from).collect();
                Ok(Tensor { rows: t.rows, cols: t.cols, data: Data::Host(values) })
            };
            let ins = inputs.iter().map(|t| copy(t)).collect::<Result<Vec<_>, _>>()?;
            let mut outs = outputs.iter().map(|t| copy(t)).collect::<Result<Vec<_>, _>>()?;
            let result = run(&host_device, &ins, &mut outs)?;
            for (t, h) in outputs.iter_mut().zip(&outs) {
                let values: Vec<f32> = host(h)?.iter().map(|v| *v as f32).collect();
                t.data = Data::Metal(self.stream.upload(&values)?);
            }
            Ok(result)
        }

        pub(super) fn select_sets(&self, (a, q): (&Tensor, &Tensor), bits: &Tensor, left: &Tensor, weight: &Tensor, mask: &mut Tensor) -> Result<(), GpuError> {
            self.through_host(&[a, q, bits, left, weight], &mut [mask], |h, i, o| h.select_sets((&i[0], &i[1]), &i[2], &i[3], &i[4], &mut o[0]))
        }

        pub(super) fn box_charge(&self, z: &Tensor, mask: &Tensor, q: &Tensor, cot: &mut Tensor, coefficient: &mut Tensor) -> Result<Vec<f64>, GpuError> {
            self.through_host(&[z, mask, q], &mut [cot, coefficient], |h, i, o| {
                let (cot, coefficient) = o.split_at_mut(1);
                h.box_charge(&i[0], &i[1], &i[2], &mut cot[0], &mut coefficient[0])
            })
        }
    }
}

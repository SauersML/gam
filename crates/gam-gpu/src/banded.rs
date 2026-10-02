//! Banded matrix products: every result carries a rigorous forward-error
//! band, whichever executor ran.
//!
//! The device route is the Apple GPU (Metal) in f32 or df64 arithmetic; the
//! CPU route is float64. Both return the same type, and the band describes
//! the arithmetic that actually ran ([`DeviceArithmetic`]), so a consumer
//! checks one band against its own tolerance and never needs to know which
//! executor ran: a device result never replaces a float64 one without its
//! band. Routing follows [`GpuPolicy`]: `off` is the CPU, `auto` takes Metal
//! when a device resolved, the operands are inside the f32 exponent range and
//! the work clears the size floor, and `required` takes Metal or refuses by
//! name, never silently running the CPU.

use crate::apple_gpu::{MetalKernel, decide_metal_under_policy};
use crate::gpu_error::GpuError;
use crate::precision_bounds::{BandRefusal, DeviceArithmetic, GemmBand, row_norms_upper};
use crate::{GpuEligibility, GpuPolicy};
use ndarray::{Array2, ArrayView2};

/// The device arithmetic a banded product asks for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BandedArithmetic {
    /// f32 values and accumulation: relative band about `(k+2)·2^-23`. For
    /// results whose role is a ranking or a proposal that a float64
    /// computation then decides.
    F32,
    /// df64 values and accumulation: relative band about `(k+2)·2^-45`.
    Df64,
}

impl BandedArithmetic {
    const fn device(self) -> DeviceArithmetic {
        match self {
            Self::F32 => DeviceArithmetic::F32,
            Self::Df64 => DeviceArithmetic::Df64,
        }
    }

    const fn kernel(self) -> MetalKernel {
        match self {
            Self::F32 => MetalKernel::GemmF32,
            Self::Df64 => MetalKernel::GemmDf64,
        }
    }

    /// The `auto` size floor, in flops `2·m·n·k`: below it the conversion,
    /// upload and launch cost more than the CPU product.
    pub const fn auto_min_flops(self) -> u128 {
        1 << 27
    }
}

/// How a product reads a resident operand `B` (`rows × cols` as stored).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Layout {
    /// `X·B`.
    AsStored,
    /// `X·Bᵀ`.
    Transposed,
}

/// A product and its band.
#[derive(Clone, Debug)]
pub struct BandedProduct {
    /// `m × n`.
    pub values: Array2<f64>,
    /// The bound on `|values[i,j] − exact|` ([`GemmBand::entry`]).
    pub band: GemmBand,
    /// GPU execution time, zero on the CPU route.
    pub device_seconds: f64,
}

fn shape_error(detail: String) -> GpuError {
    GpuError::DriverCallFailed {
        reason: format!("banded product shape mismatch: {detail}"),
    }
}

fn host_product(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Result<BandedProduct, GpuError> {
    let band = GemmBand::derive(DeviceArithmetic::HostF64, a, b).map_err(|refusal| {
        GpuError::NoDeviceKernel {
            reason: format!("banded product: no float64 band for these operands: {refusal}"),
        }
    })?;
    Ok(BandedProduct {
        values: a.dot(&b),
        band,
        device_seconds: 0.0,
    })
}

/// `A·B` (any layouts) with its band, routed by `policy`.
pub fn banded_matmul(
    policy: GpuPolicy,
    requested: BandedArithmetic,
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
) -> Result<BandedProduct, GpuError> {
    let ((m, k), (bk, n)) = (a.dim(), b.dim());
    if k != bk {
        return Err(shape_error(format!("A is {m}x{k}, B is {bk}x{n}")));
    }
    let device_band = GemmBand::derive(requested.device(), a, b);
    let flops = 2u128 * m as u128 * n as u128 * k as u128;
    let eligibility = eligibility(&device_band, m * n == 0, flops, requested);
    let decision = decide_metal_under_policy(policy, requested.kernel(), eligibility)?;
    decision.require_supported()?;
    #[cfg(target_os = "macos")]
    if let (Some(runtime), Ok(band)) = (decision.device(), &device_band) {
        let context = &runtime.context;
        let (values, timing) = match requested {
            BandedArithmetic::F32 => crate::metal::ops::matmul_f32(context, a, b)?,
            BandedArithmetic::Df64 => crate::metal::ops::matmul_df64(context, a, b)?,
        };
        return Ok(BandedProduct {
            values,
            band: band.clone(),
            device_seconds: timing.gpu_seconds,
        });
    }
    host_product(a, b)
}

fn eligibility(
    band: &Result<GemmBand, BandRefusal>,
    empty: bool,
    flops: u128,
    requested: BandedArithmetic,
) -> GpuEligibility {
    match band {
        Err(refusal) => {
            log::debug!("[banded] device band refused: {refusal}");
            GpuEligibility::CapabilityMissing {
                missing: "operands inside the f32 exponent range and an informative band",
            }
        }
        Ok(_) if empty || flops < requested.auto_min_flops() => GpuEligibility::WorkloadBelowThreshold,
        Ok(_) => GpuEligibility::Eligible,
    }
}

/// A matrix `B` (`rows × cols`) held on the Apple GPU for repeated products
/// `X·B` and `X·Bᵀ` against it (a network weight read by every batch of
/// rows). Uploading once is what makes the device worth it: a product then
/// moves only `X` and the result.
///
/// It exists only when the policy selected Metal for products in its
/// arithmetic ([`resident_operand`]); off the device there is nothing to hold.
#[derive(Debug)]
pub struct ResidentOperand {
    rows: usize,
    cols: usize,
    arithmetic: BandedArithmetic,
    /// Upper bounds on the norms of the stored rows and columns: the band's
    /// right factors for `X·Bᵀ` and `X·B`.
    row_norms: Vec<f64>,
    col_norms: Vec<f64>,
    policy: GpuPolicy,
    #[cfg(target_os = "macos")]
    runtime: &'static crate::apple_gpu::MetalRuntime,
    #[cfg(target_os = "macos")]
    buffer: crate::metal::DeviceBuffer,
    #[cfg(target_os = "macos")]
    stride: usize,
}

/// The `auto` floor of [`resident_operand`], in entries of `B`: below it a
/// weight's products are too small for the device to pay for its launches.
pub const RESIDENT_AUTO_MIN_ENTRIES: usize = 1 << 18;

/// Put `b` on the Apple GPU in `arithmetic` when `policy` selects Metal for
/// its products, `None` when the CPU route is chosen. `auto` requires entries
/// inside the f32 exponent range and at least [`RESIDENT_AUTO_MIN_ENTRIES`];
/// `required` uploads or refuses.
pub fn resident_operand(
    policy: GpuPolicy,
    arithmetic: BandedArithmetic,
    b: ArrayView2<'_, f64>,
) -> Result<Option<ResidentOperand>, GpuError> {
    let (rows, cols) = b.dim();
    let device = arithmetic.device();
    let norms = row_norms_upper(device, "resident operand", b)
        .and_then(|row_norms| Ok((row_norms, row_norms_upper(device, "resident operand", b.t())?)));
    let eligibility = match &norms {
        Err(refusal) => {
            log::debug!("[banded] resident operand refused: {refusal}");
            GpuEligibility::CapabilityMissing {
                missing: "operands inside the f32 exponent range and an informative band",
            }
        }
        Ok(_) if rows * cols < RESIDENT_AUTO_MIN_ENTRIES => GpuEligibility::WorkloadBelowThreshold,
        Ok(_) => GpuEligibility::Eligible,
    };
    let decision = decide_metal_under_policy(policy, arithmetic.kernel(), eligibility)?;
    decision.require_supported()?;
    #[cfg(target_os = "macos")]
    if let (Some(runtime), Ok((row_norms, col_norms))) = (decision.device(), norms) {
        use crate::metal::{msl, ops};
        let context = &runtime.context;
        let (buffer, stride) = match arithmetic {
            BandedArithmetic::F32 => {
                let tile = msl::GEMM_F32_TILE;
                let stride = ops::padded(cols, tile);
                (ops::upload_f32_padded(context, b, ops::padded(rows, tile), stride)?, stride)
            }
            BandedArithmetic::Df64 => (ops::upload_df64(context, b)?, cols),
        };
        return Ok(Some(ResidentOperand {
            rows,
            cols,
            arithmetic,
            row_norms,
            col_norms,
            policy,
            runtime,
            buffer,
            stride,
        }));
    }
    Ok(None)
}

impl ResidentOperand {
    #[must_use]
    pub fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[must_use]
    pub fn arithmetic(&self) -> BandedArithmetic {
        self.arithmetic
    }

    /// `X·B` or `X·Bᵀ` on the device with its band, or `None` when `X` is
    /// too short for the device under `auto` or its band is refused; the
    /// caller then runs its CPU product.
    pub fn product(
        &self,
        x: ArrayView2<'_, f64>,
        layout: Layout,
    ) -> Result<Option<BandedProduct>, GpuError> {
        let (m, k) = x.dim();
        let (inner, n, right_norms) = match layout {
            Layout::AsStored => (self.rows, self.cols, &self.col_norms),
            Layout::Transposed => (self.cols, self.rows, &self.row_norms),
        };
        if k != inner {
            return Err(shape_error(format!(
                "X is {m}x{k}, resident B is {}x{} read {layout:?}",
                self.rows, self.cols
            )));
        }
        let flops = 2u128 * m as u128 * n as u128 * k as u128;
        if m == 0
            || (self.policy != GpuPolicy::Required && flops < self.arithmetic.auto_min_flops())
        {
            return Ok(None);
        }
        let device = self.arithmetic.device();
        let band = match row_norms_upper(device, "left operand", x)
            .and_then(|rows| GemmBand::from_norms(device, k, rows, right_norms.clone()))
        {
            Ok(band) => band,
            Err(refusal) if self.policy == GpuPolicy::Required => {
                return Err(GpuError::RequiredDeviceUnavailable {
                    reason: format!("gpu=required: resident product refused: {refusal}"),
                });
            }
            Err(refusal) => {
                log::debug!("[banded] resident product refused: {refusal}");
                return Ok(None);
            }
        };
        self.device_product(x, layout, band)
    }

    #[cfg(target_os = "macos")]
    fn device_product(
        &self,
        x: ArrayView2<'_, f64>,
        layout: Layout,
        band: GemmBand,
    ) -> Result<Option<BandedProduct>, GpuError> {
        use crate::metal::ops::{self, RightLayout, RightOperand};
        let right = RightOperand {
            buffer: &self.buffer,
            rows: self.rows,
            cols: self.cols,
            stride: self.stride,
        };
        let layout = match layout {
            Layout::AsStored => RightLayout::AsStored,
            Layout::Transposed => RightLayout::Transposed,
        };
        let context = &self.runtime.context;
        let (values, timing) = match self.arithmetic {
            BandedArithmetic::F32 => ops::matmul_f32_resident(context, x, right, layout)?,
            BandedArithmetic::Df64 => ops::matmul_df64_resident(context, x, right, layout)?,
        };
        Ok(Some(BandedProduct {
            values,
            band,
            device_seconds: timing.gpu_seconds,
        }))
    }

    /// Off macOS no operand is ever resident ([`resident_operand`] returns `None`).
    #[cfg(not(target_os = "macos"))]
    fn device_product(
        &self,
        x: ArrayView2<'_, f64>,
        layout: Layout,
        band: GemmBand,
    ) -> Result<Option<BandedProduct>, GpuError> {
        Err(GpuError::NoDeviceKernel {
            reason: format!(
                "no Metal device on this platform for a {}x{} product read {layout:?} \
                 (band {:e})",
                x.nrows(),
                x.ncols(),
                band.max_entry()
            ),
        })
    }
}

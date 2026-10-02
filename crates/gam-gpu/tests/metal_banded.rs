//! The banded products against compensated float64 references: every result
//! lies inside its derived band, whichever executor ran, and the policy
//! promises hold. On a host without Metal the same tests exercise the CPU
//! route and the refusals.

use gam_gpu::GpuPolicy;
use gam_gpu::apple_gpu::{MetalAvailabilityRef, MetalRuntime};
use gam_gpu::banded::{BandedArithmetic, BandedProduct, Layout, banded_matmul, resident_operand};
use gam_gpu::precision_bounds::DeviceArithmetic;
use ndarray::{Array2, ArrayView1};

fn metal_present() -> bool {
    matches!(MetalRuntime::availability(), Ok(MetalAvailabilityRef::Available(_)))
}

/// Deterministic values in `[-scale, scale)` with a spread of exponents.
fn matrix(rows: usize, cols: usize, seed: u64, scale: f64) -> Array2<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    Array2::from_shape_simple_fn((rows, cols), || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
        let exponent = ((state >> 3) % 8) as i32 - 4;
        (2.0 * unit - 1.0) * scale * 2f64.powi(exponent)
    })
}

/// Dot2 (compensated) float64 dot product, exact to about `2^-106`.
fn exact_dot(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    let (mut sum, mut compensation) = (0.0_f64, 0.0_f64);
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

fn assert_inside(product: &BandedProduct, a: &Array2<f64>, b: &Array2<f64>) {
    for i in 0..a.nrows() {
        for j in 0..b.ncols() {
            let error = (product.values[[i, j]] - exact_dot(a.row(i), b.column(j))).abs();
            let band = product.band.entry(i, j);
            assert!(error <= band, "{:?} ({i},{j}): error {error:e} outside band {band:e}", product.band.arithmetic);
        }
    }
}

#[test]
fn products_lie_inside_their_bands_on_every_route() {
    let (a, b) = (matrix(70, 150, 11, 3.0), matrix(150, 90, 29, 2.0));
    for arithmetic in [BandedArithmetic::F32, BandedArithmetic::Df64] {
        let off = banded_matmul(GpuPolicy::Off, arithmetic, a.view(), b.view()).expect("cpu");
        assert_eq!(off.band.arithmetic, DeviceArithmetic::HostF64);
        assert_inside(&off, &a, &b);
        let auto = banded_matmul(GpuPolicy::Auto, arithmetic, a.view(), b.view()).expect("auto");
        assert_inside(&auto, &a, &b);
        let transposed = b.t().to_owned();
        let required = banded_matmul(GpuPolicy::Required, arithmetic, a.view(), transposed.t());
        if metal_present() {
            let required = required.expect("required runs on the device");
            assert!(required.band.arithmetic.is_device());
            assert_inside(&required, &a, &b);
            // A transposed view on the left is read transposed in place.
            let stored = a.t().to_owned();
            let left = banded_matmul(GpuPolicy::Required, arithmetic, stored.t(), b.view()).expect("device");
            assert_inside(&left, &a, &b);
        } else {
            assert!(required.is_err(), "required must never silently run the CPU");
        }
    }
}

#[test]
fn resident_products_read_both_layouts_inside_their_bands() {
    let (x, stored) = (matrix(130, 600, 5, 1.0), matrix(600, 520, 7, 1.0));
    let xt = matrix(130, 520, 9, 1.0);
    assert!(resident_operand(GpuPolicy::Off, BandedArithmetic::F32, stored.view()).expect("off").is_none());
    for arithmetic in [BandedArithmetic::F32, BandedArithmetic::Df64] {
        let resident = resident_operand(GpuPolicy::Required, arithmetic, stored.view());
        if !metal_present() {
            assert!(resident.is_err());
            continue;
        }
        let resident = resident.expect("uploads").expect("resident under required");
        let plain = resident.product(x.view(), Layout::AsStored).expect("runs").expect("device");
        assert_inside(&plain, &x, &stored);
        let transposed = resident.product(xt.view(), Layout::Transposed).expect("runs").expect("device");
        assert_inside(&transposed, &xt, &stored.t().to_owned());
    }
}

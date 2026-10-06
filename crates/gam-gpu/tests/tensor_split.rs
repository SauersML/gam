//! The split products (`Arithmetic::Tf32x3`, `Arithmetic::Bf16x3`) against the exact product, on the
//! host and on CUDA, and the row moves of `Device::gather_ranges` and `Device::scatter_ranges`. Each
//! CUDA test runs when a CUDA device resolves and has nothing to run otherwise.

use gam_gpu::GpuPolicy;
use gam_gpu::tensor::{Arithmetic, Device, Op, Storage};
use ndarray::Array2;

fn cuda() -> Option<Device> {
    let wide = Device::accelerator(GpuPolicy::Auto).expect("a probe that does not fault")?;
    Some(wide.with_storage(Storage::F32).expect("CUDA holds f32"))
}

fn matrix(rows: usize, cols: usize, seed: u64, scale: f64) -> Array2<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    Array2::from_shape_simple_fn((rows, cols), || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
        f64::from(((2.0 * unit - 1.0) * scale) as f32)
    })
}

/// `a` within `band` of `b` entrywise.
fn close(what: &str, a: &Array2<f64>, b: &Array2<f64>, band: impl Fn(usize, usize) -> f64) {
    assert_eq!(a.dim(), b.dim(), "{what}: shapes");
    for ((i, j), x) in a.indexed_iter() {
        let allowed = band(i, j);
        assert!((x - b[[i, j]]).abs() <= allowed, "{what} ({i},{j}): {x} against {} (band {allowed:e})", b[[i, j]]);
    }
}

/// `op(a) op(b)` in `arithmetic` on `device` (f32 operands, and `b` also as a bfloat16 copy where
/// `frozen`), and the float64 product's magnitude `Σ |a| |b|` per entry.
fn split_case(device: &Device, arithmetic: Arithmetic, frozen: bool) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let host = Device::host();
    let (rows, cols, inner) = (40, 56, 700);
    let (a, b) = (matrix(rows, inner, 31, 1.0), matrix(cols, inner, 37, 1.0));
    let b = if frozen { host.download(&host.bf16_copy(&host.upload(b.view()).unwrap()).unwrap()).unwrap() } else { b };
    let (da, db) = (device.upload(a.view()).unwrap(), device.upload(b.view()).unwrap());
    let db = if frozen && !device.is_host() { device.bf16_copy(&db).unwrap() } else { db };
    let mut c = device.zeros(rows, cols).unwrap();
    device.gemm(&mut c, 1.0, &da, Op::N, &db, Op::T, 0.0, arithmetic).unwrap();
    (device.download(&c).unwrap(), a.dot(&b.t()), a.mapv(f64::abs).dot(&b.mapv(f64::abs).t()))
}

/// The largest difference of `c` from `exact` relative to the magnitude summed.
fn worst(c: &Array2<f64>, exact: &Array2<f64>, magnitude: &Array2<f64>) -> f64 {
    c.iter().zip(exact).zip(magnitude).fold(0.0_f64, |m, ((x, e), g)| m.max((x - e).abs() / g))
}

/// A split product is within `2 u` of each term's magnitude plus f32's sums (`(k + 2) 2⁻²⁴`) of the
/// exact product, `u` its unit roundoff; one product of its parts' arithmetic is 64 times further
/// off at least.
#[test]
fn split_products_keep_f32_accuracy_on_the_host() {
    let host = Device::host();
    for arithmetic in [Arithmetic::Tf32x3, Arithmetic::Bf16x3] {
        let mut split = 0.0_f64;
        for frozen in [false, true] {
            let (c, exact, magnitude) = split_case(&host, arithmetic, frozen);
            let bound = 2.0 * arithmetic.unit_roundoff() + 702.0 * f64::from(f32::EPSILON) / 2.0;
            close(&format!("{arithmetic:?} (frozen {frozen})"), &c, &exact, |i, j| bound * magnitude[[i, j]]);
            split = split.max(worst(&c, &exact, &magnitude));
        }
        let single = arithmetic.split().unwrap();
        let (c, exact, magnitude) = split_case(&host, single, false);
        let alone = worst(&c, &exact, &magnitude);
        assert!(alone > 64.0 * split, "{single:?} alone is within {alone:e} of the magnitude, {arithmetic:?} within {split:e}");
    }
}

/// The CUDA split products against the host's, which round alike: within the bound of the exact
/// product either keeps.
#[test]
fn split_products_match_the_host_on_cuda() {
    let Some(d) = cuda() else { return };
    for arithmetic in [Arithmetic::Tf32x3, Arithmetic::Bf16x3] {
        for frozen in [false, true] {
            let (c, exact, magnitude) = split_case(&d, arithmetic, frozen);
            let bound = 2.0 * arithmetic.unit_roundoff() + 702.0 * f64::from(f32::EPSILON) / 2.0;
            close(&format!("CUDA {arithmetic:?} (frozen {frozen})"), &c, &exact, |i, j| bound * magnitude[[i, j]]);
        }
    }
}

/// A call's rows gathered from a stream buffer and scattered back (`Device::gather_ranges`,
/// `scatter_ranges`), exactly, with more ranges than one CUDA launch takes; on the host always.
#[test]
fn row_ranges_move_exactly() {
    let devices: Vec<Device> = std::iter::once(Device::host()).chain(cuda()).collect();
    let x = matrix(2000, 7, 41, 1.0);
    let mut ranges: Vec<std::ops::Range<usize>> = vec![30..41, 3..10, 0..2];
    ranges.extend((0..400).map(|i| 1000 + 2 * i..1001 + 2 * i));
    for dev in &devices {
        let t = dev.upload(x.view()).unwrap();
        let gathered = dev.download(&dev.gather_ranges(&t, &ranges).unwrap()).unwrap();
        let expected: Vec<f64> = ranges.iter().flat_map(|r| x.rows().into_iter().skip(r.start).take(r.len()).flat_map(|row| row.to_vec())).collect();
        assert_eq!(gathered.iter().copied().collect::<Vec<_>>(), expected, "gathered rows");
        let mut back = dev.zeros(2000, 7).unwrap();
        dev.scatter_ranges(&mut back, &ranges, &dev.upload(gathered.view()).unwrap()).unwrap();
        let back = dev.download(&back).unwrap();
        for r in &ranges {
            for i in r.clone() {
                assert_eq!(back.row(i), x.row(i), "row {i} scattered back");
            }
        }
    }
}

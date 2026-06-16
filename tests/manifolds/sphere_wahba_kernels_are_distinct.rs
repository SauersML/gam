//! Sanity check: the Sobolev and pseudo-spline Wahba kernels must produce
//! NUMERICALLY DIFFERENT Gram matrices on the same center set. If they
//! gave identical Gram matrices, the `wahba_kernel` selector would be
//! a no-op — a real bug.

use gam::basis::{SphereWahbaKernel, spherical_wahba_kernel_matrix_with_kind};
use ndarray::array;

fn sample_centers() -> ndarray::Array2<f64> {
    // 12 quasi-uniform points on S² (Fibonacci-ish, lat/lon in degrees).
    let n = 12_usize;
    let mut centers = ndarray::Array2::<f64>::zeros((n, 2));
    let golden = 137.5_f64;
    for i in 0..n {
        let z = (2.0 * i as f64 + 1.0) / (n as f64) - 1.0;
        let lat = z.asin().to_degrees();
        let mut lon = (i as f64) * golden;
        lon = lon.rem_euclid(360.0);
        if lon > 180.0 {
            lon -= 360.0;
        }
        centers[[i, 0]] = lat;
        centers[[i, 1]] = lon;
    }
    centers
}

#[test]
fn sobolev_and_pseudo_kernels_differ_substantially() {
    let centers = sample_centers();
    for m in 1..=4 {
        let k_sob = spherical_wahba_kernel_matrix_with_kind(
            centers.view(),
            centers.view(),
            m,
            false,
            SphereWahbaKernel::Sobolev,
        )
        .expect("Sobolev kernel");
        let k_pse = spherical_wahba_kernel_matrix_with_kind(
            centers.view(),
            centers.view(),
            m,
            false,
            SphereWahbaKernel::Pseudo,
        )
        .expect("Pseudo kernel");
        let max_abs_diff: f64 = k_sob
            .iter()
            .zip(k_pse.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let frob_sob: f64 = k_sob.iter().map(|v| v * v).sum::<f64>().sqrt();
        let frob_pse: f64 = k_pse.iter().map(|v| v * v).sum::<f64>().sqrt();
        eprintln!(
            "[kernel-distinct] m={m} ‖K_sob‖_F={frob_sob:.4e} ‖K_pse‖_F={frob_pse:.4e} max|Δ|={max_abs_diff:.4e}"
        );
        // Demand at least a 1% relative difference somewhere in the matrix.
        let rel = max_abs_diff / frob_sob.max(frob_pse).max(1e-30);
        assert!(
            rel > 0.01,
            "m={m}: Sobolev and pseudo-spline kernels match within {rel:.3e} — \
             the wahba_kernel selector is a no-op",
        );
    }
}

#[test]
fn sobolev_kernel_at_north_pole_matches_paper_closed_form() {
    // Beatson & zu Castell 2018 (Section 6.2) for S²:
    //   k_{3,1}(x) = -ln(u) - 1, with u = (1 - cos γ)/2 = sin²(γ/2)
    // Our wahba_sphere_kernel_matrix already divides by 4π (the
    // surface-area normalization), so we compare against (-ln u - 1)/(4π).
    let p = array![[90.0_f64, 0.0]]; // north pole
    let q = array![[60.0_f64, 0.0]]; // 30° from pole → γ = 30° = π/6
    let k = spherical_wahba_kernel_matrix_with_kind(
        p.view(),
        q.view(),
        1,
        false,
        SphereWahbaKernel::Sobolev,
    )
    .expect("ok");
    let gamma = std::f64::consts::PI / 6.0;
    let u = (1.0 - gamma.cos()) / 2.0;
    let expected = (-u.ln() - 1.0) / (4.0 * std::f64::consts::PI);
    let got = k[[0, 0]];
    eprintln!("[sob-closed] K_sob(γ=π/6, m=1) = {got:.10e}, expected (paper) = {expected:.10e}");
    assert!(
        (got - expected).abs() < 1e-12,
        "K_sob(γ=π/6, m=1) = {got:.6e} != Beatson-zu Castell paper {expected:.6e}",
    );
}

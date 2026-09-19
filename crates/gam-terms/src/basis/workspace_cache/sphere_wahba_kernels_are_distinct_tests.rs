#![cfg(test)]
//! The Sobolev Wahba kernel matrix builders against the Beatson–zu Castell
//! closed forms, and the structural `m = 1` refusal.

use crate::basis::{SphereWahbaKernel, spherical_wahba_kernel_matrix_with_kind};
use ndarray::array;

/// The untruncated Sobolev kernel at `m = 1` is refused at every public matrix
/// entry point (#2475): `K_1 = (−ln u − 1)/4π` is log-singular at coincidence,
/// so its Gram diagonal does not exist and any finite value would be a choice
/// of spectral resolution rather than a limit. That is the same statement as
/// `H¹(S²)` having no bounded point evaluation (Sobolev embedding needs
/// `s > d/2 = 1`), so `m = 1` is not a reproducing kernel on S² at all.
///
/// The refusal is structural — a property of the requested kernel family, not a
/// floating-point coincidence test — so these tests carry the `m = 1` contract
/// as an explicit refusal assertion and run the numerical comparisons over the
/// orders whose diagonals exist.
fn assert_untruncated_sobolev_m1_refused(result: Result<ndarray::Array2<f64>, impl ToString>) {
    let message = match result {
        Ok(_) => panic!("untruncated Sobolev m=1 has no Gram diagonal and must be refused"),
        Err(error) => error.to_string(),
    };
    assert!(
        message.contains("log-singular") && message.contains("SobolevTruncated"),
        "the m=1 refusal must name both the mathematical defect and the \
         explicit-resolution remedy; got: {message}"
    );
}

#[test]
fn sobolev_kernel_at_north_pole_matches_paper_closed_form() {
    // Beatson & zu Castell, "Thinplate Splines on the Sphere", SIGMA 14 (2018)
    // 083, Section 6.2, with u = (1 − cos γ)/2 = sin²(γ/2), all over 4π (the
    // surface-area normalization the matrix builder already applies):
    //
    //   m = 1:  −ln u − 1                                   (refused: see below)
    //   m = 2:  Li₂(1 − u) + 1 − π²/6
    //   m = 3:  −2 Li₃(u) − Li₂(1 − u) + ln(u)·Li₂(u) + 2ζ₃ + π²/6 − 2
    //
    // The reference values below are the paper's expressions evaluated at
    // γ = π/6 in 40-digit arithmetic (mpmath), NOT recomputed from this crate's
    // polylogarithm primitives — otherwise the check would be circular. Each
    // was cross-checked against the spectral definition
    // K_m = (1/4π) Σ_{l≥1} (2l+1)[l(l+1)]^{-m} P_l(cos γ) summed to l = 6000,
    // which reproduces m=3 to 20 digits and m=2 to 12 (the m=2 series tail is
    // the limit there, not the closed form).
    let p = array![[90.0_f64, 0.0]]; // north pole
    let q = array![[60.0_f64, 0.0]]; // 30° from pole → γ = 30° = π/6

    // m = 1 is the paper's k_{3,1} = −ln u − 1. It is log-singular at
    // coincidence, so it has no Gram diagonal and the matrix builders refuse
    // the whole family (#2475) — the same fact as H¹(S²) having no bounded
    // point evaluation. The contract is asserted rather than the value.
    assert_untruncated_sobolev_m1_refused(spherical_wahba_kernel_matrix_with_kind(
        p.view(),
        q.view(),
        1,
        false,
        SphereWahbaKernel::Sobolev,
    ));

    // Orders whose diagonals exist are checked against the paper directly.
    const PAPER_AT_PI_OVER_SIX: [(usize, f64); 2] = [
        (2, 0.059_239_237_216_339_744_164_91),
        (3, 0.027_085_169_433_674_185_310_11),
    ];
    for (m, expected) in PAPER_AT_PI_OVER_SIX {
        let k = spherical_wahba_kernel_matrix_with_kind(
            p.view(),
            q.view(),
            m,
            false,
            SphereWahbaKernel::Sobolev,
        )
        .expect("untruncated Sobolev m>=2 has a finite closed-form diagonal");
        let got = k[[0, 0]];
        let rel = (got - expected).abs() / expected.abs();
        eprintln!(
            "[sob-closed] K_sob(γ=π/6, m={m}) = {got:.17e}, paper = {expected:.17e} rel={rel:.2e}"
        );
        // The evaluation is a handful of polylogarithm calls and O(1)
        // arithmetic on quantities of size ~1, so the achievable bound is a few
        // ulps of the result. 16·f64::EPSILON leaves room for that without
        // admitting anything a formula error could hide in.
        assert!(
            rel <= 16.0 * f64::EPSILON,
            "K_sob(γ=π/6, m={m}) = {got:.17e} != Beatson-zu Castell {expected:.17e} \
             (rel={rel:.3e} > {:.3e})",
            16.0 * f64::EPSILON,
        );
    }
}

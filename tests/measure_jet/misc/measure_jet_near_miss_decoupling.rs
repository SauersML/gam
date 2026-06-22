//! Measure-jet frame acceptance gate 3 (docs/measure_jet_frame.md §7.3):
//! near-miss strand decoupling, asserted at the estimand level against the
//! current energy.
//!
//! Two parallel 1-D strands in 2-D at separation δ = 2× the along-strand
//! center spacing are close enough that every mid-band scale sees both, yet
//! a two-level vector (one constant per strand) is LOCALLY AFFINE on the
//! support of the measure: the offset direction is spanned by the local jet
//! features, so the multiscale jet-residual energy may charge it only the
//! τ-ridge toll — never a diffusion-style value-coupling toll. A crossing
//! (X) geometry breaks the affine compatibility at the shared center region
//! and must charge the same two-level vector at full strength.
//!
//! Deterministic coordinates throughout; no RNG.

use gam::basis::{MeasureJetBand, measure_jet_band, measure_jet_energy_form};
use ndarray::{Array1, Array2};

/// Centers per parallel strand.
const M1: usize = 20;
/// Along-strand center spacing.
const H: f64 = 0.25;
/// Strand separation: 2× the along-strand spacing (mandated by the gate —
/// close enough that every scale from the band floor up sees both strands:
/// the Gaussian truncation radius is 3ε ≥ 3·H > δ already at the floor).
const DELTA: f64 = 2.0 * H;
/// The two strand levels (c1 on strand 1, c2 on strand 2).
const C1: f64 = 0.0;
const C2: f64 = 1.0;
/// `MeasureJetBasisSpec` defaults: the `order_s = 0.0` sentinel realizes
/// s = 1.5 (MEASURE_JET_DEFAULT_ORDER_S), α = 1, τ = 1e-3.
const ORDER_S: f64 = 1.5;
const ALPHA: f64 = 1.0;
const TAU_DEFAULT: f64 = 1e-3;
/// Gaussian profile truncation in units of ε — mirrors the module's
/// MEASURE_JET_PROFILE_CUTOFF so the diffusion comparator below sums the
/// same kernel support the energy itself uses.
const PROFILE_CUTOFF: f64 = 3.0;

/// Two parallel strands: the same x-grid at y = 0 and y = δ, uniform masses.
fn parallel_strand_centers() -> (Array2<f64>, Array1<f64>) {
    let m = 2 * M1;
    let centers = Array2::<f64>::from_shape_fn((m, 2), |(i, k)| {
        let strand = i / M1;
        let j = i % M1;
        if k == 0 {
            j as f64 * H
        } else {
            strand as f64 * DELTA
        }
    });
    let masses = Array1::<f64>::from_elem(m, 1.0 / m as f64);
    (centers, masses)
}

/// X-shaped crossing: strand 1 horizontal, strand 2 vertical, each carrying
/// its own center AT the crossing point — the shared center region where
/// value compatibility is forced (the two coincident quadrature points carry
/// both strand values into every local fit that sees them).
const N_ARM: usize = 5;

fn crossing_strand_centers() -> (Array2<f64>, Array1<f64>) {
    let per_strand = 2 * N_ARM + 1;
    let m = 2 * per_strand;
    let centers = Array2::<f64>::from_shape_fn((m, 2), |(i, k)| {
        let strand = i / per_strand;
        let t = (i % per_strand) as f64 - N_ARM as f64;
        if (strand == 0) == (k == 0) {
            t * H
        } else {
            0.0
        }
    });
    let masses = Array1::<f64>::from_elem(m, 1.0 / m as f64);
    (centers, masses)
}

fn band_for(centers: &Array2<f64>) -> MeasureJetBand {
    measure_jet_band(centers.view(), 0).expect("auto band over deterministic centers")
}

/// The two-level vector: c1 on the first strand's centers, c2 on the second's.
fn two_level_vector(m_first: usize, m_total: usize) -> Array1<f64> {
    Array1::from_shape_fn(m_total, |i| if i < m_first { C1 } else { C2 })
}

fn quadratic_form(q: &Array2<f64>, v: &Array1<f64>) -> f64 {
    v.dot(&q.dot(v))
}

/// The parallel two-level energy at the fitted defaults — shared by both
/// tests so the crossing contrast compares against exactly the quantity the
/// decoupling test bounds.
fn parallel_two_level_energy_at_defaults() -> f64 {
    let (centers, masses) = parallel_strand_centers();
    let band = band_for(&centers);
    let q = measure_jet_energy_form(
        centers.view(),
        masses.view(),
        &band,
        ORDER_S,
        ALPHA,
        TAU_DEFAULT,
    )
    .expect("default-τ energy form");
    let offset = two_level_vector(M1, 2 * M1);
    quadratic_form(&q, &offset)
}

/// Gate 3 proper. On the support {y = 0} ∪ {y = δ} the two-level vector
/// equals the ambient-affine function c1 + (c2−c1)·y/δ, so wherever both
/// strands are visible the local affine fit absorbs the offset exactly
/// (τ = 0) or up to the ridge toll (τ > 0):
///
///   vᵀR_i v = γ²·q_i·Σ_k c_k²·λ_k τ/(λ_k + τ) ≤ γ²·q_i·τ,   γ = (c2−c1)·ε/δ,
///
/// (centered values Cv = γ·Φ̃e_y lie IN the feature span; λ_k = local Gram
/// eigenvalues), giving the closed-form ceiling
///
///   vᵀQv ≤ τ·(c2−c1)²·δ⁻²·log_step·Σ_ℓ ε_ℓ^{2−2s}  ≈ 1.97e-2 here,
///
/// while the alternating checkerboard pays the full multiscale residual
/// (≈ 43 here) and a diffusion-style coupling would pay W_cross ≈ 3.8
/// (derivation at the contrast gate below). Both 1e-2 gates therefore hold
/// with an order of magnitude to spare, and at τ = 0 the offset energy is
/// exactly zero up to roundoff.
#[test]
fn parallel_strands_share_no_value_coupling_at_affine_order() {
    let (centers, masses) = parallel_strand_centers();
    let band = band_for(&centers);
    let m = 2 * M1;

    // Mid-band scales must genuinely see both strands: the band floor is the
    // median nearest-center spacing H, and 3·H > δ, so even the finest scale
    // couples the strands through the kernel — the "near miss" is real.
    assert!(
        band.eps
            .iter()
            .copied()
            .any(|eps| PROFILE_CUTOFF * eps >= DELTA),
        "no band scale sees both strands — the geometry is not a near miss"
    );

    let offset = two_level_vector(M1, m);
    // Alternating ±1 checkerboard on the SAME centers (strand-2 parity
    // flipped so the pattern alternates across strands too).
    let checker = Array1::<f64>::from_shape_fn(m, |i| {
        let parity = (i % M1) + (i / M1);
        if parity % 2 == 0 { 1.0 } else { -1.0 }
    });

    // Default τ: only the ridge toll may remain (affine-damping tolerance,
    // mirroring energy_form_annihilates_affine_when_unridged in-module).
    let q_default = measure_jet_energy_form(
        centers.view(),
        masses.view(),
        &band,
        ORDER_S,
        ALPHA,
        TAU_DEFAULT,
    )
    .expect("default-τ energy form");
    let e_offset = quadratic_form(&q_default, &offset);
    let e_checker = quadratic_form(&q_default, &checker);
    assert!(
        e_checker > 0.0,
        "checkerboard must pay energy; got {e_checker:.3e}"
    );
    assert!(
        e_offset <= 1e-2 * e_checker,
        "two-level offset across parallel strands is not locally affine to the \
         energy: vᵀQv = {e_offset:.3e} vs 1e-2 × checkerboard {e_checker:.3e}"
    );

    // τ = 0 (pseudo-inverse oracle mode): the offset is EXACTLY in the local
    // affine span at every scale — machine-precision annihilation.
    let q_unridged =
        measure_jet_energy_form(centers.view(), masses.view(), &band, ORDER_S, ALPHA, 0.0)
            .expect("unridged energy form");
    let e_offset0 = quadratic_form(&q_unridged, &offset);
    let e_checker0 = quadratic_form(&q_unridged, &checker);
    assert!(
        e_offset0.abs() <= 1e-8 * e_checker0,
        "unridged offset energy {e_offset0:.3e} vs 1e-8 × checkerboard {e_checker0:.3e}"
    );

    // CONTRAST gate: a diffusion-style (value-only) coupling would charge
    // the offset at full strength. Derivation: dropping the jet (slope)
    // correction, the local block is CᵀWC alone and charges, at center i
    // and scale ε, the kernel-weighted variance of the values:
    //
    //   vᵀ(CᵀWC)v = Σ_j w_j·(v_j − v̄_i)²,   w_j = mass_j·e^{−d_ij²/(2ε²)}.
    //
    // For the two-level vector — strand-1 kernel mass a_i, strand-2 kernel
    // mass b_i, q_i = a_i + b_i — the weighted mean is v̄ = (a c1 + b c2)/q
    // and the variance collapses to
    //
    //   a_i·(c1 − v̄)² + b_i·(c2 − v̄)² = (c2 − c1)²·a_i·b_i/q_i,
    //
    // i.e. (c2−c1)² times the harmonic cross-strand kernel weight a·b/q —
    // the Gaussian mass the two strands exchange at this scale. Scattered
    // with the energy's own outer quadrature weight
    // log_step·ε^{−2s}·mass_i·q_i^{1−2α} at the FINEST scale that sees both
    // strands alone (coarser scales only add), this is the floor any
    // diffusion-style coupling would charge. The jet energy must sit at
    // least two orders below it: the offset lives in the affine span, so
    // only the τ-toll (≤ τ·Δc²·δ⁻²·log_step·Σ_ℓ ε_ℓ^{2−2s}, see above)
    // survives — a ratio of ≈ 5e-3 here.
    let eps_star = band
        .eps
        .iter()
        .copied()
        .find(|&eps| PROFILE_CUTOFF * eps >= DELTA)
        .expect("a finest both-strands-visible scale exists");
    let cutoff2 = (PROFILE_CUTOFF * eps_star) * (PROFILE_CUTOFF * eps_star);
    let inv_two_eps2 = 1.0 / (2.0 * eps_star * eps_star);
    let mut w_cross = 0.0_f64;
    for i in 0..m {
        let mut strand1_mass = 0.0_f64;
        let mut strand2_mass = 0.0_f64;
        for j in 0..m {
            let dx = centers[(i, 0)] - centers[(j, 0)];
            let dy = centers[(i, 1)] - centers[(j, 1)];
            let d2 = dx * dx + dy * dy;
            if d2 <= cutoff2 {
                let w = masses[j] * (-d2 * inv_two_eps2).exp();
                if j < M1 {
                    strand1_mass += w;
                } else {
                    strand2_mass += w;
                }
            }
        }
        let q_i = strand1_mass + strand2_mass;
        let outer =
            band.log_step * eps_star.powf(-2.0 * ORDER_S) * masses[i] * q_i.powf(1.0 - 2.0 * ALPHA);
        w_cross += outer * strand1_mass * strand2_mass / q_i;
    }
    assert!(
        w_cross > 0.0,
        "cross-strand kernel weight vanished at ε* = {eps_star:.3}"
    );
    let dc2 = (C2 - C1) * (C2 - C1);
    assert!(
        e_offset <= 1e-2 * dc2 * w_cross,
        "jet energy {e_offset:.3e} is not decoupled from the diffusion-strength \
         charge {:.3e} (= Δc² × cross-strand kernel weight {w_cross:.3e} at ε* = \
         {eps_star:.3})",
        dc2 * w_cross
    );
}

/// The counter-gate: strands that CROSS in an X, each contributing its own
/// center at the crossing, force value compatibility at the shared points.
/// No affine function can be c1 on one full line through the crossing and
/// c2 on the other (at the crossing it would have to take both values, and
/// the symmetric arms kill every slope: by the x → −x / y → −y symmetry the
/// local fit at the crossing is the constant (c1+c2)/2, leaving the FULL
/// two-point variance (Δc²/4)·q as residual). The same two-level vector
/// that rode free across the near-miss gap must now pay two orders of
/// magnitude more than the parallel case (measured ≈ 194×).
#[test]
fn true_crossing_couples_values() {
    let e_parallel = parallel_two_level_energy_at_defaults();
    assert!(
        e_parallel > 0.0,
        "parallel τ-toll must be positive; got {e_parallel:.3e}"
    );

    let (centers, masses) = crossing_strand_centers();
    let band = band_for(&centers);
    let per_strand = 2 * N_ARM + 1;
    let offset = two_level_vector(per_strand, 2 * per_strand);
    let q = measure_jet_energy_form(
        centers.view(),
        masses.view(),
        &band,
        ORDER_S,
        ALPHA,
        TAU_DEFAULT,
    )
    .expect("crossing energy form");
    let e_cross = quadratic_form(&q, &offset);

    assert!(
        e_cross >= 100.0 * e_parallel,
        "a true crossing must couple the strand values: e_cross = {e_cross:.3e} \
         vs 100 × parallel near-miss {e_parallel:.3e}"
    );
}

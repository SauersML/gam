//! Measure-jet frame acceptance gate 3 (docs/measure_jet_frame.md §7.3):
//! near-miss strand decoupling, asserted at the estimand level against the
//! current energy.
//!
//! Two parallel 1-D strands in 2-D at separation δ = 2× the along-strand
//! center spacing are close enough that every mid-band scale sees both, yet
//! a two-level vector (one constant per strand) is LOCALLY AFFINE on the
//! support of the measure: the offset direction is spanned by the local jet
//! features, so the multiscale jet-residual energy annihilates it — never a
//! diffusion-style value-coupling toll. A crossing
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
/// s = 1.5 (MEASURE_JET_DEFAULT_ORDER_S) and α = 1.
const ORDER_S: f64 = 1.5;
const ALPHA: f64 = 1.0;
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

/// The parallel near miss's two-level and checkerboard energies at the fitted
/// defaults, read off one form. The crossing gate uses the offset energy as its
/// decoupled reference and bounds it by the checkerboard.
fn parallel_two_level_and_checkerboard_energies_at_defaults() -> (f64, f64) {
    let (centers, masses) = parallel_strand_centers();
    let band = band_for(&centers);
    let q = measure_jet_energy_form(centers.view(), masses.view(), &band, ORDER_S, ALPHA)
        .expect("energy form");
    let m = 2 * M1;
    let offset = two_level_vector(M1, m);
    let checker = Array1::<f64>::from_shape_fn(m, |i| {
        let parity = (i % M1) + (i / M1);
        if parity % 2 == 0 { 1.0 } else { -1.0 }
    });
    (quadratic_form(&q, &offset), quadratic_form(&q, &checker))
}

/// A derived lower bound on the two-level energy of an X whose strands each
/// carry a center at the crossing.
///
/// The coincident crossing centers `a` and `b` hold c1 and c2. No function,
/// affine or not, takes both values at one point, so every local fit that sees
/// both leaves at least
///
///   w_a·(c1 − ℓ(0))² + w_b·(c2 − ℓ(0))²  ≥  Δc²·w_a·w_b/(w_a + w_b)  ≥  Δc²·min(w_a, w_b)/2
///
/// of residual on them. The energy scatters
/// `log_step·ε^(−η)·net_mass_o·q_o^(1−2α)·vᵀR_o v` over a greedy ε/2-net `o` of
/// the centers, each center's mass aggregated to its nearest net member, and
/// every term is nonnegative.
/// - A center within ε of the crossing aggregates to a member within 1.5ε of
///   it. That member's local fit sees both crossing centers (the kernel cutoff
///   is 3ε) with weights at least m₀·e^(−1.5²/2), where m₀ is the smaller
///   crossing mass.
/// - For α ≥ 1/2, q^(1−2α) is at least (Σ masses)^(1−2α), because q is a
///   truncated, kernel-damped sum of the masses.
///
/// Summing over the band,
///
///   vᵀQv  ≥  Σ_ℓ log_step·ε_ℓ^(−η)·M(ε_ℓ)·(Σ masses)^(1−2α)·Δc²·m₀·e^(−9/8)/2,
///
/// with M(ε) the mass within ε of the crossing. Projecting Q onto the PSD cone
/// only raises vᵀQv.
fn coincident_crossing_floor(
    centers: &Array2<f64>,
    masses: &Array1<f64>,
    band: &MeasureJetBand,
    a: usize,
    b: usize,
) -> f64 {
    assert!(ALPHA >= 0.5, "the q-factor bound needs 1 − 2α ≤ 0");
    assert!(
        1.5 <= PROFILE_CUTOFF,
        "the net member must see the crossing inside the kernel cutoff"
    );
    assert_eq!(
        centers.row(a),
        centers.row(b),
        "the crossing centers must coincide"
    );
    let dc2 = (C2 - C1) * (C2 - C1);
    let m0 = masses[a].min(masses[b]);
    let q_factor = masses.sum().powf(1.0 - 2.0 * ALPHA);
    let eta = 2.0 * ORDER_S + centers.ncols() as f64 * (2.0 - 2.0 * ALPHA);
    band.eps
        .iter()
        .map(|&eps| {
            let near_mass: f64 = (0..centers.nrows())
                .filter(|&j| {
                    let dx = centers[(j, 0)] - centers[(a, 0)];
                    let dy = centers[(j, 1)] - centers[(a, 1)];
                    dx * dx + dy * dy <= eps * eps
                })
                .map(|j| masses[j])
                .sum();
            band.log_step * eps.powf(-eta) * near_mass * q_factor * dc2 * m0
                * (-9.0_f64 / 8.0).exp()
                / 2.0
        })
        .sum()
}

/// Gate 3 proper. On the support {y = 0} ∪ {y = δ} the two-level vector
/// equals the ambient-affine function c1 + (c2−c1)·y/δ, so wherever both
/// strands are visible the centered values Cv = γ·Φ̃e_y (γ = (c2−c1)·ε/δ) lie
/// IN the local feature span and the rank-revealing local affine fit absorbs
/// the offset exactly: vᵀR_i v = 0. The offset energy is therefore zero up to
/// roundoff, while the alternating checkerboard pays the full multiscale
/// residual (≈ 43 here) and a diffusion-style coupling would pay
/// W_cross ≈ 3.8 (derivation at the contrast gate below).
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

    // The offset is EXACTLY in the local affine span at every scale —
    // machine-precision annihilation (mirroring
    // energy_form_annihilates_affine_exactly in-module).
    let q = measure_jet_energy_form(centers.view(), masses.view(), &band, ORDER_S, ALPHA)
        .expect("energy form");
    let e_offset = quadratic_form(&q, &offset);
    let e_checker = quadratic_form(&q, &checker);
    assert!(
        e_checker > 0.0,
        "checkerboard must pay energy; got {e_checker:.3e}"
    );
    assert!(
        e_offset.abs() <= 1e-8 * e_checker,
        "two-level offset across parallel strands is not locally affine to the \
         energy: vᵀQv = {e_offset:.3e} vs 1e-8 × checkerboard {e_checker:.3e}"
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
    // only roundoff survives.
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
/// center at the crossing, force value compatibility at the shared point. No
/// function can be c1 on one full line through the crossing and c2 on the
/// other, because at the crossing it would have to take both values.
/// `coincident_crossing_floor` turns that into a derived lower bound on the X's
/// two-level energy. The same two-level vector that rode free across the
/// near-miss gap pays only roundoff there, which is below that bound.
#[test]
fn true_crossing_couples_values() {
    let (e_parallel, e_checker) = parallel_two_level_and_checkerboard_energies_at_defaults();
    assert!(
        e_parallel.abs() <= 1e-8 * e_checker,
        "parallel offset energy {e_parallel:.3e} vs 1e-8 × checkerboard {e_checker:.3e}"
    );

    let (centers, masses) = crossing_strand_centers();
    let band = band_for(&centers);
    let per_strand = 2 * N_ARM + 1;
    let offset = two_level_vector(per_strand, 2 * per_strand);
    let q = measure_jet_energy_form(centers.view(), masses.view(), &band, ORDER_S, ALPHA)
        .expect("crossing energy form");
    let e_cross = quadratic_form(&q, &offset);
    let floor = coincident_crossing_floor(&centers, &masses, &band, N_ARM, per_strand + N_ARM);

    // Positive control: the bar refuses what a decoupled geometry pays.
    assert!(
        e_parallel < floor,
        "the crossing floor {floor:.3e} does not separate a decoupled near miss ({e_parallel:.3e})"
    );
    assert!(
        e_cross >= floor,
        "a true crossing must pay at least the coincident-value floor: e_cross = {e_cross:.3e} \
         vs floor {floor:.3e}"
    );
}

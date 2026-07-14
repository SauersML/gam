//! Measure-jet frame acceptance battery (docs/measure_jet_frame.md §7),
//! landed against the CURRENT public measure-jet surface
//! (`gam::basis::measure_jet_*`). These gates restate the charter's §7
//! contracts as properties of the shipped single-scale realization — they do not
//! wait for the unlanded jet-frame basis (charter §8 slice 4). Every bound is
//! a principled ceiling derived below from the energy/variance structure, not
//! a tuned threshold, and the geometry is fully deterministic (no RNG).
//!
//! Three gates, one per §7 item that is landable today:
//!
//!  §7.1 — exact affine pass-through at the DEFAULT settings
//!         (`exact_affine_passes_through_at_default_tau`). The realized term
//!         today is the single-scale/multiscale energy of `measure_jet_energy_form`, not
//!         the §1 unpenalized-head frame basis, so machine-exact affine
//!         pass-through requires that future frame block. The strongest
//!         property the current energy gives is asserted: an ambient-affine
//!         function over the centers is damped to ≤ 1e-2× a rough vector at
//!         the DEFAULT τ = 1e-3 (not merely at τ = 0), AND exactly annihilated
//!         (≤ 1e-8× rough) at τ = 0 (the rank-revealing local-affine
//!         projection). UPGRADE: when the §1 unpenalized polynomial head lands
//!         this becomes machine-exact at the default τ too — tighten the
//!         default-τ bound to the τ = 0 tolerance.
//!
//!  §7.2 — off-support variance growth obeys the support-domination theorem
//!         (`support_domination_variance_monotone`): plain Euclidean
//!         monotonicity is explicitly NOT a valid gate (§5); the valid
//!         statement is that a query whose support curve is nowhere larger has
//!         extrapolation variance no smaller. Built from a real
//!         `build_measure_jet_basis` geometry and the public
//!         `measure_jet_support_curve` / `measure_jet_extrapolation_variance`.
//!
//!  §7.3 — near-miss strand decoupling, re-verified under the single-scale-mode
//!         default (`near_miss_decoupling_holds_in_single_scale_mode`): two parallel
//!         strands at a near-miss separation pay at most the τ-ridge toll for
//!         the cross-strand value offset — ≤ 1e-2× the checkerboard energy —
//!         because the offset is ambient-affine on the support.

use gam::basis::{
    BasisMetadata, CenterStrategy, MeasureJetBand, MeasureJetBasisSpec,
    MeasureJetExtrapolationSpectrum, build_measure_jet_basis, measure_jet_band,
    measure_jet_energy_form, measure_jet_extrapolation_variance, measure_jet_support_curve,
};
use ndarray::{Array1, Array2};

/// `MeasureJetBasisSpec` default dials, made explicit for the energy-form
/// gates: the `order_s = 0.0` sentinel realizes s = 1.5
/// (`MEASURE_JET_DEFAULT_ORDER_S`), α = 1, and the DEFAULT τ ridge is 1e-3.
/// These mirror the constants the in-module and near-miss tests pin so the
/// gates speak to exactly the displayed analysis-form target.
const ORDER_S: f64 = 1.5;
const ALPHA: f64 = 1.0;
const TAU_DEFAULT: f64 = 1e-3;
/// Affine damping ceiling at the default τ (§7.1): the local affine residual
/// of a function already in the affine span survives only through the ridge
/// `λ_k τ / (λ_k + τ) ≤ τ`, two orders below a rough vector's full residual.
const AFFINE_DEFAULT_TAU_RATIO: f64 = 1e-2;
/// Machine-precision affine annihilation at τ = 0 (rank-revealing projection).
const AFFINE_EXACT_RATIO: f64 = 1e-8;

fn quadratic_form(q: &Array2<f64>, v: &Array1<f64>) -> f64 {
    v.dot(&q.dot(v))
}

// ===========================================================================
// Gate §7.1 — exact affine pass-through at the DEFAULT τ.
// ===========================================================================

/// Deterministic 2-D center cloud: a `GRID × GRID` lattice on [0, 1]² with a
/// fixed irrational shear so no axis is privileged and the local Gram blocks
/// are full-rank in the affine features {1, x, y}. No RNG.
const GRID: usize = 7;

fn lattice_centers() -> (Array2<f64>, Array1<f64>) {
    let m = GRID * GRID;
    let centers = Array2::<f64>::from_shape_fn((m, 2), |(i, k)| {
        let gx = (i % GRID) as f64 / (GRID - 1) as f64;
        let gy = (i / GRID) as f64 / (GRID - 1) as f64;
        if k == 0 {
            gx + 0.17 * gy
        } else {
            gy - 0.11 * gx
        }
    });
    let masses = Array1::<f64>::from_elem(m, 1.0 / m as f64);
    (centers, masses)
}

/// An ambient-affine function sampled at the centers: f(x, y) = a + b·x + c·y.
/// On any support this lies in every local affine fit's column span, so the
/// jet-residual energy can charge it only the τ-ridge toll.
fn affine_over_centers(centers: &Array2<f64>) -> Array1<f64> {
    let a = 0.4;
    let b = 1.3;
    let c = -0.7;
    Array1::from_shape_fn(centers.nrows(), |i| {
        a + b * centers[(i, 0)] + c * centers[(i, 1)]
    })
}

/// A rough, deterministic alternating field on the lattice — the comparator
/// the affine residual is measured against. Checkerboard parity makes it
/// maximally non-affine at every scale, so it pays the full multiscale
/// residual.
fn rough_over_centers(centers: &Array2<f64>) -> Array1<f64> {
    Array1::from_shape_fn(centers.nrows(), |i| {
        let parity = (i % GRID) + (i / GRID);
        if parity % 2 == 0 { 1.0 } else { -1.0 }
    })
}

/// §7.1. The ambient-affine field over the centers is near-null for the
/// jet-residual energy at the DEFAULT τ = 1e-3 — damped to ≤ 1e-2× the rough
/// comparator — and EXACTLY annihilated (≤ 1e-8× rough) at τ = 0, the
/// rank-revealing local-affine projection. This is the strongest affine
/// pass-through property the current energy realization exposes; the
/// machine-exact-at-default-τ form needs the unlanded §1 unpenalized
/// polynomial head (see module docs UPGRADE note).
#[test]
fn exact_affine_passes_through_at_default_tau() {
    let (centers, masses) = lattice_centers();
    let band: MeasureJetBand =
        measure_jet_band(centers.view(), 0).expect("auto band over deterministic lattice");

    let affine = affine_over_centers(&centers);
    let rough = rough_over_centers(&centers);

    // Default τ = 1e-3: only the ridge toll on the affine field may survive.
    let q_default = measure_jet_energy_form(
        centers.view(),
        masses.view(),
        &band,
        ORDER_S,
        ALPHA,
        TAU_DEFAULT,
    )
    .expect("default-τ energy form");
    let e_affine = quadratic_form(&q_default, &affine);
    let e_rough = quadratic_form(&q_default, &rough);
    assert!(
        e_rough > 0.0,
        "the rough comparator must pay energy; got {e_rough:.3e}"
    );
    assert!(
        e_affine >= 0.0,
        "energy form must be PSD; affine energy {e_affine:.3e} is negative"
    );
    assert!(
        e_affine <= AFFINE_DEFAULT_TAU_RATIO * e_rough,
        "ambient-affine field is not near-null at the DEFAULT τ: vᵀQv = {e_affine:.3e} \
         vs {AFFINE_DEFAULT_TAU_RATIO:.0e} × rough {e_rough:.3e}"
    );

    // τ = 0 (pseudo-inverse oracle mode): machine-precision affine
    // annihilation — the field lies exactly in the local affine span.
    let q_unridged =
        measure_jet_energy_form(centers.view(), masses.view(), &band, ORDER_S, ALPHA, 0.0)
            .expect("unridged energy form");
    let e_affine0 = quadratic_form(&q_unridged, &affine);
    let e_rough0 = quadratic_form(&q_unridged, &rough);
    assert!(
        e_rough0 > 0.0,
        "rough comparator energy must stay positive at τ = 0; got {e_rough0:.3e}"
    );
    assert!(
        e_affine0.abs() <= AFFINE_EXACT_RATIO * e_rough0,
        "unridged affine energy {e_affine0:.3e} is not annihilated vs \
         {AFFINE_EXACT_RATIO:.0e} × rough {e_rough0:.3e}"
    );
}

// ===========================================================================
// Gate §7.2 — support-domination variance monotonicity.
// ===========================================================================

/// Deterministic 2-D filament for the support-domination gate: a single
/// 1-D strand of `STRAND_N` points along a sheared line through [0, 1]², dense
/// enough that `build_measure_jet_basis` realizes a non-degenerate band and a
/// well-populated web. No RNG.
const STRAND_N: usize = 120;
const STRAND_CENTERS: usize = 24;

fn strand_data() -> Array2<f64> {
    Array2::<f64>::from_shape_fn((STRAND_N, 2), |(i, k)| {
        let t = i as f64 / (STRAND_N - 1) as f64;
        if k == 0 { t } else { 0.3 + 0.5 * t }
    })
}

/// §7.2 (charter §5 theorem). Build a real measure-jet geometry, take two
/// queries — one ON the web, one FAR off it — whose support curves satisfy
/// pointwise domination (`q_far ≤ q_near` at every band scale), and assert the
/// dominated (far) query's extrapolation variance is NO SMALLER. Plain
/// Euclidean monotonicity is deliberately NOT asserted (§5 forbids it as a
/// gate); the support-curve domination is verified directly before the
/// variance comparison so the implication is exactly the theorem's hypothesis.
#[test]
fn support_domination_variance_monotone() {
    let data = strand_data();
    let spec = MeasureJetBasisSpec {
        center_strategy: CenterStrategy::FarthestPoint {
            num_centers: STRAND_CENTERS,
        },
        ..Default::default()
    };
    let built = build_measure_jet_basis(data.view(), &spec)
        .expect("measure-jet build over the deterministic strand");

    let BasisMetadata::MeasureJet {
        centers,
        eps_band,
        masses,
        support_means,
        ..
    } = &built.metadata
    else {
        panic!("measure-jet build must carry MeasureJet metadata");
    };

    // Two queries in the SAME raw coordinate space the build used (the build
    // consumes `data` verbatim — input_scale = None). The near query sits on
    // the strand mid-line; the far query is pushed far off the web so its
    // Gaussian support mass is nowhere larger at any scale.
    let mut queries = Array2::<f64>::zeros((2, 2));
    queries[(0, 0)] = 0.5;
    queries[(0, 1)] = 0.3 + 0.5 * 0.5; // on the strand
    queries[(1, 0)] = 25.0;
    queries[(1, 1)] = -25.0; // far off the web

    let support =
        measure_jet_support_curve(queries.view(), centers.view(), masses.view(), eps_band)
            .expect("support curve from the built geometry");

    let near = support.row(0);
    let far = support.row(1);

    // Verify the §5 hypothesis directly: the far query pointwise-dominates
    // (is nowhere larger than) the near query at every band scale.
    let dominates = far
        .iter()
        .zip(near.iter())
        .all(|(&qf, &qn)| qf <= qn + 1e-15);
    assert!(
        dominates,
        "constructed far query does not pointwise-underrun the near query — \
         support-domination hypothesis fails: near = {near:?}, far = {far:?}"
    );
    // And the geometry is non-trivial: the near query genuinely sees the web.
    assert!(
        near.iter().copied().fold(0.0_f64, f64::max) > 0.0,
        "near query sees no web support — degenerate geometry"
    );

    // A fixed, positive per-level precision spectrum: the theorem holds for
    // ANY positive spectrum (every λ̂_ℓ⁻¹ > 0), so a deterministic descending
    // spectrum suffices — no fitted model needed for the estimand-level gate.
    let lambda_phys: Vec<f64> = (0..eps_band.len())
        .map(|l| 1.0 / (1.0 + l as f64))
        .collect();
    let spectrum = MeasureJetExtrapolationSpectrum::PerLevel(&lambda_phys);
    let coverage_floor = 0.05;

    let var_near =
        measure_jet_extrapolation_variance(near, eps_band, support_means, spectrum, coverage_floor)
            .expect("extrapolation variance at the near query");
    let var_far =
        measure_jet_extrapolation_variance(far, eps_band, support_means, spectrum, coverage_floor)
            .expect("extrapolation variance at the far query");

    // The §5 distance-honesty theorem: dominated support ⇒ no-smaller variance.
    assert!(
        var_far + 1e-12 >= var_near,
        "support-domination theorem violated: dominated far query has SMALLER \
         extrapolation variance ({var_far:.6e}) than the near query ({var_near:.6e})"
    );
    // And the off-web query must be strictly more uncertain than the on-web
    // one here — the honest off-support widening the current path lacked.
    assert!(
        var_far > var_near,
        "far-off-web query is not strictly more uncertain than the on-web query: \
         far {var_far:.6e} vs near {var_near:.6e}"
    );
}

// ===========================================================================
// Gate §7.3 — near-miss strand decoupling under the single-scale-mode default.
// ===========================================================================

/// Centers per parallel strand. Multiscale is opt-in (#1116), so the default
/// spec exercises the single-scale-mode energy at any center count.
const NM_M1: usize = 20;
/// Along-strand center spacing.
const NM_H: f64 = 0.25;
/// Strand separation: 2× the along-strand spacing — a genuine near miss
/// (3ε ≥ 3·H > δ already at the band floor, so every scale sees both strands).
const NM_DELTA: f64 = 2.0 * NM_H;
/// The two strand levels (offset across strands).
const NM_C1: f64 = 0.0;
const NM_C2: f64 = 1.0;
/// Gaussian profile truncation in units of ε (mirrors the module cutoff so the
/// near-miss diagnostic below sums the same kernel support the energy uses).
const NM_PROFILE_CUTOFF: f64 = 3.0;
/// Decoupling ceiling: the cross-strand value offset pays ≤ 1e-2× the
/// checkerboard energy.
const NM_DECOUPLE_RATIO: f64 = 1e-2;

fn parallel_strand_centers() -> (Array2<f64>, Array1<f64>) {
    let m = 2 * NM_M1;
    let centers = Array2::<f64>::from_shape_fn((m, 2), |(i, k)| {
        let strand = i / NM_M1;
        let j = i % NM_M1;
        if k == 0 {
            j as f64 * NM_H
        } else {
            strand as f64 * NM_DELTA
        }
    });
    let masses = Array1::<f64>::from_elem(m, 1.0 / m as f64);
    (centers, masses)
}

/// §7.3. Two parallel strands at a near-miss separation, evaluated with the
/// single-scale-mode default energy (default τ = 1e-3). The cross-strand two-level
/// offset (c1 on strand 1, c2 on strand 2) equals the ambient-affine function
/// c1 + (c2−c1)·y/δ on the support {y = 0} ∪ {y = δ}, so it lives in the local
/// affine span and the energy charges it only the τ-ridge toll — ≤ 1e-2× the
/// checkerboard energy, which pays the full multiscale residual. Re-verifies
/// the §7.3 affine-order decoupling under the new default.
#[test]
fn near_miss_decoupling_holds_in_single_scale_mode() {
    let (centers, masses) = parallel_strand_centers();
    let m = 2 * NM_M1;
    let band = measure_jet_band(centers.view(), 0).expect("auto band over parallel strands");

    // The geometry must be a genuine near miss: some band scale sees both
    // strands through the Gaussian kernel (3ε ≥ δ), otherwise the decoupling
    // is trivially true and gates nothing.
    assert!(
        band.eps
            .iter()
            .copied()
            .any(|eps| NM_PROFILE_CUTOFF * eps >= NM_DELTA),
        "no band scale sees both strands — the geometry is not a near miss"
    );

    // Cross-strand value offset and the rough checkerboard comparator on the
    // SAME centers (strand-2 parity flipped so the pattern alternates across
    // strands too — maximally non-affine).
    let offset = Array1::<f64>::from_shape_fn(m, |i| if i < NM_M1 { NM_C1 } else { NM_C2 });
    let checker = Array1::<f64>::from_shape_fn(m, |i| {
        let parity = (i % NM_M1) + (i / NM_M1);
        if parity % 2 == 0 { 1.0 } else { -1.0 }
    });

    // single-scale-mode default τ: only the affine-order ridge toll may survive.
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
        e_offset >= 0.0,
        "energy form must be PSD; offset energy {e_offset:.3e} is negative"
    );
    assert!(
        e_offset <= NM_DECOUPLE_RATIO * e_checker,
        "parallel-strand value offset is not decoupled at affine order in single-scale mode: \
         vᵀQv = {e_offset:.3e} vs {NM_DECOUPLE_RATIO:.0e} × checkerboard {e_checker:.3e}"
    );

    // τ = 0 floor check: the offset lives EXACTLY in the local affine span, so
    // the single-scale-mode coupling above is genuinely the ridge toll, not a small
    // residual that happens to clear the ratio.
    let q_unridged =
        measure_jet_energy_form(centers.view(), masses.view(), &band, ORDER_S, ALPHA, 0.0)
            .expect("unridged energy form");
    let e_offset0 = quadratic_form(&q_unridged, &offset);
    let e_checker0 = quadratic_form(&q_unridged, &checker);
    assert!(
        e_offset0.abs() <= AFFINE_EXACT_RATIO * e_checker0,
        "unridged parallel offset energy {e_offset0:.3e} vs {AFFINE_EXACT_RATIO:.0e} × \
         checkerboard {e_checker0:.3e} — the near-miss offset is not exactly affine"
    );
}

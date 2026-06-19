//! TARGET behavior: the tension/mass collocation sample of the redesigned
//! non-periodic Euclidean Duchon penalty must RESOLVE every basis direction.
//!
//! The Hilbert-scale `OperatorTension` (`Σ‖∇f‖²`) and `OperatorMass` (`Σ(f−f̄)²`)
//! penalties are collocated on an O(k) farthest-point sample of the data. If
//! that sample were degenerate — too few points, or clustered so the collocation
//! design dropped rank — the emitted operator penalty would be rank-deficient in
//! directions the sample fails to see, and REML could not control the wiggle it
//! is meant to. This file guards that the collocation sample is well-conditioned
//! by probing the EMITTED penalty.
//!
//! This is a test of the EVENTUAL redesigned behavior; it will FAIL until the
//! core lands (honest red).
//!
//! METHODOLOGY (coordinate-free, mirroring `duchon_structural_seminorms.rs`).
//! The emitted penalty `S` acts on the design's own coefficient space: for a
//! coefficient vector `c`, `cᵀ S c` equals the (collocated) first-order energy of
//! the function `f = X c`. For a target `g` sampled at the data rows we recover
//! representing coefficients `c = argmin ‖X c − g‖²` and read off `cᵀ S c`. A
//! well-conditioned tension penalty (i) is non-degenerate — its trace is strictly
//! positive, so it penalizes SOME direction; (ii) ANNIHILATES constants — a flat
//! function has zero gradient energy on any sample; and (iii) genuinely PENALIZES
//! a wiggly target — a high-frequency function has strictly positive collocated
//! gradient energy, which can only hold if the collocation sample resolves the
//! wiggly directions. We assert with absolute energy bounds tied to the penalty's
//! own trace, never to any reference tool.
//!
//! NOTE ON A PUBLIC σ_min HOOK. `gam::basis` exposes
//! `build_duchon_collocation_operator_matrices`, which returns a public
//! `CollocationOperatorMatrices { d0, d1, d2, collocation_points, .. }` from
//! which a caller could form `D1ᵀD1` and take its σ_min directly. We deliberately
//! probe the penalty EMITTED by `build_duchon_basis` instead: that is the matrix
//! the fit actually uses, and the farthest-point collocation SAMPLE selection is
//! internal to the build (the public matrix builder takes externally supplied
//! centers/weights). Probing the emitted block guards the end-to-end contract —
//! that the sample chosen by the redesign resolves all basis directions.

use faer::Side;
use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, PenaltySource, SpatialIdentifiability, build_duchon_basis,
};
use gam::faer_ndarray::FaerCholesky;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

// ── fixtures ────────────────────────────────────────────────────────────────

/// `n` rows in `[-1, 1]^d`, deterministic from `seed`.
fn synthetic_data(n: usize, d: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Uniform::new(-1.0_f64, 1.0).expect("uniform params valid");
    let mut data = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            data[[i, j]] = dist.sample(&mut rng);
        }
    }
    data
}

/// The DEFAULT non-periodic Euclidean Duchon spec (the all-on Hilbert scale).
fn default_duchon_spec(k: usize) -> DuchonBasisSpec {
    DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: k },
        periodic: None,
        length_scale: None,
        power: 0.5,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::default(),
    }
}

/// One built Duchon basis: dense design plus the emitted `OperatorTension`
/// block. `tension` is `None` when the build did not emit a collocated tension
/// penalty — itself a contract violation the test surfaces.
struct BuiltDuchon {
    design: Array2<f64>,
    tension: Option<Array2<f64>>,
}

fn build(data: &Array2<f64>, spec: &DuchonBasisSpec) -> BuiltDuchon {
    let result = build_duchon_basis(data.view(), spec).expect("build_duchon_basis succeeded");
    let design: Array2<f64> = result
        .design
        .try_to_dense_arc("collocation-conditioning test")
        .expect("design can be materialized")
        .as_ref()
        .clone();

    // `penalties` holds only the ACTIVE blocks, parallel to the active subset of
    // `penaltyinfo`.
    let mut tension = None;
    let active_sources = result
        .penaltyinfo
        .iter()
        .filter(|info| info.active)
        .map(|info| info.source.clone());
    for (matrix, source) in result.penalties.iter().zip(active_sources) {
        if matches!(source, PenaltySource::OperatorTension) {
            tension = Some(matrix.clone());
        }
    }

    BuiltDuchon { design, tension }
}

/// Coefficients `c` minimizing `‖X c − g‖²` via the normal equations with a
/// vanishing relative ridge for conditioning. Asserts the target is represented
/// (small relative residual) so the recovered `cᵀ S c` is meaningful.
fn coeff_for_target(x: &Array2<f64>, g: &Array1<f64>, what: &str) -> Array1<f64> {
    let p = x.ncols();
    let xtx = x.t().dot(x);
    let max_diag = xtx.diag().iter().cloned().fold(1.0_f64, f64::max);
    let mut gram = xtx.clone();
    let eps = 1e-10 * max_diag;
    for i in 0..p {
        gram[[i, i]] += eps;
    }
    let xtg = x.t().dot(g);
    let chol = gram
        .cholesky(Side::Lower)
        .expect("Cholesky of X'X + εI for target projection");
    let c = chol.solvevec(&xtg);

    let fit = x.dot(&c);
    let resid: f64 = fit
        .iter()
        .zip(g.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale = g.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
    assert!(
        resid / scale < 1e-4,
        "design cannot represent the {what} target (rel residual {:.3e}); the probe \
         requires the target be representable in the design's column space",
        resid / scale
    );
    c
}

/// `cᵀ S c`.
fn quad(s: &Array2<f64>, c: &Array1<f64>) -> f64 {
    c.dot(&s.dot(c))
}

fn constant_target(data: &Array2<f64>) -> Array1<f64> {
    Array1::from_elem(data.nrows(), 0.7)
}

/// A genuinely wiggly target sampled at the data rows: a high-frequency sine of
/// the first coordinate (rescaled from [-1,1] to roughly four cycles). Its
/// gradient energy is strictly positive, so a tension penalty whose collocation
/// sample resolves the wiggly directions must charge it a positive cost.
fn wiggly_target(data: &Array2<f64>) -> Array1<f64> {
    let n = data.nrows();
    let mut g = Array1::zeros(n);
    let freq = 2.0 * std::f64::consts::PI * 2.0; // ~2 cycles over [-1, 1]
    for i in 0..n {
        g[i] = (freq * data[[i, 0]]).sin();
    }
    g
}

// ── (5) the collocated tension penalty resolves all basis directions ─────────

/// COLLOCATION SAMPLE WELL-CONDITIONED. The emitted `OperatorTension` penalty is
/// non-degenerate and resolves the basis directions: (i) its trace is strictly
/// positive (it penalizes SOME gradient energy — the collocation sample is not
/// rank-collapsed); (ii) it ANNIHILATES constants (a flat function has zero
/// gradient energy on any sample); and (iii) it charges a STRICTLY POSITIVE cost
/// to a wiggly target (a high-frequency function is penalized — only possible if
/// the O(k) farthest-point collocation sample resolves the wiggly directions
/// rather than missing them). A degenerate / clustered sample would leave some
/// wiggly direction unseen and the wiggly energy would collapse toward the
/// constant's; the gap between them is the conditioning guard.
#[test]
fn collocation_sample_well_conditioned() {
    let data = synthetic_data(400, 1, 11);
    let spec = default_duchon_spec(24);
    let built = build(&data, &spec);
    let tension = built.tension.as_ref().expect(
        "the all-on default Duchon must emit a collocated OperatorTension penalty \
         (first-order energy Σ‖∇f‖² on the farthest-point sample)",
    );

    // (i) Non-degeneracy: the tension penalty charges SOME direction. Its trace
    // is the reference scale for the "≈ 0 on constants" and "≫ 0 on wiggle"
    // bounds below.
    let trace: f64 = (0..tension.nrows()).map(|i| tension[[i, i]]).sum();
    assert!(
        trace > 1e-6,
        "collocated tension penalty is degenerate (trace={trace:.3e}); the O(k) \
         farthest-point sample must resolve gradient energy, not collapse to rank 0"
    );

    // (ii) Constants are annihilated: zero gradient energy on any sample.
    let c_const = coeff_for_target(&built.design, &constant_target(&data), "constant");
    let energy_const = quad(tension, &c_const);
    let const_bound = 1e-6 * trace.max(1.0);
    assert!(
        energy_const <= const_bound,
        "collocated tension penalty must annihilate constants (∇const = 0): \
         energy_const={energy_const:.3e} vs trace={trace:.3e}"
    );

    // (iii) A wiggly target is genuinely penalized. If the collocation sample
    // resolved all basis directions, the high-frequency target carries real
    // gradient energy — strictly positive and far above the constant's residual.
    let c_wiggle = coeff_for_target(&built.design, &wiggly_target(&data), "wiggly");
    let energy_wiggle = quad(tension, &c_wiggle);
    eprintln!(
        "duchon-collocation-conditioning: k=24 trace={trace:.4} \
         energy_const={energy_const:.3e} energy_wiggle={energy_wiggle:.3e}"
    );
    // The wiggle's collocated gradient energy must dominate: a non-trivial
    // fraction of the penalty's own trace scale, and orders of magnitude above
    // the constant's residual. A sample that failed to resolve the wiggly
    // directions would let this collapse toward `energy_const`.
    assert!(
        energy_wiggle > 1e-3 * trace.max(1.0),
        "collocated tension penalty fails to penalize a wiggly target \
         (energy_wiggle={energy_wiggle:.3e} vs trace={trace:.3e}); the collocation \
         sample does not resolve the high-frequency directions"
    );
    assert!(
        energy_wiggle > 1e3 * energy_const.max(f64::MIN_POSITIVE),
        "collocated tension penalty does not separate wiggle from constant \
         (energy_wiggle={energy_wiggle:.3e}, energy_const={energy_const:.3e}); a \
         well-conditioned sample must charge curvature far more than a flat function"
    );
}

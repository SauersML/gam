//! Structural semantics of the redesigned non-periodic Euclidean Duchon
//! smoother.
//!
//! CONTRACT (the object these tests pin down): a non-periodic Euclidean Duchon
//! term is a polyharmonic smoother built on the cubic (`r^3`) basis, penalized
//! by its own native reproducing-kernel norm (the analytic, closed-form analogue
//! of mgcv's `bs="ds"`). Its penalty has two analytic blocks:
//!
//!   * `Primary` — the native roughness Gram `α²·Zᵀ K_CC Z`, which IS the exact
//!            `(m+s)`-order Duchon seminorm. Null space = the AFFINE functions
//!            (constant + linear): a polynomial in the null space has zero
//!            `(m+s)`-order roughness. It penalizes curvature / wiggle.
//!   * `DoublePenaltyNullspace` — an analytic shrinkage ridge on the affine null
//!            space (the Wood `select=TRUE` pattern). It penalizes the affine
//!            TREND, so the linear part is not left unpenalized; the global mean
//!            is freed only later, by the model's identifiability transform.
//!
//! The headline correctness claim — and the regression these tests guard — is
//! that the linear TREND is genuinely penalized (by the ridge) while its
//! ROUGHNESS is correctly annihilated (by the native Gram). Under the old
//! zero-padded-polynomial-block construction the affine columns sat in an
//! unpenalized side block; that is the behaviour these tests must reject.
//!
//! METHODOLOGY. The penalty matrices returned by `build_duchon_basis` act on the
//! design's own coefficient space: for a coefficient vector `c`, the penalty value
//! is the quadratic form `cᵀ S c` and equals the energy of the function `f = X c`.
//! We never need the internal coordinates: for a target function `g` sampled at the
//! data rows we recover the representing coefficients `c = argmin ‖X c − g‖²` by
//! solving the normal equations (the design carries the affine columns, so constant
//! and linear targets are represented exactly, residual ≈ 0), then read off
//! `cᵀ S c`. This is a coordinate-free, principled probe of the penalty null
//! spaces. We assert the structural pattern with absolute energy bounds tied to the
//! penalty normalization, not to any reference tool.

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

fn duchon_spec(k: usize, order: DuchonNullspaceOrder) -> DuchonBasisSpec {
    DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: k },
        periodic: None,
        length_scale: None,
        // d=2 cubic default s=(d-1)/2=0.5: satisfies CPD (2s<d) and yields the
        // r^3 kernel, whose value is finite at the center self-pairs that the
        // native Gram K_CC evaluates (the r^2 log r kernel at power=0 is the
        // thin-plate case; r^3 keeps the analytic seminorm clean here).
        power: 0.5,
        nullspace_order: order,
        // No extra intercept centering: we want the affine columns present in
        // the raw design so constant / linear targets are exactly representable.
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::default(),
    }
}

/// One built Duchon basis: dense design plus the two analytic penalty blocks
/// keyed by source. Each is `None` when the construction did not emit that block
/// — a missing roughness Gram or null-space ridge is itself a contract violation
/// the tests below surface explicitly.
struct BuiltDuchon {
    design: Array2<f64>,
    /// Native reproducing-norm roughness Gram `α²·Zᵀ K_CC Z` — the exact
    /// `(m+s)`-order Duchon seminorm. Annihilates the affine null space
    /// (constant + linear); penalizes curvature / wiggle.
    primary: Option<Array2<f64>>,
    /// Analytic null-space shrinkage ridge. Penalizes the affine trend's slope
    /// (mean-free — the constant direction sits in its null space).
    ridge: Option<Array2<f64>>,
}

fn build(data: &Array2<f64>, spec: &DuchonBasisSpec) -> BuiltDuchon {
    let result = build_duchon_basis(data.view(), spec).expect("build_duchon_basis succeeded");
    let design: Array2<f64> = result
        .design
        .try_to_dense_arc("structural-seminorm test")
        .expect("design can be materialized")
        .as_ref()
        .clone();

    // `penalties` holds only the ACTIVE blocks, parallel to the active subset of
    // `penaltyinfo`. The analytic Duchon penalty emits a Primary roughness Gram
    // and (when the null space is non-trivial) a DoublePenaltyNullspace ridge.
    let mut primary = None;
    let mut ridge = None;
    let active_sources = result
        .penaltyinfo
        .iter()
        .filter(|info| info.active)
        .map(|info| info.source.clone());
    for (matrix, source) in result.penalties.iter().zip(active_sources) {
        match source {
            PenaltySource::Primary => primary = Some(matrix.clone()),
            PenaltySource::DoublePenaltyNullspace => ridge = Some(matrix.clone()),
            _ => {}
        }
    }

    BuiltDuchon {
        design,
        primary,
        ridge,
    }
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

    // Sanity: the design must represent the target essentially exactly.
    let fit = x.dot(&c);
    let resid: f64 = fit
        .iter()
        .zip(g.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale = g.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
    assert!(
        resid / scale < 1e-6,
        "design cannot represent the {what} target (rel residual {:.3e}); \
         the affine columns must be in the design for this probe to be valid",
        resid / scale
    );
    c
}

/// `cᵀ S c`.
fn quad(s: &Array2<f64>, c: &Array1<f64>) -> f64 {
    c.dot(&s.dot(c))
}

/// Constant, linear, and quadratic targets sampled at the data rows.
fn constant_target(data: &Array2<f64>) -> Array1<f64> {
    Array1::from_elem(data.nrows(), 0.7)
}

fn linear_target(data: &Array2<f64>) -> Array1<f64> {
    // 0.4 + Σ_j (j + 1) · x_j  — a genuinely non-constant affine map.
    let n = data.nrows();
    let d = data.ncols();
    let mut g = Array1::from_elem(n, 0.4);
    for i in 0..n {
        for j in 0..d {
            g[i] += (j as f64 + 1.0) * data[[i, j]];
        }
    }
    g
}

// ── (a) native Gram null space = affine; the ridge penalizes the trend ──────

/// The native reproducing-norm Gram `α²·Zᵀ K_CC Z` is the exact `(m+s)`-order
/// Duchon seminorm, so it annihilates the AFFINE null space (constants AND
/// pure-linear maps) and penalizes curvature. With an affine (m=2) null space the
/// linear trend has zero `(m+s)`-order roughness, while the kernel block carries
/// real roughness energy (non-degenerate Gram).
#[test]
fn native_gram_roughness_null_space_is_affine() {
    let data = synthetic_data(180, 2, 11);
    let spec = duchon_spec(16, DuchonNullspaceOrder::Linear);
    let built = build(&data, &spec);
    let primary = built
        .primary
        .as_ref()
        .expect("native Gram (Primary) penalty must be present");

    // Affine targets are exactly representable by the design's affine columns,
    // so their projected coefficients are exact.
    let c_const = coeff_for_target(&built.design, &constant_target(&data), "constant");
    let c_lin = coeff_for_target(&built.design, &linear_target(&data), "linear");
    let energy_const = quad(primary, &c_const);
    let energy_lin = quad(primary, &c_lin);

    // Non-degeneracy: the Gram penalizes curvature somewhere — its trace is the
    // reference scale that makes the "≈ 0" affine bounds meaningful.
    let trace: f64 = (0..primary.nrows()).map(|i| primary[[i, i]]).sum();
    assert!(
        trace > 1e-6,
        "native Gram must be non-degenerate (penalize curvature); trace={trace:.3e}"
    );
    let bound = 1e-8 * trace.max(1.0);
    assert!(
        energy_const <= bound,
        "native Gram must annihilate constants: {energy_const:.3e} vs trace {trace:.3e}"
    );
    assert!(
        energy_lin <= bound,
        "native Gram must annihilate the AFFINE null space (a linear map has zero \
         (m+s)-order roughness): linear {energy_lin:.3e} vs trace {trace:.3e}"
    );
}

// ── (b) the null-space ridge penalizes the affine trend ─────────────────────

/// The analytic null-space ridge (`DoublePenaltyNullspace`) penalizes the affine
/// trend's slope, so the linear trend is NOT left fully unpenalized: a
/// non-constant linear function carries strictly positive ridge energy. (In a
/// fitted model the sum-to-zero identifiability transform removes the constant so
/// only the global mean stays free; here, with `identifiability = None`, we verify
/// the trend itself is penalized — the native Gram annihilates it, the ridge does
/// not.) This is the analytic counterpart of "the trend is not in an unpenalized
/// side block."
#[test]
fn null_space_ridge_penalizes_the_linear_trend() {
    let data = synthetic_data(200, 2, 21);
    let spec = duchon_spec(20, DuchonNullspaceOrder::Linear);
    let built = build(&data, &spec);
    let ridge = built.ridge.as_ref().expect(
        "null-space ridge (DoublePenaltyNullspace) must be present for an affine null space",
    );

    let c_lin = coeff_for_target(&built.design, &linear_target(&data), "linear");
    let ridge_energy = quad(ridge, &c_lin);
    // Reference: the linear function's own L2 energy at the data sites, so
    // "nonzero" is a real bound rather than float wobble.
    let g = linear_target(&data);
    let g_energy: f64 = g.iter().map(|v| v * v).sum();
    assert!(
        ridge_energy > 1e-6 * g_energy,
        "the null-space ridge must penalize a linear trend's slope (the trend is \
         not left unpenalized): ridge energy {ridge_energy:.3e}, reference L2 {g_energy:.3e}"
    );

    // The native roughness Gram, by contrast, annihilates the linear trend (it is
    // in the (m+s)-order seminorm's null space): trend control comes from the
    // ridge, not from the roughness penalty.
    if let Some(primary) = built.primary.as_ref() {
        let primary_energy = quad(primary, &c_lin);
        let trace: f64 = (0..primary.nrows()).map(|i| primary[[i, i]]).sum();
        assert!(
            primary_energy <= 1e-6 * trace.max(1.0),
            "the native Gram must leave the linear trend's roughness ≈ 0: \
             {primary_energy:.3e} vs trace {trace:.3e}"
        );
    }
}

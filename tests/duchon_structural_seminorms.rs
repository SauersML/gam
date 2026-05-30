//! Structural semantics of the redesigned non-periodic Euclidean Duchon
//! smoother.
//!
//! CONTRACT (the object these tests pin down): a non-periodic Euclidean Duchon
//! term is a *structural* amplitude / slope / curvature smoother built on the
//! cubic (`r^3`) polyharmonic basis. Its penalty is the sum of three data-jet
//! seminorms over the basis, each with its own smoothing parameter:
//!
//!   * `S0` — MASS / amplitude (centered deviation): `Σ_i (f(x_i) − f̄)²`.
//!            Null space = the constant functions (the global mean is free).
//!   * `S1` — TENSION / slope: `Σ_i ‖∇f(x_i)‖²`.
//!            Null space = the constant functions (gradient kills constants).
//!   * `S2` — STIFFNESS / curvature: `Σ_i ‖H f(x_i)‖_F²`.
//!            Null space = the AFFINE functions (Hessian kills affine maps).
//!
//! Crucially, the affine polynomial columns are INCLUDED in the slope and
//! curvature penalties. The headline correctness claim — and the regression
//! these tests guard — is that a pure-linear function therefore receives a
//! NONZERO tension (`S1`) penalty. Under the old zero-padded-polynomial-block
//! construction the affine columns sat in an unpenalized side block, so a
//! linear function got exactly zero tension; that is the behaviour these tests
//! must reject.
//!
//! METHODOLOGY. The penalty matrices `S_k` returned by `build_duchon_basis`
//! act on the design's own coefficient space: for a coefficient vector `c`,
//! the seminorm value is the quadratic form `cᵀ S_k c` and equals the data-jet
//! energy of the function `f = X c`. We never need the internal coordinates:
//! for a target function `g` sampled at the data rows we recover the
//! representing coefficients `c = argmin ‖X c − g‖²` by solving the normal
//! equations (the design carries the affine columns, so constant / linear / and
//! — at order ≥ 2 — quadratic targets are represented exactly, residual ≈ 0),
//! then read off `cᵀ S_k c`. This is a coordinate-free, principled probe of the
//! seminorm null spaces. We assert the structural pattern with absolute energy
//! bounds tied to the penalty normalization, not to any reference tool.

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, PenaltySource, SpatialIdentifiability, build_duchon_basis,
};
use faer::Side;
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
        center_strategy: CenterStrategy::FarthestPoint { num_centers: k },
        periodic: None,
        length_scale: None,
        power: 1.0,
        nullspace_order: order,
        // No extra intercept centering: we want the affine columns present in
        // the raw design so constant / linear targets are exactly representable.
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::default(),
    }
}

/// One built Duchon basis: dense design plus the three structural seminorm
/// matrices keyed by source. Each `S_k` is `None` when the construction dropped
/// that block (numerically rank-zero) — a dropped tension/curvature block is
/// itself a contract violation the tests below surface explicitly.
struct BuiltDuchon {
    design: Array2<f64>,
    mass: Option<Array2<f64>>,
    tension: Option<Array2<f64>>,
    stiffness: Option<Array2<f64>>,
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
    // `penaltyinfo` (dropped blocks appear in `penaltyinfo` but not in
    // `penalties`). Walk them together to key each matrix by its source.
    let mut mass = None;
    let mut tension = None;
    let mut stiffness = None;
    let active_sources = result
        .penaltyinfo
        .iter()
        .filter(|info| info.active)
        .map(|info| info.source.clone());
    for (matrix, source) in result.penalties.iter().zip(active_sources) {
        match source {
            PenaltySource::OperatorMass => mass = Some(matrix.clone()),
            PenaltySource::OperatorTension => tension = Some(matrix.clone()),
            PenaltySource::OperatorStiffness => stiffness = Some(matrix.clone()),
            _ => {}
        }
    }

    BuiltDuchon {
        design,
        mass,
        tension,
        stiffness,
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

fn quadratic_target(data: &Array2<f64>) -> Array1<f64> {
    // 0.3 + Σ_j x_j² — non-affine, so the Hessian (stiffness) is nonzero.
    let n = data.nrows();
    let d = data.ncols();
    let mut g = Array1::from_elem(n, 0.3);
    for i in 0..n {
        for j in 0..d {
            g[i] += data[[i, j]] * data[[i, j]];
        }
    }
    g
}

// ── (a) null spaces: S0 = constants, S1 = constants, S2 = affine ────────────

/// The MASS seminorm (centered deviation) annihilates constants: a constant
/// function has zero centered amplitude.
#[test]
fn mass_seminorm_null_space_is_constants() {
    let data = synthetic_data(180, 2, 11);
    let spec = duchon_spec(16, DuchonNullspaceOrder::Linear);
    let built = build(&data, &spec);
    let mass = built
        .mass
        .as_ref()
        .expect("mass (S0) penalty must be present");

    let c_const = coeff_for_target(&built.design, &constant_target(&data), "constant");
    let energy_const = quad(mass, &c_const);

    // Reference scale: a non-constant linear function should carry real mass
    // energy, so we compare the constant's mass against that to make the "≈ 0"
    // bound meaningful rather than absolute.
    let c_lin = coeff_for_target(&built.design, &linear_target(&data), "linear");
    let energy_lin = quad(mass, &c_lin);
    assert!(
        energy_lin > 1e-6,
        "a non-constant linear function must carry nonzero MASS energy; got {energy_lin:.3e}"
    );
    assert!(
        energy_const <= 1e-8 * energy_lin.max(1.0),
        "MASS (S0) must annihilate constants: constant energy {energy_const:.3e} \
         is not negligible vs linear-mass {energy_lin:.3e}"
    );
}

/// The TENSION seminorm (slope) annihilates constants: ∇(const) = 0.
#[test]
fn tension_seminorm_null_space_is_constants() {
    let data = synthetic_data(180, 2, 12);
    let spec = duchon_spec(16, DuchonNullspaceOrder::Linear);
    let built = build(&data, &spec);
    let tension = built
        .tension
        .as_ref()
        .expect("tension (S1) penalty must be present and non-dropped");

    let c_const = coeff_for_target(&built.design, &constant_target(&data), "constant");
    let energy_const = quad(tension, &c_const);

    let c_lin = coeff_for_target(&built.design, &linear_target(&data), "linear");
    let energy_lin = quad(tension, &c_lin);
    assert!(
        energy_lin > 1e-6,
        "a non-constant linear function must carry nonzero TENSION energy; got {energy_lin:.3e}"
    );
    assert!(
        energy_const <= 1e-8 * energy_lin.max(1.0),
        "TENSION (S1) must annihilate constants: constant energy {energy_const:.3e} \
         is not negligible vs linear-slope {energy_lin:.3e}"
    );
}

/// The STIFFNESS seminorm (curvature) annihilates AFFINE functions: both
/// constants and pure-linear maps have zero Hessian.
#[test]
fn stiffness_seminorm_null_space_is_affine() {
    let data = synthetic_data(220, 2, 13);
    // order=2 → affine + quadratic representable; quadratic target then carries
    // real stiffness while affine targets must not.
    let spec = duchon_spec(24, DuchonNullspaceOrder::Degree(2));
    let built = build(&data, &spec);
    let stiffness = built
        .stiffness
        .as_ref()
        .expect("stiffness (S2) penalty must be present");

    let c_const = coeff_for_target(&built.design, &constant_target(&data), "constant");
    let c_lin = coeff_for_target(&built.design, &linear_target(&data), "linear");
    let c_quad = coeff_for_target(&built.design, &quadratic_target(&data), "quadratic");

    let energy_const = quad(stiffness, &c_const);
    let energy_lin = quad(stiffness, &c_lin);
    let energy_quad = quad(stiffness, &c_quad);

    assert!(
        energy_quad > 1e-6,
        "a quadratic function must carry nonzero STIFFNESS energy; got {energy_quad:.3e}"
    );
    let bound = 1e-8 * energy_quad.max(1.0);
    assert!(
        energy_const <= bound,
        "STIFFNESS (S2) must annihilate constants: {energy_const:.3e} vs quad {energy_quad:.3e}"
    );
    assert!(
        energy_lin <= bound,
        "STIFFNESS (S2) must annihilate AFFINE maps: linear-curvature {energy_lin:.3e} \
         is not negligible vs quadratic-curvature {energy_quad:.3e}"
    );
}

// ── (b) HEADLINE: a linear column gets NONZERO tension ──────────────────────

/// The redesign's headline correctness property: with the affine columns folded
/// INTO the slope penalty, a pure-linear function receives a strictly nonzero
/// TENSION (S1) penalty — while still receiving zero STIFFNESS (S2), because a
/// linear map is curvature-free. The old zero-padded path would fail the first
/// assertion (linear lives in an unpenalized side block ⇒ zero tension).
#[test]
fn linear_function_receives_nonzero_tension_but_zero_stiffness() {
    let data = synthetic_data(200, 2, 21);
    let spec = duchon_spec(20, DuchonNullspaceOrder::Linear);
    let built = build(&data, &spec);

    let tension = built
        .tension
        .as_ref()
        .expect("tension (S1) penalty must exist and not be dropped to a zero block");

    let c_lin = coeff_for_target(&built.design, &linear_target(&data), "linear");
    let tension_energy = quad(tension, &c_lin);

    // A reference magnitude so "nonzero" is a real bound, not a float wobble:
    // the linear function's own L2 energy at the data sites.
    let g = linear_target(&data);
    let g_energy: f64 = g.iter().map(|v| v * v).sum();

    assert!(
        tension_energy > 1e-6 * g_energy,
        "HEADLINE FIX FAILED: a pure-linear function got ~zero TENSION energy \
         ({tension_energy:.3e}; reference L2 {g_energy:.3e}). The affine columns \
         must be INCLUDED in the slope (S1) penalty — not zero-padded into an \
         unpenalized side block."
    );

    // Curvature still ignores the linear part: a linear map is exactly flat.
    if let Some(stiffness) = built.stiffness.as_ref() {
        let stiffness_energy = quad(stiffness, &c_lin);
        // Compare against a quadratic's stiffness for scale.
        let c_quad = coeff_for_target(&built.design, &quadratic_target(&data), "quadratic");
        let quad_stiffness = quad(stiffness, &c_quad).max(1.0);
        assert!(
            stiffness_energy <= 1e-6 * quad_stiffness,
            "a linear function must have ~zero STIFFNESS: {stiffness_energy:.3e} \
             vs quadratic {quad_stiffness:.3e}"
        );
    }
}

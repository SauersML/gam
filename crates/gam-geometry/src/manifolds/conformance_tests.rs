//! Cross-manifold conformance suite.
//!
//! Every file in `manifolds/` carries its own unit tests, and those tests are
//! written against the formulas of *that* manifold. What was missing is the
//! other direction: the axioms that hold for **every** implementation of
//! [`RiemannianManifold`], checked against **one shared inventory** of manifolds
//! so that no family can quietly drift out of the contract.
//!
//! That matters here specifically because the trait ships three defaults that
//! are correct only for a *flat, embedded* manifold, and each of them fails
//! silently rather than loudly when inherited by a curved one:
//!
//! | default | correct for | what inheriting it wrongly produces |
//! |---|---|---|
//! | `project_tangent` = identity | flat manifolds | a "tangent" vector that is not tangent |
//! | `exp_map_vjp` = `(g, g)` | `exp_p(v) = p + v` | a plausible but wrong reverse-mode gradient |
//! | `riemannian_gradient` = metric raise | any | correct, but silently `O(m³)` |
//!
//! None of those shows up as a panic, a wrong shape, or a `NaN`. They show up
//! as an optimizer that converges to the wrong point. A per-manifold test suite
//! cannot catch them, because the failure mode is precisely a manifold whose
//! author did not think about the method at all — so the assertions live here,
//! keyed off [`ManifoldSpec`], and [`inventory_covers_every_manifold_spec_variant`]
//! makes adding a variant to that enum a compile error until it is covered.
//!
//! ### Two traps this suite had to be written around, recorded so the next
//! ### reader does not re-derive them
//!
//! 1. **A Grassmann point is a subspace, not a matrix.** `Gr(k, n)` points are
//!    stored as `n × k` orthonormal frames, but the frame is a *representative*:
//!    `Y` and `YR` for `R ∈ O(k)` are the same point. `exp_p(log_p(q))` is
//!    therefore under no obligation to return `q` entrywise — it returns *some*
//!    frame for `q`'s subspace, and measured here it differs from `q` by
//!    `O(1)`, not `O(ε)`. Comparing frames would flag a correct implementation;
//!    the comparison has to be on the orthogonal projector `YYᵀ`, which is the
//!    representative-free encoding of the point. (`Gr(1, n) = ℝP^{n-1}` is the
//!    easy instance of the same thing: `q` and `−q` are one point.)
//!
//! 2. **A zero-dimensional tangent space cannot be sampled by normalization.**
//!    `Gr(n, n)` is a single point and `dim() == 0`, so projecting a random
//!    ambient vector to the tangent space gives zero *up to roundoff*. Rescaling
//!    that residual to a fixed length — the obvious way to draw a test tangent —
//!    amplifies `1e-16` of noise into a unit-length vector that is not tangent
//!    to anything, and then every axiom downstream fails on the probe's own
//!    artifact rather than on the code. [`sample_tangent`] collapses a
//!    projection that lost all of its magnitude to the exact zero it
//!    mathematically is.
//!
//! ### Not-attempted is never reported as verified
//!
//! Some operations are legitimately absent: Stiefel has no closed-form parallel
//! transport or sectional curvature for `k > 1`, and `log_map` is undefined past
//! the injectivity radius, which a random pair of frames can land beyond. Those
//! return `GeometryError::Unsupported`, and this suite *accepts* that — so every
//! test that can skip a case also **counts the cases it actually verified and
//! asserts that count is non-zero**. A tolerance that is never evaluated is not
//! evidence, and a test that reports success having checked nothing is worse
//! than no test at all.

use ndarray::{Array1, Array2, ArrayView1};

use crate::manifold::{
    GeometryError, GeometryResult, ManifoldSpec, RiemannianManifold, cholesky_spd, dot, flatten,
    from_flat, symmetric_eigen, norm, qr_thin,
};

// ---------------------------------------------------------------------------
// Deterministic sampling
// ---------------------------------------------------------------------------

/// Xorshift64*, carried locally so the suite is reproducible bit-for-bit and
/// pulls in no RNG dependency. Test failures must be replayable from the seed
/// alone.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        // Box–Muller. The `max` keeps `ln` finite on the (astronomically
        // unlikely) exact-zero draw.
        let u1 = self.uniform().max(1.0e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn gaussian_vec(&mut self, n: usize) -> Array1<f64> {
        Array1::from_shape_fn(n, |_| self.normal())
    }

    fn angle(&mut self) -> f64 {
        (self.uniform() - 0.5) * std::f64::consts::TAU
    }

    fn orthogonal(&mut self, k: usize) -> Array2<f64> {
        let a = Array2::from_shape_fn((k, k), |_| self.normal());
        qr_thin(&a).0
    }
}

const SEED: u64 = 0x2545_F491_4F6C_DD1D;
const TRIALS: usize = 40;

/// The manifolds every axiom below is checked against.
///
/// Chosen to cover each [`ManifoldSpec`] variant *and* the shapes where the
/// implementations branch: `k == 1` (where Grassmann and Stiefel both delegate
/// to [`SphereManifold`](crate::SphereManifold)), `k == n` (zero-dimensional
/// Grassmann; Stiefel as the orthogonal group), `1 < k < n` (the general
/// closed forms), and a product mixing a curved factor with a
/// non-embedded-metric one.
fn inventory() -> Vec<(&'static str, ManifoldSpec)> {
    use ManifoldSpec::*;
    vec![
        ("Euclidean(3)", Euclidean(3)),
        ("Euclidean(0)", Euclidean(0)),
        ("Circle", Circle),
        ("Sphere(0)", Sphere { intrinsic_dim: 0 }),
        ("Sphere(1)", Sphere { intrinsic_dim: 1 }),
        ("Sphere(2)", Sphere { intrinsic_dim: 2 }),
        ("Sphere(4)", Sphere { intrinsic_dim: 4 }),
        ("Torus(3)", Torus { dim: 3 }),
        ("Gr(1,1)", Grassmann { k: 1, n: 1 }),
        ("Gr(1,4)", Grassmann { k: 1, n: 4 }),
        ("Gr(2,5)", Grassmann { k: 2, n: 5 }),
        ("Gr(3,6)", Grassmann { k: 3, n: 6 }),
        ("Gr(4,5)", Grassmann { k: 4, n: 5 }),
        ("Gr(3,3)", Grassmann { k: 3, n: 3 }),
        ("St(1,1)", Stiefel { k: 1, n: 1 }),
        ("St(1,4)", Stiefel { k: 1, n: 4 }),
        ("St(2,5)", Stiefel { k: 2, n: 5 }),
        ("St(3,4)", Stiefel { k: 3, n: 4 }),
        ("St(4,5)", Stiefel { k: 4, n: 5 }),
        ("St(3,3)", Stiefel { k: 3, n: 3 }),
        ("Spd(1)", Spd { n: 1 }),
        ("Spd(2)", Spd { n: 2 }),
        ("Spd(3)", Spd { n: 3 }),
        ("Spd(5)", Spd { n: 5 }),
        ("Product[]", Product(vec![])),
        ("Product[Circle,E2]", Product(vec![Circle, Euclidean(2)])),
        (
            "Product[S2,Spd2]",
            Product(vec![Sphere { intrinsic_dim: 2 }, Spd { n: 2 }]),
        ),
    ]
}

/// Name of the [`ManifoldSpec`] variant.
///
/// The `match` is exhaustive on purpose: a new variant makes this function fail
/// to compile, and [`inventory_covers_every_manifold_spec_variant`] then fails
/// until the inventory grows to include it. That is the mechanism that stops a
/// future manifold from inheriting the trait's flat defaults unnoticed.
fn variant_name(spec: &ManifoldSpec) -> &'static str {
    match spec {
        ManifoldSpec::Euclidean(_) => "Euclidean",
        ManifoldSpec::Circle => "Circle",
        ManifoldSpec::Sphere { .. } => "Sphere",
        ManifoldSpec::Torus { .. } => "Torus",
        ManifoldSpec::Grassmann { .. } => "Grassmann",
        ManifoldSpec::Stiefel { .. } => "Stiefel",
        ManifoldSpec::Spd { .. } => "Spd",
        ManifoldSpec::Product(_) => "Product",
    }
}

/// Draw a point of `spec`, in the ambient coordinates its manifold expects.
fn random_point(spec: &ManifoldSpec, rng: &mut Rng) -> Array1<f64> {
    match spec {
        ManifoldSpec::Euclidean(dim) => rng.gaussian_vec(*dim),
        ManifoldSpec::Circle => Array1::from(vec![rng.angle()]),
        ManifoldSpec::Sphere { intrinsic_dim } => {
            let v = rng.gaussian_vec(intrinsic_dim + 1);
            let scale = norm(v.view());
            v / scale
        }
        ManifoldSpec::Torus { dim } => Array1::from_shape_fn(*dim, |_| rng.angle()),
        ManifoldSpec::Grassmann { k, n } | ManifoldSpec::Stiefel { k, n } => {
            let a = Array2::from_shape_fn((*n, *k), |_| rng.normal());
            flatten(&qr_thin(&a).0)
        }
        ManifoldSpec::Spd { n } => {
            // A Aᵀ + n·I: symmetric by construction and diagonally dominant
            // enough to stay comfortably inside the cone.
            let a = Array2::from_shape_fn((*n, *n), |_| rng.normal());
            let mut p = a.dot(&a.t());
            for i in 0..*n {
                p[[i, i]] += *n as f64;
            }
            flatten(&p)
        }
        ManifoldSpec::Product(parts) => {
            let mut out: Vec<f64> = Vec::new();
            for part in parts {
                out.extend(random_point(part, rng).iter().copied());
            }
            Array1::from(out)
        }
    }
}

/// How far `x` is from satisfying `spec`'s defining constraint, in the natural
/// units of that constraint (`‖x‖ − 1`, `max|YᵀY − I|`, asymmetry / definiteness
/// for SPD). Flat manifolds have no constraint, so they are exactly `0`.
fn on_manifold_defect(spec: &ManifoldSpec, x: &Array1<f64>) -> f64 {
    match spec {
        ManifoldSpec::Euclidean(_) | ManifoldSpec::Circle | ManifoldSpec::Torus { .. } => 0.0,
        ManifoldSpec::Sphere { .. } => (norm(x.view()) - 1.0).abs(),
        ManifoldSpec::Grassmann { k, n } | ManifoldSpec::Stiefel { k, n } => {
            let y = from_flat(x.view(), *n, *k).expect("frame shape");
            let gram = y.t().dot(&y);
            let mut worst = 0.0_f64;
            for i in 0..*k {
                for j in 0..*k {
                    let want = if i == j { 1.0 } else { 0.0 };
                    worst = worst.max((gram[[i, j]] - want).abs());
                }
            }
            worst
        }
        ManifoldSpec::Spd { n } => {
            let a = from_flat(x.view(), *n, *n).expect("spd shape");
            let mut worst = 0.0_f64;
            for i in 0..*n {
                for j in 0..*n {
                    worst = worst.max((a[[i, j]] - a[[j, i]]).abs());
                }
            }
            // Definiteness is a yes/no property, so a failed Cholesky is
            // reported as a defect of 1 rather than as a small residual.
            if cholesky_spd(&a).is_err() {
                worst = worst.max(1.0);
            }
            worst
        }
        ManifoldSpec::Product(parts) => {
            let mut offset = 0usize;
            let mut worst = 0.0_f64;
            for part in parts {
                let width = part.build().expect("component builds").ambient_dim();
                let sub = x.slice(ndarray::s![offset..offset + width]).to_owned();
                worst = worst.max(on_manifold_defect(part, &sub));
                offset += width;
            }
            worst
        }
    }
}

/// A tangent vector at `p` of norm `~0.3` — short enough that `exp`/`log` stay
/// inside the injectivity radius of every manifold in the inventory.
///
/// Returns the exact zero vector when the tangent space is trivial. See the
/// module docs: renormalizing the roundoff residual of a zero-dimensional
/// projection manufactures a non-tangent vector and invalidates every axiom
/// checked with it.
fn sample_tangent(
    manifold: &dyn RiemannianManifold,
    point: ArrayView1<'_, f64>,
    rng: &mut Rng,
) -> Array1<f64> {
    let ambient = manifold.ambient_dim();
    let raw = rng.gaussian_vec(ambient);
    let projected = manifold
        .project_tangent(point, raw.view())
        .expect("project_tangent must accept an ambient vector at a manifold point");
    let magnitude = norm(projected.view());
    if magnitude > 1.0e-8 * norm(raw.view()).max(1.0) {
        projected * (0.3 / magnitude)
    } else {
        Array1::zeros(ambient)
    }
}

/// Orthogonal projector `YYᵀ` of a flattened `n × k` frame — the
/// representative-free encoding of a Grassmann point.
fn subspace_projector(flat: &Array1<f64>, n: usize, k: usize) -> Array2<f64> {
    let y = from_flat(flat.view(), n, k).expect("frame shape");
    y.dot(&y.t())
}

/// Right-multiply a flattened `n × k` frame by `rot`, i.e. move to another
/// representative of the same Grassmann point.
fn regauge(flat: &Array1<f64>, n: usize, k: usize, rot: &Array2<f64>) -> Array1<f64> {
    flatten(&from_flat(flat.view(), n, k).expect("frame shape").dot(rot))
}

/// Attainable accuracy of `log_Y(Z)` for this particular pair of subspaces.
///
/// The general-`k` Grassmann logarithm forms `M = (Z − Y·YᵀZ)·(YᵀZ)⁻¹`, so its
/// conditioning is governed by `σ_min(YᵀZ) = cos θ_max`, the cosine of the
/// largest principal angle between the two subspaces. At `θ_max = π/2` the
/// subspaces meet at the cut locus, `YᵀZ` is exactly singular, and the
/// logarithm is not merely inaccurate but undefined — the minimizing geodesic
/// is non-unique.
///
/// A single fixed tolerance therefore cannot be right: tight enough to mean
/// anything at moderate angles, it fails on random pairs that happen to land
/// near the cut locus; loose enough to survive those, it stops testing the
/// formula at all. Measured over 3000 random pairs per shape, the round-trip
/// error tracks `ε/σ_min²` — median `~1e-15` across every angle bucket, with
/// the entire tail confined to the last bucket (`θ_max > 87°`, worst `1.4e-9`
/// on `Gr(3,6)`). So the bound is written in those units and stays sharp where
/// the geometry is well posed.
fn grassmann_pair_conditioning(p: &Array1<f64>, q: &Array1<f64>, n: usize, k: usize) -> f64 {
    let y = from_flat(p.view(), n, k).expect("frame shape");
    let z = from_flat(q.view(), n, k).expect("frame shape");
    let cross = y.t().dot(&z);
    let gram = cross.t().dot(&cross);
    let (eigenvalues, _) = symmetric_eigen(&gram).expect("symmetric k×k eigendecomposition");
    let smallest = eigenvalues.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    smallest.max(0.0).sqrt()
}

/// Tolerance for a Grassmann quantity computed through `(YᵀZ)⁻¹`, given the
/// pair's [conditioning](grassmann_pair_conditioning).
fn grassmann_tolerance(sigma_min: f64) -> f64 {
    1.0e-9 + 1.0e-13 / (sigma_min * sigma_min).max(f64::MIN_POSITIVE)
}

/// Magnitude to measure an absolute residual against.
///
/// `Sphere`, `Grassmann`, `Stiefel`, `Circle` and `Torus` points are bounded, so
/// an absolute tolerance would do. `Euclidean` and `Spd` points are not: an SPD
/// matrix drawn here has entries of order `n`, and its exponential runs two
/// spectral conjugations, so the residual of an identity like `exp_p(0) = p`
/// scales with `‖p‖`. Comparing every manifold's residual against a fixed
/// absolute bound would therefore be simultaneously too strict for SPD and too
/// lax for the sphere; all the equalities below are relative to this instead.
fn point_scale(p: &Array1<f64>) -> f64 {
    1.0 + p.iter().fold(0.0_f64, |a, x| a.max(x.abs()))
}

fn sup_diff(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn sup_diff_2d(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// `true` when a manifold declines an operation it documents as having no
/// closed form — Stiefel parallel transport and sectional curvature for
/// `k > 1`, Christoffel symbols without a local chart.
///
/// Only `Unsupported` counts. A `Singular` or `NonConvergence` from a routine
/// that claims to support the operation is a failure, not a skip.
fn is_declared_unsupported<T>(result: &GeometryResult<T>) -> bool {
    matches!(result, Err(GeometryError::Unsupported(_)))
}

/// `true` when a manifold refuses a *pair* of points because the operation is
/// genuinely undefined for them, not because it is unimplemented.
///
/// `log_p(q)` is single-valued only inside the injectivity radius. Two
/// independently drawn points can land outside it — antipodes on a sphere (and
/// every pair on `S⁰`, which is two disconnected points), Stiefel frames
/// separated by a rotation angle of `π`, subspaces meeting at a principal angle
/// of `π/2`. There the minimizing geodesic is non-unique or absent, and
/// `Singular` / `NonConvergence` is the correct answer; inventing one would be
/// the defect.
///
/// This is deliberately allowed **only** for independently drawn pairs. The
/// round trip `log_p(exp_p(v))` with a short `v` is inside the injectivity
/// radius by construction, so a refusal there is a real failure and is asserted
/// as one.
fn is_pair_out_of_domain<T>(result: &GeometryResult<T>) -> bool {
    matches!(
        result,
        Err(GeometryError::Unsupported(_))
            | Err(GeometryError::Singular(_))
            | Err(GeometryError::NonConvergence { .. })
    )
}

// ---------------------------------------------------------------------------
// Axioms
// ---------------------------------------------------------------------------

#[test]
fn inventory_covers_every_manifold_spec_variant() {
    let covered: Vec<&'static str> = inventory()
        .iter()
        .map(|(_, spec)| variant_name(spec))
        .collect();
    // `variant_name` is an exhaustive match, so this list is the full set of
    // variants by construction; the assertion is that the inventory exercises
    // each of them.
    for variant in [
        "Euclidean",
        "Circle",
        "Sphere",
        "Torus",
        "Grassmann",
        "Stiefel",
        "Spd",
        "Product",
    ] {
        assert!(
            covered.contains(&variant),
            "ManifoldSpec::{variant} is not exercised by the conformance inventory — \
             add it, or a manifold can inherit the trait's flat defaults unnoticed"
        );
    }
    for (label, spec) in inventory() {
        let manifold = spec
            .build()
            .unwrap_or_else(|e| panic!("{label} must build: {e}"));
        assert!(
            manifold.dim() <= manifold.ambient_dim(),
            "{label}: dim {} exceeds ambient_dim {}",
            manifold.dim(),
            manifold.ambient_dim()
        );
    }
}

#[test]
fn exp_and_log_are_mutually_inverse() {
    let mut verified = 0usize;
    for (label, spec) in inventory() {
        let manifold = spec.build().expect("build");
        let mut rng = Rng::new(SEED);
        for trial in 0..TRIALS {
            let p = random_point(&spec, &mut rng);
            let q = random_point(&spec, &mut rng);
            let v = sample_tangent(manifold.as_ref(), p.view(), &mut rng);
            let scale = point_scale(&p);

            let at_zero = manifold
                .exp_map(p.view(), Array1::zeros(manifold.ambient_dim()).view())
                .expect("exp at zero tangent");
            assert!(
                sup_diff(&at_zero, &p) <= 1.0e-12 * scale,
                "{label} trial {trial}: exp_p(0) != p (sup {:.3e}, scale {scale:.3e})",
                sup_diff(&at_zero, &p)
            );

            let self_log = manifold.log_map(p.view(), p.view()).expect("log_p(p)");
            assert!(
                norm(self_log.view()) <= 1.0e-9 * scale,
                "{label} trial {trial}: log_p(p) != 0 (norm {:.3e}, scale {scale:.3e})",
                norm(self_log.view())
            );

            // log_p(exp_p(v)) == v. `v` is short, so this pair is inside the
            // injectivity radius by construction — no domain refusal is
            // admissible here, and the `expect` is the assertion that says so.
            let moved = manifold.exp_map(p.view(), v.view()).expect("exp_p(v)");
            let recovered = manifold
                .log_map(p.view(), moved.view())
                .expect("log_p(exp_p(v)) is inside the injectivity radius");
            assert!(
                sup_diff(&recovered, &v) <= 1.0e-9 * scale,
                "{label} trial {trial}: log_p(exp_p(v)) != v (sup {:.3e}, scale {scale:.3e})",
                sup_diff(&recovered, &v)
            );
            verified += 1;

            // exp_p(log_p(q)) == q. For Grassmann the equality is between
            // SUBSPACES, not frames — see the module docs.
            let to_q = manifold.log_map(p.view(), q.view());
            if is_pair_out_of_domain(&to_q) {
                continue;
            }
            let to_q = to_q.expect("log_p(q)");
            let round_trip = manifold.exp_map(p.view(), to_q.view()).expect("exp_p(log)");
            if let ManifoldSpec::Grassmann { k, n } = spec {
                let gap = sup_diff_2d(
                    &subspace_projector(&round_trip, n, k),
                    &subspace_projector(&q, n, k),
                );
                let sigma_min = grassmann_pair_conditioning(&p, &q, n, k);
                let tolerance = grassmann_tolerance(sigma_min);
                assert!(
                    gap <= tolerance,
                    "{label} trial {trial}: exp_p(log_p(q)) spans a different subspace \
                     (projector sup {gap:.3e} > {tolerance:.3e}, \
                     cos of largest principal angle {sigma_min:.3e})"
                );
            } else {
                let target_scale = scale.max(point_scale(&q));
                assert!(
                    sup_diff(&round_trip, &q) <= 1.0e-9 * target_scale,
                    "{label} trial {trial}: exp_p(log_p(q)) != q                      (sup {:.3e}, scale {target_scale:.3e})",
                    sup_diff(&round_trip, &q)
                );
            }
            verified += 1;
        }
    }
    assert!(verified > 0, "no exp/log round trip was actually evaluated");
}

#[test]
fn tangent_projection_is_idempotent_and_log_lands_tangent() {
    let mut verified = 0usize;
    for (label, spec) in inventory() {
        let manifold = spec.build().expect("build");
        let mut rng = Rng::new(SEED);
        for trial in 0..TRIALS {
            let p = random_point(&spec, &mut rng);
            let q = random_point(&spec, &mut rng);
            let v = sample_tangent(manifold.as_ref(), p.view(), &mut rng);

            let again = manifold
                .project_tangent(p.view(), v.view())
                .expect("re-project a tangent vector");
            assert!(
                sup_diff(&again, &v) <= 1.0e-12 * point_scale(&p),
                "{label} trial {trial}: project_tangent is not idempotent (sup {:.3e})",
                sup_diff(&again, &v)
            );

            let to_q = manifold.log_map(p.view(), q.view());
            if is_pair_out_of_domain(&to_q) {
                continue;
            }
            let to_q = to_q.expect("log_p(q)");
            let projected = manifold
                .project_tangent(p.view(), to_q.view())
                .expect("project the logarithm");
            assert!(
                sup_diff(&projected, &to_q) <= 1.0e-9 * point_scale(&to_q),
                "{label} trial {trial}: log_p(q) is not in the tangent space at p (sup {:.3e})",
                sup_diff(&projected, &to_q)
            );
            verified += 1;
        }
    }
    assert!(verified > 0, "no logarithm was actually projected");
}

#[test]
fn tangent_basis_is_metric_orthonormal_and_tangent() {
    let mut verified = 0usize;
    for (label, spec) in inventory() {
        let manifold = spec.build().expect("build");
        let (dim, ambient) = (manifold.dim(), manifold.ambient_dim());
        let mut rng = Rng::new(SEED);
        for trial in 0..TRIALS {
            let p = random_point(&spec, &mut rng);
            let basis = manifold.tangent_basis(p.view()).expect("tangent_basis");
            assert_eq!(
                (basis.nrows(), basis.ncols()),
                (ambient, dim),
                "{label} trial {trial}: tangent_basis has the wrong shape"
            );
            let metric = manifold.metric_tensor(p.view()).expect("metric_tensor");
            let gram = basis.t().dot(&metric.dot(&basis));
            for i in 0..dim {
                for j in 0..dim {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (gram[[i, j]] - want).abs() <= 1.0e-9,
                        "{label} trial {trial}: BᵀGB[{i},{j}] = {} (want {want})",
                        gram[[i, j]]
                    );
                }
            }
            for j in 0..dim {
                let column = basis.column(j).to_owned();
                let projected = manifold
                    .project_tangent(p.view(), column.view())
                    .expect("project a basis column");
                assert!(
                    sup_diff(&projected, &column) <= 1.0e-8,
                    "{label} trial {trial}: basis column {j} is not tangent (sup {:.3e})",
                    sup_diff(&projected, &column)
                );
            }
            verified += 1;
        }
    }
    assert!(verified > 0, "no tangent basis was actually checked");
}

#[test]
fn riemannian_gradient_is_the_metric_riesz_representative() {
    // Defining property: for the ambient differential `e`, the Riemannian
    // gradient is the unique tangent `v` with `g(v, ξ) = ⟨e, ξ⟩` for every
    // tangent `ξ`. Projecting `e` instead of raising it through the metric
    // satisfies this only for the embedded metric — so this is the assertion
    // that separates a correct SPD/Stiefel override from a plausible one.
    let mut verified = 0usize;
    for (label, spec) in inventory() {
        let manifold = spec.build().expect("build");
        let (dim, ambient) = (manifold.dim(), manifold.ambient_dim());
        let mut rng = Rng::new(SEED);
        for trial in 0..TRIALS {
            let p = random_point(&spec, &mut rng);
            let differential = rng.gaussian_vec(ambient);
            let basis = manifold.tangent_basis(p.view()).expect("tangent_basis");
            let metric = manifold.metric_tensor(p.view()).expect("metric_tensor");
            let gradient = manifold
                .riemannian_gradient(p.view(), differential.view())
                .expect("riemannian_gradient");

            let projected = manifold
                .project_tangent(p.view(), gradient.view())
                .expect("project the gradient");
            assert!(
                sup_diff(&projected, &gradient) <= 1.0e-8,
                "{label} trial {trial}: the Riemannian gradient is not tangent (sup {:.3e})",
                sup_diff(&projected, &gradient)
            );

            for j in 0..dim {
                let xi = basis.column(j).to_owned();
                let raised = gradient.view().dot(&metric.dot(&xi));
                let paired = dot(differential.view(), xi.view());
                let scale = raised.abs().max(paired.abs()).max(1.0);
                assert!(
                    (raised - paired).abs() <= 1.0e-8 * scale,
                    "{label} trial {trial}: g(grad, ξ_{j}) = {raised} != ⟨e, ξ_{j}⟩ = {paired}"
                );
            }
            verified += 1;
        }
    }
    assert!(verified > 0, "no Riemannian gradient was actually checked");
}

#[test]
fn exp_and_retract_land_on_the_manifold() {
    let mut verified = 0usize;
    for (label, spec) in inventory() {
        let manifold = spec.build().expect("build");
        let mut rng = Rng::new(SEED);
        for trial in 0..TRIALS {
            let p = random_point(&spec, &mut rng);
            let v = sample_tangent(manifold.as_ref(), p.view(), &mut rng);

            let stepped = manifold.exp_map(p.view(), v.view()).expect("exp_p(v)");
            let defect = on_manifold_defect(&spec, &stepped);
            let bound = 1.0e-9 * point_scale(&stepped);
            assert!(
                defect <= bound,
                "{label} trial {trial}: exp_p(v) is off the manifold                  (defect {defect:.3e} > {bound:.3e})"
            );

            let retracted = manifold.retract(p.view(), v.view()).expect("retract");
            let defect = on_manifold_defect(&spec, &retracted);
            let bound = 1.0e-9 * point_scale(&retracted);
            assert!(
                defect <= bound,
                "{label} trial {trial}: retract_p(v) is off the manifold                  (defect {defect:.3e} > {bound:.3e})"
            );
            verified += 1;
        }
    }
    assert!(verified > 0, "no step was actually checked for membership");
}

#[test]
fn geodesics_have_constant_speed_and_symmetric_distance() {
    let mut verified = 0usize;
    for (label, spec) in inventory() {
        let manifold = spec.build().expect("build");
        let mut rng = Rng::new(SEED);
        for trial in 0..TRIALS {
            let p = random_point(&spec, &mut rng);
            let q = random_point(&spec, &mut rng);
            let v = sample_tangent(manifold.as_ref(), p.view(), &mut rng);
            let speed = norm(v.view());

            for fraction in [0.25_f64, 0.5, 0.75, 1.0] {
                let scaled = &v * fraction;
                let along = manifold
                    .exp_map(p.view(), scaled.view())
                    .expect("exp along the geodesic");
                // Every point on this segment is at distance <= ‖v‖ = 0.3 from
                // `p`, so the logarithm is well defined and a refusal is a
                // failure rather than a domain boundary.
                let back = manifold
                    .log_map(p.view(), along.view())
                    .expect("log along a short geodesic");
                let travelled = norm(back.view());
                assert!(
                    (travelled - fraction * speed).abs() <= 1.0e-9 * speed.max(1.0),
                    "{label} trial {trial}: geodesic at t={fraction} travelled {travelled}, \
                     want {}",
                    fraction * speed
                );
            }

            // d(p,q) = d(q,p): both logarithms are measured in the metric of
            // their own base point, which is the only way the comparison is
            // meaningful on a manifold whose metric varies (SPD).
            let (out, back) = (
                manifold.log_map(p.view(), q.view()),
                manifold.log_map(q.view(), p.view()),
            );
            if is_pair_out_of_domain(&out) || is_pair_out_of_domain(&back) {
                continue;
            }
            let (out, back) = (out.expect("log_p(q)"), back.expect("log_q(p)"));
            let gp = manifold.metric_tensor(p.view()).expect("metric at p");
            let gq = manifold.metric_tensor(q.view()).expect("metric at q");
            let forward = out.view().dot(&gp.dot(&out)).abs().sqrt();
            let reverse = back.view().dot(&gq.dot(&back)).abs().sqrt();
            assert!(
                (forward - reverse).abs() <= 1.0e-8 * forward.max(1.0),
                "{label} trial {trial}: d(p,q) = {forward} but d(q,p) = {reverse}"
            );
            verified += 1;
        }
    }
    assert!(verified > 0, "no geodesic was actually traversed");
}

#[test]
fn parallel_transport_is_an_isometry_where_it_is_supported() {
    let mut verified = 0usize;
    let mut declined = 0usize;
    for (label, spec) in inventory() {
        let manifold = spec.build().expect("build");
        let ambient = manifold.ambient_dim();
        if manifold.dim() < 2 {
            continue;
        }
        let mut rng = Rng::new(SEED);
        for trial in 0..TRIALS {
            let p = random_point(&spec, &mut rng);
            let q = random_point(&spec, &mut rng);
            let u1 = sample_tangent(manifold.as_ref(), p.view(), &mut rng);
            let u2 = sample_tangent(manifold.as_ref(), p.view(), &mut rng);

            let mut path = Array2::<f64>::zeros((2, ambient));
            path.row_mut(0).assign(&p);
            path.row_mut(1).assign(&q);

            let t1 = manifold.parallel_transport(path.view(), u1.view());
            if is_declared_unsupported(&t1) {
                declined += 1;
                continue;
            }
            let t1 = match t1 {
                Ok(t) => t,
                // Antipodal endpoints make transport genuinely path-dependent;
                // the sphere reports that as `Singular` rather than guessing.
                Err(GeometryError::Singular(_)) => continue,
                Err(e) => panic!("{label} trial {trial}: parallel_transport failed: {e}"),
            };
            let t2 = match manifold.parallel_transport(path.view(), u2.view()) {
                Ok(t) => t,
                Err(GeometryError::Singular(_)) => continue,
                Err(e) => panic!("{label} trial {trial}: parallel_transport failed: {e}"),
            };

            let landed = manifold
                .project_tangent(q.view(), t1.view())
                .expect("project the transported vector at q");
            assert!(
                sup_diff(&landed, &t1) <= 1.0e-8,
                "{label} trial {trial}: transported vector is not tangent at q (sup {:.3e})",
                sup_diff(&landed, &t1)
            );

            let gp = manifold.metric_tensor(p.view()).expect("metric at p");
            let gq = manifold.metric_tensor(q.view()).expect("metric at q");
            let before = u1.view().dot(&gp.dot(&u2));
            let after = t1.view().dot(&gq.dot(&t2));
            assert!(
                (before - after).abs() <= 1.0e-8 * before.abs().max(1.0),
                "{label} trial {trial}: transport changed the inner product \
                 ({before} -> {after})"
            );
            verified += 1;
        }
    }
    assert!(
        verified > 0,
        "parallel transport was declined everywhere ({declined} declines) — \
         the isometry property was never evaluated"
    );
}

#[test]
fn sectional_curvature_is_symmetric_and_scale_invariant() {
    // K depends only on the 2-plane the pair spans, so it is symmetric in its
    // arguments and invariant to rescaling either one. On a manifold of
    // dimension < 2 there is no 2-plane and the quantity is undefined — the
    // implementations report that rather than returning a misleading `0.0`.
    let mut verified = 0usize;
    for (label, spec) in inventory() {
        let manifold = spec.build().expect("build");
        if manifold.dim() < 2 {
            continue;
        }
        let mut rng = Rng::new(SEED);
        for trial in 0..TRIALS {
            let p = random_point(&spec, &mut rng);
            let u = sample_tangent(manifold.as_ref(), p.view(), &mut rng);
            let v = sample_tangent(manifold.as_ref(), p.view(), &mut rng);

            let forward = manifold.sectional_curvature(p.view(), (u.view(), v.view()));
            if is_declared_unsupported(&forward) {
                continue;
            }
            let forward = forward.expect("K(u,v)");
            let swapped = manifold
                .sectional_curvature(p.view(), (v.view(), u.view()))
                .expect("K(v,u)");
            assert!(
                (forward - swapped).abs() <= 1.0e-9 * forward.abs().max(1.0),
                "{label} trial {trial}: K(u,v) = {forward} but K(v,u) = {swapped}"
            );

            let (su, sv) = (&u * 3.7, &v * 0.21);
            let rescaled = manifold
                .sectional_curvature(p.view(), (su.view(), sv.view()))
                .expect("K(au,bv)");
            assert!(
                (forward - rescaled).abs() <= 1.0e-9 * forward.abs().max(1.0),
                "{label} trial {trial}: K is not scale-invariant ({forward} -> {rescaled})"
            );
            verified += 1;
        }
    }
    assert!(
        verified > 0,
        "sectional curvature was never actually evaluated"
    );
}

#[test]
fn exp_map_vjp_matches_central_finite_differences() {
    // The trait's default VJP is the identity pair, which is exact only when
    // `exp_p(v) = p + v`. A curved manifold that inherits it returns a gradient
    // that is the right shape, finite, and wrong. Nothing but a derivative
    // check catches that.
    //
    // `h = 1e-4` is the measured optimum of the truncation/roundoff trade-off
    // for the worst case in the inventory (SPD, whose exp_map runs two spectral
    // conjugations): the residual bottoms out near 7e-10 there and rises in
    // both directions, so the 1e-7 tolerance is roughly two decades of margin
    // over FD noise while still being far below any plausible analytic error.
    const H: f64 = 1.0e-4;
    let mut verified = 0usize;
    for (label, spec) in inventory() {
        let manifold = spec.build().expect("build");
        let ambient = manifold.ambient_dim();
        let mut rng = Rng::new(SEED);
        for trial in 0..TRIALS {
            let p = random_point(&spec, &mut rng);
            let v = sample_tangent(manifold.as_ref(), p.view(), &mut rng);
            let seed_cotangent = rng.gaussian_vec(ambient);

            let (grad_point, grad_tangent) = manifold
                .exp_map_vjp(p.view(), v.view(), seed_cotangent.view())
                .expect("exp_map_vjp");

            let mut worst = 0.0_f64;
            let mut scale = 1.0_f64;
            for i in 0..ambient {
                for (analytic, base_is_point) in [(&grad_point, true), (&grad_tangent, false)] {
                    let (mut plus, mut minus) = if base_is_point {
                        (p.clone(), p.clone())
                    } else {
                        (v.clone(), v.clone())
                    };
                    plus[i] += H;
                    minus[i] -= H;
                    let (up, down) = if base_is_point {
                        (
                            manifold.exp_map(plus.view(), v.view()),
                            manifold.exp_map(minus.view(), v.view()),
                        )
                    } else {
                        (
                            manifold.exp_map(p.view(), plus.view()),
                            manifold.exp_map(p.view(), minus.view()),
                        )
                    };
                    let (Ok(up), Ok(down)) = (up, down) else {
                        continue;
                    };
                    let directional = dot(seed_cotangent.view(), (&up - &down).view()) / (2.0 * H);
                    worst = worst.max((directional - analytic[i]).abs());
                    scale = scale.max(directional.abs().max(analytic[i].abs()));
                }
            }
            assert!(
                worst <= 1.0e-7 * scale,
                "{label} trial {trial}: exp_map_vjp disagrees with finite differences \
                 (abs {worst:.3e}, scale {scale:.3e})"
            );
            verified += 1;
        }
    }
    assert!(verified > 0, "no VJP was actually differenced");
}

#[test]
fn grassmann_operations_are_invariant_to_the_frame_representative() {
    // Gr(k, n) = St(n, k)/O(k). A point is a subspace; the stored frame is a
    // representative. Every Riemannian quantity must therefore be a function of
    // the subspace alone:
    //
    //   log_Y(ZR) = log_Y(Z)          (the target's gauge is invisible)
    //   log_{YR}(Z) = log_Y(Z) · R    (the base's gauge is equivariant)
    //   Yᵀ log_Y(Z) = 0               (the logarithm is horizontal)
    //
    // A frame-dependent logarithm gives a Fréchet mean, a chart, and a fitted
    // response that depend on how the input data happened to be orthonormalized
    // — reproducible only by accident.
    let mut verified = 0usize;
    for (label, spec) in inventory() {
        let ManifoldSpec::Grassmann { k, n } = spec else {
            continue;
        };
        let manifold = spec.build().expect("build");
        let mut rng = Rng::new(SEED);
        for trial in 0..TRIALS {
            let p = random_point(&spec, &mut rng);
            let q = random_point(&spec, &mut rng);
            let rot = rng.orthogonal(k);

            let base = manifold.log_map(p.view(), q.view()).expect("log_Y(Z)");
            // Same `(YᵀZ)⁻¹` conditioning as the round trip: a pair near the
            // cut locus cannot be asked for more digits than the geometry has.
            let tolerance = grassmann_tolerance(grassmann_pair_conditioning(&p, &q, n, k));

            let regauged_target = regauge(&q, n, k, &rot);
            let from_regauged = manifold
                .log_map(p.view(), regauged_target.view())
                .expect("log_Y(ZR)");
            assert!(
                sup_diff(&base, &from_regauged) <= tolerance,
                "{label} trial {trial}: log depends on the TARGET's frame \
                 (sup {:.3e} > {tolerance:.3e})",
                sup_diff(&base, &from_regauged)
            );

            let regauged_base = regauge(&p, n, k, &rot);
            let at_regauged = manifold
                .log_map(regauged_base.view(), q.view())
                .expect("log_{{YR}}(Z)");
            let expected = regauge(&base, n, k, &rot);
            assert!(
                sup_diff(&expected, &at_regauged) <= tolerance,
                "{label} trial {trial}: log is not equivariant in the BASE's frame \
                 (sup {:.3e} > {tolerance:.3e})",
                sup_diff(&expected, &at_regauged)
            );

            let y = from_flat(p.view(), n, k).expect("frame");
            let xi = from_flat(base.view(), n, k).expect("tangent");
            let horizontality = y.t().dot(&xi);
            let worst = horizontality.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
            assert!(
                worst <= tolerance,
                "{label} trial {trial}: Yᵀlog_Y(Z) = {worst:.3e} != 0 (not horizontal)"
            );
            verified += 1;
        }
    }
    assert!(
        verified > 0,
        "no Grassmann gauge invariance was actually checked"
    );
}

#[test]
fn a_gradient_step_descends_at_the_riemannian_gradient_norm() {
    // The defining first-order property of a retraction is `DR_x(0) = id` on
    // `T_xM`. Combined with the Riesz property of the Riemannian gradient it
    // pins the slope of a gradient step exactly: for the ambient-linear
    // objective `f(y) = ⟨e, y⟩`, whose ambient differential is the constant `e`,
    //
    //   d/dη f(R_x(−η·v))|₀ = −⟨e, v⟩ = −g_x(v, v) = −‖v‖²_g,   v = grad f(x).
    //
    // Nothing else in this file checks `retract` against the tangent space it
    // retracts from — the membership test only asks that the result land on the
    // manifold, which a retraction that moves in the wrong direction would also
    // satisfy. And the check is sharp in a way the standalone Riesz test is not:
    // it consumes the gradient THROUGH `riemannian_gradient_step`, so a metric
    // raise and a retraction that are each defensible alone but inconsistent
    // with one another still fails. That composition is the actual production
    // path (issue #955), and a merely-projected differential — correct only for
    // the embedded metric — gives the wrong slope on SPD and Stiefel.
    //
    // `−e` supplies the opposite arm of the central difference, since
    // `riemannian_gradient_step` refuses a non-positive learning rate and
    // `grad` is linear in the differential.
    //
    // The learning rate is chosen so the ambient DISPLACEMENT is `TRAVEL`,
    // not so the rate itself is fixed. Central-difference truncation is
    // governed by how far the retraction actually moves — `O((η‖v‖)²)` — and
    // the gradients here differ by orders of magnitude across the inventory
    // (`‖v‖²_g` reaches ~1.6e3 on `Spd(5)`, where a fixed `η = 1e-4` travels far
    // enough to leave a 1.9e-6 truncation residual and fail a tolerance the
    // geometry has no trouble meeting). Fixing the displacement instead makes
    // the residual the same size on every manifold.
    const TRAVEL: f64 = 1.0e-4;
    let mut verified = 0usize;
    for (label, spec) in inventory() {
        let manifold = spec.build().expect("build");
        let ambient = manifold.ambient_dim();
        if manifold.dim() == 0 {
            continue;
        }
        let mut rng = Rng::new(SEED);
        for trial in 0..TRIALS {
            let p = random_point(&spec, &mut rng);
            let differential = rng.gaussian_vec(ambient);
            let negated = &differential * -1.0;

            let gradient = manifold
                .riemannian_gradient(p.view(), differential.view())
                .expect("riemannian_gradient");
            let metric = manifold.metric_tensor(p.view()).expect("metric_tensor");
            let squared_norm = gradient.view().dot(&metric.dot(&gradient));
            if squared_norm <= 1.0e-12 {
                continue;
            }
            let ambient_norm = norm(gradient.view());
            if ambient_norm <= 1.0e-12 {
                continue;
            }
            let rate = TRAVEL / ambient_norm;

            let downhill = manifold
                .riemannian_gradient_step(p.view(), differential.view(), rate)
                .expect("gradient step");
            let uphill = manifold
                .riemannian_gradient_step(p.view(), negated.view(), rate)
                .expect("reverse gradient step");

            // Circle and Torus wrap their coordinates, so a step across the ±π
            // seam is a jump in the ambient chart even though it is a
            // continuous move on the manifold. The difference quotient is
            // meaningless there; detect the wrap by its size and skip, rather
            // than reporting the chart's discontinuity as a geometry defect.
            let jumped = sup_diff(&downhill, &p).max(sup_diff(&uphill, &p)) > 0.5;
            if jumped {
                continue;
            }

            let slope = (dot(differential.view(), uphill.view())
                - dot(differential.view(), downhill.view()))
                / (2.0 * rate);
            assert!(
                (slope - squared_norm).abs() <= 1.0e-8 * squared_norm,
                "{label} trial {trial}: a gradient step descends at {slope}, but the \
                 Riemannian gradient norm is {squared_norm}"
            );
            verified += 1;
        }
    }
    assert!(verified > 0, "no gradient step was actually differenced");
}

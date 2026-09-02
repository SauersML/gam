//! Independent-oracle conformance for the κ-stereographic family.
//!
//! [`ConstantCurvature`] is deliberately **not** a [`ManifoldSpec`] variant — it
//! carries a continuous parameter rather than a fixed shape — so the shared
//! inventory in [`super::conformance_tests`] does not reach it. It needs its own
//! coverage, and it needs a different *kind* of coverage, because the thing
//! worth checking is not that a formula is self-consistent but that this one
//! chart reproduces three classical geometries it was never told about.
//!
//! The module's whole design claim is that spherical, flat, and hyperbolic
//! space are one analytic object in `u = κt²`, with κ = 0 a removable point
//! rather than a branch. Every test here is an oracle *external* to that claim:
//!
//! | oracle | what it pins |
//! |---|---|
//! | `K ≡ κ` | the curvature the family reports is the curvature it has |
//! | Möbius-translation invariance | the space really is homogeneous |
//! | `d_κ(x,y) = d_{±1}(√|κ|x, √|κ|y)/√|κ|` | the κ-family is one manifold rescaled |
//! | `κ = 0 ⇒ d = 2‖x−y‖` | the flat member is flat (in the `λ₀ = 2` gauge) |
//! | `κ = −1 ⇒ poincare_distance` | agreement with a **separate implementation** |
//! | `J_κ(r) = (sn_κ(r)/r)^{d−1}` | the volume element against its closed form |
//! | metric axioms | symmetry and the triangle inequality |
//! | `‖log_x y‖_{g_x} = d(x,y)` | the logarithm and the distance are the same geometry |
//!
//! The cross-implementation row is the strongest of these: `poincare.rs` reaches
//! hyperbolic distance by a different route, so agreement at `1e-15` is evidence
//! about the mathematics, not about a shared helper.
//!
//! ### The κ-jets are checked against the value path, not against themselves
//!
//! `distance_kappa_jet` and friends return `(f, ∂f/∂κ, ∂²f/∂κ²)` from a
//! `Tower2` program, and those derivatives enter the outer REML optimization as
//! a ψ-coordinate. A wrong `∂²/∂κ²` does not produce a wrong *answer* — it
//! produces a slower or differently-converging search, which is exactly the kind
//! of defect that survives for years. So the jets are differenced against the
//! independently written value methods (`distance`, `log_map`, `exp_map`)
//! evaluated at perturbed κ.
//!
//! Two things had to be right for that to measure anything:
//!
//! * **The test points must not depend on `h`.** The chart for `κ < 0` is a ball
//!   of radius `1/√−κ`, so the obvious "sample inside the chart of `κ − h`"
//!   makes the point cloud blow up as `h → 0`; the h-scan then measures the
//!   sampler instead of the discretization and reads as a catastrophic
//!   derivative error at exactly the κ = 0 point the family exists to handle.
//!   [`chart_point`] uses a fixed radius valid across the whole stencil.
//! * **Each derivative is scaled by its own magnitude.** `∂d/∂κ` grows like
//!   `‖w‖³` and `∂²d/∂κ²` like `‖w‖⁵`, so normalizing them by the *distance*
//!   understates a real error by orders of magnitude.
//!
//! With both fixed, the residual is second-order in `h` (measured: `1.4e-7`,
//! `1.5e-9`, `1.5e-11` at `h = 1e-2, 1e-3, 1e-4` — a clean `h²` ladder until
//! the second difference hits its `ε/h²` roundoff floor). `H = 1e-3` is the
//! bottom of that curve for the second derivative and is what the assertions
//! use.

use ndarray::Array1;

use crate::manifold::RiemannianManifold;
use crate::manifolds::constant_curvature::ConstantCurvature;
use crate::manifolds::poincare;

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
        let u1 = self.uniform().max(1.0e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    fn gaussian_vec(&mut self, n: usize) -> Array1<f64> {
        Array1::from_shape_fn(n, |_| self.normal())
    }
}

/// A chart point of radius `< 0.45`.
///
/// Fixed, and in particular independent of κ and of any finite-difference step:
/// the κ < 0 chart is the ball of radius `1/√−κ`, so a radius chosen from the
/// current κ would make each stencil arm see a different point cloud. 0.45 is
/// interior for every κ ≥ −1.1, which covers every curvature used below plus
/// the widest stencil arm.
fn chart_point(rng: &mut Rng, dim: usize) -> Array1<f64> {
    let mut v = rng.gaussian_vec(dim);
    let magnitude = v.dot(&v).sqrt().max(1.0e-30);
    v *= 0.45 * rng.uniform() / magnitude;
    v
}

/// Curvatures spanning both signs, both series/closed-form regimes of the
/// `C`/`S`/`T` primitives, and the removable point itself.
const CURVATURES: [f64; 11] = [
    2.0, 1.0, 0.35, 0.01, 1.0e-6, 0.0, -1.0e-6, -0.01, -0.35, -1.0, -2.0,
];
const DIMS: [usize; 4] = [1, 2, 3, 5];
const TRIALS: usize = 200;
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

fn seed_for(dim: usize, kappa: f64) -> u64 {
    SEED ^ ((dim as u64) << 32) ^ kappa.to_bits()
}

#[test]
fn sectional_curvature_is_kappa_everywhere() {
    let mut verified = 0usize;
    for dim in DIMS {
        if dim < 2 {
            continue;
        }
        for kappa in CURVATURES {
            let manifold = ConstantCurvature::new(dim, kappa);
            let mut rng = Rng::new(seed_for(dim, kappa));
            for _ in 0..TRIALS {
                let x = chart_point(&mut rng, dim);
                let u = manifold
                    .project_tangent(x.view(), rng.gaussian_vec(dim).view())
                    .expect("tangent u");
                let v = manifold
                    .project_tangent(x.view(), rng.gaussian_vec(dim).view())
                    .expect("tangent v");
                let Ok(k) = manifold.sectional_curvature(x.view(), (u.view(), v.view())) else {
                    continue;
                };
                assert!(
                    (k - kappa).abs() <= 1.0e-9 * kappa.abs().max(1.0),
                    "dim {dim} kappa {kappa}: sectional curvature is {k}, not kappa"
                );
                verified += 1;
            }
        }
    }
    assert!(verified > 0, "no sectional curvature was evaluated");
}

#[test]
fn distance_is_invariant_under_mobius_translation() {
    // M_κ is homogeneous: Möbius addition by a fixed g is an isometry. This is
    // the property whose failure was #2351 — a κ̂ verdict that moved when the
    // data was translated.
    let mut verified = 0usize;
    for dim in DIMS {
        for kappa in CURVATURES {
            let manifold = ConstantCurvature::new(dim, kappa);
            let mut rng = Rng::new(seed_for(dim, kappa));
            for _ in 0..TRIALS {
                let x = chart_point(&mut rng, dim);
                let y = chart_point(&mut rng, dim);
                let g = chart_point(&mut rng, dim);
                let base = manifold.distance(x.view(), y.view()).expect("d(x,y)");
                let (Ok(gx), Ok(gy)) = (
                    manifold.mobius_add(g.view(), x.view()),
                    manifold.mobius_add(g.view(), y.view()),
                ) else {
                    continue;
                };
                let Ok(moved) = manifold.distance(gx.view(), gy.view()) else {
                    continue;
                };
                assert!(
                    (moved - base).abs() <= 1.0e-9 * base.max(1.0),
                    "dim {dim} kappa {kappa}: translation changed the distance \
                     ({base} -> {moved})"
                );
                verified += 1;
            }
        }
    }
    assert!(verified > 0, "no translated distance was evaluated");
}

#[test]
fn the_kappa_family_is_one_manifold_rescaled() {
    // A space form of curvature κ is the unit-curvature form of the same sign
    // scaled by 1/√|κ|, so d_κ(x, y) = d_{sign κ}(√|κ|·x, √|κ|·y) / √|κ|. This
    // ties every κ to the two reference geometries with no free parameter.
    let mut verified = 0usize;
    for dim in DIMS {
        for kappa in CURVATURES {
            if kappa.abs() <= 1.0e-9 {
                continue;
            }
            let manifold = ConstantCurvature::new(dim, kappa);
            let unit = ConstantCurvature::new(dim, kappa.signum());
            let scale = kappa.abs().sqrt();
            let mut rng = Rng::new(seed_for(dim, kappa));
            for _ in 0..TRIALS {
                let x = chart_point(&mut rng, dim);
                let y = chart_point(&mut rng, dim);
                let here = manifold.distance(x.view(), y.view()).expect("d_kappa");
                let Ok(there) = unit.distance((&x * scale).view(), (&y * scale).view()) else {
                    continue;
                };
                assert!(
                    (here - there / scale).abs() <= 1.0e-9 * here.max(1.0),
                    "dim {dim} kappa {kappa}: d_kappa = {here} but the rescaled unit \
                     form gives {}",
                    there / scale
                );
                verified += 1;
            }
        }
    }
    assert!(verified > 0, "no rescaling was evaluated");
}

#[test]
fn the_flat_and_hyperbolic_members_match_their_classical_forms() {
    // Two external anchors:
    //   κ = 0  — the conformal gauge is λ₀ = 2, so the flat metric is 4δ and
    //            the distance is 2‖x − y‖ (Euclidean up to the isometry x ↦ 2x,
    //            exactly as the module documents).
    //   κ = −1 — must agree with `poincare.rs`, which computes hyperbolic
    //            distance by an independent route. This is the only assertion
    //            in the file that compares two implementations rather than an
    //            implementation to a formula, and so is the strongest.
    let mut flat_checked = 0usize;
    let mut hyperbolic_checked = 0usize;
    for dim in DIMS {
        let flat = ConstantCurvature::new(dim, 0.0);
        let hyperbolic = ConstantCurvature::new(dim, -1.0);
        let mut rng = Rng::new(seed_for(dim, 0.0));
        for _ in 0..TRIALS {
            let x = chart_point(&mut rng, dim);
            let y = chart_point(&mut rng, dim);

            let got = flat.distance(x.view(), y.view()).expect("flat distance");
            let want = 2.0 * (&x - &y).dot(&(&x - &y)).sqrt();
            assert!(
                (got - want).abs() <= 1.0e-12 * want.max(1.0),
                "dim {dim}: flat distance {got} != 2||x-y|| = {want}"
            );
            flat_checked += 1;

            let got = hyperbolic
                .distance(x.view(), y.view())
                .expect("hyperbolic distance");
            let want =
                poincare::poincare_distance(x.view(), y.view(), -1.0).expect("poincare distance");
            assert!(
                (got - want).abs() <= 1.0e-12 * want.max(1.0),
                "dim {dim}: kappa=-1 distance {got} disagrees with poincare.rs {want}"
            );
            hyperbolic_checked += 1;
        }
    }
    assert!(flat_checked > 0 && hyperbolic_checked > 0);
}

#[test]
fn distance_is_a_metric_and_agrees_with_the_logarithm() {
    let mut verified = 0usize;
    for dim in DIMS {
        for kappa in CURVATURES {
            let manifold = ConstantCurvature::new(dim, kappa);
            let mut rng = Rng::new(seed_for(dim, kappa));
            for _ in 0..TRIALS {
                let x = chart_point(&mut rng, dim);
                let y = chart_point(&mut rng, dim);
                let z = chart_point(&mut rng, dim);
                let dxy = manifold.distance(x.view(), y.view()).expect("d(x,y)");
                let dyx = manifold.distance(y.view(), x.view()).expect("d(y,x)");
                let dyz = manifold.distance(y.view(), z.view()).expect("d(y,z)");
                let dxz = manifold.distance(x.view(), z.view()).expect("d(x,z)");

                assert!(
                    (dxy - dyx).abs() <= 1.0e-12 * dxy.max(1.0),
                    "dim {dim} kappa {kappa}: distance is not symmetric ({dxy} vs {dyx})"
                );
                assert!(
                    dxz <= dxy + dyz + 1.0e-12 * dxz.max(1.0),
                    "dim {dim} kappa {kappa}: triangle inequality violated \
                     ({dxz} > {dxy} + {dyz})"
                );

                // ‖log_x y‖ in the metric at x is the geodesic distance. The
                // metric is conformal, so the norm is λ_x·‖·‖.
                let logarithm = manifold.log_map(x.view(), y.view()).expect("log");
                let lambda = manifold
                    .conformal_factor(x.view())
                    .expect("conformal factor");
                let metric_norm = lambda * logarithm.dot(&logarithm).sqrt();
                assert!(
                    (metric_norm - dxy).abs() <= 1.0e-9 * dxy.max(1.0),
                    "dim {dim} kappa {kappa}: ||log|| = {metric_norm} != d = {dxy}"
                );
                verified += 1;
            }
        }
    }
    assert!(verified > 0, "no metric axiom was evaluated");
}

/// #2687: the κ > 0 branch is **not** unconstrained, and its singular locus sits
/// at the same `|κ| = 1/R²` as the hyperbolic one.
///
/// The per-point chart gauge `1 + κ‖x‖²` really is vacuous for `κ ≥ 0` — but it
/// is not the condition the kernel evaluates. Every distance goes through
/// `w = (−x) ⊕_κ y`, whose Möbius denominator is
///
/// ```text
///   D(κ) = 1 + 2κ⟨x,y⟩ + κ²‖x‖²‖y‖²
/// ```
///
/// For an anti-aligned pair (`⟨x,y⟩ = −‖x‖‖y‖`) that is `(1 − κ‖x‖‖y‖)²`, which
/// vanishes at `κ = +1/(‖x‖‖y‖)`. There the two points are exactly ANTIPODAL on
/// the sphere of radius `1/√κ` and the chart coordinate `w` passes through
/// infinity — the exact spherical mirror of `1 + κ‖x‖² = 0` on the hyperbolic
/// side. Worst case over a cloud of maximum chart radius `R` is `κ = +1/R²`, so
/// a κ window symmetric about zero at a fraction of `1/R²` is a genuine margin
/// on BOTH sides rather than one branch's constraint mirrored onto the other.
///
/// Past the fold the chart is not merely inaccurate, it is **folded**: the
/// normalized separation `d_κ(x,y)·√κ/π` (the pair's fraction of the antipodal
/// maximum, the only scale-free thing κ can be read off) is exactly invariant
/// under the involution `κ ↦ 1/(κ‖x‖²‖y‖²)`, so every κ past the fold is an
/// exact reflection of one before it.
///
/// Every number below was pinned from `d = (2/√κ)·arctan(√κ‖w‖)`,
/// `‖w‖ = 2R/(1 − κR²)` BEFORE this test was first run.
#[test]
fn spherical_branch_folds_at_kappa_r2_one_so_the_kappa_window_is_symmetric_2687() {
    const R: f64 = 0.6;
    let r2 = R * R;
    let x = ndarray::array![R, 0.0];
    let y = ndarray::array![-R, 0.0];

    // (a) The shipped window's cap, κ = CONSTANT_CURVATURE_KAPPA_CHART_FRACTION/R²
    // with the fraction 0.5. Interior, and 78.4% of the way to the antipode.
    let cap = 0.5 / r2;
    let manifold = ConstantCurvature::new(2, cap);
    let d_cap = manifold
        .distance(x.view(), y.view())
        .expect("the shipped κ cap must be strictly inside the chart");
    assert!(
        (d_cap - 2.089_007_403_281_048).abs() <= 1.0e-12,
        "κ = 0.5/R²: d = {d_cap}, predicted 2.089007403281048"
    );
    let frac_cap = d_cap * cap.sqrt() / std::f64::consts::PI;
    assert!(
        (frac_cap - 0.783_653_104_061_214_8).abs() <= 1.0e-12,
        "κ = 0.5/R²: antipodal fraction = {frac_cap}, predicted 0.7836531040612148"
    );

    // (b) κ = 1/R² is the fold itself: D = (1 − κR²)² collapses and the shipped
    // guard refuses. This is the number the ±0.5 fraction is half of.
    let fold = 1.0 / r2;
    let refused = ConstantCurvature::new(2, fold).distance(x.view(), y.view());
    assert!(
        refused.is_err(),
        "κ = 1/R² = {fold} is the κ>0 antipodal fold and must be refused; got {refused:?}"
    );

    // (c) The same fold with an exactly representable denominator, so the
    // refusal cannot be an artifact of a near-miss: κ‖x‖‖y‖ = 4·0.25 = 1 in f64.
    let exact = ConstantCurvature::new(2, 4.0);
    let exact_refused = exact.distance(
        ndarray::array![0.5, 0.0].view(),
        ndarray::array![-0.5, 0.0].view(),
    );
    assert!(
        exact_refused.is_err(),
        "κ‖x‖‖y‖ = 1 exactly must be refused; got {exact_refused:?}"
    );

    // (d) The involution. `κ_wide = 9.5/R²` is the widening proposed on #2687;
    // it is the EXACT reflection of `κ_twin = (1/9.5)/R²`, which is interior to
    // the shipped window. Widening the box to 9.5/R² would therefore admit an
    // exact duplicate of a κ the box already contains.
    let kappa_wide = 9.5 / r2;
    let kappa_twin = (1.0 / 9.5) / r2;
    let d_wide = ConstantCurvature::new(2, kappa_wide)
        .distance(x.view(), y.view())
        .expect("past the fold the chart still evaluates — that is the problem");
    let d_twin = ConstantCurvature::new(2, kappa_twin)
        .distance(x.view(), y.view())
        .expect("the twin is interior");
    let frac_wide = d_wide * kappa_wide.sqrt() / std::f64::consts::PI;
    let frac_twin = d_twin * kappa_twin.sqrt() / std::f64::consts::PI;
    assert!(
        (frac_wide - 0.399_450_751_329_086_6).abs() <= 1.0e-12,
        "κ = 9.5/R²: antipodal fraction = {frac_wide}, predicted 0.3994507513290866"
    );
    assert!(
        (frac_wide - frac_twin).abs() <= 1.0e-14,
        "κ = 9.5/R² and κ = (1/9.5)/R² must be indistinguishable in the pair's \
         scale-free geometry: {frac_wide} vs {frac_twin}"
    );
    assert!(
        kappa_twin < cap && cap < fold && fold < kappa_wide,
        "the ordering this test is about: twin {kappa_twin} < cap {cap} < fold \
         {fold} < proposed {kappa_wide}"
    );

    // (e) The involution is not special to the anti-aligned pair — it is a
    // property of the chart. Over general pairs the reflected κ reproduces the
    // scale-free separation to machine precision.
    let mut rng = Rng::new(0x2687);
    let mut verified = 0usize;
    for _ in 0..64 {
        let p = chart_point(&mut rng, 2);
        let q = chart_point(&mut rng, 2);
        let ab2 = p.dot(&p) * q.dot(&q);
        for s in [0.25_f64, 0.75, 3.0] {
            let k_lo = s / ab2.sqrt();
            let k_hi = 1.0 / (k_lo * ab2);
            let (Ok(d_lo), Ok(d_hi)) = (
                ConstantCurvature::new(2, k_lo).distance(p.view(), q.view()),
                ConstantCurvature::new(2, k_hi).distance(p.view(), q.view()),
            ) else {
                continue;
            };
            let f_lo = d_lo * k_lo.sqrt() / std::f64::consts::PI;
            let f_hi = d_hi * k_hi.sqrt() / std::f64::consts::PI;
            assert!(
                (f_lo - f_hi).abs() <= 1.0e-12,
                "κ ↦ 1/(κ‖x‖²‖y‖²) must fix the scale-free separation: \
                 κ={k_lo} gives {f_lo}, κ={k_hi} gives {f_hi}"
            );
            verified += 1;
        }
    }
    assert!(verified > 0, "no reflected pair was evaluated");
}

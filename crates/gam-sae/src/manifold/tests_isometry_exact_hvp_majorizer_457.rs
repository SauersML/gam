//! Exact isometry-penalty Hessian-vector-product and PSD Gauss-Newton
//! majorizer liveness tests for the SAE manifold (#457 / #857).
//!
//! Split out of `tests.rs` by cohesive concern (issue #780 line-count gate):
//! the exact-HVP-vs-grad-FD checks, the zero-residual exact/GN collapse, the
//! PSD-majorizer liveness probes, and the multi-atom isometry cache-pairing
//! regression form one self-contained cluster sharing the
//! `build_isometry_atom_for_evaluator` fixture and its `deterministic_decoder`.

use super::*;
use approx::assert_abs_diff_eq;
use gam_terms::analytic_penalties::IsometryReference;
use ndarray::array;

pub(crate) fn deterministic_decoder(n_basis: usize, p_out: usize, seed: f64) -> Array2<f64> {
    Array2::<f64>::from_shape_fn((n_basis, p_out), |(i, j)| {
        let x = seed + 0.371 * (i as f64) - 0.193 * (j as f64) + 0.047 * ((i * j + 1) as f64);
        0.8 * x.sin() + 0.35 * (1.7 * x).cos()
    })
}

pub(crate) fn build_isometry_atom_for_evaluator(
    evaluator: Arc<dyn SaeBasisSecondJet>,
    kind: SaeAtomBasisKind,
    coords: &Array2<f64>,
    p_out: usize,
    seed: f64,
) -> (SaeManifoldAtom, IsometryPenalty, Array1<f64>) {
    let (phi, jet) = evaluator
        .evaluate(coords.view())
        .expect("the fixture's coordinate block is a valid input for this evaluator");
    let m = phi.ncols();
    let decoder = deterministic_decoder(m, p_out, seed);
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "exact_hvp_atom",
        kind,
        coords.ncols(),
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .expect("the fixture's basis, decoder and Gram blocks agree in dimension")
    .with_basis_second_jet(evaluator);
    let target_flat: Array1<f64> = coords.iter().copied().collect();
    let penalty = IsometryPenalty::new_euclidean(
        PsiSlice::full(target_flat.len(), Some(coords.ncols())),
        p_out,
    );
    (atom, penalty, target_flat)
}

pub(crate) fn assert_exact_isometry_hvp_matches_grad_fd(
    evaluator: Arc<dyn SaeBasisSecondJet>,
    kind: SaeAtomBasisKind,
    coords: Array2<f64>,
    p_out: usize,
    direction: Array2<f64>,
) {
    let (atom, penalty, target_flat) =
        build_isometry_atom_for_evaluator(evaluator, kind, &coords, p_out, 0.91);
    let rho = array![0.0_f64];
    let installed = refresh_isometry_caches_from_atom(&penalty, &atom, coords.view())
        .expect("the fixture's isometry penalty and atom share one coordinate block");
    assert!(
        installed,
        "second-jet cache must be installed for exact HVP test"
    );
    assert!(
        penalty.third_decoder_derivative().is_some(),
        "non-Duchon exact HVP requires a live refreshed third-decoder-jet cache"
    );
    let v: Array1<f64> = direction.iter().copied().collect();
    let exact = penalty.hvp(target_flat.view(), rho.view(), v.view());
    assert!(
        exact.iter().any(|x| x.abs() > 1.0e-7),
        "exact isometry HVP should be nonzero after K refresh; got {exact:?}"
    );

    let eps = 1.0e-6;
    let coords_plus = &coords + &(direction.mapv(|x| eps * x));
    let coords_minus = &coords - &(direction.mapv(|x| eps * x));
    let target_plus: Array1<f64> = coords_plus.iter().copied().collect();
    let target_minus: Array1<f64> = coords_minus.iter().copied().collect();

    refresh_isometry_caches_from_atom(&penalty, &atom, coords_plus.view())
        .expect("the fixture's isometry penalty and atom share one coordinate block");
    let grad_plus = penalty.grad_target(target_plus.view(), rho.view());
    refresh_isometry_caches_from_atom(&penalty, &atom, coords_minus.view())
        .expect("the fixture's isometry penalty and atom share one coordinate block");
    let grad_minus = penalty.grad_target(target_minus.view(), rho.view());
    refresh_isometry_caches_from_atom(&penalty, &atom, coords.view())
        .expect("the fixture's isometry penalty and atom share one coordinate block");

    let fd = (&grad_plus - &grad_minus).mapv(|x| x / (2.0 * eps));
    for i in 0..exact.len() {
        let err = (exact[i] - fd[i]).abs();
        let tol = 2.0e-4 + 3.0e-5 * exact[i].abs().max(fd[i].abs());
        assert!(
            err <= tol,
            "exact isometry HVP/grad-FD mismatch at flat index {i}: exact={:.12e}, fd={:.12e}, err={:.6e}, tol={:.6e}",
            exact[i],
            fd[i],
            err,
            tol
        );
    }
}

pub(crate) fn assert_exact_isometry_hvp_collapses_to_gn_at_zero_residual(
    evaluator: Arc<dyn SaeBasisSecondJet>,
    kind: SaeAtomBasisKind,
    coords: Array2<f64>,
    p_out: usize,
    direction: Array2<f64>,
) {
    let (atom, penalty, target_flat) =
        build_isometry_atom_for_evaluator(evaluator, kind, &coords, p_out, 1.37);
    let rho = array![0.0_f64];
    let d = coords.ncols();

    // Build the reference metric from the EXACT SAME cache the exact HVP
    // differences against (#857). The exact HVP computes its residual
    // `diff = g/gbar − g_ref` where `g = penalty.pullback_metric(d)` is read
    // from `penalty`'s own Jacobian cache, and skips the third-jet `K` term
    // only when `diff == 0.0` (a bit-exact float compare). Previously `g_ref` was
    // built from a SEPARATE `scratch` penalty's cache, so a last-ULP
    // difference between the two independent refreshes left `diff` ~1e-16
    // rather than exactly 0; multiplied by the large third decoder jet
    // (`K ~ ω³`) for the torus/sphere bases, that leaked past the 1e-10
    // exact-equality bound. Refreshing `penalty` once and seeding the
    // UserSupplied reference from the normalized `penalty.pullback_metric(d)`
    // makes `g_ref` the identical array `g/gbar` is recomputed from, so the
    // residual is bit-zero and the K term is genuinely skipped — leaving
    // exactly the GN term. `with_reference` moves the penalty by value and
    // preserves every cache slot, so the J/J2/K caches read by the HVP are
    // unchanged.
    refresh_isometry_caches_from_atom(&penalty, &atom, coords.view())
        .expect("the fixture's isometry penalty and atom share one coordinate block");
    let mut g_ref = penalty
        .pullback_metric(d)
        .expect("pullback metric is available after the cache refresh");
    let mut trace_sum = 0.0_f64;
    for row in 0..g_ref.nrows() {
        for axis in 0..d {
            trace_sum += g_ref[[row, axis * d + axis]];
        }
    }
    let normalizer = trace_sum / (g_ref.nrows() * d) as f64;
    for value in g_ref.iter_mut() {
        *value /= normalizer;
    }
    let penalty = penalty.with_reference(IsometryReference::UserSupplied(Arc::new(g_ref)));
    assert!(
        penalty.third_decoder_derivative().is_some(),
        "zero-residual exact/GN test must still carry the real refreshed K cache"
    );
    let v: Array1<f64> = direction.iter().copied().collect();
    let exact = penalty.hvp(target_flat.view(), rho.view(), v.view());
    let gn = penalty.psd_majorizer_hvp(target_flat.view(), rho.view(), v.view());
    assert!(
        gn.iter().any(|x| x.abs() > 1.0e-8),
        "GN block should be nonzero so exact/GN equality is not vacuous"
    );
    for i in 0..exact.len() {
        assert_abs_diff_eq!(exact[i], gn[i], epsilon = 1.0e-10);
    }
}

#[test]
pub(crate) fn isometry_exact_hvp_sphere_matches_grad_fd_and_uses_refreshed_k() {
    assert_exact_isometry_hvp_matches_grad_fd(
        Arc::new(AmbientSphereHarmonicEvaluator::new(2).unwrap()),
        SaeAtomBasisKind::Sphere,
        array![
            [0.0, 0.0, 1.0],
            [0.6, -0.8, 0.0],
            [0.36, 0.48, 0.8],
            [-0.48, 0.6, -0.64]
        ],
        4,
        // One tangent-direction entry per AMBIENT axis, matching the 3-wide
        // sphere coordinate above.
        array![
            [0.31, -0.27, 0.11],
            [-0.18, 0.22, -0.07],
            [0.14, 0.19, 0.23],
            [-0.25, -0.11, 0.16]
        ],
    );
}

#[test]
pub(crate) fn isometry_exact_hvp_torus_matches_grad_fd_and_uses_refreshed_k() {
    assert_exact_isometry_hvp_matches_grad_fd(
        Arc::new(TorusHarmonicEvaluator::new(2, 2).unwrap()),
        SaeAtomBasisKind::Torus,
        array![[0.13, 0.42], [0.66, 0.19], [0.88, 0.55]],
        3,
        array![[0.21, -0.16], [-0.24, 0.18], [0.13, 0.27]],
    );
}

#[test]
pub(crate) fn isometry_exact_hvp_sphere_and_torus_collapse_to_gn_at_zero_residual() {
    assert_exact_isometry_hvp_collapses_to_gn_at_zero_residual(
        Arc::new(AmbientSphereHarmonicEvaluator::new(2).unwrap()),
        SaeAtomBasisKind::Sphere,
        array![[0.0, 0.0, 1.0], [0.6, -0.8, 0.0], [0.36, 0.48, 0.8]],
        4,
        array![
            [0.17, -0.21, 0.09],
            [-0.13, 0.08, -0.24],
            [0.22, 0.19, 0.05]
        ],
    );
    assert_exact_isometry_hvp_collapses_to_gn_at_zero_residual(
        Arc::new(TorusHarmonicEvaluator::new(2, 2).unwrap()),
        SaeAtomBasisKind::Torus,
        array![[0.19, 0.31], [0.57, 0.73], [0.84, 0.12]],
        3,
        array![[0.11, -0.14], [-0.20, 0.07], [0.16, 0.23]],
    );
}

/// #457 root-cause regression: for every **non-Duchon** SAE basis the
/// isometry penalty's *exact* `hvp` returns the zero vector (no third jet
/// `K` cache outside the radial-Duchon source), so the Arrow-Schur coord
/// curvature block — which routes through `psd_majorizer_hvp` — would carry
/// **no isometry contribution at all**, and the pole fit diverges. The fix
/// is the PSD Gauss-Newton majorizer override, which needs only the first
/// and second decoder jets that `refresh_isometry_caches_from_atom`
/// installs for any basis with an analytic second jet.
///
/// This drives the real cache-refresh path with the sphere / circle /
/// torus evaluators against the **Euclidean** reference (so the residual
/// `g − I` is genuinely nonzero — the live production condition, unlike the
/// zero-residual collapse test), then asserts the curvature operator the
/// inner solve actually consumes is:
///   * genuinely **nonzero** (the bug was a silent zero block),
///   * **symmetric**, and
///   * **positive-semidefinite** (`vᵀB v ≥ 0`),
/// pinning the exact seam #457 is about, end-to-end from the evaluator.
pub(crate) fn assert_isometry_psd_majorizer_live_after_atom_refresh(
    evaluator: Arc<dyn SaeBasisSecondJet>,
    kind: SaeAtomBasisKind,
    coords: Array2<f64>,
    p_out: usize,
    probes: &[Array2<f64>],
) {
    let (atom, penalty, target_flat) =
        build_isometry_atom_for_evaluator(evaluator, kind, &coords, p_out, 0.53);
    let rho = array![0.0_f64];

    // Before any refresh the safe default is the zero block: confirm the
    // precondition so the post-refresh contrast is the genuine fix, not a
    // coincidence of a probe direction.
    let n = target_flat.len();
    let unit0 = {
        let mut e = Array1::<f64>::zeros(n);
        e[0] = 1.0;
        e
    };
    let pre = penalty.psd_majorizer_hvp(target_flat.view(), rho.view(), unit0.view());
    assert!(
        pre.iter().all(|x| *x == 0.0),
        "psd_majorizer_hvp without a cache must be the zero block; got {pre:?}"
    );

    let installed = refresh_isometry_caches_from_atom(&penalty, &atom, coords.view())
        .expect("the fixture's isometry penalty and atom share one coordinate block");
    assert!(
        installed,
        "second-jet cache must install for the PSD-majorizer liveness test"
    );

    // The Euclidean reference makes g/gbar − I nonzero on this non-orthonormal
    // decoder; verify the residual is real so the curvature seam is the
    // production one (and not vacuously the zero-residual case).
    let d = coords.ncols();
    let g = penalty
        .pullback_metric(d)
        .expect("pullback metric available after refresh");
    let mut trace_sum = 0.0_f64;
    for row in 0..g.nrows() {
        for axis in 0..d {
            trace_sum += g[[row, axis * d + axis]];
        }
    }
    let normalizer = trace_sum / (g.nrows() * d) as f64;
    let mut residual_mass = 0.0_f64;
    for row in 0..g.nrows() {
        for a in 0..d {
            for b in 0..d {
                // Euclidean reference is the identity metric I_d.
                let g_ref = if a == b { 1.0 } else { 0.0 };
                residual_mass += (g[[row, a * d + b]] / normalizer - g_ref).abs();
            }
        }
    }
    assert!(
        residual_mass > 1.0e-3,
        "Euclidean-reference residual must be nonzero for a real curvature test; \
             got residual mass {residual_mass:.3e}"
    );

    // Assemble the dense majorizer column-by-column via unit probes.
    let mut bmat = Array2::<f64>::zeros((n, n));
    for k in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[k] = 1.0;
        let col = penalty.psd_majorizer_hvp(target_flat.view(), rho.view(), e.view());
        for r in 0..n {
            bmat[[r, k]] = col[r];
        }
    }

    // Nonzero: the bug was a silent all-zero curvature block.
    let max_abs = bmat.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
    assert!(
        max_abs > 1.0e-6,
        "isometry GN majorizer must be nonzero for a non-Duchon basis after refresh; \
             max |B| = {max_abs:.3e}"
    );

    // Symmetry: B = Σ_n (∂g/∂t)ᵀ(∂g/∂t) is symmetric by construction.
    for r in 0..n {
        for c in 0..n {
            assert_abs_diff_eq!(bmat[[r, c]], bmat[[c, r]], epsilon = 1.0e-10);
        }
    }

    // PSD: vᵀ B v ≥ 0 over a spread of probe directions.
    for probe in probes {
        let v: Array1<f64> = probe.iter().copied().collect();
        assert_eq!(v.len(), n, "probe must match the flattened target length");
        let bv = penalty.psd_majorizer_hvp(target_flat.view(), rho.view(), v.view());
        let quad = v.dot(&bv);
        assert!(
            quad >= -1.0e-9,
            "isometry GN majorizer must be PSD; got vᵀBv = {quad:.3e}"
        );
    }
}

#[test]
pub(crate) fn isometry_psd_majorizer_live_after_sphere_refresh() {
    assert_isometry_psd_majorizer_live_after_atom_refresh(
        Arc::new(AmbientSphereHarmonicEvaluator::new(2).unwrap()),
        SaeAtomBasisKind::Sphere,
        array![[0.0, 0.0, 1.0], [0.6, -0.8, 0.0], [0.36, 0.48, 0.8]],
        4,
        &[
            array![[0.31, -0.27, 0.91], [-0.18, 0.22, 0.96], [0.14, 0.19, 0.97]],
            array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            array![[-2.3, 0.6, 0.4], [-0.1, 1.4, 0.9], [0.8, -1.7, 0.2]],
        ],
    );
}

#[test]
pub(crate) fn isometry_psd_majorizer_live_after_circle_refresh() {
    assert_isometry_psd_majorizer_live_after_atom_refresh(
        Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap()),
        SaeAtomBasisKind::Periodic,
        array![[0.12], [0.37], [0.58], [0.81]],
        3,
        &[
            array![[0.4], [-1.1], [0.7], [0.3]],
            array![[1.0], [1.0], [1.0], [1.0]],
            array![[-2.3], [0.6], [-0.1], [1.4]],
        ],
    );
}

#[test]
pub(crate) fn isometry_psd_majorizer_live_after_torus_refresh() {
    assert_isometry_psd_majorizer_live_after_atom_refresh(
        Arc::new(TorusHarmonicEvaluator::new(2, 2).unwrap()),
        SaeAtomBasisKind::Torus,
        array![[0.13, 0.42], [0.66, 0.19], [0.88, 0.55]],
        3,
        &[
            array![[0.21, -0.16], [-0.24, 0.18], [0.13, 0.27]],
            array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            array![[-1.2, 0.5], [0.3, -0.9], [0.7, 0.2]],
        ],
    );
}

/// A heterogeneous SAE owns one registry-level isometry penalty whose target
/// is sized at `d_max`, but evaluates that penalty against every atom's own
/// compact coordinate block. The corrected per-atom clone must therefore
/// carry the atom-local `(N * d_atom, d_atom)` target before its Jacobian cache
/// is refreshed. Otherwise the d=1 atom below installs `(N, p)` and `value`
/// reads the stale wider target, requesting `(N, 3p)` and panicking. The
/// heterogeneity is now d=1 against d=3, since the sphere atom carries an
/// ambient 3-vector — a wider spread than the d=1/d=2 pair it replaced, so the
/// retarget it exercises is strictly harder.
#[test]
pub(crate) fn corrected_isometry_penalty_retargets_mixed_dimension_atoms() {
    let p_out = 4usize;
    let coords_d1 = array![[0.05], [0.20], [0.55], [0.80]];
    let coords_d3 = array![
        [0.0, 0.0, 1.0],
        [0.6, -0.8, 0.0],
        [0.36, 0.48, 0.8],
        [-0.48, 0.6, -0.64]
    ];

    let (atom_d1, _, _) = build_isometry_atom_for_evaluator(
        Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap()),
        SaeAtomBasisKind::Periodic,
        &coords_d1,
        p_out,
        0.53,
    );
    let (atom_d3, _, _) = build_isometry_atom_for_evaluator(
        Arc::new(AmbientSphereHarmonicEvaluator::new(2).unwrap()),
        SaeAtomBasisKind::Sphere,
        &coords_d3,
        p_out,
        1.37,
    );

    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((coords_d1.nrows(), 2)),
        vec![coords_d1, coords_d3],
        vec![
            LatentManifold::Circle { period: 1.0 },
            // `coords_d3` is S^2 in AMBIENT R^3: the evaluator is
            // `AmbientSphereHarmonicEvaluator::new(2)`, and every row is a unit
            // vector ([0,0,1], [0.6,-0.8,0], [0.36,0.48,0.8] -> 0.1296+0.2304+
            // 0.64 = 1.0, [-0.48,0.6,-0.64]). So the latent manifold is a
            // 3-wide sphere.
            //
            // It used to declare `Product([Interval, Circle])`, which is TWO
            // coordinates: `ambient_dim` sums its parts (Interval -> 1,
            // Circle -> 1), so the spec claimed width 2 for a 3-column block.
            // `project_all_rows_to_manifold` asserts
            // `manifold.ambient_dim(latent_dim) == latent_dim` before it strides
            // the flat buffer, and it fired exactly as it should:
            // `left: 2, right: 3`. Striding a 3-wide block as if it were 2-wide
            // would have walked rows out of alignment, so the assert was the
            // only thing standing between this fixture and silent corruption.
            //
            // `Sphere { dim }` carries the AMBIENT width (`ambient_dim` returns
            // `dim` directly, not `dim + 1`), so S^2 in R^3 is `dim: 3`.
            LatentManifold::Sphere { dim: 3 },
        ],
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, true),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom_d1, atom_d3], assignment).unwrap();

    // This mirrors production: one default-on registry penalty is sized from
    // the heterogeneous latent spec's maximum dimension and shared by both
    // atoms. Per-atom correction must keep it live, not disable it.
    let mut registry = AnalyticPenaltyRegistry::new();
    registry.push(AnalyticPenaltyKind::Isometry(Arc::new(
        // Sized from the heterogeneous spec's MAXIMUM latent dimension, which is
        // atom 1's 3 (was 2, from the old Product spec).
        IsometryPenalty::new_euclidean(PsiSlice::full(term.n_obs() * 3, Some(3)), p_out),
    )));
    let registry_iso = match &registry.penalties[0] {
        AnalyticPenaltyKind::Isometry(penalty) => penalty,
        _ => panic!("expected isometry penalty"),
    };
    // Recreate the reported stale shared state deliberately: the registry
    // descriptor currently carries a d=2 Jacobian when production evaluates
    // atom 0 at d=1. `corrected_isometry_penalty` must clone the descriptor and
    // overwrite the clone with atom 0's own d=1 cache before any value read;
    // mutating or directly reading this registry cache would respectively
    // violate per-atom ownership or trip the hard dimensional invariant.
    let stale_registry_jacobian =
        Arc::new(Array2::<f64>::from_elem((term.n_obs(), p_out * 2), -7.0));
    registry_iso.refresh_caches(Some(stale_registry_jacobian.clone()), None);
    let rho = array![0.0_f64];

    // Atom 1 is S^2 in ambient R^3, so its latent width is 3 -- see the
    // `Sphere { dim: 3 }` note above. This loop said 2, matching the old
    // 2-coordinate Product spec rather than the atom's actual coordinates.
    for (atom_idx, expected_dim) in [(0usize, 1usize), (1usize, 3usize)] {
        let coord = &term.assignment.coords[atom_idx];
        let corrected = term
            .corrected_isometry_penalty(registry_iso, atom_idx, coord)
            .unwrap();
        let corrected_iso = match &corrected {
            AnalyticPenaltyKind::Isometry(penalty) => penalty,
            _ => panic!("expected corrected isometry penalty"),
        };
        assert_eq!(corrected_iso.target.latent_dim, Some(expected_dim));
        assert_eq!(corrected_iso.target.len(), term.n_obs() * expected_dim);
        assert_eq!(
            corrected_iso
                .jacobian_cache()
                .expect("corrected penalty must have a live Jacobian cache")
                .ncols(),
            p_out * expected_dim
        );
        let value = corrected.value(coord.as_flat().view(), rho.view());
        assert!(
            value.is_finite() && value > 0.0,
            "atom {atom_idx} must retain live, nonzero default isometry; got {value}"
        );
        let registry_jacobian = registry_iso
            .jacobian_cache()
            .expect("registry descriptor keeps its seeded cache");
        assert!(
            Arc::ptr_eq(&registry_jacobian, &stale_registry_jacobian),
            "per-atom correction must refresh an owned clone, not mutate shared registry state"
        );
    }

    let total = term
        .isometry_penalty_value_total(&registry)
        .expect("mixed-dimension default isometry must evaluate without a shape panic");
    assert!(
        total.is_finite() && total > 0.0,
        "mixed-dimension isometry total must remain live; got {total}"
    );
}

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
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let m = phi.ncols();
    let decoder = deterministic_decoder(m, p_out, seed);
    let atom = SaeManifoldAtom::new(
        "exact_hvp_atom",
        kind,
        coords.ncols(),
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(m),
    )
    .unwrap()
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
    let installed = refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();
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

    refresh_isometry_caches_from_atom(&penalty, &atom, coords_plus.view()).unwrap();
    let grad_plus = penalty.grad_target(target_plus.view(), rho.view());
    refresh_isometry_caches_from_atom(&penalty, &atom, coords_minus.view()).unwrap();
    let grad_minus = penalty.grad_target(target_minus.view(), rho.view());
    refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();

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
    refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();
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
        Arc::new(SphereChartEvaluator),
        SaeAtomBasisKind::Sphere,
        array![[-0.61, 0.23], [-0.18, -1.07], [0.42, 0.81], [0.73, -0.39]],
        4,
        array![[0.31, -0.27], [-0.18, 0.22], [0.14, 0.19], [-0.25, -0.11]],
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
        Arc::new(SphereChartEvaluator),
        SaeAtomBasisKind::Sphere,
        array![[-0.52, 0.17], [-0.11, -0.93], [0.39, 0.74]],
        4,
        array![[0.17, -0.21], [-0.13, 0.08], [0.22, 0.19]],
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

    let installed = refresh_isometry_caches_from_atom(&penalty, &atom, coords.view()).unwrap();
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
        Arc::new(SphereChartEvaluator),
        SaeAtomBasisKind::Sphere,
        array![[-0.61, 0.23], [-0.18, -1.07], [0.42, 0.81]],
        4,
        &[
            array![[0.31, -0.27], [-0.18, 0.22], [0.14, 0.19]],
            array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            array![[-2.3, 0.6], [-0.1, 1.4], [0.8, -1.7]],
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

/// Multi-atom isometry pairing regression.
///
/// Two SAE atoms share the same `(latent_dim, p_out)` signature but live
/// on different coordinate blocks. The registry holds one isometry penalty
/// per atom. The previous `find()` first-match logic paired *both*
/// penalties to atom 0, so atom 1's coords were never installed into the
/// second penalty's Jacobian cache — silently mislabeling the second
/// atom's geometry as the first's. The positional pairing must instead
/// refresh penalty `i` against atom `i`.
///
/// We pin this by computing, independently, the Jacobian cache each atom
/// would produce in isolation, then asserting that after
/// `refresh_isometry_caches_from_term` the two registry penalties carry
/// *distinct* caches matching their *own* atoms.
#[test]
pub(crate) fn refresh_isometry_caches_pairs_each_penalty_to_its_own_atom() {
    let latent_dim = 1usize;
    let p_out = 3usize;
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(5).unwrap());

    // Distinct coords per atom so the cached Jacobians must differ.
    let coords0 = array![[0.05], [0.20], [0.55], [0.80]];
    let coords1 = array![[0.13], [0.41], [0.62], [0.91]];

    let build_atom = |name: &str, coords: &Array2<f64>, seed: f64| {
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let m = phi.ncols();
        let mut decoder = Array2::<f64>::zeros((m, p_out));
        for i in 0..m {
            for j in 0..p_out {
                let x = (i as f64) * 0.371 + (j as f64) * 0.193 + seed;
                decoder[[i, j]] = (x.sin() * 0.9) + 0.1 * ((i + j) as f64).cos();
            }
        }
        let smooth = Array2::<f64>::eye(m);
        SaeManifoldAtom::new(
            name,
            SaeAtomBasisKind::Periodic,
            latent_dim,
            phi,
            jet,
            decoder,
            smooth,
        )
        .unwrap()
        .with_basis_second_jet(evaluator.clone() as Arc<dyn SaeBasisSecondJet>)
    };

    let atom0 = build_atom("atom0", &coords0, 0.5);
    let atom1 = build_atom("atom1", &coords1, 1.7);

    // Independent ground-truth caches: refresh a standalone penalty
    // against each atom in isolation.
    let slice0 = PsiSlice::full(coords0.nrows() * latent_dim, Some(latent_dim));
    let control0 = IsometryPenalty::new_euclidean(slice0, p_out);
    refresh_isometry_caches_from_atom(&control0, &atom0, coords0.view()).unwrap();
    let expected0 = control0
        .jacobian_cache()
        .expect("control penalty 0 must have a Jacobian cache");

    let slice1 = PsiSlice::full(coords1.nrows() * latent_dim, Some(latent_dim));
    let control1 = IsometryPenalty::new_euclidean(slice1, p_out);
    refresh_isometry_caches_from_atom(&control1, &atom1, coords1.view()).unwrap();
    let expected1 = control1
        .jacobian_cache()
        .expect("control penalty 1 must have a Jacobian cache");

    // The two atoms genuinely differ, else the test is vacuous.
    assert_ne!(
        *expected0, *expected1,
        "atom 0 and atom 1 must produce distinct Jacobian caches"
    );

    // Build the term and a registry with one isometry penalty per atom.
    let logits = Array2::<f64>::zeros((coords0.nrows(), 2));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0.clone(), coords1.clone()],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::ibp_map(0.7, 1.0, true),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();

    let mut registry = AnalyticPenaltyRegistry::new();
    let pslice0 = PsiSlice::full(coords0.nrows() * latent_dim, Some(latent_dim));
    let pslice1 = PsiSlice::full(coords1.nrows() * latent_dim, Some(latent_dim));
    registry.push(AnalyticPenaltyKind::Isometry(Arc::new(
        IsometryPenalty::new_euclidean(pslice0, p_out),
    )));
    registry.push(AnalyticPenaltyKind::Isometry(Arc::new(
        IsometryPenalty::new_euclidean(pslice1, p_out),
    )));

    let coords_per_atom = vec![coords0.clone(), coords1.clone()];
    let refreshed = refresh_isometry_caches_from_term(&registry, &term, &coords_per_atom).unwrap();
    assert_eq!(refreshed, 2, "both penalties should install second caches");

    let cache0 = match &registry.penalties[0] {
        AnalyticPenaltyKind::Isometry(p) => p
            .jacobian_cache()
            .expect("penalty 0 cache must be populated"),
        _ => panic!("expected isometry penalty at index 0"),
    };
    let cache1 = match &registry.penalties[1] {
        AnalyticPenaltyKind::Isometry(p) => p
            .jacobian_cache()
            .expect("penalty 1 cache must be populated"),
        _ => panic!("expected isometry penalty at index 1"),
    };

    // Penalty i must carry atom i's cache — not both atom 0's.
    assert_eq!(
        *cache0, *expected0,
        "penalty 0 must be refreshed against atom 0"
    );
    assert_eq!(
        *cache1, *expected1,
        "penalty 1 must be refreshed against atom 1 (regression: old find() paired it to atom 0)"
    );
    assert_ne!(
        *cache0, *cache1,
        "the two penalties must not collapse onto the same atom"
    );
}

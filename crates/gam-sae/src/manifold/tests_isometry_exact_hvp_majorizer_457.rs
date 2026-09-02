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


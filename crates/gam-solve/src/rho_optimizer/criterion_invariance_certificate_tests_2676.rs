// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration
// there): #2676 — the outer curvature certificate must not decide on a
// direction along which the criterion is EXACTLY constant. Scope comes from the
// parent via `use super::*`; the split is purely physical.
//
// ─── The defect, as a property rather than an instance ───
//
// `rho = log lambda` is a nonlinear reparameterisation, so for ANY smooth `V`
//
//     H_rho = diag(lambda) H_lambda diag(lambda) + diag(g_rho)
//
// holds exactly. Every criterion here sees `lambda` only through
// `sum_i lambda_i S_i`, so a `w` with `sum_i w_i S_i = 0` makes `V` constant
// along `lambda + s w`, i.e. `H_lambda w = 0`. Lift by `t = diag(lambda)^-1 w`:
//
//     t' H_rho t         = sum_k g_k t_k^2
//     t' (H_rho + diag|g|) t = 2 sum_k g_k^+ t_k^2
//
// and where the gradient components on `t`'s support are NEGATIVE the second
// line is exactly ZERO. So the certificate's `H + diag(|g|) PSD?` test is not
// merely close to failing on that direction — it is a numerical zero against a
// zero, and the verdict is the sign of the assembly residual.
//
// The gates below assert exactly that: a perturbation of HALF the gate's own
// gradient floor — the quantity it adds to the diagonal precisely so that
// residues of that size cannot be called a saddle — flips the undeflated
// verdict, and does not flip the deflated one.

use super::*;
use ndarray::{Array1, Array2, array};

/// THE IDENTITY the certificate's floor is supposed to protect against, and
/// the one direction where it protects by exactly ZERO.
///
/// `H + diag(|g|)` exists so a curvature residue of size `O(|g_k|)` cannot be
/// called a saddle. Along the criterion's invariance that protection is
/// identically cancelled:
///
///     t' (H_rho + diag|g|) t = t' diag(l) H_lambda diag(l) t + sum_k (g_k + |g_k|) t_k^2
///                            =            0                 +      2 sum_k g_k^+ t_k^2
///
/// and with the gradient NEGATIVE on `t`'s support the second term is zero too.
/// Everywhere else the same floor contributes at least `min_k |g_k|`. So the
/// floor is worth `|g|` on every direction except the one it is being asked to
/// adjudicate, where it is worth nothing.
#[test]
fn the_gradient_floor_protects_every_direction_except_the_invariance_2676() {
    let (h_rho, g_rho, invariance) = redundant_rho_system(0.75);
    let mut floored = h_rho.clone();
    for k in 0..3 {
        floored[[k, k]] += g_rho[k].abs();
    }
    let matrix_scale = h_rho.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    let t = invariance.column(0).to_owned();
    let on_invariance = t.dot(&floored.dot(&t));
    assert!(
        on_invariance.abs() <= 64.0 * f64::EPSILON * matrix_scale,
        "the floored quotient on the invariance must be zero to round-off of the matrix scale \
         {matrix_scale:.3e}, got {on_invariance:.6e}"
    );
    // And the floor really is doing work elsewhere: on the judged complement
    // the same matrix is bounded away from zero by orders.
    let judged = crate::penalty_invariance::judged_subspace_basis(3, &[], Some(&invariance))
        .expect("a 2-dimensional complement");
    let compressed = crate::penalty_invariance::compress_to_judged_subspace(&floored, &judged);
    use gam_linalg::faer_ndarray::FaerEigh;
    let (eigenvalues, _) = compressed.eigh(faer::Side::Lower).expect("eigh");
    let minimum = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        minimum > 1.0e-3,
        "the judged complement must carry real curvature (got {minimum:.6e}), or this fixture \
         is not separating the artifact from the criterion"
    );
}

/// THE ACCEPTANCE, stated as the property and not as one host's rounding.
///
/// Perturb the assembled `H_rho` ALONG the invariance by `+-delta` with
/// `delta = 0.5 * max_k |g_k|` — HALF the very floor this same test adds to the
/// diagonal to absorb `O(|g|)` residues, and orders below the matrix's own
/// scale. A gate that flips there is deciding on the gradient, not on the
/// curvature.
///
/// Undeflated: the two perturbations give OPPOSITE verdicts.
/// Deflated: both give `Some(true)`, and there is nothing left to flip.
#[test]
fn a_perturbation_below_the_certificates_own_floor_flips_the_undeflated_verdict_2676() {
    let (h_rho, g_rho, invariance) = redundant_rho_system(0.75);
    let matrix_scale = h_rho.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    let gradient_floor = g_rho.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    let delta = 0.5 * gradient_floor;
    assert!(
        delta < 1.0e-5 * matrix_scale,
        "the perturbation {delta:.3e} must be negligible against the matrix scale \
         {matrix_scale:.3e}, or the fixture is not making the point"
    );

    let mut undeflated = Vec::new();
    let mut deflated = Vec::new();
    for sign in [-1.0_f64, 1.0] {
        let t = invariance.column(0).to_owned();
        let mut perturbed = h_rho.clone();
        for i in 0..3 {
            for j in 0..3 {
                perturbed[[i, j]] += sign * delta * t[i] * t[j];
            }
        }
        undeflated.push(certificate_hessian_is_psd_off_railed_above_gradient_floor(
            &perturbed, &[], &g_rho, None,
        ));
        deflated.push(certificate_hessian_is_psd_off_railed_above_gradient_floor(
            &perturbed,
            &[],
            &g_rho,
            Some(&invariance),
        ));
    }

    assert_eq!(
        undeflated,
        vec![Some(false), Some(true)],
        "the pre-#2676 gate must be shown to flip under a {delta:.3e} perturbation — half its \
         own gradient floor — of a matrix whose scale is {matrix_scale:.3e}; if it does not, \
         this fixture no longer reproduces the defect and the gate below proves nothing"
    );
    assert_eq!(
        deflated,
        vec![Some(true), Some(true)],
        "deflating the criterion's own invariance must reach the SAME verdict on both signs: \
         the residual's sign is not evidence about the fit"
    );
}

/// The other half of the same statement: deflation must not make the gate
/// permissive. A genuine saddle — #2665's regime, `lambda_min` orders above the
/// gradient floor — still refuses with the invariance deflated, because it does
/// not live in the deflated subspace.
#[test]
fn a_genuine_saddle_still_refuses_with_the_invariance_deflated_2676() {
    let (h_rho, g_rho, invariance) = redundant_rho_system(0.75);
    // Plant negative curvature on the coordinate the invariance does NOT touch
    // (`t_1 = 0` by construction), so the saddle is entirely inside the judged
    // complement.
    let mut saddle = h_rho.clone();
    saddle[[1, 1]] = -1.5;
    assert_eq!(
        certificate_hessian_is_psd_off_railed_above_gradient_floor(
            &saddle,
            &[],
            &g_rho,
            Some(&invariance),
        ),
        Some(false),
        "a -1.5 direction against a {:.1e} gradient floor must refuse whatever is deflated",
        g_rho.iter().copied().map(f64::abs).fold(0.0_f64, f64::max),
    );
    // And the same matrix without the saddle certifies, so the refusal above is
    // the planted curvature and not the deflation machinery.
    assert_eq!(
        certificate_hessian_is_psd_off_railed_above_gradient_floor(
            &h_rho,
            &[],
            &g_rho,
            Some(&invariance),
        ),
        Some(true),
    );
}

/// Inertness: with `invariance = None` every verdict is what it was before
/// #2676, including on matrices where the deflated answer differs. This is the
/// guarantee that ordinary models — every objective that declares no invariance
/// — do not move.
#[test]
fn no_declared_invariance_reproduces_the_pre_2676_verdicts_2676() {
    let (h_rho, g_rho, invariance) = redundant_rho_system(0.75);
    // A matrix whose undeflated verdict is `false` and whose deflated verdict
    // is `true`: the two must not be confused.
    let t = invariance.column(0).to_owned();
    let gradient_floor = g_rho.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    let mut perturbed = h_rho.clone();
    for i in 0..3 {
        for j in 0..3 {
            perturbed[[i, j]] -= 0.5 * gradient_floor * t[i] * t[j];
        }
    }
    assert_eq!(
        certificate_hessian_is_psd_off_railed_above_gradient_floor(&perturbed, &[], &g_rho, None),
        Some(false),
        "the undeflated gate must still see the perturbed invariance as indefinite"
    );
    assert_eq!(
        certificate_hessian_is_psd_off_railed_above_gradient_floor(
            &perturbed,
            &[],
            &g_rho,
            Some(&invariance),
        ),
        Some(true),
    );
    // Railed-coordinate handling is unchanged on the `None` path: the judged
    // block is the interior sub-block, bit for bit.
    let indefinite = array![
        [1.0_f64, 0.0, 0.0],
        [0.0, -3.0, 0.0],
        [0.0, 0.0, 2.0],
    ];
    let zero_gradient = Array1::<f64>::zeros(3);
    assert_eq!(
        certificate_hessian_is_psd_off_railed_above_gradient_floor(
            &indefinite,
            &[1],
            &zero_gradient,
            None,
        ),
        Some(true),
        "excluding the negative coordinate leaves a PD sub-block"
    );
    assert_eq!(
        certificate_hessian_is_psd_off_railed_above_gradient_floor(
            &indefinite,
            &[0],
            &zero_gradient,
            None,
        ),
        Some(false),
        "excluding a positive coordinate leaves the negative one in the judged block"
    );
}

/// The clearance record must speak about the block the verdict was reached on.
/// Before #2676 it reported the raw interior minimum beside a verdict taken
/// elsewhere, which is how `[INDEF-HESS]` lines came to name a direction the
/// decision did not involve.
#[test]
fn the_recorded_minimum_is_taken_on_the_judged_block_2676() {
    let (h_rho, g_rho, invariance) = redundant_rho_system(0.75);
    let raw = interior_curvature_floor_clearance(&h_rho, &[], &g_rho, None)
        .expect("a finite 3x3 block has a clearance");
    let judged = interior_curvature_floor_clearance(&h_rho, &[], &g_rho, Some(&invariance))
        .expect("a finite 3x3 block has a clearance");
    assert!(
        raw.interior_min_eigenvalue < 1.0e-4,
        "the raw minimum is the chain-rule artifact, of gradient scale (got {:.3e})",
        raw.interior_min_eigenvalue
    );
    assert!(
        judged.interior_min_eigenvalue > 1.0e-3,
        "the judged minimum must be the criterion's own curvature, orders above it (got {:.3e})",
        judged.interior_min_eigenvalue
    );
    assert_eq!(
        raw.gradient_floor, judged.gradient_floor,
        "the floor is a property of the gradient over the judged COORDINATES and does not move"
    );
}

/// THE GUARANTEE, not an instance of it: **Cauchy interlacing bounds what
/// deflation can hide.**
///
/// `Z' H Z` is the compression of a symmetric `H` onto a subspace of
/// codimension `d`, so with eigenvalues written ascending,
///
/// ```text
///     lambda_1(H) <= lambda_1(Z'HZ) <= lambda_{d+1}(H).
/// ```
///
/// Deflating `d` directions can therefore "lose" at most the `d` smallest
/// eigenvalues, and NEVER an eigenvalue beyond that. With the one-dimensional
/// invariance this issue is about, a matrix carrying TWO negative eigenvalues
/// still refuses: the second one survives the compression by the upper bound
/// above. That is the general form of
/// `a_genuine_saddle_still_refuses_with_the_invariance_deflated`, which
/// exhibits a single instance.
///
/// Swept over a deterministic family of matrices and deflation directions so
/// the bound is asserted as a law rather than at one point.
#[test]
fn deflation_cannot_hide_more_than_the_smallest_eigenvalue_cauchy_interlacing_2676() {
    use gam_linalg::faer_ndarray::FaerEigh;

    // A deterministic pseudo-random symmetric family; no RNG, so this is the
    // same sweep on every host.
    let mut state = 0x2676_u64;
    let mut next = move || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let dimension = 5usize;
    for case in 0..24 {
        let mut h = Array2::<f64>::zeros((dimension, dimension));
        for i in 0..dimension {
            for j in i..dimension {
                let value = next() * 4.0;
                h[[i, j]] = value;
                h[[j, i]] = value;
            }
        }
        // A deflation direction, orthonormalised by the module under test.
        let mut raw = Array2::<f64>::zeros((dimension, 1));
        for row in 0..dimension {
            raw[[row, 0]] = next();
        }
        let Some(deflation) = crate::penalty_invariance::orthonormalize_columns(&raw) else {
            continue;
        };
        let Some(judged) =
            crate::penalty_invariance::judged_subspace_basis(dimension, &[], Some(&deflation))
        else {
            continue;
        };
        assert_eq!(judged.ncols(), dimension - 1, "case {case}");
        let compressed = crate::penalty_invariance::compress_to_judged_subspace(&h, &judged);

        let (full, _) = h.eigh(faer::Side::Lower).expect("eigh");
        let (part, _) = compressed.eigh(faer::Side::Lower).expect("eigh");
        let mut full_sorted: Vec<f64> = full.to_vec();
        full_sorted.sort_by(f64::total_cmp);
        let mut part_sorted: Vec<f64> = part.to_vec();
        part_sorted.sort_by(f64::total_cmp);
        let scale = h.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
        let slack = 1.0e-9 * scale.max(1.0);
        assert!(
            part_sorted[0] >= full_sorted[0] - slack,
            "case {case}: interlacing lower bound violated ({:.6e} < {:.6e})",
            part_sorted[0],
            full_sorted[0],
        );
        assert!(
            part_sorted[0] <= full_sorted[1] + slack,
            "case {case}: deflating ONE direction must not push the minimum past the SECOND \
             smallest eigenvalue ({:.6e} > {:.6e}) — that upper bound is what stops a matrix \
             with two negative directions from being certified",
            part_sorted[0],
            full_sorted[1],
        );
    }
}

/// The consequence that matters here, stated directly: a spectrum with TWO
/// negative directions still refuses however the single invariance is chosen.
#[test]
fn two_negative_directions_still_refuse_with_one_direction_deflated_2676() {
    let (h_rho, g_rho, invariance) = redundant_rho_system(0.75);
    // The invariance's own (artifact) curvature is already negative here; plant
    // a SECOND, genuine negative direction in the complement.
    let mut two_negatives = h_rho.clone();
    two_negatives[[1, 1]] = -0.75;
    assert_eq!(
        certificate_hessian_is_psd_off_railed_above_gradient_floor(
            &two_negatives,
            &[],
            &g_rho,
            Some(&invariance),
        ),
        Some(false),
        "deflating one direction cannot certify a matrix carrying two negative ones"
    );
}

/// The instrument itself: with the pair consistent the measurement is zero, and
/// with an error injected on the deflated direction it recovers it exactly.
#[test]
fn the_outer_certificate_measures_its_own_assembly_error_2748() {
    let (consistent, gradient, invariance) = outer_resolution_fixture(1.0e-6, 0.0);
    let clean = measured_outer_curvature_resolution(&consistent, &[], &gradient, Some(&invariance));
    // Judged against the arithmetic this measurement itself runs at, not
    // against literal zero: the residual is a Rayleigh quotient of a matrix
    // whose scale is 1, so its own round-off is what a clean pair returns.
    let matrix_scale = consistent
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    assert!(
        clean <= 64.0 * f64::EPSILON * matrix_scale,
        "a consistent (H, g) pair on the invariance must measure round-off of the matrix \
         scale {matrix_scale:.6e}; got {clean:.6e}"
    );

    let injected = 1.0e-6_f64;
    let (perturbed, gradient, invariance) = outer_resolution_fixture(1.0e-6, injected);
    let measured =
        measured_outer_curvature_resolution(&perturbed, &[], &gradient, Some(&invariance));
    assert!(
        (measured - injected).abs() <= 64.0 * f64::EPSILON,
        "expected the injected {injected:.6e}, measured {measured:.6e}"
    );

    // And it is unavailable, not zero-by-assumption, when no invariance is
    // declared: that path keeps the historical shift, which is the inertness
    // guarantee for every model without a redundant penalty map.
    assert_eq!(
        measured_outer_curvature_resolution(&perturbed, &[], &gradient, None),
        0.0
    );
}

/// The verdict moves with the MEASUREMENT and with nothing else: the same
/// judged curvature is refused when the assembly is clean and admitted when the
/// assembly has demonstrated, on its own exactly-zero identity, that it cannot
/// resolve it.
#[test]
fn a_judged_curvature_inside_the_measured_resolution_is_not_a_saddle_2748() {
    let curvature = 5.0e-7_f64;

    // Clean assembly: nothing measured, the historical sqrt(eps) shift decides,
    // and a 5e-7 negative direction is a saddle.
    let (clean, gradient, invariance) = outer_resolution_fixture(curvature, 0.0);
    assert_eq!(
        certificate_hessian_is_psd_off_railed_above_gradient_floor(
            &clean,
            &[],
            &gradient,
            Some(&invariance),
        ),
        Some(false),
        "with no measured error the historical shift (1.49e-8) still refuses 5e-7"
    );

    // Same judged curvature, same gradient, but the assembly has measured
    // itself inconsistent by 1e-6 on a direction whose exact answer is zero.
    let (noisy, gradient, invariance) = outer_resolution_fixture(curvature, 1.0e-6);
    assert_eq!(
        certificate_hessian_is_psd_off_railed_above_gradient_floor(
            &noisy,
            &[],
            &gradient,
            Some(&invariance),
        ),
        Some(true),
        "a curvature inside the assembly's own measured error is unresolved, not a saddle"
    );

    // The control: an order MORE negative curvature against the same measured
    // 1e-6 is outside it and still refuses. A resolution cannot swallow a
    // quantity above it.
    let (real_saddle, gradient, invariance) = outer_resolution_fixture(1.0e-5, 1.0e-6);
    assert_eq!(
        certificate_hessian_is_psd_off_railed_above_gradient_floor(
            &real_saddle,
            &[],
            &gradient,
            Some(&invariance),
        ),
        Some(false),
        "a curvature an order above the measured resolution is a genuine saddle"
    );
}

/// The measured resolution can only ADMIT, never refuse: `max` with the
/// historical shift means every point the pre-#2748 rule accepted is still
/// accepted, whatever the measurement says.
#[test]
fn the_measured_resolution_can_only_admit_2748() {
    for curvature in [1.0e-12_f64, 1.0e-9, 1.0e-7, 1.0e-4, 1.0e-1] {
        let (clean, gradient, invariance) = outer_resolution_fixture(curvature, 0.0);
        let before = certificate_hessian_is_psd_off_railed_above_gradient_floor(
            &clean,
            &[],
            &gradient,
            Some(&invariance),
        );
        let (noisy, gradient, invariance) = outer_resolution_fixture(curvature, 1.0e-6);
        let after = certificate_hessian_is_psd_off_railed_above_gradient_floor(
            &noisy,
            &[],
            &gradient,
            Some(&invariance),
        );
        if before == Some(true) {
            assert_eq!(
                after,
                Some(true),
                "curvature {curvature:.1e} was accepted before the measurement and must \
                 stay accepted"
            );
        }
    }
}

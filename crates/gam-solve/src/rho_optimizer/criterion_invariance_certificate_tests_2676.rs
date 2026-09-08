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

use ndarray::Array2;

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


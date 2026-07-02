//! Split-out sibling test module, carved from `tests.rs` to keep that tracked
//! file under the 10k-line build gate (`build.rs` `MAX_TRACKED_FILE_LINES`),
//! mirroring the existing `tests_parallelism_invariance_1557` /
//! `tests_logdet_adjoint_780` splits. These three regressions need only the
//! manifold re-exports plus the shared `gamma_fd_tiny_fixture` helper, which
//! remains defined in `tests.rs`:
//!
//! * `streaming_polar_refresh_reorients_frame` — the closed-form streaming polar
//!   frame refresh re-orients the decoder frame toward an accumulated
//!   cross-moment span while staying column-orthonormal.
//! * `small_p_zero_decoder_stays_full_b` — a zero decoder at small ambient `p`
//!   keeps the full-`B` border (frame activation is a no-op).
//! * `forward_alpha_data_derivative_skips_ungated_atom_1026` — the learnable-α
//!   forward data-derivative gives an ungated background-tier atom zero
//!   α-sensitivity (FD-checked against the data NLL).

use super::*;
use super::tests::gamma_fd_tiny_fixture;
use approx::assert_abs_diff_eq;

/// Streaming polar refresh from an accumulated cross-moment re-orients the
/// frame toward the cross-moment span and keeps `B_k`'s in-span component
/// while staying column-orthonormal (the closed-form streaming step).
#[test]
pub(crate) fn streaming_polar_refresh_reorients_frame() {
    let m = 4usize;
    let p = 12usize;
    let r = 2usize;
    let mut frame0 = Array2::<f64>::zeros((p, r));
    frame0[[0, 0]] = 1.0;
    frame0[[1, 1]] = 1.0;
    let mut c0 = Array2::<f64>::zeros((m, r));
    for mu in 0..m {
        c0[[mu, 0]] = 1.0 + mu as f64;
        c0[[mu, 1]] = 0.5 - mu as f64;
    }
    let decoder = fast_abt(&c0, &frame0);
    let mut phi = Array2::<f64>::zeros((m, m));
    let mut jet = Array3::<f64>::zeros((m, m, 1));
    for mu in 0..m {
        phi[[mu, mu]] = 1.0;
        jet[[mu, mu, 0]] = 1.0;
    }
    let s_raw = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    let mut atom = SaeManifoldAtom::new(
        "stream",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        decoder,
        s_raw,
    )
    .unwrap();
    atom.maybe_activate_decoder_frame().expect("activate");
    // New cross-moment pointing at axes {2,3}: refreshed frame must span them.
    let mut cross = Array2::<f64>::zeros((p, r));
    cross[[2, 0]] = 3.0;
    cross[[3, 1]] = 2.0;
    atom.refresh_frame_from_cross_moment(cross.view())
        .expect("refresh");
    let frame = atom.decoder_frame.as_ref().expect("frame");
    // Frame stays orthonormal.
    let gram = fast_atb(&frame.frame().to_owned(), &frame.frame().to_owned());
    for i in 0..r {
        for j in 0..r {
            let expect = if i == j { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(gram[[i, j]], expect, epsilon = 1.0e-9);
        }
    }
    // Refreshed span aligns with the cross-moment axes {2,3} (angle ~0).
    let mut target_span = Array2::<f64>::zeros((p, r));
    target_span[[2, 0]] = 1.0;
    target_span[[3, 1]] = 1.0;
    let angle = frame
        .max_principal_angle(target_span.view())
        .expect("angle");
    assert_abs_diff_eq!(angle, 0.0, epsilon = 1.0e-9);
}

#[test]
pub(crate) fn small_p_zero_decoder_stays_full_b() {
    let m = 3usize;
    let p = 8usize;
    let mut phi = Array2::<f64>::zeros((m, m));
    let mut jet = Array3::<f64>::zeros((m, m, 1));
    for row in 0..m {
        phi[[row, row]] = 1.0;
        jet[[row, row, 0]] = 1.0;
    }
    let smooth_penalty = gam_terms::basis::create_difference_penalty_matrix(m, 2, None).unwrap();
    let mut atom = SaeManifoldAtom::new(
        "small-p-zero",
        SaeAtomBasisKind::EuclideanPatch,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((m, p)),
        smooth_penalty,
    )
    .unwrap();

    assert_eq!(atom.decoder_frame_activation_rank().unwrap(), None);
    assert_eq!(atom.maybe_activate_decoder_frame().unwrap(), None);
    assert_eq!(atom.border_frame_rank(), p);
}

/// #1026/#1417: the learnable-α forward data-derivative must give an UNGATED
/// (background-tier) atom ZERO α-sensitivity. An ungated atom's gate is forced
/// to 1.0 (`has_ungated` override), so its mass `a_k ≡ 1` is α-independent and
/// `∂a_k/∂logα = 0` — the `π_k(α)` chain applies only to gated atoms. Before the
/// fix the code credited the ungated atom `(1/π_k)·dπ_k/dρ ≠ 0`, biasing the
/// data α-gradient. FD-check the analytic against the data NLL ½Σ‖fitted−target‖²
/// (where the ungated atom's reconstruction is α-constant) on a 2-atom fixture
/// with atom 1 ungated.
#[test]
pub(crate) fn forward_alpha_data_derivative_skips_ungated_atom_1026() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, true);
    // Atom 1 is the #1026 ungated background tier (gate ≡ 1).
    term.assignment = term
        .assignment
        .clone()
        .with_ungated(vec![false, true])
        .unwrap();
    rho.log_lambda_sparse = 0.3;

    let analytic = term
        .learnable_ibp_forward_alpha_data_derivative(&rho, target.view())
        .unwrap();

    // FD of the data NLL ½Σ‖fitted−target‖² wrt ρ₀ (= logα offset, since
    // α = α₀·e^{ρ₀} ⇒ ∂logα/∂ρ₀ = 1). The ungated atom's fitted contribution is
    // α-constant, so the FD sees only the gated atom's π-derivative.
    let data_nll = |t: &SaeManifoldTerm, r: &SaeManifoldRho| -> f64 {
        let fitted = t.try_fitted_for_rho(r).unwrap();
        let mut s = 0.0_f64;
        for row in 0..fitted.nrows() {
            for c in 0..fitted.ncols() {
                let d = fitted[[row, c]] - target[[row, c]];
                s += d * d;
            }
        }
        0.5 * s
    };
    let h = 1.0e-6;
    let mut rp = rho.clone();
    let mut rm = rho.clone();
    rp.log_lambda_sparse += h;
    rm.log_lambda_sparse -= h;
    let fd = (data_nll(&term, &rp) - data_nll(&term, &rm)) / (2.0 * h);
    assert!(
        (analytic - fd).abs() <= 1.0e-5 * (1.0 + fd.abs()),
        "forward-α data derivative must match FD with an ungated atom: \
         analytic={analytic:.8e}, fd={fd:.8e}"
    );
    // Non-vacuity: the gated atom must give a materially nonzero derivative
    // (otherwise the test would pass even if everything were zeroed).
    assert!(
        fd.abs() > 1.0e-6,
        "fixture must exercise a nonzero gated-atom α-derivative; fd={fd:.3e}"
    );
}

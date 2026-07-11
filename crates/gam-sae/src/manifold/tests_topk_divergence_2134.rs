//! #2134 wall #2 — the softmax `top_k > 1` JOINT-fit divergence, and the
//! regression guard for the [`SaeManifoldTerm::softmax_active_plan`] fix.
//!
//! On genuinely MULTI-ACTIVE data (each row is the SUM of two disjoint circles,
//! so a faithful reconstruction needs ~2 atoms carrying every row) a softmax
//! dictionary fit with a HARD `top_k = 2` fit-time cap co-collapses: the per-row
//! top-2 support flips across outer Newton iterations, the truncated,
//! non-renormalized softmax reconstruction `Σ_{k∈top_2} a_k B_k g_k` is a
//! support-DEPENDENT surrogate (nothing forces the softmax to concentrate onto
//! exactly two atoms per row), the objective jumps at each re-selection, monotone
//! descent breaks, and the joint (t, B) solve oscillates into the signal-free
//! null. The DENSE softmax fit on the SAME data — support-independent objective —
//! recovers materially positive reconstruction EV.
//!
//! The fix (`softmax_active_plan`) routes a JOINT-fit `cap >= 2` through the
//! memory-budget lever only (dense at this small K), so the capped fit now
//! matches the dense fit; `top_k` becomes a post-fit projection. The
//! winner-take-all `cap = 1` (#2132 saddle escape) and the FIXED-DECODER encode
//! cap (the large-K compact contract) are untouched — so this is a strict
//! recovery of the multi-active joint fit, not a weakening of the sparse lanes.

use super::tests::deterministic_circle_noise;
use super::*;

/// Multi-active target: circle A on the even ambient columns, circle B on the
/// odd ones (disjoint deterministic near-orthonormal 2-frames), driven by two
/// incommensurate phases so the circles are not row-aligned. Each ROW is the SUM
/// of BOTH circles, so ~2 atoms must be simultaneously active to reconstruct it
/// (the exact regime a hard `top_k = 2` cap is meant to serve). Columns are
/// standardized to zero mean / unit variance (the whitening proxy).
fn two_circle_multi_active_target(n: usize, p: usize, sigma: f64) -> Array2<f64> {
    let mut fa = Array2::<f64>::zeros((2, p));
    let mut fb = Array2::<f64>::zeros((2, p));
    for j in 0..p {
        if j % 2 == 0 {
            fa[[0, j]] = deterministic_circle_noise(j, 0);
            fa[[1, j]] = deterministic_circle_noise(j, 1);
        } else {
            fb[[0, j]] = deterministic_circle_noise(j, 2);
            fb[[1, j]] = deterministic_circle_noise(j, 3);
        }
    }
    for f in [&mut fa, &mut fb] {
        for r in 0..2 {
            let nrm = (0..p).map(|j| f[[r, j]] * f[[r, j]]).sum::<f64>().sqrt();
            for j in 0..p {
                f[[r, j]] /= nrm.max(1.0e-300);
            }
        }
    }
    let mut z = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        let ta = std::f64::consts::TAU * (row as f64) / (n as f64);
        let tb = std::f64::consts::TAU * (2.0 * row as f64 + 0.37) / (n as f64);
        let (ca, sa) = (ta.cos(), ta.sin());
        let (cb, sb) = (tb.cos(), tb.sin());
        for j in 0..p {
            z[[row, j]] = ca * fa[[0, j]]
                + sa * fa[[1, j]]
                + cb * fb[[0, j]]
                + sb * fb[[1, j]]
                + sigma * deterministic_circle_noise(row, j + 7);
        }
    }
    for j in 0..p {
        let mut mean = 0.0_f64;
        for row in 0..n {
            mean += z[[row, j]];
        }
        mean /= n as f64;
        let mut var = 0.0_f64;
        for row in 0..n {
            let d = z[[row, j]] - mean;
            var += d * d;
        }
        let sd = (var / n as f64).sqrt().max(1.0e-12);
        for row in 0..n {
            z[[row, j]] = (z[[row, j]] - mean) / sd;
        }
    }
    z
}

/// Fresh softmax K-atom periodic term on the multi-active target, production PCA
/// seed, decoders cold at zero, and a mild round-robin favored-atom routing seed
/// (the stand-in for the alternating routing refine: every atom carries mass, no
/// degenerate symmetry). `active_cap` installs the fit-time softmax top-k cap.
fn softmax_multi_active_term(
    n: usize,
    p: usize,
    m: usize,
    k: usize,
    active_cap: Option<usize>,
) -> (SaeManifoldTerm, Array2<f64>) {
    let d = 1usize;
    let target = two_circle_multi_active_target(n, p, 0.05);
    let basis_kinds = vec![SaeAtomBasisKind::Periodic; k];
    let dims = vec![d; k];
    let seed = sae_pca_seed_initial_coords(target.view(), &basis_kinds, &dims).unwrap();
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap());
    let mut basis_values = Array3::<f64>::zeros((k, n, m));
    let mut basis_jacobian = Array4::<f64>::zeros((k, n, m, d));
    let decoder = Array3::<f64>::zeros((k, m, p));
    let mut penalties = Array3::<f64>::zeros((k, m, m));
    let mut coords_vec: Vec<Array2<f64>> = Vec::new();
    for atom in 0..k {
        let coords = seed.slice(s![atom, .., 0..d]).to_owned();
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        basis_values.slice_mut(s![atom, .., ..]).assign(&phi);
        basis_jacobian.slice_mut(s![atom, .., .., ..]).assign(&jet);
        penalties
            .slice_mut(s![atom, .., ..])
            .assign(&Array2::<f64>::eye(m));
        coords_vec.push(coords);
    }
    // Round-robin routing seed with a MODERATE favored bump: enough to break the
    // degenerate uniform saddle, but spread enough that the softmax genuinely
    // distributes mass across more than `top_k` atoms per row (favored ≈ 0.60,
    // each other ≈ 0.13 at K=4) — the regime where the hard top-k truncation
    // drops material mass and the support competes/flips.
    let mut logits = Array2::<f64>::zeros((n, k));
    for row in 0..n {
        logits[[row, row % k]] = 1.5;
    }
    let mut evaluators: Vec<Option<Arc<dyn SaeBasisSecondJet>>> = Vec::new();
    for _ in 0..k {
        evaluators.push(Some(evaluator.clone()));
    }
    let mut term = term_from_padded_blocks_with_mode(
        n,
        p,
        &basis_kinds,
        basis_values.view(),
        basis_jacobian.view(),
        &vec![m; k],
        &dims,
        decoder.view(),
        penalties.view(),
        logits.view(),
        &coords_vec,
        AssignmentMode::softmax(1.0),
        &evaluators,
    )
    .unwrap();
    term.set_softmax_active_cap(active_cap);
    (term, target)
}

/// Run the joint (t, B) Arrow-Schur fit and return the dictionary reconstruction
/// EV at the converged state.
fn fit_and_ev(mut term: SaeManifoldTerm, target: &Array2<f64>) -> f64 {
    let k = term.k_atoms();
    let mut rho = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1); k]);
    let loss = term
        .run_joint_fit_arrow_schur(target.view(), &mut rho, None, 60, 0.05, 1.0e-3, 1.0e-3)
        .expect("softmax joint fit must complete without a fatal solver abort");
    assert!(loss.total().is_finite(), "loss must stay finite");
    term.dictionary_reconstruction_ev(target.view(), &rho)
        .expect("reconstruction EV must be defined")
}

/// #2134 — a softmax `top_k = 2` JOINT fit on multi-active (two-circle-sum) data
/// must recover EV comparable to the DENSE softmax fit on the identical data, NOT
/// co-collapse. The dense fit is the in-test reference so the guard self-calibrates
/// to whatever EV this fixture supports (no brittle absolute bar): the capped fit
/// is a regression iff it lands materially below the dense fit.
///
/// Before the fix the hard top-2 cap made the compact reconstruction support-
/// dependent; the per-row top-2 support flipped across outer iterations and the
/// dictionary co-collapsed toward the signal-free null (EV ≈ 0) while the dense
/// fit recovered the planted rank-4 subspace. The fix routes the joint-fit
/// `cap >= 2` through the budget lever (dense here), so the two fits now agree.
#[test]
fn softmax_top_k_2_joint_fit_matches_dense_on_multi_active_2134() {
    let n = 96usize;
    let p = 16usize;
    let m = 5usize; // [1, sin2πt, cos2πt, sin4πt, cos4πt]
    let k = 4usize;

    // Dense reference (no fit-time cap): the support-independent softmax objective.
    let (dense_term, target) = softmax_multi_active_term(n, p, m, k, None);
    let dense_ev = fit_and_ev(dense_term, &target);

    // The reported path: a hard `top_k = 2` fit-time cap on the SAME data.
    let (capped_term, target2) = softmax_multi_active_term(n, p, m, k, Some(2));
    assert_eq!(target, target2, "both fits must see identical data");
    let capped_ev = fit_and_ev(capped_term, &target2);

    eprintln!(
        "[#2134 repro] two-circle multi-active K={k}: dense_ev={dense_ev:.4} \
         top_k2_ev={capped_ev:.4} gap={:.4}",
        dense_ev - capped_ev
    );

    // Sanity: the dense softmax fit must actually engage the planted structure
    // (two disjoint circles span a rank-4 subspace of the whitened cloud).
    assert!(
        dense_ev > 0.15,
        "dense softmax fit did not recover the multi-active structure (EV={dense_ev:.4}); \
         the fixture, not the top-k cap, is the problem"
    );
    // The regression guard: the `top_k = 2` fit must not co-collapse relative to
    // the dense fit. A support-dependent truncation drives EV toward the null
    // floor (≈0), which is a large gap below the dense reference.
    assert!(
        capped_ev > dense_ev - 0.10,
        "softmax top_k=2 joint fit co-collapsed on multi-active data: EV={capped_ev:.4} \
         vs dense EV={dense_ev:.4} (#2134 top_k>1 divergence — the hard fit-time cap made \
         the compact reconstruction support-dependent; the per-row top-2 support flips across \
         outer iterations and the dictionary collapses to the null)"
    );
}

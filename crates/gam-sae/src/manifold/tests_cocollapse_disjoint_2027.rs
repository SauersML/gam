//! #2027 — deterministic repro for the K≥2 whitened dictionary CO-COLLAPSE, and
//! the regression guard for the disjoint-subspace / ownership-anchor / reseed-
//! hysteresis fix.
//!
//! Two planted circles live in DISJOINT 2-planes of an ambient `p`-dim cloud; the
//! per-column-standardized ("whitened") target is their sum, so a faithful K=2
//! reconstruction REQUIRES both atoms to carry signal on different subspaces.
//! Before the fix the joint decoder refit at the co-collapse reseed re-spread one
//! residual direction across both atoms and the gate let them trade rows, so the
//! dictionary re-symmetrised into a single shared basin and the reconstruction EV
//! sat near the signal-free null floor. With the greedy disjoint-subspace decoder
//! refit + soft row-ownership anchor + reseed cooldown the two atoms hold distinct
//! territories and the fit recovers a materially positive EV.

use super::tests::deterministic_circle_noise;
use super::*;

/// Whitened two-circle target: circle A lives on the even ambient columns, circle
/// B on the odd ones (disjoint deterministic near-orthonormal 2-frames), driven by
/// two INCOMMENSURATE phases so the circles are not row-aligned. Each column is
/// standardized to zero mean / unit variance (the whitening proxy that puts both
/// circles on a common scale, the regime the real-data co-collapse lives in).
fn two_circle_whitened_target(n: usize, p: usize, sigma: f64) -> Array2<f64> {
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

/// Build a fresh K=2 periodic term seeded (production PCA seed) from the whitened
/// two-circle target, decoders cold at zero.
fn two_circle_k2_term(n: usize, p: usize, m: usize) -> (SaeManifoldTerm, Array2<f64>) {
    let d = 1usize;
    let k = 2usize;
    let target = two_circle_whitened_target(n, p, 0.05);
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
    let logits = Array2::<f64>::zeros((n, k));
    let mut evaluators: Vec<Option<Arc<dyn SaeBasisSecondJet>>> = Vec::new();
    for _ in 0..k {
        evaluators.push(Some(evaluator.clone()));
    }
    let term = term_from_padded_blocks_with_mode(
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
        AssignmentMode::ibp_map(1.0, 1.0, false),
        &evaluators,
    )
    .unwrap();
    (term, target)
}

/// Generalized K=`k` build of the disjoint two-circle fixture (decoders cold at
/// zero) — used by the majority-collapse breach regression.
fn two_circle_kn_term(n: usize, p: usize, m: usize, k: usize) -> SaeManifoldTerm {
    let d = 1usize;
    let target = two_circle_whitened_target(n, p, 0.05);
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
    let logits = Array2::<f64>::zeros((n, k));
    let mut evaluators: Vec<Option<Arc<dyn SaeBasisSecondJet>>> = Vec::new();
    for _ in 0..k {
        evaluators.push(Some(evaluator.clone()));
    }
    term_from_padded_blocks_with_mode(
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
        AssignmentMode::ibp_map(1.0, 1.0, false),
        &evaluators,
    )
    .unwrap()
}

/// #1939 regression — the breach reference must NOT degenerate when the MAJORITY
/// of atoms collapse. Real IBP data is `‖B‖=[2.6, ~0, ~0]` (K=3, two co-vanished):
/// the MEDIAN norm is set BY the collapsed atoms (tiny or 0), so a median-keyed
/// `1e-3·median` floor sits below them and `retract_collapsed_decoders_in_loop`
/// would skip the collapse — a silent no-op on exactly the failure it targets.
/// Keying on the max survivor must retract the two collapsed atoms while leaving
/// the healthy one alone. Uses tiny-NONZERO collapsed norms (5e-4), the case a
/// median-fallback-only fix would still miss (median 5e-4 > 0).
#[test]
fn cone_atom_breach_survives_majority_collapse() {
    let mut term = two_circle_kn_term(48, 8, 5, 3);
    // Put all decoder mass on one entry so each atom's norm is exactly the target:
    // atom 0 healthy (2.6), atoms 1 & 2 collapsed but RETRACTABLE (5e-4, above the
    // direction floor 1e-12·2.6). Median = 5e-4 (nonzero); only the max reference
    // (2.6) flags 1 & 2.
    for (idx, target_norm) in [(0usize, 2.6_f64), (1, 5.0e-4), (2, 5.0e-4)] {
        let dec = &mut term.atoms[idx].decoder_coefficients;
        for v in dec.iter_mut() {
            *v = 0.0;
        }
        dec[[0, 0]] = target_norm;
    }
    let retracted = term.retract_collapsed_decoders_in_loop();
    assert_eq!(
        retracted, 2,
        "the two majority-collapsed atoms must be retracted against the max survivor, \
         not skipped because the median is set by the collapse"
    );
    let norm = |t: &SaeManifoldTerm, i: usize| -> f64 {
        t.atoms[i]
            .decoder_coefficients
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
    };
    assert!(
        (norm(&term, 0) - 2.6).abs() < 1.0e-9,
        "the healthy survivor must be untouched"
    );
    for idx in [1usize, 2] {
        assert!(
            (norm(&term, idx) - 1.0).abs() < 1.0e-9,
            "retracted atom {idx} must be unit-Frobenius"
        );
    }
}

/// The K=2 whitened two-circle fit must recover a materially positive
/// reconstruction EV — NOT co-collapse to the signal-free null floor. Two disjoint
/// circles together span a rank-4 subspace of the whitened cloud, so an honest K=2
/// dictionary explains a large fraction of the variance (the torch proxy reaches
/// ≈0.47 on the sibling nursery experiment). The disjoint-subspace decoder refit,
/// row-ownership anchor, and reseed cooldown keep the two atoms on distinct
/// territories through the joint solve.
#[test]
pub(crate) fn two_circle_whitened_k2_recovers_disjoint_signal_2027() {
    let n = 96usize;
    let p = 16usize;
    let m = 5usize; // [1, sin2πt, cos2πt, sin4πt, cos4πt]
    let (mut term, target) = two_circle_k2_term(n, p, m);

    let mut rho = SaeManifoldRho::new(
        0.0,
        -6.0,
        vec![Array1::<f64>::zeros(1), Array1::<f64>::zeros(1)],
    );
    let loss = term
        .run_joint_fit_arrow_schur(target.view(), &mut rho, None, 60, 0.05, 1.0e-3, 1.0e-3)
        .unwrap();
    assert!(loss.total().is_finite(), "loss must stay finite");

    let ev = term
        .dictionary_reconstruction_ev(target.view(), &rho)
        .unwrap();
    eprintln!(
        "[#2027 repro] K=2 whitened two-circle EV = {ev:.4}, cocollapse_reseeds = {}",
        term.dictionary_cocollapse_reseeds
    );
    assert!(
        ev > 0.20,
        "K=2 whitened two-circle dictionary co-collapsed: EV={ev:.4} (expected > 0.20; \
         two disjoint circles span a rank-4 subspace that an honest K=2 fit recovers)"
    );
}

/// The greedy disjoint-subspace decoder refit must, on a co-collapsed reseed,
/// leave BOTH atoms carrying material decoder norm — never let one atom take all
/// the residual while the other stays ≈0 (the relative-norm collapse the joint
/// refit permitted). A direct unit check of `refit_decoder_sequential_deflation`.
#[test]
pub(crate) fn sequential_deflation_gives_both_atoms_material_norm_2027() {
    let n = 96usize;
    let p = 16usize;
    let m = 5usize;
    let (mut term, target) = two_circle_k2_term(n, p, m);
    let rho = SaeManifoldRho::new(
        0.0,
        -6.0,
        vec![Array1::<f64>::zeros(1), Array1::<f64>::zeros(1)],
    );
    term.refit_decoder_sequential_deflation(target.view(), &rho)
        .unwrap();
    let mut norms = [0.0_f64; 2];
    for (atom_idx, atom) in term.atoms.iter().enumerate() {
        norms[atom_idx] = atom
            .decoder_coefficients
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
    }
    let (lo, hi) = if norms[0] <= norms[1] {
        (norms[0], norms[1])
    } else {
        (norms[1], norms[0])
    };
    eprintln!("[#2027 repro] deflation decoder norms = {norms:?}");
    assert!(hi > 0.0, "at least one atom must carry decoder norm");
    assert!(
        lo > 1.0e-3 * hi,
        "both atoms must carry material decoder norm after deflation: norms={norms:?}"
    );
}

/// #2027 WIDTH-SCALING + STRUCTURE-RECOVERY guard — the discriminating test.
///
/// Two facts from the sibling nursery evidence make raw EV an INSUFFICIENT guard:
///   1. The pathology is WIDTH-dependent — the REML control converges at `p = 16`
///      but hangs / co-collapses at `p ≈ 96`. A fix must be checked at BOTH widths:
///      the narrow arm must stay healthy, the wide arm is the regime being rescued.
///   2. Its fingerprint is a co-collapse that posts a DECENT reconstruction EV while
///      recovering NEITHER planted circle — both atoms pile into one shared subspace
///      and ride one circle plus noise (torch proxy: EV 0.63, adjacency 0.43/0.25).
///      Asserting EV alone therefore passes a co-collapsed fit.
///
/// The two circles are planted on DISJOINT ambient column PARITIES (circle A on the
/// even output channels, circle B on the odd), so an honest K=2 dictionary MUST
/// separate: one atom's decoder concentrates its Frobenius energy on the even
/// channels, the other on the odd. Co-collapse piles both atoms onto the same
/// channels — detected here as both atoms landing on the SAME side of the 0.5
/// even-energy split. We require, at both widths: finite loss (no thrash), a
/// materially positive EV, and the two atoms SEPARATED onto opposite-parity
/// subspaces (the structure the disjoint-deflation + ownership-anchor fix restores).
///
/// NOTE: this exercises the INNER joint solve (`run_joint_fit_arrow_schur` at a
/// fixed ρ) — the co-collapse / structure-recovery layer the seeding + anchoring fix
/// lives in — not the outer REML ρ-search whose non-PD-Hessian retries are the
/// separate Python-side "hang" at wide `p`.
#[test]
pub(crate) fn two_circle_separates_at_narrow_and_wide_widths_2027() {
    let m = 5usize;
    // (n, p): a NARROW arm that must stay healthy and the WIDE arm being rescued.
    for &(n, p) in &[(96usize, 16usize), (120usize, 96usize)] {
        let (mut term, target) = two_circle_k2_term(n, p, m);
        let mut rho = SaeManifoldRho::new(
            0.0,
            -6.0,
            vec![Array1::<f64>::zeros(1), Array1::<f64>::zeros(1)],
        );
        let loss = term
            .run_joint_fit_arrow_schur(target.view(), &mut rho, None, 60, 0.05, 1.0e-3, 1.0e-3)
            .unwrap();
        assert!(
            loss.total().is_finite(),
            "p={p}: joint fit must return a finite loss (no thrash / NaN)"
        );
        let ev = term
            .dictionary_reconstruction_ev(target.view(), &rho)
            .unwrap();

        // Per-atom decoder energy split across even vs odd output channels: circle A
        // lives on even channels, circle B on odd.
        let mut even_frac = [0.0_f64; 2];
        for (atom_idx, atom) in term.atoms.iter().enumerate() {
            let b = &atom.decoder_coefficients; // (m × p)
            let mut e_even = 0.0_f64;
            let mut e_odd = 0.0_f64;
            for col in 0..b.nrows() {
                for out in 0..p {
                    let v = b[[col, out]] * b[[col, out]];
                    if out % 2 == 0 {
                        e_even += v;
                    } else {
                        e_odd += v;
                    }
                }
            }
            even_frac[atom_idx] = e_even / (e_even + e_odd).max(1.0e-300);
        }
        eprintln!(
            "[#2027 repro] p={p}: EV={ev:.4}, per-atom even-energy fraction={even_frac:?}, \
             cocollapse_reseeds={}",
            term.dictionary_cocollapse_reseeds
        );
        assert!(
            ev > 0.20,
            "p={p}: dictionary co-collapsed to the null floor (EV={ev:.4} <= 0.20)"
        );
        // STRUCTURE RECOVERY: the atoms must land on OPPOSITE planted subspaces — one
        // even-dominant, one odd-dominant. Both on the same side of 0.5 is the
        // co-collapse signature (acceptable EV, neither circle recovered).
        let (lo, hi) = if even_frac[0] <= even_frac[1] {
            (even_frac[0], even_frac[1])
        } else {
            (even_frac[1], even_frac[0])
        };
        assert!(
            lo < 0.5 && hi > 0.5,
            "p={p}: atoms did NOT separate onto the two planted circles \
             (even-energy fractions {even_frac:?} both on one side of 0.5 = co-collapse: \
             EV looks fine but neither circle is recovered)"
        );
    }
}

/// #2082 — the STRUCTURAL coherence detector must fire on atoms that decode the
/// SAME output subspace (the "high EV, no structure" collapse the two-width test
/// catches) and stay SILENT on atoms whose decoders span ORTHOGONAL subspaces
/// (a healthy well-separated dictionary), keying on the derived random-subspace
/// null — so a live fit's guard reseeds a merged pair without touching separated
/// atoms.
#[test]
pub(crate) fn structural_coherence_detector_fires_on_duplicate_not_orthogonal_2082() {
    let n = 48usize;
    let p = 8usize;
    let m = 5usize;
    let (mut term, _target) = two_circle_k2_term(n, p, m);

    // ORTHOGONAL output subspaces: atom 0 decodes only EVEN output channels, atom 1
    // only ODD → their output frames are orthogonal → coherence 0 → NOT flagged.
    for atom in 0..2 {
        let mut b = Array2::<f64>::zeros((m, p));
        for col in 0..m {
            let out = (if atom == 0 { 0 } else { 1 }) + 2 * (col % (p / 2));
            if out < p {
                b[[col, out]] = 1.0;
            }
        }
        term.atoms[atom].decoder_coefficients = b;
    }
    assert!(
        term.structural_coherence_collapse_detected()
            .unwrap()
            .is_none(),
        "orthogonal-subspace atoms must NOT be flagged as structurally collapsed"
    );

    // DUPLICATE output subspaces: both atoms decode the SAME two output directions
    // → identical output frames → coherence ~1 → flagged (well above the ½(μ_null+1)
    // bar; for p=8, rank-2, μ_null ≈ 0.9 so the bar ≈ 0.95, and 1.0 clears it).
    let mut dup = Array2::<f64>::zeros((m, p));
    dup[[0, 0]] = 1.0;
    dup[[1, 1]] = 1.0;
    term.atoms[0].decoder_coefficients = dup.clone();
    term.atoms[1].decoder_coefficients = dup;
    let hit = term
        .structural_coherence_collapse_detected()
        .unwrap()
        .expect("duplicate-subspace atoms must be flagged as structurally collapsed");
    assert_eq!((hit.0, hit.1), (0, 1), "the offending pair is (0, 1)");
    assert!(
        hit.2 > 0.99,
        "duplicate output frames must report coherence ~1, got {}",
        hit.2
    );
}

/// #2100 — turning the `quotient_scale` SCALE-gauge lever ON must NOT detonate a
/// healthy fit. The pre-fix #2022 decoder peel folded EVERY atom's ‖B_k‖ into its
/// log-amplitude on EVERY β-Newton line-search TRIAL (inside `apply_newton_step_impl`),
/// forcing ‖B_k‖≡1 mid-solve and fighting the β-Newton magnitude step; the next step
/// took a runaway magnitude correction that compounded to a reconstruction EV of
/// ≈ −5.8e128 on the healthy K=2 two-circle dictionary (norms crushed to
/// [1.00, 0.187]). The fix removes that per-trial fold and moves the unit-Frobenius
/// retraction to the ACCEPTED-iterate boundary, gated to COLLAPSED atoms only
/// (`retract_collapsed_decoders_in_loop`) — so on a HEALTHY dictionary nothing
/// breaches and the ON path reproduces the OFF baseline (EV ≈ +0.43). Pre-fix this
/// test FAILS (EV explodes to ≈ −1e128); post-fix it PASSES.
#[test]
pub(crate) fn quotient_scale_on_does_not_detonate_healthy_k2_two_circle_2100() {
    let n = 96usize;
    let p = 16usize;
    let m = 5usize;

    // OFF baseline (default lever) — the healthy EV the ON path must match.
    let (mut off, target) = two_circle_k2_term(n, p, m);
    let mut rho_off = SaeManifoldRho::new(
        0.0,
        -6.0,
        vec![Array1::<f64>::zeros(1), Array1::<f64>::zeros(1)],
    );
    off.run_joint_fit_arrow_schur(target.view(), &mut rho_off, None, 60, 0.05, 1.0e-3, 1.0e-3)
        .unwrap();
    let ev_off = off
        .dictionary_reconstruction_ev(target.view(), &rho_off)
        .unwrap();
    assert!(
        ev_off.is_finite() && ev_off > 0.3,
        "OFF baseline must be a healthy positive EV (~+0.43); got {ev_off:.4}"
    );

    // ON — identical fixture, `quotient_scale` engaged.
    let (mut on, _t) = two_circle_k2_term(n, p, m);
    on.set_quotient_scale(true);
    let mut rho_on = SaeManifoldRho::new(
        0.0,
        -6.0,
        vec![Array1::<f64>::zeros(1), Array1::<f64>::zeros(1)],
    );
    let loss = on
        .run_joint_fit_arrow_schur(target.view(), &mut rho_on, None, 60, 0.05, 1.0e-3, 1.0e-3)
        .unwrap();
    assert!(
        loss.total().is_finite(),
        "quotient_scale ON: joint fit loss must stay finite (no runaway magnitude)"
    );
    let ev_on = on
        .dictionary_reconstruction_ev(target.view(), &rho_on)
        .unwrap();
    let norms: Vec<f64> = on
        .atoms
        .iter()
        .map(|a| {
            a.decoder_coefficients
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    eprintln!(
        "[#2100 repro] quotient_scale ON: EV={ev_on:.4} (OFF={ev_off:.4}), norms={norms:.4?}"
    );
    assert!(
        ev_on.is_finite() && ev_on > 0.3,
        "quotient_scale ON DETONATED a healthy K=2 fit: EV={ev_on:.4e} (expected finite > 0.3, \
         matching the OFF baseline {ev_off:.4}). The #2022 per-β-Newton decoder peel must NOT \
         run mid-solve — the unit-Frobenius retraction belongs at the accepted-iterate boundary, \
         gated to collapsed atoms."
    );
    // On a healthy dictionary nothing breaches the collapse ratio, so the ON path is
    // effectively inert: the recovered EV must track the OFF baseline closely (not a
    // coincidentally-positive but degraded fit).
    assert!(
        (ev_on - ev_off).abs() <= 0.1 * (1.0 + ev_off.abs()),
        "quotient_scale ON must track the OFF baseline on a healthy fit: ON={ev_on:.4} vs \
         OFF={ev_off:.4}"
    );
}

/// #2100 (K=1 arm) — turning `quotient_scale` ON must NOT crash a single-atom fit.
/// The issue's second failure mode: a healthy K=1 fit collapsed (EV 0.9975 → 0.0,
/// s → −43) because the per-β-Newton fold at `apply_newton_step_impl` folded the lone
/// decoder's ‖B‖ into `s` on every trial, so `s` ran away and `B` was crushed. With
/// that per-trial fold removed and the boundary retraction early-outing at K<2 (a lone
/// low-amplitude decoder is healthy, never retracted), the ON fit must stay finite and
/// healthy — tracking the OFF baseline closely. (It is NOT bit-for-bit identical: the
/// pre-existing #2022 refit-peel legitimately re-expresses the absolute decoder as a
/// unit frame + `s` at the seeding refit, an image-frozen coordinate change that nudges
/// the optimization trajectory by ~1e-5 without harming the fit — so this asserts
/// crash-freedom and OFF-tracking, not equality.) Pre-fix the ON fit collapses
/// (EV → 0); post-fix it is healthy.
#[test]
pub(crate) fn quotient_scale_on_does_not_crash_k1_2100() {
    let n = 96usize;
    let p = 16usize;
    let m = 5usize;

    let target = two_circle_whitened_target(n, p, 0.05);

    let mut off = two_circle_kn_term(n, p, m, 1);
    let mut rho_off = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1)]);
    off.run_joint_fit_arrow_schur(target.view(), &mut rho_off, None, 60, 0.05, 1.0e-3, 1.0e-3)
        .unwrap();
    let ev_off = off
        .dictionary_reconstruction_ev(target.view(), &rho_off)
        .unwrap();
    assert!(
        ev_off.is_finite() && ev_off > 0.2,
        "K=1 OFF baseline must be a healthy positive EV; got {ev_off:.4}"
    );

    let mut on = two_circle_kn_term(n, p, m, 1);
    on.set_quotient_scale(true);
    let mut rho_on = SaeManifoldRho::new(0.0, -6.0, vec![Array1::<f64>::zeros(1)]);
    let loss = on
        .run_joint_fit_arrow_schur(target.view(), &mut rho_on, None, 60, 0.05, 1.0e-3, 1.0e-3)
        .unwrap();
    assert!(loss.total().is_finite(), "K=1 ON: loss must stay finite");
    let ev_on = on
        .dictionary_reconstruction_ev(target.view(), &rho_on)
        .unwrap();
    let s = on.atoms[0].log_amplitude;

    eprintln!("[#2100 repro] K=1 quotient_scale ON EV={ev_on:.6} (OFF={ev_off:.6}), s={s:.4}");
    assert!(
        ev_on.is_finite() && ev_on > 0.2,
        "quotient_scale ON CRASHED a healthy K=1 fit: EV={ev_on:.4e} (expected finite > 0.2, \
         the issue's collapse was EV → 0.0 with s → −43); s={s:.4}"
    );
    // The ON fit must track the OFF baseline — a healthy K=1 decoder is never
    // retracted, so the only quotient effect is the image-frozen refit-peel.
    assert!(
        (ev_on - ev_off).abs() <= 1.0e-3 * (1.0 + ev_off.abs()),
        "quotient_scale ON must track the OFF baseline at K=1: ON={ev_on:.6} vs OFF={ev_off:.6}"
    );
}

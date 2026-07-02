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

use super::*;
use super::tests::deterministic_circle_noise;

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
        norms[atom_idx] = atom.decoder_coefficients.iter().map(|v| v * v).sum::<f64>().sqrt();
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

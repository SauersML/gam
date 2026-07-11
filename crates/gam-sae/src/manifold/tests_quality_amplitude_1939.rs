//! #1939 OBJECTIVE-QUALITY acceptance bar — existence/intensity DECOUPLING in the
//! physical dictionary. The current representation carries intensity directly in
//! each fitted decoder. *Existence* (does this atom explain held-out structure)
//! must be identified separately from *intensity* (how large its contribution is).
//!
//! We plant that ground truth: two live circles on DISJOINT output subspaces whose
//! amplitudes differ by ~an order of magnitude, plus a DEAD atom slot with no
//! planted signal, and fit a K=3 dictionary. The objective is truth recovery —
//! the planted amplitude ratio and the dead/alive partition — NOT reproduction of
//! any reference tool's fitted parameters.
//!
//! Intensity lives in the decoder magnitude `‖B_k‖`; the executable objective bar
//! below therefore identifies a dead atom by its held-out contribution while
//! separately checking the recovered intensity ratio of the two live atoms.

use super::tests::deterministic_circle_noise;
use super::*;

/// Two circles on disjoint output-column parities with UNEQUAL amplitudes, plus a
/// third (dead) subspace that carries no planted signal — returned UN-whitened so
/// the planted amplitudes survive into the target (a column-standardized target
/// would quotient exactly the intensity this test measures).
///
/// * circle A: even output columns, amplitude `amp_a` (the strong atom),
/// * circle B: odd output columns, amplitude `amp_b` (the weak-but-real atom),
/// * the remaining variance is small isotropic noise (no third circle).
fn two_unequal_circles_plus_dead(
    n: usize,
    p: usize,
    amp_a: f64,
    amp_b: f64,
    sigma: f64,
) -> Array2<f64> {
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
    // Orthonormalize each planted 2-frame so the circle's ambient radius is 1 and
    // the decoded amplitude is exactly `amp_*` (not tangled with frame scale).
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
        // Use frequency 3 for the weak circle. Each fitted circle basis reaches
        // only harmonic 2, so the strong frequency-1 chart cannot also absorb the
        // weak signal as its second harmonic. A frequency-2 planting would make
        // decoder ownership non-identifiable before existence/intensity is tested.
        let tb = std::f64::consts::TAU * (3.0 * row as f64 + 0.37) / (n as f64);
        let (ca, sa) = (ta.cos(), ta.sin());
        let (cb, sb) = (tb.cos(), tb.sin());
        for j in 0..p {
            z[[row, j]] = amp_a * (ca * fa[[0, j]] + sa * fa[[1, j]])
                + amp_b * (cb * fb[[0, j]] + sb * fb[[1, j]])
                + sigma * deterministic_circle_noise(row, j + 7);
        }
    }
    z
}

/// Build a fresh K periodic term with the production PCA coordinates and joint
/// decoder-LSQ seed at the given atom count on the UN-whitened target.
fn kterm_periodic(target: &Array2<f64>, k: usize, m: usize) -> SaeManifoldTerm {
    let n = target.nrows();
    let p = target.ncols();
    let d = 1usize;
    let basis_kinds = vec![SaeAtomBasisKind::Periodic; k];
    let dims = vec![d; k];
    let seed = sae_pca_seed_initial_coords(target.view(), &basis_kinds, &dims).unwrap();
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap());

    let mut basis_values = Array3::<f64>::zeros((k, n, m));
    let mut basis_jacobian = Array4::<f64>::zeros((k, n, m, d));
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
    let basis_sizes = vec![m; k];
    let decoder = sae_decoder_lsq_init(
        basis_values.view(),
        &basis_sizes,
        target.view(),
        logits.view(),
        "ordered_beta_bernoulli",
        1.0,
        1.0,
        0.0,
        None,
    )
    .unwrap();
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
        &basis_sizes,
        &dims,
        decoder.view(),
        penalties.view(),
        logits.view(),
        &coords_vec,
        AssignmentMode::ordered_beta_bernoulli(1.0, 1.0, false),
        &evaluators,
    )
    .unwrap()
}

/// Per-atom RMS of the GATED decoded contribution `a_ik · (Φ_k B_k)_i` over all
/// rows and output columns — the atom's physical intensity as it enters the
/// reconstruction (existence × intensity combined into an output-energy readout).
fn per_atom_contribution_rms(term: &SaeManifoldTerm) -> Vec<f64> {
    let n = term.n_obs();
    let p = term.output_dim();
    let k = term.k_atoms();
    let mut sumsq = vec![0.0_f64; k];
    let mut buf = vec![0.0_f64; p];
    for row in 0..n {
        let weights = term.assignment.try_assignments_row(row).unwrap();
        for atom in 0..k {
            let a_k = weights[atom];
            term.atoms[atom].fill_decoded_row(row, &mut buf);
            for &g in buf.iter() {
                let v = a_k * g;
                sumsq[atom] += v * v;
            }
        }
    }
    sumsq
        .into_iter()
        .map(|s| (s / (n as f64 * p as f64)).sqrt())
        .collect()
}

/// Even-column energy fraction of an atom's decoder — the subspace fingerprint
/// used to match a fitted atom to circle A (even, ~1.0) vs circle B (odd, ~0.0).
fn even_energy_fraction(atom: &SaeManifoldAtom, p: usize) -> f64 {
    let b = &atom.decoder_coefficients;
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
    e_even / (e_even + e_odd).max(1.0e-300)
}

/// OBJECTIVE BAR (reachable) — a K=3 fit of two unequal circles + one dead slot
/// recovers the planted structure: the reconstruction is faithful, the two live
/// atoms land on the two planted (disjoint-parity) subspaces with intensities in
/// the planted ~8:1 ratio, and the dead atom is separately identified as
/// EXISTENCE-negative — it explains ~no held-out variance — even though the WEAK
/// live atom is also small in magnitude. That last clause is the #1939 payoff:
/// existence and intensity are decoupled, so "small" (weak circle) is not confused
/// with "absent" (dead slot).
#[test]
fn existence_and_intensity_are_separately_identified_1939() {
    let n = 144usize;
    let p = 16usize;
    let m = 5usize; // [1, sin2πt, cos2πt, sin4πt, cos4πt]
    let amp_a = 8.0_f64;
    let amp_b = 1.0_f64;
    let target = two_unequal_circles_plus_dead(n, p, amp_a, amp_b, 0.02);
    let mut term = kterm_periodic(&target, 3, m);

    let mut rho = SaeManifoldRho::new(
        0.0,
        -6.0,
        vec![
            Array1::<f64>::zeros(1),
            Array1::<f64>::zeros(1),
            Array1::<f64>::zeros(1),
        ],
    );
    let loss = term
        .run_joint_fit_arrow_schur(target.view(), &mut rho, None, 80, 0.05, 1.0e-3, 1.0e-3)
        .unwrap();
    assert!(loss.total().is_finite(), "loss must stay finite");

    let ev = term
        .dictionary_reconstruction_ev(target.view(), &rho)
        .unwrap();
    let loao = term
        .per_atom_loao_explained_variance(target.view(), &rho)
        .unwrap();
    let contrib = per_atom_contribution_rms(&term);
    let even_frac: Vec<f64> = term
        .atoms
        .iter()
        .map(|a| even_energy_fraction(a, p))
        .collect();
    eprintln!(
        "[#1939] EV={ev:.4}, contrib_rms={contrib:?}, loao_ev={loao:?}, even_frac={even_frac:?}"
    );

    // Faithful reconstruction: two rank-2 circles dominate a 16-dim cloud, so an
    // honest K=3 dictionary explains most of the variance.
    assert!(
        ev > 0.80,
        "the two-circle target must be well reconstructed (EV={ev:.4})"
    );

    // Rank atoms by physical contribution. The strongest and second-strongest are
    // the live atoms; the weakest is the dead-slot candidate.
    let mut order: Vec<usize> = (0..3).collect();
    order.sort_by(|&i, &j| contrib[j].partial_cmp(&contrib[i]).unwrap());
    let (strong, weak, dead) = (order[0], order[1], order[2]);

    // (b) INTENSITY RECOVERY — the two live atoms carry the planted amplitude
    // ratio. The gate weight is common (uniform-ish ibp gate), so the ratio of
    // decoded contribution RMS tracks amp_a/amp_b; allow a generous factor since
    // fit noise and the shared gate perturb the absolute scale but not the order.
    let planted_ratio = amp_a / amp_b;
    let recovered_ratio = contrib[strong] / contrib[weak].max(1.0e-300);
    eprintln!("[#1939] planted amp ratio={planted_ratio:.2}, recovered={recovered_ratio:.2}");
    assert!(
        recovered_ratio > 2.0 && recovered_ratio < 4.0 * planted_ratio,
        "the live intensity ratio {recovered_ratio:.2} must recover the planted order of \
         magnitude {planted_ratio:.2} (strong ≫ weak, not collapsed to parity)"
    );

    // The two live atoms occupy OPPOSITE planted subspaces (one even-dominant, one
    // odd-dominant) — they recovered distinct circles, not one shared basin.
    let (lo, hi) = if even_frac[strong] <= even_frac[weak] {
        (even_frac[strong], even_frac[weak])
    } else {
        (even_frac[weak], even_frac[strong])
    };
    assert!(
        lo < 0.5 && hi > 0.5,
        "the two live atoms must separate onto the two planted (disjoint-parity) circles; \
         even-fractions strong={:.3} weak={:.3}",
        even_frac[strong],
        even_frac[weak]
    );

    // (c) EXISTENCE identified SEPARATELY from intensity — the crux. Both the weak
    // live atom and the dead atom are small in magnitude, but the weak atom EXISTS
    // (it explains real held-out variance) while the dead atom does not. So the
    // dead atom's leave-one-atom-out EV drop must be near zero AND far below the
    // weak live atom's — proving "weak" was not mistaken for "absent".
    let loao_weak = loao[weak].unwrap_or(0.0);
    let loao_dead = loao[dead].unwrap_or(0.0);
    eprintln!("[#1939] loao(weak live)={loao_weak:.4}, loao(dead)={loao_dead:.4}");
    assert!(
        loao_weak > 0.01,
        "the WEAK live atom must explain real held-out variance (existence-positive); \
         loao_weak={loao_weak:.4}"
    );
    assert!(
        loao_dead < 0.25 * loao_weak,
        "the DEAD atom must be existence-negative — its held-out contribution \
         ({loao_dead:.4}) must sit far below the weak live atom's ({loao_weak:.4}), so a \
         small-but-real intensity is never confused with absence"
    );
    // And its physical contribution must be the smallest by a clear margin.
    assert!(
        contrib[dead] < 0.5 * contrib[weak],
        "the dead atom's decoded contribution {:.4e} must be materially below the weak \
         live atom's {:.4e}",
        contrib[dead],
        contrib[weak]
    );
}

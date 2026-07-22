//! Decoder-collapse detection & prevention tests, split out of `tests.rs` to
//! keep that file under the #780 line-count gate. These exercise the whole
//! collapse-prevention stack: the decoder-norm guard and its residual reseed
//! (single-atom and total K-way co-collapse), the keep-best multi-start, the
//! decoder-repulsion collinearity gate, the #1522/#1610/#1625 separation
//! interior-point barrier (value/gradient/scale-invariance/evidence-derived
//! strength), and the #1026 hybrid-collapse dominance / top-k / OOS
//! reconstruction paths. They share the parent module's fixtures via
//! `super::tests`.

use super::tests::{
    TestPeriodicEvaluator, periodic_basis, small_two_atom_periodic_term, trivial_k1_euclidean_term,
};
use super::*;
use ndarray::array;

/// #976 decoder arm (prevention): a K>1 fit whose second atom's decoder has
/// collapsed to ≈0 — gates still spread, so the gate-mass guard is satisfied
/// — is caught by [`SaeManifoldTerm::enforce_decoder_norm_guard`], which
/// reseeds the collapsed atom onto the reconstruction residual and re-fits
/// the decoders so the atom recovers a NON-degenerate, DISTINCT decoder.
/// This is the disease the real-data K=2/K=3 OLMo fits hit (every decoder →
/// 0 ⇒ EV=0 ⇒ every per-row H_tt gauge-flat ⇒ the 0→K·n deflation abort).
#[test]
pub(crate) fn decoder_norm_guard_reseeds_collapsed_atom_to_distinct_nonzero() {
    let (term0, target, rho) = small_two_atom_periodic_term();
    let mut term = term0.clone();
    // Collapse atom 1's decoder to ≈0 while leaving its assignment gates
    // spread (the mass guard sees nothing wrong). Atom 0 keeps its signal.
    term.atoms[1].decoder_coefficients.fill(0.0);

    let norm = |a: &SaeManifoldAtom| -> f64 {
        a.decoder_coefficients
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
    };
    assert!(norm(&term.atoms[1]) < 1e-12, "atom 1 starts collapsed");

    term.enforce_decoder_norm_guard(target.view(), 0, &rho, None)
        .expect("decoder-norm guard must not error on a recoverable collapse");

    // The guard recorded a Reseeded collapse event for the collapsed atom.
    let reseeded = term
        .collapse_events()
        .iter()
        .any(|e| e.atom == 1 && e.action == CollapseAction::Reseeded);
    assert!(
        reseeded,
        "collapsed atom 1 must be recorded as Reseeded; events: {:?}",
        term.collapse_events()
    );

    // After the reseed + joint LSQ refit, atom 1 carries a non-degenerate
    // decoder again (well above the collapse floor relative to atom 0).
    let n1 = norm(&term.atoms[1]);
    let n0 = norm(&term.atoms[0]);
    assert!(
        n0 > 0.0 && n1 > SAE_ATOM_DECODER_NORM_COLLAPSE_RATIO * n0,
        "reseeded atom 1 decoder must be non-degenerate: ‖B0‖={n0:.3e} ‖B1‖={n1:.3e}"
    );

    // The reseeded atom's coordinates are diversified (not a single
    // collapsed constant), so its design column is non-degenerate.
    let c1 = term.assignment.coords[1].as_matrix();
    let (lo, hi) = c1
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    assert!(
        hi - lo > 1e-6,
        "reseeded atom 1 coordinates must span a non-trivial range; got [{lo}, {hi}]"
    );

    // The reseeded decoder is DISTINCT from atom 0's (not a duplicate): the
    // residual-seeded coordinates point atom 1 at unexplained signal, so the
    // two decoder column-spaces are not collinear.
    let b0 = &term.atoms[0].decoder_coefficients;
    let b1 = &term.atoms[1].decoder_coefficients;
    let dot: f64 = b0.iter().zip(b1.iter()).map(|(x, y)| x * y).sum();
    let cos = dot.abs() / (n0 * n1);
    assert!(
        cos < 0.999,
        "reseeded atom 1 decoder must be distinct from atom 0 (|cos|={cos:.4})"
    );
}

/// decoder.
#[test]
pub(crate) fn decoder_norm_guard_reseeds_all_atoms_on_total_co_collapse_k3() {
    // Three periodic (circle) atoms, p=3 output so three distinct residual PCs
    // exist for the disjoint-PC reseed to land each atom on its own direction.
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45], [0.10]];
    let coords2 = array![[0.25], [0.40], [0.75], [0.05], [0.60], [0.85]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let (phi2, jet2) = periodic_basis(&coords2);
    // Decoders are tiny-but-NONZERO and of comparable magnitude across atoms:
    // the dictionary co-collapsed (EV ≈ 0) yet has a usable median scale, so it
    // reaches the absolute-EV co-collapse arm (an exactly-zero dictionary would
    // hit the `median == 0` early return — the cold-seed case, handled by the
    // mass guard/inner solve, not here) and no atom is *relatively* behind its
    // peers (all norms within ~1.5×, none below `1e-3·median`).
    let make_atom = |name: &str, phi: Array2<f64>, jet: Array3<f64>, scale: f64| {
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            Array2::<f64>::from_elem((3, 3), scale),
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
    };
    let atom0 = make_atom("periodic0", phi0, jet0, 1.0e-5);
    let atom1 = make_atom("periodic1", phi1, jet1, 1.2e-5);
    let atom2 = make_atom("periodic2", phi2, jet2, 0.8e-5);
    // Gates stay spread across rows/atoms — the gate-mass guard is satisfied,
    // so only the absolute-EV co-collapse arm can catch this failure.
    let logits = array![
        [0.7, -0.2, 0.3],
        [0.1, 0.4, -0.1],
        [-0.3, 0.5, 0.2],
        [0.6, -0.1, 0.4],
        [0.2, 0.3, -0.2],
        [0.4, 0.1, 0.5]
    ];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0, coords1, coords2],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(0.8),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom0, atom1, atom2], assignment).unwrap();
    // A target with genuine 3-direction structure so the residual (≈ target,
    // since the dictionary explains ≈0) carries three distinct PCs.
    let target = array![
        [0.40, -0.10, 0.05],
        [-0.20, 0.35, -0.15],
        [0.10, 0.05, 0.30],
        [0.25, -0.30, -0.05],
        [-0.15, 0.20, 0.18],
        [0.30, 0.12, -0.22]
    ];
    let rho = SaeManifoldRho::new(
        (-0.3_f64).exp().ln(),
        0.7_f64.ln(),
        vec![
            array![0.9_f64.ln()],
            array![1.0_f64.ln()],
            array![1.1_f64.ln()],
        ],
    );

    // Confirm the precondition: the dictionary is co-collapsed (EV below the
    // floor) with NO atom relatively behind its peers (all norms ≈0).
    let ev_before = term
        .dictionary_reconstruction_ev(target.view(), &rho)
        .expect("EV evaluates");
    assert!(
        ev_before < 0.28_f64,
        "test precondition: dictionary must start co-collapsed; EV={ev_before:.4}"
    );

    // S1: the EV co-collapse arm is armed only at iteration > 0 (iteration 0 = cold seed).
    term.enforce_decoder_norm_guard(target.view(), 1, &rho, None)
        .expect("co-collapse guard must recover, not error");

    // EVERY atom — including the one the old code preserved as anchor — must be
    // recorded as Reseeded. This is the regression the fix targets.
    for atom in 0..3 {
        let reseeded = term
            .collapse_events()
            .iter()
            .any(|e| e.atom == atom && e.action == CollapseAction::Reseeded);
        assert!(
            reseeded,
            "total co-collapse must reseed ALL atoms; atom {atom} was not reseeded. events: {:?}",
            term.collapse_events()
        );
    }

    // After the reseed + joint LSQ refit every atom carries a non-degenerate
    // decoder again, and the three decoders are pairwise distinct (each landed
    // on its own residual PC, so no two column-spaces are collinear).
    let norm = |a: &SaeManifoldAtom| -> f64 {
        a.decoder_coefficients
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
    };
    let norms: Vec<f64> = (0..3).map(|a| norm(&term.atoms[a])).collect();
    for (atom, &nrm) in norms.iter().enumerate() {
        assert!(
            nrm > 1e-9,
            "reseeded atom {atom} decoder must be non-degenerate; ‖B‖={nrm:.3e}"
        );
    }
    for a in 0..3 {
        for b in (a + 1)..3 {
            let ba = &term.atoms[a].decoder_coefficients;
            let bb = &term.atoms[b].decoder_coefficients;
            let dot: f64 = ba.iter().zip(bb.iter()).map(|(x, y)| x * y).sum();
            let cos = dot.abs() / (norms[a] * norms[b]);
            assert!(
                cos < 0.999,
                "reseeded atoms {a},{b} decoders must be distinct (|cos|={cos:.4})"
            );
        }
    }

    // The dictionary is no longer co-collapsed: the reseed + LSQ refit explains
    // strictly more variance than the degenerate start.
    let ev_after = term
        .dictionary_reconstruction_ev(target.view(), &rho)
        .expect("EV evaluates post-reseed");
    assert!(
        ev_after > ev_before,
        "co-collapse reseed must improve EV; before={ev_before:.4} after={ev_after:.4}"
    );
}

/// #1026 keep-best multi-start: the full-dictionary co-collapse reseed is a
/// bounded multi-start over distinct residual subspaces, but successive reseeds
/// can land in STRICTLY WORSE basins (real OLMo K=4: the seed explains EV 0.127
/// while later reseeds fall to −1.0). A multi-start must return the BEST basin it
/// visited, never the last. The guard retains the highest-EV state seen across
/// the reseeds and restores it once the reseed budget is spent, so the final
/// dictionary EV is no worse than the best intermediate attempt.
#[test]
pub(crate) fn co_collapse_multistart_restores_best_basin_not_last_reseed() {
    // Same co-collapsed K=3 periodic dictionary as
    // `decoder_norm_guard_reseeds_all_atoms_on_total_co_collapse_k3`, driven
    // through the WHOLE reseed budget so the budget-exhaustion restore fires.
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45], [0.10]];
    let coords2 = array![[0.25], [0.40], [0.75], [0.05], [0.60], [0.85]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let (phi2, jet2) = periodic_basis(&coords2);
    let make_atom = |name: &str, phi: Array2<f64>, jet: Array3<f64>, scale: f64| {
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            Array2::<f64>::from_elem((3, 3), scale),
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
    };
    let atom0 = make_atom("periodic0", phi0, jet0, 1.0e-5);
    let atom1 = make_atom("periodic1", phi1, jet1, 1.2e-5);
    let atom2 = make_atom("periodic2", phi2, jet2, 0.8e-5);
    let logits = array![
        [0.7, -0.2, 0.3],
        [0.1, 0.4, -0.1],
        [-0.3, 0.5, 0.2],
        [0.6, -0.1, 0.4],
        [0.2, 0.3, -0.2],
        [0.4, 0.1, 0.5]
    ];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0, coords1, coords2],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(0.8),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom0, atom1, atom2], assignment).unwrap();
    let target = array![
        [0.40, -0.10, 0.05],
        [-0.20, 0.35, -0.15],
        [0.10, 0.05, 0.30],
        [0.25, -0.30, -0.05],
        [-0.15, 0.20, 0.18],
        [0.30, 0.12, -0.22]
    ];
    let rho = SaeManifoldRho::new(
        (-0.3_f64).exp().ln(),
        0.7_f64.ln(),
        vec![
            array![0.9_f64.ln()],
            array![1.0_f64.ln()],
            array![1.1_f64.ln()],
        ],
    );

    // Drive the guard once per "outer iteration" through the whole multi-start
    // budget plus the budget-exhaustion call, recording the dictionary EV the
    // guard observes at the start of each call (the candidate basin it may bank).
    // The guard reseeds in place, so each call's pre-reseed EV is a distinct
    // multi-start attempt; the best of these is what the final state must match.
    // S1: the EV arm is armed only at iteration > 0; drive 1..=BUDGET+1 (BUDGET reseeds + restore).
    let mut best_seen = f64::NEG_INFINITY;
    for iteration in 1..=(SAE_DICTIONARY_COCOLLAPSE_RESEED_BUDGET + 1) {
        let ev_at_entry = term
            .dictionary_reconstruction_ev(target.view(), &rho)
            .expect("EV evaluates");
        if ev_at_entry < 0.28_f64 {
            best_seen = best_seen.max(ev_at_entry);
        }
        term.enforce_decoder_norm_guard(target.view(), iteration, &rho, None)
            .expect("co-collapse guard must recover, not error");
    }

    // After the budget is spent the guard has restored the best basin it banked,
    // so the final dictionary EV is at least the best attempt seen — never the
    // (possibly catastrophic) last reseed.
    let ev_final = term
        .dictionary_reconstruction_ev(target.view(), &rho)
        .expect("EV evaluates");
    assert!(
        best_seen.is_finite(),
        "test precondition: at least one co-collapsed attempt must be observed"
    );
    assert!(
        ev_final >= best_seen - 1e-9,
        "multi-start must return its BEST basin, not the last reseed: \
         final EV={ev_final:.6} < best seen={best_seen:.6}"
    );
}

/// #1026 decoder-repulsion gate safety: the collinearity gate must be a STRICT
/// no-op for well-separated atoms (orthogonal decoders → gate `None`, so no
/// value/gradient/curvature is added and healthy fits are byte-identical) and
/// must ENGAGE for near-collinear atoms (the co-collapse geometry it conditions).
/// Built on a K=2 periodic fixture whose decoders we set directly.
#[test]
pub(crate) fn decoder_repulsion_gate_off_when_separated_on_when_collinear() {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45], [0.10]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    // Periodic basis is M=3 wide; output p=3. Build two atoms; decoders set below.
    let make_atom = |name: &str, phi: Array2<f64>, jet: Array3<f64>, decoder: Array2<f64>| {
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
    };
    let logits = array![
        [0.7, -0.2],
        [0.1, 0.4],
        [-0.3, 0.5],
        [0.6, -0.1],
        [0.2, 0.3],
        [0.4, 0.1]
    ];
    let build = |dec0: Array2<f64>, dec1: Array2<f64>| {
        let atom0 = make_atom("periodic0", phi0.clone(), jet0.clone(), dec0);
        let atom1 = make_atom("periodic1", phi1.clone(), jet1.clone(), dec1);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits.clone(),
            vec![coords0.clone(), coords1.clone()],
            vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ],
            AssignmentMode::softmax(0.8),
        )
        .unwrap();
        SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap()
    };

    // ORTHOGONAL decoders: atom0 writes output channel 0, atom1 writes channel 1.
    // Their cross-Gram B_0 B_1ᵀ = 0 ⇒ s_01 = 0 ⇒ gate exactly 0 ⇒ field `None`.
    let mut dec0 = Array2::<f64>::zeros((3, 3));
    dec0[[0, 0]] = 1.0;
    let mut dec1 = Array2::<f64>::zeros((3, 3));
    dec1[[0, 1]] = 1.0;
    let mut sep = build(dec0, dec1);
    sep.refresh_decoder_repulsion_gate();
    assert!(
        sep.decoder_repulsion_gate.is_none(),
        "orthogonal decoders must leave the repulsion gate OFF (strict no-op): {:?}",
        sep.decoder_repulsion_gate
    );
    assert_eq!(
        sep.decoder_repulsion_value(1.0),
        0.0,
        "orthogonal decoders must contribute zero repulsion value"
    );

    // COLLINEAR decoders: both atoms write the SAME output channel 0 with the
    // same basis-row pattern ⇒ s_01 = 1 ⇒ gate fully engaged ⇒ field `Some`.
    let mut dec0c = Array2::<f64>::zeros((3, 3));
    dec0c[[0, 0]] = 1.0;
    let mut dec1c = Array2::<f64>::zeros((3, 3));
    dec1c[[0, 0]] = 1.0;
    let mut col = build(dec0c, dec1c);
    col.refresh_decoder_repulsion_gate();
    let gate = col
        .decoder_repulsion_gate
        .as_ref()
        .expect("collinear decoders must ENGAGE the repulsion gate");
    assert!(
        gate.iter().any(|&(j, k, w)| j == 0 && k == 1 && w > 0.0),
        "engaged gate must carry a positive weight on pair (0,1): {gate:?}"
    );
    assert!(
        col.decoder_repulsion_value(1.0) > 0.0,
        "collinear decoders must contribute positive repulsion value"
    );
}

/// #1522 — the SEPARATION interior-point barrier is the deterministic collapse
/// PREVENTION (not a detect-then-reseed bandaid). On a constructed collapse-prone
/// fixture — two co-firing K=2 atoms whose decoders point nearly the same way
/// (normalized alignment `c² ≈ 0.8`, the geometry that drives the per-row `H_tt`
/// near-singular and the whole dictionary into the co-collapse basin) — this
/// pins that the barrier:
///   1. WITH it (`scale = 1`): adds a positive penalty AND a genuine SEPARATING
///      force — one gradient-descent step along `-∂P_sep/∂B` strictly REDUCES the
///      alignment `c²`, i.e. the atoms move apart (collapse is prevented in the
///      optimizer, not patched after the fact).
///   2. WITHOUT it (`scale = 0` ⇒ `μ = 0`, the LOCAL "no prevention" arm — no
///      process-global override toggled, so it is parallelism-safe): value `0`
///      and an all-zero gradient. The aligned atoms feel NO restoring force and
///      would stay collapsed — this is the "collapses without the prevention"
///      half of the pin.
///   3. INTERIOR-POINT divergence: a MORE-aligned configuration carries a strictly
///      LARGER barrier value than a less-aligned one, so the force grows without
///      bound toward the collapse boundary (`c² → 1`).
///   4. NON-REGRESSION: ORTHOGONAL (healthy, well-separated) decoders get value
///      `0` and an all-zero gradient even with the barrier ON, so the prevention
///      is a strict no-op away from collapse and healthy fits stay byte-identical
///      (the reseed backstop can remain as defense-in-depth and rarely fires).
#[test]
pub(crate) fn separation_barrier_is_collapse_prevention_not_bandaid_1522() {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45], [0.10]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    // softmax routing ⇒ every atom carries strictly positive mass on every row,
    // so the pair co-fires (`q_01 > 0`) and the separation barrier engages.
    let logits = array![
        [0.7, -0.2],
        [0.1, 0.4],
        [-0.3, 0.5],
        [0.6, -0.1],
        [0.2, 0.3],
        [0.4, 0.1]
    ];
    let build = |dec0: Array2<f64>, dec1: Array2<f64>| {
        let make = |name: &str, phi: Array2<f64>, jet: Array3<f64>, decoder: Array2<f64>| {
            SaeManifoldAtom::new_with_provided_function_gram(
                name,
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(3),
            )
            .unwrap()
            .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
        };
        let atom0 = make("periodic0", phi0.clone(), jet0.clone(), dec0);
        let atom1 = make("periodic1", phi1.clone(), jet1.clone(), dec1);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits.clone(),
            vec![coords0.clone(), coords1.clone()],
            vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ],
            AssignmentMode::softmax(0.8),
        )
        .unwrap();
        SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap()
    };
    // Single-row decoders so the normalized alignment `c²` is exactly the squared
    // cosine of the two output-direction vectors. Channel choices give `c² = 0.8`
    // (cosθ = √0.8): high enough to drive collapse, low enough that the barrier
    // gradient (`α ∝ 1/(1-c²+ε)`) is finite and a small step stays in the basin.
    let row_decoder = |r: [f64; 3]| {
        let mut d = Array2::<f64>::zeros((3, 3));
        d[[0, 0]] = r[0];
        d[[0, 1]] = r[1];
        d[[0, 2]] = r[2];
        d
    };
    // Normalized-alignment c² between two single-row decoders, read straight off
    // the atom decoder coefficients (the same quantity the barrier penalizes).
    let alignment_c2 = |b0: &Array2<f64>, b1: &Array2<f64>| -> f64 {
        let (m0, p) = (b0.nrows(), b0.ncols());
        let m1 = b1.nrows();
        let mut cross = 0.0_f64;
        for a in 0..m0 {
            for b in 0..m1 {
                let mut c = 0.0_f64;
                for o in 0..p {
                    c += b0[[a, o]] * b1[[b, o]];
                }
                cross += c * c;
            }
        }
        let n0: f64 = b0.iter().map(|v| v * v).sum();
        let n1: f64 = b1.iter().map(|v| v * v).sum();
        cross / (n0 * n1)
    };

    let dec0 = row_decoder([1.0, 0.0, 0.0]);
    // cosθ = √0.8 ≈ 0.894427, sinθ = √0.2 ≈ 0.447214 ⇒ unit-norm, c² = 0.8.
    let dec1 = row_decoder([0.894_427_191, 0.447_213_595, 0.0]);
    let c2_before = alignment_c2(&dec0, &dec1);
    assert!(
        (c2_before - 0.8).abs() < 1e-6,
        "fixture precondition: aligned decoders must start at c² ≈ 0.8, got {c2_before}"
    );

    let term = build(dec0.clone(), dec1.clone());

    // ── Arm 2 (do this first): barrier OFF (scale 0 ⇒ μ = 0) is a no-op. ──
    let (value_off, grad_off) = term.separation_barrier_value_and_grad_for_test(0.0);
    assert_eq!(
        value_off, 0.0,
        "barrier OFF must contribute zero value (the no-prevention arm)"
    );
    assert!(
        grad_off.iter().all(|&g| g == 0.0),
        "barrier OFF must leave the gradient identically zero — aligned atoms feel \
         NO separating force, so without prevention they stay collapsed"
    );

    // ── Arm 1: barrier ON supplies a positive penalty and a separating force. ──
    let (value_on, grad_on) = term.separation_barrier_value_and_grad_for_test(1.0);
    assert!(
        value_on > 0.0,
        "barrier ON must penalize the aligned, co-firing pair (value {value_on} ≤ 0)"
    );
    assert!(
        grad_on.iter().any(|&g| g != 0.0),
        "barrier ON must produce a non-zero separating gradient on the aligned pair"
    );

    // One gradient-descent step `B ← B - η·∂P/∂B` must REDUCE the alignment c².
    // η is small relative to the decoder scale so the step stays inside the basin.
    let eta = 1.0e-3;
    let offsets = term.beta_offsets();
    let p = term.output_dim();
    let stepped = |atom: usize, base: &Array2<f64>| -> Array2<f64> {
        let mut out = base.clone();
        let off = offsets[atom];
        for a in 0..out.nrows() {
            for o in 0..p {
                out[[a, o]] -= eta * grad_on[off + a * p + o];
            }
        }
        out
    };
    let dec0_stepped = stepped(0, &dec0);
    let dec1_stepped = stepped(1, &dec1);
    let c2_after = alignment_c2(&dec0_stepped, &dec1_stepped);
    assert!(
        c2_after < c2_before - 1e-9,
        "a descent step along the barrier gradient must SEPARATE the atoms \
         (c² must fall): before={c2_before:.6} after={c2_after:.6}"
    );

    // ── Arm 3: interior-point divergence — more alignment ⇒ strictly larger value. ──
    // Less aligned: r_k = (0.6, 0.8, 0) ⇒ c² = 0.36. More aligned: c² ≈ 0.98.
    let term_less = build(dec0.clone(), row_decoder([0.6, 0.8, 0.0]));
    let term_more = build(dec0.clone(), row_decoder([0.989_949_49, 0.141_421_36, 0.0]));
    let value_less = term_less.separation_barrier_value(1.0);
    let value_more = term_more.separation_barrier_value(1.0);
    assert!(
        value_more > value_on && value_on > value_less,
        "barrier value must grow with alignment toward the collapse boundary: \
         less(c²=.36)={value_less:.6} < base(c²=.8)={value_on:.6} < more(c²=.98)={value_more:.6}"
    );

    // ── Arm 4: non-regression — orthogonal (healthy) decoders are a strict no-op
    // in the FORCE. The separating gradient (and hence the optimizer trajectory)
    // is identically zero, so a well-separated fit is steered exactly as if no
    // barrier were present; the scalar value carries only the negligible constant
    // `-μ·q·log(1+ε) ≈ -1e-5` eps-softening offset (a constant in the objective,
    // which cannot move the optimum or fire the reseed). ──
    let term_ortho = build(row_decoder([1.0, 0.0, 0.0]), row_decoder([0.0, 1.0, 0.0]));
    let (value_ortho, grad_ortho) = term_ortho.separation_barrier_value_and_grad_for_test(1.0);
    assert!(
        grad_ortho.iter().all(|&g| g == 0.0),
        "orthogonal (well-separated) decoders must leave the separating gradient \
         identically zero (strict no-op force) — healthy fits steer unchanged: {grad_ortho:?}"
    );
    assert!(
        value_ortho.abs() < 1.0e-4,
        "orthogonal decoders' barrier value must be negligible (only the ε-softening \
         constant), got {value_ortho}"
    );
}

/// #1625 — build a 2-atom periodic SAE term whose single-row decoders realize a
/// chosen squared alignment `c² = cos²θ` (`dec0 = e0`, `dec1 = (cosθ, sinθ, 0)`),
/// co-firing under softmax so the separation barrier's coactivation `q_01 > 0`.
/// The shared regression fixture for the collinearity-gate guards below.
fn aligned_two_atom_term_with_c2(c2: f64) -> SaeManifoldTerm {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45], [0.10]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let logits = array![
        [0.7, -0.2],
        [0.1, 0.4],
        [-0.3, 0.5],
        [0.6, -0.1],
        [0.2, 0.3],
        [0.4, 0.1]
    ];
    let cos = c2.sqrt();
    let sin = (1.0 - c2).max(0.0).sqrt();
    let row_decoder = |r: [f64; 3]| {
        let mut d = Array2::<f64>::zeros((3, 3));
        d[[0, 0]] = r[0];
        d[[0, 1]] = r[1];
        d[[0, 2]] = r[2];
        d
    };
    let make = |name: &str, phi: Array2<f64>, jet: Array3<f64>, decoder: Array2<f64>| {
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
    };
    let atom0 = make("periodic0", phi0, jet0, row_decoder([1.0, 0.0, 0.0]));
    let atom1 = make("periodic1", phi1, jet1, row_decoder([cos, sin, 0.0]));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0, coords1],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(0.8),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap()
}

/// The Jeffreys separation barrier `−½ log det F` is a SOFT interior-point
/// COLLAPSE-prevention barrier with NO hard collinearity gate: its force
/// `q²o/(1−q²o²)·∂o/∂B` vanishes as `O(o)` for separated atoms and diverges as
/// `det F → 0`, so the soft interior-point structure IS the gate. This is the
/// root-cause cure for the #1625 stall WITHOUT a tuned threshold: the ungated
/// pairwise `−log(1−c²+ε)` exerted an O(1) force at moderate `c²` (its
/// `∂/∂o = 1/(1−o) → 1` even as `o → 0`) that dragged a healthy fit off the data
/// optimum, which the old code suppressed with a hard `w(c²)` smoothstep; the
/// Jeffreys force instead falls off smoothly on its own. Guard: the separating
/// force must GROW with alignment and become NEGLIGIBLE (relative to the near-
/// collapse force) as the atoms separate, so a well-separated dictionary is steered
/// as if the barrier were absent.
#[test]
fn separation_barrier_force_vanishes_smoothly_as_atoms_separate() {
    let grad_norm = |c2: f64| -> (f64, f64) {
        let term = aligned_two_atom_term_with_c2(c2);
        let (v, g) = term.separation_barrier_value_and_grad_for_test(1.0);
        (v, g.iter().map(|x| x * x).sum::<f64>().sqrt())
    };
    // Near-orthogonal, moderate, and near-collapse alignments.
    let (v_lo, f_lo) = grad_norm(0.02);
    let (v_mid, f_mid) = grad_norm(0.36);
    let (v_hi, f_hi) = grad_norm(0.8);

    // Interior-point growth: value strictly increases with alignment toward the
    // collapse boundary (no hard gate flattening the low-overlap regime to 0).
    assert!(
        v_lo < v_mid && v_mid < v_hi,
        "barrier value must grow with alignment: lo(c²=.02)={v_lo:.3e} < mid(.36)={v_mid:.3e} < hi(.8)={v_hi:.3e}"
    );
    // The near-collapse pair feels a genuine separating force.
    assert!(
        f_hi > 0.0,
        "the near-collapse pair must feel a separating force, got {f_hi}"
    );
    // Force GROWS monotonically toward collapse and VANISHES as atoms separate:
    // the soft auto-gate that replaces the hard `w(c²)` threshold and prevents the
    // #1625 stall. At c²=0.02 the force is a small fraction of the near-collapse
    // force (it scales like O(o)), so it cannot dominate a healthy fit's residual.
    assert!(
        f_lo < f_mid && f_mid < f_hi,
        "separating force must grow with alignment: lo={f_lo:.3e} < mid={f_mid:.3e} < hi={f_hi:.3e}"
    );
    assert!(
        f_lo < 0.2 * f_hi,
        "the separating force must become negligible as atoms separate (O(o) soft \
         auto-gate): near-orthogonal force {f_lo:.3e} must be ≪ near-collapse force {f_hi:.3e}"
    );
}

/// The Jeffreys barrier's analytic gradient `Σ_e α_e·∂o_e/∂B`,
/// `α_e = −F⁻¹[jₑ,kₑ]·q_e`, must match the finite difference of its OWN value
/// `−½ log det F` — the exact value/gradient-consistency contract the line search
/// relies on (a desync would let the value see a force the Newton step never
/// modelled). Evaluated at `c² = 0.7`, on the interior-point interior where the
/// force is materially nonzero, so the `F⁻¹` prefactor and the rank-aware
/// `∂o/∂B` carrier are both load-bearing.
#[test]
fn separation_barrier_gated_gradient_matches_fd_1625() {
    let c2 = 0.7_f64;
    let cos = c2.sqrt();
    let sin = (1.0 - c2).sqrt();
    // Rebuild the term from explicit decoders so we can perturb a single
    // decoder coefficient and recompute the value.
    let build = |d1: [f64; 3]| -> SaeManifoldTerm {
        let mut t = aligned_two_atom_term_with_c2(c2);
        // Overwrite atom1's decoder row 0 with the perturbed direction.
        t.atoms[1].decoder_coefficients[[0, 0]] = d1[0];
        t.atoms[1].decoder_coefficients[[0, 1]] = d1[1];
        t.atoms[1].decoder_coefficients[[0, 2]] = d1[2];
        t
    };
    let base = build([cos, sin, 0.0]);
    let (_v, grad) = base.separation_barrier_value_and_grad_for_test(1.0);
    let offsets = base.beta_offsets();
    let p = base.output_dim();
    // FD each of atom1's row-0 decoder coefficients against the value.
    let h = 1.0e-7;
    let mut max_rel = 0.0_f64;
    for o in 0..3 {
        let mut plus = [cos, sin, 0.0];
        let mut minus = [cos, sin, 0.0];
        plus[o] += h;
        minus[o] -= h;
        let vp = build(plus).separation_barrier_value(1.0);
        let vm = build(minus).separation_barrier_value(1.0);
        let fd = (vp - vm) / (2.0 * h);
        let analytic = grad[offsets[1] + 0 * p + o];
        let rel = (fd - analytic).abs() / (1.0 + fd.abs().max(analytic.abs()));
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_rel < 1.0e-5,
        "gated barrier analytic ∂P/∂B must match FD of the value (incl. the smoothstep \
         w'(c²) term) on the ramp: max rel err {max_rel:.3e}"
    );
}

/// #1625 — within a Newton step the barrier's normalized coactivation `q_jk` is a
/// FROZEN weight (the gradient differentiates only the decoder shape `c²`), so the
/// line-search VALUE must read the same frozen `q` even after the trial logits
/// move — otherwise value and gradient desync in the logit block (the original
/// #1625 defect, surfaced as a phantom logit gradient the Newton step never
/// modelled). After an assembly freezes the coactivation, perturbing a logit must
/// leave `separation_barrier_value` unchanged (the decoders are untouched, and `q`
/// is frozen). Uses an aligned (above-gate) term so the barrier is genuinely live.
#[test]
fn separation_barrier_value_frozen_coactivation_invariant_to_logit_moves_1625() {
    let mut term = aligned_two_atom_term_with_c2(0.8);
    let target = Array2::<f64>::zeros((term.n_obs(), term.output_dim()));
    let rho = SaeManifoldRho::new(
        -2.0,
        -2.0,
        vec![Array1::from_vec(vec![-2.0]), Array1::from_vec(vec![-2.0])],
    );
    // Assemble once to FREEZE the coactivation gate at the current logits.
    term.assemble_arrow_schur(target.view(), &rho, None)
        .expect("assemble freezes the barrier coactivation");
    let value_before = term.separation_barrier_value(1.0);
    assert!(value_before > 0.0, "aligned pair must have a live barrier");
    // Move the logits substantially WITHOUT re-assembling (mimics a line-search
    // trial). The frozen coactivation must keep the barrier value pinned.
    for v in term.assignment.logits.iter_mut() {
        *v += 0.37;
    }
    let value_after = term.separation_barrier_value(1.0);
    assert!(
        (value_after - value_before).abs() <= 1.0e-12 * (1.0 + value_before.abs()),
        "frozen coactivation must hold the barrier value across logit moves: \
         before={value_before:.12e} after={value_after:.12e}"
    );
}

/// #1610 — the separation-barrier collapse-threshold (the decoder-norm floor
/// below which an atom is shape-undefined and the barrier abstains) must be
/// DATA-DERIVED / scale-invariant, not an absolute magic constant.
///
/// Direct-helper arm: `barrier_norm_floor_sq` is exactly
/// `SAE_BARRIER_ACTIVE_NORM_REL_FLOOR² · max_k ‖B_k‖²_F`, equivariant under a
/// global rescaling of the decoders by `s²`, and reduces to the historical
/// absolute `1e-6²` floor at unit decoder scale (`max ‖B_k‖²_F = 1`). The
/// all-zero dictionary yields `0` (no live shape).
#[test]
fn barrier_norm_floor_is_data_derived_scale_invariant_1610() {
    // max ‖B_k‖²_F = 4.0 ⇒ floor² = (1e-6)²·4 = 4e-12.
    let norm_sq = [1.0_f64, 4.0, 0.25];
    let floor = SaeManifoldTerm::barrier_norm_floor_sq(&norm_sq);
    let rel = SAE_BARRIER_ACTIVE_NORM_REL_FLOOR;
    assert!(
        (floor - rel * rel * 4.0).abs() <= 1e-30,
        "floor² must be rel²·max‖B_k‖²_F = {}, got {floor}",
        rel * rel * 4.0
    );
    // At the canonical unit decoder scale this reduces to the historical 1e-6
    // absolute floor (floor² = 1e-12), so existing unit-scale fits are unchanged.
    let unit = SaeManifoldTerm::barrier_norm_floor_sq(&[1.0]);
    assert!(
        (unit - 1.0e-12).abs() <= 1e-27,
        "at unit decoder scale the floor must equal the historical 1e-6² = 1e-12, got {unit}"
    );
    // Equivariance: scaling every ‖B_k‖²_F by s² scales the floor² by s².
    for &s2 in &[1.0e-12_f64, 1.0e6, 9.0] {
        let scaled: Vec<f64> = norm_sq.iter().map(|v| v * s2).collect();
        let f_scaled = SaeManifoldTerm::barrier_norm_floor_sq(&scaled);
        assert!(
            (f_scaled - s2 * floor).abs() <= s2 * floor * 1e-9 + 1e-30,
            "floor² must scale by s² under a global ‖B‖² rescaling: s²={s2}, \
             expected {}, got {f_scaled}",
            s2 * floor
        );
    }
    // All-zero dictionary: no live atom to be a shape ⇒ floor 0 (the exactly-0
    // self-norm check abstains every pair anyway).
    assert_eq!(SaeManifoldTerm::barrier_norm_floor_sq(&[0.0, 0.0]), 0.0);
}

/// #1610 — END-TO-END scale invariance of collapse prevention: the separation
/// barrier penalizes the SHAPE alignment `c²` weighted by the (normalized)
/// coactivation `q`, both of which are scale-free, so the barrier VALUE is
/// invariant under a global rescaling of the decoders. The OLD absolute
/// `1e-6` norm floor broke this: a corpus whose natural decoder scale fell below
/// the floor had its decoders classified as shape-undefined and collapse
/// prevention was silently disabled (value → 0). With the data-derived relative
/// floor the barrier engages identically at any decoder scale.
#[test]
fn separation_barrier_collapse_prevention_is_scale_invariant_1610() {
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45], [0.10]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let logits = array![
        [0.7, -0.2],
        [0.1, 0.4],
        [-0.3, 0.5],
        [0.6, -0.1],
        [0.2, 0.3],
        [0.4, 0.1]
    ];
    let row_decoder = |r: [f64; 3]| {
        let mut d = Array2::<f64>::zeros((3, 3));
        d[[0, 0]] = r[0];
        d[[0, 1]] = r[1];
        d[[0, 2]] = r[2];
        d
    };
    // Aligned (c² = 0.8), co-firing under softmax — the collapse-prone pair.
    let dir0 = [1.0, 0.0, 0.0];
    let dir1 = [0.894_427_191, 0.447_213_595, 0.0];
    let build_at_scale = |s: f64| {
        let scale_row = |r: [f64; 3]| [r[0] * s, r[1] * s, r[2] * s];
        let make = |name: &str, phi: Array2<f64>, jet: Array3<f64>, decoder: Array2<f64>| {
            SaeManifoldAtom::new_with_provided_function_gram(
                name,
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(3),
            )
            .unwrap()
            .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
        };
        let atom0 = make(
            "p0",
            phi0.clone(),
            jet0.clone(),
            row_decoder(scale_row(dir0)),
        );
        let atom1 = make(
            "p1",
            phi1.clone(),
            jet1.clone(),
            row_decoder(scale_row(dir1)),
        );
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits.clone(),
            vec![coords0.clone(), coords1.clone()],
            vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ],
            AssignmentMode::softmax(0.8),
        )
        .unwrap();
        SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap()
    };

    // Unit scale: the barrier engages and penalizes the aligned pair.
    let value_unit = build_at_scale(1.0).separation_barrier_value(1.0);
    assert!(
        value_unit > 0.0,
        "barrier must engage on the aligned, co-firing pair at unit scale, got {value_unit}"
    );
    // Tiny scale: decoder entries ~1e-7 ⇒ ‖B_k‖²_F ~1e-14 < the OLD absolute
    // floor² (1e-12). Under the old absolute floor the barrier would have
    // abstained (value 0 — collapse prevention disabled). The data-derived floor
    // keeps it engaged with the SAME value (c² and q are scale-free).
    let value_tiny = build_at_scale(1.0e-7).separation_barrier_value(1.0);
    assert!(
        value_tiny > 0.0,
        "data-derived floor must keep collapse prevention ENGAGED at a tiny decoder \
         scale where the old absolute 1e-6 floor disabled it, got {value_tiny}"
    );
    assert!(
        (value_tiny - value_unit).abs() <= value_unit.abs() * 1e-9,
        "the barrier value is scale-free (shape + coactivation only): unit={value_unit} \
         must equal tiny-scale={value_tiny}"
    );
    // And a HUGE scale leaves it unchanged too (symmetry of the invariance).
    let value_huge = build_at_scale(1.0e6).separation_barrier_value(1.0);
    assert!(
        (value_huge - value_unit).abs() <= value_unit.abs() * 1e-9,
        "barrier value must be invariant at large decoder scale too: unit={value_unit} \
         huge={value_huge}"
    );
}

/// #1610 — the decoder-repulsion collapse-prevention conditioner must be
/// PRINCIPLED, not a hand-picked absolute magic constant:
///   1. its strength is a DERIVED dimensionless fraction of the primary
///      separation-barrier strength (`μ_rep = ratio · μ_sep`), not an
///      independent `1e-3`; and
///   2. after the #1610 energy normalization the realized repulsion penalty is a
///      function of the dimensionless collinearity `c_jk² ∈ [0,1]` ALONE, so it
///      is INVARIANT under a global corpus rescaling `B_k → s·B_k`.
///
/// Property (2) is the property the OLD absolute constant VIOLATED: it weighted
/// the un-normalized cross-Gram energy `‖B_jB_kᵀ‖²_F = c²·‖B_j‖²_F·‖B_k‖²_F`, so
/// the repulsion value scaled as `s⁴` under a rescaling by `s` while the
/// collapse geometry (`c²`, the gate) was identical — the same scale bug #1610
/// fixed for the separation barrier's norm floor. The test builds a fixed,
/// near-collinear (gate-engaged) K=2 fixture and asserts the repulsion value is
/// equal across decoder scales spanning 13 orders of magnitude. With the old
/// `½·STRENGTH·c²·s⁴` weighting these would differ by `s⁴` (up to `1e52`), so
/// this fails before the normalization and passes after.
#[test]
pub(crate) fn decoder_repulsion_strength_is_derived_and_scale_invariant_1610() {
    // (1) Strength is a DERIVED dimensionless fraction of the data-derived
    // separation-barrier strength μ_C, not an independent absolute constant.
    // (Checked on a constructed term below, after the fixture builder — μ_C is
    // now read from the data-fit inseparability of the live design/routing, not a
    // global constant or a rank-count heuristic.)

    // (2) End-to-end scale invariance of the repulsion value.
    let coords0 = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let coords1 = array![[0.15], [0.30], [0.65], [0.90], [0.45], [0.10]];
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    let logits = array![
        [0.7, -0.2],
        [0.1, 0.4],
        [-0.3, 0.5],
        [0.6, -0.1],
        [0.2, 0.3],
        [0.4, 0.1]
    ];
    // Two atoms whose decoders are NEAR-collinear (cosine 0.9 ⇒ c² = 0.81, above
    // the 0.5 gate but strictly < 1), so the gate is partially engaged and the
    // penalty is strictly positive and finite. Rank-1 decoders (only row 0
    // nonzero) keep `‖B_k‖²_F` trivial to reason about: at scale `s`,
    // `‖B_0‖²_F = ‖B_1‖²_F = s²` and `c² = 0.81` (scale-free).
    let build_at_scale = |s: f64| {
        let mut dec0 = Array2::<f64>::zeros((3, 3));
        dec0[[0, 0]] = s;
        let mut dec1 = Array2::<f64>::zeros((3, 3));
        dec1[[0, 0]] = 0.9 * s;
        dec1[[0, 1]] = (1.0 - 0.9 * 0.9_f64).sqrt() * s; // ‖row‖ = s, cosine with dec0 = 0.9
        let make = |name: &str, phi: Array2<f64>, jet: Array3<f64>, decoder: Array2<f64>| {
            SaeManifoldAtom::new_with_provided_function_gram(
                name,
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                decoder,
                Array2::<f64>::eye(3),
            )
            .unwrap()
            .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
        };
        let atom0 = make("rep0", phi0.clone(), jet0.clone(), dec0);
        let atom1 = make("rep1", phi1.clone(), jet1.clone(), dec1);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits.clone(),
            vec![coords0.clone(), coords1.clone()],
            vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Circle { period: 1.0 },
            ],
            AssignmentMode::softmax(0.8),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap();
        term.refresh_decoder_repulsion_gate();
        term
    };

    // (1) — the repulsion strength is the derived fraction
    // `SAE_DECODER_REPULSION_BARRIER_RATIO · μ_C` of the separation-barrier
    // strength, and μ_C is itself EVIDENCE-DERIVED — the worst-case data-fit
    // inseparability strength `γ/(1-γ)` over the co-active pairs (#1610), NOT a
    // hand-picked magnitude and NOT a rank-count heuristic. Checked on a
    // constructed unit-scale term (μ_C is a per-term, per-pair quantity).
    let unit_term = build_at_scale(1.0);
    let expected = SAE_DECODER_REPULSION_BARRIER_RATIO * unit_term.separation_barrier_strength();
    assert_eq!(
        unit_term.decoder_repulsion_strength(),
        expected,
        "repulsion strength must be the derived fraction {SAE_DECODER_REPULSION_BARRIER_RATIO} \
         of the evidence-derived separation-barrier strength {}, got {}",
        unit_term.separation_barrier_strength(),
        unit_term.decoder_repulsion_strength(),
    );
    // The evidence-derived strength is a strictly positive, finite number for a
    // genuinely co-active pair (the data-fit couples them, so γ > 0), and it is
    // NOT the old overcompleteness ratio (which for two periodic M=3 atoms in p=3
    // was pinned at exactly 2.0). It is the reciprocal-margin `γ/(1-γ)` to the
    // data-fit's co-collapse boundary, read from the chart design + routing.
    let mu_c = unit_term.separation_barrier_strength();
    assert!(
        mu_c > 0.0 && mu_c.is_finite(),
        "μ_C must be a positive finite evidence-derived strength for a co-active \
         pair, got {mu_c}"
    );
    // Decoder-scale invariance of the STRENGTH: γ (hence μ_C) is read from the
    // chart design + routing, not the decoder magnitudes, so rescaling the whole
    // dictionary leaves the strength unchanged (unlike a REML `λ ∝ σ²/τ²`).
    let mu_c_tiny = build_at_scale(1.0e-7).separation_barrier_strength();
    let mu_c_huge = build_at_scale(1.0e6).separation_barrier_strength();
    let rel_mu = |a: f64, b: f64| (a - b).abs() / b.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel_mu(mu_c_tiny, mu_c) <= 1e-9 && rel_mu(mu_c_huge, mu_c) <= 1e-9,
        "evidence-derived μ_C must be decoder-scale invariant: unit={mu_c} \
         tiny={mu_c_tiny} huge={mu_c_huge}"
    );

    let value_unit = build_at_scale(1.0).decoder_repulsion_value(1.0);
    assert!(
        value_unit > 0.0 && value_unit.is_finite(),
        "near-collinear gate-engaged pair must yield a positive finite repulsion \
         value at unit scale, got {value_unit}"
    );
    // Same collapse geometry (c², gate identical) at a tiny and a huge corpus
    // scale: the energy-normalized penalty is invariant. The OLD un-normalized
    // weighting would scale these by s⁴ = 1e-28 and 1e24 respectively.
    let value_tiny = build_at_scale(1.0e-7).decoder_repulsion_value(1.0);
    let value_huge = build_at_scale(1.0e6).decoder_repulsion_value(1.0);
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(f64::MIN_POSITIVE);
    assert!(
        rel(value_tiny, value_unit) <= 1e-9,
        "repulsion value must be scale-invariant: unit={value_unit} tiny={value_tiny} \
         (old absolute constant scaled this by s⁴)"
    );
    assert!(
        rel(value_huge, value_unit) <= 1e-9,
        "repulsion value must be scale-invariant: unit={value_unit} huge={value_huge} \
         (old absolute constant scaled this by s⁴)"
    );
}

/// #1610 — the separation-barrier strength is EVIDENCE-DERIVED: the per-pair
/// strength `μ_jk = γ_jk/(1-γ_jk)` is a MONOTONE function of the data-fit
/// inseparability `γ_jk` (the largest canonical correlation of the two atoms'
/// coactivation-weighted chart designs — the quantity that decides whether the
/// joint inner penalized quasi-Laplace Hessian stays PD). This replaces the old geometry
/// heuristic `Σ min(M_k,p)/min(n,p)`, which was blind to the actual design/routing
/// and so gave the SAME strength to a data-separable pair and a data-degenerate
/// one. Here two atoms with IDENTICAL chart designs are driven from data-fit
/// SEPARABLE (disjoint routing ⇒ γ ≈ 0 ⇒ μ ≈ 0) to data-fit DEGENERATE
/// (overlapping routing on a shared design ⇒ γ → 1 ⇒ μ large), and the strength
/// must rise accordingly. γ (hence μ) is read from the design + routing only, so
/// it is decoder-scale free.
#[test]
pub(crate) fn barrier_strength_tracks_data_fit_inseparability_1610() {
    let coords = array![[0.05], [0.20], [0.55], [0.80], [0.35], [0.65]];
    let (phi, jet) = periodic_basis(&coords);
    // Two atoms with the SAME chart design (identical Φ) so the ONLY thing that
    // sets γ is the coactivation-weighted routing overlap we pass in.
    let make = |name: &str, decoder: Array2<f64>| {
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi.clone(),
            jet.clone(),
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
    };
    let mut dec0 = Array2::<f64>::zeros((3, 3));
    dec0[[0, 0]] = 1.0;
    let mut dec1 = Array2::<f64>::zeros((3, 3));
    dec1[[0, 1]] = 1.0;
    let logits = array![
        [0.7, -0.2],
        [0.1, 0.4],
        [-0.3, 0.5],
        [0.6, -0.1],
        [0.2, 0.3],
        [0.4, 0.1]
    ];
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords.clone(), coords.clone()],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(0.8),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![make("a0", dec0), make("a1", dec1)], assignment).unwrap();

    // DISJOINT routing: atom 0 fires only on the first three rows, atom 1 only on
    // the last three. No row co-fires, so the weighted cross-design Gram is 0 ⇒
    // γ ≈ 0 ⇒ the data-fit already separates the pair ⇒ μ ≈ 0 (no safeguard owed).
    let gates_disjoint = array![
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0]
    ];
    let gamma_sep = term.design_inseparability_with_gates(gates_disjoint.view(), 0, 1);
    let mu_sep = term.barrier_pair_strength_with_gates(gates_disjoint.view(), 0, 1);
    assert!(
        gamma_sep <= 1e-9,
        "disjoint routing on any design ⇒ data-fit separable ⇒ γ ≈ 0, got {gamma_sep}"
    );
    assert!(
        mu_sep <= 1e-6,
        "a data-fit-separable pair owes ~no separation barrier, got μ = {mu_sep}"
    );

    // OVERLAPPING routing on the SHARED design: both atoms fire together on every
    // row, so the two coactivation-weighted design column spaces COINCIDE ⇒ γ → 1
    // (the data-fit cannot tell them apart) ⇒ μ = γ/(1-γ) is large.
    let gates_overlap = array![
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0]
    ];
    let gamma_deg = term.design_inseparability_with_gates(gates_overlap.view(), 0, 1);
    let mu_deg = term.barrier_pair_strength_with_gates(gates_overlap.view(), 0, 1);
    assert!(
        gamma_deg > 0.999,
        "identical designs + identical routing ⇒ perfectly inseparable ⇒ γ → 1, got {gamma_deg}"
    );
    assert!(
        mu_deg > mu_sep + 1.0,
        "the barrier strength MUST rise as the data-fit inseparability rises: \
         separable μ={mu_sep} vs degenerate μ={mu_deg}"
    );
    // μ = γ/(1-γ) exactly (evidence-derived reciprocal margin), no hidden magic.
    let eps = SAE_SEPARATION_BARRIER_EPS;
    let expected_deg = gamma_deg / (1.0 - gamma_deg).max(eps);
    assert!(
        (mu_deg - expected_deg).abs() <= expected_deg.abs() * 1e-9 + 1e-12,
        "μ must equal γ/max(1-γ,ε): γ={gamma_deg} expected={expected_deg} got={mu_deg}"
    );

    // γ (hence μ) is a DESIGN/ROUTING quantity, independent of decoder magnitude:
    // rescaling the decoders leaves both unchanged.
    let mut big0 = Array2::<f64>::zeros((3, 3));
    big0[[0, 0]] = 1.0e6;
    let mut big1 = Array2::<f64>::zeros((3, 3));
    big1[[0, 1]] = 1.0e6;
    let assignment2 = SaeAssignment::from_blocks_with_mode_and_manifolds(
        array![
            [0.7, -0.2],
            [0.1, 0.4],
            [-0.3, 0.5],
            [0.6, -0.1],
            [0.2, 0.3],
            [0.4, 0.1]
        ],
        vec![coords.clone(), coords.clone()],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(0.8),
    )
    .unwrap();
    let term_big =
        SaeManifoldTerm::new(vec![make("a0", big0), make("a1", big1)], assignment2).unwrap();
    let mu_deg_big = term_big.barrier_pair_strength_with_gates(gates_overlap.view(), 0, 1);
    assert!(
        (mu_deg_big - mu_deg).abs() <= mu_deg.abs() * 1e-9,
        "evidence-derived μ must be decoder-scale invariant: unit={mu_deg} big={mu_deg_big}"
    );
}

/// #976 distinct-basin lever: the co-collapse multi-start reseed must read a
/// DIFFERENT principal subspace on each retry. The PC-pair rotation offset (=
/// the 0-based retry index) shifts which residual PC pair each periodic atom
/// reads, so two consecutive multi-start attempts produce seed coordinates that
/// are not bit-identical. Without the rotation every retry re-reads the same
/// leading PCs of the (unchanged) residual and the budget-N multi-start is N
/// identical attempts — the K=3 coin-flip this fix targets.
#[test]
pub(crate) fn co_collapse_reseed_rotation_explores_distinct_subspaces() {
    // A residual with three well-separated PC directions (p = 6 so >= 6 PCs
    // exist and the offset can rotate through several disjoint pairs).
    let residual = array![
        [3.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        [-3.0, -0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.2, 0.0, 0.0],
        [0.0, 0.0, -2.0, -0.2, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, -1.0, -0.3],
    ];
    let kinds = vec![
        SaeAtomBasisKind::Periodic,
        SaeAtomBasisKind::Periodic,
        SaeAtomBasisKind::Periodic,
    ];
    let dims = vec![1usize, 1, 1];
    let seed0 = sae_pca_seed_initial_coords_with_pc_offset(residual.view(), &kinds, &dims, 0)
        .expect("offset-0 seed");
    let seed1 = sae_pca_seed_initial_coords_with_pc_offset(residual.view(), &kinds, &dims, 1)
        .expect("offset-1 seed");
    let seed2 = sae_pca_seed_initial_coords_with_pc_offset(residual.view(), &kinds, &dims, 2)
        .expect("offset-2 seed");
    let maxdiff = |a: &Array3<f64>, b: &Array3<f64>| -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    };
    assert!(
        maxdiff(&seed0, &seed1) > 1e-3,
        "retry 0 vs 1 must read distinct PC pairs (max coord diff = {:.3e})",
        maxdiff(&seed0, &seed1)
    );
    assert!(
        maxdiff(&seed1, &seed2) > 1e-3,
        "retry 1 vs 2 must read distinct PC pairs (max coord diff = {:.3e})",
        maxdiff(&seed1, &seed2)
    );
    // Offset 0 must be byte-identical to the no-offset entry point (the K=1 and
    // initial-fit seed paths must be untouched).
    let seed_plain =
        sae_pca_seed_initial_coords(residual.view(), &kinds, &dims).expect("plain seed");
    assert_eq!(
        seed0, seed_plain,
        "offset-0 seed must equal the no-offset seed bit-for-bit"
    );
}

/// #976 determinism (issue requirement: identical inputs ⇒ identical output
/// run-to-run). The PCA seed is the SAE-fit entry the owner flagged as flipping
/// the collapse basin between runs. Pin that repeated calls on identical input
/// are bit-identical under the process-default (global Rayon) faer backend, so
/// the now-rotated multi-start is a fixed pass rather than a coin-flip. (The
/// cross-thread-count arm is exercised on the cluster via RAYON_NUM_THREADS; faer's
/// blocked factorizations keep a fixed per-element reduction order, so the
/// global-state-mutating Seq/Par toggle is deliberately NOT done here — it would
/// race the rest of the suite's parallel tests.)
#[test]
pub(crate) fn pca_seed_is_run_to_run_reproducible() {
    let residual = array![
        [3.0, 0.1, -0.2, 0.4, 0.0, 0.05],
        [-3.0, -0.1, 0.2, -0.4, 0.0, -0.05],
        [0.3, 0.0, 2.0, 0.2, 0.1, 0.0],
        [-0.3, 0.0, -2.0, -0.2, -0.1, 0.0],
        [0.0, 0.2, 0.1, 0.0, 1.0, 0.3],
        [0.0, -0.2, -0.1, 0.0, -1.0, -0.3],
    ];
    let kinds = vec![SaeAtomBasisKind::Periodic, SaeAtomBasisKind::Periodic];
    let dims = vec![1usize, 1];
    let seed_a = sae_pca_seed_initial_coords(residual.view(), &kinds, &dims).expect("seed #1");
    let seed_b = sae_pca_seed_initial_coords(residual.view(), &kinds, &dims).expect("seed #2");
    assert_eq!(
        seed_a, seed_b,
        "PCA seed must be bit-identical run-to-run (the issue's determinism \
         requirement)"
    );
}

/// #976 decoder arm is a strict no-op for K=1: a single atom has no peer to
/// fall behind, so the guard must never reseed or record an event even when
/// the lone decoder is tiny. This pins the "K=1 path unchanged" guarantee.
#[test]
pub(crate) fn decoder_norm_guard_is_noop_for_k1() {
    let mut term = trivial_k1_euclidean_term();
    let n = term.n_obs();
    let p = term.output_dim();
    let target = Array2::<f64>::zeros((n, p));
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![0.0_f64]]);
    let before = term.atoms[0].decoder_coefficients.clone();
    term.enforce_decoder_norm_guard(target.view(), 0, &rho, None)
        .expect("K=1 decoder-norm guard must be a no-op, never error");
    assert!(
        term.collapse_events().is_empty(),
        "K=1 must record no decoder-collapse events"
    );
    assert_eq!(
        term.atoms[0].decoder_coefficients, before,
        "K=1 decoder must be untouched by the guard"
    );
}

/// #1026 — the hybrid split is **load-bearing on the reconstruction**: a slot
/// whose verdict selects LINEAR has its curved decoded image replaced by its
/// fitted straight sub-model, and that substitution match-or-beats the
/// all-curved reconstruction on explained variance at strictly fewer
/// parameters (the strict-generalization dominance floor of #1026).
///
/// The test pins two regimes:
///  * No report ⇒ the collapsed reconstruction is bit-identical to the curved
///    one (the verdict cannot silently alter the fit before it is computed).
///  * A genuinely STRAIGHT atom (its decoded image is a line) forces the
///    dominance floor to select linear; collapsing it leaves the
///    reconstruction essentially unchanged (a line collapsed to its own line),
///    so EV is preserved, while the slot sheds its `M·p − 2·p` curved
///    coefficients — EV-per-parameter strictly improves.
#[test]
pub(crate) fn hybrid_collapse_is_load_bearing_and_dominates() {
    let (mut term, _t, rho) = small_two_atom_periodic_term();

    // (1) Before the report exists, collapse == curved reconstruction.
    let curved = term
        .try_fitted_for_rho(&rho)
        .expect("curved reconstruction assembles");
    let pre = term
        .hybrid_collapsed_reconstruction(&rho)
        .expect("collapse with no report returns the curved fit");
    assert!(
        (&curved - &pre).iter().all(|d| d.abs() < 1e-15),
        "with no hybrid-split report the collapse must equal the curved fit"
    );

    // Make atom 0 genuinely STRAIGHT: a single nonzero basis-0 coefficient
    // decodes γ(t) = φ₀(t)·b, and we additionally drive its decoded image to a
    // pure line by zeroing the higher harmonics — Θ → 0 ⇒ the dominance floor
    // must select linear for this slot.
    for basis_row in 1..term.atoms[0].decoder_coefficients.nrows() {
        for out_col in 0..term.atoms[0].decoder_coefficients.ncols() {
            term.atoms[0].decoder_coefficients[[basis_row, out_col]] = 0.0;
        }
    }

    // Target = the term's own curved reconstruction (after straightening atom 0)
    // ⇒ EV(curved) = 1 exactly, and each atom's leave-this-atom-out response
    // residual `y_resp` equals its own mass-scaled contribution `a_k·γ_k`. The
    // common-evidence selector (#1202) scores both candidates against that
    // residual, so the target is required.
    // A PHYSICAL target: the term's curved reconstruction plus a tiny smooth
    // residual, so the term does NOT reconstruct it to machine zero. This is
    // REQUIRED for the hybrid split to run at all. `compute_hybrid_split_report`
    // derives the rank-charge noise floor as `phi_hat = ||target - full||^2/(n*p)`;
    // with `target == full` (the exact self-reconstruction this fixture used to
    // pass) that floor is EXACTLY 0, at which `build_atom_candidates` cannot price
    // the evidence and refuses every atom as `Unadjudicable` (#2362) -> an empty
    // report. A real fit never reconstructs its target to zero. The perturbation
    // is orders of magnitude below either atom's own contribution, so the
    // straight-vs-curved discrimination below is unchanged; it only lifts phi_hat
    // off the degenerate zero.
    let full = term
        .try_fitted_for_rho(&rho)
        .expect("post-straighten curved reconstruction assembles");
    let mut target = full.clone();
    for i in 0..target.nrows() {
        for j in 0..target.ncols() {
            target[[i, j]] += 1.0e-3 * (0.7 * (i as f64 + 1.0) + 1.3 * (j as f64 + 1.0)).sin();
        }
    }

    // Compute and install the real hybrid-split report (closed-form, no outer
    // fit — sidesteps #1051).
    let report = term
        .compute_hybrid_split_report(&rho, Some(target.view()))
        .expect("hybrid split report computes")
        .expect("eligible d=1 atoms present a report");
    term.hybrid_split_report = Some(report);

    // The straight atom 0 must have collapsed to linear (its verdict carries a
    // straight sub-model).
    let collapsed_any = term
        .hybrid_split_report
        .as_ref()
        .unwrap()
        .verdicts
        .iter()
        .any(|v| v.linear_image.is_some());
    assert!(
        collapsed_any,
        "a straight atom must collapse at least one slot to the linear tail"
    );

    // EV of the ALL-CURVED reconstruction against the physical target. The
    // perturbation is tiny, so the curved fit still explains essentially all of
    // the target — but not tautologically 1. (The prior `EV(target, target)`
    // measured SELF-EV, which is 1 for any input and only looked meaningful when
    // `target == full`.) The dominance floor below compares the COLLAPSED
    // reconstruction against THIS curved baseline; because collapsing the
    // straightened slot replaces its constant curve with the best straight line to
    // the same residual, it can only match or beat the curved fit.
    let ev_curved = reconstruction_explained_variance(target.view(), full.view())
        .expect("curved-reconstruction EV defined");
    assert!(
        ev_curved > 0.99,
        "the curved fit must explain essentially all of the barely-perturbed \
         target; got {ev_curved}"
    );

    // The collapsed dictionary (straight slot decoded by its line) must
    // match-or-beat the curved EV up to the line-fit residual of an already
    // straight image — which is ~0. This is the dominance floor measured on
    // the EV axis: collapsing a straight atom costs no reconstruction.
    let ev_collapsed = term
        .hybrid_collapsed_explained_variance(target.view(), &rho)
        .expect("collapsed EV evaluates")
        .expect("collapsed EV defined");
    assert!(
        ev_collapsed >= ev_curved - 1e-6,
        "collapsing a straight atom must preserve EV (match-or-beat dominance \
             floor): curved {ev_curved:.9}, collapsed {ev_collapsed:.9}"
    );

    // And the collapsed slot sheds curved coefficients: its evidence-priced
    // parameter count is the 2·p linear budget, strictly below the M·p curved
    // decoder it replaced (M ≥ 3 basis rows here).
    let verdict = term
        .hybrid_split_report
        .as_ref()
        .unwrap()
        .verdicts
        .iter()
        .find(|v| v.linear_image.is_some())
        .expect("a collapsed slot exists");
    let collapsed_idx = verdict.linear_image.as_ref().unwrap().atom_idx;
    let curved_params = term.atoms[collapsed_idx].decoder_coefficients.len();
    assert!(
        verdict.choice.num_parameters < curved_params,
        "the linear-collapsed slot must shed curved coefficients: linear \
             {} < curved {}",
        verdict.choice.num_parameters,
        curved_params
    );

    // #1026 EV-vs-Θ frontier as STRUCTURED report data: recompute the report
    // WITH the reconstruction target so each verdict carries the `(Θ, ΔEV)`
    // pair the roadmap reports against (previously this lived only as a
    // transient `log::info!` line). The target is the term's own curved
    // reconstruction, so every atom's leave-one-atom-out drop `ΔEV_k` is the
    // real EV it earns; the report must surface a finite `(Θ, ΔEV)` for every
    // adjudicated d = 1 slot, and the collapsed-to-linear slot must read its
    // straight signature `Θ ≈ 0`.
    let report_with_ev = term
        .compute_hybrid_split_report(&rho, Some(target.view()))
        .expect("hybrid split report with target computes")
        .expect("eligible d=1 atoms present a report");
    assert!(
        !report_with_ev.verdicts.is_empty(),
        "the report must adjudicate at least one d=1 slot"
    );
    for v in &report_with_ev.verdicts {
        let theta = v
            .fitted_turning
            .unwrap_or_else(|| panic!("verdict '{}' must carry a fitted turning Θ", v.atom_name));
        let dev = v
            .train_loao_delta_ev
            .unwrap_or_else(|| panic!("verdict '{}' must carry a training LOAO ΔEV", v.atom_name));
        assert!(
            theta.is_finite() && theta >= 0.0,
            "fitted turning Θ must be a finite non-negative arc-curvature integral; \
             got {theta} for '{}'",
            v.atom_name
        );
        assert!(
            dev.is_finite(),
            "training LOAO ΔEV must be finite; got {dev} for '{}'",
            v.atom_name
        );
        // The slot that collapsed to the linear tail is straight by definition:
        // its decoded curve integrates ~zero turning.
        if !v.kept_curved {
            assert!(
                theta <= 1e-3,
                "a linear-tail slot must read Θ ≈ 0 (straight image); got {theta} for '{}'",
                v.atom_name
            );
        }
    }

    // #1026 — the POSITIVE arm of the EV-preservation discrimination. The fixture
    // mixes a straightened slot (atom 0: its curved fit IS a line, so collapsing
    // it is lossless — asserted above) with a LOAD-BEARING slot (atom 1: nonzero
    // higher harmonics make its decoded warp a genuinely non-linear function of
    // the coordinate, so collapsing it to a straight secant would raise the
    // reconstruction SSR and DROP EV). The EV-preservation gate keys on exactly
    // that EV loss (`collapse_ssr_increase`), so a correct adjudication must do
    // BOTH: release the straight slot to the linear tail AND keep the load-bearing
    // slot curved while it earns reconstruction. At least one adjudicated slot
    // must therefore be kept curved and carry a strictly positive training LOAO
    // ΔEV — a curveable atom doing real reconstruction work the straight tail
    // cannot capture.
    //
    // On Θ: this fixture reconstructs a 1-D target, and a scalar curve has no
    // geometric turning — the wedge ‖γ' ∧ γ''‖ vanishes identically in one
    // dimension — so every atom honestly reports Θ = 0 here (pinned finite, not
    // the historical `None`, by the loop above, which exercises the constant-image
    // → `Some(0.0)` fix). The geometric Θ-discrimination (high Θ for a real loop,
    // ≈ 0 for a line) is a ≥ 2-D property and is covered where it is meaningful:
    // the real-circle `chart_canonicalization::turning_tests` (→ 2π) and the
    // evidence-level `hybrid_split::tests::turning_residual_selects_curved_on_evidence`.
    // The gate never reads Θ, so this end-to-end test asserts the EV-axis
    // discrimination the gate actually performs, not a turning the fixture's
    // dimensionality cannot exhibit.
    let curved_earner = report_with_ev
        .verdicts
        .iter()
        .find(|v| v.kept_curved && v.train_loao_delta_ev.map(|d| d > 0.0).unwrap_or(false));
    assert!(
        curved_earner.is_some(),
        "a load-bearing curveable slot must be kept curved AND earn positive training \
         LOAO ΔEV (collapsing it would drop reconstruction EV); verdicts = {:?}",
        report_with_ev
            .verdicts
            .iter()
            .map(|v| (
                v.atom_name.clone(),
                v.kept_curved,
                v.fitted_turning,
                v.train_loao_delta_ev
            ))
            .collect::<Vec<_>>()
    );

    // The split is sharp and keyed to the atom identities, not a coincidental
    // count: the slot we straightened (atom 0) is the one released to the linear
    // tail, while the untouched load-bearing slot (atom 1) is the one kept curved.
    // A vacuous "keep everything curved" or "collapse the wrong atom" adjudication
    // fails one of these halves.
    assert_eq!(
        curved_earner.unwrap().atom_name,
        "periodic1",
        "the load-bearing (untouched) atom must be the one kept curved"
    );
    for v in &report_with_ev.verdicts {
        if !v.kept_curved {
            assert_eq!(
                v.atom_name, "periodic0",
                "only the straightened atom may be released to the linear tail; \
                 '{}' collapsed unexpectedly",
                v.atom_name
            );
        }
    }
}

/// #1233 — the hard `top_k` reconstruction must compose with the #1026 hybrid
/// collapse. The FFI top-k path reconstructs from a PROJECTED assignment matrix
/// through [`SaeManifoldTerm::reconstruct_from_assignments`]; that shared
/// assembler must decode a verdict-linear `d = 1` slot by its straight
/// sub-model image (exactly as the production `fitted()` does), not by the
/// original curved decoder. The regression: with `top_k == K` (every atom kept,
/// i.e. the full soft assignment), the collapse-aware projected reconstruction
/// must EXACTLY equal the non-projected collapsed reconstruction, INCLUDING when
/// a slot is hybrid-collapsed linear — and must DIFFER from the curved-only
/// reconstruction, proving the collapse is genuinely engaged on this path.
#[test]
pub(crate) fn topk_reconstruction_composes_with_hybrid_collapse() {
    let (mut term, _t, rho) = small_two_atom_periodic_term();

    // Straighten atom 0 so its verdict collapses to the linear tail.
    for basis_row in 1..term.atoms[0].decoder_coefficients.nrows() {
        for out_col in 0..term.atoms[0].decoder_coefficients.ncols() {
            term.atoms[0].decoder_coefficients[[basis_row, out_col]] = 0.0;
        }
    }
    // A PHYSICAL target: the term's curved reconstruction plus a tiny smooth
    // residual, so the term does NOT reconstruct it to machine zero. This is
    // REQUIRED for the hybrid split to run at all. `compute_hybrid_split_report`
    // derives the rank-charge noise floor as `phi_hat = ||target - full||^2/(n*p)`;
    // with `target == full` (the exact self-reconstruction this fixture used to
    // pass) that floor is EXACTLY 0, at which `build_atom_candidates` cannot price
    // the evidence and refuses every atom as `Unadjudicable` (#2362) -> an empty
    // report. A real fit never reconstructs its target to zero. The perturbation
    // is orders of magnitude below either atom's own contribution, so the
    // straight-vs-curved discrimination below is unchanged; it only lifts phi_hat
    // off the degenerate zero.
    let full = term
        .try_fitted_for_rho(&rho)
        .expect("post-straighten curved reconstruction assembles");
    let mut target = full.clone();
    for i in 0..target.nrows() {
        for j in 0..target.ncols() {
            target[[i, j]] += 1.0e-3 * (0.7 * (i as f64 + 1.0) + 1.3 * (j as f64 + 1.0)).sin();
        }
    }
    let report = term
        .compute_hybrid_split_report(&rho, Some(target.view()))
        .expect("hybrid split report computes")
        .expect("eligible d=1 atoms present a report");
    term.hybrid_split_report = Some(report);
    assert!(
        term.hybrid_linear_image_map().contains_key(&0),
        "atom 0 must have collapsed to a linear image for this regression"
    );

    // #1233 WITNESS. The straightened atom is a CONSTANT (its periodic basis row 0
    // is the DC term), so its fitted linear image equals its own curve and
    // collapsing it is a numerical no-op — on its own it cannot exercise the
    // collapse-aware reconstruction. Install a genuinely SLOPED straight image
    // into the collapsed slot: still a line (zero turning — a legitimate linear
    // tail, NOT the EV-losing over-collapse the gate prevents), but now
    // `b₀ + (t − t̄)·b₁` differs from the constant curve by a real, per-row,
    // measurable amount. The collapse-aware reconstruction MUST decode THIS image,
    // so the composition / engagement assertions below become non-vacuous: they
    // would fail if the top-k path skipped the collapse or decoded a different
    // image.
    const WITNESS_SLOPE: f64 = 0.4;
    {
        let report = term.hybrid_split_report.as_mut().unwrap();
        let img = report
            .verdicts
            .iter_mut()
            .find_map(|v| v.linear_image.as_mut())
            .expect("the collapsed slot must carry a linear image to install a witness into");
        for slope in img.b1.iter_mut() {
            *slope += WITNESS_SLOPE;
        }
    }

    // `top_k == K` keeps every atom: the projected assignment matrix IS the full
    // soft assignment, so the projected (collapse-aware) reconstruction must
    // match the production collapsed `fitted()` bit-for-bit.
    let full_assignments = term.assignment.assignments();
    let projected_collapsed = term
        .reconstruct_from_assignments(full_assignments.view(), true)
        .expect("collapse-aware projected reconstruction assembles");
    let production_collapsed = term.fitted();
    let max_gap = (&projected_collapsed - &production_collapsed)
        .iter()
        .fold(0.0_f64, |m, d| m.max(d.abs()));
    assert!(
        max_gap < 1e-12,
        "top_k==K collapse-aware reconstruction must equal the non-projected \
         collapsed fitted() (incl. the linear-collapsed slot); max gap {max_gap:e}"
    );

    // And it must DIFFER from the curved-only assembly — otherwise the collapse
    // is a silent no-op and the test would pass vacuously.
    let projected_curved = term
        .reconstruct_from_assignments(full_assignments.view(), false)
        .expect("curved projected reconstruction assembles");
    let curved_gap = (&projected_collapsed - &projected_curved)
        .iter()
        .fold(0.0_f64, |m, d| m.max(d.abs()));
    assert!(
        curved_gap > 1e-9,
        "the collapsed slot must change the reconstruction vs the curved decoder \
         (collapse engaged); max gap {curved_gap:e}"
    );
}

/// #1228 — an OOS term must reconstruct a hybrid-collapsed `d = 1` slot by the
/// trained dictionary's straight sub-model when those images are attached via
/// [`SaeManifoldTerm::set_hybrid_linear_images`], matching the train-side
/// collapse policy instead of the original curved decoder.
#[test]
pub(crate) fn oos_linear_images_drive_collapsed_reconstruction() {
    let (mut term, _t, rho) = small_two_atom_periodic_term();
    for basis_row in 1..term.atoms[0].decoder_coefficients.nrows() {
        for out_col in 0..term.atoms[0].decoder_coefficients.ncols() {
            term.atoms[0].decoder_coefficients[[basis_row, out_col]] = 0.0;
        }
    }
    // A PHYSICAL target: the term's curved reconstruction plus a tiny smooth
    // residual, so the term does NOT reconstruct it to machine zero. This is
    // REQUIRED for the hybrid split to run at all. `compute_hybrid_split_report`
    // derives the rank-charge noise floor as `phi_hat = ||target - full||^2/(n*p)`;
    // with `target == full` (the exact self-reconstruction this fixture used to
    // pass) that floor is EXACTLY 0, at which `build_atom_candidates` cannot price
    // the evidence and refuses every atom as `Unadjudicable` (#2362) -> an empty
    // report. A real fit never reconstructs its target to zero. The perturbation
    // is orders of magnitude below either atom's own contribution, so the
    // straight-vs-curved discrimination below is unchanged; it only lifts phi_hat
    // off the degenerate zero.
    let full = term
        .try_fitted_for_rho(&rho)
        .expect("curved reconstruction assembles");
    let mut target = full.clone();
    for i in 0..target.nrows() {
        for j in 0..target.ncols() {
            target[[i, j]] += 1.0e-3 * (0.7 * (i as f64 + 1.0) + 1.3 * (j as f64 + 1.0)).sin();
        }
    }
    let report = term
        .compute_hybrid_split_report(&rho, Some(target.view()))
        .expect("hybrid split report computes")
        .expect("eligible d=1 atoms present a report");

    // Install the report so `fitted()` reconstructs the verdict-linear slot by its
    // straight sub-model (the train-side collapsed reconstruction).
    term.hybrid_split_report = Some(report);

    // #1228 WITNESS. The straightened atom is a CONSTANT (periodic basis row 0 is
    // the DC term), so its fitted linear image equals its own curve and collapsing
    // it changes nothing — the train-vs-OOS threading could not be observed.
    // Install a genuinely SLOPED straight image into the collapsed slot: still a
    // line (zero turning — a legitimate linear tail, NOT the EV-losing
    // over-collapse the gate prevents), but now it differs from the constant curve
    // by a real, measurable amount, so the train-side collapse is non-trivial and
    // the OOS reproduction below genuinely exercises the image threading.
    const WITNESS_SLOPE: f64 = 0.4;
    {
        let report = term.hybrid_split_report.as_mut().unwrap();
        let img = report
            .verdicts
            .iter_mut()
            .find_map(|v| v.linear_image.as_mut())
            .expect("the collapsed slot must carry a linear image to install a witness into");
        for slope in img.b1.iter_mut() {
            *slope += WITNESS_SLOPE;
        }
    }

    // Harvest the trained (witness-sloped) linear images to thread to a fresh OOS
    // term that knows the decoder but not the in-fit report, then drop the report.
    let images: Vec<_> = term
        .hybrid_split_report
        .as_ref()
        .unwrap()
        .verdicts
        .iter()
        .filter_map(|v| v.linear_image.clone())
        .collect();
    assert!(
        !images.is_empty(),
        "the straight slot must yield at least one linear image to thread to OOS"
    );
    let collapsed_with_report = term.fitted();
    term.hybrid_split_report = None;

    // Without images attached, the fresh term reconstructs all-curved.
    let curved = term.fitted();
    assert!(
        (&curved - &collapsed_with_report)
            .iter()
            .any(|d| d.abs() > 1e-9),
        "with no images attached the OOS reconstruction must be the curved one"
    );

    // Attaching the trained images restores the collapsed reconstruction exactly.
    term.set_hybrid_linear_images(images)
        .expect("valid linear images attach");
    let collapsed_oos = term.fitted();
    let gap = (&collapsed_oos - &collapsed_with_report)
        .iter()
        .fold(0.0_f64, |m, d| m.max(d.abs()));
    assert!(
        gap < 1e-12,
        "attached OOS linear images must reproduce the train-side collapsed \
         reconstruction; max gap {gap:e}"
    );
}

/// Shared builder for the Jeffreys barrier tests: a K=2
/// periodic term over `n` rows with explicit single-row decoders and a routing
/// where BOTH atoms carry non-negligible mass on every row (so the pair
/// co-fires everywhere, every gate sits far above the relative-mass floor, and
/// the truncated-support energies equal the plain full sums — the reference
/// formulas below are exact). Single-row decoders make the rank-aware overlap
/// `o_01` exactly the squared cosine of the two direction vectors.
fn jeffreys_two_atom_term(n: usize, dec0: [f64; 3], dec1: [f64; 3]) -> SaeManifoldTerm {
    let coords0 = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| (i as f64 * 0.618_034).fract());
    let coords1 = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| (i as f64 * 0.414_214).fract());
    let (phi0, jet0) = periodic_basis(&coords0);
    let (phi1, jet1) = periodic_basis(&coords1);
    // Bounded, row-varying logits: both softmax gates stay within a small
    // factor of each other (far above the 1e-3 relative-mass floor), while the
    // variation keeps the coactivation cosine q strictly inside (0, 1).
    let logits = Array2::<f64>::from_shape_fn((n, 2), |(i, j)| {
        if j == 0 {
            0.4 * (i as f64 * 0.7).sin()
        } else {
            0.3 * (i as f64 * 1.1).cos()
        }
    });
    let row_decoder = |r: [f64; 3]| {
        let mut d = Array2::<f64>::zeros((3, 3));
        d[[0, 0]] = r[0];
        d[[0, 1]] = r[1];
        d[[0, 2]] = r[2];
        d
    };
    let make = |name: &str, phi: Array2<f64>, jet: Array3<f64>, decoder: Array2<f64>| {
        SaeManifoldAtom::new_with_provided_function_gram(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(TestPeriodicEvaluator))
    };
    let atom0 = make("occ0", phi0, jet0, row_decoder(dec0));
    let atom1 = make("occ1", phi1, jet1, row_decoder(dec1));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0, coords1],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::softmax(0.8),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap()
}

/// The production separation barrier's SMOOTH spectral floor
/// `m(λ) = ε·softplus((λ + ε)/ε)` ([`SaeManifoldTerm::barrier_spectral_m`]),
/// restated here in closed form so the references below price the same object
/// the production value does.
///
/// This floor — NOT the hard `m(λ) = λ + ε` these references used to hard-code —
/// is applied to every eigenvalue of the component matrix `F`. The two agree only
/// in the asymptotic regime `(λ + ε)/ε ≫ 1`, and this K=2 softmax fixture is far
/// from it: two softmax gates over n = 48/64 rows give `N_eff ≈ 12/16`, hence
/// `ε_C = 2·√(2/N_eff) ≈ 0.80/0.69`, so `(λ + ε)/ε` sits at ≈1–3 where softplus is
/// still strongly curved. Pricing these fixtures with the hard floor overstated the
/// barrier by ≈2.03× (n=48, o=0.98) and ≈1.85× (n=64, o=0.93) — the historical
/// references predate the smooth floor and were never re-derived.
///
/// What stays independent is the part that actually tests the assembly: the
/// component's `F = [[1, r], [r, 1]]` has eigenvalues `1 ± r` in CLOSED FORM, and
/// `r = q·o`, `ε_C` are rebuilt here from the realized routing rather than read
/// back from production. Only the scalar floor is a shared definition.
fn barrier_spectral_m_reference(lam: f64, eps: f64) -> f64 {
    let x = (lam + eps) / eps;
    if x >= 30.0 {
        lam + eps
    } else if x <= -30.0 {
        eps * x.exp()
    } else {
        eps * x.exp().ln_1p()
    }
}

/// Independently reconstruct the two-atom Jeffreys value from the realized
/// routing. Returns `(value, q, eps)`, where `q` is the coactivation cosine and
/// `eps` is the sampling-resolution shift.
fn two_atom_jeffreys_reference(term: &SaeManifoldTerm, overlap: f64) -> (f64, f64, f64) {
    let gates = term.assignment.assignments();
    let (mut cross, mut e0, mut e1) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..gates.nrows() {
        let a0 = gates[[i, 0]];
        let a1 = gates[[i, 1]];
        cross += a0 * a1;
        e0 += a0 * a0;
        e1 += a1 * a1;
    }
    let q = cross / (e0 * e1).sqrt();
    let eps = 2.0 * (2.0 / e0.min(e1)).sqrt();
    let r = q * overlap;
    // `F = [[1, r], [r, 1]]` ⇒ closed-form eigenvalues `1 ± r`; the production
    // value is `−½·Σ_i [ln m(λ_i) − ln m(1)]` under the smooth spectral floor.
    let value = -0.5
        * (barrier_spectral_m_reference(1.0 + r, eps).ln()
            + barrier_spectral_m_reference(1.0 - r, eps).ln()
            - 2.0 * barrier_spectral_m_reference(1.0, eps).ln());
    (value, q, eps)
}

/// The unscaled Jeffreys barrier matches its closed-form value and analytic
/// decoder gradient at a near-collapsed pair (`o = c² = 0.98`). The component
/// value is `−½ log det(F+εI)` relative to the shifted identity; occupancy
/// controls `q` and the resolution shift `ε`, but never multiplies the log-det.
#[test]
fn unscaled_jeffreys_value_and_gradient_match_closed_form() {
    let c2 = 0.98_f64;
    let cos = c2.sqrt();
    let sin = (1.0 - c2).sqrt();
    let n = 48_usize;
    let dec0 = [1.0, 0.0, 0.0];
    let dec1 = [cos, sin, 0.0];
    let term = jeffreys_two_atom_term(n, dec0, dec1);

    let (expected, q, eps) = two_atom_jeffreys_reference(&term, c2);
    assert!(
        q > 0.0 && q < 1.0,
        "fixture: coactivation must be strictly interior, got q={q}"
    );
    let value = term.separation_barrier_value(1.0);
    assert!(
        value > 0.0,
        "near-collapsed co-firing pair must be penalized, got {value}"
    );
    assert!(
        (value - expected).abs() <= expected.abs() * 1e-9,
        "Jeffreys value must equal the unscaled closed form \
         −½·log det(F+ε_C·I): expected={expected:.12e} got={value:.12e} \
         (q={q:.6}, ε_C={eps:.6})"
    );

    // FD contract on every decoder coefficient of both atoms.
    let (_v, grad) = term.separation_barrier_value_and_grad_for_test(1.0);
    let offsets = term.beta_offsets();
    let p = term.output_dim();
    let h = 1.0e-6;
    let mut max_rel = 0.0_f64;
    for atom in 0..2 {
        let m = term.atoms[atom].decoder_coefficients.nrows();
        for a in 0..m {
            for o in 0..p {
                let mut plus = jeffreys_two_atom_term(n, dec0, dec1);
                plus.atoms[atom].decoder_coefficients[[a, o]] += h;
                let mut minus = jeffreys_two_atom_term(n, dec0, dec1);
                minus.atoms[atom].decoder_coefficients[[a, o]] -= h;
                let fd = (plus.separation_barrier_value(1.0) - minus.separation_barrier_value(1.0))
                    / (2.0 * h);
                let analytic = grad[offsets[atom] + a * p + o];
                let rel = (fd - analytic).abs() / (1.0 + fd.abs().max(analytic.abs()));
                max_rel = max_rel.max(rel);
            }
        }
    }
    assert!(
        max_rel < 1.0e-5,
        "unscaled analytic ∂P/∂B must match FD of the unscaled value at \
         the near-collapsed pair: max rel err {max_rel:.3e}"
    );
}

/// Sample size factors out of a fixed-dimensional Jeffreys volume ratio. For
/// `I_N = N_eff(F+εI)`, both the fitted Fisher determinant and its identity
/// reference acquire the same `s log N_eff` term, which cancels. This test
/// evaluates that total-information expression over six orders of magnitude and
/// pins it to the production component value.
#[test]
fn jeffreys_total_information_factorization_is_sample_size_invariant() {
    let overlap = 0.93_f64;
    let term = jeffreys_two_atom_term(
        64,
        [1.0, 0.0, 0.0],
        [overlap.sqrt(), (1.0 - overlap).sqrt(), 0.0],
    );
    let (expected, q, eps) = two_atom_jeffreys_reference(&term, overlap);
    let production = term.separation_barrier_value(1.0);
    assert!((production - expected).abs() <= 1.0e-12);

    let r = q * overlap;
    for sample_mass in [1.0_f64, 7.0, 1.0e3, 1.0e6] {
        let total_information_value = -0.5
            * ((sample_mass * barrier_spectral_m_reference(1.0 + r, eps)).ln()
                + (sample_mass * barrier_spectral_m_reference(1.0 - r, eps)).ln()
                - 2.0 * (sample_mass * barrier_spectral_m_reference(1.0, eps)).ln());
        assert!(
            (total_information_value - production).abs() <= 2.0e-12,
            "the common s·log(N_eff) factor must cancel: N_eff={sample_mass:e}, \
             total-information value={total_information_value:.12e}, \
             production={production:.12e}"
        );
    }
}

/// SEAM GATE for the unscaled Jeffreys barrier on the ASSEMBLED path: the
/// arrow-Schur `gb` (which carries the barrier force `α_e = −G·q` plus the
/// frozen-gate repulsion, data, and smoothness) must
/// match central finite differences of `penalized_objective_total` — the ONE
/// objective the production line search evaluates, and the ONLY value consumer
/// of `separation_barrier_value` — under the production snapshot/restore
/// discipline (frozen gates reinstalled on every FD trial, exactly like the
/// optimizer, which restores decoders but never re-derives the gates).
///
/// This is the K≥2, barrier-ENGAGED twin of the K=1
/// `sae_d1_assembled_gradient_matches_loss_central_fd` (where every barrier path
/// early-returns at `k_atoms < 2`). Green means the unscaled value and gradient
/// price one object end-to-end through the real assembly.
#[test]
fn unscaled_jeffreys_assembled_gradient_matches_penalized_objective_fd() {
    let n = 32_usize;
    let c2 = 0.8_f64;
    let dec0 = [1.0, 0.0, 0.0];
    let dec1 = [c2.sqrt(), (1.0 - c2).sqrt(), 0.0];
    let term0 = jeffreys_two_atom_term(n, dec0, dec1);
    let p = term0.output_dim();
    // Deterministic non-trivial target so the data gradient is exercised too.
    let target = Array2::<f64>::from_shape_fn((n, p), |(i, j)| {
        0.21 * (0.31 * (i as f64 + 1.0) + 0.47 * (j as f64 + 1.0)).sin()
            - 0.13 * (0.19 * (i as f64 + 1.0) * (j as f64 + 1.0)).cos()
    });
    let rho = SaeManifoldRho::new(
        -2.0,
        -2.0,
        vec![Array1::from_vec(vec![-2.0]), Array1::from_vec(vec![-2.0])],
    );

    // Production lagged-diffusivity discipline: freeze the gates once at the
    // base state; every FD trial reinstalls the SAME frozen gates (clone resets
    // them — they are transient per-assembly state).
    let mut base = term0.clone();
    base.refresh_decoder_repulsion_gate();
    base.refresh_barrier_coactivation_gate();
    base.refresh_amplitude_barrier_gate(); // #2343 — third frozen per-assembly gate
    let base = base;
    let reinstall_frozen_gates = |t: &mut SaeManifoldTerm| {
        t.decoder_repulsion_gate = base.decoder_repulsion_gate.clone();
        t.barrier_coactivation_gate = base.barrier_coactivation_gate.clone();
        t.amplitude_barrier_gate = base.amplitude_barrier_gate;
    };

    // The barrier must be live, otherwise the seam check would be vacuous.
    let barrier_value = base.separation_barrier_value(1.0);
    assert!(
        barrier_value > 1.0e-4,
        "fixture: the unscaled barrier must be live for aligned, co-firing \
         atoms; got {barrier_value}"
    );

    // Assemble on a clone (assembly re-freezes the gates from the identical
    // state, so they equal `base`'s), and FD the production objective.
    let mut assembled = base.clone();
    let sys = assembled
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("K=2 assembly succeeds");
    let beta = base.flatten_beta();
    assert_eq!(sys.gb.len(), beta.len());
    let h = 1.0e-6;
    let mut worst_rel = 0.0_f64;
    let mut worst_idx = 0_usize;
    for idx in 0..beta.len() {
        let mut beta_plus = beta.clone();
        beta_plus[idx] += h;
        let mut plus = base.clone();
        reinstall_frozen_gates(&mut plus);
        plus.set_flat_beta(beta_plus.view()).expect("set beta plus");
        let obj_plus = plus
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("objective at plus");

        let mut beta_minus = beta.clone();
        beta_minus[idx] -= h;
        let mut minus = base.clone();
        reinstall_frozen_gates(&mut minus);
        minus
            .set_flat_beta(beta_minus.view())
            .expect("set beta minus");
        let obj_minus = minus
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("objective at minus");

        let fd = (obj_plus - obj_minus) / (2.0 * h);
        let analytic = sys.gb[idx];
        let rel = (fd - analytic).abs() / fd.abs().max(analytic.abs()).max(1.0e-9);
        if rel > worst_rel {
            worst_rel = rel;
            worst_idx = idx;
        }
    }
    assert!(
        worst_rel < 5.0e-5,
        "assembled gb must be the exact gradient of the line-search objective \
         (unscaled Jeffreys barrier included on both sides): worst rel err \
         {worst_rel:.3e} at beta index {worst_idx}"
    );
}

/// #2343 — IN-SITU acceptance: at the decoder-collapse point the interior
/// AMPLITUDE barrier is the sole meaningful radial (amplitude) force, and the
/// decoder repulsion is inert against it. Two clauses pin the root cause:
///
///  (1) the repulsion's OWN in-situ radial force — measured through the exact
///      assembled penalty (`live_decoder_repulsion_penalty`, frozen gate + LIVE
///      norms) — is negligible relative to the amplitude barrier. With the
///      normalizer live the coherence is homogeneous degree 0 in each decoder
///      radius, so its radial gradient vanishes by Euler's theorem
///      (`Σ_{a,o} B·∂P/∂B ≡ 0`); the pre-fix FROZEN normalizer instead left the
///      term degree 2 in the live radius with a stale `1/‖B‖⁴` amplification that
///      produced a `+2.57e7` INWARD radial force — `5×` the barrier's outward one.
///
///  (2) the NET radial β-gradient of the collapsing atom therefore equals the
///      barrier's ANALYTIC `∂P_A/∂B = g_coef·B` (the only term that prices
///      amplitude) to within the small common-mode residual of the other live
///      terms (data-fit / separation), `~1e-8` relative here.
///
/// Scale note: being INSIDE the barrier's turn-on radius forces `u = ‖B‖²_F ≲ f =
/// 1e-12·max_k‖B_k‖²_F`, i.e. `u ~ 1e-13`. In that regime the exact Euler
/// cancellation `E − (E/N)·N` is amplified by `κ ∝ 1/N ~ 1e15`, so finite double
/// precision caps the repulsion's residual radial force at `~1e-7` ABSOLUTE — still
/// `~1e-14` of the barrier (`~2e7`) and physically inert. The MACHINE-precision
/// degree-0 property at O(1) decoder scale is pinned separately by the green
/// `decoder_incoherence_repulsion_is_radially_free_euler` gam-terms unit test; the
/// in-situ bounds below are relative to the barrier, the scale-invariant quantity
/// the pre-fix bug violated (repulsion/barrier `~5`, wrong sign).
#[test]
fn repulsion_is_radially_inert_net_radial_is_analytic_barrier_2343() {
    use gam_terms::analytic_penalties::AnalyticPenalty;

    let (term0, target0, _rho) = small_two_atom_periodic_term();
    let mut term = term0.clone();
    let p = term.output_dim();

    // Atom 1's decoder = ε · atom 0's decoder: EXACTLY collinear (output-Gram
    // cosine² = 1, the maximal-coherence worst case for a radial leak) and deep
    // inside the amplitude barrier's turn-on radius (‖B_1‖²_F ≪ f).
    let eps = 1.0e-7_f64;
    let b0 = term.atoms[0].decoder_coefficients.clone();
    term.atoms[1].decoder_coefficients = &b0 * eps;

    // Zero target so the data-fit gradient on the ≈0 decoder is itself O(ε): the
    // dominant radial force on atom 1's block is the collapse-prevention stack.
    let target = Array2::<f64>::zeros(target0.raw_dim());
    let rho = SaeManifoldRho::new((1.0e-4_f64).ln(), (1.0e-4_f64).ln(), vec![array![0.0], array![0.0]]);
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("assembly must succeed at the collapse point");

    // The repulsion must actually be live on this pair, or both clauses are
    // vacuous (this is exactly the configuration the pre-fix inward force hit).
    let gate = term
        .decoder_repulsion_gate
        .clone()
        .expect("#2343: the decoder repulsion gate must be ENGAGED on the collinear pair");
    assert!(
        gate.iter().any(|&(j, k, w)| (j, k) == (0, 1) && w > 0.0),
        "#2343: pair (0,1) must carry positive repulsion weight; gate = {gate:?}"
    );

    let offsets = term.beta_offsets();
    let off1 = offsets[1];
    let b1 = term.atoms[1].decoder_coefficients.clone();
    let u: f64 = b1.iter().map(|v| v * v).sum();
    let s = u.sqrt();
    assert!(s > 0.0, "collapsing atom must retain a radial direction");
    assert_eq!(b1.ncols(), p, "decoder block must be M×p_out");
    let dir: Vec<f64> = b1.iter().map(|v| v / s).collect();

    // Analytic amplitude barrier at the SAME frozen turn-on radius — the scale
    // both clauses measure against (it is what prices amplitude here).
    let norm_sq: Vec<f64> = term
        .atoms
        .iter()
        .map(|atom| atom.decoder_coefficients.iter().map(|v| v * v).sum::<f64>())
        .collect();
    let f = SaeManifoldTerm::barrier_norm_floor_sq(&norm_sq);
    let mu = SAE_AMPLITUDE_BARRIER_STRENGTH;
    assert!(u < f, "atom 1 must sit inside the barrier turn-on radius: u={u:e} f={f:e}");
    let g_coef = -2.0 * mu * f / (u * (u + f));
    let expected = g_coef * s; // barrier radial force (outward, < 0)
    assert!(expected < 0.0, "barrier radial force must be outward");

    // ---- Clause (1): the repulsion's own in-situ radial force ≪ barrier. ----
    let rep = term
        .live_decoder_repulsion_penalty()
        .expect("#2343: live repulsion penalty must exist when the gate is engaged");
    let beta = term.flatten_beta();
    let rep_grad = rep.grad_target(beta.view(), Array1::<f64>::zeros(0).view());
    let rep_radial: f64 = (0..b1.len()).map(|i| rep_grad[off1 + i] * dir[i]).sum();
    let rep_rel = rep_radial.abs() / expected.abs();
    assert!(
        rep_rel <= 1.0e-9,
        "#2343 clause (1): the live-normalized repulsion must be radially INERT — its \
         own in-situ radial β-gradient on the collapsing atom must be negligible \
         against the amplitude barrier (degree-0 homogeneity, Euler). Got \
         {rep_radial:.3e} vs barrier {expected:.3e} (relative {rep_rel:e}); the pre-fix \
         frozen normalizer made this +2.57e7, i.e. ~5× the barrier."
    );

    // ---- Clause (2): the NET radial equals the analytic amplitude barrier. ----
    let radial: f64 = (0..b1.len()).map(|i| sys.gb[off1 + i] * dir[i]).sum();
    assert!(
        radial < 0.0,
        "#2343 clause (2): the net radial force must be OUTWARD (negative): \
         radial={radial:e} expected={expected:e}"
    );
    let rel = (radial - expected).abs() / expected.abs();
    assert!(
        rel <= 1.0e-6,
        "#2343 clause (2): with the repulsion radially inert (clause 1), the net radial \
         β-gradient must equal the amplitude barrier's analytic g_coef·‖B_1‖ alone, to \
         within the ~1e-8 common-mode residual of the other live terms (data-fit / \
         separation): measured {radial:.12e} vs analytic {expected:.12e} (relative gap \
         {rel:e}). The pre-fix frozen-normalizer repulsion flipped this to +2.07e7 \
         (INWARD) — an O(1) sign change, far above this bound."
    );
}


// #2253 co-collapse instrumentation (diagnostic; zz_measure). Sweep the 2-atom
// alignment c2 toward collapse and report the separation-barrier restoring force
// (grad norm) + value. If the force PLATEAUS at O(1) as c2->1, the tiny fixture
// is in the WEAK-barrier (large eps_C = 2*sqrt(s/N_eff), small-N_eff) regime, not
// the softplus curvature-cap (small-eps) regime.
#[test]
fn zz_measure_separation_force_vs_c2_2253() {
    for &c2 in &[0.5_f64, 0.9, 0.99, 0.999, 0.9999, 0.999999] {
        let term = aligned_two_atom_term_with_c2(c2);
        let (v, g) = term.separation_barrier_value_and_grad_for_test(1.0);
        let fnorm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        eprintln!("SEPFORCE c2={c2:.6} value={v:.6e} force_gradnorm={fnorm:.6e}");
    }
}


// #2253 co-collapse — confirm the gate-inside defect on the REAL failing
// fixtures: report the co-firing weight q, the decoder coherence o=c2, and the
// 2-atom collapsing eigenvalue lambda_min=1-q*o, both as constructed and with
// the decoders forcibly ALIGNED (o->1). If q<1 with o->1, lambda_min saturates
// at 1-q>0 (bounded away from the pole) — regime-1 confirmed on real fixtures.
#[test]
fn zz_measure_real_fixture_barrier_q_2253() {
    use crate::manifold::tests::small_two_atom_periodic_term;
    use crate::manifold::tests_recovery_split_780::gamma_fd_tiny_fixture;
    let report = |tag: &str, term: &SaeManifoldTerm| {
        let (pairs, _neff) = term.barrier_coactivation_pairs();
        for (j, k, q) in &pairs {
            let o = term.decoder_gram_cosine_sq(*j, *k);
            eprintln!(
                "REALQ {tag} pair=({j},{k}) q={q:.6e} o_c2={o:.6e} lam_min=1-q*o={:.6e}",
                1.0 - q * o
            );
        }
        if pairs.is_empty() {
            eprintln!("REALQ {tag} NO co-firing pairs");
        }
    };
    // recompute config: gamma_fd_tiny + ordered-Beta--Bernoulli gate + sparse 0.5.
    let (mut term, _t, _r) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, true);
    term.refresh_barrier_coactivation_gate();
    report("recompute_asbuilt", &term);
    let b0 = term.atoms[0].decoder_coefficients.clone();
    if term.atoms[1].decoder_coefficients.dim() == b0.dim() {
        term.atoms[1].decoder_coefficients = b0.clone();
    }
    term.refresh_barrier_coactivation_gate();
    report("recompute_aligned", &term);
    // hutchinson: small_two_atom_periodic_term.
    let (mut h, _t2, _r2) = small_two_atom_periodic_term();
    h.refresh_barrier_coactivation_gate();
    report("hutchinson_asbuilt", &h);
    let hb0 = h.atoms[0].decoder_coefficients.clone();
    if h.atoms[1].decoder_coefficients.dim() == hb0.dim() {
        h.atoms[1].decoder_coefficients = hb0.clone();
    }
    h.refresh_barrier_coactivation_gate();
    report("hutchinson_aligned", &h);
}

// #2253 Q2/Q3 — under-power vs solver: SVD the tiny-fixture TARGETS to see
// whether the data supports rank-2 (K=2) at all. If sigma2/sigma1 is tiny the
// reseeder verdict "cannot anchor K=2" is CORRECT (fixture under-power), not a
// barrier or solver failure.
#[test]
fn zz_measure_tiny_fixture_target_rank_2253() {
    use gam_linalg::faer_ndarray::FaerSvd;
    use crate::manifold::tests::small_two_atom_periodic_term;
    use crate::manifold::tests_recovery_split_780::gamma_fd_tiny_fixture;
    let svd_report = |tag: &str, target: &Array2<f64>| {
        let (_u, sv, _vt) = target.svd(false, false).expect("svd");
        let s: Vec<f64> = sv.iter().copied().collect();
        let s1 = s.first().copied().unwrap_or(0.0);
        let s2 = s.get(1).copied().unwrap_or(0.0);
        let ratio = if s1 > 0.0 { s2 / s1 } else { 0.0 };
        let sfmt: Vec<String> = s.iter().map(|x| format!("{x:.4e}")).collect();
        eprintln!(
            "TARGETRANK {tag} dim={:?} sigmas={:?} sigma2_over_sigma1={ratio:.6e}",
            target.dim(),
            sfmt
        );
    };
    let (_t, tgt_r, _r) = gamma_fd_tiny_fixture();
    svd_report("recompute_gamma_fd_tiny", &tgt_r);
    let (_t2, tgt_h, _r2) = small_two_atom_periodic_term();
    svd_report("hutchinson_small_two_atom", &tgt_h);
}

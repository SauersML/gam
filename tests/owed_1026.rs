//! Owed-work regression gate for #1026 (SAE reconstruction parity / scale-K).
//!
//! #1026's real inner-convergence wall (per the issue thread + the
//! `project_1026_cocollapse_inner_convergence` disposition) is the K>1
//! co-collapse: two atoms drift onto the SAME decoder direction, a per-row
//! `H_tt` block goes near-singular, the reduced β-Schur over-subtracts and
//! goes non-PD, the inner joint-Newton stalls, and EV cliffs to 0. The landed
//! convergence cure (commit `60db8ba3f`) is the collinearity-gated decoder
//! REPULSION: a PSD `DecoderIncoherencePenalty` whose per-pair weight is a C1
//! smoothstep that is EXACTLY 0 below the collinearity gate and ramps up as two
//! decoders become collinear, so the assembled β-Hessian gains strictly
//! positive curvature in the collapse direction — atoms stay distinct mid-solve,
//! the per-row `H_tt` never collapses, the Schur stays PD, the inner Newton
//! converges.
//!
//! This integration test pins the load-bearing property at assembly scope:
//! when two atoms' decoders are COLLINEAR the assembled penalty operator carries
//! strictly more curvature in the collapse direction than when they are
//! ORTHOGONAL, and the gap is at least the documented repulsion strength. Each
//! atom's OWN decoder Gram (hence its smoothness-penalty curvature) is held
//! identical across the two cases by writing the same basis-row pattern to a
//! different output channel, so the curvature DIFFERENCE in the atom's own block
//! isolates the repulsion alone — no data-fit / smoothness confound. A
//! regression that silently disengaged the repulsion (re-introducing the K>1
//! collapse) would drive that gap to ~0 and fail here.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use std::sync::Arc;

use ndarray::{Array1, Array2, array};

use gam::solver::arrow_schur::{
    ArrowSchurError, ArrowSchurSystem, ArrowSolveOptions, BetaPenaltyOp,
    solve_arrow_newton_step_with_options,
};
use gam::terms::latent::LatentManifold;
use gam::terms::{
    ArdSharing, AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
    SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};

const M: usize = 3; // periodic basis: [const, sin, cos]
const P: usize = 3; // output channels

/// Build a two-atom periodic SAE term over a fixed set of circle coordinates,
/// with caller-supplied decoders. Latent dim 1, IBP-MAP routing (the production
/// real-data path in the #1026 thread).
fn build_two_atom_term(dec0: Array2<f64>, dec1: Array2<f64>) -> SaeManifoldTerm {
    let coords0 = array![[0.05], [0.22], [0.55], [0.81], [0.34], [0.66], [0.12], [0.90]];
    let coords1 = array![[0.15], [0.31], [0.64], [0.92], [0.47], [0.09], [0.73], [0.40]];
    let eval = PeriodicHarmonicEvaluator::new(M).unwrap();
    let (phi0, jet0) = eval.evaluate(coords0.view()).unwrap();
    let (phi1, jet1) = eval.evaluate(coords1.view()).unwrap();
    let make = |name: &str, phi: Array2<f64>, jet, dec: Array2<f64>| {
        SaeManifoldAtom::new(
            name,
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            dec,
            Array2::<f64>::eye(M),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(M).unwrap()))
    };
    let atom0 = make("circle_0", phi0, jet0, dec0);
    let atom1 = make("circle_1", phi1, jet1, dec1);
    let n = coords0.nrows();
    // Mild, non-degenerate logits so both atoms are routed-on (gate map active).
    let logits = Array2::from_shape_fn((n, 2), |(i, k)| {
        0.3 + 0.1 * (i as f64) - 0.05 * (k as f64)
    });
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0, coords1],
        vec![
            LatentManifold::Circle { period: 1.0 },
            LatentManifold::Circle { period: 1.0 },
        ],
        AssignmentMode::ibp_map(0.5, 1.0, false),
    )
    .unwrap();
    SaeManifoldTerm::new(vec![atom0, atom1], assignment).unwrap()
}

/// `vᵀ P v` for the assembled β-penalty operator `P` and probe direction `v`.
fn penalty_quadratic_form(op: &Arc<dyn BetaPenaltyOp>, v: &[f64]) -> f64 {
    let mut pv = vec![0.0_f64; v.len()];
    op.matvec(v, &mut pv);
    v.iter().zip(pv.iter()).map(|(a, b)| a * b).sum()
}

/// #1026: the collinearity-gated decoder repulsion (the landed K>1 co-collapse
/// convergence cure, commit `60db8ba3f`) must load strictly positive curvature
/// into the assembled β-penalty operator along the collapse direction when two
/// atoms' decoders are collinear, and be a strict no-op when they are
/// orthogonal. The probe direction is supported ONLY on atom 1's decoder block,
/// and atom 1's own decoder pattern (hence its smoothness-penalty curvature) is
/// identical in both cases, so the curvature DIFFERENCE isolates the repulsion.
#[test]
fn decoder_repulsion_conditions_collapse_direction_1026() {
    // Atom 1's sin/cos rows define the collapse-probe direction in its own
    // decoder block. Same UNIT pattern in both cases; only the output channel it
    // writes differs (channel 0 ⇒ collinear with atom 0; channel 1 ⇒ orthogonal).
    // The β layout concatenates per-atom (M×P) row-major decoder blocks, so atom
    // 1's block starts at offset M*P.
    let beta_dim = 2 * M * P;
    let atom1_block_start = M * P;

    // --- COLLINEAR: both atoms write output channel 0 with the same sin/cos.
    let mut dec0_col = Array2::<f64>::zeros((M, P));
    dec0_col[[1, 0]] = 1.0; // sin -> ch0
    dec0_col[[2, 0]] = 1.0; // cos -> ch0
    let mut dec1_col = Array2::<f64>::zeros((M, P));
    dec1_col[[1, 0]] = 1.0;
    dec1_col[[2, 0]] = 1.0;

    // --- ORTHOGONAL: atom 1 writes channel 1 instead (same pattern, disjoint
    // span ⇒ cross-Gram 0 ⇒ repulsion gate exactly off). Atom 1's own decoder
    // Gram BᵀB is IDENTICAL to the collinear case (a permutation of output
    // channels), so its smoothness curvature is unchanged.
    let dec0_orth = dec0_col.clone();
    let mut dec1_orth = Array2::<f64>::zeros((M, P));
    dec1_orth[[1, 1]] = 1.0;
    dec1_orth[[2, 1]] = 1.0;

    // Probe direction: atom 1's sin & cos rows moving together (the collapse
    // motion the repulsion resists). Placed at the SAME block positions in both
    // assemblies (atom 1's block), targeting the channel each case actually
    // writes, so the smoothness self-curvature along the probe matches.
    // Decoder block is row-major (M basis rows × P channels): basis row `r`,
    // channel `c` sits at `r*P + c`. Rows 1 (sin) and 2 (cos) are the circle
    // plane; the collapse probe excites both together.
    let sin_row = P;
    let cos_row = 2 * P;
    let mut v_col = vec![0.0_f64; beta_dim];
    v_col[atom1_block_start + sin_row] = 1.0; // sin -> ch0
    v_col[atom1_block_start + cos_row] = 1.0; // cos -> ch0
    let mut v_orth = vec![0.0_f64; beta_dim];
    v_orth[atom1_block_start + sin_row + 1] = 1.0; // sin -> ch1
    v_orth[atom1_block_start + cos_row + 1] = 1.0; // cos -> ch1

    let target = Array2::<f64>::zeros((8, P));
    // Tiny smoothness/sparsity so the SAME small smoothness curvature is present
    // in both cases (it cancels in the isolated atom-1-block difference) and the
    // repulsion is the only collinearity-dependent term.
    let rho = SaeManifoldRho::new(
        (1.0e-3_f64).ln(),
        (1.0e-3_f64).ln(),
        vec![Array1::<f64>::zeros(0); 2],
    );

    let mut term_col = build_two_atom_term(dec0_col, dec1_col);
    let sys_col = term_col
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("collinear assembly must succeed");
    let mut term_orth = build_two_atom_term(dec0_orth, dec1_orth);
    let sys_orth = term_orth
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("orthogonal assembly must succeed");

    let op_col = sys_col
        .penalty_op
        .as_ref()
        .expect("collinear assembly must install a β-penalty operator");
    let op_orth = sys_orth
        .penalty_op
        .as_ref()
        .expect("orthogonal assembly must install a β-penalty operator");

    let q_col = penalty_quadratic_form(op_col, &v_col);
    let q_orth = penalty_quadratic_form(op_orth, &v_orth);

    // The repulsion is PSD, so it can only ADD curvature in the collapse
    // direction; orthogonal decoders engage no repulsion. The atom-1-block
    // smoothness curvature is identical across the two cases by construction, so
    // the strictly positive gap is the repulsion alone. The per-pair weight is
    // SAE_DECODER_REPULSION_STRENGTH (1e-3) at full collinearity, and the
    // quadratic form on a 2-row unit probe is O(strength) — assert a margin well
    // clear of rounding but below the analytic value so the gate engaging is the
    // load-bearing assertion, not a brittle exact number.
    assert!(
        q_col.is_finite() && q_orth.is_finite(),
        "penalty quadratic forms must be finite: collinear={q_col}, orthogonal={q_orth}"
    );
    assert!(
        q_col > q_orth + 1.0e-6,
        "#1026 decoder repulsion must add curvature in the collapse direction \
         when decoders are collinear (the K>1 co-collapse convergence cure): \
         collinear vᵀPv={q_col:e} must exceed orthogonal vᵀPv={q_orth:e} by the \
         repulsion strength; a near-zero gap means the repulsion silently \
         disengaged and the co-collapse stall is back."
    );
}

/// #1026 companion: with two near-collinear atoms, assembling the Arrow-Schur
/// system must SUCCEED and produce a finite, well-formed β tier — i.e. the
/// repulsion-conditioned assembly does not itself introduce a non-finite or
/// degenerate operator on exactly the geometry that used to drive the reduced
/// Schur non-PD. This guards the assembly half of the convergence cure.
#[test]
fn collinear_two_atom_assembly_is_finite_and_well_formed_1026() {
    // Strongly collinear (identical decoders) — the worst-case collapse geometry.
    let mut dec = Array2::<f64>::zeros((M, P));
    dec[[1, 0]] = 1.0;
    dec[[2, 0]] = 1.0;
    let mut term = build_two_atom_term(dec.clone(), dec);
    let target = Array2::<f64>::zeros((8, P));
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(0); 2]);
    let sys = term
        .assemble_arrow_schur(target.view(), &rho, None)
        .expect("collinear assembly must succeed (no panic / no degenerate refusal)");

    let op = sys
        .penalty_op
        .as_ref()
        .expect("β-penalty operator must be installed for K=2");
    // The penalty operator's diagonal (Jacobi seed) must be finite and the
    // matvec must be finite on a unit probe — the conditioned β tier is usable
    // by the downstream Schur factor.
    let beta_dim = 2 * M * P;
    let mut diag = vec![0.0_f64; beta_dim];
    op.diagonal(&mut diag);
    assert!(
        diag.iter().all(|v| v.is_finite()),
        "penalty operator diagonal must be finite under collinear decoders"
    );
    assert!(
        sys.gb.iter().all(|v| v.is_finite()),
        "assembled β gradient must be finite under collinear decoders"
    );
}

/// Build a small arrow system whose REDUCED SCHUR is INDEFINITE — the exact
/// #1026 K>1 co-collapse signature. Two atoms (k=2, d=1) sit on near-singular
/// per-row latent Hessians `H_tt^(i) = [[h_tt]]` (PD and well-conditioned, so no
/// per-row ridge fires) with substantial cross-coupling `H_tβ`, so the
/// accumulated `Σ_i H_tβᵀ (H_tt)⁻¹ H_tβ` OVER-SUBTRACTS the small `H_ββ` into a
/// matrix with negative eigenvalues:
///   S = diag(h_bb, h_bb) − diag(c²/h_tt, c²/h_tt).
/// With `h_tt = 0.01`, `c = 1`, `h_bb = 1`: S = diag(1 − 100, 1 − 100) =
/// diag(−99, −99) ≺ 0. This is the collapsed geometry the reduced Schur
/// Cholesky refuses.
fn indefinite_collapsed_schur_system(h_tt: f64, c: f64, h_bb: f64) -> ArrowSchurSystem {
    let k = 2usize;
    let d = 1usize;
    let n = 2usize;
    let hbb = Array2::<f64>::zeros((k, k));
    let mut sys = ArrowSchurSystem::new_with_hbb(n, d, k, hbb);
    // H_ββ: small diagonal (will be over-subtracted).
    sys.hbb[[0, 0]] = h_bb;
    sys.hbb[[1, 1]] = h_bb;
    // Row 0 couples β0; row 1 couples β1. Each on a small PD H_tt.
    sys.rows[0].htt[[0, 0]] = h_tt;
    sys.rows[0].htbeta[[0, 0]] = c;
    sys.rows[0].gt[0] = 0.3;
    sys.rows[1].htt[[0, 0]] = h_tt;
    sys.rows[1].htbeta[[0, 1]] = c;
    sys.rows[1].gt[0] = -0.2;
    sys.gb[0] = 0.5;
    sys.gb[1] = -0.4;
    sys.refresh_row_hessian_fingerprint();
    sys
}

/// #1026 hardening — the spectral floor must preserve the EXACT Newton step in
/// the HEALTHY β subspace while conditioning only the collapsed direction. Build
/// a reduced Schur `S = diag(+5, −99)`: direction 0 is well-conditioned
/// (`h_bb=6`, `c=1`, `h_tt=1` ⇒ `S_00 = 6 − 1 = 5`), direction 1 is collapsed
/// (`h_bb=1`, `c=1`, `h_tt=0.01` ⇒ `S_11 = 1 − 100 = −99`). The Schur is diagonal
/// (the two atoms couple disjoint β columns), so the healthy block's Newton
/// solve is exactly `5·Δβ_0 = rhs_0` and the floor — which lifts only
/// eigenvalues below `floor·max|λ| = 1e-8·99 ≈ 1e-6`, leaving `5 ≫ 1e-6`
/// untouched — must return that EXACT `Δβ_0` while still producing a finite
/// `Δβ_1`. This pins the load-bearing property: damping is confined to the
/// collapsed subspace; the converged dictionary is unperturbed elsewhere.
#[test]
fn spectral_floor_preserves_healthy_subspace_under_mixed_collapse_1026() {
    let k = 2usize;
    let d = 1usize;
    let n = 2usize;
    let mut sys = ArrowSchurSystem::new_with_hbb(n, d, k, Array2::<f64>::zeros((k, k)));
    sys.hbb[[0, 0]] = 6.0; // healthy: S_00 = 6 - 1 = 5
    sys.hbb[[1, 1]] = 1.0; // collapsed: S_11 = 1 - 100 = -99
    sys.rows[0].htt[[0, 0]] = 1.0;
    sys.rows[0].htbeta[[0, 0]] = 1.0;
    sys.rows[0].gt[0] = 0.0;
    sys.rows[1].htt[[0, 0]] = 0.01;
    sys.rows[1].htbeta[[0, 1]] = 1.0;
    sys.rows[1].gt[0] = 0.0;
    // Pure β gradient (no per-row gt), so the reduced rhs_β is exactly -g_β and
    // the healthy-direction Newton solve is the clean `S_00·Δβ_0 = -g_β,0`.
    sys.gb[0] = -10.0; // expect Δβ_0 = -g/S = 10/5 = 2.0 exactly
    sys.gb[1] = 0.7;
    sys.refresh_row_hessian_fingerprint();

    let mut floored = ArrowSolveOptions::direct();
    floored.schur_pd_floor = Some(1.0e-8);
    let (_delta_t, delta_beta, _cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &floored)
            .expect("mixed-collapse floored solve must succeed");

    // The reduced system solves `S Δβ = -g_β` (arrow Newton convention). Healthy
    // direction: 5·Δβ_0 = 10 ⇒ Δβ_0 = 2.0 EXACTLY — the floor (which only lifts
    // eigenvalues ≤ 1e-6) must not perturb the λ=5 direction.
    assert!(
        (delta_beta[0] - 2.0).abs() < 1e-9,
        "healthy β direction must keep its EXACT Newton step Δβ_0 = -g/S = 2.0; \
         got {} (floor must not perturb the λ=5 subspace)",
        delta_beta[0]
    );
    assert!(
        delta_beta[1].is_finite(),
        "collapsed β direction must yield a finite (minimally-damped) step; got {}",
        delta_beta[1]
    );
}

/// #1026 — the reduced-Schur spectral PD-floor (the SAE co-collapse SOLVE-path
/// cure). On the EXACT collapsed geometry that makes the reduced Schur
/// indefinite, the solve must (a) REFUSE with `SchurFactorFailed` under the
/// default options (no floor) — reproducing the non-PD abort that forces the LM
/// loop to inflate `ridge_β` and the inner Newton to crawl — and (b) SUCCEED
/// with a finite step once the floor is engaged, because the floor lifts only
/// the collapsed eigen-directions to a small positive stiffness. A regression
/// that dropped the floor (or never wired it on the SAE path) fails (b); a
/// regression that floored a HEALTHY system fails (a).
#[test]
fn reduced_schur_pd_floor_recovers_indefinite_collapse_1026() {
    let sys = indefinite_collapsed_schur_system(0.01, 1.0, 1.0);

    // (a) Default options: strict non-PD refusal.
    let strict = ArrowSolveOptions::direct();
    assert!(
        strict.schur_pd_floor.is_none(),
        "default options must NOT floor the Schur (strict contract for BA / non-SAE callers)"
    );
    let strict_result = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &strict);
    match strict_result {
        Err(ArrowSchurError::SchurFactorFailed { .. }) => {}
        other => panic!(
            "indefinite reduced Schur must REFUSE without the PD-floor (the co-collapse \
             non-PD abort); got {other:?}"
        ),
    }

    // (b) Floor engaged: a finite step on the conditioned (PD-floored) Schur.
    let mut floored = ArrowSolveOptions::direct();
    floored.schur_pd_floor = Some(1.0e-8);
    let (delta_t, delta_beta, _cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &floored)
            .expect("spectral PD-floor must convert the non-PD refusal into a usable step");
    assert!(
        delta_t.iter().all(|v| v.is_finite()) && delta_beta.iter().all(|v| v.is_finite()),
        "PD-floored solve must return a finite Newton step: Δt={delta_t:?}, Δβ={delta_beta:?}"
    );
    assert!(
        delta_t.len() == 2 && delta_beta.len() == 2,
        "step dimensions must match the system (d·n=2 latent, k=2 border)"
    );
    // The step must be non-trivial (a genuine descent move, not a frozen zero).
    let step_norm: f64 = delta_t
        .iter()
        .chain(delta_beta.iter())
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    assert!(
        step_norm > 0.0,
        "PD-floored solve must take a genuine (non-zero) step, not stall at zero"
    );
}

/// #1026 — the floor is a strict NO-OP on a healthy (PD) reduced Schur: a
/// well-conditioned system solves IDENTICALLY with and without the floor, so the
/// cure cannot perturb the converged dictionary anywhere the collapse does not
/// occur. (`h_bb` large vs the subtraction keeps S positive definite.)
#[test]
fn reduced_schur_pd_floor_is_noop_on_healthy_system_1026() {
    // h_bb = 1000 ≫ c²/h_tt = 100, so S = diag(900, 900) ≻ 0 (healthy).
    let sys = indefinite_collapsed_schur_system(0.01, 1.0, 1000.0);

    let strict = ArrowSolveOptions::direct();
    let mut floored = ArrowSolveOptions::direct();
    floored.schur_pd_floor = Some(1.0e-8);

    let (dt_strict, db_strict, _c0) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &strict)
            .expect("healthy PD Schur must solve without the floor");
    let (dt_floor, db_floor, _c1) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &floored)
            .expect("healthy PD Schur must also solve with the floor enabled");

    // Cholesky of the genuine PD Schur is taken FIRST; the floor branch is never
    // reached, so the two steps are bit-for-bit identical.
    for (a, b) in dt_strict.iter().zip(dt_floor.iter()) {
        assert_eq!(a, b, "healthy-system Δt must be identical with/without the floor");
    }
    for (a, b) in db_strict.iter().zip(db_floor.iter()) {
        assert_eq!(a, b, "healthy-system Δβ must be identical with/without the floor");
    }
}

/// #1026 — the MATRIX-FREE (InexactPCG) reduced-Schur curvature floor: the REAL
/// K≥4 production path. At K≥4 the SAE border (`beta_dim = K·M·p`) far exceeds
/// the Direct/dense threshold (2000), so the inner solve runs unbounded
/// (trust radius = ∞) Steihaug-PCG against the matrix-free Schur operator. On
/// the collapsed geometry the reduced Schur is indefinite, so the very first CG
/// direction hits `pᵀSp ≤ 0`; with no trust radius there is no boundary to step
/// to. WITHOUT the floor this surfaces `UnboundedNegativeCurvature` (the stall
/// that makes the outer LM loop inflate `ridge_β` and the inner Newton crawl —
/// `‖Π⊥Δ‖` never drops). WITH the floor the operator is minimally ridged along
/// the collapsed direction and CG returns a finite descent step.
///
/// We force `InexactPCG` (the K≥4 mode) on the same indefinite fixture; its
/// default trust radius is already `f64::INFINITY` (the SAE inner-solve
/// condition).
#[test]
fn matrix_free_pcg_curvature_floor_recovers_unbounded_negative_curvature_1026() {
    let sys = indefinite_collapsed_schur_system(0.01, 1.0, 1.0);

    // WITHOUT the floor: unbounded InexactPCG hits negative curvature and fails
    // (the co-collapse stall). The default trust radius is ∞.
    let mut pcg_strict = ArrowSolveOptions::inexact_pcg();
    assert!(
        pcg_strict.trust_region.radius == f64::INFINITY,
        "the SAE inner solve runs UNBOUNDED (radius = ∞); fixture must match that"
    );
    assert!(pcg_strict.schur_pd_floor.is_none());
    // Keep it strictly the unbounded matrix-free path.
    pcg_strict.gpu_matvec = None;
    let strict_result = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &pcg_strict);
    // Without the floor the unbounded matrix-free solve FAILS on the indefinite
    // reduced Schur — either at the Jacobi-diagonal refusal (`PcgFailed`, since
    // `S_ii = H_ββ,ii − Σ … < 0` so the preconditioner cannot invert it) or, for
    // a Schur whose diagonal stays positive while an off-diagonal direction is
    // indefinite, at the CG negative-curvature event (`UnboundedNegativeCurvature`).
    // Either way it is the co-collapse stall the floor exists to cure.
    match strict_result {
        Err(ArrowSchurError::UnboundedNegativeCurvature { curvature, .. }) => {
            assert!(
                curvature <= 0.0,
                "the reported curvature must be the offending non-positive pᵀSp; got {curvature}"
            );
        }
        Err(ArrowSchurError::PcgFailed { .. }) => {}
        other => panic!(
            "unbounded matrix-free PCG on an indefinite reduced Schur must FAIL \
             without the floor (the co-collapse stall: non-PD Jacobi diagonal or \
             CG negative curvature); got {other:?}"
        ),
    }

    // WITH the floor: the curvature-floor retry conditions the collapsed
    // direction and the unbounded PCG returns a finite step.
    let mut pcg_floor = ArrowSolveOptions::inexact_pcg();
    pcg_floor.schur_pd_floor = Some(1.0e-8);
    pcg_floor.gpu_matvec = None;
    let (delta_t, delta_beta, _cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &pcg_floor).expect(
            "matrix-free curvature floor must convert the unbounded negative-curvature \
             stall into a usable step",
        );
    assert!(
        delta_t.iter().all(|v| v.is_finite()) && delta_beta.iter().all(|v| v.is_finite()),
        "floored matrix-free solve must return a finite step: Δt={delta_t:?}, Δβ={delta_beta:?}"
    );
    let step_norm: f64 = delta_t
        .iter()
        .chain(delta_beta.iter())
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    assert!(
        step_norm > 0.0,
        "floored matrix-free solve must take a genuine (non-zero) descent step, not stall"
    );
}

/// #1026 — the matrix-free curvature floor is a strict NO-OP on a healthy (PD)
/// reduced Schur: a well-conditioned system never hits `pᵀSp ≤ 0`, so the
/// unbounded InexactPCG solve is identical with and without the floor enabled.
#[test]
fn matrix_free_pcg_curvature_floor_is_noop_on_healthy_system_1026() {
    // h_bb = 1000 ≫ subtraction 100 → S = diag(900, 900) ≻ 0 (healthy).
    let sys = indefinite_collapsed_schur_system(0.01, 1.0, 1000.0);

    let mut pcg_strict = ArrowSolveOptions::inexact_pcg();
    pcg_strict.gpu_matvec = None;
    let mut pcg_floor = ArrowSolveOptions::inexact_pcg();
    pcg_floor.schur_pd_floor = Some(1.0e-8);
    pcg_floor.gpu_matvec = None;

    let (dt_strict, db_strict, _c0) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &pcg_strict)
            .expect("healthy PD Schur must solve without the floor");
    let (dt_floor, db_floor, _c1) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &pcg_floor)
            .expect("healthy PD Schur must also solve with the floor enabled");

    // No negative curvature is ever hit, so the floor branch never fires and the
    // two unbounded-PCG iterates match to tight tolerance (CG arithmetic is the
    // same; the floor only adds an alternate failure-recovery path).
    assert_eq!(dt_strict.len(), dt_floor.len());
    for (a, b) in dt_strict.iter().zip(dt_floor.iter()) {
        assert!(
            (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs())),
            "healthy-system Δt must match with/without the floor: {a} vs {b}"
        );
    }
    for (a, b) in db_strict.iter().zip(db_floor.iter()) {
        assert!(
            (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs())),
            "healthy-system Δβ must match with/without the floor: {a} vs {b}"
        );
    }
}

/// #1026 — shared-hyperparameter ARD at large K. The OUTER REML optimizer
/// searches over the flat `to_flat()` coordinate vector, so its dimension is
/// what makes a large-K fit tractable or not.
///
/// Per-atom ARD (the small/moderate-K default) gives one independent outer
/// coordinate per atom per axis: `2 + Σ_k d_k`. At K = 32_768 1-D atoms that is
/// 32_770 outer hyperparameters, each outer eval refitting the whole dictionary
/// — intractable for a generic outer optimizer. The shared mode collapses ARD
/// to one strength per intrinsic axis (`max_d`), broadcast to every atom, so the
/// flat vector is a CONSTANT `2 + max_d` regardless of K.
///
/// This gate pins both regimes: per-atom keeps the historical `2 + Σ d_k` count
/// (so existing fits are unchanged), and shared is O(1) in K. It also checks the
/// broadcast round-trip: `from_flat` rebuilds the full per-atom precision table
/// the inner solve consumes, with every atom sharing the per-axis strength.
#[test]
fn shared_ard_collapses_outer_param_count_at_large_k() {
    // d=1 atoms (the worst case: one ARD axis per atom).
    let d_per_atom = 1usize;
    for &k in &[2usize, 32usize, 1000usize, 32_768usize] {
        let log_ard: Vec<ndarray::Array1<f64>> =
            (0..k).map(|_| ndarray::Array1::<f64>::zeros(d_per_atom)).collect();

        let per_atom = SaeManifoldRho::new(-0.5, -0.5, log_ard.clone());
        assert_eq!(per_atom.ard_sharing, ArdSharing::PerAtom);
        // Per-atom: 2 + Σ_k d_k = 2 + K (d=1). This is exactly the historical
        // layout — existing small-K fits keep their parameterization.
        assert_eq!(
            per_atom.to_flat().len(),
            2 + k * d_per_atom,
            "per-atom ARD must keep 2 + Σ d_k outer coords (K={k})"
        );

        let shared = SaeManifoldRho::new_shared_ard(-0.5, -0.5, log_ard.clone());
        assert_eq!(shared.ard_sharing, ArdSharing::Shared);
        // Shared: a CONSTANT 2 + max_d outer coords, independent of K.
        assert_eq!(
            shared.to_flat().len(),
            2 + d_per_atom,
            "shared ARD must be O(1) in K (got K={k})"
        );
    }

    // Round-trip: shared flat broadcasts back to a full per-atom table, every
    // atom carrying the shared per-axis strength, preserving heterogeneous d_k.
    let log_ard = vec![
        ndarray::Array1::<f64>::zeros(2), // a 2-axis atom
        ndarray::Array1::<f64>::zeros(1), // a 1-axis atom
        ndarray::Array1::<f64>::zeros(2), // another 2-axis atom
    ];
    let shared = SaeManifoldRho::new_shared_ard(0.0, 0.0, log_ard);
    let flat = shared.to_flat();
    // max_d = 2 → 2 + 2 = 4 outer coords.
    assert_eq!(flat.len(), 4);

    // Drive the two shared per-axis strengths to distinct values and broadcast.
    let mut moved = flat.clone();
    moved[2] = 1.5; // axis-0 shared log-precision
    moved[3] = -2.5; // axis-1 shared log-precision
    let rebuilt = shared.from_flat(moved.view());
    assert_eq!(rebuilt.ard_sharing, ArdSharing::Shared);
    // Every atom that owns axis 0 sees 1.5; every atom owning axis 1 sees -2.5;
    // the 1-axis atom keeps its single axis.
    assert_eq!(rebuilt.log_ard.len(), 3);
    assert_eq!(rebuilt.log_ard[0].as_slice().unwrap(), &[1.5, -2.5]);
    assert_eq!(rebuilt.log_ard[1].as_slice().unwrap(), &[1.5]);
    assert_eq!(rebuilt.log_ard[2].as_slice().unwrap(), &[1.5, -2.5]);

    // to_flat is the exact inverse of the broadcast (read-back is exact when the
    // table is uniform across owners, which the broadcast guarantees).
    let reflat = rebuilt.to_flat();
    for (a, b) in moved.iter().zip(reflat.iter()) {
        assert!((a - b).abs() <= 1e-12, "shared ARD round-trip must be exact: {a} vs {b}");
    }
}

/// BEHAVIORAL half of the #1026 shared-ARD collapse: the count-collapse above is
/// only useful if the SHARED parameterization is still a VALID convergent outer
/// coordinate — a smaller flat vector is worthless if the inner joint fit no
/// longer converges under it. This drives the real production inner solve
/// (`reml_criterion_with_cache`, the same entry the outer optimizer steps on)
/// at a `new_shared_ard` ρ and asserts:
///   1. the shared flat coordinate is strictly SHORTER than the per-atom one
///      (2 + max_d  <  2 + Σ_k d_k for K>1) — the O(1)-in-K collapse, and
///   2. the inner fit CONVERGES to a FINITE Laplace/REML criterion, i.e. the
///      shared `from_flat` broadcast feeds the inner solve a well-posed per-atom
///      precision table that the joint Newton can actually reach a minimum on.
/// A regression that broke the broadcast (or made the shared coordinate
/// non-convergent) would either shorten nothing or return a non-finite criterion
/// here. Built at small K (CPU-feasible) because the shared coordinate's validity
/// is K-independent — that independence is the whole point of the collapse.
#[test]
fn shared_ard_is_a_convergent_outer_coordinate_1026() {
    // Two distinct, non-collinear decoders so the inner data-fit is well posed
    // and the joint fit has a genuine (non-degenerate) minimum to converge to.
    let mut dec0 = Array2::<f64>::zeros((M, P));
    dec0[[0, 0]] = 0.5;
    dec0[[1, 1]] = 1.0;
    dec0[[2, 2]] = 1.0;
    let mut dec1 = Array2::<f64>::zeros((M, P));
    dec1[[0, 1]] = 0.5;
    dec1[[1, 2]] = 1.0;
    dec1[[2, 0]] = 1.0;
    let mut term = build_two_atom_term(dec0, dec1);
    let n = term.n_obs();
    let target = Array2::from_shape_fn((n, P), |(i, c)| 0.1 * ((i as f64) * 0.3 + (c as f64)).sin());

    // d=1 circle atoms ⇒ max_d = 1. Per-atom flat = 2 + K*d = 2 + 2 = 4; shared
    // flat = 2 + max_d = 3. The collapse is real even at K=2; it widens as K→∞.
    let log_ard = vec![Array1::<f64>::from_elem(1, (1.0e-1_f64).ln()); 2];
    let per_atom = SaeManifoldRho::new((1.0e-2_f64).ln(), (1.0e-2_f64).ln(), log_ard.clone());
    let shared = SaeManifoldRho::new_shared_ard((1.0e-2_f64).ln(), (1.0e-2_f64).ln(), log_ard);
    assert_eq!(shared.ard_sharing, ArdSharing::Shared);
    assert!(
        shared.to_flat().len() < per_atom.to_flat().len(),
        "shared ARD outer coordinate ({}) must be strictly shorter than per-atom ({})",
        shared.to_flat().len(),
        per_atom.to_flat().len()
    );

    // Inner-solve knobs mirroring the production outer objective's defaults.
    let (cost, _loss, _cache) = term
        .reml_criterion_with_cache(target.view(), &shared, None, 64, 1.0, 1.0e-8, 1.0e-8)
        .expect("inner joint fit must converge at the shared-ARD ρ");
    assert!(
        cost.is_finite(),
        "shared-ARD inner fit must reach a FINITE REML criterion (convergence); got {cost}"
    );
}

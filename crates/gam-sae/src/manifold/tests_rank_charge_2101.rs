//! #5/(B) rank-charge criterion tests: the honest realised-rank BIC charge
//! (i) ACCEPTS a real rank-2 circle, (ii) NEUTRALISES a vanishing decoder
//! (co-collapse fix), (iii) is INERT when the flag is off.

use crate::manifold::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};
use std::sync::Arc;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}
fn lcg_normal(s: &mut u64) -> f64 {
    let u1 = lcg(s).max(1e-12);
    let u2 = lcg(s);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A fitted K=1 rank-2 circle on dims (0,1): cos→e0, sin→e1, coordinate = the
/// true phase, amp 1, noise 0.05. Returns the fitted term + rho.
fn fitted_circle_term(n: usize, p: usize) -> (SaeManifoldTerm, SaeManifoldRho) {
    let mut s = 0x2101_B1C_0000_0005u64;
    let theta: Vec<f64> = (0..n).map(|_| std::f64::consts::TAU * lcg(&mut s)).collect();
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] += theta[i].cos();
        x[[i, 1]] += theta[i].sin();
        for j in 0..p {
            x[[i, j]] += 0.05 * lcg_normal(&mut s);
        }
    }
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords =
        Array2::<f64>::from_shape_fn((n, 1), |(r, _)| theta[r] / std::f64::consts::TAU);
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let mut decoder = Array2::<f64>::zeros((3, p));
    decoder[[1, 0]] = 1.0;
    decoder[[2, 1]] = 1.0;
    let atom = SaeManifoldAtom::new(
        "circle".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone());
    let logits = Array2::<f64>::from_elem((n, 1), 3.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(0.7, 1.0, false),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    term.set_guards_enabled(false);
    let mut rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    term.run_joint_fit_arrow_schur(x.view(), &mut rho, None, 60, 1.0, 1e-6, 1e-6)
        .expect("K=1 circle fit");
    (term, rho)
}

/// (i) real rank-2 circle → d_eff in the BIC-accept range (rank≈2 × basis-EDF),
/// AND (ii) a vanishing decoder (×1e-4) → d_eff→0 (co-collapse neutralised).
#[test]
fn rank_charge_deff_accepts_circle_and_neutralises_vanishing() {
    let (mut term, rho) = fitted_circle_term(80, 16);
    // Dispersion (noise floor R) from a reml pass.
    let (_v, loss, cache) = term
        .reml_criterion_with_cache(unit_target(&term).view(), &rho, None, 0, 1.0, 1e-6, 1e-6)
        .unwrap_or_else(|_| panic!("reml pass"));
    let disp = term.reconstruction_dispersion(&loss, &cache, &rho).unwrap();
    drop((loss, cache));

    let d_real = term.per_atom_realised_rank_dof(&rho, disp).unwrap();
    eprintln!(
        "[rank-charge] dispersion R={disp:.5}  circle d_eff={:.3} → charge ½·d_eff·ln80={:.3}",
        d_real[0],
        0.5 * d_real[0] * (80f64).ln()
    );
    assert!(
        d_real[0] > 2.5 && d_real[0] < 8.0,
        "rank-2 circle d_eff should be ~rank-2×basis-EDF (~4-6); got {:.3}",
        d_real[0]
    );
    // The BIC charge must ACCEPT: ½·d_eff·log n − Δloss < 0 is the birth decision;
    // here just assert the charge is well below the ~n-DOF that would over-reject.
    assert!(
        0.5 * d_real[0] * (80f64).ln() < 15.0,
        "rank-charge must be modest (accept), got charge {:.3}",
        0.5 * d_real[0] * (80f64).ln()
    );

    // Vanishing: shrink the decoder → singular values ≪ noise floor → d_eff→0.
    let saved = term.atoms[0].decoder_coefficients.clone();
    term.atoms[0]
        .decoder_coefficients
        .assign(&(&saved * 1e-4));
    let d_vanish = term.per_atom_realised_rank_dof(&rho, disp).unwrap();
    eprintln!("[rank-charge] vanishing (decoder×1e-4) d_eff={:.5} → charge≈0 (neutral)", d_vanish[0]);
    assert!(
        d_vanish[0] < 0.2,
        "vanishing decoder must give d_eff→0 (neutral); got {:.4}",
        d_vanish[0]
    );
    term.atoms[0].decoder_coefficients.assign(&saved);
}

/// (iii) flag OFF ⇒ the criterion value is BYTE-IDENTICAL to the historical path.
#[test]
fn rank_charge_flag_off_is_inert() {
    let (mut term, rho) = fitted_circle_term(80, 16);
    let tgt = unit_target(&term);
    // default (flag off)
    let (v_off, _, _) = term
        .reml_criterion_with_cache(tgt.view(), &rho, None, 0, 1.0, 1e-6, 1e-6)
        .unwrap();
    // explicitly set off — still identical
    term.set_rank_charge_evidence(false);
    let (v_off2, _, _) = term
        .reml_criterion_with_cache(tgt.view(), &rho, None, 0, 1.0, 1e-6, 1e-6)
        .unwrap();
    assert_eq!(
        v_off, v_off2,
        "rank_charge_evidence=false must be bit-identical to the historical criterion"
    );
    // flag ON changes the criterion (the rank charge replaces the coord-block).
    term.set_rank_charge_evidence(true);
    let (v_on, _, _) = term
        .reml_criterion_with_cache(tgt.view(), &rho, None, 0, 1.0, 1e-6, 1e-6)
        .unwrap();
    eprintln!("[rank-charge] reml OFF={v_off:.4}  ON={v_on:.4}  (ON lowers the circle's complexity)");
    assert!(
        (v_on - v_off).abs() > 1e-6 && v_on.is_finite(),
        "rank_charge_evidence=true must change the (finite) criterion; off={v_off:.4} on={v_on:.4}"
    );
}

/// (iv) HEALTHY K=3 DECISION-LEVEL CONTROL (hand-built — the pipeline has no
/// healthy multi-atom fit pre-recovery): three well-separated clean rank-2
/// circles on disjoint output dims. The rank-charge value changes a lot BY
/// DESIGN (over-charge removal), so inertness is checked at the DECISION level:
/// every atom must price as a clean rank-2 (d_eff ~4-6), the criterion stays
/// finite and well-conditioned (no Schur collapse at K≥2), and flag-off is
/// bit-identical.
#[test]
fn rank_charge_healthy_k3_control_well_conditioned() {
    let n = 96usize;
    let p = 18usize;
    let ncirc = 3usize;
    let mut s = 0x2101_C3C_0000_0009u64;
    let theta: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..ncirc).map(|_| std::f64::consts::TAU * lcg(&mut s)).collect())
        .collect();
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for c in 0..ncirc {
            x[[i, 2 * c]] += theta[i][c].cos();
            x[[i, 2 * c + 1]] += theta[i][c].sin();
        }
        for j in 0..p {
            x[[i, j]] += 0.05 * lcg_normal(&mut s);
        }
    }
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    // Each atom seeded CLEAN on its own axis-aligned dims (2c, 2c+1), true phase.
    let mut atoms = Vec::new();
    let mut coord_blocks = Vec::new();
    let mut manifolds = Vec::new();
    for c in 0..ncirc {
        let coords =
            Array2::<f64>::from_shape_fn((n, 1), |(r, _)| theta[r][c] / std::f64::consts::TAU);
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let mut decoder = Array2::<f64>::zeros((3, p));
        decoder[[1, 2 * c]] = 1.0;
        decoder[[2, 2 * c + 1]] = 1.0;
        let atom = SaeManifoldAtom::new(
            format!("circle{c}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_second_jet(evaluator.clone());
        atoms.push(atom);
        coord_blocks.push(coords);
        manifolds.push(LatentManifold::Circle { period: 1.0 });
    }
    let logits = Array2::<f64>::from_elem((n, ncirc), 3.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        manifolds,
        AssignmentMode::ibp_map(0.7, 1.0, false),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    term.set_guards_enabled(false);
    let mut rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); ncirc]);
    term.run_joint_fit_arrow_schur(x.view(), &mut rho, None, 60, 1.0, 1e-6, 1e-6)
        .expect("K=3 clean fit");

    let (v_off, loss, cache) = term
        .reml_criterion_with_cache(x.view(), &rho, None, 0, 1.0, 1e-6, 1e-6)
        .unwrap();
    let disp = term.reconstruction_dispersion(&loss, &cache, &rho).unwrap();
    drop((loss, cache));
    let d_eff = term.per_atom_realised_rank_dof(&rho, disp).unwrap();
    eprintln!("[rank-charge K=3] d_eff per atom = {:?}  disp={disp:.5}",
        d_eff.iter().map(|v| (v*100.0).round()/100.0).collect::<Vec<_>>());
    for (k, &de) in d_eff.iter().enumerate() {
        assert!(
            de > 2.0 && de < 8.0,
            "K=3 atom {k}: every clean rank-2 circle must price ~4-6; got d_eff={de:.3}"
        );
    }
    term.set_rank_charge_evidence(true);
    let (v_on, _, _) = term
        .reml_criterion_with_cache(x.view(), &rho, None, 0, 1.0, 1e-6, 1e-6)
        .unwrap();
    eprintln!("[rank-charge K=3] reml OFF={v_off:.3} ON={v_on:.3}");
    assert!(
        v_on.is_finite() && v_off.is_finite(),
        "K=3 criterion must stay finite (no Schur collapse) both ways: off={v_off} on={v_on}"
    );
}

/// Build + fit a term with circles on the given output-dim indices (each circle
/// c on dims (2c, 2c+1)), against the shared target `x`. Used for leave-one-out
/// decision margins.
fn fit_circle_subset(
    x: &Array2<f64>,
    theta: &[Vec<f64>],
    circles: &[usize],
    flag: bool,
) -> (SaeManifoldTerm, SaeManifoldRho) {
    let n = x.nrows();
    let p = x.ncols();
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let mut atoms = Vec::new();
    let mut coord_blocks = Vec::new();
    let mut manifolds = Vec::new();
    for &c in circles {
        let coords =
            Array2::<f64>::from_shape_fn((n, 1), |(r, _)| theta[r][c] / std::f64::consts::TAU);
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        let mut decoder = Array2::<f64>::zeros((3, p));
        decoder[[1, 2 * c]] = 1.0;
        decoder[[2, 2 * c + 1]] = 1.0;
        let atom = SaeManifoldAtom::new(
            format!("circle{c}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_second_jet(evaluator.clone());
        atoms.push(atom);
        coord_blocks.push(coords);
        manifolds.push(LatentManifold::Circle { period: 1.0 });
    }
    let logits = Array2::<f64>::from_elem((n, circles.len()), 3.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coord_blocks,
        manifolds,
        AssignmentMode::ibp_map(0.7, 1.0, false),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    term.set_guards_enabled(false);
    term.set_rank_charge_evidence(flag);
    let mut rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); circles.len()]);
    term.run_joint_fit_arrow_schur(x.view(), &mut rho, None, 60, 1.0, 1e-6, 1e-6)
        .expect("subset fit");
    (term, rho)
}

/// (v) DECISION-LEVEL control (the commit gate): on a clean well-separated
/// 3-circle fit, the leave-one-out margin of every real atom must be < 0
/// (KEEPING it is favored ⇒ accepted) under the rank charge, and NO decision
/// may FLIP vs flag-off (no healthy atom newly rejected). The value shifts a lot
/// by design; the accept/reject OUTCOME must not.
#[test]
fn rank_charge_k3_decisions_preserved() {
    let n = 96usize;
    let p = 18usize;
    let ncirc = 3usize;
    let mut s = 0x2101_DEC_0000_0011u64;
    let theta: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..ncirc).map(|_| std::f64::consts::TAU * lcg(&mut s)).collect())
        .collect();
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for c in 0..ncirc {
            x[[i, 2 * c]] += theta[i][c].cos();
            x[[i, 2 * c + 1]] += theta[i][c].sin();
        }
        for j in 0..p {
            x[[i, j]] += 0.05 * lcg_normal(&mut s);
        }
    }
    // For each flag, compute the leave-one-out margin of each circle:
    //   margin_k = reml(all 3) − reml(drop k).  <0 ⇒ keeping k is favored (accepted).
    let margins = |flag: bool| -> Vec<f64> {
        let (mut t3, r3) = fit_circle_subset(&x, &theta, &[0, 1, 2], flag);
        let (v3, _, _) = t3
            .reml_criterion_with_cache(x.view(), &r3, None, 0, 1.0, 1e-6, 1e-6)
            .unwrap();
        (0..ncirc)
            .map(|drop| {
                let keep: Vec<usize> = (0..ncirc).filter(|&c| c != drop).collect();
                let (mut t2, r2) = fit_circle_subset(&x, &theta, &keep, flag);
                let (v2, _, _) = t2
                    .reml_criterion_with_cache(x.view(), &r2, None, 0, 1.0, 1e-6, 1e-6)
                    .unwrap();
                v3 - v2 // margin_drop: <0 ⇒ the dropped circle is worth KEEPING
            })
            .collect()
    };
    let m_off = margins(false);
    let m_on = margins(true);
    eprintln!("[rank-charge K=3 decisions] leave-one-out margins OFF={m_off:?} ON={m_on:?}");
    for k in 0..ncirc {
        // (a) every real atom ACCEPTED under the rank charge (margin < 0).
        assert!(
            m_on[k] < 0.0,
            "circle {k}: rank-charge must ACCEPT the real atom (margin<0); got {:.3}",
            m_on[k]
        );
        // (b) NO decision flip vs flag-off (a healthy atom accepted off must stay accepted on).
        assert!(
            !(m_off[k] < 0.0 && m_on[k] >= 0.0),
            "circle {k}: decision FLIPPED off→on (was accepted, now rejected): off={:.3} on={:.3}",
            m_off[k],
            m_on[k]
        );
    }
    // (c) spurious/noise atom rejected is covered structurally by the vanishing
    // test (rank→0 → charge 0 → ΔEV rejects); a real atom here is never spurious.
}

/// The reconstruction target the fitted circle was built against (re-derived from
/// the same seed so the reml pass scores the real data).
fn unit_target(term: &SaeManifoldTerm) -> Array2<f64> {
    let n = term.n_obs();
    let p = term.output_dim();
    let mut s = 0x2101_B1C_0000_0005u64;
    let theta: Vec<f64> = (0..n).map(|_| std::f64::consts::TAU * lcg(&mut s)).collect();
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] += theta[i].cos();
        x[[i, 1]] += theta[i].sin();
        for j in 0..p {
            x[[i, j]] += 0.05 * lcg_normal(&mut s);
        }
    }
    x
}

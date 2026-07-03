//! #2101 DIAGNOSTIC PROBE (not a pass/fail guard): localize WHERE the born
//! decoder dies in the real IBP regime — the K=1 birth SUB-FIT
//! (`fit_single_atom_response_in_place`, penalty-on-scale-B) vs the JOINT BACKFIT
//! (gate-dependent deflation). Reproduces red-tree's disjoint 6-circle ibp_map
//! recovery (n=80, p=16, distinct amps 1.0..0.55, noise 0.05, structured_whitening
//! OFF) and records the last-born atom's ‖B‖ at every SAC progress event via the
//! callback (no driver edit). Run with `-- --nocapture` to read the trajectory.

use crate::manifold::{
    fit_stagewise, AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind,
    SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm, StagewiseConfig,
    StagewiseProgress,
};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};
use std::cell::RefCell;
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

#[test]
fn probe_2101_birth_locus_disjoint_6circle_ibp() {
    let n = 80usize;
    let p = 16usize;
    let ncirc = 6usize;
    // distinct amps 1.0..0.55 (mirror driver.gen); disjoint AXIS-ALIGNED frames:
    // circle c lives on output dims (2c, 2c+1). Same difficulty class as red-tree's
    // random-orthonormal disjoint frames for the born-decoder mechanism.
    let amps: Vec<f64> = (0..ncirc)
        .map(|c| 1.0 - 0.45 * (c as f64) / ((ncirc - 1) as f64))
        .collect();
    let mut s = 0x2101_D150_0000_0001u64;
    let theta: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..ncirc).map(|_| std::f64::consts::TAU * lcg(&mut s)).collect())
        .collect();
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for c in 0..ncirc {
            x[[i, 2 * c]] += amps[c] * theta[i][c].cos();
            x[[i, 2 * c + 1]] += amps[c] * theta[i][c].sin();
        }
        for j in 0..p {
            x[[i, j]] += 0.05 * lcg_normal(&mut s);
        }
    }

    // K=1 ibp seed: one circle atom on dims (0,1), coords ~ row fraction.
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(r, _)| r as f64 / n as f64);
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let mut decoder = Array2::<f64>::zeros((3, p));
    decoder[[1, 0]] = 1.0;
    decoder[[2, 1]] = 1.0;
    let atom = SaeManifoldAtom::new(
        "seed".to_string(),
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
    let mut seed_term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    seed_term.set_guards_enabled(false);
    let mut rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    let config = StagewiseConfig {
        inner_max_iter: 40,
        learning_rate: 1.0,
        ridge_ext_coord: 1e-6,
        ridge_beta: 1e-6,
        max_births: 8,
        max_backfit_sweeps: 6,
        min_effect_ev: 0.0,
        max_factor_rank: 8,
        structured_whitening: false,
    };
    // Fit the K=1 seed before entering stagewise (mirrors the pipeline's seed fit).
    seed_term
        .run_joint_fit_arrow_schur(
            x.view(),
            &mut rho,
            None,
            config.inner_max_iter,
            config.learning_rate,
            config.ridge_ext_coord,
            config.ridge_beta,
        )
        .expect("seed K=1 fit");

    let log: RefCell<Vec<String>> = RefCell::new(Vec::new());
    {
        let mut cb = |pg: StagewiseProgress<'_>| -> Result<(), String> {
            let k = pg.k_atoms;
            let (born_norm, rows) = if k >= 1 {
                let d = &pg.term.atoms[k - 1].decoder_coefficients;
                let bn = d.iter().map(|v| v * v).sum::<f64>().sqrt();
                let rn: Vec<f64> = (0..d.nrows())
                    .map(|r| d.row(r).iter().map(|v| v * v).sum::<f64>().sqrt())
                    .collect();
                (bn, rn)
            } else {
                (0.0, vec![])
            };
            log.borrow_mut().push(format!(
                "{:?} round={} sweep={} k={} born|B|={:.3e} rows={:?} ev={:?} reml_after={:?}",
                pg.event,
                pg.birth_round,
                pg.backfit_sweep,
                k,
                born_norm,
                rows.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
                pg.ev.map(|e| (e * 10000.0).round() / 10000.0),
                pg.joint_reml_after.map(|r| (r * 100.0).round() / 100.0),
            ));
            Ok(())
        };
        let res = fit_stagewise(seed_term, rho, x.view(), None, None, &config, Some(&mut cb));
        log.borrow_mut()
            .push(format!("FINAL: {:?}", res.map(|r| (r.term.k_atoms(), r.report.births_accepted))));
    }
    eprintln!("\n==== #2101 BIRTH LOCUS TRAJECTORY (disjoint 6-circle ibp) ====");
    for line in log.borrow().iter() {
        eprintln!("{line}");
    }
    eprintln!("==== END #2101 PROBE ====\n");
}

/// #2101 UNIFICATION TEST (decides one-fix-vs-two): does the K=1 fit DESTROY a
/// PROPERLY-SEEDED rank-2 circle? Seed a K=1 atom as a perfect circle (cos→e0,
/// sin→e1, coordinate = the TRUE phase) on a single-circle target, run the same
/// K=1 driver the birth sub-fit uses, and read the cos/sin decoder-row norms
/// before/after. If cos/sin SURVIVE ⇒ the fit is fine and #2101 is purely the
/// DC-SEED (seed cos/sin, my lane, SEPARATE from #5). If cos/sin COLLAPSE ⇒ the
/// fit/criterion crushes circles (the ‖B‖→0 / harmonic-row reward — UNIFIED with
/// fit-robustness's #5). Runs both ibp and softmax gate modes.
#[test]
fn probe_2101_proper_circle_seed_survival() {
    let n = 80usize;
    let p = 16usize;
    let mut s = 0x2101_C15C_0000_0007u64;
    // Single circle on dims (0,1), amp 1, independent phase, noise 0.05.
    let theta: Vec<f64> = (0..n).map(|_| std::f64::consts::TAU * lcg(&mut s)).collect();
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] += theta[i].cos();
        x[[i, 1]] += theta[i].sin();
        for j in 0..p {
            x[[i, j]] += 0.05 * lcg_normal(&mut s);
        }
    }
    // The PeriodicHarmonicEvaluator maps coord t∈[0,1) to angle 2πt; seed the
    // coordinate at the true phase so the circle is perfectly aligned.
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(r, _)| {
        theta[r] / std::f64::consts::TAU
    });
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();

    for (mode_name, logit) in [
        ("ibp_map", 3.0f64),
        ("ibp_map", -4.0),
        ("softmax", 3.0),
        ("softmax", -4.0),
    ] {
        let mode = if mode_name == "ibp_map" {
            AssignmentMode::ibp_map(0.7, 1.0, false)
        } else {
            AssignmentMode::softmax(1.0)
        };
        // Perfect circle decoder: DC=0, cos-row→e0, sin-row→e1.
        let mut decoder = Array2::<f64>::zeros((3, p));
        decoder[[1, 0]] = 1.0;
        decoder[[2, 1]] = 1.0;
        let atom = SaeManifoldAtom::new(
            "circle".to_string(),
            SaeAtomBasisKind::Periodic,
            1,
            phi.clone(),
            jet.clone(),
            decoder,
            Array2::<f64>::eye(3),
        )
        .unwrap()
        .with_basis_second_jet(evaluator.clone());
        let logits = Array2::<f64>::from_elem((n, 1), logit);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            vec![coords.clone()],
            vec![LatentManifold::Circle { period: 1.0 }],
            mode,
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        term.set_guards_enabled(false);
        let mut rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
        let rows_before: Vec<f64> = {
            let d = &term.atoms[0].decoder_coefficients;
            (0..d.nrows())
                .map(|r| d.row(r).iter().map(|v| v * v).sum::<f64>().sqrt())
                .collect()
        };
        term.run_joint_fit_arrow_schur(x.view(), &mut rho, None, 60, 1.0, 1e-6, 1e-6)
            .expect("K=1 circle fit");
        let rows_after: Vec<f64> = {
            let d = &term.atoms[0].decoder_coefficients;
            (0..d.nrows())
                .map(|r| d.row(r).iter().map(|v| v * v).sum::<f64>().sqrt())
                .collect()
        };
        let harm_before = (rows_before[1].powi(2) + rows_before[2].powi(2)).sqrt();
        let harm_after = (rows_after[1].powi(2) + rows_after[2].powi(2)).sqrt();
        eprintln!(
            "\n==== #2101 PROPER-CIRCLE-SEED SURVIVAL ({mode_name}, logit={logit}) ====\n  rows before = {:?}\n  rows after  = {:?}\n  harmonic ‖(cos,sin)‖: before={:.4} after={:.4}  (SURVIVES if after stays ~before; COLLAPSE if after≈0)",
            rows_before.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
            rows_after.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>(),
            harm_before,
            harm_after,
        );
    }
    eprintln!("==== END #2101 SURVIVAL PROBE ====\n");
}

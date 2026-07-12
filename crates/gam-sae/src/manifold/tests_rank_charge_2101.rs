//! #5/(B) rank-charge criterion tests: the honest realised-rank BIC charge
//! (i) ACCEPTS a real rank-2 circle, (ii) NEUTRALISES a vanishing decoder
//! (co-collapse fix), and the canonical criterion is dense/streaming invariant.

use crate::manifold::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2};
use std::sync::{Arc, Mutex};

/// The two K=3 controls each run several joint fits; cargo runs tests in-binary
/// on a thread pool, so left unguarded they can execute simultaneously and, under
/// a loaded host, starve each other (observed as a spurious "hang"/kill, not a
/// logic failure). Serialising them against each other caps peak concurrency to
/// one heavy multi-atom fit at a time. Poison-tolerant: a panic in one test must
/// surface as that test's failure, not poison-fail the sibling.
static K3_SERIAL: Mutex<()> = Mutex::new(());
fn k3_guard() -> std::sync::MutexGuard<'static, ()> {
    K3_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

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
    let theta: Vec<f64> = (0..n)
        .map(|_| std::f64::consts::TAU * lcg(&mut s))
        .collect();
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] += theta[i].cos();
        x[[i, 1]] += theta[i].sin();
        for j in 0..p {
            x[[i, j]] += 0.05 * lcg_normal(&mut s);
        }
    }
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(r, _)| theta[r] / std::f64::consts::TAU);
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let mut decoder = Array2::<f64>::zeros((3, p));
    decoder[[1, 0]] = 1.0;
    decoder[[2, 1]] = 1.0;
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false),
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
/// AND (ii) a decoder scaled far enough below the explicit degeneracy threshold
/// `RANK_VANISHED_REL·R` → d_eff=0 (co-collapse neutralised).
#[test]
fn rank_charge_deff_accepts_circle_and_neutralises_vanishing() {
    let (mut term, rho) = fitted_circle_term(80, 16);
    // Dispersion (noise floor R) from a reml pass.
    let (_v, loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            unit_target(&term).view(),
            &rho,
            None,
            0,
            1.0,
            1e-6,
            1e-6,
        )
        .unwrap_or_else(|_| panic!("reml pass"));
    let disp = term
        .reconstruction_dispersion(&loss, &cache, &rho, None)
        .unwrap();
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

    // Vanishing is NOT synonymous with merely falling below the MP detection
    // edge. The latter is an alive-but-unresolved atom and production correctly
    // promotes it to chargeable rank 1. Derive a scale with a factor-four safety
    // margin below the separate categorical threshold instead of guessing an
    // absolute multiplier: under D↦cD every reconstruction eigenvalue obeys
    // μ_k(c)=c²μ_k(1).
    let saved = term.atoms[0].decoder_coefficients.clone();
    let mut grams = term.empty_decoder_gram_accumulator();
    term.accumulate_decoder_gram(&mut grams);
    let n_eff: f64 = term
        .assignment
        .assignments()
        .column(0)
        .iter()
        .map(|&a| a * a)
        .sum();
    let lam = rho.lambda_smooth_vec();
    let spectrum = super::wbic_audit::recon_spectrum(
        &grams[0],
        &saved,
        n_eff,
        term.output_dim() as f64,
        disp,
        lam[0],
        Some(&term.atoms[0].smooth_penalty),
    )
    .unwrap();
    let top_signal = spectrum
        .reconstruction_energies()
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let vanished_edge = super::construction::RANK_VANISHED_REL * disp;
    assert!(
        top_signal > vanished_edge && vanished_edge > 0.0,
        "the fitted circle must begin alive before the vanishing-scale certificate: \
         top μ={top_signal:.6e}, vanished edge={vanished_edge:.6e}"
    );
    let vanished_scale = 0.5 * (vanished_edge / top_signal).sqrt();
    let certified_top_signal = vanished_scale * vanished_scale * top_signal;
    assert!(
        certified_top_signal < 0.5 * vanished_edge,
        "derived vanishing scale must retain its factor-four analytic threshold margin \
         after floating-point evaluation"
    );
    term.atoms[0]
        .decoder_coefficients
        .assign(&(&saved * vanished_scale));
    let d_vanish = term.per_atom_realised_rank_dof(&rho, disp).unwrap();
    eprintln!(
        "[rank-charge] vanishing (decoder×{vanished_scale:.3e}) \
         top μ≤{certified_top_signal:.3e} < {vanished_edge:.3e}; \
         d_eff={:.5} → charge=0 (neutral)",
        d_vanish[0],
    );
    assert_eq!(
        d_vanish[0], 0.0,
        "decoder certified below RANK_VANISHED_REL·R must give d_eff=0; got {:.4}",
        d_vanish[0],
    );
    term.atoms[0].decoder_coefficients.assign(&saved);
}

/// (iii) HEALTHY K=3 DECISION-LEVEL CONTROL (hand-built — the pipeline has no
/// healthy multi-atom fit pre-recovery): three well-separated clean rank-2
/// circles on disjoint output dims. Every atom must price as a clean rank-2
/// (d_eff ~4-6), and the canonical criterion must stay finite and
/// well-conditioned (no Schur collapse at K≥2).
#[test]
fn rank_charge_healthy_k3_control_well_conditioned() {
    let serial = k3_guard();
    let n = 96usize;
    let p = 18usize;
    let ncirc = 3usize;
    let mut s = 0x2101_C3C_0000_0009u64;
    let theta: Vec<Vec<f64>> = (0..n)
        .map(|_| {
            (0..ncirc)
                .map(|_| std::f64::consts::TAU * lcg(&mut s))
                .collect()
        })
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
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    term.set_guards_enabled(false);
    let mut rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); ncirc]);
    term.run_joint_fit_arrow_schur(x.view(), &mut rho, None, 60, 1.0, 1e-6, 1e-6)
        .expect("K=3 clean fit");

    let (criterion, loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(x.view(), &rho, None, 0, 1.0, 1e-6, 1e-6)
        .unwrap();
    let disp = term
        .reconstruction_dispersion(&loss, &cache, &rho, None)
        .unwrap();
    drop((loss, cache));
    let d_eff = term.per_atom_realised_rank_dof(&rho, disp).unwrap();
    eprintln!(
        "[rank-charge K=3] d_eff per atom = {:?}  disp={disp:.5}",
        d_eff
            .iter()
            .map(|v| (v * 100.0).round() / 100.0)
            .collect::<Vec<_>>()
    );
    for (k, &de) in d_eff.iter().enumerate() {
        assert!(
            de > 2.0 && de < 8.0,
            "K=3 atom {k}: every clean rank-2 circle must price ~4-6; got d_eff={de:.3}"
        );
    }
    assert!(
        criterion.is_finite(),
        "K=3 rank-charge criterion must stay finite (no Schur collapse): {criterion}"
    );
    drop(serial); // hold the K=3 serialisation lock across the whole fit
}

/// Build + fit a term with circles on the given output-dim indices (each circle
/// c on dims (2c, 2c+1)), against the shared target `x`. Used for leave-one-out
/// decision margins.
fn fit_circle_subset(
    x: &Array2<f64>,
    theta: &[Vec<f64>],
    circles: &[usize],
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
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
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
        AssignmentMode::ordered_beta_bernoulli(0.7, 1.0, false),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    term.set_guards_enabled(false);
    let mut rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1); circles.len()]);
    term.run_joint_fit_arrow_schur(x.view(), &mut rho, None, 60, 1.0, 1e-6, 1e-6)
        .expect("subset fit");
    (term, rho)
}

/// (iv) DECISION-LEVEL control: on a clean well-separated 3-circle fit, the
/// canonical criterion's leave-one-out margin of every real atom must be < 0
/// (KEEPING it is favored ⇒ accepted).
#[test]
fn rank_charge_k3_accepts_clean_atoms() {
    let serial = k3_guard();
    let n = 96usize;
    let p = 18usize;
    let ncirc = 3usize;
    let mut s = 0x2101_DEC_0000_0011u64;
    let theta: Vec<Vec<f64>> = (0..n)
        .map(|_| {
            (0..ncirc)
                .map(|_| std::f64::consts::TAU * lcg(&mut s))
                .collect()
        })
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
    // Compute each circle's leave-one-out margin:
    // margin_k = reml(all 3) − reml(drop k). <0 ⇒ keeping k is favored.
    let margins = || -> Vec<f64> {
        let (mut t3, r3) = fit_circle_subset(&x, &theta, &[0, 1, 2]);
        let (v3, _, _) = t3
            .penalized_quasi_laplace_criterion_with_cache(x.view(), &r3, None, 0, 1.0, 1e-6, 1e-6)
            .unwrap();
        (0..ncirc)
            .map(|drop| {
                let keep: Vec<usize> = (0..ncirc).filter(|&c| c != drop).collect();
                let (mut t2, r2) = fit_circle_subset(&x, &theta, &keep);
                let (v2, _, _) = t2
                    .penalized_quasi_laplace_criterion_with_cache(
                        x.view(),
                        &r2,
                        None,
                        0,
                        1.0,
                        1e-6,
                        1e-6,
                    )
                    .unwrap();
                v3 - v2 // margin_drop: <0 ⇒ the dropped circle is worth KEEPING
            })
            .collect()
    };
    // The K=3 joint fits use rayon parallel reductions whose order is thread-timing
    // dependent; the leave-one-out margin is a difference of two large independent
    // fits, which amplifies that into occasional sign flips under parallel test
    // execution. Pin the fits to a ONE-thread rayon pool so they converge to the
    // identical (correct) optimum every run — the single-thread values ARE the
    // optimum (verified stable across runs).
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread rayon pool for deterministic K=3 fits");
    let margins = pool.install(margins);
    eprintln!("[rank-charge K=3 decisions] leave-one-out margins={margins:?}");
    for (k, margin) in margins.iter().enumerate() {
        assert!(
            *margin < 0.0,
            "circle {k}: rank-charge must ACCEPT the real atom (margin<0); got {:.3}",
            margin
        );
    }
    // A spurious/noise atom is covered structurally by the vanishing
    // test (rank→0 → charge 0 → ΔEV rejects); a real atom here is never spurious.
    drop(serial); // hold the K=3 serialisation lock across the whole fit
}

/// (v) #9 DENSE-vs-STREAMING PARITY (the #9 correctness proof): the streaming
/// criterion must price the rank charge IDENTICALLY to the dense path. The load-
/// bearing invariant is that the streaming chunk-accumulated per-atom Grams +
/// effective sample sizes equal the dense `accumulate_decoder_gram`/`Σa²` (so the
/// shared `rank_dof_from_grams` returns the same d_eff), and the end-to-end
/// criterion values agree to ε.
#[test]
fn rank_charge_dense_streaming_parity() {
    let serial = k3_guard();
    let (mut term, rho) = fitted_circle_term(80, 16);
    let tgt = unit_target(&term);

    // Dense per-atom Grams + N_eff (what per_atom_realised_rank_dof builds).
    let mut dense_grams = term.empty_decoder_gram_accumulator();
    term.accumulate_decoder_gram(&mut dense_grams);
    let dense_n_eff: Vec<f64> = (0..term.k_atoms())
        .map(|k| {
            term.assignment
                .assignments()
                .column(k)
                .iter()
                .map(|&a| a * a)
                .sum()
        })
        .collect();

    // Streaming: pull the chunk-accumulated Grams + N_eff via the log-det pass.
    let mut ri = super::construction::StreamingRankInputs::default();
    term.streaming_exact_arrow_log_det(tgt.view(), &rho, None, Some(&mut ri))
        .expect("streaming log-det with rank inputs");

    assert_eq!(ri.grams.len(), dense_grams.len(), "atom count parity");
    for k in 0..dense_grams.len() {
        let (dg, sg) = (&dense_grams[k], &ri.grams[k]);
        let max_abs = dg
            .iter()
            .zip(sg.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        eprintln!(
            "[#9 parity] atom {k}: max|G_dense−G_stream|={max_abs:.3e}  N_eff dense={:.4} stream={:.4}",
            dense_n_eff[k], ri.n_eff[k]
        );
        assert!(
            max_abs < 1e-9,
            "atom {k}: streaming Gram must match dense (chunk-additive ΦᵀWΦ); max|Δ|={max_abs:.3e}"
        );
        assert!(
            (dense_n_eff[k] - ri.n_eff[k]).abs() < 1e-9,
            "atom {k}: streaming N_eff must match dense Σa²"
        );
    }

    // d_eff parity through the shared core (identical grams ⇒ identical count).
    let disp = 0.003_f64; // fixed R so both price against the same floor
    let d_dense = term
        .rank_dof_from_grams(&dense_grams, &dense_n_eff, &rho, disp)
        .unwrap();
    let d_stream = term
        .rank_dof_from_grams(&ri.grams, &ri.n_eff, &rho, disp)
        .unwrap();
    eprintln!("[#9 parity] d_eff dense={d_dense:?} stream={d_stream:?}");
    for k in 0..d_dense.len() {
        assert!(
            (d_dense[k] - d_stream[k]).abs() < 1e-9,
            "atom {k}: d_eff parity dense={} stream={}",
            d_dense[k],
            d_stream[k]
        );
    }

    // End-to-end canonical criterion parity: dense vs streaming to ε.
    let (v_dense, _, _) = term
        .penalized_quasi_laplace_criterion_with_cache(tgt.view(), &rho, None, 0, 1.0, 1e-6, 1e-6)
        .unwrap();
    let (v_stream, _) = term
        .penalized_quasi_laplace_criterion_streaming_exact(
            tgt.view(),
            &rho,
            None,
            0,
            1.0,
            1e-6,
            1e-6,
        )
        .unwrap();
    eprintln!("[#9 parity] criterion dense={v_dense:.6} stream={v_stream:.6}");
    assert!(
        (v_dense - v_stream).abs() < 1e-5,
        "dense vs streaming rank-charge criterion must agree: dense={v_dense} stream={v_stream}"
    );
    drop(serial); // hold the K=3 serialisation lock across the whole fit
}

/// (vii) #16 SHARED PRIMITIVE parity. The free `realised_rank_charge_dof` must price an
/// atom IDENTICALLY to the term-level `per_atom_realised_rank_dof` — the single source of
/// truth the #2023 tier PROMOTE/DEMOTE sites will both call, guaranteeing they adjudicate
/// in one currency.
#[test]
fn rank_charge_shared_primitive_parity() {
    let (mut term, rho) = fitted_circle_term(80, 16);
    let tgt = unit_target(&term);
    let (_v, loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(tgt.view(), &rho, None, 0, 1.0, 1e-6, 1e-6)
        .unwrap();
    let disp = term
        .reconstruction_dispersion(&loss, &cache, &rho, None)
        .unwrap();
    drop((loss, cache));

    // Term-level d_eff (the currency the joint REML charges).
    let d_term = term.per_atom_realised_rank_dof(&rho, disp).unwrap();

    // Free-fn d_eff from the SAME atom's gram/decoder/N_eff — must be bit-identical.
    let mut grams = term.empty_decoder_gram_accumulator();
    term.accumulate_decoder_gram(&mut grams);
    let n_eff: f64 = term
        .assignment
        .assignments()
        .column(0)
        .iter()
        .map(|&a| a * a)
        .sum();
    let lam = rho.lambda_smooth_vec();
    let d_free = super::construction::realised_rank_charge_dof(
        &grams[0],
        &term.atoms[0].decoder_coefficients,
        n_eff,
        term.output_dim() as f64,
        disp,
        lam[0],
        Some(&term.atoms[0].smooth_penalty),
    )
    .unwrap();
    eprintln!(
        "[#16 primitive] d_term={:.12} d_free={:.12}",
        d_term[0], d_free
    );
    assert_eq!(
        d_term[0], d_free,
        "shared realised_rank_charge_dof must match the term-level pricing bit-for-bit"
    );
}

/// (vii) #5 VETO — the blend-null null-license fix (recov matrix 12484591). A
/// zero-realised-rank atom (rank_eff==0 ⟺ d_eff==0) reconstructs nothing, so
/// its quasi-Laplace score is invalid (the β-Schur log-det → −∞ was letting
/// zero-‖B‖ atoms get born on a featureless residual), so the criterion must reject
/// it categorically (v → +∞) — not merely neutralise its charge. A real circle is
/// untouched (rank_eff=2).
#[test]
fn rank_charge_vetoes_zero_realised_rank_atom() {
    let (mut term, rho) = fitted_circle_term(80, 16);
    let tgt = unit_target(&term);
    let saved = term.atoms[0].decoder_coefficients.clone();

    // A real circle (rank_eff=2, d_eff≈5.5) remains finite.
    let (v_real, _, _) = term
        .penalized_quasi_laplace_criterion_with_cache(tgt.view(), &rho, None, 0, 1.0, 1e-6, 1e-6)
        .unwrap();
    eprintln!("[#5 veto] real circle v={v_real:.4} (finite, accepted)");
    assert!(
        v_real.is_finite(),
        "real rank-2 circle must NOT be vetoed: {v_real}"
    );

    // A vanishing decoder (×1e-6 → rank_eff=0, d_eff=0) is vetoed.
    term.atoms[0].decoder_coefficients.assign(&(&saved * 1e-6));
    let (v_vanish, _, _) = term
        .penalized_quasi_laplace_criterion_with_cache(tgt.view(), &rho, None, 0, 1.0, 1e-6, 1e-6)
        .unwrap();
    eprintln!("[#5 veto] vanishing atom v={v_vanish} (must be +∞)");
    assert!(
        v_vanish.is_infinite() && v_vanish > 0.0,
        "a zero-realised-rank (vanishing) atom must be VETOED to +∞; got {v_vanish}"
    );
    term.atoms[0].decoder_coefficients.assign(&saved);
}

/// The reconstruction target the fitted circle was built against (re-derived from
/// the same seed so the reml pass scores the real data).
fn unit_target(term: &SaeManifoldTerm) -> Array2<f64> {
    let n = term.n_obs();
    let p = term.output_dim();
    let mut s = 0x2101_B1C_0000_0005u64;
    let theta: Vec<f64> = (0..n)
        .map(|_| std::f64::consts::TAU * lcg(&mut s))
        .collect();
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

/// (viii) #2022/#2099 PIECEWISE SCALE-INSENSITIVITY of the evidence, rebuilt against HEAD's
/// architecture. The removed `fit_level_decoder_rescale_invariance_2099` gate held
/// the reconstruction IMAGE fixed under `B↦cB` via a compensating `s↦s−ln c`
/// amplitude — a symmetry that no longer exists on HEAD (the free amplitude dof was
/// removed; the inner walk is `(δt, δβ)` with `‖B‖` data-pinned). What DOES survive
/// as the evidence-side quotient is that the rank charge `½·d_eff·ln N_eff` is a
/// piecewise-constant rank count, not the removed continuously scale-dependent
/// `½·log(a²‖B‖²)` log-volume. Concretely `d_eff = rank_eff·basis_edf` is
/// INVARIANT under a uniform decoder rescale `D↦cD` on the resolved-rank plateau:
/// `basis_edf` reads only the (decoder-free) basis Gram, while each reconstruction
/// eigenvalue scales as `μ_k(c)=c²μ_k(1)` against the FIXED MP edge
/// `R·(1+√(p/N_eff))²`. Crossing `c_k=√(edge/μ_k)` therefore causes an honest,
/// monotone integer-rank transition; away from those crossings the charge is
/// exactly constant. The old log-volume proxy `½·ln‖D‖²` instead shifts
/// continuously by `ln c` under the same rescale — the scale degeneracy #2022
/// wanted out of the evidence. As `c→0`, the final transition is the separate
/// degeneracy cutoff `max μ(c)≤RANK_VANISHED_REL·R`, where `d_eff=0` and the
/// categorical veto applies. Convergence-independent (prices the primitive
/// directly, no fit).
#[test]
fn rank_charge_deff_is_piecewise_constant_with_monotone_scale_transitions_2099() {
    let (mut term, rho) = fitted_circle_term(80, 16);
    let tgt = unit_target(&term);
    let (_v, loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(tgt.view(), &rho, None, 0, 1.0, 1e-6, 1e-6)
        .unwrap();
    let disp = term
        .reconstruction_dispersion(&loss, &cache, &rho, None)
        .unwrap();
    drop((loss, cache));

    let mut grams = term.empty_decoder_gram_accumulator();
    term.accumulate_decoder_gram(&mut grams);
    let n_eff: f64 = term
        .assignment
        .assignments()
        .column(0)
        .iter()
        .map(|&a| a * a)
        .sum();
    let lam = rho.lambda_smooth_vec();
    let p_out = term.output_dim() as f64;
    let base_decoder = term.atoms[0].decoder_coefficients.clone();
    let d_eff = |decoder: &Array2<f64>| -> f64 {
        super::construction::realised_rank_charge_dof(
            &grams[0],
            decoder,
            n_eff,
            p_out,
            disp,
            lam[0],
            Some(&term.atoms[0].smooth_penalty),
        )
        .unwrap()
    };
    let spectrum = |decoder: &Array2<f64>| {
        super::wbic_audit::recon_spectrum(
            &grams[0],
            decoder,
            n_eff,
            p_out,
            disp,
            lam[0],
            Some(&term.atoms[0].smooth_penalty),
        )
        .unwrap()
    };

    let base_spectrum = spectrum(&base_decoder);
    let d0 = d_eff(&base_decoder);
    assert!(
        base_spectrum.mp_detection_rank() >= 1 && d0 > 0.0,
        "resolved circle must begin above the MP edge; rank={}, d_eff={d0}",
        base_spectrum.mp_detection_rank(),
    );

    // First certify the continuous part of the scaling law. The edge and EDF are
    // decoder-free, while every μ is homogeneous of degree two in the decoder.
    let n0: f64 = base_decoder.iter().map(|v| v * v).sum();
    for &c in &[0.5_f64, 2.0, 4.0] {
        let scaled = base_decoder.mapv(|v| c * v);
        let scaled_spectrum = spectrum(&scaled);
        assert_eq!(
            scaled_spectrum.mp_detection_edge(),
            base_spectrum.mp_detection_edge(),
            "fixed dispersion and aspect ratio imply a fixed MP edge"
        );
        assert_eq!(
            scaled_spectrum.basis_edf(),
            base_spectrum.basis_edf(),
            "basis EDF must not depend on decoder scale"
        );
        for (axis, (&mu_c, &mu_0)) in scaled_spectrum
            .reconstruction_energies()
            .iter()
            .zip(base_spectrum.reconstruction_energies().iter())
            .enumerate()
        {
            let expected = c * c * mu_0;
            let tolerance = 512.0 * f64::EPSILON * expected.max(base_spectrum.mp_detection_edge());
            assert!(
                (mu_c - expected).abs() <= tolerance,
                "decoder scaling law failed on mode {axis}: \
                 μ(c)={mu_c:.17e}, c²μ(1)={expected:.17e}, tolerance={tolerance:.3e}"
            );
        }
        let nc: f64 = scaled.iter().map(|v| v * v).sum();
        let old_proxy_shift = 0.5 * (nc.ln() - n0.ln());
        eprintln!(
            "[#2099 scale law] c={c:>4}: hard rank={} | old ½log‖B‖² shift={old_proxy_shift:+.4}",
            scaled_spectrum.mp_detection_rank(),
        );
        assert!(
            old_proxy_shift.abs() > 0.1,
            "sanity: the old log-volume proxy MUST be scale-dependent (shift {old_proxy_shift})"
        );
    }

    let top_mu = base_spectrum
        .reconstruction_energies()
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let vanished_edge = super::construction::RANK_VANISHED_REL * disp;
    let vanished_transition = (vanished_edge / top_mu).sqrt();
    let mut hard_transitions: Vec<f64> = base_spectrum
        .reconstruction_energies()
        .iter()
        .copied()
        .filter(|&mu| mu > 0.0)
        .map(|mu| (base_spectrum.mp_detection_edge() / mu).sqrt())
        .collect();
    hard_transitions.sort_by(f64::total_cmp);
    assert!(
        vanished_transition < hard_transitions[0],
        "the degeneracy boundary must lie below the first MP detection crossing"
    );

    // The base scale's plateau is bounded by the nearest MP crossings. Probe two
    // interior scales, never a guessed multiplier that might cross an edge.
    let lower = hard_transitions
        .iter()
        .copied()
        .filter(|&transition| transition < 1.0)
        .fold(0.0_f64, f64::max);
    let upper = hard_transitions
        .iter()
        .copied()
        .find(|&transition| transition >= 1.0)
        .unwrap_or(4.0);
    assert!(lower > 0.0 && upper > 1.0);
    for c in [lower.sqrt(), upper.sqrt()] {
        let d_c = d_eff(&base_decoder.mapv(|value| c * value));
        assert_eq!(
            d_c, d0,
            "d_eff must be exactly invariant inside one no-crossing plateau \
             (c={c:.6e}, plateau=({lower:.6e}, {upper:.6e}))"
        );
    }

    // Finally probe one point in every non-empty interval cut out by the
    // categorical and MP thresholds. The production value must equal the rank
    // predicted directly from μ(c)=c²μ(1), and those ranks may only increase as
    // decoder scale increases. Near-coincident MP crossings intentionally have
    // no unstable microscopic probe between them; the adjacent safe intervals
    // still certify their combined integer jump.
    let mut transitions = vec![vanished_transition];
    transitions.extend(hard_transitions.iter().copied());
    transitions.sort_by(f64::total_cmp);
    let mut probes = vec![0.5 * transitions[0]];
    for pair in transitions.windows(2) {
        if pair[1] > pair[0] * (1.0 + 1.0e-8) {
            probes.push((pair[0] * pair[1]).sqrt());
        }
    }
    probes.push(2.0 * transitions[transitions.len() - 1]);

    let mut previous_rank = 0usize;
    let mut saw_alive_below_mp = false;
    for c in probes {
        let c2 = c * c;
        let hard_rank = base_spectrum
            .reconstruction_energies()
            .iter()
            .filter(|&&mu| c2 * mu > base_spectrum.mp_detection_edge())
            .count();
        let expected_rank = if hard_rank > 0 {
            hard_rank
        } else if c2 * top_mu > vanished_edge {
            saw_alive_below_mp = true;
            1
        } else {
            0
        };
        let d_c = d_eff(&base_decoder.mapv(|value| c * value));
        let expected_d_eff = expected_rank as f64 * base_spectrum.basis_edf();
        let tolerance = 128.0 * f64::EPSILON * expected_d_eff.max(1.0);
        eprintln!("[#2099 transition] c={c:.6e}: expected rank={expected_rank}, d_eff={d_c:.12}");
        assert!(
            (d_c - expected_d_eff).abs() <= tolerance,
            "rank-charge transition mismatch at c={c:.6e}: \
             d_eff={d_c:.17e}, expected={expected_d_eff:.17e}"
        );
        assert!(
            expected_rank >= previous_rank,
            "rank charge must be monotone under increasing positive decoder scale"
        );
        previous_rank = expected_rank;
    }
    assert!(
        saw_alive_below_mp,
        "the probes must cover the alive-but-below-MP rank-1 regime"
    );
}

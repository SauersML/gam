//! #2111 INTEGRATED dense-torus acceptance test — the end-to-end bar the issue
//! specifies, closing the gap between the ISA producer / pair-κ machinery (unit-
//! tested) and the full birth pipeline (`fit_stagewise`).
//!
//! FIXTURE. A dense product-of-circles torus: `k = 6` circles on DISJOINT
//! axis-aligned output frames (circle `c` on dims `(2c, 2c+1)`), every row on
//! EVERY circle (density 6 — the degenerate regime where single-plane ring-ness
//! fails and only 4th-order ISA separates the factors), distinct amplitudes
//! `1.0 … 0.55`, independent angles, small isotropic noise. This is the
//! `probe_2101_birth_locus_disjoint_6circle_ibp` structure at a sample size
//! (`n = 700`) clear of the dense-case small-sample floor (`n ≥ 300`; the
//! gated-edge `ISA_SUBSAMPLE_FLOOR` resolution bound concerns `q → ½` gates,
//! not this density-1 fixture).
//!
//! PIPELINE. Seed a single K=1 circle atom on circle 0's true coordinate, then run
//! the integrated forward-birth + backfit engine [`fit_stagewise`]. On a disjoint
//! residual the shared-factor model is rank-0, so births fall through to the ISA
//! fallback seed (`residual_principal_birth_candidate` → `isa_extract_certified_plane`)
//! — the exact machinery #2111 specifies. The seed contributes circle 0; the ISA
//! births must recover circles 1–5 from the blended dense residual.
//!
//! ACCEPTANCE BAR (#2111). 6 born atoms; every atom's decoder output-plane matches
//! a distinct true circle at overlap ≥ 0.9; every decoder is clean (singular-value
//! participation ratio ≤ 3 — a rank-2 circle decoder has PR ≈ 2); `n_distinct = 6`,
//! `n_clean = 6` (best overlap ≥ 0.9 AND second-best ≤ 0.2); and the forward phase
//! exits NATURALLY (`stopped_reason != MaxBirths`).
//!
//! If the bar is not met, the printed per-atom overlap / PR table + birth ledger
//! localise WHICH stage drops the ball (ISA rotation, birth acceptance, or joint
//! backfit) — the failure mode is the finding.

use crate::manifold::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm, StagewiseConfig, StagewiseStop,
    fit_stagewise,
};
use gam_linalg::faer_ndarray::FaerSvd;
use gam_terms::latent::LatentManifold;
use ndarray::{Array1, Array2, ArrayView2};
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

/// Dense product-of-circles torus target (`n × p`) + the true axis-aligned circle
/// planes. Circle `c` lives on dims `(2c, 2c+1)` with amplitude `amps[c]` and an
/// independent per-row angle; every row carries every circle (dense).
fn dense_torus(
    n: usize,
    p: usize,
    k: usize,
    amps: &[f64],
    sigma: f64,
    seed: u64,
) -> (Array2<f64>, Vec<Array2<f64>>, Vec<Array2<f64>>) {
    assert!(p >= 2 * k && amps.len() == k);
    let mut s = seed;
    let mut data = Array2::<f64>::zeros((n, p));
    // Per-circle true per-row angle (turns) — kept to seed circle 0 and to sanity
    // the fixture; the fitter never sees the angles of circles 1..k.
    let mut turns = vec![Array2::<f64>::zeros((n, 1)); k];
    for i in 0..n {
        for c in 0..k {
            let t = lcg(&mut s);
            turns[c][[i, 0]] = t;
            let ang = std::f64::consts::TAU * t;
            data[[i, 2 * c]] += amps[c] * ang.cos();
            data[[i, 2 * c + 1]] += amps[c] * ang.sin();
        }
        for j in 0..p {
            data[[i, j]] += sigma * lcg_normal(&mut s);
        }
    }
    let true_planes: Vec<Array2<f64>> = (0..k)
        .map(|c| {
            Array2::from_shape_fn(
                (p, 2),
                |(row, col)| {
                    if row == 2 * c + col { 1.0 } else { 0.0 }
                },
            )
        })
        .collect();
    (data, true_planes, turns)
}

/// Top-2 right-singular (output-space) plane of a decoder `D` (`m × p`): the two
/// leading rows of `Vᵀ` transposed into a `(p, 2)` orthonormal basis. This is the
/// ambient 2-plane the atom reconstructs into.
fn decoder_output_plane(decoder: &Array2<f64>) -> Array2<f64> {
    let (_, _, vt) = decoder
        .svd(false, true)
        .expect("decoder svd for output plane");
    let vt = vt.expect("vt present");
    let p = decoder.ncols();
    Array2::from_shape_fn((p, 2), |(row, col)| vt[[col, row]])
}

/// Singular-value participation ratio `PR = (Σσ²)² / Σσ⁴` of a decoder. A clean
/// rank-2 circle decoder (two comparable singular values, the rest ~0) gives
/// PR ≈ 2; a blended / higher-rank decoder gives PR > 3.
fn decoder_sv_pr(decoder: &Array2<f64>) -> f64 {
    let (_, sv, _) = decoder.svd(false, false).expect("decoder svd for PR");
    let s2: Vec<f64> = sv.iter().map(|&s| s * s).collect();
    let num: f64 = s2.iter().sum::<f64>().powi(2);
    let den: f64 = s2.iter().map(|v| v * v).sum::<f64>();
    if den > 0.0 { num / den } else { 0.0 }
}

/// Subspace affinity `‖UᵀV‖²_F / 2 ∈ [0, 1]` between two orthonormal `(p, 2)`
/// planes (1 = identical plane).
fn plane_overlap(u: &Array2<f64>, v: &Array2<f64>) -> f64 {
    let m = u.t().dot(v);
    m.iter().map(|x| x * x).sum::<f64>() / 2.0
}

/// A degree-1 periodic circle atom whose decoder plants cos→`dir_a`, sin→`dir_b`.
fn circle_atom(
    name: &str,
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    dir_a: usize,
    dir_b: usize,
    p: usize,
) -> SaeManifoldAtom {
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
    let mut decoder = Array2::<f64>::zeros((3, p));
    decoder[[1, dir_a % p]] = 1.0;
    decoder[[2, dir_b % p]] = 1.0;
    SaeManifoldAtom::new(
        name.to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_second_jet(evaluator.clone())
}

/// K=1 seed term: circle 0 on its true coordinate, active on every row.
fn seed_term(coords0: &Array2<f64>, p: usize) -> (SaeManifoldTerm, SaeManifoldRho) {
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let atom = circle_atom("seed_c0", &evaluator, coords0, 0, 1, p);
    let n = coords0.nrows();
    let logits = Array2::<f64>::from_elem((n, 1), 6.0);
    // IBP (independent per-atom Bernoulli) assignment — NOT softmax. The dense
    // torus has every row co-active on EVERY circle; softmax forces per-row
    // competition (probabilities sum to 1) and its entropy sparsity prior
    // maximally penalises exactly the dense co-activation the fixture requires,
    // so no born circle can clear the frozen-ρ evidence gate. This matches the
    // `probe_2101_birth_locus_disjoint_6circle_ibp` structure this test
    // reproduces (see the module docstring) and every sibling circle-recovery
    // fixture (#2027/#2089/#2101), all of which seed `ibp_map(0.7, 1.0, false)`.
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords0.clone()],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(0.7, 1.0, false),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    term.set_guards_enabled(false);
    // Charge births on the occupancy-aware BIC rank scale (½·d_eff·ln N_eff),
    // not the raw per-row coordinate-block Laplace log-det ½log|H_tt|. At n=700
    // the raw coordinate log-det grows ≈ O(n) per born atom with NO compensating
    // occam offset (λ_smooth = 1 ⇒ occam = 0), so it dwarfs the deviance gain and
    // the frozen-ρ birth gate rejects every genuinely-good circle (measured: a
    // clean rank-2 circle that lifts EV 0.27→0.43 raised REML by +391 purely
    // through +598 of uncompensated ½log|H_tt|). The rank charge is the honest,
    // rotation-invariant Laplace complexity for a realised-rank decoder and is
    // exactly the decoder-scale-mispricing remedy this path documents. The flag
    // propagates to every fit_stagewise clone (see term-clone), so setting it on
    // the seed engages it for the whole forward-birth sweep.
    term.set_rank_charge_evidence(true);
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    (term, rho)
}

/// Score every born atom against the true planes; returns
/// `(n_distinct, n_clean, n_real, per_atom rows)` where each row is
/// `(best_overlap, second_overlap, best_circle, pr)`.
fn score_atoms(
    term: &SaeManifoldTerm,
    true_planes: &[Array2<f64>],
) -> (usize, usize, usize, Vec<(f64, f64, usize, f64)>) {
    let mut claimed = std::collections::HashSet::new();
    let (mut n_real, mut n_clean) = (0usize, 0usize);
    let mut rows = Vec::new();
    for k in 0..term.k_atoms() {
        let decoder = &term.atoms[k].decoder_coefficients;
        let plane = decoder_output_plane(decoder);
        let pr = decoder_sv_pr(decoder);
        let mut ov: Vec<(f64, usize)> = true_planes
            .iter()
            .enumerate()
            .map(|(idx, tp)| (plane_overlap(&plane, tp), idx))
            .collect();
        ov.sort_by(|a, b| b.0.total_cmp(&a.0));
        let best = ov[0].0;
        let second = ov.get(1).map(|x| x.0).unwrap_or(0.0);
        claimed.insert(ov[0].1);
        if best >= 0.9 {
            n_real += 1;
            if second <= 0.2 {
                n_clean += 1;
            }
        }
        rows.push((best, second, ov[0].1, pr));
    }
    (claimed.len(), n_clean, n_real, rows)
}

/// THE #2111 INTEGRATED ACCEPTANCE TEST.
#[test]
fn dense_torus_integrated_birth_recovery_2111() {
    let k = 6usize;
    let n = 700usize;
    let p = 16usize;
    // Distinct amplitudes 1.0 … 0.55 (the real fixture's identifying signal).
    let amps: Vec<f64> = (0..k)
        .map(|c| 1.0 - 0.45 * (c as f64) / ((k - 1) as f64))
        .collect();
    let (data, true_planes, turns) = dense_torus(n, p, k, &amps, 0.05, 0x2111_D0_7A_5EED);

    let config = StagewiseConfig {
        inner_max_iter: 24,
        learning_rate: 1.0,
        ridge_ext_coord: 1e-6,
        ridge_beta: 1e-6,
        max_births: 10, // safety BOUND (> k so a natural stop is observable)
        max_backfit_sweeps: 3,
        min_effect_ev: 0.0,
        max_factor_rank: 4,
        structured_whitening: false,
    };

    // Seed circle 0 on its true coordinate; fit the K=1 seed before SAC entry.
    let (mut seed, mut rho) = seed_term(&turns[0], p);
    seed.run_joint_fit_arrow_schur(
        data.view(),
        &mut rho,
        None,
        config.inner_max_iter,
        config.learning_rate,
        config.ridge_ext_coord,
        config.ridge_beta,
    )
    .expect("K=1 seed fit must complete");

    let result = fit_stagewise(seed, rho, data.view(), None, None, &config, None, None)
        .expect("fit_stagewise must complete on the dense torus");

    let (n_distinct, n_clean, n_real, rows) = score_atoms(&result.term, &true_planes);
    let natural_exit = result.report.stopped_reason != StagewiseStop::MaxBirths;

    eprintln!(
        "\n[#2111 dense torus] K={} births_accepted={} births_rejected={} stop={:?} natural_exit={}",
        result.term.k_atoms(),
        result.report.births_accepted,
        result.report.births_rejected,
        result.report.stopped_reason,
        natural_exit,
    );
    eprintln!("[#2111] per-atom decoder plane vs true circles:");
    eprintln!("   atom  best_overlap  second_overlap  best_circle   sv_PR");
    for (k, (best, second, circ, pr)) in rows.iter().enumerate() {
        eprintln!("   {k:>4}  {best:>12.4}  {second:>14.4}  {circ:>11}   {pr:>6.3}");
    }
    eprintln!(
        "[#2111] n_distinct={n_distinct} n_real={n_real} n_clean={n_clean}  (bar: 6/6/6, all PR<=3, natural_exit)"
    );
    eprintln!("[#2111] birth ledger:");
    for (r, br) in result.report.birth_records.iter().enumerate() {
        eprintln!(
            "   round {r}: accepted={} kind={:?} dEV={:.5} factor_energy={:.5}",
            br.accepted, br.kind, br.delta_ev, br.factor_energy
        );
    }

    // ── The #2111 acceptance bar ────────────────────────────────────────────────
    let all_real = rows.iter().all(|(b, _, _, _)| *b >= 0.9);
    let all_clean = rows.iter().all(|(b, s, _, _)| *b >= 0.9 && *s <= 0.2);
    let all_pr_ok = rows.iter().all(|(_, _, _, pr)| *pr <= 3.0);
    assert!(
        result.term.k_atoms() == k,
        "expected K={k} born atoms, got {}",
        result.term.k_atoms()
    );
    assert!(
        all_real,
        "every atom must match a true circle at overlap >= 0.9"
    );
    assert!(
        all_clean,
        "every atom must be CLEAN (best>=0.9 AND second<=0.2)"
    );
    assert!(
        all_pr_ok,
        "every decoder must have SV participation ratio <= 3"
    );
    assert!(n_distinct == k, "n_distinct must be {k}; got {n_distinct}");
    assert!(n_real == k, "n_real must be {k}; got {n_real}");
    assert!(n_clean == k, "n_clean must be {k}; got {n_clean}");
    assert!(
        natural_exit,
        "forward phase must exit naturally (not MaxBirths)"
    );
}

/// Fixture sanity (fast, no fit): the planted dense torus really carries `2k`
/// above-noise directions in the expected axis-aligned frames — a guard that the
/// integrated test above is exercising the intended structure, not a degenerate
/// input. Uses the shared column-second-moment eigenstructure.
#[test]
fn dense_torus_fixture_has_2k_signal_dirs_2111() {
    let k = 6usize;
    let (data, _planes, _turns) = dense_torus(700, 16, k, &vec![1.0; k], 0.05, 0x2111_F1F7);
    let signal = column_signal_rank(data.view(), 0.05 * 0.05);
    eprintln!(
        "[#2111 fixture] above-noise signal directions = {signal} (expect {})",
        2 * k
    );
    assert!(
        signal == 2 * k,
        "dense {k}-torus must show exactly {} signal directions; got {signal}",
        2 * k
    );
}

/// Count column-covariance eigenvalues above an isotropic-noise Marchenko–Pastur
/// edge — the number of real signal directions in the centered data.
fn column_signal_rank(data: ArrayView2<'_, f64>, noise_var: f64) -> usize {
    use gam_linalg::faer_ndarray::FaerEigh;
    let (n, p) = data.dim();
    let mut mean = Array1::<f64>::zeros(p);
    for i in 0..n {
        for j in 0..p {
            mean[j] += data[[i, j]];
        }
    }
    mean.mapv_inplace(|v| v / n as f64);
    let mut cov = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        for a in 0..p {
            let ra = data[[i, a]] - mean[a];
            for b in a..p {
                cov[[a, b]] += ra * (data[[i, b]] - mean[b]);
            }
        }
    }
    for a in 0..p {
        for b in a..p {
            let v = cov[[a, b]] / n as f64;
            cov[[a, b]] = v;
            cov[[b, a]] = v;
        }
    }
    let (evals, _) = cov.eigh(crate::manifold::Side::Lower).expect("cov eigh");
    let edge = noise_var * (1.0 + (p as f64 / n as f64).sqrt()).powi(2);
    evals.iter().filter(|&&e| e > edge).count()
}

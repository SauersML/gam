//! #981 — the report's own falsification (the load-bearing test).
//!
//! Two replicate fits of the SAME planted single-circle SAE problem, run with
//! different row orders and different latent-coordinate seed gauges, must agree
//! **up to exactly the group the residual-gauge certificate reports — no more,
//! no less**:
//!
//! * Both replicates' certificates ([`residual_gauge`] via the production
//!   `fit_diagnostics_report` lowering) must name the SAME group
//!   (`group_signature()` equality), under the same named metric (Euclidean —
//!   the smallest honest scope: single atom, no harvested metric, no isometry
//!   pin, so the certificate escalates to `diffeomorphism-unpinned` and the
//!   reported group contains the circle's `Isom(S¹) = O(2)` we align by).
//! * The replicates must **NOT** agree raw: the latent circle coordinates of
//!   the two fits differ by a genuine gauge element (the seeds are offset by a
//!   reflection + phase shift the objective cannot see), so a vacuous test
//!   that skipped the alignment would fail here — this assertion makes the
//!   alignment load-bearing.
//! * After aligning by the reported gauge — an O(2) Procrustes on the
//!   `(cos θ, sin θ)` circle embedding, exactly the `Isom(S¹)` factor of the
//!   certified group — the replicates' coordinates must agree.
//! * The orbit invariants (fitted reconstructions) must agree WITHOUT any
//!   alignment: the gauge lives only in the coordinates, never in the fit.
//!
//! The fits run the production outer engine (`OuterProblem::run` around
//! `SaeManifoldOuterObjective`), exactly like the two-circle recovery pin.

use faer::Side as FaerSide;
use gam::inference::row_metric::MetricProvenance;
use gam::linalg::faer_ndarray::{FaerCholesky, FaerSvd, fast_ata, fast_atb};
use gam::solver::rho_optimizer::OuterProblem;
use gam::terms::latent::LatentManifold;
use gam::terms::sae::identifiability::GeneratorFamily;
use gam::terms::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

// ---- production defaults (gamfit `sae_manifold_fit`, ibp_map path) ---------
const N: usize = 200;
const P: usize = 12;
const M: usize = 3; // const + 1 harmonic (sin, cos) -> circle
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const SPARSITY: f64 = 1.0;
const SMOOTHNESS: f64 = 1.0;
const INNER_MAX_ITER: usize = 50;
const LEARNING_RATE: f64 = 1.0;
const RIDGE_EXT_COORD: f64 = 1.0e-6;
const RIDGE_BETA: f64 = 1.0e-6;

// ---- planted DGP ------------------------------------------------------------
const RADIUS: f64 = 1.0;
/// Replicate A latent seed gauge: identity + small phase shift.
const SEED_SHIFT_A: f64 = 0.05;
/// Replicate B latent seed gauge: reflection through 0.20 (θ ↦ 0.40 − θ),
/// a genuine non-identity element of the circle's O(2) gauge.
const SEED_REFLECT_B: f64 = 0.40;
/// Deterministic row-order stride for replicate B (coprime to N = 200).
const ROW_STRIDE_B: usize = 7;

/// Deterministic Lehmer-style uniform in [0,1) keyed purely by index.
fn idx_uniform(seed: u64) -> f64 {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((state >> 11) as f64) * f64::from_bits(0x3CA0000000000000)
}

/// One planted P×2 orthonormal frame (Gram-Schmidt of two deterministic
/// full-rank ambient vectors).
fn planted_frame() -> Array2<f64> {
    let mut raw = Array2::<f64>::zeros((P, 2));
    for j in 0..2 {
        for i in 0..P {
            raw[[i, j]] = ((i as f64 + 1.0) * 0.37 * (j as f64 + 1.0)).sin()
                + 0.5 * ((i as f64) * 0.11 - (j as f64) * 0.9).cos();
        }
    }
    let mut q = Array2::<f64>::zeros((P, 2));
    for j in 0..2 {
        let mut v = raw.column(j).to_owned();
        for prev in 0..j {
            let qp = q.column(prev);
            let dot: f64 = qp.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            for i in 0..P {
                v[i] -= dot * qp[i];
            }
        }
        let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for i in 0..P {
            q[[i, j]] = v[i] / nrm;
        }
    }
    q
}

/// Planted truth: every row active on the single circle atom.
fn planted_theta_amp() -> (Vec<f64>, Vec<f64>) {
    let mut theta = vec![0.0_f64; N];
    let mut amp = vec![0.0_f64; N];
    for i in 0..N {
        theta[i] = ((i as f64) * 0.061_803 + 0.13).rem_euclid(1.0);
        amp[i] = 0.85 + 0.30 * idx_uniform(i as u64 * 2 + 1);
    }
    (theta, amp)
}

/// Planted response Z[i] = amp_i · R · (cosθ_i u_1 + sinθ_i u_2) + noise.
fn planted_response(theta: &[f64], amp: &[f64], frame: &Array2<f64>) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((N, P));
    let mut signal_sq = 0.0_f64;
    for i in 0..N {
        let ang = std::f64::consts::TAU * theta[i];
        let (c, sn) = (ang.cos(), ang.sin());
        let scale = amp[i] * RADIUS;
        for col in 0..P {
            let contrib = scale * (c * frame[[col, 0]] + sn * frame[[col, 1]]);
            z[[i, col]] += contrib;
            signal_sq += contrib * contrib;
        }
    }
    let sigma = 0.03 * (signal_sq / (N * P) as f64).sqrt();
    for i in 0..N {
        for col in 0..P {
            let u = idx_uniform(((i * P + col) as u64) * 7 + 3).max(1.0e-12);
            let u2 = idx_uniform(((i * P + col) as u64) * 7 + 5);
            let g = (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            z[[i, col]] += sigma * g;
        }
    }
    z
}

/// Production cold decoder init for the single-atom ibp_map path: weighted LSQ
/// at the zero-logit gate `a = σ(0) = 0.5` (the K = 1 specialization of the
/// pyffi `sae_decoder_lsq_init`, whose residual-seed logits are identically
/// zero for a single atom).
fn decoder_lsq_init_single(phi: &Array2<f64>, z: &Array2<f64>) -> Array2<f64> {
    let x = phi * 0.5;
    let mut xtx = fast_ata(&x);
    let mut trace = 0.0_f64;
    for i in 0..M {
        trace += xtx[[i, i]];
    }
    let jitter = (trace / M as f64).max(1.0) * 1.0e-8;
    for i in 0..M {
        xtx[[i, i]] += jitter;
    }
    let xtz = fast_atb(&x, z);
    xtx.cholesky(FaerSide::Lower)
        .expect("decoder LSQ Cholesky")
        .solve_mat(&xtz)
}

/// Fit the single-circle SAE through the production outer engine on the given
/// row matrix, seeding the latent circle coordinates with `coord_seed`
/// (one value per row of `z`, in `z`'s row order). Returns the fitted term.
fn run_production_fit(z: &Array2<f64>, coord_seed: &[f64], label: &str) -> SaeManifoldTerm {
    let evaluator = PeriodicHarmonicEvaluator::new(M).expect("evaluator");
    let coords = Array2::from_shape_fn((N, 1), |(i, _)| coord_seed[i]);
    let (phi, jet) = evaluator.evaluate(coords.view()).expect("basis evaluation");
    let decoder = decoder_lsq_init_single(&phi, z);
    let atom = SaeManifoldAtom::new(
        "circle".to_string(),
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(M),
    )
    .expect("atom")
    .with_basis_evaluator(Arc::new(
        PeriodicHarmonicEvaluator::new(M).expect("evaluator"),
    ));
    // Single atom ⇒ the production residual-energy seed logits are identically
    // zero (the k ≤ 1 early return of `sae_residual_seed_logits`).
    let logits = Array2::<f64>::zeros((N, 1));
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(TAU, ALPHA, false),
    )
    .expect("assignment");
    let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");

    let init_rho = SaeManifoldRho::new(
        SPARSITY.ln(),
        SMOOTHNESS.ln(),
        vec![Array1::<f64>::zeros(0); 1],
    );
    let init_rho_flat = init_rho.to_flat();
    let n_params = init_rho_flat.len();
    let mut objective = SaeManifoldOuterObjective::new(
        term,
        z.clone(),
        None,
        init_rho,
        INNER_MAX_ITER,
        LEARNING_RATE,
        RIDGE_EXT_COORD,
        RIDGE_BETA,
    );
    let problem = OuterProblem::new(n_params).with_initial_rho(init_rho_flat);
    problem
        .run(&mut objective, label)
        .expect("outer cascade must complete");
    objective
        .into_fitted()
        .expect("outer fit was evaluated")
        .term
}

/// `(cos 2πθ, sin 2πθ)` circle embedding of a coordinate vector (period 1).
fn embed(theta: &[f64]) -> Array2<f64> {
    let mut e = Array2::<f64>::zeros((theta.len(), 2));
    for (i, &t) in theta.iter().enumerate() {
        let a = std::f64::consts::TAU * t;
        e[[i, 0]] = a.cos();
        e[[i, 1]] = a.sin();
    }
    e
}

/// Raw (unaligned) agreement: mean over rows of the embedding dot product,
/// i.e. mean cos of the per-row angular difference. 1 ⇒ identical coordinates.
fn unaligned_mean_cos(a: &[f64], b: &[f64]) -> f64 {
    let ea = embed(a);
    let eb = embed(b);
    let mut acc = 0.0_f64;
    for i in 0..a.len() {
        acc += ea[[i, 0]] * eb[[i, 0]] + ea[[i, 1]] * eb[[i, 1]];
    }
    acc / a.len() as f64
}

/// Agreement after aligning `b` to `a` by the circle's O(2) gauge: the
/// orthogonal Procrustes rotation/reflection `R = U Vᵀ` from the SVD of
/// `e_bᵀ e_a` — exactly the `Isom(S¹)` factor of the certified residual gauge.
/// Returns the mean cos of the aligned per-row angular difference.
fn o2_aligned_mean_cos(a: &[f64], b: &[f64]) -> f64 {
    let ea = embed(a);
    let eb = embed(b);
    let cross = fast_atb(&eb, &ea);
    let (u, sv, vt) = cross.svd(true, true).expect("Procrustes SVD");
    let u = u.expect("U");
    let vt = vt.expect("Vt");
    // A near-zero second singular value would mean the embeddings do not span
    // the circle and the O(2) alignment is ill-posed — the planted angles fill
    // the circle, so the alignment must be well-determined.
    assert!(
        sv[1] > 1.0e-3 * sv[0].max(1.0e-300),
        "O(2) Procrustes alignment must be well-posed (singular values {sv:?})"
    );
    let aligned = eb.dot(&u.dot(&vt));
    let mut acc = 0.0_f64;
    for i in 0..a.len() {
        acc += aligned[[i, 0]] * ea[[i, 0]] + aligned[[i, 1]] * ea[[i, 1]];
    }
    acc / a.len() as f64
}

#[test]
fn replicate_fits_agree_up_to_exactly_the_reported_gauge() {
    let frame = planted_frame();
    let (theta_true, amp) = planted_theta_amp();
    let z = planted_response(&theta_true, &amp, &frame);

    // ---- replicate A: natural row order, identity-ish seed gauge ----------
    let seed_a: Vec<f64> = theta_true
        .iter()
        .map(|t| (t + SEED_SHIFT_A).rem_euclid(1.0))
        .collect();
    let term_a = run_production_fit(&z, &seed_a, "SAE replicate gauge agreement (A)");

    // ---- replicate B: strided row order, reflected+shifted seed gauge -----
    let perm: Vec<usize> = (0..N).map(|r| (r * ROW_STRIDE_B + 3) % N).collect();
    let mut z_b = Array2::<f64>::zeros((N, P));
    let mut seed_b = vec![0.0_f64; N];
    for (r, &src) in perm.iter().enumerate() {
        z_b.row_mut(r).assign(&z.row(src));
        seed_b[r] = (SEED_REFLECT_B - theta_true[src]).rem_euclid(1.0);
    }
    let term_b = run_production_fit(&z_b, &seed_b, "SAE replicate gauge agreement (B)");

    // ---- fitted circle coordinates, replicate B mapped back to natural rows
    let coords_a = term_a.assignment.coords[0].as_matrix();
    let coords_b_fit = term_b.assignment.coords[0].as_matrix();
    let theta_a: Vec<f64> = (0..N).map(|i| coords_a[[i, 0]]).collect();
    let mut theta_b = vec![0.0_f64; N];
    for (r, &src) in perm.iter().enumerate() {
        theta_b[src] = coords_b_fit[[r, 0]];
    }

    // ---- residual-gauge certificates (production lowering, Euclidean, no
    // isometry pin: the smallest honest scope) ------------------------------
    let report_a = term_a
        .fit_diagnostics_report(None, false, None, None)
        .expect("replicate A diagnostics")
        .residual_gauge;
    let report_b = term_b
        .fit_diagnostics_report(None, false, None, None)
        .expect("replicate B diagnostics")
        .residual_gauge;

    println!("replicate A: {}", report_a.summary);
    println!("replicate B: {}", report_b.summary);

    // Both certificates computed in the same named metric.
    assert_eq!(report_a.metric_provenance, MetricProvenance::Euclidean);
    assert_eq!(report_b.metric_provenance, MetricProvenance::Euclidean);

    // The two replicates must be identified up to EXACTLY the same group.
    assert_eq!(
        report_a.group_signature(),
        report_b.group_signature(),
        "replicate fits of the same planted problem must be certified up to the \
         same residual gauge group.\nA: {}\nB: {}",
        report_a.summary,
        report_b.summary
    );

    // No isometry pin was installed ⇒ both reports must honestly escalate to
    // diffeomorphism-unpinned; the circle's Isom(S¹) = O(2) we align by below
    // is contained in that reported freedom (O(2) ⊂ Diff(S¹)). A certificate
    // claiming the latent parameterization pinned here would be the rung-2
    // theorem's conditions violated — exactly what this test exists to catch.
    assert!(
        report_a.diffeomorphism_unpinned && report_b.diffeomorphism_unpinned,
        "without an isometry pin both certificates must escalate to \
         diffeomorphism-unpinned (the reported group must contain the O(2) \
         gauge the replicates actually differ by).\nA: {}\nB: {}",
        report_a.summary,
        report_b.summary
    );

    // ---- #998 oracle: the U(1) phase freedom is an EXACT data-null --------
    // With no isometry pin installed, the production path certifies the
    // circle's within-atom gauge on its exact orbit in (decoder, coordinate)
    // space: the harmonic basis is closed under phase shifts, so the
    // LS-compensated orbit motion is a data-null by construction — the
    // verdict must be unpinned with the compensation residual at the noise
    // floor and NO lowering-error calibration involved (scale 0; nothing is
    // compressed on this path). This closes the loop the frame certificate
    // could only refuse: the freedom the replicates demonstrably differ by
    // (the alignment block below) is now positively certified, exactly.
    for (label, report) in [("A", &report_a), ("B", &report_b)] {
        let phase = report
            .generators
            .iter()
            .find(|g| {
                g.family == GeneratorFamily::IsomAtom && g.description.contains("exact orbit")
            })
            .unwrap_or_else(|| {
                panic!(
                    "replicate {label}: the circle atom must carry an exact-orbit \
                     Isom generator; got: {:?}",
                    report
                        .generators
                        .iter()
                        .map(|g| g.description.clone())
                        .collect::<Vec<_>>()
                )
            });
        assert!(
            phase.generator_norm > 0.0,
            "replicate {label}: the phase orbit moves the fit (nonzero \
             uncompensated motion) — a zero norm means the view was empty"
        );
        assert!(
            phase.pinned_energy_fraction <= 1.0e-6,
            "replicate {label}: the harmonic basis is closed under phase \
             shifts, so the compensation residual must sit at the numerical \
             noise floor, got {} ({})",
            phase.pinned_energy_fraction,
            phase.description
        );
        assert_eq!(
            phase.lowering_error_scale, 0.0,
            "replicate {label}: the exact path needs no lowering-error \
             calibration"
        );
        assert!(
            phase.unpinned,
            "replicate {label}: the U(1) phase freedom must be certified \
             unpinned — this is the over-claim guard the frame certificate \
             could only satisfy by refusing. {}",
            report.summary
        );
        assert!(
            report.group_signature().contains("Isom(M_k)"),
            "replicate {label}: the certified group must name the surviving \
             O(2)/Isom factor, got: {}",
            report.group_signature()
        );
    }

    // ---- both replicates recover the planted circle (truth guard) ---------
    let truth_cos_a = o2_aligned_mean_cos(&theta_true, &theta_a);
    let truth_cos_b = o2_aligned_mean_cos(&theta_true, &theta_b);
    println!("truth recovery (O(2)-aligned mean cos): A={truth_cos_a:.6} B={truth_cos_b:.6}");
    assert!(
        truth_cos_a >= 0.9 && truth_cos_b >= 0.9,
        "both replicates must recover the planted circle coordinates up to \
         gauge (A={truth_cos_a:.6}, B={truth_cos_b:.6})"
    );

    // ---- the load-bearing pair -------------------------------------------
    // (1) WITHOUT gauge alignment the replicates must NOT agree: the seeds
    // differ by a reflection + phase shift the objective is invariant to, so
    // the raw coordinates land on different orbit representatives. This is
    // the non-vacuousness guard — if the fits happened to share a gauge, the
    // alignment below would prove nothing.
    let raw_cos = unaligned_mean_cos(&theta_a, &theta_b);
    // (2) WITH the reported gauge — O(2) Procrustes on the circle embedding,
    // the Isom(S¹) factor of the certified group — they must agree.
    let aligned_cos = o2_aligned_mean_cos(&theta_a, &theta_b);
    println!(
        "replicate agreement: raw mean cos={raw_cos:.6}, O(2)-aligned mean cos={aligned_cos:.6}"
    );
    assert!(
        raw_cos < 0.5,
        "replicates must NOT agree without gauge alignment (raw mean cos = \
         {raw_cos:.6}); the seeds plant a reflection+shift gauge difference — \
         if this fails the test has gone vacuous"
    );
    assert!(
        aligned_cos >= 0.95,
        "replicates must agree after aligning by the reported O(2) circle \
         gauge (aligned mean cos = {aligned_cos:.6}, raw = {raw_cos:.6})"
    );

    // ---- orbit invariants agree with NO alignment --------------------------
    // The gauge lives in the coordinates only: the fitted reconstructions of
    // the two replicates (replicate B un-permuted) must already agree.
    let fitted_a = term_a.fitted();
    let fitted_b_perm = term_b.fitted();
    let mut fitted_b = Array2::<f64>::zeros((N, P));
    for (r, &src) in perm.iter().enumerate() {
        fitted_b.row_mut(src).assign(&fitted_b_perm.row(r));
    }
    let mut diff_sq = 0.0_f64;
    let mut norm_sq = 0.0_f64;
    for i in 0..N {
        for j in 0..P {
            let d = fitted_a[[i, j]] - fitted_b[[i, j]];
            diff_sq += d * d;
            norm_sq += fitted_a[[i, j]] * fitted_a[[i, j]];
        }
    }
    let rel = (diff_sq / norm_sq.max(1.0e-300)).sqrt();
    println!("fitted-reconstruction relative Frobenius gap (no alignment): {rel:.6}");
    assert!(
        rel <= 0.1,
        "orbit invariants (fitted reconstructions) must agree without any \
         alignment; relative Frobenius gap = {rel:.6}"
    );
}

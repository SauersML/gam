//! Fit-level verification battery for the #972 Grassmann decoder frames
//! (issue #992, items 2 and 3).
//!
//! * **Evidence equality at small p** — a dictionary of truly low-rank atoms
//!   evaluated through the SAME criterion entry (`reml_criterion`) twice: once
//!   on the full-`B` border, once with the Grassmann frames activated. Frame
//!   activation is an exact re-representation (the decoder matrix is
//!   unchanged; only the border coordinates and the Laplace dimension
//!   accounting move), so the criterion must agree within a small relative
//!   tolerance — this is the test that catches a drift in
//!   `grassmann_evidence_dimension()` → `reml_occam_term` (the profiled-frame
//!   normalizer), the objective↔gradient-desync class wearing evidence
//!   clothing. The border-size collapse is asserted as a hard invariant
//!   alongside (`factored_border_dim == Σ M_k·r_k < beta_dim`).
//!
//! * **Alternating-update stationarity** — the closed-form streaming polar
//!   update is exact block-coordinate ascent for the frame: at an alternating
//!   `(U, C)` fixed point of the factored least-squares objective the joint
//!   gradient restricted to the Stiefel tangent space at `U` must vanish.
//!   Probed by central finite differences along a basis of tangent directions
//!   (both the `U·A` antisymmetric leg and the `U_⊥·B` normal leg), with the
//!   in-tree `GrassmannCrossMoment` / `GrassmannFrame::polar_update`
//!   primitives driving the alternation — the same code path the term's
//!   frame refresh uses.

use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::terms::latent_coord::LatentManifold;
use gam::terms::sae_manifold::{GrassmannCrossMoment, GrassmannFrame};
use gam::terms::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

use faer::Side as FaerSide;

const M: usize = 3; // const + 1 harmonic -> circle basis
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const INNER_MAX_ITER: usize = 30;
const LEARNING_RATE: f64 = 1.0;
const RIDGE_EXT_COORD: f64 = 1.0e-6;
const RIDGE_BETA: f64 = 1.0e-6;

/// Deterministic uniform in [0,1) keyed purely by index (no clock).
fn idx_uniform(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Two disjoint planted circle atoms in ambient dimension `p`: atom `a` lives
/// in the orthogonal coordinate plane spanned by axes `(2a, 2a+1)` plus a
/// small constant offset along axis `4 + a`. Rows alternate which atom is
/// active; active rows carry mass `gate` on their atom.
struct SmallTruth {
    n: usize,
    k: usize,
    theta: Vec<Vec<f64>>,
    active: Vec<Vec<bool>>,
}

fn small_truth(n: usize, k: usize) -> SmallTruth {
    let mut theta = vec![vec![0.0_f64; n]; k];
    let mut active = vec![vec![false; n]; k];
    for i in 0..n {
        let a = i % k;
        active[a][i] = true;
        theta[a][i] = idx_uniform((i as u64) * 7 + a as u64);
    }
    SmallTruth {
        n,
        k,
        theta,
        active,
    }
}

fn planted_z(truth: &SmallTruth, p: usize) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((truth.n, p));
    for a in 0..truth.k {
        for i in 0..truth.n {
            if !truth.active[a][i] {
                continue;
            }
            let ang = 2.0 * std::f64::consts::PI * truth.theta[a][i];
            // Rank-3 atom span: {const axis, cos axis, sin axis}.
            z[[i, 2 * a]] += ang.cos();
            z[[i, 2 * a + 1]] += ang.sin();
            z[[i, 4 + a]] += 0.5;
            // Tiny deterministic noise off-plane so nothing is exactly zero.
            for j in 0..p {
                z[[i, j]] += 1.0e-3 * (idx_uniform((i * p + j) as u64 ^ 0xABCD) - 0.5);
            }
        }
    }
    z
}

/// Build a small seeded term over the planted data: coords near truth, logits
/// hard ±4 by planted activity, decoder per atom from a ridge LS of `z` on the
/// atom's basis over its active rows (rank ≤ M = 3 ≪ p by construction, so the
/// frame activation margin is satisfied for every atom).
fn build_small_term(truth: &SmallTruth, z: &Array2<f64>, p: usize) -> SaeManifoldTerm {
    let n = truth.n;
    let k = truth.k;
    let evaluator = PeriodicHarmonicEvaluator::new(M).unwrap();
    let mut atoms = Vec::with_capacity(k);
    let mut coords_k = Vec::with_capacity(k);
    let mut logits = Array2::<f64>::zeros((n, k));
    for a in 0..k {
        let coords = Array2::from_shape_fn((n, 1), |(i, _)| {
            (truth.theta[a][i] + 0.03).rem_euclid(1.0)
        });
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        // Ridge LS decoder over the atom's active rows: B_a = (ΦᵀΦ+εI)⁻¹ΦᵀZ.
        let act_rows: Vec<usize> = (0..n).filter(|&i| truth.active[a][i]).collect();
        let mut phi_act = Array2::<f64>::zeros((act_rows.len(), M));
        let mut z_act = Array2::<f64>::zeros((act_rows.len(), p));
        for (r, &i) in act_rows.iter().enumerate() {
            for c in 0..M {
                phi_act[[r, c]] = phi[[i, c]];
            }
            for j in 0..p {
                z_act[[r, j]] = z[[i, j]];
            }
        }
        let mut xtx = fast_ata(&phi_act);
        for d in 0..M {
            xtx[[d, d]] += 1.0e-8;
        }
        let xtz = fast_atb(&phi_act, &z_act);
        let b = xtx
            .cholesky(FaerSide::Lower)
            .expect("decoder LS Cholesky")
            .solve_mat(&xtz);
        for i in 0..n {
            logits[[i, a]] = if truth.active[a][i] { 4.0 } else { -4.0 };
        }
        let atom = SaeManifoldAtom::new(
            format!("circle_{a}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            b,
            Array2::<f64>::eye(M),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(M).unwrap()));
        atoms.push(atom);
        coords_k.push(coords);
    }
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords_k,
        vec![LatentManifold::Circle { period: 1.0 }; k],
        AssignmentMode::ibp_map(TAU, ALPHA, false),
    )
    .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
}

#[test]
fn frame_factored_evidence_matches_full_b_at_small_p() {
    let p = 12usize;
    let k = 2usize;
    let n = 240usize;
    let truth = small_truth(n, k);
    let z = planted_z(&truth, p);
    let base = build_small_term(&truth, &z, p);
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::zeros(1); k]);

    // Arm 1: full-B border (frames never activated).
    let mut full = base.clone();
    for atom in &mut full.atoms {
        atom.deactivate_decoder_frame();
    }
    // Arm 2: frame-factored border. Every atom's decoder has rank ≤ M = 3 ≪ p,
    // so the activation margin is satisfied and activation must engage.
    let mut framed = base.clone();
    let mut total_rank = 0usize;
    for atom in &mut framed.atoms {
        let r = atom
            .maybe_activate_decoder_frame()
            .expect("frame activation")
            .expect("rank-3-in-p=12 atom must activate its frame");
        assert!(r <= M, "activated rank {r} cannot exceed basis size {M}");
        total_rank += r;
    }

    // Exactness discipline: activation re-REPRESENTS the decoder, it does not
    // change it. Fitted values must be identical bit-for-bit.
    let f_full = full.fitted();
    let f_framed = framed.fitted();
    for (a, b) in f_full.iter().zip(f_framed.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "frame activation changed a fitted value"
        );
    }

    // The border-size collapse is a tested invariant, not a hope.
    gam::terms::sae_manifold::grassmann_assert_border_dim_invariant(&framed)
        .expect("border-dim invariant");
    assert!(
        framed.factored_border_dim() < full.factored_border_dim(),
        "factored border {} must be strictly below the full-B border {}",
        framed.factored_border_dim(),
        full.factored_border_dim()
    );
    assert!(
        framed.grassmann_evidence_dimension() > 0,
        "profiled Grassmann dims must be counted for the evidence normalizer"
    );
    let _ = total_rank;

    // Evidence consistency: the same criterion entry on both representations
    // of the same model state. The factored log-det plus the profiled-frame
    // dimension term must reproduce the full-B evidence within a small
    // relative tolerance; a drift here is exactly the dimension-accounting
    // bug class the test exists to catch.
    let (v_full, loss_full) = full
        .reml_criterion(
            z.view(),
            &rho,
            None,
            INNER_MAX_ITER,
            LEARNING_RATE,
            RIDGE_EXT_COORD,
            RIDGE_BETA,
        )
        .expect("full-B criterion");
    let (v_framed, loss_framed) = framed
        .reml_criterion(
            z.view(),
            &rho,
            None,
            INNER_MAX_ITER,
            LEARNING_RATE,
            RIDGE_EXT_COORD,
            RIDGE_BETA,
        )
        .expect("framed criterion");
    let scale = v_full.abs().max(1.0);
    let gap = (v_framed - v_full).abs();
    println!(
        "evidence: full={v_full:.8} framed={v_framed:.8} gap={gap:.3e} (rel {:.3e}); \
         data_fit full={:.6} framed={:.6}",
        gap / scale,
        loss_full.data_fit,
        loss_framed.data_fit
    );
    assert!(
        gap <= 1.0e-2 * scale,
        "EVIDENCE DRIFT: frame-factored criterion {v_framed:.8} vs full-B {v_full:.8} \
         (gap {gap:.3e}, scale {scale:.3e}) — check grassmann_evidence_dimension / \
         reml_occam_term accounting"
    );
}

// ---------------------------------------------------------------------------
// Designed-subsample honesty at the fit level (#991 acceptance): a fit on a
// non-uniformly designed, HT-weighted subsample must recover the same planted
// structure as the full fit.
// ---------------------------------------------------------------------------

/// Orthonormal basis of `span(Bᵀ)` (the atom's output plane), via thin SVD.
fn decoder_plane(b: &Array2<f64>, p: usize, r: usize) -> Array2<f64> {
    use gam::linalg::faer_ndarray::FaerSvd;
    let (_u, sv, vt) = b.svd(false, true).expect("plane SVD");
    let vt = vt.expect("right factor");
    let keep = r.min(vt.nrows()).min(sv.len());
    let mut q = Array2::<f64>::zeros((p, keep));
    for c in 0..keep {
        for i in 0..p {
            q[[i, c]] = vt[[c, i]];
        }
    }
    q
}

/// Largest principal angle between two orthonormal frames (radians).
fn max_principal_angle(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    use gam::linalg::faer_ndarray::FaerSvd;
    let m = a.t().dot(b);
    let (_x, sv, _y) = m.svd(false, false).expect("angle SVD");
    let smin = sv.iter().copied().fold(f64::INFINITY, f64::min);
    smin.clamp(-1.0, 1.0).acos()
}

#[test]
fn designed_weighted_subsample_fit_recovers_what_the_full_fit_recovers() {
    use gam::inference::row_measure::RowMeasure;
    use gam::inference::row_metric::RowMetric;

    let p = 12usize;
    let k = 2usize;
    let n = 240usize;
    let truth = small_truth(n, k);
    let z = planted_z(&truth, p);
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::zeros(1); k]);

    // Planted plane of atom a: {e_{2a}, e_{2a+1}, e_{4+a}}.
    let planted_plane = |a: usize| -> Array2<f64> {
        let mut q = Array2::<f64>::zeros((p, 3));
        q[[2 * a, 0]] = 1.0;
        q[[2 * a + 1, 1]] = 1.0;
        q[[4 + a, 2]] = 1.0;
        q
    };

    // ---- full fit -----------------------------------------------------------
    let mut full = build_small_term(&truth, &z, p);
    full.reml_criterion(
        z.view(),
        &rho,
        None,
        INNER_MAX_ITER,
        LEARNING_RATE,
        RIDGE_EXT_COORD,
        RIDGE_BETA,
    )
    .expect("full fit");

    // ---- designed, non-uniform, HT-weighted subsample fit -------------------
    // A deliberately biased design measure: atom-0 rows carry 3× the mass of
    // atom-1 rows (p_metric = 1, rank = 1 ⇒ mass = u²), so the design
    // oversamples atom 0 and the 1/π weights must undo exactly that bias.
    let factors = Array2::from_shape_fn((n, 1), |(i, _)| {
        if truth.active[0][i] { 3.0_f64.sqrt() } else { 1.0 }
    });
    let metric = RowMetric::output_fisher(Arc::new(factors), 1, 1).expect("design metric");
    let measure = RowMeasure::from_metric(&metric);
    let budget = 150usize;
    let sample = measure.designed_subsample(budget, 23);
    assert!(
        sample.rows.len() < n,
        "design must be a proper subsample ({} of {n})",
        sample.rows.len()
    );

    // Slice the planted world onto the designed rows and build + weight the
    // subsample term (what `collect_designed_target` + the term seam do at
    // corpus scale, materialized inline here on the in-memory fixture).
    let n_sub = sample.rows.len();
    let mut sub_truth = SmallTruth {
        n: n_sub,
        k,
        theta: vec![vec![0.0; n_sub]; k],
        active: vec![vec![false; n_sub]; k],
    };
    let mut z_sub = Array2::<f64>::zeros((n_sub, p));
    for (s_idx, &row) in sample.rows.iter().enumerate() {
        for a in 0..k {
            sub_truth.theta[a][s_idx] = truth.theta[a][row];
            sub_truth.active[a][s_idx] = truth.active[a][row];
        }
        for j in 0..p {
            z_sub[[s_idx, j]] = z[[row, j]];
        }
    }
    let mut designed = build_small_term(&sub_truth, &z_sub, p);
    designed
        .set_row_loss_weights(sample.likelihood_weights.clone())
        .expect("install honesty weights");
    assert!(
        designed.row_loss_weights().is_some(),
        "a non-uniform design must install non-trivial weights"
    );
    designed
        .reml_criterion(
            z_sub.view(),
            &rho,
            None,
            INNER_MAX_ITER,
            LEARNING_RATE,
            RIDGE_EXT_COORD,
            RIDGE_BETA,
        )
        .expect("designed fit");

    // ---- both arms recover the planted planes --------------------------------
    for a in 0..k {
        let truth_q = planted_plane(a);
        let full_q = decoder_plane(&full.atoms[a].decoder_coefficients, p, 3);
        let designed_q = decoder_plane(&designed.atoms[a].decoder_coefficients, p, 3);
        let ang_full = max_principal_angle(&full_q, &truth_q);
        let ang_designed = max_principal_angle(&designed_q, &truth_q);
        println!(
            "atom {a}: principal angle to plant — full={ang_full:.4} rad, \
             designed+weighted={ang_designed:.4} rad"
        );
        assert!(
            ang_full < 0.30,
            "full fit failed to recover atom {a}'s plane ({ang_full:.4} rad)"
        );
        assert!(
            ang_designed < 0.30,
            "designed+weighted fit failed to recover atom {a}'s plane \
             ({ang_designed:.4} rad) — honesty weighting did not preserve the estimand"
        );
    }
}

// ---------------------------------------------------------------------------
// Item 3: alternating polar/LS stationarity (FD against the joint gradient).
// ---------------------------------------------------------------------------

/// Factored LS objective `F(U, C) = ½ ‖Z − Φ C Uᵀ‖²_F` with `U ∈ St(p, r)`.
fn factored_objective(z: &Array2<f64>, phi: &Array2<f64>, c: &Array2<f64>, u: &Array2<f64>) -> f64 {
    let recon = phi.dot(c).dot(&u.t());
    let mut acc = 0.0;
    for (zv, rv) in z.iter().zip(recon.iter()) {
        let d = zv - rv;
        acc += 0.5 * d * d;
    }
    acc
}

/// Thin-QR retraction of `U + step` back onto the Stiefel manifold via the
/// same polar primitive the production frame refresh uses (the polar factor of
/// a near-orthonormal matrix IS its closest orthonormal frame).
fn retract(u_plus_step: &Array2<f64>) -> Array2<f64> {
    GrassmannFrame::polar_update(u_plus_step.view())
        .expect("retraction polar")
        .frame()
        .to_owned()
}

#[test]
fn alternating_polar_ls_fixed_point_is_jointly_stationary() {
    let p = 12usize;
    let r = 2usize;
    let n = 300usize;

    // Planted low-rank data Z = Φ C* U*ᵀ + small noise.
    let evaluator = PeriodicHarmonicEvaluator::new(M).unwrap();
    let coords = Array2::from_shape_fn((n, 1), |(i, _)| idx_uniform(i as u64 * 13 + 1));
    let (phi, _jet) = evaluator.evaluate(coords.view()).unwrap();
    // Planted orthonormal U*: the first r coordinate axes.
    let u_star = {
        let mut u0 = Array2::<f64>::zeros((p, r));
        u0[[0, 0]] = 1.0;
        u0[[1, 1]] = 1.0;
        u0
    };
    let c_star = Array2::from_shape_fn((M, r), |(a, b)| {
        1.0 + 0.3 * ((a * r + b) as f64)
    });
    let mut z = phi.dot(&c_star).dot(&u_star.t());
    for (idx, v) in z.iter_mut().enumerate() {
        *v += 1.0e-3 * (idx_uniform(idx as u64 ^ 0x5EED) - 0.5);
    }

    // Alternate: C from exact LS given U; U from the streaming polar of the
    // cross-moment Σ zᵢ (φᵢ C)ᵀ given C — the production update.
    let mut u = {
        // Cold start: a deterministic non-planted frame.
        let mut u0 = Array2::<f64>::zeros((p, r));
        for i in 0..p {
            for j in 0..r {
                u0[[i, j]] = idx_uniform((i * r + j) as u64 * 31 + 5) - 0.5;
            }
        }
        retract(&u0)
    };
    let mut c = Array2::<f64>::zeros((M, r));
    for _ in 0..60 {
        // C-step: minimize over C exactly — (ΦᵀΦ) C = Φᵀ Z U.
        let mut xtx = fast_ata(&phi);
        for d in 0..M {
            xtx[[d, d]] += 1.0e-12;
        }
        let rhs = fast_atb(&phi, &z.dot(&u));
        c = xtx
            .cholesky(FaerSide::Lower)
            .expect("C-step Cholesky")
            .solve_mat(&rhs);
        // U-step: streaming cross-moment + closed-form polar.
        let mut cross = GrassmannCrossMoment::new(p, r);
        cross
            .accumulate(z.view(), phi.dot(&c).view())
            .expect("cross-moment accumulate");
        u = cross.polar_frame().expect("polar frame").frame().to_owned();
    }

    let f0 = factored_objective(&z, &phi, &c, &u);

    // FD stationarity over a tangent basis at U: T_U St(p,r) =
    // { U·A (A antisym) } ⊕ { U⊥·B }. Directional derivative of F along each
    // tangent direction (through the polar retraction) must vanish at the
    // alternating fixed point — the closed-form polar step IS exact
    // block-coordinate ascent, so its fixed point is jointly stationary.
    let h = 1.0e-5;
    let mut worst = 0.0_f64;
    let mut probe = |dir: &Array2<f64>, label: &str| {
        let norm: f64 = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1.0e-12 {
            return;
        }
        let unit = dir.mapv(|v| v / norm);
        let up = retract(&(&u + &unit.mapv(|v| v * h)));
        let um = retract(&(&u - &unit.mapv(|v| v * h)));
        let fp = factored_objective(&z, &phi, &c, &up);
        let fm = factored_objective(&z, &phi, &c, &um);
        let deriv = (fp - fm) / (2.0 * h);
        let rel = deriv.abs() / f0.abs().max(1.0);
        println!("tangent {label}: directional derivative {deriv:.3e} (rel {rel:.3e})");
        if rel > worst {
            worst = rel;
        }
    };

    // Antisymmetric leg: the single independent A for r = 2.
    {
        let mut a = Array2::<f64>::zeros((r, r));
        a[[0, 1]] = 1.0;
        a[[1, 0]] = -1.0;
        let dir = u.dot(&a);
        probe(&dir, "U·A (inner rotation)");
    }
    // Normal leg: a few deterministic U⊥ directions (project ambient axes off
    // range(U)).
    for axis in [2usize, 5, 9] {
        let mut e = Array2::<f64>::zeros((p, 1));
        e[[axis, 0]] = 1.0;
        let proj = u.dot(&u.t().dot(&e));
        let perp = &e - &proj;
        for col in 0..r {
            let mut dir = Array2::<f64>::zeros((p, r));
            for i in 0..p {
                dir[[i, col]] = perp[[i, 0]];
            }
            probe(&dir, &format!("U⊥ axis {axis} → col {col}"));
        }
    }

    assert!(
        worst <= 1.0e-4,
        "STATIONARITY FAIL: worst relative tangent directional derivative {worst:.3e} \
         exceeds 1e-4 — the alternating polar/LS fixed point is not a joint stationary \
         point (polar step optimizing a different objective than the C solve?)"
    );
}

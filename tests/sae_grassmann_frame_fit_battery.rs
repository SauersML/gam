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
use gam::solver::outer_strategy::OuterProblem;
use gam::terms::latent_coord::LatentManifold;
use gam::terms::sae_manifold::{GrassmannCrossMoment, GrassmannFrame};
use gam::terms::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};
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

/// Residual-energy IBP cold seed (the production `sae_residual_seed_logits`
/// path): logit ∝ −gain·(per-atom reconstruction residual − row mean)/row mean,
/// so each row routes toward the atom that explains it best. Data-driven and
/// soft (no hard ±gain saturation), which keeps the inner t-block PD-reachable.
fn residual_seed_logits(
    basis_values: ArrayView3<'_, f64>,
    basis_sizes: &[usize],
    z: ArrayView2<'_, f64>,
    gain: f64,
) -> Array2<f64> {
    let k_atoms = basis_sizes.len();
    let (n_obs, p_out) = z.dim();
    let mut logits = Array2::<f64>::zeros((n_obs, k_atoms));
    if n_obs == 0 || p_out == 0 || k_atoms <= 1 {
        return logits;
    }
    let z_owned = z.to_owned();
    let mut resid = Array2::<f64>::zeros((n_obs, k_atoms));
    for atom_idx in 0..k_atoms {
        let m_k = basis_sizes[atom_idx];
        let mut phi = Array2::<f64>::zeros((n_obs, m_k));
        for row in 0..n_obs {
            for c in 0..m_k {
                phi[[row, c]] = basis_values[[atom_idx, row, c]];
            }
        }
        let mut gram = fast_ata(&phi);
        let mut trace = 0.0_f64;
        for i in 0..m_k {
            trace += gram[[i, i]];
        }
        let jitter = (trace / m_k as f64).max(1.0).max(1.0e-12) * 1.0e-8;
        for i in 0..m_k {
            gram[[i, i]] += jitter;
        }
        let rhs = fast_atb(&phi, &z_owned);
        let factor = gram
            .cholesky(FaerSide::Lower)
            .expect("residual seed Cholesky");
        let b_k = factor.solve_mat(&rhs);
        let fitted = phi.dot(&b_k);
        for row in 0..n_obs {
            let mut e = 0.0_f64;
            for col in 0..p_out {
                let d = z[[row, col]] - fitted[[row, col]];
                e += d * d;
            }
            resid[[row, atom_idx]] = e;
        }
    }
    let mut global_mean = 0.0_f64;
    for row in 0..n_obs {
        for k in 0..k_atoms {
            global_mean += resid[[row, k]];
        }
    }
    global_mean /= (n_obs * k_atoms) as f64;
    let floor = (global_mean * 1.0e-6).max(1.0e-12);
    for row in 0..n_obs {
        let mut row_mean = 0.0_f64;
        for k in 0..k_atoms {
            row_mean += resid[[row, k]];
        }
        row_mean = (row_mean / k_atoms as f64).max(floor);
        for k in 0..k_atoms {
            logits[[row, k]] = -gain * (resid[[row, k]] - row_mean) / row_mean;
        }
    }
    logits
}

/// Weighted-LSQ decoder seed at the IBP gate (the production
/// `sae_decoder_lsq_init` path): per-atom decoder from a joint ridge LS of `z`
/// on the gate-weighted bases.
fn decoder_lsq_init(
    basis_values: ArrayView3<'_, f64>,
    basis_sizes: &[usize],
    z: ArrayView2<'_, f64>,
    initial_logits: ArrayView2<'_, f64>,
    tau: f64,
) -> Array3<f64> {
    let k_atoms = basis_sizes.len();
    let (n_obs, p_out) = z.dim();
    let m_max = basis_sizes.iter().copied().max().unwrap_or(1).max(1);
    let mut out = Array3::<f64>::zeros((k_atoms, m_max, p_out));
    let mut a_init = Array2::<f64>::zeros((n_obs, k_atoms));
    let inv_tau = 1.0 / tau;
    for row in 0..n_obs {
        for k in 0..k_atoms {
            let x = initial_logits[[row, k]] * inv_tau;
            a_init[[row, k]] = if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let ex = x.exp();
                ex / (1.0 + ex)
            };
        }
    }
    let offsets: Vec<usize> = {
        let mut acc = 0usize;
        let mut v = Vec::with_capacity(k_atoms + 1);
        v.push(0);
        for &m in basis_sizes {
            acc += m;
            v.push(acc);
        }
        v
    };
    let m_total = offsets[k_atoms];
    let mut x = Array2::<f64>::zeros((n_obs, m_total));
    for atom_idx in 0..k_atoms {
        let m_k = basis_sizes[atom_idx];
        let off = offsets[atom_idx];
        for row in 0..n_obs {
            let w = a_init[[row, atom_idx]];
            for c in 0..m_k {
                x[[row, off + c]] = w * basis_values[[atom_idx, row, c]];
            }
        }
    }
    let mut xtx = fast_ata(&x);
    let mut trace = 0.0_f64;
    for i in 0..m_total {
        trace += xtx[[i, i]];
    }
    let jitter = (trace / m_total as f64).max(1.0) * 1.0e-8;
    for i in 0..m_total {
        xtx[[i, i]] += jitter;
    }
    let xtz = fast_atb(&x, &z.to_owned());
    let b_joint = xtx
        .cholesky(FaerSide::Lower)
        .expect("decoder LSQ Cholesky")
        .solve_mat(&xtz);
    for atom_idx in 0..k_atoms {
        let m_k = basis_sizes[atom_idx];
        let off = offsets[atom_idx];
        for c in 0..m_k {
            for j in 0..p_out {
                out[[atom_idx, c, j]] = b_joint[[off + c, j]];
            }
        }
    }
    out
}

/// Build a small cold term over the planted data with the production
/// residual-energy seed + weighted-LSQ decoder init (the basin the engine
/// fits from), coords seeded near truth.
fn build_small_term(truth: &SmallTruth, z: &Array2<f64>) -> SaeManifoldTerm {
    let n = truth.n;
    let k = truth.k;
    let evaluator = PeriodicHarmonicEvaluator::new(M).unwrap();
    let mut coords_k: Vec<Array2<f64>> = Vec::with_capacity(k);
    let mut phi_k: Vec<Array2<f64>> = Vec::with_capacity(k);
    let mut jet_k: Vec<Array3<f64>> = Vec::with_capacity(k);
    let mut basis_values = Array3::<f64>::zeros((k, n, M));
    for a in 0..k {
        let coords =
            Array2::from_shape_fn((n, 1), |(i, _)| (truth.theta[a][i] + 0.03).rem_euclid(1.0));
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        for row in 0..n {
            for c in 0..M {
                basis_values[[a, row, c]] = phi[[row, c]];
            }
        }
        coords_k.push(coords);
        phi_k.push(phi);
        jet_k.push(jet);
    }
    let basis_sizes = vec![M; k];
    let logits = residual_seed_logits(basis_values.view(), &basis_sizes, z.view(), 4.0);
    let decoder = decoder_lsq_init(basis_values.view(), &basis_sizes, z.view(), logits.view(), TAU);

    let mut atoms = Vec::with_capacity(k);
    for a in 0..k {
        let b = decoder.slice(ndarray::s![a, 0..M, ..]).to_owned();
        let atom = SaeManifoldAtom::new(
            format!("circle_{a}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi_k[a].clone(),
            jet_k[a].clone(),
            b,
            Array2::<f64>::eye(M),
        )
        .unwrap()
        .with_basis_evaluator(Arc::new(PeriodicHarmonicEvaluator::new(M).unwrap()));
        atoms.push(atom);
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

/// Drive a term through the production outer engine (the only blessed fit
/// entry; it handles the inner-solve ridging that reaches a PD basin). Returns
/// the fitted term and the engine's converged criterion value. Empty ARD axes
/// mirror the proven small-SAE fixtures.
fn fit_via_engine(term: SaeManifoldTerm, z: &Array2<f64>, label: &str) -> (SaeManifoldTerm, f64) {
    let k = term.atoms.len();
    let init_rho = SaeManifoldRho::new(
        1.0_f64.ln(),
        1.0_f64.ln(),
        vec![Array1::<f64>::zeros(0); k],
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
    let result = problem
        .run(&mut objective, label)
        .expect("outer cascade must complete");
    let (fitted, _rho, _loss) = objective.into_fitted();
    (fitted, result.final_value)
}

#[test]
fn frame_factored_evidence_matches_full_b_at_small_p() {
    let p = 12usize;
    let k = 2usize;
    let n = 240usize;
    let truth = small_truth(n, k);
    let z = planted_z(&truth, p);
    let base = build_small_term(&truth, &z);

    // --- Structural exactness (no fit needed): activation re-REPRESENTS the
    // decoder; it must not change the reconstruction, must shrink the border,
    // and must register profiled Grassmann dims for the evidence normalizer.
    let mut full_state = base.clone();
    for atom in &mut full_state.atoms {
        atom.deactivate_decoder_frame();
    }
    let mut framed_state = base.clone();
    for atom in &mut framed_state.atoms {
        let r = atom
            .maybe_activate_decoder_frame()
            .expect("frame activation")
            .expect("rank-≤3-in-p=12 atom must activate its frame");
        assert!(r <= M, "activated rank {r} cannot exceed basis size {M}");
    }
    let f_full = full_state.fitted();
    let f_framed = framed_state.fitted();
    for (a, b) in f_full.iter().zip(f_framed.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "frame activation changed a fitted value"
        );
    }
    gam::terms::sae_manifold::grassmann_assert_border_dim_invariant(&framed_state)
        .expect("border-dim invariant");
    assert!(
        framed_state.factored_border_dim() < full_state.factored_border_dim(),
        "factored border {} must be strictly below the full-B border {}",
        framed_state.factored_border_dim(),
        full_state.factored_border_dim()
    );
    assert!(
        framed_state.grassmann_evidence_dimension() > 0,
        "profiled Grassmann dims must be counted for the evidence normalizer"
    );

    // --- Evidence consistency at a FIXED smoothness λ = 1 (log λ = 0).
    //
    // The criterion is `V = data_fit + ½ log|H| − occam`. The reconstruction
    // data-fit and log-det are assembled on the FULL decoder in both arms (the
    // frame profiles the border representation but never enters the assembly).
    // The ONLY frame-dependent term is the occam normalizer's frame-dimension
    // contribution `½·grassmann_dim·log λ`, which VANISHES at log λ = 0. So at
    // a fixed log λ = 0 a correct framed path must reproduce the full-B
    // criterion to round-off; a gap means the factored representation leaked
    // into the data-fit/log-det or the dimension accounting drifted (the
    // desync class this guards). (Note: at the *optimized* λ ≠ 1 the two
    // criteria correctly DIFFER by that occam term — equality is a fixed-λ=0
    // property, not an optimized-fit property.)
    //
    // We first drive the term to a PD inner basin through the engine, then
    // evaluate the criterion at the fixed log λ = 0 ρ on both representations
    // (re-solving the inner state from the converged decoder stays PD).
    let (converged, _) = fit_via_engine(base.clone(), &z, "frame battery seed fit");
    let rho0 = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(0); k]);

    let mut full = converged.clone();
    for atom in &mut full.atoms {
        atom.deactivate_decoder_frame();
    }
    let mut framed = converged.clone();
    for atom in &mut framed.atoms {
        atom.maybe_activate_decoder_frame().expect("activate");
    }
    let (v_full, _) = full
        .reml_criterion(
            z.view(),
            &rho0,
            None,
            INNER_MAX_ITER,
            LEARNING_RATE,
            RIDGE_EXT_COORD,
            RIDGE_BETA,
        )
        .expect("full-B criterion at log λ=0");
    let (v_framed, _) = framed
        .reml_criterion(
            z.view(),
            &rho0,
            None,
            INNER_MAX_ITER,
            LEARNING_RATE,
            RIDGE_EXT_COORD,
            RIDGE_BETA,
        )
        .expect("framed criterion at log λ=0");
    let scale = v_full.abs().max(1.0);
    let gap = (v_framed - v_full).abs();
    println!(
        "evidence@logλ=0: full={v_full:.8} framed={v_framed:.8} gap={gap:.3e} (rel {:.3e})",
        gap / scale
    );
    assert!(
        gap <= 1.0e-6 * scale,
        "EVIDENCE DRIFT: frame-factored criterion {v_framed:.8} vs full-B {v_full:.8} \
         (gap {gap:.3e}) at log λ=0 — the factored representation leaked into the \
         log-det / data-fit, or grassmann_evidence_dimension/reml_occam_term drifted"
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

    // Planted plane of atom a: {e_{2a}, e_{2a+1}, e_{4+a}}.
    let planted_plane = |a: usize| -> Array2<f64> {
        let mut q = Array2::<f64>::zeros((p, 3));
        q[[2 * a, 0]] = 1.0;
        q[[2 * a + 1, 1]] = 1.0;
        q[[4 + a, 2]] = 1.0;
        q
    };

    // ---- full fit (production engine) ---------------------------------------
    let full = build_small_term(&truth, &z);
    let (full_fit, _) = fit_via_engine(full, &z, "designed-fit full");

    // ---- designed, non-uniform, HT-weighted subsample fit -------------------
    // A deliberately biased design measure: atom-0 rows carry 3× the mass of
    // atom-1 rows (p_metric = 1, rank = 1 ⇒ mass = u²), so the design
    // oversamples atom 0 and the 1/π weights must undo exactly that bias.
    let factors = Array2::from_shape_fn((n, 1), |(i, _)| {
        if truth.active[0][i] {
            3.0_f64.sqrt()
        } else {
            1.0
        }
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
    let mut designed = build_small_term(&sub_truth, &z_sub);
    designed
        .set_row_loss_weights(sample.likelihood_weights.clone())
        .expect("install honesty weights");
    assert!(
        designed.row_loss_weights().is_some(),
        "a non-uniform design must install non-trivial weights"
    );
    let (designed_fit, _) = fit_via_engine(designed, &z_sub, "designed-fit weighted subsample");

    // ---- both arms recover the planted planes --------------------------------
    for a in 0..k {
        let truth_q = planted_plane(a);
        let full_q = decoder_plane(&full_fit.atoms[a].decoder_coefficients, p, 3);
        let designed_q = decoder_plane(&designed_fit.atoms[a].decoder_coefficients, p, 3);
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
    let c_star = Array2::from_shape_fn((M, r), |(a, b)| 1.0 + 0.3 * ((a * r + b) as f64));
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

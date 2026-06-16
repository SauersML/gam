//! End-to-end verification of the frame-factored Grassmann decoder solve
//! (#972 / #977 T1) driven through the PRODUCTION outer engine.
//!
//! The SAE manifold term factors each atom's decoder `B_k = C_k Uᵀ` onto a
//! Grassmann frame `U_k ∈ St(p, r_k)` so the arrow-Schur border collapses from
//! `Σ_k M_k·p` to `Σ_k M_k·r_k`. The frame auto-activates (magic, no flag) when
//! the decoder's numerical rank is materially below the ambient output dim `p`
//! (`r ≤ (1 − margin)·p`). These tests plant TRULY low-rank circle atoms in an
//! ambient `p` where the frame is genuinely beneficial, fit them through the
//! same engine the production driver uses (`OuterProblem::run` around
//! `SaeManifoldOuterObjective`, cold residual-energy seed + weighted-LSQ decoder
//! init), and assert that the factored solve:
//!
//!   1. recovers the planted decoder PLANE (principal angle small) while the
//!      border actually collapses (`factored_border_dim < beta_dim`,
//!      `border_frame_rank < p`) and the border-dim invariant holds;
//!   2. matches the full-`B` fit's plane recovery AND reconstruction R² on a
//!      truly low-rank atom (the factorization is exact re-representation, it
//!      must not cost accuracy);
//!   3. handles a MIXED dictionary (one low-rank atom frames, one full-rank-ish
//!      atom stays full-`B`) — the variable-`r` path;
//!   4. keeps the evidence criterion consistent: at a FIXED smoothness
//!      `log λ = 0` the framed and full-`B` `reml_criterion` agree to round-off
//!      (the only frame-dependent term — the Grassmann-dimension occam
//!      normalizer — is linear in `log λ` and vanishes at `log λ = 0`; at the
//!      optimized `λ ≠ 1` they correctly differ by that occam term).
//!
//! These are HONEST gates: principled tolerances picked up front, never weakened
//! to make a fit pass. A real fit that cannot reach them is a finding.

use gam::linalg::faer_ndarray::{FaerCholesky, FaerSvd, fast_ata, fast_atb};
use gam::solver::rho_optimizer::OuterProblem;
use gam::terms::latent::LatentManifold;
use gam::terms::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};
use std::sync::Arc;

use faer::Side as FaerSide;

// ---- production defaults (gamfit `sae_manifold_fit`, ibp_map path) ----------
const M: usize = 3; // const + 1 harmonic (sin, cos) -> circle basis, rank ≤ 3
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const INNER_MAX_ITER: usize = 20;
const LEARNING_RATE: f64 = 1.0;
const RIDGE_EXT_COORD: f64 = 1.0e-6;
const RIDGE_BETA: f64 = 1.0e-6;
const RESIDUAL_SEED_GAIN: f64 = 4.0;

// ---- honest tolerances ------------------------------------------------------
/// Max principal angle (rad) between a fitted decoder plane and the plant. ~0.30
/// rad ≈ 17°: a clean rank-3 planted signal must be recovered well inside this.
const PLANE_ANGLE_TOL: f64 = 0.30;
/// Reconstruction R² agreement between the factored and full-`B` fits on a
/// truly low-rank atom — the factorization is exact, so they must agree closely.
const R2_AGREE_TOL: f64 = 0.02;
/// Floor each fit must clear on a clean low-rank planted signal.
const R2_FLOOR: f64 = 0.85;
/// Relative tolerance for the fixed-`log λ = 0` evidence agreement (round-off).
const EVIDENCE_REL_TOL: f64 = 1.0e-6;

/// Deterministic uniform in [0,1) keyed purely by index (no clock).
fn idx_uniform(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------
// Planted low-rank world. Each circle atom `a` lives in a 3-dimensional output
// plane (the rank-3 span {const-axis, cos-axis, sin-axis}). We embed those three
// axes far apart inside a much larger ambient `p` so the decoder is GENUINELY
// rank ≤ 3 ≪ p and the Grassmann frame is beneficial.
// ---------------------------------------------------------------------------

struct SmallTruth {
    n: usize,
    k: usize,
    theta: Vec<Vec<f64>>,
    active: Vec<Vec<bool>>,
    /// `planted_axes[a] = (cos_axis, sin_axis, offset_axis)` for atom `a`.
    planted_axes: Vec<(usize, usize, usize)>,
}

/// `k` disjoint planted circle atoms in ambient `p`. Atom `a` occupies axes
/// `(2a, 2a+1)` for its cos/sin plane plus axis `2K + a` for a constant offset,
/// so the planted planes are mutually orthogonal coordinate triples. Rows
/// round-robin which atom is active.
fn small_truth(n: usize, k: usize, p: usize) -> SmallTruth {
    let mut theta = vec![vec![0.0_f64; n]; k];
    let mut active = vec![vec![false; n]; k];
    let mut planted_axes = Vec::with_capacity(k);
    for a in 0..k {
        let cos_axis = 2 * a;
        let sin_axis = 2 * a + 1;
        let off_axis = 2 * k + a;
        assert!(
            off_axis < p,
            "ambient p={p} too small for {k} disjoint rank-3 planted atoms"
        );
        planted_axes.push((cos_axis, sin_axis, off_axis));
    }
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
        planted_axes,
    }
}

fn planted_z(truth: &SmallTruth, p: usize) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((truth.n, p));
    for a in 0..truth.k {
        let (cos_axis, sin_axis, off_axis) = truth.planted_axes[a];
        for i in 0..truth.n {
            if !truth.active[a][i] {
                continue;
            }
            let ang = 2.0 * std::f64::consts::PI * truth.theta[a][i];
            z[[i, cos_axis]] += ang.cos();
            z[[i, sin_axis]] += ang.sin();
            z[[i, off_axis]] += 0.5;
            // Tiny deterministic off-plane noise so nothing is exactly zero.
            for j in 0..p {
                z[[i, j]] += 1.0e-3 * (idx_uniform((i * p + j) as u64 ^ 0xABCD) - 0.5);
            }
        }
    }
    z
}

/// The planted `p×3` orthonormal plane of atom `a` (its three coordinate axes).
fn planted_plane(truth: &SmallTruth, a: usize, p: usize) -> Array2<f64> {
    let (cos_axis, sin_axis, off_axis) = truth.planted_axes[a];
    let mut q = Array2::<f64>::zeros((p, 3));
    q[[cos_axis, 0]] = 1.0;
    q[[sin_axis, 1]] = 1.0;
    q[[off_axis, 2]] = 1.0;
    q
}

// ---------------------------------------------------------------------------
// Production cold seed (residual-energy IBP logits + weighted-LSQ decoder init),
// ported VERBATIM from the battery fixtures.
// ---------------------------------------------------------------------------

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

/// Build a cold term over the planted data (production residual-energy seed +
/// weighted-LSQ decoder init), coords seeded near the planted angles.
fn build_small_term(truth: &SmallTruth, z: &Array2<f64>, p: usize) -> SaeManifoldTerm {
    let n = truth.n;
    let k = truth.k;
    assert_eq!(z.ncols(), p, "z ambient dim must match p");
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
    let logits = residual_seed_logits(
        basis_values.view(),
        &basis_sizes,
        z.view(),
        RESIDUAL_SEED_GAIN,
    );
    let decoder = decoder_lsq_init(
        basis_values.view(),
        &basis_sizes,
        z.view(),
        logits.view(),
        TAU,
    );

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

/// Drive a (possibly frame-activated) term through the production outer engine.
/// Returns the fitted term and the engine's converged criterion value. Frame
/// state on the input term is preserved by the inner solve (the engine fits the
/// representation it is handed), so to fit the FACTORED solve we activate frames
/// on the term before calling this; for the full-`B` arm we leave them off.
fn fit_via_engine(term: SaeManifoldTerm, z: &Array2<f64>, label: &str) -> (SaeManifoldTerm, f64) {
    let k = term.atoms.len();
    let init_rho =
        SaeManifoldRho::new(1.0_f64.ln(), 1.0_f64.ln(), vec![Array1::<f64>::zeros(0); k]);
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
    let problem = OuterProblem::new(n_params)
        .with_initial_rho(init_rho_flat)
        .with_max_iter(25);
    let result = problem
        .run(&mut objective, label)
        .expect("outer cascade must complete");
    let (fitted, _rho, _loss) = objective.into_fitted();
    (fitted, result.final_value)
}

/// Orthonormal basis of `span(Bᵀ)` (the atom's output plane), via thin SVD —
/// the right-singular subspace of the decoder. `decoder_coefficients` always
/// holds the reconstructed full `B_k` (the frame only re-represents the border),
/// so this reads the plane identically on framed and full-`B` atoms.
fn decoder_plane(b: &Array2<f64>, p: usize, r: usize) -> Array2<f64> {
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
    let m = a.t().dot(b);
    let (_x, sv, _y) = m.svd(false, false).expect("angle SVD");
    let smin = sv.iter().copied().fold(f64::INFINITY, f64::min);
    smin.clamp(-1.0, 1.0).acos()
}

/// Reconstruction R² of a fitted term against the planted response.
fn reconstruction_r2(term: &SaeManifoldTerm, z: &Array2<f64>) -> f64 {
    let fitted = term.fitted();
    let (n, p) = z.dim();
    let mut zbar = 0.0;
    for i in 0..n {
        for j in 0..p {
            zbar += z[[i, j]];
        }
    }
    zbar /= (n * p) as f64;
    let mut ssr = 0.0;
    let mut sst = 0.0;
    for i in 0..n {
        for j in 0..p {
            let r = z[[i, j]] - fitted[[i, j]];
            ssr += r * r;
            let d = z[[i, j]] - zbar;
            sst += d * d;
        }
    }
    1.0 - ssr / sst.max(1.0e-12)
}

// ===========================================================================
// Test 1: the factored fit recovers the planted low-rank atoms while the border
// actually collapses.
// ===========================================================================
#[test]
fn factored_fit_recovers_planted_low_rank_atoms() {
    let p = 12usize;
    let k = 3usize; // rank-3 atoms in p=12 ⇒ frames strongly beneficial
    let n = 150usize;
    let truth = small_truth(n, k, p);
    let z = planted_z(&truth, p);

    // Build the cold term, activate frames (the magic auto-activation the
    // production driver runs before the fit), then fit through the engine.
    let mut term = build_small_term(&truth, &z, p);
    let activated = term
        .auto_activate_decoder_frames()
        .expect("frame auto-activation");
    println!("test1: auto-activated frames on {activated}/{k} atoms (p={p}, rank≤{M})");
    assert_eq!(
        activated, k,
        "every rank-≤{M}-in-p={p} atom must auto-activate its Grassmann frame"
    );

    // Border must collapse before the fit (it is a structural property of the
    // representation, not the fit).
    let beta_dim = term.beta_dim();
    let border_dim = term.factored_border_dim();
    println!(
        "test1: beta_dim={beta_dim} factored_border_dim={border_dim} (collapse ratio={:.3})",
        border_dim as f64 / beta_dim as f64
    );
    assert!(
        border_dim < beta_dim,
        "factored border {border_dim} must be strictly below full-B border {beta_dim}"
    );
    gam::terms::sae::manifold::grassmann_assert_border_dim_invariant(&term)
        .expect("border-dim invariant before fit");

    let (fitted, final_v) = fit_via_engine(term, &z, "test1 factored low-rank fit");
    println!("test1: engine final criterion = {final_v:.6}");

    // Frames must still be active after the fit, with each border rank below p.
    gam::terms::sae::manifold::grassmann_assert_border_dim_invariant(&fitted)
        .expect("border-dim invariant after fit");
    assert!(
        fitted.factored_border_dim() < fitted.beta_dim(),
        "border must remain collapsed after the fit: {} vs {}",
        fitted.factored_border_dim(),
        fitted.beta_dim()
    );
    for a in 0..k {
        let r = fitted.atoms[a].border_frame_rank();
        println!("test1: atom {a} border_frame_rank={r} (p={p})");
        assert!(
            r < p,
            "atom {a} frame must keep border rank {r} < p={p} after the fit"
        );
    }

    // Each fitted atom recovers its planted decoder plane.
    for a in 0..k {
        let truth_q = planted_plane(&truth, a, p);
        let fit_q = decoder_plane(&fitted.atoms[a].decoder_coefficients, p, 3);
        let ang = max_principal_angle(&fit_q, &truth_q);
        println!("test1: atom {a} max principal angle to plant = {ang:.4} rad");
        assert!(
            ang < PLANE_ANGLE_TOL,
            "factored fit failed to recover atom {a}'s plane ({ang:.4} rad >= {PLANE_ANGLE_TOL})"
        );
    }

    let r2 = reconstruction_r2(&fitted, &z);
    println!("test1: reconstruction R2 = {r2:.6}");
    assert!(
        r2 >= R2_FLOOR,
        "factored fit reconstruction R2 {r2:.6} < floor {R2_FLOOR}"
    );
}

// ===========================================================================
// Test 2: the factored fit matches the full-B fit on a truly low-rank atom —
// both recover the plant, and their reconstruction R² agree closely.
// ===========================================================================
#[test]
fn factored_matches_full_b_recovery() {
    let p = 12usize;
    let k = 2usize;
    let n = 144usize;
    let truth = small_truth(n, k, p);
    let z = planted_z(&truth, p);

    // ---- full-B arm: frames force-DEACTIVATED throughout ----
    let mut full_term = build_small_term(&truth, &z, p);
    for atom in &mut full_term.atoms {
        atom.deactivate_decoder_frame();
    }
    assert!(
        !full_term.frames_active(),
        "full-B arm must carry no active frames"
    );
    assert_eq!(
        full_term.factored_border_dim(),
        full_term.beta_dim(),
        "full-B border must equal beta_dim"
    );
    let (full_fit, full_v) = fit_via_engine(full_term, &z, "test2 full-B fit");

    // ---- factored arm: frames auto-activated ----
    let mut framed_term = build_small_term(&truth, &z, p);
    let activated = framed_term
        .auto_activate_decoder_frames()
        .expect("frame auto-activation");
    assert_eq!(activated, k, "both low-rank atoms must activate frames");
    assert!(
        framed_term.factored_border_dim() < framed_term.beta_dim(),
        "factored border must collapse below full-B"
    );
    let (framed_fit, framed_v) = fit_via_engine(framed_term, &z, "test2 factored fit");

    println!(
        "test2: full-B final={full_v:.6} factored final={framed_v:.6} | \
         full-B border={} factored border={}",
        full_fit.beta_dim(),
        framed_fit.factored_border_dim()
    );

    // Both arms recover both planted planes.
    for a in 0..k {
        let truth_q = planted_plane(&truth, a, p);
        let full_q = decoder_plane(&full_fit.atoms[a].decoder_coefficients, p, 3);
        let framed_q = decoder_plane(&framed_fit.atoms[a].decoder_coefficients, p, 3);
        let ang_full = max_principal_angle(&full_q, &truth_q);
        let ang_framed = max_principal_angle(&framed_q, &truth_q);
        println!(
            "test2: atom {a} principal angle — full-B={ang_full:.4} rad, factored={ang_framed:.4} rad"
        );
        assert!(
            ang_full < PLANE_ANGLE_TOL,
            "full-B fit failed to recover atom {a}'s plane ({ang_full:.4} rad)"
        );
        assert!(
            ang_framed < PLANE_ANGLE_TOL,
            "factored fit failed to recover atom {a}'s plane ({ang_framed:.4} rad)"
        );
    }

    // Reconstruction R² must agree: the factorization is exact re-representation
    // of a truly low-rank atom, so the factored solve must not lose accuracy.
    let r2_full = reconstruction_r2(&full_fit, &z);
    let r2_framed = reconstruction_r2(&framed_fit, &z);
    let gap = (r2_full - r2_framed).abs();
    println!("test2: R2 full-B={r2_full:.6} factored={r2_framed:.6} (|gap|={gap:.3e})");
    assert!(
        r2_full >= R2_FLOOR && r2_framed >= R2_FLOOR,
        "both arms must clear the R2 floor {R2_FLOOR}: full={r2_full:.6} factored={r2_framed:.6}"
    );
    assert!(
        gap <= R2_AGREE_TOL,
        "factored R2 {r2_framed:.6} must agree with full-B R2 {r2_full:.6} within {R2_AGREE_TOL} \
         (|gap|={gap:.3e}) — the low-rank factorization lost accuracy"
    );
}

// ===========================================================================
// Test 3: a mixed dictionary — one truly low-rank atom (frame activates) and one
// genuinely full-rank-ish atom (frame stays OFF). Exercises the variable-r path.
// ===========================================================================
#[test]
fn mixed_framed_and_full_rank_atoms() {
    let p = 12usize;
    let k = 2usize;
    let n = 144usize;
    let truth = small_truth(n, k, p);
    // Atom 0: planted rank-3 (low rank). Atom 1: inflate its decoder to nearly
    // full rank by planting energy across MANY ambient axes so its numerical
    // rank approaches p and the frame is NOT beneficial.
    let mut z = planted_z(&truth, p);
    for i in 0..n {
        if !truth.active[1][i] {
            continue;
        }
        let ang = 2.0 * std::f64::consts::PI * truth.theta[1][i];
        // Spread the atom-1 signal across all p axes with distinct per-axis
        // gains so range(B_1ᵀ) fills the ambient space (numerical rank ≈ p).
        for j in 0..p {
            let gain = 1.0 + 0.5 * idx_uniform((j as u64) * 31 + 7);
            let basis = if j % 2 == 0 { ang.cos() } else { ang.sin() };
            z[[i, j]] += 0.6 * gain * basis + 0.3 * gain;
        }
    }

    let mut term = build_small_term(&truth, &z, p);
    // Per-atom activation so we can read which atoms framed.
    let r0 = term.atoms[0]
        .maybe_activate_decoder_frame()
        .expect("atom 0 activation attempt");
    let r1 = term.atoms[1]
        .maybe_activate_decoder_frame()
        .expect("atom 1 activation attempt");
    println!(
        "test3: atom0 frame={:?} (rank {}), atom1 frame={:?} (rank {})",
        r0,
        term.atoms[0].border_frame_rank(),
        r1,
        term.atoms[1].border_frame_rank()
    );
    assert!(
        r0.is_some(),
        "the truly low-rank atom 0 must activate its frame"
    );
    assert!(
        r1.is_none(),
        "the full-rank-ish atom 1 must NOT activate a frame (stays full-B)"
    );
    assert!(
        term.atoms[0].border_frame_rank() < p,
        "framed atom 0 must collapse its border rank below p"
    );
    assert_eq!(
        term.atoms[1].border_frame_rank(),
        p,
        "full-B atom 1 must keep the full border rank p"
    );

    // Mixed-r border is strictly between an all-full-B and an all-framed border.
    let beta_dim = term.beta_dim();
    let border_dim = term.factored_border_dim();
    println!("test3: beta_dim={beta_dim} mixed factored_border_dim={border_dim}");
    assert!(
        border_dim < beta_dim,
        "the framed atom must shrink the mixed border below beta_dim"
    );
    gam::terms::sae::manifold::grassmann_assert_border_dim_invariant(&term)
        .expect("mixed border-dim invariant before fit");

    let (fitted, final_v) = fit_via_engine(term, &z, "test3 mixed fit");
    println!("test3: engine final criterion = {final_v:.6}");
    gam::terms::sae::manifold::grassmann_assert_border_dim_invariant(&fitted)
        .expect("mixed border-dim invariant after fit");

    // Both atoms recover their planted planes through the mixed solve.
    for a in 0..k {
        let truth_q = planted_plane(&truth, a, p);
        let fit_q = decoder_plane(&fitted.atoms[a].decoder_coefficients, p, 3);
        let ang = max_principal_angle(&fit_q, &truth_q);
        println!("test3: atom {a} max principal angle to plant = {ang:.4} rad");
        assert!(
            ang < PLANE_ANGLE_TOL,
            "mixed fit failed to recover atom {a}'s plane ({ang:.4} rad >= {PLANE_ANGLE_TOL})"
        );
    }
}

// ===========================================================================
// Test 4: evidence consistency at a fixed smoothness λ. Mirror of the battery's
// `frame_factored_evidence_matches_full_b_at_small_p`, but driven to a PD basin
// through the WIRED production solve. At fixed log λ = 0 the framed and full-B
// criteria must agree to round-off (#999 dissolved: the factored log-det now
// matches the occam accounting). At the optimized λ ≠ 1 they correctly differ by
// the linear occam term, so we only assert equality at the log λ = 0 anchor.
// ===========================================================================
#[test]
fn evidence_consistency_at_fixed_lambda() {
    let p = 12usize;
    let k = 2usize;
    let n = 144usize;
    let truth = small_truth(n, k, p);
    let z = planted_z(&truth, p);

    // Drive the term to a PD inner basin through the engine (full-B seed).
    let base = build_small_term(&truth, &z, p);
    let (converged, _) = fit_via_engine(base, &z, "test4 evidence seed fit");

    // Evaluate the criterion at FIXED smoothness log λ on a full-B clone vs a
    // frame-activated clone, re-solving each arm's inner state from the
    // converged decoder. The only frame-dependent criterion term is the occam
    // normalizer's Grassmann-dimension contribution `½·grassmann_dim·log λ`,
    // which vanishes at log λ = 0.
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(0); k]);

    let mut full = converged.clone();
    for atom in &mut full.atoms {
        atom.deactivate_decoder_frame();
    }
    assert!(!full.frames_active(), "full-B clone must have no frames");

    let mut framed = converged.clone();
    let activated = framed
        .auto_activate_decoder_frames()
        .expect("frame activation on converged decoder");
    println!("test4: activated {activated}/{k} frames on the converged decoder");
    assert_eq!(activated, k, "both low-rank atoms must frame at the basin");
    assert!(
        framed.factored_border_dim() < framed.beta_dim(),
        "framed evidence clone border must collapse"
    );
    assert!(
        framed.grassmann_evidence_dimension() > 0,
        "profiled Grassmann dims must be counted for the occam normalizer"
    );

    let (v_full, _) = full
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
    let (v_framed, _) = framed
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

    let gap = v_framed - v_full;
    let scale = v_full.abs().max(1.0);
    println!(
        "test4: V_full(λ=1)={v_full:.8} V_framed(λ=1)={v_framed:.8} | gap={gap:.3e} (rel {:.3e})",
        gap.abs() / scale
    );
    assert!(
        gap.abs() <= EVIDENCE_REL_TOL * scale,
        "EVIDENCE DRIFT: at log λ = 0 the framed criterion {v_framed:.8} must match the full-B \
         criterion {v_full:.8} to round-off (gap {gap:.3e}, rel {:.3e} > {EVIDENCE_REL_TOL}) — the \
         factored representation leaked into the data-fit / log-det",
        gap.abs() / scale
    );
}

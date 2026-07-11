//! K-ladder recovery pin for the SAE manifold dictionary learner (#985 part 2).
//!
//! The K=2 fixture (`sae_manifold_joint_two_circle_recovery`) shows the cold
//! IBP-MAP joint fit does not collapse for two planted circle atoms. This file
//! climbs the K ladder: K=64 and K=1024 planted circle-atom dictionaries, with
//! mutually-orthogonal-ish planted frames embedded in a larger ambient space p,
//! driven through the SAME production engine (cold IBP-MAP residual-energy seed
//! logits, weighted-LSQ decoder init, generic outer cascade `OuterProblem::run`
//! around `SaeManifoldOuterObjective`).
//!
//! The collapse signature is per-atom mean active mass crashing toward ~0.03 on
//! the rows where the atom is *truly* active (planted ~0.2). A failing run here
//! REPRODUCES that at scale; a passing run REFUTES it. Either way the test
//! prints verbatim numbers (active masses, principal angles).
//!
//! Construction-path fidelity note: the gam crate cannot reach the pyffi-only
//! seed helpers. As the K=2 fixture documents, we replicate the two seed stages
//! that determine the routing collapse VERBATIM (residual-energy IBP logits at
//! gain 4.0; weighted-LSQ decoder init at the IBP gate), and seed the latent
//! coordinates from the planted angles (the production PCA / cluster coordinate
//! seed is a separate stage; this is a routing/active-mass + span-recovery
//! failure, not a coordinate-recovery one).
//!
//! The full battery per K:
//!   (a) NO COLLAPSE: per-atom mean active mass on truly-active rows stays
//!       within a principled band of the planted mass (does not crash to ~0.03).
//!   (b) RECOVERY: planted atom column-spans recovered by PRINCIPAL ANGLES;
//!       max principal angle small per matched atom (atoms matched by best
//!       alignment / smallest principal angle).
//!   (c) DETERMINISM: two runs with identical seed produce bit-identical fitted
//!       coords / masses.
//!   (d) LEDGER DETERMINISM (#992 / #976): the collapse-guard event ledger
//!       replays identically across the two runs, and a green rung records no
//!       terminal collapse.
//!   (e) ACTIVE-SET SPARSITY (#992): mean fitted per-row active-set size stays
//!       within a factor-2 band of the planted count, and mean off-support
//!       assignment mass stays below a quarter of the planted active mass.

use gam::linalg::faer_ndarray::{FaerCholesky, FaerSvd, fast_ata, fast_atb};
use gam::solver::rho_optimizer::OuterProblem;
use gam::terms::latent::LatentManifold;
use gam::terms::{
    sae::manifold::AssignmentMode, sae::manifold::PeriodicHarmonicEvaluator,
    sae::manifold::SaeAssignment, sae::manifold::SaeAtomBasisKind,
    sae::manifold::SaeBasisEvaluator, sae::manifold::SaeManifoldAtom,
    sae::manifold::SaeManifoldOuterObjective, sae::manifold::SaeManifoldRho,
    sae::manifold::SaeManifoldTerm,
};
use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3, s};
use std::sync::Arc;

use faer::Side as FaerSide;

// ---- production defaults (gamfit `sae_manifold_fit`, ordered_beta_bernoulli path) ----------
const M: usize = 3; // const + 1 harmonic (sin, cos) -> circle
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const SPARSITY: f64 = 1.0;
const SMOOTHNESS: f64 = 1.0;
const INNER_MAX_ITER: usize = 50;
const LEARNING_RATE: f64 = 1.0;
const RIDGE_EXT_COORD: f64 = 1.0e-6;
const RIDGE_BETA: f64 = 1.0e-6;
const RESIDUAL_SEED_GAIN: f64 = 4.0; // SAE_RESIDUAL_SEED_GAIN in pyffi

// ---- planted DGP --------------------------------------------------------
const PLANTED_ACTIVE_MASS: f64 = 0.2; // spec: codes'-units planted mean
const COLLAPSE_FLOOR: f64 = 0.03; // the failure-mode active mass we must beat

/// A single K-rung: dictionary cardinality, ambient dimension, observations.
struct Rung {
    k: usize,
    p: usize,
    n: usize,
}

/// Deterministic Lehmer-style uniform in [0,1) keyed purely by index (no clock).
fn idx_uniform(seed: u64) -> f64 {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((state >> 11) as f64) * f64::from_bits(0x3CA0000000000000)
}

/// K planted p×2 orthonormal frames whose 2K spanning columns are
/// mutually-orthonormal-ish: build 2K deterministic ambient vectors and
/// Gram-Schmidt them to an orthonormal set, then carve out consecutive pairs.
/// Requires 2K <= p so the planted planes are genuinely mutually orthogonal.
fn planted_frames(k: usize, p: usize) -> Vec<Array2<f64>> {
    let cols = 2 * k;
    assert!(
        cols <= p,
        "need ambient p >= 2K for mutually-orthogonal planted frames (p={p}, K={k})"
    );
    let mut raw = Array2::<f64>::zeros((p, cols));
    for j in 0..cols {
        for i in 0..p {
            // smooth, distinct, full-rank deterministic columns
            raw[[i, j]] = ((i as f64 + 1.0) * 0.37 * (j as f64 + 1.0)).sin()
                + 0.5 * ((i as f64) * 0.11 - (j as f64) * 0.9).cos()
                + 0.25 * (((i as f64) * 0.017 + (j as f64) * 0.041) * 1.7).sin();
        }
    }
    // Modified Gram-Schmidt to an orthonormal set of `cols` columns.
    let mut q = Array2::<f64>::zeros((p, cols));
    for j in 0..cols {
        let mut v = raw.column(j).to_owned();
        for prev in 0..j {
            let qp = q.column(prev);
            let dot: f64 = qp.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            for i in 0..p {
                v[i] -= dot * qp[i];
            }
        }
        let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            nrm > 1.0e-9,
            "planted ambient basis rank-deficient at col {j}"
        );
        for i in 0..p {
            q[[i, j]] = v[i] / nrm;
        }
    }
    (0..k)
        .map(|atom| q.slice(s![.., 2 * atom..2 * atom + 2]).to_owned())
        .collect()
}

/// Per-row truth: angles, gates (truly-active flags), and amplitudes.
/// Each atom is sparsely active (target ~PLANTED_ACTIVE_MASS share of rows
/// hold any single atom's gate) with a deterministic, well-spread support so
/// every atom sees enough active rows to recover its plane and angle.
struct Truth {
    k: usize,
    n: usize,
    theta: Vec<Vec<f64>>,   // [k][n]
    active: Vec<Vec<bool>>, // [k][n]
    amp: Vec<Vec<f64>>,     // [k][n]
    radii: Vec<f64>,
}

fn planted_truth(rung: &Rung) -> Truth {
    let k = rung.k;
    let n = rung.n;
    let mut theta = vec![vec![0.0; n]; k];
    let mut active = vec![vec![false; n]; k];
    let mut amp = vec![vec![0.0; n]; k];
    let radii: Vec<f64> = (0..k)
        .map(|a| 1.0 + 0.1 * (a as f64 / k.max(1) as f64))
        .collect();

    // We want each atom active on ~PLANTED_ACTIVE_MASS * n rows, with the
    // supports spread evenly so no atom is starved. Assign each atom a
    // contiguous-with-stride block of active rows of size n_active_target.
    let n_active_target = ((PLANTED_ACTIVE_MASS * n as f64).round() as usize).max(8);

    for a in 0..k {
        // Distinct irrational-ish stride per atom so angles fill the circle.
        let stride = 0.045 + 0.0007 * (a as f64);
        let phase = idx_uniform(a as u64 * 7 + 11);
        for i in 0..n {
            theta[a][i] = ((i as f64) * stride + phase).rem_euclid(1.0);
        }
        // Active support: a deterministic, atom-shifted modular pattern so the
        // union of supports tiles the rows and co-activation is common.
        let base = (a * n) / k.max(1); // spread starts across the row range
        for t in 0..n_active_target {
            // jittered stride keeps supports overlapping (co-active rows) yet
            // distinct per atom.
            let jit = (idx_uniform((a as u64) * 131 + (t as u64) * 17 + 3) * 5.0) as usize;
            let row = (base + t * 3 + jit) % n;
            active[a][row] = true;
        }
        for i in 0..n {
            amp[a][i] = if active[a][i] {
                0.85 + 0.30 * idx_uniform((a as u64) * 977 + (i as u64) * 2 + 1)
            } else {
                0.0
            };
        }
    }
    Truth {
        k,
        n,
        theta,
        active,
        amp,
        radii,
    }
}

/// Planted response Z = Σ_a gate_a · amp_a · r_a (cosθ u_a1 + sinθ u_a2) + noise.
fn planted_response(truth: &Truth, frames: &[Array2<f64>], p: usize) -> (Array2<f64>, f64) {
    let n = truth.n;
    let mut z = Array2::<f64>::zeros((n, p));
    let mut signal_sq = 0.0_f64;
    for i in 0..n {
        for a in 0..truth.k {
            if !truth.active[a][i] {
                continue;
            }
            let ang = std::f64::consts::TAU * truth.theta[a][i];
            let c = ang.cos();
            let s = ang.sin();
            let scale = truth.amp[a][i] * truth.radii[a];
            for col in 0..p {
                let contrib = scale * (c * frames[a][[col, 0]] + s * frames[a][[col, 1]]);
                z[[i, col]] += contrib;
                signal_sq += contrib * contrib;
            }
        }
    }
    let signal_scale = (signal_sq / (n * p) as f64).sqrt();
    let sigma = 0.04 * signal_scale; // ~4% of signal scale
    for i in 0..n {
        for col in 0..p {
            let u = idx_uniform(((i * p + col) as u64) * 7 + 3);
            let u2 = idx_uniform(((i * p + col) as u64) * 7 + 5);
            // Box-Muller, deterministic
            let g = (-2.0 * (u.max(1.0e-12)).ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            z[[i, col]] += sigma * g;
        }
    }
    (z, signal_scale)
}

/// VERBATIM port of pyffi `sae_residual_seed_logits` (ordered_beta_bernoulli cold seed).
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

/// VERBATIM port of pyffi `sae_decoder_lsq_init` (ordered_beta_bernoulli branch).
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
            let a = if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let ex = x.exp();
                ex / (1.0 + ex)
            };
            a_init[[row, k]] = a;
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

/// Build the cold term the production driver would hand to the outer engine.
fn build_cold_term(truth: &Truth, z: ArrayView2<'_, f64>, p: usize) -> SaeManifoldTerm {
    let k = truth.k;
    let n = truth.n;
    assert_eq!(z.ncols(), p, "ambient dim of z must match the caller's p");
    let evaluator = PeriodicHarmonicEvaluator::new(M).unwrap();
    // Seed latent coordinates from the planted angles (slightly offset, per the
    // inline torus oracle, so coordinate recovery is not what is under test).
    let mut coords_k: Vec<Array2<f64>> = Vec::with_capacity(k);
    let mut phi_k: Vec<Array2<f64>> = Vec::with_capacity(k);
    let mut jet_k: Vec<Array3<f64>> = Vec::with_capacity(k);
    for a in 0..k {
        // small atom-specific offset so the seed is not exactly the truth.
        let offset = 0.04 + 0.013 * ((a % 5) as f64);
        let coords = Array2::from_shape_fn((n, 1), |(i, _)| {
            (truth.theta[a][i] + offset).rem_euclid(1.0)
        });
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        coords_k.push(coords);
        phi_k.push(phi);
        jet_k.push(jet);
    }

    let basis_sizes = vec![M; k];
    // (K, N, M) padded basis-value stack for the seed ports.
    let mut basis_values = Array3::<f64>::zeros((k, n, M));
    for a in 0..k {
        for row in 0..n {
            for c in 0..M {
                basis_values[[a, row, c]] = phi_k[a][[row, c]];
            }
        }
    }
    // Production cold IBP-MAP routing seed + weighted-LSQ decoder init.
    let logits = residual_seed_logits(basis_values.view(), &basis_sizes, z, RESIDUAL_SEED_GAIN);
    let decoder = decoder_lsq_init(basis_values.view(), &basis_sizes, z, logits.view(), TAU);

    let mut atoms = Vec::with_capacity(k);
    for a in 0..k {
        let b = decoder.slice(s![a, 0..M, ..]).to_owned();
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
        AssignmentMode::ordered_beta_bernoulli(TAU, ALPHA, false),
    )
    .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
}

/// Drive the fit through the production outer engine; return the fitted term,
/// the converged assignments, and the engine's final criterion value.
fn run_production_fit(
    truth: &Truth,
    z: &Array2<f64>,
    p: usize,
    label: &str,
) -> (SaeManifoldTerm, Array2<f64>, f64) {
    let k = truth.k;
    let term = build_cold_term(truth, z.view(), p);
    let init_rho = SaeManifoldRho::new(
        SPARSITY.ln(),
        SMOOTHNESS.ln(),
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
    objective
        .certify_outer_result(&result)
        .expect("K-ladder outer result must certify the installed state");
    let fitted_term = objective
        .into_fitted()
        .expect("outer fit was evaluated")
        .term;
    let assignments = fitted_term.assignment.assignments();
    (fitted_term, assignments, result.final_value)
}

/// Decoder sin/cos rows span the fitted atom's circle plane. Returns the p×2
/// orthonormalized plane basis (cos-row, sin-row order matches u_a1, u_a2).
fn fitted_plane(atom_decoder: ArrayView2<'_, f64>, p: usize) -> Array2<f64> {
    // Periodic basis layout: row0=const, row1=sin, row2=cos.
    // g(θ)=B^T φ = B_const + sinθ·B_sin + cosθ·B_cos, plane = span(B_cos, B_sin)
    let cos_row = atom_decoder.row(2).to_owned();
    let sin_row = atom_decoder.row(1).to_owned();
    let mut plane = Array2::<f64>::zeros((p, 2));
    for i in 0..p {
        plane[[i, 0]] = cos_row[i];
        plane[[i, 1]] = sin_row[i];
    }
    // Gram-Schmidt orthonormalize the two columns.
    let n0 = plane.column(0).iter().map(|x| x * x).sum::<f64>().sqrt();
    for i in 0..p {
        plane[[i, 0]] /= n0.max(1.0e-300);
    }
    let dot: f64 = (0..p).map(|i| plane[[i, 0]] * plane[[i, 1]]).sum();
    for i in 0..p {
        plane[[i, 1]] -= dot * plane[[i, 0]];
    }
    let n1 = plane.column(1).iter().map(|x| x * x).sum::<f64>().sqrt();
    for i in 0..p {
        plane[[i, 1]] /= n1.max(1.0e-300);
    }
    plane
}

/// Principal angles (radians) between two p×2 orthonormal column spans, sorted
/// ascending. σ_i = singular values of U_fitᵀ U_true ∈ [0,1]; θ_i = acos σ_i.
/// Returns (theta_min, theta_max); theta_max small ⇒ identical 2-D span.
fn principal_angles(u_fit: &Array2<f64>, u_true: &Array2<f64>) -> (f64, f64) {
    let m = u_fit.t().dot(u_true);
    let (_u, sv, _vt) = m.svd(false, false).expect("principal-angle SVD");
    let mut angles: Vec<f64> = sv.iter().map(|&s| s.clamp(-1.0, 1.0).acos()).collect();
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let theta_min = angles.first().copied().unwrap_or(0.0);
    let theta_max = angles.last().copied().unwrap_or(0.0);
    (theta_min, theta_max)
}

/// Greedy best-alignment matching: for each planted atom (in descending order
/// of how cleanly its best fitted candidate matches), claim the unused fitted
/// atom with the smallest max-principal-angle. Returns match[t] = fitted index.
fn match_atoms(fitted_planes: &[Array2<f64>], planted_planes: &[Array2<f64>]) -> Vec<usize> {
    let k = planted_planes.len();
    // cost[t][f] = max principal angle between planted t and fitted f.
    let mut cost = vec![vec![0.0_f64; k]; k];
    for t in 0..k {
        for f in 0..k {
            let (_lo, hi) = principal_angles(&fitted_planes[f], &planted_planes[t]);
            cost[t][f] = hi;
        }
    }
    let mut used = vec![false; k];
    let mut matched = vec![usize::MAX; k];
    // Order planted atoms by their best available cost (clearest first) to keep
    // greedy matching robust at large K.
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| {
        let ca = (0..k).map(|f| cost[a][f]).fold(f64::INFINITY, f64::min);
        let cb = (0..k).map(|f| cost[b][f]).fold(f64::INFINITY, f64::min);
        ca.partial_cmp(&cb).unwrap()
    });
    for &t in &order {
        let mut best_f = usize::MAX;
        let mut best_c = f64::INFINITY;
        for f in 0..k {
            if used[f] {
                continue;
            }
            if cost[t][f] < best_c {
                best_c = cost[t][f];
                best_f = f;
            }
        }
        used[best_f] = true;
        matched[t] = best_f;
    }
    matched
}

/// One K-rung end-to-end: fit twice (determinism), match, assert the battery.
fn run_rung(rung: &Rung) {
    let p = rung.p;
    let k = rung.k;
    let frames = planted_frames(k, p);
    let truth = planted_truth(rung);
    let (z, signal_scale) = planted_response(&truth, &frames, p);

    let label = format!("SAE K-ladder recovery K={k}");
    // ---- determinism: identical cold fit twice ----------------------------
    let (term1, assign1, final1) = run_production_fit(&truth, &z, p, &label);
    let (term2, assign2, final2) = run_production_fit(&truth, &z, p, &label);
    let determinism_gap = (final1 - final2).abs();
    let final_bit_identical = final1.to_bits() == final2.to_bits();

    // Bit-identical fitted coords across the two runs.
    let mut coords_bit_identical = true;
    for a in 0..k {
        let c1 = term1.assignment.coords[a].as_matrix();
        let c2 = term2.assignment.coords[a].as_matrix();
        if c1.dim() != c2.dim() {
            coords_bit_identical = false;
            break;
        }
        for (v1, v2) in c1.iter().zip(c2.iter()) {
            if v1.to_bits() != v2.to_bits() {
                coords_bit_identical = false;
                break;
            }
        }
        if !coords_bit_identical {
            break;
        }
    }
    // Bit-identical active masses across the two runs.
    let mut mass_bit_identical = assign1.dim() == assign2.dim();
    if mass_bit_identical {
        for (v1, v2) in assign1.iter().zip(assign2.iter()) {
            if v1.to_bits() != v2.to_bits() {
                mass_bit_identical = false;
                break;
            }
        }
    }
    let deterministic = final_bit_identical && coords_bit_identical && mass_bit_identical;

    // ---- match fitted -> planted by principal angles ----------------------
    let planted_planes: Vec<Array2<f64>> = frames.clone();
    let fitted_planes: Vec<Array2<f64>> = (0..k)
        .map(|a| fitted_plane(term1.atoms[a].decoder_coefficients.view(), p))
        .collect();
    let matched = match_atoms(&fitted_planes, &planted_planes);

    // ---- per-atom metrics on truly-active rows ----------------------------
    let mut theta_max = vec![0.0_f64; k];
    let mut fitted_mass = vec![0.0_f64; k];
    let mut n_active = vec![0usize; k];
    for t in 0..k {
        let f = matched[t];
        let (_lo, hi) = principal_angles(&fitted_planes[f], &planted_planes[t]);
        theta_max[t] = hi;
        let mut acc = 0.0;
        let mut cnt = 0usize;
        for i in 0..truth.n {
            if truth.active[t][i] {
                acc += assign1[[i, f]];
                cnt += 1;
            }
        }
        n_active[t] = cnt;
        fitted_mass[t] = if cnt > 0 { acc / cnt as f64 } else { 0.0 };
    }

    // ---- reconstruction R^2 -----------------------------------------------
    let fitted = term1.fitted();
    let mut ssr = 0.0;
    let mut sst = 0.0;
    let mut zbar = 0.0;
    for i in 0..truth.n {
        for j in 0..p {
            zbar += z[[i, j]];
        }
    }
    zbar /= (truth.n * p) as f64;
    for i in 0..truth.n {
        for j in 0..p {
            let r = z[[i, j]] - fitted[[i, j]];
            ssr += r * r;
            let d = z[[i, j]] - zbar;
            sst += d * d;
        }
    }
    let r2 = 1.0 - ssr / sst.max(1.0e-12);

    // ---- aggregate stats over the dictionary ------------------------------
    let mean_mass = fitted_mass.iter().sum::<f64>() / k as f64;
    let min_mass = fitted_mass.iter().copied().fold(f64::INFINITY, f64::min);
    let mean_theta = theta_max.iter().sum::<f64>() / k as f64;
    let worst_theta = theta_max.iter().copied().fold(0.0_f64, f64::max);
    let n_collapsed = fitted_mass
        .iter()
        .filter(|&&m| m < 0.5 * PLANTED_ACTIVE_MASS)
        .count();
    let n_recovered = theta_max.iter().filter(|&&t| t < 0.25).count();

    // ---- VERBATIM diagnostic dump -----------------------------------------
    println!(
        "=== SAE K-ladder recovery (IBP-MAP, production cold driver) K={k}, p={p}, N={} ===",
        truth.n
    );
    println!(
        "signal_scale={signal_scale:.6}  noise_sigma={:.6}",
        0.04 * signal_scale
    );
    println!(
        "determinism: |final1-final2|={determinism_gap:.3e} final_bits_eq={final_bit_identical} coords_bits_eq={coords_bit_identical} masses_bits_eq={mass_bit_identical} => deterministic={deterministic}"
    );
    println!(
        "active mass: mean={mean_mass:.6} min={min_mass:.6} planted={PLANTED_ACTIVE_MASS:.4} collapse_floor={COLLAPSE_FLOOR:.4}  (n_collapsed={n_collapsed}/{k})"
    );
    println!(
        "principal angle (max per atom, rad): mean={mean_theta:.6} worst={worst_theta:.6}  (n_recovered<0.25rad={n_recovered}/{k})"
    );
    println!("reconstruction R2={r2:.6}  (ssr={ssr:.4}, sst={sst:.4})");
    // print a representative sample of per-atom numbers verbatim.
    let sample: Vec<usize> = {
        let mut v: Vec<usize> = Vec::new();
        let step = (k / 8).max(1);
        let mut a = 0;
        while a < k && v.len() < 8 {
            v.push(a);
            a += step;
        }
        if !v.contains(&(k - 1)) {
            v.push(k - 1);
        }
        v
    };
    for &t in &sample {
        println!(
            "  atom {t}: fitted<-{}  n_active={}  active_mass={:.6} (ratio={:.4})  theta_max={:.6}rad",
            matched[t],
            n_active[t],
            fitted_mass[t],
            fitted_mass[t] / PLANTED_ACTIVE_MASS,
            theta_max[t],
        );
    }

    // ---- assertions --------------------------------------------------------
    // (a) NO COLLAPSE: no atom's active mass may crash toward the ~0.03 floor;
    //     each must stay within a principled band of the planted ~0.2 mass.
    //     Lower band = halfway between the collapse floor and half-planted so a
    //     single collapsed atom trips it; we additionally require it above the
    //     raw collapse floor with margin.
    let lower_band = 0.5 * PLANTED_ACTIVE_MASS; // 0.10, well above 0.03 collapse
    assert!(
        min_mass >= lower_band,
        "COLLAPSE at K={k}: min active mass {min_mass:.6} < band {lower_band:.4} \
         (planted {PLANTED_ACTIVE_MASS}, collapse floor {COLLAPSE_FLOOR}); \
         {n_collapsed}/{k} atoms collapsed; mean mass {mean_mass:.6}"
    );
    assert!(
        min_mass > COLLAPSE_FLOOR * 2.0,
        "COLLAPSE at K={k}: min active mass {min_mass:.6} within 2x the collapse floor {COLLAPSE_FLOOR}"
    );
    // (b) RECOVERY: every matched atom recovers its planted plane to a small
    //     max principal angle.
    assert!(
        worst_theta < 0.25,
        "RECOVERY FAIL at K={k}: worst max principal angle {worst_theta:.6} rad >= 0.25 \
         ({} / {k} atoms recovered < 0.25 rad; mean {mean_theta:.6})",
        n_recovered
    );
    // (c) DETERMINISM: identical seed -> bit-identical fit.
    assert!(
        deterministic,
        "DETERMINISM FAIL at K={k}: final_bits_eq={final_bit_identical} coords_bits_eq={coords_bit_identical} masses_bits_eq={mass_bit_identical} (gap {determinism_gap:.3e})"
    );
    // (d) LEDGER DETERMINISM (#992 / #976): the structure ledger the search
    //     reads — the collapse-guard event sequence — must be identical across
    //     the two runs (same breaches, same iterations, same actions, masses
    //     bit-equal), and a green rung must contain no TERMINAL collapse (a
    //     terminal event contradicts the (a)/(b) assertions above by
    //     construction). Reseed events are legal (the guard giving an atom a
    //     second basin) but must replay identically.
    let ev1 = term1.collapse_events();
    let ev2 = term2.collapse_events();
    assert_eq!(
        ev1.len(),
        ev2.len(),
        "LEDGER DETERMINISM FAIL at K={k}: {} vs {} collapse events",
        ev1.len(),
        ev2.len()
    );
    for (e1, e2) in ev1.iter().zip(ev2.iter()) {
        assert!(
            e1.iteration == e2.iteration
                && e1.atom == e2.atom
                && e1.action == e2.action
                && e1.max_active_mass.to_bits() == e2.max_active_mass.to_bits()
                && e1.floor.to_bits() == e2.floor.to_bits(),
            "LEDGER DETERMINISM FAIL at K={k}: event mismatch {e1:?} vs {e2:?}"
        );
    }
    assert!(
        !ev1.iter()
            .any(|e| e.action == gam::solver::structure_search::CollapseAction::Terminal),
        "LEDGER FAIL at K={k}: terminal collapse recorded on a rung whose mass/recovery \
         assertions passed: {ev1:?}"
    );
    // (e) ACTIVE-SET SPARSITY (#992): the fitted per-row active set must stay
    //     near the planted one — scaling K must not smear mass across the
    //     dictionary. Two checks: (i) the mean fitted active-set size (atoms
    //     whose mass exceeds half the planted level on a row) stays within a
    //     factor-2 band of the planted per-row active count; (ii) the mean
    //     OFF-SUPPORT mass (assignment mass on (row, atom) pairs the plant
    //     left inactive) stays below a quarter of the planted active mass —
    //     the smear statistic that row-level set sizes alone can miss.
    let set_threshold = 0.5 * PLANTED_ACTIVE_MASS;
    let mut fitted_set_total = 0usize;
    let mut planted_set_total = 0usize;
    let mut off_mass_acc = 0.0_f64;
    let mut off_pairs = 0usize;
    for i in 0..truth.n {
        for t in 0..k {
            let f = matched[t];
            if truth.active[t][i] {
                planted_set_total += 1;
            } else {
                off_mass_acc += assign1[[i, f]];
                off_pairs += 1;
            }
            if assign1[[i, f]] > set_threshold {
                fitted_set_total += 1;
            }
        }
    }
    let mean_fitted_set = fitted_set_total as f64 / truth.n as f64;
    let mean_planted_set = planted_set_total as f64 / truth.n as f64;
    let mean_off_mass = if off_pairs > 0 {
        off_mass_acc / off_pairs as f64
    } else {
        0.0
    };
    println!(
        "active set: mean fitted size={mean_fitted_set:.3} planted={mean_planted_set:.3} \
         (threshold {set_threshold:.3}); mean off-support mass={mean_off_mass:.6}"
    );
    assert!(
        mean_fitted_set <= 2.0 * mean_planted_set && mean_fitted_set >= 0.5 * mean_planted_set,
        "SPARSITY FAIL at K={k}: mean fitted active-set size {mean_fitted_set:.3} outside \
         [0.5, 2.0]x planted {mean_planted_set:.3}"
    );
    assert!(
        mean_off_mass <= 0.25 * PLANTED_ACTIVE_MASS,
        "SPARSITY FAIL at K={k}: mean off-support mass {mean_off_mass:.6} exceeds a quarter \
         of the planted active mass {PLANTED_ACTIVE_MASS}"
    );
    // Reconstruction sanity: the dictionary must explain the planted signal.
    assert!(
        r2 >= 0.85,
        "RECONSTRUCTION FAIL at K={k}: R2 {r2:.6} < 0.85"
    );
}

#[test]
fn sae_manifold_k_ladder_recovery_k64() {
    // 64 mutually-orthogonal-ish planted circle atoms: 2K=128 spanning columns
    // need p >= 128; scale N so every atom sees enough active rows.
    let rung = Rung {
        k: 64,
        p: 160,
        n: 4000,
    };
    run_rung(&rung);
}

#[test]
fn sae_manifold_k_ladder_recovery_k1024() {
    // 1024 planted circle atoms: 2K=2048 spanning columns need p >= 2048; N
    // scaled so each atom has a healthy active support.
    let rung = Rung {
        k: 1024,
        p: 2100,
        n: 40000,
    };
    run_rung(&rung);
}

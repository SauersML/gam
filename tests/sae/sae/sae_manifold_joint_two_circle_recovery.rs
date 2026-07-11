//! Regression pin for the "K≥2 SAE joint fit collapses cold" failure
//! (#853 class). Two planted circle atoms are recovered against the planted
//! truth, driving the fit *exactly the way production does*: cold IBP-MAP
//! residual-energy seed logits, weighted-LSQ decoder init, and the generic
//! outer cascade (`OuterProblem::run`) around `SaeManifoldOuterObjective` —
//! the same engine `crates/gam-pyffi` `sae_manifold_fit_minimal` drives.
//!
//! The collapse signature is per-atom mean active mass crashing to ~0.03 on
//! the rows where the atom is *truly* active (vs a planted ~0.2). A failing
//! run here REPRODUCES that; a passing run REFUTES it. Either way the test
//! prints verbatim numbers.
//!
//! Construction-path fidelity note: the gam crate cannot reach the pyffi-only
//! seed helpers (`sae_pca_seed_initial_coords` cluster-refine,
//! `sae_residual_seed_logits`, `sae_decoder_lsq_init`,
//! `sae_refine_routing_seed`). We replicate the two seed stages that
//! determine the routing collapse VERBATIM from those bodies (residual-energy
//! IBP logits at gain 4.0; weighted-LSQ decoder init at the IBP gate), and
//! seed the latent coordinates from the planted angles (the production PCA /
//! cluster coordinate seed is a separate stage; #853 is a routing/active-mass
//! failure, not a coordinate-recovery one — mirroring the inline torus oracle
//! `ordered_beta_bernoulli_k2_periodic_torus_recovers_signal_with_lsq_init`).

use gam::linalg::faer_ndarray::{FaerCholesky, FaerSvd, fast_ata, fast_atb};
use gam::solver::rho_optimizer::OuterProblem;
use gam::terms::latent::LatentManifold;
use gam::terms::sae::manifold::CurvatureWalkReport;
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
const N: usize = 600;
const P: usize = 24;
const K: usize = 2;
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
const R_A: f64 = 1.0;
const R_B: f64 = 1.1;
const PLANTED_ACTIVE_MASS: f64 = 0.2; // spec: codes'-units planted mean

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

/// Two p×2 orthonormal frames whose column spans are mutually orthogonal.
/// Columns 0..2 -> atom A plane, columns 2..4 -> atom B plane, of an
/// orthonormalized deterministic ambient basis.
fn planted_frames() -> (Array2<f64>, Array2<f64>) {
    // Build 4 deterministic ambient vectors, Gram-Schmidt to orthonormal.
    let mut raw = Array2::<f64>::zeros((P, 4));
    for j in 0..4 {
        for i in 0..P {
            // smooth, distinct, full-rank deterministic columns
            raw[[i, j]] = ((i as f64 + 1.0) * 0.37 * (j as f64 + 1.0)).sin()
                + 0.5 * ((i as f64) * 0.11 - (j as f64) * 0.9).cos();
        }
    }
    let mut q = Array2::<f64>::zeros((P, 4));
    for j in 0..4 {
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
    let u_a = q.slice(s![.., 0..2]).to_owned();
    let u_b = q.slice(s![.., 2..4]).to_owned();
    (u_a, u_b)
}

/// Per-row truth: angles, gates (truly-active flags), and amplitudes.
/// 30% only-A, 30% only-B, 40% co-active, partitioned by index.
struct Truth {
    theta: [Vec<f64>; K],
    active: [Vec<bool>; K],
    amp: [Vec<f64>; K],
}

fn planted_truth() -> Truth {
    let mut theta = [vec![0.0; N], vec![0.0; N]];
    let mut active = [vec![false; N], vec![false; N]];
    let mut amp = [vec![0.0; N], vec![0.0; N]];
    for i in 0..N {
        // distinct irrational-ish strides so angles fill the circle
        theta[0][i] = ((i as f64) * 0.061_803 + 0.13).rem_euclid(1.0);
        theta[1][i] = ((i as f64) * 0.098_765 + 0.57).rem_euclid(1.0);
        let bucket = i % 10;
        let (a_on, b_on) = if bucket < 3 {
            (true, false) // 30% only-A
        } else if bucket < 6 {
            (false, true) // 30% only-B
        } else {
            (true, true) // 40% co-active
        };
        active[0][i] = a_on;
        active[1][i] = b_on;
        // mild amplitude spread ~1
        amp[0][i] = if a_on {
            0.85 + 0.30 * idx_uniform(i as u64 * 2 + 1)
        } else {
            0.0
        };
        amp[1][i] = if b_on {
            0.85 + 0.30 * idx_uniform(i as u64 * 2 + 2)
        } else {
            0.0
        };
    }
    Truth { theta, active, amp }
}

/// Planted response Z = Σ_k gate_k · amp_k · r_k (cosθ u_k1 + sinθ u_k2) + noise.
fn planted_response(truth: &Truth, u_a: &Array2<f64>, u_b: &Array2<f64>) -> (Array2<f64>, f64) {
    let frames = [u_a, u_b];
    let radii = [R_A, R_B];
    let mut z = Array2::<f64>::zeros((N, P));
    let mut signal_sq = 0.0_f64;
    for i in 0..N {
        for k in 0..K {
            if !truth.active[k][i] {
                continue;
            }
            let ang = std::f64::consts::TAU * truth.theta[k][i];
            let c = ang.cos();
            let s = ang.sin();
            let scale = truth.amp[k][i] * radii[k];
            for col in 0..P {
                let contrib = scale * (c * frames[k][[col, 0]] + s * frames[k][[col, 1]]);
                z[[i, col]] += contrib;
                signal_sq += contrib * contrib;
            }
        }
    }
    let signal_scale = (signal_sq / (N * P) as f64).sqrt();
    let sigma = 0.04 * signal_scale; // ~4% of signal scale
    for i in 0..N {
        for col in 0..P {
            let u = idx_uniform(((i * P + col) as u64) * 7 + 3);
            let u2 = idx_uniform(((i * P + col) as u64) * 7 + 5);
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
fn build_cold_term(truth: &Truth, z: ArrayView2<'_, f64>) -> SaeManifoldTerm {
    let evaluator = PeriodicHarmonicEvaluator::new(M).unwrap();
    // Seed latent coordinates from the planted angles (slightly offset, per the
    // inline torus oracle, so coordinate recovery is not what is under test).
    let mut coords_k: Vec<Array2<f64>> = Vec::with_capacity(K);
    let mut phi_k: Vec<Array2<f64>> = Vec::with_capacity(K);
    let mut jet_k: Vec<Array3<f64>> = Vec::with_capacity(K);
    let offsets = [0.05_f64, 0.07_f64];
    for k in 0..K {
        let coords = Array2::from_shape_fn((N, 1), |(i, _)| {
            (truth.theta[k][i] + offsets[k]).rem_euclid(1.0)
        });
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        coords_k.push(coords);
        phi_k.push(phi);
        jet_k.push(jet);
    }

    let basis_sizes = [M, M];
    // (K, N, M) padded basis-value stack for the seed ports.
    let mut basis_values = Array3::<f64>::zeros((K, N, M));
    for k in 0..K {
        for row in 0..N {
            for c in 0..M {
                basis_values[[k, row, c]] = phi_k[k][[row, c]];
            }
        }
    }
    // Production cold IBP-MAP routing seed + weighted-LSQ decoder init.
    let logits = residual_seed_logits(basis_values.view(), &basis_sizes, z, RESIDUAL_SEED_GAIN);
    let decoder = decoder_lsq_init(basis_values.view(), &basis_sizes, z, logits.view(), TAU);

    let mut atoms = Vec::with_capacity(K);
    for k in 0..K {
        let b = decoder.slice(s![k, 0..M, ..]).to_owned();
        let atom = SaeManifoldAtom::new(
            format!("circle_{k}"),
            SaeAtomBasisKind::Periodic,
            1,
            phi_k[k].clone(),
            jet_k[k].clone(),
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
        vec![LatentManifold::Circle { period: 1.0 }; K],
        AssignmentMode::ordered_beta_bernoulli(TAU, ALPHA, false),
    )
    .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
}

/// Drive the fit through the production outer engine; return the fitted term,
/// the converged assignments, and the engine's final criterion value.
fn run_production_fit(truth: &Truth, z: &Array2<f64>) -> (SaeManifoldTerm, Array2<f64>, f64) {
    let term = build_cold_term(truth, z.view());
    let init_rho = dimensionless_entry_rho(&term, z);
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
        .run(&mut objective, "SAE manifold two-circle recovery")
        .expect("outer cascade must complete");
    objective
        .certify_outer_result(&result)
        .expect("two-circle outer result must certify the installed state");
    let fitted_term = objective
        .into_fitted()
        .expect("outer fit was evaluated")
        .term;
    let assignments = fitted_term.assignment.assignments();
    (fitted_term, assignments, result.final_value)
}

/// Decoder sin/cos rows span the fitted atom's circle plane. Returns the p×2
/// orthonormalized plane basis (cos-row, sin-row order matches u_k1, u_k2).
fn fitted_plane(atom_decoder: ArrayView2<'_, f64>) -> Array2<f64> {
    // Periodic basis layout: row0=const, row1=sin, row2=cos.
    // g(θ)=B^T φ = B_const + sinθ·B_sin + cosθ·B_cos, plane = span(B_cos, B_sin)
    let cos_row = atom_decoder.row(2).to_owned();
    let sin_row = atom_decoder.row(1).to_owned();
    let mut plane = Array2::<f64>::zeros((P, 2));
    for i in 0..P {
        plane[[i, 0]] = cos_row[i];
        plane[[i, 1]] = sin_row[i];
    }
    // Gram-Schmidt orthonormalize the two columns.
    let n0 = plane.column(0).iter().map(|x| x * x).sum::<f64>().sqrt();
    for i in 0..P {
        plane[[i, 0]] /= n0.max(1.0e-300);
    }
    let dot: f64 = (0..P).map(|i| plane[[i, 0]] * plane[[i, 1]]).sum();
    for i in 0..P {
        plane[[i, 1]] -= dot * plane[[i, 0]];
    }
    let n1 = plane.column(1).iter().map(|x| x * x).sum::<f64>().sqrt();
    for i in 0..P {
        plane[[i, 1]] /= n1.max(1.0e-300);
    }
    plane
}

/// σ_min of U_fitᵀ U_true (∈ [0,1]); 1 ⇒ identical span.
fn plane_sigma_min(u_fit: &Array2<f64>, u_true: &Array2<f64>) -> f64 {
    let m = u_fit.t().dot(u_true);
    let (_u, sv, _vt) = m.svd(false, false).expect("plane SVD");
    sv.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Circular correlation of fitted vs planted θ up to O(2) gauge: Procrustes on
/// (cosθ, sinθ) then correlation of the aligned unit vectors, restricted to the
/// truly-active rows. Returns value in [-1, 1] (≈1 ⇒ recovered).
fn circular_corr_active(theta_true: &[f64], theta_fit: &[f64], active: &[bool]) -> f64 {
    let idx: Vec<usize> = (0..theta_true.len()).filter(|&i| active[i]).collect();
    if idx.len() < 3 {
        return 0.0;
    }
    // Embed both on the circle.
    let mut at = Array2::<f64>::zeros((idx.len(), 2));
    let mut af = Array2::<f64>::zeros((idx.len(), 2));
    for (r, &i) in idx.iter().enumerate() {
        let tt = std::f64::consts::TAU * theta_true[i];
        let tf = std::f64::consts::TAU * theta_fit[i];
        at[[r, 0]] = tt.cos();
        at[[r, 1]] = tt.sin();
        af[[r, 0]] = tf.cos();
        af[[r, 1]] = tf.sin();
    }
    // Orthogonal Procrustes: R = U Vᵀ from SVD of afᵀ·at (rotation+reflection
    // = the O(2) circle gauge). Align fitted to true, then correlate.
    let cross = af.t().dot(&at);
    let (u, _sv, vt) = cross.svd(true, true).expect("procrustes SVD");
    let u = u.expect("U");
    let vt = vt.expect("Vt");
    let rot = u.dot(&vt); // 2x2
    let aligned = af.dot(&rot);
    // Pearson correlation between aligned fitted embedding and true embedding,
    // flattened over the 2 circle channels (gauge-invariant similarity).
    let n = (idx.len() * 2) as f64;
    let mut mx = 0.0;
    let mut my = 0.0;
    for r in 0..idx.len() {
        for c in 0..2 {
            mx += aligned[[r, c]];
            my += at[[r, c]];
        }
    }
    mx /= n;
    my /= n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for r in 0..idx.len() {
        for c in 0..2 {
            let dx = aligned[[r, c]] - mx;
            let dy = at[[r, c]] - my;
            sxy += dx * dy;
            sxx += dx * dx;
            syy += dy * dy;
        }
    }
    sxy / (sxx.sqrt() * syy.sqrt()).max(1.0e-300)
}

#[test]
fn sae_manifold_joint_two_circle_recovery_ordered_beta_bernoulli() {
    let (u_a, u_b) = planted_frames();
    let truth = planted_truth();
    let (z, signal_scale) = planted_response(&truth, &u_a, &u_b);

    // ---- determinism: identical cold fit twice, criterion to ~1e-12 -------
    let (term1, assign1, final1) = run_production_fit(&truth, &z);
    let (_term2, _assign2, final2) = run_production_fit(&truth, &z);
    let determinism_gap = (final1 - final2).abs();
    let deterministic = determinism_gap <= 1.0e-12;

    // ---- Hungarian (2x2 brute force) match fitted -> planted ---------------
    let planted_planes = [u_a.clone(), u_b.clone()];
    let fitted_planes: Vec<Array2<f64>> = (0..K)
        .map(|k| fitted_plane(term1.atoms[k].decoder_coefficients.view()))
        .collect();
    // cost[f][t] = 1 - sigma_min(plane match); pick the permutation minimizing total.
    let mut cost = [[0.0_f64; K]; K];
    for f in 0..K {
        for t in 0..K {
            cost[f][t] = 1.0 - plane_sigma_min(&fitted_planes[f], &planted_planes[t]);
        }
    }
    // Two permutations for K=2.
    let perm_id_cost = cost[0][0] + cost[1][1];
    let perm_swap_cost = cost[0][1] + cost[1][0];
    // match[t] = fitted atom index assigned to planted atom t.
    let matched: [usize; K] = if perm_id_cost <= perm_swap_cost {
        [0, 1]
    } else {
        [1, 0]
    };

    // ---- per-atom metrics on truly-active rows -----------------------------
    let mut sigma_min = [0.0_f64; K];
    let mut circ_corr = [0.0_f64; K];
    let mut planted_mass = [0.0_f64; K];
    let mut fitted_mass = [0.0_f64; K];
    let mut n_active = [0usize; K];

    // fitted per-atom latent coords (for circular correlation).
    let fitted_coords: Vec<Vec<f64>> = (0..K)
        .map(|k| {
            let c = term1.assignment.coords[k].as_matrix();
            (0..N).map(|i| c[[i, 0]]).collect()
        })
        .collect();

    for t in 0..K {
        let f = matched[t];
        sigma_min[t] = plane_sigma_min(&fitted_planes[f], &planted_planes[t]);
        circ_corr[t] = circular_corr_active(&truth.theta[t], &fitted_coords[f], &truth.active[t]);
        // active-mass on truly-active rows: mean fitted gate vs planted reference.
        let mut acc = 0.0;
        let mut cnt = 0usize;
        for i in 0..N {
            if truth.active[t][i] {
                acc += assign1[[i, f]];
                cnt += 1;
            }
        }
        n_active[t] = cnt;
        fitted_mass[t] = if cnt > 0 { acc / cnt as f64 } else { 0.0 };
        planted_mass[t] = PLANTED_ACTIVE_MASS;
    }

    // ---- reconstruction R^2 -----------------------------------------------
    let fitted = term1.fitted();
    let mut ssr = 0.0;
    let mut sst = 0.0;
    let mut zbar = 0.0;
    for i in 0..N {
        for j in 0..P {
            zbar += z[[i, j]];
        }
    }
    zbar /= (N * P) as f64;
    for i in 0..N {
        for j in 0..P {
            let r = z[[i, j]] - fitted[[i, j]];
            ssr += r * r;
            let d = z[[i, j]] - zbar;
            sst += d * d;
        }
    }
    let r2 = 1.0 - ssr / sst.max(1.0e-12);

    // ---- VERBATIM diagnostic dump -----------------------------------------
    println!("=== SAE two-circle recovery (IBP-MAP, production cold driver) ===");
    println!(
        "signal_scale={signal_scale:.6}  noise_sigma={:.6}",
        0.04 * signal_scale
    );
    println!(
        "Hungarian match planted->fitted: A<-{}  B<-{}",
        matched[0], matched[1]
    );
    println!(
        "determinism: |final1-final2|={determinism_gap:.3e} (final1={final1:.12}, final2={final2:.12}) deterministic={deterministic}"
    );
    for t in 0..K {
        let label = if t == 0 { "A" } else { "B" };
        println!(
            "atom {label}: n_active={}  active_mass planted={:.4} fitted={:.6}  (ratio={:.4})  sigma_min={:.6}  circ_corr={:.6}",
            n_active[t],
            planted_mass[t],
            fitted_mass[t],
            fitted_mass[t] / planted_mass[t],
            sigma_min[t],
            circ_corr[t],
        );
    }
    println!("reconstruction R2={r2:.6}  (ssr={ssr:.4}, sst={sst:.4})");

    // ---- assertions --------------------------------------------------------
    let mut failures: Vec<usize> = Vec::new();
    // 1. No collapse: fitted active-mass on truly-active rows >= 50% planted.
    for t in 0..K {
        if !(fitted_mass[t] >= 0.5 * PLANTED_ACTIVE_MASS) {
            if !failures.contains(&1) {
                failures.push(1);
            }
        }
    }
    // 2. Plane recovery: sigma_min > 0.95 per matched atom.
    for t in 0..K {
        if !(sigma_min[t] > 0.95) && !failures.contains(&2) {
            failures.push(2);
        }
    }
    // 3. Coordinate recovery: circular corr >= 0.9 per atom on active rows.
    for t in 0..K {
        if !(circ_corr[t] >= 0.9) && !failures.contains(&3) {
            failures.push(3);
        }
    }
    // 4. Reconstruction R^2 >= 0.9.
    if !(r2 >= 0.9) {
        failures.push(4);
    }
    // 5. Determinism: identical fits agree to ~1e-12.
    if !deterministic {
        failures.push(5);
    }

    assert!(
        failures.is_empty(),
        "SAE two-circle recovery FAILED assertions {failures:?}; \
         active_mass fitted=[{:.6}, {:.6}] (planted {PLANTED_ACTIVE_MASS}); \
         sigma_min=[{:.6}, {:.6}]; circ_corr=[{:.6}, {:.6}]; R2={r2:.6}; \
         determinism_gap={determinism_gap:.3e}",
        fitted_mass[0],
        fitted_mass[1],
        sigma_min[0],
        sigma_min[1],
        circ_corr[0],
        circ_corr[1],
    );
}

/// Build the production objective for the two-circle fixture (same cold term and
/// entry ρ as `run_production_fit`), without driving the outer cascade — the
/// caller exercises the curvature-homotopy entry walk directly.
fn build_objective(truth: &Truth, z: &Array2<f64>) -> (SaeManifoldOuterObjective, Array1<f64>) {
    let term = build_cold_term(truth, z.view());
    let init_rho = dimensionless_entry_rho(&term, z);
    let init_rho_flat = init_rho.to_flat();
    (
        SaeManifoldOuterObjective::new(
            term,
            z.clone(),
            None,
            init_rho,
            INNER_MAX_ITER,
            LEARNING_RATE,
            RIDGE_EXT_COORD,
            RIDGE_BETA,
        ),
        init_rho_flat,
    )
}

fn dimensionless_entry_rho(term: &SaeManifoldTerm, z: &Array2<f64>) -> SaeManifoldRho {
    let seed_dispersion = term
        .seed_reconstruction_dispersion(z.view())
        .expect("seed reconstruction dispersion");
    assert!(seed_dispersion.is_finite() && seed_dispersion > 0.0);
    SaeManifoldRho::new(
        SPARSITY.ln(),
        SMOOTHNESS.ln(),
        vec![Array1::<f64>::zeros(0); K],
    )
    .seed_scaled_by_dispersion(seed_dispersion)
    .expect("dimensionless seed scaling by the profiled reconstruction dispersion")
}

#[test]
fn sae_two_circle_seed_dispersion_diagnostic() {
    let (u_a, u_b) = planted_frames();
    let truth = planted_truth();
    let (z, signal_scale) = planted_response(&truth, &u_a, &u_b);
    let term = build_cold_term(&truth, z.view());
    let seed_dispersion = term
        .seed_reconstruction_dispersion(z.view())
        .expect("seed reconstruction dispersion");
    let seed_r2 = {
        let fitted = term.fitted();
        let mut ssr = 0.0;
        let mut sst = 0.0;
        let mut zbar = 0.0;
        for i in 0..N {
            for j in 0..P {
                zbar += z[[i, j]];
            }
        }
        zbar /= (N * P) as f64;
        for i in 0..N {
            for j in 0..P {
                let r = z[[i, j]] - fitted[[i, j]];
                ssr += r * r;
                let d = z[[i, j]] - zbar;
                sst += d * d;
            }
        }
        1.0 - ssr / sst.max(1.0e-12)
    };
    println!(
        "two-circle seed signal_scale={signal_scale:.6} seed_phi={seed_dispersion:.6e} seed_r2={seed_r2:.6}"
    );
    assert!(seed_dispersion.is_finite() && seed_dispersion > 0.0);
}

/// #1007 walk-path oracle: the certified curvature-homotopy entry walk reaches
/// the gate-0 two-circle fixture's optimum from the Eckart-Young anchor with
/// ZERO reseeds, no recorded bifurcation, and no inner active-mass collapse —
/// the certified-anchor replacement for the blind multi-seed multistart.
///
/// This drives `run_curvature_homotopy_entry` directly (the same call the outer
/// seed loop makes as its entry leg) and asserts:
///   1. the walk ARRIVED at `η = 1` on the certified optimal branch;
///   2. it recorded NO bifurcation (the arrow-factor min pivot stayed above the
///      safe-SPD floor across the whole walk);
///   3. it triggered ZERO scaffold reseeds and observed NO inner collapse
///      events — a clean walk from the global anchor does not need them;
///   4. the post-walk reconstruction recovers the planted two-plane structure
///      (R² high), i.e. the certified branch reached a genuinely good optimum,
///      not merely "arrived".
#[test]
fn sae_two_circle_curvature_homotopy_entry_arrives_zero_reseed() {
    let (u_a, u_b) = planted_frames();
    let truth = planted_truth();
    let (z, signal_scale) = planted_response(&truth, &u_a, &u_b);

    let (mut objective, init_rho_flat) = build_objective(&truth, &z);
    let arrived = objective
        .run_curvature_homotopy_entry()
        .expect("curvature-homotopy entry walk must not hard-error on the gate-0 fixture");

    let report: CurvatureWalkReport = objective
        .curvature_walk_report()
        .expect("a walk that ran must record a report")
        .clone();

    // Re-converge the η=1 arrival at the same fixed ρ through the one public
    // fixed-ρ fit entry. The homotopy report certifies the branch walk, while
    // `fit_at_fixed_rho` is the ownership gate that may mint a fitted object.
    objective
        .fit_at_fixed_rho(init_rho_flat.view())
        .expect("homotopy arrival must converge at its fixed rho");
    // Post-walk reconstruction R² at the certified η = 1 state.
    let fitted_term = objective
        .into_fitted()
        .expect("outer fit was evaluated")
        .term;
    let fitted = fitted_term.fitted();
    let mut ssr = 0.0;
    let mut sst = 0.0;
    let mut zbar = 0.0;
    for i in 0..N {
        for j in 0..P {
            zbar += z[[i, j]];
        }
    }
    zbar /= (N * P) as f64;
    for i in 0..N {
        for j in 0..P {
            let r = z[[i, j]] - fitted[[i, j]];
            ssr += r * r;
            let d = z[[i, j]] - zbar;
            sst += d * d;
        }
    }
    let r2 = 1.0 - ssr / sst.max(1.0e-12);

    println!("=== SAE two-circle curvature-homotopy entry walk (#1007) ===");
    println!("signal_scale={signal_scale:.6}");
    println!(
        "walk: arrived={} eta_steps={} step_halvings={} reseeds={} collapse_events={} \
         anchor_residual_norm_sq={:.6e} bifurcation={:?}",
        report.arrived,
        report.eta_steps,
        report.step_halvings,
        report.reseeds,
        report.collapse_events,
        report.anchor_residual_norm_sq,
        report.bifurcation,
    );
    println!("post-walk reconstruction R2={r2:.6}");

    let mut failures: Vec<&str> = Vec::new();
    if !arrived || !report.arrived {
        failures.push("walk did not arrive at eta=1 on the certified branch");
    }
    if report.bifurcation.is_some() {
        failures.push("walk recorded a branch bifurcation");
    }
    if report.reseeds != 0 {
        failures.push("walk triggered a scaffold reseed");
    }
    if report.collapse_events != 0 {
        failures.push("walk observed an inner active-mass collapse");
    }
    if !(report.eta_steps >= 1) {
        failures.push("walk recorded no accepted eta waypoint");
    }
    if !(r2 >= 0.9) {
        failures.push("post-walk reconstruction R2 below 0.9");
    }

    assert!(
        failures.is_empty(),
        "SAE two-circle curvature-homotopy entry FAILED {failures:?}; \
         arrived={arrived}/{} eta_steps={} step_halvings={} reseeds={} collapse_events={} \
         bifurcation={:?} R2={r2:.6}",
        report.arrived,
        report.eta_steps,
        report.step_halvings,
        report.reseeds,
        report.collapse_events,
        report.bifurcation,
    );
}

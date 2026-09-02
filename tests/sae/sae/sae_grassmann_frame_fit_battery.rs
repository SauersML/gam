//! Fit-level verification battery for the #972 Grassmann decoder frames
//! (issue #992, items 2 and 3).
//!
//! * **Evidence equality at small p** — a dictionary of truly low-rank atoms
//!   evaluated through the SAME criterion entry (`penalized_quasi_laplace_criterion`) twice: once
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
use gam::solver::rho_optimizer::OuterProblem;
use gam::terms::latent::LatentManifold;
use gam::terms::sae::manifold::{GrassmannCrossMoment, GrassmannFrame};
use gam::terms::{
    sae::manifold::AssignmentMode, sae::manifold::PeriodicHarmonicEvaluator,
    sae::manifold::SaeAssignment, sae::manifold::SaeAtomBasisKind,
    sae::manifold::SaeBasisEvaluator, sae::manifold::SaeManifoldAtom,
    sae::manifold::SaeManifoldOuterObjective, sae::manifold::SaeManifoldRho,
    sae::manifold::SaeManifoldTerm,
};
use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3};
use std::sync::Arc;

use faer::Side as FaerSide;

const M: usize = 3; // const + 1 harmonic -> circle basis
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const INNER_MAX_ITER: usize = 20;
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

use super::shared_seed_fixtures::residual_seed_logits;

/// Weighted-LSQ decoder seed at the ordered independent Beta--Bernoulli gate (the production
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
        let atom = SaeManifoldAtom::new_with_provided_function_gram(
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

/// Drive a term through the production outer engine (the only blessed fit
/// entry; it handles the inner-solve ridging that reaches a PD basin). Returns
/// the fitted term and the engine's converged criterion value. Empty ARD axes
/// mirror the proven small-SAE fixtures.
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
    // Bound the outer cascade: the planted signal is clean, so a capped fit
    // recovers the planes and reaches a PD basin without the multi-minute
    // full-convergence the larger reference fixtures pay for.
    let problem = OuterProblem::new(n_params)
        .with_initial_rho(init_rho_flat)
        .with_max_iter(25);
    let result = problem
        .run(&mut objective, label)
        .expect("outer cascade must complete");
    objective
        .certify_outer_result(&result)
        .expect("Grassmann-frame outer result must certify the installed state");
    let fitted = objective
        .into_fitted()
        .expect("outer fit was evaluated")
        .term;
    (fitted, result.final_value)
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


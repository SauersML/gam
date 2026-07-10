//! #1026 — MEASURED reconstruction parity: hybrid curved + linear-tail
//! dictionary vs a pure-linear SAE baseline, at matched dictionary size K.
//!
//! This is the discriminating measurement the #1026 roadmap calls for, run end
//! to end through the SAME production engine the recovery pins use
//! (`SaeManifoldOuterObjective` + the generic `OuterProblem::run` cascade, cold
//! IBP-MAP residual-energy seed logits + weighted-LSQ decoder init at gain 4.0 /
//! τ = 0.5). The DGP is the proven planted-circle dictionary from
//! `sae_manifold_k_ladder_recovery` (K genuinely-curved circle features —
//! integrated turning Θ = 2π each — embedded in mutually-orthogonal planted
//! frames of a larger ambient space p, sparse per-row gates, ~4 % noise).
//!
//! The hybrid-dominance argument (issue #1026, high-confidence half): a
//! dictionary whose atom set INCLUDES the linear atom as the Θ = 0 special case
//! cannot lose to a pure-linear dictionary at matched active budget. Its
//! quantitative core: a linear SAE shatters a curved feature of turning Θ into
//!
//!     N(ε) ≈ Θ / (2 √(2ε))
//!
//! rank-1 directions to hit relative reconstruction error ε (radius cancels —
//! scale-free). For a full hue circle (Θ = 2π) at ε ≈ 0.02 this is ≈ 16; at
//! ε ≈ 0.05, ≈ 10. So the measured, falsifiable parity statement is:
//!
//!   1. CURVED match-or-beats LINEAR at matched K: one periodic atom per circle
//!      reconstructs the curved DGP; one linear atom per circle can only fit a
//!      secant (a diameter) and is starved — EV(curved) ≥ EV(linear) at the
//!      SAME K, by a wide margin (the shatter penalty, measured).
//!   2. CURVED clears a high absolute reconstruction bar on this clean DGP.
//!   3. LINEAR needs the SHATTER BUDGET (≈ N(ε)·K linear atoms) to climb back to
//!      curved parity — quantitatively confirming the Θ/√ε law: the same linear
//!      basis, given ~10× the atoms, finally reaches EV within a small margin of
//!      the K-atom curved dictionary.
//!
//! All three dictionaries are fit identically; the only difference is the atom
//! basis (periodic vs euclidean-degree-1) and, for arm 3, the atom count. The
//! reconstruction metric is the per-column explained variance the recovery pins
//! report (`1 − SSR/SST`), measured the same way for every arm so the
//! comparison is unbiased.

use gam::linalg::faer_ndarray::{FaerCholesky, fast_ata, fast_atb};
use gam::solver::rho_optimizer::OuterProblem;
use gam::terms::latent::LatentManifold;
use gam::terms::sae::manifold::EuclideanPatchEvaluator;
use gam::terms::{
    AssignmentMode, PeriodicHarmonicEvaluator, SaeAssignment, SaeAtomBasisKind, SaeBasisEvaluator,
    SaeManifoldAtom, SaeManifoldOuterObjective, SaeManifoldRho, SaeManifoldTerm,
};
use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3, s};
use std::sync::Arc;

use faer::Side as FaerSide;

// ---- production defaults (gamfit `sae_manifold_fit`, ibp_map path) ----------
const M_CIRCLE: usize = 3; // const + 1 harmonic (sin, cos) -> circle, Θ = 2π
const M_LINEAR: usize = 2; // const + linear monomial -> a direction, Θ = 0
const TAU: f64 = 0.5;
const ALPHA: f64 = 1.0;
const SPARSITY: f64 = 1.0;
const SMOOTHNESS: f64 = 1.0;
const INNER_MAX_ITER: usize = 50;
const LEARNING_RATE: f64 = 1.0;
const RIDGE_EXT_COORD: f64 = 1.0e-6;
const RIDGE_BETA: f64 = 1.0e-6;
const RESIDUAL_SEED_GAIN: f64 = 4.0;

// ---- planted DGP --------------------------------------------------------
const PLANTED_ACTIVE_MASS: f64 = 0.25;

/// Which basis a slot carries: a curved circle (periodic harmonic, Θ = 2π) or a
/// linear direction (euclidean degree-1 monomial, Θ = 0 — the linear-SAE atom).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Circle,
    Linear,
}

impl Kind {
    fn basis_size(self) -> usize {
        match self {
            Kind::Circle => M_CIRCLE,
            Kind::Linear => M_LINEAR,
        }
    }
    fn basis_kind(self) -> SaeAtomBasisKind {
        match self {
            Kind::Circle => SaeAtomBasisKind::Periodic,
            Kind::Linear => SaeAtomBasisKind::EuclideanPatch,
        }
    }
    fn evaluator(self) -> Arc<dyn SaeBasisEvaluator> {
        match self {
            Kind::Circle => Arc::new(PeriodicHarmonicEvaluator::new(M_CIRCLE).unwrap()),
            Kind::Linear => Arc::new(EuclideanPatchEvaluator::new(1, 1).unwrap()),
        }
    }
    fn manifold(self) -> LatentManifold {
        match self {
            Kind::Circle => LatentManifold::Circle { period: 1.0 },
            Kind::Linear => LatentManifold::Euclidean,
        }
    }
}

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
/// mutually-orthonormal-ish (Gram-Schmidt of 2K deterministic ambient vectors).
fn planted_frames(k: usize, p: usize) -> Vec<Array2<f64>> {
    let cols = 2 * k;
    assert!(cols <= p, "need ambient p >= 2K (p={p}, K={k})");
    let mut raw = Array2::<f64>::zeros((p, cols));
    for j in 0..cols {
        for i in 0..p {
            raw[[i, j]] = ((i as f64 + 1.0) * 0.37 * (j as f64 + 1.0)).sin()
                + 0.5 * ((i as f64) * 0.11 - (j as f64) * 0.9).cos()
                + 0.25 * (((i as f64) * 0.017 + (j as f64) * 0.041) * 1.7).sin();
        }
    }
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

/// Per-circle planted truth: angles θ, sparse active gates, amplitudes.
struct Truth {
    k: usize,
    n: usize,
    theta: Vec<Vec<f64>>,
    active: Vec<Vec<bool>>,
    amp: Vec<Vec<f64>>,
    radii: Vec<f64>,
}

fn planted_truth(k: usize, n: usize) -> Truth {
    let mut theta = vec![vec![0.0; n]; k];
    let mut active = vec![vec![false; n]; k];
    let mut amp = vec![vec![0.0; n]; k];
    let radii: Vec<f64> = (0..k)
        .map(|a| 1.0 + 0.1 * (a as f64 / k.max(1) as f64))
        .collect();
    let n_active_target = ((PLANTED_ACTIVE_MASS * n as f64).round() as usize).max(8);
    for a in 0..k {
        let stride = 0.045 + 0.0007 * (a as f64);
        let phase = idx_uniform(a as u64 * 7 + 11);
        for i in 0..n {
            theta[a][i] = ((i as f64) * stride + phase).rem_euclid(1.0);
        }
        let base = (a * n) / k.max(1);
        for t in 0..n_active_target {
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
fn planted_response(truth: &Truth, frames: &[Array2<f64>], p: usize) -> Array2<f64> {
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
    let sigma = 0.04 * signal_scale;
    for i in 0..n {
        for col in 0..p {
            let u = idx_uniform(((i * p + col) as u64) * 7 + 3);
            let u2 = idx_uniform(((i * p + col) as u64) * 7 + 5);
            let g = (-2.0 * (u.max(1.0e-12)).ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            z[[i, col]] += sigma * g;
        }
    }
    z
}

/// VERBATIM port of pyffi `sae_residual_seed_logits` (ibp_map cold seed). Works
/// for any per-atom basis (generic over `basis_values` / `basis_sizes`).
fn residual_seed_logits(
    basis_values: ArrayView3<'_, f64>,
    basis_sizes: &[usize],
    z: ArrayView2<'_, f64>,
    gain: f64,
) -> Array2<f64> {
    let k_atoms = basis_sizes.len();
    let (n_obs, p_out) = z.dim();
    let mut logits = Array2::<f64>::zeros((n_obs, k_atoms));
    let mut resid = z.to_owned();
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
        let jitter = (trace / m_k as f64).max(1.0) * 1.0e-8;
        for i in 0..m_k {
            gram[[i, i]] += jitter;
        }
        let rhs = fast_atb(&phi, &resid);
        let b = gram
            .cholesky(FaerSide::Lower)
            .expect("residual-seed Cholesky")
            .solve_mat(&rhs);
        let fitted = phi.dot(&b);
        let mut energy = vec![0.0_f64; n_obs];
        let mut mean_energy = 0.0_f64;
        for row in 0..n_obs {
            let mut e = 0.0;
            for col in 0..p_out {
                e += fitted[[row, col]] * fitted[[row, col]];
            }
            energy[row] = e.sqrt();
            mean_energy += energy[row];
        }
        mean_energy /= n_obs as f64;
        let denom = mean_energy.max(1.0e-12);
        for row in 0..n_obs {
            logits[[row, atom_idx]] = gain * (energy[row] / denom - 1.0);
            for col in 0..p_out {
                resid[[row, col]] -= fitted[[row, col]];
            }
        }
    }
    logits
}

/// VERBATIM port of pyffi `sae_decoder_lsq_init` (ibp_map branch).
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

/// One dictionary slot: which circle it is seeded onto, its basis kind, and the
/// arc-segment it owns (for the shatter arm, each linear atom owns one secant of
/// its circle so the seed coordinate / support tiles the [0,1) angle range).
struct Slot {
    circle: usize,
    kind: Kind,
    /// Segment center in [0,1) the slot's coordinate seed is offset toward, and
    /// the half-width of the segment it is preferentially active on. A single
    /// curved atom owns the whole circle (`seg_center` arbitrary, width 0.5).
    seg_center: f64,
    seg_halfwidth: f64,
}

/// Build the cold term for an arbitrary slot list, through the production cold
/// IBP-MAP routing seed + weighted-LSQ decoder init (identical to the recovery
/// pins, generalized over heterogeneous atom bases).
fn build_cold_term(
    truth: &Truth,
    z: ArrayView2<'_, f64>,
    p: usize,
    slots: &[Slot],
) -> SaeManifoldTerm {
    let n = truth.n;
    let k_atoms = slots.len();
    assert_eq!(z.ncols(), p);

    let mut coords_k: Vec<Array2<f64>> = Vec::with_capacity(k_atoms);
    let mut phi_k: Vec<Array2<f64>> = Vec::with_capacity(k_atoms);
    let mut jet_k: Vec<Array3<f64>> = Vec::with_capacity(k_atoms);
    let basis_sizes: Vec<usize> = slots.iter().map(|s| s.kind.basis_size()).collect();
    let m_max = basis_sizes.iter().copied().max().unwrap();

    for slot in slots {
        let a = slot.circle;
        // Seed the latent coordinate from the planted angle of the slot's circle
        // (slightly offset so coordinate recovery is not what is under test). For
        // a linear secant slot, the coordinate is the SAME angle — the linear
        // atom then fits the best line through the arc it is routed onto.
        let offset = 0.04 + 0.013 * ((a % 5) as f64);
        let coords = Array2::from_shape_fn((n, 1), |(i, _)| {
            (truth.theta[a][i] + offset).rem_euclid(1.0)
        });
        let evaluator = slot.kind.evaluator();
        let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();
        coords_k.push(coords);
        phi_k.push(phi);
        jet_k.push(jet);
    }

    // (K, N, m_max) padded basis-value stack for the seed ports.
    let mut basis_values = Array3::<f64>::zeros((k_atoms, n, m_max));
    for (ai, slot) in slots.iter().enumerate() {
        let m_k = slot.kind.basis_size();
        for row in 0..n {
            for c in 0..m_k {
                basis_values[[ai, row, c]] = phi_k[ai][[row, c]];
            }
        }
    }

    let mut logits = residual_seed_logits(basis_values.view(), &basis_sizes, z, RESIDUAL_SEED_GAIN);
    // Shatter arm: bias each linear secant's seed logit toward the arc segment it
    // owns, so the K·S linear atoms tile their circle's angle range instead of
    // all collapsing onto the same diameter. A circular bump on |θ − center|.
    for (ai, slot) in slots.iter().enumerate() {
        if slot.kind == Kind::Linear && slot.seg_halfwidth < 0.49 {
            let a = slot.circle;
            for row in 0..n {
                let d = {
                    let raw = (truth.theta[a][row] - slot.seg_center).rem_euclid(1.0);
                    raw.min(1.0 - raw) // circular distance in [0, 0.5]
                };
                // +gain inside the owned segment, −gain outside (smooth ramp).
                let bump = RESIDUAL_SEED_GAIN * (slot.seg_halfwidth - d) / slot.seg_halfwidth;
                logits[[row, ai]] += bump;
            }
        }
    }
    let decoder = decoder_lsq_init(basis_values.view(), &basis_sizes, z, logits.view(), TAU);

    let mut atoms = Vec::with_capacity(k_atoms);
    for (ai, slot) in slots.iter().enumerate() {
        let m_k = slot.kind.basis_size();
        let b = decoder.slice(s![ai, 0..m_k, ..]).to_owned();
        let atom = SaeManifoldAtom::new(
            format!(
                "{}_{ai}",
                match slot.kind {
                    Kind::Circle => "circle",
                    Kind::Linear => "linear",
                }
            ),
            slot.kind.basis_kind(),
            1,
            phi_k[ai].clone(),
            jet_k[ai].clone(),
            b,
            Array2::<f64>::eye(m_k),
        )
        .unwrap()
        .with_basis_evaluator(slot.kind.evaluator());
        atoms.push(atom);
    }
    let manifolds: Vec<LatentManifold> = slots.iter().map(|s| s.kind.manifold()).collect();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords_k,
        manifolds,
        AssignmentMode::ibp_map(TAU, ALPHA, false),
    )
    .unwrap();
    SaeManifoldTerm::new(atoms, assignment).unwrap()
}

/// Drive the fit through the production outer engine; return the fitted term.
fn run_production_fit(
    truth: &Truth,
    z: &Array2<f64>,
    p: usize,
    slots: &[Slot],
    label: &str,
) -> SaeManifoldTerm {
    let k_atoms = slots.len();
    let term = build_cold_term(truth, z.view(), p, slots);
    let init_rho = SaeManifoldRho::new(
        SPARSITY.ln(),
        SMOOTHNESS.ln(),
        vec![Array1::<f64>::zeros(0); k_atoms],
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
        .expect("reconstruction-parity outer result must certify the installed state");
    let fitted_term = objective
        .into_fitted()
        .expect("outer fit was evaluated")
        .term;
    fitted_term
}

/// Reconstruction explained variance `1 − SSR/SST` (per-column centered),
/// measured identically for every arm so the comparison is unbiased.
fn reconstruction_ev(z: &Array2<f64>, fitted: &Array2<f64>) -> f64 {
    let (n, p) = z.dim();
    let mut means = vec![0.0_f64; p];
    for j in 0..p {
        let mut acc = 0.0;
        for i in 0..n {
            acc += z[[i, j]];
        }
        means[j] = acc / n as f64;
    }
    let mut ssr = 0.0_f64;
    let mut sst = 0.0_f64;
    for i in 0..n {
        for j in 0..p {
            let r = z[[i, j]] - fitted[[i, j]];
            ssr += r * r;
            let d = z[[i, j]] - means[j];
            sst += d * d;
        }
    }
    1.0 - ssr / sst.max(1.0e-12)
}

/// One curved atom per circle: the SAE-manifold dictionary at matched K.
fn curved_slots(k: usize) -> Vec<Slot> {
    (0..k)
        .map(|c| Slot {
            circle: c,
            kind: Kind::Circle,
            seg_center: 0.0,
            seg_halfwidth: 0.5,
        })
        .collect()
}

/// One linear atom per circle: the pure-linear SAE baseline at matched K.
fn linear_matched_slots(k: usize) -> Vec<Slot> {
    (0..k)
        .map(|c| Slot {
            circle: c,
            kind: Kind::Linear,
            seg_center: 0.0,
            seg_halfwidth: 0.5,
        })
        .collect()
}

/// `shards` linear atoms per circle, each owning one arc segment: the linear
/// baseline at the SHATTER BUDGET (≈ N(ε)·K atoms).
fn linear_shatter_slots(k: usize, shards: usize) -> Vec<Slot> {
    let mut slots = Vec::with_capacity(k * shards);
    let halfwidth = 0.5 / shards as f64;
    for c in 0..k {
        for s in 0..shards {
            let center = (s as f64 + 0.5) / shards as f64;
            slots.push(Slot {
                circle: c,
                kind: Kind::Linear,
                seg_center: center,
                seg_halfwidth: halfwidth,
            });
        }
    }
    slots
}

#[test]
fn sae_reconstruction_parity_curved_beats_linear_and_shatter_budget_recovers() {
    // A modest K so all three arms (including the K·shards shatter dictionary)
    // fit fast and memory-light. K=8 circles in p=48 ambient dims, n=2400
    // rows. 2K = 16 <= p, so the planted planes are mutually orthogonal.
    let k = 8usize;
    let p = 48usize;
    let n = 2400usize;
    // Shatter budget per circle for a full-turn (Θ = 2π) feature: N(ε) ≈
    // Θ/(2√(2ε)) = π/√(2ε). At ε ≈ 0.05, N ≈ 10. We give the linear baseline 10
    // secants per circle and ask it to reach within a small margin of curved.
    let shards = 10usize;
    let eps_used = {
        // The ε this shard count targets, from N = π/√(2ε)  =>  ε = π²/(2 N²).
        std::f64::consts::PI.powi(2) / (2.0 * (shards as f64).powi(2))
    };

    let truth = planted_truth(k, n);
    let frames = planted_frames(k, p);
    let z = planted_response(&truth, &frames, p);

    let curved = run_production_fit(&truth, &z, p, &curved_slots(k), "parity-curved-K");
    let linear_k = run_production_fit(&truth, &z, p, &linear_matched_slots(k), "parity-linear-K");
    let linear_shatter = run_production_fit(
        &truth,
        &z,
        p,
        &linear_shatter_slots(k, shards),
        "parity-linear-shatter",
    );

    let ev_curved = reconstruction_ev(&z, &curved.fitted());
    let ev_linear_k = reconstruction_ev(&z, &linear_k.fitted());
    let ev_linear_shatter = reconstruction_ev(&z, &linear_shatter.fitted());

    println!("=== #1026 reconstruction parity (production engine, planted K-circle DGP) ===");
    println!("K={k} circles, p={p} ambient, N={n}, Θ=2π/circle, noise≈4% of signal scale");
    println!(
        "shatter budget = {shards} linear secants/circle (targets ε≈{eps_used:.4}, N(ε)≈π/√(2ε)≈{:.1})",
        std::f64::consts::PI / (2.0 * eps_used).sqrt()
    );
    println!("reconstruction EV (1 − SSR/SST):");
    println!("  CURVED   (K={k} periodic atoms)          EV = {ev_curved:.6}");
    println!("  LINEAR   (K={k} linear atoms, matched)   EV = {ev_linear_k:.6}");
    println!(
        "  LINEAR   (K·{shards}={} linear secants)   EV = {ev_linear_shatter:.6}",
        k * shards
    );
    println!(
        "  curved − linear(matched K) margin       = {:.6}",
        ev_curved - ev_linear_k
    );
    println!(
        "  curved − linear(shatter budget) margin  = {:.6}",
        ev_curved - ev_linear_shatter
    );

    // (1) CURVED match-or-beats LINEAR at matched K — by a wide margin. One
    //     periodic atom captures a full circle; one linear atom captures only a
    //     secant. This is the shatter penalty, MEASURED. The margin bar is loose
    //     (0.10 EV) but the gap is far larger in practice.
    assert!(
        ev_curved >= ev_linear_k + 0.10,
        "PARITY (1) FAIL: curved EV {ev_curved:.6} did not beat matched-K linear EV \
         {ev_linear_k:.6} by the 0.10 shatter margin (Θ=2π features; one linear atom \
         per circle can only fit a diameter)"
    );

    // (2) CURVED clears a high absolute reconstruction bar on this clean DGP —
    //     the same 0.85 bar the K-ladder recovery pins hold (4% noise caps EV
    //     near 1; one curved atom per circle is the sufficient parameterization).
    assert!(
        ev_curved >= 0.85,
        "PARITY (2) FAIL: curved EV {ev_curved:.6} below the 0.85 reconstruction bar \
         (the curved dictionary must explain the planted curved signal)"
    );

    // (3) LINEAR reaches curved PARITY when given the SHATTER BUDGET (≈N(ε)·K
    //     atoms) — the Θ/√ε shatter law, measured: the SAME linear basis, given
    //     ~10× the atoms, climbs to within a small margin of the K-atom curved
    //     dictionary. (It need not exceed curved — a secant approximation of a
    //     curve cannot beat the exact curved atom — only reach parity.)
    assert!(
        ev_linear_shatter >= ev_curved - 0.05,
        "PARITY (3) FAIL: linear at the shatter budget (K·{shards} atoms) EV \
         {ev_linear_shatter:.6} did not reach within 0.05 of curved EV {ev_curved:.6} — \
         the Θ/√ε shatter law predicts ~{shards} linear secants recover a Θ=2π circle"
    );

    // (4) And the shatter budget must STRICTLY improve on the matched-K linear
    //     baseline (more secants = finer chord approximation = higher EV). This
    //     pins that the climb is real and monotone, not a fluke of the seed.
    assert!(
        ev_linear_shatter >= ev_linear_k + 0.10,
        "PARITY (4) FAIL: shatter-budget linear EV {ev_linear_shatter:.6} did not improve \
         on matched-K linear EV {ev_linear_k:.6} by 0.10 — adding secants must refine the \
         chord approximation of each circle"
    );
}

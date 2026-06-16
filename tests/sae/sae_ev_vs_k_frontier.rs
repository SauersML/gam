//! #1026 — the EV-vs-K discriminating frontier measurement.
//!
//! The #1026 roadmap calls this measurement "wanted regardless": the held-out
//! reconstruction explained variance EV(K) as a function of the dictionary size
//! K. Its scientific job is to DISCRIMINATE two hypotheses about an activation
//! corpus:
//!
//!   * H_flat  — the variance is mostly **unstructured / linear bulk**. Then a
//!     linear dictionary already captures it; EV(K) climbs for the first few
//!     atoms (the dominant linear directions) and then **flattens early**:
//!     doubling K past the bulk's effective rank buys almost nothing.
//!   * H_curved — the variance carries **genuine curved families** (a hue circle,
//!     Θ = 2π, etc.). A linear dictionary must *shatter* each curved feature of
//!     turning Θ into N(ε) ≈ Θ/(2√(2ε)) rank-1 secants to reach error ε, so a
//!     pure-linear EV(K) keeps climbing slowly for many atoms; a **curved /
//!     hybrid** dictionary that can spend ONE curved atom per family captures it
//!     at once and its EV(K) climbs *fast* then flattens at small K.
//!
//! So the frontier both (a) measures the climb-then-flatten shape that tells you
//! WHEN the structured part is captured, and (b) contrasts the hybrid
//! curved+linear dictionary against the pure-linear baseline at matched active
//! budget. The hybrid includes the linear atom as the Θ = 0 special case, so it
//! cannot lose at matched K (issue #1026, high-confidence dominance half) — and
//! on a curved DGP it must climb *faster* (capture the curved families with
//! fewer atoms) than the pure-linear shatter.
//!
//! This is a HELD-OUT measurement: the dictionary is fit on a TRAIN split and the
//! EV is measured on a disjoint TEST split, evaluated with the **frozen** trained
//! decoder (each trained curved/linear atom re-seated onto the test-row latent
//! coordinates; the decoder coefficients B_k are never re-fit on the test rows).
//! That makes EV(K) a genuine generalization curve, not an in-sample fit
//! statistic — the climb-then-flatten shape is the real bias/variance frontier,
//! not memorization.
//!
//! Everything runs through the SAME production engine the recovery pins use
//! (`SaeManifoldOuterObjective` + `OuterProblem::run`, cold IBP-MAP
//! residual-energy seed logits at gain 4.0, weighted-LSQ decoder init at τ = 0.5).
//! The only knob swept is K (and, for the contrast, the per-slot basis:
//! curved circle vs euclidean degree-1 linear). The reconstruction metric is the
//! per-column explained variance `1 − SSR/SST` measured identically for every K
//! and every arm.

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
const PLANTED_ACTIVE_MASS: f64 = 0.25;

/// Which basis a dictionary slot carries.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A curved circle (periodic harmonic, integrated turning Θ = 2π).
    Circle,
    /// A linear direction (euclidean degree-1 monomial, Θ = 0). This is exactly
    /// the pure-linear SAE atom — the hybrid dictionary's Θ = 0 special case.
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

/// `cols` mutually-orthonormal planted ambient directions (Gram-Schmidt of
/// deterministic ambient vectors), partitioned by the caller into curved-family
/// frames (2 columns each) and linear-bulk directions (1 column each).
fn planted_basis(cols: usize, p: usize) -> Array2<f64> {
    assert!(cols <= p, "need ambient p >= cols (p={p}, cols={cols})");
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
    q
}

/// A corpus with KNOWN structure: `k_curved` planted curved circles (Θ = 2π) in
/// mutually-orthogonal planes, plus a `linear_bulk` of orthogonal linear
/// directions carrying lower-rank unstructured variance. The split index `rows`
/// is a disjoint row partition (train vs test) drawn from the SAME generative
/// process — independent gates / angles / noise per row — so EV on the held-out
/// rows is a genuine generalization measurement.
struct Corpus {
    k_curved: usize,
    linear_bulk: usize,
    p: usize,
    /// 2·k_curved curved frame columns followed by `linear_bulk` linear columns.
    basis: Array2<f64>,
}

impl Corpus {
    fn new(k_curved: usize, linear_bulk: usize, p: usize) -> Self {
        let cols = 2 * k_curved + linear_bulk;
        let basis = planted_basis(cols, p);
        Self {
            k_curved,
            linear_bulk,
            p,
            basis,
        }
    }

    fn curved_frame(&self, c: usize) -> (Array1<f64>, Array1<f64>) {
        (
            self.basis.column(2 * c).to_owned(),
            self.basis.column(2 * c + 1).to_owned(),
        )
    }

    fn linear_dir(&self, l: usize) -> Array1<f64> {
        self.basis.column(2 * self.k_curved + l).to_owned()
    }

    /// Draw a row block of size `n` with a per-block `salt` so train and test
    /// blocks are independent. Returns `(z, theta)` where `theta[c][i]` is the
    /// planted angle of curved family `c` at row `i` (used to seed coordinates).
    fn draw(&self, n: usize, salt: u64) -> (Array2<f64>, Vec<Vec<f64>>) {
        let mut z = Array2::<f64>::zeros((n, self.p));
        let mut theta = vec![vec![0.0_f64; n]; self.k_curved];
        let mut signal_sq = 0.0_f64;
        let n_active = ((PLANTED_ACTIVE_MASS * n as f64).round() as usize).max(8);

        // Curved families: sparse gated full-turn circles.
        for c in 0..self.k_curved {
            let (u0, u1) = self.curved_frame(c);
            let radius = 1.0 + 0.1 * (c as f64 / self.k_curved.max(1) as f64);
            let stride = 0.045 + 0.0007 * (c as f64);
            let phase = idx_uniform(salt.wrapping_add(c as u64) * 7 + 11);
            let mut active = vec![false; n];
            let base = (c * n) / self.k_curved.max(1);
            for t in 0..n_active {
                let jit = (idx_uniform(salt.wrapping_add((c as u64) * 131 + (t as u64) * 17 + 3))
                    * 5.0) as usize;
                active[(base + t * 3 + jit) % n] = true;
            }
            for i in 0..n {
                let ang = ((i as f64) * stride + phase).rem_euclid(1.0);
                theta[c][i] = ang;
                if !active[i] {
                    continue;
                }
                let amp = 0.85
                    + 0.30 * idx_uniform(salt.wrapping_add((c as u64) * 977 + (i as u64) * 2 + 1));
                let scale = amp * radius;
                let phi = std::f64::consts::TAU * ang;
                let (cph, sph) = (phi.cos(), phi.sin());
                for col in 0..self.p {
                    let contrib = scale * (cph * u0[col] + sph * u1[col]);
                    z[[i, col]] += contrib;
                    signal_sq += contrib * contrib;
                }
            }
        }

        // Linear bulk: each direction is a sparse-gated rank-1 ray (a feature a
        // single linear atom captures exactly). This is the unstructured-linear
        // variance the discriminating frontier must distinguish from curvature.
        for l in 0..self.linear_bulk {
            let dir = self.linear_dir(l);
            let scale_amp = 0.6; // a touch below the curved scale so curves dominate
            for i in 0..n {
                let gate = idx_uniform(salt.wrapping_add((l as u64) * 5003 + (i as u64) * 9 + 7));
                if gate > PLANTED_ACTIVE_MASS {
                    continue;
                }
                let coeff = scale_amp
                    * (2.0 * idx_uniform(salt.wrapping_add((l as u64) * 41 + (i as u64) * 3 + 2))
                        - 1.0);
                for col in 0..self.p {
                    let contrib = coeff * dir[col];
                    z[[i, col]] += contrib;
                    signal_sq += contrib * contrib;
                }
            }
        }

        // Additive noise at ~4% of the signal scale (the recovery-pin DGP level).
        let signal_scale = (signal_sq / (n * self.p) as f64).sqrt().max(1.0e-12);
        let sigma = 0.04 * signal_scale;
        for i in 0..n {
            for col in 0..self.p {
                let u = idx_uniform(salt.wrapping_add(((i * self.p + col) as u64) * 7 + 3));
                let u2 = idx_uniform(salt.wrapping_add(((i * self.p + col) as u64) * 7 + 5));
                let g = (-2.0 * (u.max(1.0e-12)).ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                z[[i, col]] += sigma * g;
            }
        }
        (z, theta)
    }
}

/// One dictionary slot: which curved family it seeds onto (if curved), and its
/// basis kind. Linear slots seed onto a curved family's angle as well (the linear
/// atom then fits the best secant through the arc it is routed onto), and the
/// extra linear slots beyond the curved families seed onto the linear bulk.
#[derive(Clone, Copy)]
struct Slot {
    /// Seed coordinate source: `Some(c)` = curved family c's angle; `None` =
    /// a linear-bulk direction projection (the slot has no angular structure).
    seed_family: Option<usize>,
    /// For `seed_family == None`, which linear-bulk direction to project onto.
    bulk_dir: usize,
    kind: Kind,
}

/// VERBATIM port of pyffi `sae_residual_seed_logits` (ibp_map cold seed).
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
        let mut mean_energy = 0.0_f64;
        let mut energy = vec![0.0_f64; n_obs];
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

/// Seed each slot's latent coordinate from the corpus, returning the per-slot
/// `(coords, phi, jet)` and the padded `(K, N, m_max)` basis-value stack.
fn seed_slot_geometry(
    slots: &[Slot],
    corpus: &Corpus,
    theta: &[Vec<f64>],
    z: ArrayView2<'_, f64>,
    n: usize,
) -> (
    Vec<Array2<f64>>,
    Vec<Array2<f64>>,
    Vec<Array3<f64>>,
    Array3<f64>,
) {
    let k_atoms = slots.len();
    let basis_sizes: Vec<usize> = slots.iter().map(|s| s.kind.basis_size()).collect();
    let m_max = basis_sizes.iter().copied().max().unwrap();
    let mut coords_k = Vec::with_capacity(k_atoms);
    let mut phi_k = Vec::with_capacity(k_atoms);
    let mut jet_k = Vec::with_capacity(k_atoms);

    for (si, slot) in slots.iter().enumerate() {
        let coords = match slot.seed_family {
            Some(c) => {
                // Curved-family seed: the planted angle, slightly offset so
                // coordinate recovery is not what is under test.
                let offset = 0.04 + 0.013 * ((si % 5) as f64);
                Array2::from_shape_fn((n, 1), |(i, _)| (theta[c][i] + offset).rem_euclid(1.0))
            }
            None => {
                // Linear-bulk seed: project each row onto the bulk direction and
                // standardize into a [0,1)-ish euclidean coordinate.
                let dir = corpus.linear_dir(slot.bulk_dir % corpus.linear_bulk.max(1));
                let mut proj = Array1::<f64>::zeros(n);
                for i in 0..n {
                    proj[i] = z.row(i).dot(&dir);
                }
                let mean = proj.sum() / n as f64;
                let var = proj.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
                let sd = var.sqrt().max(1.0e-9);
                Array2::from_shape_fn((n, 1), |(i, _)| (proj[i] - mean) / sd)
            }
        };
        let (phi, jet) = slot.kind.evaluator().evaluate(coords.view()).unwrap();
        coords_k.push(coords);
        phi_k.push(phi);
        jet_k.push(jet);
    }

    let mut basis_values = Array3::<f64>::zeros((k_atoms, n, m_max));
    for (ai, slot) in slots.iter().enumerate() {
        for row in 0..n {
            for c in 0..slot.kind.basis_size() {
                basis_values[[ai, row, c]] = phi_k[ai][[row, c]];
            }
        }
    }
    (coords_k, phi_k, jet_k, basis_values)
}

/// Build the cold term for `slots` on a corpus row block, through the production
/// cold IBP-MAP routing seed + weighted-LSQ decoder init.
fn build_cold_term(
    slots: &[Slot],
    corpus: &Corpus,
    theta: &[Vec<f64>],
    z: ArrayView2<'_, f64>,
    n: usize,
) -> SaeManifoldTerm {
    let k_atoms = slots.len();
    let basis_sizes: Vec<usize> = slots.iter().map(|s| s.kind.basis_size()).collect();
    let (coords_k, phi_k, jet_k, basis_values) = seed_slot_geometry(slots, corpus, theta, z, n);

    let logits = residual_seed_logits(basis_values.view(), &basis_sizes, z, RESIDUAL_SEED_GAIN);
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

/// Drive the cold term through the production outer engine; return the fitted
/// term and its fitted rho.
fn run_production_fit(
    slots: &[Slot],
    corpus: &Corpus,
    theta: &[Vec<f64>],
    z: &Array2<f64>,
    n: usize,
    label: &str,
) -> (SaeManifoldTerm, SaeManifoldRho) {
    let k_atoms = slots.len();
    let term = build_cold_term(slots, corpus, theta, z.view(), n);
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
    problem
        .run(&mut objective, label)
        .expect("outer cascade must complete");
    let (fitted_term, rho, _loss) = objective.into_fitted();
    (fitted_term, rho)
}

/// Held-out reconstruction EV: re-seat the FITTED atoms (decoder coefficients
/// frozen) onto the TEST-row latent coordinates and reconstruct, measuring EV on
/// the test rows. The decoder is never re-fit on test rows — only the per-row
/// assignment masses are seeded (the same cold residual-energy IBP routing the
/// production predict path uses), so this is a genuine generalization measurement
/// of the trained DECODER.
///
/// HONESTY CAVEAT (stated, not hidden): the curved-atom test latent coordinates
/// are seeded from the planted `theta_test` (plus the same fixed per-slot offset
/// the train arm uses), not encoded from `z_test` alone — coordinate *recovery*
/// is deliberately not what this frontier tests, and seeding both arms from the
/// same planted angles keeps the hybrid-vs-linear contrast apples-to-apples. What
/// IS held out is the decoder `B_k`: it is frozen from the TRAIN fit and never
/// sees a test row, so EV(K) measures how well the trained decoder curve
/// generalizes to fresh draws at known coordinates. The linear-bulk coordinate is
/// genuinely encoded from `z_test` (projection onto the bulk direction), so the
/// bulk arm is a full encode+decode generalization; only the curved coordinate is
/// oracle-seeded.
fn held_out_ev(
    fitted: &SaeManifoldTerm,
    slots: &[Slot],
    corpus: &Corpus,
    theta_test: &[Vec<f64>],
    z_test: &Array2<f64>,
    n_test: usize,
) -> f64 {
    let k_atoms = slots.len();
    let basis_sizes: Vec<usize> = slots.iter().map(|s| s.kind.basis_size()).collect();
    let m_max = basis_sizes.iter().copied().max().unwrap();

    // Test-row geometry (coords + phi) for each slot.
    let (coords_k, phi_k, jet_k, _bv) =
        seed_slot_geometry(slots, corpus, theta_test, z_test.view(), n_test);

    // Build held-out atoms: test-row phi/jet, but the FROZEN trained decoder
    // coefficients B_k. The decoded curve g_k(t) = Φ_test(t) B_k^trained is the
    // trained atom evaluated at the test coordinates.
    let mut atoms = Vec::with_capacity(k_atoms);
    let mut basis_values = Array3::<f64>::zeros((k_atoms, n_test, m_max));
    for (ai, slot) in slots.iter().enumerate() {
        let m_k = slot.kind.basis_size();
        let b = fitted.atoms[ai].decoder_coefficients.clone();
        for row in 0..n_test {
            for c in 0..m_k {
                basis_values[[ai, row, c]] = phi_k[ai][[row, c]];
            }
        }
        let atom = SaeManifoldAtom::new(
            slot_name(slot, ai),
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

    // Seed the test-row assignment from the residual energy of the FROZEN
    // decoded curves (cold routing only — no decoder re-fit).
    let logits = residual_seed_logits(
        basis_values.view(),
        &basis_sizes,
        z_test.view(),
        RESIDUAL_SEED_GAIN,
    );
    let manifolds: Vec<LatentManifold> = slots.iter().map(|s| s.kind.manifold()).collect();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        coords_k,
        manifolds,
        AssignmentMode::ibp_map(TAU, ALPHA, false),
    )
    .unwrap();
    let test_term = SaeManifoldTerm::new(atoms, assignment).unwrap();
    reconstruction_ev(z_test, &test_term.fitted())
}

fn slot_name(slot: &Slot, ai: usize) -> String {
    format!(
        "{}_{ai}",
        match slot.kind {
            Kind::Circle => "circle",
            Kind::Linear => "linear",
        }
    )
}

/// Reconstruction explained variance `1 − SSR/SST` (per-column centered),
/// measured identically for every K and every arm so the frontier is unbiased.
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

/// The HYBRID dictionary at size K: spend the first `min(K, k_curved)` atoms on
/// curved circle bases (one per curved family, in order), and any remaining atoms
/// on linear directions onto the linear bulk. This is the dictionary whose atom
/// set INCLUDES the linear atom as the Θ = 0 special case — it can always do at
/// least as well as a pure-linear dictionary of the same K, and on a curved DGP
/// it captures each circle with one atom instead of shattering it.
fn hybrid_slots(k: usize, corpus: &Corpus) -> Vec<Slot> {
    let mut slots = Vec::with_capacity(k);
    for i in 0..k {
        if i < corpus.k_curved {
            slots.push(Slot {
                seed_family: Some(i),
                bulk_dir: 0,
                kind: Kind::Circle,
            });
        } else {
            slots.push(Slot {
                seed_family: None,
                bulk_dir: i - corpus.k_curved,
                kind: Kind::Linear,
            });
        }
    }
    slots
}

/// The PURE-LINEAR dictionary at size K (the linear-SAE baseline at matched
/// active budget). The first atoms tile the curved families' angle ranges with
/// secants (each linear atom owns one chord of a circle), and any remaining atoms
/// take the linear bulk. With K atoms a pure-linear dictionary must SHATTER each
/// Θ = 2π circle into chords, so its EV(K) climbs only as fast as the secant
/// approximation refines.
fn linear_slots(k: usize, corpus: &Corpus) -> Vec<Slot> {
    let mut slots = Vec::with_capacity(k);
    // Distribute the K linear atoms round-robin across the curved families first
    // (so each circle gets chords as K grows), then the linear bulk.
    for i in 0..k {
        let family = i % corpus.k_curved.max(1);
        slots.push(Slot {
            seed_family: Some(family),
            bulk_dir: 0,
            kind: Kind::Linear,
        });
    }
    // If there is a linear bulk and spare atoms beyond a chord per family, point
    // the tail atoms at the bulk directions.
    let chords_per_family = k / corpus.k_curved.max(1);
    if chords_per_family >= 2 {
        for l in 0..corpus.linear_bulk.min(k) {
            let idx = k - 1 - l;
            slots[idx] = Slot {
                seed_family: None,
                bulk_dir: l,
                kind: Kind::Linear,
            };
        }
    }
    slots
}

#[test]
fn ev_vs_k_frontier_discriminates_curved_from_linear_and_hybrid_dominates() {
    // KNOWN structure: 4 curved circle families (Θ = 2π each) + a 3-direction
    // linear bulk, embedded in p = 40 ambient dims. 2·4 + 3 = 11 <= p, so the
    // planted directions are mutually orthogonal. The effective structured rank
    // is 4 (the curved families dominate the variance); the frontier must climb
    // through K ≈ 4 and then flatten.
    let k_curved = 4usize;
    let linear_bulk = 3usize;
    let p = 40usize;
    let n_train = 1800usize;
    let n_test = 1200usize;

    let corpus = Corpus::new(k_curved, linear_bulk, p);
    let (z_train, theta_train) = corpus.draw(n_train, 0x1026_0001);
    let (z_test, theta_test) = corpus.draw(n_test, 0x1026_BEEF);

    // The K ladder: powers of two through 2·k_curved so we straddle the
    // structured rank (climb region: K < k_curved; flatten region: K >= k_curved).
    let ks = [1usize, 2, 4, 8];

    let mut hybrid_ev: Vec<(usize, f64)> = Vec::new();
    let mut linear_ev: Vec<(usize, f64)> = Vec::new();

    for &k in &ks {
        let hslots = hybrid_slots(k, &corpus);
        let (hfit, _hrho) = run_production_fit(
            &hslots,
            &corpus,
            &theta_train,
            &z_train,
            n_train,
            &format!("frontier-hybrid-K{k}"),
        );
        let h_ev = held_out_ev(&hfit, &hslots, &corpus, &theta_test, &z_test, n_test);
        hybrid_ev.push((k, h_ev));

        let lslots = linear_slots(k, &corpus);
        let (lfit, _lrho) = run_production_fit(
            &lslots,
            &corpus,
            &theta_train,
            &z_train,
            n_train,
            &format!("frontier-linear-K{k}"),
        );
        let l_ev = held_out_ev(&lfit, &lslots, &corpus, &theta_test, &z_test, n_test);
        linear_ev.push((k, l_ev));
    }

    println!("=== #1026 EV-vs-K frontier (HELD-OUT, production engine) ===");
    println!(
        "DGP: {k_curved} curved circles (Θ=2π) + {linear_bulk} linear-bulk dirs, p={p}, \
         n_train={n_train}, n_test={n_test}, noise≈4%"
    );
    println!("K   hybrid_EV   linear_EV   (hybrid − linear)");
    for i in 0..ks.len() {
        let (k, h) = hybrid_ev[i];
        let (_, l) = linear_ev[i];
        println!("{k:<3} {h:>9.6}   {l:>9.6}   {:>+9.6}", h - l);
    }

    // --- Assertion 1: the frontier CLIMBS while structured atoms are found. ---
    // From K = 1 to K = k_curved the hybrid dictionary captures one more curved
    // family per atom, so EV must rise substantially. The structured families
    // carry the bulk of the variance, so going K=1 -> K=k_curved is the climb
    // region: a large, monotone-ish gain.
    let ev_k1 = hybrid_ev[0].1;
    let ev_kc = hybrid_ev
        .iter()
        .find(|(k, _)| *k == k_curved)
        .map(|(_, v)| *v)
        .expect("k_curved is on the ladder");
    assert!(
        ev_kc >= ev_k1 + 0.25,
        "FRONTIER CLIMB FAIL: hybrid EV at K={k_curved} ({ev_kc:.6}) did not exceed EV at K=1 \
         ({ev_k1:.6}) by 0.25 — the structured curved families must drive a steep early climb"
    );

    // The climb must also be (weakly) monotone across the climb region: each
    // added structured atom cannot REDUCE held-out EV by more than a small noise
    // wobble.
    for w in hybrid_ev.windows(2) {
        let ((k0, e0), (k1, e1)) = (w[0], w[1]);
        if k1 <= k_curved {
            assert!(
                e1 >= e0 - 0.02,
                "FRONTIER MONOTONE FAIL: hybrid EV dropped from K={k0} ({e0:.6}) to K={k1} \
                 ({e1:.6}) inside the climb region — adding a structured atom must not lose EV"
            );
        }
    }

    // --- Assertion 2: the frontier FLATTENS once structure is captured. ---
    // Past K = k_curved the curved families are all spent; extra atoms can only
    // chase the small linear bulk and noise, so the marginal EV per doubling must
    // collapse far below the climb-region gain. We compare the gain over the LAST
    // doubling (k_curved -> 2·k_curved) against the gain over the FIRST doubling
    // toward structure.
    let ev_k2c = hybrid_ev
        .iter()
        .find(|(k, _)| *k == 2 * k_curved)
        .map(|(_, v)| *v)
        .expect("2*k_curved is on the ladder");
    let tail_gain = ev_k2c - ev_kc;
    let climb_gain = ev_kc - ev_k1;
    assert!(
        tail_gain <= 0.5 * climb_gain,
        "FRONTIER FLATTEN FAIL: tail gain ({tail_gain:.6}, K={k_curved}->{}) is not at most \
         half the climb gain ({climb_gain:.6}, K=1->{k_curved}) — the frontier must flatten once \
         the curved families are captured (the discriminating signature of structured variance)",
        2 * k_curved
    );

    // --- Assertion 3: HYBRID DOMINATES pure-linear at every matched budget. ---
    // The hybrid atom set includes the linear atom as the Θ = 0 special case, so
    // it cannot lose to a pure-linear dictionary at matched K. On this curved DGP
    // it must STRICTLY win for K <= k_curved (one curved atom captures a circle a
    // single secant cannot), and at least tie (within a small slack) beyond.
    for i in 0..ks.len() {
        let (k, h) = hybrid_ev[i];
        let (_, l) = linear_ev[i];
        assert!(
            h >= l - 0.02,
            "DOMINANCE FAIL: hybrid EV {h:.6} fell below pure-linear EV {l:.6} at K={k} — \
             the hybrid includes the linear atom as a special case and cannot lose at matched K"
        );
    }
    // And a STRICT win at the matched structured budget K = k_curved: the curved
    // dictionary captures the Θ = 2π families with one atom each, while the
    // pure-linear dictionary can only place k_curved secants and is starved.
    let lin_kc = linear_ev
        .iter()
        .find(|(k, _)| *k == k_curved)
        .map(|(_, v)| *v)
        .expect("k_curved is on the ladder");
    assert!(
        ev_kc >= lin_kc + 0.10,
        "DOMINANCE (strict) FAIL: at the matched structured budget K={k_curved}, hybrid EV \
         {ev_kc:.6} did not beat pure-linear EV {lin_kc:.6} by 0.10 — the Θ/√ε shatter penalty \
         means one secant per circle cannot match one curved atom per circle"
    );

    // --- Assertion 4: the curved structure is genuinely captured (not noise). ---
    // The hybrid frontier must clear a high absolute held-out bar once the curved
    // families are in: a 4% noise DGP caps EV near 1, and one curved atom per
    // family is the sufficient parameterization.
    assert!(
        ev_kc >= 0.70,
        "ABSOLUTE BAR FAIL: hybrid held-out EV at K={k_curved} ({ev_kc:.6}) below 0.70 — the \
         curved dictionary must explain the planted curved signal out of sample"
    );

    // --- Assertion 5: the DISCRIMINATING signature — the pure-linear arm has NOT
    // flattened where the hybrid has. ---
    // This is the actual #1026 hypothesis test (H_flat vs H_curved). Under the
    // Θ/(2√(2ε)) shatter law a pure-linear dictionary must keep adding secants to
    // refine each circle, so its EV(K) is still climbing across K=k_curved->2·k_curved
    // exactly where the hybrid (one curved atom per family) has run out of structure
    // and flattened. Encode that as: the pure-linear arm climbs MORE over the tail
    // doubling than the hybrid does, AND that linear tail climb is non-trivial. A
    // corpus on which BOTH arms flatten together (H_flat) would fail this — which is
    // the discrimination the frontier exists to perform.
    let lin_k2c = linear_ev
        .iter()
        .find(|(k, _)| *k == 2 * k_curved)
        .map(|(_, v)| *v)
        .expect("2*k_curved is on the ladder");
    let lin_k1 = linear_ev[0].1;
    let lin_tail_gain = lin_k2c - lin_kc;
    let lin_climb_gain = lin_kc - lin_k1;
    println!(
        "discrimination: hybrid tail/climb = {:.4} (flattens), linear tail/climb = {:.4} (still shattering)",
        if climb_gain.abs() > 1e-9 {
            tail_gain / climb_gain
        } else {
            f64::NAN
        },
        if lin_climb_gain.abs() > 1e-9 {
            lin_tail_gain / lin_climb_gain
        } else {
            f64::NAN
        }
    );
    // The pure-linear arm is still meaningfully refining over the tail doubling: a
    // single secant per circle leaves the largest secant error, and doubling the
    // chord count is exactly where the shatter law buys the most. We require the
    // linear tail gain to clear a real floor (not noise) AND to exceed the hybrid's
    // collapsed tail by a clear margin (the curvature/shatter contrast itself).
    assert!(
        lin_tail_gain >= 0.03,
        "DISCRIMINATION FAIL: pure-linear tail gain ({lin_tail_gain:.6}, K={k_curved}->{}) is below \
         0.03 — a linear dictionary must KEEP climbing by shattering circles into finer secants; \
         if it has already flattened the corpus is not curvature-discriminating",
        2 * k_curved
    );
    assert!(
        lin_tail_gain >= tail_gain + 0.02,
        "DISCRIMINATION FAIL: pure-linear tail gain ({lin_tail_gain:.6}) did not exceed the hybrid \
         tail gain ({tail_gain:.6}) by 0.02 — the discriminating signature is that the curved/hybrid \
         dictionary flattens while the pure-linear shatter is still refining at the same budget"
    );
}

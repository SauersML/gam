//! Multiresolution residual cascade for scattered 2-3D smooths at huge n
//! (compute-first primitive #3, #1032; siblings: the 1-D scan in
//! [`crate::solver::spline_scan`], the 2-D grid in
//! [`crate::solver::grid_spline_2d`]).
//!
//! Model. In metric-scaled coordinates `z = diag(metric)·x` the smooth is
//!   `f(z) = P(z)'γ + Σ_l Σ_j c_{l,j} · φ((z − ξ_{l,j})/δ_l)`,
//! an unpenalized linear polynomial layer `P = {1, z_1, …, z_d}` at the root
//! plus, per level `l = 0..L`, compactly supported Wendland bumps
//! `φ(r) = (1−r)₊⁴(4r+1)` (positive definite and C² on ℝ³) of support radius
//! `δ_l = OVERLAP·h_l` planted on the NEW centers of a nested net with
//! covering radius `h_l = h₀·2^{−l}`. Coefficients are a-priori independent,
//! `c_{l,j} ~ N(0, τ²·4^{−l(s−d/2)})` — the standard multilevel frame whose
//! diagonal prior norm is equivalent to the Sobolev-`s` (semi)norm on
//! quasi-uniform nested nets (Narcowich–Ward inverse estimates + Le Gia–
//! Wendland multilevel stability; `d/2 < s ≤ (d+3)/2`, the native smoothness
//! of the Wendland-(3,1) bump). The assembled claim is certified in-test
//! against a dense kernel solve on small n (#904 style), not assumed.
//!
//! Nets. Each level's center set is a greedy hash-grid ε-net scanned in data
//! order, seeded with the previous level's net: covering radius ≤ h_l over
//! the data AND separation ≥ h_l — the same quasi-uniformity guarantees
//! farthest-point sampling gives, at O(n) per level (each point checks the
//! 3^d neighboring cells of one hash grid of cell size h_l). Nets are nested
//! (`Ξ_0 ⊂ Ξ_1 ⊂ …`); a center carries a bump only at its birth level.
//!
//! Fit. With `W = diag(w)`, `D = diag(0 on the polynomial layer, d_l =
//! 4^{l(s−d/2)} on level-l bumps)` and `λ = σ²/τ²`, the posterior mode solves
//! `(X'WX + λD)c = X'Wy`. `X` is sparse — a row touches the O(1) bumps per
//! level whose supports cover it, O(qL) nonzeros — and is held in CSR. For
//! moderate column counts (`m ≤ DENSE_GRAM_MAX`) the normal equations are
//! solved by dense Cholesky with the EXACT log-determinant (same route as the
//! grid sibling); beyond that the solve is preconditioned CG with the two-level
//! additive-Schwarz coarse-space preconditioner `P = blockdiag(A_CC,
//! diag(A_FF))`. The multilevel Wendland frame is redundant across scales — a
//! coarse bump and the fine bumps in its support are strongly correlated — so
//! the data-fit Gram `X'WX` couples levels and a pure-diagonal preconditioner
//! leaves a conditioning that GROWS with the number of data-identified levels
//! (hence with n). The coarse space `C` (polynomial layer + the data-dominated
//! coarsest levels, see `coarse_space_cols`) is solved EXACTLY by a small dense
//! Cholesky and the penalty-dominated fine tail `F` — where `A_ll ≈ λ d_l I` is
//! already uniformly conditioned — by its Jacobi diagonal. That deflation is
//! what makes `P^{−1/2}(X'WX+λD)P^{−1/2}` uniformly conditioned, so the CG
//! iteration count is genuinely n-independent (the in-test gate asserts an
//! ADDITIVE bound across a 4× n jump, not a multiplicative one). Every CG solve
//! reports its relative residual `‖b − Ac‖/‖b‖`: a computable backward-error
//! certificate (`c` solves a system perturbed by no more than that fraction)
//! inherited by every linear functional of the solution.
//!
//! REML. λ maximizes the profiled-σ² restricted criterion
//!   `ℓ_R(λ) = −½[ log|X'WX+λD| − log|λD|₊ + (n−d−1)·log σ̂²(λ) ] + const`,
//! `log|λD|₊ = r·logλ + Σ_j log d_j` over the `r` penalized columns and
//! `σ̂² = (y'Wy − c'X'Wy)/(n−d−1)` — the same shape as the siblings, with the
//! penalty-logdet constant kept so criteria are comparable across cascade
//! depths. On the dense route `log|X'WX+λD|` is exact; on the iterative
//! route it is `Σ_j log P_jj + tr log(P^{−1/2}AP^{−1/2})` with the trace
//! estimated by stochastic Lanczos quadrature on FIXED Rademacher probes
//! (deterministic seed, shared across every λ trial — common random numbers
//! make the criterion a smooth deterministic function of λ, so the coarse-
//! grid + golden-section search is exactly as deterministic as the
//! siblings'). The diagonal split is the level-block control variate: the
//! dominant λ-dependence rides the exactly-computed `Σ log P_jj` term and
//! SLQ only sees the well-conditioned remainder.
//!
//! Refinement certificate. After fitting L levels, the candidate level L+1
//! is constructed (O(n)) and the EXACT objective decrease available from
//! adding it is bounded: for the penalized objective `F(c) = ‖√W(y−Xc)‖² +
//! λc'Dc`, appending columns `X₂` with penalty `λd_{L+1}I` decreases the
//! minimum by `g'S⁻¹g`, `g = X₂'W r̂`, `S` the Schur complement; since
//! `A₁₁ ⪰ X₁'WX₁` and `X₂'W^{1/2}·proj·W^{1/2}X₂ ⪯ X₂'WX₂`, `S ⪰ λd_{L+1}I`,
//! so the decrease is at most `‖X₂'W r̂‖²/(λ·d_{L+1})` — a computable
//! discretization certificate. The cascade refines (adds the level, refits,
//! re-selects λ) until that bound drops below `REFINE_TOL` of the penalized
//! residual, the net stops producing new centers (every point is a center),
//! or the level/center caps are reached: certified-or-fallback, the same
//! discipline as the radial-profile GL ladder.
//!
//! Posterior. Coefficient covariance is `σ²(X'WX+λD)^{−1}`; pointwise
//! prediction variance routes the basis row through one (certified) solve.
//! Exact posterior samples come from perturb-and-solve: `c_s = A^{−1}(X'Wy +
//! σ(X'W^{1/2}z₁ + √λ D^{1/2}z₂))` with iid standard-normal `z₁, z₂` has
//! mean `ĉ` and covariance exactly `σ²A^{−1}` (deterministically seeded; one
//! certified solve per sample).
//!
//! Payoff. Build O(n·(L + 3^d)), fit O(nnz · iters) per λ trial with
//! n-independent iters — O(n log n) end to end, against the dense n×k kernel
//! Gram + O(k³) per trial that duchon/matern pay today. Gap behavior is
//! mechanical: levels wider than a gap keep support across it (polynomial +
//! coarse bumps bridge), finer levels have no data and revert to their prior
//! variance, so the posterior mean bridges instead of sagging while the
//! variance grows into the gap.

use std::collections::HashMap;
use std::sync::Arc;

use super::grid_spline_2d::{chol_solve, cholesky_logdet};

/// Bump support radius as a multiple of the level's covering radius:
/// `δ_l = OVERLAP·h_l`. Separation ≥ h_l caps the bumps covering a point at
/// a packing constant per level (O(q) row nonzeros per level).
const OVERLAP: f64 = 2.0;
/// Root covering radius as a fraction of the largest scaled axis range.
const H0_FRACTION: f64 = 0.5;
/// Levels in the initial cascade before refinement certificates run.
const INITIAL_LEVELS: usize = 3;
/// Hard cap on cascade depth (h shrinks 2^16-fold below the root).
const MAX_LEVELS: usize = 16;
/// Hard cap on total centers across all levels.
const MAX_CENTERS: usize = 200_000;
/// Refinement stops when the exact next-level gain bound falls below this
/// fraction of the penalized residual.
const REFINE_TOL: f64 = 1e-3;

/// Column count up to which the normal equations go through dense Cholesky
/// (exact logdet, no iteration); above it, PCG + SLQ. 1536² doubles ≈ 18 MB.
const DENSE_GRAM_MAX: usize = 1536;

/// Deterministic coarse-grid width and bounds for the log-λ search (same
/// scheme as the siblings), then golden-section refinement.
const LOG_LAMBDA_GRID: usize = 25;
const LOG_LAMBDA_LO: f64 = -18.0;
const LOG_LAMBDA_HI: f64 = 18.0;
const LOG_LAMBDA_TOL: f64 = 1e-6;

/// PCG convergence: relative residual ‖b − Ac‖/‖b‖ (the backward-error
/// certificate) demanded of every solve, and the iteration cap past which
/// the solve is an error rather than a silent approximation. The certification
/// suite gates the iterative route at 1e-9; asking for more burns matvecs
/// without strengthening any downstream certificate.
const CG_RTOL: f64 = 1e-9;
const CG_MAX_ITERS: usize = 4000;

/// Coarse-space additive-Schwarz preconditioner controls (issue #1032: the
/// "BPX/level-diagonal preconditioned CG, n-independent iters" spec).
///
/// The multilevel Wendland frame is redundant across scales — a coarse bump and
/// the fine bumps inside its support are strongly correlated — so the data-fit
/// Gram `X'WX` couples levels and a pure-diagonal (Jacobi) preconditioner leaves
/// a conditioning that grows with the number of *data-identified* levels, hence
/// with `n` (more rows ⇒ finer levels carry data ⇒ another collinear coarse
/// scale the diagonal can't decouple). The cure is the textbook two-level
/// additive Schwarz coarse space: solve the coarse block — the polynomial layer
/// plus every level the penalty has NOT yet made diagonally dominant — EXACTLY,
/// and precondition the remaining penalty-dominated fine levels (where
/// `A_ll ≈ λ d_l I` is already uniformly conditioned) by their Jacobi diagonal.
///
/// A level is "data-dominated" while `λ d_l < COARSE_DOMINANCE · median diag
/// (X'WX) over the level`. Because columns are laid out poly, level-0, level-1,
/// … and `d_l` increases while the per-level data weight decreases, the
/// data-dominated levels are exactly the coarsest prefix `[0, ncoarse)`, so the
/// coarse space is a contiguous column prefix and the cut is a single scan. The
/// crossover level grows only as `½ log₄(n/λ)` — `ncoarse = O(√(n/λ))` columns —
/// so the exact coarse factorization stays small against the sparse matvecs at
/// every n the primitive serves. [`COARSE_SPACE_MAX`] caps it as a safety valve
/// (past the cap the finer data-dominated levels fall back to Jacobi and the
/// iteration count rises, but the CG residual certificate still guarantees the
/// solve); [`MIN_COARSE_LEVELS`] always deflates the two coarsest scales, which
/// are near-collinear with the polynomial layer at every λ.
const COARSE_DOMINANCE: f64 = 4.0;
/// Safety ceiling on the exact-coarse column count. It must NOT bind at the n
/// the primitive serves: the n-independent iteration count rests on the coarse
/// block containing the WHOLE data-dominated prefix (`O(√(n/λ))` columns), so a
/// cap that truncates that prefix is exactly what makes the iteration count
/// climb with n (a finer data-dominated level demoted to Jacobi cannot be
/// decoupled from the coarse scales it is collinear with). At the n-scales the
/// iterative route engages (tens of thousands of rows → a ≈1.4k-column
/// prefix) this is non-binding headroom; it only triggers in the genuinely
/// degenerate case the quasi-uniformity guard is meant to catch first. The
/// realized coarse factorization runs at the actual prefix length, not the cap,
/// so the ceiling costs nothing until it fires.
const COARSE_SPACE_MAX: usize = 4096;
const MIN_COARSE_LEVELS: usize = 2;

/// Quasi-uniformity guard (issue #1032, caveat 2). The BPX n-independent CG
/// iteration bound rests on the nested ε-nets being quasi-uniform *in the
/// metric-scaled coordinates `z = diag(metric)·x` the bumps live in*. The
/// greedy net guarantees covering ≤ h and separation ≥ h in `z` by
/// construction, so the only way the BPX norm-equivalence constant blows up is
/// when the metric is so anisotropic that the metric-scaled point cloud is
/// effectively degenerate along a direction — the data collapses onto a lower
/// dimension in `z`, the root covering radius `h₀ = ½·max_a range_a` swamps the
/// collapsed axis, the level-`l` bumps overlap pathologically, and the
/// preconditioner constant (hence the iteration count) grows without an
/// n-independent bound. The realized symptom is `solve_iters` climbing toward
/// [`CG_MAX_ITERS`]; this guard detects the *cause* up front from the
/// metric-scaled per-axis spread so the auto-route can fall back to the dense
/// kernel BEFORE paying an unbounded iterative solve, rather than discovering
/// the blow-up only after `CG_MAX_ITERS` work.
///
/// Condition measure: the ratio of the largest to smallest metric-scaled
/// per-axis standard deviation (a scale-free aspect ratio of the scaled
/// cloud). Past this threshold the net is no longer quasi-uniform in every
/// direction and the BPX bound is not trustworthy. Derived, not a knob: a
/// `10³` aspect ratio means the collapsed axis carries <0.1% of the dominant
/// axis's variation, at which point its bumps span the whole cloud and the
/// multilevel hierarchy degenerates to a single ill-conditioned level.
const QUASI_UNIFORMITY_MAX_ASPECT: f64 = 1.0e3;

/// SLQ controls: fixed Rademacher probes (shared across λ trials) and the
/// Lanczos depth per probe (full reorthogonalization; early exit on
/// breakdown).
const SLQ_PROBES: usize = 24;
const SLQ_LANCZOS_STEPS: usize = 48;

/// Deterministic seed for the SLQ probes and posterior samples.
const RNG_SEED: u64 = 0x1032_CA5C_ADE0_5EED;

/// Floor for eigenvalues/pivots before the system is declared singular.
const EIG_FLOOR: f64 = 1e-300;

// ───────────────────────────── deterministic RNG ────────────────────────────

/// SplitMix64: tiny, deterministic, full-period stream generator.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1): 53-bit mantissa, shifted off zero.
    fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    /// Standard normal via Box–Muller.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Rademacher ±1.
    fn next_sign(&mut self) -> f64 {
        if self.next_u64() & 1 == 0 { 1.0 } else { -1.0 }
    }
}

// ─────────────────────────────── hash grids ─────────────────────────────────

/// Integer cell of a point at a given cell width (coordinates are already
/// metric-scaled and shifted to be ≥ 0, so indices are small and exact).
#[inline]
fn cell_of(z: &[f64; 3], dim: usize, width: f64) -> (i32, i32, i32) {
    let mut c = [0_i32; 3];
    for a in 0..dim {
        c[a] = (z[a] / width).floor() as i32;
    }
    (c[0], c[1], c[2])
}

/// Hash grid over a point set: cell → indices. Lookup scans the 3^d
/// neighborhood, which covers every point within one cell width.
struct HashGrid {
    width: f64,
    dim: usize,
    cells: HashMap<(i32, i32, i32), Vec<u32>>,
}

impl HashGrid {
    fn new(width: f64, dim: usize) -> Self {
        HashGrid {
            width,
            dim,
            cells: HashMap::new(),
        }
    }

    fn insert(&mut self, idx: u32, z: &[f64; 3]) {
        let key = cell_of(z, self.dim, self.width);
        self.cells.entry(key).or_default().push(idx);
    }

    /// Visit every stored index in the 3^d cells around `z` (deterministic
    /// order: lexicographic cells, insertion order within a cell).
    fn for_neighbors(&self, z: &[f64; 3], mut visit: impl FnMut(u32)) {
        let (c0, c1, c2) = cell_of(z, self.dim, self.width);
        let d2 = if self.dim > 2 { 1 } else { 0 };
        let d1 = if self.dim > 1 { 1 } else { 0 };
        for i0 in -1..=1_i32 {
            for i1 in -d1..=d1 {
                for i2 in -d2..=d2 {
                    if let Some(bucket) = self.cells.get(&(c0 + i0, c1 + i1, c2 + i2)) {
                        for &idx in bucket {
                            visit(idx);
                        }
                    }
                }
            }
        }
    }
}

#[inline]
fn dist2(a: &[f64; 3], b: &[f64; 3], dim: usize) -> f64 {
    let mut s = 0.0;
    for k in 0..dim {
        let d = a[k] - b[k];
        s += d * d;
    }
    s
}

/// Wendland-(3,1) bump `(1−r)₊⁴(4r+1)`: positive definite on ℝ^d, d ≤ 3,
/// C², native space H^{(d+3)/2}.
#[inline]
fn wendland(r: f64) -> f64 {
    if r >= 1.0 {
        return 0.0;
    }
    let v = 1.0 - r;
    let v2 = v * v;
    v2 * v2 * (4.0 * r + 1.0)
}

// ───────────────────────────── design assembly ──────────────────────────────

/// One resolution level: its NEW centers (scaled coordinates), covering
/// radius, support radius, prior precision weight, and a lookup grid of cell
/// width δ_l over those centers.
struct Level {
    h: f64,
    delta: f64,
    /// Prior precision weight `d_l = 4^{l(s−d/2)}` (prior variance τ²/d_l).
    weight: f64,
    centers: Vec<[f64; 3]>,
    /// First flat column index of this level's coefficients.
    col_offset: usize,
    grid: HashGrid,
}

/// Immutable fitted-design core shared between the design handle and fits.
struct Core {
    dim: usize,
    metric: [f64; 3],
    /// Lower corner / range of the scaled bounding box (polynomial layer
    /// coordinates are `2(z − lo)/range − 1` for conditioning).
    z_lo: [f64; 3],
    z_range: [f64; 3],
    sobolev_s: f64,
    levels: Vec<Level>,
    /// Full nested net Ξ_L (scaled coords), retained so the candidate level
    /// L+1 can extend it without re-deriving coarser levels.
    net: Vec<[f64; 3]>,
    /// Total columns: `dim + 1` polynomial + all level centers.
    m: usize,
    /// CSR design rows (column-sorted within a row).
    row_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    vals: Vec<f64>,
    /// Inputs retained for matvecs, residuals, and refinement.
    w: Vec<f64>,
    y: Vec<f64>,
    /// Scaled data coordinates (shifted to the box corner).
    z: Vec<[f64; 3]>,
    /// `X'Wy`, `y'Wy`, `diag(X'WX)`.
    rhs: Vec<f64>,
    ytwy: f64,
    gram_diag: Vec<f64>,
    /// Per-column prior precision weight (0 on the polynomial layer).
    pen_diag: Vec<f64>,
    /// `Σ_j log d_j` over penalized columns (the λ-free part of log|λD|₊,
    /// kept so REML criteria compare across cascade depths).
    pen_logdet_const: f64,
    /// Dense upper-triangular `X'WX` when `m ≤ DENSE_GRAM_MAX` (row-major
    /// m×m, lower mirror filled at solve time); None on the iterative route.
    dense_gram: Option<Vec<f64>>,
    /// Predict-only factored precision: the lower Cholesky factor `L` of
    /// `A = X'WX + λD` at the FIT's λ, populated only on a core rebuilt from a
    /// persisted [`ResidualCascadeState`] (where the training CSR is dropped).
    /// When present, `solve_coeff` replays the posterior-variance solve through
    /// this factor instead of the absent training design; `None` on a
    /// training-built core, which solves through `dense_gram`/PCG as usual.
    predict_chol: Option<Vec<f64>>,
}

/// Solver route a fit took for its log-determinant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogdetMethod {
    /// Dense Cholesky: exact.
    DenseExact,
    /// Diagonal control variate + stochastic Lanczos quadrature on fixed
    /// deterministic probes.
    Slq,
}

/// Computable certificates attached to a fit.
#[derive(Clone, Copy, Debug)]
pub struct CascadeCertificate {
    /// Backward error of the coefficient solve: ‖b − Aĉ‖/‖b‖ (0 on the dense
    /// route).
    pub solve_rel_residual: f64,
    /// CG iterations of the coefficient solve (0 on the dense route); the
    /// n-independence gate watches this.
    pub solve_iters: usize,
    /// Route the log-determinant took.
    pub logdet_method: LogdetMethod,
}

/// Discretization certificate of the refinement loop: the exact upper bound
/// on the penalized-objective decrease available from one more level.
#[derive(Clone, Copy, Debug)]
pub struct RefinementCertificate {
    /// `‖X_{L+1}'W r̂‖² / (λ·d_{L+1})` at the accepted fit.
    pub next_level_gain_bound: f64,
    /// The absolute tolerance it was compared against (`REFINE_TOL·rss_pen`).
    pub tolerance: f64,
    /// True when refinement stopped because the net produced no new centers
    /// or a cap was reached rather than because the bound passed.
    pub exhausted: bool,
}

/// Multiresolution residual-cascade design: nested nets, sparse design,
/// diagonal multilevel prior — everything needed to evaluate the REML
/// criterion and solve at any λ.
pub struct ResidualCascadeDesign {
    core: Arc<Core>,
}

/// Fitted cascade with factored-by-solve posterior access.
pub struct ResidualCascadeFit {
    core: Arc<Core>,
    /// Dense-route prediction factor at the fit's λ. When present, pointwise
    /// variance uses this one Cholesky factor instead of refactoring the same
    /// precision matrix for every prediction point.
    predict_chol: Option<Vec<f64>>,
    /// Coefficients: `dim+1` polynomial entries, then level blocks.
    pub coeff: Vec<f64>,
    /// Selected (or supplied) log smoothing parameter `log λ = log σ²/τ²`.
    pub log_lambda: f64,
    /// Profiled (or supplied) observation variance σ².
    pub sigma2: f64,
    /// Restricted log-likelihood at the fit, up to λ- and data-independent
    /// additive constants (exact REML differences across λ on the dense
    /// route; SLQ-estimated on the iterative route).
    pub restricted_loglik: f64,
    /// Penalized residual quadratic `y'Wy − c'X'Wy`.
    pub rss_pen: f64,
    /// Solve/logdet certificates.
    pub certificate: CascadeCertificate,
    /// Present when the fit came from the refinement loop.
    pub refinement: Option<RefinementCertificate>,
}

/// One resolution level's geometry in a persisted snapshot: the data needed to
/// rebuild a [`Level`] (its lookup grid, bumps, and column block) without the
/// training rows. Centers are flattened `dim`-major (`dim` floats per center).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LevelState {
    pub h: f64,
    pub delta: f64,
    pub weight: f64,
    pub col_offset: u64,
    /// `dim·n_centers` scaled-coordinate floats, center-major.
    pub centers: Vec<f64>,
}

/// Serializable snapshot of a [`ResidualCascadeFit`] (#1032 persistence
/// prerequisite). Holds everything `predict` needs and NOTHING about the
/// training rows:
/// - MEAN: the nested geometry (`dim`/`metric`/box/`sobolev_s` + per-level
///   centers/δ/weights/col-offsets) and the root polynomial layer are all that
///   `basis_row_scaled`·`coeff` reads;
/// - VARIANCE: the factored precision `predict_chol` — the lower Cholesky factor
///   `L` of `A = X'WX + λD` at the fit's λ — which the posterior-variance solve
///   `x'A⁻¹x` replays against (the training design that originally assembled `A`
///   is dropped).
///
/// `from_state` rebuilds a predict-capable fit whose `Core` carries empty
/// training CSR and `predict_chol = Some(L)`; `solve_coeff` then routes the
/// variance solve through `L`. The reconstructed fit cannot be re-fit or
/// resampled (it has no rows), only predicted from — exactly the persistence
/// contract.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResidualCascadeState {
    pub dim: u64,
    /// Per-axis metric scaling (length 3; trailing entries are 1 for `dim < 3`).
    pub metric: [f64; 3],
    pub z_lo: [f64; 3],
    pub z_range: [f64; 3],
    pub sobolev_s: f64,
    pub levels: Vec<LevelState>,
    /// Total column count `dim + 1 + Σ centers`.
    pub m: u64,
    /// `Σ_j log d_j` over penalized columns (kept so restored REML scalars stay
    /// comparable across cascade depths).
    pub pen_logdet_const: f64,
    /// Posterior-mode coefficients (length `m`).
    pub coeff: Vec<f64>,
    pub log_lambda: f64,
    pub sigma2: f64,
    pub restricted_loglik: f64,
    pub rss_pen: f64,
    /// Lower Cholesky factor `L` of `A = X'WX + λD` at the fit's λ, `m × m`
    /// row-major — the factored precision the variance solve replays through.
    pub predict_chol: Vec<f64>,
}

/// Forward substitution `L y = b` (lower factor, row-major) into `out`.
fn forward_sub_into(l: &[f64], p: usize, b: &[f64], out: &mut [f64]) {
    for i in 0..p {
        let mut s = b[i];
        for t in 0..i {
            s -= l[i * p + t] * out[t];
        }
        out[i] = s / l[i * p + i];
    }
}

/// Back substitution `Lᵀ z = y` (lower factor, row-major) into `out`.
fn back_sub_into(l: &[f64], p: usize, y: &[f64], out: &mut [f64]) {
    for i in (0..p).rev() {
        let mut s = y[i];
        for t in i + 1..p {
            s -= l[t * p + i] * out[t];
        }
        out[i] = s / l[i * p + i];
    }
}

/// Coarse-space additive-Schwarz preconditioner for the iterative route
/// (issue #1032). `A = X'WX + λD` is preconditioned by the symmetric positive
/// definite block-diagonal `P = blockdiag(A_CC, diag(A_FF))`, where the coarse
/// index set `C = [0, ncoarse)` is the polynomial layer plus the data-dominated
/// (coarsest) levels and `F` the penalty-dominated fine tail — see the
/// [`COARSE_DOMINANCE`]/[`COARSE_SPACE_MAX`] docs for why this delivers
/// n-independent CG iteration counts where the pure-Jacobi diagonal does not.
///
/// `solve` applies `P⁻¹` (exact coarse Cholesky solve ⊕ fine Jacobi). For the
/// SLQ log-determinant the symmetric factor `R = blockdiag(L_CC, diag√A_FF)`
/// with `P = R Rᵀ` is exposed through `apply_r_inv`/`apply_r_inv_t`, and
/// `log|P| = log|A_CC| + Σ_F log A_jj`.
struct Preconditioner {
    /// First fine column; coarse block is the principal `[0, ncoarse)` submatrix.
    ncoarse: usize,
    /// Lower Cholesky factor of the coarse block `A_CC` (`ncoarse × ncoarse`).
    coarse_chol: Vec<f64>,
    /// `log|A_CC|` (exact).
    coarse_logdet: f64,
    /// `1/A_jj` on the fine columns `[ncoarse, m)`.
    inv_fine: Vec<f64>,
    /// `1/√A_jj` on the fine columns (the `R⁻¹`/`R⁻ᵀ` fine scaling).
    inv_sqrt_fine: Vec<f64>,
    /// `Σ_F log A_jj` (the fine part of `log|P|`).
    fine_logdet: f64,
}

impl Preconditioner {
    /// `out = P⁻¹ r`: exact coarse solve on `[0, ncoarse)`, Jacobi on the tail.
    fn solve(&self, r: &[f64], out: &mut [f64]) {
        let nc = self.ncoarse;
        let zc = chol_solve(&self.coarse_chol, nc, &r[..nc]);
        out[..nc].copy_from_slice(&zc);
        for (k, o) in out[nc..].iter_mut().enumerate() {
            *o = r[nc + k] * self.inv_fine[k];
        }
    }

    /// `out = R⁻ᵀ v` (coarse: `L_CCᵀ` back-solve; fine: `/√A_jj`).
    fn apply_r_inv_t(&self, v: &[f64], out: &mut [f64]) {
        let nc = self.ncoarse;
        back_sub_into(&self.coarse_chol, nc, &v[..nc], &mut out[..nc]);
        for (k, o) in out[nc..].iter_mut().enumerate() {
            *o = v[nc + k] * self.inv_sqrt_fine[k];
        }
    }

    /// `out = R⁻¹ v` (coarse: `L_CC` forward-solve; fine: `/√A_jj`).
    fn apply_r_inv(&self, v: &[f64], out: &mut [f64]) {
        let nc = self.ncoarse;
        forward_sub_into(&self.coarse_chol, nc, &v[..nc], &mut out[..nc]);
        for (k, o) in out[nc..].iter_mut().enumerate() {
            *o = v[nc + k] * self.inv_sqrt_fine[k];
        }
    }

    /// `log|P| = log|A_CC| + Σ_F log A_jj`.
    fn logdet(&self) -> f64 {
        self.coarse_logdet + self.fine_logdet
    }
}

impl Core {
    /// Scale a raw point into shifted metric coordinates.
    fn scale_point(&self, x: &[f64]) -> [f64; 3] {
        let mut z = [0.0_f64; 3];
        for a in 0..self.dim {
            z[a] = self.metric[a] * x[a] - self.z_lo[a];
        }
        z
    }

    /// Sparse basis row at a scaled point: polynomial layer then every bump
    /// whose support covers it, as (column, value) pairs sorted by column.
    fn basis_row_scaled(&self, z: &[f64; 3]) -> Vec<(usize, f64)> {
        let mut row = Vec::with_capacity(self.dim + 1 + self.levels.len() * 8);
        row.push((0, 1.0));
        for a in 0..self.dim {
            row.push((a + 1, 2.0 * z[a] / self.z_range[a] - 1.0));
        }
        for level in &self.levels {
            let start = row.len();
            level.grid.for_neighbors(z, |j| {
                let c = &level.centers[j as usize];
                let r = dist2(z, c, self.dim).sqrt() / level.delta;
                let v = wendland(r);
                if v > 0.0 {
                    row.push((level.col_offset + j as usize, v));
                }
            });
            row[start..].sort_unstable_by_key(|&(col, _)| col);
        }
        row
    }

    /// `out = (X'WX + λD)·v` through the CSR rows: O(nnz).
    fn matvec(&self, lambda: f64, v: &[f64], out: &mut [f64]) {
        for (o, (&d, &x)) in out.iter_mut().zip(self.pen_diag.iter().zip(v.iter())) {
            *o = lambda * d * x;
        }
        for i in 0..self.w.len() {
            let lo = self.row_ptr[i];
            let hi = self.row_ptr[i + 1];
            let mut t = 0.0;
            for e in lo..hi {
                t += self.vals[e] * v[self.col_idx[e] as usize];
            }
            t *= self.w[i];
            for e in lo..hi {
                out[self.col_idx[e] as usize] += self.vals[e] * t;
            }
        }
    }

    /// Jacobi / level-diagonal preconditioner: `diag(X'WX) + λ·diag(λD)`.
    /// Levels share a constant prior weight, so this IS the level-block
    /// (BPX-flavored) diagonal in the multilevel frame.
    /// Coarse column count of the additive-Schwarz coarse space at `λ`: the
    /// polynomial layer plus the longest prefix of data-dominated levels
    /// (`λ d_l < COARSE_DOMINANCE · median diag(X'WX) over the level`), with the
    /// two coarsest levels always deflated and the total capped at
    /// [`COARSE_SPACE_MAX`]. Because `d_l` rises while the per-level data weight
    /// falls, the data-dominated set is a contiguous prefix, so one scan from the
    /// coarsest level finds the cut. (See [`COARSE_DOMINANCE`].)
    fn coarse_space_cols(&self, lambda: f64) -> usize {
        let mut ncoarse = self.nullity();
        let mut buf: Vec<f64> = Vec::new();
        for (li, level) in self.levels.iter().enumerate() {
            let a = level.col_offset;
            let b = a + level.centers.len();
            if b <= a {
                continue;
            }
            if b > COARSE_SPACE_MAX {
                break;
            }
            let dominated = if li < MIN_COARSE_LEVELS {
                true
            } else {
                buf.clear();
                buf.extend_from_slice(&self.gram_diag[a..b]);
                buf.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
                let gram_median = buf[buf.len() / 2];
                lambda * level.weight < COARSE_DOMINANCE * gram_median
            };
            if dominated {
                ncoarse = b;
            } else {
                break;
            }
        }
        // Keep at least one fine column so the split is well-defined; if every
        // level is coarse the iterative route is degenerate anyway and the dense
        // route would have been taken, but guard regardless.
        let ncoarse = ncoarse.min(self.m);
        // Debug-only coarse-space layout trace (#1032). Gated on the log level so
        // the per-call string build stays out of this preconditioner hot path,
        // and routed through `log` (an `eprintln!` here trips the src banned-macro
        // gate and broke the build).
        if log::log_enabled!(log::Level::Debug) {
            let mut s = String::new();
            for (li, level) in self.levels.iter().enumerate() {
                let a = level.col_offset;
                let b = a + level.centers.len();
                let mut buf: Vec<f64> = self.gram_diag[a..b].to_vec();
                buf.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
                let med = if buf.is_empty() {
                    0.0
                } else {
                    buf[buf.len() / 2]
                };
                let coarse = b <= ncoarse;
                s.push_str(&format!(
                    " L{li}[{}c off{a} w={:.2e} λw={:.2e} med={:.2e} {}]",
                    level.centers.len(),
                    level.weight,
                    lambda * level.weight,
                    med,
                    if coarse { "C" } else { "F" }
                ));
            }
            log::debug!(
                "[1032-COARSE] λ={lambda:.3e} m={} ncoarse={ncoarse} cap={COARSE_SPACE_MAX}{s}",
                self.m
            );
        }
        ncoarse
    }

    /// Build the coarse-space additive-Schwarz preconditioner at `λ`: assemble
    /// and factor the coarse block `A_CC` from the CSR (coarse columns are the
    /// prefix `[0, ncoarse)`, and each CSR row is column-sorted, so a row's
    /// coarse entries are its leading run), then the Jacobi diagonal on the fine
    /// tail. `O(n · q_C²) + O(ncoarse³)` — paid once per `λ`, not per CG step.
    fn build_preconditioner(&self, lambda: f64) -> Result<Preconditioner, String> {
        let m = self.m;
        let nc = self.coarse_space_cols(lambda);
        let mut acc = vec![0.0_f64; nc * nc];
        for i in 0..self.w.len() {
            let lo = self.row_ptr[i];
            let hi = self.row_ptr[i + 1];
            // Leading run of coarse columns (CSR rows are column-sorted).
            let mut end = lo;
            while end < hi && (self.col_idx[end] as usize) < nc {
                end += 1;
            }
            for ea in lo..end {
                let ca = self.col_idx[ea] as usize;
                let va = self.w[i] * self.vals[ea];
                for eb in ea..end {
                    let cb = self.col_idx[eb] as usize;
                    acc[ca * nc + cb] += va * self.vals[eb];
                }
            }
        }
        for i in 0..nc {
            for j in i + 1..nc {
                acc[j * nc + i] = acc[i * nc + j];
            }
        }
        for i in 0..nc {
            acc[i * nc + i] += lambda * self.pen_diag[i];
        }
        let coarse_logdet = cholesky_logdet(&mut acc, nc)?;
        let mut inv_fine = Vec::with_capacity(m - nc);
        let mut inv_sqrt_fine = Vec::with_capacity(m - nc);
        let mut fine_logdet = 0.0;
        for j in nc..m {
            let p = self.gram_diag[j] + lambda * self.pen_diag[j];
            if !(p.is_finite() && p > EIG_FLOOR) {
                return Err(format!(
                    "residual cascade: non-positive preconditioner diagonal {p} at column {j}"
                ));
            }
            inv_fine.push(1.0 / p);
            inv_sqrt_fine.push(1.0 / p.sqrt());
            fine_logdet += p.ln();
        }
        Ok(Preconditioner {
            ncoarse: nc,
            coarse_chol: acc,
            coarse_logdet,
            inv_fine,
            inv_sqrt_fine,
            fine_logdet,
        })
    }

    /// Preconditioned CG on `(X'WX + λD)c = b` to relative residual CG_RTOL.
    /// Returns the solution with its backward-error certificate.
    fn pcg(
        &self,
        lambda: f64,
        b: &[f64],
        warm: Option<&[f64]>,
    ) -> Result<(Vec<f64>, f64, usize), String> {
        let m = self.m;
        let prec = self.build_preconditioner(lambda)?;
        let b_norm = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        if b_norm == 0.0 {
            return Ok((vec![0.0; m], 0.0, 0));
        }
        let mut zv = vec![0.0; m];
        let mut x = match warm {
            Some(x0) => {
                if x0.len() != m {
                    return Err(format!(
                        "residual cascade: warm-start length {} != system size {m}",
                        x0.len()
                    ));
                }
                x0.to_vec()
            }
            None => {
                prec.solve(b, &mut zv);
                zv.clone()
            }
        };
        let mut r = vec![0.0; m];
        self.matvec(lambda, &x, &mut r);
        for (ri, &bi) in r.iter_mut().zip(b.iter()) {
            *ri = bi - *ri;
        }
        prec.solve(&r, &mut zv);
        let mut p_dir = zv.clone();
        let mut rz: f64 = r.iter().zip(zv.iter()).map(|(&a, &c)| a * c).sum();
        let mut ap = vec![0.0; m];
        for iter in 0..CG_MAX_ITERS {
            let r_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r_norm <= CG_RTOL * b_norm {
                return Ok((x, r_norm / b_norm, iter));
            }
            self.matvec(lambda, &p_dir, &mut ap);
            let pap: f64 = p_dir.iter().zip(ap.iter()).map(|(&a, &c)| a * c).sum();
            if !(pap.is_finite() && pap > 0.0) {
                return Err(format!(
                    "residual cascade: CG curvature breakdown (p'Ap = {pap}) at iteration {iter}"
                ));
            }
            let alpha = rz / pap;
            for j in 0..m {
                x[j] += alpha * p_dir[j];
                r[j] -= alpha * ap[j];
            }
            prec.solve(&r, &mut zv);
            let rz_new: f64 = r.iter().zip(zv.iter()).map(|(&a, &c)| a * c).sum();
            let beta = rz_new / rz;
            rz = rz_new;
            for j in 0..m {
                p_dir[j] = zv[j] + beta * p_dir[j];
            }
        }
        Err(format!(
            "residual cascade: CG failed to reach relative residual {CG_RTOL} within \
             {CG_MAX_ITERS} iterations (the coarse-space additive-Schwarz preconditioner should \
             make this n-independent; this indicates a degenerate design)"
        ))
    }

    /// Expand the cached dense upper Gram + λD into a full symmetric matrix.
    fn dense_system(&self, lambda: f64) -> Option<Vec<f64>> {
        let gram = self.dense_gram.as_ref()?;
        let m = self.m;
        let mut a = vec![0.0; m * m];
        for i in 0..m {
            for j in i..m {
                let mut v = gram[i * m + j];
                if i == j {
                    v += lambda * self.pen_diag[i];
                }
                a[i * m + j] = v;
                a[j * m + i] = v;
            }
        }
        Some(a)
    }

    /// Exact log-determinant of `X'WX + λD` by dense Cholesky. Errors when
    /// the design is past the dense sizing cap.
    fn logdet_dense(&self, lambda: f64) -> Result<f64, String> {
        let mut a = self.dense_system(lambda).ok_or_else(|| {
            format!(
                "residual cascade: dense logdet requested past the sizing cap \
                 (m = {} > {DENSE_GRAM_MAX})",
                self.m
            )
        })?;
        cholesky_logdet(&mut a, self.m)
    }

    /// SLQ log-determinant: exact control variate `log|P|` (the coarse-space
    /// additive-Schwarz preconditioner's own log-determinant — `log|A_CC|` plus
    /// the fine Jacobi `Σ_F log A_jj`) plus stochastic Lanczos quadrature for
    /// `tr log(R⁻¹ A R⁻ᵀ)`, `P = R Rᵀ`, on fixed deterministic Rademacher probes
    /// shared across every λ (common random numbers ⇒ the REML criterion is a
    /// smooth deterministic function of λ). The same coarse deflation that makes
    /// the PCG iteration count n-independent makes `R⁻¹ A R⁻ᵀ` uniformly
    /// conditioned, so the Lanczos quadrature converges in a depth-independent
    /// number of steps too.
    fn logdet_slq(&self, lambda: f64) -> Result<f64, String> {
        let m = self.m;
        let prec = self.build_preconditioner(lambda)?;
        let logdet = prec.logdet();
        // M·v = R⁻¹ A R⁻ᵀ v (eigenvalues of P^{−1/2} A P^{−1/2}) without forming M.
        let mut scratch_in = vec![0.0; m];
        let mut scratch_out = vec![0.0; m];
        let mut vbuf = vec![0.0; m];
        let mut trace_est = 0.0;
        let steps = SLQ_LANCZOS_STEPS.min(m);
        let mut basis: Vec<Vec<f64>> = Vec::with_capacity(steps);
        for probe in 0..SLQ_PROBES {
            let mut rng =
                SplitMix64::new(RNG_SEED ^ (probe as u64).wrapping_mul(0xD134_2543_DE82_EF95));
            let mut q = vec![0.0; m];
            for qj in q.iter_mut() {
                *qj = rng.next_sign();
            }
            let z_norm2 = m as f64;
            let inv_norm = 1.0 / (m as f64).sqrt();
            for qj in q.iter_mut() {
                *qj *= inv_norm;
            }
            // Lanczos with full reorthogonalization.
            basis.clear();
            let mut alpha = Vec::with_capacity(steps);
            let mut beta: Vec<f64> = Vec::with_capacity(steps);
            let mut q_prev: Option<Vec<f64>> = None;
            for _step in 0..steps {
                // v = R⁻¹ A R⁻ᵀ q.
                prec.apply_r_inv_t(&q, &mut scratch_in);
                self.matvec(lambda, &scratch_in, &mut scratch_out);
                prec.apply_r_inv(&scratch_out, &mut vbuf);
                let mut v: Vec<f64> = vbuf.clone();
                let a: f64 = v.iter().zip(q.iter()).map(|(&x, &y)| x * y).sum();
                alpha.push(a);
                for j in 0..m {
                    v[j] -= a * q[j];
                }
                if let Some(prev) = &q_prev {
                    let b_prev = beta.last().copied().unwrap_or(0.0);
                    for j in 0..m {
                        v[j] -= b_prev * prev[j];
                    }
                }
                // Full reorthogonalization against the stored basis.
                basis.push(q.clone());
                for qb in &basis {
                    let proj: f64 = v.iter().zip(qb.iter()).map(|(&x, &y)| x * y).sum();
                    for j in 0..m {
                        v[j] -= proj * qb[j];
                    }
                }
                let b: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if !(b.is_finite()) {
                    return Err("residual cascade: Lanczos breakdown (non-finite norm)".into());
                }
                if b < 1e-13 {
                    break;
                }
                beta.push(b);
                q_prev = Some(std::mem::replace(&mut q, v));
                for qj in q.iter_mut() {
                    *qj /= b;
                }
            }
            beta.truncate(alpha.len().saturating_sub(1));
            let (theta, tau) = symmetric_tridiagonal_eigen(&alpha, &beta)?;
            let mut quad = 0.0;
            for (&t, &w0) in theta.iter().zip(tau.iter()) {
                if !(t.is_finite() && t > EIG_FLOOR) {
                    return Err(format!(
                        "residual cascade: non-positive Ritz value {t} in SLQ (system not PD)"
                    ));
                }
                quad += w0 * w0 * t.ln();
            }
            trace_est += z_norm2 * quad;
        }
        Ok(logdet + trace_est / SLQ_PROBES as f64)
    }

    /// Log-determinant through the route the sizing contract picks.
    fn logdet(&self, lambda: f64) -> Result<(f64, LogdetMethod), String> {
        if self.dense_gram.is_some() {
            Ok((self.logdet_dense(lambda)?, LogdetMethod::DenseExact))
        } else {
            Ok((self.logdet_slq(lambda)?, LogdetMethod::Slq))
        }
    }

    /// Coefficient solve at λ: dense Cholesky when cached, else certified PCG.
    fn solve_coeff(
        &self,
        lambda: f64,
        b: &[f64],
        warm: Option<&[f64]>,
    ) -> Result<(Vec<f64>, f64, usize), String> {
        // A core rebuilt from a persisted state carries no training design, only
        // the factored precision `L` of `A = X'WX + λD` at the fit's λ. Replay
        // the solve through it (exact — predict always solves at that same λ).
        if let Some(l) = &self.predict_chol {
            return Ok((chol_solve(l, self.m, b), 0.0, 0));
        }
        if let Some(mut a) = self.dense_system(lambda) {
            cholesky_logdet(&mut a, self.m)?;
            return Ok((chol_solve(&a, self.m, b), 0.0, 0));
        }
        self.pcg(lambda, b, warm)
    }

    /// Assemble the lower Cholesky factor `L` of `A = X'WX + λD` as a dense
    /// `m × m` row-major matrix — the factored precision a persisted predict
    /// replays its posterior-variance solve through. Uses the cached dense Gram
    /// when present; otherwise scatters the CSR row outer products into the
    /// upper triangle (one O(nnz·q) pass), the same assembly `build` uses under
    /// the sizing cap, just without the cap. Factoring is O(m³) — paid once at
    /// snapshot time, not per predict.
    fn assemble_predict_factor(&self, lambda: f64) -> Result<Vec<f64>, String> {
        let m = self.m;
        let mut a = vec![0.0_f64; m * m];
        if let Some(gram) = &self.dense_gram {
            for i in 0..m {
                for j in i..m {
                    let v = gram[i * m + j];
                    a[i * m + j] = v;
                    a[j * m + i] = v;
                }
            }
        } else {
            for i in 0..self.w.len() {
                let lo = self.row_ptr[i];
                let hi = self.row_ptr[i + 1];
                for ea in lo..hi {
                    let ca = self.col_idx[ea] as usize;
                    let va = self.w[i] * self.vals[ea];
                    for eb in ea..hi {
                        let cb = self.col_idx[eb] as usize;
                        a[ca * m + cb] += va * self.vals[eb];
                    }
                }
            }
            // Mirror the upper triangle into the lower.
            for i in 0..m {
                for j in i + 1..m {
                    a[j * m + i] = a[i * m + j];
                }
            }
        }
        for (i, d) in self.pen_diag.iter().enumerate() {
            a[i * m + i] += lambda * d;
        }
        cholesky_logdet(&mut a, m)?;
        Ok(a)
    }

    /// Penalized residual quadratic at a solution: `y'Wy − c'X'Wy`.
    fn rss_pen(&self, coeff: &[f64]) -> f64 {
        let mut quad = 0.0;
        for (c, r) in coeff.iter().zip(self.rhs.iter()) {
            quad += c * r;
        }
        self.ytwy - quad
    }

    /// Number of unpenalized (polynomial) columns.
    fn nullity(&self) -> usize {
        self.dim + 1
    }

    /// Working residual `r_i = y_i − (Xc)_i`.
    fn residuals(&self, coeff: &[f64]) -> Vec<f64> {
        let n = self.y.len();
        let mut r = Vec::with_capacity(n);
        for i in 0..n {
            let mut fit = 0.0;
            for e in self.row_ptr[i]..self.row_ptr[i + 1] {
                fit += self.vals[e] * coeff[self.col_idx[e] as usize];
            }
            r.push(self.y[i] - fit);
        }
        r
    }
}

// ──────────────────── symmetric tridiagonal eigensolver ─────────────────────

/// Eigenvalues and FIRST eigenvector components of a symmetric tridiagonal
/// matrix (diag `d`, off-diagonal `e`), by implicit-shift QL with the
/// first-row vector carried through the rotations — exactly what Lanczos
/// quadrature needs.
fn symmetric_tridiagonal_eigen(d: &[f64], e: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n = d.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut diag = d.to_vec();
    let mut off = vec![0.0; n];
    off[..n - 1].copy_from_slice(&e[..n - 1]);
    let mut first = vec![0.0; n];
    first[0] = 1.0;
    for l in 0..n {
        let mut iter = 0;
        loop {
            // Find a negligible off-diagonal to split at.
            let mut msplit = n - 1;
            for mm in l..n - 1 {
                let dd = diag[mm].abs() + diag[mm + 1].abs();
                if off[mm].abs() <= f64::EPSILON * dd {
                    msplit = mm;
                    break;
                }
            }
            if msplit == l {
                break;
            }
            iter += 1;
            if iter > 60 {
                return Err("residual cascade: tridiagonal QL failed to converge".into());
            }
            let mut g = (diag[l + 1] - diag[l]) / (2.0 * off[l]);
            let mut r = g.hypot(1.0);
            g = diag[msplit] - diag[l] + off[l] / (g + r.copysign(g));
            let (mut s, mut c) = (1.0, 1.0);
            let mut p = 0.0;
            let mut broke_early = false;
            for i in (l..msplit).rev() {
                let mut f = s * off[i];
                let b = c * off[i];
                r = f.hypot(g);
                off[i + 1] = r;
                if r == 0.0 {
                    diag[i + 1] -= p;
                    off[msplit] = 0.0;
                    broke_early = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = diag[i + 1] - p;
                r = (diag[i] - g) * s + 2.0 * c * b;
                p = s * r;
                diag[i + 1] = g + p;
                g = c * r - b;
                // Carry the first-row eigenvector components.
                f = first[i + 1];
                first[i + 1] = s * first[i] + c * f;
                first[i] = c * first[i] - s * f;
            }
            if broke_early {
                continue;
            }
            diag[l] -= p;
            off[l] = g;
            off[msplit] = 0.0;
        }
    }
    Ok((diag, first))
}

// ───────────────────────────── net construction ─────────────────────────────

/// Extend a nested net to covering radius `h` over the DOMAIN: first every data
/// point further than `h` from the (seeded) net becomes a new center, then every
/// cell of the `h`-grid over the bounding box `[0, box_hi]` whose centre is not
/// yet within `h` of the net is filled with a synthetic center. O((n + box
/// cells)·3^d). Returns the new centers.
///
/// Covering the box, not merely the data cloud, is what the multilevel Wendland
/// norm-equivalence (Narcowich–Ward inverse estimates + Le Gia–Wendland
/// multilevel stability) actually requires: the nested centres must be
/// quasi-uniform over the domain Ω. In data-dense regions every cell is already
/// covered by a data center, so the fill is a no-op there; in a data void it
/// plants the fine centres whose coefficients carry no data and revert to the
/// prior — the mechanism by which the posterior mean bridges a gap (coarse
/// data-pinned bumps) while the posterior variance GROWS into it (fine void
/// bumps the data cannot pin). The synthetic centres carry (almost) no data
/// rows, so their Gram diagonal is ~0 and they land in the penalty-dominated
/// fine block where the Jacobi preconditioner is exact — they neither perturb
/// the coarse factorization nor the n-independent iteration count.
fn extend_net(
    net: &mut Vec<[f64; 3]>,
    points: &[[f64; 3]],
    dim: usize,
    h: f64,
    box_hi: &[f64; 3],
) -> Vec<[f64; 3]> {
    let mut grid = HashGrid::new(h, dim);
    for (idx, c) in net.iter().enumerate() {
        grid.insert(idx as u32, c);
    }
    let h2 = h * h;
    let mut new_centers = Vec::new();
    let try_add = |net: &mut Vec<[f64; 3]>,
                   grid: &mut HashGrid,
                   new_centers: &mut Vec<[f64; 3]>,
                   p: &[f64; 3]| {
        let mut covered = false;
        grid.for_neighbors(p, |j| {
            if !covered && dist2(p, &net[j as usize], dim) <= h2 {
                covered = true;
            }
        });
        if !covered {
            let idx = net.len() as u32;
            net.push(*p);
            grid.insert(idx, p);
            new_centers.push(*p);
        }
    };
    for p in points {
        try_add(net, &mut grid, &mut new_centers, p);
    }
    // Fill the bounding box so the net covers the domain, not just the data.
    let mut cells = [1_i64; 3];
    for a in 0..dim {
        cells[a] = (box_hi[a] / h).ceil() as i64 + 1;
    }
    let mut c = [0.0_f64; 3];
    for i0 in 0..cells[0] {
        c[0] = (i0 as f64 + 0.5) * h;
        for i1 in 0..cells[1] {
            if dim > 1 {
                c[1] = (i1 as f64 + 0.5) * h;
            }
            for i2 in 0..cells[2] {
                if dim > 2 {
                    c[2] = (i2 as f64 + 0.5) * h;
                }
                try_add(net, &mut grid, &mut new_centers, &c);
            }
        }
    }
    new_centers
}

impl ResidualCascadeDesign {
    /// Build the cascade design: validate, scale by the metric, grow `levels`
    /// nested nets, and assemble the sparse design plus its sufficient
    /// statistics in O(n·(levels + 3^d)).
    ///
    /// `xs` holds one slice per axis (2 or 3 of them), `metric` the positive
    /// per-axis scaling of the learned metric, `sobolev_s` the Sobolev order
    /// of the equivalent (semi)norm — must satisfy `d/2 < s ≤ (d+3)/2` (the
    /// Wendland-(3,1) native smoothness).
    pub fn build(
        xs: &[&[f64]],
        y: &[f64],
        w: &[f64],
        metric: &[f64],
        sobolev_s: f64,
        levels: usize,
    ) -> Result<Self, String> {
        let dim = xs.len();
        if !(dim == 2 || dim == 3) {
            return Err(format!(
                "residual cascade: built for scattered 2-3D smooths, got {dim} axes"
            ));
        }
        let n = y.len();
        if w.len() != n || xs.iter().any(|x| x.len() != n) {
            return Err(format!(
                "residual cascade: length mismatch (y={n}, w={}, axes={:?})",
                w.len(),
                xs.iter().map(|x| x.len()).collect::<Vec<_>>()
            ));
        }
        if n <= dim + 1 {
            return Err(format!(
                "residual cascade: needs more than {} rows for the profiled REML degrees of \
                 freedom, got {n}",
                dim + 1
            ));
        }
        if metric.len() != dim || metric.iter().any(|&s| !(s.is_finite() && s > 0.0)) {
            return Err(format!(
                "residual cascade: metric must be {dim} finite positive scales, got {metric:?}"
            ));
        }
        if !(sobolev_s > dim as f64 / 2.0 && sobolev_s <= (dim as f64 + 3.0) / 2.0) {
            return Err(format!(
                "residual cascade: sobolev_s must lie in (d/2, (d+3)/2] = ({}, {}] for the \
                 Wendland-(3,1) bump, got {sobolev_s}",
                dim as f64 / 2.0,
                (dim as f64 + 3.0) / 2.0
            ));
        }
        if levels == 0 || levels > MAX_LEVELS {
            return Err(format!(
                "residual cascade: levels must be in 1..={MAX_LEVELS}, got {levels}"
            ));
        }
        for i in 0..n {
            if !(y[i].is_finite() && w[i].is_finite() && w[i] > 0.0)
                || xs.iter().any(|x| !x[i].is_finite())
            {
                return Err(format!(
                    "residual cascade: non-finite or non-positive input at row {i}"
                ));
            }
        }
        // Scaled, corner-shifted coordinates.
        let mut z_lo = [f64::INFINITY; 3];
        let mut z_hi = [f64::NEG_INFINITY; 3];
        for a in 0..dim {
            for &v in xs[a] {
                let s = metric[a] * v;
                z_lo[a] = z_lo[a].min(s);
                z_hi[a] = z_hi[a].max(s);
            }
        }
        let mut z_range = [1.0_f64; 3];
        let mut max_range = 0.0_f64;
        for a in 0..dim {
            if !(z_hi[a] > z_lo[a]) {
                return Err(format!(
                    "residual cascade: degenerate axis {a} bounding box [{}, {}]",
                    z_lo[a], z_hi[a]
                ));
            }
            z_range[a] = z_hi[a] - z_lo[a];
            max_range = max_range.max(z_range[a]);
        }
        for a in dim..3 {
            z_lo[a] = 0.0;
        }
        let z: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                let mut p = [0.0_f64; 3];
                for a in 0..dim {
                    p[a] = metric[a] * xs[a][i] - z_lo[a];
                }
                p
            })
            .collect();
        let mut metric3 = [1.0_f64; 3];
        metric3[..dim].copy_from_slice(metric);

        let h0 = H0_FRACTION * max_range;
        let mut net: Vec<[f64; 3]> = Vec::new();
        let mut level_specs = Vec::with_capacity(levels);
        let mut col = dim + 1;
        let mut pen_logdet_const = 0.0;
        for l in 0..levels {
            let h = h0 * 0.5_f64.powi(l as i32);
            let new_centers = extend_net(&mut net, &z, dim, h, &z_range);
            if net.len() > MAX_CENTERS {
                return Err(format!(
                    "residual cascade: center cap {MAX_CENTERS} exceeded at level {l}"
                ));
            }
            let weight = level_weight(l, sobolev_s, dim);
            pen_logdet_const += new_centers.len() as f64 * weight.ln();
            let delta = OVERLAP * h;
            let mut grid = HashGrid::new(delta, dim);
            for (j, c) in new_centers.iter().enumerate() {
                grid.insert(j as u32, c);
            }
            let col_offset = col;
            col += new_centers.len();
            level_specs.push(Level {
                h,
                delta,
                weight,
                centers: new_centers,
                col_offset,
                grid,
            });
        }
        let m = col;

        // CSR assembly + sufficient statistics in one pass.
        let mut row_ptr = Vec::with_capacity(n + 1);
        row_ptr.push(0_usize);
        let mut col_idx: Vec<u32> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        let mut rhs = vec![0.0_f64; m];
        let mut gram_diag = vec![0.0_f64; m];
        let mut ytwy = 0.0_f64;
        let probe_core = CoreScaffold {
            dim,
            z_range,
            levels: &level_specs,
        };
        for i in 0..n {
            let row = probe_core.basis_row(&z[i]);
            for &(c, v) in &row {
                col_idx.push(c as u32);
                vals.push(v);
                rhs[c] += w[i] * y[i] * v;
                gram_diag[c] += w[i] * v * v;
            }
            ytwy += w[i] * y[i] * y[i];
            row_ptr.push(col_idx.len());
        }
        let mut pen_diag = vec![0.0_f64; m];
        for level in &level_specs {
            for j in 0..level.centers.len() {
                pen_diag[level.col_offset + j] = level.weight;
            }
        }

        // Dense Gram cache under the sizing cap: O(n·q²) scatter of row outer
        // products into the upper triangle.
        let dense_gram = if m <= DENSE_GRAM_MAX {
            let mut gram = vec![0.0_f64; m * m];
            for i in 0..n {
                let lo = row_ptr[i];
                let hi = row_ptr[i + 1];
                for ea in lo..hi {
                    let ca = col_idx[ea] as usize;
                    let va = w[i] * vals[ea];
                    for eb in ea..hi {
                        gram[ca * m + col_idx[eb] as usize] += va * vals[eb];
                    }
                }
            }
            Some(gram)
        } else {
            None
        };

        Ok(ResidualCascadeDesign {
            core: Arc::new(Core {
                dim,
                metric: metric3,
                z_lo,
                z_range,
                sobolev_s,
                levels: level_specs,
                net,
                m,
                row_ptr,
                col_idx,
                vals,
                w: w.to_vec(),
                y: y.to_vec(),
                z,
                rhs,
                ytwy,
                gram_diag,
                pen_diag,
                pen_logdet_const,
                dense_gram,
                predict_chol: None,
            }),
        })
    }

    /// Number of resolution levels.
    pub fn num_levels(&self) -> usize {
        self.core.levels.len()
    }

    /// Aspect ratio of the metric-scaled point cloud: the ratio of the largest
    /// to smallest per-axis standard deviation of the scaled coordinates `z`.
    /// This is the metric-condition measure the quasi-uniformity guard (issue
    /// #1032, caveat 2) keys on — see [`QUASI_UNIFORMITY_MAX_ASPECT`]. A value
    /// near 1 is an isotropic (benign) cloud; a large value means the metric
    /// has collapsed the data onto a lower-dimensional sheet in `z`, breaking
    /// the BPX n-independent iteration bound.
    pub fn metric_scaled_aspect_ratio(&self) -> f64 {
        let dim = self.core.dim;
        let n = self.core.z.len();
        if dim == 0 || n == 0 {
            return 1.0;
        }
        let mut mean = [0.0_f64; 3];
        for p in &self.core.z {
            for a in 0..dim {
                mean[a] += p[a];
            }
        }
        for m in mean.iter_mut().take(dim) {
            *m /= n as f64;
        }
        let mut var = [0.0_f64; 3];
        for p in &self.core.z {
            for a in 0..dim {
                let d = p[a] - mean[a];
                var[a] += d * d;
            }
        }
        let mut sd_lo = f64::INFINITY;
        let mut sd_hi = 0.0_f64;
        for v in var.iter().take(dim) {
            let sd = (v / n as f64).sqrt();
            sd_lo = sd_lo.min(sd);
            sd_hi = sd_hi.max(sd);
        }
        if !(sd_lo > 0.0 && sd_lo.is_finite()) {
            // A collapsed axis (zero scaled spread) is maximally degenerate.
            return f64::INFINITY;
        }
        sd_hi / sd_lo
    }

    /// Quasi-uniformity certificate (issue #1032, caveat 2): `true` iff the
    /// metric-scaled cloud is isotropic enough that the BPX n-independent CG
    /// iteration bound is trustworthy. When this returns `false` the auto-route
    /// MUST fall back to the dense kernel path rather than pay an iterative
    /// solve whose iteration count is no longer n-independent — the CG residual
    /// certificate would still *catch* a mis-solve at [`CG_MAX_ITERS`], but the
    /// guard prevents the silent O(n·iters) blow-up up front.
    pub fn quasi_uniformity_certified(&self) -> bool {
        self.metric_scaled_aspect_ratio() <= QUASI_UNIFORMITY_MAX_ASPECT
    }

    /// Number of columns `ncoarse` in the additive-Schwarz coarse space at `log
    /// λ` (the polynomial layer plus the data-dominated coarsest levels). The
    /// iterative-route preconditioner solves the principal `[0, ncoarse)` block
    /// of `A = X'WX + λD` exactly and Jacobi-preconditions the fine tail; exposed
    /// so the conditioning oracle can reconstruct that block-arrow preconditioner
    /// from the public dense system and certify it is uniformly conditioned in
    /// depth. See [`COARSE_DOMINANCE`].
    pub fn coarse_space_cols(&self, log_lambda: f64) -> usize {
        self.core.coarse_space_cols(log_lambda.exp())
    }

    /// Total coefficient count (`dim + 1` polynomial + all centers).
    pub fn num_coeffs(&self) -> usize {
        self.core.m
    }

    /// Number of stored nonzeros in the CSR design.
    pub fn num_nonzeros(&self) -> usize {
        self.core.vals.len()
    }

    /// Total centers across all levels.
    pub fn num_centers(&self) -> usize {
        self.core.m - self.core.nullity()
    }

    /// NEW centers of one level in ORIGINAL (unscaled) coordinates.
    pub fn centers(&self, level: usize) -> Vec<Vec<f64>> {
        let lv = &self.core.levels[level];
        lv.centers
            .iter()
            .map(|c| {
                (0..self.core.dim)
                    .map(|a| (c[a] + self.core.z_lo[a]) / self.core.metric[a])
                    .collect()
            })
            .collect()
    }

    /// Support radius δ_l of one level, in scaled coordinates.
    pub fn support_radius(&self, level: usize) -> f64 {
        self.core.levels[level].delta
    }

    /// Prior precision weight `d_l = 4^{l(s−d/2)}` of one level.
    pub fn level_weight(&self, level: usize) -> f64 {
        self.core.levels[level].weight
    }

    /// Sparse basis row at a raw point, as (column, value) pairs sorted by
    /// column within each block — the exact row the fit used for training
    /// rows, exposed so oracles can assemble the dense system independently.
    pub fn basis_row(&self, x: &[f64]) -> Result<Vec<(usize, f64)>, String> {
        self.check_point(x)?;
        Ok(self.core.basis_row_scaled(&self.core.scale_point(x)))
    }

    fn check_point(&self, x: &[f64]) -> Result<(), String> {
        if x.len() != self.core.dim || x.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "residual cascade: point must be {} finite coordinates, got {x:?}",
                self.core.dim
            ));
        }
        Ok(())
    }

    /// Exact penalty quadratic `c'Dc` (unit-λ multilevel prior energy).
    pub fn penalty_value(&self, coeff: &[f64]) -> Result<f64, String> {
        if coeff.len() != self.core.m {
            return Err(format!(
                "residual cascade: coefficient length {} != {}",
                coeff.len(),
                self.core.m
            ));
        }
        Ok(coeff
            .iter()
            .zip(self.core.pen_diag.iter())
            .map(|(&c, &d)| d * c * c)
            .sum())
    }

    /// Exact dense log-determinant of `X'WX + λD` (errors past the sizing
    /// cap) — exposed for the in-test SLQ-vs-exact oracle.
    pub fn logdet_exact(&self, log_lambda: f64) -> Result<f64, String> {
        self.core.logdet_dense(log_lambda.exp())
    }

    /// SLQ log-determinant estimate on the fixed deterministic probes —
    /// exposed for the in-test SLQ-vs-exact oracle.
    pub fn logdet_slq(&self, log_lambda: f64) -> Result<f64, String> {
        self.core.logdet_slq(log_lambda.exp())
    }

    /// Profiled-σ² REML criterion at `log λ` (differences across λ are exact
    /// REML differences on the dense route; SLQ-estimated past the cap).
    pub fn criterion(&self, log_lambda: f64) -> Result<f64, String> {
        Ok(self.criterion_with_warm(log_lambda, None)?.0)
    }

    fn criterion_with_warm(
        &self,
        log_lambda: f64,
        warm: Option<&[f64]>,
    ) -> Result<(f64, Vec<f64>), String> {
        if !log_lambda.is_finite() {
            return Err(format!(
                "residual cascade: non-finite log lambda {log_lambda}"
            ));
        }
        let core = &self.core;
        let lambda = log_lambda.exp();
        let (coeff, _, _) = core.solve_coeff(lambda, &core.rhs, warm)?;
        let rss_pen = core.rss_pen(&coeff);
        if !(rss_pen > 0.0) {
            return Err(format!(
                "residual cascade: degenerate penalized residual {rss_pen}"
            ));
        }
        let (logdet, _) = core.logdet(lambda)?;
        let dof = (core.y.len() - core.nullity()) as f64;
        let r = (core.m - core.nullity()) as f64;
        let sigma2 = rss_pen / dof;
        Ok((
            -0.5 * (logdet - r * log_lambda - core.pen_logdet_const + dof * sigma2.ln()),
            coeff,
        ))
    }

    /// Fit at a FIXED `log λ`, with σ² either supplied or profiled.
    pub fn fit_at(
        &self,
        log_lambda: f64,
        sigma2: Option<f64>,
    ) -> Result<ResidualCascadeFit, String> {
        self.fit_at_with_warm(log_lambda, sigma2, None)
    }

    fn fit_at_with_warm(
        &self,
        log_lambda: f64,
        sigma2: Option<f64>,
        warm: Option<&[f64]>,
    ) -> Result<ResidualCascadeFit, String> {
        if !log_lambda.is_finite() {
            return Err(format!(
                "residual cascade: non-finite log lambda {log_lambda}"
            ));
        }
        let core = &self.core;
        let lambda = log_lambda.exp();
        let (coeff, rel_res, iters) = core.solve_coeff(lambda, &core.rhs, warm)?;
        let rss_pen = core.rss_pen(&coeff);
        let dof = (core.y.len() - core.nullity()) as f64;
        let sigma2 = match sigma2 {
            Some(s) => {
                if !(s.is_finite() && s > 0.0) {
                    return Err(format!("residual cascade: invalid sigma2 {s}"));
                }
                s
            }
            None => {
                if !(rss_pen > 0.0) {
                    return Err(format!(
                        "residual cascade: degenerate penalized residual {rss_pen}"
                    ));
                }
                rss_pen / dof
            }
        };
        let (logdet, logdet_method) = core.logdet(lambda)?;
        let r = (core.m - core.nullity()) as f64;
        // Full restricted log-likelihood at this (λ, σ²) up to λ- and σ-free
        // constants; at the profiled σ̂² the quadratic collapses to `dof`.
        let restricted_loglik = -0.5
            * (logdet - r * log_lambda - core.pen_logdet_const
                + dof * sigma2.ln()
                + rss_pen / sigma2);
        let predict_chol = if core.dense_gram.is_some() {
            Some(core.assemble_predict_factor(lambda)?)
        } else {
            None
        };
        Ok(ResidualCascadeFit {
            core: Arc::clone(&self.core),
            predict_chol,
            coeff,
            log_lambda,
            sigma2,
            restricted_loglik,
            rss_pen,
            certificate: CascadeCertificate {
                solve_rel_residual: rel_res,
                solve_iters: iters,
                logdet_method,
            },
            refinement: None,
        })
    }

    /// Fit with `log λ` selected by the profiled REML criterion:
    /// deterministic coarse grid then golden-section refinement (the SLQ
    /// probes are fixed, so the iterative-route criterion is just as
    /// deterministic — same data, same fit).
    pub fn fit_reml(&self) -> Result<ResidualCascadeFit, String> {
        let mut best_i = 0usize;
        let mut best_v = f64::NEG_INFINITY;
        let mut best_coeff = Vec::new();
        let mut warm: Option<Vec<f64>> = None;
        let step = (LOG_LAMBDA_HI - LOG_LAMBDA_LO) / (LOG_LAMBDA_GRID - 1) as f64;
        for i in 0..LOG_LAMBDA_GRID {
            let ll = LOG_LAMBDA_LO + step * i as f64;
            let (v, coeff) = self.criterion_with_warm(ll, warm.as_deref())?;
            if v > best_v {
                best_v = v;
                best_i = i;
                best_coeff = coeff.clone();
            }
            warm = Some(coeff);
        }
        let mut lo = LOG_LAMBDA_LO + step * best_i.saturating_sub(1) as f64;
        let mut hi = (LOG_LAMBDA_LO + step * (best_i + 1) as f64).min(LOG_LAMBDA_HI);
        let inv_phi = 0.618_033_988_749_894_9_f64;
        let mut x1 = hi - inv_phi * (hi - lo);
        let mut x2 = lo + inv_phi * (hi - lo);
        let (mut f1, mut c1) = self.criterion_with_warm(x1, Some(&best_coeff))?;
        let (mut f2, mut c2) = self.criterion_with_warm(x2, Some(&c1))?;
        while hi - lo > LOG_LAMBDA_TOL {
            if f1 < f2 {
                lo = x1;
                x1 = x2;
                f1 = f2;
                c1 = c2;
                x2 = lo + inv_phi * (hi - lo);
                (f2, c2) = self.criterion_with_warm(x2, Some(&c1))?;
            } else {
                hi = x2;
                x2 = x1;
                f2 = f1;
                c2 = c1;
                x1 = hi - inv_phi * (hi - lo);
                (f1, c1) = self.criterion_with_warm(x1, Some(&c2))?;
            }
        }
        let warm = if f1 >= f2 { &c1 } else { &c2 };
        self.fit_at_with_warm(0.5 * (lo + hi), None, Some(warm))
    }

    /// Exact upper bound on the penalized-objective decrease available from
    /// appending the candidate level L+1 at this fit's λ:
    /// `‖X₂'W r̂‖² / (λ·d_{L+1})` (see the module header for the Schur-
    /// complement argument). Returns `None` when the net is exhausted (no new
    /// centers — every data point is already a center).
    pub fn next_level_gain_bound(&self, fit: &ResidualCascadeFit) -> Result<Option<f64>, String> {
        let core = &self.core;
        if !Arc::ptr_eq(core, &fit.core) {
            return Err("residual cascade: fit does not belong to this design".into());
        }
        let next_l = core.levels.len();
        if next_l >= MAX_LEVELS {
            return Ok(None);
        }
        let h = core.levels[next_l - 1].h * 0.5;
        let mut net = core.net.clone();
        let candidates = extend_net(&mut net, &core.z, core.dim, h, &core.z_range);
        if candidates.is_empty() || net.len() > MAX_CENTERS {
            return Ok(None);
        }
        let delta = OVERLAP * h;
        let mut grid = HashGrid::new(delta, core.dim);
        for (j, c) in candidates.iter().enumerate() {
            grid.insert(j as u32, c);
        }
        let r = core.residuals(&fit.coeff);
        let mut g = vec![0.0_f64; candidates.len()];
        for (i, zi) in core.z.iter().enumerate() {
            let wr = core.w[i] * r[i];
            grid.for_neighbors(zi, |j| {
                let rad = dist2(zi, &candidates[j as usize], core.dim).sqrt() / delta;
                g[j as usize] += wr * wendland(rad);
            });
        }
        let g2: f64 = g.iter().map(|v| v * v).sum();
        let d_next = level_weight(next_l, core.sobolev_s, core.dim);
        Ok(Some(g2 / (fit.log_lambda.exp() * d_next)))
    }
}

/// Prior precision weight of level `l`: `4^{l(s−d/2)}`.
fn level_weight(l: usize, sobolev_s: f64, dim: usize) -> f64 {
    (4.0_f64).powf(l as f64 * (sobolev_s - dim as f64 / 2.0))
}

/// Lightweight view used during assembly, before the Core exists: shares the
/// exact basis-row logic with [`Core::basis_row_scaled`] so the assembled CSR
/// and later prediction rows cannot drift apart.
struct CoreScaffold<'a> {
    dim: usize,
    z_range: [f64; 3],
    levels: &'a [Level],
}

impl CoreScaffold<'_> {
    fn basis_row(&self, z: &[f64; 3]) -> Vec<(usize, f64)> {
        let mut row = Vec::with_capacity(self.dim + 1 + self.levels.len() * 8);
        row.push((0, 1.0));
        for a in 0..self.dim {
            row.push((a + 1, 2.0 * z[a] / self.z_range[a] - 1.0));
        }
        for level in self.levels {
            let start = row.len();
            level.grid.for_neighbors(z, |j| {
                let c = &level.centers[j as usize];
                let r = dist2(z, c, self.dim).sqrt() / level.delta;
                let v = wendland(r);
                if v > 0.0 {
                    row.push((level.col_offset + j as usize, v));
                }
            });
            row[start..].sort_unstable_by_key(|&(col, _)| col);
        }
        row
    }
}

impl ResidualCascadeFit {
    /// Posterior `(mean, variance)` at a raw point: the sparse basis row
    /// dotted with the coefficients, and `σ̂²·x'(X'WX+λD)^{−1}x` through one
    /// certified solve.
    pub fn predict(&self, x: &[f64]) -> Result<(f64, f64), String> {
        let core = &self.core;
        if x.len() != core.dim || x.iter().any(|v| !v.is_finite()) {
            return Err(format!(
                "residual cascade: prediction point must be {} finite coordinates, got {x:?}",
                core.dim
            ));
        }
        let row = core.basis_row_scaled(&core.scale_point(x));
        let mut mean = 0.0;
        let mut dense_row = vec![0.0_f64; core.m];
        for &(c, v) in &row {
            mean += v * self.coeff[c];
            dense_row[c] += v;
        }
        let lambda = self.log_lambda.exp();
        let zsol = if let Some(l) = &self.predict_chol {
            chol_solve(l, core.m, &dense_row)
        } else {
            core.solve_coeff(lambda, &dense_row, None)?.0
        };
        let mut quad = 0.0;
        for (a, b) in dense_row.iter().zip(zsol.iter()) {
            quad += a * b;
        }
        Ok((mean, self.sigma2 * quad))
    }

    /// EXACT posterior coefficient samples by perturb-and-solve:
    /// `c_s = A^{−1}(X'Wy + σ(X'W^{1/2}z₁ + √λ D^{1/2}z₂))` has mean ĉ and
    /// covariance exactly `σ̂²A^{−1}`. Deterministically seeded; one certified
    /// solve per sample (warm-started at the mode).
    pub fn sample_coefficients(&self, n_samples: usize) -> Result<Vec<Vec<f64>>, String> {
        let core = &self.core;
        let lambda = self.log_lambda.exp();
        let sigma = self.sigma2.sqrt();
        let sqrt_lambda = lambda.sqrt();
        let n = core.y.len();
        let mut rng = SplitMix64::new(RNG_SEED ^ 0xA11C_E5A_u64);
        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut b = core.rhs.clone();
            // X'W^{1/2} z₁: one CSR pass with per-row factor √w_i·z₁_i.
            for i in 0..n {
                let f = sigma * core.w[i].sqrt() * rng.next_normal();
                for e in core.row_ptr[i]..core.row_ptr[i + 1] {
                    b[core.col_idx[e] as usize] += f * core.vals[e];
                }
            }
            // √λ D^{1/2} z₂ on the penalized columns.
            for (bj, &dj) in b.iter_mut().zip(core.pen_diag.iter()) {
                if dj > 0.0 {
                    *bj += sigma * sqrt_lambda * dj.sqrt() * rng.next_normal();
                }
            }
            let (c, _, _) = core.solve_coeff(lambda, &b, Some(&self.coeff))?;
            samples.push(c);
        }
        Ok(samples)
    }

    /// Number of resolution levels in the fitted cascade.
    pub fn num_levels(&self) -> usize {
        self.core.levels.len()
    }

    /// Total coefficient count.
    pub fn num_coeffs(&self) -> usize {
        self.core.m
    }

    /// Total centers across all fitted resolution levels.
    pub fn num_centers(&self) -> usize {
        self.core.m - self.core.nullity()
    }

    /// Snapshot the fit for persistence (#1032). Assembles the factored
    /// precision `L` of `A = X'WX + λD` at the fit's λ (O(m³) once) and copies
    /// the nested geometry + coefficients, dropping all training rows. The
    /// resulting [`ResidualCascadeState`] is predict-complete: `from_state`
    /// replays the posterior mean+variance bit-for-bit.
    pub fn to_state(&self) -> Result<ResidualCascadeState, String> {
        let core = &self.core;
        let lambda = self.log_lambda.exp();
        let predict_chol = if let Some(l) = &self.predict_chol {
            l.clone()
        } else if let Some(l) = &core.predict_chol {
            l.clone()
        } else {
            core.assemble_predict_factor(lambda)?
        };
        let dim = core.dim;
        let levels = core
            .levels
            .iter()
            .map(|level| {
                let mut centers = Vec::with_capacity(level.centers.len() * dim);
                for c in &level.centers {
                    centers.extend_from_slice(&c[..dim]);
                }
                LevelState {
                    h: level.h,
                    delta: level.delta,
                    weight: level.weight,
                    col_offset: level.col_offset as u64,
                    centers,
                }
            })
            .collect();
        Ok(ResidualCascadeState {
            dim: dim as u64,
            metric: core.metric,
            z_lo: core.z_lo,
            z_range: core.z_range,
            sobolev_s: core.sobolev_s,
            levels,
            m: core.m as u64,
            pen_logdet_const: core.pen_logdet_const,
            coeff: self.coeff.clone(),
            log_lambda: self.log_lambda,
            sigma2: self.sigma2,
            restricted_loglik: self.restricted_loglik,
            rss_pen: self.rss_pen,
            predict_chol,
        })
    }

    /// Rebuild a predict-capable fit from a snapshot (#1032). Validates shape,
    /// finiteness, the Sobolev/Wendland window, strictly-positive level weights
    /// and box ranges, the column accounting (`m = dim+1 + Σ centers`, matching
    /// `col_offset`s), positive σ², and that `predict_chol` is a valid `m × m`
    /// lower factor (positive pivots) — so a corrupt payload fails here, not in
    /// a later `predict`. The restored `Core` has empty training CSR and
    /// `predict_chol = Some(L)`; its `predict` reads only geometry (mean) and
    /// the factor (variance), replaying both exactly.
    pub fn from_state(state: &ResidualCascadeState) -> Result<Self, String> {
        let dim = state.dim as usize;
        if !(dim == 2 || dim == 3) {
            return Err(format!(
                "residual cascade state: dim must be 2 or 3, got {dim}"
            ));
        }
        if !(state.sobolev_s > dim as f64 / 2.0 && state.sobolev_s <= (dim as f64 + 3.0) / 2.0) {
            return Err(format!(
                "residual cascade state: sobolev_s {} outside the Wendland window ({}, {}]",
                state.sobolev_s,
                dim as f64 / 2.0,
                (dim as f64 + 3.0) / 2.0
            ));
        }
        for a in 0..dim {
            if !(state.metric[a].is_finite() && state.metric[a] > 0.0) {
                return Err(format!(
                    "residual cascade state: metric axis {a} must be finite positive, got {}",
                    state.metric[a]
                ));
            }
            if !(state.z_range[a].is_finite()
                && state.z_range[a] > 0.0
                && state.z_lo[a].is_finite())
            {
                return Err(format!(
                    "residual cascade state: degenerate box on axis {a} (lo={}, range={})",
                    state.z_lo[a], state.z_range[a]
                ));
            }
        }
        let m = state.m as usize;
        let mut metric3 = [1.0_f64; 3];
        metric3[..dim].copy_from_slice(&state.metric[..dim]);
        let mut z_lo = [0.0_f64; 3];
        let mut z_range = [1.0_f64; 3];
        z_lo[..dim].copy_from_slice(&state.z_lo[..dim]);
        z_range[..dim].copy_from_slice(&state.z_range[..dim]);

        // Rebuild the levels and their lookup grids from the flattened centers,
        // checking the column accounting matches the polynomial layer + blocks.
        let mut levels = Vec::with_capacity(state.levels.len());
        let mut net: Vec<[f64; 3]> = Vec::new();
        let mut pen_diag = vec![0.0_f64; m];
        let mut expected_offset = dim + 1;
        for (li, ls) in state.levels.iter().enumerate() {
            if !(ls.h.is_finite() && ls.h > 0.0 && ls.delta.is_finite() && ls.delta > 0.0) {
                return Err(format!(
                    "residual cascade state: level {li} has non-positive h/delta ({}, {})",
                    ls.h, ls.delta
                ));
            }
            if !(ls.weight.is_finite() && ls.weight > 0.0) {
                return Err(format!(
                    "residual cascade state: level {li} has non-positive prior weight {}",
                    ls.weight
                ));
            }
            if ls.centers.len() % dim != 0 {
                return Err(format!(
                    "residual cascade state: level {li} centers length {} not a multiple of dim {dim}",
                    ls.centers.len()
                ));
            }
            let n_centers = ls.centers.len() / dim;
            let col_offset = ls.col_offset as usize;
            if col_offset != expected_offset {
                return Err(format!(
                    "residual cascade state: level {li} col_offset {col_offset} ≠ expected {expected_offset}"
                ));
            }
            let mut grid = HashGrid::new(ls.delta, dim);
            let mut centers = Vec::with_capacity(n_centers);
            for j in 0..n_centers {
                let mut c = [0.0_f64; 3];
                for a in 0..dim {
                    let v = ls.centers[j * dim + a];
                    if !v.is_finite() {
                        return Err(format!(
                            "residual cascade state: non-finite center coordinate at level {li}, center {j}"
                        ));
                    }
                    c[a] = v;
                }
                grid.insert(j as u32, &c);
                centers.push(c);
                net.push(c);
                let col = col_offset + j;
                if col >= m {
                    return Err(format!(
                        "residual cascade state: level {li} column {col} exceeds m {m}"
                    ));
                }
                pen_diag[col] = ls.weight;
            }
            expected_offset = col_offset + n_centers;
            levels.push(Level {
                h: ls.h,
                delta: ls.delta,
                weight: ls.weight,
                centers,
                col_offset,
                grid,
            });
        }
        if expected_offset != m {
            return Err(format!(
                "residual cascade state: column accounting mismatch (dim+1+Σcenters = {expected_offset} ≠ m {m})"
            ));
        }
        if state.coeff.len() != m {
            return Err(format!(
                "residual cascade state: coeff length {} ≠ m {m}",
                state.coeff.len()
            ));
        }
        if state.predict_chol.len() != m * m {
            return Err(format!(
                "residual cascade state: predict_chol must be m×m = {m}² = {}, got {}",
                m * m,
                state.predict_chol.len()
            ));
        }
        for (i, v) in state
            .coeff
            .iter()
            .chain(state.predict_chol.iter())
            .enumerate()
        {
            if !v.is_finite() {
                return Err(format!("residual cascade state: non-finite entry at {i}"));
            }
        }
        for g in 0..m {
            let piv = state.predict_chol[g * m + g];
            if !(piv.is_finite() && piv > 0.0) {
                return Err(format!(
                    "residual cascade state: non-positive Cholesky pivot {piv} at index {g}"
                ));
            }
        }
        if !(state.log_lambda.is_finite()
            && state.sigma2.is_finite()
            && state.sigma2 > 0.0
            && state.restricted_loglik.is_finite()
            && state.rss_pen.is_finite())
        {
            return Err(format!(
                "residual cascade state: invalid scalars (log_lambda={}, sigma2={}, restricted_loglik={}, rss_pen={})",
                state.log_lambda, state.sigma2, state.restricted_loglik, state.rss_pen
            ));
        }
        let core = Core {
            dim,
            metric: metric3,
            z_lo,
            z_range,
            sobolev_s: state.sobolev_s,
            levels,
            net,
            m,
            row_ptr: Vec::new(),
            col_idx: Vec::new(),
            vals: Vec::new(),
            w: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            rhs: Vec::new(),
            ytwy: 0.0,
            gram_diag: Vec::new(),
            pen_diag,
            pen_logdet_const: state.pen_logdet_const,
            dense_gram: None,
            predict_chol: Some(state.predict_chol.clone()),
        };
        Ok(ResidualCascadeFit {
            core: Arc::new(core),
            predict_chol: None,
            coeff: state.coeff.clone(),
            log_lambda: state.log_lambda,
            sigma2: state.sigma2,
            restricted_loglik: state.restricted_loglik,
            rss_pen: state.rss_pen,
            certificate: CascadeCertificate {
                solve_rel_residual: 0.0,
                solve_iters: 0,
                logdet_method: LogdetMethod::DenseExact,
            },
            refinement: None,
        })
    }
}

/// Fit the full magic-default cascade: start at [`INITIAL_LEVELS`], REML-fit,
/// and refine (add a level, refit, re-select λ) until the exact next-level
/// gain bound certifies that one more level cannot move the penalized
/// objective by more than [`REFINE_TOL`] of the penalized residual — or the
/// net/cap is exhausted. Returns the certified fit.
pub fn fit_residual_cascade(
    xs: &[&[f64]],
    y: &[f64],
    w: &[f64],
    metric: &[f64],
    sobolev_s: f64,
) -> Result<ResidualCascadeFit, String> {
    let mut levels = INITIAL_LEVELS;
    loop {
        let design = ResidualCascadeDesign::build(xs, y, w, metric, sobolev_s, levels)?;
        // Quasi-uniformity guard (issue #1032, caveat 2): if the metric has
        // collapsed the cloud onto a near-degenerate sheet in scaled
        // coordinates, the BPX iteration bound no longer holds. Refuse the
        // iterative solve up front with a typed signal so the auto-route falls
        // back to the dense kernel BEFORE paying an unbounded CG, rather than
        // grinding to CG_MAX_ITERS. (The guard is checked at the root level
        // only — refinement adds finer nets to the SAME scaled cloud, so the
        // aspect ratio is invariant under added levels.)
        if levels == INITIAL_LEVELS && !design.quasi_uniformity_certified() {
            return Err(format!(
                "residual cascade: metric-scaled aspect ratio {:.3e} exceeds the \
                 quasi-uniformity ceiling {QUASI_UNIFORMITY_MAX_ASPECT:.0e}; the BPX \
                 iteration bound is not trustworthy on this (near-degenerate) metric — \
                 fall back to the dense kernel path",
                design.metric_scaled_aspect_ratio()
            ));
        }
        let mut fit = design.fit_reml()?;
        // The realized CG iteration count at this cascade depth is the runtime
        // tell of the BPX n-independence bound (issue #1032 caveat: a count
        // creeping toward CG_MAX_ITERS means the quasi-uniformity guard's static
        // aspect-ratio check was too lenient for this cloud). It is exposed
        // STRUCTURALLY rather than over stderr: the per-depth count and backward
        // error ride on `fit.certificate` (`solve_iters` — 0 on the dense route,
        // the PCG count on the iterative route — and `solve_rel_residual`), so a
        // caller that wants to watch the bound reads them off the returned fit
        // instead of scraping log lines. (A library solve never writes to
        // stderr.)
        let gain = design.next_level_gain_bound(&fit)?;
        let tolerance = REFINE_TOL * fit.rss_pen;
        match gain {
            None => {
                fit.refinement = Some(RefinementCertificate {
                    next_level_gain_bound: 0.0,
                    tolerance,
                    exhausted: true,
                });
                return Ok(fit);
            }
            Some(bound) if bound <= tolerance || levels >= MAX_LEVELS => {
                fit.refinement = Some(RefinementCertificate {
                    next_level_gain_bound: bound,
                    tolerance,
                    exhausted: bound > tolerance,
                });
                return Ok(fit);
            }
            Some(_) => {
                levels += 1;
            }
        }
    }
}

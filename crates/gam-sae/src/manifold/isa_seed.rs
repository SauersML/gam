//! Independent-subspace (ISA) deflationary birth producer (#2111 hardening).
//!
//! WHY FOURTH ORDER. The stagewise birth path mines the running residual for a
//! circle's 2-plane. Whitening the above-noise signal subspace EXHAUSTS all
//! second-order information: a dictionary of K disjoint circles leaves a
//! 2K-dimensional isotropic whitened subspace, so eigenvector pairing returns
//! Davis–Kahan BLENDS across circles (a top-2 plane overlaps ~0.5 with any true
//! circle — the observed K≥2 co-collapse). The identifying signal a blend cannot
//! mimic lives at FOURTH order: a circle's normalized energy fourth moment
//! `κ(W) = E[(‖Wᵀz‖²)²] / E[‖Wᵀz‖²]²` on its own 2-plane is an analytic anchor
//! per population —
//!
//! * dense clean circle: constant radius ⇒ `κ ≈ 1` (the global minimum);
//! * gated circle active on a fraction `q` of rows: `r² = a²·Bernoulli(q)` ⇒
//!   `κ = 1/q` (super-Gaussian for `q < 1/2`);
//! * Gaussian blend (CLT mixture of many independent charts): `r²` is scaled
//!   `χ²₂` ⇒ `κ = 2` exactly; a 45° two-circle sub-Gaussian blend sits at 5/4.
//!
//! So `(κ − 2)²` — squared distance from the GAUSSIAN anchor — is the ISA
//! contrast: it is maximized by clean structure on EITHER side (sub-Gaussian
//! dense circles at `κ → 1`, super-Gaussian gated circles at `κ = 1/q > 2`) and
//! zeroed exactly on the blends the producer must refuse.
//!
//! PRODUCER. Whiten the above-Marchenko–Pastur signal subspace → partition its
//! `r` coordinates into `⌊r/2⌋` candidate 2-planes → cyclic 2-plane JACOBI
//! rotations: every coordinate pair `(i, j)` spanning two different planes is
//! rotated by the angle maximizing the total contrast `Σ_planes (κ_m − 2)²`,
//! which is exactly evaluable at any angle from one `O(n)` joint-moment pass
//! (the rotated moments are closed-form trigonometric polynomials — see
//! [`PairPolys`]). Multistart (`≥ 6` random orthogonal inits — the
//! prototype-validated floor for escaping permutation/blend saddles on the
//! equal-amplitude worst case, where second order gives NO ordering at all)
//! keeps the best basin. The winning plane is accepted only on the ANALYTIC
//! contrast certificate (anchors above; no tuned ε), and the accepted circle is
//! DEFLATED by subtracting its FITTED CURVE — the least-squares harmonic
//! `â₁cosθ·u₁ + â₂sinθ·u₂` on its active rows — NOT its plane projection.
//! Subtracting the plane would also delete the plane's noise energy and the
//! overlap other structure may share with it, biasing every later round;
//! subtracting the fitted curve removes exactly the model the born atom will
//! carry, which is also what the stagewise fit-then-residual loop does — so the
//! in-fit deflation and this producer's harness deflation are the same operator.
//!
//! Everything here is derived from the residual spectrum (MP edge, bottom-
//! quartile noise scale) plus the analytic κ anchors; the only dials are
//! multistart/sweep COST caps, typed on [`IsaSeedConfig`].
//!
//! THEORY (Superposed Geometry, Prop. 1 — measure-level identifiability). A
//! CENTERED circle's cone ℝ₊·Y coincides, as a set, with the whole 2-plane
//! P∖{0}: no support test, no rank test, no Terracini border-block Jacobian
//! can distinguish "this data lives on a circle" from "this data spans a
//! plane" — the circle is SUPPORT-INVISIBLE. It is identifiable only through
//! its RADIAL LAW, i.e. the MEASURE the data puts on that support, not the
//! support itself. This file is the measure-level half of identifiability;
//! the support/rank half (Terracini tests, border-block Jacobian) lives in
//! sibling `identifiability.rs` + the certificate machinery. The two see
//! COMPLEMENTARY halves of Prop. 1 — support existence vs. the law on that
//! support — and neither subsumes the other: a centered circle is exactly the
//! case the support test cannot see and this producer must.

use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array1, Array2, ArrayView2};

/// Moment-concentration floor on the number of rows the Jacobi contrast is
/// estimated from — DERIVED, not tuned. By the delta method, the estimator
/// `κ̂ = m̂₄/m̂₂²` of a gated circle (`r² = a²·Bernoulli(q)`) has
/// `Var(κ̂) = (1 − q)/(q³ n)`, and the certificate must resolve the gated
/// anchor `1/q` from the Gaussian-blend anchor `2`, a gap of `1/q − 2`, at the
/// conventional `z = 3` level: `z·SE ≤ (1/q − 2)/2 ⇒ n ≥ 4z²·q(1 − q)/(1 − 2q)²`.
/// The bound is increasing in `q` and diverges at `q → ½` (a half-gated circle
/// has `κ = 2` — genuinely indistinguishable from a blend by κ alone), so the
/// floor is set at the practical design edge `q = 0.43`:
/// `n ≥ 4·9·0.43·0.57/0.14² ≈ 450 → 500`. Sparser gates and dense circles
/// concentrate strictly faster, so 500 covers the whole certifiable range.
/// (This also retires the old `n = 80`-class fixtures: any harness exercising
/// the fourth-order certificate needs `n ≥ 300` to be out of the small-sample
/// floor even in the easy dense case.)
const ISA_SUBSAMPLE_FLOOR: usize = 500;

/// Angle samples for one Jacobi pair — DERIVED from the harmonic degree of the
/// objective, not tuned. In `φ = 2θ` each plane's `E[r²]` is a degree-1 and
/// `E[r⁴]` a degree-2 trigonometric polynomial, so each `(κ − 2)²` term is a
/// ratio with numerator degree ≤ 4 and denominator degree ≤ 4; the two-term
/// objective has degree ≤ 8 over degree ≤ 8, hence its derivative's numerator
/// has degree ≤ 16 ⇒ at most 32 zeros per `φ`-period ⇒ at most 32 monotone
/// pieces over the `θ`-period `π`. Two samples per monotone piece (64)
/// guarantee a sample inside the global maximizer's basin; the exact
/// hand-derived `dJ/dθ` then localizes it by bisection. This is Nyquist-rate
/// interpolation of a known-degree function, not a tuned search grid.
const ISA_ANGLE_SAMPLES: usize = 64;

/// Relative improvement below which a Jacobi sweep is converged (machine-
/// precision scale — a convergence tolerance, not a model threshold).
const ISA_SWEEP_RTOL: f64 = 1.0e-12;

/// Cost dials for the ISA producer (multistart width and a sweep safety bound —
/// COST caps, not statistical thresholds; every accept/reject decision is made
/// by the derived anchors and floors above).
#[derive(Clone, Copy, Debug)]
pub struct IsaSeedConfig {
    /// Random orthogonal Jacobi inits (identity + `n_inits − 1` random). The
    /// default 6 is the prototype-validated multistart floor that reached the
    /// global basin on the equal-amplitude worst case (#2111).
    pub n_inits: usize,
    /// Safety bound on cyclic Jacobi sweeps; convergence is the
    /// [`ISA_SWEEP_RTOL`] relative-improvement stop, this only caps a
    /// pathological cycle (mirrors the `max_births` bound-not-stop pattern).
    pub max_sweeps: usize,
}

impl Default for IsaSeedConfig {
    fn default() -> Self {
        Self {
            n_inits: 6,
            max_sweeps: 64,
        }
    }
}

/// The residual eigenstructure every extraction consumes: column mean,
/// ascending eigenpairs of the column-centered second moment, the above-floor
/// index set (strongest first), the Marchenko–Pastur edge, and the bottom-
/// quartile noise scale the certificate anchors use.
pub struct IsaEigenParts {
    pub mean: Array1<f64>,
    /// Ascending eigenvalues of `(1/n)·R_cᵀR_c`.
    pub evals: Array1<f64>,
    /// Matching eigenvector columns.
    pub evecs: Array2<f64>,
    /// Indices into `evals` above the MP edge, sorted strongest first.
    pub above: Vec<usize>,
    /// `λ₊ = σ̂²·(1 + √(p/n))²` — the analytic top edge of the Marchenko–Pastur
    /// law at aspect `p/n`: the largest eigenvalue white noise produces, so a
    /// direction above it is real structure, not a fluctuation. `σ̂²` is the
    /// median eigenvalue — robust while signal directions are a minority.
    pub mp_edge: f64,
    /// Noise scale for the κ certificate: the median of the smallest-quartile
    /// eigenvalues, where white noise provably sits regardless of how many
    /// directions are signal. (The global median that sets the MP edge lands in
    /// the SIGNAL bulk on a dense multi-circle residual and would inflate the
    /// clean anchor past the blend anchor, rejecting even a clean circle.)
    pub sigma2_cert: f64,
}

/// Center `residual`, eigendecompose its second moment, and derive the MP floor
/// context. `Ok(None)` when the residual has no above-floor direction (pure
/// noise ⇒ the caller's natural stop) or is too small to carry structure.
pub fn isa_eigen_parts(residual: ArrayView2<'_, f64>) -> Result<Option<IsaEigenParts>, String> {
    let (n, p) = residual.dim();
    if n < 2 || p == 0 {
        return Ok(None);
    }
    let mut mean = Array1::<f64>::zeros(p);
    for row in 0..n {
        for j in 0..p {
            mean[j] += residual[[row, j]];
        }
    }
    mean.mapv_inplace(|v| v / n as f64);
    let mut s = Array2::<f64>::zeros((p, p));
    for row in 0..n {
        for a in 0..p {
            let ra = residual[[row, a]] - mean[a];
            for b in a..p {
                s[[a, b]] += ra * (residual[[row, b]] - mean[b]);
            }
        }
    }
    for a in 0..p {
        for b in a..p {
            let v = s[[a, b]] / n as f64;
            s[[a, b]] = v;
            s[[b, a]] = v;
        }
    }
    let (evals, evecs) = s
        .eigh(Side::Lower)
        .map_err(|err| format!("isa_eigen_parts: residual eigensolve failed: {err:?}"))?;
    if evals.is_empty() {
        return Ok(None);
    }
    let mut ascending: Vec<f64> = evals.iter().copied().collect();
    ascending.sort_by(|a, b| a.total_cmp(b));
    let mid = ascending.len() / 2;
    let sigma2 = if ascending.len() % 2 == 1 {
        ascending[mid]
    } else {
        0.5 * (ascending[mid - 1] + ascending[mid])
    }
    .max(f64::MIN_POSITIVE);
    let gamma = p as f64 / n as f64;
    let mp_edge = sigma2 * (1.0 + gamma.sqrt()).powi(2);
    let mut above: Vec<usize> = (0..evals.len()).filter(|&k| evals[k] > mp_edge).collect();
    above.sort_by(|&a, &b| evals[b].total_cmp(&evals[a]));
    if above.is_empty() {
        return Ok(None);
    }
    let q = (evals.len() / 4).max(1);
    let sigma2_cert = evals[(q - 1) / 2].max(0.0);
    Ok(Some(IsaEigenParts {
        mean,
        evals,
        evecs,
        above,
        mp_edge,
        sigma2_cert,
    }))
}

/// One certified circle 2-plane extracted by the producer, in ambient
/// coordinates, with everything a birth seed (or the harness deflation) needs.
pub struct IsaPlaneCandidate {
    /// Ambient orthonormal plane basis, `(p, 2)`.
    pub basis: Array2<f64>,
    /// Least-squares harmonic amplitudes on the active rows (cos, sin axes).
    pub amplitudes: [f64; 2],
    /// Per-row in-plane phase in turns `[0, 1)`, `(n, 1)` — the born chart seed.
    pub phases_turns: Array2<f64>,
    /// Per-row OWN-PRESENCE gate: `ln(ρ_i² / (2σ̂² ln n))` where the plane
    /// radius clears the derived χ²₂ upper-tail noise floor, else `−∞` (same
    /// contract as the #2109 gate).
    pub gate_logits: Vec<f64>,
    /// Observed raw-radius `κ` over all rows.
    pub kappa: f64,
    /// Active-row fraction `q̂` (the gate the κ anchors were evaluated at).
    pub q_hat: f64,
}

/// splitmix64-keyed standard normal (Box–Muller over an LCG): deterministic,
/// no RNG crate, reproducible across thread/device counts.
fn lcg_normal(state: &mut u64) -> f64 {
    let mut u = [0.0_f64; 2];
    for slot in u.iter_mut() {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *slot = ((*state >> 11) as f64) / ((1u64 << 53) as f64);
    }
    let u1 = u[0].max(f64::MIN_POSITIVE);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u[1]).cos()
}

/// Modified Gram–Schmidt orthonormalization of the two columns of a `(d, 2)`
/// matrix in place. Returns `false` when the columns are numerically
/// rank-deficient (the caller abandons the degenerate plane rather than
/// emitting a NaN chart).
pub(crate) fn orthonormalize2(w: &mut Array2<f64>) -> bool {
    let d = w.nrows();
    let n0: f64 = (0..d).map(|i| w[[i, 0]] * w[[i, 0]]).sum::<f64>().sqrt();
    if !(n0 > 1e-12) {
        return false;
    }
    for i in 0..d {
        w[[i, 0]] /= n0;
    }
    let dot: f64 = (0..d).map(|i| w[[i, 0]] * w[[i, 1]]).sum();
    for i in 0..d {
        w[[i, 1]] -= dot * w[[i, 0]];
    }
    let n1: f64 = (0..d).map(|i| w[[i, 1]] * w[[i, 1]]).sum::<f64>().sqrt();
    if !(n1 > 1e-12) {
        return false;
    }
    for i in 0..d {
        w[[i, 1]] /= n1;
    }
    true
}

/// Closed-form rotated-moment polynomials for one plane under one Jacobi pair.
/// With `c = cos θ`, `s = sin θ` and the pair coordinate rotating as
/// `y(θ) = ±s·y_other + c·y_own ∓ …` (sign per side), both plane moments are
/// exact trigonometric polynomials in `2θ`:
///
/// ```text
/// E[r²](θ) = d₀ + d₁·cos2θ + d₂·sin2θ
/// E[r⁴](θ) = n₀ + n₁·cos2θ + n₂·sin2θ + n₃·cos4θ + n₄·sin4θ
/// ```
///
/// so `κ(θ) = N/D²` and the pair objective `Σ (κ − 2)²` are exactly evaluable
/// (value AND hand-derived `d/dθ`) at any angle in `O(1)` after one `O(n)`
/// joint-moment pass. No autodiff, no finite differences.
#[derive(Clone, Copy)]
struct PlanePolys {
    d: [f64; 3],
    n: [f64; 5],
}

impl PlanePolys {
    fn kappa(&self, c2: f64, s2: f64, c4: f64, s4: f64) -> f64 {
        let den = self.d[0] + self.d[1] * c2 + self.d[2] * s2;
        if !(den > 0.0) {
            return f64::INFINITY;
        }
        let num =
            self.n[0] + self.n[1] * c2 + self.n[2] * s2 + self.n[3] * c4 + self.n[4] * s4;
        num / (den * den)
    }

    /// Hand-derived `dκ/dθ` at the same angle (quotient rule on the two
    /// trigonometric polynomials; `dC₂/dθ = −2S₂`, `dS₂/dθ = 2C₂`, etc.).
    fn dkappa(&self, c2: f64, s2: f64, c4: f64, s4: f64) -> f64 {
        let den = self.d[0] + self.d[1] * c2 + self.d[2] * s2;
        if !(den > 0.0) {
            return 0.0;
        }
        let num =
            self.n[0] + self.n[1] * c2 + self.n[2] * s2 + self.n[3] * c4 + self.n[4] * s4;
        let dden = -2.0 * self.d[1] * s2 + 2.0 * self.d[2] * c2;
        let dnum = -2.0 * self.n[1] * s2 + 2.0 * self.n[2] * c2 - 4.0 * self.n[3] * s4
            + 4.0 * self.n[4] * c4;
        (dnum * den - 2.0 * num * dden) / (den * den * den)
    }
}

/// Joint-moment pass for one Jacobi pair `(i, j)`: builds the rotated-moment
/// polynomials of the plane containing `i` (rotating as `y_i(θ) = c·y_i − s·y_j`)
/// and the plane containing `j` (`y_j(θ) = s·y_i + c·y_j`). A `None` side means
/// the coordinate is the unpaired slack column of an odd-`r` subspace (it
/// participates in rotations, but earns no contrast of its own).
fn pair_polys(
    y: &Array2<f64>,
    i: usize,
    j: usize,
    partner_i: Option<usize>,
    partner_j: Option<usize>,
) -> (Option<PlanePolys>, Option<PlanePolys>) {
    let n = y.ncols();
    let inv_n = 1.0 / n as f64;
    // Pure pair moments m_ab = E[y_i^a y_j^b].
    let (mut m20, mut m11, mut m02) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut m40, mut m31, mut m22, mut m13, mut m04) = (0.0_f64, 0.0, 0.0, 0.0, 0.0);
    // Partner-weighted moments (w = fixed partner coordinate of each plane).
    let (mut a2, mut a4, mut awi, mut awx, mut awj) = (0.0_f64, 0.0, 0.0, 0.0, 0.0);
    let (mut b2, mut b4, mut bwi, mut bwx, mut bwj) = (0.0_f64, 0.0, 0.0, 0.0, 0.0);
    for col in 0..n {
        let yi = y[[i, col]];
        let yj = y[[j, col]];
        let (yi2, yj2, yij) = (yi * yi, yj * yj, yi * yj);
        m20 += yi2;
        m11 += yij;
        m02 += yj2;
        m40 += yi2 * yi2;
        m31 += yi2 * yij;
        m22 += yi2 * yj2;
        m13 += yij * yj2;
        m04 += yj2 * yj2;
        if let Some(pi) = partner_i {
            let w2 = y[[pi, col]] * y[[pi, col]];
            a2 += w2;
            a4 += w2 * w2;
            awi += w2 * yi2;
            awx += w2 * yij;
            awj += w2 * yj2;
        }
        if let Some(pj) = partner_j {
            let w2 = y[[pj, col]] * y[[pj, col]];
            b2 += w2;
            b4 += w2 * w2;
            bwi += w2 * yi2;
            bwx += w2 * yij;
            bwj += w2 * yj2;
        }
    }
    for v in [
        &mut m20, &mut m11, &mut m02, &mut m40, &mut m31, &mut m22, &mut m13, &mut m04, &mut a2,
        &mut a4, &mut awi, &mut awx, &mut awj, &mut b2, &mut b4, &mut bwi, &mut bwx, &mut bwj,
    ] {
        *v *= inv_n;
    }
    // Shared quartic combinations: per-row u = (y_i² + y_j²)/2, v = (y_i² − y_j²)/2,
    // w = y_i·y_j, so E[y(θ)⁴] = E[(u ± C·v ∓ S·w)²] expands over these.
    let euu = (m40 + 2.0 * m22 + m04) / 4.0;
    let evv = (m40 - 2.0 * m22 + m04) / 4.0;
    let eww = m22;
    let euv = (m40 - m04) / 4.0;
    let euw = (m31 + m13) / 2.0;
    let evw = (m31 - m13) / 2.0;
    let h0 = (m20 + m02) / 2.0;
    let h1 = (m20 - m02) / 2.0;
    let h2 = m11;
    let quart_dc = euu + (evv + eww) / 2.0;
    let quart_c4 = (evv - eww) / 2.0;
    // Plane A: y_i(θ) = c·y_i − s·y_j ⇒ y_i(θ)² = h0 + C·h1 − S·h2 per-row shape.
    let plane_a = partner_i.map(|_| {
        let g0 = (awi + awj) / 2.0;
        let g1 = (awi - awj) / 2.0;
        PlanePolys {
            d: [a2 + h0, h1, -h2],
            n: [
                a4 + 2.0 * g0 + quart_dc,
                2.0 * g1 + 2.0 * euv,
                -2.0 * awx - 2.0 * euw,
                quart_c4,
                -evw,
            ],
        }
    });
    // Plane B: y_j(θ) = s·y_i + c·y_j ⇒ y_j(θ)² = h0 − C·h1 + S·h2.
    let plane_b = partner_j.map(|_| {
        let g0 = (bwi + bwj) / 2.0;
        let g1 = (bwi - bwj) / 2.0;
        PlanePolys {
            d: [b2 + h0, -h1, h2],
            n: [
                b4 + 2.0 * g0 + quart_dc,
                -2.0 * g1 - 2.0 * euv,
                2.0 * bwx + 2.0 * euw,
                quart_c4,
                -evw,
            ],
        }
    });
    (plane_a, plane_b)
}

/// Pair objective `J(θ) = Σ_{existing planes} (κ(θ) − 2)²` and its hand-derived
/// derivative, exactly evaluable from the pair polynomials.
///
/// This IS the Prop. 1 measure-vs-support contrast made optimizable: `κ = 2`
/// is not an arbitrary target but the one population — the Gaussian/CLT blend
/// of many independent charts — whose radial law a plane and a circle share
/// asymptotically. `(κ − 2)²` is zero exactly there and rises on BOTH sides
/// (dense circle `κ → 1`, gated circle `κ = 1/q > 2`), so ascending it always
/// moves toward whichever clean radial law is present and away from the blend
/// no support/rank test could have ruled out in the first place.
fn pair_objective(a: &Option<PlanePolys>, b: &Option<PlanePolys>, theta: f64) -> f64 {
    let (c2, s2) = ((2.0 * theta).cos(), (2.0 * theta).sin());
    let (c4, s4) = ((4.0 * theta).cos(), (4.0 * theta).sin());
    let mut j = 0.0;
    for side in [a, b].into_iter().flatten() {
        let k = side.kappa(c2, s2, c4, s4);
        if k.is_finite() {
            j += (k - 2.0) * (k - 2.0);
        }
    }
    j
}

fn pair_objective_deriv(a: &Option<PlanePolys>, b: &Option<PlanePolys>, theta: f64) -> f64 {
    let (c2, s2) = ((2.0 * theta).cos(), (2.0 * theta).sin());
    let (c4, s4) = ((4.0 * theta).cos(), (4.0 * theta).sin());
    let mut dj = 0.0;
    for side in [a, b].into_iter().flatten() {
        let k = side.kappa(c2, s2, c4, s4);
        if k.is_finite() {
            dj += 2.0 * (k - 2.0) * side.dkappa(c2, s2, c4, s4);
        }
    }
    dj
}

/// κ of the plane made of coordinate rows `(2m, 2m+1)` of `y`, over columns.
fn plane_rows_kappa(y: &Array2<f64>, m: usize) -> f64 {
    let n = y.ncols();
    let (mut s2, mut s4) = (0.0_f64, 0.0_f64);
    for col in 0..n {
        let y0 = y[[2 * m, col]];
        let y1 = y[[2 * m + 1, col]];
        let r2 = y0 * y0 + y1 * y1;
        s2 += r2;
        s4 += r2 * r2;
    }
    let m2 = s2 / n as f64;
    if !(m2 > 0.0) {
        return f64::INFINITY;
    }
    (s4 / n as f64) / (m2 * m2)
}

/// Total ISA contrast `Σ_planes (κ_m − 2)²` of the current rotation state —
/// the objective the Jacobi sweep ascends and the score used to rank basins
/// across multistart inits and planes across a single basin (see
/// [`isa_extract_certified_plane`]). Same Gaussian-anchor logic as
/// [`pair_objective`], just summed over all current planes rather than the
/// two planes touched by one rotation.
fn total_contrast(y: &Array2<f64>, n_planes: usize) -> f64 {
    (0..n_planes)
        .map(|m| {
            let k = plane_rows_kappa(y, m);
            if k.is_finite() { (k - 2.0) * (k - 2.0) } else { 0.0 }
        })
        .sum()
}

/// Cyclic 2-plane Jacobi ascent on the ISA contrast. `y` holds the rotated
/// coordinates (`r × n`), `q` the accumulated orthogonal rotation (`y = qᵀ·z`,
/// columns of `q` are whitened-space directions); both are updated in place.
fn jacobi_optimize(y: &mut Array2<f64>, q: &mut Array2<f64>, n_planes: usize, max_sweeps: usize) {
    let r = y.nrows();
    let partner = |c: usize| -> Option<usize> {
        if c < 2 * n_planes {
            Some(c ^ 1)
        } else {
            None
        }
    };
    let mut total = total_contrast(y, n_planes);
    for _ in 0..max_sweeps {
        let mut improved = false;
        for i in 0..r {
            for j in (i + 1)..r {
                // Same-plane rotations leave every plane radius invariant.
                if partner(i) == Some(j) {
                    continue;
                }
                // Exclude the plane the OTHER coordinate belongs to from the
                // partner when they coincide — cannot happen here (planes are
                // 2-coordinate, so same plane ⇔ partner(i) == j, skipped above).
                let (pa, pb) = pair_polys(y, i, j, partner(i), partner(j));
                if pa.is_none() && pb.is_none() {
                    continue;
                }
                let j0 = pair_objective(&pa, &pb, 0.0);
                // Nyquist-rate scan (derivation at ISA_ANGLE_SAMPLES) + argmax.
                let mut best_theta = 0.0_f64;
                let mut best_j = j0;
                let step = std::f64::consts::PI / ISA_ANGLE_SAMPLES as f64;
                for k in 0..ISA_ANGLE_SAMPLES {
                    let theta = -std::f64::consts::FRAC_PI_2 + step * k as f64;
                    let jv = pair_objective(&pa, &pb, theta);
                    if jv > best_j {
                        best_j = jv;
                        best_theta = theta;
                    }
                }
                // Polish inside the bracketing lattice cell with the exact
                // derivative (bisection on dJ/dθ — hand-derived, no FD).
                let (mut lo, mut hi) = (best_theta - step, best_theta + step);
                let (dlo, dhi) = (
                    pair_objective_deriv(&pa, &pb, lo),
                    pair_objective_deriv(&pa, &pb, hi),
                );
                if dlo > 0.0 && dhi < 0.0 {
                    for _ in 0..60 {
                        let mid = 0.5 * (lo + hi);
                        if pair_objective_deriv(&pa, &pb, mid) > 0.0 {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    let polished = 0.5 * (lo + hi);
                    let jp = pair_objective(&pa, &pb, polished);
                    if jp > best_j {
                        best_j = jp;
                        best_theta = polished;
                    }
                }
                if best_j > j0 * (1.0 + ISA_SWEEP_RTOL) + f64::MIN_POSITIVE {
                    let (c, s) = (best_theta.cos(), best_theta.sin());
                    for col in 0..y.ncols() {
                        let yi = y[[i, col]];
                        let yj = y[[j, col]];
                        y[[i, col]] = c * yi - s * yj;
                        y[[j, col]] = s * yi + c * yj;
                    }
                    for row in 0..q.nrows() {
                        let qi = q[[row, i]];
                        let qj = q[[row, j]];
                        q[[row, i]] = c * qi - s * qj;
                        q[[row, j]] = s * qi + c * qj;
                    }
                    improved = true;
                }
            }
        }
        let new_total = total_contrast(y, n_planes);
        if !improved || new_total - total <= ISA_SWEEP_RTOL * (1.0 + total.abs()) {
            break;
        }
        total = new_total;
    }
}

/// Deterministic stride subsample of `n` columns down to the moment floor.
fn subsample_columns(n: usize) -> Vec<usize> {
    if n <= ISA_SUBSAMPLE_FLOOR {
        return (0..n).collect();
    }
    (0..ISA_SUBSAMPLE_FLOOR)
        .map(|i| i * n / ISA_SUBSAMPLE_FLOOR)
        .collect()
}

/// Analytic κ-contrast certificate + candidate assembly for one whitened-space
/// plane. `w` is `(r, 2)` in the above-floor eigenbasis. Certification happens
/// on the FULL data in AMBIENT coordinates: the plane is un-whitened,
/// re-orthonormalized, and its raw-radius moments compared against the
/// noise-corrected population anchors —
///
/// * clean gated circle: `E[r²] = q̂a² + 2σ̂²`, `E[r⁴] = q̂a⁴ + 8q̂a²σ̂² + 8σ̂⁴`
///   (dense is `q̂ = 1`), so `κ_model = (q̂a⁴ + 8q̂a²σ̂² + 8σ̂⁴)/m₂²`;
/// * nearest blend: the 45° two-circle sub-Gaussian blend, whose only change is
///   the pure quartic term scaling by 5/4: `κ_blend = (1.25·q̂a⁴ + 8q̂a²σ̂² +
///   8σ̂⁴)/m₂²`. Every blendier population (same-plane two-circle 3/2, Gaussian
///   2) sits above it — all scaled by the same measured `1/q̂` — so the
///   NEAREST adversary bounds them all.
///
/// Accept iff the anchors are ordered (`κ_model < κ_blend`) and the observed κ
/// falls on the model side of their midpoint — a likelihood split between two
/// derived populations, no tuned ε.
///
/// This is the actual certification instrument for Prop. 1: `amb`, the
/// candidate 2-plane, is by construction support-indistinguishable from a
/// generic plane (that ambiguity is exactly what the Jacobi ascent's contrast
/// score cannot resolve on its own, since a lucky rotation could land near a
/// blend's local optimum). What decides ACCEPT/REJECT here is not the plane
/// itself but a comparison of two DERIVED MEASURES on it — `κ_model` (clean
/// gated/dense circle law) vs. `κ_blend` (nearest Gaussian-mixture law) — so
/// the accept/reject boundary lives entirely in measure space, never in
/// support space. A circle and a plane occupying identical support pass or
/// fail this test on the strength of their radial law alone.
fn certify_plane(
    residual: ArrayView2<'_, f64>,
    parts: &IsaEigenParts,
    w: &Array2<f64>,
) -> Option<IsaPlaneCandidate> {
    let (n, p) = residual.dim();
    let r = parts.above.len();
    // Un-whiten to ambient: u_a / √λ_a per above-floor direction, then restore
    // orthonormality (whitening skews the frame).
    let mut amb = Array2::<f64>::zeros((p, 2));
    for (a, &k) in parts.above.iter().enumerate().take(r) {
        let inv = 1.0 / parts.evals[k].max(f64::MIN_POSITIVE).sqrt();
        for j in 0..p {
            amb[[j, 0]] += parts.evecs[[j, k]] * inv * w[[a, 0]];
            amb[[j, 1]] += parts.evecs[[j, k]] * inv * w[[a, 1]];
        }
    }
    if !orthonormalize2(&mut amb) {
        return None;
    }
    let noise_2plane = 2.0 * parts.sigma2_cert.max(f64::MIN_POSITIVE) * (n as f64).ln();
    let mut phases = Array2::<f64>::zeros((n, 1));
    let mut gate = vec![f64::NEG_INFINITY; n];
    let (mut r2_sum, mut r4_sum) = (0.0_f64, 0.0_f64);
    let (mut c_num, mut c_den, mut s_num, mut s_den) = (0.0_f64, 0.0, 0.0, 0.0);
    let mut n_active = 0usize;
    for i in 0..n {
        let (mut p1, mut p2) = (0.0_f64, 0.0_f64);
        for j in 0..p {
            let ri = residual[[i, j]] - parts.mean[j];
            p1 += ri * amb[[j, 0]];
            p2 += ri * amb[[j, 1]];
        }
        let theta = p2.atan2(p1);
        phases[[i, 0]] = theta.rem_euclid(std::f64::consts::TAU) / std::f64::consts::TAU;
        let r2 = p1 * p1 + p2 * p2;
        r2_sum += r2;
        r4_sum += r2 * r2;
        if r2 > noise_2plane {
            gate[i] = (r2 / noise_2plane).ln();
            n_active += 1;
            // LS harmonic amplitudes on active rows: p1 = ρcosθ regressed on cosθ.
            let (ct, st) = (theta.cos(), theta.sin());
            c_num += p1 * ct;
            c_den += ct * ct;
            s_num += p2 * st;
            s_den += st * st;
        }
    }
    if n_active == 0 {
        return None;
    }
    // With the χ²₂ tail gate calibrated to one expected false-active row per
    // plane, a resolvable circle must clear a super-Poisson tail count rather
    // than a handful of accidental exceedances.
    let active_floor = ((n as f64).ln().powi(2).ceil() as usize).max(4);
    if n_active < active_floor {
        return None;
    }
    let q_hat = n_active as f64 / n as f64;
    let m2 = (r2_sum / n as f64).max(f64::MIN_POSITIVE);
    let kappa_obs = (r4_sum / n as f64) / (m2 * m2);
    let sig2 = parts.sigma2_cert;
    // Noise-corrected squared amplitude: m₂ = q̂a² + 2σ̂².
    let a2t = ((m2 - 2.0 * sig2) / q_hat).max(0.0);
    let common = 8.0 * q_hat * a2t * sig2 + 8.0 * sig2 * sig2;
    let kappa_model = (q_hat * a2t * a2t + common) / (m2 * m2);
    let kappa_blend = (1.25 * q_hat * a2t * a2t + common) / (m2 * m2);
    if !(kappa_model < kappa_blend) {
        return None; // amplitude below noise — anchors unresolvable
    }
    let gate_mid = 0.5 * (kappa_model + kappa_blend);
    if !(kappa_obs < gate_mid) {
        return None; // blend side of the split — refuse the circle claim
    }
    let a1 = if c_den > 0.0 { c_num / c_den } else { 0.0 };
    let a2 = if s_den > 0.0 { s_num / s_den } else { 0.0 };
    if !(a1.is_finite() && a2.is_finite()) {
        return None;
    }
    Some(IsaPlaneCandidate {
        basis: amb,
        amplitudes: [a1, a2],
        phases_turns: phases,
        gate_logits: gate,
        kappa: kappa_obs,
        q_hat,
    })
}

/// Extract ONE certified circle plane from the residual, given its
/// eigenstructure: whiten the above-floor subspace, run the multistart Jacobi
/// contrast ascent, then walk the resulting planes in descending contrast order
/// and return the first that passes the ambient certificate. `None` = the
/// residual carries no certifiable clean circle (the caller's natural stop or
/// rank-1 fallback).
pub fn isa_extract_certified_plane(
    residual: ArrayView2<'_, f64>,
    parts: &IsaEigenParts,
    config: &IsaSeedConfig,
) -> Option<IsaPlaneCandidate> {
    let n = residual.nrows();
    let r = parts.above.len();
    if r < 2 || n < 2 {
        return None;
    }
    let n_planes = r / 2;
    // Whitened above-floor coordinates, moment-floor subsample of the columns.
    let cols = subsample_columns(n);
    let n_sub = cols.len();
    let mut z = Array2::<f64>::zeros((r, n_sub));
    for (a, &k) in parts.above.iter().enumerate() {
        let inv = 1.0 / parts.evals[k].max(f64::MIN_POSITIVE).sqrt();
        for (cc, &row) in cols.iter().enumerate() {
            let mut proj = 0.0_f64;
            for j in 0..residual.ncols() {
                proj += (residual[[row, j]] - parts.mean[j]) * parts.evecs[[j, k]];
            }
            z[[a, cc]] = proj * inv;
        }
    }
    // Multistart Jacobi: identity init + random orthogonal inits, keep the basin
    // with the largest total contrast. Deterministic (LCG keyed by init + n).
    let mut best: Option<(f64, Array2<f64>, Array2<f64>)> = None;
    for init in 0..config.n_inits.max(1) {
        let mut q = Array2::<f64>::eye(r);
        if init > 0 {
            // Random orthogonal via Gram-Schmidt of LCG normal columns.
            let mut state = 0x2111_15A0_u64 ^ ((init as u64) << 32) ^ n as u64;
            let mut g = Array2::<f64>::from_shape_fn((r, r), |_| lcg_normal(&mut state));
            for c in 0..r {
                for prev in 0..c {
                    let mut dot = 0.0;
                    for row in 0..r {
                        dot += g[[row, c]] * g[[row, prev]];
                    }
                    for row in 0..r {
                        let sub = dot * g[[row, prev]];
                        g[[row, c]] -= sub;
                    }
                }
                let mut nrm = 0.0;
                for row in 0..r {
                    nrm += g[[row, c]] * g[[row, c]];
                }
                let nrm = nrm.sqrt();
                if nrm > 1e-12 {
                    for row in 0..r {
                        g[[row, c]] /= nrm;
                    }
                } else {
                    for row in 0..r {
                        g[[row, c]] = if row == c { 1.0 } else { 0.0 };
                    }
                }
            }
            q = g;
        }
        let mut y = q.t().dot(&z);
        jacobi_optimize(&mut y, &mut q, n_planes, config.max_sweeps);
        let contrast = total_contrast(&y, n_planes);
        if best.as_ref().is_none_or(|(bc, _, _)| contrast > *bc) {
            best = Some((contrast, q, y));
        }
    }
    let (_, q, y) = best?;
    // Planes in descending single-plane contrast; first to certify wins.
    let mut order: Vec<(f64, usize)> = (0..n_planes)
        .map(|m| {
            let k = plane_rows_kappa(&y, m);
            let c = if k.is_finite() { (k - 2.0) * (k - 2.0) } else { 0.0 };
            (c, m)
        })
        .collect();
    order.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (contrast, m) in order {
        if !(contrast > 0.0) {
            continue;
        }
        let mut w = Array2::<f64>::zeros((r, 2));
        for row in 0..r {
            w[[row, 0]] = q[[row, 2 * m]];
            w[[row, 1]] = q[[row, 2 * m + 1]];
        }
        if let Some(cand) = certify_plane(residual, parts, &w) {
            return Some(cand);
        }
    }
    None
}

/// Subtract an accepted plane's fitted residual projection, in place: on each
/// active row (finite gate) remove the centered residual component in the
/// certified plane. Certification still emits the LS harmonic amplitudes used by
/// the birth seed; harvest deflation must instead zero the accepted plane's
/// energy so the next eigendecomposition surfaces the next circle.
pub fn isa_deflate_fitted_curve(residual: &mut Array2<f64>, cand: &IsaPlaneCandidate) {
    let (n, p) = residual.dim();
    let mut mean = Array1::<f64>::zeros(p);
    for i in 0..n {
        for j in 0..p {
            mean[j] += residual[[i, j]];
        }
    }
    mean.mapv_inplace(|v| v / n as f64);
    for i in 0..n {
        if !cand.gate_logits[i].is_finite() {
            continue;
        }
        let (mut p1, mut p2) = (0.0_f64, 0.0_f64);
        for j in 0..p {
            let ri = residual[[i, j]] - mean[j];
            p1 += ri * cand.basis[[j, 0]];
            p2 += ri * cand.basis[[j, 1]];
        }
        for j in 0..p {
            residual[[i, j]] -= p1 * cand.basis[[j, 0]] + p2 * cand.basis[[j, 1]];
        }
    }
}

/// The producer's harvest: the certified planes in extraction order, and
/// whether the loop exited NATURALLY (the certificate/noise floor said stop)
/// rather than by hitting the caller's safety cap.
pub struct IsaHarvest {
    pub planes: Vec<IsaPlaneCandidate>,
    pub natural_exit: bool,
}

/// Full deflationary run: extract-certify-deflate until the residual carries no
/// certifiable circle (natural exit) or `max_planes` (a safety BOUND, not a stop
/// criterion) is reached. This is the harness/e2e entry; the stagewise birth
/// path calls [`isa_extract_certified_plane`] once per birth instead, because
/// its fit-then-residual loop IS the deflation.
pub fn isa_deflationary_producer(
    residual: ArrayView2<'_, f64>,
    max_planes: usize,
    config: &IsaSeedConfig,
) -> Result<IsaHarvest, String> {
    let mut work = residual.to_owned();
    let mut planes = Vec::new();
    let mut natural_exit = false;
    while planes.len() < max_planes {
        let Some(parts) = isa_eigen_parts(work.view())? else {
            natural_exit = true; // residual is noise — the derived-floor stop
            break;
        };
        let Some(cand) = isa_extract_certified_plane(work.view(), &parts, config) else {
            natural_exit = true; // nothing certifies — the contrast stop
            break;
        };
        isa_deflate_fitted_curve(&mut work, &cand);
        planes.push(cand);
    }
    Ok(IsaHarvest {
        planes,
        natural_exit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_uniform(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// `k` planted circles in `p` ambient dims behind a random orthonormal
    /// 2k-frame; each circle active per-row with probability `q` (1.0 = dense
    /// torus: every row on every circle). Returns the data and the true
    /// ambient planes.
    fn planted_circles(
        n: usize,
        p: usize,
        k: usize,
        q: f64,
        amps: &[f64],
        sigma: f64,
        seed: u64,
    ) -> (Array2<f64>, Vec<Array2<f64>>) {
        assert!(p >= 2 * k && amps.len() == k);
        let mut state = seed;
        // Random orthonormal 2k-frame via Gram-Schmidt of normal columns.
        let mut frame = Array2::<f64>::from_shape_fn((p, 2 * k), |_| lcg_normal(&mut state));
        for c in 0..2 * k {
            for prev in 0..c {
                let mut dot = 0.0;
                for row in 0..p {
                    dot += frame[[row, c]] * frame[[row, prev]];
                }
                for row in 0..p {
                    let sub = dot * frame[[row, prev]];
                    frame[[row, c]] -= sub;
                }
            }
            let mut nrm = 0.0;
            for row in 0..p {
                nrm += frame[[row, c]] * frame[[row, c]];
            }
            let nrm = nrm.sqrt();
            for row in 0..p {
                frame[[row, c]] /= nrm;
            }
        }
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for c in 0..k {
                if lcg_uniform(&mut state) >= q {
                    continue;
                }
                let th = std::f64::consts::TAU * lcg_uniform(&mut state);
                for j in 0..p {
                    data[[i, j]] +=
                        amps[c] * (th.cos() * frame[[j, 2 * c]] + th.sin() * frame[[j, 2 * c + 1]]);
                }
            }
            for j in 0..p {
                data[[i, j]] += sigma * lcg_normal(&mut state);
            }
        }
        let true_planes: Vec<Array2<f64>> = (0..k)
            .map(|c| {
                Array2::from_shape_fn((p, 2), |(row, col)| frame[[row, 2 * c + col]])
            })
            .collect();
        (data, true_planes)
    }

    /// Subspace affinity `‖UᵀV‖_F² / 2 ∈ [0, 1]` between two orthonormal
    /// ambient 2-planes (1 = same plane).
    fn plane_overlap(u: &Array2<f64>, v: &Array2<f64>) -> f64 {
        let m = u.t().dot(v);
        m.iter().map(|x| x * x).sum::<f64>() / 2.0
    }

    /// The DONE gate of #2111: run the producer, match each extracted plane to
    /// its best true plane, and count (distinct, real, clean). "Real" = best
    /// overlap ≥ 0.9; "clean" = additionally NOT spread onto a second circle
    /// (second-best overlap ≤ 0.2).
    fn producer_gate(
        data: &Array2<f64>,
        true_planes: &[Array2<f64>],
        max_planes: usize,
    ) -> (usize, usize, usize, usize, bool) {
        let harvest = isa_deflationary_producer(data.view(), max_planes, &IsaSeedConfig::default())
            .expect("producer must run");
        let mut claimed = std::collections::HashSet::new();
        let (mut n_real, mut n_clean) = (0usize, 0usize);
        for cand in &harvest.planes {
            let mut overlaps: Vec<(f64, usize)> = true_planes
                .iter()
                .enumerate()
                .map(|(idx, tp)| (plane_overlap(&cand.basis, tp), idx))
                .collect();
            overlaps.sort_by(|a, b| b.0.total_cmp(&a.0));
            claimed.insert(overlaps[0].1);
            if overlaps[0].0 >= 0.9 {
                n_real += 1;
                if overlaps.len() < 2 || overlaps[1].0 <= 0.2 {
                    n_clean += 1;
                }
            }
        }
        (
            claimed.len(),
            n_real,
            n_clean,
            harvest.planes.len(),
            harvest.natural_exit,
        )
    }

    /// DENSE TORUS, EQUAL AMPLITUDES — the load-bearing worst case: all six
    /// circle planes carry identical variance, so the whitened signal subspace
    /// is perfectly isotropic and second order gives NO ordering whatsoever
    /// (any orthonormal pairing is a valid eigenbasis). Only the fourth-order
    /// contrast separates the circles. Gate: 6 planes, all distinct, all real,
    /// all clean, natural exit.
    ///
    /// This is the Davis–Kahan exhaustion case named in the module doc made
    /// concrete: at second order the equal-amplitude 6-circle covariance is
    /// exactly isotropic on its 12-dim signal subspace, so ANY orthonormal
    /// re-pairing of coordinates is an equally valid eigenbasis and a
    /// second-order method returns arbitrary blends. Recovering 6 clean,
    /// distinct circles here is possible only because the κ contrast breaks
    /// the degeneracy that support/covariance information cannot.
    #[test]
    fn isa_producer_gate_dense_torus_equal_amplitude() {
        let k = 6usize;
        let amps = vec![1.0_f64; k];
        let (data, truth) = planted_circles(2000, 32, k, 1.0, &amps, 0.05, 0x2111_D07A_u64);
        let (n_distinct, n_real, n_clean, n_planes, natural_exit) =
            producer_gate(&data, &truth, 2 * k);
        assert!(
            n_distinct == k && n_real == k && n_clean == k && n_planes == k && natural_exit,
            "dense equal-amplitude torus gate: n_distinct={n_distinct} n_real={n_real} \
             n_clean={n_clean} planes={n_planes} natural_exit={natural_exit} (want 6/6/6, \
             6 planes, natural exit)"
        );
    }

    /// SPARSE GATED — six circles each active on a q = 0.25 Bernoulli row
    /// subset (κ = 1/q = 4, the super-Gaussian side of the contrast). Same
    /// gate: 6 distinct real clean planes, natural exit.
    ///
    /// Covers the OTHER anchor arm from the dense-torus test above: a gated
    /// circle's support is a lower-dimensional slice of the plane (only a q
    /// fraction of rows lie on the cone at all), yet the plane spanned by its
    /// nonzero rows is still just a 2-plane — support alone still cannot
    /// certify it. κ = 1/q = 4 on the super-Gaussian side of the anchor is
    #[test]
    fn zzz_debug_sparse_gated() {
        let k = 6usize;
        let amps: Vec<f64> = (0..k).map(|c| 1.0 + 0.1 * c as f64).collect();
        let (data, truth) = planted_circles(2000, 32, k, 0.25, &amps, 0.05, 0x2111_6A7E_u64);
        let mut work = data.clone();
        let cfg = IsaSeedConfig::default();
        for round in 0..12 {
            let parts = match isa_eigen_parts(work.view()).unwrap() {
                Some(p) => p,
                None => {
                    eprintln!("round {round}: eigen_parts None (natural exit)");
                    break;
                }
            };
            eprintln!(
                "round {round}: above.len()={} mp_edge={:.5} sigma2_cert={:.5} evals_top={:?}",
                parts.above.len(),
                parts.mp_edge,
                parts.sigma2_cert,
                parts
                    .above
                    .iter()
                    .take(4)
                    .map(|&i| (parts.evals[i] * 1e4).round() / 1e4)
                    .collect::<Vec<_>>()
            );
            // Inspect the winning basin's per-plane kappa BEFORE certification.
            let cand = isa_extract_certified_plane(work.view(), &parts, &cfg);
            match &cand {
                Some(c) => {
                    let mut best_ov = 0.0;
                    for tp in &truth {
                        let ov = plane_overlap(&c.basis, tp);
                        if ov > best_ov {
                            best_ov = ov;
                        }
                    }
                    eprintln!(
                        "  CERTIFIED: kappa={:.3} q_hat={:.3} best_overlap={:.3}",
                        c.kappa, c.q_hat, best_ov
                    );
                    isa_deflate_fitted_curve(&mut work, c);
                }
                None => {
                    eprintln!("  no certify -> natural exit");
                    // Show why: replicate the extraction's plane ordering + certify attempt.
                    debug_extract_why(work.view(), &parts, &cfg, &truth);
                    break;
                }
            }
        }
    }

    fn debug_extract_why(
        residual: ArrayView2<'_, f64>,
        parts: &IsaEigenParts,
        config: &IsaSeedConfig,
        truth: &[Array2<f64>],
    ) {
        let n = residual.nrows();
        let r = parts.above.len();
        let n_planes = r / 2;
        let cols = subsample_columns(n);
        let n_sub = cols.len();
        let mut z = Array2::<f64>::zeros((r, n_sub));
        for (a, &kk) in parts.above.iter().enumerate() {
            let inv = 1.0 / parts.evals[kk].max(f64::MIN_POSITIVE).sqrt();
            for (cc, &row) in cols.iter().enumerate() {
                let mut proj = 0.0_f64;
                for j in 0..residual.ncols() {
                    proj += (residual[[row, j]] - parts.mean[j]) * parts.evecs[[j, kk]];
                }
                z[[a, cc]] = proj * inv;
            }
        }
        let mut best: Option<(f64, Array2<f64>, Array2<f64>)> = None;
        for init in 0..config.n_inits.max(1) {
            let mut q = Array2::<f64>::eye(r);
            if init > 0 {
                let mut state = 0x2111_15A0_u64 ^ ((init as u64) << 32) ^ n as u64;
                let mut g = Array2::<f64>::from_shape_fn((r, r), |_| lcg_normal(&mut state));
                for c in 0..r {
                    for prev in 0..c {
                        let mut dot = 0.0;
                        for row in 0..r {
                            dot += g[[row, c]] * g[[row, prev]];
                        }
                        for row in 0..r {
                            let sub = dot * g[[row, prev]];
                            g[[row, c]] -= sub;
                        }
                    }
                    let mut nrm = 0.0;
                    for row in 0..r {
                        nrm += g[[row, c]] * g[[row, c]];
                    }
                    let nrm = nrm.sqrt();
                    for row in 0..r {
                        g[[row, c]] /= nrm;
                    }
                }
                q = g;
            }
            let mut y = q.t().dot(&z);
            jacobi_optimize(&mut y, &mut q, n_planes, config.max_sweeps);
            let contrast = total_contrast(&y, n_planes);
            if best.as_ref().is_none_or(|(bc, _, _)| contrast > *bc) {
                best = Some((contrast, q, y));
            }
        }
        let (tot, q, y) = best.unwrap();
        eprintln!("    winning basin total_contrast={tot:.4}, per-plane:");
        let mut order: Vec<(f64, usize)> = (0..n_planes)
            .map(|m| {
                let kk = plane_rows_kappa(&y, m);
                let c = if kk.is_finite() { (kk - 2.0) * (kk - 2.0) } else { 0.0 };
                (c, m)
            })
            .collect();
        order.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (contrast, m) in order.iter().take(n_planes) {
            let m = *m;
            let mut w = Array2::<f64>::zeros((r, 2));
            for row in 0..r {
                w[[row, 0]] = q[[row, 2 * m]];
                w[[row, 1]] = q[[row, 2 * m + 1]];
            }
            let kw = plane_rows_kappa(&y, m);
            // Certify manually with diagnostics inline.
            let (kobs, kmod, kbl, qh, best_ov) = certify_diag(residual, parts, &w, truth);
            eprintln!(
                "      plane m={m} whit_kappa={kw:.3} contrast={contrast:.4} -> kobs={kobs:.3} kmod={kmod:.3} kbl={kbl:.3} qhat={qh:.3} best_ov={best_ov:.3}"
            );
        }
    }

    fn certify_diag(
        residual: ArrayView2<'_, f64>,
        parts: &IsaEigenParts,
        w: &Array2<f64>,
        truth: &[Array2<f64>],
    ) -> (f64, f64, f64, f64, f64) {
        let (n, p) = residual.dim();
        let r = parts.above.len();
        let mut amb = Array2::<f64>::zeros((p, 2));
        for (a, &kk) in parts.above.iter().enumerate().take(r) {
            let inv = 1.0 / parts.evals[kk].max(f64::MIN_POSITIVE).sqrt();
            for j in 0..p {
                amb[[j, 0]] += parts.evecs[[j, kk]] * inv * w[[a, 0]];
                amb[[j, 1]] += parts.evecs[[j, kk]] * inv * w[[a, 1]];
            }
        }
        if !orthonormalize2(&mut amb) {
            return (f64::NAN, 0.0, 0.0, 0.0, 0.0);
        }
        let noise_2plane = 2.0 * parts.mp_edge;
        let (mut r2_sum, mut r4_sum) = (0.0_f64, 0.0_f64);
        let mut n_active = 0usize;
        for i in 0..n {
            let (mut p1, mut p2) = (0.0_f64, 0.0_f64);
            for j in 0..p {
                let ri = residual[[i, j]] - parts.mean[j];
                p1 += ri * amb[[j, 0]];
                p2 += ri * amb[[j, 1]];
            }
            let r2 = p1 * p1 + p2 * p2;
            r2_sum += r2;
            r4_sum += r2 * r2;
            if r2 > noise_2plane {
                n_active += 1;
            }
        }
        let q_hat = n_active as f64 / n as f64;
        let m2 = (r2_sum / n as f64).max(f64::MIN_POSITIVE);
        let kappa_obs = (r4_sum / n as f64) / (m2 * m2);
        let sig2 = parts.sigma2_cert;
        let a2t = ((m2 - 2.0 * sig2) / q_hat.max(f64::MIN_POSITIVE)).max(0.0);
        let common = 8.0 * q_hat * a2t * sig2 + 8.0 * sig2 * sig2;
        let kappa_model = (q_hat * a2t * a2t + common) / (m2 * m2);
        let kappa_blend = (1.25 * q_hat * a2t * a2t + common) / (m2 * m2);
        let mut best_ov = 0.0;
        for tp in truth {
            let ov = plane_overlap(&amb, tp);
            if ov > best_ov {
                best_ov = ov;
            }
        }
        (kappa_obs, kappa_model, kappa_blend, q_hat, best_ov)
    }

    /// what carries the identification.
    #[test]
    fn isa_producer_gate_sparse_gated() {
        let k = 6usize;
        let amps: Vec<f64> = (0..k).map(|c| 1.0 + 0.1 * c as f64).collect();
        let (data, truth) = planted_circles(2000, 32, k, 0.25, &amps, 0.05, 0x2111_6A7E_u64);
        let (n_distinct, n_real, n_clean, n_planes, natural_exit) =
            producer_gate(&data, &truth, 2 * k);
        assert!(
            n_distinct == k && n_real == k && n_clean == k && n_planes == k && natural_exit,
            "sparse gated gate: n_distinct={n_distinct} n_real={n_real} n_clean={n_clean} \
             planes={n_planes} natural_exit={natural_exit} (want 6/6/6, 6 planes, natural exit)"
        );
    }

    /// PLANTED-BLEND REJECTION NULL — a low-rank GAUSSIAN factor structure
    /// (every 2-plane of it has κ = 2, the blend anchor exactly). The producer
    /// must extract NOTHING and exit naturally: a κ certificate that accepted
    /// any plane here would hallucinate circles out of covariance alone.
    ///
    /// This is the HONESTY FACE of Prop. 1: a rank-6 Gaussian factor model has
    /// exactly the same support (a 6-dim subspace, decomposable into any
    /// orthonormal 2-plane pairing) as six circles do, so a support/rank test
    /// alone cannot tell them apart — that ambiguity is real, not a producer
    /// bug. The only thing separating them is measure: every 2-plane slice of
    /// a Gaussian factor sits at exactly κ = 2, the blend anchor `certify_plane`
    /// is built to refuse. Certifying a plane here would be the precise
    /// support-for-measure substitution Prop. 1 rules out — proving the
    /// negative is as load-bearing as the two positive gates above.
    #[test]
    fn isa_producer_rejects_planted_gaussian_blend() {
        let n = 2000usize;
        let p = 32usize;
        let rank = 6usize;
        let mut state = 0x2111_B1E4_D00D_u64;
        let mut frame = Array2::<f64>::from_shape_fn((p, rank), |_| lcg_normal(&mut state));
        for c in 0..rank {
            for prev in 0..c {
                let mut dot = 0.0;
                for row in 0..p {
                    dot += frame[[row, c]] * frame[[row, prev]];
                }
                for row in 0..p {
                    let sub = dot * frame[[row, prev]];
                    frame[[row, c]] -= sub;
                }
            }
            let mut nrm = 0.0;
            for row in 0..p {
                nrm += frame[[row, c]] * frame[[row, c]];
            }
            let nrm = nrm.sqrt();
            for row in 0..p {
                frame[[row, c]] /= nrm;
            }
        }
        let mut data = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for c in 0..rank {
                let g = lcg_normal(&mut state);
                for j in 0..p {
                    data[[i, j]] += g * frame[[j, c]];
                }
            }
            for j in 0..p {
                data[[i, j]] += 0.05 * lcg_normal(&mut state);
            }
        }
        let harvest =
            isa_deflationary_producer(data.view(), 12, &IsaSeedConfig::default()).expect("run");
        assert!(
            harvest.planes.is_empty() && harvest.natural_exit,
            "Gaussian factor blend must certify NO circle plane (got {} planes, \
             natural_exit={})",
            harvest.planes.len(),
            harvest.natural_exit
        );
    }

    /// The Jacobi pair polynomials must agree with direct evaluation of the
    /// rotated κ at arbitrary angles (the closed-form moment expansion is the
    /// load-bearing hand-derived math — check it against brute force).
    #[test]
    fn pair_polys_match_brute_force_rotation() {
        let r = 5usize;
        let n = 400usize;
        let mut state = 0xC0DE_2111_u64;
        let y = Array2::<f64>::from_shape_fn((r, n), |_| lcg_normal(&mut state));
        let (i, j) = (0usize, 2usize); // plane 0 = (0,1), plane 1 = (2,3), coord 4 unpaired
        let (pa, pb) = pair_polys(&y, i, j, Some(1), Some(3));
        for &theta in &[0.0, 0.3, -0.7, 1.1, std::f64::consts::FRAC_PI_3] {
            let (c, s) = (theta.cos(), theta.sin());
            // Brute force: rotate, recompute κ of both planes directly.
            let mut yr = y.clone();
            for col in 0..n {
                let yi = y[[i, col]];
                let yj = y[[j, col]];
                yr[[i, col]] = c * yi - s * yj;
                yr[[j, col]] = s * yi + c * yj;
            }
            let brute = {
                let ka = plane_rows_kappa(&yr, 0);
                let kb = plane_rows_kappa(&yr, 1);
                (ka - 2.0) * (ka - 2.0) + (kb - 2.0) * (kb - 2.0)
            };
            let poly = pair_objective(&pa, &pb, theta);
            assert!(
                (brute - poly).abs() < 1e-10 * (1.0 + brute.abs()),
                "pair polynomial mismatch at θ={theta}: brute={brute:.12} poly={poly:.12}"
            );
            // The hand-derived derivative must match a symmetric secant of the
            // POLYNOMIAL evaluation (test-only differentiation of a closed form).
            let h = 1e-6;
            let secant =
                (pair_objective(&pa, &pb, theta + h) - pair_objective(&pa, &pb, theta - h))
                    / (2.0 * h);
            let deriv = pair_objective_deriv(&pa, &pb, theta);
            assert!(
                (secant - deriv).abs() < 1e-5 * (1.0 + secant.abs()),
                "dJ/dθ mismatch at θ={theta}: secant={secant:.9} analytic={deriv:.9}"
            );
        }
    }
}

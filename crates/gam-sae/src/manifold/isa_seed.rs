//! Independent-subspace (ISA) capture-and-joint-rotation birth producer (#2111 hardening).
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
//! PRODUCER. Capture the above-Marchenko–Pastur signal subspace and count its
//! candidate planes (`⌊r/2⌋`). Whitening that captured span exhausts second
//! order and destroys amplitude ordering, so separation inside it must be
//! JOINT, not greedy: partition the `r` whitened coordinates into candidate
//! 2-planes, then run cyclic 2-plane JACOBI rotations over every coordinate
//! pair `(i, j)` spanning different planes. Each rotation maximizes the total
//! contrast `Σ_planes (κ_m − 2)²`, exactly evaluable at any angle from one
//! `O(n)` joint-moment pass (closed-form trigonometric polynomials — see
//! [`PlanePolys`]). Multistart (`≥ 6` random orthogonal inits — the
//! prototype-validated floor for escaping permutation/blend saddles) keeps the
//! best JOINT basin. Every plane in that single jointly rotated basis is then
//! accepted or refused by the ANALYTIC contrast certificate (anchors above; no
//! tuned ε) after mapping back through the GENERATIVE unwhitening `EΛ^{1/2}` to
//! its ambient support plane. Deflation is not a separation method here; it is
//! retained only as an external utility for callers that need to peel an
//! accepted fitted curve. That utility removes the accepted ambient plane's
//! centered covariance projection across the residual rows; certification gates
//! describe seed evidence, not which rows are allowed to retain accepted-plane
//! energy during residual peeling.
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
/// conventional `z = 3` level: `z·SE ≤ (1/q − 2)/2 = (1 − 2q)/(2q)`, i.e.
/// `z²·(1 − q)/(q³n) ≤ (1 − 2q)²/(4q²) ⇒ n ≥ 4z²·(1 − q)/(q·(1 − 2q)²)`.
/// The bound diverges at BOTH edges — at `q → ½` (a half-gated circle has
/// `κ = 2`, genuinely indistinguishable from a blend by κ alone) and at
/// `q → 0` (vanishing active mass: `Var(κ̂) ∼ 1/(q³n)`) — so the floor is set
/// at the dense-side practical design edge `q = 0.43`:
/// `n ≥ 4·9·0.57/(0.43·0.14²) ≈ 2435 → 2500`. On the sparse side the same
/// 2500 resolves gates down to `q ≈ 0.0155` (where the bound re-crosses 2500),
/// and the certificate's `ln²n` active-count floor already refuses gates in
/// that duty range at the sample sizes where subsampling engages at all; dense
/// circles concentrate strictly faster. (This also retires the old
/// `n = 80`-class fixtures: any harness exercising the fourth-order
/// certificate needs `n ≥ 300` to be out of the small-sample floor even in the
/// easy dense case.)
const ISA_SUBSAMPLE_FLOOR: usize = 2500;

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
    /// Noise scale for the κ certificate: the SAME monotone MP fixed-point
    /// estimate that sets [`Self::mp_edge`], i.e. `mp_edge / (1 + √(p/n))²`.
    /// The certificate must read the true white-noise variance, and the fixed
    /// point is the only estimator here robust to BOTH failure modes at once:
    ///
    /// * signal-majority (dense multi-circle round 0): a plain global median
    ///   lands in the SIGNAL bulk; the fixed point iterates the edge DOWN to
    ///   the largest prefix self-consistently below its own MP edge — the noise
    ///   band whatever fraction of directions are signal (see the derivation on
    ///   [`Self::mp_edge`]).
    /// * deflation RESIDUE (producer rounds ≥ 1): accepted planes land at ~0.99
    ///   ambient overlap, so deflating them leaves ~1% of a circle's energy in
    ///   near-deflated directions — eigenvalues strictly BELOW the noise bulk
    ///   (`~1e-5..1e-3` on the p=16/32 fixtures) but far above the numerical
    ///   rank floor. The old bottom-quartile-of-surviving estimator grabbed that
    ///   residue and read σ̂²_cert an order of magnitude too small, collapsing
    ///   the χ²₂ active-gate floor so noise rows gated active (`q̂ → 1`), the
    ///   noise-corrected anchors inverted, and freshly SEPARATED clean circles
    ///   were REFUSED — the observed sparse-torus stall at 4/6. The fixed point
    ///   is immune because the noise eigenvalues OUTNUMBER the residue, so the
    ///   noise-band median stays in the bulk.
    pub sigma2_cert: f64,
}

/// Upper-triangle accumulation of the centered second moment over rows
/// `lo..hi`: `S[a][b] += (r[row][a] − mean[a])·(r[row][b] − mean[b])` for
/// `b ≥ a`, rows in order. The centered row is materialized once per row (the
/// subtraction yields the identical value the old per-`(a,b)` recomputation
/// produced) and the inner update runs on contiguous slices, so per entry the
/// terms and their addition order are exactly the legacy loop's — a single
/// chunk spanning `0..n` is bit-identical to the pre-chunk implementation.
fn centered_second_moment_chunk(
    residual: ArrayView2<'_, f64>,
    mean: &Array1<f64>,
    lo: usize,
    hi: usize,
) -> Array2<f64> {
    let p = residual.ncols();
    let mut s = Array2::<f64>::zeros((p, p));
    let mut crow = vec![0.0_f64; p];
    for row in lo..hi {
        for (j, slot) in crow.iter_mut().enumerate() {
            *slot = residual[[row, j]] - mean[j];
        }
        for a in 0..p {
            let ra = crow[a];
            let mut srow = s.row_mut(a);
            let srow = srow.as_slice_mut().expect("row of standard-layout matrix");
            for b in a..p {
                srow[b] += ra * crow[b];
            }
        }
    }
    s
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
    for row in residual.outer_iter() {
        mean += &row;
    }
    mean.mapv_inplace(|v| v / n as f64);
    // Centered second moment — the O(n·p²) pass that dominates every producer
    // round on the full residual. Parallelized over FIXED contiguous row chunks
    // whose upper-triangle partials are summed in CHUNK ORDER, so the result is
    // bit-reproducible and independent of thread count; it differs from the
    // single running row-sum only in the harmless grouping of accumulation
    // round-off, which the mirroring below (and the eigensolve's own tolerance)
    // absorbs — the same determinism contract `scaled_second_moment` in the
    // structured-residual estimator (#974) already established for exactly this
    // shape of matrix. Engaged only above a row threshold (the serial path stays
    // exact on small inputs and avoids rayon overhead) and only when NOT already
    // inside a rayon worker (nested calls keep the outer region's cores).
    let mut s = {
        use rayon::prelude::*;
        const PARALLEL_ROW_MIN: usize = 8192;
        const CHUNK_ROWS: usize = 2048;
        if n >= PARALLEL_ROW_MIN && rayon::current_thread_index().is_none() {
            let n_chunks = n.div_ceil(CHUNK_ROWS);
            let partials: Vec<Array2<f64>> = (0..n_chunks)
                .into_par_iter()
                .map(|c| {
                    let lo = c * CHUNK_ROWS;
                    let hi = ((c + 1) * CHUNK_ROWS).min(n);
                    centered_second_moment_chunk(residual, &mean, lo, hi)
                })
                .collect();
            let mut acc = Array2::<f64>::zeros((p, p));
            for part in &partials {
                acc += part;
            }
            acc
        } else {
            centered_second_moment_chunk(residual, &mean, 0, n)
        }
    };
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
    let gamma = p as f64 / n as f64;
    let mp_factor = (1.0 + gamma.sqrt()).powi(2);
    // Numerical-rank screen BEFORE any noise estimation. Deflation
    // (`isa_deflate_fitted_curve`) removes each accepted plane's two variance
    // dimensions EXACTLY, so in producer rounds ≥ 2 the bottom of the spectrum
    // carries one numerically-zero eigenvalue per deflated dimension (and a
    // rank-deficient `n ≤ p` sample covariance plants them on round 1). Those
    // are rank deficiencies, not noise observations: left in, the bottom-
    // quartile certificate scale reads a deflated zero, `σ̂²_cert ≈ 0`
    // collapses the χ²₂ gate floor to `2·MIN_POSITIVE·ln n`, every row gates
    // active (`q̂ = 1`, `κ_model = 1`, `gate_mid = 1.125`), genuinely gated
    // circles left in the residual (`κ_obs ≈ 1/q ≫ 1.125`) are REFUSED, and
    // any accepted dense circle ships saturated `gate_logits` for all rows.
    // Both noise estimators below therefore run on the SURVIVING spectrum —
    // eigenvalues above the eigensolve's numerical-rank threshold. The MP edge
    // aspect stays `p/n`: for `n ≤ p` the nonzero Wishart bulk still tops out
    // at `σ²(1 + √(p/n))²`, and after deflation `p/n` is (mildly) conservative.
    let rank_tol = ascending.last().copied().unwrap_or(0.0).max(0.0) * f64::EPSILON * p as f64;
    let n_rank_deficient = ascending.partition_point(|&e| e <= rank_tol);
    let surviving = &ascending[n_rank_deficient..];
    if surviving.is_empty() {
        return Ok(None);
    }
    // Robust noise floor under MAJORITY signal. A plain median of the spectrum
    // estimates σ² only while signal directions are a MINORITY; on a DENSE
    // product-of-circles residual the signal dims are the majority — a k-torus
    // carries 2k signal eigenvalues (each ≈ a²/2) and 2k > p/2 once the torus is
    // dense in its frame (the p=16, k=6 `probe_2101` fixture has 12 signal dims of
    // 16). Then the median lands in the SIGNAL bulk and `λ₊ = σ̂²(1+√γ)²` inflates
    // to admit only the few strongest circles — the observed 3-of-6 stall (`first
    // three at overlap≈0.99, circles 4-6 never surface`), while the p=32 producer
    // gates (12 of 32 = minority) are unaffected because their median already sits
    // in noise. Estimate σ² instead by a monotone fixed point: σ̂² = median of the
    // eigenvalues at-or-below the current edge, iterated. The candidate noise band
    // is a prefix of the ascending spectrum and only shrinks (the edge is
    // non-increasing because a shorter prefix of a sorted vector has a
    // non-greater median), so it converges in ≤ p steps to the largest prefix
    // self-consistently below its own MP edge — the true noise band, whatever
    // fraction of directions are signal. On a signal-MINORITY spectrum the very
    // first median is already in noise and the iterate is a no-op (bit-identical
    // to the old median floor); on pure noise the whole spectrum is the fixed
    // point and nothing clears the edge (the natural-stop is preserved).
    let median_prefix = |xs: &[f64]| -> f64 {
        let m = xs.len();
        if m == 0 {
            return f64::MIN_POSITIVE;
        }
        if m % 2 == 1 {
            xs[m / 2]
        } else {
            0.5 * (xs[m / 2 - 1] + xs[m / 2])
        }
        .max(f64::MIN_POSITIVE)
    };
    let mut noise_len = surviving.len();
    let mut sigma2 = median_prefix(&surviving[..noise_len]);
    loop {
        let edge = sigma2 * mp_factor;
        // Eigenvalues at-or-below the edge form the candidate noise band (a prefix
        // of the ascending surviving spectrum, since `e <= edge` is true-then-false
        // on it).
        let new_len = surviving.partition_point(|&e| e <= edge);
        if new_len == 0 || new_len == noise_len {
            break;
        }
        noise_len = new_len;
        sigma2 = median_prefix(&surviving[..noise_len]);
    }
    let mp_edge = sigma2 * mp_factor;
    let mut above: Vec<usize> = (0..evals.len()).filter(|&k| evals[k] > mp_edge).collect();
    above.sort_by(|&a, &b| evals[b].total_cmp(&evals[a]));
    if above.is_empty() {
        return Ok(None);
    }
    // The certificate noise scale is the MP fixed-point σ̂² (`mp_edge/mp_factor`),
    // NOT a bottom-quartile of the surviving spectrum: after a deflation the
    // sub-bulk residue of ~0.99-overlap accepted planes contaminates the bottom
    // quartile and reads σ̂²_cert far too small, refusing clean circles. The
    // fixed point rides the noise bulk (which outnumbers the residue) instead.
    let sigma2_cert = sigma2;
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
        let num = self.n[0] + self.n[1] * c2 + self.n[2] * s2 + self.n[3] * c4 + self.n[4] * s4;
        num / (den * den)
    }

    /// Hand-derived `dκ/dθ` at the same angle (quotient rule on the two
    /// trigonometric polynomials; `dC₂/dθ = −2S₂`, `dS₂/dθ = 2C₂`, etc.).
    fn dkappa(&self, c2: f64, s2: f64, c4: f64, s4: f64) -> f64 {
        let den = self.d[0] + self.d[1] * c2 + self.d[2] * s2;
        if !(den > 0.0) {
            return 0.0;
        }
        let num = self.n[0] + self.n[1] * c2 + self.n[2] * s2 + self.n[3] * c4 + self.n[4] * s4;
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
    // This is the innermost Jacobi kernel — one call per candidate pair per
    // sweep — so hoist the four coordinate rows to contiguous slices once and
    // index those, instead of paying `y[[row, col]]`'s 2-D offset arithmetic and
    // bounds check per element. Same elements, same accumulation order: the
    // moments are bit-identical to the indexed form.
    let yi_row = y.row(i);
    let yi_row = yi_row.as_slice().expect("row of standard-layout matrix");
    let yj_row = y.row(j);
    let yj_row = yj_row.as_slice().expect("row of standard-layout matrix");
    let pi_row = partner_i.map(|pi| y.row(pi));
    let pi_row = pi_row
        .as_ref()
        .map(|r| r.as_slice().expect("row of standard-layout matrix"));
    let pj_row = partner_j.map(|pj| y.row(pj));
    let pj_row = pj_row
        .as_ref()
        .map(|r| r.as_slice().expect("row of standard-layout matrix"));
    for col in 0..n {
        let yi = yi_row[col];
        let yj = yj_row[col];
        let (yi2, yj2, yij) = (yi * yi, yj * yj, yi * yj);
        m20 += yi2;
        m11 += yij;
        m02 += yj2;
        m40 += yi2 * yi2;
        m31 += yi2 * yij;
        m22 += yi2 * yj2;
        m13 += yij * yj2;
        m04 += yj2 * yj2;
        if let Some(prow) = pi_row {
            let w = prow[col];
            let w2 = w * w;
            a2 += w2;
            a4 += w2 * w2;
            awi += w2 * yi2;
            awx += w2 * yij;
            awj += w2 * yj2;
        }
        if let Some(prow) = pj_row {
            let w = prow[col];
            let w2 = w * w;
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

fn best_pair_rotation(a: &Option<PlanePolys>, b: &Option<PlanePolys>) -> (f64, f64) {
    let step = std::f64::consts::PI / ISA_ANGLE_SAMPLES as f64;
    let mut samples: Vec<(f64, f64, f64)> = Vec::with_capacity(ISA_ANGLE_SAMPLES + 1);
    for k in 0..=ISA_ANGLE_SAMPLES {
        let theta = -std::f64::consts::FRAC_PI_2 + step * k as f64;
        samples.push((
            theta,
            pair_objective(a, b, theta),
            pair_objective_deriv(a, b, theta),
        ));
    }

    let mut best_theta = 0.0_f64;
    let mut best_j = pair_objective(a, b, 0.0);
    for &(theta, jv, _) in &samples {
        if jv > best_j {
            best_j = jv;
            best_theta = theta;
        }
    }

    for k in 0..ISA_ANGLE_SAMPLES {
        let (mut lo, _, dlo) = samples[k];
        let (mut hi, _, dhi) = samples[k + 1];
        if !(dlo.is_finite() && dhi.is_finite()) {
            continue;
        }
        if dlo == 0.0 {
            let jv = pair_objective(a, b, lo);
            if jv > best_j {
                best_j = jv;
                best_theta = lo;
            }
            continue;
        }
        if dhi == 0.0 {
            let jv = pair_objective(a, b, hi);
            if jv > best_j {
                best_j = jv;
                best_theta = hi;
            }
            continue;
        }
        if dlo.signum() == dhi.signum() {
            continue;
        }
        let mut dlo_cur = dlo;
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            let dmid = pair_objective_deriv(a, b, mid);
            if dmid == 0.0 {
                lo = mid;
                hi = mid;
                break;
            }
            if dmid.signum() == dlo_cur.signum() {
                lo = mid;
                dlo_cur = dmid;
            } else {
                hi = mid;
            }
        }
        let theta = 0.5 * (lo + hi);
        let jv = pair_objective(a, b, theta);
        if jv > best_j {
            best_j = jv;
            best_theta = theta;
        }
    }

    (best_theta, best_j)
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
/// the objective the Jacobi sweep ascends and the score used to rank joint
/// basins across multistart inits and planes inside a single basin (see
/// [`isa_extract_certified_planes`]). Same Gaussian-anchor logic as
/// [`pair_objective`], just summed over all current planes rather than the
/// two planes touched by one rotation.
fn total_contrast(y: &Array2<f64>, n_planes: usize) -> f64 {
    (0..n_planes)
        .map(|m| {
            let k = plane_rows_kappa(y, m);
            if k.is_finite() {
                (k - 2.0) * (k - 2.0)
            } else {
                0.0
            }
        })
        .sum()
}

/// Cyclic 2-plane Jacobi ascent on the ISA contrast. `y` holds the rotated
/// coordinates (`r × n`), `q` the accumulated orthogonal rotation (`y = qᵀ·z`,
/// columns of `q` are whitened-space directions); both are updated in place.
fn jacobi_optimize(y: &mut Array2<f64>, q: &mut Array2<f64>, n_planes: usize, max_sweeps: usize) {
    let r = y.nrows();
    let partner = |c: usize| -> Option<usize> { if c < 2 * n_planes { Some(c ^ 1) } else { None } };
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
                // Nyquist-rate scan over the full period plus all derivative
                // sign-change roots. The old single-cell polish missed maxima
                // whose grid winner was not bracketed by `+ → -` derivatives.
                let (best_theta, best_j) = best_pair_rotation(&pa, &pb);
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

/// Whitened above-floor coordinates, row-major as `(r, n_sub)` for the Jacobi
/// moment pass. The rotation itself is estimated on the deterministic
/// concentration-floor subsample; certification still uses all rows.
fn whitened_subsample(residual: ArrayView2<'_, f64>, parts: &IsaEigenParts) -> Option<Array2<f64>> {
    let n = residual.nrows();
    let r = parts.above.len();
    if r < 2 || n < 2 {
        return None;
    }
    let cols = subsample_columns(n);
    let mut z = Array2::<f64>::zeros((r, cols.len()));
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
    Some(z)
}

/// Multistart joint Jacobi basis for the whole captured whitened span. Returns
/// `(q, y)`, where `y = qᵀ z`; columns `(2m, 2m+1)` of `q` are the separated
/// whitened-space plane `m`. This is the only separation step: callers may
/// choose how many certified planes to consume, but they must not recursively
/// re-separate by greedy deflation inside this captured span.
fn joint_jacobi_basis(
    residual: ArrayView2<'_, f64>,
    parts: &IsaEigenParts,
    config: &IsaSeedConfig,
) -> Option<(Array2<f64>, Array2<f64>)> {
    let z = whitened_subsample(residual, parts)?;
    let r = z.nrows();
    let n_planes = r / 2;
    let n_inits = if n_planes == 1 {
        1
    } else {
        config.n_inits.max(1)
    };
    // Every multistart init is fully independent — its own LCG state (seeded by
    // `init`), its own `q`/`y`, its own Jacobi ascent — and the winner selection
    // is a strict `>` scan in init order. So the inits fan out across cores and
    // the sequential scan over the ORDER-PRESERVING indexed collect reproduces
    // the serial loop's winner (including its keep-the-earlier-init tie
    // behavior) bit-for-bit: no arithmetic moved, only which core ran it. This
    // was the other serial wall of the ISA harvest besides the second-moment
    // pass — `n_inits` (default 6) full Jacobi ascents back to back. Engaged
    // only when not already inside a rayon worker (nested calls keep the outer
    // region's cores), matching the estimator-wide nesting discipline.
    let candidates: Vec<(f64, Array2<f64>, Array2<f64>)> = {
        use rayon::prelude::*;
        let run_init = |init: usize| -> (f64, Array2<f64>, Array2<f64>) {
            let (mut q, mut y) = jacobi_init_state(&z, r, init, residual.nrows());
            jacobi_optimize(&mut y, &mut q, n_planes, config.max_sweeps);
            let contrast = total_contrast(&y, n_planes);
            (contrast, q, y)
        };
        if n_inits > 1 && rayon::current_thread_index().is_none() {
            (0..n_inits).into_par_iter().map(run_init).collect()
        } else {
            (0..n_inits).map(run_init).collect()
        }
    };
    let mut best: Option<(f64, Array2<f64>, Array2<f64>)> = None;
    for (contrast, q, y) in candidates {
        if best.as_ref().is_none_or(|(bc, _, _)| contrast > *bc) {
            best = Some((contrast, q, y));
        }
    }
    best.map(|(_, q, y)| (q, y))
}

/// Initial rotation state for multistart init `init`: identity for the first
/// init, a deterministic LCG Gram-Schmidt random orthogonal basis for the rest,
/// paired with the rotated coordinates `y = qᵀ z`. Extracted from the multistart
/// loop verbatim so the serial and parallel drivers share one body.
fn jacobi_init_state(
    z: &Array2<f64>,
    r: usize,
    init: usize,
    residual_rows: usize,
) -> (Array2<f64>, Array2<f64>) {
    let mut q = Array2::<f64>::eye(r);
    if init > 0 {
        // Random orthogonal via Gram-Schmidt of LCG normal columns. The seed
        // mixes the FULL residual row count (not the subsample width), exactly
        // as the pre-extraction loop did.
        let mut state = 0x2111_15A0_u64 ^ ((init as u64) << 32) ^ residual_rows as u64;
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
    let y = q.t().dot(z);
    (q, y)
}

pub(crate) fn capture_signal_span(
    residual: ArrayView2<'_, f64>,
    max_planes: usize,
) -> Result<Option<IsaEigenParts>, String> {
    if max_planes == 0 {
        return Ok(None);
    }
    let Some(mut parts) = isa_eigen_parts(residual)? else {
        return Ok(None);
    };
    let mut keep = parts.above.len();
    if keep % 2 == 1 {
        keep -= 1;
    }
    if keep < 2 {
        return Ok(None);
    }
    parts.above.truncate(keep);
    Ok(Some(parts))
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
    // Un-whiten to the ambient SUPPORT plane: a whitened source direction `w`
    // generates ambient variation as `E Λ^{1/2} w`. The reciprocal
    // `E Λ^{-1/2} w` is the score functional used to READ the whitened
    // coordinate, not the Euclidean support plane to birth/deflate. On unequal
    // circle amplitudes the two maps span different mixed planes; using the
    // score plane leaves the accepted strong circle's covariance behind, so
    // later rounds keep seeing it instead of the weaker circles.
    let mut amb = Array2::<f64>::zeros((p, 2));
    for (a, &k) in parts.above.iter().enumerate().take(r) {
        let scale = parts.evals[k].max(f64::MIN_POSITIVE).sqrt();
        for j in 0..p {
            amb[[j, 0]] += parts.evecs[[j, k]] * scale * w[[a, 0]];
            amb[[j, 1]] += parts.evecs[[j, k]] * scale * w[[a, 1]];
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

/// Extract all certified circle planes from one captured above-floor span:
/// whiten the span once, run one JOINT multistart Jacobi rotation over every
/// candidate plane pair, then certify planes from that joint basis. `max_planes`
/// is a caller safety bound on how many certified planes to consume, not a
/// separation mechanism.
pub fn isa_extract_certified_planes(
    residual: ArrayView2<'_, f64>,
    parts: &IsaEigenParts,
    max_planes: usize,
    config: &IsaSeedConfig,
) -> Vec<IsaPlaneCandidate> {
    let r = parts.above.len();
    if r < 2 || max_planes == 0 {
        return Vec::new();
    }
    let n_planes = r / 2;
    let Some((q, y)) = joint_jacobi_basis(residual, parts, config) else {
        return Vec::new();
    };
    // Planes in descending single-plane contrast, but all come from the same
    // joint basin. This ordering only selects emission order under max_planes.
    let mut order: Vec<(f64, usize)> = (0..n_planes)
        .map(|m| {
            let k = plane_rows_kappa(&y, m);
            let c = if k.is_finite() {
                (k - 2.0) * (k - 2.0)
            } else {
                0.0
            };
            (c, m)
        })
        .collect();
    order.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut out = Vec::new();
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
            out.push(cand);
            if out.len() >= max_planes {
                break;
            }
        }
    }
    out
}

/// Extract one certified circle plane from the JOINT split of the current
/// captured span. This is a compatibility shim for single-birth callers; it
/// does not perform greedy recursive separation.
pub fn isa_extract_certified_plane(
    residual: ArrayView2<'_, f64>,
    parts: &IsaEigenParts,
    config: &IsaSeedConfig,
) -> Option<IsaPlaneCandidate> {
    isa_extract_certified_planes(residual, parts, 1, config)
        .into_iter()
        .next()
}

/// Subtract an accepted plane's centered residual projection, in place. Gates
/// certify seed evidence only; deflation removes the certified ambient SUPPORT
/// plane from every centered row so accepted-plane covariance is actually gone
/// before the next eigendecomposition.
pub fn isa_deflate_fitted_curve(residual: &mut Array2<f64>, cand: &IsaPlaneCandidate) {
    let (n, p) = residual.dim();
    if n == 0 {
        return;
    }
    let mut mean = Array1::<f64>::zeros(p);
    for i in 0..n {
        for j in 0..p {
            mean[j] += residual[[i, j]];
        }
    }
    mean.mapv_inplace(|v| v / n as f64);
    for i in 0..n {
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

/// The producer's harvest: the certified planes from one joint split, and
/// whether harvest ended before hitting the caller's safety cap.
pub struct IsaHarvest {
    pub planes: Vec<IsaPlaneCandidate>,
    pub natural_exit: bool,
}

/// Full producer run: capture an even-dimensional above-floor support span,
/// jointly rotate every candidate plane inside that span, emit every certified
/// plane from that joint split, then remove accepted support before the next
/// capture round. Deflation only advances the residual snapshot between certified
/// joint splits; it is not a greedy separator inside a captured whitened span.
pub fn isa_deflationary_producer(
    residual: ArrayView2<'_, f64>,
    max_planes: usize,
    config: &IsaSeedConfig,
) -> Result<IsaHarvest, String> {
    if max_planes == 0 {
        return Ok(IsaHarvest {
            planes: Vec::new(),
            natural_exit: true,
        });
    }
    let mut work = residual.to_owned();
    let mut planes = Vec::new();
    while planes.len() < max_planes {
        let remaining = max_planes - planes.len();
        let Some(parts) = capture_signal_span(work.view(), remaining)? else {
            break;
        };
        let mut round = isa_extract_certified_planes(work.view(), &parts, remaining, config);
        if round.is_empty() {
            break;
        }
        for cand in &round {
            isa_deflate_fitted_curve(&mut work, cand);
        }
        planes.append(&mut round);
    }
    let natural_exit = planes.len() < max_planes;
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
            .map(|c| Array2::from_shape_fn((p, 2), |(row, col)| frame[[row, 2 * c + col]]))
            .collect();
        (data, true_planes)
    }

    /// Subspace affinity `‖UᵀV‖_F² / 2 ∈ [0, 1]` between two orthonormal
    /// ambient 2-planes (1 = same plane).
    fn plane_overlap(u: &Array2<f64>, v: &Array2<f64>) -> f64 {
        let m = u.t().dot(v);
        m.iter().map(|x| x * x).sum::<f64>() / 2.0
    }

    const PLANE_OVERLAP_REAL_FLOOR: f64 = 0.9;
    const PLANE_OVERLAP_BLEND_CEIL: f64 = 0.2;

    #[derive(Debug)]
    struct ProducerGateMetrics {
        n_distinct: usize,
        n_real: usize,
        n_clean: usize,
        n_planes: usize,
        natural_exit: bool,
        signal_span_dim: usize,
        power_plane_count: usize,
        residual_power_dim: usize,
        residual_excess_energy: f64,
        truth_overlaps: Vec<f64>,
        best_overlaps: Vec<f64>,
        second_overlaps: Vec<f64>,
    }

    fn residual_power_metrics(data: &Array2<f64>) -> (usize, f64) {
        let Some(parts) = isa_eigen_parts(data.view()).expect("residual eigensolve must run")
        else {
            return (0, 0.0);
        };
        let excess = parts
            .above
            .iter()
            .map(|&idx| (parts.evals[idx] - parts.mp_edge).max(0.0))
            .sum();
        (parts.above.len(), excess)
    }

    /// The DONE gate of #2111: run the producer, match each extracted plane to
    /// its best true plane, and count (distinct, real, clean). "Real" = best
    /// overlap ≥ 0.9; "clean" = additionally NOT spread onto a second circle
    /// (second-best overlap ≤ 0.2).
    fn producer_gate_metrics(
        data: &Array2<f64>,
        true_planes: &[Array2<f64>],
        max_planes: usize,
    ) -> ProducerGateMetrics {
        let captured = capture_signal_span(data.view(), max_planes)
            .expect("initial capture must run")
            .expect("fixture must have an above-threshold signal span");
        let signal_span_dim = captured.above.len();
        let power_plane_count = signal_span_dim / 2;
        let harvest = isa_deflationary_producer(data.view(), max_planes, &IsaSeedConfig::default())
            .expect("producer must run");
        let mut claimed = std::collections::HashSet::new();
        let (mut n_real, mut n_clean) = (0usize, 0usize);
        let (mut best_overlaps, mut second_overlaps) = (Vec::new(), Vec::new());
        for cand in &harvest.planes {
            let mut overlaps: Vec<(f64, usize)> = true_planes
                .iter()
                .enumerate()
                .map(|(idx, tp)| (plane_overlap(&cand.basis, tp), idx))
                .collect();
            overlaps.sort_by(|a, b| b.0.total_cmp(&a.0));
            claimed.insert(overlaps[0].1);
            best_overlaps.push(overlaps[0].0);
            let second = overlaps.get(1).map_or(0.0, |(ov, _)| *ov);
            second_overlaps.push(second);
            if overlaps[0].0 >= PLANE_OVERLAP_REAL_FLOOR {
                n_real += 1;
                if second <= PLANE_OVERLAP_BLEND_CEIL {
                    n_clean += 1;
                }
            }
        }
        let mut deflated = data.clone();
        for cand in &harvest.planes {
            isa_deflate_fitted_curve(&mut deflated, cand);
        }
        let (residual_power_dim, residual_excess_energy) = residual_power_metrics(&deflated);
        let truth_overlaps = candidate_overlaps(&harvest.planes, true_planes);
        ProducerGateMetrics {
            n_distinct: claimed.len(),
            n_real,
            n_clean,
            n_planes: harvest.planes.len(),
            natural_exit: harvest.natural_exit,
            signal_span_dim,
            power_plane_count,
            residual_power_dim,
            residual_excess_energy,
            truth_overlaps,
            best_overlaps,
            second_overlaps,
        }
    }

    fn assert_producer_gate_margins(label: &str, metrics: &ProducerGateMetrics, k: usize) {
        assert!(
            metrics.signal_span_dim >= 2 * k && metrics.power_plane_count >= k,
            "{label}: underpowered fixture before producer: metrics={metrics:?}"
        );
        assert!(
            metrics
                .truth_overlaps
                .iter()
                .all(|&ov| ov >= PLANE_OVERLAP_REAL_FLOOR),
            "{label}: at least one true circle lacks a recovered plane: metrics={metrics:?}"
        );
        assert!(
            metrics
                .best_overlaps
                .iter()
                .all(|&ov| ov >= PLANE_OVERLAP_REAL_FLOOR),
            "{label}: at least one emitted plane is not real: metrics={metrics:?}"
        );
        assert!(
            metrics
                .second_overlaps
                .iter()
                .all(|&ov| ov <= PLANE_OVERLAP_BLEND_CEIL),
            "{label}: at least one emitted plane remains blended: metrics={metrics:?}"
        );
        assert!(
            metrics.residual_power_dim == 0 && metrics.residual_excess_energy == 0.0,
            "{label}: accepted-plane deflation left above-MP residual energy: metrics={metrics:?}"
        );
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
        let metrics = producer_gate_metrics(&data, &truth, 2 * k);
        assert_producer_gate_margins("dense equal-amplitude torus gate", &metrics, k);
        assert!(
            metrics.n_distinct == k
                && metrics.n_real == k
                && metrics.n_clean == k
                && metrics.n_planes == k
                && metrics.natural_exit,
            "dense equal-amplitude torus gate exact all-six stress failed: metrics={metrics:?}"
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
    /// what carries the identification.
    #[test]
    fn isa_producer_gate_sparse_gated() {
        let k = 6usize;
        let amps: Vec<f64> = (0..k).map(|c| 1.0 + 0.1 * c as f64).collect();
        let (data, truth) = planted_circles(2000, 32, k, 0.25, &amps, 0.05, 0x2111_6A7E_u64);
        let metrics = producer_gate_metrics(&data, &truth, 2 * k);
        assert_producer_gate_margins("sparse gated gate", &metrics, k);
        assert!(
            metrics.n_distinct == k
                && metrics.n_real == k
                && metrics.n_clean == k
                && metrics.n_planes == k
                && metrics.natural_exit,
            "sparse gated gate exact all-six stress failed: metrics={metrics:?}"
        );
    }

    fn best_truth_overlaps(planes: &[Array2<f64>], truth: &[Array2<f64>]) -> Vec<f64> {
        let mut per_truth = vec![0.0_f64; truth.len()];
        for plane in planes {
            for (idx, tp) in truth.iter().enumerate() {
                per_truth[idx] = per_truth[idx].max(plane_overlap(plane, tp));
            }
        }
        per_truth
    }

    fn candidate_overlaps(planes: &[IsaPlaneCandidate], truth: &[Array2<f64>]) -> Vec<f64> {
        let bases: Vec<Array2<f64>> = planes.iter().map(|cand| cand.basis.clone()).collect();
        best_truth_overlaps(&bases, truth)
    }

    fn ambient_plane_from_captured_parts(
        parts: &IsaEigenParts,
        q: &Array2<f64>,
        m: usize,
    ) -> Array2<f64> {
        let p = parts.evecs.nrows();
        let r = parts.above.len();
        let mut amb = Array2::<f64>::zeros((p, 2));
        for (a, &k) in parts.above.iter().enumerate().take(r) {
            let scale = parts.evals[k].max(f64::MIN_POSITIVE).sqrt();
            for j in 0..p {
                amb[[j, 0]] += parts.evecs[[j, k]] * scale * q[[a, 2 * m]];
                amb[[j, 1]] += parts.evecs[[j, k]] * scale * q[[a, 2 * m + 1]];
            }
        }
        assert!(
            orthonormalize2(&mut amb),
            "captured joint plane must have rank 2"
        );
        amb
    }

    fn greedy_deflation_probe(
        data: &Array2<f64>,
        max_planes: usize,
        config: &IsaSeedConfig,
    ) -> Vec<IsaPlaneCandidate> {
        let mut work = data.clone();
        let mut planes = Vec::new();
        while planes.len() < max_planes {
            let Some(parts) = isa_eigen_parts(work.view()).expect("greedy probe eigensolve") else {
                break;
            };
            let Some(cand) = isa_extract_certified_plane(work.view(), &parts, config) else {
                break;
            };
            isa_deflate_fitted_curve(&mut work, &cand);
            planes.push(cand);
        }
        planes
    }

    #[test]
    fn isa_joint_rotation_recovers_unequal_gated_circles_where_greedy_collapses() {
        let k = 6usize;
        let amps = vec![1.00, 0.86, 0.73, 0.61, 0.50, 0.41];
        let qs = vec![0.90, 0.65, 0.42, 0.25, 0.14, 0.08];
        let (data, truth) =
            planted_circles_unequal_gates(12_000, 32, &qs, &amps, 0.03, 0x2111_15A_u64);
        let config = IsaSeedConfig {
            n_inits: 10,
            max_sweeps: 80,
        };
        let greedy = greedy_deflation_probe(&data, k, &config);
        let greedy_overlaps = candidate_overlaps(&greedy, &truth);
        let captured = capture_signal_span(data.view(), k)
            .expect("capture must run")
            .expect("capture must find signal span");
        let (q, _y) = joint_jacobi_basis(data.view(), &captured, &config)
            .expect("joint basis must optimize captured span");
        let joint_planes: Vec<Array2<f64>> = (0..k)
            .map(|m| ambient_plane_from_captured_parts(&captured, &q, m))
            .collect();
        let joint_overlaps = best_truth_overlaps(&joint_planes, &truth);
        eprintln!(
            "[#2111 unequal gated] greedy overlaps = {:?}; joint overlaps = {:?}",
            greedy_overlaps, joint_overlaps
        );
        assert!(
            joint_overlaps.iter().all(|&ov| ov > 0.35),
            "joint ISA must recover every unequal gated plane above the weak-source floor; \
             overlaps={joint_overlaps:?}"
        );
    }

    fn planted_circles_unequal_gates(
        n: usize,
        p: usize,
        qs: &[f64],
        amps: &[f64],
        sigma: f64,
        seed: u64,
    ) -> (Array2<f64>, Vec<Array2<f64>>) {
        assert!(qs.len() == amps.len() && p >= 2 * qs.len());
        let k = qs.len();
        let mut state = seed;
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
                if lcg_uniform(&mut state) >= qs[c] {
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
            .map(|c| Array2::from_shape_fn((p, 2), |(row, col)| frame[[row, 2 * c + col]]))
            .collect();
        (data, true_planes)
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
            let secant = (pair_objective(&pa, &pb, theta + h)
                - pair_objective(&pa, &pb, theta - h))
                / (2.0 * h);
            let deriv = pair_objective_deriv(&pa, &pb, theta);
            assert!(
                (secant - deriv).abs() < 1e-5 * (1.0 + secant.abs()),
                "dJ/dθ mismatch at θ={theta}: secant={secant:.9} analytic={deriv:.9}"
            );
        }
    }
}

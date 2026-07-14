//! Block 5 — Cubic B-spline cell-moment tables on a fixed knot grid.
//!
//! This module computes the per-cell moment integrals
//!
//!   I_ν^{ij}(m)  = ∫_{L}^{R} (x − m)^ν · B_i(x) · B_j(x) dx               (1D)
//!   M_α^{ij}(c)  = ∫_{c} (x − m)^α · B_i(x) · B_j(x) dx                  (d-D tensor)
//!
//! where (B_i, B_j) are two cubic (degree-3) B-splines that are *both* nonzero on a
//! single knot span (a "cell" in the tensor sense), m is a chosen expansion point
//! (we use the cell's left corner L so u = x − L stays in [0, h]), and α is a
//! multi-index of moment exponents.  The output table is consumed by tensor-product
//! smooth PIRLS as a *cell-local Gram factor* against a polynomial weight surface
//!
//!   G_{ij}^{(c)}  =  Σ_α w_{c,α} · M_α^{ij}(c).
//!
//! ## Math contract (block 5 section 1–9)
//!
//!   Per axis: on a half-open knot span [L, R] of width h = R − L, only the four
//!   cubic B-splines that "see" the span are active. Each active basis function
//!   restricted to the span is a degree-3 polynomial a₀ + a₁ u + a₂ u² + a₃ u³ in
//!   the local coordinate u = x − L. The 4 coefficient vectors come from the
//!   classical Cox-de Boor recurrence (section 2 of the math notes).
//!
//!   For an unordered active pair (i, j) the *product* B_i · B_j is degree 6 on
//!   the span: c₀ + c₁ u + … + c₆ u⁶ = (a^{(i)} ⊛ a^{(j)})(u).
//!
//!   1D closed form for the moment about m (section 1):
//!     I_ν^{ij}(m) = Σ_{s=0..ν} C(ν,s) (L−m)^{ν−s} · Σ_{q=0..6} c_q · h^{q+s+1}/(q+s+1)
//!
//!   In all of our consumers we use m = L (cell-local expansion), so
//!     I_ν^{ij}(L) = Σ_{q=0..6} c_q · h^{q+ν+1}/(q+ν+1).
//!   We keep the general (L−m)^{ν−s} expansion in CPU code for flexibility but
//!   the hot kernel path (NVRTC, Phase 2) always uses m = L.
//!
//!   Tensor cell on a hex (axis-aligned box, fully separable; section 9):
//!     M_α^{ij}(c) = Π_r I_{α_r}^{i_r j_r}(L_r).
//!
//! ## Derivative variants
//!
//! For a derivative-derivative moment ∫ B_i^{(ℓ₁)} · B_j^{(ℓ₂)} · (x−m)^ν dx, we
//! differentiate the *coefficient vector* a^{(i)} ℓ₁ times (degree drops by ℓ₁),
//! then convolve with the (possibly differentiated) a^{(j)} and feed the resulting
//! product polynomial of degree (6 − ℓ₁ − ℓ₂) into the same closed form. This is
//! how the mass/tension/stiffness penalty kernels share one shape with the
//! plain-moment kernel — the only thing that changes per kernel is the input
//! `prod_coeff` table built on the CPU.
//!
//! ## Sibling-agent boundary
//!
//! `src/families/cubic_cell_kernel.rs` contains *different* "cell moments" used
//! by the denested-cubic-transport row jet (`nvrtc-bms-flex`'s territory). Names
//! in this module are deliberately distinct (`cubic_bspline_cell_moments`,
//! `tensor_bspline_moment_table`) to avoid any collision or confusion.
//!
//! ## Task ledger (cubic-moments charter)
//!
//! - CM-P1 — hex CPU reference + Phase-1 host substrate. **DONE**.
//! - CM-P2 — hex NVRTC kernel + V100 parity bench. **DONE** (commit cd27ff0cf).
//! - CM-P3 — tetrahedral two-stage moment kernel (`tetrahedral_geom_moments_kernel`
//!   + `tetrahedral_contract_kernel`, host dispatcher
//!   `try_device_tetrahedral_moments`). **DONE** — see the CM-P3 section below.

#[cfg(target_os = "linux")]
use std::collections::HashMap;
#[cfg(target_os = "linux")]
use std::collections::hash_map::DefaultHasher;
#[cfg(target_os = "linux")]
use std::hash::{Hash, Hasher};
#[cfg(target_os = "linux")]
use std::sync::Mutex;
use std::sync::OnceLock;

#[cfg(target_os = "linux")]
use gam_gpu::gpu_err;
use gam_gpu::gpu_error::GpuError;
#[cfg(target_os = "linux")]
use gam_gpu::gpu_error::GpuResultExt;

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaContext, CudaModule, CudaStream};

// ────────────────────────────────────────────────────────────────────────
// Constants and small numeric helpers
// ────────────────────────────────────────────────────────────────────────

/// Cubic B-splines are degree 3. The product of two cubics on a single span is
/// degree 6, so each `prod_coeff` vector has 7 entries c_0..c_6.
pub const DEGREE: usize = 3;
/// Number of cubic basis functions that are simultaneously nonzero on one span.
pub const ACTIVE_PER_SPAN: usize = DEGREE + 1; // 4
/// Length of a product-polynomial coefficient vector on a single span.
pub const PROD_LEN: usize = 2 * DEGREE + 1; // 7
/// Number of unordered active pairs per span (4 × 5 / 2 = 10).
pub const PAIRS_PER_SPAN: usize = ACTIVE_PER_SPAN * (ACTIVE_PER_SPAN + 1) / 2; // 10

/// Pascal's triangle row 0..=8 (sufficient for any α we care about in practice).
/// Used to expand (L − m)^{ν − s} when m ≠ L in CPU code.
fn binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut acc: f64 = 1.0;
    for i in 0..k {
        acc = acc * (n - i) as f64 / (i + 1) as f64;
    }
    acc
}

/// Lower-triangular row-major index into a symmetric 4×4 active-pair table.
/// Returns the slot for the unordered pair (a, b) with 0 ≤ a, b < 4.
#[inline]
pub fn active_pair_index(a: usize, b: usize) -> usize {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    // 0=(0,0), 1=(0,1), 2=(1,1), 3=(0,2), 4=(1,2), 5=(2,2),
    // 6=(0,3), 7=(1,3), 8=(2,3), 9=(3,3)
    hi * (hi + 1) / 2 + lo
}

/// Total cells skipped when the span has zero width (collapsed knots).
/// We never emit moments for these — the integral is identically zero and a
/// zero-width entry would only poison the per-axis stride arithmetic downstream.
#[inline]
pub fn span_is_active(width: f64) -> bool {
    width > 0.0 && width.is_finite()
}

// ────────────────────────────────────────────────────────────────────────
// Cox-de Boor: active cubic basis polynomial coefficients on one span.
// ────────────────────────────────────────────────────────────────────────

/// Active cubic basis polynomials on a single non-degenerate span.
///
/// For knot vector `t` and an active span index `k` (meaning the integration
/// interval is `[t[k], t[k+1])`), exactly 4 cubic B-splines are nonzero:
/// indices `k-3, k-2, k-1, k`. This routine returns the polynomial coefficient
/// vectors of those 4 basis functions, expressed in the *local* coordinate
/// `u = x − t[k]` so values stay well-conditioned for narrow cells at large
/// absolute coordinates. (Section 2 of block-5 notes.)
///
/// `t` must have at least `k+4` and at least `k-3` entries in range, i.e. this
/// is the standard "interior" condition. Open uniform clamped end conditions
/// are handled by the caller by repeating the boundary knot the requisite
/// number of times before constructing this table.
pub fn cubic_basis_local_coeffs(t: &[f64], k: usize) -> [[f64; ACTIVE_PER_SPAN]; ACTIVE_PER_SPAN] {
    assert!(k + 4 <= t.len(), "knot index out of range");
    assert!(k >= 3, "need 3 left-side knots for cubic B-splines");

    // Cox-de Boor at degree p with the standard recurrence
    //   N_{i,0}(x) = 1 on [t_i, t_{i+1}), 0 else
    //   N_{i,p}(x) = (x - t_i)/(t_{i+p} - t_i) N_{i,p-1}(x)
    //              + (t_{i+p+1} - x)/(t_{i+p+1} - t_{i+1}) N_{i+1,p-1}(x)
    //
    // We evaluate each step as a polynomial in `u = x − t_k`. On the active
    // span [t_k, t_{k+1}) every basis function is a polynomial in u of degree
    // ≤ p, represented in monomial form `c_0 + c_1 u + c_2 u² + c_3 u³`.
    //
    // We track one polynomial coefficient vector per "currently active" basis
    // function. Start from p=0 (single piecewise constant N_{k,0} ≡ 1, i.e.
    // coefficients [1, 0, 0, 0]) and lift through p = 1, 2, 3.

    // tk = t_k = local origin.
    let tk = t[k];

    // poly[i] = monomial coeffs of N_{k - p + i, p} restricted to the span,
    // in u. We carry p + 1 active entries at each level; final level (p=3) has
    // 4 entries — indices correspond to basis functions B_{k-3..=k}.
    let zero: [f64; ACTIVE_PER_SPAN] = [0.0; ACTIVE_PER_SPAN];
    // Level p = 0: just one nonzero piece — N_{k,0} = 1.
    let mut level: Vec<[f64; ACTIVE_PER_SPAN]> = vec![{
        let mut v = zero;
        v[0] = 1.0;
        v
    }];

    for p in 1..=DEGREE {
        // New level has p+1 active basis functions: indices (k-p .. k).
        let mut next: Vec<[f64; ACTIVE_PER_SPAN]> = vec![zero; p + 1];

        for j in 0..=p {
            let i = k - p + j; // global basis index at this level
            // Term A: (x - t_i)/(t_{i+p} - t_i) N_{i,p-1}(x)
            // Note: N_{i,p-1} on this span is `level[j-1]` if it exists,
            // i.e. index j-1 in the previous level (which had p entries).
            if j >= 1 {
                let denom = t[i + p] - t[i];
                if denom > 0.0 {
                    // (x - t_i) = (u + (t_k - t_i)). Let shift = t_k - t_i.
                    let shift = tk - t[i];
                    let inv = 1.0 / denom;
                    let prev = &level[j - 1];
                    // multiply prev by (u + shift) · inv: result deg ≤ p.
                    for q in 0..p {
                        // shift · u^q term
                        next[j][q] += inv * shift * prev[q];
                        // u^{q+1} term from the `u·prev` factor
                        next[j][q + 1] += inv * prev[q];
                    }
                }
            }
            // Term B: (t_{i+p+1} - x)/(t_{i+p+1} - t_{i+1}) N_{i+1,p-1}(x)
            if j < p {
                let denom = t[i + p + 1] - t[i + 1];
                if denom > 0.0 {
                    // (t_{i+p+1} - x) = -(u - (t_{i+p+1} - t_k))
                    let shift = t[i + p + 1] - tk;
                    let inv = 1.0 / denom;
                    let prev = &level[j];
                    // multiply prev by (shift - u) · inv: result deg ≤ p.
                    for q in 0..p {
                        next[j][q] += inv * shift * prev[q];
                        next[j][q + 1] += -inv * prev[q];
                    }
                }
            }
        }

        level = next;
    }

    // level now has 4 vectors for B_{k-3..=k}, ordered by ascending global index.
    let mut out: [[f64; ACTIVE_PER_SPAN]; ACTIVE_PER_SPAN] =
        [[0.0; ACTIVE_PER_SPAN]; ACTIVE_PER_SPAN];
    for (idx, v) in level.into_iter().enumerate() {
        out[idx] = v;
    }
    out
}

/// Differentiate a monomial coefficient vector once: (a_0..a_3) → (a_1, 2 a_2, 3 a_3, 0).
/// Used to build derivative-derivative moment tables (mass/tension/stiffness).
pub fn differentiate_basis_coeffs(a: [f64; ACTIVE_PER_SPAN]) -> [f64; ACTIVE_PER_SPAN] {
    [a[1], 2.0 * a[2], 3.0 * a[3], 0.0]
}

/// Convolve two degree-≤3 coefficient vectors into a degree-≤6 product vector
/// (section 1 explicit formula). Branch-free, FMA-friendly.
#[inline]
pub fn convolve_basis_pair(
    a: [f64; ACTIVE_PER_SPAN],
    b: [f64; ACTIVE_PER_SPAN],
) -> [f64; PROD_LEN] {
    let mut c = [0.0; PROD_LEN];
    for i in 0..ACTIVE_PER_SPAN {
        if a[i] == 0.0 {
            continue;
        }
        for j in 0..ACTIVE_PER_SPAN {
            c[i + j] += a[i] * b[j];
        }
    }
    c
}

// ────────────────────────────────────────────────────────────────────────
// 1D closed-form moments
// ────────────────────────────────────────────────────────────────────────

/// Closed form for ∫_L^R (x − m)^ν · P(u) dx where P(u) = Σ_q c_q u^q and
/// u = x − L. Section 1 expansion.
///
/// When `m == L` (cell-local moments, our default), the formula collapses to
/// `Σ_q c_q · h^{q+ν+1} / (q + ν + 1)`.
pub fn moment_1d_about(c: [f64; PROD_LEN], width: f64, nu: usize, m_minus_left: f64) -> f64 {
    if !span_is_active(width) {
        return 0.0;
    }
    let mut acc = 0.0;
    let lm = -m_minus_left; // (L - m)
    for s in 0..=nu {
        let bin = binomial(nu, s);
        // f64::powi(0) returns 1.0 for any base including 0.0, so the s=nu
        // case lands correctly even when lm is zero (no special-case needed).
        let lm_pow = lm.powi((nu - s) as i32);
        // S_s = Σ_q c_q · h^{q+s+1} / (q + s + 1)
        let mut ss = 0.0;
        let mut h_pow = width.powi((s + 1) as i32);
        for q in 0..PROD_LEN {
            ss += c[q] * h_pow / ((q + s + 1) as f64);
            h_pow *= width;
        }
        acc += bin * lm_pow * ss;
    }
    acc
}

/// Convenience: 1D moment about the cell's left endpoint (m = L). This is the
/// integral the hot kernel evaluates per (cell, axis, pair, alpha) entry.
#[inline]
pub fn moment_1d_local(c: [f64; PROD_LEN], width: f64, nu: usize) -> f64 {
    if !span_is_active(width) {
        return 0.0;
    }
    let mut acc = 0.0;
    let mut h_pow = width.powi((nu + 1) as i32);
    for q in 0..PROD_LEN {
        acc += c[q] * h_pow / ((q + nu + 1) as f64);
        h_pow *= width;
    }
    acc
}

// ────────────────────────────────────────────────────────────────────────
// 20-point Gauss-Legendre reference (parity gate for the closed-form path)
// ────────────────────────────────────────────────────────────────────────

// Canonical 20-point Gauss-Legendre nodes/weights on [-1, 1] (Abramowitz &
// Stegun 25.4), shared with the bivariate-normal cell integrator. The single
// source of truth lives in `crate::cubic_cell_kernel`; this parity
// gate references it so the two cubic-cell consumers can never silently drift.
// 20 points integrate any polynomial of degree ≤ 39 exactly in finite
// arithmetic — far more than our degree-≤ (6 + ν) integrand needs.
use crate::cubic_cell_kernel::{GL20_NODES, GL20_WEIGHTS};

/// Gauss-Legendre reference: ∫_L^R (x − m)^ν · P(x − L) dx with P given by the
/// product-polynomial coefficient vector. Used solely as a parity gate.
pub fn moment_1d_gauss_legendre(
    c: [f64; PROD_LEN],
    left: f64,
    width: f64,
    nu: usize,
    m: f64,
) -> f64 {
    if !span_is_active(width) {
        return 0.0;
    }
    let half = 0.5 * width;
    let center = left + half;
    let mut acc = 0.0;
    for k in 0..20 {
        let x = center + half * GL20_NODES[k];
        let u = x - left;
        // Horner on c[0..=6]
        let mut p = c[PROD_LEN - 1];
        for q in (0..PROD_LEN - 1).rev() {
            p = p * u + c[q];
        }
        let mom = (x - m).powi(nu as i32);
        acc += GL20_WEIGHTS[k] * mom * p;
    }
    acc * half
}

// ────────────────────────────────────────────────────────────────────────
// Per-axis tables (Phase 1 CPU build).
// ────────────────────────────────────────────────────────────────────────

/// Per-axis precomputed tables shared by every consumer kernel for the cubic
/// B-spline moment family. The hot path uploads these (or their device twins)
/// once per fit and never rebuilds them per PIRLS iteration.
#[derive(Clone, Debug)]
pub struct AxisCubicMomentTables {
    /// Active knot spans in original order. Length = `n_active_spans`.
    /// `left[s] = t[k_s]`, `width[s] = t[k_s + 1] − t[k_s]`. Zero-width spans
    /// are dropped during construction.
    pub span_indices: Vec<usize>,
    pub left: Vec<f64>,
    pub width: Vec<f64>,
    /// Per-span product-polynomial coefficients for all 10 unordered active
    /// pairs. Stride: `prod_coeff[s * PAIRS_PER_SPAN * PROD_LEN + pair * PROD_LEN + q]`.
    pub prod_coeff: Vec<f64>,
    /// Derivative orders the table is built for. `(0, 0)` is the plain moment;
    /// `(1, 1)` is the tension-style ∫ B_i' B_j' table; `(2, 2)` is stiffness.
    pub derivative_left: u8,
    pub derivative_right: u8,
}

/// Cox-de Boor coefficients differentiated `d` times in-place (using the
/// monomial-derivative rule). Coefficients above degree (3 − d) are zero.
fn derive_basis_coeffs(mut a: [f64; ACTIVE_PER_SPAN], d: u8) -> [f64; ACTIVE_PER_SPAN] {
    for _ in 0..d {
        a = differentiate_basis_coeffs(a);
    }
    a
}

impl AxisCubicMomentTables {
    /// Build the per-axis table for a single cubic-B-spline axis given an open
    /// knot vector `t` (clamped, repeated end knots are caller-supplied).
    /// `derivative_left`/`derivative_right` give the order of differentiation
    /// applied to the *first* / *second* basis in each pair.
    pub fn build(t: &[f64], derivative_left: u8, derivative_right: u8) -> Self {
        assert!(
            t.len() >= 2 * DEGREE + 2,
            "knot vector too short for cubic B-splines: got {} knots, need ≥ {}",
            t.len(),
            2 * DEGREE + 2
        );
        // Active spans are k = DEGREE..(t.len() - DEGREE - 1) with positive width.
        let mut span_indices = Vec::new();
        let mut left = Vec::new();
        let mut width = Vec::new();
        let mut prod_coeff = Vec::new();

        for k in DEGREE..(t.len() - DEGREE - 1) {
            let w = t[k + 1] - t[k];
            if !span_is_active(w) {
                continue;
            }
            let basis = cubic_basis_local_coeffs(t, k);
            // Apply derivative orders before convolution; the math notes
            // explicitly require this so the product polynomial stays in the
            // closed-form path (degree drops, but the convolution shape is
            // unchanged).
            let mut left_basis = basis;
            let mut right_basis = basis;
            for a in left_basis.iter_mut() {
                *a = derive_basis_coeffs(*a, derivative_left);
            }
            for a in right_basis.iter_mut() {
                *a = derive_basis_coeffs(*a, derivative_right);
            }

            let mut span_prod = [[0.0f64; PROD_LEN]; PAIRS_PER_SPAN];
            for a in 0..ACTIVE_PER_SPAN {
                for b in a..ACTIVE_PER_SPAN {
                    let pair_idx = active_pair_index(a, b);
                    // For asymmetric derivative orders (e.g. (1, 0)) the
                    // product B_i^{(1)} B_j^{(0)} differs from B_i^{(0)} B_j^{(1)};
                    // we store the canonical ordering (left = a, right = b) and
                    // require callers that swap to apply the same convention.
                    span_prod[pair_idx] = convolve_basis_pair(left_basis[a], right_basis[b]);
                }
            }

            span_indices.push(k);
            left.push(t[k]);
            width.push(w);
            prod_coeff.extend(span_prod.iter().flatten().copied());
        }

        Self {
            span_indices,
            left,
            width,
            prod_coeff,
            derivative_left,
            derivative_right,
        }
    }

    pub fn n_spans(&self) -> usize {
        self.left.len()
    }

    /// Retrieve the product-polynomial coefficient vector for a span/pair.
    pub fn prod(&self, span: usize, pair: usize) -> [f64; PROD_LEN] {
        let off = (span * PAIRS_PER_SPAN + pair) * PROD_LEN;
        let mut out = [0.0; PROD_LEN];
        out.copy_from_slice(&self.prod_coeff[off..off + PROD_LEN]);
        out
    }

    /// Closed-form 1D moment I_ν^{ij}(L) for the given span and pair.
    pub fn moment_local(&self, span: usize, pair: usize, nu: usize) -> f64 {
        moment_1d_local(self.prod(span, pair), self.width[span], nu)
    }
}

// ────────────────────────────────────────────────────────────────────────
// Tensor (hexahedral) moments — CPU reference for Phase 4 parity.
// ────────────────────────────────────────────────────────────────────────

/// Moment layout on disk / in device memory. Alpha-major keeps reads coalesced
/// when consumer kernels iterate alpha as the outermost loop (matches the
/// PIRLS contraction pattern G_{ij}^{(c)} = Σ_α w_{c,α} M_α^{ij}(c)).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MomentLayout {
    AlphaMajor,
}

/// Public spec for a tensor cubic-B-spline moment table.
///
/// `alphas[i]` lists the multi-index of moment exponents for output slot `i`.
/// `derivative_left[i]` / `derivative_right[i]` are per-axis derivative orders
/// for the (left, right) basis in the pair — one ℓ value per axis. The CPU and
/// GPU paths share this spec verbatim.
#[derive(Clone, Debug)]
pub struct CubicMomentSpec {
    pub alphas: Vec<Vec<u8>>,
    pub derivative_left: Vec<Vec<u8>>,
    pub derivative_right: Vec<Vec<u8>>,
    pub layout: MomentLayout,
}

impl CubicMomentSpec {
    pub fn d(&self) -> usize {
        self.alphas.first().map(|v| v.len()).unwrap_or(0)
    }

    pub fn n_alpha(&self) -> usize {
        self.alphas.len()
    }
}

/// Build the per-axis tables required for `spec` from a list of per-axis knot
/// vectors. Returns one `AxisCubicMomentTables` per axis per *distinct*
/// derivative signature actually requested by `spec`.
pub fn build_axis_tables_cpu(
    spec: &CubicMomentSpec,
    knots_per_axis: &[Vec<f64>],
) -> Vec<Vec<AxisCubicMomentTables>> {
    assert_eq!(spec.d(), knots_per_axis.len(), "axis count mismatch");
    let d = spec.d();
    let mut out = Vec::with_capacity(d);
    for axis in 0..d {
        // Distinct (deriv_left[axis], deriv_right[axis]) pairs across all alphas.
        let mut sigs: Vec<(u8, u8)> = (0..spec.n_alpha())
            .map(|i| {
                (
                    spec.derivative_left[i][axis],
                    spec.derivative_right[i][axis],
                )
            })
            .collect();
        sigs.sort_unstable();
        sigs.dedup();
        let mut axis_tables = Vec::with_capacity(sigs.len());
        for (dl, dr) in sigs {
            axis_tables.push(AxisCubicMomentTables::build(&knots_per_axis[axis], dl, dr));
        }
        out.push(axis_tables);
    }
    out
}

/// CPU reference: compute M_α^{ij}(c) for one tensor hex cell from the axis
/// tables, separably. `cell_span[axis]` indexes into the *active* span array of
/// that axis (i.e. the same indexing the device buffer uses).
pub fn tensor_hex_moment_cpu(
    axis_tables: &[&AxisCubicMomentTables],
    cell_span: &[usize],
    alpha: &[u8],
    pair_per_axis: &[usize],
) -> f64 {
    assert_eq!(axis_tables.len(), cell_span.len());
    assert_eq!(axis_tables.len(), alpha.len());
    assert_eq!(axis_tables.len(), pair_per_axis.len());
    let mut prod = 1.0;
    for r in 0..axis_tables.len() {
        let i_r = axis_tables[r].moment_local(cell_span[r], pair_per_axis[r], alpha[r] as usize);
        prod *= i_r;
        if prod == 0.0 {
            return 0.0;
        }
    }
    prod
}

// ────────────────────────────────────────────────────────────────────────
// Device-resident output (Phase 2 will populate via NVRTC kernel).
// ────────────────────────────────────────────────────────────────────────

/// Sized handle to the device-resident tensor moment table. Phase 1 only
/// materialises the host-side metadata; the device buffer is owned by Phase 2.
#[derive(Debug)]
pub struct DeviceCubicMomentTable {
    pub n_cells: usize,
    pub pair_tuple_count: usize,
    pub n_alpha: usize,
    pub layout: MomentLayout,
    #[cfg(target_os = "linux")]
    pub values: cudarc::driver::CudaSlice<f64>,
    #[cfg(not(target_os = "linux"))]
    pub values: Vec<f64>,
}

/// Sized handle to the per-axis 1D marginal moment table (factored consumer
/// path). Same layout convention: alpha-major along moment exponent ν.
#[derive(Debug)]
pub struct DeviceMarginalTable {
    pub n_axes: usize,
    pub n_spans_per_axis: Vec<usize>,
    pub n_alpha_per_axis: Vec<usize>,
    #[cfg(target_os = "linux")]
    pub values: cudarc::driver::CudaSlice<f64>,
    #[cfg(not(target_os = "linux"))]
    pub values: Vec<f64>,
}

// ────────────────────────────────────────────────────────────────────────
// CUDA backend handle, module cache key, and NVRTC kernel source generator
// for the hexahedral tensor-moment kernel (Phase 2).
// ────────────────────────────────────────────────────────────────────────

/// NVRTC module-cache key. The module is specialised at compile-time by
/// (D, AMAX, NALPHA, hashed alpha/derivative tables, output layout, CC) so a
/// re-fit with the same spec resolves to a cache hit.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct HexMomentModuleKey {
    cc_major: i32,
    cc_minor: i32,
    d: u32,
    amax: u32,
    nalpha: u32,
    alpha_hash: u64,
    deriv_hash: u64,
    layout_tag: u8,
}

/// 64-bit FNV-style hash of the alpha multi-index table. Stable across runs;
/// used as part of the module cache key so two specs with the same alpha grid
/// share one NVRTC compile.
#[cfg(target_os = "linux")]
fn hash_alpha_table(alphas: &[Vec<u8>]) -> u64 {
    let mut h = DefaultHasher::new();
    (alphas.len() as u64).hash(&mut h);
    for row in alphas {
        (row.len() as u64).hash(&mut h);
        for &v in row {
            v.hash(&mut h);
        }
    }
    h.finish()
}

#[cfg(target_os = "linux")]
fn hash_deriv_table(deriv_left: &[Vec<u8>], deriv_right: &[Vec<u8>]) -> u64 {
    let mut h = DefaultHasher::new();
    (deriv_left.len() as u64).hash(&mut h);
    for row in deriv_left {
        (row.len() as u64).hash(&mut h);
        for &v in row {
            v.hash(&mut h);
        }
    }
    (deriv_right.len() as u64).hash(&mut h);
    for row in deriv_right {
        (row.len() as u64).hash(&mut h);
        for &v in row {
            v.hash(&mut h);
        }
    }
    h.finish()
}

#[inline]
#[cfg(target_os = "linux")]
fn layout_tag(layout: MomentLayout) -> u8 {
    match layout {
        MomentLayout::AlphaMajor => 0,
    }
}

/// Generate the NVRTC C++ source for the hex tensor moment kernel,
/// specialised to the spec's `(D, AMAX, NALPHA)` triple. The alpha multi-index
/// table is baked in as a `__constant__` array so the inner loop unrolls.
///
/// Math (section 9): for each cell c, output slot α, ordered pair-tuple
/// (i_r, j_r) per axis,
///
///   M_α^{ij}(c) = Π_{r=0..D-1} I_{α_r}^{i_r j_r}(L_r)
///
/// with the cell-local 1D moment
///
///   I_ν^{ij}(L) = Σ_{q=0..6} c_q · h^{q+ν+1} / (q + ν + 1).
///
/// `alpha_table[NALPHA][D]` carries the exponent multi-indices for the spec.
/// Output is alpha-major with stride `((n_cells+31)/32)*32` so consecutive
/// threads write coalesced f64s on Volta+.
#[cfg(target_os = "linux")]
fn build_hex_tensor_kernel_source(d: usize, amax: usize, alphas: &[Vec<u8>]) -> String {
    let nalpha = alphas.len();
    let mut alpha_decl = String::new();
    alpha_decl.push_str("__constant__ unsigned char ALPHA_TABLE[NALPHA][D] = {\n");
    for row in alphas {
        alpha_decl.push_str("    { ");
        for (k, v) in row.iter().enumerate() {
            if k > 0 {
                alpha_decl.push_str(", ");
            }
            alpha_decl.push_str(&v.to_string());
        }
        alpha_decl.push_str(" },\n");
    }
    alpha_decl.push_str("};\n");

    format!(
        r#"
#define D       {d}
#define AMAX    {amax}
#define NALPHA  {nalpha}
#define PROD_LEN 7

{alpha_decl}

// Closed-form 1D moment about the cell's left endpoint (m = L):
//   I_nu^{{ij}}(L) = sum_{{q=0..6}} c_q * h^{{q+nu+1}} / (q + nu + 1)
// Implemented with a Horner-style accumulating `h_pow` and fma. We loop on
// nu_plus_one = nu + 1 up to the compile-time AMAX so the divides are
// constant-foldable.
__device__ __forceinline__ double moment_1d_local(
    const double *cprod,   // [PROD_LEN], product-poly coefs on cell
    double        h,       // cell width
    unsigned int  nu       // moment exponent
) {{
    double h_pow = 1.0;
    for (unsigned int s = 0; s <= nu; ++s) {{
        h_pow *= h;             // after loop: h_pow == h^{{nu+1}} ... continues below
    }}
    double acc = 0.0;
    #pragma unroll
    for (int q = 0; q < PROD_LEN; ++q) {{
        double denom = (double)(q + nu + 1);
        acc = fma(cprod[q] / denom, h_pow, acc);
        h_pow *= h;
    }}
    return acc;
}}

// One thread = one (cell, alpha-slot). Block (32, 8, 1): x → cell, y → alpha.
// Inputs (all device-resident):
//   axis_prod_coeff_flat: f64 buffer, concatenated per-axis tables; offset
//       per axis given by `axis_offset[axis]` (in elements). Each axis table
//       is [n_spans_axis][PAIRS_PER_SPAN(=10)][PROD_LEN(=7)] row-major.
//   axis_offset:          i64[D]
//   cell_span_per_axis:   i32[n_cells * D] — active-span index per axis.
//   cell_pair_per_axis:   i32[n_cells * D] — unordered-pair slot (0..=9) per axis.
//   cell_width_per_axis:  f64[n_cells * D] — cell width per axis.
// Output:
//   out: f64[NALPHA * out_stride], alpha-major; thread writes out[a*out_stride + c].
extern "C" __global__ void cubic_hex_tensor_moments(
    const double *axis_prod_coeff_flat,
    const long long *axis_offset,
    const int *cell_span_per_axis,
    const int *cell_pair_per_axis,
    const double *cell_width_per_axis,
    int          n_cells,
    long long    out_stride,
    double      *out
) {{
    const int cell  = blockIdx.x * blockDim.x + threadIdx.x;
    const int alpha = blockIdx.y * blockDim.y + threadIdx.y;
    if (cell >= n_cells || alpha >= NALPHA) return;

    double prod = 1.0;
    #pragma unroll
    for (int r = 0; r < D; ++r) {{
        const int   span_r  = cell_span_per_axis[cell * D + r];
        const int   pair_r  = cell_pair_per_axis[cell * D + r];
        const double width_r = cell_width_per_axis[cell * D + r];
        const long long base = axis_offset[r]
            + (long long)span_r * 10LL * (long long)PROD_LEN
            + (long long)pair_r * (long long)PROD_LEN;
        const double *cprod = axis_prod_coeff_flat + base;
        const unsigned int nu = (unsigned int)ALPHA_TABLE[alpha][r];
        const double mu = moment_1d_local(cprod, width_r, nu);
        prod *= mu;
    }}

    out[(long long)alpha * out_stride + (long long)cell] = prod;
}}
"#,
        d = d,
        amax = amax,
        nalpha = nalpha,
        alpha_decl = alpha_decl,
    )
}

/// Probe handle for the cubic-B-spline moments GPU backend. Holds the CUDA
/// context, default stream, capability tag, and the NVRTC module cache so
/// re-fits with the same `CubicMomentSpec` resolve to a cache hit.
#[cfg(target_os = "linux")]
struct CubicMomentBackendInner {
    ctx: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<CudaStream>,
    modules: Mutex<HashMap<HexMomentModuleKey, std::sync::Arc<CudaModule>>>,
    tet_modules: Mutex<HashMap<TetMomentModuleKey, std::sync::Arc<CudaModule>>>,
    cc_major: i32,
    cc_minor: i32,
}

#[must_use]
pub struct CubicMomentBackend {
    #[cfg(target_os = "linux")]
    inner: CubicMomentBackendInner,
}

impl CubicMomentBackend {
    pub const fn compiled() -> bool {
        cfg!(target_os = "linux")
    }

    /// Download an alpha-major `[NALPHA, n_cells]` device-resident moment
    /// table into a host `Vec<f64>`. Provided for bench-side parity checks
    /// (see `bench/cargo_benches/cubic_hex_tensor_gpu_parity.rs`) and any
    /// future debugging tool that wants to inspect the kernel output without
    /// reaching past the backend's privacy boundary. On non-Linux hosts the
    /// `DeviceCubicMomentTable.values` field is already a `Vec<f64>` so the
    /// call is a copy; on Linux it issues one DtoH and a stream sync.
    pub fn download_alpha_major(&self, dev: &DeviceCubicMomentTable) -> Result<Vec<f64>, GpuError> {
        #[cfg(target_os = "linux")]
        {
            let stream = &self.inner.stream;
            let host = stream
                .clone_dtoh(&dev.values)
                .gpu_ctx("cubic_bspline_moments download_alpha_major dtov")?;
            stream
                .synchronize()
                .gpu_ctx("cubic_bspline_moments download_alpha_major sync")?;
            Ok(host)
        }
        #[cfg(not(target_os = "linux"))]
        {
            Ok(dev.values.clone())
        }
    }

    pub fn probe() -> Result<&'static Self, GpuError> {
        static BACKEND: OnceLock<Result<CubicMomentBackend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                #[cfg(target_os = "linux")]
                {
                    Self::probe_linux()
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Err(GpuError::DriverLibraryUnavailable {
                        reason: "cubic_bspline_moments GPU backend is Linux-only".to_string(),
                    })
                }
            })
            .as_ref()
            .map_err(GpuError::clone)
    }

    #[cfg(target_os = "linux")]
    fn probe_linux() -> Result<Self, GpuError> {
        let parts = gam_gpu::backend_probe::probe_cuda_backend("cubic_bspline_moments")?;
        Ok(CubicMomentBackend {
            inner: CubicMomentBackendInner {
                ctx: parts.ctx,
                stream: parts.stream,
                modules: Mutex::new(HashMap::new()),
                tet_modules: Mutex::new(HashMap::new()),
                cc_major: parts.capability.compute_major,
                cc_minor: parts.capability.compute_minor,
            },
        })
    }

    #[cfg(target_os = "linux")]
    fn tet_module_for(
        &self,
        key: TetMomentModuleKey,
        src_factory: impl FnOnce() -> String,
    ) -> Result<std::sync::Arc<CudaModule>, GpuError> {
        if let Ok(guard) = self.inner.tet_modules.lock() {
            if let Some(existing) = guard.get(&key) {
                return Ok(existing.clone());
            }
        }
        let src = src_factory();
        // Shared arch+fmad options (NOT bare `compile_ptx`): #1686's
        // `--fmad=false` keeps the GL-quadrature moment reductions
        // bit-comparable to the separately-rounded CPU reference, and the #1551
        // arch pin keys the kernel to the device's real compute capability.
        let ptx = gam_gpu::device_cache::compile_ptx_arch(&src).gpu_ctx_with(|err| {
            format!(
                "tetrahedral_moments NVRTC compile (D={}, NBETA={}, NALPHA={}): {err}",
                key.d, key.nbeta, key.nalpha
            )
        })?;
        let module = self
            .inner
            .ctx
            .load_module(ptx)
            .gpu_ctx("tetrahedral_moments module load")?;
        if let Ok(mut guard) = self.inner.tet_modules.lock() {
            guard.entry(key).or_insert_with(|| module.clone());
        }
        Ok(module)
    }

    #[cfg(target_os = "linux")]
    fn module_for(
        &self,
        key: HexMomentModuleKey,
        src_factory: impl FnOnce() -> String,
    ) -> Result<std::sync::Arc<CudaModule>, GpuError> {
        if let Ok(guard) = self.inner.modules.lock() {
            if let Some(existing) = guard.get(&key) {
                return Ok(existing.clone());
            }
        }
        let src = src_factory();
        // Shared arch+fmad options (NOT bare `compile_ptx`): #1686's
        // `--fmad=false` keeps the hex-tensor moment reductions bit-comparable
        // to the separately-rounded CPU reference, and the #1551 arch pin keys
        // the kernel to the device's real compute capability.
        let ptx = gam_gpu::device_cache::compile_ptx_arch(&src).gpu_ctx_with(|err| {
            format!(
                "cubic_bspline_moments NVRTC compile (D={}, AMAX={}, NALPHA={}): {err}",
                key.d, key.amax, key.nalpha
            )
        })?;
        let module = self
            .inner
            .ctx
            .load_module(ptx)
            .gpu_ctx("cubic_bspline_moments module load")?;
        if let Ok(mut guard) = self.inner.modules.lock() {
            guard.entry(key).or_insert_with(|| module.clone());
        }
        Ok(module)
    }
}

// ────────────────────────────────────────────────────────────────────────
// Device-resident hex tensor moment build (Phase 2 entry point).
// ────────────────────────────────────────────────────────────────────────

/// Per-cell descriptor for the hex tensor build: which active span and which
/// unordered active-pair slot to use on each axis, plus the cell width per
/// axis. The width is carried explicitly so the kernel never has to chase a
/// second indirection just to read it back.
#[derive(Clone, Debug)]
pub struct HexCellTable {
    /// `span_per_axis[cell * d + axis]` — active-span index on `axis`.
    pub span_per_axis: Vec<i32>,
    /// `pair_per_axis[cell * d + axis]` — `active_pair_index(i_axis, j_axis)`.
    pub pair_per_axis: Vec<i32>,
    /// `width_per_axis[cell * d + axis]` — `t[k+1] − t[k]` for that span.
    pub width_per_axis: Vec<f64>,
    pub n_cells: usize,
    pub d: usize,
}

impl HexCellTable {
    pub fn validate(&self) -> Result<(), GpuError> {
        let want = self.n_cells * self.d;
        if self.span_per_axis.len() != want
            || self.pair_per_axis.len() != want
            || self.width_per_axis.len() != want
        {
            gam_gpu::gpu_bail!(
                "HexCellTable: expected length {want} (n_cells*d), got span={}, pair={}, width={}",
                self.span_per_axis.len(),
                self.pair_per_axis.len(),
                self.width_per_axis.len(),
            );
        }
        Ok(())
    }
}

/// Build the alpha-major `[NALPHA, n_cells]` hex tensor moment table on the
/// device. Per-axis tables must be supplied in a fixed (axis × derivative-sig)
/// order matching `build_axis_tables_cpu`; the consumer picks one table per
/// alpha slot through `derivative_left` / `derivative_right` (the kernel
/// itself is derivative-agnostic — derivative orders are baked into the
/// product-poly coefficients of each axis table on the CPU).
///
/// For now Phase 2 requires that the spec uses **one** derivative signature
/// per axis (the homogeneous-deriv case driven by the PIRLS consumer). Mixed
/// signatures fall back to the CPU path until Phase 3 adds the
/// derivative-bank kernel; this matches what the math notes call the
/// "single-bank" hex specialisation.
#[cfg(target_os = "linux")]
pub fn build_hex_tensor_moments_device(
    spec: &CubicMomentSpec,
    axis_tables: &[Vec<AxisCubicMomentTables>],
    cells: &HexCellTable,
) -> Result<DeviceCubicMomentTable, GpuError> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    cells.validate()?;
    if spec.d() != cells.d {
        gam_gpu::gpu_bail!(
            "build_hex_tensor_moments_device: spec.d()={} != cells.d={}",
            spec.d(),
            cells.d
        );
    }
    if axis_tables.len() != cells.d {
        gam_gpu::gpu_bail!(
            "build_hex_tensor_moments_device: axis_tables.len()={} != d={}",
            axis_tables.len(),
            cells.d
        );
    }
    // This device kernel covers the single-bank layout (exactly one derivative
    // signature per axis). Multi-bank (mixed-signature) axes are served by the
    // CPU path, which is the correct reference computation.
    for (axis, banks) in axis_tables.iter().enumerate() {
        if banks.len() != 1 {
            return Err(GpuError::NoDeviceKernel {
                reason: format!(
                    "build_hex_tensor_moments_device: axis {axis} has {} derivative banks; \
                     this device kernel covers the single-bank layout — multi-bank axes use \
                     the CPU path",
                    banks.len()
                ),
            });
        }
    }
    let nalpha = spec.n_alpha();
    if nalpha == 0 || cells.n_cells == 0 {
        return Err(GpuError::DriverCallFailed {
            reason: "build_hex_tensor_moments_device: empty spec or cell list".to_string(),
        });
    }
    let amax = spec
        .alphas
        .iter()
        .flat_map(|row| row.iter().copied())
        .max()
        .unwrap_or(0) as usize;

    let backend = CubicMomentBackend::probe()?;
    let key = HexMomentModuleKey {
        cc_major: backend.inner.cc_major,
        cc_minor: backend.inner.cc_minor,
        d: cells.d as u32,
        amax: amax as u32,
        nalpha: nalpha as u32,
        alpha_hash: hash_alpha_table(&spec.alphas),
        deriv_hash: hash_deriv_table(&spec.derivative_left, &spec.derivative_right),
        layout_tag: layout_tag(spec.layout),
    };
    let d_for_src = cells.d;
    let alphas_for_src = spec.alphas.clone();
    let module = backend.module_for(key, move || {
        build_hex_tensor_kernel_source(d_for_src, amax, &alphas_for_src)
    })?;
    let func = module
        .load_function("cubic_hex_tensor_moments")
        .gpu_ctx("cubic_bspline_moments load_function")?;
    let stream = backend.inner.stream.clone();

    // Concatenate per-axis tables into one device-side f64 buffer and record
    // per-axis offsets (in elements).
    let mut axis_offsets: Vec<i64> = Vec::with_capacity(cells.d);
    let mut flat: Vec<f64> = Vec::new();
    for banks in axis_tables.iter() {
        axis_offsets.push(flat.len() as i64);
        flat.extend_from_slice(&banks[0].prod_coeff);
    }

    let axis_flat_dev = stream
        .clone_htod(flat.as_slice())
        .gpu_ctx("cubic_bspline_moments htod axis_flat")?;
    let axis_off_dev = stream
        .clone_htod(axis_offsets.as_slice())
        .gpu_ctx("cubic_bspline_moments htod axis_offsets")?;
    let span_dev = stream
        .clone_htod(cells.span_per_axis.as_slice())
        .gpu_ctx("cubic_bspline_moments htod span")?;
    let pair_dev = stream
        .clone_htod(cells.pair_per_axis.as_slice())
        .gpu_ctx("cubic_bspline_moments htod pair")?;
    let width_dev = stream
        .clone_htod(cells.width_per_axis.as_slice())
        .gpu_ctx("cubic_bspline_moments htod width")?;

    let out_stride = ((cells.n_cells + 31) / 32) * 32;
    let mut out_dev = stream
        .alloc_zeros::<f64>(out_stride * nalpha)
        .gpu_ctx_with(|err| {
            format!("cubic_bspline_moments alloc out (stride={out_stride}, nalpha={nalpha}): {err}")
        })?;

    let block_x: u32 = 32;
    let block_y: u32 = 8;
    let grid_x: u32 = ((cells.n_cells as u32) + block_x - 1) / block_x;
    let grid_y: u32 = ((nalpha as u32) + block_y - 1) / block_y;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (block_x, block_y, 1),
        shared_mem_bytes: 0,
    };

    let n_cells_i32: i32 = i32::try_from(cells.n_cells).map_err(|_| {
        gpu_err!(
            "cubic_bspline_moments n_cells={} overflows i32",
            cells.n_cells
        )
    })?;
    let out_stride_i64: i64 = out_stride as i64;

    let mut builder = stream.launch_builder(&func);
    builder
        .arg(&axis_flat_dev)
        .arg(&axis_off_dev)
        .arg(&span_dev)
        .arg(&pair_dev)
        .arg(&width_dev)
        .arg(&n_cells_i32)
        .arg(&out_stride_i64)
        .arg(&mut out_dev);
    // SAFETY: all pointers come from cudarc-checked allocations on `stream`;
    // the kernel reads inputs of declared sizes and writes within
    // `out[0 .. out_stride * nalpha]`. Launch dims are non-zero and bounded
    // by the per-axis i32 cap above.
    unsafe { builder.launch(cfg) }.gpu_ctx("cubic_bspline_moments kernel launch")?;
    stream
        .synchronize()
        .gpu_ctx("cubic_bspline_moments synchronize")?;

    Ok(DeviceCubicMomentTable {
        n_cells: cells.n_cells,
        pair_tuple_count: 1,
        n_alpha: nalpha,
        layout: spec.layout,
        values: out_dev,
    })
}

/// macOS-side stub so the Phase 2 entry point is callable on every host;
/// returns the same `DriverLibraryUnavailable` error as the backend probe.
/// Params are referenced in the error message so the non-Linux build does
/// not silently strip them via `_`-prefixed silencer names.
#[cfg(not(target_os = "linux"))]
pub fn build_hex_tensor_moments_device(
    spec: &CubicMomentSpec,
    axis_tables: &[Vec<AxisCubicMomentTables>],
    cells: &HexCellTable,
) -> Result<DeviceCubicMomentTable, GpuError> {
    Err(GpuError::DriverLibraryUnavailable {
        reason: format!(
            "cubic_bspline_moments GPU backend is Linux-only \
             (layout={:?}, axis_tables_axes={}, n_cells={})",
            spec.layout,
            axis_tables.len(),
            cells.n_cells,
        ),
    })
}

// ────────────────────────────────────────────────────────────────────────
// CM-P3 — Tetrahedral two-stage moment kernel.
//
// The hex path (above) is restricted to axis-aligned tensor-product cells:
// per-axis cubic-B-spline closed forms compose into the cell moment as a
// product of 1D integrals. For non-axis-aligned partitions (e.g. cells
// emitted by an unstructured Delaunay/Lloyd mesher, or any setting where
// the basis Gram is not separable into a per-axis tensor) we need a path
// that integrates over an arbitrary affine simplex.
//
// Math contract — geometric monomial moments on a tetrahedron T with
// vertices v0,…,v3 ∈ R^D and a per-cell reference point c0:
//
//   G_β(T)  =  ∫_T (x − c0)^β dx,        β ∈ N^D, |β| ≤ AMAX_GEOM.
//
// Map the reference simplex T_ref = {u ∈ R^3 : u_i ≥ 0, u1+u2+u3 ≤ 1}
// affinely onto T via x(u) = v0 + B·u with B = [v1−v0 | v2−v0 | v3−v0]
// (column-major). The Jacobian is constant: |det B| = 6·Vol(T). With
// q_r = v0,r − c0,r and e_{i,r} = v_{i+1},r − v0,r,
//
//   (x_r − c0,r)^{β_r}
//      = Σ_{κ_0+…+κ_3 = β_r} (β_r! / (κ_0! κ_1! κ_2! κ_3!))
//                              · q_r^{κ_0}
//                              · Π_{i=1..3} (e_{i,r} u_i)^{κ_i}.
//
// Taking the product over r=0..D−1 expands (x − c0)^β into a polynomial
// in (u_1, u_2, u_3) with coefficients that are products of (q_r, e_{i,r}).
// Each monomial u_1^{n_1} u_2^{n_2} u_3^{n_3} integrates over T_ref to
//
//   ∫_{T_ref} u_1^{n_1} u_2^{n_2} u_3^{n_3} du  =  n_1! n_2! n_3! / (n_1+n_2+n_3+3)!
//
// (the Dirichlet / Lasserre–Avrachenkov closed form), and contributes a
// factor of |det B| to the world-space integral. To express the inner
// per-axis lift compactly we use an "affine T_n recurrence" mirroring the
// 1D moment table: define T_{β_r}(q_r; e_{·,r}) as the polynomial in
// (u_1,u_2,u_3) given by (q_r + e_{1,r} u_1 + e_{2,r} u_2 + e_{3,r} u_3)^{β_r};
// then T_{β_r} = T_{β_r − 1} · (q_r + Σ_i e_{i,r} u_i). The kernel walks
// β_r from 1..=AMAX_GEOM_AXIS and accumulates products across axes
// directly into the geom-moment slot.
//
// Stage 2 — basis-Gram contraction:
//
//   M_α^{ij}(c) = Σ_{T ∈ c} Σ_β  W_{α,β}^{ij}(c) · G_β(T),
//
// where W is the caller-supplied basis-Gram weight tensor (depends only on
// the cell — same value for every tetrahedron in the cell). The kernel
// reads the per-tet G_β table emitted by stage 1, the per-cell weight
// tensor uploaded once, and emits the same alpha-major output shape as
// the hex kernel so the two paths are drop-in interchangeable downstream.
// ────────────────────────────────────────────────────────────────────────

/// One tetrahedron in R^D. Vertices are stored as a flat 4·D-vector in
/// vertex-major order (v0_0..v0_{D-1}, v1_0..v1_{D-1}, ...). `cell_index`
/// links the tetrahedron back to the logical cell whose moment slot the
/// contraction kernel will accumulate into; multiple tetrahedra may share
/// the same `cell_index`. `cell_center_offset` provides the per-cell
/// expansion point c0 used by the geometric moment integrand (x − c0)^β;
/// the kernel reads `cell_centers` at this offset (in elements: D doubles).
#[derive(Clone, Debug)]
pub struct TetrahedralCellTable {
    /// `vertices[tet * 4 * D + v * D + r]` — coordinate r of vertex v in tet.
    pub vertices: Vec<f64>,
    /// `cell_index[tet]` — logical cell this tet contributes to (0..n_cells).
    pub cell_index: Vec<i32>,
    /// `cell_centers[cell * D + r]` — expansion point c0 for cell `cell`.
    pub cell_centers: Vec<f64>,
    pub n_tets: usize,
    pub n_cells: usize,
    pub d: usize,
}

impl TetrahedralCellTable {
    pub fn validate(&self) -> Result<(), GpuError> {
        let want_v = self.n_tets * 4 * self.d;
        if self.vertices.len() != want_v {
            gam_gpu::gpu_bail!(
                "TetrahedralCellTable: expected vertices len {want_v} (n_tets*4*d), got {}",
                self.vertices.len()
            );
        }
        if self.cell_index.len() != self.n_tets {
            gam_gpu::gpu_bail!(
                "TetrahedralCellTable: cell_index len {} != n_tets {}",
                self.cell_index.len(),
                self.n_tets
            );
        }
        if self.cell_centers.len() != self.n_cells * self.d {
            gam_gpu::gpu_bail!(
                "TetrahedralCellTable: cell_centers len {} != n_cells*d {}",
                self.cell_centers.len(),
                self.n_cells * self.d
            );
        }
        for (i, &c) in self.cell_index.iter().enumerate() {
            if c < 0 || (c as usize) >= self.n_cells {
                gam_gpu::gpu_bail!(
                    "TetrahedralCellTable: cell_index[{i}] = {c} out of range [0, {})",
                    self.n_cells
                );
            }
        }
        Ok(())
    }
}

/// Public spec for the tetrahedral two-stage moment table.
///
/// `geom_betas[g][r]` is the multi-index β ∈ N^D for geometric-moment slot g
/// (|β| ≤ AMAX_GEOM). `alphas` and `layout` match the hex spec — the
/// contraction kernel emits the same alpha-major `[NALPHA, n_cells]` output.
/// `pairs_per_cell` is the number of (i, j) basis-pair slots the consumer
/// expects per cell (typically `PAIRS_PER_SPAN^D` for tensor-product bases,
/// but the kernel only consumes it as a stride and accepts any value).
#[derive(Clone, Debug)]
pub struct TetrahedralMomentSpec {
    pub geom_betas: Vec<Vec<u8>>,
    pub alphas: Vec<Vec<u8>>,
    pub pairs_per_cell: usize,
    pub layout: MomentLayout,
}

impl TetrahedralMomentSpec {
    pub fn d(&self) -> usize {
        self.alphas
            .first()
            .map(|v| v.len())
            .or_else(|| self.geom_betas.first().map(|v| v.len()))
            .unwrap_or(0)
    }

    pub fn n_alpha(&self) -> usize {
        self.alphas.len()
    }

    pub fn n_beta(&self) -> usize {
        self.geom_betas.len()
    }
}

/// 64-bit FNV-style hash of the β multi-index table for the NVRTC module
/// cache key.
#[cfg(target_os = "linux")]
fn hash_beta_table(betas: &[Vec<u8>]) -> u64 {
    let mut h = DefaultHasher::new();
    (betas.len() as u64).hash(&mut h);
    for row in betas {
        (row.len() as u64).hash(&mut h);
        for &v in row {
            v.hash(&mut h);
        }
    }
    h.finish()
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[cfg(target_os = "linux")]
struct TetMomentModuleKey {
    cc_major: i32,
    cc_minor: i32,
    kind: u8, // 0 = geom, 1 = contract
    d: u32,
    nbeta: u32,
    nalpha: u32,
    pairs: u32,
    beta_hash: u64,
    alpha_hash: u64,
    layout_tag: u8,
}

/// Closed-form factorial for the Dirichlet integral on T_ref. Up to 20! is
/// representable exactly in f64; we cap at 12 (geometric moment exponents
/// only ever reach AMAX_GEOM ≈ 6 in production specs, so n_1+n_2+n_3+3 ≤ 9).
#[inline]
fn fact_f64(n: u32) -> f64 {
    let mut acc = 1.0f64;
    for k in 2..=n {
        acc *= k as f64;
    }
    acc
}

/// Dirichlet integral over the reference 3-simplex.
///
///   ∫_{T_ref} u_1^{n1} u_2^{n2} u_3^{n3} du = n1!·n2!·n3! / (n1+n2+n3+3)!
#[inline]
fn dirichlet_ref_simplex(n1: u32, n2: u32, n3: u32) -> f64 {
    fact_f64(n1) * fact_f64(n2) * fact_f64(n3) / fact_f64(n1 + n2 + n3 + 3)
}

/// CPU reference: geometric moment G_β(T) = ∫_T (x − c0)^β dx for one
/// tetrahedron via the expansion described at the top of this section.
/// Used by the parity test and as the canonical reference if a non-Linux
/// host wants to evaluate the same quantity without a GPU.
pub fn tetrahedral_geom_moment_cpu(
    vertices: &[f64], // 4*D
    cell_center: &[f64],
    beta: &[u8],
    d: usize,
) -> f64 {
    assert_eq!(vertices.len(), 4 * d);
    assert_eq!(cell_center.len(), d);
    assert_eq!(beta.len(), d);

    // q[r] = v0[r] − c0[r], e[i][r] = v_{i+1}[r] − v0[r]
    let mut q = vec![0.0f64; d];
    let mut e = [vec![0.0f64; d], vec![0.0f64; d], vec![0.0f64; d]];
    for r in 0..d {
        q[r] = vertices[r] - cell_center[r];
        for i in 0..3 {
            e[i][r] = vertices[(i + 1) * d + r] - vertices[r];
        }
    }

    // |det B| with B = [e1 | e2 | e3] (D×3). Only the D = 3 case has a
    // unique determinant. For D > 3 we use the 3D Gram-determinant
    // sqrt(det(BᵀB)) interpretation (i.e. 6·Vol of the 3-simplex embedded
    // in R^D). For D = 2 we treat the third edge as zero-extended and
    // fall back to a planar |det| with e3 ignored — but the entry point
    // refuses D < 3 to keep the geometry well-posed.
    assert!(d >= 3, "tetrahedral path requires D ≥ 3");
    let det_b = if d == 3 {
        let m = [
            [e[0][0], e[1][0], e[2][0]],
            [e[0][1], e[1][1], e[2][1]],
            [e[0][2], e[1][2], e[2][2]],
        ];
        (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
            .abs()
    } else {
        // sqrt(det(BᵀB))
        let mut g = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for r in 0..d {
                    acc += e[i][r] * e[j][r];
                }
                g[i][j] = acc;
            }
        }
        let det_g = g[0][0] * (g[1][1] * g[2][2] - g[1][2] * g[2][1])
            - g[0][1] * (g[1][0] * g[2][2] - g[1][2] * g[2][0])
            + g[0][2] * (g[1][0] * g[2][1] - g[1][1] * g[2][0]);
        det_g.max(0.0).sqrt()
    };

    // For each axis r build the polynomial in (u_1, u_2, u_3) representing
    // (q_r + e_{1,r} u_1 + e_{2,r} u_2 + e_{3,r} u_3)^{β_r} using the
    // affine T_n recurrence T_β = T_{β-1} · (q + Σ_i e_i u_i). The
    // polynomial is stored as a dense [n_max+1; 3] tensor over u_1, u_2, u_3
    // exponents, capped at β_r along each axis.
    //
    // Per-axis polynomial size: (β_r + 1)^3 doubles. Then we multiply
    // across axes into a global polynomial in (u_1, u_2, u_3) capped at
    // (|β|, |β|, |β|) (loose but safe). Total: (|β|+1)^3 doubles, summed
    // term-by-term via the Dirichlet integral.
    let beta_total: u32 = beta.iter().map(|&v| v as u32).sum();
    let n_max = beta_total as usize;
    let stride = n_max + 1;
    let size = stride * stride * stride;
    // `poly[i + stride*(j + stride*k)] = coefficient of u_1^i u_2^j u_3^k`.
    // Start with the unit polynomial 1.
    let mut poly = vec![0.0f64; size];
    poly[0] = 1.0;

    for r in 0..d {
        let br = beta[r] as u32;
        if br == 0 {
            continue;
        }
        let qr = q[r];
        let e1 = e[0][r];
        let e2 = e[1][r];
        let e3 = e[2][r];
        for _ in 0..br {
            // poly := poly · (qr + e1·u_1 + e2·u_2 + e3·u_3). Shift-add
            // pattern; we accumulate into a fresh buffer to avoid
            // aliasing. Bounds: indices stay ≤ n_max by construction
            // because total degree at most |β|.
            let mut next = vec![0.0f64; size];
            for k in 0..stride {
                for j in 0..stride {
                    for i in 0..stride {
                        let v = poly[i + stride * (j + stride * k)];
                        if v == 0.0 {
                            continue;
                        }
                        next[i + stride * (j + stride * k)] += qr * v;
                        if i + 1 < stride {
                            next[(i + 1) + stride * (j + stride * k)] += e1 * v;
                        }
                        if j + 1 < stride {
                            next[i + stride * ((j + 1) + stride * k)] += e2 * v;
                        }
                        if k + 1 < stride {
                            next[i + stride * (j + stride * (k + 1))] += e3 * v;
                        }
                    }
                }
            }
            poly = next;
        }
    }

    // Integrate term-by-term against the Dirichlet kernel and scale by
    // |det B| (the constant Jacobian).
    let mut acc = 0.0f64;
    for k in 0..stride {
        for j in 0..stride {
            for i in 0..stride {
                let coeff = poly[i + stride * (j + stride * k)];
                if coeff == 0.0 {
                    continue;
                }
                acc += coeff * dirichlet_ref_simplex(i as u32, j as u32, k as u32);
            }
        }
    }
    acc * det_b
}

/// Generate NVRTC C++ source for the geometric-moment kernel. One thread =
/// one (tet, β-slot). The β multi-index table is baked in as a
/// `__constant__` array so the per-axis loop unrolls.
#[cfg(target_os = "linux")]
fn build_tet_geom_kernel_source(d: usize, betas: &[Vec<u8>]) -> String {
    let nbeta = betas.len();
    let beta_total_max: u32 = betas
        .iter()
        .map(|row| row.iter().map(|&v| v as u32).sum::<u32>())
        .max()
        .unwrap_or(0);
    let stride = (beta_total_max + 1) as usize;
    // We allocate the dense (stride)^3 polynomial buffer in registers /
    // local memory. Total doubles = stride^3 — keep it bounded.
    let poly_len = stride * stride * stride;

    let mut beta_decl = String::new();
    beta_decl.push_str("__constant__ unsigned char BETA_TABLE[NBETA][D] = {\n");
    for row in betas {
        beta_decl.push_str("    { ");
        for (k, v) in row.iter().enumerate() {
            if k > 0 {
                beta_decl.push_str(", ");
            }
            beta_decl.push_str(&v.to_string());
        }
        beta_decl.push_str(" },\n");
    }
    beta_decl.push_str("};\n");

    format!(
        r#"
#define D       {d}
#define NBETA   {nbeta}
#define STRIDE  {stride}
#define POLYLEN {poly_len}

{beta_decl}

// Dense polynomial in (u_1, u_2, u_3) with per-axis degree cap STRIDE-1.
// Coefficient of u_1^i u_2^j u_3^k stored at poly[i + STRIDE*(j + STRIDE*k)].

// Factorials up to 12 (covers any β-sum we ever ship).
__constant__ double FACT_LUT[13] = {{
    1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
    3628800.0, 39916800.0, 479001600.0
}};

__device__ __forceinline__ double dirichlet_3(int n1, int n2, int n3) {{
    int s = n1 + n2 + n3 + 3;
    return FACT_LUT[n1] * FACT_LUT[n2] * FACT_LUT[n3] / FACT_LUT[s];
}}

// Inputs (all device-resident):
//   vertices_flat: f64[n_tets * 4 * D] — tet vertex coordinates.
//   cell_index:    i32[n_tets]         — logical cell per tet (only used by
//                                          stage 2; the geom kernel forwards
//                                          it through unmodified output).
//   cell_centers:  f64[n_cells * D]    — per-cell expansion point c0.
// Output:
//   out: f64[NBETA * out_stride], β-major; thread (tet, β) writes
//        out[β * out_stride + tet].
extern "C" __global__ void tetrahedral_geom_moments_kernel(
    const double *vertices_flat,
    const int    *cell_index,
    const double *cell_centers,
    int           n_tets,
    long long     out_stride,
    double       *out
) {{
    const int tet  = blockIdx.x * blockDim.x + threadIdx.x;
    const int bidx = blockIdx.y * blockDim.y + threadIdx.y;
    if (tet >= n_tets || bidx >= NBETA) return;

    const int cell = cell_index[tet];
    const double *vptr  = vertices_flat + (long long)tet * 4LL * (long long)D;
    const double *c0ptr = cell_centers  + (long long)cell * (long long)D;

    // Per-axis q[r] and e[i][r] (i = 0..2 → edges v1−v0, v2−v0, v3−v0).
    double q[D];
    double e[3][D];
    #pragma unroll
    for (int r = 0; r < D; ++r) {{
        const double v0r = vptr[r];
        q[r]    = v0r - c0ptr[r];
        e[0][r] = vptr[1*D + r] - v0r;
        e[1][r] = vptr[2*D + r] - v0r;
        e[2][r] = vptr[3*D + r] - v0r;
    }}

    // |det B| (D >= 3). For D > 3 fall back to sqrt(det(BᵀB)). Compile-time
    // branched so the D = 3 fast path stays a single 3×3 cofactor expansion.
    double det_b;
#if D == 3
    det_b = fabs(
        e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
      - e[1][0] * (e[0][1] * e[2][2] - e[0][2] * e[2][1])
      + e[2][0] * (e[0][1] * e[1][2] - e[0][2] * e[1][1])
    );
#else
    double g[3][3];
    #pragma unroll
    for (int i = 0; i < 3; ++i) {{
        #pragma unroll
        for (int j = 0; j < 3; ++j) {{
            double acc = 0.0;
            #pragma unroll
            for (int r = 0; r < D; ++r) acc += e[i][r] * e[j][r];
            g[i][j] = acc;
        }}
    }}
    double det_g =
        g[0][0] * (g[1][1] * g[2][2] - g[1][2] * g[2][1])
      - g[0][1] * (g[1][0] * g[2][2] - g[1][2] * g[2][0])
      + g[0][2] * (g[1][0] * g[2][1] - g[1][1] * g[2][0]);
    det_b = sqrt(det_g > 0.0 ? det_g : 0.0);
#endif

    // Dense polynomial buffer in (u_1, u_2, u_3). Starts as the constant
    // polynomial 1, then iteratively multiplied by (q_r + Σ_i e_{{i,r}} u_i)
    // for each axis r, β_r times. We use two ping-pong buffers in local
    // memory; the compiler keeps them in registers when POLYLEN is small.
    double poly_a[POLYLEN];
    double poly_b[POLYLEN];
    #pragma unroll
    for (int t = 0; t < POLYLEN; ++t) {{ poly_a[t] = 0.0; poly_b[t] = 0.0; }}
    poly_a[0] = 1.0;
    bool a_is_src = true;

    #pragma unroll
    for (int r = 0; r < D; ++r) {{
        const unsigned int br = (unsigned int)BETA_TABLE[bidx][r];
        const double qr = q[r];
        const double e1 = e[0][r];
        const double e2 = e[1][r];
        const double e3 = e[2][r];
        for (unsigned int rep = 0; rep < br; ++rep) {{
            double *src = a_is_src ? poly_a : poly_b;
            double *dst = a_is_src ? poly_b : poly_a;
            #pragma unroll
            for (int t = 0; t < POLYLEN; ++t) dst[t] = 0.0;
            // Shift-add: dst = qr*src + e1*shift_i(src) + e2*shift_j(src) + e3*shift_k(src)
            for (int k = 0; k < STRIDE; ++k) {{
                for (int j = 0; j < STRIDE; ++j) {{
                    for (int i = 0; i < STRIDE; ++i) {{
                        const int idx = i + STRIDE * (j + STRIDE * k);
                        const double v = src[idx];
                        if (v == 0.0) continue;
                        dst[idx] += qr * v;
                        if (i + 1 < STRIDE) dst[(i+1) + STRIDE*(j + STRIDE*k)] += e1 * v;
                        if (j + 1 < STRIDE) dst[i + STRIDE*((j+1) + STRIDE*k)] += e2 * v;
                        if (k + 1 < STRIDE) dst[i + STRIDE*(j + STRIDE*(k+1))] += e3 * v;
                    }}
                }}
            }}
            a_is_src = !a_is_src;
        }}
    }}

    const double *final_poly = a_is_src ? poly_a : poly_b;
    double acc = 0.0;
    for (int k = 0; k < STRIDE; ++k) {{
        for (int j = 0; j < STRIDE; ++j) {{
            for (int i = 0; i < STRIDE; ++i) {{
                const double coeff = final_poly[i + STRIDE * (j + STRIDE * k)];
                if (coeff == 0.0) continue;
                acc = fma(coeff, dirichlet_3(i, j, k), acc);
            }}
        }}
    }}

    out[(long long)bidx * out_stride + (long long)tet] = acc * det_b;
}}
"#,
        d = d,
        nbeta = nbeta,
        stride = stride,
        poly_len = poly_len,
        beta_decl = beta_decl,
    )
}

/// Generate NVRTC C++ source for the basis-Gram contraction kernel. One
/// thread = one (cell, α-slot, pair-slot). Reads the per-tet geometric
/// moment table emitted by stage 1 plus the per-cell weight tensor and
/// accumulates the basis-pair moment for the cell.
#[cfg(target_os = "linux")]
fn build_tet_contract_kernel_source(nalpha: usize, nbeta: usize, pairs: usize) -> String {
    format!(
        r#"
#define NALPHA  {nalpha}
#define NBETA   {nbeta}
#define PAIRS   {pairs}

// Inputs:
//   geom: f64[NBETA * geom_stride] β-major; geom[β*geom_stride + tet] = G_β(tet).
//   weights: f64[n_cells * NALPHA * NBETA * PAIRS]
//       row-major (cell, α, β, pair). Caller supplies the basis-Gram tensor
//       per cell — for tensor-product cubics this is the same coefficient
//       array used by the hex kernel, expressed as a coefficient on the
//       geometric monomial basis (see the math contract block above).
//   tet_to_cell: i32[n_tets] — same array as the stage-1 cell_index input.
//   tet_offsets: i32[n_cells + 1] — CSR-style segment offsets so the kernel
//       walks all tets of cell `c` in tet_to_cell[tet_offsets[c]..tet_offsets[c+1]].
//
// Output:
//   out: f64[NALPHA * PAIRS * out_stride], α-major along outermost, then
//        pair, then cell; thread (cell, α, pair) writes
//        out[(α * PAIRS + pair) * out_stride + cell].
extern "C" __global__ void tetrahedral_contract_kernel(
    const double *geom,
    long long     geom_stride,
    const double *weights,
    const int    *tet_offsets,
    const int    *tet_index_in_segment,
    int           n_cells,
    long long     out_stride,
    double       *out
) {{
    const int cell  = blockIdx.x * blockDim.x + threadIdx.x;
    const int alpha = blockIdx.y * blockDim.y + threadIdx.y;
    const int pair  = blockIdx.z * blockDim.z + threadIdx.z;
    if (cell >= n_cells || alpha >= NALPHA || pair >= PAIRS) return;

    const int beg = tet_offsets[cell];
    const int end = tet_offsets[cell + 1];

    // Per-cell weight base: weights[cell, α, β, pair] flattened.
    const long long w_base = ((long long)cell * (long long)NALPHA + (long long)alpha)
                                * (long long)NBETA * (long long)PAIRS
                              + (long long)pair;

    double acc = 0.0;
    for (int t = beg; t < end; ++t) {{
        const int tet = tet_index_in_segment[t];
        #pragma unroll
        for (int b = 0; b < NBETA; ++b) {{
            const double g_b = geom[(long long)b * geom_stride + (long long)tet];
            const double w   = weights[w_base + (long long)b * (long long)PAIRS];
            acc = fma(w, g_b, acc);
        }}
    }}

    out[((long long)alpha * (long long)PAIRS + (long long)pair) * out_stride
        + (long long)cell] = acc;
}}
"#,
        nalpha = nalpha,
        nbeta = nbeta,
        pairs = pairs,
    )
}

/// Inputs to the tetrahedral two-stage path. The caller assembles the
/// per-cell basis-Gram weight tensor on the CPU once per fit (it depends
/// only on the cell geometry and the chosen basis, not on PIRLS state)
/// and hands it in along with the CSR-style tet→cell index.
#[derive(Clone, Debug)]
pub struct TetrahedralMomentInputs<'a> {
    pub spec: &'a TetrahedralMomentSpec,
    pub cells: &'a TetrahedralCellTable,
    /// `tet_offsets[cell + 1] − tet_offsets[cell]` = number of tets in cell.
    /// Length = n_cells + 1. Caller is responsible for emitting this in the
    /// same order as `cell_index` is partitioned (i.e. sort tets by
    /// `cell_index`, then `tet_index_in_segment[t] = original_tet_index`).
    pub tet_offsets: &'a [i32],
    pub tet_index_in_segment: &'a [i32],
    /// `weights[cell, α, β, pair]` flattened row-major. Length
    /// = n_cells · NALPHA · NBETA · PAIRS_PER_CELL.
    pub weights: &'a [f64],
}

/// Device-side dispatcher for the CM-P3 tetrahedral path. Mirrors the
/// `Ok(None)` availability pattern used by `try_device_moments` in
/// `src/gpu/cubic_cell/device.rs`: returns `Ok(None)` only when configured
/// policy permits CPU and CUDA is genuinely absent, `Ok(Some(_))` on a
/// successful launch, and `Err(_)` on probe, driver, NVRTC, or shape failure.
#[cfg(target_os = "linux")]
pub fn try_device_tetrahedral_moments(
    inputs: &TetrahedralMomentInputs<'_>,
) -> Result<Option<DeviceCubicMomentTable>, GpuError> {
    let Some(_) = gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::global_policy())? else {
        return Ok(None);
    };
    let backend = CubicMomentBackend::probe()?;
    build_tetrahedral_moments_device(backend, inputs).map(Some)
}

/// Non-Linux dispatcher. `Auto` and `Off` resolve to `Ok(None)`; `Required`
/// returns the typed required-device error produced by runtime resolution.
#[cfg(not(target_os = "linux"))]
pub fn try_device_tetrahedral_moments(
    inputs: &TetrahedralMomentInputs<'_>,
) -> Result<Option<DeviceCubicMomentTable>, GpuError> {
    let runtime = gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::global_policy())?;
    // Auto/Off on a non-Linux host reaches this validation before reporting
    // typed device absence as Ok(None).
    inputs.cells.validate()?;
    if inputs.cells.d < 3 {
        gam_gpu::gpu_bail!(
            "try_device_tetrahedral_moments: tetrahedral path requires D >= 3 (got {})",
            inputs.cells.d
        );
    }
    if runtime.is_some() {
        gam_gpu::gpu_bail!(
            "try_device_tetrahedral_moments: CUDA runtime resolved on a platform without the compiled CUDA backend"
        );
    }
    Ok(None)
}

#[cfg(target_os = "linux")]
fn build_tetrahedral_moments_device(
    backend: &CubicMomentBackend,
    inputs: &TetrahedralMomentInputs<'_>,
) -> Result<DeviceCubicMomentTable, GpuError> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    inputs.cells.validate()?;
    let spec = inputs.spec;
    let cells = inputs.cells;
    if spec.d() != cells.d {
        gam_gpu::gpu_bail!(
            "build_tetrahedral_moments_device: spec.d()={} != cells.d={}",
            spec.d(),
            cells.d
        );
    }
    if cells.d < 3 {
        gam_gpu::gpu_bail!(
            "build_tetrahedral_moments_device: tetrahedral path requires D >= 3 (got {})",
            cells.d
        );
    }
    let nalpha = spec.n_alpha();
    let nbeta = spec.n_beta();
    let pairs = spec.pairs_per_cell;
    if nalpha == 0 || nbeta == 0 || pairs == 0 || cells.n_tets == 0 || cells.n_cells == 0 {
        gam_gpu::gpu_bail!(
            "build_tetrahedral_moments_device: empty spec or cell list \
             (nalpha={nalpha}, nbeta={nbeta}, pairs={pairs}, n_tets={}, n_cells={})",
            cells.n_tets,
            cells.n_cells
        );
    }
    let want_off = cells.n_cells + 1;
    if inputs.tet_offsets.len() != want_off {
        gam_gpu::gpu_bail!(
            "build_tetrahedral_moments_device: tet_offsets len {} != n_cells+1 {}",
            inputs.tet_offsets.len(),
            want_off
        );
    }
    if inputs.tet_index_in_segment.len() != cells.n_tets {
        gam_gpu::gpu_bail!(
            "build_tetrahedral_moments_device: tet_index_in_segment len {} != n_tets {}",
            inputs.tet_index_in_segment.len(),
            cells.n_tets
        );
    }
    let want_w = cells.n_cells * nalpha * nbeta * pairs;
    if inputs.weights.len() != want_w {
        gam_gpu::gpu_bail!(
            "build_tetrahedral_moments_device: weights len {} != n_cells*nalpha*nbeta*pairs {}",
            inputs.weights.len(),
            want_w
        );
    }

    let stream = backend.inner.stream.clone();

    // ───── Stage 1: geometric moments ─────
    let geom_key = TetMomentModuleKey {
        cc_major: backend.inner.cc_major,
        cc_minor: backend.inner.cc_minor,
        kind: 0,
        d: cells.d as u32,
        nbeta: nbeta as u32,
        nalpha: 0,
        pairs: 0,
        beta_hash: hash_beta_table(&spec.geom_betas),
        alpha_hash: 0,
        layout_tag: layout_tag(spec.layout),
    };
    let d_for_geom = cells.d;
    let betas_for_geom = spec.geom_betas.clone();
    let geom_module = backend.tet_module_for(geom_key, move || {
        build_tet_geom_kernel_source(d_for_geom, &betas_for_geom)
    })?;
    let geom_func = geom_module
        .load_function("tetrahedral_geom_moments_kernel")
        .gpu_ctx("tetrahedral_moments load_function geom")?;

    let vertices_dev = stream
        .clone_htod(cells.vertices.as_slice())
        .gpu_ctx("tetrahedral_moments htod vertices")?;
    let cell_index_dev = stream
        .clone_htod(cells.cell_index.as_slice())
        .gpu_ctx("tetrahedral_moments htod cell_index")?;
    let cell_centers_dev = stream
        .clone_htod(cells.cell_centers.as_slice())
        .gpu_ctx("tetrahedral_moments htod cell_centers")?;

    let geom_stride = ((cells.n_tets + 31) / 32) * 32;
    let mut geom_dev = stream
        .alloc_zeros::<f64>(geom_stride * nbeta)
        .gpu_ctx_with(|err| {
            format!("tetrahedral_moments alloc geom (stride={geom_stride}, nbeta={nbeta}): {err}")
        })?;

    let block_x: u32 = 32;
    let block_y: u32 = 8;
    let grid_x: u32 = ((cells.n_tets as u32) + block_x - 1) / block_x;
    let grid_y: u32 = ((nbeta as u32) + block_y - 1) / block_y;
    let cfg_geom = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (block_x, block_y, 1),
        shared_mem_bytes: 0,
    };
    let n_tets_i32: i32 = i32::try_from(cells.n_tets)
        .map_err(|_| gpu_err!("tetrahedral_moments n_tets={} overflows i32", cells.n_tets))?;
    let geom_stride_i64: i64 = geom_stride as i64;

    let mut builder = stream.launch_builder(&geom_func);
    builder
        .arg(&vertices_dev)
        .arg(&cell_index_dev)
        .arg(&cell_centers_dev)
        .arg(&n_tets_i32)
        .arg(&geom_stride_i64)
        .arg(&mut geom_dev);
    // SAFETY: pointers come from cudarc-checked allocations on `stream`;
    // kernel reads inputs of declared sizes and writes within
    // geom[0 .. geom_stride * nbeta]. Launch dims are non-zero, bounded
    // by the i32 cap above.
    unsafe { builder.launch(cfg_geom) }.gpu_ctx("tetrahedral_moments geom kernel launch")?;

    // ───── Stage 2: basis-Gram contraction ─────
    let con_key = TetMomentModuleKey {
        cc_major: backend.inner.cc_major,
        cc_minor: backend.inner.cc_minor,
        kind: 1,
        d: cells.d as u32,
        nbeta: nbeta as u32,
        nalpha: nalpha as u32,
        pairs: pairs as u32,
        beta_hash: hash_beta_table(&spec.geom_betas),
        alpha_hash: hash_alpha_table(&spec.alphas),
        layout_tag: layout_tag(spec.layout),
    };
    let con_module = backend.tet_module_for(con_key, move || {
        build_tet_contract_kernel_source(nalpha, nbeta, pairs)
    })?;
    let con_func = con_module
        .load_function("tetrahedral_contract_kernel")
        .gpu_ctx("tetrahedral_moments load_function contract")?;

    let weights_dev = stream
        .clone_htod(inputs.weights)
        .gpu_ctx("tetrahedral_moments htod weights")?;
    let offsets_dev = stream
        .clone_htod(inputs.tet_offsets)
        .gpu_ctx("tetrahedral_moments htod offsets")?;
    let segidx_dev = stream
        .clone_htod(inputs.tet_index_in_segment)
        .gpu_ctx("tetrahedral_moments htod segidx")?;

    let out_stride = ((cells.n_cells + 31) / 32) * 32;
    let mut out_dev = stream
        .alloc_zeros::<f64>(out_stride * nalpha * pairs)
        .gpu_ctx_with(|err| format!(
            "tetrahedral_moments alloc out (stride={out_stride}, nalpha={nalpha}, pairs={pairs}): {err}"
        ))?;

    let block_cx: u32 = 16;
    let block_cy: u32 = 4;
    let block_cz: u32 = 4;
    let grid_cx: u32 = ((cells.n_cells as u32) + block_cx - 1) / block_cx;
    let grid_cy: u32 = ((nalpha as u32) + block_cy - 1) / block_cy;
    let grid_cz: u32 = ((pairs as u32) + block_cz - 1) / block_cz;
    let cfg_con = LaunchConfig {
        grid_dim: (grid_cx, grid_cy, grid_cz),
        block_dim: (block_cx, block_cy, block_cz),
        shared_mem_bytes: 0,
    };
    let n_cells_i32: i32 = i32::try_from(cells.n_cells).map_err(|_| {
        gpu_err!(
            "tetrahedral_moments n_cells={} overflows i32",
            cells.n_cells
        )
    })?;
    let out_stride_i64: i64 = out_stride as i64;

    let mut builder = stream.launch_builder(&con_func);
    builder
        .arg(&geom_dev)
        .arg(&geom_stride_i64)
        .arg(&weights_dev)
        .arg(&offsets_dev)
        .arg(&segidx_dev)
        .arg(&n_cells_i32)
        .arg(&out_stride_i64)
        .arg(&mut out_dev);
    // SAFETY: pointers come from cudarc-checked allocations on `stream`;
    // kernel reads inputs of declared sizes and writes within
    // out[0 .. out_stride * nalpha * pairs]. Launch dims are non-zero.
    unsafe { builder.launch(cfg_con) }.gpu_ctx("tetrahedral_moments contract kernel launch")?;
    stream
        .synchronize()
        .gpu_ctx("tetrahedral_moments synchronize")?;

    Ok(DeviceCubicMomentTable {
        n_cells: cells.n_cells,
        pair_tuple_count: pairs,
        n_alpha: nalpha,
        layout: spec.layout,
        values: out_dev,
    })
}

// ────────────────────────────────────────────────────────────────────────
// Tests (Phase 1 validation)
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod cubic_bspline_moments_tests {
    use super::*;

    fn open_uniform_knots(n_basis: usize) -> Vec<f64> {
        // Open uniform clamped knot vector for n_basis cubic B-splines on [0,1].
        let n_int = n_basis - DEGREE; // interior spans
        let mut t = Vec::with_capacity(n_basis + DEGREE + 1);
        for _ in 0..=DEGREE {
            t.push(0.0);
        }
        for i in 1..n_int {
            t.push(i as f64 / n_int as f64);
        }
        for _ in 0..=DEGREE {
            t.push(1.0);
        }
        t
    }

    fn nonuniform_knots() -> Vec<f64> {
        // 8 cubic basis functions on a deliberately non-uniform mesh in [-2, 3].
        let interior = [-1.7, -0.4, 0.1, 0.9, 1.55];
        let mut t = Vec::new();
        for _ in 0..=DEGREE {
            t.push(-2.0);
        }
        t.extend_from_slice(&interior);
        for _ in 0..=DEGREE {
            t.push(3.0);
        }
        t
    }

    /// Asserts `got` matches `expected` within `abs + rel * max(1, |expected|)`.
    /// Implemented as a macro (not a fn) so each call site inlines an `assert!`
    /// — keeps the build's "test bodies must contain assertions" scanner happy.
    macro_rules! assert_close {
        ($label:expr, $got:expr, $expected:expr, $rel:expr, $abs:expr $(,)?) => {{
            let got_v: f64 = $got;
            let expected_v: f64 = $expected;
            let rel_v: f64 = $rel;
            let abs_v: f64 = $abs;
            assert!(
                got_v.is_finite() && expected_v.is_finite(),
                "{}: non-finite (got={}, expected={})",
                $label,
                got_v,
                expected_v
            );
            let diff = (got_v - expected_v).abs();
            let bound = abs_v + rel_v * expected_v.abs().max(1.0);
            assert!(
                diff <= bound,
                "{}: |{} - {}| = {} exceeds tol abs={}, rel={} (bound {})",
                $label,
                got_v,
                expected_v,
                diff,
                abs_v,
                rel_v,
                bound
            );
        }};
    }

    /// Cox-de Boor basis on a single span must satisfy partition of unity:
    /// the four active cubics sum to 1 at every point in the span.
    #[test]
    fn cox_de_boor_partition_of_unity_uniform() {
        let t = open_uniform_knots(8);
        for k in DEGREE..(t.len() - DEGREE - 1) {
            let width = t[k + 1] - t[k];
            if !span_is_active(width) {
                continue;
            }
            let coeffs = cubic_basis_local_coeffs(&t, k);
            for step in 0..=4 {
                let u = step as f64 * width / 4.0;
                let mut sum = 0.0;
                for a in 0..ACTIVE_PER_SPAN {
                    // Horner
                    let c = &coeffs[a];
                    let mut p = c[3];
                    p = p * u + c[2];
                    p = p * u + c[1];
                    p = p * u + c[0];
                    sum += p;
                }
                assert_close!(
                    &format!("partition span={k} step={step}"),
                    sum,
                    1.0,
                    1e-13,
                    1e-13,
                );
            }
        }
    }

    /// 1D closed form vs 20-pt Gauss-Legendre on a non-uniform knot vector,
    /// for several moment exponents. Required tolerance: 1e-13 relative.
    #[test]
    fn one_d_closed_form_matches_gauss_legendre_nonuniform() {
        let t = nonuniform_knots();
        let tables = AxisCubicMomentTables::build(&t, 0, 0);
        for span in 0..tables.n_spans() {
            let width = tables.width[span];
            let left = tables.left[span];
            for pair in 0..PAIRS_PER_SPAN {
                let c = tables.prod(span, pair);
                for nu in 0..=4usize {
                    let closed = moment_1d_local(c, width, nu);
                    let gl = moment_1d_gauss_legendre(c, left, width, nu, left);
                    assert_close!(
                        &format!("span={span} pair={pair} nu={nu}"),
                        closed,
                        gl,
                        1e-13,
                        1e-14,
                    );
                }
            }
        }
    }

    /// Closed form with m ≠ L (general-purpose CPU path) must also agree
    /// with Gauss-Legendre on shifted moment expansion.
    #[test]
    fn one_d_closed_form_shifted_moments_match_gauss_legendre() {
        let t = nonuniform_knots();
        let tables = AxisCubicMomentTables::build(&t, 0, 0);
        for span in 0..tables.n_spans() {
            let width = tables.width[span];
            let left = tables.left[span];
            for pair in 0..PAIRS_PER_SPAN {
                let c = tables.prod(span, pair);
                for nu in 0..=3usize {
                    for &m in &[
                        left - 0.3,
                        left + 0.1,
                        left + 0.5 * width,
                        left + width + 0.2,
                    ] {
                        let closed = moment_1d_about(c, width, nu, m - left);
                        let gl = moment_1d_gauss_legendre(c, left, width, nu, m);
                        assert_close!(
                            &format!("span={span} pair={pair} nu={nu} m={m}"),
                            closed,
                            gl,
                            1e-12,
                            1e-13,
                        );
                    }
                }
            }
        }
    }

    /// Partition-of-unity moment test (math section 13 item 4):
    /// for α = 0 (plain integral, no monomial weight) Σ_{i,j} M_0^{ij}
    /// over the 16 ordered active pairs equals the span width (1D),
    /// because Σ_i B_i(x) = 1.
    #[test]
    fn partition_of_unity_zeroth_moment_equals_span_width() {
        let t = nonuniform_knots();
        let tables = AxisCubicMomentTables::build(&t, 0, 0);
        for span in 0..tables.n_spans() {
            let width = tables.width[span];
            let mut sum = 0.0;
            // ordered pairs: each unordered pair (a, b) with a != b is counted twice.
            for a in 0..ACTIVE_PER_SPAN {
                for b in 0..ACTIVE_PER_SPAN {
                    let m = tables.moment_local(span, active_pair_index(a, b), 0);
                    sum += m;
                }
            }
            assert_close!(&format!("partition span={span}"), sum, width, 1e-13, 1e-14);
        }
    }

    /// Tensor separability (math section 13 item 2): for any 2D cell the
    /// product moment equals the product of its 1D marginals, to ~1e-14.
    #[test]
    fn tensor_separability_2d() {
        let t = nonuniform_knots();
        let table_x = AxisCubicMomentTables::build(&t, 0, 0);
        let table_y = AxisCubicMomentTables::build(&t, 0, 0);
        let axes: Vec<&AxisCubicMomentTables> = vec![&table_x, &table_y];
        for sx in 0..table_x.n_spans() {
            for sy in 0..table_y.n_spans() {
                for pa in [0usize, 4, 9] {
                    for pb in [0usize, 3, 7] {
                        for alpha in &[[0u8, 0u8], [1, 0], [0, 1], [2, 1], [3, 3]] {
                            let m_tensor =
                                tensor_hex_moment_cpu(&axes, &[sx, sy], alpha, &[pa, pb]);
                            let m_marginal = table_x.moment_local(sx, pa, alpha[0] as usize)
                                * table_y.moment_local(sy, pb, alpha[1] as usize);
                            assert_close!(
                                &format!("tensor sx={sx} sy={sy} pa={pa} pb={pb}"),
                                m_tensor,
                                m_marginal,
                                1e-14,
                                1e-15,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Symmetry: M_α^{ij} = M_α^{ji} for plain (non-derivative) moments,
    /// because B_i B_j is commutative. Encoded by reusing the same prod_coeff.
    #[test]
    fn symmetry_pair_swap_gives_same_moment() {
        let t = nonuniform_knots();
        let tables = AxisCubicMomentTables::build(&t, 0, 0);
        for span in 0..tables.n_spans() {
            for a in 0..ACTIVE_PER_SPAN {
                for b in 0..ACTIVE_PER_SPAN {
                    for nu in 0..=3usize {
                        let m_ab = tables.moment_local(span, active_pair_index(a, b), nu);
                        let m_ba = tables.moment_local(span, active_pair_index(b, a), nu);
                        assert_eq!(
                            m_ab.to_bits(),
                            m_ba.to_bits(),
                            "span={span} pair=({a},{b}) nu={nu}: pair index must be unordered"
                        );
                    }
                }
            }
        }
    }

    /// Derivative-derivative moment ∫ B_i' B_j' dx must equal the corresponding
    /// Gauss-Legendre integral computed by convolving differentiated basis
    /// coefficient vectors on the fly. This is the "tension" penalty kernel
    /// shape (math section: derivative variants).
    #[test]
    fn derivative_moment_matches_gauss_legendre() {
        let t = nonuniform_knots();
        let tables = AxisCubicMomentTables::build(&t, 1, 1);
        for span in 0..tables.n_spans() {
            let left = tables.left[span];
            let width = tables.width[span];
            // Build the plain-basis coefficients to compare against.
            let k = tables.span_indices[span];
            let basis = cubic_basis_local_coeffs(&t, k);
            for a in 0..ACTIVE_PER_SPAN {
                for b in a..ACTIVE_PER_SPAN {
                    let pair = active_pair_index(a, b);
                    let da = differentiate_basis_coeffs(basis[a]);
                    let db = differentiate_basis_coeffs(basis[b]);
                    let prod = convolve_basis_pair(da, db);
                    for nu in 0..=2usize {
                        let closed = tables.moment_local(span, pair, nu);
                        let reference = moment_1d_gauss_legendre(prod, left, width, nu, left);
                        assert_close!(
                            &format!("d/dx span={span} pair=({a},{b}) nu={nu}"),
                            closed,
                            reference,
                            1e-13,
                            1e-14,
                        );
                    }
                }
            }
        }
    }

    /// GPU vs CPU parity for the hex tensor moment build: every
    /// (cell, alpha) entry must match the CPU reference to 1e-12 relative.
    /// Skips silently when no CUDA runtime is reachable so the test runs on
    /// macOS dev hosts as a smoke check of the host-side glue.
    #[cfg(target_os = "linux")]
    #[test]
    fn gpu_hex_tensor_moments_match_cpu_reference() {
        match gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto) {
            Ok(Some(_)) => {}
            Ok(None) => {
                eprintln!("skipping GPU parity test: no CUDA device");
                return;
            }
            Err(error) => panic!("GPU parity CUDA probe failed: {error}"),
        }
        let t = nonuniform_knots();
        let table = AxisCubicMomentTables::build(&t, 0, 0);
        let axes_cpu: Vec<&AxisCubicMomentTables> = vec![&table, &table];
        let axes_for_build: Vec<Vec<AxisCubicMomentTables>> =
            vec![vec![table.clone()], vec![table.clone()]];

        let alphas: Vec<Vec<u8>> = vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 1], vec![3, 3]];
        let deriv = vec![vec![0u8, 0u8]; alphas.len()];
        let spec = CubicMomentSpec {
            alphas: alphas.clone(),
            derivative_left: deriv.clone(),
            derivative_right: deriv.clone(),
            layout: MomentLayout::AlphaMajor,
        };

        // Build a small cell list: every (sx, sy) pair × a few pair-tuples.
        let pair_choices: [usize; 3] = [0, 4, 9];
        let mut span_per_axis: Vec<i32> = Vec::new();
        let mut pair_per_axis: Vec<i32> = Vec::new();
        let mut width_per_axis: Vec<f64> = Vec::new();
        let mut cell_meta: Vec<(usize, usize, usize, usize)> = Vec::new();
        for sx in 0..table.n_spans() {
            for sy in 0..table.n_spans() {
                for &pa in &pair_choices {
                    for &pb in &pair_choices {
                        span_per_axis.push(sx as i32);
                        span_per_axis.push(sy as i32);
                        pair_per_axis.push(pa as i32);
                        pair_per_axis.push(pb as i32);
                        width_per_axis.push(table.width[sx]);
                        width_per_axis.push(table.width[sy]);
                        cell_meta.push((sx, sy, pa, pb));
                    }
                }
            }
        }
        let n_cells = cell_meta.len();
        let cells = HexCellTable {
            span_per_axis,
            pair_per_axis,
            width_per_axis,
            n_cells,
            d: 2,
        };

        let dev = super::build_hex_tensor_moments_device(&spec, &axes_for_build, &cells)
            .expect("GPU hex-tensor moment build must succeed after CUDA admission");

        // Copy the alpha-major device buffer back to host.
        let stream = CubicMomentBackend::probe()
            .expect("backend probe ok after a successful build")
            .inner
            .stream
            .clone();
        let host_vals = stream
            .clone_dtoh(&dev.values)
            .expect("dtov of device moments");
        let out_stride = host_vals.len() / spec.n_alpha();
        assert!(
            out_stride >= n_cells,
            "out_stride={out_stride} < n_cells={n_cells}"
        );

        for (a_idx, alpha) in alphas.iter().enumerate() {
            for (cell, &(sx, sy, pa, pb)) in cell_meta.iter().enumerate() {
                let expected = tensor_hex_moment_cpu(&axes_cpu, &[sx, sy], alpha, &[pa, pb]);
                let got = host_vals[a_idx * out_stride + cell];
                assert_close!(
                    &format!("gpu cell={cell} alpha={alpha:?}"),
                    got,
                    expected,
                    1e-12,
                    1e-13,
                );
            }
        }

        // Alpha-major layout invariant: stride is the 32-aligned `n_cells` and
        // the full buffer length is `stride * n_alpha`. Catches a silent regression
        // to cell-major or to a stride that drops the warp-coalesced padding.
        assert_eq!(
            out_stride,
            ((n_cells + 31) / 32) * 32,
            "alpha-major stride must be 32-aligned n_cells"
        );
        assert_eq!(
            host_vals.len(),
            out_stride * spec.n_alpha(),
            "alpha-major total = stride * n_alpha"
        );
    }

    /// Module-cache hit on re-fit with the same spec. The NVRTC compile is
    /// the dominant per-call latency; the cache key
    /// (cc_major, cc_minor, d, amax, nalpha, alpha_hash, deriv_hash, layout_tag)
    /// must collide for two structurally-identical specs so the second build
    /// reuses the module rather than re-compiling.
    #[cfg(target_os = "linux")]
    #[test]
    fn hex_tensor_module_cache_hits_on_repeat_spec() {
        match gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto) {
            Ok(Some(_)) => {}
            Ok(None) => {
                eprintln!("skipping module-cache test: no CUDA device");
                return;
            }
            Err(error) => panic!("module-cache CUDA probe failed: {error}"),
        }
        let t = nonuniform_knots();
        let table = AxisCubicMomentTables::build(&t, 0, 0);
        let axes_for_build: Vec<Vec<AxisCubicMomentTables>> =
            vec![vec![table.clone()], vec![table.clone()]];
        let alphas: Vec<Vec<u8>> = vec![vec![0, 0], vec![1, 0], vec![2, 1]];
        let deriv = vec![vec![0u8, 0u8]; alphas.len()];
        let spec = CubicMomentSpec {
            alphas,
            derivative_left: deriv.clone(),
            derivative_right: deriv,
            layout: MomentLayout::AlphaMajor,
        };
        // One cell on (sx=0, sy=0), pair (0, 0). Avoids redoing the big sweep.
        let cells = HexCellTable {
            span_per_axis: vec![0, 0],
            pair_per_axis: vec![0, 0],
            width_per_axis: vec![table.width[0], table.width[0]],
            n_cells: 1,
            d: 2,
        };

        // First build — compiles the module.
        let first = super::build_hex_tensor_moments_device(&spec, &axes_for_build, &cells)
            .expect("GPU hex-tensor module-cache build must succeed after CUDA admission");
        let backend = CubicMomentBackend::probe().expect("backend probe");
        let cache_len_after_first = {
            let g = backend.inner.modules.lock().expect("cache lock");
            g.len()
        };
        assert!(
            cache_len_after_first >= 1,
            "module cache must hold ≥1 entry after first build"
        );

        // Second build with an identical spec — must not grow the cache. The
        // returned table is sanity-checked rather than discarded to keep the
        // banned-`let _` scanner happy.
        let second = super::build_hex_tensor_moments_device(&spec, &axes_for_build, &cells)
            .expect("second build with identical spec must succeed");
        assert_eq!(
            second.n_alpha, first.n_alpha,
            "cache hit must yield the same n_alpha as the first build"
        );
        assert_eq!(
            second.n_cells, first.n_cells,
            "cache hit must yield the same n_cells as the first build"
        );
        let cache_len_after_second = {
            let g = backend.inner.modules.lock().expect("cache lock");
            g.len()
        };
        assert_eq!(
            cache_len_after_first, cache_len_after_second,
            "identical spec must hit the cache (no new module compiled)"
        );
    }

    /// Hex tensor kernel source generator must include the requested D, AMAX,
    /// NALPHA macros, the alpha table, and the entry-point symbol. This is the
    /// host-side guard that the NVRTC template stays callable from the dispatcher
    /// even when nobody can run NVRTC (macOS CI). Compiles only on rendering.
    #[test]
    #[cfg(target_os = "linux")]
    fn hex_tensor_kernel_source_contains_required_symbols() {
        let alphas = vec![vec![0u8, 0u8], vec![1, 0], vec![0, 1], vec![2, 1]];
        let src = super::build_hex_tensor_kernel_source(2, 2, &alphas);
        assert!(
            src.contains("#define D       2"),
            "D macro missing in:\n{src}"
        );
        assert!(
            src.contains("#define AMAX    2"),
            "AMAX macro missing in:\n{src}"
        );
        assert!(
            src.contains("#define NALPHA  4"),
            "NALPHA macro missing in:\n{src}"
        );
        assert!(
            src.contains("cubic_hex_tensor_moments"),
            "kernel entry-point name missing"
        );
        assert!(
            src.contains("ALPHA_TABLE[NALPHA][D]"),
            "constant alpha table missing"
        );
        // Each alpha row should appear as a brace-list. Spot-check the (2,1)
        // entry to confirm the constant initialiser is byte-exact.
        assert!(src.contains("{ 2, 1 }"), "alpha row (2,1) missing");
    }

    /// Alpha-table hash is stable across construction order and changes
    /// whenever any byte in the table changes. Required so the NVRTC module
    /// cache key stays canonical for the same spec.
    #[test]
    #[cfg(target_os = "linux")]
    fn alpha_table_hash_is_stable_and_sensitive() {
        let a = vec![vec![0u8, 0u8], vec![1, 0], vec![0, 1]];
        let b = vec![vec![0u8, 0u8], vec![1, 0], vec![0, 1]];
        let c = vec![vec![0u8, 0u8], vec![1, 0], vec![0, 2]];
        assert_eq!(super::hash_alpha_table(&a), super::hash_alpha_table(&b));
        assert_ne!(super::hash_alpha_table(&a), super::hash_alpha_table(&c));
    }

    /// CM-P3 sanity: the CPU reference for a single unit-volume tet with
    /// vertices at the canonical basis matches the analytic Dirichlet
    /// formula for several β. Asserts the affine T_n expansion on the CPU
    /// side (and thus the formula the NVRTC kernel implements verbatim).
    #[test]
    fn tetrahedral_geom_moment_cpu_matches_dirichlet_unit_simplex() {
        // Vertices: v0 = 0, v1 = e1, v2 = e2, v3 = e3. So T = T_ref and
        // |det B| = 1; x − c0 = u (with c0 = 0). Then
        //   ∫_T u_1^{β_1} u_2^{β_2} u_3^{β_3} du
        //     = β_1! β_2! β_3! / (β_1+β_2+β_3+3)!
        let v0 = [0.0, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let v3 = [0.0, 0.0, 1.0];
        let mut verts = Vec::new();
        verts.extend_from_slice(&v0);
        verts.extend_from_slice(&v1);
        verts.extend_from_slice(&v2);
        verts.extend_from_slice(&v3);
        let c0 = [0.0f64, 0.0, 0.0];
        for beta in &[
            [0u8, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [2, 1, 0],
            [1, 1, 1],
            [2, 2, 1],
        ] {
            let got = super::tetrahedral_geom_moment_cpu(&verts, &c0, beta, 3);
            let want = super::dirichlet_ref_simplex(beta[0] as u32, beta[1] as u32, beta[2] as u32);
            assert_close!(&format!("dirichlet β={:?}", beta), got, want, 1e-14, 1e-15,);
        }
    }

    /// CM-P3: scaled / translated tet — vertices on a non-degenerate
    /// parallelepiped should yield G_β(T) = |det B| · ∫_{T_ref} (Bu + q)^β du,
    /// matched against a brute 6th-order Gauss-quadrature reference on
    /// the reference simplex (more than enough for |β| ≤ 3 polynomial
    /// integrand × the affine pull-back). The point of this test is to
    /// catch any sign / index error in the T_n expansion that the unit
    /// simplex test (B = I, q = 0) cannot see.
    #[test]
    fn tetrahedral_geom_moment_cpu_matches_quadrature_general_tet() {
        // Non-degenerate tetrahedron with shifted v0 and asymmetric edges.
        let v0 = [0.3f64, -0.2, 0.7];
        let v1 = [1.1, 0.4, 0.6];
        let v2 = [0.5, 0.9, 1.1];
        let v3 = [0.7, -0.1, 1.8];
        let mut verts = Vec::new();
        verts.extend_from_slice(&v0);
        verts.extend_from_slice(&v1);
        verts.extend_from_slice(&v2);
        verts.extend_from_slice(&v3);
        let c0 = [0.1f64, 0.05, 0.2];

        // 14-point Stroud-degree-5 rule on the 3-simplex. We use the
        // simpler approach: tensor 8-pt GL on [0,1]^3 with the standard
        // Duffy transform from the cube to the simplex,
        //   u_1 = ξ,  u_2 = (1-ξ) η,  u_3 = (1-ξ)(1-η) ζ,
        //   du_1 du_2 du_3 = (1-ξ)^2 (1-η) dξ dη dζ.
        // Exact for polynomials of total degree ≤ 15 (8-pt GL is exact
        // through degree 15 per axis), more than enough for |β| ≤ 3.
        const GL8_X01: [f64; 8] = [
            0.019_855_071_751_231_88,
            0.101_666_761_293_186_63,
            0.237_233_795_041_835_50,
            0.408_282_678_752_175_10,
            0.591_717_321_247_824_90,
            0.762_766_204_958_164_50,
            0.898_333_238_706_813_30,
            0.980_144_928_248_768_10,
        ];
        const GL8_W01: [f64; 8] = [
            0.050_614_268_145_188_18,
            0.111_190_517_226_687_24,
            0.156_853_322_938_943_55,
            0.181_341_891_689_180_92,
            0.181_341_891_689_180_92,
            0.156_853_322_938_943_55,
            0.111_190_517_226_687_24,
            0.050_614_268_145_188_18,
        ];

        for beta in &[
            [0u8, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 1, 0],
            [1, 1, 1],
            [3, 0, 0],
        ] {
            let got = super::tetrahedral_geom_moment_cpu(&verts, &c0, beta, 3);
            // Reference via Duffy + tensor 8-pt GL.
            let mut ref_acc = 0.0f64;
            for ix in 0..8 {
                for iy in 0..8 {
                    for iz in 0..8 {
                        let xi = GL8_X01[ix];
                        let et = GL8_X01[iy];
                        let ze = GL8_X01[iz];
                        let w = GL8_W01[ix] * GL8_W01[iy] * GL8_W01[iz];
                        let u1 = xi;
                        let u2 = (1.0 - xi) * et;
                        let u3 = (1.0 - xi) * (1.0 - et) * ze;
                        let jac = (1.0 - xi) * (1.0 - xi) * (1.0 - et);
                        // x(u) = v0 + u1 (v1-v0) + u2 (v2-v0) + u3 (v3-v0)
                        let mut x = [0.0f64; 3];
                        for r in 0..3 {
                            x[r] = v0[r]
                                + u1 * (v1[r] - v0[r])
                                + u2 * (v2[r] - v0[r])
                                + u3 * (v3[r] - v0[r]);
                        }
                        let mut integrand = 1.0;
                        for r in 0..3 {
                            integrand *= (x[r] - c0[r]).powi(beta[r] as i32);
                        }
                        ref_acc += w * jac * integrand;
                    }
                }
            }
            // |det B|
            let b = [
                [v1[0] - v0[0], v2[0] - v0[0], v3[0] - v0[0]],
                [v1[1] - v0[1], v2[1] - v0[1], v3[1] - v0[1]],
                [v1[2] - v0[2], v2[2] - v0[2], v3[2] - v0[2]],
            ];
            let det = (b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
                - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
                + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]))
                .abs();
            let want = ref_acc * det;
            assert_close!(&format!("tet β={:?}", beta), got, want, 1e-12, 1e-13,);
        }
    }

    /// CM-P3: TetrahedralCellTable::validate catches mis-sized vertex / cell
    /// arrays and out-of-range cell indices. The validator is the only
    /// gatekeeper between the host caller and the NVRTC kernel, so its
    /// rejection set must stay tight.
    #[test]
    fn tetrahedral_cell_table_validate_catches_misshapen_input() {
        let good = super::TetrahedralCellTable {
            vertices: vec![0.0; 4 * 3],
            cell_index: vec![0],
            cell_centers: vec![0.0, 0.0, 0.0],
            n_tets: 1,
            n_cells: 1,
            d: 3,
        };
        assert!(good.validate().is_ok(), "well-formed table validates");

        let bad_verts = super::TetrahedralCellTable {
            vertices: vec![0.0; 4 * 3 - 1],
            ..good.clone()
        };
        assert!(bad_verts.validate().is_err(), "short vertex array rejected");

        let bad_idx = super::TetrahedralCellTable {
            cell_index: vec![3],
            ..good.clone()
        };
        assert!(
            bad_idx.validate().is_err(),
            "out-of-range cell index rejected"
        );
    }

    /// CM-P3 kernel source must contain the two entry-point symbols the
    /// host dispatcher looks up by name. Catches any future rename that
    /// would surface only as a runtime "function not found" failure.
    #[test]
    #[cfg(target_os = "linux")]
    fn tetrahedral_kernel_sources_contain_required_symbols() {
        let betas = vec![vec![0u8, 0, 0], vec![1, 0, 0]];
        let geom_src = super::build_tet_geom_kernel_source(3, &betas);
        assert!(
            geom_src.contains("tetrahedral_geom_moments_kernel"),
            "geom kernel missing entry-point symbol"
        );
        assert!(
            geom_src.contains("BETA_TABLE"),
            "geom kernel missing baked-in β table"
        );
        let con_src = super::build_tet_contract_kernel_source(4, 3, 10);
        assert!(
            con_src.contains("tetrahedral_contract_kernel"),
            "contract kernel missing entry-point symbol"
        );
        assert!(
            con_src.contains("NALPHA  4") && con_src.contains("NBETA   3"),
            "contract kernel missing NALPHA / NBETA defines"
        );
    }

    /// Backend `compiled()` reflects the platform and is callable on every
    /// host (no-op probe is fine on macOS — `probe()` will return Err there).
    #[test]
    fn backend_compiled_flag_matches_platform() {
        assert_eq!(CubicMomentBackend::compiled(), cfg!(target_os = "linux"));
        if cfg!(target_os = "linux") {
            // `probe()` must not panic (reaching here proves that). Strengthen the
            // old tautological `is_ok() || is_err()` placeholder: when a CUDA
            // runtime IS present the probe must SUCCEED — a probe failure with a
            // live runtime is a real backend-init fault (device-PCG skip-pass
            // class, eee12f6b2). With no runtime, an Err is the legitimate outcome.
            match gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto) {
                Ok(Some(_)) => {
                    let probe = CubicMomentBackend::probe();
                    assert!(
                        probe.is_ok(),
                        "CubicMomentBackend::probe() must succeed when CUDA is present, got {:?}",
                        probe.err()
                    );
                }
                Ok(None) => assert!(
                    CubicMomentBackend::probe().is_err(),
                    "probe() must return Err on a Linux host with no CUDA device"
                ),
                Err(error) => panic!("CubicMomentBackend CUDA probe failed: {error}"),
            }
        } else {
            assert!(
                CubicMomentBackend::probe().is_err(),
                "non-Linux probe must return Err"
            );
        }
    }
}

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

// ────────────────────────────────────────────────────────────────────────
// Cox-de Boor: active cubic basis polynomial coefficients on one span.
// ────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────
// 1D closed-form moments
// ────────────────────────────────────────────────────────────────────────

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

    pub fn n_beta(&self) -> usize {
        self.geom_betas.len()
    }
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

        // #2422 EVERY HOST: the CPU reference this fixture grades the device
        // against must be non-degenerate. If every expected value were zero (an
        // empty span table, a collapsed pair choice), the device parity loop
        // below would compare zeros to zeros and pass while proving nothing.
        // The per-axis moments themselves are pinned to Gauss-Legendre
        // elsewhere in this module; what is checked here is that THIS fixture
        // exercises them.
        let mut nonzero_expected = 0usize;
        for alpha in alphas.iter() {
            for &(sx, sy, pa, pb) in cell_meta.iter() {
                let expected = tensor_hex_moment_cpu(&axes_cpu, &[sx, sy], alpha, &[pa, pb]);
                assert!(
                    expected.is_finite(),
                    "CPU hex-tensor reference produced a non-finite moment at \
                     alpha={alpha:?} span=({sx}, {sy}) pair=({pa}, {pb})"
                );
                if expected != 0.0 {
                    nonzero_expected += 1;
                }
            }
        }
        assert!(
            nonzero_expected * 4 >= alphas.len() * cell_meta.len(),
            "device parity fixture is near-degenerate: only {nonzero_expected} of {} expected \
             moments are nonzero, so the parity comparison would prove little",
            alphas.len() * cell_meta.len()
        );

        match gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto) {
            Ok(Some(_)) => {}
            Ok(None) => {
                // Device-free host: the admitted-only device entry must REFUSE
                // rather than fabricate a host-side answer (#1551 class). The
                // parity claim itself needs a device and gets no stand-in.
                eprintln!("no CUDA device: asserting the hex-tensor device entry declines");
                assert!(
                    super::build_hex_tensor_moments_device(&spec, &axes_for_build, &cells)
                        .is_err(),
                    "no CUDA runtime on this host, yet the hex-tensor device build returned a \
                     table — the admitted-only device path fabricated a host answer"
                );
                return;
            }
            Err(error) => panic!("GPU parity CUDA probe failed: {error}"),
        }

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
        // #2422 EVERY HOST: the cache-key claim — "two structurally-identical
        // specs must collide, a differing spec must not" — is a pure host-side
        // property of the hashed alpha/derivative tables. Only the cache
        // LOOKUP needs a device, so the discriminating half runs everywhere.
        {
            let alphas_a: Vec<Vec<u8>> = vec![vec![0, 0], vec![1, 0], vec![2, 1]];
            let alphas_b: Vec<Vec<u8>> = vec![vec![0, 0], vec![1, 0], vec![2, 1]];
            let alphas_c: Vec<Vec<u8>> = vec![vec![0, 0], vec![1, 0], vec![2, 2]];
            assert_eq!(
                hash_alpha_table(&alphas_a),
                hash_alpha_table(&alphas_b),
                "structurally identical alpha tables must hash equal, else every re-fit \
                 recompiles the module"
            );
            assert_ne!(
                hash_alpha_table(&alphas_a),
                hash_alpha_table(&alphas_c),
                "a different alpha grid must not collide, else a re-fit reuses a module \
                 specialised for the wrong grid"
            );
            let deriv_zero = vec![vec![0u8, 0u8]; 3];
            let deriv_one = vec![vec![0u8, 1u8]; 3];
            assert_eq!(
                hash_deriv_table(&deriv_zero, &deriv_zero),
                hash_deriv_table(&deriv_zero, &deriv_zero)
            );
            assert_ne!(
                hash_deriv_table(&deriv_zero, &deriv_zero),
                hash_deriv_table(&deriv_zero, &deriv_one),
                "a different derivative table must not collide with the undifferentiated one"
            );
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

        match gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto) {
            Ok(Some(_)) => {}
            Ok(None) => {
                eprintln!("no CUDA device: asserting the hex-tensor device entry declines");
                assert!(
                    super::build_hex_tensor_moments_device(&spec, &axes_for_build, &cells)
                        .is_err(),
                    "no CUDA runtime on this host, yet the hex-tensor device build returned a \
                     table — the admitted-only device path fabricated a host answer"
                );
                return;
            }
            Err(error) => panic!("module-cache CUDA probe failed: {error}"),
        }

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

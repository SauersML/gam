//! Device-resident SAE row program that emits ARROW SUFFICIENT STATISTICS
//! instead of a materialized derivative tower (#1017 / AUDIT §16 / #2304).
//!
//! # Why this module exists
//!
//! [`crate::gpu_kernels::sae_rowjet`] is device-correct but its *interface* is
//! wrong for production: it uploads a row tile, launches four kernels, and then
//! downloads the FULL row jet — `q·p + q²·p + n_beta·p + q·n_beta·p` doubles per
//! row — so that the host can immediately contract it down to blocks that are a
//! factor `p` smaller. The tower crosses PCIe only to be destroyed.
//!
//! The inner solve never needs the tower. For a reconstructed row
//! `f(ξ, β) ∈ R^p` with (metric-whitened, `√w`-scaled) residual `r ∈ R^p`, the
//! arrow system needs exactly
//!
//! ```text
//! g_ξ   = J_ξᵀ r                                  (per row, length q)
//! H_ξξ  = J_ξᵀ J_ξ + s · Σ_c r_c ∇²_ξξ f_c        (per row, q × q)
//! H_ξβ  = J_ξᵀ J_β + s · Σ_c r_c ∇²_ξβ f_c        (per row, q × n_beta)
//! g_β   = Σ_rows J_βᵀ r                           (shared, length n_beta)
//! H_ββ  = Σ_rows J_βᵀ J_β                         (shared, n_beta × n_beta)
//! ```
//!
//! with `s = 0` for the Gauss–Newton block and `s = 1` for the exact residual
//! curvature ([`ArrowCurvature`]). Reconstruction is LINEAR in `β`, so `H_ββ`
//! carries no residual-curvature term at all — `∇²_ββ f ≡ 0`. The `ξ`-blocks
//! are per-row because the arrow's `t` block is block-diagonal by row; only the
//! `β` blocks are reduced across rows.
//!
//! So this module's device boundary is: **the GPU evaluates the row program and
//! accumulates the arrow blocks in-kernel; only the reduced blocks cross PCIe.**
//! The per-row download shrinks from `p·(q + q² + q·n_beta + n_beta)` doubles to
//! `q + q² + q·n_beta` doubles (the `β` blocks are reduced to a single copy for
//! the whole tile), i.e. a factor ≈ `p` less traffic with the `q²·p` tower gone.
//!
//! # Determinism (certificate-grade)
//!
//! Bit-recurrence certificates may only be minted from a reduction whose
//! association order is a pure function of the problem shape — never of thread
//! count, block scheduling, or atomic arrival order. Therefore:
//!
//! * No `atomicAdd`. Every output element is produced by exactly ONE thread that
//!   sums its contributions in ascending index order.
//! * The cross-row `β` reduction uses the canonical tree defined here: contiguous
//!   leaves of [`ARROW_REDUCTION_LEAF_ROWS`] rows summed left-to-right, then a
//!   strict binary pairing over leaves (`out[i] = in[2i] + in[2i+1]`, with an odd
//!   final leaf CARRIED, not added to a zero pad). The leaf size is
//!   [`gam_linalg::pairwise_reduce::BASE_CHUNK`], the same base block the CPU
//!   deterministic fold uses, so the tree shape is shared by both backends.
//! * The device kernels use `__dmul_rn` / `__dadd_rn`, which forbids FMA
//!   contraction. Together with the shared tree this makes the device result
//!   bit-identical to the host mirror, not merely close.
//!
//! [`ResidentRowJetHandle::deterministic`] is therefore always true today, and
//! the accessor exists so that any future throughput-first backend (atomics,
//! split-K, TF32) must announce itself and be refused by the certificate path.
//!
//! # Third-order extension point (#2253 / Path C)
//!
//! The outer exact HVP needs a DIRECTIONAL third derivative
//! `T[v]_{ab} = Σ_c r_c · ∂³f_c/∂ξ_a ∂ξ_b ∂ξ_v · v_v` — a `q × q` matrix, NOT a
//! `q³` (or `q³·p`) tensor. Every ingredient is already resident: for softmax
//! gates the third logit derivative is the centered third moment
//! `∂³f_c/∂ℓ_a∂ℓ_b∂ℓ_d = τ⁻³ · Σ …` built from the SAME `z`, `decoded`, and the
//! centered deviations `decoded[a][c] − mean_c` that
//! [`RowChannels::second`] already forms, and the coordinate channels need only a
//! `d3` slot channel alongside `decoded_second`.
//!
//! The extension seam is exactly: add a `direction: Option<&[f64]>` (length
//! `n_rows · q`) to [`DeviceRequest`], add one kernel `sae_arrow_third_dir` with
//! the same `(row, a, b)` thread mapping and the same `__dadd_rn` accumulation as
//! `sae_arrow_htt`, and add a `t3: Vec<f64>` block (shape `[n_rows, q, q]`) to
//! [`ArrowBlocks`]. No new tensor is materialized and no new transfer shape is
//! introduced — the contraction with `v` happens inside the kernel, so the
//! download stays `O(q²)` per row. `ArrowCurvature` gains no variant: the third
//! order is a separate directional product, not a curvature mode.

use crate::gpu_kernels::sae_rowjet::{SaeRowJetPath, SaeRowJetPrimary, SaeSoftmaxRowJetInput};

/// Leaf size of the canonical cross-row reduction tree. Shared with the host
/// deterministic fold so that both backends associate additions identically.
pub const ARROW_REDUCTION_LEAF_ROWS: usize = gam_linalg::pairwise_reduce::BASE_CHUNK;

/// Which curvature the arrow blocks carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrowCurvature {
    /// `H = JᵀJ` only. The residual-curvature term is dropped (PSD by
    /// construction).
    GaussNewton,
    /// `H = JᵀJ + Σ_c r_c ∇²f_c`. The exact (possibly indefinite) block.
    ExactNewton,
}

impl ArrowCurvature {
    /// Multiplier `s` on the residual-curvature term.
    #[inline]
    pub fn residual_scale(self) -> f64 {
        match self {
            Self::GaussNewton => 0.0,
            Self::ExactNewton => 1.0,
        }
    }
}

/// Reduced arrow sufficient statistics for one row tile.
///
/// The `ξ` (latent `t`) blocks are per row — the arrow's `t` block is
/// block-diagonal by row — while the `β` blocks are summed over the tile.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrowBlocks {
    pub n_rows: usize,
    pub q: usize,
    pub n_beta: usize,
    /// `[n_rows, q]` — `g_ξ = J_ξᵀ r`.
    pub g_t: Vec<f64>,
    /// `[n_rows, q, q]` — `H_ξξ`.
    pub h_tt: Vec<f64>,
    /// `[n_rows, q, n_beta]` — `H_ξβ`.
    pub h_tb: Vec<f64>,
    /// `[n_beta]` — `g_β = Σ_rows J_βᵀ r`.
    pub g_beta: Vec<f64>,
    /// `[n_beta, n_beta]` — `H_ββ = Σ_rows J_βᵀ J_β` (no residual curvature:
    /// reconstruction is linear in `β`).
    pub h_bb: Vec<f64>,
}

impl ArrowBlocks {
    fn zeros(n_rows: usize, q: usize, n_beta: usize) -> Self {
        Self {
            n_rows,
            q,
            n_beta,
            g_t: vec![0.0; n_rows * q],
            h_tt: vec![0.0; n_rows * q * q],
            h_tb: vec![0.0; n_rows * q * n_beta],
            g_beta: vec![0.0; n_beta],
            h_bb: vec![0.0; n_beta * n_beta],
        }
    }

    #[inline]
    pub fn row_g_t(&self, row: usize) -> &[f64] {
        &self.g_t[row * self.q..(row + 1) * self.q]
    }

    #[inline]
    pub fn row_h_tt(&self, row: usize) -> &[f64] {
        let len = self.q * self.q;
        &self.h_tt[row * len..(row + 1) * len]
    }

    #[inline]
    pub fn row_h_tb(&self, row: usize) -> &[f64] {
        let len = self.q * self.n_beta;
        &self.h_tb[row * len..(row + 1) * len]
    }

    /// Exact data-part arrow Hessian-vector product `H · v`, evaluated on the
    /// REDUCED blocks — downstream never sees the derivative tower, and a Krylov
    /// loop pays zero PCIe traffic per apply (accumulate once, apply many).
    ///
    /// `direction.t` is `[n_rows, q]`, `direction.beta` is `[n_beta]`.
    pub fn apply_exact_hvp_data(
        &self,
        direction: &ArrowDirection,
    ) -> Result<ArrowDirection, String> {
        if direction.t.len() != self.n_rows * self.q || direction.beta.len() != self.n_beta {
            return Err(format!(
                "arrow HVP direction shape ({}, {}) != blocks ({}, {})",
                direction.t.len(),
                direction.beta.len(),
                self.n_rows * self.q,
                self.n_beta
            ));
        }
        let mut out_t = vec![0.0; self.n_rows * self.q];
        let mut out_beta = vec![0.0; self.n_beta];
        // Ascending, fixed-order accumulation: the same association the device
        // reduction uses, so an HVP is reproducible bit-for-bit.
        for row in 0..self.n_rows {
            let v_t = &direction.t[row * self.q..(row + 1) * self.q];
            let h_tt = self.row_h_tt(row);
            let h_tb = self.row_h_tb(row);
            for a in 0..self.q {
                let mut acc = 0.0;
                for b in 0..self.q {
                    acc += h_tt[a * self.q + b] * v_t[b];
                }
                for j in 0..self.n_beta {
                    acc += h_tb[a * self.n_beta + j] * direction.beta[j];
                }
                out_t[row * self.q + a] = acc;
            }
            for j in 0..self.n_beta {
                let mut acc = 0.0;
                for a in 0..self.q {
                    acc += h_tb[a * self.n_beta + j] * v_t[a];
                }
                out_beta[j] += acc;
            }
        }
        for i in 0..self.n_beta {
            let mut acc = 0.0;
            for j in 0..self.n_beta {
                acc += self.h_bb[i * self.n_beta + j] * direction.beta[j];
            }
            out_beta[i] += acc;
        }
        Ok(ArrowDirection {
            t: out_t,
            beta: out_beta,
        })
    }
}

/// A `(t, β)` direction / product in the arrow coordinates of one tile.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrowDirection {
    /// `[n_rows, q]`.
    pub t: Vec<f64>,
    /// `[n_beta]`.
    pub beta: Vec<f64>,
}

/// Score-only reduction: `g_ξ` per row and the tile's shared `g_β`.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrowScore {
    pub n_rows: usize,
    pub q: usize,
    pub n_beta: usize,
    pub g_t: Vec<f64>,
    pub g_beta: Vec<f64>,
}

/// Row-program channels evaluated ON DEMAND from the resident row inputs.
///
/// This is the single authority for the fused algebra: the CUDA `__device__`
/// helpers in [`RESIDENT_ARROW_KERNEL_SOURCE`] are a line-for-line mirror of
/// these four methods, and the host backend calls them directly. Nothing here
/// stores a `q²·p` tower — each channel element is an `O(1)` (logit channels
/// `O(K)` through the shared row mean) closed form.
struct RowChannels<'a> {
    input: &'a SaeSoftmaxRowJetInput,
    inv_tau: f64,
    /// `mean_c = Σ_{active a} z_a · decoded[a][c]`, formed once per row.
    mean: Vec<f64>,
}

impl<'a> RowChannels<'a> {
    fn new(input: &'a SaeSoftmaxRowJetInput, inv_tau: f64) -> Self {
        let p = input.out_dim;
        let mut mean = vec![0.0; p];
        for c in 0..p {
            let mut acc = 0.0;
            for atom in 0..input.n_atoms {
                if input.active_atoms[atom] {
                    acc += input.gate_values[atom] * input.decoded[atom * p + c];
                }
            }
            mean[c] = acc;
        }
        Self {
            input,
            inv_tau,
            mean,
        }
    }

    #[inline]
    fn kind_atom(&self, slot: usize) -> (bool, usize) {
        match self.input.primaries[slot] {
            SaeRowJetPrimary::Logit { atom } => (true, atom),
            SaeRowJetPrimary::Coordinate { atom, .. } => (false, atom),
        }
    }

    /// `√w · ∂f_c/∂ξ_slot`. Mirror of `sae_rowjet_first`.
    #[inline]
    fn first(&self, slot: usize, c: usize) -> f64 {
        let p = self.input.out_dim;
        let (is_logit, atom) = self.kind_atom(slot);
        let root = self.input.sqrt_row_weight;
        if is_logit {
            let component = if self.input.active_atoms[atom] {
                self.input.decoded[atom * p + c]
            } else {
                0.0
            };
            let centered = component - self.mean[c];
            root * (self.inv_tau * self.input.gate_values[atom]) * centered
        } else {
            if !self.input.active_atoms[atom] {
                return 0.0;
            }
            self.input.gate_values[atom] * root * self.input.decoded_first[slot * p + c]
        }
    }

    /// `√w · ∂²f_c/∂ξ_a ∂ξ_b`. Mirror of `sae_rowjet_second`.
    #[inline]
    fn second(&self, slot_a: usize, slot_b: usize, c: usize) -> f64 {
        let p = self.input.out_dim;
        let q = self.input.primaries.len();
        let (logit_a, atom_a) = self.kind_atom(slot_a);
        let (logit_b, atom_b) = self.kind_atom(slot_b);
        let root = self.input.sqrt_row_weight;
        let z = &self.input.gate_values;
        if logit_a && logit_b {
            let component_a = if self.input.active_atoms[atom_a] {
                self.input.decoded[atom_a * p + c]
            } else {
                0.0
            };
            let component_b = if self.input.active_atoms[atom_b] {
                self.input.decoded[atom_b * p + c]
            } else {
                0.0
            };
            let centered_a = component_a - self.mean[c];
            let centered_b = component_b - self.mean[c];
            let diagonal = if atom_a == atom_b { 1.0 } else { 0.0 };
            let common = self.inv_tau * self.inv_tau * z[atom_a];
            let coefficient_a = root * (common * (diagonal - z[atom_b]));
            let coefficient_b = root * (-common * z[atom_b]);
            coefficient_a * centered_a + coefficient_b * centered_b
        } else if logit_a || logit_b {
            let logit_atom = if logit_a { atom_a } else { atom_b };
            let coord_atom = if logit_a { atom_b } else { atom_a };
            let coord_slot = if logit_a { slot_b } else { slot_a };
            if !self.input.active_atoms[coord_atom] {
                return 0.0;
            }
            let diagonal = if coord_atom == logit_atom { 1.0 } else { 0.0 };
            let coefficient = z[coord_atom] * (diagonal - z[logit_atom]) * self.inv_tau * root;
            coefficient * self.input.decoded_first[coord_slot * p + c]
        } else if atom_a == atom_b {
            if !self.input.active_atoms[atom_a] {
                return 0.0;
            }
            z[atom_a] * root * self.input.decoded_second[(slot_a * q + slot_b) * p + c]
        } else {
            0.0
        }
    }

    /// `√w · ∂f_c/∂β_border`. Mirror of `sae_rowjet_beta`.
    #[inline]
    fn beta(&self, border: usize, c: usize) -> f64 {
        let p = self.input.out_dim;
        let atom = self.input.beta_atoms[border];
        if !self.input.active_atoms[atom] {
            return 0.0;
        }
        let base = self.input.gate_values[atom]
            * self.input.beta_basis_values[border]
            * self.input.sqrt_row_weight;
        base * self.input.beta_outputs[border * p + c]
    }

    /// `√w · ∂²f_c/∂ξ_slot ∂β_border`. Mirror of `sae_rowjet_beta_mixed`.
    #[inline]
    fn mixed(&self, slot: usize, border: usize, c: usize) -> f64 {
        let p = self.input.out_dim;
        let n_beta = self.input.beta_atoms.len();
        let target = self.input.beta_atoms[border];
        if !self.input.active_atoms[target] {
            return 0.0;
        }
        let (is_logit, source_atom) = self.kind_atom(slot);
        let z = &self.input.gate_values;
        let mut scalar = if is_logit {
            let diagonal = if target == source_atom { 1.0 } else { 0.0 };
            z[target]
                * (diagonal - z[source_atom])
                * self.inv_tau
                * self.input.beta_basis_values[border]
        } else if source_atom == target {
            z[target] * self.input.beta_basis_first[slot * n_beta + border]
        } else {
            0.0
        };
        scalar *= self.input.sqrt_row_weight;
        scalar * self.input.beta_outputs[border * p + c]
    }
}

/// Fused per-row arrow contribution: the reduced blocks for ONE row.
struct RowArrow {
    g_t: Vec<f64>,
    h_tt: Vec<f64>,
    h_tb: Vec<f64>,
    g_beta: Vec<f64>,
    h_bb: Vec<f64>,
}

/// Host mirror of the fused device kernels. Evaluates the row program and
/// contracts it in one pass: no `q²·p` tower is ever allocated.
fn fused_row_arrow(
    input: &SaeSoftmaxRowJetInput,
    residual: &[f64],
    inv_tau: f64,
    curvature: ArrowCurvature,
) -> RowArrow {
    let p = input.out_dim;
    let q = input.primaries.len();
    let n_beta = input.beta_atoms.len();
    let scale = curvature.residual_scale();
    let channels = RowChannels::new(input, inv_tau);

    let mut g_t = vec![0.0; q];
    let mut h_tt = vec![0.0; q * q];
    let mut h_tb = vec![0.0; q * n_beta];
    let mut g_beta = vec![0.0; n_beta];
    let mut h_bb = vec![0.0; n_beta * n_beta];

    for a in 0..q {
        let mut acc = 0.0;
        for c in 0..p {
            acc += channels.first(a, c) * residual[c];
        }
        g_t[a] = acc;
    }
    for a in 0..q {
        for b in 0..q {
            let mut acc = 0.0;
            for c in 0..p {
                let gauss_newton = channels.first(a, c) * channels.first(b, c);
                let curvature_term = scale * (residual[c] * channels.second(a, b, c));
                acc += gauss_newton + curvature_term;
            }
            h_tt[a * q + b] = acc;
        }
    }
    for a in 0..q {
        for j in 0..n_beta {
            let mut acc = 0.0;
            for c in 0..p {
                let gauss_newton = channels.first(a, c) * channels.beta(j, c);
                let curvature_term = scale * (residual[c] * channels.mixed(a, j, c));
                acc += gauss_newton + curvature_term;
            }
            h_tb[a * n_beta + j] = acc;
        }
    }
    for j in 0..n_beta {
        let mut acc = 0.0;
        for c in 0..p {
            acc += channels.beta(j, c) * residual[c];
        }
        g_beta[j] = acc;
    }
    for i in 0..n_beta {
        for j in 0..n_beta {
            let mut acc = 0.0;
            for c in 0..p {
                // Reconstruction is linear in β ⇒ ∇²_ββ f ≡ 0: Gauss-Newton and
                // exact Newton agree on this block for every curvature mode.
                acc += channels.beta(i, c) * channels.beta(j, c);
            }
            h_bb[i * n_beta + j] = acc;
        }
    }
    RowArrow {
        g_t,
        h_tt,
        h_tb,
        g_beta,
        h_bb,
    }
}

/// Canonical cross-row reduction tree (host side).
///
/// `leaf(range)` folds a contiguous run of rows left-to-right; leaves are then
/// paired strictly (`in[2i] + in[2i+1]`, odd tail CARRIED). Identical in shape
/// to the device `sae_arrow_beta_leaf` + `sae_arrow_beta_merge` pair, so the two
/// backends associate their additions the same way.
fn canonical_row_tree_reduce(
    n_rows: usize,
    width: usize,
    leaf: impl Fn(usize) -> Vec<f64>,
) -> Vec<f64> {
    if n_rows == 0 || width == 0 {
        return vec![0.0; width];
    }
    let n_leaves = n_rows.div_ceil(ARROW_REDUCTION_LEAF_ROWS);
    let mut level: Vec<Vec<f64>> = (0..n_leaves).map(&leaf).collect();
    while level.len() > 1 {
        let mut next: Vec<Vec<f64>> = Vec::with_capacity(level.len().div_ceil(2));
        let mut index = 0;
        while index + 1 < level.len() {
            let mut left = std::mem::take(&mut level[index]);
            let right = &level[index + 1];
            for (slot, value) in left.iter_mut().enumerate() {
                *value += right[slot];
            }
            next.push(left);
            index += 2;
        }
        if index < level.len() {
            // Odd tail is CARRIED, never added to a zero pad: `x + 0.0` is exact
            // for finite `x`, but carrying keeps the tree shape explicit.
            next.push(std::mem::take(&mut level[index]));
        }
        level = next;
    }
    level.pop().unwrap_or_else(|| vec![0.0; width])
}

/// Persistent, device-resident handle for one row-tile SHAPE.
///
/// Buffers are allocated once at construction (`capacity_rows`) and reused for
/// every call: there is no per-call device allocation, and — critically — no
/// per-call transfer of a derivative tower in either direction. Only the reduced
/// arrow blocks come back.
pub struct ResidentRowJetHandle {
    n_atoms: usize,
    q: usize,
    p: usize,
    n_beta: usize,
    inv_tau: f64,
    capacity_rows: usize,
    path: SaeRowJetPath,
    #[cfg(target_os = "linux")]
    device: Option<device::DeviceResidency>,
}

impl std::fmt::Debug for ResidentRowJetHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidentRowJetHandle")
            .field("n_atoms", &self.n_atoms)
            .field("q", &self.q)
            .field("p", &self.p)
            .field("n_beta", &self.n_beta)
            .field("inv_tau", &self.inv_tau)
            .field("capacity_rows", &self.capacity_rows)
            .field("path", &self.path)
            .finish()
    }
}

impl ResidentRowJetHandle {
    /// Allocate the resident buffers for a tile shape. `capacity_rows` bounds
    /// every later call; the buffers are never reallocated.
    pub fn new(
        n_atoms: usize,
        q: usize,
        p: usize,
        n_beta: usize,
        inv_tau: f64,
        capacity_rows: usize,
        path: SaeRowJetPath,
    ) -> Result<Self, String> {
        if n_atoms == 0 || p == 0 {
            return Err(format!(
                "resident arrow handle requires nonzero atoms/output dimension; got K={n_atoms}, p={p}"
            ));
        }
        if !inv_tau.is_finite() || inv_tau <= 0.0 {
            return Err(format!(
                "resident arrow handle inverse temperature must be finite and positive; got {inv_tau}"
            ));
        }
        #[cfg(target_os = "linux")]
        let device = match path {
            SaeRowJetPath::Device => Some(
                device::DeviceResidency::allocate(n_atoms, q, p, n_beta, capacity_rows)
                    .map_err(|error| error.to_string())?,
            ),
            SaeRowJetPath::Cpu => None,
        };
        #[cfg(not(target_os = "linux"))]
        if path == SaeRowJetPath::Device {
            return Err("resident arrow device handle requested on a non-Linux host".to_string());
        }
        Ok(Self {
            n_atoms,
            q,
            p,
            n_beta,
            inv_tau,
            capacity_rows,
            path,
            #[cfg(target_os = "linux")]
            device,
        })
    }

    /// True when the reduction this handle performs is the canonical
    /// fixed-order tree, i.e. when it is admissible for a bit-recurrence
    /// certificate. Both backends implemented here satisfy it; a future
    /// throughput-first (atomic / split-K) backend must return `false` and the
    /// certificate path must refuse it.
    #[inline]
    pub fn deterministic(&self) -> bool {
        true
    }

    #[inline]
    pub fn path(&self) -> SaeRowJetPath {
        self.path
    }

    fn validate_tile(
        &self,
        rows: &[SaeSoftmaxRowJetInput],
        residual: &[f64],
    ) -> Result<(), String> {
        if rows.len() > self.capacity_rows {
            return Err(format!(
                "resident arrow tile has {} rows, exceeding the resident capacity {}",
                rows.len(),
                self.capacity_rows
            ));
        }
        if residual.len() != rows.len() * self.p {
            return Err(format!(
                "resident arrow residual length {} != rows*p = {}",
                residual.len(),
                rows.len() * self.p
            ));
        }
        if let Some((index, value)) = residual
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "resident arrow residual[{index}] must be finite; got {value}"
            ));
        }
        for (index, row) in rows.iter().enumerate() {
            row.validate()?;
            let shape = (
                row.n_atoms,
                row.n_primaries(),
                row.out_dim,
                row.n_beta_borders(),
            );
            if shape != (self.n_atoms, self.q, self.p, self.n_beta) {
                return Err(format!(
                    "resident arrow row {index} shape {shape:?} != handle shape ({}, {}, {}, {})",
                    self.n_atoms, self.q, self.p, self.n_beta
                ));
            }
            if row.beta_atoms != rows[0].beta_atoms || row.beta_outputs != rows[0].beta_outputs {
                return Err(format!(
                    "resident arrow row {index} has a different decoder-border frame"
                ));
            }
        }
        Ok(())
    }

    /// Evaluate the row program and accumulate the ARROW SUFFICIENT STATISTICS.
    /// Only the reduced blocks cross the host boundary.
    ///
    /// `residual` is the metric-whitened, `√w`-scaled row residual `r`, laid out
    /// `[n_rows, p]` — the same whitening the row jets carry, so `g = Jᵀ r` and
    /// `H = JᵀJ + s Σ r ∇²f` each carry exactly one factor of `w`.
    pub fn accumulate_arrow_blocks(
        &mut self,
        rows: &[SaeSoftmaxRowJetInput],
        residual: &[f64],
        curvature: ArrowCurvature,
    ) -> Result<ArrowBlocks, String> {
        self.validate_tile(rows, residual)?;
        if rows.is_empty() {
            return Ok(ArrowBlocks::zeros(0, self.q, self.n_beta));
        }
        match self.path {
            SaeRowJetPath::Cpu => Ok(self.host_blocks(rows, residual, curvature)),
            SaeRowJetPath::Device => {
                #[cfg(target_os = "linux")]
                {
                    let residency = self
                        .device
                        .as_mut()
                        .ok_or_else(|| "resident arrow device path has no residency".to_string())?;
                    residency
                        .accumulate(rows, residual, self.inv_tau, curvature)
                        .map_err(|error| error.to_string())
                }
                #[cfg(not(target_os = "linux"))]
                {
                    Err("resident arrow device tile requested on a non-Linux host".to_string())
                }
            }
        }
    }

    /// Score-only contraction `g = Jᵀ r`. Skips every curvature block; used by
    /// the gradient-only outer passes where the Hessian is not needed.
    pub fn contract_score_rhs(
        &mut self,
        rows: &[SaeSoftmaxRowJetInput],
        residual: &[f64],
    ) -> Result<ArrowScore, String> {
        let blocks = self.accumulate_arrow_blocks(rows, residual, ArrowCurvature::GaussNewton)?;
        Ok(ArrowScore {
            n_rows: blocks.n_rows,
            q: blocks.q,
            n_beta: blocks.n_beta,
            g_t: blocks.g_t,
            g_beta: blocks.g_beta,
        })
    }

    /// Exact data-part arrow HVP through the reduced blocks. Accumulate once,
    /// apply many — a Krylov cycle pays zero device traffic per apply.
    pub fn apply_exact_hvp_data(
        &self,
        blocks: &ArrowBlocks,
        direction: &ArrowDirection,
    ) -> Result<ArrowDirection, String> {
        blocks.apply_exact_hvp_data(direction)
    }

    /// Host backend: the fused algebra, contracted with the canonical tree.
    fn host_blocks(
        &self,
        rows: &[SaeSoftmaxRowJetInput],
        residual: &[f64],
        curvature: ArrowCurvature,
    ) -> ArrowBlocks {
        let mut out = ArrowBlocks::zeros(rows.len(), self.q, self.n_beta);
        let per_row: Vec<RowArrow> = rows
            .iter()
            .enumerate()
            .map(|(row, input)| {
                fused_row_arrow(
                    input,
                    &residual[row * self.p..(row + 1) * self.p],
                    self.inv_tau,
                    curvature,
                )
            })
            .collect();
        let q = self.q;
        let n_beta = self.n_beta;
        for (row, arrow) in per_row.iter().enumerate() {
            out.g_t[row * q..(row + 1) * q].copy_from_slice(&arrow.g_t);
            out.h_tt[row * q * q..(row + 1) * q * q].copy_from_slice(&arrow.h_tt);
            out.h_tb[row * q * n_beta..(row + 1) * q * n_beta].copy_from_slice(&arrow.h_tb);
        }
        // Cross-row β reduction on the canonical tree.
        let leaf_g = |leaf: usize| -> Vec<f64> {
            let start = leaf * ARROW_REDUCTION_LEAF_ROWS;
            let end = (start + ARROW_REDUCTION_LEAF_ROWS).min(per_row.len());
            let mut acc = per_row[start].g_beta.clone();
            for arrow in &per_row[start + 1..end] {
                for (slot, value) in acc.iter_mut().enumerate() {
                    *value += arrow.g_beta[slot];
                }
            }
            acc
        };
        let leaf_h = |leaf: usize| -> Vec<f64> {
            let start = leaf * ARROW_REDUCTION_LEAF_ROWS;
            let end = (start + ARROW_REDUCTION_LEAF_ROWS).min(per_row.len());
            let mut acc = per_row[start].h_bb.clone();
            for arrow in &per_row[start + 1..end] {
                for (slot, value) in acc.iter_mut().enumerate() {
                    *value += arrow.h_bb[slot];
                }
            }
            acc
        };
        out.g_beta = canonical_row_tree_reduce(rows.len(), n_beta, leaf_g);
        out.h_bb = canonical_row_tree_reduce(rows.len(), n_beta * n_beta, leaf_h);
        out
    }
}

/// Reference boundary: the OLD interface's arrow blocks, obtained by
/// materializing the full derivative tower with the existing row-jet program and
/// contracting it on the host. This is the oracle the fused formulation must
/// reproduce; it is deliberately written against
/// [`crate::gpu_kernels::sae_rowjet::execute_softmax_row_jet_tile`] so the two
/// implementations share no code below the row-input struct.
pub fn arrow_blocks_from_materialized_tower(
    rows: &[SaeSoftmaxRowJetInput],
    residual: &[f64],
    inv_tau: f64,
    curvature: ArrowCurvature,
) -> Result<ArrowBlocks, String> {
    let channels = crate::gpu_kernels::sae_rowjet::execute_softmax_row_jet_tile(
        rows,
        inv_tau,
        SaeRowJetPath::Cpu,
    )?;
    let (n, q, p, n_beta) = (channels.n_rows, channels.q, channels.p, channels.n_beta);
    if residual.len() != n * p {
        return Err(format!(
            "materialized-tower reference residual length {} != rows*p = {}",
            residual.len(),
            n * p
        ));
    }
    let scale = curvature.residual_scale();
    let mut out = ArrowBlocks::zeros(n, q, n_beta);
    let first = |row: usize, a: usize, c: usize| channels.first[(row * q + a) * p + c];
    let second =
        |row: usize, a: usize, b: usize, c: usize| channels.second[((row * q + a) * q + b) * p + c];
    let beta = |row: usize, j: usize, c: usize| channels.beta[(row * n_beta + j) * p + c];
    let mixed = |row: usize, a: usize, j: usize, c: usize| {
        channels.beta_mixed[((row * q + a) * n_beta + j) * p + c]
    };
    for row in 0..n {
        let r = &residual[row * p..(row + 1) * p];
        for a in 0..q {
            let mut acc = 0.0;
            for c in 0..p {
                acc += first(row, a, c) * r[c];
            }
            out.g_t[row * q + a] = acc;
            for b in 0..q {
                let mut acc = 0.0;
                for c in 0..p {
                    acc +=
                        first(row, a, c) * first(row, b, c) + scale * (r[c] * second(row, a, b, c));
                }
                out.h_tt[(row * q + a) * q + b] = acc;
            }
            for j in 0..n_beta {
                let mut acc = 0.0;
                for c in 0..p {
                    acc +=
                        first(row, a, c) * beta(row, j, c) + scale * (r[c] * mixed(row, a, j, c));
                }
                out.h_tb[(row * q + a) * n_beta + j] = acc;
            }
        }
        for i in 0..n_beta {
            let mut acc = 0.0;
            for c in 0..p {
                acc += beta(row, i, c) * r[c];
            }
            out.g_beta[i] += acc;
            for j in 0..n_beta {
                let mut acc = 0.0;
                for c in 0..p {
                    acc += beta(row, i, c) * beta(row, j, c);
                }
                out.h_bb[i * n_beta + j] += acc;
            }
        }
    }
    Ok(out)
}

/// Fused arrow kernels. Every output element is written by exactly one thread;
/// there is no `atomicAdd` anywhere, and `__dmul_rn` / `__dadd_rn` forbid FMA
/// contraction so the device result is bit-identical to the host mirror.
///
/// Thread mappings:
///   * `sae_arrow_gt`   : `(row, a)`            → `g_ξ`
///   * `sae_arrow_htt`  : `(row, a, b)`         → `H_ξξ`
///   * `sae_arrow_htb`  : `(row, a, j)`         → `H_ξβ`
///   * `sae_arrow_beta_leaf`  : `(leaf, elem)`  → per-leaf `g_β` / `H_ββ` partials
///   * `sae_arrow_beta_merge` : `(node, elem)`  → one level of the strict pairing
///
/// Third-order extension point: add `sae_arrow_third_dir` with the `(row, a, b)`
/// mapping of `sae_arrow_htt`, an extra `const double* v` argument, and the
/// third centered-moment channel — it writes a `q × q` block, so no additional
/// tensor is materialized or transferred.
pub const RESIDENT_ARROW_KERNEL_SOURCE: &str = r#"
__device__ __forceinline__ double row_mean(
    const double* z, const int* active, const double* decoded,
    int row, int k, int p, int c)
{
  double mean=0.0;
  for(int a=0;a<k;++a){
    if(active[row*k+a]) mean=__dadd_rn(mean, __dmul_rn(z[row*k+a], decoded[(row*k+a)*p+c]));
  }
  return mean;
}

__device__ __forceinline__ double channel_first(
    const double* z, const int* active, const int* kind, const int* atom,
    const double* decoded, const double* d1, const double* sqrt_w,
    double inv_tau, int k, int q, int p, int row, int slot, int c, double mean)
{
  int a=atom[row*q+slot];
  double root=sqrt_w[row];
  if(kind[row*q+slot]==0){
    double component = active[row*k+a] ? decoded[(row*k+a)*p+c] : 0.0;
    double centered = component - mean;
    double coefficient = __dmul_rn(root, __dmul_rn(inv_tau, z[row*k+a]));
    return __dmul_rn(coefficient, centered);
  }
  if(!active[row*k+a]) return 0.0;
  double coefficient=__dmul_rn(z[row*k+a], root);
  return __dmul_rn(coefficient, d1[(row*q+slot)*p+c]);
}

__device__ __forceinline__ double channel_second(
    const double* z, const int* active, const int* kind, const int* atom,
    const double* decoded, const double* d1, const double* d2,
    const double* sqrt_w, double inv_tau, int k, int q, int p,
    int row, int slot_a, int slot_b, int c, double mean)
{
  int ka=kind[row*q+slot_a], kb=kind[row*q+slot_b];
  int aa=atom[row*q+slot_a], ab=atom[row*q+slot_b];
  double root=sqrt_w[row];
  if(ka==0 && kb==0){
    double component_a = active[row*k+aa] ? decoded[(row*k+aa)*p+c] : 0.0;
    double component_b = active[row*k+ab] ? decoded[(row*k+ab)*p+c] : 0.0;
    double centered_a = component_a - mean;
    double centered_b = component_b - mean;
    double za=z[row*k+aa], zb=z[row*k+ab];
    double diagonal = aa==ab ? 1.0 : 0.0;
    double common=__dmul_rn(__dmul_rn(inv_tau, inv_tau), za);
    double coefficient_a=__dmul_rn(root, __dmul_rn(common, diagonal-zb));
    double coefficient_b=__dmul_rn(root, __dmul_rn(-common, zb));
    return __dadd_rn(__dmul_rn(coefficient_a, centered_a),
                     __dmul_rn(coefficient_b, centered_b));
  }
  if(ka==0 || kb==0){
    int logit_atom = ka==0 ? aa : ab;
    int coord_atom = ka==0 ? ab : aa;
    int coord_slot = ka==0 ? slot_b : slot_a;
    if(!active[row*k+coord_atom]) return 0.0;
    double diagonal = coord_atom==logit_atom ? 1.0 : 0.0;
    double coefficient=__dmul_rn(__dmul_rn(z[row*k+coord_atom], diagonal-z[row*k+logit_atom]), inv_tau);
    coefficient=__dmul_rn(coefficient, root);
    return __dmul_rn(coefficient, d1[(row*q+coord_slot)*p+c]);
  }
  if(aa==ab){
    if(!active[row*k+aa]) return 0.0;
    double coefficient=__dmul_rn(z[row*k+aa], root);
    return __dmul_rn(coefficient, d2[((row*q+slot_a)*q+slot_b)*p+c]);
  }
  return 0.0;
}

__device__ __forceinline__ double channel_beta(
    const double* z, const int* active, const int* beta_atom,
    const double* beta_phi, const double* beta_output, const double* sqrt_w,
    int k, int p, int nb, int row, int border, int c)
{
  int a=beta_atom[border];
  if(!active[row*k+a]) return 0.0;
  double base=__dmul_rn(z[row*k+a], beta_phi[row*nb+border]);
  base=__dmul_rn(base, sqrt_w[row]);
  return __dmul_rn(base, beta_output[border*p+c]);
}

__device__ __forceinline__ double channel_mixed(
    const double* z, const int* active, const int* kind, const int* atom,
    const int* beta_atom, const double* beta_phi, const double* beta_first,
    const double* beta_output, const double* sqrt_w, double inv_tau,
    int k, int q, int p, int nb, int row, int slot, int border, int c)
{
  int target=beta_atom[border];
  if(!active[row*k+target]) return 0.0;
  int source_atom=atom[row*q+slot];
  double scalar;
  if(kind[row*q+slot]==0){
    double diagonal = target==source_atom ? 1.0 : 0.0;
    scalar=__dmul_rn(__dmul_rn(z[row*k+target], diagonal-z[row*k+source_atom]), inv_tau);
    scalar=__dmul_rn(scalar, beta_phi[row*nb+border]);
  }else if(source_atom==target){
    scalar=__dmul_rn(z[row*k+target], beta_first[(row*q+slot)*nb+border]);
  }else{
    scalar=0.0;
  }
  scalar=__dmul_rn(scalar, sqrt_w[row]);
  return __dmul_rn(scalar, beta_output[border*p+c]);
}

extern "C" __global__ void sae_arrow_gt(
    const double* z, const int* active, const int* kind, const int* atom,
    const double* decoded, const double* d1, const double* sqrt_w,
    const double* residual, double inv_tau, int k, int q, int p,
    unsigned long long total, double* g_t)
{
  unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=total) return;
  int slot=(int)(index%(unsigned long long)q);
  int row=(int)(index/(unsigned long long)q);
  double acc=0.0;
  for(int c=0;c<p;++c){
    double mean=row_mean(z,active,decoded,row,k,p,c);
    double f=channel_first(z,active,kind,atom,decoded,d1,sqrt_w,inv_tau,k,q,p,row,slot,c,mean);
    acc=__dadd_rn(acc, __dmul_rn(f, residual[row*p+c]));
  }
  g_t[index]=acc;
}

extern "C" __global__ void sae_arrow_htt(
    const double* z, const int* active, const int* kind, const int* atom,
    const double* decoded, const double* d1, const double* d2,
    const double* sqrt_w, const double* residual, double inv_tau,
    double scale, int k, int q, int p, unsigned long long total, double* h_tt)
{
  unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=total) return;
  int slot_b=(int)(index%(unsigned long long)q);
  unsigned long long rem=index/(unsigned long long)q;
  int slot_a=(int)(rem%(unsigned long long)q);
  int row=(int)(rem/(unsigned long long)q);
  double acc=0.0;
  for(int c=0;c<p;++c){
    double mean=row_mean(z,active,decoded,row,k,p,c);
    double fa=channel_first(z,active,kind,atom,decoded,d1,sqrt_w,inv_tau,k,q,p,row,slot_a,c,mean);
    double fb=channel_first(z,active,kind,atom,decoded,d1,sqrt_w,inv_tau,k,q,p,row,slot_b,c,mean);
    double s2=channel_second(z,active,kind,atom,decoded,d1,d2,sqrt_w,inv_tau,k,q,p,row,slot_a,slot_b,c,mean);
    double gauss_newton=__dmul_rn(fa, fb);
    double curvature=__dmul_rn(scale, __dmul_rn(residual[row*p+c], s2));
    acc=__dadd_rn(acc, __dadd_rn(gauss_newton, curvature));
  }
  h_tt[index]=acc;
}

extern "C" __global__ void sae_arrow_htb(
    const double* z, const int* active, const int* kind, const int* atom,
    const int* beta_atom, const double* decoded, const double* d1,
    const double* beta_phi, const double* beta_first, const double* beta_output,
    const double* sqrt_w, const double* residual, double inv_tau, double scale,
    int k, int q, int p, int nb, unsigned long long total, double* h_tb)
{
  unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=total) return;
  int border=(int)(index%(unsigned long long)nb);
  unsigned long long rem=index/(unsigned long long)nb;
  int slot=(int)(rem%(unsigned long long)q);
  int row=(int)(rem/(unsigned long long)q);
  double acc=0.0;
  for(int c=0;c<p;++c){
    double mean=row_mean(z,active,decoded,row,k,p,c);
    double f=channel_first(z,active,kind,atom,decoded,d1,sqrt_w,inv_tau,k,q,p,row,slot,c,mean);
    double b=channel_beta(z,active,beta_atom,beta_phi,beta_output,sqrt_w,k,p,nb,row,border,c);
    double m=channel_mixed(z,active,kind,atom,beta_atom,beta_phi,beta_first,beta_output,
                           sqrt_w,inv_tau,k,q,p,nb,row,slot,border,c);
    double gauss_newton=__dmul_rn(f, b);
    double curvature=__dmul_rn(scale, __dmul_rn(residual[row*p+c], m));
    acc=__dadd_rn(acc, __dadd_rn(gauss_newton, curvature));
  }
  h_tb[index]=acc;
}

// Per-leaf partials of the shared beta blocks. Element layout per leaf:
//   [0, nb)              -> g_beta
//   [nb, nb + nb*nb)     -> h_bb (row-major)
// Rows inside a leaf are folded in ASCENDING order, matching the host mirror.
extern "C" __global__ void sae_arrow_beta_leaf(
    const double* z, const int* active, const int* beta_atom,
    const double* beta_phi, const double* beta_output, const double* sqrt_w,
    const double* residual, int k, int p, int nb, int leaf_rows, int n_rows,
    unsigned long long total, double* partials)
{
  unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=total) return;
  int width=nb+nb*nb;
  int elem=(int)(index%(unsigned long long)width);
  int leaf=(int)(index/(unsigned long long)width);
  int start=leaf*leaf_rows;
  int end=start+leaf_rows; if(end>n_rows) end=n_rows;
  double acc=0.0;
  for(int row=start;row<end;++row){
    double contribution=0.0;
    if(elem<nb){
      int i=elem;
      for(int c=0;c<p;++c){
        double bi=channel_beta(z,active,beta_atom,beta_phi,beta_output,sqrt_w,k,p,nb,row,i,c);
        contribution=__dadd_rn(contribution, __dmul_rn(bi, residual[row*p+c]));
      }
    }else{
      int flat=elem-nb;
      int i=flat/nb, j=flat%nb;
      for(int c=0;c<p;++c){
        double bi=channel_beta(z,active,beta_atom,beta_phi,beta_output,sqrt_w,k,p,nb,row,i,c);
        double bj=channel_beta(z,active,beta_atom,beta_phi,beta_output,sqrt_w,k,p,nb,row,j,c);
        contribution=__dadd_rn(contribution, __dmul_rn(bi, bj));
      }
    }
    acc=__dadd_rn(acc, contribution);
  }
  partials[index]=acc;
}

// One level of the strict binary pairing: out[node] = in[2*node] + in[2*node+1],
// with an odd tail CARRIED. Fixed pairing ⇒ association order is a pure function
// of the leaf count.
extern "C" __global__ void sae_arrow_beta_merge(
    const double* in_level, int in_nodes, int width,
    unsigned long long total, double* out_level)
{
  unsigned long long index=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(index>=total) return;
  int elem=(int)(index%(unsigned long long)width);
  int node=(int)(index/(unsigned long long)width);
  int left=2*node;
  int right=left+1;
  double value=in_level[(long long)left*width+elem];
  if(right<in_nodes){
    value=__dadd_rn(value, in_level[(long long)right*width+elem]);
  }
  out_level[index]=value;
}
"#;

#[cfg(target_os = "linux")]
mod device {
    use super::{
        ARROW_REDUCTION_LEAF_ROWS, ArrowBlocks, ArrowCurvature, RESIDENT_ARROW_KERNEL_SOURCE,
    };
    use crate::gpu_kernels::sae_rowjet::{SaeRowJetPrimary, SaeSoftmaxRowJetInput};
    use cudarc::driver::{CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
    use gam_gpu::gpu_error::{GpuError, GpuResultExt};
    use std::sync::{Arc, OnceLock};

    struct Backend {
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
    }

    fn backend() -> Result<&'static Backend, GpuError> {
        static BACKEND: OnceLock<Result<Backend, GpuError>> = OnceLock::new();
        BACKEND
            .get_or_init(|| {
                let parts = gam_gpu::backend_probe::probe_cuda_backend("sae_resident_arrow")?;
                let ptx = gam_gpu::device_cache::compile_ptx_arch(RESIDENT_ARROW_KERNEL_SOURCE)
                    .gpu_ctx("resident arrow NVRTC compile")?;
                let module = parts
                    .ctx
                    .load_module(ptx)
                    .gpu_ctx("resident arrow module load")?;
                Ok(Backend {
                    stream: parts.stream,
                    module,
                })
            })
            .as_ref()
            .map_err(GpuError::clone)
    }

    fn grid(total: usize) -> Result<LaunchConfig, GpuError> {
        const THREADS: u32 = 8 * 32;
        let total_u64 = u64::try_from(total)
            .map_err(|_| gam_gpu::gpu_err!("resident arrow output length overflow"))?;
        let blocks = u32::try_from(total_u64.div_ceil(u64::from(THREADS)))
            .map_err(|_| gam_gpu::gpu_err!("resident arrow grid overflow"))?;
        Ok(LaunchConfig {
            grid_dim: (blocks.max(1), 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        })
    }

    /// Persistent device buffers. Allocated once; every later tile reuses them.
    pub(super) struct DeviceResidency {
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        k: usize,
        q: usize,
        p: usize,
        nb: usize,
        capacity_rows: usize,
        z: CudaSlice<f64>,
        active: CudaSlice<i32>,
        kind: CudaSlice<i32>,
        atom: CudaSlice<i32>,
        decoded: CudaSlice<f64>,
        d1: CudaSlice<f64>,
        d2: CudaSlice<f64>,
        sqrt_w: CudaSlice<f64>,
        residual: CudaSlice<f64>,
        beta_atom: CudaSlice<i32>,
        beta_phi: CudaSlice<f64>,
        beta_first: CudaSlice<f64>,
        beta_output: CudaSlice<f64>,
        g_t: CudaSlice<f64>,
        h_tt: CudaSlice<f64>,
        h_tb: CudaSlice<f64>,
        /// The two scratch levels of the canonical β reduction tree. They are
        /// `Option` only so the merge loop can take them out of `self` and swap
        /// owned locals — no reallocation ever happens.
        beta_ping: Option<CudaSlice<f64>>,
        beta_pong: Option<CudaSlice<f64>>,
    }

    impl DeviceResidency {
        pub(super) fn allocate(
            k: usize,
            q: usize,
            p: usize,
            nb: usize,
            capacity_rows: usize,
        ) -> Result<Self, GpuError> {
            let b = backend()?;
            let stream = b.stream.clone();
            let n = capacity_rows.max(1);
            let width = (nb + nb * nb).max(1);
            let leaves = n.div_ceil(ARROW_REDUCTION_LEAF_ROWS).max(1);
            let alloc_f64 = |len: usize, what: &'static str| -> Result<CudaSlice<f64>, GpuError> {
                stream.alloc_zeros::<f64>(len.max(1)).gpu_ctx(what)
            };
            let alloc_i32 = |len: usize, what: &'static str| -> Result<CudaSlice<i32>, GpuError> {
                stream.alloc_zeros::<i32>(len.max(1)).gpu_ctx(what)
            };
            Ok(Self {
                k,
                q,
                p,
                nb,
                capacity_rows,
                z: alloc_f64(n * k, "resident arrow alloc gates")?,
                active: alloc_i32(n * k, "resident arrow alloc active")?,
                kind: alloc_i32(n * q, "resident arrow alloc kinds")?,
                atom: alloc_i32(n * q, "resident arrow alloc atoms")?,
                decoded: alloc_f64(n * k * p, "resident arrow alloc decoded")?,
                d1: alloc_f64(n * q * p, "resident arrow alloc decoded first")?,
                d2: alloc_f64(n * q * q * p, "resident arrow alloc decoded second")?,
                sqrt_w: alloc_f64(n, "resident arrow alloc row weights")?,
                residual: alloc_f64(n * p, "resident arrow alloc residual")?,
                beta_atom: alloc_i32(nb, "resident arrow alloc beta atoms")?,
                beta_phi: alloc_f64(n * nb, "resident arrow alloc beta basis")?,
                beta_first: alloc_f64(n * q * nb, "resident arrow alloc beta basis first")?,
                beta_output: alloc_f64(nb * p, "resident arrow alloc beta outputs")?,
                g_t: alloc_f64(n * q, "resident arrow alloc g_t")?,
                h_tt: alloc_f64(n * q * q, "resident arrow alloc h_tt")?,
                h_tb: alloc_f64(n * q * nb, "resident arrow alloc h_tb")?,
                // Both tree levels are allocated at the full leaf width: the merge
                // loop SWAPS the two owned buffers, so either one can host the
                // widest (leaf) level on a later call.
                beta_ping: Some(alloc_f64(leaves * width, "resident arrow alloc beta ping")?),
                beta_pong: Some(alloc_f64(leaves * width, "resident arrow alloc beta pong")?),
                stream,
                module: b.module.clone(),
            })
        }

        /// Upload the row inputs into the resident buffers, launch the fused
        /// kernels, and download ONLY the reduced arrow blocks.
        pub(super) fn accumulate(
            &mut self,
            rows: &[SaeSoftmaxRowJetInput],
            residual: &[f64],
            inv_tau: f64,
            curvature: ArrowCurvature,
        ) -> Result<ArrowBlocks, GpuError> {
            let (k, q, p, nb) = (self.k, self.q, self.p, self.nb);
            let n = rows.len();
            if n > self.capacity_rows {
                return Err(gam_gpu::gpu_err!(
                    "resident arrow tile exceeds the resident capacity"
                ));
            }
            let stream = self.stream.clone();

            let mut z = Vec::with_capacity(n * k);
            let mut active = Vec::with_capacity(n * k);
            let mut kind = Vec::with_capacity(n * q);
            let mut atom = Vec::with_capacity(n * q);
            let mut decoded = Vec::with_capacity(n * k * p);
            let mut d1 = Vec::with_capacity(n * q * p);
            let mut d2 = Vec::with_capacity(n * q * q * p);
            let mut sqrt_w = Vec::with_capacity(n);
            let mut beta_phi = Vec::with_capacity(n * nb);
            let mut beta_first = Vec::with_capacity(n * q * nb);
            for input in rows {
                z.extend_from_slice(&input.gate_values);
                active.extend(input.active_atoms.iter().map(|&value| i32::from(value)));
                for primary in &input.primaries {
                    let (tag, source) = match *primary {
                        SaeRowJetPrimary::Logit { atom: source } => (0_i32, source),
                        SaeRowJetPrimary::Coordinate { atom: source, .. } => (1_i32, source),
                    };
                    kind.push(tag);
                    atom.push(i32::try_from(source).map_err(|_| {
                        gam_gpu::gpu_err!("resident arrow atom index overflows i32")
                    })?);
                }
                decoded.extend_from_slice(&input.decoded);
                d1.extend_from_slice(&input.decoded_first);
                d2.extend_from_slice(&input.decoded_second);
                sqrt_w.push(input.sqrt_row_weight);
                beta_phi.extend_from_slice(&input.beta_basis_values);
                beta_first.extend_from_slice(&input.beta_basis_first);
            }
            let beta_atom: Vec<i32> = rows[0]
                .beta_atoms
                .iter()
                .map(|&source| {
                    i32::try_from(source).map_err(|_| {
                        gam_gpu::gpu_err!("resident arrow beta atom index overflows i32")
                    })
                })
                .collect::<Result<_, _>>()?;

            // Uploads write into the PERSISTENT buffers: no per-call allocation.
            upload_f64(&stream, &z, &mut self.z, "resident arrow htod gates")?;
            upload_i32(
                &stream,
                &active,
                &mut self.active,
                "resident arrow htod active",
            )?;
            upload_i32(&stream, &kind, &mut self.kind, "resident arrow htod kinds")?;
            upload_i32(&stream, &atom, &mut self.atom, "resident arrow htod atoms")?;
            upload_f64(
                &stream,
                &decoded,
                &mut self.decoded,
                "resident arrow htod decoded",
            )?;
            upload_f64(
                &stream,
                &d1,
                &mut self.d1,
                "resident arrow htod decoded first",
            )?;
            upload_f64(
                &stream,
                &d2,
                &mut self.d2,
                "resident arrow htod decoded second",
            )?;
            upload_f64(
                &stream,
                &sqrt_w,
                &mut self.sqrt_w,
                "resident arrow htod weights",
            )?;
            upload_f64(
                &stream,
                residual,
                &mut self.residual,
                "resident arrow htod residual",
            )?;
            upload_i32(
                &stream,
                &beta_atom,
                &mut self.beta_atom,
                "resident arrow htod beta atoms",
            )?;
            upload_f64(
                &stream,
                &beta_phi,
                &mut self.beta_phi,
                "resident arrow htod beta basis",
            )?;
            upload_f64(
                &stream,
                &beta_first,
                &mut self.beta_first,
                "resident arrow htod beta basis first",
            )?;
            upload_f64(
                &stream,
                rows[0].beta_outputs.as_ref(),
                &mut self.beta_output,
                "resident arrow htod beta outputs",
            )?;

            let scale = curvature.residual_scale();
            let k_i32 = i32::try_from(k)
                .map_err(|_| gam_gpu::gpu_err!("resident arrow K overflows i32"))?;
            let q_i32 = i32::try_from(q)
                .map_err(|_| gam_gpu::gpu_err!("resident arrow q overflows i32"))?;
            let p_i32 = i32::try_from(p)
                .map_err(|_| gam_gpu::gpu_err!("resident arrow p overflows i32"))?;
            let nb_i32 = i32::try_from(nb)
                .map_err(|_| gam_gpu::gpu_err!("resident arrow beta count overflows i32"))?;

            let gt_len = n * q;
            if gt_len != 0 {
                let function = self
                    .module
                    .load_function("sae_arrow_gt")
                    .gpu_ctx("resident arrow g_t load")?;
                let total = gt_len as u64;
                let mut launch = stream.launch_builder(&function);
                launch
                    .arg(&self.z)
                    .arg(&self.active)
                    .arg(&self.kind)
                    .arg(&self.atom)
                    .arg(&self.decoded)
                    .arg(&self.d1)
                    .arg(&self.sqrt_w)
                    .arg(&self.residual)
                    .arg(&inv_tau)
                    .arg(&k_i32)
                    .arg(&q_i32)
                    .arg(&p_i32)
                    .arg(&total)
                    .arg(&mut self.g_t);
                // SAFETY: the kernel's argument ABI matches this builder and the
                // grid covers exactly the `gt_len` allocated outputs.
                unsafe { launch.launch(grid(gt_len)?) }.gpu_ctx("resident arrow g_t launch")?;
            }
            let htt_len = n * q * q;
            if htt_len != 0 {
                let function = self
                    .module
                    .load_function("sae_arrow_htt")
                    .gpu_ctx("resident arrow h_tt load")?;
                let total = htt_len as u64;
                let mut launch = stream.launch_builder(&function);
                launch
                    .arg(&self.z)
                    .arg(&self.active)
                    .arg(&self.kind)
                    .arg(&self.atom)
                    .arg(&self.decoded)
                    .arg(&self.d1)
                    .arg(&self.d2)
                    .arg(&self.sqrt_w)
                    .arg(&self.residual)
                    .arg(&inv_tau)
                    .arg(&scale)
                    .arg(&k_i32)
                    .arg(&q_i32)
                    .arg(&p_i32)
                    .arg(&total)
                    .arg(&mut self.h_tt);
                // SAFETY: as above; one thread per `h_tt` element.
                unsafe { launch.launch(grid(htt_len)?) }.gpu_ctx("resident arrow h_tt launch")?;
            }
            let htb_len = n * q * nb;
            if htb_len != 0 {
                let function = self
                    .module
                    .load_function("sae_arrow_htb")
                    .gpu_ctx("resident arrow h_tb load")?;
                let total = htb_len as u64;
                let mut launch = stream.launch_builder(&function);
                launch
                    .arg(&self.z)
                    .arg(&self.active)
                    .arg(&self.kind)
                    .arg(&self.atom)
                    .arg(&self.beta_atom)
                    .arg(&self.decoded)
                    .arg(&self.d1)
                    .arg(&self.beta_phi)
                    .arg(&self.beta_first)
                    .arg(&self.beta_output)
                    .arg(&self.sqrt_w)
                    .arg(&self.residual)
                    .arg(&inv_tau)
                    .arg(&scale)
                    .arg(&k_i32)
                    .arg(&q_i32)
                    .arg(&p_i32)
                    .arg(&nb_i32)
                    .arg(&total)
                    .arg(&mut self.h_tb);
                // SAFETY: as above; one thread per `h_tb` element.
                unsafe { launch.launch(grid(htb_len)?) }.gpu_ctx("resident arrow h_tb launch")?;
            }

            let mut blocks = ArrowBlocks::zeros(n, q, nb);
            if gt_len != 0 {
                stream
                    .memcpy_dtoh(&self.g_t.slice(0..gt_len), &mut blocks.g_t)
                    .gpu_ctx("resident arrow dtoh g_t")?;
            }
            if htt_len != 0 {
                stream
                    .memcpy_dtoh(&self.h_tt.slice(0..htt_len), &mut blocks.h_tt)
                    .gpu_ctx("resident arrow dtoh h_tt")?;
            }
            if htb_len != 0 {
                stream
                    .memcpy_dtoh(&self.h_tb.slice(0..htb_len), &mut blocks.h_tb)
                    .gpu_ctx("resident arrow dtoh h_tb")?;
            }

            if nb != 0 {
                let width = nb + nb * nb;
                let leaves = n.div_ceil(ARROW_REDUCTION_LEAF_ROWS);
                let leaf_total = leaves * width;
                // Own the two tree levels for the whole reduction: swapping owned
                // locals keeps the buffers resident without aliasing `self`.
                let mut ping = self.beta_ping.take().ok_or_else(|| {
                    gam_gpu::gpu_err!("resident arrow beta scratch was not restored")
                })?;
                let mut pong = self.beta_pong.take().ok_or_else(|| {
                    gam_gpu::gpu_err!("resident arrow beta scratch was not restored")
                })?;
                let leaf_rows_i32 = i32::try_from(ARROW_REDUCTION_LEAF_ROWS)
                    .map_err(|_| gam_gpu::gpu_err!("resident arrow leaf size overflows i32"))?;
                let n_rows_i32 = i32::try_from(n)
                    .map_err(|_| gam_gpu::gpu_err!("resident arrow row count overflows i32"))?;
                let width_i32 = i32::try_from(width)
                    .map_err(|_| gam_gpu::gpu_err!("resident arrow beta width overflows i32"))?;
                let leaf_result = (|| -> Result<Vec<f64>, GpuError> {
                    let function = self
                        .module
                        .load_function("sae_arrow_beta_leaf")
                        .gpu_ctx("resident arrow beta leaf load")?;
                    let total = leaf_total as u64;
                    let mut launch = stream.launch_builder(&function);
                    launch
                        .arg(&self.z)
                        .arg(&self.active)
                        .arg(&self.beta_atom)
                        .arg(&self.beta_phi)
                        .arg(&self.beta_output)
                        .arg(&self.sqrt_w)
                        .arg(&self.residual)
                        .arg(&k_i32)
                        .arg(&p_i32)
                        .arg(&nb_i32)
                        .arg(&leaf_rows_i32)
                        .arg(&n_rows_i32)
                        .arg(&total)
                        .arg(&mut ping);
                    // SAFETY: the kernel ABI matches this builder; one thread per
                    // (leaf, beta-block element) covers exactly `leaf_total`.
                    unsafe { launch.launch(grid(leaf_total)?) }
                        .gpu_ctx("resident arrow beta leaf launch")?;

                    // Strict binary pairing, one kernel per level. The pairing is a
                    // pure function of the leaf count, so the association order —
                    // hence the exact f64 result — is scheduling-independent.
                    let mut nodes = leaves;
                    while nodes > 1 {
                        let next_nodes = nodes.div_ceil(2);
                        let level_total = next_nodes * width;
                        let function = self
                            .module
                            .load_function("sae_arrow_beta_merge")
                            .gpu_ctx("resident arrow beta merge load")?;
                        let total = level_total as u64;
                        let in_nodes = i32::try_from(nodes).map_err(|_| {
                            gam_gpu::gpu_err!("resident arrow beta node count overflows i32")
                        })?;
                        let mut launch = stream.launch_builder(&function);
                        launch
                            .arg(&ping)
                            .arg(&in_nodes)
                            .arg(&width_i32)
                            .arg(&total)
                            .arg(&mut pong);
                        // SAFETY: the kernel ABI matches; the grid covers exactly
                        // the `level_total` outputs of this level, and `ping` /
                        // `pong` are distinct resident buffers.
                        unsafe { launch.launch(grid(level_total)?) }
                            .gpu_ctx("resident arrow beta merge launch")?;
                        std::mem::swap(&mut ping, &mut pong);
                        nodes = next_nodes;
                    }

                    let mut root = vec![0.0_f64; width];
                    stream
                        .memcpy_dtoh(&ping.slice(0..width), &mut root)
                        .gpu_ctx("resident arrow dtoh beta blocks")?;
                    Ok(root)
                })();
                self.beta_ping = Some(ping);
                self.beta_pong = Some(pong);
                let root = leaf_result?;
                blocks.g_beta.copy_from_slice(&root[..nb]);
                blocks.h_bb.copy_from_slice(&root[nb..]);
            }

            stream.synchronize().gpu_ctx("resident arrow synchronize")?;
            Ok(blocks)
        }
    }

    fn upload_f64(
        stream: &Arc<CudaStream>,
        host: &[f64],
        device: &mut CudaSlice<f64>,
        what: &'static str,
    ) -> Result<(), GpuError> {
        if host.is_empty() {
            return Ok(());
        }
        stream
            .memcpy_htod(host, &mut device.slice_mut(0..host.len()))
            .gpu_ctx(what)
    }

    fn upload_i32(
        stream: &Arc<CudaStream>,
        host: &[i32],
        device: &mut CudaSlice<i32>,
        what: &'static str,
    ) -> Result<(), GpuError> {
        if host.is_empty() {
            return Ok(());
        }
        stream
            .memcpy_htod(host, &mut device.slice_mut(0..host.len()))
            .gpu_ctx(what)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(n: usize) -> (Vec<SaeSoftmaxRowJetInput>, Vec<f64>) {
        let k = 3;
        let p = 4;
        let primaries = vec![
            SaeRowJetPrimary::Logit { atom: 0 },
            SaeRowJetPrimary::Logit { atom: 1 },
            SaeRowJetPrimary::Coordinate { atom: 0, axis: 0 },
            SaeRowJetPrimary::Coordinate { atom: 1, axis: 0 },
            SaeRowJetPrimary::Coordinate { atom: 1, axis: 1 },
            SaeRowJetPrimary::Coordinate { atom: 2, axis: 0 },
        ];
        let q = primaries.len();
        let rows: Vec<SaeSoftmaxRowJetInput> = (0..n)
            .map(|row| {
                let logits: Vec<f64> = (0..k)
                    .map(|atom| 0.4 * ((row * 17 + atom * 11 + 1) as f64 * 0.07).sin())
                    .collect();
                let shift = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = logits.iter().map(|value| (value - shift).exp()).collect();
                let sum: f64 = exps.iter().sum();
                let gate_values: Vec<f64> = exps.iter().map(|value| value / sum).collect();
                let decoded: Vec<f64> = (0..k * p)
                    .map(|index| ((row * 13 + index * 7 + 3) as f64 * 0.09).cos())
                    .collect();
                let mut decoded_first = vec![0.0; q * p];
                for slot in 2..q {
                    for c in 0..p {
                        decoded_first[slot * p + c] =
                            ((row * 19 + slot * 5 + c + 2) as f64 * 0.04).sin();
                    }
                }
                let mut decoded_second = vec![0.0; q * q * p];
                for a in 2..q {
                    for b in 2..q {
                        let same_atom = matches!(
                            (primaries[a], primaries[b]),
                            (
                                SaeRowJetPrimary::Coordinate { atom: left, .. },
                                SaeRowJetPrimary::Coordinate { atom: right, .. },
                            ) if left == right
                        );
                        if same_atom {
                            for c in 0..p {
                                decoded_second[(a * q + b) * p + c] =
                                    ((row * 23 + a * 7 + b * 3 + c + 1) as f64 * 0.03).cos();
                            }
                        }
                    }
                }
                let beta_atoms = vec![0_usize, 1, 2];
                let n_beta = beta_atoms.len();
                let beta_basis_values = vec![0.8, -0.3, 0.5];
                let mut beta_basis_first = vec![0.0; q * n_beta];
                beta_basis_first[2 * n_beta] = 0.2;
                beta_basis_first[3 * n_beta + 1] = -0.4;
                beta_basis_first[4 * n_beta + 1] = 0.7;
                beta_basis_first[5 * n_beta + 2] = -0.1;
                let beta_outputs: Vec<f64> = (0..n_beta * p)
                    .map(|index| ((index * 5 + 1) as f64 * 0.11).sin())
                    .collect();
                SaeSoftmaxRowJetInput {
                    n_atoms: k,
                    out_dim: p,
                    coordinate_slots: SaeSoftmaxRowJetInput::coordinate_slots_for(&primaries),
                    primaries: primaries.clone(),
                    gate_values,
                    active_atoms: vec![true, true, row % 3 != 0],
                    sqrt_row_weight: (1.0 + row as f64 * 0.1).sqrt(),
                    decoded,
                    decoded_first,
                    decoded_second,
                    beta_atoms: beta_atoms.into(),
                    beta_basis_values,
                    beta_basis_first,
                    beta_outputs: beta_outputs.into(),
                }
            })
            .collect();
        let residual: Vec<f64> = (0..n * p)
            .map(|index| 0.3 * ((index * 3 + 1) as f64 * 0.17).sin())
            .collect();
        (rows, residual)
    }

    fn assert_blocks_close(fused: &ArrowBlocks, reference: &ArrowBlocks, tol: f64, label: &str) {
        assert_eq!(fused.n_rows, reference.n_rows, "{label}: row count");
        let pairs: [(&str, &Vec<f64>, &Vec<f64>); 5] = [
            ("g_t", &fused.g_t, &reference.g_t),
            ("h_tt", &fused.h_tt, &reference.h_tt),
            ("h_tb", &fused.h_tb, &reference.h_tb),
            ("g_beta", &fused.g_beta, &reference.g_beta),
            ("h_bb", &fused.h_bb, &reference.h_bb),
        ];
        let mut nonzero = false;
        for (name, left, right) in pairs {
            assert_eq!(left.len(), right.len(), "{label}: {name} length");
            for (index, (&a, &b)) in left.iter().zip(right.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= tol * (1.0 + b.abs()),
                    "{label}: {name}[{index}] fused {a:e} != reference {b:e}"
                );
                if b != 0.0 {
                    nonzero = true;
                }
            }
        }
        assert!(nonzero, "{label}: reference blocks are entirely zero");
    }

    /// CPU-parity: the fused resident formulation reproduces the arrow blocks of
    /// the OLD boundary (materialize the full tower, then contract on the host).
    /// Runs with no GPU.
    #[test]
    fn resident_arrow_blocks_match_materialized_tower_contraction_1017() {
        // More rows than one reduction leaf, so the cross-row β tree has an
        // interior node AND an odd tail.
        let n = ARROW_REDUCTION_LEAF_ROWS * 2 + 7;
        let (rows, residual) = fixture(n);
        for curvature in [ArrowCurvature::GaussNewton, ArrowCurvature::ExactNewton] {
            let mut handle =
                ResidentRowJetHandle::new(3, 6, 4, 3, 1.3, n, SaeRowJetPath::Cpu).expect("handle");
            assert!(handle.deterministic());
            let fused = handle
                .accumulate_arrow_blocks(&rows, &residual, curvature)
                .expect("fused arrow blocks");
            let reference = arrow_blocks_from_materialized_tower(&rows, &residual, 1.3, curvature)
                .expect("materialized-tower reference");
            assert_blocks_close(&fused, &reference, 1.0e-12, &format!("{curvature:?}"));
        }
    }

    /// The Gauss-Newton and exact-Newton blocks must differ (otherwise the
    /// residual-curvature channel is silently dead and the parity test above is
    /// vacuous), while `H_ββ` must agree exactly: reconstruction is linear in β.
    #[test]
    fn resident_arrow_curvature_channel_is_live_and_beta_block_is_linear_1017() {
        let n = 32;
        let (rows, residual) = fixture(n);
        let mut handle =
            ResidentRowJetHandle::new(3, 6, 4, 3, 1.3, n, SaeRowJetPath::Cpu).expect("handle");
        let gn = handle
            .accumulate_arrow_blocks(&rows, &residual, ArrowCurvature::GaussNewton)
            .expect("gauss-newton blocks");
        let exact = handle
            .accumulate_arrow_blocks(&rows, &residual, ArrowCurvature::ExactNewton)
            .expect("exact-newton blocks");
        assert_eq!(gn.g_t, exact.g_t, "the score must not depend on curvature");
        assert_eq!(
            gn.h_bb, exact.h_bb,
            "β is linear: H_ββ carries no curvature"
        );
        let htt_gap = gn
            .h_tt
            .iter()
            .zip(exact.h_tt.iter())
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        let htb_gap = gn
            .h_tb
            .iter()
            .zip(exact.h_tb.iter())
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(htt_gap > 1.0e-8, "residual curvature is dead in H_ξξ");
        assert!(htb_gap > 1.0e-8, "residual curvature is dead in H_ξβ");
    }

    /// The reduced-block HVP equals the dense product of the same blocks — the
    /// downstream Krylov apply never needs the tower.
    #[test]
    fn resident_arrow_hvp_matches_dense_block_product_1017() {
        let n = 5;
        let (rows, residual) = fixture(n);
        let mut handle =
            ResidentRowJetHandle::new(3, 6, 4, 3, 1.3, n, SaeRowJetPath::Cpu).expect("handle");
        let blocks = handle
            .accumulate_arrow_blocks(&rows, &residual, ArrowCurvature::ExactNewton)
            .expect("blocks");
        let (q, nb) = (blocks.q, blocks.n_beta);
        let direction = ArrowDirection {
            t: (0..n * q)
                .map(|index| ((index * 7 + 1) as f64 * 0.21).cos())
                .collect(),
            beta: (0..nb)
                .map(|index| ((index * 11 + 2) as f64 * 0.13).sin())
                .collect(),
        };
        let product = handle
            .apply_exact_hvp_data(&blocks, &direction)
            .expect("hvp");

        // Independent dense assembly of the arrow operator's action.
        let mut expect_t = vec![0.0; n * q];
        let mut expect_beta = vec![0.0; nb];
        for row in 0..n {
            for a in 0..q {
                let mut acc = 0.0;
                for b in 0..q {
                    acc += blocks.h_tt[(row * q + a) * q + b] * direction.t[row * q + b];
                }
                for j in 0..nb {
                    acc += blocks.h_tb[(row * q + a) * nb + j] * direction.beta[j];
                }
                expect_t[row * q + a] = acc;
            }
            for j in 0..nb {
                for a in 0..q {
                    expect_beta[j] +=
                        blocks.h_tb[(row * q + a) * nb + j] * direction.t[row * q + a];
                }
            }
        }
        for i in 0..nb {
            for j in 0..nb {
                expect_beta[i] += blocks.h_bb[i * nb + j] * direction.beta[j];
            }
        }
        for (index, (&got, &want)) in product.t.iter().zip(expect_t.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1.0e-12 * (1.0 + want.abs()),
                "hvp t[{index}] {got:e} != {want:e}"
            );
        }
        for (index, (&got, &want)) in product.beta.iter().zip(expect_beta.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1.0e-12 * (1.0 + want.abs()),
                "hvp beta[{index}] {got:e} != {want:e}"
            );
        }
    }

    /// The device path must reproduce the host mirror on every reduced block.
    /// Skips ONLY when CUDA is genuinely absent; a real driver error fails loud.
    #[cfg(target_os = "linux")]
    #[test]
    fn resident_arrow_device_matches_host_reduced_blocks_1017() {
        match gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto) {
            Ok(Some(_)) => {}
            Ok(None) => return,
            Err(error) => panic!("resident arrow CUDA admission failed: {error}"),
        }
        let n = ARROW_REDUCTION_LEAF_ROWS * 5 + 3;
        let (rows, residual) = fixture(n);
        for curvature in [ArrowCurvature::GaussNewton, ArrowCurvature::ExactNewton] {
            let mut host =
                ResidentRowJetHandle::new(3, 6, 4, 3, 1.3, n, SaeRowJetPath::Cpu).expect("host");
            let expected = host
                .accumulate_arrow_blocks(&rows, &residual, curvature)
                .expect("host blocks");
            let mut resident = ResidentRowJetHandle::new(3, 6, 4, 3, 1.3, n, SaeRowJetPath::Device)
                .expect("device residency");
            // Two passes through the SAME resident buffers: no reallocation, and
            // the second pass must be bit-identical to the first.
            let first = resident
                .accumulate_arrow_blocks(&rows, &residual, curvature)
                .expect("device blocks");
            let second = resident
                .accumulate_arrow_blocks(&rows, &residual, curvature)
                .expect("device blocks (repeat)");
            assert_eq!(
                first, second,
                "resident device reduction is not reproducible"
            );
            assert_blocks_close(&first, &expected, 1.0e-12, &format!("device {curvature:?}"));
        }
    }
}

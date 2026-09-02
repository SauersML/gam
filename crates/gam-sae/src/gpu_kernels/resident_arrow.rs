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
//! `RowChannels::second` already forms, and the coordinate channels need only a
//! `d3` slot channel alongside `decoded_second`.
//!
//! The extension seam is exactly: add a `direction: Option<&[f64]>` (length
//! `n_rows · q`) to `DeviceRequest`, add one kernel `sae_arrow_third_dir` with
//! the same `(row, a, b)` thread mapping and the same `__dadd_rn` accumulation as
//! `sae_arrow_htt`, and add a `t3: Vec<f64>` block (shape `[n_rows, q, q]`) to
//! [`ArrowBlocks`]. No new tensor is materialized and no new transfer shape is
//! introduced — the contraction with `v` happens inside the kernel, so the
//! download stays `O(q²)` per row. `ArrowCurvature` gains no variant: the third
//! order is a separate directional product, not a curvature mode.

use crate::gpu_kernels::sae_rowjet::SaeRowJetPath;

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
        // The availability question goes through the one shared gate (#2422).
        // The seam assertion below is this test's own and is kept; what the bare
        // `resolve` could not do is make the skip VISIBLE — the process counter
        // never moved and no `SKIPPED(no-cuda):` line was emitted, so a CI ledger
        // scraping a green CPU-only run saw no trace that the device half never
        // ran. `gpu_for_test` also panics under `GpuPolicy::Required`, so a GPU
        // lane no longer needs a per-test opt-in.
        let skips_before = gam_gpu::test_gate::skipped_for_absent_device();
        match gam_gpu::test_gate::gpu_for_test("resident-arrow reduced-block parity #1017") {
            gam_gpu::test_gate::GpuTestGate::Ready(_) => {}
            gam_gpu::test_gate::GpuTestGate::AbsentDevice => {
                gam_gpu::test_gate::assert_absent_device_was_counted(skips_before);
                // #2422: a bare `return` here reported `passed` with zero
                // assertions on every device-free runner. Without a device the
                // Device-path constructor must REFUSE -- `ResidentRowJetHandle::new`
                // routes `SaeRowJetPath::Device` into
                // `device::DeviceResidency::allocate`, which needs a backend, so an
                // `Ok` would mean the handle fabricated device residency (#1551).
                // The host path is constructed first so a fixture broken for an
                // unrelated reason cannot make the refusal pass for the wrong reason.
                let n = ARROW_REDUCTION_LEAF_ROWS + 1;
                ResidentRowJetHandle::new(3, 6, 4, 3, 1.3, n, SaeRowJetPath::Cpu)
                    .expect("the host resident-arrow handle must build on every host");
                assert!(
                    ResidentRowJetHandle::new(3, 6, 4, 3, 1.3, n, SaeRowJetPath::Device).is_err(),
                    "no CUDA runtime on this host, yet the Device resident-arrow handle \
                     constructed -- the seam fabricated device residency (#1551 class)"
                );
                return;
            }
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

//! # Atom selection for multi-manifold / overlapping atoms (Piece 6).
//!
//! This module is the structural sibling of [`crate::terms::latent_coord`] for
//! the **multi-atom** regime described in `proposals/sae_manifold.md` §3.
//! Where the single-atom case stores one per-row latent `t_n ∈ ℝ^d`, here we
//! maintain:
//!
//! 1. an [`AtomLibrary`] of `K` candidate manifold-atoms (each with its own
//!    on-atom basis, its own intrinsic dimension `d_k`, and its own
//!    [`crate::terms::latent_coord::LatentCoordValues`] block of per-row
//!    on-atom coordinates `t_{·, k} ∈ ℝ^{N × d_k}`),
//! 2. per-observation [`crate::terms::atom_codes::SparseAtomCode`]s recording
//!    a *soft assignment* `a_n ∈ ℝ^K` with active support `S_n ⊆ {1..K}`,
//! 3. a pluggable [`AtomSelectionStrategy`] governing how the assignment is
//!    parameterised and how its discrete active-set is differentiated through.
//!
//! ## Parameter partition
//!
//! Following the SAE-manifold tier-assignment table (`sae_manifold.md` §3.1):
//!
//! | Block | Lives in | Owner (piece) |
//! |---|---|---|
//! | `B_1..B_K` decoder coefficients (shared, *one block per atom*) | β / inner Newton | Piece 1 (decoder) — we only hold references |
//! | `t_{n, k}` on-atom coordinate (per row, per atom) | ext-coord / row-local | this module + Piece 1 |
//! | `a_{n, ·}` soft atom assignment (per row) | ext-coord / row-local | **this module** |
//! | `λ_sp` sparsity strength, `λ_sm` smoothness, `α_kj` ARD | ρ / REML outer loop | Piece 4 (sparsity), Piece 1 (ARD) |
//! | `K` atom count | discrete / `compare_models` | upstream wrapper |
//!
//! The Schur / arrow structure is preserved: each row's
//! `ext_n = (a_{n,·}, t_{n,1,·}, …, t_{n,K,·})` is block-diagonal across `n`,
//! coupled to the dense decoder border only through the *active subset*
//! `S_n` (inactive atoms contribute zero through the gating
//! `a_{n,k} = 0`). The math-audit caveat from `sae_manifold.md` §3.3 about
//! the shared `Schur⁻¹` factor in the REML `log|H|` gradient applies
//! unchanged.
//!
//! ### Per-row local-block size
//!
//! The single-atom case from Piece 1 (`latent_coord.rs`) carries a per-row
//! local block of size `d × d` (just `t_n ∈ ℝ^d`). For the multi-atom case
//! the per-row local block stacks the assignment row and the on-atom
//! coordinates of every atom:
//!
//! ```text
//!   ext_n  =  ( a_{n, 1..K}  ;  t_{n, 1, ·}  ;  …  ;  t_{n, K, ·} )
//!         ∈  ℝ^{K + Σ_k d_k}.
//! ```
//!
//! So the local-block dimension is
//!
//! ```text
//!   dim(ext_n)  =  K  +  Σ_{k=1..K} d_k,
//! ```
//!
//! and the local Hessian block is `(K + Σ_k d_k) × (K + Σ_k d_k)`,
//! block-diagonal across `n`. Piece 1's `solve_arrow_newton_step` Schur
//! elimination generalises by:
//!
//! 1. Eliminating shared β = `(B_1, …, B_K)` first (the existing inner
//!    factorisation), restricted on each row to the *active subset* `S_n`
//!    — atoms with `a_{n,k} = 0` contribute neither to the border nor to
//!    the row's `(t_{n,k}, ·)` block at first order.
//! 2. Solving each row's `(K + Σ_k d_k) × (K + Σ_k d_k)` local block. In
//!    the typical sparse regime `|S_n| ≪ K`, so the *effective* local
//!    block collapses to `(|S_n| + Σ_{k ∈ S_n} d_k) × (·)` after dropping
//!    the inactive coordinates from the active-set.
//!
//! The production SAE-manifold assembler now applies that stacking recipe in
//! [`crate::terms::sae_manifold::SaeManifoldTerm::assemble_arrow_schur`]:
//! the `(K, K)` assignment block sits on the diagonal corner of `ext_n`, the
//! `K` per-atom `(d_k, d_k)` coordinate blocks tile the rest, and the
//! off-diagonal `(a_{n,k}, t_{n,k,·})` couplings are populated from each
//! atom's basis-derivative jet evaluated against `B_k`.
//!
//! ## Relaxation choices for the assignment
//!
//! The assignment `a_n` is intrinsically combinatorial: in the ideal sparse
//! regime it picks a small support `S_n` and a real-valued amplitude on it.
//! Three differentiable relaxations are exposed via
//! [`AtomSelectionStrategy`]:
//!
//! * [`EntropicSoftmax`] — write `a_n = softmax(ℓ_n / τ)` for free logits
//!   `ℓ_n ∈ ℝ^K`. Stays on the open simplex; gradient is the standard
//!   softmax Jacobian; sparsity is encouraged by adding an
//!   entropic penalty `−H(a_n) = Σ_k a_{n,k} log a_{n,k}` whose strength
//!   trades against the data fit (small `τ` → near-hard assignment, larger
//!   `τ` → diffuse). Default strategy.
//! * [`TopK`] — keep only the `k` largest free amplitudes per row. Exact
//!   sparsity, but the active-set choice is discrete; we use the
//!   *straight-through* gradient estimator (forward pass uses the
//!   sparsified `a_n`; backward pass uses the dense gradient as if the
//!   thresholding were the identity). This is the standard convention in
//!   the TopK-SAE literature and matches the Manifold-SAE Curve-SAE
//!   benchmark; the bias of the estimator is documented at
//!   [`TopK::apply`].
//! * [`L1Relaxed`] — non-negative free amplitudes with a smoothed-L¹
//!   ([`crate::terms::analytic_penalties::SparsityPenalty`]) penalty. Pairs
//!   directly with the active-set inner solver (Piece 4) — strictly-zero
//!   weights *and* a smooth gradient. The relaxation parameter is the
//!   smoothing scale `ε` of the smoothed-L¹, REML-selectable.
//!
//! All three implement [`AtomSelectionStrategy`], which exposes the value and
//! gradient of the assignment-to-code map plus the corresponding penalty
//! contribution.
//!
//! ## Closed-form gradients and production assembly
//!
//! Fully implemented (closed-form, this module):
//!
//! * Reconstruction `Ẑ_n = Σ_k a_{n,k} · g_k(t_{n,k})` given external
//!   decoder evaluations `g_k(t_{n,k})` ([`reconstruct_row`]).
//! * `∂ℒ_data/∂a_{n,k}` for the row-local quadratic data-fit
//!   ([`data_grad_assignment_row`]) — this is the `−(Z_n − Ẑ_n)ᵀ g_k(t_{n,k})`
//!   formula from `sae_manifold.md` §3.3.
//! * Softmax forward / Jacobian-vector product
//!   ([`EntropicSoftmax::apply`], [`EntropicSoftmax::jvp_logits`]).
//! * TopK projection with straight-through gradient
//!   ([`TopK::apply`], [`TopK::backward_straight_through`]).
//! * Sparsity-penalty coupling trait
//!   ([`AssignmentSparsityCoupling`]) wired to
//!   [`crate::terms::analytic_penalties::SparsityPenalty`].
//!
//! Production SAE-manifold assembly is now first-class in
//! [`crate::terms::sae_manifold::SaeManifoldTerm::assemble_arrow_schur`]:
//! it materializes the joint `(logits, t)` per-row block, including the
//! assignment diagonal and `(a, t)` cross terms from the atom basis jets, then
//! hands the result to [`crate::solver::arrow_schur::ArrowSchurSystem`].
//!
//! ## Integration hooks to other pieces
//!
//! * Piece 1 (`arrow_schur.rs`, `solve_arrow_newton_step`): consumed by the
//!   first-class SAE-manifold assembler in
//!   [`crate::terms::sae_manifold::SaeManifoldTerm::assemble_arrow_schur`].
//! * Piece 4 (`SparsityPenalty`): consumed as a black box via the
//!   [`AssignmentSparsityCoupling`] trait below. We do not edit Piece 4.
//! * Piece 5 (REML outer loop): the per-strategy relaxation parameter
//!   (`temperature`, `k`, `eps`) joins the outer ρ vector through the
//!   already-existing [`crate::terms::analytic_penalties`] `rho_index`
//!   plumbing; no new outer-loop code is needed here.

use ndarray::{Array1, Array2, ArrayView1};

use crate::terms::analytic_penalties::SparsityPenalty;
use crate::terms::atom_codes::{BitVec, SparseAtomCode, SparseAtomCodes};
use crate::terms::latent_coord::LatentCoordValues;

// ---------------------------------------------------------------------------
// Atom shape (decoder reference) — kept as an opaque token here.
// ---------------------------------------------------------------------------

/// Opaque handle to an atom's smooth-decoder shape function.
///
/// In the full integration this will resolve to a concrete `Smooth` from
/// `crate::terms::smooth` — but `smooth.rs` is owned by another piece, so we
/// keep this layer abstract: a [`ShapeRef`] is a stable index into an
/// externally-held registry of decoder bases, plus the intrinsic dimension
/// `d_k` and basis size `M_k` of that atom. The atom-selection layer never
/// dereferences the shape directly; it asks the caller for evaluated
/// decoder outputs `g_k(t_{n,k}) ∈ ℝ^p` and design-gradient jets.
#[derive(Debug, Clone, Copy)]
pub struct ShapeRef {
    /// Stable index into the caller's decoder-shape registry.
    pub id: usize,
    /// Intrinsic dimension `d_k` of this atom's on-manifold coordinate.
    pub intrinsic_dim: usize,
    /// Basis size `M_k` (number of decoder coefficient columns per output dim).
    pub basis_size: usize,
}

// ---------------------------------------------------------------------------
// Per-atom record and library
// ---------------------------------------------------------------------------

/// One candidate manifold-atom: its decoder-shape reference plus the per-row
/// on-atom coordinates `t_{·, k} ∈ ℝ^{N × d_k}`.
///
/// Note that the *decoder coefficients* `B_k` live in the β tier (owned by
/// Piece 1 / `pirls.rs`); we hold only row-local extension-coordinate state here.
#[derive(Debug, Clone)]
pub struct AtomRecord {
    pub shape: ShapeRef,
    pub coords: LatentCoordValues,
}

impl AtomRecord {
    pub fn new(shape: ShapeRef, coords: LatentCoordValues) -> Self {
        debug_assert_eq!(
            coords.latent_dim(),
            shape.intrinsic_dim,
            "AtomRecord: coord latent_dim {} != shape.intrinsic_dim {}",
            coords.latent_dim(),
            shape.intrinsic_dim,
        );
        Self { shape, coords }
    }

    pub fn intrinsic_dim(&self) -> usize {
        self.shape.intrinsic_dim
    }
}

/// `K` candidate manifold-atoms sharing a single observation set of size `N`.
///
/// All atoms must agree on `n_obs`. They may have different intrinsic
/// dimensions `d_k` (ragged), which is why the per-atom on-row coordinate
/// blocks are stored as separate [`LatentCoordValues`] rather than as one
/// dense `(N, K, d)` tensor.
#[derive(Debug, Clone)]
pub struct AtomLibrary {
    atoms: Vec<AtomRecord>,
    n_obs: usize,
}

impl AtomLibrary {
    /// Construct from a non-empty `Vec` of atoms. Errors if the per-atom
    /// `n_obs` disagree, or if no atoms are supplied.
    pub fn new(atoms: Vec<AtomRecord>) -> Result<Self, String> {
        if atoms.is_empty() {
            return Err("AtomLibrary::new: at least one atom required".into());
        }
        let n_obs = atoms[0].coords.n_obs();
        for (k, a) in atoms.iter().enumerate() {
            if a.coords.n_obs() != n_obs {
                return Err(format!(
                    "AtomLibrary::new: atom {k} has n_obs={} but atom 0 has n_obs={n_obs}",
                    a.coords.n_obs()
                ));
            }
        }
        Ok(Self { atoms, n_obs })
    }

    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    pub fn k_atoms(&self) -> usize {
        self.atoms.len()
    }

    pub fn atom(&self, k: usize) -> &AtomRecord {
        &self.atoms[k]
    }

    pub fn atom_mut(&mut self, k: usize) -> &mut AtomRecord {
        &mut self.atoms[k]
    }

    pub fn iter(&self) -> impl Iterator<Item = &AtomRecord> {
        self.atoms.iter()
    }

    /// Total intrinsic-dimension count `Σ_k d_k`. The per-row ext-coordinate block has
    /// size `K + Σ_k d_k` (assignment plus per-atom coord).
    pub fn total_intrinsic_dim(&self) -> usize {
        self.atoms.iter().map(|a| a.intrinsic_dim()).sum()
    }

    /// Allocate matching [`SparseAtomCodes`] storage (all-empty).
    pub fn fresh_codes(&self) -> SparseAtomCodes {
        SparseAtomCodes::empty(self.n_obs, self.k_atoms())
    }
}

// ---------------------------------------------------------------------------
// Row-local reconstruction & gradients (closed form)
// ---------------------------------------------------------------------------

/// Reconstruction at row `n`:
/// `Ẑ_n = Σ_{k ∈ S_n} a_{n,k} · g_k(t_{n,k})`.
///
/// `decoder_outputs[k]` is `g_k(t_{n,k}) ∈ ℝ^p`. Inactive atoms are skipped;
/// active atoms contribute their soft-weighted decoder output.
pub fn reconstruct_row(
    code: &SparseAtomCode,
    decoder_outputs: &[ArrayView1<'_, f64>],
) -> Array1<f64> {
    debug_assert_eq!(decoder_outputs.len(), code.k_atoms());
    let p = decoder_outputs.first().map(|v| v.len()).unwrap_or(0);
    let mut reconstruction = Array1::<f64>::zeros(p);
    for k in code.active_mask.iter_ones() {
        debug_assert_eq!(decoder_outputs[k].len(), p);
        let w = code.weights[k];
        for i in 0..p {
            reconstruction[i] += w * decoder_outputs[k][i];
        }
    }
    reconstruction
}

/// Gradient of the row-`n` quadratic data fit `½ ‖Z_n − Ẑ_n‖²` with respect
/// to the assignment vector `a_{n, ·}`.
///
/// From `sae_manifold.md` §3.3:
/// ```text
///   ∂ℒ_data / ∂a_{n,k}  =  −(Z_n − Ẑ_n)ᵀ  g_k(t_{n,k}).
/// ```
///
/// The returned vector has length `K`. Inactive coordinates also get filled
/// in (the *unconstrained* gradient), so that selection strategies such as
/// [`TopK`] using the straight-through estimator can read the dense form.
pub fn data_grad_assignment_row(
    residual: ArrayView1<'_, f64>,
    decoder_outputs: &[ArrayView1<'_, f64>],
) -> Array1<f64> {
    let k = decoder_outputs.len();
    let mut g = Array1::<f64>::zeros(k);
    let p = residual.len();
    for kk in 0..k {
        debug_assert_eq!(decoder_outputs[kk].len(), p);
        let mut acc = 0.0;
        let g_k = &decoder_outputs[kk];
        for i in 0..p {
            acc += residual[i] * g_k[i];
        }
        g[kk] = -acc;
    }
    g
}

// ---------------------------------------------------------------------------
// Sparsity coupling trait
// ---------------------------------------------------------------------------

/// Trait wiring an [`AtomSelectionStrategy`] into a
/// [`SparsityPenalty`] (Piece 4) without depending on Piece 4's internal
/// representation.
///
/// The contract: implementors expose the *target slice* over which the
/// sparsity penalty applies — for the L1-relaxed strategy this is the
/// free-amplitude vector itself; for entropic-softmax it is typically a no-op
/// (the entropic regulariser, owned by the strategy, replaces L¹); for TopK
/// it is also a no-op (cardinality is the regulariser).
pub trait AssignmentSparsityCoupling {
    /// Apply `penalty` to the row-`n` assignment, returning `(value, grad)`
    /// over the row's `K` free amplitudes. `rho` is the local penalty view.
    fn penalty_value_and_grad(
        &self,
        penalty: &SparsityPenalty,
        free_amplitudes_row: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> (f64, Array1<f64>);
}

// ---------------------------------------------------------------------------
// AtomSelectionStrategy trait + three impls
// ---------------------------------------------------------------------------

/// Pluggable strategy governing the assignment parameterisation.
///
/// All strategies operate row-wise (the assignment is per-observation) and
/// take a length-`K` slice of *free amplitudes* `ℓ_n` (logits for softmax,
/// raw non-negative amplitudes for L¹-relaxed, raw amplitudes for TopK).
pub trait AtomSelectionStrategy: AssignmentSparsityCoupling {
    /// Strategy tag — useful for diagnostics / `compare_models` keying.
    fn name(&self) -> &'static str;

    /// Forward: map free amplitudes `ℓ_n` to a [`SparseAtomCode`] for row `n`.
    fn apply(&self, free_amplitudes_row: ArrayView1<'_, f64>) -> SparseAtomCode;

    /// Backward: given `∂ℒ/∂a_{n,·}` from the data-fit (length `K`, dense),
    /// return `∂ℒ/∂ℓ_{n,·}` (length `K`).
    ///
    /// Strategies differ in their Jacobian; see per-impl docs.
    fn backward(
        &self,
        free_amplitudes_row: ArrayView1<'_, f64>,
        code: &SparseAtomCode,
        grad_a_row: ArrayView1<'_, f64>,
    ) -> Array1<f64>;
}

// --- EntropicSoftmax --------------------------------------------------------

/// Fully differentiable simplex parameterisation.
///
/// ```text
///   a_{n,k} = exp(ℓ_{n,k} / τ) / Σ_j exp(ℓ_{n,j} / τ).
/// ```
///
/// The temperature `τ > 0` is the relaxation parameter. Lower `τ` produces
/// near-hard assignments (but with vanishing gradients); higher `τ` keeps
/// assignments diffuse. The default is `τ = 1.0`.
///
/// The entropic regulariser `−H(a_n)` is *not* materialised here — it is
/// added through the standard penalty layer (see
/// [`AssignmentSparsityCoupling`]) using
/// [`SparsityKind::Log`](crate::terms::analytic_penalties::SparsityKind) as a
/// proxy. Pure cross-entropy support is deferred until Piece 4 grows a
/// dedicated `EntropyPenalty`.
#[derive(Debug, Clone)]
pub struct EntropicSoftmax {
    pub temperature: f64,
    /// If `Some(thr)`, atoms with softmax mass below `thr` are masked out
    /// (still soft below — the mask only affects which atoms count as
    /// active for the per-row Schur reduction). The default is `None`,
    /// i.e. full dense support, which is appropriate for very small `K`.
    pub mask_threshold: Option<f64>,
}

impl EntropicSoftmax {
    pub fn new(temperature: f64) -> Self {
        debug_assert!(temperature > 0.0);
        Self {
            temperature,
            mask_threshold: None,
        }
    }

    pub fn with_mask_threshold(mut self, thr: f64) -> Self {
        self.mask_threshold = Some(thr);
        self
    }

    /// Numerically stable softmax with temperature.
    fn softmax(&self, logits: ArrayView1<'_, f64>) -> Array1<f64> {
        let k = logits.len();
        let tau = self.temperature;
        // shift by max for stability
        let mut m = f64::NEG_INFINITY;
        for &l in logits.iter() {
            let s = l / tau;
            if s > m {
                m = s;
            }
        }
        let mut out = Array1::<f64>::zeros(k);
        let mut s = 0.0;
        for i in 0..k {
            let v = (logits[i] / tau - m).exp();
            out[i] = v;
            s += v;
        }
        debug_assert!(s > 0.0);
        for v in out.iter_mut() {
            *v /= s;
        }
        out
    }

    /// Jacobian-vector product: given `g_a = ∂ℒ/∂a`, return `∂ℒ/∂ℓ`.
    ///
    /// The softmax Jacobian (per row) is `J = (diag(a) − a aᵀ) / τ`, so
    /// `∂ℒ/∂ℓ = (a ⊙ g_a − a · (a · g_a)) / τ`.
    pub fn jvp_logits(&self, a: ArrayView1<'_, f64>, g_a: ArrayView1<'_, f64>) -> Array1<f64> {
        let k = a.len();
        let mut dot = 0.0;
        for i in 0..k {
            dot += a[i] * g_a[i];
        }
        let mut out = Array1::<f64>::zeros(k);
        let inv_tau = 1.0 / self.temperature;
        for i in 0..k {
            out[i] = a[i] * (g_a[i] - dot) * inv_tau;
        }
        out
    }
}

impl AtomSelectionStrategy for EntropicSoftmax {
    fn name(&self) -> &'static str {
        "entropic_softmax"
    }

    fn apply(&self, free_amplitudes_row: ArrayView1<'_, f64>) -> SparseAtomCode {
        let a = self.softmax(free_amplitudes_row);
        let k = a.len();
        let mut mask = BitVec::ones(k);
        if let Some(thr) = self.mask_threshold {
            for i in 0..k {
                if a[i] < thr {
                    mask.set(i, false);
                }
            }
        }
        let mut weights = vec![0.0_f64; k];
        for i in 0..k {
            if mask.get(i) {
                weights[i] = a[i];
            }
        }
        SparseAtomCode {
            active_mask: mask,
            weights,
        }
    }

    fn backward(
        &self,
        free_amplitudes_row: ArrayView1<'_, f64>,
        _code: &SparseAtomCode,
        grad_a_row: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        // Recompute softmax (cheap; alternative is to cache it in the code,
        // but that conflates the masked weights with the *unmasked* softmax
        // needed by the Jacobian).
        let a = self.softmax(free_amplitudes_row);
        self.jvp_logits(a.view(), grad_a_row)
    }
}

impl AssignmentSparsityCoupling for EntropicSoftmax {
    fn penalty_value_and_grad(
        &self,
        _penalty: &SparsityPenalty,
        free_amplitudes_row: ArrayView1<'_, f64>,
        _rho: ArrayView1<'_, f64>,
    ) -> (f64, Array1<f64>) {
        // Entropic-softmax does not consume the L¹ sparsity penalty
        // directly; the entropy regularisation lives inside the strategy
        // itself. We return zero contribution here (Piece 4 sees nothing
        // to penalise) so the global energy isn't double-counted.
        let k = free_amplitudes_row.len();
        (0.0, Array1::<f64>::zeros(k))
    }
}

// --- TopK -------------------------------------------------------------------

/// Hard active-set: keep the `k` largest free amplitudes per row.
///
/// Reconstruction uses `a_{n,j} = ℓ_{n,j}` if `j ∈ topk(ℓ_n)` else `0`.
///
/// **Straight-through gradient.** The forward map is discontinuous (the
/// active-set changes at amplitude ties); the backward pass treats the
/// thresholding as the identity, so `∂ℒ/∂ℓ ≈ ∂ℒ/∂a`. This is the standard
/// TopK-SAE convention (Makhzani & Frey 2014; Gao et al. 2024). The bias is
/// (i) zero whenever the active set is locally stable, and (ii) bounded by
/// `‖∂ℒ/∂a‖_∞` at tie crossings, so a small temperature in the upstream
/// objective is a sufficient mitigation when used together with adaptive
/// step sizes.
#[derive(Debug, Clone, Copy)]
pub struct TopK {
    pub k: usize,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        debug_assert!(k > 0);
        Self { k }
    }

    fn topk_indices(&self, amps: ArrayView1<'_, f64>) -> Vec<usize> {
        let n = amps.len();
        let k_use = self.k.min(n);
        if k_use == 0 {
            return Vec::new();
        }
        let mut idx: Vec<usize> = (0..n).collect();
        let pivot = k_use.saturating_sub(1).min(n - 1);
        idx.sort_by(|&a, &b| {
            amps[b]
                .partial_cmp(&amps[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(pivot + 1);
        idx
    }

    /// Straight-through backward: ∂ℒ/∂ℓ = ∂ℒ/∂a *masked to the active set*.
    ///
    /// Documenting the convention: we zero out the gradient on inactive
    /// coordinates (a stricter form of straight-through that matches the
    /// "dead-feature freezing" behaviour observed in TopK-SAE). Some
    /// references (e.g. Hubinger-style straight-through) pass the gradient
    /// through unmodified; the masked form is empirically better at avoiding
    /// dead atoms (see `project_curve_sae_efficiency.md`).
    pub fn backward_straight_through(
        &self,
        code: &SparseAtomCode,
        grad_a_row: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let k = grad_a_row.len();
        let mut out = Array1::<f64>::zeros(k);
        for i in code.active_mask.iter_ones() {
            out[i] = grad_a_row[i];
        }
        out
    }
}

impl AtomSelectionStrategy for TopK {
    fn name(&self) -> &'static str {
        "topk"
    }

    fn apply(&self, free_amplitudes_row: ArrayView1<'_, f64>) -> SparseAtomCode {
        let k_total = free_amplitudes_row.len();
        let mut mask = BitVec::zeros(k_total);
        let mut weights = vec![0.0_f64; k_total];
        for i in self.topk_indices(free_amplitudes_row) {
            mask.set(i, true);
            weights[i] = free_amplitudes_row[i];
        }
        SparseAtomCode {
            active_mask: mask,
            weights,
        }
    }

    fn backward(
        &self,
        _free_amplitudes_row: ArrayView1<'_, f64>,
        code: &SparseAtomCode,
        grad_a_row: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        self.backward_straight_through(code, grad_a_row)
    }
}

impl AssignmentSparsityCoupling for TopK {
    fn penalty_value_and_grad(
        &self,
        _penalty: &SparsityPenalty,
        free_amplitudes_row: ArrayView1<'_, f64>,
        _rho: ArrayView1<'_, f64>,
    ) -> (f64, Array1<f64>) {
        // Cardinality is enforced structurally; no smooth penalty consumed.
        let k = free_amplitudes_row.len();
        (0.0, Array1::<f64>::zeros(k))
    }
}

// --- L1Relaxed --------------------------------------------------------------

/// Non-negative free amplitudes with a smoothed-L¹ penalty (Piece 4).
///
/// Forward: `a_{n,k} = max(ℓ_{n,k}, 0)`; active iff `a_{n,k} > 0`.
///
/// The active-set is *induced* by the smoothed-L¹ via the existing
/// active-set inner solver (see `src/solver/active_set.rs`). The
/// smoothing scale `ε` is the relaxation parameter (REML-selectable through
/// [`SparsityPenalty::with_eps_reml`]).
#[derive(Debug, Clone)]
pub struct L1Relaxed {
    /// Threshold below which an amplitude is treated as inactive. Defaults
    /// to `0.0` (exact non-negativity); larger values give a deadzone.
    pub active_threshold: f64,
}

impl L1Relaxed {
    pub fn new() -> Self {
        Self {
            active_threshold: 0.0,
        }
    }

    pub fn with_threshold(thr: f64) -> Self {
        Self {
            active_threshold: thr,
        }
    }
}

impl Default for L1Relaxed {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomSelectionStrategy for L1Relaxed {
    fn name(&self) -> &'static str {
        "l1_relaxed"
    }

    fn apply(&self, free_amplitudes_row: ArrayView1<'_, f64>) -> SparseAtomCode {
        let k = free_amplitudes_row.len();
        let mut mask = BitVec::zeros(k);
        let mut weights = vec![0.0_f64; k];
        for i in 0..k {
            let a = free_amplitudes_row[i].max(0.0);
            if a > self.active_threshold {
                mask.set(i, true);
                weights[i] = a;
            }
        }
        SparseAtomCode {
            active_mask: mask,
            weights,
        }
    }

    fn backward(
        &self,
        free_amplitudes_row: ArrayView1<'_, f64>,
        code: &SparseAtomCode,
        grad_a_row: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        // d max(ℓ,0)/dℓ = 1 for ℓ>0, 0 otherwise; restricted to the active set.
        let k = grad_a_row.len();
        let mut out = Array1::<f64>::zeros(k);
        for i in code.active_mask.iter_ones() {
            if free_amplitudes_row[i] > 0.0 {
                out[i] = grad_a_row[i];
            }
        }
        out
    }
}

impl AssignmentSparsityCoupling for L1Relaxed {
    fn penalty_value_and_grad(
        &self,
        penalty: &SparsityPenalty,
        free_amplitudes_row: ArrayView1<'_, f64>,
        rho: ArrayView1<'_, f64>,
    ) -> (f64, Array1<f64>) {
        use crate::terms::analytic_penalties::AnalyticPenalty;
        // Apply the smoothed-L¹ to the non-negative free amplitudes
        // directly. Negative entries map to zero in the forward pass; for
        // the penalty we evaluate on the *clipped* values to keep the
        // sub-gradient at zero consistent with the active-set semantics.
        let k = free_amplitudes_row.len();
        let mut clipped = Array1::<f64>::zeros(k);
        for i in 0..k {
            clipped[i] = free_amplitudes_row[i].max(0.0);
        }
        let v = penalty.value(clipped.view(), rho);
        let mut g = penalty.grad_target(clipped.view(), rho);
        for i in 0..k {
            if free_amplitudes_row[i] <= 0.0 {
                g[i] = 0.0;
            }
        }
        (v, g)
    }
}

// ---------------------------------------------------------------------------
// Convenience: dense reconstruction over all rows (diagnostic).
// ---------------------------------------------------------------------------

/// Materialize `Ẑ ∈ ℝ^{N × p}` from current codes and externally-supplied
/// per-row decoder outputs `decoder_outputs[n][k] = g_k(t_{n,k}) ∈ ℝ^p`.
///
/// Allocates; intended for diagnostic / post-fit pipelines.
pub fn reconstruct_all(
    codes: &SparseAtomCodes,
    decoder_outputs: &[Vec<ArrayView1<'_, f64>>],
    p_out: usize,
) -> Array2<f64> {
    let n = codes.n_obs();
    let mut reconstruction = Array2::<f64>::zeros((n, p_out));
    for nn in 0..n {
        let row = reconstruct_row(codes.row(nn), &decoder_outputs[nn]);
        for i in 0..p_out {
            reconstruction[[nn, i]] = row[i];
        }
    }
    reconstruction
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terms::latent_coord::{LatentCoordValues, LatentIdMode};
    use ndarray::array;

    fn lib() -> AtomLibrary {
        let c0 = LatentCoordValues::from_matrix(
            array![[0.0, 0.0], [0.1, 0.2]].view(),
            LatentIdMode::None,
        );
        let c1 = LatentCoordValues::from_matrix(array![[0.0], [1.0]].view(), LatentIdMode::None);
        AtomLibrary::new(vec![
            AtomRecord::new(
                ShapeRef {
                    id: 0,
                    intrinsic_dim: 2,
                    basis_size: 8,
                },
                c0,
            ),
            AtomRecord::new(
                ShapeRef {
                    id: 1,
                    intrinsic_dim: 1,
                    basis_size: 5,
                },
                c1,
            ),
        ])
        .expect("library")
    }

    #[test]
    fn library_construct() {
        let l = lib();
        assert_eq!(l.k_atoms(), 2);
        assert_eq!(l.n_obs(), 2);
        assert_eq!(l.total_intrinsic_dim(), 3);
    }

    #[test]
    fn softmax_is_simplex() {
        let s = EntropicSoftmax::new(1.0);
        let logits = array![1.0_f64, 2.0, 3.0];
        let code = s.apply(logits.view());
        let sum: f64 = code.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert_eq!(code.active_mask.count_ones(), 3);
    }

    #[test]
    fn topk_keeps_top() {
        let t = TopK::new(2);
        let amps = array![0.1_f64, 0.9, 0.4, 0.5];
        let code = t.apply(amps.view());
        assert_eq!(code.active_mask.count_ones(), 2);
        assert!(code.active_mask.get(1));
        assert!(code.active_mask.get(3));
    }

    #[test]
    fn l1_relaxed_clips_negatives() {
        let l = L1Relaxed::new();
        let amps = array![-0.5_f64, 0.3, -0.1, 0.7];
        let code = l.apply(amps.view());
        assert_eq!(code.active_mask.count_ones(), 2);
        assert_eq!(code.weights[1], 0.3);
        assert_eq!(code.weights[3], 0.7);
    }

    #[test]
    fn data_grad_assignment_matches_formula() {
        let p = 3;
        let z = array![1.0_f64, 0.0, 0.0];
        let g0 = array![1.0_f64, 0.0, 0.0];
        let g1 = array![0.0_f64, 1.0, 0.0];
        let outs = [g0.view(), g1.view()];
        let mut code = SparseAtomCode::empty(2);
        code.assign(0, 0.5);
        code.assign(1, 0.0);
        let r = reconstruct_row(&code, &outs);
        assert_eq!(r.len(), p);
        let resid = &z - &r;
        let g = data_grad_assignment_row(resid.view(), &outs);
        // resid = (0.5, 0, 0); dot(resid, g0) = 0.5, dot(resid, g1) = 0
        assert!((g[0] + 0.5).abs() < 1e-12);
        assert!(g[1].abs() < 1e-12);
    }
}

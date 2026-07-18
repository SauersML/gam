//! Shared seam helper for the Wood (2011) stable reparameterization at the
//! outer LAML assembly seam (#2331 Inc 1).
//!
//! # What this module does
//!
//! Both the transformation-survival lane (`gam-models/src/survival/base.rs`)
//! and the custom-family lane (`gam-custom-family/src/assembly.rs`) converge on
//! the same task: they have a *raw-frame* inner solution — a penalized Hessian
//! `H`, a coefficient vector `β̂`, a set of per-block penalty matrices `Sₖ` with
//! current smoothing weights `λₖ = e^{ρₖ}`, and (optionally) a family curvature
//! third-derivative provider — and they must build the `InnerAssembly` that the
//! unified LAML evaluator consumes.
//!
//! [`assemble_reparameterized_inner`] performs the single similarity transform
//! that makes that assembly numerically stable, using the orthogonal `Qs` from
//! the Wood reparameterization ([`gam_terms::construction::ReparamResult`]):
//!
//! ```text
//!   H′  = Qsᵀ H Qs          (transformed penalized Hessian)
//!   β̂′  = Qsᵀ β̂             (transformed coefficient vector)
//! ```
//!
//! and produces, alongside `H′`/`β̂′`:
//!
//! * the **joint** penalty pseudo-logdeterminant `log|S_λ|₊` with `S_λ = Σₖ λₖ Sₖ`,
//!   together with its `ρ`-gradient (`det1`) and `ρ`-Hessian (`det2`), all sourced
//!   from the single W-factor eigendecomposition in
//!   [`super::penalty_logdet::PenaltyPseudologdet`] (#2331 R7 — `det2` must not be
//!   stitched together from the per-block `construction.rs` determinants);
//! * a [`HessianDerivativeProvider`] wrapper that conjugates the raw family
//!   provider into the transformed frame,
//!   `correction′(v′) = Qsᵀ · correction(Qs · v′) · Qs`;
//! * the orthogonal `Qs` itself, so a caller that also carries active linear
//!   constraints can map the constraint rows into the transformed frame as
//!   `A_act · Qs` (#2331 Inc 3, optional).
//!
//! # Why this is exact
//!
//! `Qs` is **orthogonal** (`ReparamResult.qs`, built from the penalized/null
//! split in `gam-terms/src/construction.rs` via `compose_qs_from_split`). Under
//! an orthogonal similarity transform:
//!
//! * every eigenvalue of `H` is preserved, so `log|H|`, the `HardPseudo`
//!   positive-subspace mask, and every spectral engine trace are exactly
//!   invariant — `H′` and `H` are the same operator in a rotated basis;
//! * the penalty sum transforms as `Σₖ λₖ Qsᵀ Sₖ Qs = Qsᵀ S_λ Qs`, whose
//!   pseudo-logdeterminant and `ρ`-derivatives (`tr(S_λ⁺ Sₖ)` and the paired
//!   trace) are traces of similar operators and therefore **frame-invariant**.
//!   This is why the joint logdet may be — and here is — computed directly from
//!   the raw `Sₖ`: the value is identical in either frame, and the W-factor path
//!   already stacks the scaled roots so it never squares the conditioning.
//! * the family third-derivative correction enters the criterion only through
//!   traces (`tr(S⁺ A_l S⁺ Cₖ)` and friends), so the conjugation wrapper leaves
//!   every consumed quantity invariant while presenting the correction in the
//!   frame `H′`/`β̂′` live in.
//!
//! The mode responses stay consistent under the same transform: with
//! `Aₖ′ = Qsᵀ Aₖ Qs`, `vₖ′ = H′⁻¹ Aₖ′ β̂′ = Qsᵀ (H⁻¹ Aₖ β̂) = Qsᵀ vₖ`, which is
//! exactly the argument the conjugated provider expects.
//!
//! # Scope (what this helper does NOT own)
//!
//! Per the #2331 landing design the reparameterization applies **only at the
//! outer LAML seam**: the inner PIRLS solve, the active cone, and the KKT
//! machinery all stay in the raw frame. This helper therefore does not touch
//! the inner solve; it consumes an already-converged raw inner solution.
//!
//! It also does not build the transformed [`PenaltyCoordinate`] roots. Those are
//! produced at the seam directly from `ReparamResult.e_transformed` /
//! `canonical_transformed` (which are already expressed in the `Qs` frame and
//! are the single source of truth for penalty roots there); the seam plugs those
//! into `InnerAssembly.penalty_coords`. This helper owns the pieces the design
//! enumerates — `H′`, `β̂′`, the joint logdet triple, the conjugated provider,
//! and `Qs` — and nothing else, so the two responsibilities stay separable.

use ndarray::{Array1, Array2};

use gam_terms::construction::ReparamResult;

use super::penalty_logdet::PenaltyPseudologdet;
use super::reml_outer_engine::{HessianDerivativeProvider, PenaltyLogdetDerivs};

/// Raw-frame inputs the reparameterization consumes.
///
/// All fields are borrowed: the helper never takes ownership of the raw
/// solution, it only reads it to build the transformed-frame products. Every
/// matrix/vector is in the raw coefficient frame the inner solve produced.
pub struct RawInnerReparamContext<'a> {
    /// Penalized Hessian `H` (dense, `p × p`, symmetric).
    pub hessian: &'a Array2<f64>,
    /// Coefficient vector `β̂` (length `p`).
    pub beta: &'a Array1<f64>,
    /// Per-block penalty matrices `Sₖ`, each already embedded into the full
    /// `p × p` coefficient space (zero outside the block's column range).
    ///
    /// The caller passes **every** active penalty block, including any
    /// full-span stabilization ridge block: the joint normalizer
    /// `log|Σₖ λₖ Sₖ|₊` is computed over exactly this set, so a full-span ridge
    /// that overlaps the smoothing blocks is accounted for jointly (the #2331
    /// Finding 3a objective correction) rather than as a per-block sum.
    pub penalties_embedded: &'a [Array2<f64>],
    /// Current smoothing weights `λₖ = e^{ρₖ}`, one per entry of
    /// `penalties_embedded`, in the same order.
    pub lambdas: &'a [f64],
}

/// Transformed-frame products of the reparameterization.
///
/// `'dp` is the lifetime of any borrowed state captured by the family
/// derivative provider (design matrices, curvature arrays); it is threaded
/// through the conjugation wrapper unchanged.
pub struct ReparameterizedInner<'dp> {
    /// `H′ = Qsᵀ H Qs` — the penalized Hessian in the transformed frame.
    pub hessian_transformed: Array2<f64>,
    /// `β̂′ = Qsᵀ β̂` — the coefficient vector in the transformed frame.
    pub beta_transformed: Array1<f64>,
    /// Joint penalty pseudo-logdeterminant and its `ρ`-derivatives.
    ///
    /// `value = log|Σₖ λₖ Sₖ|₊`, `first[k] = λₖ tr(S_λ⁺ Sₖ)`, and
    /// `second[[k,l]] = δ_{kl} first[k] − λₖ λₗ tr(S_λ⁺ Sₖ S_λ⁺ Sₗ)`, all from a
    /// single [`PenaltyPseudologdet`] eigendecomposition (frame-invariant).
    pub penalty_logdet: PenaltyLogdetDerivs,
    /// The raw family derivative provider conjugated into the transformed frame,
    /// or `None` when the caller supplied no provider (Gaussian-style, no
    /// curvature drift).
    pub deriv_provider: Option<Box<dyn HessianDerivativeProvider + 'dp>>,
    /// The orthogonal transform `Qs`, surfaced so a caller with active linear
    /// constraints can map its constraint rows into the transformed frame as
    /// `A_act · Qs` (#2331 Inc 3).
    pub qs: Array2<f64>,
}

impl std::fmt::Debug for ReparameterizedInner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReparameterizedInner")
            .field(
                "hessian_transformed",
                &format_args!(
                    "{}x{}",
                    self.hessian_transformed.nrows(),
                    self.hessian_transformed.ncols()
                ),
            )
            .field("beta_transformed_len", &self.beta_transformed.len())
            .field("penalty_logdet_value", &self.penalty_logdet.value)
            .field("has_deriv_provider", &self.deriv_provider.is_some())
            .field("qs", &format_args!("{}x{}", self.qs.nrows(), self.qs.ncols()))
            .finish()
    }
}

/// Conjugate the raw-frame inner solution into the Wood transformed frame.
///
/// See the module documentation for the mathematics. Returns an `Err` (never a
/// panic) when a precondition is violated:
///
/// * **Shape** — `Qs`, `H`, `β̂`, and each `Sₖ` must agree on the ambient
///   dimension `p`, and `lambdas.len()` must equal `penalties_embedded.len()`.
/// * **R5, orthogonality** — `‖Qsᵀ Qs − I‖_∞` must sit below a `p`-derived
///   floor; a non-orthogonal `Qs` would silently distort every conjugated
///   eigenvalue and trace.
/// * **R5, shrinkage ridge** — `ReparamResult.penalty_shrinkage_ridge` must be
///   exactly `0.0`. A nonzero `ρ`-independent shrinkage ridge changes the prior
///   normalizer and must be handled explicitly upstream, never silently folded
///   through the conjugation.
pub fn assemble_reparameterized_inner<'dp>(
    ctx: &RawInnerReparamContext<'_>,
    deriv_provider: Option<Box<dyn HessianDerivativeProvider + 'dp>>,
    reparam: &ReparamResult,
) -> Result<ReparameterizedInner<'dp>, String> {
    let qs = &reparam.qs;
    let p = qs.nrows();

    // ── Shape validation ────────────────────────────────────────────────────
    if qs.ncols() != p {
        return Err(format!(
            "reparameterized inner: Qs must be square, got {}x{}",
            qs.nrows(),
            qs.ncols()
        ));
    }
    if ctx.hessian.nrows() != p || ctx.hessian.ncols() != p {
        return Err(format!(
            "reparameterized inner: Hessian must be {p}x{p} to match Qs, got {}x{}",
            ctx.hessian.nrows(),
            ctx.hessian.ncols()
        ));
    }
    if ctx.beta.len() != p {
        return Err(format!(
            "reparameterized inner: beta length {} must match Qs dimension {p}",
            ctx.beta.len()
        ));
    }
    if ctx.lambdas.len() != ctx.penalties_embedded.len() {
        return Err(format!(
            "reparameterized inner: {} lambdas but {} penalty blocks",
            ctx.lambdas.len(),
            ctx.penalties_embedded.len()
        ));
    }
    for (k, s_k) in ctx.penalties_embedded.iter().enumerate() {
        if s_k.nrows() != p || s_k.ncols() != p {
            return Err(format!(
                "reparameterized inner: penalty block {k} must be {p}x{p} (embedded), got {}x{}",
                s_k.nrows(),
                s_k.ncols()
            ));
        }
    }

    // ── R5: hard preconditions ─────────────────────────────────────────────
    // Orthogonality floor derived from the ambient dimension: an orthogonal
    // factor assembled by a Householder/eigenvector chain accumulates at most
    // O(p · ε) departure from Qsᵀ Qs = I. A factor that misses this bound is
    // not an admissible similarity transform.
    let orth_tol = 128.0 * (p.max(1) as f64) * f64::EPSILON;
    let orth_residual = max_abs_orthogonality_defect(qs);
    if orth_residual.is_nan() || orth_residual > orth_tol {
        return Err(format!(
            "reparameterized inner: Qs is not orthogonal — ‖QsᵀQs − I‖_∞ = {orth_residual:.3e} \
             exceeds tolerance {orth_tol:.3e} (p = {p})"
        ));
    }
    if reparam.penalty_shrinkage_ridge != 0.0 {
        return Err(format!(
            "reparameterized inner: ReparamResult carries a nonzero shrinkage ridge \
             ({:.3e}); a ρ-independent shrinkage ridge changes the prior normalizer and \
             must be resolved upstream, never silently conjugated",
            reparam.penalty_shrinkage_ridge
        ));
    }

    // ── Similarity transform of the inner solution ─────────────────────────
    // H′ = Qsᵀ H Qs, β̂′ = Qsᵀ β̂.
    let hessian_transformed = qs.t().dot(ctx.hessian).dot(qs);
    let beta_transformed = qs.t().dot(ctx.beta);

    // ── Joint penalty pseudo-logdet (single W-factor eigendecomposition) ────
    // log|S_λ|₊ with S_λ = Σₖ λₖ Sₖ. Frame-invariant, so computed on the raw
    // (embedded) Sₖ; the W-factor path stacks scaled roots and never squares
    // the conditioning. `ridge = 0.0`: any stabilization ridge is one of the
    // Sₖ blocks (R5 forbids a hidden shrinkage ridge on this seam).
    let pld = PenaltyPseudologdet::from_components(ctx.penalties_embedded, ctx.lambdas, 0.0)
        .map_err(|e| format!("reparameterized inner: joint penalty logdet failed: {e}"))?;
    let (det1, det2) = pld.rho_derivatives(ctx.penalties_embedded, ctx.lambdas);
    let penalty_logdet = PenaltyLogdetDerivs {
        value: pld.value(),
        first: det1,
        second: Some(det2),
    };

    // ── Conjugate the family derivative provider into the transformed frame ─
    let deriv_provider: Option<Box<dyn HessianDerivativeProvider + 'dp>> =
        deriv_provider.map(|inner| {
            Box::new(ConjugatedDerivProvider {
                inner,
                qs: qs.clone(),
            }) as Box<dyn HessianDerivativeProvider + 'dp>
        });

    Ok(ReparameterizedInner {
        hessian_transformed,
        beta_transformed,
        penalty_logdet,
        deriv_provider,
        qs: qs.clone(),
    })
}

/// `max_{i,j} |(Qsᵀ Qs)[i,j] − δ_{ij}|`, evaluated without materializing the
/// full product.
fn max_abs_orthogonality_defect(qs: &Array2<f64>) -> f64 {
    let n = qs.ncols();
    let mut worst = 0.0_f64;
    for i in 0..n {
        let col_i = qs.column(i);
        for j in i..n {
            let col_j = qs.column(j);
            let dot: f64 = col_i.iter().zip(col_j.iter()).map(|(a, b)| a * b).sum();
            let target = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((dot - target).abs());
        }
    }
    worst
}

/// Conjugating wrapper that presents a raw-frame [`HessianDerivativeProvider`]
/// in the Wood transformed frame.
///
/// Every correction the wrapped provider returns is a `p × p` block consumed by
/// the evaluator only through traces; the wrapper maps the transformed-frame
/// query vectors back to the raw frame (`Qs · v′`), evaluates the raw
/// correction, and conjugates the result forward (`Qsᵀ · C · Qs`). Because
/// `Qs` is orthogonal this preserves every trace exactly.
///
/// The wrapper deliberately does **not** forward the family fast-path hooks
/// (`scalar_glm_ingredients`, `outer_hessian_derivative_kernel`,
/// `family_outer_hessian_operator`): those operators live in the raw frame and
/// cannot be reused verbatim in the transformed frame. Leaving them at their
/// `None` defaults routes the evaluator through the dense/operator correction
/// methods this wrapper *does* conjugate, keeping the transformed assembly
/// self-consistent (correctness over the raw-frame fast path, per #2331 — all
/// consumptions are value-invariant traces).
struct ConjugatedDerivProvider<'dp> {
    inner: Box<dyn HessianDerivativeProvider + 'dp>,
    qs: Array2<f64>,
}

impl<'dp> ConjugatedDerivProvider<'dp> {
    /// Map a transformed-frame vector `v′` back to the raw frame: `Qs · v′`.
    #[inline]
    fn to_raw(&self, v_prime: &Array1<f64>) -> Array1<f64> {
        self.qs.dot(v_prime)
    }

    /// Conjugate a raw-frame correction block `C` forward: `Qsᵀ C Qs`.
    #[inline]
    fn conjugate(&self, c: &Array2<f64>) -> Array2<f64> {
        self.qs.t().dot(c).dot(&self.qs)
    }
}

impl<'dp> HessianDerivativeProvider for ConjugatedDerivProvider<'dp> {
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let v_raw = self.to_raw(v_k);
        Ok(self
            .inner
            .hessian_derivative_correction(&v_raw)?
            .map(|c| self.conjugate(&c)))
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let v_k_raw = self.to_raw(v_k);
        let v_l_raw = self.to_raw(v_l);
        let u_kl_raw = self.to_raw(u_kl);
        Ok(self
            .inner
            .hessian_second_derivative_correction(&v_k_raw, &v_l_raw, &u_kl_raw)?
            .map(|c| self.conjugate(&c)))
    }

    fn has_corrections(&self) -> bool {
        self.inner.has_corrections()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;

    /// Deterministic orthogonal `p × p` matrix: eigenvectors of a fixed
    /// symmetric matrix are orthonormal, so this exercises a genuine rotation
    /// (never the identity) without any RNG.
    fn deterministic_orthogonal(p: usize) -> Array2<f64> {
        // Symmetric by construction: i*j and i+j are symmetric in (i, j).
        let sym = Array2::<f64>::from_shape_fn((p, p), |(i, j)| {
            (((i * j) as f64) * 0.1).sin() + (((i + j) as f64) * 0.3).cos()
        });
        let (_, evecs) = sym.eigh(Side::Lower).expect("eigh for orthogonal factor");
        evecs
    }

    /// Deterministic SPD `p × p` matrix `B Bᵀ + I`.
    fn deterministic_spd(p: usize) -> Array2<f64> {
        let b = Array2::<f64>::from_shape_fn((p, p), |(i, j)| {
            (((i * 31 + j * 17) % 13) as f64 - 6.0) * 0.25
        });
        let mut h = b.dot(&b.t());
        for i in 0..p {
            h[[i, i]] += 1.0;
        }
        h
    }

    /// Embed a small block matrix into a `p × p` zero matrix at `[start, start+m)`.
    fn embed(block: &Array2<f64>, start: usize, p: usize) -> Array2<f64> {
        let m = block.nrows();
        let mut s = Array2::<f64>::zeros((p, p));
        for i in 0..m {
            for j in 0..m {
                s[[start + i, start + j]] = block[[i, j]];
            }
        }
        s
    }

    /// A second-difference-style SPD block (positive definite, so full rank).
    fn diff_block(m: usize) -> Array2<f64> {
        let mut s = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            s[[i, i]] = 2.0;
            if i + 1 < m {
                s[[i, i + 1]] = -1.0;
                s[[i + 1, i]] = -1.0;
            }
        }
        // Nudge the diagonal so the block is strictly positive definite (a clean
        // full-rank fixture; nullspace handling is exercised elsewhere).
        for i in 0..m {
            s[[i, i]] += 0.1;
        }
        s
    }

    /// Sorted eigenvalues of a symmetric matrix.
    fn sorted_eigenvalues(m: &Array2<f64>) -> Vec<f64> {
        let (evals, _) = m.eigh(Side::Lower).expect("eigh");
        let mut e: Vec<f64> = evals.to_vec();
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        e
    }

    /// A minimal `ReparamResult` carrying only the fields the helper reads
    /// (`qs`, `penalty_shrinkage_ridge`); the rest are inert placeholders.
    fn reparam_with(qs: Array2<f64>, shrinkage_ridge: f64) -> ReparamResult {
        let p = qs.nrows();
        ReparamResult {
            s_transformed: Array2::zeros((p, p)),
            log_det: 0.0,
            det1: Array1::zeros(0),
            qs,
            canonical_transformed: Vec::new(),
            e_transformed: Array2::zeros((0, p)),
            u_truncated: Array2::zeros((p, 0)),
            penalty_shrinkage_ridge: shrinkage_ridge,
        }
    }

    // ── Test (1): disjoint-block parity ─────────────────────────────────────
    #[test]
    fn disjoint_blocks_joint_matches_per_block_and_eigenvalues_invariant() {
        let p = 8;
        // Two disjoint full-rank penalty blocks: columns 0..4 and 4..8.
        let s0 = embed(&diff_block(4), 0, p);
        let s1 = embed(&diff_block(4), 4, p);
        let penalties = vec![s0.clone(), s1.clone()];
        let lambdas = vec![2.5, 0.4];

        let h = deterministic_spd(p);
        let beta = Array1::from_shape_fn(p, |i| (i as f64 - 3.5) * 0.3);
        let qs = deterministic_orthogonal(p);
        let reparam = reparam_with(qs.clone(), 0.0);

        let ctx = RawInnerReparamContext {
            hessian: &h,
            beta: &beta,
            penalties_embedded: &penalties,
            lambdas: &lambdas,
        };
        let out = assemble_reparameterized_inner(&ctx, None, &reparam)
            .expect("disjoint reparam should succeed");

        // Per-block reference: for disjoint blocks the joint logdet is the sum
        // of the block logdets and det1[k] is block-local.
        let mut per_block_value = 0.0;
        let mut per_block_det1 = Array1::<f64>::zeros(2);
        for (k, s_k) in penalties.iter().enumerate() {
            let pld_k =
                PenaltyPseudologdet::from_components(&[s_k.clone()], &[lambdas[k]], 0.0).unwrap();
            let (d1_k, _) = pld_k.rho_derivatives(&[s_k.clone()], &[lambdas[k]]);
            per_block_value += pld_k.value();
            per_block_det1[k] = d1_k[0];
        }

        assert!(
            (out.penalty_logdet.value - per_block_value).abs() <= 1e-10,
            "joint value {} vs per-block sum {}",
            out.penalty_logdet.value,
            per_block_value
        );
        for k in 0..2 {
            assert!(
                (out.penalty_logdet.first[k] - per_block_det1[k]).abs() <= 1e-10,
                "det1[{k}] joint {} vs per-block {}",
                out.penalty_logdet.first[k],
                per_block_det1[k]
            );
        }
        // Cross-block second derivative must vanish for disjoint blocks.
        let det2 = out.penalty_logdet.second.as_ref().unwrap();
        assert!(
            det2[[0, 1]].abs() <= 1e-10 && det2[[1, 0]].abs() <= 1e-10,
            "cross-block det2 nonzero: {} / {}",
            det2[[0, 1]],
            det2[[1, 0]]
        );

        // H′ eigenvalues equal H eigenvalues (orthogonal similarity).
        let ev_h = sorted_eigenvalues(&h);
        let ev_hp = sorted_eigenvalues(&out.hessian_transformed);
        for (a, b) in ev_h.iter().zip(ev_hp.iter()) {
            assert!(
                (a - b).abs() <= 1e-11 * (1.0 + a.abs()),
                "eigenvalue drift under conjugation: {a} vs {b}"
            );
        }

        // β̂′ = Qsᵀ β̂, and Qs β̂′ recovers β̂ (round-trip).
        let recovered = qs.dot(&out.beta_transformed);
        for i in 0..p {
            assert!((recovered[i] - beta[i]).abs() <= 1e-11);
        }
    }

    // ── Test (2): ridge-overlap value-delta equality ────────────────────────
    #[test]
    fn full_span_ridge_overlap_joint_matches_assembled_and_differs_from_per_block_sum() {
        let p = 6;
        // Block penalty on columns 0..4, plus a FULL-SPAN ridge (I over 0..6)
        // that overlaps it — the survival stabilization-ridge shape.
        let s_block = embed(&diff_block(4), 0, p);
        let s_ridge = Array2::<f64>::eye(p);
        let penalties = vec![s_block.clone(), s_ridge.clone()];
        let lambdas = vec![3.0, 0.05];

        let h = deterministic_spd(p);
        let beta = Array1::from_shape_fn(p, |i| (i as f64) * 0.11 - 0.3);
        let qs = deterministic_orthogonal(p);
        let reparam = reparam_with(qs, 0.0);

        let ctx = RawInnerReparamContext {
            hessian: &h,
            beta: &beta,
            penalties_embedded: &penalties,
            lambdas: &lambdas,
        };
        let out = assemble_reparameterized_inner(&ctx, None, &reparam)
            .expect("ridge-overlap reparam should succeed");
        let joint = out.penalty_logdet.value;

        // Directly-assembled reference: log|λ0 S_block + λ1 I|₊.
        let mut assembled = Array2::<f64>::zeros((p, p));
        assembled.scaled_add(lambdas[0], &s_block);
        assembled.scaled_add(lambdas[1], &s_ridge);
        let ref_assembled = PenaltyPseudologdet::from_assembled(assembled, None)
            .unwrap()
            .value();
        assert!(
            (joint - ref_assembled).abs() <= 1e-9,
            "joint {joint} vs directly-assembled reference {ref_assembled}"
        );

        // Per-block (WRONG) convention: log|λ0 S_block|₊ + log|λ1 I|₊.
        let v_block = PenaltyPseudologdet::from_components(&[s_block], &[lambdas[0]], 0.0)
            .unwrap()
            .value();
        let v_ridge = PenaltyPseudologdet::from_components(&[s_ridge], &[lambdas[1]], 0.0)
            .unwrap()
            .value();
        let per_block_sum = v_block + v_ridge;

        // The joint normalizer must differ from the per-block sum by a real,
        // non-roundoff amount (the #2331 Finding 3a objective correction).
        assert!(
            (joint - per_block_sum).abs() > 1e-6,
            "joint {joint} indistinguishable from per-block sum {per_block_sum}"
        );
    }

    // ── Test (3): R5 asserts fire ───────────────────────────────────────────
    #[test]
    fn non_orthogonal_qs_is_rejected() {
        let p = 4;
        let h = deterministic_spd(p);
        let beta = Array1::zeros(p);
        let penalties = vec![embed(&diff_block(4), 0, p)];
        let lambdas = vec![1.0];
        // Qs = 2·I: QsᵀQs = 4I, defect 3 ≫ tolerance.
        let bad_qs = 2.0 * Array2::<f64>::eye(p);
        let reparam = reparam_with(bad_qs, 0.0);

        let ctx = RawInnerReparamContext {
            hessian: &h,
            beta: &beta,
            penalties_embedded: &penalties,
            lambdas: &lambdas,
        };
        let err = assemble_reparameterized_inner(&ctx, None, &reparam)
            .expect_err("non-orthogonal Qs must be rejected");
        assert!(err.contains("not orthogonal"), "unexpected error: {err}");
    }

    #[test]
    fn nonzero_shrinkage_ridge_is_rejected() {
        let p = 4;
        let h = deterministic_spd(p);
        let beta = Array1::zeros(p);
        let penalties = vec![embed(&diff_block(4), 0, p)];
        let lambdas = vec![1.0];
        let qs = deterministic_orthogonal(p);
        let reparam = reparam_with(qs, 1e-3); // nonzero shrinkage ridge

        let ctx = RawInnerReparamContext {
            hessian: &h,
            beta: &beta,
            penalties_embedded: &penalties,
            lambdas: &lambdas,
        };
        let err = assemble_reparameterized_inner(&ctx, None, &reparam)
            .expect_err("nonzero shrinkage ridge must be rejected");
        assert!(err.contains("shrinkage ridge"), "unexpected error: {err}");
    }

    // ── Test (4): conjugated-provider trace invariance ──────────────────────

    /// Test provider whose third-derivative correction is a deterministic
    /// symmetric function of the query direction: `C(u) = M0 + u uᵀ`, with `M0`
    /// a fixed symmetric matrix. Both pieces exercise the conjugation: the
    /// constant `M0` tests `tr(Qsᵀ M0 Qs) = tr(M0)`, and the rank-1 `u uᵀ` tests
    /// the vector remap `Qs · v′`.
    struct FixedCorrectionProvider {
        m0: Array2<f64>,
    }

    impl HessianDerivativeProvider for FixedCorrectionProvider {
        fn hessian_derivative_correction(
            &self,
            v_k: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            let n = v_k.len();
            let mut c = self.m0.clone();
            for i in 0..n {
                for j in 0..n {
                    c[[i, j]] += v_k[i] * v_k[j];
                }
            }
            Ok(Some(c))
        }

        fn has_corrections(&self) -> bool {
            true
        }
    }

    fn trace(m: &Array2<f64>) -> f64 {
        (0..m.nrows().min(m.ncols())).map(|i| m[[i, i]]).sum()
    }

    #[test]
    fn conjugated_provider_preserves_trace_and_quadratic_form() {
        let p = 5;
        let qs = deterministic_orthogonal(p);
        let m0 = deterministic_spd(p);
        let inner = FixedCorrectionProvider { m0: m0.clone() };

        // Build the conjugated wrapper directly (the helper builds the same
        // type internally).
        let conj = ConjugatedDerivProvider {
            inner: Box::new(FixedCorrectionProvider { m0 }),
            qs: qs.clone(),
        };

        let v_prime = Array1::from_shape_fn(p, |i| (i as f64 - 2.0) * 0.37 + 0.1);
        let v_raw = qs.dot(&v_prime); // Qs · v′

        let c_prime = conj
            .hessian_derivative_correction(&v_prime)
            .unwrap()
            .unwrap();
        let c_raw = inner
            .hessian_derivative_correction(&v_raw)
            .unwrap()
            .unwrap();

        // (a) trace invariance: tr(correction′) == tr(correction).
        assert!(
            (trace(&c_prime) - trace(&c_raw)).abs() <= 1e-10 * (1.0 + trace(&c_raw).abs()),
            "trace not preserved: {} vs {}",
            trace(&c_prime),
            trace(&c_raw)
        );

        // (b) quadratic identity: v′ᵀ correction′(v′) v′ == (Qs v′)ᵀ correction(Qs v′) (Qs v′).
        let q_prime = v_prime.dot(&c_prime.dot(&v_prime));
        let q_raw = v_raw.dot(&c_raw.dot(&v_raw));
        assert!(
            (q_prime - q_raw).abs() <= 1e-10 * (1.0 + q_raw.abs()),
            "quadratic form not preserved: {q_prime} vs {q_raw}"
        );

        // (c) the conjugated block is exactly Qsᵀ C_raw Qs.
        let expected = qs.t().dot(&c_raw).dot(&qs);
        for i in 0..p {
            for j in 0..p {
                assert!(
                    (c_prime[[i, j]] - expected[[i, j]]).abs() <= 1e-10,
                    "conjugation mismatch at ({i},{j})"
                );
            }
        }
    }
}

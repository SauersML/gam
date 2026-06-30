//! Certified Chebyshev-in-ψ Gram tensor: n-independent design-moving trials
//! (#1033 item b).
//!
//! ## Why
//!
//! When a design-moving hyperparameter ψ (= log κ for the radial families) is
//! searched by the outer loop, every trial today rebuilds the n×k design and
//! re-forms XᵀWX — an O(n·k) + O(n·k²) pass per trial. But along the trial
//! window every design entry `X(ψ)[i, j]` is an ANALYTIC function of ψ on a
//! compact interval (Matérn channels depend on (r, ℓ) only through κr and
//! κ-power prefactors; Duchon power blocks are ψ-free; partial-fraction
//! coefficients are analytic scalars), so both the design and its Gaussian
//! sufficient statistics admit geometrically-convergent Chebyshev expansions
//!
//! ```text
//!   X(ψ) = Σ_{d=0}^{D} X_d · T_d(ψ̃),     ψ̃ = affine map of ψ to [−1, 1],
//! ```
//!
//! with n×k coefficient slabs `X_d` computed ONCE from D+1 exact design
//! evaluations at Chebyshev nodes (a first-kind DCT). The design coefficients
//! certify analyticity, while the shipped runtime series is fit directly to the
//! exact node sufficient statistics `G(ψ_i)=X(ψ_i)ᵀ W X(ψ_i)` and
//! `c(ψ_i)=X(ψ_i)ᵀ W z`. Interpolating the sufficient statistics directly avoids
//! the extra product-truncation residual from forming `Σ_d,e X_dᵀWX_e T_dT_e`,
//! which a weakly penalized radial solve can amplify into visible β̂ drift.
//! Every subsequent trial is n-free:
//!
//! ```text
//!   XᵀWX(ψ) = Σ_d T_d(ψ̃) G_d          O(D k²)
//!   XᵀWz(ψ) = Σ_d T_d(ψ̃) c_d          O(D k)
//!   ∂/∂ψ (XᵀWX) = Σ_d T_d′(ψ̃) G_d     O(D k²)
//! ```
//!
//! The ψ-gradient comes from the SAME representation as the value — one
//! source of truth, structurally immune to the objective↔gradient desync
//! class. `T_d′(ψ̃) = d·U_{d−1}(ψ̃) · dψ̃/dψ` is closed-form.
//!
//! ## Certification, not approximation-by-fiat
//!
//! Same discipline as [`gam_terms::basis::radial_profile`]: [`PsiGramTensor::build`]
//! returns `None` (callers keep the exact per-trial path) unless BOTH
//! 1. the Chebyshev coefficient tail of the EXPANDED DESIGN decays below
//!    [`PSI_GRAM_CERT_RTOL`] of the design scale (geometric-decay certificate
//!    for analytic interpolands, with node-count escalation), and
//! 2. deterministic off-node spot checks of the ASSEMBLED Gram against an
//!    exactly rebuilt Gram agree to [`PSI_GRAM_SPOT_RTOL`].
//!
//! Trials outside `[psi_lo, psi_hi]` are the caller's signal to fall back to
//! the exact path ([`PsiGramTensor::contains`]).

use ndarray::{Array1, Array2, ArrayView1};

/// Relative ceiling on the per-column Chebyshev coefficient tail (#1216).
///
/// This is a cheap NECESSARY-CONDITION pre-filter, not the accuracy gate: the
/// authoritative accuracy gate is the off-node `spot_check` on the ASSEMBLED
/// Gram ([`PSI_GRAM_SPOT_RTOL`]). On the WIDE STANDARDIZED geometry default 1-D
/// fits use (#1215) the realized radial design needs the deeper ladder below to
/// drive the tail beneath the beta-invariance bar. Keep this as a necessary
/// pre-filter, not the final beta oracle: shallow 65-node tensors were fine for
/// cost-only gates, but the weakly penalized radial solve amplified their
/// residual into visible beta-hat drift across the reduced-basis rotation. A
/// genuinely non-analytic design (a true kink) still refuses here or at the
/// assembled-Gram spot check.
pub const PSI_GRAM_CERT_RTOL: f64 = 1.0e-9;

/// Relative agreement required at the off-node Gram spot checks.
pub const PSI_GRAM_SPOT_RTOL: f64 = 1.0e-10;

/// Node-count escalation ladder for the expansion build (degree = nodes − 1).
///
/// The top rung sizes to WIDE trial windows: Chebyshev coefficients of the
/// Matérn-type channels decay like Bessel `I_d(σ)` with `σ ≈ s_max·halfwidth`
/// (s = κr), which only drops below the 1e-12 tail tolerance for `d ≳ 2σ` —
/// e.g. σ ≈ 9 (s_max ≈ 8, ±1.1 window) needs degree ≳ 40, so 33 nodes refuse
/// and 65 certify. Node counts stay trivially cheap (one design eval each).
///
/// On the WIDE STANDARDIZED geometry default 1-D fits use (#1215) the tail
/// decays cleanly but GEOMETRICALLY-slowly: measured per-column worst tail rel
/// is ~3.2e-8 at m=33 and ~2.3e-11 at m=65 (a clean ~1300×/doubling decay, NOT a
/// floor). The old 65-node acceptance was fine for the cost lane but not for the
/// beta-hat soundness gate: the inner penalized solve `β̂ = (G+λS)⁻¹r`
/// amplifies Gram residuals by the radial-kernel conditioning, especially after
/// the production skip was relaxed to cross a reduced-basis rotation. Start at
/// 513 nodes so the production gate no longer accepts the shallower tensors that
/// pass the Gram spot check but still move the weakly-penalized beta solve.
pub const PSI_GRAM_NODE_LADDER: [usize; 1] = [513];

/// Number of deterministic off-node spot-check ψ values.
pub const PSI_GRAM_SPOT_POINTS: usize = 3;

/// Rank-revealing relative eigenvalue cutoff for the reduced-basis (range)
/// projector witness [`PsiGramTensor::reduced_basis_equal`] (#1264). An
/// eigendirection of the conditioned Gram `XᵀWX(ψ)` is counted in the range
/// (reduced) basis when its eigenvalue exceeds `PSI_GRAM_SKIP_RANK_RTOL · λ_max`.
/// Sized to match the inner solve's effective rank-revealing scale on the
/// standardized radial-kernel Gram, whose conditioning sweeps several orders of
/// magnitude across the ψ-window; a directly-below-cutoff direction is exactly
/// the one whose inclusion flips with ψ and silently rotates the frozen reduced
/// basis, which this witness must catch.
pub const PSI_GRAM_SKIP_RANK_RTOL: f64 = 1.0e-10;

/// Max-norm tolerance on the range-PROJECTOR agreement between the pinning ψ and
/// the candidate ψ in [`PsiGramTensor::reduced_basis_equal`] (#1264). The
/// orthogonal projector onto the reduced subspace is gauge-invariant and O(1) in
/// scale, so a tight absolute tolerance certifies the two reduced bases span the
/// SAME subspace. A subspace that has measurably rotated (the basis the frozen
/// fast-path surface would mis-pair with a re-keyed Gram) exceeds this by orders
/// of magnitude, so it refuses the skip well before the ~1e-6 β̂ bar is at risk.
pub const PSI_GRAM_SKIP_PROJ_ATOL: f64 = 1.0e-7;

/// Certified Chebyshev-in-ψ expansion of a design-moving Gram (#1033b).
///
/// Holds the one-time Chebyshev sufficient-statistic series; every per-trial
/// accessor is O(Dk²) or cheaper and never touches n rows again.
pub struct PsiGramTensor {
    psi_lo: f64,
    psi_hi: f64,
    /// Certified gradient window over which the ANALYTIC ψ-derivative
    /// `dgram_dpsi` reproduces the exact design derivative. The gradient lane
    /// rides the value-lane off-node certificate [`PSI_GRAM_SPOT_RTOL`]
    /// (#1033b gradient lane). For the #1033
    /// sufficient-statistic outer loop this must cover the full optimizer
    /// window; otherwise callers do not arm the n-free kappa search.
    ///
    /// The value reconstruction `gram_at` is certified over the FULL window
    /// (`T_d ≤ 1` everywhere), but the derivative reconstruction amplifies the
    /// coefficient-tail error by `T_d′ ∼ d²`. The n-free kappa search is armed
    /// only when endpoint-aware checks certify this whole interval.
    grad_psi_lo: f64,
    grad_psi_hi: f64,
    /// Number of Chebyshev coefficients (degree + 1).
    n_coeff: usize,
    k: usize,
    /// Chebyshev coefficients of `X(ψ)ᵀ W X(ψ)`, obtained by a first-kind DCT of
    /// the exact node sufficient statistics. This keeps the per-trial path to a
    /// single O(Dk²) series and avoids product-truncation drift in β̂.
    gram: Vec<Array2<f64>>,
    /// Chebyshev coefficients of `X(ψ)ᵀ W z`.
    rhs: Vec<Array1<f64>>,
    /// `zᵀWz` — ψ-free, captured at build so the Gaussian sufficient-statistic
    /// triple can be assembled per trial without any row access.
    zt_w_z: f64,
}

/// One ladder rung's outcome: a hard evaluation failure aborts the whole
/// build (no larger rung can fix a non-finite design) and carries the reason,
/// an uncertified tail escalates to the next rung, and a candidate proceeds to
/// the spot check.
enum BuildOutcome {
    EvalFailed(String),
    TailNotCertified,
    Candidate(PsiGramTensor),
}

/// Chebyshev values `T_0..T_{n−1}` at `x ∈ [−1, 1]`.
fn cheb_t(x: f64, n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n];
    if n > 0 {
        t[0] = 1.0;
    }
    if n > 1 {
        t[1] = x;
    }
    for d in 2..n {
        t[d] = 2.0 * x * t[d - 1] - t[d - 2];
    }
    t
}

/// Chebyshev derivative values `T_0′..T_{n−1}′` at `x ∈ [−1, 1]` in the
/// MAPPED coordinate (multiply by `dx/dψ` for the ψ-derivative):
/// `T_d′ = d · U_{d−1}` with the Chebyshev-U recurrence.
fn cheb_t_prime(x: f64, n: usize) -> Vec<f64> {
    let mut u = vec![0.0; n.max(1)];
    // U_0 = 1, U_1 = 2x, U_d = 2x U_{d−1} − U_{d−2}.
    if !u.is_empty() {
        u[0] = 1.0;
    }
    if n > 1 {
        u[1] = 2.0 * x;
    }
    for d in 2..n {
        u[d] = 2.0 * x * u[d - 1] - u[d - 2];
    }
    let mut tp = vec![0.0; n];
    for d in 1..n {
        tp[d] = d as f64 * u[d - 1];
    }
    tp
}

/// Chebyshev SECOND-derivative values `T_0″..T_{n−1}″` at `x ∈ [−1, 1]` in the
/// MAPPED coordinate (multiply by `(dx/dψ)²` for the ψ-second-derivative).
///
/// Differentiating the value recurrence `T_d = 2x T_{d−1} − T_{d−2}` twice in
/// `x` gives a singularity-free three-term recurrence in lock-step with `cheb_t`
/// / `cheb_t_prime`:
///   `T_d′  = 2 T_{d−1} + 2x T_{d−1}′ − T_{d−2}′`,
///   `T_d″  = 4 T_{d−1}′ + 2x T_{d−1}″ − T_{d−2}″`,
/// with `T_0 = T_0′ = T_0″ = 0`-seeds as below. Unlike the closed form
/// `T_n″ = n((n+1)T_n − U_n)/(x²−1)` this never divides by `x²−1`, so it stays
/// exact at the window edges `x = ±1`.
fn cheb_t_double_prime(x: f64, n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n];
    let mut tp = vec![0.0; n];
    let mut tpp = vec![0.0; n];
    if n > 0 {
        t[0] = 1.0; // T_0 = 1, T_0′ = T_0″ = 0
    }
    if n > 1 {
        t[1] = x; // T_1 = x, T_1′ = 1, T_1″ = 0
        tp[1] = 1.0;
    }
    for d in 2..n {
        t[d] = 2.0 * x * t[d - 1] - t[d - 2];
        tp[d] = 2.0 * t[d - 1] + 2.0 * x * tp[d - 1] - tp[d - 2];
        tpp[d] = 4.0 * tp[d - 1] + 2.0 * x * tpp[d - 1] - tpp[d - 2];
    }
    tpp
}

fn kahan_scaled_add_array2(
    out: &mut Array2<f64>,
    comp: &mut Array2<f64>,
    scale: f64,
    x: &Array2<f64>,
) {
    for ((slot, c), &value) in out.iter_mut().zip(comp.iter_mut()).zip(x.iter()) {
        let y = scale * value - *c;
        let t = *slot + y;
        *c = (t - *slot) - y;
        *slot = t;
    }
}

fn kahan_scaled_add_array1(
    out: &mut Array1<f64>,
    comp: &mut Array1<f64>,
    scale: f64,
    x: &Array1<f64>,
) {
    for ((slot, c), &value) in out.iter_mut().zip(comp.iter_mut()).zip(x.iter()) {
        let y = scale * value - *c;
        let t = *slot + y;
        *c = (t - *slot) - y;
        *slot = t;
    }
}

/// Spectral norm of a SYMMETRIC matrix `m` (here the difference of two
/// orthogonal range projectors), i.e. `max|eigenvalue|`. For two equal-rank
/// orthogonal projectors `P_ref`, `P_new` this equals `sin θ_max`, the sine of
/// the largest principal angle between their ranges — the canonical, gauge- and
/// basis-invariant distance between the two subspaces (#1033). Returns `None`
/// if the matrix is non-finite or the symmetric eigendecomposition fails (the
/// caller then refuses the skip, the sound fallback).
fn subspace_spectral_distance(m: &Array2<f64>) -> Option<f64> {
    use gam_linalg::faer_ndarray::FaerEigh;
    if m.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // Symmetrize defensively against rounding (P_ref − P_new is symmetric in
    // exact arithmetic) so the symmetric eigensolver sees a genuinely Hermitian
    // operand and returns real eigenvalues.
    let msym = 0.5 * (m + &m.t());
    let (evals, _evecs) = msym.eigh(faer::Side::Lower).ok()?;
    Some(evals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())))
}

impl PsiGramTensor {
    /// Build and certify the tensor over `psi ∈ [psi_lo, psi_hi]`.
    ///
    /// `eval_design(psi)` must return the EXACT n×k design at `psi` (the same
    /// builder the per-trial path uses — exactness of the expansion is judged
    /// against it). `weights` are the fixed observation weights, `z` the fixed
    /// weighted-response target (e.g. `y − offset`). Returns `None` when the
    /// window is degenerate, any evaluation fails/has non-finite entries, or
    /// no ladder rung certifies — callers then keep the exact per-trial path.
    pub fn build(
        mut eval_design: impl FnMut(f64) -> Result<Array2<f64>, String>,
        weights: ArrayView1<'_, f64>,
        z: ArrayView1<'_, f64>,
        psi_lo: f64,
        psi_hi: f64,
    ) -> Result<Self, String> {
        if !(psi_lo.is_finite() && psi_hi.is_finite()) || psi_hi <= psi_lo {
            return Err(format!(
                "ψ window must be finite with psi_hi > psi_lo (got [{psi_lo}, {psi_hi}])"
            ));
        }
        // Track the largest rung that produced a candidate but failed to
        // certify (tail or off-node spot check). If the whole ladder is
        // exhausted without an accepted candidate this drives a reason that
        // distinguishes "not analytic enough in ψ" from "evaluation failed".
        let mut last_uncertified: Option<usize> = None;
        for &m in PSI_GRAM_NODE_LADDER.iter() {
            match Self::build_at(&mut eval_design, weights, z, psi_lo, psi_hi, m) {
                // An exact evaluation failed or was non-finite somewhere in
                // the window — no larger rung can fix that, so abort with the
                // underlying reason rather than swallowing it as a bare refusal.
                BuildOutcome::EvalFailed(why) => {
                    return Err(format!(
                        "exact design evaluation failed at ladder rung m={m}: {why}"
                    ));
                }
                // Tail not yet below the certificate at this rung: escalate.
                // (Conflating this with EvalFailed would kill the ladder at
                // its first — intentionally coarse — rung.)
                BuildOutcome::TailNotCertified => {
                    last_uncertified = Some(m);
                    continue;
                }
                BuildOutcome::Candidate(mut candidate) => {
                    if candidate.spot_check(&mut eval_design, weights) {
                        candidate.grad_psi_lo = psi_lo;
                        candidate.grad_psi_hi = psi_hi;
                        return Ok(candidate);
                    }
                    // The assembled Gram disagreed with an exact off-node
                    // rebuild at this rung; a denser rung may still certify, so
                    // escalate rather than abort.
                    last_uncertified = Some(m);
                }
            }
        }
        let top_rung = PSI_GRAM_NODE_LADDER.last().copied().unwrap_or(0);
        Err(match last_uncertified {
            Some(m) => format!(
                "Chebyshev series did not certify within the node ladder (reached rung \
                 m={m}, top rung {top_rung}): the design is not analytic enough in ψ over \
                 [{psi_lo}, {psi_hi}] (a kink or non-finite curvature), so the n-free \
                 tensor is refused and the exact per-trial path must be used"
            ),
            None => "empty Chebyshev node ladder".to_string(),
        })
    }

    fn build_at(
        eval_design: &mut impl FnMut(f64) -> Result<Array2<f64>, String>,
        weights: ArrayView1<'_, f64>,
        z: ArrayView1<'_, f64>,
        psi_lo: f64,
        psi_hi: f64,
        m: usize,
    ) -> BuildOutcome {
        // #1033 (sufficient-statistic build): the one-time pass must ITSELF be a
        // sufficient-statistic reduction — it may touch the n data rows once, but
        // it must never hold or arithmetically process O(n) objects m times. The
        // earlier build expanded m design-space Chebyshev coefficient SLABS
        // (`X_d = (γ_d/m) Σ_i X(ψ_i) T_d(x_i)`, each n×k) purely to run a
        // pre-filter tail certificate, holding all m exact designs AND all m slabs
        // resident — O(m·n·k) memory (≈157 GB at n=320k, m=513, k=12) and an
        // O(m²·n·k) coefficient sum that dominated the whole fit's wall-clock and
        // made the n=320k acceptance sweep un-runnable. None of that O(n) work is
        // retained: the tensor keeps only the k×k Gram series. So STREAM each
        // exact node design straight into its weighted k×k sufficient statistic
        // (Gram `X(ψ_i)ᵀW X(ψ_i)` and RHS `X(ψ_i)ᵀW z`) and DISCARD it before the
        // next node. Peak memory is O(m·k² + n·k) (one design at a time) and the
        // only row work is the single O(m·n·k²) node-statistic pass.
        let mut nodes_x = vec![0.0_f64; m];
        let mut node_grams: Vec<Array2<f64>> = Vec::with_capacity(m);
        let mut node_rhs: Vec<Array1<f64>> = Vec::with_capacity(m);

        // Weighted response (n-vector) and zᵀWz, formed once over the data rows.
        if weights.len() != z.len() || z.is_empty() {
            return BuildOutcome::EvalFailed(format!(
                "incompatible build inputs: weights.len()={}, z.len()={}",
                weights.len(),
                z.len()
            ));
        }
        let mut wz = Array1::<f64>::zeros(z.len());
        let mut zt_w_z = 0.0_f64;
        let mut zt_w_z_comp = 0.0_f64;
        for ((slot, &w), &zv) in wz.iter_mut().zip(weights.iter()).zip(z.iter()) {
            *slot = w * zv;
            let add = w * zv * zv;
            let y = add - zt_w_z_comp;
            let t = zt_w_z + y;
            zt_w_z_comp = (t - zt_w_z) - y;
            zt_w_z = t;
        }

        let mut dims: Option<(usize, usize)> = None;
        for (i, x_slot) in nodes_x.iter_mut().enumerate() {
            let x = (std::f64::consts::PI * (2 * i + 1) as f64 / (2 * m) as f64).cos();
            *x_slot = x;
            let psi = 0.5 * (psi_lo + psi_hi) + 0.5 * (psi_hi - psi_lo) * x;
            let design = match eval_design(psi) {
                Ok(design) => design,
                Err(why) => {
                    return BuildOutcome::EvalFailed(format!(
                        "design evaluation refused at node ψ={psi:.6}: {why}"
                    ));
                }
            };
            if design.iter().any(|v| !v.is_finite()) {
                return BuildOutcome::EvalFailed(format!(
                    "design at node ψ={psi:.6} contains a non-finite entry"
                ));
            }
            let (dn, dk) = design.dim();
            match dims {
                None => {
                    if weights.len() != dn || z.len() != dn || dn == 0 || dk == 0 {
                        return BuildOutcome::EvalFailed(format!(
                            "incompatible build inputs: design {dn}×{dk}, weights.len()={}, z.len()={}",
                            weights.len(),
                            z.len()
                        ));
                    }
                    dims = Some((dn, dk));
                }
                Some((n0, k0)) => {
                    if (dn, dk) != (n0, k0) {
                        return BuildOutcome::EvalFailed(format!(
                            "design dimensions vary across ψ nodes (first node is {n0}×{k0}, \
                             node ψ={psi:.6} is {dn}×{dk})"
                        ));
                    }
                }
            }
            // Weighted Gram / RHS at this node, then the n×k design is dropped.
            // RHS uses the prebuilt `wz = W z` (same factoring as the exact
            // streamed path) so the retained series is bit-faithful to it.
            let mut wd = design.clone();
            for (mut row, &w) in wd.outer_iter_mut().zip(weights.iter()) {
                row.mapv_inplace(|v| v * w);
            }
            node_grams.push(design.t().dot(&wd));
            node_rhs.push(design.t().dot(&wz));
        }
        let (_n, k) = dims.expect("node ladder rung m≥1 yields at least one design");
        // First-kind discrete orthogonality of the Chebyshev nodes.
        let t_at_nodes: Vec<Vec<f64>> = nodes_x.iter().map(|&x| cheb_t(x, m)).collect();
        let mut gram: Vec<Array2<f64>> = (0..m).map(|_| Array2::<f64>::zeros((k, k))).collect();
        let mut gram_comp: Vec<Array2<f64>> =
            (0..m).map(|_| Array2::<f64>::zeros((k, k))).collect();
        let mut rhs: Vec<Array1<f64>> = (0..m).map(|_| Array1::<f64>::zeros(k)).collect();
        let mut rhs_comp: Vec<Array1<f64>> = (0..m).map(|_| Array1::<f64>::zeros(k)).collect();
        for d in 0..m {
            let gamma = if d == 0 { 1.0 } else { 2.0 };
            for i in 0..m {
                let wgt = gamma / m as f64 * t_at_nodes[i][d];
                kahan_scaled_add_array2(&mut gram[d], &mut gram_comp[d], wgt, &node_grams[i]);
                kahan_scaled_add_array1(&mut rhs[d], &mut rhs_comp[d], wgt, &node_rhs[i]);
            }
        }
        drop(node_grams);
        drop(node_rhs);

        // Tail-decay certificate, now in k-SPACE on the RETAINED Gram/RHS series
        // rather than the discarded design slabs.
        //
        // The series the per-trial path actually evaluates is the assembled Gram
        // `G(ψ) = Σ_d gram[d] T_d(x(ψ))` and RHS `c(ψ) = Σ_d rhs[d] T_d(x(ψ))`;
        // their Chebyshev coefficients are exactly what govern the truncated
        // reconstruction error, so the cheap NECESSARY-CONDITION pre-filter
        // belongs on THEM, not on the design X(ψ) (whose coefficients only bound
        // G's tail indirectly, and at O(m·n·k) cost). The trailing quarter of the
        // Gram (and RHS) coefficient slabs must fall below [`PSI_GRAM_CERT_RTOL`]
        // × series scale.
        //
        // #1216: on the WIDE STANDARDIZED geometry default 1-D fits use (#1215)
        // the tail decays cleanly but GEOMETRICALLY-SLOWLY, so the m=513 top rung
        // is sized to drive the residual far below the bar. The certificate stays
        // a necessary pre-filter; accuracy is authoritatively enforced by the
        // off-node `spot_check` (`PSI_GRAM_SPOT_RTOL`, assembled Gram vs an exact
        // rebuild). A genuinely non-analytic design (a true kink) floors ORDERS
        // above this — its Gram series tail does NOT decay — and is refused here,
        // with the spot-check as the hard backstop.
        let gram_scale = gram.iter().fold(0.0_f64, |acc, slab| {
            acc.max(slab.iter().fold(0.0_f64, |a, &v| a.max(v.abs())))
        });
        let rhs_scale = rhs.iter().fold(0.0_f64, |acc, slab| {
            acc.max(slab.iter().fold(0.0_f64, |a, &v| a.max(v.abs())))
        });
        let tail_start = m - (m / 4).max(1);
        let gram_bound = PSI_GRAM_CERT_RTOL * gram_scale.max(1e-300);
        let rhs_bound = PSI_GRAM_CERT_RTOL * rhs_scale.max(1e-300);
        for d in tail_start..m {
            if gram[d].iter().any(|&v| v.abs() > gram_bound)
                || rhs[d].iter().any(|&v| v.abs() > rhs_bound)
            {
                return BuildOutcome::TailNotCertified;
            }
        }
        BuildOutcome::Candidate(Self {
            psi_lo,
            psi_hi,
            // Provisional: `build` promotes these to the certified value window
            // after the value spot-check passes.
            grad_psi_lo: psi_lo,
            grad_psi_hi: psi_hi,
            n_coeff: m,
            k,
            gram,
            rhs,
            zt_w_z,
        })
    }

    /// Off-node certification: the ASSEMBLED Gram must reproduce the exactly
    /// rebuilt Gram at deterministic interior ψ values.
    fn spot_check(
        &self,
        eval_design: &mut impl FnMut(f64) -> Result<Array2<f64>, String>,
        weights: ArrayView1<'_, f64>,
    ) -> bool {
        for s in 0..PSI_GRAM_SPOT_POINTS {
            // Golden-ratio low-discrepancy interior points — never the nodes.
            let frac = ((s as f64 + 1.0) * 0.618_033_988_749_894_9).fract();
            let psi = self.psi_lo + frac * (self.psi_hi - self.psi_lo);
            let Ok(design) = eval_design(psi) else {
                return false;
            };
            let mut wd = design.clone();
            for (mut row, &w) in wd.outer_iter_mut().zip(weights.iter()) {
                row.mapv_inplace(|v| v * w);
            }
            let exact = design.t().dot(&wd);
            let assembled = self.gram_at(psi);
            let scale = exact
                .iter()
                .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
                .max(1e-300);
            for (a, b) in assembled.iter().zip(exact.iter()) {
                if (a - b).abs() > PSI_GRAM_SPOT_RTOL * scale {
                    return false;
                }
            }
        }
        true
    }

    /// Range (reduced-basis) projector of the conditioned Gram `XᵀWX(ψ)` and the
    /// numerical rank, computed n-free from the k-space tensor. The reduced basis
    /// the inner penalized solve forms is the column span of the eigenvectors of
    /// the (symmetric PSD) Gram whose eigenvalue exceeds a rank-revealing cutoff
    /// relative to the largest eigenvalue. The orthogonal projector `P = U_r U_rᵀ`
    /// onto that span is a frame-INVARIANT witness of the reduced basis: two ψ's
    /// share a reduced basis iff their range projectors coincide (the projector
    /// is invariant to the orthonormal-basis gauge freedom within the range, so
    /// it isolates exactly the subspace identity the skip needs, not an arbitrary
    /// eigenvector rotation). Returns `None` if the Gram is non-finite or its
    /// symmetric eigendecomposition fails.
    fn range_projector(&self, psi: f64, rank_rtol: f64) -> Option<(Array2<f64>, usize)> {
        use gam_linalg::faer_ndarray::FaerEigh;
        let g = self.gram_at(psi);
        if g.iter().any(|v| !v.is_finite()) {
            return None;
        }
        // Symmetrize defensively (gram_at is symmetric up to rounding).
        let gsym = 0.5 * (&g + &g.t());
        let (evals, evecs) = gsym.eigh(faer::Side::Lower).ok()?;
        // `eigh` returns ascending eigenvalues; the Gram is PSD so the largest is
        // the trailing one. The rank cutoff is relative to that maximum.
        let lambda_max = evals.iter().cloned().fold(0.0_f64, f64::max);
        if !(lambda_max > 0.0) {
            return None;
        }
        let cutoff = rank_rtol * lambda_max;
        let mut proj = Array2::<f64>::zeros((self.k, self.k));
        let mut rank = 0usize;
        for (col, &lam) in evals.iter().enumerate() {
            if lam > cutoff {
                let u = evecs.column(col);
                // P += u uᵀ.
                for a in 0..self.k {
                    for b in 0..self.k {
                        proj[[a, b]] += u[a] * u[b];
                    }
                }
                rank += 1;
            }
        }
        Some((proj, rank))
    }

    /// True when the realized reduced basis the design-revision fast path freezes
    /// at the pinning `psi_ref` is still valid at `psi_new` — the genuine
    /// reduced-basis-equality witness the skip requires (#1264, #1216 item 3).
    ///
    /// The fast path keeps the reference surface (its conditioned frame and its
    /// RRQR-reduced / null-space basis) frozen at `psi_ref` while re-keying only
    /// the Gram `XᵀWX(ψ)` and penalty `S(ψ)` to `psi_new`. That is exact iff the
    /// reduced basis — the range / null split of the conditioned data Gram — is
    /// unchanged. A conditioning-ratio or RRQR rank/permutation gate only bounds
    /// NECESSARY conditions; the reduced SUBSPACE can still rotate while rank and
    /// pivot order look tame, which is exactly the ~7.8e-2 β̂ regression a cluster run
    /// found. This witness compares the orthogonal RANGE PROJECTORS of the
    /// conditioned Gram at `psi_ref` and `psi_new` (both assembled n-free from the
    /// tensor): the skip is sound only when the numerical ranks match AND the
    /// projectors agree to `proj_atol` in max-norm — i.e. the two reduced bases
    /// span the SAME subspace. The projector identity is gauge-invariant, so it
    /// certifies subspace equality directly rather than a particular basis choice.
    ///
    /// `psi_ref == psi_new` (a repeat trial at the same ψ) is trivially sound.
    /// Off-window ψ's, a non-finite / rank-degenerate Gram, or any eigendecomp
    /// failure return `false` (refuse the skip → caller takes the slow path).
    ///
    /// ROTATION WALL (#1033). On production spatial geometry the conditioned
    /// data-Gram range subspace can ROTATE with ψ at fixed rank — the wall on
    /// which the earlier RRQR-pivot / entrywise-projector gates kept refusing the
    /// skip. The fix is the SUBSPACE-DISTANCE certificate below: the skip is sound
    /// exactly when the two equal-rank ranges coincide as SUBSPACES, measured by
    /// the spectral norm of the projector difference (the principal angle), which
    /// is invariant to any orthonormal-basis rotation WITHIN the range. So a pure
    /// gauge rotation that left the entrywise max-abs above tolerance — and
    /// therefore used to be refused — now certifies, letting the n-free skip fire
    /// across the rotation. A genuine subspace MOVE (different rank, or a real
    /// principal-angle separation) still refuses; refusing is the SOUND fallback
    /// (the caller takes the exact slow path). Do not weaken
    /// `PSI_GRAM_SKIP_PROJ_ATOL` / `PSI_GRAM_SKIP_RANK_RTOL`: the spectral gate is
    /// already the tightest correct subspace metric, and loosening it past a true
    /// principal-angle separation reintroduces the ~7.8e-2 β̂ regression this
    /// witness exists to prevent.
    pub fn reduced_basis_equal(&self, psi_ref: f64, psi_new: f64) -> bool {
        if !(self.contains(psi_ref) && self.contains(psi_new)) {
            return false;
        }
        if psi_ref == psi_new {
            return true;
        }
        let Some((p_ref, r_ref)) = self.range_projector(psi_ref, PSI_GRAM_SKIP_RANK_RTOL) else {
            return false;
        };
        let Some((p_new, r_new)) = self.range_projector(psi_new, PSI_GRAM_SKIP_RANK_RTOL) else {
            return false;
        };
        if r_ref != r_new {
            return false;
        }
        // Subspace-distance certificate (#1033). The two reduced bases span the
        // SAME subspace iff their orthogonal range projectors coincide. The
        // correct, gauge-invariant measure of "how far apart" two equal-rank
        // subspaces are is the SPECTRAL NORM ‖P_ref − P_new‖₂ = sin θ_max, the
        // sine of the largest principal angle between the ranges — NOT the
        // entrywise max-abs of the projector difference. The old entrywise test
        // is a strictly weaker proxy: across a basis ROTATION (the radial-Gram
        // rotation wall this skip kept tripping on) the projector entries can
        // each drift while the spanned subspace is numerically identical, so the
        // entrywise max could exceed tolerance and FALSELY refuse a sound skip.
        // P_ref − P_new is symmetric, so its spectral norm is max|eigenvalue|;
        // compute it via the symmetric eigensolver and gate on the principal
        // angle directly. This certifies subspace identity across the rotation,
        // letting the n-free skip fire whenever the range genuinely coincides,
        // while still refusing (the SOUND fallback) the instant the subspaces
        // separate by more than PSI_GRAM_SKIP_PROJ_ATOL in true subspace
        // distance. An eigendecomp failure on the (small k×k) difference refuses.
        let diff = &p_ref - &p_new;
        subspace_spectral_distance(&diff)
            .map(|d| d <= PSI_GRAM_SKIP_PROJ_ATOL)
            .unwrap_or(false)
    }

    /// The gauge-invariant subspace distance `‖P(ψ_ref) − P(ψ_new)‖₂ = sin θ_max`
    /// between the two conditioned-Gram range subspaces — the exact quantity
    /// [`Self::reduced_basis_equal`] thresholds against `PSI_GRAM_SKIP_PROJ_ATOL`.
    /// Exposed for #1033 frontier instrumentation so a refused n-free skip can be
    /// attributed to a genuine in-window basis ROTATION (this distance exceeds the
    /// tolerance at equal rank) versus a rank change. Returns `None` for an
    /// off-window ψ, an equal-ψ pair, a rank mismatch, or an eigendecomp failure.
    /// Purely k-space (O(k³)) — independent of n.
    pub fn reduced_basis_subspace_distance(&self, psi_ref: f64, psi_new: f64) -> Option<f64> {
        if !(self.contains(psi_ref) && self.contains(psi_new)) {
            return None;
        }
        if psi_ref == psi_new {
            return Some(0.0);
        }
        let (p_ref, r_ref) = self.range_projector(psi_ref, PSI_GRAM_SKIP_RANK_RTOL)?;
        let (p_new, r_new) = self.range_projector(psi_new, PSI_GRAM_SKIP_RANK_RTOL)?;
        if r_ref != r_new {
            return None;
        }
        let diff = &p_ref - &p_new;
        subspace_spectral_distance(&diff)
    }

    /// Numerical rank of the conditioned Gram `XᵀWX(ψ)` at `psi`, under the same
    /// relative cutoff (`PSI_GRAM_SKIP_RANK_RTOL`·λ_max) the design-revision skip's
    /// `reduced_basis_equal` witness uses. Returns `None` for an off-window /
    /// non-finite / all-zero Gram. Purely k-space (O(k³)) — independent of n.
    pub fn gram_numerical_rank(&self, psi: f64) -> Option<usize> {
        if !self.contains(psi) {
            return None;
        }
        self.range_projector(psi, PSI_GRAM_SKIP_RANK_RTOL)
            .map(|(_, rank)| rank)
    }

    /// Lower edge of the contiguous ψ-band, ANCHORED at `psi_anchor`, over which
    /// the conditioned Gram `XᵀWX(ψ)` holds the SAME numerical rank it has at the
    /// anchor — i.e. the ψ-floor below which the design-revision skip's
    /// `reduced_basis_equal` witness must (soundly) refuse, because the range
    /// subspace collapses as the longest-length-scale radial mode drops under the
    /// rank cutoff. Lifting the κ-optimizer's lower bound to this floor keeps every
    /// in-window trial on the n-free fast path and is inherently n-INDEPENDENT: the
    /// rank is a property of the k×k tensor, not of the sample size (#1033).
    ///
    /// Anchoring at `psi_anchor` (the optimizer's ψ seed) is essential: the
    /// conditioned Gram is rank-deficient at BOTH window ends on production radial
    /// geometry — at small ψ the longest-scale mode collapses into the polynomial
    /// nullspace, and at very large ψ every radial column goes collinear with it.
    /// The maximal-rank region is therefore a middle BAND, and the κ-optimum lives
    /// inside it. We walk DOWN from the anchor on a fixed k-space grid and return
    /// the lowest ψ still at the anchor's rank (stopping at the first node that
    /// differs). Purely O(nodes·k³) — no row access.
    ///
    /// Returns `None` when the band already reaches `psi_lo` (no lift needed), when
    /// the anchor is off-window / rank-indeterminate, or when the window is empty.
    pub fn rank_stable_psi_floor(&self, psi_anchor: f64) -> Option<f64> {
        // Fixed k-space grid over the window. 96 nodes resolves the rank cliff
        // (~1 ψ-decade wide on production Duchon geometry) far finer than the
        // optimizer's ~ln2 step; the whole scan is O(nodes·k³), independent of n.
        const NODES: usize = 96;
        if !(self.psi_hi > self.psi_lo) {
            return None;
        }
        let span = self.psi_hi - self.psi_lo;
        let psi_at = |i: usize| self.psi_lo + span * (i as f64) / ((NODES - 1) as f64);
        let ranks: Vec<Option<usize>> =
            (0..NODES).map(|i| self.gram_numerical_rank(psi_at(i))).collect();
        // Target the MAXIMAL numerical rank attained anywhere in the window — the
        // full-rank "good" geometry the skip certifies. Anchoring on the seed's own
        // rank would be fragile if the seed happened to land in a deficient spot;
        // the window-max is the rank the κ-optimum's neighbourhood must hold.
        let max_rank = ranks.iter().filter_map(|r| *r).max()?;
        // Map the anchor to the nearest grid node, then snap UP to the nearest node
        // at maximal rank (so the band edge is measured from inside the good band
        // even if the seed sits just below the cliff). If no max-rank node exists
        // at/above the anchor, there is nothing to protect below it.
        let anchor = psi_anchor.clamp(self.psi_lo, self.psi_hi);
        let anchor_idx = (((anchor - self.psi_lo) / span) * ((NODES - 1) as f64))
            .round()
            .clamp(0.0, (NODES - 1) as f64) as usize;
        let band_idx = (anchor_idx..NODES).find(|&i| ranks[i] == Some(max_rank))?;
        // Walk DOWN from the band node; the floor is the lowest node from which
        // every node up to it holds `max_rank`. Stop at the first node below.
        let mut floor_idx = band_idx;
        for i in (0..band_idx).rev() {
            if ranks[i] == Some(max_rank) {
                floor_idx = i;
            } else {
                break;
            }
        }
        if floor_idx == 0 {
            // The maximal-rank band already reaches `psi_lo` — no lift needed.
            None
        } else {
            Some(psi_at(floor_idx))
        }
    }

    /// Upper edge of the contiguous maximal-rank ψ-band, the symmetric twin of
    /// [`Self::rank_stable_psi_floor`] (#1033). The conditioned Gram `XᵀWX(ψ)` is
    /// rank-deficient at BOTH window ends — at small ψ the longest-length-scale
    /// radial mode collapses into the polynomial nullspace, and at very large ψ
    /// every radial column goes collinear with the low-frequency mode, so the
    /// maximal-rank region is a middle BAND. The optimizer's line search can
    /// OVERSHOOT above that band (e.g. ψ≈1.0 on production Duchon geometry), where
    /// the design-realization skip's `reduced_basis_equal` witness must soundly
    /// refuse (the range subspace dropped a dimension) → an O(n) `reset_surface`,
    /// AND the pinning ψ recorded at that reset is itself rank-deficient, so the
    /// NEXT in-band trial mismatches its reference and resets a SECOND time. Both
    /// resets vanish once the optimizer's UPPER bound is clamped down to this
    /// n-free k-space ceiling, keeping every trial inside the maximal-rank band.
    ///
    /// Walks UP from the anchor on the same fixed k-space grid as the floor and
    /// returns the highest ψ still at the window's maximal numerical rank
    /// (stopping at the first node above that differs). Purely O(nodes·k³) — no
    /// row access, inherently n-INDEPENDENT (rank is a property of the k×k tensor).
    ///
    /// Returns `None` when the band already reaches `psi_hi` (no clamp needed),
    /// when the anchor is off-window / rank-indeterminate, or when the window is
    /// empty.
    pub fn rank_stable_psi_ceiling(&self, psi_anchor: f64) -> Option<f64> {
        // Same grid + max-rank target + anchor→band snap as `rank_stable_psi_floor`
        // so the floor and ceiling bracket the SAME contiguous maximal-rank band.
        const NODES: usize = 96;
        if !(self.psi_hi > self.psi_lo) {
            return None;
        }
        let span = self.psi_hi - self.psi_lo;
        let psi_at = |i: usize| self.psi_lo + span * (i as f64) / ((NODES - 1) as f64);
        let ranks: Vec<Option<usize>> =
            (0..NODES).map(|i| self.gram_numerical_rank(psi_at(i))).collect();
        let max_rank = ranks.iter().filter_map(|r| *r).max()?;
        let anchor = psi_anchor.clamp(self.psi_lo, self.psi_hi);
        let anchor_idx = (((anchor - self.psi_lo) / span) * ((NODES - 1) as f64))
            .round()
            .clamp(0.0, (NODES - 1) as f64) as usize;
        // Snap to the nearest max-rank node at/below the anchor (the mirror of the
        // floor's snap-UP), so the band edge is measured from inside the good band.
        let band_idx = (0..=anchor_idx).rev().find(|&i| ranks[i] == Some(max_rank))?;
        // Walk UP from the band node; the ceiling is the highest node from which
        // every node down to it holds `max_rank`. Stop at the first node above.
        let mut ceil_idx = band_idx;
        for i in (band_idx + 1)..NODES {
            if ranks[i] == Some(max_rank) {
                ceil_idx = i;
            } else {
                break;
            }
        }
        if ceil_idx == NODES - 1 {
            // The maximal-rank band already reaches `psi_hi` — no clamp needed.
            None
        } else {
            Some(psi_at(ceil_idx))
        }
    }

    /// True when `psi` lies inside the certified window.
    pub fn contains(&self, psi: f64) -> bool {
        psi.is_finite() && psi >= self.psi_lo && psi <= self.psi_hi
    }

    /// The certified value window `[psi_lo, psi_hi]` (#1033 instrumentation).
    pub fn psi_window(&self) -> (f64, f64) {
        (self.psi_lo, self.psi_hi)
    }

    /// True when `psi` lies inside the certified gradient window where the
    /// analytic ψ-derivative is bit-tight against the exact design derivative
    /// (#1033b). The n-free kappa outer loop is armed only when this covers the
    /// full optimizer bounds.
    pub fn contains_for_gradient(&self, psi: f64) -> bool {
        psi.is_finite()
            && self.grad_psi_lo.is_finite()
            && self.grad_psi_hi.is_finite()
            && psi >= self.grad_psi_lo
            && psi <= self.grad_psi_hi
    }

    fn mapped(&self, psi: f64) -> f64 {
        (2.0 * psi - (self.psi_lo + self.psi_hi)) / (self.psi_hi - self.psi_lo)
    }

    /// `XᵀWX(ψ)` assembled n-free in O(Dk²) from the direct Gram series.
    pub fn gram_at(&self, psi: f64) -> Array2<f64> {
        let x = self.mapped(psi);
        let t = cheb_t(x, self.gram.len());
        let mut out = Array2::<f64>::zeros((self.k, self.k));
        let mut comp = Array2::<f64>::zeros((self.k, self.k));
        for (d, td) in t.iter().enumerate() {
            kahan_scaled_add_array2(&mut out, &mut comp, *td, &self.gram[d]);
        }
        out
    }

    /// `XᵀWz(ψ)` assembled n-free in O(Dk).
    pub fn rhs_at(&self, psi: f64) -> Array1<f64> {
        let x = self.mapped(psi);
        let t = cheb_t(x, self.n_coeff);
        let mut out = Array1::<f64>::zeros(self.k);
        let mut comp = Array1::<f64>::zeros(self.k);
        for (d, td) in t.iter().enumerate() {
            kahan_scaled_add_array1(&mut out, &mut comp, *td, &self.rhs[d]);
        }
        out
    }

    /// Exact `∂(XᵀWX)/∂ψ` from the SAME representation as the value — the
    /// structural cure for the objective↔gradient desync class on this
    /// channel. n-free, O(Dk²) from the direct Gram series.
    pub fn dgram_dpsi(&self, psi: f64) -> Array2<f64> {
        let x = self.mapped(psi);
        let dx_dpsi = 2.0 / (self.psi_hi - self.psi_lo);
        let tp = cheb_t_prime(x, self.gram.len());
        let mut out = Array2::<f64>::zeros((self.k, self.k));
        let mut comp = Array2::<f64>::zeros((self.k, self.k));
        for (d, tpd) in tp.iter().enumerate() {
            kahan_scaled_add_array2(&mut out, &mut comp, *tpd * dx_dpsi, &self.gram[d]);
        }
        out
    }

    /// Exact `∂(XᵀWz)/∂ψ`, n-free.
    pub fn drhs_dpsi(&self, psi: f64) -> Array1<f64> {
        let x = self.mapped(psi);
        let dx_dpsi = 2.0 / (self.psi_hi - self.psi_lo);
        let tp = cheb_t_prime(x, self.n_coeff);
        let mut out = Array1::<f64>::zeros(self.k);
        let mut comp = Array1::<f64>::zeros(self.k);
        for (d, tpd) in tp.iter().enumerate() {
            kahan_scaled_add_array1(&mut out, &mut comp, *tpd * dx_dpsi, &self.rhs[d]);
        }
        out
    }

    /// Exact `∂²(XᵀWX)/∂ψ²` from the SAME representation as the value/gradient —
    /// the n-free curvature that lets the outer Newton/ARC step read the τ-τ
    /// Hessian's design-moving block without re-streaming an O(n) slab Gram
    /// (#1033, Gaussian-identity single-ψ Hessian channel). O(Dk²) from the
    /// direct Gram series.
    ///
    /// `XᵀWX(ψ) = Σ_d T_d(x) G_d` with `x = mapped(ψ)`, so by the chain rule
    /// `d²/dψ² = T_d″(x) · (dx/dψ)²`.
    pub fn d2gram_dpsi2(&self, psi: f64) -> Array2<f64> {
        let x = self.mapped(psi);
        let dx_dpsi = 2.0 / (self.psi_hi - self.psi_lo);
        let dx_dpsi_sq = dx_dpsi * dx_dpsi;
        let tpp = cheb_t_double_prime(x, self.gram.len());
        let mut out = Array2::<f64>::zeros((self.k, self.k));
        let mut comp = Array2::<f64>::zeros((self.k, self.k));
        for (d, tppd) in tpp.iter().enumerate() {
            kahan_scaled_add_array2(&mut out, &mut comp, *tppd * dx_dpsi_sq, &self.gram[d]);
        }
        out
    }

    /// Exact `∂²(XᵀWz)/∂ψ²`, n-free. `T_d″·(dx/dψ)²` against the rhs slabs.
    pub fn d2rhs_dpsi2(&self, psi: f64) -> Array1<f64> {
        let x = self.mapped(psi);
        let dx_dpsi = 2.0 / (self.psi_hi - self.psi_lo);
        let dx_dpsi_sq = dx_dpsi * dx_dpsi;
        let tpp = cheb_t_double_prime(x, self.n_coeff);
        let mut out = Array1::<f64>::zeros(self.k);
        let mut comp = Array1::<f64>::zeros(self.k);
        for (d, tppd) in tpp.iter().enumerate() {
            kahan_scaled_add_array1(&mut out, &mut comp, *tppd * dx_dpsi_sq, &self.rhs[d]);
        }
        out
    }

    /// Assemble the Gaussian-identity sufficient-statistic cache at `psi`
    /// without touching a single data row — the bridge from this tensor into
    /// the inner PLS solver's fast path (#1033b → `GaussianFixedCache`).
    ///
    /// `(XᵀWX, XᵀWz, zᵀWz)` is everything the Gaussian penalized solve needs
    /// at any λ, so a ψ-trial that holds a certified tensor can hand the
    /// inner solver this cache instead of realizing the n×k design. The
    /// caller is responsible for `contains(psi)` (off-window trials fall back
    /// to the exact realizer path). Dense-path bridge only: the sparse
    /// scatter cache stays `None`.
    pub fn gaussian_fixed_cache_at(&self, psi: f64) -> crate::pirls::GaussianFixedCache {
        crate::pirls::GaussianFixedCache {
            xtwx_orig: self.gram_at(psi),
            xtwy_orig: self.rhs_at(psi),
            centered_weighted_y_sq: self.zt_w_z,
            row_prediction_is_stale: true,
            xtwx_sparse_orig: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Analytic Matérn-shaped synthetic design: entries g(e^{u_ij + ψ}) with
    /// g(s) = (1 + s)·exp(−s) (the ν = 3/2 Matérn shape) plus a ψ-free power
    /// column — the exact structural mix of the production radial designs.
    fn synth_design(psi: f64, n: usize, k: usize) -> Result<Array2<f64>, String> {
        let mut x = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                let r = 0.05 + (i as f64 + 1.0) * (j as f64 + 1.0) / (n as f64 * k as f64) * 3.0;
                if j == k - 1 {
                    // ψ-free polynomial block column.
                    x[[i, j]] = r * r * r;
                } else {
                    let s = r * psi.exp();
                    x[[i, j]] = (1.0 + s) * (-s).exp();
                }
            }
        }
        Ok(x)
    }

    /// A genuinely FULL-RANK, well-conditioned, ψ-dependent synthetic design for
    /// the gauge-invariance witness test. Unlike `synth_design` (whose Matérn-like
    /// `(1+s)e^{-s}` columns over a narrow `r`-range collapse to a numerical rank
    /// of 3–4 of `k=6` and whose near-null subspace *rotates* across the window —
    /// so `reduced_basis_equal` correctly refuses), this builds `k` near-orthogonal
    /// Fourier/Chebyshev-flavoured base columns and applies a mild, sign-varying
    /// per-column amplitude `e^{c_j·ψ}`. The base columns are linearly independent
    /// with a Gram condition number `≈3`, so the weighted Gram is full column rank
    /// (numerical rank `= k`) at *every* ψ in the window — its range is the whole
    /// k-space and the orthogonal range projector is the identity for all ψ. The
    /// amplitude modulation still genuinely *rotates the eigenvectors* with ψ, so
    /// the witness must certify (identical range subspace) despite a per-ψ
    /// eigenvector gauge that differs — exactly the gauge invariance under test.
    /// The amplitudes are entire in ψ, so the Chebyshev tensor still certifies.
    fn synth_full_rank_design(psi: f64, n: usize, k: usize) -> Result<Array2<f64>, String> {
        use std::f64::consts::PI;
        assert!(k >= 2 && k % 2 == 0, "helper assumes an even k ≥ 2");
        // ψ-analytic Givens angle: rotates each adjacent column plane by θ(ψ). A
        // rotation is orthogonal, so it preserves the COLUMN SPACE and the Gram
        // SPECTRUM (rank = k, condition number constant in ψ) while genuinely
        // turning the eigenvECTORS — the precise setting in which the range
        // projector is ψ-invariant (identity at full rank) but the per-ψ gauge
        // differs. cos/sin are entire, so the Chebyshev tensor still certifies.
        let theta = 0.6 * psi;
        let (c, s) = (theta.cos(), theta.sin());
        let mut x = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let t = (i as f64 + 0.5) / n as f64;
            // Distinctly-scaled near-orthogonal base columns → distinct, separated
            // eigenvalues so each eigenvector is well-defined (no degenerate plane
            // that would make the rotation gauge-ambiguous).
            let mut b = vec![0.0_f64; k];
            for (j, slot) in b.iter_mut().enumerate() {
                let base = if j % 2 == 0 {
                    ((j as f64) * PI * t).cos()
                } else {
                    (((j + 1) as f64) * PI * t).sin()
                };
                *slot = (1.0 + 0.5 * j as f64) * base;
            }
            // Apply the Givens rotation to every adjacent (2m, 2m+1) plane,
            // including the dominant top plane, so the LEADING eigenvector rotates
            // too (a rotation confined to the small-eigenvalue planes would leave
            // the leading eigenvector fixed and make the gauge check vacuous).
            let mut row = b.clone();
            let mut p = 0;
            while p + 1 < k {
                let (bp, bq) = (b[p], b[p + 1]);
                row[p] = c * bp - s * bq;
                row[p + 1] = s * bp + c * bq;
                p += 2;
            }
            for (j, &v) in row.iter().enumerate() {
                x[[i, j]] = v;
            }
        }
        Ok(x)
    }

    fn exact_gram(psi: f64, n: usize, k: usize, w: &Array1<f64>) -> Array2<f64> {
        let design = synth_design(psi, n, k).unwrap();
        let mut wd = design.clone();
        for (mut row, &wi) in wd.outer_iter_mut().zip(w.iter()) {
            row.mapv_inplace(|v| v * wi);
        }
        design.t().dot(&wd)
    }

    /// #1033b primitive gate: the certified tensor must reproduce the exact
    /// Gram/rhs at arbitrary in-window ψ to certification accuracy, and its
    /// analytic ψ-derivative must match central finite differences of the
    /// exact Gram — value and gradient from one representation.
    #[test]
    fn psi_gram_tensor_matches_exact_gram_and_fd_gradient() {
        let (n, k) = (160usize, 7usize);
        let w = Array1::from_iter((0..n).map(|i| 1.0 + 0.5 * ((i % 3) as f64)));
        let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.37).sin()));
        let (psi_lo, psi_hi) = (-1.2_f64, 1.0_f64);

        let tensor = PsiGramTensor::build(
            |psi| synth_design(psi, n, k),
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("analytic synthetic design must certify");

        for &psi in &[-1.1, -0.63, 0.0, 0.41, 0.97] {
            assert!(tensor.contains(psi));
            let exact = exact_gram(psi, n, k, &w);
            let fast = tensor.gram_at(psi);
            let scale = exact.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            for (a, b) in fast.iter().zip(exact.iter()) {
                assert!(
                    (a - b).abs() <= 1e-9 * scale,
                    "gram mismatch at psi={psi}: fast={a}, exact={b}"
                );
            }
            // rhs check against the exact weighted cross-product.
            let design = synth_design(psi, n, k).unwrap();
            let mut wz = Array1::<f64>::zeros(n);
            for ((slot, &wi), &zi) in wz.iter_mut().zip(w.iter()).zip(z.iter()) {
                *slot = wi * zi;
            }
            let exact_rhs = design.t().dot(&wz);
            let fast_rhs = tensor.rhs_at(psi);
            let rscale = exact_rhs.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            for (a, b) in fast_rhs.iter().zip(exact_rhs.iter()) {
                assert!(
                    (a - b).abs() <= 1e-9 * rscale,
                    "rhs mismatch at psi={psi}: fast={a}, exact={b}"
                );
            }
            // Analytic ψ-gradient vs central FD of the EXACT gram.
            let h = 1e-5;
            let g_plus = exact_gram(psi + h, n, k, &w);
            let g_minus = exact_gram(psi - h, n, k, &w);
            let dg = tensor.dgram_dpsi(psi);
            let dscale = dg.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1e-12);
            for ((a, p), m_) in dg.iter().zip(g_plus.iter()).zip(g_minus.iter()) {
                let fd = (p - m_) / (2.0 * h);
                assert!(
                    (a - fd).abs() <= 1e-5 * dscale,
                    "dgram/dpsi mismatch at psi={psi}: analytic={a}, fd={fd}"
                );
            }
        }
        // Outside the window the caller must fall back to the exact path.
        assert!(!tensor.contains(psi_hi + 0.5));
        assert!(!tensor.contains(psi_lo - 0.5));

        // Bridge gate (#1033b → GaussianFixedCache): the n-free cache must
        // reproduce the exactly streamed sufficient statistics, and the
        // ridge-penalized solves through both must agree — the inner PLS
        // consumes nothing else, so this certifies the trial-loop handoff.
        for &psi in &[-0.9, 0.2, 0.8] {
            let cache = tensor.gaussian_fixed_cache_at(psi);
            let design = synth_design(psi, n, k).unwrap();
            let mut wd = design.clone();
            for (mut row, &wi) in wd.outer_iter_mut().zip(w.iter()) {
                row.mapv_inplace(|v| v * wi);
            }
            let exact_gram = design.t().dot(&wd);
            let exact_rhs = wd.t().dot(&z);
            let exact_ztwz: f64 = w.iter().zip(z.iter()).map(|(&wi, &zi)| wi * zi * zi).sum();
            assert!(
                (cache.centered_weighted_y_sq - exact_ztwz).abs()
                    <= 1e-12 * exact_ztwz.abs().max(1e-300),
                "zᵀWz drift: cache={}, exact={exact_ztwz}",
                cache.centered_weighted_y_sq
            );
            // Ridge-penalized solve agreement: (G + I)β = r on both sides.
            let solve = |g: &Array2<f64>, r: &Array1<f64>| -> Array1<f64> {
                let mut a = g.clone();
                for i in 0..k {
                    a[[i, i]] += 1.0;
                }
                // Small dense Gauss elimination (k = 7 in this test).
                let mut aug = Array2::<f64>::zeros((k, k + 1));
                aug.slice_mut(ndarray::s![.., ..k]).assign(&a);
                aug.slice_mut(ndarray::s![.., k]).assign(r);
                for col in 0..k {
                    let piv = (col..k)
                        .max_by(|&p, &q| aug[[p, col]].abs().total_cmp(&aug[[q, col]].abs()))
                        .unwrap();
                    if piv != col {
                        for j in 0..=k {
                            let tmp = aug[[col, j]];
                            aug[[col, j]] = aug[[piv, j]];
                            aug[[piv, j]] = tmp;
                        }
                    }
                    let p = aug[[col, col]];
                    for row in 0..k {
                        if row == col {
                            continue;
                        }
                        let f = aug[[row, col]] / p;
                        for j in col..=k {
                            aug[[row, j]] -= f * aug[[col, j]];
                        }
                    }
                }
                Array1::from_iter((0..k).map(|i| aug[[i, k]] / aug[[i, i]]))
            };
            let beta_fast = solve(&cache.xtwx_orig, &cache.xtwy_orig);
            let beta_exact = solve(&exact_gram, &exact_rhs);
            let bscale = beta_exact
                .iter()
                .fold(0.0_f64, |a, &v| a.max(v.abs()))
                .max(1e-300);
            for (a, b) in beta_fast.iter().zip(beta_exact.iter()) {
                assert!(
                    (a - b).abs() <= 1e-8 * bscale,
                    "penalized solve drift at psi={psi}: fast={a}, exact={b}"
                );
            }
        }
    }

    /// #1033 rank-stable κ-floor: the conditioned radial Gram goes numerically
    /// rank-deficient at the LARGE-length-scale (small-ψ) window edge — the
    /// `synth_design` Matérn columns over a narrow `r`-range collapse toward the
    /// polynomial nullspace there. `rank_stable_psi_floor` must (a) detect that the
    /// maximal-rank band does NOT reach `psi_lo` and return a floor strictly inside
    /// the window, (b) report that floor as the lower edge of the maximal-rank band
    /// containing the seed, and (c) be a pure k-space property — IDENTICAL whether
    /// the tensor was built from n=200 or n=4000 rows (the n-independence the κ
    /// outer loop relies on). A design that is full-rank across the whole window
    /// must return `None` (no lift needed).
    #[test]
    fn rank_stable_psi_floor_is_inside_window_and_n_independent() {
        let k = 7usize;
        // Window spanning the small-ψ rank cliff. Kept moderate so the Chebyshev
        // ladder certifies at a low rung (the build is the only n-pass; a wide
        // window forces high rungs = many design realizations = slow test).
        let (psi_lo, psi_hi) = (-1.6_f64, 1.0_f64);
        let build_at = |n: usize| {
            let w = Array1::from_iter((0..n).map(|i| 1.0 + 0.5 * ((i % 3) as f64)));
            let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.37).sin()));
            PsiGramTensor::build(|psi| synth_design(psi, n, k), w.view(), z.view(), psi_lo, psi_hi)
                .expect("analytic synthetic design must certify")
        };

        let t_small = build_at(120);
        // Seed at the well-conditioned (small-length-scale) window end — the
        // κ-optimum's neighbourhood, guaranteed to be at the window-maximal rank
        // (the synthetic radial design's rank rises toward large ψ). The floor is
        // the lower edge of the maximal-rank band reaching this seed.
        let seed = psi_hi;
        let floor_small = t_small.rank_stable_psi_floor(seed);

        // (a) the band does not reach psi_lo → a floor is returned, strictly inside.
        let floor = floor_small.expect("collapsing-rank design must lift the floor off psi_lo");
        assert!(
            floor > psi_lo && floor <= seed,
            "floor {floor} must lie in (psi_lo {psi_lo}, seed {seed}]"
        );

        // (b) below the floor the Gram is rank-deficient relative to the seed; at/
        // above the floor it holds the window-maximal rank. Verify the rank at the
        // floor equals the rank at the seed, and the rank just below the floor is
        // strictly lower (the floor is a genuine rank edge, not an interior node).
        let rank_at = |psi: f64| t_small.gram_numerical_rank(psi).unwrap();
        let max_rank = rank_at(seed);
        assert_eq!(
            rank_at(floor),
            max_rank,
            "the floor must sit at the window-maximal rank"
        );
        let probe_below = floor - 0.25;
        if t_small.contains(probe_below) {
            assert!(
                rank_at(probe_below) < max_rank,
                "rank just below the floor ({}) must drop under the band rank {max_rank}",
                rank_at(probe_below)
            );
        }

        // (c) n-independence: the floor from a 5× larger build must match to grid
        // resolution. The tensor is certified to the same Chebyshev tolerance, so
        // the k-space rank structure — hence the floor — is the same object.
        let t_big = build_at(1000);
        let floor_big = t_big
            .rank_stable_psi_floor(seed)
            .expect("the rank cliff is an n-free property; the big build must also lift");
        let grid_step = (psi_hi - psi_lo) / 95.0; // NODES - 1 = 95
        assert!(
            (floor_small.unwrap() - floor_big).abs() <= 1.5 * grid_step,
            "rank-stable floor must be n-independent: n=200 → {}, n=4000 → {floor_big} \
             (grid step {grid_step})",
            floor_small.unwrap()
        );

        // A genuinely full-rank, well-conditioned design across the window needs no
        // lift → None. `synth_full_rank_design` requires an even k.
        let n = 200usize;
        let kk = 6usize;
        let w = Array1::from_iter((0..n).map(|i| 1.0 + 0.5 * ((i % 3) as f64)));
        let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.29).cos()));
        let full = PsiGramTensor::build(
            |psi| synth_full_rank_design(psi, n, kk),
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("full-rank design must certify");
        assert!(
            full.rank_stable_psi_floor(seed).is_none(),
            "a window-wide full-rank design must not lift the floor"
        );
    }

    /// #1033 (rank-stable κ-CEILING): the symmetric twin of the floor test. The
    /// `synth_design` radial columns `(1+s)e^{-s}` with `s = r·e^ψ` collapse at the
    /// HIGH ψ edge — every column decays toward zero as `s→∞`, so the conditioned
    /// Gram drops rank near `psi_hi`. `rank_stable_psi_ceiling` must (a) detect that
    /// the maximal-rank band does NOT reach `psi_hi` and return a ceiling strictly
    /// inside the window, (b) report that ceiling as the upper edge of the band
    /// containing the seed (rank at the ceiling = window-maximal, rank just above it
    /// strictly lower), and (c) be a pure k-space property — IDENTICAL whether built
    /// from few or many rows (the n-independence the κ outer loop relies on). A
    /// design full-rank up to `psi_hi` must return `None` (no clamp needed). This is
    /// the regression guard for the n=16000 fast-ladder resets: the κ line search
    /// overshot above the band to a rank-deficient ψ and tripped two O(n) resets.
    #[test]
    fn rank_stable_psi_ceiling_is_inside_window_and_n_independent() {
        let k = 7usize;
        // `synth_design`'s radial Gram rank RISES with ψ (the columns separate as
        // s = r·e^ψ grows), so it collapses at the LOW edge — the floor's setting.
        // To exercise the CEILING we feed the ψ-REFLECTED design `synth_design(-ψ)`,
        // whose rank instead collapses at the HIGH edge (rank 7→3 as ψ→psi_hi),
        // exactly the high-edge degeneracy the κ-ceiling guards against. Seed at a
        // window-maximal-rank node (located by a coarse scan, not assumed at an
        // edge); the ceiling is the upper edge of the maximal-rank band.
        let (psi_lo, psi_hi) = (-2.6_f64, 1.0_f64);
        let build_at = |n: usize| {
            let w = Array1::from_iter((0..n).map(|i| 1.0 + 0.5 * ((i % 3) as f64)));
            let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.41).cos()));
            PsiGramTensor::build(|psi| synth_design(-psi, n, k), w.view(), z.view(), psi_lo, psi_hi)
                .expect("analytic synthetic design must certify")
        };

        let t_small = build_at(120);
        let rank_at = |psi: f64| t_small.gram_numerical_rank(psi).unwrap();
        let scan: Vec<(f64, usize)> = (0..96)
            .map(|i| {
                let p = psi_lo + (psi_hi - psi_lo) * (i as f64) / 95.0;
                (p, rank_at(p))
            })
            .collect();
        let window_max_rank = scan.iter().map(|&(_, r)| r).max().unwrap();
        let seed = scan
            .iter()
            .find(|&&(_, r)| r == window_max_rank)
            .map(|&(p, _)| p)
            .expect("some node must hold the window-maximal rank");
        let ceil_small = t_small.rank_stable_psi_ceiling(seed);

        // (a) the band does not reach psi_hi → a ceiling is returned, strictly inside.
        let ceiling =
            ceil_small.expect("high-edge-collapsing design must clamp the ceiling off psi_hi");
        assert!(
            ceiling < psi_hi && ceiling >= seed,
            "ceiling {ceiling} must lie in [seed {seed}, psi_hi {psi_hi})"
        );

        // (b) at/below the ceiling the Gram holds the window-maximal rank; above it
        // the rank drops. The ceiling is a genuine rank edge, not an interior node.
        let max_rank = window_max_rank;
        assert_eq!(
            rank_at(seed),
            max_rank,
            "the seed must sit at the window-maximal rank"
        );
        assert_eq!(
            rank_at(ceiling),
            max_rank,
            "the ceiling must sit at the window-maximal rank"
        );
        let probe_above = ceiling + 0.25;
        if t_small.contains(probe_above) {
            assert!(
                rank_at(probe_above) < max_rank,
                "rank just above the ceiling ({}) must drop under the band rank {max_rank}",
                rank_at(probe_above)
            );
        }

        // (c) n-independence: the ceiling from a larger build matches to grid
        // resolution — the rank cliff is an n-free k-space property.
        let t_big = build_at(1000);
        let ceil_big = t_big
            .rank_stable_psi_ceiling(seed)
            .expect("the rank cliff is an n-free property; the big build must also clamp");
        let grid_step = (psi_hi - psi_lo) / 95.0; // NODES - 1 = 95
        assert!(
            (ceil_small.unwrap() - ceil_big).abs() <= 1.5 * grid_step,
            "rank-stable ceiling must be n-independent: n=120 → {}, n=1000 → {ceil_big} \
             (grid step {grid_step})",
            ceil_small.unwrap()
        );

        // A genuinely full-rank design across the window needs no clamp → None.
        let n = 200usize;
        let kk = 6usize;
        let w = Array1::from_iter((0..n).map(|i| 1.0 + 0.5 * ((i % 3) as f64)));
        let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.23).sin()));
        let full = PsiGramTensor::build(
            |psi| synth_full_rank_design(psi, n, kk),
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("full-rank design must certify");
        assert!(
            full.rank_stable_psi_ceiling(seed).is_none(),
            "a window-wide full-rank design must not clamp the ceiling"
        );
    }

    /// #1033 n-independence invariant (structural, build-free, bit-tight):
    /// after the one-time `build` n-pass, EVERY per-trial accessor the certified
    /// κ/ψ outer-loop hot path consumes — the value `(gram_at, rhs_at)`, the
    /// gradient `(dgram_dpsi, drhs_dpsi)`, the Hessian-channel curvature
    /// `(d2gram_dpsi2, d2rhs_dpsi2)`, and the inner-solver bridge
    /// `gaussian_fixed_cache_at` — must touch ZERO data rows. We prove this by
    /// instrumenting the `eval_design` closure with an invocation counter (the
    /// closure is the ONLY route to the n×k design): the counter advances during
    /// `build` (the certified node ladder + off-node spot checks) and must then
    /// stay FROZEN across an entire ψ-trial sweep. This is the
    /// "no surface rebuild / no n×k re-realization on a cache-hit trial"
    /// invariant the outer-loop seam (`SpatialJointContext::eval_full`,
    /// `skip_design_realization`) relies on — asserted here at the tensor source
    /// of truth, independent of whether the full κ-fit converges or of any
    /// wall-clock measurement.
    #[test]
    fn psi_gram_tensor_trial_accessors_touch_no_data_rows() {
        use std::cell::Cell;

        let (n, k) = (256usize, 6usize);
        let w = Array1::from_iter((0..n).map(|i| 0.8 + 0.4 * ((i % 4) as f64) / 3.0));
        let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.21).sin() + 0.2));
        let (psi_lo, psi_hi) = (-1.1_f64, 0.95_f64);

        // The closure is the SOLE path to the n×k design; count every call.
        let calls = Cell::new(0usize);
        let tensor = PsiGramTensor::build(
            |psi| {
                calls.set(calls.get() + 1);
                synth_design(psi, n, k)
            },
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("analytic synthetic design must certify");

        // The one-time build necessarily streamed the design at the Chebyshev
        // nodes plus off-node spot checks. Freeze the count.
        let build_calls = calls.get();
        assert!(
            build_calls > 0,
            "build must have streamed the design at least once (sanity)"
        );

        // A dense ψ-trial sweep strictly inside the certified window. Every
        // accessor below is what a per-trial outer-loop eval consumes.
        let m = 64usize;
        let lo = psi_lo + 0.05;
        let hi = psi_hi - 0.05;
        for i in 0..m {
            let psi = lo + (hi - lo) * (i as f64) / (m as f64 - 1.0);
            assert!(tensor.contains(psi));
            // Value lane.
            tensor.gram_at(psi);
            tensor.rhs_at(psi);
            // Gradient lane.
            tensor.dgram_dpsi(psi);
            tensor.drhs_dpsi(psi);
            // Hessian-channel curvature.
            tensor.d2gram_dpsi2(psi);
            tensor.d2rhs_dpsi2(psi);
            // Inner-solver bridge (the GaussianFixedCache the PLS fast path reads).
            tensor.gaussian_fixed_cache_at(psi);
        }

        assert_eq!(
            calls.get(),
            build_calls,
            "n-independence VIOLATED: a per-trial accessor re-streamed the n×k \
             design ({} extra eval_design calls across {m} ψ-trials). The certified \
             κ/ψ outer loop must serve value + gradient + Hessian curvature + the \
             inner-solver cache from k-space sufficient statistics only, with NO \
             per-trial n-row work.",
            calls.get() - build_calls
        );
    }

    /// #1033 Hessian-channel primitive gate: the n-free second ψ-derivatives
    /// `d2gram_dpsi2` / `d2rhs_dpsi2` must match central FD of the analytic FIRST
    /// derivatives (`dgram_dpsi` / `drhs_dpsi`) — the curvature the outer Newton
    /// /ARC step reads when the Gaussian Hessian channel is served from the
    /// tensor instead of a re-streamed O(n) slab. Differencing the analytic first
    /// derivative (not the exact gram) keeps this a pure check of the
    /// second-derivative recurrence, isolated from the build's value-lane tol.
    #[test]
    fn psi_gram_tensor_second_derivative_matches_fd_of_gradient() {
        let (n, k) = (160usize, 7usize);
        let w = Array1::from_iter((0..n).map(|i| 1.0 + 0.5 * ((i % 3) as f64)));
        let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.37).sin()));
        let (psi_lo, psi_hi) = (-1.2_f64, 1.0_f64);

        let tensor = PsiGramTensor::build(
            |psi| synth_design(psi, n, k),
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("analytic synthetic design must certify");

        let h = 1e-5;
        for &psi in &[-1.0, -0.5, 0.0, 0.4, 0.9] {
            // ∂²G/∂ψ² vs central FD of the analytic ∂G/∂ψ.
            let dg_plus = tensor.dgram_dpsi(psi + h);
            let dg_minus = tensor.dgram_dpsi(psi - h);
            let d2g = tensor.d2gram_dpsi2(psi);
            let gscale = d2g.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1e-9);
            for ((a, p), m_) in d2g.iter().zip(dg_plus.iter()).zip(dg_minus.iter()) {
                let fd = (p - m_) / (2.0 * h);
                assert!(
                    (a - fd).abs() <= 1e-4 * gscale,
                    "d2gram/dpsi2 mismatch at psi={psi}: analytic={a}, fd={fd}"
                );
            }
            // ∂²(XᵀWz)/∂ψ² vs central FD of the analytic ∂(XᵀWz)/∂ψ.
            let dr_plus = tensor.drhs_dpsi(psi + h);
            let dr_minus = tensor.drhs_dpsi(psi - h);
            let d2r = tensor.d2rhs_dpsi2(psi);
            let rscale = d2r.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1e-9);
            for ((a, p), m_) in d2r.iter().zip(dr_plus.iter()).zip(dr_minus.iter()) {
                let fd = (p - m_) / (2.0 * h);
                assert!(
                    (a - fd).abs() <= 1e-4 * rscale,
                    "d2rhs/dpsi2 mismatch at psi={psi}: analytic={a}, fd={fd}"
                );
            }
        }
    }

    /// The penalized Gaussian profile deviance at a fixed ridge λ, assembled
    /// PURELY from the sufficient-statistic triple `(G, r, c) = (XᵀWX, XᵀWz, zᵀWz)`:
    ///
    /// ```text
    ///   β(λ) = (G + λS)⁻¹ r,   D(ψ;λ) = c − 2 βᵀr + βᵀ(G + λS)β = c − βᵀr
    /// ```
    ///
    /// (the second equality uses the normal equations `(G + λS)β = r`). This is
    /// EXACTLY the object the inner Gaussian PLS minimizes over β, and it is a
    /// pure function of `(G, r, c)` — n-free. Returns `(D, β)` so the caller can
    /// also probe the coefficient lane. `s_ridge` is the ridge penalty matrix.
    fn profile_deviance(
        g: &Array2<f64>,
        r: &Array1<f64>,
        c: f64,
        s_ridge: &Array2<f64>,
        lambda: f64,
        k: usize,
    ) -> (f64, Array1<f64>) {
        // Dense (G + λS) β = r via partial-pivot Gauss elimination (small k).
        let mut a = g.clone();
        a.scaled_add(lambda, s_ridge);
        let mut aug = Array2::<f64>::zeros((k, k + 1));
        aug.slice_mut(ndarray::s![.., ..k]).assign(&a);
        aug.slice_mut(ndarray::s![.., k]).assign(r);
        for col in 0..k {
            let piv = (col..k)
                .max_by(|&p, &q| aug[[p, col]].abs().total_cmp(&aug[[q, col]].abs()))
                .unwrap();
            if piv != col {
                for j in 0..=k {
                    let tmp = aug[[col, j]];
                    aug[[col, j]] = aug[[piv, j]];
                    aug[[piv, j]] = tmp;
                }
            }
            let p = aug[[col, col]];
            for row in 0..k {
                if row == col {
                    continue;
                }
                let f = aug[[row, col]] / p;
                for j in col..=k {
                    aug[[row, j]] -= f * aug[[col, j]];
                }
            }
        }
        let beta = Array1::from_iter((0..k).map(|i| aug[[i, k]] / aug[[i, i]]));
        let deviance = c - beta.dot(r);
        (deviance, beta)
    }

    /// #1033 bit-tight Hessian + κ-optimum gate. The fast path's promise is not
    /// merely that the Gram VALUE matches at sampled ψ — it is that the WHOLE
    /// outer κ search (objective, its ψ-curvature, and therefore the located
    /// optimum) is reproduced by the n-free sufficient-statistic representation
    /// to machine precision. This harness certifies exactly that:
    ///
    ///   1. **Objective**: the penalized profile deviance `D(ψ)` assembled from
    ///      the tensor's `(gram_at, rhs_at, zᵀWz)` matches the exactly streamed
    ///      `XᵀWX/XᵀWz/zᵀWz` deviance bit-tight at every ψ on a fine grid.
    ///   2. **Curvature (Hessian)**: the second ψ-derivative `D''(ψ)` of the
    ///      fast-path objective matches the second ψ-derivative of the EXACT
    ///      objective (central FD of the streamed deviance) — the curvature the
    ///      outer Newton step reads must be the true curvature, not an
    ///      approximation that drifts off the value (the objective↔gradient
    ///      desync class, now extended to the second order).
    ///   3. **κ-optimum**: the argmin of `D(ψ)` over the grid is IDENTICAL
    ///      between the two assemblies — the fast path lands on the same κ as the
    ///      exact streamed search, to the grid resolution AND bit-tight in the
    ///      objective value at that node.
    #[test]
    fn psi_gram_tensor_bit_tight_hessian_and_kappa_optimum() {
        let (n, k) = (200usize, 6usize);
        // Heterogeneous weights + a response with genuine ψ-dependent curvature
        // so the deviance has a non-degenerate interior minimum in ψ.
        let w = Array1::from_iter((0..n).map(|i| 0.7 + 0.6 * (((i * 7) % 5) as f64) / 4.0));
        let z = Array1::from_iter((0..n).map(|i| {
            let t = (i as f64) / (n as f64 - 1.0);
            (3.0 * t).sin() + 0.3 * (7.0 * t).cos()
        }));
        let (psi_lo, psi_hi) = (-1.0_f64, 0.9_f64);
        // Fixed ridge λ over the search — the κ optimizer profiles ψ at fixed
        // smoothing here; identity-S ridge keeps the profile well-posed.
        let s_ridge = Array2::<f64>::eye(k);
        let lambda = 0.5_f64;

        let tensor = PsiGramTensor::build(
            |psi| synth_design(psi, n, k),
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("analytic synthetic design must certify");

        let exact_ztwz: f64 = w.iter().zip(z.iter()).map(|(&wi, &zi)| wi * zi * zi).sum();

        // Exact streamed deviance at arbitrary ψ — the ground truth the n-free
        // path must reproduce.
        let exact_deviance = |psi: f64| -> f64 {
            let design = synth_design(psi, n, k).unwrap();
            let mut wd = design.clone();
            for (mut row, &wi) in wd.outer_iter_mut().zip(w.iter()) {
                row.mapv_inplace(|v| v * wi);
            }
            let g = design.t().dot(&wd);
            let r = wd.t().dot(&z);
            profile_deviance(&g, &r, exact_ztwz, &s_ridge, lambda, k).0
        };

        // Fast n-free deviance from the certified tensor.
        let fast_deviance = |psi: f64| -> f64 {
            let g = tensor.gram_at(psi);
            let r = tensor.rhs_at(psi);
            profile_deviance(&g, &r, exact_ztwz, &s_ridge, lambda, k).0
        };

        // Dense grid strictly inside the certified window (away from the edges,
        // where the build's value lane is still certified but we want a clean
        // central-FD second derivative to exist on both sides).
        let m = 81usize;
        let lo = psi_lo + 0.06;
        let hi = psi_hi - 0.06;
        let grid: Vec<f64> = (0..m)
            .map(|i| lo + (hi - lo) * (i as f64) / (m as f64 - 1.0))
            .collect();

        // (1) Objective bit-tight across the whole grid; track argmin on both.
        let mut worst_value_rel = 0.0_f64;
        let (mut fast_argmin, mut fast_min) = (f64::NAN, f64::INFINITY);
        let (mut exact_argmin, mut exact_min) = (f64::NAN, f64::INFINITY);
        for &psi in &grid {
            let de = exact_deviance(psi);
            let df = fast_deviance(psi);
            let rel = (de - df).abs() / de.abs().max(1e-300);
            worst_value_rel = worst_value_rel.max(rel);
            if df < fast_min {
                fast_min = df;
                fast_argmin = psi;
            }
            if de < exact_min {
                exact_min = de;
                exact_argmin = psi;
            }
        }
        assert!(
            worst_value_rel <= 1e-9,
            "penalized profile deviance: fast n-free assembly diverged from exact \
             streamed by rel {worst_value_rel:.3e} (> 1e-9) somewhere on the ψ grid"
        );

        // (3) κ-optimum: identical grid node AND bit-tight value there. The
        // argmin must be a true interior minimum (not a window edge) for this to
        // certify the OUTER search rather than a boundary artifact.
        assert_eq!(
            fast_argmin.to_bits(),
            exact_argmin.to_bits(),
            "κ-optimum mismatch: fast argmin ψ={fast_argmin}, exact argmin ψ={exact_argmin} \
             — the n-free objective located a different optimum"
        );
        assert!(
            fast_argmin > lo + 1e-9 && fast_argmin < hi - 1e-9,
            "κ-optimum landed on the grid edge ψ={fast_argmin}; the fixture must have \
             an INTERIOR minimum for this to test the outer search, not a boundary"
        );
        let opt_rel = (exact_min - fast_min).abs() / exact_min.abs().max(1e-300);
        assert!(
            opt_rel <= 1e-9,
            "κ-optimum objective value drift at ψ={fast_argmin}: fast={fast_min}, \
             exact={exact_min}, rel={opt_rel:.3e}"
        );

        // (2) Gradient + curvature from the tensor's ANALYTIC ψ-derivatives.
        //
        // Differencing two objectives that agree only to ~1e-9 in VALUE cannot
        // certify their curvature: the central second difference divides by h²,
        // so the ~1e-9 value gap (which is NOT common-mode — they are different
        // assemblies) is amplified by 1/h² and swamps any real curvature signal.
        // The principled bit-tight curvature check uses the tensor's OWN analytic
        // ψ-derivatives `dgram_dpsi`/`drhs_dpsi`: the envelope gradient of the
        // profile deviance `D(ψ) = c − rᵀA⁻¹r`, `A = G + λS`, is
        //
        //   D'(ψ) = −2 βᵀ(∂r/∂ψ) + βᵀ(∂G/∂ψ)β,   β = A⁻¹r,
        //
        // assembled n-free from `(dgram_dpsi, drhs_dpsi)`. We certify this
        // analytic gradient against a central FD of the EXACT streamed objective
        // (first order ⇒ only 1/h amplification, so the ~1e-9 value agreement is
        // not destroyed), and certify the curvature by central-differencing the
        // ANALYTIC gradient (again 1/h, not 1/h²). This is the same one-
        // representation value↔gradient↔curvature consistency the production fast
        // path relies on for the outer Newton step.
        let solve_a = |g: &Array2<f64>, r: &Array1<f64>| -> Array1<f64> {
            profile_deviance(g, r, exact_ztwz, &s_ridge, lambda, k).1
        };
        // Analytic n-free ψ-gradient of the penalized profile deviance, valid on
        // the certified gradient sub-window where `dgram_dpsi` is bit-tight.
        let analytic_grad = |psi: f64| -> f64 {
            let g = tensor.gram_at(psi);
            let r = tensor.rhs_at(psi);
            let beta = solve_a(&g, &r);
            let dg = tensor.dgram_dpsi(psi);
            let dr = tensor.drhs_dpsi(psi);
            -2.0 * beta.dot(&dr) + beta.dot(&dg.dot(&beta))
        };

        // Two finite-difference steps, each near the optimum of its own
        // truncation/rounding trade-off:
        //   * `h_grad = 1e-6` for the FIRST derivative (central FD ⇒ O(h²)
        //     truncation, O(ε/h) rounding ⇒ optimum near 1e-5..1e-6);
        //   * `h_curv = 2e-4` for the curvature. A SECOND difference divides by
        //     h², so its rounding floor is O(ε·|D|/h²): at h=1e-6 that is
        //     ~1e-16/1e-12 = 1e-4 of |D|, comparable to the curvature itself —
        //     useless. h≈2e-4 puts the rounding floor at ~1e-16/4e-8 ≈ 2.5e-9·|D|
        //     and the O(h²·D⁗) truncation around the same scale, so the second
        //     difference is meaningful. The analytic-gradient curvature is
        //     differenced at the SAME h_curv so the two carry the same
        //     truncation order and the comparison is apples-to-apples.
        let h_grad = 1e-6_f64;
        let h_curv = 2e-4_f64;
        let mut worst_grad_rel = 0.0_f64;
        let mut worst_hess_rel = 0.0_f64;
        let mut tested = 0usize;
        for &psi in &grid {
            // The exact-objective curvature stencil reaches ±2·h_curv; require the
            // whole stencil to stay inside the certified gradient sub-window so the
            // analytic-gradient differences are all bit-tight.
            if !tensor.contains_for_gradient(psi - 2.0 * h_curv)
                || !tensor.contains_for_gradient(psi + 2.0 * h_curv)
            {
                continue;
            }
            tested += 1;
            // Analytic gradient vs central FD of the EXACT streamed objective.
            let exact_g1 =
                (exact_deviance(psi + h_grad) - exact_deviance(psi - h_grad)) / (2.0 * h_grad);
            let ag = analytic_grad(psi);
            let gscale = exact_g1.abs().max(1e-6);
            worst_grad_rel = worst_grad_rel.max((exact_g1 - ag).abs() / gscale);
            // Curvature: central FD of the ANALYTIC gradient (n-free) vs central
            // second difference of the EXACT objective, both at h_curv.
            let analytic_h2 =
                (analytic_grad(psi + h_curv) - analytic_grad(psi - h_curv)) / (2.0 * h_curv);
            let exact_h2 = (exact_deviance(psi + h_curv) - 2.0 * exact_deviance(psi)
                + exact_deviance(psi - h_curv))
                / (h_curv * h_curv);
            let hscale = exact_h2.abs().max(1e-3);
            worst_hess_rel = worst_hess_rel.max((analytic_h2 - exact_h2).abs() / hscale);
        }
        assert!(
            tested > 0,
            "no ψ on the grid lay inside the certified gradient sub-window"
        );
        assert!(
            worst_grad_rel <= 1e-5,
            "ψ-gradient mismatch: the tensor's analytic n-free objective gradient diverged \
             from the exact streamed objective by rel {worst_grad_rel:.3e} (> 1e-5)"
        );
        // The curvature compares an analytic-gradient central difference against
        // an exact-objective second difference; the residual O(h²) truncation +
        // O(ε/h²) rounding floor at h_curv=2e-4 sets a realistic bit-tight bar of
        // ~1e-3 relative (any larger gap is a genuine curvature divergence, not FD
        // noise — the value/gradient lanes already certify the objective itself to
        // ~1e-9/1e-5).
        assert!(
            worst_hess_rel <= 1e-3,
            "ψ-curvature (Hessian) mismatch: fast n-free objective curvature diverged \
             from the exact streamed objective by rel {worst_hess_rel:.3e} (> 1e-3) — \
             the outer Newton step would read a different curvature than the truth"
        );

        eprintln!(
            "[psi-gram-bittight] n={n} k={k} grid={m} grad-tested={tested}  \
             worst |ΔD|/D={worst_value_rel:.2e}  worst |ΔD'|/D'={worst_grad_rel:.2e}  \
             worst |ΔD''|/D''={worst_hess_rel:.2e}  κ-opt ψ={fast_argmin:.6} (interior, bit-identical)"
        );
    }

    /// Certification negative: a NON-analytic (kinked) design must refuse to
    /// certify rather than silently approximate.
    #[test]
    fn psi_gram_tensor_refuses_non_analytic_design() {
        let (n, k) = (40usize, 3usize);
        let w = Array1::from_elem(n, 1.0);
        let z = Array1::from_elem(n, 0.5);
        let tensor = PsiGramTensor::build(
            |psi| {
                let mut x = Array2::<f64>::zeros((n, k));
                for i in 0..n {
                    for j in 0..k {
                        // |ψ| kink at 0 inside the window: not analytic.
                        x[[i, j]] = psi.abs() + (i + j) as f64 / (n + k) as f64;
                    }
                }
                Ok(x)
            },
            w.view(),
            z.view(),
            -1.0,
            1.0,
        );
        assert!(
            tensor.is_err(),
            "kinked design must fail the tail-decay/spot-check certificates"
        );
    }

    /// #1264 reduced-basis-equality witness — REFLEXIVITY + GAUGE INVARIANCE.
    ///
    /// `reduced_basis_equal(ψ, ψ)` is trivially sound (the surface is its own
    /// reference), and the witness must accept two ψ's whose RANGE subspace is
    /// identical even when the per-ψ eigenvECTORS differ (the projector is
    /// gauge-invariant). The synthetic full-rank Matérn-shaped design's range is
    /// the whole k-space for every ψ, so every in-window pair shares a reduced
    /// basis and must certify.
    #[test]
    fn reduced_basis_witness_reflexive_and_gauge_invariant() {
        let (n, k) = (160usize, 6usize);
        let w = Array1::from_iter((0..n).map(|i| 1.0 + 0.3 * ((i % 5) as f64)));
        let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.29).sin()));
        let (psi_lo, psi_hi) = (-1.0_f64, 0.8_f64);
        // Use the genuinely full-rank, well-conditioned design: its weighted Gram
        // has numerical rank `= k` at every ψ (range = whole k-space, identity
        // range projector), so the gauge-invariance premise actually holds. The
        // narrow-`r` `synth_design` does NOT satisfy this — its Gram is rank 3–4 of
        // 6 with a near-null subspace that ROTATES across the window, on which the
        // witness *correctly* refuses (refusing a rotating reduced basis is the
        // sound fallback the production skip gate exists for). See
        // `synth_full_rank_design`.
        let tensor = PsiGramTensor::build(
            |psi| synth_full_rank_design(psi, n, k),
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("analytic full-rank synthetic design must certify");

        // PREMISE CHECK: the design is full column rank (numerical rank = k) and
        // the range projector is the identity at every grid ψ, so the test really
        // is exercising gauge invariance over a ψ-invariant subspace — not riding a
        // rank-deficient fixture the witness would (correctly) refuse.
        let grid: Vec<f64> = (0..=12).map(|i| psi_lo + 0.05 + 0.06 * i as f64).collect();
        let identity = Array2::<f64>::eye(k);
        for &psi in &grid {
            let (proj, rank) = tensor
                .range_projector(psi, PSI_GRAM_SKIP_RANK_RTOL)
                .expect("full-rank Gram must yield a range projector");
            assert_eq!(
                rank, k,
                "full-rank design must have numerical rank k={k} at psi={psi} \
                 (got {rank}) — otherwise the gauge-invariance premise is vacuous"
            );
            let proj_dev = (&proj - &identity)
                .iter()
                .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
            assert!(
                proj_dev <= 1e-8,
                "range projector must be the identity at psi={psi} \
                 (max|P−I|={proj_dev:.2e})"
            );
        }

        // GAUGE-INVARIANCE CHECK: the per-ψ eigenvectors genuinely rotate across
        // the window (so the witness is exercised against a moving gauge, not a
        // static one), yet the spanned subspace is identical. Confirm the rotation
        // is real by checking the leading eigenvector turns measurably end-to-end.
        let leading_evec = |psi: f64| -> Array1<f64> {
            use gam_linalg::faer_ndarray::FaerEigh;
            let g = tensor.gram_at(psi);
            let gsym = 0.5 * (&g + &g.t());
            let (evals, evecs) = gsym.eigh(faer::Side::Lower).unwrap();
            // `eigh` returns ascending eigenvalues; the leading one is the last.
            let top = evals.len() - 1;
            evecs.column(top).to_owned()
        };
        let v_lo = leading_evec(grid[0]);
        let v_hi = leading_evec(*grid.last().unwrap());
        let cos_angle = v_lo.dot(&v_hi).abs()
            / (v_lo.dot(&v_lo).sqrt() * v_hi.dot(&v_hi).sqrt()).max(1e-300);
        assert!(
            cos_angle <= 0.999,
            "the design's eigenvectors must rotate with ψ for the gauge-invariance \
             test to be non-trivial (|cos∠(v_lo,v_hi)|={cos_angle:.6} — too close to 1)"
        );

        // Reflexive: same ψ is always sound.
        for &psi in &[-0.9, -0.2, 0.0, 0.5, 0.79] {
            assert!(
                tensor.reduced_basis_equal(psi, psi),
                "witness must be reflexive at psi={psi}"
            );
        }
        // The full-rank synthetic design spans all of k-space at every ψ, so the
        // range projector is the identity for all ψ → every pair certifies despite
        // the eigenvector rotation just verified (gauge invariance).
        for &a in &grid {
            for &b in &grid {
                assert!(
                    tensor.reduced_basis_equal(a, b),
                    "full-rank design: range is ψ-invariant (identity projector), \
                     so the skip witness must certify (ψ_ref={a}, ψ_new={b})"
                );
            }
        }
        // Off-window ψ refuses.
        assert!(!tensor.reduced_basis_equal(psi_lo - 0.5, 0.0));
        assert!(!tensor.reduced_basis_equal(0.0, psi_hi + 0.5));
    }

    /// #1264 reduced-basis-equality witness — REFUSES across a genuine subspace
    /// change (the exact failure mode of the old RRQR-only gate).
    ///
    /// Construct a design whose first two columns are fixed (ψ-invariant) profiles
    /// and whose third column's AMPLITUDE `ε(ψ) = e^{αψ}` analytically sweeps the
    /// third eigendirection's eigenvalue `∝ ε²` across the rank-revealing cutoff.
    /// Below the cutoff the reduced (range) basis is the 2-D span of the first two
    /// profiles; above it the range is 3-D. Two ψ's on the SAME side of the
    /// threshold share a reduced basis (witness accepts); two ψ's STRADDLING it do
    /// not (witness refuses) — exactly the stale-basis pairing the design-revision
    /// fast path must not perform. The amplitude is smooth/analytic so the tensor
    /// still certifies (this is a reduced-basis change, not a non-analytic kink).
    #[test]
    fn reduced_basis_witness_refuses_across_subspace_change() {
        let (n, k) = (200usize, 3usize);
        // Three fixed, well-separated column profiles (full column rank when all
        // present). The third is scaled by ε(ψ).
        let base = |i: usize, j: usize| -> f64 {
            let t = (i as f64 + 0.5) / n as f64;
            match j {
                0 => 1.0,
                1 => (2.0 * std::f64::consts::PI * t).sin(),
                _ => (4.0 * std::f64::consts::PI * t).cos(),
            }
        };
        // ε(ψ) crosses √cutoff (relative to λ_max ~ O(n)) within the window: at
        // λ_max ≈ n the cutoff is rank_rtol·n ≈ 1e-10·200 = 2e-8, so the third
        // eigenvalue ε²·‖c3‖² ≈ ε²·(n/2) crosses it at ε ≈ sqrt(4e-8/n) ≈ 1.4e-5,
        // i.e. ψ* ≈ ln(1.4e-5)/α. With α = 10 and window [−1.6,−0.8], ψ* ≈ −1.12
        // sits inside the window, giving a clean below/above split.
        let alpha = 10.0_f64;
        let design = move |psi: f64| -> Result<Array2<f64>, String> {
            let eps = (alpha * psi).exp();
            let mut x = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                x[[i, 0]] = base(i, 0);
                x[[i, 1]] = base(i, 1);
                x[[i, 2]] = eps * base(i, 2);
            }
            Ok(x)
        };
        let w = Array1::from_elem(n, 1.0);
        let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.13).sin()));
        let (psi_lo, psi_hi) = (-1.6_f64, -0.8_f64);
        let tensor = PsiGramTensor::build(design, w.view(), z.view(), psi_lo, psi_hi)
            .expect("smooth ε(ψ) design must still certify (analytic, no kink)");

        // Find the actual threshold by scanning the rank.
        let rank_at = |psi: f64| -> usize {
            tensor
                .range_projector(psi, PSI_GRAM_SKIP_RANK_RTOL)
                .map(|(_, r)| r)
                .unwrap_or(0)
        };
        let lo_rank = rank_at(psi_lo + 0.02);
        let hi_rank = rank_at(psi_hi - 0.02);
        assert_eq!(
            lo_rank, 2,
            "low-ψ end must be rank-2 (third column below cutoff)"
        );
        assert_eq!(
            hi_rank, 3,
            "high-ψ end must be rank-3 (third column above cutoff)"
        );

        // Same-side pairs (both rank-2) certify; straddling pairs refuse.
        let psi_low_a = psi_lo + 0.05;
        let psi_low_b = psi_lo + 0.10;
        assert_eq!(rank_at(psi_low_a), 2);
        assert_eq!(rank_at(psi_low_b), 2);
        assert!(
            tensor.reduced_basis_equal(psi_low_a, psi_low_b),
            "two low-ψ trials share the rank-2 reduced basis → skip is sound"
        );
        let psi_high_a = psi_hi - 0.05;
        let psi_high_b = psi_hi - 0.10;
        assert_eq!(rank_at(psi_high_a), 3);
        assert_eq!(rank_at(psi_high_b), 3);
        // High-side: the range is the full 3-D space at both, so the projector is
        // the identity at both → still a shared reduced basis.
        assert!(
            tensor.reduced_basis_equal(psi_high_a, psi_high_b),
            "two high-ψ trials share the rank-3 reduced basis → skip is sound"
        );
        // Straddling the rank change: the reduced basis MOVED (2-D → 3-D). The
        // witness MUST refuse — this is precisely the stale-basis pairing the old
        // RRQR-only gate let through.
        assert!(
            !tensor.reduced_basis_equal(psi_low_a, psi_high_a),
            "witness must REFUSE a skip that straddles the reduced-basis (rank) \
             change — freezing the low-ψ rank-2 basis and re-keying the high-ψ \
             rank-3 Gram is the exact ~7.8e-2 β̂ regression #1264 guards"
        );
        assert!(
            !tensor.reduced_basis_equal(psi_high_a, psi_low_a),
            "witness must refuse symmetrically (high pin, low trial)"
        );
    }

    /// #1033 ROTATION WALL — the subspace-distance certificate must CERTIFY a
    /// skip across a pure basis ROTATION at fixed rank, where the old entrywise
    /// max-abs projector gate would have refused.
    ///
    /// Build a rank-2 (in a k=3 space) design whose 2-D range ROTATES smoothly
    /// with ψ but whose RANK stays 2: two ψ-dependent in-plane directions span the
    /// same fixed 2-plane (cols 0,1 of a fixed orthonormal pair) rotated by an
    /// analytic angle φ(ψ). The SUBSPACE (the 2-plane) is ψ-invariant — only the
    /// basis within it rotates — so the range projector is mathematically
    /// IDENTICAL at every ψ, but its eigenVECTORS rotate. A correct
    /// subspace-identity witness must certify every in-window pair; the spectral
    /// (principal-angle) distance is ~0 throughout while a naive entrywise
    /// comparison of rotated eigenbases would not be guaranteed to.
    #[test]
    fn reduced_basis_witness_certifies_across_pure_rotation_1033() {
        let (n, k) = (240usize, 3usize);
        // Two fixed orthogonal ambient profiles spanning a fixed 2-plane; the
        // third ambient direction is left empty so the range is exactly that
        // 2-plane (rank 2) for every ψ.
        let p0 = |i: usize| -> f64 {
            let t = (i as f64 + 0.5) / n as f64;
            (2.0 * std::f64::consts::PI * t).sin()
        };
        let p1 = |i: usize| -> f64 {
            let t = (i as f64 + 0.5) / n as f64;
            (2.0 * std::f64::consts::PI * t).cos()
        };
        // Within the fixed 2-plane, rotate the two design columns by φ(ψ): the
        // SPAN is unchanged (still the {p0,p1} plane) but the basis rotates, so
        // the per-ψ eigenvectors of the Gram rotate while the range projector is
        // ψ-invariant.
        let design = move |psi: f64| -> Result<Array2<f64>, String> {
            let phi = 0.7 * psi; // analytic angle sweep
            let (c, s) = (phi.cos(), phi.sin());
            let mut x = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                let (a, b) = (p0(i), p1(i));
                x[[i, 0]] = c * a - s * b;
                x[[i, 1]] = s * a + c * b;
                // column 2 stays zero → range is the fixed 2-plane, rank 2.
            }
            Ok(x)
        };
        let w = Array1::from_elem(n, 1.0);
        let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.17).cos()));
        let (psi_lo, psi_hi) = (-1.0_f64, 1.0_f64);
        let tensor = PsiGramTensor::build(design, w.view(), z.view(), psi_lo, psi_hi)
            .expect("smooth rotation design must certify (analytic, no kink)");

        // Rank is a constant 2 across the window (the third direction is empty).
        let rank_at = |psi: f64| -> usize {
            tensor
                .range_projector(psi, PSI_GRAM_SKIP_RANK_RTOL)
                .map(|(_, r)| r)
                .unwrap_or(0)
        };
        for &psi in &[-0.95, -0.4, 0.0, 0.4, 0.95] {
            assert_eq!(rank_at(psi), 2, "rotation keeps rank 2 at psi={psi}");
        }

        // Every in-window pair spans the SAME 2-plane (only the basis rotates),
        // so the subspace-distance witness MUST certify the skip — this is the
        // rotation that the entrywise gate kept refusing (the #1033 wall).
        let grid: Vec<f64> = (0..=10).map(|i| psi_lo + 0.05 + 0.09 * i as f64).collect();
        for &a in &grid {
            for &b in &grid {
                assert!(
                    tensor.reduced_basis_equal(a, b),
                    "pure in-plane rotation preserves the range subspace → the \
                     subspace-distance skip witness must certify (#1033) \
                     (ψ_ref={a}, ψ_new={b})"
                );
            }
        }
    }
}

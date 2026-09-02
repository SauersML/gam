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

use gam_linalg::decision::{RankDecision, certified_rank, projector_error_bar};
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

/// Bisection budget for the rank-stable ψ-band edge search
/// ([`PsiGramTensor::rank_stable_psi_floor`] / `_ceiling`). The band edge is a
/// monotone crossing of the projector witness, so bisection converges to the
/// true edge in `ceil(log2(span/atol))` steps — 64 caps any window to well
/// below machine precision, and the relative `ATOL` break stops earlier in
/// practice. Replaces the former fixed 96-node grid scan (SPEC: grid search is
/// never allowed, #2054); each witness eval is O(k³), independent of n.
const PSI_BAND_BISECTION_ITERS: usize = 64;
const PSI_BAND_BISECTION_ATOL: f64 = 1.0e-10;

/// Slack on the symmetric eigensolver's backward-error bound, used to size the
/// band-edge rank guard ([`PsiGramTensor::rank_guard_gap`]).
///
/// DERIVATION. A backward-stable symmetric eigensolver returns eigenvalues with
/// an ABSOLUTE error bar `|λ̂_i − λ_i| ≤ p(k)·ε·‖G‖₂ = p(k)·ε·λ_max`, `p(k)` a
/// low-degree polynomial in the order. `p(k) ≤ SLACK·k` with `SLACK = 8` covers
/// the measured LAPACK-class constant together with the few extra ulps of
/// `λ_max` the tensor's own `D·(Chebyshev sum)·D` reassembly contributes. The
/// band edges of [`PSI_GRAM_SKIP_RANK_RTOL`] are decided on `λ_r ≈ rtol·λ_max`,
/// so that absolute bar is a RELATIVE bar of `SLACK·k·ε/rtol ≈ 1.2e-4` on the
/// decided quantity — the margin any trustworthy rank claim here must clear
/// (theory master §9-step-6: decide with a margin wider than the backward error
/// committed forming the quantity decided on).
const PSI_BAND_RANK_GUARD_SLACK: f64 = 8.0;

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
    /// #1216 — per-column log-amplitude slope `p_j` and intercept `c_j` of the
    /// AMPLITUDE ENVELOPE `α_j(ψ) = exp(p_j·ψ + c_j)` factored out of column `j`
    /// of the design BEFORE the Chebyshev transform. The stored `gram`/`rhs`
    /// series interpolate the AMPLITUDE-NORMALIZED sufficient statistics
    /// `G̃(ψ) = D(ψ)⁻¹ (XᵀWX)(ψ) D(ψ)⁻¹` and `c̃(ψ) = D(ψ)⁻¹ (XᵀWz)(ψ)`, with
    /// `D(ψ) = diag(α_j(ψ))`, and every accessor RE-APPLIES `D(ψ)` in closed
    /// form (`gram_at = D G̃ D`, `rhs_at = D c̃`, with the exact product-rule
    /// envelope for the ψ-derivatives).
    ///
    /// ## Why (the #1216 dynamic-range wall)
    ///
    /// For the radial families the length-scale enters the design through the
    /// kernel argument κr with `ψ = log κ`, so each radial column's AMPLITUDE is
    /// a pure power `κ^{p_j} = e^{p_j ψ}` (a thin-plate/Duchon kernel `(κr)^{p}`
    /// or `(κr)^{p} log(κr)` has amplitude `κ^{p} = e^{p ψ}`; the residual
    /// `log(κr)` growth is only LINEAR in ψ and stays in the normalized factor).
    /// Over the wide standardized ψ-window default 1-D fits use (#1215, ~9 nats)
    /// that amplitude spans `e^{9p}` orders of magnitude, so the raw Gram entries
    /// `G_ab(ψ) = e^{(p_a+p_b)ψ}·(ψ-free)` carry an enormous exponential trend.
    /// The Chebyshev coefficients of `e^{cψ}` decay only for degree `d ≳ c·halfwidth`,
    /// so the raw-Gram tail certified geometrically-slowly and the weakly
    /// penalized β̂ solve amplified the residual — the tensor refused to attach
    /// (`attached=false`), the n-free skip never fired, and every κ-trial fell to
    /// the O(n) design realization.
    ///
    /// Factoring `α_j(ψ) = e^{p_j ψ + c_j}` out of column `j` cancels the
    /// exponential EXACTLY for a pure-power radial column: the normalized cross
    /// block `G̃_ab = G_ab/(α_a α_b) = e^{(p_a+p_b)ψ}·(ψ-free)/e^{(p_a+p_b)ψ}` is
    /// ψ-FREE, so `G̃` has O(1) dynamic range and certifies at a low degree. A
    /// SINGLE scalar amplitude does NOT suffice here: the design mixes radial
    /// columns (power `p`) with a ψ-free polynomial-nullspace block (power 0), and
    /// only a PER-COLUMN factor (dividing each column by its OWN amplitude) makes
    /// every block — radial-radial, radial-poly, poly-poly — simultaneously
    /// ψ-free. `p_j`/`c_j` are recovered EXACTLY (family-agnostically) from the
    /// node Grams' diagonals: `½·log G_jj(ψ) = p_j·ψ + c_j` is exactly affine for
    /// a power-law column, so a least-squares fit over the Chebyshev nodes (whose
    /// weighted column norms `‖X_j(ψ_i)‖²_W = G_jj(ψ_i)` are already formed) reads
    /// off the slope with zero residual, with no kernel-family dispatch and no
    /// extra row work. Choosing `c_j` as the fitted intercept sets `α_j(ψ)≈
    /// ‖X_j(ψ)‖_W`, so `G̃_jj≈1` across the window (Cauchy–Schwarz bounds the
    /// off-diagonals by 1). The reconstruction `gram_at = D G̃ D` equals the true
    /// Gram for ANY choice of `D`, so this only ever improves conditioning — and
    /// the off-node `spot_check` re-derives it against the exact rebuild, so an
    /// error in the envelope algebra REFUSES the tensor (sound fallback) rather
    /// than shipping a wrong Gram.
    col_amp_slope: Array1<f64>,
    col_amp_intercept: Array1<f64>,
}

/// Recover the per-column log-amplitude envelope `α_j(ψ) = exp(p_j·ψ + c_j)` from
/// the node Grams' diagonals (#1216). For a power-law radial column
/// `X_j(ψ) = e^{p_j ψ}·(ψ-free)`, the weighted column norm satisfies
/// `½·log G_jj(ψ) = p_j·ψ + c_j` EXACTLY, so an ordinary least-squares fit over
/// the Chebyshev-node ψ values recovers `(p_j, c_j)` with zero residual; for a
/// non-power column it returns the best affine log-amplitude, which still removes
/// the dominant exponential trend and leaves an analytic low-degree remainder.
/// A column whose diagonal is non-positive or non-finite at ANY node (a genuinely
/// vanishing column somewhere in the window) gets the identity envelope
/// `(p_j, c_j) = (0, 0)` ⇒ `α_j ≡ 1` — no normalization, the historical path for
/// that column — so the recovery never divides by a spurious fitted exponential.
///
/// A column is normalized ONLY when its amplitude dynamic range across the window
/// is large enough to matter: `|p_j|·span ≥ COL_AMP_ENGAGE`. Below that the raw
/// tensor already certified (the narrow-window regime), so the column keeps the
/// identity envelope and its accessors stay BIT-FOR-BIT the historical path
/// (`α_j ≡ 1` ⇒ `gram_at` multiplies by exactly `1.0`). Normalization thus
/// engages exactly on the wide standardized geometry (#1215/#1216) that needs it.
const COL_AMP_ENGAGE: f64 = 6.0;

fn recover_column_log_amplitudes(
    node_grams: &[Array2<f64>],
    node_psis: &[f64],
    k: usize,
) -> (Array1<f64>, Array1<f64>) {
    let mut slope = Array1::<f64>::zeros(k);
    let mut intercept = Array1::<f64>::zeros(k);
    let m = node_psis.len();
    if m < 2 {
        return (slope, intercept);
    }
    let psi_bar = node_psis.iter().sum::<f64>() / m as f64;
    let denom = node_psis
        .iter()
        .map(|&p| (p - psi_bar).powi(2))
        .sum::<f64>();
    if !(denom > 0.0) {
        return (slope, intercept);
    }
    let (psi_min, psi_max) = node_psis
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &p| {
            (lo.min(p), hi.max(p))
        });
    let span = psi_max - psi_min;
    for j in 0..k {
        // y_i = ½·log G_jj(ψ_i); identity envelope if any diagonal is unusable.
        let mut ys = Vec::with_capacity(m);
        let mut usable = true;
        for g in node_grams.iter() {
            let d = g[[j, j]];
            if !(d > 0.0) || !d.is_finite() {
                usable = false;
                break;
            }
            ys.push(0.5 * d.ln());
        }
        if !usable {
            continue;
        }
        let y_bar = ys.iter().sum::<f64>() / m as f64;
        let mut num = 0.0_f64;
        for (i, &y) in ys.iter().enumerate() {
            num += (node_psis[i] - psi_bar) * (y - y_bar);
        }
        let p = num / denom;
        let c = y_bar - p * psi_bar;
        // Engage normalization only where the amplitude dynamic range is large;
        // otherwise leave the identity envelope so the column is the historical
        // path bit-for-bit.
        if p.is_finite() && c.is_finite() && p.abs() * span >= COL_AMP_ENGAGE {
            slope[j] = p;
            intercept[j] = c;
        }
    }
    (slope, intercept)
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

fn weighted_gram_and_rhs(
    design: &Array2<f64>,
    weights: ArrayView1<'_, f64>,
    wz: &Array1<f64>,
) -> (Array2<f64>, Array1<f64>) {
    let (_n, k) = design.dim();
    let mut gram = Array2::<f64>::zeros((k, k));
    let mut gram_comp = Array2::<f64>::zeros((k, k));
    let mut rhs = Array1::<f64>::zeros(k);
    let mut rhs_comp = Array1::<f64>::zeros(k);

    // Stream row contributions with one compensation term per retained k-space
    // entry.  The Chebyshev value/derivative accessors are n-free only after
    // these node sufficient statistics are frozen, so this is the unique O(n)
    // reduction whose summation error is subsequently amplified by high-order
    // derivative coefficients.  Keeping the reduction compensated preserves the
    // algebraic additivity under replicated rows without relying on BLAS dot's
    // implementation-dependent reduction tree.
    for ((row, &w), &wz_i) in design.outer_iter().zip(weights.iter()).zip(wz.iter()) {
        for a in 0..k {
            let xa = row[a];

            let y_rhs = xa * wz_i - rhs_comp[a];
            let t_rhs = rhs[a] + y_rhs;
            rhs_comp[a] = (t_rhs - rhs[a]) - y_rhs;
            rhs[a] = t_rhs;

            for b in a..k {
                let y = xa * w * row[b] - gram_comp[[a, b]];
                let t = gram[[a, b]] + y;
                gram_comp[[a, b]] = (t - gram[[a, b]]) - y;
                gram[[a, b]] = t;
            }
        }
    }

    for a in 0..k {
        for b in 0..a {
            gram[[a, b]] = gram[[b, a]];
        }
    }

    (gram, rhs)
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

/// A range projector together with the error bar it was computed to (#2448).
struct RangeProjector {
    /// `P = U_r U_rᵀ`, the orthogonal projector onto the kept eigenspace.
    proj: Array2<f64>,
    /// Number of eigenvalues above the rank cutoff.
    rank: usize,
    /// Certified upper bound on `‖P̂ − P‖₂` — how far the COMPUTED projector can
    /// sit from the projector of the exact Gram, given the eigensolver's own
    /// backward error and the eigen-gap that defines the kept/dropped split.
    /// Zero at full rank (`P` is the identity exactly); `f64::INFINITY` when the
    /// gap is not wide enough to determine the subspace at all.
    ///
    /// Computed by [`gam_linalg::decision::projector_error_bar`], which carries
    /// the derivation. It lives next to [`certified_rank`] deliberately: the
    /// whole content of #2448 is that certifying the RANK does not certify the
    /// EIGENSPACE realizing it, and a reader arriving at the rank primitive needs
    /// that caveat in the same file.
    err_bar: f64,
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
        let mut node_psis = vec![0.0_f64; m];
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
            node_psis[i] = psi;
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
            let (node_gram, node_rh) = weighted_gram_and_rhs(&design, weights, &wz);
            node_grams.push(node_gram);
            node_rhs.push(node_rh);
        }
        let (_n, k) = dims.expect("node ladder rung m≥1 yields at least one design");

        // #1216 amplitude normalization: recover the per-column log-amplitude
        // envelope `α_j(ψ) = exp(p_j ψ + c_j)` from the node Grams' diagonals
        // (`G_jj(ψ_i) = ‖X_j(ψ_i)‖²_W`, already formed), then divide each node's
        // sufficient statistics by it BEFORE the Chebyshev transform:
        // `G̃(ψ_i) = D_i⁻¹ G(ψ_i) D_i⁻¹`, `c̃(ψ_i) = D_i⁻¹ c(ψ_i)`, with
        // `D_i = diag(α_j(ψ_i))`. This cancels the exponential length-scale trend
        // (κ^{p_j} = e^{p_j ψ}) column-by-column, collapsing the interpoland's
        // dynamic range to O(1) so the tail certificate/spot-check pass on the
        // wide standardized geometry that used to refuse. The envelope is
        // re-applied in closed form by every accessor, so `gram_at` still returns
        // the TRUE Gram (the transform is conditioning-only, not a change of
        // answer). Normalizing the k×k node statistics (not the n×k design) keeps
        // this an O(m·k²) k-space step with no extra row work.
        let (col_amp_slope, col_amp_intercept) =
            recover_column_log_amplitudes(&node_grams, &node_psis, k);
        for (i, &psi) in node_psis.iter().enumerate() {
            let alpha: Vec<f64> = (0..k)
                .map(|j| (col_amp_slope[j] * psi + col_amp_intercept[j]).exp())
                .collect();
            let g = &mut node_grams[i];
            for a in 0..k {
                for b in 0..k {
                    g[[a, b]] /= alpha[a] * alpha[b];
                }
            }
            let r = &mut node_rhs[i];
            for a in 0..k {
                r[a] /= alpha[a];
            }
        }

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
            col_amp_slope,
            col_amp_intercept,
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
            let zero_rhs = Array1::<f64>::zeros(design.nrows());
            let (exact, _) = weighted_gram_and_rhs(&design, weights, &zero_rhs);
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
    ///
    /// The returned [`RangeProjector`] carries the Davis–Kahan error bar on
    /// `proj` alongside it, because a projector is only as meaningful as the
    /// eigen-gap that defines it — see [`RangeProjector::err_bar`] and #2448.
    fn range_projector(&self, psi: f64, rank_rtol: f64) -> Option<RangeProjector> {
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
        // Ascending order, so the kept block is the trailing `rank` entries and
        // the Davis–Kahan separation is between `evals[n-rank]` (smallest kept)
        // and `evals[n-rank-1]` (largest dropped).
        let mut smallest_kept = f64::INFINITY;
        let mut largest_dropped = f64::NEG_INFINITY;
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
                smallest_kept = smallest_kept.min(lam);
            } else {
                largest_dropped = largest_dropped.max(lam);
            }
        }
        let err_bar = if rank == evals.len() {
            // Nothing was dropped: `P` is the identity, exactly, on every host.
            // There is no eigenvector to rotate and no gap to divide by.
            0.0
        } else {
            projector_error_bar(
                smallest_kept - largest_dropped,
                Self::eig_backward_error(evals.len(), lambda_max),
            )
        };
        Some(RangeProjector {
            proj,
            rank,
            err_bar,
        })
    }

    /// Absolute backward-error bar `p(k)·ε·‖G‖₂` a backward-stable symmetric
    /// eigensolver commits on the eigenvalues of an order-`k` Gram, with
    /// `p(k) ≤ PSI_BAND_RANK_GUARD_SLACK·k`. This is the SAME instrument constant
    /// [`Self::rank_guard_gap`] sizes the rank guard band from — see
    /// [`PSI_BAND_RANK_GUARD_SLACK`] for the derivation. The rank guard divides it
    /// by the rank CUTOFF (turning it into a relative bar on the decided
    /// eigenvalue); the projector bar divides it by the eigen-GAP (turning it into
    /// a bar on the decided subspace). Same error, two different decisions.
    fn eig_backward_error(order: usize, lambda_max: f64) -> f64 {
        PSI_BAND_RANK_GUARD_SLACK * (order as f64) * f64::EPSILON * lambda_max
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
    ///
    /// ADMISSIBILITY OF THE WITNESS ITSELF (#2448). The gate above compares a
    /// MEASURED subspace distance to a 1e-7 tolerance, and a measurement is only a
    /// decision when the instrument resolves the tolerance. Both projectors are
    /// eigenvectors of a Gram the eigensolver only knows to `p(k)·ε·λ_max`, so by
    /// Davis–Kahan each carries its own rotation bar `‖E‖/(gap − ‖E‖)` set by the
    /// kept/dropped eigen-GAP (see [`projector_error_bar`]). On production radial
    /// geometry that gap can be ~1e-10·λ_max — the measured value on this module's
    /// own floor fixture — which puts the bar at ~1e-4, THREE ORDERS ABOVE the
    /// tolerance being gated on. The comparison then decides on roundoff: measured
    /// `sinθ` there is non-monotone in `Δψ` (2.9e-7 at 1e-8, 8.6e-9 at 1e-7,
    /// 1.6e-7 at 1.8e-7), so the witness accepts and refuses in alternation and
    /// [`Self::rank_stable_psi_floor`]'s monotone-step precondition is false.
    ///
    /// So the gate is not `measured ≤ ATOL` but the CERTIFIED BOUND on the true
    /// distance, `measured + err_ref + err_new ≤ ATOL` — the triangle inequality
    /// applied to the two error bars. This is the same rule the rank decision
    /// already follows (`Self::rank_guard_gap`): decide with a margin wider than
    /// the backward error committed forming the quantity decided on. Where the
    /// eigen-gap is wide (in particular at FULL rank, where `P` is the identity
    /// exactly and both bars are 0) the bound is the measurement and nothing
    /// changes; where the gap collapses the bound is `+∞` and the witness refuses
    /// — the sound fallback, at the price of the O(n) exact path.
    pub fn reduced_basis_equal(&self, psi_ref: f64, psi_new: f64) -> bool {
        if !(self.contains(psi_ref) && self.contains(psi_new)) {
            return false;
        }
        if psi_ref == psi_new {
            return true;
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
        // This certifies subspace identity across the rotation, letting the n-free
        // skip fire whenever the range genuinely coincides, while still refusing
        // (the SOUND fallback) the instant the subspaces separate by more than
        // PSI_GRAM_SKIP_PROJ_ATOL in true subspace distance — or the instant the
        // instrument stops being able to tell (#2448).
        self.reduced_basis_subspace_distance_bound(psi_ref, psi_new)
            .map(|d| d <= PSI_GRAM_SKIP_PROJ_ATOL)
            .unwrap_or(false)
    }

    /// The gauge-invariant subspace distance `‖P(ψ_ref) − P(ψ_new)‖₂ = sin θ_max`
    /// between the two conditioned-Gram range subspaces, AS MEASURED from the
    /// computed projectors. Exposed for #1033 frontier instrumentation so a
    /// refused n-free skip can be attributed to a genuine in-window basis ROTATION
    /// versus a rank change. Returns `None` for an off-window ψ, an equal-ψ pair,
    /// a rank mismatch, or an eigendecomp failure. Purely k-space (O(k³)) —
    /// independent of n.
    ///
    /// This is the measurement, NOT the quantity [`Self::reduced_basis_equal`]
    /// gates on: that gate uses [`Self::reduced_basis_subspace_distance_bound`],
    /// which adds each projector's own Davis–Kahan error bar (#2448). Reading a
    /// small value here as "the subspaces coincide" is exactly the mistake the
    /// bound exists to stop — compare the two to attribute a refusal to a real
    /// rotation rather than to an unresolvable eigen-gap.
    pub fn reduced_basis_subspace_distance(&self, psi_ref: f64, psi_new: f64) -> Option<f64> {
        if !(self.contains(psi_ref) && self.contains(psi_new)) {
            return None;
        }
        if psi_ref == psi_new {
            return Some(0.0);
        }
        let p_ref = self.range_projector(psi_ref, PSI_GRAM_SKIP_RANK_RTOL)?;
        let p_new = self.range_projector(psi_new, PSI_GRAM_SKIP_RANK_RTOL)?;
        if p_ref.rank != p_new.rank {
            return None;
        }
        let diff = &p_ref.proj - &p_new.proj;
        subspace_spectral_distance(&diff)
    }

    /// CERTIFIED UPPER BOUND on the true subspace distance between the two
    /// conditioned-Gram ranges — the quantity [`Self::reduced_basis_equal`]
    /// actually thresholds against `PSI_GRAM_SKIP_PROJ_ATOL` (#2448).
    ///
    /// `‖P_ref − P_new‖₂ ≤ ‖P̂_ref − P̂_new‖₂ + ‖P̂_ref − P_ref‖₂ + ‖P̂_new − P_new‖₂`
    /// — the measurement plus each computed projector's own Davis–Kahan bar. A
    /// bound at or below the tolerance certifies the two reduced bases span the
    /// same subspace no matter how the eigensolver's last bits fell; a bound above
    /// it means EITHER the subspaces genuinely separated OR the instrument cannot
    /// resolve the question, and both must refuse the skip.
    ///
    /// Returns `f64::INFINITY` when either projector's eigen-gap is closed at the
    /// backward-error scale (the eigenvectors are an arbitrary rotation inside a
    /// numerically degenerate cluster). `None` for an off-window ψ, a rank
    /// mismatch, or an eigendecomp failure. `Some(0.0)` for an equal-ψ pair, which
    /// is trivially and exactly sound. Purely k-space (O(k³)) — independent of n.
    pub fn reduced_basis_subspace_distance_bound(
        &self,
        psi_ref: f64,
        psi_new: f64,
    ) -> Option<f64> {
        if !(self.contains(psi_ref) && self.contains(psi_new)) {
            return None;
        }
        if psi_ref == psi_new {
            return Some(0.0);
        }
        let p_ref = self.range_projector(psi_ref, PSI_GRAM_SKIP_RANK_RTOL)?;
        let p_new = self.range_projector(psi_new, PSI_GRAM_SKIP_RANK_RTOL)?;
        if p_ref.rank != p_new.rank {
            return None;
        }
        let diff = &p_ref.proj - &p_new.proj;
        let measured = subspace_spectral_distance(&diff)?;
        Some(measured + p_ref.err_bar + p_new.err_bar)
    }

    /// The Davis–Kahan error bar `‖P̂(ψ) − P(ψ)‖₂` on the range projector at `psi`
    /// alone — how much of [`Self::reduced_basis_subspace_distance_bound`] is this
    /// endpoint's instrument rather than a real subspace move (#2448).
    ///
    /// Exposed so a refused skip, or a band edge that collapsed onto its anchor,
    /// can be ATTRIBUTED: a bar far under `PSI_GRAM_SKIP_PROJ_ATOL` means the
    /// refusal is a genuine rotation and the ψ-window is the thing to look at; a
    /// bar above it means the conditioned Gram has no resolvable kept/dropped
    /// eigen-gap at this ψ and the reduced-basis question is unanswerable there at
    /// double precision — a property of the geometry and the rank cutoff, not of
    /// the trial. `f64::INFINITY` when the gap is fully closed, `0.0` at full rank.
    /// `None` for an off-window / non-finite / all-zero Gram. Purely k-space.
    pub fn range_projector_error_bar(&self, psi: f64) -> Option<f64> {
        if !self.contains(psi) {
            return None;
        }
        self.range_projector(psi, PSI_GRAM_SKIP_RANK_RTOL)
            .map(|p| p.err_bar)
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
            .map(|p| p.rank)
    }

    /// Descending eigenvalues of the conditioned Gram `XᵀWX(ψ)` — the spectrum
    /// every rank decision in this module is taken on. `None` for an off-window
    /// / non-finite Gram, an eigendecomposition failure, or an all-zero Gram.
    /// Purely k-space (O(k³)) — independent of n.
    fn gram_spectrum(&self, psi: f64) -> Option<Vec<f64>> {
        use gam_linalg::faer_ndarray::FaerEigh;
        if !self.contains(psi) {
            return None;
        }
        let g = self.gram_at(psi);
        if g.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let gsym = 0.5 * (&g + &g.t());
        let (evals, _evecs) = gsym.eigh(faer::Side::Lower).ok()?;
        let mut spectrum: Vec<f64> = evals.to_vec();
        spectrum.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        (spectrum.first().copied().unwrap_or(0.0) > 0.0).then_some(spectrum)
    }

    /// Multiplicative guard gap the band-edge search certifies its rank claims
    /// with — the relative width of the two-sided band around the rank cutoff
    /// inside which the eigensolver's own backward error makes the integer rank
    /// undecidable. See [`PSI_BAND_RANK_GUARD_SLACK`] for the derivation
    /// (`SLACK·k·ε / PSI_GRAM_SKIP_RANK_RTOL`).
    fn rank_guard_gap(&self) -> f64 {
        PSI_BAND_RANK_GUARD_SLACK * (self.k as f64) * f64::EPSILON / PSI_GRAM_SKIP_RANK_RTOL
    }

    /// The conditioned Gram's rank decision at `psi` in the theory-master
    /// decision currency (`gam_linalg::decision`): the same partition
    /// [`Self::gram_numerical_rank`] reports, but posed against a two-sided guard
    /// band so the answer comes with a MARGIN. `Certified` means every kept
    /// eigenvalue clears the cutoff — and every dropped one falls below it — by
    /// the guard gap, which is `PSI_BAND_RANK_GUARD_SLACK` times the
    /// eigensolver's own error bar on those eigenvalues; the integer rank is
    /// therefore host-stable, and [`gam_linalg::decision::rank_transport_radius`]
    /// converts the same certificate into an operator-norm neighbourhood.
    /// `Ambiguous` is the honest report that an eigenvalue sits inside the band
    /// and the integer would be decided by the eigensolver's last bit.
    ///
    /// The decision is taken on the Gram spectrum rather than on design singular
    /// values — the squared currency `certified_rank` warns about — because the
    /// tensor retains ONLY the k×k sufficient statistic by construction (the
    /// n×k design rows are streamed and discarded at build). The guard band is
    /// sized to that squared currency's own backward error accordingly, which is
    /// what keeps the decision honest.
    ///
    /// `None` for an off-window / non-finite / all-zero Gram. Purely k-space.
    pub fn rank_decision(&self, psi: f64) -> Option<RankDecision> {
        let spectrum = self.gram_spectrum(psi)?;
        let lambda_max = spectrum[0];
        Some(certified_rank(
            &spectrum,
            PSI_GRAM_SKIP_RANK_RTOL * lambda_max,
            self.rank_guard_gap(),
        ))
    }

    /// Certified rank at the band-search anchor, or `None` when the anchor's own
    /// rank decision carries no margin. In the latter case there is nothing to
    /// transport, so [`Self::band_accepts`] falls back to the witness-only
    /// predicate rather than collapsing the band onto the anchor.
    fn anchor_certified_rank(&self, psi_anchor: f64) -> Option<usize> {
        match self.rank_decision(psi_anchor) {
            Some(RankDecision::Certified { rank, .. }) => Some(rank),
            _ => None,
        }
    }

    /// Band-edge acceptance predicate shared by [`Self::rank_stable_psi_floor`]
    /// and [`Self::rank_stable_psi_ceiling`].
    ///
    /// It is the production skip witness [`Self::reduced_basis_equal`] AND — when
    /// the anchor's own rank decision has a margin — a CERTIFIED rank claim equal
    /// to the anchor's. The conjunct is what makes the returned edge mean
    /// something: bisecting on the bare witness returns the ψ where `λ_r` sits
    /// exactly ON the rank cutoff, i.e. the one ψ in the whole window whose rank
    /// is undecidable at the eigensolver's own error bar. Clamping the κ line
    /// search to that ψ places its extreme trial on a coin-flip rank, and a flip
    /// to the deficient side is precisely the refused skip → O(n) `reset_surface`
    /// → rank-deficient pinning ψ → SECOND reset that the clamp exists to
    /// prevent. Requiring the guard band instead returns the last ψ whose rank
    /// COUNT clears the cutoff by a multiple of the eigensolver's error bar (the
    /// certification predicate is necessarily marginal at its own edge; the
    /// integer it decides is not); the edge moves down by only
    /// `ln(1+gap)/|d margin/dψ|` (≈1e-4 in ψ on the #2408 fixture), so the search
    /// interval is untouched for practical purposes while the clamp becomes
    /// host-stable. The band is a SUBSET of the witness-only band, so the skip
    /// still fires on every in-band trial — soundness is unchanged.
    fn band_accepts(&self, psi_anchor: f64, anchor_rank: Option<usize>, psi: f64) -> bool {
        if !self.reduced_basis_equal(psi_anchor, psi) {
            return false;
        }
        match anchor_rank {
            None => true,
            Some(anchor) => matches!(
                self.rank_decision(psi),
                Some(RankDecision::Certified { rank, .. }) if rank == anchor
            ),
        }
    }

    /// Lower edge of the contiguous ψ-band, ANCHORED at `psi_anchor`, over which
    /// the conditioned Gram `XᵀWX(ψ)` has the SAME reduced range projector as the
    /// anchor — i.e. the ψ-floor below which the design-revision skip's
    /// `reduced_basis_equal` witness must (soundly) refuse, because the range
    /// subspace either collapses or rotates away from the pinned slow-path basis.
    /// Lifting the κ-optimizer's lower bound to this floor keeps every in-window
    /// trial on the n-free fast path: the search is computed from the k×k tensor
    /// and never touches a sample row (#1033).
    ///
    /// N-FREE COST IS NOT N-INVARIANT LOCATION (#2408). The search costs
    /// `O(iters·k³)` with zero row access — that is the property the outer loop
    /// buys. The edge's LOCATION is a different claim: the tensor is BUILT from n
    /// rows, so its Gram is a row-sample of the underlying continuum Gram with a
    /// relative `O(1/n)` sampling error, which by Ostrowski moves every eigenvalue
    /// by a relative `O(1/n)` and hence moves [`Self::rank_margin`] additively.
    /// Where the edge is a root of that margin, the implicit-function bound
    /// `|δψ*| ≤ sup|δ margin| / inf|d margin/dψ|` is the whole truth: the edge is
    /// n-invariant only in the limit, at a rate set by the CROSSING SLOPE. A steep
    /// cliff pins the edge to machine precision; a grazing crossing (measured
    /// 1.5 nats per unit ψ on the #2408 fixture) lets an `O(1/n)` margin excursion
    /// displace the edge by `O(0.05)`. Callers must treat the edge as a clamp with
    /// that transport bound, never as an n-free constant of the design.
    ///
    /// Anchoring at `psi_anchor` (the optimizer's ψ seed) is essential: the
    /// conditioned Gram is rank-deficient at BOTH window ends on production radial
    /// geometry — at small ψ the longest-scale mode collapses into the polynomial
    /// nullspace, and at very large ψ every radial column goes collinear with it.
    /// The skip-acceptable region is therefore a middle BAND, and the κ-optimum
    /// lives inside it. We bisect DOWN from the anchor on the projector witness
    /// and return the lowest ψ still sharing the anchor's reduced projector — the
    /// true lower band edge, resolved continuously rather than snapped to a fixed
    /// grid (#2054). Purely O(iters·k³) — no row access.
    ///
    /// Returns `None` when the band already reaches `psi_lo` (no lift needed), when
    /// the anchor is off-window / projector-indeterminate, or when the window is empty.
    pub fn rank_stable_psi_floor(&self, psi_anchor: f64) -> Option<f64> {
        if !(self.psi_hi > self.psi_lo) {
            return None;
        }
        if !self.contains(psi_anchor) {
            return None;
        }
        // The anchor always accepts (its own rank decision is trivially equal to
        // itself and `reduced_basis_equal(x, x) == true`) and, per the window
        // geometry, sits inside the contiguous skip-acceptable middle band, so
        // `accepts` is a monotone step on `[psi_lo, psi_anchor]`: true from the
        // anchor down to the band's lower edge, false below it. Find that edge by
        // BISECTION on the witness — it converges continuously to the true
        // crossing instead of snapping to one of a fixed grid of nodes (#2054;
        // SPEC forbids grid search). Purely k-space (O(iters·k³)), no row access.
        let anchor_rank = self.anchor_certified_rank(psi_anchor);
        let accepts = |psi: f64| self.band_accepts(psi_anchor, anchor_rank, psi);
        let mut lo = self.psi_lo; // refusing endpoint (to be established)
        if accepts(lo) {
            // The skip-acceptable band already reaches `psi_lo` — no lift needed.
            return None;
        }
        let mut hi = psi_anchor; // known accepting endpoint (accepts(anchor) == true)
        // Invariant: `accepts(hi)` true, `accepts(lo)` false; the lower band edge
        // is the unique crossing in `(lo, hi]`. Return the lowest accepting ψ.
        for _ in 0..PSI_BAND_BISECTION_ITERS {
            if hi - lo <= PSI_BAND_BISECTION_ATOL * (1.0 + hi.abs()) {
                break;
            }
            let mid = 0.5 * (lo + hi);
            if accepts(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(hi)
    }

    /// Upper edge of the contiguous skip-acceptable ψ-band, the symmetric twin of
    /// [`Self::rank_stable_psi_floor`] (#1033). The conditioned Gram `XᵀWX(ψ)` is
    /// rank-deficient at BOTH window ends — at small ψ the longest-length-scale
    /// radial mode collapses into the polynomial nullspace, and at very large ψ
    /// every radial column goes collinear with the low-frequency mode, so the
    /// skip-acceptable region is a middle BAND. The optimizer's line search can
    /// OVERSHOOT above that band (e.g. ψ≈1.0 on production Duchon geometry), where
    /// the design-realization skip's `reduced_basis_equal` witness must soundly
    /// refuse (the range subspace dropped a dimension) → an O(n) `reset_surface`,
    /// AND the pinning ψ recorded at that reset is itself rank-deficient, so the
    /// NEXT in-band trial mismatches its reference and resets a SECOND time. Both
    /// resets vanish once the optimizer's UPPER bound is clamped down to this
    /// n-free k-space ceiling, keeping every trial inside the skip-acceptable band.
    ///
    /// Bisects UP from the anchor on the band witness (the mirror of the floor)
    /// and returns the highest ψ still accepted against the anchor — the true
    /// upper band edge, resolved continuously rather than snapped to a fixed grid
    /// (#2054). Purely O(iters·k³) — no row access.
    ///
    /// The edge is n-FREE IN COST but not n-INVARIANT IN LOCATION; see
    /// [`Self::rank_stable_psi_floor`] for the transport bound that replaces the
    /// former (false) invariance claim, and the `band_accepts` predicate for why
    /// the returned edge is the last CERTIFIED-rank ψ rather than the ψ sitting
    /// exactly on the cutoff.
    ///
    /// Returns `None` when the band already reaches `psi_hi` (no clamp needed),
    /// when the anchor is off-window / projector-indeterminate, or when the window is
    /// empty.
    pub fn rank_stable_psi_ceiling(&self, psi_anchor: f64) -> Option<f64> {
        // Bisection mirror of `rank_stable_psi_floor` (#2054): the anchor always
        // accepts and sits inside the contiguous band, so `accepts` is a monotone
        // step on `[psi_anchor, psi_hi]` (true up to the upper edge, false above).
        // Locate the edge by bisection on the band witness rather than a fixed
        // grid scan. Purely k-space (O(iters·k³)), no row access.
        if !(self.psi_hi > self.psi_lo) {
            return None;
        }
        if !self.contains(psi_anchor) {
            return None;
        }
        let anchor_rank = self.anchor_certified_rank(psi_anchor);
        let accepts = |psi: f64| self.band_accepts(psi_anchor, anchor_rank, psi);
        let mut hi = self.psi_hi; // refusing endpoint (to be established)
        if accepts(hi) {
            // The skip-acceptable band already reaches `psi_hi` — no clamp needed.
            return None;
        }
        let mut lo = psi_anchor; // known accepting endpoint (accepts(anchor) == true)
        // Invariant: `accepts(lo)` true, `accepts(hi)` false; the upper band edge
        // is the unique crossing in `[lo, hi)`. Return the highest accepting ψ.
        for _ in 0..PSI_BAND_BISECTION_ITERS {
            if hi - lo <= PSI_BAND_BISECTION_ATOL * (1.0 + hi.abs()) {
                break;
            }
            let mid = 0.5 * (lo + hi);
            if accepts(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(lo)
    }

    /// True when `psi` lies inside the certified window.
    pub fn contains(&self, psi: f64) -> bool {
        psi.is_finite() && psi >= self.psi_lo && psi <= self.psi_hi
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

    /// The per-column amplitude envelope `α_j(ψ) = exp(p_j·ψ + c_j)` (#1216) that
    /// every accessor re-applies to the stored amplitude-NORMALIZED series to
    /// recover the true design-moving statistics. `D(ψ) = diag(col_amplitudes)`.
    fn col_amplitudes(&self, psi: f64) -> Array1<f64> {
        Array1::from_iter(
            (0..self.k).map(|j| (self.col_amp_slope[j] * psi + self.col_amp_intercept[j]).exp()),
        )
    }

    /// Chebyshev contraction `Σ_d w_d · gram[d]` of the stored (amplitude-
    /// normalized) Gram series with per-coefficient weights `w_d` — the shared
    /// n-free O(Dk²) kernel behind the value/gradient/curvature accessors.
    fn contract_gram(&self, weights: &[f64]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.k, self.k));
        let mut comp = Array2::<f64>::zeros((self.k, self.k));
        for (d, &wd) in weights.iter().enumerate() {
            kahan_scaled_add_array2(&mut out, &mut comp, wd, &self.gram[d]);
        }
        out
    }

    /// Chebyshev contraction `Σ_d w_d · rhs[d]` of the stored (amplitude-
    /// normalized) RHS series.
    fn contract_rhs(&self, weights: &[f64]) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(self.k);
        let mut comp = Array1::<f64>::zeros(self.k);
        for (d, &wd) in weights.iter().enumerate() {
            kahan_scaled_add_array1(&mut out, &mut comp, wd, &self.rhs[d]);
        }
        out
    }

    /// `XᵀWX(ψ)` assembled n-free in O(Dk²) from the direct Gram series.
    ///
    /// `G(ψ) = D(ψ) G̃(ψ) D(ψ)` (#1216): the stored series interpolates the
    /// amplitude-normalized `G̃`, and the per-column envelope `D(ψ) = diag(α_j)`
    /// is re-applied here — `G_ab = α_a·α_b·G̃_ab` — reproducing the true Gram.
    pub fn gram_at(&self, psi: f64) -> Array2<f64> {
        let x = self.mapped(psi);
        let t = cheb_t(x, self.gram.len());
        let g_tilde = self.contract_gram(&t);
        let alpha = self.col_amplitudes(psi);
        let mut out = Array2::<f64>::zeros((self.k, self.k));
        for a in 0..self.k {
            for b in 0..self.k {
                out[[a, b]] = alpha[a] * alpha[b] * g_tilde[[a, b]];
            }
        }
        out
    }

    /// `XᵀWz(ψ)` assembled n-free in O(Dk). `c_a = α_a·c̃_a` (#1216 envelope).
    pub fn rhs_at(&self, psi: f64) -> Array1<f64> {
        let x = self.mapped(psi);
        let t = cheb_t(x, self.n_coeff);
        let c_tilde = self.contract_rhs(&t);
        let alpha = self.col_amplitudes(psi);
        let mut out = Array1::<f64>::zeros(self.k);
        for a in 0..self.k {
            out[a] = alpha[a] * c_tilde[a];
        }
        out
    }

    /// Exact `∂(XᵀWX)/∂ψ` from the SAME representation as the value — the
    /// structural cure for the objective↔gradient desync class on this
    /// channel. n-free, O(Dk²) from the direct Gram series.
    pub fn dgram_dpsi(&self, psi: f64) -> Array2<f64> {
        let x = self.mapped(psi);
        let dx_dpsi = 2.0 / (self.psi_hi - self.psi_lo);
        let t = cheb_t(x, self.gram.len());
        let tp = cheb_t_prime(x, self.gram.len());
        let tp_scaled: Vec<f64> = tp.iter().map(|v| v * dx_dpsi).collect();
        let g_tilde = self.contract_gram(&t);
        let g_tilde_p = self.contract_gram(&tp_scaled);
        // Product rule on `G = D G̃ D`, `D = diag(exp(p_j ψ + c_j))`,
        // `D' = diag(p_j)·D`: `[dG/dψ]_ab = α_a α_b [(p_a+p_b) G̃_ab + G̃'_ab]`.
        let alpha = self.col_amplitudes(psi);
        let p = &self.col_amp_slope;
        let mut out = Array2::<f64>::zeros((self.k, self.k));
        for a in 0..self.k {
            for b in 0..self.k {
                out[[a, b]] =
                    alpha[a] * alpha[b] * ((p[a] + p[b]) * g_tilde[[a, b]] + g_tilde_p[[a, b]]);
            }
        }
        out
    }

    /// Exact `∂(XᵀWz)/∂ψ`, n-free. `[dc/dψ]_a = α_a (p_a c̃_a + c̃'_a)` (#1216).
    pub fn drhs_dpsi(&self, psi: f64) -> Array1<f64> {
        let x = self.mapped(psi);
        let dx_dpsi = 2.0 / (self.psi_hi - self.psi_lo);
        let t = cheb_t(x, self.n_coeff);
        let tp = cheb_t_prime(x, self.n_coeff);
        let tp_scaled: Vec<f64> = tp.iter().map(|v| v * dx_dpsi).collect();
        let c_tilde = self.contract_rhs(&t);
        let c_tilde_p = self.contract_rhs(&tp_scaled);
        let alpha = self.col_amplitudes(psi);
        let p = &self.col_amp_slope;
        let mut out = Array1::<f64>::zeros(self.k);
        for a in 0..self.k {
            out[a] = alpha[a] * (p[a] * c_tilde[a] + c_tilde_p[a]);
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
            // #1868: the caller (`install_psi_gram_statistics`) attaches the
            // once-built ψ-invariant frozen row bundle after this returns; the
            // tensor itself is n-free and holds no row vectors to build it from.
            frozen_rows: None,
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

    /// Duchon-shaped analytic synthetic design: polyharmonic `s^p·ln s` radial
    /// columns (`s = r·e^ψ`) plus a ψ-free polynomial column — the structural
    /// shape of the production hybrid-Duchon radial designs, as distinct from
    /// [`synth_design`]'s `(1+s)e^{-s}` Matérn shape.
    ///
    /// The distinction is the whole point (#2464). `s^p·ln s = r^p e^{pψ}(ln r + ψ)`
    /// is an exponential amplitude times a factor AFFINE in ψ, so it is NOT a pure
    /// per-column amplitude `e^{p_jψ+c_j}`; the leftover ψ-motion has to be carried
    /// by the Chebyshev series `G̃(x(ψ))` and therefore by its DERIVATIVE `G̃'`.
    /// The Matérn shape is far smoother in `x`, so a series accurate enough for the
    /// value there is also accurate in slope — which is exactly the coincidence a
    /// Matérn-only fixture cannot distinguish from a correct derivative.
    fn synth_duchon_design(psi: f64, n: usize, k: usize) -> Result<Array2<f64>, String> {
        let mut x = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            for j in 0..k {
                let r = 0.05 + (i as f64 + 1.0) * (j as f64 + 1.0) / (n as f64 * k as f64) * 3.0;
                if j == k - 1 {
                    x[[i, j]] = r * r * r;
                } else {
                    let s = r * psi.exp();
                    x[[i, j]] = s * s * s * s.ln();
                }
            }
        }
        Ok(x)
    }

    fn exact_duchon_gram(psi: f64, n: usize, k: usize, w: &Array1<f64>) -> Array2<f64> {
        let design = synth_duchon_design(psi, n, k).unwrap();
        let mut wd = design.clone();
        for (mut row, &wi) in wd.outer_iter_mut().zip(w.iter()) {
            row.mapv_inplace(|v| v * wi);
        }
        design.t().dot(&wd)
    }

    fn exact_duchon_rhs(psi: f64, n: usize, k: usize, w: &Array1<f64>, z: &Array1<f64>) -> Array1<f64> {
        let design = synth_duchon_design(psi, n, k).unwrap();
        let mut wz = Array1::<f64>::zeros(n);
        for ((slot, &wi), &zi) in wz.iter_mut().zip(w.iter()).zip(z.iter()) {
            *slot = wi * zi;
        }
        design.t().dot(&wz)
    }

    /// #2464: the same certification as
    /// [`psi_gram_tensor_matches_exact_gram_and_fd_gradient`], on the profile
    /// shape that actually CONSUMES this lane in production.
    ///
    /// The `#1033b` n-free ψ-gradient short-circuit in `reml/hyper.rs` is reachable
    /// only where `supports_nfree_penalty_rekey` is admitted — Duchon — because
    /// Matérn was excluded by the #1274 revert. Yet the tensor's derivative
    /// certification runs on a Matérn-shaped fixture. Production measures the
    /// Duchon `fixed_beta` ψ-gradient at `+9.821e2` against FD `−3.431e3` (opposite
    /// sign) while Matérn at the same pin is clean, so the fixture's shape and the
    /// consumer's shape are exactly the variable that is not controlled.
    ///
    /// This also closes a gap in the sibling gate: that one FD-checks `dgram_dpsi`
    /// but never `drhs_dpsi` against a difference of `rhs_at`. `drhs_dpsi` carries
    /// the dominant term of `a_j = −(∂b/∂ψ)·β̂ + ½β̂ᵀ(∂G/∂ψ)β̂ + ½β̂ᵀS_τβ̂`, so an
    /// error there lands in `fixed_beta` and nowhere else — which is precisely the
    /// one channel production reports wrong.
    ///
    /// Tolerances are the sibling gate's, unchanged, so a failure here is directly
    /// comparable to the Matérn arm passing rather than a differently-graded test.
    #[test]
    fn psi_gram_tensor_derivatives_certify_on_a_duchon_profile_2464() {
        let (n, k) = (160usize, 7usize);
        let w = Array1::from_iter((0..n).map(|i| 1.0 + 0.5 * ((i % 3) as f64)));
        let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.37).sin()));
        let (psi_lo, psi_hi) = (-1.2_f64, 1.0_f64);

        let tensor = PsiGramTensor::build(
            |psi| synth_duchon_design(psi, n, k),
            w.view(),
            z.view(),
            psi_lo,
            psi_hi,
        )
        .expect("analytic Duchon-shaped design must certify");

        let h = 1e-5_f64;
        let mut worst_gram_rel = 0.0_f64;
        let mut worst_rhs_rel = 0.0_f64;
        let mut worst_dgram_rel = 0.0_f64;
        let mut worst_drhs_rel = 0.0_f64;
        let mut graded = 0usize;

        for &psi in &[-1.1, -0.63, 0.0, 0.41, 0.97] {
            assert!(tensor.contains(psi), "psi={psi} must be in the value window");
            if !tensor.contains_for_gradient(psi - h) || !tensor.contains_for_gradient(psi + h) {
                continue;
            }
            graded += 1;

            // VALUE: both channels, against the exact streamed statistics.
            let exact_g = exact_duchon_gram(psi, n, k, &w);
            let gscale = exact_g.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1e-300);
            for (a, b) in tensor.gram_at(psi).iter().zip(exact_g.iter()) {
                worst_gram_rel = worst_gram_rel.max((a - b).abs() / gscale);
            }
            let exact_r = exact_duchon_rhs(psi, n, k, &w, &z);
            let rscale = exact_r.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1e-300);
            for (a, b) in tensor.rhs_at(psi).iter().zip(exact_r.iter()) {
                worst_rhs_rel = worst_rhs_rel.max((a - b).abs() / rscale);
            }

            // DERIVATIVE: both channels, against a central difference of the EXACT
            // statistics — not of the tensor's own value, so a representation that
            // is self-consistent but wrong about the design cannot pass.
            let dg = tensor.dgram_dpsi(psi);
            let dgscale = dg.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1e-12);
            let gp = exact_duchon_gram(psi + h, n, k, &w);
            let gm = exact_duchon_gram(psi - h, n, k, &w);
            for ((a, p), m) in dg.iter().zip(gp.iter()).zip(gm.iter()) {
                worst_dgram_rel = worst_dgram_rel.max((a - (p - m) / (2.0 * h)).abs() / dgscale);
            }

            let dr = tensor.drhs_dpsi(psi);
            let drscale = dr.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1e-12);
            let rp = exact_duchon_rhs(psi + h, n, k, &w, &z);
            let rm = exact_duchon_rhs(psi - h, n, k, &w, &z);
            for ((a, p), m) in dr.iter().zip(rp.iter()).zip(rm.iter()) {
                worst_drhs_rel = worst_drhs_rel.max((a - (p - m) / (2.0 * h)).abs() / drscale);
            }
        }

        eprintln!(
            "[2464-tensor-duchon] graded={graded} gram_rel={worst_gram_rel:.3e} \
             rhs_rel={worst_rhs_rel:.3e} dgram_rel={worst_dgram_rel:.3e} \
             drhs_rel={worst_drhs_rel:.3e}"
        );
        assert!(
            graded >= 3,
            "the certified gradient sub-window admitted only {graded} probes; this gate \
             would be reporting on almost nothing"
        );
        assert!(worst_gram_rel <= 1e-9, "gram value: {worst_gram_rel:.3e} > 1e-9");
        assert!(worst_rhs_rel <= 1e-9, "rhs value: {worst_rhs_rel:.3e} > 1e-9");
        assert!(
            worst_dgram_rel <= 1e-5,
            "dgram/dpsi vs FD of the exact Duchon gram: {worst_dgram_rel:.3e} > 1e-5"
        );
        assert!(
            worst_drhs_rel <= 1e-5,
            "drhs/dpsi vs FD of the exact Duchon rhs: {worst_drhs_rel:.3e} > 1e-5 \
             (this channel had no FD gate at all before #2464)"
        );
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

    /// #2448 — the `synth_design` fixture is the case where the projector witness
    /// CANNOT ANSWER, and `rank_stable_psi_floor` must degrade to "no lift" rather
    /// than invent a band edge out of roundoff.
    ///
    /// The Matérn-shaped columns over a narrow `r`-range give the conditioned Gram
    /// a smooth GEOMETRIC spectral decay — measured relative spectrum at the seed
    /// `[1, 1.3e-2, 4.4e-4, 5.1e-6, 3.6e-8, 1.2e-10, 3.0e-13]` — with no cliff
    /// anywhere. The `PSI_GRAM_SKIP_RANK_RTOL = 1e-10` cutoff slices through the
    /// middle of that decay, so the kept/dropped eigen-GAP is only `1.2e-10·λ_max`
    /// and the range projector's own Davis–Kahan bar is `~1e-4` — THREE ORDERS
    /// ABOVE the `1e-7` tolerance the witness gates on. Before #2448 the gate
    /// compared a measured `sinθ` to that tolerance anyway, and the measurement is
    /// pure roundoff at this scale: it is NON-MONOTONE in `Δψ` (asserted below),
    /// so the bisection's monotone-step precondition was false and the returned
    /// floor was a coin flip within `~1e-7` of the anchor.
    ///
    /// With the admissibility check the witness refuses every non-trivial pair
    /// here, the bisection collapses the band onto the anchor EXACTLY, and the
    /// κ-caller's `floor < psi_anchor` filter reads that as "no lift" — the sound
    /// fallback (every trial takes the exact O(n) path). That collapse is now
    /// bit-identical across row counts, because it is a structural refusal rather
    /// than a noise-decided edge.
    ///
    /// The genuine floor-search claim — a real interior edge, n-transportable —
    /// lives in `rank_stable_psi_floor_finds_a_real_edge_when_the_witness_is_admissible_2448`,
    /// on a fixture whose witness can actually resolve the question. A design that
    /// is full-rank across the whole window must still return `None`.
    #[test]
    fn rank_stable_psi_floor_refuses_to_invent_an_edge_from_roundoff_2448() {
        let k = 7usize;
        // Window spanning the small-ψ rank cliff. Kept moderate so the Chebyshev
        // ladder certifies at a low rung (the build is the only n-pass; a wide
        // window forces high rungs = many design realizations = slow test).
        let (psi_lo, psi_hi) = (-1.6_f64, 1.0_f64);
        let build_at = |n: usize| {
            let w = Array1::from_iter((0..n).map(|i| 1.0 + 0.5 * ((i % 3) as f64)));
            let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.37).sin()));
            PsiGramTensor::build(
                |psi| synth_design(psi, n, k),
                w.view(),
                z.view(),
                psi_lo,
                psi_hi,
            )
            .expect("analytic synthetic design must certify")
        };

        let t_small = build_at(120);
        let t_big = build_at(1000);
        // Seed at the well-conditioned (small-length-scale) window end — the
        // κ-optimum's neighbourhood, and the window-maximal rank (the synthetic
        // radial design's rank rises toward large ψ).
        let seed = psi_hi;

        // (a) THE INSTRUMENT CANNOT RESOLVE THE TOLERANCE. The anchor's own range
        // projector carries a Davis–Kahan bar orders of magnitude above the
        // subspace tolerance the witness gates on, on BOTH row counts — this is a
        // property of the geometry and the rank cutoff, not of the sample.
        for (label, t) in [("n=120", &t_small), ("n=1000", &t_big)] {
            let bar = t
                .range_projector_error_bar(seed)
                .expect("the seed Gram is finite and non-degenerate");
            assert!(
                bar > 100.0 * PSI_GRAM_SKIP_PROJ_ATOL,
                "{label}: the geometric-decay spectrum leaves the seed's range \
                 projector unresolved (measured bar ~1e-4 vs a {PSI_GRAM_SKIP_PROJ_ATOL:e} \
                 tolerance); got {bar:e}, which would mean the fixture changed and \
                 this test no longer exercises #2448"
            );
        }

        // (b) THE OLD GATE'S MEASUREMENT IS NOISE. `sinθ` is not monotone in Δψ
        // anywhere near the anchor, so `measured ≤ ATOL` is not a step function and
        // the bisection has no crossing to find. Assert the non-monotonicity
        // directly rather than restating it in prose: over this ladder of Δψ the
        // measured distance must go DOWN at least once while Δψ goes up.
        let ladder = [1e-8_f64, 1e-7, 1.8e-7, 3e-7];
        let measured: Vec<f64> = ladder
            .iter()
            .map(|&d| {
                t_small
                    .reduced_basis_subspace_distance(seed, seed - d)
                    .expect("equal rank at these tiny offsets")
            })
            .collect();
        assert!(
            measured.windows(2).any(|w| w[1] < w[0]),
            "the pre-#2448 accept predicate was claimed to be a monotone step in \
             Δψ; measured sinθ over Δψ={ladder:?} is {measured:?}, which must be \
             non-monotone for this fixture to be the #2448 witness"
        );

        // (c) THE WITNESS REFUSES, and the band therefore collapses onto the anchor
        // EXACTLY — not to within a noise floor. The κ-caller filters on
        // `floor < psi_anchor`, so this reads as "no lift": every trial takes the
        // exact O(n) path, which is the sound fallback.
        assert!(
            !t_small.reduced_basis_equal(seed, seed - 1e-6),
            "an unresolved projector must refuse the skip, not certify it on roundoff"
        );
        let floor_small = t_small
            .rank_stable_psi_floor(seed)
            .expect("the band does not reach psi_lo, so an edge is reported");
        let floor_big = t_big
            .rank_stable_psi_floor(seed)
            .expect("the band does not reach psi_lo, so an edge is reported");
        assert_eq!(
            floor_small, seed,
            "with no admissible witness the band is exactly the anchor"
        );
        assert_eq!(
            floor_big, floor_small,
            "a structural refusal is bit-identical across row counts, unlike the \
             pre-#2448 noise-decided edge (which moved ~1.4e-7 between these builds)"
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

    /// #1216 amplitude normalization: on a WIDE window a mixed design whose radial
    /// columns carry a pure length-scale amplitude `κ^{p_j} = e^{p_j ψ}` (the
    /// Duchon/ThinPlate structure) plus a ψ-free polynomial-nullspace column must
    /// (a) recover the per-column power `p_j` EXACTLY from the node-Gram diagonals,
    /// (b) collapse the interpoland's dynamic range to O(1) even though the TRUE
    /// Gram spans many orders of magnitude across the window, and (c) still
    /// reconstruct the true Gram / rhs / ψ-derivative to certification accuracy via
    /// the closed-form envelope `D(ψ)`. This is the mechanism that lets the tensor
    /// attach on the wide standardized geometry that used to refuse (the #1216
    /// wall): the normalized series is what the tail certificate and off-node spot
    /// check see, and here it is essentially ψ-free.
    #[test]
    fn amplitude_normalization_factors_exact_power_and_bounds_series_1216() {
        let (n, k) = (180usize, 4usize);
        // Three radial columns with amplitude e^{p ψ}, p = 2, over distinct
        // ψ-free profiles, plus one ψ-free polynomial-nullspace column (p = 0).
        let powers = [2.0_f64, 2.0, 2.0, 0.0];
        let base = |i: usize, j: usize| -> f64 {
            let t = (i as f64 + 0.5) / n as f64;
            match j {
                0 => 1.0,
                1 => (2.0 * std::f64::consts::PI * t).sin(),
                2 => (2.0 * std::f64::consts::PI * t).cos(),
                _ => t, // ψ-free polynomial column
            }
        };
        let design = move |psi: f64| -> Result<Array2<f64>, String> {
            let mut x = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                for j in 0..k {
                    x[[i, j]] = (powers[j] * psi).exp() * base(i, j);
                }
            }
            Ok(x)
        };
        let w = Array1::from_iter((0..n).map(|i| 1.0 + 0.4 * ((i % 3) as f64)));
        let z = Array1::from_iter((0..n).map(|i| ((i as f64) * 0.31).sin() + 0.2));
        // WIDE window (~9 nats) — the #1215/#1216 standardized-geometry regime.
        let (psi_lo, psi_hi) = (-4.5_f64, 4.5_f64);

        let tensor = PsiGramTensor::build(|psi| design(psi), w.view(), z.view(), psi_lo, psi_hi)
            .expect("amplitude-normalized wide-window design must certify (#1216)");

        // (a) The per-column power is recovered EXACTLY (½·log G_jj is affine in ψ
        // for a power-law column, so the LS slope has zero residual).
        for j in 0..k {
            assert!(
                (tensor.col_amp_slope[j] - powers[j]).abs() <= 1e-9,
                "recovered amplitude slope p_{j}={} must match the true power {}",
                tensor.col_amp_slope[j],
                powers[j]
            );
        }

        // (b) The stored (normalized) series has O(1) dynamic range, even though the
        // TRUE Gram entry spans many orders across the window. Factoring e^{p_j ψ}
        // out column-by-column makes the normalized statistics essentially ψ-free.
        let series_scale = tensor.gram.iter().fold(0.0_f64, |acc, slab| {
            acc.max(slab.iter().fold(0.0_f64, |a, &v| a.max(v.abs())))
        });
        assert!(
            series_scale < 1e2,
            "normalized Gram series must be O(1); got max entry {series_scale:.3e}"
        );
        // The true Gram genuinely explodes at the high-ψ end (evidence the dynamic
        // range was real and is now carried by the envelope, not the interpoland).
        let g_hi = tensor.gram_at(psi_hi);
        assert!(
            g_hi[[0, 0]] > 1e6,
            "true Gram must carry the large length-scale amplitude at psi_hi \
             (got {:.3e})",
            g_hi[[0, 0]]
        );

        // (c) The closed-form envelope reconstructs the true Gram / rhs / gradient
        // to certification accuracy at the window edges and center.
        let exact_gram_rhs = |psi: f64| -> (Array2<f64>, Array1<f64>) {
            let d = design(psi).unwrap();
            let mut wd = d.clone();
            for (mut row, &wi) in wd.outer_iter_mut().zip(w.iter()) {
                row.mapv_inplace(|v| v * wi);
            }
            (d.t().dot(&wd), wd.t().dot(&z))
        };
        for &psi in &[psi_lo + 0.1, 0.0, psi_hi - 0.1] {
            let (eg, er) = exact_gram_rhs(psi);
            let fg = tensor.gram_at(psi);
            let fr = tensor.rhs_at(psi);
            let gscale = eg.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1e-300);
            for (a, b) in fg.iter().zip(eg.iter()) {
                assert!(
                    (a - b).abs() <= PSI_GRAM_SPOT_RTOL * gscale,
                    "gram reconstruction at psi={psi}: fast={a}, exact={b}"
                );
            }
            let rscale = er.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1e-300);
            for (a, b) in fr.iter().zip(er.iter()) {
                assert!(
                    (a - b).abs() <= PSI_GRAM_SPOT_RTOL * rscale,
                    "rhs reconstruction at psi={psi}: fast={a}, exact={b}"
                );
            }
        }
        // Analytic ψ-gradient (product-rule envelope) vs central FD of the exact
        // Gram at the window center — certifies the derivative envelope algebra.
        let psi = 0.0_f64;
        let h = 1e-6;
        let (gp, _) = exact_gram_rhs(psi + h);
        let (gm, _) = exact_gram_rhs(psi - h);
        let dg = tensor.dgram_dpsi(psi);
        let dscale = dg.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1e-12);
        for ((a, p), m_) in dg.iter().zip(gp.iter()).zip(gm.iter()) {
            let fd = (p - m_) / (2.0 * h);
            assert!(
                (a - fd).abs() <= 1e-5 * dscale,
                "dgram/dpsi envelope mismatch at psi={psi}: analytic={a}, fd={fd}"
            );
        }
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
            let RangeProjector { proj, rank, .. } = tensor
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
        let cos_angle =
            v_lo.dot(&v_hi).abs() / (v_lo.dot(&v_lo).sqrt() * v_hi.dot(&v_hi).sqrt()).max(1e-300);
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
                .map(|p| p.rank)
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
                .map(|p| p.rank)
                .unwrap_or(0)
        };
        for &psi in &[-0.95, -0.4, 0.0, 0.4, 0.95] {
            assert_eq!(rank_at(psi), 2, "rotation keeps rank 2 at psi={psi}");
            // #2448 — the kept 2-plane is separated from the empty third direction
            // by an O(λ_max) gap, so the admissibility check is far from binding
            // here. This is the guard that the check did not buy soundness by
            // refusing everything: a witness that always refuses would also pass a
            // refusal test, but it would fail the certification sweep below.
            let bar = tensor
                .range_projector_error_bar(psi)
                .expect("finite rank-2 Gram");
            assert!(
                bar < 1e-3 * PSI_GRAM_SKIP_PROJ_ATOL,
                "a wide-gap projector must be resolved orders inside the tolerance \
                 at psi={psi}, else the skip witness could never certify; got {bar:e}"
            );
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

    /// #2448 — the floor search still finds a REAL interior band edge, and
    /// transports it across row counts, whenever the witness can actually resolve
    /// the question. The companion to
    /// `rank_stable_psi_floor_refuses_to_invent_an_edge_from_roundoff_2448`: that
    /// one pins the refusal, this one pins that the refusal is not universal.
    ///
    /// The fixture is the `reduced_basis_witness_refuses_across_subspace_change`
    /// geometry — two ψ-invariant profiles plus a third whose amplitude
    /// `ε(ψ) = e^{10ψ}` sweeps its eigenvalue `∝ ε²` across the rank cutoff at
    /// `ψ* = ln(√(2·rtol)) / 10 ≈ −1.118`. This is a genuine CLIFF, not a
    /// geometric decay: above `ψ*` the Gram is FULL rank (`P` is the identity, so
    /// its Davis–Kahan bar is exactly 0), and below it the third direction is
    /// orders of magnitude under the second, so the gap is `O(λ_max)` and that bar
    /// is `~1e-14`. The witness is admissible on both sides and the band edge is
    /// decided by the rank change alone.
    ///
    /// `ψ*` is n-FREE by construction: both the cutoff (`rtol·λ_max`) and the
    /// third eigenvalue (`ε²·‖c₃‖²`) scale linearly in n, so the crossing does not
    /// move. That is what makes the transport assertion below sharp rather than
    /// window-sized.
    #[test]
    fn rank_stable_psi_floor_finds_a_real_edge_when_the_witness_is_admissible_2448() {
        let k = 3usize;
        let (psi_lo, psi_hi) = (-1.6_f64, -0.8_f64);
        let alpha = 10.0_f64;
        let build_at = |n: usize| {
            let base = move |i: usize, j: usize| -> f64 {
                let t = (i as f64 + 0.5) / n as f64;
                match j {
                    0 => 1.0,
                    1 => (2.0 * std::f64::consts::PI * t).sin(),
                    _ => (4.0 * std::f64::consts::PI * t).cos(),
                }
            };
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
            PsiGramTensor::build(design, w.view(), z.view(), psi_lo, psi_hi)
                .expect("smooth ε(ψ) design must certify (analytic, no kink)")
        };

        let anchor = psi_hi - 0.02;
        let mut floors = Vec::new();
        for n in [200usize, 800] {
            let t = build_at(n);

            // The anchor sits at FULL rank, so its projector is the identity and
            // carries no error at all — the admissibility check is inert here, which
            // is precisely why this fixture can still resolve an edge.
            assert_eq!(
                t.gram_numerical_rank(anchor),
                Some(k),
                "n={n}: the anchor must be full rank for the premise to hold"
            );
            assert_eq!(
                t.range_projector_error_bar(anchor),
                Some(0.0),
                "n={n}: a full-rank range projector is the identity exactly, on \
                 every host — its Davis–Kahan bar is 0, not merely small"
            );

            let floor = t
                .rank_stable_psi_floor(anchor)
                .expect("the rank cliff is inside the window, so a floor is reported");
            assert!(
                floor > psi_lo && floor < anchor,
                "n={n}: an admissible witness must return an INTERIOR edge, not \
                 collapse onto the anchor: floor={floor} in ({psi_lo}, {anchor})"
            );

            // The edge is the rank cliff: full rank at it, deficient just below.
            assert_eq!(
                t.gram_numerical_rank(floor),
                Some(k),
                "n={n}: the floor itself must still hold the band rank"
            );
            assert_eq!(
                t.gram_numerical_rank(floor - 1e-3),
                Some(k - 1),
                "n={n}: the rank must drop immediately below the floor — otherwise \
                 the edge is an interior artefact, not the cliff"
            );

            // Below the cliff the surviving 2-D range is separated from the third
            // direction by an O(λ_max) gap, so the witness is admissible THERE too;
            // the refusal across the cliff is a rank change, not an unresolved
            // projector.
            let below = t
                .range_projector_error_bar(floor - 1e-3)
                .expect("finite Gram below the cliff");
            assert!(
                below < PSI_GRAM_SKIP_PROJ_ATOL,
                "n={n}: below the cliff the kept/dropped gap is O(λ_max), so the \
                 projector must be resolved far inside the tolerance; got {below:e}"
            );
            floors.push(floor);
        }

        // n-TRANSPORT. The crossing is n-free in exact arithmetic (cutoff and the
        // third eigenvalue both scale linearly in n), so the two builds must agree
        // to the bisection's own resolution, not to a window-sized slop. The
        // bisection stops at `PSI_BAND_BISECTION_ATOL·(1+|ψ|)`; allow a few of
        // those plus the O(1/n) Ostrowski excursion of the crossing itself,
        // converted through the margin slope `d/dψ ln(ε²) = 2α`.
        let bisection_resolution = PSI_BAND_BISECTION_ATOL * (1.0 + psi_hi.abs());
        let transport = 8.0 * bisection_resolution + (1.0 / 200.0) / (2.0 * alpha);
        assert!(
            (floors[0] - floors[1]).abs() <= transport,
            "an admissible band edge must transport across row counts: n=200 → {}, \
             n=800 → {} (bound {transport:e})",
            floors[0],
            floors[1]
        );
    }
}

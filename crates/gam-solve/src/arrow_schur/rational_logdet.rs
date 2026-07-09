//! Desync-safe stochastic log-determinant: a FIXED rational surrogate whose
//! value and parameter-gradient are the same deterministic functional (#2080).
//!
//! The wide-`p` REML criterion needs `½·log det S(ρ)` for the reduced evidence
//! Schur `S` (border dim `k = Σ M_k·p`), whose *dense assembly* is the
//! dominant per-eval cost at LLM widths (`O(n·q·k²)`, an order of magnitude
//! above even the `O(k³)` Cholesky at #2230 shapes). Plain SLQ
//! ([`super::slq_logdet`]) removes the assembly but re-opens the
//! objective↔gradient desync class: a stochastic VALUE paired with the exact
//! analytic gradient hands the outer line search a gradient of a *different*
//! function, and fresh probes per eval turn the criterion into noise on the
//! scale of the stall tolerances.
//!
//! This module closes that structurally. Fix, per outer solve:
//!
//! * a probe block `V = [v_1 … v_m]` (Rademacher, common random numbers across
//!   every ρ evaluation), and
//! * a fixed quadrature `{(t_ℓ, w_ℓ)}` for the integral representation
//!
//!   `log x = ∫₀^∞ ( 1/(1+t) − 1/(x+t) ) dt`,
//!
//! and define the SURROGATE
//!
//! `L̃(ρ) = Σ_ℓ w_ℓ · [ k/(1+t_ℓ) − (1/m)·Σ_j v_jᵀ (S(ρ)+t_ℓ I)⁻¹ v_j ]`.
//!
//! `L̃` is a smooth deterministic function of ρ (probes and nodes never move),
//! `E_V[L̃] = Σ_ℓ w_ℓ·[k/(1+t_ℓ) − tr(S+t_ℓ)⁻¹] ≈ log det S` to quadrature
//! accuracy, and its EXACT ρ-derivative along a direction `∂S` is
//!
//! `∂L̃ = (1/m)·Σ_j Σ_ℓ w_ℓ · y_{jℓ}ᵀ (∂S) y_{jℓ}`,  `y_{jℓ} = (S+t_ℓ I)⁻¹ v_j`
//!
//! — computable from the SAME shifted solves as the value. The outer optimizer
//! therefore descends a function whose gradient is its own: the desync class is
//! closed by construction, not by tolerance tuning. Probe-set bias is a
//! terminal concern (the fluctuation is a fixed smooth `O(m^{-1/2})`
//! perturbation of the criterion surface), certified once at the accepted ρ̂
//! by an independent probe block or one dense factorization.
//!
//! Quadrature: the half-line integral is mapped by the exp-sinh
//! double-exponential substitution `t = c·exp(sinh(u)·π/2)` and truncated
//! trapezoid in `u`. The integrand `g(t) = k/(1+t) − tr(S+t)⁻¹` is analytic on
//! `t > 0`, finite at `t → 0⁺`, and decays like `1/t²`, so the DE-trapezoid
//! error decays double-exponentially in the node count; the node window is
//! sized from the caller's spectral bracket `[λ_min, λ_max]` so the transition
//! region of every eigenvalue is inside the resolved range.
//!
//! Shifted solves: each `(S + t_ℓ I) y = v` is SPD with conditioning
//! `(λ_max+t)/(λ_min+t)` — large shifts converge in a handful of CG steps, and
//! the ladder is walked from the LARGEST shift down with warm starts (`y(t)` is
//! smooth in `t`), so only the smallest-shift solves pay meaningful iteration
//! counts. The apply is only ever consumed through a caller-provided matvec, so
//! `S` is never formed.

use super::prelude::*;
use gam_linalg::utils::splitmix64;

/// Top-subspace (Hutch++) deflation configuration for the surrogate. When a plan
/// carries one, [`RationalLogdetPlan::evaluate`] peels an `r`-dimensional
/// orthonormal subspace `Q` of the heavy (top) directions from the operator and
/// splits the log-determinant by the EXACT identity (no invariance assumed)
///
/// `tr log(S/c) = tr(Qᵀ log(S/c) Q) + tr(P log(S/c) P)`,  `P = I − QQᵀ`,
///
/// evaluating the first block deterministically over the `r` basis columns and
/// the second by Hutchinson over the PROJECTED probes `u_j = P v_j` (each with
/// its own reference norm `‖u_j‖²`, so the `k − r` bookkeeping is automatic).
/// The Hutchinson variance then rides only on the off-diagonal mass of
/// `P log(S/c) P` — small once `Q` captures the heavy directions — collapsing the
/// error bar that raw probes carry on a wide spectrum. The decomposition is
/// EXACT for ANY orthonormal `Q`; the subspace iteration only steers `Q` toward
/// the top space to reduce variance, it can never bias the estimate.
///
/// The basis is FROZEN here (built once by [`RationalLogdetPlan::with_deflation`]
/// from the operator at the plan's ρ), NOT rebuilt per evaluation. This is what
/// keeps value and gradient the SAME functional: with the estimated `term2`, the
/// sum `term1 + term2` is `Q`-dependent, so a `Q` that moved with ρ would put an
/// un-modelled `∂Q/∂ρ` term in the true gradient. A frozen `Q` makes the
/// fixed-`Q` directional derivative EXACT for the surrogate, at the cost of `Q`
/// going slightly stale as the line search moves ρ (which only relaxes the
/// variance reduction — never biases the value, since the decomposition is exact
/// for any fixed orthonormal `Q`).
#[derive(Clone)]
pub struct DeflationSpec {
    /// Frozen orthonormal top-subspace basis `Q` (columns `q_i`), built once from
    /// the operator. Reused verbatim across every ρ evaluation (CRN).
    pub basis: Vec<Array1<f64>>,
}

/// Fixed probes + fixed quadrature for one outer solve. Build once (per ρ
/// search), reuse for every criterion/gradient evaluation so the surrogate is
/// one deterministic function of ρ.
#[derive(Clone)]
pub struct RationalLogdetPlan {
    /// Operator dimension `k`.
    pub dim: usize,
    /// Rademacher probe block, `m` columns of length `dim` (CRN across ρ).
    pub probes: Vec<Array1<f64>>,
    /// Quadrature nodes `(t_ℓ, w_ℓ)` for `∫₀^∞ g(t) dt`, ordered ascending in
    /// `t` (the solve ladder walks them descending).
    pub nodes: Vec<(f64, f64)>,
    /// `ln c` for the bracket-centred representation: the estimate is
    /// `k·ln c + Σ_ℓ w_ℓ·[k/(c+t_ℓ) − tr-est (S+t_ℓ)⁻¹]`.
    pub log_center: f64,
    /// The bracket centre `c = √(λ_min·λ_max)` itself.
    pub center: f64,
    /// Optional top-subspace (Hutch++) deflation. `None` (the default from
    /// [`Self::build`]) reproduces the bare-Hutchinson path bit-for-bit; set via
    /// [`Self::with_deflation`].
    pub deflation: Option<DeflationSpec>,
}

/// One evaluation of the surrogate: the value and the per-(probe, node) solve
/// bundle `y_{jℓ}` needed to contract the exact gradient against any `∂S`
/// direction without re-solving.
pub struct RationalLogdetEval {
    /// `L̃ ≈ log det S` (surrogate value; deterministic given the plan).
    pub estimate: f64,
    /// Hutchinson standard error: sample sd of the per-probe estimates over
    /// `√m`. Zero for a single probe. The QUADRATURE part of the error is not
    /// in this bar (it is deterministic and bounded by the plan's `rel_tol`).
    pub std_err: f64,
    /// `y_{jℓ} = (S + t_ℓ I)⁻¹ u_j`, outer index `ℓ` (node), inner `j` (probe).
    /// `u_j = P v_j` are the deflation-PROJECTED probes when the plan carries a
    /// [`DeflationSpec`] (`u_j = v_j` — the raw probes — otherwise).
    pub shifted_solves: Vec<Vec<Array1<f64>>>,
    /// `y_{q_iℓ} = (S + t_ℓ I)⁻¹ q_i` for each deflation-basis column `q_i`,
    /// outer index `ℓ` (node), inner `i` (basis column). Empty without deflation.
    /// Carried so the directional derivative contracts the deterministic
    /// `tr(Qᵀ log(S/c) Q)` block against `∂S` from the SAME shifted solves.
    pub deflation_solves: Vec<Vec<Array1<f64>>>,
    /// The orthonormal deflation basis `Q` (columns `q_i`) actually realised for
    /// this evaluation; empty without deflation, and possibly shorter than the
    /// requested rank if the block collapsed.
    pub deflation_basis: Vec<Array1<f64>>,
    /// Total CG iterations spent (diagnostic).
    pub cg_iterations: usize,
}

impl RationalLogdetPlan {
    /// Build a plan for spectrum bracket `[lambda_min, lambda_max]` (rough
    /// estimates are fine — the window is padded two decades on each side),
    /// `num_probes` Rademacher probes, and a target quadrature accuracy of
    /// roughly `rel_tol` on `log det`.
    pub fn build(
        dim: usize,
        num_probes: usize,
        seed: u64,
        lambda_min: f64,
        lambda_max: f64,
        rel_tol: f64,
    ) -> Option<Self> {
        if dim == 0
            || num_probes == 0
            || !(lambda_min.is_finite() && lambda_max.is_finite())
            || lambda_min <= 0.0
            || lambda_max < lambda_min
            || !(rel_tol.is_finite() && rel_tol > 0.0 && rel_tol < 1.0)
        {
            return None;
        }
        let mut probes = Vec::with_capacity(num_probes);
        for p in 0..num_probes {
            let mut v = Array1::<f64>::zeros(dim);
            let mut state = seed.wrapping_add(p as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut bits: u64 = 0;
            let mut remaining: u32 = 0;
            for value in v.iter_mut() {
                if remaining == 0 {
                    bits = splitmix64(&mut state);
                    remaining = 64;
                }
                *value = if bits & 1 == 1 { 1.0 } else { -1.0 };
                bits >>= 1;
                remaining -= 1;
            }
            probes.push(v);
        }
        // Bracket-centred exp-sinh DE nodes for the shifted representation
        //
        //   log x = log c + ∫₀^∞ ( 1/(c+t) − 1/(x+t) ) dt,   c = √(λ_min·λ_max),
        //
        // with t(u) = c·exp(π/2·sinh u), dt = t·(π/2)·cosh u du. Centring at the
        // geometric bracket midpoint keeps the integrand's complex poles
        // (t = −λ_i, i.e. u where c·exp(π/2·sinh u) = −λ_i) as far from the
        // real u-axis as the spectrum allows. The nearest pole sits at height
        // d(λ) ≈ (π/2)/cosh(u_λ), u_λ = asinh((2/π)·ln(λ/c)), which SHRINKS
        // with the bracket width — the reason a fixed h fails at wide κ. Size
        // the step from the trapezoid-DE bound err ~ exp(−2π·d_min/h):
        // h = 2π·d_min/ln(1/tol).
        //
        // TRUNCATION WINDOW must be sized by rel_tol, NOT a fixed decade pad. The
        // dropped tails of the t-integral are, for the EXTREME eigenvalues,
        //   low : ∫₀^{t_lo}(1/(c+t) − 1/(λ_min+t))dt ≈ (1/c − 1/λ_min)·t_lo
        //         ≈ −t_lo/λ_min,          bounded by rel_tol ⟺ t_lo = λ_min·rel_tol
        //   high: ∫_{t_hi}^∞(1/(c+t) − 1/(λ_max+t))dt ≈ (λ_max − c)/t_hi
        //         ≈ λ_max/t_hi,           bounded by rel_tol ⟺ t_hi = λ_max/rel_tol.
        // The former fixed two-decade pad (t_lo = (λ_min/c)·1e-2, t_hi =
        // (λ_max/c)·1e2) left these tails at O(1e-2/c) and O(1e-2) — orders ABOVE
        // rel_tol — so the estimate lost the extreme (esp. TOP) eigenvalues' tail
        // mass and was biased LOW, worst at wide κ. The DE transform compresses
        // the wider t-window into a modest u-range (double-exponential), so the
        // node count grows only logarithmically.
        let c = (lambda_min * lambda_max).sqrt();
        let t_lo = lambda_min * rel_tol;
        let t_hi = lambda_max / rel_tol;
        let u_of = |ratio: f64| ((2.0 / std::f64::consts::PI) * ratio.ln()).asinh();
        let u_lo = u_of(t_lo);
        let u_hi = u_of(t_hi);
        // Worst-case pole height over the padded bracket (evaluate at both
        // ends; the pole of the reference term at t = −c sits at u = 0 with
        // height π/2, never the minimum).
        let pole_height = |lam_over_c: f64| -> f64 {
            let s = (2.0 / std::f64::consts::PI) * lam_over_c.ln();
            std::f64::consts::FRAC_PI_2 / (1.0 + s * s).sqrt()
        };
        let d_min = pole_height(lambda_min / c)
            .min(pole_height(lambda_max / c))
            .min(std::f64::consts::FRAC_PI_2);
        let h_bound = 2.0 * std::f64::consts::PI * d_min / (1.0f64 / rel_tol).ln();
        let steps = (((u_hi - u_lo) / h_bound).ceil() as usize).max(16);
        let h = (u_hi - u_lo) / steps as f64;
        let mut nodes = Vec::with_capacity(steps + 1);
        for s in 0..=steps {
            let u = u_lo + h * s as f64;
            let t = c * (std::f64::consts::FRAC_PI_2 * u.sinh()).exp();
            let w = h * t * std::f64::consts::FRAC_PI_2 * u.cosh();
            if t.is_finite() && w.is_finite() && w > 0.0 {
                nodes.push((t, w));
            }
        }
        if nodes.is_empty() {
            return None;
        }
        Some(Self {
            dim,
            probes,
            nodes,
            log_center: c.ln(),
            center: c,
            deflation: None,
        })
    }

    /// Attach top-subspace (Hutch++) deflation, FREEZING an orthonormal basis `Q`
    /// of up to `rank` columns built now from `matvec` by `subspace_iters`
    /// block-power steps (`Q ← orthonormalise(S·Q)`) from a `seed`-deterministic
    /// Rademacher start. The frozen `Q` is reused for every subsequent
    /// [`Self::evaluate`], so the surrogate stays one deterministic function of ρ
    /// with the fixed-`Q` directional derivative as its EXACT gradient (see
    /// [`DeflationSpec`]). Build this at the plan's ρ, from the same operator the
    /// evaluations use. `rank = 0` (or a fully-collapsed block) yields the
    /// bare-Hutchinson plan unchanged.
    pub fn with_deflation(
        mut self,
        matvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
        rank: usize,
        subspace_iters: usize,
        seed: u64,
    ) -> Self {
        let basis = build_deflation_basis(matvec, self.dim, rank, subspace_iters, seed);
        self.deflation = (!basis.is_empty()).then_some(DeflationSpec { basis });
        self
    }

    /// Evaluate the surrogate `L̃ ≈ log det S` through `matvec(v) = S·v`.
    ///
    /// Each shifted system is solved by plain CG to relative residual
    /// `cg_rel_tol`, walking the shift ladder from the largest `t` (near-trivial
    /// solves) down to the smallest, warm-starting each solve from the previous
    /// shift's solution for the same probe.
    pub fn evaluate(
        &self,
        matvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
        cg_rel_tol: f64,
        cg_max_iters: usize,
    ) -> Option<RationalLogdetEval> {
        let m = self.probes.len();
        let k = self.dim as f64;
        // Ladder: descending t (warm starts carry per vector across shifts).
        let mut order: Vec<usize> = (0..self.nodes.len()).collect();
        order.sort_by(|&a, &b| {
            self.nodes[b]
                .0
                .partial_cmp(&self.nodes[a].0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // FROZEN top-subspace deflation basis Q (empty without a DeflationSpec).
        // Built once at plan creation from the operator at the plan's ρ; reused
        // verbatim here so the surrogate is one fixed-Q function of ρ.
        let basis: &[Array1<f64>] = self
            .deflation
            .as_ref()
            .map(|d| d.basis.as_slice())
            .unwrap_or(&[]);

        // Deflation-projected probes u_j = P v_j = v_j − Q(Qᵀ v_j). With no
        // basis this is v_j unchanged, so the bare-Hutchinson path is recovered
        // bit-for-bit (‖u_j‖² = k, one solve family, no term1).
        let probes_proj: Vec<Array1<f64>> = self
            .probes
            .iter()
            .map(|v| {
                let mut u = v.clone();
                for q in basis {
                    let c = u.dot(q);
                    u.scaled_add(-c, q);
                }
                u
            })
            .collect();

        // Shifted solves for the projected probes and (if any) the basis columns.
        let (shifted, iters_probe) =
            solve_shift_ladder(matvec, &self.nodes, &order, &probes_proj, cg_rel_tol, cg_max_iters)?;
        let (deflation_solves, iters_basis) = if basis.is_empty() {
            (Vec::new(), 0)
        } else {
            solve_shift_ladder(matvec, &self.nodes, &order, basis, cg_rel_tol, cg_max_iters)?
        };
        let total_iters = iters_probe + iters_basis;

        // term1 = tr(Qᵀ log(S/c) Q) = Σ_i Σ_ℓ w_ℓ (‖q_i‖²/(c+t_ℓ) − q_iᵀ y_{q_iℓ}),
        // ‖q_i‖² = 1. Deterministic (no probe variance).
        let mut term1 = 0.0_f64;
        for (ell, &(t, w)) in self.nodes.iter().enumerate() {
            let reference = 1.0 / (self.center + t);
            for (i, q) in basis.iter().enumerate() {
                term1 += w * (reference - q.dot(&deflation_solves[ell][i]));
            }
        }

        // term2 per-probe: e_j = Σ_ℓ w_ℓ (‖u_j‖²/(c+t_ℓ) − u_jᵀ y_{u_jℓ}). The
        // PER-VECTOR reference norm ‖u_j‖² makes the (k−r) count automatic and
        // exact. The surrogate value is k·ln c + term1 + mean_j e_j; the
        // Hutchinson error bar is the spread of the e_j (term1 is deterministic,
        // so it carries no variance).
        let u_norm_sq: Vec<f64> = probes_proj.iter().map(|u| u.dot(u)).collect();
        let mut per_probe = vec![0.0_f64; m];
        for (ell, &(t, w)) in self.nodes.iter().enumerate() {
            let inv = 1.0 / (self.center + t);
            for j in 0..m {
                per_probe[j] += w * (u_norm_sq[j] * inv - probes_proj[j].dot(&shifted[ell][j]));
            }
        }
        let term2 = per_probe.iter().sum::<f64>() / m as f64;
        let estimate = k * self.log_center + term1 + term2;
        let std_err = if m > 1 {
            let var = per_probe
                .iter()
                .map(|e| (e - term2) * (e - term2))
                .sum::<f64>()
                / (m as f64 - 1.0);
            (var / m as f64).sqrt()
        } else {
            0.0
        };
        if !(estimate.is_finite() && std_err.is_finite()) {
            return None;
        }
        Some(RationalLogdetEval {
            estimate,
            std_err,
            shifted_solves: shifted,
            deflation_solves,
            deflation_basis: basis.to_vec(),
            cg_iterations: total_iters,
        })
    }

    /// Exact derivative of the surrogate along a Hessian direction: given
    /// `dmatvec(v) = (∂S)·v`, returns `∂L̃ = (1/m)·Σ_{j,ℓ} w_ℓ · y_{jℓ}ᵀ(∂S)y_{jℓ}`.
    ///
    /// This is the true gradient of the SAME function [`Self::evaluate`]
    /// returned — value and gradient can never desync.
    pub fn directional_derivative(
        &self,
        eval: &RationalLogdetEval,
        dmatvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
    ) -> Option<f64> {
        let m = self.probes.len() as f64;
        // Projected-probe block (Hutchinson, averaged over m) and the
        // deterministic deflation block (Σ over the r basis columns, NOT
        // averaged) — the exact derivative of `term2` and `term1` respectively.
        // The `k·ln c` term is ρ-independent and contributes nothing.
        let mut acc_probe = 0.0;
        let mut acc_defl = 0.0;
        for (ell, &(_, w)) in self.nodes.iter().enumerate() {
            for y in &eval.shifted_solves[ell] {
                let dy = dmatvec(y.view());
                acc_probe += w * y.dot(&dy);
            }
            if let Some(defl) = eval.deflation_solves.get(ell) {
                for y in defl {
                    let dy = dmatvec(y.view());
                    acc_defl += w * y.dot(&dy);
                }
            }
        }
        let acc = acc_defl + acc_probe / m;
        acc.is_finite().then_some(acc)
    }
}

/// Plain CG on `(A + t·I) y = b` through the un-shifted `matvec(v) = A·v`,
/// warm-started from `y0`. Returns the solution and the iteration count, or
/// `None` on a non-finite breakdown (SPD + t > 0 makes that a caller bug or a
/// non-finite operator, both of which must surface).
fn shifted_cg(
    matvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
    t: f64,
    b: &Array1<f64>,
    y0: &Array1<f64>,
    rel_tol: f64,
    max_iters: usize,
) -> Option<(Array1<f64>, usize)> {
    let apply = |v: ArrayView1<f64>| -> Array1<f64> {
        let mut out = matvec(v);
        out.scaled_add(t, &v.to_owned());
        out
    };
    let mut y = y0.clone();
    let mut r = b - &apply(y.view());
    let b_norm = b.dot(b).sqrt().max(f64::MIN_POSITIVE);
    let mut p = r.clone();
    let mut rs = r.dot(&r);
    if !rs.is_finite() {
        return None;
    }
    let tol = rel_tol * b_norm;
    let mut iters = 0usize;
    while rs.sqrt() > tol && iters < max_iters {
        let ap = apply(p.view());
        let denom = p.dot(&ap);
        if !(denom.is_finite() && denom > 0.0) {
            return None;
        }
        let alpha = rs / denom;
        y.scaled_add(alpha, &p);
        r.scaled_add(-alpha, &ap);
        let rs_new = r.dot(&r);
        if !rs_new.is_finite() {
            return None;
        }
        p = &r + &(&p * (rs_new / rs));
        rs = rs_new;
        iters += 1;
    }
    Some((y, iters))
}

/// Modified Gram-Schmidt orthonormalisation of a column block, DROPPING any
/// column whose residual norm collapses (linear dependence / rank deficiency).
/// The realised rank is `out.len()`, which may be below the input count.
fn orthonormalize(cols: &[Array1<f64>]) -> Vec<Array1<f64>> {
    let mut out: Vec<Array1<f64>> = Vec::with_capacity(cols.len());
    for col in cols {
        let mut v = col.clone();
        // TWO MGS passes ("twice is enough", Kahan/Parlett): block-power drives
        // the columns of S·Q toward the dominant eigenvector, so the input block
        // is ill-conditioned and a SINGLE pass leaves orthogonality error O(κ·ε).
        // Q enters the DETERMINISTIC term1 = tr(Qᵀ log(S/c) Q), where any
        // QᵀQ ≠ I directly biases the estimate (a slack basis would only widen
        // the Hutchinson bar, but a non-orthonormal one shifts the value). The
        // second pass restores orthogonality to O(ε). The collapse test uses the
        // FIRST-pass residual norm (relative to the pre-orthogonalisation norm) so
        // a genuinely dependent column is still dropped, not merely re-cleaned.
        let v0_norm = v.dot(&v).sqrt();
        for basis in &out {
            let proj = v.dot(basis);
            v.scaled_add(-proj, basis);
        }
        let norm_after_first = v.dot(&v).sqrt();
        for basis in &out {
            let proj = v.dot(basis);
            v.scaled_add(-proj, basis);
        }
        let norm = v.dot(&v).sqrt();
        let collapsed = !(norm_after_first.is_finite())
            || norm_after_first <= 1e-12 * v0_norm.max(1e-300)
            || !(norm.is_finite())
            || norm <= 1e-12;
        if !collapsed {
            v.mapv_inplace(|x| x / norm);
            out.push(v);
        }
    }
    out
}

/// Build the Hutch++ top-subspace basis `Q` (`≤ rank` orthonormal columns) by
/// block-power (subspace) iteration on the operator: a `seed`-deterministic
/// Rademacher start block, orthonormalised, then `iters` rounds of
/// `Q ← orthonormalise(S·Q)`. The result steers toward the top eigenspace so the
/// deflated Hutchinson variance is small; the log-det decomposition is EXACT for
/// any orthonormal `Q`, so a slack `Q` cannot bias the estimate (only widen the
/// error bar). Deterministic for a fixed `(matvec, dim, rank, iters, seed)`.
fn build_deflation_basis(
    matvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
    dim: usize,
    rank: usize,
    iters: usize,
    seed: u64,
) -> Vec<Array1<f64>> {
    let r = rank.min(dim);
    if r == 0 {
        return Vec::new();
    }
    let mut cols: Vec<Array1<f64>> = (0..r)
        .map(|col| {
            let mut v = Array1::<f64>::zeros(dim);
            let mut state = seed
                .wrapping_add((col as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .wrapping_add(0xD1B5_4A32_D192_ED03);
            let mut bits: u64 = 0;
            let mut remaining: u32 = 0;
            for value in v.iter_mut() {
                if remaining == 0 {
                    bits = splitmix64(&mut state);
                    remaining = 64;
                }
                *value = if bits & 1 == 1 { 1.0 } else { -1.0 };
                bits >>= 1;
                remaining -= 1;
            }
            v
        })
        .collect();
    cols = orthonormalize(&cols);
    for _ in 0..iters.max(1) {
        if cols.is_empty() {
            break;
        }
        let applied: Vec<Array1<f64>> = cols.iter().map(|c| matvec(c.view())).collect();
        cols = orthonormalize(&applied);
    }
    cols
}

/// Solve `(S + t_ℓ I) y = v` for every input vector across the whole shift
/// ladder, walking `order` (descending `t`) with per-vector warm starts (the
/// solution is smooth in `t`, so the previous shift seeds the next). Returns
/// `solves[ℓ][j]` and the total CG iteration count, or `None` on a shifted-CG
/// breakdown. Shared by the projected-probe and deflation-basis solve families
/// so both warm-start identically.
fn solve_shift_ladder(
    matvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
    nodes: &[(f64, f64)],
    order: &[usize],
    vectors: &[Array1<f64>],
    cg_rel_tol: f64,
    cg_max_iters: usize,
) -> Option<(Vec<Vec<Array1<f64>>>, usize)> {
    let m = vectors.len();
    let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
    let mut solves: Vec<Vec<Array1<f64>>> = vec![Vec::with_capacity(m); nodes.len()];
    let mut warm: Vec<Array1<f64>> = vec![Array1::zeros(dim); m];
    let mut total = 0usize;
    for &ell in order {
        let (t, _) = nodes[ell];
        let mut per = Vec::with_capacity(m);
        for (j, v) in vectors.iter().enumerate() {
            let (y, iters) = shifted_cg(matvec, t, v, &warm[j], cg_rel_tol, cg_max_iters)?;
            total += iters;
            warm[j] = y.clone();
            per.push(y);
        }
        solves[ell] = per;
    }
    Some((solves, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn next_uniform(state: &mut u64, lo: f64, hi: f64) -> f64 {
        let bits = splitmix64(state) >> 11;
        let unit = (bits as f64) / ((1u64 << 53) as f64);
        lo + (hi - lo) * unit
    }

    /// Random SPD `A = Q diag(λ) Qᵀ` with a prescribed spectrum, returned with
    /// its exact `log det` and eigen-pieces for derivative oracles.
    fn spd_with_spectrum(dim: usize, lambdas: &[f64], seed: u64) -> (Array2<f64>, f64) {
        let mut state = seed;
        let mut g = Array2::<f64>::zeros((dim, dim));
        for v in g.iter_mut() {
            // Box-Muller from two uniforms.
            let u1 = next_uniform(&mut state, 1e-12, 1.0);
            let u2 = next_uniform(&mut state, 0.0, 1.0);
            *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
        // QR via Gram-Schmidt for an orthonormal Q (dim is small in tests).
        let mut q = Array2::<f64>::zeros((dim, dim));
        for c in 0..dim {
            let mut col = g.column(c).to_owned();
            for prev in 0..c {
                let proj = q.column(prev).dot(&col);
                let prev_col = q.column(prev).to_owned();
                col.scaled_add(-proj, &prev_col);
            }
            let norm = col.dot(&col).sqrt();
            let col = col / norm;
            q.column_mut(c).assign(&col);
        }
        let mut a = Array2::<f64>::zeros((dim, dim));
        for (i, &l) in lambdas.iter().enumerate() {
            let qi = q.column(i);
            for r in 0..dim {
                for c in 0..dim {
                    a[[r, c]] += l * qi[r] * qi[c];
                }
            }
        }
        let logdet: f64 = lambdas.iter().map(|l| l.ln()).sum();
        (a, logdet)
    }

    #[test]
    fn quadrature_is_exact_on_scalar_spectrum() {
        // dim=1: Hutchinson is exact (v = ±1), so the only error is quadrature.
        for &x in &[1e-6, 1e-3, 0.5, 1.0, 7.3, 1e4, 1e8] {
            let plan = RationalLogdetPlan::build(1, 1, 7, x, x, 1e-10).expect("plan");
            let a = Array2::from_elem((1, 1), x);
            let eval = plan
                .evaluate(&|v: ArrayView1<f64>| a.dot(&v), 1e-14, 10_000)
                .expect("eval");
            let err = (eval.estimate - x.ln()).abs() / x.ln().abs().max(1.0);
            assert!(
                err < 1e-8,
                "quadrature error {err:.3e} at x={x:e} (est {} vs {})",
                eval.estimate,
                x.ln()
            );
        }
    }

    #[test]
    fn matches_dense_logdet_within_probe_error_at_wide_kappa() {
        // κ = 1e8 spectrum, log-uniform. With m probes the Hutchinson std-err
        // scales like sqrt(2 Σ (stuff)/m); assert against a generous multiple
        // of the exact dense answer's scale rather than tuning to luck.
        let dim = 96;
        let mut state = 42u64;
        let lambdas: Vec<f64> = (0..dim)
            .map(|_| 10f64.powf(next_uniform(&mut state, -4.0, 4.0)))
            .collect();
        let (a, logdet) = spd_with_spectrum(dim, &lambdas, 1234);
        let lmin = lambdas.iter().cloned().fold(f64::INFINITY, f64::min);
        let lmax = lambdas.iter().cloned().fold(0.0f64, f64::max);
        let plan = RationalLogdetPlan::build(dim, 64, 11, lmin, lmax, 1e-9).expect("plan");
        let eval = plan
            .evaluate(&|v: ArrayView1<f64>| a.dot(&v), 1e-12, 50_000)
            .expect("eval");
        // The probe fluctuation on a wide spectrum is genuinely large (Hutchinson
        // variance ~ 2·off-diag mass of log S), so assert the estimator against
        // its OWN error bar (5σ ⇒ false-failure odds ~1e-6) plus a small
        // deterministic quadrature budget — this validates estimate AND bar.
        let err = (eval.estimate - logdet).abs();
        let budget = 5.0 * eval.std_err + 1e-3 * logdet.abs().max(1.0);
        assert!(
            err < budget,
            "estimate {} vs exact {} — |err| {err:.3e} exceeds 5σ+quad budget {budget:.3e} \
             (std_err {:.3e})",
            eval.estimate,
            logdet,
            eval.std_err
        );
        assert!(
            eval.std_err.is_finite() && eval.std_err > 0.0,
            "multi-probe eval must report a positive error bar"
        );
    }

    #[test]
    fn directional_derivative_matches_fd_of_the_surrogate_itself() {
        // THE contract: the reported gradient is the exact derivative of the
        // SURROGATE (same probes, same nodes), not of the true log det. Central
        // FD of evaluate() along a random SPD direction must agree tightly.
        let dim = 40;
        let mut state = 9u64;
        let lambdas: Vec<f64> = (0..dim)
            .map(|_| 10f64.powf(next_uniform(&mut state, -2.0, 2.0)))
            .collect();
        let (a, _) = spd_with_spectrum(dim, &lambdas, 77);
        let d_lambdas: Vec<f64> = (0..dim)
            .map(|_| next_uniform(&mut state, 0.1, 1.0))
            .collect();
        let (da, _) = spd_with_spectrum(dim, &d_lambdas, 78);
        let plan = RationalLogdetPlan::build(dim, 8, 5, 1e-2, 1e2, 1e-9).expect("plan");
        let eval = plan
            .evaluate(&|v: ArrayView1<f64>| a.dot(&v), 1e-13, 20_000)
            .expect("eval");
        let grad = plan
            .directional_derivative(&eval, &|v: ArrayView1<f64>| da.dot(&v))
            .expect("grad");
        let h = 1e-5;
        let a_plus = &a + &(&da * h);
        let a_minus = &a - &(&da * h);
        let f_plus = plan
            .evaluate(&|v: ArrayView1<f64>| a_plus.dot(&v), 1e-13, 20_000)
            .expect("eval+")
            .estimate;
        let f_minus = plan
            .evaluate(&|v: ArrayView1<f64>| a_minus.dot(&v), 1e-13, 20_000)
            .expect("eval-")
            .estimate;
        let fd = (f_plus - f_minus) / (2.0 * h);
        let rel = (grad - fd).abs() / fd.abs().max(1e-12);
        assert!(
            rel < 1e-5,
            "surrogate gradient {grad:.9e} vs its own FD {fd:.9e} (rel {rel:.3e})"
        );
        // Sign sanity: derivative of log det along an SPD direction is positive.
        assert!(grad > 0.0, "SPD direction must increase log det, got {grad}");
    }

    #[test]
    fn evaluate_is_deterministic_across_calls() {
        let dim = 24;
        let lambdas: Vec<f64> = (1..=dim).map(|i| i as f64).collect();
        let (a, _) = spd_with_spectrum(dim, &lambdas, 3);
        let plan = RationalLogdetPlan::build(dim, 4, 99, 1.0, dim as f64, 1e-8).expect("plan");
        let e1 = plan
            .evaluate(&|v: ArrayView1<f64>| a.dot(&v), 1e-12, 10_000)
            .expect("eval1")
            .estimate;
        let e2 = plan
            .evaluate(&|v: ArrayView1<f64>| a.dot(&v), 1e-12, 10_000)
            .expect("eval2")
            .estimate;
        assert_eq!(e1, e2, "fixed plan must be bit-deterministic");
    }

    #[test]
    fn full_rank_deflation_is_exact_no_hutchinson() {
        // Deflating the ENTIRE space (rank = dim) makes P = 0: every probe
        // projects to zero, term2 vanishes with no variance, and the estimate is
        // the deterministic quadrature tr log(S/c) over a full orthonormal basis
        // = exact log det. Pins the term1 / decomposition bookkeeping as
        // UNBIASED (a wrong reference count or projector would shift it).
        let dim = 28;
        let lambdas: Vec<f64> = (1..=dim).map(|i| 0.3 + 0.7 * i as f64).collect();
        let (a, logdet) = spd_with_spectrum(dim, &lambdas, 31);
        let lmin = lambdas.iter().cloned().fold(f64::INFINITY, f64::min);
        let lmax = lambdas.iter().cloned().fold(0.0f64, f64::max);
        let matvec = |v: ArrayView1<f64>| a.dot(&v);
        let plan = RationalLogdetPlan::build(dim, 4, 5, lmin, lmax, 1e-11)
            .expect("plan")
            .with_deflation(&matvec, dim, 2, 123);
        let eval = plan.evaluate(&matvec, 1e-14, 20_000).expect("eval");
        assert_eq!(
            eval.deflation_basis.len(),
            dim,
            "full-rank block must realise dim orthonormal columns"
        );
        assert!(
            eval.std_err < 1e-8,
            "full deflation leaves ~no Hutchinson variance (P ≈ 0), got std_err={:.3e}",
            eval.std_err
        );
        let rel = (eval.estimate - logdet).abs() / logdet.abs().max(1.0);
        assert!(
            rel < 1e-6,
            "full-rank deflation must be exact to quadrature: rel {rel:.3e} \
             (est {} vs {logdet})",
            eval.estimate
        );
    }

    #[test]
    fn deflation_cuts_error_bar_and_stays_accurate_at_wide_kappa() {
        // κ = 1e8 log-uniform: raw Hutchinson carries a large bar; peeling the
        // top-16 directions collapses it while the estimate stays accurate (the
        // decomposition is exact for any Q, term2 unbiased for the projected
        // probes).
        let dim = 96;
        let mut state = 2026u64;
        let lambdas: Vec<f64> = (0..dim)
            .map(|_| 10f64.powf(next_uniform(&mut state, -4.0, 4.0)))
            .collect();
        let (a, logdet) = spd_with_spectrum(dim, &lambdas, 4321);
        let lmin = lambdas.iter().cloned().fold(f64::INFINITY, f64::min);
        let lmax = lambdas.iter().cloned().fold(0.0f64, f64::max);
        let matvec = |v: ArrayView1<f64>| a.dot(&v);
        let plain = RationalLogdetPlan::build(dim, 32, 17, lmin, lmax, 1e-9).expect("plan");
        let defl = plain.clone().with_deflation(&matvec, 16, 3, 555);
        let e_plain = plain.evaluate(&matvec, 1e-12, 50_000).expect("plain");
        let e_defl = defl.evaluate(&matvec, 1e-12, 50_000).expect("defl");
        let rel = (e_defl.estimate - logdet).abs() / logdet.abs().max(1.0);
        eprintln!(
            "wide-κ: plain std_err={:.3e} defl std_err={:.3e} defl rel={:.3e}",
            e_plain.std_err, e_defl.std_err, rel
        );
        assert!(
            rel < 0.05,
            "deflated estimate rel err {rel:.3e} (est {} vs exact {logdet})",
            e_defl.estimate
        );
        assert!(
            e_defl.std_err < e_plain.std_err,
            "deflation must shrink the Hutchinson error bar (plain {:.3e} vs defl {:.3e})",
            e_plain.std_err,
            e_defl.std_err
        );
    }

    #[test]
    fn deflated_directional_derivative_matches_fd_of_surrogate() {
        // The value↔gradient no-desync contract WITH deflation: the fixed-Q
        // directional derivative is the exact derivative of the surrogate value
        // because Q is FROZEN. Building the plan's Q once from `a` and reusing it
        // for the a ± h·da evaluations holds Q fixed, so the central FD matches
        // the analytic gradient tightly.
        let dim = 40;
        let mut state = 9u64;
        let lambdas: Vec<f64> = (0..dim)
            .map(|_| 10f64.powf(next_uniform(&mut state, -2.0, 2.0)))
            .collect();
        let (a, _) = spd_with_spectrum(dim, &lambdas, 77);
        let d_lambdas: Vec<f64> = (0..dim).map(|_| next_uniform(&mut state, 0.1, 1.0)).collect();
        let (da, _) = spd_with_spectrum(dim, &d_lambdas, 78);
        let matvec = |v: ArrayView1<f64>| a.dot(&v);
        let plan = RationalLogdetPlan::build(dim, 8, 5, 1e-2, 1e2, 1e-9)
            .expect("plan")
            .with_deflation(&matvec, 6, 3, 4242);
        assert!(
            plan.deflation.as_ref().is_some_and(|d| !d.basis.is_empty()),
            "deflation basis must have been frozen"
        );
        let eval = plan.evaluate(&matvec, 1e-13, 20_000).expect("eval");
        let grad = plan
            .directional_derivative(&eval, &|v: ArrayView1<f64>| da.dot(&v))
            .expect("grad");
        let h = 1e-5;
        let a_plus = &a + &(&da * h);
        let a_minus = &a - &(&da * h);
        let f_plus = plan
            .evaluate(&|v: ArrayView1<f64>| a_plus.dot(&v), 1e-13, 20_000)
            .expect("eval+")
            .estimate;
        let f_minus = plan
            .evaluate(&|v: ArrayView1<f64>| a_minus.dot(&v), 1e-13, 20_000)
            .expect("eval-")
            .estimate;
        let fd = (f_plus - f_minus) / (2.0 * h);
        let rel = (grad - fd).abs() / fd.abs().max(1e-12);
        assert!(
            rel < 1e-5,
            "deflated surrogate gradient {grad:.9e} vs its own FD {fd:.9e} (rel {rel:.3e})"
        );
        assert!(grad > 0.0, "SPD direction must increase log det, got {grad}");
    }
}

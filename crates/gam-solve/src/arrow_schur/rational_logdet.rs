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
use gam_linalg::utils::{splitmix64, splitmix64_hash};

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

/// Lossless low-rank representation of the derivative of one fixed rational
/// log-determinant evaluation.
///
/// For every symmetric operator direction `D`,
///
/// `plan.directional_derivative(eval, D) = (1/r) Σ_a x_a^T D x_a`,
///
/// where `x_a` are [`Self::vectors`] and `r` is their count.  The vectors fold
/// in every quadrature weight, the Hutchinson `1/m`, and the deterministic
/// deflation block.  Consequently consumers that already assemble arrow
/// selected-inverse contractions from probe pairs can use `(vectors, vectors)`
/// without pretending that the vectors are raw probes or unshifted `S^-1`
/// solves.  This representation is the derivative of the rational SURROGATE,
/// not an estimator of the derivative of the exact log determinant.
pub struct RationalLogdetDerivativeBundle {
    pub vectors: Vec<Array1<f64>>,
}

impl RationalLogdetDerivativeBundle {
    /// Apply the represented derivative to a symmetric operator direction.
    pub fn directional_derivative(
        &self,
        dmatvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
    ) -> Option<f64> {
        if self.vectors.is_empty() {
            return None;
        }
        let inv_rank = 1.0 / self.vectors.len() as f64;
        let derivative = self
            .vectors
            .iter()
            .map(|vector| vector.dot(&dmatvec(vector.view())))
            .sum::<f64>()
            * inv_rank;
        derivative.is_finite().then_some(derivative)
    }
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
        // ONE sequential master stream for ALL probes. The former per-probe
        // initial state `(seed + p)·γ` (γ = the splitmix64 increment) made
        // probe `p` of seed `s` BIT-IDENTICAL to probe `p+1` of seed `s−1`
        // (a splitmix stream from x₀ emits the words at x₀+γ, x₀+2γ, …, so
        // any two starts differing by a multiple of γ are the same stream
        // shifted), and within one plan made probe `p+1`'s word stream probe
        // `p`'s shifted by one word — a sliding window sharing sign words
        // between consecutive probes. Each probe was still individually
        // uniform Rademacher (Hutchinson stays unbiased), but the probes were
        // NOT jointly independent: the std_err bookkeeping and any
        // seed-averaged inference (the wide-κ multiseed discriminator, whose
        // 96 seeds at unit spacing drew ~128 distinct probe vectors instead
        // of 3072 and reported a common Hutchinson fluctuation as a "5.57σ
        // deterministic bias") were invalidated. Sequential consumption from
        // one hashed master state has no window structure and no cross-seed
        // stream aliasing; determinism per seed (the CRN contract) is kept.
        let mut master = splitmix64_hash(seed);
        let probes = rademacher_block(&mut master, num_probes, dim);
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
        // Invert t(u) = c·exp(π/2·sinh u): u(t) = asinh((2/π)·ln(t/c)). The /c is
        // load-bearing — t_lo/t_hi below are ABSOLUTE truncation points, so a node
        // at u_of(t) must land at t, not c·t (which shifts the resolved window by a
        // full factor of c and under-resolves the extreme-eigenvalue tails). Mirrors
        // the /c the pole_height ratio uses just below.
        let u_of = |t: f64| ((2.0 / std::f64::consts::PI) * (t / c).ln()).asinh();
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

    /// Attach TWO-SIDED spectral deflation: freeze an orthonormal basis `Q`
    /// spanning BOTH the `top_rank` largest-λ directions (block power on `S`) and
    /// the `bottom_rank` smallest-λ directions (inverse iteration on `S⁻¹`, matrix-
    /// free via CG), merged and re-orthonormalised into one basis.
    ///
    /// This is the wide-κ variance-reduction lever. The surrogate's Hutchinson bar
    /// is `√(2·‖offdiag(P·log(S/c)·P)‖_F²)` — purely off-diagonal, so a
    /// diagonal/scalar control variate buys NOTHING (Rademacher already resolves
    /// the diagonal exactly). The off-diagonal mass of `log(S/c)` is loaded
    /// SYMMETRICALLY onto the two spectral tails (`|log(λ/c)|` peaks at both
    /// `λ_max` and `λ_min`), so the one-sided [`Self::with_deflation`] removes only
    /// half of it and stalls near `½·lnκ`-scale error bars at wide κ. Peeling both
    /// tails is a rank-`(top+bottom)` low-rank control variate whose deterministic
    /// `tr(Qᵀ log(S/c) Q)` block (term1) is computed exactly and whose complement
    /// carries only the interior — small — off-diagonal mass. At EQUAL total rank
    /// this cuts the wide-κ bar by ≈`√2`·(tail/interior ratio) over one-sided
    /// deflation; the decomposition stays EXACT for any orthonormal `Q`, so the
    /// value is never biased (only the bar shrinks). `top_rank = bottom_rank = 0`
    /// reduces to the bare-Hutchinson plan.
    ///
    /// The `cg` budget `(rel_tol, max_iters)` bounds the inverse-iteration solves;
    /// it may be loose (an approximate bottom `Q` only relaxes the variance
    /// reduction, never biases the estimate). Build this once at the plan's ρ, from
    /// the same operator the evaluations use.
    pub fn with_two_sided_deflation(
        mut self,
        matvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
        top_rank: usize,
        bottom_rank: usize,
        subspace_iters: usize,
        seed: u64,
        cg: (f64, usize),
    ) -> Option<Self> {
        let (cg_rel_tol, cg_max_iters) = cg;
        let mut cols = build_deflation_basis(matvec, self.dim, top_rank, subspace_iters, seed);
        cols.extend(build_inverse_deflation_basis(
            matvec,
            self.dim,
            bottom_rank,
            subspace_iters,
            seed,
            cg_rel_tol,
            cg_max_iters,
        )?);
        // Merge the two orthonormal families into ONE orthonormal basis (the top
        // and bottom blocks are near-orthogonal but not exactly; the second MGS
        // pass in `orthonormalize` cleans the cross terms and drops any collapsed
        // column, so `Q` stays exactly orthonormal — the property term1 needs).
        let basis = orthonormalize(&cols);
        self.deflation = (!basis.is_empty()).then_some(DeflationSpec { basis });
        Some(self)
    }

    /// Evaluate the surrogate `L̃ ≈ log det S` through `matvec(v) = S·v`.
    ///
    /// Each shifted system is solved by plain CG to normwise backward error
    /// `cg_rel_tol`, walking the shift ladder from the largest `t` (near-trivial
    /// solves) down to the smallest, warm-starting each solve from the previous
    /// shift's solution for the same probe. A stricter RHS-relative residual
    /// also terminates the solve when it is attainable.
    pub fn evaluate(
        &self,
        matvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
        cg_rel_tol: f64,
        cg_max_iters: usize,
    ) -> Option<RationalLogdetEval> {
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

        // Deflation-projected probes u_j = P v_j (raw probes without a basis).
        let probes_proj = self.projected_probes(basis);

        // Shifted solves for the projected probes and (if any) the basis columns.
        let (shifted, iters_probe) = solve_shift_ladder(
            matvec,
            &self.nodes,
            &order,
            &probes_proj,
            cg_rel_tol,
            cg_max_iters,
        )?;
        let (deflation_solves, iters_basis) = if basis.is_empty() {
            (Vec::new(), 0)
        } else {
            solve_shift_ladder(matvec, &self.nodes, &order, basis, cg_rel_tol, cg_max_iters)?
        };
        self.assemble_eval(
            probes_proj,
            basis,
            shifted,
            deflation_solves,
            iters_probe + iters_basis,
        )
    }

    /// Deflation-projected probes `u_j = P v_j = v_j − Q(Qᵀ v_j)` (the raw probes
    /// bit-for-bit when `basis` is empty: `‖u_j‖² = k`, no term1). Shared by
    /// [`Self::evaluate`] and the wide-κ discriminator's exact-solve audit arm so
    /// BOTH project against the identical frozen `Q` the term1 columns use — the
    /// one place the "exact for any orthonormal Q" proof could silently break is a
    /// `Q` that differs between the probe projector and term1, so they must draw
    /// from the same `basis` slice.
    fn projected_probes(&self, basis: &[Array1<f64>]) -> Vec<Array1<f64>> {
        self.probes
            .iter()
            .map(|v| {
                let mut u = v.clone();
                for q in basis {
                    let c = u.dot(q);
                    u.scaled_add(-c, q);
                }
                u
            })
            .collect()
    }

    /// Assemble the surrogate value, error bar, and carried solves from the two
    /// shifted-solve ladders — `shifted[ℓ][j]` for the projected probes and
    /// `deflation_solves[ℓ][i]` for the basis columns, both indexed by node `ℓ`.
    /// The ONLY solver-dependent inputs are those two ladders, so [`Self::evaluate`]
    /// (CG) and any exact-solve audit that feeds the same ladders produce
    /// byte-identical term1/term2/std_err bookkeeping — the property the wide-κ
    /// discriminator's exact arm relies on to isolate solve error from a structural
    /// split bias.
    fn assemble_eval(
        &self,
        probes_proj: Vec<Array1<f64>>,
        basis: &[Array1<f64>],
        shifted: Vec<Vec<Array1<f64>>>,
        deflation_solves: Vec<Vec<Array1<f64>>>,
        total_iters: usize,
    ) -> Option<RationalLogdetEval> {
        let m = self.probes.len();
        let k = self.dim as f64;
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

    /// Collapse [`RationalLogdetEval`]'s complete shifted-solve ladder into a
    /// lossless weighted low-rank derivative representation.
    ///
    /// This is deliberately derived from the same evaluation that produced the
    /// value.  Re-solving only the raw probes at shift zero would instead encode
    /// `tr(S^-1 D)`, which is generally NOT the derivative of this fixed-node
    /// rational surrogate and would reopen the objective/gradient desynchrony
    /// the surrogate exists to prevent.
    pub fn into_directional_derivative_bundle(
        &self,
        eval: RationalLogdetEval,
    ) -> Option<RationalLogdetDerivativeBundle> {
        let expected_deflation_nodes =
            usize::from(!eval.deflation_basis.is_empty()) * self.nodes.len();
        if eval.shifted_solves.len() != self.nodes.len()
            || eval.deflation_solves.len() != expected_deflation_nodes
        {
            return None;
        }
        let probe_count = self.probes.len();
        if probe_count == 0
            || eval
                .shifted_solves
                .iter()
                .any(|solves| solves.len() != probe_count)
            || eval
                .deflation_solves
                .iter()
                .any(|solves| solves.len() != eval.deflation_basis.len())
        {
            return None;
        }
        let term_count = self.nodes.len().checked_mul(
            probe_count.checked_add(eval.deflation_basis.len())?,
        )?;
        if term_count == 0 {
            return None;
        }
        let mut vectors = Vec::with_capacity(term_count);
        let rank = term_count as f64;
        let probes = probe_count as f64;
        let mut deflation_by_node = eval.deflation_solves;
        if deflation_by_node.is_empty() {
            deflation_by_node.resize_with(self.nodes.len(), Vec::new);
        }
        for ((mut probe_solves, mut deflation_solves), &(_, weight)) in eval
            .shifted_solves
            .into_iter()
            .zip(deflation_by_node)
            .zip(&self.nodes)
        {
            if !(weight.is_finite() && weight > 0.0) {
                return None;
            }
            let probe_scale = (rank * weight / probes).sqrt();
            let deflation_scale = (rank * weight).sqrt();
            if !(probe_scale.is_finite() && deflation_scale.is_finite()) {
                return None;
            }
            for mut solve in probe_solves.drain(..) {
                if solve.len() != self.dim {
                    return None;
                }
                solve *= probe_scale;
                vectors.push(solve);
            }
            for mut solve in deflation_solves.drain(..) {
                if solve.len() != self.dim {
                    return None;
                }
                solve *= deflation_scale;
                vectors.push(solve);
            }
        }
        Some(RationalLogdetDerivativeBundle { vectors })
    }
}

/// Plain CG on `(A + t·I) y = b` through the un-shifted `matvec(v) = A·v`,
/// warm-started from `y0`. Returns the solution and the iteration count only
/// after the TRUE residual certifies either the stricter RHS-relative residual
/// or the requested normwise backward error; exhaustion and non-finite/SPD
/// breakdowns return `None`. The matrix-free backward-error denominator uses
/// the largest Rayleigh quotient observed over the CG directions. For SPD `A`,
/// this is a lower bound on `||A||₂`, hence
///
/// `||r||₂ / (lambda_observed ||y||₂ + ||b||₂)`
///
/// is a conservative upper bound on the usual normwise backward error. This
/// closes the f64 roundoff gap where `||r||/||b||` cannot reach a requested
/// tolerance even though the computed solution already solves a nearby system
/// to that tolerance. When the recursively updated CG residual reaches the
/// RHS-relative threshold before the true residual does, the recurrence is
/// restarted from the true residual (reliable residual replacement) rather
/// than rejecting a recoverable solve. Returning an uncertified
/// iteration-capped last iterate would make the value consume an uncontrolled
/// approximate inverse while the derivative formula differentiates an exact
/// inverse, re-opening the #2080 objective/gradient desynchronisation this
/// module exists to prevent.
fn shifted_cg(
    matvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
    t: f64,
    b: &Array1<f64>,
    y0: &Array1<f64>,
    rel_tol: f64,
    max_iters: usize,
) -> Option<(Array1<f64>, usize)> {
    if !(rel_tol.is_finite() && rel_tol > 0.0) {
        return None;
    }
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
    let mut observed_operator_norm = 0.0_f64;
    loop {
        if rs.sqrt() <= tol {
            // Recursive CG residuals lose their equality to `b - A y` through
            // roundoff, especially on the smallest shifts.  A recursive
            // convergence report is therefore only a prompt to inspect the
            // actual residual.  If it has not converged, restart the Krylov
            // recurrence from that exact residual and spend the remaining
            // caller-provided iteration budget.  The former terminal check
            // returned `None` immediately here, even when one reliable update
            // was enough to satisfy the requested contract.
            let true_residual = b - &apply(y.view());
            let true_rs = true_residual.dot(&true_residual);
            if !true_rs.is_finite() {
                return None;
            }
            let true_residual_norm = true_rs.sqrt();
            let y_norm = y.dot(&y).sqrt();
            if !y_norm.is_finite() {
                return None;
            }
            // Evaluate the backward-error ratio in the log domain. The scale
            // `lambda_observed * ||y|| + ||b||` can overflow even when every
            // operand and the certified ratio are representable.
            let backward_error_certified = if observed_operator_norm > 0.0 && y_norm > 0.0 {
                let log_operator_solution = observed_operator_norm.ln() + y_norm.ln();
                let log_rhs = b_norm.ln();
                let log_scale = log_operator_solution.max(log_rhs);
                let log_denominator = log_scale
                    + ((log_operator_solution - log_scale).exp()
                        + (log_rhs - log_scale).exp())
                    .ln();
                true_residual_norm.ln() - log_denominator <= rel_tol.ln()
            } else {
                false
            };
            if true_residual_norm <= tol || backward_error_certified {
                return Some((y, iters));
            }
            if iters >= max_iters {
                return None;
            }
            r = true_residual;
            rs = true_rs;
            p = r.clone();
        }
        if iters >= max_iters {
            return None;
        }
        let ap = apply(p.view());
        let denom = p.dot(&ap);
        if !(denom.is_finite() && denom > 0.0) {
            return None;
        }
        let p_norm_sq = p.dot(&p);
        if !(p_norm_sq.is_finite() && p_norm_sq > 0.0) {
            return None;
        }
        let rayleigh = denom / p_norm_sq;
        if rayleigh.is_finite() {
            observed_operator_norm = observed_operator_norm.max(rayleigh);
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
        // Numerical rank, not a tuned absolute knob: below √ε of the source
        // column's norm, orthogonal residuals carry no stable direction.
        let rank_tol = f64::EPSILON.sqrt() * v0_norm;
        let collapsed = !(v0_norm.is_finite() && v0_norm > 0.0)
            || !(norm_after_first.is_finite())
            || norm_after_first <= rank_tol
            || !(norm.is_finite())
            || norm <= rank_tol;
        if !collapsed {
            v.mapv_inplace(|x| x / norm);
            out.push(v);
        }
    }
    out
}

/// Draw `ncols` length-`dim` Rademacher (±1) vectors by consuming ONE sequential
/// splitmix stream from `master` (LSB-first, 64 signs per word), the bit buffer
/// reset per column. Single home for the probe/start-block generation shared by
/// [`RationalLogdetPlan::build`], [`build_deflation_basis`], and
/// [`build_inverse_deflation_basis`]; consuming from one advancing `master`
/// (rather than a per-column `(seed + col)·γ` restart) is what removes the
/// cross-column / cross-seed stream aliasing documented in `build`.
fn rademacher_block(master: &mut u64, ncols: usize, dim: usize) -> Vec<Array1<f64>> {
    (0..ncols)
        .map(|_| {
            let mut v = Array1::<f64>::zeros(dim);
            let mut bits: u64 = 0;
            let mut remaining: u32 = 0;
            for value in v.iter_mut() {
                if remaining == 0 {
                    bits = splitmix64(master);
                    remaining = 64;
                }
                *value = if bits & 1 == 1 { 1.0 } else { -1.0 };
                bits >>= 1;
                remaining -= 1;
            }
            v
        })
        .collect()
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
    // One sequential master stream for the whole start block — same
    // decorrelation as the probe generation in `RationalLogdetPlan::build`:
    // the former per-column start `seed + col·γ + const` (γ = the splitmix64
    // increment) made column c+1's word stream column c's shifted by one
    // word (sliding-window sharing). Harmless to the EXACTNESS of the
    // deflated split (any orthonormal Q is valid), but a correlated start
    // block weakens the subspace iteration's coverage of the top eigenspace
    // for no reason. Determinism per seed is kept.
    let mut master = splitmix64_hash(seed.wrapping_add(0xD1B5_4A32_D192_ED03));
    let mut cols = orthonormalize(&rademacher_block(&mut master, r, dim));
    for _ in 0..iters {
        if cols.is_empty() {
            break;
        }
        let applied: Vec<Array1<f64>> = cols.iter().map(|c| matvec(c.view())).collect();
        cols = orthonormalize(&applied);
    }
    cols
}

/// Build the BOTTOM (smallest-λ) subspace basis by INVERSE subspace iteration:
/// the same block-power as [`build_deflation_basis`] but with the operator
/// replaced by `S⁻¹` (applied matrix-free by plain CG through `matvec`), so the
/// rounds `Q ← orthonormalise(S⁻¹·Q)` amplify the SMALLEST eigenvalues instead of
/// the largest. This is the second arm of the two-sided control variate
/// ([`RationalLogdetPlan::with_two_sided_deflation`]): the Hutchinson variance of
/// the surrogate rides on the off-diagonal Frobenius mass of `log(S/c)`, which a
/// wide spectrum loads SYMMETRICALLY onto both tails (`log(λ_max/c) = +½lnκ` and
/// `log(λ_min/c) = −½lnκ`), so peeling only the top leaves the entire bottom-tail
/// contribution in the bar. A polynomial filter `(μI − S)` cannot reach the
/// bottom on a dense log-uniform spectrum (the relative gap `(μ−λ_1)/(μ−λ_2) ≈ 1`
/// gives no separation); genuine bottom amplification needs `S⁻¹`, whence the CG
/// inverse iteration here.
///
/// The solves may use a loose requested tolerance — an approximate bottom `Q`
/// only relaxes variance reduction and cannot bias the exact split — but every
/// requested solve must still CONVERGE to that tolerance. Exhaustion propagates
/// as `None`; silently retaining an un-amplified start column would falsify the
/// requested two-sided variance contract. The whole build is a ONE-TIME frozen
/// cost per outer solve, never per evaluation.
fn build_inverse_deflation_basis(
    matvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
    dim: usize,
    rank: usize,
    iters: usize,
    seed: u64,
    cg_rel_tol: f64,
    cg_max_iters: usize,
) -> Option<Vec<Array1<f64>>> {
    let r = rank.min(dim);
    if r == 0 {
        return Some(Vec::new());
    }
    // Distinct master stream from the top-basis start (a different additive
    // offset into splitmix) so the top and bottom start blocks are not aliased.
    let mut master = splitmix64_hash(seed.wrapping_add(0x2545_F491_4F6C_DD1D));
    let mut cols = orthonormalize(&rademacher_block(&mut master, r, dim));
    let zero = Array1::<f64>::zeros(dim);
    for _ in 0..iters {
        if cols.is_empty() {
            break;
        }
        // Inverse iteration step: apply S⁻¹ column-wise via plain CG (shift 0 on
        // the SPD operator). Every solve must meet the caller's (possibly loose)
        // tolerance; an exhausted solve invalidates the requested bottom peel.
        let applied: Option<Vec<Array1<f64>>> = cols
            .iter()
            .map(|c| shifted_cg(matvec, 0.0, c, &zero, cg_rel_tol, cg_max_iters).map(|(y, _)| y))
            .collect();
        cols = orthonormalize(&applied?);
    }
    Some(cols)
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
    use ndarray::array;

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
        assert!(
            grad > 0.0,
            "SPD direction must increase log det, got {grad}"
        );
    }

    #[test]
    fn fixed_probe_derivative_bundle_matches_rational_directional_not_raw_inverse() {
        // Use an intentionally coarse, fixed quadrature window so the rational
        // surrogate's derivative is decisively different from the exact
        // shift-zero trace.  The production bundle must reproduce the former:
        // substituting `(v, S^-1 v)` here would make this regression fail.
        let diagonal = array![0.2, 3.0, 17.0];
        let direction = array![1.0, 2.0, 4.0];
        let matvec = |v: ArrayView1<f64>| &diagonal * &v;
        let dmatvec = |v: ArrayView1<f64>| &direction * &v;
        let plan = RationalLogdetPlan::build(3, 3, 71, 0.2, 17.0, 0.25)
            .expect("fixed rational plan");
        let eval = plan
            .evaluate(&matvec, 1.0e-13, 64)
            .expect("fixed rational evaluation");
        let authority = plan
            .directional_derivative(&eval, &dmatvec)
            .expect("rational directional derivative");
        let bundle = plan
            .into_directional_derivative_bundle(eval)
            .expect("lossless rational derivative bundle");
        let represented = bundle
            .directional_derivative(&dmatvec)
            .expect("represented directional derivative");
        let scale = authority.abs().max(1.0);
        assert!(
            (represented - authority).abs() <= 64.0 * f64::EPSILON * scale,
            "lossless bundle derivative {represented:.16e} != rational authority \
             {authority:.16e}"
        );

        // Rademacher probes resolve this diagonal exact-inverse trace exactly;
        // it is therefore a clean stand-in for the obsolete raw t=0 bundle.
        let raw_shift_zero = direction
            .iter()
            .zip(diagonal.iter())
            .map(|(&d, &s)| d / s)
            .sum::<f64>();
        assert!(
            (raw_shift_zero - authority).abs() > 1.0e-4,
            "fixture must separate the rational derivative ({authority:.9e}) from the \
             raw shift-zero inverse trace ({raw_shift_zero:.9e})"
        );
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
    fn shifted_cg_refuses_an_unconverged_iteration_cap() {
        let a = array![[1.0, 0.0], [0.0, 4.0]];
        let b = array![1.0, 1.0];
        let zero = Array1::<f64>::zeros(2);
        let matvec = |v: ArrayView1<f64>| a.dot(&v);

        assert!(
            shifted_cg(&matvec, 0.0, &b, &zero, 1.0e-12, 1).is_none(),
            "one CG step cannot solve a two-eigenvalue system to 1e-12; the \
             iteration-capped last iterate must be refused"
        );
        let (solved, iterations) = shifted_cg(&matvec, 0.0, &b, &zero, 1.0e-12, 2)
            .expect("two-dimensional SPD CG must converge in at most two steps");
        let residual = &b - &matvec(solved.view());
        assert!(
            residual.dot(&residual).sqrt() <= 1.0e-12 * b.dot(&b).sqrt(),
            "returned shifted solve must satisfy its true-residual contract"
        );
        assert_eq!(iterations, 2);
    }

    #[test]
    fn two_sided_deflation_propagates_bottom_solve_nonconvergence() {
        let a = array![[1.0, 0.0], [0.0, 4.0]];
        let matvec = |v: ArrayView1<f64>| a.dot(&v);
        let plan =
            RationalLogdetPlan::build(2, 2, 17, 1.0, 4.0, 1.0e-9).expect("valid rational plan");
        assert!(
            plan.with_two_sided_deflation(&matvec, 0, 1, 1, 91, (1.0e-12, 0))
                .is_none(),
            "a requested bottom-tail inverse solve may not silently fall back to \
             the unamplified start column"
        );
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
    fn full_rank_deflation_is_exact_at_wide_kappa_deterministic_bias_localizer() {
        // #2080 DEFINITIVE deterministic-bias localizer at WIDE κ. The sibling
        // `full_rank_deflation_is_exact_no_hutchinson` pins the split at κ≈20
        // (narrow); this pins it on the SAME κ≈1e8 log-uniform spectrum the wide-κ
        // multiseed discriminator uses. Deflating the ENTIRE space (rank = dim)
        // makes P = 0: term2 (Hutchinson) vanishes with NO variance, so the estimate
        // is the PURE deterministic quadrature of `tr log(S/c)` over a full
        // orthonormal basis — and `tr(Qᵀ M Q) = tr(M)` for ANY orthonormal full-rank
        // `Q`, so this is independent of which basis the block power realises.
        //
        // This is the discriminator the multiseed test could not be: if the wide-κ
        // "+2.87 (5.57σ)" were a genuine quadrature or split DEFECT it would surface
        // HERE, at rel ≈ 5%, with ZERO probe noise to hide behind. It does not — the
        // exp-sinh DE quadrature resolves the [λ_min, λ_max] = 1e8 bracket to ~1e-10
        // per eigenvalue and the term1/term2 decomposition is exact by construction.
        // A nonzero value here (rel ≥ 1e-6) is the ONLY thing that would justify
        // "quadrature/split derivation work"; its passing localises the multiseed
        // residual entirely to Hutchinson VARIANCE on the deflated complement (fixed
        // by more probes / deeper rank / a control variate, NOT a re-derivation).
        // Uses the EXACT dense-Cholesky arm so there is not even CG error to blame.
        let dim = 96;
        let mut state = 2026u64;
        let lambdas: Vec<f64> = (0..dim)
            .map(|_| 10f64.powf(next_uniform(&mut state, -4.0, 4.0)))
            .collect();
        let (a, logdet) = spd_with_spectrum(dim, &lambdas, 4321);
        let lmin = lambdas.iter().cloned().fold(f64::INFINITY, f64::min);
        let lmax = lambdas.iter().cloned().fold(0.0f64, f64::max);
        let matvec = |v: ArrayView1<f64>| a.dot(&v);
        let plan = RationalLogdetPlan::build(dim, 8, 5, lmin, lmax, 1e-9)
            .expect("plan")
            .with_deflation(&matvec, dim, 2, 555);
        let eval = evaluate_exact(&plan, &a);
        assert_eq!(
            eval.deflation_basis.len(),
            dim,
            "the rank=dim block power must realise a full orthonormal basis even at \
             κ≈1e8 (got {}); if it collapses, term2 is nonzero and this stops being a \
             zero-variance deterministic check",
            eval.deflation_basis.len()
        );
        assert!(
            eval.std_err < 1e-8,
            "full deflation must leave ~no Hutchinson variance (P ≈ 0) at wide κ, got \
             std_err={:.3e}",
            eval.std_err
        );
        let rel = (eval.estimate - logdet).abs() / logdet.abs().max(1.0);
        assert!(
            rel < 1e-6,
            "wide-κ full-rank deflation must be exact to quadrature — a nonzero value \
             is the ONLY signature of a genuine deterministic quadrature/split bias: \
             rel {rel:.3e} (est {} vs exact {logdet})",
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

    /// Exact (dense Cholesky) audit arm: solve every shifted system
    /// `(A + t_ℓ I) y = v` directly, replacing the CG ladder, but feed the results
    /// through the SAME [`RationalLogdetPlan::assemble_eval`] the production
    /// `evaluate` uses — identical probes, nodes, frozen `Q`, and term1/term2
    /// bookkeeping. Any gap between this and the exact log-det is then split/Q
    /// structure (or quadrature/probe), never CG solve error, since there is none.
    fn evaluate_exact(plan: &RationalLogdetPlan, a: &Array2<f64>) -> RationalLogdetEval {
        use gam_linalg::triangular::{
            CholeskyGuard, cholesky_factor_in_place, cholesky_solve_vector,
        };
        let basis: &[Array1<f64>] = plan
            .deflation
            .as_ref()
            .map(|d| d.basis.as_slice())
            .unwrap_or(&[]);
        let probes_proj = plan.projected_probes(basis);
        let exact_ladder = |vectors: &[Array1<f64>]| -> Vec<Vec<Array1<f64>>> {
            plan.nodes
                .iter()
                .map(|&(t, _)| {
                    let mut at = a.clone();
                    for i in 0..a.nrows() {
                        at[[i, i]] += t;
                    }
                    let l = cholesky_factor_in_place(at.view(), CholeskyGuard::FiniteStrict)
                        .expect("shifted SPD system must factor");
                    vectors
                        .iter()
                        .map(|v| cholesky_solve_vector(l.view(), v.view()))
                        .collect()
                })
                .collect()
        };
        let shifted = exact_ladder(&probes_proj);
        let deflation_solves = if basis.is_empty() {
            Vec::new()
        } else {
            exact_ladder(basis)
        };
        plan.assemble_eval(probes_proj, basis, shifted, deflation_solves, 0)
            .expect("exact assemble")
    }

    #[test]
    fn deflation_wide_kappa_bias_cg_convergence_discriminator() {
        // #2080 loop discriminator (battery_5e59e646b). With BOTH the quadrature
        // window fix and the `/c` node-placement fix landed,
        // `deflation_cuts_error_bar_and_stays_accurate_at_wide_kappa` still fails
        // ~10% low (est ≈51.54 vs exact ≈57.39) — NEARLY byte-identical to the
        // pre-fix 51.529 — while `full_rank_deflation_is_exact_no_hutchinson` now
        // PASSES. So pure quadrature is exonerated and the residual bias appears
        // ONLY with deflation at wide κ. Prime suspect: CG under-resolution of the
        // SMALL-shift solves on the κ=1e8 operator — those directions carry most of
        // the log-det magnitude, and a `cg_rel_tol` scaled to ‖b‖ can stop before
        // resolving the small-eigenvalue mass, dropping it deterministically (which
        // is why the FD gates, sharing the same value path, still pass — a
        // self-consistent bias).
        //
        // Discriminator: rerun the EXACT failing fixture with `cg_rel_tol`
        // tightened 1e-12→1e-15 and `cg_max_iters` ×10. If the deflated estimate
        // becomes accurate the bias is SOLVE-CONVERGENCE (production fix = an honest
        // per-shift iteration/tolerance budget from the shift-dependent conditioning
        // `κ_ℓ = (λmax + t)/(λmin + t)`); if it stays ~10% low (Δest ≈ 0) the bias is
        // STRUCTURAL in the deflated split and needs a fresh derivation. The loop
        // battery verdicts this test's pass/fail.
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
        // Three arms on the SAME deflated plan: loose CG (the failing fixture's
        // budget), tightened CG, and EXACT dense-Cholesky solves. The exact arm is
        // the DEFINITIVE one — it carries no CG error, so its residual is purely
        // split/Q/quadrature structure.
        let e_loose = defl.evaluate(&matvec, 1e-12, 50_000).expect("loose");
        let e_tight = defl.evaluate(&matvec, 1e-15, 500_000).expect("tight");
        let e_exact = evaluate_exact(&defl, &a);
        let rel_loose = (e_loose.estimate - logdet).abs() / logdet.abs().max(1.0);
        let rel_tight = (e_tight.estimate - logdet).abs() / logdet.abs().max(1.0);
        let rel_exact = (e_exact.estimate - logdet).abs() / logdet.abs().max(1.0);
        // Structural suspect 1: is the loose bias many σ (systematic) or within the
        // Hutchinson bar (a probe fluctuation)? gap ≈ 5.85 vs the reported std_err.
        let gap = logdet - e_loose.estimate;
        let sigma_ratio = gap.abs() / e_loose.std_err.max(1e-300);
        eprintln!(
            "wide-κ 3-arm discriminator: exact_logdet={logdet:.6}\n  \
             loose(1e-12,50k)  est={:.6} rel={rel_loose:.3e} std_err={:.3e}\n  \
             tight(1e-15,500k) est={:.6} rel={rel_tight:.3e} Δvs_loose={:.3e}\n  \
             EXACT(cholesky)   est={:.6} rel={rel_exact:.3e} Δvs_loose={:.3e}\n  \
             gap={gap:.4} = {sigma_ratio:.1}σ (loose std_err)",
            e_loose.estimate,
            e_loose.std_err,
            e_tight.estimate,
            (e_tight.estimate - e_loose.estimate).abs(),
            e_exact.estimate,
            (e_exact.estimate - e_loose.estimate).abs(),
        );

        // Structural suspect 2: the "exact for any orthonormal Q" proof breaks only
        // if the probe projector and term1 use a DIFFERENT or non-orthonormal Q.
        // Verify the realised Q is the frozen basis, is orthonormal, and that the
        // projected probes are truly Q-orthogonal (P = I − QQᵀ applied).
        let frozen: &[Array1<f64>] = defl
            .deflation
            .as_ref()
            .map(|d| d.basis.as_slice())
            .unwrap_or(&[]);
        assert_eq!(
            e_exact.deflation_basis.len(),
            frozen.len(),
            "exact arm must realise the same frozen Q rank as the plan"
        );
        for (qe, qf) in e_exact.deflation_basis.iter().zip(frozen) {
            assert!(
                (qe - qf).mapv(f64::abs).sum() < 1e-12,
                "term1's Q must be the plan's frozen Q (no drift)"
            );
        }
        for (i, qi) in frozen.iter().enumerate() {
            for (j, qj) in frozen.iter().enumerate() {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qi.dot(qj) - expect).abs() < 1e-9,
                    "frozen Q must be orthonormal: QᵀQ[{i},{j}] = {}",
                    qi.dot(qj)
                );
            }
        }
        let proj = defl.projected_probes(frozen);
        for u in &proj {
            for q in frozen {
                assert!(
                    u.dot(q).abs() < 1e-9,
                    "projected probe must be Q-orthogonal (same P as term1)"
                );
            }
        }

        // DEFINITIVE verdict via the exact arm (no CG error possible):
        //   GREEN (rel_exact < 0.05) ⇒ the deflated split / frozen-Q structure is
        //     SOUND; ALL residual bias in the CG path is solve error → the fix is
        //     per-shift preconditioning (the κ·ε≈2e-8 residual floor at κ=1e8 means
        //     a tighter cg_rel_tol alone cannot reach it; Jacobi / shifted-system
        //     preconditioning is the lever, not tolerance).
        //   RED with rel_exact ≈ rel_loose ⇒ the bias is STRUCTURAL in the deflated
        //     split with the frozen subspace-iteration Q at wide κ → fresh derivation.
        // The loop verdicts this test's pass/fail alongside the three logged arms.
        assert!(
            rel_exact < 0.05,
            "EXACT-solve deflated estimate rel err {rel_exact:.3e} (est {} vs exact {logdet}); \
             CG loose rel {rel_loose:.3e}, tight rel {rel_tight:.3e}. With no CG error possible, \
             rel_exact ≈ rel_loose ⇒ the wide-κ bias is STRUCTURAL in the deflated split, not \
             solve convergence",
            e_exact.estimate
        );
    }

    #[test]
    fn deflation_wide_kappa_variance_vs_bias_multiseed() {
        // #2080 DEFINITIVE variance-vs-bias split for the wide-κ "10% low" verdict.
        //
        // The sibling `..._cg_convergence_discriminator` proves the residual is NOT
        // CG error (loose/tight/EXACT-Cholesky arms are byte-identical) and reports
        // gap ≈ 5.85 = 0.6σ against a std_err of 9.25 (≈16% of |logdet|=57.4). A
        // SINGLE-seed draw at 0.6σ cannot distinguish a structural split/quadrature
        // BIAS from an unlucky Hutchinson VARIANCE draw — the two have completely
        // different production fixes (fresh derivation vs. more probes / deeper
        // deflation / a control variate). This test settles it by AVERAGING the
        // estimate over K independent probe seeds while holding the deterministic
        // pieces fixed:
        //   • frozen Q (seed 555, rank 16) — term1 = tr(Qᵀ log(S/c) Q) is IDENTICAL
        //     across seeds, so it drops out of the seed-to-seed spread;
        //   • EXACT Cholesky solves — zero CG error, as the sibling established;
        //   • only the 32 Rademacher probes (and thus term2's Hutchinson draw) vary.
        // Hutchinson is UNBIASED over Rademacher probes, so
        //   E_seed[estimate] = k·log c + term1 + E_seed[term2]
        //                    → k·log c + tr_quad(Qᵀ·Q) + tr_quad(P·P)
        //                    = the QUADRATURE approximation of tr log S.
        // Hence bias_of_mean isolates the DETERMINISTIC (split+quadrature) error
        // with the Hutchinson variance averaged away as 1/√K. Verdict:
        //   |mean − logdet| ≲ 3·se_mean  ⇒  VARIANCE-dominated: the split+quadrature
        //     are unbiased at κ=1e8; the fix is variance reduction (more probes,
        //     deeper deflation rank, or a control variate), NOT a re-derivation.
        //   |mean − logdet| ≫ se_mean    ⇒  a genuine deterministic bias remains
        //     (quadrature under-resolution or a split defect) → derivation work.
        let dim = 96;
        let mut state = 2026u64;
        let lambdas: Vec<f64> = (0..dim)
            .map(|_| 10f64.powf(next_uniform(&mut state, -4.0, 4.0)))
            .collect();
        let (a, logdet) = spd_with_spectrum(dim, &lambdas, 4321);
        let lmin = lambdas.iter().cloned().fold(f64::INFINITY, f64::min);
        let lmax = lambdas.iter().cloned().fold(0.0f64, f64::max);
        let matvec = |v: ArrayView1<f64>| a.dot(&v);

        // Fixed frozen Q (identical to the sibling's), varied ONLY by probe seed.
        let k_seeds = 96usize;
        let mut ests = Vec::with_capacity(k_seeds);
        let mut internal_bars = Vec::with_capacity(k_seeds);
        // Guard the PRECONDITION this discriminator rests on: the K·m probe vectors
        // must be JOINTLY INDEPENDENT across the unit-spaced seeds. The former
        // per-probe RNG init `(seed + p)·γ` aliased them — a splitmix stream from
        // `x₀` and one from `x₀ + γ` are the same stream shifted, so
        // `(seed=9000+s, probe=p)` and `(9000+s−1, p+1)` were BIT-IDENTICAL and the
        // 96 seeds drew only ~128 distinct vectors of 96·32=3072. Averaging
        // correlated draws does not reduce variance as 1/√K, so `se_mean` collapsed
        // and a shared Hutchinson fluctuation was reported as a "5.57σ deterministic
        // bias". Fingerprint every probe's sign pattern (dim ≤ 128) and require near
        // all distinct, so an RNG regression that re-aliases the seeds fails HERE
        // rather than resurfacing as a phantom quadrature/split bias.
        let mut probe_fingerprints: std::collections::HashSet<u128> =
            std::collections::HashSet::new();
        for s in 0..k_seeds {
            let plan = RationalLogdetPlan::build(dim, 32, 9000 + s as u64, lmin, lmax, 1e-9)
                .expect("plan")
                .with_deflation(&matvec, 16, 3, 555);
            for probe in &plan.probes {
                let mut fp = 0u128;
                for (i, &x) in probe.iter().enumerate() {
                    if x > 0.0 {
                        fp |= 1u128 << i;
                    }
                }
                probe_fingerprints.insert(fp);
            }
            let e = evaluate_exact(&plan, &a);
            ests.push(e.estimate);
            internal_bars.push(e.std_err);
        }
        let total_pairs = k_seeds * 32;
        let distinct = probe_fingerprints.len();
        assert!(
            distinct as f64 > 0.95 * total_pairs as f64,
            "probe vectors must be jointly independent across seeds for this \
             variance-vs-bias split to be valid: only {distinct} distinct of \
             {total_pairs} (seed, probe) pairs — the RNG has re-aliased unit-spaced \
             seeds (expected ~{total_pairs}), so any reported σ is meaningless"
        );
        let n = ests.len() as f64;
        let mean = ests.iter().sum::<f64>() / n;
        let var = ests.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let sd = var.sqrt();
        let se_mean = sd / n.sqrt();
        let mean_internal_bar = internal_bars.iter().sum::<f64>() / n;
        let bias = mean - logdet;
        let bias_frac = bias.abs() / logdet.abs().max(1.0);
        let bias_sigma = bias.abs() / se_mean.max(1e-300);
        eprintln!(
            "wide-κ variance-vs-bias ({k_seeds} seeds, fixed Q, EXACT solves): exact={logdet:.6}\n  \
             mean={mean:.6}  bias={bias:+.6} ({bias_frac:.3e} rel, {bias_sigma:.2}σ of the mean)\n  \
             seed-to-seed sd={sd:.4}  se_mean={se_mean:.4}  ⟨internal std_err⟩={mean_internal_bar:.4}\n  \
             VERDICT: {}",
            if bias_sigma < 3.0 {
                "VARIANCE-dominated — split+quadrature UNBIASED at κ=1e8; fix = variance reduction (probes/rank/control-variate), NOT re-derivation"
            } else {
                "genuine DETERMINISTIC bias survives probe-averaging — quadrature/split derivation work needed"
            }
        );
        // Cross-check: the internal per-eval Hutchinson bar must PREDICT the
        // observed seed-to-seed spread (both estimate the same term2 variance);
        // a gross mismatch would mean the reported std_err is itself miscalibrated.
        assert!(
            (mean_internal_bar / sd).ln().abs() < 1.0,
            "internal std_err ({mean_internal_bar:.3}) must track the empirical seed spread ({sd:.3}) \
             within a factor e; a mismatch means the surrogate's error bar is miscalibrated"
        );
        // The definitive verdict for #2080's deflation lane: with the split and
        // quadrature deterministic and solves exact, the probe-averaged estimate is
        // an UNBIASED estimator of tr log S. If this holds, the single-seed "10%"
        // is variance and the sibling discriminator's STRUCTURAL framing is too
        // strong — the production lever is variance reduction, not a new derivation.
        assert!(
            bias_sigma < 3.0 || bias_frac < 0.02,
            "probe-averaged estimate is biased by {bias:+.4} ({bias_frac:.3e} rel, {bias_sigma:.2}σ): \
             deterministic split/quadrature bias survives — genuine derivation work, not variance"
        );
    }

    #[test]
    fn two_sided_deflation_drops_wide_kappa_std_err_below_two_percent() {
        // #2080 wide-κ VARIANCE-REDUCTION deliverable. The multiseed discriminator
        // established the surrogate is UNBIASED at κ=1e8 but too NOISY: the wide-κ
        // Hutchinson bar is ~14% of |logdet| with one-sided top deflation — too
        // loose for the outer REML to trust one evaluation. Root cause (see
        // `build_inverse_deflation_basis`): the bar is `√(2‖offdiag(P log(S/c) P)‖_F²)`,
        // and `log(S/c)`'s off-diagonal mass sits SYMMETRICALLY on both spectral
        // tails, so one-sided (top-only) deflation removes only half of it. Peeling
        // BOTH tails — the two-sided low-rank control variate — collapses the bar.
        //
        // This measures the bar three ways on IDENTICAL probes through the EXACT
        // dense-Cholesky estimator arm (so `std_err` reflects the ESTIMATOR variance,
        // not CG solve noise) and asserts the two-sided bar (a) falls below 2% of
        // |logdet|, and (b) beats ONE-sided deflation AT EQUAL TOTAL RANK — the
        // apples-to-apples proof that the win is the two-sidedness, not merely more
        // deflated columns. The value stays unbiased throughout (exact split for any
        // orthonormal Q), checked against the estimator's own 5σ bar.
        let dim = 96;
        let mut state = 2026u64;
        let lambdas: Vec<f64> = (0..dim)
            .map(|_| 10f64.powf(next_uniform(&mut state, -4.0, 4.0)))
            .collect();
        let (a, logdet) = spd_with_spectrum(dim, &lambdas, 4321);
        let lmin = lambdas.iter().cloned().fold(f64::INFINITY, f64::min);
        let lmax = lambdas.iter().cloned().fold(0.0f64, f64::max);
        let matvec = |v: ArrayView1<f64>| a.dot(&v);

        // Common 256-probe block (CRN); the three plans differ ONLY in the frozen Q.
        let base = RationalLogdetPlan::build(dim, 256, 17, lmin, lmax, 1e-9).expect("plan");
        let top16 = base.clone().with_deflation(&matvec, 16, 3, 555); // current wide-κ config
        let top64 = base.clone().with_deflation(&matvec, 64, 3, 555); // one-sided, EQUAL total rank
        let two = base
            .clone()
            .with_two_sided_deflation(&matvec, 32, 32, 3, 555, (1e-3, 5000))
            .expect("bottom-tail inverse iteration must converge"); // 32 top + 32 bottom

        let e16 = evaluate_exact(&top16, &a);
        let e64 = evaluate_exact(&top64, &a);
        let e2 = evaluate_exact(&two, &a);
        let f = |se: f64| se / logdet.abs().max(1.0);
        eprintln!(
            "wide-κ variance reduction (256 probes, EXACT estimator): |logdet|={:.4}\n  \
             top-only  r16 (current): std_err={:.4} ({:.4} of |ld|)\n  \
             top-only  r64 (eq-rank): std_err={:.4} ({:.4} of |ld|)\n  \
             two-sided 32+32:         std_err={:.4} ({:.4} of |ld|)  rel={:.4}\n  \
             => vs top-r16 {:.2}×, vs eq-rank top-r64 {:.2}×",
            logdet.abs(),
            e16.std_err,
            f(e16.std_err),
            e64.std_err,
            f(e64.std_err),
            e2.std_err,
            f(e2.std_err),
            (e2.estimate - logdet).abs() / logdet.abs().max(1.0),
            e16.std_err / e2.std_err.max(1e-300),
            e64.std_err / e2.std_err.max(1e-300),
        );
        assert_eq!(
            e2.deflation_basis.len(),
            64,
            "two-sided block must realise 32 top + 32 bottom orthonormal columns (got {})",
            e2.deflation_basis.len()
        );
        // (a) below the 2%-of-|logdet| production target.
        assert!(
            e2.std_err < 0.02 * logdet.abs(),
            "two-sided wide-κ std_err {:.4} must fall below 2% of |logdet| ({:.4})",
            e2.std_err,
            0.02 * logdet.abs()
        );
        // (b) beats ONE-sided deflation at EQUAL total rank (the two-sidedness is
        // the lever, not the column count) — calibrated ratio ≈ 2.75×, assert ≥ 2×.
        assert!(
            e2.std_err < 0.5 * e64.std_err,
            "two-sided ({:.4}) must beat equal-rank one-sided ({:.4}) by ≥2× — the win is \
             peeling BOTH tails, not merely deflating more columns",
            e2.std_err,
            e64.std_err
        );
        // Value stays unbiased (exact split for any orthonormal Q): the estimate is
        // within its own honest 5σ bar of the exact log-det.
        assert!(
            (e2.estimate - logdet).abs() < 5.0 * e2.std_err,
            "two-sided estimate {:.4} must stay within 5σ ({:.4}) of exact {:.4} — variance \
             reduction must not bias the value",
            e2.estimate,
            5.0 * e2.std_err,
            logdet
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
        let d_lambdas: Vec<f64> = (0..dim)
            .map(|_| next_uniform(&mut state, 0.1, 1.0))
            .collect();
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
        assert!(
            grad > 0.0,
            "SPD direction must increase log det, got {grad}"
        );
    }
}

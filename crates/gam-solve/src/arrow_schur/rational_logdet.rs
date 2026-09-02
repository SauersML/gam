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

/// Work and shape diagnostics for one evaluation of a frozen rational
/// log-determinant plan. These describe the exact shifted-solve ladder whose
/// value and derivative bundle were emitted together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RationalLogdetEvaluationMetrics {
    /// Total certified shifted-CG iterations across probe and deflation solves.
    pub cg_iterations: usize,
    /// Number of rational quadrature nodes in the frozen plan.
    pub node_count: usize,
    /// Number of frozen deflation directions actually realised.
    pub deflation_rank: usize,
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
    metrics: RationalLogdetEvaluationMetrics,
}

impl RationalLogdetDerivativeBundle {
    /// Diagnostics for the evaluation that produced this derivative bundle.
    /// Keeping them on the bundle makes it impossible to report work from one
    /// operator alongside the derivative of another.
    #[must_use]
    pub fn evaluation_metrics(&self) -> RationalLogdetEvaluationMetrics {
        self.metrics
    }

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

    /// [`Self::with_two_sided_deflation`] with the same diagonal preconditioner
    /// the evaluations use.
    ///
    /// The bottom-tail basis comes from INVERSE iteration — plain CG on the
    /// UNSHIFTED operator at full `κ` — which is the single worst-conditioned
    /// solve family in the whole surrogate. Preconditioning it is not optional
    /// bookkeeping: without it, a wide-diagonal border makes the deflation ladder
    /// (which doubles the rank until the error bar clears) spend its entire
    /// budget inside `build_inverse_deflation_basis` (#2576). The basis only
    /// steers variance reduction, so this cannot bias the value either way.
    pub fn with_two_sided_deflation_preconditioned(
        mut self,
        matvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
        preconditioner: &ShiftedDiagonalPreconditioner,
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
            preconditioner,
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
        self.evaluate_preconditioned(
            matvec,
            &IDENTITY_SHIFT_PRECONDITIONER,
            cg_rel_tol,
            cg_max_iters,
        )
    }

    /// Evaluate with a diagonal preconditioner on the shifted solves.
    ///
    /// The plan is the STATISTICAL functional — probes, quadrature nodes,
    /// deflation basis — and the shifted inverse is the NUMERICAL means of
    /// evaluating it. A preconditioner changes only the second, so the value
    /// this returns is the same function of the operator that
    /// [`Self::evaluate`] returns, converged to the same certified residual,
    /// and its [`Self::directional_derivative`] is still that value's exact
    /// gradient. What changes is how many iterations each shifted solve takes:
    /// on an operator whose diagonal spans orders of magnitude — the
    /// overcomplete arrow border's atom firing counts — that is the difference
    /// between converging and running to the iteration cap (#2576).
    pub fn evaluate_preconditioned(
        &self,
        matvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
        preconditioner: &ShiftedDiagonalPreconditioner,
        cg_rel_tol: f64,
        cg_max_iters: usize,
    ) -> Option<RationalLogdetEval> {
        let solve = |shift: f64, rhs: &Array1<f64>, warm: &Array1<f64>| {
            shifted_pcg(
                matvec,
                preconditioner,
                shift,
                rhs,
                warm,
                cg_rel_tol,
                cg_max_iters,
            )
        };
        self.evaluate_with_shifted_solver(&solve)
    }

    /// Evaluate this frozen rational functional with a caller-owned shifted
    /// linear solver.
    ///
    /// The solver must return the converged solution of
    /// `(S + shift·I)y = rhs` and an iteration count. `warm` is the solution
    /// from the preceding, larger shift for the same right-hand side. Separating
    /// the statistical functional (fixed probes, nodes, deflation basis, value
    /// assembly, and derivative bundle) from the numerical inverse lets callers
    /// use a structured preconditioner without changing the criterion. In
    /// particular, an exact-observed-information operator can use its positive
    /// majorizer only as a preconditioner; storage strategy can no longer require
    /// a different log-determinant definition.
    pub fn evaluate_with_shifted_solver(
        &self,
        solve: &(impl Fn(
            f64,
            &Array1<f64>,
            &Array1<f64>,
        ) -> Option<(Array1<f64>, usize)>
                  + Sync),
    ) -> Option<RationalLogdetEval> {
        // Ladder: descending shift (warm starts carry per vector across shifts).
        let mut order: Vec<usize> = (0..self.nodes.len()).collect();
        order.sort_by(|&a, &b| {
            self.nodes[b]
                .0
                .partial_cmp(&self.nodes[a].0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // FROZEN top-subspace deflation basis Q (empty without a DeflationSpec).
        // Built once at plan creation from the operator at the plan's rho; reused
        // verbatim here so the surrogate is one fixed-Q function of rho.
        let basis: &[Array1<f64>] = self
            .deflation
            .as_ref()
            .map(|d| d.basis.as_slice())
            .unwrap_or(&[]);

        // Deflation-projected probes u_j = P v_j (raw probes without a basis).
        let probes_proj = self.projected_probes(basis);

        // Both solve families use the same injected solver and shift ordering.
        let (shifted, iters_probe) =
            solve_shift_ladder_with(solve, &self.nodes, &order, &probes_proj)?;
        let (deflation_solves, iters_basis) = if basis.is_empty() {
            (Vec::new(), 0)
        } else {
            solve_shift_ladder_with(solve, &self.nodes, &order, basis)?
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
        let metrics = RationalLogdetEvaluationMetrics {
            cg_iterations: eval.cg_iterations,
            node_count: self.nodes.len(),
            deflation_rank: eval.deflation_basis.len(),
        };
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
        Some(RationalLogdetDerivativeBundle { vectors, metrics })
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
/// The no-op preconditioner. `shifted_pcg` under it is bit-for-bit the plain CG
/// this module ran before #2576 — `z == r`, so `rᵀz` is `rᵀr` and the direction
/// update is `p ← r + βp` exactly — which is why there is ONE shifted-solve
/// implementation and one convergence certificate here rather than two that
/// could drift apart.
pub(crate) const IDENTITY_SHIFT_PRECONDITIONER: ShiftedDiagonalPreconditioner =
    ShiftedDiagonalPreconditioner { diagonal: None };

/// Diagonal preconditioner for the SHIFTED systems `(A + tI)` a rational
/// log-determinant plan solves.
///
/// The caller supplies the *unshifted* operator's diagonal; the shift is added
/// per solve, exactly as it is added to the operator. That single fact is what
/// makes one diagonal serve the whole shift ladder: `diag(A + tI) = diag(A) + t`.
///
/// `diagonal: None` is the identity, i.e. the unpreconditioned iteration. On the
/// overcomplete arrow border the supplied diagonal is the shared block's own —
/// the atom FIRING-COUNT distribution, orders of magnitude wide — and it is
/// exactly the spread that stalls an unpreconditioned CG (#2576).
#[derive(Debug, Clone)]
pub struct ShiftedDiagonalPreconditioner {
    diagonal: Option<Array1<f64>>,
}

impl ShiftedDiagonalPreconditioner {
    /// Build from the unshifted operator's diagonal. A non-finite or
    /// non-positive entry means there is no usable scale, and the identity is
    /// returned rather than a fabricated one: the iteration is then exactly the
    /// unpreconditioned one, which is still correct for SPD `A`.
    pub fn from_operator_diagonal(diagonal: &Array1<f64>) -> Self {
        if diagonal
            .iter()
            .any(|value| !(value.is_finite() && *value > 0.0))
        {
            return Self { diagonal: None };
        }
        Self {
            diagonal: Some(diagonal.clone()),
        }
    }

    pub fn identity() -> Self {
        Self { diagonal: None }
    }

    fn apply(&self, residual: &Array1<f64>, shift: f64) -> Array1<f64> {
        match &self.diagonal {
            Some(diagonal) => {
                let mut out = residual.clone();
                for (value, scale) in out.iter_mut().zip(diagonal.iter()) {
                    let denominator = scale + shift;
                    // `from_operator_diagonal` admits only strictly positive
                    // finite entries and the plan's shifts are non-negative, so
                    // this can fail only on overflow. Leaving that entry
                    // unscaled keeps `M` symmetric positive definite (unit on
                    // that coordinate) instead of emitting an infinity that
                    // would break the recurrence.
                    if denominator > 0.0 && denominator.is_finite() {
                        *value /= denominator;
                    }
                }
                out
            }
            None => residual.clone(),
        }
    }
}

/// Preconditioned shifted CG. Identical certificate, restart policy and
/// refusal contract as [`shifted_cg`] — the preconditioner steers the search
/// directions and nothing else, so the returned iterate still satisfies the
/// SAME true-residual / backward-error test before it is accepted.
fn shifted_pcg(
    matvec: &(impl Fn(ArrayView1<f64>) -> Array1<f64> + Sync),
    preconditioner: &ShiftedDiagonalPreconditioner,
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
    let mut z = preconditioner.apply(&r, t);
    let mut p = z.clone();
    // `rs` is the PRECONDITIONED inner product `rᵀz` that drives the recurrence;
    // `residual_norm_sq` is the plain `rᵀr` every convergence test below reads.
    // Under the identity preconditioner they coincide, so this is the same
    // iteration `shifted_cg` always ran.
    let mut rs = r.dot(&z);
    let mut residual_norm_sq = r.dot(&r);
    if !(rs.is_finite() && residual_norm_sq.is_finite()) {
        return None;
    }
    let tol = rel_tol * b_norm;
    let mut iters = 0usize;
    let mut observed_operator_norm = 0.0_f64;
    loop {
        if residual_norm_sq.sqrt() <= tol {
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
            let backward_error_certified =
                if observed_operator_norm > 0.0 && y_norm > 0.0 {
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
            // The restart's `true_rs` is not stored: control falls straight
            // into the matvec below, which recomputes `residual_norm_sq` from
            // the updated residual before the next convergence test reads it.
            r = true_residual;
            z = preconditioner.apply(&r, t);
            rs = r.dot(&z);
            if !rs.is_finite() {
                return None;
            }
            p = z.clone();
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
        // `rs = rᵀM⁻¹r` is zero only when `r` is, and a zero residual reaches
        // the certified-exit branch above (`tol > 0` always). Reaching here with
        // `rs == 0` means round-off destroyed the SPD-by-construction
        // preconditioned inner product; the update would divide by zero.
        if rs == 0.0 {
            return None;
        }
        let alpha = rs / denom;
        y.scaled_add(alpha, &p);
        r.scaled_add(-alpha, &ap);
        residual_norm_sq = r.dot(&r);
        z = preconditioner.apply(&r, t);
        let rs_new = r.dot(&z);
        if !(rs_new.is_finite() && residual_norm_sq.is_finite()) {
            return None;
        }
        p = &z + &(&p * (rs_new / rs));
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
    preconditioner: &ShiftedDiagonalPreconditioner,
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
            .map(|c| {
                shifted_pcg(
                    matvec,
                    preconditioner,
                    0.0,
                    c,
                    &zero,
                    cg_rel_tol,
                    cg_max_iters,
                )
                .map(|(y, _)| y)
            })
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
fn solve_shift_ladder_with(
    solve: &(impl Fn(
        f64,
        &Array1<f64>,
        &Array1<f64>,
    ) -> Option<(Array1<f64>, usize)>
              + Sync),
    nodes: &[(f64, f64)],
    order: &[usize],
    vectors: &[Array1<f64>],
) -> Option<(Vec<Vec<Array1<f64>>>, usize)> {
    let m = vectors.len();
    let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
    let mut solves: Vec<Vec<Array1<f64>>> = vec![Vec::with_capacity(m); nodes.len()];
    let mut warm: Vec<Array1<f64>> = vec![Array1::zeros(dim); m];
    let mut total = 0usize;
    for &ell in order {
        let (shift, _) = nodes[ell];
        let mut per = Vec::with_capacity(m);
        for (j, rhs) in vectors.iter().enumerate() {
            let (solution, iters) = solve(shift, rhs, &warm[j])?;
            if solution.len() != dim || solution.iter().any(|value| !value.is_finite()) {
                return None;
            }
            total = total.checked_add(iters)?;
            warm[j] = solution.clone();
            per.push(solution);
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
            shifted_pcg(&matvec, &IDENTITY_SHIFT_PRECONDITIONER, 0.0, &b, &zero, 1.0e-12, 1)
                .is_none(),
            "one CG step cannot solve a two-eigenvalue system to 1e-12; the \
             iteration-capped last iterate must be refused"
        );
        let (solved, iterations) =
            shifted_pcg(&matvec, &IDENTITY_SHIFT_PRECONDITIONER, 0.0, &b, &zero, 1.0e-12, 2)
                .expect("two-dimensional SPD CG must converge in at most two steps");
        let residual = &b - &matvec(solved.view());
        assert!(
            residual.dot(&residual).sqrt() <= 1.0e-12 * b.dot(&b).sqrt(),
            "returned shifted solve must satisfy its true-residual contract"
        );
        assert_eq!(iterations, 2);
    }

}

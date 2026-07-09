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
    /// `y_{jℓ} = (S + t_ℓ I)⁻¹ v_j`, outer index `ℓ` (node), inner `j` (probe).
    pub shifted_solves: Vec<Vec<Array1<f64>>>,
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
        // h = 2π·d_min/ln(1/tol); truncation window padded two decades past
        // the bracket (the tail beyond contributes O(t_lo/λ_min) ≪ tol).
        let c = (lambda_min * lambda_max).sqrt();
        let t_lo = (lambda_min / c) * 1e-2;
        let t_hi = (lambda_max / c) * 1e2;
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
        })
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
        let mut shifted: Vec<Vec<Array1<f64>>> =
            vec![Vec::with_capacity(m); self.nodes.len()];
        let mut total_iters = 0usize;
        // Ladder: descending t. Warm starts carry per-probe across shifts.
        let mut order: Vec<usize> = (0..self.nodes.len()).collect();
        order.sort_by(|&a, &b| {
            self.nodes[b]
                .0
                .partial_cmp(&self.nodes[a].0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut warm: Vec<Array1<f64>> = vec![Array1::zeros(self.dim); m];
        for &ell in &order {
            let (t, _) = self.nodes[ell];
            let mut per_probe = Vec::with_capacity(m);
            for (j, v) in self.probes.iter().enumerate() {
                let (y, iters) = shifted_cg(matvec, t, v, &warm[j], cg_rel_tol, cg_max_iters)?;
                total_iters += iters;
                warm[j] = y.clone();
                per_probe.push(y);
            }
            shifted[ell] = per_probe;
        }
        // Per-probe estimates: e_j = k·ln c + Σ_ℓ w_ℓ·(k/(c+t_ℓ) − v_jᵀ y_{jℓ});
        // the surrogate is their mean, the Hutchinson error bar their spread.
        let mut per_probe = vec![k * self.log_center; m];
        for (ell, &(t, w)) in self.nodes.iter().enumerate() {
            let reference = k / (self.center + t);
            for (j, v) in self.probes.iter().enumerate() {
                per_probe[j] += w * (reference - v.dot(&shifted[ell][j]));
            }
        }
        let estimate = per_probe.iter().sum::<f64>() / m as f64;
        let std_err = if m > 1 {
            let var = per_probe
                .iter()
                .map(|e| (e - estimate) * (e - estimate))
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
        let mut acc = 0.0;
        for (ell, &(_, w)) in self.nodes.iter().enumerate() {
            for y in &eval.shifted_solves[ell] {
                let dy = dmatvec(y.view());
                acc += w * y.dot(&dy);
            }
        }
        acc /= m;
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
}

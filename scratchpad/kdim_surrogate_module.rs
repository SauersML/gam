//! #1033 — k-dim sufficient-statistic basin ranking for the REML seed grid.
//!
//! The seed grid (`crate::seeding::select_objective_seed_on_log_lambda_grid`)
//! ranks candidate log-smoothing vectors `ρ` by the REML/LAML objective `V(ρ)`.
//! Today every candidate pays a full-n inner P-IRLS solve (#1575: ~20-32 such
//! solves before the optimizer even starts). This module summarizes a converged
//! reference point `ρ₀` by **k-dim sufficient statistics** — the objective value
//! `V₀`, its gradient `g = ∂V/∂ρ|ρ₀` (length `k`) and Hessian
//! `H = ∂²V/∂ρ²|ρ₀` (`k×k`) — and ranks NEARBY candidates by the exact
//! second-order Taylor model the outer Newton itself uses:
//!
//! ```text
//!   Ṽ(ρ₀ + δ) = V₀ + gᵀδ + ½ δᵀ H δ
//! ```
//!
//! An interior candidate within the trust region is then ranked for `O(k²)`
//! flops instead of an `O(n·p²·iters)` solve.
//!
//! ## Bit-safety / quality-guard contract (#1266 / #1464 / #1548 / #1426)
//!
//! The surrogate is a **ranking pre-filter, never a basin filter**:
//! * FAR candidates — the over-smoothing saturation corner (#1266), the
//!   null-space keep corner (#1548) and the collapsing-kernel basin (#1464) —
//!   sit outside any sane trust radius, so [`BasinReference::within_trust`]
//!   reports `false` for them and [`rank_candidates`] RETAINS them (flagged) so
//!   the caller still evaluates them at full `n`. The surrogate can never drop a
//!   corner probe.
//! * The adopted seed is always verified by a real full-n solve before the
//!   optimizer runs, so the #1426 λ→0 trap (a capped solve reporting a
//!   spuriously-low cost) is untouched — the surrogate only reorders which
//!   candidates are tried, it never reports a fit's accepted cost.
//!
//! The numeric core ([`surrogate_cost`], [`within_trust`]) is intentionally on
//! plain `&[f64]` slices and is validated bit-for-bit against an independent
//! brute-force quadratic and a non-quadratic Richardson check (see the unit
//! tests, mirrored from `scratchpad/kdim_surrogate_standalone.rs`).

use ndarray::{Array1, Array2};

/// k-dim sufficient statistics summarizing the REML objective basin around a
/// converged reference `ρ₀`. All fields are `O(k)`/`O(k²)` — no n-sized state.
#[derive(Clone, Debug)]
pub(crate) struct BasinReference {
    /// Reference point `ρ₀` (length `k`).
    pub(crate) rho0: Array1<f64>,
    /// Objective value `V(ρ₀)`.
    pub(crate) v0: f64,
    /// Gradient `g = ∂V/∂ρ|ρ₀` (length `k`).
    pub(crate) grad: Array1<f64>,
    /// Hessian `H = ∂²V/∂ρ²|ρ₀` (`k×k`, symmetric).
    pub(crate) hess: Array2<f64>,
}

/// One ranked seed candidate. `within_trust=false` marks a candidate outside the
/// Taylor trust region (a far corner) that the caller MUST still evaluate at full
/// `n`; it is never dropped.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RankedCandidate {
    /// Index into the caller's candidate list.
    pub(crate) index: usize,
    /// Second-order surrogate cost `Ṽ(ρ)`.
    pub(crate) surrogate_cost: f64,
    /// Whether `ρ` lies within the L∞ trust radius of the reference.
    pub(crate) within_trust: bool,
}

/// Second-order Taylor model of `V` at `ρ₀ + δ`:
/// `Ṽ = V₀ + gᵀδ + ½ δᵀ H δ`. `grad` is length `k`, `hess_rowmajor` is the
/// `k×k` Hessian in row-major order (assumed symmetric), `delta = ρ − ρ₀`.
fn surrogate_cost(v0: f64, grad: &[f64], hess_rowmajor: &[f64], delta: &[f64]) -> f64 {
    let k = grad.len();
    assert_eq!(delta.len(), k);
    assert_eq!(hess_rowmajor.len(), k * k);
    let mut lin = 0.0_f64;
    for i in 0..k {
        lin += grad[i] * delta[i];
    }
    // ½ δᵀ H δ
    let mut quad = 0.0_f64;
    for i in 0..k {
        let row = &hess_rowmajor[i * k..i * k + k];
        let mut hd_i = 0.0_f64;
        for (j, &h_ij) in row.iter().enumerate() {
            hd_i += h_ij * delta[j];
        }
        quad += delta[i] * hd_i;
    }
    v0 + lin + 0.5 * quad
}

/// L∞ trust check: the local Taylor model is trusted only within `radius` of
/// `ρ₀` in every coordinate. Far candidates (the #1266/#1548/#1464 corners) fail
/// this and must be evaluated at full `n` by the caller.
fn within_trust(delta: &[f64], radius: f64) -> bool {
    delta.iter().all(|d| d.abs() <= radius)
}

impl BasinReference {
    /// Build a reference from k-dim sufficient statistics. `hess` must be `k×k`
    /// with `k = rho0.len() = grad.len()`.
    pub(crate) fn new(rho0: Array1<f64>, v0: f64, grad: Array1<f64>, hess: Array2<f64>) -> Self {
        assert_eq!(grad.len(), rho0.len());
        assert_eq!(hess.nrows(), rho0.len());
        assert_eq!(hess.ncols(), rho0.len());
        Self {
            rho0,
            v0,
            grad,
            hess,
        }
    }

    /// Second-order surrogate cost at `rho` (full ρ vector, not a delta).
    pub(crate) fn surrogate_cost_at(&self, rho: &Array1<f64>) -> f64 {
        let delta: Vec<f64> = rho
            .iter()
            .zip(self.rho0.iter())
            .map(|(&r, &r0)| r - r0)
            .collect();
        // Row-major contiguous copy of H (ndarray default layout is row-major,
        // but a view may be non-standard; copy to guarantee the slice contract).
        let hess_rm: Vec<f64> = self.hess.iter().copied().collect();
        let grad_slice: Vec<f64> = self.grad.iter().copied().collect();
        surrogate_cost(self.v0, &grad_slice, &hess_rm, &delta)
    }

    /// Whether `rho` lies within the L∞ `radius` trust region of the reference.
    pub(crate) fn within_trust_at(&self, rho: &Array1<f64>, radius: f64) -> bool {
        let delta: Vec<f64> = rho
            .iter()
            .zip(self.rho0.iter())
            .map(|(&r, &r0)| r - r0)
            .collect();
        within_trust(&delta, radius)
    }
}

/// Rank candidate ρ-points by surrogate cost (ascending). Out-of-trust
/// candidates are RETAINED (flagged `within_trust=false`) so the caller still
/// evaluates them at full `n`; the surrogate is a ranking pre-filter, never a
/// basin filter (#1266/#1464/#1548 corners are always preserved).
pub(crate) fn rank_candidates(
    reference: &BasinReference,
    candidates: &[Array1<f64>],
    trust_radius: f64,
) -> Vec<RankedCandidate> {
    let mut ranked: Vec<RankedCandidate> = candidates
        .iter()
        .enumerate()
        .map(|(index, rho)| RankedCandidate {
            index,
            surrogate_cost: reference.surrogate_cost_at(rho),
            within_trust: reference.within_trust_at(rho, trust_radius),
        })
        .collect();
    ranked.sort_by(|a, b| {
        a.surrogate_cost
            .partial_cmp(&b.surrogate_cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent brute-force quadratic evaluator (different code path than the
    // shipped surrogate) so the equality tests are not tautological.
    fn brute_quadratic(v0: f64, g: &[f64], h: &[f64], delta: &[f64]) -> f64 {
        let k = g.len();
        let mut acc = v0;
        for i in 0..k {
            acc += g[i] * delta[i];
        }
        for i in 0..k {
            for j in 0..k {
                acc += 0.5 * delta[i] * h[i * k + j] * delta[j];
            }
        }
        acc
    }

    fn reference(v0: f64, rho0: &[f64], g: &[f64], h_rowmajor: &[f64]) -> BasinReference {
        let k = rho0.len();
        BasinReference::new(
            Array1::from(rho0.to_vec()),
            v0,
            Array1::from(g.to_vec()),
            Array2::from_shape_vec((k, k), h_rowmajor.to_vec()).unwrap(),
        )
    }

    #[test]
    fn surrogate_is_exact_on_a_known_quadratic_1033() {
        let v0 = 1.5;
        let g = [0.3, -1.2, 0.7];
        let h = [2.0, 0.5, -0.1, 0.5, 1.3, 0.2, -0.1, 0.2, 0.9];
        let rho0 = [0.0, 0.0, 0.0];
        let r = reference(v0, &rho0, &g, &h);
        for delta in [
            vec![0.0, 0.0, 0.0],
            vec![1.0, -2.0, 0.5],
            vec![-3.0, 4.0, -1.0],
            vec![0.25, 0.25, 0.25],
        ] {
            let rho = Array1::from(delta.clone());
            let s = r.surrogate_cost_at(&rho);
            let b = brute_quadratic(v0, &g, &h, &delta);
            assert!((s - b).abs() <= 1e-12, "delta={delta:?} s={s} b={b}");
        }
    }

    #[test]
    fn surrogate_minimizer_is_minus_hinv_g_1033() {
        let v0 = 0.0;
        let g = [2.0, -4.0, 8.0];
        let h = [2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0];
        let rho0 = [0.0, 0.0, 0.0];
        let r = reference(v0, &rho0, &g, &h);
        let dstar = [-1.0, 1.0, -1.0];
        let smin = r.surrogate_cost_at(&Array1::from(dstar.to_vec()));
        for eps in [0.1_f64, -0.1, 0.3, -0.3] {
            for axis in 0..3 {
                let mut d = dstar;
                d[axis] += eps;
                let s = r.surrogate_cost_at(&Array1::from(d.to_vec()));
                assert!(s > smin - 1e-15, "axis {axis} eps {eps}: {s} !> {smin}");
            }
        }
    }

    #[test]
    fn ranking_matches_true_objective_on_a_quadratic_1033() {
        let v0 = -2.0;
        let g = [0.5, 0.5];
        let h = [3.0, 1.0, 1.0, 2.0];
        let rho0 = [0.0, 0.0];
        let r = reference(v0, &rho0, &g, &h);
        let raw = [
            vec![0.1, 0.1],
            vec![-0.5, 0.2],
            vec![0.3, -0.4],
            vec![0.0, 0.0],
            vec![-0.2, -0.2],
        ];
        let cands: Vec<Array1<f64>> = raw.iter().cloned().map(Array1::from).collect();
        let ranked = rank_candidates(&r, &cands, 10.0);
        let mut expected: Vec<(usize, f64)> = raw
            .iter()
            .enumerate()
            .map(|(i, d)| (i, brute_quadratic(v0, &g, &h, d)))
            .collect();
        expected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let got: Vec<usize> = ranked.iter().map(|r| r.index).collect();
        let exp: Vec<usize> = expected.iter().map(|e| e.0).collect();
        assert_eq!(got, exp, "ranking order mismatch");
    }

    #[test]
    fn trust_flags_far_corners_out_and_never_drops_them_1033() {
        let radius = 3.0;
        let g = [0.0, 0.0, 0.0];
        let h = [0.0; 9];
        let rho0 = [0.0, 0.0, 0.0];
        let r = reference(0.0, &rho0, &g, &h);
        assert!(r.within_trust_at(&Array1::from(vec![3.0, -1.0, 0.0]), radius));
        assert!(!r.within_trust_at(&Array1::from(vec![12.0, 0.0, 0.0]), radius)); // #1266
        assert!(!r.within_trust_at(&Array1::from(vec![0.0, -9.0, 0.0]), radius)); // #1548
        let cands: Vec<Array1<f64>> = [vec![1.0, 0.0, 0.0], vec![12.0, 0.0, 0.0]]
            .iter()
            .cloned()
            .map(Array1::from)
            .collect();
        let ranked = rank_candidates(&r, &cands, radius);
        assert_eq!(ranked.len(), 2, "no candidate dropped");
        let far = ranked.iter().find(|c| c.index == 1).unwrap();
        assert!(!far.within_trust, "far corner must be flagged out-of-trust");
    }

    #[test]
    fn second_order_taylor_accuracy_on_a_nonquadratic_1033() {
        // V(ρ)=Σ exp(ρ_i): genuinely non-quadratic. At ρ₀=0: V₀=k, g_i=1, H=I.
        // 2nd-order error is O(‖δ‖³): halving δ shrinks error ~8× (a 1st-order
        // model would only shrink ~4×), proving this is a real 2nd-order model.
        let k = 3usize;
        let v0 = k as f64;
        let g = vec![1.0_f64; k];
        let mut h = vec![0.0_f64; k * k];
        for i in 0..k {
            h[i * k + i] = 1.0;
        }
        let rho0 = vec![0.0_f64; k];
        let r = reference(v0, &rho0, &g, &h);
        let true_v = |delta: &[f64]| -> f64 { delta.iter().map(|d| d.exp()).sum() };
        let base = [0.6_f64, -0.4, 0.5];
        let err = |scale: f64| -> f64 {
            let d: Vec<f64> = base.iter().map(|b| b * scale).collect();
            (r.surrogate_cost_at(&Array1::from(d.clone())) - true_v(&d)).abs()
        };
        let r1 = err(1.0) / err(0.5);
        let r2 = err(0.5) / err(0.25);
        assert!(r1 > 6.0 && r1 < 10.0, "ratio1 {r1} not ≈8");
        assert!(r2 > 6.0 && r2 < 10.0, "ratio2 {r2} not ≈8");
    }
}

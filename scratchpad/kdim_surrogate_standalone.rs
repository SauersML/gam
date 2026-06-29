// Standalone validation of the #1033 k-dim sufficient-statistic surrogate core.
// Compile + run with:  rustc --test -O scratchpad/kdim_surrogate_standalone.rs -o /tmp/kdim_test && /tmp/kdim_test
// The functions below are the EXACT numeric core that ships in
// crates/gam-solve/src/reml/kdim_surrogate.rs (plain &[f64] slices, no deps),
// so these tests validate the shipped arithmetic, not a paraphrase.

// ---- shipped core (keep byte-identical to reml/kdim_surrogate.rs) ----

/// Second-order Taylor model of the REML/LAML objective V at ρ = ρ₀ + δ:
///   Ṽ(ρ₀+δ) = V₀ + gᵀδ + ½ δᵀ H δ
/// `grad` is g = ∂V/∂ρ|ρ₀ (length k); `hess_rowmajor` is H = ∂²V/∂ρ²|ρ₀
/// (k×k, row-major, assumed symmetric); `delta` is ρ − ρ₀ (length k).
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
        for j in 0..k {
            hd_i += row[j] * delta[j];
        }
        quad += delta[i] * hd_i;
    }
    v0 + lin + 0.5 * quad
}

/// L∞ trust check: the local Taylor model is only trusted within `radius` of ρ₀
/// in every coordinate. Far candidates (the #1266/#1548/#1464 corners) fail this
/// and MUST be evaluated full-n by the caller — the surrogate never suppresses
/// a corner probe.
fn within_trust(delta: &[f64], radius: f64) -> bool {
    delta.iter().all(|d| d.abs() <= radius)
}

#[derive(Clone, Debug, PartialEq)]
struct RankedCandidate {
    index: usize,
    surrogate_cost: f64,
    within_trust: bool,
}

/// Rank candidate deltas by surrogate cost (ascending). Out-of-trust candidates
/// are RETAINED (flagged `within_trust=false`) so the caller still full-evaluates
/// them; the surrogate is a ranking pre-filter, never a basin filter.
fn rank_candidates(
    v0: f64,
    grad: &[f64],
    hess_rowmajor: &[f64],
    candidate_deltas: &[Vec<f64>],
    trust_radius: f64,
) -> Vec<RankedCandidate> {
    let mut ranked: Vec<RankedCandidate> = candidate_deltas
        .iter()
        .enumerate()
        .map(|(index, d)| RankedCandidate {
            index,
            surrogate_cost: surrogate_cost(v0, grad, hess_rowmajor, d),
            within_trust: within_trust(d, trust_radius),
        })
        .collect();
    ranked.sort_by(|a, b| {
        a.surrogate_cost
            .partial_cmp(&b.surrogate_cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
}

// ---- standalone tests ----

#[cfg(test)]
mod tests {
    use super::*;

    // Independent brute-force quadratic evaluator (different code path than the
    // shipped surrogate) so the test is not a tautology.
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

    #[test]
    fn surrogate_is_exact_on_a_known_quadratic() {
        // k=3, asymmetric-looking but symmetric H.
        let v0 = 1.5;
        let g = [0.3, -1.2, 0.7];
        let h = [
            2.0, 0.5, -0.1, //
            0.5, 1.3, 0.2, //
            -0.1, 0.2, 0.9,
        ];
        for delta in [
            vec![0.0, 0.0, 0.0],
            vec![1.0, -2.0, 0.5],
            vec![-3.0, 4.0, -1.0],
            vec![0.25, 0.25, 0.25],
        ] {
            let s = surrogate_cost(v0, &g, &h, &delta);
            let b = brute_quadratic(v0, &g, &h, &delta);
            assert!((s - b).abs() <= 1e-12, "delta={delta:?} s={s} b={b}");
        }
    }

    #[test]
    fn surrogate_minimizer_is_minus_hinv_g() {
        // For V = v0 + gᵀδ + ½δᵀHδ with SPD H, the min is at δ* = -H⁻¹ g and any
        // perturbation off δ* raises the surrogate. H = diag(2,4,8) → δ* trivial.
        let v0 = 0.0;
        let g = [2.0, -4.0, 8.0];
        let h = [
            2.0, 0.0, 0.0, //
            0.0, 4.0, 0.0, //
            0.0, 0.0, 8.0,
        ];
        let dstar = vec![-1.0, 1.0, -1.0]; // -g_i / h_ii
        let smin = surrogate_cost(v0, &g, &h, &dstar);
        for eps in [0.1_f64, -0.1, 0.3, -0.3] {
            for axis in 0..3 {
                let mut d = dstar.clone();
                d[axis] += eps;
                let s = surrogate_cost(v0, &g, &h, &d);
                assert!(s > smin - 1e-15, "perturb axis {axis} eps {eps}: {s} !> {smin}");
            }
        }
    }

    #[test]
    fn ranking_matches_true_objective_on_a_quadratic() {
        // On a quadratic the surrogate IS exact, so its ranking must equal the
        // ranking by the (independent) true objective.
        let v0 = -2.0;
        let g = [0.5, 0.5];
        let h = [3.0, 1.0, 1.0, 2.0];
        let cands = vec![
            vec![0.1, 0.1],
            vec![-0.5, 0.2],
            vec![0.3, -0.4],
            vec![0.0, 0.0],
            vec![-0.2, -0.2],
        ];
        let ranked = rank_candidates(v0, &g, &h, &cands, 10.0);
        // independent expected order
        let mut expected: Vec<(usize, f64)> = cands
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
    fn trust_flags_far_corners_out() {
        let radius = 3.0;
        // interior ±3 step: in trust; saturation corner (far): out.
        assert!(within_trust(&[3.0, -1.0, 0.0], radius));
        assert!(within_trust(&[-3.0, 3.0, 2.999], radius));
        assert!(!within_trust(&[12.0, 0.0, 0.0], radius)); // #1266 saturation corner
        assert!(!within_trust(&[0.0, -9.0, 0.0], radius)); // #1548 keep corner
        // ranking retains the far candidate (flagged), never drops it.
        let g = [0.0, 0.0, 0.0];
        let h = [0.0; 9];
        let cands = vec![vec![1.0, 0.0, 0.0], vec![12.0, 0.0, 0.0]];
        let ranked = rank_candidates(0.0, &g, &h, &cands, radius);
        assert_eq!(ranked.len(), 2, "no candidate dropped");
        let far = ranked.iter().find(|r| r.index == 1).unwrap();
        assert!(!far.within_trust, "far corner must be flagged out-of-trust");
    }

    #[test]
    fn second_order_taylor_accuracy_on_a_nonquadratic() {
        // V(ρ) = Σ exp(ρ_i): genuinely non-quadratic. At ρ₀=0, V₀=k, g_i=1,
        // H=I. The 2nd-order surrogate error must be O(‖δ‖³): halving δ shrinks
        // the error by ~8×. This proves it's a real 2nd-order model, not a
        // tautology (a 1st-order model would only shrink by ~4×).
        let k = 3usize;
        let v0 = k as f64; // Σ exp(0)
        let g = vec![1.0_f64; k]; // d/dρ exp(ρ) = exp(ρ) = 1 at 0
        let mut h = vec![0.0_f64; k * k]; // d²/dρ² exp = exp = 1 at 0 (diagonal)
        for i in 0..k {
            h[i * k + i] = 1.0;
        }
        let true_v = |delta: &[f64]| -> f64 { delta.iter().map(|d| d.exp()).sum() };
        let base = [0.6_f64, -0.4, 0.5];
        let err = |scale: f64| -> f64 {
            let d: Vec<f64> = base.iter().map(|b| b * scale).collect();
            (surrogate_cost(v0, &g, &h, &d) - true_v(&d)).abs()
        };
        let e1 = err(1.0);
        let e_half = err(0.5);
        let e_quarter = err(0.25);
        // ratio ≈ 8 for a true 2nd-order model (cubic leading error).
        let r1 = e1 / e_half;
        let r2 = e_half / e_quarter;
        assert!(r1 > 6.0 && r1 < 10.0, "halving ratio1 {r1} not ≈8 (e1={e1}, e_half={e_half})");
        assert!(r2 > 6.0 && r2 < 10.0, "halving ratio2 {r2} not ≈8");
    }
}

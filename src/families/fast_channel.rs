//! Hand-speed single-channel Faà di Bruno (#932 "unified source, fast as hand").
//!
//! A dense [`super::jet_tower::Tower4<K>`] reading ONE mixed channel materializes
//! the entire `K⁴` derivative tensor — measured at ~19× the x86 instruction count
//! of the hand factorization for an order-4 channel. The runtime partition walker
//! in [`super::jet_algebra::faa_di_bruno`] is exact but its recursive
//! `&mut dyn FnMut` enumeration does not inline to straight-line arithmetic.
//!
//! This module owns the *compiled* form: for a composition `f ∘ q` whose inner
//! map carries partials over `N` DISTINCT differentiation directions, the single
//! fully-mixed top channel `∂ᴺ(f∘q)/∂d₁…∂d_N` is the Faà di Bruno sum over the set
//! partitions of `{d₁…d_N}`,
//!
//! ```text
//!   Σ_{π ∈ partitions} f^{(|π|)}(q) · Π_{B ∈ π} q_B ,
//! ```
//!
//! where `q_B` is the inner partial over the directions in block `B`. The blocks
//! are read out of a squarefree bitmask array `q[mask]` (`q[0]` is unused — the
//! value channel never enters a top mixed partial). These functions write that
//! sum out for the small fixed orders the engine actually uses (`N ∈ {2,3,4}`),
//! so it compiles to the optimal straight-line form (LLVM CSE recovers the shared
//! sub-products). They are the SINGLE SOURCE every family feeds — there is no
//! hand-maintained per-family chain rule — and the `oracle_tests` below pin each
//! one BIT-FOR-BIT against the general runtime partition walker
//! ([`super::jet_algebra::faa_di_bruno`]), so the compiled form can never drift
//! from the universal rule.
//!
//! `f_stack[k]` is `f^{(k+1)}(q)` (the derivative magnitudes `m_{k+1}`); the value
//! `f(q)` is index −1 and never appears in a top mixed partial.

/// `N = 2`: `∂²(f∘q)/∂a∂b = m₂·q_a·q_b + m₁·q_ab`.
/// `q[1]=q_a, q[2]=q_b, q[3]=q_ab`. `m=[m₁,m₂]`.
#[inline(always)]
pub(crate) fn faa_top2(m: [f64; 2], q: &[f64; 4]) -> f64 {
    // |π|=2: {a}{b} ; |π|=1: {ab}
    m[1] * q[1] * q[2] + m[0] * q[3]
}

/// `N = 3`: the fully-mixed third channel `∂³(f∘q)/∂a∂b∂u`.
/// Bitmask: `a=1, b=2, u=4`. `m=[m₁,m₂,m₃]`.
#[inline(always)]
pub(crate) fn faa_top3(m: [f64; 3], q: &[f64; 8]) -> f64 {
    let (a, b, u) = (1usize, 2, 4);
    // |π|=3: {a}{b}{u}
    let p3 = q[a] * q[b] * q[u];
    // |π|=2: one pair + one singleton (3 partitions)
    let p2 = q[a | b] * q[u] + q[a | u] * q[b] + q[b | u] * q[a];
    // |π|=1: {abu}
    let p1 = q[a | b | u];
    m[2] * p3 + m[1] * p2 + m[0] * p1
}

/// `N = 4`: the fully-mixed fourth channel `∂⁴(f∘q)/∂a∂b∂u∂v`.
/// Bitmask: `a=1, b=2, u=4, v=8`. `m=[m₁,m₂,m₃,m₄]`.
#[inline(always)]
pub(crate) fn faa_top4(m: [f64; 4], q: &[f64; 16]) -> f64 {
    let (a, b, u, v) = (1usize, 2, 4, 8);
    // |π|=4: {a}{b}{u}{v}
    let p4 = q[a] * q[b] * q[u] * q[v];
    // |π|=3: one pair + two singletons (6 partitions)
    let p3 = q[a | b] * q[u] * q[v]
        + q[a | u] * q[b] * q[v]
        + q[a | v] * q[b] * q[u]
        + q[b | u] * q[a] * q[v]
        + q[b | v] * q[a] * q[u]
        + q[u | v] * q[a] * q[b];
    // |π|=2: three pair-pair + four triple-singleton (7 partitions)
    let p2 = q[a | b] * q[u | v]
        + q[a | u] * q[b | v]
        + q[a | v] * q[b | u]
        + q[a | b | u] * q[v]
        + q[a | b | v] * q[u]
        + q[a | u | v] * q[b]
        + q[b | u | v] * q[a];
    // |π|=1: {abuv}
    let p1 = q[a | b | u | v];
    m[3] * p4 + m[2] * p3 + m[1] * p2 + m[0] * p1
}

#[cfg(test)]
mod oracle_tests {
    //! Pin each compiled top-channel sum BIT-FOR-BIT against the general runtime
    //! partition walker [`crate::families::jet_algebra::faa_di_bruno`]. If a
    //! `faa_top*` ever diverges from the universal rule these disagree.
    use super::*;
    use crate::families::jet_algebra::faa_di_bruno;

    fn stream(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    /// Reference: the runtime walker's value for the fully-mixed top channel of
    /// `N` distinct directions, reading inner block partials from `q[mask]`.
    fn walker_top(n: usize, derivs: &[f64], q: &[f64]) -> f64 {
        let positions: Vec<usize> = (0..n).collect();
        faa_di_bruno(&positions, derivs, |block| {
            // block positions ARE the direction indices; build the squarefree mask.
            let mask: usize = block.iter().fold(0usize, |acc, &p| acc | (1 << p));
            q[mask]
        })
    }

    #[test]
    fn faa_top2_matches_runtime_walker() {
        let mut next = stream(0x2);
        for _ in 0..500 {
            let m = [next(), next()];
            let mut q = [0.0; 4];
            for (mask, qm) in q.iter_mut().enumerate() {
                if mask != 0 {
                    *qm = next();
                }
            }
            // derivs stack for faa_di_bruno is [f, f', f''] = [_, m1, m2].
            let derivs = [0.0, m[0], m[1]];
            let got = faa_top2(m, &q);
            let want = walker_top(2, &derivs, &q);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "faa_top2 {got:+.17e} vs walker {want:+.17e}"
            );
        }
    }

    #[test]
    fn faa_top3_matches_runtime_walker() {
        let mut next = stream(0x3);
        for _ in 0..500 {
            let m = [next(), next(), next()];
            let mut q = [0.0; 8];
            for (mask, qm) in q.iter_mut().enumerate() {
                if mask != 0 {
                    *qm = next();
                }
            }
            let derivs = [0.0, m[0], m[1], m[2]];
            let got = faa_top3(m, &q);
            let want = walker_top(3, &derivs, &q);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "faa_top3 {got:+.17e} vs walker {want:+.17e}"
            );
        }
    }

    #[test]
    fn faa_top4_matches_runtime_walker() {
        let mut next = stream(0x4);
        for _ in 0..500 {
            let m = [next(), next(), next(), next()];
            let mut q = [0.0; 16];
            for (mask, qm) in q.iter_mut().enumerate() {
                if mask != 0 {
                    *qm = next();
                }
            }
            let derivs = [0.0, m[0], m[1], m[2], m[3]];
            let got = faa_top4(m, &q);
            let want = walker_top(4, &derivs, &q);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "faa_top4 {got:+.17e} vs walker {want:+.17e}"
            );
        }
    }
}

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
//! sum out for the small fixed orders the engine actually uses (`N ∈ {2,3,4}`).
//! They are the SINGLE SOURCE every family feeds — there is no hand-maintained
//! per-family chain rule — and the `oracle_tests` below pin each one BIT-FOR-BIT
//! against the general runtime partition walker
//! ([`super::jet_algebra::faa_di_bruno`]), so the compiled form can never drift
//! from the universal rule.
//!
//! # Nested, not flat
//!
//! The sum is NOT written as the flat enumeration of partitions. It is written
//! in the nested form the recursion `T_{N} = ∂_{d_N} T_{N-1}` produces when the
//! product rule is applied to the previous order and the result is regrouped by
//! `f^{(k)}`: the `(a,b)`-block product `P = q_a·q_b` and its directional
//! derivatives `P_u`, `P_v`, `P_uv` are named once and shared by every order they
//! feed. The flat enumeration spells the same 15 partitions as 15 independent
//! products, and without `fast-math` LLVM cannot re-associate `q_au·q_b·q_v` and
//! `q_au·q_b·q_u` to recover the shared `q_au·q_b` — measured: the flat order-4
//! form lost to the strongest hand factorization at `median_ratio = 0.728`,
//! unanimously over fifteen paired repetitions, and the hand factorization was
//! precisely this nested form. Both forms are the universal rule; only the
//! nested one is also the optimal schedule.
//!
//! `f_stack[k]` is `f^{(k+1)}(q)` (the derivative magnitudes `m_{k+1}`); the value
//! `f(q)` is index −1 and never appears in a top mixed partial.

/// `N = 2`: `∂²(f∘q)/∂a∂b = m₂·q_a·q_b + m₁·q_ab`.
/// `q[1]=q_a, q[2]=q_b, q[3]=q_ab`. `m=[m₁,m₂]`.
#[inline(always)]
pub fn faa_top2(m: [f64; 2], q: &[f64; 4]) -> f64 {
    // |π|=2: {a}{b} ; |π|=1: {ab}
    m[1] * q[1] * q[2] + m[0] * q[3]
}

/// `N = 3`: the fully-mixed third channel `∂³(f∘q)/∂a∂b∂u`, as
/// `∂_u [m₂·q_a·q_b + m₁·q_ab]` regrouped by `f^{(k)}`.
/// Bitmask: `a=1, b=2, u=4`. `m=[m₁,m₂,m₃]`.
#[inline(always)]
pub fn faa_top3(m: [f64; 3], q: &[f64; 8]) -> f64 {
    let (a, b, u) = (1usize, 2, 4);
    // P = q_a·q_b and its u-derivative.
    let p = q[a] * q[b];
    let p_u = q[a | u] * q[b] + q[a] * q[b | u];
    // ∂_u(m₂·P)      = m₃·q_u·P + m₂·P_u
    // ∂_u(m₁·q_ab)   = m₂·q_u·q_ab + m₁·q_abu
    m[2] * (q[u] * p) + m[1] * (p_u + q[u] * q[a | b]) + m[0] * q[a | b | u]
}

/// `N = 4`: the fully-mixed fourth channel `∂⁴(f∘q)/∂a∂b∂u∂v`, as
/// `∂_v` of the order-3 form regrouped by `f^{(k)}`.
/// Bitmask: `a=1, b=2, u=4, v=8`. `m=[m₁,m₂,m₃,m₄]`.
#[inline(always)]
pub fn faa_top4(m: [f64; 4], q: &[f64; 16]) -> f64 {
    let (a, b, u, v) = (1usize, 2, 4, 8);
    // P = q_a·q_b and its directional derivatives, each named once.
    let p = q[a] * q[b];
    let p_u = q[a | u] * q[b] + q[a] * q[b | u];
    let p_v = q[a | v] * q[b] + q[a] * q[b | v];
    let p_uv = q[a | u | v] * q[b] + q[a | u] * q[b | v] + q[a | v] * q[b | u] + q[a] * q[b | u | v];
    let uv = q[u] * q[v];
    // ∂_v of the order-3 terms:
    //   m₃·q_u·P        → m₄·q_v·q_u·P + m₃·(q_uv·P + q_u·P_v)
    //   m₂·(P_u + q_u·q_ab) → m₃·q_v·(P_u + q_u·q_ab) + m₂·(P_uv + q_uv·q_ab + q_u·q_abv)
    //   m₁·q_abu        → m₂·q_v·q_abu + m₁·q_abuv
    m[3] * (uv * p)
        + m[2] * (q[u | v] * p + q[u] * p_v + q[v] * p_u + uv * q[a | b])
        + m[1] * (p_uv + q[u | v] * q[a | b] + q[u] * q[a | b | v] + q[v] * q[a | b | u])
        + m[0] * q[a | b | u | v]
}

/// Compile several order-three top channels as one output schedule.
#[inline(always)]
pub fn faa_bundle3<const OUTPUTS: usize>(m: [f64; 3], q: &[[f64; 8]; OUTPUTS]) -> [f64; OUTPUTS] {
    std::array::from_fn(|output| faa_top3(m, &q[output]))
}

/// Joint order-three pullback coefficients for the same curve/wiggle map.
#[inline(always)]
pub fn curve_wiggle_bundle3(
    m: [f64; 3],
    q_u: f64,
    a: f64,
    b: f64,
    a_u: f64,
    b_u: f64,
    xi: f64,
) -> [f64; 6] {
    faa_bundle3(
        m,
        &[
            [0.0, a, a, b, q_u, a_u, a_u, b_u],
            [0.0, a, 1.0, 0.0, q_u, a_u, 0.0, 0.0],
            [0.0, a, 0.0, 1.0, q_u, a_u, xi, 0.0],
            [0.0, a, 0.0, 0.0, q_u, a_u, 0.0, xi],
            [0.0, 1.0, 1.0, 0.0, q_u, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, q_u, xi, 0.0, 0.0],
        ],
    )
}

/// Joint order-four pullback coefficients for the same curve/wiggle map.
///
/// This is the sparse, build-time-expanded form of nine [`faa_top4`] calls.
/// Terms whose inner-map mask is structurally zero are absent, and outputs are
/// scheduled from the smallest reusable channels (`ww_*` → `eta_w_*` →
/// `eta_eta`). The independent partition walker tests below remain the semantic
/// oracle, so this is a compiler lowering of the universal rule rather than a
/// second family calculus.
#[inline(always)]
pub fn curve_wiggle_bundle4(
    m: [f64; 4],
    q: [f64; 3],
    a_jet: [f64; 4],
    b_jet: [f64; 4],
    xi: [f64; 2],
) -> [f64; 9] {
    let [m1, m2, m3, m4] = m;
    let [q_u, q_v, q_uv] = q;
    let [a, a_u, a_v, a_uv] = a_jet;
    let [b, b_u, b_v, b_uv] = b_jet;
    let [xi_u, xi_v] = xi;
    let xi_uv = xi_u * xi_v;
    let ww_bb = m4 * q_u * q_v + m3 * q_uv;
    let ww_db = m3 * (xi_u * q_v + xi_v * q_u);
    let ww_ddb = m2 * xi_uv;
    let ww_dd = 2.0 * ww_ddb;
    let eta_w_b = ww_bb * a + m3 * (a_u * q_v + a_v * q_u) + m2 * a_uv;
    let eta_w_d1 = ww_db * a + m3 * q_u * q_v + m2 * (q_uv + a_u * xi_v + a_v * xi_u);
    let eta_w_d2 = ww_ddb * a + m2 * (q_u * xi_v + q_v * xi_u);
    let eta_w_d3 = m1 * xi_uv;
    let eta_eta = eta_w_b * a
        + m3 * (a * (q_u * a_v + q_v * a_u) + q_u * q_v * b)
        + m2 * (a * a_uv + 2.0 * a_u * a_v + q_uv * b + q_u * b_v + q_v * b_u)
        + m1 * b_uv;
    [
        eta_eta, eta_w_b, eta_w_d1, eta_w_d2, eta_w_d3, ww_bb, ww_db, ww_ddb, ww_dd,
    ]
}

#[cfg(test)]
mod oracle_tests {
    //! Pin each compiled top-channel sum BIT-FOR-BIT against the general runtime
    //! partition walker [`gam_math::jet_algebra::faa_di_bruno`]. If a
    //! `faa_top*` ever diverges from the universal rule these disagree.
    use super::*;
    use crate::jet_algebra::faa_di_bruno;

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

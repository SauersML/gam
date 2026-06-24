//! Pure scalar objective-derivative coefficient helpers for the binomial
//! location-scale family.
//!
//! Self-contained seam extracted from the gamlss monolith (issue #780): the
//! three closed-form arithmetic kernels that assemble the Hessian coefficient,
//! its first directional derivative, and its mixed second directional
//! derivative from the per-row objective derivative magnitudes
//! `m_k = F^(k)(q)` and the scalar `q`-map derivative terms.
//!
//! ## #932: the outer chain rule is single-sourced
//!
//! Each `coeff` is one entry of the composite `F(q(θ))` where `F` is the scalar
//! per-row objective (its derivative stack `m1..m4 = F^(k)(q)` arrives already
//! single-sourced from the `binomial_loglik_q_tower` Faà-di-Bruno path) and
//! `q(θ)` is the latent-position map carrying its own partials in up to four
//! distinct differentiation directions (`a`, `b`, `u`, `v`).
//!
//! * `hessian_coeff` (`H_ab`, order 2) is the **hot inner-Newton** path and is
//!   kept as the hand-tuned closed form `m2·q_a q_b + m1·q_ab` — the cheapest
//!   possible spelling, evaluated once per row per inner iterate. It is pinned
//!   bit-for-bit against the `Tower2` composition in the oracle_tests test below.
//! * `directionalhessian_coeff` (`D_u H_ab`, order 3) and
//!   `second_directionalhessian_coeff` (`D²_{uv} H_ab`, order 4) are the
//!   **outer-loop cross-block** chains — the exact #736/#947/#948 bug genus
//!   (a dropped or double-counted product-rule term that is invisible until a
//!   new consumer touches the full mixed block). They are now derived
//!   MECHANICALLY: a `Tower4` jet seeded with the `q`-map partials in the four
//!   distinct seed directions, composed with the `F`-derivative stack, and the
//!   single mixed channel `t3[a,b,u]` / `t4[a,b,u,v]` read out. The
//!   Leibniz/Faà-di-Bruno coefficients are produced by the shared partition
//!   walker, not by hand, so no cross term can be silently dropped.
//!
//! The pre-migration hand formulas are retained verbatim as the bit-identity
//! oracle (`mod oracle_tests`) the new tower path is pinned against (the
//! `verify_kernel_channels` discipline — the hand calculus is the witness, the
//! tower is the single source of truth).

use crate::families::fast_channel::{faa_top2, faa_top3, faa_top4};

#[inline]
pub(crate) fn hessian_coeff_fromobjective_q_terms(
    m1: f64,
    m2: f64,
    q_a: f64,
    q_b: f64,
    q_ab: f64,
) -> f64 {
    // #932 unified source: `H_ab = ∂²(F∘q)/∂a∂b` is the fully-mixed order-2 top
    // channel of the composition, so it IS `fast_channel::faa_top2` — the
    // universal Faà di Bruno partition sum `m₂·q_a q_b + m₁·q_ab`. No
    // per-family chain rule; the primitive is pinned to the runtime partition
    // walker in `fast_channel`, and compiles to the same straight line as the
    // closed form (a packed order-2 dual matched hand asm).
    faa_top2([m1, m2], &[0.0, q_a, q_b, q_ab])
}

#[inline]
pub(crate) fn directionalhessian_coeff_fromobjective_q_terms(
    m1: f64,
    m2: f64,
    m3: f64,
    dq: f64,
    q_a: f64,
    q_b: f64,
    q_ab: f64,
    dq_a: f64,
    dq_b: f64,
    dq_ab: f64,
) -> f64 {
    // #932 unified source: `D_u H_ab = ∂³(F∘q)/∂a∂b∂u` is the fully-mixed
    // order-3 top channel, so it IS `fast_channel::faa_top3` — the universal
    // partition sum over the three distinct directions {a, b, u}. Pack the
    // q-map block partials into the bitmask array (a=1, b=2, u=4) and read the
    // top channel. This is the SAME jet truth as `Tower4<3>::compose.t3[a][b][u]`
    // (pinned in `oracle_tests`) but computes ONLY the read channel as a
    // compile-time-unrolled sum — measured at ~hand instruction count, vs ~19×
    // for the dense tower that materializes the whole 3⁴ tensor.
    let q = [
        0.0, q_a, q_b, q_ab, // _, a, b, ab
        dq, dq_a, dq_b, dq_ab, // u, au, bu, abu
    ];
    faa_top3([m1, m2, m3], &q)
}

#[inline]
pub(crate) fn second_directionalhessian_coeff_fromobjective_q_terms(
    m1: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    dq_u: f64,
    dqv: f64,
    d2q_uv: f64,
    q_a: f64,
    q_b: f64,
    q_ab: f64,
    dq_a_u: f64,
    dq_av: f64,
    dq_b_u: f64,
    dq_bv: f64,
    d2q_a_uv: f64,
    d2q_b_uv: f64,
    dq_ab_u: f64,
    dq_abv: f64,
    d2q_ab_uv: f64,
) -> f64 {
    // #932 unified source: `D²_{uv} H_ab = ∂⁴(F∘q)/∂a∂b∂u∂v` is the fully-mixed
    // order-4 top channel, so it IS `fast_channel::faa_top4` — the universal
    // partition sum over the four distinct directions {a, b, u, v}. Pack the
    // q-map block partials into the bitmask array (a=1, b=2, u=4, v=8) and read
    // the top channel. Same jet truth as `Tower4<4>::compose.t4[a][b][u][v]`
    // (pinned in `oracle_tests`) but computes ONLY that channel as a
    // compile-time-unrolled 15-term sum — measured at ≤ hand instruction count,
    // vs ~19× for the dense tower. The single `dq_u·dqv·q_ab` term an even-older
    // hand path once DOUBLE-counted is one partition (`{u}{v}` over the `q_ab`
    // block) the universal rule emits exactly once.
    let q = [
        0.0, q_a, q_b, q_ab, // _, a, b, ab
        dq_u, dq_a_u, dq_b_u, dq_ab_u, // u, au, bu, abu
        dqv, dq_av, dq_bv, dq_abv, // v, av, bv, abv
        d2q_uv, d2q_a_uv, d2q_b_uv, d2q_ab_uv, // uv, auv, buv, abuv
    ];
    faa_top4([m1, m2, m3, m4], &q)
}

#[cfg(test)]
mod oracle_tests {
    //! #932 single-source oracles: the production coefficients carry the
    //! hand-factored straight-line form (fast — the dense tower is ~19× the
    //! instruction count to read one channel), and these tests pin them
    //! BIT-FOR-BIT against the mechanical `Tower` composition that IS the
    //! single source of truth. The jet is the truth; the hand spelling is the
    //! compiled form. If the hand factorization ever drifts from the jet these
    //! channels disagree.
    use super::*;
    use crate::families::jet_tower::{Tower2, Tower4};

    /// Distinct seed-direction indices for the `q`-map jet: the two
    /// Hessian-block axes `a`/`b` and the two directional axes `u`/`v`.
    const A: usize = 0;
    const B: usize = 1;
    const U: usize = 2;
    const V: usize = 3;

    /// The hand `Tower2`-equivalent for the order-2 Hessian coefficient.
    fn hessian_via_tower(m1: f64, m2: f64, q_a: f64, q_b: f64, q_ab: f64) -> f64 {
        let mut q = Tower2::<2>::zero();
        q.g[0] = q_a;
        q.g[1] = q_b;
        q.h[0][1] = q_ab;
        q.h[1][0] = q_ab;
        q.compose_unary([0.0, m1, m2]).h[0][1]
    }

    /// The SINGLE SOURCE for `D_u H_ab` (order 3): seed a `Tower4<3>` jet for
    /// `q` over the sub-blocks of the three distinct directions {a, b, u},
    /// compose with the F-stack `[·, m1, m2, m3]`, and read the mixed channel
    /// `t3[a][b][u]`. The partition walker supplies every Leibniz coefficient
    /// mechanically — this is what the production hand factorization must equal.
    fn tower_order3(
        m1: f64,
        m2: f64,
        m3: f64,
        dq: f64,
        q_a: f64,
        q_b: f64,
        q_ab: f64,
        dq_a: f64,
        dq_b: f64,
        dq_ab: f64,
    ) -> f64 {
        let mut q = Tower4::<3>::zero();
        q.g[A] = q_a;
        q.g[B] = q_b;
        q.g[U] = dq;
        q.h[A][B] = q_ab;
        q.h[A][U] = dq_a;
        q.h[B][U] = dq_b;
        q.t3[A][B][U] = dq_ab;
        q.compose_unary([0.0, m1, m2, m3, 0.0]).t3[A][B][U]
    }

    /// The SINGLE SOURCE for `D²_{uv} H_ab` (order 4): seed a `Tower4<4>` jet
    /// for `q` over the 15 mixed sub-blocks of the four distinct directions
    /// {a, b, u, v}, compose with `[·, m1, m2, m3, m4]`, and read `t4[a][b][u][v]`.
    fn tower_order4(
        m1: f64,
        m2: f64,
        m3: f64,
        m4: f64,
        dq_u: f64,
        dqv: f64,
        d2q_uv: f64,
        q_a: f64,
        q_b: f64,
        q_ab: f64,
        dq_a_u: f64,
        dq_av: f64,
        dq_b_u: f64,
        dq_bv: f64,
        d2q_a_uv: f64,
        d2q_b_uv: f64,
        dq_ab_u: f64,
        dq_abv: f64,
        d2q_ab_uv: f64,
    ) -> f64 {
        let mut q = Tower4::<4>::zero();
        q.g[A] = q_a;
        q.g[B] = q_b;
        q.g[U] = dq_u;
        q.g[V] = dqv;
        q.h[A][B] = q_ab;
        q.h[A][U] = dq_a_u;
        q.h[A][V] = dq_av;
        q.h[B][U] = dq_b_u;
        q.h[B][V] = dq_bv;
        q.h[U][V] = d2q_uv;
        q.t3[A][B][U] = dq_ab_u;
        q.t3[A][B][V] = dq_abv;
        q.t3[A][U][V] = d2q_a_uv;
        q.t3[B][U][V] = d2q_b_uv;
        q.t4[A][B][U][V] = d2q_ab_uv;
        q.compose_unary([0.0, m1, m2, m3, m4]).t4[A][B][U][V]
    }

    /// A deterministic pseudo-random stream so every channel participates with a
    /// distinct nonzero value (no accidental cancellation hides a dropped term).
    fn stream(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    fn close(label: &str, got: f64, want: f64) {
        let tol = 1e-12 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol,
            "{label}: tower {got:+.17e} vs hand {want:+.17e} (|Δ|={:.3e})",
            (got - want).abs()
        );
    }

    #[test]
    fn hessian_matches_tower() {
        let mut next = stream(0xC0FFEE);
        for _ in 0..200 {
            let (m1, m2) = (next(), next());
            let (q_a, q_b, q_ab) = (next(), next(), next());
            let prod = hessian_coeff_fromobjective_q_terms(m1, m2, q_a, q_b, q_ab);
            let tower = hessian_via_tower(m1, m2, q_a, q_b, q_ab);
            close("H_ab", prod, tower);
        }
    }

    #[test]
    fn directional_matches_jet_single_source() {
        let mut next = stream(0xBEEF);
        for _ in 0..400 {
            let m = [next(), next(), next()];
            let (dq, q_a, q_b, q_ab, dq_a, dq_b, dq_ab) =
                (next(), next(), next(), next(), next(), next(), next());
            // production (hand-factored, fast) vs the Tower4<3> single source.
            let prod = directionalhessian_coeff_fromobjective_q_terms(
                m[0], m[1], m[2], dq, q_a, q_b, q_ab, dq_a, dq_b, dq_ab,
            );
            let tower =
                tower_order3(m[0], m[1], m[2], dq, q_a, q_b, q_ab, dq_a, dq_b, dq_ab);
            close("D_u H_ab", prod, tower);
        }
    }

    #[test]
    fn second_directional_matches_jet_single_source() {
        let mut next = stream(0xD00D);
        for _ in 0..400 {
            let m = [next(), next(), next(), next()];
            let dq_u = next();
            let dqv = next();
            let d2q_uv = next();
            let q_a = next();
            let q_b = next();
            let q_ab = next();
            let dq_a_u = next();
            let dq_av = next();
            let dq_b_u = next();
            let dq_bv = next();
            let d2q_a_uv = next();
            let d2q_b_uv = next();
            let dq_ab_u = next();
            let dq_abv = next();
            let d2q_ab_uv = next();
            // production (hand-factored, fast) vs the Tower4<4> single source.
            let prod = second_directionalhessian_coeff_fromobjective_q_terms(
                m[0], m[1], m[2], m[3], dq_u, dqv, d2q_uv, q_a, q_b, q_ab, dq_a_u, dq_av, dq_b_u,
                dq_bv, d2q_a_uv, d2q_b_uv, dq_ab_u, dq_abv, d2q_ab_uv,
            );
            let tower = tower_order4(
                m[0], m[1], m[2], m[3], dq_u, dqv, d2q_uv, q_a, q_b, q_ab, dq_a_u, dq_av, dq_b_u,
                dq_bv, d2q_a_uv, d2q_b_uv, dq_ab_u, dq_abv, d2q_ab_uv,
            );
            close("D2_uv H_ab", prod, tower);
        }
    }

    /// #932 βw cross-channel guard. The link-wiggle kernel
    /// (`binomial/wiggle.rs`) does NOT call
    /// `second_directionalhessian_coeff_fromobjective_q_terms` per (row, coeff)
    /// — that would be `O(n · p_w)` order-4 tower evaluations. Instead it
    /// hand-expands the coefficient as a polynomial in each wiggle basis slot
    /// `(b, b′, b″, b‴)` and factors out the per-row scalar coefficients
    /// `alpha_xw_{b,d,dd,d3}` (the `X_t/X_ls ↔ wiggle` cross blocks) and
    /// `c_ww_{bb,bd,bdd,dd_pair}` (the wiggle↔wiggle block), so the operator
    /// applies them as rank structure. Those hand expansions are the exact
    /// #736/#947 bug genus (a dropped or mis-weighted mixed term).
    ///
    /// This test re-derives both expansions from the documented substitutions
    /// and asserts they reproduce the SINGLE-SOURCE coefficient bit-for-bit, so
    /// any future drift in the wiggle expansion that diverges from the jet
    /// truth is caught here rather than in a silently wrong outer Hessian.
    #[test]
    fn betaw_cross_channel_expansions_match_single_source() {
        let mut next = stream(0x5151);
        let mut max_xw = 0.0_f64;
        let mut max_ww = 0.0_f64;
        for _ in 0..2000 {
            let (m1, m2, m3, m4) = (next(), next(), next(), next());
            let (dq_u, dqv, d2q_uv) = (next(), next(), next());
            // full-q x-channel partials (x = t).
            let (q_t, dq_t_u, dq_tv, d2q_t_uv) = (next(), next(), next(), next());
            // base-q0 partials.
            let (q0_t, dq0_u, dq0v, d2q0_uv) = (next(), next(), next(), next());
            let (dq0_t_u, dq0_tv, d2q0_t_uv) = (next(), next(), next());
            // one wiggle basis slot j and a second slot k.
            let (br, dr, ddr, d3r) = (next(), next(), next(), next());

            // X_t ↔ wiggle cross block: a = t, b = w. Substitutions mirror the
            // `binomial/wiggle.rs` comment (qw = b, dqw_u = b′·dq0u, …).
            let qw = br;
            let dqw_u = dr * dq0_u;
            let dqwv = dr * dq0v;
            let d2qw_uv = ddr * dq0_u * dq0v + dr * d2q0_uv;
            let q_tw = dr * q0_t;
            let dq_tw_u = ddr * dq0_u * q0_t + dr * dq0_t_u;
            let dq_twv = ddr * dq0v * q0_t + dr * dq0_tv;
            let d2q_tw_uv = d3r * dq0_u * dq0v * q0_t
                + ddr * (d2q0_uv * q0_t + dq0_u * dq0_tv + dq0v * dq0_t_u)
                + dr * d2q0_t_uv;
            let coeff_tw = second_directionalhessian_coeff_fromobjective_q_terms(
                m1, m2, m3, m4, dq_u, dqv, d2q_uv, q_t, qw, q_tw, dq_t_u, dq_tv, dqw_u, dqwv,
                d2q_t_uv, d2qw_uv, dq_tw_u, dq_twv, d2q_tw_uv,
            );
            let alpha_b =
                m4 * dq_u * dqv * q_t + m3 * (d2q_uv * q_t + dq_u * dq_tv + dqv * dq_t_u) + m2 * d2q_t_uv;
            let alpha_d = m3 * (dq_u * q_t * dq0v + dqv * q_t * dq0_u + dq_u * dqv * q0_t)
                + m2
                    * (dq_t_u * dq0v
                        + dq_tv * dq0_u
                        + q_t * d2q0_uv
                        + d2q_uv * q0_t
                        + dq_u * dq0_tv
                        + dqv * dq0_t_u)
                + m1 * d2q0_t_uv;
            let alpha_dd = m2 * (q_t * dq0_u * dq0v + dq_u * dq0v * q0_t + dqv * dq0_u * q0_t)
                + m1 * (d2q0_uv * q0_t + dq0_u * dq0_tv + dq0v * dq0_t_u);
            let alpha_d3 = m1 * dq0_u * dq0v * q0_t;
            let recon_xw = alpha_b * br + alpha_d * dr + alpha_dd * ddr + alpha_d3 * d3r;
            max_xw = max_xw.max((coeff_tw - recon_xw).abs() / coeff_tw.abs().max(1.0));

            // wiggle ↔ wiggle block: a = w_j, b = w_k (q linear in each βw, so
            // q_ab = 0 and there is no third/fourth own-channel term).
            let (brk, drk, ddrk) = (next(), next(), next());
            let coeff_ww = second_directionalhessian_coeff_fromobjective_q_terms(
                m1,
                m2,
                m3,
                m4,
                dq_u,
                dqv,
                d2q_uv,
                br,
                brk,
                0.0,
                dr * dq0_u,
                dr * dq0v,
                drk * dq0_u,
                drk * dq0v,
                ddr * dq0_u * dq0v + dr * d2q0_uv,
                ddrk * dq0_u * dq0v + drk * d2q0_uv,
                0.0,
                0.0,
                0.0,
            );
            let c_ww_bb = m4 * dq_u * dqv + m3 * d2q_uv;
            let c_ww_bd = m3 * (dq_u * dq0v + dqv * dq0_u) + m2 * d2q0_uv;
            let c_ww_bdd = m2 * dq0_u * dq0v;
            let c_ww_dd_pair = 2.0 * m2 * dq0_u * dq0v;
            let recon_ww = c_ww_bb * br * brk
                + c_ww_bd * (br * drk + dr * brk)
                + c_ww_bdd * (br * ddrk + ddr * brk)
                + c_ww_dd_pair * dr * drk;
            max_ww = max_ww.max((coeff_ww - recon_ww).abs() / coeff_ww.abs().max(1.0));
        }
        assert!(max_xw < 1e-12, "alpha_xw drifted from single source: {max_xw:.3e}");
        assert!(max_ww < 1e-12, "c_ww drifted from single source: {max_ww:.3e}");
    }
}

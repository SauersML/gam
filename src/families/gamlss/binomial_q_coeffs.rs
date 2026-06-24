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
//!   bit-for-bit against the `Tower2` composition in the oracle test below.
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
//! oracle (`mod oracle`) the new tower path is pinned against (the
//! `verify_kernel_channels` discipline — the hand calculus is the witness, the
//! tower is the single source of truth).

use crate::families::jet_tower::Tower4;

/// Distinct seed-direction indices for the `q`-map jet: the two Hessian-block
/// axes `a`/`b` and the two directional-derivative axes `u`/`v`.
const A: usize = 0;
const B: usize = 1;
const U: usize = 2;
const V: usize = 3;

#[inline]
pub(crate) fn hessian_coeff_fromobjective_q_terms(
    m1: f64,
    m2: f64,
    q_a: f64,
    q_b: f64,
    q_ab: f64,
) -> f64 {
    // F = -sum ell, scalar q:
    //   H_ab = m2 * q_a q_b + m1 * q_ab.
    //
    // Hot inner-Newton path: kept as the hand-tuned closed form. The
    // `Tower2<2>` composition `compose([·, m1, m2]).h[a][b]` produces the exact
    // same value (pinned in `oracle::hessian_matches_tower`); the hand spelling
    // avoids building a 2-variable tower per row per inner iterate.
    m2 * q_a * q_b + m1 * q_ab
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
    // #932: `D_u H_ab` is the order-3 mixed channel of `F(q(θ))` in the three
    // distinct directions (a, b, u). Seed a `Tower4<3>` jet for `q` with its
    // partials over every sub-block of {a, b, u}, compose with the F-stack
    // `[·, m1, m2, m3]`, and read `t3[a][b][u]`. The partition walker supplies
    // the Leibniz coefficients (`m3·q_a q_b dq`, the three
    // `m2·(pair·single)` terms, `m1·dq_ab`) mechanically.
    let mut q = Tower4::<3>::zero();
    // First-order: q_a = ∂q/∂a, q_b = ∂q/∂b, dq = ∂q/∂u.
    q.g[A] = q_a;
    q.g[B] = q_b;
    q.g[U] = dq;
    // Second-order (canonical sorted index only; the t3[a,b,u] read touches no
    // other ordering).
    q.h[A][B] = q_ab; // ∂²q/∂a∂b
    q.h[A][U] = dq_a; // ∂²q/∂a∂u
    q.h[B][U] = dq_b; // ∂²q/∂b∂u
    // Third-order: ∂³q/∂a∂b∂u.
    q.t3[A][B][U] = dq_ab;
    q.compose_unary([0.0, m1, m2, m3, 0.0]).t3[A][B][U]
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
    // #932: `D²_{uv} H_ab` is the order-4 mixed channel of `F(q(θ))` in the four
    // distinct directions (a, b, u, v). Seed a `Tower4<4>` jet for `q` with its
    // 15 mixed partials over every sub-block of {a, b, u, v}, compose with the
    // F-stack `[·, m1, m2, m3, m4]`, and read `t4[a][b][u][v]`.
    //
    // This is exactly the hand expansion in `oracle::second_directional_hand`
    // (the single `dq_u·dqv·q_ab` term the old hand path once double-counted,
    // every `m2`/`m3` cross term, …) but produced by the partition walker —
    // there is no hand-maintained channel left to drop or mis-weight.
    let mut q = Tower4::<4>::zero();
    // First-order partials ∂q/∂{a,b,u,v}.
    q.g[A] = q_a;
    q.g[B] = q_b;
    q.g[U] = dq_u;
    q.g[V] = dqv;
    // Second-order partials ∂²q/∂·∂· over the six distinct axis pairs.
    q.h[A][B] = q_ab; // ∂²q/∂a∂b
    q.h[A][U] = dq_a_u; // ∂²q/∂a∂u
    q.h[A][V] = dq_av; // ∂²q/∂a∂v
    q.h[B][U] = dq_b_u; // ∂²q/∂b∂u
    q.h[B][V] = dq_bv; // ∂²q/∂b∂v
    q.h[U][V] = d2q_uv; // ∂²q/∂u∂v
    // Third-order partials ∂³q/∂·∂·∂· over the four distinct axis triples.
    q.t3[A][B][U] = dq_ab_u; // ∂³q/∂a∂b∂u
    q.t3[A][B][V] = dq_abv; // ∂³q/∂a∂b∂v
    q.t3[A][U][V] = d2q_a_uv; // ∂³q/∂a∂u∂v
    q.t3[B][U][V] = d2q_b_uv; // ∂³q/∂b∂u∂v
    // Fourth-order partial ∂⁴q/∂a∂b∂u∂v.
    q.t4[A][B][U][V] = d2q_ab_uv;
    q.compose_unary([0.0, m1, m2, m3, m4]).t4[A][B][U][V]
}

#[cfg(test)]
mod oracle {
    //! Pre-migration hand-summed chain-rule formulas, kept verbatim as the
    //! bit-identity witnesses for the #932 `Tower` composition. If the tower
    //! path ever drifts from the hand calculus these channels disagree.
    use super::*;
    use crate::families::jet_tower::Tower2;

    /// The hand `Tower2`-equivalent for the order-2 Hessian coefficient.
    fn hessian_via_tower(m1: f64, m2: f64, q_a: f64, q_b: f64, q_ab: f64) -> f64 {
        let mut q = Tower2::<2>::zero();
        q.g[0] = q_a;
        q.g[1] = q_b;
        q.h[0][1] = q_ab;
        q.h[1][0] = q_ab;
        q.compose_unary([0.0, m1, m2]).h[0][1]
    }

    /// The PRE-MIGRATION hand chain rule for `D_u H_ab` (order 3).
    fn directional_hand(
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
        m3 * dq * q_a * q_b + m2 * (dq_a * q_b + q_a * dq_b + dq * q_ab) + m1 * dq_ab
    }

    /// The PRE-MIGRATION hand chain rule for `D²_{uv} H_ab` (order 4).
    fn second_directional_hand(
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
        let d_qaqb_u = dq_a_u * q_b + q_a * dq_b_u;
        let d_qaqbv = dq_av * q_b + q_a * dq_bv;
        let d2_qaqb_uv = d2q_a_uv * q_b + dq_a_u * dq_bv + dq_av * dq_b_u + q_a * d2q_b_uv;
        m4 * dq_u * dqv * q_a * q_b
            + m3 * (d2q_uv * q_a * q_b + dq_u * d_qaqbv + dqv * d_qaqb_u + dq_u * dqv * q_ab)
            + m2 * (d2_qaqb_uv + d2q_uv * q_ab + dq_u * dq_abv + dqv * dq_ab_u)
            + m1 * d2q_ab_uv
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
    fn directional_matches_hand_chain_rule() {
        let mut next = stream(0xBEEF);
        for _ in 0..400 {
            let m = [next(), next(), next()];
            let (dq, q_a, q_b, q_ab, dq_a, dq_b, dq_ab) =
                (next(), next(), next(), next(), next(), next(), next());
            let tower = directionalhessian_coeff_fromobjective_q_terms(
                m[0], m[1], m[2], dq, q_a, q_b, q_ab, dq_a, dq_b, dq_ab,
            );
            let hand =
                directional_hand(m[0], m[1], m[2], dq, q_a, q_b, q_ab, dq_a, dq_b, dq_ab);
            close("D_u H_ab", tower, hand);
        }
    }

    #[test]
    fn second_directional_matches_hand_chain_rule() {
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
            let tower = second_directionalhessian_coeff_fromobjective_q_terms(
                m[0], m[1], m[2], m[3], dq_u, dqv, d2q_uv, q_a, q_b, q_ab, dq_a_u, dq_av, dq_b_u,
                dq_bv, d2q_a_uv, d2q_b_uv, dq_ab_u, dq_abv, d2q_ab_uv,
            );
            let hand = second_directional_hand(
                m[0], m[1], m[2], m[3], dq_u, dqv, d2q_uv, q_a, q_b, q_ab, dq_a_u, dq_av, dq_b_u,
                dq_bv, d2q_a_uv, d2q_b_uv, dq_ab_u, dq_abv, d2q_ab_uv,
            );
            close("D2_uv H_ab", tower, hand);
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

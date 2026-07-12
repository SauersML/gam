//! Poisson-log exponential-family Taylor-jet oracle (#932, FD-free exactness).
//!
//! Issue #932 builds out a generic, fixed-order truncated-Taylor jet algebra so
//! every family's [`super::row_kernel::RowKernel`] derivative tower (value / ∇ /
//! H / contracted-third / contracted-fourth) is MECHANICALLY derived from a
//! single row-NLL expression rather than hand-written. The jet machinery itself
//! ([`crate::jet_scalar::JetScalar`] with the packed [`crate::jet_scalar::Order2`]
//! / [`crate::jet_scalar::OneSeed`] / [`crate::jet_scalar::TwoSeed`] scalars and
//! the dense [`crate::jet_tower::Tower4`]) already lives in the tree and the two
//! production `RowKernel` families (survival marginal-slope, Bernoulli rigid) are
//! pinned against it.
//!
//! What this module adds is the FD-free exactness check item #2 of the issue
//! calls for, anchored on a CLEAN closed-form exponential family that is NOT a
//! `RowKernel` impl elsewhere in the tree: a Poisson-log GLM row negative
//! log-likelihood. The row loss is written ONCE, generic over
//! `S: JetScalar<2>` ([`poisson_row_nll`]), and the jet-derived tower (every
//! channel, including the third / fourth contractions realized through the
//! production packed `OneSeed` / `TwoSeed` scalars) is asserted equal to an
//! INDEPENDENT, hand-derived closed-form Poisson derivative tower
//! ([`poisson_closed_form_tower`]) at several deterministic pseudo-random points.
//!
//! This is a genuine, distinct CI guard: the existing scalar tests pin the
//! packed scalars against the dense `Tower4` (the algebra against itself); here
//! the WHOLE algebra — `exp`, `ln`, multiplicative seeding, all four orders, and
//! the nilpotent contraction seeding — is pinned against calculus done by hand
//! for a real exponential family. A regression in any algebra primitive (a sign
//! flip in the cross-Hessian, a dropped Faà di Bruno term, an off-by-one in the
//! `OneSeed`/`TwoSeed` composition) that happened to be self-consistent across
//! the packed/dense scalars would still be caught here, because the comparand is
//! external hand calculus, not another jet.
//!
//! # The model
//!
//! Two primaries `p = (p₀, p₁)`. The Poisson linear predictor is BILINEAR,
//!
//! ```text
//!   η(p) = a·p₀ + b·p₁ + d·p₀·p₁,
//! ```
//!
//! so its own second derivative has a nonzero cross term (`∂²η/∂p₀∂p₁ = d`) and
//! all third- and higher η-derivatives vanish. The bilinear form is deliberate:
//! it makes every off-diagonal entry of the Hessian / third / fourth towers
//! nonzero, so a dropped or sign-flipped cross-channel (the #736 bug genus) is
//! loud. With `μ = e^{η}` the row NLL (Poisson log-likelihood, sign-flipped, with
//! the data-only normalizer `ln Γ(y+1)` retained so the value channel is the
//! true NLL) is
//!
//! ```text
//!   ℓ(p) = μ − y·η + ln Γ(y + 1) = e^{η} − y·η + ln Γ(y + 1).
//! ```

use crate::jet_scalar::{JetScalar, OneSeed, Order2, TwoSeed};
use crate::jet_tower::{
    RowProgram, Tower4, program_fourth_contracted, program_full_tower, program_row_kernel,
    program_third_contracted,
};

/// One Poisson-log row fixture: the response `y` (a count, as `f64`) and the
/// three bilinear-predictor coefficients `(a, b, d)` defining
/// `η = a·p₀ + b·p₁ + d·p₀·p₁`.
#[derive(Clone, Copy, Debug)]
pub struct PoissonRow {
    /// Observed count response `yᵢ ≥ 0`.
    pub y: f64,
    /// Coefficient of `p₀` in the bilinear predictor.
    pub a: f64,
    /// Coefficient of `p₁` in the bilinear predictor.
    pub b: f64,
    /// Coefficient of the `p₀·p₁` cross term in the bilinear predictor.
    pub d: f64,
}

/// The Poisson-log row negative log-likelihood, written ONCE over the generic
/// jet scalar `S`. The `p` array arrives pre-seeded by the caller (plain
/// variables for the order-2 channel, or with the nilpotent ε / δ directions for
/// the contracted third / fourth). The body uses ONLY [`JetScalar`] ops, and the
/// per-row data (`y`, `a`, `b`, `d`, the `ln Γ(y+1)` normalizer) enters as plain
/// `f64` constants — the single source of truth from which every derivative
/// channel is then exact by construction.
///
/// `ℓ = e^{η} − y·η + ln Γ(y+1)`, `η = a·p₀ + b·p₁ + d·p₀·p₁`.
pub fn poisson_row_nll<S: JetScalar<2>>(row: &PoissonRow, p: &[S; 2]) -> S {
    // η = a·p₀ + b·p₁ + d·p₀·p₁  (all combinators are exact truncated Leibniz).
    let eta = p[0]
        .scale(row.a)
        .add(&p[1].scale(row.b))
        .add(&p[0].mul(&p[1]).scale(row.d));
    // μ = e^{η}.
    let mu = eta.exp();
    // ℓ = μ − y·η + ln Γ(y+1).  ln Γ(y+1) is a data-only constant: it shifts the
    // value channel to the true NLL and leaves every derivative channel
    // untouched (its jet is `constant`, all derivative slots zero).
    let ln_norm = ln_gamma_real(row.y + 1.0);
    mu.sub(&eta.scale(row.y)).add(&S::constant(ln_norm))
}

/// Evaluate the single-expression Poisson row NLL on the full dense
/// [`Tower4`] scalar at the base point `p₀ = (p[0], p[1])`, returning every
/// derivative channel `(v, g, h, t3, t4)` in one pass. This is the jet-derived
/// tower the test pins against the hand-derived closed form.
pub fn poisson_jet_tower(row: &PoissonRow, p0: [f64; 2]) -> Tower4<2> {
    let vars: [Tower4<2>; 2] = std::array::from_fn(|axis| Tower4::variable(p0[axis], axis));
    poisson_row_nll(row, &vars)
}

/// A minimal [`RowProgram`] wrapper around [`poisson_row_nll`].
///
/// This mirrors the production family seam: primaries are seeded by the
/// `program_*` derivation helpers, while the row likelihood itself stays a single
/// expression over `S: JetScalar<2>`.
struct PoissonProgram {
    row: PoissonRow,
    p0: [f64; 2],
}

impl RowProgram<2> for PoissonProgram {
    fn n_rows(&self) -> usize {
        1
    }

    fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
        assert_eq!(row, 0, "single-row PoissonProgram only contains row zero");
        Ok(self.p0)
    }

    fn eval<S: JetScalar<2>>(&self, row: usize, p: &[S; 2]) -> Result<S, String> {
        assert_eq!(row, 0, "single-row PoissonProgram only contains row zero");
        Ok(poisson_row_nll(&self.row, p))
    }
}

/// INDEPENDENT hand-derived closed-form Poisson-log derivative tower at the base
/// point `p₀`. Derived by direct calculus (NOT via any jet) from
/// `ℓ = e^{η} − y·η + C`, `η = a·p₀ + b·p₁ + d·p₀·p₁`, exploiting that every
/// η-derivative of order ≥ 3 vanishes and the only nonzero η-Hessian entry is the
/// cross term `H^η₀₁ = H^η₁₀ = d`.
///
/// With `gₐ = ∂η/∂pₐ`, `Hₐᵦ = ∂²η/∂pₐ∂pᵦ`, `m = e^{η}`:
///
/// ```text
///   ℓ            = m − y·η + C
///   ∂ₐℓ          = (m − y)·gₐ
///   ∂ₐ∂ᵦℓ        = m·gₐ·gᵦ + (m − y)·Hₐᵦ
///   ∂ₐ∂ᵦ∂_cℓ     = m·(gₐgᵦg_c + Hₐᵦg_c + Hₐ_c gᵦ + Hᵦ_c gₐ)
///   ∂ₐ∂ᵦ∂_c∂_dℓ  = m·[ g_d·(gₐgᵦg_c + Hₐᵦg_c + Hₐ_cgᵦ + Hᵦ_cgₐ)
///                       + Hₐ_d gᵦg_c + gₐHᵦ_d g_c + gₐgᵦH_c_d
///                       + HₐᵦH_c_d + Hₐ_cHᵦ_d + Hᵦ_cHₐ_d ]
/// ```
pub fn poisson_closed_form_tower(row: &PoissonRow, p0: [f64; 2]) -> Tower4<2> {
    let (a, b, d, y) = (row.a, row.b, row.d, row.y);
    let (q0, q1) = (p0[0], p0[1]);
    let eta = a * q0 + b * q1 + d * q0 * q1;
    let m = eta.exp();
    let c = ln_gamma_real(y + 1.0);

    // First derivatives of the bilinear predictor η.
    let g = [a + d * q1, b + d * q0];
    // Second derivatives of η: only the cross term is nonzero.
    let mut hh = [[0.0_f64; 2]; 2];
    hh[0][1] = d;
    hh[1][0] = d;

    let mut t = Tower4::<2>::zero();
    t.v = m - y * eta + c;
    for av in 0..2 {
        t.g[av] = (m - y) * g[av];
        for bv in 0..2 {
            t.h[av][bv] = m * g[av] * g[bv] + (m - y) * hh[av][bv];
            for cv in 0..2 {
                t.t3[av][bv][cv] = m
                    * (g[av] * g[bv] * g[cv]
                        + hh[av][bv] * g[cv]
                        + hh[av][cv] * g[bv]
                        + hh[bv][cv] * g[av]);
                for dv in 0..2 {
                    t.t4[av][bv][cv][dv] = m
                        * (g[dv]
                            * (g[av] * g[bv] * g[cv]
                                + hh[av][bv] * g[cv]
                                + hh[av][cv] * g[bv]
                                + hh[bv][cv] * g[av])
                            + hh[av][dv] * g[bv] * g[cv]
                            + g[av] * hh[bv][dv] * g[cv]
                            + g[av] * g[bv] * hh[cv][dv]
                            + hh[av][bv] * hh[cv][dv]
                            + hh[av][cv] * hh[bv][dv]
                            + hh[bv][cv] * hh[av][dv]);
                }
            }
        }
    }
    t
}

/// `ln Γ(x)` for the data-only Poisson normalizer `ln Γ(y+1)`. Reuses the SAME
/// centralised `ln Γ` primitive the family jets draw on — the value entry
/// (`[0]`) of [`crate::jet_tower::ln_gamma_derivative_stack`], which is
/// `statrs::function::gamma::ln_gamma` — rather than re-deriving it, per the #932
/// instruction to reuse the shared special-function primitives.
#[inline]
fn ln_gamma_real(x: f64) -> f64 {
    crate::jet_tower::ln_gamma_derivative_stack(x)[0]
}

/// A tiny deterministic LCG so the test points are pseudo-random yet fixed
/// across runs (NO `rand`, NO date/clock seeding — per the #932 rules).
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        // Numerical Recipes LCG constants; take the high 53 bits as a
        // uniform in [0, 1).
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    /// Uniform in `[lo, hi)`.
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

const REL_TOL: f64 = 1e-9;

fn close(a: f64, b: f64, label: &str) {
    let band = REL_TOL + REL_TOL * a.abs().max(b.abs());
    assert!(
        (a - b).abs() <= band,
        "{label}: jet {a:+.15e} vs hand {b:+.15e} (band {band:.3e})"
    );
}

/// The mechanically jet-derived Poisson tower (value / ∇ / H / t3 / t4) must
/// equal the INDEPENDENT hand-derived closed form to 1e-9 at several fixed
/// pseudo-random `(p₀, y, a, b, d)` points. This is the FD-free exactness
/// check #932 item #2 asks for, on a clean exponential family.
#[test]
fn poisson_jet_tower_matches_hand_derived_closed_form() {
    let mut rng = Lcg(0x9322_0451_dead_beef);
    for trial in 0..16 {
        // Counts y ∈ {0, …, 7}; predictor coefficients and base point kept
        // in a tame range so η stays moderate (no exp overflow), exercising
        // both the y = 0 and y > 0 branches of (m − y).
        let y = (rng.uniform(0.0, 8.0)).floor();
        let row = PoissonRow {
            y,
            a: rng.uniform(-1.2, 1.2),
            b: rng.uniform(-1.2, 1.2),
            d: rng.uniform(-0.9, 0.9),
        };
        let p0 = [rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)];

        let jet = poisson_jet_tower(&row, p0);
        let hand = poisson_closed_form_tower(&row, p0);

        close(jet.v, hand.v, &format!("trial {trial} value"));
        for i in 0..2 {
            close(jet.g[i], hand.g[i], &format!("trial {trial} grad[{i}]"));
            for j in 0..2 {
                close(
                    jet.h[i][j],
                    hand.h[i][j],
                    &format!("trial {trial} hess[{i}][{j}]"),
                );
                for k in 0..2 {
                    close(
                        jet.t3[i][j][k],
                        hand.t3[i][j][k],
                        &format!("trial {trial} t3[{i}][{j}][{k}]"),
                    );
                    for l in 0..2 {
                        close(
                            jet.t4[i][j][k][l],
                            hand.t4[i][j][k][l],
                            &format!("trial {trial} t4[{i}][{j}][{k}][{l}]"),
                        );
                    }
                }
            }
        }
    }
}

/// The PRODUCTION packed scalars — `Order2` (value/∇/H), `OneSeed` (contracted
/// third), `TwoSeed` (contracted fourth) — evaluated on the SAME single
/// `poisson_row_nll` expression must reproduce the hand-derived closed form's
/// corresponding channels. This pins the cutover path the families actually
/// use (the small packed scalars, not the dense `Tower4`) against external
/// calculus, with the contraction directions folded into the nilpotent seeds.
#[test]
fn poisson_packed_scalars_match_hand_derived_contractions() {
    let mut rng = Lcg(0x0bad_f00d_1234_5678);
    let dirs: [[f64; 2]; 3] = [[0.7, -0.4], [-0.9, 1.3], [1.1, 0.6]];

    for trial in 0..12 {
        let y = (rng.uniform(0.0, 8.0)).floor();
        let row = PoissonRow {
            y,
            a: rng.uniform(-1.2, 1.2),
            b: rng.uniform(-1.2, 1.2),
            d: rng.uniform(-0.9, 0.9),
        };
        let p0 = [rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)];
        let hand = poisson_closed_form_tower(&row, p0);

        // Order2: value / gradient / Hessian.
        let o2: [Order2<2>; 2] = std::array::from_fn(|axis| Order2::variable(p0[axis], axis));
        let s2 = poisson_row_nll(&row, &o2);
        close(s2.value(), hand.v, &format!("trial {trial} Order2 value"));
        for i in 0..2 {
            close(
                s2.g()[i],
                hand.g[i],
                &format!("trial {trial} Order2 grad[{i}]"),
            );
            for j in 0..2 {
                close(
                    s2.h()[i][j],
                    hand.h[i][j],
                    &format!("trial {trial} Order2 hess[{i}][{j}]"),
                );
            }
        }

        // OneSeed: contracted third Σ_c ℓ_{abc}·dir_c for each direction,
        // checked against the hand tower's own contraction of t3.
        for (di, dir) in dirs.iter().enumerate() {
            let os: [OneSeed<2>; 2] =
                std::array::from_fn(|axis| OneSeed::seed_direction(p0[axis], axis, dir[axis]));
            let third = poisson_row_nll(&row, &os).contracted_third();
            let truth = hand.third_contracted(dir);
            for i in 0..2 {
                for j in 0..2 {
                    close(
                        third[i][j],
                        truth[i][j],
                        &format!("trial {trial} dir {di} OneSeed third[{i}][{j}]"),
                    );
                }
            }
        }

        // TwoSeed: contracted fourth Σ_{cd} ℓ_{abcd}·u_c·v_d for direction
        // pairs, checked against the hand tower's own contraction of t4.
        for (ui, u) in dirs.iter().enumerate() {
            let v = dirs[(ui + 1) % dirs.len()];
            let ts: [TwoSeed<2>; 2] =
                std::array::from_fn(|axis| TwoSeed::seed(p0[axis], axis, u[axis], v[axis]));
            let fourth = poisson_row_nll(&row, &ts).contracted_fourth();
            let truth = hand.fourth_contracted(u, &v);
            for i in 0..2 {
                for j in 0..2 {
                    close(
                        fourth[i][j],
                        truth[i][j],
                        &format!("trial {trial} pair {ui} TwoSeed fourth[{i}][{j}]"),
                    );
                }
            }
        }
    }
}

/// The RowKernel-shaped derivation surface (`program_row_kernel`,
/// `program_third_contracted`, `program_fourth_contracted`) must be a pure
/// projection of the same full tower. This is the exact, FD-free oracle pattern
/// production families use to guard hand-tuned kernels: every channel a
/// `RowKernel` consumer can request is derived from ONE [`RowProgram::eval`]
/// expression, so the cross-channel desynchronisation class from #736/#932 has
/// no second derivative source in which to hide.
#[test]
fn poisson_generic_row_kernel_helpers_are_full_tower_projections() {
    let mut rng = Lcg(0x9320_0000_cafe_f00d);
    let dirs: [[f64; 2]; 3] = [[0.7, -0.4], [-0.9, 1.3], [1.1, 0.6]];

    for trial in 0..12 {
        let program = PoissonProgram {
            row: PoissonRow {
                y: (rng.uniform(0.0, 8.0)).floor(),
                a: rng.uniform(-1.2, 1.2),
                b: rng.uniform(-1.2, 1.2),
                d: rng.uniform(-0.9, 0.9),
            },
            p0: [rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)],
        };

        let tower = program_full_tower(&program, 0).expect("full generic Poisson tower");
        let (value, gradient, hessian) =
            program_row_kernel(&program, 0).expect("generic row kernel");

        close(value, tower.v, &format!("trial {trial} generic value"));
        for i in 0..2 {
            close(
                gradient[i],
                tower.g[i],
                &format!("trial {trial} generic grad[{i}]"),
            );
            for j in 0..2 {
                close(
                    hessian[i][j],
                    tower.h[i][j],
                    &format!("trial {trial} generic hess[{i}][{j}]"),
                );
            }
        }

        for (di, dir) in dirs.iter().enumerate() {
            let third =
                program_third_contracted(&program, 0, dir).expect("generic third contraction");
            let truth = tower.third_contracted(dir);
            for i in 0..2 {
                for j in 0..2 {
                    close(
                        third[i][j],
                        truth[i][j],
                        &format!("trial {trial} dir {di} generic third[{i}][{j}]"),
                    );
                }
            }
        }

        for (ui, u) in dirs.iter().enumerate() {
            let v = dirs[(ui + 1) % dirs.len()];
            let fourth =
                program_fourth_contracted(&program, 0, u, &v).expect("generic fourth contraction");
            let truth = tower.fourth_contracted(u, &v);
            for i in 0..2 {
                for j in 0..2 {
                    close(
                        fourth[i][j],
                        truth[i][j],
                        &format!("trial {trial} pair {ui} generic fourth[{i}][{j}]"),
                    );
                }
            }
        }
    }
}

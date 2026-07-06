//! Gamma-dispersion Taylor-jet oracle (#932): the `ln_gamma` / `digamma`
//! composition path, pinned against an INDEPENDENT finite-difference oracle.
//!
//! Issue #932 builds a fixed-order truncated-Taylor jet algebra so every
//! family's [`gam_models::row_kernel::RowKernel`]-shaped derivative tower
//! (value / ∇ / H / contracted-third / contracted-fourth) is MECHANICALLY
//! derived from a single row-NLL expression rather than hand-written. The
//! reusable machinery already lives in the tree
//! ([`crate::jet_tower::RowNllProgramGeneric`], the packed
//! [`crate::jet_scalar::Order2`] / [`crate::jet_scalar::OneSeed`] /
//! [`crate::jet_scalar::TwoSeed`] scalars, the dense [`crate::jet_tower::Tower4`],
//! and the channel comparator [`crate::jet_tower::verify_kernel_channels`]), and
//! the elementary Gaussian / Poisson oracles pin the `exp` / `ln` composition
//! paths against external hand calculus.
//!
//! What those elementary oracles do NOT exercise is the SPECIAL-FUNCTION
//! composition path: the log-gamma normalizer `ln Γ(a)` with a
//! parameter-dependent argument `a`, whose derivative tower threads the
//! hand-certified polygamma stacks ([`crate::jet_tower::ln_gamma_derivative_stack`]
//! / [`crate::jet_tower::digamma_derivative_stack`]) through the multivariate
//! Faà di Bruno composition. That path is what the Gamma / negative-binomial
//! GAMLSS dispersion families ride, and a bug there (a mis-placed stack entry, a
//! dropped composition cross-term, an inaccurate polygamma) would silently
//! corrupt every dispersion fit. This module closes that oracle gap.
//!
//! # The model
//!
//! Rigby–Stasinopoulos Gamma ("GA"): mean `μ > 0`, scale `σ > 0`, variance
//! `μ² σ²`, shape `a = 1/σ²`. Two primaries `p = (p₀, p₁)` with the natural
//! GAMLSS links `μ = e^{p₀}` (`p₀ = log μ`) and `σ = e^{p₁}` (`p₁ = log σ`), so
//! `a = e^{−2 p₁}` and `σ² μ = e^{p₀ + 2 p₁}`. The row NLL (`−log f`) is
//!
//! ```text
//!   ℓ(p) = −(a − 1) log y + y / (σ² μ) + a · log(σ² μ) + log Γ(a)
//!        = −(a − 1) log y + y e^{−(p₀ + 2 p₁)} + a (p₀ + 2 p₁) + log Γ(a),
//! ```
//!
//! with `a = e^{−2 p₁}`. The `log Γ(a)` term makes the tower a genuine
//! special-function composition with a NONLINEAR inner argument (`a` is not
//! linear in `p₁`), so every order of the Faà di Bruno chain — `ψ · a′`,
//! `ψ′ · a′² + ψ · a″`, … — is exercised, and every `p₁`-block of the tower
//! (including the `∂p₀∂p₁` cross blocks, the #736 bug shape) is nonzero.
//!
//! # Why the oracle is independent
//!
//! The comparand is a tensor-product 5-point central-difference tower of the
//! SAME row NLL evaluated in plain `f64`, whose `log Γ` term calls `statrs`'
//! `ln_gamma` DIRECTLY (never the jet's polygamma stacks). Finite differences
//! share no code with the truncated-Taylor algebra: they catch a bug in the
//! Leibniz / Faà di Bruno combinatorics, in the `.ln_gamma()` / `.digamma()`
//! stack windowing, AND in the polygamma stack accuracy — any of which would be
//! invisible to a jet-vs-jet self-check. Each fourth-order partial is
//! Richardson-extrapolated (two step sizes) so the FD truncation floor is well
//! below the asserted tolerance.
//!
//! A second, EXACT test pins the stack *placement* directly: with a purely
//! affine inner argument `x(p) = c₀ p₀ + c₁ p₁ + c₂`, the composite
//! `ln Γ(x)` (and `ψ(x)`) has the closed tower `∂^α = (∏ c) · f^{(|α|)}(x₀)`,
//! so the jet's every channel must equal `c^α` times the certified stack entry
//! to ~1e-11 — a bit-tight check that each stack derivative lands at its correct
//! tensor order.

use crate::jet_scalar::JetScalar;
use crate::jet_tower::{
    Tower4, digamma_derivative_stack, generic_fourth_contracted, generic_full_tower,
    generic_row_kernel, generic_third_contracted, ln_gamma_derivative_stack,
};

/// A tiny deterministic LCG so the test points are pseudo-random yet fixed
/// across runs (NO `rand`, NO clock seeding — per the #932 rules).
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

/// One Gamma row fixture: response `y > 0` and the base primaries `(p₀, p₁)`
/// (`p₀ = log μ`, `p₁ = log σ`) at which the row tower is linearized.
#[derive(Clone, Copy, Debug)]
struct GammaRow {
    /// Observed positive response `yᵢ`.
    y: f64,
    /// Log-mean primary `p₀ = log μᵢ`.
    p0: f64,
    /// Log-scale primary `p₁ = log σᵢ`.
    p1: f64,
}

/// The Rigby–Stasinopoulos Gamma row NLL, written ONCE as a generic
/// [`RowNllProgramGeneric<2>`](crate::jet_tower::RowNllProgramGeneric) over the
/// jet scalar `S`. The body uses ONLY [`JetScalar`] ops (`scale`, `add`, `exp`,
/// `ln_gamma`); the per-row data (`y`, `log y`) enters as plain `f64` constants —
/// the single source of truth from which every derivative channel is then exact
/// by construction.
struct GammaLocScaleRow {
    rows: Vec<GammaRow>,
}

impl crate::jet_tower::RowNllProgramGeneric<2> for GammaLocScaleRow {
    fn n_rows(&self) -> usize {
        self.rows.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
        let r = self
            .rows
            .get(row)
            .ok_or_else(|| format!("GammaLocScaleRow: row {row} out of range"))?;
        // Primary order matches the f64 oracle: index 0 = p₀ = log μ, 1 = p₁ = log σ.
        Ok([r.p0, r.p1])
    }

    fn row_nll_generic<S: JetScalar<2>>(&self, row: usize, p: &[S; 2]) -> Result<S, String> {
        let data = self
            .rows
            .get(row)
            .ok_or_else(|| format!("GammaLocScaleRow: row {row} out of range"))?;
        let p0 = &p[0];
        let p1 = &p[1];
        // a = 1/σ² = e^{−2 p₁}  (the nonlinear ln Γ argument).
        let a = p1.scale(-2.0).exp();
        // q = log(σ² μ) = p₀ + 2 p₁  (linear).
        let q = p0.add(&p1.scale(2.0));
        // y / (σ² μ) = y · e^{−q}.
        let data_over_scale = S::constant(data.y).mul(&q.neg().exp());
        // a · q  (the a·log(σ²μ) term).
        let a_times_q = a.mul(&q);
        // ln Γ(a)  — the special-function composition under test.
        let ln_gamma_a = a.ln_gamma();
        // −(a − 1) log y = −a·log y + log y.
        let log_y = data.y.ln();
        let neg_a_log_y = a.scale(-log_y);
        // ℓ = −a log y + log y + y e^{−q} + a q + ln Γ(a).
        Ok(neg_a_log_y
            .add(&S::constant(log_y))
            .add(&data_over_scale)
            .add(&a_times_q)
            .add(&ln_gamma_a))
    }
}

/// The Gamma row NLL evaluated in plain `f64` — the INDEPENDENT comparand for
/// the finite-difference oracle. The `log Γ(a)` term calls `statrs`' `ln_gamma`
/// DIRECTLY, sharing no code with the jet's polygamma derivative stacks.
fn gamma_nll_f64(row: &GammaRow, p0: f64, p1: f64) -> f64 {
    let a = (-2.0 * p1).exp();
    let q = p0 + 2.0 * p1;
    let log_y = row.y.ln();
    -a * log_y + log_y + row.y * (-q).exp() + a * q + statrs::function::gamma::ln_gamma(a)
}

/// 1D central-difference stencils on the symmetric node set `{−2,−1,0,1,2}·h`.
/// Each entry is `(coefficients, 1/normalizer, truncation_order)` for the
/// derivative order equal to its index (0..=4). The `f^{(n)}` estimate at step
/// `h` is `(Σ coeff·f(node)) / (normalizer · hⁿ)`, accurate to `O(h^trunc)`.
const STENCIL_COEFF: [[f64; 5]; 5] = [
    [0.0, 0.0, 1.0, 0.0, 0.0],       // order 0
    [1.0, -8.0, 0.0, 8.0, -1.0],     // order 1  (/12)
    [-1.0, 16.0, -30.0, 16.0, -1.0], // order 2  (/12)
    [-1.0, 2.0, 0.0, -2.0, 1.0],     // order 3  (/2)
    [1.0, -4.0, 6.0, -4.0, 1.0],     // order 4  (/1)
];
const STENCIL_NORM: [f64; 5] = [1.0, 12.0, 12.0, 2.0, 1.0];
/// Leading truncation order of each stencil above (`O(h^trunc)`).
const STENCIL_TRUNC: [i32; 5] = [64, 4, 4, 2, 2];

/// Mixed partial `∂^a_{p₀} ∂^b_{p₁} ℓ` at one step size `h`, via the tensor
/// product of the two 1D central-difference stencils. `a + b ≤ 4`.
fn fd_partial_at(row: &GammaRow, a: usize, b: usize, h: f64) -> f64 {
    let ca = STENCIL_COEFF[a];
    let cb = STENCIL_COEFF[b];
    let mut acc = 0.0;
    for (i, &cai) in ca.iter().enumerate() {
        if cai == 0.0 {
            continue;
        }
        for (j, &cbj) in cb.iter().enumerate() {
            if cbj == 0.0 {
                continue;
            }
            let x0 = row.p0 + (i as f64 - 2.0) * h;
            let x1 = row.p1 + (j as f64 - 2.0) * h;
            acc += cai * cbj * gamma_nll_f64(row, x0, x1);
        }
    }
    let denom = STENCIL_NORM[a] * STENCIL_NORM[b] * h.powi((a + b) as i32);
    acc / denom
}

/// Richardson-extrapolated mixed partial `∂^a_{p₀} ∂^b_{p₁} ℓ`. Combining two
/// step sizes cancels the leading `O(h^p)` truncation term (`p` is the minimum
/// truncation order over the DIFFERENTIATED directions), pushing the FD floor
/// well under the asserted tolerance even for the fourth-order channels.
fn fd_partial(row: &GammaRow, a: usize, b: usize) -> f64 {
    if a == 0 && b == 0 {
        return gamma_nll_f64(row, row.p0, row.p1);
    }
    // Truncation order limited by the coarsest differentiated direction.
    let mut p = i32::MAX;
    if a >= 1 {
        p = p.min(STENCIL_TRUNC[a]);
    }
    if b >= 1 {
        p = p.min(STENCIL_TRUNC[b]);
    }
    // The fourth-order tensor-product stencil subtracts nearly equal row-NLL
    // values before dividing by `h^4` (or `h^(a+b)` for mixed entries).  A
    // `1e-2` step is inside the roundoff-dominated side of that balance for the
    // p₀⁴ channel of the deterministic fixtures: the Gamma row NLL contains an
    // `O(1)` log-normalizer, while the p₀⁴ signal is only the
    // `y·exp(-(p₀+2p₁))` term, so cancellation in the constant-in-p₀ pieces can
    // exceed the oracle's stated band.  Use a dyadic log-primary step instead:
    // `2^-5` is still local on the fixture domain (`p ∈ [-0.5, 0.5] ×
    // [-0.3, 0.3]`, stencil reaches only `±1/16` at the coarse level) but keeps
    // the fourth-order denominator large enough that the Richardson oracle is
    // measuring the row expression rather than floating-point cancellation.
    let h = 1.0 / 32.0;
    let coarse = fd_partial_at(row, a, b, h);
    let fine = fd_partial_at(row, a, b, h * 0.5);
    let two_p = 2f64.powi(p);
    (two_p * fine - coarse) / (two_p - 1.0)
}

/// Deterministic Gamma fixtures with moderate, well-scaled primaries so the FD
/// stencils and the polygamma stacks are both comfortably in their accurate
/// regime (`a = e^{−2p₁} ∈ [0.5, 1.9]`, `y ∈ [0.5, 3]`).
fn gamma_fixtures(seed: u64, n: usize) -> Vec<GammaRow> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|_| GammaRow {
            y: rng.uniform(0.5, 3.0),
            p0: rng.uniform(-0.5, 0.5),
            p1: rng.uniform(-0.3, 0.3),
        })
        .collect()
}

/// The mechanically jet-derived Gamma dispersion tower (value / ∇ / H / t3 / t4)
/// must equal an INDEPENDENT `statrs`-based finite-difference tower of the same
/// row NLL, tensor entry by tensor entry, at every order through the fourth.
/// This is the FD-free-*algebra*, FD-*oracle* exactness check #932 asks for,
/// applied to the special-function (`ln Γ`) composition path the elementary
/// Gaussian / Poisson oracles never touch.
#[test]
fn gamma_dispersion_jet_tower_matches_independent_fd_oracle() {
    let rows = gamma_fixtures(0x9322_0203_6a6d_6d61, 20);
    let program = GammaLocScaleRow { rows: rows.clone() };

    // Richardson-extrapolated FD floor sits near ~1e-8 for the fourth-order
    // channels on these well-scaled points; 5e-6 is a strong, method-honest
    // bound that a sign flip / dropped term / mis-windowed stack blows past.
    const REL_TOL: f64 = 5e-6;
    const ATOL: f64 = 1e-7;
    let close = |jet: f64, fd: f64, label: &str| {
        let band = ATOL + REL_TOL * jet.abs().max(fd.abs());
        assert!(
            (jet - fd).abs() <= band,
            "{label}: jet {jet:+.12e} vs FD {fd:+.12e} (|Δ| {:.3e} > band {band:.3e})",
            (jet - fd).abs()
        );
    };

    for (row, fixture) in rows.iter().enumerate() {
        let tower: Tower4<2> = generic_full_tower(&program, row).expect("gamma jet tower");

        // Value.
        close(
            tower.v,
            fd_partial(fixture, 0, 0),
            &format!("row {row} value"),
        );

        // Gradient: g[i] = ∂_{p_i} ℓ.
        for i in 0..2 {
            let (a, b) = order_counts(&[i]);
            close(
                tower.g[i],
                fd_partial(fixture, a, b),
                &format!("row {row} grad[{i}]"),
            );
        }
        // Hessian: h[i][j] = ∂_{p_i}∂_{p_j} ℓ (symmetry implicitly checked).
        for i in 0..2 {
            for j in 0..2 {
                let (a, b) = order_counts(&[i, j]);
                close(
                    tower.h[i][j],
                    fd_partial(fixture, a, b),
                    &format!("row {row} hess[{i}][{j}]"),
                );
            }
        }
        // Third tensor.
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let (a, b) = order_counts(&[i, j, k]);
                    close(
                        tower.t3[i][j][k],
                        fd_partial(fixture, a, b),
                        &format!("row {row} t3[{i}][{j}][{k}]"),
                    );
                }
            }
        }
        // Fourth tensor.
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        let (a, b) = order_counts(&[i, j, k, l]);
                        close(
                            tower.t4[i][j][k][l],
                            fd_partial(fixture, a, b),
                            &format!("row {row} t4[{i}][{j}][{k}][{l}]"),
                        );
                    }
                }
            }
        }
    }
}

/// Split a list of primary axes into `(count of axis 0, count of axis 1)` — the
/// `(a, b)` differentiation orders the FD tensor-product stencil consumes for
/// the fully-symmetric tower entry whose indices are `axes`.
fn order_counts(axes: &[usize]) -> (usize, usize) {
    let a = axes.iter().filter(|&&x| x == 0).count();
    let b = axes.iter().filter(|&&x| x == 1).count();
    (a, b)
}

/// The PRODUCTION packed scalars — `Order2` (value/∇/H), `OneSeed` (contracted
/// third), `TwoSeed` (contracted fourth) — evaluated on the SAME single
/// `row_nll_generic` Gamma expression must reproduce the dense `Tower4`
/// contractions. This pins the actual cutover path a `ln Γ`-carrying family
/// would use (the small packed scalars, with the contraction directions folded
/// into the nilpotent seeds) rather than the dense oracle tower.
#[test]
fn gamma_dispersion_packed_scalars_match_dense_tower_contractions() {
    let rows = gamma_fixtures(0x0bad_c0de_6a6d_6d61, 16);
    let program = GammaLocScaleRow { rows: rows.clone() };

    let third_dirs: [[f64; 2]; 3] = [[0.9, -0.5], [-1.1, 0.3], [0.2, 1.4]];

    const REL_TOL: f64 = 1e-11;
    let close = |a: f64, b: f64, label: &str| {
        let band = REL_TOL + REL_TOL * a.abs().max(b.abs());
        assert!(
            (a - b).abs() <= band,
            "{label}: packed {a:+.15e} vs dense {b:+.15e} (band {band:.3e})"
        );
    };

    for row in 0..rows.len() {
        let tower: Tower4<2> = generic_full_tower(&program, row).expect("tower");

        // Order2: value / gradient / Hessian via the production packed scalar.
        let (v, g, h) = generic_row_kernel(&program, row).expect("Order2 channel");
        close(v, tower.v, &format!("row {row} Order2 value"));
        for i in 0..2 {
            close(g[i], tower.g[i], &format!("row {row} Order2 grad[{i}]"));
            for j in 0..2 {
                close(
                    h[i][j],
                    tower.h[i][j],
                    &format!("row {row} Order2 hess[{i}][{j}]"),
                );
            }
        }

        // OneSeed: contracted third Σ_c ℓ_{abc}·dir_c vs the dense t3 contraction.
        for (di, dir) in third_dirs.iter().enumerate() {
            let third = generic_third_contracted(&program, row, dir).expect("OneSeed third");
            let truth = tower.third_contracted(dir);
            for i in 0..2 {
                for j in 0..2 {
                    close(
                        third[i][j],
                        truth[i][j],
                        &format!("row {row} dir {di} OneSeed third[{i}][{j}]"),
                    );
                }
            }
        }

        // TwoSeed: contracted fourth Σ_{cd} ℓ_{abcd}·u_c·v_d vs the dense t4.
        for (ui, u) in third_dirs.iter().enumerate() {
            let v = third_dirs[(ui + 1) % third_dirs.len()];
            let fourth = generic_fourth_contracted(&program, row, u, &v).expect("TwoSeed fourth");
            let truth = tower.fourth_contracted(u, &v);
            for i in 0..2 {
                for j in 0..2 {
                    close(
                        fourth[i][j],
                        truth[i][j],
                        &format!("row {row} pair {ui} TwoSeed fourth[{i}][{j}]"),
                    );
                }
            }
        }
    }
}

/// An affine inner argument `x(p) = c₀ p₀ + c₁ p₁ + c₂` composed with a unary
/// special function `f`, written ONCE over the jet scalar. Because the inner
/// map is affine, the composite tower is exactly `∂^α (f∘x) = (∏ c_axis)·
/// f^{(|α|)}(x₀)` — a closed form that pins each certified stack entry to its
/// correct tensor order WITHOUT re-deriving Faà di Bruno.
struct AffineComposeRow {
    c0: f64,
    c1: f64,
    c2: f64,
    p0: f64,
    p1: f64,
    /// `false` → compose `ln Γ`; `true` → compose `ψ` (digamma).
    digamma: bool,
}

impl crate::jet_tower::RowNllProgramGeneric<2> for AffineComposeRow {
    fn n_rows(&self) -> usize {
        1
    }
    fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
        if row >= self.n_rows() {
            return Err(format!("AffineComposeRow: row {row} out of range"));
        }
        Ok([self.p0, self.p1])
    }
    fn row_nll_generic<S: JetScalar<2>>(&self, row: usize, p: &[S; 2]) -> Result<S, String> {
        if row >= self.n_rows() {
            return Err(format!("AffineComposeRow: row {row} out of range"));
        }
        let x = p[0]
            .scale(self.c0)
            .add(&p[1].scale(self.c1))
            .add(&S::constant(self.c2));
        Ok(if self.digamma {
            x.digamma()
        } else {
            x.ln_gamma()
        })
    }
}

/// The certified `ln_gamma` / `digamma` derivative stacks must land at their
/// correct tower orders. With an AFFINE inner argument the composite tower is
/// the closed form `∂^α = c^α · f^{(|α|)}(x₀)`, so this test compares the jet's
/// every channel to `c`-power × the SAME certified stack the composition
/// consumes — an EXACT (~1e-11) check of stack placement, independent of the
/// Faà di Bruno cross-terms (which the nonlinear FD oracle above exercises).
#[test]
fn affine_special_function_composition_places_certified_stack_by_order() {
    // Choose (c, p, offset) so the inner argument x₀ is a comfortable, positive
    // value where both stacks are accurate.
    let cases: [(bool, f64, f64, f64, f64, f64); 4] = [
        // (digamma, c0, c1, c2, p0, p1)
        (false, 0.7, -1.3, 4.0, 0.5, -0.25),
        (false, 1.1, 0.4, 3.0, -0.3, 0.6),
        (true, 0.6, -0.9, 5.0, 0.2, 0.4),
        (true, -0.8, 1.2, 6.0, -0.4, 0.1),
    ];

    const REL_TOL: f64 = 1e-11;
    let close = |jet: f64, closed: f64, label: &str| {
        let band = REL_TOL + REL_TOL * jet.abs().max(closed.abs());
        assert!(
            (jet - closed).abs() <= band,
            "{label}: jet {jet:+.15e} vs closed {closed:+.15e} (band {band:.3e})"
        );
    };

    for (ci, &(digamma, c0, c1, c2, p0, p1)) in cases.iter().enumerate() {
        let program = AffineComposeRow {
            c0,
            c1,
            c2,
            p0,
            p1,
            digamma,
        };
        let x0 = c0 * p0 + c1 * p1 + c2;
        // The SAME certified stack the `.ln_gamma()` / `.digamma()` compose reads.
        let stack = if digamma {
            digamma_derivative_stack(x0)
        } else {
            ln_gamma_derivative_stack(x0)
        };
        let c = [c0, c1];
        let tower: Tower4<2> = generic_full_tower(&program, 0).expect("affine tower");

        let tag = if digamma { "digamma" } else { "ln_gamma" };
        // Value.
        close(tower.v, stack[0], &format!("case {ci} {tag} value"));
        // First order: g[i] = c_i · f'(x₀).
        for i in 0..2 {
            close(
                tower.g[i],
                c[i] * stack[1],
                &format!("case {ci} {tag} grad[{i}]"),
            );
        }
        // Second order.
        for i in 0..2 {
            for j in 0..2 {
                close(
                    tower.h[i][j],
                    c[i] * c[j] * stack[2],
                    &format!("case {ci} {tag} hess[{i}][{j}]"),
                );
            }
        }
        // Third order.
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    close(
                        tower.t3[i][j][k],
                        c[i] * c[j] * c[k] * stack[3],
                        &format!("case {ci} {tag} t3[{i}][{j}][{k}]"),
                    );
                }
            }
        }
        // Fourth order.
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        close(
                            tower.t4[i][j][k][l],
                            c[i] * c[j] * c[k] * c[l] * stack[4],
                            &format!("case {ci} {tag} t4[{i}][{j}][{k}][{l}]"),
                        );
                    }
                }
            }
        }
    }
}

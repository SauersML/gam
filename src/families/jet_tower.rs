//! Taylor-jet tower algebra: write each family's row log-likelihood ONCE,
//! derive the entire `RowKernel<K>` derivative tower mechanically (#932).
//!
//! # The object
//!
//! [`Tower4<K>`] is a truncated multivariate Taylor scalar in `K` primary
//! variables, carrying the value and ALL partial derivatives through fourth
//! order as full (unsymmetrized) tensors:
//!
//! ```text
//!   v        ℓ
//!   g[a]     ∂ℓ/∂p_a
//!   h[a][b]  ∂²ℓ/∂p_a∂p_b
//!   t3[abc]  ∂³ℓ/∂p_a∂p_b∂p_c
//!   t4[abcd] ∂⁴ℓ/∂p_a∂p_b∂p_c∂p_d
//! ```
//!
//! Arithmetic (`+ − × ÷`, scalar mixes) propagates the tower by the exact
//! Leibniz rule; unary transcendentals propagate by the exact multivariate
//! Faà di Bruno formula given a `[f, f′, f″, f‴, f⁗]` stack evaluated at the
//! inner value. This is truncated Taylor ALGEBRA — exact derivatives of the
//! evaluated expression, not finite differences, not an approximation —
//! fully compatible with the exact-REML-only policy.
//!
//! One evaluation of a row NLL program at seeded variables yields, in a
//! single pass, every channel the [`super::row_kernel::RowKernel`] trait
//! demands: `row_kernel` (value/∇/H), `row_third_contracted(dir)` (contract
//! `t3` with `dir`), and `row_fourth_contracted(u, v)` (contract `t4` with
//! `u` and `v`). The directional cross-channels that hand-written towers
//! drop (#736's residual gap) cannot be dropped here: there is no separate
//! "channel" to forget — every derivative of the one expression is carried.
//!
//! # Why this exists (the bug genus)
//!
//! Every family today hand-writes its tower: value in one function,
//! gradient in another, `pdfthird_derivative`/`pdffourth_derivative`,
//! entry/exit-specific cross blocks — thousands of lines of calculus that
//! drift. #736 was a sign flip in a hand-written cross-Hessian block,
//! invisible until a new consumer touched it; #948 is a derivative path
//! that is not the derivative of the evaluated row loss (clamped-μ
//! surrogate); the objective↔gradient desync class is the same disease at
//! the criterion level. A tower-derived kernel is exact-by-construction:
//! the value channel IS the production loss expression, so its derivative
//! channels cannot desync from it.
//!
//! # Relation to `jet_partitions::MultiDirJet`
//!
//! The tree already carries a *directional* jet (bitmask coefficients over
//! distinct seeded directions, heap-allocated, Bell-partition compose) used
//! inside the marginal-slope and latent-survival families. It answers "the
//! derivative along THESE specific directions" and must be re-seeded and
//! re-evaluated per direction tuple (e.g. 10 symmetric `(a,b)` pairs for a
//! K=4 fourth contraction). `Tower4` answers ALL of them from one
//! evaluation: contraction happens AFTER differentiation, as plain linear
//! algebra on the stored tensors. Use `MultiDirJet` when you need a handful
//! of directions of a huge-K expression; use `Tower4` when you need the
//! complete small-K tower — which is exactly the `RowKernel<K≤4>` shape.
//! The `[f64; 5]` unary-derivative stacks
//! (`unary_derivatives_neglog_phi`, …) are signature-compatible with
//! [`Tower4::compose_unary`], so the families' existing special-function
//! stacks are directly reusable.
//!
//! # Stability discipline (why this is NOT autodiff)
//!
//! Differentiating the primal code path inherits its instabilities: a jet
//! pushed through a naive `ln(1 + e^η)` is garbage in the saturated tail
//! even though the true derivative σ(η) is benign there. This module
//! therefore splits responsibility: **humans own primitive stability,
//! the algebra owns combinatorics**. Tail-critical special functions enter
//! a program ONLY as hand-certified `[f64; 5]` derivative stacks through
//! [`Tower4::compose_unary`] — the same stacks the families already write
//! (`unary_derivatives_neglog_phi` and friends, built on erfcx/log_ndtr) —
//! and the tower mechanizes only the Leibniz/Faà di Bruno composition,
//! which is where hand-written towers actually fail (#736 was a
//! composition sign flip, not a primitive error). Program authors must use
//! a stable primitive stack wherever the f64 production loss does; the
//! convenience methods (`exp`, `ln`, `sqrt`, …) are for expressions whose
//! arguments are tame by construction.
//!
//! # Storage convention
//!
//! Tensors are stored FULL, not symmetric-packed: `t4` for K=4 is 256
//! doubles where 35 would do. This is deliberate clarity-over-speed for the
//! oracle role — indexing is trivially auditable, contraction loops are
//! obvious, and the redundancy is itself a checked invariant (the algebra
//! only ever writes symmetric values). Symmetric packing is a later,
//! profile-justified optimization behind the same API.
//!
//! # Deployment ladder (#932)
//!
//! 1. This module: the algebra + the program seam + the oracle.
//! 2. Universal oracle: every hand-written `RowKernel` gains a CI test
//!    asserting channel-by-channel agreement with a `RowNllProgram` written
//!    once — see [`verify_kernel_channels`]. This alone would have caught
//!    #736 at introduction.
//! 3. Migrate error-dense / cold towers to [`derived_row_kernel`] et al.;
//!    keep hand-tuned hot paths, now verified against the single-expression
//!    truth instead of being the only definition.
//! 4. New families (#914/#916/#917 ZI/ordinal/expectile, #921's location-
//!    scale port) implement ONLY `RowNllProgram` and get an exact
//!    fourth-order tower for the price of writing the likelihood.

/// Truncated fourth-order multivariate Taylor scalar in `K` variables.
///
/// See the module documentation for semantics and conventions. `Copy` is
/// intentional despite the size (2 KiB at K=4): towers are per-row
/// temporaries that live entirely in registers/stack during a row program,
/// and value semantics keep program code readable (`a * b + c`).
#[derive(Clone, Copy, Debug)]
pub struct Tower4<const K: usize> {
    /// Value ℓ.
    pub v: f64,
    /// Gradient ∂ℓ/∂p_a.
    pub g: [f64; K],
    /// Hessian ∂²ℓ/∂p_a∂p_b (symmetric).
    pub h: [[f64; K]; K],
    /// Third derivatives ∂³ℓ/∂p_a∂p_b∂p_c (fully symmetric).
    pub t3: [[[f64; K]; K]; K],
    /// Fourth derivatives ∂⁴ℓ/∂p_a∂p_b∂p_c∂p_d (fully symmetric).
    pub t4: [[[[f64; K]; K]; K]; K],
}

impl<const K: usize> Tower4<K> {
    /// The additive identity.
    pub fn zero() -> Self {
        Self {
            v: 0.0,
            g: [0.0; K],
            h: [[0.0; K]; K],
            t3: [[[0.0; K]; K]; K],
            t4: [[[[0.0; K]; K]; K]; K],
        }
    }

    /// A constant: value `c`, all derivatives zero.
    pub fn constant(c: f64) -> Self {
        let mut out = Self::zero();
        out.v = c;
        out
    }

    /// The seeded variable `p_idx` with current value `value`:
    /// unit first derivative in slot `idx`, zero elsewhere and above.
    pub fn variable(value: f64, idx: usize) -> Self {
        let mut out = Self::constant(value);
        out.g[idx] = 1.0;
        out
    }

    /// Exact truncated Leibniz product.
    ///
    /// Each output derivative `D_S(ab)` is the sum over all subsets of the
    /// index multiset of `D_T(a) · D_{S∖T}(b)` — written out explicitly per
    /// order so every term is auditable against the general formula.
    pub fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        let mut out = Self::zero();
        out.v = a.v * b.v;
        for i in 0..K {
            out.g[i] = a.v * b.g[i] + a.g[i] * b.v;
        }
        for i in 0..K {
            for j in 0..K {
                out.h[i][j] = a.v * b.h[i][j] + a.g[i] * b.g[j] + a.g[j] * b.g[i] + a.h[i][j] * b.v;
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    out.t3[i][j][k] = a.v * b.t3[i][j][k]
                        + a.g[i] * b.h[j][k]
                        + a.g[j] * b.h[i][k]
                        + a.g[k] * b.h[i][j]
                        + a.h[j][k] * b.g[i]
                        + a.h[i][k] * b.g[j]
                        + a.h[i][j] * b.g[k]
                        + a.t3[i][j][k] * b.v;
                }
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    for l in 0..K {
                        // 16 subsets of {i,j,k,l}: ∅ | singles | pairs | triples | full.
                        out.t4[i][j][k][l] = a.v * b.t4[i][j][k][l]
                            + a.g[i] * b.t3[j][k][l]
                            + a.g[j] * b.t3[i][k][l]
                            + a.g[k] * b.t3[i][j][l]
                            + a.g[l] * b.t3[i][j][k]
                            + a.h[i][j] * b.h[k][l]
                            + a.h[i][k] * b.h[j][l]
                            + a.h[i][l] * b.h[j][k]
                            + a.h[j][k] * b.h[i][l]
                            + a.h[j][l] * b.h[i][k]
                            + a.h[k][l] * b.h[i][j]
                            + a.t3[i][j][k] * b.g[l]
                            + a.t3[i][j][l] * b.g[k]
                            + a.t3[i][k][l] * b.g[j]
                            + a.t3[j][k][l] * b.g[i]
                            + a.t4[i][j][k][l] * b.v;
                    }
                }
            }
        }
        out
    }

    /// Exact multivariate Faà di Bruno composition `f ∘ self`.
    ///
    /// `d = [f(u), f′(u), f″(u), f‴(u), f⁗(u)]` evaluated at `u = self.v` —
    /// the SAME `[f64; 5]` stack shape the families' existing
    /// `unary_derivatives_*` helpers produce, so those special-function
    /// stacks (Φ, log-Φ, normal pdf, …) plug in directly.
    ///
    /// The order-m output sums over the set partitions of the m indices
    /// (Bell(3) = 5 terms at order 3, Bell(4) = 15 at order 4), grouped by
    /// block count: each partition into r blocks contributes
    /// `f⁽ʳ⁾ · Π_blocks D_block(u)`.
    pub fn compose_unary(&self, d: [f64; 5]) -> Self {
        let u = self;
        let mut out = Self::zero();
        out.v = d[0];
        for i in 0..K {
            out.g[i] = d[1] * u.g[i];
        }
        for i in 0..K {
            for j in 0..K {
                out.h[i][j] = d[1] * u.h[i][j] + d[2] * u.g[i] * u.g[j];
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    out.t3[i][j][k] = d[1] * u.t3[i][j][k]
                        + d[2] * (u.g[i] * u.h[j][k] + u.g[j] * u.h[i][k] + u.g[k] * u.h[i][j])
                        + d[3] * u.g[i] * u.g[j] * u.g[k];
                }
            }
        }
        for i in 0..K {
            for j in 0..K {
                for k in 0..K {
                    for l in 0..K {
                        out.t4[i][j][k][l] = d[1] * u.t4[i][j][k][l]
                            + d[2]
                                * (u.g[i] * u.t3[j][k][l]
                                    + u.g[j] * u.t3[i][k][l]
                                    + u.g[k] * u.t3[i][j][l]
                                    + u.g[l] * u.t3[i][j][k]
                                    + u.h[i][j] * u.h[k][l]
                                    + u.h[i][k] * u.h[j][l]
                                    + u.h[i][l] * u.h[j][k])
                            + d[3]
                                * (u.g[i] * u.g[j] * u.h[k][l]
                                    + u.g[i] * u.g[k] * u.h[j][l]
                                    + u.g[i] * u.g[l] * u.h[j][k]
                                    + u.g[j] * u.g[k] * u.h[i][l]
                                    + u.g[j] * u.g[l] * u.h[i][k]
                                    + u.g[k] * u.g[l] * u.h[i][j])
                            + d[4] * u.g[i] * u.g[j] * u.g[k] * u.g[l];
                    }
                }
            }
        }
        out
    }

    /// Multiply every channel by a plain scalar.
    pub fn scale(&self, s: f64) -> Self {
        let mut out = *self;
        out.v *= s;
        for i in 0..K {
            out.g[i] *= s;
            for j in 0..K {
                out.h[i][j] *= s;
                for k in 0..K {
                    out.t3[i][j][k] *= s;
                    for l in 0..K {
                        out.t4[i][j][k][l] *= s;
                    }
                }
            }
        }
        out
    }

    /// e^self.
    pub fn exp(&self) -> Self {
        let e = self.v.exp();
        self.compose_unary([e, e, e, e, e])
    }

    /// ln(self). Caller guarantees positivity (likelihood programs do).
    pub fn ln(&self) -> Self {
        let u = self.v;
        let r = 1.0 / u;
        self.compose_unary([u.ln(), r, -r * r, 2.0 * r * r * r, -6.0 * r * r * r * r])
    }

    /// 1/self.
    pub fn recip(&self) -> Self {
        let r = 1.0 / self.v;
        let r2 = r * r;
        self.compose_unary([r, -r2, 2.0 * r2 * r, -6.0 * r2 * r2, 24.0 * r2 * r2 * r])
    }

    /// √self. Caller guarantees positivity.
    pub fn sqrt(&self) -> Self {
        let u = self.v;
        let s = u.sqrt();
        self.compose_unary([
            s,
            0.5 / s,
            -0.25 / (u * s),
            0.375 / (u * u * s),
            -0.9375 / (u * u * u * s),
        ])
    }

    /// self^a for real exponent `a`. Caller guarantees a positive base.
    pub fn powf(&self, a: f64) -> Self {
        let u = self.v;
        let f0 = u.powf(a);
        let f1 = a * u.powf(a - 1.0);
        let f2 = a * (a - 1.0) * u.powf(a - 2.0);
        let f3 = a * (a - 1.0) * (a - 2.0) * u.powf(a - 3.0);
        let f4 = a * (a - 1.0) * (a - 2.0) * (a - 3.0) * u.powf(a - 4.0);
        self.compose_unary([f0, f1, f2, f3, f4])
    }

    /// ln Γ(self). Caller guarantees positivity.
    pub fn ln_gamma(&self) -> Self {
        self.compose_unary(ln_gamma_derivative_stack(self.v))
    }

    /// ψ(self), the digamma function. Caller guarantees positivity.
    pub fn digamma(&self) -> Self {
        self.compose_unary(digamma_derivative_stack(self.v))
    }

    /// ψ′(self), the trigamma function. Caller guarantees positivity.
    pub fn trigamma(&self) -> Self {
        self.compose_unary(trigamma_derivative_stack(self.v))
    }

    /// Contract `t3` with one primary-space direction:
    /// `out[a][b] = Σ_c t3[a][b][c] · dir[c]` — exactly the
    /// `row_third_contracted` shape.
    pub fn third_contracted(&self, dir: &[f64; K]) -> [[f64; K]; K] {
        let mut out = [[0.0; K]; K];
        for a in 0..K {
            for b in 0..K {
                let mut acc = 0.0;
                for c in 0..K {
                    acc += self.t3[a][b][c] * dir[c];
                }
                out[a][b] = acc;
            }
        }
        out
    }

    /// Contract `t4` with two primary-space directions:
    /// `out[a][b] = Σ_{c,d} t4[a][b][c][d] · u[c] · v[d]` — exactly the
    /// `row_fourth_contracted` shape.
    pub fn fourth_contracted(&self, u: &[f64; K], w: &[f64; K]) -> [[f64; K]; K] {
        let mut out = [[0.0; K]; K];
        for a in 0..K {
            for b in 0..K {
                let mut acc = 0.0;
                for c in 0..K {
                    for d in 0..K {
                        acc += self.t4[a][b][c][d] * u[c] * w[d];
                    }
                }
                out[a][b] = acc;
            }
        }
        out
    }
}

pub fn ln_gamma_derivative_stack(x: f64) -> [f64; 5] {
    [
        statrs::function::gamma::ln_gamma(x),
        statrs::function::gamma::digamma(x),
        polygamma_positive(1, x),
        polygamma_positive(2, x),
        polygamma_positive(3, x),
    ]
}

pub fn digamma_derivative_stack(x: f64) -> [f64; 5] {
    [
        statrs::function::gamma::digamma(x),
        polygamma_positive(1, x),
        polygamma_positive(2, x),
        polygamma_positive(3, x),
        polygamma_positive(4, x),
    ]
}

pub fn trigamma_derivative_stack(x: f64) -> [f64; 5] {
    [
        polygamma_positive(1, x),
        polygamma_positive(2, x),
        polygamma_positive(3, x),
        polygamma_positive(4, x),
        polygamma_positive(5, x),
    ]
}

fn polygamma_positive(order: usize, mut x: f64) -> f64 {
    if !(x.is_finite() && x > 0.0) {
        return f64::NAN;
    }
    let mut acc = 0.0;
    while x < POLYGAMMA_ASYMPTOTIC_MIN_X {
        acc += polygamma_recurrence_term(order, x);
        x += 1.0;
    }
    acc + polygamma_asymptotic(order, x)
}

const POLYGAMMA_ASYMPTOTIC_MIN_X: f64 = 20.0;
const BERNOULLI_EVEN: [(usize, f64); 10] = [
    (2, 1.0 / 6.0),
    (4, -1.0 / 30.0),
    (6, 1.0 / 42.0),
    (8, -1.0 / 30.0),
    (10, 5.0 / 66.0),
    (12, -691.0 / 2730.0),
    (14, 7.0 / 6.0),
    (16, -3617.0 / 510.0),
    (18, 43867.0 / 798.0),
    (20, -174611.0 / 330.0),
];

fn polygamma_recurrence_term(order: usize, x: f64) -> f64 {
    let sign = if order % 2 == 1 { 1.0 } else { -1.0 };
    sign * factorial(order) / x.powi((order + 1) as i32)
}

fn polygamma_asymptotic(order: usize, x: f64) -> f64 {
    if !(1..=5).contains(&order) {
        return f64::NAN;
    }

    let order_factorial = factorial(order);
    let leading_sign = if order % 2 == 1 { 1.0 } else { -1.0 };
    let mut out = leading_sign * factorial(order - 1) / x.powi(order as i32)
        + leading_sign * order_factorial / (2.0 * x.powi((order + 1) as i32));

    let bernoulli_sign = if order % 2 == 1 { 1.0 } else { -1.0 };
    for (bernoulli_order, bernoulli) in BERNOULLI_EVEN {
        let rising = rising_factorial(bernoulli_order, order);
        out += bernoulli_sign * bernoulli * rising / bernoulli_order as f64
            / x.powi((bernoulli_order + order) as i32);
    }
    out
}

fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0, |acc, k| acc * k as f64)
}

fn rising_factorial(start: usize, len: usize) -> f64 {
    (start..start + len).fold(1.0, |acc, k| acc * k as f64)
}

impl<const K: usize> std::ops::Add for Tower4<K> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        let mut out = self;
        out.v += o.v;
        for i in 0..K {
            out.g[i] += o.g[i];
            for j in 0..K {
                out.h[i][j] += o.h[i][j];
                for k in 0..K {
                    out.t3[i][j][k] += o.t3[i][j][k];
                    for l in 0..K {
                        out.t4[i][j][k][l] += o.t4[i][j][k][l];
                    }
                }
            }
        }
        out
    }
}

impl<const K: usize> std::ops::Sub for Tower4<K> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        self + o.scale(-1.0)
    }
}

impl<const K: usize> std::ops::Neg for Tower4<K> {
    type Output = Self;
    fn neg(self) -> Self {
        self.scale(-1.0)
    }
}

impl<const K: usize> std::ops::Mul for Tower4<K> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Tower4::mul(&self, &o)
    }
}

impl<const K: usize> std::ops::Div for Tower4<K> {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        Tower4::mul(&self, &o.recip())
    }
}

impl<const K: usize> std::ops::Add<f64> for Tower4<K> {
    type Output = Self;
    fn add(self, c: f64) -> Self {
        let mut out = self;
        out.v += c;
        out
    }
}

impl<const K: usize> std::ops::Sub<f64> for Tower4<K> {
    type Output = Self;
    fn sub(self, c: f64) -> Self {
        self + (-c)
    }
}

impl<const K: usize> std::ops::Mul<f64> for Tower4<K> {
    type Output = Self;
    fn mul(self, c: f64) -> Self {
        self.scale(c)
    }
}

// ── The program seam ─────────────────────────────────────────────────

/// A family's row negative log-likelihood written ONCE over tower scalars.
///
/// This is the single source of truth #932 asks for: the value channel of
/// the returned tower must BE the production row NLL (same branches, same
/// guards, same numerics), and every derivative channel is then exact by
/// construction. The linear Jacobian wiring (coefficients ↔ primaries) is
/// NOT part of this trait — it is family data, not calculus, and stays on
/// the `RowKernel` implementor.
pub trait RowNllProgram<const K: usize>: Send + Sync {
    /// Number of observations the program covers.
    fn n_rows(&self) -> usize;

    /// Current primary-scalar values for `row` (where to seed the tower).
    fn primaries(&self, row: usize) -> Result<[f64; K], String>;

    /// The row NLL evaluated on tower scalars. `p[a]` arrives pre-seeded as
    /// variable `a` at the current primary value; implementations combine
    /// them with `Tower4` arithmetic and per-row data (response, censoring
    /// indicators, offsets) entering as constants.
    fn row_nll(&self, row: usize, p: &[Tower4<K>; K]) -> Result<Tower4<K>, String>;
}

/// Evaluate a program's full tower at the current primaries for one row.
///
/// One call yields every `RowKernel` calculus channel; callers that need
/// several contractions of the same row should hold the returned tower and
/// contract repeatedly rather than re-evaluating.
pub fn evaluate_program<const K: usize, P: RowNllProgram<K> + ?Sized>(
    prog: &P,
    row: usize,
) -> Result<Tower4<K>, String> {
    let p = prog.primaries(row)?;
    let vars: [Tower4<K>; K] = std::array::from_fn(|a| Tower4::variable(p[a], a));
    prog.row_nll(row, &vars)
}

/// Mechanically derived `row_kernel` channel: `(nll, ∇, H)`.
pub fn derived_row_kernel<const K: usize, P: RowNllProgram<K> + ?Sized>(
    prog: &P,
    row: usize,
) -> Result<(f64, [f64; K], [[f64; K]; K]), String> {
    let t = evaluate_program(prog, row)?;
    Ok((t.v, t.g, t.h))
}

/// Mechanically derived `row_third_contracted` channel.
pub fn derived_third_contracted<const K: usize, P: RowNllProgram<K> + ?Sized>(
    prog: &P,
    row: usize,
    dir: &[f64; K],
) -> Result<[[f64; K]; K], String> {
    Ok(evaluate_program(prog, row)?.third_contracted(dir))
}

/// Mechanically derived `row_fourth_contracted` channel.
pub fn derived_fourth_contracted<const K: usize, P: RowNllProgram<K> + ?Sized>(
    prog: &P,
    row: usize,
    dir_u: &[f64; K],
    dir_v: &[f64; K],
) -> Result<[[f64; K]; K], String> {
    Ok(evaluate_program(prog, row)?.fourth_contracted(dir_u, dir_v))
}

// ── The oracle ───────────────────────────────────────────────────────

/// One row's worth of hand-written kernel outputs, as claimed by a
/// `RowKernel` implementation, packaged for verification against the
/// tower truth. Plain data (no trait coupling) so any kernel — whatever
/// its visibility — can be audited from its own test module.
pub struct KernelChannels<const K: usize> {
    /// Claimed `(nll, ∇, H)` from `row_kernel`.
    pub value: f64,
    /// Claimed gradient.
    pub gradient: [f64; K],
    /// Claimed Hessian.
    pub hessian: [[f64; K]; K],
    /// Claimed `row_third_contracted(dir)` outputs as `(dir, claim)` pairs.
    pub third: Vec<([f64; K], [[f64; K]; K])>,
    /// Claimed `row_fourth_contracted(u, v)` outputs as `(u, v, claim)`.
    pub fourth: Vec<([f64; K], [f64; K], [[f64; K]; K])>,
}

/// Channel-by-channel audit of a hand-written kernel against the
/// single-expression tower truth. Returns `Err` naming the first channel,
/// index, claimed and true values on disagreement — designed as the body
/// of the per-family CI oracle tests (#932 deployment step 2).
///
/// Tolerance is RELATIVE to a per-channel magnitude floor: each comparison
/// uses `|claim − truth| ≤ rel_tol · max(|truth|, floor)` where `floor`
/// is the largest absolute entry of the true channel — so zero entries of
/// structurally sparse towers don't demand absolute equality, while genuine
/// sign flips (#736) and dropped channels are loud.
pub fn verify_kernel_channels<const K: usize>(
    tower: &Tower4<K>,
    claims: &KernelChannels<K>,
    rel_tol: f64,
) -> Result<(), String> {
    let check = |label: &str, claim: f64, truth: f64, floor: f64| -> Result<(), String> {
        let scale = truth.abs().max(floor).max(1e-300);
        if (claim - truth).abs() > rel_tol * scale {
            return Err(format!(
                "row-kernel oracle: {label} disagrees: claimed {claim:+.12e}, tower {truth:+.12e} (rel_tol {rel_tol:.1e}, scale {scale:.3e})"
            ));
        }
        Ok(())
    };

    check("value", claims.value, tower.v, 1.0)?;

    let g_floor = tower.g.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    for a in 0..K {
        check(
            &format!("gradient[{a}]"),
            claims.gradient[a],
            tower.g[a],
            g_floor,
        )?;
    }

    let h_floor = tower
        .h
        .iter()
        .flatten()
        .fold(0.0_f64, |m, x| m.max(x.abs()));
    for a in 0..K {
        for b in 0..K {
            check(
                &format!("hessian[{a}][{b}]"),
                claims.hessian[a][b],
                tower.h[a][b],
                h_floor,
            )?;
        }
    }

    for (t_idx, (dir, claim)) in claims.third.iter().enumerate() {
        let truth = tower.third_contracted(dir);
        let floor = truth.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()));
        for a in 0..K {
            for b in 0..K {
                check(
                    &format!("third[{t_idx}][{a}][{b}]"),
                    claim[a][b],
                    truth[a][b],
                    floor,
                )?;
            }
        }
    }

    for (f_idx, (u, w, claim)) in claims.fourth.iter().enumerate() {
        let truth = tower.fourth_contracted(u, w);
        let floor = truth.iter().flatten().fold(0.0_f64, |m, x| m.max(x.abs()));
        for a in 0..K {
            for b in 0..K {
                check(
                    &format!("fourth[{f_idx}][{a}][{b}]"),
                    claim[a][b],
                    truth[a][b],
                    floor,
                )?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Binomial-logit row NLL, K=1: ℓ(η) = ln(1 + e^η) − y·η.
    /// The entire tower has textbook closed forms in μ = σ(η); this test
    /// pins the algebra (exp, ln, scalar mixes, Leibniz/Faà di Bruno) to
    /// analytic truth at near-machine precision.
    struct LogitProgram {
        eta: Vec<f64>,
        y: Vec<f64>,
    }

    impl RowNllProgram<1> for LogitProgram {
        fn n_rows(&self) -> usize {
            self.eta.len()
        }
        fn primaries(&self, row: usize) -> Result<[f64; 1], String> {
            Ok([self.eta[row]])
        }
        fn row_nll(&self, row: usize, p: &[Tower4<1>; 1]) -> Result<Tower4<1>, String> {
            let eta = p[0];
            Ok((eta.exp() + 1.0).ln() - eta * self.y[row])
        }
    }

    #[test]
    fn logit_tower_matches_closed_forms() {
        let prog = LogitProgram {
            eta: vec![-2.3, -0.4, 0.0, 0.9, 3.1],
            y: vec![1.0, 0.0, 1.0, 0.0, 1.0],
        };
        for row in 0..prog.n_rows() {
            let t = evaluate_program(&prog, row).expect("logit program");
            let eta = prog.eta[row];
            let y = prog.y[row];
            let mu = 1.0 / (1.0 + (-eta).exp());
            let w = mu * (1.0 - mu);
            let expect = [
                (t.v, (1.0 + eta.exp()).ln() - y * eta, "value"),
                (t.g[0], mu - y, "grad"),
                (t.h[0][0], w, "hess"),
                (t.t3[0][0][0], w * (1.0 - 2.0 * mu), "third"),
                (
                    t.t4[0][0][0][0],
                    w * (1.0 - 6.0 * mu + 6.0 * mu * mu),
                    "fourth",
                ),
            ];
            for (got, want, label) in expect {
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "row {row} {label}: got {got:+.15e} want {want:+.15e}"
                );
            }
        }
    }

    /// Gaussian location-scale row NLL, K=2 primaries (η, s = log σ):
    /// ℓ = s + ½ e^{−2s} (y − η)². Mixed cross blocks — the #736 fragility
    /// shape — all have one-line closed forms here.
    struct LocScaleProgram {
        eta: Vec<f64>,
        s: Vec<f64>,
        y: Vec<f64>,
    }

    impl RowNllProgram<2> for LocScaleProgram {
        fn n_rows(&self) -> usize {
            self.eta.len()
        }
        fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
            Ok([self.eta[row], self.s[row]])
        }
        fn row_nll(&self, row: usize, p: &[Tower4<2>; 2]) -> Result<Tower4<2>, String> {
            let r = -(p[0] - self.y[row]);
            Ok(p[1] + (p[1] * (-2.0)).exp() * r * r * 0.5)
        }
    }

    #[test]
    fn locscale_tower_matches_closed_forms_including_cross_blocks() {
        let prog = LocScaleProgram {
            eta: vec![0.3, -1.1, 2.0],
            s: vec![-0.5, 0.2, 0.8],
            y: vec![1.0, -2.0, 2.5],
        };
        let tol = 1e-12;
        for row in 0..prog.n_rows() {
            let t = evaluate_program(&prog, row).expect("locscale program");
            let r = prog.y[row] - prog.eta[row];
            let w = (-2.0 * prog.s[row]).exp();
            // (η, s) = indices (0, 1).
            let truth_g = [-w * r, 1.0 - w * r * r];
            let truth_h = [[w, 2.0 * w * r], [2.0 * w * r, 2.0 * w * r * r]];
            // Third tensor: distinct-entry closed forms.
            // ∂ηηη = 0, ∂ηηs = −2w, ∂ηss = −4wr, ∂sss = −4wr².
            let t3_truth = |a: usize, b: usize, c: usize| -> f64 {
                match a + b + c {
                    0 => 0.0,
                    1 => -2.0 * w,
                    2 => -4.0 * w * r,
                    _ => -4.0 * w * r * r,
                }
            };
            // Fourth tensor: ∂ηηηη = 0, ∂ηηηs = 0? No: d/ds(∂ηηη)=0 ✓;
            // ∂ηηss = 4w, ∂ηsss = 8wr, ∂ssss = 8wr².
            let t4_truth = |a: usize, b: usize, c: usize, d: usize| -> f64 {
                match a + b + c + d {
                    0 | 1 => 0.0,
                    2 => 4.0 * w,
                    3 => 8.0 * w * r,
                    _ => 8.0 * w * r * r,
                }
            };
            for a in 0..2 {
                assert!(
                    (t.g[a] - truth_g[a]).abs() <= tol * truth_g[a].abs().max(1.0),
                    "row {row} grad[{a}]"
                );
                for b in 0..2 {
                    assert!(
                        (t.h[a][b] - truth_h[a][b]).abs() <= tol * w.max(1.0) * (1.0 + r.abs()),
                        "row {row} hess[{a}][{b}]: got {} want {}",
                        t.h[a][b],
                        truth_h[a][b]
                    );
                    for c in 0..2 {
                        assert!(
                            (t.t3[a][b][c] - t3_truth(a, b, c)).abs()
                                <= tol * 8.0 * w.max(1.0) * (1.0 + r.abs() + r * r),
                            "row {row} t3[{a}][{b}][{c}]: got {} want {}",
                            t.t3[a][b][c],
                            t3_truth(a, b, c)
                        );
                        for d in 0..2 {
                            assert!(
                                (t.t4[a][b][c][d] - t4_truth(a, b, c, d)).abs()
                                    <= tol * 16.0 * w.max(1.0) * (1.0 + r.abs() + r * r),
                                "row {row} t4[{a}][{b}][{c}][{d}]: got {} want {}",
                                t.t4[a][b][c][d],
                                t4_truth(a, b, c, d)
                            );
                        }
                    }
                }
            }
            // The derived trait-surface helpers agree with direct contraction.
            let dir = [0.7, -1.3];
            let third = derived_third_contracted(&prog, row, &dir).expect("third");
            for a in 0..2 {
                for b in 0..2 {
                    let want = t.t3[a][b][0] * dir[0] + t.t3[a][b][1] * dir[1];
                    assert!((third[a][b] - want).abs() <= 1e-13 * want.abs().max(1.0));
                }
            }
        }
    }

    /// FD cross-check on a deliberately gnarly composition (div, sqrt,
    /// powf, nested exp/ln) in K=3, where no closed form is consulted:
    /// every tower channel is checked against central finite differences
    /// of the channel one order below — value→grad, grad→hess, hess→t3,
    /// t3→t4 — so each order is independently anchored.
    ///
    /// The program carries a per-row primary fixture plus a per-row offset
    /// `tau[row]` that enters the loss as a constant, so `row` genuinely
    /// drives both the seed point and the evaluated expression.
    struct GnarlyProgram {
        primaries: Vec<[f64; 3]>,
        tau: Vec<f64>,
    }

    impl GnarlyProgram {
        fn fixture() -> Self {
            Self {
                primaries: vec![[0.4, -0.7, 1.2], [-0.9, 0.6, 0.3], [1.1, -0.2, -0.8]],
                tau: vec![0.15, -0.35, 0.5],
            }
        }
    }

    impl RowNllProgram<3> for GnarlyProgram {
        fn n_rows(&self) -> usize {
            self.primaries.len()
        }
        fn primaries(&self, row: usize) -> Result<[f64; 3], String> {
            self.primaries
                .get(row)
                .copied()
                .ok_or_else(|| format!("gnarly: row {row} out of range"))
        }
        fn row_nll(&self, row: usize, p: &[Tower4<3>; 3]) -> Result<Tower4<3>, String> {
            let tau = *self
                .tau
                .get(row)
                .ok_or_else(|| format!("gnarly: tau row {row} out of range"))?;
            let a = (p[0] * p[1]).exp();
            let b = (p[2] * p[2] + 1.0).sqrt();
            let c = (a + b + tau).ln();
            let d = (p[1] * 0.5 + 2.0).powf(1.7);
            Ok(c / d + (p[0] - p[2]) * (p[0] - p[2]) * 0.25)
        }
    }

    /// Evaluate the gnarly program's tower at an ARBITRARY seed point for
    /// `row` (used to drive central differences off the fixture grid),
    /// while keeping `row`'s per-row data (`tau`) in the loss.
    fn gnarly_tower_at(prog: &GnarlyProgram, row: usize, p: [f64; 3]) -> Tower4<3> {
        struct At<'a> {
            base: &'a GnarlyProgram,
            row: usize,
            p: [f64; 3],
        }
        impl RowNllProgram<3> for At<'_> {
            fn n_rows(&self) -> usize {
                1
            }
            fn primaries(&self, row: usize) -> Result<[f64; 3], String> {
                if row != 0 {
                    return Err(format!("gnarly-at: row {row} out of range"));
                }
                Ok(self.p)
            }
            fn row_nll(&self, eval_row: usize, vars: &[Tower4<3>; 3]) -> Result<Tower4<3>, String> {
                if eval_row != 0 {
                    return Err(format!("gnarly-at: eval row {eval_row} out of range"));
                }
                self.base.row_nll(self.row, vars)
            }
        }
        evaluate_program(&At { base: prog, row, p }, 0).expect("gnarly tower")
    }

    #[test]
    fn gnarly_tower_is_fd_consistent_order_by_order() {
        let prog = GnarlyProgram::fixture();
        for row in 0..prog.n_rows() {
            let base = prog.primaries(row).expect("primaries");
            let t = gnarly_tower_at(&prog, row, base);
            let h_step = 1e-5;
            let tol = 1e-6;
            for c in 0..3 {
                let mut up = base;
                let mut dn = base;
                up[c] += h_step;
                dn[c] -= h_step;
                let t_up = gnarly_tower_at(&prog, row, up);
                let t_dn = gnarly_tower_at(&prog, row, dn);
                // value → gradient.
                let fd_g = (t_up.v - t_dn.v) / (2.0 * h_step);
                assert!(
                    (t.g[c] - fd_g).abs() <= tol * fd_g.abs().max(1.0),
                    "grad[{c}]: analytic {} fd {}",
                    t.g[c],
                    fd_g
                );
                for a in 0..3 {
                    // gradient → Hessian.
                    let fd_h = (t_up.g[a] - t_dn.g[a]) / (2.0 * h_step);
                    assert!(
                        (t.h[a][c] - fd_h).abs() <= tol * fd_h.abs().max(1.0),
                        "hess[{a}][{c}]: analytic {} fd {}",
                        t.h[a][c],
                        fd_h
                    );
                    for b in 0..3 {
                        // Hessian → third.
                        let fd_t3 = (t_up.h[a][b] - t_dn.h[a][b]) / (2.0 * h_step);
                        assert!(
                            (t.t3[a][b][c] - fd_t3).abs() <= tol * fd_t3.abs().max(1.0),
                            "t3[{a}][{b}][{c}]: analytic {} fd {}",
                            t.t3[a][b][c],
                            fd_t3
                        );
                        for d in 0..3 {
                            // third → fourth.
                            let fd_t4 = (t_up.t3[a][b][d] - t_dn.t3[a][b][d]) / (2.0 * h_step);
                            assert!(
                                (t.t4[a][b][d][c] - fd_t4).abs() <= tol * fd_t4.abs().max(1.0),
                                "t4[{a}][{b}][{d}][{c}]: analytic {} fd {}",
                                t.t4[a][b][d][c],
                                fd_t4
                            );
                        }
                    }
                }
            }
        }
    }

    /// The oracle harness catches a planted #736-style sign flip in a
    /// cross block and reports the channel by name.
    #[test]
    fn oracle_catches_planted_cross_block_sign_flip() {
        let prog = LocScaleProgram {
            eta: vec![0.3],
            s: vec![-0.5],
            y: vec![1.0],
        };
        let t = evaluate_program(&prog, 0).expect("tower");
        let dir = [0.6, -0.2];
        let mut third = t.third_contracted(&dir);
        let honest = KernelChannels {
            value: t.v,
            gradient: t.g,
            hessian: t.h,
            third: vec![(dir, third)],
            fourth: vec![(dir, [1.0, 0.5], t.fourth_contracted(&dir, &[1.0, 0.5]))],
        };
        verify_kernel_channels(&t, &honest, 1e-10).expect("honest kernel must pass");

        // Plant the #736 flip: negate one mixed cross entry.
        third[0][1] = -third[0][1];
        let flipped = KernelChannels {
            value: t.v,
            gradient: t.g,
            hessian: t.h,
            third: vec![(dir, third)],
            fourth: vec![],
        };
        let err = verify_kernel_channels(&t, &flipped, 1e-10)
            .expect_err("planted sign flip must be caught");
        assert!(
            err.contains("third[0][0][1]"),
            "oracle must name the flipped channel, got: {err}"
        );
    }

    /// The third- and fourth-order tensors must be FULLY symmetric under
    /// index permutation (mixed partials commute). The tower stores them
    /// unsymmetrized, so equal-by-construction is a real invariant of the
    /// Leibniz/Faà di Bruno writes — a cheap typo tripwire. Asserted on a
    /// nontrivial K=3 tower with all of div/sqrt/powf/exp/ln exercised, so
    /// every composition path contributes. Lives in a test (not the hot
    /// per-op path) on purpose.
    #[test]
    fn t3_t4_are_fully_index_symmetric() {
        let prog = GnarlyProgram::fixture();
        // 3! = 6 permutations of three indices.
        let perms3: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        // 4! = 24 permutations of four indices.
        let perms4: [[usize; 4]; 24] = [
            [0, 1, 2, 3],
            [0, 1, 3, 2],
            [0, 2, 1, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
            [0, 3, 2, 1],
            [1, 0, 2, 3],
            [1, 0, 3, 2],
            [1, 2, 0, 3],
            [1, 2, 3, 0],
            [1, 3, 0, 2],
            [1, 3, 2, 0],
            [2, 0, 1, 3],
            [2, 0, 3, 1],
            [2, 1, 0, 3],
            [2, 1, 3, 0],
            [2, 3, 0, 1],
            [2, 3, 1, 0],
            [3, 0, 1, 2],
            [3, 0, 2, 1],
            [3, 1, 0, 2],
            [3, 1, 2, 0],
            [3, 2, 0, 1],
            [3, 2, 1, 0],
        ];
        for row in 0..prog.n_rows() {
            let t = evaluate_program(&prog, row).expect("gnarly tower");
            let scale_t3 =
                t.t3.iter()
                    .flatten()
                    .flatten()
                    .fold(0.0_f64, |m, x| m.max(x.abs()))
                    .max(1.0);
            let scale_t4 =
                t.t4.iter()
                    .flatten()
                    .flatten()
                    .flatten()
                    .fold(0.0_f64, |m, x| m.max(x.abs()))
                    .max(1.0);
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        let base = t.t3[i][j][k];
                        let idx = [i, j, k];
                        for p in &perms3 {
                            let permed = t.t3[idx[p[0]]][idx[p[1]]][idx[p[2]]];
                            assert!(
                                (base - permed).abs() <= 1e-12 * scale_t3,
                                "row {row}: t3[{i}][{j}][{k}]={base:+.15e} != \
                                 permuted {permed:+.15e} under {p:?}"
                            );
                        }
                        for l in 0..3 {
                            let base4 = t.t4[i][j][k][l];
                            let idx4 = [i, j, k, l];
                            for p in &perms4 {
                                let permed = t.t4[idx4[p[0]]][idx4[p[1]]][idx4[p[2]]][idx4[p[3]]];
                                assert!(
                                    (base4 - permed).abs() <= 1e-12 * scale_t4,
                                    "row {row}: t4[{i}][{j}][{k}][{l}]={base4:+.15e} != \
                                     permuted {permed:+.15e} under {p:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Stable derivative stack for `log Φ(x)` through fourth order.
///
/// The value and Mills ratio come from the shared probit primitive, so the
/// deep left tail uses the same erfcx path as production log-CDF code.
#[inline]
pub(crate) fn unary_derivatives_normal_logcdf(x: f64) -> [f64; 5] {
    let (log_cdf, lambda) = crate::probability::signed_probit_logcdf_and_mills_ratio(x);
    let lambda2 = lambda * lambda;
    let lambda3 = lambda2 * lambda;
    let x2 = x * x;
    [
        log_cdf,
        lambda,
        -lambda * (x + lambda),
        lambda * (x2 - 1.0 + 3.0 * x * lambda + 2.0 * lambda2),
        -lambda
            * ((x * x2 - 3.0 * x) + (7.0 * x2 - 4.0) * lambda + 12.0 * x * lambda2 + 6.0 * lambda3),
    ]
}

/// Stable derivative stack for `log(1 - exp(-x))`, `x > 0`, through fourth order.
#[inline]
pub(crate) fn unary_derivatives_log1mexp_positive(x: f64) -> [f64; 5] {
    let r = 1.0 / x.exp_m1();
    [
        crate::probability::log1mexp_positive(x),
        r,
        -r * (1.0 + r),
        r * (1.0 + r) * (1.0 + 2.0 * r),
        -r * (1.0 + r) * (1.0 + 6.0 * r + 6.0 * r * r),
    ]
}

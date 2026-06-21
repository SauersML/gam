# Jet-Tower Cutover: Math Derivations (#932)

This document gives the complete, self-contained derivations needed to finish
the #932 *mechanical-tower cutover*: replacing every hand-written family
derivative tower with a single row log-likelihood expression evaluated over a
truncated-Taylor scalar algebra. It is grounded directly in the production code:

- `src/families/jet_tower.rs` — `Tower4<K>` / `Tower2<K>` algebra,
  `substitute_intercept`, `implicit_solve`, `RowNllProgram`,
  `verify_kernel_channels`, and the moving-boundary combinators.
- `src/families/survival/location_scale/row_kernel.rs` — the 9-primary
  hand-derived path (`SurvivalLsRowKernel`, `SLS_ROW_K = 9`) and the two gates
  `row_kernel_joint_hessian_supported()` / `row_kernel_directional_supported()`
  that currently `return false`.
- `src/families/survival/marginal_slope/timepoint_exact/{first_full,directional,bidirectional}.rs`
  — the flex implicit-intercept calibration, the moving-density boundary flux,
  and the hand-written cross blocks.
- `src/terms/sae/row_jet_program.rs` — the SAE reconstruction row
  (`SaeReconstructionRowProgram`, `RowGate`, `AtomRowBasisJet`).

The notation throughout: a *primary* is a coordinate `p_a` (a linear predictor,
a gate logit, a latent coordinate) in which the row expression is written; the
row loss is `ℓ(p)`; index letters `a,b,c,d` range over `0..K`; `∂_a ≡ ∂/∂p_a`.
We write `ℓ_a = ∂_a ℓ`, `ℓ_{ab} = ∂_a∂_b ℓ`, etc. All tensors are fully
symmetric in their lower indices. "Channel" means one of the outputs a
`RowKernel` must produce: value, gradient `ℓ_a`, Hessian `ℓ_{ab}`,
third-contracted `Σ_c ℓ_{abc} d_c`, fourth-contracted `Σ_{cd} ℓ_{abcd} u_c v_d`.

---

## A. Order-specific scalar algebras

### A.0 The generic interface

Every concrete jet scalar carries a *value* and a fixed bundle of derivative
channels, and supports the same five operations. The cutover wants the row loss
written **once** against this interface and then re-instantiated at whatever
order/representation a given consumer needs:

```rust
/// A truncated-Taylor scalar carrying derivatives through order ORDER in K
/// primaries. All concrete towers (Order2, directional, two-seed, full Tower4)
/// implement this with the same algebra; only the carried channel set differs.
pub trait JetScalar<const K: usize>: Copy {
    /// A constant: value c, every derivative zero.
    fn constant(c: f64) -> Self;
    /// The seeded variable p_axis at value `x`: unit first-derivative in slot
    /// `axis`, all higher channels zero.
    fn variable(x: f64, axis: usize) -> Self;
    /// The value channel ℓ(p).
    fn value(&self) -> f64;

    // exact truncated arithmetic (Leibniz)
    fn add(&self, o: &Self) -> Self;
    fn sub(&self, o: &Self) -> Self;
    fn mul(&self, o: &Self) -> Self;
    fn scale(&self, s: f64) -> Self;

    /// Exact multivariate Faà di Bruno composition f∘self, given the outer
    /// derivative stack d = [f(u), f'(u), f''(u), …] at u = self.value().
    /// (Stable special-function stacks are passed here — humans own primitive
    ///  stability, the algebra owns combinatorics; see jet_tower.rs module doc.)
    fn compose_unary(&self, d: &[f64]) -> Self;
}
```

The single source of truth is the program trait already in the codebase
(`jet_tower.rs::RowNllProgram`), restated generically over `JetScalar`:

```rust
pub trait RowNllProgram<const K: usize> {
    fn n_rows(&self) -> usize;
    fn primaries(&self, row: usize) -> Result<[f64; K], String>;
    /// The row NLL on jet scalars. p[a] arrives pre-seeded as variable a at the
    /// current primary value; the body uses ONLY JetScalar ops + per-row data
    /// (response, censoring, offsets) entering as constants.
    fn row_nll<S: JetScalar<K>>(&self, row: usize, p: &[S; K]) -> Result<S, String>;
}
```

Because `row_nll` is generic in `S`, *the very same expression* yields different
channel sets when instantiated at different scalars. The point of this section is
to prove that each instantiation extracts the channel a `RowKernel` consumer
asks for — exactly, not approximately.

The seeding convention (production `evaluate_program`, `jet_tower.rs:1118`):

```rust
let p = prog.primaries(row)?;
let vars: [S; K] = from_fn(|a| S::variable(p[a], a));
prog.row_nll(row, &vars)
```

so primary `a` carries the unit tangent `e_a` and the evaluated tower's
derivative channels are partials with respect to `p_a` at the current point.

### A.1 `Order2<K>` — value / gradient / Hessian

`Order2<K>` carries `(v, g_a, H_{ab})`. This is the production `Tower2<K>`
(`jet_tower.rs:404`). Its arithmetic is the order-≤2 truncation of the Leibniz /
Faà di Bruno rules:

**Product** (`Tower2::mul`, lines 456–470):
$$
(ab).v = a.v\,b.v,\qquad
(ab).g_i = a.v\,b.g_i + a.g_i\,b.v,
$$
$$
(ab).H_{ij} = a.v\,b.H_{ij} + a.g_i\,b.g_j + a.g_j\,b.g_i + a.H_{ij}\,b.v .
$$
These are exactly $\partial_i(ab)$ and $\partial_i\partial_j(ab)$ by the Leibniz
rule, dropping nothing at order ≤ 2.

**Composition** $f\circ s$ (`Tower2::compose_unary`, `d=[f,f',f'']`):
$$
v = f(s.v),\quad
g_i = f'\,s.g_i,\quad
H_{ij} = f''\,s.g_i\,s.g_j + f'\,s.H_{ij}.
$$
This is the multivariate Faà di Bruno formula truncated to order 2.

**Claim.** Seeding `p_a = Order2::variable(p_a^0, a)` and evaluating the row
program gives `v = ℓ(p^0)`, `g_a = ℓ_a(p^0)`, `H_{ab} = ℓ_{ab}(p^0)`.

**Proof.** Each `variable(x,a)` has `g = e_a`, `H = 0`; it is the order-2 jet of
the coordinate function `p ↦ p_a`. The program is a finite composition of
`add/sub/mul/scale/compose_unary` and constants; each operation above is the
*exact* order-≤2 jet of the corresponding real operation (product rule, chain
rule). By induction over the expression tree, the output's order-≤2 channels are
the exact order-≤2 partials of the composed real function `ℓ` at `p^0`. ∎

This is the channel `RowKernel::row_kernel` returns `(v, g, H)`.

> **Bit-identity with the full tower.** The order-2 Leibniz/Faà terms read only
> the order-≤2 channels of their inputs (see `Tower4::mul` line 187: `out.h[i][j]`
> never touches `t3`/`t4`). So `Order2<K>` returns *bit-identical* `(v,g,H)` to
> a full `Tower4<K>`. This is what makes the value-only / Hessian-only inner
> Newton step a free, exact truncation — the rationale of `Tower2` in
> `jet_tower.rs:387`.

### A.2 One-seed directional — contracted third

Carry an `Order2<K>` base plus **one nilpotent** `ε` with `ε² = 0`. A scalar is
$$
s = s^{(0)} + \varepsilon\, s^{(\varepsilon)},
$$
where `s^{(0)}` is an `Order2<K>` (value/grad/Hessian channels) and
`s^{(ε)}` is **another `Order2<K>`** holding the ε-coefficient's own value/grad/
Hessian. Arithmetic is `ε² = 0` truncation of the product:
$$
(ab)^{(0)} = a^{(0)}b^{(0)},\qquad
(ab)^{(\varepsilon)} = a^{(0)}b^{(\varepsilon)} + a^{(\varepsilon)}b^{(0)},
$$
each product an `Order2` product (A.1). Composition pushes `ε` through one extra
derivative of the outer stack: with `u = s^{(0)}.v`,
$$
(f\circ s)^{(0)} = f\circ s^{(0)}\ (\text{Order2}),\qquad
(f\circ s)^{(\varepsilon)} = \big[\,\text{Order2 product of } f'(\cdot)\text{-stack and } s^{(\varepsilon)}\big].
$$

**Seeding.** Set primary `a` to
$$
p_a = p_a^0 + x_a + \varepsilon\,u_a,
$$
i.e. `s^{(0)} = variable(p_a^0, a)` and `s^{(ε)} = constant(u_a)`. Then the
base directions are the unit `e_a` (giving the usual Hessian channel) and the
ε-direction is the fixed vector `u`.

**Claim.** The ε-component of the Hessian channel of the evaluated program is the
contracted third derivative:
$$
\big[(\text{program})^{(\varepsilon)}\big].H_{ab} \;=\; \sum_c \ell_{abc}\, u_c .
$$

**Proof.** Introduce a real bookkeeping parameter `τ` and consider the family
`p_a(τ) = p_a^0 + x_a + τ u_a`. Define `Ψ(τ) = ℓ(p(τ))`. The base-Hessian
channel as a function of `τ` is `H_{ab}(τ) = ∂_a∂_b Ψ` evaluated with the
$x$-seeds carrying $e_a$, i.e. `H_{ab}(τ) = ℓ_{ab}(p(τ))`. The nilpotent `ε`
implements first-order Taylor in `τ` exactly: `s(τ) = s^{(0)} + τ s^{(ε)} + O(τ²)`,
and dropping `τ² (≡ε²)` keeps precisely `d/dτ` at `τ = 0`. Hence
$$
\big[(\text{program})^{(\varepsilon)}\big].H_{ab}
= \left.\frac{d}{d\tau}\right|_{0} \ell_{ab}\big(p(\tau)\big)
= \sum_c \ell_{abc}(p^0)\,\frac{dp_c}{d\tau}
= \sum_c \ell_{abc}\,u_c. \qquad\blacksquare
$$

This is exactly `RowKernel::row_third_contracted(dir = u)` (compare the
production full-tower contraction `Tower4::third_contracted`, line 309). The
one-seed scalar reaches it without materializing the full `t3` tensor.

### A.3 Two-seed — contracted fourth

Carry an `Order2<K>` base plus **two** nilpotents `ε, δ` with
`ε² = δ² = 0` (and `εδ` retained). A scalar is
$$
s = s^{(0)} + \varepsilon\,s^{(\varepsilon)} + \delta\,s^{(\delta)} + \varepsilon\delta\,s^{(\varepsilon\delta)},
$$
four `Order2<K>` parts. Product (truncating `ε²=δ²=0`):
$$
(ab)^{(0)} = a^{(0)}b^{(0)},\quad
(ab)^{(\varepsilon)} = a^{(0)}b^{(\varepsilon)}+a^{(\varepsilon)}b^{(0)},\quad
(ab)^{(\delta)} = a^{(0)}b^{(\delta)}+a^{(\delta)}b^{(0)},
$$
$$
(ab)^{(\varepsilon\delta)} = a^{(0)}b^{(\varepsilon\delta)} + a^{(\varepsilon)}b^{(\delta)} + a^{(\delta)}b^{(\varepsilon)} + a^{(\varepsilon\delta)}b^{(0)}.
$$
Composition: with `u = s^{(0)}.v`, the four parts pick up successively higher
outer derivatives (`f` on the base, `f'` on each single-seed part, and
`f''·s^{(ε)}s^{(δ)} + f'·s^{(εδ)}` on the cross part — the second Faà di Bruno
term).

**Seeding.** Set primary `a` to
$$
p_a = p_a^0 + x_a + \varepsilon\,u_a + \delta\,v_a,
$$
so `s^{(0)} = variable(p_a^0,a)`, `s^{(ε)} = constant(u_a)`,
`s^{(δ)} = constant(v_a)`, `s^{(εδ)} = constant(0)`.

**Claim.**
$$
\big[(\text{program})^{(\varepsilon\delta)}\big].H_{ab} \;=\; \sum_{c,d} \ell_{abcd}\, u_c\, v_d .
$$

**Proof.** Two bookkeeping parameters `σ, ρ`; `p_a(σ,ρ) = p_a^0 + x_a + σ u_a + ρ v_a`.
`ε` implements `∂_σ|_0`, `δ` implements `∂_ρ|_0` (each exact at first order by
nilpotency), and `εδ` keeps the single mixed term `∂_σ∂_ρ|_0` (no `σ²` or `ρ²`
contamination). With the base Hessian channel `H_{ab}(σ,ρ) = ℓ_{ab}(p(σ,ρ))`,
$$
\big[(\text{program})^{(\varepsilon\delta)}\big].H_{ab}
= \partial_\sigma\partial_\rho\big|_{0}\,\ell_{ab}(p(\sigma,\rho))
= \sum_{c,d}\ell_{abcd}\,\frac{\partial p_c}{\partial\sigma}\frac{\partial p_d}{\partial\rho}
= \sum_{c,d}\ell_{abcd}\,u_c v_d. \qquad\blacksquare
$$

This is exactly `RowKernel::row_fourth_contracted(u, v)` (compare
`Tower4::fourth_contracted`, line 326), again without materializing `t4`.

### A.4 Storage / cost accounting

For `K` primaries:

| scalar | doubles carried | formula | K=9 doubles | K=9 bytes |
|---|---|---|---|---|
| dense `Tower4<K>` | `1+K+K²+K³+K⁴` | full tensors | **7381** | **59 048** |
| symmetric-distinct `Tower4` | `1 + C(K,1) + C(K+1,2) + C(K+2,3) + C(K+3,4)` | symmetric packing | **715** | **5 720** |
| `Order2<K>` | `1+K+K²` | value/grad/Hess | 91 | 728 |
| one-seed directional | `2(1+K+K²)` | two `Order2` | 182 | 1 456 |
| two-seed fourth | `4(1+K+K²)` | four `Order2` | **364** | **≈ 2.8 KiB** |

Checks at `K=9`: dense `1+9+81+729+6561 = 7381`; symmetric
`1 + 9 + C(10,2)=45 + C(11,3)=165 + C(12,4)=495 = 715`; two-seed
`4·(1+9+81) = 4·91 = 364` doubles `= 2912 B ≈ 2.84 KiB`.

**Why this dissolves the `Tower4<9>` objection.** The location-scale gates
`row_kernel_directional_supported()` and `row_kernel_joint_hessian_supported()`
return `false` (`row_kernel.rs:770, 782`) explicitly because the dense
`Tower4<9>` "materializes a complete fourth-order tensor over the nine survival
location-scale primary channels … every row program builds multiple ~50 KiB
tower values and operator chains copy them by value, which has caused stack
overflows and severe timeouts." The 59 KB figure is precisely the
`1+9+81+729+6561` count above.

The contracted scalars eliminate that object entirely:

- The directional consumer needs only `row_third_contracted(dir)`, which is the
  **one-seed** scalar at 1.46 KiB — a 40× reduction, and it never forms `t3` or
  `t4` (contraction happens *during* differentiation, not after).
- The joint-Hessian/fourth consumer needs only `row_fourth_contracted(u,v)`,
  which is the **two-seed** scalar at 2.8 KiB — a 20× reduction.
- Each consumer re-seeds per `(dir)` or `(u,v)` tuple, which is exactly the
  contraction loop the consumer was going to run anyway; the cost moves from
  "build a 59 KB tensor then contract" to "contract while differentiating." For
  the handful of directions a Newton/logdet step needs, total work is far below
  the dense tower. (When a consumer genuinely needs *all* contractions of one
  row, the symmetric-packed `Tower4` at 5.7 KiB is the right object; the gate's
  "until the generic tower grows a packed/heap-backed tensor layout" caveat is
  satisfied by either route.)

So the cutover does **not** require shipping `Tower4<9>` per row. It requires the
`Order2`/one-seed/two-seed family, each of which is small, stack-friendly, and
exact for the channel it serves.

---

## B. Implicit-function recursion to 4th order

The flex marginal-slope and any calibrated family thread an **implicit**
intercept `a(θ)` defined by a constraint
$$
F\big(a(\theta),\,\theta\big) \equiv 0,
$$
where `θ ∈ ℝ^K` are the primaries. We need the full derivative tower of `a`:
`a_i, a_{ij}, a_{ijk}, a_{ijkl}`. This is `jet_tower.rs::implicit_solve`
(line 793), and it must replace the hand recursion in `first_full.rs` /
`directional.rs` (e.g. the pre-pass `a_dir = -f_dir/f_a` at `directional.rs:56`,
and `a_u = -f_u/f_a`, `a_{uv} = -(f_{uv}+f_{au}a_v+f_{av}a_u+f_{aa}a_u a_v)/f_a`).

Write `F_a = ∂F/∂a` (slot 0), `F_i = ∂F/∂θ_i`, `F_{ai}`, etc. Differentiate the
identity `F(a(θ),θ) = 0` using the chain rule, with `a_I` the mixed θ-partial of
`a` for multi-index `I`. Throughout, `F_a ≠ 0` (the constraint is strictly
monotone in the intercept — guarded at `implicit_solve` line 799).

### Order 1

$$
\partial_i\,F = F_a\,a_i + F_i = 0
\;\Longrightarrow\;
\boxed{\,a_i = -\,\frac{F_i}{F_a}\,}.
$$

### Order 2

Differentiate `F_a a_i + F_i = 0` by `θ_j`. The total θ-derivative of `F_a` is
`(F_{aa} a_j + F_{aj})`, of `F_i` is `(F_{ai} a_j + F_{ij})`:
$$
(F_{aa}a_j + F_{aj})\,a_i + F_a\,a_{ij} + F_{ai}a_j + F_{ij} = 0,
$$
$$
\boxed{\,a_{ij} = -\frac{1}{F_a}\Big(F_{ij} + F_{ai}a_j + F_{aj}a_i + F_{aa}a_i a_j\Big)\,}.
$$
This is exactly the hand formula in the `directional.rs` doc comment — but here
it falls out of the recursion, not a memorized string.

### Order 3

Differentiate the order-2 identity
`F_a a_{ij} + (F_{aa}a_j + F_{aj})a_i + F_{ai}a_j + F_{ij} = 0` by `θ_k`. Using
total derivatives `D_k F_{aa} = F_{aaa}a_k + F_{aak}`, `D_k F_{aj} = F_{aaj}a_k + F_{ajk}`,
`D_k F_{ai} = F_{aai}a_k + F_{aik}`, `D_k F_{ij} = F_{aij}a_k + F_{ijk}`, and
`D_k F_a = F_{aa}a_k + F_{ak}`:
$$
\begin{aligned}
F_a\,a_{ijk} = -\Big[&\,F_{ijk}
+ \big(F_{aij}a_k + F_{aik}a_j + F_{ajk}a_i\big)
+ F_{aa}\big(a_i a_{jk} + a_j a_{ik} + a_k a_{ij}\big)\\
&+ \big(F_{ai}a_{jk} + F_{aj}a_{ik} + F_{ak}a_{ij}\big)
+ \big(F_{aai}a_j a_k + F_{aaj}a_i a_k + F_{aak}a_i a_j\big)\\
&+ F_{aaa}\,a_i a_j a_k\,\Big],
\end{aligned}
$$
$$
\boxed{\,a_{ijk} = -\frac{1}{F_a}\big[\cdots\big]\,}\quad(\text{bracket above, symmetric in } i,j,k).
$$

### Order 4 — via the exact substitution recursion

Writing the order-4 closed form by hand is exactly the error-prone activity #932
removes (the `(g,w0)` third was 3× short for this reason). The production method
(`implicit_solve` lines 830–871) computes **every** order by one mechanism, and
it is provably exact per order. Define the composed residual
$$
G(\theta) \;=\; F\big(A(\theta),\,\theta\big),\qquad A(\theta)\ \text{the running estimate of}\ a(\theta),
$$
assembled by `substitute_intercept` (line 913), which is the exact 4th-order
multivariate Taylor evaluation of `F` with slot 0 fed the increment `δa = A − A.v`
and slots `1..K` fed the unit-seeded primaries.

**Key structural fact.** The order-`m` θ-tensor of `G` depends on the order-`m`
tensor of `A` **only** through the single linear term `F_a · A_{(m)}`, with unit
chain coefficient (slot 0 enters `substitute_intercept`'s first-order part as a
plain seeded variable). Every other contribution to `G_{(m)}` involves only
**lower-order** tensors of `A` (already fixed) and partials of `F`.

**Recursion (exact per order).** Start `A := constant(a_0)` (correct at order 0
because `F(a_0,θ_0) = 0` — the genuine-root precondition). For `m = 1,2,3,4`:

1. Form `G = substitute_intercept(F, A)` (exact through order 4).
2. Set `A_{(m)} \mathrel{-}= G_{(m)} / F_a`.

After step `m`, `G_{(m)} = 0` exactly: before the update `G_{(m)} = (\text{lower-order terms}) + F_a A_{(m)}^{\text{old}}`, and subtracting `G_{(m)}/F_a` from `A_{(m)}` cancels the residual while not disturbing any lower order (those depend only on already-fixed lower tensors). Induction gives `G_{(1..4)} = 0`, i.e. `A` matches the true root curve `a(θ)` through 4th order. This is precisely the loop body at `implicit_solve:831–871`.

This recursion **is** the order-4 formula: expanding the four substitution passes
symbolically reproduces the boxed order-1/2/3 results above and their order-4
continuation, with none of the 4th-order partition terms hand-omitted.

### Preconditions (must hold; all guarded in `implicit_solve`)

1. **Genuine root.** `a_0` solves `F(a_0,θ_0)=0`. Operationally: the single
   Newton correction `|F|/|F_a| ≤ root_tol·(1+|a_0|)` (line 817, `root_tol=1e-9`).
   The recursion cancels orders 1..4 but **never touches order 0**, so a non-root
   `a_0` would produce the Taylor expansion of the *level set* `F = F(a_0)`, not
   the root curve `F = 0` (line 790–792).
2. **Well-conditioned `F_a`.** `F_a` finite and non-zero (line 799); the
   production strict-monotonicity guard guarantees this.
3. **Expansion-point match.** The constraint tower `F` must be expanded about the
   *same* `a` value that the primaries/intercept are seeded at: `f.v` is the
   residual at `(a_0, θ_0)` and `a_0` is that solved intercept. A mismatch
   silently expands the wrong curve.

### Composed-residual post-check (mandatory)

After building `A`, recompute `G = substitute_intercept(F, A)` and assert every
channel (value, grad, Hessian, `t3`, `t4`) is below a scale-aware floor
`resid_tol = 1e-7·(1+|F_a|)` (lines 878–898). By construction orders 1..4
vanish and `G.v = F(a_0,θ_0) ≈ 0`; the check makes any arithmetic regression in
the substitution recursion **loud** instead of silently shipping a level-set
expansion.

### Contrast: the BMS-flex pinned-Newton injects O(r) derivative error

The BMS-flex hand path takes a **single pinned-Newton step from a frozen reset**
each iterate (constant coefficient reset, not a converged implicit solve). Model
it as: start from `a_i = 0` (the reset has no encoded sensitivity), take one
Newton step on the order-1 stationarity. The order-1 update solves a *linearized*
constraint about the (non-root) reset point. Let `r = F(a_0, θ_0)` be the
**root residual** at the reset. Expanding the one pinned step:
$$
a_i^{\text{new}} \;=\; -\frac{F_i}{F_a} \;+\; r\,\frac{F_{ai}}{F_a^{2}} \;+\; O(r^2).
$$
The first term is the correct `a_i`; the second is a **spurious O(r)
contamination** of the *derivative* — even though `r` is "small," it multiplies
the curvature `F_{ai}` and is divided by `F_a²`, so a tolerated constraint
residual leaks linearly into every downstream derivative channel (and worse at
higher orders, where the contamination compounds across the un-recursed terms —
the observed 3× shortfall on the contracted third). `implicit_solve` removes this
by (a) requiring a genuine root so `r ≤ 1e-9·(1+|a_0|)`, and (b) building all
orders from the *converged* point rather than one step from a reset.

---

## C. Crossing-edge motion

A cell boundary sits at the standardized residual
$$
z_c(\theta) \;=\; \frac{\tau(\theta) - a(\theta)}{b(\theta)},
$$
where `τ` is the (possibly θ-dependent) threshold, `a` the implicit intercept
(Section B), `b` the scale. The edge **moves** with θ, so every θ-derivative of a
cell integral carries `z`-motion terms. The production hand path computes the
edge velocity term-by-term (`first_full.rs::moving_density_boundary_flux`,
edge velocity `-(a_u[axis] + direct_g)/b` at lines 22–30 — note this is the
`b`-linear, `τ`-via-`a` special case).

### First derivative

$$
z_i = \frac{\tau_i - a_i}{b} - \frac{(\tau - a)\,b_i}{b^{2}}
    = \frac{\tau_i - a_i - z_c\,b_i}{b}.
$$
With `b` constant (`b_i = 0`) and `τ` entering only through `a`, this collapses to
`z_i = -a_i/b`, matching `-(a_u[axis])/b`; the `direct_g` term is the explicit
`τ_i = z·(∂…)` piece for the `g`-axis. (The production formula is the
specialization, the formula above is the general case.)

### Second derivative (general `b`)

Differentiate `b z_i = τ_i - a_i - z b_i` by `θ_j`:
$$
b_j z_i + b\,z_{ij} = \tau_{ij} - a_{ij} - z_j b_i - z\, b_{ij},
$$
$$
\boxed{\,z_{ij} = \frac{\tau_{ij} - a_{ij} - z_i b_j - z_j b_i - z\,b_{ij}}{b}\,}.
$$
With `b` linear in θ (so `b_{ij} = 0`) this is
`z_{ij} = (τ_{ij} - a_{ij} - z_i b_j - z_j b_i)/b`.

### General multi-index recursion

For a multi-index `I`, differentiate `b z = τ - a` `|I|` times by the Leibniz
rule on the product `b·z`:
$$
\sum_{J \subseteq I} \binom{I}{J} b_J\, z_{I\setminus J} = \tau_I - a_I,
$$
isolating the `J = ∅` term `b·z_I`:
$$
\boxed{\,z_I = \frac{\tau_I - a_I - \displaystyle\sum_{\varnothing \ne J \subseteq I} \binom{I}{J} b_J\, z_{I\setminus J}}{b}\,}.
$$

### Mechanization — no hand flux formula needed

The crucial observation: **the recursion above is exactly what a tower division
computes.** If `τ`, `a`, `b` are each `Tower4<K>` (with `a` produced by
`implicit_solve`, Section B), then
```rust
let z = (tau - a) / b;   // Tower4 Sub + Div (= Mul by recip), jet_tower.rs:719
```
evaluates the value channel `(τ.v − a.v)/b.v` **and** automatically carries every
`z_i, z_{ij}, z_{ijk}, z_{ijkl}` through the Leibniz product/reciprocal algebra.
`Tower4::recip` (line 261) supplies the exact derivative stack of `1/b`, and
`mul` (line 175) applies the subset-Leibniz walker; their composition reproduces
the boxed `z_I` recursion term-for-term. There is **no separate flux channel** to
derive or forget — the edge tower *is* the answer at every order.

---

## D. Moving-domain integrals

A cell contributes
$$
I(\theta) \;=\; \int_{L(\theta)}^{R(\theta)} G(z,\theta)\,dz,
$$
with both limits `L,R` moving (Section C) and the integrand `G` itself
θ-dependent (the density weight `w = e^{-q}/2π` and the cell coefficients move
with η). This is the flex moving-density flux
(`first_full.rs::moving_density_boundary_flux`) and the value-less boundary
combinators `moving_limit_boundary_tower*` (`jet_tower.rs:1000, 1058`).

### First derivative (Leibniz)

$$
I_i = \underbrace{G(R,\theta)\,R_i - G(L,\theta)\,L_i}_{\text{boundary motion}}
    + \underbrace{\int_{L}^{R} G_{\theta_i}(z,\theta)\,dz}_{\text{interior}}.
$$

### Second derivative (full Leibniz with boundary motion)

Differentiate `I_i` by `θ_j`. The boundary term `G(R,θ)R_i` differentiates by
chain rule (`R` depends on θ, and `G` depends on θ both directly and through its
first argument evaluated at `R`):
$$
\partial_j\big[G(R,\theta)R_i\big]
= \big(G_z(R)\,R_j + G_{\theta_j}(R)\big)R_i + G(R)\,R_{ij},
$$
and symmetrically at `L`. The interior term contributes its own boundary motion
when differentiated (`∂_j ∫_L^R G_{θ_i} = G_{θ_i}(R)R_j − G_{θ_i}(L)L_j + ∫ G_{θ_iθ_j}`). Collecting:
$$
\begin{aligned}
I_{ij} =\;& G_z(R)\,R_i R_j + G_{\theta_i}(R)R_j + G_{\theta_j}(R)R_i + G(R)\,R_{ij} \\
        &- \big[G_z(L)\,L_i L_j + G_{\theta_i}(L)L_j + G_{\theta_j}(L)L_i + G(L)\,L_{ij}\big] \\
        &+ \int_{L}^{R} G_{\theta_i\theta_j}(z,\theta)\,dz.
\end{aligned}
$$
The boxed comparison: the directional hand path historically dropped the
`G(R)R_{ij}` term (the `G·z_{uv}` self-term) and mis-signed the `G_z R_i R_j`
self-flux — exactly the documented `(g,w0)` shortfall.

### Shared-edge jump collapse

When two adjacent cells share an edge `z_c` (the right edge of one equals the
left edge of the next), the boundary contributions partly cancel. The net
contribution of the shared edge to `I_i` is the **jump**:
$$
z_{c,i}\,\big(G_{\text{left}}(z_c,\theta) - G_{\text{right}}(z_c,\theta)\big),
$$
i.e. the edge velocity times the integrand discontinuity across the partition.
For a continuous integrand the jump is 0 and only the cell endpoints survive;
for the flex cubic-cell partition the per-cell polynomials differ, so the jump is
the surviving cross-cell flux. (This is why `cell_moving_boundary_flux_tower`
differences the two edge contributions, `jet_tower.rs:1015`.)

### Mechanization — eliminate the flux channels entirely

Two strategies, both already seeded in `jet_tower.rs`, remove every hand-written
flux term.

**(1) Antiderivative substitution.** Let `Φ` satisfy `Φ_z = G` (z-antiderivative;
`Φ` defined up to a θ-dependent constant). Then
$$
I(\theta) = \Phi(R,\theta) - \Phi(L,\theta),
$$
and **every** θ-derivative of `I` (interior *and* boundary motion) is the tower
of `Φ(R,θ) − Φ(L,θ)` — one composition of a mixed `(z,θ)` jet of `Φ` with the
edge towers `R(θ), L(θ)`. For a θ-only-through-edge integrand,
`moving_limit_boundary_tower(z_edge, [B,B',B'',B'''])` composes `z_edge` with
`[0, B, B', B'', B''']` (the leading `0` discards the meaningless `Φ(z_0)`
constant; the value channel is unused — lines 1000–1005). For a θ-dependent
integrand `G(z;θ)`, `moving_limit_boundary_tower_theta_integrand` takes the full
`(K+1)`-variable jet `phi_jet` of `Φ` (slot 0 = z, slots 1..K = θ) and returns
$$
\Phi(z_{\text{edge}}(\theta);\theta) - \Phi(z_0;\theta)
= \texttt{substitute\_intercept}(\phi\_jet, z\_edge) - \texttt{substitute\_intercept}(\phi\_jet, \text{const}(z_0)),
$$
isolating exactly the edge-motion part — every Leibniz boundary term derived by
the substitution algebra, none hand-omitted (lines 1058–1071). The
`G·R_{ij}`, `G_z R_i R_j`, and mixed `G_{θ_i}(R)R_j` terms all appear
automatically as channels of this single difference.

**(2) Fixed-domain pullback.** Map the moving cell to the unit interval:
$$
z(t,\theta) = L(\theta) + \big(R(\theta) - L(\theta)\big)\,t,\quad t \in [0,1],\quad dz = (R-L)\,dt,
$$
$$
I(\theta) = (R(\theta) - L(\theta))\int_0^1 G\big(z(t,\theta),\theta\big)\,dt.
$$
Now the **limits are constants** (0 and 1), so differentiating under the integral
has *no* boundary terms — boundary motion becomes ordinary multiplication by the
tower `(R − L)` and composition `G(z(t,θ),θ)` evaluated at fixed quadrature nodes
`t`. Each derivative of `I` is `Tower4` arithmetic on `(R−L)` and on the
per-node integrand tower; the moving-boundary calculus has been *traded for*
plain product/chain rule that the algebra already does exactly. This is the
preferred strategy when a numerical quadrature already discretizes the cell.

Both strategies make the moving-boundary flux a *consequence* of tower
arithmetic, not a separately-maintained derivative — closing the genus that
produced the flex shortfalls.

---

## E. Faà di Bruno pullback for location-scale link-wiggle

The location-scale link-wiggle path has the row loss depend on coefficients `β`
through a **nonlinear** map `p(β)` before the row likelihood `ℓ`:
$$
L(\beta) \;=\; \ell\big(p(\beta)\big),
$$
where `p` is the basis-evaluated link warp `q(η_threshold, η_log_sigma)` — the
reason `row_kernel_joint_hessian_supported()` returns `false`
(`row_kernel.rs:766–777`: "its basis rows are evaluated at
`q(eta_threshold, eta_log_sigma)`, so the row design itself changes with beta and
contributes `dJ/dβ` terms outside the current trait contract").

### Second derivative

For directions `v_1, v_2` in β-space, with `Dp[v]` the directional derivative of
the map and `D²p[v_1,v_2]` its second:
$$
\boxed{\,D^2 L[v_1,v_2] = D^2\ell\big[Dp[v_1],\,Dp[v_2]\big] \;+\; D\ell\big[D^2 p[v_1,v_2]\big]\,}.
$$
In index form, with `J_{ai} = ∂p_a/∂β_i` and `H^{(p)}_{aij} = ∂²p_a/∂β_i∂β_j`:
$$
L_{ij} = \sum_{a,b} \ell_{ab} J_{ai} J_{bj} \;+\; \sum_a \ell_a\, H^{(p)}_{aij}.
$$

### Third derivative

$$
\begin{aligned}
D^3 L[v_1,v_2,v_3] =\;& D^3\ell[Dp[v_1],Dp[v_2],Dp[v_3]] \\
&+ D^2\ell[D^2 p[v_1,v_2],\,Dp[v_3]] + D^2\ell[D^2 p[v_1,v_3],\,Dp[v_2]] + D^2\ell[D^2 p[v_2,v_3],\,Dp[v_1]] \\
&+ D\ell[D^3 p[v_1,v_2,v_3]].
\end{aligned}
$$

### General set-partition (Faà di Bruno) formula

$$
\boxed{\,D^n(\ell\circ p)[v_1,\dots,v_n]
= \sum_{\pi \in \Pi_n} D^{|\pi|}\ell\Big[\,D^{|B|}p\big[v_i : i\in B\big] : B \in \pi\,\Big]\,},
$$
where `Π_n` is the set of partitions of `{1,…,n}`, `|π|` the number of blocks,
and each block `B` contributes the `|B|`-th derivative of `p` along the `v_i`
indexed by `B`, fed as one argument slot to the `|π|`-th derivative of `ℓ`.
(Bell(2)=2, Bell(3)=5, Bell(4)=15 terms — matching the partition counts in
`Tower4::compose_unary`, line 224.)

### Why `JᵀHJ` alone is wrong

A naive pullback uses only the **leading** partition (all singletons),
$$
L_{ij} \approx \sum_{a,b}\ell_{ab}J_{ai}J_{bj} = (J^\top H J)_{ij},
$$
which **drops the term `Dℓ[D²p]` = `Σ_a ℓ_a H^{(p)}_{aij}`** — the contribution
of the curvature of the map weighted by the loss gradient. That term is nonzero
exactly because `p` is nonlinear in `β` (it is the `dJ/dβ` the gate comment
flags). Omitting it under-counts curvature whenever `ℓ_a ≠ 0` (i.e. away from a
per-index stationary point) and breaks at every higher order even more severely
(third order drops three `D²ℓ[D²p,Dp]` terms plus `Dℓ[D³p]`).

### Recommendation — seed coefficient-space directions into the row program

Do **not** try to bolt the partition formula onto a fixed-Jacobian
`RowKernel`. Instead make the pullback *intrinsic*: write the row program so its
primaries are the **coefficient-space directions**, evaluating `p(β)` itself with
tower scalars inside `row_nll`. Concretely, for a fourth-contracted consumer,
seed (Section A.3) the two-seed scalar with the **β-direction tangents**:
the base seeds carry the Hessian directions, `ε`/`δ` carry `u,v` in β-space, and
`row_nll` computes `q(η(β))` and then `ℓ(q)` entirely over the jet scalar. Then
the evaluated channels are derivatives of `L = ℓ∘p` directly — the full Faà di
Bruno partition sum is produced by the algebra (every `D^{|B|}p` block is a
channel of the tower for `p`, every `D^{|π|}ℓ` is a channel of the tower for
`ℓ`, and `Tower4::mul`/`compose_unary` assemble the partitions). No `J`, `H^{(p)}`,
or `dJ/dβ` is ever materialized by hand; the gate `row_kernel_joint_hessian_supported()`
can flip to `true` once the row program evaluates the warp `q(η)` on tower
scalars rather than reading a precomputed fixed Jacobian.

---

## Cutover checklist — derivation → production site

| Derivation | Production site it replaces / unblocks | Action |
|---|---|---|
| **A.1 `Order2<K>`** | `Tower2<K>` (`jet_tower.rs:404`); `RowKernel::row_kernel` `(v,g,H)` | Already live; route inner Newton steps through it (bit-identical `(v,g,H)`). |
| **A.2 one-seed directional** | `row_kernel_directional_supported()` = `false` (`row_kernel.rs:782`) | Add a one-seed scalar; flip the gate to `true` — 1.46 KiB replaces the 59 KB `Tower4<9>` the gate rejects. |
| **A.3 two-seed fourth** | `row_kernel_joint_hessian_supported()` = `false` (`row_kernel.rs:770`); `row_fourth_contracted` | Add a two-seed scalar (2.8 KiB); the dense-`Tower4<9>` cost objection no longer applies. |
| **A.4 storage accounting** | the explicit "~50 KiB tower / stack overflow / timeout" justification in both gate comments | Numbers above quantify the 20–40× reduction; cite when flipping the gates. |
| **B implicit recursion + post-check** | `implicit_solve` (`jet_tower.rs:793`); flex `a_u/a_{uv}` hand recursion (`first_full.rs`, `directional.rs:56–83`) | Replace the hand `a_*` formulas with one `implicit_solve` call; keep the composed-residual self-check armed. |
| **B pinned-Newton contrast** | BMS-flex pinned-Newton reset closed-forms (`families/bms/*`) | Replace the per-iterate frozen-reset step with a genuine-root `implicit_solve`; eliminates the O(r) derivative contamination. |
| **C crossing-edge `z = (τ−a)/b`** | `moving_density_boundary_flux` edge-velocity (`first_full.rs:22–47`) | Build `z_edge` as `(tau − a)/b` over `Tower4`; the per-order `z_I` recursion is automatic, no hand flux. |
| **D moving-domain integral (antiderivative / fixed-domain)** | `moving_limit_boundary_tower*` (`jet_tower.rs:1000,1058`); flex `directional.rs`/`bidirectional.rs` boundary cross blocks | Express each cell integral via `Φ(R;θ)−Φ(L;θ)` (strategy 1) or the unit-interval pullback (strategy 2); deletes the hand-summed `G·z_{uv}`, `G_z z_u z_v`, mixed terms. |
| **D shared-edge jump** | flex inter-cell flux (`cell_moving_boundary_flux_tower`, `jet_tower.rs:1015`) | Difference the two edge towers; the jump term is automatic. |
| **E Faà di Bruno pullback** | link-wiggle `dJ/dβ` gap (`row_kernel.rs:766–777`) | Seed β-space directions into `row_nll`, evaluate `q(η(β))` on tower scalars; flip `row_kernel_joint_hessian_supported()` to `true`. |
| **Unifying: flex one program** | `first_full.rs` + `directional.rs` + `bidirectional.rs` (three hand kernels) | Collapse to ONE `RowNllProgram`: calibration via B, edges via C, cell integrals via D, link warp via E. |
| **SAE border channels** | `SaeReconstructionRowProgram` / `row_jets_for_logdet` hand-packed first/second (`row_jet_program.rs`) | Fold the gate `ζ(ℓ)` + basis `Φ(t)` + decoder `B` into the row expression over `Tower4`; the #1006 logdet adjoint `Γ_a = tr(H⁻¹ ∂H/∂θ_a)` consumes derived channels — no hand cross blocks. |

The universal safety net is `verify_kernel_channels` (`jet_tower.rs:1189`): every
hand kernel kept for speed gains a CI oracle asserting channel-by-channel
agreement (mixed abs/rel tol, strict non-finite handling) against the
single-expression tower truth — the test that would have caught #736 at
introduction.

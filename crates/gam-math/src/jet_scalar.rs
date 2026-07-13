//! Order-specific Taylor-jet SCALAR algebras (#932 cutover, doc §A).
//!
//! [`crate::jet_tower::Tower4`] carries the full value/gradient/Hessian/`t3`/`t4`
//! tensor stack: it answers EVERY channel a [`super::row_kernel::RowKernel`]
//! consumer can ask for, but at `K = 9` that is a ~50 KiB per-row object whose
//! by-value copies overflowed the stack and timed out the location-scale fit.
//! The cutover therefore does NOT instantiate the dense `Tower4<9>` per row; it
//! carries, per consumer, only the one channel that consumer serves:
//!
//! | consumer | channel | scalar here | K=9 size |
//! |---|---|---|---|
//! | inner Newton / `row_kernel` | `(v, g, H)` | [`Order2`] | 728 B |
//! | `row_third_contracted(dir)` | `Σ_c ℓ_{abc} dir_c` | [`OneSeed`] | 1.46 KiB |
//! | `row_fourth_contracted(u, v)` | `Σ_{cd} ℓ_{abcd} u_c v_d` | [`TwoSeed`] | 2.8 KiB |
//!
//! Each is built on [`Order2`] (value/grad/Hessian), which is the production
//! [`crate::jet_tower::Tower2`] re-expressed behind a generic interface: a row
//! loss written ONCE against [`JetScalar`] re-instantiates at whatever order /
//! representation a consumer needs, with the contraction folded INTO the
//! differentiation (the nilpotent ε / δ directions), so `t3` / `t4` are never
//! materialised. The single source of truth is the same one expression — the
//! genus of #736 cross-block drift cannot reappear because there is no separate
//! channel to forget.
//!
//! # Why each scalar is exact (doc §A.1–A.3)
//!
//! * [`Order2`] is the order-≤2 truncation of the Leibniz / Faà di Bruno rules.
//!   Those order-2 terms read ONLY the order-≤2 channels of their inputs (see
//!   [`crate::jet_tower::Tower4::mul`]: `out.h[i][j]` never touches `t3`/`t4`),
//!   so its `(v, g, H)` is BIT-IDENTICAL to a full `Tower4<K>` — and identical
//!   to [`crate::jet_tower::Tower2`], over which it is a thin newtype.
//! * [`OneSeed`] carries an [`Order2`] base plus one nilpotent ε (`ε² = 0`)
//!   holding another [`Order2`]. Seeding ε with the fixed direction `u` makes the
//!   ε-component of the Hessian channel the contracted third `Σ_c ℓ_{abc} u_c`
//!   (the nilpotent implements `d/dτ|₀` of `ℓ_{ab}(p + τu)` exactly).
//! * [`TwoSeed`] carries an [`Order2`] base plus ε, δ (`ε² = δ² = 0`, `εδ`
//!   retained) — four [`Order2`] parts. Seeding ε, δ with `u, v` makes the
//!   εδ-component of the Hessian channel the contracted fourth
//!   `Σ_{cd} ℓ_{abcd} u_c v_d` (the single mixed `∂_σ∂_ρ|₀` term, no `σ²`/`ρ²`
//!   contamination).
//!
//! # Stability discipline
//!
//! As in [`crate::jet_tower`], humans own primitive stability and the algebra
//! owns combinatorics: tail-critical special functions enter ONLY as
//! hand-certified `[f64; 5]` derivative stacks through [`crate::nested_dual::JetField::compose_unary`]
//! (each scalar consumes the leading entries its order needs), never by
//! differentiating an unstable primal.
//!
//! # Production scalars and the test-only all-channels oracle
//!
//! The `JetScalar` trait below is production: it is the bound on the canonical
//! [`crate::jet_tower::RowProgram::eval`] seam a family row loss is written
//! against. The order-specific scalars that *consume* it —
//! [`Order2`] (value/grad/Hessian), [`OneSeed`] (contracted third) and
//! [`TwoSeed`] (contracted fourth) — are production: the survival location-scale
//! `RowKernel<9>` builds its joint Hessian / directional derivatives through them
//! (`survival::location_scale::row_kernel`), paying only the small packed scalar
//! per row instead of the ~50 KiB dense [`crate::jet_tower::Tower4`].
//!
//! The [`crate::jet_tower::Tower4`] all-channels `JetScalar` impl is test-only: it
//! is the oracle that pins the contracted scalars against the dense
//! value/grad/Hessian/`t3`/`t4` truth, so it lives in the `#[cfg(test)]` module.

/// Symmetric coefficient operator used by the universal quadratic-form jet
/// primitive.
///
/// Implementations own representation-specific matrix access (diagonal,
/// dense, low rank, structured, or matrix free). Jet scalars own the one
/// derivative rule for `x' A x`; callers never spell its chain rule.
pub trait SymmetricQuadraticCoefficients {
    /// Input-space dimension of the symmetric operator.
    fn dimension(&self) -> usize;

    /// Compute `output = A * input` without materializing `A`.
    fn multiply(&self, input: &[f64], output: &mut [f64]);

    /// One symmetric coefficient `A[row, column]`.
    fn coefficient(&self, row: usize, column: usize) -> f64;

    /// Evaluate the primal quadratic form without materializing the input as a
    /// separate `f64` vector. Structured operators should override this to
    /// preserve their representation's natural complexity (for example O(KR)
    /// for a K-by-R low-rank factor).
    fn quadratic_value<T, F>(&self, inputs: &[T], value: F) -> f64
    where
        F: Fn(&T) -> f64,
    {
        assert_eq!(
            inputs.len(),
            self.dimension(),
            "symmetric quadratic-form dimension mismatch"
        );
        let mut out = 0.0;
        for row in 0..inputs.len() {
            let row_value = value(&inputs[row]);
            out += self.coefficient(row, row) * row_value * row_value;
            for column in row + 1..inputs.len() {
                out += 2.0 * self.coefficient(row, column) * row_value * value(&inputs[column]);
            }
        }
        out
    }
}

fn symmetric_quadratic_form_default<T, C>(
    inputs: &[T],
    coefficients: &C,
    constant: impl Fn(f64) -> T,
    add: impl Fn(&T, &T) -> T,
    mul: impl Fn(&T, &T) -> T,
    scale: impl Fn(&T, f64) -> T,
) -> T
where
    C: SymmetricQuadraticCoefficients,
{
    assert_eq!(
        inputs.len(),
        coefficients.dimension(),
        "symmetric quadratic-form dimension mismatch"
    );
    let mut out = constant(0.0);
    for row in 0..inputs.len() {
        let diagonal = mul(&inputs[row], &inputs[row]);
        out = add(&out, &scale(&diagonal, coefficients.coefficient(row, row)));
        for column in row + 1..inputs.len() {
            let cross = mul(&inputs[row], &inputs[column]);
            out = add(
                &out,
                &scale(&cross, 2.0 * coefficients.coefficient(row, column)),
            );
        }
    }
    out
}

fn linear_combination_default<T>(
    inputs: &[T],
    weights: &[f64],
    constant: impl Fn(f64) -> T,
    add: impl Fn(&T, &T) -> T,
    scale: impl Fn(&T, f64) -> T,
) -> T {
    assert_eq!(
        inputs.len(),
        weights.len(),
        "linear-combination dimension mismatch"
    );
    inputs
        .iter()
        .zip(weights)
        .fold(constant(0.0), |sum, (input, &weight)| {
            add(&sum, &scale(input, weight))
        })
}

fn multiply_add_default<T>(
    left: &T,
    right: &T,
    addend: &T,
    mul: impl Fn(&T, &T) -> T,
    add: impl Fn(&T, &T) -> T,
) -> T {
    add(&mul(left, right), addend)
}

fn composed_sum_default<T>(
    inputs: &[T],
    derivative_stacks: &[[f64; 5]],
    constant: impl Fn(f64) -> T,
    add: impl Fn(&T, &T) -> T,
    compose: impl Fn(&T, [f64; 5]) -> T,
) -> T {
    assert_eq!(
        inputs.len(),
        derivative_stacks.len(),
        "composed-sum term-count mismatch"
    );
    inputs
        .iter()
        .zip(derivative_stacks)
        .fold(constant(0.0), |sum, (input, &stack)| {
            add(&sum, &compose(input, stack))
        })
}

fn affine_compose_default<T>(
    input: &T,
    input_scale: f64,
    input_shift: f64,
    derivative_stack: [f64; 5],
    scale: impl Fn(&T, f64) -> T,
    add_constant: impl Fn(&T, f64) -> T,
    compose: impl Fn(&T, [f64; 5]) -> T,
) -> T {
    compose(
        &add_constant(&scale(input, input_scale), input_shift),
        derivative_stack,
    )
}

fn affine_composed_sum_default<T>(
    inputs: &[T],
    input_scales: &[f64],
    derivative_stacks: &[[f64; 5]],
    constant: impl Fn(f64) -> T,
    add: impl Fn(&T, &T) -> T,
    scale: impl Fn(&T, f64) -> T,
    add_constant: impl Fn(&T, f64) -> T,
    compose: impl Fn(&T, [f64; 5]) -> T,
) -> T {
    assert_eq!(inputs.len(), input_scales.len());
    assert_eq!(inputs.len(), derivative_stacks.len());
    inputs.iter().zip(input_scales).zip(derivative_stacks).fold(
        constant(0.0),
        |sum, ((input, &input_scale), &stack)| {
            add(
                &sum,
                &affine_compose_default(
                    input,
                    input_scale,
                    0.0,
                    stack,
                    &scale,
                    &add_constant,
                    &compose,
                ),
            )
        },
    )
}

fn shared_multiply_add_affine_composed_sum_default<T, const N: usize>(
    lefts: &[&T; N],
    right: &T,
    addend: &T,
    addend_scales: &[f64; N],
    input_scales: &[f64; N],
    derivative_stacks: &[[f64; 5]; N],
    constant: impl Fn(f64) -> T,
    add: impl Fn(&T, &T) -> T,
    mul: impl Fn(&T, &T) -> T,
    scale: impl Fn(&T, f64) -> T,
    multiply_add: impl Fn(&T, &T, &T) -> T,
    affine_compose: impl Fn(&T, f64, f64, [f64; 5]) -> T,
) -> T {
    (0..N).fold(constant(0.0), |sum, term| {
        let inner = if addend_scales[term] == 0.0 {
            mul(lefts[term], right)
        } else if addend_scales[term] == 1.0 {
            multiply_add(lefts[term], right, addend)
        } else {
            multiply_add(lefts[term], right, &scale(addend, addend_scales[term]))
        };
        let composed = affine_compose(&inner, input_scales[term], 0.0, derivative_stacks[term]);
        add(&sum, &composed)
    })
}

/// Canonical representatives for repeated inner sources in a shared composed sum.
///
/// Backends define source identity at their representation boundary: eager jets
/// use reference identity, while compiled graphs use node identity. Terms are
/// equivalent only when their left operand and structural addend coefficient
/// name the same inner expression. The returned term-to-source map lets every
/// backend sum outer first/second coefficients before propagating that source.
#[inline(always)]
pub(crate) fn canonical_shared_source_schedule<const N: usize>(
    mut equivalent: impl FnMut(usize, usize) -> bool,
) -> ([usize; N], [usize; N], usize) {
    let mut representatives = [0; N];
    let mut term_sources = [0; N];
    let mut source_count = 0;
    for term in 0..N {
        let mut source = 0;
        while source < source_count && !equivalent(term, representatives[source]) {
            source += 1;
        }
        if source == source_count {
            representatives[source] = term;
            source_count += 1;
        }
        term_sources[term] = source;
    }
    (representatives, term_sources, source_count)
}

/// A truncated-Taylor scalar carrying derivatives in `K` primaries.
///
/// All concrete scalars here ([`Order2`], [`OneSeed`], [`TwoSeed`]) and the full
/// [`crate::jet_tower::Tower4`] implement the SAME algebra; only the carried
/// channel set differs. A row loss written once against this interface yields a
/// different channel set per instantiation, all exact for the channel they serve
/// (doc §A.0).
pub trait JetScalar<const K: usize>: crate::nested_dual::JetField + Copy {
    /// A constant: value `c`, every derivative channel zero.
    fn constant(c: f64) -> Self;

    /// The seeded variable `p_axis` at value `x`: unit first derivative in slot
    /// `axis`, all higher channels zero. (The nilpotent / cross channels of the
    /// directional scalars are seeded zero — callers set ε/δ directions through
    /// the scalar-specific [`OneSeed::seed_direction`] / [`TwoSeed::seed`].)
    fn variable(x: f64, axis: usize) -> Self;

    /// Evaluate `inputs' A inputs` from one universal semantic primitive.
    /// Order-specific scalars may lower the mechanically derived channels
    /// directly; the default is the exact scalar program over `mul/add/scale`.
    fn symmetric_quadratic_form<C: SymmetricQuadraticCoefficients>(
        inputs: &[Self],
        coefficients: &C,
    ) -> Self {
        symmetric_quadratic_form_default(
            inputs,
            coefficients,
            Self::constant,
            crate::nested_dual::JetField::add,
            crate::nested_dual::JetField::mul,
            crate::nested_dual::JetField::scale,
        )
    }

    /// Evaluate `sum_i weights[i] * inputs[i]` in one semantic primitive.
    fn linear_combination(inputs: &[Self], weights: &[f64]) -> Self {
        linear_combination_default(
            inputs,
            weights,
            Self::constant,
            crate::nested_dual::JetField::add,
            crate::nested_dual::JetField::scale,
        )
    }

    /// Add a primal constant without changing derivative channels.
    fn add_constant(&self, constant: f64) -> Self {
        self.add(&Self::constant(constant))
    }

    /// Evaluate `self * right + addend` in one semantic primitive.
    fn multiply_add(&self, right: &Self, addend: &Self) -> Self {
        multiply_add_default(
            self,
            right,
            addend,
            crate::nested_dual::JetField::mul,
            crate::nested_dual::JetField::add,
        )
    }

    /// Sum unary compositions directly from certified derivative stacks.
    fn composed_sum(inputs: &[Self], derivative_stacks: &[[f64; 5]]) -> Self {
        composed_sum_default(
            inputs,
            derivative_stacks,
            Self::constant,
            crate::nested_dual::JetField::add,
            crate::nested_dual::JetField::compose_unary,
        )
    }

    /// Exact product as an explicit compiled graph node.
    fn product(&self, right: &Self) -> Self {
        self.mul(right)
    }

    /// Compose a certified outer stack after the affine map
    /// `u = input_scale * self + input_shift`.
    fn affine_compose(
        &self,
        input_scale: f64,
        input_shift: f64,
        derivative_stack: [f64; 5],
    ) -> Self {
        affine_compose_default(
            self,
            input_scale,
            input_shift,
            derivative_stack,
            crate::nested_dual::JetField::scale,
            Self::add_constant,
            crate::nested_dual::JetField::compose_unary,
        )
    }

    /// Sum unary compositions whose inputs each carry an affine scale.
    fn affine_composed_sum(
        inputs: &[Self],
        input_scales: &[f64],
        derivative_stacks: &[[f64; 5]],
    ) -> Self {
        affine_composed_sum_default(
            inputs,
            input_scales,
            derivative_stacks,
            Self::constant,
            crate::nested_dual::JetField::add,
            crate::nested_dual::JetField::scale,
            Self::add_constant,
            crate::nested_dual::JetField::compose_unary,
        )
    }

    /// Evaluate
    /// `Σ_i f_i(input_scale_i · (left_i · right + addend_scale_i · addend))`
    /// from the certified derivative stack of each `f_i`. The shared operands
    /// make expression-level common subexpressions explicit, so optimized
    /// backends apply their inherited derivative channels once. Expression
    /// arity is part of the type, and borrowed operands never copy a full tower.
    /// An exact-zero addend scale (either sign of IEEE zero) removes that addend
    /// from the corresponding term entirely.
    fn shared_multiply_add_affine_composed_sum<const N: usize>(
        lefts: &[&Self; N],
        right: &Self,
        addend: &Self,
        addend_scales: &[f64; N],
        input_scales: &[f64; N],
        derivative_stacks: &[[f64; 5]; N],
    ) -> Self {
        shared_multiply_add_affine_composed_sum_default(
            lefts,
            right,
            addend,
            addend_scales,
            input_scales,
            derivative_stacks,
            Self::constant,
            crate::nested_dual::JetField::add,
            crate::nested_dual::JetField::mul,
            crate::nested_dual::JetField::scale,
            Self::multiply_add,
            Self::affine_compose,
        )
    }

    // The scalar-field algebra — `value`, `add`, `sub`, `mul`, `neg`, `scale`,
    // and the single Faà di Bruno `compose_unary([f64; 5])` — is declared ONCE on
    // the shared [`crate::nested_dual::JetField`] supertrait (which the runtime-`p`
    // flex jets also extend), so it is not re-declared here. The value/composition
    // convention is identical (`u = self.value()`, the `[f64; 5]` stack shape
    // [`crate::jet_tower::Tower4`] consumes). The helpers below build on it.

    /// Compose with a unary special-function whose derivative stack is built from
    /// the scalar base value through `stack_fn`. This evaluates
    /// `stack_fn(self.value())` once and forwards to
    /// [`compose_unary`](Self::compose_unary), so it is bit-identical to the
    /// explicit `self.compose_unary(stack_fn(self.value()))` form.
    fn compose_unary_with(&self, stack_fn: impl Fn(f64) -> [f64; 5]) -> Self {
        self.compose_unary(stack_fn(self.value()))
    }

    /// `e^self`. Convenience for tame arguments (see module stability note).
    fn exp(&self) -> Self {
        let e = self.value().exp();
        self.compose_unary([e, e, e, e, e])
    }

    /// `√self`. Caller guarantees positivity.
    fn sqrt(&self) -> Self {
        let u = self.value();
        let s = u.sqrt();
        self.compose_unary([
            s,
            0.5 / s,
            -0.25 / (u * s),
            0.375 / (u * u * s),
            -0.9375 / (u * u * u * s),
        ])
    }

    /// `ln(self)`. Caller guarantees positivity. Same derivative stack
    /// [`crate::jet_tower::Tower4::ln`] uses, so any program written over both
    /// matches term-for-term.
    fn ln(&self) -> Self {
        let u = self.value();
        let r = 1.0 / u;
        self.compose_unary([u.ln(), r, -r * r, 2.0 * r * r * r, -6.0 * r * r * r * r])
    }

    /// `1/self`.
    fn recip(&self) -> Self {
        let r = 1.0 / self.value();
        let r2 = r * r;
        self.compose_unary([r, -r2, 2.0 * r2 * r, -6.0 * r2 * r2, 24.0 * r2 * r2 * r])
    }

    /// `self^a` for real exponent `a`. Caller guarantees a positive base.
    /// Mirrors [`crate::jet_tower::Tower4::powf`] (falling-factorial stack).
    fn powf(&self, a: f64) -> Self {
        let u = self.value();
        self.compose_unary([
            u.powf(a),
            a * u.powf(a - 1.0),
            a * (a - 1.0) * u.powf(a - 2.0),
            a * (a - 1.0) * (a - 2.0) * u.powf(a - 3.0),
            a * (a - 1.0) * (a - 2.0) * (a - 3.0) * u.powf(a - 4.0),
        ])
    }

    /// `ln Γ(self)`. Caller guarantees a positive argument. Uses the SAME
    /// hand-certified derivative stack [`crate::jet_tower::Tower4::ln_gamma`]
    /// consumes ([`crate::jet_tower::ln_gamma_derivative_stack`]), so any
    /// program written over both matches term-for-term.
    fn ln_gamma(&self) -> Self {
        self.compose_unary(crate::jet_tower::ln_gamma_derivative_stack(self.value()))
    }

    /// `ψ(self) = d/dx ln Γ(x)` (digamma). Caller guarantees a positive
    /// argument. Same hand-certified stack
    /// [`crate::jet_tower::digamma_derivative_stack`].
    fn digamma(&self) -> Self {
        self.compose_unary(crate::jet_tower::digamma_derivative_stack(self.value()))
    }
}

/// A Taylor-jet scalar whose primary dimension is selected at runtime.
///
/// This is the dimensioned counterpart of [`JetScalar`].  Its algebra is the
/// same; only the constructors receive the row's actual primary count.  The
/// fixed-size scalar implementations below bridge to this trait as well, which
/// lets one row program serve both the const-generic derivative oracles and the
/// runtime-sized production backends without duplicating the expression.
pub trait RuntimeJetScalar<'arena>: Clone {
    /// Storage arena used by runtime-backed scalars. Fixed derivative oracles
    /// use the unit type because their storage is inline.
    type Workspace: ?Sized;

    /// A constant in a `dimension`-primary algebra.
    fn constant(c: f64, dimension: usize, workspace: &'arena Self::Workspace) -> Self;
    /// A seeded variable in a `dimension`-primary algebra.
    fn variable(x: f64, axis: usize, dimension: usize, workspace: &'arena Self::Workspace) -> Self;

    /// Evaluate `inputs' A inputs` from the same universal semantic primitive
    /// as [`JetScalar::symmetric_quadratic_form`].
    fn symmetric_quadratic_form<C: SymmetricQuadraticCoefficients>(
        inputs: &[Self],
        coefficients: &C,
        dimension: usize,
        workspace: &'arena Self::Workspace,
    ) -> Self {
        symmetric_quadratic_form_default(
            inputs,
            coefficients,
            |value| Self::constant(value, dimension, workspace),
            Self::add,
            Self::mul,
            Self::scale,
        )
    }

    /// Evaluate `sum_i weights[i] * inputs[i]` in one semantic primitive.
    fn linear_combination(
        inputs: &[Self],
        weights: &[f64],
        dimension: usize,
        workspace: &'arena Self::Workspace,
    ) -> Self {
        linear_combination_default(
            inputs,
            weights,
            |value| Self::constant(value, dimension, workspace),
            Self::add,
            Self::scale,
        )
    }

    /// Add a primal constant without changing derivative channels.
    fn add_constant(&self, constant: f64, workspace: &'arena Self::Workspace) -> Self {
        self.add(&Self::constant(constant, self.dimension(), workspace))
    }

    /// Evaluate `self * right + addend` in one semantic primitive.
    fn multiply_add(&self, right: &Self, addend: &Self) -> Self {
        multiply_add_default(self, right, addend, Self::mul, Self::add)
    }

    /// Sum unary compositions directly from certified derivative stacks.
    fn composed_sum(
        inputs: &[Self],
        derivative_stacks: &[[f64; 5]],
        dimension: usize,
        workspace: &'arena Self::Workspace,
    ) -> Self {
        composed_sum_default(
            inputs,
            derivative_stacks,
            |value| Self::constant(value, dimension, workspace),
            Self::add,
            Self::compose_unary,
        )
    }

    /// Exact product as an explicit compiled graph node.
    fn product(&self, right: &Self) -> Self {
        self.mul(right)
    }

    /// Compose a certified outer stack after an affine input map.
    fn affine_compose(
        &self,
        input_scale: f64,
        input_shift: f64,
        derivative_stack: [f64; 5],
        workspace: &'arena Self::Workspace,
    ) -> Self {
        affine_compose_default(
            self,
            input_scale,
            input_shift,
            derivative_stack,
            Self::scale,
            |value, constant| value.add_constant(constant, workspace),
            Self::compose_unary,
        )
    }

    /// Sum unary compositions whose inputs each carry an affine scale.
    fn affine_composed_sum(
        inputs: &[Self],
        input_scales: &[f64],
        derivative_stacks: &[[f64; 5]],
        dimension: usize,
        workspace: &'arena Self::Workspace,
    ) -> Self {
        affine_composed_sum_default(
            inputs,
            input_scales,
            derivative_stacks,
            |value| Self::constant(value, dimension, workspace),
            Self::add,
            Self::scale,
            |value, constant| value.add_constant(constant, workspace),
            Self::compose_unary,
        )
    }

    /// Runtime-dimension lowering of
    /// `Σ_i f_i(input_scale_i · (left_i · right + addend_scale_i · addend))`
    /// from certified derivative stacks. Const expression arity replaces an
    /// implementation-specific term cap, while shared operands expose universal
    /// common-subexpression elimination. If every addend scale is exact zero
    /// (including `-0.0`), the addend has no dimension, workspace, or
    /// derivative-channel obligations.
    fn shared_multiply_add_affine_composed_sum<const N: usize>(
        lefts: &[&Self; N],
        right: &Self,
        addend: &Self,
        addend_scales: &[f64; N],
        input_scales: &[f64; N],
        derivative_stacks: &[[f64; 5]; N],
        dimension: usize,
        workspace: &'arena Self::Workspace,
    ) -> Self {
        shared_multiply_add_affine_composed_sum_default(
            lefts,
            right,
            addend,
            addend_scales,
            input_scales,
            derivative_stacks,
            |value| Self::constant(value, dimension, workspace),
            Self::add,
            Self::mul,
            Self::scale,
            Self::multiply_add,
            |input, scale, shift, stack| input.affine_compose(scale, shift, stack, workspace),
        )
    }
    /// Number of primary derivative axes carried by this scalar.
    fn dimension(&self) -> usize;
    /// Value channel.
    fn value(&self) -> f64;
    /// Exact truncated sum.
    fn add(&self, o: &Self) -> Self;
    /// Exact truncated difference.
    fn sub(&self, o: &Self) -> Self;
    /// Exact truncated product.
    fn mul(&self, o: &Self) -> Self;
    /// Negate every channel.
    fn neg(&self) -> Self;
    /// Scale every channel.
    fn scale(&self, s: f64) -> Self;
    /// Exact unary composition from the certified derivative stack.
    fn compose_unary(&self, d: [f64; 5]) -> Self;

    /// `e^self`.
    fn exp(&self) -> Self {
        let e = self.value().exp();
        self.compose_unary([e, e, e, e, e])
    }

    /// `1/self`.
    fn recip(&self) -> Self {
        let r = 1.0 / self.value();
        let r2 = r * r;
        self.compose_unary([r, -r2, 2.0 * r2 * r, -6.0 * r2 * r2, 24.0 * r2 * r2 * r])
    }
}

/// Zero-order runtime scalar for evaluating the primal value of a canonical
/// runtime-width row program without constructing derivative channels.
///
/// The primary dimension remains part of the scalar so it obeys the same
/// runtime algebra contract as derivative-carrying implementations. Unary
/// derivative stacks contribute only their value entry, while structured
/// quadratic operators retain their representation-specific primal lowering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuntimeValue {
    value: f64,
    dimension: usize,
}

impl<'arena> RuntimeJetScalar<'arena> for RuntimeValue {
    type Workspace = ();

    #[inline(always)]
    fn constant(c: f64, dimension: usize, &(): &'arena Self::Workspace) -> Self {
        Self {
            value: c,
            dimension,
        }
    }

    #[inline(always)]
    fn variable(x: f64, axis: usize, dimension: usize, &(): &'arena Self::Workspace) -> Self {
        assert!(
            axis < dimension,
            "runtime value variable axis out of bounds"
        );
        Self {
            value: x,
            dimension,
        }
    }

    #[inline(always)]
    fn symmetric_quadratic_form<C: SymmetricQuadraticCoefficients>(
        inputs: &[Self],
        coefficients: &C,
        dimension: usize,
        &(): &'arena Self::Workspace,
    ) -> Self {
        assert_eq!(inputs.len(), coefficients.dimension());
        assert!(inputs.iter().all(|input| input.dimension == dimension));
        Self {
            value: coefficients.quadratic_value(inputs, |input| input.value),
            dimension,
        }
    }

    #[inline(always)]
    fn linear_combination(
        inputs: &[Self],
        weights: &[f64],
        dimension: usize,
        &(): &'arena Self::Workspace,
    ) -> Self {
        assert_eq!(inputs.len(), weights.len());
        assert!(inputs.iter().all(|input| input.dimension == dimension));
        let value = inputs
            .iter()
            .zip(weights)
            .map(|(input, &weight)| input.value * weight)
            .sum();
        Self { value, dimension }
    }

    #[inline(always)]
    fn add_constant(&self, constant: f64, &(): &'arena Self::Workspace) -> Self {
        Self {
            value: self.value + constant,
            dimension: self.dimension,
        }
    }

    #[inline(always)]
    fn multiply_add(&self, right: &Self, addend: &Self) -> Self {
        self.assert_same_dimension(right);
        self.assert_same_dimension(addend);
        Self {
            value: self.value * right.value + addend.value,
            dimension: self.dimension,
        }
    }

    #[inline(always)]
    fn composed_sum(
        inputs: &[Self],
        derivative_stacks: &[[f64; 5]],
        dimension: usize,
        &(): &'arena Self::Workspace,
    ) -> Self {
        assert_eq!(inputs.len(), derivative_stacks.len());
        assert!(inputs.iter().all(|input| input.dimension == dimension));
        Self {
            value: derivative_stacks.iter().map(|stack| stack[0]).sum(),
            dimension,
        }
    }

    #[inline(always)]
    fn product(&self, right: &Self) -> Self {
        self.mul(right)
    }

    #[inline(always)]
    fn affine_compose(
        &self,
        _: f64,
        _: f64,
        derivative_stack: [f64; 5],
        &(): &'arena Self::Workspace,
    ) -> Self {
        Self {
            value: derivative_stack[0],
            dimension: self.dimension,
        }
    }

    #[inline(always)]
    fn affine_composed_sum(
        inputs: &[Self],
        input_scales: &[f64],
        derivative_stacks: &[[f64; 5]],
        dimension: usize,
        &(): &'arena Self::Workspace,
    ) -> Self {
        assert_eq!(inputs.len(), input_scales.len());
        assert_eq!(inputs.len(), derivative_stacks.len());
        assert!(inputs.iter().all(|input| input.dimension == dimension));
        Self {
            value: derivative_stacks.iter().map(|stack| stack[0]).sum(),
            dimension,
        }
    }

    #[inline(always)]
    fn dimension(&self) -> usize {
        self.dimension
    }

    #[inline(always)]
    fn value(&self) -> f64 {
        self.value
    }

    #[inline(always)]
    fn add(&self, other: &Self) -> Self {
        self.assert_same_dimension(other);
        Self {
            value: self.value + other.value,
            dimension: self.dimension,
        }
    }

    #[inline(always)]
    fn sub(&self, other: &Self) -> Self {
        self.assert_same_dimension(other);
        Self {
            value: self.value - other.value,
            dimension: self.dimension,
        }
    }

    #[inline(always)]
    fn mul(&self, other: &Self) -> Self {
        self.assert_same_dimension(other);
        Self {
            value: self.value * other.value,
            dimension: self.dimension,
        }
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        Self {
            value: -self.value,
            dimension: self.dimension,
        }
    }

    #[inline(always)]
    fn scale(&self, scale: f64) -> Self {
        Self {
            value: self.value * scale,
            dimension: self.dimension,
        }
    }

    #[inline(always)]
    fn compose_unary(&self, derivative_stack: [f64; 5]) -> Self {
        Self {
            value: derivative_stack[0],
            dimension: self.dimension,
        }
    }
}

impl RuntimeValue {
    #[inline(always)]
    fn assert_same_dimension(&self, other: &Self) {
        assert_eq!(self.dimension, other.dimension);
    }
}

/// Adapter that presents any const-generic [`JetScalar<K>`] through the
/// runtime-dimension interface.  It is used by derivative oracles so the same
/// row program can be instantiated at a fixed tower and at a dynamic packed
/// scalar; production code unwraps the inner fixed tower after evaluation.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct FixedRuntimeJet<S, const K: usize> {
    inner: S,
}

impl<S, const K: usize> FixedRuntimeJet<S, K> {
    /// Wrap a fixed-dimension scalar so a runtime-dimension row expression can
    /// evaluate it without authoring a second const-generic body.
    #[inline(always)]
    #[must_use]
    pub fn from_inner(inner: S) -> Self {
        Self { inner }
    }

    /// Recover the wrapped const-generic scalar.
    #[inline(always)]
    #[must_use]
    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<'arena, S: JetScalar<K>, const K: usize> RuntimeJetScalar<'arena> for FixedRuntimeJet<S, K> {
    type Workspace = ();

    #[inline(always)]
    fn constant(c: f64, dimension: usize, &(): &'arena Self::Workspace) -> Self {
        assert_eq!(dimension, K, "fixed jet dimension mismatch");
        Self {
            inner: S::constant(c),
        }
    }

    #[inline(always)]
    fn variable(x: f64, axis: usize, dimension: usize, &(): &'arena Self::Workspace) -> Self {
        assert_eq!(dimension, K, "fixed jet dimension mismatch");
        Self {
            inner: S::variable(x, axis),
        }
    }

    #[inline(always)]
    fn symmetric_quadratic_form<C: SymmetricQuadraticCoefficients>(
        inputs: &[Self],
        coefficients: &C,
        dimension: usize,
        &(): &'arena Self::Workspace,
    ) -> Self {
        assert_eq!(dimension, K, "fixed jet dimension mismatch");
        assert_eq!(inputs.len(), coefficients.dimension());
        // SAFETY: `FixedRuntimeJet<S, K>` is `repr(transparent)` with exactly one
        // non-zero-sized field of type `S`. Therefore each element has the same
        // layout and alignment as `S`; the cast preserves the allocation,
        // provenance, element count, and shared lifetime of `inputs`, and no
        // mutable reference is created while the original slice is borrowed.
        let inner =
            unsafe { std::slice::from_raw_parts(inputs.as_ptr().cast::<S>(), inputs.len()) };
        Self {
            inner: S::symmetric_quadratic_form(inner, coefficients),
        }
    }

    #[inline(always)]
    fn linear_combination(
        inputs: &[Self],
        weights: &[f64],
        dimension: usize,
        &(): &'arena Self::Workspace,
    ) -> Self {
        assert_eq!(dimension, K, "fixed jet dimension mismatch");
        assert_eq!(inputs.len(), weights.len());
        // SAFETY: `FixedRuntimeJet<S, K>` is `repr(transparent)` over its sole
        // non-zero-sized `S` field, so the shared slice has identical element
        // layout, alignment, provenance, length, and lifetime after the cast.
        let inner =
            unsafe { std::slice::from_raw_parts(inputs.as_ptr().cast::<S>(), inputs.len()) };
        Self {
            inner: S::linear_combination(inner, weights),
        }
    }

    #[inline(always)]
    fn add_constant(&self, constant: f64, &(): &'arena Self::Workspace) -> Self {
        Self {
            inner: self.inner.add_constant(constant),
        }
    }

    #[inline(always)]
    fn multiply_add(&self, right: &Self, addend: &Self) -> Self {
        Self {
            inner: self.inner.multiply_add(&right.inner, &addend.inner),
        }
    }

    #[inline(always)]
    fn composed_sum(
        inputs: &[Self],
        derivative_stacks: &[[f64; 5]],
        dimension: usize,
        &(): &'arena Self::Workspace,
    ) -> Self {
        assert_eq!(dimension, K, "fixed jet dimension mismatch");
        // SAFETY: `FixedRuntimeJet<S, K>` is `repr(transparent)` over its sole
        // non-zero-sized `S` field, so the shared slice has identical element
        // layout, alignment, provenance, length, and lifetime after the cast.
        let inner =
            unsafe { std::slice::from_raw_parts(inputs.as_ptr().cast::<S>(), inputs.len()) };
        Self {
            inner: S::composed_sum(inner, derivative_stacks),
        }
    }

    #[inline(always)]
    fn product(&self, right: &Self) -> Self {
        Self {
            inner: self.inner.product(&right.inner),
        }
    }

    #[inline(always)]
    fn affine_compose(
        &self,
        input_scale: f64,
        input_shift: f64,
        derivative_stack: [f64; 5],
        &(): &'arena Self::Workspace,
    ) -> Self {
        Self {
            inner: self
                .inner
                .affine_compose(input_scale, input_shift, derivative_stack),
        }
    }

    #[inline(always)]
    fn affine_composed_sum(
        inputs: &[Self],
        input_scales: &[f64],
        derivative_stacks: &[[f64; 5]],
        dimension: usize,
        &(): &'arena Self::Workspace,
    ) -> Self {
        assert_eq!(dimension, K, "fixed jet dimension mismatch");
        // SAFETY: `FixedRuntimeJet<S, K>` is `repr(transparent)` over its sole
        // non-zero-sized `S` field, so the shared slice has identical element
        // layout, alignment, provenance, length, and lifetime after the cast.
        let inner =
            unsafe { std::slice::from_raw_parts(inputs.as_ptr().cast::<S>(), inputs.len()) };
        Self {
            inner: S::affine_composed_sum(inner, input_scales, derivative_stacks),
        }
    }

    #[inline(always)]
    fn shared_multiply_add_affine_composed_sum<const N: usize>(
        lefts: &[&Self; N],
        right: &Self,
        addend: &Self,
        addend_scales: &[f64; N],
        input_scales: &[f64; N],
        derivative_stacks: &[[f64; 5]; N],
        dimension: usize,
        &(): &'arena Self::Workspace,
    ) -> Self {
        assert_eq!(dimension, K, "fixed jet dimension mismatch");
        let left_inner: [&S; N] = std::array::from_fn(|term| &lefts[term].inner);
        Self {
            inner: S::shared_multiply_add_affine_composed_sum(
                &left_inner,
                &right.inner,
                &addend.inner,
                addend_scales,
                input_scales,
                derivative_stacks,
            ),
        }
    }

    #[inline(always)]
    fn dimension(&self) -> usize {
        K
    }

    #[inline(always)]
    fn value(&self) -> f64 {
        self.inner.value()
    }

    #[inline(always)]
    fn add(&self, o: &Self) -> Self {
        Self {
            inner: self.inner.add(&o.inner),
        }
    }

    #[inline(always)]
    fn sub(&self, o: &Self) -> Self {
        Self {
            inner: self.inner.sub(&o.inner),
        }
    }

    #[inline(always)]
    fn mul(&self, o: &Self) -> Self {
        Self {
            inner: self.inner.mul(&o.inner),
        }
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        Self {
            inner: self.inner.neg(),
        }
    }

    #[inline(always)]
    fn scale(&self, s: f64) -> Self {
        Self {
            inner: self.inner.scale(s),
        }
    }

    #[inline(always)]
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        Self {
            inner: self.inner.compose_unary(d),
        }
    }
}

/// Reusable storage for runtime-sized packed jets. Scalar primitives write
/// into this bump arena, so arithmetic performs no heap allocation. Callers
/// reserve once per worker/chunk and [`reset`](Self::reset) between rows.
#[derive(Debug)]
pub struct DynamicJetArena {
    bump: bumpalo::Bump,
}

impl DynamicJetArena {
    /// Create an arena with the allocator's default initial chunk.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bump: bumpalo::Bump::new(),
        }
    }

    /// Create an arena with a row-program-selected initial byte capacity.
    #[must_use]
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            bump: bumpalo::Bump::with_capacity(bytes),
        }
    }

    /// Reclaim all scalar outputs and compact a fragmented high-water mark
    /// into one retained chunk.
    ///
    /// `bumpalo::Bump::reset` keeps only its newest chunk and returns every
    /// older chunk to the global allocator. Runtime jet tapes routinely span
    /// several geometrically-grown chunks, so a bare reset would allocate
    /// those discarded chunks again on every row. When reset exposes that
    /// fragmentation, replace the empty arena once with a single chunk large
    /// enough for the complete prior tape. Later equal-or-smaller rows reuse
    /// that chunk without allocator traffic.
    pub fn reset(&mut self) {
        let high_water = self.bump.allocated_bytes();
        self.bump.reset();
        if self.bump.allocated_bytes() < high_water {
            self.bump = bumpalo::Bump::with_capacity(high_water);
        }
    }

    /// Bytes currently reserved from the global allocator. A warm-reset-warm
    /// benchmark uses this to prove the second row requires no arena growth.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.bump.allocated_bytes()
    }

    #[inline(always)]
    fn zeros(&self, len: usize) -> &mut [f64] {
        self.bump.alloc_slice_fill_copy(len, 0.0)
    }

    /// Allocate and initialize a runtime-sized slice in the arena. Row programs
    /// use this for their primary-scalar arrays so those arrays share the same
    /// reusable workspace as derivative channels.
    #[inline(always)]
    pub fn alloc_slice_fill_with<T>(&self, len: usize, fill: impl FnMut(usize) -> T) -> &mut [T] {
        self.bump.alloc_slice_fill_with(len, fill)
    }
}

impl Default for DynamicJetArena {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime-sized packed first-order scalar: value plus arena-backed gradient.
#[derive(Clone, Copy, Debug)]
pub struct DynamicOrder1<'arena> {
    arena: &'arena DynamicJetArena,
    /// Value channel.
    pub v: f64,
    /// Gradient channel, length [`Self::dimension`].
    pub g: &'arena [f64],
}

impl DynamicOrder1<'_> {
    /// Gradient channel.
    #[inline]
    #[must_use]
    pub fn g(&self) -> &[f64] {
        self.g
    }

    #[inline]
    fn assert_compatible(&self, o: &Self) {
        assert_eq!(
            self.g.len(),
            o.g.len(),
            "dynamic first-order jet dimension mismatch"
        );
        assert!(
            std::ptr::eq(self.arena, o.arena),
            "dynamic jets belong to different arenas"
        );
    }
}

impl<'arena> RuntimeJetScalar<'arena> for DynamicOrder1<'arena> {
    type Workspace = DynamicJetArena;

    fn constant(c: f64, dimension: usize, arena: &'arena DynamicJetArena) -> Self {
        Self {
            arena,
            v: c,
            g: arena.zeros(dimension),
        }
    }

    fn variable(x: f64, axis: usize, dimension: usize, arena: &'arena DynamicJetArena) -> Self {
        assert!(
            axis < dimension,
            "dynamic first-order jet axis out of bounds"
        );
        let g = arena.zeros(dimension);
        g[axis] = 1.0;
        Self { arena, v: x, g }
    }

    fn dimension(&self) -> usize {
        self.g.len()
    }
    fn value(&self) -> f64 {
        self.v
    }

    fn add(&self, o: &Self) -> Self {
        self.assert_compatible(o);
        let g = self.arena.zeros(self.dimension());
        for i in 0..g.len() {
            g[i] = self.g[i] + o.g[i];
        }
        Self {
            arena: self.arena,
            v: self.v + o.v,
            g,
        }
    }

    fn sub(&self, o: &Self) -> Self {
        self.assert_compatible(o);
        let g = self.arena.zeros(self.dimension());
        for i in 0..g.len() {
            g[i] = self.g[i] - o.g[i];
        }
        Self {
            arena: self.arena,
            v: self.v - o.v,
            g,
        }
    }

    fn mul(&self, o: &Self) -> Self {
        self.assert_compatible(o);
        let g = self.arena.zeros(self.dimension());
        for i in 0..g.len() {
            g[i] = self.v * o.g[i] + self.g[i] * o.v;
        }
        Self {
            arena: self.arena,
            v: self.v * o.v,
            g,
        }
    }

    fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    fn scale(&self, s: f64) -> Self {
        let g = self.arena.zeros(self.dimension());
        for i in 0..g.len() {
            g[i] = self.g[i] * s;
        }
        Self {
            arena: self.arena,
            v: self.v * s,
            g,
        }
    }

    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let g = self.arena.zeros(self.dimension());
        for i in 0..g.len() {
            g[i] = d[1] * self.g[i];
        }
        Self {
            arena: self.arena,
            v: d[0],
            g,
        }
    }
}

/// Runtime-sized packed second-order scalar: value, gradient, and a row-major
/// Hessian. Storage is `O(K^2)` in the row's actual primary dimension and comes
/// from the row's reusable [`DynamicJetArena`].
#[derive(Clone, Copy, Debug)]
pub struct DynamicOrder2<'arena> {
    arena: &'arena DynamicJetArena,
    /// Value channel.
    pub v: f64,
    /// Gradient channel.
    pub g: &'arena [f64],
    /// Row-major Hessian channel.
    pub h: &'arena [f64],
}

impl DynamicOrder2<'_> {
    /// Construct an arena-backed second-order scalar from channel functions.
    ///
    /// This is the allocation-free extension seam for exact projected products
    /// whose channel law is not ordinary scalar multiplication. Each channel is
    /// written directly into the row arena, so downstream runtime-jet programs
    /// can keep their specialized algebra without materializing temporary
    /// `Vec`s or exposing the arena pointer stored by [`DynamicOrder2`]. The
    /// Hessian function is evaluated on the upper triangle and mirrored because
    /// a scalar Hessian is symmetric.
    #[inline]
    #[must_use]
    pub fn from_channel_functions<'arena>(
        value: f64,
        dimension: usize,
        arena: &'arena DynamicJetArena,
        mut gradient: impl FnMut(usize) -> f64,
        mut hessian: impl FnMut(usize, usize) -> f64,
    ) -> DynamicOrder2<'arena> {
        let g = arena.alloc_slice_fill_with(dimension, |axis| gradient(axis));
        let h = arena.zeros(dimension * dimension);
        for row in 0..dimension {
            for column in row..dimension {
                let channel = hessian(row, column);
                h[row * dimension + column] = channel;
                h[column * dimension + row] = channel;
            }
        }
        DynamicOrder2 {
            arena,
            v: value,
            g,
            h,
        }
    }

    /// Gradient channel.
    #[inline]
    #[must_use]
    pub fn g(&self) -> &[f64] {
        self.g
    }

    /// Row-major Hessian channel.
    #[inline]
    #[must_use]
    pub fn h(&self) -> &[f64] {
        self.h
    }

    /// Hessian entry `(row, col)`.
    #[inline]
    #[must_use]
    pub fn h_at(&self, row: usize, col: usize) -> f64 {
        self.h[row * self.dimension() + col]
    }

    #[inline(always)]
    fn assert_compatible(&self, o: &Self) {
        assert_eq!(
            self.g.len(),
            o.g.len(),
            "dynamic second-order jet dimension mismatch"
        );
        assert_eq!(
            self.h.len(),
            o.h.len(),
            "dynamic second-order jet Hessian mismatch"
        );
        assert!(
            std::ptr::eq(self.arena, o.arena),
            "dynamic jets belong to different arenas"
        );
    }
}

impl<'arena> RuntimeJetScalar<'arena> for DynamicOrder2<'arena> {
    type Workspace = DynamicJetArena;

    #[inline(always)]
    fn constant(c: f64, dimension: usize, arena: &'arena DynamicJetArena) -> Self {
        Self {
            arena,
            v: c,
            g: arena.zeros(dimension),
            h: arena.zeros(dimension * dimension),
        }
    }

    #[inline(always)]
    fn variable(x: f64, axis: usize, dimension: usize, arena: &'arena DynamicJetArena) -> Self {
        assert!(
            axis < dimension,
            "dynamic second-order jet axis out of bounds"
        );
        let g = arena.zeros(dimension);
        g[axis] = 1.0;
        Self {
            arena,
            v: x,
            g,
            h: arena.zeros(dimension * dimension),
        }
    }

    #[inline(always)]
    fn symmetric_quadratic_form<C: SymmetricQuadraticCoefficients>(
        inputs: &[Self],
        coefficients: &C,
        dimension: usize,
        arena: &'arena DynamicJetArena,
    ) -> Self {
        assert_eq!(inputs.len(), coefficients.dimension());
        assert!(
            inputs.iter().all(|input| {
                input.dimension() == dimension && std::ptr::eq(input.arena, arena)
            }),
            "dynamic quadratic-form jets must share dimension and arena"
        );
        let input_dimension = inputs.len();
        let values = arena.zeros(input_dimension);
        for (value, input) in values.iter_mut().zip(inputs) {
            *value = input.v;
        }
        let projected = arena.zeros(input_dimension);
        coefficients.multiply(values, projected);

        let mut value = 0.0;
        for axis in 0..input_dimension {
            value += values[axis] * projected[axis];
        }
        let gradient = arena.zeros(dimension);
        for primary in 0..dimension {
            let mut channel = 0.0;
            for axis in 0..input_dimension {
                channel += projected[axis] * inputs[axis].g[primary];
            }
            gradient[primary] = 2.0 * channel;
        }
        let hessian = arena.zeros(dimension * dimension);
        let input_gradient = arena.zeros(input_dimension);
        let projected_gradient = arena.zeros(input_dimension);
        for primary_b in 0..dimension {
            for row in 0..input_dimension {
                input_gradient[row] = inputs[row].g[primary_b];
            }
            coefficients.multiply(input_gradient, projected_gradient);
            for primary_a in 0..=primary_b {
                let mut inherited = 0.0;
                let mut curvature = 0.0;
                for row in 0..input_dimension {
                    inherited += projected[row] * inputs[row].h[primary_a * dimension + primary_b];
                    curvature += inputs[row].g[primary_a] * projected_gradient[row];
                }
                let channel = 2.0 * (inherited + curvature);
                hessian[primary_a * dimension + primary_b] = channel;
                hessian[primary_b * dimension + primary_a] = channel;
            }
        }
        Self {
            arena,
            v: value,
            g: gradient,
            h: hessian,
        }
    }

    #[inline(always)]
    fn product(&self, right: &Self) -> Self {
        self.assert_compatible(right);
        let dimension = self.dimension();
        let gradient = self.arena.zeros(dimension);
        let hessian = self.arena.zeros(dimension * dimension);
        for primary in 0..dimension {
            gradient[primary] = self.v * right.g[primary] + self.g[primary] * right.v;
            for other in primary..dimension {
                let index = primary * dimension + other;
                let channel = self.v * right.h[index]
                    + self.g[primary] * right.g[other]
                    + self.g[other] * right.g[primary]
                    + self.h[index] * right.v;
                hessian[index] = channel;
                hessian[other * dimension + primary] = channel;
            }
        }
        Self {
            arena: self.arena,
            v: self.v * right.v,
            g: gradient,
            h: hessian,
        }
    }

    #[inline(always)]
    fn affine_compose(
        &self,
        input_scale: f64,
        input_shift: f64,
        derivative_stack: [f64; 5],
        arena: &'arena DynamicJetArena,
    ) -> Self {
        assert!(std::ptr::eq(self.arena, arena));
        assert!(input_shift.is_finite(), "affine input shift must be finite");
        let dimension = self.dimension();
        let first = derivative_stack[1] * input_scale;
        let second = derivative_stack[2] * input_scale * input_scale;
        let gradient = arena.zeros(dimension);
        let hessian = arena.zeros(dimension * dimension);
        for primary in 0..dimension {
            gradient[primary] = first * self.g[primary];
            for other in primary..dimension {
                let index = primary * dimension + other;
                let channel = first * self.h[index] + second * self.g[primary] * self.g[other];
                hessian[index] = channel;
                hessian[other * dimension + primary] = channel;
            }
        }
        Self {
            arena,
            v: derivative_stack[0],
            g: gradient,
            h: hessian,
        }
    }

    #[inline(always)]
    fn affine_composed_sum(
        inputs: &[Self],
        input_scales: &[f64],
        derivative_stacks: &[[f64; 5]],
        dimension: usize,
        arena: &'arena DynamicJetArena,
    ) -> Self {
        assert_eq!(inputs.len(), input_scales.len());
        assert_eq!(inputs.len(), derivative_stacks.len());
        assert!(
            inputs.iter().all(|input| {
                input.dimension() == dimension && std::ptr::eq(input.arena, arena)
            }),
            "dynamic affine-composed-sum jets must share dimension and arena"
        );
        let gradient = arena.zeros(dimension);
        let hessian = arena.zeros(dimension * dimension);
        let mut value = 0.0;
        for ((input, &input_scale), stack) in inputs.iter().zip(input_scales).zip(derivative_stacks)
        {
            let first = stack[1] * input_scale;
            let second = stack[2] * input_scale * input_scale;
            value += stack[0];
            for primary in 0..dimension {
                gradient[primary] += first * input.g[primary];
                for other in primary..dimension {
                    let index = primary * dimension + other;
                    hessian[index] +=
                        first * input.h[index] + second * input.g[primary] * input.g[other];
                }
            }
        }
        for primary in 0..dimension {
            for other in primary + 1..dimension {
                hessian[other * dimension + primary] = hessian[primary * dimension + other];
            }
        }
        Self {
            arena,
            v: value,
            g: gradient,
            h: hessian,
        }
    }

    #[inline(always)]
    fn shared_multiply_add_affine_composed_sum<const N: usize>(
        lefts: &[&Self; N],
        right: &Self,
        addend: &Self,
        addend_scales: &[f64; N],
        input_scales: &[f64; N],
        derivative_stacks: &[[f64; 5]; N],
        dimension: usize,
        arena: &'arena DynamicJetArena,
    ) -> Self {
        assert!(
            lefts.iter().all(|input| {
                input.dimension() == dimension && std::ptr::eq(input.arena, arena)
            }) && (N == 0 || (right.dimension() == dimension && std::ptr::eq(right.arena, arena))),
            "dynamic fused product-composition jets must share dimension and arena"
        );
        let addend_live = addend_scales.iter().any(|&scale| scale != 0.0);
        assert!(
            !addend_live || (addend.dimension() == dimension && std::ptr::eq(addend.arena, arena)),
            "live dynamic fused addends must share dimension and arena"
        );
        let (representatives, term_sources, source_count) =
            canonical_shared_source_schedule(|term, representative| {
                std::ptr::eq(lefts[term], lefts[representative])
                    && addend_scales[term] == addend_scales[representative]
            });
        let source_gradients = arena.zeros(source_count * dimension);
        let mut source_firsts = [0.0; N];
        let mut source_seconds = [0.0; N];
        let gradient = arena.zeros(dimension);
        let hessian = arena.zeros(dimension * dimension);
        let mut value = 0.0;
        let mut right_first = 0.0;
        let mut addend_first = 0.0;
        for term in 0..N {
            let first = derivative_stacks[term][1] * input_scales[term];
            let second = derivative_stacks[term][2] * input_scales[term] * input_scales[term];
            let source = term_sources[term];
            source_firsts[source] += first;
            source_seconds[source] += second;
            value += derivative_stacks[term][0];
        }
        for source in 0..source_count {
            let term = representatives[source];
            let first = source_firsts[source];
            right_first += first * lefts[term].v;
            addend_first += first * addend_scales[term];
            for primary in 0..dimension {
                let product_gradient =
                    lefts[term].v * right.g[primary] + lefts[term].g[primary] * right.v;
                let inner_gradient = if addend_scales[term] == 0.0 {
                    product_gradient
                } else if addend_scales[term] == 1.0 {
                    product_gradient + addend.g[primary]
                } else {
                    product_gradient + addend_scales[term] * addend.g[primary]
                };
                source_gradients[source * dimension + primary] = inner_gradient;
                gradient[primary] += first * lefts[term].g[primary] * right.v;
            }
        }
        if N != 0 {
            for primary in 0..dimension {
                gradient[primary] += right_first * right.g[primary];
            }
        }
        if addend_live {
            for primary in 0..dimension {
                gradient[primary] += addend_first * addend.g[primary];
            }
        }
        for primary in 0..dimension {
            for other in primary..dimension {
                let index = primary * dimension + other;
                let mut channel = if N == 0 {
                    0.0
                } else {
                    right_first * right.h[index]
                };
                if addend_live {
                    channel += addend_first * addend.h[index];
                }
                for source in 0..source_count {
                    let term = representatives[source];
                    let local_product_hessian = lefts[term].g[primary] * right.g[other]
                        + lefts[term].g[other] * right.g[primary]
                        + lefts[term].h[index] * right.v;
                    channel += source_firsts[source] * local_product_hessian
                        + source_seconds[source]
                            * source_gradients[source * dimension + primary]
                            * source_gradients[source * dimension + other];
                }
                hessian[index] = channel;
                hessian[other * dimension + primary] = channel;
            }
        }
        Self {
            arena,
            v: value,
            g: gradient,
            h: hessian,
        }
    }

    #[inline(always)]
    fn add_constant(&self, constant: f64, arena: &'arena DynamicJetArena) -> Self {
        assert!(std::ptr::eq(self.arena, arena));
        Self {
            arena,
            v: self.v + constant,
            g: self.g,
            h: self.h,
        }
    }

    #[inline(always)]
    fn multiply_add(&self, right: &Self, addend: &Self) -> Self {
        self.assert_compatible(right);
        self.assert_compatible(addend);
        let dimension = self.dimension();
        let gradient = self.arena.zeros(dimension);
        let hessian = self.arena.zeros(dimension * dimension);
        for primary in 0..dimension {
            gradient[primary] =
                self.v * right.g[primary] + self.g[primary] * right.v + addend.g[primary];
            for other in primary..dimension {
                let index = primary * dimension + other;
                let channel = self.v * right.h[index]
                    + self.g[primary] * right.g[other]
                    + self.g[other] * right.g[primary]
                    + self.h[index] * right.v
                    + addend.h[index];
                hessian[index] = channel;
                hessian[other * dimension + primary] = channel;
            }
        }
        Self {
            arena: self.arena,
            v: self.v * right.v + addend.v,
            g: gradient,
            h: hessian,
        }
    }

    #[inline(always)]
    fn composed_sum(
        inputs: &[Self],
        derivative_stacks: &[[f64; 5]],
        dimension: usize,
        arena: &'arena DynamicJetArena,
    ) -> Self {
        assert_eq!(inputs.len(), derivative_stacks.len());
        assert!(
            inputs.iter().all(|input| {
                input.dimension() == dimension && std::ptr::eq(input.arena, arena)
            }),
            "dynamic composed-sum jets must share dimension and arena"
        );
        let gradient = arena.zeros(dimension);
        let hessian = arena.zeros(dimension * dimension);
        let mut value = 0.0;
        for (input, stack) in inputs.iter().zip(derivative_stacks) {
            value += stack[0];
            for primary in 0..dimension {
                gradient[primary] += stack[1] * input.g[primary];
                for other in primary..dimension {
                    let index = primary * dimension + other;
                    hessian[index] +=
                        stack[1] * input.h[index] + stack[2] * input.g[primary] * input.g[other];
                }
            }
        }
        for primary in 0..dimension {
            for other in primary + 1..dimension {
                hessian[other * dimension + primary] = hessian[primary * dimension + other];
            }
        }
        Self {
            arena,
            v: value,
            g: gradient,
            h: hessian,
        }
    }

    #[inline(always)]
    fn linear_combination(
        inputs: &[Self],
        weights: &[f64],
        dimension: usize,
        arena: &'arena DynamicJetArena,
    ) -> Self {
        assert_eq!(inputs.len(), weights.len());
        assert!(
            inputs.iter().all(|input| {
                input.dimension() == dimension && std::ptr::eq(input.arena, arena)
            }),
            "dynamic linear-combination jets must share dimension and arena"
        );
        let mut value = 0.0;
        for (input, &weight) in inputs.iter().zip(weights) {
            value += input.v * weight;
        }
        let gradient = arena.zeros(dimension);
        let hessian = arena.zeros(dimension * dimension);
        for primary in 0..dimension {
            for (input, &weight) in inputs.iter().zip(weights) {
                gradient[primary] += input.g[primary] * weight;
            }
            for other in primary..dimension {
                let index = primary * dimension + other;
                for (input, &weight) in inputs.iter().zip(weights) {
                    hessian[index] += input.h[index] * weight;
                }
                hessian[other * dimension + primary] = hessian[index];
            }
        }
        Self {
            arena,
            v: value,
            g: gradient,
            h: hessian,
        }
    }

    #[inline(always)]
    fn dimension(&self) -> usize {
        self.g.len()
    }

    #[inline(always)]
    fn value(&self) -> f64 {
        self.v
    }

    #[inline(always)]
    fn add(&self, o: &Self) -> Self {
        self.assert_compatible(o);
        let dimension = self.dimension();
        let g = self.arena.zeros(dimension);
        let h = self.arena.zeros(self.h.len());
        for i in 0..g.len() {
            g[i] = self.g[i] + o.g[i];
        }
        for row in 0..dimension {
            for column in row..dimension {
                let index = row * dimension + column;
                let channel = self.h[index] + o.h[index];
                h[index] = channel;
                h[column * dimension + row] = channel;
            }
        }
        Self {
            arena: self.arena,
            v: self.v + o.v,
            g,
            h,
        }
    }

    #[inline(always)]
    fn sub(&self, o: &Self) -> Self {
        self.assert_compatible(o);
        let dimension = self.dimension();
        let g = self.arena.zeros(dimension);
        let h = self.arena.zeros(self.h.len());
        for i in 0..g.len() {
            g[i] = self.g[i] - o.g[i];
        }
        for row in 0..dimension {
            for column in row..dimension {
                let index = row * dimension + column;
                let channel = self.h[index] - o.h[index];
                h[index] = channel;
                h[column * dimension + row] = channel;
            }
        }
        Self {
            arena: self.arena,
            v: self.v - o.v,
            g,
            h,
        }
    }

    #[inline(always)]
    fn mul(&self, o: &Self) -> Self {
        self.assert_compatible(o);
        let n = self.dimension();
        let g = self.arena.zeros(n);
        let h = self.arena.zeros(n * n);
        for i in 0..n {
            g[i] = self.v * o.g[i] + self.g[i] * o.v;
        }
        for i in 0..n {
            for j in i..n {
                let ij = i * n + j;
                let hij =
                    self.v * o.h[ij] + self.g[i] * o.g[j] + self.g[j] * o.g[i] + self.h[ij] * o.v;
                h[ij] = hij;
                h[j * n + i] = hij;
            }
        }
        Self {
            arena: self.arena,
            v: self.v * o.v,
            g,
            h,
        }
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    #[inline(always)]
    fn scale(&self, s: f64) -> Self {
        let dimension = self.dimension();
        let g = self.arena.zeros(dimension);
        let h = self.arena.zeros(self.h.len());
        for i in 0..g.len() {
            g[i] = self.g[i] * s;
        }
        for row in 0..dimension {
            for column in row..dimension {
                let index = row * dimension + column;
                let channel = self.h[index] * s;
                h[index] = channel;
                h[column * dimension + row] = channel;
            }
        }
        Self {
            arena: self.arena,
            v: self.v * s,
            g,
            h,
        }
    }

    #[inline(always)]
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let n = self.dimension();
        let g = self.arena.zeros(n);
        let h = self.arena.zeros(n * n);
        for i in 0..n {
            g[i] = d[1] * self.g[i];
        }
        for i in 0..n {
            for j in i..n {
                let ij = i * n + j;
                let channel = d[1] * self.h[ij] + d[2] * self.g[i] * self.g[j];
                h[ij] = channel;
                h[j * n + i] = channel;
            }
        }
        Self {
            arena: self.arena,
            v: d[0],
            g,
            h,
        }
    }
}

/// Runtime-sized one-seed scalar for a Hessian-contracted third derivative.
#[derive(Clone, Copy, Debug)]
pub struct DynamicOneSeed<'arena> {
    /// Base value/gradient/Hessian channels.
    pub base: DynamicOrder2<'arena>,
    /// Nilpotent `epsilon` coefficient.
    pub eps: DynamicOrder2<'arena>,
}

impl<'arena> DynamicOneSeed<'arena> {
    /// Seed one primary with the supplied contraction direction component.
    #[inline(always)]
    #[must_use]
    pub fn seed_direction(
        x: f64,
        axis: usize,
        u_axis: f64,
        dimension: usize,
        arena: &'arena DynamicJetArena,
    ) -> Self {
        Self {
            base: DynamicOrder2::variable(x, axis, dimension, arena),
            eps: DynamicOrder2::constant(u_axis, dimension, arena),
        }
    }

    /// Row-major contracted-third matrix.
    #[inline(always)]
    #[must_use]
    pub fn contracted_third(&self) -> &[f64] {
        self.eps.h()
    }
}

impl<'arena> RuntimeJetScalar<'arena> for DynamicOneSeed<'arena> {
    type Workspace = DynamicJetArena;

    #[inline(always)]
    fn constant(c: f64, dimension: usize, arena: &'arena DynamicJetArena) -> Self {
        Self {
            base: DynamicOrder2::constant(c, dimension, arena),
            eps: DynamicOrder2::constant(0.0, dimension, arena),
        }
    }

    #[inline(always)]
    fn variable(x: f64, axis: usize, dimension: usize, arena: &'arena DynamicJetArena) -> Self {
        Self {
            base: DynamicOrder2::variable(x, axis, dimension, arena),
            eps: DynamicOrder2::constant(0.0, dimension, arena),
        }
    }

    #[inline(always)]
    fn dimension(&self) -> usize {
        self.base.dimension()
    }

    #[inline(always)]
    fn value(&self) -> f64 {
        self.base.value()
    }

    #[inline(always)]
    fn add(&self, o: &Self) -> Self {
        Self {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
        }
    }

    #[inline(always)]
    fn sub(&self, o: &Self) -> Self {
        Self {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
        }
    }

    #[inline(always)]
    fn mul(&self, o: &Self) -> Self {
        self.base.assert_compatible(&o.base);
        self.eps.assert_compatible(&o.eps);
        Self {
            base: self.base.mul(&o.base),
            eps: DynamicOrder2::from_channel_functions(
                self.base.v * o.eps.v + self.eps.v * o.base.v,
                self.dimension(),
                self.base.arena,
                |i| {
                    self.base.v * o.eps.g[i]
                        + self.base.g[i] * o.eps.v
                        + self.eps.v * o.base.g[i]
                        + self.eps.g[i] * o.base.v
                },
                |i, j| {
                    let ij = i * self.dimension() + j;
                    self.base.v * o.eps.h[ij]
                        + self.base.g[i] * o.eps.g[j]
                        + self.base.g[j] * o.eps.g[i]
                        + self.base.h[ij] * o.eps.v
                        + self.eps.v * o.base.h[ij]
                        + self.eps.g[i] * o.base.g[j]
                        + self.eps.g[j] * o.base.g[i]
                        + self.eps.h[ij] * o.base.v
                },
            ),
        }
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        Self {
            base: self.base.neg(),
            eps: self.eps.neg(),
        }
    }

    #[inline(always)]
    fn scale(&self, s: f64) -> Self {
        Self {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
        }
    }

    #[inline(always)]
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let base = self.base.compose_unary(d);
        let dimension = self.dimension();
        let eps = DynamicOrder2::from_channel_functions(
            d[1] * self.eps.v,
            dimension,
            self.base.arena,
            |i| d[2] * self.base.g[i] * self.eps.v + d[1] * self.eps.g[i],
            |i, j| {
                let ij = i * dimension + j;
                d[1] * self.eps.h[ij]
                    + d[2]
                        * (self.base.g[i] * self.eps.g[j]
                            + self.base.g[j] * self.eps.g[i]
                            + self.base.h[ij] * self.eps.v)
                    + d[3] * self.base.g[i] * self.base.g[j] * self.eps.v
            },
        );
        Self { base, eps }
    }
}

/// Reusable arena plus lane count for runtime directional batches.
///
/// A caller resets this once per row/chunk and evaluates every requested
/// third- or fourth-derivative contraction in one row-program pass. The bump
/// retains its largest chunk, so warmed rows do not return to the global
/// allocator.
#[derive(Debug)]
pub struct DynamicJetBatchWorkspace {
    arena: DynamicJetArena,
    lanes: usize,
}

impl DynamicJetBatchWorkspace {
    /// Create a reusable workspace for `lanes` simultaneous directions.
    #[must_use]
    pub fn new(lanes: usize) -> Self {
        Self {
            arena: DynamicJetArena::new(),
            lanes,
        }
    }

    /// Reclaim all jet storage and select the next evaluation's lane count.
    pub fn reset(&mut self, lanes: usize) {
        self.arena.reset();
        self.lanes = lanes;
    }

    /// Bytes retained by the bump allocator after the largest evaluation.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.arena.allocated_bytes()
    }

    /// Allocate a primary array in the same arena as every scalar channel.
    #[inline(always)]
    pub fn alloc_slice_fill_with<T>(&self, len: usize, fill: impl FnMut(usize) -> T) -> &mut [T] {
        self.arena.alloc_slice_fill_with(len, fill)
    }
}

/// Runtime-sized batch of one-seed contractions sharing one order-two base.
///
/// Lane `l` is algebraically identical to a standalone [`DynamicOneSeed`]
/// seeded by direction `u_l`, but the value/gradient/Hessian base is evaluated
/// once rather than once per direction. The output lane's epsilon Hessian is
/// `sum_c d³f/(dx_a dx_b dx_c) u_l[c]`.
#[derive(Clone, Copy, Debug)]
pub struct DynamicOneSeedBatch<'arena> {
    /// Shared value/gradient/Hessian channels.
    pub base: DynamicOrder2<'arena>,
    /// One nilpotent epsilon coefficient per contraction direction.
    eps: &'arena [DynamicOrder2<'arena>],
}

impl<'arena> DynamicOneSeedBatch<'arena> {
    /// Seed one primary across every direction lane.
    #[inline(always)]
    #[must_use]
    pub fn seed_directions(
        x: f64,
        axis: usize,
        dimension: usize,
        workspace: &'arena DynamicJetBatchWorkspace,
        mut direction_at: impl FnMut(usize) -> f64,
    ) -> Self {
        let eps = workspace
            .arena
            .alloc_slice_fill_with(workspace.lanes, |lane| {
                DynamicOrder2::constant(direction_at(lane), dimension, &workspace.arena)
            });
        Self {
            base: DynamicOrder2::variable(x, axis, dimension, &workspace.arena),
            eps,
        }
    }

    /// Number of simultaneous contraction directions.
    #[inline(always)]
    #[must_use]
    pub fn lanes(&self) -> usize {
        self.eps.len()
    }

    /// Row-major contracted-third matrix for one direction lane.
    #[inline(always)]
    #[must_use]
    pub fn contracted_third(&self, lane: usize) -> &[f64] {
        self.eps[lane].h()
    }

    #[inline(always)]
    fn assert_compatible(&self, other: &Self) {
        self.base.assert_compatible(&other.base);
        assert_eq!(
            self.eps.len(),
            other.eps.len(),
            "dynamic one-seed batch lane mismatch"
        );
    }
}

impl<'arena> RuntimeJetScalar<'arena> for DynamicOneSeedBatch<'arena> {
    type Workspace = DynamicJetBatchWorkspace;

    #[inline(always)]
    fn constant(c: f64, dimension: usize, workspace: &'arena DynamicJetBatchWorkspace) -> Self {
        let eps = workspace.arena.alloc_slice_fill_with(workspace.lanes, |_| {
            DynamicOrder2::constant(0.0, dimension, &workspace.arena)
        });
        Self {
            base: DynamicOrder2::constant(c, dimension, &workspace.arena),
            eps,
        }
    }

    #[inline(always)]
    fn variable(
        x: f64,
        axis: usize,
        dimension: usize,
        workspace: &'arena DynamicJetBatchWorkspace,
    ) -> Self {
        let eps = workspace.arena.alloc_slice_fill_with(workspace.lanes, |_| {
            DynamicOrder2::constant(0.0, dimension, &workspace.arena)
        });
        Self {
            base: DynamicOrder2::variable(x, axis, dimension, &workspace.arena),
            eps,
        }
    }

    #[inline(always)]
    fn dimension(&self) -> usize {
        self.base.dimension()
    }

    #[inline(always)]
    fn value(&self) -> f64 {
        self.base.value()
    }

    #[inline(always)]
    fn add(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let eps = self
            .base
            .arena
            .alloc_slice_fill_with(self.eps.len(), |lane| self.eps[lane].add(&other.eps[lane]));
        Self {
            base: self.base.add(&other.base),
            eps,
        }
    }

    #[inline(always)]
    fn sub(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let eps = self
            .base
            .arena
            .alloc_slice_fill_with(self.eps.len(), |lane| self.eps[lane].sub(&other.eps[lane]));
        Self {
            base: self.base.sub(&other.base),
            eps,
        }
    }

    #[inline(always)]
    fn mul(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let eps = self
            .base
            .arena
            .alloc_slice_fill_with(self.eps.len(), |lane| {
                self.base
                    .mul(&other.eps[lane])
                    .add(&self.eps[lane].mul(&other.base))
            });
        Self {
            base: self.base.mul(&other.base),
            eps,
        }
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    #[inline(always)]
    fn scale(&self, scale: f64) -> Self {
        let eps = self
            .base
            .arena
            .alloc_slice_fill_with(self.eps.len(), |lane| self.eps[lane].scale(scale));
        Self {
            base: self.base.scale(scale),
            eps,
        }
    }

    #[inline(always)]
    fn compose_unary(&self, derivatives: [f64; 5]) -> Self {
        let fprime = self.base.compose_unary([
            derivatives[1],
            derivatives[2],
            derivatives[3],
            derivatives[4],
            derivatives[4],
        ]);
        let eps = self
            .base
            .arena
            .alloc_slice_fill_with(self.eps.len(), |lane| fprime.mul(&self.eps[lane]));
        Self {
            base: self.base.compose_unary(derivatives),
            eps,
        }
    }
}

/// Runtime-sized batch of two-seed contractions sharing one order-two base.
///
/// Lane `l` is algebraically identical to a standalone [`DynamicTwoSeed`]
/// seeded by the direction pair `(u_l, v_l)`, while the base row expression is
/// traversed once for the entire pair batch. Storage is `O(lanes * K^2)`; no
/// dense fourth-order tensor is formed.
#[derive(Clone, Copy, Debug)]
pub struct DynamicTwoSeedBatch<'arena> {
    /// Shared value/gradient/Hessian channels.
    pub base: DynamicOrder2<'arena>,
    eps: &'arena [DynamicOrder2<'arena>],
    del: &'arena [DynamicOrder2<'arena>],
    eps_del: &'arena [DynamicOrder2<'arena>],
}

impl<'arena> DynamicTwoSeedBatch<'arena> {
    /// Seed one primary across every direction-pair lane.
    #[inline(always)]
    #[must_use]
    pub fn seed_direction_pairs(
        x: f64,
        axis: usize,
        dimension: usize,
        workspace: &'arena DynamicJetBatchWorkspace,
        mut direction_pair_at: impl FnMut(usize) -> (f64, f64),
    ) -> Self {
        let directions = workspace
            .arena
            .alloc_slice_fill_with(workspace.lanes, |lane| direction_pair_at(lane));
        let eps = workspace
            .arena
            .alloc_slice_fill_with(workspace.lanes, |lane| {
                DynamicOrder2::constant(directions[lane].0, dimension, &workspace.arena)
            });
        let del = workspace
            .arena
            .alloc_slice_fill_with(workspace.lanes, |lane| {
                DynamicOrder2::constant(directions[lane].1, dimension, &workspace.arena)
            });
        let eps_del = workspace.arena.alloc_slice_fill_with(workspace.lanes, |_| {
            DynamicOrder2::constant(0.0, dimension, &workspace.arena)
        });
        Self {
            base: DynamicOrder2::variable(x, axis, dimension, &workspace.arena),
            eps,
            del,
            eps_del,
        }
    }

    /// Number of simultaneous contraction pairs.
    #[inline(always)]
    #[must_use]
    pub fn lanes(&self) -> usize {
        self.eps.len()
    }

    /// Row-major contracted-fourth matrix for one direction-pair lane.
    #[inline(always)]
    #[must_use]
    pub fn contracted_fourth(&self, lane: usize) -> &[f64] {
        self.eps_del[lane].h()
    }

    #[inline(always)]
    fn assert_compatible(&self, other: &Self) {
        self.base.assert_compatible(&other.base);
        assert_eq!(
            self.eps.len(),
            other.eps.len(),
            "dynamic two-seed batch lane mismatch"
        );
        assert_eq!(
            self.del.len(),
            self.eps.len(),
            "dynamic two-seed batch delta mismatch"
        );
        assert_eq!(
            self.eps_del.len(),
            self.eps.len(),
            "dynamic two-seed batch cross mismatch"
        );
    }
}

impl<'arena> RuntimeJetScalar<'arena> for DynamicTwoSeedBatch<'arena> {
    type Workspace = DynamicJetBatchWorkspace;

    #[inline(always)]
    fn constant(c: f64, dimension: usize, workspace: &'arena Self::Workspace) -> Self {
        let zero = workspace.arena.alloc_slice_fill_with(workspace.lanes, |_| {
            DynamicOrder2::constant(0.0, dimension, &workspace.arena)
        });
        Self {
            base: DynamicOrder2::constant(c, dimension, &workspace.arena),
            eps: zero,
            del: zero,
            eps_del: zero,
        }
    }

    #[inline(always)]
    fn variable(x: f64, axis: usize, dimension: usize, workspace: &'arena Self::Workspace) -> Self {
        let zero = workspace.arena.alloc_slice_fill_with(workspace.lanes, |_| {
            DynamicOrder2::constant(0.0, dimension, &workspace.arena)
        });
        Self {
            base: DynamicOrder2::variable(x, axis, dimension, &workspace.arena),
            eps: zero,
            del: zero,
            eps_del: zero,
        }
    }

    #[inline(always)]
    fn dimension(&self) -> usize {
        self.base.dimension()
    }

    #[inline(always)]
    fn value(&self) -> f64 {
        self.base.value()
    }

    #[inline(always)]
    fn add(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let arena = self.base.arena;
        let eps =
            arena.alloc_slice_fill_with(self.lanes(), |lane| self.eps[lane].add(&other.eps[lane]));
        let del =
            arena.alloc_slice_fill_with(self.lanes(), |lane| self.del[lane].add(&other.del[lane]));
        let eps_del = arena.alloc_slice_fill_with(self.lanes(), |lane| {
            self.eps_del[lane].add(&other.eps_del[lane])
        });
        Self {
            base: self.base.add(&other.base),
            eps,
            del,
            eps_del,
        }
    }

    #[inline(always)]
    fn sub(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let arena = self.base.arena;
        let eps =
            arena.alloc_slice_fill_with(self.lanes(), |lane| self.eps[lane].sub(&other.eps[lane]));
        let del =
            arena.alloc_slice_fill_with(self.lanes(), |lane| self.del[lane].sub(&other.del[lane]));
        let eps_del = arena.alloc_slice_fill_with(self.lanes(), |lane| {
            self.eps_del[lane].sub(&other.eps_del[lane])
        });
        Self {
            base: self.base.sub(&other.base),
            eps,
            del,
            eps_del,
        }
    }

    #[inline(always)]
    fn mul(&self, other: &Self) -> Self {
        self.assert_compatible(other);
        let arena = self.base.arena;
        let eps = arena.alloc_slice_fill_with(self.lanes(), |lane| {
            self.base
                .mul(&other.eps[lane])
                .add(&self.eps[lane].mul(&other.base))
        });
        let del = arena.alloc_slice_fill_with(self.lanes(), |lane| {
            self.base
                .mul(&other.del[lane])
                .add(&self.del[lane].mul(&other.base))
        });
        let eps_del = arena.alloc_slice_fill_with(self.lanes(), |lane| {
            self.base
                .mul(&other.eps_del[lane])
                .add(&self.eps[lane].mul(&other.del[lane]))
                .add(&self.del[lane].mul(&other.eps[lane]))
                .add(&self.eps_del[lane].mul(&other.base))
        });
        Self {
            base: self.base.mul(&other.base),
            eps,
            del,
            eps_del,
        }
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    #[inline(always)]
    fn scale(&self, scale: f64) -> Self {
        let arena = self.base.arena;
        let eps = arena.alloc_slice_fill_with(self.lanes(), |lane| self.eps[lane].scale(scale));
        let del = arena.alloc_slice_fill_with(self.lanes(), |lane| self.del[lane].scale(scale));
        let eps_del =
            arena.alloc_slice_fill_with(self.lanes(), |lane| self.eps_del[lane].scale(scale));
        Self {
            base: self.base.scale(scale),
            eps,
            del,
            eps_del,
        }
    }

    #[inline(always)]
    fn compose_unary(&self, derivatives: [f64; 5]) -> Self {
        let arena = self.base.arena;
        let fprime = self.base.compose_unary([
            derivatives[1],
            derivatives[2],
            derivatives[3],
            derivatives[4],
            derivatives[4],
        ]);
        let fsecond = self.base.compose_unary([
            derivatives[2],
            derivatives[3],
            derivatives[4],
            derivatives[4],
            derivatives[4],
        ]);
        let eps = arena.alloc_slice_fill_with(self.lanes(), |lane| fprime.mul(&self.eps[lane]));
        let del = arena.alloc_slice_fill_with(self.lanes(), |lane| fprime.mul(&self.del[lane]));
        let eps_del = arena.alloc_slice_fill_with(self.lanes(), |lane| {
            fsecond
                .mul(&self.eps[lane])
                .mul(&self.del[lane])
                .add(&fprime.mul(&self.eps_del[lane]))
        });
        Self {
            base: self.base.compose_unary(derivatives),
            eps,
            del,
            eps_del,
        }
    }
}

/// Runtime-sized two-seed scalar for a Hessian-contracted fourth derivative.
#[derive(Clone, Copy, Debug)]
pub struct DynamicTwoSeed<'arena> {
    /// Base value/gradient/Hessian channels.
    pub base: DynamicOrder2<'arena>,
    /// Nilpotent `epsilon` coefficient.
    pub eps: DynamicOrder2<'arena>,
    /// Nilpotent `delta` coefficient.
    pub del: DynamicOrder2<'arena>,
    /// Mixed `epsilon delta` coefficient.
    pub eps_del: DynamicOrder2<'arena>,
}

impl<'arena> DynamicTwoSeed<'arena> {
    /// Seed one primary with both contraction direction components.
    #[inline(always)]
    #[must_use]
    pub fn seed(
        x: f64,
        axis: usize,
        u_axis: f64,
        v_axis: f64,
        dimension: usize,
        arena: &'arena DynamicJetArena,
    ) -> Self {
        Self {
            base: DynamicOrder2::variable(x, axis, dimension, arena),
            eps: DynamicOrder2::constant(u_axis, dimension, arena),
            del: DynamicOrder2::constant(v_axis, dimension, arena),
            eps_del: DynamicOrder2::constant(0.0, dimension, arena),
        }
    }

    /// Row-major contracted-fourth matrix.
    #[inline(always)]
    #[must_use]
    pub fn contracted_fourth(&self) -> &[f64] {
        self.eps_del.h()
    }
}

impl<'arena> RuntimeJetScalar<'arena> for DynamicTwoSeed<'arena> {
    type Workspace = DynamicJetArena;

    #[inline(always)]
    fn constant(c: f64, dimension: usize, arena: &'arena DynamicJetArena) -> Self {
        Self {
            base: DynamicOrder2::constant(c, dimension, arena),
            eps: DynamicOrder2::constant(0.0, dimension, arena),
            del: DynamicOrder2::constant(0.0, dimension, arena),
            eps_del: DynamicOrder2::constant(0.0, dimension, arena),
        }
    }

    #[inline(always)]
    fn variable(x: f64, axis: usize, dimension: usize, arena: &'arena DynamicJetArena) -> Self {
        Self {
            base: DynamicOrder2::variable(x, axis, dimension, arena),
            eps: DynamicOrder2::constant(0.0, dimension, arena),
            del: DynamicOrder2::constant(0.0, dimension, arena),
            eps_del: DynamicOrder2::constant(0.0, dimension, arena),
        }
    }

    #[inline(always)]
    fn dimension(&self) -> usize {
        self.base.dimension()
    }

    #[inline(always)]
    fn value(&self) -> f64 {
        self.base.value()
    }

    #[inline(always)]
    fn add(&self, o: &Self) -> Self {
        Self {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
            del: self.del.add(&o.del),
            eps_del: self.eps_del.add(&o.eps_del),
        }
    }

    #[inline(always)]
    fn sub(&self, o: &Self) -> Self {
        Self {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
            del: self.del.sub(&o.del),
            eps_del: self.eps_del.sub(&o.eps_del),
        }
    }

    #[inline(always)]
    fn mul(&self, o: &Self) -> Self {
        let base = self.base.mul(&o.base);
        let eps = self.base.mul(&o.eps).add(&self.eps.mul(&o.base));
        let del = self.base.mul(&o.del).add(&self.del.mul(&o.base));
        let eps_del = self
            .base
            .mul(&o.eps_del)
            .add(&self.eps.mul(&o.del))
            .add(&self.del.mul(&o.eps))
            .add(&self.eps_del.mul(&o.base));
        Self {
            base,
            eps,
            del,
            eps_del,
        }
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        Self {
            base: self.base.neg(),
            eps: self.eps.neg(),
            del: self.del.neg(),
            eps_del: self.eps_del.neg(),
        }
    }

    #[inline(always)]
    fn scale(&self, s: f64) -> Self {
        Self {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
            del: self.del.scale(s),
            eps_del: self.eps_del.scale(s),
        }
    }

    #[inline(always)]
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let base = self.base.compose_unary(d);
        let fprime = self.base.compose_unary([d[1], d[2], d[3], d[4], d[4]]);
        let fsecond = self.base.compose_unary([d[2], d[3], d[4], d[4], d[4]]);
        let eps = fprime.mul(&self.eps);
        let del = fprime.mul(&self.del);
        let eps_del = fsecond
            .mul(&self.eps)
            .mul(&self.del)
            .add(&fprime.mul(&self.eps_del));
        Self {
            base,
            eps,
            del,
            eps_del,
        }
    }
}

// ── Order2<K> ergonomic operator overloads (doc §A.1) ───────────────────
//
// The dispersion-family row NLLs are written with `+`/`-`/`*` operators over
// the primaries (mirroring how they read as `Tower4` expressions). These
// delegate channel-for-channel to the inner `Tower2` arithmetic (which has
// `Add`/`Mul`; `Sub`/`Neg` are expressed as `+ (-1)·rhs` exactly as the
// `crate::nested_dual::JetField::sub` / `crate::nested_dual::JetField::neg` impls do), so an `Order2` expression is
// bit-identical to the same `Tower4` expression's order-≤2 channels.

impl<const K: usize> std::ops::Add for Order2<K> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Order2(self.0 + o.0)
    }
}

impl<const K: usize> std::ops::Add<f64> for Order2<K> {
    type Output = Self;
    #[inline]
    fn add(self, c: f64) -> Self {
        Order2(self.0 + c)
    }
}

impl<const K: usize> std::ops::Sub for Order2<K> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Order2(self.0 + o.0.scale(-1.0))
    }
}

impl<const K: usize> std::ops::Sub<f64> for Order2<K> {
    type Output = Self;
    #[inline]
    fn sub(self, c: f64) -> Self {
        Order2(self.0 + (-c))
    }
}

impl<const K: usize> std::ops::Mul for Order2<K> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Order2(crate::jet_tower::Tower2::mul(&self.0, &o.0))
    }
}

impl<const K: usize> std::ops::Mul<f64> for Order2<K> {
    type Output = Self;
    #[inline]
    fn mul(self, c: f64) -> Self {
        Order2(self.0.scale(c))
    }
}

impl<const K: usize> std::ops::Neg for Order2<K> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Order2(self.0.scale(-1.0))
    }
}

/// Filtered Hensel lift of a SCALAR implicit state `a(θ)` defined by the
/// constraint `F(a, θ) = 0`, evaluated in ANY [`JetScalar`] algebra `S` (doc
/// §11, "A generic implicit-lift operator for every production scalar").
///
/// This is the perf-respecting alternative to lifting through a dense
/// `Tower4<K+1>` (which carries the implicit variable as an extra dense axis):
/// the state `a` lives directly in the consumer's own `K`-primary algebra
/// `S` — `Order2<K>` for value/gradient/Hessian, `Tower4<K>` for the full
/// `t3`/`t4` — never paying for an extra variable.
///
/// **Method.** Fixed-Jacobian Newton in the nilpotent algebra. By the
/// filtered-lift theorem (doc §11.1), if `F_a := ∂F/∂a(a₀, θ₀)` is the primal
/// Jacobian at the base point and `inv_fa = 1/F_a`, then the iteration
/// `A ← A − inv_fa · F(A, θ)` raises the filtration degree of the residual by
/// at least one per step: each step kills exactly one graded layer. Starting
/// from `A = const(a₀)` (whose residual lies in `F¹` because `θ − θ₀ ∈ 𝔫`),
/// `iters` equal to the algebra's nilpotency order returns the *exact* lifted
/// jet (`Order2`: 2, `OneSeed`: 3, `Tower4`/`TwoSeed`: 4). The value channel of
/// `A` never moves — `F(A, θ).value() = F(a₀, θ₀) = 0` at the certified root —
/// so a caller may precompute every primitive's derivative stack at the fixed
/// base index once and let the cheap polynomial composition repeat per step.
///
/// `f` evaluates the constraint `F(a, θ)` in `S` (capturing the seeded
/// parameter jets `θ`); `a0` is the certified scalar root `F(a₀, θ₀) ≈ 0`.
pub fn filtered_implicit_solve_scalar<const K: usize, S: JetScalar<K>>(
    a0: f64,
    inv_fa: f64,
    iters: usize,
    f: impl Fn(&S) -> S,
) -> S {
    let mut a = S::constant(a0);
    for _ in 0..iters {
        let residual = f(&a);
        a = a.sub(&residual.scale(inv_fa));
    }
    a
}

// ── PatternedOrder2<P, K, H>: compile-time sparse Hessian ───────────────

/// Runtime-dimension form of [`filtered_implicit_solve_scalar`].
///
/// The state is lifted directly in the caller's [`RuntimeJetScalar`] algebra,
/// so one row expression can serve a runtime-sized value/gradient/Hessian
/// evaluation (`DynamicOrder2`), a Hessian-contracted third derivative
/// (`DynamicOneSeed`), and a Hessian-contracted fourth derivative
/// (`DynamicTwoSeed`). `iters` is the algebra's nilpotency order: two, three,
/// and four respectively. No implicit axis is appended to the derivative
/// storage.
pub fn filtered_implicit_solve_runtime_scalar<'arena, S: RuntimeJetScalar<'arena>>(
    a0: f64,
    inv_fa: f64,
    iters: usize,
    dimension: usize,
    workspace: &'arena S::Workspace,
    f: impl Fn(&S) -> S,
) -> S {
    let mut a = S::constant(a0, dimension, workspace);
    for _ in 0..iters {
        let residual = f(&a);
        a = a.sub(&residual.scale(inv_fa));
    }
    a
}

/// Compile-time upper-triangle Hessian pattern for [`PatternedOrder2`].
///
/// `PAIRS` contains exactly the `(row, column)` channels a row program can
/// produce, with `row <= column`.  Pointwise Taylor algebra never couples one
/// Hessian pair through another: output `H[i,j]` depends only on the two input
/// `H[i,j]` channels and their `g[i]`/`g[j]` channels.  It is therefore exact to
/// omit structurally impossible pairs rather than multiplying their zeros.
pub trait HessianPattern<const K: usize, const H: usize> {
    const PAIRS: [(usize, usize); H];
    const PAIR_BITS: [[u128; K]; K];
}

/// Build the symmetric axis-pair → patterned-slot lookup used by dependency
/// propagation in [`PatternedOrder2`].
pub const fn hessian_pair_bits<const K: usize, const H: usize>(
    pairs: [(usize, usize); H],
) -> [[u128; K]; K] {
    let mut table = [[0u128; K]; K];
    let mut slot = 0;
    while slot < H {
        let (i, j) = pairs[slot];
        let bit = 1u128 << slot;
        table[i][j] = bit;
        table[j][i] = bit;
        slot += 1;
    }
    table
}

/// Exact order-two jet with a dense gradient and a compile-time patterned
/// upper-triangle Hessian.
///
/// This carries `1 + K + H` scalars instead of `1 + K + K²`.  Its arithmetic is
/// the same Leibniz/Faà-di-Bruno algebra as [`Order2`], evaluated only for the
/// Hessian pairs declared by `P`.  A family row NLL remains written once over
/// [`JetScalar`]; the pattern is an execution schedule, not a second derivative
/// formula.
#[derive(Debug)]
pub struct PatternedOrder2<P, const K: usize, const H: usize> {
    v: f64,
    g: [f64; K],
    h: [f64; H],
    gradient_mask: u128,
    hessian_mask: u128,
    pattern: std::marker::PhantomData<fn() -> P>,
}

impl<P, const K: usize, const H: usize> Copy for PatternedOrder2<P, K, H> {}

impl<P, const K: usize, const H: usize> Clone for PatternedOrder2<P, K, H> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P, const K: usize, const H: usize> PatternedOrder2<P, K, H>
where
    P: HessianPattern<K, H>,
{
    #[inline]
    #[must_use]
    pub fn g(&self) -> [f64; K] {
        self.g
    }

    /// Expand the patterned upper triangle into the dense symmetric shape
    /// required by the existing `RowKernel` interface. Missing pairs are exact
    /// structural zeros.
    #[inline]
    #[must_use]
    pub fn h(&self) -> [[f64; K]; K] {
        let mut dense = [[0.0; K]; K];
        for (slot, &(i, j)) in P::PAIRS.iter().enumerate() {
            dense[i][j] = self.h[slot];
            dense[j][i] = self.h[slot];
        }
        dense
    }

    #[inline]
    fn pair_mask_between(left: u128, right: u128) -> u128 {
        let mut result = 0u128;
        let mut left_axes = left;
        while left_axes != 0 {
            let i = left_axes.trailing_zeros() as usize;
            left_axes &= left_axes - 1;
            let mut right_axes = right;
            while right_axes != 0 {
                let j = right_axes.trailing_zeros() as usize;
                right_axes &= right_axes - 1;
                result |= P::PAIR_BITS[i][j];
            }
        }
        result
    }
}

impl<P, const K: usize, const H: usize> JetScalar<K> for PatternedOrder2<P, K, H>
where
    P: HessianPattern<K, H>,
{
    #[inline]
    fn constant(c: f64) -> Self {
        Self {
            v: c,
            g: [0.0; K],
            h: [0.0; H],
            gradient_mask: 0,
            hessian_mask: 0,
            pattern: std::marker::PhantomData,
        }
    }

    #[inline]
    fn variable(x: f64, axis: usize) -> Self {
        let mut out = Self::constant(x);
        if axis < K {
            out.g[axis] = 1.0;
            out.gradient_mask = 1u128 << axis;
        }
        out
    }
}

impl<P, const K: usize, const H: usize> crate::nested_dual::JetField for PatternedOrder2<P, K, H>
where
    P: HessianPattern<K, H>,
{
    #[inline]
    fn value(&self) -> f64 {
        self.v
    }

    #[inline]
    fn add(&self, other: &Self) -> Self {
        let mut out = Self::constant(self.v + other.v);
        out.gradient_mask = self.gradient_mask | other.gradient_mask;
        let mut gradient_mask = out.gradient_mask;
        while gradient_mask != 0 {
            let i = gradient_mask.trailing_zeros() as usize;
            gradient_mask &= gradient_mask - 1;
            out.g[i] = self.g[i] + other.g[i];
        }
        out.hessian_mask = self.hessian_mask | other.hessian_mask;
        let mut hessian_mask = out.hessian_mask;
        while hessian_mask != 0 {
            let slot = hessian_mask.trailing_zeros() as usize;
            hessian_mask &= hessian_mask - 1;
            out.h[slot] = self.h[slot] + other.h[slot];
        }
        out
    }

    #[inline]
    fn sub(&self, other: &Self) -> Self {
        let mut out = Self::constant(self.v - other.v);
        out.gradient_mask = self.gradient_mask | other.gradient_mask;
        let mut gradient_mask = out.gradient_mask;
        while gradient_mask != 0 {
            let i = gradient_mask.trailing_zeros() as usize;
            gradient_mask &= gradient_mask - 1;
            out.g[i] = self.g[i] - other.g[i];
        }
        out.hessian_mask = self.hessian_mask | other.hessian_mask;
        let mut hessian_mask = out.hessian_mask;
        while hessian_mask != 0 {
            let slot = hessian_mask.trailing_zeros() as usize;
            hessian_mask &= hessian_mask - 1;
            out.h[slot] = self.h[slot] - other.h[slot];
        }
        out
    }

    #[inline]
    fn mul(&self, other: &Self) -> Self {
        let mut out = Self::constant(self.v * other.v);
        out.gradient_mask = self.gradient_mask | other.gradient_mask;
        let mut gradient_mask = out.gradient_mask;
        while gradient_mask != 0 {
            let i = gradient_mask.trailing_zeros() as usize;
            gradient_mask &= gradient_mask - 1;
            out.g[i] = self.v * other.g[i] + self.g[i] * other.v;
        }
        out.hessian_mask = self.hessian_mask
            | other.hessian_mask
            | Self::pair_mask_between(self.gradient_mask, other.gradient_mask);
        let mut hessian_mask = out.hessian_mask;
        while hessian_mask != 0 {
            let slot = hessian_mask.trailing_zeros() as usize;
            hessian_mask &= hessian_mask - 1;
            let (i, j) = P::PAIRS[slot];
            out.h[slot] = self.v * other.h[slot]
                + self.g[i] * other.g[j]
                + self.g[j] * other.g[i]
                + self.h[slot] * other.v;
        }
        out
    }

    #[inline]
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    #[inline]
    fn scale(&self, scale: f64) -> Self {
        let mut out = Self::constant(self.v * scale);
        out.gradient_mask = self.gradient_mask;
        let mut gradient_mask = out.gradient_mask;
        while gradient_mask != 0 {
            let i = gradient_mask.trailing_zeros() as usize;
            gradient_mask &= gradient_mask - 1;
            out.g[i] = self.g[i] * scale;
        }
        out.hessian_mask = self.hessian_mask;
        let mut hessian_mask = out.hessian_mask;
        while hessian_mask != 0 {
            let slot = hessian_mask.trailing_zeros() as usize;
            hessian_mask &= hessian_mask - 1;
            out.h[slot] = self.h[slot] * scale;
        }
        out
    }

    #[inline]
    fn compose_unary(&self, derivatives: [f64; 5]) -> Self {
        let mut out = Self::constant(derivatives[0]);
        out.gradient_mask = self.gradient_mask;
        let mut gradient_mask = out.gradient_mask;
        while gradient_mask != 0 {
            let i = gradient_mask.trailing_zeros() as usize;
            gradient_mask &= gradient_mask - 1;
            out.g[i] = derivatives[1] * self.g[i];
        }
        out.hessian_mask =
            self.hessian_mask | Self::pair_mask_between(self.gradient_mask, self.gradient_mask);
        let mut hessian_mask = out.hessian_mask;
        while hessian_mask != 0 {
            let slot = hessian_mask.trailing_zeros() as usize;
            hessian_mask &= hessian_mask - 1;
            let (i, j) = P::PAIRS[slot];
            out.h[slot] = derivatives[2] * self.g[i] * self.g[j] + derivatives[1] * self.h[slot];
        }
        out
    }
}

// ── Order2<K>: value / gradient / Hessian (doc §A.1) ────────────────────

/// Truncated SECOND-order scalar: value `v`, gradient `g_a`, Hessian `H_{ab}`.
///
/// This is a thin newtype over the production [`crate::jet_tower::Tower2`], so
/// its `(v, g, H)` channels are obtained by the SAME formulas — and are
/// therefore bit-identical to both [`crate::jet_tower::Tower2`] and the order-≤2
/// channels of a full [`crate::jet_tower::Tower4`] (doc §A.1, "Bit-identity with
/// the full tower"). The wrapper exists only to satisfy the generic
/// [`JetScalar`] interface (the `compose_unary` / `add` / `sub` / `neg` /
/// `recip` the trait demands, which `Tower2` does not expose by that shape) —
/// every channel is delegated to `Tower2` arithmetic unchanged.
#[derive(Clone, Copy, Debug)]
pub struct Order2<const K: usize>(pub crate::jet_tower::Tower2<K>);

impl<const K: usize> Order2<K> {
    /// Read the gradient channel `g_a = ∂ℓ/∂p_a`.
    #[inline]
    pub fn g(&self) -> [f64; K] {
        self.0.g
    }

    /// Read the Hessian channel.
    #[inline]
    pub fn h(&self) -> [[f64; K]; K] {
        self.0.h
    }
}

impl<const K: usize> JetScalar<K> for Order2<K> {
    fn constant(c: f64) -> Self {
        Order2(crate::jet_tower::Tower2::constant(c))
    }
    fn variable(x: f64, axis: usize) -> Self {
        Order2(crate::jet_tower::Tower2::variable(x, axis))
    }

    #[inline(always)]
    fn symmetric_quadratic_form<C: SymmetricQuadraticCoefficients>(
        inputs: &[Self],
        coefficients: &C,
    ) -> Self {
        assert_eq!(inputs.len(), coefficients.dimension());
        let input_dimension = inputs.len();
        assert!(input_dimension <= K);
        let mut values = [0.0; K];
        for axis in 0..input_dimension {
            values[axis] = inputs[axis].0.v;
        }
        let mut projected = [0.0; K];
        coefficients.multiply(
            &values[..input_dimension],
            &mut projected[..input_dimension],
        );

        let mut out = crate::jet_tower::Tower2::zero();
        for axis in 0..input_dimension {
            out.v += values[axis] * projected[axis];
        }
        for primary in 0..K {
            let mut channel = 0.0;
            for axis in 0..input_dimension {
                channel += projected[axis] * inputs[axis].0.g[primary];
            }
            out.g[primary] = 2.0 * channel;
        }
        let mut input_gradient = [0.0; K];
        let mut projected_gradient = [0.0; K];
        for primary_b in 0..K {
            for row in 0..input_dimension {
                input_gradient[row] = inputs[row].0.g[primary_b];
            }
            coefficients.multiply(
                &input_gradient[..input_dimension],
                &mut projected_gradient[..input_dimension],
            );
            for primary_a in 0..=primary_b {
                let mut inherited = 0.0;
                let mut curvature = 0.0;
                for row in 0..input_dimension {
                    inherited += projected[row] * inputs[row].0.h[primary_a][primary_b];
                    curvature += inputs[row].0.g[primary_a] * projected_gradient[row];
                }
                let channel = 2.0 * (inherited + curvature);
                out.h[primary_a][primary_b] = channel;
                out.h[primary_b][primary_a] = channel;
            }
        }
        Order2(out)
    }

    #[inline(always)]
    fn linear_combination(inputs: &[Self], weights: &[f64]) -> Self {
        assert_eq!(inputs.len(), weights.len());
        let mut out = crate::jet_tower::Tower2::zero();
        for (input, &weight) in inputs.iter().zip(weights) {
            out.v += input.0.v * weight;
        }
        for primary in 0..K {
            for (input, &weight) in inputs.iter().zip(weights) {
                out.g[primary] += input.0.g[primary] * weight;
            }
            for other in primary..K {
                for (input, &weight) in inputs.iter().zip(weights) {
                    out.h[primary][other] += input.0.h[primary][other] * weight;
                }
                out.h[other][primary] = out.h[primary][other];
            }
        }
        Order2(out)
    }

    #[inline(always)]
    fn add_constant(&self, constant: f64) -> Self {
        let mut out = *self;
        out.0.v += constant;
        out
    }

    #[inline(always)]
    fn multiply_add(&self, right: &Self, addend: &Self) -> Self {
        let mut out = crate::jet_tower::Tower2::zero();
        out.v = self.0.v * right.0.v + addend.0.v;
        for primary in 0..K {
            out.g[primary] =
                self.0.v * right.0.g[primary] + self.0.g[primary] * right.0.v + addend.0.g[primary];
            for other in primary..K {
                let channel = self.0.v * right.0.h[primary][other]
                    + self.0.g[primary] * right.0.g[other]
                    + self.0.g[other] * right.0.g[primary]
                    + self.0.h[primary][other] * right.0.v
                    + addend.0.h[primary][other];
                out.h[primary][other] = channel;
                out.h[other][primary] = channel;
            }
        }
        Order2(out)
    }

    #[inline(always)]
    fn product(&self, right: &Self) -> Self {
        let mut out = crate::jet_tower::Tower2::zero();
        out.v = self.0.v * right.0.v;
        for primary in 0..K {
            out.g[primary] = self.0.v * right.0.g[primary] + self.0.g[primary] * right.0.v;
            for other in primary..K {
                let channel = self.0.v * right.0.h[primary][other]
                    + self.0.g[primary] * right.0.g[other]
                    + self.0.g[other] * right.0.g[primary]
                    + self.0.h[primary][other] * right.0.v;
                out.h[primary][other] = channel;
                out.h[other][primary] = channel;
            }
        }
        Order2(out)
    }

    #[inline(always)]
    fn affine_compose(
        &self,
        input_scale: f64,
        input_shift: f64,
        derivative_stack: [f64; 5],
    ) -> Self {
        assert!(input_shift.is_finite(), "affine input shift must be finite");
        let first = derivative_stack[1] * input_scale;
        let second = derivative_stack[2] * input_scale * input_scale;
        let mut out = crate::jet_tower::Tower2::zero();
        out.v = derivative_stack[0];
        for primary in 0..K {
            out.g[primary] = first * self.0.g[primary];
            for other in primary..K {
                let channel =
                    first * self.0.h[primary][other] + second * self.0.g[primary] * self.0.g[other];
                out.h[primary][other] = channel;
                out.h[other][primary] = channel;
            }
        }
        Order2(out)
    }

    #[inline(always)]
    fn affine_composed_sum(
        inputs: &[Self],
        input_scales: &[f64],
        derivative_stacks: &[[f64; 5]],
    ) -> Self {
        assert_eq!(inputs.len(), input_scales.len());
        assert_eq!(inputs.len(), derivative_stacks.len());
        let mut out = crate::jet_tower::Tower2::zero();
        for ((input, &input_scale), stack) in inputs.iter().zip(input_scales).zip(derivative_stacks)
        {
            let first = stack[1] * input_scale;
            let second = stack[2] * input_scale * input_scale;
            out.v += stack[0];
            for primary in 0..K {
                out.g[primary] += first * input.0.g[primary];
                for other in primary..K {
                    out.h[primary][other] += first * input.0.h[primary][other]
                        + second * input.0.g[primary] * input.0.g[other];
                }
            }
        }
        for primary in 0..K {
            for other in primary + 1..K {
                out.h[other][primary] = out.h[primary][other];
            }
        }
        Order2(out)
    }

    #[inline(always)]
    fn shared_multiply_add_affine_composed_sum<const N: usize>(
        lefts: &[&Self; N],
        right: &Self,
        addend: &Self,
        addend_scales: &[f64; N],
        input_scales: &[f64; N],
        derivative_stacks: &[[f64; 5]; N],
    ) -> Self {
        let (representatives, term_sources, source_count) =
            canonical_shared_source_schedule(|term, representative| {
                std::ptr::eq(lefts[term], lefts[representative])
                    && addend_scales[term] == addend_scales[representative]
            });
        let mut source_gradients = [[0.0; K]; N];
        let mut source_firsts = [0.0; N];
        let mut source_seconds = [0.0; N];
        let mut out = crate::jet_tower::Tower2::zero();
        let mut right_first = 0.0;
        let mut addend_first = 0.0;
        for term in 0..N {
            let first = derivative_stacks[term][1] * input_scales[term];
            let second = derivative_stacks[term][2] * input_scales[term] * input_scales[term];
            let source = term_sources[term];
            source_firsts[source] += first;
            source_seconds[source] += second;
            out.v += derivative_stacks[term][0];
        }
        for source in 0..source_count {
            let term = representatives[source];
            let first = source_firsts[source];
            right_first += first * lefts[term].0.v;
            addend_first += first * addend_scales[term];
            for primary in 0..K {
                let product_gradient =
                    lefts[term].0.v * right.0.g[primary] + lefts[term].0.g[primary] * right.0.v;
                let inner_gradient = if addend_scales[term] == 0.0 {
                    product_gradient
                } else if addend_scales[term] == 1.0 {
                    product_gradient + addend.0.g[primary]
                } else {
                    product_gradient + addend_scales[term] * addend.0.g[primary]
                };
                source_gradients[source][primary] = inner_gradient;
                out.g[primary] += first * lefts[term].0.g[primary] * right.0.v;
            }
        }
        if N != 0 {
            for primary in 0..K {
                out.g[primary] += right_first * right.0.g[primary];
            }
        }
        let addend_live = addend_scales.iter().any(|&scale| scale != 0.0);
        if addend_live {
            for primary in 0..K {
                out.g[primary] += addend_first * addend.0.g[primary];
            }
        }
        for primary in 0..K {
            for other in primary..K {
                let mut channel = if N == 0 {
                    0.0
                } else {
                    right_first * right.0.h[primary][other]
                };
                if addend_live {
                    channel += addend_first * addend.0.h[primary][other];
                }
                for source in 0..source_count {
                    let term = representatives[source];
                    let local_product_hessian = lefts[term].0.g[primary] * right.0.g[other]
                        + lefts[term].0.g[other] * right.0.g[primary]
                        + lefts[term].0.h[primary][other] * right.0.v;
                    channel += source_firsts[source] * local_product_hessian
                        + source_seconds[source]
                            * source_gradients[source][primary]
                            * source_gradients[source][other];
                }
                out.h[primary][other] = channel;
                out.h[other][primary] = channel;
            }
        }
        Order2(out)
    }

    #[inline(always)]
    fn composed_sum(inputs: &[Self], derivative_stacks: &[[f64; 5]]) -> Self {
        assert_eq!(inputs.len(), derivative_stacks.len());
        let mut out = crate::jet_tower::Tower2::zero();
        for (input, stack) in inputs.iter().zip(derivative_stacks) {
            out.v += stack[0];
            for primary in 0..K {
                out.g[primary] += stack[1] * input.0.g[primary];
                for other in primary..K {
                    out.h[primary][other] += stack[1] * input.0.h[primary][other]
                        + stack[2] * input.0.g[primary] * input.0.g[other];
                }
            }
        }
        for primary in 0..K {
            for other in primary + 1..K {
                out.h[other][primary] = out.h[primary][other];
            }
        }
        Order2(out)
    }
}

impl<const K: usize> crate::nested_dual::JetField for Order2<K> {
    fn value(&self) -> f64 {
        self.0.v
    }
    fn add(&self, o: &Self) -> Self {
        Order2(self.0 + o.0)
    }
    fn sub(&self, o: &Self) -> Self {
        // Tower2 has no Sub op; subtract by adding the negation, matching
        // Tower4::sub (self + o.scale(-1.0)).
        Order2(self.0 + o.0.scale(-1.0))
    }
    fn mul(&self, o: &Self) -> Self {
        Order2(crate::jet_tower::Tower2::mul(&self.0, &o.0))
    }
    fn neg(&self) -> Self {
        Order2(self.0.scale(-1.0))
    }
    fn scale(&self, s: f64) -> Self {
        Order2(self.0.scale(s))
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // Order-≤2 reads only [f, f', f''] of the stack.
        Order2(self.0.compose_unary([d[0], d[1], d[2]]))
    }
}

/// Static lowering target for a sum of composed, low-dimensional row atoms.
///
/// A row likelihood often depends on a large global primary vector only through
/// a few small independent indices.  Evaluating the whole expression in an
/// `Order2<K>` then pays dense `K²` arithmetic at every intermediate; carrying
/// runtime dependency masks replaces that arithmetic with branches and bit
/// scans.  This accumulator provides the ahead-of-time alternative: evaluate
/// each index once in its natural local dimension `N`, then scatter the exact
/// second-order composition into the global `K` channels through a fixed axis
/// map.  The family still owns only its scalar index expression and certified
/// unary derivative stack; this type owns the universal chain rule.
///
/// For an atom `q(x_local)` and outer stack `[f(q), f'(q), f''(q)]`, the lowered
/// channels are
///
/// ```text
/// g[a_i]       += f' q_i
/// H[a_i, a_j] += f' q_ij + f'' q_i q_j.
/// ```
///
/// Only the local upper triangle is evaluated, then mirrored into the global
/// symmetric output.  With literal `axes` and fixed `N`, LLVM unrolls this into
/// straight-line arithmetic: no dependency masks, sparse-pair lookups, or jet
/// temporaries survive into the generated schedule.
#[derive(Clone, Copy, Debug)]
pub struct MappedOrder2Accumulator<const K: usize> {
    value: f64,
    gradient: [f64; K],
    hessian: [[f64; K]; K],
}

/// Compile-time-symbolic value/gradient/Hessian of one local row atom.
///
/// The Hessian stores only its upper triangle, in row-major triangular order.
/// Instances are emitted by `gam-row-macros`; unlike a runtime forward jet,
/// they contain only final live channels and therefore introduce no seeded
/// identity arrays, dependency masks, or zero arithmetic.
#[derive(Clone, Copy, Debug)]
pub struct StaticOrder2Atom<
    const N: usize,
    const H: usize,
    const GRADIENT_BITS: u128,
    const HESSIAN_BITS: u128,
> {
    value: f64,
    gradient: [f64; N],
    hessian: [f64; H],
}

impl<const N: usize, const H: usize, const G: u128, const Q: u128> StaticOrder2Atom<N, H, G, Q> {
    /// Construct a generated atom. `H` must equal `N(N+1)/2`.
    #[inline(always)]
    #[must_use]
    pub fn new(value: f64, gradient: [f64; N], hessian: [f64; H]) -> Self {
        assert!(H == N * (N + 1) / 2, "invalid packed order-two shape");
        assert!(N <= 128 && H <= 128, "static atom sparsity mask overflow");
        Self {
            value,
            gradient,
            hessian,
        }
    }

    /// Scalar value of the generated atom.
    #[inline(always)]
    #[must_use]
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Generated local gradient.
    #[inline(always)]
    #[must_use]
    pub fn gradient(&self) -> [f64; N] {
        self.gradient
    }

    /// Generated local Hessian entry. The matrix is symmetric.
    #[inline(always)]
    #[must_use]
    pub fn hessian_at(&self, row: usize, column: usize) -> f64 {
        assert!(
            row < N && column < N,
            "static atom Hessian axis out of range"
        );
        let (row, column) = if row <= column {
            (row, column)
        } else {
            (column, row)
        };
        let index = row * (2 * N - row + 1) / 2 + column - row;
        self.hessian[index]
    }
}

/// Order-two channel reader accepted by [`MappedOrder2Accumulator`].
///
/// Both ordinary forward jets and build-time-symbolic atoms implement this
/// interface. The accumulator owns the only global scatter/chain rule.
pub trait Order2AtomChannels<const N: usize> {
    /// Structurally live local gradient channels.
    const GRADIENT_BITS: u128;
    /// Structurally live packed upper-Hessian channels.
    const HESSIAN_BITS: u128;
    /// Local gradient entry.
    fn gradient_at(&self, axis: usize) -> f64;
    /// Local Hessian entry.
    fn hessian_at(&self, row: usize, column: usize) -> f64;
}

impl<const N: usize> Order2AtomChannels<N> for Order2<N> {
    const GRADIENT_BITS: u128 = low_mask(N);
    const HESSIAN_BITS: u128 = low_mask(N * (N + 1) / 2);

    #[inline(always)]
    fn gradient_at(&self, axis: usize) -> f64 {
        self.0.g[axis]
    }

    #[inline(always)]
    fn hessian_at(&self, row: usize, column: usize) -> f64 {
        self.0.h[row][column]
    }
}

impl<const N: usize, const H: usize, const G: u128, const Q: u128> Order2AtomChannels<N>
    for StaticOrder2Atom<N, H, G, Q>
{
    const GRADIENT_BITS: u128 = G;
    const HESSIAN_BITS: u128 = Q;

    #[inline(always)]
    fn gradient_at(&self, axis: usize) -> f64 {
        self.gradient[axis]
    }

    #[inline(always)]
    fn hessian_at(&self, row: usize, column: usize) -> f64 {
        StaticOrder2Atom::hessian_at(self, row, column)
    }
}

const fn low_mask(channels: usize) -> u128 {
    if channels >= 128 {
        u128::MAX
    } else {
        (1u128 << channels) - 1
    }
}

impl<const K: usize> MappedOrder2Accumulator<K> {
    /// Empty additive accumulator.
    #[inline(always)]
    #[must_use]
    pub fn zero() -> Self {
        Self {
            value: 0.0,
            gradient: [0.0; K],
            hessian: [[0.0; K]; K],
        }
    }

    /// Scatter `f(atom)` using the exact order-two Faà di Bruno rule.
    ///
    /// `axes[i]` maps local derivative axis `i` to its global primary axis. The
    /// map must be injective. Repeated axes describe a non-injective linear
    /// pullback whose identified cross terms need multiplicity that this simple
    /// scatter deliberately does not represent.
    #[inline(always)]
    pub fn add_composed<const N: usize, const H: usize, A: Order2AtomChannels<N>>(
        &mut self,
        atom: &A,
        axes: [usize; N],
        derivatives: [f64; 3],
        value_add: bool,
        gradient_add: [bool; N],
        hessian_add: [bool; H],
    ) {
        assert!(H == N * (N + 1) / 2, "invalid mapped Hessian write shape");
        assert!(N <= 128 && H <= 128, "mapped atom sparsity mask overflow");
        assert!(
            axes.iter().all(|&axis| axis < K),
            "mapped atom axis must be within the global primary dimension"
        );
        assert!(
            axes.iter()
                .enumerate()
                .all(|(i, axis)| !axes[..i].contains(axis)),
            "mapped atom axes must be injective"
        );

        if value_add {
            self.value += derivatives[0];
        } else {
            self.value = derivatives[0];
        }
        let mut packed = 0;
        for local_i in 0..N {
            let global_i = axes[local_i];
            if A::GRADIENT_BITS & (1u128 << local_i) != 0 {
                let channel = derivatives[1] * atom.gradient_at(local_i);
                if gradient_add[local_i] {
                    self.gradient[global_i] += channel;
                } else {
                    self.gradient[global_i] = channel;
                }
            }
            for local_j in local_i..N {
                let global_j = axes[local_j];
                let inner_live = A::HESSIAN_BITS & (1u128 << packed) != 0;
                let outer_live = A::GRADIENT_BITS & (1u128 << local_i) != 0
                    && A::GRADIENT_BITS & (1u128 << local_j) != 0;
                let channel = if inner_live {
                    let inner = derivatives[1] * atom.hessian_at(local_i, local_j);
                    if outer_live {
                        inner
                            + derivatives[2] * atom.gradient_at(local_i) * atom.gradient_at(local_j)
                    } else {
                        inner
                    }
                } else if outer_live {
                    derivatives[2] * atom.gradient_at(local_i) * atom.gradient_at(local_j)
                } else {
                    packed += 1;
                    continue;
                };
                if hessian_add[packed] {
                    self.hessian[global_i][global_j] += channel;
                    if global_i != global_j {
                        self.hessian[global_j][global_i] += channel;
                    }
                } else {
                    self.hessian[global_i][global_j] = channel;
                    if global_i != global_j {
                        self.hessian[global_j][global_i] = channel;
                    }
                }
                packed += 1;
            }
        }
    }

    /// Finish the lowering in the standard row-kernel channel layout.
    #[inline(always)]
    #[must_use]
    pub fn into_channels(self) -> (f64, [f64; K], [[f64; K]; K]) {
        (self.value, self.gradient, self.hessian)
    }
}

/// One inner scalar in a runtime-width additive order-two composition.
///
/// Implementors expose only the inner gradient and Hessian. The corresponding
/// outer function's first and second derivatives live beside the source in the
/// caller's term type, so several terms that share an inner scalar may be fused
/// before entering [`DynamicOrder2Accumulator`].
pub trait DynamicOrder2Term {
    /// Outer first derivative `f'(q)` for this already-fused source.
    fn outer_first(&self) -> f64;

    /// Outer second derivative `f''(q)` for this already-fused source.
    fn outer_second(&self) -> f64;

    /// Inner first derivative `q_i`.
    fn inner_gradient(&self, axis: usize) -> f64;

    /// Inner second derivative `q_ij`.
    fn inner_hessian(&self, row: usize, column: usize) -> f64;
}

/// Allocation-minimal runtime-width lowering of an additive order-two program.
///
/// This is the dynamic-width sibling of [`MappedOrder2Accumulator`]. A caller
/// first reduces its mathematical row program to a scalar value plus `N`
/// independent composed sources. This accumulator then performs the universal
/// chain rule
///
/// ```text
/// g_i  = sum_t (f_t' q_t,i)
/// H_ij = sum_t (f_t'' q_t,i q_t,j + f_t' q_t,ij)
/// ```
///
/// in one gradient pass and one upper-triangular Hessian pass. It owns exactly
/// the two buffers that become the returned gradient and Hessian; no per-term
/// derivative buffer or jet temporary is allocated. `N` is const-generic so a
/// small fixed source set is visible to LLVM and can be unrolled.
#[derive(Debug)]
pub struct DynamicOrder2Accumulator {
    value: f64,
    gradient: Vec<f64>,
    hessian: Vec<f64>,
}

impl DynamicOrder2Accumulator {
    /// Lower a fused additive source set at runtime primary width `dimension`.
    #[inline(always)]
    #[must_use]
    pub fn from_composed_sum<T: DynamicOrder2Term, const N: usize>(
        dimension: usize,
        value: f64,
        terms: &[T; N],
    ) -> Self {
        let mut gradient = vec![0.0; dimension];
        let mut hessian = vec![0.0; dimension * dimension];

        for axis in 0..dimension {
            let mut channel = 0.0;
            for term in terms {
                channel += term.outer_first() * term.inner_gradient(axis);
            }
            gradient[axis] = channel;
        }

        for row in 0..dimension {
            for column in row..dimension {
                let mut channel = 0.0;
                for term in terms {
                    let row_gradient = term.inner_gradient(row);
                    let column_gradient = term.inner_gradient(column);
                    channel += term.outer_second() * row_gradient * column_gradient
                        + term.outer_first() * term.inner_hessian(row, column);
                }
                hessian[row * dimension + column] = channel;
                hessian[column * dimension + row] = channel;
            }
        }

        Self {
            value,
            gradient,
            hessian,
        }
    }

    /// Finish the lowering as `(value, gradient, row-major symmetric Hessian)`.
    #[inline(always)]
    #[must_use]
    pub fn into_channels(self) -> (f64, Vec<f64>, Vec<f64>) {
        (self.value, self.gradient, self.hessian)
    }
}

// ── Lane-batched Order-2 scalar: 4 rows per pass in SIMD lanes (perf) ────
//
// The hot per-row jet kernels evaluate ONE row's `(v, g, H)` tower at a time in
// scalar `f64`. A hand-written scalar derivative does the same. The throughput
// lever a jet has that scalar hand-code cannot is **row batching in SIMD
// lanes**: the order-≤2 Leibniz product `Order2::mul` is `O(K²)` independent
// per-channel float ops, and EVERY row runs the identical op graph on different
// data — the textbook SPMD shape. Packing `LANES = 4` rows into a `wide::f64x4`
// and running the algebra once per 4 rows replaces 4 scalar passes with one
// vector pass: the `K²` Hessian channel updates become `K²` NEON `.2d` / SSE2
// `pd` instructions covering 4 rows each, ~4× fewer FP instructions per row.
//
// The carried scalar field is abstracted by [`Lane`] so the SAME algebra body
// instantiates at `f64` (1 row, used as the bit-identity oracle) or
// [`wide::f64x4`] (4 rows). Bit-identity is structural, not approximate:
//
//   * Every arithmetic op is a plain lane-wise `+` / `-` / `*` (NEVER a fused
//     `mul_add`), and IEEE-754 double `+`/`-`/`*`/`/` are correctly rounded and
//     deterministic, so lane `i` of an `f64x4` op equals the scalar `f64` op on
//     that lane's inputs bit-for-bit.
//   * The transcendental derivative STACKS (`exp`/`ln`/`sqrt`/…) are produced
//     **per lane by the identical scalar code** ([`Lane::unary3`] unpacks, runs
//     the same `[f64; 3]` stack closure the scalar path runs, repacks), so the
//     only thing vectorised is the cheap rational tensor composition — the
//     library transcendental itself is the exact same `f64::exp` call per lane.
//   * The op order mirrors [`crate::jet_tower::Tower2`] term-for-term, so
//     [`Order2Lane<f64, K>`] is `to_bits`-identical to the production
//     [`Order2<K>`] (= `Tower2<K>`), and [`Order2Lane<f64x4, K>`] lane `i` is
//     `to_bits`-identical to that — proven by the `batch_tests` oracle below
//     (≥2000 random 4-row batches across `K ∈ {2,3,4,9}`).

/// The scalar field a [`Order2Lane`] carries: either a single `f64` (one row,
/// the oracle) or a [`wide::f64x4`] (four rows evaluated in SIMD lanes). All ops
/// are plain lane-wise IEEE arithmetic, so a vector op equals the scalar op on
/// each lane bit-for-bit.
pub trait Lane: Copy {
    /// Broadcast a scalar to every lane.
    fn splat(x: f64) -> Self;
    /// Lane-wise `self + o`.
    fn add(self, o: Self) -> Self;
    /// Lane-wise `self - o`.
    fn sub(self, o: Self) -> Self;
    /// Lane-wise `self * o`.
    fn mul(self, o: Self) -> Self;
    /// The `f64` in lane `i` (`i < LANES`; `f64` ignores `i`).
    fn lane(self, i: usize) -> f64;
    /// Build the order-≤2 derivative stack `[f(u), f′(u), f″(u)]` **per lane**
    /// from the lane value `u`, via the SAME scalar `stack` closure the
    /// per-row path runs (so the transcendental/rational stack is bit-identical
    /// to the scalar evaluation — only the subsequent tensor composition is
    /// vectorised).
    fn unary3(self, stack: impl Fn(f64) -> [f64; 3]) -> [Self; 3];
    /// Build the order-≤4 derivative stack `[f, f′, f″, f‴, f⁗]` **per lane**
    /// from the lane value `u`, via the SAME scalar `stack` closure the per-row
    /// path runs. The one-/two-seed scalars ([`OneSeedLane`] / [`TwoSeedLane`])
    /// need outer derivatives one / two orders beyond their order-2 base, so
    /// they build their composition stack through this five-entry variant. As
    /// with [`unary3`](Lane::unary3), only the transcendental/rational stack is
    /// evaluated per lane (bit-identically to the scalar path); the subsequent
    /// tensor composition is vectorised.
    fn unary5(self, stack: impl Fn(f64) -> [f64; 5]) -> [Self; 5];
}

impl Lane for f64 {
    #[inline]
    fn splat(x: f64) -> Self {
        x
    }
    #[inline]
    fn add(self, o: Self) -> Self {
        self + o
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        self - o
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        self * o
    }
    #[inline]
    fn lane(self, _: usize) -> f64 {
        self
    }
    #[inline]
    fn unary3(self, stack: impl Fn(f64) -> [f64; 3]) -> [Self; 3] {
        stack(self)
    }
    #[inline]
    fn unary5(self, stack: impl Fn(f64) -> [f64; 5]) -> [Self; 5] {
        stack(self)
    }
}

impl Lane for wide::f64x4 {
    #[inline]
    fn splat(x: f64) -> Self {
        wide::f64x4::splat(x)
    }
    #[inline]
    fn add(self, o: Self) -> Self {
        self + o
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        self - o
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        self * o
    }
    #[inline]
    fn lane(self, i: usize) -> f64 {
        self.to_array()[i]
    }
    #[inline]
    fn unary3(self, stack: impl Fn(f64) -> [f64; 3]) -> [Self; 3] {
        let a = self.to_array();
        let mut d0 = [0.0_f64; 4];
        let mut d1 = [0.0_f64; 4];
        let mut d2 = [0.0_f64; 4];
        for i in 0..4 {
            let s = stack(a[i]);
            d0[i] = s[0];
            d1[i] = s[1];
            d2[i] = s[2];
        }
        [
            wide::f64x4::new(d0),
            wide::f64x4::new(d1),
            wide::f64x4::new(d2),
        ]
    }
    #[inline]
    fn unary5(self, stack: impl Fn(f64) -> [f64; 5]) -> [Self; 5] {
        let a = self.to_array();
        let mut d = [[0.0_f64; 4]; 5];
        for i in 0..4 {
            let s = stack(a[i]);
            for (k, dk) in d.iter_mut().enumerate() {
                dk[i] = s[k];
            }
        }
        [
            wide::f64x4::new(d[0]),
            wide::f64x4::new(d[1]),
            wide::f64x4::new(d[2]),
            wide::f64x4::new(d[3]),
            wide::f64x4::new(d[4]),
        ]
    }
}

/// A lane-batched order-≤2 Taylor scalar: value / gradient / Hessian carried in
/// a SIMD field [`L: Lane`](Lane). With `L = f64x4` one instance carries FOUR
/// rows at once, so the row loop processes 4 rows per vector pass instead of one
/// per scalar pass.
///
/// The channel layout and every float op mirror [`crate::jet_tower::Tower2`]
/// term-for-term, so `Order2Lane<f64, K>` is `to_bits`-identical to the
/// production [`Order2<K>`] and `Order2Lane<f64x4, K>` lane `i` is
/// `to_bits`-identical to that (see the module note and `batch_tests`).
#[derive(Clone, Copy, Debug)]
pub struct Order2Lane<L: Lane, const K: usize> {
    /// Value channel `ℓ` (one entry per lane/row).
    pub v: L,
    /// Gradient channel `∂ℓ/∂p_a`.
    pub g: [L; K],
    /// Hessian channel `∂²ℓ/∂p_a∂p_b` (symmetric).
    pub h: [[L; K]; K],
}

/// The 4-rows-per-pass batched order-≤2 scalar (`wide::f64x4` lanes).
pub type Order2Batch<const K: usize> = Order2Lane<wide::f64x4, K>;

impl<L: Lane, const K: usize> Order2Lane<L, K> {
    /// A constant: value `c` in every channel-zero slot.
    #[inline]
    pub fn constant(c: L) -> Self {
        Order2Lane {
            v: c,
            g: [L::splat(0.0); K],
            h: [[L::splat(0.0); K]; K],
        }
    }

    /// The seeded variable `p_axis` at (per-lane) value `value`: unit first
    /// derivative in slot `axis`. With `L = f64x4`, `value` packs the four
    /// rows' values of primary `axis`.
    #[inline]
    pub fn variable(value: L, axis: usize) -> Self {
        let mut out = Self::constant(value);
        out.g[axis] = L::splat(1.0);
        out
    }

    /// Lane-wise `self + o` (mirrors `Tower2` Add: per-channel add).
    #[inline]
    pub fn add(&self, o: &Self) -> Self {
        let mut out = *self;
        out.v = self.v.add(o.v);
        for i in 0..K {
            out.g[i] = self.g[i].add(o.g[i]);
            for j in 0..K {
                out.h[i][j] = self.h[i][j].add(o.h[i][j]);
            }
        }
        out
    }

    /// Multiply every channel by the plain scalar `s` (mirrors `Tower2::scale`).
    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        let sl = L::splat(s);
        let mut out = *self;
        out.v = self.v.mul(sl);
        for i in 0..K {
            out.g[i] = self.g[i].mul(sl);
            for j in 0..K {
                out.h[i][j] = self.h[i][j].mul(sl);
            }
        }
        out
    }

    /// Lane-wise `self - o`, expressed as `self + o·(-1)` exactly as
    /// [`Order2::sub`] / `Tower4::sub` do, so signed-zero handling matches.
    #[inline]
    pub fn sub(&self, o: &Self) -> Self {
        self.add(&o.scale(-1.0))
    }

    /// Negate every channel (= `scale(-1.0)`, matching [`Order2::neg`]).
    #[inline]
    pub fn neg(&self) -> Self {
        self.scale(-1.0)
    }

    /// Exact order-≤2 Leibniz product, term-for-term identical to
    /// [`crate::jet_tower::Tower2::mul`] (same factor order, no `mul_add`).
    ///
    /// The Hessian channel is symmetric under `i ↔ j` (see
    /// [`crate::jet_tower::Tower2::mul`] for why the invariant always holds), so
    /// we compute the upper triangle `j ≥ i` and mirror it — `K(K+1)/2` lane
    /// entry-chains instead of `K²`. Because each lane entry is already a full
    /// SIMD op (no cross-`j` lane packing to lose), halving the entry count is a
    /// direct throughput win (~18 % on `Order2Lane<f64x4, 9>`, the survival batch
    /// kernel, and ~2× on the `f64` oracle). The upper triangle uses the EXACT
    /// term order of `Tower2::mul`, so `Order2Lane<f64>` stays `to_bits`-identical
    /// to `Order2` (= `Tower2`) and `Order2Lane<f64x4>` lane `i` stays
    /// `to_bits`-identical to that; the mirror makes the batch Hessian exactly
    /// symmetric, matching the scalar `Tower2::mul` (which mirrors identically).
    #[inline]
    pub fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        let mut out = Self::constant(a.v.mul(b.v));
        for i in 0..K {
            // a.v*b.g[i] + a.g[i]*b.v
            out.g[i] = a.v.mul(b.g[i]).add(a.g[i].mul(b.v));
        }
        for i in 0..K {
            for j in i..K {
                // a.v*b.h + a.g[i]*b.g[j] + a.g[j]*b.g[i] + a.h*b.v
                let hij =
                    a.v.mul(b.h[i][j])
                        .add(a.g[i].mul(b.g[j]))
                        .add(a.g[j].mul(b.g[i]))
                        .add(a.h[i][j].mul(b.v));
                out.h[i][j] = hij;
                out.h[j][i] = hij;
            }
        }
        out
    }

    /// Exact order-≤2 Faà di Bruno composition `f ∘ self`, given the per-lane
    /// derivative stack `d = [f(u), f′(u), f″(u)]`. Mirrors
    /// [`crate::jet_tower::Tower2::compose_unary`] term-for-term (`acc` starts at
    /// `0` then accumulates, so signed-zero collapses identically).
    #[inline]
    pub fn compose_unary(&self, d: [L; 3]) -> Self {
        let mut out = Self::constant(d[0]);
        for i in 0..K {
            let mut acc = L::splat(0.0);
            acc = acc.add(d[1].mul(self.g[i]));
            out.g[i] = acc;
        }
        for i in 0..K {
            for j in 0..K {
                let mut acc = L::splat(0.0);
                acc = acc.add(d[1].mul(self.h[i][j]));
                acc = acc.add(d[2].mul(self.g[i]).mul(self.g[j]));
                out.h[i][j] = acc;
            }
        }
        out
    }

    /// `e^self`, per-lane stack `[e, e, e]` (matches the [`JetScalar::exp`]
    /// default forwarded through `Order2`).
    #[inline]
    pub fn exp(&self) -> Self {
        let d = self.v.unary3(|u| {
            let e = u.exp();
            [e, e, e]
        });
        self.compose_unary(d)
    }

    /// `ln(self)`; caller guarantees positivity. Per-lane stack
    /// `[ln u, 1/u, -1/u²]` (matches [`JetScalar::ln`] truncated to order 2).
    #[inline]
    pub fn ln(&self) -> Self {
        let d = self.v.unary3(|u| {
            let r = 1.0 / u;
            [u.ln(), r, -r * r]
        });
        self.compose_unary(d)
    }

    /// `√self`; caller guarantees positivity. Per-lane stack
    /// `[s, 0.5/s, -0.25/(u·s)]` (matches [`JetScalar::sqrt`]).
    #[inline]
    pub fn sqrt(&self) -> Self {
        let d = self.v.unary3(|u| {
            let s = u.sqrt();
            [s, 0.5 / s, -0.25 / (u * s)]
        });
        self.compose_unary(d)
    }

    /// `1/self`. Per-lane stack `[r, -r², 2r³]` (matches [`JetScalar::recip`]).
    #[inline]
    pub fn recip(&self) -> Self {
        let d = self.v.unary3(|u| {
            let r = 1.0 / u;
            let r2 = r * r;
            [r, -r2, 2.0 * r2 * r]
        });
        self.compose_unary(d)
    }

    /// `self^a` for real `a`; caller guarantees a positive base. Per-lane
    /// falling-factorial stack (matches [`JetScalar::powf`]).
    #[inline]
    pub fn powf(&self, a: f64) -> Self {
        let d = self.v.unary3(|u| {
            [
                u.powf(a),
                a * u.powf(a - 1.0),
                a * (a - 1.0) * u.powf(a - 2.0),
            ]
        });
        self.compose_unary(d)
    }
}

impl<const K: usize> Order2Batch<K> {
    /// Extract lane `i`'s `(v, g, H)` as a production [`Order2<K>`] scalar.
    /// Lane `i` is `to_bits`-identical to evaluating the same program at
    /// [`Order2<K>`] on row `i` (see `batch_tests`).
    #[inline]
    #[must_use]
    pub fn lane(&self, i: usize) -> Order2<K> {
        let mut t = crate::jet_tower::Tower2::<K>::constant(self.v.lane(i));
        for a in 0..K {
            t.g[a] = self.g[a].lane(i);
            for b in 0..K {
                t.h[a][b] = self.h[a][b].lane(i);
            }
        }
        Order2(t)
    }
}

// ── Order1<K>: value / gradient only (doc §A.1, first-order prune) ──────

/// Truncated FIRST-order scalar: value `v` and gradient `g_a` only — NO Hessian.
///
/// This is [`Order2`] with the K×K Hessian channel deleted. Its value and
/// gradient are computed by the SAME order-≤1 truncation of the Leibniz / Faà
/// di Bruno rules that [`Order2`] uses for those two channels, with the float
/// operations applied in the identical order — so its `(v, g)` is BIT-IDENTICAL
/// to both [`Order2`]'s and a full [`crate::jet_tower::Tower4`]'s order-≤1
/// channels. Use it at a consumer that reads ONLY value + gradient (the SAE
/// β-border channel: the reconstruction is linear in β, so the Hessian-in-β
/// vanishes and the dense K×K Hessian product `Tower2::mul` would build is pure
/// discarded work). Order-≤1 value/gradient never read any input's Hessian, so
/// dropping that channel changes neither result nor float-op order — it only
/// removes the `K²` arithmetic that produced an unread tensor.
#[derive(Clone, Copy, Debug)]
pub struct Order1<const K: usize> {
    /// Value ℓ.
    pub v: f64,
    /// Gradient ∂ℓ/∂p_a.
    pub g: [f64; K],
}

impl<const K: usize> Order1<K> {
    /// Read the gradient channel `g_a = ∂ℓ/∂p_a`.
    #[inline]
    pub fn g(&self) -> [f64; K] {
        self.g
    }
}

impl<const K: usize> JetScalar<K> for Order1<K> {
    fn constant(c: f64) -> Self {
        // Order2::constant -> Tower2::constant: value c, all derivatives zero.
        Order1 { v: c, g: [0.0; K] }
    }
    fn variable(x: f64, axis: usize) -> Self {
        // Order2::variable -> Tower2::variable: unit first derivative in `axis`.
        let mut g = [0.0; K];
        g[axis] = 1.0;
        Order1 { v: x, g }
    }
}

impl<const K: usize> crate::nested_dual::JetField for Order1<K> {
    fn value(&self) -> f64 {
        self.v
    }
    fn add(&self, o: &Self) -> Self {
        // Tower2 Add: out.v += o.v; out.g[i] += o.g[i] (same float order).
        let mut g = self.g;
        for i in 0..K {
            g[i] += o.g[i];
        }
        Order1 { v: self.v + o.v, g }
    }
    fn sub(&self, o: &Self) -> Self {
        // Mirror Order2::sub == self + o.scale(-1.0) exactly: scale then add.
        self.add(&o.scale(-1.0))
    }
    fn mul(&self, o: &Self) -> Self {
        // Tower2::mul value/grad terms, identical float order:
        //   v = a.v*b.v;  g[i] = a.v*b.g[i] + a.g[i]*b.v.
        // (The Hessian loop `a.v*b.h + a.g*b.g + ... + a.h*b.v` is the discarded
        //  work this type exists to skip; it never feeds v or g.)
        let a = self;
        let b = o;
        let mut g = [0.0; K];
        for i in 0..K {
            g[i] = a.v * b.g[i] + a.g[i] * b.v;
        }
        Order1 { v: a.v * b.v, g }
    }
    fn neg(&self) -> Self {
        // Order2::neg == self.0.scale(-1.0).
        self.scale(-1.0)
    }
    fn scale(&self, s: f64) -> Self {
        // Tower2::scale: out.v *= s; out.g[i] *= s (same float order).
        let mut g = self.g;
        for i in 0..K {
            g[i] *= s;
        }
        Order1 { v: self.v * s, g }
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // Faà di Bruno truncated to order ≤ 1 (matches `faa_di_bruno` /
        // `Tower2::compose_unary` for the value and gradient channels):
        //   value channel (m=0): d[0].
        //   grad channel (positions=[i], single partition {{0}}): d[1]·g[i].
        // Order-≤1 reads only d[0], d[1]; trailing stack entries are unused.
        let mut g = [0.0; K];
        for i in 0..K {
            g[i] = d[1] * self.g[i];
        }
        Order1 { v: d[0], g }
    }
}

// ── OneSeed<K>: one-seed directional, contracted third (doc §A.2) ───────

/// One-seed directional scalar: an [`Order2`] base plus ONE nilpotent ε
/// (`ε² = 0`) whose coefficient is itself an [`Order2`].
///
/// A scalar is `s = base + ε·eps`. Arithmetic is the `ε² = 0` truncation of the
/// product (doc §A.2): the base parts multiply as ordinary [`Order2`] products,
/// and the ε-coefficient picks up `a.base·b.eps + a.eps·b.base`. Composition
/// pushes ε through one extra outer derivative.
///
/// Seed each primary with [`seed_direction`](Self::seed_direction): the base is
/// the usual seeded variable (carrying `e_a` for the Hessian channel) and the
/// ε-coefficient is the FIXED contraction direction `u_a` (a constant). Then the
/// ε-component of the evaluated Hessian channel is the contracted third
/// `[eps.h][a][b] = Σ_c ℓ_{abc} u_c` — exactly `row_third_contracted(dir = u)`,
/// without materialising `t3`.
#[derive(Clone, Copy, Debug)]
pub struct OneSeed<const K: usize> {
    /// The `ε⁰` part: value / gradient / Hessian of `ℓ`.
    pub base: Order2<K>,
    /// The `ε¹` part: value / gradient / Hessian of the ε-coefficient. After a
    /// `seed_direction(u)` evaluation, `eps.h[a][b] = Σ_c ℓ_{abc} u_c`.
    pub eps: Order2<K>,
}

impl<const K: usize> OneSeed<K> {
    /// Seed primary `axis` at value `x` with ε-direction component `u_axis`:
    /// `p_axis = p_axis⁰ + x-seed + ε·u_axis`, i.e. base = `variable(x, axis)`
    /// and eps = `constant(u_axis)` (doc §A.2 "Seeding").
    pub fn seed_direction(x: f64, axis: usize, u_axis: f64) -> Self {
        OneSeed {
            base: Order2::variable(x, axis),
            eps: Order2::constant(u_axis),
        }
    }

    /// The contracted-third channel after a `seed_direction(u)` evaluation:
    /// `out[a][b] = Σ_c ℓ_{abc} u_c`, i.e. the ε-coefficient's Hessian (doc §A.2).
    pub fn contracted_third(&self) -> [[f64; K]; K] {
        self.eps.h()
    }
}

impl<const K: usize> JetScalar<K> for OneSeed<K> {
    fn constant(c: f64) -> Self {
        OneSeed {
            base: Order2::constant(c),
            eps: Order2::constant(0.0),
        }
    }
    fn variable(x: f64, axis: usize) -> Self {
        // No ε-direction unless seeded via `seed_direction`.
        OneSeed {
            base: Order2::variable(x, axis),
            eps: Order2::constant(0.0),
        }
    }
}

impl<const K: usize> crate::nested_dual::JetField for OneSeed<K> {
    fn value(&self) -> f64 {
        self.base.value()
    }
    fn add(&self, o: &Self) -> Self {
        OneSeed {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
        }
    }
    fn sub(&self, o: &Self) -> Self {
        OneSeed {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
        }
    }
    fn mul(&self, o: &Self) -> Self {
        // (a.base + ε a.eps)(b.base + ε b.eps), dropping ε².  Build
        // the epsilon channels directly instead of materialising two complete
        // Order2 products and adding them.  The upper triangle is sufficient:
        // every input Hessian is symmetric and this polynomial preserves that
        // invariant exactly.
        let ab = &self.base.0;
        let ae = &self.eps.0;
        let bb = &o.base.0;
        let be = &o.eps.0;
        let mut eps = crate::jet_tower::Tower2::<K>::zero();
        eps.v = ab.v * be.v + ae.v * bb.v;
        for i in 0..K {
            eps.g[i] = ab.v * be.g[i] + ab.g[i] * be.v + ae.v * bb.g[i] + ae.g[i] * bb.v;
        }
        for i in 0..K {
            for j in i..K {
                let channel = ab.v * be.h[i][j]
                    + ab.g[i] * be.g[j]
                    + ab.g[j] * be.g[i]
                    + ab.h[i][j] * be.v
                    + ae.v * bb.h[i][j]
                    + ae.g[i] * bb.g[j]
                    + ae.g[j] * bb.g[i]
                    + ae.h[i][j] * bb.v;
                eps.h[i][j] = channel;
                eps.h[j][i] = channel;
            }
        }
        OneSeed {
            base: self.base.mul(&o.base),
            eps: Order2(eps),
        }
    }
    fn neg(&self) -> Self {
        OneSeed {
            base: self.base.neg(),
            eps: self.eps.neg(),
        }
    }
    fn scale(&self, s: f64) -> Self {
        OneSeed {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
        }
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // f(base + ε eps) = f(base) + ε · f'(base)·eps (ε² = 0).
        // Fuse the shifted composition and product into their final channels;
        // this removes one temporary Order2 composition and one Order2 product
        // from every unary primitive in a directional row program.
        let base = self.base.compose_unary([d[0], d[1], d[2], d[3], d[4]]);
        let b = &self.base.0;
        let e = &self.eps.0;
        let mut eps = crate::jet_tower::Tower2::<K>::zero();
        eps.v = d[1] * e.v;
        for i in 0..K {
            eps.g[i] = d[2] * b.g[i] * e.v + d[1] * e.g[i];
        }
        for i in 0..K {
            for j in i..K {
                let channel = d[1] * e.h[i][j]
                    + d[2] * (b.g[i] * e.g[j] + b.g[j] * e.g[i] + b.h[i][j] * e.v)
                    + d[3] * b.g[i] * b.g[j] * e.v;
                eps.h[i][j] = channel;
                eps.h[j][i] = channel;
            }
        }
        OneSeed {
            base,
            eps: Order2(eps),
        }
    }
}

// ── OneSeedLane<L, K>: lane-batched one-seed directional (doc §A.2) ──────

/// Lane-batched [`OneSeed`]: the same one-seed directional scalar with its two
/// [`Order2`] parts re-typed to [`Order2Lane<L, K>`], so one `L = f64x4`
/// instance carries FOUR rows' contracted-third evaluations per vector pass.
///
/// Every operation (`add`/`sub`/`mul`/`neg`/`scale`/`compose_unary` and the
/// transcendentals) is a term-for-term structural re-type of the scalar
/// [`OneSeed`] ops onto the lane-implemented [`Order2Lane`] algebra. With
/// `L = f64`, `OneSeedLane<f64, K>` is `to_bits`-identical to [`OneSeed<K>`];
/// with `L = f64x4`, lane `i` is `to_bits`-identical to that (see `batch_tests`).
#[derive(Clone, Copy, Debug)]
pub struct OneSeedLane<L: Lane, const K: usize> {
    /// The `ε⁰` part (lane-batched value / gradient / Hessian of `ℓ`).
    pub base: Order2Lane<L, K>,
    /// The `ε¹` part. After a `seed_direction(u)` evaluation,
    /// `eps.h[a][b]` lane `i` is row `i`'s `Σ_c ℓ_{abc} u_c`.
    pub eps: Order2Lane<L, K>,
}

/// The 4-rows-per-pass batched one-seed scalar (`wide::f64x4` lanes).
pub type OneSeedBatch<const K: usize> = OneSeedLane<wide::f64x4, K>;

impl<L: Lane, const K: usize> OneSeedLane<L, K> {
    /// A constant: base = `constant(c)`, ε-part zero (mirrors [`OneSeed::constant`]).
    #[inline]
    pub fn constant(c: L) -> Self {
        OneSeedLane {
            base: Order2Lane::constant(c),
            eps: Order2Lane::constant(L::splat(0.0)),
        }
    }

    /// The seeded variable `p_axis` at (per-lane) value `value`, no ε-direction
    /// (mirrors [`OneSeed::variable`]).
    #[inline]
    pub fn variable(value: L, axis: usize) -> Self {
        OneSeedLane {
            base: Order2Lane::variable(value, axis),
            eps: Order2Lane::constant(L::splat(0.0)),
        }
    }

    /// Seed primary `axis` at (per-lane) value `value` with ε-direction
    /// component `u_axis`: base = `variable(value, axis)`, eps = `constant(u_axis)`
    /// (mirrors [`OneSeed::seed_direction`]). With `L = f64x4`, `value` / `u_axis`
    /// pack the four rows' values / directions of primary `axis`.
    #[inline]
    pub fn seed_direction(value: L, axis: usize, u_axis: L) -> Self {
        OneSeedLane {
            base: Order2Lane::variable(value, axis),
            eps: Order2Lane::constant(u_axis),
        }
    }

    /// The contracted-third channel after a `seed_direction(u)` evaluation:
    /// `out[a][b]` lane `i` is row `i`'s `Σ_c ℓ_{abc} u_c` (the ε-part Hessian).
    #[inline]
    #[must_use]
    pub fn contracted_third(&self) -> [[L; K]; K] {
        self.eps.h
    }

    /// Lane-wise `self + o` (mirrors [`OneSeed::add`]).
    #[inline]
    pub fn add(&self, o: &Self) -> Self {
        OneSeedLane {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
        }
    }

    /// Lane-wise `self - o` (mirrors [`OneSeed::sub`]).
    #[inline]
    pub fn sub(&self, o: &Self) -> Self {
        OneSeedLane {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
        }
    }

    /// Lane-wise `self · o`, ε² = 0 truncation (mirrors [`OneSeed::mul`]).
    #[inline]
    pub fn mul(&self, o: &Self) -> Self {
        let ab = &self.base;
        let ae = &self.eps;
        let bb = &o.base;
        let be = &o.eps;
        let mut eps = Order2Lane::constant(ab.v.mul(be.v).add(ae.v.mul(bb.v)));
        for i in 0..K {
            eps.g[i] =
                ab.v.mul(be.g[i])
                    .add(ab.g[i].mul(be.v))
                    .add(ae.v.mul(bb.g[i]))
                    .add(ae.g[i].mul(bb.v));
        }
        for i in 0..K {
            for j in i..K {
                let channel =
                    ab.v.mul(be.h[i][j])
                        .add(ab.g[i].mul(be.g[j]))
                        .add(ab.g[j].mul(be.g[i]))
                        .add(ab.h[i][j].mul(be.v))
                        .add(ae.v.mul(bb.h[i][j]))
                        .add(ae.g[i].mul(bb.g[j]))
                        .add(ae.g[j].mul(bb.g[i]))
                        .add(ae.h[i][j].mul(bb.v));
                eps.h[i][j] = channel;
                eps.h[j][i] = channel;
            }
        }
        OneSeedLane {
            base: self.base.mul(&o.base),
            eps,
        }
    }

    /// Negate every part (mirrors [`OneSeed::neg`]).
    #[inline]
    pub fn neg(&self) -> Self {
        OneSeedLane {
            base: self.base.neg(),
            eps: self.eps.neg(),
        }
    }

    /// Multiply every part by the plain scalar `s` (mirrors [`OneSeed::scale`]).
    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        OneSeedLane {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
        }
    }

    /// Exact order-≤2-per-part Faà di Bruno composition `f ∘ self`, given the
    /// per-lane outer-derivative stack `d = [f, f′, f″, f‴, f⁗]`. Term-for-term
    /// identical to the fused [`OneSeed::compose_unary`] channels: the base reads
    /// `d[0..=2]`, while the ε-coefficient directly evaluates
    /// `f′(base)·eps` from `d[1..=3]` without an intermediate jet.
    #[inline]
    pub fn compose_unary(&self, d: [L; 5]) -> Self {
        let base = self.base.compose_unary([d[0], d[1], d[2]]);
        let b = &self.base;
        let e = &self.eps;
        let mut eps = Order2Lane::constant(d[1].mul(e.v));
        for i in 0..K {
            eps.g[i] = d[2].mul(b.g[i]).mul(e.v).add(d[1].mul(e.g[i]));
        }
        for i in 0..K {
            for j in i..K {
                let mixed = b.g[i]
                    .mul(e.g[j])
                    .add(b.g[j].mul(e.g[i]))
                    .add(b.h[i][j].mul(e.v));
                let channel = d[1]
                    .mul(e.h[i][j])
                    .add(d[2].mul(mixed))
                    .add(d[3].mul(b.g[i]).mul(b.g[j]).mul(e.v));
                eps.h[i][j] = channel;
                eps.h[j][i] = channel;
            }
        }
        OneSeedLane { base, eps }
    }

    /// `e^self`, per-lane stack `[e, e, e, e, e]` (matches [`JetScalar::exp`]).
    #[inline]
    pub fn exp(&self) -> Self {
        let d = self.base.v.unary5(|u| {
            let e = u.exp();
            [e, e, e, e, e]
        });
        self.compose_unary(d)
    }

    /// `ln(self)`; caller guarantees positivity (matches [`JetScalar::ln`]).
    #[inline]
    pub fn ln(&self) -> Self {
        let d = self.base.v.unary5(|u| {
            let r = 1.0 / u;
            [u.ln(), r, -r * r, 2.0 * r * r * r, -6.0 * r * r * r * r]
        });
        self.compose_unary(d)
    }

    /// `√self`; caller guarantees positivity (matches [`JetScalar::sqrt`]).
    #[inline]
    pub fn sqrt(&self) -> Self {
        let d = self.base.v.unary5(|u| {
            let s = u.sqrt();
            [
                s,
                0.5 / s,
                -0.25 / (u * s),
                0.375 / (u * u * s),
                -0.9375 / (u * u * u * s),
            ]
        });
        self.compose_unary(d)
    }

    /// `1/self` (matches [`JetScalar::recip`]).
    #[inline]
    pub fn recip(&self) -> Self {
        let d = self.base.v.unary5(|u| {
            let r = 1.0 / u;
            let r2 = r * r;
            [r, -r2, 2.0 * r2 * r, -6.0 * r2 * r2, 24.0 * r2 * r2 * r]
        });
        self.compose_unary(d)
    }

    /// `self^a` for real `a`; caller guarantees a positive base (matches
    /// [`JetScalar::powf`]).
    #[inline]
    pub fn powf(&self, a: f64) -> Self {
        let d = self.base.v.unary5(|u| {
            [
                u.powf(a),
                a * u.powf(a - 1.0),
                a * (a - 1.0) * u.powf(a - 2.0),
                a * (a - 1.0) * (a - 2.0) * u.powf(a - 3.0),
                a * (a - 1.0) * (a - 2.0) * (a - 3.0) * u.powf(a - 4.0),
            ]
        });
        self.compose_unary(d)
    }

    /// `ln Γ(self)`; caller guarantees positivity (matches [`JetScalar::ln_gamma`],
    /// same hand-certified stack).
    #[inline]
    pub fn ln_gamma(&self) -> Self {
        let d = self
            .base
            .v
            .unary5(crate::jet_tower::ln_gamma_derivative_stack);
        self.compose_unary(d)
    }

    /// `ψ(self)` digamma; caller guarantees positivity (matches
    /// [`JetScalar::digamma`], same hand-certified stack).
    #[inline]
    pub fn digamma(&self) -> Self {
        let d = self
            .base
            .v
            .unary5(crate::jet_tower::digamma_derivative_stack);
        self.compose_unary(d)
    }
}

impl<const K: usize> OneSeedBatch<K> {
    /// Extract lane `i`'s parts as a production [`OneSeed<K>`]. Lane `i` is
    /// `to_bits`-identical to evaluating the same program at [`OneSeed<K>`] on
    /// row `i` (see `batch_tests`).
    #[inline]
    #[must_use]
    pub fn lane(&self, i: usize) -> OneSeed<K> {
        OneSeed {
            base: self.base.lane(i),
            eps: self.eps.lane(i),
        }
    }
}

// ── TwoSeed<K>: two-seed, contracted fourth (doc §A.3) ──────────────────

/// Two-seed scalar: an [`Order2`] base plus TWO nilpotents ε, δ
/// (`ε² = δ² = 0`, `εδ` retained) — four [`Order2`] parts
/// `s = base + ε·eps + δ·del + εδ·eps_del`.
///
/// Product truncates `ε² = δ² = 0` (doc §A.3): each part is built from
/// [`Order2`] products of the four input parts. Composition picks up
/// successively higher outer derivatives, the cross part carrying the second
/// Faà di Bruno term `f''·eps·del + f'·eps_del`.
///
/// Seed each primary with [`seed`](Self::seed): base = `variable(x, axis)`,
/// eps = `constant(u_axis)`, del = `constant(v_axis)`, eps_del = `constant(0)`.
/// Then the εδ-component of the evaluated Hessian channel is the contracted
/// fourth `[eps_del.h][a][b] = Σ_{cd} ℓ_{abcd} u_c v_d` — exactly
/// `row_fourth_contracted(u, v)`, without materialising `t4`.
#[derive(Clone, Copy, Debug)]
pub struct TwoSeed<const K: usize> {
    /// The `ε⁰δ⁰` part: value / grad / Hessian of `ℓ`.
    pub base: Order2<K>,
    /// The `ε¹δ⁰` part.
    pub eps: Order2<K>,
    /// The `ε⁰δ¹` part.
    pub del: Order2<K>,
    /// The `ε¹δ¹` part. After a `seed(u, v)` evaluation,
    /// `eps_del.h[a][b] = Σ_{cd} ℓ_{abcd} u_c v_d`.
    pub eps_del: Order2<K>,
}

impl<const K: usize> TwoSeed<K> {
    /// Seed primary `axis` at value `x` with ε-direction `u_axis` and
    /// δ-direction `v_axis`:
    /// `p_axis = p_axis⁰ + x-seed + ε·u_axis + δ·v_axis` (doc §A.3 "Seeding").
    pub fn seed(x: f64, axis: usize, u_axis: f64, v_axis: f64) -> Self {
        TwoSeed {
            base: Order2::variable(x, axis),
            eps: Order2::constant(u_axis),
            del: Order2::constant(v_axis),
            eps_del: Order2::constant(0.0),
        }
    }

    /// The contracted-fourth channel after a `seed(u, v)` evaluation:
    /// `out[a][b] = Σ_{cd} ℓ_{abcd} u_c v_d`, i.e. the εδ-coefficient's Hessian.
    pub fn contracted_fourth(&self) -> [[f64; K]; K] {
        self.eps_del.h()
    }
}

impl<const K: usize> JetScalar<K> for TwoSeed<K> {
    fn constant(c: f64) -> Self {
        TwoSeed {
            base: Order2::constant(c),
            eps: Order2::constant(0.0),
            del: Order2::constant(0.0),
            eps_del: Order2::constant(0.0),
        }
    }
    fn variable(x: f64, axis: usize) -> Self {
        TwoSeed {
            base: Order2::variable(x, axis),
            eps: Order2::constant(0.0),
            del: Order2::constant(0.0),
            eps_del: Order2::constant(0.0),
        }
    }
}

impl<const K: usize> crate::nested_dual::JetField for TwoSeed<K> {
    fn value(&self) -> f64 {
        self.base.value()
    }
    fn add(&self, o: &Self) -> Self {
        TwoSeed {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
            del: self.del.add(&o.del),
            eps_del: self.eps_del.add(&o.eps_del),
        }
    }
    fn sub(&self, o: &Self) -> Self {
        TwoSeed {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
            del: self.del.sub(&o.del),
            eps_del: self.eps_del.sub(&o.eps_del),
        }
    }
    fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        // Truncate ε² = δ² = 0 (doc §A.3 product table).
        let base = a.base.mul(&b.base);
        let eps = a.base.mul(&b.eps).add(&a.eps.mul(&b.base));
        let del = a.base.mul(&b.del).add(&a.del.mul(&b.base));
        let eps_del = a
            .base
            .mul(&b.eps_del)
            .add(&a.eps.mul(&b.del))
            .add(&a.del.mul(&b.eps))
            .add(&a.eps_del.mul(&b.base));
        TwoSeed {
            base,
            eps,
            del,
            eps_del,
        }
    }
    fn neg(&self) -> Self {
        TwoSeed {
            base: self.base.neg(),
            eps: self.eps.neg(),
            del: self.del.neg(),
            eps_del: self.eps_del.neg(),
        }
    }
    fn scale(&self, s: f64) -> Self {
        TwoSeed {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
            del: self.del.scale(s),
            eps_del: self.eps_del.scale(s),
        }
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // f(s) with s = base + ε eps + δ del + εδ eps_del, ε²=δ²=0:
        //   f(s) = f(base)
        //        + ε · f'(base)·eps
        //        + δ · f'(base)·del
        //        + εδ · ( f''(base)·eps·del + f'(base)·eps_del ).
        // Each f^{(r)}(base) is the Order2 composition of base with the stack
        // shifted r entries (doc §A.3 composition). Order2 reads only the
        // leading three entries of whatever stack it is handed, so the trailing
        // padding slots are unused (the fixed-length array makes this total).
        let base = self.base.compose_unary([d[0], d[1], d[2], d[3], d[4]]);
        let fprime = self.base.compose_unary([d[1], d[2], d[3], d[4], d[4]]); // f'(base) as Order2
        let fsecond = self.base.compose_unary([d[2], d[3], d[4], d[4], d[4]]); // f''(base) as Order2
        let eps = fprime.mul(&self.eps);
        let del = fprime.mul(&self.del);
        let eps_del = fsecond
            .mul(&self.eps)
            .mul(&self.del)
            .add(&fprime.mul(&self.eps_del));
        TwoSeed {
            base,
            eps,
            del,
            eps_del,
        }
    }
}

// ── TwoSeedLane<L, K>: lane-batched two-seed, contracted fourth (doc §A.3) ─

/// Lane-batched [`TwoSeed`]: the same two-seed scalar with its four [`Order2`]
/// parts re-typed to [`Order2Lane<L, K>`], so one `L = f64x4` instance carries
/// FOUR rows' contracted-fourth evaluations per vector pass.
///
/// Every operation is a term-for-term structural re-type of the scalar
/// [`TwoSeed`] ops onto the lane-implemented [`Order2Lane`] algebra. With
/// `L = f64`, `TwoSeedLane<f64, K>` is `to_bits`-identical to [`TwoSeed<K>`];
/// with `L = f64x4`, lane `i` is `to_bits`-identical to that (see `batch_tests`).
#[derive(Clone, Copy, Debug)]
pub struct TwoSeedLane<L: Lane, const K: usize> {
    /// The `ε⁰δ⁰` part.
    pub base: Order2Lane<L, K>,
    /// The `ε¹δ⁰` part.
    pub eps: Order2Lane<L, K>,
    /// The `ε⁰δ¹` part.
    pub del: Order2Lane<L, K>,
    /// The `ε¹δ¹` part. After a `seed(u, v)` evaluation, `eps_del.h[a][b]`
    /// lane `i` is row `i`'s `Σ_{cd} ℓ_{abcd} u_c v_d`.
    pub eps_del: Order2Lane<L, K>,
}

/// The 4-rows-per-pass batched two-seed scalar (`wide::f64x4` lanes).
pub type TwoSeedBatch<const K: usize> = TwoSeedLane<wide::f64x4, K>;

impl<L: Lane, const K: usize> TwoSeedLane<L, K> {
    /// A constant: base = `constant(c)`, all seed parts zero (mirrors
    /// [`TwoSeed::constant`]).
    #[inline]
    pub fn constant(c: L) -> Self {
        let z = Order2Lane::constant(L::splat(0.0));
        TwoSeedLane {
            base: Order2Lane::constant(c),
            eps: z,
            del: z,
            eps_del: z,
        }
    }

    /// The seeded variable `p_axis` at (per-lane) value `value`, no ε/δ direction
    /// (mirrors [`TwoSeed::variable`]).
    #[inline]
    pub fn variable(value: L, axis: usize) -> Self {
        let z = Order2Lane::constant(L::splat(0.0));
        TwoSeedLane {
            base: Order2Lane::variable(value, axis),
            eps: z,
            del: z,
            eps_del: z,
        }
    }

    /// Seed primary `axis` at (per-lane) value `value` with ε-direction `u_axis`
    /// and δ-direction `v_axis` (mirrors [`TwoSeed::seed`]). With `L = f64x4`,
    /// each argument packs the four rows' values for primary `axis`.
    #[inline]
    pub fn seed(value: L, axis: usize, u_axis: L, v_axis: L) -> Self {
        TwoSeedLane {
            base: Order2Lane::variable(value, axis),
            eps: Order2Lane::constant(u_axis),
            del: Order2Lane::constant(v_axis),
            eps_del: Order2Lane::constant(L::splat(0.0)),
        }
    }

    /// The contracted-fourth channel after a `seed(u, v)` evaluation:
    /// `out[a][b]` lane `i` is row `i`'s `Σ_{cd} ℓ_{abcd} u_c v_d`
    /// (the εδ-part Hessian).
    #[inline]
    #[must_use]
    pub fn contracted_fourth(&self) -> [[L; K]; K] {
        self.eps_del.h
    }

    /// Lane-wise `self + o` (mirrors [`TwoSeed::add`]).
    #[inline]
    pub fn add(&self, o: &Self) -> Self {
        TwoSeedLane {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
            del: self.del.add(&o.del),
            eps_del: self.eps_del.add(&o.eps_del),
        }
    }

    /// Lane-wise `self - o` (mirrors [`TwoSeed::sub`]).
    #[inline]
    pub fn sub(&self, o: &Self) -> Self {
        TwoSeedLane {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
            del: self.del.sub(&o.del),
            eps_del: self.eps_del.sub(&o.eps_del),
        }
    }

    /// Lane-wise `self · o`, ε² = δ² = 0 truncation (mirrors [`TwoSeed::mul`]).
    #[inline]
    pub fn mul(&self, o: &Self) -> Self {
        let a = self;
        let b = o;
        let base = a.base.mul(&b.base);
        let eps = a.base.mul(&b.eps).add(&a.eps.mul(&b.base));
        let del = a.base.mul(&b.del).add(&a.del.mul(&b.base));
        let eps_del = a
            .base
            .mul(&b.eps_del)
            .add(&a.eps.mul(&b.del))
            .add(&a.del.mul(&b.eps))
            .add(&a.eps_del.mul(&b.base));
        TwoSeedLane {
            base,
            eps,
            del,
            eps_del,
        }
    }

    /// Negate every part (mirrors [`TwoSeed::neg`]).
    #[inline]
    pub fn neg(&self) -> Self {
        TwoSeedLane {
            base: self.base.neg(),
            eps: self.eps.neg(),
            del: self.del.neg(),
            eps_del: self.eps_del.neg(),
        }
    }

    /// Multiply every part by the plain scalar `s` (mirrors [`TwoSeed::scale`]).
    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        TwoSeedLane {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
            del: self.del.scale(s),
            eps_del: self.eps_del.scale(s),
        }
    }

    /// Exact composition `f ∘ self`, given the per-lane outer-derivative stack
    /// `d = [f, f′, f″, f‴, f⁗]`. Term-for-term identical to
    /// [`TwoSeed::compose_unary`]: base reads `d[0..=2]`, `f′(base)` reads
    /// `d[1..=3]`, `f″(base)` reads `d[2..=4]`, and the cross part carries
    /// `f″·eps·del + f′·eps_del`.
    #[inline]
    pub fn compose_unary(&self, d: [L; 5]) -> Self {
        let base = self.base.compose_unary([d[0], d[1], d[2]]);
        let fprime = self.base.compose_unary([d[1], d[2], d[3]]);
        let fsecond = self.base.compose_unary([d[2], d[3], d[4]]);
        let eps = fprime.mul(&self.eps);
        let del = fprime.mul(&self.del);
        let eps_del = fsecond
            .mul(&self.eps)
            .mul(&self.del)
            .add(&fprime.mul(&self.eps_del));
        TwoSeedLane {
            base,
            eps,
            del,
            eps_del,
        }
    }

    /// `e^self`, per-lane stack `[e; 5]` (matches [`JetScalar::exp`]).
    #[inline]
    pub fn exp(&self) -> Self {
        let d = self.base.v.unary5(|u| {
            let e = u.exp();
            [e, e, e, e, e]
        });
        self.compose_unary(d)
    }

    /// `ln(self)`; caller guarantees positivity (matches [`JetScalar::ln`]).
    #[inline]
    pub fn ln(&self) -> Self {
        let d = self.base.v.unary5(|u| {
            let r = 1.0 / u;
            [u.ln(), r, -r * r, 2.0 * r * r * r, -6.0 * r * r * r * r]
        });
        self.compose_unary(d)
    }

    /// `√self`; caller guarantees positivity (matches [`JetScalar::sqrt`]).
    #[inline]
    pub fn sqrt(&self) -> Self {
        let d = self.base.v.unary5(|u| {
            let s = u.sqrt();
            [
                s,
                0.5 / s,
                -0.25 / (u * s),
                0.375 / (u * u * s),
                -0.9375 / (u * u * u * s),
            ]
        });
        self.compose_unary(d)
    }

    /// `1/self` (matches [`JetScalar::recip`]).
    #[inline]
    pub fn recip(&self) -> Self {
        let d = self.base.v.unary5(|u| {
            let r = 1.0 / u;
            let r2 = r * r;
            [r, -r2, 2.0 * r2 * r, -6.0 * r2 * r2, 24.0 * r2 * r2 * r]
        });
        self.compose_unary(d)
    }

    /// `self^a` for real `a`; caller guarantees a positive base (matches
    /// [`JetScalar::powf`]).
    #[inline]
    pub fn powf(&self, a: f64) -> Self {
        let d = self.base.v.unary5(|u| {
            [
                u.powf(a),
                a * u.powf(a - 1.0),
                a * (a - 1.0) * u.powf(a - 2.0),
                a * (a - 1.0) * (a - 2.0) * u.powf(a - 3.0),
                a * (a - 1.0) * (a - 2.0) * (a - 3.0) * u.powf(a - 4.0),
            ]
        });
        self.compose_unary(d)
    }

    /// `ln Γ(self)`; caller guarantees positivity (matches [`JetScalar::ln_gamma`]).
    #[inline]
    pub fn ln_gamma(&self) -> Self {
        let d = self
            .base
            .v
            .unary5(crate::jet_tower::ln_gamma_derivative_stack);
        self.compose_unary(d)
    }

    /// `ψ(self)` digamma; caller guarantees positivity (matches
    /// [`JetScalar::digamma`]).
    #[inline]
    pub fn digamma(&self) -> Self {
        let d = self
            .base
            .v
            .unary5(crate::jet_tower::digamma_derivative_stack);
        self.compose_unary(d)
    }
}

impl<const K: usize> TwoSeedBatch<K> {
    /// Extract lane `i`'s parts as a production [`TwoSeed<K>`]. Lane `i` is
    /// `to_bits`-identical to evaluating the same program at [`TwoSeed<K>`] on
    /// row `i` (see `batch_tests`).
    #[inline]
    #[must_use]
    pub fn lane(&self, i: usize) -> TwoSeed<K> {
        TwoSeed {
            base: self.base.lane(i),
            eps: self.eps.lane(i),
            del: self.del.lane(i),
            eps_del: self.eps_del.lane(i),
        }
    }
}

// ── Tower3<K>: value / gradient / Hessian / third tensor ────────────────

/// The order-≤3 [`crate::jet_tower::Tower3`] is also a [`JetScalar`]. It serves
/// consumers that read `.t3` but never `.t4`, avoiding the fourth-tensor
/// product/composition work while preserving the lower channels
/// bit-for-bit against [`crate::jet_tower::Tower4`].
impl<const K: usize> JetScalar<K> for crate::jet_tower::Tower3<K> {
    fn constant(c: f64) -> Self {
        crate::jet_tower::Tower3::constant(c)
    }
    fn variable(x: f64, axis: usize) -> Self {
        crate::jet_tower::Tower3::variable(x, axis)
    }
}

impl<const K: usize> crate::nested_dual::JetField for crate::jet_tower::Tower3<K> {
    fn value(&self) -> f64 {
        self.v
    }
    fn add(&self, o: &Self) -> Self {
        *self + *o
    }
    fn sub(&self, o: &Self) -> Self {
        *self + o.scale(-1.0)
    }
    fn mul(&self, o: &Self) -> Self {
        crate::jet_tower::Tower3::mul(self, o)
    }
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }
    fn scale(&self, s: f64) -> Self {
        crate::jet_tower::Tower3::scale(self, s)
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        crate::jet_tower::Tower3::compose_unary(self, [d[0], d[1], d[2], d[3]])
    }
}

// ── Tower4<K>: full dense tower as a JetScalar (the all-channels scalar) ─

/// The full dense [`crate::jet_tower::Tower4`] is itself a [`JetScalar`]: it
/// carries EVERY channel, so a row expression written ONCE against [`JetScalar`]
/// can be evaluated at `Tower4` to obtain the full `(v, g, H, t3, t4)` in one
/// pass. This is BOTH the #932 oracle ground truth the packed [`Order2`] /
/// [`OneSeed`] / [`TwoSeed`] scalars are pinned against, AND a production scalar:
/// a family whose uncontracted third / fourth derivative tensors are needed
/// (the BMS rigid `third_full` / `fourth_full` caches) evaluates the SAME
/// generic row-NLL expression at `Tower4` and reads `.t3` / `.t4` off the
/// result — so the dense tensors come from the single source of truth, not a
/// separately hand-written jet. The packed scalars serve the consumers that
/// need only `(v, g, H)` (`Order2`) or one / two contractions
/// (`OneSeed` / `TwoSeed`) without paying for the dense tensors.
impl<const K: usize> JetScalar<K> for crate::jet_tower::Tower4<K> {
    fn constant(c: f64) -> Self {
        crate::jet_tower::Tower4::constant(c)
    }
    fn variable(x: f64, axis: usize) -> Self {
        crate::jet_tower::Tower4::variable(x, axis)
    }
}

impl<const K: usize> crate::nested_dual::JetField for crate::jet_tower::Tower4<K> {
    fn value(&self) -> f64 {
        self.v
    }
    fn add(&self, o: &Self) -> Self {
        *self + *o
    }
    fn sub(&self, o: &Self) -> Self {
        *self - *o
    }
    fn mul(&self, o: &Self) -> Self {
        crate::jet_tower::Tower4::mul(self, o)
    }
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }
    fn scale(&self, s: f64) -> Self {
        crate::jet_tower::Tower4::scale(self, s)
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        crate::jet_tower::Tower4::compose_unary(self, d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jet_tower::{RowProgram, Tower4, program_full_tower};
    use crate::nested_dual::JetField;

    struct DenseSymmetric3([[f64; 3]; 3]);

    impl SymmetricQuadraticCoefficients for DenseSymmetric3 {
        fn dimension(&self) -> usize {
            3
        }

        fn multiply(&self, input: &[f64], output: &mut [f64]) {
            assert_eq!(input.len(), 3);
            assert_eq!(output.len(), 3);
            for (row, output) in output.iter_mut().enumerate() {
                *output = (0..3)
                    .map(|column| self.0[row][column] * input[column])
                    .sum();
            }
        }

        fn coefficient(&self, row: usize, column: usize) -> f64 {
            self.0[row][column]
        }
    }

    #[test]
    fn symmetric_quadratic_order2_lowerings_match_scalar_program() {
        const K: usize = 4;
        let coefficients = DenseSymmetric3([[1.2, 0.3, -0.2], [0.3, 0.8, 0.15], [-0.2, 0.15, 1.5]]);
        let values = [0.4, -0.7, 1.1, 0.25];
        let fixed_vars: [Order2<K>; K] =
            std::array::from_fn(|axis| Order2::variable(values[axis], axis));
        let fixed_inputs = [
            fixed_vars[0].mul(&fixed_vars[1]).add(&fixed_vars[3]),
            fixed_vars[1].exp().add(&fixed_vars[2].scale(0.4)),
            fixed_vars[2].mul(&fixed_vars[2]).sub(&fixed_vars[0]),
        ];
        let fixed_direct = Order2::symmetric_quadratic_form(&fixed_inputs, &coefficients);
        let fixed_scalar = symmetric_quadratic_form_default(
            &fixed_inputs,
            &coefficients,
            Order2::constant,
            JetField::add,
            JetField::mul,
            JetField::scale,
        );
        let weights = [0.7, -1.1, 0.35];
        let fixed_linear_direct = Order2::linear_combination(&fixed_inputs, &weights);
        let fixed_linear_scalar = linear_combination_default(
            &fixed_inputs,
            &weights,
            Order2::constant,
            JetField::add,
            JetField::scale,
        );
        let derivative_stacks = [
            [0.8, -0.3, 0.7, 0.0, 0.0],
            [-0.2, 1.1, -0.4, 0.0, 0.0],
            [1.4, 0.25, 0.6, 0.0, 0.0],
        ];
        let fixed_add_direct = fixed_inputs[0].add_constant(0.65);
        let fixed_add_scalar = fixed_inputs[0].add(&Order2::constant(0.65));
        let fixed_multiply_add_direct =
            fixed_inputs[0].multiply_add(&fixed_inputs[1], &fixed_inputs[2]);
        let fixed_multiply_add_scalar = multiply_add_default(
            &fixed_inputs[0],
            &fixed_inputs[1],
            &fixed_inputs[2],
            JetField::mul,
            JetField::add,
        );
        let fixed_composed_direct = Order2::composed_sum(&fixed_inputs, &derivative_stacks);
        let fixed_composed_scalar = composed_sum_default(
            &fixed_inputs,
            &derivative_stacks,
            Order2::constant,
            JetField::add,
            JetField::compose_unary,
        );

        let value_vars: [RuntimeValue; K] =
            std::array::from_fn(|axis| RuntimeValue::variable(values[axis], axis, K, &()));
        let value_inputs = [
            value_vars[0].mul(&value_vars[1]).add(&value_vars[3]),
            value_vars[1].exp().add(&value_vars[2].scale(0.4)),
            value_vars[2].mul(&value_vars[2]).sub(&value_vars[0]),
        ];
        let value_quadratic =
            RuntimeValue::symmetric_quadratic_form(&value_inputs, &coefficients, K, &());
        let value_linear = RuntimeValue::linear_combination(&value_inputs, &weights, K, &());
        let value_composed = RuntimeValue::composed_sum(&value_inputs, &derivative_stacks, K, &());

        let arena = DynamicJetArena::new();
        let dynamic_vars: [DynamicOrder2<'_>; K] =
            std::array::from_fn(|axis| DynamicOrder2::variable(values[axis], axis, K, &arena));
        let dynamic_inputs = [
            dynamic_vars[0].mul(&dynamic_vars[1]).add(&dynamic_vars[3]),
            dynamic_vars[1].exp().add(&dynamic_vars[2].scale(0.4)),
            dynamic_vars[2].mul(&dynamic_vars[2]).sub(&dynamic_vars[0]),
        ];
        let dynamic_direct =
            DynamicOrder2::symmetric_quadratic_form(&dynamic_inputs, &coefficients, K, &arena);
        let dynamic_scalar = symmetric_quadratic_form_default(
            &dynamic_inputs,
            &coefficients,
            |value| DynamicOrder2::constant(value, K, &arena),
            RuntimeJetScalar::add,
            RuntimeJetScalar::mul,
            RuntimeJetScalar::scale,
        );
        let dynamic_linear_direct =
            DynamicOrder2::linear_combination(&dynamic_inputs, &weights, K, &arena);
        let dynamic_linear_scalar = linear_combination_default(
            &dynamic_inputs,
            &weights,
            |value| DynamicOrder2::constant(value, K, &arena),
            RuntimeJetScalar::add,
            RuntimeJetScalar::scale,
        );
        let dynamic_add_direct = dynamic_inputs[0].add_constant(0.65, &arena);
        let dynamic_add_scalar = dynamic_inputs[0].add(&DynamicOrder2::constant(0.65, K, &arena));
        let dynamic_multiply_add_direct =
            dynamic_inputs[0].multiply_add(&dynamic_inputs[1], &dynamic_inputs[2]);
        let dynamic_multiply_add_scalar = multiply_add_default(
            &dynamic_inputs[0],
            &dynamic_inputs[1],
            &dynamic_inputs[2],
            RuntimeJetScalar::mul,
            RuntimeJetScalar::add,
        );
        let dynamic_composed_direct =
            DynamicOrder2::composed_sum(&dynamic_inputs, &derivative_stacks, K, &arena);
        let dynamic_composed_scalar = composed_sum_default(
            &dynamic_inputs,
            &derivative_stacks,
            |value| DynamicOrder2::constant(value, K, &arena),
            RuntimeJetScalar::add,
            RuntimeJetScalar::compose_unary,
        );

        let tolerance = 2.0e-13;
        for (label, actual, expected) in [
            ("fixed value", fixed_direct.value(), fixed_scalar.value()),
            (
                "zero-order quadratic value",
                value_quadratic.value(),
                fixed_direct.value(),
            ),
            (
                "zero-order linear value",
                value_linear.value(),
                fixed_linear_direct.value(),
            ),
            (
                "zero-order composed value",
                value_composed.value(),
                fixed_composed_direct.value(),
            ),
            (
                "dynamic value",
                dynamic_direct.value(),
                dynamic_scalar.value(),
            ),
            (
                "fixed linear value",
                fixed_linear_direct.value(),
                fixed_linear_scalar.value(),
            ),
            (
                "dynamic linear value",
                dynamic_linear_direct.value(),
                dynamic_linear_scalar.value(),
            ),
            (
                "fixed add-constant value",
                fixed_add_direct.value(),
                fixed_add_scalar.value(),
            ),
            (
                "dynamic add-constant value",
                dynamic_add_direct.value(),
                dynamic_add_scalar.value(),
            ),
            (
                "fixed multiply-add value",
                fixed_multiply_add_direct.value(),
                fixed_multiply_add_scalar.value(),
            ),
            (
                "dynamic multiply-add value",
                dynamic_multiply_add_direct.value(),
                dynamic_multiply_add_scalar.value(),
            ),
            (
                "fixed composed-sum value",
                fixed_composed_direct.value(),
                fixed_composed_scalar.value(),
            ),
            (
                "dynamic composed-sum value",
                dynamic_composed_direct.value(),
                dynamic_composed_scalar.value(),
            ),
        ] {
            assert!(
                (actual - expected).abs() <= tolerance * actual.abs().max(expected.abs()).max(1.0),
                "{label}: direct={actual:+.16e}, scalar={expected:+.16e}"
            );
        }
        for primary_a in 0..K {
            for (label, actual, expected) in [
                (
                    "fixed gradient",
                    fixed_direct.g()[primary_a],
                    fixed_scalar.g()[primary_a],
                ),
                (
                    "dynamic gradient",
                    dynamic_direct.g()[primary_a],
                    dynamic_scalar.g()[primary_a],
                ),
                (
                    "fixed linear gradient",
                    fixed_linear_direct.g()[primary_a],
                    fixed_linear_scalar.g()[primary_a],
                ),
                (
                    "dynamic linear gradient",
                    dynamic_linear_direct.g()[primary_a],
                    dynamic_linear_scalar.g()[primary_a],
                ),
                (
                    "fixed add-constant gradient",
                    fixed_add_direct.g()[primary_a],
                    fixed_add_scalar.g()[primary_a],
                ),
                (
                    "dynamic add-constant gradient",
                    dynamic_add_direct.g()[primary_a],
                    dynamic_add_scalar.g()[primary_a],
                ),
                (
                    "fixed multiply-add gradient",
                    fixed_multiply_add_direct.g()[primary_a],
                    fixed_multiply_add_scalar.g()[primary_a],
                ),
                (
                    "dynamic multiply-add gradient",
                    dynamic_multiply_add_direct.g()[primary_a],
                    dynamic_multiply_add_scalar.g()[primary_a],
                ),
                (
                    "fixed composed-sum gradient",
                    fixed_composed_direct.g()[primary_a],
                    fixed_composed_scalar.g()[primary_a],
                ),
                (
                    "dynamic composed-sum gradient",
                    dynamic_composed_direct.g()[primary_a],
                    dynamic_composed_scalar.g()[primary_a],
                ),
            ] {
                assert!(
                    (actual - expected).abs()
                        <= tolerance * actual.abs().max(expected.abs()).max(1.0),
                    "{label}[{primary_a}]: direct={actual:+.16e}, scalar={expected:+.16e}"
                );
            }
            for primary_b in 0..K {
                for (label, actual, expected) in [
                    (
                        "fixed Hessian",
                        fixed_direct.h()[primary_a][primary_b],
                        fixed_scalar.h()[primary_a][primary_b],
                    ),
                    (
                        "dynamic Hessian",
                        dynamic_direct.h_at(primary_a, primary_b),
                        dynamic_scalar.h_at(primary_a, primary_b),
                    ),
                    (
                        "fixed linear Hessian",
                        fixed_linear_direct.h()[primary_a][primary_b],
                        fixed_linear_scalar.h()[primary_a][primary_b],
                    ),
                    (
                        "dynamic linear Hessian",
                        dynamic_linear_direct.h_at(primary_a, primary_b),
                        dynamic_linear_scalar.h_at(primary_a, primary_b),
                    ),
                    (
                        "fixed add-constant Hessian",
                        fixed_add_direct.h()[primary_a][primary_b],
                        fixed_add_scalar.h()[primary_a][primary_b],
                    ),
                    (
                        "dynamic add-constant Hessian",
                        dynamic_add_direct.h_at(primary_a, primary_b),
                        dynamic_add_scalar.h_at(primary_a, primary_b),
                    ),
                    (
                        "fixed multiply-add Hessian",
                        fixed_multiply_add_direct.h()[primary_a][primary_b],
                        fixed_multiply_add_scalar.h()[primary_a][primary_b],
                    ),
                    (
                        "dynamic multiply-add Hessian",
                        dynamic_multiply_add_direct.h_at(primary_a, primary_b),
                        dynamic_multiply_add_scalar.h_at(primary_a, primary_b),
                    ),
                    (
                        "fixed composed-sum Hessian",
                        fixed_composed_direct.h()[primary_a][primary_b],
                        fixed_composed_scalar.h()[primary_a][primary_b],
                    ),
                    (
                        "dynamic composed-sum Hessian",
                        dynamic_composed_direct.h_at(primary_a, primary_b),
                        dynamic_composed_scalar.h_at(primary_a, primary_b),
                    ),
                ] {
                    assert!(
                        (actual - expected).abs()
                            <= tolerance * actual.abs().max(expected.abs()).max(1.0),
                        "{label}[{primary_a},{primary_b}]: direct={actual:+.16e}, scalar={expected:+.16e}"
                    );
                }
            }
        }
    }

    #[test]
    fn compiled_product_affine_and_fused_nodes_match_scalar_program_randomized() {
        const K: usize = 4;
        const TERMS: usize = 10;

        fn sample(state: &mut u64) -> f64 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            let unit = (*state >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64));
            2.0 * unit - 1.0
        }

        fn arbitrary_order2<const K: usize>(state: &mut u64) -> Order2<K> {
            let mut tower = crate::jet_tower::Tower2::zero();
            tower.v = sample(state);
            for primary in 0..K {
                tower.g[primary] = sample(state);
                for other in primary..K {
                    let channel = sample(state);
                    tower.h[primary][other] = channel;
                    tower.h[other][primary] = channel;
                }
            }
            Order2(tower)
        }

        fn close(actual: f64, expected: f64, case: usize, label: &str) {
            let tolerance = 2.0e-12 * actual.abs().max(expected.abs()).max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "case {case} {label}: direct={actual:+.16e}, scalar={expected:+.16e}, tolerance={tolerance:.3e}"
            );
        }

        let mut state = 0x932a_ff1e_c0de_5eed_u64;
        for case in 0..256 {
            // Arbitrary value, gradient, and symmetric-Hessian inputs model the
            // complete local state of nonlinear upstream expressions. This is
            // stronger than testing only independently seeded variables.
            let fixed_inputs: [Order2<K>; TERMS] =
                std::array::from_fn(|_| arbitrary_order2(&mut state));
            let mut input_scales: [f64; TERMS] = std::array::from_fn(|_| sample(&mut state));
            input_scales[0] = -1.25;
            input_scales[1] = 0.0;
            input_scales[4] = 1.25;
            let addend_scales: [f64; TERMS] = std::array::from_fn(|term| match term % 4 {
                0 => 0.0,
                1 => 1.0,
                2 => -0.75,
                _ => 0.35,
            });
            let derivative_stacks: [[f64; 5]; TERMS] =
                std::array::from_fn(|_| std::array::from_fn(|_| sample(&mut state)));
            let input_shift = sample(&mut state);
            let mut fixed_lefts = std::array::from_fn(|term| &fixed_inputs[term]);
            fixed_lefts[4] = &fixed_inputs[0];
            fixed_lefts[5] = &fixed_inputs[0];
            let fixed_right = &fixed_inputs[1];
            let fixed_addend = &fixed_inputs[2];

            let fixed_product_direct = fixed_inputs[0].product(&fixed_inputs[1]);
            let fixed_product_scalar = fixed_inputs[0].mul(&fixed_inputs[1]);
            let fixed_affine_direct =
                fixed_inputs[2].affine_compose(input_scales[2], input_shift, derivative_stacks[2]);
            let fixed_affine_scalar = affine_compose_default(
                &fixed_inputs[2],
                input_scales[2],
                input_shift,
                derivative_stacks[2],
                JetField::scale,
                Order2::add_constant,
                JetField::compose_unary,
            );
            let fixed_sum_direct =
                Order2::affine_composed_sum(&fixed_inputs, &input_scales, &derivative_stacks);
            let fixed_sum_scalar = affine_composed_sum_default(
                &fixed_inputs,
                &input_scales,
                &derivative_stacks,
                Order2::constant,
                JetField::add,
                JetField::scale,
                Order2::add_constant,
                JetField::compose_unary,
            );
            let fixed_fused_direct = Order2::shared_multiply_add_affine_composed_sum(
                &fixed_lefts,
                fixed_right,
                fixed_addend,
                &addend_scales,
                &input_scales,
                &derivative_stacks,
            );
            let fixed_fused_scalar = shared_multiply_add_affine_composed_sum_default(
                &fixed_lefts,
                fixed_right,
                fixed_addend,
                &addend_scales,
                &input_scales,
                &derivative_stacks,
                Order2::constant,
                JetField::add,
                JetField::mul,
                JetField::scale,
                Order2::multiply_add,
                Order2::affine_compose,
            );

            let arena = DynamicJetArena::new();
            let dynamic_inputs: [DynamicOrder2<'_>; TERMS] = std::array::from_fn(|term| {
                DynamicOrder2::from_channel_functions(
                    fixed_inputs[term].value(),
                    K,
                    &arena,
                    |primary| fixed_inputs[term].g()[primary],
                    |primary, other| fixed_inputs[term].h()[primary][other],
                )
            });
            let dynamic_product_direct = dynamic_inputs[0].product(&dynamic_inputs[1]);
            let dynamic_product_scalar = dynamic_inputs[0].mul(&dynamic_inputs[1]);
            let dynamic_affine_direct = dynamic_inputs[2].affine_compose(
                input_scales[2],
                input_shift,
                derivative_stacks[2],
                &arena,
            );
            let dynamic_affine_scalar = affine_compose_default(
                &dynamic_inputs[2],
                input_scales[2],
                input_shift,
                derivative_stacks[2],
                RuntimeJetScalar::scale,
                |input, constant| input.add_constant(constant, &arena),
                RuntimeJetScalar::compose_unary,
            );
            let dynamic_sum_direct = DynamicOrder2::affine_composed_sum(
                &dynamic_inputs,
                &input_scales,
                &derivative_stacks,
                K,
                &arena,
            );
            let dynamic_sum_scalar = affine_composed_sum_default(
                &dynamic_inputs,
                &input_scales,
                &derivative_stacks,
                |value| DynamicOrder2::constant(value, K, &arena),
                RuntimeJetScalar::add,
                RuntimeJetScalar::scale,
                |input, constant| input.add_constant(constant, &arena),
                RuntimeJetScalar::compose_unary,
            );
            let mut dynamic_lefts = std::array::from_fn(|term| &dynamic_inputs[term]);
            dynamic_lefts[4] = &dynamic_inputs[0];
            dynamic_lefts[5] = &dynamic_inputs[0];
            let dynamic_right = &dynamic_inputs[1];
            let dynamic_addend = &dynamic_inputs[2];
            let dynamic_fused_direct = DynamicOrder2::shared_multiply_add_affine_composed_sum(
                &dynamic_lefts,
                dynamic_right,
                dynamic_addend,
                &addend_scales,
                &input_scales,
                &derivative_stacks,
                K,
                &arena,
            );
            let dynamic_fused_scalar = shared_multiply_add_affine_composed_sum_default(
                &dynamic_lefts,
                dynamic_right,
                dynamic_addend,
                &addend_scales,
                &input_scales,
                &derivative_stacks,
                |value| DynamicOrder2::constant(value, K, &arena),
                RuntimeJetScalar::add,
                RuntimeJetScalar::mul,
                RuntimeJetScalar::scale,
                RuntimeJetScalar::multiply_add,
                |input, scale, shift, stack| input.affine_compose(scale, shift, stack, &arena),
            );

            for (label, actual, expected) in [
                (
                    "fixed product value",
                    fixed_product_direct.value(),
                    fixed_product_scalar.value(),
                ),
                (
                    "fixed affine value",
                    fixed_affine_direct.value(),
                    fixed_affine_scalar.value(),
                ),
                (
                    "fixed affine sum value",
                    fixed_sum_direct.value(),
                    fixed_sum_scalar.value(),
                ),
                (
                    "fixed fused value",
                    fixed_fused_direct.value(),
                    fixed_fused_scalar.value(),
                ),
                (
                    "dynamic product value",
                    dynamic_product_direct.value(),
                    dynamic_product_scalar.value(),
                ),
                (
                    "dynamic affine value",
                    dynamic_affine_direct.value(),
                    dynamic_affine_scalar.value(),
                ),
                (
                    "dynamic affine sum value",
                    dynamic_sum_direct.value(),
                    dynamic_sum_scalar.value(),
                ),
                (
                    "dynamic fused value",
                    dynamic_fused_direct.value(),
                    dynamic_fused_scalar.value(),
                ),
            ] {
                close(actual, expected, case, label);
            }
            for primary in 0..K {
                for (label, actual, expected) in [
                    (
                        "fixed product gradient",
                        fixed_product_direct.g()[primary],
                        fixed_product_scalar.g()[primary],
                    ),
                    (
                        "fixed affine gradient",
                        fixed_affine_direct.g()[primary],
                        fixed_affine_scalar.g()[primary],
                    ),
                    (
                        "fixed affine sum gradient",
                        fixed_sum_direct.g()[primary],
                        fixed_sum_scalar.g()[primary],
                    ),
                    (
                        "fixed fused gradient",
                        fixed_fused_direct.g()[primary],
                        fixed_fused_scalar.g()[primary],
                    ),
                    (
                        "dynamic product gradient",
                        dynamic_product_direct.g()[primary],
                        dynamic_product_scalar.g()[primary],
                    ),
                    (
                        "dynamic affine gradient",
                        dynamic_affine_direct.g()[primary],
                        dynamic_affine_scalar.g()[primary],
                    ),
                    (
                        "dynamic affine sum gradient",
                        dynamic_sum_direct.g()[primary],
                        dynamic_sum_scalar.g()[primary],
                    ),
                    (
                        "dynamic fused gradient",
                        dynamic_fused_direct.g()[primary],
                        dynamic_fused_scalar.g()[primary],
                    ),
                ] {
                    close(actual, expected, case, label);
                }
                for other in 0..K {
                    for (label, actual, expected) in [
                        (
                            "fixed product Hessian",
                            fixed_product_direct.h()[primary][other],
                            fixed_product_scalar.h()[primary][other],
                        ),
                        (
                            "fixed affine Hessian",
                            fixed_affine_direct.h()[primary][other],
                            fixed_affine_scalar.h()[primary][other],
                        ),
                        (
                            "fixed affine sum Hessian",
                            fixed_sum_direct.h()[primary][other],
                            fixed_sum_scalar.h()[primary][other],
                        ),
                        (
                            "fixed fused Hessian",
                            fixed_fused_direct.h()[primary][other],
                            fixed_fused_scalar.h()[primary][other],
                        ),
                        (
                            "dynamic product Hessian",
                            dynamic_product_direct.h_at(primary, other),
                            dynamic_product_scalar.h_at(primary, other),
                        ),
                        (
                            "dynamic affine Hessian",
                            dynamic_affine_direct.h_at(primary, other),
                            dynamic_affine_scalar.h_at(primary, other),
                        ),
                        (
                            "dynamic affine sum Hessian",
                            dynamic_sum_direct.h_at(primary, other),
                            dynamic_sum_scalar.h_at(primary, other),
                        ),
                        (
                            "dynamic fused Hessian",
                            dynamic_fused_direct.h_at(primary, other),
                            dynamic_fused_scalar.h_at(primary, other),
                        ),
                    ] {
                        close(actual, expected, case, label);
                    }
                }
            }
        }
    }

    #[test]
    fn shared_product_composition_accepts_empty_expression() {
        const K: usize = 4;
        let fixed_terms: [&Order2<K>; 0] = [];
        let fixed_shared = Order2::constant(1.0);
        let scales: [f64; 0] = [];
        let stacks: [[f64; 5]; 0] = [];
        let fixed = Order2::shared_multiply_add_affine_composed_sum(
            &fixed_terms,
            &fixed_shared,
            &fixed_shared,
            &scales,
            &scales,
            &stacks,
        );
        assert_eq!(fixed.value().to_bits(), 0.0_f64.to_bits());
        assert!(fixed.g().iter().all(|&channel| channel == 0.0));
        assert!(fixed.h().iter().flatten().all(|&channel| channel == 0.0));

        let value_terms: [&RuntimeValue; 0] = [];
        let value_shared = RuntimeValue::constant(1.0, K, &());
        let value = RuntimeValue::shared_multiply_add_affine_composed_sum(
            &value_terms,
            &value_shared,
            &value_shared,
            &scales,
            &scales,
            &stacks,
            K,
            &(),
        );
        assert_eq!(value.value().to_bits(), 0.0_f64.to_bits());
        assert_eq!(value.dimension(), K);

        let arena = DynamicJetArena::new();
        let dynamic_terms: [&DynamicOrder2<'_>; 0] = [];
        let dynamic_shared = DynamicOrder2::constant(1.0, K, &arena);
        let dynamic = DynamicOrder2::shared_multiply_add_affine_composed_sum(
            &dynamic_terms,
            &dynamic_shared,
            &dynamic_shared,
            &scales,
            &scales,
            &stacks,
            K,
            &arena,
        );
        assert_eq!(dynamic.value().to_bits(), 0.0_f64.to_bits());
        assert!((0..K).all(|axis| dynamic.g()[axis] == 0.0));
        assert!((0..K).all(|row| (0..K).all(|column| dynamic.h_at(row, column) == 0.0)));
    }

    #[test]
    fn runtime_fused_product_composition_preserves_tower4_channels() {
        const K: usize = 2;
        const N: usize = 9;
        let vars = [
            Tower4::<K>::variable(0.37, 0),
            Tower4::<K>::variable(-0.61, 1),
        ];
        let upstream = [
            vars[0].mul(&vars[1]).add(&vars[0].exp()),
            vars[1].mul(&vars[1]).add(&vars[0].scale(0.3)),
            vars[0].mul(&vars[0]).sub(&vars[1].scale(-0.2)),
        ];
        let lefts: [Tower4<K>; N] = std::array::from_fn(|term| upstream[term % upstream.len()]);
        let right = upstream[1];
        let addend = upstream[2];
        let addend_scales: [f64; N] = std::array::from_fn(|term| [-0.0, 1.0, -0.7, 0.25][term % 4]);
        let input_scales: [f64; N] = std::array::from_fn(|term| [0.0, -1.3, 0.45, 1.1][term % 4]);
        let stacks: [[f64; 5]; N] = std::array::from_fn(|term| {
            let t = term as f64 + 1.0;
            [0.17 * t, -0.11 * t, 0.07 * t, -0.03 * t, 0.013 * t]
        });

        let expected = (0..N).fold(Tower4::<K>::constant(0.0), |sum, term| {
            let inner = if addend_scales[term] == 0.0 {
                lefts[term].mul(&right)
            } else if addend_scales[term] == 1.0 {
                JetScalar::multiply_add(&lefts[term], &right, &addend)
            } else {
                JetScalar::multiply_add(&lefts[term], &right, &addend.scale(addend_scales[term]))
            };
            sum.add(&JetScalar::affine_compose(
                &inner,
                input_scales[term],
                0.0,
                stacks[term],
            ))
        });
        let wrapped_lefts: [FixedRuntimeJet<Tower4<K>, K>; N] =
            std::array::from_fn(|term| FixedRuntimeJet::from_inner(lefts[term]));
        let wrapped_right = FixedRuntimeJet::from_inner(right);
        let wrapped_addend = FixedRuntimeJet::from_inner(addend);
        let wrapped_left_refs: [&FixedRuntimeJet<Tower4<K>, K>; N] =
            std::array::from_fn(|term| &wrapped_lefts[term]);
        let actual = FixedRuntimeJet::<Tower4<K>, K>::shared_multiply_add_affine_composed_sum(
            &wrapped_left_refs,
            &wrapped_right,
            &wrapped_addend,
            &addend_scales,
            &input_scales,
            &stacks,
            K,
            &(),
        )
        .into_inner();

        let same = |label: &str, got: f64, want: f64| {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{label}: got={got:+.17e}, want={want:+.17e}"
            );
        };
        same("value", actual.v, expected.v);
        for a in 0..K {
            same("gradient", actual.g[a], expected.g[a]);
            for b in 0..K {
                same("Hessian", actual.h[a][b], expected.h[a][b]);
                for c in 0..K {
                    same("third", actual.t3[a][b][c], expected.t3[a][b][c]);
                    for d in 0..K {
                        same("fourth", actual.t4[a][b][c][d], expected.t4[a][b][c][d]);
                    }
                }
            }
        }
    }

    /// A small polynomial-plus-unary row expression written ONCE, generically
    /// over `S: JetScalar<2>`, so it can be evaluated against every scalar:
    /// `ℓ = (e^{p0·p1} + 2) · √(p0·p0 + 1) − p1·p1·0.5`.
    /// Exercises mul, add/sub, scale, exp, sqrt — every algebra op.
    fn row_expr<S: JetScalar<2>>(p: &[S; 2]) -> S {
        let g = p[0].mul(&p[1]).exp();
        let inner = g.add(&S::constant(2.0));
        let radic = p[0].mul(&p[0]).add(&S::constant(1.0)).sqrt();
        inner.mul(&radic).sub(&p[1].mul(&p[1]).scale(0.5))
    }

    /// The same expression exposed through the canonical generic row-program seam.
    struct ExprProgram {
        p: [f64; 2],
    }
    impl RowProgram<2> for ExprProgram {
        fn n_rows(&self) -> usize {
            1
        }
        fn primaries(&self, row: usize) -> Result<[f64; 2], String> {
            if row >= self.n_rows() {
                return Err(format!("ExprProgram: row {row} out of range"));
            }
            Ok(self.p)
        }
        fn eval<S: JetScalar<2>>(&self, row: usize, p: &[S; 2]) -> Result<S, String> {
            if row >= self.n_rows() {
                return Err(format!("ExprProgram: row {row} out of range"));
            }
            Ok(row_expr(p))
        }
    }

    const SEED: [f64; 2] = [0.37, -0.81];
    const U: [f64; 2] = [0.6, -0.2];
    const V: [f64; 2] = [-0.4, 1.1];
    const TOL: f64 = 1e-10;

    fn close(a: f64, b: f64, label: &str) {
        let band = TOL + TOL * a.abs().max(b.abs());
        assert!(
            (a - b).abs() <= band,
            "{label}: {a:+.15e} vs {b:+.15e} (band {band:.3e})"
        );
    }

    fn tower() -> Tower4<2> {
        program_full_tower(&ExprProgram { p: SEED }, 0).expect("tower")
    }

    /// Order2 reproduces Tower4's value/grad/Hessian channels exactly.
    #[test]
    fn order2_matches_tower_value_grad_hessian() {
        let t = tower();
        let vars: [Order2<2>; 2] = std::array::from_fn(|a| Order2::variable(SEED[a], a));
        let s = row_expr(&vars);
        close(s.value(), t.v, "value");
        for a in 0..2 {
            close(s.0.g[a], t.g[a], &format!("grad[{a}]"));
            for b in 0..2 {
                close(s.h()[a][b], t.h[a][b], &format!("hess[{a}][{b}]"));
            }
        }
    }

    #[test]
    fn mapped_order2_accumulator_matches_dense_overlapping_atoms() {
        const K: usize = 4;
        let p = [0.2_f64, 0.7, -0.4, 0.3];
        let dense_vars: [Order2<K>; K] =
            std::array::from_fn(|axis| Order2::variable(p[axis], axis));
        let dense_q0 = dense_vars[3].mul(&dense_vars[1]).add(&dense_vars[3].exp());
        let dense_q1 = dense_vars[1].mul(&dense_vars[2]).sub(&dense_vars[2]);
        let dense = dense_q0.ln().add(&dense_q1.exp());

        let local_q0_vars: [Order2<2>; 2] =
            std::array::from_fn(|axis| Order2::variable(p[[3, 1][axis]], axis));
        let local_q0 = local_q0_vars[0]
            .mul(&local_q0_vars[1])
            .add(&local_q0_vars[0].exp());
        let local_q1_vars: [Order2<2>; 2] =
            std::array::from_fn(|axis| Order2::variable(p[[1, 2][axis]], axis));
        let local_q1 = local_q1_vars[0]
            .mul(&local_q1_vars[1])
            .sub(&local_q1_vars[1]);

        let q0 = local_q0.value();
        let q1_exp = local_q1.value().exp();
        let mut lowered = MappedOrder2Accumulator::<K>::zero();
        lowered.add_composed(
            &local_q0,
            [3, 1],
            [q0.ln(), q0.recip(), -1.0 / (q0 * q0)],
            false,
            [false, false],
            [false, false, false],
        );
        lowered.add_composed(
            &local_q1,
            [1, 2],
            [q1_exp, q1_exp, q1_exp],
            true,
            [true, false],
            [true, false, false],
        );
        let (value, gradient, hessian) = lowered.into_channels();

        close(value, dense.value(), "mapped value");
        for i in 0..K {
            close(gradient[i], dense.g()[i], &format!("mapped gradient[{i}]"));
            for j in 0..K {
                close(
                    hessian[i][j],
                    dense.h()[i][j],
                    &format!("mapped Hessian[{i},{j}]"),
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "mapped atom axes must be injective")]
    fn mapped_order2_accumulator_rejects_duplicate_axes() {
        let vars: [Order2<2>; 2] = std::array::from_fn(|axis| Order2::variable(0.2, axis));
        let atom = vars[0].add(&vars[1]);
        let mut lowered = MappedOrder2Accumulator::<2>::zero();
        lowered.add_composed(
            &atom,
            [1, 1],
            [0.4, 1.0, 0.0],
            false,
            [false, false],
            [false, false, false],
        );
    }

    #[test]
    #[should_panic(expected = "mapped atom axis must be within")]
    fn mapped_order2_accumulator_rejects_out_of_range_axes() {
        let atom = Order2::<1>::variable(0.2, 0);
        let mut lowered = MappedOrder2Accumulator::<2>::zero();
        lowered.add_composed(&atom, [2], [0.2, 1.0, 0.0], false, [false], [false]);
    }

    #[test]
    fn dynamic_order2_accumulator_matches_dense_composed_sum() {
        const K: usize = 4;

        struct Term {
            first: f64,
            second: f64,
            gradient: [f64; K],
            hessian: [[f64; K]; K],
        }

        impl DynamicOrder2Term for Term {
            fn outer_first(&self) -> f64 {
                self.first
            }

            fn outer_second(&self) -> f64 {
                self.second
            }

            fn inner_gradient(&self, axis: usize) -> f64 {
                self.gradient[axis]
            }

            fn inner_hessian(&self, row: usize, column: usize) -> f64 {
                self.hessian[row][column]
            }
        }

        let p = [0.7, -0.3, 0.2, 0.8];
        let vars: [Order2<K>; K] = std::array::from_fn(|axis| Order2::variable(p[axis], axis));
        let first_atom = vars[0]
            .mul(&vars[1])
            .add(&vars[2].exp())
            .add(&Order2::constant(1.5));
        let second_atom = vars[1].mul(&vars[3]).sub(&vars[0]);
        let first_value = first_atom.value();
        let second_exp = second_atom.value().exp();
        let first_stack = [
            first_value.ln(),
            first_value.recip(),
            -1.0 / (first_value * first_value),
            0.0,
            0.0,
        ];
        let second_stack = [second_exp, second_exp, second_exp, second_exp, second_exp];
        let dense = first_atom
            .compose_unary(first_stack)
            .add(&second_atom.compose_unary(second_stack));
        let terms = [
            Term {
                first: first_stack[1],
                second: first_stack[2],
                gradient: first_atom.g(),
                hessian: first_atom.h(),
            },
            Term {
                first: second_stack[1],
                second: second_stack[2],
                gradient: second_atom.g(),
                hessian: second_atom.h(),
            },
        ];
        let (value, gradient, hessian) = DynamicOrder2Accumulator::from_composed_sum(
            K,
            first_stack[0] + second_stack[0],
            &terms,
        )
        .into_channels();

        close(value, dense.value(), "dynamic value");
        for row in 0..K {
            close(
                gradient[row],
                dense.g()[row],
                &format!("dynamic gradient[{row}]"),
            );
            for column in 0..K {
                close(
                    hessian[row * K + column],
                    dense.h()[row][column],
                    &format!("dynamic Hessian[{row},{column}]"),
                );
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct FullTwoPattern;

    impl HessianPattern<2, 3> for FullTwoPattern {
        const PAIRS: [(usize, usize); 3] = [(0, 0), (0, 1), (1, 1)];
        const PAIR_BITS: [[u128; 2]; 2] = hessian_pair_bits(Self::PAIRS);
    }

    /// Patterned order-two arithmetic is channel-identical to dense `Order2`
    /// when the pattern covers the expression's complete Hessian support.
    #[test]
    fn patterned_order2_matches_dense_order2() {
        type Sparse = PatternedOrder2<FullTwoPattern, 2, 3>;
        let dense_vars: [Order2<2>; 2] = std::array::from_fn(|a| Order2::variable(SEED[a], a));
        let sparse_vars: [Sparse; 2] = std::array::from_fn(|a| Sparse::variable(SEED[a], a));
        let dense = row_expr(&dense_vars);
        let sparse = row_expr(&sparse_vars);
        close(sparse.value(), dense.value(), "patterned value");
        for i in 0..2 {
            close(sparse.g()[i], dense.g()[i], &format!("patterned grad[{i}]"));
            for j in 0..2 {
                close(
                    sparse.h()[i][j],
                    dense.h()[i][j],
                    &format!("patterned hess[{i}][{j}]"),
                );
            }
        }
    }

    /// `compose_unary_with` is `to_bits`-identical to the explicit
    /// `compose_unary(stack_fn(value))` form. Exercised on [`Order2`] across
    /// `K ∈ {2,3,4,9}`, ≥ 4000 random seeded inputs.
    #[test]
    fn compose_unary_with_scalar_seam_bit_identical() {
        fn rand_unit(state: &mut u64) -> f64 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            2.0 * ((x >> 11) as f64 / ((1u64 << 53) as f64)) - 1.0
        }
        // A base-value-dependent finite stack standing in for a family stack.
        fn stack(u: f64) -> [f64; 5] {
            [
                u.sin(),
                u.cos(),
                (2.0 * u).sin(),
                (0.5 * u).cos(),
                u * u - 0.3,
            ]
        }
        fn run<const K: usize>(state: &mut u64, n: usize) -> usize {
            for _ in 0..n {
                // A non-trivial Order2<K> jet: a seeded variable pushed through a
                // couple of algebra ops so g/h are dense, then exercise the seam.
                let base = rand_unit(state);
                let mut s = Order2::<K>::variable(base, 0);
                for a in 1..K {
                    s = crate::nested_dual::JetField::mul(
                        &s,
                        &Order2::<K>::variable(rand_unit(state), a),
                    );
                }
                let with = s.compose_unary_with(stack);
                let explicit = s.compose_unary(stack(s.value()));
                assert_eq!(with.value().to_bits(), explicit.value().to_bits(), "value");
                for a in 0..K {
                    assert_eq!(with.g()[a].to_bits(), explicit.g()[a].to_bits(), "g[{a}]");
                    for b in 0..K {
                        assert_eq!(
                            with.h()[a][b].to_bits(),
                            explicit.h()[a][b].to_bits(),
                            "h[{a}][{b}]"
                        );
                    }
                }
            }
            n
        }
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let total = run::<2>(&mut st, 1100)
            + run::<3>(&mut st, 1100)
            + run::<4>(&mut st, 1100)
            + run::<9>(&mut st, 1100);
        assert_eq!(total, 4400);
    }

    /// OneSeed's ε-Hessian is the contracted third Σ_c ℓ_{abc} u_c, matching
    /// `Tower4::third_contracted(u)`. Base channels also match the tower.
    #[test]
    fn one_seed_matches_tower_third_contracted() {
        let t = tower();
        let truth = t.third_contracted(&U);
        let vars: [OneSeed<2>; 2] =
            std::array::from_fn(|a| OneSeed::seed_direction(SEED[a], a, U[a]));
        let s = row_expr(&vars);
        // Base channels are the plain (v, g, H).
        close(s.value(), t.v, "value");
        for a in 0..2 {
            for b in 0..2 {
                close(s.base.h()[a][b], t.h[a][b], &format!("base hess[{a}][{b}]"));
            }
        }
        let third = s.contracted_third();
        for a in 0..2 {
            for b in 0..2 {
                close(third[a][b], truth[a][b], &format!("third[{a}][{b}]"));
            }
        }
    }

    /// The fused fixed-width OneSeed product/composition preserves every
    /// channel of the defining unfused nilpotent algebra. Random symmetric
    /// Order2 inputs exercise dense non-seed intermediates, while exact
    /// mirrored symmetry is checked as an independent production invariant.
    #[test]
    fn fused_one_seed_channels_match_unfused_definition_932() {
        const K: usize = 8;

        fn random_scalar(state: &mut u64) -> f64 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            ((*state >> 11) as f64 / ((1_u64 << 53) as f64)) * 2.0 - 1.0
        }

        fn random_order2<const N: usize>(state: &mut u64) -> Order2<N> {
            let mut tower = crate::jet_tower::Tower2::<N>::zero();
            tower.v = random_scalar(state);
            for axis in 0..N {
                tower.g[axis] = random_scalar(state);
            }
            for row in 0..N {
                for column in row..N {
                    let channel = random_scalar(state);
                    tower.h[row][column] = channel;
                    tower.h[column][row] = channel;
                }
            }
            Order2(tower)
        }

        fn assert_channels_close<const N: usize>(
            label: &str,
            actual: &OneSeed<N>,
            expected: &OneSeed<N>,
        ) {
            for (part_label, actual_part, expected_part, require_exact_symmetry) in [
                ("base", &actual.base.0, &expected.base.0, false),
                ("eps", &actual.eps.0, &expected.eps.0, true),
            ] {
                let check = |channel: &str, got: f64, want: f64| {
                    let tolerance = 2.0e-14 * got.abs().max(want.abs()).max(1.0);
                    assert!(
                        (got - want).abs() <= tolerance,
                        "{label} {part_label} {channel}: got={got:+.17e} want={want:+.17e}"
                    );
                };
                check("value", actual_part.v, expected_part.v);
                for row in 0..N {
                    check(
                        &format!("gradient[{row}]"),
                        actual_part.g[row],
                        expected_part.g[row],
                    );
                    for column in 0..N {
                        check(
                            &format!("hessian[{row},{column}]"),
                            actual_part.h[row][column],
                            expected_part.h[row][column],
                        );
                        if require_exact_symmetry {
                            assert_eq!(
                                actual_part.h[row][column].to_bits(),
                                actual_part.h[column][row].to_bits(),
                                "{label} {part_label} Hessian symmetry at [{row},{column}]"
                            );
                        }
                    }
                }
            }
        }

        let mut state = 0x9320_1eed_5eed_cafe_u64;
        for sample in 0..256 {
            let left = OneSeed {
                base: random_order2::<K>(&mut state),
                eps: random_order2::<K>(&mut state),
            };
            let right = OneSeed {
                base: random_order2::<K>(&mut state),
                eps: random_order2::<K>(&mut state),
            };

            let fused_product = left.mul(&right);
            let unfused_product = OneSeed {
                base: left.base.mul(&right.base),
                eps: left.base.mul(&right.eps).add(&left.eps.mul(&right.base)),
            };
            assert_channels_close(
                &format!("sample {sample} product"),
                &fused_product,
                &unfused_product,
            );

            let derivatives: [f64; 5] = std::array::from_fn(|_| random_scalar(&mut state));
            let fused_composition = left.compose_unary(derivatives);
            let unfused_composition = OneSeed {
                base: left.base.compose_unary(derivatives),
                eps: left
                    .base
                    .compose_unary([
                        derivatives[1],
                        derivatives[2],
                        derivatives[3],
                        derivatives[4],
                        derivatives[4],
                    ])
                    .mul(&left.eps),
            };
            assert_channels_close(
                &format!("sample {sample} composition"),
                &fused_composition,
                &unfused_composition,
            );
        }
    }

    /// TwoSeed's εδ-Hessian is the contracted fourth Σ_{cd} ℓ_{abcd} u_c v_d,
    /// matching `Tower4::fourth_contracted(u, v)`. The ε / δ single-seed parts
    /// reproduce the two third contractions Σ_c ℓ_{abc} u_c and …v_d.
    #[test]
    fn two_seed_matches_tower_fourth_contracted() {
        let t = tower();
        let truth4 = t.fourth_contracted(&U, &V);
        let truth3_u = t.third_contracted(&U);
        let truth3_v = t.third_contracted(&V);
        let vars: [TwoSeed<2>; 2] = std::array::from_fn(|a| TwoSeed::seed(SEED[a], a, U[a], V[a]));
        let s = row_expr(&vars);
        close(s.value(), t.v, "value");
        for a in 0..2 {
            close(s.base.0.g[a], t.g[a], &format!("grad[{a}]"));
            for b in 0..2 {
                close(s.base.h()[a][b], t.h[a][b], &format!("base hess[{a}][{b}]"));
                close(
                    s.eps.h()[a][b],
                    truth3_u[a][b],
                    &format!("eps third_u[{a}][{b}]"),
                );
                close(
                    s.del.h()[a][b],
                    truth3_v[a][b],
                    &format!("del third_v[{a}][{b}]"),
                );
            }
        }
        let fourth = s.contracted_fourth();
        for a in 0..2 {
            for b in 0..2 {
                close(fourth[a][b], truth4[a][b], &format!("fourth[{a}][{b}]"));
            }
        }
    }

    /// The generic `row_nll_generic` seam (added to Tower4's program trait
    /// surface) evaluates the SAME expression on each scalar and extracts the
    /// channel a consumer asks for, agreeing with the direct Tower4 contraction.
    #[test]
    fn generic_program_seam_matches_tower_for_every_channel() {
        let t = tower();
        // Order2 via generic seam.
        let o2: [Order2<2>; 2] = std::array::from_fn(|a| Order2::variable(SEED[a], a));
        let so2 = row_expr(&o2);
        close(so2.value(), t.v, "seam order2 value");
        // OneSeed third.
        let os: [OneSeed<2>; 2] =
            std::array::from_fn(|a| OneSeed::seed_direction(SEED[a], a, U[a]));
        let third = row_expr(&os).contracted_third();
        let truth3 = t.third_contracted(&U);
        for a in 0..2 {
            for b in 0..2 {
                close(third[a][b], truth3[a][b], &format!("seam third[{a}][{b}]"));
            }
        }
        // TwoSeed fourth.
        let ts: [TwoSeed<2>; 2] = std::array::from_fn(|a| TwoSeed::seed(SEED[a], a, U[a], V[a]));
        let fourth = row_expr(&ts).contracted_fourth();
        let truth4 = t.fourth_contracted(&U, &V);
        for a in 0..2 {
            for b in 0..2 {
                close(
                    fourth[a][b],
                    truth4[a][b],
                    &format!("seam fourth[{a}][{b}]"),
                );
            }
        }
    }

    /// The (test-only) `Tower4: JetScalar` impl is the all-channels oracle scalar:
    /// evaluating the SAME generic `row_expr` at `S = Tower4` (through the
    /// `JetScalar` trait ops) must reproduce, channel-for-channel, the `Tower4`
    /// obtained through the canonical `RowProgram` path (`program_full_tower`).
    /// This pins that the full-tower scalar and the contracted scalar oracles above
    /// all execute the same generic algebra surface.
    #[test]
    fn tower4_as_jetscalar_matches_program_tower_all_channels() {
        let t = tower();
        let vars: [Tower4<2>; 2] = std::array::from_fn(|a| Tower4::variable(SEED[a], a));
        let s = row_expr(&vars);
        close(s.v, t.v, "tower-jetscalar value");
        for a in 0..2 {
            close(s.g[a], t.g[a], &format!("tower-jetscalar grad[{a}]"));
            for b in 0..2 {
                close(
                    s.h[a][b],
                    t.h[a][b],
                    &format!("tower-jetscalar hess[{a}][{b}]"),
                );
                for c in 0..2 {
                    close(
                        s.t3[a][b][c],
                        t.t3[a][b][c],
                        &format!("tower-jetscalar t3[{a}][{b}][{c}]"),
                    );
                    for d in 0..2 {
                        close(
                            s.t4[a][b][c][d],
                            t.t4[a][b][c][d],
                            &format!("tower-jetscalar t4[{a}][{b}][{c}][{d}]"),
                        );
                    }
                }
            }
        }
    }

    /// Arena-backed runtime directional jets must reproduce the fixed packed
    /// algebras channel-for-channel. This isolates the runtime scalar algebra
    /// from every family row program before the SLS/SAE integration oracles.
    #[test]
    fn runtime_directional_jets_match_fixed_packed_algebra_932() {
        fn expression<'arena, S: RuntimeJetScalar<'arena>>(vars: &[S]) -> S {
            let bilinear = vars[0].mul(&vars[1]);
            let curved = vars[2].scale(0.7).add(&vars[3].mul(&vars[3]).scale(-0.2));
            bilinear
                .add(&curved)
                .exp()
                .mul(&vars[4].compose_unary([0.4, -0.3, 0.2, -0.1, 0.05]))
        }

        const K: usize = 5;
        let values = [0.2, -0.7, 0.4, 1.1, -0.3];
        let direction_u = [0.5, -0.2, 0.7, -0.4, 0.1];
        let direction_v = [-0.3, 0.8, 0.2, 0.6, -0.5];
        let close = |actual: f64, expected: f64| {
            let tolerance = 1.0e-13 * (1.0 + actual.abs().max(expected.abs()));
            assert!((actual - expected).abs() <= tolerance);
        };

        let fixed_one: Vec<FixedRuntimeJet<OneSeed<K>, K>> = (0..K)
            .map(|axis| FixedRuntimeJet {
                inner: OneSeed::seed_direction(values[axis], axis, direction_u[axis]),
            })
            .collect();
        let arena_one = DynamicJetArena::new();
        let dynamic_one: Vec<DynamicOneSeed<'_>> = (0..K)
            .map(|axis| {
                DynamicOneSeed::seed_direction(values[axis], axis, direction_u[axis], K, &arena_one)
            })
            .collect();
        let fixed_third = expression(&fixed_one).into_inner().contracted_third();
        let dynamic_third = expression(&dynamic_one);
        for a in 0..K {
            for b in 0..K {
                assert_eq!(
                    dynamic_third.contracted_third()[a * K + b].to_bits(),
                    dynamic_third.contracted_third()[b * K + a].to_bits(),
                    "arena third Hessian must be exactly symmetric at ({a},{b})"
                );
                close(
                    dynamic_third.contracted_third()[a * K + b],
                    fixed_third[a][b],
                );
            }
        }

        let fixed_one_v: Vec<FixedRuntimeJet<OneSeed<K>, K>> = (0..K)
            .map(|axis| FixedRuntimeJet {
                inner: OneSeed::seed_direction(values[axis], axis, direction_v[axis]),
            })
            .collect();
        let fixed_third_v = expression(&fixed_one_v).into_inner().contracted_third();
        let batch_workspace = DynamicJetBatchWorkspace::new(2);
        let directions = [direction_u, direction_v];
        let batch_vars = batch_workspace.alloc_slice_fill_with(K, |axis| {
            DynamicOneSeedBatch::seed_directions(values[axis], axis, K, &batch_workspace, |lane| {
                directions[lane][axis]
            })
        });
        let dynamic_batch = expression(batch_vars);
        assert_eq!(dynamic_batch.lanes(), 2);
        for lane in 0..2 {
            let expected = if lane == 0 {
                &fixed_third
            } else {
                &fixed_third_v
            };
            for a in 0..K {
                for b in 0..K {
                    close(
                        dynamic_batch.contracted_third(lane)[a * K + b],
                        expected[a][b],
                    );
                }
            }
        }

        let fixed_two: Vec<FixedRuntimeJet<TwoSeed<K>, K>> = (0..K)
            .map(|axis| FixedRuntimeJet {
                inner: TwoSeed::seed(values[axis], axis, direction_u[axis], direction_v[axis]),
            })
            .collect();
        let arena_two = DynamicJetArena::new();
        let dynamic_two: Vec<DynamicTwoSeed<'_>> = (0..K)
            .map(|axis| {
                DynamicTwoSeed::seed(
                    values[axis],
                    axis,
                    direction_u[axis],
                    direction_v[axis],
                    K,
                    &arena_two,
                )
            })
            .collect();
        let fixed_fourth = expression(&fixed_two).into_inner().contracted_fourth();
        let dynamic_fourth = expression(&dynamic_two);
        for a in 0..K {
            for b in 0..K {
                close(
                    dynamic_fourth.contracted_fourth()[a * K + b],
                    fixed_fourth[a][b],
                );
            }
        }

        let fixed_two_swapped: Vec<FixedRuntimeJet<TwoSeed<K>, K>> = (0..K)
            .map(|axis| {
                FixedRuntimeJet::from_inner(TwoSeed::seed(
                    values[axis],
                    axis,
                    direction_v[axis],
                    direction_u[axis],
                ))
            })
            .collect();
        let fixed_fourth_swapped = expression(&fixed_two_swapped)
            .into_inner()
            .contracted_fourth();
        let pair_workspace = DynamicJetBatchWorkspace::new(2);
        let direction_pairs = [(direction_u, direction_v), (direction_v, direction_u)];
        let pair_vars = pair_workspace.alloc_slice_fill_with(K, |axis| {
            DynamicTwoSeedBatch::seed_direction_pairs(
                values[axis],
                axis,
                K,
                &pair_workspace,
                |lane| (direction_pairs[lane].0[axis], direction_pairs[lane].1[axis]),
            )
        });
        let dynamic_pair_batch = expression(pair_vars);
        assert_eq!(dynamic_pair_batch.lanes(), 2);
        for lane in 0..2 {
            let expected = if lane == 0 {
                &fixed_fourth
            } else {
                &fixed_fourth_swapped
            };
            for a in 0..K {
                for b in 0..K {
                    close(
                        dynamic_pair_batch.contracted_fourth(lane)[a * K + b],
                        expected[a][b],
                    );
                }
            }
        }
    }

    #[test]
    fn dynamic_jet_arena_compacts_fragmented_high_water_932() {
        const WORDS_PER_ALLOCATION: usize = 1 << 17;
        const ALLOCATIONS: usize = 6;

        let mut arena = DynamicJetArena::new();
        for lane in 0..ALLOCATIONS {
            let allocation = arena.alloc_slice_fill_with(WORDS_PER_ALLOCATION, |_| lane as u64);
            std::hint::black_box(allocation);
        }
        let fragmented_high_water = arena.allocated_bytes();

        arena.reset();
        let compact_high_water = arena.allocated_bytes();
        assert!(
            compact_high_water >= fragmented_high_water,
            "compacted arena must retain the complete fragmented tape"
        );

        for lane in 0..ALLOCATIONS {
            let allocation = arena.alloc_slice_fill_with(WORDS_PER_ALLOCATION, |_| lane as u64);
            std::hint::black_box(allocation);
        }
        assert_eq!(
            arena.allocated_bytes(),
            compact_high_water,
            "equal replay must fit in the compacted chunk"
        );

        arena.reset();
        assert_eq!(
            arena.allocated_bytes(),
            compact_high_water,
            "stable reset must retain the compacted chunk"
        );
    }
}

#[cfg(test)]
mod batch_tests {
    //! SIMD row-batching oracle: prove [`Order2Batch<K>`] (4 rows in
    //! `wide::f64x4` lanes) is `to_bits`-identical, on every value/gradient/
    //! Hessian channel, to the production [`Order2<K>`] evaluated per row — and
    //! that the new scalar field [`Order2Lane<f64, K>`] is too. Composing the two
    //! claims, batch lane `i` reproduces the production scalar for row `i` bit
    //! for bit, so the 4× throughput is a free lunch (no result change).

    use super::{
        JetScalar, Lane, OneSeed, OneSeedBatch, OneSeedLane, Order2, Order2Batch, Order2Lane,
        TwoSeed, TwoSeedBatch, TwoSeedLane,
    };
    // The scalar-field algebra (`value`, `add`, …) lives on the shared `JetField`
    // base now, so the concrete-typed channel reads below need it in scope.
    use crate::nested_dual::JetField;

    /// The ops the witness row expression needs, so ONE generic body evaluates
    /// at the production [`Order2<K>`], the new scalar [`Order2Lane<f64, K>`],
    /// and the batched [`Order2Batch<K>`].
    trait RowAlg<const K: usize>: Copy {
        fn constant(c: f64) -> Self;
        fn add(&self, o: &Self) -> Self;
        fn sub(&self, o: &Self) -> Self;
        fn mul(&self, o: &Self) -> Self;
        fn scale(&self, s: f64) -> Self;
        fn exp(&self) -> Self;
        fn sqrt(&self) -> Self;
        fn recip(&self) -> Self;
    }

    impl<const K: usize> RowAlg<K> for Order2<K> {
        fn constant(c: f64) -> Self {
            <Self as JetScalar<K>>::constant(c)
        }
        fn add(&self, o: &Self) -> Self {
            crate::nested_dual::JetField::add(self, o)
        }
        fn sub(&self, o: &Self) -> Self {
            crate::nested_dual::JetField::sub(self, o)
        }
        fn mul(&self, o: &Self) -> Self {
            crate::nested_dual::JetField::mul(self, o)
        }
        fn scale(&self, s: f64) -> Self {
            crate::nested_dual::JetField::scale(self, s)
        }
        fn exp(&self) -> Self {
            JetScalar::exp(self)
        }
        fn sqrt(&self) -> Self {
            JetScalar::sqrt(self)
        }
        fn recip(&self) -> Self {
            JetScalar::recip(self)
        }
    }

    impl<L: Lane, const K: usize> RowAlg<K> for Order2Lane<L, K> {
        fn constant(c: f64) -> Self {
            Order2Lane::constant(L::splat(c))
        }
        fn add(&self, o: &Self) -> Self {
            Order2Lane::add(self, o)
        }
        fn sub(&self, o: &Self) -> Self {
            Order2Lane::sub(self, o)
        }
        fn mul(&self, o: &Self) -> Self {
            Order2Lane::mul(self, o)
        }
        fn scale(&self, s: f64) -> Self {
            Order2Lane::scale(self, s)
        }
        fn exp(&self) -> Self {
            Order2Lane::exp(self)
        }
        fn sqrt(&self) -> Self {
            Order2Lane::sqrt(self)
        }
        fn recip(&self) -> Self {
            Order2Lane::recip(self)
        }
    }

    /// A dense witness row expression touching every algebra op (mul, add, sub,
    /// scale, exp, sqrt, recip) over ALL `K` primaries, so the gradient and the
    /// full `K×K` Hessian are dense (no trivially-zero channel). All transcend.
    /// arguments are kept finite/positive: `sqrt(s²+1) > 0`, `recip(exp+2) > 0`.
    fn row_expr<const K: usize, A: RowAlg<K>>(p: &[A; K]) -> A {
        let mut s = A::constant(0.3);
        for a in 0..K {
            let b = (a + 1) % K;
            s = s.add(&p[a].mul(&p[b]).scale(0.1 + 0.05 * a as f64));
        }
        let e = s.exp();
        let r = s.mul(&s).add(&A::constant(1.0)).sqrt();
        let denom = e.add(&A::constant(2.0));
        e.mul(&r).sub(&s.scale(0.5)).mul(&denom.recip())
    }

    /// xorshift64 → `f64` in `[-1, 1)`.
    fn rand_unit(state: &mut u64) -> f64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        let u = (x >> 11) as f64 / ((1u64 << 53) as f64); // [0, 1)
        2.0 * u - 1.0
    }

    /// Returns the number of (batch, row) pairs whose every channel was
    /// verified bit-identical, so the caller can assert the expected total ran.
    fn check_k<const K: usize>(state: &mut u64, batches: usize) -> usize {
        let mut verified_rows = 0usize;
        for _ in 0..batches {
            // Four independent rows of K primary values.
            let rows: [[f64; K]; 4] =
                std::array::from_fn(|_| std::array::from_fn(|_| rand_unit(state)));

            // Production ground truth, evaluated per row at Order2<K>.
            let prod: [Order2<K>; 4] = std::array::from_fn(|r| {
                let p: [Order2<K>; K] = std::array::from_fn(|a| Order2::variable(rows[r][a], a));
                row_expr(&p)
            });

            // New scalar field (Order2Lane<f64>), per row.
            let scal: [Order2Lane<f64, K>; 4] = std::array::from_fn(|r| {
                let p: [Order2Lane<f64, K>; K] =
                    std::array::from_fn(|a| Order2Lane::variable(rows[r][a], a));
                row_expr(&p)
            });

            // Batched: 4 rows packed into f64x4 lanes, ONE vector pass.
            let pbatch: [Order2Batch<K>; K] = std::array::from_fn(|a| {
                let packed = wide::f64x4::new([rows[0][a], rows[1][a], rows[2][a], rows[3][a]]);
                Order2Batch::variable(packed, a)
            });
            let batch = row_expr(&pbatch);

            for r in 0..4 {
                let g = prod[r].0;
                // Order2Lane<f64> == Order2<K> (bit-identical scalar field).
                assert_eq!(scal[r].v.to_bits(), g.v.to_bits(), "K={K} scalar v");
                // Batch lane r == Order2<K> for row r.
                let lr = batch.lane(r).0;
                assert_eq!(lr.v.to_bits(), g.v.to_bits(), "K={K} batch lane {r} v");
                for a in 0..K {
                    assert_eq!(
                        scal[r].g[a].to_bits(),
                        g.g[a].to_bits(),
                        "K={K} scalar g[{a}]"
                    );
                    assert_eq!(
                        lr.g[a].to_bits(),
                        g.g[a].to_bits(),
                        "K={K} batch lane {r} g[{a}]"
                    );
                    for b in 0..K {
                        assert_eq!(
                            scal[r].h[a][b].to_bits(),
                            g.h[a][b].to_bits(),
                            "K={K} scalar h[{a}][{b}]"
                        );
                        assert_eq!(
                            lr.h[a][b].to_bits(),
                            g.h[a][b].to_bits(),
                            "K={K} batch lane {r} h[{a}][{b}]"
                        );
                    }
                }
                verified_rows += 1;
            }
        }
        verified_rows
    }

    /// ≥2000 random 4-row batches per K, across K ∈ {2,3,4,9}: every channel of
    /// every lane is `to_bits`-identical to the production scalar per row.
    #[test]
    fn batch_lanes_bit_identical_to_scalar_per_row() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut verified = 0usize;
        verified += check_k::<2>(&mut state, 2000);
        verified += check_k::<3>(&mut state, 2000);
        verified += check_k::<4>(&mut state, 2000);
        verified += check_k::<9>(&mut state, 2000);
        // 4 K-values × 2000 batches × 4 packed rows each, all bit-identical.
        assert_eq!(verified, 4 * 2000 * 4, "every batch row must be verified");
    }

    // ── One-/two-seed lane oracles ──────────────────────────────────────────
    //
    // The same dense `row_expr` witness program runs over the SEEDED directional
    // scalars: the scalar `OneSeed`/`TwoSeed` per row, the `f64`-lane re-type
    // (`*SeedLane<f64>`), and the 4-rows-per-pass batch (`*SeedBatch`). The
    // headline claim is that the contracted-third / contracted-fourth channel of
    // every lane is `to_bits`-identical to the production scalar's per row.

    impl<const K: usize> RowAlg<K> for OneSeed<K> {
        fn constant(c: f64) -> Self {
            <Self as JetScalar<K>>::constant(c)
        }
        fn add(&self, o: &Self) -> Self {
            crate::nested_dual::JetField::add(self, o)
        }
        fn sub(&self, o: &Self) -> Self {
            crate::nested_dual::JetField::sub(self, o)
        }
        fn mul(&self, o: &Self) -> Self {
            crate::nested_dual::JetField::mul(self, o)
        }
        fn scale(&self, s: f64) -> Self {
            crate::nested_dual::JetField::scale(self, s)
        }
        fn exp(&self) -> Self {
            JetScalar::exp(self)
        }
        fn sqrt(&self) -> Self {
            JetScalar::sqrt(self)
        }
        fn recip(&self) -> Self {
            JetScalar::recip(self)
        }
    }

    impl<L: Lane, const K: usize> RowAlg<K> for OneSeedLane<L, K> {
        fn constant(c: f64) -> Self {
            OneSeedLane::constant(L::splat(c))
        }
        fn add(&self, o: &Self) -> Self {
            OneSeedLane::add(self, o)
        }
        fn sub(&self, o: &Self) -> Self {
            OneSeedLane::sub(self, o)
        }
        fn mul(&self, o: &Self) -> Self {
            OneSeedLane::mul(self, o)
        }
        fn scale(&self, s: f64) -> Self {
            OneSeedLane::scale(self, s)
        }
        fn exp(&self) -> Self {
            OneSeedLane::exp(self)
        }
        fn sqrt(&self) -> Self {
            OneSeedLane::sqrt(self)
        }
        fn recip(&self) -> Self {
            OneSeedLane::recip(self)
        }
    }

    impl<const K: usize> RowAlg<K> for TwoSeed<K> {
        fn constant(c: f64) -> Self {
            <Self as JetScalar<K>>::constant(c)
        }
        fn add(&self, o: &Self) -> Self {
            crate::nested_dual::JetField::add(self, o)
        }
        fn sub(&self, o: &Self) -> Self {
            crate::nested_dual::JetField::sub(self, o)
        }
        fn mul(&self, o: &Self) -> Self {
            crate::nested_dual::JetField::mul(self, o)
        }
        fn scale(&self, s: f64) -> Self {
            crate::nested_dual::JetField::scale(self, s)
        }
        fn exp(&self) -> Self {
            JetScalar::exp(self)
        }
        fn sqrt(&self) -> Self {
            JetScalar::sqrt(self)
        }
        fn recip(&self) -> Self {
            JetScalar::recip(self)
        }
    }

    impl<L: Lane, const K: usize> RowAlg<K> for TwoSeedLane<L, K> {
        fn constant(c: f64) -> Self {
            TwoSeedLane::constant(L::splat(c))
        }
        fn add(&self, o: &Self) -> Self {
            TwoSeedLane::add(self, o)
        }
        fn sub(&self, o: &Self) -> Self {
            TwoSeedLane::sub(self, o)
        }
        fn mul(&self, o: &Self) -> Self {
            TwoSeedLane::mul(self, o)
        }
        fn scale(&self, s: f64) -> Self {
            TwoSeedLane::scale(self, s)
        }
        fn exp(&self) -> Self {
            TwoSeedLane::exp(self)
        }
        fn sqrt(&self) -> Self {
            TwoSeedLane::sqrt(self)
        }
        fn recip(&self) -> Self {
            TwoSeedLane::recip(self)
        }
    }

    fn check_oneseed<const K: usize>(state: &mut u64, batches: usize) -> usize {
        let mut rows_checked = 0;
        for _ in 0..batches {
            let rows: [[f64; K]; 4] =
                std::array::from_fn(|_| std::array::from_fn(|_| rand_unit(state)));
            // Per-row ε-direction.
            let u: [[f64; K]; 4] =
                std::array::from_fn(|_| std::array::from_fn(|_| rand_unit(state)));

            // Production ground truth (scalar OneSeed per row).
            let prod: [OneSeed<K>; 4] = std::array::from_fn(|r| {
                let p: [OneSeed<K>; K] =
                    std::array::from_fn(|a| OneSeed::seed_direction(rows[r][a], a, u[r][a]));
                row_expr(&p)
            });

            // f64-lane re-type per row.
            let scal: [OneSeedLane<f64, K>; 4] = std::array::from_fn(|r| {
                let p: [OneSeedLane<f64, K>; K] =
                    std::array::from_fn(|a| OneSeedLane::seed_direction(rows[r][a], a, u[r][a]));
                row_expr(&p)
            });

            // 4-rows-per-pass batch.
            let pbatch: [OneSeedBatch<K>; K] = std::array::from_fn(|a| {
                let val = wide::f64x4::new([rows[0][a], rows[1][a], rows[2][a], rows[3][a]]);
                let uu = wide::f64x4::new([u[0][a], u[1][a], u[2][a], u[3][a]]);
                OneSeedBatch::seed_direction(val, a, uu)
            });
            let batch = row_expr(&pbatch);

            for r in 0..4 {
                let want = prod[r].contracted_third();
                let got_scal = scal[r].contracted_third();
                let got_batch = batch.lane(r).contracted_third();
                // Value channel too (sanity that the base program agrees).
                assert_eq!(
                    scal[r].base.v.to_bits(),
                    prod[r].base.value().to_bits(),
                    "OneSeed K={K} scalar value"
                );
                assert_eq!(
                    batch.lane(r).base.value().to_bits(),
                    prod[r].base.value().to_bits(),
                    "OneSeed K={K} batch lane {r} value"
                );
                for a in 0..K {
                    for b in 0..K {
                        assert_eq!(
                            got_scal[a][b].to_bits(),
                            want[a][b].to_bits(),
                            "OneSeed K={K} scalar third[{a}][{b}]"
                        );
                        assert_eq!(
                            got_batch[a][b].to_bits(),
                            want[a][b].to_bits(),
                            "OneSeed K={K} batch lane {r} third[{a}][{b}]"
                        );
                    }
                }
                rows_checked += 1;
            }
        }
        rows_checked
    }

    fn check_twoseed<const K: usize>(state: &mut u64, batches: usize) -> usize {
        let mut rows_checked = 0;
        for _ in 0..batches {
            let rows: [[f64; K]; 4] =
                std::array::from_fn(|_| std::array::from_fn(|_| rand_unit(state)));
            let u: [[f64; K]; 4] =
                std::array::from_fn(|_| std::array::from_fn(|_| rand_unit(state)));
            let v: [[f64; K]; 4] =
                std::array::from_fn(|_| std::array::from_fn(|_| rand_unit(state)));

            let prod: [TwoSeed<K>; 4] = std::array::from_fn(|r| {
                let p: [TwoSeed<K>; K] =
                    std::array::from_fn(|a| TwoSeed::seed(rows[r][a], a, u[r][a], v[r][a]));
                row_expr(&p)
            });

            let scal: [TwoSeedLane<f64, K>; 4] = std::array::from_fn(|r| {
                let p: [TwoSeedLane<f64, K>; K] =
                    std::array::from_fn(|a| TwoSeedLane::seed(rows[r][a], a, u[r][a], v[r][a]));
                row_expr(&p)
            });

            let pbatch: [TwoSeedBatch<K>; K] = std::array::from_fn(|a| {
                let val = wide::f64x4::new([rows[0][a], rows[1][a], rows[2][a], rows[3][a]]);
                let uu = wide::f64x4::new([u[0][a], u[1][a], u[2][a], u[3][a]]);
                let vv = wide::f64x4::new([v[0][a], v[1][a], v[2][a], v[3][a]]);
                TwoSeedBatch::seed(val, a, uu, vv)
            });
            let batch = row_expr(&pbatch);

            for r in 0..4 {
                let want = prod[r].contracted_fourth();
                let got_scal = scal[r].contracted_fourth();
                let got_batch = batch.lane(r).contracted_fourth();
                assert_eq!(
                    scal[r].base.v.to_bits(),
                    prod[r].base.value().to_bits(),
                    "TwoSeed K={K} scalar value"
                );
                assert_eq!(
                    batch.lane(r).base.value().to_bits(),
                    prod[r].base.value().to_bits(),
                    "TwoSeed K={K} batch lane {r} value"
                );
                for a in 0..K {
                    for b in 0..K {
                        assert_eq!(
                            got_scal[a][b].to_bits(),
                            want[a][b].to_bits(),
                            "TwoSeed K={K} scalar fourth[{a}][{b}]"
                        );
                        assert_eq!(
                            got_batch[a][b].to_bits(),
                            want[a][b].to_bits(),
                            "TwoSeed K={K} batch lane {r} fourth[{a}][{b}]"
                        );
                    }
                }
                rows_checked += 1;
            }
        }
        rows_checked
    }

    /// ≥2000 random 4-row batches per K, across K ∈ {2,3,4,9}: the
    /// contracted-third channel of every `OneSeedLane` lane is `to_bits`-identical
    /// to the production [`OneSeed`] per row.
    #[test]
    fn oneseed_lanes_contracted_third_bit_identical() {
        let mut state = 0x1234_5678_9ABC_DEF0_u64;
        let batches = 2000;
        let rows_checked = check_oneseed::<2>(&mut state, batches)
            + check_oneseed::<3>(&mut state, batches)
            + check_oneseed::<4>(&mut state, batches)
            + check_oneseed::<9>(&mut state, batches);
        // 4 widths × `batches` batches × 4 rows each: a silently empty inner
        // loop would leave this at zero instead of passing as a no-op.
        assert_eq!(rows_checked, 4 * batches * 4);
    }

    /// ≥2000 random 4-row batches per K, across K ∈ {2,3,4,9}: the
    /// contracted-fourth channel of every `TwoSeedLane` lane is `to_bits`-identical
    /// to the production [`TwoSeed`] per row.
    #[test]
    fn twoseed_lanes_contracted_fourth_bit_identical() {
        let mut state = 0x0FED_CBA9_8765_4321_u64;
        let batches = 2000;
        let rows_checked = check_twoseed::<2>(&mut state, batches)
            + check_twoseed::<3>(&mut state, batches)
            + check_twoseed::<4>(&mut state, batches)
            + check_twoseed::<9>(&mut state, batches);
        // 4 widths × `batches` batches × 4 rows each: a silently empty inner
        // loop would leave this at zero instead of passing as a no-op.
        assert_eq!(rows_checked, 4 * batches * 4);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{JetScalar, Order1, Order2, filtered_implicit_solve_scalar};
    use crate::nested_dual::JetField;

    // ── Order2 direct property tests ─────────────────────────────────────────

    /// `Order2::constant(c)` carries value `c` and zero everywhere else.
    #[test]
    fn order2_constant_has_zero_derivatives() {
        let s = Order2::<3>::constant(7.5);
        assert_eq!(s.value(), 7.5);
        for a in 0..3 {
            assert_eq!(s.g()[a], 0.0, "grad[{a}] should be zero");
            for b in 0..3 {
                assert_eq!(s.h()[a][b], 0.0, "hess[{a}][{b}] should be zero");
            }
        }
    }

    /// `Order2::variable(x, axis)` has unit gradient in slot `axis` and zero Hessian.
    #[test]
    fn order2_variable_has_unit_gradient_in_seeded_slot() {
        let x = -2.5_f64;
        let s = Order2::<4>::variable(x, 2);
        assert_eq!(s.value(), x);
        for a in 0..4 {
            let expected_g = if a == 2 { 1.0 } else { 0.0 };
            assert_eq!(s.g()[a], expected_g, "grad[{a}]");
            for b in 0..4 {
                assert_eq!(s.h()[a][b], 0.0, "hess[{a}][{b}] should be zero");
            }
        }
    }

    /// `Order2::add` sums gradient channels; `sub` is the inverse on gradients.
    /// Uses integer-valued primaries so the value roundtrip is also exact.
    #[test]
    fn order2_add_sub_roundtrip() {
        let p = Order2::<2>::variable(3.0, 0);
        let q = Order2::<2>::variable(2.0, 1);
        let pq = crate::nested_dual::JetField::add(&p, &q);
        // value = 3 + 2 = 5
        assert_eq!(pq.value(), 5.0, "add value");
        let back = crate::nested_dual::JetField::sub(&pq, &q);
        // (p + q) - q gradient should equal p's gradient exactly
        for a in 0..2 {
            assert_eq!(back.g()[a], p.g()[a], "grad[{a}] roundtrip");
        }
    }

    /// `Order2::mul` of two variables satisfies the Leibniz product rule:
    ///   ∂(p·q)/∂p = q,  ∂(p·q)/∂q = p,  ∂²(p·q)/∂p∂q = 1.
    #[test]
    fn order2_mul_satisfies_leibniz_rule() {
        let pv = 3.0_f64;
        let qv = -2.0_f64;
        let p = Order2::<2>::variable(pv, 0);
        let q = Order2::<2>::variable(qv, 1);
        let pq = crate::nested_dual::JetField::mul(&p, &q);
        assert_eq!(pq.value(), pv * qv, "value = p·q");
        assert_eq!(pq.g()[0], qv, "∂(p·q)/∂p = q");
        assert_eq!(pq.g()[1], pv, "∂(p·q)/∂q = p");
        assert_eq!(pq.h()[0][1], 1.0, "∂²(p·q)/∂p∂q = 1");
        assert_eq!(pq.h()[1][0], 1.0, "∂²(p·q)/∂q∂p = 1 (symmetric)");
        assert_eq!(pq.h()[0][0], 0.0, "∂²(p·q)/∂p² = 0");
        assert_eq!(pq.h()[1][1], 0.0, "∂²(p·q)/∂q² = 0");
    }

    /// `Order2::scale(s)` multiplies every channel by `s`.
    #[test]
    fn order2_scale_multiplies_all_channels() {
        let p = Order2::<2>::variable(4.0, 0);
        let s = 2.5_f64;
        let ps = crate::nested_dual::JetField::scale(&p, s);
        assert_eq!(ps.value(), 4.0 * s);
        assert_eq!(ps.g()[0], 1.0 * s);
        assert_eq!(ps.g()[1], 0.0);
    }

    /// `Order2::exp` at a constant has value `e^c`, gradient `e^c * g`, Hessian `e^c * (g⊗g + H)`.
    /// At a seeded variable `p₀`, the first derivative is `e^{p₀}` and second is `e^{p₀}`.
    #[test]
    fn order2_exp_derivative_stack_correct() {
        let p0 = 1.0_f64;
        let p = Order2::<1>::variable(p0, 0);
        let ep = JetScalar::exp(&p);
        let e = p0.exp();
        assert!((ep.value() - e).abs() < 1e-15, "exp value");
        assert!((ep.g()[0] - e).abs() < 1e-15, "d/dp exp(p) = exp(p)");
        assert!((ep.h()[0][0] - e).abs() < 1e-15, "d²/dp² exp(p) = exp(p)");
    }

    /// `Order2::ln` at a seeded variable: d/dp ln(p) = 1/p, d²/dp² ln(p) = -1/p².
    #[test]
    fn order2_ln_derivative_stack_correct() {
        let p0 = 2.0_f64;
        let p = Order2::<1>::variable(p0, 0);
        let lnp = JetScalar::ln(&p);
        assert!((lnp.value() - p0.ln()).abs() < 1e-15, "ln value");
        assert!((lnp.g()[0] - 1.0 / p0).abs() < 1e-15, "d/dp ln(p) = 1/p");
        assert!(
            (lnp.h()[0][0] - (-1.0 / (p0 * p0))).abs() < 1e-15,
            "d²/dp² ln(p) = -1/p²"
        );
    }

    /// `exp` and `ln` are mutual inverses: `ln(exp(p)).value() == p` at the scalar.
    #[test]
    fn order2_exp_ln_roundtrip_at_value() {
        let p0 = 0.8_f64;
        let p = Order2::<1>::variable(p0, 0);
        let roundtrip = JetScalar::ln(&JetScalar::exp(&p));
        assert!((roundtrip.value() - p0).abs() < 1e-14, "ln(exp(p)) ≈ p");
    }

    // ── Order1 tests ─────────────────────────────────────────────────────────

    /// `Order1::constant` carries the correct value with all-zero gradient.
    #[test]
    fn order1_constant_has_zero_gradient() {
        let s = Order1::<3>::constant(-5.0);
        assert_eq!(s.value(), -5.0);
        for a in 0..3 {
            assert_eq!(s.g()[a], 0.0, "g[{a}] should be zero");
        }
    }

    /// `Order1::variable(x, axis)` has unit gradient only in `axis`.
    #[test]
    fn order1_variable_has_unit_gradient_in_seeded_slot() {
        let s = Order1::<3>::variable(2.0, 1);
        assert_eq!(s.value(), 2.0);
        assert_eq!(s.g()[0], 0.0);
        assert_eq!(s.g()[1], 1.0);
        assert_eq!(s.g()[2], 0.0);
    }

    /// `Order1::mul` satisfies the product rule (value and gradient, no Hessian).
    #[test]
    fn order1_mul_satisfies_product_rule() {
        let pv = 3.0_f64;
        let qv = -2.0_f64;
        let p = Order1::<2>::variable(pv, 0);
        let q = Order1::<2>::variable(qv, 1);
        let pq = crate::nested_dual::JetField::mul(&p, &q);
        assert_eq!(pq.value(), pv * qv);
        assert_eq!(pq.g()[0], qv, "∂(p·q)/∂p = q");
        assert_eq!(pq.g()[1], pv, "∂(p·q)/∂q = p");
    }

    /// `Order1::exp` carries the correct value and gradient `e^{p₀}`.
    #[test]
    fn order1_exp_has_correct_value_and_gradient() {
        let p0 = 0.5_f64;
        let p = Order1::<2>::variable(p0, 0);
        let ep = JetScalar::exp(&p);
        let e = p0.exp();
        assert!((ep.value() - e).abs() < 1e-15, "exp value");
        assert!((ep.g()[0] - e).abs() < 1e-15, "d/dp exp(p)");
        assert_eq!(ep.g()[1], 0.0, "irrelevant gradient slot is zero");
    }

    /// `Order1` and `Order2` agree on value and gradient for the same expression.
    #[test]
    fn order1_and_order2_agree_on_value_and_gradient() {
        let p0 = 1.3_f64;
        let q0 = -0.7_f64;
        // evaluate (p * q + p).exp() at (p0, q0)
        let p1 = Order1::<2>::variable(p0, 0);
        let q1 = Order1::<2>::variable(q0, 1);
        let expr1 = JetScalar::exp(&crate::nested_dual::JetField::add(
            &crate::nested_dual::JetField::mul(&p1, &q1),
            &p1,
        ));

        let p2 = Order2::<2>::variable(p0, 0);
        let q2 = Order2::<2>::variable(q0, 1);
        let expr2 = JetScalar::exp(&crate::nested_dual::JetField::add(
            &crate::nested_dual::JetField::mul(&p2, &q2),
            &p2,
        ));

        assert!(
            (expr1.value() - expr2.value()).abs() < 1e-14,
            "value mismatch"
        );
        for a in 0..2 {
            assert!(
                (expr1.g()[a] - expr2.g()[a]).abs() < 1e-14,
                "gradient[{a}] mismatch"
            );
        }
    }

    // ── filtered_implicit_solve_scalar ────────────────────────────────────────

    /// Lift the trivial linear constraint F(a, θ) = a - θ = 0 through `Order2<1>`.
    /// The exact lifted jet is a(θ) = θ, so value=θ₀, gradient=1.
    #[test]
    fn filtered_implicit_solve_linear_constraint_gives_exact_jet() {
        let theta0 = 3.0_f64;
        let theta = Order2::<1>::variable(theta0, 0);
        // a0 = theta0, F_a = 1, inv_fa = 1; 2 iters suffice for Order2.
        let a = filtered_implicit_solve_scalar::<1, Order2<1>>(theta0, 1.0, 2, |a_jet| {
            crate::nested_dual::JetField::sub(a_jet, &theta)
        });
        assert!((a.value() - theta0).abs() < 1e-14, "value = theta0");
        // da/dtheta = 1 (identity)
        assert!((a.g()[0] - 1.0).abs() < 1e-14, "gradient = 1");
        // d²a/dtheta² = 0 (linear)
        assert!(a.h()[0][0].abs() < 1e-14, "hessian = 0");
    }

    /// `filtered_implicit_solve_scalar` on a quadratic constraint F(a,θ)=a²-θ=0
    /// with primal root a₀=√θ₀, giving da/dθ = 1/(2√θ₀), d²a/dθ² = -1/(4θ₀^{3/2}).
    #[test]
    fn filtered_implicit_solve_quadratic_constraint_matches_analytic_derivatives() {
        let theta0 = 4.0_f64;
        let a0 = theta0.sqrt();
        let inv_fa = 1.0 / (2.0 * a0);
        let theta = Order2::<1>::variable(theta0, 0);
        // F(a,theta) = a*a - theta
        let a = filtered_implicit_solve_scalar::<1, Order2<1>>(a0, inv_fa, 2, |a_jet| {
            let aa = crate::nested_dual::JetField::mul(a_jet, a_jet);
            crate::nested_dual::JetField::sub(&aa, &theta)
        });
        let tol = 1e-12;
        assert!((a.value() - a0).abs() < tol, "value = sqrt(theta0)");
        let expected_g = 0.5 / a0;
        assert!(
            (a.g()[0] - expected_g).abs() < tol,
            "da/dtheta = 1/(2*sqrt)"
        );
        let expected_h = -0.25 / (theta0 * a0);
        assert!(
            (a.h()[0][0] - expected_h).abs() < tol,
            "d2a/dtheta2 = -1/(4*theta^1.5)"
        );
    }
}

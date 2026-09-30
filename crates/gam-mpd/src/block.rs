//! Native linear reads and RMSNorm evaluations with forward-error radii (#2951).
//!
//! [`linear_read`] is the native read `x Wᵀ` of rows that carry a radius against their exact
//! values, with apply.rs's kernel and its rounding band; [`rms_norm_band`] is the rounding band
//! of the source's RMSNorm. Each computed bound adds the underflow its own products could have
//! lost and is divided by the factor its own arithmetic could have lost, every operation rounded
//! up, so a stack of layers ending in a [`linear_read`] carries one radius from its input rows to
//! its logits. Every magnitude is computed one output column at a time, so no `|A|` is formed.

use super::apply::{ApplyError, native_linear};
use super::attention::ProjectedRows;
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use gam_runtime::resource::{Governed, MemoryGovernor};
use ndarray::{Array2, ArrayView1, ArrayView2, Zip};
use std::fmt;

/// binary64's subnormal spacing `2^-1074`. A product or quotient whose result rounds into
/// the subnormal range moves by an absolute amount of at most half of it, not relatively;
/// additions and subtractions are exact there. Half the spacing, `2^-1075`, is not a
/// binary64 number (`f64::MIN_POSITIVE * UNIT_ROUNDOFF` rounds to zero), so a band carries
/// the whole spacing.
pub const SUBNORMAL_SPACING: f64 = f64::MIN_POSITIVE * f64::EPSILON;

/// `value` stepped one float up: an upper bound on the exact result of an operation that
/// rounded to nearest to `value`.
pub(super) fn up(value: f64) -> f64 {
    value.next_up()
}

/// `value` stepped one float down, clamped at zero: a lower bound on a nonnegative exact
/// result that rounded to nearest to `value`.
pub(super) fn down(value: f64) -> f64 {
    value.next_down().max(0.0)
}

/// An upper bound on the absolute underflow of one normalized entry `fl(w fl(x ν̂))` beyond
/// its relative rounding: `|w| 2^-1075 (1 + u)` from `x ν̂` and `2^-1075` from the gain's
/// product, carried as `|w| 2^-1074 + 2^-1074`, which also covers a following bias
/// addition's `1 + u`.
fn underflow_reach(weight: f64) -> f64 {
    up(up(weight.abs() * SUBNORMAL_SPACING) + SUBNORMAL_SPACING)
}

/// The relative growth `λ` of one normalized entry `fl(w fl(x ν̂))` of the row `values`
/// (`x`, `d` wide) against its exact value `w x ν` beyond [`underflow_reach`], with
/// `operations = k` relative roundings on the entry's path, or `None` when no finite `λ < 1`
/// bounds it.
///
/// The computed argument of the square root is `X (1 + θ) + A`, with `X = mean(x²) + ε`,
/// `|θ| ≤ γ_(d+2)` and `|A| ≤ 2^-1074 (1 + γ_(d+1))`: each square and the mean's division
/// round by at most `2^-1075` absolute into the subnormal range, while the sum and `+ ε`
/// stay relative. With `X ≥ X₀ = max(ε, max_j x_j² / d)`, the absolute part is a relative
/// `α`, `|α| ≤ ω = 2 · 2^-1074 / (X₀ (1 − γ_k))`, which moves `ν̂` by `(1 + α)^(-1/2)`, within
/// `ρ = ω / (2 (1 − ω))` of one for `ω < 1`: `(1 − ω)^(-1/2) − 1 = ω / (√(1 − ω) (1 + √(1 − ω)))`
/// and `1 − ω ≤ √(1 − ω)`. So `λ = (1 + γ_k)(1 + ρ) − 1 = γ_k + ρ + γ_k ρ`. A row with `ε = 0`
/// whose mean square is not resolved above the underflow (`ω ≥ 1`) has no bound. Every
/// operation rounds to nearest and then steps one float up (down where it divides), so `λ`
/// is an upper bound computed in binary64.
fn normalization_growth(values: ArrayView1<'_, f64>, epsilon: f64, operations: usize) -> Option<f64> {
    let largest = values.iter().fold(0.0_f64, |largest, value| largest.max(value.abs()));
    let floor = epsilon.max(down(down(largest * largest) / values.len() as f64));
    let growth = up(accumulation_growth(operations));
    let omega = up(2.0 * SUBNORMAL_SPACING / down(floor * down(1.0 - growth)));
    if !(omega < 1.0) {
        return None;
    }
    let rho = up(omega / down(2.0 * down(1.0 - omega)));
    let lambda = up(up(growth + rho) + up(growth * rho));
    (lambda < 1.0).then_some(lambda)
}

/// The rounding band of one binary64 evaluation of an RMSNorm, `MaskedNorm::Rms`'s program
/// `fl(w_j fl(x_j ν̂))`, against the exact RMSNorm `y = w x (mean(x²) + ε)^(-1/2)` of the input
/// rows `inputs` (`d` wide, with the gain `gain`), from its computed rows `normalized`.
///
/// The squares, `d − 1` additions, the mean, `+ ε`, the square root, the reciprocal and the
/// products with the row and the gain are `γ_(d+6)` relative while their results stay
/// normal. A square, the mean or a product whose result is subnormal rounds by an absolute
/// `2^-1075` instead: through `ν̂` that is the relative `ρ` of `normalization_growth`, and on
/// the entry it is the absolute `a_j = |w_j| 2^-1074 + 2^-1074` of `underflow_reach`. So
/// `|ŷ − y| ≤ λ |y| + a` with `λ = (1 + γ_(d+6))(1 + ρ) − 1`, and `|y| ≤ (|ŷ| + a)/(1 − λ)`,
/// so the band is `λ (|ŷ| + a)/(1 − λ) + a`. Every operation rounds to nearest and then steps
/// one float up (down where it divides), so the band is an upper bound computed in binary64.
///
/// `inputs` and `normalized` share a shape and `gain` is as wide; a row with `ε = 0` whose
/// mean square is not resolved above binary64's underflow has no band and is refused.
pub fn rms_norm_band(
    epsilon: f64,
    gain: ArrayView1<'_, f64>,
    inputs: ArrayView2<'_, f64>,
    normalized: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, NormBandUnbounded> {
    let width = inputs.ncols();
    let mut band = Array2::<f64>::zeros(normalized.raw_dim());
    for (row, (values, (computed, mut band_row))) in inputs
        .rows()
        .into_iter()
        .zip(normalized.rows().into_iter().zip(band.rows_mut()))
        .enumerate()
    {
        let growth = normalization_growth(values, epsilon, width + 6).ok_or(NormBandUnbounded { row })?;
        let complement = down(1.0 - growth);
        Zip::from(&mut band_row)
            .and(gain)
            .and(computed)
            .for_each(|slot, &weight, &value| {
                let absolute = underflow_reach(weight);
                *slot = up(up(up(growth * up(value.abs() + absolute)) / complement) + absolute);
            });
    }
    Ok(band)
}

/// A normalization row whose rounding band has no finite bound: a row with `ε = 0` whose mean
/// square is not resolved above binary64's underflow ([`rms_norm_band`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormBandUnbounded {
    pub row: usize,
}

impl fmt::Display for NormBandUnbounded {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "normalization row {} with zero epsilon has no finite rounding band: its mean square is not \
             resolved above binary64's underflow, or its centring error is too large for a slope bound",
            self.row
        )
    }
}

/// `|M| r` for nonnegative rows `r` (`tokens × n`) and a matrix `M` (`outputs × n`), one
/// output column at a time, so `|M|` is never formed.
fn abs_map(matrix: ArrayView2<'_, f64>, rows: ArrayView2<'_, f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((rows.nrows(), matrix.nrows()));
    for (column, entries) in matrix.outer_iter().enumerate() {
        out.column_mut(column).assign(&rows.dot(&entries.mapv(f64::abs)));
    }
    out
}

/// `|A| r` for nonnegative rows `r`, with `A` the map a read applies to each row, the rounded
/// operations of that read's product (`n`) and of this magnitude (`k`), per row, and the
/// read's underflow reach per entry.
///
/// Rounding in binary64 is `fl(a ∘ b) = (a ∘ b)(1 + δ) + η` with `|δ| ≤ u`, where `|η| ≤ 2^-1075`
/// for a product or quotient that rounds into the subnormal range and `η = 0` for a sum, which is
/// exact there (a fused multiply-add rounds once and adds one `η`). Higham's `γ_n |A| |x|` covers
/// the `δ` terms only. Each product's `η` passes the later sums with a factor of at most
/// `1 + γ_n ≤ 2`, and a later product scales an earlier stage's `η` by the magnitude it
/// multiplies. So the entry's absolute term is at most `𝒜 · 2^-1074`, with the allowance `𝒜` of
/// [`native_magnitude`]. The magnitude program runs the read's products on nonnegative operands,
/// so `𝒜` bounds its underflow as well. `underflow` holds `𝒜 · 2^-1074`, rounded up.
struct ReadMagnitude {
    magnitude: Array2<f64>,
    underflow: Array2<f64>,
    read_operations: Vec<usize>,
    magnitude_operations: Vec<usize>,
}

impl ReadMagnitude {
    /// The magnitude itself as a bound on `|A| r`. Its `k` operations on nonnegative terms each
    /// scale by `1 + δ`, `|δ| ≤ u`, and its products add at most the underflow reach `a`, so
    /// `M̂ ≥ (1 − γ_k) |A| r − a` and the exact value is at most `(M̂ + a) / (1 − γ_k)`. Every
    /// operation of that bound rounds to nearest and steps one float up (down for the
    /// divisor), so a magnitude that underflowed to zero still carries `a`.
    fn dominated(self) -> Array2<f64> {
        dominate(self.magnitude, self.underflow.view(), &self.magnitude_operations)
    }

    /// `|fl(A x) − A x| ≤ γ_n |A| |x| + a` per entry: the forward error of a product with `n`
    /// rounded operations per entry in any summation order (Higham ASNA Lemma 3.1), plus the
    /// read's underflow reach `a`, with `|A| |x|` bounded as in [`Self::dominated`] from a
    /// magnitude of `k` operations. Evaluated with every operation stepped one float up, so a
    /// subnormal band does not round to zero.
    fn rounding(self) -> Array2<f64> {
        let mut band = dominate(self.magnitude, self.underflow.view(), &self.magnitude_operations);
        for ((mut row, reach), &read) in band
            .outer_iter_mut()
            .zip(self.underflow.outer_iter())
            .zip(&self.read_operations)
        {
            let growth = up(accumulation_growth(read));
            Zip::from(&mut row)
                .and(&reach)
                .for_each(|value, &reach| *value = up(up(growth * *value) + reach));
        }
        band
    }
}

/// `(M̂ + a) / (1 − γ_k)` per entry, each operation rounded to nearest and stepped one float up
/// (down for the divisor): the upper bound [`ReadMagnitude::dominated`] derives.
fn dominate(magnitude: Array2<f64>, underflow: ArrayView2<'_, f64>, operations: &[usize]) -> Array2<f64> {
    let mut bound = magnitude;
    for ((mut row, reach), &operations) in bound.outer_iter_mut().zip(underflow.outer_iter()).zip(operations) {
        let dominance = down(1.0 - up(accumulation_growth(operations)));
        Zip::from(&mut row)
            .and(&reach)
            .for_each(|value, &reach| *value = up(up(*value + reach) / dominance));
    }
    bound
}

/// `a + b` for two nonnegative bounds, rounded up: the exact sum is at most the computed
/// one divided by `1 − u`.
fn sum_of_bounds(first: Array2<f64>, second: Array2<f64>) -> Array2<f64> {
    (first + &second).mapv(|value| value / (1.0 - UNIT_ROUNDOFF))
}

/// Whether rows with this radius are exact. Their input radius then carries nothing into a
/// read, and a bound plus exact zeros is the bound itself, with no addition to round.
fn is_exact(radius: ArrayView2<'_, f64>) -> bool {
    radius.iter().all(|&entry| entry == 0.0)
}

/// `|W| r` for a native read of the nonnegative rows `r`: the inner product takes `n = d`
/// rounded operations, and so does this magnitude. Its `d` products give the allowance
/// `𝒜 = d`, and `d · 2^-1074` is exact for `d < 2^53`.
fn native_magnitude(weight: ArrayView2<'_, f64>, rows: ArrayView2<'_, f64>) -> ReadMagnitude {
    let (tokens, width) = rows.dim();
    ReadMagnitude {
        magnitude: abs_map(weight, rows),
        underflow: Array2::from_elem((tokens, weight.nrows()), width as f64 * SUBNORMAL_SPACING),
        read_operations: vec![width; tokens],
        magnitude_operations: vec![width; tokens],
    }
}

/// A native linear read `x Wᵀ` of rows that carry a radius against their exact values:
/// apply.rs's `native_linear`, its rounding band `γ_d |W| |x̂| + d · 2^-1074` (the second term
/// for its products that round into the subnormal range), and `|W| r`, the most the exact read
/// can move between the computed rows and the exact ones. A stack of layers ends in such a
/// read, its unembedding, so the logits carry the whole stack's radius. Returns the read and
/// its radius against the exact read of the exact rows.
pub fn linear_read(
    governor: &MemoryGovernor,
    weight: ArrayView2<'_, f64>,
    rows: ProjectedRows<'_>,
) -> Result<(Governed<Array2<f64>>, Array2<f64>), ApplyError> {
    if rows.radius.dim() != rows.values.dim() {
        return Err(ApplyError::Shape {
            operand: "input radius",
            expected: rows.values.dim(),
            found: rows.radius.dim(),
        });
    }
    let values = native_linear(governor, weight, rows.values)?;
    let rounding = native_magnitude(weight, rows.values.mapv(f64::abs).view()).rounding();
    let radius = if is_exact(rows.radius) {
        rounding
    } else {
        sum_of_bounds(rounding, native_magnitude(weight, rows.radius).dominated())
    };
    Ok((values, radius))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::test_governor;
    use crate::gated_rewrite::MaskedNorm;
    use ndarray::Array1;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// Qwen3's declared `rms_norm_eps`.
    const RMS_EPSILON: f64 = 1.0e-6;
    const LAYER_WIDTH: usize = 8;
    const LAYER_TOKENS: usize = 6;

    /// Eighths in `[-1, 1]`.
    fn eighths(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_simple_fn((rows, cols), || rng.random_range(-8..=8) as f64 / 8.0)
    }

    fn eighths_vector(rng: &mut StdRng, len: usize) -> Array1<f64> {
        Array1::from_shape_simple_fn(len, || rng.random_range(-8..=8) as f64 / 8.0)
    }

    /// `rms_norm_band` covers the double-double RMSNorm of the rows `MaskedNorm::apply`
    /// normalized. Positive control: the computed rows are not the double-double ones, so a
    /// zero band is exceeded and the band's coverage is not vacuous.
    #[test]
    fn a_normalization_band_covers_the_double_double_normalization() {
        use qd::Quad;
        let mut rng = StdRng::seed_from_u64(2997);
        let gain = eighths_vector(&mut rng, LAYER_WIDTH);
        let rows = Array2::from_shape_simple_fn((LAYER_TOKENS, LAYER_WIDTH), || {
            rng.random_range(-48..=48) as f64 / 24.0
        });
        let quad = Quad::from_f64;
        let reference = |row: usize| -> Vec<Quad> {
            let values: Vec<Quad> = rows.row(row).iter().map(|&value| quad(value)).collect();
            let square = values.iter().fold(quad(0.0), |sum, &value| sum + value * value);
            let inverse_root = quad(1.0) / (square / quad(LAYER_WIDTH as f64) + quad(RMS_EPSILON)).sqrt();
            values
                .iter()
                .enumerate()
                .map(|(column, &value)| quad(gain[column]) * value * inverse_root)
                .collect()
        };
        let excess = |computed: &Array2<f64>, band: &Array2<f64>| {
            (0..LAYER_TOKENS)
                .map(|row| {
                    let exact = reference(row);
                    (0..LAYER_WIDTH)
                        .filter(|&column| (quad(computed[[row, column]]) - exact[column]).0.abs() > band[[row, column]])
                        .count()
                })
                .sum::<usize>()
        };
        let normalized = MaskedNorm::Rms { epsilon: RMS_EPSILON, gain: gain.view() }
            .apply(rows.view())
            .expect("finite rows");
        let band = rms_norm_band(RMS_EPSILON, gain.view(), rows.view(), normalized.view()).expect("finite rows");
        assert_eq!(excess(&normalized, &band), 0, "a row left its rounding band around the double-double RMSNorm");
        let zero = Array2::<f64>::zeros(normalized.raw_dim());
        assert!(
            excess(&normalized, &zero) > 0,
            "positive control: the computed rows must differ from the double-double normalization somewhere"
        );
    }

    /// An RMSNorm row whose product `x ν̂` falls in the subnormal range rounds it by an
    /// absolute `2^-1075`, not relatively: `[1000, 1e-310]` with gain `[1, 1e10]` has
    /// `x ν̂ ≈ 1.4e-313`, whose rounding the gain lifts to about `9e-315` on a normal output
    /// of `1.4e-303`. The band covers the double-double normalization there. Positive
    /// control: the relative band alone, `γ_(d+6) |ŷ| / (1 − γ_(d+6))`, about `1e-318`, is
    /// exceeded. A zero-epsilon row whose mean square is the smallest subnormal has no
    /// band and is refused, typed, though the owner normalizes it.
    #[test]
    fn a_normalization_band_covers_a_row_whose_products_underflow() {
        use qd::Quad;
        let quad = Quad::from_f64;
        let rows = ndarray::array![[1000.0, 1.0e-310]];
        let gain = ndarray::array![1.0, 1.0e10];
        let normalized = MaskedNorm::Rms { epsilon: RMS_EPSILON, gain: gain.view() }
            .apply(rows.view())
            .expect("finite rows");
        let band = rms_norm_band(RMS_EPSILON, gain.view(), rows.view(), normalized.view()).expect("finite rows");
        let square = quad(rows[[0, 0]]) * quad(rows[[0, 0]]) + quad(rows[[0, 1]]) * quad(rows[[0, 1]]);
        let inverse_root = quad(1.0) / (square / quad(2.0) + quad(RMS_EPSILON)).sqrt();
        // `w x` is normal, so the reference's products keep their accuracy.
        let exact = quad(gain[1]) * quad(rows[[0, 1]]) * inverse_root;
        let error = (quad(normalized[[0, 1]]) - exact).0.abs();
        assert!(
            error <= band[[0, 1]],
            "the band {:e} must cover the underflowed product's error {error:e}",
            band[[0, 1]]
        );
        let growth = accumulation_growth(2 + 6);
        let relative_only = growth * normalized[[0, 1]].abs() / (1.0 - growth);
        assert!(
            error > relative_only,
            "positive control: the relative band {relative_only:e} must not cover the error {error:e}"
        );

        let unresolved = ndarray::array![[2.0e-162, 2.0e-162]];
        let unit_gain = ndarray::array![1.0, 1.0];
        let owner = MaskedNorm::Rms { epsilon: 0.0, gain: unit_gain.view() }.apply(unresolved.view());
        assert!(owner.is_ok(), "control: the owner normalizes a row whose mean square is the smallest subnormal");
        let owner = owner.expect("checked above");
        assert!(
            matches!(
                rms_norm_band(0.0, unit_gain.view(), unresolved.view(), owner.view()),
                Err(NormBandUnbounded { row: 0 })
            ),
            "a zero-epsilon row whose mean square is not resolved above the underflow must be refused"
        );
    }

    /// A read whose products round into the subnormal range (#4005). Each of eight products
    /// `3e-161 · 7e-162 ≈ 42.504 · 2^-1074` rounds to a multiple of `2^-1074` and loses about
    /// `0.496 · 2^-1074`, which no relative band sees: the read's error is near
    /// `3.96 · 2^-1074`. Every quantity here is measured in units of `2^-1074`, so each
    /// comparison is between plain normal numbers.
    ///
    /// The units matter and are not cosmetic (#2822). Written as `3.5 · SUBNORMAL_SPACING`
    /// the bar is not the bar: the subnormal grid holds only integer multiples of `2^-1074`,
    /// so that product rounds to `4 · 2^-1074` and demands a whole spacing more than it says,
    /// which the measured `3.964` does not clear. Scaling the comparison into the normal range
    /// by `scale · scale` compounds it, because `2^600 · 2^600` overflows: the bar saturates
    /// at `f64::MAX`, and a diagnostic dividing by it reports an error of zero for an error of
    /// four spacings. The operands are still scaled by `2^600` to read each product's rounding
    /// exactly through a fused multiply-add, but the scaling is undone before anything is
    /// compared, and `scale · (scale · SUBNORMAL_SPACING)` is associated so no factor leaves
    /// the normal range.
    ///
    /// Positive control: the relative band alone, `γ_d |W| |x̂| / (1 − γ_(d+3))`, misses the
    /// error entirely — at this magnitude it underflows to exactly zero, so the whole band is
    /// the read's underflow reach.
    #[test]
    fn a_read_band_encloses_products_that_round_into_the_subnormal_range() {
        const PRODUCTS: usize = 8;
        let scale = 2.0_f64.powi(600);
        let (weight_entry, row_entry) = (3.0e-161, 7.0e-162);
        let weight = Array2::from_elem((1, PRODUCTS), weight_entry);
        let rows = Array2::from_elem((1, PRODUCTS), row_entry);
        let (read, band) = linear_read(test_governor(), weight.view(), ProjectedRows::exact(rows.view())).expect("subnormal read");
        let (weight_scaled, row_scaled) = (weight_entry * scale, row_entry * scale);
        let product = weight_scaled * row_scaled;
        let residual = weight_scaled.mul_add(row_scaled, -product);
        // One spacing at the scaled magnitude, `2^600 · 2^600 · 2^-1074 = 2^126`. Associated
        // so the inner factor is `2^-474`: no intermediate leaves the normal range.
        let scaled_spacing = scale * (scale * SUBNORMAL_SPACING);
        // Every division below is by a power of two and so is exact. The computed read is an
        // integer multiple of the spacing (subnormal sums are exact), and the exact read is
        // `8 (product + residual)` at the scaled magnitude.
        let computed_spacings = read[[0, 0]] / SUBNORMAL_SPACING;
        let terms = PRODUCTS as f64;
        let exact_spacings = terms * product / scaled_spacing + terms * residual / scaled_spacing;
        let error_spacings = (computed_spacings - exact_spacings).abs();
        assert!(
            error_spacings > 3.5,
            "the fixture must lose most of four subnormal spacings to underflow, \
             lost {error_spacings:.4} (computed {computed_spacings}, exact {exact_spacings:.6})"
        );
        let band_spacings = band[[0, 0]] / SUBNORMAL_SPACING;
        assert!(
            error_spacings <= band_spacings,
            "the read band of {band_spacings:.4} spacings must enclose the underflow error of \
             {error_spacings:.4} spacings"
        );
        let magnitude = abs_map(weight.view(), rows.view())[[0, 0]];
        let relative = accumulation_growth(PRODUCTS) * magnitude / (1.0 - accumulation_growth(PRODUCTS + 3));
        assert!(
            relative / SUBNORMAL_SPACING < error_spacings,
            "positive control: the relative band of {:.4} spacings alone must miss the \
             underflow error of {error_spacings:.4} spacings",
            relative / SUBNORMAL_SPACING
        );
    }

    /// A radius read `|W| r` whose products underflow to zero still carries their allowance
    /// (#4005): `1e-170 · 1e-170` rounds to zero, yet rows within `r = 1e-170` of the computed
    /// ones can move the exact read by `1e-340`. The carried radius is at least one subnormal
    /// spacing and exceeds the radius of the same read of exact rows. Positive control: the
    /// computed magnitude `|W| r` is zero.
    #[test]
    fn an_underflowed_radius_read_carries_its_allowance() {
        let weight = Array2::from_elem((1, 1), 1.0e-170);
        let rows = Array2::from_elem((1, 1), 1.0e-170);
        let radius = Array2::from_elem((1, 1), 1.0e-170);
        assert_eq!(
            native_magnitude(weight.view(), radius.view()).magnitude[[0, 0]],
            0.0,
            "positive control: the magnitude of the radius read underflows to zero"
        );
        let (_, exact) = linear_read(test_governor(), weight.view(), ProjectedRows::exact(rows.view())).expect("exact rows");
        let (_, carried) = linear_read(
            test_governor(),
            weight.view(),
            ProjectedRows {
                values: rows.view(),
                radius: radius.view(),
            },
        )
        .expect("rows with a radius");
        assert!(
            carried[[0, 0]] >= SUBNORMAL_SPACING && carried[[0, 0]] > exact[[0, 0]],
            "an underflowed radius read must carry its allowance: carried {:e}, exact rows {:e}",
            carried[[0, 0]],
            exact[[0, 0]]
        );
    }

    /// At normal scale the underflow allowance and the upward steps leave a read band at its
    /// relative size `γ_d |W| |x̂| / (1 − γ_(d+3))`, plus at most `2 d · 2^-1074`. The band's five
    /// stepped operations and its stepped divisor each scale by at most `(1 + u)(1 + 2u)`, and
    /// this test's relative band rounds four times, so the ratio stays below
    /// `(1 + 3u)^6 (1 + u)^4 < 1 + 16 ε`.
    #[test]
    fn the_underflow_allowance_leaves_a_normal_read_band_at_its_relative_size() {
        let mut rng = StdRng::seed_from_u64(4005);
        let weight = eighths(&mut rng, LAYER_WIDTH, LAYER_WIDTH);
        let residual = Array2::from_shape_simple_fn((LAYER_TOKENS, LAYER_WIDTH), || {
            rng.random_range(-48..=48) as f64 / 24.0
        });
        let weight = weight.view();
        let width = weight.ncols();
        let (_, band) = linear_read(test_governor(), weight, ProjectedRows::exact(residual.view())).expect("native read");
        let magnitude = abs_map(weight, residual.mapv(f64::abs).view());
        let relative = magnitude
            .mapv(|value| accumulation_growth(width) * value / (1.0 - accumulation_growth(width + 3)));
        Zip::from(&band).and(&relative).for_each(|&band, &relative| {
            assert!(
                band <= relative * (1.0 + 16.0 * f64::EPSILON) + 2.0 * width as f64 * SUBNORMAL_SPACING,
                "a normal read band {band:e} must stay at its relative size {relative:e}"
            );
        });
        assert!(
            relative.iter().any(|&value| value > 0.0),
            "the fixture's reads must round"
        );
    }

    #[test]
    fn a_linear_read_refuses_a_radius_of_another_shape_than_its_rows() {
        let mut rng = StdRng::seed_from_u64(2994);
        let weight = eighths(&mut rng, LAYER_WIDTH, LAYER_WIDTH);
        let rows = eighths(&mut rng, LAYER_TOKENS, LAYER_WIDTH);
        let radius = rows.mapv(|value| value.abs() * 2.0_f64.powi(-20));
        assert!(
            linear_read(test_governor(), weight.view(), ProjectedRows { values: rows.view(), radius: radius.view() }).is_ok(),
            "control: a radius of its rows' shape is read"
        );
        let short = radius.slice(ndarray::s![.., ..3]);
        assert!(
            matches!(
                linear_read(test_governor(), weight.view(), ProjectedRows { values: rows.view(), radius: short }),
                Err(ApplyError::Shape { operand: "input radius", .. })
            ),
            "a radius of another shape than its rows must be refused"
        );
    }
}

//! The family's **primary row geometry**: how many primaries the slope index
//! occupies, and how those primaries enter the nine-feature frame the sole
//! `row_program!` declaration consumes.
//!
//! The location index `q` has always carried three follow-up channels — its
//! value at entry, at exit, and its exit-time derivative — because the
//! likelihood is `log S(t₁) − log S(t₀)` for a censored row and picks up
//! `log η′(t₁)` for an event. The slope index carried one, which is exactly the
//! restriction gam#2765 / gam#2767 name: `b` could not move along follow-up.
//!
//! [`StaticSlopeGeometry`] is the four-primary `(q₀, q₁, q̇₁, g)` frame the
//! family has always used and is the `db/dt = 0` face of
//! [`DynamicSlopeGeometry`], the six-primary `(q₀, q₁, q̇₁, g₀, g₁, ġ₁)` frame.
//! Both feed the SAME likelihood declaration; only the feature map differs, so
//! there is still exactly one place where the survival marginal-slope
//! log-likelihood is written down.
//!
//! Keeping the static frame at four primaries is not a micro-optimisation. The
//! row towers are dense in the primary count (`Order2` is `P + P²`, the
//! fourth-order tower is `P⁴`), so a model that does not ask for a
//! follow-up-varying slope must not pay `6⁴/4⁴ = 5×` for three channels whose
//! two extra columns are structurally a copy and a zero.

use super::*;

use gam_math::nested_dual::JetField;

/// Entry-time location channel.
pub(crate) const PRIMARY_Q0: usize = 0;
/// Exit-time location channel.
pub(crate) const PRIMARY_Q1: usize = 1;
/// Exit-time location derivative channel.
pub(crate) const PRIMARY_QD1: usize = 2;
/// First slope primary: the sole `g` of the static frame, `g₀` of the dynamic
/// one. Every geometry's slope primaries are contiguous from here.
pub(crate) const PRIMARY_SLOPE: usize = 3;
/// Exit-time slope channel of the dynamic frame.
pub(crate) const PRIMARY_SLOPE_EXIT: usize = 4;
/// Exit-time slope-derivative channel of the dynamic frame.
pub(crate) const PRIMARY_SLOPE_RATE: usize = 5;

/// The number of primaries in the time-constant-slope frame.
pub(crate) const STATIC_SLOPE_PRIMARIES: usize = 4;
/// The number of primaries in the follow-up-varying-slope frame.
pub(crate) const DYNAMIC_SLOPE_PRIMARIES: usize = 6;

/// Which primaries the row program is *affine* in. The higher-order sparse
/// towers elide every derivative block that is structurally zero under this
/// declaration, and `check_contract` asserts the premise at each elision site.
///
/// Only the three location channels qualify. `η₀`, `η₁` and `η′₁` are each
/// degree ≤ 1 in every `q`, and carry no `q·q` product — that survives the
/// follow-up-varying slope unchanged, because the term it adds to `η′₁` is
/// `q₁·c′₁` and `c′₁` has no `q` dependence at all. The slope channels are
/// genuinely nonlinear (through `c`), and `ġ₁` — although the program is
/// degree 1 in it — must stay out of the mask: `dV₁ = 2·cov·g₁·ġ₁` makes
/// `∂³η′₁/∂ġ₁∂q₁∂g₁` nonzero, and that block has two "linear" indices, so
/// declaring `ġ₁` affine would silently drop real curvature.
pub(crate) const RIGID_LINEAR_MASK: u32 =
    (1 << PRIMARY_Q0) | (1 << PRIMARY_Q1) | (1 << PRIMARY_QD1);

/// A primary frame for the survival marginal-slope row program.
///
/// Implementors own only the *feature map* — the likelihood itself is the one
/// `row_program!` declaration in [`super::row_math`]. Everything a consumer
/// needs to pull derivatives back from feature space into primary space is
/// declared here, so a new frame cannot forget a channel: the Jacobian, the
/// per-axis active-feature schedule the sparse pullback walks, and the feature
/// map's own curvature.
pub(crate) trait SlopeRowGeometry<const P: usize>: Copy + Send + Sync + 'static {
    /// Whether the frame lets `b` move along follow-up.
    const FOLLOW_UP_VARYING: bool;

    /// Human-readable name used in diagnostics.
    const NAME: &'static str;

    /// The nine semantic features at this frame's primaries.
    ///
    /// Generic over the scalar so the value path, every compile-time jet, and
    /// the higher-order towers all read one expression. A frame that needs a
    /// zero of the carrier builds it with `JetField::constant_like`, which
    /// inherits the derivative width from a primary rather than requiring the
    /// caller to supply one.
    fn feature_frame<T: JetField + Clone>(
        primaries: &[T; P],
        inputs: &RigidRowInputs,
    ) -> [T; RIGID_FEATURE_DIMENSION];

    /// `∂feature/∂primary`, one row per feature, exactly as
    /// [`order2_feature_pullback_into`] indexes it.
    fn feature_jacobian(
        primaries: &[f64; P],
        inputs: &RigidRowInputs,
    ) -> [[f64; P]; RIGID_FEATURE_DIMENSION];

    /// How many features primary `axis` actually reaches.
    fn active_feature_count(axis: usize) -> usize;

    /// The `slot`-th feature primary `axis` reaches.
    fn active_feature(axis: usize, slot: usize) -> usize;

    /// Accumulate `Σ_f g_f · ∂²f/∂p_a∂p_b` into the `P×P` primary Hessian.
    ///
    /// Every entry is a constant multiple of the score covariance because the
    /// location features are linear in the slope and the variance features are
    /// quadratic — there is no third-or-higher structure in the map itself.
    fn add_feature_curvature(
        feature_gradient: &[f64; RIGID_FEATURE_DIMENSION],
        inputs: &RigidRowInputs,
        hessian: &mut [[f64; P]; P],
    );

    /// `∂feature/∂z_sum` at fixed primaries, and `∂/∂z_sum` of the Jacobian
    /// column of primary `axis`. Together these are everything the Murphy–Topel
    /// generated-regressor correction needs from the frame (gam#2768).
    fn score_sensitivity(
        primaries: &[f64; P],
        inputs: &RigidRowInputs,
    ) -> ScoreSensitivity<P>;
}

/// The frame's dependence on the latent score value itself.
pub(crate) struct ScoreSensitivity<const P: usize> {
    /// `∂feature/∂z_sum`, per feature.
    pub(crate) feature: [f64; RIGID_FEATURE_DIMENSION],
    /// `∂²feature/∂z_sum∂p_a`, per `(feature, primary)` in the same layout as
    /// [`SlopeRowGeometry::feature_jacobian`].
    pub(crate) jacobian: [[f64; P]; RIGID_FEATURE_DIMENSION],
}

/// The frame's follow-up variation as the row program's activity constant:
/// the program evaluates the slope-rate terms of `η′₁` only when it is `1.0`.
#[inline(always)]
pub(crate) fn follow_up_varying_flag<const P: usize, G: SlopeRowGeometry<P>>() -> f64 {
    if G::FOLLOW_UP_VARYING { 1.0 } else { 0.0 }
}

// ── The time-constant slope frame ───────────────────────────────────────

#[derive(Clone, Copy)]
pub(crate) struct StaticSlopeGeometry;

impl SlopeRowGeometry<STATIC_SLOPE_PRIMARIES> for StaticSlopeGeometry {
    const FOLLOW_UP_VARYING: bool = false;
    const NAME: &'static str = "time-constant slope";

    #[inline(always)]
    fn feature_frame<T: JetField + Clone>(
        primaries: &[T; STATIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> [T; RIGID_FEATURE_DIMENSION] {
        let slope = &primaries[PRIMARY_SLOPE];
        let zero = slope.constant_like(0.0);
        let observed = slope.scale(inputs.probit_scale);
        let linear = observed.scale(inputs.z_sum);
        let variance = slope.mul(slope).scale(inputs.covariance_ones);
        static_slope_feature_frame(
            primaries[PRIMARY_Q0].clone(),
            primaries[PRIMARY_Q1].clone(),
            primaries[PRIMARY_QD1].clone(),
            linear,
            variance,
            zero,
        )
    }

    #[inline(always)]
    fn feature_jacobian(
        primaries: &[f64; STATIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> [[f64; STATIC_SLOPE_PRIMARIES]; RIGID_FEATURE_DIMENSION] {
        const P: usize = STATIC_SLOPE_PRIMARIES;
        let mut jacobian = [[0.0; P]; RIGID_FEATURE_DIMENSION];
        jacobian[FEATURE_Q0][PRIMARY_Q0] = 1.0;
        jacobian[FEATURE_Q1][PRIMARY_Q1] = 1.0;
        jacobian[FEATURE_QD1][PRIMARY_QD1] = 1.0;
        let d_linear = inputs.probit_scale * inputs.z_sum;
        let d_variance = 2.0 * primaries[PRIMARY_SLOPE] * inputs.covariance_ones;
        jacobian[FEATURE_LINEAR0][PRIMARY_SLOPE] = d_linear;
        jacobian[FEATURE_LINEAR1][PRIMARY_SLOPE] = d_linear;
        jacobian[FEATURE_VARIANCE0][PRIMARY_SLOPE] = d_variance;
        jacobian[FEATURE_VARIANCE1][PRIMARY_SLOPE] = d_variance;
        jacobian
    }

    #[inline(always)]
    fn active_feature_count(axis: usize) -> usize {
        if axis < PRIMARY_SLOPE {
            1
        } else {
            STATIC_SLOPE_ACTIVE_FEATURES.len()
        }
    }

    #[inline(always)]
    fn active_feature(axis: usize, slot: usize) -> usize {
        if axis < PRIMARY_SLOPE {
            axis
        } else {
            STATIC_SLOPE_ACTIVE_FEATURES[slot]
        }
    }

    #[inline(always)]
    fn add_feature_curvature(
        feature_gradient: &[f64; RIGID_FEATURE_DIMENSION],
        inputs: &RigidRowInputs,
        hessian: &mut [[f64; STATIC_SLOPE_PRIMARIES]; STATIC_SLOPE_PRIMARIES],
    ) {
        hessian[PRIMARY_SLOPE][PRIMARY_SLOPE] += (feature_gradient[FEATURE_VARIANCE0]
            + feature_gradient[FEATURE_VARIANCE1])
            * 2.0
            * inputs.covariance_ones;
    }

    #[inline(always)]
    fn score_sensitivity(
        primaries: &[f64; STATIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> ScoreSensitivity<STATIC_SLOPE_PRIMARIES> {
        const P: usize = STATIC_SLOPE_PRIMARIES;
        let observed_slope = inputs.probit_scale * primaries[PRIMARY_SLOPE];
        let mut feature = [0.0; RIGID_FEATURE_DIMENSION];
        feature[FEATURE_LINEAR0] = observed_slope;
        feature[FEATURE_LINEAR1] = observed_slope;
        let mut jacobian = [[0.0; P]; RIGID_FEATURE_DIMENSION];
        jacobian[FEATURE_LINEAR0][PRIMARY_SLOPE] = inputs.probit_scale;
        jacobian[FEATURE_LINEAR1][PRIMARY_SLOPE] = inputs.probit_scale;
        ScoreSensitivity { feature, jacobian }
    }
}

// ── The follow-up-varying slope frame ──────────────────────────────────

#[derive(Clone, Copy)]
pub(crate) struct DynamicSlopeGeometry;

impl DynamicSlopeGeometry {
    /// `∂²V/∂g² = 2·cov`, which is also the coefficient of the bilinear
    /// variance rate `dV₁ = 2·cov·g₁·ġ₁`. Written once because the frame, its
    /// Jacobian and its curvature all need the same constant.
    #[inline(always)]
    fn variance_curvature_scale(inputs: &RigidRowInputs) -> f64 {
        2.0 * inputs.covariance_ones
    }
}

impl SlopeRowGeometry<DYNAMIC_SLOPE_PRIMARIES> for DynamicSlopeGeometry {
    const FOLLOW_UP_VARYING: bool = true;
    const NAME: &'static str = "follow-up-varying slope";

    #[inline(always)]
    fn feature_frame<T: JetField + Clone>(
        primaries: &[T; DYNAMIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> [T; RIGID_FEATURE_DIMENSION] {
        let entry = &primaries[PRIMARY_SLOPE];
        let exit = &primaries[PRIMARY_SLOPE_EXIT];
        let rate = &primaries[PRIMARY_SLOPE_RATE];
        let location = |slope: &T| slope.scale(inputs.probit_scale).scale(inputs.z_sum);
        [
            primaries[PRIMARY_Q0].clone(),
            primaries[PRIMARY_Q1].clone(),
            primaries[PRIMARY_QD1].clone(),
            location(entry),
            location(exit),
            location(rate),
            entry.mul(entry).scale(inputs.covariance_ones),
            exit.mul(exit).scale(inputs.covariance_ones),
            exit.mul(rate).scale(Self::variance_curvature_scale(inputs)),
        ]
    }

    #[inline(always)]
    fn feature_jacobian(
        primaries: &[f64; DYNAMIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> [[f64; DYNAMIC_SLOPE_PRIMARIES]; RIGID_FEATURE_DIMENSION] {
        const P: usize = DYNAMIC_SLOPE_PRIMARIES;
        let mut jacobian = [[0.0; P]; RIGID_FEATURE_DIMENSION];
        jacobian[FEATURE_Q0][PRIMARY_Q0] = 1.0;
        jacobian[FEATURE_Q1][PRIMARY_Q1] = 1.0;
        jacobian[FEATURE_QD1][PRIMARY_QD1] = 1.0;
        let d_linear = inputs.probit_scale * inputs.z_sum;
        jacobian[FEATURE_LINEAR0][PRIMARY_SLOPE] = d_linear;
        jacobian[FEATURE_LINEAR1][PRIMARY_SLOPE_EXIT] = d_linear;
        jacobian[FEATURE_DLINEAR1][PRIMARY_SLOPE_RATE] = d_linear;
        let curvature_scale = Self::variance_curvature_scale(inputs);
        jacobian[FEATURE_VARIANCE0][PRIMARY_SLOPE] = curvature_scale * primaries[PRIMARY_SLOPE];
        jacobian[FEATURE_VARIANCE1][PRIMARY_SLOPE_EXIT] =
            curvature_scale * primaries[PRIMARY_SLOPE_EXIT];
        jacobian[FEATURE_DVARIANCE1][PRIMARY_SLOPE_EXIT] =
            curvature_scale * primaries[PRIMARY_SLOPE_RATE];
        jacobian[FEATURE_DVARIANCE1][PRIMARY_SLOPE_RATE] =
            curvature_scale * primaries[PRIMARY_SLOPE_EXIT];
        jacobian
    }

    #[inline(always)]
    fn active_feature_count(axis: usize) -> usize {
        match axis {
            PRIMARY_SLOPE => 2,
            PRIMARY_SLOPE_EXIT => 3,
            PRIMARY_SLOPE_RATE => 2,
            _ => 1,
        }
    }

    #[inline(always)]
    fn active_feature(axis: usize, slot: usize) -> usize {
        match (axis, slot) {
            (PRIMARY_SLOPE, 0) => FEATURE_LINEAR0,
            (PRIMARY_SLOPE, _) => FEATURE_VARIANCE0,
            (PRIMARY_SLOPE_EXIT, 0) => FEATURE_LINEAR1,
            (PRIMARY_SLOPE_EXIT, 1) => FEATURE_VARIANCE1,
            (PRIMARY_SLOPE_EXIT, _) => FEATURE_DVARIANCE1,
            (PRIMARY_SLOPE_RATE, 0) => FEATURE_DLINEAR1,
            (PRIMARY_SLOPE_RATE, _) => FEATURE_DVARIANCE1,
            (identity, _) => identity,
        }
    }

    #[inline(always)]
    fn add_feature_curvature(
        feature_gradient: &[f64; RIGID_FEATURE_DIMENSION],
        inputs: &RigidRowInputs,
        hessian: &mut [[f64; DYNAMIC_SLOPE_PRIMARIES]; DYNAMIC_SLOPE_PRIMARIES],
    ) {
        let curvature_scale = Self::variance_curvature_scale(inputs);
        // `∂²V₀/∂g₀² = ∂²V₁/∂g₁² = 2·cov`
        hessian[PRIMARY_SLOPE][PRIMARY_SLOPE] +=
            feature_gradient[FEATURE_VARIANCE0] * curvature_scale;
        hessian[PRIMARY_SLOPE_EXIT][PRIMARY_SLOPE_EXIT] +=
            feature_gradient[FEATURE_VARIANCE1] * curvature_scale;
        // `∂²(dV₁)/∂g₁∂ġ₁ = 2·cov`; the map has no other second derivative.
        let mixed = feature_gradient[FEATURE_DVARIANCE1] * curvature_scale;
        hessian[PRIMARY_SLOPE_EXIT][PRIMARY_SLOPE_RATE] += mixed;
        hessian[PRIMARY_SLOPE_RATE][PRIMARY_SLOPE_EXIT] += mixed;
    }

    #[inline(always)]
    fn score_sensitivity(
        primaries: &[f64; DYNAMIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> ScoreSensitivity<DYNAMIC_SLOPE_PRIMARIES> {
        const P: usize = DYNAMIC_SLOPE_PRIMARIES;
        let mut feature = [0.0; RIGID_FEATURE_DIMENSION];
        feature[FEATURE_LINEAR0] = inputs.probit_scale * primaries[PRIMARY_SLOPE];
        feature[FEATURE_LINEAR1] = inputs.probit_scale * primaries[PRIMARY_SLOPE_EXIT];
        feature[FEATURE_DLINEAR1] = inputs.probit_scale * primaries[PRIMARY_SLOPE_RATE];
        let mut jacobian = [[0.0; P]; RIGID_FEATURE_DIMENSION];
        jacobian[FEATURE_LINEAR0][PRIMARY_SLOPE] = inputs.probit_scale;
        jacobian[FEATURE_LINEAR1][PRIMARY_SLOPE_EXIT] = inputs.probit_scale;
        jacobian[FEATURE_DLINEAR1][PRIMARY_SLOPE_RATE] = inputs.probit_scale;
        ScoreSensitivity { feature, jacobian }
    }
}

/// Run a block in whichever primary frame the family's slope layout selects.
///
/// The two frames have different primary counts, so a value that still carries
/// `P` cannot cross this boundary — the block must reduce to a frame-free type
/// (an `ndarray` value, a `dyn` workspace, a scalar). That is the point: it
/// forces every dispatch to name where the frame stops mattering, instead of
/// letting a four-primary assumption leak downstream.
macro_rules! in_slope_frame {
    ($family:expr, $primaries:ident, $geometry:ident, $body:block) => {{
        if $family.slope_is_follow_up_varying() {
            const $primaries: usize = DYNAMIC_SLOPE_PRIMARIES;
            type $geometry = DynamicSlopeGeometry;
            $body
        } else {
            const $primaries: usize = STATIC_SLOPE_PRIMARIES;
            type $geometry = StaticSlopeGeometry;
            $body
        }
    }};
}

pub(crate) use in_slope_frame;

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::matrix::DesignMatrix;
    use ndarray::array;

    fn inputs(probit_scale: f64, z_sum: f64, covariance_ones: f64, di: f64) -> RigidRowInputs {
        RigidRowInputs {
            row: 0,
            wi: 0.75,
            di,
            z_sum,
            covariance_ones,
            probit_scale,
            qd1_lower: -1.0,
        }
    }

    /// Deterministic xorshift grid, no RNG dependency (matching the style of the
    /// other row-program oracles in this crate).
    struct Grid(u64);

    impl Grid {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    /// The static frame IS the `db/dt = 0` face of the dynamic one.
    ///
    /// Not a smoke test: it is the statement that gam#2765's generalization does
    /// not change the model anybody already fitted. Under `g₀ = g₁ = g`,
    /// `ġ₁ = 0`, the six-primary lowering must reproduce the four-primary one
    /// channel for channel, with the two slope columns of the dynamic frame
    /// summing to the single slope column of the static one — because `g` there
    /// is literally the same coefficient functional read at both endpoints.
    #[test]
    fn dynamic_frame_reduces_to_the_static_frame_when_the_slope_does_not_move_2765() {
        let mut grid = Grid(0x9E3779B97F4A7C15);
        let mut worst = 0.0_f64;
        for _ in 0..2000 {
            let q0 = grid.next() * 1.5;
            let q1 = grid.next() * 1.5;
            let qd1 = 0.5 + grid.next().abs() * 2.0;
            let g = grid.next() * 1.2;
            let row = inputs(
                0.6 + grid.next().abs(),
                grid.next() * 1.2,
                0.7 + grid.next().abs(),
                if grid.next() > 0.0 { 1.0 } else { 0.0 },
            );

            let (static_value, static_gradient, static_hessian) =
                rigid_row_order2::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(
                    &[q0, q1, qd1, g],
                    &row,
                )
                .expect("static frame admits the row");
            let (dynamic_value, dynamic_gradient, dynamic_hessian) =
                rigid_row_order2::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>(
                    &[q0, q1, qd1, g, g, 0.0],
                    &row,
                )
                .expect("dynamic frame admits the same row");

            let mut check = |left: f64, right: f64, what: &str| {
                let tolerance = 1e-12 * (1.0 + left.abs().max(right.abs()));
                let error = (left - right).abs();
                worst = worst.max(error / (1.0 + left.abs().max(right.abs())));
                assert!(
                    error <= tolerance,
                    "{what}: static {left:+.17e} vs dynamic {right:+.17e}"
                );
            };

            check(static_value, dynamic_value, "value");
            for axis in 0..PRIMARY_SLOPE {
                check(
                    static_gradient[axis],
                    dynamic_gradient[axis],
                    "location gradient",
                );
            }
            // The slope column splits across the entry and exit channels.
            check(
                static_gradient[PRIMARY_SLOPE],
                dynamic_gradient[PRIMARY_SLOPE] + dynamic_gradient[PRIMARY_SLOPE_EXIT],
                "slope gradient",
            );
            for left in 0..PRIMARY_SLOPE {
                for right in 0..PRIMARY_SLOPE {
                    check(
                        static_hessian[left][right],
                        dynamic_hessian[left][right],
                        "location Hessian",
                    );
                }
                check(
                    static_hessian[left][PRIMARY_SLOPE],
                    dynamic_hessian[left][PRIMARY_SLOPE]
                        + dynamic_hessian[left][PRIMARY_SLOPE_EXIT],
                    "location/slope Hessian",
                );
            }
            check(
                static_hessian[PRIMARY_SLOPE][PRIMARY_SLOPE],
                dynamic_hessian[PRIMARY_SLOPE][PRIMARY_SLOPE]
                    + dynamic_hessian[PRIMARY_SLOPE][PRIMARY_SLOPE_EXIT]
                    + dynamic_hessian[PRIMARY_SLOPE_EXIT][PRIMARY_SLOPE]
                    + dynamic_hessian[PRIMARY_SLOPE_EXIT][PRIMARY_SLOPE_EXIT],
                "slope Hessian",
            );
        }
        assert!(worst <= 1e-12, "worst relative disagreement {worst:.3e}");
    }

    /// The dynamic frame's analytic gradient and Hessian against central
    /// differences of the same row program. This is what certifies the two
    /// genuinely new channels — `ġ₁`, and `g₁`'s extra route into `η′₁` through
    /// `dV₁` — rather than only the ones the static frame already exercised.
    #[test]
    fn dynamic_frame_derivatives_match_central_differences_2765() {
        let mut grid = Grid(0xD1B54A32D192ED03);
        let mut worst = 0.0_f64;
        for _ in 0..400 {
            let row = inputs(
                0.7 + grid.next().abs() * 0.3,
                grid.next() * 1.2,
                0.7 + grid.next().abs(),
                if grid.next() > 0.0 { 1.0 } else { 0.0 },
            );
            // Keep `η′₁` comfortably positive: the row program takes its log, so
            // the admissible set is an open half space, not all of R⁶.
            let primaries = [
                grid.next() * 1.0,
                grid.next() * 1.0,
                2.0 + grid.next().abs(),
                grid.next() * 0.6,
                grid.next() * 0.6,
                grid.next() * 0.15,
            ];
            let value_at = |point: &[f64; DYNAMIC_SLOPE_PRIMARIES]| -> Option<f64> {
                rigid_row_order2::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>(point, &row)
                    .ok()
                    .map(|(value, _, _)| value)
            };
            let Some((_, gradient, hessian)) =
                rigid_row_order2::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>(
                    &primaries, &row,
                )
                .ok()
            else {
                continue;
            };

            let step = 1e-5;
            for axis in 0..DYNAMIC_SLOPE_PRIMARIES {
                let mut up = primaries;
                let mut down = primaries;
                up[axis] += step;
                down[axis] -= step;
                let (Some(up_value), Some(down_value)) = (value_at(&up), value_at(&down)) else {
                    continue;
                };
                let finite = (up_value - down_value) / (2.0 * step);
                let scale = 1.0 + finite.abs().max(gradient[axis].abs());
                let error = (finite - gradient[axis]).abs() / scale;
                worst = worst.max(error);
                assert!(
                    error <= 5e-6,
                    "{} gradient axis {axis}: analytic {:+.12e} vs central difference {finite:+.12e}",
                    DynamicSlopeGeometry::NAME,
                    gradient[axis],
                );

                // Second derivative along the same axis.
                let Some(centre) = value_at(&primaries) else {
                    continue;
                };
                let finite_second = (up_value - 2.0 * centre + down_value) / (step * step);
                let scale = 1.0 + finite_second.abs().max(hessian[axis][axis].abs());
                let error = (finite_second - hessian[axis][axis]).abs() / scale;
                assert!(
                    error <= 5e-4,
                    "{} Hessian axis {axis}: analytic {:+.12e} vs central difference {finite_second:+.12e}",
                    DynamicSlopeGeometry::NAME,
                    hessian[axis][axis],
                );
            }
        }
        assert!(worst <= 5e-6, "worst relative gradient error {worst:.3e}");
    }

    /// `η′₁` is the follow-up derivative of `η₁`, including the two terms a
    /// time-constant slope zeroes out. Differentiating the model's own
    /// definition of `η(t)` in `t` and comparing to the program's witness is the
    /// statement that gam#2765's kernel really did gain the right terms.
    #[test]
    fn adjusted_derivative_is_the_follow_up_derivative_of_eta_2767() {
        let probit_scale = 0.83;
        let z_sum = -0.7;
        let covariance_ones = 1.4;
        let row = inputs(probit_scale, z_sum, covariance_ones, 1.0);

        // Explicit, smooth `q(t)` and `g(t)`; nothing about them is special
        // beyond being nonlinear in `t` so every term is exercised.
        let q = |t: f64| 0.4 + 0.9 * t + 0.25 * t * t;
        let q_rate = |t: f64| 0.9 + 0.5 * t;
        let g = |t: f64| 0.3 + 0.7 * t - 0.2 * t * t;
        let g_rate = |t: f64| 0.7 - 0.4 * t;
        let eta = |t: f64| {
            let slope = g(t);
            let correction = (1.0 + probit_scale * probit_scale * slope * slope * covariance_ones)
                .sqrt();
            q(t) * correction + probit_scale * slope * z_sum
        };

        for step_exponent in [4, 5, 6] {
            let t = 0.6_f64;
            let h = 10.0_f64.powi(-step_exponent);
            let finite = (eta(t + h) - eta(t - h)) / (2.0 * h);

            let features = [
                0.0,
                q(t),
                q_rate(t),
                0.0,
                probit_scale * g(t) * z_sum,
                probit_scale * g_rate(t) * z_sum,
                0.0,
                g(t) * g(t) * covariance_ones,
                2.0 * g(t) * g_rate(t) * covariance_ones,
            ];
            let (_, _, _, [_, _, adjusted_derivative]) = rigid_feature_frame_order2(
                &features,
                row.wi,
                row.di,
                probit_scale,
                follow_up_varying_flag::<DYNAMIC_SLOPE_PRIMARIES, DynamicSlopeGeometry>(),
            );

            let error = (finite - adjusted_derivative).abs()
                / (1.0 + finite.abs().max(adjusted_derivative.abs()));
            assert!(
                error <= 1e-7,
                "h=1e-{step_exponent}: program η′₁ {adjusted_derivative:+.12e} vs central \
                 difference of η(t) {finite:+.12e}"
            );
        }
    }

    /// A layout with follow-up designs reports the three channels the row frame
    /// consumes, and the time-constant layout collapses them exactly.
    #[test]
    fn follow_up_layout_reports_entry_exit_and_rate_channels_2765() {
        let exit = DesignMatrix::from(array![[1.0, 2.0], [1.0, 5.0]]);
        let entry = DesignMatrix::from(array![[1.0, 1.0], [1.0, 3.0]]);
        let rate = DesignMatrix::from(array![[0.0, 0.5], [0.0, 0.25]]);
        let offset = array![0.125, -0.25];
        let beta = array![3.0, -1.5];

        let static_layout = SlopeTopology::shared()
            .materialize_identity(exit.clone(), &offset)
            .expect("shared layout");
        assert!(!static_layout.is_follow_up_varying());
        let static_channels = static_layout
            .row_channels(1, &beta, 42.0)
            .expect("static channels");
        assert_eq!(static_channels.entry, 42.0);
        assert_eq!(static_channels.exit, 42.0);
        assert_eq!(static_channels.rate, 0.0);
        assert_eq!(static_layout.primary_channels().as_slice().len(), 1);

        let dynamic_layout = static_layout
            .with_follow_up(entry, rate)
            .expect("shared layouts accept a follow-up margin");
        assert!(dynamic_layout.is_follow_up_varying());
        let exit_eta = 1.0 * 3.0 + 5.0 * -1.5 + offset[1];
        let channels = dynamic_layout
            .row_channels(1, &beta, exit_eta)
            .expect("dynamic channels");
        assert_eq!(channels.entry, 1.0 * 3.0 + 3.0 * -1.5 + offset[1]);
        assert_eq!(channels.exit, exit_eta);
        assert_eq!(channels.rate, 0.0 * 3.0 + 0.25 * -1.5);
        let primaries = dynamic_layout.primary_channels();
        assert_eq!(
            primaries
                .as_slice()
                .iter()
                .map(|&(primary, _)| primary)
                .collect::<Vec<_>>(),
            vec![PRIMARY_SLOPE, PRIMARY_SLOPE_EXIT, PRIMARY_SLOPE_RATE],
        );
    }

    /// A per-score topology cannot carry a single time margin, and says so.
    #[test]
    fn per_score_layout_refuses_a_follow_up_margin_2765() {
        let raw = array![[2.0, 3.0], [7.0, 11.0]];
        let layout = SlopeTopology::per_score(vec![0..1, 1..2], 2)
            .expect("per-score topology")
            .materialize_identity(DesignMatrix::from(raw.clone()), &array![0.0, 0.0])
            .expect("per-score layout");
        let Err(error) = layout.with_follow_up(
            DesignMatrix::from(raw.clone()),
            DesignMatrix::from(raw),
        ) else {
            panic!(
                // SAFETY (test): the refusal is the property under test; if the
                // call succeeds there is nothing left to assert on.
                "per-score plus a time margin must be refused"
            );
        };
        assert!(error.contains("per-score"), "{error}");
    }
}

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

use gam_math::jet_scalar::JetScalar;
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
/// `row_program!` declaration in [`super::row_math`]. A frame declares its
/// features over any jet, the two direct second-order lowerings every consumer
/// reads, and the towers its higher-order paths are built on.
pub(crate) trait SlopeRowGeometry<const P: usize>: Copy + Send + Sync + 'static {
    /// Whether the frame lets `b` move along follow-up.
    const FOLLOW_UP_VARYING: bool;

    /// Whether the frame anchors the index on a declared latent law
    /// (gam#2923) instead of the Gaussian closed form. An anchored frame reads
    /// [`RigidRowInputs::anchor`], is nonlinear in every primary, and has no
    /// device lowering.
    const ANCHORED: bool;

    /// Human-readable name used in diagnostics.
    const NAME: &'static str;

    /// The order-≤3 tower the all-axes first-directional path builds once per
    /// row. Static sparsity is a property of the FRAME: the Gaussian frames are
    /// affine in the three location primaries and elide those blocks, the
    /// anchored frame is affine in `q̇₁` alone and elides only those blocks.
    type Tower3: JetScalar<P> + SparseThird<P> + Send + Sync;

    /// The order-≤4 tower of the second-directional all-axes path.
    type Tower4: JetScalar<P> + SparseThird<P> + SparseFourth<P> + Send + Sync;

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

    /// `∂feature/∂z_sum` at fixed primaries, and `∂/∂z_sum` of the Jacobian
    /// column of primary `axis`. Together these are everything the Murphy–Topel
    /// generated-regressor correction needs from the frame (gam#2768).
    fn score_sensitivity(
        primaries: &[f64; P],
        inputs: &RigidRowInputs,
    ) -> ScoreSensitivity<P>;

    /// Direct value/gradient/Hessian lowering of the canonical nine-feature row
    /// program followed by the second-order pullback into this frame.
    fn row_order2(
        primaries: &[f64; P],
        inputs: &RigidRowInputs,
    ) -> Result<(f64, [f64; P], [[f64; P]; P]), String>;

    /// `∂/∂z` of the rigid row NLL's PRIMARY gradient: the mixed `(primary,
    /// latent score)` second derivative, before any block-Jacobian scatter
    /// (gam#2768) — the row channel the Murphy–Topel generated-regressor
    /// covariance correction contracts against.
    fn row_primary_mixed_in_z(
        primaries: &[f64; P],
        inputs: &RigidRowInputs,
    ) -> Result<[f64; P], String>;
}

/// The feature map of a GAUSSIAN frame: one whose features are at most
/// quadratic in the primaries, so the order-two pullback needs only a Jacobian
/// and a constant curvature. Everything the pullback needs is declared here, so
/// a new frame cannot forget a channel: the Jacobian, the per-axis
/// active-feature schedule the sparse pullback walks, and the feature map's
/// own curvature.
pub(crate) trait GaussianFeatureMap<const P: usize>: SlopeRowGeometry<P> {
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
}

/// Direct value/gradient/Hessian lowering of the canonical nine-feature row
/// program followed by the universal second-order pullback into a Gaussian
/// frame `G`. The fixed stack buffers and active-feature map expose only the
/// channels the frame's slope primaries actually reach.
#[inline(always)]
pub(crate) fn gaussian_row_order2<const P: usize, G: GaussianFeatureMap<P>>(
    primaries: &[f64; P],
    inputs: &RigidRowInputs,
) -> Result<(f64, [f64; P], [[f64; P]; P]), String> {
    let features = G::feature_frame(primaries, inputs);
    let (value, feature_gradient, feature_hessian, [neg_eta0, neg_eta1, adjusted_derivative]) =
        rigid_feature_frame_order2(
            &features,
            inputs.wi,
            inputs.wi_entry,
            inputs.di,
            inputs.probit_scale,
            follow_up_varying_flag::<P, G>(),
        );
    validate_rigid_row_admission::<P, G>(
        primaries[PRIMARY_QD1],
        inputs,
        neg_eta0,
        neg_eta1,
        adjusted_derivative,
    )?;

    let jacobian = G::feature_jacobian(primaries, inputs);
    let mut gradient = [0.0; P];
    let mut hessian = [[0.0; P]; P];
    order2_feature_pullback_into(
        &feature_gradient,
        &feature_hessian,
        &jacobian,
        G::active_feature_count,
        G::active_feature,
        &mut gradient,
        &mut hessian,
        |gradient, hessian| G::add_feature_curvature(gradient, inputs, hessian),
    );
    Ok((value, gradient, hessian))
}

/// `∂/∂z` of the rigid row NLL's PRIMARY gradient on a Gaussian frame: the
/// mixed `(primary, latent score)` second derivative, before any block-Jacobian
/// scatter (gam#2768).
///
/// This is the row channel the Murphy–Topel generated-regressor covariance
/// correction contracts against — `s_i = ∂(score_β,i)/∂ζ_i` is exactly this
/// vector pushed through the same primary→β Jacobian the gradient uses — and it
/// is derived MECHANICALLY from the sole `rigid_feature_program` declaration
/// rather than by hand, in the spirit of the single-source contract on
/// [`rigid_row_nll`].
///
/// Differentiating the pullback `∂ℓ/∂p_a = Σ_f g_f·J_{f a}` gives
///
/// ```text
///     ∂²ℓ/∂p_a ∂z_sum = Σ_f (Σ_h H_{f h}·∂h/∂z)·J_{f a}  +  Σ_f g_f·∂J_{f a}/∂z,
/// ```
///
/// and both `∂h/∂z` and `∂J/∂z` are owned by the frame
/// ([`SlopeRowGeometry::score_sensitivity`]) — for a static slope the score
/// reaches the entry AND exit location channels, which is what the single
/// `linear` feature used to be. For `K = 1` (`z_sum = z`, the only shape the
/// conditional latent calibration is persisted for) this is the row's exact
/// `∂/∂z`.
#[inline]
pub(crate) fn gaussian_row_primary_mixed_in_z<const P: usize, G: GaussianFeatureMap<P>>(
    primaries: &[f64; P],
    inputs: &RigidRowInputs,
) -> Result<[f64; P], String> {
    let features = G::feature_frame(primaries, inputs);
    let (_, feature_gradient, feature_hessian, [neg_eta0, neg_eta1, adjusted_derivative]) =
        rigid_feature_frame_order2(
            &features,
            inputs.wi,
            inputs.wi_entry,
            inputs.di,
            inputs.probit_scale,
            follow_up_varying_flag::<P, G>(),
        );
    validate_rigid_row_admission::<P, G>(
        primaries[PRIMARY_QD1],
        inputs,
        neg_eta0,
        neg_eta1,
        adjusted_derivative,
    )?;
    let jacobian = G::feature_jacobian(primaries, inputs);
    Ok(mixed_in_z_from_feature_derivatives(
        &feature_gradient,
        &feature_hessian,
        &jacobian,
        G::active_feature_count,
        G::active_feature,
        &G::score_sensitivity(primaries, inputs),
    ))
}

/// The `∂/∂z_sum` contraction of [`SlopeRowGeometry::row_primary_mixed_in_z`]
/// once the feature derivatives, the Jacobian, the active-feature schedule and
/// the frame's score sensitivity are in hand.
#[inline]
pub(crate) fn mixed_in_z_from_feature_derivatives<const P: usize>(
    feature_gradient: &[f64; RIGID_FEATURE_DIMENSION],
    feature_hessian: &[[f64; RIGID_FEATURE_DIMENSION]; RIGID_FEATURE_DIMENSION],
    jacobian: &[[f64; P]; RIGID_FEATURE_DIMENSION],
    active_feature_count: impl Fn(usize) -> usize,
    active_feature: impl Fn(usize, usize) -> usize,
    sensitivity: &ScoreSensitivity<P>,
) -> [f64; P] {
    // `Σ_h H_{f h}·∂h/∂z_sum`, one entry per feature.
    let hessian_in_z: [f64; RIGID_FEATURE_DIMENSION] = std::array::from_fn(|feature| {
        let mut channel = 0.0;
        for other in 0..RIGID_FEATURE_DIMENSION {
            channel += feature_hessian[feature][other] * sensitivity.feature[other];
        }
        channel
    });
    let mut mixed = [0.0; P];
    for axis in 0..P {
        let mut channel = 0.0;
        for slot in 0..active_feature_count(axis) {
            let feature = active_feature(axis, slot);
            channel += hessian_in_z[feature] * jacobian[feature][axis]
                + feature_gradient[feature] * sensitivity.jacobian[feature][axis];
        }
        mixed[axis] = channel;
    }
    mixed
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
    const ANCHORED: bool = false;
    const NAME: &'static str = "time-constant slope";
    type Tower3 = SparseTower3<STATIC_SLOPE_PRIMARIES, RIGID_LINEAR_MASK>;
    type Tower4 = SparseTower4<STATIC_SLOPE_PRIMARIES, RIGID_LINEAR_MASK>;

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

    #[inline(always)]
    fn row_order2(
        primaries: &[f64; STATIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> Result<
        (
            f64,
            [f64; STATIC_SLOPE_PRIMARIES],
            [[f64; STATIC_SLOPE_PRIMARIES]; STATIC_SLOPE_PRIMARIES],
        ),
        String,
    > {
        gaussian_row_order2::<STATIC_SLOPE_PRIMARIES, Self>(primaries, inputs)
    }

    #[inline]
    fn row_primary_mixed_in_z(
        primaries: &[f64; STATIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> Result<[f64; STATIC_SLOPE_PRIMARIES], String> {
        gaussian_row_primary_mixed_in_z::<STATIC_SLOPE_PRIMARIES, Self>(primaries, inputs)
    }
}

impl GaussianFeatureMap<STATIC_SLOPE_PRIMARIES> for StaticSlopeGeometry {
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
    const ANCHORED: bool = false;
    const NAME: &'static str = "follow-up-varying slope";
    type Tower3 = SparseTower3<DYNAMIC_SLOPE_PRIMARIES, RIGID_LINEAR_MASK>;
    type Tower4 = SparseTower4<DYNAMIC_SLOPE_PRIMARIES, RIGID_LINEAR_MASK>;

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

    #[inline(always)]
    fn row_order2(
        primaries: &[f64; DYNAMIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> Result<
        (
            f64,
            [f64; DYNAMIC_SLOPE_PRIMARIES],
            [[f64; DYNAMIC_SLOPE_PRIMARIES]; DYNAMIC_SLOPE_PRIMARIES],
        ),
        String,
    > {
        gaussian_row_order2::<DYNAMIC_SLOPE_PRIMARIES, Self>(primaries, inputs)
    }

    #[inline]
    fn row_primary_mixed_in_z(
        primaries: &[f64; DYNAMIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> Result<[f64; DYNAMIC_SLOPE_PRIMARIES], String> {
        gaussian_row_primary_mixed_in_z::<DYNAMIC_SLOPE_PRIMARIES, Self>(primaries, inputs)
    }
}

impl GaussianFeatureMap<DYNAMIC_SLOPE_PRIMARIES> for DynamicSlopeGeometry {
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
}

// ── The anchored time-constant slope frame (gam#2923) ──────────────────

/// The four-primary `(q₀, q₁, q̇₁, g)` frame whose location channels are
/// anchored on a declared finite law instead of lowered in closed form.
///
/// The row program is unchanged: it still computes `η = q·√(1 + s²V) + L` and
/// `η′₁ = q̇₁·√(1 + s²V₁) + …`. This frame feeds it `q_j := α(q_j, b)`,
/// `V_j := 0` and `q̇₁ := α_q(q₁, b)·q̇₁`, so that what it evaluates is
///
/// ```text
///     η_j = α(q_j, b) + b·z,        η′₁ = ∂α/∂q (q₁, b) · q̇₁,
/// ```
///
/// with `α` the root of `Σ_k w_k Φ(−(α + b u_k)) = Φ(−q)` on the row's law
/// ([`super::anchor`]). On a Gaussian law `α = q·√(1 + b²)` to quadrature
/// tolerance and the frame IS [`StaticSlopeGeometry`]; on any other law the
/// closed form is wrong by exactly the amount the anchoring equation says.
///
/// The map is nonlinear in `q₀`, `q₁` and `g` — `α` in `q` as well as in `g` —
/// so of the location channels only `q̇₁` stays affine
/// ([`ANCHORED_LINEAR_MASK`]), and `η′₁ > 0` still follows from `q̇₁ > 0`
/// because `α_q = φ(q)/Σ_k w_k φ(η_k)` is strictly positive.
#[derive(Clone, Copy)]
pub(crate) struct AnchoredStaticSlopeGeometry;

/// The anchored frame's active-feature schedule for the slope primary: it
/// reaches every anchored channel and both linear channels.
const ANCHORED_SLOPE_ACTIVE_FEATURES: [usize; 5] = [
    FEATURE_Q0,
    FEATURE_Q1,
    FEATURE_QD1,
    FEATURE_LINEAR0,
    FEATURE_LINEAR1,
];

/// Both anchors of a row, solved and differentiated once.
struct AnchoredRowState {
    entry: AnchorDerivatives,
    exit: AnchorDerivatives,
    observed_slope: f64,
}

impl AnchoredStaticSlopeGeometry {
    #[inline]
    fn context<'a>(inputs: &RigidRowInputs<'a>) -> AnchorRowContext<'a> {
        inputs
            .anchor
            .expect("the anchored slope frame is only constructed with a declared latent law")
    }

    /// Solve one location channel's anchor through the row's slot.
    #[inline]
    fn anchor(
        q: f64,
        observed_slope: f64,
        inputs: &RigidRowInputs,
        slot: SurvivalInterceptSlotKind,
    ) -> Result<f64, String> {
        solve_anchor_in_slot(
            q,
            observed_slope,
            Self::context(inputs),
            inputs.row,
            survival_anchor_slot(slot),
        )
    }

    /// One location channel's anchor with its implicit derivatives, through
    /// the row's slot.
    #[inline]
    fn derivatives(
        q: f64,
        observed_slope: f64,
        inputs: &RigidRowInputs,
        slot: SurvivalInterceptSlotKind,
    ) -> Result<AnchorDerivatives, String> {
        anchor_derivatives_in_slot(
            q,
            observed_slope,
            Self::context(inputs),
            inputs.row,
            survival_anchor_slot(slot),
        )
    }

    /// One location channel's Taylor table, through the row's slot: every
    /// consumer of the row at one iterate reads the one table the first of
    /// them differentiated.
    #[inline]
    fn taylor(
        q: f64,
        observed_slope: f64,
        inputs: &RigidRowInputs,
        slot: SurvivalInterceptSlotKind,
    ) -> Result<AnchorTaylor, String> {
        anchor_taylor_in_slot(
            q,
            observed_slope,
            Self::context(inputs),
            inputs.row,
            survival_anchor_slot(slot),
        )
    }

    /// `[α(q₀, b), α(q₁, b), α_q(q₁, b)·q̇₁]` over any jet (gam#2928).
    ///
    /// The roots are solved on the real values. A carrier with derivative
    /// channels then receives each anchor's Taylor table through order five
    /// ([`AnchorTaylor`]), read through the row's slot and composed with its
    /// own primaries: a few carrier products per anchor, where Newton's
    /// iteration in the jet algebra (the test oracle `anchor_jet`) walked the
    /// whole law three times.
    fn anchored_channels<T: JetField + Clone>(
        primaries: &[T; STATIC_SLOPE_PRIMARIES],
        observed: &T,
        inputs: &RigidRowInputs,
    ) -> Result<[T; 3], String> {
        let q0 = &primaries[PRIMARY_Q0];
        let q1 = &primaries[PRIMARY_Q1];
        let qd1 = &primaries[PRIMARY_QD1];
        let b = observed.value();
        if std::mem::size_of::<T>() == std::mem::size_of::<f64>() {
            let alpha0 = Self::anchor(q0.value(), b, inputs, SurvivalInterceptSlotKind::Entry)?;
            let alpha1 = Self::anchor(q1.value(), b, inputs, SurvivalInterceptSlotKind::Exit)?;
            // A carrier the size of one `f64` has no derivative channel: the
            // roots and `α_q(q₁, b)` are the whole answer, read from the exit
            // slot's derivatives (the same bits `φ(q₁)/Σ_k w_k φ(η_k)` gives).
            let rate_factor =
                Self::derivatives(q1.value(), b, inputs, SurvivalInterceptSlotKind::Exit)?.a_q;
            return Ok([
                q0.constant_like(alpha0),
                q1.constant_like(alpha1),
                qd1.scale(rate_factor),
            ]);
        }
        let delta_b = observed.compose_unary([0.0, 1.0, 0.0, 0.0, 0.0]);
        let entry = Self::taylor(q0.value(), b, inputs, SurvivalInterceptSlotKind::Entry)?;
        let exit = Self::taylor(q1.value(), b, inputs, SurvivalInterceptSlotKind::Exit)?;
        Ok([
            entry.lift(q0, &delta_b),
            exit.lift(q1, &delta_b),
            exit.lift_q_derivative(q1, &delta_b).mul(qd1),
        ])
    }

    fn solve_row(
        primaries: &[f64; STATIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> Result<AnchoredRowState, String> {
        let observed_slope = inputs.probit_scale * primaries[PRIMARY_SLOPE];
        let entry = Self::derivatives(
            primaries[PRIMARY_Q0],
            observed_slope,
            inputs,
            SurvivalInterceptSlotKind::Entry,
        )?;
        let exit = Self::derivatives(
            primaries[PRIMARY_Q1],
            observed_slope,
            inputs,
            SurvivalInterceptSlotKind::Exit,
        )?;
        Ok(AnchoredRowState {
            entry,
            exit,
            observed_slope,
        })
    }

    #[inline]
    fn features_from(
        state: &AnchoredRowState,
        primaries: &[f64; STATIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> [f64; RIGID_FEATURE_DIMENSION] {
        let linear = state.observed_slope * inputs.z_sum;
        [
            state.entry.alpha,
            state.exit.alpha,
            state.exit.a_q * primaries[PRIMARY_QD1],
            linear,
            linear,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    }

    /// `∂feature/∂primary` of the anchored map, one row per feature.
    #[inline]
    fn jacobian_from(
        state: &AnchoredRowState,
        primaries: &[f64; STATIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> [[f64; STATIC_SLOPE_PRIMARIES]; RIGID_FEATURE_DIMENSION] {
        const P: usize = STATIC_SLOPE_PRIMARIES;
        let s = inputs.probit_scale;
        let qd1 = primaries[PRIMARY_QD1];
        let mut jacobian = [[0.0; P]; RIGID_FEATURE_DIMENSION];
        jacobian[FEATURE_Q0][PRIMARY_Q0] = state.entry.a_q;
        jacobian[FEATURE_Q0][PRIMARY_SLOPE] = state.entry.a_b * s;
        jacobian[FEATURE_Q1][PRIMARY_Q1] = state.exit.a_q;
        jacobian[FEATURE_Q1][PRIMARY_SLOPE] = state.exit.a_b * s;
        // α̇₁ = α_q(q₁, b)·q̇₁.
        jacobian[FEATURE_QD1][PRIMARY_Q1] = state.exit.a_qq * qd1;
        jacobian[FEATURE_QD1][PRIMARY_QD1] = state.exit.a_q;
        jacobian[FEATURE_QD1][PRIMARY_SLOPE] = state.exit.a_qb * s * qd1;
        let d_linear = s * inputs.z_sum;
        jacobian[FEATURE_LINEAR0][PRIMARY_SLOPE] = d_linear;
        jacobian[FEATURE_LINEAR1][PRIMARY_SLOPE] = d_linear;
        jacobian
    }

    /// `Σ_f g_f · ∂²f/∂p_a∂p_b` for the anchored map: the two anchors'
    /// curvature in `(q, b)` and the rate feature's, which is a third
    /// derivative of `α`.
    #[inline]
    fn add_curvature_from(
        state: &AnchoredRowState,
        primaries: &[f64; STATIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
        feature_gradient: &[f64; RIGID_FEATURE_DIMENSION],
        hessian: &mut [[f64; STATIC_SLOPE_PRIMARIES]; STATIC_SLOPE_PRIMARIES],
    ) {
        let s = inputs.probit_scale;
        let s2 = s * s;
        let qd1 = primaries[PRIMARY_QD1];
        let mut add = |a: usize, b: usize, value: f64| {
            hessian[a][b] += value;
            if a != b {
                hessian[b][a] += value;
            }
        };
        let g0 = feature_gradient[FEATURE_Q0];
        if g0 != 0.0 {
            add(PRIMARY_Q0, PRIMARY_Q0, g0 * state.entry.a_qq);
            add(PRIMARY_Q0, PRIMARY_SLOPE, g0 * state.entry.a_qb * s);
            add(PRIMARY_SLOPE, PRIMARY_SLOPE, g0 * state.entry.a_bb * s2);
        }
        let g1 = feature_gradient[FEATURE_Q1];
        if g1 != 0.0 {
            add(PRIMARY_Q1, PRIMARY_Q1, g1 * state.exit.a_qq);
            add(PRIMARY_Q1, PRIMARY_SLOPE, g1 * state.exit.a_qb * s);
            add(PRIMARY_SLOPE, PRIMARY_SLOPE, g1 * state.exit.a_bb * s2);
        }
        let gd = feature_gradient[FEATURE_QD1];
        if gd != 0.0 {
            add(PRIMARY_Q1, PRIMARY_Q1, gd * state.exit.a_qqq * qd1);
            add(PRIMARY_Q1, PRIMARY_QD1, gd * state.exit.a_qq);
            add(PRIMARY_Q1, PRIMARY_SLOPE, gd * state.exit.a_qqb * s * qd1);
            add(PRIMARY_QD1, PRIMARY_SLOPE, gd * state.exit.a_qb * s);
            add(PRIMARY_SLOPE, PRIMARY_SLOPE, gd * state.exit.a_qbb * s2 * qd1);
        }
    }

    /// How many features primary `axis` reaches.
    #[inline(always)]
    fn active_feature_count(axis: usize) -> usize {
        match axis {
            PRIMARY_Q1 => 2,
            PRIMARY_SLOPE => ANCHORED_SLOPE_ACTIVE_FEATURES.len(),
            _ => 1,
        }
    }

    /// The `slot`-th feature primary `axis` reaches.
    #[inline(always)]
    fn active_feature(axis: usize, slot: usize) -> usize {
        match (axis, slot) {
            (PRIMARY_Q1, 0) => FEATURE_Q1,
            (PRIMARY_Q1, _) => FEATURE_QD1,
            (PRIMARY_SLOPE, _) => ANCHORED_SLOPE_ACTIVE_FEATURES[slot],
            (identity, _) => identity,
        }
    }

    /// The f64 features and the feature-space derivatives of the row program
    /// at a solved row state, admission included.
    #[inline]
    fn program_order2(
        state: &AnchoredRowState,
        primaries: &[f64; STATIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> Result<
        (
            f64,
            [f64; RIGID_FEATURE_DIMENSION],
            [[f64; RIGID_FEATURE_DIMENSION]; RIGID_FEATURE_DIMENSION],
        ),
        String,
    > {
        let features = Self::features_from(state, primaries, inputs);
        let (value, feature_gradient, feature_hessian, [neg_eta0, neg_eta1, adjusted_derivative]) =
            rigid_feature_frame_order2(
                &features,
                inputs.wi,
                inputs.wi_entry,
                inputs.di,
                inputs.probit_scale,
                follow_up_varying_flag::<STATIC_SLOPE_PRIMARIES, Self>(),
            );
        validate_rigid_row_admission::<STATIC_SLOPE_PRIMARIES, Self>(
            primaries[PRIMARY_QD1],
            inputs,
            neg_eta0,
            neg_eta1,
            adjusted_derivative,
        )?;
        Ok((value, feature_gradient, feature_hessian))
    }
}

/// The anchored frame's affine primaries: `q̇₁` alone (gam#2928). `α(q, b)` is
/// nonlinear in both location channels, but the rate feature `α_q(q₁, b)·q̇₁`
/// is degree one in `q̇₁` and nothing else the frame builds reads `q̇₁`, so
/// every block with two `q̇₁` legs is structurally zero exactly as in
/// [`RIGID_LINEAR_MASK`].
pub(crate) const ANCHORED_LINEAR_MASK: u32 = 1 << PRIMARY_QD1;

impl SlopeRowGeometry<STATIC_SLOPE_PRIMARIES> for AnchoredStaticSlopeGeometry {
    const FOLLOW_UP_VARYING: bool = false;
    const ANCHORED: bool = true;
    const NAME: &'static str = "time-constant slope on a declared latent law";
    type Tower3 = SparseTower3<STATIC_SLOPE_PRIMARIES, ANCHORED_LINEAR_MASK>;
    type Tower4 = SparseTower4<STATIC_SLOPE_PRIMARIES, ANCHORED_LINEAR_MASK>;

    /// The anchored frame over any jet. Both anchors are solved on the real
    /// values through the row's slots, and a jet carrier receives each one's
    /// Taylor table composed with its own primaries
    /// ([`AnchoredStaticSlopeGeometry::anchored_channels`]). A failed solve or
    /// an underflowed table leaves the location channels `NaN`, which the row
    /// program's admission rejects as a non-finite signed margin.
    #[inline]
    fn feature_frame<T: JetField + Clone>(
        primaries: &[T; STATIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> [T; RIGID_FEATURE_DIMENSION] {
        let slope = &primaries[PRIMARY_SLOPE];
        let observed = slope.scale(inputs.probit_scale);
        let zero = slope.constant_like(0.0);
        let linear = observed.scale(inputs.z_sum);
        let [alpha0, alpha1, rate] =
            Self::anchored_channels(primaries, &observed, inputs).unwrap_or_else(|reason| {
                log::trace!(
                    "[survival-marginal-slope anchor] row {}: {reason}; the row is refused \
                     through its non-finite signed margin",
                    inputs.row
                );
                let refused = slope.constant_like(f64::NAN);
                [refused.clone(), refused.clone(), refused]
            });
        [
            alpha0,
            alpha1,
            rate,
            linear.clone(),
            linear,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero,
        ]
    }

    #[inline]
    fn score_sensitivity(
        primaries: &[f64; STATIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> ScoreSensitivity<STATIC_SLOPE_PRIMARIES> {
        // The anchor is a functional of the LAW, not of the row's own score,
        // so the score reaches the frame exactly as it does the Gaussian one:
        // through the two linear channels.
        StaticSlopeGeometry::score_sensitivity(primaries, inputs)
    }

    fn row_order2(
        primaries: &[f64; STATIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> Result<
        (
            f64,
            [f64; STATIC_SLOPE_PRIMARIES],
            [[f64; STATIC_SLOPE_PRIMARIES]; STATIC_SLOPE_PRIMARIES],
        ),
        String,
    > {
        const P: usize = STATIC_SLOPE_PRIMARIES;
        let state = Self::solve_row(primaries, inputs)?;
        let (value, feature_gradient, feature_hessian) =
            Self::program_order2(&state, primaries, inputs)?;
        let jacobian = Self::jacobian_from(&state, primaries, inputs);
        let mut gradient = [0.0; P];
        let mut hessian = [[0.0; P]; P];
        order2_feature_pullback_into(
            &feature_gradient,
            &feature_hessian,
            &jacobian,
            Self::active_feature_count,
            Self::active_feature,
            &mut gradient,
            &mut hessian,
            |gradient, hessian| {
                Self::add_curvature_from(&state, primaries, inputs, gradient, hessian)
            },
        );
        Ok((value, gradient, hessian))
    }

    fn row_primary_mixed_in_z(
        primaries: &[f64; STATIC_SLOPE_PRIMARIES],
        inputs: &RigidRowInputs,
    ) -> Result<[f64; STATIC_SLOPE_PRIMARIES], String> {
        let state = Self::solve_row(primaries, inputs)?;
        let (_, feature_gradient, feature_hessian) =
            Self::program_order2(&state, primaries, inputs)?;
        let jacobian = Self::jacobian_from(&state, primaries, inputs);
        Ok(mixed_in_z_from_feature_derivatives(
            &feature_gradient,
            &feature_hessian,
            &jacobian,
            Self::active_feature_count,
            Self::active_feature,
            &Self::score_sensitivity(primaries, inputs),
        ))
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
        if $family.anchored_law_active() {
            // A declared latent law runs the anchored frame (gam#2923). The
            // family refuses to combine it with a follow-up-varying slope at
            // construction, so this branch is the time-constant frame only.
            const $primaries: usize = STATIC_SLOPE_PRIMARIES;
            type $geometry = AnchoredStaticSlopeGeometry;
            $body
        } else if $family.slope_is_follow_up_varying() {
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

    fn inputs(probit_scale: f64, z_sum: f64, covariance_ones: f64, di: f64) -> RigidRowInputs<'static> {
        RigidRowInputs {
            row: 0,
            wi: 0.75,
            wi_entry: 0.75,
            di,
            z_sum,
            covariance_ones,
            probit_scale,
            qd1_lower: -1.0,
            anchor: None,
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

/// The anchored frame's derivative blocks (gam#2923), every one against an
/// independent witness: the Gaussian frame on a Gaussian law, central
/// differences on a skewed one, and the jet towers against the direct
/// lowering.
#[cfg(test)]
mod anchored_frame_tests {
    use super::super::test_support::{gauss_hermite_probabilists, skewed_grid};
    use super::*;
    use gam_math::jet_scalar::JetScalar;

    fn gaussian_law() -> AnchorGridOwned {
        let (nodes, weights) = gauss_hermite_probabilists(65).expect("Gauss–Hermite law");
        AnchorGridOwned::new(nodes, weights)
    }

    fn skewed_law() -> AnchorGridOwned {
        skewed_grid()
    }

    fn inputs<'a>(
        law: &'a AnchorGridOwned,
        probit_scale: f64,
        z_sum: f64,
        wi: f64,
        di: f64,
    ) -> RigidRowInputs<'a> {
        RigidRowInputs {
            row: 0,
            wi,
            wi_entry: wi,
            di,
            z_sum,
            covariance_ones: 1.0,
            probit_scale,
            qd1_lower: 1e-6,
            anchor: Some(AnchorRowContext {
                grid: law.view(),
                roots: None,
            }),
        }
    }

    struct Grid(u64);

    impl Grid {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    fn close(left: f64, right: f64, tol: f64) -> bool {
        (left - right).abs() <= tol * (1.0 + left.abs().max(right.abs()))
    }

    /// Gaussian is the special case: on a Gauss–Hermite law the anchored frame
    /// reproduces the closed-form frame's value, gradient and Hessian to
    /// quadrature tolerance, on event and censored rows, with and without a
    /// frailty scale.
    #[test]
    fn anchored_frame_on_a_gaussian_law_is_the_gaussian_frame() {
        let law = gaussian_law();
        let mut grid = Grid(0x2923_0001);
        let mut worst = 0.0_f64;
        for _ in 0..300 {
            let probit_scale = if grid.next() > 0.0 { 1.0 } else { 0.85 };
            let z_sum = grid.next() * 1.5;
            let wi = 0.6 + grid.next().abs();
            let di = if grid.next() > 0.0 { 1.0 } else { 0.0 };
            let anchored = inputs(&law, probit_scale, z_sum, wi, di);
            let gaussian = RigidRowInputs {
                anchor: None,
                covariance_ones: 1.0,
                ..anchored
            };
            let primaries = [
                grid.next() * 1.5,
                grid.next() * 1.5,
                0.5 + grid.next().abs() * 2.0,
                grid.next() * 1.2,
            ];
            let (value_a, gradient_a, hessian_a) =
                rigid_row_order2::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>(
                    &primaries, &anchored,
                )
                .expect("anchored frame admits the row");
            let (value_g, gradient_g, hessian_g) =
                rigid_row_order2::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(
                    &primaries, &gaussian,
                )
                .expect("Gaussian frame admits the row");
            let mut check = |left: f64, right: f64, what: &str| {
                let scale = 1.0 + left.abs().max(right.abs());
                worst = worst.max((left - right).abs() / scale);
                assert!(
                    close(left, right, 1e-7),
                    "{what}: anchored {left:+.12e} vs Gaussian {right:+.12e}"
                );
            };
            check(value_a, value_g, "value");
            for axis in 0..STATIC_SLOPE_PRIMARIES {
                check(gradient_a[axis], gradient_g[axis], "gradient");
                for other in 0..STATIC_SLOPE_PRIMARIES {
                    check(hessian_a[axis][other], hessian_g[axis][other], "Hessian");
                }
            }
        }
        assert!(worst <= 1e-7, "worst relative disagreement {worst:.3e}");
    }

    /// On a skewed law the anchored gradient and Hessian match central
    /// differences of the anchored value, and the closed form is measurably
    /// NOT the same model.
    #[test]
    fn anchored_frame_derivatives_match_central_differences() {
        let law = skewed_law();
        let mut grid = Grid(0x2923_0002);
        let mut worst = 0.0_f64;
        let mut gaussian_gap = 0.0_f64;
        for _ in 0..200 {
            let row = inputs(
                &law,
                if grid.next() > 0.0 { 1.0 } else { 0.9 },
                grid.next() * 1.2,
                0.7 + grid.next().abs() * 0.3,
                if grid.next() > 0.0 { 1.0 } else { 0.0 },
            );
            let primaries = [
                grid.next() * 1.2,
                grid.next() * 1.2,
                0.8 + grid.next().abs() * 2.0,
                grid.next() * 0.9,
            ];
            let value_at = |point: &[f64; STATIC_SLOPE_PRIMARIES]| -> f64 {
                rigid_row_value::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>(point, &row)
                    .expect("anchored value")
            };
            let (value, gradient, hessian) =
                rigid_row_order2::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>(
                    &primaries, &row,
                )
                .expect("anchored frame admits the row");
            assert!(close(value, value_at(&primaries), 1e-13), "value paths agree");
            let gaussian = RigidRowInputs {
                anchor: None,
                ..row
            };
            let (closed_form, _, _) =
                rigid_row_order2::<STATIC_SLOPE_PRIMARIES, StaticSlopeGeometry>(
                    &primaries, &gaussian,
                )
                .expect("Gaussian frame admits the row");
            gaussian_gap = gaussian_gap.max((closed_form - value).abs());

            let step = 1e-5;
            for axis in 0..STATIC_SLOPE_PRIMARIES {
                let mut up = primaries;
                let mut down = primaries;
                up[axis] += step;
                down[axis] -= step;
                let finite = (value_at(&up) - value_at(&down)) / (2.0 * step);
                let error = (finite - gradient[axis]).abs() / (1.0 + finite.abs());
                worst = worst.max(error);
                assert!(
                    error <= 5e-6,
                    "gradient axis {axis}: analytic {:+.12e} vs central difference {finite:+.12e}",
                    gradient[axis]
                );
                // Hessian rows from central differences of the analytic gradient.
                let (_, gradient_up, _) =
                    rigid_row_order2::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>(
                        &up, &row,
                    )
                    .expect("up");
                let (_, gradient_down, _) =
                    rigid_row_order2::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>(
                        &down, &row,
                    )
                    .expect("down");
                for other in 0..STATIC_SLOPE_PRIMARIES {
                    let finite = (gradient_up[other] - gradient_down[other]) / (2.0 * step);
                    let error = (finite - hessian[axis][other]).abs() / (1.0 + finite.abs());
                    worst = worst.max(error);
                    assert!(
                        error <= 5e-6,
                        "Hessian [{axis}][{other}]: analytic {:+.12e} vs central difference {finite:+.12e}",
                        hessian[axis][other]
                    );
                }
            }
        }
        assert!(worst <= 5e-6, "worst relative disagreement {worst:.3e}");
        assert!(
            gaussian_gap > 1e-3,
            "on a skewed law the closed form must be a different likelihood; largest row gap {gaussian_gap:.3e}"
        );
    }

    /// The dense towers the all-axes paths build carry the same gradient and
    /// Hessian as the direct lowering, and their third and fourth tensors are
    /// the finite differences of the orders below.
    #[test]
    fn anchored_frame_towers_match_the_direct_lowering_and_finite_differences() {
        let law = skewed_law();
        let mut grid = Grid(0x2923_0003);
        for _ in 0..60 {
            let row = inputs(
                &law,
                1.0,
                grid.next() * 1.2,
                0.7 + grid.next().abs() * 0.3,
                if grid.next() > 0.0 { 1.0 } else { 0.0 },
            );
            let primaries = [
                grid.next() * 1.0,
                grid.next() * 1.0,
                0.8 + grid.next().abs() * 1.5,
                grid.next() * 0.8,
            ];
            // The frame's own production tower, with its `q̇₁` mask: a wrong
            // linearity declaration panics in `check_contract` here.
            let tower_at = |point: &[f64; STATIC_SLOPE_PRIMARIES]| {
                let vars: [<AnchoredStaticSlopeGeometry as SlopeRowGeometry<
                    STATIC_SLOPE_PRIMARIES,
                >>::Tower4; STATIC_SLOPE_PRIMARIES] =
                    std::array::from_fn(|axis| JetScalar::variable(point[axis], axis));
                rigid_row_nll::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry, _>(
                    &vars, &row,
                )
                .expect("anchored tower")
            };
            let tower = tower_at(&primaries);
            let (value, gradient, hessian) =
                rigid_row_order2::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>(
                    &primaries, &row,
                )
                .expect("direct lowering");
            assert!(close(tower.v, value, 1e-12), "tower value {} vs {value}", tower.v);
            for a in 0..STATIC_SLOPE_PRIMARIES {
                assert!(
                    close(tower.g[a], gradient[a], 1e-9),
                    "tower gradient [{a}] {} vs direct {}",
                    tower.g[a],
                    gradient[a]
                );
                for b in 0..STATIC_SLOPE_PRIMARIES {
                    assert!(
                        close(tower.h[a][b], hessian[a][b], 1e-8),
                        "tower Hessian [{a}][{b}] {} vs direct {}",
                        tower.h[a][b],
                        hessian[a][b]
                    );
                }
            }
            let step = 1e-4;
            for c in 0..STATIC_SLOPE_PRIMARIES {
                let mut up = primaries;
                let mut down = primaries;
                up[c] += step;
                down[c] -= step;
                let tower_up = tower_at(&up);
                let tower_down = tower_at(&down);
                for a in 0..STATIC_SLOPE_PRIMARIES {
                    for b in 0..STATIC_SLOPE_PRIMARIES {
                        let finite = (tower_up.h[a][b] - tower_down.h[a][b]) / (2.0 * step);
                        assert!(
                            close(tower.t3[a][b][c], finite, 2e-5),
                            "t3 [{a}][{b}][{c}] {} vs central difference {finite}",
                            tower.t3[a][b][c]
                        );
                        for d in 0..STATIC_SLOPE_PRIMARIES {
                            let finite =
                                (tower_up.t3[a][b][d] - tower_down.t3[a][b][d]) / (2.0 * step);
                            assert!(
                                close(tower.t4[a][b][d][c], finite, 5e-5),
                                "t4 [{a}][{b}][{d}][{c}] {} vs central difference {finite}",
                                tower.t4[a][b][d][c]
                            );
                        }
                    }
                }
            }
        }
    }

    /// Delayed-entry rows deep in both tails (gam#2941): the entry anchor at
    /// `q₀` and the exit anchor at `q₁` both at `|q|` of 84–90, where every
    /// node's density underflows — a 32-row delayed-entry calibration refused
    /// all its startup seeds with `G_α = 0` at `q ≈ −84`. On a skewed and a
    /// Gauss–Hermite law, event and censored, the direct lowering and the
    /// production tower admit the row with finite channels and agree, and the
    /// lowering's gradient and Hessian are central differences of the value.
    #[test]
    fn anchored_frame_admits_delayed_entry_rows_deep_in_both_tails() {
        let skewed = skewed_law();
        let gaussian = gaussian_law();
        for (label, law) in [("skewed", &skewed), ("gaussian", &gaussian)] {
            // −83.970… with slope −0.0966 is the point a 32-row delayed-entry
            // calibration refused on every startup seed.
            for &q0 in &[-90.0, -84.0, -83.97022428333848, 84.0, 90.0] {
                for &slope in &[-2.5, -0.09658543743257131, 0.4, 3.0] {
                    for &di in &[0.0, 1.0] {
                        let row = inputs(law, 1.0, 0.8, 1.0, di);
                        // The exit index after the entry index, as follow-up moves on.
                        let primaries = [q0, q0 + 0.6, 1.3, slope];
                        let context = format!("{label} q₀={q0} q₁={} g={slope} event={di}", q0 + 0.6);
                        let (value, gradient, hessian) =
                            rigid_row_order2::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>(
                                &primaries, &row,
                            )
                            .unwrap_or_else(|e| panic!("{context}: direct lowering refused: {e}"));
                        assert!(
                            value.is_finite()
                                && gradient.iter().all(|g| g.is_finite())
                                && hessian.iter().flatten().all(|h| h.is_finite()),
                            "{context}: non-finite direct lowering"
                        );
                        let vars: [<AnchoredStaticSlopeGeometry as SlopeRowGeometry<
                            STATIC_SLOPE_PRIMARIES,
                        >>::Tower4; STATIC_SLOPE_PRIMARIES] =
                            std::array::from_fn(|axis| JetScalar::variable(primaries[axis], axis));
                        let tower =
                            rigid_row_nll::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry, _>(
                                &vars, &row,
                            )
                            .unwrap_or_else(|e| panic!("{context}: tower refused: {e}"));
                        assert!(close(tower.v, value, 1e-12), "{context}: tower value {} vs {value}", tower.v);
                        for a in 0..STATIC_SLOPE_PRIMARIES {
                            assert!(
                                close(tower.g[a], gradient[a], 1e-9),
                                "{context}: tower gradient [{a}] {} vs direct {}",
                                tower.g[a],
                                gradient[a]
                            );
                            for b in 0..STATIC_SLOPE_PRIMARIES {
                                assert!(
                                    close(tower.h[a][b], hessian[a][b], 1e-8),
                                    "{context}: tower Hessian [{a}][{b}] {} vs direct {}",
                                    tower.h[a][b],
                                    hessian[a][b]
                                );
                            }
                        }
                        let value_at = |point: &[f64; STATIC_SLOPE_PRIMARIES]| -> f64 {
                            rigid_row_value::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>(point, &row)
                                .unwrap_or_else(|e| panic!("{context}: value refused: {e}"))
                        };
                        for axis in 0..STATIC_SLOPE_PRIMARIES {
                            let step = 1e-5 * (1.0 + primaries[axis].abs());
                            let mut up = primaries;
                            let mut down = primaries;
                            up[axis] += step;
                            down[axis] -= step;
                            let width = up[axis] - down[axis];
                            let finite = (value_at(&up) - value_at(&down)) / width;
                            assert!(
                                close(finite, gradient[axis], 5e-6),
                                "{context}: gradient axis {axis}: analytic {:+.12e} vs central difference {finite:+.12e}",
                                gradient[axis]
                            );
                            let (_, gradient_up, _) =
                                rigid_row_order2::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>(
                                    &up, &row,
                                )
                                .expect("up");
                            let (_, gradient_down, _) =
                                rigid_row_order2::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>(
                                    &down, &row,
                                )
                                .expect("down");
                            for other in 0..STATIC_SLOPE_PRIMARIES {
                                let finite = (gradient_up[other] - gradient_down[other]) / width;
                                assert!(
                                    close(finite, hessian[axis][other], 5e-6),
                                    "{context}: Hessian [{axis}][{other}]: analytic {:+.12e} vs central difference {finite:+.12e}",
                                    hessian[axis][other]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The generated-regressor channel `∂(∇ℓ)/∂z` of the anchored frame is the
    /// finite difference of its gradient in the row's score.
    #[test]
    fn anchored_frame_mixed_in_z_matches_central_differences() {
        let law = skewed_law();
        let mut grid = Grid(0x2923_0004);
        for _ in 0..100 {
            let z_sum = grid.next() * 1.2;
            let wi = 0.7 + grid.next().abs() * 0.3;
            let di = if grid.next() > 0.0 { 1.0 } else { 0.0 };
            let primaries = [
                grid.next() * 1.0,
                grid.next() * 1.0,
                0.8 + grid.next().abs() * 1.5,
                grid.next() * 0.8,
            ];
            let row = inputs(&law, 0.95, z_sum, wi, di);
            let mixed = rigid_row_primary_mixed_in_z::<
                STATIC_SLOPE_PRIMARIES,
                AnchoredStaticSlopeGeometry,
            >(&primaries, &row)
            .expect("mixed channel");
            let step = 1e-5;
            let gradient_at = |z: f64| {
                let row = inputs(&law, 0.95, z, wi, di);
                rigid_row_order2::<STATIC_SLOPE_PRIMARIES, AnchoredStaticSlopeGeometry>(
                    &primaries, &row,
                )
                .expect("gradient")
                .1
            };
            let up = gradient_at(z_sum + step);
            let down = gradient_at(z_sum - step);
            for axis in 0..STATIC_SLOPE_PRIMARIES {
                let finite = (up[axis] - down[axis]) / (2.0 * step);
                assert!(
                    close(mixed[axis], finite, 5e-6),
                    "mixed [{axis}] {} vs central difference {finite}",
                    mixed[axis]
                );
            }
        }
    }
}

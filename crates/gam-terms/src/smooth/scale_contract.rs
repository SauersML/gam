//! Exhaustive coordinate-scale contracts for smooth basis construction.
//!
//! A scale contract is not a human-readable tag.  It is a typed description of
//! the pullback that must be applied to every coordinate-bearing part of a
//! basis: design, penalty, coordinate derivatives, null geometry, and
//! dimensionful hyperparameters.  [`SmoothBasisSpec::scale_contract`] matches
//! every enum arm without a wildcard, so adding a new basis variant is a compile
//! error until its law is declared here.

use super::*;
use crate::basis::{OneDimensionalBoundary, SphereMethod};

/// Builder-level basis family.  Variants that share a `SmoothBasisSpec` arm but
/// use different mathematics (cyclic/open/natural-cubic, Wahba/harmonic sphere,
/// pure/hybrid Duchon, and factor-wrapper flavours) remain distinct here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasisScaleFamily {
    ByVariableNumeric,
    ByVariableFactor,
    FactorSumToZero,
    OpenBSpline,
    CyclicBSpline,
    NaturalCubic,
    BySmoothNumeric,
    BySmoothFactor,
    FactorSmoothFs,
    FactorSmoothSz,
    FactorSmoothRe,
    ThinPlate,
    SphereWahba,
    SphereHarmonic,
    ConstantCurvature,
    Matern,
    MeasureJet,
    PureDuchon,
    HybridDuchon,
    Pca,
    TensorBSpline,
}

impl BasisScaleFamily {
    /// Canonical registry order.  The registry-completeness test walks this
    /// array and compares it with a concrete `SmoothBasisSpec` zoo.
    pub const ALL: [Self; 21] = [
        Self::ByVariableNumeric,
        Self::ByVariableFactor,
        Self::FactorSumToZero,
        Self::OpenBSpline,
        Self::CyclicBSpline,
        Self::NaturalCubic,
        Self::BySmoothNumeric,
        Self::BySmoothFactor,
        Self::FactorSmoothFs,
        Self::FactorSmoothSz,
        Self::FactorSmoothRe,
        Self::ThinPlate,
        Self::SphereWahba,
        Self::SphereHarmonic,
        Self::ConstantCurvature,
        Self::Matern,
        Self::MeasureJet,
        Self::PureDuchon,
        Self::HybridDuchon,
        Self::Pca,
        Self::TensorBSpline,
    ];
}

/// Coordinate action under which the declared law is exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisCoordinateScaleAction {
    /// `x' = a*x + b`, `a > 0`; all knots/endpoints move by the same map.
    PositiveAffineAbscissa,
    /// Each tensor margin has its own positive affine map.
    IndependentPositiveAffineAxes,
    /// `x' = a*x + b`, with one `a > 0` shared by every Euclidean axis;
    /// centers, periodic lengths, and kernel range move with the same map.
    /// (A diagonal anisotropy cannot represent a general rotated metric, so
    /// rotation is deliberately not claimed by this scale-only contract.)
    UniformEuclideanScale,
    /// Degrees/radians are two coordinate encodings of the same point on S².
    IntrinsicAngularUnitConversion,
    /// `x'=a*x`, `kappa'=kappa/a^2`, `ell'=a*ell` in the stereographic chart.
    ConstantCurvatureChartSimilarity,
    /// `x'=a*x`, `mean'=a*mean`, `loadings'=loadings/a`, preserving PCA scores.
    PcaScoreGauge,
    /// Multiply an inner smooth by a numeric `by` coordinate.
    NumericModulation,
    /// Replicate/gate an inner smooth using dimensionless categorical labels.
    DiscreteReplication,
}

/// Transformation of the emitted design matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisDesignScaleLaw {
    /// The declared joint coordinate/parameter action leaves every design entry
    /// invariant (up to floating-point roundoff).
    Invariant,
    /// Scaling only the numeric multiplier by `a` sends `X -> a*X`; scaling an
    /// inner coordinate follows the child's law.
    NumericMultiplierDegreeOne,
    /// Wrapper design is a row gate/replication of its invariant inner design.
    ReplicatedInner,
    /// Tensor design is the row-wise product of invariant marginal designs.
    TensorProductOfMarginals,
    /// A random-intercept/slope block has alternating degree-zero and
    /// degree-one columns in each factor level: intercept columns are invariant
    /// and slope columns gain one power of the abscissa scale.
    RandomInterceptSlopeDegreesZeroAndOne,
}

/// Transformation of active penalty matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisPenaltyScaleLaw {
    /// The raw functional has the stated coordinate homogeneity, while the
    /// emitted unit-Frobenius matrix is invariant and its recorded
    /// `normalization_scale` carries that power.
    FrobeniusNormalizedRawPower(i32),
    /// The raw derivative-energy functional has exact homogeneity
    /// `a^(1 - 2*order)`; keeping `order` typed avoids truncating a structural
    /// basis parameter into a fixed-width exponent.
    FrobeniusNormalizedDerivativeOrder { order: usize },
    /// Each tensor marginal carries its own `1 - 2*marginal_order` raw power;
    /// every emitted normalized Kronecker penalty is invariant.
    FrobeniusNormalizedPerMarginalDerivativeOrder,
    /// Kernel/energy construction and its normalization are invariant under the
    /// complete declared parameter pullback.
    FrobeniusNormalizedInvariant,
    /// The physical RKHS Gram is emitted without arbitrary normalization but
    /// is itself invariant under the complete coordinate/parameter pullback.
    InvariantPhysicalRkhsGram,
    /// Penalties are copied/congruence-transformed into wrapper blocks without
    /// introducing a coordinate scale of their own.
    ReplicatedInner,
    /// PCA uses the empirical function-mass Gram `X_score'X_score/n`; invariant
    /// scores therefore give an exactly invariant penalty and null geometry.
    InvariantFunctionMass,
}

/// Transformation of analytic input/hyperparameter derivatives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisDerivativeScaleLaw {
    /// An order-q coordinate derivative transforms as `D'^q = a^-q D^q`.
    InverseCoordinatePower { maximum_order: usize },
    /// The same coordinate law, while derivatives in `log(kappa)=-log(ell)`
    /// remain invariant under `ell -> a*ell`.
    InverseCoordinatePowerAndInvariantLogRange { maximum_order: usize },
    /// Under `kappa'=kappa/a^2`, first and second derivatives with respect to
    /// the numeric curvature coordinate gain factors `a^2` and `a^4`.
    ConstantCurvatureParameterPowers,
    /// Derivatives live in the intrinsic angular chart; changing degrees to
    /// radians applies the ordinary inverse unit-conversion chain rule.
    IntrinsicAngularChainRule,
    /// Wrapper derivatives obey the product rule with the numeric multiplier.
    NumericModulationProductRule,
    /// Discrete gates/replications introduce no derivative coordinate.
    DelegatedToInner,
    /// Tensor partial derivatives take the inverse power on the differentiated
    /// margin and multiply the invariant values of all other margins.
    TensorMarginalProductRule { maximum_order: usize },
}

/// Transformation of the structural penalty null space / identifiability
/// section.  This is explicit because equal penalty ranks alone do not prove
/// that the same unpenalized functions survived a coordinate change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisNullGeometryScaleLaw {
    PolynomialPullback,
    CyclicConstantMode,
    TensorProductPullback,
    EuclideanPolynomialPullback,
    CenterConstraintPullback,
    MeasureJetAffineHeadPullback,
    IntrinsicHarmonicSubspace,
    ConstantCurvatureCenterConstraint,
    PcaScoreCongruence,
    FullRankRandomEffect,
    ReplicatedInner,
}

/// Every dimensionful object that must move with a coordinate-unit change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DimensionfulBasisParameter {
    Knots,
    DomainEndpoints,
    Periods,
    Centers,
    InputStandardDeviations,
    LengthScale,
    MeasureJetScaleBand,
    MeasureJetCoordinateNoise,
    Curvature,
    PcaCenterMean,
    PcaLoadings,
    NumericByMultiplier,
}

/// Integer homogeneity of a dimensionful parameter under `x -> a*x`.
/// `power=1` means multiply by `a`, `-1` divide by `a`, and `-2` divide by
/// `a^2`.  Angular units use `power=1` with the degree/radian conversion in
/// place of `a`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimensionfulParameterScale {
    pub parameter: DimensionfulBasisParameter,
    pub power: i8,
}

impl DimensionfulParameterScale {
    const fn new(parameter: DimensionfulBasisParameter, power: i8) -> Self {
        Self { parameter, power }
    }
}

/// How construction realizes the input frame.  This is deliberately private:
/// callers consume it through [`BasisScaleContract::normalize_euclidean_frame`]
/// rather than branching on a second public policy enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputFrameNormalization {
    Parameterized,
    AutoStandardizedOriginalUnits,
    AutoStandardizedFreshOriginalReplayRealized,
    Intrinsic,
    PcaGauge,
    Delegated,
}

/// Complete scale declaration for one node of a smooth-basis tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasisScaleContract {
    pub family: BasisScaleFamily,
    pub coordinate_action: BasisCoordinateScaleAction,
    pub design: BasisDesignScaleLaw,
    pub penalty: BasisPenaltyScaleLaw,
    pub derivatives: BasisDerivativeScaleLaw,
    pub null_geometry: BasisNullGeometryScaleLaw,
    pub dimensionful_parameters: Vec<DimensionfulParameterScale>,
    /// Inner/marginal contracts in construction order.  Wrappers have one
    /// child; tensors have one child per marginal; leaf bases have none.
    pub children: Vec<BasisScaleContract>,
    input_frame: InputFrameNormalization,
}

/// Realized Euclidean input frame returned to the spatial constructors.
pub(super) struct NormalizedEuclideanFrame {
    pub coordinates: Array2<f64>,
    pub input_scales: Option<Vec<f64>>,
    pub length_scale: Option<f64>,
}

impl BasisScaleContract {
    fn leaf(
        family: BasisScaleFamily,
        coordinate_action: BasisCoordinateScaleAction,
        design: BasisDesignScaleLaw,
        penalty: BasisPenaltyScaleLaw,
        derivatives: BasisDerivativeScaleLaw,
        null_geometry: BasisNullGeometryScaleLaw,
        dimensionful_parameters: Vec<DimensionfulParameterScale>,
        input_frame: InputFrameNormalization,
    ) -> Self {
        Self {
            family,
            coordinate_action,
            design,
            penalty,
            derivatives,
            null_geometry,
            dimensionful_parameters,
            children: Vec::new(),
            input_frame,
        }
    }

    fn wrapper(
        family: BasisScaleFamily,
        coordinate_action: BasisCoordinateScaleAction,
        design: BasisDesignScaleLaw,
        derivatives: BasisDerivativeScaleLaw,
        child: BasisScaleContract,
        dimensionful_parameters: Vec<DimensionfulParameterScale>,
    ) -> Self {
        Self {
            family,
            coordinate_action,
            design,
            penalty: BasisPenaltyScaleLaw::ReplicatedInner,
            derivatives,
            null_geometry: BasisNullGeometryScaleLaw::ReplicatedInner,
            dimensionful_parameters,
            children: vec![child],
            input_frame: InputFrameNormalization::Delegated,
        }
    }

    /// Standardize a Euclidean spatial input using this family's declared
    /// original-unit/replay law.  This is the sole construction path for the
    /// ThinPlate, Matérn, Duchon, and MeasureJet `input_scales` fields.
    pub(super) fn normalize_euclidean_frame(
        &self,
        mut coordinates: Array2<f64>,
        stored_scales: Option<&[f64]>,
        length_scale: Option<f64>,
    ) -> Result<NormalizedEuclideanFrame, BasisError> {
        let replay = stored_scales.is_some();
        let replay_range_is_already_realized = match self.input_frame {
            InputFrameNormalization::AutoStandardizedOriginalUnits => false,
            InputFrameNormalization::AutoStandardizedFreshOriginalReplayRealized => true,
            InputFrameNormalization::Parameterized
            | InputFrameNormalization::Intrinsic
            | InputFrameNormalization::PcaGauge
            | InputFrameNormalization::Delegated => {
                return Err(BasisError::InvalidInput(format!(
                    "basis {:?} does not declare Euclidean auto-standardization",
                    self.family
                )));
            }
        };
        let scales = match stored_scales {
            Some(scales) => {
                validate_input_scale_vector(scales, coordinates.ncols(), self.family)?;
                Some(scales.to_vec())
            }
            None => compute_spatial_input_scales(coordinates.view()),
        };

        let transformed_length = if let Some(scales) = scales.as_deref() {
            apply_input_standardization(&mut coordinates, scales);
            if replay_range_is_already_realized && replay {
                length_scale
            } else if replay_range_is_already_realized {
                length_scale.map(|ell| {
                    if ell > 0.0 {
                        compensate_length_scale_for_standardization(ell, scales)
                    } else {
                        ell
                    }
                })
            } else {
                length_scale.map(|ell| compensate_length_scale_for_standardization(ell, scales))
            }
        } else {
            length_scale
        };

        Ok(NormalizedEuclideanFrame {
            coordinates,
            input_scales: scales,
            length_scale: transformed_length,
        })
    }
}

fn scale(power: i8, parameter: DimensionfulBasisParameter) -> DimensionfulParameterScale {
    DimensionfulParameterScale::new(parameter, power)
}

fn bspline_contract(spec: &BSplineBasisSpec) -> BasisScaleContract {
    let (family, null_geometry, mut parameters) = match (&spec.knotspec, &spec.boundary) {
        (BSplineKnotSpec::NaturalCubicRegression { .. }, _) => (
            BasisScaleFamily::NaturalCubic,
            BasisNullGeometryScaleLaw::PolynomialPullback,
            vec![scale(1, DimensionfulBasisParameter::Knots)],
        ),
        (BSplineKnotSpec::PeriodicUniform { .. }, _)
        | (_, OneDimensionalBoundary::Cyclic { .. }) => (
            BasisScaleFamily::CyclicBSpline,
            BasisNullGeometryScaleLaw::CyclicConstantMode,
            vec![
                scale(1, DimensionfulBasisParameter::Knots),
                scale(1, DimensionfulBasisParameter::DomainEndpoints),
                scale(1, DimensionfulBasisParameter::Periods),
            ],
        ),
        (BSplineKnotSpec::Generate { .. }, OneDimensionalBoundary::Open)
        | (BSplineKnotSpec::Automatic { .. }, OneDimensionalBoundary::Open)
        | (BSplineKnotSpec::Provided(_), OneDimensionalBoundary::Open) => (
            BasisScaleFamily::OpenBSpline,
            BasisNullGeometryScaleLaw::PolynomialPullback,
            vec![scale(1, DimensionfulBasisParameter::Knots)],
        ),
    };
    if matches!(spec.knotspec, BSplineKnotSpec::Generate { .. }) {
        parameters.push(scale(1, DimensionfulBasisParameter::DomainEndpoints));
    }
    let penalty = if family == BasisScaleFamily::NaturalCubic {
        BasisPenaltyScaleLaw::FrobeniusNormalizedRawPower(-3)
    } else {
        BasisPenaltyScaleLaw::FrobeniusNormalizedDerivativeOrder {
            order: spec.penalty_order,
        }
    };
    BasisScaleContract::leaf(
        family,
        BasisCoordinateScaleAction::PositiveAffineAbscissa,
        BasisDesignScaleLaw::Invariant,
        penalty,
        BasisDerivativeScaleLaw::InverseCoordinatePower { maximum_order: 2 },
        null_geometry,
        parameters,
        InputFrameNormalization::Parameterized,
    )
}

fn spatial_parameters(
    include_length: bool,
    include_periods: bool,
) -> Vec<DimensionfulParameterScale> {
    let mut parameters = vec![
        scale(1, DimensionfulBasisParameter::Centers),
        scale(1, DimensionfulBasisParameter::InputStandardDeviations),
    ];
    if include_length {
        parameters.push(scale(1, DimensionfulBasisParameter::LengthScale));
    }
    if include_periods {
        parameters.push(scale(1, DimensionfulBasisParameter::Periods));
    }
    parameters
}

impl SmoothBasisSpec {
    /// Return the complete scale contract for this concrete basis tree.
    ///
    /// There is intentionally no wildcard arm: a new `SmoothBasisSpec` variant
    /// cannot compile until its design, penalty, derivative, null-space, and
    /// dimensionful-parameter laws are all selected here.
    pub fn scale_contract(&self) -> BasisScaleContract {
        match self {
            SmoothBasisSpec::ByVariable { inner, by, .. } => match by {
                ByVariableSpec::Numeric => BasisScaleContract::wrapper(
                    BasisScaleFamily::ByVariableNumeric,
                    BasisCoordinateScaleAction::NumericModulation,
                    BasisDesignScaleLaw::NumericMultiplierDegreeOne,
                    BasisDerivativeScaleLaw::NumericModulationProductRule,
                    inner.scale_contract(),
                    vec![scale(1, DimensionfulBasisParameter::NumericByMultiplier)],
                ),
                ByVariableSpec::Level { .. } => BasisScaleContract::wrapper(
                    BasisScaleFamily::ByVariableFactor,
                    BasisCoordinateScaleAction::DiscreteReplication,
                    BasisDesignScaleLaw::ReplicatedInner,
                    BasisDerivativeScaleLaw::DelegatedToInner,
                    inner.scale_contract(),
                    Vec::new(),
                ),
            },
            SmoothBasisSpec::FactorSumToZero { inner, .. } => BasisScaleContract::wrapper(
                BasisScaleFamily::FactorSumToZero,
                BasisCoordinateScaleAction::DiscreteReplication,
                BasisDesignScaleLaw::ReplicatedInner,
                BasisDerivativeScaleLaw::DelegatedToInner,
                inner.scale_contract(),
                Vec::new(),
            ),
            SmoothBasisSpec::BSpline1D { spec, .. } => bspline_contract(spec),
            SmoothBasisSpec::BySmooth { smooth, by_kind } => match by_kind {
                ByVarKind::Numeric { .. } => BasisScaleContract::wrapper(
                    BasisScaleFamily::BySmoothNumeric,
                    BasisCoordinateScaleAction::NumericModulation,
                    BasisDesignScaleLaw::NumericMultiplierDegreeOne,
                    BasisDerivativeScaleLaw::NumericModulationProductRule,
                    smooth.scale_contract(),
                    vec![scale(1, DimensionfulBasisParameter::NumericByMultiplier)],
                ),
                ByVarKind::Factor { .. } => BasisScaleContract::wrapper(
                    BasisScaleFamily::BySmoothFactor,
                    BasisCoordinateScaleAction::DiscreteReplication,
                    BasisDesignScaleLaw::ReplicatedInner,
                    BasisDerivativeScaleLaw::DelegatedToInner,
                    smooth.scale_contract(),
                    Vec::new(),
                ),
            },
            SmoothBasisSpec::FactorSmooth { spec } => match &spec.flavour {
                FactorSmoothFlavour::Fs { .. } => BasisScaleContract::wrapper(
                    BasisScaleFamily::FactorSmoothFs,
                    BasisCoordinateScaleAction::DiscreteReplication,
                    BasisDesignScaleLaw::ReplicatedInner,
                    BasisDerivativeScaleLaw::DelegatedToInner,
                    bspline_contract(&spec.marginal),
                    Vec::new(),
                ),
                FactorSmoothFlavour::Sz => BasisScaleContract::wrapper(
                    BasisScaleFamily::FactorSmoothSz,
                    BasisCoordinateScaleAction::DiscreteReplication,
                    BasisDesignScaleLaw::ReplicatedInner,
                    BasisDerivativeScaleLaw::DelegatedToInner,
                    bspline_contract(&spec.marginal),
                    Vec::new(),
                ),
                FactorSmoothFlavour::Re => BasisScaleContract {
                    family: BasisScaleFamily::FactorSmoothRe,
                    coordinate_action: BasisCoordinateScaleAction::PositiveAffineAbscissa,
                    design: BasisDesignScaleLaw::RandomInterceptSlopeDegreesZeroAndOne,
                    penalty: BasisPenaltyScaleLaw::FrobeniusNormalizedInvariant,
                    derivatives: BasisDerivativeScaleLaw::InverseCoordinatePower {
                        maximum_order: 1,
                    },
                    null_geometry: BasisNullGeometryScaleLaw::FullRankRandomEffect,
                    dimensionful_parameters: vec![scale(1, DimensionfulBasisParameter::Knots)],
                    children: vec![bspline_contract(&spec.marginal)],
                    input_frame: InputFrameNormalization::Delegated,
                },
            },
            SmoothBasisSpec::ThinPlate { .. } => BasisScaleContract::leaf(
                BasisScaleFamily::ThinPlate,
                BasisCoordinateScaleAction::UniformEuclideanScale,
                BasisDesignScaleLaw::Invariant,
                BasisPenaltyScaleLaw::FrobeniusNormalizedInvariant,
                BasisDerivativeScaleLaw::InverseCoordinatePowerAndInvariantLogRange {
                    maximum_order: 2,
                },
                BasisNullGeometryScaleLaw::EuclideanPolynomialPullback,
                spatial_parameters(true, true),
                InputFrameNormalization::AutoStandardizedOriginalUnits,
            ),
            SmoothBasisSpec::Sphere { spec, .. } => {
                let (family, parameters) = match spec.method {
                    SphereMethod::Wahba => (
                        BasisScaleFamily::SphereWahba,
                        vec![scale(1, DimensionfulBasisParameter::Centers)],
                    ),
                    SphereMethod::Harmonic => (BasisScaleFamily::SphereHarmonic, Vec::new()),
                };
                BasisScaleContract::leaf(
                    family,
                    BasisCoordinateScaleAction::IntrinsicAngularUnitConversion,
                    BasisDesignScaleLaw::Invariant,
                    BasisPenaltyScaleLaw::FrobeniusNormalizedInvariant,
                    BasisDerivativeScaleLaw::IntrinsicAngularChainRule,
                    BasisNullGeometryScaleLaw::IntrinsicHarmonicSubspace,
                    parameters,
                    InputFrameNormalization::Intrinsic,
                )
            }
            SmoothBasisSpec::ConstantCurvature { .. } => BasisScaleContract::leaf(
                BasisScaleFamily::ConstantCurvature,
                BasisCoordinateScaleAction::ConstantCurvatureChartSimilarity,
                BasisDesignScaleLaw::Invariant,
                BasisPenaltyScaleLaw::InvariantPhysicalRkhsGram,
                BasisDerivativeScaleLaw::ConstantCurvatureParameterPowers,
                BasisNullGeometryScaleLaw::ConstantCurvatureCenterConstraint,
                vec![
                    scale(1, DimensionfulBasisParameter::Centers),
                    scale(1, DimensionfulBasisParameter::LengthScale),
                    scale(-2, DimensionfulBasisParameter::Curvature),
                ],
                InputFrameNormalization::Intrinsic,
            ),
            SmoothBasisSpec::Matern { .. } => BasisScaleContract::leaf(
                BasisScaleFamily::Matern,
                BasisCoordinateScaleAction::UniformEuclideanScale,
                BasisDesignScaleLaw::Invariant,
                BasisPenaltyScaleLaw::FrobeniusNormalizedInvariant,
                BasisDerivativeScaleLaw::InverseCoordinatePowerAndInvariantLogRange {
                    maximum_order: 2,
                },
                BasisNullGeometryScaleLaw::CenterConstraintPullback,
                spatial_parameters(true, true),
                InputFrameNormalization::AutoStandardizedOriginalUnits,
            ),
            SmoothBasisSpec::MeasureJet { .. } => BasisScaleContract::leaf(
                BasisScaleFamily::MeasureJet,
                BasisCoordinateScaleAction::UniformEuclideanScale,
                BasisDesignScaleLaw::Invariant,
                BasisPenaltyScaleLaw::FrobeniusNormalizedInvariant,
                BasisDerivativeScaleLaw::InverseCoordinatePowerAndInvariantLogRange {
                    maximum_order: 2,
                },
                BasisNullGeometryScaleLaw::MeasureJetAffineHeadPullback,
                {
                    let mut parameters = spatial_parameters(true, false);
                    parameters.push(scale(1, DimensionfulBasisParameter::MeasureJetScaleBand));
                    parameters.push(scale(
                        1,
                        DimensionfulBasisParameter::MeasureJetCoordinateNoise,
                    ));
                    parameters
                },
                InputFrameNormalization::AutoStandardizedFreshOriginalReplayRealized,
            ),
            SmoothBasisSpec::Duchon { spec, .. } => {
                let family = if spec.length_scale.is_some() {
                    BasisScaleFamily::HybridDuchon
                } else {
                    BasisScaleFamily::PureDuchon
                };
                BasisScaleContract::leaf(
                    family,
                    BasisCoordinateScaleAction::UniformEuclideanScale,
                    BasisDesignScaleLaw::Invariant,
                    BasisPenaltyScaleLaw::FrobeniusNormalizedInvariant,
                    if spec.length_scale.is_some() {
                        BasisDerivativeScaleLaw::InverseCoordinatePowerAndInvariantLogRange {
                            maximum_order: 2,
                        }
                    } else {
                        BasisDerivativeScaleLaw::InverseCoordinatePower { maximum_order: 2 }
                    },
                    BasisNullGeometryScaleLaw::EuclideanPolynomialPullback,
                    spatial_parameters(spec.length_scale.is_some(), true),
                    InputFrameNormalization::AutoStandardizedOriginalUnits,
                )
            }
            SmoothBasisSpec::Pca { .. } => BasisScaleContract::leaf(
                BasisScaleFamily::Pca,
                BasisCoordinateScaleAction::PcaScoreGauge,
                BasisDesignScaleLaw::Invariant,
                BasisPenaltyScaleLaw::InvariantFunctionMass,
                BasisDerivativeScaleLaw::InverseCoordinatePower { maximum_order: 1 },
                BasisNullGeometryScaleLaw::PcaScoreCongruence,
                vec![
                    scale(1, DimensionfulBasisParameter::PcaCenterMean),
                    scale(-1, DimensionfulBasisParameter::PcaLoadings),
                ],
                InputFrameNormalization::PcaGauge,
            ),
            SmoothBasisSpec::TensorBSpline { spec, .. } => BasisScaleContract {
                family: BasisScaleFamily::TensorBSpline,
                coordinate_action: BasisCoordinateScaleAction::IndependentPositiveAffineAxes,
                design: BasisDesignScaleLaw::TensorProductOfMarginals,
                penalty: BasisPenaltyScaleLaw::FrobeniusNormalizedPerMarginalDerivativeOrder,
                derivatives: BasisDerivativeScaleLaw::TensorMarginalProductRule {
                    maximum_order: 2,
                },
                null_geometry: BasisNullGeometryScaleLaw::TensorProductPullback,
                dimensionful_parameters: Vec::new(),
                children: spec.marginalspecs.iter().map(bspline_contract).collect(),
                input_frame: InputFrameNormalization::Delegated,
            },
        }
    }

    /// Validate every scale-bearing field before construction or frozen replay.
    /// Wrapper recursion is exhaustive; no basis can bypass this check.
    pub fn validate_scale_configuration(&self) -> Result<(), BasisError> {
        let contract = self.scale_contract();
        match self {
            SmoothBasisSpec::ByVariable { inner, .. }
            | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
                inner.validate_scale_configuration()
            }
            SmoothBasisSpec::BSpline1D { .. } => Ok(()),
            SmoothBasisSpec::BySmooth { smooth, .. } => smooth.validate_scale_configuration(),
            SmoothBasisSpec::FactorSmooth { .. } => Ok(()),
            SmoothBasisSpec::ThinPlate {
                feature_cols,
                input_scales,
                ..
            }
            | SmoothBasisSpec::Matern {
                feature_cols,
                input_scales,
                ..
            }
            | SmoothBasisSpec::MeasureJet {
                feature_cols,
                input_scales,
                ..
            }
            | SmoothBasisSpec::Duchon {
                feature_cols,
                input_scales,
                ..
            } => {
                if feature_cols.is_empty() {
                    return Err(BasisError::InvalidInput(format!(
                        "basis {:?} requires at least one coordinate axis",
                        contract.family
                    )));
                }
                if let Some(scales) = input_scales {
                    validate_input_scale_vector(scales, feature_cols.len(), contract.family)?;
                }
                Ok(())
            }
            SmoothBasisSpec::Sphere { feature_cols, .. }
            | SmoothBasisSpec::ConstantCurvature { feature_cols, .. } => {
                if feature_cols.is_empty() {
                    return Err(BasisError::InvalidInput(format!(
                        "basis {:?} requires at least one coordinate axis",
                        contract.family
                    )));
                }
                Ok(())
            }
            SmoothBasisSpec::Pca { .. } => Ok(()),
            SmoothBasisSpec::TensorBSpline { feature_cols, spec } => {
                if feature_cols.len() != spec.marginalspecs.len() {
                    return Err(BasisError::DimensionMismatch(format!(
                        "TensorBSpline has {} feature axes but {} marginal scale contracts",
                        feature_cols.len(),
                        spec.marginalspecs.len()
                    )));
                }
                Ok(())
            }
        }
    }
}

fn validate_input_scale_vector(
    scales: &[f64],
    expected: usize,
    family: BasisScaleFamily,
) -> Result<(), BasisError> {
    if scales.len() != expected {
        return Err(BasisError::DimensionMismatch(format!(
            "basis {family:?} has {} stored input scales for {expected} coordinate axes",
            scales.len()
        )));
    }
    if let Some((axis, value)) = scales
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || *value <= 0.0)
    {
        return Err(BasisError::InvalidInput(format!(
            "basis {family:?} input scale at axis {axis} must be positive and finite, got {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{
        BSplineBoundaryConditions, BasisWorkspace, ConstantCurvatureIdentifiability,
        DuchonNullspaceOrder, DuchonOperatorPenaltySpec, MaternIdentifiability, MaternNu,
        MeasureJetIdentifiability, SphereWahbaKernel, SphericalSplineIdentifiability,
        build_constant_curvature_basis, build_spherical_spline_basis,
        constant_curvature_kernel_kappa_jets,
    };
    use ndarray::{Array1, Array2, array};
    use std::collections::HashSet;

    fn open_marginal(scale: f64) -> BSplineBasisSpec {
        BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Provided(Array1::from(vec![
                0.0,
                0.0,
                0.0,
                0.0,
                0.25 * scale,
                0.6 * scale,
                scale,
                scale,
                scale,
                scale,
            ])),
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
            boundary: OneDimensionalBoundary::Open,
            boundary_conditions: BSplineBoundaryConditions::default(),
        }
    }

    fn basis(feature_col: usize, scale: f64) -> SmoothBasisSpec {
        SmoothBasisSpec::BSpline1D {
            feature_col,
            spec: open_marginal(scale),
        }
    }

    fn factor_spec(flavour: FactorSmoothFlavour) -> SmoothBasisSpec {
        SmoothBasisSpec::FactorSmooth {
            spec: FactorSmoothSpec {
                continuous_cols: vec![0],
                group_col: 1,
                marginal: open_marginal(1.0),
                flavour,
                group_frozen_levels: Some(vec![0.0_f64.to_bits(), 1.0_f64.to_bits()]),
                frozen_global_orthogonality: None,
            },
        }
    }

    fn scale_contract_zoo() -> Vec<SmoothBasisSpec> {
        let by_level_bits = 1.0_f64.to_bits();
        let mut sphere_harmonic = SphericalSplineBasisSpec::default();
        sphere_harmonic.method = SphereMethod::Harmonic;
        sphere_harmonic.max_degree = Some(3);
        vec![
            SmoothBasisSpec::ByVariable {
                inner: Box::new(basis(0, 1.0)),
                by_col: 1,
                kind: BySmoothKind::Numeric,
                by: ByVariableSpec::Numeric,
            },
            SmoothBasisSpec::ByVariable {
                inner: Box::new(basis(0, 1.0)),
                by_col: 1,
                kind: BySmoothKind::Level {
                    level_bits: by_level_bits,
                },
                by: ByVariableSpec::Level {
                    value_bits: by_level_bits,
                    label: "one".to_string(),
                },
            },
            SmoothBasisSpec::FactorSumToZero {
                inner: Box::new(basis(0, 1.0)),
                by_col: 1,
                levels: vec![0.0_f64.to_bits(), by_level_bits],
                frozen_global_orthogonality: None,
            },
            basis(0, 1.0),
            SmoothBasisSpec::BSpline1D {
                feature_col: 0,
                spec: BSplineBasisSpec {
                    knotspec: BSplineKnotSpec::PeriodicUniform {
                        data_range: (0.0, 1.0),
                        num_basis: 8,
                    },
                    boundary: OneDimensionalBoundary::Cyclic {
                        start: 0.0,
                        end: 1.0,
                    },
                    ..open_marginal(1.0)
                },
            },
            SmoothBasisSpec::BSpline1D {
                feature_col: 0,
                spec: BSplineBasisSpec {
                    knotspec: BSplineKnotSpec::NaturalCubicRegression {
                        knots: array![0.0, 0.2, 0.5, 0.8, 1.0],
                    },
                    ..open_marginal(1.0)
                },
            },
            SmoothBasisSpec::BySmooth {
                smooth: Box::new(basis(0, 1.0)),
                by_kind: ByVarKind::Numeric { feature_col: 1 },
            },
            SmoothBasisSpec::BySmooth {
                smooth: Box::new(basis(0, 1.0)),
                by_kind: ByVarKind::Factor {
                    feature_col: 1,
                    ordered: false,
                    frozen_levels: Some(vec![0.0_f64.to_bits(), by_level_bits]),
                },
            },
            factor_spec(FactorSmoothFlavour::Fs {
                m_null_penalty_orders: vec![1],
            }),
            factor_spec(FactorSmoothFlavour::Sz),
            factor_spec(FactorSmoothFlavour::Re),
            SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0, 1],
                spec: ThinPlateBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                    periodic: None,
                    length_scale: 0.7.into(),
                    double_penalty: false,
                    identifiability: SpatialIdentifiability::None,
                    radial_reparam: None,
                },
                input_scales: None,
            },
            SmoothBasisSpec::Sphere {
                feature_cols: vec![0, 1],
                spec: SphericalSplineBasisSpec::default(),
            },
            SmoothBasisSpec::Sphere {
                feature_cols: vec![0, 1],
                spec: sphere_harmonic,
            },
            SmoothBasisSpec::ConstantCurvature {
                feature_cols: vec![0, 1],
                spec: ConstantCurvatureBasisSpec::default(),
            },
            SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                    periodic: None,
                    length_scale: 0.7.into(),
                    nu: MaternNu::ThreeHalves,
                    include_intercept: false,
                    double_penalty: false,
                    identifiability: MaternIdentifiability::None,
                    aniso_log_scales: None,
                },
                input_scales: None,
            },
            SmoothBasisSpec::MeasureJet {
                feature_cols: vec![0, 1],
                spec: MeasureJetBasisSpec::default(),
                input_scales: None,
            },
            SmoothBasisSpec::Duchon {
                feature_cols: vec![0, 1],
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                    periodic: None,
                    length_scale: None,
                    power: 0.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::None,
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                    radial_reparam: None,
                },
                input_scales: None,
            },
            SmoothBasisSpec::Duchon {
                feature_cols: vec![0, 1],
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
                    periodic: None,
                    length_scale: Some(0.7),
                    power: 1.0,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    identifiability: SpatialIdentifiability::None,
                    aniso_log_scales: None,
                    operator_penalties: DuchonOperatorPenaltySpec::default(),
                    boundary: OneDimensionalBoundary::Open,
                    radial_reparam: None,
                },
                input_scales: None,
            },
            SmoothBasisSpec::Pca {
                feature_cols: vec![0, 1],
                basis_matrix: array![[1.0], [0.0]],
                centered: true,
                smooth_penalty: 1.0,
                center_mean: Some(array![0.0, 0.0]),
                pca_basis_path: None,
                chunk_size: 32,
            },
            SmoothBasisSpec::TensorBSpline {
                feature_cols: vec![0, 1],
                spec: TensorBSplineSpec {
                    marginalspecs: vec![open_marginal(1.0), open_marginal(1.0)],
                    periods: vec![None, None],
                    double_penalty: false,
                    identifiability: TensorBSplineIdentifiability::None,
                    penalty_decomposition: TensorBSplinePenaltyDecomposition::MarginalKroneckerSum,
                },
            },
        ]
    }

    fn zoo_basis(family: BasisScaleFamily) -> SmoothBasisSpec {
        scale_contract_zoo()
            .into_iter()
            .find(|basis| basis.scale_contract().family == family)
            .unwrap_or_else(|| panic!("scale-contract zoo is missing {family:?}"))
    }

    #[test]
    fn scale_contract_registry_is_exhaustive_unique_and_typed_2315() {
        let zoo = scale_contract_zoo();
        let observed: HashSet<_> = zoo
            .iter()
            .map(|basis| basis.scale_contract().family)
            .collect();
        let expected: HashSet<_> = BasisScaleFamily::ALL.into_iter().collect();
        assert_eq!(zoo.len(), BasisScaleFamily::ALL.len());
        assert_eq!(observed, expected);

        for basis in &zoo {
            let contract = basis.scale_contract();
            match contract.family {
                BasisScaleFamily::ByVariableNumeric
                | BasisScaleFamily::ByVariableFactor
                | BasisScaleFamily::FactorSumToZero
                | BasisScaleFamily::BySmoothNumeric
                | BasisScaleFamily::BySmoothFactor
                | BasisScaleFamily::FactorSmoothFs
                | BasisScaleFamily::FactorSmoothSz
                | BasisScaleFamily::FactorSmoothRe => {
                    assert_eq!(contract.children.len(), 1, "{:?}", contract.family);
                }
                BasisScaleFamily::TensorBSpline => {
                    assert_eq!(contract.children.len(), 2);
                }
                BasisScaleFamily::OpenBSpline
                | BasisScaleFamily::CyclicBSpline
                | BasisScaleFamily::NaturalCubic
                | BasisScaleFamily::ThinPlate
                | BasisScaleFamily::SphereWahba
                | BasisScaleFamily::SphereHarmonic
                | BasisScaleFamily::ConstantCurvature
                | BasisScaleFamily::Matern
                | BasisScaleFamily::MeasureJet
                | BasisScaleFamily::PureDuchon
                | BasisScaleFamily::HybridDuchon
                | BasisScaleFamily::Pca => assert!(contract.children.is_empty()),
            }
            let unique_parameters: HashSet<_> = contract
                .dimensionful_parameters
                .iter()
                .map(|parameter| parameter.parameter)
                .collect();
            assert_eq!(
                unique_parameters.len(),
                contract.dimensionful_parameters.len(),
                "duplicate dimensionful parameter in {:?}",
                contract.family
            );
        }
    }

    fn assert_matrix_close(actual: &Array2<f64>, expected: &Array2<f64>, tolerance: f64) {
        assert_eq!(actual.dim(), expected.dim());
        for ((row, col), &target) in expected.indexed_iter() {
            let observed = actual[[row, col]];
            assert!(
                (observed - target).abs() <= tolerance * (1.0 + target.abs()),
                "matrix[{row},{col}] differs: observed={observed:.16e}, target={target:.16e}"
            );
        }
    }

    fn assert_build_geometry_close(
        actual: &BasisBuildResult,
        expected: &BasisBuildResult,
        tolerance: f64,
    ) {
        assert_matrix_close(
            &actual.design.to_dense(),
            &expected.design.to_dense(),
            tolerance,
        );
        assert_eq!(
            actual.active_penalties.len(),
            expected.active_penalties.len()
        );
        for (observed, target) in actual
            .active_penalties
            .iter()
            .zip(expected.active_penalties.iter())
        {
            assert_eq!(observed.info.source, target.info.source);
            assert_eq!(observed.info.effective_rank, target.info.effective_rank);
            assert_eq!(observed.nullity, target.nullity);
            assert_matrix_close(&observed.matrix, &target.matrix, tolerance);
        }
    }

    fn assert_local_geometry_close(
        actual: &LocalSmoothTermBuild,
        expected: &LocalSmoothTermBuild,
        tolerance: f64,
    ) {
        assert_matrix_close(
            &actual.design.to_dense(),
            &expected.design.to_dense(),
            tolerance,
        );
        assert_eq!(
            actual.active_penalties.len(),
            expected.active_penalties.len()
        );
        for (observed, target) in actual
            .active_penalties
            .iter()
            .zip(expected.active_penalties.iter())
        {
            assert_eq!(observed.info.source, target.info.source);
            assert_eq!(observed.info.effective_rank, target.info.effective_rank);
            assert_eq!(observed.nullity, target.nullity);
            assert_matrix_close(&observed.matrix, &target.matrix, tolerance);
        }
    }

    fn scaled_cyclic_marginal(scale: f64) -> BSplineBasisSpec {
        BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::PeriodicUniform {
                data_range: (-0.4 * scale, 1.6 * scale),
                num_basis: 9,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
            boundary: OneDimensionalBoundary::Cyclic {
                start: -0.4 * scale,
                end: 1.6 * scale,
            },
            boundary_conditions: BSplineBoundaryConditions::default(),
        }
    }

    fn scaled_natural_marginal(scale: f64) -> BSplineBasisSpec {
        BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::NaturalCubicRegression {
                knots: array![0.0, 0.17, 0.43, 0.71, 1.0].mapv(|value| value * scale),
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
            boundary: OneDimensionalBoundary::Open,
            boundary_conditions: BSplineBoundaryConditions::default(),
        }
    }

    #[test]
    fn declared_spline_builders_obey_design_penalty_and_null_scale_laws_2315() {
        let open_points = array![0.0, 0.05, 0.3, 0.55, 0.72, 0.95, 1.0];
        let cyclic_points = array![-2.4, -0.4, -0.13, 0.2, 1.1, 1.6, 3.6];
        let natural_points = array![-0.35, 0.0, 0.09, 0.43, 0.86, 1.0, 1.28];

        for (family, points, build_spec) in [
            (
                BasisScaleFamily::OpenBSpline,
                open_points,
                open_marginal as fn(f64) -> BSplineBasisSpec,
            ),
            (
                BasisScaleFamily::CyclicBSpline,
                cyclic_points,
                scaled_cyclic_marginal as fn(f64) -> BSplineBasisSpec,
            ),
            (
                BasisScaleFamily::NaturalCubic,
                natural_points,
                scaled_natural_marginal as fn(f64) -> BSplineBasisSpec,
            ),
        ] {
            let reference_spec = build_spec(1.0);
            assert_eq!(bspline_contract(&reference_spec).family, family);
            let reference = build_bspline_basis_1d(points.view(), &reference_spec)
                .expect("reference scalar basis");
            for factor in [1e-9_f64, 1.0, 1e9] {
                let actual = build_bspline_basis_1d(
                    points.mapv(|value| value * factor).view(),
                    &build_spec(factor),
                )
                .expect("rescaled scalar basis");
                assert_build_geometry_close(&actual, &reference, 8e-10);
                for (observed, target) in actual
                    .active_penalties
                    .iter()
                    .zip(reference.active_penalties.iter())
                {
                    let rescaled = observed.info.normalization_scale * factor.powi(3);
                    assert!(
                        (rescaled - target.info.normalization_scale).abs()
                            <= 2e-8 * (1.0 + target.info.normalization_scale.abs()),
                        "{family:?} raw penalty normalizer violated a^-3 at factor {factor}"
                    );
                }
            }
        }

        let tensor_data = array![
            [0.00, 0.13],
            [0.08, 0.91],
            [0.22, 0.37],
            [0.41, 1.00],
            [0.58, 0.02],
            [0.73, 0.66],
            [0.89, 0.48],
            [1.00, 0.00]
        ];
        let build_tensor = |x_scale: f64, y_scale: f64| {
            let mut scaled = tensor_data.clone();
            scaled.column_mut(0).mapv_inplace(|value| value * x_scale);
            scaled.column_mut(1).mapv_inplace(|value| value * y_scale);
            let spec = TensorBSplineSpec {
                marginalspecs: vec![open_marginal(x_scale), open_marginal(y_scale)],
                periods: vec![None, None],
                double_penalty: false,
                identifiability: TensorBSplineIdentifiability::None,
                penalty_decomposition: TensorBSplinePenaltyDecomposition::MarginalKroneckerSum,
            };
            build_tensor_bspline_basis(scaled.view(), &[0, 1], &spec)
                .expect("rescaled tensor basis")
        };
        let reference = build_tensor(1.0, 1.0);
        for x_scale in [1e-9_f64, 1.0, 1e9] {
            for y_scale in [1e-9_f64, 1.0, 1e9] {
                let actual = build_tensor(x_scale, y_scale);
                assert_build_geometry_close(&actual, &reference, 1e-9);
            }
        }
    }

    fn wrapper_basis(family: BasisScaleFamily, abscissa_scale: f64) -> SmoothBasisSpec {
        let inner = || Box::new(basis(0, abscissa_scale));
        let levels = vec![0.0_f64.to_bits(), 1.0_f64.to_bits(), 2.0_f64.to_bits()];
        let factor_smooth = |flavour| SmoothBasisSpec::FactorSmooth {
            spec: FactorSmoothSpec {
                continuous_cols: vec![0],
                group_col: 1,
                marginal: open_marginal(abscissa_scale),
                flavour,
                group_frozen_levels: Some(levels.clone()),
                frozen_global_orthogonality: None,
            },
        };
        match family {
            BasisScaleFamily::ByVariableNumeric => SmoothBasisSpec::ByVariable {
                inner: inner(),
                by_col: 1,
                kind: BySmoothKind::Numeric,
                by: ByVariableSpec::Numeric,
            },
            BasisScaleFamily::ByVariableFactor => SmoothBasisSpec::ByVariable {
                inner: inner(),
                by_col: 1,
                kind: BySmoothKind::Level {
                    level_bits: 1.0_f64.to_bits(),
                },
                by: ByVariableSpec::Level {
                    value_bits: 1.0_f64.to_bits(),
                    label: "one".to_string(),
                },
            },
            BasisScaleFamily::FactorSumToZero => SmoothBasisSpec::FactorSumToZero {
                inner: inner(),
                by_col: 1,
                levels,
                frozen_global_orthogonality: None,
            },
            BasisScaleFamily::BySmoothNumeric => SmoothBasisSpec::BySmooth {
                smooth: inner(),
                by_kind: ByVarKind::Numeric { feature_col: 1 },
            },
            BasisScaleFamily::BySmoothFactor => SmoothBasisSpec::BySmooth {
                smooth: inner(),
                by_kind: ByVarKind::Factor {
                    feature_col: 1,
                    ordered: false,
                    frozen_levels: Some(levels),
                },
            },
            BasisScaleFamily::FactorSmoothFs => factor_smooth(FactorSmoothFlavour::Fs {
                m_null_penalty_orders: vec![1],
            }),
            BasisScaleFamily::FactorSmoothSz => factor_smooth(FactorSmoothFlavour::Sz),
            BasisScaleFamily::FactorSmoothRe => factor_smooth(FactorSmoothFlavour::Re),
            BasisScaleFamily::OpenBSpline
            | BasisScaleFamily::CyclicBSpline
            | BasisScaleFamily::NaturalCubic
            | BasisScaleFamily::ThinPlate
            | BasisScaleFamily::SphereWahba
            | BasisScaleFamily::SphereHarmonic
            | BasisScaleFamily::ConstantCurvature
            | BasisScaleFamily::Matern
            | BasisScaleFamily::MeasureJet
            | BasisScaleFamily::PureDuchon
            | BasisScaleFamily::HybridDuchon
            | BasisScaleFamily::Pca
            | BasisScaleFamily::TensorBSpline => {
                panic!("{family:?} is not a wrapper fixture")
            }
        }
    }

    fn build_local(data: &Array2<f64>, basis: SmoothBasisSpec) -> LocalSmoothTermBuild {
        build_single_local_smooth_term(
            data.view(),
            &SmoothTermSpec {
                name: "scale-contract-wrapper".to_string(),
                basis,
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            },
            &mut BasisWorkspace::new(),
        )
        .expect("wrapper scale fixture must build")
    }

    #[test]
    fn every_wrapper_preserves_its_declared_inner_abscissa_pullback_2315() {
        let data = array![
            [0.00, 0.0],
            [0.08, 1.0],
            [0.17, 2.0],
            [0.26, 0.0],
            [0.35, 1.0],
            [0.44, 2.0],
            [0.56, 0.0],
            [0.65, 1.0],
            [0.74, 2.0],
            [0.83, 0.0],
            [0.92, 1.0],
            [1.00, 2.0]
        ];
        let invariant_families = [
            BasisScaleFamily::ByVariableNumeric,
            BasisScaleFamily::ByVariableFactor,
            BasisScaleFamily::FactorSumToZero,
            BasisScaleFamily::BySmoothNumeric,
            BasisScaleFamily::BySmoothFactor,
            BasisScaleFamily::FactorSmoothFs,
            BasisScaleFamily::FactorSmoothSz,
        ];
        for family in invariant_families {
            let reference = build_local(&data, wrapper_basis(family, 1.0));
            for factor in [1e-9_f64, 1.0, 1e9] {
                let mut scaled = data.clone();
                scaled.column_mut(0).mapv_inplace(|value| factor * value);
                let actual = build_local(&scaled, wrapper_basis(family, factor));
                assert_local_geometry_close(&actual, &reference, 2e-8);
            }
        }

        // `bs="re"` is a random intercept+slope, so each level carries one
        // invariant intercept column and one degree-one slope column.
        let family = BasisScaleFamily::FactorSmoothRe;
        let reference = build_local(&data, wrapper_basis(family, 1.0));
        for factor in [1e-9_f64, 1.0, 1e9] {
            let mut scaled = data.clone();
            scaled.column_mut(0).mapv_inplace(|value| factor * value);
            let actual = build_local(&scaled, wrapper_basis(family, factor));
            let mut pulled_back = actual.design.to_dense();
            for slope_col in (1..pulled_back.ncols()).step_by(2) {
                pulled_back
                    .column_mut(slope_col)
                    .mapv_inplace(|value| value / factor);
            }
            assert_matrix_close(&pulled_back, &reference.design.to_dense(), 2e-9);
            assert_eq!(
                actual.active_penalties.len(),
                reference.active_penalties.len()
            );
            for (observed, target) in actual
                .active_penalties
                .iter()
                .zip(reference.active_penalties.iter())
            {
                assert_matrix_close(&observed.matrix, &target.matrix, 2e-10);
                assert_eq!(observed.nullity, target.nullity);
            }
            assert_eq!(
                joint_unpenalized_dim(actual.dim, &actual.active_penalties),
                0,
                "the combined random-intercept/slope penalty must be full rank"
            );
        }
    }

    #[test]
    fn numeric_modulator_has_exact_degree_one_design_and_invariant_penalty_2315() {
        let data = array![
            [0.00, 0.4],
            [0.08, 0.7],
            [0.17, 1.1],
            [0.26, 0.8],
            [0.35, 1.4],
            [0.44, 0.6],
            [0.56, 1.2],
            [0.65, 0.9],
            [0.74, 1.5],
            [0.83, 0.5],
            [0.92, 1.3],
            [1.00, 1.0]
        ];
        for family in [
            BasisScaleFamily::ByVariableNumeric,
            BasisScaleFamily::BySmoothNumeric,
        ] {
            let reference = build_local(&data, wrapper_basis(family, 1.0));
            for factor in [1e-9_f64, 1.0, 1e9] {
                let mut scaled = data.clone();
                scaled.column_mut(1).mapv_inplace(|value| factor * value);
                let actual = build_local(&scaled, wrapper_basis(family, 1.0));
                let pulled_back = actual.design.to_dense().mapv(|value| value / factor);
                assert_matrix_close(&pulled_back, &reference.design.to_dense(), 2e-10);
                assert_eq!(
                    actual.active_penalties.len(),
                    reference.active_penalties.len()
                );
                for (observed, target) in actual
                    .active_penalties
                    .iter()
                    .zip(reference.active_penalties.iter())
                {
                    assert_matrix_close(&observed.matrix, &target.matrix, 2e-10);
                }
            }
        }
    }

    fn spherical_spec(method: SphereMethod, radians: bool) -> SphericalSplineBasisSpec {
        SphericalSplineBasisSpec {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
            penalty_order: 2,
            double_penalty: false,
            radians,
            method,
            max_degree: Some(3),
            wahba_kernel: SphereWahbaKernel::Sobolev,
            identifiability: SphericalSplineIdentifiability::CenterSumToZero,
        }
    }

    #[test]
    fn sphere_constant_curvature_and_pca_obey_their_non_euclidean_gauges_2315() {
        let degrees = array![
            [-62.0, -150.0],
            [-41.0, -77.0],
            [-18.0, -12.0],
            [4.0, 39.0],
            [23.0, 101.0],
            [47.0, 166.0],
            [66.0, -115.0],
            [11.0, -171.0]
        ];
        let radians = degrees.mapv(f64::to_radians);
        for method in [SphereMethod::Wahba, SphereMethod::Harmonic] {
            let in_degrees =
                build_spherical_spline_basis(degrees.view(), &spherical_spec(method, false))
                    .expect("degree-encoded sphere basis");
            let in_radians =
                build_spherical_spline_basis(radians.view(), &spherical_spec(method, true))
                    .expect("radian-encoded sphere basis");
            assert_build_geometry_close(&in_radians, &in_degrees, 2e-9);
        }

        let chart_data = array![
            [-0.42, -0.18],
            [-0.31, 0.22],
            [-0.08, -0.34],
            [0.13, 0.29],
            [0.27, -0.11],
            [0.38, 0.17]
        ];
        let centers = array![[-0.36, -0.04], [-0.12, 0.25], [0.16, -0.21], [0.34, 0.13]];
        let kappa = -0.7_f64;
        let length_scale = 0.55_f64;
        let reference_spec = ConstantCurvatureBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            kappa,
            kappa_fixed: true,
            length_scale,
            double_penalty: false,
            identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
        };
        let reference = build_constant_curvature_basis(chart_data.view(), &reference_spec)
            .expect("reference constant-curvature basis");
        let (_, dk_reference, dkk_reference) = constant_curvature_kernel_kappa_jets(
            chart_data.view(),
            centers.view(),
            kappa,
            length_scale,
        )
        .expect("reference curvature jets");
        for factor in [1e-9_f64, 1.0, 1e9] {
            let scaled_data = chart_data.mapv(|value| value * factor);
            let scaled_centers = centers.mapv(|value| value * factor);
            let scaled_kappa = kappa / factor.powi(2);
            let scaled_length = length_scale * factor;
            let actual = build_constant_curvature_basis(
                scaled_data.view(),
                &ConstantCurvatureBasisSpec {
                    center_strategy: CenterStrategy::UserProvided(scaled_centers.clone()),
                    kappa: scaled_kappa,
                    kappa_fixed: true,
                    length_scale: scaled_length,
                    double_penalty: false,
                    identifiability: ConstantCurvatureIdentifiability::CenterSumToZero,
                },
            )
            .expect("rescaled constant-curvature basis");
            assert_build_geometry_close(&actual, &reference, 2e-8);
            let (_, dk, dkk) = constant_curvature_kernel_kappa_jets(
                scaled_data.view(),
                scaled_centers.view(),
                scaled_kappa,
                scaled_length,
            )
            .expect("rescaled curvature jets");
            assert_matrix_close(
                &dk.mapv(|value| value / factor.powi(2)),
                &dk_reference,
                2e-8,
            );
            assert_matrix_close(
                &dkk.mapv(|value| value / factor.powi(4)),
                &dkk_reference,
                3e-7,
            );
        }

        let pca_data = array![
            [-1.0, 0.3],
            [-0.4, 1.1],
            [0.2, -0.7],
            [0.8, 0.5],
            [1.3, -0.2],
            [1.7, 0.9]
        ];
        let center_mean = array![0.35, 0.15];
        let loadings = array![[0.8, -0.3], [0.6, 0.9]];
        let reference = build_pca_smooth_basis(
            pca_data.view(),
            &[0, 1],
            &loadings,
            true,
            1.7,
            Some(&center_mean),
            None,
            32,
        )
        .expect("reference PCA score gauge");
        for factor in [1e-9_f64, 1.0, 1e9] {
            let actual = build_pca_smooth_basis(
                pca_data.mapv(|value| factor * value).view(),
                &[0, 1],
                &loadings.mapv(|value| value / factor),
                true,
                1.7,
                Some(&center_mean.mapv(|value| factor * value)),
                None,
                32,
            )
            .expect("rescaled PCA score gauge");
            assert_build_geometry_close(&actual, &reference, 2e-10);
        }
    }

    #[test]
    fn euclidean_registry_frames_are_similarity_invariant_for_every_declared_family_2315() {
        let coordinates = array![
            [-0.9, 0.2],
            [-0.4, 0.8],
            [0.1, -0.6],
            [0.7, 0.4],
            [1.2, -0.1]
        ];
        let cases = [
            (BasisScaleFamily::ThinPlate, Some(0.7)),
            (BasisScaleFamily::Matern, Some(0.7)),
            (BasisScaleFamily::MeasureJet, Some(0.7)),
            (BasisScaleFamily::PureDuchon, None),
            (BasisScaleFamily::HybridDuchon, Some(0.7)),
        ];
        for (family, length_scale) in cases {
            let basis = zoo_basis(family);
            let contract = basis.scale_contract();
            assert_eq!(contract.family, family);
            let reference = contract
                .normalize_euclidean_frame(coordinates.clone(), None, length_scale)
                .expect("reference scale frame");
            for factor in [1e-9_f64, 1.0, 1e9] {
                let actual = contract
                    .normalize_euclidean_frame(
                        coordinates.mapv(|value| factor * value),
                        None,
                        length_scale.map(|ell| factor * ell),
                    )
                    .expect("rescaled frame");
                assert_matrix_close(&actual.coordinates, &reference.coordinates, 3e-12);
                match (actual.length_scale, reference.length_scale) {
                    (Some(observed), Some(target)) => assert!(
                        (observed - target).abs() <= 3e-12 * (1.0 + target.abs()),
                        "{family:?} effective range changed at factor {factor}: {observed} vs {target}"
                    ),
                    (None, None) => {}
                    pair => panic!("{family:?} changed optional range shape: {pair:?}"),
                }
            }
        }
    }

    fn euclidean_basis(family: BasisScaleFamily, factor: f64) -> SmoothBasisSpec {
        let centers = CenterStrategy::FarthestPoint { num_centers: 8 };
        match family {
            BasisScaleFamily::ThinPlate => SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0, 1],
                spec: ThinPlateBasisSpec {
                    center_strategy: centers,
                    periodic: None,
                    length_scale: (0.55 * factor).into(),
                    double_penalty: false,
                    identifiability: SpatialIdentifiability::None,
                    radial_reparam: None,
                },
                input_scales: None,
            },
            BasisScaleFamily::Matern => SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    center_strategy: centers,
                    periodic: None,
                    length_scale: (0.55 * factor).into(),
                    nu: MaternNu::ThreeHalves,
                    include_intercept: false,
                    double_penalty: false,
                    identifiability: MaternIdentifiability::None,
                    aniso_log_scales: None,
                },
                input_scales: None,
            },
            BasisScaleFamily::MeasureJet => SmoothBasisSpec::MeasureJet {
                feature_cols: vec![0, 1],
                spec: MeasureJetBasisSpec {
                    center_strategy: centers,
                    order_s: 1.5,
                    alpha: 1.0,
                    tau0: 1e-3,
                    num_scales: 3,
                    length_scale: 0.55 * factor,
                    double_penalty: false,
                    learn_length_scale: false,
                    multiscale: false,
                    identifiability: MeasureJetIdentifiability::CenterSumToZero,
                    frozen_quadrature: None,
                },
                input_scales: None,
            },
            BasisScaleFamily::PureDuchon | BasisScaleFamily::HybridDuchon => {
                SmoothBasisSpec::Duchon {
                    feature_cols: vec![0, 1],
                    spec: DuchonBasisSpec {
                        center_strategy: centers,
                        periodic: None,
                        length_scale: (family == BasisScaleFamily::HybridDuchon)
                            .then_some(0.55 * factor),
                        power: if family == BasisScaleFamily::HybridDuchon {
                            1.0
                        } else {
                            0.0
                        },
                        nullspace_order: DuchonNullspaceOrder::Linear,
                        identifiability: SpatialIdentifiability::None,
                        aniso_log_scales: None,
                        operator_penalties: DuchonOperatorPenaltySpec::all_disabled(),
                        boundary: OneDimensionalBoundary::Open,
                        radial_reparam: None,
                    },
                    input_scales: None,
                }
            }
            BasisScaleFamily::ByVariableNumeric
            | BasisScaleFamily::ByVariableFactor
            | BasisScaleFamily::FactorSumToZero
            | BasisScaleFamily::OpenBSpline
            | BasisScaleFamily::CyclicBSpline
            | BasisScaleFamily::NaturalCubic
            | BasisScaleFamily::BySmoothNumeric
            | BasisScaleFamily::BySmoothFactor
            | BasisScaleFamily::FactorSmoothFs
            | BasisScaleFamily::FactorSmoothSz
            | BasisScaleFamily::FactorSmoothRe
            | BasisScaleFamily::SphereWahba
            | BasisScaleFamily::SphereHarmonic
            | BasisScaleFamily::ConstantCurvature
            | BasisScaleFamily::Pca
            | BasisScaleFamily::TensorBSpline => {
                panic!("{family:?} is not a Euclidean spatial fixture")
            }
        }
    }

    #[test]
    fn every_euclidean_spatial_builder_obeys_its_registry_pullback_2315() {
        let data = array![
            [-0.95, -0.22],
            [-0.82, 0.31],
            [-0.63, -0.47],
            [-0.48, 0.62],
            [-0.27, -0.08],
            [-0.11, 0.41],
            [0.06, -0.55],
            [0.21, 0.17],
            [0.38, 0.73],
            [0.54, -0.31],
            [0.69, 0.49],
            [0.83, -0.66],
            [0.97, 0.04],
            [1.08, 0.58],
            [1.19, -0.39],
            [1.31, 0.26]
        ];
        for family in [
            BasisScaleFamily::ThinPlate,
            BasisScaleFamily::Matern,
            BasisScaleFamily::MeasureJet,
            BasisScaleFamily::PureDuchon,
            BasisScaleFamily::HybridDuchon,
        ] {
            let reference = build_local(&data, euclidean_basis(family, 1.0));
            for factor in [1e-9_f64, 1.0, 1e9] {
                let scaled = data.mapv(|value| factor * value);
                let actual = build_local(&scaled, euclidean_basis(family, factor));
                assert_local_geometry_close(&actual, &reference, 3e-7);
            }
        }
    }

    #[test]
    fn malformed_frozen_scale_vectors_are_rejected_before_indexing_2315() {
        let mut basis = zoo_basis(BasisScaleFamily::Matern);
        for invalid in [vec![1.0], vec![1.0, 0.0], vec![1.0, f64::NAN]] {
            match &mut basis {
                SmoothBasisSpec::Matern { input_scales, .. } => {
                    *input_scales = Some(invalid);
                }
                other => panic!("family lookup returned {other:?} instead of Matérn"),
            }
            assert!(basis.validate_scale_configuration().is_err());
        }
    }
}

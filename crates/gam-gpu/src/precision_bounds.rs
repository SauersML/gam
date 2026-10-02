//! Rigorous forward-error bands for the reduced-precision device kernels.
//!
//! Apple GPUs (Metal) have no float64. gam's numerics and every certificate
//! are float64, so a Metal result is admitted only in one of two roles:
//!
//! * **screening / ordering**: the value decides which candidates the float64
//!   CPU path looks at next, and every decision the screen makes is sound
//!   against the band derived here (a candidate is discarded only when its
//!   band excludes it);
//! * **bounded**: the value carries a band derived here that covers the whole
//!   device computation, and the consumer checks that band against the
//!   tolerance its certificate needs before using the value.
//!
//! Nothing in this module trusts a device result. It derives, from the shape
//! of a computation and float64 norms of its operands computed on the host, a
//! bound on how far the device's answer can be from the exact answer. The
//! bound is a pure function of those host quantities, so the device can never
//! argue for its own accuracy.
//!
//! # Arithmetic model
//!
//! Every bound assumes the device performs each f32 `+`, `*`, `fma` and
//! f64→f32 conversion as an IEEE-754 binary32 operation that returns one of the
//! two floats adjacent to the exact result (faithful rounding), and may flush
//! results and operands below the normal range (`2^-126`) to zero. Metal's safe
//! math mode promises correctly rounded `+`/`*`/`fma`, which is stronger; the
//! simdgroup matrix unit is not documented at all, so the GEMM bands use the
//! faithful unit `2^-23` instead of the round-to-nearest unit `2^-24`. The
//! Metal backend checks this model on its own device at probe time
//! (`metal::self_test`) and refuses to come up when the device violates it.
//!
//! Relative error per operation is `unit`; flushed underflow contributes an
//! absolute error of at most `2^-126` per operation, collected into the
//! `absolute` terms below with a generous margin.
//!
//! # Double-float (df64)
//!
//! A df64 value is an unevaluated sum `hi + lo` of two f32 with
//! `|lo| ≤ ulp(hi)/2`. The kernels use the accurate double-word algorithms of
//! Joldes, Muller and Popescu, "Tight and rigorous error bounds for basic
//! building blocks of double-word arithmetic" (ACM TOMS 2017), with the
//! corrections of Muller and Rideau (ACM TOMS 2022):
//!
//! * `DWPlusDW` (accurate, Algorithm 6): relative error `≤ 3u²/(1−4u)`;
//! * `DWTimesDW3` (Algorithm 12, with fma): relative error `≤ 5u²`;
//!
//! with `u = 2^-24`, so one df64 multiply-add has relative error below
//! `5u²(1+2^-20) < 2^-45.67`. [`DF64_OP_UNIT`] is `2^-45`, a factor 1.6 above
//! that. The f64→df64 split of an operand (`hi = f32(x)`, `lo = f32(x − hi)`)
//! has relative error `≤ u² = 2^-48`. A df64 value therefore carries about 45
//! to 48 significant bits against float64's 53, and only f32's exponent range:
//! it is a genuine accuracy loss of at least `2^-45 / 2^-53 = 256×` per
//! operation, and is admitted only where a band derived here is inside the
//! consumer's tolerance.

use ndarray::{ArrayView2, Axis};

/// Round-to-nearest unit roundoff of binary32, `2^-24`.
pub const F32_UNIT_ROUNDOFF: f64 = 5.960_464_477_539_063e-8;

/// Faithful-rounding unit of binary32, `2^-23`: the relative error of any
/// operation that returns one of the two floats adjacent to the exact result.
/// The device GEMM bands use it because the simdgroup matrix unit's rounding
/// is undocumented.
pub const F32_FAITHFUL_UNIT: f64 = 1.192_092_895_507_812_5e-7;

/// Relative error of one df64 multiply-add (`DWTimesDW3` then accurate
/// `DWPlusDW`), `2^-45`; the derived value is `5u²(1+2^-20) < 2^-45.67`.
pub const DF64_OP_UNIT: f64 = 2.842_170_943_040_400_7e-14;

/// Relative error of splitting a float64 into a df64 pair, `u² = 2^-48`.
pub const DF64_SPLIT_UNIT: f64 = 3.552_713_678_800_501e-15;

/// The smallest positive normal binary32, `2^-126`.
pub const F32_MIN_NORMAL: f64 = 1.175_494_350_822_287_5e-38;

/// Absolute error one f32 multiply-add may lose to flushed underflow, with
/// margin: `2^-124`, four times the `2^-126` a single flush can cost, covering
/// the product and the sum of a multiply-add.
pub const F32_UNDERFLOW_PER_OP: f64 = 4.0 * F32_MIN_NORMAL;

/// Absolute error one df64 multiply-add may lose to flushed underflow, with
/// margin: `2^-120`. A df64 multiply-add is about twenty f32 operations, each of
/// which can lose at most `2^-126` to a flush.
pub const DF64_UNDERFLOW_PER_OP: f64 = 64.0 * F32_MIN_NORMAL;

/// The largest operand magnitude the device kernels admit, `2^60`. With every
/// operand below it, no product exceeds `2^120` and no sum of the admitted
/// sizes reaches the binary32 overflow threshold `2^128`.
pub const F32_ADMISSIBLE_MAGNITUDE: f64 = 1.152_921_504_606_847e18;

/// The largest bound on an accumulated magnitude (`‖a‖₂·‖b‖₂` for a dot
/// product) the device kernels admit, `2^120`. Partial sums of a dot product
/// never exceed `(1+γ)·Σ|aₗbₗ| ≤ (1+γ)·‖a‖₂‖b‖₂`, so they stay 8 binades below
/// binary32 overflow.
pub const F32_ADMISSIBLE_ACCUMULATION: f64 = 1.329_227_995_784_916e36;

/// Relative inflation applied to every host-side float64 evaluation of a
/// bound, `2^-40`. The bound formulas take a few dozen float64 operations, each
/// with relative error `2^-53`; `2^-40` covers them with a factor of 100
/// spare, so a bound computed on the host is never smaller than the exact
/// value of its formula.
const HOST_EVALUATION_INFLATION: f64 = 1.0 + 9.094_947_017_729_282e-13;

/// The classical `γₙ = n·unit / (1 − n·unit)` of Higham, *Accuracy and
/// Stability of Numerical Algorithms*, Lemma 3.1: `∏(1+δᵢ)^{±1} = 1 + θₙ` with
/// `|θₙ| ≤ γₙ` for `n` operations of relative error at most `unit`. `None`
/// when `n·unit ≥ 1/2`, where the bound is no longer informative and the
/// device route is refused instead.
#[must_use]
pub fn gamma(n: usize, unit: f64) -> Option<f64> {
    let nu = (n as f64) * unit;
    if !(nu.is_finite() && nu < 0.5) {
        return None;
    }
    Some(nu / (1.0 - nu) * HOST_EVALUATION_INFLATION)
}

/// Round-to-nearest unit roundoff of binary64, `2^-53`.
pub const F64_UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;

/// The arithmetic a kernel carries values in: one of the two device
/// arithmetics, or the host's float64 that every CPU fallback uses, so a
/// caller reads one kind of band whichever executor ran.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceArithmetic {
    /// Binary32 values with f32 accumulation (Metal).
    F32,
    /// Double-float (`hi + lo`) values with df64 accumulation (Metal).
    Df64,
    /// Binary64 on the host CPU, any summation order.
    HostF64,
}

impl DeviceArithmetic {
    /// Relative error of one multiply-add in this arithmetic.
    #[must_use]
    pub const fn op_unit(self) -> f64 {
        match self {
            Self::F32 => F32_FAITHFUL_UNIT,
            Self::Df64 => DF64_OP_UNIT,
            Self::HostF64 => F64_UNIT_ROUNDOFF,
        }
    }

    /// Relative error of converting one float64 operand into this arithmetic.
    #[must_use]
    pub const fn split_unit(self) -> f64 {
        match self {
            Self::F32 => F32_FAITHFUL_UNIT,
            Self::Df64 => DF64_SPLIT_UNIT,
            Self::HostF64 => 0.0,
        }
    }

    /// Absolute error one multiply-add may lose to underflow. The host keeps
    /// gradual underflow, so a float64 operation loses at most `2^-1074`;
    /// `2^-1070` is used.
    #[must_use]
    pub const fn underflow_per_op(self) -> f64 {
        match self {
            Self::F32 => F32_UNDERFLOW_PER_OP,
            Self::Df64 => DF64_UNDERFLOW_PER_OP,
            Self::HostF64 => 7.905_050_788_573_489e-323,
        }
    }

    /// The largest operand magnitude this arithmetic admits.
    #[must_use]
    pub const fn admissible_magnitude(self) -> f64 {
        match self {
            Self::F32 | Self::Df64 => F32_ADMISSIBLE_MAGNITUDE,
            Self::HostF64 => 1.0e150,
        }
    }

    /// The largest accumulated magnitude this arithmetic admits.
    #[must_use]
    pub const fn admissible_accumulation(self) -> f64 {
        match self {
            Self::F32 | Self::Df64 => F32_ADMISSIBLE_ACCUMULATION,
            Self::HostF64 => 1.0e300,
        }
    }

    /// Whether this arithmetic runs on the device.
    #[must_use]
    pub const fn is_device(self) -> bool {
        match self {
            Self::F32 | Self::Df64 => true,
            Self::HostF64 => false,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Df64 => "df64",
            Self::HostF64 => "host-f64",
        }
    }
}

/// Why a device computation cannot be given a band.
#[derive(Clone, Debug, PartialEq)]
pub enum BandRefusal {
    /// An operand is NaN or infinite.
    NonFinite { what: &'static str },
    /// An operand exceeds [`F32_ADMISSIBLE_MAGNITUDE`].
    MagnitudeOutOfRange { what: &'static str, magnitude: f64 },
    /// An accumulated magnitude exceeds [`F32_ADMISSIBLE_ACCUMULATION`].
    AccumulationOutOfRange { magnitude: f64 },
    /// The operation count makes `γ` uninformative.
    DepthTooLarge { operations: usize, unit: f64 },
}

impl std::fmt::Display for BandRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite { what } => write!(f, "{what} holds a non-finite value"),
            Self::MagnitudeOutOfRange { what, magnitude } => write!(
                f,
                "{what} holds a value of magnitude {magnitude:e}, above the admission \
                 limit of the arithmetic it would run in"
            ),
            Self::AccumulationOutOfRange { magnitude } => write!(
                f,
                "an accumulated magnitude {magnitude:e} exceeds the admission limit of \
                 the arithmetic it would run in"
            ),
            Self::DepthTooLarge { operations, unit } => write!(
                f,
                "{operations} accumulated operations at unit {unit:e} leave no informative \
                 error band"
            ),
        }
    }
}

/// Upper bounds on the Euclidean norms of the rows of `matrix` (any layout),
/// evaluated in float64 and inflated so each bounds the exact norm. Refuses a
/// non-finite entry or one above the arithmetic's admitted magnitude. Column
/// norms are the row norms of the transposed view.
pub fn row_norms_upper(
    arithmetic: DeviceArithmetic,
    what: &'static str,
    matrix: ArrayView2<'_, f64>,
) -> Result<Vec<f64>, BandRefusal> {
    use ndarray::parallel::prelude::*;
    // Summing `cols` squares in float64 has relative error at most γ_{cols+1}
    // at unit 2^-53 in any order; the square root halves it. Inflate by γ at
    // unit 2^-52.
    let inflation = 1.0
        + gamma(matrix.ncols() + 2, f64::EPSILON).ok_or(BandRefusal::DepthTooLarge {
            operations: matrix.ncols() + 2,
            unit: f64::EPSILON,
        })?;
    let limit = arithmetic.admissible_magnitude();
    let check = |value: f64| -> Result<f64, BandRefusal> {
        if !value.is_finite() {
            return Err(BandRefusal::NonFinite { what });
        }
        let magnitude = value.abs();
        if magnitude > limit {
            return Err(BandRefusal::MagnitudeOutOfRange { what, magnitude });
        }
        Ok(value * value)
    };
    if !matrix.is_standard_layout() && matrix.t().is_standard_layout() {
        // A transposed view of a row-major matrix: accumulate its rows'
        // squares by walking the stored rows contiguously, in parallel blocks.
        let stored = matrix.t();
        let block = 64;
        let partials: Result<Vec<Vec<f64>>, BandRefusal> = stored
            .axis_chunks_iter(Axis(0), block)
            .into_par_iter()
            .map(|chunk| {
                let mut sums = vec![0.0_f64; stored.ncols()];
                for row in chunk.rows() {
                    for (sum, &value) in sums.iter_mut().zip(row) {
                        *sum += check(value)?;
                    }
                }
                Ok(sums)
            })
            .collect();
        let mut sums = vec![0.0_f64; stored.ncols()];
        for partial in partials? {
            for (sum, value) in sums.iter_mut().zip(partial) {
                *sum += value;
            }
        }
        return Ok(sums.into_iter().map(|sum| sum.sqrt() * inflation).collect());
    }
    matrix
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| {
            let mut sum = 0.0_f64;
            for &value in row {
                sum += check(value)?;
            }
            Ok(sum.sqrt() * inflation)
        })
        .collect()
}

/// The forward-error band of one device matrix product `C = A·B` with inner
/// dimension `k`, computed from float64 operands converted on the host.
///
/// For every entry,
///
/// ```text
/// |ĉᵢⱼ − cᵢⱼ| ≤ γ_{k+2}(unit)·Σₗ|aᵢₗ||bₗⱼ| + η·(k + Σₗ(|aᵢₗ| + |bₗⱼ|))
///            ≤ coefficient·‖aᵢ‖₂‖bⱼ‖₂ + absolute·(k + √k·(‖aᵢ‖₂ + ‖bⱼ‖₂))
/// ```
///
/// The first line is Higham's dot-product bound (Theorem 3.5, valid for any
/// summation order) with the two operand conversions counted as two more
/// rounded operations per term; the second is Cauchy–Schwarz. `η` is the
/// per-operation underflow loss of the arithmetic.
#[derive(Clone, Debug, PartialEq)]
pub struct GemmBand {
    pub arithmetic: DeviceArithmetic,
    pub k: usize,
    /// `γ_{k+2}(unit)`.
    pub coefficient: f64,
    /// The per-operation underflow loss `η`.
    pub absolute: f64,
    /// Upper bounds on `‖aᵢ‖₂`, one per row of `A`.
    pub row_norms: Vec<f64>,
    /// Upper bounds on `‖bⱼ‖₂`, one per column of `B`.
    pub col_norms: Vec<f64>,
}

impl GemmBand {
    /// Derive the band for `A (m×k) · B (k×n)`, both float64 in any layout.
    /// Refuses operands outside the admitted range of the arithmetic or a
    /// depth that leaves `γ` uninformative.
    pub fn derive(
        arithmetic: DeviceArithmetic,
        a: ArrayView2<'_, f64>,
        b: ArrayView2<'_, f64>,
    ) -> Result<Self, BandRefusal> {
        let row_norms = row_norms_upper(arithmetic, "left operand", a)?;
        let col_norms = row_norms_upper(arithmetic, "right operand", b.t())?;
        Self::from_norms(arithmetic, a.ncols(), row_norms, col_norms)
    }

    /// The band from precomputed upper bounds on the operand norms.
    pub fn from_norms(
        arithmetic: DeviceArithmetic,
        k: usize,
        row_norms: Vec<f64>,
        col_norms: Vec<f64>,
    ) -> Result<Self, BandRefusal> {
        let unit = arithmetic.op_unit().max(arithmetic.split_unit());
        let coefficient = gamma(k + 2, unit).ok_or(BandRefusal::DepthTooLarge {
            operations: k + 2,
            unit,
        })?;
        let largest_row = row_norms.iter().copied().fold(0.0_f64, f64::max);
        let largest_col = col_norms.iter().copied().fold(0.0_f64, f64::max);
        let accumulation = largest_row * largest_col;
        if !(accumulation <= arithmetic.admissible_accumulation()) {
            return Err(BandRefusal::AccumulationOutOfRange {
                magnitude: accumulation,
            });
        }
        Ok(Self {
            arithmetic,
            k,
            coefficient,
            absolute: arithmetic.underflow_per_op() * HOST_EVALUATION_INFLATION,
            row_norms,
            col_norms,
        })
    }

    /// The bound on `|ĉᵢⱼ − cᵢⱼ|`.
    #[must_use]
    pub fn entry(&self, i: usize, j: usize) -> f64 {
        let (ra, cb) = (self.row_norms[i], self.col_norms[j]);
        let k = self.k as f64;
        (self.coefficient * ra * cb + self.absolute * (k + k.sqrt() * (ra + cb)))
            * HOST_EVALUATION_INFLATION
    }

    /// The largest entry bound over the whole product.
    #[must_use]
    pub fn max_entry(&self) -> f64 {
        let ra = self.row_norms.iter().copied().fold(0.0_f64, f64::max);
        let cb = self.col_norms.iter().copied().fold(0.0_f64, f64::max);
        let k = self.k as f64;
        (self.coefficient * ra * cb + self.absolute * (k + k.sqrt() * (ra + cb)))
            * HOST_EVALUATION_INFLATION
    }

    /// The largest entry bound in row `i`, `maxⱼ entry(i, j)`.
    #[must_use]
    pub fn row_max(&self, i: usize) -> f64 {
        let ra = self.row_norms[i];
        let cb = self.col_norms.iter().copied().fold(0.0_f64, f64::max);
        let k = self.k as f64;
        (self.coefficient * ra * cb + self.absolute * (k + k.sqrt() * (ra + cb)))
            * HOST_EVALUATION_INFLATION
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_the_stated_powers_of_two() {
        assert_eq!(F32_UNIT_ROUNDOFF, 2f64.powi(-24));
        assert_eq!(F32_FAITHFUL_UNIT, 2f64.powi(-23));
        assert_eq!(DF64_OP_UNIT, 2f64.powi(-45));
        assert_eq!(DF64_SPLIT_UNIT, 2f64.powi(-48));
        assert_eq!(F32_MIN_NORMAL, 2f64.powi(-126));
        assert_eq!(F32_MIN_NORMAL, f64::from(f32::MIN_POSITIVE));
        assert_eq!(F32_ADMISSIBLE_MAGNITUDE, 2f64.powi(60));
        assert_eq!(F32_ADMISSIBLE_ACCUMULATION, 2f64.powi(120));
        assert_eq!(HOST_EVALUATION_INFLATION, 1.0 + 2f64.powi(-40));
    }

    /// The derived df64 multiply-add error `5u²(1+2^-20)` (JMP/MR bounds at
    /// `u = 2^-24`) sits under the constant the bands use.
    #[test]
    fn df64_op_unit_covers_the_double_word_bounds() {
        let u = F32_UNIT_ROUNDOFF;
        let times = 5.0 * u * u;
        let plus = 3.0 * u * u / (1.0 - 4.0 * u);
        let derived = times.max(plus) * (1.0 + 2f64.powi(-20));
        assert!(derived < DF64_OP_UNIT, "{derived:e} vs {DF64_OP_UNIT:e}");
        assert!(DF64_SPLIT_UNIT >= u * u);
    }

    #[test]
    fn gamma_refuses_uninformative_depths() {
        assert!(gamma(1 << 21, F32_FAITHFUL_UNIT).is_some());
        assert!(gamma(1 << 22, F32_FAITHFUL_UNIT).is_none());
        let g = gamma(10, F32_UNIT_ROUNDOFF).expect("small depth");
        assert!(g > 10.0 * F32_UNIT_ROUNDOFF);
    }

    /// A band is an upper bound on the exact quantity: norms are inflated
    /// above their float64 evaluation, and the entry bound dominates the
    /// componentwise sum it replaces.
    #[test]
    fn gemm_band_dominates_the_componentwise_bound() {
        let (m, k, n) = (3, 5, 4);
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.71).cos() * 2.0).collect();
        let (a, b) = (
            ndarray::Array2::from_shape_vec((m, k), a).expect("shape"),
            ndarray::Array2::from_shape_vec((k, n), b).expect("shape"),
        );
        let band = GemmBand::derive(DeviceArithmetic::F32, a.view(), b.view()).expect("in range");
        for i in 0..m {
            for j in 0..n {
                let componentwise: f64 = (0..k).map(|l| (a[[i, l]] * b[[l, j]]).abs()).sum();
                assert!(band.entry(i, j) >= band.coefficient * componentwise);
            }
        }
        assert!(band.max_entry() >= band.row_max(1));
        assert!(band.row_max(1) >= band.entry(1, 2));
    }

    #[test]
    fn out_of_range_operands_are_refused_by_name() {
        use ndarray::Array2;
        let big = Array2::from_shape_vec((1, 2), vec![1.0, 2f64.powi(61)]).expect("shape");
        let ones = Array2::from_elem((2, 1), 1.0);
        let refusal = GemmBand::derive(DeviceArithmetic::F32, big.view(), ones.view())
            .expect_err("above the admitted magnitude");
        assert!(matches!(refusal, BandRefusal::MagnitudeOutOfRange { what: "left operand", .. }));
        let one = Array2::from_elem((1, 1), 1.0);
        let nan = Array2::from_elem((1, 1), f64::NAN);
        assert!(matches!(
            GemmBand::derive(DeviceArithmetic::Df64, one.view(), nan.view()),
            Err(BandRefusal::NonFinite { what: "right operand" })
        ));
        let row = Array2::from_elem((1, 1 << 10), 2f64.powi(59));
        let col = Array2::from_elem((1 << 10, 1), 2f64.powi(59));
        assert!(matches!(
            GemmBand::derive(DeviceArithmetic::F32, row.view(), col.view()),
            Err(BandRefusal::AccumulationOutOfRange { .. })
        ));
    }
}

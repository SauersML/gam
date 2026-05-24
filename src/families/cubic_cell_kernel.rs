use crate::probability::normal_cdf;
use crate::resource::{ByteLruCache, ResidentBytes};
use smallvec::{SmallVec, smallvec};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Typed errors raised by the de-nested cubic transport kernel.
///
/// Sibling families (`bernoulli_marginal_slope`, `survival_marginal_slope`,
/// `marginal_slope_shared`) currently consume the kernel's public surface via
/// `Result<_, String>`. To stay source-compatible, the kernel converts errors
/// to `String` at the boundary via `From<CubicCellKernelError> for String` and
/// keeps the public function signatures returning `Result<_, String>`.
/// `Display` is exact-byte-equivalent to the previous `format!(...)` strings.
#[derive(Clone, Debug)]
pub enum CubicCellKernelError {
    /// Interval probe / cell-bounds preconditions (ordered bounds, supported
    /// infinity patterns, positive finite width).
    InvalidInterval { reason: String },
    /// Cell-shape / branch-classification failure: tail cells not affine,
    /// finite cells with non-positive width, non-finite affine coefficients,
    /// non-affine cell with infinite bounds, leading-coefficient degeneracy
    /// in the moment recurrence, etc.
    InvalidCellShape { reason: String },
    /// Reduced moment vector (or polynomial-convolution scratch) is shorter
    /// than the polynomial degree the leaf needs to evaluate.
    InsufficientMoments { reason: String },
    /// Bivariate-normal CDF domain validation (non-finite/non-infinite
    /// argument, non-finite correlation).
    BivariateNormalDomain { reason: String },
}

impl std::fmt::Display for CubicCellKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CubicCellKernelError::InvalidInterval { reason }
            | CubicCellKernelError::InvalidCellShape { reason }
            | CubicCellKernelError::InsufficientMoments { reason }
            | CubicCellKernelError::BivariateNormalDomain { reason } => f.write_str(reason),
        }
    }
}

impl std::error::Error for CubicCellKernelError {}

impl From<CubicCellKernelError> for String {
    fn from(err: CubicCellKernelError) -> String {
        err.to_string()
    }
}

impl CubicCellKernelError {
    #[inline]
    fn invalid_interval(reason: impl Into<String>) -> Self {
        CubicCellKernelError::InvalidInterval {
            reason: reason.into(),
        }
    }
    #[inline]
    fn invalid_cell_shape(reason: impl Into<String>) -> Self {
        CubicCellKernelError::InvalidCellShape {
            reason: reason.into(),
        }
    }
    #[inline]
    fn insufficient_moments(reason: impl Into<String>) -> Self {
        CubicCellKernelError::InsufficientMoments {
            reason: reason.into(),
        }
    }
    #[inline]
    fn bivariate_normal_domain(reason: impl Into<String>) -> Self {
        CubicCellKernelError::BivariateNormalDomain {
            reason: reason.into(),
        }
    }
}

// De-nested cubic transport kernel.
//
// This module implements the de-nested flexible-link/score-warp model
//
//   eta(z) = a + b*z + b*delta_h(z) + delta_w(a + b*z)
//
// where delta_h is the score warp and delta_w is the link deviation.
// This is not the literal nested composition L(a + b*H(z)); it is an
// additive-correction model around the affine core a + b*z.
//
// On each partition cell, both deviations are cubic polynomials, so eta is
// at most sextic in z and q(z) = 0.5*(z^2 + eta^2) is at most degree 12.
// The integral of exp(-q(z)) is evaluated by transporting from the affine
// anchor (c2=c3=0, where q is Gaussian and the integral reduces to BVN)
// to the target non-affine cell via the polynomial moment recurrence.
//
// The partition covers (-∞, +∞) with:
//   • two semi-infinite affine TAIL cells (outside all deviation support),
//   • finitely many interior cells (each a sextic microcell).
// Because tail cells have constant deviations (c2=c3=0), their bounds
// are parameter-independent, so no Leibniz boundary-motion corrections
// appear in the derivatives.
//
// Shared by bernoulli_marginal_slope and survival_marginal_slope families.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalSpanCubic {
    pub left: f64,
    pub right: f64,
    pub c0: f64,
    pub c1: f64,
    pub c2: f64,
    pub c3: f64,
}

impl LocalSpanCubic {
    #[inline]
    pub fn evaluate(self, x: f64) -> f64 {
        let t = x - self.left;
        self.c0 + self.c1 * t + self.c2 * t * t + self.c3 * t * t * t
    }

    #[inline]
    pub fn first_derivative(self, x: f64) -> f64 {
        let t = x - self.left;
        self.c1 + 2.0 * self.c2 * t + 3.0 * self.c3 * t * t
    }

    #[inline]
    pub fn second_derivative(self, x: f64) -> f64 {
        let t = x - self.left;
        2.0 * self.c2 + 6.0 * self.c3 * t
    }
}

pub const ANCHORED_DEVIATION_KERNEL: &str = "DenestedCubicTransport";
/// Default normalized non-affine branch tolerance used by [`branch_cell`].
///
/// Keep this cutoff explicit and hill-climbable: the biobank-shape cycle-0
/// sweep evaluated `{1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3}` against the
/// legacy transport path.  The more aggressive candidates require an
/// end-to-end beta acceptance run before promotion; the default therefore
/// remains the legacy `1e-10` value to preserve bit-for-bit model behavior.
pub const NORMALIZED_CELL_BRANCH_TOL: f64 = 1e-10;

const INV_TWO_PI: f64 = 1.0 / std::f64::consts::TAU;

const GL_NODES: [f64; 384] = [
    -9.999_804_411_726_474e-1,
    -9.998_969_471_378_596e-1,
    -9.997_467_408_113_523e-1,
    -9.995_297_988_558_859e-1,
    -9.992_461_316_671_845e-1,
    -9.988_957_572_063_257e-1,
    -9.984_786_985_384_589e-1,
    -9.979_949_833_727_938e-1,
    -9.974_446_439_389_107e-1,
    -9.968_277_169_440_913e-1,
    -9.961_442_435_551_087e-1,
    -9.953_942_693_885_953e-1,
    -9.945_778_445_047_068e-1,
    -9.936_950_234_020_883e-1,
    -9.927_458_650_133_153e-1,
    -9.917_304_327_004_32e-1,
    -9.906_487_942_504_061e-1,
    -9.895_010_218_704_087e-1,
    -9.882_871_921_828_699e-1,
    -9.870_073_862_202_815e-1,
    -9.856_616_894_197_333e-1,
    -9.842_501_916_171_713e-1,
    -9.827_729_870_413_743e-1,
    -9.812_301_743_076_443e-1,
    -9.796_218_564_112_101e-1,
    -9.779_481_407_203_411e-1,
    -9.762_091_389_691_724e-1,
    -9.744_049_672_502_397e-1,
    -9.725_357_460_067_257e-1,
    -9.706_016_000_244_151e-1,
    -9.686_026_584_233_628e-1,
    -9.665_390_546_492_71e-1,
    -9.644_109_264_645_802e-1,
    -9.622_184_159_392_698e-1,
    -9.599_616_694_413_742e-1,
    -9.576_408_376_272_095e-1,
    -9.552_560_754_313_16e-1,
    -9.528_075_420_561_144e-1,
    -9.502_954_009_612_771e-1,
    -9.477_198_198_528_157e-1,
    -9.450_809_706_718_851e-1,
    -9.423_790_295_833_044e-1,
    -9.396_141_769_637_963e-1,
    -9.367_865_973_899_459e-1,
    -9.338_964_796_258_775e-1,
    -9.309_440_166_106_54e-1,
    -9.279_294_054_453_956e-1,
    -9.248_528_473_801_222e-1,
    -9.217_145_478_003_181e-1,
    -9.185_147_162_132_208e-1,
    -9.152_535_662_338_34e-1,
    -9.119_313_155_706_682e-1,
    -9.085_481_860_112_055e-1,
    -9.051_044_034_070_944e-1,
    -9.016_001_976_590_722e-1,
    -8.980_358_027_016_164e-1,
    -8.944_114_564_873_288e-1,
    -8.907_274_009_710_492e-1,
    -8.869_838_820_937_034e-1,
    -8.831_811_497_658_847e-1,
    -8.793_194_578_511_7e-1,
    -8.753_990_641_491_725e-1,
    -8.714_202_303_783_312e-1,
    -8.673_832_221_584_393e-1,
    -8.632_883_089_929_12e-1,
    -8.591_357_642_507_945e-1,
    -8.549_258_651_485_127e-1,
    -8.506_588_927_313_666e-1,
    -8.463_351_318_547_683e-1,
    -8.419_548_711_652_254e-1,
    -8.375_184_030_810_715e-1,
    -8.330_260_237_729_452e-1,
    -8.284_780_331_440_178e-1,
    -8.238_747_348_099_726e-1,
    -8.192_164_360_787_36e-1,
    -8.145_034_479_299_62e-1,
    -8.097_360_849_942_72e-1,
    -8.049_146_655_322_506e-1,
    -8.000_395_114_131_988e-1,
    -7.951_109_480_936_471e-1,
    -7.901_293_045_956_28e-1,
    -7.850_949_134_847_117e-1,
    -7.800_081_108_478_04e-1,
    -7.748_692_362_707_1e-1,
    -7.696_786_328_154_644e-1,
    -7.644_366_469_974_285e-1,
    -7.591_436_287_621_58e-1,
    -7.537_999_314_620_412e-1,
    -7.484_059_118_327_094e-1,
    -7.429_619_299_692_227e-1,
    -7.374_683_493_020_299e-1,
    -7.319_255_365_727_068e-1,
    -7.263_338_618_094_733e-1,
    -7.206_936_983_024_912e-1,
    -7.150_054_225_789_432e-1,
    -7.092_694_143_778_975e-1,
    -7.034_860_566_249_567e-1,
    -6.976_557_354_066_943e-1,
    -6.917_788_399_448_808e-1,
    -6.858_557_625_704_99e-1,
    -6.798_868_986_975_534e-1,
    -6.738_726_467_966_731e-1,
    -6.678_134_083_685_102e-1,
    -6.617_095_879_169_366e-1,
    -6.555_615_929_220_4e-1,
    -6.493_698_338_129_212e-1,
    -6.431_347_239_402_948e-1,
    -6.368_566_795_488_945e-1,
    -6.305_361_197_496_849e-1,
    -6.241_734_664_918_837e-1,
    -6.177_691_445_347_913e-1,
    -6.113_235_814_194_364e-1,
    -6.048_372_074_400_329e-1,
    -5.983_104_556_152_549e-1,
    -5.917_437_616_593_286e-1,
    -5.851_375_639_529_456e-1,
    -5.784_923_035_139_965e-1,
    -5.718_084_239_681_3e-1,
    -5.650_863_715_191_369e-1,
    -5.583_265_949_191_623e-1,
    -5.515_295_454_387_482e-1,
    -5.446_956_768_367_068e-1,
    -5.378_254_453_298_289e-1,
    -5.309_193_095_624_275e-1,
    -5.239_777_305_757_194e-1,
    -5.170_011_717_770_473e-1,
    -5.099_900_989_089_429e-1,
    -5.029_449_800_180_356e-1,
    -4.958_662_854_238_058_4e-1,
    -4.887_544_876_871_878e-1,
    -4.816_100_615_790_221e-1,
    -4.744_334_840_483_605_5e-1,
    -4.672_252_341_906_264e-1,
    -4.599_857_932_156_304e-1,
    -4.527_156_444_154_463_7e-1,
    -4.454_152_731_321_473_5e-1,
    -4.380_851_667_254_05e-1,
    -4.307_258_145_399_544_5e-1,
    -4.233_377_078_729_265e-1,
    -4.159_213_399_410_494e-1,
    -4.084_772_058_477_228e-1,
    -4.010_058_025_499_653e-1,
    -3.935_076_288_252_386e-1,
    -3.859_831_852_381_500_6e-1,
    -3.784_329_741_070_358_6e-1,
    -3.708_574_994_704_271e-1,
    -3.632_572_670_534_011e-1,
    -3.556_327_842_338_202e-1,
    -3.479_845_600_084_600_6e-1,
    -3.403_131_049_590_297e-1,
    -3.326_189_312_180_866e-1,
    -3.249_025_524_348_469_5e-1,
    -3.171_644_837_408_958_4e-1,
    -3.094_052_417_157_978e-1,
    -3.016_253_443_526_109e-1,
    -2.938_253_110_233_064_5e-1,
    -2.860_056_624_440_967_5e-1,
    -2.781_669_206_406_729e-1,
    -2.703_096_089_133_553e-1,
    -2.624_342_518_021_592_4e-1,
    -2.545_413_750_517_773e-1,
    -2.466_315_055_764_817_5e-1,
    -2.387_051_714_249_486_3e-1,
    -2.307_629_017_450_062e-1,
    -2.228_052_267_483_099_4e-1,
    -2.148_326_776_749_466_5e-1,
    -2.068_457_867_579_697_5e-1,
    -1.988_450_871_878_683_4e-1,
    -1.908_311_130_769_724_5e-1,
    -1.828_043_994_237_965_6e-1,
    -1.747_654_820_773_241_2e-1,
    -1.667_148_977_012_352_4e-1,
    -1.586_531_837_380_799_3e-1,
    -1.505_808_783_733_995e-1,
    -1.424_985_204_997_981_4e-1,
    -1.344_066_496_809_674_7e-1,
    -1.263_058_061_156_663e-1,
    -1.181_965_306_016_578_4e-1,
    -1.100_793_644_996_070_4e-1,
    -1.019_548_496_969_403_7e-1,
    -9.382_352_857_167_028e-2,
    -8.568_594_395_618_719e-2,
    -7.754_263_910_102_077e-2,
    -6.939_415_763_857_37e-2,
    -6.124_104_354_682_962e-2,
    -5.308_384_111_303_817_6e-2,
    -4.492_309_489_737_94e-2,
    -3.675_934_969_660_982e-2,
    -2.859_315_050_769_284_7e-2,
    -2.042_504_249_141_571e-2,
    -1.225_557_093_599_553_8e-2,
    -4.085_281_220_676_868e-3,
    4.085_281_220_676_868e-3,
    1.225_557_093_599_553_8e-2,
    2.042_504_249_141_571e-2,
    2.859_315_050_769_284_7e-2,
    3.675_934_969_660_982e-2,
    4.492_309_489_737_94e-2,
    5.308_384_111_303_817_6e-2,
    6.124_104_354_682_962e-2,
    6.939_415_763_857_37e-2,
    7.754_263_910_102_077e-2,
    8.568_594_395_618_719e-2,
    9.382_352_857_167_028e-2,
    1.019_548_496_969_403_7e-1,
    1.100_793_644_996_070_4e-1,
    1.181_965_306_016_578_4e-1,
    1.263_058_061_156_663e-1,
    1.344_066_496_809_674_7e-1,
    1.424_985_204_997_981_4e-1,
    1.505_808_783_733_995e-1,
    1.586_531_837_380_799_3e-1,
    1.667_148_977_012_352_4e-1,
    1.747_654_820_773_241_2e-1,
    1.828_043_994_237_965_6e-1,
    1.908_311_130_769_724_5e-1,
    1.988_450_871_878_683_4e-1,
    2.068_457_867_579_697_5e-1,
    2.148_326_776_749_466_5e-1,
    2.228_052_267_483_099_4e-1,
    2.307_629_017_450_062e-1,
    2.387_051_714_249_486_3e-1,
    2.466_315_055_764_817_5e-1,
    2.545_413_750_517_773e-1,
    2.624_342_518_021_592_4e-1,
    2.703_096_089_133_553e-1,
    2.781_669_206_406_729e-1,
    2.860_056_624_440_967_5e-1,
    2.938_253_110_233_064_5e-1,
    3.016_253_443_526_109e-1,
    3.094_052_417_157_978e-1,
    3.171_644_837_408_958_4e-1,
    3.249_025_524_348_469_5e-1,
    3.326_189_312_180_866e-1,
    3.403_131_049_590_297e-1,
    3.479_845_600_084_600_6e-1,
    3.556_327_842_338_202e-1,
    3.632_572_670_534_011e-1,
    3.708_574_994_704_271e-1,
    3.784_329_741_070_358_6e-1,
    3.859_831_852_381_500_6e-1,
    3.935_076_288_252_386e-1,
    4.010_058_025_499_653e-1,
    4.084_772_058_477_228e-1,
    4.159_213_399_410_494e-1,
    4.233_377_078_729_265e-1,
    4.307_258_145_399_544_5e-1,
    4.380_851_667_254_05e-1,
    4.454_152_731_321_473_5e-1,
    4.527_156_444_154_463_7e-1,
    4.599_857_932_156_304e-1,
    4.672_252_341_906_264e-1,
    4.744_334_840_483_605_5e-1,
    4.816_100_615_790_221e-1,
    4.887_544_876_871_878e-1,
    4.958_662_854_238_058_4e-1,
    5.029_449_800_180_356e-1,
    5.099_900_989_089_429e-1,
    5.170_011_717_770_473e-1,
    5.239_777_305_757_194e-1,
    5.309_193_095_624_275e-1,
    5.378_254_453_298_289e-1,
    5.446_956_768_367_068e-1,
    5.515_295_454_387_482e-1,
    5.583_265_949_191_623e-1,
    5.650_863_715_191_369e-1,
    5.718_084_239_681_3e-1,
    5.784_923_035_139_965e-1,
    5.851_375_639_529_456e-1,
    5.917_437_616_593_286e-1,
    5.983_104_556_152_549e-1,
    6.048_372_074_400_329e-1,
    6.113_235_814_194_364e-1,
    6.177_691_445_347_913e-1,
    6.241_734_664_918_837e-1,
    6.305_361_197_496_849e-1,
    6.368_566_795_488_945e-1,
    6.431_347_239_402_948e-1,
    6.493_698_338_129_212e-1,
    6.555_615_929_220_4e-1,
    6.617_095_879_169_366e-1,
    6.678_134_083_685_102e-1,
    6.738_726_467_966_731e-1,
    6.798_868_986_975_534e-1,
    6.858_557_625_704_99e-1,
    6.917_788_399_448_808e-1,
    6.976_557_354_066_943e-1,
    7.034_860_566_249_567e-1,
    7.092_694_143_778_975e-1,
    7.150_054_225_789_432e-1,
    7.206_936_983_024_912e-1,
    7.263_338_618_094_733e-1,
    7.319_255_365_727_068e-1,
    7.374_683_493_020_299e-1,
    7.429_619_299_692_227e-1,
    7.484_059_118_327_094e-1,
    7.537_999_314_620_412e-1,
    7.591_436_287_621_58e-1,
    7.644_366_469_974_285e-1,
    7.696_786_328_154_644e-1,
    7.748_692_362_707_1e-1,
    7.800_081_108_478_04e-1,
    7.850_949_134_847_117e-1,
    7.901_293_045_956_28e-1,
    7.951_109_480_936_471e-1,
    8.000_395_114_131_988e-1,
    8.049_146_655_322_506e-1,
    8.097_360_849_942_72e-1,
    8.145_034_479_299_62e-1,
    8.192_164_360_787_36e-1,
    8.238_747_348_099_726e-1,
    8.284_780_331_440_178e-1,
    8.330_260_237_729_452e-1,
    8.375_184_030_810_715e-1,
    8.419_548_711_652_254e-1,
    8.463_351_318_547_683e-1,
    8.506_588_927_313_666e-1,
    8.549_258_651_485_127e-1,
    8.591_357_642_507_945e-1,
    8.632_883_089_929_12e-1,
    8.673_832_221_584_393e-1,
    8.714_202_303_783_312e-1,
    8.753_990_641_491_725e-1,
    8.793_194_578_511_7e-1,
    8.831_811_497_658_847e-1,
    8.869_838_820_937_034e-1,
    8.907_274_009_710_492e-1,
    8.944_114_564_873_288e-1,
    8.980_358_027_016_164e-1,
    9.016_001_976_590_722e-1,
    9.051_044_034_070_944e-1,
    9.085_481_860_112_055e-1,
    9.119_313_155_706_682e-1,
    9.152_535_662_338_34e-1,
    9.185_147_162_132_208e-1,
    9.217_145_478_003_181e-1,
    9.248_528_473_801_222e-1,
    9.279_294_054_453_956e-1,
    9.309_440_166_106_54e-1,
    9.338_964_796_258_775e-1,
    9.367_865_973_899_459e-1,
    9.396_141_769_637_963e-1,
    9.423_790_295_833_044e-1,
    9.450_809_706_718_851e-1,
    9.477_198_198_528_157e-1,
    9.502_954_009_612_771e-1,
    9.528_075_420_561_144e-1,
    9.552_560_754_313_16e-1,
    9.576_408_376_272_095e-1,
    9.599_616_694_413_742e-1,
    9.622_184_159_392_698e-1,
    9.644_109_264_645_802e-1,
    9.665_390_546_492_71e-1,
    9.686_026_584_233_628e-1,
    9.706_016_000_244_151e-1,
    9.725_357_460_067_257e-1,
    9.744_049_672_502_397e-1,
    9.762_091_389_691_724e-1,
    9.779_481_407_203_411e-1,
    9.796_218_564_112_101e-1,
    9.812_301_743_076_443e-1,
    9.827_729_870_413_743e-1,
    9.842_501_916_171_713e-1,
    9.856_616_894_197_333e-1,
    9.870_073_862_202_815e-1,
    9.882_871_921_828_699e-1,
    9.895_010_218_704_087e-1,
    9.906_487_942_504_061e-1,
    9.917_304_327_004_32e-1,
    9.927_458_650_133_153e-1,
    9.936_950_234_020_883e-1,
    9.945_778_445_047_068e-1,
    9.953_942_693_885_953e-1,
    9.961_442_435_551_087e-1,
    9.968_277_169_440_913e-1,
    9.974_446_439_389_107e-1,
    9.979_949_833_727_938e-1,
    9.984_786_985_384_589e-1,
    9.988_957_572_063_257e-1,
    9.992_461_316_671_845e-1,
    9.995_297_988_558_859e-1,
    9.997_467_408_113_523e-1,
    9.998_969_471_378_596e-1,
    9.999_804_411_726_474e-1,
];
const GL_WEIGHTS: [f64; 384] = [
    5.019_410_348_676_869_6e-5,
    1.168_390_665_730_266_3e-4,
    1.835_749_193_551_655_8e-4,
    2.503_070_890_844_105e-4,
    3.170_242_698_112_815e-4,
    3.837_208_020_912_921_4e-4,
    4.503_919_137_716_827e-4,
    5.170_330_453_491_649e-4,
    5.836_397_042_630_135e-4,
    6.502_074_240_969_948e-4,
    7.167_317_509_947_801e-4,
    7.832_082_385_905_168e-4,
    8.496_324_460_039_209e-4,
    9.159_999_370_632_641e-4,
    9.823_062_800_663_463e-4,
    1.048_547_047_793_689_5e-3,
    1.114_717_817_647_310_6e-3,
    1.180_814_171_855_922e-3,
    1.246_831_697_715_441_5e-3,
    1.312_765_987_850_66e-3,
    1.378_612_640_487_646_8e-3,
    1.444_367_259_734_736e-3,
    1.510_025_455_865_810_3e-3,
    1.575_582_845_607_936_8e-3,
    1.641_035_052_429_271_5e-3,
    1.706_377_706_828_447_1e-3,
    1.771_606_446_623_834_7e-3,
    1.836_716_917_243_567_5e-3,
    1.901_704_772_014_899_2e-3,
    1.966_565_672_453_437e-3,
    2.031_295_288_552_398_4e-3,
    2.095_889_299_071_020_6e-3,
    2.160_343_391_822_734_3e-3,
    2.224_653_263_962_713e-3,
    2.288_814_622_274_955e-3,
    2.352_823_183_458_769e-3,
    2.416_674_674_414_340_5e-3,
    2.480_364_832_528_265_6e-3,
    2.543_889_405_957_74e-3,
    2.607_244_153_914_452e-3,
    2.670_424_846_947_554e-3,
    2.733_427_267_226_093_3e-3,
    2.796_247_208_820_428e-3,
    2.858_880_477_983_06e-3,
    2.921_322_893_428_515_3e-3,
    2.983_570_286_612_554_5e-3,
    3.045_618_502_010_327_8e-3,
    3.107_463_397_393_755_5e-3,
    3.169_100_844_108_32e-3,
    3.230_526_727_348_174e-3,
    3.291_736_946_431_361e-3,
    3.352_727_415_073_250_3e-3,
    3.413_494_061_659_418_4e-3,
    3.474_032_829_517_317e-3,
    3.534_339_677_187_348_4e-3,
    3.594_410_578_692_452e-3,
    3.654_241_523_806_987e-3,
    3.713_828_518_324_312_5e-3,
    3.773_167_584_323_583_5e-3,
    3.832_254_760_435_171e-3,
    3.891_086_102_105_193_4e-3,
    3.949_657_681_858_895e-3,
    4.007_965_589_562_678e-3,
    4.066_005_932_685_269e-3,
    4.123_774_836_557_6e-3,
    4.181_268_444_631_281e-3,
    4.238_482_918_736_289e-3,
    4.295_414_439_336_925e-3,
    4.352_059_205_787_275e-3,
    4.408_413_436_584_285e-3,
    4.464_473_369_620_78e-3,
    4.520_235_262_436_235e-3,
    4.575_695_392_466_791e-3,
    4.630_850_057_293_894e-3,
    4.685_695_574_891_041e-3,
    4.740_228_283_870_022e-3,
    4.794_444_543_725_102e-3,
    4.848_340_735_076_109e-3,
    4.901_913_259_910_197e-3,
    4.955_158_541_821_682_4e-3,
    5.008_073_026_251_332e-3,
    5.060_653_180_723_101_4e-3,
    5.112_895_495_080_397e-3,
    5.164_796_481_720_011e-3,
    5.216_352_675_825_451e-3,
    5.267_560_635_597_735e-3,
    5.318_416_942_485_385e-3,
    5.368_918_201_412_827e-3,
    5.419_061_041_006_627e-3,
    5.468_842_113_820_941e-3,
    5.518_258_096_560_71e-3,
    5.567_305_690_303_767e-3,
    5.615_981_620_720_803e-3,
    5.664_282_638_294_182e-3,
    5.712_205_518_534_655e-3,
    5.759_747_062_196_925_5e-3,
    5.806_904_095_492_818e-3,
    5.853_673_470_303_617_4e-3,
    5.900_052_064_389_824e-3,
    5.946_036_781_599_814e-3,
    5.991_624_552_076_468e-3,
    6.036_812_332_462_087e-3,
    6.081_597_106_101_673e-3,
    6.125_975_883_244_196e-3,
    6.169_945_701_242_237e-3,
    6.213_503_624_749_591e-3,
    6.256_646_745_917_723e-3,
    6.299_372_184_589_237e-3,
    6.341_677_088_490_664e-3,
    6.383_558_633_422_572e-3,
    6.425_014_023_448_273e-3,
    6.466_040_491_080_434e-3,
    6.506_635_297_465_724e-3,
    6.546_795_732_567_842_5e-3,
    6.586_519_115_348_261e-3,
    6.625_802_793_945_317e-3,
    6.664_644_145_851_14e-3,
    6.703_040_578_086_941e-3,
    6.740_989_527_375_895e-3,
    6.778_488_460_314_126e-3,
    6.815_534_873_540_5e-3,
    6.852_126_293_902_878e-3,
    6.888_260_278_623_754e-3,
    6.923_934_415_463_31e-3,
    6.959_146_322_880_146_5e-3,
    6.993_893_650_190_702e-3,
    7.028_174_077_725_734e-3,
    7.061_985_316_985_506e-3,
    7.095_325_110_792_439e-3,
    7.128_191_233_441_844e-3,
    7.160_581_490_850_321e-3,
    7.192_493_720_702_486e-3,
    7.223_925_792_595_309e-3,
    7.254_875_608_179_984e-3,
    7.285_341_101_302_512e-3,
    7.315_320_238_141_324_5e-3,
    7.344_811_017_343_063e-3,
    7.373_811_470_156_258e-3,
    7.402_319_660_562_818e-3,
    7.430_333_685_407_178e-3,
    7.457_851_674_523_319e-3,
    7.484_871_790_859_79e-3,
    7.511_392_230_602_079e-3,
    7.537_411_223_293_362e-3,
    7.562_927_031_952_382e-3,
    7.587_937_953_189_561_5e-3,
    7.612_442_317_320_796e-3,
    7.636_438_488_478_739e-3,
    7.659_924_864_722_064e-3,
    7.682_899_878_142_539e-3,
    7.705_361_994_969_524e-3,
    7.727_309_715_672_44e-3,
    7.748_741_575_060_914e-3,
    7.769_656_142_382_462e-3,
    7.790_052_021_418_226e-3,
    7.809_927_850_575_903e-3,
    7.829_282_302_980_82e-3,
    7.848_114_086_564_56e-3,
    7.866_421_944_151_094e-3,
    7.884_204_653_540_665e-3,
    7.901_461_027_591_6e-3,
    7.918_189_914_299_318e-3,
    7.934_390_196_873_448e-3,
    7.950_060_793_812_204e-3,
    7.965_200_658_974_709e-3,
    7.979_808_781_650_77e-3,
    7.993_884_186_628_266e-3,
    8.007_425_934_258_548e-3,
    8.020_433_120_518_866e-3,
    8.032_904_877_072_8e-3,
    8.044_840_371_328_26e-3,
    8.056_238_806_493_175e-3,
    8.067_099_421_628_42e-3,
    8.077_421_491_698_82e-3,
    8.087_204_327_621_594e-3,
    8.096_447_276_312_202e-3,
    8.105_149_720_727_933e-3,
    8.113_311_079_909_208e-3,
    8.120_930_809_018_415e-3,
    8.128_008_399_376_085e-3,
    8.134_543_378_495_033e-3,
    8.140_535_310_111_77e-3,
    8.145_983_794_215_77e-3,
    8.150_888_467_075_875e-3,
    8.155_249_001_265_092e-3,
    8.159_065_105_681_899e-3,
    8.162_336_525_570_1e-3,
    8.165_063_042_535_465e-3,
    8.167_244_474_560_707e-3,
    8.168_880_676_017_344e-3,
    8.169_971_537_675_47e-3,
    8.170_516_986_711_104e-3,
    8.170_516_986_711_104e-3,
    8.169_971_537_675_47e-3,
    8.168_880_676_017_344e-3,
    8.167_244_474_560_707e-3,
    8.165_063_042_535_465e-3,
    8.162_336_525_570_1e-3,
    8.159_065_105_681_899e-3,
    8.155_249_001_265_092e-3,
    8.150_888_467_075_875e-3,
    8.145_983_794_215_77e-3,
    8.140_535_310_111_77e-3,
    8.134_543_378_495_033e-3,
    8.128_008_399_376_085e-3,
    8.120_930_809_018_415e-3,
    8.113_311_079_909_208e-3,
    8.105_149_720_727_933e-3,
    8.096_447_276_312_202e-3,
    8.087_204_327_621_594e-3,
    8.077_421_491_698_82e-3,
    8.067_099_421_628_42e-3,
    8.056_238_806_493_175e-3,
    8.044_840_371_328_26e-3,
    8.032_904_877_072_8e-3,
    8.020_433_120_518_866e-3,
    8.007_425_934_258_548e-3,
    7.993_884_186_628_266e-3,
    7.979_808_781_650_77e-3,
    7.965_200_658_974_709e-3,
    7.950_060_793_812_204e-3,
    7.934_390_196_873_448e-3,
    7.918_189_914_299_318e-3,
    7.901_461_027_591_6e-3,
    7.884_204_653_540_665e-3,
    7.866_421_944_151_094e-3,
    7.848_114_086_564_56e-3,
    7.829_282_302_980_82e-3,
    7.809_927_850_575_903e-3,
    7.790_052_021_418_226e-3,
    7.769_656_142_382_462e-3,
    7.748_741_575_060_914e-3,
    7.727_309_715_672_44e-3,
    7.705_361_994_969_524e-3,
    7.682_899_878_142_539e-3,
    7.659_924_864_722_064e-3,
    7.636_438_488_478_739e-3,
    7.612_442_317_320_796e-3,
    7.587_937_953_189_561_5e-3,
    7.562_927_031_952_382e-3,
    7.537_411_223_293_362e-3,
    7.511_392_230_602_079e-3,
    7.484_871_790_859_79e-3,
    7.457_851_674_523_319e-3,
    7.430_333_685_407_178e-3,
    7.402_319_660_562_818e-3,
    7.373_811_470_156_258e-3,
    7.344_811_017_343_063e-3,
    7.315_320_238_141_324_5e-3,
    7.285_341_101_302_512e-3,
    7.254_875_608_179_984e-3,
    7.223_925_792_595_309e-3,
    7.192_493_720_702_486e-3,
    7.160_581_490_850_321e-3,
    7.128_191_233_441_844e-3,
    7.095_325_110_792_439e-3,
    7.061_985_316_985_506e-3,
    7.028_174_077_725_734e-3,
    6.993_893_650_190_702e-3,
    6.959_146_322_880_146_5e-3,
    6.923_934_415_463_31e-3,
    6.888_260_278_623_754e-3,
    6.852_126_293_902_878e-3,
    6.815_534_873_540_5e-3,
    6.778_488_460_314_126e-3,
    6.740_989_527_375_895e-3,
    6.703_040_578_086_941e-3,
    6.664_644_145_851_14e-3,
    6.625_802_793_945_317e-3,
    6.586_519_115_348_261e-3,
    6.546_795_732_567_842_5e-3,
    6.506_635_297_465_724e-3,
    6.466_040_491_080_434e-3,
    6.425_014_023_448_273e-3,
    6.383_558_633_422_572e-3,
    6.341_677_088_490_664e-3,
    6.299_372_184_589_237e-3,
    6.256_646_745_917_723e-3,
    6.213_503_624_749_591e-3,
    6.169_945_701_242_237e-3,
    6.125_975_883_244_196e-3,
    6.081_597_106_101_673e-3,
    6.036_812_332_462_087e-3,
    5.991_624_552_076_468e-3,
    5.946_036_781_599_814e-3,
    5.900_052_064_389_824e-3,
    5.853_673_470_303_617_4e-3,
    5.806_904_095_492_818e-3,
    5.759_747_062_196_925_5e-3,
    5.712_205_518_534_655e-3,
    5.664_282_638_294_182e-3,
    5.615_981_620_720_803e-3,
    5.567_305_690_303_767e-3,
    5.518_258_096_560_71e-3,
    5.468_842_113_820_941e-3,
    5.419_061_041_006_627e-3,
    5.368_918_201_412_827e-3,
    5.318_416_942_485_385e-3,
    5.267_560_635_597_735e-3,
    5.216_352_675_825_451e-3,
    5.164_796_481_720_011e-3,
    5.112_895_495_080_397e-3,
    5.060_653_180_723_101_4e-3,
    5.008_073_026_251_332e-3,
    4.955_158_541_821_682_4e-3,
    4.901_913_259_910_197e-3,
    4.848_340_735_076_109e-3,
    4.794_444_543_725_102e-3,
    4.740_228_283_870_022e-3,
    4.685_695_574_891_041e-3,
    4.630_850_057_293_894e-3,
    4.575_695_392_466_791e-3,
    4.520_235_262_436_235e-3,
    4.464_473_369_620_78e-3,
    4.408_413_436_584_285e-3,
    4.352_059_205_787_275e-3,
    4.295_414_439_336_925e-3,
    4.238_482_918_736_289e-3,
    4.181_268_444_631_281e-3,
    4.123_774_836_557_6e-3,
    4.066_005_932_685_269e-3,
    4.007_965_589_562_678e-3,
    3.949_657_681_858_895e-3,
    3.891_086_102_105_193_4e-3,
    3.832_254_760_435_171e-3,
    3.773_167_584_323_583_5e-3,
    3.713_828_518_324_312_5e-3,
    3.654_241_523_806_987e-3,
    3.594_410_578_692_452e-3,
    3.534_339_677_187_348_4e-3,
    3.474_032_829_517_317e-3,
    3.413_494_061_659_418_4e-3,
    3.352_727_415_073_250_3e-3,
    3.291_736_946_431_361e-3,
    3.230_526_727_348_174e-3,
    3.169_100_844_108_32e-3,
    3.107_463_397_393_755_5e-3,
    3.045_618_502_010_327_8e-3,
    2.983_570_286_612_554_5e-3,
    2.921_322_893_428_515_3e-3,
    2.858_880_477_983_06e-3,
    2.796_247_208_820_428e-3,
    2.733_427_267_226_093_3e-3,
    2.670_424_846_947_554e-3,
    2.607_244_153_914_452e-3,
    2.543_889_405_957_74e-3,
    2.480_364_832_528_265_6e-3,
    2.416_674_674_414_340_5e-3,
    2.352_823_183_458_769e-3,
    2.288_814_622_274_955e-3,
    2.224_653_263_962_713e-3,
    2.160_343_391_822_734_3e-3,
    2.095_889_299_071_020_6e-3,
    2.031_295_288_552_398_4e-3,
    1.966_565_672_453_437e-3,
    1.901_704_772_014_899_2e-3,
    1.836_716_917_243_567_5e-3,
    1.771_606_446_623_834_7e-3,
    1.706_377_706_828_447_1e-3,
    1.641_035_052_429_271_5e-3,
    1.575_582_845_607_936_8e-3,
    1.510_025_455_865_810_3e-3,
    1.444_367_259_734_736e-3,
    1.378_612_640_487_646_8e-3,
    1.312_765_987_850_66e-3,
    1.246_831_697_715_441_5e-3,
    1.180_814_171_855_922e-3,
    1.114_717_817_647_310_6e-3,
    1.048_547_047_793_689_5e-3,
    9.823_062_800_663_463e-4,
    9.159_999_370_632_641e-4,
    8.496_324_460_039_209e-4,
    7.832_082_385_905_168e-4,
    7.167_317_509_947_801e-4,
    6.502_074_240_969_948e-4,
    5.836_397_042_630_135e-4,
    5.170_330_453_491_649e-4,
    4.503_919_137_716_827e-4,
    3.837_208_020_912_921_4e-4,
    3.170_242_698_112_815e-4,
    2.503_070_890_844_105e-4,
    1.835_749_193_551_655_8e-4,
    1.168_390_665_730_266_3e-4,
    5.019_410_348_676_869_6e-5,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExactCellBranch {
    Affine,
    Quartic,
    Sextic,
}

/// Auto-tune the per-cell affine/non-affine branch tolerance from the cell's
/// own coefficient magnitudes.
///
/// The legacy `branch_cell` compared the normalized cubic coefficients
/// `(k2, k3)` against a single global constant.  That constant is calibrated
/// for cells whose anchor coefficients `(c0, c1)` are O(1).  When the anchor
/// dominates — e.g. a tail cell with `|c0|, |c1| >> 1` — a relative criterion
/// against the anchor magnitude is more numerically meaningful than the bare
/// global threshold, because the affine contribution to `eta` already absorbs
/// any difference at the chosen scale.
///
/// The returned tolerance is always at least [`NORMALIZED_CELL_BRANCH_TOL`],
/// so cells with O(1) anchors recover bit-identical classification with the
/// legacy code path.  This preserves numerical equivalence for the
/// established `cubic_cell_kernel` tests, including the
/// `tuned_branch_tolerance_matches_legacy_non_affine_transport_grid` grid.
#[inline]
fn effective_branch_tol(cell: DenestedCubicCell) -> f64 {
    let anchor_scale = cell.c0.abs().max(cell.c1.abs()).max(1.0);
    NORMALIZED_CELL_BRANCH_TOL * anchor_scale
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenestedCubicCell {
    pub left: f64,
    pub right: f64,
    pub c0: f64,
    pub c1: f64,
    pub c2: f64,
    pub c3: f64,
}

impl DenestedCubicCell {
    #[inline]
    pub fn eta(self, z: f64) -> f64 {
        self.c0 + self.c1 * z + self.c2 * z * z + self.c3 * z * z * z
    }

    #[inline]
    pub fn q(self, z: f64) -> f64 {
        let eta = self.eta(z);
        0.5 * (z * z + eta * eta)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CellMomentFingerprint {
    pub hash: u64,
    bins: [u64; 6],
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CellMomentCacheKey {
    pub fingerprint: CellMomentFingerprint,
    pub max_degree: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CellMomentDedupStats {
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
}

impl CellMomentDedupStats {
    #[inline]
    pub fn hit_rate(self) -> f64 {
        if self.lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.lookups as f64
        }
    }
}

pub const DEFAULT_CELL_MOMENT_DEDUP_EPSILON: f64 = 0.0;

#[inline]
fn splitmix64(x: u64) -> u64 {
    crate::linalg::utils::splitmix64_hash(x)
}

#[inline]
fn mix_fingerprint_words(words: &[u64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &word in words {
        h ^= splitmix64(word);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

#[inline]
fn quantized_cell_word(x: f64, epsilon: f64) -> u64 {
    if epsilon == 0.0 || !epsilon.is_finite() || epsilon < 0.0 || !x.is_finite() {
        return x.to_bits();
    }
    (x / epsilon).round().to_bits()
}

/// Returns a deterministic geometric fingerprint for a de-nested cubic cell.
///
/// With `epsilon == 0.0`, each coordinate is represented by its exact IEEE-754
/// bit pattern, so equal fingerprints imply bit-equal `(left, right, c0, c1,
/// c2, c3)` tuples.  With `epsilon > 0`, finite coordinates are binned to the
/// nearest multiple of `epsilon`; callers should treat this as an approximate
/// cache key and validate the resulting model error for their data.
pub fn cell_moment_fingerprint(cell: DenestedCubicCell, epsilon: f64) -> CellMomentFingerprint {
    let bins = [
        quantized_cell_word(cell.left, epsilon),
        quantized_cell_word(cell.right, epsilon),
        quantized_cell_word(cell.c0, epsilon),
        quantized_cell_word(cell.c1, epsilon),
        quantized_cell_word(cell.c2, epsilon),
        quantized_cell_word(cell.c3, epsilon),
    ];
    CellMomentFingerprint {
        hash: mix_fingerprint_words(&bins),
        bins,
    }
}

#[inline]
pub fn cell_moment_cache_key(
    cell: DenestedCubicCell,
    max_degree: usize,
    epsilon: f64,
) -> CellMomentCacheKey {
    CellMomentCacheKey {
        fingerprint: cell_moment_fingerprint(cell, epsilon),
        max_degree,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenestedPartitionCell {
    pub cell: DenestedCubicCell,
    pub score_span: LocalSpanCubic,
    pub link_span: LocalSpanCubic,
}

impl DenestedPartitionCell {}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct TailCellMomentCacheKey {
    c0_bits: u64,
    c1_bits: u64,
    endpoint_bits: u64,
    side: i8,
    max_degree: usize,
}

const TAIL_CELL_MOMENT_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;
const TAIL_CELL_MOMENT_CACHE_MAX_ENTRIES: usize = 262_144;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TailCellMomentCacheStats {
    pub hits: usize,
    pub misses: usize,
    pub entries: usize,
}

impl TailCellMomentCacheStats {
    #[inline]
    pub fn requests(self) -> usize {
        self.hits + self.misses
    }

    #[inline]
    pub fn hit_rate(self) -> f64 {
        let requests = self.requests();
        if requests == 0 {
            0.0
        } else {
            self.hits as f64 / requests as f64
        }
    }
}

/// Affine-tail cell-moment memo.
///
/// Stand-alone instances (`TailCellMomentCache::new()`) are useful when a
/// caller needs deterministic hit/miss bookkeeping that is not polluted by
/// concurrent traffic on the global memo. The production path uses the
/// global instance behind [`evaluate_cell_moments`].
#[derive(Debug)]
pub struct TailCellMomentCache {
    moments: ByteLruCache<TailCellMomentCacheKey, CellMomentState>,
    hits: usize,
    misses: usize,
}

impl Default for TailCellMomentCache {
    fn default() -> Self {
        Self {
            moments: ByteLruCache::with_max_entries(
                TAIL_CELL_MOMENT_CACHE_MAX_BYTES,
                TAIL_CELL_MOMENT_CACHE_MAX_ENTRIES,
            ),
            hits: 0,
            misses: 0,
        }
    }
}

impl TailCellMomentCache {
    /// Construct an empty cache. Hits/misses start at zero.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset the cache to its empty state. Existing entries are dropped and
    /// the hit/miss counters are zeroed.
    #[inline]
    pub fn clear(&mut self) {
        self.moments.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Snapshot of the cache's current usage stats.
    #[inline]
    pub fn stats(&self) -> TailCellMomentCacheStats {
        TailCellMomentCacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.moments.len(),
        }
    }

    /// Look up `cell` at `max_degree`, computing and inserting the result on
    /// miss. Cells outside the affine-tail keyset bypass the cache and run
    /// the uncached evaluator directly without touching the counters.
    ///
    /// Stat semantics: a **miss** is counted iff this call inserted a new
    /// entry into the bounded LRU; every cache hit increments `hits`.
    pub fn evaluate(
        &mut self,
        cell: DenestedCubicCell,
        max_degree: usize,
    ) -> Result<CellMomentState, String> {
        let Some(key) = tail_cell_cache_key(cell, max_degree) else {
            return evaluate_cell_moments_uncached(cell, max_degree);
        };
        if let Some(state) = self.moments.get(&key) {
            self.hits += 1;
            return Ok(state);
        }
        let state = evaluate_cell_moments_uncached(cell, max_degree)?;
        self.misses += 1;
        self.moments.insert(key, state.clone());
        Ok(state)
    }
}

static TAIL_CELL_MOMENT_CACHE: std::sync::OnceLock<std::sync::Mutex<TailCellMomentCache>> =
    std::sync::OnceLock::new();
static TAIL_CELL_MOMENT_CACHE_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

fn tail_cell_moment_cache() -> &'static std::sync::Mutex<TailCellMomentCache> {
    TAIL_CELL_MOMENT_CACHE.get_or_init(|| std::sync::Mutex::new(TailCellMomentCache::default()))
}

#[inline]
fn tail_cell_cache_key(
    cell: DenestedCubicCell,
    max_degree: usize,
) -> Option<TailCellMomentCacheKey> {
    if cell.c2.abs() > NORMALIZED_CELL_BRANCH_TOL || cell.c3.abs() > NORMALIZED_CELL_BRANCH_TOL {
        return None;
    }
    match (!cell.left.is_finite(), !cell.right.is_finite()) {
        (true, false) if cell.right.is_finite() => Some(TailCellMomentCacheKey {
            c0_bits: cell.c0.to_bits(),
            c1_bits: cell.c1.to_bits(),
            endpoint_bits: cell.right.to_bits(),
            side: -1,
            max_degree,
        }),
        (false, true) if cell.left.is_finite() => Some(TailCellMomentCacheKey {
            c0_bits: cell.c0.to_bits(),
            c1_bits: cell.c1.to_bits(),
            endpoint_bits: cell.left.to_bits(),
            side: 1,
            max_degree,
        }),
        _ => None,
    }
}

pub fn set_tail_cell_moment_cache_enabled(enabled: bool) {
    TAIL_CELL_MOMENT_CACHE_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

pub fn reset_tail_cell_moment_cache() {
    let mut cache = tail_cell_moment_cache()
        .lock()
        .expect("tail cell moment cache mutex poisoned");
    cache.clear();
}

pub fn tail_cell_moment_cache_stats() -> TailCellMomentCacheStats {
    let cache = tail_cell_moment_cache()
        .lock()
        .expect("tail cell moment cache mutex poisoned");
    cache.stats()
}

#[derive(Clone, Copy, Debug, Eq)]
pub struct CellFingerprint {
    c0: u64,
    c1: u64,
    c2: u64,
    c3: u64,
    left: u64,
    right: u64,
}

impl CellFingerprint {
    #[inline]
    pub fn new(cell: DenestedCubicCell) -> Self {
        Self {
            c0: cell.c0.to_bits(),
            c1: cell.c1.to_bits(),
            c2: cell.c2.to_bits(),
            c3: cell.c3.to_bits(),
            left: cell.left.to_bits(),
            right: cell.right.to_bits(),
        }
    }
}

impl PartialEq for CellFingerprint {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.c0 == other.c0
            && self.c1 == other.c1
            && self.c2 == other.c2
            && self.c3 == other.c3
            && self.left == other.left
            && self.right == other.right
    }
}

impl Hash for CellFingerprint {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.c0.hash(state);
        self.c1.hash(state);
        self.c2.hash(state);
        self.c3.hash(state);
        self.left.hash(state);
        self.right.hash(state);
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CachedCellMoments {
    /// Regular (value) cell moments, populated by
    /// `evaluate_cell_moments_cached`. None when only derivative moments
    /// have been cached for this cell. Wrapped in `Arc` so `ByteLruCache`
    /// returns lookups through cheap refcount bumps instead of deep-cloning
    /// the inline `SmallVec<[f64; 10]>` (which spills on every degree-`>= 10`
    /// request) on every hot-path LRU hit.
    state: Option<Arc<CellMomentState>>,
    /// Derivative moments, populated by
    /// `evaluate_cell_derivative_moments_cached`. None when only value
    /// moments have been cached for this cell. Both variants share the
    /// same `CellFingerprint` key so derivative-only callers do not evict
    /// pre-cached value entries and vice versa. Same `Arc` wrapping rationale
    /// as `state` above.
    derivative_state: Option<Arc<CellDerivativeMomentState>>,
}

impl CachedCellMoments {
    #[inline]
    pub fn new(state: Arc<CellMomentState>) -> Self {
        Self {
            state: Some(state),
            derivative_state: None,
        }
    }

    #[inline]
    pub fn new_derivative(state: Arc<CellDerivativeMomentState>) -> Self {
        Self {
            state: None,
            derivative_state: Some(state),
        }
    }

    #[inline]
    pub fn state_for_degree(&self, max_degree: usize) -> Option<CellMomentState> {
        let state = self.state.as_ref()?;
        if state.moments.len().saturating_sub(1) < max_degree {
            return None;
        }
        // Cached `Arc<CellMomentState>` is shared across LRU hits, so we
        // cannot reuse the inner vector in place. Clone the underlying state
        // and (rarely) truncate down to the requested degree to honour the
        // public moment-length contract.
        let mut state = (**state).clone();
        state.moments.truncate(max_degree + 1);
        Some(state)
    }

    #[inline]
    pub fn derivative_state_for_degree(
        &self,
        max_degree: usize,
    ) -> Option<CellDerivativeMomentState> {
        let state = self.derivative_state.as_ref()?;
        if state.moments.len().saturating_sub(1) < max_degree {
            return None;
        }
        // See `state_for_degree`: shared `Arc` forces an inner clone here.
        let mut state = (**state).clone();
        state.moments.truncate(max_degree + 1);
        Some(state)
    }

    #[inline]
    pub fn with_value(mut self, state: Arc<CellMomentState>) -> Self {
        self.state = Some(state);
        self
    }

    #[inline]
    pub fn with_derivative(mut self, state: Arc<CellDerivativeMomentState>) -> Self {
        self.derivative_state = Some(state);
        self
    }
}

impl ResidentBytes for CachedCellMoments {
    fn resident_bytes(&self) -> usize {
        let value_bytes = self
            .state
            .as_ref()
            .map_or(0, |state| state.resident_bytes());
        let derivative_bytes = self
            .derivative_state
            .as_ref()
            .map_or(0, |state| state.resident_bytes());
        std::mem::size_of::<Self>()
            .saturating_add(value_bytes)
            .saturating_add(derivative_bytes)
    }
}

#[derive(Debug, Default)]
pub struct CellMomentCacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
}

impl CellMomentCacheStats {
    #[inline]
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn snapshot(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }

    #[inline]
    pub fn hit_rate_delta(&self, before: (u64, u64)) -> (u64, u64, f64) {
        let (hits, misses) = self.snapshot();
        let dh = hits.saturating_sub(before.0);
        let dm = misses.saturating_sub(before.1);
        let total = dh + dm;
        let rate = if total == 0 {
            0.0
        } else {
            dh as f64 / total as f64
        };
        (dh, dm, rate)
    }
}

pub type CellMomentLruCache = ByteLruCache<CellFingerprint, CachedCellMoments>;

pub const CELL_MOMENT_INLINE_CAPACITY: usize = 10;

pub type CellMomentVec = SmallVec<[f64; CELL_MOMENT_INLINE_CAPACITY]>;

#[derive(Clone, Debug, PartialEq)]
pub struct CellMomentState {
    pub branch: ExactCellBranch,
    pub value: f64,
    pub moments: CellMomentVec,
}

impl ResidentBytes for CellMomentState {
    fn resident_bytes(&self) -> usize {
        let spilled_bytes = if self.moments.spilled() {
            self.moments
                .capacity()
                .saturating_mul(std::mem::size_of::<f64>())
        } else {
            0
        };
        std::mem::size_of::<Self>().saturating_add(spilled_bytes)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CellDerivativeMomentState {
    pub branch: ExactCellBranch,
    pub moments: CellMomentVec,
}

impl ResidentBytes for CellDerivativeMomentState {
    fn resident_bytes(&self) -> usize {
        let spilled_bytes = if self.moments.spilled() {
            self.moments
                .capacity()
                .saturating_mul(std::mem::size_of::<f64>())
        } else {
            0
        };
        std::mem::size_of::<Self>().saturating_add(spilled_bytes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellMomentStateRef<'a> {
    pub branch: ExactCellBranch,
    pub value: f64,
    pub moments: &'a [f64],
}

#[derive(Clone, Debug, Default)]
pub struct CellMomentScratch {
    moments: Vec<f64>,
}

impl CellMomentScratch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(max_degree: usize) -> Self {
        Self {
            moments: Vec::with_capacity(max_degree + 1),
        }
    }

    #[inline]
    fn prepare_moments(&mut self, len: usize) -> &mut [f64] {
        #[cfg(test)]
        if self.moments.capacity() < len {
            CELL_MOMENT_TEST_REALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.moments.resize(len, 0.0);
        self.moments.fill(0.0);
        &mut self.moments
    }
}

#[cfg(test)]
static CELL_MOMENT_TEST_REALLOCS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
pub(crate) fn reset_cell_moment_test_reallocs() {
    CELL_MOMENT_TEST_REALLOCS.store(0, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn cell_moment_test_reallocs() -> usize {
    CELL_MOMENT_TEST_REALLOCS.load(std::sync::atomic::Ordering::Relaxed)
}

/// 20-point Gauss–Legendre nodes on [-1, 1] for the Drezner–Wesolowsky
/// bivariate normal CDF representation.  20 points give >30-digit accuracy
/// for the smooth arcsin-transformed integrand, ensuring the BVN value is
/// exact to f64 precision for all (h, k, ρ).
const BVN_GL_NODES_20: [f64; 20] = [
    -0.993_128_599_185_094_9,
    -0.963_971_927_277_913_8,
    -0.912_234_428_251_326,
    -0.839_116_971_822_218_8,
    -0.746_331_906_460_150_8,
    -0.636_053_680_726_515,
    -0.510_867_001_950_827_1,
    -0.373_706_088_715_419_6,
    -0.227_785_851_141_645_1,
    -0.076_526_521_133_497_33,
    0.076_526_521_133_497_33,
    0.227_785_851_141_645_1,
    0.373_706_088_715_419_6,
    0.510_867_001_950_827_1,
    0.636_053_680_726_515,
    0.746_331_906_460_150_8,
    0.839_116_971_822_218_8,
    0.912_234_428_251_326,
    0.963_971_927_277_913_8,
    0.993_128_599_185_094_9,
];

const BVN_GL_WEIGHTS_20: [f64; 20] = [
    0.017_614_007_139_152_12,
    0.040_601_429_800_386_94,
    0.062_672_048_334_109_06,
    0.083_276_741_576_704_75,
    0.101_930_119_817_240_4,
    0.118_194_531_961_518_4,
    0.131_688_638_449_176_6,
    0.142_096_109_318_382_1,
    0.149_172_986_472_603_7,
    0.152_753_387_130_725_9,
    0.152_753_387_130_725_9,
    0.149_172_986_472_603_7,
    0.142_096_109_318_382_1,
    0.131_688_638_449_176_6,
    0.118_194_531_961_518_4,
    0.101_930_119_817_240_4,
    0.083_276_741_576_704_75,
    0.062_672_048_334_109_06,
    0.040_601_429_800_386_94,
    0.017_614_007_139_152_12,
];

fn dedup_sorted_breakpoints(points: &mut Vec<f64>) {
    points.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    points.dedup_by(|lhs, rhs| {
        if *lhs == *rhs {
            true
        } else if lhs.is_finite() && rhs.is_finite() {
            (*lhs - *rhs).abs() <= 1e-12
        } else {
            false
        }
    });
}

#[inline]
pub fn interval_probe_point(left: f64, right: f64) -> Result<f64, String> {
    if !(left < right) {
        return Err(CubicCellKernelError::invalid_interval(format!(
            "interval probe requires ordered bounds, got [{left}, {right}]"
        ))
        .into());
    }
    if left.is_finite() && right.is_finite() {
        Ok(0.5 * (left + right))
    } else if left == f64::NEG_INFINITY && right == f64::INFINITY {
        Ok(0.0)
    } else if left == f64::NEG_INFINITY && right.is_finite() {
        Ok(right - 1.0)
    } else if left.is_finite() && right == f64::INFINITY {
        Ok(left + 1.0)
    } else {
        Err(CubicCellKernelError::invalid_interval(format!(
            "interval probe requires finite bounds or full infinities, got [{left}, {right}]"
        ))
        .into())
    }
}

#[inline]
pub fn quartic_qprime_coefficients(c0: f64, c1: f64, c2: f64) -> [f64; 4] {
    [
        c0 * c1,
        1.0 + c1 * c1 + 2.0 * c0 * c2,
        3.0 * c1 * c2,
        2.0 * c2 * c2,
    ]
}

#[inline]
pub fn sextic_qprime_coefficients(c0: f64, c1: f64, c2: f64, c3: f64) -> [f64; 6] {
    [
        c0 * c1,
        1.0 + c1 * c1 + 2.0 * c0 * c2,
        3.0 * c0 * c3 + 3.0 * c1 * c2,
        4.0 * c1 * c3 + 2.0 * c2 * c2,
        5.0 * c2 * c3,
        3.0 * c3 * c3,
    ]
}

#[inline]
pub fn moment_boundary_term(cell: DenestedCubicCell, n: usize) -> f64 {
    let left_term = if cell.left.is_infinite() {
        0.0
    } else {
        cell.left.powi(n as i32) * (-cell.q(cell.left)).exp()
    };
    let right_term = if cell.right.is_infinite() {
        0.0
    } else {
        cell.right.powi(n as i32) * (-cell.q(cell.right)).exp()
    };
    right_term - left_term
}

/// Same as [`moment_boundary_term`] but takes precomputed `left^n` and
/// `right^n` so callers can roll the powers across a recurrence — each
/// iteration becomes one multiply instead of a fresh `powi(n)`.
#[inline]
fn moment_boundary_term_with_powers(
    cell: DenestedCubicCell,
    left_pow_n: f64,
    right_pow_n: f64,
) -> f64 {
    let left_term = if cell.left.is_infinite() {
        0.0
    } else {
        left_pow_n * (-cell.q(cell.left)).exp()
    };
    let right_term = if cell.right.is_infinite() {
        0.0
    } else {
        right_pow_n * (-cell.q(cell.right)).exp()
    };
    right_term - left_term
}

pub fn reduce_quartic_moments(
    cell: DenestedCubicCell,
    base_m0_m2: [f64; 3],
    max_degree: usize,
) -> Result<Vec<f64>, String> {
    if max_degree <= 2 {
        return Ok(base_m0_m2[..=max_degree].to_vec());
    }
    let d = quartic_qprime_coefficients(cell.c0, cell.c1, cell.c2);
    let lead = d[3];
    if !lead.is_finite() || lead.abs() <= 1e-18 {
        return Err(CubicCellKernelError::invalid_cell_shape(format!(
            "quartic moment reduction requires nonzero leading coefficient, got {lead:.3e}"
        ))
        .into());
    }
    let mut moments = vec![0.0; max_degree + 1];
    moments[0] = base_m0_m2[0];
    moments[1] = base_m0_m2[1];
    moments[2] = base_m0_m2[2];
    // Roll left^n / right^n across the recurrence rather than calling
    // `powi(n)` each iteration. Skip the multiply when an endpoint is
    // infinite — the boundary helper ignores the power in that case, and
    // ∞·0 would produce a NaN we'd then have to mask off anyway.
    let left_finite = cell.left.is_finite();
    let right_finite = cell.right.is_finite();
    let mut left_pow_n = if left_finite { 1.0 } else { 0.0 };
    let mut right_pow_n = if right_finite { 1.0 } else { 0.0 };
    for n in 0..=(max_degree - 3) {
        let b_n = moment_boundary_term_with_powers(cell, left_pow_n, right_pow_n);
        let mut numer = if n == 0 {
            0.0
        } else {
            (n as f64) * moments[n - 1]
        };
        for j in 0..=2 {
            numer -= d[j] * moments[n + j];
        }
        numer -= b_n;
        moments[n + 3] = numer / lead;
        if left_finite {
            left_pow_n *= cell.left;
        }
        if right_finite {
            right_pow_n *= cell.right;
        }
    }
    Ok(moments)
}

pub fn reduce_sextic_moments(
    cell: DenestedCubicCell,
    base_m0_m4: [f64; 5],
    max_degree: usize,
) -> Result<Vec<f64>, String> {
    if max_degree <= 4 {
        return Ok(base_m0_m4[..=max_degree].to_vec());
    }
    let d = sextic_qprime_coefficients(cell.c0, cell.c1, cell.c2, cell.c3);
    let lead = d[5];
    if !lead.is_finite() {
        return Err(CubicCellKernelError::invalid_cell_shape(format!(
            "sextic moment reduction encountered non-finite leading coefficient: {lead:.3e}"
        ))
        .into());
    }
    if let Some(lower_branch) = degenerate_sextic_branch(cell, lead)? {
        if lower_branch == ExactCellBranch::Quartic {
            return evaluate_non_affine_cell_state(
                DenestedCubicCell { c3: 0.0, ..cell },
                ExactCellBranch::Quartic,
                max_degree,
            )
            .map(|state| state.moments.into_vec());
        }
        return evaluate_affine_cell_state(
            DenestedCubicCell {
                left: cell.left,
                right: cell.right,
                c0: cell.c0,
                c1: cell.c1,
                c2: 0.0,
                c3: 0.0,
            },
            max_degree,
        )
        .map(|state| state.moments.into_vec());
    }
    let mut moments = vec![0.0; max_degree + 1];
    for (idx, value) in base_m0_m4.into_iter().enumerate() {
        moments[idx] = value;
    }
    let left_finite = cell.left.is_finite();
    let right_finite = cell.right.is_finite();
    let mut left_pow_n = if left_finite { 1.0 } else { 0.0 };
    let mut right_pow_n = if right_finite { 1.0 } else { 0.0 };
    for n in 0..=(max_degree - 5) {
        let b_n = moment_boundary_term_with_powers(cell, left_pow_n, right_pow_n);
        let mut numer = if n == 0 {
            0.0
        } else {
            (n as f64) * moments[n - 1]
        };
        for j in 0..=4 {
            numer -= d[j] * moments[n + j];
        }
        numer -= b_n;
        moments[n + 5] = numer / lead;
        if left_finite {
            left_pow_n *= cell.left;
        }
        if right_finite {
            right_pow_n *= cell.right;
        }
    }
    Ok(moments)
}

#[cfg(test)]
#[inline]
pub fn polynomial_value(coefficients: &[f64], z: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |acc, &coeff| acc * z + coeff)
}

#[inline]
pub fn cell_first_derivative_from_moments(
    derivative_coefficients: &[f64],
    moments: &[f64],
) -> Result<f64, String> {
    let value = moment_dot_with_coefficients(derivative_coefficients, moments, "first derivative")?;
    Ok(value * INV_TWO_PI)
}

#[inline]
pub fn cell_polynomial_integral_from_moments(
    polynomial_coefficients: &[f64],
    moments: &[f64],
    label: &str,
) -> Result<f64, String> {
    let value = moment_dot_with_coefficients(polynomial_coefficients, moments, label)?;
    Ok(value * INV_TWO_PI)
}

#[inline]
pub fn cell_second_derivative_from_moments(
    cell: DenestedCubicCell,
    first_coefficients_r: &[f64],
    first_coefficients_s: &[f64],
    second_coefficients_rs: &[f64],
    moments: &[f64],
) -> Result<f64, String> {
    let second_degree = second_coefficients_rs.len().saturating_sub(1);
    let product_degree = first_coefficients_r.len().saturating_sub(1)
        + first_coefficients_s.len().saturating_sub(1)
        + 3;
    let needed = second_degree.max(product_degree) + 1;
    if needed > moments.len() {
        return Err(CubicCellKernelError::insufficient_moments(format!(
            "insufficient reduced moments for second derivative: need {}, have {}",
            needed,
            moments.len()
        ))
        .into());
    }
    let second_term = moment_dot_with_coefficients_unchecked(second_coefficients_rs, moments);
    // Fold `Σ_{e,i,j} eta[e]·r[i]·s[j]·moments[e+i+j]` into a single dot
    // against `moments`. Convolving `eta ⊗ r ⊗ s` first turns the original
    // `len(eta)·len(r)·len(s)` triple loop (typically 4·4·4 = 64 mul-adds
    // per call) into `len(eta)·len(r) + (len(eta)+len(r)-1)·len(s) +
    // len(out)` ≈ 16 + 28 + 10 = 54 mul-adds, with the inner loops now in
    // straight-line FMA-friendly form.
    let cubic = [cell.c0, cell.c1, cell.c2, cell.c3];
    // Capacity bound: cubic (4) + first_r (≤MAX) + first_s (≤MAX) - 2.
    // First-coefficient slices are passed in as `[f64; 4]` from every
    // production caller; sizing to 32 covers any realistic test input.
    const SCRATCH: usize = 32;
    let mut eta_r = [0.0_f64; SCRATCH];
    let mut eta_rs = [0.0_f64; SCRATCH];
    let er_len = poly_conv_into(&cubic, first_coefficients_r, &mut eta_r);
    let ers_len = poly_conv_into(&eta_r[..er_len], first_coefficients_s, &mut eta_rs);
    let mut eta_term = 0.0;
    for k in 0..ers_len {
        eta_term = eta_rs[k].mul_add(moments[k], eta_term);
    }
    Ok((second_term - eta_term) * INV_TWO_PI)
}

#[inline]
fn moment_dot_with_coefficients(
    coefficients: &[f64],
    moments: &[f64],
    label: &str,
) -> Result<f64, String> {
    if coefficients.len() > moments.len() {
        return Err(CubicCellKernelError::insufficient_moments(format!(
            "insufficient reduced moments for {label}: need {}, have {}",
            coefficients.len(),
            moments.len()
        ))
        .into());
    }
    Ok(moment_dot_with_coefficients_unchecked(
        coefficients,
        moments,
    ))
}

#[inline]
fn moment_dot_with_coefficients_unchecked(coefficients: &[f64], moments: &[f64]) -> f64 {
    let mut acc = 0.0;
    for (idx, &coeff) in coefficients.iter().enumerate() {
        acc = coeff.mul_add(moments[idx], acc);
    }
    acc
}

/// Convolve two polynomial coefficient slices into a fixed-capacity output
/// buffer. Returns the populated length (`lhs.len() + rhs.len() - 1` when
/// both are non-empty). The buffer's tail (beyond the returned length) is
/// not zeroed; callers must use only the returned prefix.
///
/// Used by the multi-derivative reductions to fold `eta · r · s · …` triple
/// and quadruple sums into a single moment dot, eliminating the
/// `O(deg^3)`/`O(deg^4)` inner-loop work that dominated the
/// `cell_*_derivative_from_moments` hot leaves on biobank-scale fits.
#[inline]
fn poly_conv_into(lhs: &[f64], rhs: &[f64], out: &mut [f64]) -> usize {
    if lhs.is_empty() || rhs.is_empty() {
        return 0;
    }
    let len = lhs.len() + rhs.len() - 1;
    debug_assert!(out.len() >= len);
    for slot in out[..len].iter_mut() {
        *slot = 0.0;
    }
    for (i, &lv) in lhs.iter().enumerate() {
        for (j, &rv) in rhs.iter().enumerate() {
            out[i + j] = lv.mul_add(rv, out[i + j]);
        }
    }
    len
}

#[inline]
fn require_moments_degree(
    required_degree: usize,
    moments: &[f64],
    label: &str,
) -> Result<(), String> {
    if required_degree >= moments.len() {
        return Err(CubicCellKernelError::insufficient_moments(format!(
            "insufficient reduced moments for {label}: need {}, have {}",
            required_degree + 1,
            moments.len()
        ))
        .into());
    }
    Ok(())
}

#[inline]
fn require_scratch_capacity(
    required_len: usize,
    capacity: usize,
    label: &str,
) -> Result<(), String> {
    if required_len > capacity {
        return Err(CubicCellKernelError::insufficient_moments(format!(
            "{label} polynomial convolution scratch too small: need {required_len}, have {capacity}"
        ))
        .into());
    }
    Ok(())
}

#[inline]
fn convolution_chain_len(lengths: &[usize]) -> usize {
    if lengths.is_empty() || lengths.contains(&0) {
        0
    } else {
        lengths.iter().sum::<usize>() - (lengths.len() - 1)
    }
}

#[inline]
fn first_coefficients_degree(label: &str, coefficients: &[f64]) -> Result<usize, String> {
    coefficients
        .len()
        .checked_sub(1)
        .ok_or_else(|| format!("{label} first-derivative coefficients must be non-empty"))
}

#[inline]
pub fn cell_third_derivative_from_moments(
    cell: DenestedCubicCell,
    first_coefficients_r: &[f64],
    first_coefficients_s: &[f64],
    first_coefficients_t: &[f64],
    second_coefficients_rs: &[f64],
    second_coefficients_rt: &[f64],
    second_coefficients_st: &[f64],
    third_coefficients_rst: &[f64],
    moments: &[f64],
) -> Result<f64, String> {
    let eta = [cell.c0, cell.c1, cell.c2, cell.c3];
    let r_degree = first_coefficients_degree("r", first_coefficients_r)?;
    let s_degree = first_coefficients_degree("s", first_coefficients_s)?;
    let t_degree = first_coefficients_degree("t", first_coefficients_t)?;
    let second_sum_degree = [
        second_coefficients_rs.len() + first_coefficients_t.len(),
        second_coefficients_rt.len() + first_coefficients_s.len(),
        second_coefficients_st.len() + first_coefficients_r.len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0)
    .saturating_sub(1);
    let triple_product_degree = r_degree + s_degree + t_degree;
    let needed = (third_coefficients_rst.len().saturating_sub(1))
        .max(3 + second_sum_degree)
        .max(6 + triple_product_degree);
    require_moments_degree(needed, moments, "third derivative")?;

    let third_term = moment_dot_with_coefficients_unchecked(third_coefficients_rst, moments);

    // This is a deliberately serial leaf kernel: each call performs only a
    // handful of fixed-size polynomial convolutions, so Rayon fan-out belongs
    // at the surrounding row/cell batch level rather than inside this hot path.
    const SCRATCH: usize = 32;
    let max_linear_conv_len = [
        convolution_chain_len(&[
            eta.len(),
            second_coefficients_rs.len(),
            first_coefficients_t.len(),
        ]),
        convolution_chain_len(&[
            eta.len(),
            second_coefficients_rt.len(),
            first_coefficients_s.len(),
        ]),
        convolution_chain_len(&[
            eta.len(),
            second_coefficients_st.len(),
            first_coefficients_r.len(),
        ]),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let max_cubic_conv_len = convolution_chain_len(&[
        7,
        first_coefficients_r.len(),
        first_coefficients_s.len(),
        first_coefficients_t.len(),
    ]);
    require_scratch_capacity(
        max_linear_conv_len.max(max_cubic_conv_len),
        SCRATCH,
        "third derivative",
    )?;
    let mut buf_a = [0.0_f64; SCRATCH];
    let mut buf_b = [0.0_f64; SCRATCH];

    // eta_second_term = Σ over (rs⊗t, rt⊗s, st⊗r) of eta⊗product · moments.
    // Fold each of the three triple sums into a single moment dot.
    let mut eta_second_term = 0.0;
    let conv_dot = |first: &[f64],
                    second: &[f64],
                    buf_a: &mut [f64; SCRATCH],
                    buf_b: &mut [f64; SCRATCH]|
     -> f64 {
        let m = poly_conv_into(first, second, buf_a);
        let n = poly_conv_into(&eta, &buf_a[..m], buf_b);
        let mut acc = 0.0;
        for k in 0..n {
            acc = buf_b[k].mul_add(moments[k], acc);
        }
        acc
    };
    eta_second_term += conv_dot(
        second_coefficients_rs,
        first_coefficients_t,
        &mut buf_a,
        &mut buf_b,
    );
    eta_second_term += conv_dot(
        second_coefficients_rt,
        first_coefficients_s,
        &mut buf_a,
        &mut buf_b,
    );
    eta_second_term += conv_dot(
        second_coefficients_st,
        first_coefficients_r,
        &mut buf_a,
        &mut buf_b,
    );

    // cubic_coeff_term = Σ_{e,i,j,k} (eta·eta − 1)[e] · r[i] · s[j] · t[k] · moments[e+i+j+k].
    // Convolve r⊗s, then ⊗t, then ⊗(eta·eta − 1), giving a single dot.
    let mut eta_sq_minus_one = [0.0_f64; 7];
    for (i, &eta_i) in eta.iter().enumerate() {
        for (j, &eta_j) in eta.iter().enumerate() {
            eta_sq_minus_one[i + j] = eta_i.mul_add(eta_j, eta_sq_minus_one[i + j]);
        }
    }
    eta_sq_minus_one[0] -= 1.0;

    let rs_len = poly_conv_into(first_coefficients_r, first_coefficients_s, &mut buf_a);
    let rst_len = poly_conv_into(&buf_a[..rs_len], first_coefficients_t, &mut buf_b);
    // buf_a now reused for (eta_sq_minus_one ⊗ rst).
    let final_len = poly_conv_into(&eta_sq_minus_one, &buf_b[..rst_len], &mut buf_a);
    let mut cubic_coeff_term = 0.0;
    for k in 0..final_len {
        cubic_coeff_term = buf_a[k].mul_add(moments[k], cubic_coeff_term);
    }

    Ok((third_term - eta_second_term + cubic_coeff_term) * INV_TWO_PI)
}

#[inline]
pub fn cell_fourth_derivative_from_moments(
    cell: DenestedCubicCell,
    first_coefficients_r: &[f64],
    first_coefficients_s: &[f64],
    first_coefficients_t: &[f64],
    first_coefficients_u: &[f64],
    second_coefficients_rs: &[f64],
    second_coefficients_rt: &[f64],
    second_coefficients_ru: &[f64],
    second_coefficients_st: &[f64],
    second_coefficients_su: &[f64],
    second_coefficients_tu: &[f64],
    third_coefficients_rst: &[f64],
    third_coefficients_rsu: &[f64],
    third_coefficients_rtu: &[f64],
    third_coefficients_stu: &[f64],
    fourth_coefficients_rstu: &[f64],
    moments: &[f64],
) -> Result<f64, String> {
    let eta = [cell.c0, cell.c1, cell.c2, cell.c3];
    let r_degree = first_coefficients_degree("r", first_coefficients_r)?;
    let s_degree = first_coefficients_degree("s", first_coefficients_s)?;
    let t_degree = first_coefficients_degree("t", first_coefficients_t)?;
    let u_degree = first_coefficients_degree("u", first_coefficients_u)?;
    let linear_sum_degree = [
        third_coefficients_rst.len() + first_coefficients_u.len(),
        third_coefficients_rsu.len() + first_coefficients_t.len(),
        third_coefficients_rtu.len() + first_coefficients_s.len(),
        third_coefficients_stu.len() + first_coefficients_r.len(),
        second_coefficients_rs.len() + second_coefficients_tu.len(),
        second_coefficients_rt.len() + second_coefficients_su.len(),
        second_coefficients_ru.len() + second_coefficients_st.len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0)
    .saturating_sub(1);
    let quad_sum_degree = [
        second_coefficients_rs.len() + first_coefficients_t.len() + first_coefficients_u.len(),
        second_coefficients_rt.len() + first_coefficients_s.len() + first_coefficients_u.len(),
        second_coefficients_ru.len() + first_coefficients_s.len() + first_coefficients_t.len(),
        second_coefficients_st.len() + first_coefficients_r.len() + first_coefficients_u.len(),
        second_coefficients_su.len() + first_coefficients_r.len() + first_coefficients_t.len(),
        second_coefficients_tu.len() + first_coefficients_r.len() + first_coefficients_s.len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0)
    .saturating_sub(2);
    let quartic_product_degree = r_degree + s_degree + t_degree + u_degree;
    let needed = (fourth_coefficients_rstu.len().saturating_sub(1))
        .max(3 + linear_sum_degree)
        .max(6 + quad_sum_degree)
        .max(9 + quartic_product_degree);
    require_moments_degree(needed, moments, "fourth derivative")?;

    let fourth_term = moment_dot_with_coefficients_unchecked(fourth_coefficients_rstu, moments);

    // This is a deliberately serial leaf kernel: each call performs only a
    // handful of fixed-size polynomial convolutions, so Rayon fan-out belongs
    // at the surrounding row/cell batch level rather than inside this hot path.
    const SCRATCH: usize = 32;
    let max_linear_conv_len = [
        convolution_chain_len(&[
            eta.len(),
            third_coefficients_rst.len(),
            first_coefficients_u.len(),
        ]),
        convolution_chain_len(&[
            eta.len(),
            third_coefficients_rsu.len(),
            first_coefficients_t.len(),
        ]),
        convolution_chain_len(&[
            eta.len(),
            third_coefficients_rtu.len(),
            first_coefficients_s.len(),
        ]),
        convolution_chain_len(&[
            eta.len(),
            third_coefficients_stu.len(),
            first_coefficients_r.len(),
        ]),
        convolution_chain_len(&[
            eta.len(),
            second_coefficients_rs.len(),
            second_coefficients_tu.len(),
        ]),
        convolution_chain_len(&[
            eta.len(),
            second_coefficients_rt.len(),
            second_coefficients_su.len(),
        ]),
        convolution_chain_len(&[
            eta.len(),
            second_coefficients_ru.len(),
            second_coefficients_st.len(),
        ]),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let max_quad_conv_len = [
        convolution_chain_len(&[
            7,
            second_coefficients_rs.len(),
            first_coefficients_t.len(),
            first_coefficients_u.len(),
        ]),
        convolution_chain_len(&[
            7,
            second_coefficients_rt.len(),
            first_coefficients_s.len(),
            first_coefficients_u.len(),
        ]),
        convolution_chain_len(&[
            7,
            second_coefficients_ru.len(),
            first_coefficients_s.len(),
            first_coefficients_t.len(),
        ]),
        convolution_chain_len(&[
            7,
            second_coefficients_st.len(),
            first_coefficients_r.len(),
            first_coefficients_u.len(),
        ]),
        convolution_chain_len(&[
            7,
            second_coefficients_su.len(),
            first_coefficients_r.len(),
            first_coefficients_t.len(),
        ]),
        convolution_chain_len(&[
            7,
            second_coefficients_tu.len(),
            first_coefficients_r.len(),
            first_coefficients_s.len(),
        ]),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let max_quartic_conv_len = convolution_chain_len(&[
        10,
        first_coefficients_r.len(),
        first_coefficients_s.len(),
        first_coefficients_t.len(),
        first_coefficients_u.len(),
    ]);
    require_scratch_capacity(
        max_linear_conv_len
            .max(max_quad_conv_len)
            .max(max_quartic_conv_len),
        SCRATCH,
        "fourth derivative",
    )?;
    let mut buf_a = [0.0_f64; SCRATCH];
    let mut buf_b = [0.0_f64; SCRATCH];

    // eta_linear_term = Σ over seven (rst⊗u, rsu⊗t, rtu⊗s, stu⊗r, rs⊗tu,
    // rt⊗su, ru⊗st) of eta⊗product · moments. Fold each triple sum into
    // a single moment dot.
    let conv_eta_dot = |first: &[f64],
                        second: &[f64],
                        buf_a: &mut [f64; SCRATCH],
                        buf_b: &mut [f64; SCRATCH]|
     -> f64 {
        let m = poly_conv_into(first, second, buf_a);
        let n = poly_conv_into(&eta, &buf_a[..m], buf_b);
        let mut acc = 0.0;
        for k in 0..n {
            acc = buf_b[k].mul_add(moments[k], acc);
        }
        acc
    };
    let mut eta_linear_term = 0.0;
    eta_linear_term += conv_eta_dot(
        third_coefficients_rst,
        first_coefficients_u,
        &mut buf_a,
        &mut buf_b,
    );
    eta_linear_term += conv_eta_dot(
        third_coefficients_rsu,
        first_coefficients_t,
        &mut buf_a,
        &mut buf_b,
    );
    eta_linear_term += conv_eta_dot(
        third_coefficients_rtu,
        first_coefficients_s,
        &mut buf_a,
        &mut buf_b,
    );
    eta_linear_term += conv_eta_dot(
        third_coefficients_stu,
        first_coefficients_r,
        &mut buf_a,
        &mut buf_b,
    );
    eta_linear_term += conv_eta_dot(
        second_coefficients_rs,
        second_coefficients_tu,
        &mut buf_a,
        &mut buf_b,
    );
    eta_linear_term += conv_eta_dot(
        second_coefficients_rt,
        second_coefficients_su,
        &mut buf_a,
        &mut buf_b,
    );
    eta_linear_term += conv_eta_dot(
        second_coefficients_ru,
        second_coefficients_st,
        &mut buf_a,
        &mut buf_b,
    );

    let mut eta_sq_minus_one = [0.0_f64; 7];
    for (i, &eta_i) in eta.iter().enumerate() {
        for (j, &eta_j) in eta.iter().enumerate() {
            eta_sq_minus_one[i + j] = eta_i.mul_add(eta_j, eta_sq_minus_one[i + j]);
        }
    }
    eta_sq_minus_one[0] -= 1.0;

    // quad_coeff_term: six (eta²−1)⊗A⊗B⊗C · moments sums, where the (A,B,C)
    // factors are: (rs,t,u), (rt,s,u), (ru,s,t), (st,r,u), (su,r,t), (tu,r,s).
    let mut buf_c = [0.0_f64; SCRATCH];
    let conv_weighted_triple_dot = |weight: &[f64],
                                    a: &[f64],
                                    b: &[f64],
                                    c: &[f64],
                                    buf_a: &mut [f64; SCRATCH],
                                    buf_b: &mut [f64; SCRATCH],
                                    buf_c: &mut [f64; SCRATCH]|
     -> f64 {
        let ab_len = poly_conv_into(a, b, buf_a);
        let abc_len = poly_conv_into(&buf_a[..ab_len], c, buf_b);
        let final_len = poly_conv_into(weight, &buf_b[..abc_len], buf_c);
        let mut acc = 0.0;
        for k in 0..final_len {
            acc = buf_c[k].mul_add(moments[k], acc);
        }
        acc
    };
    let mut quad_coeff_term = 0.0;
    quad_coeff_term += conv_weighted_triple_dot(
        &eta_sq_minus_one,
        second_coefficients_rs,
        first_coefficients_t,
        first_coefficients_u,
        &mut buf_a,
        &mut buf_b,
        &mut buf_c,
    );
    quad_coeff_term += conv_weighted_triple_dot(
        &eta_sq_minus_one,
        second_coefficients_rt,
        first_coefficients_s,
        first_coefficients_u,
        &mut buf_a,
        &mut buf_b,
        &mut buf_c,
    );
    quad_coeff_term += conv_weighted_triple_dot(
        &eta_sq_minus_one,
        second_coefficients_ru,
        first_coefficients_s,
        first_coefficients_t,
        &mut buf_a,
        &mut buf_b,
        &mut buf_c,
    );
    quad_coeff_term += conv_weighted_triple_dot(
        &eta_sq_minus_one,
        second_coefficients_st,
        first_coefficients_r,
        first_coefficients_u,
        &mut buf_a,
        &mut buf_b,
        &mut buf_c,
    );
    quad_coeff_term += conv_weighted_triple_dot(
        &eta_sq_minus_one,
        second_coefficients_su,
        first_coefficients_r,
        first_coefficients_t,
        &mut buf_a,
        &mut buf_b,
        &mut buf_c,
    );
    quad_coeff_term += conv_weighted_triple_dot(
        &eta_sq_minus_one,
        second_coefficients_tu,
        first_coefficients_r,
        first_coefficients_s,
        &mut buf_a,
        &mut buf_b,
        &mut buf_c,
    );

    // cubic_weight = 3·eta − eta³ (same as the prior expansion: eta_sq*eta
    // negated, plus the 3·eta linear correction).
    let mut eta_sq = [0.0_f64; 7];
    for (i, &eta_i) in eta.iter().enumerate() {
        for (j, &eta_j) in eta.iter().enumerate() {
            eta_sq[i + j] = eta_i.mul_add(eta_j, eta_sq[i + j]);
        }
    }
    let mut cubic_weight = [0.0_f64; 10];
    for (i, &eta_sq_i) in eta_sq.iter().enumerate() {
        for (j, &eta_j) in eta.iter().enumerate() {
            cubic_weight[i + j] = (-eta_sq_i).mul_add(eta_j, cubic_weight[i + j]);
        }
    }
    for (idx, &eta_coeff) in eta.iter().enumerate() {
        cubic_weight[idx] += 3.0 * eta_coeff;
    }

    // quartic_coeff_term: cubic_weight ⊗ r ⊗ s ⊗ t ⊗ u · moments. The
    // original quintuple loop did 10·4·4·4·4 = 2560 mul-adds per call;
    // four sequential convolutions plus one moment dot drop this to
    // ~16+28+40+52+16 ≈ 152 mul-adds.
    let rs_len = poly_conv_into(first_coefficients_r, first_coefficients_s, &mut buf_a);
    let rst_len = poly_conv_into(&buf_a[..rs_len], first_coefficients_t, &mut buf_b);
    let rstu_len = poly_conv_into(&buf_b[..rst_len], first_coefficients_u, &mut buf_a);
    let final_len = poly_conv_into(&cubic_weight, &buf_a[..rstu_len], &mut buf_b);
    let mut quartic_coeff_term = 0.0;
    for k in 0..final_len {
        quartic_coeff_term = buf_b[k].mul_add(moments[k], quartic_coeff_term);
    }

    Ok((fourth_term - eta_linear_term + quad_coeff_term + quartic_coeff_term) * INV_TWO_PI)
}

#[inline]
pub fn global_cubic_from_local(span: LocalSpanCubic) -> (f64, f64, f64, f64) {
    let left = span.left;
    let q0 = span.c0 - span.c1 * left + span.c2 * left * left - span.c3 * left * left * left;
    let q1 = span.c1 - 2.0 * span.c2 * left + 3.0 * span.c3 * left * left;
    let q2 = span.c2 - 3.0 * span.c3 * left;
    let q3 = span.c3;
    (q0, q1, q2, q3)
}

#[inline]
pub fn transformed_link_cubic(link_span: LocalSpanCubic, a: f64, b: f64) -> (f64, f64, f64, f64) {
    let shift = a - link_span.left;
    let d0 = link_span.c0
        + link_span.c1 * shift
        + link_span.c2 * shift * shift
        + link_span.c3 * shift * shift * shift;
    let d1 = b * (link_span.c1 + 2.0 * link_span.c2 * shift + 3.0 * link_span.c3 * shift * shift);
    let d2 = b * b * (link_span.c2 + 3.0 * link_span.c3 * shift);
    let d3 = link_span.c3 * b * b * b;
    (d0, d1, d2, d3)
}

#[inline]
pub fn denested_cell_coefficients(
    score_span: LocalSpanCubic,
    link_span: LocalSpanCubic,
    a: f64,
    b: f64,
) -> [f64; 4] {
    let (h0, h1, h2, h3) = global_cubic_from_local(score_span);
    let (d0, d1, d2, d3) = transformed_link_cubic(link_span, a, b);
    [a + b * h0 + d0, b + b * h1 + d1, b * h2 + d2, b * h3 + d3]
}

#[inline]
pub fn denested_cell_coefficient_partials(
    score_span: LocalSpanCubic,
    link_span: LocalSpanCubic,
    a: f64,
    b: f64,
) -> ([f64; 4], [f64; 4]) {
    let (h0, h1, h2, h3) = global_cubic_from_local(score_span);
    let shift = a - link_span.left;
    let alpha1 = link_span.c1;
    let alpha2 = link_span.c2;
    let alpha3 = link_span.c3;
    let dc_da = [
        1.0 + alpha1 + 2.0 * alpha2 * shift + 3.0 * alpha3 * shift * shift,
        b * (2.0 * alpha2 + 6.0 * alpha3 * shift),
        3.0 * alpha3 * b * b,
        0.0,
    ];
    let dc_db = [
        h0,
        1.0 + h1 + alpha1 + 2.0 * alpha2 * shift + 3.0 * alpha3 * shift * shift,
        h2 + 2.0 * b * (alpha2 + 3.0 * alpha3 * shift),
        h3 + 3.0 * alpha3 * b * b,
    ];
    (dc_da, dc_db)
}

#[inline]
fn link_cubic_second_partials(
    link_span: LocalSpanCubic,
    a: f64,
    b: f64,
) -> ([f64; 4], [f64; 4], [f64; 4]) {
    let shift = a - link_span.left;
    let alpha2 = link_span.c2;
    let alpha3 = link_span.c3;
    let dc_daa = [
        2.0 * alpha2 + 6.0 * alpha3 * shift,
        6.0 * alpha3 * b,
        0.0,
        0.0,
    ];
    let dc_dab = [
        0.0,
        2.0 * alpha2 + 6.0 * alpha3 * shift,
        6.0 * alpha3 * b,
        0.0,
    ];
    let dc_dbb = [
        0.0,
        0.0,
        2.0 * (alpha2 + 3.0 * alpha3 * shift),
        6.0 * alpha3 * b,
    ];
    (dc_daa, dc_dab, dc_dbb)
}

#[inline]
pub fn denested_cell_second_partials(
    score_span: LocalSpanCubic,
    link_span: LocalSpanCubic,
    a: f64,
    b: f64,
) -> ([f64; 4], [f64; 4], [f64; 4]) {
    let score_left = score_span.left;
    if !score_left.is_finite() {
        return ([f64::NAN; 4], [f64::NAN; 4], [f64::NAN; 4]);
    }
    link_cubic_second_partials(link_span, a, b)
}

#[inline]
fn link_cubic_third_partials(
    link_span: LocalSpanCubic,
) -> ([f64; 4], [f64; 4], [f64; 4], [f64; 4]) {
    let alpha3 = link_span.c3;
    (
        [6.0 * alpha3, 0.0, 0.0, 0.0],
        [0.0, 6.0 * alpha3, 0.0, 0.0],
        [0.0, 0.0, 6.0 * alpha3, 0.0],
        [0.0, 0.0, 0.0, 6.0 * alpha3],
    )
}

#[inline]
pub fn denested_cell_third_partials(
    link_span: LocalSpanCubic,
) -> ([f64; 4], [f64; 4], [f64; 4], [f64; 4]) {
    link_cubic_third_partials(link_span)
}

#[inline]
pub fn score_basis_cell_coefficients(score_basis_span: LocalSpanCubic, b: f64) -> [f64; 4] {
    let (h0, h1, h2, h3) = global_cubic_from_local(score_basis_span);
    [b * h0, b * h1, b * h2, b * h3]
}

#[inline]
pub fn link_basis_cell_coefficients(link_basis_span: LocalSpanCubic, a: f64, b: f64) -> [f64; 4] {
    let (d0, d1, d2, d3) = transformed_link_cubic(link_basis_span, a, b);
    [d0, d1, d2, d3]
}

#[inline]
pub fn link_basis_cell_coefficient_partials(
    link_basis_span: LocalSpanCubic,
    a: f64,
    b: f64,
) -> ([f64; 4], [f64; 4]) {
    let shift = a - link_basis_span.left;
    let alpha1 = link_basis_span.c1;
    let alpha2 = link_basis_span.c2;
    let alpha3 = link_basis_span.c3;
    let dc_da = [
        alpha1 + 2.0 * alpha2 * shift + 3.0 * alpha3 * shift * shift,
        b * (2.0 * alpha2 + 6.0 * alpha3 * shift),
        3.0 * alpha3 * b * b,
        0.0,
    ];
    let dc_db = [
        0.0,
        alpha1 + 2.0 * alpha2 * shift + 3.0 * alpha3 * shift * shift,
        2.0 * b * (alpha2 + 3.0 * alpha3 * shift),
        3.0 * alpha3 * b * b,
    ];
    (dc_da, dc_db)
}

#[inline]
pub fn link_basis_cell_second_partials(
    link_basis_span: LocalSpanCubic,
    a: f64,
    b: f64,
) -> ([f64; 4], [f64; 4], [f64; 4]) {
    link_cubic_second_partials(link_basis_span, a, b)
}

#[inline]
pub fn link_basis_cell_third_partials(
    link_basis_span: LocalSpanCubic,
) -> ([f64; 4], [f64; 4], [f64; 4], [f64; 4]) {
    link_cubic_third_partials(link_basis_span)
}

pub fn build_denested_partition_cells<FS, FL>(
    a: f64,
    b: f64,
    score_breaks: &[f64],
    link_breaks: &[f64],
    score_span_at: FS,
    link_span_at: FL,
) -> Result<Vec<DenestedPartitionCell>, String>
where
    FS: FnMut(f64) -> Result<LocalSpanCubic, String>,
    FL: FnMut(f64) -> Result<LocalSpanCubic, String>,
{
    build_denested_partition_cells_with_tails(
        a,
        b,
        score_breaks,
        link_breaks,
        score_span_at,
        link_span_at,
    )
}

/// Build a partition covering `(-∞, +∞)` with parameter-independent outer
/// bounds.  Interior cells use the same finite-cell polynomial algebra.
/// The two tail cells are guaranteed affine (c2=c3=0) because both
/// deviations saturate to constants outside their knot support.
///
/// The tail cells' score/link spans come from the same closures evaluated
/// at a representative point in the tail region — the closures must return
/// constant (c1=c2=c3=0) cubics for points outside support.
pub fn build_denested_partition_cells_with_tails<FS, FL>(
    a: f64,
    b: f64,
    score_breaks: &[f64],
    link_breaks: &[f64],
    mut score_span_at: FS,
    mut link_span_at: FL,
) -> Result<Vec<DenestedPartitionCell>, String>
where
    FS: FnMut(f64) -> Result<LocalSpanCubic, String>,
    FL: FnMut(f64) -> Result<LocalSpanCubic, String>,
{
    // Collect all INTERNAL split points (finite).
    let mut split_points = score_breaks.to_vec();
    if b.abs() > 1e-12 {
        for &tau in link_breaks {
            let z = (tau - a) / b;
            if z.is_finite() {
                split_points.push(z);
            }
        }
    }
    dedup_sorted_breakpoints(&mut split_points);

    let mut out = Vec::new();

    if split_points.is_empty() {
        let score_span = score_span_at(0.0)?;
        let link_span = link_span_at(a)?;
        let coeffs = denested_cell_coefficients(score_span, link_span, a, b);
        return Ok(vec![DenestedPartitionCell {
            cell: DenestedCubicCell {
                left: f64::NEG_INFINITY,
                right: f64::INFINITY,
                c0: coeffs[0],
                c1: coeffs[1],
                c2: 0.0,
                c3: 0.0,
            },
            score_span,
            link_span,
        }]);
    }

    // ── Left tail cell: (-∞, leftmost_split] ──
    let leftmost = split_points[0];
    // Evaluate spans at a point just left of the leftmost split.  The
    // closures return constant tail cubics for this region.
    let left_probe = interval_probe_point(f64::NEG_INFINITY, leftmost)?;
    let left_score_span = score_span_at(left_probe)?;
    let left_link_span = link_span_at(a + b * left_probe)?;
    let left_coeffs = denested_cell_coefficients(left_score_span, left_link_span, a, b);
    if left_coeffs[2].abs() > NORMALIZED_CELL_BRANCH_TOL
        || left_coeffs[3].abs() > NORMALIZED_CELL_BRANCH_TOL
    {
        return Err(CubicCellKernelError::invalid_cell_shape(format!(
            "left tail cell must be affine (deviations constant outside support), \
             got c2={:.3e}, c3={:.3e}",
            left_coeffs[2], left_coeffs[3]
        ))
        .into());
    }
    out.push(DenestedPartitionCell {
        cell: DenestedCubicCell {
            left: f64::NEG_INFINITY,
            right: leftmost,
            c0: left_coeffs[0],
            c1: left_coeffs[1],
            c2: 0.0,
            c3: 0.0,
        },
        score_span: left_score_span,
        link_span: left_link_span,
    });

    // ── Interior cells (all finite) ──
    for window in split_points.windows(2) {
        let left = window[0];
        let right = window[1];
        if !left.is_finite() || !right.is_finite() || right - left <= 1e-12 {
            continue;
        }
        let mid = interval_probe_point(left, right)?;
        let score_span = score_span_at(mid)?;
        let link_span = link_span_at(a + b * mid)?;
        let coeffs = denested_cell_coefficients(score_span, link_span, a, b);
        out.push(DenestedPartitionCell {
            cell: DenestedCubicCell {
                left,
                right,
                c0: coeffs[0],
                c1: coeffs[1],
                c2: coeffs[2],
                c3: coeffs[3],
            },
            score_span,
            link_span,
        });
    }

    // ── Right tail cell: [rightmost_split, +∞) ──
    let rightmost = *split_points.last().unwrap();
    let right_probe = interval_probe_point(rightmost, f64::INFINITY)?;
    let right_score_span = score_span_at(right_probe)?;
    let right_link_span = link_span_at(a + b * right_probe)?;
    let right_coeffs = denested_cell_coefficients(right_score_span, right_link_span, a, b);
    if right_coeffs[2].abs() > NORMALIZED_CELL_BRANCH_TOL
        || right_coeffs[3].abs() > NORMALIZED_CELL_BRANCH_TOL
    {
        return Err(CubicCellKernelError::invalid_cell_shape(format!(
            "right tail cell must be affine (deviations constant outside support), \
             got c2={:.3e}, c3={:.3e}",
            right_coeffs[2], right_coeffs[3]
        ))
        .into());
    }
    out.push(DenestedPartitionCell {
        cell: DenestedCubicCell {
            left: rightmost,
            right: f64::INFINITY,
            c0: right_coeffs[0],
            c1: right_coeffs[1],
            c2: 0.0,
            c3: 0.0,
        },
        score_span: right_score_span,
        link_span: right_link_span,
    });

    Ok(out)
}

#[inline]
pub fn normalized_non_affine_coefficients(
    left: f64,
    right: f64,
    c0: f64,
    c1: f64,
    c2: f64,
    c3: f64,
) -> Result<(f64, f64), String> {
    let width = right - left;
    if !width.is_finite() || width <= 0.0 {
        return Err(CubicCellKernelError::invalid_cell_shape(format!(
            "normalized cubic coefficients require a positive finite cell width, got left={left}, right={right}"
        ))
        .into());
    }
    let anchor_scale = c0.abs() + c1.abs();
    if !anchor_scale.is_finite() {
        return Err(CubicCellKernelError::invalid_cell_shape(format!(
            "normalized cubic coefficients require finite affine coefficients, got c0={c0}, c1={c1}"
        ))
        .into());
    }
    let mid = 0.5 * (left + right);
    let half = 0.5 * width;
    let k2 = half * half * (c2 + 3.0 * c3 * mid);
    let k3 = c3 * half * half * half;
    Ok((k2, k3))
}

#[inline]
pub fn branch_cell(cell: DenestedCubicCell) -> Result<ExactCellBranch, String> {
    let tol = effective_branch_tol(cell);
    if !cell.left.is_finite() || !cell.right.is_finite() {
        if cell.c2.abs() <= tol && cell.c3.abs() <= tol {
            return Ok(ExactCellBranch::Affine);
        }
        return Err(CubicCellKernelError::invalid_cell_shape(format!(
            "non-affine cells require finite bounds, got [{}, {}] with c2={:.6e}, c3={:.6e}",
            cell.left, cell.right, cell.c2, cell.c3
        ))
        .into());
    }
    let (k2, k3) = normalized_non_affine_coefficients(
        cell.left, cell.right, cell.c0, cell.c1, cell.c2, cell.c3,
    )?;
    if k2.abs() <= tol && k3.abs() <= tol {
        Ok(ExactCellBranch::Affine)
    } else if k3.abs() <= tol {
        Ok(ExactCellBranch::Quartic)
    } else {
        Ok(ExactCellBranch::Sextic)
    }
}

#[inline]
fn degenerate_sextic_branch(
    cell: DenestedCubicCell,
    lead: f64,
) -> Result<Option<ExactCellBranch>, String> {
    // The sextic recurrence divides by `lead = 3*c3^2`. When that division is
    // unstable, lower the polynomial degree without discarding a material
    // quadratic coefficient.
    let (normalized_k2, normalized_k3) = normalized_non_affine_coefficients(
        cell.left, cell.right, cell.c0, cell.c1, cell.c2, cell.c3,
    )?;
    if normalized_k3.abs() > NORMALIZED_CELL_BRANCH_TOL && lead.abs() > 1e-18 {
        return Ok(None);
    }
    if normalized_k2.abs() > NORMALIZED_CELL_BRANCH_TOL {
        Ok(Some(ExactCellBranch::Quartic))
    } else {
        Ok(Some(ExactCellBranch::Affine))
    }
}

#[inline]
fn validate_bvn_args(h: f64, k: f64, rho: f64) -> Result<(), String> {
    if !h.is_finite() && !h.is_infinite() {
        return Err(CubicCellKernelError::bivariate_normal_domain(
            "bivariate normal cdf requires finite or infinite h",
        )
        .into());
    }
    if !k.is_finite() && !k.is_infinite() {
        return Err(CubicCellKernelError::bivariate_normal_domain(
            "bivariate normal cdf requires finite or infinite k",
        )
        .into());
    }
    if !rho.is_finite() {
        return Err(CubicCellKernelError::bivariate_normal_domain(format!(
            "bivariate normal cdf requires finite correlation, got {rho}"
        ))
        .into());
    }
    Ok(())
}

#[inline]
fn bvn_gl_sum(h: f64, k: f64, rho_clamped: f64, asr: f64) -> f64 {
    // The Drezner-Wesolowsky arcsin representation is integrated with the
    // same 20-point Gauss-Legendre rule as before, but mirrored node pairs are
    // evaluated with one sin_cos for the half-angle offset rather than two
    // independent sin calls.  This preserves the quadrature rule (and hence
    // the accuracy envelope) while reducing the transcendental work in the
    // dominant finite-bound path from 20 sin calls to 11 sin/cos evaluations.
    if rho_clamped == 0.0 {
        return 0.0;
    }
    let hs = 0.5 * (h * h + k * k);
    let hk = h * k;
    let half_asr = 0.5 * asr;
    let (sin_mid, cos_mid) = half_asr.sin_cos();
    let mut sum = 0.0;
    for i in 0..10 {
        let node = BVN_GL_NODES_20[i].abs();
        let weight = BVN_GL_WEIGHTS_20[i];
        let (sin_delta, cos_delta) = (half_asr * node).sin_cos();

        let sn_lo = sin_mid * cos_delta - cos_mid * sin_delta;
        let one_minus_lo = 1.0 - sn_lo * sn_lo;
        let expo_lo = ((sn_lo * hk) - hs) / one_minus_lo;

        let sn_hi = sin_mid * cos_delta + cos_mid * sin_delta;
        let one_minus_hi = 1.0 - sn_hi * sn_hi;
        let expo_hi = ((sn_hi * hk) - hs) / one_minus_hi;

        sum += weight * (expo_lo.exp() + expo_hi.exp());
    }
    sum
}

pub fn bivariate_normal_cdf(h: f64, k: f64, rho: f64) -> Result<f64, String> {
    validate_bvn_args(h, k, rho)?;
    if h == f64::NEG_INFINITY || k == f64::NEG_INFINITY {
        return Ok(0.0);
    }
    if h == f64::INFINITY {
        return Ok(normal_cdf(k));
    }
    if k == f64::INFINITY {
        return Ok(normal_cdf(h));
    }

    let rho_clamped = rho.clamp(-1.0, 1.0);
    if rho_clamped >= 1.0 - 1e-12 {
        return Ok(normal_cdf(h.min(k)));
    }
    if rho_clamped <= -1.0 + 1e-12 {
        return Ok((normal_cdf(h) - normal_cdf(-k)).clamp(0.0, 1.0));
    }
    if rho_clamped == 0.0 {
        return Ok((normal_cdf(h) * normal_cdf(k)).clamp(0.0, 1.0));
    }
    if h == 0.0 && k == 0.0 {
        return Ok((0.25 + rho_clamped.asin() / std::f64::consts::TAU).clamp(0.0, 1.0));
    }

    let asr = rho_clamped.asin();
    let sum = bvn_gl_sum(h, k, rho_clamped, asr);
    Ok((normal_cdf(h) * normal_cdf(k) + asr * sum / (4.0 * std::f64::consts::PI)).clamp(0.0, 1.0))
}

#[inline]
fn bvn_gl_sum_interval(h: f64, left: f64, right: f64, rho_clamped: f64, asr: f64) -> f64 {
    if rho_clamped == 0.0 {
        return 0.0;
    }
    let h2 = h * h;
    let right_hs = 0.5 * (h2 + right * right);
    let left_hs = 0.5 * (h2 + left * left);
    let half_asr = 0.5 * asr;
    let (sin_mid, cos_mid) = half_asr.sin_cos();
    let mut sum = 0.0;
    for i in 0..10 {
        let node = BVN_GL_NODES_20[i].abs();
        let weight = BVN_GL_WEIGHTS_20[i];
        let (sin_delta, cos_delta) = (half_asr * node).sin_cos();

        let sn_lo = sin_mid * cos_delta - cos_mid * sin_delta;
        let one_minus_lo = 1.0 - sn_lo * sn_lo;
        let lo_right = (((sn_lo * h * right) - right_hs) / one_minus_lo).exp();
        let lo_left = (((sn_lo * h * left) - left_hs) / one_minus_lo).exp();

        let sn_hi = sin_mid * cos_delta + cos_mid * sin_delta;
        let one_minus_hi = 1.0 - sn_hi * sn_hi;
        let hi_right = (((sn_hi * h * right) - right_hs) / one_minus_hi).exp();
        let hi_left = (((sn_hi * h * left) - left_hs) / one_minus_hi).exp();

        sum += weight * ((lo_right - lo_left) + (hi_right - hi_left));
    }
    sum
}

fn bivariate_normal_cdf_interval(h: f64, left: f64, right: f64, rho: f64) -> Result<f64, String> {
    if right <= left {
        return Ok(0.0);
    }
    if left == f64::NEG_INFINITY && right == f64::INFINITY {
        return Ok(normal_cdf(h));
    }
    if !left.is_finite() || !right.is_finite() {
        let upper = bivariate_normal_cdf(h, right, rho)?;
        let lower = bivariate_normal_cdf(h, left, rho)?;
        return Ok((upper - lower).clamp(0.0, 1.0));
    }
    validate_bvn_args(h, left, rho)?;
    validate_bvn_args(h, right, rho)?;
    if h == f64::NEG_INFINITY {
        return Ok(0.0);
    }
    if h == f64::INFINITY {
        return Ok((normal_cdf(right) - normal_cdf(left)).clamp(0.0, 1.0));
    }

    let rho_clamped = rho.clamp(-1.0, 1.0);
    if rho_clamped >= 1.0 - 1e-12 || rho_clamped <= -1.0 + 1e-12 {
        let upper = bivariate_normal_cdf(h, right, rho_clamped)?;
        let lower = bivariate_normal_cdf(h, left, rho_clamped)?;
        return Ok((upper - lower).clamp(0.0, 1.0));
    }

    let cdf_h = normal_cdf(h);
    let normal_part = cdf_h * (normal_cdf(right) - normal_cdf(left));
    if rho_clamped == 0.0 {
        return Ok(normal_part.clamp(0.0, 1.0));
    }
    let asr = rho_clamped.asin();
    let sum = bvn_gl_sum_interval(h, left, right, rho_clamped, asr);
    Ok((normal_part + asr * sum / (4.0 * std::f64::consts::PI)).clamp(0.0, 1.0))
}

fn exp_neg_half_square(x: f64) -> f64 {
    if x.is_infinite() {
        0.0
    } else {
        (-0.5 * x * x).exp()
    }
}

/// Fill `out[0..=max_degree]` with the raw truncated standard-normal moments
///
/// ```text
/// T_n(a, b) = ∫_a^b z^n exp(-z²/2) dz
/// ```
///
/// using the integration-by-parts recurrence
///
/// ```text
/// T_0(a, b) = √(2π) (Φ(b) − Φ(a))
/// T_1(a, b) = exp(−a²/2) − exp(−b²/2)
/// T_n(a, b) = a^(n−1) e^{−a²/2} − b^(n−1) e^{−b²/2} + (n−1) T_{n−2}(a, b)
/// ```
///
/// Computed in one forward sweep so each call evaluates `erf` and
/// `exp(−x²/2)` exactly twice (once at `a`, once at `b`) regardless of the
/// requested degree. The naive form — calling `T_n` recursively for each
/// `n = 0..=max_degree` — re-evaluated `erf`/`exp` about `max_degree²/4`
/// times per affine cell, which dominated the wall time of the
/// transformation-normal and bernoulli-marginal-slope inner solves with
/// `max_degree = 64` (the transport order's required degree budget).
fn fill_truncated_gaussian_moments(a: f64, b: f64, out: &mut [f64]) {
    if out.is_empty() {
        return;
    }
    let cdf = |x: f64| -> f64 {
        if x.is_infinite() {
            if x.is_sign_positive() { 1.0 } else { 0.0 }
        } else {
            0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2))
        }
    };
    out[0] = (2.0 * std::f64::consts::PI).sqrt() * (cdf(b) - cdf(a));
    if out.len() == 1 {
        return;
    }
    let ea = exp_neg_half_square(a);
    let eb = exp_neg_half_square(b);
    out[1] = ea - eb;
    if out.len() == 2 {
        return;
    }
    let a_finite = a.is_finite();
    let b_finite = b.is_finite();
    // For n in 2..=max_degree we need a^{n-1} e^{-a²/2} (resp. b). Carry the
    // running powers a^{n-1}, b^{n-1} forward by a single multiply per step.
    // Infinite endpoints contribute 0 (the integrand decays at the rate of
    // exp(−x²/2)), matching the prior `is_infinite` branch in the recursive
    // implementation; we still update the running power so the iteration
    // stays branchless when both endpoints are finite.
    let mut a_pow_n_minus_1 = a; // a^1, used at n = 2
    let mut b_pow_n_minus_1 = b;
    for n in 2..out.len() {
        let left = if a_finite { a_pow_n_minus_1 * ea } else { 0.0 };
        let right = if b_finite { b_pow_n_minus_1 * eb } else { 0.0 };
        out[n] = left - right + (n as f64 - 1.0) * out[n - 2];
        a_pow_n_minus_1 *= a;
        b_pow_n_minus_1 *= b;
    }
}

/// Stack-array bound for `affine_anchor_moment_vector_into`. Public callers
/// use up to ~24 (largest is the bernoulli-margslope outer-step degree-21
/// reduction); 64 leaves comfortable headroom without growing the per-call
/// stack footprint meaningfully.
const MAX_AFFINE_ANCHOR_DEGREE: usize = 64;

pub fn affine_anchor_moment_vector(
    alpha: f64,
    beta: f64,
    left: f64,
    right: f64,
    max_degree: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; max_degree + 1];
    affine_anchor_moment_vector_into(alpha, beta, left, right, max_degree, &mut out);
    out
}

fn affine_anchor_moment_vector_into(
    alpha: f64,
    beta: f64,
    left: f64,
    right: f64,
    max_degree: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(out.len(), max_degree + 1);
    let s = (1.0 + beta * beta).sqrt();
    let mu = -alpha * beta / (1.0 + beta * beta);
    let y_left = if left.is_infinite() {
        if left.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        s * (left - mu)
    };
    let y_right = if right.is_infinite() {
        if right.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        s * (right - mu)
    };
    let anchor = (-alpha * alpha / (2.0 * s * s)).exp() / s;
    debug_assert!(
        max_degree <= MAX_AFFINE_ANCHOR_DEGREE,
        "affine_anchor_moment_vector max_degree {} exceeds compile-time bound {}",
        max_degree,
        MAX_AFFINE_ANCHOR_DEGREE
    );
    let mut t = [0.0_f64; MAX_AFFINE_ANCHOR_DEGREE + 1];
    fill_truncated_gaussian_moments(y_left, y_right, &mut t[..=max_degree]);
    // Build mu^k and s^{-k} tables once. The inner sum is the binomial
    // expansion of the affine change-of-variables, and computing the
    // binomial coefficient via Pascal's row recurrence + carrying mu/s
    // powers eliminates the per-(n, k) `powi` and binomial calls that
    // otherwise dominated the inner loop at large `max_degree`.
    let mut mu_pow = [1.0_f64; MAX_AFFINE_ANCHOR_DEGREE + 1];
    for k in 1..=max_degree {
        mu_pow[k] = mu_pow[k - 1] * mu;
    }
    let inv_s = 1.0 / s;
    let mut inv_s_pow = [1.0_f64; MAX_AFFINE_ANCHOR_DEGREE + 1];
    for k in 1..=max_degree {
        inv_s_pow[k] = inv_s_pow[k - 1] * inv_s;
    }
    out.fill(0.0);
    for n in 0..=max_degree {
        let mut acc = 0.0;
        // C(n, k+1) = C(n, k) · (n − k) / (k + 1).
        let mut binom = 1.0;
        for k in 0..=n {
            let term = binom * mu_pow[n - k] * inv_s_pow[k];
            acc = term.mul_add(t[k], acc);
            if k < n {
                binom = binom * (n - k) as f64 / (k + 1) as f64;
            }
        }
        out[n] = anchor * acc;
    }
}

fn affine_value_from_moment_primitive(alpha: f64, beta: f64, left: f64, right: f64) -> f64 {
    // Exact formula via bivariate normal CDF.
    //
    // V(α,β,l,r) = ∫_l^r Φ(α+βz)φ(z)dz
    //            = P(U ≤ α+βZ, l ≤ Z ≤ r)    where U,Z iid N(0,1)
    //            = Φ₂(h, r; ρ) − Φ₂(h, l; ρ)
    //
    // with h = α/√(1+β²) and ρ = −β/√(1+β²).
    //
    // This is exact to floating-point precision via the high-accuracy
    // Drezner-Wesolowsky BVN routine, replacing the previous fixed 20-point
    // Gauss-Legendre numerical integration of the derivative primitive.
    let s = (1.0 + beta * beta).sqrt();
    let h = alpha / s;
    let rho = -beta / s;
    bivariate_normal_cdf_interval(h, left, right, rho).unwrap_or(0.0)
}

/// Evaluate an affine cell (c2=c3=0) with a value/moment-consistent primitive.
///
/// Value and moments are now generated from the same affine moment primitive.
/// The zero-moment derivative is exact, and `value` is reconstructed by
/// integrating `d value / d alpha = INV_TWO_PI * moments[0]` over `alpha`
/// on a transformed semi-infinite domain.
pub fn evaluate_affine_cell_state(
    cell: DenestedCubicCell,
    max_degree: usize,
) -> Result<CellMomentState, String> {
    let alpha = cell.c0;
    let beta = cell.c1;
    let value = affine_value_from_moment_primitive(alpha, beta, cell.left, cell.right);
    let moments = affine_anchor_moment_vector(alpha, beta, cell.left, cell.right, max_degree);
    Ok(CellMomentState {
        branch: ExactCellBranch::Affine,
        value,
        moments: moments.into(),
    })
}

fn evaluate_affine_cell_derivative_state(
    cell: DenestedCubicCell,
    max_degree: usize,
) -> Result<CellDerivativeMomentState, String> {
    let alpha = cell.c0;
    let beta = cell.c1;
    let moments = affine_anchor_moment_vector(alpha, beta, cell.left, cell.right, max_degree);
    Ok(CellDerivativeMomentState {
        branch: ExactCellBranch::Affine,
        moments: moments.into(),
    })
}

/// Accumulate `mw * z^k` into `moments[k]` for k=0..moments.len(). The
/// "unrolled4" name is historical — this is the plain scalar accumulator
/// that the SIMD outer loop calls per lane. Moment counts are small enough
/// (max_degree + 1 <= ~10) that explicit 4-way unrolling does not measurably
/// improve throughput over the iterator path; the wide::f64x4::exp savings
/// in the SIMD outer dominate the kernel's runtime.
#[inline]
fn accumulate_moments_unrolled4(moments: &mut [f64], mw: f64, z: f64) {
    let mut z_pow = 1.0_f64;
    for slot in moments.iter_mut() {
        *slot += mw * z_pow;
        z_pow *= z;
    }
}

fn evaluate_non_affine_cell_state(
    cell: DenestedCubicCell,
    branch: ExactCellBranch,
    max_degree: usize,
) -> Result<CellMomentState, String> {
    // Direct 384-point Gauss-Legendre on [left, right] is exact to machine
    // precision for any quartic/sextic q on a bounded interval; the prior
    // adaptive transport path expanded basis_moments via the forward 3-/5-
    // step recurrences in reduce_quartic/sextic_moments, which amplify
    // roundoff by (1/lead)^n with lead = 2c2²/3c3² and overflow to NaN for
    // small c2/c3 cells that arise naturally in production.
    let mut moments: CellMomentVec = smallvec![0.0_f64; max_degree + 1];
    let mut value_integral = 0.0_f64;
    let center = 0.5 * (cell.left + cell.right);
    let half_width = 0.5 * (cell.right - cell.left);
    let inv_sqrt_tau = 1.0 / (std::f64::consts::TAU).sqrt();
    let c0 = cell.c0;
    let c1 = cell.c1;
    let c2 = cell.c2;
    let c3 = cell.c3;
    let moments_slice: &mut [f64] = &mut moments;
    debug_assert_eq!(GL_NODES.len(), GL_WEIGHTS.len());
    // SIMD path: process 4 GL nodes per outer iteration, batching the two
    // scalar `exp` calls into single 4-wide `wide::f64x4::exp` invocations.
    // 384 is divisible by 4, so no scalar tail is needed for the GL sweep.
    // The inner moment accumulation is then run scalar per-lane but with a
    // 4-way unrolled slab over the moment slots to break the `z_pow *= z`
    // serial dependency chain.
    use wide::f64x4;
    let center_v = f64x4::splat(center);
    let half_width_v = f64x4::splat(half_width);
    let c0_v = f64x4::splat(c0);
    let c1_v = f64x4::splat(c1);
    let c2_v = f64x4::splat(c2);
    let c3_v = f64x4::splat(c3);
    let neg_half_v = f64x4::splat(-0.5);
    let n_total = GL_NODES.len();
    let n_simd = n_total - (n_total % 4);
    let mut i = 0;
    while i < n_simd {
        let node_v = f64x4::from([
            GL_NODES[i],
            GL_NODES[i + 1],
            GL_NODES[i + 2],
            GL_NODES[i + 3],
        ]);
        let weight_v = f64x4::from([
            GL_WEIGHTS[i],
            GL_WEIGHTS[i + 1],
            GL_WEIGHTS[i + 2],
            GL_WEIGHTS[i + 3],
        ]);
        let z_v = half_width_v.mul_add(node_v, center_v);
        // Horner: ((c3*z + c2)*z + c1)*z + c0
        let eta_v = c3_v
            .mul_add(z_v, c2_v)
            .mul_add(z_v, c1_v)
            .mul_add(z_v, c0_v);
        let z2_v = z_v * z_v;
        let neg_q_v = neg_half_v * (z2_v + eta_v * eta_v);
        let scaled_weight_v = weight_v * half_width_v;
        let exp_negq_v = neg_q_v.exp();
        let neg_half_z2_v = neg_half_v * z2_v;
        let exp_neg_half_z2_v = neg_half_z2_v.exp();
        let moment_weight_v = scaled_weight_v * exp_negq_v;
        let value_term_v = scaled_weight_v * exp_neg_half_z2_v;
        let z_arr = z_v.to_array();
        let eta_arr = eta_v.to_array();
        let mw_arr = moment_weight_v.to_array();
        let vt_arr = value_term_v.to_array();
        for lane in 0..4 {
            let z = z_arr[lane];
            let mw = mw_arr[lane];
            accumulate_moments_unrolled4(moments_slice, mw, z);
            value_integral = vt_arr[lane].mul_add(normal_cdf(eta_arr[lane]), value_integral);
        }
        i += 4;
    }
    while i < n_total {
        let node = GL_NODES[i];
        let weight = GL_WEIGHTS[i];
        let z = center + half_width * node;
        let eta = c3.mul_add(z, c2).mul_add(z, c1).mul_add(z, c0);
        let q = 0.5 * (z * z + eta * eta);
        let scaled_weight = weight * half_width;
        let moment_weight = scaled_weight * (-q).exp();
        accumulate_moments_unrolled4(moments_slice, moment_weight, z);
        value_integral =
            (scaled_weight * (-0.5 * z * z).exp()).mul_add(normal_cdf(eta), value_integral);
        i += 1;
    }
    Ok(CellMomentState {
        branch,
        value: value_integral * inv_sqrt_tau,
        moments,
    })
}

fn evaluate_non_affine_cell_derivative_state(
    cell: DenestedCubicCell,
    branch: ExactCellBranch,
    max_degree: usize,
) -> Result<CellDerivativeMomentState, String> {
    let mut moments: CellMomentVec = smallvec![0.0_f64; max_degree + 1];
    let center = 0.5 * (cell.left + cell.right);
    let half_width = 0.5 * (cell.right - cell.left);
    for (&node, &weight) in GL_NODES.iter().zip(GL_WEIGHTS.iter()) {
        let z = center + half_width * node;
        let moment_weight = weight * (-cell.q(z)).exp();
        let mut z_pow = 1.0_f64;
        for moment in &mut moments {
            *moment = moment_weight.mul_add(z_pow, *moment);
            z_pow *= z;
        }
    }
    for moment in &mut moments {
        *moment *= half_width;
    }
    Ok(CellDerivativeMomentState { branch, moments })
}

/// De-nested cubic cell evaluator.
///
/// Affine cells use the closed-form affine anchor; non-affine cells (Quartic
/// and Sextic branches) are evaluated in a single pass over a fixed
/// high-order Gauss-Legendre rule on `[left, right]`.
pub fn evaluate_cell_moments(
    cell: DenestedCubicCell,
    max_degree: usize,
) -> Result<CellMomentState, String> {
    if !TAIL_CELL_MOMENT_CACHE_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return evaluate_cell_moments_uncached(cell, max_degree);
    }
    let Some(key) = tail_cell_cache_key(cell, max_degree) else {
        return evaluate_cell_moments_uncached(cell, max_degree);
    };
    // Fast probe: serve cached state without holding the mutex during the
    // uncached evaluator below.
    {
        let mut cache = tail_cell_moment_cache()
            .lock()
            .expect("tail cell moment cache mutex poisoned");
        if let Some(state) = cache.moments.get(&key) {
            cache.hits += 1;
            return Ok(state);
        }
    }
    let state = evaluate_cell_moments_uncached(cell, max_degree)?;
    let mut cache = tail_cell_moment_cache()
        .lock()
        .expect("tail cell moment cache mutex poisoned");
    let cache = &mut *cache;
    if let Some(value) = cache.moments.get(&key) {
        cache.hits += 1;
        Ok(value)
    } else {
        cache.misses += 1;
        cache.moments.insert(key, state.clone());
        Ok(state)
    }
}

/// Evaluate cell moments without consulting the global affine-tail memo.
///
/// This is retained for regression tests and before/after microbenchmarks;
/// production callers should use [`evaluate_cell_moments`].
pub fn evaluate_cell_moments_uncached(
    cell: DenestedCubicCell,
    max_degree: usize,
) -> Result<CellMomentState, String> {
    evaluate_cell_state_dispatched(
        cell,
        max_degree,
        evaluate_affine_cell_state,
        evaluate_non_affine_cell_state,
    )
}

/// Evaluate only the moment vector needed by derivative contractions.
///
/// This deliberately does not compute the cell probability value
/// `∫ φ(z) Φ(η(z)) dz`. Derivative contractions consume
/// `∫ z^k exp(-q(z)) dz` moments only, so keeping the value out of the return
/// type prevents this cheaper evaluator from satisfying value-bearing calls.
pub fn evaluate_cell_derivative_moments_uncached(
    cell: DenestedCubicCell,
    max_degree: usize,
) -> Result<CellDerivativeMomentState, String> {
    evaluate_cell_state_dispatched(
        cell,
        max_degree,
        evaluate_affine_cell_derivative_state,
        evaluate_non_affine_cell_derivative_state,
    )
}

/// Shared branch dispatch for the value-bearing and derivative-only cell
/// evaluators. Both walk the same decision tree (semi-infinite tail → must
/// be affine; finite cell → branch-by-coefficients with the sextic
/// degenerate-lowering path), differing only in which pair of
/// `(affine, non_affine)` evaluator helpers to delegate to.  The two helpers
/// are passed as `fn` pointers so the dispatch monomorphizes per `S` and
/// keeps the existing pre-condition errors / unreachable branch handling
/// in lockstep across both evaluators.
fn evaluate_cell_state_dispatched<S>(
    cell: DenestedCubicCell,
    max_degree: usize,
    affine: fn(DenestedCubicCell, usize) -> Result<S, String>,
    non_affine: fn(DenestedCubicCell, ExactCellBranch, usize) -> Result<S, String>,
) -> Result<S, String> {
    let left_inf = !cell.left.is_finite();
    let right_inf = !cell.right.is_finite();
    if left_inf || right_inf {
        // Semi-infinite tail cells must be affine: the deviation saturates
        // to a constant outside support, so c2=c3=0.  Both the BVN CDF
        // and the truncated-Gaussian moment vector handle infinite bounds.
        if cell.c2.abs() > NORMALIZED_CELL_BRANCH_TOL || cell.c3.abs() > NORMALIZED_CELL_BRANCH_TOL
        {
            return Err(CubicCellKernelError::invalid_cell_shape(format!(
                "semi-infinite cell [{}, {}] must be affine (c2=c3=0), got c2={:.3e}, c3={:.3e}",
                cell.left, cell.right, cell.c2, cell.c3
            ))
            .into());
        }
        return affine(cell, max_degree);
    }
    if cell.right <= cell.left {
        return Err(CubicCellKernelError::invalid_cell_shape(format!(
            "finite cell must have left < right, got [{}, {}]",
            cell.left, cell.right
        ))
        .into());
    }
    let branch = branch_cell(cell)?;
    if branch == ExactCellBranch::Affine {
        return affine(cell, max_degree);
    }
    if branch == ExactCellBranch::Sextic {
        let lead = sextic_qprime_coefficients(cell.c0, cell.c1, cell.c2, cell.c3)[5];
        if !lead.is_finite() {
            return Err(CubicCellKernelError::invalid_cell_shape(format!(
                "sextic cell evaluation encountered non-finite leading coefficient: {lead:.3e}"
            ))
            .into());
        }
        if let Some(lower_branch) = degenerate_sextic_branch(cell, lead)? {
            return match lower_branch {
                ExactCellBranch::Quartic => non_affine(
                    DenestedCubicCell { c3: 0.0, ..cell },
                    ExactCellBranch::Quartic,
                    max_degree,
                ),
                ExactCellBranch::Affine => affine(
                    DenestedCubicCell {
                        c2: 0.0,
                        c3: 0.0,
                        ..cell
                    },
                    max_degree,
                ),
                ExactCellBranch::Sextic => Err(CubicCellKernelError::invalid_cell_shape(
                    "internal: degenerate_sextic_branch returned Sextic as a lowered branch",
                )
                .into()),
            };
        }
    }
    non_affine(cell, branch, max_degree)
}

/// Evaluate a de-nested cubic cell through a fit-lifetime byte-limited LRU cache.
///
/// The fingerprint is an exact bit-cast of `(c0, c1, c2, c3, left, right)`, so
/// eviction and reuse cannot alias nearby-but-different cells.  A cached entry
/// computed to a higher degree may satisfy a lower-degree request by truncating
/// the moment vector, preserving the public [`evaluate_cell_moments`] contract.
pub fn evaluate_cell_moments_cached(
    cell: DenestedCubicCell,
    max_degree: usize,
    cache: &CellMomentLruCache,
    stats: Option<&CellMomentCacheStats>,
) -> Result<CellMomentState, String> {
    let key = CellFingerprint::new(cell);
    let existing_derivative = match cache.get(&key) {
        Some(cached) => {
            if let Some(state) = cached.state_for_degree(max_degree) {
                if let Some(stats) = stats {
                    stats.hits.fetch_add(1, Ordering::Relaxed);
                }
                return Ok(state);
            }
            // `cached.derivative_state` is `Option<Arc<_>>`; `.clone()` here
            // is the cheap refcount bump the audit-39 fix targets, not a
            // full moment-vector deep clone.
            cached.derivative_state.clone()
        }
        None => None,
    };
    if let Some(stats) = stats {
        stats.misses.fetch_add(1, Ordering::Relaxed);
    }
    let state = evaluate_cell_moments(cell, max_degree)?;
    // Wrap the freshly-computed state in `Arc` once, share it with the cache
    // through `Arc::clone`, and return the underlying value by unwrapping the
    // unique-reference (caller-side) `Arc`. This replaces the prior
    // `state.clone()` deep copy at the insert site.
    let shared = Arc::new(state);
    let mut entry = CachedCellMoments::new(Arc::clone(&shared));
    if let Some(derivative) = existing_derivative {
        entry = entry.with_derivative(derivative);
    }
    cache.insert(key, entry);
    Ok(Arc::try_unwrap(shared).unwrap_or_else(|a| (*a).clone()))
}

/// Derivative-moment counterpart to [`evaluate_cell_moments_cached`]. Shares
/// the value-moment LRU by storing both moment kinds in a single
/// [`CachedCellMoments`] entry keyed on the cell fingerprint — derivative
/// insertions preserve any pre-existing value state and vice versa, so the
/// two callers never evict each other's work.
pub fn evaluate_cell_derivative_moments_cached(
    cell: DenestedCubicCell,
    max_degree: usize,
    cache: &CellMomentLruCache,
    stats: Option<&CellMomentCacheStats>,
) -> Result<CellDerivativeMomentState, String> {
    let key = CellFingerprint::new(cell);
    let existing_value = match cache.get(&key) {
        Some(cached) => {
            if let Some(state) = cached.derivative_state_for_degree(max_degree) {
                if let Some(stats) = stats {
                    stats.hits.fetch_add(1, Ordering::Relaxed);
                }
                return Ok(state);
            }
            // `cached.state` is `Option<Arc<_>>`; `.clone()` here is the cheap
            // refcount bump the audit-39 fix targets, not a full moment-vector
            // deep clone.
            cached.state.clone()
        }
        None => None,
    };
    if let Some(stats) = stats {
        stats.misses.fetch_add(1, Ordering::Relaxed);
    }
    let state = evaluate_cell_derivative_moments_uncached(cell, max_degree)?;
    // Wrap the freshly-computed state in `Arc` once, share it with the cache
    // through `Arc::clone`, and return the underlying value by unwrapping the
    // unique-reference (caller-side) `Arc`. This replaces the prior
    // `state.clone()` deep copy at the insert site.
    let shared = Arc::new(state);
    let mut entry = CachedCellMoments::new_derivative(Arc::clone(&shared));
    if let Some(value) = existing_value {
        entry = entry.with_value(value);
    }
    cache.insert(key, entry);
    Ok(Arc::try_unwrap(shared).unwrap_or_else(|a| (*a).clone()))
}

/// Scratch-backed variant of [`evaluate_cell_moments`].
///
/// Reuses the supplied [`CellMomentScratch`] for the returned moments slice,
/// so repeated calls with the same scratch (and a sufficient initial capacity)
/// avoid per-call `Vec` allocations on the hot inner-PIRLS row-intercept
/// solver path. Internal transport allocations are unchanged.
pub fn evaluate_cell_moments_with_scratch<'a>(
    cell: DenestedCubicCell,
    max_degree: usize,
    scratch: &'a mut CellMomentScratch,
) -> Result<CellMomentStateRef<'a>, String> {
    let state = evaluate_cell_moments(cell, max_degree)?;
    let out = scratch.prepare_moments(max_degree + 1);
    out.copy_from_slice(&state.moments);
    Ok(CellMomentStateRef {
        branch: state.branch,
        value: state.value,
        moments: out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probability::normal_pdf;

    fn assert_close_rel(label: &str, actual: f64, expected: f64, tol: f64) {
        let denom = expected.abs().max(1.0);
        let rel = (actual - expected).abs() / denom;
        assert!(
            rel <= tol,
            "{label}: actual={actual:.17e} expected={expected:.17e} rel={rel:.3e} tol={tol:.3e}"
        );
    }

    #[test]
    fn non_affine_cell_state_grid_matches_public_cell_moments_reference() {
        let cells = [
            DenestedCubicCell {
                left: -1.25,
                right: -0.2,
                c0: -0.35,
                c1: 0.85,
                c2: 0.04,
                c3: -0.015,
            },
            DenestedCubicCell {
                left: -0.2,
                right: 0.55,
                c0: 0.12,
                c1: -0.65,
                c2: -0.025,
                c3: 0.02,
            },
            DenestedCubicCell {
                left: 0.55,
                right: 1.6,
                c0: 0.42,
                c1: 0.35,
                c2: 0.018,
                c3: 0.012,
            },
        ];
        for cell in cells {
            let branch = branch_cell(cell).expect("branch");
            assert_ne!(branch, ExactCellBranch::Affine);
            for max_degree in [0usize, 2, 4, 9, 16] {
                let direct = evaluate_non_affine_cell_state(cell, branch, max_degree)
                    .expect("direct non-affine transport");
                let public = evaluate_cell_moments(cell, max_degree).expect("public evaluator");
                assert_eq!(direct.branch, public.branch);
                assert_eq!(direct.moments.len(), public.moments.len());
                let value_scale = direct.value.abs().max(public.value.abs()).max(1.0);
                assert!(
                    (direct.value - public.value).abs() <= 1e-10 * value_scale,
                    "value mismatch for {cell:?} degree {max_degree}: direct={} public={}",
                    direct.value,
                    public.value
                );
                for (degree, (lhs, rhs)) in
                    direct.moments.iter().zip(public.moments.iter()).enumerate()
                {
                    let scale = lhs.abs().max(rhs.abs()).max(1.0);
                    assert!(
                        (lhs - rhs).abs() <= 1e-10 * scale,
                        "moment {degree} mismatch for {cell:?} degree {max_degree}: {lhs} vs {rhs}"
                    );
                }
            }
        }
    }

    #[test]
    fn affine_tail_cell_memo_matches_uncached_grid_and_records_hits() {
        // Use a dedicated local cache so the test's hit/miss/entry counters
        // are not perturbed by concurrent tests that drive the shared
        // global memo through `evaluate_cell_moments`. Asserting on the
        // global counters made this test race-flaky when the suite ran in
        // parallel.
        let mut cache = TailCellMomentCache::new();
        let c0s = [-2.0, -0.25, 0.0, 1.5];
        let c1s = [-1.2, -0.05, 0.0, 0.8];
        let endpoints = [-4.0, -1.0, 0.0, 2.5, 6.0];
        let degrees = [0_usize, 4, 9, 16, 24];

        for &c0 in &c0s {
            for &c1 in &c1s {
                for &endpoint in &endpoints {
                    for &max_degree in &degrees {
                        for &(left, right) in
                            &[(f64::NEG_INFINITY, endpoint), (endpoint, f64::INFINITY)]
                        {
                            let cell = DenestedCubicCell {
                                left,
                                right,
                                c0,
                                c1,
                                c2: 0.0,
                                c3: 0.0,
                            };
                            let expected = evaluate_cell_moments_uncached(cell, max_degree)
                                .expect("uncached affine tail moments");
                            let actual = cache
                                .evaluate(cell, max_degree)
                                .expect("cached affine tail moments miss");
                            let repeat = cache
                                .evaluate(cell, max_degree)
                                .expect("cached affine tail moments hit");
                            assert_eq!(actual.branch, expected.branch);
                            assert_eq!(repeat.branch, expected.branch);
                            assert_close_rel(
                                "tail value miss",
                                actual.value,
                                expected.value,
                                1e-14,
                            );
                            assert_close_rel("tail value hit", repeat.value, expected.value, 1e-14);
                            assert_eq!(actual.moments.len(), expected.moments.len());
                            assert_eq!(repeat.moments.len(), expected.moments.len());
                            for (idx, ((a, r), e)) in actual
                                .moments
                                .iter()
                                .zip(repeat.moments.iter())
                                .zip(expected.moments.iter())
                                .enumerate()
                            {
                                assert_close_rel(
                                    &format!("tail moment miss[{idx}]"),
                                    *a,
                                    *e,
                                    1e-14,
                                );
                                assert_close_rel(&format!("tail moment hit[{idx}]"), *r, *e, 1e-14);
                            }
                        }
                    }
                }
            }
        }

        let stats = cache.stats();
        assert_eq!(stats.misses, stats.entries);
        assert!(
            stats.hits >= stats.misses,
            "expected repeat hits: {stats:?}"
        );
        assert!(
            stats.hit_rate() >= 0.5,
            "unexpected low hit rate: {stats:?}"
        );
    }

    fn reference_bivariate_normal_cdf_20(h: f64, k: f64, rho: f64) -> f64 {
        if h == f64::NEG_INFINITY || k == f64::NEG_INFINITY {
            return 0.0;
        }
        if h == f64::INFINITY {
            return normal_cdf(k);
        }
        if k == f64::INFINITY {
            return normal_cdf(h);
        }
        let rho_clamped = rho.clamp(-1.0, 1.0);
        if rho_clamped >= 1.0 - 1e-12 {
            return normal_cdf(h.min(k));
        }
        if rho_clamped <= -1.0 + 1e-12 {
            return (normal_cdf(h) - normal_cdf(-k)).clamp(0.0, 1.0);
        }

        let hs = 0.5 * (h * h + k * k);
        let asr = rho_clamped.asin();
        let mut sum = 0.0;
        for (&node, &weight) in BVN_GL_NODES_20.iter().zip(BVN_GL_WEIGHTS_20.iter()) {
            let sn = (0.5 * asr * (node + 1.0)).sin();
            let one_minus = 1.0 - sn * sn;
            let expo = ((sn * h * k) - hs) / one_minus;
            sum += weight * expo.exp();
        }
        (normal_cdf(h) * normal_cdf(k) + asr * sum / (4.0 * std::f64::consts::PI)).clamp(0.0, 1.0)
    }

    #[test]
    fn non_affine_cell_state_reference_grid_matches_public_moments() {
        let c0s = [-0.4, 0.0, 0.35];
        let c1s = [-0.8, 0.25, 1.1];
        let c2s = [-0.12, 0.08];
        let c3s = [-0.04, 0.03];
        let intervals = [(-1.25, -0.2), (-0.5, 0.75), (0.1, 1.4)];
        let degrees = [3usize, 6, 9, 12];

        for &c0 in &c0s {
            for &c1 in &c1s {
                for &c2 in &c2s {
                    for &c3 in &c3s {
                        for &(left, right) in &intervals {
                            let cell = DenestedCubicCell {
                                left,
                                right,
                                c0,
                                c1,
                                c2,
                                c3,
                            };
                            let branch = branch_cell(cell).expect("branch");
                            assert_ne!(branch, ExactCellBranch::Affine);
                            for &degree in &degrees {
                                let direct = evaluate_non_affine_cell_state(cell, branch, degree)
                                    .expect("direct non-affine state");
                                let public = evaluate_cell_moments(cell, degree)
                                    .expect("public non-affine state");
                                assert_eq!(direct.branch, public.branch);
                                let value_scale =
                                    direct.value.abs().max(public.value.abs()).max(1.0);
                                assert!(
                                    (direct.value - public.value).abs() / value_scale <= 1.0e-15,
                                    "value mismatch for {cell:?}, degree {degree}: direct={:.17e}, public={:.17e}",
                                    direct.value,
                                    public.value
                                );
                                assert_eq!(direct.moments.len(), public.moments.len());
                                for (idx, (&a, &b)) in
                                    direct.moments.iter().zip(public.moments.iter()).enumerate()
                                {
                                    let scale = a.abs().max(b.abs()).max(1.0);
                                    assert!(
                                        (a - b).abs() / scale <= 1.0e-15,
                                        "moment {idx} mismatch for {cell:?}, degree {degree}: direct={a:.17e}, public={b:.17e}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn bivariate_normal_cdf_matches_reference_grid_to_1e_minus_10() {
        let hs = [-8.0, -5.0, -3.0, -1.5, -0.5, 0.0, 0.25, 1.0, 2.5, 5.0, 8.0];
        let ks = [-8.0, -4.0, -2.0, -0.75, 0.0, 0.4, 1.25, 3.0, 6.0, 8.0];
        let rhos = [
            -0.999_999_999_999,
            -0.999,
            -0.95,
            -0.7,
            -0.3,
            -1.0e-12,
            0.0,
            1.0e-12,
            0.3,
            0.7,
            0.95,
            0.999,
            0.999_999_999_999,
        ];
        for &h in &hs {
            for &k in &ks {
                for &rho in &rhos {
                    let actual = bivariate_normal_cdf(h, k, rho).expect("bvn");
                    let expected = reference_bivariate_normal_cdf_20(h, k, rho);
                    let scale = expected.abs().max(1.0e-300);
                    let rel = (actual - expected).abs() / scale;
                    assert!(
                        rel < 1.0e-10 || (actual - expected).abs() < 1.0e-14,
                        "h={h} k={k} rho={rho} actual={actual:.17e} expected={expected:.17e} rel={rel:.3e}"
                    );
                }
            }
        }
    }

    #[test]
    fn bivariate_normal_cdf_matches_reference_lcg_property_samples() {
        let mut seed = 0x5eed_cafe_f00d_u64;
        let mut next_unit = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
        };
        for _ in 0..4096 {
            let h = -8.0 + 16.0 * next_unit();
            let k = -8.0 + 16.0 * next_unit();
            let rho = -0.999 + 1.998 * next_unit();
            let actual = bivariate_normal_cdf(h, k, rho).expect("bvn");
            let expected = reference_bivariate_normal_cdf_20(h, k, rho);
            let scale = expected.abs().max(1.0e-300);
            let rel = (actual - expected).abs() / scale;
            assert!(
                rel < 1.0e-10 || (actual - expected).abs() < 1.0e-14,
                "h={h} k={k} rho={rho} actual={actual:.17e} expected={expected:.17e} rel={rel:.3e}"
            );
        }
    }

    #[test]
    fn affine_bvn_interval_primitive_matches_two_cdf_difference() {
        let hs = [-6.0, -2.0, -0.25, 0.0, 0.8, 3.0, 6.0];
        let bounds = [
            (-5.0, -2.0),
            (-3.0, -0.1),
            (-1.0, 0.0),
            (-0.25, 0.75),
            (0.2, 3.5),
            (2.0, 7.0),
        ];
        let rhos = [-0.98, -0.8, -0.25, 0.0, 0.25, 0.8, 0.98];
        for &h in &hs {
            for &(left, right) in &bounds {
                for &rho in &rhos {
                    let actual =
                        bivariate_normal_cdf_interval(h, left, right, rho).expect("interval");
                    let expected = (reference_bivariate_normal_cdf_20(h, right, rho)
                        - reference_bivariate_normal_cdf_20(h, left, rho))
                    .clamp(0.0, 1.0);
                    let scale = expected.abs().max(1.0e-300);
                    let rel = (actual - expected).abs() / scale;
                    assert!(
                        rel < 1.0e-10 || (actual - expected).abs() < 1.0e-12,
                        "h={h} left={left} right={right} rho={rho} actual={actual:.17e} expected={expected:.17e} rel={rel:.3e}"
                    );
                }
            }
        }
    }

    fn simpson_integral<F>(left: f64, right: f64, steps: usize, f: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let n = if steps.is_multiple_of(2) {
            steps
        } else {
            steps + 1
        };
        let h = (right - left) / n as f64;
        let mut acc = f(left) + f(right);
        for k in 1..n {
            let x = left + h * k as f64;
            let w = if k % 2 == 0 { 2.0 } else { 4.0 };
            acc += w * f(x);
        }
        acc * h / 3.0
    }

    #[test]
    fn global_transform_preserves_local_span_polynomial() {
        let span = LocalSpanCubic {
            left: -1.2,
            right: 0.8,
            c0: 0.3,
            c1: -0.25,
            c2: 0.11,
            c3: -0.04,
        };
        let (g0, g1, g2, g3) = global_cubic_from_local(span);
        for &x in &[-1.2, -0.7, -0.1, 0.4, 0.8] {
            let local = span.evaluate(x);
            let global = g0 + g1 * x + g2 * x * x + g3 * x * x * x;
            assert!((local - global).abs() < 1e-12);
        }
    }

    #[test]
    fn bivariate_normal_cdf_independent_factorizes() {
        let h = -0.35;
        let k = 0.8;
        let out = bivariate_normal_cdf(h, k, 0.0).expect("bvn");
        let target = normal_cdf(h) * normal_cdf(k);
        assert!((out - target).abs() < 1e-12);
    }

    #[test]
    fn evaluate_affine_cell_state_matches_numeric_integrals() {
        let cell = DenestedCubicCell {
            left: -0.9,
            right: 0.8,
            c0: 0.15,
            c1: -0.35,
            c2: 0.0,
            c3: 0.0,
        };
        let state = evaluate_affine_cell_state(cell, 6).expect("affine cell");
        let value_numeric = simpson_integral(cell.left, cell.right, 4000, |z| {
            super::normal_cdf(cell.eta(z)) * normal_pdf(z)
        });
        assert_eq!(state.branch, ExactCellBranch::Affine);
        assert!((state.value - value_numeric).abs() < 1e-9);
        for degree in 0..=6 {
            let target = simpson_integral(cell.left, cell.right, 4000, |z| {
                z.powi(degree as i32) * (-cell.q(z)).exp()
            });
            assert!((state.moments[degree] - target).abs() < 1e-9);
        }
    }

    #[test]
    fn affine_cell_value_matches_zero_moment_derivative() {
        let cell = DenestedCubicCell {
            left: -1.1,
            right: 0.7,
            c0: 0.23,
            c1: -0.41,
            c2: 0.0,
            c3: 0.0,
        };
        let h = 1e-6;
        let plus = evaluate_affine_cell_state(
            DenestedCubicCell {
                c0: cell.c0 + h,
                ..cell
            },
            0,
        )
        .expect("affine plus");
        let minus = evaluate_affine_cell_state(
            DenestedCubicCell {
                c0: cell.c0 - h,
                ..cell
            },
            0,
        )
        .expect("affine minus");
        let center = evaluate_affine_cell_state(cell, 0).expect("affine center");
        let d_value = (plus.value - minus.value) / (2.0 * h);
        let target = INV_TWO_PI * center.moments[0];
        assert!((d_value - target).abs() < 1e-8);
    }

    #[test]
    fn coefficient_partials_match_exact_span_derivatives() {
        let score_span = LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: 0.08,
            c1: -0.03,
            c2: 0.02,
            c3: -0.01,
        };
        let link_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: -0.05,
            c1: 0.04,
            c2: -0.02,
            c3: 0.015,
        };
        let a = 0.3;
        let b = -0.7;
        let (dc_da, dc_db) = denested_cell_coefficient_partials(score_span, link_span, a, b);
        for &z in &[-0.75, -0.4, -0.1, 0.2] {
            let u = a + b * z;
            let eta_a = 1.0 + link_span.first_derivative(u);
            let eta_b = z + score_span.evaluate(z) + z * link_span.first_derivative(u);
            assert!((polynomial_value(&dc_da, z) - eta_a).abs() < 1e-12);
            assert!((polynomial_value(&dc_db, z) - eta_b).abs() < 1e-12);
        }
    }

    #[test]
    fn second_coefficient_partials_match_exact_span_derivatives() {
        let score_span = LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: 0.08,
            c1: -0.03,
            c2: 0.02,
            c3: -0.01,
        };
        let link_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: -0.05,
            c1: 0.04,
            c2: -0.02,
            c3: 0.015,
        };
        let a = 0.3;
        let b = -0.7;
        let second_partials = denested_cell_second_partials(score_span, link_span, a, b);
        let dc_daa = second_partials.0;
        let dc_dab = second_partials.1;
        let dc_dbb = second_partials.2;
        for &z in &[-0.75, -0.4, -0.1, 0.2] {
            let u = a + b * z;
            let eta_aa = link_span.second_derivative(u);
            let eta_ab = z * link_span.second_derivative(u);
            let eta_bb = z * z * link_span.second_derivative(u);
            assert!((polynomial_value(&dc_daa, z) - eta_aa).abs() < 1e-12);
            assert!((polynomial_value(&dc_dab, z) - eta_ab).abs() < 1e-12);
            assert!((polynomial_value(&dc_dbb, z) - eta_bb).abs() < 1e-12);
        }
    }

    #[test]
    fn higher_derivative_moment_helpers_reject_empty_first_coefficients() {
        let cell = DenestedCubicCell {
            left: -1.0,
            right: 1.0,
            c0: 0.0,
            c1: 1.0,
            c2: 0.0,
            c3: 0.0,
        };
        let moments = [1.0; 16];

        let third_err = cell_third_derivative_from_moments(
            cell,
            &[],
            &[1.0],
            &[1.0],
            &[],
            &[],
            &[],
            &[],
            &moments,
        )
        .expect_err("empty first coefficients should be rejected");
        assert!(third_err.contains("r first-derivative coefficients must be non-empty"));

        let fourth_err = cell_fourth_derivative_from_moments(
            cell,
            &[1.0],
            &[],
            &[1.0],
            &[1.0],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
            &moments,
        )
        .expect_err("empty first coefficients should be rejected");
        assert!(fourth_err.contains("s first-derivative coefficients must be non-empty"));
    }

    #[test]
    fn fourth_derivative_rejects_overlong_scratch_convolutions() {
        let cell = DenestedCubicCell {
            left: -1.0,
            right: 1.0,
            c0: 0.0,
            c1: 1.0,
            c2: 0.0,
            c3: 0.0,
        };
        let long_first = [1.0; 10];
        let zero = [0.0; 1];
        let moments = [1.0; 64];

        let err = cell_fourth_derivative_from_moments(
            cell,
            &long_first,
            &long_first,
            &long_first,
            &long_first,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &moments,
        )
        .expect_err("oversized convolution should be rejected before writing scratch");
        assert!(err.contains("fourth derivative polynomial convolution scratch too small"));
    }

    #[test]
    fn score_and_link_basis_cell_coefficients_match_direct_construction() {
        let score_basis_span = LocalSpanCubic {
            left: -0.7,
            right: 0.4,
            c0: 0.2,
            c1: -0.04,
            c2: 0.03,
            c3: -0.01,
        };
        let link_basis_span = LocalSpanCubic {
            left: -0.5,
            right: 1.1,
            c0: -0.03,
            c1: 0.05,
            c2: -0.02,
            c3: 0.01,
        };
        let a = 0.25;
        let b = -0.8;
        let score_coeffs = score_basis_cell_coefficients(score_basis_span, b);
        let link_coeffs = link_basis_cell_coefficients(link_basis_span, a, b);
        for &z in &[-0.7, -0.1, 0.2, 0.4] {
            let score_poly = polynomial_value(&score_coeffs, z);
            let link_poly = polynomial_value(&link_coeffs, z);
            assert!((score_poly - b * score_basis_span.evaluate(z)).abs() < 1e-12);
            assert!((link_poly - link_basis_span.evaluate(a + b * z)).abs() < 1e-12);
        }
    }

    #[test]
    fn link_basis_partials_match_exact_span_derivatives() {
        let link_basis_span = LocalSpanCubic {
            left: -0.5,
            right: 1.1,
            c0: -0.03,
            c1: 0.05,
            c2: -0.02,
            c3: 0.01,
        };
        let a = 0.25;
        let b = -0.8;
        let (dc_da, dc_db) = link_basis_cell_coefficient_partials(link_basis_span, a, b);
        let (dc_daa, dc_dab, dc_dbb) = link_basis_cell_second_partials(link_basis_span, a, b);
        for &z in &[-0.6, -0.2, 0.15, 0.5] {
            let u = a + b * z;
            let eta_a = link_basis_span.first_derivative(u);
            let eta_b = z * link_basis_span.first_derivative(u);
            let eta_aa = link_basis_span.second_derivative(u);
            let eta_ab = z * link_basis_span.second_derivative(u);
            let eta_bb = z * z * link_basis_span.second_derivative(u);
            assert!((polynomial_value(&dc_da, z) - eta_a).abs() < 1e-12);
            assert!((polynomial_value(&dc_db, z) - eta_b).abs() < 1e-12);
            assert!((polynomial_value(&dc_daa, z) - eta_aa).abs() < 1e-12);
            assert!((polynomial_value(&dc_dab, z) - eta_ab).abs() < 1e-12);
            assert!((polynomial_value(&dc_dbb, z) - eta_bb).abs() < 1e-12);
        }
    }

    #[test]
    fn denested_third_partials_match_exact_span_derivatives() {
        let link_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: -0.05,
            c1: 0.04,
            c2: -0.02,
            c3: 0.015,
        };
        let (dc_daaa, dc_daab, dc_dabb, dc_dbbb) = denested_cell_third_partials(link_span);
        let link_third = 6.0 * link_span.c3;
        for &z in &[-0.75, -0.4, -0.1, 0.2] {
            let eta_aaa = link_third;
            let eta_aab = z * link_third;
            let eta_abb = z * z * link_third;
            let eta_bbb = z * z * z * link_third;
            assert!((polynomial_value(&dc_daaa, z) - eta_aaa).abs() < 1e-12);
            assert!((polynomial_value(&dc_daab, z) - eta_aab).abs() < 1e-12);
            assert!((polynomial_value(&dc_dabb, z) - eta_abb).abs() < 1e-12);
            assert!((polynomial_value(&dc_dbbb, z) - eta_bbb).abs() < 1e-12);
        }
    }

    #[test]
    fn link_basis_third_partials_match_exact_span_derivatives() {
        let link_basis_span = LocalSpanCubic {
            left: -0.5,
            right: 1.1,
            c0: -0.03,
            c1: 0.05,
            c2: -0.02,
            c3: 0.01,
        };
        let (dc_daaa, dc_daab, dc_dabb, dc_dbbb) = link_basis_cell_third_partials(link_basis_span);
        let link_third = 6.0 * link_basis_span.c3;
        for &z in &[-0.6, -0.2, 0.15, 0.5] {
            let eta_aaa = link_third;
            let eta_aab = z * link_third;
            let eta_abb = z * z * link_third;
            let eta_bbb = z * z * z * link_third;
            assert!((polynomial_value(&dc_daaa, z) - eta_aaa).abs() < 1e-12);
            assert!((polynomial_value(&dc_daab, z) - eta_aab).abs() < 1e-12);
            assert!((polynomial_value(&dc_dabb, z) - eta_abb).abs() < 1e-12);
            assert!((polynomial_value(&dc_dbbb, z) - eta_bbb).abs() < 1e-12);
        }
    }

    #[test]
    fn branch_selection_uses_normalized_non_affine_coefficients() {
        let affine = DenestedCubicCell {
            left: -1.0,
            right: 1.0,
            c0: 0.1,
            c1: -0.4,
            c2: 1e-13,
            c3: -1e-13,
        };
        let quartic = DenestedCubicCell {
            c2: 2e-4,
            c3: 1e-13,
            ..affine
        };
        let sextic = DenestedCubicCell {
            c2: 2e-4,
            c3: 5e-3,
            ..affine
        };
        assert_eq!(branch_cell(affine).unwrap(), ExactCellBranch::Affine);
        assert_eq!(branch_cell(quartic).unwrap(), ExactCellBranch::Quartic);
        assert_eq!(branch_cell(sextic).unwrap(), ExactCellBranch::Sextic);
    }

    #[test]
    fn affine_anchor_moments_match_whole_line_closed_forms() {
        let out = affine_anchor_moment_vector(0.0, 0.0, f64::NEG_INFINITY, f64::INFINITY, 4);
        let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
        assert!((out[0] - sqrt_2pi).abs() < 1e-12);
        assert!(out[1].abs() < 1e-12);
        assert!((out[2] - sqrt_2pi).abs() < 1e-12);
    }

    #[test]
    fn affine_anchor_moments_match_shifted_gaussian_whole_line() {
        let alpha = 0.7;
        let beta = -0.4;
        let out = affine_anchor_moment_vector(alpha, beta, f64::NEG_INFINITY, f64::INFINITY, 4);
        let s = (1.0 + beta * beta).sqrt();
        let mu = -alpha * beta / (1.0 + beta * beta);
        let scale = (-alpha * alpha / (2.0 * s * s)).exp() / s;
        let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
        assert!((out[0] - scale * sqrt_2pi).abs() < 1e-12);
        assert!((out[1] - scale * sqrt_2pi * mu).abs() < 1e-12);
        assert!((out[2] - scale * sqrt_2pi * (mu * mu + 1.0 / (s * s))).abs() < 1e-10);
    }

    #[test]
    fn quartic_recurrence_reduces_higher_moments() {
        let cell = DenestedCubicCell {
            left: -1.0,
            right: 0.9,
            c0: 0.2,
            c1: -0.3,
            c2: 0.18,
            c3: 0.0,
        };
        let exact = |k: usize| {
            simpson_integral(cell.left, cell.right, 2000, |z| {
                z.powi(k as i32) * (-cell.q(z)).exp()
            })
        };
        let reduced = reduce_quartic_moments(cell, [exact(0), exact(1), exact(2)], 6)
            .expect("quartic reduction");
        for k in 0..=6 {
            let target = exact(k);
            assert!(
                (reduced[k] - target).abs() < 1e-7,
                "quartic reduced moment M{k} mismatch: {} vs {}",
                reduced[k],
                target
            );
        }
    }

    #[test]
    fn sextic_recurrence_reduces_higher_moments() {
        let cell = DenestedCubicCell {
            left: -0.8,
            right: 0.7,
            c0: -0.1,
            c1: 0.25,
            c2: -0.14,
            c3: 0.22,
        };
        let exact = |k: usize| {
            simpson_integral(cell.left, cell.right, 3000, |z| {
                z.powi(k as i32) * (-cell.q(z)).exp()
            })
        };
        let reduced =
            reduce_sextic_moments(cell, [exact(0), exact(1), exact(2), exact(3), exact(4)], 9)
                .expect("sextic reduction");
        for k in 0..=9 {
            let target = exact(k);
            assert!(
                (reduced[k] - target).abs() < 1e-7,
                "sextic reduced moment M{k} mismatch: {} vs {}",
                reduced[k],
                target
            );
        }
    }

    #[test]
    fn degenerate_sextic_branch_preserves_quadratic_coefficient() {
        let cell = DenestedCubicCell {
            left: -1.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.1,
            c3: 2.0e-10,
        };
        assert_eq!(branch_cell(cell).unwrap(), ExactCellBranch::Sextic);

        let state = evaluate_cell_moments(cell, 9).expect("degenerate sextic cell");
        let quartic_cell = DenestedCubicCell { c3: 0.0, ..cell };
        let quartic = evaluate_cell_moments(quartic_cell, 9).expect("quartic cell");
        let affine = evaluate_affine_cell_state(
            DenestedCubicCell {
                c2: 0.0,
                c3: 0.0,
                ..cell
            },
            9,
        )
        .expect("affine cell");

        assert_eq!(state.branch, ExactCellBranch::Quartic);
        for k in 0..=9 {
            assert!(
                (state.moments[k] - quartic.moments[k]).abs() < 1e-12,
                "lowered moment M{k} should match the quartic cell: {} vs {}",
                state.moments[k],
                quartic.moments[k]
            );
        }
        assert!(
            (state.moments[0] - affine.moments[0]).abs() > 1e-4,
            "degenerate sextic handling must not drop the nonzero c2 term"
        );
    }

    #[test]
    fn moment_reduced_first_and_second_derivatives_match_numeric_integrals() {
        let cell = DenestedCubicCell {
            left: -0.9,
            right: 0.6,
            c0: 0.15,
            c1: -0.2,
            c2: 0.08,
            c3: 0.17,
        };
        let moments = reduce_sextic_moments(
            cell,
            [
                simpson_integral(cell.left, cell.right, 3000, |z| (-cell.q(z)).exp()),
                simpson_integral(cell.left, cell.right, 3000, |z| z * (-cell.q(z)).exp()),
                simpson_integral(cell.left, cell.right, 3000, |z| z * z * (-cell.q(z)).exp()),
                simpson_integral(cell.left, cell.right, 3000, |z| {
                    z.powi(3) * (-cell.q(z)).exp()
                }),
                simpson_integral(cell.left, cell.right, 3000, |z| {
                    z.powi(4) * (-cell.q(z)).exp()
                }),
            ],
            9,
        )
        .expect("reduced moments");

        let r = [0.7, -0.1, 0.3];
        let s = [0.2, 0.5];
        let second = [0.4, -0.2, 0.1];
        let exact_first = cell_first_derivative_from_moments(&r, &moments).expect("first");
        let exact_second =
            cell_second_derivative_from_moments(cell, &r, &s, &second, &moments).expect("second");

        let numeric_first = simpson_integral(cell.left, cell.right, 3000, |z| {
            polynomial_value(&r, z) * (-cell.q(z)).exp() / (2.0 * std::f64::consts::PI)
        });
        let numeric_second = simpson_integral(cell.left, cell.right, 3000, |z| {
            let eta = cell.eta(z);
            (polynomial_value(&second, z) - eta * polynomial_value(&r, z) * polynomial_value(&s, z))
                * (-cell.q(z)).exp()
                / (2.0 * std::f64::consts::PI)
        });

        assert!((exact_first - numeric_first).abs() < 1e-7);
        assert!((exact_second - numeric_second).abs() < 1e-7);
    }

    #[test]
    fn moment_reduced_third_derivative_matches_numeric_integral() {
        let cell = DenestedCubicCell {
            left: -0.85,
            right: 0.7,
            c0: -0.12,
            c1: 0.18,
            c2: 0.09,
            c3: -0.11,
        };
        let moments = evaluate_cell_moments(cell, 12).expect("cell moments");
        let r = [0.35, -0.12, 0.08];
        let s = [0.17, 0.09];
        let t = [-0.21, 0.14, -0.04];
        let rs = [0.11, -0.07, 0.05];
        let rt = [-0.06, 0.03];
        let st = [0.08, -0.02, 0.01];
        let rst = [0.04, -0.05, 0.02];

        let exact_third = cell_third_derivative_from_moments(
            cell,
            &r,
            &s,
            &t,
            &rs,
            &rt,
            &st,
            &rst,
            &moments.moments,
        )
        .expect("third derivative");
        let numeric_third = simpson_integral(cell.left, cell.right, 4000, |z| {
            let eta = cell.eta(z);
            let rz = polynomial_value(&r, z);
            let sz = polynomial_value(&s, z);
            let tz = polynomial_value(&t, z);
            let rsz = polynomial_value(&rs, z);
            let rtz = polynomial_value(&rt, z);
            let stz = polynomial_value(&st, z);
            let rstz = polynomial_value(&rst, z);
            (rstz - eta * (rsz * tz + rtz * sz + stz * rz) + (eta * eta - 1.0) * rz * sz * tz)
                * (-cell.q(z)).exp()
                / (2.0 * std::f64::consts::PI)
        });

        assert!((exact_third - numeric_third).abs() < 1e-7);
    }

    #[test]
    fn moment_reduced_fourth_derivative_matches_numeric_integral() {
        let cell = DenestedCubicCell {
            left: -0.8,
            right: 0.65,
            c0: 0.11,
            c1: -0.22,
            c2: 0.07,
            c3: 0.13,
        };
        let moments = evaluate_cell_moments(cell, 16).expect("cell moments");
        let r = [0.21, -0.13, 0.06];
        let s = [-0.18, 0.04];
        let t = [0.09, 0.07, -0.03];
        let u = [-0.14, 0.05];
        let rs = [0.08, -0.03, 0.02];
        let rt = [-0.05, 0.01];
        let ru = [0.04, -0.02, 0.01];
        let st = [0.03, 0.02];
        let su = [-0.02, 0.05, -0.01];
        let tu = [0.07, -0.04];
        let rst = [0.03, -0.01, 0.02];
        let rsu = [-0.02, 0.04];
        let rtu = [0.01, 0.02, -0.01];
        let stu = [-0.03, 0.02];
        let rstu = [0.02, -0.01, 0.01];

        let exact_fourth = cell_fourth_derivative_from_moments(
            cell,
            &r,
            &s,
            &t,
            &u,
            &rs,
            &rt,
            &ru,
            &st,
            &su,
            &tu,
            &rst,
            &rsu,
            &rtu,
            &stu,
            &rstu,
            &moments.moments,
        )
        .expect("fourth derivative");
        let numeric_fourth = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let rz = polynomial_value(&r, z);
            let sz = polynomial_value(&s, z);
            let tz = polynomial_value(&t, z);
            let uz = polynomial_value(&u, z);
            let rsz = polynomial_value(&rs, z);
            let rtz = polynomial_value(&rt, z);
            let ruz = polynomial_value(&ru, z);
            let stz = polynomial_value(&st, z);
            let suz = polynomial_value(&su, z);
            let tuz = polynomial_value(&tu, z);
            let rstz = polynomial_value(&rst, z);
            let rsuz = polynomial_value(&rsu, z);
            let rtuz = polynomial_value(&rtu, z);
            let stuz = polynomial_value(&stu, z);
            let rstuz = polynomial_value(&rstu, z);
            let linear =
                rstz * uz + rsuz * tz + rtuz * sz + stuz * rz + rsz * tuz + rtz * suz + ruz * stz;
            let quadratic = rsz * tz * uz
                + rtz * sz * uz
                + ruz * sz * tz
                + stz * rz * uz
                + suz * rz * tz
                + tuz * rz * sz;
            let quartic = rz * sz * tz * uz;
            (rstuz - eta * linear
                + (eta * eta - 1.0) * quadratic
                + (-eta * eta * eta + 3.0 * eta) * quartic)
                * (-cell.q(z)).exp()
                / (2.0 * std::f64::consts::PI)
        });

        assert!((exact_fourth - numeric_fourth).abs() < 2e-7);
    }

    #[test]
    fn denested_cell_parameter_derivatives_match_exact_integrands() {
        let score_span = LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: 0.08,
            c1: -0.03,
            c2: 0.02,
            c3: -0.01,
        };
        let link_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: -0.05,
            c1: 0.04,
            c2: -0.02,
            c3: 0.015,
        };
        let a = 0.3;
        let b = -0.7;
        let coeffs = denested_cell_coefficients(score_span, link_span, a, b);
        let cell = DenestedCubicCell {
            left: score_span.left,
            right: score_span.right,
            c0: coeffs[0],
            c1: coeffs[1],
            c2: coeffs[2],
            c3: coeffs[3],
        };
        let state = evaluate_cell_moments(cell, 24).expect("cell moments");
        let (dc_da, dc_db) = denested_cell_coefficient_partials(score_span, link_span, a, b);
        let (dc_daa, dc_dab, dc_dbb) = denested_cell_second_partials(score_span, link_span, a, b);
        let (dc_daaa, dc_daab, dc_dabb, dc_dbbb) = denested_cell_third_partials(link_span);
        let zero = [0.0; 4];
        let link_third = 6.0 * link_span.c3;

        let eta_a = |z: f64| 1.0 + link_span.first_derivative(a + b * z);
        let eta_b = |z: f64| z + score_span.evaluate(z) + z * link_span.first_derivative(a + b * z);
        let eta_aa = |z: f64| link_span.second_derivative(a + b * z);
        let eta_ab = |z: f64| z * link_span.second_derivative(a + b * z);
        let eta_bb = |z: f64| z * z * link_span.second_derivative(a + b * z);
        let eta_aaa = |z: f64| link_third + 0.0 * z;
        let eta_aab = |z: f64| z * link_third;
        let eta_abb = |z: f64| z * z * link_third;
        let eta_bbb = |z: f64| z * z * z * link_third;

        let exact_a = cell_first_derivative_from_moments(&dc_da, &state.moments).expect("a");
        let exact_b = cell_first_derivative_from_moments(&dc_db, &state.moments).expect("b");
        let exact_aa =
            cell_second_derivative_from_moments(cell, &dc_da, &dc_da, &dc_daa, &state.moments)
                .expect("aa");
        let exact_ab =
            cell_second_derivative_from_moments(cell, &dc_da, &dc_db, &dc_dab, &state.moments)
                .expect("ab");
        let exact_bb =
            cell_second_derivative_from_moments(cell, &dc_db, &dc_db, &dc_dbb, &state.moments)
                .expect("bb");
        let exact_aaa = cell_third_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &dc_da,
            &dc_daa,
            &dc_daa,
            &dc_daa,
            &dc_daaa,
            &state.moments,
        )
        .expect("aaa");
        let exact_aab = cell_third_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &dc_db,
            &dc_daa,
            &dc_dab,
            &dc_dab,
            &dc_daab,
            &state.moments,
        )
        .expect("aab");
        let exact_abb = cell_third_derivative_from_moments(
            cell,
            &dc_da,
            &dc_db,
            &dc_db,
            &dc_dab,
            &dc_dab,
            &dc_dbb,
            &dc_dabb,
            &state.moments,
        )
        .expect("abb");
        let exact_bbb = cell_third_derivative_from_moments(
            cell,
            &dc_db,
            &dc_db,
            &dc_db,
            &dc_dbb,
            &dc_dbb,
            &dc_dbb,
            &dc_dbbb,
            &state.moments,
        )
        .expect("bbb");
        let exact_aaaa = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &dc_da,
            &dc_da,
            &dc_daa,
            &dc_daa,
            &dc_daa,
            &dc_daa,
            &dc_daa,
            &dc_daa,
            &dc_daaa,
            &dc_daaa,
            &dc_daaa,
            &dc_daaa,
            &zero,
            &state.moments,
        )
        .expect("aaaa");
        let exact_aaab = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &dc_da,
            &dc_db,
            &dc_daa,
            &dc_daa,
            &dc_dab,
            &dc_daa,
            &dc_dab,
            &dc_dab,
            &dc_daaa,
            &dc_daab,
            &dc_daab,
            &dc_daab,
            &zero,
            &state.moments,
        )
        .expect("aaab");
        let exact_aabb = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &dc_db,
            &dc_db,
            &dc_daa,
            &dc_dab,
            &dc_dab,
            &dc_dab,
            &dc_dab,
            &dc_dbb,
            &dc_daab,
            &dc_daab,
            &dc_dabb,
            &dc_dabb,
            &zero,
            &state.moments,
        )
        .expect("aabb");
        let exact_abbb = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_db,
            &dc_db,
            &dc_db,
            &dc_dab,
            &dc_dab,
            &dc_dab,
            &dc_dbb,
            &dc_dbb,
            &dc_dbb,
            &dc_dabb,
            &dc_dabb,
            &dc_dabb,
            &dc_dbbb,
            &zero,
            &state.moments,
        )
        .expect("abbb");
        let exact_bbbb = cell_fourth_derivative_from_moments(
            cell,
            &dc_db,
            &dc_db,
            &dc_db,
            &dc_db,
            &dc_dbb,
            &dc_dbb,
            &dc_dbb,
            &dc_dbb,
            &dc_dbb,
            &dc_dbb,
            &dc_dbbb,
            &dc_dbbb,
            &dc_dbbb,
            &dc_dbbb,
            &zero,
            &state.moments,
        )
        .expect("bbbb");

        let numeric_a = simpson_integral(cell.left, cell.right, 5000, |z| {
            eta_a(z) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_b = simpson_integral(cell.left, cell.right, 5000, |z| {
            eta_b(z) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_aa = simpson_integral(cell.left, cell.right, 5000, |z| {
            (eta_aa(z) - cell.eta(z) * eta_a(z) * eta_a(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_ab = simpson_integral(cell.left, cell.right, 5000, |z| {
            (eta_ab(z) - cell.eta(z) * eta_a(z) * eta_b(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_bb = simpson_integral(cell.left, cell.right, 5000, |z| {
            (eta_bb(z) - cell.eta(z) * eta_b(z) * eta_b(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_aaa = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (eta_aaa(z) - 3.0 * eta * eta_aa(z) * eta_a(z) + (eta * eta - 1.0) * eta_a(z).powi(3))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_aab = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let b_z = eta_b(z);
            (eta_aab(z) - eta * (eta_aa(z) * b_z + 2.0 * eta_ab(z) * a_z)
                + (eta * eta - 1.0) * a_z * a_z * b_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_abb = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let b_z = eta_b(z);
            (eta_abb(z) - eta * (2.0 * eta_ab(z) * b_z + eta_bb(z) * a_z)
                + (eta * eta - 1.0) * a_z * b_z * b_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bbb = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (eta_bbb(z) - 3.0 * eta * eta_bb(z) * eta_b(z) + (eta * eta - 1.0) * eta_b(z).powi(3))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_aaaa = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let eta_a_z = eta_a(z);
            let eta_aa_z = eta_aa(z);
            let eta_aaa_z = eta_aaa(z);
            (-eta * (4.0 * eta_aaa_z * eta_a_z + 3.0 * eta_aa_z * eta_aa_z)
                + (eta * eta - 1.0) * (6.0 * eta_aa_z * eta_a_z * eta_a_z)
                + (-eta * eta * eta + 3.0 * eta) * eta_a_z.powi(4))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_aaab = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let b_z = eta_b(z);
            let aa_z = eta_aa(z);
            let ab_z = eta_ab(z);
            let aaa_z = eta_aaa(z);
            let aab_z = eta_aab(z);
            (-eta * (aaa_z * b_z + 3.0 * aab_z * a_z + 3.0 * aa_z * ab_z)
                + (eta * eta - 1.0) * (3.0 * aa_z * a_z * b_z + 3.0 * ab_z * a_z * a_z)
                + (-eta * eta * eta + 3.0 * eta) * a_z.powi(3) * b_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_aabb = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let b_z = eta_b(z);
            let aa_z = eta_aa(z);
            let ab_z = eta_ab(z);
            let bb_z = eta_bb(z);
            let aab_z = eta_aab(z);
            let abb_z = eta_abb(z);
            (-eta * (2.0 * aab_z * b_z + 2.0 * abb_z * a_z + aa_z * bb_z + 2.0 * ab_z * ab_z)
                + (eta * eta - 1.0)
                    * (aa_z * b_z * b_z + 4.0 * ab_z * a_z * b_z + bb_z * a_z * a_z)
                + (-eta * eta * eta + 3.0 * eta) * a_z * a_z * b_z * b_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_abbb = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let b_z = eta_b(z);
            let ab_z = eta_ab(z);
            let bb_z = eta_bb(z);
            let abb_z = eta_abb(z);
            let bbb_z = eta_bbb(z);
            (-eta * (3.0 * abb_z * b_z + bbb_z * a_z + 3.0 * ab_z * bb_z)
                + (eta * eta - 1.0) * (3.0 * ab_z * b_z * b_z + 3.0 * bb_z * a_z * b_z)
                + (-eta * eta * eta + 3.0 * eta) * a_z * b_z.powi(3))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bbbb = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let eta_b_z = eta_b(z);
            let eta_bb_z = eta_bb(z);
            let eta_bbb_z = eta_bbb(z);
            (-eta * (4.0 * eta_bbb_z * eta_b_z + 3.0 * eta_bb_z * eta_bb_z)
                + (eta * eta - 1.0) * (6.0 * eta_bb_z * eta_b_z * eta_b_z)
                + (-eta * eta * eta + 3.0 * eta) * eta_b_z.powi(4))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });

        assert!((exact_a - numeric_a).abs() < 1e-8);
        assert!((exact_b - numeric_b).abs() < 1e-8);
        assert!((exact_aa - numeric_aa).abs() < 1e-8);
        assert!((exact_ab - numeric_ab).abs() < 1e-8);
        assert!((exact_bb - numeric_bb).abs() < 1e-8);
        assert!((exact_aaa - numeric_aaa).abs() < 2e-7);
        assert!((exact_aab - numeric_aab).abs() < 2e-7);
        assert!((exact_abb - numeric_abb).abs() < 2e-7);
        assert!((exact_bbb - numeric_bbb).abs() < 2e-7);
        assert!((exact_aaaa - numeric_aaaa).abs() < 2e-6);
        assert!((exact_aaab - numeric_aaab).abs() < 2e-6);
        assert!((exact_aabb - numeric_aabb).abs() < 2e-6);
        assert!((exact_abbb - numeric_abbb).abs() < 2e-6);
        assert!((exact_bbbb - numeric_bbbb).abs() < 2e-6);
    }

    #[test]
    fn link_basis_cell_derivatives_match_exact_integrands() {
        let score_span = LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: 0.08,
            c1: -0.03,
            c2: 0.02,
            c3: -0.01,
        };
        let link_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: -0.05,
            c1: 0.04,
            c2: -0.02,
            c3: 0.015,
        };
        let link_basis_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: 0.02,
            c1: -0.01,
            c2: 0.03,
            c3: -0.02,
        };
        let a = 0.3;
        let b = -0.7;
        let coeffs = denested_cell_coefficients(score_span, link_span, a, b);
        let cell = DenestedCubicCell {
            left: score_span.left,
            right: score_span.right,
            c0: coeffs[0],
            c1: coeffs[1],
            c2: coeffs[2],
            c3: coeffs[3],
        };
        let state = evaluate_cell_moments(cell, 24).expect("cell moments");
        let (dc_da, dc_db) = denested_cell_coefficient_partials(score_span, link_span, a, b);
        let second_partials = denested_cell_second_partials(score_span, link_span, a, b);
        let dc_daa = second_partials.0;
        let dc_dab = second_partials.1;
        let dc_dbb = second_partials.2;
        let denested_third = denested_cell_third_partials(link_span);
        let dc_daaa = denested_third.0;
        let dc_dbbb = denested_third.3;

        let coeff_w = link_basis_cell_coefficients(link_basis_span, a, b);
        let (coeff_aw, coeff_bw) = link_basis_cell_coefficient_partials(link_basis_span, a, b);
        let (coeff_aaw, coeff_abw, coeff_bbw) =
            link_basis_cell_second_partials(link_basis_span, a, b);
        let link_basis_third = link_basis_cell_third_partials(link_basis_span);
        let coeff_aaaw = link_basis_third.0;
        let coeff_bbbw = link_basis_third.3;
        let zero = [0.0; 4];
        let basis_third = 6.0 * link_basis_span.c3;

        let eta_a = |z: f64| 1.0 + link_span.first_derivative(a + b * z);
        let eta_b = |z: f64| z + score_span.evaluate(z) + z * link_span.first_derivative(a + b * z);
        let eta_aa = |z: f64| link_span.second_derivative(a + b * z);
        let eta_ab = |z: f64| z * link_span.second_derivative(a + b * z);
        let eta_bb = |z: f64| z * z * link_span.second_derivative(a + b * z);
        let eta_w = |z: f64| link_basis_span.evaluate(a + b * z);
        let eta_aw = |z: f64| link_basis_span.first_derivative(a + b * z);
        let eta_bw = |z: f64| z * link_basis_span.first_derivative(a + b * z);
        let eta_aaw = |z: f64| link_basis_span.second_derivative(a + b * z);
        let eta_abw = |z: f64| z * link_basis_span.second_derivative(a + b * z);
        let eta_bbw = |z: f64| z * z * link_basis_span.second_derivative(a + b * z);
        let eta_aaaw = |z: f64| basis_third + 0.0 * z;
        let eta_bbbw = |z: f64| z * z * z * basis_third;

        let exact_w = cell_first_derivative_from_moments(&coeff_w, &state.moments).expect("w");
        let exact_aw =
            cell_second_derivative_from_moments(cell, &dc_da, &coeff_w, &coeff_aw, &state.moments)
                .expect("aw");
        let exact_bw =
            cell_second_derivative_from_moments(cell, &dc_db, &coeff_w, &coeff_bw, &state.moments)
                .expect("bw");
        let exact_ww =
            cell_second_derivative_from_moments(cell, &coeff_w, &coeff_w, &zero, &state.moments)
                .expect("ww");
        let exact_aaw = cell_third_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &coeff_w,
            &dc_daa,
            &coeff_aw,
            &coeff_aw,
            &coeff_aaw,
            &state.moments,
        )
        .expect("aaw");
        let exact_abw = cell_third_derivative_from_moments(
            cell,
            &dc_da,
            &dc_db,
            &coeff_w,
            &dc_dab,
            &coeff_aw,
            &coeff_bw,
            &coeff_abw,
            &state.moments,
        )
        .expect("abw");
        let exact_bbw = cell_third_derivative_from_moments(
            cell,
            &dc_db,
            &dc_db,
            &coeff_w,
            &dc_dbb,
            &coeff_bw,
            &coeff_bw,
            &coeff_bbw,
            &state.moments,
        )
        .expect("bbw");
        let exact_www = cell_third_derivative_from_moments(
            cell,
            &coeff_w,
            &coeff_w,
            &coeff_w,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("www");
        let exact_aaaw = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &dc_da,
            &coeff_w,
            &dc_daa,
            &dc_daa,
            &coeff_aw,
            &dc_daa,
            &coeff_aw,
            &coeff_aw,
            &dc_daaa,
            &coeff_aaw,
            &coeff_aaw,
            &coeff_aaw,
            &coeff_aaaw,
            &state.moments,
        )
        .expect("aaaw");
        let exact_aaww = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &coeff_w,
            &coeff_w,
            &dc_daa,
            &coeff_aw,
            &coeff_aw,
            &coeff_aw,
            &coeff_aw,
            &zero,
            &coeff_aaw,
            &coeff_aaw,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("aaww");
        let exact_abww = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_db,
            &coeff_w,
            &coeff_w,
            &dc_dab,
            &coeff_aw,
            &coeff_aw,
            &coeff_bw,
            &coeff_bw,
            &zero,
            &coeff_abw,
            &coeff_abw,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("abww");
        let exact_bbww = cell_fourth_derivative_from_moments(
            cell,
            &dc_db,
            &dc_db,
            &coeff_w,
            &coeff_w,
            &dc_dbb,
            &coeff_bw,
            &coeff_bw,
            &coeff_bw,
            &coeff_bw,
            &zero,
            &coeff_bbw,
            &coeff_bbw,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("bbww");
        let exact_bbbw = cell_fourth_derivative_from_moments(
            cell,
            &dc_db,
            &dc_db,
            &dc_db,
            &coeff_w,
            &dc_dbb,
            &dc_dbb,
            &coeff_bw,
            &dc_dbb,
            &coeff_bw,
            &coeff_bw,
            &dc_dbbb,
            &coeff_bbw,
            &coeff_bbw,
            &coeff_bbw,
            &coeff_bbbw,
            &state.moments,
        )
        .expect("bbbw");
        let exact_wwww = cell_fourth_derivative_from_moments(
            cell,
            &coeff_w,
            &coeff_w,
            &coeff_w,
            &coeff_w,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("wwww");

        let numeric_w = simpson_integral(cell.left, cell.right, 5000, |z| {
            eta_w(z) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_aw = simpson_integral(cell.left, cell.right, 5000, |z| {
            (eta_aw(z) - cell.eta(z) * eta_a(z) * eta_w(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_bw = simpson_integral(cell.left, cell.right, 5000, |z| {
            (eta_bw(z) - cell.eta(z) * eta_b(z) * eta_w(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_ww = simpson_integral(cell.left, cell.right, 5000, |z| {
            (-cell.eta(z) * eta_w(z) * eta_w(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_aaw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let w_z = eta_w(z);
            let a_z = eta_a(z);
            (eta_aaw(z) - eta * (eta_aa(z) * w_z + 2.0 * eta_aw(z) * a_z)
                + (eta * eta - 1.0) * a_z * a_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_abw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let w_z = eta_w(z);
            let a_z = eta_a(z);
            let b_z = eta_b(z);
            (eta_abw(z) - eta * (eta_ab(z) * w_z + eta_aw(z) * b_z + eta_bw(z) * a_z)
                + (eta * eta - 1.0) * a_z * b_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bbw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let w_z = eta_w(z);
            let b_z = eta_b(z);
            (eta_bbw(z) - eta * (eta_bb(z) * w_z + 2.0 * eta_bw(z) * b_z)
                + (eta * eta - 1.0) * b_z * b_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_www = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let w_z = eta_w(z);
            ((eta * eta - 1.0) * w_z * w_z * w_z) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_aaaw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let w_z = eta_w(z);
            let aa_z = eta_aa(z);
            let aw_z = eta_aw(z);
            (eta_aaaw(z)
                - eta * ((dc_daaa[0] + 0.0 * z) * w_z + 3.0 * eta_aaw(z) * a_z + 3.0 * aa_z * aw_z)
                + (eta * eta - 1.0) * (3.0 * aa_z * a_z * w_z + 3.0 * aw_z * a_z * a_z)
                + (-eta * eta * eta + 3.0 * eta) * a_z * a_z * a_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_aaww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let w_z = eta_w(z);
            let aw_z = eta_aw(z);
            (-(2.0 * eta * (eta_aaw(z) * w_z + aw_z * aw_z))
                + (eta * eta - 1.0) * (eta_aa(z) * w_z * w_z + 4.0 * aw_z * a_z * w_z)
                + (-eta * eta * eta + 3.0 * eta) * a_z * a_z * w_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_abww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let b_z = eta_b(z);
            let w_z = eta_w(z);
            let aw_z = eta_aw(z);
            let bw_z = eta_bw(z);
            (-(2.0 * eta * (eta_abw(z) * w_z + aw_z * bw_z))
                + (eta * eta - 1.0)
                    * (eta_ab(z) * w_z * w_z + 2.0 * aw_z * b_z * w_z + 2.0 * bw_z * a_z * w_z)
                + (-eta * eta * eta + 3.0 * eta) * a_z * b_z * w_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bbww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let b_z = eta_b(z);
            let w_z = eta_w(z);
            let bw_z = eta_bw(z);
            (-(2.0 * eta * (eta_bbw(z) * w_z + bw_z * bw_z))
                + (eta * eta - 1.0) * (eta_bb(z) * w_z * w_z + 4.0 * bw_z * b_z * w_z)
                + (-eta * eta * eta + 3.0 * eta) * b_z * b_z * w_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bbbw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let b_z = eta_b(z);
            let w_z = eta_w(z);
            let bb_z = eta_bb(z);
            let bw_z = eta_bw(z);
            (eta_bbbw(z)
                - eta
                    * ((dc_dbbb[3] * z * z * z) * w_z + 3.0 * eta_bbw(z) * b_z + 3.0 * bb_z * bw_z)
                + (eta * eta - 1.0) * (3.0 * bb_z * b_z * w_z + 3.0 * bw_z * b_z * b_z)
                + (-eta * eta * eta + 3.0 * eta) * b_z * b_z * b_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_wwww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let w_z = eta_w(z);
            ((-eta * eta * eta + 3.0 * eta) * w_z * w_z * w_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });

        assert!((exact_w - numeric_w).abs() < 1e-8);
        assert!((exact_aw - numeric_aw).abs() < 1e-7);
        assert!((exact_bw - numeric_bw).abs() < 1e-7);
        assert!((exact_ww - numeric_ww).abs() < 1e-7);
        assert!((exact_aaw - numeric_aaw).abs() < 2e-6);
        assert!((exact_abw - numeric_abw).abs() < 2e-6);
        assert!((exact_bbw - numeric_bbw).abs() < 2e-6);
        assert!((exact_www - numeric_www).abs() < 2e-6);
        assert!((exact_aaaw - numeric_aaaw).abs() < 3e-6);
        assert!((exact_aaww - numeric_aaww).abs() < 3e-6);
        assert!((exact_abww - numeric_abww).abs() < 3e-6);
        assert!((exact_bbww - numeric_bbww).abs() < 3e-6);
        assert!((exact_bbbw - numeric_bbbw).abs() < 3e-6);
        assert!((exact_wwww - numeric_wwww).abs() < 3e-6);
    }

    #[test]
    fn score_basis_cell_derivatives_match_exact_integrands() {
        let score_span = LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: 0.08,
            c1: -0.03,
            c2: 0.02,
            c3: -0.01,
        };
        let score_basis_span = LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: -0.04,
            c1: 0.06,
            c2: -0.01,
            c3: 0.02,
        };
        let link_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: -0.05,
            c1: 0.04,
            c2: -0.02,
            c3: 0.015,
        };
        let a = 0.3;
        let b = -0.7;
        let coeffs = denested_cell_coefficients(score_span, link_span, a, b);
        let cell = DenestedCubicCell {
            left: score_span.left,
            right: score_span.right,
            c0: coeffs[0],
            c1: coeffs[1],
            c2: coeffs[2],
            c3: coeffs[3],
        };
        let state = evaluate_cell_moments(cell, 24).expect("cell moments");
        let (dc_da, dc_db) = denested_cell_coefficient_partials(score_span, link_span, a, b);
        let second_partials = denested_cell_second_partials(score_span, link_span, a, b);
        let dc_daa = second_partials.0;
        let dc_dab = second_partials.1;
        let dc_dbb = second_partials.2;
        let denested_third = denested_cell_third_partials(link_span);
        let dc_dbbb = denested_third.3;

        let coeff_h = score_basis_cell_coefficients(score_basis_span, b);
        let coeff_bh = score_basis_cell_coefficients(score_basis_span, 1.0);
        let zero = [0.0; 4];

        let eta_a = |z: f64| 1.0 + link_span.first_derivative(a + b * z);
        let eta_b = |z: f64| z + score_span.evaluate(z) + z * link_span.first_derivative(a + b * z);
        let eta_ab = |z: f64| z * link_span.second_derivative(a + b * z);
        let eta_bb = |z: f64| z * z * link_span.second_derivative(a + b * z);
        let eta_h = |z: f64| b * score_basis_span.evaluate(z);
        let eta_bh = |z: f64| score_basis_span.evaluate(z);

        let exact_h = cell_first_derivative_from_moments(&coeff_h, &state.moments).expect("h");
        let exact_ah =
            cell_second_derivative_from_moments(cell, &dc_da, &coeff_h, &zero, &state.moments)
                .expect("ah");
        let exact_bh =
            cell_second_derivative_from_moments(cell, &dc_db, &coeff_h, &coeff_bh, &state.moments)
                .expect("bh");
        let exact_hh =
            cell_second_derivative_from_moments(cell, &coeff_h, &coeff_h, &zero, &state.moments)
                .expect("hh");
        let exact_abh = cell_third_derivative_from_moments(
            cell,
            &dc_da,
            &dc_db,
            &coeff_h,
            &dc_dab,
            &zero,
            &coeff_bh,
            &zero,
            &state.moments,
        )
        .expect("abh");
        let exact_bbh = cell_third_derivative_from_moments(
            cell,
            &dc_db,
            &dc_db,
            &coeff_h,
            &dc_dbb,
            &coeff_bh,
            &coeff_bh,
            &zero,
            &state.moments,
        )
        .expect("bbh");
        let exact_bhh = cell_third_derivative_from_moments(
            cell,
            &dc_db,
            &coeff_h,
            &coeff_h,
            &coeff_bh,
            &coeff_bh,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("bhh");
        let exact_hhh = cell_third_derivative_from_moments(
            cell,
            &coeff_h,
            &coeff_h,
            &coeff_h,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("hhh");
        let exact_bbbh = cell_fourth_derivative_from_moments(
            cell,
            &dc_db,
            &dc_db,
            &dc_db,
            &coeff_h,
            &dc_dbb,
            &dc_dbb,
            &coeff_bh,
            &dc_dbb,
            &coeff_bh,
            &coeff_bh,
            &dc_dbbb,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("bbbh");
        let exact_aahh = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &coeff_h,
            &coeff_h,
            &dc_daa,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("aahh");
        let exact_abhh = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_db,
            &coeff_h,
            &coeff_h,
            &dc_dab,
            &zero,
            &zero,
            &coeff_bh,
            &coeff_bh,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("abhh");
        let exact_bbhh = cell_fourth_derivative_from_moments(
            cell,
            &dc_db,
            &dc_db,
            &coeff_h,
            &coeff_h,
            &dc_dbb,
            &coeff_bh,
            &coeff_bh,
            &coeff_bh,
            &coeff_bh,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("bbhh");
        let exact_bhhh = cell_fourth_derivative_from_moments(
            cell,
            &dc_db,
            &coeff_h,
            &coeff_h,
            &coeff_h,
            &coeff_bh,
            &coeff_bh,
            &coeff_bh,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("bhhh");
        let exact_hhhh = cell_fourth_derivative_from_moments(
            cell,
            &coeff_h,
            &coeff_h,
            &coeff_h,
            &coeff_h,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("hhhh");

        let numeric_h = simpson_integral(cell.left, cell.right, 5000, |z| {
            eta_h(z) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_ah = simpson_integral(cell.left, cell.right, 5000, |z| {
            (-cell.eta(z) * eta_a(z) * eta_h(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_bh = simpson_integral(cell.left, cell.right, 5000, |z| {
            (eta_bh(z) - cell.eta(z) * eta_b(z) * eta_h(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_hh = simpson_integral(cell.left, cell.right, 5000, |z| {
            (-cell.eta(z) * eta_h(z) * eta_h(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_abh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (-(eta * (eta_ab(z) * eta_h(z) + eta_bh(z) * eta_a(z)))
                + (eta * eta - 1.0) * eta_a(z) * eta_b(z) * eta_h(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bbh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (-(eta * (eta_bb(z) * eta_h(z) + 2.0 * eta_bh(z) * eta_b(z)))
                + (eta * eta - 1.0) * eta_b(z) * eta_b(z) * eta_h(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bhh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (-(2.0 * eta * eta_bh(z) * eta_h(z))
                + (eta * eta - 1.0) * eta_b(z) * eta_h(z) * eta_h(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_hhh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            ((eta * eta - 1.0) * eta_h(z) * eta_h(z) * eta_h(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_bbbh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let b_z = eta_b(z);
            let h_z = eta_h(z);
            let bb_z = eta_bb(z);
            let bh_z = eta_bh(z);
            (-(eta * ((dc_dbbb[3] * z * z * z) * h_z + 3.0 * bb_z * bh_z))
                + (eta * eta - 1.0) * (3.0 * bb_z * b_z * h_z + 3.0 * bh_z * b_z * b_z)
                + (-eta * eta * eta + 3.0 * eta) * b_z * b_z * b_z * h_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_aahh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let h_z = eta_h(z);
            ((eta * eta - 1.0) * polynomial_value(&dc_daa, z) * h_z * h_z
                + (-eta * eta * eta + 3.0 * eta) * a_z * a_z * h_z * h_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_abhh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let a_z = eta_a(z);
            let b_z = eta_b(z);
            let h_z = eta_h(z);
            ((eta * eta - 1.0) * (eta_ab(z) * h_z * h_z + 2.0 * eta_bh(z) * a_z * h_z)
                + (-eta * eta * eta + 3.0 * eta) * a_z * b_z * h_z * h_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bbhh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let b_z = eta_b(z);
            let h_z = eta_h(z);
            let bh_z = eta_bh(z);
            (-(2.0 * eta * bh_z * bh_z)
                + (eta * eta - 1.0) * (eta_bb(z) * h_z * h_z + 4.0 * bh_z * b_z * h_z)
                + (-eta * eta * eta + 3.0 * eta) * b_z * b_z * h_z * h_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bhhh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let h_z = eta_h(z);
            (-(eta * (3.0 * eta_bh(z) * h_z * h_z))
                + (eta * eta - 1.0) * (3.0 * eta_bh(z) * h_z * h_z)
                + (-eta * eta * eta + 3.0 * eta) * eta_b(z) * h_z * h_z * h_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_hhhh = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let h_z = eta_h(z);
            ((-eta * eta * eta + 3.0 * eta) * h_z * h_z * h_z * h_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });

        assert!((exact_h - numeric_h).abs() < 1e-8);
        assert!((exact_ah - numeric_ah).abs() < 1e-7);
        assert!((exact_bh - numeric_bh).abs() < 1e-7);
        assert!((exact_hh - numeric_hh).abs() < 1e-7);
        assert!((exact_abh - numeric_abh).abs() < 2e-6);
        assert!((exact_bbh - numeric_bbh).abs() < 2e-6);
        assert!((exact_bhh - numeric_bhh).abs() < 2e-6);
        assert!((exact_hhh - numeric_hhh).abs() < 2e-6);
        assert!((exact_bbbh - numeric_bbbh).abs() < 3e-6);
        assert!((exact_aahh - numeric_aahh).abs() < 3e-6);
        assert!((exact_abhh - numeric_abhh).abs() < 3e-6);
        assert!((exact_bbhh - numeric_bbhh).abs() < 3e-6);
        assert!((exact_bhhh - numeric_bhhh).abs() < 3e-6);
        assert!((exact_hhhh - numeric_hhhh).abs() < 3e-6);
    }

    #[test]
    fn cross_basis_cell_derivatives_match_exact_integrands() {
        let score_span = LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: 0.08,
            c1: -0.03,
            c2: 0.02,
            c3: -0.01,
        };
        let score_basis_span = LocalSpanCubic {
            left: -0.75,
            right: 0.25,
            c0: -0.04,
            c1: 0.06,
            c2: -0.01,
            c3: 0.02,
        };
        let link_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: -0.05,
            c1: 0.04,
            c2: -0.02,
            c3: 0.015,
        };
        let link_basis_span = LocalSpanCubic {
            left: -0.6,
            right: 0.9,
            c0: 0.02,
            c1: -0.01,
            c2: 0.03,
            c3: -0.02,
        };
        let a = 0.3;
        let b = -0.7;
        let coeffs = denested_cell_coefficients(score_span, link_span, a, b);
        let cell = DenestedCubicCell {
            left: score_span.left,
            right: score_span.right,
            c0: coeffs[0],
            c1: coeffs[1],
            c2: coeffs[2],
            c3: coeffs[3],
        };
        let state = evaluate_cell_moments(cell, 24).expect("cell moments");
        let (dc_da, dc_db) = denested_cell_coefficient_partials(score_span, link_span, a, b);
        let (dc_daa, dc_dab, _) = denested_cell_second_partials(score_span, link_span, a, b);

        let coeff_h = score_basis_cell_coefficients(score_basis_span, b);
        let coeff_bh = score_basis_cell_coefficients(score_basis_span, 1.0);
        let coeff_w = link_basis_cell_coefficients(link_basis_span, a, b);
        let (coeff_aw, coeff_bw) = link_basis_cell_coefficient_partials(link_basis_span, a, b);
        let (coeff_aaw, coeff_abw, _) = link_basis_cell_second_partials(link_basis_span, a, b);
        let zero = [0.0; 4];

        let eta_a = |z: f64| 1.0 + link_span.first_derivative(a + b * z);
        let eta_b = |z: f64| z + score_span.evaluate(z) + z * link_span.first_derivative(a + b * z);
        let eta_h = |z: f64| b * score_basis_span.evaluate(z);
        let eta_bh = |z: f64| score_basis_span.evaluate(z);
        let eta_w = |z: f64| link_basis_span.evaluate(a + b * z);
        let eta_ab = |z: f64| z * link_span.second_derivative(a + b * z);
        let eta_aw = |z: f64| link_basis_span.first_derivative(a + b * z);
        let eta_bw = |z: f64| z * link_basis_span.first_derivative(a + b * z);

        let exact_hw =
            cell_second_derivative_from_moments(cell, &coeff_h, &coeff_w, &zero, &state.moments)
                .expect("hw");
        let exact_ahw = cell_third_derivative_from_moments(
            cell,
            &dc_da,
            &coeff_h,
            &coeff_w,
            &zero,
            &coeff_aw,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("ahw");
        let exact_bhw = cell_third_derivative_from_moments(
            cell,
            &dc_db,
            &coeff_h,
            &coeff_w,
            &coeff_bh,
            &coeff_bw,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("bhw");
        let exact_hhw = cell_third_derivative_from_moments(
            cell,
            &coeff_h,
            &coeff_h,
            &coeff_w,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("hhw");
        let exact_hww = cell_third_derivative_from_moments(
            cell,
            &coeff_h,
            &coeff_w,
            &coeff_w,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("hww");
        let exact_aahw = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_da,
            &coeff_h,
            &coeff_w,
            &dc_daa,
            &zero,
            &coeff_aw,
            &zero,
            &coeff_aw,
            &zero,
            &zero,
            &coeff_aaw,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("aahw");
        let exact_hhww = cell_fourth_derivative_from_moments(
            cell,
            &coeff_h,
            &coeff_h,
            &coeff_w,
            &coeff_w,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("hhww");
        let exact_hhhw = cell_fourth_derivative_from_moments(
            cell,
            &coeff_h,
            &coeff_h,
            &coeff_h,
            &coeff_w,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("hhhw");
        let exact_abhw = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &dc_db,
            &coeff_h,
            &coeff_w,
            &dc_dab,
            &zero,
            &coeff_aw,
            &coeff_bh,
            &coeff_bw,
            &zero,
            &zero,
            &coeff_abw,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("abhw");
        let exact_ahww = cell_fourth_derivative_from_moments(
            cell,
            &dc_da,
            &coeff_h,
            &coeff_w,
            &coeff_w,
            &zero,
            &coeff_aw,
            &coeff_aw,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("ahww");
        let exact_bhww = cell_fourth_derivative_from_moments(
            cell,
            &dc_db,
            &coeff_h,
            &coeff_w,
            &coeff_w,
            &coeff_bh,
            &coeff_bw,
            &coeff_bw,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("bhww");
        let exact_hwww = cell_fourth_derivative_from_moments(
            cell,
            &coeff_h,
            &coeff_w,
            &coeff_w,
            &coeff_w,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &zero,
            &state.moments,
        )
        .expect("hwww");

        let numeric_hw = simpson_integral(cell.left, cell.right, 5000, |z| {
            (-cell.eta(z) * eta_h(z) * eta_w(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_ahw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (-(eta * eta_aw(z) * eta_h(z)) + (eta * eta - 1.0) * eta_a(z) * eta_h(z) * eta_w(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bhw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (-(eta * (eta_bh(z) * eta_w(z) + eta_bw(z) * eta_h(z)))
                + (eta * eta - 1.0) * eta_b(z) * eta_h(z) * eta_w(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_hhw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            ((eta * eta - 1.0) * eta_h(z) * eta_h(z) * eta_w(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_hww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            ((eta * eta - 1.0) * eta_h(z) * eta_w(z) * eta_w(z)) * (-cell.q(z)).exp() * INV_TWO_PI
        });
        let numeric_aahw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (-(eta * polynomial_value(&coeff_aaw, z) * eta_h(z))
                + (eta * eta - 1.0)
                    * (polynomial_value(&dc_daa, z) * eta_h(z) * eta_w(z)
                        + 2.0 * eta_aw(z) * eta_a(z) * eta_h(z))
                + (-eta * eta * eta + 3.0 * eta) * eta_a(z) * eta_a(z) * eta_h(z) * eta_w(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_hhww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            ((-eta * eta * eta + 3.0 * eta) * eta_h(z) * eta_h(z) * eta_w(z) * eta_w(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_hhhw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            ((-eta * eta * eta + 3.0 * eta) * eta_h(z) * eta_h(z) * eta_h(z) * eta_w(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_abhw = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (-(eta * polynomial_value(&coeff_abw, z) * eta_h(z) + eta * eta_aw(z) * eta_bh(z))
                + (eta * eta - 1.0)
                    * (eta_ab(z) * eta_h(z) * eta_w(z)
                        + eta_aw(z) * eta_b(z) * eta_h(z)
                        + eta_bh(z) * eta_a(z) * eta_w(z)
                        + eta_bw(z) * eta_a(z) * eta_h(z))
                + (-eta * eta * eta + 3.0 * eta) * eta_a(z) * eta_b(z) * eta_h(z) * eta_w(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_ahww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            (2.0 * (eta * eta - 1.0) * eta_aw(z) * eta_h(z) * eta_w(z)
                + (-eta * eta * eta + 3.0 * eta) * eta_a(z) * eta_h(z) * eta_w(z) * eta_w(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_bhww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            let h_z = eta_h(z);
            let w_z = eta_w(z);
            ((eta * eta - 1.0) * (eta_bh(z) * w_z * w_z + 2.0 * eta_bw(z) * h_z * w_z)
                + (-eta * eta * eta + 3.0 * eta) * eta_b(z) * h_z * w_z * w_z)
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });
        let numeric_hwww = simpson_integral(cell.left, cell.right, 5000, |z| {
            let eta = cell.eta(z);
            ((-eta * eta * eta + 3.0 * eta) * eta_h(z) * eta_w(z) * eta_w(z) * eta_w(z))
                * (-cell.q(z)).exp()
                * INV_TWO_PI
        });

        assert!((exact_hw - numeric_hw).abs() < 1e-7);
        assert!((exact_ahw - numeric_ahw).abs() < 2e-6);
        assert!((exact_bhw - numeric_bhw).abs() < 2e-6);
        assert!((exact_hhw - numeric_hhw).abs() < 2e-6);
        assert!((exact_hww - numeric_hww).abs() < 2e-6);
        assert!((exact_aahw - numeric_aahw).abs() < 3e-6);
        assert!((exact_hhww - numeric_hhww).abs() < 3e-6);
        assert!((exact_hhhw - numeric_hhhw).abs() < 3e-6);
        assert!((exact_abhw - numeric_abhw).abs() < 3e-6);
        assert!((exact_ahww - numeric_ahww).abs() < 3e-6);
        assert!((exact_bhww - numeric_bhww).abs() < 3e-6);
        assert!((exact_hwww - numeric_hwww).abs() < 3e-6);
    }

    #[test]
    fn cell_moment_scratch_reuses_buffers_under_margslope_like_pressure() {
        let cells = [
            DenestedCubicCell {
                left: -1.2,
                right: -0.35,
                c0: 0.18,
                c1: 0.72,
                c2: -0.045,
                c3: 0.018,
            },
            DenestedCubicCell {
                left: -0.35,
                right: 0.48,
                c0: -0.08,
                c1: 0.91,
                c2: 0.038,
                c3: -0.014,
            },
            DenestedCubicCell {
                left: 0.48,
                right: 1.4,
                c0: 0.11,
                c1: 0.83,
                c2: 0.022,
                c3: 0.012,
            },
        ];
        let mut scratch = CellMomentScratch::with_capacity(MAX_AFFINE_ANCHOR_DEGREE);
        for cell in cells {
            let baseline = evaluate_cell_moments(cell, 9).expect("baseline moments");
            let scratch_state =
                evaluate_cell_moments_with_scratch(cell, 9, &mut scratch).expect("scratch moments");
            assert_eq!(baseline.branch, scratch_state.branch);
            assert!((baseline.value - scratch_state.value).abs() <= 1e-10);
            assert_eq!(baseline.moments.len(), scratch_state.moments.len());
            for (lhs, rhs) in baseline.moments.iter().zip(scratch_state.moments.iter()) {
                assert!((lhs - rhs).abs() <= 1e-10, "{lhs} vs {rhs}");
            }
        }

        reset_cell_moment_test_reallocs();
        let mut checksum = 0.0;
        for i in 0..5_000 {
            let cell = cells[i % cells.len()];
            let state = evaluate_cell_moments_with_scratch(cell, 9, &mut scratch)
                .expect("scratch moments under repeated pressure");
            checksum += state.value + state.moments[0] * 1e-12;
        }
        assert!(checksum.is_finite());
        assert_eq!(
            cell_moment_test_reallocs(),
            0,
            "scratch-backed inner cell-moment calls should not grow Vec buffers"
        );
    }

    #[test]
    fn evaluate_cell_moments_matches_numeric_integrals() {
        let cell = DenestedCubicCell {
            left: -0.9,
            right: 0.8,
            c0: 0.15,
            c1: -0.35,
            c2: 0.11,
            c3: -0.07,
        };
        let state = evaluate_cell_moments(cell, 6).expect("cell moments");
        let value_numeric = simpson_integral(cell.left, cell.right, 4000, |z| {
            super::normal_cdf(cell.eta(z)) * normal_pdf(z)
        });
        assert!((state.value - value_numeric).abs() < 1e-9);
        for degree in 0..=6 {
            let target = simpson_integral(cell.left, cell.right, 4000, |z| {
                z.powi(degree as i32) * (-cell.q(z)).exp()
            });
            assert!((state.moments[degree] - target).abs() < 1e-9);
        }
    }

    #[test]
    fn partition_builder_moves_link_preimages_with_intercept() {
        let score_breaks = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let link_breaks = [-1.5, -0.5, 0.5, 1.5];
        let score_span = |z: f64| {
            let left = if z < -1.0 {
                -2.0
            } else if z < 0.0 {
                -1.0
            } else if z < 1.0 {
                0.0
            } else {
                1.0
            };
            Ok(LocalSpanCubic {
                left,
                right: left + 1.0,
                c0: 0.1,
                c1: 0.2,
                c2: 0.0,
                c3: 0.0,
            })
        };
        let link_span = |u: f64| {
            let left = if u < -0.5 {
                -1.5
            } else if u < 0.5 {
                -0.5
            } else {
                0.5
            };
            Ok(LocalSpanCubic {
                left,
                right: left + 1.0,
                c0: -0.05,
                c1: 0.1,
                c2: 0.0,
                c3: 0.0,
            })
        };
        let cells_a0 = build_denested_partition_cells(
            0.25,
            0.9,
            &score_breaks,
            &link_breaks,
            score_span,
            link_span,
        )
        .expect("cells a0");
        let cells_a1 = build_denested_partition_cells(
            0.55,
            0.9,
            &score_breaks,
            &link_breaks,
            score_span,
            link_span,
        )
        .expect("cells a1");
        assert!(cells_a0.len() >= score_breaks.len() - 1);
        assert!(
            cells_a0
                .windows(2)
                .all(|w| (w[0].cell.right - w[1].cell.left).abs() <= 1e-12)
        );
        assert!(
            cells_a0
                .iter()
                .zip(cells_a1.iter())
                .any(|(lhs, rhs)| (lhs.cell.left - rhs.cell.left).abs() > 1e-10)
        );
        assert!(cells_a0.first().unwrap().cell.left.is_infinite());
        assert!(cells_a0.last().unwrap().cell.right.is_infinite());
    }

    #[test]
    fn partition_builder_without_breaks_returns_single_global_cell() {
        let cells = build_denested_partition_cells_with_tails(
            0.3,
            -0.4,
            &[],
            &[],
            |z| {
                if z.is_nan() {
                    return Err("probe z is NaN".to_string());
                }
                Ok(LocalSpanCubic {
                    left: 0.0,
                    right: 1.0,
                    c0: 0.0,
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                })
            },
            |u| {
                if u.is_nan() {
                    return Err("probe u is NaN".to_string());
                }
                Ok(LocalSpanCubic {
                    left: 0.0,
                    right: 1.0,
                    c0: 0.0,
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                })
            },
        )
        .expect("global cell");
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].cell.left, f64::NEG_INFINITY);
        assert_eq!(cells[0].cell.right, f64::INFINITY);
        assert!(cells[0].cell.c2.abs() < 1e-12);
        assert!(cells[0].cell.c3.abs() < 1e-12);
    }

    #[test]
    fn polynomial_integral_helper_matches_moment_sum() {
        let cell = DenestedCubicCell {
            left: -1.5,
            right: 1.25,
            c0: 0.2,
            c1: -0.4,
            c2: 0.15,
            c3: 0.03,
        };
        let state = evaluate_cell_moments(cell, 8).expect("cell moments");
        let coeffs = [1.5, -0.25, 0.75, 0.1];
        let expected = INV_TWO_PI
            * coeffs
                .iter()
                .enumerate()
                .map(|(idx, coeff)| coeff * state.moments[idx])
                .sum::<f64>();
        let got = cell_polynomial_integral_from_moments(&coeffs, &state.moments, "test poly")
            .expect("poly integral");
        assert!((got - expected).abs() < 1e-14);
    }

    #[test]
    fn batched_cell_moment_max_degree_matches_direct_non_affine_grid() {
        let cells = [
            DenestedCubicCell {
                left: -2.0,
                right: -0.25,
                c0: -0.7,
                c1: 0.8,
                c2: 0.015,
                c3: -0.004,
            },
            DenestedCubicCell {
                left: -0.5,
                right: 0.75,
                c0: 0.2,
                c1: -0.35,
                c2: -0.025,
                c3: 0.0,
            },
            DenestedCubicCell {
                left: 0.1,
                right: 1.6,
                c0: 0.4,
                c1: 0.25,
                c2: 0.01,
                c3: 0.006,
            },
            DenestedCubicCell {
                left: -1.25,
                right: 2.25,
                c0: -0.1,
                c1: 0.55,
                c2: -0.012,
                c3: 0.003,
            },
        ];
        for cell in cells {
            let branch = branch_cell(cell).expect("branch");
            if branch == ExactCellBranch::Affine {
                continue;
            }
            let batched =
                evaluate_non_affine_cell_state(cell, branch, 21).expect("degree-21 state");
            for degree in [9usize, 15, 21] {
                let direct =
                    evaluate_non_affine_cell_state(cell, branch, degree).expect("direct state");
                assert_eq!(batched.branch, direct.branch);
                let denom = direct.value.abs().max(1.0);
                assert!(((batched.value - direct.value).abs() / denom) < 1e-10);
                for k in 0..=degree {
                    let denom = direct.moments[k].abs().max(1.0);
                    let rel = (batched.moments[k] - direct.moments[k]).abs() / denom;
                    assert!(
                        rel < 1e-10,
                        "cell={cell:?} degree={degree} moment={k} rel={rel:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn derivative_moment_evaluator_matches_value_evaluator_moments() {
        let cells = [
            DenestedCubicCell {
                left: -2.0,
                right: -0.4,
                c0: 0.15,
                c1: -0.8,
                c2: 0.0,
                c3: 0.0,
            },
            DenestedCubicCell {
                left: -0.75,
                right: 1.4,
                c0: -0.25,
                c1: 0.6,
                c2: 0.12,
                c3: 0.0,
            },
            DenestedCubicCell {
                left: -1.1,
                right: 0.9,
                c0: 0.35,
                c1: -0.3,
                c2: 0.05,
                c3: -0.015,
            },
        ];
        for cell in cells {
            for degree in [4usize, 9, 15, 21] {
                let full = evaluate_cell_moments_uncached(cell, degree).expect("full moments");
                let derivative = evaluate_cell_derivative_moments_uncached(cell, degree)
                    .expect("derivative moments");
                assert_eq!(full.branch, derivative.branch);
                assert_eq!(full.moments.len(), derivative.moments.len());
                for k in 0..full.moments.len() {
                    assert_eq!(full.moments[k].to_bits(), derivative.moments[k].to_bits());
                }
            }
        }
    }

    #[test]
    fn cell_moment_lru_matches_uncached_non_affine_grid() {
        let cache = CellMomentLruCache::new(16 * 1024 * 1024);
        let stats = CellMomentCacheStats::default();
        let c0s = [-0.75, 0.0, 0.5];
        let c1s = [-1.2, 0.25, 1.1];
        let c2s = [-0.18, 0.07];
        let c3s = [0.0, 0.025];
        let bounds = [(-2.0, -0.5), (-0.25, 1.5)];
        let degrees = [4usize, 9, 15, 21];
        for &c0 in &c0s {
            for &c1 in &c1s {
                for &c2 in &c2s {
                    for &c3 in &c3s {
                        for &(left, right) in &bounds {
                            for &max_degree in &degrees {
                                let cell = DenestedCubicCell {
                                    left,
                                    right,
                                    c0,
                                    c1,
                                    c2,
                                    c3,
                                };
                                let branch = branch_cell(cell).expect("branch");
                                if branch == ExactCellBranch::Affine {
                                    continue;
                                }
                                let expected =
                                    evaluate_non_affine_cell_state(cell, branch, max_degree)
                                        .expect("uncached non-affine moments");
                                let got = evaluate_cell_moments_cached(
                                    cell,
                                    max_degree,
                                    &cache,
                                    Some(&stats),
                                )
                                .expect("cached moments");
                                assert_eq!(got.branch, expected.branch);
                                assert_eq!(got.moments.len(), max_degree + 1);
                                let denom = expected.value.abs().max(1.0);
                                assert!(
                                    ((got.value - expected.value).abs() / denom) < 1e-10,
                                    "value mismatch for {cell:?} degree {max_degree}: got {} expected {}",
                                    got.value,
                                    expected.value
                                );
                                for (idx, (&lhs, &rhs)) in
                                    got.moments.iter().zip(expected.moments.iter()).enumerate()
                                {
                                    let denom = rhs.abs().max(1.0);
                                    assert!(
                                        ((lhs - rhs).abs() / denom) < 1e-10,
                                        "moment {idx} mismatch for {cell:?} degree {max_degree}: got {lhs} expected {rhs}"
                                    );
                                }
                                let warm = evaluate_cell_moments_cached(
                                    cell,
                                    max_degree,
                                    &cache,
                                    Some(&stats),
                                )
                                .expect("warm cached moments");
                                assert_eq!(warm, got);
                            }
                        }
                    }
                }
            }
        }
        let (hits, misses) = stats.snapshot();
        assert!(hits > 0, "expected warm LRU hits");
        assert!(misses > 0, "expected cold LRU misses");
    }

    #[test]
    fn cell_moment_fingerprint_exact_cache_matches_current_evaluator() {
        let cells = [
            DenestedCubicCell {
                left: -1.75,
                right: -0.25,
                c0: 0.15,
                c1: -0.35,
                c2: 0.08,
                c3: -0.015,
            },
            DenestedCubicCell {
                left: -0.5,
                right: 0.8,
                c0: -0.2,
                c1: 0.45,
                c2: -0.12,
                c3: 0.025,
            },
            DenestedCubicCell {
                left: 0.1,
                right: 1.6,
                c0: 0.05,
                c1: 0.2,
                c2: 0.03,
                c3: 0.004,
            },
        ];
        let mut cache = std::collections::HashMap::new();
        for max_degree in [0usize, 3, 4, 9, 16] {
            for cell in cells {
                let baseline = evaluate_cell_moments(cell, max_degree).expect("baseline moments");
                let key = cell_moment_cache_key(cell, max_degree, 0.0);
                let cached = cache.entry(key).or_insert_with(|| {
                    evaluate_cell_moments(cell, max_degree).expect("cached moments")
                });
                assert_eq!(baseline.branch, cached.branch);
                assert_eq!(baseline.value.to_bits(), cached.value.to_bits());
                assert_eq!(baseline.moments.len(), cached.moments.len());
                for (lhs, rhs) in baseline.moments.iter().zip(cached.moments.iter()) {
                    assert_eq!(lhs.to_bits(), rhs.to_bits());
                }
            }
        }
    }

    #[test]
    fn fuzzy_cell_moment_fingerprint_error_scales_with_epsilon() {
        for epsilon in [1e-8, 1e-6] {
            let base = DenestedCubicCell {
                left: -1.25,
                right: 1.1,
                c0: 0.1,
                c1: -0.25,
                c2: 0.04,
                c3: -0.006,
            };
            let perturbed = DenestedCubicCell {
                left: base.left + 0.001 * epsilon,
                right: base.right - 0.001 * epsilon,
                c0: base.c0 + 0.001 * epsilon,
                c1: base.c1 - 0.001 * epsilon,
                c2: base.c2 + 0.001 * epsilon,
                c3: base.c3 - 0.001 * epsilon,
            };
            assert_eq!(
                cell_moment_cache_key(base, 9, epsilon),
                cell_moment_cache_key(perturbed, 9, epsilon)
            );
            let lhs = evaluate_cell_moments(base, 9).expect("base moments");
            let rhs = evaluate_cell_moments(perturbed, 9).expect("perturbed moments");
            let max_rel = lhs
                .moments
                .iter()
                .zip(rhs.moments.iter())
                .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1.0))
                .fold(0.0_f64, f64::max);
            assert!(
                max_rel <= 10.0 * epsilon,
                "epsilon={epsilon:.1e} max_rel={max_rel:.3e}"
            );
        }
    }

    /// Locks in numerical equivalence of the optimized
    /// `evaluate_non_affine_cell_state` against an inline reference
    /// implementation that mirrors the prior pre-fold structure
    /// (separate `cell.eta(z)` / `cell.q(z)` calls; post-loop
    /// `* half_width`; trailing `value_integral * half_width / sqrt(TAU)`).
    /// Any drift larger than 1e-13 relative would indicate the hot-path
    /// rewrite changed the math.
    #[test]
    fn non_affine_cell_state_matches_prefold_reference_to_1e_minus_13() {
        // Reference: byte-for-byte the structure of the previous
        // implementation. Kept local to this test to avoid leaking a second
        // public surface.
        fn reference(
            cell: DenestedCubicCell,
            branch: ExactCellBranch,
            max_degree: usize,
        ) -> CellMomentState {
            let mut moments: CellMomentVec = smallvec![0.0_f64; max_degree + 1];
            let mut value_integral = 0.0_f64;
            let center = 0.5 * (cell.left + cell.right);
            let half_width = 0.5 * (cell.right - cell.left);
            for (&node, &weight) in GL_NODES.iter().zip(GL_WEIGHTS.iter()) {
                let z = center + half_width * node;
                let eta = cell.eta(z);
                let moment_weight = weight * (-cell.q(z)).exp();
                let mut z_pow = 1.0_f64;
                for moment in &mut moments {
                    *moment = moment_weight.mul_add(z_pow, *moment);
                    z_pow *= z;
                }
                value_integral += weight * (-0.5 * z * z).exp() * normal_cdf(eta);
            }
            for moment in &mut moments {
                *moment *= half_width;
            }
            CellMomentState {
                branch,
                value: value_integral * half_width / (std::f64::consts::TAU).sqrt(),
                moments,
            }
        }

        // Hand-rolled inputs that cross both Quartic and Sextic branches and
        // exercise positive/negative coefficients, asymmetric intervals, and
        // a wide degree range (matches survival_marginal_slope's degree=9
        // production call as well as the bernoulli outer-step degree=24).
        let cells = [
            DenestedCubicCell {
                left: -1.25,
                right: -0.2,
                c0: -0.35,
                c1: 0.85,
                c2: 0.04,
                c3: -0.015,
            },
            DenestedCubicCell {
                left: -0.2,
                right: 0.55,
                c0: 0.12,
                c1: -0.65,
                c2: -0.025,
                c3: 0.02,
            },
            DenestedCubicCell {
                left: 0.55,
                right: 1.6,
                c0: 0.42,
                c1: 0.35,
                c2: 0.018,
                c3: 0.012,
            },
            DenestedCubicCell {
                left: -3.0,
                right: -1.0,
                c0: 1.7,
                c1: -0.4,
                c2: 0.11,
                c3: -0.07,
            },
        ];
        let degrees = [0_usize, 4, 9, 16, 24];
        for cell in cells {
            let branch = branch_cell(cell).expect("branch");
            assert_ne!(branch, ExactCellBranch::Affine);
            for max_degree in degrees {
                let actual = evaluate_non_affine_cell_state(cell, branch, max_degree)
                    .expect("optimized non-affine");
                let expected = reference(cell, branch, max_degree);
                assert_eq!(actual.branch, expected.branch);
                assert_eq!(actual.moments.len(), expected.moments.len());
                let denom_v = expected.value.abs().max(1.0);
                let rel_v = (actual.value - expected.value).abs() / denom_v;
                let actual_v = actual.value;
                let expected_v = expected.value;
                assert!(
                    rel_v <= 1e-13,
                    "value rel mismatch for {cell:?} degree {max_degree}: \
                     actual={actual_v:.17e} expected={expected_v:.17e} rel={rel_v:.3e}"
                );
                for (k, (lhs, rhs)) in actual
                    .moments
                    .iter()
                    .zip(expected.moments.iter())
                    .enumerate()
                {
                    let denom = rhs.abs().max(1.0);
                    let rel = (lhs - rhs).abs() / denom;
                    assert!(
                        rel <= 1e-13,
                        "moment {k} rel mismatch for {cell:?} degree {max_degree}: \
                         actual={lhs:.17e} expected={rhs:.17e} rel={rel:.3e}"
                    );
                }

                // Also lock in the derivative-state path on the same
                // inputs so the (parallel) edit there can't drift.
                let actual_deriv =
                    evaluate_non_affine_cell_derivative_state(cell, branch, max_degree)
                        .expect("optimized derivative");
                for (k, (lhs, rhs)) in actual_deriv
                    .moments
                    .iter()
                    .zip(expected.moments.iter())
                    .enumerate()
                {
                    let denom = rhs.abs().max(1.0);
                    let rel = (lhs - rhs).abs() / denom;
                    assert!(
                        rel <= 1e-13,
                        "deriv moment {k} rel mismatch for {cell:?} degree {max_degree}: \
                         actual={lhs:.17e} expected={rhs:.17e} rel={rel:.3e}"
                    );
                }
            }
        }
    }
}

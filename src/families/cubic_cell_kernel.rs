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
    -9.99980441172647394e-01,
    -9.99896947137859593e-01,
    -9.99746740811352286e-01,
    -9.99529798855885887e-01,
    -9.99246131667184456e-01,
    -9.98895757206325730e-01,
    -9.98478698538458942e-01,
    -9.97994983372793798e-01,
    -9.97444643938910747e-01,
    -9.96827716944091335e-01,
    -9.96144243555108666e-01,
    -9.95394269388595321e-01,
    -9.94577844504706765e-01,
    -9.93695023402088262e-01,
    -9.92745865013315298e-01,
    -9.91730432700432041e-01,
    -9.90648794250406084e-01,
    -9.89501021870408670e-01,
    -9.88287192182869867e-01,
    -9.87007386220281502e-01,
    -9.85661689419733311e-01,
    -9.84250191617171311e-01,
    -9.82772987041374280e-01,
    -9.81230174307644254e-01,
    -9.79621856411210135e-01,
    -9.77948140720341086e-01,
    -9.76209138969172385e-01,
    -9.74404967250239729e-01,
    -9.72535746006725654e-01,
    -9.70601600024415090e-01,
    -9.68602658423362795e-01,
    -9.66539054649271034e-01,
    -9.64410926464580154e-01,
    -9.62218415939269822e-01,
    -9.59961669441374177e-01,
    -9.57640837627209529e-01,
    -9.55256075431315965e-01,
    -9.52807542056114398e-01,
    -9.50295400961277070e-01,
    -9.47719819852815726e-01,
    -9.45080970671885123e-01,
    -9.42379029583304439e-01,
    -9.39614176963796344e-01,
    -9.36786597389945852e-01,
    -9.33896479625877518e-01,
    -9.30944016610653957e-01,
    -9.27929405445395594e-01,
    -9.24852847380122189e-01,
    -9.21714547800318140e-01,
    -9.18514716213220783e-01,
    -9.15253566233834026e-01,
    -9.11931315570668199e-01,
    -9.08548186011205461e-01,
    -9.05104403407094416e-01,
    -9.01600197659072178e-01,
    -8.98035802701616426e-01,
    -8.94411456487328782e-01,
    -8.90727400971049188e-01,
    -8.86983882093703380e-01,
    -8.83181149765884688e-01,
    -8.79319457851170050e-01,
    -8.75399064149172457e-01,
    -8.71420230378331162e-01,
    -8.67383222158439326e-01,
    -8.63288308992911979e-01,
    -8.59135764250794520e-01,
    -8.54925865148512654e-01,
    -8.50658892731366634e-01,
    -8.46335131854768274e-01,
    -8.41954871165225383e-01,
    -8.37518403081071505e-01,
    -8.33026023772945212e-01,
    -8.28478033144017800e-01,
    -8.23874734809972642e-01,
    -8.19216436078735954e-01,
    -8.14503447929961988e-01,
    -8.09736084994271965e-01,
    -8.04914665532250551e-01,
    -8.00039511413198845e-01,
    -7.95110948093647130e-01,
    -7.90129304595628024e-01,
    -7.85094913484711721e-01,
    -7.80008110847803970e-01,
    -7.74869236270710027e-01,
    -7.69678632815464447e-01,
    -7.64436646997428526e-01,
    -7.59143628762158018e-01,
    -7.53799931462041162e-01,
    -7.48405911832709436e-01,
    -7.42961929969222723e-01,
    -7.37468349302029869e-01,
    -7.31925536572706759e-01,
    -7.26333861809473347e-01,
    -7.20693698302491192e-01,
    -7.15005422578943173e-01,
    -7.09269414377897478e-01,
    -7.03486056624956668e-01,
    -6.97655735406694344e-01,
    -6.91778839944880763e-01,
    -6.85855762570498961e-01,
    -6.79886898697553366e-01,
    -6.73872646796673136e-01,
    -6.67813408368510220e-01,
    -6.61709587916936570e-01,
    -6.55561592922039971e-01,
    -6.49369833812921238e-01,
    -6.43134723940294806e-01,
    -6.36856679548894467e-01,
    -6.30536119749684931e-01,
    -6.24173466491883655e-01,
    -6.17769144534791259e-01,
    -6.11323581419436435e-01,
    -6.04837207440032887e-01,
    -5.98310455615254866e-01,
    -5.91743761659328626e-01,
    -5.85137563952945583e-01,
    -5.78492303513996498e-01,
    -5.71808423968130031e-01,
    -5.65086371519136876e-01,
    -5.58326594919162256e-01,
    -5.51529545438748214e-01,
    -5.44695676836706832e-01,
    -5.37825445329828900e-01,
    -5.30919309562427499e-01,
    -5.23977730575719391e-01,
    -5.17001171777047297e-01,
    -5.09990098908942868e-01,
    -5.02944980018035559e-01,
    -4.95866285423805842e-01,
    -4.88754487687187822e-01,
    -4.81610061579022086e-01,
    -4.74433484048360554e-01,
    -4.67225234190626393e-01,
    -4.59985793215630379e-01,
    -4.52715644415446372e-01,
    -4.45415273132147349e-01,
    -4.38085166725404984e-01,
    -4.30725814539954455e-01,
    -4.23337707872926516e-01,
    -4.15921339941049406e-01,
    -4.08477205847722791e-01,
    -4.01005802549965318e-01,
    -3.93507628825238598e-01,
    -3.85983185238150062e-01,
    -3.78432974107035858e-01,
    -3.70857499470427121e-01,
    -3.63257267053401112e-01,
    -3.55632784233820221e-01,
    -3.47984560008460064e-01,
    -3.40313104959029722e-01,
    -3.32618931218086611e-01,
    -3.24902552434846947e-01,
    -3.17164483740895842e-01,
    -3.09405241715797807e-01,
    -3.01625344352610902e-01,
    -2.93825311023306446e-01,
    -2.86005662444096753e-01,
    -2.78166920640672888e-01,
    -2.70309608913355315e-01,
    -2.62434251802159235e-01,
    -2.54541375051777319e-01,
    -2.46631505576481752e-01,
    -2.38705171424948626e-01,
    -2.30762901745006188e-01,
    -2.22805226748309937e-01,
    -2.14832677674946654e-01,
    -2.06845786757969746e-01,
    -1.98845087187868336e-01,
    -1.90831113076972453e-01,
    -1.82804399423796560e-01,
    -1.74765482077324119e-01,
    -1.66714897701235237e-01,
    -1.58653183738079934e-01,
    -1.50580878373399507e-01,
    -1.42498520499798137e-01,
    -1.34406649680967466e-01,
    -1.26305806115666303e-01,
    -1.18196530601657843e-01,
    -1.10079364499607035e-01,
    -1.01954849696940367e-01,
    -9.38235285716702838e-02,
    -8.56859439561871855e-02,
    -7.75426391010207716e-02,
    -6.93941576385737058e-02,
    -6.12410435468296183e-02,
    -5.30838411130381757e-02,
    -4.49230948973793967e-02,
    -3.67593496966098221e-02,
    -2.85931505076928473e-02,
    -2.04250424914157111e-02,
    -1.22555709359955384e-02,
    -4.08528122067686791e-03,
    4.08528122067686791e-03,
    1.22555709359955384e-02,
    2.04250424914157111e-02,
    2.85931505076928473e-02,
    3.67593496966098221e-02,
    4.49230948973793967e-02,
    5.30838411130381757e-02,
    6.12410435468296183e-02,
    6.93941576385737058e-02,
    7.75426391010207716e-02,
    8.56859439561871855e-02,
    9.38235285716702838e-02,
    1.01954849696940367e-01,
    1.10079364499607035e-01,
    1.18196530601657843e-01,
    1.26305806115666303e-01,
    1.34406649680967466e-01,
    1.42498520499798137e-01,
    1.50580878373399507e-01,
    1.58653183738079934e-01,
    1.66714897701235237e-01,
    1.74765482077324119e-01,
    1.82804399423796560e-01,
    1.90831113076972453e-01,
    1.98845087187868336e-01,
    2.06845786757969746e-01,
    2.14832677674946654e-01,
    2.22805226748309937e-01,
    2.30762901745006188e-01,
    2.38705171424948626e-01,
    2.46631505576481752e-01,
    2.54541375051777319e-01,
    2.62434251802159235e-01,
    2.70309608913355315e-01,
    2.78166920640672888e-01,
    2.86005662444096753e-01,
    2.93825311023306446e-01,
    3.01625344352610902e-01,
    3.09405241715797807e-01,
    3.17164483740895842e-01,
    3.24902552434846947e-01,
    3.32618931218086611e-01,
    3.40313104959029722e-01,
    3.47984560008460064e-01,
    3.55632784233820221e-01,
    3.63257267053401112e-01,
    3.70857499470427121e-01,
    3.78432974107035858e-01,
    3.85983185238150062e-01,
    3.93507628825238598e-01,
    4.01005802549965318e-01,
    4.08477205847722791e-01,
    4.15921339941049406e-01,
    4.23337707872926516e-01,
    4.30725814539954455e-01,
    4.38085166725404984e-01,
    4.45415273132147349e-01,
    4.52715644415446372e-01,
    4.59985793215630379e-01,
    4.67225234190626393e-01,
    4.74433484048360554e-01,
    4.81610061579022086e-01,
    4.88754487687187822e-01,
    4.95866285423805842e-01,
    5.02944980018035559e-01,
    5.09990098908942868e-01,
    5.17001171777047297e-01,
    5.23977730575719391e-01,
    5.30919309562427499e-01,
    5.37825445329828900e-01,
    5.44695676836706832e-01,
    5.51529545438748214e-01,
    5.58326594919162256e-01,
    5.65086371519136876e-01,
    5.71808423968130031e-01,
    5.78492303513996498e-01,
    5.85137563952945583e-01,
    5.91743761659328626e-01,
    5.98310455615254866e-01,
    6.04837207440032887e-01,
    6.11323581419436435e-01,
    6.17769144534791259e-01,
    6.24173466491883655e-01,
    6.30536119749684931e-01,
    6.36856679548894467e-01,
    6.43134723940294806e-01,
    6.49369833812921238e-01,
    6.55561592922039971e-01,
    6.61709587916936570e-01,
    6.67813408368510220e-01,
    6.73872646796673136e-01,
    6.79886898697553366e-01,
    6.85855762570498961e-01,
    6.91778839944880763e-01,
    6.97655735406694344e-01,
    7.03486056624956668e-01,
    7.09269414377897478e-01,
    7.15005422578943173e-01,
    7.20693698302491192e-01,
    7.26333861809473347e-01,
    7.31925536572706759e-01,
    7.37468349302029869e-01,
    7.42961929969222723e-01,
    7.48405911832709436e-01,
    7.53799931462041162e-01,
    7.59143628762158018e-01,
    7.64436646997428526e-01,
    7.69678632815464447e-01,
    7.74869236270710027e-01,
    7.80008110847803970e-01,
    7.85094913484711721e-01,
    7.90129304595628024e-01,
    7.95110948093647130e-01,
    8.00039511413198845e-01,
    8.04914665532250551e-01,
    8.09736084994271965e-01,
    8.14503447929961988e-01,
    8.19216436078735954e-01,
    8.23874734809972642e-01,
    8.28478033144017800e-01,
    8.33026023772945212e-01,
    8.37518403081071505e-01,
    8.41954871165225383e-01,
    8.46335131854768274e-01,
    8.50658892731366634e-01,
    8.54925865148512654e-01,
    8.59135764250794520e-01,
    8.63288308992911979e-01,
    8.67383222158439326e-01,
    8.71420230378331162e-01,
    8.75399064149172457e-01,
    8.79319457851170050e-01,
    8.83181149765884688e-01,
    8.86983882093703380e-01,
    8.90727400971049188e-01,
    8.94411456487328782e-01,
    8.98035802701616426e-01,
    9.01600197659072178e-01,
    9.05104403407094416e-01,
    9.08548186011205461e-01,
    9.11931315570668199e-01,
    9.15253566233834026e-01,
    9.18514716213220783e-01,
    9.21714547800318140e-01,
    9.24852847380122189e-01,
    9.27929405445395594e-01,
    9.30944016610653957e-01,
    9.33896479625877518e-01,
    9.36786597389945852e-01,
    9.39614176963796344e-01,
    9.42379029583304439e-01,
    9.45080970671885123e-01,
    9.47719819852815726e-01,
    9.50295400961277070e-01,
    9.52807542056114398e-01,
    9.55256075431315965e-01,
    9.57640837627209529e-01,
    9.59961669441374177e-01,
    9.62218415939269822e-01,
    9.64410926464580154e-01,
    9.66539054649271034e-01,
    9.68602658423362795e-01,
    9.70601600024415090e-01,
    9.72535746006725654e-01,
    9.74404967250239729e-01,
    9.76209138969172385e-01,
    9.77948140720341086e-01,
    9.79621856411210135e-01,
    9.81230174307644254e-01,
    9.82772987041374280e-01,
    9.84250191617171311e-01,
    9.85661689419733311e-01,
    9.87007386220281502e-01,
    9.88287192182869867e-01,
    9.89501021870408670e-01,
    9.90648794250406084e-01,
    9.91730432700432041e-01,
    9.92745865013315298e-01,
    9.93695023402088262e-01,
    9.94577844504706765e-01,
    9.95394269388595321e-01,
    9.96144243555108666e-01,
    9.96827716944091335e-01,
    9.97444643938910747e-01,
    9.97994983372793798e-01,
    9.98478698538458942e-01,
    9.98895757206325730e-01,
    9.99246131667184456e-01,
    9.99529798855885887e-01,
    9.99746740811352286e-01,
    9.99896947137859593e-01,
    9.99980441172647394e-01,
];
const GL_WEIGHTS: [f64; 384] = [
    5.01941034867686955e-05,
    1.16839066573026630e-04,
    1.83574919355165579e-04,
    2.50307089084410490e-04,
    3.17024269811281504e-04,
    3.83720802091292140e-04,
    4.50391913771682687e-04,
    5.17033045349164929e-04,
    5.83639704263013521e-04,
    6.50207424096994804e-04,
    7.16731750994780109e-04,
    7.83208238590516782e-04,
    8.49632446003920874e-04,
    9.15999937063264138e-04,
    9.82306280066346264e-04,
    1.04854704779368951e-03,
    1.11471781764731063e-03,
    1.18081417185592194e-03,
    1.24683169771544152e-03,
    1.31276598785066002e-03,
    1.37861264048764684e-03,
    1.44436725973473589e-03,
    1.51002545586581029e-03,
    1.57558284560793682e-03,
    1.64103505242927153e-03,
    1.70637770682844714e-03,
    1.77160644662383471e-03,
    1.83671691724356751e-03,
    1.90170477201489921e-03,
    1.96656567245343710e-03,
    2.03129528855239836e-03,
    2.09588929907102064e-03,
    2.16034339182273431e-03,
    2.22465326396271285e-03,
    2.28881462227495521e-03,
    2.35282318345876899e-03,
    2.41667467441434051e-03,
    2.48036483252826560e-03,
    2.54388940595774021e-03,
    2.60724415391445181e-03,
    2.67042484694755399e-03,
    2.73342726722609333e-03,
    2.79624720882042793e-03,
    2.85888047798306009e-03,
    2.92132289342851527e-03,
    2.98357028661255450e-03,
    3.04561850201032778e-03,
    3.10746339739375547e-03,
    3.16910084410831992e-03,
    3.23052672734817393e-03,
    3.29173694643136120e-03,
    3.35272741507325030e-03,
    3.41349406165941845e-03,
    3.47403282951731698e-03,
    3.53433967718734835e-03,
    3.59441057869245217e-03,
    3.65424152380698685e-03,
    3.71382851832431247e-03,
    3.77316758432358346e-03,
    3.83225476043517121e-03,
    3.89108610210519342e-03,
    3.94965768185889538e-03,
    4.00796558956267781e-03,
    4.06600593268526857e-03,
    4.12377483655760029e-03,
    4.18126844463128072e-03,
    4.23848291873628932e-03,
    4.29541443933692535e-03,
    4.35205920578727538e-03,
    4.40841343658428513e-03,
    4.46447336962078022e-03,
    4.52023526243623512e-03,
    4.57569539246679112e-03,
    4.63085005729389418e-03,
    4.68569557489104116e-03,
    4.74022828387002208e-03,
    4.79444454372510240e-03,
    4.84834073507610895e-03,
    4.90191325991019659e-03,
    4.95515854182168244e-03,
    5.00807302625133188e-03,
    5.06065318072310144e-03,
    5.11289549508039693e-03,
    5.16479648172001125e-03,
    5.21635267582545078e-03,
    5.26756063559773516e-03,
    5.31841694248538516e-03,
    5.36891820141282682e-03,
    5.41906104100662710e-03,
    5.46884211382094119e-03,
    5.51825809656070979e-03,
    5.56730569030376710e-03,
    5.61598162072080281e-03,
    5.66428263829418192e-03,
    5.71220551853465503e-03,
    5.75974706219692552e-03,
    5.80690409549281840e-03,
    5.85367347030361744e-03,
    5.90005206438982358e-03,
    5.94603678159981441e-03,
    5.99162455207646796e-03,
    6.03681233246208728e-03,
    6.08159710610167280e-03,
    6.12597588324419618e-03,
    6.16994570124223662e-03,
    6.21350362474959103e-03,
    6.25664674591772281e-03,
    6.29937218458923726e-03,
    6.34167708849066401e-03,
    6.38355863342257232e-03,
    6.42501402344827343e-03,
    6.46604049108043429e-03,
    6.50663529746572412e-03,
    6.54679573256784254e-03,
    6.58651911534826121e-03,
    6.62580279394531669e-03,
    6.66464414585114014e-03,
    6.70304057808694068e-03,
    6.74098952737589487e-03,
    6.77848846031412627e-03,
    6.81553487354050018e-03,
    6.85212629390287778e-03,
    6.88826027862375363e-03,
    6.92393441546331010e-03,
    6.95914632288014649e-03,
    6.99389365019070178e-03,
    7.02817407772573371e-03,
    7.06198531698550621e-03,
    7.09532511079243901e-03,
    7.12819123344184385e-03,
    7.16058149085032139e-03,
    7.19249372070248586e-03,
    7.22392579259530879e-03,
    7.25487560817998384e-03,
    7.28534110130251225e-03,
    7.31532023814132452e-03,
    7.34481101734306332e-03,
    7.37381147015625776e-03,
    7.40231966056281807e-03,
    7.43033368540717797e-03,
    7.45785167452331886e-03,
    7.48487179085978969e-03,
    7.51139223060207865e-03,
    7.53741122329336204e-03,
    7.56292703195238225e-03,
    7.58793795318956146e-03,
    7.61244231732079602e-03,
    7.63643848847873896e-03,
    7.65992486472206417e-03,
    7.68289987814253870e-03,
    7.70536199496952429e-03,
    7.72730971567243963e-03,
    7.74874157506091378e-03,
    7.76965614238246238e-03,
    7.79005202141822613e-03,
    7.80992785057590261e-03,
    7.82928230298081916e-03,
    7.84811408656456086e-03,
    7.86642194415109429e-03,
    7.88420465354066485e-03,
    7.90146102759159939e-03,
    7.91818991429931764e-03,
    7.93439019687344760e-03,
    7.95006079381220358e-03,
    7.96520065897470854e-03,
    7.97980878165076984e-03,
    7.99388418662826637e-03,
    8.00742593425854757e-03,
    8.02043312051886614e-03,
    8.03290487707280024e-03,
    8.04484037132826083e-03,
    8.05623880649317461e-03,
    8.06709942162842086e-03,
    8.07742149169882011e-03,
    8.08720432762159366e-03,
    8.09644727631220196e-03,
    8.10514972072793292e-03,
    8.11331107990920751e-03,
    8.12093080901841492e-03,
    8.12800839937608509e-03,
    8.13454337849503321e-03,
    8.14053531011177042e-03,
    8.14598379421577028e-03,
    8.15088846707587474e-03,
    8.15524900126509203e-03,
    8.15906510568189906e-03,
    8.16233652557010053e-03,
    8.16506304253546492e-03,
    8.16724447456070729e-03,
    8.16888067601734442e-03,
    8.16997153767546995e-03,
    8.17051698671110438e-03,
    8.17051698671110438e-03,
    8.16997153767546995e-03,
    8.16888067601734442e-03,
    8.16724447456070729e-03,
    8.16506304253546492e-03,
    8.16233652557010053e-03,
    8.15906510568189906e-03,
    8.15524900126509203e-03,
    8.15088846707587474e-03,
    8.14598379421577028e-03,
    8.14053531011177042e-03,
    8.13454337849503321e-03,
    8.12800839937608509e-03,
    8.12093080901841492e-03,
    8.11331107990920751e-03,
    8.10514972072793292e-03,
    8.09644727631220196e-03,
    8.08720432762159366e-03,
    8.07742149169882011e-03,
    8.06709942162842086e-03,
    8.05623880649317461e-03,
    8.04484037132826083e-03,
    8.03290487707280024e-03,
    8.02043312051886614e-03,
    8.00742593425854757e-03,
    7.99388418662826637e-03,
    7.97980878165076984e-03,
    7.96520065897470854e-03,
    7.95006079381220358e-03,
    7.93439019687344760e-03,
    7.91818991429931764e-03,
    7.90146102759159939e-03,
    7.88420465354066485e-03,
    7.86642194415109429e-03,
    7.84811408656456086e-03,
    7.82928230298081916e-03,
    7.80992785057590261e-03,
    7.79005202141822613e-03,
    7.76965614238246238e-03,
    7.74874157506091378e-03,
    7.72730971567243963e-03,
    7.70536199496952429e-03,
    7.68289987814253870e-03,
    7.65992486472206417e-03,
    7.63643848847873896e-03,
    7.61244231732079602e-03,
    7.58793795318956146e-03,
    7.56292703195238225e-03,
    7.53741122329336204e-03,
    7.51139223060207865e-03,
    7.48487179085978969e-03,
    7.45785167452331886e-03,
    7.43033368540717797e-03,
    7.40231966056281807e-03,
    7.37381147015625776e-03,
    7.34481101734306332e-03,
    7.31532023814132452e-03,
    7.28534110130251225e-03,
    7.25487560817998384e-03,
    7.22392579259530879e-03,
    7.19249372070248586e-03,
    7.16058149085032139e-03,
    7.12819123344184385e-03,
    7.09532511079243901e-03,
    7.06198531698550621e-03,
    7.02817407772573371e-03,
    6.99389365019070178e-03,
    6.95914632288014649e-03,
    6.92393441546331010e-03,
    6.88826027862375363e-03,
    6.85212629390287778e-03,
    6.81553487354050018e-03,
    6.77848846031412627e-03,
    6.74098952737589487e-03,
    6.70304057808694068e-03,
    6.66464414585114014e-03,
    6.62580279394531669e-03,
    6.58651911534826121e-03,
    6.54679573256784254e-03,
    6.50663529746572412e-03,
    6.46604049108043429e-03,
    6.42501402344827343e-03,
    6.38355863342257232e-03,
    6.34167708849066401e-03,
    6.29937218458923726e-03,
    6.25664674591772281e-03,
    6.21350362474959103e-03,
    6.16994570124223662e-03,
    6.12597588324419618e-03,
    6.08159710610167280e-03,
    6.03681233246208728e-03,
    5.99162455207646796e-03,
    5.94603678159981441e-03,
    5.90005206438982358e-03,
    5.85367347030361744e-03,
    5.80690409549281840e-03,
    5.75974706219692552e-03,
    5.71220551853465503e-03,
    5.66428263829418192e-03,
    5.61598162072080281e-03,
    5.56730569030376710e-03,
    5.51825809656070979e-03,
    5.46884211382094119e-03,
    5.41906104100662710e-03,
    5.36891820141282682e-03,
    5.31841694248538516e-03,
    5.26756063559773516e-03,
    5.21635267582545078e-03,
    5.16479648172001125e-03,
    5.11289549508039693e-03,
    5.06065318072310144e-03,
    5.00807302625133188e-03,
    4.95515854182168244e-03,
    4.90191325991019659e-03,
    4.84834073507610895e-03,
    4.79444454372510240e-03,
    4.74022828387002208e-03,
    4.68569557489104116e-03,
    4.63085005729389418e-03,
    4.57569539246679112e-03,
    4.52023526243623512e-03,
    4.46447336962078022e-03,
    4.40841343658428513e-03,
    4.35205920578727538e-03,
    4.29541443933692535e-03,
    4.23848291873628932e-03,
    4.18126844463128072e-03,
    4.12377483655760029e-03,
    4.06600593268526857e-03,
    4.00796558956267781e-03,
    3.94965768185889538e-03,
    3.89108610210519342e-03,
    3.83225476043517121e-03,
    3.77316758432358346e-03,
    3.71382851832431247e-03,
    3.65424152380698685e-03,
    3.59441057869245217e-03,
    3.53433967718734835e-03,
    3.47403282951731698e-03,
    3.41349406165941845e-03,
    3.35272741507325030e-03,
    3.29173694643136120e-03,
    3.23052672734817393e-03,
    3.16910084410831992e-03,
    3.10746339739375547e-03,
    3.04561850201032778e-03,
    2.98357028661255450e-03,
    2.92132289342851527e-03,
    2.85888047798306009e-03,
    2.79624720882042793e-03,
    2.73342726722609333e-03,
    2.67042484694755399e-03,
    2.60724415391445181e-03,
    2.54388940595774021e-03,
    2.48036483252826560e-03,
    2.41667467441434051e-03,
    2.35282318345876899e-03,
    2.28881462227495521e-03,
    2.22465326396271285e-03,
    2.16034339182273431e-03,
    2.09588929907102064e-03,
    2.03129528855239836e-03,
    1.96656567245343710e-03,
    1.90170477201489921e-03,
    1.83671691724356751e-03,
    1.77160644662383471e-03,
    1.70637770682844714e-03,
    1.64103505242927153e-03,
    1.57558284560793682e-03,
    1.51002545586581029e-03,
    1.44436725973473589e-03,
    1.37861264048764684e-03,
    1.31276598785066002e-03,
    1.24683169771544152e-03,
    1.18081417185592194e-03,
    1.11471781764731063e-03,
    1.04854704779368951e-03,
    9.82306280066346264e-04,
    9.15999937063264138e-04,
    8.49632446003920874e-04,
    7.83208238590516782e-04,
    7.16731750994780109e-04,
    6.50207424096994804e-04,
    5.83639704263013521e-04,
    5.17033045349164929e-04,
    4.50391913771682687e-04,
    3.83720802091292140e-04,
    3.17024269811281504e-04,
    2.50307089084410490e-04,
    1.83574919355165579e-04,
    1.16839066573026630e-04,
    5.01941034867686955e-05,
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
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
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

#[derive(Debug, Default)]
struct TailCellMomentCache {
    moments: std::collections::HashMap<TailCellMomentCacheKey, CellMomentState>,
    hits: usize,
    misses: usize,
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
    cache.moments.clear();
    cache.hits = 0;
    cache.misses = 0;
}

pub fn tail_cell_moment_cache_stats() -> TailCellMomentCacheStats {
    let cache = tail_cell_moment_cache()
        .lock()
        .expect("tail cell moment cache mutex poisoned");
    TailCellMomentCacheStats {
        hits: cache.hits,
        misses: cache.misses,
        entries: cache.moments.len(),
    }
}

#[cfg(test)]
fn evaluate_cell_moments_with_locked_tail_cache(
    cache: &mut TailCellMomentCache,
    cell: DenestedCubicCell,
    max_degree: usize,
) -> Result<CellMomentState, String> {
    if TAIL_CELL_MOMENT_CACHE_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
        && let Some(key) = tail_cell_cache_key(cell, max_degree)
    {
        if let Some(state) = cache.moments.get(&key).cloned() {
            cache.hits += 1;
            return Ok(state);
        }
        cache.misses += 1;
        let state = evaluate_cell_moments_uncached(cell, max_degree)?;
        let state = cache.moments.entry(key).or_insert_with(|| state.clone());
        return Ok(state.clone());
    }
    evaluate_cell_moments_uncached(cell, max_degree)
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
        let value_spilled = self
            .state
            .as_ref()
            .map(|s| {
                let s = s.as_ref();
                if s.moments.spilled() {
                    s.moments
                        .capacity()
                        .saturating_mul(std::mem::size_of::<f64>())
                } else {
                    0
                }
            })
            .unwrap_or(0);
        let deriv_spilled = self
            .derivative_state
            .as_ref()
            .map(|s| {
                let s = s.as_ref();
                if s.moments.spilled() {
                    s.moments
                        .capacity()
                        .saturating_mul(std::mem::size_of::<f64>())
                } else {
                    0
                }
            })
            .unwrap_or(0);
        std::mem::size_of::<Self>()
            .saturating_add(value_spilled)
            .saturating_add(deriv_spilled)
    }
}

#[derive(Debug, Default)]
pub struct CellMomentCacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
}

impl CellMomentCacheStats {
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

#[derive(Clone, Debug, PartialEq)]
pub struct CellDerivativeMomentState {
    pub branch: ExactCellBranch,
    pub moments: CellMomentVec,
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
    -0.912_234_428_251_325_9,
    -0.839_116_971_822_218_8,
    -0.746_331_906_460_150_8,
    -0.636_053_680_726_515_0,
    -0.510_867_001_950_827_1,
    -0.373_706_088_715_419_6,
    -0.227_785_851_141_645_1,
    -0.076_526_521_133_497_33,
    0.076_526_521_133_497_33,
    0.227_785_851_141_645_1,
    0.373_706_088_715_419_6,
    0.510_867_001_950_827_1,
    0.636_053_680_726_515_0,
    0.746_331_906_460_150_8,
    0.839_116_971_822_218_8,
    0.912_234_428_251_325_9,
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
    if lengths.is_empty() || lengths.iter().any(|&len| len == 0) {
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
    //
    // Hot-path notes:
    //   * `cell.eta(z)` is computed once via Horner and reused both to form
    //     `q(z) = 0.5*(z² + η²)` and to feed `normal_cdf(eta)`; the previous
    //     formulation called `cell.q(z)` which redundantly recomputed η.
    //   * `half_width` is folded into the per-node weight, eliminating the
    //     post-loop scaling pass over `moments` and the trailing
    //     `* half_width` on `value_integral`.
    //   * The inner moment accumulation reborrows the moments slice as `&mut
    //     [f64]` and steps via raw indexing so the compiler emits a clean
    //     bounds-check-hoisted FMA chain.
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
    let half_v = f64x4::splat(0.5);
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
    let c0 = cell.c0;
    let c1 = cell.c1;
    let c2 = cell.c2;
    let c3 = cell.c3;
    let moments_slice: &mut [f64] = &mut moments;
    debug_assert_eq!(GL_NODES.len(), GL_WEIGHTS.len());
    // See `evaluate_non_affine_cell_state` for the SIMD strategy. The
    // derivative variant only needs `exp(-q)` (no value integral and so no
    // second `exp` per node), but the same 4-wide GL batching still amortizes
    // the single `exp` call across 4 lanes.
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
        let eta_v = c3_v
            .mul_add(z_v, c2_v)
            .mul_add(z_v, c1_v)
            .mul_add(z_v, c0_v);
        let z2_v = z_v * z_v;
        let neg_q_v = neg_half_v * (z2_v + eta_v * eta_v);
        let scaled_weight_v = weight_v * half_width_v;
        let moment_weight_v = scaled_weight_v * neg_q_v.exp();
        let z_arr = z_v.to_array();
        let mw_arr = moment_weight_v.to_array();
        for lane in 0..4 {
            accumulate_moments_unrolled4(moments_slice, mw_arr[lane], z_arr[lane]);
        }
        i += 4;
    }
    while i < n_total {
        let node = GL_NODES[i];
        let weight = GL_WEIGHTS[i];
        let z = center + half_width * node;
        let eta = c3.mul_add(z, c2).mul_add(z, c1).mul_add(z, c0);
        let q = 0.5 * (z * z + eta * eta);
        let moment_weight = (weight * half_width) * (-q).exp();
        accumulate_moments_unrolled4(moments_slice, moment_weight, z);
        i += 1;
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
    if TAIL_CELL_MOMENT_CACHE_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
        && let Some(key) = tail_cell_cache_key(cell, max_degree)
    {
        let mut cache = tail_cell_moment_cache()
            .lock()
            .expect("tail cell moment cache mutex poisoned");
        if let Some(state) = cache.moments.get(&key).cloned() {
            cache.hits += 1;
            return Ok(state);
        }
        cache.misses += 1;
        drop(cache);

        let state = evaluate_cell_moments_uncached(cell, max_degree)?;

        let mut cache = tail_cell_moment_cache()
            .lock()
            .expect("tail cell moment cache mutex poisoned");
        let state = cache.moments.entry(key).or_insert_with(|| state.clone());
        return Ok(state.clone());
    }
    evaluate_cell_moments_uncached(cell, max_degree)
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
                ExactCellBranch::Sextic => unreachable!("sextic cannot be a lowered branch"),
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
        let mut tail_cache = tail_cell_moment_cache()
            .lock()
            .expect("tail cell moment cache mutex poisoned");
        *tail_cache = TailCellMomentCache::default();
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
                            let actual = evaluate_cell_moments_with_locked_tail_cache(
                                &mut tail_cache,
                                cell,
                                max_degree,
                            )
                            .expect("cached affine tail moments miss");
                            let repeat = evaluate_cell_moments_with_locked_tail_cache(
                                &mut tail_cache,
                                cell,
                                max_degree,
                            )
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

        let stats = TailCellMomentCacheStats {
            hits: tail_cache.hits,
            misses: tail_cache.misses,
            entries: tail_cache.moments.len(),
        };
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
        let n = if steps % 2 == 0 { steps } else { steps + 1 };
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

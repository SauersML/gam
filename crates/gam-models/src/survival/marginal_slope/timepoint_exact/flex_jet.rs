//! Single-source flex survival row NLL over a runtime-`K` jet algebra (#932,
//! doc §C/§D/Unifying).
//!
//! The flex marginal-slope row negative log-likelihood is
//! ```text
//! ℓ = w·[ logΦ(−η₀) − (1−d)·logΦ(−η₁)
//!         + d·½η₁² − d·logχ₁ + d·½q₁² + d·logD₁ − d·logqd₁ + d·ln2π ]
//! ```
//! (`flex_sensitivity.rs:105`). [`flex_row_nll`] writes this **once** over a
//! generic [`FlexJet`] scalar; instantiating it at [`Jet2`] yields value /
//! gradient / Hessian, at [`Jet3`] yields the contracted third `D_dir H[u,v]`,
//! and at [`Jet4`] the contracted fourth. The directional / bidirectional
//! contraction "directions" fall out of the nilpotent ε / δ seeds of the timepoint
//! jets, exactly as the packed `Order2`/`OneSeed`/`TwoSeed` scalars do for
//! location-scale — but here over a **runtime** primary count `p` (the flex
//! primary count `4 + |h| + |w| + 1` is large and variable, so a `Vec`-backed
//! jet avoids the const-generic monomorphization blow-up the packed scalars would
//! incur).
//!
//! The timepoint quantities `η₀, η₁, χ₁, D₁` are produced **by the jet path
//! itself** in production: [`flex_timepoint_inputs_generic`] builds them from the
//! per-cell coefficient θ-jets and the moving-edge / moment recurrence, seeded at
//! `Jet2` (value/grad/Hess), `Jet3` (+directional) or `Jet4` (+bidirectional).
//! `q₁`/`qd₁` are seeded as plain primaries. The single-source probit derivative
//! stack `surv_stack` and the `ln` stack carry the only special functions (humans
//! own primitive stability, the algebra owns combinatorics).
//!
//! Verification is deliberately independent of the production composition:
//!
//! - scalar finite differences re-solve the real moving-boundary intercept;
//! - `Dual22` checks fourth order through a distinct 2+2 nesting;
//! - the compiled order-two lowering is pinned to the generic plan;
//! - rigid zero-deviation rows are pinned to the independent `Tower4` program.
//!
//! The Euler projector in [`MomentTerm::moment_term`] supplies the universal
//! `j/(j+m)` distinguished-derivative weight, and [`lift_intercept_flex`] runs
//! enough frozen-inverse iterations for the represented jet order. Those two
//! rules are shared by every timepoint channel.
//!

use super::*;
use crate::bms::signed_probit_neglog_unary_stack;
use crate::survival::marginal_slope::timewiggle_geometry::{
    TimewiggleBasisDerivativeRows, TimewiggleQBaseValues, timewiggle_q_from_basis_derivative_rows,
};
use gam_math::jet_scalar::{
    DynamicJetArena, DynamicOneSeed, DynamicOrder2, DynamicOrder2Accumulator, DynamicOrder2Term,
    OneSeed, Order2, RuntimeJetScalar,
};
use gam_math::jet_tower::Tower2;

/// Canonical value/gradient/Hessian timepoint channels consumed by the
/// contracted row expression.
#[derive(Clone, Debug)]
pub(crate) struct FlexTimepointBasePack {
    pub(crate) eta: f64,
    pub(crate) chi: f64,
    pub(crate) d: f64,
    pub(crate) eta_u: Vec<f64>,
    pub(crate) eta_uv: Vec<f64>,
    pub(crate) chi_u: Vec<f64>,
    pub(crate) chi_uv: Vec<f64>,
    pub(crate) d_u: Vec<f64>,
    pub(crate) d_uv: Vec<f64>,
}

/// Single-direction extension of the canonical timepoint channels.
#[derive(Clone, Debug)]
pub(crate) struct FlexTimepointDirectionalPack {
    pub(crate) eta_uv_dir: Vec<f64>,
    pub(crate) eta_u_dir: Vec<f64>,
    pub(crate) chi_u_dir: Vec<f64>,
    pub(crate) chi_uv_dir: Vec<f64>,
    pub(crate) d_u_dir: Vec<f64>,
    pub(crate) d_uv_dir: Vec<f64>,
}

/// Mixed second-direction extension of the canonical timepoint channels.
#[derive(Clone, Debug)]
pub(crate) struct FlexTimepointBidirectionalPack {
    pub(crate) eta_uv_uv: Vec<f64>,
    pub(crate) chi_uv_uv: Vec<f64>,
    pub(crate) d_uv_uv: Vec<f64>,
}

/// Motion of every family-owned scalar entering one FLEX row along one outer
/// direction.
///
/// The offset entries are deliberately not derivatives of the final
/// `(q0, q1, qd1)` coordinates: when time wiggle is active, the generic q-map
/// owns that nonlinear composition. `probit_scale` is motion of the physical
/// multiplicative probit scale (not log-sigma); a baseline direction sets it to
/// zero, while a learned-scale direction supplies the exact analytic scale
/// derivative stack. This carrier therefore serves baseline, scale, and their
/// combined/polarized directions without another row formula.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FlexFamilyRowDirection {
    pub(crate) entry: f64,
    pub(crate) exit: f64,
    pub(crate) derivative_exit: f64,
    pub(crate) probit_scale: f64,
}

/// One family derivative channel of a complete FLEX row in flattened
/// coefficient coordinates.
#[derive(Clone, Debug)]
pub(crate) struct FlexFamilyCoefficientTerms {
    pub(crate) objective: f64,
    pub(crate) gradient: Array1<f64>,
    pub(crate) hessian: Array2<f64>,
}

/// Exact first and same-direction second family derivatives of a complete FLEX
/// row, plus an optional coefficient-map drift of the first channel.
#[derive(Clone, Debug)]
pub(crate) struct FlexFamilyDirectionRowTerms {
    pub(crate) first: FlexFamilyCoefficientTerms,
    pub(crate) second: FlexFamilyCoefficientTerms,
    pub(crate) directional: Option<FlexFamilyCoefficientTerms>,
}

pub(crate) fn pack_flex_timepoint_base(base: &SurvivalFlexTimepointExact) -> FlexTimepointBasePack {
    FlexTimepointBasePack {
        eta: base.eta,
        chi: base.chi,
        d: base.d,
        eta_u: base.eta_u.to_vec(),
        eta_uv: base.eta_uv.iter().copied().collect(),
        chi_u: base.chi_u.to_vec(),
        chi_uv: base.chi_uv.iter().copied().collect(),
        d_u: base.d_u.to_vec(),
        d_uv: base.d_uv.iter().copied().collect(),
    }
}

thread_local! {
    /// Per-worker FLEX directional workspace. The largest row tape is retained
    /// across contractions, so warmed production calls do not revisit the
    /// global allocator for runtime-sized derivative channels.
    static FLEX_THIRD_JET_ARENA: std::cell::RefCell<DynamicJetArena> =
        std::cell::RefCell::new(DynamicJetArena::new());
}

pub(crate) fn with_flex_third_jet_arena<R>(evaluate: impl FnOnce(&mut DynamicJetArena) -> R) -> R {
    FLEX_THIRD_JET_ARENA.with(|workspace| {
        let mut arena = workspace.borrow_mut();
        arena.reset();
        let result = evaluate(&mut arena);
        arena.reset();
        result
    })
}

/// The `[f64; 5]` Faà di Bruno stack of `g(η) = logΦ(−η)` at `η`.
///
/// With `N(m) = −logΦ(m)` and `(k1,k2,k3,k4) = N′…N⁗(m)` at `m = −η`
/// (`signed_probit_neglog_derivatives_up_to_fourth`), the chain rule on
/// `g(η) = −N(−η)` gives `g′ = k1`, `g″ = −k2`, `g‴ = k3`, `g⁗ = −k4`. This is
/// the entry/exit survival stack; composing the timepoint η-jet with it
/// supplies the canonical entry/exit survival derivatives consumed by every
/// generated flex-row lowering.
#[inline]
fn surv_stack(eta: f64) -> Result<[f64; 5], String> {
    let signed_margin = -eta;
    if signed_margin != f64::INFINITY && !signed_margin.is_finite() {
        return Err(format!(
            "non-finite signed margin in exact probit derivative helper: {signed_margin}"
        ));
    }
    // `weight = -1` makes the fused BMS primitive return the derivative stack
    // of `logΦ(m)` from ONE Mills-ratio evaluation. Compose with `m = -η` by
    // flipping odd derivative orders. This replaces the old two-call sequence
    // (one `logcdf`, then another `logcdf` discarded by the derivative helper).
    let m_stack = signed_probit_neglog_unary_stack(signed_margin, -1.0);
    Ok([m_stack[0], -m_stack[1], m_stack[2], -m_stack[3], m_stack[4]])
}

/// The `[f64; 5]` Faà di Bruno stack of `ln(x)`.
#[inline]
fn ln_stack(x: f64) -> [f64; 5] {
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    [x.ln(), inv, -inv2, 2.0 * inv2 * inv, -6.0 * inv2 * inv2]
}

/// A runtime-`K` truncated-Taylor scalar: the row loss is written once against
/// this interface and re-instantiated at [`Jet1`] / [`Jet2`] / [`Jet3`] /
/// [`Jet4`] for gradient-only, value/grad/Hessian, contracted-third, and
/// contracted-fourth channels.
/// The runtime-`p` (Vec-backed) INNER timepoint-geometry jet layer. It EXTENDS
/// the shared [`JetField`] scalar-field algebra (value / add / sub / mul / neg /
/// scale / the single Faà di Bruno `compose_unary`) with only the one thing the
/// flex geometry adds on top: the truncation order [`FlexJet::ORDER`]. The
/// const-`K` packed row seam (`JetScalar`) extends the SAME `JetField` base — but
/// the two program layers stay distinct: `FlexJet` is runtime-`p`, not const-`K`.
trait FlexJet: JetField + Clone {
    /// Highest derivative order represented by this nilpotent algebra.
    /// A value-zero perturbation raised to `ORDER + 1` is identically zero, so
    /// generic Taylor-polynomial builders must stop at this exact order.
    const ORDER: usize;

    /// Scale every derivative channel by its total homogeneous order. Entry
    /// `factors[r]` applies to channels carrying exactly `r` derivatives.
    /// This is the representation-independent Euler-operator seam used by the
    /// distinguished calibration projector.
    fn scale_homogeneous_orders(&self, factors: [f64; 5]) -> Self;
}

impl FlexJet for f64 {
    const ORDER: usize = 0;

    #[inline]
    fn scale_homogeneous_orders(&self, factors: [f64; 5]) -> Self {
        factors[0] * self
    }
}

/// Lift any flex scalar through one independent second-order family direction.
///
/// A channel in `v`, `g`, or `h` carries respectively zero, one, or two outer
/// derivatives in addition to its inner homogeneous order. Shifting the Euler
/// factors by that outer order is essential: applying the unshifted factors to
/// all three fields would silently mis-scale calibration's distinguished
/// family derivative. Channels above the row program's certified total order
/// four are truncated explicitly.
impl<J: FlexJet> FlexJet for Dual2<J> {
    const ORDER: usize = if J::ORDER >= 2 { 4 } else { J::ORDER + 2 };

    #[inline]
    fn scale_homogeneous_orders(&self, factors: [f64; 5]) -> Self {
        let shifted = |outer_order: usize| {
            std::array::from_fn(|inner_order| {
                factors
                    .get(inner_order + outer_order)
                    .copied()
                    .unwrap_or(0.0)
            })
        };
        Self {
            v: self.v.scale_homogeneous_orders(shifted(0)),
            g: self.g.scale_homogeneous_orders(shifted(1)),
            h: self.h.scale_homogeneous_orders(shifted(2)),
        }
    }
}

const FLEX_OUTER_SOURCE_COUNT: usize = 6;
const FLEX_OUTER_TERM_COUNT: usize = 7;

/// The six independent inner scalars used by the seven-term flex row NLL.
/// `eta1` appears twice (survival and Gaussian-density terms), and the order-two
/// compiler fuses those two outer stacks before touching its derivative views.
#[derive(Clone, Copy)]
#[repr(usize)]
enum FlexOuterSource {
    Eta0 = 0,
    Eta1 = 1,
    Q1 = 2,
    Chi1 = 3,
    D1 = 4,
    Qd1 = 5,
}

impl FlexOuterSource {
    #[inline(always)]
    fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy)]
enum FlexOuterTransform {
    Compose([f64; 5]),
    Square,
}

#[derive(Clone, Copy)]
enum FlexOuterCombine {
    Add,
    Subtract,
}

/// One term of the flex row NLL. This term list is the sole mathematical
/// definition consumed by both the generic high-order jets and the compiled
/// allocation-minimal order-two lowering.
#[derive(Clone, Copy)]
struct FlexOuterTerm {
    source: FlexOuterSource,
    transform: FlexOuterTransform,
    scale: f64,
    combine: FlexOuterCombine,
}

impl FlexOuterTerm {
    #[inline(always)]
    fn evaluate<J: FlexJet>(&self, sources: &FlexJetSources<'_, J>) -> J {
        let source = sources.get(self.source);
        let transformed = match self.transform {
            FlexOuterTransform::Compose(stack) => source.compose_unary(stack),
            FlexOuterTransform::Square => source.mul(source),
        };
        transformed.scale(self.scale)
    }

    #[inline(always)]
    fn order2_channels(&self, source_value: f64) -> [f64; 3] {
        let mut channels = match self.transform {
            FlexOuterTransform::Compose(stack) => [stack[0], stack[1], stack[2]],
            FlexOuterTransform::Square => [
                source_value * source_value,
                source_value + source_value,
                2.0,
            ],
        };
        for channel in &mut channels {
            *channel *= self.scale;
        }
        channels
    }
}

struct FlexJetSources<'a, J> {
    eta0: &'a J,
    eta1: &'a J,
    q1: &'a J,
    chi1: &'a J,
    d1: &'a J,
    qd1: &'a J,
}

impl<'a, J> FlexJetSources<'a, J> {
    #[inline(always)]
    fn get(&self, source: FlexOuterSource) -> &'a J {
        match source {
            FlexOuterSource::Eta0 => self.eta0,
            FlexOuterSource::Eta1 => self.eta1,
            FlexOuterSource::Q1 => self.q1,
            FlexOuterSource::Chi1 => self.chi1,
            FlexOuterSource::D1 => self.d1,
            FlexOuterSource::Qd1 => self.qd1,
        }
    }
}

/// The single flex row-NLL source, excluding the additive `w·d·ln2π`
/// constant. The ordered seven-term plan preserves the generic evaluator's
/// historical arithmetic while exposing the same source/transform metadata to
/// the order-two compiler.
struct FlexOuterPlan {
    terms: [FlexOuterTerm; FLEX_OUTER_TERM_COUNT],
}

impl FlexOuterPlan {
    #[inline(always)]
    fn new(
        chi1: f64,
        d1: f64,
        qd1: f64,
        surv0: [f64; 5],
        surv1: [f64; 5],
        wi: f64,
        di: f64,
    ) -> Self {
        let wd = wi * di;
        Self {
            terms: [
                FlexOuterTerm {
                    source: FlexOuterSource::Eta0,
                    transform: FlexOuterTransform::Compose(surv0),
                    scale: wi,
                    combine: FlexOuterCombine::Add,
                },
                FlexOuterTerm {
                    source: FlexOuterSource::Eta1,
                    transform: FlexOuterTransform::Compose(surv1),
                    scale: -wi * (1.0 - di),
                    combine: FlexOuterCombine::Add,
                },
                FlexOuterTerm {
                    source: FlexOuterSource::Eta1,
                    transform: FlexOuterTransform::Square,
                    scale: 0.5 * wd,
                    combine: FlexOuterCombine::Add,
                },
                FlexOuterTerm {
                    source: FlexOuterSource::Q1,
                    transform: FlexOuterTransform::Square,
                    scale: 0.5 * wd,
                    combine: FlexOuterCombine::Add,
                },
                FlexOuterTerm {
                    source: FlexOuterSource::Chi1,
                    transform: FlexOuterTransform::Compose(ln_stack(chi1)),
                    scale: wd,
                    combine: FlexOuterCombine::Subtract,
                },
                FlexOuterTerm {
                    source: FlexOuterSource::D1,
                    transform: FlexOuterTransform::Compose(ln_stack(d1)),
                    scale: wd,
                    combine: FlexOuterCombine::Add,
                },
                FlexOuterTerm {
                    source: FlexOuterSource::Qd1,
                    transform: FlexOuterTransform::Compose(ln_stack(qd1)),
                    scale: wd,
                    combine: FlexOuterCombine::Subtract,
                },
            ],
        }
    }

    #[inline(always)]
    fn evaluate<J: FlexJet>(&self, sources: &FlexJetSources<'_, J>) -> J {
        let mut output = self.terms[0].evaluate(sources);
        for term in &self.terms[1..] {
            let contribution = term.evaluate(sources);
            output = match term.combine {
                FlexOuterCombine::Add => output.add(&contribution),
                FlexOuterCombine::Subtract => output.sub(&contribution),
            };
        }
        output
    }

    /// Compile the ordered term list to one value plus one `(f', f'')` pair per
    /// independent source. Terms sharing a source are fused before the
    /// runtime-width gradient/Hessian sweep.
    #[inline(always)]
    fn compile_order2(&self, source_values: [f64; FLEX_OUTER_SOURCE_COUNT]) -> FlexOrder2Plan {
        let mut value = 0.0;
        let mut derivatives = [[0.0; 2]; FLEX_OUTER_SOURCE_COUNT];
        for term in &self.terms {
            let channels = term.order2_channels(source_values[term.source.index()]);
            let sign = match term.combine {
                FlexOuterCombine::Add => 1.0,
                FlexOuterCombine::Subtract => -1.0,
            };
            value += sign * channels[0];
            derivatives[term.source.index()][0] += sign * channels[1];
            derivatives[term.source.index()][1] += sign * channels[2];
        }
        FlexOrder2Plan { value, derivatives }
    }
}

/// The single-source flex row NLL **minus** the additive `w·d·ln2π` constant
/// (which the caller adds to the value channel — it has no derivative). Written
/// once over `FlexJet`; the instantiating scalar selects the channel.
#[inline]
fn flex_row_nll<J: FlexJet>(
    eta0: &J,
    eta1: &J,
    chi1: &J,
    d1: &J,
    q1: &J,
    qd1: &J,
    surv0: [f64; 5],
    surv1: [f64; 5],
    wi: f64,
    di: f64,
) -> J {
    FlexOuterPlan::new(chi1.value(), d1.value(), qd1.value(), surv0, surv1, wi, di).evaluate(
        &FlexJetSources {
            eta0,
            eta1,
            q1,
            chi1,
            d1,
            qd1,
        },
    )
}

// ── Compiled two-allocation order-two lowering (value/grad/Hessian) ────────
//
// Instantiating the plan through the generic [`Jet2`] algebra would allocate
// roughly 18 intermediate vectors. Instead [`FlexOuterPlan::compile_order2`]
// fuses the ordered term list to six source coefficients and the universal
// [`DynamicOrder2Accumulator`] reads the ndarray channels in place. The only
// allocations are the two buffers returned to the caller. Both representations
// consume the same plan; there is no second likelihood formula to maintain.
struct FlexOrder2Plan {
    value: f64,
    derivatives: [[f64; 2]; FLEX_OUTER_SOURCE_COUNT],
}

struct FlexOrder2View<'a> {
    value: f64,
    gradient: ndarray::ArrayView1<'a, f64>,
    hessian: ndarray::ArrayView2<'a, f64>,
}

struct FlexOrder2Inputs<'a> {
    eta0: FlexOrder2View<'a>,
    eta1: FlexOrder2View<'a>,
    q1: (f64, usize),
    chi1: FlexOrder2View<'a>,
    d1: FlexOrder2View<'a>,
    qd1: (f64, usize),
}

enum FlexOrder2Term<'a> {
    Dense {
        outer: [f64; 2],
        gradient: ndarray::ArrayView1<'a, f64>,
        hessian: ndarray::ArrayView2<'a, f64>,
    },
    Axis {
        outer: [f64; 2],
        axis: usize,
    },
}

impl DynamicOrder2Term for FlexOrder2Term<'_> {
    #[inline(always)]
    fn outer_first(&self) -> f64 {
        match self {
            Self::Dense { outer, .. } | Self::Axis { outer, .. } => outer[0],
        }
    }

    #[inline(always)]
    fn outer_second(&self) -> f64 {
        match self {
            Self::Dense { outer, .. } | Self::Axis { outer, .. } => outer[1],
        }
    }

    #[inline(always)]
    fn inner_gradient(&self, axis: usize) -> f64 {
        match self {
            Self::Dense { gradient, .. } => gradient[axis],
            Self::Axis { axis: source, .. } => {
                if axis == *source {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    #[inline(always)]
    fn inner_hessian(&self, row: usize, column: usize) -> f64 {
        match self {
            Self::Dense { hessian, .. } => hessian[[row, column]],
            Self::Axis { .. } => 0.0,
        }
    }
}

/// Allocation-minimal order-two lowering of the same [`FlexOuterPlan`] used by
/// the generic high-order jets. The plan fuses the two `eta1` terms into one
/// outer derivative pair; [`DynamicOrder2Accumulator`] then performs one fused
/// gradient pass and one fused Hessian pass into exactly two owned buffers.
#[inline]
fn lower_flex_outer_plan_order2(
    plan: &FlexOuterPlan,
    inputs: FlexOrder2Inputs<'_>,
    dimension: usize,
) -> (f64, Vec<f64>, Vec<f64>) {
    let source_values = [
        inputs.eta0.value,
        inputs.eta1.value,
        inputs.q1.0,
        inputs.chi1.value,
        inputs.d1.value,
        inputs.qd1.0,
    ];
    let compiled = plan.compile_order2(source_values);
    let terms = [
        FlexOrder2Term::Dense {
            outer: compiled.derivatives[FlexOuterSource::Eta0.index()],
            gradient: inputs.eta0.gradient,
            hessian: inputs.eta0.hessian,
        },
        FlexOrder2Term::Dense {
            outer: compiled.derivatives[FlexOuterSource::Eta1.index()],
            gradient: inputs.eta1.gradient,
            hessian: inputs.eta1.hessian,
        },
        FlexOrder2Term::Axis {
            outer: compiled.derivatives[FlexOuterSource::Q1.index()],
            axis: inputs.q1.1,
        },
        FlexOrder2Term::Dense {
            outer: compiled.derivatives[FlexOuterSource::Chi1.index()],
            gradient: inputs.chi1.gradient,
            hessian: inputs.chi1.hessian,
        },
        FlexOrder2Term::Dense {
            outer: compiled.derivatives[FlexOuterSource::D1.index()],
            gradient: inputs.d1.gradient,
            hessian: inputs.d1.hessian,
        },
        FlexOrder2Term::Axis {
            outer: compiled.derivatives[FlexOuterSource::Qd1.index()],
            axis: inputs.qd1.1,
        },
    ];
    DynamicOrder2Accumulator::from_composed_sum(dimension, compiled.value, &terms).into_channels()
}

// ── Jet2: value / gradient / Hessian (runtime K) ───────────────────────────

/// Value `v`, gradient `g[i]`, Hessian `h[i*p+j]` (row-major, symmetric) over a
/// runtime primary count `p = g.len()`. The order-≤2 truncation of the Leibniz /
/// Faà di Bruno rules — bit-identical to [`super::super::super::jet_tower::Tower2`]
/// channel-for-channel, just `Vec`-backed.
#[derive(Clone)]
struct Jet2 {
    v: f64,
    g: Vec<f64>,
    h: Vec<f64>,
}

impl Jet2 {
    /// A jet from explicit channels: `g` length `p`, `h` length `p*p` (or empty
    /// for the grad-only path, treated as the zero Hessian).
    fn from_parts(v: f64, g: &[f64], h: &[f64]) -> Self {
        let p = g.len();
        let hv = if h.is_empty() {
            vec![0.0; p * p]
        } else {
            assert_eq!(h.len(), p * p, "Jet2::from_parts Hessian length");
            h.to_vec()
        };
        Jet2 {
            v,
            g: g.to_vec(),
            h: hv,
        }
    }

    /// The seeded primary `p_axis` at value `x`: unit gradient in slot `axis`,
    /// zero Hessian.
    fn primary(x: f64, axis: usize, p: usize) -> Self {
        let mut g = vec![0.0; p];
        if axis < p {
            g[axis] = 1.0;
        }
        Jet2 {
            v: x,
            g,
            h: vec![0.0; p * p],
        }
    }

    #[inline]
    fn p(&self) -> usize {
        self.g.len()
    }

    #[inline]
    fn scale_homogeneous_from(&self, offset: usize, factors: [f64; 5]) -> Self {
        assert!(offset + 2 < factors.len());
        Jet2 {
            v: factors[offset] * self.v,
            g: self
                .g
                .iter()
                .map(|&channel| factors[offset + 1] * channel)
                .collect(),
            h: self
                .h
                .iter()
                .map(|&channel| factors[offset + 2] * channel)
                .collect(),
        }
    }
}

impl JetField for Jet2 {
    #[inline]
    fn value(&self) -> f64 {
        self.v
    }
    fn add(&self, o: &Self) -> Self {
        let p = self.p();
        let mut g = vec![0.0; p];
        let mut h = vec![0.0; p * p];
        for i in 0..p {
            g[i] = self.g[i] + o.g[i];
        }
        for k in 0..p * p {
            h[k] = self.h[k] + o.h[k];
        }
        Jet2 {
            v: self.v + o.v,
            g,
            h,
        }
    }
    fn sub(&self, o: &Self) -> Self {
        let p = self.p();
        let mut g = vec![0.0; p];
        let mut h = vec![0.0; p * p];
        for i in 0..p {
            g[i] = self.g[i] - o.g[i];
        }
        for k in 0..p * p {
            h[k] = self.h[k] - o.h[k];
        }
        Jet2 {
            v: self.v - o.v,
            g,
            h,
        }
    }
    fn mul(&self, o: &Self) -> Self {
        let p = self.p();
        let mut g = vec![0.0; p];
        let mut h = vec![0.0; p * p];
        for i in 0..p {
            g[i] = self.v * o.g[i] + self.g[i] * o.v;
        }
        for i in 0..p {
            for j in 0..p {
                h[i * p + j] = self.v * o.h[i * p + j]
                    + self.g[i] * o.g[j]
                    + self.g[j] * o.g[i]
                    + self.h[i * p + j] * o.v;
            }
        }
        Jet2 {
            v: self.v * o.v,
            g,
            h,
        }
    }
    fn scale(&self, s: f64) -> Self {
        Jet2 {
            v: self.v * s,
            g: self.g.iter().map(|&x| x * s).collect(),
            h: self.h.iter().map(|&x| x * s).collect(),
        }
    }
    #[inline]
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // Order-≤2 reads only [f, f', f''].
        let p = self.p();
        let (f, f1, f2) = (d[0], d[1], d[2]);
        let mut g = vec![0.0; p];
        let mut h = vec![0.0; p * p];
        for i in 0..p {
            g[i] = f1 * self.g[i];
        }
        for i in 0..p {
            for j in 0..p {
                h[i * p + j] = f2 * self.g[i] * self.g[j] + f1 * self.h[i * p + j];
            }
        }
        Jet2 { v: f, g, h }
    }
}

impl FlexJet for Jet2 {
    const ORDER: usize = 2;

    #[inline]
    fn scale_homogeneous_orders(&self, factors: [f64; 5]) -> Self {
        self.scale_homogeneous_from(0, factors)
    }
}

// ── Jet1: value + gradient only (grad-only path, no discarded Hessian) ──────

/// The order-≤1 truncation of the same Leibniz / Faà di Bruno rules `Jet2`
/// realises — value + gradient, **no Hessian channel**. The value and gradient
/// arithmetic of `Jet2` never reads its own `h` channel (`v`/`g` depend only on
/// `v`/`g`), so instantiating [`flex_row_nll`] at `Jet1` yields the value and
/// gradient `f64::to_bits`-identical to the `Jet2` path while eliminating the
/// entire `O(p²)` Hessian allocation + arithmetic. Used by the grad-only
/// production caller (`flex_sensitivity.rs`, `want_hess == false`), which builds
/// and then discards a full `p×p` Hessian under `Jet2`.
#[derive(Clone)]
struct Jet1 {
    v: f64,
    g: Vec<f64>,
}

impl Jet1 {
    /// A jet from a value + gradient view (the grad-only timepoint packs carry no
    /// `*_uv`, so only the value/gradient channels are ever supplied).
    fn from_view(v: f64, g: ndarray::ArrayView1<'_, f64>) -> Self {
        Jet1 {
            v,
            g: g.iter().copied().collect(),
        }
    }

    /// The seeded primary `p_axis` at value `x`: unit gradient in slot `axis`.
    fn primary(x: f64, axis: usize, p: usize) -> Self {
        let mut g = vec![0.0; p];
        if axis < p {
            g[axis] = 1.0;
        }
        Jet1 { v: x, g }
    }

    #[inline]
    fn p(&self) -> usize {
        self.g.len()
    }
}

impl JetField for Jet1 {
    #[inline]
    fn value(&self) -> f64 {
        self.v
    }
    fn add(&self, o: &Self) -> Self {
        let p = self.p();
        let mut g = vec![0.0; p];
        for i in 0..p {
            g[i] = self.g[i] + o.g[i];
        }
        Jet1 { v: self.v + o.v, g }
    }
    fn sub(&self, o: &Self) -> Self {
        let p = self.p();
        let mut g = vec![0.0; p];
        for i in 0..p {
            g[i] = self.g[i] - o.g[i];
        }
        Jet1 { v: self.v - o.v, g }
    }
    fn mul(&self, o: &Self) -> Self {
        let p = self.p();
        let mut g = vec![0.0; p];
        for i in 0..p {
            g[i] = self.v * o.g[i] + self.g[i] * o.v;
        }
        Jet1 { v: self.v * o.v, g }
    }
    fn scale(&self, s: f64) -> Self {
        Jet1 {
            v: self.v * s,
            g: self.g.iter().map(|&x| x * s).collect(),
        }
    }
    #[inline]
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        // Order-≤1 reads only [f, f'].
        let p = self.p();
        let (f, f1) = (d[0], d[1]);
        let mut g = vec![0.0; p];
        for i in 0..p {
            g[i] = f1 * self.g[i];
        }
        Jet1 { v: f, g }
    }
}

impl FlexJet for Jet1 {
    const ORDER: usize = 1;

    #[inline]
    fn scale_homogeneous_orders(&self, factors: [f64; 5]) -> Self {
        Jet1 {
            v: factors[0] * self.v,
            g: self.g.iter().map(|&channel| factors[1] * channel).collect(),
        }
    }
}

// ── Jet3: one-seed directional, contracted third (doc §A.2) ────────────────

/// An [`Jet2`] base plus one nilpotent ε (`ε² = 0`) holding another [`Jet2`].
/// After seeding the timepoint jets' ε-parts with their directional derivatives,
/// the ε-Hessian of the evaluated NLL is `Σ_c ℓ_{abc} dir_c = (D_dir H)[a][b]`.
#[derive(Clone)]
struct Jet3 {
    base: Jet2,
    eps: Jet2,
}

impl Jet3 {
    /// Seeded primary: base = `primary(x, axis)`, ε = constant `dir[axis]`.
    fn primary(x: f64, axis: usize, p: usize, dir_axis: f64) -> Self {
        Jet3 {
            base: Jet2::primary(x, axis, p),
            eps: Jet2::from_parts(dir_axis, &vec![0.0; p], &[]),
        }
    }
    /// The contracted-third channel `Σ_c ℓ_{abc} dir_c` (the ε-Hessian).
    fn contracted_third(&self) -> Vec<f64> {
        self.eps.h.clone()
    }
}

impl JetField for Jet3 {
    #[inline]
    fn value(&self) -> f64 {
        self.base.v
    }
    fn add(&self, o: &Self) -> Self {
        Jet3 {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
        }
    }
    fn sub(&self, o: &Self) -> Self {
        Jet3 {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
        }
    }
    fn mul(&self, o: &Self) -> Self {
        Jet3 {
            base: self.base.mul(&o.base),
            eps: self.base.mul(&o.eps).add(&self.eps.mul(&o.base)),
        }
    }
    fn scale(&self, s: f64) -> Self {
        Jet3 {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
        }
    }
    #[inline]
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let base = self.base.compose_unary([d[0], d[1], d[2], d[3], d[4]]);
        // f'(base) as a Jet2 (consumes [f', f'', f''']).
        let fprime = self.base.compose_unary([d[1], d[2], d[3], d[4], d[4]]);
        let eps = fprime.mul(&self.eps);
        Jet3 { base, eps }
    }
}

impl FlexJet for Jet3 {
    const ORDER: usize = 3;

    #[inline]
    fn scale_homogeneous_orders(&self, factors: [f64; 5]) -> Self {
        Jet3 {
            base: self.base.scale_homogeneous_from(0, factors),
            eps: self.eps.scale_homogeneous_from(1, factors),
        }
    }
}

/// Inner coefficient algebra used by the baseline-family row evaluator.
///
/// `Jet2` owns the coefficient value/score/Hessian channels. `Jet3` adds one
/// nilpotent coefficient direction whose epsilon part is the exact directional
/// drift of those same channels. The outer family coordinate is supplied by
/// `Dual2<J>` and is intentionally absent from this constructor.
trait FlexCoefficientJet: FlexJet {
    fn affine(
        value: f64,
        gradient: Vec<f64>,
        directional_value: f64,
        directional_gradient: Vec<f64>,
    ) -> Self;

    fn constant(value: f64, dimension: usize) -> Self {
        Self::affine(value, vec![0.0; dimension], 0.0, vec![0.0; dimension])
    }

    fn owned_base_terms(&self) -> FlexFamilyCoefficientTerms;

    fn owned_directional_terms(&self) -> Option<FlexFamilyCoefficientTerms>;
}

fn owned_jet2_terms(jet: &Jet2) -> FlexFamilyCoefficientTerms {
    let dimension = jet.g.len();
    FlexFamilyCoefficientTerms {
        objective: jet.v,
        gradient: Array1::from_vec(jet.g.clone()),
        hessian: Array2::from_shape_vec((dimension, dimension), jet.h.clone())
            .expect("Jet2 coefficient Hessian shape invariant"),
    }
}

impl FlexCoefficientJet for Jet2 {
    fn affine(
        value: f64,
        gradient: Vec<f64>,
        directional_value: f64,
        directional_gradient: Vec<f64>,
    ) -> Self {
        assert!(
            directional_value == 0.0
                && directional_gradient.iter().all(|value| *value == 0.0),
            "Jet2 cannot erase a coefficient-direction seed; use Jet3"
        );
        Self::from_parts(value, &gradient, &[])
    }

    fn owned_base_terms(&self) -> FlexFamilyCoefficientTerms {
        owned_jet2_terms(self)
    }

    fn owned_directional_terms(&self) -> Option<FlexFamilyCoefficientTerms> {
        None
    }
}

impl FlexCoefficientJet for Jet3 {
    fn affine(
        value: f64,
        gradient: Vec<f64>,
        directional_value: f64,
        directional_gradient: Vec<f64>,
    ) -> Self {
        Self {
            base: Jet2::from_parts(value, &gradient, &[]),
            eps: Jet2::from_parts(directional_value, &directional_gradient, &[]),
        }
    }

    fn owned_base_terms(&self) -> FlexFamilyCoefficientTerms {
        owned_jet2_terms(&self.base)
    }

    fn owned_directional_terms(&self) -> Option<FlexFamilyCoefficientTerms> {
        Some(owned_jet2_terms(&self.eps))
    }
}

/// Arena-backed one-seed scalar for the production directional timepoint
/// program. It implements the same [`FlexJet`] algebra as [`Jet3`], while every
/// derivative channel is allocated from one resettable row arena instead of a
/// fresh `Vec` per primitive operation.
#[derive(Clone, Copy)]
struct ArenaJet3<'arena> {
    arena: &'arena DynamicJetArena,
    inner: DynamicOneSeed<'arena>,
}

impl<'arena> ArenaJet3<'arena> {
    #[inline]
    fn primary(
        x: f64,
        axis: usize,
        p: usize,
        direction: f64,
        arena: &'arena DynamicJetArena,
    ) -> Self {
        let inner = if axis < p {
            DynamicOneSeed::seed_direction(x, axis, direction, p, arena)
        } else {
            DynamicOneSeed {
                base: DynamicOrder2::constant(x, p, arena),
                eps: DynamicOrder2::constant(direction, p, arena),
            }
        };
        Self { arena, inner }
    }
}

impl JetField for ArenaJet3<'_> {
    #[inline(always)]
    fn value(&self) -> f64 {
        self.inner.value()
    }

    #[inline(always)]
    fn add(&self, other: &Self) -> Self {
        Self {
            arena: self.arena,
            inner: self.inner.add(&other.inner),
        }
    }

    #[inline(always)]
    fn sub(&self, other: &Self) -> Self {
        Self {
            arena: self.arena,
            inner: self.inner.sub(&other.inner),
        }
    }

    #[inline(always)]
    fn mul(&self, other: &Self) -> Self {
        Self {
            arena: self.arena,
            inner: self.inner.mul(&other.inner),
        }
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        Self {
            arena: self.arena,
            inner: self.inner.neg(),
        }
    }

    #[inline(always)]
    fn scale(&self, scale: f64) -> Self {
        Self {
            arena: self.arena,
            inner: self.inner.scale(scale),
        }
    }

    #[inline(always)]
    fn compose_unary(&self, derivatives: [f64; 5]) -> Self {
        Self {
            arena: self.arena,
            inner: self.inner.compose_unary(derivatives),
        }
    }
}

impl FlexJet for ArenaJet3<'_> {
    const ORDER: usize = 3;

    #[inline]
    fn scale_homogeneous_orders(&self, factors: [f64; 5]) -> Self {
        let dimension = self.inner.base.g.len();
        let scale_order2 = |channels: &DynamicOrder2<'_>, offset: usize| {
            DynamicOrder2::from_channel_functions(
                factors[offset] * channels.v,
                dimension,
                self.arena,
                |axis| factors[offset + 1] * channels.g[axis],
                |row, column| factors[offset + 2] * channels.h[row * dimension + column],
            )
        };
        Self {
            arena: self.arena,
            inner: DynamicOneSeed {
                base: scale_order2(&self.inner.base, 0),
                eps: scale_order2(&self.inner.eps, 1),
            },
        }
    }
}

/// Inline fixed-width one-seed scalar for the common eight-primary FLEX row.
/// It instantiates the same generic timepoint program as [`ArenaJet3`], but the
/// compiler can scalarize every value/gradient/Hessian channel and eliminate
/// runtime-width indexing and arena traffic.
#[derive(Clone, Copy)]
struct FixedJet3<const K: usize> {
    inner: OneSeed<K>,
}

impl<const K: usize> FixedJet3<K> {
    #[inline]
    fn primary(x: f64, axis: usize, p: usize, direction: f64) -> Self {
        assert_eq!(p, K, "fixed FLEX Jet3 width mismatch");
        let inner = if axis < K {
            OneSeed::seed_direction(x, axis, direction)
        } else {
            OneSeed {
                base: <Order2<K> as gam_math::jet_scalar::JetScalar<K>>::constant(x),
                eps: <Order2<K> as gam_math::jet_scalar::JetScalar<K>>::constant(direction),
            }
        };
        Self { inner }
    }
}

impl<const K: usize> JetField for FixedJet3<K> {
    #[inline(always)]
    fn value(&self) -> f64 {
        self.inner.value()
    }

    #[inline(always)]
    fn add(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.add(&other.inner),
        }
    }

    #[inline(always)]
    fn sub(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.sub(&other.inner),
        }
    }

    #[inline(always)]
    fn mul(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.mul(&other.inner),
        }
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        Self {
            inner: self.inner.neg(),
        }
    }

    #[inline(always)]
    fn scale(&self, scale: f64) -> Self {
        Self {
            inner: self.inner.scale(scale),
        }
    }

    #[inline(always)]
    fn compose_unary(&self, derivatives: [f64; 5]) -> Self {
        Self {
            inner: self.inner.compose_unary(derivatives),
        }
    }
}

impl<const K: usize> FlexJet for FixedJet3<K> {
    const ORDER: usize = 3;

    #[inline]
    fn scale_homogeneous_orders(&self, factors: [f64; 5]) -> Self {
        let scale_order2 = |channels: &Order2<K>, offset: usize| {
            let mut scaled = Tower2::<K>::zero();
            scaled.v = factors[offset] * channels.0.v;
            for axis in 0..K {
                scaled.g[axis] = factors[offset + 1] * channels.0.g[axis];
            }
            for row in 0..K {
                for column in 0..K {
                    scaled.h[row][column] = factors[offset + 2] * channels.0.h[row][column];
                }
            }
            Order2(scaled)
        };
        Self {
            inner: OneSeed {
                base: scale_order2(&self.inner.base, 0),
                eps: scale_order2(&self.inner.eps, 1),
            },
        }
    }
}

trait FlexThirdOutput: FlexJet + MomentTerm {
    fn pack_timepoint_outputs(
        eta: &Self,
        chi: &Self,
        d: &Self,
    ) -> (FlexTimepointBasePack, FlexTimepointDirectionalPack);
}

impl FlexThirdOutput for ArenaJet3<'_> {
    fn pack_timepoint_outputs(
        eta: &Self,
        chi: &Self,
        d: &Self,
    ) -> (FlexTimepointBasePack, FlexTimepointDirectionalPack) {
        (
            FlexTimepointBasePack {
                eta: eta.inner.base.v,
                chi: chi.inner.base.v,
                d: d.inner.base.v,
                eta_u: eta.inner.base.g.to_vec(),
                eta_uv: eta.inner.base.h.to_vec(),
                chi_u: chi.inner.base.g.to_vec(),
                chi_uv: chi.inner.base.h.to_vec(),
                d_u: d.inner.base.g.to_vec(),
                d_uv: d.inner.base.h.to_vec(),
            },
            FlexTimepointDirectionalPack {
                eta_u_dir: eta.inner.eps.g.to_vec(),
                eta_uv_dir: eta.inner.eps.h.to_vec(),
                chi_u_dir: chi.inner.eps.g.to_vec(),
                chi_uv_dir: chi.inner.eps.h.to_vec(),
                d_u_dir: d.inner.eps.g.to_vec(),
                d_uv_dir: d.inner.eps.h.to_vec(),
            },
        )
    }
}

impl<const K: usize> FlexThirdOutput for FixedJet3<K> {
    fn pack_timepoint_outputs(
        eta: &Self,
        chi: &Self,
        d: &Self,
    ) -> (FlexTimepointBasePack, FlexTimepointDirectionalPack) {
        let flatten =
            |matrix: &[[f64; K]; K]| matrix.iter().flat_map(|row| row.iter().copied()).collect();
        (
            FlexTimepointBasePack {
                eta: eta.inner.base.0.v,
                chi: chi.inner.base.0.v,
                d: d.inner.base.0.v,
                eta_u: eta.inner.base.0.g.to_vec(),
                eta_uv: flatten(&eta.inner.base.0.h),
                chi_u: chi.inner.base.0.g.to_vec(),
                chi_uv: flatten(&chi.inner.base.0.h),
                d_u: d.inner.base.0.g.to_vec(),
                d_uv: flatten(&d.inner.base.0.h),
            },
            FlexTimepointDirectionalPack {
                eta_u_dir: eta.inner.eps.0.g.to_vec(),
                eta_uv_dir: flatten(&eta.inner.eps.0.h),
                chi_u_dir: chi.inner.eps.0.g.to_vec(),
                chi_uv_dir: flatten(&chi.inner.eps.0.h),
                d_u_dir: d.inner.eps.0.g.to_vec(),
                d_uv_dir: flatten(&d.inner.eps.0.h),
            },
        )
    }
}

// ── Jet4: two-seed, contracted fourth (doc §A.3) ───────────────────────────

/// An [`Jet2`] base plus ε, δ (`ε² = δ² = 0`, `εδ` retained) — four [`Jet2`]
/// parts. After seeding with both directions, the εδ-Hessian of the NLL is
/// `Σ_{cd} ℓ_{abcd} u_c v_d`.
#[derive(Clone)]
struct Jet4 {
    base: Jet2,
    eps: Jet2,
    del: Jet2,
    eps_del: Jet2,
}

impl Jet4 {
    fn primary(x: f64, axis: usize, p: usize, du: f64, dv: f64) -> Self {
        let zero = vec![0.0; p];
        Jet4 {
            base: Jet2::primary(x, axis, p),
            eps: Jet2::from_parts(du, &zero, &[]),
            del: Jet2::from_parts(dv, &zero, &[]),
            eps_del: Jet2::from_parts(0.0, &zero, &[]),
        }
    }
    fn contracted_fourth(&self) -> Vec<f64> {
        self.eps_del.h.clone()
    }
}

impl JetField for Jet4 {
    #[inline]
    fn value(&self) -> f64 {
        self.base.v
    }
    fn add(&self, o: &Self) -> Self {
        Jet4 {
            base: self.base.add(&o.base),
            eps: self.eps.add(&o.eps),
            del: self.del.add(&o.del),
            eps_del: self.eps_del.add(&o.eps_del),
        }
    }
    fn sub(&self, o: &Self) -> Self {
        Jet4 {
            base: self.base.sub(&o.base),
            eps: self.eps.sub(&o.eps),
            del: self.del.sub(&o.del),
            eps_del: self.eps_del.sub(&o.eps_del),
        }
    }
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
        Jet4 {
            base,
            eps,
            del,
            eps_del,
        }
    }
    fn scale(&self, s: f64) -> Self {
        Jet4 {
            base: self.base.scale(s),
            eps: self.eps.scale(s),
            del: self.del.scale(s),
            eps_del: self.eps_del.scale(s),
        }
    }
    #[inline]
    fn neg(&self) -> Self {
        self.scale(-1.0)
    }
    fn compose_unary(&self, d: [f64; 5]) -> Self {
        let base = self.base.compose_unary([d[0], d[1], d[2], d[3], d[4]]);
        let fprime = self.base.compose_unary([d[1], d[2], d[3], d[4], d[4]]);
        let fsecond = self.base.compose_unary([d[2], d[3], d[4], d[4], d[4]]);
        let eps = fprime.mul(&self.eps);
        let del = fprime.mul(&self.del);
        let eps_del = fsecond
            .mul(&self.eps)
            .mul(&self.del)
            .add(&fprime.mul(&self.eps_del));
        Jet4 {
            base,
            eps,
            del,
            eps_del,
        }
    }
}

impl FlexJet for Jet4 {
    const ORDER: usize = 4;

    #[inline]
    fn scale_homogeneous_orders(&self, factors: [f64; 5]) -> Self {
        Jet4 {
            base: self.base.scale_homogeneous_from(0, factors),
            eps: self.eps.scale_homogeneous_from(1, factors),
            del: self.del.scale_homogeneous_from(1, factors),
            eps_del: self.eps_del.scale_homogeneous_from(2, factors),
        }
    }
}

/// `Σ_i x[i]·y[i]` over equal-length slices.
#[inline]
fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
}

/// `out[i] = Σ_j m[i*p+j]·v[j]` for a row-major `p×p` matrix `m`.
fn mat_vec(m: &[f64], v: &[f64], p: usize) -> Vec<f64> {
    let mut out = vec![0.0; p];
    for i in 0..p {
        let mut acc = 0.0;
        for j in 0..p {
            acc += m[i * p + j] * v[j];
        }
        out[i] = acc;
    }
    out
}

/// `v1ᵀ m v2` for a row-major `p×p` matrix `m`.
fn quad_form(m: &[f64], v1: &[f64], v2: &[f64], p: usize) -> f64 {
    let mut acc = 0.0;
    for i in 0..p {
        let mi = &m[i * p..i * p + p];
        acc += v1[i] * dot(mi, v2);
    }
    acc
}

/// Order-≤2 jet channels (value, gradient view, optional Hessian view) for the
/// four flex row-NLL inputs (entry η, exit η, observed χ, observed d), bundled
/// so `flex_row_nll_value_grad_hess` stays under the argument-count gate.
pub(crate) struct FlexRowJet2Channels<'a> {
    pub eta0_v: f64,
    pub eta0_g: ndarray::ArrayView1<'a, f64>,
    pub eta0_h: Option<ndarray::ArrayView2<'a, f64>>,
    pub eta1_v: f64,
    pub eta1_g: ndarray::ArrayView1<'a, f64>,
    pub eta1_h: Option<ndarray::ArrayView2<'a, f64>>,
    pub chi1_v: f64,
    pub chi1_g: ndarray::ArrayView1<'a, f64>,
    pub chi1_h: Option<ndarray::ArrayView2<'a, f64>>,
    pub d1_v: f64,
    pub d1_g: ndarray::ArrayView1<'a, f64>,
    pub d1_h: Option<ndarray::ArrayView2<'a, f64>>,
}

/// Entry/exit base + directional timepoint packs for the contracted-third path,
/// bundled to keep `flex_row_nll_third_contracted` under the argument-count gate.
pub(crate) struct FlexThirdPacks<'a> {
    pub entry_base: &'a FlexTimepointBasePack,
    pub exit_base: &'a FlexTimepointBasePack,
    pub entry_ext: &'a FlexTimepointDirectionalPack,
    pub exit_ext: &'a FlexTimepointDirectionalPack,
}

/// Entry/exit base + both directional + bidirectional timepoint packs for the
/// contracted-fourth path, bundled to keep `flex_row_nll_fourth_contracted`
/// under the argument-count gate.
pub(crate) struct FlexFourthPacks<'a> {
    pub entry_base: &'a FlexTimepointBasePack,
    pub exit_base: &'a FlexTimepointBasePack,
    pub entry_ext_u: &'a FlexTimepointDirectionalPack,
    pub exit_ext_u: &'a FlexTimepointDirectionalPack,
    pub entry_ext_v: &'a FlexTimepointDirectionalPack,
    pub exit_ext_v: &'a FlexTimepointDirectionalPack,
    pub entry_bi: &'a FlexTimepointBidirectionalPack,
    pub exit_bi: &'a FlexTimepointBidirectionalPack,
}

impl SurvivalMarginalSlopeFamily {
    /// Single-source flex row value + gradient (+ Hessian if `hess_h*` non-empty)
    /// from the entry/exit timepoint packs. The Hessian channel is returned only
    /// when the `*_uv` slices are supplied; the grad-only caller passes empty
    /// `h` slices (the value/gradient channels do not read the Hessian).
    ///
    /// `g_*` are the length-`p` gradient packs, `h_*` the `p*p` row-major Hessian
    /// packs (empty for grad-only). Replaces the hand value/grad/Hessian
    /// assembly in `flex_sensitivity.rs`.
    pub(crate) fn flex_row_nll_value_grad_hess(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q1: f64,
        qd1: f64,
        ch: FlexRowJet2Channels<'_>,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let FlexRowJet2Channels {
            eta0_v,
            eta0_g,
            eta0_h,
            eta1_v,
            eta1_g,
            eta1_h,
            chi1_v,
            chi1_g,
            chi1_h,
            d1_v,
            d1_g,
            d1_h,
        } = ch;
        let p = primary.total;
        let wi = self.weights[row];
        let di = self.event[row];
        let surv0 = surv_stack(eta0_v)?;
        let surv1 = surv_stack(eta1_v)?;
        let want_hess = eta1_h.is_some();
        if !want_hess {
            // Grad-only: the value/gradient channels never read the Hessian, so a
            // value+gradient-only `Jet1` yields a `to_bits`-identical value+grad
            // while dropping the discarded `O(p²)` Hessian alloc + arithmetic.
            let eta0 = Jet1::from_view(eta0_v, eta0_g);
            let eta1 = Jet1::from_view(eta1_v, eta1_g);
            let chi1 = Jet1::from_view(chi1_v, chi1_g);
            let d1 = Jet1::from_view(d1_v, d1_g);
            let q1j = Jet1::primary(q1, primary.q1, p);
            let qd1j = Jet1::primary(qd1, primary.qd1, p);
            let out = flex_row_nll(&eta0, &eta1, &chi1, &d1, &q1j, &qd1j, surv0, surv1, wi, di);
            let value = out.v + wi * di * std::f64::consts::TAU.ln();
            let grad = Array1::from(out.g);
            return Ok((value, grad, Array2::zeros((p, p))));
        }
        // Compile the SAME seven-term outer plan the generic Jet1/Jet3/Jet4
        // evaluators consume, then lower its six independent sources in one
        // order-two pass. Only the returned gradient and Hessian allocate.
        let eta0_h = eta0_h.ok_or("flex order-two lowering: missing eta0 Hessian")?;
        let chi1_h = chi1_h.ok_or("flex order-two lowering: missing chi1 Hessian")?;
        let d1_h = d1_h.ok_or("flex order-two lowering: missing d1 Hessian")?;
        let eta1_h = eta1_h.ok_or("flex order-two lowering: missing eta1 Hessian")?;
        let plan = FlexOuterPlan::new(chi1_v, d1_v, qd1, surv0, surv1, wi, di);
        let (row_value, row_gradient, row_hessian) = lower_flex_outer_plan_order2(
            &plan,
            FlexOrder2Inputs {
                eta0: FlexOrder2View {
                    value: eta0_v,
                    gradient: eta0_g,
                    hessian: eta0_h,
                },
                eta1: FlexOrder2View {
                    value: eta1_v,
                    gradient: eta1_g,
                    hessian: eta1_h,
                },
                q1: (q1, primary.q1),
                chi1: FlexOrder2View {
                    value: chi1_v,
                    gradient: chi1_g,
                    hessian: chi1_h,
                },
                d1: FlexOrder2View {
                    value: d1_v,
                    gradient: d1_g,
                    hessian: d1_h,
                },
                qd1: (qd1, primary.qd1),
            },
            p,
        );
        let value = row_value + wi * di * std::f64::consts::TAU.ln();
        let grad = Array1::from(row_gradient);
        let hess = Array2::from_shape_vec((p, p), row_hessian).map_err(|e| e.to_string())?;
        Ok((value, grad, hess))
    }

    /// Single-source flex contracted third `D_dir H[u,v]` from the entry/exit
    /// base + directional packs through the canonical flex-row plan.
    pub(crate) fn flex_row_nll_third_contracted(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q1: f64,
        qd1: f64,
        dir: &[f64],
        packs: FlexThirdPacks<'_>,
    ) -> Result<Array2<f64>, String> {
        let FlexThirdPacks {
            entry_base,
            exit_base,
            entry_ext,
            exit_ext,
        } = packs;
        let p = primary.total;
        let wi = self.weights[row];
        let di = self.event[row];
        let surv0 = surv_stack(entry_base.eta)?;
        let surv1 = surv_stack(exit_base.eta)?;

        let mk =
            |base_v: f64, base_g: &[f64], base_h: &[f64], ext_g: &[f64], ext_h: &[f64]| -> Jet3 {
                Jet3 {
                    base: Jet2::from_parts(base_v, base_g, base_h),
                    eps: Jet2::from_parts(dot(base_g, dir), ext_g, ext_h),
                }
            };
        let eta0 = mk(
            entry_base.eta,
            &entry_base.eta_u,
            &entry_base.eta_uv,
            &entry_ext.eta_u_dir,
            &entry_ext.eta_uv_dir,
        );
        let eta1 = mk(
            exit_base.eta,
            &exit_base.eta_u,
            &exit_base.eta_uv,
            &exit_ext.eta_u_dir,
            &exit_ext.eta_uv_dir,
        );
        let chi1 = mk(
            exit_base.chi,
            &exit_base.chi_u,
            &exit_base.chi_uv,
            &exit_ext.chi_u_dir,
            &exit_ext.chi_uv_dir,
        );
        let d1 = mk(
            exit_base.d,
            &exit_base.d_u,
            &exit_base.d_uv,
            &exit_ext.d_u_dir,
            &exit_ext.d_uv_dir,
        );
        let q1j = Jet3::primary(q1, primary.q1, p, dir[primary.q1]);
        let qd1j = Jet3::primary(qd1, primary.qd1, p, dir[primary.qd1]);
        let out = flex_row_nll(&eta0, &eta1, &chi1, &d1, &q1j, &qd1j, surv0, surv1, wi, di);
        Array2::from_shape_vec((p, p), out.contracted_third()).map_err(|e| e.to_string())
    }

    /// Single-source flex contracted fourth `Σ_{cd} ℓ_{abcd} u_c v_d` from the
    /// entry/exit base + both directional packs + bidirectional packs. Replaces
    /// the same canonical flex-row plan at fourth order.
    pub(crate) fn flex_row_nll_fourth_contracted(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q1: f64,
        qd1: f64,
        dir_u: &[f64],
        dir_v: &[f64],
        packs: FlexFourthPacks<'_>,
    ) -> Result<Array2<f64>, String> {
        let FlexFourthPacks {
            entry_base,
            exit_base,
            entry_ext_u,
            exit_ext_u,
            entry_ext_v,
            exit_ext_v,
            entry_bi,
            exit_bi,
        } = packs;
        let p = primary.total;
        let wi = self.weights[row];
        let di = self.event[row];
        let surv0 = surv_stack(entry_base.eta)?;
        let surv1 = surv_stack(exit_base.eta)?;

        // eps_del.v = uᵀ·H·v, eps_del.g = (H_dir_u)·v, eps_del.h = bi.
        let mk = |base_v: f64,
                  base_g: &[f64],
                  base_h: &[f64],
                  ext_u_g: &[f64],
                  ext_u_h: &[f64],
                  ext_v_g: &[f64],
                  ext_v_h: &[f64],
                  bi_h: &[f64]|
         -> Jet4 {
            let eps_del_v = quad_form(base_h, dir_u, dir_v, p);
            let eps_del_g = mat_vec(ext_u_h, dir_v, p);
            Jet4 {
                base: Jet2::from_parts(base_v, base_g, base_h),
                eps: Jet2::from_parts(dot(base_g, dir_u), ext_u_g, ext_u_h),
                del: Jet2::from_parts(dot(base_g, dir_v), ext_v_g, ext_v_h),
                eps_del: Jet2::from_parts(eps_del_v, &eps_del_g, bi_h),
            }
        };
        let eta0 = mk(
            entry_base.eta,
            &entry_base.eta_u,
            &entry_base.eta_uv,
            &entry_ext_u.eta_u_dir,
            &entry_ext_u.eta_uv_dir,
            &entry_ext_v.eta_u_dir,
            &entry_ext_v.eta_uv_dir,
            &entry_bi.eta_uv_uv,
        );
        let eta1 = mk(
            exit_base.eta,
            &exit_base.eta_u,
            &exit_base.eta_uv,
            &exit_ext_u.eta_u_dir,
            &exit_ext_u.eta_uv_dir,
            &exit_ext_v.eta_u_dir,
            &exit_ext_v.eta_uv_dir,
            &exit_bi.eta_uv_uv,
        );
        let chi1 = mk(
            exit_base.chi,
            &exit_base.chi_u,
            &exit_base.chi_uv,
            &exit_ext_u.chi_u_dir,
            &exit_ext_u.chi_uv_dir,
            &exit_ext_v.chi_u_dir,
            &exit_ext_v.chi_uv_dir,
            &exit_bi.chi_uv_uv,
        );
        let d1 = mk(
            exit_base.d,
            &exit_base.d_u,
            &exit_base.d_uv,
            &exit_ext_u.d_u_dir,
            &exit_ext_u.d_uv_dir,
            &exit_ext_v.d_u_dir,
            &exit_ext_v.d_uv_dir,
            &exit_bi.d_uv_uv,
        );
        let q1j = Jet4::primary(q1, primary.q1, p, dir_u[primary.q1], dir_v[primary.q1]);
        let qd1j = Jet4::primary(qd1, primary.qd1, p, dir_u[primary.qd1], dir_v[primary.qd1]);
        let out = flex_row_nll(&eta0, &eta1, &chi1, &d1, &q1j, &qd1j, surv0, surv1, wi, di);
        Array2::from_shape_vec((p, p), out.contracted_fourth()).map_err(|e| e.to_string())
    }
}

// ── #932-2 PRODUCTION jet timepoint machinery (promoted from the
// `moment_engine_tests` oracle module) ──────────────────────────────────────
//
// The single-source flex timepoint `(eta, chi, d)` jet builder
// `flex_timepoint_inputs_generic` and its helpers (the `FlexJet` moment
// recurrence, intercept lift, cell-coefficient / chi-poly / moving-edge jets, and
// the observed / calibration input bridges) live here at module scope, consumed by
// the `compute_survival_timepoint_exact_jet` Jet2 wrapper below. Independent
// scalar-FD and nested-dual gates live in `moment_engine_tests`.

// #932: the `recip`/`exp`/`add_const` jet helpers (formerly `FlexJet` default
// methods) live here as free generic fns — only the relocated moment-engine /
// Phase-C builders below consume them, so keeping them inside the test module
// avoids the orphaned-`dead_code` gate while preserving the exact derivations.
fn recip<J: FlexJet>(x: &J) -> J {
    let v = x.value();
    let inv = 1.0 / v;
    let inv2 = inv * inv;
    x.compose_unary([
        inv,
        -inv2,
        2.0 * inv2 * inv,
        -6.0 * inv2 * inv2,
        24.0 * inv2 * inv2 * inv,
    ])
}
fn exp_jet<J: FlexJet>(x: &J) -> J {
    let e = x.value().exp();
    x.compose_unary([e, e, e, e, e])
}
fn add_const<J: FlexJet>(x: &J, c: f64) -> J {
    x.compose_unary([x.value() + c, 1.0, 0.0, 0.0, 0.0])
}

/// The calibration residual term `C·M` as a **distinguished-derivative
/// projector**, NOT an ordinary product. (`C = self` the coefficient jet,
/// `M = m` the moment jet.)
///
/// ## Why a projector and not `mul`
///
/// The calibration residual is `R = ∫ η e^{−q} dz = Σ_k C_k M_k`, and its
/// derivatives must equal the calibration constraint tensors `∂_θ R = ∫ η_θ
/// e^{−q}` whose FIRST (lead) index is forced onto the coefficient η. The
/// moment carries the `e^{−q}` motion (`M_a = −∫ z^k η η_a e^{−q}`), so when
/// both `C` and `M` move with the same θ, an ordinary jet product
/// `tangent(C)·M` double-counts the shared η-motion: at order n it gives
/// every `(j,m)` split the binomial weight, which is too large by `(j+m)/j`.
///
/// ## The exact law (distinguished-derivative averaging)
///
/// Average over which of the `r = |I|` derivative slots is the distinguished
/// (lead) one; a term `C_A M_B` (A on the coefficient, B on the moment)
/// survives iff the lead slot lies in A, which happens with probability
/// `|A|/|I|`:
///
/// ```text
///   P_I(C,M) = Σ_{A⊔B=I, A≠∅}  (|A| / |I|)  C_A M_B ,   weight j/(j+m), j=|A|.
/// ```
///
/// Orders 1–4 (the `Jet2`/`Jet3`/`Jet4` impls below realise exactly these):
///
/// ```text
///   P_i    = C_i M
///   P_ij   = C_ij M + ½(C_i M_j + C_j M_i)
///   P_ijk  = C_ijk M + ⅔ Σ C_ij M_k + ⅓ Σ C_i M_jk
///   P_ijkl = C_ijkl M + ¾ Σ C_ijk M_l + ½ Σ C_ij M_kl + ¼ Σ C_i M_jkl
/// ```
///
/// Along a scalar path the law collapses to the closed form
/// `P_n = Σ_j C(n,j)·(j/n)·C^(j)M^(n−j) = Σ_j C(n−1,j−1) C^(j)M^(n−j)
///      = d^(n−1)/dt^(n−1) (C′M)` — i.e. `½/⅔,⅓/¾,½,¼` are not empirical
/// fudge factors but `binom(n−1,j−1)/binom(n,j)`. The implementation below
/// generates them for every channel as `E⁻¹((E C)M)`; no order-specific table
/// remains. It is verified channel-for-channel against the true `R_ij…`
/// integrals (gam#932).
///
/// `moment_term` was formerly a `FlexJet` trait method, but the production
/// single-source NLL assembles its residual directly. The private extension
/// trait expresses the projector once for every production and oracle algebra.
trait MomentTerm: FlexJet {
    fn moment_term(&self, moment: &Self) -> Self {
        // Let E be the Euler operator that multiplies every homogeneous
        // derivative channel of order j by j. For total order n=j+m,
        //
        //     E⁻¹((E C) M)
        //
        // assigns the ordinary Leibniz split C_A M_B the required
        // distinguished-slot weight j/n. The jet product supplies every
        // partition and its multiplicity; the two order maps below therefore
        // generate the complete projector for every represented order without
        // an order-specific coefficient table.
        const EULER: [f64; 5] = [0.0, 1.0, 2.0, 3.0, 4.0];
        const EULER_INVERSE: [f64; 5] = [0.0, 1.0, 0.5, 1.0 / 3.0, 0.25];
        self.scale_homogeneous_orders(EULER)
            .mul(moment)
            .scale_homogeneous_orders(EULER_INVERSE)
    }
}

impl<J: FlexJet> MomentTerm for J {}

/// #932 item-2 Phase B-base: the normalization base moments `M_0..M_4` as jets,
/// carrying their exact θ-derivatives (incl. the moving-edge flux), built from
/// the cell's already-computed NUMERIC moment vector (`numeric_moments`) plus the
/// cell-coefficient jets `c` and the moving edge jets `(z_left, z_right)`.
///
/// `M_n = ∫_{z_L(θ)}^{z_R(θ)} zⁿ e^{−q(z,θ)} dz`, `q = ½(z² + η(z)²)`, `η = c0+c1z
/// +c2z²+c3z³` with `(c, z_L, z_R)` all θ-dependent.
///
/// This single-sources the hand `survival_flex_base_d_u`/`_d_uv`/`f_au`/`f_aa`
/// base normalization derivatives over a generic `FlexJet` order — exact to ALL
/// jet orders (Jet2/Jet3/Jet4), not just first. The value channel is
/// bit-identical to `numeric_moments[n]`; the derivative channels are
/// finite-difference-pinned against `evaluate_cell_moments` on perturbed cells
/// (`base_moment_jets_first_derivative_matches_fd_932`,
/// `base_moment_jets_second_derivative_matches_fd_932`).
///
/// EXACTNESS to all orders (the self-consistent closure): write
/// `M_n(θ) = ∫ zⁿ e^{−q(z,θ)} dz = ∫ zⁿ e^{−q(z,θ₀)}·e^{−Δq(z)} dz`,
/// `Δq(z) = q(z,θ) − q(z,θ₀) = ½(η(z,θ)² − η(z,θ₀)²)` (the `z²` term cancels).
/// The factor `e^{−Δq}` has VALUE channel 1 (Δq=0 at θ₀) and its derivative
/// channels carry the full `(−∂q)` / `(−∂²q + (∂q)²)` / … expansion. Expanding
/// `e^{−Δq}` as a jet-coefficient polynomial in `z` (`S(z)=Σ_m S_m zᵐ`, `S_m`
/// jets) and dotting against the NUMERIC moments gives the interior
/// `Σ_m S_m·M_{n+m}^{numeric}` — exact to every order because the `e^{−Δq}`
/// expansion already contains the `(∂q)²` cross-term and higher. The truncation
/// `e^{−Δq} ≈ Σ_{k≤J::ORDER} (−Δq)^k/k!` is exact in each instantiated
/// nilpotent algebra (`Δq` has value 0, so the next power only feeds derivative
/// orders that `J` does not represent). The boundary is the Leibniz flux
/// `+ f(z_R)·z_R' − f(z_L)·z_L'`,
/// integrand VALUE at the moving endpoint times the edge θ-velocity jet (exact to
/// all orders via the edge-jet algebra).
fn base_moment_jets<J: FlexJet>(
    c: &[J; 4],
    z_left: &J,
    left_finite: bool,
    z_right: &J,
    right_finite: bool,
    numeric_moments: &[f64],
) -> [J; 5] {
    assert!(
        (1..=4).contains(&J::ORDER),
        "base_moment_jets supports exact derivative orders 1 through 4"
    );
    let required_moments = 5 + 6 * J::ORDER;
    assert!(
        numeric_moments.len() >= required_moments,
        "order-{} base-moment jet requires numeric M_0..M_{}, got only {} moments",
        J::ORDER,
        required_moments - 1,
        numeric_moments.len()
    );

    // η₀ = value-only coefficient jets; jet-polynomial convolution helper.
    let c0_const: [J; 4] = std::array::from_fn(|k| const_jet_like(&c[k], c[k].value()));
    let conv = |lhs: &[J], rhs: &[J]| -> Vec<J> {
        let mut out: Vec<J> = (0..lhs.len() + rhs.len() - 1)
            .map(|_| const_jet_like(&c[0], 0.0))
            .collect();
        for (i, li) in lhs.iter().enumerate() {
            for (j, rj) in rhs.iter().enumerate() {
                out[i + j] = out[i + j].add(&li.mul(rj));
            }
        }
        out
    };
    // −Δq(z) = −½(η² − η₀²), a jet-coefficient polynomial in z (value channel 0).
    let eta_sq = conv(c, c);
    let eta0_sq = conv(&c0_const, &c0_const);
    let neg_dq: Vec<J> = eta_sq
        .iter()
        .zip(eta0_sq.iter())
        .map(|(a, b)| a.sub(b).scale(-0.5))
        .collect();
    // S(z) = e^{−Δq} = Σ_{k=0}^{J::ORDER} (−Δq)^k / k!
    // (jet-coefficient polynomial),
    // splitting e^{−q(θ)} = e^{−q0}·e^{−Δq}, Δq = q(θ)−q0. Truncating at
    // p=J::ORDER is EXACT: value(−Δq)=0 ⇒ −Δq ∈ m (nilpotent) ⇒
    // (−Δq)^{p+1} = 0. Using the instantiated order is not an approximation:
    // evaluating the order-four polynomial in Jet1/2/3 previously spent most
    // of its time constructing channels that are identically zero.
    //
    // MOMENT-DEGREE BUDGET. η is cubic ⇒ deg_z(Δq) ≤ 6, so deg_z(S) ≤ 6p, and
    // the interior dot `Σ_m S_m·M_{n+m}` below reaches `M_{n+6p}`. An order-`p`
    // jet for `M_n` therefore needs numeric base moments through `n + 6p`: for
    // p=4 that is `n+24` (n≤4 base moments → M_28; n≤3 calibration → M_27). The
    // cached partition builds to 32 (margin), which is why 27/32 are not magic.
    let mut s_poly: Vec<J> = vec![const_jet_like(&c[0], 1.0)];
    let mut power: Vec<J> = s_poly.clone();
    let factorials = [1.0_f64, 1.0, 2.0, 6.0, 24.0];
    for fact in factorials.iter().take(J::ORDER + 1).skip(1) {
        power = conv(&power, &neg_dq);
        for (m, coeff) in power.iter().enumerate() {
            let term = coeff.scale(1.0 / fact);
            if m < s_poly.len() {
                s_poly[m] = s_poly[m].add(&term);
            } else {
                s_poly.push(term);
            }
        }
    }
    // The interior `Σ_m S_m·M_{n+m}^{numeric}` integrates `g(z,θ)=zⁿe^{−q(z,θ)}`
    // over the FIXED value-channel limits `[z_L0, z_R0]` (the numeric moments are
    // those fixed-limit integrals). The MOVING-limit correction is the thin
    // sliver `∫_{z_R0}^{z_R(θ)} g dz − ∫_{z_L0}^{z_L(θ)} g dz` (`edge_sliver_jet`),
    // exact to all jet orders.
    //
    // SHARED-EDGE JUMP COLLAPSE: when the caller sums these per-cell moments over
    // a partition, an INTERIOR edge shared by cells `i`,`i+1` enters as `+sliver`
    // (cell i's right) and `−sliver` (cell i+1's left) at the SAME moving `z` with
    // the SAME `g` — so the two flux contributions telescope to zero automatically.
    // Only GENUINE boundaries (the timepoint Crossing edge with no cancel partner,
    // §D) survive. No hand jump/flux formula is needed; the cancellation is exact
    // in the jet algebra because both slivers are the identical jet with opposite
    // sign.
    std::array::from_fn(|n| {
        let mut acc = const_jet_like(&c[0], 0.0);
        for (m, s_m) in s_poly.iter().enumerate() {
            let m_npm = numeric_moments[n + m];
            if m_npm != 0.0 {
                acc = acc.add(&s_m.scale(m_npm));
            }
        }
        if let Some(sr) = edge_sliver_jet(n, c, z_right, right_finite) {
            acc = acc.add(&sr);
        }
        if let Some(sl) = edge_sliver_jet(n, c, z_left, left_finite) {
            acc = acc.sub(&sl);
        }
        acc
    })
}

/// The moving-edge sliver `∫_{z_E0}^{z_E(θ)} zⁿ e^{−q(z,θ)} dz` as a jet (value
/// 0, derivative channels = the §D moving-boundary flux to all orders). With
/// `δ = z_E − z_E0` (jet, value 0) and `g(z) = zⁿ e^{−q}`,
/// `∫_{z_E0}^{z_E} g dz = g·δ + ½ g_z δ² + ⅙ g_zz δ³ + (1/24) g_zzz δ⁴` (Taylor
/// in δ; the instantiated [`FlexJet::ORDER`] selects the exact prefix because
/// the next δ power vanishes in that nilpotent quotient). `g`, `g_z`, … are
/// evaluated at the FIXED edge `z_E0` but with the θ-dependent coefficient jets
/// `c`, so the sliver carries the full coefficient × edge cross-motion.
/// `q = ½(z² + η²)`,
/// `q_z = z + η η_z`, `η_z = c1 + 2c2 z + 3c3 z²`; the `g`-stack follows from
/// `g_z = (n/z − q_z) g` by the product/chain rule.
///
/// The four `gδ`/`½g_z δ²`/`⅙g_zz δ³`/`(1/24)g_zzz δ⁴` terms are JET products, so
/// the θ-jet of the sliver automatically contains every coefficient×edge cross
/// channel of the full Faà di Bruno expansion. Concretely, writing `d_k = δ^(k)`
/// and `G_r^[s] = ∂_t^s ∂_z^r g`, the 4th θ-derivative is
/// `S'''' = G_0 d_4 + 4G_0^[1] d_3 + 6G_0^[2] d_2 + 4G_0^[3] d_1 + 4G_1 d_1 d_3
///        + 3G_1 d_2² + 12G_1^[1] d_1 d_2 + 6G_1^[2] d_1² + 6G_2 d_1² d_2
///        + 4G_2^[1] d_1³ + G_3 d_1⁴`. So a 4th-ORDER-only crossing-edge mismatch
/// is NOT uniquely the `g_zzz δ⁴` term — it can equally be a wrong `z_4` (edge
/// 4th deriv) or a coefficient-edge CROSS channel (`G_1^[2] d_1²`, `G_2^[1] d_1³`,
/// `G_1 d_2²`). NOTE: the `n/z` `g`-stack form has a removable singularity at
/// `z_E0=0` (special-cased); the singularity-free polynomial form
/// `g_z = e^{−q}(n z^{n−1} − q_z z^n)`, `g_zz = e^{−q}(n(n−1)z^{n−2} − 2n q_z z^{n−1}
/// + (q_z²−q_zz)z^n)`, … is preferable when `z_E0` may be near 0.
fn edge_sliver_jet<J: FlexJet>(n: usize, c: &[J; 4], z_e: &J, finite: bool) -> Option<J> {
    if !finite {
        return None;
    }
    let z0 = z_e.value();
    let zc = const_jet_like(z_e, z0); // fixed edge, value-only
    // η at the fixed edge as a jet in c. Higher z-derivatives are constructed
    // lazily below only when the instantiated nilpotent order can consume them.
    let eta = c[3]
        .mul(&zc)
        .add(&c[2])
        .mul(&zc)
        .add(&c[1])
        .mul(&zc)
        .add(&c[0]);
    // g = zⁿ e^{−q}.
    let z_pow = {
        let mut zk = const_jet_like(z_e, 1.0);
        for _ in 0..n {
            zk = zk.mul(&zc);
        }
        zk
    };
    let q = zc.mul(&zc).add(&eta.mul(&eta)).scale(0.5);
    let w = exp_jet(&q.scale(-1.0));
    let g = z_pow.mul(&w);
    let delta = tangent_jet(z_e);
    let mut sliver = g.mul(&delta);
    if J::ORDER == 1 {
        return Some(sliver);
    }

    // η_z and q_z are needed starting at the δ² term.
    let eta_z = c[2]
        .scale(2.0)
        .add(&c[3].scale(3.0).mul(&zc))
        .mul(&zc)
        .add(&c[1]); // c1 + 2c2 z + 3c3 z²
    let q_z = zc.add(&eta.mul(&eta_z));
    // n/z^k constants (z held at the fixed edge); 0 when n=0 or z0=0.
    let nz = |power: i32| -> J {
        if n == 0 || z0 == 0.0 {
            const_jet_like(z_e, 0.0)
        } else {
            const_jet_like(z_e, n as f64 / z0.powi(power))
        }
    };
    // g_z/g = a1 = n/z − q_z.
    let a1 = nz(1).sub(&q_z);
    let g_z = a1.mul(&g);
    let d2 = delta.mul(&delta);
    sliver = sliver.add(&g_z.mul(&d2).scale(0.5));
    if J::ORDER == 2 {
        return Some(sliver);
    }

    // η_zz, q_zz and a1' first contribute through the δ³ term.
    let eta_zz = c[2].scale(2.0).add(&c[3].scale(6.0).mul(&zc)); // 2c2 + 6c3 z
    let q_zz = add_const(&eta_z.mul(&eta_z).add(&eta.mul(&eta_zz)), 1.0);
    let a1p = nz(2).scale(-1.0).sub(&q_zz);
    // g_zz/g = b2 = a1' + a1².
    let b2 = a1p.add(&a1.mul(&a1));
    let g_zz = b2.mul(&g);
    let d3 = d2.mul(&delta);
    sliver = sliver.add(&g_zz.mul(&d3).scale(1.0 / 6.0));
    if J::ORDER == 3 {
        return Some(sliver);
    }

    // The fourth-order quotient is the only one that can observe g_zzz·δ⁴.
    assert_eq!(
        J::ORDER,
        4,
        "edge sliver supports derivative orders 1 through 4"
    );
    let eta_zzz = c[3].scale(6.0); // 6c3
    let q_zzz = eta_z.scale(3.0).mul(&eta_zz).add(&eta.mul(&eta_zzz));
    let a1pp = nz(3).scale(2.0).sub(&q_zzz);
    // g_zzz/g = b2' + a1 b2, b2' = a1'' + 2 a1 a1'.
    let b2p = a1pp.add(&a1.mul(&a1p).scale(2.0));
    let g_zzz = b2p.add(&a1.mul(&b2)).mul(&g);
    let d4 = d3.mul(&delta);
    Some(sliver.add(&g_zzz.mul(&d4).scale(1.0 / 24.0)))
}

/// #932 item-2 STEP 3c: the GENERIC-order timepoint `(eta, chi, d)` builder over
/// ANY `FlexJet` order (`Jet2`/`Jet3`/`Jet4`). It consumes only jet algebra, so
/// instantiating it at `Jet3` (one directional seed) yields the directional
/// extension `D_dir(eta,chi,d)` in the `eps` channel, and at `Jet4` (two seeds)
/// the mixed second-directional `D_d1 D_d2` in the `eps_del` channel.
///
/// The caller pre-seeds `b_jet` (the slope `g` primary), `du[u]` (the unit
/// per-primary jets), `template` (a zero jet shaped at the right order/`p`), and
/// `q_jet` (the complete time-quantile jet, including both coefficient and
/// family-owned directions), plus `scale_ratio_jet` (the current probit scale
/// divided by the f64 scale already baked into the cached coefficient packs),
/// and
/// supplies the OBSERVED-channel jets `rho_jet`/`tau_jet` (the `h`/`w`/`infl`
/// linear channels added to `eta`/`chi`; pass zero jets for a pure `g` model,
/// where the full `(a,b)` observed-coeff pack already carries every `g` order).
/// The full `(a,b)` Taylor runs against the REAL `b_jet` here (no `db=0`), so the
/// `g`-axis is single-sourced through the pack at every order.
///
/// Returns the three output jets `(eta, chi, d)`; the caller extracts the value /
/// gradient / Hessian / directional channels it needs.
fn flex_timepoint_inputs_generic<J: FlexJet + MomentTerm>(
    template: &J,
    b_jet: &J,
    du: &[J],
    a0: f64,
    d_check: f64,
    primary_g: usize,
    infl: Option<usize>,
    q_jet: &J,
    scale_ratio_jet: &J,
    z_obs: f64,
    o_infl: f64,
    obs_coeff: [f64; 4],
    obs_fixed: &DenestedCellPrimaryFixedPartials,
    cells: &[CalibrationCellJetInputs<'_>],
) -> Result<(J, J, J), String> {
    // Intercept lift to order `J` (value/grad/Hess/… per the seed). The lift's
    // residual closure rebuilds the per-cell coefficient + moment jets at the
    // current iterate, so the lifted `a_jet` carries the intercept's full
    // θ-jet (incl. directional channels) to order `J` automatically.
    //
    // The filtered (frozen-inverse) Newton chord gains exactly ONE derivative
    // order per iteration, so the iterate count must reach the highest jet
    // order in play: 2 for `Jet2`, 3 for the `Jet3` directional Hessian, 4 for
    // the `Jet4` mixed-second-directional channel. Use that algebraic order
    // directly. Although a later correction is mathematically zero once every
    // represented order has converged, evaluating it still rebuilds the full
    // per-cell residual program; an unconditional fourth pass therefore wasted
    // one third of the Jet3 lift work and half of the Jet2 lift work. (A
    // hardcoded 2 left the Jet3/Jet4 mixed intercept derivatives one iteration
    // short — `eta_uv` converged but `eta_uv_dir` did not; gam#932.)
    let residual =
        |a: &J| calibration_residual_jet(a, b_jet, primary_g, du, q_jet, scale_ratio_jet, cells);
    let a_jet = lift_intercept_flex(template, a0, 1.0 / d_check, J::ORDER, residual);

    // Observed eta/chi: the OBSERVED cell coefficient `c_k(a, {θ_u})` and its
    // `∂_a` (= χ) built as MULTIVARIATE jets over ALL primaries (g/h/w) via
    // `cell_coeff_jets`/`cell_chi_poly_jets` on the OBSERVED-point fixed pack
    // `obs_fixed` (the analogue of the calibration cells' pack: `coeff_u[g]=dc_db`,
    // `coeff_u[h]=b·H(z_obs)`, `coeff_u[w]=link_basis(a,b)`, with their a/b
    // partials). Composing with the lifted `a_jet` + the directional `du` seeds
    // carries the h/w cross-derivatives to ALL orders automatically — replacing
    // the (a,b)-only `observed_coeff_component_jet` + frozen-scalar channels.
    // `eta = Σ_k c_k·z_obs^k + o_infl (+ the infl primary's unit partial)`.
    let da = tangent_jet(&a_jet);
    let eta_coeff_base = cell_coeff_jets(&a_jet, obs_coeff, obs_fixed, primary_g, &da, du);
    let chi_coeff_base = cell_chi_poly_jets(&a_jet, obs_fixed, primary_g, &da, du);
    let eta_coeff =
        std::array::from_fn(|coefficient| eta_coeff_base[coefficient].mul(scale_ratio_jet));
    let chi_coeff =
        std::array::from_fn(|coefficient| chi_coeff_base[coefficient].mul(scale_ratio_jet));
    let mut eta = add_const(&eval_coeff_jet_at(&eta_coeff, z_obs), o_infl);
    if let Some(infl_axis) = infl {
        // ∂η₁/∂o_infl = 1: the absorbed-influence offset shifts η₁ additively
        // (#461), independent of the calibration cells, so its only partial is
        // the unit slope on its own primary.
        eta = eta.add(&du[infl_axis]);
    }
    let chi = eval_coeff_jet_at(&chi_coeff, z_obs);

    // D normalization = Σ_cells INV_TWO_PI·Σ_k χ_k·M_k, with the cell coeff /
    // chi-poly jets through the lifted `a_jet` (the `da` tangent above) and the
    // moving-edge jets through `(a_jet, b_jet)`.
    let mut d = const_jet_like(template, 0.0);
    for cell in cells {
        let c_pos_base =
            cell_coeff_jets(&a_jet, cell.base_pos_coeffs, cell.fixed, primary_g, &da, du);
        let chi_jets_base = cell_chi_poly_jets(&a_jet, cell.fixed, primary_g, &da, du);
        let c_pos = std::array::from_fn(|coefficient| c_pos_base[coefficient].mul(scale_ratio_jet));
        let chi_jets =
            std::array::from_fn(|coefficient| chi_jets_base[coefficient].mul(scale_ratio_jet));
        let edge_l = cell_edge_jet(&a_jet, b_jet, cell.left_edge, cell.cell_left);
        let edge_r = cell_edge_jet(&a_jet, b_jet, cell.right_edge, cell.cell_right);
        d = d.add(&flex_timepoint_d_cell(
            template,
            &c_pos,
            &chi_jets,
            &edge_l,
            cell.cell_left.is_finite(),
            &edge_r,
            cell.cell_right.is_finite(),
            cell.numeric_moments,
        ));
    }

    Ok((eta, chi, d))
}

/// A value-zero "tangent" jet `x_jet − x.value()`: value 0, derivative channels
/// preserved. Used as the perturbation argument of the bivariate Taylor below.
#[inline]
fn tangent_jet<J: FlexJet>(x: &J) -> J {
    add_const(x, -x.value())
}

/// A constant jet (value `v`, all derivative channels zero), shaped like
/// `template` (so it carries the right runtime primary count).
#[inline]
fn const_jet_like<J: FlexJet>(template: &J, v: f64) -> J {
    add_const(&template.scale(0.0), v)
}

/// #932 item-2 Phase C STEP 2: the generic-order intercept Newton lift over a
/// runtime `FlexJet` — the linchpin that produces the 3rd/4th intercept
/// θ-derivatives the base Hessian lacks. Mirrors `lift_intercept_order2` /
/// `filtered_implicit_solve_scalar` but over a runtime `Jet2`/`Jet3`/`Jet4`.
///
/// The calibration constraint `F(a(θ), θ) = 0` is solved by the filtered Newton
/// step `A ← A − R(A)·inv_fa` (`inv_fa = 1/D`, `D = |F_a|`), iterated `iters`
/// times (the jet nilpotency order). `R(A)` is the calibration RESIDUAL JET built
/// by the caller-supplied `residual` closure from the per-cell coefficient jets
/// and moment jets:
///
///   R(A) = Σ_cells INV_TWO_PI · Σ_k tangent(c_posₖ(A)) · Mₖ(A)   (+ q self-term)
///
/// where `c_posₖ(A)` are the POSITIVE cell coefficients as jets in `A` (and the
/// primaries) and `Mₖ(A)` the cell's normalization moment jets. This is the EXACT
/// calibration θ-jet to all orders: `∂_θ R = INV_TWO_PI ∫ η_θ e^{−q} = −f_u`,
/// `∂²_θ R = INV_TWO_PI ∫ (η_θθ − η η_θ²) e^{−q}` (the `−η η_θ²` falling out of
/// `Mₖ`'s own `e^{−Δq}` motion `M_θ = −∫ η η_θ e^{−q}`), reproducing the hand
/// `f_u`/`f_uv`/`f_aa` moment dots and their 3rd/4th extensions automatically. The
/// value channel is the scalar calibration `f` (driven to ~0 by seeding
/// `A.value = a0` from the scalar solve), so only the derivative channels solve.
///
/// ## What the iteration converges to: the implicit-function tower
///
/// The fixed point is the exact `a(θ)` of `F(a(θ),θ) = 0`. Differentiating
/// `F` repeatedly (with `F_{pq} = ∂^{p+q}F/∂a^p∂t^q` along a path, `A=a_1`,
/// `B=a_2`, `C=a_3`, `D=a_4`) gives the standard IFT recursion the jet recovers
/// channel-for-channel — only `F_a·a_n` carries `a_n` linearly, all else is the
/// already-known lower orders:
/// ```text
///   a_i   = −F_i / F_a
///   a_ij  = −(F_ij + F_ai a_j + F_aj a_i + F_aa a_i a_j) / F_a
///   A = −F_01/F_10;  B = −(F_02 + 2F_11 A + F_20 A²)/F_10
///   C = −(F_03 + 3F_12 A + 3F_21 A² + F_30 A³ + 3(F_11+F_20 A)B)/F_10
///   D = −(F_04 + 4F_13 A + 6F_22 A² + 4F_31 A³ + F_40 A⁴
///         + 6F_12 B + 12F_21 A B + 6F_30 A² B + 3F_20 B²
///         + 4(F_11+F_20 A)C) / F_10
/// ```
///
/// ## Why an order-`p` jet needs exactly `p` iterations (NOT quadratic Newton)
///
/// `inv_fa` is the FROZEN scalar inverse `1/F_a(a0,0)` — its derivative
/// channels are dropped — so this is a chord/modified-Newton step, not true
/// Newton. Let `m` be the nilpotent ideal of the order-`p` jet algebra
/// (`m^{p+1} = 0`), and `e_r = A_r − a*` the jet error against the exact root.
/// Taylor-expanding `F(a*+e_r) = F_a·e_r + O(e_r²)`,
///
/// ```text
///   e_{r+1} = (1 − inv_fa·F_a(a*,θ))·e_r + O(e_r²).
/// ```
///
/// The constant part of `1 − inv_fa·F_a` vanishes (`inv_fa·F_a(a0,0) = 1`), so
/// `1 − inv_fa·F_a ∈ m`; and `e_r² ∈ m^{2k} ⊆ m^{k+1}`. Hence
/// `e_r ∈ m^k ⟹ e_{r+1} ∈ m^{k+1}`. The seed `A_0 = const(a0)` has no nilpotent
/// channels (`e_0 ∈ m`), so by induction `e_r ∈ m^{r+1}` and `e_p = 0`:
/// **each iteration recovers exactly one additional homogeneous Taylor degree.**
/// So `Jet2 → 2`, `Jet3 → 3`, `Jet4 → 4` (and any extra passes are exact no-ops,
/// since `R(a*) = 0` once converged). A hardcoded `2` left the Jet3/Jet4 mixed
/// intercept derivatives one+ iterations short (gam#932). Callers pass `iters = 4`.
///
/// `template` carries the runtime primary count; `a0` the solved intercept value.
fn lift_intercept_flex<J: FlexJet>(
    template: &J,
    a0: f64,
    inv_fa: f64,
    iters: usize,
    residual: impl Fn(&J) -> J,
) -> J {
    let mut a = const_jet_like(template, a0);
    for _ in 0..iters {
        let r = residual(&a);
        a = a.sub(&r.scale(inv_fa));
    }
    a
}

/// The per-row calibration residual jet `R(A)` for [`lift_intercept_flex`],
/// summed over a timepoint's cells: `Σ_cells INV_TWO_PI·Σ_k tangent(c_posₖ(A))·
/// Mₖ(A)` plus the q-marginal self-term carried by the complete generic `q_jet`
/// and the complete generic probit `scale_ratio_jet` (the
/// historical `f_u[q] += φ(q)` boundary term of the calibration). The cells are
/// supplied as `(base_pos_coeffs, fixed, edges, finiteness, numeric_moments)` so
/// the coefficient jets and moment jets are rebuilt at the current iterate `A`.
fn calibration_residual_jet<J: FlexJet + MomentTerm>(
    a_jet: &J,
    b_jet: &J,
    g_axis: usize,
    du: &[J],
    q_jet: &J,
    scale_ratio_jet: &J,
    cells: &[CalibrationCellJetInputs<'_>],
) -> J {
    let da = tangent_jet(a_jet);
    let inv_two_pi = std::f64::consts::TAU.recip();
    let mut r = const_jet_like(a_jet, 0.0);
    for cell in cells {
        // Positive cell coefficients as jets in (A, primaries).
        let c_pos_base = cell_coeff_jets(a_jet, cell.base_pos_coeffs, cell.fixed, g_axis, &da, du);
        let c_pos = std::array::from_fn(|coefficient| c_pos_base[coefficient].mul(scale_ratio_jet));
        // Moving edge jets: Crossing edges move with A/b, Fixed edges are static.
        let edge_l = cell_edge_jet(a_jet, b_jet, cell.left_edge, cell.cell_left);
        let edge_r = cell_edge_jet(a_jet, b_jet, cell.right_edge, cell.cell_right);
        let m = base_moment_jets(
            &c_pos,
            &edge_l,
            cell.cell_left.is_finite(),
            &edge_r,
            cell.cell_right.is_finite(),
            cell.numeric_moments,
        );
        // Σ_k moment_term(c_posₖ, Mₖ): the EXACT de-nested calibration residual
        // `∫ η_θ e^{−q}`. `moment_term` strips c's value (F's VALUE is carried by
        // the scalar seed) AND applies the `j/(j+m)` Leibniz weights so the lead
        // derivative always lands on the coefficient polynomial η — a plain
        // `tangent(c)·M` over-counts every split-derivative term by its binomial
        // weight, doubling the lifted intercept Hessian `a_uv` (gam#932 base gates).
        let mut cell_r = const_jet_like(a_jet, 0.0);
        for k in 0..4 {
            cell_r = cell_r.add(&c_pos[k].moment_term(&m[k]));
        }
        r = r.add(&cell_r.scale(inv_two_pi));
    }
    // q-marginal self-term, carried to ALL orders as the derivative channels of
    // `g(q) = Φ(−q)` composed with the q-primary jet `q_jet = q + δq`. The hand
    // adds, to the calibration F, `f_u[q] += φ(q)`, `f_uv[[q,q]] += −q·φ(q)`, and
    // the directional `f_uv_dir[[q,q]] += dir[q]·(q²−1)·φ(q)` (first_full.rs:711,
    // directional.rs:351). With `R = −F` and `g'(q)=−φ(q)`, `g''(q)=q·φ(q)`,
    // `g'''(q)=(1−q²)·φ(q)`, `g''''(q)=(q³−3q)·φ(q)`, ADDING `g(q_jet)` (minus its
    // value, to keep this term's value contribution 0 as the scalar seed already
    // drives R≈0) reproduces every order: grad[q]=−φ(q), Hess[q,q]=q·φ(q), and the
    // ε/εδ channels carry the directional `(q²−1)φ` / `(q³−3q)φ` q-self terms the
    // FLAT `−φ(q)·δq` form dropped (the bug the Jet3/Jet4 gates pin).
    let q = q_jet.value();
    let phi_q = crate::probability::normal_pdf(q);
    let g0 = crate::probability::normal_cdf(-q);
    let g1 = -phi_q;
    let g2 = q * phi_q;
    let g3 = (1.0 - q * q) * phi_q;
    let g4 = (q * q * q - 3.0 * q) * phi_q;
    let q_self = add_const(&q_jet.compose_unary([g0, g1, g2, g3, g4]), -g0);
    r = r.add(&q_self);
    r
}

/// Per-cell inputs for [`calibration_residual_jet`]: the positive base
/// coefficients, the fixed-partial pack, the cell edges (location + provenance),
/// and the numeric moment vector. Borrowed from the cached partition.
struct CalibrationCellJetInputs<'a> {
    base_pos_coeffs: [f64; 4],
    fixed: &'a DenestedCellPrimaryFixedPartials,
    cell_left: f64,
    cell_right: f64,
    left_edge: crate::cubic_cell_kernel::PartitionEdge,
    right_edge: crate::cubic_cell_kernel::PartitionEdge,
    numeric_moments: &'a [f64],
}

/// The moving cell-edge `z` as a jet: a `Crossing { tau }` edge sits at
/// `z = (τ − a)/b` and moves with the intercept jet `a_jet` and slope jet
/// `b_jet`; a `Fixed(z)` edge is static (a constant jet, no θ-motion).
///
/// Evaluating `(τ−a)·(1/b)` in the jet algebra reproduces the entire §C
/// crossing-edge velocity recursion for free — no hand flux formula. From the
/// defining identity `b·z = τ − a`, differentiating `n` times along a path
/// gives `Σ_{k=0}^n binom(n,k) b^(k) z^(n−k) = τ^(n) − a^(n)`, i.e.
/// `z^(n) = (τ^(n) − a^(n) − Σ_{k=1}^n binom(n,k) b^(k) z^(n−k)) / b`
/// (`z_1 = (τ_1−a_1−b_1 z)/b`, …, `z_4 = (τ_4−a_4−4b_1 z_3−6b_2 z_2−4b_3 z_1−b_4 z)/b`).
/// That is exactly what `sub` + `mul(&recip(b))` compute channel-for-channel,
/// so the moving-boundary edge velocities the hand `directional`/`bidirectional`
/// assemble by explicit flux drop out of the seed.
fn cell_edge_jet<J: FlexJet>(
    a_jet: &J,
    b_jet: &J,
    edge: crate::cubic_cell_kernel::PartitionEdge,
    z_value: f64,
) -> J {
    match edge {
        crate::cubic_cell_kernel::PartitionEdge::Crossing { tau } => {
            // z = (τ − a)·(1/b).
            const_jet_like(a_jet, tau).sub(a_jet).mul(&recip(b_jet))
        }
        crate::cubic_cell_kernel::PartitionEdge::Fixed(_) => const_jet_like(a_jet, z_value),
    }
}

/// The per-cell de-nested coefficient `c_k` (k = 0..4) as a jet, built from the
/// cell's `DenestedCellPrimaryFixedPartials` pack composed with the intercept
/// perturbation `da = tangent(a_jet)` and the per-primary perturbations
/// `du[u] = tangent(primary_u)`. This is the cell analogue of
/// `observed_coeff_component_jet`, carrying ALL primaries (not just a,b): it is
/// the multivariate Taylor of `c_k(a, {θ_u})` whose cross-partials are the pack
/// fields. Matches the hand `eta_u_poly`/`eta_uv_poly`/`chi_*` assembly in
/// `first_full`/`directional`/`bidirectional` term for term — the FlexJet algebra
/// raises it to the contracted third/fourth automatically.
///
/// Taylor structure (per k, `g_axis` = the slope `b` primary):
///   c = c0
///     + dc_da·da + ½dc_daa·da² + ⅙dc_daaa·da³                       (pure a)
///     + Σ_u coeff_u[u]·du                                           (pure u, lin)
///     + Σ_u coeff_au[u]·da·du + ½Σ_u coeff_aau[u]·da²·du            (a×u)
///     + Σ_u coeff_bu[u]·db·du + Σ_u coeff_abu[u]·da·db·du           (b×u)
///       + ½Σ_u coeff_bbu[u]·db²·du
///     + ⅙Σ_u coeff_aaau[u]·da³·du + ½Σ_u coeff_aabu[u]·da²·db·du    (3rd in a/b ×u)
///       + ½Σ_u coeff_abbu[u]·da·db²·du + ⅙Σ_u coeff_bbbu[u]·db³·du
/// where `db = du[g_axis]` (the slope perturbation). The `coeff_u`-family terms
/// are LINEAR in `du` (each cell coefficient is at most linear in any single
/// non-a/non-b primary), so no `du²` term is needed beyond the b-channel ones.
fn cell_coeff_jets<J: FlexJet>(
    template: &J,
    base_c: [f64; 4],
    fixed: &DenestedCellPrimaryFixedPartials,
    g_axis: usize,
    da: &J,
    du: &[J],
) -> [J; 4] {
    let p = du.len();
    let dada = da.mul(da);
    let dadada = dada.mul(da);
    let db = &du[g_axis];
    let dadb = da.mul(db);
    let dbdb = db.mul(db);
    std::array::from_fn(|k| {
        let mut c = const_jet_like(template, base_c[k]);
        // Pure-a chain.
        c = c
            .add(&da.scale(fixed.dc_da[k]))
            .add(&dada.scale(0.5 * fixed.dc_daa[k]))
            .add(&dadada.scale(fixed.dc_daaa[k] / 6.0));
        // Per-primary chains for the LINEAR axes (u != g): each cell coefficient
        // is at most linear in any single non-a/non-b primary, so every
        // `coeff_*u[u]·…·du[u]` term is genuinely BILINEAR in `du[u]` (factor 1,
        // off-diagonal — no Taylor factorial on `du[u]`). The slope axis `g`
        // (= `b`) is handled separately below because there `du[g] == db` appears
        // as a REPEATED factor: `coeff_bu[g]·db·du[g] = dc_dbb·db²` would
        // DOUBLE-count the pure-`b²` Taylor term (Jet2::mul gives `db·db` a
        // Hessian of 2 — gam#932 g-diagonal fix), and likewise `coeff_abu[g]`
        // (`a·b²`, needs ½) and `coeff_bbu[g]` (`b³`, needs ⅙ not ½).
        for u in 0..p {
            if u == g_axis {
                continue;
            }
            let duu = &du[u];
            let mut chain = duu.scale(fixed.coeff_u[u][k]);
            chain = chain
                .add(&da.mul(duu).scale(fixed.coeff_au[u][k]))
                .add(&dada.mul(duu).scale(0.5 * fixed.coeff_aau[u][k]));
            chain = chain
                .add(&db.mul(duu).scale(fixed.coeff_bu[u][k]))
                .add(&dadb.mul(duu).scale(fixed.coeff_abu[u][k]))
                .add(&dbdb.mul(duu).scale(0.5 * fixed.coeff_bbu[u][k]));
            chain = chain
                .add(&dadada.mul(duu).scale(fixed.coeff_aaau[u][k] / 6.0))
                .add(&dada.mul(db).mul(duu).scale(0.5 * fixed.coeff_aabu[u][k]))
                .add(&dadb.mul(db).mul(duu).scale(0.5 * fixed.coeff_abbu[u][k]))
                .add(&dbdb.mul(db).mul(duu).scale(fixed.coeff_bbbu[u][k] / 6.0));
            c = c.add(&chain);
        }
        // Slope-axis (`b`) chain with the correct Taylor factorials (the
        // `coeff_*u[g]` pack values are `dc_db`/`dc_dab`/`dc_dbb`/`dc_daab`/
        // `dc_dabb`/`dc_dbbb`; the third-order `aaau/aabu/abbu/bbbu[g]` are 0):
        //   dc_db·db + dc_dab·(da·db) + ½dc_daab·(da²·db)
        //            + ½dc_dbb·db² + ½dc_dabb·(da·db²) + ⅙dc_dbbb·db³.
        // This matches `observed_coeff_component_jet`'s b-terms exactly.
        c = c
            .add(&db.scale(fixed.coeff_u[g_axis][k]))
            .add(&dadb.scale(fixed.coeff_au[g_axis][k]))
            .add(&dada.mul(db).scale(0.5 * fixed.coeff_aau[g_axis][k]))
            .add(&dbdb.scale(0.5 * fixed.coeff_bu[g_axis][k]))
            .add(&dadb.mul(db).scale(0.5 * fixed.coeff_abu[g_axis][k]))
            .add(&dbdb.mul(db).scale(fixed.coeff_bbu[g_axis][k] / 6.0));
        c
    })
}

/// The per-cell `χ = ∂η/∂a` polynomial coefficients `dc_da[k]` (k = 0..4) as
/// jets, the `∂_a`-shifted analogue of [`cell_coeff_jets`]: the cell coefficient
/// family whose base is `dc_da`, whose `a`-derivatives are `dc_daa`/`dc_daaa`,
/// whose per-primary derivatives are `coeff_au`/`coeff_aau` (= `∂(dc_da)/∂u` and
/// `∂²(dc_da)/∂a∂u`), and whose `b`-cross is `coeff_abu` (= `∂²(dc_da)/∂b∂u`).
/// These are the `χ_u`/`χ_uv` chains the hand `first_full` assembles by hand
/// (`chi_u_poly = dc_daa·a_u + coeff_au`); the FlexJet algebra raises them.
fn cell_chi_poly_jets<J: FlexJet>(
    template: &J,
    fixed: &DenestedCellPrimaryFixedPartials,
    g_axis: usize,
    da: &J,
    du: &[J],
) -> [J; 4] {
    let p = du.len();
    let dada = da.mul(da);
    let db = &du[g_axis];
    std::array::from_fn(|k| {
        // Base = dc_da; a-chain = dc_daa·da + ½dc_daaa·da².
        let mut c = const_jet_like(template, fixed.dc_da[k]);
        c = c
            .add(&da.scale(fixed.dc_daa[k]))
            .add(&dada.scale(0.5 * fixed.dc_daaa[k]));
        // Linear axes (u != g): χ per-primary is bilinear in du[u] (factor 1).
        // The slope axis g (= b) is handled separately: there `coeff_abu[g]·db·
        // du[g] = dc_dabb·db²` repeats the b factor and would DOUBLE-count χ's
        // pure-`b²` Taylor term (gam#932 g-diagonal fix, mirroring cell_coeff_jets).
        let dbdb = db.mul(db);
        let dadb = da.mul(db);
        for u in 0..p {
            if u == g_axis {
                continue;
            }
            let duu = &du[u];
            // χ = ∂_a η, so χ's per-primary chain is ∂_a of η's per-primary chain
            // (`cell_coeff_jets`), dropping one a-order:
            //   coeff_au·du + coeff_aau·da·du + coeff_abu·db·du
            //   + ½coeff_aaau·da²·du + coeff_aabu·(da·db)·du + ½coeff_abbu·db²·du.
            // The three second-order terms are zero for a g-only family but
            // NON-zero on h/w channels, where χ_uv's directional (Jet3) / mixed
            // (Jet4) channel needs them — without them χ_uv_dir under-counts
            // (gam#932 ghw gate).
            let chain = duu
                .scale(fixed.coeff_au[u][k])
                .add(&da.mul(duu).scale(fixed.coeff_aau[u][k]))
                .add(&db.mul(duu).scale(fixed.coeff_abu[u][k]))
                .add(&dada.mul(duu).scale(0.5 * fixed.coeff_aaau[u][k]))
                .add(&dadb.mul(duu).scale(fixed.coeff_aabu[u][k]))
                .add(&dbdb.mul(duu).scale(0.5 * fixed.coeff_abbu[u][k]));
            c = c.add(&chain);
        }
        // Slope-axis (b) χ-chain with correct factorials: coeff_au[g]·db +
        // coeff_aau[g]·(da·db) + ½coeff_abu[g]·db²  (= dc_dab·db + dc_daab·da·db
        // + ½dc_dabb·db²); the higher χ b-terms (coeff for b³/a²b on g) are 0.
        c = c
            .add(&db.scale(fixed.coeff_au[g_axis][k]))
            .add(&da.mul(db).scale(fixed.coeff_aau[g_axis][k]))
            .add(&dbdb.scale(0.5 * fixed.coeff_abu[g_axis][k]));
        c
    })
}

/// #932 item-2 Phase C: the per-row density normalization `D = Σ_cells ∫ G0 dz`
/// (`G0 = χ·w`, `w = e^{−q}/2π`) as a jet at any `FlexJet` order, carrying its
/// exact θ-derivatives (the hand D-path `d_u`/`d_uv` are this jet's grad/Hess).
///
/// Per cell `D_cell = INV_TWO_PI · Σ_k χ_k · M_k`, where `χ_k` are the cell's
/// `dc_da` polynomial coefficients as jets ([`cell_chi_poly_jets`]) and `M_k` are
/// the cell's normalization moments as jets ([`base_moment_jets`], carrying both
/// the coefficient motion and the moving-edge sliver). The single-source magic:
/// the hand path forms `d_u` by EXPLICITLY assembling `χ_u − χ·η·η_u` + boundary
/// flux; the jet product `χ_k·M_k` reproduces all three terms automatically —
/// `χ_u` from `χ_k`'s motion, `−χ·η·η_u` from `M_k`'s interior `e^{−Δq}` factor
/// (`∂M_k = −Σ_m(η∂η)_m M_{k+m}`), and the boundary flux from `M_k`'s edge
/// sliver. `c_jets` are the cell's `c0..c3` jets ([`cell_coeff_jets`]) feeding the
/// moment exponent; `edge_l`/`edge_r` the moving edge jets; `moments` the cell's
/// NUMERIC moment vector (≥ `4 + 6` entries for the `e^{−Δq}` expansion).
fn flex_timepoint_d_cell<J: FlexJet>(
    template: &J,
    c_jets: &[J; 4],
    chi_jets: &[J; 4],
    edge_l: &J,
    left_finite: bool,
    edge_r: &J,
    right_finite: bool,
    numeric_moments: &[f64],
) -> J {
    let m = base_moment_jets(
        c_jets,
        edge_l,
        left_finite,
        edge_r,
        right_finite,
        numeric_moments,
    );
    let mut acc = const_jet_like(template, 0.0);
    for (k, chi_k) in chi_jets.iter().enumerate() {
        acc = acc.add(&chi_k.mul(&m[k]));
    }
    acc.scale(std::f64::consts::TAU.recip())
}

/// Evaluate a 4-coefficient cell polynomial jet `Σ_k coeff_jet[k]·z^k` at the
/// fixed observation point `z` (the jet image of `eval_coeff4_at`).
#[inline]
fn eval_coeff_jet_at<J: FlexJet>(coeff_jet: &[J; 4], z: f64) -> J {
    let mut zk = 1.0;
    let mut acc = const_jet_like(&coeff_jet[0], 0.0);
    for c in coeff_jet.iter() {
        acc = acc.add(&c.scale(zk));
        zk *= z;
    }
    acc
}

/// Build the `flex_timepoint_inputs_generic` cell inputs (`CalibrationCellJet
/// Inputs`) for a timepoint from a cached partition — the `cached → jet-inputs`
/// bridge the production cutover will promote. Borrows the cached cells.
fn cells_from_cached(cached: &CachedPartitionCells) -> Vec<CalibrationCellJetInputs<'_>> {
    cached
        .cells
        .iter()
        .map(|entry| {
            let cell = entry.partition_cell.cell;
            CalibrationCellJetInputs {
                base_pos_coeffs: [cell.c0, cell.c1, cell.c2, cell.c3],
                fixed: &entry.fixed,
                cell_left: cell.left,
                cell_right: cell.right,
                left_edge: entry.partition_cell.left_edge,
                right_edge: entry.partition_cell.right_edge,
                numeric_moments: entry.state.moments.as_slice(),
            }
        })
        .collect()
}

/// The OBSERVED-point coefficient + per-primary fixed-partial pack for the generic
/// builder — the observed-point analogue of `denested_cell_primary_fixed_partials`
/// (the calibration cells' pack). Returns `(obs_coeff, obs_fixed)` where:
///   - the `(a,b)` columns (the `g` slope axis) come from `observed_denested_cell
///     _partials` (`coeff_u[g]=dc_db`, `coeff_au[g]=dc_dab`, `coeff_bu[g]=dc_dbb`,
///     `coeff_aau[g]=dc_daab`, `coeff_abu[g]=dc_dabb`, `coeff_bbu[g]=dc_dbbb`),
///   - the score-warp `h` columns from `observed_score_basis_coefficients` at
///     `z_obs` (`coeff_u[h]=b·H(z_obs)`, `coeff_bu[h]=H(z_obs)`; a-independent, so
///     every `a`-cross column is zero),
///   - the link-dev `w` columns from `link_basis_cell_coefficients` at `u_obs`
///     (`coeff_u[w]`) and its first/second/third `(a,b)` partials.
/// Feeding this to `cell_coeff_jets`/`cell_chi_poly_jets` builds the observed
/// `eta`/`chi` as multivariate jets over g/h/w to ALL orders — the same machinery
/// the calibration cells / D path use. `scale` = the probit-frailty scale.
fn observed_fixed_for(
    family: &SurvivalMarginalSlopeFamily,
    primary: &FlexPrimarySlices,
    row: usize,
    a: f64,
    b: f64,
    beta_h: Option<&Array1<f64>>,
    beta_w: Option<&Array1<f64>>,
) -> Result<([f64; 4], DenestedCellPrimaryFixedPartials), String> {
    let r = primary.total;
    let scale = family.probit_frailty_scale();
    let z_obs = family.observed_score_projection(row);
    let u_obs = a + b * z_obs;
    let obs = family.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;

    let mut coeff_u = vec![[0.0; 4]; r];
    let mut coeff_au = vec![[0.0; 4]; r];
    let mut coeff_bu = vec![[0.0; 4]; r];
    let mut coeff_aau = vec![[0.0; 4]; r];
    let mut coeff_abu = vec![[0.0; 4]; r];
    let mut coeff_bbu = vec![[0.0; 4]; r];
    let mut coeff_aaau = vec![[0.0; 4]; r];
    let mut coeff_aabu = vec![[0.0; 4]; r];
    let mut coeff_abbu = vec![[0.0; 4]; r];
    let mut coeff_bbbu = vec![[0.0; 4]; r];

    // g (slope) axis = the observed (a,b) pack columns.
    coeff_u[primary.g] = obs.dc_db;
    coeff_au[primary.g] = obs.dc_dab;
    coeff_bu[primary.g] = obs.dc_dbb;
    coeff_aau[primary.g] = obs.dc_daab;
    coeff_abu[primary.g] = obs.dc_dabb;
    coeff_bbu[primary.g] = obs.dc_dbbb;

    // h (score-warp) axis: `coeff_h(z_obs) = b·H(z_obs)` — linear in b, a-free.
    if let Some(h_range) = primary.h.as_ref().filter(|_| family.score_warp.is_some()) {
        for local_idx in 0..h_range.len() {
            let idx = h_range.start + local_idx;
            coeff_u[idx] = scale_coeff4(
                family.observed_score_basis_coefficients(row, local_idx, z_obs, b)?,
                scale,
            );
            coeff_bu[idx] = scale_coeff4(
                family.observed_score_basis_coefficients(row, local_idx, z_obs, 1.0)?,
                scale,
            );
        }
    }

    // w (link-dev) axis: `coeff_w = link_basis(u_obs, a, b)` + its (a,b) partials.
    if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), family.link_dev.as_ref()) {
        for local_idx in 0..w_range.len() {
            let span = runtime.basis_cubic_at(local_idx, u_obs)?;
            let idx = w_range.start + local_idx;
            coeff_u[idx] = scale_coeff4(
                exact_kernel::link_basis_cell_coefficients(span, a, b),
                scale,
            );
            let (dc_aw, dc_bw) = exact_kernel::link_basis_cell_coefficient_partials(span, a, b);
            let (dc_aaw, dc_abw, dc_bbw) =
                exact_kernel::link_basis_cell_second_partials(span, a, b);
            let (dc_aaaw, dc_aabw, dc_abbw, dc_bbbw) =
                exact_kernel::link_basis_cell_third_partials(span);
            coeff_au[idx] = scale_coeff4(dc_aw, scale);
            coeff_bu[idx] = scale_coeff4(dc_bw, scale);
            coeff_aau[idx] = scale_coeff4(dc_aaw, scale);
            coeff_abu[idx] = scale_coeff4(dc_abw, scale);
            coeff_bbu[idx] = scale_coeff4(dc_bbw, scale);
            coeff_aaau[idx] = scale_coeff4(dc_aaaw, scale);
            coeff_aabu[idx] = scale_coeff4(dc_aabw, scale);
            coeff_abbu[idx] = scale_coeff4(dc_abbw, scale);
            coeff_bbbu[idx] = scale_coeff4(dc_bbbw, scale);
        }
    }

    let fixed = DenestedCellPrimaryFixedPartials {
        dc_da: obs.dc_da,
        dc_daa: obs.dc_daa,
        dc_daaa: obs.dc_daaa,
        coeff_u,
        coeff_au,
        coeff_bu,
        coeff_aau,
        coeff_abu,
        coeff_bbu,
        coeff_aaau,
        coeff_aabu,
        coeff_abbu,
        coeff_bbbu,
    };
    Ok((obs.coeff, fixed))
}

impl SurvivalMarginalSlopeFamily {
    /// #932-2 PRODUCTION cutover: the exact timepoint `(eta, chi, d)` value /
    /// gradient / Hessian via the single-source `flex_timepoint_inputs_generic`
    /// jet builder at [`Jet2`]. Independent finite differences pin its value,
    /// gradient, and Hessian channels in `moment_engine_tests`.
    pub(crate) fn compute_survival_timepoint_exact_jet(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        o_infl: f64,
    ) -> Result<SurvivalFlexTimepointExact, String> {
        let cached = self.build_cached_partition(primary, a, b, beta_h, beta_w)?;
        self.compute_survival_timepoint_exact_jet_from_cached(
            row, primary, q, q_index, a, b, beta_h, beta_w, o_infl, &cached,
        )
    }

    /// `compute_survival_timepoint_exact_jet` over a pre-built cached partition.
    pub(crate) fn compute_survival_timepoint_exact_jet_from_cached(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        o_infl: f64,
        cached: &CachedPartitionCells,
    ) -> Result<SurvivalFlexTimepointExact, String> {
        let p = primary.total;
        let d_check = self.evaluate_survival_denom_d(a, b, beta_h, beta_w)?;
        let z_obs = self.observed_score_projection(row);
        let (obs_coeff, obs_fixed) = observed_fixed_for(self, primary, row, a, b, beta_h, beta_w)?;
        let cells = cells_from_cached(cached);

        let template = Jet2::primary(0.0, usize::MAX, p);
        let b_jet = Jet2::primary(b, primary.g, p);
        let du: Vec<Jet2> = (0..p).map(|u| Jet2::primary(0.0, u, p)).collect();
        let q_jet = add_const(&du[q_index], q);
        let (eta, chi, d) = flex_timepoint_inputs_generic(
            &template,
            &b_jet,
            &du,
            a,
            d_check,
            primary.g,
            primary.infl,
            &q_jet,
            &const_jet_like(&template, 1.0),
            z_obs,
            o_infl,
            obs_coeff,
            &obs_fixed,
            &cells,
        )?;

        let to_g = |j: &Jet2| Array1::from(j.g.clone());
        let to_h = |j: &Jet2| -> Result<Array2<f64>, String> {
            Array2::from_shape_vec((p, p), j.h.clone()).map_err(|e| e.to_string())
        };
        Ok(SurvivalFlexTimepointExact {
            eta: eta.value(),
            chi: chi.value(),
            d: d.value(),
            eta_u: to_g(&eta),
            eta_uv: to_h(&eta)?,
            chi_u: to_g(&chi),
            chi_uv: to_h(&chi)?,
            d_u: to_g(&d),
            d_uv: to_h(&d)?,
        })
    }

    /// #932 grad-only single-source: the exact timepoint `(eta, chi, d)` VALUE +
    /// GRADIENT via the SAME single-source `flex_timepoint_inputs_generic` jet
    /// builder as [`Self::compute_survival_timepoint_exact_jet`], instantiated at
    /// [`Jet1`] (no second-order channel). This replaces the hand first-order
    /// cell-moment / IFT `a_u` / moving-flux `d_u` / `ρ`-`τ` `eta_u`/`chi_u`
    /// assembly that used to live in `first_full`: there is now ONE definition of
    /// the flex timepoint geometry, and the grad-only path is its order-≤1
    /// truncation. Because `Jet1` shares `add`/`sub`/`mul`/`scale`/`compose_unary`
    /// and the `moment_term` gradient line with `Jet2` op-for-op, the returned
    /// value/gradient are bit-identical to the [`Jet2`] path's base + gradient
    /// channels; the identity is pinned by
    /// `flex_timepoint_first_order_matches_jet2_and_fd_932`.
    pub(crate) fn compute_survival_timepoint_first_order_exact(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        o_infl: f64,
    ) -> Result<SurvivalFlexTimepointFirstOrderExact, String> {
        let cached = self.build_cached_partition(primary, a, b, beta_h, beta_w)?;
        let p = primary.total;
        let d_check = self.evaluate_survival_denom_d(a, b, beta_h, beta_w)?;
        let z_obs = self.observed_score_projection(row);
        let (obs_coeff, obs_fixed) = observed_fixed_for(self, primary, row, a, b, beta_h, beta_w)?;
        let cells = cells_from_cached(&cached);

        let template = Jet1::primary(0.0, usize::MAX, p);
        let b_jet = Jet1::primary(b, primary.g, p);
        let du: Vec<Jet1> = (0..p).map(|u| Jet1::primary(0.0, u, p)).collect();
        let q_jet = add_const(&du[q_index], q);
        let (eta, chi, d) = flex_timepoint_inputs_generic(
            &template,
            &b_jet,
            &du,
            a,
            d_check,
            primary.g,
            primary.infl,
            &q_jet,
            &const_jet_like(&template, 1.0),
            z_obs,
            o_infl,
            obs_coeff,
            &obs_fixed,
            &cells,
        )?;

        let to_g = |j: &Jet1| Array1::from(j.g.clone());
        Ok(SurvivalFlexTimepointFirstOrderExact {
            eta: eta.value(),
            chi: chi.value(),
            d: d.value(),
            eta_u: to_g(&eta),
            chi_u: to_g(&chi),
            d_u: to_g(&d),
        })
    }
}

// #932-2 increment 2: Jet3 directional and Jet4 mixed-second-directional
// channels instantiate the same Euler-generated `j/(j+m)` projector as the
// order-one/two and nested-dual algebras.

impl SurvivalMarginalSlopeFamily {
    /// #932-2 PRODUCTION cutover (increment 2): the directional timepoint
    /// extension `D_dir(eta_u/eta_uv/chi_u/chi_uv/d_u/d_uv)` via the single-source
    /// `flex_timepoint_inputs_generic` jet builder at [`Jet3`] (one nilpotent ε
    /// seed = the contraction direction). Returns the directional pack
    /// directly (the ε channel `.eps.g`/`.eps.h` of the `(eta, chi, d)` jets).
    /// The row-level contracted tensor is pinned by independent rigid-tower and
    /// finite-difference witnesses.
    pub(crate) fn compute_survival_timepoint_directional_jet_from_cached(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        o_infl: f64,
        cached: &CachedPartitionCells,
        dir: &Array1<f64>,
        arena: &DynamicJetArena,
    ) -> Result<(FlexTimepointBasePack, FlexTimepointDirectionalPack), String> {
        let p = primary.total;
        let d_check = self.evaluate_survival_denom_d(a, b, beta_h, beta_w)?;
        let z_obs = self.observed_score_projection(row);
        let (obs_coeff, obs_fixed) = observed_fixed_for(self, primary, row, a, b, beta_h, beta_w)?;
        let cells = cells_from_cached(cached);

        macro_rules! evaluate {
            ($template:expr, $b_jet:expr, $du:expr) => {{
                let template = $template;
                let b_jet = $b_jet;
                let du = $du;
                let q_jet = add_const(&du[q_index], q);
                let (eta, chi, d) = flex_timepoint_inputs_generic(
                    &template,
                    &b_jet,
                    &du,
                    a,
                    d_check,
                    primary.g,
                    primary.infl,
                    &q_jet,
                    &const_jet_like(&template, 1.0),
                    z_obs,
                    o_infl,
                    obs_coeff,
                    &obs_fixed,
                    &cells,
                )?;
                Ok(FlexThirdOutput::pack_timepoint_outputs(&eta, &chi, &d))
            }};
        }

        match p {
            8 => evaluate!(
                FixedJet3::<8>::primary(0.0, usize::MAX, p, 0.0),
                FixedJet3::<8>::primary(b, primary.g, p, dir[primary.g]),
                (0..p)
                    .map(|axis| FixedJet3::<8>::primary(0.0, axis, p, dir[axis]))
                    .collect::<Vec<_>>()
            ),
            _ => evaluate!(
                ArenaJet3::primary(0.0, usize::MAX, p, 0.0, arena),
                ArenaJet3::primary(b, primary.g, p, dir[primary.g], arena),
                (0..p)
                    .map(|axis| ArenaJet3::primary(0.0, axis, p, dir[axis], arena))
                    .collect::<Vec<_>>()
            ),
        }
    }

    /// #932-2 PRODUCTION cutover (increment 2): the mixed second-directional
    /// timepoint extension `D_{d1} D_{d2}(eta_uv/chi_uv/d_uv)` via the single-source
    /// builder at [`Jet4`] (two nilpotent seeds ε = `dir1`, δ = `dir2`). Returns the
    /// bidirectional pack directly (the εδ-Hessian channel `.eps_del.h`).
    /// Nested-dual and scalar finite-difference gates independently pin this
    /// fourth-order channel.
    pub(crate) fn compute_survival_timepoint_bidirectional_jet_from_cached(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        cached: &CachedPartitionCells,
        dir1: &Array1<f64>,
        dir2: &Array1<f64>,
    ) -> Result<FlexTimepointBidirectionalPack, String> {
        let p = primary.total;
        let d_check = self.evaluate_survival_denom_d(a, b, beta_h, beta_w)?;
        let z_obs = self.observed_score_projection(row);
        let (obs_coeff, obs_fixed) = observed_fixed_for(self, primary, row, a, b, beta_h, beta_w)?;
        let cells = cells_from_cached(cached);

        let template = Jet4::primary(0.0, usize::MAX, p, 0.0, 0.0);
        let b_jet = Jet4::primary(b, primary.g, p, dir1[primary.g], dir2[primary.g]);
        let du: Vec<Jet4> = (0..p)
            .map(|u| Jet4::primary(0.0, u, p, dir1[u], dir2[u]))
            .collect();
        let q_jet = add_const(&du[q_index], q);
        let (eta, chi, d) = flex_timepoint_inputs_generic(
            &template,
            &b_jet,
            &du,
            a,
            d_check,
            primary.g,
            primary.infl,
            &q_jet,
            &const_jet_like(&template, 1.0),
            z_obs,
            0.0,
            obs_coeff,
            &obs_fixed,
            &cells,
        )?;

        Ok(FlexTimepointBidirectionalPack {
            eta_uv_uv: eta.eps_del.h.clone(),
            chi_uv_uv: chi.eps_del.h.clone(),
            d_uv_uv: d.eps_del.h.clone(),
        })
    }
}

struct FlexFamilyCoefficientJets<J> {
    template: Dual2<J>,
    q0: Dual2<J>,
    q1: Dual2<J>,
    qd1: Dual2<J>,
    g: Dual2<J>,
    scale_ratio: Dual2<J>,
    du: Vec<Dual2<J>>,
    o_infl: f64,
}

enum FlexCoefficientRowDirection<'a> {
    /// Move the canonical flattened coefficient vector while holding every
    /// coefficient-to-row map fixed.
    Beta(&'a Array1<f64>),
    /// Move one coefficient-to-row design map.  The epsilon seed owns both
    /// `X_psi beta` and `X_psi`, so the resulting Jet3 channel includes the
    /// derivative of the score/Hessian pullback, not only row-value motion.
    Design {
        block: usize,
        derivative_row: &'a Array1<f64>,
        beta: &'a Array1<f64>,
        coefficient_range: std::ops::Range<usize>,
    },
}

fn add_coefficient_row(
    gradient: &mut [f64],
    range: &std::ops::Range<usize>,
    row: ndarray::ArrayView1<'_, f64>,
    channel: &str,
) -> Result<(), String> {
    if range.len() != row.len() {
        return Err(format!(
            "survival marginal-slope FLEX family {channel} coefficient row width {} != flattened range width {}",
            row.len(),
            range.len(),
        ));
    }
    for (axis, value) in range.clone().zip(row.iter().copied()) {
        gradient[axis] += value;
    }
    Ok(())
}

fn lifted_coefficient_affine<J: FlexCoefficientJet>(
    value: f64,
    gradient: Vec<f64>,
    coefficient_direction: Option<&FlexCoefficientRowDirection<'_>>,
    design_affected: bool,
    family_first: f64,
    family_second: f64,
) -> Dual2<J> {
    let dimension = gradient.len();
    let mut directional_gradient = vec![0.0; dimension];
    let directional_value = match coefficient_direction {
        None => 0.0,
        Some(FlexCoefficientRowDirection::Beta(direction)) => gradient
            .iter()
            .zip(direction.iter())
            .map(|(&coefficient, &step)| coefficient * step)
            .sum(),
        Some(FlexCoefficientRowDirection::Design {
            derivative_row,
            beta,
            coefficient_range,
            ..
        }) if design_affected => {
            for (axis, value) in coefficient_range
                .clone()
                .zip(derivative_row.iter().copied())
            {
                directional_gradient[axis] = value;
            }
            derivative_row.dot(*beta)
        }
        Some(FlexCoefficientRowDirection::Design { .. }) => 0.0,
    };
    Dual2 {
        v: J::affine(value, gradient, directional_value, directional_gradient),
        g: J::constant(family_first, dimension),
        h: J::constant(family_second, dimension),
    }
}

fn one_hot_coefficient_gradient(dimension: usize, axis: usize) -> Vec<f64> {
    let mut gradient = vec![0.0; dimension];
    gradient[axis] = 1.0;
    gradient
}

impl SurvivalMarginalSlopeFamily {
    fn validate_flex_family_coefficient_state(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
        first: FlexFamilyRowDirection,
        second: FlexFamilyRowDirection,
        coefficient_direction: Option<&FlexCoefficientRowDirection<'_>>,
    ) -> Result<(), String> {
        if row >= self.n {
            return Err(format!(
                "survival marginal-slope FLEX family row {row} is out of range for n={}",
                self.n
            ));
        }
        let directions = [
            first.entry,
            first.exit,
            first.derivative_exit,
            first.probit_scale,
            second.entry,
            second.exit,
            second.derivative_exit,
            second.probit_scale,
        ];
        if directions.iter().any(|value| !value.is_finite()) {
            return Err(
                "survival marginal-slope FLEX family row directions must be finite".to_string(),
            );
        }
        match coefficient_direction {
            Some(FlexCoefficientRowDirection::Beta(direction)) => {
                if direction.len() != slices.total {
                    return Err(format!(
                        "survival marginal-slope FLEX family beta direction length {} != flattened coefficient width {}",
                        direction.len(),
                        slices.total,
                    ));
                }
                if direction.iter().any(|value| !value.is_finite()) {
                    return Err(
                        "survival marginal-slope FLEX family beta direction must be finite"
                            .to_string(),
                    );
                }
            }
            Some(FlexCoefficientRowDirection::Design {
                block,
                derivative_row,
                beta,
                coefficient_range,
            }) => {
                if !matches!(*block, 1 | 2) {
                    return Err(format!(
                        "survival marginal-slope FLEX family design direction supports marginal/logslope blocks 1 or 2, got block {block}"
                    ));
                }
                let expected_range = if *block == 1 {
                    &slices.marginal
                } else {
                    &slices.logslope
                };
                if coefficient_range != expected_range {
                    return Err(format!(
                        "survival marginal-slope FLEX family design direction block {block} range {:?} != canonical {:?}",
                        coefficient_range, expected_range,
                    ));
                }
                if derivative_row.len() != coefficient_range.len()
                    || beta.len() != coefficient_range.len()
                {
                    return Err(format!(
                        "survival marginal-slope FLEX family design direction block {block} widths disagree: derivative={}, beta={}, range={}",
                        derivative_row.len(),
                        beta.len(),
                        coefficient_range.len(),
                    ));
                }
                if coefficient_range.end > slices.total {
                    return Err(format!(
                        "survival marginal-slope FLEX family design direction range {:?} exceeds flattened width {}",
                        coefficient_range, slices.total,
                    ));
                }
                if derivative_row.iter().any(|value| !value.is_finite()) {
                    return Err(
                        "survival marginal-slope FLEX family design derivative row must be finite"
                            .to_string(),
                    );
                }
            }
            None => {}
        }

        let mut expected = vec![
            ("time", slices.time.clone()),
            ("marginal", slices.marginal.clone()),
            ("logslope", slices.logslope.clone()),
        ];
        if let Some(range) = slices.score_warp.as_ref() {
            expected.push(("score warp", range.clone()));
        }
        if let Some(range) = slices.link_dev.as_ref() {
            expected.push(("link deviation", range.clone()));
        }
        if let Some(range) = slices.influence.as_ref() {
            expected.push(("influence", range.clone()));
        }
        if block_states.len() != expected.len() {
            return Err(format!(
                "survival marginal-slope FLEX family coefficient map expects {} block states, got {}",
                expected.len(),
                block_states.len(),
            ));
        }
        for (block, (name, range)) in block_states.iter().zip(expected.iter()).enumerate() {
            if block.beta.len() != range.len() {
                return Err(format!(
                    "survival marginal-slope FLEX family {name} block {block} beta width {} != layout width {}",
                    block.beta.len(),
                    range.len(),
                ));
            }
        }
        for (block, name) in [(0usize, "time"), (1, "marginal"), (2, "logslope")] {
            if block_states[block].eta.len() <= row {
                return Err(format!(
                    "survival marginal-slope FLEX family {name} block eta length {} does not contain row {row}",
                    block_states[block].eta.len(),
                ));
            }
        }
        Ok(())
    }

    fn build_flex_family_coefficient_jets<J: FlexCoefficientJet>(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary: &FlexPrimarySlices,
        slices: &BlockSlices,
        first: FlexFamilyRowDirection,
        second: FlexFamilyRowDirection,
        coefficient_direction: Option<&FlexCoefficientRowDirection<'_>>,
    ) -> Result<FlexFamilyCoefficientJets<J>, String> {
        let dimension = slices.total;
        let design_targets_marginal = matches!(
            coefficient_direction,
            Some(FlexCoefficientRowDirection::Design { block: 1, .. })
        );
        let design_targets_logslope = matches!(
            coefficient_direction,
            Some(FlexCoefficientRowDirection::Design { block: 2, .. })
        );
        let q_values = self.row_dynamic_q_values(row, block_states)?;
        let time_entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|error| format!("FLEX family design_entry row: {error}"))?;
        let time_exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|error| format!("FLEX family design_exit row: {error}"))?;
        let time_derivative_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|error| format!("FLEX family design_derivative_exit row: {error}"))?;
        let marginal_chunk = self
            .marginal_design
            .try_row_chunk(row..row + 1)
            .map_err(|error| format!("FLEX family marginal_design row: {error}"))?;
        let logslope_chunk = self
            .logslope_layout
            .coefficient_design()
            .try_row_chunk(row..row + 1)
            .map_err(|error| format!("FLEX family logslope design row: {error}"))?;
        let entry_row = time_entry_chunk.row(0);
        let exit_row = time_exit_chunk.row(0);
        let derivative_row = time_derivative_chunk.row(0);
        let marginal_row = marginal_chunk.row(0);
        let logslope_row = logslope_chunk.row(0);

        let mut q0_gradient = vec![0.0; dimension];
        let mut q1_gradient = vec![0.0; dimension];
        let mut qd1_gradient = vec![0.0; dimension];
        let (q0, q1, qd1) = if self.flex_timewiggle_active() {
            let time_tail = self.time_wiggle_range();
            let base_width = time_tail.start;
            let base_range = slices.time.start..slices.time.start + base_width;
            add_coefficient_row(
                &mut q0_gradient,
                &base_range,
                entry_row.slice(s![..base_width]),
                "timewiggle entry base",
            )?;
            add_coefficient_row(
                &mut q1_gradient,
                &base_range,
                exit_row.slice(s![..base_width]),
                "timewiggle exit base",
            )?;
            add_coefficient_row(
                &mut qd1_gradient,
                &base_range,
                derivative_row.slice(s![..base_width]),
                "timewiggle derivative base",
            )?;
            add_coefficient_row(
                &mut q0_gradient,
                &slices.marginal,
                marginal_row,
                "timewiggle entry marginal",
            )?;
            add_coefficient_row(
                &mut q1_gradient,
                &slices.marginal,
                marginal_chunk.row(0),
                "timewiggle exit marginal",
            )?;

            let beta_time = &block_states[0].beta;
            let beta_time_base = beta_time.slice(s![..base_width]);
            let beta_time_wiggle = beta_time.slice(s![time_tail.clone()]);
            let h0 = entry_row.slice(s![..base_width]).dot(&beta_time_base)
                + self.offset_entry[row]
                + block_states[1].eta[row];
            let h1 = exit_row.slice(s![..base_width]).dot(&beta_time_base)
                + self.offset_exit[row]
                + block_states[1].eta[row];
            let d_raw = derivative_row.slice(s![..base_width]).dot(&beta_time_base)
                + self.derivative_offset_exit[row];
            let h0_jet = lifted_coefficient_affine::<J>(
                h0,
                q0_gradient,
                coefficient_direction,
                design_targets_marginal,
                first.entry,
                second.entry,
            );
            let h1_jet = lifted_coefficient_affine::<J>(
                h1,
                q1_gradient,
                coefficient_direction,
                design_targets_marginal,
                first.exit,
                second.exit,
            );
            let d_raw_jet = lifted_coefficient_affine::<J>(
                d_raw,
                qd1_gradient,
                coefficient_direction,
                false,
                first.derivative_exit,
                second.derivative_exit,
            );
            let beta_wiggle_jets: Vec<Dual2<J>> = time_tail
                .clone()
                .map(|local_axis| {
                    lifted_coefficient_affine::<J>(
                        beta_time[local_axis],
                        one_hot_coefficient_gradient(dimension, slices.time.start + local_axis),
                        coefficient_direction,
                        false,
                        0.0,
                        0.0,
                    )
                })
                .collect();
            let (entry_geometry, entry_basis_d5) = self
                .time_wiggle_geometry_with_basis_d5(
                    Array1::from_vec(vec![h0]).view(),
                    beta_time_wiggle,
                )?
                .ok_or_else(|| {
                    "FLEX family timewiggle entry geometry is unavailable".to_string()
                })?;
            let (exit_geometry, exit_basis_d5) = self
                .time_wiggle_geometry_with_basis_d5(
                    Array1::from_vec(vec![h1]).view(),
                    beta_time.slice(s![time_tail]),
                )?
                .ok_or_else(|| "FLEX family timewiggle exit geometry is unavailable".to_string())?;
            let entry_basis =
                TimewiggleBasisDerivativeRows::from_geometry(&entry_geometry, &entry_basis_d5, 0);
            let exit_basis =
                TimewiggleBasisDerivativeRows::from_geometry(&exit_geometry, &exit_basis_d5, 0);
            let q = timewiggle_q_from_basis_derivative_rows(
                &h0_jet,
                &h1_jet,
                &d_raw_jet,
                &beta_wiggle_jets,
                &entry_basis,
                &exit_basis,
                TimewiggleQBaseValues {
                    q0: q_values.q0,
                    q1: q_values.q1,
                    dq1_dh1: exit_geometry.dq_dq0[0],
                },
            )?;
            if q.q0.value().to_bits() != q_values.q0.to_bits()
                || q.q1.value().to_bits() != q_values.q1.to_bits()
                || q.qd1.value().to_bits() != q_values.qd1.to_bits()
            {
                return Err(
                    "FLEX family generic timewiggle q values drifted from the current f64 row"
                        .to_string(),
                );
            }
            (q.q0, q.q1, q.qd1)
        } else {
            add_coefficient_row(&mut q0_gradient, &slices.time, entry_row, "entry time")?;
            add_coefficient_row(&mut q1_gradient, &slices.time, exit_row, "exit time")?;
            add_coefficient_row(
                &mut qd1_gradient,
                &slices.time,
                derivative_row,
                "derivative time",
            )?;
            add_coefficient_row(
                &mut q0_gradient,
                &slices.marginal,
                marginal_row,
                "entry marginal",
            )?;
            add_coefficient_row(
                &mut q1_gradient,
                &slices.marginal,
                marginal_chunk.row(0),
                "exit marginal",
            )?;
            (
                lifted_coefficient_affine::<J>(
                    q_values.q0,
                    q0_gradient,
                    coefficient_direction,
                    design_targets_marginal,
                    first.entry,
                    second.entry,
                ),
                lifted_coefficient_affine::<J>(
                    q_values.q1,
                    q1_gradient,
                    coefficient_direction,
                    design_targets_marginal,
                    first.exit,
                    second.exit,
                ),
                lifted_coefficient_affine::<J>(
                    q_values.qd1,
                    qd1_gradient,
                    coefficient_direction,
                    false,
                    first.derivative_exit,
                    second.derivative_exit,
                ),
            )
        };

        let mut g_gradient = vec![0.0; dimension];
        add_coefficient_row(&mut g_gradient, &slices.logslope, logslope_row, "logslope")?;
        let g = lifted_coefficient_affine::<J>(
            block_states[2].eta[row],
            g_gradient,
            coefficient_direction,
            design_targets_logslope,
            0.0,
            0.0,
        );
        let zero =
            || lifted_coefficient_affine::<J>(0.0, vec![0.0; dimension], None, false, 0.0, 0.0);
        let template = zero();
        let mut du = vec![template.clone(); primary.total];
        du[primary.q0] = tangent_jet(&q0);
        du[primary.q1] = tangent_jet(&q1);
        du[primary.qd1] = tangent_jet(&qd1);
        du[primary.g] = tangent_jet(&g);

        if let (Some(primary_range), Some(coefficient_range), Some(beta_h)) = (
            primary.h.as_ref(),
            slices.score_warp.as_ref(),
            self.flex_score_beta(block_states)?,
        ) {
            if primary_range.len() != coefficient_range.len() {
                return Err("FLEX family score-warp primary/coefficient widths differ".to_string());
            }
            for local_axis in 0..primary_range.len() {
                let coefficient_axis = coefficient_range.start + local_axis;
                let coefficient = lifted_coefficient_affine::<J>(
                    beta_h[local_axis],
                    one_hot_coefficient_gradient(dimension, coefficient_axis),
                    coefficient_direction,
                    false,
                    0.0,
                    0.0,
                );
                du[primary_range.start + local_axis] = tangent_jet(&coefficient);
            }
        }
        if let (Some(primary_range), Some(coefficient_range), Some(beta_w)) = (
            primary.w.as_ref(),
            slices.link_dev.as_ref(),
            self.flex_link_beta(block_states)?,
        ) {
            if primary_range.len() != coefficient_range.len() {
                return Err(
                    "FLEX family link-deviation primary/coefficient widths differ".to_string(),
                );
            }
            for local_axis in 0..primary_range.len() {
                let coefficient_axis = coefficient_range.start + local_axis;
                let coefficient = lifted_coefficient_affine::<J>(
                    beta_w[local_axis],
                    one_hot_coefficient_gradient(dimension, coefficient_axis),
                    coefficient_direction,
                    false,
                    0.0,
                    0.0,
                );
                du[primary_range.start + local_axis] = tangent_jet(&coefficient);
            }
        }

        let o_infl = self.influence_index_offset(row, block_states)?;
        if let (Some(primary_axis), Some(coefficient_range), Some(influence)) = (
            primary.infl,
            slices.influence.as_ref(),
            self.influence_absorber.as_ref(),
        ) {
            let mut influence_gradient = vec![0.0; dimension];
            add_coefficient_row(
                &mut influence_gradient,
                coefficient_range,
                influence.row(row),
                "influence",
            )?;
            let influence_jet = lifted_coefficient_affine::<J>(
                o_infl,
                influence_gradient,
                coefficient_direction,
                false,
                0.0,
                0.0,
            );
            du[primary_axis] = tangent_jet(&influence_jet);
        }

        let probit_scale = self.probit_frailty_scale();
        if !probit_scale.is_finite() || probit_scale <= 0.0 {
            return Err(format!(
                "survival marginal-slope FLEX family probit scale must be finite and positive, got {probit_scale}"
            ));
        }
        let scale_ratio = Dual2 {
            v: J::constant(1.0, dimension),
            g: J::constant(first.probit_scale / probit_scale, dimension),
            h: J::constant(second.probit_scale / probit_scale, dimension),
        };
        Ok(FlexFamilyCoefficientJets {
            template,
            q0,
            q1,
            qd1,
            g,
            scale_ratio,
            du,
            o_infl,
        })
    }

    fn flex_family_direction_row_terms_generic<J: FlexCoefficientJet>(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        first: FlexFamilyRowDirection,
        second: FlexFamilyRowDirection,
        coefficient_direction: Option<&FlexCoefficientRowDirection<'_>>,
    ) -> Result<FlexFamilyDirectionRowTerms, String> {
        self.ensure_scalar_flex_exact_score_geometry("FLEX family-direction row program")?;
        let expected_blocks = 3
            + usize::from(self.score_warp.is_some())
            + usize::from(self.link_dev.is_some())
            + usize::from(self.influence_absorber.is_some());
        if block_states.len() != expected_blocks {
            return Err(format!(
                "survival marginal-slope FLEX family coefficient map expects {expected_blocks} block states, got {}",
                block_states.len(),
            ));
        }
        let primary = flex_primary_slices(self);
        let slices = block_slices(self, block_states);
        self.validate_flex_family_coefficient_state(
            row,
            block_states,
            &slices,
            first,
            second,
            coefficient_direction,
        )?;
        let jets = self.build_flex_family_coefficient_jets::<J>(
            row,
            block_states,
            &primary,
            &slices,
            first,
            second,
            coefficient_direction,
        )?;
        if survival_derivative_guard_violated(jets.qd1.value(), self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope monotonicity violated at row {row}: qd1={:.3e} < guard={:.3e}",
                    jets.qd1.value(),
                    self.derivative_guard,
                ),
            }
            .into());
        }

        let g_value = jets.g.value();
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        let (a0, _) = self.solve_row_survival_intercept_with_slot(
            jets.q0.value(),
            g_value,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, _) = self.solve_row_survival_intercept_with_slot(
            jets.q1.value(),
            g_value,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;
        let entry_cached = self.build_cached_partition(&primary, a0, g_value, beta_h, beta_w)?;
        let exit_cached = self.build_cached_partition(&primary, a1, g_value, beta_h, beta_w)?;
        let evaluate_timepoint =
            |q: &Dual2<J>, a: f64, cached: &CachedPartitionCells| -> Result<_, String> {
                let d_check = self.evaluate_survival_denom_d(a, g_value, beta_h, beta_w)?;
                let (obs_coeff, obs_fixed) =
                    observed_fixed_for(self, &primary, row, a, g_value, beta_h, beta_w)?;
                let cells = cells_from_cached(cached);
                flex_timepoint_inputs_generic(
                    &jets.template,
                    &jets.g,
                    &jets.du,
                    a,
                    d_check,
                    primary.g,
                    primary.infl,
                    q,
                    &jets.scale_ratio,
                    self.observed_score_projection(row),
                    jets.o_infl,
                    obs_coeff,
                    &obs_fixed,
                    &cells,
                )
            };
        let (eta0, _, _) = evaluate_timepoint(&jets.q0, a0, &entry_cached)?;
        let (eta1, chi1, d1) = evaluate_timepoint(&jets.q1, a1, &exit_cached)?;
        if !chi1.value().is_finite() || chi1.value() <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive observed chi1={:.3e}",
                    chi1.value(),
                ),
            }
            .into());
        }
        let output = flex_row_nll(
            &eta0,
            &eta1,
            &chi1,
            &d1,
            &jets.q1,
            &jets.qd1,
            surv_stack(eta0.value())?,
            surv_stack(eta1.value())?,
            self.weights[row],
            self.event[row],
        );
        Ok(FlexFamilyDirectionRowTerms {
            first: output.g.owned_base_terms(),
            second: output.h.owned_base_terms(),
            directional: output.g.owned_directional_terms(),
        })
    }

    /// Evaluate one complete FLEX row under one arbitrary family direction.
    ///
    /// The inner width is the canonical flattened coefficient layout
    /// `time | marginal | logslope | score-warp? | link-dev? | influence?`.
    /// Consequently `first.gradient` is the complete family-by-coefficient
    /// mixed-partial row in one evaluation.  This is distinct from a
    /// family-by-design-hyper partial: the latter must use
    /// [`Self::flex_family_design_direction_row_terms`] so `X_psi beta` and
    /// the `X_psi` pullback drift are both represented.
    /// The outer [`Dual2`] owns the supplied family first/second motion. With no
    /// beta direction the row runs as `Dual2<Jet2>`; with one it runs as
    /// `Dual2<Jet3>` and returns the exact directional drift of the first family
    /// value/score/Hessian channel. Baseline offsets, nonlinear time wiggle,
    /// learned probit scale, calibration, moving moments, and the row likelihood
    /// are all evaluated by this one program.
    pub(crate) fn flex_family_direction_row_terms(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        first: FlexFamilyRowDirection,
        second: FlexFamilyRowDirection,
        beta_direction: Option<&Array1<f64>>,
    ) -> Result<FlexFamilyDirectionRowTerms, String> {
        if let Some(beta_direction) = beta_direction {
            let coefficient_direction = FlexCoefficientRowDirection::Beta(beta_direction);
            self.flex_family_direction_row_terms_generic::<Jet3>(
                row,
                block_states,
                first,
                second,
                Some(&coefficient_direction),
            )
        } else {
            self.flex_family_direction_row_terms_generic::<Jet2>(
                row,
                block_states,
                first,
                second,
                None,
            )
        }
    }

    /// Evaluate the derivative of the complete family row channel with respect
    /// to one marginal/logslope design coordinate at fixed coefficients.
    ///
    /// `derivative_row` is one row of `X_psi`.  Its Jet3 epsilon part is seeded
    /// with both `X_psi beta` and the coefficient gradient `X_psi`; therefore
    /// `directional` includes `X_psi^T score` and both Hessian pullback-map
    /// drift terms automatically, including their nonlinear time-wiggle
    /// composition.
    pub(crate) fn flex_family_design_direction_row_terms(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        first: FlexFamilyRowDirection,
        second: FlexFamilyRowDirection,
        block: usize,
        derivative_row: &Array1<f64>,
    ) -> Result<FlexFamilyDirectionRowTerms, String> {
        let slices = block_slices(self, block_states);
        let (beta, coefficient_range) = match block {
            1 => (&block_states[1].beta, slices.marginal),
            2 => (&block_states[2].beta, slices.logslope),
            _ => {
                return Err(format!(
                    "survival marginal-slope FLEX family design direction supports marginal/logslope blocks 1 or 2, got block {block}"
                ));
            }
        };
        let coefficient_direction = FlexCoefficientRowDirection::Design {
            block,
            derivative_row,
            beta,
            coefficient_range,
        };
        self.flex_family_direction_row_terms_generic::<Jet3>(
            row,
            block_states,
            first,
            second,
            Some(&coefficient_direction),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// #932 nested-dual ORACLE instantiation of the single-source flex geometry.
//
// `gam_math::nested_dual::Dual22 = Dual2<Dual2<f64>>` carries a truncation-FREE
// forward-over-forward derivative in TWO independent scalar directions `(s, t)`,
// each to 2nd order (2 + 2 = 4th-order bidirectional). Instantiating the SAME
// `flex_timepoint_inputs_generic` geometry over it — with every primary seeded
// as `base_i + s·d1_i + t·d2_i` — makes the `∂²_s ∂²_t` channel (`channels()[8]`)
// equal to the full arbitrary-weight bidirectional contraction
//   Σ_abcd ℓ_abcd · d1_a d1_b d2_c d2_d
// via a DIFFERENT composition ordering than the production p-primary `Jet4`.
// This is the truncation-free replacement for the flex jet4 bidirectional's
// scalar-FD sanity gate (the last FD-limited seam in the #932 tower). It is an
// ORACLE only — never used on the production sweep — so it lives beside, not
// inside, the p-primary jet types.
use gam_math::nested_dual::{Dual2, Dual22, JetField};

#[cfg(test)]
mod moment_engine_tests {
    use super::*;
    use crate::cubic_cell_kernel::{DenestedCubicCell, reduce_sextic_moments};
    use crate::marginal_slope_shared::eval_coeff4_at;
    use gam_math::jet_scalar::{Order2, filtered_implicit_solve_scalar};
    use gam_math::jet_tower::Tower2;
    use std::hint::black_box;
    use std::time::Instant;

    #[test]
    fn dual2_flexjet_scales_runtime_channels_by_total_homogeneous_order() {
        let factors = [2.0, 3.0, 5.0, 7.0, 11.0];
        let original = Dual2 {
            v: Jet2::from_parts(1.0, &[2.0, 3.0], &[4.0, 5.0, 6.0, 7.0]),
            g: Jet2::from_parts(8.0, &[9.0, 10.0], &[11.0, 12.0, 13.0, 14.0]),
            h: Jet2::from_parts(15.0, &[16.0, 17.0], &[18.0, 19.0, 20.0, 21.0]),
        };
        let scaled = original.scale_homogeneous_orders(factors);
        let assert_part = |actual: &Jet2, expected: &Jet2, outer_order: usize| {
            assert_eq!(actual.v, factors[outer_order] * expected.v);
            for axis in 0..expected.g.len() {
                assert_eq!(actual.g[axis], factors[outer_order + 1] * expected.g[axis]);
            }
            for axis in 0..expected.h.len() {
                assert_eq!(actual.h[axis], factors[outer_order + 2] * expected.h[axis]);
            }
        };
        assert_eq!(<Dual2<Jet2> as FlexJet>::ORDER, 4);
        assert_part(&scaled.v, &original.v, 0);
        assert_part(&scaled.g, &original.g, 1);
        assert_part(&scaled.h, &original.h, 2);
    }

    #[test]
    fn recursive_dual22_flexjet_scaling_matches_nested_total_order() {
        let factors = [2.0, 3.0, 5.0, 7.0, 11.0];
        let original_channels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let scaled = Dual22::from_channels(original_channels)
            .scale_homogeneous_orders(factors)
            .channels();
        // `(outer order, inner order)` in `Dual22::channels()` order.
        let total_orders = [0usize, 1, 1, 2, 2, 2, 3, 3, 4];
        for channel in 0..scaled.len() {
            assert_eq!(
                scaled[channel],
                factors[total_orders[channel]] * original_channels[channel]
            );
        }
    }

    #[test]
    fn generic_q_jet_carries_exact_beta_family_calibration_channels() {
        let q = 0.37;
        let zero = Jet2::from_parts(0.0, &[0.0], &[]);
        let template = Dual2 {
            v: zero.clone(),
            g: zero.clone(),
            h: zero.clone(),
        };
        // q = q0 + beta + theta: independent unit inner-beta and outer-family
        // directions, with no curvature in the input chart itself.
        let q_jet = Dual2 {
            v: Jet2::primary(q, 0, 1),
            g: Jet2::from_parts(1.0, &[0.0], &[]),
            h: zero,
        };
        let scale_ratio = const_jet_like(&template, 1.0);
        let residual =
            calibration_residual_jet(&template, &template, 0, &[], &q_jet, &scale_ratio, &[]);
        let phi = crate::probability::normal_pdf(q);
        let expected = [
            -phi,
            q * phi,
            (1.0 - q * q) * phi,
            (q * q * q - 3.0 * q) * phi,
        ];
        let check = |actual: f64, wanted: f64, channel: &str| {
            let tolerance = 256.0 * f64::EPSILON * (1.0 + actual.abs().max(wanted.abs()));
            assert!(
                (actual - wanted).abs() <= tolerance,
                "{channel}: actual={actual:.17e}, wanted={wanted:.17e}, tolerance={tolerance:.3e}"
            );
        };
        check(residual.v.v, 0.0, "value");
        check(residual.v.g[0], expected[0], "beta first");
        check(residual.v.h[0], expected[1], "beta second");
        check(residual.g.v, expected[0], "family first");
        check(residual.g.g[0], expected[1], "family-beta");
        check(residual.g.h[0], expected[2], "family-beta-beta");
        check(residual.h.v, expected[1], "family second");
        check(residual.h.g[0], expected[2], "family-family-beta");
        check(residual.h.h[0], expected[3], "family-family-beta-beta");
    }

    /// Test-only execution policy that runs the historical order-four moment
    /// construction over a lower-order algebra. The wrapped arithmetic is
    /// unchanged, so this is an exact pre-optimization timing baseline rather
    /// than a separately re-derived moment formula.
    #[derive(Clone)]
    struct ForcedOrder4<J>(J);

    impl<J: FlexJet> JetField for ForcedOrder4<J> {
        #[inline(always)]
        fn value(&self) -> f64 {
            self.0.value()
        }

        #[inline(always)]
        fn add(&self, other: &Self) -> Self {
            Self(self.0.add(&other.0))
        }

        #[inline(always)]
        fn sub(&self, other: &Self) -> Self {
            Self(self.0.sub(&other.0))
        }

        #[inline(always)]
        fn mul(&self, other: &Self) -> Self {
            Self(self.0.mul(&other.0))
        }

        #[inline(always)]
        fn neg(&self) -> Self {
            Self(self.0.neg())
        }

        #[inline(always)]
        fn scale(&self, scale: f64) -> Self {
            Self(self.0.scale(scale))
        }

        #[inline(always)]
        fn compose_unary(&self, derivatives: [f64; 5]) -> Self {
            Self(self.0.compose_unary(derivatives))
        }
    }

    impl<J: FlexJet> FlexJet for ForcedOrder4<J> {
        const ORDER: usize = 4;

        fn scale_homogeneous_orders(&self, factors: [f64; 5]) -> Self {
            Self(self.0.scale_homogeneous_orders(factors))
        }
    }

    // ── §B moment engine: the de-nested cell moments over a FlexJet ─────────────
    //
    // #932 Item 2 (doc §D). The per-cell moments `M_n = ∫_{z_L}^{z_R} z^n e^{−q(z)} dz`
    // (sextic `q`, no closed antiderivative) satisfy the SAME raising recurrence the
    // numeric `cubic_cell_kernel::reduce_sextic_moments` uses —
    //   `M_{n+5} = (n·M_{n−1} − Σ_{j=0}^{4} d[j]·M_{n+j} − b_n) / d[5]`,
    // with `d = q'(z)` coefficients (`sextic_qprime_coefficients`) and boundary term
    // `b_n = z_R^n e^{−q(z_R)} − z_L^n e^{−q(z_L)}` — so it ports to ANY `FlexJet`
    // scalar verbatim. Carrying the cell coefficients `c0..c3` and the (moving) edges
    // `z_L,z_R` as jets propagates the moments' θ-derivatives mechanically: the
    // `Σ d[j]·M_{n+j}` term is the interior coefficient sensitivity and the boundary
    // term `b_n` is exactly the §D moving-boundary flux (its edge-jet derivatives are
    // the Leibniz `[z^n e^{−q}·z_edge']` contributions the hand `directional` path
    // assembles by hand). The base moments `M_0..M_4` (the normalization integrals)
    // arrive as jets from the cell evaluator — those carry the only transcendental
    // (erf/series) content; the algebra owns the rest.

    /// `q'(z)` coefficient jets `[d0..d5]` for `q = ½(z² + η²)`, `η = c0+c1 z+c2 z²+
    /// c3 z³`, over `FlexJet` cell-coefficient jets — the jet image of
    /// [`crate::cubic_cell_kernel::sextic_qprime_coefficients`].
    fn qprime_coeffs_jet<J: FlexJet>(c: &[J; 4]) -> [J; 6] {
        let (c0, c1, c2, c3) = (&c[0], &c[1], &c[2], &c[3]);
        // d0 = c0·c1
        let d0 = c0.mul(c1);
        // d1 = 1 + c1² + 2·c0·c2   (the leading `+z` of q' supplies the constant 1)
        let d1 = add_const(&c1.mul(c1).add(&c0.mul(c2).scale(2.0)), 1.0);
        // d2 = 3·c0·c3 + 3·c1·c2
        let d2 = c0.mul(c3).add(&c1.mul(c2)).scale(3.0);
        // d3 = 4·c1·c3 + 2·c2²
        let d3 = c1.mul(c3).scale(4.0).add(&c2.mul(c2).scale(2.0));
        // d4 = 5·c2·c3
        let d4 = c2.mul(c3).scale(5.0);
        // d5 = 3·c3²
        let d5 = c3.mul(c3).scale(3.0);
        [d0, d1, d2, d3, d4, d5]
    }

    /// `q(z) = ½(z² + η(z)²)` evaluated at an edge jet `z`, with `η` from the cell
    /// coefficient jets — the exponent whose `e^{−q}` is the boundary weight.
    fn cell_q_at_jet<J: FlexJet>(c: &[J; 4], z: &J) -> J {
        // η = c0 + c1 z + c2 z² + c3 z³  (Horner)
        let eta = c[3].mul(z).add(&c[2]).mul(z).add(&c[1]).mul(z).add(&c[0]);
        // ½(z² + η²)
        z.mul(z).add(&eta.mul(&eta)).scale(0.5)
    }

    /// One boundary term `z^n·e^{−q(z)}` at a (possibly infinite) moving edge jet.
    /// An infinite edge contributes nothing (matching the numeric
    /// `moment_boundary_term_with_powers` short-circuit).
    fn boundary_edge_term_jet<J: FlexJet>(
        c: &[J; 4],
        z: &J,
        z_pow_n: &J,
        finite: bool,
    ) -> Option<J> {
        if !finite {
            return None;
        }
        let q = cell_q_at_jet(c, z);
        let w = exp_jet(&q.scale(-1.0));
        Some(z_pow_n.mul(&w))
    }

    /// The sextic moment recurrence over a `FlexJet`: given the cell coefficient
    /// jets `c`, the moving edge jets `(z_left, z_right)` with their finiteness, and
    /// the base moment jets `M_0..M_4`, return `M_0..M_max` as jets. Bit-faithful to
    /// `reduce_sextic_moments` term for term, but every operation in the `FlexJet`
    /// algebra so the moments carry their exact θ-derivatives.
    fn cell_moment_recurrence_jet<J: FlexJet>(
        c: &[J; 4],
        z_left: &J,
        left_finite: bool,
        z_right: &J,
        right_finite: bool,
        base_m0_m4: &[J; 5],
        max_degree: usize,
    ) -> Vec<J> {
        let d = qprime_coeffs_jet(c);
        let inv_lead = recip(&d[5]);
        let mut moments: Vec<J> = base_m0_m4.iter().cloned().collect();
        if max_degree < 5 {
            moments.truncate(max_degree + 1);
            return moments;
        }
        // Rolling z^n at each edge (jets), starting at n = 0 (z^0 = 1 = z/z).
        let one_l = recip(z_left).mul(z_left);
        let one_r = recip(z_right).mul(z_right);
        let mut left_pow = one_l;
        let mut right_pow = one_r;
        for n in 0..=(max_degree - 5) {
            let b_left = boundary_edge_term_jet(c, z_left, &left_pow, left_finite);
            let b_right = boundary_edge_term_jet(c, z_right, &right_pow, right_finite);
            // b_n = right − left, missing edges contribute zero.
            let mut b_n = match (b_right, b_left) {
                (Some(r), Some(l)) => r.sub(&l),
                (Some(r), None) => r,
                (None, Some(l)) => l.scale(-1.0),
                (None, None) => moments[0].scale(0.0),
            };
            // numer = n·M_{n−1} − Σ_{j=0}^{4} d[j]·M_{n+j} − b_n
            let mut numer = if n == 0 {
                moments[0].scale(0.0)
            } else {
                moments[n - 1].scale(n as f64)
            };
            for j in 0..=4 {
                numer = numer.sub(&d[j].mul(&moments[n + j]));
            }
            numer = numer.sub(&b_n);
            moments.push(numer.mul(&inv_lead));
            // Roll powers: z^{n+1} = z^n · z.
            left_pow = if left_finite {
                left_pow.mul(z_left)
            } else {
                b_n.scale(0.0)
            };
            right_pow = if right_finite {
                right_pow.mul(z_right)
            } else {
                // reuse b_n as a zero-jet scratch source of the right `p`
                b_n = b_n.scale(0.0);
                b_n
            };
        }
        moments
    }

    // ── §C: observed cell-coefficient jets + eta/chi point-eval (Phase C core) ──
    //
    // The observed cell coefficients `coeff[k]` are a smooth function of the
    // intercept `a(θ)` and the slope `b` (= the `g` primary), with the score-warp
    // (`h`) and link-dev (`w`) channels entering linearly on top. Their full
    // bivariate Taylor in `(a,b)` is exactly the `observed_denested_cell_partials`
    // pack (`dc_da…dc_dbbb`). Composing that Taylor with the intercept jet `a_jet`
    // and the slope jet `b_jet` (both carrying their θ-derivatives) yields each
    // `coeff[k]` AS a jet — so `eta = Σ_k coeff[k]·z_obs^k` and `chi = Σ_k
    // dc_da[k]·z_obs^k` (point-evals at the fixed observation `z_obs`) carry their
    // exact θ-derivatives mechanically, replacing the hand `eta_u = chi·a_u + rho`
    // / `eta_uv = …` chain in `first_full`/`directional`/`bidirectional`.

    /// One observed cell coefficient `coeff[k]` as a jet: the bivariate `(a,b)`
    /// Taylor (up to 3rd order, matching the `dc_d{a,b}…` pack) composed with the
    /// intercept tangent `da` and slope tangent `db` jets. Terms with a 0/6/2/… are
    /// the multinomial Taylor weights `coeff + Σ (1/(i!j!)) ∂^{i+j}coeff/∂a^i∂b^j ·
    /// da^i db^j`.
    fn observed_coeff_component_jet<J: FlexJet>(
        template: &J,
        k: usize,
        coeff: [f64; 4],
        dc_da: [f64; 4],
        dc_db: [f64; 4],
        dc_daa: [f64; 4],
        dc_dab: [f64; 4],
        dc_dbb: [f64; 4],
        dc_daaa: [f64; 4],
        dc_daab: [f64; 4],
        dc_dabb: [f64; 4],
        dc_dbbb: [f64; 4],
        da: &J,
        db: &J,
    ) -> J {
        let dada = da.mul(da);
        let dadb = da.mul(db);
        let dbdb = db.mul(db);
        let mut c = const_jet_like(template, coeff[k]);
        c = c.add(&da.scale(dc_da[k])).add(&db.scale(dc_db[k]));
        c = c
            .add(&dada.scale(0.5 * dc_daa[k]))
            .add(&dadb.scale(dc_dab[k]))
            .add(&dbdb.scale(0.5 * dc_dbb[k]));
        let inv6 = 1.0 / 6.0;
        let half = 0.5;
        c = c
            .add(&dada.mul(da).scale(inv6 * dc_daaa[k]))
            .add(&dada.mul(db).scale(half * dc_daab[k]))
            .add(&dadb.mul(db).scale(half * dc_dabb[k]))
            .add(&dbdb.mul(db).scale(inv6 * dc_dbbb[k]));
        c
    }

    /// The observed cell-coefficient partial pack (`coeff`/`dc_d{a,b}…/dbbb`) passed
    /// through `observed_denested_cell_partials`, bundled so the generic eta/chi
    /// builder stays under the argument-count gate.
    struct ObservedCoeffPack {
        coeff: [f64; 4],
        dc_da: [f64; 4],
        dc_db: [f64; 4],
        dc_daa: [f64; 4],
        dc_dab: [f64; 4],
        dc_dbb: [f64; 4],
        dc_daaa: [f64; 4],
        dc_daab: [f64; 4],
        dc_dabb: [f64; 4],
        dc_dbbb: [f64; 4],
    }

    /// Phase C-complete (generic order): the observed timepoint `eta` and `chi` as
    /// jets at ANY `FlexJet` order, from the intercept jet `a_jet` (carrying its
    /// θ-derivatives to that order) and the slope jet `b_jet`, the observed
    /// cell-coefficient pack, and pre-built score-warp(`h`)/link-dev(`w`) `rho`/`tau`
    /// channel jets. `eta`/`chi` carry their exact θ-derivatives by composing the
    /// coefficients' bivariate `(a,b)` Taylor with the intercept/slope jets, then
    /// adding the linear `h`/`w`/`infl` channels — replacing the hand
    /// `eta_u = chi·a_u + rho`, `eta_uv = …` chains in
    /// `first_full`/`directional`/`bidirectional`.
    ///
    /// `rho_jet`/`tau_jet` are the already-seeded jets carrying the linear `h`/`w`/
    /// `infl` channels' θ-dependence on their own primaries (the caller builds them
    /// at the correct order with the right directional seeds — order-specific
    /// seeding context lives at the call site, not here). `eta += rho_jet`,
    /// `chi += tau_jet`.
    fn flex_timepoint_eta_chi<J: FlexJet>(
        a_jet: &J,
        b_jet: &J,
        z_obs: f64,
        o_infl: f64,
        pack: &ObservedCoeffPack,
        rho_jet: &J,
        tau_jet: &J,
    ) -> (J, J) {
        let da = tangent_jet(a_jet);
        let db = tangent_jet(b_jet);
        let zero4 = [0.0_f64; 4];

        // eta coefficients: the coeff pack composed with (da, db).
        let coeff_jets: [J; 4] = std::array::from_fn(|k| {
            observed_coeff_component_jet(
                a_jet,
                k,
                pack.coeff,
                pack.dc_da,
                pack.dc_db,
                pack.dc_daa,
                pack.dc_dab,
                pack.dc_dbb,
                pack.dc_daaa,
                pack.dc_daab,
                pack.dc_dabb,
                pack.dc_dbbb,
                &da,
                &db,
            )
        });
        let eta = add_const(&eval_coeff_jet_at(&coeff_jets, z_obs), o_infl).add(rho_jet);

        // chi = ∂eta/∂a coefficients = the dc_da pack, whose own (a,b)-Taylor is the
        // once-`a`-shifted pack (dc_daa as ∂/∂a, dc_dab as ∂/∂b, dc_daaa/daab/dabb as
        // the seconds; the dc_da pack carries no third-order term, so those are 0).
        let chi_jets: [J; 4] = std::array::from_fn(|k| {
            observed_coeff_component_jet(
                a_jet,
                k,
                pack.dc_da,
                pack.dc_daa,
                pack.dc_dab,
                pack.dc_daaa,
                pack.dc_daab,
                pack.dc_dabb,
                zero4,
                zero4,
                zero4,
                zero4,
                &da,
                &db,
            )
        });
        let chi = eval_coeff_jet_at(&chi_jets, z_obs).add(tau_jet);

        (eta, chi)
    }

    /// #932 item-2 increment 1: the FlexJet moment recurrence must reproduce the
    /// numeric `reduce_sextic_moments` on the VALUE channel term-for-term (a
    /// generic non-degenerate sextic cell), proving the port of the raising
    /// recurrence + boundary term to the jet algebra is exact. Independent
    /// finite-difference and nested-dual gates exercise the derivative channels.
    #[test]
    fn cell_moment_recurrence_jet_value_matches_numeric_932() {
        let cell = DenestedCubicCell {
            left: -1.5,
            right: 2.0,
            c0: 0.3,
            c1: -0.4,
            c2: 0.5,
            c3: 0.2,
        };
        let base = [1.0_f64, 0.1, 0.6, -0.05, 0.4];
        let max_degree = 12usize;
        let reference =
            reduce_sextic_moments(cell, base, max_degree).expect("numeric sextic moments");

        let p = 3usize;
        let konst = |x: f64| Jet2::from_parts(x, &vec![0.0; p], &[]);
        let c = [
            konst(cell.c0),
            konst(cell.c1),
            konst(cell.c2),
            konst(cell.c3),
        ];
        let zl = konst(cell.left);
        let zr = konst(cell.right);
        let base_jets = [
            konst(base[0]),
            konst(base[1]),
            konst(base[2]),
            konst(base[3]),
            konst(base[4]),
        ];
        let moments = cell_moment_recurrence_jet(
            &c,
            &zl,
            cell.left.is_finite(),
            &zr,
            cell.right.is_finite(),
            &base_jets,
            max_degree,
        );
        assert_eq!(moments.len(), reference.len(), "moment count");
        for (n, (m, r)) in moments.iter().zip(reference.iter()).enumerate() {
            assert!(
                (m.value() - r).abs() <= 1e-9 * (1.0 + r.abs()),
                "moment {n}: jet value {} != numeric {}",
                m.value(),
                r
            );
        }
    }

    /// The order-aware nilpotent schedule must preserve every represented
    /// channel while eliminating the historical order-four work from Jet1/2.
    /// `ForcedOrder4` delegates to the identical underlying arithmetic and only
    /// overrides the execution order, making it an exact pre-change baseline.
    #[test]
    fn measure_base_moment_instantiated_order_vs_forced_four_932() {
        use crate::cubic_cell_kernel::evaluate_cell_moments;

        let cell = DenestedCubicCell {
            left: -1.2,
            right: 1.7,
            c0: 0.25,
            c1: -0.35,
            c2: 0.4,
            c3: 0.15,
        };
        let numeric = evaluate_cell_moments(cell, 28)
            .expect("numeric cell moments")
            .moments
            .into_vec();
        let p = 6usize;
        let gradient = |scale: f64| -> Vec<f64> {
            (0..p)
                .map(|axis| scale * (axis as f64 + 1.0) / p as f64)
                .collect()
        };
        let hessian = |scale: f64| -> Vec<f64> {
            let mut out = vec![0.0; p * p];
            for i in 0..p {
                for j in 0..p {
                    out[i * p + j] = scale * (i + j + 1) as f64 / (p * p) as f64;
                }
            }
            out
        };

        let c1: [Jet1; 4] = [
            Jet1 {
                v: cell.c0,
                g: gradient(0.13),
            },
            Jet1 {
                v: cell.c1,
                g: gradient(-0.21),
            },
            Jet1 {
                v: cell.c2,
                g: gradient(0.17),
            },
            Jet1 {
                v: cell.c3,
                g: gradient(0.09),
            },
        ];
        let left1 = Jet1 {
            v: cell.left,
            g: gradient(-0.23),
        };
        let right1 = Jet1 {
            v: cell.right,
            g: gradient(0.31),
        };
        let forced_c1 = c1.clone().map(ForcedOrder4);
        let forced_left1 = ForcedOrder4(left1.clone());
        let forced_right1 = ForcedOrder4(right1.clone());

        let c2: [Jet2; 4] = [
            Jet2::from_parts(cell.c0, &gradient(0.13), &hessian(0.017)),
            Jet2::from_parts(cell.c1, &gradient(-0.21), &hessian(-0.011)),
            Jet2::from_parts(cell.c2, &gradient(0.17), &hessian(0.019)),
            Jet2::from_parts(cell.c3, &gradient(0.09), &hessian(-0.007)),
        ];
        let left2 = Jet2::from_parts(cell.left, &gradient(-0.23), &hessian(0.013));
        let right2 = Jet2::from_parts(cell.right, &gradient(0.31), &hessian(-0.015));
        let forced_c2 = c2.clone().map(ForcedOrder4);
        let forced_left2 = ForcedOrder4(left2.clone());
        let forced_right2 = ForcedOrder4(right2.clone());

        let native1 = base_moment_jets(&c1, &left1, true, &right1, true, &numeric);
        let historical1 = base_moment_jets(
            &forced_c1,
            &forced_left1,
            true,
            &forced_right1,
            true,
            &numeric,
        );
        let native2 = base_moment_jets(&c2, &left2, true, &right2, true, &numeric);
        let historical2 = base_moment_jets(
            &forced_c2,
            &forced_left2,
            true,
            &forced_right2,
            true,
            &numeric,
        );
        let close = |got: f64, want: f64, label: &str| {
            let tolerance = 1e-12 * got.abs().max(want.abs()).max(1.0);
            assert!((got - want).abs() <= tolerance, "{label}: {got} != {want}");
        };
        for n in 0..5 {
            close(
                native1[n].v,
                historical1[n].0.v,
                &format!("Jet1 M{n} value"),
            );
            for i in 0..p {
                close(
                    native1[n].g[i],
                    historical1[n].0.g[i],
                    &format!("Jet1 M{n} gradient[{i}]"),
                );
            }
            close(
                native2[n].v,
                historical2[n].0.v,
                &format!("Jet2 M{n} value"),
            );
            for i in 0..p {
                close(
                    native2[n].g[i],
                    historical2[n].0.g[i],
                    &format!("Jet2 M{n} gradient[{i}]"),
                );
                for j in 0..p {
                    close(
                        native2[n].h[i * p + j],
                        historical2[n].0.h[i * p + j],
                        &format!("Jet2 M{n} Hessian[{i},{j}]"),
                    );
                }
            }
        }

        let iterations = if cfg!(debug_assertions) { 4 } else { 5_000 };
        let mut native1_best = f64::INFINITY;
        let mut historical1_best = f64::INFINITY;
        let mut native2_best = f64::INFINITY;
        let mut historical2_best = f64::INFINITY;
        for _ in 0..5 {
            let start = Instant::now();
            for _ in 0..iterations {
                black_box(base_moment_jets(
                    black_box(&c1),
                    black_box(&left1),
                    true,
                    black_box(&right1),
                    true,
                    black_box(&numeric),
                ));
            }
            native1_best = native1_best.min(start.elapsed().as_secs_f64());

            let start = Instant::now();
            for _ in 0..iterations {
                black_box(base_moment_jets(
                    black_box(&forced_c1),
                    black_box(&forced_left1),
                    true,
                    black_box(&forced_right1),
                    true,
                    black_box(&numeric),
                ));
            }
            historical1_best = historical1_best.min(start.elapsed().as_secs_f64());

            let start = Instant::now();
            for _ in 0..iterations {
                black_box(base_moment_jets(
                    black_box(&c2),
                    black_box(&left2),
                    true,
                    black_box(&right2),
                    true,
                    black_box(&numeric),
                ));
            }
            native2_best = native2_best.min(start.elapsed().as_secs_f64());

            let start = Instant::now();
            for _ in 0..iterations {
                black_box(base_moment_jets(
                    black_box(&forced_c2),
                    black_box(&forced_left2),
                    true,
                    black_box(&forced_right2),
                    true,
                    black_box(&numeric),
                ));
            }
            historical2_best = historical2_best.min(start.elapsed().as_secs_f64());
        }
        let to_ns = |seconds: f64| seconds * 1e9 / iterations as f64;
        let native1_ns = to_ns(native1_best);
        let historical1_ns = to_ns(historical1_best);
        let native2_ns = to_ns(native2_best);
        let historical2_ns = to_ns(historical2_best);
        eprintln!(
            "FLEX-MOMENT-ORDER-932 jet1={native1_ns:.2} ns historical-order4-jet1={historical1_ns:.2} ns speedup1={:.3}x jet2={native2_ns:.2} ns historical-order4-jet2={historical2_ns:.2} ns speedup2={:.3}x",
            historical1_ns / native1_ns,
            historical2_ns / native2_ns,
        );
    }

    /// #932 item-2 Phase B-base: the base-moment jet builder `base_moment_jets`
    /// must reproduce the FIRST θ-derivatives of the normalization base moments
    /// `M_0..M_4` (interior `Σ_m S_m M_{n+m}` + moving-edge sliver flux) against a
    /// central finite difference of `evaluate_cell_moments` on a smooth one-
    /// parameter cell family `c_k(θ)=c_k0+θ·dc_k`, `z_{L,R}(θ)=z0+θ·v`. The
    /// gradient channel of the order-exact `Jet1` (seeded with `dc`/`v` in primary slot 0) is
    /// the analytic `dM_n/dθ`; the value channel is the numeric `M_n`.
    #[test]
    fn base_moment_jets_first_derivative_matches_fd_932() {
        use crate::cubic_cell_kernel::evaluate_cell_moments;

        // Smooth one-parameter family (θ scalar). Edges move; coefficients move.
        let c0 = [0.25_f64, -0.35, 0.4, 0.15];
        let zl0 = -1.2_f64;
        let zr0 = 1.7_f64;
        let dc = [0.13_f64, 0.21, -0.17, 0.09];
        let v_l = -0.23_f64;
        let v_r = 0.31_f64;
        let cell_at = |theta: f64| DenestedCubicCell {
            left: zl0 + theta * v_l,
            right: zr0 + theta * v_r,
            c0: c0[0] + theta * dc[0],
            c1: c0[1] + theta * dc[1],
            c2: c0[2] + theta * dc[2],
            c3: c0[3] + theta * dc[3],
        };
        let max_degree = 10usize;
        let moments_at = |theta: f64| -> Vec<f64> {
            evaluate_cell_moments(cell_at(theta), max_degree)
                .expect("numeric cell moments")
                .moments
                .into_vec()
        };
        let numeric0 = moments_at(0.0);

        // Seed the jets in primary slot 0 of a width-1 primary space: each
        // coefficient/edge jet carries its θ-velocity as its slot-0 gradient.
        let p = 1usize;
        let seeded = |x: f64, vel: f64| {
            let mut g = vec![0.0; p];
            g[0] = vel;
            Jet1 { v: x, g }
        };
        let c_jets = [
            seeded(c0[0], dc[0]),
            seeded(c0[1], dc[1]),
            seeded(c0[2], dc[2]),
            seeded(c0[3], dc[3]),
        ];
        let zl_jet = seeded(zl0, v_l);
        let zr_jet = seeded(zr0, v_r);
        let m_jets = base_moment_jets(&c_jets, &zl_jet, true, &zr_jet, true, &numeric0);

        // Central finite difference of each M_n.
        let h = 1e-6_f64;
        let mp = moments_at(h);
        let mm = moments_at(-h);
        for n in 0..5 {
            let fd = (mp[n] - mm[n]) / (2.0 * h);
            let jet = &m_jets[n];
            assert!(
                (jet.value() - numeric0[n]).abs() <= 1e-12 * (1.0 + numeric0[n].abs()),
                "M_{n} value {} != numeric {}",
                jet.value(),
                numeric0[n]
            );
            assert!(
                (jet.g[0] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                "M_{n} dθ analytic {} != FD {}",
                jet.g[0],
                fd
            );
        }
    }

    /// #932 item-2 Phase B-base closure: the SECOND θ-derivative (the self-
    /// consistent `e^{−Δq}` interior `(∂q)²` cross-term + the second-order moving-
    /// edge sliver) must match a central finite difference of the analytic FIRST
    /// derivative. Probes the `Jet2` Hessian channel `h[0]` (= `d²M_n/dθ²`) of
    /// `base_moment_jets`, the all-orders exactness the Jet3/Jet4 contractions
    /// depend on.
    #[test]
    fn base_moment_jets_second_derivative_matches_fd_932() {
        use crate::cubic_cell_kernel::evaluate_cell_moments;

        let c0 = [0.25_f64, -0.35, 0.4, 0.15];
        let zl0 = -1.2_f64;
        let zr0 = 1.7_f64;
        let dc = [0.13_f64, 0.21, -0.17, 0.09];
        let v_l = -0.23_f64;
        let v_r = 0.31_f64;
        let cell_at = |theta: f64| DenestedCubicCell {
            left: zl0 + theta * v_l,
            right: zr0 + theta * v_r,
            c0: c0[0] + theta * dc[0],
            c1: c0[1] + theta * dc[1],
            c2: c0[2] + theta * dc[2],
            c3: c0[3] + theta * dc[3],
        };
        // The order-two `e^{−Δq}` interior closure retains
        // `S(z)=Σ_{k≤2}(−Δq)^k/k!`. Its `(−Δq)²` term is degree 12 in z (η is
        // cubic, hence Δq degree 6) and reaches `M_{n+12}`, up to `M_16` for
        // n≤4. Production builds cached moments beyond that; match the complete
        // budget so every Jet2 Hessian channel is exact.
        let max_degree = 27usize;
        let moments_at = |theta: f64| -> Vec<f64> {
            evaluate_cell_moments(cell_at(theta), max_degree)
                .expect("numeric cell moments")
                .moments
                .into_vec()
        };
        // Analytic first derivative dM_n/dθ from base_moment_jets at parameter θ.
        let analytic_first = |theta: f64, n: usize| -> f64 {
            let numeric = moments_at(theta);
            let seeded = |x: f64, vel: f64| {
                let g = vec![vel];
                Jet2::from_parts(x, &g, &[])
            };
            let cell = cell_at(theta);
            let c_jets = [
                seeded(cell.c0, dc[0]),
                seeded(cell.c1, dc[1]),
                seeded(cell.c2, dc[2]),
                seeded(cell.c3, dc[3]),
            ];
            let zl_jet = seeded(cell.left, v_l);
            let zr_jet = seeded(cell.right, v_r);
            let m = base_moment_jets(&c_jets, &zl_jet, true, &zr_jet, true, &numeric);
            m[n].g[0]
        };

        // Analytic second derivative from the Jet2 Hessian channel at θ=0.
        let numeric0 = moments_at(0.0);
        let seeded = |x: f64, vel: f64| {
            let g = vec![vel];
            Jet2::from_parts(x, &g, &[])
        };
        let c_jets = [
            seeded(c0[0], dc[0]),
            seeded(c0[1], dc[1]),
            seeded(c0[2], dc[2]),
            seeded(c0[3], dc[3]),
        ];
        let zl_jet = seeded(zl0, v_l);
        let zr_jet = seeded(zr0, v_r);
        let m_jets = base_moment_jets(&c_jets, &zl_jet, true, &zr_jet, true, &numeric0);

        let h = 1e-5_f64;
        for n in 0..5 {
            let fd2 = (analytic_first(h, n) - analytic_first(-h, n)) / (2.0 * h);
            let hess = m_jets[n].h[0];
            assert!(
                (hess - fd2).abs() <= 2e-4 * (1.0 + fd2.abs()),
                "M_{n} d²θ analytic {} != FD-of-analytic {}",
                hess,
                fd2
            );
        }
    }

    /// #932 item-2 Phase C: the generic `flex_timepoint_eta_chi<J>` builder must
    /// reproduce `eta = eval_coeff4_at(coeff, z) + o_infl + rho` and `chi =
    /// eval_coeff4_at(dc_da, z) + tau` on the VALUE channel, and the a/b θ-motion
    /// on the gradient channel vs a central finite difference of the same scalar
    /// expression along a smooth `(a,b)` family. Pins the bivariate-Taylor compose
    /// + the linear rho/tau add at `Jet2`.
    #[test]
    fn flex_timepoint_eta_chi_value_and_grad_932() {
        let z_obs = 0.7_f64;
        let o_infl = 0.05_f64;
        let pack = ObservedCoeffPack {
            coeff: [0.2, -0.3, 0.15, 0.05],
            dc_da: [1.1, 0.2, 0.03, 0.0],
            dc_db: [0.4, 1.05, 0.1, 0.02],
            dc_daa: [0.07, 0.02, 0.0, 0.0],
            dc_dab: [0.2, 0.09, 0.01, 0.0],
            dc_dbb: [0.11, 0.04, 0.005, 0.0],
            dc_daaa: [0.003, 0.0, 0.0, 0.0],
            dc_daab: [0.006, 0.001, 0.0, 0.0],
            dc_dabb: [0.004, 0.002, 0.0, 0.0],
            dc_dbbb: [0.008, 0.001, 0.0, 0.0],
        };
        // Single-primary family θ: a(θ)=a0+θ·a_u, b(θ)=b0+θ·b_u.
        let a0 = 0.3_f64;
        let b0 = 1.2_f64;
        let a_u = 0.25_f64;
        let b_u = -0.4_f64;
        let p = 1usize;
        let a_jet = Jet2::from_parts(a0, &[a_u], &[]);
        let b_jet = Jet2::from_parts(b0, &[b_u], &[]);
        // No rho/tau channels for this probe.
        let zero = Jet2::from_parts(0.0, &vec![0.0; p], &[]);
        let (eta, chi) = flex_timepoint_eta_chi(&a_jet, &b_jet, z_obs, o_infl, &pack, &zero, &zero);

        // Scalar reference: eta(a,b) = Σ_k c_k(a,b) z^k, c_k composed from the
        // bivariate Taylor of the pack about (a0,b0).
        let coeff_scalar = |da: f64, db: f64| -> [f64; 4] {
            std::array::from_fn(|k| {
                pack.coeff[k]
                    + pack.dc_da[k] * da
                    + pack.dc_db[k] * db
                    + 0.5 * pack.dc_daa[k] * da * da
                    + pack.dc_dab[k] * da * db
                    + 0.5 * pack.dc_dbb[k] * db * db
                    + pack.dc_daaa[k] * da * da * da / 6.0
                    + 0.5 * pack.dc_daab[k] * da * da * db
                    + 0.5 * pack.dc_dabb[k] * da * db * db
                    + pack.dc_dbbb[k] * db * db * db / 6.0
            })
        };
        let eta_scalar = |theta: f64| -> f64 {
            let c = coeff_scalar(a_u * theta, b_u * theta);
            eval_coeff4_scalar(&c, z_obs) + o_infl
        };
        let chi_scalar = |theta: f64| -> f64 {
            let dc = coeff_scalar_da(&pack, a_u * theta, b_u * theta);
            eval_coeff4_scalar(&dc, z_obs)
        };
        assert!(
            (eta.value() - eta_scalar(0.0)).abs() <= 1e-12 * (1.0 + eta_scalar(0.0).abs()),
            "eta value {} != {}",
            eta.value(),
            eta_scalar(0.0)
        );
        assert!(
            (chi.value() - chi_scalar(0.0)).abs() <= 1e-12 * (1.0 + chi_scalar(0.0).abs()),
            "chi value {} != {}",
            chi.value(),
            chi_scalar(0.0)
        );
        let h = 1e-6_f64;
        let eta_fd = (eta_scalar(h) - eta_scalar(-h)) / (2.0 * h);
        let chi_fd = (chi_scalar(h) - chi_scalar(-h)) / (2.0 * h);
        assert!(
            (eta.g[0] - eta_fd).abs() <= 1e-5 * (1.0 + eta_fd.abs()),
            "eta grad {} != FD {}",
            eta.g[0],
            eta_fd
        );
        assert!(
            (chi.g[0] - chi_fd).abs() <= 1e-5 * (1.0 + chi_fd.abs()),
            "chi grad {} != FD {}",
            chi.g[0],
            chi_fd
        );
    }

    /// Scalar Horner `Σ_k c[k] z^k` (the `f64` image of `eval_coeff4_at`).
    fn eval_coeff4_scalar(c: &[f64; 4], z: f64) -> f64 {
        let mut acc = 0.0;
        for &ck in c.iter().rev() {
            acc = acc * z + ck;
        }
        acc
    }

    /// Scalar `∂_a coeff` Taylor (the dc_da pack composed about (a0,b0)) — the
    /// reference for `chi`'s value/grad in the test above.
    fn coeff_scalar_da(pack: &ObservedCoeffPack, da: f64, db: f64) -> [f64; 4] {
        std::array::from_fn(|k| {
            pack.dc_da[k]
                + pack.dc_daa[k] * da
                + pack.dc_dab[k] * db
                + 0.5 * pack.dc_daaa[k] * da * da
                + pack.dc_daab[k] * da * db
                + 0.5 * pack.dc_dabb[k] * db * db
        })
    }

    /// #932 item-2 Phase C: `cell_coeff_jets` must reproduce the hand cell-coeff
    /// total θ-derivatives `c_k(a(θ), θ)` value + gradient (the `eta_u_poly =
    /// dc_da·a_u + coeff_u` / `coeff_bu·db` structure) vs a central FD of the
    /// scalar multivariate Taylor along a smooth `(a, {θ_u})` family. p=3 with the
    /// g-slope at axis 1.
    #[test]
    fn cell_coeff_jets_value_and_grad_932() {
        let p = 3usize;
        let g_axis = 1usize;
        let base_c = [0.2_f64, -0.3, 0.15, 0.05];
        // Minimal but fully-populated fixed pack (per-primary length p).
        let mk_run = |seed: f64| -> Vec<[f64; 4]> {
            (0..p)
                .map(|u| std::array::from_fn(|k| seed * (1.0 + u as f64) * (1.0 + k as f64) * 0.01))
                .collect()
        };
        let fixed = DenestedCellPrimaryFixedPartials {
            dc_da: [1.1, 0.2, 0.03, 0.0],
            dc_daa: [0.07, 0.02, 0.0, 0.0],
            dc_daaa: [0.003, 0.0, 0.0, 0.0],
            coeff_u: mk_run(0.9),
            coeff_au: mk_run(0.4),
            coeff_bu: mk_run(0.5),
            coeff_aau: mk_run(0.12),
            coeff_abu: mk_run(0.16),
            coeff_bbu: mk_run(0.11),
            coeff_aaau: mk_run(0.02),
            coeff_aabu: mk_run(0.03),
            coeff_abbu: mk_run(0.04),
            coeff_bbbu: mk_run(0.05),
        };
        // Smooth family θ: a(θ)=a0+θ·a_u, θ_u(θ)=u0[u]+θ·v[u].
        let a0 = 0.3_f64;
        let a_u = 0.25_f64;
        let v = [0.2_f64, -0.4, 0.33];

        let seeded = |x: f64, vel: f64| {
            let g = vec![vel];
            Jet2::from_parts(x, &g, &[])
        };
        let a_jet = seeded(a0, a_u);
        let da = tangent_jet(&a_jet);
        // du[u] = value-0 tangent carrying velocity v[u] in slot 0.
        let du: Vec<Jet2> = (0..p).map(|u| seeded(0.0, v[u])).collect();
        let jets = cell_coeff_jets(&a_jet, base_c, &fixed, g_axis, &da, &du);

        // Scalar reference: the TRUE multivariate Taylor `c_k(a(θ), {θ_u(θ)})` with
        // CORRECT factorials on every repeated axis. The non-g axes are linear in
        // `du[u]` (factor 1). The slope axis `g` (= `b`) carries its own pure-`b`
        // powers `½dc_dbb·db²`, `½dc_dabb·da·db²`, `⅙dc_dbbb·db³` (Taylor 1/n!) — NOT
        // the `coeff_bu[g]·db·du[g]` bilinear form, which would double/triple-count
        // the g-diagonal (the gam#932 fix the jet now implements).
        let scalar_c = |theta: f64| -> [f64; 4] {
            let da = a_u * theta;
            let db = v[g_axis] * theta;
            std::array::from_fn(|k| {
                let mut acc = base_c[k]
                    + fixed.dc_da[k] * da
                    + 0.5 * fixed.dc_daa[k] * da * da
                    + fixed.dc_daaa[k] * da * da * da / 6.0;
                // Linear (u != g) axes: bilinear in du[u].
                for u in 0..p {
                    if u == g_axis {
                        continue;
                    }
                    let duu = v[u] * theta;
                    acc += fixed.coeff_u[u][k] * duu
                        + fixed.coeff_au[u][k] * da * duu
                        + 0.5 * fixed.coeff_aau[u][k] * da * da * duu
                        + fixed.coeff_bu[u][k] * db * duu
                        + fixed.coeff_abu[u][k] * da * db * duu
                        + 0.5 * fixed.coeff_bbu[u][k] * db * db * duu
                        + fixed.coeff_aaau[u][k] * da * da * da * duu / 6.0
                        + 0.5 * fixed.coeff_aabu[u][k] * da * da * db * duu
                        + 0.5 * fixed.coeff_abbu[u][k] * da * db * db * duu
                        + fixed.coeff_bbbu[u][k] * db * db * db * duu / 6.0;
                }
                // Slope axis g (= b): pure-b Taylor with 1/n! factorials.
                acc += fixed.coeff_u[g_axis][k] * db
                    + fixed.coeff_au[g_axis][k] * da * db
                    + 0.5 * fixed.coeff_aau[g_axis][k] * da * da * db
                    + 0.5 * fixed.coeff_bu[g_axis][k] * db * db
                    + 0.5 * fixed.coeff_abu[g_axis][k] * da * db * db
                    + fixed.coeff_bbu[g_axis][k] * db * db * db / 6.0;
                acc
            })
        };
        let h = 1e-6_f64;
        let c0 = scalar_c(0.0);
        let cp = scalar_c(h);
        let cm = scalar_c(-h);
        for k in 0..4 {
            assert!(
                (jets[k].value() - c0[k]).abs() <= 1e-12 * (1.0 + c0[k].abs()),
                "c_{k} value {} != {}",
                jets[k].value(),
                c0[k]
            );
            let fd = (cp[k] - cm[k]) / (2.0 * h);
            assert!(
                (jets[k].g[0] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                "c_{k} grad {} != FD {}",
                jets[k].g[0],
                fd
            );
        }

        // gam#932 g-DIAGONAL Hessian: with each primary in its OWN slot, the cell
        // coefficient jet's `Hess[g,g] = ∂²c/∂b²` MUST equal `coeff_bu[g] = dc_dbb`
        // EXACTLY — NOT `2·dc_dbb`. The pre-fix `coeff_bu[g]·db·du[g]` term gave
        // `dc_dbb·db²` whose Jet2 Hessian is `2·dc_dbb` (mul's symmetric 2×); the
        // ½-factorial fix restores the true second partial. (`da` here is a pure
        // intercept perturbation with NO primary grad, so it does not leak into the
        // g-slot Hessian — isolating `∂²c/∂b²` cleanly.)
        let da_iso = Jet2::from_parts(0.0, &vec![0.0; p], &[]);
        let du_iso: Vec<Jet2> = (0..p).map(|u| Jet2::primary(0.0, u, p)).collect();
        let jets_iso = cell_coeff_jets(&da_iso, base_c, &fixed, g_axis, &da_iso, &du_iso);
        for k in 0..4 {
            let hgg = jets_iso[k].h[g_axis * p + g_axis];
            assert!(
                (hgg - fixed.coeff_bu[g_axis][k]).abs()
                    <= 1e-12 * (1.0 + fixed.coeff_bu[g_axis][k].abs()),
                "c_{k} Hess[g,g] {} != dc_dbb {} (2× = the pre-fix g-diagonal bug)",
                hgg,
                fixed.coeff_bu[g_axis][k]
            );
        }
    }

    /// #932 item-2 Phase C STEP 1: `flex_timepoint_d_cell` (the per-cell density
    /// normalization `D_cell = INV_TWO_PI·Σ_k χ_k·M_k`) must reproduce the value
    /// `INV_TWO_PI·Σ_k dc_da[k]·M_k` and the θ-gradient `d_u` (the hand
    /// `survival_flex_base_d_u` quantity) vs a central FD of the same scalar `D(θ)`
    /// on a smooth intercept-only family `a(θ)=a0+θ` with FIXED edges (isolates
    /// the interior coefficient motion + the M_k `e^{−Δq}` `−χηη_u` term — the edge
    /// flux is exercised by the moving-edge `base_moment_jets` tests). The cell's
    /// `c_k` and `dc_da_k` move with `a` per the chosen `dc_daa`/`dc_daaa` pack.
    #[test]
    fn flex_timepoint_d_cell_value_and_grad_932() {
        use crate::cubic_cell_kernel::evaluate_cell_moments;

        let zl = -1.1_f64;
        let zr = 1.6_f64;
        // a-only family: c_k and dc_da_k are cubic/quadratic in θ via the pack.
        let c_base = [0.2_f64, -0.3, 0.18, 0.06];
        let dc_da = [1.05_f64, 0.22, 0.04, 0.0];
        let dc_daa = [0.08_f64, 0.03, 0.0, 0.0];
        let dc_daaa = [0.004_f64, 0.0, 0.0, 0.0];
        let cell_at = |theta: f64| {
            let c: [f64; 4] = std::array::from_fn(|k| {
                c_base[k]
                    + dc_da[k] * theta
                    + 0.5 * dc_daa[k] * theta * theta
                    + dc_daaa[k] * theta * theta * theta / 6.0
            });
            DenestedCubicCell {
                left: zl,
                right: zr,
                c0: c[0],
                c1: c[1],
                c2: c[2],
                c3: c[3],
            }
        };
        let dc_da_at = |theta: f64| -> [f64; 4] {
            std::array::from_fn(|k| dc_da[k] + dc_daa[k] * theta + 0.5 * dc_daaa[k] * theta * theta)
        };
        let max_degree = 10usize;
        let moments_at = |theta: f64| -> Vec<f64> {
            evaluate_cell_moments(cell_at(theta), max_degree)
                .expect("numeric cell moments")
                .moments
                .into_vec()
        };
        let d_scalar = |theta: f64| -> f64 {
            let m = moments_at(theta);
            let chi = dc_da_at(theta);
            let mut acc = 0.0;
            for k in 0..4 {
                acc += chi[k] * m[k];
            }
            acc * std::f64::consts::TAU.recip()
        };

        // Jet build: single primary (slot 0) = the intercept axis, velocity 1.
        let seeded = |x: f64, vel: f64| {
            let g = vec![vel];
            Jet1 { v: x, g }
        };
        let cell0 = cell_at(0.0);
        let c_jets = [
            seeded(cell0.c0, dc_da[0]),
            seeded(cell0.c1, dc_da[1]),
            seeded(cell0.c2, dc_da[2]),
            seeded(cell0.c3, dc_da[3]),
        ];
        // chi_jets carry dc_da value + dc_daa motion (a-only family).
        let dc_da0 = dc_da_at(0.0);
        let chi_jets = [
            seeded(dc_da0[0], dc_daa[0]),
            seeded(dc_da0[1], dc_daa[1]),
            seeded(dc_da0[2], dc_daa[2]),
            seeded(dc_da0[3], dc_daa[3]),
        ];
        let template = seeded(0.0, 0.0);
        let edge_l = seeded(zl, 0.0); // fixed edge: no motion
        let edge_r = seeded(zr, 0.0);
        let numeric0 = moments_at(0.0);
        let d_jet = flex_timepoint_d_cell(
            &template, &c_jets, &chi_jets, &edge_l, true, &edge_r, true, &numeric0,
        );

        assert!(
            (d_jet.value() - d_scalar(0.0)).abs() <= 1e-10 * (1.0 + d_scalar(0.0).abs()),
            "D value {} != {}",
            d_jet.value(),
            d_scalar(0.0)
        );
        let h = 1e-6_f64;
        let fd = (d_scalar(h) - d_scalar(-h)) / (2.0 * h);
        assert!(
            (d_jet.g[0] - fd).abs() <= 1e-4 * (1.0 + fd.abs()),
            "D grad (d_u) {} != FD {}",
            d_jet.g[0],
            fd
        );
    }

    /// #932 item-2 Phase C: `cell_chi_poly_jets` (the `∂η/∂a = dc_da` family as
    /// jets) value channel == `dc_da[k]`, gradient channel == the hand
    /// `chi_u_poly = dc_daa·a_u + coeff_au` chain, vs a central FD of the scalar
    /// `dc_da_k(a(θ), {θ_u})` Taylor. p=2 with the g-slope at axis 1.
    #[test]
    fn cell_chi_poly_jets_value_and_grad_932() {
        let p = 2usize;
        let g_axis = 1usize;
        let mk_run = |seed: f64| -> Vec<[f64; 4]> {
            (0..p)
                .map(|u| std::array::from_fn(|k| seed * (1.0 + u as f64) * (1.0 + k as f64) * 0.01))
                .collect()
        };
        let fixed = DenestedCellPrimaryFixedPartials {
            dc_da: [1.05, 0.22, 0.04, 0.0],
            dc_daa: [0.08, 0.03, 0.0, 0.0],
            dc_daaa: [0.004, 0.0, 0.0, 0.0],
            coeff_u: mk_run(0.9),
            coeff_au: mk_run(0.4),
            coeff_bu: mk_run(0.5),
            coeff_aau: mk_run(0.12),
            coeff_abu: mk_run(0.16),
            coeff_bbu: mk_run(0.11),
            coeff_aaau: mk_run(0.02),
            coeff_aabu: mk_run(0.03),
            coeff_abbu: mk_run(0.04),
            coeff_bbbu: mk_run(0.05),
        };
        let a_u = 0.25_f64;
        let v = [0.2_f64, -0.4];
        let seeded = |x: f64, vel: f64| {
            let g = vec![vel];
            Jet2::from_parts(x, &g, &[])
        };
        let a_jet = seeded(0.3, a_u);
        let da = tangent_jet(&a_jet);
        let du: Vec<Jet2> = (0..p).map(|u| seeded(0.0, v[u])).collect();
        let chi = cell_chi_poly_jets(&a_jet, &fixed, g_axis, &da, &du);

        let chi_scalar = |theta: f64| -> [f64; 4] {
            let da = a_u * theta;
            let db = v[g_axis] * theta;
            std::array::from_fn(|k| {
                let mut acc =
                    fixed.dc_da[k] + fixed.dc_daa[k] * da + 0.5 * fixed.dc_daaa[k] * da * da;
                for u in 0..p {
                    let duu = v[u] * theta;
                    acc += fixed.coeff_au[u][k] * duu
                        + fixed.coeff_aau[u][k] * da * duu
                        + fixed.coeff_abu[u][k] * db * duu;
                }
                acc
            })
        };
        let h = 1e-6_f64;
        let c0 = chi_scalar(0.0);
        let cp = chi_scalar(h);
        let cm = chi_scalar(-h);
        for k in 0..4 {
            assert!(
                (chi[k].value() - c0[k]).abs() <= 1e-12 * (1.0 + c0[k].abs()),
                "chi_{k} value {} != {}",
                chi[k].value(),
                c0[k]
            );
            let fd = (cp[k] - cm[k]) / (2.0 * h);
            assert!(
                (chi[k].g[0] - fd).abs() <= 1e-5 * (1.0 + fd.abs()),
                "chi_{k} grad {} != FD {}",
                chi[k].g[0],
                fd
            );
        }
    }
    // ── §3c: real-family finite-difference and nested-dual gates ──────────────
    /// A minimal g-only survival marginal-slope family for the §3c directional gates:
    /// scalar score covariance, raw `z`, a 1-col marginal + 1-col logslope design, no
    /// score-warp/link-dev/wiggle/absorber. Deterministic synthetic data.
    fn make_g_only_flex_family(n: usize) -> SurvivalMarginalSlopeFamily {
        let event: Array1<f64> =
            Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
        let weights: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 5) as f64 * 0.1));
        let z: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| -1.0 + 2.0 * (((i * 17 + 5) % n) as f64 + 0.5) / (n as f64)),
        );
        let offset_entry: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| -0.4 + 0.7 * (((i * 11 + 3) % n) as f64 + 0.5) / (n as f64)),
        );
        let offset_exit: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| 0.1 + 0.6 * (((i * 19 + 7) % n) as f64 + 0.5) / (n as f64)),
        );
        let derivative_offset_exit: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 0.5 + 0.05 * ((i * 23 + 1) % 3) as f64));
        let marginal_design = Array2::from_shape_fn((n, 1), |(i, _)| {
            0.3 + 0.4 * (((i * 29 + 11) % n) as f64) / (n as f64)
        });
        let logslope_design = Array2::from_shape_fn((n, 1), |(i, _)| {
            0.2 + 0.5 * (((i * 37 + 9) % n) as f64) / (n as f64)
        });
        SurvivalMarginalSlopeFamily {
            n,
            event: Arc::new(event),
            weights: Arc::new(weights),
            z: Arc::new(z.insert_axis(Axis(1))),
            score_covariance: MarginalSlopeCovariance::diagonal(Array1::from(vec![1.0])).unwrap(),
            gaussian_frailty_sd: None,
            family_hyper: SurvivalMarginalSlopeFamilyHyperState::default(),
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((n, 0))),
            design_exit: DesignMatrix::from(Array2::zeros((n, 0))),
            design_derivative_exit: DesignMatrix::from(Array2::zeros((n, 0))),
            offset_entry: Arc::new(offset_entry),
            offset_exit: Arc::new(offset_exit),
            derivative_offset_exit: Arc::new(derivative_offset_exit),
            marginal_design: DesignMatrix::from(marginal_design),
            logslope_layout: DesignMatrix::from(logslope_design).into(),
            score_warp: None,
            link_dev: None,
            influence_absorber: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        }
    }

    /// The runtime Jet3 arena retains one bounded tape per worker and reuses it
    /// for an identical-width production row.
    #[test]
    fn flex_third_arena_reuses_warmed_tape_932() {
        let family = make_g_only_flex_family(16);
        let primary = flex_primary_slices(&family);
        let row = 5usize;
        let g = 0.21_f64;
        let q1 = family.offset_exit[row] + family.marginal_design.to_dense()[[row, 0]] * 0.15;
        let a1 = family
            .solve_row_survival_intercept_with_slot(
                q1,
                g,
                None,
                None,
                Some((row, SurvivalInterceptSlotKind::Exit)),
            )
            .expect("intercept solve")
            .0;
        let cached = family
            .build_cached_partition(&primary, a1, g, None, None)
            .expect("cached partition");
        let dir = Array1::from_iter((0..primary.total).map(|axis| 0.1 + 0.03 * axis as f64));

        let run = || {
            with_flex_third_jet_arena(|arena| {
                family
                    .compute_survival_timepoint_directional_jet_from_cached(
                        row, &primary, q1, primary.q1, a1, g, None, None, 0.0, &cached, &dir, arena,
                    )
                    .expect("production third-order timepoint")
            })
        };
        let retained_bytes = || with_flex_third_jet_arena(|arena| arena.allocated_bytes());
        run();
        let first = retained_bytes();
        assert!(first > 0, "FLEX third arena did not retain its warm tape");
        run();
        assert_eq!(
            retained_bytes(),
            first,
            "same-width FLEX third row grew its warmed arena"
        );
    }

    /// #932 grad-only cutover: the production grad-only timepoint
    /// (`compute_survival_timepoint_first_order_exact`, [`Jet1`]) is pinned two ways.
    /// (1) PARITY: its value + `eta_u`/`chi_u`/`d_u` equal the grad+Hessian
    /// production path (`compute_survival_timepoint_exact_jet`, [`Jet2`]) base +
    /// gradient channels to ≤1e-9 relative — the single-source claim (both drive the
    /// one `flex_timepoint_inputs_generic` builder). (2) INDEPENDENT FD WITNESS: the
    /// TOTAL derivatives `∂(eta,chi,d)/∂g` and `∂(eta,chi,d)/∂q` — each threading the
    /// per-row intercept CALIBRATION re-solve `a(g,q)` the audit flagged — match a
    /// central finite difference of the production-returned VALUE channels.
    #[test]
    fn flex_timepoint_first_order_matches_jet2_and_fd_932() {
        let n = 16usize;
        let family = make_g_only_flex_family(n);
        let primary = flex_primary_slices(&family);
        let p = primary.total;
        let row = 6usize;
        let g = 0.19_f64;
        let o_infl = 0.0_f64;

        let m_beta = 0.15_f64;
        let q1 = family.offset_exit[row] + family.marginal_design.to_dense()[[row, 0]] * m_beta;

        // Base-point intercept solve + both production timepoint packs.
        let (a1, _d1) = family
            .solve_row_survival_intercept_with_slot(
                q1,
                g,
                None,
                None,
                Some((row, SurvivalInterceptSlotKind::Exit)),
            )
            .expect("intercept solve");
        let first = family
            .compute_survival_timepoint_first_order_exact(
                row, &primary, q1, primary.q1, a1, g, None, None, o_infl,
            )
            .expect("grad-only jet1");
        let full = family
            .compute_survival_timepoint_exact_jet(
                row, &primary, q1, primary.q1, a1, g, None, None, o_infl,
            )
            .expect("grad+hess jet2");

        // (1) Parity against the Jet2 production path (single-source).
        let rel = |a: f64, b: f64| (a - b).abs() <= 1e-9 * (1.0 + b.abs());
        assert!(
            rel(first.eta, full.eta),
            "eta {} != {}",
            first.eta,
            full.eta
        );
        assert!(
            rel(first.chi, full.chi),
            "chi {} != {}",
            first.chi,
            full.chi
        );
        assert!(rel(first.d, full.d), "d {} != {}", first.d, full.d);
        for u in 0..p {
            assert!(
                rel(first.eta_u[u], full.eta_u[u]),
                "eta_u[{u}] {} != {}",
                first.eta_u[u],
                full.eta_u[u]
            );
            assert!(
                rel(first.chi_u[u], full.chi_u[u]),
                "chi_u[{u}] {} != {}",
                first.chi_u[u],
                full.chi_u[u]
            );
            assert!(
                rel(first.d_u[u], full.d_u[u]),
                "d_u[{u}] {} != {}",
                first.d_u[u],
                full.d_u[u]
            );
        }

        // (2) Independent central-FD witness of the returned VALUE channels. The
        // closure re-solves the calibration intercept `a(g,q)` exactly as production
        // does, so the FD is the genuine TOTAL derivative (intercept chain included).
        let eval = |gg: f64, qq: f64| -> (f64, f64, f64) {
            let (aa, _) = family
                .solve_row_survival_intercept_with_slot(
                    qq,
                    gg,
                    None,
                    None,
                    Some((row, SurvivalInterceptSlotKind::Exit)),
                )
                .expect("fd intercept solve");
            let tp = family
                .compute_survival_timepoint_first_order_exact(
                    row, &primary, qq, primary.q1, aa, gg, None, None, o_infl,
                )
                .expect("fd grad-only");
            (tp.eta, tp.chi, tp.d)
        };
        let h = 1e-6_f64;
        let (eta_gp, chi_gp, d_gp) = eval(g + h, q1);
        let (eta_gm, chi_gm, d_gm) = eval(g - h, q1);
        let (eta_qp, chi_qp, d_qp) = eval(g, q1 + h);
        let (eta_qm, chi_qm, d_qm) = eval(g, q1 - h);
        let fd = |plus: f64, minus: f64| (plus - minus) / (2.0 * h);
        let check = |label: &str, analytic: f64, numeric: f64| {
            assert!(
                (analytic - numeric).abs() <= 1e-5 * (1.0 + analytic.abs()),
                "{label}: analytic {analytic} != fd {numeric}"
            );
        };
        check("d eta/dg", first.eta_u[primary.g], fd(eta_gp, eta_gm));
        check("d chi/dg", first.chi_u[primary.g], fd(chi_gp, chi_gm));
        check("d d/dg", first.d_u[primary.g], fd(d_gp, d_gm));
        check("d eta/dq", first.eta_u[primary.q1], fd(eta_qp, eta_qm));
        check("d chi/dq", first.chi_u[primary.q1], fd(chi_qp, chi_qm));
        check("d d/dq", first.d_u[primary.q1], fd(d_qp, d_qm));
    }

    /// #932 item-2 STEP 4 (TRUNCATION-FREE gate): the nested-dual ORACLE `Dual22`
    /// instantiation of `flex_timepoint_inputs_generic` reproduces the production
    /// p-primary `Jet4` instantiation's mixed 4th-order bidirectional contraction
    /// EXACTLY, via an INDEPENDENT `2 + 2` composition order (not `2 + 1 + 1`).
    ///
    /// `Dual22` seeds two scalar directions `s = d1`, `t = d2` (each to 2nd order)
    /// and reads the `∂²_s ∂²_t` channel `= Σ_abcd ℓ_abcd d1_a d1_b d2_c d2_d`.
    /// The `Jet4` path seeds `eps = del = d2` — matrix `Σ_cd ℓ_uvcd d2_c d2_d` —
    /// then contracts both free axes by `d1`; the same scalar by ℓ's symmetry.
    /// Neither path uses a finite difference nor the incomplete §D hand oracle, so
    /// this replaces the flex jet4 bidirectional's 1e-3 scalar-FD sanity (the last
    /// FD-limited seam in the #932 tower) with a machine-precision independent
    /// check. Several random `(d1, d2)` pin the whole tensor, not one slice.
    #[test]
    fn flex_timepoint_inputs_nested_dual_matches_jet4_contraction_932() {
        let n = 16usize;
        let family = make_ghw_flex_family(n);
        let primary = flex_primary_slices(&family);
        let p = primary.total;
        let row = 6usize;
        let g = 0.2_f64;

        let h_len = primary.h.as_ref().map(|r| r.len()).unwrap_or(0);
        let w_len = primary.w.as_ref().map(|r| r.len()).unwrap_or(0);
        let beta_h = Array1::from_iter(
            (0..h_len).map(|i| 0.1 + 0.05 * (i as f64) - 0.02 * ((i % 2) as f64)),
        );
        let beta_w = Array1::from_iter(
            (0..w_len).map(|i| -0.08 + 0.04 * (i as f64) + 0.01 * ((i % 3) as f64)),
        );
        let bh = Some(&beta_h);
        let bw = Some(&beta_w);

        let m_beta = 0.15_f64;
        let q1 = family.offset_exit[row] + family.marginal_design.to_dense()[[row, 0]] * m_beta;
        let o_infl = 0.0_f64;
        let a1 = family
            .solve_row_survival_intercept_with_slot(
                q1,
                g,
                bh,
                bw,
                Some((row, SurvivalInterceptSlotKind::Exit)),
            )
            .expect("intercept solve")
            .0;
        let cached = family
            .build_cached_partition(&primary, a1, g, bh, bw)
            .expect("cached partition");
        let (obs_coeff, obs_fixed) =
            observed_fixed_for(&family, &primary, row, a1, g, bh, bw).expect("obs fixed");
        let cells = cells_from_cached(&cached);
        let z_obs = family.observed_score_projection(row);
        let d_check = family
            .evaluate_survival_denom_d(a1, g, bh, bw)
            .expect("denom");

        for trial in 0..4usize {
            let f = trial as f64;
            let d1 =
                Array1::from_iter((0..p).map(|c| {
                    0.11 + 0.03 * (c as f64) - 0.02 * (((c + trial) % 2) as f64) + 0.01 * f
                }));
            let d2 = Array1::from_iter((0..p).map(|c| {
                -0.06 + 0.045 * (((c + trial) % 3) as f64) + 0.02 * (c as f64) - 0.015 * f
            }));

            // Production p-primary Jet4: eps = del = d2 ⇒ matrix Σ_cd ℓ_uvcd d2_c d2_d.
            let tpl4 = Jet4::primary(0.0, usize::MAX, p, 0.0, 0.0);
            let b4 = Jet4::primary(g, primary.g, p, d2[primary.g], d2[primary.g]);
            let du4: Vec<Jet4> = (0..p)
                .map(|u| Jet4::primary(0.0, u, p, d2[u], d2[u]))
                .collect();
            let q4 = add_const(&du4[primary.q1], q1);
            let (eta4, chi4, d4) = flex_timepoint_inputs_generic(
                &tpl4,
                &b4,
                &du4,
                a1,
                d_check,
                primary.g,
                primary.infl,
                &q4,
                &const_jet_like(&tpl4, 1.0),
                z_obs,
                o_infl,
                obs_coeff,
                &obs_fixed,
                &cells,
            )
            .expect("generic jet4");
            let contract = |m: &Vec<f64>| -> f64 {
                let mut s = 0.0;
                for u in 0..p {
                    for v in 0..p {
                        s += d1[u] * d1[v] * m[u * p + v];
                    }
                }
                s
            };
            let eta_ref = contract(&eta4.eps_del.h);
            let chi_ref = contract(&chi4.eps_del.h);
            let d_ref = contract(&d4.eps_del.h);

            // Nested-dual ORACLE: s = d1, t = d2 ⇒ channel ∂²_s ∂²_t.
            let tpl2 = Dual22::seed_directional(0.0, 0.0, 0.0);
            let b2 = Dual22::seed_directional(g, d1[primary.g], d2[primary.g]);
            let du2: Vec<Dual22> = (0..p)
                .map(|u| Dual22::seed_directional(0.0, d1[u], d2[u]))
                .collect();
            let q2 = add_const(&du2[primary.q1], q1);
            let (eta2, chi2, d2n) = flex_timepoint_inputs_generic(
                &tpl2,
                &b2,
                &du2,
                a1,
                d_check,
                primary.g,
                primary.infl,
                &q2,
                &const_jet_like(&tpl2, 1.0),
                z_obs,
                o_infl,
                obs_coeff,
                &obs_fixed,
                &cells,
            )
            .expect("generic dual22");
            let eta_got = eta2.channels()[8];
            let chi_got = chi2.channels()[8];
            let d_got = d2n.channels()[8];

            let check = |label: &str, got: f64, refv: f64| {
                assert!(
                    (got - refv).abs() <= 1e-9 * (1.0 + refv.abs()),
                    "trial {trial} {label}: nested-dual {got} != jet4 contraction {refv}"
                );
            };
            check("eta_uv_uv", eta_got, eta_ref);
            check("chi_uv_uv", chi_got, chi_ref);
            check("d_uv_uv", d_got, d_ref);
        }
    }

    // ── §3c h/w channels: g+h+w scalar-FD gate ────────────────────────────────
    //
    // With score-warp(`h`) AND link-dev(`w`) active, the OBSERVED eta/chi carry the
    // `h`/`w` primaries, and their bidirectional derivatives involve the
    // h/w channel weights' OWN (a,b)-Taylor (w: `link_basis_cell_*partials`; h:
    // a-independent, `coeff_u[h]=b·H(z_obs)`/`coeff_bu[h]=H(z_obs)`). `observed_fixed_
    // for` packs these into a `DenestedCellPrimaryFixedPartials` at the observed point;
    // `cell_coeff_jets`/`cell_chi_poly_jets` raise them to all orders. This gate pins
    // the g+h+w Jet4 contraction against an independent scalar finite difference.

    /// A score-warp / link-dev deviation runtime for the g+h+w gate.
    fn flex_test_deviation_runtime() -> DeviationRuntime {
        build_score_warp_deviation_block_from_seed(
            &Array1::from(vec![-1.0, 0.0, 1.0]),
            &DeviationBlockConfig {
                degree: 3,
                num_internal_knots: 1,
                penalty_order: 2,
                penalty_orders: vec![1, 2, 3],
                double_penalty: false,
                monotonicity_eps: 1e-4,
            },
        )
        .expect("build test deviation runtime")
        .runtime
    }

    /// A g+h+w survival family: like `make_g_only_flex_family` but with BOTH a
    /// score-warp and a link-dev runtime installed (scalar score dim).
    fn make_ghw_flex_family(n: usize) -> SurvivalMarginalSlopeFamily {
        let mut family = make_g_only_flex_family(n);
        family.score_warp = Some(flex_test_deviation_runtime());
        family.link_dev = Some(flex_test_deviation_runtime());
        family
    }

    fn make_complete_family_map_fixture() -> (SurvivalMarginalSlopeFamily, Vec<ParameterBlockState>)
    {
        let mut family = make_ghw_flex_family(1);
        let knots = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let degree = 3usize;
        let wiggle_width = time_wiggle_basis_ncols(&knots, degree).expect("timewiggle basis width");
        let time_width = 1 + wiggle_width;
        let mut entry_design = Array2::zeros((1, time_width));
        let mut exit_design = Array2::zeros((1, time_width));
        let mut derivative_design = Array2::zeros((1, time_width));
        entry_design[[0, 0]] = 0.25;
        exit_design[[0, 0]] = 0.45;
        derivative_design[[0, 0]] = 0.15;
        family.design_entry = DesignMatrix::from(entry_design);
        family.design_exit = DesignMatrix::from(exit_design);
        family.design_derivative_exit = DesignMatrix::from(derivative_design);
        family.offset_entry = Arc::new(Array1::from(vec![0.20]));
        family.offset_exit = Arc::new(Array1::from(vec![0.35]));
        family.derivative_offset_exit = Arc::new(Array1::from(vec![0.80]));
        family.marginal_design = DesignMatrix::from(
            Array2::from_shape_vec((1, 1), vec![0.30]).expect("marginal fixture shape"),
        );
        family.logslope_layout = DesignMatrix::from(
            Array2::from_shape_vec((1, 1), vec![0.40]).expect("logslope fixture shape"),
        )
        .into();
        family.influence_absorber = Some(
            Array2::from_shape_vec((1, 2), vec![0.20, -0.15]).expect("influence fixture shape"),
        );
        family.time_wiggle_knots = Some(knots);
        family.time_wiggle_degree = Some(degree);
        family.time_wiggle_ncols = wiggle_width;
        family.event = Arc::new(Array1::from(vec![1.0]));
        family.weights = Arc::new(Array1::from(vec![0.9]));
        family.z =
            Arc::new(Array2::from_shape_vec((1, 1), vec![0.2]).expect("score fixture shape"));
        family.gaussian_frailty_sd = Some(0.6);

        let mut beta_time = Array1::zeros(time_width);
        beta_time[0] = 0.10;
        for local in 0..wiggle_width {
            beta_time[1 + local] = 0.006 + 0.002 * local as f64;
        }
        let beta_marginal = Array1::from(vec![0.12]);
        let beta_logslope = Array1::from(vec![0.20]);
        let score_width = family
            .score_warp
            .as_ref()
            .expect("score runtime")
            .basis_dim();
        let link_width = family.link_dev.as_ref().expect("link runtime").basis_dim();
        let beta_score =
            Array1::from_iter((0..score_width).map(|axis| 0.004 * (1.0 + axis as f64)));
        let beta_link = Array1::from_iter((0..link_width).map(|axis| -0.003 + 0.001 * axis as f64));
        let beta_influence = Array1::from(vec![0.03, -0.02]);
        let time_eta = family.design_exit.dot_row(0, &beta_time) + family.offset_exit[0];
        let marginal_eta = family.marginal_design.dot_row(0, &beta_marginal);
        let logslope_eta = family
            .logslope_layout
            .coefficient_design()
            .dot_row(0, &beta_logslope);
        let states = vec![
            ParameterBlockState {
                beta: beta_time,
                eta: Array1::from(vec![time_eta]),
            },
            ParameterBlockState {
                beta: beta_marginal,
                eta: Array1::from(vec![marginal_eta]),
            },
            ParameterBlockState {
                beta: beta_logslope,
                eta: Array1::from(vec![logslope_eta]),
            },
            ParameterBlockState {
                beta: beta_score,
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: beta_link,
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: beta_influence,
                eta: Array1::zeros(1),
            },
        ];
        (family, states)
    }

    #[test]
    fn complete_family_map_owns_every_coefficient_block_and_log_sigma_scale_stack() {
        let (family, states) = make_complete_family_map_fixture();
        let primary = flex_primary_slices(&family);
        let slices = block_slices(&family, &states);
        let sigma = family.gaussian_frailty_sd.expect("fixture sigma");
        let sigma2 = sigma * sigma;
        let alpha = sigma2 / (1.0 + sigma2);
        let scale = family.probit_frailty_scale();
        let first = FlexFamilyRowDirection {
            entry: 0.07,
            exit: -0.04,
            derivative_exit: 0.03,
            probit_scale: -scale * alpha,
        };
        let second = FlexFamilyRowDirection {
            entry: -0.02,
            exit: 0.05,
            derivative_exit: -0.01,
            probit_scale: scale * alpha * (3.0 * alpha - 2.0),
        };
        let jets = family
            .build_flex_family_coefficient_jets::<Jet2>(
                0, &states, &primary, &slices, first, second, None,
            )
            .expect("complete family coefficient map");
        let q = family
            .row_dynamic_q_geometry(0, &states)
            .expect("analytic dynamic q geometry");
        let check = |actual: f64, expected: f64, channel: &str| {
            let tolerance = 1024.0 * f64::EPSILON * (1.0 + actual.abs().max(expected.abs()));
            assert!(
                (actual - expected).abs() <= tolerance,
                "{channel}: actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
            );
        };
        for local in 0..slices.time.len() {
            let axis = slices.time.start + local;
            check(jets.q0.v.g[axis], q.dq0_time[local], "q0 time");
            check(jets.q1.v.g[axis], q.dq1_time[local], "q1 time");
            check(jets.qd1.v.g[axis], q.dqd1_time[local], "qd1 time");
        }
        for local in 0..slices.marginal.len() {
            let axis = slices.marginal.start + local;
            check(jets.q0.v.g[axis], q.dq0_marginal[local], "q0 marginal");
            check(jets.q1.v.g[axis], q.dq1_marginal[local], "q1 marginal");
            check(jets.qd1.v.g[axis], q.dqd1_marginal[local], "qd1 marginal");
        }
        assert_eq!(
            jets.g.v.g[slices.logslope.start],
            family.logslope_layout.coefficient_design().to_dense()[[0, 0]],
        );
        let h_primary = primary.h.as_ref().expect("score primary").start;
        let h_coefficient = slices.score_warp.as_ref().expect("score slice").start;
        assert_eq!(jets.du[h_primary].v.g[h_coefficient], 1.0);
        let w_primary = primary.w.as_ref().expect("link primary").start;
        let w_coefficient = slices.link_dev.as_ref().expect("link slice").start;
        assert_eq!(jets.du[w_primary].v.g[w_coefficient], 1.0);
        let influence_primary = primary.infl.expect("influence primary");
        let influence_range = slices.influence.as_ref().expect("influence slice");
        assert_eq!(jets.du[influence_primary].v.g[influence_range.start], 0.20);
        assert_eq!(
            jets.du[influence_primary].v.g[influence_range.start + 1],
            -0.15
        );
        assert_eq!(jets.scale_ratio.v.value(), 1.0);
        check(jets.scale_ratio.g.value(), -alpha, "ds/s");
        check(
            jets.scale_ratio.h.value(),
            alpha * (3.0 * alpha - 2.0),
            "d2s/s",
        );
        assert!(jets.scale_ratio.g.g.iter().all(|value| *value == 0.0));
        assert!(jets.scale_ratio.h.h.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn complete_family_row_dual2_jet2_and_jet3_share_channels_without_fd() {
        let (family, states) = make_complete_family_map_fixture();
        let slices = block_slices(&family, &states);
        let scale = family.probit_frailty_scale();
        let first = FlexFamilyRowDirection {
            entry: 0.04,
            exit: -0.03,
            derivative_exit: 0.02,
            probit_scale: -0.08 * scale,
        };
        let second = FlexFamilyRowDirection {
            entry: -0.01,
            exit: 0.025,
            derivative_exit: 0.015,
            probit_scale: 0.03 * scale,
        };
        let order2 = family
            .flex_family_direction_row_terms(0, &states, first, second, None)
            .expect("Dual2<Jet2> family row");
        let direction =
            Array1::from_iter((0..slices.total).map(|axis| -0.035 + 0.007 * (axis % 9) as f64));
        let order3 = family
            .flex_family_direction_row_terms(0, &states, first, second, Some(&direction))
            .expect("Dual2<Jet3> family row");
        let assert_same = |left: &FlexFamilyCoefficientTerms,
                           right: &FlexFamilyCoefficientTerms,
                           channel: &str| {
            assert_eq!(
                left.objective.to_bits(),
                right.objective.to_bits(),
                "{channel} V"
            );
            for axis in 0..slices.total {
                assert_eq!(
                    left.gradient[axis].to_bits(),
                    right.gradient[axis].to_bits(),
                    "{channel} g[{axis}]"
                );
                for other in 0..slices.total {
                    assert_eq!(
                        left.hessian[[axis, other]].to_bits(),
                        right.hessian[[axis, other]].to_bits(),
                        "{channel} H[{axis},{other}]"
                    );
                }
            }
        };
        assert_same(&order2.first, &order3.first, "first");
        assert_same(&order2.second, &order3.second, "second");
        let drift = order3.directional.expect("nonzero Jet3 beta drift");
        let expected_value = order2.first.gradient.dot(&direction);
        let tolerance =
            |actual: f64, expected: f64| 1.0e-10 * (1.0 + actual.abs().max(expected.abs()));
        assert!(
            (drift.objective - expected_value).abs() <= tolerance(drift.objective, expected_value),
            "family beta drift V {} != g·d {}",
            drift.objective,
            expected_value,
        );
        for axis in 0..slices.total {
            let expected_gradient = order2.first.hessian.row(axis).dot(&direction);
            assert!(
                (drift.gradient[axis] - expected_gradient).abs()
                    <= tolerance(drift.gradient[axis], expected_gradient),
                "family beta drift g[{axis}] {} != H[row]·d {}",
                drift.gradient[axis],
                expected_gradient,
            );
        }
        assert!(drift.hessian.iter().all(|value| value.is_finite()));
        assert!(drift.hessian.iter().any(|value| *value != 0.0));

        // A design direction is not a beta direction: its epsilon seed owns
        // both X_psi beta and X_psi.  With the fixture's scalar marginal
        // design this gives two independent analytic identities, including
        // the pullback-map contribution to the marginal score.
        let x = family.marginal_design.to_dense()[[0, 0]];
        let x_psi = 0.17;
        let beta = states[1].beta[0];
        let design = family
            .flex_family_design_direction_row_terms(
                0,
                &states,
                first,
                second,
                1,
                &Array1::from(vec![x_psi]),
            )
            .expect("Dual2<Jet3> family-by-design row");
        assert_same(&order2.first, &design.first, "design first base");
        assert_same(&order2.second, &design.second, "design second base");
        let design_drift = design.directional.expect("nonzero design drift");
        let marginal_axis = slices.marginal.start;
        let h_psi = x_psi * beta;
        let expected_design_objective = order2.first.gradient[marginal_axis] * h_psi / x;
        assert!(
            (design_drift.objective - expected_design_objective).abs()
                <= tolerance(design_drift.objective, expected_design_objective),
            "family-by-design objective {} != analytic {}",
            design_drift.objective,
            expected_design_objective,
        );
        let expected_design_score = x_psi * order2.first.gradient[marginal_axis] / x
            + h_psi * order2.first.hessian[[marginal_axis, marginal_axis]] / x;
        assert!(
            (design_drift.gradient[marginal_axis] - expected_design_score).abs()
                <= tolerance(design_drift.gradient[marginal_axis], expected_design_score,),
            "family-by-design marginal score {} != analytic pullback {}",
            design_drift.gradient[marginal_axis],
            expected_design_score,
        );
        assert!(design_drift.hessian.iter().all(|value| value.is_finite()));
        assert!(design_drift.hessian.iter().any(|value| *value != 0.0));
    }

    /// `flex_timepoint_inputs_generic` at Jet4 must match a scalar finite difference
    /// of the real intercept solve with active score-warp and link-deviation primaries.
    #[test]
    fn flex_timepoint_inputs_ghw_jet4_matches_scalar_fd_932() {
        let n = 16usize;
        let family = make_ghw_flex_family(n);
        let primary = flex_primary_slices(&family);
        let p = primary.total;
        let row = 6usize;
        let g = 0.2_f64;

        // Non-trivial h/w coefficients (lengths from the primary layout).
        let h_len = primary.h.as_ref().map(|r| r.len()).unwrap_or(0);
        let w_len = primary.w.as_ref().map(|r| r.len()).unwrap_or(0);
        let beta_h = Array1::from_iter(
            (0..h_len).map(|i| 0.1 + 0.05 * (i as f64) - 0.02 * ((i % 2) as f64)),
        );
        let beta_w = Array1::from_iter(
            (0..w_len).map(|i| -0.08 + 0.04 * (i as f64) + 0.01 * ((i % 3) as f64)),
        );
        let bh = Some(&beta_h);
        let bw = Some(&beta_w);

        let m_beta = 0.15_f64;
        let q1 = family.offset_exit[row] + family.marginal_design.to_dense()[[row, 0]] * m_beta;
        let o_infl = 0.0_f64;
        let a1 = family
            .solve_row_survival_intercept_with_slot(
                q1,
                g,
                bh,
                bw,
                Some((row, SurvivalInterceptSlotKind::Exit)),
            )
            .expect("intercept solve")
            .0;
        let cached = family
            .build_cached_partition(&primary, a1, g, bh, bw)
            .expect("cached partition");

        let (obs_coeff, obs_fixed) =
            observed_fixed_for(&family, &primary, row, a1, g, bh, bw).expect("obs fixed");
        let cells = cells_from_cached(&cached);
        let z_obs = family.observed_score_projection(row);
        let d_check = family
            .evaluate_survival_denom_d(a1, g, bh, bw)
            .expect("denom");

        // ── #932 scalar-FD oracle: the authoritative [q1,q1]
        // bidirectional reference.
        //
        // `q1` (= primary.q1) enters the OBSERVED eta/chi/D only through the lifted
        // intercept a (no de-nested-cell coefficient dependence: the eta_aa·a_u² and
        // r_uv terms vanish at the q1 axis), so eta_uv_uv[q1,q1] = D_d1 D_d2(∂²_q1
        // eta_obs) is a pure chain through the intercept solve a(q1, β). Finite-
        // difference the observed scalars eta_obs/chi_obs/D as functions of the q1
        // marginal and the (dir1, dir2)-perturbed primaries. The intercept root is
        // solved at every stencil point, so the oracle includes the complete moving
        // boundary response. The probe also pins the lifted intercept itself.
        let (oracle_eta_uvuv, oracle_chi_uvuv, oracle_d_uvuv) = {
            let dir1 = Array1::from_iter(
                (0..p).map(|c| 0.12 + 0.04 * (c as f64) - 0.01 * ((c % 2) as f64)),
            );
            let dir2 = Array1::from_iter(
                (0..p).map(|c| -0.07 + 0.05 * ((c % 3) as f64) + 0.02 * (c as f64)),
            );
            // Solve the intercept at θ + (an arbitrary per-primary perturbation `pert`)
            // and read back (a, eta_obs, chi_obs, D). `pert[u]` shifts primary u's
            // scalar: q1→the marginal q1 argument, g→the slope, h-range[i]→β_h[i],
            // w-range[i]→β_w[i]; every other primary (q0/qd1/infl) leaves the intercept
            // unchanged (a depends only on q1/g/β_h/β_w). This drives BOTH the directional
            // contractions and the (u,v) Hessian stencils below.
            let scalars_of = |pert: &Array1<f64>| -> (f64, f64, f64, f64) {
                let q1_pert = q1 + pert[primary.q1];
                let g_pert = g + pert[primary.g];
                let bh_pert: Array1<f64> = Array1::from_iter(
                    (0..h_len).map(|i| beta_h[i] + pert[primary.h.as_ref().unwrap().start + i]),
                );
                let bw_pert: Array1<f64> = Array1::from_iter(
                    (0..w_len).map(|i| beta_w[i] + pert[primary.w.as_ref().unwrap().start + i]),
                );
                let a_pert = family
                    .solve_row_survival_intercept_with_slot(
                        q1_pert,
                        g_pert,
                        Some(&bh_pert),
                        Some(&bw_pert),
                        None,
                    )
                    .expect("oracle intercept solve")
                    .0;
                let obs = family
                    .observed_denested_cell_partials(
                        row,
                        a_pert,
                        g_pert,
                        Some(&bh_pert),
                        Some(&bw_pert),
                    )
                    .expect("oracle observed partials");
                let d_pert = family
                    .evaluate_survival_denom_d(a_pert, g_pert, Some(&bh_pert), Some(&bw_pert))
                    .expect("oracle denom");
                (
                    a_pert,
                    eval_coeff4_at(&obs.coeff, z_obs) + o_infl,
                    eval_coeff4_at(&obs.dc_da, z_obs),
                    d_pert,
                )
            };

            let hq = 2.0e-3_f64;
            let ht = 3.0e-3_f64;
            // Build the per-primary perturbation vector for a stencil point:
            // su·e_u + sv·e_v + t1·dir1 + t2·dir2 (u and v may coincide → su,sv add).
            let pert_vec =
                |su: f64, u: usize, sv: f64, v: usize, t1: f64, t2: f64| -> Array1<f64> {
                    let mut pert = &dir1 * t1 + &dir2 * t2;
                    pert[u] += su;
                    pert[v] += sv;
                    pert
                };
            // The mixed bidirectional Hessian entry [u,v] of the observed scalars
            // (a, eta_obs, chi_obs, D) all at once: ∂_u ∂_v (D_dir1 D_dir2 ·). For u==v
            // a ∂²_u 3-point stencil; for u!=v the 2×2 cross stencil. The directional
            // second cross is the outer 2×2 over (t1,t2) — a 4×4 stencil product
            // (Hessian × directional). Returns one tuple so the four scalars share the
            // (expensive) intercept solves.
            let mixed = |u: usize, v: usize| -> (f64, f64, f64, f64) {
                let acc =
                    |w: f64, su: f64, sv: f64, t1: f64, t2: f64, out: &mut (f64, f64, f64, f64)| {
                        let s = scalars_of(&pert_vec(su, u, sv, v, t1, t2));
                        out.0 += w * s.0;
                        out.1 += w * s.1;
                        out.2 += w * s.2;
                        out.3 += w * s.3;
                    };
                let hess_uv = |t1: f64, t2: f64| -> (f64, f64, f64, f64) {
                    let mut o = (0.0, 0.0, 0.0, 0.0);
                    if u == v {
                        acc(1.0, hq, 0.0, t1, t2, &mut o);
                        acc(-2.0, 0.0, 0.0, t1, t2, &mut o);
                        acc(1.0, -hq, 0.0, t1, t2, &mut o);
                        let inv = 1.0 / (hq * hq);
                        (o.0 * inv, o.1 * inv, o.2 * inv, o.3 * inv)
                    } else {
                        acc(1.0, hq, hq, t1, t2, &mut o);
                        acc(-1.0, hq, -hq, t1, t2, &mut o);
                        acc(-1.0, -hq, hq, t1, t2, &mut o);
                        acc(1.0, -hq, -hq, t1, t2, &mut o);
                        let inv = 1.0 / (4.0 * hq * hq);
                        (o.0 * inv, o.1 * inv, o.2 * inv, o.3 * inv)
                    }
                };
                let a = hess_uv(ht, ht);
                let b = hess_uv(ht, -ht);
                let c = hess_uv(-ht, ht);
                let d = hess_uv(-ht, -ht);
                let inv = 1.0 / (4.0 * ht * ht);
                (
                    (a.0 - b.0 - c.0 + d.0) * inv,
                    (a.1 - b.1 - c.1 + d.1) * inv,
                    (a.2 - b.2 - c.2 + d.2) * inv,
                    (a.3 - b.3 - c.3 + d.3) * inv,
                )
            };

            // Cross-check: the jet's lifted intercept a_uv_uv[q1,q1] (eps_del Hessian)
            // == the scalar-FD a-Hessian (the verdict that confirmed the jet correct).
            let template4 = Jet4::primary(0.0, usize::MAX, p, 0.0, 0.0);
            let b_jet4 = Jet4::primary(g, primary.g, p, dir1[primary.g], dir2[primary.g]);
            let du4: Vec<Jet4> = (0..p)
                .map(|u| Jet4::primary(0.0, u, p, dir1[u], dir2[u]))
                .collect();
            let q_jet4 = add_const(&du4[primary.q1], q1);
            let scale_ratio4 = const_jet_like(&template4, 1.0);
            let residual_probe = |a: &Jet4| {
                calibration_residual_jet(
                    a,
                    &b_jet4,
                    primary.g,
                    &du4,
                    &q_jet4,
                    &scale_ratio4,
                    &cells,
                )
            };
            let a_jet_probe = lift_intercept_flex(&template4, a1, 1.0 / d_check, 4, residual_probe);
            let jet_a_uvuv = a_jet_probe.eps_del.h[primary.q1 * p + primary.q1];

            // Oracle matrices: the FULL bidirectional (a, eta, chi, D) Hessian for EVERY
            // (u,v) — the scalar-FD of the real intercept solve is ground truth at every
            // entry, so the gate asserts the whole eta/chi/D_uv_uv matrix against it.
            // The Hessian is symmetric, so compute v>=u and mirror. `a`-channel
            // diagonal [q1,q1] feeds the lifted-intercept cross-check below.
            let mut o_eta = Array2::<f64>::zeros((p, p));
            let mut o_chi = Array2::<f64>::zeros((p, p));
            let mut o_d = Array2::<f64>::zeros((p, p));
            let mut ref_a_uvuv = 0.0_f64;
            for u in 0..p {
                for v in u..p {
                    let (a_uvuv, eta_uvuv, chi_uvuv, d_uvuv) = mixed(u, v);
                    o_eta[[u, v]] = eta_uvuv;
                    o_eta[[v, u]] = eta_uvuv;
                    o_chi[[u, v]] = chi_uvuv;
                    o_chi[[v, u]] = chi_uvuv;
                    o_d[[u, v]] = d_uvuv;
                    o_d[[v, u]] = d_uvuv;
                    if u == primary.q1 && v == primary.q1 {
                        ref_a_uvuv = a_uvuv;
                    }
                }
            }
            assert!(
                (jet_a_uvuv - ref_a_uvuv).abs() <= 1e-3 * (1.0 + ref_a_uvuv.abs()),
                "#932 PROBE a_uv_uv[q1,q1]: jet {jet_a_uvuv} != scalar-FD {ref_a_uvuv} \
                 (diff {})",
                jet_a_uvuv - ref_a_uvuv,
            );
            (o_eta, o_chi, o_d)
        };

        // ── Jet4 bidirectional ──
        let dir1 =
            Array1::from_iter((0..p).map(|c| 0.12 + 0.04 * (c as f64) - 0.01 * ((c % 2) as f64)));
        let dir2 =
            Array1::from_iter((0..p).map(|c| -0.07 + 0.05 * ((c % 3) as f64) + 0.02 * (c as f64)));
        let template4 = Jet4::primary(0.0, usize::MAX, p, 0.0, 0.0);
        let b_jet4 = Jet4::primary(g, primary.g, p, dir1[primary.g], dir2[primary.g]);
        let du4: Vec<Jet4> = (0..p)
            .map(|u| Jet4::primary(0.0, u, p, dir1[u], dir2[u]))
            .collect();
        let q_jet4 = add_const(&du4[primary.q1], q1);
        let (eta4, chi4, d4) = flex_timepoint_inputs_generic(
            &template4,
            &b_jet4,
            &du4,
            a1,
            d_check,
            primary.g,
            primary.infl,
            &q_jet4,
            &const_jet_like(&template4, 1.0),
            z_obs,
            o_infl,
            obs_coeff,
            &obs_fixed,
            &cells,
        )
        .expect("generic jet4");

        // Assert the full bidirectional Hessian against the scalar-FD oracle at every
        // entry. Report every mismatch so a channel regression is immediately local.
        let cmp_mat_oracle = |label: &str, jet: &Vec<f64>, oracle: &Array2<f64>| {
            let mut fails: Vec<String> = Vec::new();
            for u in 0..p {
                for v in 0..p {
                    let o = oracle[[u, v]];
                    let j = jet[u * p + v];
                    if (j - o).abs() > 1e-3 * (1.0 + o.abs()) {
                        let rel = (j - o).abs() / (1.0 + o.abs());
                        fails.push(format!("[{u},{v}] jet {j:.6} oracle {o:.6} rel {rel:.2e}"));
                    }
                }
            }
            assert!(
                fails.is_empty(),
                "{label} jet != scalar-FD oracle at {} entr{}: {}",
                fails.len(),
                if fails.len() == 1 { "y" } else { "ies" },
                fails.join("; "),
            );
        };
        cmp_mat_oracle("eta_uv_uv", &eta4.eps_del.h, &oracle_eta_uvuv);
        cmp_mat_oracle("chi_uv_uv", &chi4.eps_del.h, &oracle_chi_uvuv);
        cmp_mat_oracle("d_uv_uv", &d4.eps_del.h, &oracle_d_uvuv);
    }
}

#[cfg(test)]
mod compiled_order2_oracle_tests {
    //! Gate: the two-output-allocation order-two lowering of [`FlexOuterPlan`]
    //! agrees to `1e-12` with the same plan evaluated through generic [`Jet2`]
    //! — value, gradient, and Hessian, channel-for-channel — over ≥2000 random
    //! inputs at several primary counts `p`.
    use super::*;

    fn xorshift(state: &mut u64) -> f64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        let u = (x >> 11) as f64 / ((1u64 << 53) as f64);
        2.0 * u - 1.0
    }

    fn rand_dense(p: usize, st: &mut u64) -> (f64, Vec<f64>, Vec<f64>) {
        let v = xorshift(st);
        let g: Vec<f64> = (0..p).map(|_| xorshift(st)).collect();
        let mut h = vec![0.0; p * p];
        for i in 0..p {
            for j in i..p {
                let x = xorshift(st);
                h[i * p + j] = x;
                h[j * p + i] = x;
            }
        }
        (v, g, h)
    }

    #[test]
    fn compiled_order2_row_nll_matches_generic_plan() {
        for &p in &[3usize, 6, 9, 12, 20] {
            let mut st = 0x5DEE_CE66_D9C7_F123u64 ^ (p as u64).wrapping_mul(0x9E3779B97F4A7C15);
            for _ in 0..2200 {
                let wi = (xorshift(&mut st) + 1.5).abs() + 0.1;
                let di = if xorshift(&mut st) > 0.0 { 1.0 } else { 0.0 };
                let surv0: [f64; 5] = std::array::from_fn(|_| xorshift(&mut st));
                let surv1: [f64; 5] = std::array::from_fn(|_| xorshift(&mut st));
                let (e0v, e0g, e0h) = rand_dense(p, &mut st);
                let (e1v, e1g, e1h) = rand_dense(p, &mut st);
                let (mut cv, cg, ch) = rand_dense(p, &mut st);
                cv = (cv + 2.0).abs() + 0.3;
                let (mut dv, dg, dh) = rand_dense(p, &mut st);
                dv = (dv + 2.0).abs() + 0.3;
                let q1v = (xorshift(&mut st) + 2.0).abs() + 0.2;
                let qd1v = (xorshift(&mut st) + 2.0).abs() + 0.2;
                let qax = p - 2;
                let qdax = p - 1;

                // Generic single source at Jet2.
                let g_out = flex_row_nll(
                    &Jet2::from_parts(e0v, &e0g, &e0h),
                    &Jet2::from_parts(e1v, &e1g, &e1h),
                    &Jet2::from_parts(cv, &cg, &ch),
                    &Jet2::from_parts(dv, &dg, &dh),
                    &Jet2::primary(q1v, qax, p),
                    &Jet2::primary(qd1v, qdax, p),
                    surv0,
                    surv1,
                    wi,
                    di,
                );

                // Production compiled lowering of that exact plan.
                let plan = FlexOuterPlan::new(cv, dv, qd1v, surv0, surv1, wi, di);
                let (f_value, f_gradient, f_hessian) = lower_flex_outer_plan_order2(
                    &plan,
                    FlexOrder2Inputs {
                        eta0: FlexOrder2View {
                            value: e0v,
                            gradient: ndarray::ArrayView1::from(&e0g),
                            hessian: ndarray::ArrayView2::from_shape((p, p), &e0h).unwrap(),
                        },
                        eta1: FlexOrder2View {
                            value: e1v,
                            gradient: ndarray::ArrayView1::from(&e1g),
                            hessian: ndarray::ArrayView2::from_shape((p, p), &e1h).unwrap(),
                        },
                        q1: (q1v, qax),
                        chi1: FlexOrder2View {
                            value: cv,
                            gradient: ndarray::ArrayView1::from(&cg),
                            hessian: ndarray::ArrayView2::from_shape((p, p), &ch).unwrap(),
                        },
                        d1: FlexOrder2View {
                            value: dv,
                            gradient: ndarray::ArrayView1::from(&dg),
                            hessian: ndarray::ArrayView2::from_shape((p, p), &dh).unwrap(),
                        },
                        qd1: (qd1v, qdax),
                    },
                    p,
                );

                let close = |left: f64, right: f64, channel: &str| {
                    let tolerance = 1e-12 * left.abs().max(right.abs()).max(1.0);
                    assert!(
                        (left - right).abs() <= tolerance,
                        "{channel} p={p}: generic={left:+.16e} compiled={right:+.16e}"
                    );
                };
                close(g_out.v, f_value, "value");
                for i in 0..p {
                    close(g_out.g[i], f_gradient[i], &format!("grad[{i}]"));
                }
                for k in 0..p * p {
                    close(g_out.h[k], f_hessian[k], &format!("hess[{k}]"));
                }
            }
        }
    }
}

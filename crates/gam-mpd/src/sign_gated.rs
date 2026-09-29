//! The sign-gated split of a SwiGLU block and its ReLU replacement contract (#2951).
//!
//! # The split
//!
//! A SwiGLU block reads `g = W_g x`, `u = W_u x` and writes `F(x) = W_d [s(g) ⊙ u]` with the
//! SiLU `s(t) = t σ(t)`. With `relu(t) = max(t, 0)`,
//!
//! ```text
//! s(t) = relu(t) + e(t),      e(t) = −|t| σ(−|t|),
//! F(x) = P(x) + R(x),         P = W_d [relu(g) ⊙ u],      R = W_d [e(g) ⊙ u].
//! ```
//!
//! Derivation: for `t ≥ 0`, `t σ(t) − t = −t (1 − σ(t)) = −t σ(−t)`; for `t < 0`, `relu(t) = 0`
//! and `t σ(t) = −|t| σ(−|t|)`. So the split is an identity, not an approximation. `P` is the
//! sign-gated bilinear law: on the set of inputs sharing one sign pattern `A = {n : g_n > 0}`
//! it is the bilinear form `W_d[:, A] (g_A ⊙ u_A)` of the two reads. `R` is the correction a
//! SiLU → ReLU replacement drops, exactly.
//!
//! # The correction's scalar
//!
//! * **Even and nonpositive.** `e(−t) = e(t) ≤ 0`, so `R(−x) = −R(x)`: `u` is odd in `x`.
//! * **Peak.** With `φ(a) = a σ(−a)` for `a ≥ 0`, `φ′(a) = σ(−a)(1 − a σ(a))`, and `a σ(a)`
//!   increases from `0` to `∞` (its derivative `σ(a) + a σ(a) σ(−a)` is positive), so `φ` has one
//!   critical point, a maximum, at `a σ(a) = 1`, i.e. `a = 1 + e^{−a}`. Writing `a = 1 + ω`
//!   gives `ω = e^{−1} e^{−ω}`, `ω e^ω = e^{−1}`, so `ω = W(e^{−1})` (principal Lambert W), and
//!   the peak value is `φ(a) = a − a σ(a) = a − 1`:
//!
//!   ```text
//!   sup_t |e(t)| = W(1/e) = 0.278464542761073795…,   attained at |t| = 1 + W(1/e).
//!   ```
//!
//!   [`correction_peak`] returns it rounded up.
//! * **Lipschitz ½.** For `a > 0`: where `a σ(a) ≤ 1`, `0 ≤ φ′(a) ≤ σ(−a) ≤ ½`; where
//!   `a σ(a) > 1`, `|φ′(a)| = σ(−a)(a σ(a) − 1) < a σ(−a) σ(a) ≤ φ(a) ≤ W(1/e) < ½`. The kink at
//!   `0` has slopes `∓½`. So `|e(t) − e(t′)| ≤ ½ |t − t′|`.
//! * **Decay.** `|e(t)| = |t| σ(−|t|) ≤ |t| e^{−|t|}`: a unit whose gate is far from zero
//!   contributes almost nothing to `R`.
//! * **By sign.** On an active unit (`t > 0`) `e(t) = −σ(−t) relu(t)`, a shrinkage of the
//!   law's own hidden value by at most one half; on an inactive unit (`t ≤ 0`) `e(t) = s(t)`,
//!   the SiLU's leak through a closed gate.
//!
//! # Bounds on `‖R‖` at one input
//!
//! Write `c = e(g) ⊙ u`, so `R = W_d c`.
//!
//! * **Executed enclosure** (the tightest). `R` is computed with a per-entry radius (below), so
//!   `‖R‖ ∈ [‖R̂‖ − ‖r‖, ‖R̂‖ + ‖r‖]` by the triangle inequality, each norm rounded outward. Its
//!   width is a few units of roundoff relative, so it is tight to that.
//! * **Operator bound.** `‖R‖ ≤ σ₁(W_d) ‖c‖`, with `σ₁` certified above by
//!   `state::spectral_norm_bounds`. It is the best bound that reads `c` only through `‖c‖`: for
//!   every value of `‖c‖`, the top right singular vector of `W_d` scaled to that norm attains
//!   it. Its excess at a token is exactly `σ₁ / (‖W_d c‖/‖c‖)`, the top gain over the realized
//!   one. A correction isotropic in the unit space has realized gain near the root-mean-square
//!   singular value `‖W_d‖_F / √I`, so the excess is near `σ₁ √I / ‖W_d‖_F`, which no
//!   restriction of `W_d` to a column subset closes (the subset keeps most of the spectrum).
//!   [`SignGatedSplit::down_gains`] reports both gains beside it so the excess is explained,
//!   not only measured.
//! * **Region bound (no input).** Behind an input RMSNorm with gain `γ`, `x = γ ⊙ z` with
//!   `‖z‖² = d · m/(m + ε) < d` (`m` the mean square of the residual), so for every residual
//!   `‖R‖ ≤ σ₁(W_d) · W(1/e) · ‖W_u diag γ‖₂ · √d`, from `|e| ≤ W(1/e)` entrywise and
//!   `‖u‖ ≤ ‖W_u diag γ‖₂ ‖z‖`. It is the only bound here that holds without an input; a
//!   post-norm block reads the raw residual, which no parameter bounds, and has none.
//!
//! # Replacement contract
//!
//! The block is a residual map `h ↦ h + w(F(x))`, `x = N_in(h)` behind an optional input
//! RMSNorm, `w` an optional output RMSNorm (`w = id` in a pre-norm layer, `N_ff` in OLMo 2's
//! post-norm layer). Swapping `s` for `relu` in this block alone changes its output by
//!
//! ```text
//! Δ(h) = w(P(x)) − w(F(x)),        Δ = −R when w = id,
//! ```
//!
//! exactly. [`SignGatedSplit::change`] states `sup_rows ‖Δ‖` over the declared rows through
//! `verify::exhaustive_supremum` (exact over the finite family, or a counterexample to a
//! declared tolerance). When the block's output residual is read by a final norm and an
//! unembedding (the last layer), [`replacement_readout`] executes both models' logits with
//! their radii and runs `verify::verify_family`, so the model-level KL of a last-layer
//! replacement is certified here; deeper layers need the rest of the network, which the caller
//! executes.
//!
//! # Bands
//!
//! Input rows are exact. Every read is `block::linear_read` (its `γ_d |W| |x̂|` band plus
//! `|W| r` for an input radius), each norm `canonical::LayerRmsNorm`, and the SiLU hidden rows
//! `canonical::swiglu_rows`. The two new hidden rows over gate and up rows within radii
//! `r_g`, `r_u`:
//!
//! * `a = relu(g) ⊙ u`: `|relu(g_e) u_e − relu(ĝ) û| ≤ r_g (|û| + r_u) + relu(ĝ) r_u` (relu is
//!   1-Lipschitz), plus the product's `γ_1 |â|` and one subnormal spacing.
//! * `c = e(g) ⊙ u`: `ê = fl(−|ĝ| σ̂(−|ĝ|))`, where gam-math's `σ` for a negative argument
//!   forms `exp` (one ulp, `2u` relative), one addition and one division, and the product one
//!   more rounding, so `|ê − e(ĝ)| ≤ γ_8 |ê| + (2|ĝ| + 2) 2^-1074` (the absolute part is an
//!   `exp` that underflows, as in `canonical::swiglu_rows`). Propagation uses the Lipschitz
//!   constant ½: `|e(g_e) u_e − e(ĝ) û| ≤ ½ r_g (|û| + r_u) + (|ê| + e_ê) r_u`, and the
//!   evaluation adds `e_ê |û| + γ_1 |ĉ| + 2^-1074`.
//!
//! The executed identity `F = P + R` is then checked entry by entry,
//! `|F̂ − P̂ − R̂| ≤ r_F + r_P + r_R + γ_2 (|F̂| + |P̂| + |R̂|)`; an execution that fails it is
//! refused, never returned.

use std::fmt;

use gam_linalg::roundoff::accumulation_growth;
use gam_math::special::logistic;
use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip, s};

use super::apply::ApplyError;
use super::attention::ProjectedRows;
use super::block::{SUBNORMAL_SPACING, down, linear_read, up};
use super::canonical::{CanonicalRefusal, LayerRmsNorm, swiglu_rows};
use super::gauge::SwigluUnits;
use super::secant::BandedMatrix;
use super::state::{SpectralNormBounds, StateError, entrywise_band_norm, spectral_norm_bounds};
use super::supports::{EvidenceStatus, EvidenceStatusError};
use super::verify::{FamilyVerification, RowValue, Tolerance, VerifyError, exhaustive_supremum, verify_family};

/// `W(1/e)` to the nearest binary64; [`correction_peak`] steps it up.
const CORRECTION_PEAK_NEAREST: f64 = 0.278_464_542_761_073_8;

/// `sup_t |e(t)| = W(1/e)`, rounded up (module docs, *The correction's scalar*).
pub fn correction_peak() -> f64 {
    up(CORRECTION_PEAK_NEAREST)
}

/// The gate magnitude `1 + W(1/e)` at which `|e|` peaks, to the nearest binary64.
pub const CORRECTION_PEAK_GATE: f64 = 1.278_464_542_761_073_7;

/// `e(t) = s(t) − relu(t) = −|t| σ(−|t|)`, evaluated without cancellation.
pub fn relu_correction(t: f64) -> f64 {
    let magnitude = t.abs();
    -magnitude * logistic(-magnitude)
}

/// A refused split.
#[derive(Debug)]
pub enum SignGatedError {
    /// A tensor or row block disagrees with the block.
    Shape {
        what: &'static str,
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// Rows hold `NaN` or `±∞`.
    NonFinite { what: &'static str },
    /// No rows were declared.
    EmptyRows,
    Apply(ApplyError),
    Canonical(Box<CanonicalRefusal>),
    State(StateError),
    Evidence(EvidenceStatusError),
    /// The executed identity `F = P + R` failed beyond its derived bands at `(row, column)` by
    /// `excess`: a defect in the evaluation, so nothing is returned.
    Identity { row: usize, column: usize, excess: f64 },
    /// The readout verification refused.
    Verify(String),
}

impl fmt::Display for SignGatedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape { what, expected, found } => {
                write!(formatter, "sign-gated split: {what} has shape {found:?}, expected {expected:?}")
            }
            Self::NonFinite { what } => write!(formatter, "sign-gated split: {what} holds a non-finite value"),
            Self::EmptyRows => write!(formatter, "sign-gated split: no rows were declared"),
            Self::Apply(error) => write!(formatter, "sign-gated split: {error}"),
            Self::Canonical(refusal) => write!(formatter, "sign-gated split: {refusal:?}"),
            Self::State(error) => write!(formatter, "sign-gated split: {error}"),
            Self::Evidence(error) => write!(formatter, "sign-gated split: {error}"),
            Self::Identity { row, column, excess } => write!(
                formatter,
                "sign-gated split: F = P + R fails at ({row}, {column}) by {excess} beyond its bands"
            ),
            Self::Verify(reason) => write!(formatter, "sign-gated readout: {reason}"),
        }
    }
}

impl std::error::Error for SignGatedError {}

impl From<ApplyError> for SignGatedError {
    fn from(error: ApplyError) -> Self {
        Self::Apply(error)
    }
}

impl From<CanonicalRefusal> for SignGatedError {
    fn from(refusal: CanonicalRefusal) -> Self {
        Self::Canonical(Box::new(refusal))
    }
}

impl From<StateError> for SignGatedError {
    fn from(error: StateError) -> Self {
        Self::State(error)
    }
}

impl From<EvidenceStatusError> for SignGatedError {
    fn from(error: EvidenceStatusError) -> Self {
        Self::Evidence(error)
    }
}

/// A residual SwiGLU block `h ↦ h + w(F(N_in(h)))` (module docs, *Replacement contract*).
#[derive(Clone, Debug)]
pub struct ResidualSwiglu {
    pub units: SwigluUnits,
    /// The RMSNorm the block reads its input through (pre-norm), or none (the MLP reads `h`).
    pub input_norm: Option<LayerRmsNorm>,
    /// The RMSNorm applied to the block's output before the residual add (post-norm), or none.
    pub output_norm: Option<LayerRmsNorm>,
}

/// Rows with a per-entry radius against their exact values.
#[derive(Clone, Debug, PartialEq)]
pub struct BandedRows {
    pub values: Array2<f64>,
    pub radius: Array2<f64>,
}

impl BandedRows {
    fn exact(values: ArrayView2<'_, f64>) -> Self {
        Self {
            values: values.to_owned(),
            radius: Array2::zeros(values.raw_dim()),
        }
    }

    fn view(&self) -> ProjectedRows<'_> {
        ProjectedRows {
            values: self.values.view(),
            radius: self.radius.view(),
        }
    }

    fn rows(&self, range: std::ops::Range<usize>) -> BandedRows {
        BandedRows {
            values: self.values.slice(s![range.clone(), ..]).to_owned(),
            radius: self.radius.slice(s![range, ..]).to_owned(),
        }
    }

    /// Per-row enclosures of the Euclidean norm of the exact rows.
    fn norms(&self) -> Vec<Interval> {
        self.values
            .rows()
            .into_iter()
            .zip(self.radius.rows())
            .map(|(values, radius)| {
                let reach = upper_norm(radius);
                Interval {
                    lower: down(lower_norm(values) - reach).max(0.0),
                    upper: up(upper_norm(values) + reach),
                }
            })
            .collect()
    }
}

/// A closed interval `[lower, upper]` holding an exact value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

/// `‖v‖₂` rounded up: every square and partial sum rounded to nearest then stepped up, and
/// the correctly rounded square root stepped up.
fn upper_norm(values: ArrayView1<'_, f64>) -> f64 {
    up(values.iter().fold(0.0, |sum, &value| up(sum + up(value * value))).sqrt())
}

/// `‖v‖₂` rounded down, the same way.
fn lower_norm(values: ArrayView1<'_, f64>) -> f64 {
    down(values.iter().fold(0.0, |sum, &value| down(sum + down(value * value))).sqrt())
}

/// Sums of nonnegative lower and upper ends, rounded outward.
fn interval_sum(terms: impl IntoIterator<Item = Interval>) -> Interval {
    terms.into_iter().fold(Interval { lower: 0.0, upper: 0.0 }, |sum, term| Interval {
        lower: down(sum.lower + term.lower),
        upper: up(sum.upper + term.upper),
    })
}

fn squared(interval: Interval) -> Interval {
    Interval {
        lower: down(interval.lower * interval.lower),
        upper: up(interval.upper * interval.upper),
    }
}

/// `1 − a/b` for nonnegative intervals, rounded outward; `None` when `b` may be zero.
fn one_minus_ratio(numerator: Interval, denominator: Interval) -> Option<Interval> {
    (denominator.lower > 0.0).then(|| Interval {
        lower: down(1.0 - up(numerator.upper / denominator.lower)),
        upper: up(1.0 - down(numerator.lower / denominator.upper)),
    })
}

/// `√(a/b)` for nonnegative intervals, rounded outward; `None` when `b` may be zero.
fn root_ratio(numerator: Interval, denominator: Interval) -> Option<Interval> {
    (denominator.lower > 0.0).then(|| Interval {
        lower: down(down(numerator.lower / denominator.upper).sqrt()),
        upper: up(up(numerator.upper / denominator.lower).sqrt()),
    })
}

/// The signs of one row's gates: certified positive, certified nonpositive, and undecided
/// (the radius reaches zero).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateSigns {
    pub active: usize,
    pub inactive: usize,
    pub undecided: usize,
}

/// `σ₁(W_d)` certified, and the root-mean-square singular value `‖W_d‖_F / √I` it is compared
/// with (module docs, *Operator bound*).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DownGains {
    pub operator: SpectralNormBounds,
    /// `‖W_d‖_F / √I`, computed (a descriptive scale, not a bound).
    pub root_mean_square: f64,
}

/// Aggregate statistics over the declared rows, each an interval holding the exact models'
/// value. `None` where a denominator's interval reaches zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitShares {
    /// `1 − Σ_t ‖R_t‖² / Σ_t ‖F_t − F̄‖²`: the variance of the block's output the law explains.
    pub law_explained_variance: Option<Interval>,
    /// `√(Σ_t ‖R_t‖² / Σ_t ‖F_t‖²)`.
    pub correction_over_native: Option<Interval>,
    /// `⟨P, F⟩ / (‖P‖ ‖F‖)` over all rows at once.
    pub law_native_cosine: Option<Interval>,
    /// `1 − Σ_t ‖Δ_t‖² / Σ_t ‖w(F_t) − mean‖²`: the same at the residual write. Equal to
    /// `law_explained_variance` without an output norm.
    pub write_explained_variance: Option<Interval>,
    /// `√(Σ_t ‖Δ_t‖² / Σ_t ‖w(F_t)‖²)`.
    pub change_over_write: Option<Interval>,
}

/// The executed sign-gated split of a residual SwiGLU block at declared rows.
#[derive(Clone, Debug)]
pub struct SignGatedSplit {
    /// The block's MLP output `F` (its native SiLU execution).
    pub native: BandedRows,
    /// The law `P = W_d[relu(g) ⊙ u]`.
    pub law: BandedRows,
    /// The correction `R = W_d[e(g) ⊙ u]`.
    pub correction: BandedRows,
    /// The residual writes `w(F)` and `w(P)`.
    pub native_write: BandedRows,
    pub replaced_write: BandedRows,
    /// The change `Δ = w(P) − w(F)` of a SiLU → ReLU replacement in this block.
    pub change: BandedRows,
    /// `max (|F̂ − P̂ − R̂| − allowance)`, at most zero (a positive value is refused).
    pub identity_excess: f64,
    /// Per-row enclosures of `‖F‖`, `‖P‖`, `‖R‖`, `‖w(F)‖` and `‖Δ‖`.
    pub native_norms: Vec<Interval>,
    pub law_norms: Vec<Interval>,
    pub correction_norms: Vec<Interval>,
    pub write_norms: Vec<Interval>,
    pub change_norms: Vec<Interval>,
    /// Per-row upper bounds `‖c‖` on the correction's hidden row `c = e(g) ⊙ u`.
    pub correction_hidden_norms: Vec<f64>,
    /// Per-row operator bounds `σ₁(W_d) ‖c‖ ≥ ‖R‖`.
    pub operator_bounds: Vec<f64>,
    pub down_gains: DownGains,
    /// `σ₁(W_d) W(1/e) ‖W_u diag γ‖₂ √d`, a bound on `‖R‖` at every residual, when the block
    /// reads through an input norm.
    pub region_bound: Option<f64>,
    /// Per-row gate signs.
    pub gate_signs: Vec<GateSigns>,
    /// Per row, the fewest units whose computed law hidden values `â_n²` reach 90% of `Σ â²`
    /// (descriptive, on the computed rows; zero for a zero row).
    pub law_units90: Vec<usize>,
    pub shares: SplitShares,
}

/// The finite family the change supremum is stated over: the declared rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChangeDomain {
    pub rows: usize,
}

impl SignGatedSplit {
    /// `sup_rows ‖Δ‖` over the declared rows: exact over the family with the largest row's
    /// enclosure as its error, or a counterexample to a declared tolerance on `‖Δ‖`.
    pub fn change_supremum(
        &self,
        tolerance: Option<f64>,
    ) -> Result<EvidenceStatus<usize, ChangeDomain>, SignGatedError> {
        let rows = self.change_norms.iter().enumerate().map(|(row, interval)| {
            let value = 0.5 * interval.lower + 0.5 * interval.upper;
            let error = up(up(interval.upper - value).max(up(value - interval.lower)));
            (row, RowValue::Resolved { value, numerical_error: error })
        });
        Ok(exhaustive_supremum(rows, tolerance, ChangeDomain { rows: self.change_norms.len() })?)
    }
}

fn expect_shape(what: &'static str, expected: (usize, usize), found: (usize, usize)) -> Result<(), SignGatedError> {
    if expected == found {
        Ok(())
    } else {
        Err(SignGatedError::Shape { what, expected, found })
    }
}

/// The law and correction hidden rows over banded gate and up rows (module docs, *Bands*).
fn split_hidden(gate: &BandedRows, up_rows: &BandedRows) -> (BandedRows, BandedRows) {
    let product = accumulation_growth(1);
    let evaluation = accumulation_growth(8);
    let shape = gate.values.raw_dim();
    let mut law = BandedRows { values: Array2::zeros(shape.clone()), radius: Array2::zeros(shape.clone()) };
    let mut correction = BandedRows { values: Array2::zeros(shape.clone()), radius: Array2::zeros(shape) };
    for (index, &g) in gate.values.indexed_iter() {
        let (g_radius, u, u_radius) = (gate.radius[index], up_rows.values[index], up_rows.radius[index]);
        let reach = up(u.abs() + u_radius);

        let relu = g.max(0.0);
        let a = relu * u;
        let propagated = up(up(g_radius * reach) + up(relu * u_radius));
        law.values[index] = a;
        law.radius[index] = up(up(propagated + up(product * a.abs())) + SUBNORMAL_SPACING);

        let e = relu_correction(g);
        let e_error = up(up(evaluation * e.abs()) + up(up(2.0 * g.abs() + 2.0) * SUBNORMAL_SPACING));
        let c = e * u;
        let propagated = up(up(0.5 * up(g_radius * reach)) + up(up(e.abs() + e_error) * u_radius));
        let evaluated = up(up(up(e_error * u.abs()) + up(product * c.abs())) + SUBNORMAL_SPACING);
        correction.values[index] = c;
        correction.radius[index] = up(propagated + evaluated);
    }
    (law, correction)
}

/// `W x` of banded rows, with its radius.
fn read(governor: &MemoryGovernor, weight: ArrayView2<'_, f64>, rows: &BandedRows) -> Result<BandedRows, SignGatedError> {
    let (values, radius) = linear_read(governor, weight, rows.view())?;
    Ok(BandedRows { values: values.to_owned(), radius })
}

/// `a − b` of banded rows: radius `r_a + r_b + γ_1 |fl(a − b)|`.
fn difference(first: &BandedRows, second: &BandedRows) -> BandedRows {
    let rounding = accumulation_growth(1);
    let values = &first.values - &second.values;
    let radius = Zip::from(&first.radius)
        .and(&second.radius)
        .and(&values)
        .map_collect(|&a, &b, &value| up(up(a + b) + up(rounding * value.abs())));
    BandedRows { values, radius }
}

/// `a + b` of banded rows: radius `r_a + r_b + γ_1 |fl(a + b)|`.
fn sum(first: &BandedRows, second: &BandedRows) -> BandedRows {
    let rounding = accumulation_growth(1);
    let values = &first.values + &second.values;
    let radius = Zip::from(&first.radius)
        .and(&second.radius)
        .and(&values)
        .map_collect(|&a, &b, &value| up(up(a + b) + up(rounding * value.abs())));
    BandedRows { values, radius }
}

/// The largest `|F̂ − P̂ − R̂| − (r_F + r_P + r_R + γ_2 (|F̂| + |P̂| + |R̂|))`, and where, or the
/// refusal when it is positive.
fn identity_excess(native: &BandedRows, law: &BandedRows, correction: &BandedRows) -> Result<f64, SignGatedError> {
    let growth = up(accumulation_growth(2));
    let mut largest = (f64::NEG_INFINITY, (0, 0));
    for (index, &f) in native.values.indexed_iter() {
        let (p, r) = (law.values[index], correction.values[index]);
        let gap = ((f - p) - r).abs();
        let magnitude = up(up(f.abs() + p.abs()) + r.abs());
        let allowance = up(up(up(native.radius[index] + law.radius[index]) + correction.radius[index]) + up(growth * magnitude));
        let excess = gap - allowance;
        if excess > largest.0 {
            largest = (excess, index);
        }
    }
    let (excess, (row, column)) = largest;
    if excess > 0.0 {
        return Err(SignGatedError::Identity { row, column, excess });
    }
    Ok(excess)
}

/// Enclosures of `Σ_t ‖X_t − X̄‖²` over banded rows. With `X_e = X̂ + δ`, `|δ| ≤ r`, the exact
/// centred row is `(X̂_t − X̄̂) + (δ_t − δ̄)` with `|δ_t − δ̄| ≤ r_t + r̄`; the mean's evaluation
/// adds `γ_(n+1) mean|X̂|` and the subtraction `γ_1` of its result.
fn centred_energy(rows: &BandedRows) -> Interval {
    let (count, width) = rows.values.dim();
    let zeros = || Array1::zeros(width);
    let mean = rows.values.mean_axis(Axis(0)).unwrap_or_else(zeros);
    let mean_magnitude = rows.values.mapv(f64::abs).mean_axis(Axis(0)).unwrap_or_else(zeros);
    let mean_radius = rows.radius.mean_axis(Axis(0)).unwrap_or_else(zeros);
    let growth = up(accumulation_growth(count + 1));
    let rounding = accumulation_growth(1);
    let centred = &rows.values - &mean.view().insert_axis(Axis(0));
    let mut radius = Array2::zeros(rows.values.raw_dim());
    for ((row, column), slot) in radius.indexed_iter_mut() {
        let spread = up(mean_radius[column] * up(1.0 + growth));
        let own = up(rows.radius[[row, column]] + up(rounding * centred[[row, column]].abs()));
        *slot = up(up(own + spread) + up(growth * mean_magnitude[column]));
    }
    interval_sum(BandedRows { values: centred, radius }.norms().into_iter().map(squared))
}

/// An enclosure of `⟨P, F⟩ / (‖P‖ ‖F‖)` over all entries.
fn cosine(first: &BandedRows, second: &BandedRows, first_norms: &[Interval], second_norms: &[Interval]) -> Option<Interval> {
    let entries = first.values.len();
    let mut dot = 0.0;
    let mut reach = 0.0;
    let mut magnitude = 0.0;
    Zip::from(&first.values)
        .and(&first.radius)
        .and(&second.values)
        .and(&second.radius)
        .for_each(|&a, &ra, &b, &rb| {
            dot += a * b;
            magnitude = up(magnitude + up(a.abs() * b.abs()));
            reach = up(reach + up(up(up(a.abs() * rb) + up(ra * b.abs())) + up(ra * rb)));
        });
    let error = up(reach + up(up(accumulation_growth(entries)) * magnitude));
    let numerator = Interval { lower: down(dot - error), upper: up(dot + error) };
    let first_total = interval_sum(first_norms.iter().copied().map(squared));
    let second_total = interval_sum(second_norms.iter().copied().map(squared));
    let lower_scale = down(down(first_total.lower.sqrt()) * down(second_total.lower.sqrt()));
    let upper_scale = up(up(first_total.upper.sqrt()) * up(second_total.upper.sqrt()));
    if !(lower_scale > 0.0) {
        return None;
    }
    let lower = if numerator.lower >= 0.0 {
        down(numerator.lower / upper_scale)
    } else {
        down(numerator.lower / lower_scale)
    };
    let upper = if numerator.upper >= 0.0 {
        up(numerator.upper / lower_scale)
    } else {
        up(numerator.upper / upper_scale)
    };
    Some(Interval { lower: lower.max(-1.0), upper: upper.min(1.0) })
}

/// The fewest units whose `a_n²` reach 90% of `Σ a²` on one computed row.
fn units90(row: ArrayView1<'_, f64>) -> usize {
    let mut energies: Vec<f64> = row.iter().map(|value| value * value).collect();
    let total: f64 = energies.iter().sum();
    if total == 0.0 {
        return 0;
    }
    energies.sort_by(|a, b| b.total_cmp(a));
    let mut covered = 0.0;
    for (count, energy) in energies.iter().enumerate() {
        covered += energy;
        if covered >= 0.9 * total {
            return count + 1;
        }
    }
    energies.len()
}

fn gate_signs(gate: &BandedRows) -> Vec<GateSigns> {
    gate.values
        .rows()
        .into_iter()
        .zip(gate.radius.rows())
        .map(|(values, radius)| {
            let mut signs = GateSigns { active: 0, inactive: 0, undecided: 0 };
            for (&g, &r) in values.iter().zip(radius.iter()) {
                if down(g - r) > 0.0 {
                    signs.active += 1;
                } else if up(g + r) <= 0.0 {
                    signs.inactive += 1;
                } else {
                    signs.undecided += 1;
                }
            }
            signs
        })
        .collect()
}

/// The block's residual write `w(y)` of banded MLP output rows.
fn write(block: &ResidualSwiglu, rows: &BandedRows) -> Result<BandedRows, SignGatedError> {
    match &block.output_norm {
        None => Ok(rows.clone()),
        Some(norm) => {
            let (values, radius) = norm.apply_rows(rows.view())?;
            Ok(BandedRows { values, radius })
        }
    }
}

impl ResidualSwiglu {
    fn check(&self, residual: ArrayView2<'_, f64>) -> Result<(), SignGatedError> {
        let width = self.units.gate().ncols();
        if residual.nrows() == 0 {
            return Err(SignGatedError::EmptyRows);
        }
        expect_shape("residual rows", (residual.nrows(), width), residual.dim())?;
        if residual.iter().any(|value| !value.is_finite()) {
            return Err(SignGatedError::NonFinite { what: "residual rows" });
        }
        if let Some(norm) = &self.input_norm {
            expect_shape("input norm gain", (width, 1), (norm.gain.len(), 1))?;
        }
        if let Some(norm) = &self.output_norm {
            let out = self.units.down().nrows();
            expect_shape("output norm gain", (out, 1), (norm.gain.len(), 1))?;
        }
        Ok(())
    }

    /// The MLP input rows `x = N_in(h)`, banded.
    fn mlp_input(&self, residual: ArrayView2<'_, f64>) -> Result<BandedRows, SignGatedError> {
        let exact = BandedRows::exact(residual);
        match &self.input_norm {
            None => Ok(exact),
            Some(norm) => {
                let (values, radius) = norm.apply_rows(exact.view())?;
                Ok(BandedRows { values, radius })
            }
        }
    }

    /// The residual rows after the block, native or replaced: `h + w`, banded.
    fn output_rows(&self, residual: ArrayView2<'_, f64>, written: &BandedRows) -> BandedRows {
        sum(&BandedRows::exact(residual), written)
    }

    /// The executed sign-gated split at exact residual rows `h` (module docs).
    pub fn split(&self, governor: &MemoryGovernor, residual: ArrayView2<'_, f64>) -> Result<SignGatedSplit, SignGatedError> {
        self.check(residual)?;
        let x = self.mlp_input(residual)?;
        let gate = read(governor, self.units.gate(), &x)?;
        let up_rows = read(governor, self.units.up(), &x)?;
        let (hidden_values, hidden_radius) = swiglu_rows(gate.view(), up_rows.view())?;
        let native_hidden = BandedRows { values: hidden_values, radius: hidden_radius };
        let (law_hidden, correction_hidden) = split_hidden(&gate, &up_rows);
        let down_weight = self.units.down();
        let native = read(governor, down_weight, &native_hidden)?;
        let law = read(governor, down_weight, &law_hidden)?;
        let correction = read(governor, down_weight, &correction_hidden)?;
        let identity = identity_excess(&native, &law, &correction)?;

        let native_write = write(self, &native)?;
        let replaced_write = write(self, &law)?;
        let change = match &self.output_norm {
            // P − F = −R exactly, and −R̂ carries R's radius.
            None => BandedRows { values: -&correction.values, radius: correction.radius.clone() },
            Some(_) => difference(&replaced_write, &native_write),
        };

        let native_norms = native.norms();
        let law_norms = law.norms();
        let correction_norms = correction.norms();
        let write_norms = native_write.norms();
        let change_norms = change.norms();
        let correction_hidden_norms: Vec<f64> = correction_hidden.norms().iter().map(|interval| interval.upper).collect();

        let units = down_weight.ncols();
        let operator = spectral_norm_bounds(governor, &down_weight.to_owned(), 0.0, "sign-gated split: down operator norm")?;
        let frobenius = down_weight.iter().map(|value| value * value).sum::<f64>().sqrt();
        let down_gains = DownGains { operator, root_mean_square: frobenius / (units as f64).sqrt() };
        let operator_bounds = correction_hidden_norms.iter().map(|&norm| up(operator.upper * norm)).collect();
        let region_bound = self.region_bound(governor, operator)?;

        let correction_energy = interval_sum(correction_norms.iter().copied().map(squared));
        let change_energy = interval_sum(change_norms.iter().copied().map(squared));
        let shares = SplitShares {
            law_explained_variance: one_minus_ratio(correction_energy, centred_energy(&native)),
            correction_over_native: root_ratio(correction_energy, interval_sum(native_norms.iter().copied().map(squared))),
            law_native_cosine: cosine(&law, &native, &law_norms, &native_norms),
            write_explained_variance: one_minus_ratio(change_energy, centred_energy(&native_write)),
            change_over_write: root_ratio(change_energy, interval_sum(write_norms.iter().copied().map(squared))),
        };
        Ok(SignGatedSplit {
            gate_signs: gate_signs(&gate),
            law_units90: law_hidden.values.rows().into_iter().map(units90).collect(),
            native,
            law,
            correction,
            native_write,
            replaced_write,
            change,
            identity_excess: identity,
            native_norms,
            law_norms,
            correction_norms,
            write_norms,
            change_norms,
            correction_hidden_norms,
            operator_bounds,
            down_gains,
            region_bound,
            shares,
        })
    }

    /// `σ₁(W_d) W(1/e) ‖W_u diag γ‖₂ √d (1 + ρ)` behind an input norm whose stored gain is within
    /// `ρ` relative of the represented one (module docs, *Region bound*). `W_u diag γ` is formed
    /// with one rounding per entry, `u |fl(W_u γ)|`, whose Frobenius norm bounds its spectral error.
    fn region_bound(&self, governor: &MemoryGovernor, down_norm: SpectralNormBounds) -> Result<Option<f64>, SignGatedError> {
        let Some(norm) = &self.input_norm else {
            return Ok(None);
        };
        let scaled = &self.units.up() * &norm.gain.view().insert_axis(Axis(0));
        let formation = up(entrywise_band_norm(scaled.mapv(|value| up(accumulation_growth(1) * value.abs()))));
        let up_norm = spectral_norm_bounds(governor, &scaled, formation, "sign-gated split: scaled up operator norm")?;
        let width = up((self.units.gate().ncols() as f64).sqrt());
        let gain_growth = up(1.0 + norm.gain_defect);
        Ok(Some(up(up(up(up(down_norm.upper * correction_peak()) * up_norm.upper) * width) * gain_growth)))
    }
}

/// A final RMSNorm and an unembedding that read a block's output residual as logits (the
/// block is the last layer).
#[derive(Clone, Debug)]
pub struct Readout {
    pub norm: LayerRmsNorm,
    /// `vocabulary × width`.
    pub unembedding: Array2<f64>,
}

/// The logits of the residual rows after a block that writes `written`, as an executable over
/// row ranges.
fn readout_logits<'a>(
    block: &'a ResidualSwiglu,
    residual: ArrayView2<'a, f64>,
    written: &'a BandedRows,
    readout: &'a Readout,
) -> impl Fn(&MemoryGovernor, &(), &std::ops::Range<usize>) -> Result<BandedMatrix, SignGatedError> + 'a {
    move |governor, _, range| {
        let out = block.output_rows(residual.slice(s![range.clone(), ..]), &written.rows(range.clone()));
        let (values, radius) = readout.norm.apply_rows(out.view())?;
        let logits = read(governor, readout.unembedding.view(), &BandedRows { values, radius })?;
        Ok(BandedMatrix { values: logits.values, bands: logits.radius })
    }
}

/// The model-level contract of a last-layer replacement: the native model's logits against
/// the replaced model's over the declared rows, in chunks of `chunk_rows` rows (the family's
/// inputs), through `verify::verify_family` (module docs, *Replacement contract*).
pub fn replacement_readout(
    governor: &MemoryGovernor,
    block: &ResidualSwiglu,
    residual: ArrayView2<'_, f64>,
    split: &SignGatedSplit,
    readout: &Readout,
    tolerance: &Tolerance,
    chunk_rows: usize,
) -> Result<FamilyVerification, SignGatedError> {
    block.check(residual)?;
    let width = residual.ncols();
    expect_shape("unembedding", (readout.unembedding.nrows(), width), readout.unembedding.dim())?;
    expect_shape("readout norm gain", (width, 1), (readout.norm.gain.len(), 1))?;
    expect_shape("split rows", residual.dim(), split.native_write.values.dim())?;
    let rows = residual.nrows();
    let chunk = chunk_rows.max(1);
    let family: Vec<std::ops::Range<usize>> = (0..rows).step_by(chunk).map(|start| start..(start + chunk).min(rows)).collect();
    verify_family(
        governor,
        &readout_logits(block, residual, &split.native_write, readout),
        &readout_logits(block, residual, &split.replaced_write, readout),
        &(),
        &family,
        tolerance,
    )
    .map_err(|error: VerifyError<SignGatedError, SignGatedError>| SignGatedError::Verify(error.to_string()))
}

#[cfg(test)]
#[path = "sign_gated_tests.rs"]
mod tests;

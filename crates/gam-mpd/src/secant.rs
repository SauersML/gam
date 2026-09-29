//! Exact two-endpoint finite-change ("secant") operators for the primitives a
//! mechanism program executes (#2951).
//!
//! A local Jacobian describes an infinitesimal change. A component ablation is a
//! finite change, and in saturated regimes the two disagree by orders of
//! magnitude. For each primitive below there is an operator `A(x, y)`, built from
//! the two endpoints only, with `f(y) − f(x) = A(x, y)(y − x)` EXACTLY (not to
//! first order), applied matrix-free. Each operator returns its values beside a
//! per-entry forward-error band derived from the arithmetic that produced them
//! (Higham's `γ_k`, through [`accumulation_growth`] and [`accumulation_band`]);
//! the band bounds the distance to the exact operator of the STORED endpoints,
//! first order in the unit roundoff `u` and then inflated by [`inflated`]. Input
//! rounding belongs to the caller. `exp`, `exp_m1` and `ln` are taken to return
//! within one unit in the last place (`2u` relatively), the contract
//! `gam_math::categorical` states for the same functions.
//!
//! # Softmax
//!
//! `p = softmax u`, `q = softmax v`, `δ = v − u`. With the logarithmic means
//! `ℓ_i = (q_i − p_i)/(log q_i − log p_i)` (`ℓ_i = p_i` when `p_i = q_i`) and
//! `s = Σ ℓ_i`,
//!
//! ```text
//! q − p = ℓ ⊙ δ − ℓ (ℓᵀδ)/s = ℓ ⊙ (δ − c·1),      c = ℓᵀδ / s.
//! ```
//!
//! Derivation: `log q_i − log p_i = δ_i − κ` with `κ = lse(v) − lse(u)`, so by the
//! definition of `ℓ`, `q_i − p_i = ℓ_i (δ_i − κ)`. Summing over `i`, `0 = Σ(q_i −
//! p_i) = ℓᵀδ − κ s`, so `κ = c`. The operator `diag ℓ − ℓℓᵀ/s` is symmetric and
//! positive semidefinite (`xᵀAx = s·Var_{ℓ/s}(x) ≥ 0`), annihilates constant
//! shifts, and tends to the softmax Jacobian `diag p − ppᵀ` as `v → u` (`ℓ → p`,
//! `s → 1`). `ℓ` is formed from the log-probabilities of
//! [`log_softmax_with_error`], never from differences of probabilities: with
//! `h = max(log p_i, log q_i)` and `d = |log q_i − log p_i|`,
//!
//! ```text
//! ℓ_i = e^h · E(d),       E(d) = (1 − e^{−d})/d = −expm1(−d)/d,  E(0) = 1.
//! ```
//!
//! `E` loses no accuracy near `d = 0` (`expm1` is relatively accurate there and
//! the quotient divides by the same `d` it was evaluated at), and
//! `|d log E/dd| = 1/d − 1/(e^d − 1) ≤ ½`, so an absolute error `e` in `d` is at
//! most the relative error `e/2` in `E`.
//!
//! # RMSNorm without gain
//!
//! `N(x) = x/r(x)`, `r(x) = √(ε + ‖x‖²/d)`, `ε > 0`. For endpoints `x`, `y` with
//! `m = (x + y)/2`, `Δ = y − x`, `r = r(x)`, `t = r(y)`:
//!
//! ```text
//! N(y) − N(x) = [½(1/r + 1/t) I − 2 m mᵀ/(d r t (r + t))] Δ.
//! ```
//!
//! Derivation: `y/t − x/r = m(1/t − 1/r) + (Δ/2)(1/t + 1/r)`, and
//! `1/t − 1/r = (r² − t²)/(r t (r + t))` with `r² − t² = (‖x‖² − ‖y‖²)/d =
//! −(x + y)ᵀ(y − x)/d = −2mᵀΔ/d`. A gain `g` multiplies on the left,
//! `g ⊙ N(y) − g ⊙ N(x) = g ⊙ (N(y) − N(x))` ([`BandedVector::gained`]).
//!
//! # Bilinear products
//!
//! With `L̄ = (L + L′)/2`, `ΔL = L′ − L` and the same for `R`,
//!
//! ```text
//! L′R′ − LR = L̄ ΔR + ΔL R̄
//! ```
//!
//! exactly: expanding the right side gives `½(L + L′)(R′ − R) + ½(L′ − L)(R + R′)
//! = ½(LR′ − LR + L′R′ − L′R) + ½(L′R + L′R′ − LR − LR′) = L′R′ − LR`. A weight
//! matrix times an input and attention weights times values are the same rule.
//!
//! # Scalar activations
//!
//! SiLU `x σ(x)` and the exact GELU `x Φ(x)` are both `x g(x)` for a gate `g`.
//! With `m = (a + b)/2`, `h = b − a`,
//!
//! ```text
//! b g(b) − a g(a) = h (g(a) + g(b))/2 + m (g(b) − g(a)),
//! [x g]_{a,b} = (g(a) + g(b))/2 + m [g]_{a,b},
//! ```
//!
//! (`m(g(b) − g(a)) + ½h(g(a) + g(b)) = ½(a + b)(g(b) − g(a)) + ½(b − a)(g(a) +
//! g(b)) = b g(b) − a g(a)`), where `[g]_{a,b}` is the divided difference, equal
//! to `g′(a)` at `a = b`. Only the gate's divided difference can cancel.
//!
//! * **Logistic.** For `lo ≤ hi`, `σ(hi) − σ(lo) = σ(hi) σ(−lo) (1 − e^{lo−hi})`
//!   (divide `σ(lo)σ(−hi)` by `σ(hi)σ(−lo)`: the ratio is `e^{lo−hi}`), so
//!   `[σ]_{lo,hi} = σ(hi) σ(−lo) E(hi − lo)` with the same `E` as the softmax. It
//!   is a product of positive factors, relatively accurate everywhere, and equals
//!   `σ(a)σ(−a) = σ′(a)` at `a = b`. No switch is needed.
//! * **Normal CDF.** No product form exists, so two evaluations are made and the
//!   one with the smaller DERIVED band is returned; that comparison is the switch.
//!   The direct quotient differences the tails that do not cancel (`Φ(hi) − Φ(lo)`
//!   for `lo < 0`, `Φ(−lo) − Φ(−hi)` for `lo ≥ 0`), with the table route's proven
//!   error ([`NORMAL_CDF_RELATIVE_ERROR`], [`NORMAL_CDF_UNDERFLOW_FLOOR`]); its band
//!   grows like `u/h`. The series integrates the Taylor expansion of `φ` about `m`
//!   over `[lo, hi]`, odd orders vanishing:
//!
//!   ```text
//!   [Φ]_{lo,hi} = ∫_{−½}^{½} φ(m + τh) dτ = φ(m) Σ_k He_{2k}(m) (h/2)^{2k}/(2k+1)!,
//!   ```
//!
//!   using `φ^{(n)} = (−1)^n He_n φ` and `∫_{−½}^{½} τ^{2k} dτ = (½)^{2k}/(2k+1)`.
//!   Its terms `Q_n = He_n(m)(h/2)^n/(n+1)!` follow the scaled Hermite recurrence
//!   `Q_{n+1} = (m h/2) Q_n/(n+2) − n (h/2)² Q_{n−1}/((n+1)(n+2))`, whose rounding is
//!   tracked term by term. Truncating after order `2K` leaves the Lagrange remainder
//!   `≤ sup_{[lo,hi]} |He_{2K+2} φ| (h/2)^{2K+2}/(2K+3)!`, bounded without
//!   constants by `|He_n(x)| ≤ Ĥ_n(|x|)`, the Hermite recurrence with every sign
//!   positive (`Ĥ_{n+1} = X Ĥ_n + n Ĥ_{n−1}`, nonnegative coefficients, so
//!   increasing in `X`), taken at the largest `|x|` of the interval, and `φ` at its
//!   point nearest zero. The series stops once its remainder is below its own
//!   rounding band, or once its rounding band already exceeds the direct band. The
//!   remainder's scaled sequence `P_n = Ĥ_n(X)(h/2)^n/(n+1)!` is positive and tends to
//!   zero superexponentially, so the loop ends; a `P` that overflows can never
//!   certify and ends it at once.

use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_band, accumulation_growth};
use gam_math::categorical::{CategoricalError, log_softmax_with_error};
use gam_math::probability::{
    NORMAL_CDF_RELATIVE_ERROR, NORMAL_CDF_UNDERFLOW_FLOOR, normal_cdf_and_pdf, normal_pdf_bounded,
};
use gam_math::roundoff::inflated;
use gam_math::special::logistic;
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use std::fmt;

/// A refused secant evaluation.
#[derive(Clone, Debug, PartialEq)]
pub enum SecantError {
    /// An operand holds `NaN` or `±∞`.
    NonFinite {
        operand: &'static str,
        index: usize,
        value: f64,
    },
    /// An operand has no entries.
    Empty { operand: &'static str },
    /// An operand's length or shape disagrees with its partner.
    Shape {
        operand: &'static str,
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// The RMSNorm offset must be finite and positive.
    NonPositiveEpsilon { epsilon: f64 },
    /// A difference or a sum of squares of finite operands overflowed.
    Overflow { quantity: &'static str },
    /// The categorical owner refused a logit vector.
    Categorical(CategoricalError),
}

impl fmt::Display for SecantError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite {
                operand,
                index,
                value,
            } => write!(
                formatter,
                "secant refused: {operand}[{index}] = {value} is not finite"
            ),
            Self::Empty { operand } => write!(formatter, "secant refused: {operand} is empty"),
            Self::Shape {
                operand,
                expected,
                found,
            } => write!(
                formatter,
                "secant refused: {operand} has shape {found:?}, its partner needs {expected:?}"
            ),
            Self::NonPositiveEpsilon { epsilon } => {
                write!(
                    formatter,
                    "secant refused: RMSNorm offset {epsilon} is not finite and positive"
                )
            }
            Self::Overflow { quantity } => {
                write!(formatter, "secant refused: {quantity} overflowed")
            }
            Self::Categorical(err) => write!(formatter, "secant refused: {err}"),
        }
    }
}

impl std::error::Error for SecantError {}

impl From<CategoricalError> for SecantError {
    fn from(err: CategoricalError) -> Self {
        Self::Categorical(err)
    }
}

/// Values with a per-entry bound on `|computed − exact|`.
#[derive(Clone, Debug, PartialEq)]
pub struct BandedVector {
    pub values: Vec<f64>,
    pub bands: Vec<f64>,
}

impl BandedVector {
    /// `g ⊙ values`: a gain applied on the left, one rounded product per entry.
    pub fn gained(&self, gain: &[f64]) -> Result<Self, SecantError> {
        require_finite("gain", gain.iter().copied())?;
        require_length("gain", self.values.len(), gain.len())?;
        let values: Vec<f64> = self
            .values
            .iter()
            .zip(gain)
            .map(|(value, g)| value * g)
            .collect();
        let bands = self
            .bands
            .iter()
            .zip(gain)
            .zip(&values)
            .map(|((band, g), value)| inflated(g.abs() * band + UNIT_ROUNDOFF * value.abs(), 1))
            .collect();
        Ok(Self { values, bands })
    }
}

/// A matrix with a per-entry bound on `|computed − exact|`.
#[derive(Clone, Debug, PartialEq)]
pub struct BandedMatrix {
    pub values: Array2<f64>,
    pub bands: Array2<f64>,
}

/// A scalar with a bound on `|computed − exact|`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandedScalar {
    pub value: f64,
    pub band: f64,
}

fn require_finite(
    operand: &'static str,
    values: impl IntoIterator<Item = f64>,
) -> Result<(), SecantError> {
    match values
        .into_iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        Some((index, value)) => Err(SecantError::NonFinite {
            operand,
            index,
            value,
        }),
        None => Ok(()),
    }
}

fn require_length(operand: &'static str, expected: usize, found: usize) -> Result<(), SecantError> {
    if expected == found {
        Ok(())
    } else {
        Err(SecantError::Shape {
            operand,
            expected: (expected, 1),
            found: (found, 1),
        })
    }
}

/// `fl(end − start)` per entry with its radius `u·|Δ_i|`, refusing an overflow.
fn rounded_differences(
    start: &[f64],
    end: &[f64],
    quantity: &'static str,
) -> Result<(Vec<f64>, Vec<f64>), SecantError> {
    let differences: Vec<f64> = start.iter().zip(end).map(|(a, b)| b - a).collect();
    if differences.iter().any(|value| !value.is_finite()) {
        return Err(SecantError::Overflow { quantity });
    }
    let radii = differences
        .iter()
        .map(|value| UNIT_ROUNDOFF * value.abs())
        .collect();
    Ok((differences, radii))
}

/// `E(d) = (1 − e^{−d})/d` for `d ≥ 0`, `E(0) = 1`: the secant slope of `1 − e^{−x}` over `[0, d]`. Relative error
/// at most `3u` (`expm1` two units, the quotient one).
fn exponential_secant(d: f64) -> f64 {
    if d == 0.0 { 1.0 } else { -(-d).exp_m1() / d }
}

/// The exact softmax secant operator between two logit vectors, held as its logarithmic means.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftmaxSecant {
    means: Vec<f64>,
    mean_bands: Vec<f64>,
    total: f64,
    total_band: f64,
}

impl SoftmaxSecant {
    /// The operator `A = diag ℓ − ℓℓᵀ/s` between `softmax(start)` and `softmax(end)`.
    ///
    /// Band of `ℓ_i`: the log-probabilities carry radii `r_p`, `r_q` from [`log_softmax_with_error`]. `e^h` takes the
    /// relative error `expm1(r_h) + 2u`, `d = |log q − log p|` the absolute error `r_p + r_q + u·d`, which moves `E` by
    /// at most half of it relatively, `E` itself `3u`, and the product `u`. `s` is a sum of positive terms: the
    /// bands of its terms plus `γ_{n−1}·s`.
    pub fn between(start: &[f64], end: &[f64]) -> Result<Self, SecantError> {
        if start.is_empty() {
            return Err(SecantError::Empty {
                operand: "start logits",
            });
        }
        require_length("end logits", start.len(), end.len())?;
        require_finite("start logits", start.iter().copied())?;
        require_finite("end logits", end.iter().copied())?;
        let (log_p, radius_p) = log_softmax_with_error(start)?;
        let (log_q, radius_q) = log_softmax_with_error(end)?;
        let u = UNIT_ROUNDOFF;
        let mut means = Vec::with_capacity(start.len());
        let mut mean_bands = Vec::with_capacity(start.len());
        for i in 0..start.len() {
            let (head, head_radius) = if log_p[i] >= log_q[i] {
                (log_p[i], radius_p[i])
            } else {
                (log_q[i], radius_q[i])
            };
            let gap = (log_q[i] - log_p[i]).abs();
            let mean = head.exp() * exponential_secant(gap);
            let gap_error = radius_p[i] + radius_q[i] + u * gap;
            let relative = head_radius.exp_m1() + 2.0 * u + 0.5 * gap_error + 3.0 * u + u;
            means.push(mean);
            mean_bands.push(inflated(mean * relative, 4));
        }
        let total: f64 = means.iter().sum();
        let total_band = inflated(
            mean_bands.iter().sum::<f64>() + accumulation_growth(means.len() - 1) * total,
            means.len(),
        );
        Ok(Self {
            means,
            mean_bands,
            total,
            total_band,
        })
    }

    /// The logarithmic means `ℓ_i`, each between `p_i` and `q_i`.
    pub fn logarithmic_means(&self) -> &[f64] {
        &self.means
    }

    /// `A δ = ℓ ⊙ (δ − c·1)`, `c = ℓᵀδ/s`, in `O(n)` for an exactly stored `δ`.
    pub fn apply(&self, direction: &[f64]) -> Result<BandedVector, SecantError> {
        require_length("direction", self.means.len(), direction.len())?;
        require_finite("direction", direction.iter().copied())?;
        Ok(self.apply_with_radii(direction, &vec![0.0; direction.len()]))
    }

    /// `A δ` for a `δ` known only within `radii` of exact, in `O(n)`: the bands also cover
    /// every `δ` in that box.
    pub fn apply_within(&self, direction: &[f64], radii: &[f64]) -> Result<BandedVector, SecantError> {
        require_length("direction", self.means.len(), direction.len())?;
        require_length("direction radii", self.means.len(), radii.len())?;
        require_finite("direction", direction.iter().copied())?;
        require_finite("direction radii", radii.iter().copied())?;
        Ok(self.apply_with_radii(direction, radii))
    }

    /// `A δ` for a `δ` known within `radii`. `ℓᵀδ` errs by `accumulation_band(n, Σ|ℓ_iδ_i|)` plus the propagated
    /// `Σ(band_i|δ_i| + ℓ_i r_i)`; `c` adds `|c|·band_s/s` and one quotient rounding; `ℓ_i(δ_i − c)` adds its factors'
    /// bands and two roundings.
    fn apply_with_radii(&self, direction: &[f64], radii: &[f64]) -> BandedVector {
        let u = UNIT_ROUNDOFF;
        let n = self.means.len();
        let mut dot = 0.0;
        let mut absolute = 0.0;
        let mut propagated = 0.0;
        for i in 0..n {
            dot += self.means[i] * direction[i];
            absolute += (self.means[i] * direction[i]).abs();
            propagated += self.mean_bands[i] * direction[i].abs() + self.means[i] * radii[i];
        }
        let dot_band = accumulation_band(n, absolute) + propagated;
        let centre = dot / self.total;
        let centre_band =
            (dot_band + centre.abs() * self.total_band) / self.total + u * centre.abs();
        let mut values = Vec::with_capacity(n);
        let mut bands = Vec::with_capacity(n);
        for i in 0..n {
            let offset = direction[i] - centre;
            let value = self.means[i] * offset;
            let offset_band = radii[i] + centre_band + u * offset.abs();
            values.push(value);
            bands.push(inflated(
                self.mean_bands[i] * offset.abs() + self.means[i] * offset_band + u * value.abs(),
                4,
            ));
        }
        BandedVector { values, bands }
    }
}

/// `softmax(end) − softmax(start)` through the exact secant operator, with `δ = fl(end − start)` and its rounding
/// carried as a radius.
pub fn softmax_change(start: &[f64], end: &[f64]) -> Result<BandedVector, SecantError> {
    let operator = SoftmaxSecant::between(start, end)?;
    let (direction, radii) = rounded_differences(start, end, "logit difference")?;
    Ok(operator.apply_with_radii(&direction, &radii))
}

/// The exact secant operator of gainless RMSNorm between two inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct RmsNormSecant {
    midpoint: Vec<f64>,
    difference: Vec<f64>,
    difference_radii: Vec<f64>,
    diagonal: f64,
    diagonal_relative: f64,
    rank_one: f64,
    rank_one_relative: f64,
}

impl RmsNormSecant {
    /// `B = ½(1/r + 1/t) I − 2 m mᵀ/(d r t (r + t))` between `start` and `end`.
    ///
    /// `r² = ε + Σx²/d` sums `d` rounded squares (`γ_d` relative, all terms positive), divides and adds (`2u`), so `r`
    /// carries `ρ_r = ½(γ_d + 2u) + u` relatively. The diagonal coefficient takes `max(ρ_r, ρ_t) + 2u` (two positive
    /// reciprocals and their sum) and the rank-one coefficient `ρ_r + ρ_t + max(ρ_r, ρ_t) + 5u` (a sum, three products
    /// and a quotient).
    pub fn between(start: &[f64], end: &[f64], epsilon: f64) -> Result<Self, SecantError> {
        if start.is_empty() {
            return Err(SecantError::Empty {
                operand: "start input",
            });
        }
        require_length("end input", start.len(), end.len())?;
        require_finite("start input", start.iter().copied())?;
        require_finite("end input", end.iter().copied())?;
        if !(epsilon.is_finite() && epsilon > 0.0) {
            return Err(SecantError::NonPositiveEpsilon { epsilon });
        }
        let u = UNIT_ROUNDOFF;
        let dimension = start.len() as f64;
        let rms = |values: &[f64]| {
            (epsilon + values.iter().map(|x| x * x).sum::<f64>() / dimension).sqrt()
        };
        let start_rms = rms(start);
        let end_rms = rms(end);
        if !(start_rms.is_finite() && end_rms.is_finite()) {
            return Err(SecantError::Overflow {
                quantity: "sum of squares",
            });
        }
        let rms_relative = 0.5 * (accumulation_growth(start.len()) + 2.0 * u) + u;
        let midpoint: Vec<f64> = start
            .iter()
            .zip(end)
            .map(|(a, b)| 0.5 * a + 0.5 * b)
            .collect();
        let (difference, difference_radii) = rounded_differences(start, end, "input difference")?;
        let diagonal = 0.5 * (1.0 / start_rms + 1.0 / end_rms);
        let rank_one = 2.0 / (dimension * start_rms * end_rms * (start_rms + end_rms));
        Ok(Self {
            midpoint,
            difference,
            difference_radii,
            diagonal,
            diagonal_relative: rms_relative + 2.0 * u,
            rank_one,
            rank_one_relative: 3.0 * rms_relative + 5.0 * u,
        })
    }

    /// `B Δ` for an exactly stored `Δ`, in `O(d)`.
    pub fn apply(&self, direction: &[f64]) -> Result<BandedVector, SecantError> {
        require_length("direction", self.midpoint.len(), direction.len())?;
        require_finite("direction", direction.iter().copied())?;
        Ok(self.apply_with_radii(direction, &vec![0.0; direction.len()]))
    }

    /// `B Δ` for a `Δ` known only within `radii` of exact, in `O(d)`: the bands also cover
    /// every `Δ` in that box.
    pub fn apply_within(&self, direction: &[f64], radii: &[f64]) -> Result<BandedVector, SecantError> {
        require_length("direction", self.midpoint.len(), direction.len())?;
        require_length("direction radii", self.midpoint.len(), radii.len())?;
        require_finite("direction", direction.iter().copied())?;
        require_finite("direction radii", radii.iter().copied())?;
        Ok(self.apply_with_radii(direction, radii))
    }

    /// `N(end) − N(start) = B·fl(end − start)`, the subtraction's rounding carried as a radius.
    pub fn change(&self) -> BandedVector {
        self.apply_with_radii(&self.difference, &self.difference_radii)
    }

    /// `B Δ = a Δ − (b·mᵀΔ) m`. `mᵀΔ` errs by `accumulation_band(d, Σ|m_iΔ_i|)` plus the midpoint's rounding `u|m_i|`
    /// and the radii; `w = b·mᵀΔ` adds `b`'s relative band and a product; each entry adds two products, a
    /// subtraction and the propagated bands.
    fn apply_with_radii(&self, direction: &[f64], radii: &[f64]) -> BandedVector {
        let u = UNIT_ROUNDOFF;
        let n = self.midpoint.len();
        let mut dot = 0.0;
        let mut absolute = 0.0;
        let mut propagated = 0.0;
        for i in 0..n {
            dot += self.midpoint[i] * direction[i];
            absolute += (self.midpoint[i] * direction[i]).abs();
            propagated += self.midpoint[i].abs() * (u * direction[i].abs() + radii[i]);
        }
        let dot_band = accumulation_band(n, absolute) + propagated;
        let weight = self.rank_one * dot;
        let weight_band = self.rank_one * dot_band + weight.abs() * (self.rank_one_relative + u);
        let mut values = Vec::with_capacity(n);
        let mut bands = Vec::with_capacity(n);
        for i in 0..n {
            let diagonal_part = self.diagonal * direction[i];
            let rank_one_part = weight * self.midpoint[i];
            let value = diagonal_part - rank_one_part;
            values.push(value);
            bands.push(inflated(
                self.diagonal * radii[i]
                    + diagonal_part.abs() * (self.diagonal_relative + u)
                    + rank_one_part.abs() * 2.0 * u
                    + self.midpoint[i].abs() * weight_band
                    + u * value.abs(),
                4,
            ));
        }
        BandedVector { values, bands }
    }
}

/// `L′R′ − LR = L̄ ΔR + ΔL R̄` for `L` (`p × q`) and `R` (`q × s`): a weight matrix times inputs, or attention
/// weights times values.
///
/// Band: each term of an entry's two length-`q` inner products has computed operands one rounding off (the
/// averages and differences), the inner products add `γ_q` and the final sum one more, so the entry errs by at most
/// `γ_{q+3}·S` with `S = |L̄||ΔR| + |ΔL||R̄|` (Higham, ASNA §3.1, for any summation order); `S` is itself a
/// computed sum of positive terms and is inflated by its own `γ_{q+1}`.
pub fn bilinear_change(
    left_start: ArrayView2<'_, f64>,
    left_end: ArrayView2<'_, f64>,
    right_start: ArrayView2<'_, f64>,
    right_end: ArrayView2<'_, f64>,
) -> Result<BandedMatrix, SecantError> {
    let expect = |operand, expected: (usize, usize), found: (usize, usize)| {
        if expected == found {
            Ok(())
        } else {
            Err(SecantError::Shape {
                operand,
                expected,
                found,
            })
        }
    };
    expect("left end", left_start.dim(), left_end.dim())?;
    expect("right end", right_start.dim(), right_end.dim())?;
    expect(
        "right start",
        (left_start.ncols(), right_start.ncols()),
        (right_start.nrows(), right_start.ncols()),
    )?;
    require_finite("left start", left_start.iter().copied())?;
    require_finite("left end", left_end.iter().copied())?;
    require_finite("right start", right_start.iter().copied())?;
    require_finite("right end", right_end.iter().copied())?;
    let left_mean = (&left_start * 0.5) + (&left_end * 0.5);
    let left_step = &left_end - &left_start;
    let right_mean = (&right_start * 0.5) + (&right_end * 0.5);
    let right_step = &right_end - &right_start;
    if left_step
        .iter()
        .chain(right_step.iter())
        .any(|value| !value.is_finite())
    {
        return Err(SecantError::Overflow {
            quantity: "bilinear endpoint difference",
        });
    }
    let values = left_mean.dot(&right_step) + left_step.dot(&right_mean);
    let absolute = left_mean.mapv(f64::abs).dot(&right_step.mapv(f64::abs))
        + left_step.mapv(f64::abs).dot(&right_mean.mapv(f64::abs));
    let inner = left_start.ncols();
    let growth = accumulation_growth(inner + 3);
    let bands = absolute.mapv(|sum| inflated(growth * sum, inner + 1));
    Ok(BandedMatrix { values, bands })
}

/// `W′h′ − Wh = W̄ Δh + ΔW h̄`: [`bilinear_change`] with `h` as one column.
pub fn bilinear_vector_change(
    weight_start: ArrayView2<'_, f64>,
    weight_end: ArrayView2<'_, f64>,
    input_start: ArrayView1<'_, f64>,
    input_end: ArrayView1<'_, f64>,
) -> Result<BandedVector, SecantError> {
    let change = bilinear_change(
        weight_start,
        weight_end,
        input_start.insert_axis(Axis(1)),
        input_end.insert_axis(Axis(1)),
    )?;
    Ok(BandedVector {
        values: change.values.column(0).to_vec(),
        bands: change.bands.column(0).to_vec(),
    })
}

/// A gated scalar activation `x g(x)` a mechanism program executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecantActivation {
    /// SiLU `x σ(x)`, gate [`logistic`].
    Silu,
    /// The exact GELU `x Φ(x)`, gate [`normal_cdf_and_pdf`].
    ExactGelu,
}

/// The divided difference `[σ(end) − σ(start)]/(end − start)`, and `σ′(start)` when the endpoints coincide, so
/// `σ(end) − σ(start) = slope·(end − start)` exactly.
///
/// Band: the gate average carries its gate's error and one rounding, `m = ½a + ½b` one rounding (it never
/// overflows), the gate's divided difference its own band, and the product and the final sum one rounding each.
pub fn activation_divided_difference(
    activation: SecantActivation,
    start: f64,
    end: f64,
) -> Result<BandedScalar, SecantError> {
    require_finite("start", [start])?;
    require_finite("end", [end])?;
    if !(end - start).is_finite() {
        return Err(SecantError::Overflow {
            quantity: "activation endpoint difference",
        });
    }
    let u = UNIT_ROUNDOFF;
    let (lo, hi) = if start <= end {
        (start, end)
    } else {
        (end, start)
    };
    let (gate_mean, gate_mean_band, gate_slope) = match activation {
        SecantActivation::Silu => {
            let mean = 0.5 * (logistic(start) + logistic(end));
            (mean, 7.0 * u * mean, logistic_divided_difference(lo, hi))
        }
        SecantActivation::ExactGelu => {
            let sum = normal_cdf_and_pdf(start).0 + normal_cdf_and_pdf(end).0;
            let mean = 0.5 * sum;
            let band = NORMAL_CDF_RELATIVE_ERROR * mean + NORMAL_CDF_UNDERFLOW_FLOOR + u * mean;
            (mean, band, normal_cdf_divided_difference(lo, hi))
        }
    };
    let midpoint = 0.5 * start + 0.5 * end;
    let product = midpoint * gate_slope.value;
    let value = gate_mean + product;
    let band = gate_mean_band
        + midpoint.abs() * gate_slope.band
        + 2.0 * u * product.abs()
        + u * value.abs();
    Ok(BandedScalar {
        value,
        band: inflated(band, 4),
    })
}

/// [`activation_divided_difference`] entry by entry.
pub fn activation_divided_differences(
    activation: SecantActivation,
    start: &[f64],
    end: &[f64],
) -> Result<BandedVector, SecantError> {
    require_length("end", start.len(), end.len())?;
    require_finite("start", start.iter().copied())?;
    require_finite("end", end.iter().copied())?;
    let mut values = Vec::with_capacity(start.len());
    let mut bands = Vec::with_capacity(start.len());
    for (a, b) in start.iter().zip(end) {
        let slope = activation_divided_difference(activation, *a, *b)?;
        values.push(slope.value);
        bands.push(slope.band);
    }
    Ok(BandedVector { values, bands })
}

/// `[σ]_{lo,hi} = σ(hi) σ(−lo) E(hi − lo)`. [`logistic`] errs by at most `6u` relatively (an exponential of two
/// units, a sum and a quotient, and in the negative branch a numerator of two units), `E` by `3u` plus the width's
/// rounding `u·h` moved by at most half, and the two products by `2u`.
fn logistic_divided_difference(lo: f64, hi: f64) -> BandedScalar {
    let u = UNIT_ROUNDOFF;
    let width = hi - lo;
    let value = logistic(hi) * logistic(-lo) * exponential_secant(width);
    let relative = 6.0 * u + 6.0 * u + 3.0 * u + 0.5 * u * width + 2.0 * u;
    BandedScalar {
        value,
        band: inflated(value * relative, 2),
    }
}

/// `[Φ]_{lo,hi}` by whichever of the direct quotient and the Taylor series carries the smaller derived band (module
/// documentation).
fn normal_cdf_divided_difference(lo: f64, hi: f64) -> BandedScalar {
    let direct = normal_cdf_direct_quotient(lo, hi);
    let series = normal_cdf_series(lo, hi, direct.band);
    if series.band <= direct.band {
        series
    } else {
        direct
    }
}

/// `(Φ(hi) − Φ(lo))/h` from the non-cancelling tails; no band at `h = 0`. Each `Φ` errs by
/// `NORMAL_CDF_RELATIVE_ERROR·Φ + NORMAL_CDF_UNDERFLOW_FLOOR`, the difference by one rounding, and the quotient by
/// one more plus the width's rounding.
fn normal_cdf_direct_quotient(lo: f64, hi: f64) -> BandedScalar {
    let width = hi - lo;
    if width == 0.0 {
        return BandedScalar {
            value: f64::NAN,
            band: f64::INFINITY,
        };
    }
    let (upper, lower) = if lo >= 0.0 {
        (normal_cdf_and_pdf(-lo).0, normal_cdf_and_pdf(-hi).0)
    } else {
        (normal_cdf_and_pdf(hi).0, normal_cdf_and_pdf(lo).0)
    };
    let difference = upper - lower;
    let difference_band = NORMAL_CDF_RELATIVE_ERROR * (upper + lower)
        + 2.0 * NORMAL_CDF_UNDERFLOW_FLOOR
        + UNIT_ROUNDOFF * difference.abs();
    let value = difference / width;
    BandedScalar {
        value,
        band: inflated(
            difference_band / width + 2.0 * UNIT_ROUNDOFF * value.abs(),
            2,
        ),
    }
}

/// The Taylor series of the module documentation, stopped by its derived remainder, or once its rounding band passes
/// `competing` (then its band is returned and the caller keeps the other route).
fn normal_cdf_series(lo: f64, hi: f64, competing: f64) -> BandedScalar {
    let u = UNIT_ROUNDOFF;
    let midpoint = 0.5 * lo + 0.5 * hi;
    let half = 0.5 * (hi - lo);
    let half_square = half * half;
    // The series expands about the computed `midpoint` over the computed half-width, so the remainder's supremum is
    // over `[midpoint − half, midpoint + half]`: `reach` rounds up and `nearest` down past their one rounding each.
    let reach = (midpoint.abs() + half) * (1.0 + 2.0 * u);
    let nearest = ((midpoint.abs() - half) * (1.0 - 2.0 * u)).max(0.0);
    let density_upper = |x: f64| {
        let (value, error) = normal_pdf_bounded(x);
        value + error
    };
    let (density, density_error) = normal_pdf_bounded(midpoint);
    let nearest_density = density_upper(nearest);
    // sup |φ′| = sup |x| φ(x) on [lo, hi]: |x|φ(x) rises on [0, 1] and falls after.
    let slope_ceiling = if nearest >= 1.0 {
        nearest * nearest_density
    } else if reach <= 1.0 {
        reach * density_upper(reach)
    } else {
        density_upper(1.0)
    };
    // `midpoint` and `2·half` each round once; ∂/∂m and ∂/∂h of the divided difference are at most sup|φ′| and
    // sup|φ′|/4.
    let sensitivity = slope_ceiling * (u * midpoint.abs() + 0.5 * u * half);

    let mid_half = midpoint * half;
    let reach_half = reach * half;
    // Q_{n-1}, Q_n with running absolute errors, and P_{n-1}, P_n with their relative bound.
    let (mut q_prev, mut q_curr) = (1.0, 0.5 * mid_half);
    let (mut q_prev_error, mut q_curr_error) = (0.0, u * q_curr.abs());
    let (mut p_prev, mut p_curr) = (1.0, 0.5 * reach_half);
    let mut order = 1_usize;
    let mut sum = 1.0_f64;
    let mut absolute = 1.0_f64;
    let mut term_error = 0.0_f64;
    let mut terms = 1_usize;
    loop {
        // Advance both recurrences from order `order` (odd) to `order + 1` (even).
        let n = order as f64;
        let leading = mid_half * q_curr / (n + 2.0);
        let trailing = n * half_square * q_prev / ((n + 1.0) * (n + 2.0));
        let q_next = leading - trailing;
        let q_next_error = (mid_half / (n + 2.0)).abs() * q_curr_error
            + (n * half_square / ((n + 1.0) * (n + 2.0))) * q_prev_error
            + 3.0 * u * leading.abs()
            + 4.0 * u * trailing.abs()
            + u * q_next.abs();
        let p_next =
            reach_half * p_curr / (n + 2.0) + n * half_square * p_prev / ((n + 1.0) * (n + 2.0));
        // Each step of the positive recurrence adds at most 4u to `P`'s relative error (3u and 5u branches, one sum).
        let remainder = inflated(p_next, 4 * (order + 1)) * nearest_density;
        let sum_error = term_error + accumulation_growth(terms) * absolute;
        let rounding = density * sum_error
            + sum.abs() * density_error
            + u * (density * sum).abs()
            + sensitivity;
        if !p_next.is_finite() {
            return BandedScalar {
                value: density * sum,
                band: f64::INFINITY,
            };
        }
        if remainder <= rounding || rounding >= competing {
            return BandedScalar {
                value: density * sum,
                band: inflated(rounding + remainder, 3),
            };
        }
        sum += q_next;
        absolute += q_next.abs();
        term_error += q_next_error;
        terms += 1;
        // One more step to the next odd order.
        let n = (order + 1) as f64;
        let leading_odd = mid_half * q_next / (n + 2.0);
        let trailing_odd = n * half_square * q_curr / ((n + 1.0) * (n + 2.0));
        let q_odd = leading_odd - trailing_odd;
        let q_odd_error = (mid_half / (n + 2.0)).abs() * q_next_error
            + (n * half_square / ((n + 1.0) * (n + 2.0))) * q_curr_error
            + 3.0 * u * leading_odd.abs()
            + 4.0 * u * trailing_odd.abs()
            + u * q_odd.abs();
        let p_odd =
            reach_half * p_next / (n + 2.0) + n * half_square * p_curr / ((n + 1.0) * (n + 2.0));
        q_prev = q_next;
        q_prev_error = q_next_error;
        q_curr = q_odd;
        q_curr_error = q_odd_error;
        p_prev = p_next;
        p_curr = p_odd;
        order += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_math::gaussian_gated::silu_derivatives;
    use ndarray::{Array1, array};

    const U: f64 = UNIT_ROUNDOFF;

    fn logits(n: usize, phase: f64, scale: f64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                scale * ((i as f64 + 1.0) * (1.7 + phase)).sin()
                    + 0.4 * ((i * i) as f64 * 0.31 + phase).cos()
            })
            .collect()
    }

    /// `softmax(z)` from the owner's log-softmax, each entry within `p·(expm1 r + 2u)` of the exact value.
    fn probabilities(logits: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (log_p, radii) = log_softmax_with_error(logits).expect("finite logits");
        let values: Vec<f64> = log_p.iter().map(|value| value.exp()).collect();
        let bands = values
            .iter()
            .zip(&radii)
            .map(|(p, r)| inflated(p * (r.exp_m1() + 2.0 * U), 2))
            .collect();
        (values, bands)
    }

    /// `q − p` evaluated directly, with its band.
    fn direct_softmax_change(start: &[f64], end: &[f64]) -> BandedVector {
        let (p, p_bands) = probabilities(start);
        let (q, q_bands) = probabilities(end);
        let values: Vec<f64> = q.iter().zip(&p).map(|(q, p)| q - p).collect();
        let bands = (0..p.len())
            .map(|i| inflated(p_bands[i] + q_bands[i] + U * values[i].abs(), 1))
            .collect();
        BandedVector { values, bands }
    }

    /// Whether `candidate` lies within `claimed + reference.bands` of `reference` at every entry.
    fn agrees(candidate: &[f64], claimed: &[f64], reference: &BandedVector) -> bool {
        (0..candidate.len())
            .all(|i| (candidate[i] - reference.values[i]).abs() <= claimed[i] + reference.bands[i])
    }

    #[test]
    fn softmax_secant_reproduces_the_finite_change_and_the_local_forms_do_not() {
        for (phase, step) in [(0.1, 0.9), (0.7, 2.5), (1.3, 6.0)] {
            let start = logits(7, phase, 2.0);
            let end: Vec<f64> = start
                .iter()
                .enumerate()
                .map(|(i, z)| z + step * ((i as f64) * 0.83 + phase).cos())
                .collect();
            let secant = softmax_change(&start, &end).expect("finite logits");
            let direct = direct_softmax_change(&start, &end);
            assert!(
                agrees(&secant.values, &secant.bands, &direct),
                "secant {:?} ± {:?} misses direct {:?} ± {:?}",
                secant.values,
                secant.bands,
                direct.values,
                direct.bands
            );
            // Positive controls, held to the secant's own band: the local Jacobian at the start, and the operator
            // without its rank-one centring.
            let (p, _) = probabilities(&start);
            let delta: Vec<f64> = start.iter().zip(&end).map(|(a, b)| b - a).collect();
            let mean: f64 = p.iter().zip(&delta).map(|(p, d)| p * d).sum();
            let jacobian: Vec<f64> = p.iter().zip(&delta).map(|(p, d)| p * (d - mean)).collect();
            assert!(
                !agrees(&jacobian, &secant.bands, &direct),
                "the local Jacobian passed the secant band"
            );
            let operator = SoftmaxSecant::between(&start, &end).expect("finite logits");
            let diagonal_only: Vec<f64> = operator
                .logarithmic_means()
                .iter()
                .zip(&delta)
                .map(|(l, d)| l * d)
                .collect();
            assert!(
                !agrees(&diagonal_only, &secant.bands, &direct),
                "dropping the rank-one term passed"
            );
            // Each logarithmic mean lies between its two probabilities.
            let (q, _) = probabilities(&end);
            for i in 0..p.len() {
                let l = operator.logarithmic_means()[i];
                let band = operator.mean_bands[i];
                assert!(l + band >= p[i].min(q[i]) * (1.0 - 4.0 * U));
                assert!(l - band <= p[i].max(q[i]) * (1.0 + 4.0 * U));
            }
        }
    }

    #[test]
    fn softmax_secant_annihilates_constant_shifts() {
        let start = logits(6, 0.4, 1.5);
        let end = logits(6, 1.1, 3.0);
        let operator = SoftmaxSecant::between(&start, &end).expect("finite logits");
        let constant = operator.apply(&[2.75; 6]).expect("finite direction");
        for (value, band) in constant.values.iter().zip(&constant.bands) {
            assert!(
                value.abs() <= *band,
                "a constant shift moved the output by {value:e} > {band:e}"
            );
        }
        // Softmax is shift invariant, and so is the secant change.
        let shifted_end: Vec<f64> = end.iter().map(|z| z + 13.0).collect();
        let direct = direct_softmax_change(&start, &end);
        let shifted = softmax_change(&start, &shifted_end).expect("finite logits");
        assert!(agrees(&shifted.values, &shifted.bands, &direct));
    }

    #[test]
    fn softmax_secant_is_symmetric_and_positive_semidefinite() {
        let start = logits(8, 0.2, 2.5);
        let end = logits(8, 0.9, 1.0);
        let operator = SoftmaxSecant::between(&start, &end).expect("finite logits");
        let x = logits(8, 2.1, 1.0);
        let y = logits(8, 3.4, 1.7);
        let ax = operator.apply(&x).expect("finite");
        let ay = operator.apply(&y).expect("finite");
        let pairing = |left: &[f64], right: &BandedVector| {
            let value: f64 = left.iter().zip(&right.values).map(|(a, b)| a * b).sum();
            let absolute: f64 = left
                .iter()
                .zip(&right.values)
                .map(|(a, b)| (a * b).abs())
                .sum();
            let propagated: f64 = left
                .iter()
                .zip(&right.bands)
                .map(|(a, b)| a.abs() * b)
                .sum();
            (
                value,
                inflated(accumulation_band(left.len(), absolute) + propagated, 1),
            )
        };
        let (xay, xay_band) = pairing(&x, &ay);
        let (yax, yax_band) = pairing(&y, &ax);
        assert!(
            (xay - yax).abs() <= xay_band + yax_band,
            "xᵀAy = {xay:e}, yᵀAx = {yax:e}"
        );
        for vector in [&x, &y] {
            let (quadratic, band) = pairing(vector, &operator.apply(vector).expect("finite"));
            assert!(quadratic >= -band, "xᵀAx = {quadratic:e} below −{band:e}");
            assert!(
                quadratic > band,
                "a non-constant direction must have positive curvature"
            );
        }
    }

    #[test]
    fn softmax_secant_tends_to_the_softmax_jacobian() {
        let start = logits(5, 0.6, 1.8);
        let direction = logits(5, 1.9, 1.0);
        let (p, p_bands) = probabilities(&start);
        let sup = direction.iter().fold(0.0_f64, |m, w| m.max(w.abs()));
        let mean: f64 = p.iter().zip(&direction).map(|(p, w)| p * w).sum();
        let jacobian: Vec<f64> = p
            .iter()
            .zip(&direction)
            .map(|(p, w)| p * (w - mean))
            .collect();
        let band_sum: f64 = p_bands.iter().sum();
        let mut previous_gap = f64::INFINITY;
        for step in [1e-1, 1e-3, 1e-5] {
            let end: Vec<f64> = start
                .iter()
                .zip(&direction)
                .map(|(z, w)| z + step * w)
                .collect();
            let action = SoftmaxSecant::between(&start, &end)
                .expect("finite")
                .apply(&direction)
                .expect("finite");
            let change = direct_softmax_change(&start, &end);
            // With ℓ = p + e, |e_i| ≤ |q_i − p_i|, and E = ‖q − p‖₁, the two operators differ on w by at most
            // ‖w‖∞(2E + 2E/(1 − E)) at every entry.
            let total: f64 = change
                .values
                .iter()
                .zip(&change.bands)
                .map(|(v, b)| v.abs() + b)
                .sum();
            let allowance = sup * (2.0 * total + 2.0 * total / (1.0 - total));
            let mut gap = 0.0_f64;
            for i in 0..p.len() {
                let jacobian_band = inflated(
                    2.0 * sup * (p_bands[i] + p[i] * band_sum + accumulation_growth(8) * p[i]),
                    2,
                );
                let difference = (action.values[i] - jacobian[i]).abs();
                assert!(difference <= allowance + action.bands[i] + jacobian_band);
                gap = gap.max(difference);
            }
            assert!(
                gap < previous_gap,
                "the secant did not approach the Jacobian"
            );
            previous_gap = gap;
        }
    }

    #[test]
    fn saturated_softmax_change_is_captured_by_the_secant_and_missed_by_the_jacobian() {
        let start = [12.0, -12.0];
        let end = [-12.0, 12.0];
        let secant = softmax_change(&start, &end).expect("finite");
        // q − p = (−tanh 12, tanh 12) exactly.
        let exact = 12.0_f64.tanh();
        for (value, (band, sign)) in secant
            .values
            .iter()
            .zip(secant.bands.iter().zip([-1.0, 1.0]))
        {
            assert!(
                (value - sign * exact).abs() <= band + U * exact,
                "{value} vs {}",
                sign * exact
            );
        }
        let norm = secant.values.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - std::f64::consts::SQRT_2 * exact).abs() <= 4.0 * U * norm);
        // The local Jacobian at the start: 48·√2·p₁p₂ = 12√2/cosh²12 ≈ 2.6e-9.
        let (p, _) = probabilities(&start);
        let delta = [-24.0, 24.0];
        // (diag p − ppᵀ)δ = p₁p₂(δ₁ − δ₂)(1, −1) for two categories, formed without the cancelling δ − pᵀδ.
        let coupling = p[0] * p[1] * (delta[0] - delta[1]);
        let jacobian = [coupling, -coupling];
        let jacobian_norm = (jacobian[0] * jacobian[0] + jacobian[1] * jacobian[1]).sqrt();
        let predicted = 12.0 * std::f64::consts::SQRT_2 / 12.0_f64.cosh().powi(2);
        assert!(
            (jacobian_norm - predicted).abs() <= 1e-6 * predicted,
            "{jacobian_norm:e} vs {predicted:e}"
        );
        assert!(jacobian_norm < 3e-9 && norm > 1.414);
        let direct = direct_softmax_change(&start, &end);
        assert!(!agrees(&jacobian, &secant.bands, &direct));
    }

    /// `x/r(x)` directly, each entry within `|N_i|(ρ_r + u)`.
    fn direct_norm(x: &[f64], epsilon: f64) -> (Vec<f64>, Vec<f64>) {
        let d = x.len() as f64;
        let r = (epsilon + x.iter().map(|v| v * v).sum::<f64>() / d).sqrt();
        let relative = 0.5 * (accumulation_growth(x.len()) + 2.0 * U) + 2.0 * U;
        let values: Vec<f64> = x.iter().map(|v| v / r).collect();
        let bands = values
            .iter()
            .map(|v| inflated(v.abs() * relative, 1))
            .collect();
        (values, bands)
    }

    fn direct_norm_change(x: &[f64], y: &[f64], epsilon: f64) -> BandedVector {
        let (nx, bx) = direct_norm(x, epsilon);
        let (ny, by) = direct_norm(y, epsilon);
        let values: Vec<f64> = ny.iter().zip(&nx).map(|(a, b)| a - b).collect();
        let bands = (0..values.len())
            .map(|i| inflated(bx[i] + by[i] + U * values[i].abs(), 1))
            .collect();
        BandedVector { values, bands }
    }

    #[test]
    fn rms_norm_secant_reproduces_the_finite_change_and_the_local_forms_do_not() {
        let epsilon = 1e-6;
        let x = logits(9, 0.3, 1.2);
        let reflected: Vec<f64> = x.iter().map(|v| -v).collect();
        // The reflection has midpoint 0, where the rank-one term vanishes and only the Jacobian control applies.
        for (y, has_rank_one) in [(logits(9, 1.4, 3.0), true), (reflected, false)] {
            let operator = RmsNormSecant::between(&x, &y, epsilon).expect("finite");
            let change = operator.change();
            let direct = direct_norm_change(&x, &y, epsilon);
            assert!(
                agrees(&change.values, &change.bands, &direct),
                "{change:?} vs {direct:?}"
            );
            let delta: Vec<f64> = x.iter().zip(&y).map(|(a, b)| b - a).collect();
            let diagonal_only: Vec<f64> = delta.iter().map(|d| operator.diagonal * d).collect();
            assert_eq!(
                !agrees(&diagonal_only, &change.bands, &direct),
                has_rank_one,
                "dropping the rank-one term changed the verdict"
            );
            let d = x.len() as f64;
            let r2 = epsilon + x.iter().map(|v| v * v).sum::<f64>() / d;
            let projection: f64 = x.iter().zip(&delta).map(|(a, b)| a * b).sum::<f64>() / (d * r2);
            let jacobian: Vec<f64> = x
                .iter()
                .zip(&delta)
                .map(|(a, b)| (b - a * projection) / r2.sqrt())
                .collect();
            assert!(
                !agrees(&jacobian, &change.bands, &direct),
                "the local Jacobian passed"
            );
            // A gain multiplies on the left.
            let gain = logits(9, 2.2, 0.8);
            let gained = change.gained(&gain).expect("finite gain");
            let gained_direct = BandedVector {
                values: direct
                    .values
                    .iter()
                    .zip(&gain)
                    .map(|(v, g)| v * g)
                    .collect(),
                bands: direct
                    .bands
                    .iter()
                    .zip(&gain)
                    .zip(&direct.values)
                    .map(|((b, g), v)| g.abs() * b + U * (v * g).abs())
                    .collect(),
            };
            assert!(agrees(&gained.values, &gained.bands, &gained_direct));
        }
    }

    #[test]
    fn rms_norm_secant_applies_to_any_direction_linearly() {
        let x = logits(5, 0.8, 1.0);
        let y = logits(5, 1.8, 2.0);
        let operator = RmsNormSecant::between(&x, &y, 1e-5).expect("finite");
        let a = logits(5, 0.1, 1.0);
        let b = logits(5, 0.5, 1.0);
        let sum: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p + q).collect();
        let (ba, bb, bs) = (
            operator.apply(&a).expect("finite"),
            operator.apply(&b).expect("finite"),
            operator.apply(&sum).expect("finite"),
        );
        let norm_bound = operator.diagonal
            + operator.rank_one * operator.midpoint.iter().map(|m| m * m).sum::<f64>();
        for i in 0..5 {
            // `sum` rounds once per entry, which B moves by at most ‖B‖₂·u‖sum‖₂.
            let rounding = norm_bound * U * sum.iter().map(|s| s * s).sum::<f64>().sqrt();
            let slack = ba.bands[i]
                + bb.bands[i]
                + bs.bands[i]
                + rounding
                + 2.0 * U * (ba.values[i].abs() + bb.values[i].abs());
            assert!((ba.values[i] + bb.values[i] - bs.values[i]).abs() <= slack);
        }
    }

    #[test]
    fn bilinear_secant_is_exact_and_the_first_order_form_is_not() {
        let fixture = |rows: usize, cols: usize, phase: f64| {
            Array2::from_shape_fn((rows, cols), |(i, j)| {
                ((i as f64 + 1.0) * (0.37 + phase) + (j as f64 + 1.0) * (0.61 - 0.5 * phase)).sin()
            })
        };
        let direct_band = |left: &Array2<f64>,
                           right: &Array2<f64>,
                           left_new: &Array2<f64>,
                           right_new: &Array2<f64>| {
            let value = left_new.dot(right_new) - left.dot(right);
            let absolute = left_new.mapv(f64::abs).dot(&right_new.mapv(f64::abs))
                + left.mapv(f64::abs).dot(&right.mapv(f64::abs));
            let inner = left.ncols();
            let band = Array2::from_shape_fn(value.dim(), |(i, k)| {
                inflated(
                    accumulation_growth(inner + 1) * absolute[[i, k]] + U * value[[i, k]].abs(),
                    inner,
                )
            });
            (value, band)
        };
        let (l, l_new) = (fixture(4, 6, 0.1), fixture(4, 6, 0.9));
        let (r, r_new) = (fixture(6, 3, 0.4), fixture(6, 3, 1.3));
        let change =
            bilinear_change(l.view(), l_new.view(), r.view(), r_new.view()).expect("finite");
        let (direct, band) = direct_band(&l, &r, &l_new, &r_new);
        let first_order = l.dot(&(&r_new - &r)) + (&l_new - &l).dot(&r);
        let mut first_order_fails = false;
        for ((i, k), value) in change.values.indexed_iter() {
            let allowance = change.bands[[i, k]] + band[[i, k]];
            assert!((value - direct[[i, k]]).abs() <= allowance);
            first_order_fails |= (first_order[[i, k]] - direct[[i, k]]).abs() > allowance;
        }
        assert!(
            first_order_fails,
            "the first-order product rule passed the secant band"
        );
        // Attention weights (rows of a softmax) times values.
        let weights = Array2::from_shape_fn((3, 4), |(i, j)| {
            probabilities(&logits(4, i as f64, 1.0)).0[j]
        });
        let weights_new = Array2::from_shape_fn((3, 4), |(i, j)| {
            probabilities(&logits(4, i as f64 + 0.5, 2.0)).0[j]
        });
        let (values, values_new) = (fixture(4, 5, 0.2), fixture(4, 5, 0.7));
        let attention = bilinear_change(
            weights.view(),
            weights_new.view(),
            values.view(),
            values_new.view(),
        )
        .expect("finite");
        let (attention_direct, attention_band) =
            direct_band(&weights, &values, &weights_new, &values_new);
        for ((i, k), value) in attention.values.indexed_iter() {
            assert!(
                (value - attention_direct[[i, k]]).abs()
                    <= attention.bands[[i, k]] + attention_band[[i, k]]
            );
        }
        // The vector form.
        let h: Array1<f64> = array![0.3, -1.2, 2.0, 0.7, -0.4, 1.1];
        let h_new: Array1<f64> = array![-0.9, 0.2, 1.5, 0.1, 0.8, -2.0];
        let vector =
            bilinear_vector_change(l.view(), l_new.view(), h.view(), h_new.view()).expect("finite");
        let column = |v: &Array1<f64>| v.clone().insert_axis(Axis(1));
        let (vector_direct, vector_band) = direct_band(&l, &column(&h), &l_new, &column(&h_new));
        for i in 0..4 {
            assert!(
                (vector.values[i] - vector_direct[[i, 0]]).abs()
                    <= vector.bands[i] + vector_band[[i, 0]]
            );
        }
    }

    fn silu(x: f64) -> (f64, f64) {
        let value = silu_derivatives(x)[0];
        (value, 7.0 * U * value.abs())
    }

    fn gelu(x: f64) -> (f64, f64) {
        let cdf = normal_cdf_and_pdf(x).0;
        let value = x * cdf;
        (
            value,
            x.abs() * (NORMAL_CDF_RELATIVE_ERROR * cdf + NORMAL_CDF_UNDERFLOW_FLOOR)
                + U * value.abs(),
        )
    }

    /// `(σ′, σ′′′, band of σ′)` at `x`. GELU: `σ′ = Φ + xφ`, `σ′′ = (2 − x²)φ`, `σ′′′ = (x³ − 4x)φ`.
    fn activation_jet(activation: SecantActivation, x: f64) -> (f64, f64, f64) {
        match activation {
            SecantActivation::Silu => {
                let jet = silu_derivatives(x);
                let band = 16.0 * U * (logistic(x) + x.abs() * logistic(x) * logistic(-x));
                (jet[1], jet[3], band)
            }
            SecantActivation::ExactGelu => {
                let (cdf, pdf) = normal_cdf_and_pdf(x);
                let band = 16.0 * U * (cdf + x.abs() * pdf) + NORMAL_CDF_UNDERFLOW_FLOOR;
                (cdf + x * pdf, (x * x * x - 4.0 * x) * pdf, band)
            }
        }
    }

    fn evaluate(activation: SecantActivation, x: f64) -> (f64, f64) {
        match activation {
            SecantActivation::Silu => silu(x),
            SecantActivation::ExactGelu => gelu(x),
        }
    }

    #[test]
    fn activation_slopes_reproduce_finite_changes() {
        let pairs = [
            (-3.0, 4.0),
            (0.5, 0.75),
            (-8.0, -6.0),
            (6.0, 9.0),
            (-0.25, 2.25),
            (-1.25, -1.125),
            (10.0, -10.0),
            (-40.0, 38.0),
        ];
        for activation in [SecantActivation::Silu, SecantActivation::ExactGelu] {
            for (a, b) in pairs {
                let slope = activation_divided_difference(activation, a, b).expect("finite");
                let width = b - a;
                let (sa, ba) = evaluate(activation, a);
                let (sb, bb) = evaluate(activation, b);
                let direct = sb - sa;
                let allowance = slope.band * width.abs() + ba + bb + 2.0 * U * direct.abs();
                assert!(
                    (slope.value * width - direct).abs() <= allowance,
                    "{activation:?} on [{a}, {b}]: {} vs {direct}",
                    slope.value * width
                );
            }
            let vector = activation_divided_differences(activation, &[0.5, -2.0], &[0.75, -2.0])
                .expect("finite");
            let single = activation_divided_difference(activation, 0.5, 0.75).expect("finite");
            assert_eq!(vector.values[0], single.value);
        }
    }

    #[test]
    fn activation_slopes_stay_accurate_where_the_naive_quotient_cancels() {
        let width = (-30.0_f64).exp2();
        for activation in [SecantActivation::Silu, SecantActivation::ExactGelu] {
            for a in [0.8, -1.25, -6.0, 7.0, 0.0] {
                // At a = b the slope is the analytic derivative.
                let (first, _, first_band) = activation_jet(activation, a);
                let at = activation_divided_difference(activation, a, a).expect("finite");
                assert!(
                    (at.value - first).abs() <= at.band + first_band,
                    "{activation:?} at {a}: {} vs {first}",
                    at.value
                );
                // Nearby, the Taylor expansion about the exact midpoint, σ′(m) + σ′′′(m)h²/24, whose next term is
                // below h⁴·sup|σ⁽⁵⁾|/1920 with sup|σ⁽⁵⁾| < 1920 for both.
                let b = a + width;
                let midpoint = a + 0.5 * width;
                let (first, third, first_band) = activation_jet(activation, midpoint);
                let reference = first + third * width * width / 24.0;
                let reference_band =
                    first_band + 4.0 * U * third.abs() * width * width + width.powi(4);
                let slope = activation_divided_difference(activation, a, b).expect("finite");
                assert!(
                    (slope.value - reference).abs() <= slope.band + reference_band,
                    "{activation:?} near {a}: {} vs {reference}",
                    slope.value
                );
                // The naive quotient loses about u·|σ|/h and must fail the same band where σ is not tiny.
                let naive = (evaluate(activation, b).0 - evaluate(activation, a).0) / width;
                if evaluate(activation, a).0.abs() > 1e-3 {
                    assert!(
                        (naive - reference).abs() > slope.band + reference_band,
                        "{activation:?} near {a}: the naive quotient passed the secant band"
                    );
                }
            }
        }
    }

    #[test]
    fn normal_cdf_series_and_direct_quotient_agree_where_both_are_resolved() {
        for (lo, hi) in [(-0.5, 1.5), (0.25, 3.0), (-4.0, -2.5), (1.0, 1.0625)] {
            let direct = normal_cdf_direct_quotient(lo, hi);
            let series = normal_cdf_series(lo, hi, f64::INFINITY);
            assert!(series.band.is_finite() && direct.band.is_finite());
            assert!(
                (series.value - direct.value).abs() <= series.band + direct.band,
                "[{lo}, {hi}]: series {series:?}, direct {direct:?}"
            );
        }
        // A tiny width is left to the series, a wide one to the direct quotient.
        let narrow_width = (-35.0_f64).exp2();
        let narrow = normal_cdf_divided_difference(0.3, 0.3 + narrow_width);
        assert!(narrow.band < normal_cdf_direct_quotient(0.3, 0.3 + narrow_width).band);
        assert!(narrow.band <= 64.0 * U * narrow.value);
        let wide_direct = normal_cdf_direct_quotient(-20.0, 25.0);
        let wide_series = normal_cdf_series(-20.0, 25.0, wide_direct.band);
        assert!(wide_series.band > wide_direct.band);
    }

    #[test]
    fn non_finite_and_malformed_inputs_are_refused() {
        let nan = f64::NAN;
        assert!(matches!(
            SoftmaxSecant::between(&[0.0, nan], &[0.0, 1.0]),
            Err(SecantError::NonFinite { index: 1, .. })
        ));
        assert!(matches!(
            softmax_change(&[0.0], &[0.0, 1.0]),
            Err(SecantError::Shape { .. })
        ));
        assert!(matches!(
            softmax_change(&[], &[]),
            Err(SecantError::Empty { .. })
        ));
        let operator = SoftmaxSecant::between(&[0.0, 1.0], &[1.0, 0.0]).expect("finite");
        assert!(matches!(
            operator.apply(&[f64::INFINITY, 0.0]),
            Err(SecantError::NonFinite { .. })
        ));
        assert!(matches!(
            RmsNormSecant::between(&[1.0], &[2.0], 0.0),
            Err(SecantError::NonPositiveEpsilon { .. })
        ));
        assert!(matches!(
            RmsNormSecant::between(&[1.0], &[nan], 1e-6),
            Err(SecantError::NonFinite { .. })
        ));
        assert!(matches!(
            RmsNormSecant::between(&[f64::MAX, f64::MAX], &[1.0, 1.0], 1e-6),
            Err(SecantError::Overflow { .. })
        ));
        let zeros = Array2::<f64>::zeros((2, 2));
        let mut bad = zeros.clone();
        bad[[1, 0]] = nan;
        assert!(matches!(
            bilinear_change(zeros.view(), bad.view(), zeros.view(), zeros.view()),
            Err(SecantError::NonFinite { .. })
        ));
        let column = Array2::<f64>::zeros((3, 1));
        assert!(matches!(
            bilinear_change(zeros.view(), zeros.view(), column.view(), column.view()),
            Err(SecantError::Shape { .. })
        ));
        assert!(matches!(
            activation_divided_difference(SecantActivation::Silu, nan, 0.0),
            Err(SecantError::NonFinite { .. })
        ));
        assert!(matches!(
            activation_divided_difference(SecantActivation::ExactGelu, -f64::MAX, f64::MAX),
            Err(SecantError::Overflow { .. })
        ));
    }
}

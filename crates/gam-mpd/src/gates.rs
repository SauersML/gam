//! Switching functions (#2951): which subcomponents are on for an input, computed by a small
//! formula over the model's own upstream amplitudes instead of a search or a separate importance
//! network. Its bits are part of the one total description length: the switching function's own
//! bits plus the per-word listing of what it gets wrong.
//!
//! # The switching function
//!
//! A subcomponent `j` reads the amplitude `a_j = v_jᵀx` of its site's input `x`, and `x` is exactly a
//! sum of upstream contributions, so `a_j` is itself an exact contraction `ℓᵀa_up` of the upstream
//! amplitudes (given the norm scales and gates the forward computes). Its switching function is a
//! logit over a few features `f`: its own `a_j` first, then amplitudes `a_k` of upstream
//! subcomponents at this position (lag 0) or at the previous position (lag 1). Only what the native
//! forward has computed when `j` runs is allowed ([`upstream`]): a site's amplitudes at this position
//! when the site precedes `j`'s read, and at the previous position only through an attention step
//! between them. Nothing downstream or in the future is read, so the switching functions are a
//! causal circuit, not an analyser:
//!
//! ```text
//!   ĝ_j(f) = β + ℓᵀf + Σ_{u<r} c_u GELU(w_uᵀf + d_u),     P(j on | f) = σ(ĝ_j(f)),
//! ```
//!
//! and `j` is on exactly when `ĝ_j(f) > 0`. With no features and no units it is the base rate.
//! Nothing about its size is fixed: the features, the number of units `r` and the precision of
//! every coefficient are chosen by the code below.
//!
//! # The code
//!
//! A switching function is sent as `L_int(d + 1)`, the feature subset (its bits given by the
//! caller, who knows the pool it was chosen from), `L_int(r + 1)`, a dyadic precision `p` (signed
//! prefix integer) and every coefficient as the signed prefix integer `round(θ·2^p)`
//! ([`Switch::function_bits`]). The per-word listing then sends the on/off labels at the inputs
//! under the decoded function, `−Σ log₂ P(y | f)` ([`Switch::listing_bits`]). The chosen function is
//! the one with the fewest total bits, measured with the coefficients the decoder rebuilds, so a
//! coefficient's precision is paid for exactly as far as the listing repays it. A group of
//! subcomponents doing many unrelated jobs needs a long switching function; [`best`] prices a
//! candidate group's on-labels so the one total can prefer splitting it.
//!
//! # The fit
//!
//! Maximum likelihood by Levenberg–Marquardt on the Gauss–Newton (Fisher) matrix of the logistic
//! likelihood, which is exact for the logit's linear part. The ladder starts from the base rate
//! (Krichevsky–Trofimov, closed form), fits the linear function, then adds one unit at a time, each
//! started where it is worth most: among hinges along each feature (both signs) at the quantiles of
//! the on-inputs' and of all inputs' feature values, the one with the largest Rao score
//! `s²/I` (`s`, `I` the likelihood's gradient and information in the new unit's output weight at
//! zero), its weight at the one-dimensional Newton step `−s/I`. The ladder stops at the first unit
//! that does not lower the total. A fit stops when the undamped Gauss–Newton step promises less
//! than a thousandth of a bit: the code is compared in bits, and coefficients are then rounded far
//! coarser than that.
//!
//! [`screen`] proposes features from a pool by the same score: per candidate `k` and function `j`,
//! the predicted bits of adding `a_k` linearly, `(Σ_t r_tj a_tk)² / (2 ln 2 Σ_t w_tj a_tk²)` with `r`
//! the residual `y − p` and `w = p(1 − p)`. It only proposes (its products may run in f32 on the
//! device); the refitted function's exact total decides.

use super::codec::{prefix_integer_len_bits, signed_prefix_integer_len_bits};
use super::device::{product_atb, proposing};
use super::masked::Site;
use super::operator_program::{Node, OperatorProgram};
use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// One feature of a switching function: the amplitude of subcomponent `piece` of site `site` at this input (`lag`
/// 0) or at the previous input of the sequence (`lag` 1, zero at a sequence's first input).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Feature {
    pub site: usize,
    pub piece: usize,
    pub lag: usize,
}

/// One nonlinear unit: `c · GELU(wᵀf + d)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Unit {
    pub w: Vec<f64>,
    pub d: f64,
    pub c: f64,
}

/// A decoded switching function (module note) with its code lengths on its training inputs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Switch {
    pub features: Vec<Feature>,
    pub beta: f64,
    pub linear: Vec<f64>,
    pub units: Vec<Unit>,
    /// The dyadic precision `p`: every coefficient is an integer multiple of `2^-p`.
    pub precision: i32,
    /// The switching function's own bits: structure, precision and coefficients.
    pub function_bits: f64,
    /// The per-word listing of the training labels under the function.
    pub listing_bits: f64,
}

impl Switch {
    pub fn total_bits(&self) -> f64 {
        self.function_bits + self.listing_bits
    }

    /// The logit at feature values `f` (in the switch's feature order).
    pub fn logit(&self, f: &[f64]) -> f64 {
        let mut g = self.beta + dot(&self.linear, f);
        for u in &self.units {
            g += u.c * gelu(dot(&u.w, f) + u.d);
        }
        g
    }

    /// Whether the switch says the subcomponent is on.
    pub fn on(&self, f: &[f64]) -> bool {
        self.logit(f) > 0.0
    }

    /// Multiply–adds per input to evaluate the switch (a unit's GELU counted as one).
    pub fn multiply_adds(&self) -> usize {
        let d = self.features.len();
        d + self.units.len() * (d + 2)
    }

    /// The switch's bits for labels `y` at feature rows `x` (`n × d`).
    pub fn label_bits(&self, x: ArrayView2<f64>, y: &[bool]) -> f64 {
        let data = row_major(x);
        let d = x.ncols();
        y.iter().enumerate().map(|(t, on)| label_bits(self.logit(&data[t * d..(t + 1) * d]), *on)).sum()
    }
}

/// `x`'s entries row after row, borrowed when `x` is already laid out so.
fn row_major(x: ArrayView2<'_, f64>) -> Cow<'_, [f64]> {
    match x.to_slice() {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(x.iter().copied().collect()),
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

/// GELU in its tanh form (the form the models it gates use).
pub fn gelu(h: f64) -> f64 {
    0.5 * h * (1.0 + (SQRT_2_OVER_PI * (h + 0.044715 * h * h * h)).tanh())
}

fn gelu_derivative(h: f64) -> f64 {
    let t = (SQRT_2_OVER_PI * (h + 0.044715 * h * h * h)).tanh();
    0.5 * (1.0 + t) + 0.5 * h * (1.0 - t * t) * SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * h * h)
}

/// `log(1 + e^g)` without overflow.
fn softplus(g: f64) -> f64 {
    if g > 0.0 { g + (-g).exp().ln_1p() } else { g.exp().ln_1p() }
}

fn sigmoid(g: f64) -> f64 {
    if g >= 0.0 { 1.0 / (1.0 + (-g).exp()) } else { g.exp() / (1.0 + g.exp()) }
}

/// `−log₂ P(y | logit g)`.
pub fn label_bits(g: f64, on: bool) -> f64 {
    (if on { softplus(-g) } else { softplus(g) }) / std::f64::consts::LN_2
}

fn int_bits(value: i64) -> f64 {
    signed_prefix_integer_len_bits(value).map_or(f64::INFINITY, |b| b as f64)
}

fn count_bits(value: usize) -> f64 {
    prefix_integer_len_bits(value as u64 + 1).map_or(f64::INFINITY, |b| b as f64)
}

/// The parameter vector `[β, ℓ (d), per unit (w (d), d_u, c_u)]` of a switch with `d` features.
#[derive(Clone, Debug)]
struct Params {
    d: usize,
    theta: Vec<f64>,
}

impl Params {
    fn units(&self) -> usize {
        (self.theta.len() - 1 - self.d) / (self.d + 2)
    }

    fn unit(&self, u: usize) -> usize {
        1 + self.d + u * (self.d + 2)
    }

    /// The logit at row `f` and, if asked, its gradient in θ.
    fn logit(&self, f: &[f64], jac: Option<&mut [f64]>) -> f64 {
        let d = self.d;
        let t = &self.theta;
        let mut g = t[0] + dot(&t[1..1 + d], f);
        let mut jac = jac;
        if let Some(j) = jac.as_deref_mut() {
            j[0] = 1.0;
            j[1..1 + d].copy_from_slice(f);
        }
        for u in 0..self.units() {
            let o = self.unit(u);
            let h = dot(&t[o..o + d], f) + t[o + d];
            let c = t[o + d + 1];
            let gh = gelu(h);
            g += c * gh;
            if let Some(j) = jac.as_deref_mut() {
                let s = c * gelu_derivative(h);
                for (k, fk) in f.iter().enumerate() {
                    j[o + k] = s * fk;
                }
                j[o + d] = s;
                j[o + d + 1] = gh;
            }
        }
        g
    }

    /// Negative log-likelihood in nats over row-major feature rows.
    fn nll(&self, x: &[f64], y: &[bool]) -> f64 {
        let d = self.d;
        y.iter()
            .enumerate()
            .map(|(t, on)| {
                let g = self.logit(&x[t * d..(t + 1) * d], None);
                if *on { softplus(-g) } else { softplus(g) }
            })
            .sum()
    }

    /// Gradient and Gauss–Newton (Fisher) matrix of the NLL.
    fn derivatives(&self, x: &[f64], y: &[bool]) -> (Vec<f64>, Vec<f64>) {
        let (p, d) = (self.theta.len(), self.d);
        let mut grad = vec![0.0; p];
        let mut fisher = vec![0.0; p * p];
        let mut jac = vec![0.0; p];
        for (t, on) in y.iter().enumerate() {
            let g = self.logit(&x[t * d..(t + 1) * d], Some(&mut jac));
            let q = sigmoid(g);
            let r = q - if *on { 1.0 } else { 0.0 };
            let w = q * (1.0 - q);
            for a in 0..p {
                grad[a] += r * jac[a];
                let wa = w * jac[a];
                for b in 0..=a {
                    fisher[a * p + b] += wa * jac[b];
                }
            }
        }
        for a in 0..p {
            for b in 0..a {
                fisher[b * p + a] = fisher[a * p + b];
            }
        }
        (grad, fisher)
    }
}

/// Solve the symmetric positive definite system `A z = b` (`A` row-major `n × n`) by Cholesky;
/// `None` when `A` is not positive definite.
fn solve_spd(a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let s = a[i * n + j] - (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum::<f64>();
            if i == j {
                if !(s > 0.0) || !s.is_finite() {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut z = b.to_vec();
    for i in 0..n {
        z[i] = (z[i] - (0..i).map(|k| l[i * n + k] * z[k]).sum::<f64>()) / l[i * n + i];
    }
    for i in (0..n).rev() {
        z[i] = (z[i] - (i + 1..n).map(|k| l[k * n + i] * z[k]).sum::<f64>()) / l[i * n + i];
    }
    Some(z)
}

/// Levenberg–Marquardt to the likelihood's optimum (module note); returns the NLL in nats.
fn optimise(params: &mut Params, x: &[f64], y: &[bool]) -> f64 {
    let stop = 1e-3 * std::f64::consts::LN_2;
    let mut nll = params.nll(x, y);
    let mut lambda = 1e-3;
    let p = params.theta.len();
    // Each accepted step must gain; a refused one raises the damping. The damping cannot grow
    // without bound and still move θ, so a ceiling on it is where no step helps. Converged when
    // the undamped Gauss–Newton step promises less than the stopping gain; the step count is only
    // a guard against a likelihood that keeps creeping (separable labels).
    for _ in 0..1000 {
        if lambda >= 1e12 {
            break;
        }
        let (grad, fisher) = params.derivatives(x, y);
        let trace = (0..p).map(|a| fisher[a * p + a]).sum::<f64>() / p as f64;
        let mut ridged = fisher.clone();
        for a in 0..p {
            ridged[a * p + a] += 1e-12 * trace.max(f64::MIN_POSITIVE);
        }
        let negative: Vec<f64> = grad.iter().map(|g| -g).collect();
        if let Some(newton) = solve_spd(&ridged, &negative)
            && 0.5 * dot(&newton, &negative) < stop
        {
            return nll;
        }
        loop {
            let mut damped = fisher.clone();
            for a in 0..p {
                damped[a * p + a] += lambda * (fisher[a * p + a] + 1e-12 * trace.max(f64::MIN_POSITIVE));
            }
            let Some(step) = solve_spd(&damped, &negative) else {
                lambda *= 10.0;
                if lambda >= 1e12 {
                    return nll;
                }
                continue;
            };
            let trial = Params { d: params.d, theta: params.theta.iter().zip(&step).map(|(t, s)| t + s).collect() };
            let next = trial.nll(x, y);
            if next.is_finite() && next < nll {
                *params = trial;
                nll = next;
                lambda = (lambda / 10.0).max(1e-12);
                break;
            }
            lambda *= 10.0;
            if lambda >= 1e12 {
                return nll;
            }
        }
    }
    nll
}

/// The switch these parameters give at the precision that minimises the total code, the coefficient
/// bits counted exactly and the labels' bits measured under the decoded coefficients.
fn encode(params: &Params, x: &[f64], y: &[bool], features: &[Feature], structure_bits: f64) -> Switch {
    let (_, fisher) = params.derivatives(x, y);
    let p = params.theta.len();
    let header = count_bits(params.d) + structure_bits + count_bits(params.units());
    let largest = params.theta.iter().fold(0.0f64, |m, t| m.max(t.abs()));
    // The coarsest useful precision rounds every coefficient to zero; past 52 fraction bits more
    // than the largest coefficient's mantissa, nothing changes.
    let coarsest = -(largest.max(f64::MIN_POSITIVE).log2().ceil() as i32) - 2;
    let finest = coarsest + 60;
    // Rank precisions by the coefficient bits plus the second-order cost of the rounding error
    // (the Fisher at the optimum); measure the best few exactly.
    let mut ranked: Vec<(f64, i32)> = (coarsest..=finest)
        .map(|prec| {
            let scale = (prec as f64).exp2();
            let ints: Vec<i64> = params.theta.iter().map(|t| (t * scale).round() as i64).collect();
            let delta: Vec<f64> = params.theta.iter().zip(&ints).map(|(t, k)| *k as f64 / scale - t).collect();
            let mut quad = 0.0;
            for a in 0..p {
                for b in 0..p {
                    quad += delta[a] * fisher[a * p + b] * delta[b];
                }
            }
            let coef: f64 = ints.iter().map(|k| int_bits(*k)).sum::<f64>() + int_bits(prec as i64);
            (coef + 0.5 * quad / std::f64::consts::LN_2, prec)
        })
        .collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut best: Option<Switch> = None;
    for &(_, prec) in ranked.iter().take(3) {
        let scale = (prec as f64).exp2();
        let ints: Vec<i64> = params.theta.iter().map(|t| (t * scale).round() as i64).collect();
        let decoded = Params { d: params.d, theta: ints.iter().map(|k| *k as f64 / scale).collect() };
        let function_bits = header + ints.iter().map(|k| int_bits(*k)).sum::<f64>() + int_bits(prec as i64);
        let switch = to_switch(&decoded, features, prec, function_bits, decoded.nll(x, y) / std::f64::consts::LN_2);
        if best.as_ref().is_none_or(|b| switch.total_bits() < b.total_bits()) {
            best = Some(switch);
        }
    }
    best.unwrap_or_else(|| to_switch(params, features, 0, f64::INFINITY, f64::INFINITY))
}

fn to_switch(params: &Params, features: &[Feature], precision: i32, function_bits: f64, listing_bits: f64) -> Switch {
    let d = params.d;
    let t = &params.theta;
    Switch {
        features: features.to_vec(),
        beta: t[0],
        linear: t[1..1 + d].to_vec(),
        units: (0..params.units())
            .map(|u| {
                let o = params.unit(u);
                Unit { w: t[o..o + d].to_vec(), d: t[o + d], c: t[o + d + 1] }
            })
            .collect(),
        precision,
        function_bits,
        listing_bits,
    }
}

fn from_switch(switch: &Switch) -> Params {
    let mut theta = vec![switch.beta];
    theta.extend_from_slice(&switch.linear);
    for u in &switch.units {
        theta.extend_from_slice(&u.w);
        theta.push(u.d);
        theta.push(u.c);
    }
    Params { d: switch.features.len(), theta }
}

/// The base-rate switch: the Krichevsky–Trofimov rate of the training labels.
pub fn base(y: &[bool]) -> Switch {
    let on = y.iter().filter(|v| **v).count() as f64;
    let rate = (on + 0.5) / (y.len() as f64 + 1.0);
    let params = Params { d: 0, theta: vec![(rate / (1.0 - rate)).ln()] };
    encode(&params, &[], y, &[], 0.0)
}

/// The fewest bits any switch with features can take: its header and at least one bit for each of
/// its precision and its three coefficients.
pub fn least_featured_bits(structure_bits: f64) -> f64 {
    count_bits(1) + structure_bits + count_bits(0) + 4.0
}

/// The best switching function for on-labels `y` (a subcomponent's, or a candidate group's) over
/// feature rows `x` (`n × d`, columns in the order of `features`), `structure_bits` the code of the
/// feature subset: the cheaper of the base rate and [`fit`]. Its `total_bits()` (function plus
/// per-word listing) is what the labels cost in the one total.
pub fn best(x: ArrayView2<f64>, y: &[bool], features: &[Feature], structure_bits: f64) -> Switch {
    let rate = base(y);
    if features.is_empty() || rate.total_bits() <= least_featured_bits(structure_bits) {
        return rate;
    }
    let fitted = fit(x, y, features, structure_bits, None);
    if fitted.total_bits() < rate.total_bits() { fitted } else { rate }
}

/// The best switch over feature rows `x` (`n × d`, columns in the order of `features`) for labels `y`,
/// `structure_bits` the code of the feature subset: the linear switch, then one unit more at a time
/// while the total falls (module note). `start` (a switch over the first `d − 1` of these features,
/// or over all of them) warm-starts the ladder.
pub fn fit(x: ArrayView2<f64>, y: &[bool], features: &[Feature], structure_bits: f64, start: Option<&Switch>) -> Switch {
    let d = features.len();
    let data = row_major(x);
    let x: &[f64] = &data;
    let mut params = match start {
        Some(switch) => {
            let mut p = from_switch(switch);
            if p.d + 1 == d {
                p = widen(&p);
            }
            Params { d, theta: p.theta.iter().take(1 + d).copied().collect() }
        }
        None => {
            let rate = base(y);
            let mut theta = vec![rate.beta];
            theta.extend(std::iter::repeat_n(0.0, d));
            Params { d, theta }
        }
    };
    optimise(&mut params, x, y);
    let mut best = encode(&params, x, y, features, structure_bits);
    loop {
        let Some(mut grown) = add_unit(&params, x, y) else { break };
        optimise(&mut grown, x, y);
        let switch = encode(&grown, x, y, features, structure_bits);
        if switch.total_bits() < best.total_bits() {
            best = switch;
            params = grown;
        } else {
            break;
        }
    }
    best
}

/// `p` with one more feature, its coefficients zero (inserted last in every block).
fn widen(p: &Params) -> Params {
    let d = p.d;
    let mut theta = p.theta[..1 + d].to_vec();
    theta.push(0.0);
    for u in 0..p.units() {
        let o = p.unit(u);
        theta.extend_from_slice(&p.theta[o..o + d]);
        theta.push(0.0);
        theta.push(p.theta[o + d]);
        theta.push(p.theta[o + d + 1]);
    }
    Params { d: d + 1, theta }
}

/// `p` with one more unit, started at the hinge with the largest Rao score (module note).
fn add_unit(p: &Params, x: &[f64], y: &[bool]) -> Option<Params> {
    let d = p.d;
    if d == 0 {
        return None;
    }
    let n = y.len();
    let logits: Vec<f64> = (0..n).map(|t| p.logit(&x[t * d..(t + 1) * d], None)).collect();
    let mut best: Option<(f64, Vec<f64>, f64, f64)> = None;
    for k in 0..d {
        let column: Vec<f64> = (0..n).map(|t| x[t * d + k]).collect();
        let mean = column.iter().sum::<f64>() / n as f64;
        let spread = (column.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
        if !(spread > 0.0) {
            continue;
        }
        for sign in [1.0, -1.0] {
            let scale = sign / spread;
            let mut on_values: Vec<f64> = column.iter().zip(y).filter(|(_, o)| **o).map(|(v, _)| v * scale).collect();
            let mut all_values: Vec<f64> = column.iter().map(|v| v * scale).collect();
            on_values.sort_by(f64::total_cmp);
            all_values.sort_by(f64::total_cmp);
            let mut hinges = Vec::new();
            for values in [&on_values, &all_values] {
                if values.is_empty() {
                    continue;
                }
                for q in [0.1, 0.3, 0.5, 0.7, 0.9] {
                    hinges.push(values[((values.len() - 1) as f64 * q).round() as usize]);
                }
            }
            for hinge in hinges {
                let (mut s, mut info) = (0.0, 0.0);
                for (t, (v, on)) in column.iter().zip(y).enumerate() {
                    let h = gelu(v * scale - hinge);
                    let q = sigmoid(logits[t]);
                    s += (q - if *on { 1.0 } else { 0.0 }) * h;
                    info += q * (1.0 - q) * h * h;
                }
                if info > 0.0 && best.as_ref().is_none_or(|b| s * s / info > b.0) {
                    let mut w = vec![0.0; d];
                    w[k] = scale;
                    best = Some((s * s / info, w, -hinge, -s / info));
                }
            }
        }
    }
    let (_, w, bias, c) = best?;
    let mut theta = p.theta.clone();
    theta.extend_from_slice(&w);
    theta.push(bias);
    theta.push(c);
    Some(Params { d, theta })
}

/// The predicted bits of adding each candidate feature linearly to each switch (module note):
/// `candidates` is `n × K`, `residuals` and `weights` are `n × J` (`y − p` and `p(1 − p)` per switch
/// at the same inputs); returns `K × J`. A proposal: its products may run in f32 on the device.
pub fn screen(candidates: &Array2<f64>, residuals: &Array2<f64>, weights: &Array2<f64>) -> Result<Array2<f64>, String> {
    let squares = candidates.mapv(|v| v * v);
    let (score, info) = proposing(|| -> Result<_, String> {
        let score = product_atb(candidates, residuals).map_err(|e| e.to_string())?;
        let info = product_atb(&squares, weights).map_err(|e| e.to_string())?;
        Ok((score, info))
    })?;
    let mut gains = score;
    gains.zip_mut_with(&info, |s, i| *s = if *i > 0.0 { *s * *s / (2.0 * std::f64::consts::LN_2 * i) } else { 0.0 });
    Ok(gains)
}

/// The sites whose amplitudes a site's switching functions may read (module note).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Upstream {
    /// At the same position: every written node precedes the site's first read node.
    pub here: Vec<usize>,
    /// At the previous position: an attention step lies after every written node and at or before
    /// the site's first read node, so the previous position reaches it only through attention.
    pub before: Vec<usize>,
}

/// Per site, its [`Upstream`] sites in `program` (nodes in topological order).
pub fn upstream(program: &OperatorProgram, sites: &[Site]) -> Vec<Upstream> {
    let mixing: Vec<usize> =
        program.nodes.iter().enumerate().filter(|(_, n)| matches!(n, Node::Attend { .. } | Node::Mix { .. })).map(|(i, _)| i).collect();
    sites
        .iter()
        .map(|b| {
            let first_read = b.reads.iter().min().copied().unwrap_or(0);
            let last_write = |a: &Site| a.writes.iter().max().copied().unwrap_or(usize::MAX);
            let here = sites.iter().enumerate().filter(|(_, a)| last_write(a) < first_read).map(|(i, _)| i).collect();
            let before = sites
                .iter()
                .enumerate()
                .filter(|(_, a)| mixing.iter().any(|m| last_write(a) < *m && *m <= first_read))
                .map(|(i, _)| i)
                .collect();
            Upstream { here, before }
        })
        .collect()
}

/// A switch's decisions over feature rows, as a column of 0/1.
pub fn decisions(switch: &Switch, x: ArrayView2<f64>) -> Array1<f64> {
    let data = row_major(x);
    let d = x.ncols();
    (0..x.nrows()).map(|t| if switch.on(&data[t * d..(t + 1) * d]) { 1.0 } else { 0.0 }).collect()
}

/// The sets the switches choose (`switches[site][piece]`), as per-site 0/1 masks (`rows × pieces`), from
/// every site's amplitudes on the clean forward (`amplitudes[site]` is `rows × pieces`) and each
/// row's previous row in its sequence.
pub fn masks(switches: &[Vec<Switch>], amplitudes: &[Array2<f64>], previous: &[Option<usize>]) -> Vec<Array2<f64>> {
    switches.iter()
        .map(|site_switches| {
            let mut m = Array2::<f64>::zeros((previous.len(), site_switches.len()));
            for (piece, switch) in site_switches.iter().enumerate() {
                let mut f = vec![0.0; switch.features.len()];
                for (r, prev) in previous.iter().enumerate() {
                    for (k, feature) in switch.features.iter().enumerate() {
                        let row = if feature.lag == 0 { Some(r) } else { *prev };
                        f[k] = row.map_or(0.0, |row| amplitudes[feature.site][[row, feature.piece]]);
                    }
                    if switch.on(&f) {
                        m[[r, piece]] = 1.0;
                    }
                }
            }
            m
        })
        .collect()
}

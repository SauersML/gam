//! A12: the absolute-risk prediction benchmark for the joint latent-signature event
//! model (#2961).
//!
//! # What this measures and why it exists
//!
//! #2961 claims that a model which infers more of a patient's shared predictive
//! state, earlier and with better uncertainty, predicts absolute risk better than a
//! diagnoses-and-genetics baseline. That claim needs numbers. This program implements
//! the A12 design (jls-verify, #2961 comment 5720829207) on cohorts drawn from a known
//! joint law. For every target `d` (a disease, or death), landmark `s` and horizon
//! `u`, it scores
//!
//! ```text
//! F_d(s, u | H_s) = P(s < T_d <= s + u, T_d < T_death | H_s)
//! ```
//!
//! against held-out subjects' realised outcomes. Every arm predicts the same subjects
//! at the same landmark from the same history `H_s`. Every arm is scored with the same
//! censoring weights, and death is a competing event throughout.
//!
//! # Scenarios
//!
//! All four are declared; a run names one with `--scenario`.
//! * `S1`: the joint law below, correctly specified for the joint model.
//! * `S2`: the null. No signature contributions, no genetic drive, no jumps: events do
//!   not depend on the state, so no arm can beat `null` beyond sampling error.
//! * `S3`: the log-linear event-history engine's own law, including its entry contract:
//!   `lambda = exp(eta0(age, sex, g) + a' z - |a|^2 / 2)` over OU atoms that start
//!   stationary at entry, with no pre-entry records, no survivor selection and no jumps.
//!   The engine's complete-case arm is correctly specified here: the positive control
//!   that the harness recovers a correctly specified model.
//! * `S4`: S1 with visit attendance rising with the first signature, which breaks the
//!   exogenous-visit contract.
//!
//! # The joint law (S1)
//!
//! Time is age in years.
//! - **Context.** Sex and four ancestry PCs.
//! - **Genetics.** Three genetic scores, linear in the PCs with correlated residuals.
//!   Each subject's scores are missing with probability 0.1, or 0.3 when PC1 > 1:
//!   missing at random given the context.
//! - **State.** Three signatures follow OU processes, at rates 0.1, 0.2 and 0.5 per
//!   year, with stationary variance one. Their means are `B(age) g + u(age, sex)`,
//!   linear in age. The path starts at age 30 from its stationary law.
//! - **Marks.** Six once-only diseases, a recurrent hospitalisation and death. Their
//!   rates are `exp(a_m + b_m (age - 55) + sex effect) (pi_m0 + sum_k pi_mk
//!   softplus(x_k))`, with sparse decoders. Three diagnoses jump their main signature.
//! - **Records.** Enrolment happens at an age uniform on [40, 70], among the living.
//!   Records start up to ten years earlier, and events before the record start are
//!   never observed.
//! - **Visits.** Annual, from the record start. Attendance depends on the context only.
//!   An attended visit measures five channels, each missing with probability 0.3:
//!   two Student-t labs, a probit binary survey, a four-level cumulative-probit survey
//!   and a negative-binomial count.
//! - **Follow-up.** Until age 85 or fifteen years after enrolment. Loss to follow-up is
//!   independent, at 0.02 per year.
//!
//! The law is simulated on a weekly age grid, with intensities and drift frozen over
//! each step, and a hospitalisation fires at most once per step. The generator, the
//! brute-force forward simulation, the Rao-Blackwellised forward paths and the particle
//! filter share this discretised law exactly. The self-check reruns the forward paths
//! at half the step.
//!
//! Censoring truncates one simulated truth at an independent loss time. So each cell
//! is scored twice on the same outcomes: with IPCW on the censored cohort, and exactly
//! on the uncensored truth.
//!
//! # Arms
//!
//! * `null`: the censored training cohort's landmark Aalen-Johansen cumulative
//!   incidence, one prediction for everyone.
//! * `oracle-hs`: `F*`, the Bayes forecast under the true law given `H_s`, from a
//!   particle filter. The filter:
//!   - draws missing scores from their law given the PCs, and the pre-record path from
//!     the law;
//!   - weights survival to entry, the observed events and non-events from the record
//!     start, and each visit's channel values (plus attendance in S4);
//!   - forces the observed jumps;
//!   - averages Rao-Blackwellised forward paths from a systematic resample of the cloud.
//!   The forecast is the mean of `R` independent replicate filters. Their spread over
//!   `sqrt(R)` is the Monte Carlo standard error, and every replicate's ESS is reported
//!   per subject. It is the attainable ceiling for the full information set.
//! * `oracle-hs-dx`: the same filter without the measurement process, so context,
//!   scores and records only. It is the ceiling for the engine arms' information.
//! * `oracle-state`: the true law given the true Markov state at `s`. No
//!   `H_s`-measurable predictor reaches it in expected proper score.
//! * `evh-dx-genetics`: the production event-history engine
//!   (`fit_event_history_formulas`, then `forecast_history`) on sex, PCs and scores.
//!   This engine requires every covariate, so a missing score enters as zero plus an
//!   indicator: the mean-imputation arm.
//! * `evh-dx-genetics-cc`: the same engine on subjects with observed scores only: the
//!   complete-case arm.
//! * `evh-dx-genetics-channels`: the mean-imputation arm plus every channel's last
//!   observed value and an ever-measured indicator, as time-varying covariates, held
//!   over the forecast window. The engine has no measurement channels.
//!
//! After entry, the engine forecasts through its history path. Its histories start at
//! enrolment, with the record window's diagnoses as prior history. One
//! `SubjectHistory` has a single risk-window start, so death cannot start at enrolment
//! while the diseases start at the record start. A history from the record start would
//! put death at risk in immortal time. At the entry landmark there is no follow-up to
//! filter, so the engine arms use the engine's population forecast for an entrant with
//! the subject's covariates: the latent state starts from its stationary prior and
//! every mark is at risk.
//!
//! * `joint-rank0`: the joint event model's rank-zero slice (`fit_joint_event_model`,
//!   then `condition` and `forecast`). It is a floor: constant rates, no covariates, no
//!   latent state. Its histories start at enrolment, so it has no entry-landmark
//!   forecast, and the tables report that landmark as not produced. K >= 1 slices join as
//!   they land.
//! * `aladyn-map`: the ALADYNOULLI-style baseline, by MAP of its published objective. Read line
//!   by line at surbut/aladynoulli2 @ 563f6cd24fe69598f8e7439848fa130ed825fabb,
//!   `pyScripts_forPublish/clust_huge_amp_vectorized.py`:
//!   - `theta = softmax(lambda)`, `pi = kappa sum_k theta phi_prob` (:221-228);
//!   - the discrete first-occurrence loss (:230-248);
//!   - the GP priors on lambda around the genetic mean, and on phi around the prevalence
//!     logits plus psi (:288-350).
//!
//!   The published fit is Adam with parameter groups for a fixed number of epochs (:353-394),
//!   with no convergence test. This arm minimises with opt's BFGS as is, and accepts the final
//!   point only when every gradient coordinate lies within its running rounding bound
//!   `eps mu_j`. An uncertified fit yields no arm, and the report names opt's termination.
//!   opt's BFGS keeps a dense inverse Hessian, so until opt carries a limited-memory solver the
//!   default `aladyn-map` arm fits nothing and reports its parameter count and the dense inverse
//!   Hessian's size as awaiting that solver. `--arms aladyn-map-dense` runs the dense fit, which
//!   only small cohorts can hold. Declared deviations:
//!   - the gated LRT penalty (:257-281) is omitted;
//!   - `kappa = exp(kappa_raw)`, with the objective refused where some `pi >= 1`, in place of
//!     the published clamp of `pi` to `[1e-6, 1 - 1e-6]`;
//!   - at a forecast, the landmark's yearly bin is the first forecast bin, not a history bin;
//!   - cyclic clusters in place of spectral clustering;
//!   - left truncation at the record start.
//!
//!   The file has no forecast. At `s` the arm refits the subject's lambda on its history before
//!   `s`, with the other parameters held, and takes `1 - prod (1 - pi)` over the horizon's
//!   yearly bins. Death is censoring in that model, so it is reported alongside the
//!   competing-risk arms, never as the same quantity, and it has no death target.
//!
//! # Metrics
//!
//! Landmarks are entry, entry + 2 and entry + 5 years; horizons are 1, 5 and 10 years.
//! The landmark set for `s` is every subject under observation at `s` whose records are
//! free of `d` there. In `(s, s+u]`, a subject is a case (`d` first), a competing death,
//! an event-free control (observed through `s+u`), or censored. `G` is the reverse
//! Kaplan-Meier censoring survival, in time since entry, among subjects under
//! observation at `s`. Cases and deaths weigh `1/G(T-)`, controls weigh `1/G((s+u)-)`,
//! and the censored weigh zero. Per cell, the program reports:
//! * the IPCW Brier score, and IPA `1 - Brier / Brier_null`;
//! * the IPCW log score of the case indicator over its finite contributions, with the count
//!   of infinite ones;
//! * the cumulative/dynamic IPCW AUC twice: against every uncensored non-case, and
//!   against event-free survivors only;
//! * the IPCW competing-risk concordance truncated at `s+u` (Wolbers). A pair against a
//!   subject still at risk weighs `1/(G(T_i-) G(T_i))`, and a pair against an earlier
//!   death weighs `1/(G(T_i-) G(T_j-))`;
//! * the IPCW logistic calibration intercept (offset `logit F`) and slope;
//! * the same intercept and slope from jackknife pseudo-observations of the landmark
//!   Aalen-Johansen estimator, regressed on `logit F` through the logit link;
//! * the IPCW observed incidence against the mean prediction;
//! * the mean `|F - F*|` against `oracle-hs`;
//! * paired bootstrap standard errors of the Brier and AUC differences against `null`,
//!   the previous arm, and each oracle filter (the regret). Each replicate resamples the
//!   cohort and re-estimates `G`. These are test-sampling errors, conditional on the fitted
//!   arms. A Brier difference within its Monte Carlo standard error, from the oracles'
//!   replicate errors, is reported as unresolved.
//!
//! Every cell is scored over two subsets: `all` (every arm except the complete-case
//! engine) and `complete-g` (the subjects with observed scores, every arm). For each arm,
//! the eligible subjects it cannot predict are listed per cell, with their case count. An
//! oracle filter that still lost an eligible subject after `FILTER_DOUBLINGS` particle
//! doublings is refused for that cell. So is `aladyn-map` when an eligible subject's refit was
//! not certified.
//!
//! Every run first runs the self-check:
//! - the metric code against closed-form cases, with bars of zero where the arithmetic is
//!   exact and derived rounding otherwise, and constant-hazard competing risks;
//! - the particle filter against the exact posterior of a static one-signature law (a
//!   Gauss-Hermite ratio of integrals), with two mutant negative controls that must fail;
//! - the forward paths against brute-force simulation;
//! - the weekly discretisation against a Gompertz incidence, deterministically, within its
//!   derived first-order bound, and the hospitalisation count against a halved step.
//! Monte Carlo bars are `z_{alpha / (2 m)}` standard errors over the `m` Monte Carlo cells,
//! with `alpha = SELF_CHECK_ALPHA`.
//!
//! On any mismatch, the program exits non-zero without benchmarking. `--self-check`
//! stops after the checks.
//!
//! # Running
//!
//! ```sh
//! cargo test --bench joint_absolute_risk -- --self-check
//! cargo test --bench joint_absolute_risk -- --scenario S3 --n-train 1000 --n-test 1000
//! ```
//!
//! `cargo test` builds this `harness = false` program with the optimised test profile
//! and runs its `main`. `--arms` takes a comma-separated subset of `oracle-hs`,
//! `oracle-hs-dx`, `oracle-state`, `joint-rank0`, `aladyn-map`, `aladyn-map-dense`,
//! `evh-dx-genetics`, `evh-dx-genetics-cc` and `evh-dx-genetics-channels`; `null` always runs. Scoring runs after the last oracle
//! and after each engine arm, over the subjects that every arm so far predicts.

use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

use gam::event_history::{
    CovariateSegment, Event, EventHistoryCohort, EventHistoryFit, FutureSegment, HistoryForecastRequest,
    MarkKind, PopulationForecastRequest, SubjectHistory, fit_event_history_formulas, forecast_history,
    population_forecast,
};
use gam::event_history::joint::fit_joint_event_model;
use gam::families::custom_family::BlockwiseFitOptions;
use ndarray::{Array1, Array2};
use opt::{Bfgs, BfgsError, FirstOrderSample, FusedObjective, MaxIterations, ObjectiveEvalError, Tolerance};
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};
use statrs::function::erf::erfc;

const SIGNATURES: usize = 3;
const SCORES: usize = 3;
const PCS: usize = 4;
const DISEASES: usize = 6;
/// Mark indices: the diseases, then the recurrent hospitalisation, then death.
const HOSPITAL: usize = DISEASES;
const DEATH: usize = DISEASES + 1;
const MARKS: usize = DISEASES + 2;
/// Scored targets: each disease, then death.
const DEATH_TARGET: usize = DISEASES;
const TARGETS: usize = DISEASES + 1;
const CHANNELS: usize = 5;
const LAB_A: usize = 0;
const LAB_B: usize = 1;
const SURVEY_BINARY: usize = 2;
const SURVEY_ORDINAL: usize = 3;
const CHANNEL_NAMES: [&str; CHANNELS] = ["lab1", "lab2", "survey_binary", "survey_ordinal", "count"];
const ORDINAL_LEVELS: usize = 4;
/// Declared family-wise false-alarm rate of the self-check's Monte Carlo checks: each bar is
/// `z_{alpha / (2 m)}` standard errors over the `m` Monte Carlo cells.
const SELF_CHECK_ALPHA: f64 = 1e-3;
/// A collapsed filter cloud reruns with twice the particles, at most this many times.
const FILTER_DOUBLINGS: usize = 3;
/// Poisson draws are sums of draws with mean at most this.
const POISSON_CHUNK: f64 = 256.0;
/// Gauss-Hermite order of the static-law posterior; the error is the change at twice it.
const STATIC_ORDER: usize = 40;
/// Gauss-Legendre order of the Gompertz incidence; the error is the change at twice it.
const GOMPERTZ_ORDER: usize = 32;

/// xoshiro256** seeded through splitmix64. It is deterministic across platforms and
/// independent of any crate's sampling API, so a seed names one cohort for good.
#[derive(Clone)]
struct Rng {
    state: [u64; 4],
}

impl Rng {
    fn new(seed: u64, stream: u64) -> Self {
        let mut mix = seed ^ stream.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        let mut state = [0u64; 4];
        for slot in state.iter_mut() {
            mix = mix.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut value = mix;
            value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *slot = value ^ (value >> 31);
        }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let shifted = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= shifted;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Uniform on the open interval (0, 1).
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn exponential(&mut self) -> f64 {
        -self.uniform().ln()
    }

    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        radius * (std::f64::consts::TAU * self.uniform()).cos()
    }

    fn student_t(&mut self, dof: usize) -> f64 {
        let numerator = self.normal();
        let mut chi_square = 0.0;
        let mut remaining = dof;
        while remaining > 0 {
            let draw = self.normal();
            chi_square += draw * draw;
            remaining -= 1;
        }
        numerator / (chi_square / dof as f64).sqrt()
    }

    /// A unit-scale gamma draw with an integer shape.
    fn gamma(&mut self, shape: usize) -> f64 {
        let mut total = 0.0;
        let mut remaining = shape;
        while remaining > 0 {
            total += self.exponential();
            remaining -= 1;
        }
        total
    }

    /// A Poisson draw as a sum of draws with mean at most `POISSON_CHUNK`, so Knuth's
    /// `exp(-mean)` limit never underflows.
    fn poisson(&mut self, mean: f64) -> usize {
        let mut remaining = mean;
        let mut count = 0;
        while remaining > 0.0 {
            let part = remaining.min(POISSON_CHUNK);
            remaining -= part;
            let limit = (-part).exp();
            let mut product = self.uniform();
            while product > limit {
                product *= self.uniform();
                count += 1;
            }
        }
        count
    }

    fn bernoulli(&mut self, probability: f64) -> bool {
        self.uniform() < probability
    }

    /// An index uniform on `0..n`, for `n > 0`.
    fn index(&mut self, n: usize) -> usize {
        ((self.uniform() * n as f64) as usize).min(n - 1)
    }
}

fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// `sigmoid(x)` by the same operations as [`sigmoid`], with its running rounding bound `mu`,
/// given the bound `mu_x` of `x`. The exponential charges its accuracy, cited from the runtime
/// libm (glibc 2.28 on MSI, < 1 ulp), times its value, plus the propagated operand bound.
fn sigmoid_bound(x: f64, mu_x: f64) -> (f64, f64) {
    if x >= 0.0 {
        let e = (-x).exp();
        let mu_e = e * mu_x + e;
        let sum = 1.0 + e;
        let mu_sum = mu_e + sum;
        let value = 1.0 / sum;
        (value, value * mu_sum / sum + value)
    } else {
        let e = x.exp();
        let mu_e = e * mu_x + e;
        let sum = 1.0 + e;
        let mu_sum = mu_e + sum;
        let value = e / sum;
        (value, (mu_e + value * mu_sum) / sum + value)
    }
}

/// A computed value with its running rounding bound (Higham, *Accuracy and Stability*, ch. 3).
/// Each operation adds its operands' propagated bounds and its own result's magnitude, so the
/// value lies within `eps * mu` of the exact result to first order. Used for the self-check's bars.
#[derive(Clone, Copy)]
struct Bounded {
    value: f64,
    mu: f64,
}

impl Bounded {
    fn exact(value: f64) -> Self {
        Self { value, mu: 0.0 }
    }

    fn add(self, other: Self) -> Self {
        let value = self.value + other.value;
        Self { value, mu: self.mu + other.mu + value.abs() }
    }

    fn sub(self, other: Self) -> Self {
        let value = self.value - other.value;
        Self { value, mu: self.mu + other.mu + value.abs() }
    }

    fn mul(self, other: Self) -> Self {
        let value = self.value * other.value;
        Self {
            value,
            mu: other.value.abs() * self.mu + self.value.abs() * other.mu + value.abs(),
        }
    }

    fn div(self, other: Self) -> Self {
        let value = self.value / other.value;
        Self {
            value,
            mu: (self.mu + value.abs() * other.mu) / other.value.abs() + value.abs(),
        }
    }

    fn negate(self) -> Self {
        Self { value: -self.value, mu: self.mu }
    }

    /// `exp`, charging its accuracy cited from the runtime libm (glibc 2.28 on MSI, < 1 ulp) times
    /// its value, plus the propagated operand bound `|f'| mu`.
    fn exp(self) -> Self {
        let value = self.value.exp();
        Self { value, mu: value * self.mu + value }
    }

    /// `ln`, with the cited libm charge and the propagated operand bound `mu / |x|`.
    fn ln(self) -> Self {
        let value = self.value.ln();
        Self { value, mu: self.mu / self.value.abs() + value.abs() }
    }

    /// `ln_1p`, charging its accuracy times its value, plus the propagated operand bound
    /// `mu / |1 + x|`. The accuracy is cited from glibc-2.28 sysdeps/ieee754/dbl-64/s_log1p.c:61-63
    /// (md5 cf019cc8): "the error is always less than 1 ulp". The x86_64 multiarch tree has no
    /// log1p variant.
    fn ln_1p(self) -> Self {
        let value = self.value.ln_1p();
        Self { value, mu: self.mu / (1.0 + self.value).abs() + value.abs() }
    }

    /// `exp_m1`, charging its accuracy times its value, plus the propagated operand bound
    /// `|e^x| mu`. The accuracy is cited from glibc-2.28 sysdeps/ieee754/dbl-64/s_expm1.c:96-98
    /// (md5 e60002aa): "the error is always less than 1 ulp".
    fn exp_m1(self) -> Self {
        let value = self.value.exp_m1();
        Self { value, mu: self.value.exp() * self.mu + value.abs() }
    }
}

/// Add a bounded term to gradient coordinate `index`, carrying the coordinate's running bound: the
/// term's own bound plus the new partial sum's magnitude.
fn accumulate(gradient: &mut [f64], bound: &mut [f64], index: usize, term: Bounded) {
    gradient[index] += term.value;
    bound[index] += term.mu + gradient[index].abs();
}

/// [`cumulative_dynamic_auc`]'s operations on bounded weights, over `(prediction, case, weight)`
/// rows sorted by prediction.
fn bounded_auc(rows: &[(f64, bool, Bounded)]) -> Bounded {
    let zero = Bounded::exact(0.0);
    let half = Bounded::exact(0.5);
    let (mut below, mut numerator, mut cases, mut controls) = (zero, zero, zero, zero);
    let mut index = 0;
    while index < rows.len() {
        let value = rows[index].0;
        let (mut case_weight, mut control_weight) = (zero, zero);
        while index < rows.len() && rows[index].0 == value {
            if rows[index].1 {
                case_weight = case_weight.add(rows[index].2);
            } else {
                control_weight = control_weight.add(rows[index].2);
            }
            index += 1;
        }
        numerator = numerator.add(case_weight.mul(below.add(half.mul(control_weight))));
        below = below.add(control_weight);
        cases = cases.add(case_weight);
        controls = controls.add(control_weight);
    }
    numerator.div(cases.mul(controls))
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

fn report(line: String) {
    let mut out = std::io::stdout().lock();
    if writeln!(out, "{line}").and_then(|()| out.flush()).is_err() {
        eprintln!("a12: stdout is closed");
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn standard_deviation(values: &[f64]) -> f64 {
    let centre = mean(values);
    let spread: f64 = values.iter().map(|value| (value - centre).powi(2)).sum();
    (spread / (values.len() as f64 - 1.0)).sqrt()
}


/// Newton's method from `start` on a polynomial evaluated with its derivative by `evaluate`,
/// until the correction stops shrinking or falls to rounding. Returns the root and the final
/// derivative.
fn polynomial_root(start: f64, evaluate: &dyn Fn(f64) -> (f64, f64)) -> (f64, f64) {
    let mut root = start;
    let mut previous = f64::INFINITY;
    loop {
        let (value, slope) = evaluate(root);
        let correction = value / slope;
        root -= correction;
        if !(correction.abs() < previous) || correction.abs() <= f64::EPSILON * root.abs() {
            return (root, evaluate(root).1);
        }
        previous = correction.abs();
    }
}

/// Gauss-Hermite nodes and weights for `integral exp(-x^2) f(x) dx`, from the orthonormal
/// Hermite recurrence with the classic asymptotic starting values (Numerical Recipes' gauher).
fn gauss_hermite(order: usize) -> Vec<(f64, f64)> {
    let n = order as f64;
    let evaluate = |x: f64| {
        let mut current = std::f64::consts::PI.powf(-0.25);
        let mut before = 0.0;
        for j in 1..=order {
            let next = x * (2.0 / j as f64).sqrt() * current - ((j as f64 - 1.0) / j as f64).sqrt() * before;
            before = current;
            current = next;
        }
        (current, (2.0 * n).sqrt() * before)
    };
    let mut roots: Vec<f64> = Vec::with_capacity(order / 2 + 1);
    let mut nodes = Vec::with_capacity(order);
    for i in 0..(order + 1) / 2 {
        let start = match i {
            0 => (2.0 * n + 1.0).sqrt() - 1.85575 * (2.0 * n + 1.0).powf(-1.0 / 6.0),
            1 => roots[0] - 1.14 * n.powf(0.426) / roots[0],
            2 => 1.86 * roots[1] - 0.86 * roots[0],
            3 => 1.91 * roots[2] - 0.91 * roots[1],
            _ => 2.0 * roots[i - 1] - roots[i - 2],
        };
        let (root, slope) = polynomial_root(start, &evaluate);
        roots.push(root);
        let weight = 2.0 / (slope * slope);
        nodes.push((root, weight));
        if 2 * i + 1 != order {
            nodes.push((-root, weight));
        }
    }
    nodes
}

/// Gauss-Legendre nodes and weights on [-1, 1].
fn gauss_legendre(order: usize) -> Vec<(f64, f64)> {
    let evaluate = |x: f64| {
        let mut current = 1.0;
        let mut before = 0.0;
        for j in 1..=order {
            let next = ((2 * j - 1) as f64 * x * current - (j - 1) as f64 * before) / j as f64;
            before = current;
            current = next;
        }
        (current, order as f64 * (x * current - before) / (x * x - 1.0))
    };
    let mut nodes = Vec::with_capacity(order);
    for i in 0..(order + 1) / 2 {
        let start = (std::f64::consts::PI * (i as f64 + 0.75) / (order as f64 + 0.5)).cos();
        let (root, slope) = polynomial_root(start, &evaluate);
        let weight = 2.0 / ((1.0 - root * root) * slope * slope);
        nodes.push((root, weight));
        if 2 * i + 1 != order {
            nodes.push((-root, weight));
        }
    }
    nodes
}

fn cholesky(matrix: [[f64; SCORES]; SCORES]) -> [[f64; SCORES]; SCORES] {
    let mut lower = [[0.0; SCORES]; SCORES];
    for i in 0..SCORES {
        for j in 0..=i {
            let mut value = matrix[i][j];
            for k in 0..j {
                value -= lower[i][k] * lower[j][k];
            }
            lower[i][j] = if i == j {
                value.sqrt()
            } else {
                value / lower[j][j]
            };
        }
    }
    lower
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Scenario {
    Joint,
    Null,
    EngineLaw,
    InformativeVisits,
}

impl Scenario {
    fn parse(text: &str) -> Option<Self> {
        match text {
            "S1" => Some(Self::Joint),
            "S2" => Some(Self::Null),
            "S3" => Some(Self::EngineLaw),
            "S4" => Some(Self::InformativeVisits),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Joint => "S1-joint",
            Self::Null => "S2-null",
            Self::EngineLaw => "S3-engine-law",
            Self::InformativeVisits => "S4-informative-visits",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum IntensityForm {
    /// `exp(baseline) (pi_m0 + sum_k pi_mk softplus(x_k))`.
    Softplus,
    /// `exp(baseline + gamma_m' g + a_m' x - |a_m|^2 / 2)`.
    LogLinear,
}

/// The synthetic population. Every number is a declared property of the simulated
/// truth, not a setting of any arm.
struct TruthLaw {
    scenario: Scenario,
    form: IntensityForm,
    /// Mean-reversion rate of each signature, per year; stationary variance is one.
    kappa: [f64; SIGNATURES],
    /// `B0[k][j]`: shift of signature k's mean per unit of score j at the reference age.
    genetic_drive: [[f64; SCORES]; SIGNATURES],
    /// `B1[k][j]`: change of `B0[k][j]` per `age_scale` years of age.
    genetic_drive_trend: [[f64; SCORES]; SIGNATURES],
    /// `u1[k]`: shift of signature k's mean for sex = 1.
    sex_drive: [f64; SIGNATURES],
    /// `u2[k]`: change of signature k's mean per `age_scale` years of age.
    drift_trend: [f64; SIGNATURES],
    /// Scores' dependence on the PCs.
    score_on_pcs: [[f64; PCS]; SCORES],
    /// Cholesky factor of the scores' residual covariance.
    score_cholesky: [[f64; SCORES]; SCORES],
    missing_genetics: f64,
    missing_genetics_high_pc1: f64,
    sex_share: f64,
    /// Log rate at the reference age, per year.
    log_rate: [f64; MARKS],
    /// Gompertz slope per year of age.
    age_slope: [f64; MARKS],
    sex_effect: [f64; MARKS],
    /// Decoder weights `(pi_m0, pi_m1, pi_m2, pi_m3)`, each row on the simplex.
    decoder: [[f64; SIGNATURES + 1]; MARKS],
    /// Log-linear loadings `a_m`.
    loading: [[f64; SIGNATURES]; MARKS],
    /// Log-linear score effects `gamma_m`.
    score_effect: [[f64; SCORES]; MARKS],
    /// State jump after each diagnosis.
    jump: [[f64; SIGNATURES]; DISEASES],
    lab_intercept: [f64; 2],
    lab_loading: [[f64; SIGNATURES]; 2],
    lab_scale: [f64; 2],
    lab_dof: usize,
    binary_intercept: f64,
    binary_loading: [f64; SIGNATURES],
    ordinal_loading: [f64; SIGNATURES],
    ordinal_thresholds: [f64; ORDINAL_LEVELS - 1],
    count_intercept: f64,
    count_loading: [f64; SIGNATURES],
    count_dispersion: usize,
    attendance_intercept: f64,
    attendance_sex: f64,
    attendance_pc2: f64,
    /// Attendance log-odds per unit of the first signature (non-zero only in S4).
    attendance_signature: f64,
    channel_missing: f64,
    /// Whether the path starts at `origin_age` (S1, S2, S4) or at entry (S3).
    pre_entry_path: bool,
    origin_age: f64,
    reference_age: f64,
    age_scale: f64,
    entry_age: [f64; 2],
    record_lookback: f64,
    follow_up: f64,
    maximum_age: f64,
    loss_rate: f64,
    steps_per_year: usize,
}

impl TruthLaw {
    fn new(scenario: Scenario) -> Self {
        let correlation = 0.3;
        let mut covariance = [[correlation; SCORES]; SCORES];
        for (i, row) in covariance.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        let mut law = Self {
            scenario,
            form: IntensityForm::Softplus,
            kappa: [0.1, 0.2, 0.5],
            genetic_drive: [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.2, 0.2, 0.4]],
            genetic_drive_trend: [[0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.1]],
            sex_drive: [0.3, -0.2, 0.0],
            drift_trend: [0.2, 0.1, 0.0],
            score_on_pcs: [
                [0.3, 0.0, 0.0, 0.0],
                [0.0, 0.3, 0.0, 0.0],
                [0.2, 0.0, 0.2, 0.0],
            ],
            score_cholesky: cholesky(covariance),
            missing_genetics: 0.1,
            missing_genetics_high_pc1: 0.3,
            sex_share: 0.5,
            log_rate: [0.008, 0.006, 0.005, 0.004, 0.006, 0.003, 0.08, 0.006].map(f64::ln),
            age_slope: [0.06, 0.05, 0.07, 0.04, 0.05, 0.08, 0.03, 0.09],
            sex_effect: [0.4, 0.0, 0.0, -0.3, 0.0, 0.0, 0.0, 0.2],
            decoder: [
                [0.4, 0.6, 0.0, 0.0],
                [0.3, 0.0, 0.7, 0.0],
                [0.5, 0.25, 0.0, 0.25],
                [0.2, 0.0, 0.4, 0.4],
                [0.6, 0.0, 0.0, 0.4],
                [0.4, 0.3, 0.3, 0.0],
                [0.4, 0.2, 0.2, 0.2],
                [0.5, 0.3, 0.0, 0.2],
            ],
            loading: [
                [0.6, 0.0, 0.0],
                [0.0, 0.6, 0.0],
                [0.3, 0.0, 0.3],
                [0.0, 0.4, 0.4],
                [0.0, 0.0, 0.5],
                [0.3, 0.3, 0.0],
                [0.2, 0.2, 0.2],
                [0.3, 0.0, 0.2],
            ],
            score_effect: [
                [0.3, 0.0, 0.0],
                [0.0, 0.3, 0.0],
                [0.0, 0.0, 0.3],
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
            ],
            jump: [
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.4],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            lab_intercept: [0.0, 0.0],
            lab_loading: [[0.8, 0.0, 0.0], [0.0, 0.6, 0.4]],
            lab_scale: [0.6, 0.6],
            lab_dof: 5,
            binary_intercept: -0.3,
            binary_loading: [0.0, 0.0, 0.7],
            ordinal_loading: [0.5, 0.0, 0.5],
            ordinal_thresholds: [-0.8, 0.2, 1.1],
            count_intercept: 0.5,
            count_loading: [0.3, 0.3, 0.0],
            count_dispersion: 4,
            attendance_intercept: 0.8,
            attendance_sex: 0.4,
            attendance_pc2: -0.3,
            attendance_signature: 0.0,
            channel_missing: 0.3,
            pre_entry_path: true,
            origin_age: 30.0,
            reference_age: 55.0,
            age_scale: 15.0,
            entry_age: [40.0, 70.0],
            record_lookback: 10.0,
            follow_up: 15.0,
            maximum_age: 85.0,
            loss_rate: 0.02,
            steps_per_year: 52,
        };
        match scenario {
            Scenario::Joint => {}
            Scenario::Null => {
                law.decoder = [[1.0, 0.0, 0.0, 0.0]; MARKS];
                law.genetic_drive = [[0.0; SCORES]; SIGNATURES];
                law.genetic_drive_trend = [[0.0; SCORES]; SIGNATURES];
                law.jump = [[0.0; SIGNATURES]; DISEASES];
            }
            Scenario::EngineLaw => {
                law.form = IntensityForm::LogLinear;
                law.genetic_drive = [[0.0; SCORES]; SIGNATURES];
                law.genetic_drive_trend = [[0.0; SCORES]; SIGNATURES];
                law.sex_drive = [0.0; SIGNATURES];
                law.drift_trend = [0.0; SIGNATURES];
                law.jump = [[0.0; SIGNATURES]; DISEASES];
                law.pre_entry_path = false;
                law.record_lookback = 0.0;
            }
            Scenario::InformativeVisits => {
                law.attendance_signature = 0.8;
            }
        }
        law
    }

    fn step_length(&self) -> f64 {
        1.0 / self.steps_per_year as f64
    }

    /// Steps from the origin age to `age`.
    fn step_of(&self, age: f64) -> usize {
        ((age - self.origin_age) * self.steps_per_year as f64).round() as usize
    }

    fn age_at(&self, step: usize) -> f64 {
        self.origin_age + step as f64 * self.step_length()
    }

    fn quantise(&self, age: f64) -> f64 {
        self.age_at(self.step_of(age))
    }

    fn draw_scores(&self, pcs: &[f64; PCS], rng: &mut Rng) -> [f64; SCORES] {
        let mut innovation = [0.0; SCORES];
        for value in innovation.iter_mut() {
            *value = rng.normal();
        }
        let mut scores = [0.0; SCORES];
        for (j, score) in scores.iter_mut().enumerate() {
            let mut value = 0.0;
            for (p, pc) in pcs.iter().enumerate() {
                value += self.score_on_pcs[j][p] * pc;
            }
            for (i, draw) in innovation.iter().enumerate() {
                value += self.score_cholesky[j][i] * draw;
            }
            *score = value;
        }
        scores
    }

    fn draw_person(&self, rng: &mut Rng) -> Person {
        let sex = if rng.bernoulli(self.sex_share) { 1.0 } else { 0.0 };
        let mut pcs = [0.0; PCS];
        for pc in pcs.iter_mut() {
            *pc = rng.normal();
        }
        let scores = self.draw_scores(&pcs, rng);
        let missing = if pcs[0] > 1.0 {
            self.missing_genetics_high_pc1
        } else {
            self.missing_genetics
        };
        let scores_observed = !rng.bernoulli(missing);
        let span = self.entry_age[1] - self.entry_age[0];
        let entry_age = self.quantise(self.entry_age[0] + span * rng.uniform());
        let record_start = self.quantise(entry_age - self.record_lookback * rng.uniform());
        Person {
            sex,
            pcs,
            scores,
            scores_observed,
            entry_age,
            record_start,
        }
    }

    fn drive_mean(&self, person: &Person, age: f64) -> [f64; SIGNATURES] {
        let trend = (age - self.reference_age) / self.age_scale;
        let mut mean = [0.0; SIGNATURES];
        for (k, slot) in mean.iter_mut().enumerate() {
            let mut value = self.sex_drive[k] * person.sex + self.drift_trend[k] * trend;
            for j in 0..SCORES {
                value += (self.genetic_drive[k][j] + self.genetic_drive_trend[k][j] * trend)
                    * person.scores[j];
            }
            *slot = value;
        }
        mean
    }

    /// The stationary state law at `age` given the drive, with no diagnoses.
    fn initial_state(&self, person: &Person, age: f64, rng: &mut Rng) -> MarkovState {
        let mean = self.drive_mean(person, age);
        let mut signatures = [0.0; SIGNATURES];
        for (k, slot) in signatures.iter_mut().enumerate() {
            *slot = mean[k] + rng.normal();
        }
        MarkovState {
            signatures,
            diagnosed: [false; DISEASES],
        }
    }

    fn intensity(&self, person: &Person, state: &MarkovState, mark: usize, age: f64) -> f64 {
        let baseline = self.log_rate[mark]
            + self.age_slope[mark] * (age - self.reference_age)
            + self.sex_effect[mark] * person.sex;
        match self.form {
            IntensityForm::Softplus => {
                let mut activity = self.decoder[mark][0];
                for k in 0..SIGNATURES {
                    activity += self.decoder[mark][k + 1] * softplus(state.signatures[k]);
                }
                baseline.exp() * activity
            }
            IntensityForm::LogLinear => {
                let mut predictor = baseline;
                let mut centring = 0.0;
                for k in 0..SIGNATURES {
                    predictor += self.loading[mark][k] * state.signatures[k];
                    centring += self.loading[mark][k] * self.loading[mark][k];
                }
                for j in 0..SCORES {
                    predictor += self.score_effect[mark][j] * person.scores[j];
                }
                (predictor - 0.5 * centring).exp()
            }
        }
    }

    /// The state's move over one step: OU diffusion toward the drive at the step's
    /// start, then the jumps of the diagnoses that fired in the step.
    fn transition(
        &self,
        person: &Person,
        state: &mut MarkovState,
        age: f64,
        fired: &[(f64, usize)],
        rng: &mut Rng,
    ) {
        let dt = self.step_length();
        let mean = self.drive_mean(person, age);
        for k in 0..SIGNATURES {
            let decay = (-self.kappa[k] * dt).exp();
            state.signatures[k] = mean[k]
                + decay * (state.signatures[k] - mean[k])
                + (1.0 - decay * decay).sqrt() * rng.normal();
        }
        for event in fired {
            let mark = event.1;
            if mark < DISEASES {
                state.diagnosed[mark] = true;
                for k in 0..SIGNATURES {
                    state.signatures[k] += self.jump[mark][k];
                }
            }
        }
    }

    /// One step from `age` to `age + dt`. Intensities are frozen at the step's start.
    /// Each mark at risk draws an independent exponential clock. Every disease and
    /// hospitalisation whose clock rings before death (or the step's end) fires.
    /// `fired` receives `(age, mark)` in time order; the return value is the death age,
    /// if death fired.
    fn step(
        &self,
        person: &Person,
        state: &mut MarkovState,
        age: f64,
        firing: Firing,
        rng: &mut Rng,
        fired: &mut Vec<(f64, usize)>,
    ) -> Option<f64> {
        let dt = self.step_length();
        fired.clear();
        let mut window = dt;
        let mut death = None;
        if firing.death {
            let clock = rng.exponential() / self.intensity(person, state, DEATH, age);
            if clock < dt {
                window = clock;
                death = Some(age + clock);
            }
        }
        for mark in 0..DEATH {
            if (mark < DISEASES && state.diagnosed[mark]) || firing.suppressed == Some(mark) {
                continue;
            }
            let clock = rng.exponential() / self.intensity(person, state, mark, age);
            if clock < window {
                fired.push((age + clock, mark));
            }
        }
        fired.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.transition(person, state, age, fired, rng);
        death
    }

    fn attendance_probability(&self, person: &Person, state: &MarkovState) -> f64 {
        sigmoid(
            self.attendance_intercept
                + self.attendance_sex * person.sex
                + self.attendance_pc2 * person.pcs[1]
                + self.attendance_signature * state.signatures[0],
        )
    }

    fn lab_location(&self, state: &MarkovState, lab: usize) -> f64 {
        let mut location = self.lab_intercept[lab];
        for k in 0..SIGNATURES {
            location += self.lab_loading[lab][k] * state.signatures[k];
        }
        location
    }

    fn linear(loading: &[f64; SIGNATURES], state: &MarkovState) -> f64 {
        loading
            .iter()
            .zip(state.signatures.iter())
            .map(|pair| pair.0 * pair.1)
            .sum()
    }

    fn binary_predictor(&self, state: &MarkovState) -> f64 {
        self.binary_intercept + Self::linear(&self.binary_loading, state)
    }

    fn count_mean(&self, state: &MarkovState) -> f64 {
        (self.count_intercept + Self::linear(&self.count_loading, state)).exp()
    }

    /// `P(level | state)` of the cumulative-probit survey.
    fn ordinal_probability(&self, state: &MarkovState, level: usize) -> f64 {
        let predictor = Self::linear(&self.ordinal_loading, state);
        let upper = if level + 1 < ORDINAL_LEVELS {
            normal_cdf(self.ordinal_thresholds[level] - predictor)
        } else {
            1.0
        };
        let lower = if level > 0 {
            normal_cdf(self.ordinal_thresholds[level - 1] - predictor)
        } else {
            0.0
        };
        upper - lower
    }

    fn measure_visit(&self, person: &Person, state: &MarkovState, age: f64, rng: &mut Rng) -> Visit {
        let attended = rng.bernoulli(self.attendance_probability(person, state));
        let mut values = [f64::NAN; CHANNELS];
        if attended {
            for (channel, slot) in values.iter_mut().enumerate() {
                if rng.bernoulli(self.channel_missing) {
                    continue;
                }
                *slot = match channel {
                    LAB_A | LAB_B => {
                        self.lab_location(state, channel) + self.lab_scale[channel] * rng.student_t(self.lab_dof)
                    }
                    SURVEY_BINARY => {
                        if self.binary_predictor(state) + rng.normal() > 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    SURVEY_ORDINAL => {
                        let latent = Self::linear(&self.ordinal_loading, state) + rng.normal();
                        self.ordinal_thresholds
                            .iter()
                            .filter(|threshold| latent > **threshold)
                            .count() as f64
                    }
                    _ => {
                        let dispersion = self.count_dispersion as f64;
                        let rate = rng.gamma(self.count_dispersion) * self.count_mean(state) / dispersion;
                        rng.poisson(rate) as f64
                    }
                };
            }
        }
        Visit {
            age,
            attended,
            values,
        }
    }

    /// The log likelihood of one visit given a state, up to terms free of the state: the
    /// attendance when it is informative, and every measured channel.
    fn visit_log_likelihood(&self, person: &Person, state: &MarkovState, visit: &Visit) -> f64 {
        let mut total = 0.0;
        if self.attendance_signature != 0.0 {
            let probability = self.attendance_probability(person, state);
            total += if visit.attended {
                probability.ln()
            } else {
                (1.0 - probability).ln()
            };
        }
        for (channel, value) in visit.values.iter().enumerate() {
            if value.is_nan() {
                continue;
            }
            total += match channel {
                LAB_A | LAB_B => {
                    let dof = self.lab_dof as f64;
                    let standardised = (value - self.lab_location(state, channel)) / self.lab_scale[channel];
                    -0.5 * (dof + 1.0) * (standardised * standardised / dof).ln_1p()
                }
                SURVEY_BINARY => {
                    let predictor = self.binary_predictor(state);
                    if *value > 0.5 {
                        normal_cdf(predictor).ln()
                    } else {
                        normal_cdf(-predictor).ln()
                    }
                }
                SURVEY_ORDINAL => self.ordinal_probability(state, *value as usize).ln(),
                _ => {
                    let rate = self.count_mean(state);
                    let dispersion = self.count_dispersion as f64;
                    dispersion * (dispersion / (dispersion + rate)).ln()
                        + value * (rate / (dispersion + rate)).ln()
                }
            };
        }
        total
    }

    /// One subject's uncensored truth, with an independent censoring age drawn
    /// alongside. Subjects who die before entry are never enrolled, so they are redrawn;
    /// the second value counts the redraws.
    fn draw_record(&self, rng: &mut Rng, offsets: &[f64]) -> (Record, usize) {
        let mut redrawn = 0;
        loop {
            let person = self.draw_person(rng);
            let entry_step = self.step_of(person.entry_age);
            let record_step = self.step_of(person.record_start);
            let start_step = if self.pre_entry_path { 0 } else { entry_step };
            let administrative_end = (person.entry_age + self.follow_up).min(self.maximum_age);
            let censor = (person.entry_age + rng.exponential() / self.loss_rate).min(administrative_end);
            let mut state = self.initial_state(&person, self.age_at(start_step), rng);
            let mut events = Vec::new();
            let mut visits = Vec::new();
            let mut death = None;
            let mut landmark_states = vec![None; offsets.len()];
            let mut fired = Vec::new();
            let firing = Firing {
                death: true,
                suppressed: None,
            };
            for n in start_step..self.step_of(administrative_end) {
                let age = self.age_at(n);
                if n >= record_step && (n - record_step) % self.steps_per_year == 0 {
                    visits.push(self.measure_visit(&person, &state, age, rng));
                }
                for (index, offset) in offsets.iter().enumerate() {
                    if n == self.step_of(person.entry_age + offset) {
                        landmark_states[index] = Some(state.clone());
                    }
                }
                let died = self.step(&person, &mut state, age, firing, rng, &mut fired);
                if n >= record_step {
                    events.extend(fired.iter().copied());
                }
                if let Some(time) = died {
                    death = Some(time);
                    break;
                }
            }
            if death.is_some_and(|time| time <= person.entry_age) {
                redrawn += 1;
                continue;
            }
            let exit = death.unwrap_or(administrative_end);
            visits.retain(|visit| visit.age < exit);
            let record = Record {
                person,
                events,
                death,
                exit,
                administrative_end,
                visits,
                landmark_states,
                censor,
            };
            return (record, redrawn);
        }
    }

    /// Per path, the Rao-Blackwellised `F_target(s, h)` at each absolute horizon under
    /// the true law, given the Markov state at `s`.
    ///
    /// Paths suppress the target (when it is a disease) and death, but carry every
    /// other mark and its jump. Within a step, the target fires before death with
    /// probability `lambda_d / Lambda (1 - exp(-Lambda dt))`, times the probability that
    /// neither fired earlier. That is exact for the discretised law, because intensities
    /// are frozen over a step and neither mark changes anything before it fires. A
    /// disease the state already carries has `F = 0`.
    fn forward_paths(
        &self,
        person: &Person,
        state: &MarkovState,
        s: f64,
        target: usize,
        horizons: &[f64],
        paths: usize,
        rng: &mut Rng,
    ) -> Vec<Vec<f64>> {
        if target < DISEASES && state.diagnosed[target] {
            return vec![vec![0.0; horizons.len()]; paths];
        }
        let dt = self.step_length();
        let start = self.step_of(s);
        let last = self.step_of(horizons[horizons.len() - 1]);
        let firing = Firing {
            death: false,
            suppressed: (target < DISEASES).then_some(target),
        };
        let mut fired = Vec::new();
        let mut result = Vec::with_capacity(paths);
        while result.len() < paths {
            let mut current = state.clone();
            let mut unresolved = 1.0;
            let mut incidence = 0.0;
            let mut values = Vec::with_capacity(horizons.len());
            for n in start..last {
                let age = self.age_at(n);
                let death_rate = self.intensity(person, &current, DEATH, age);
                let rate = if target < DISEASES {
                    self.intensity(person, &current, target, age)
                } else {
                    death_rate
                };
                let total = if target < DISEASES { rate + death_rate } else { death_rate };
                let leave = -(-total * dt).exp_m1();
                incidence += unresolved * rate / total * leave;
                unresolved *= 1.0 - leave;
                self.step(person, &mut current, age, firing, rng, &mut fired);
                while values.len() < horizons.len() && self.step_of(horizons[values.len()]) == n + 1 {
                    values.push(incidence);
                }
            }
            result.push(values);
        }
        result
    }

    /// The fraction of forward simulations of the full law from the Markov state at `s`
    /// in which the target fires (a disease before death) by each horizon: the
    /// brute-force counterpart of [`Self::forward_paths`].
    fn simulated_incidence(
        &self,
        person: &Person,
        state: &MarkovState,
        s: f64,
        target: usize,
        horizons: &[f64],
        paths: usize,
        rng: &mut Rng,
    ) -> Vec<f64> {
        let start = self.step_of(s);
        let last = self.step_of(horizons[horizons.len() - 1]);
        let firing = Firing {
            death: true,
            suppressed: None,
        };
        let mut fired = Vec::new();
        let mut counts = vec![0usize; horizons.len()];
        let mut done = 0;
        while done < paths {
            let mut current = state.clone();
            let mut onset = None;
            for n in start..last {
                let died = self.step(person, &mut current, self.age_at(n), firing, rng, &mut fired);
                if let Some(event) = fired.iter().find(|event| event.1 == target && target < DISEASES) {
                    onset = Some(event.0);
                    break;
                }
                if let Some(time) = died {
                    if target == DEATH_TARGET {
                        onset = Some(time);
                    }
                    break;
                }
            }
            if let Some(time) = onset {
                for (count, horizon) in counts.iter_mut().zip(horizons) {
                    if time <= *horizon {
                        *count += 1;
                    }
                }
            }
            done += 1;
        }
        counts.iter().map(|count| *count as f64 / paths as f64).collect()
    }

    /// Per forward simulation of the full law from the Markov state at `s`, the number of
    /// hospitalisations before `horizon` or death.
    fn simulated_hospitalisations(
        &self,
        person: &Person,
        state: &MarkovState,
        s: f64,
        horizon: f64,
        paths: usize,
        rng: &mut Rng,
    ) -> Vec<f64> {
        let firing = Firing {
            death: true,
            suppressed: None,
        };
        let mut fired = Vec::new();
        let mut counts = Vec::with_capacity(paths);
        while counts.len() < paths {
            let mut current = state.clone();
            let mut count = 0usize;
            for n in self.step_of(s)..self.step_of(horizon) {
                let died = self.step(person, &mut current, self.age_at(n), firing, rng, &mut fired);
                count += fired.iter().filter(|event| event.1 == HOSPITAL).count();
                if died.is_some() {
                    break;
                }
            }
            counts.push(count as f64);
        }
        counts
    }

    /// `F*` at every landmark, target and horizon for one subject: a particle filter on
    /// the true law given the subject's history, with the measurement process included
    /// when `measurements`.
    fn filtered_forecasts(
        &self,
        record: &Record,
        grid: &Grid,
        particles: usize,
        forecast_paths: usize,
        measurements: bool,
        mutation: Mutation,
        rng: &mut Rng,
    ) -> FilterOutcome {
        let cells = grid.offsets.len() * TARGETS * grid.horizons.len();
        let mut outcome = FilterOutcome {
            values: vec![f64::NAN; cells],
            minimum_ess: f64::NAN,
        };
        let person = &record.person;
        let dt = self.step_length();
        let entry_step = self.step_of(person.entry_age);
        let record_step = self.step_of(person.record_start);
        let start_step = if self.pre_entry_path { 0 } else { entry_step };
        let last_step = self.step_of(person.entry_age + grid.offsets[grid.offsets.len() - 1]);
        let mut cloud = Vec::with_capacity(particles);
        while cloud.len() < particles {
            let mut drawn = person.clone();
            if !drawn.scores_observed {
                drawn.scores = self.draw_scores(&drawn.pcs, rng);
            }
            let state = self.initial_state(&drawn, self.age_at(start_step), rng);
            cloud.push(Particle {
                person: drawn,
                state,
                log_weight: 0.0,
            });
        }
        let unobserved = Firing {
            death: false,
            suppressed: None,
        };
        let mut fired = Vec::new();
        let mut observed = Vec::new();
        let mut next_event = 0;
        let mut next_visit = 0;
        let mut minimum_ess = particles as f64;
        for n in start_step..=last_step {
            let age = self.age_at(n);
            for (li, offset) in grid.offsets.iter().enumerate() {
                let s = person.entry_age + offset;
                if n == self.step_of(s) && record.exit > s {
                    self.forecast_cloud(&cloud, record, s, li, grid, forecast_paths, rng, &mut outcome);
                }
            }
            if n == last_step || age >= record.exit {
                break;
            }
            if n < record_step {
                // Before the record start nothing is observed, but the subject lived.
                for particle in cloud.iter_mut() {
                    if mutation != Mutation::DropPreRecordSurvival {
                        particle.log_weight -= self.intensity(&particle.person, &particle.state, DEATH, age) * dt;
                    }
                    self.step(&particle.person, &mut particle.state, age, unobserved, rng, &mut fired);
                }
            } else {
                observed.clear();
                while next_event < record.events.len() && record.events[next_event].0 < age + dt {
                    observed.push(record.events[next_event]);
                    next_event += 1;
                }
                let mut visit = None;
                if next_visit < record.visits.len() && record.visits[next_visit].age < age + dt {
                    visit = Some(record.visits[next_visit]);
                    next_visit += 1;
                }
                for particle in cloud.iter_mut() {
                    if measurements && mutation != Mutation::DropVisitTerm {
                        if let Some(visit) = &visit {
                            particle.log_weight += self.visit_log_likelihood(&particle.person, &particle.state, visit);
                        }
                    }
                    particle.log_weight -= self.intensity(&particle.person, &particle.state, DEATH, age) * dt;
                    for mark in 0..DEATH {
                        let event = observed.iter().find(|event| event.1 == mark);
                        if mark < DISEASES && particle.state.diagnosed[mark] {
                            if event.is_some() {
                                particle.log_weight = f64::NEG_INFINITY;
                            }
                            continue;
                        }
                        let rate = self.intensity(&particle.person, &particle.state, mark, age);
                        particle.log_weight += match event {
                            Some(event) if mutation == Mutation::DropEventTerm => -rate * (event.0 - age),
                            Some(event) => rate.ln() - rate * (event.0 - age),
                            None => -rate * dt,
                        };
                    }
                    self.transition(&particle.person, &mut particle.state, age, &observed, rng);
                }
            }
            if cloud.iter().all(|particle| particle.log_weight == f64::NEG_INFINITY) {
                return outcome;
            }
            let ess = effective_sample_size(&cloud);
            minimum_ess = minimum_ess.min(ess);
            if ess < 0.5 * particles as f64 {
                resample(&mut cloud, rng);
            }
        }
        outcome.minimum_ess = minimum_ess;
        outcome
    }

    /// The weighted cloud's forecast at landmark `s`: the mean of one Rao-Blackwellised
    /// path per systematically resampled particle, for every target at risk. Resampled
    /// particles are correlated, so the standard error comes from independent replicate
    /// filters, not from this spread.
    fn forecast_cloud(
        &self,
        cloud: &[Particle],
        record: &Record,
        s: f64,
        li: usize,
        grid: &Grid,
        forecast_paths: usize,
        rng: &mut Rng,
        outcome: &mut FilterOutcome,
    ) {
        let horizons: Vec<f64> = grid.horizons.iter().map(|u| s + u).collect();
        let chosen = systematic_indices(&normalised_weights(cloud), forecast_paths, rng);
        for target in 0..TARGETS {
            if record.diagnosed_by(target, s) {
                continue;
            }
            let mut paths = Vec::with_capacity(chosen.len());
            for index in chosen.iter() {
                let particle = &cloud[*index];
                paths.extend(self.forward_paths(&particle.person, &particle.state, s, target, &horizons, 1, rng));
            }
            for ui in 0..horizons.len() {
                let column: Vec<f64> = paths.iter().map(|path| path[ui]).collect();
                let slot = (li * TARGETS + target) * grid.horizons.len() + ui;
                outcome.values[slot] = mean(&column);
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Person {
    sex: f64,
    pcs: [f64; PCS],
    scores: [f64; SCORES],
    scores_observed: bool,
    entry_age: f64,
    record_start: f64,
}

#[derive(Clone, Debug)]
struct MarkovState {
    signatures: [f64; SIGNATURES],
    diagnosed: [bool; DISEASES],
}

/// Which marks one step may fire. The forward paths suppress their target and death,
/// and integrate their competition analytically.
#[derive(Clone, Copy)]
struct Firing {
    death: bool,
    suppressed: Option<usize>,
}

/// A deliberate defect in the particle filter, used only by the self-check's negative controls.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mutation {
    Faithful,
    /// Drop `ln lambda` at an observed event.
    DropEventTerm,
    /// Drop the survival weight before the record start.
    DropPreRecordSurvival,
    /// Drop every visit's likelihood.
    DropVisitTerm,
}

/// One scheduled visit. A channel that was not measured holds NaN.
#[derive(Clone, Copy, Debug)]
struct Visit {
    age: f64,
    attended: bool,
    values: [f64; CHANNELS],
}

#[derive(Clone)]
struct Particle {
    person: Person,
    state: MarkovState,
    log_weight: f64,
}

struct FilterOutcome {
    /// `F*`, laid out `(landmark * TARGETS + target) * horizons + horizon`; NaN where
    /// the subject is not in the landmark set or the filter lost every particle.
    values: Vec<f64>,
    minimum_ess: f64,
}

fn normalised_weights(cloud: &[Particle]) -> Vec<f64> {
    let top = cloud
        .iter()
        .map(|particle| particle.log_weight)
        .fold(f64::NEG_INFINITY, f64::max);
    let raw: Vec<f64> = cloud
        .iter()
        .map(|particle| (particle.log_weight - top).exp())
        .collect();
    let total: f64 = raw.iter().sum();
    raw.into_iter().map(|weight| weight / total).collect()
}

fn effective_sample_size(cloud: &[Particle]) -> f64 {
    let weights = normalised_weights(cloud);
    1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>()
}

/// `count` systematic draws of indices from normalised `weights`.
fn systematic_indices(weights: &[f64], count: usize, rng: &mut Rng) -> Vec<usize> {
    let offset = rng.uniform() / count as f64;
    let mut chosen = Vec::with_capacity(count);
    let mut source = 0;
    let mut cumulative = weights[0];
    while chosen.len() < count {
        let target = offset + chosen.len() as f64 / count as f64;
        while cumulative < target && source + 1 < weights.len() {
            source += 1;
            cumulative += weights[source];
        }
        chosen.push(source);
    }
    chosen
}

fn resample(cloud: &mut Vec<Particle>, rng: &mut Rng) {
    let chosen = systematic_indices(&normalised_weights(cloud), cloud.len(), rng);
    let next: Vec<Particle> = chosen
        .iter()
        .map(|index| {
            let mut particle = cloud[*index].clone();
            particle.log_weight = 0.0;
            particle
        })
        .collect();
    *cloud = next;
}

struct Record {
    person: Person,
    /// Observed diseases and hospitalisations `(age, mark)` from the record start, in time
    /// order.
    events: Vec<(f64, usize)>,
    death: Option<f64>,
    exit: f64,
    /// Age 85 or fifteen years after entry.
    administrative_end: f64,
    /// Scheduled visits before exit, in time order.
    visits: Vec<Visit>,
    /// The true Markov state at each landmark the subject reached alive (uncensored truth
    /// only).
    landmark_states: Vec<Option<MarkovState>>,
    /// The independent censoring age: loss to follow-up or the administrative end.
    censor: f64,
}

impl Record {
    fn first_occurrence(&self, mark: usize) -> Option<f64> {
        self.events
            .iter()
            .find(|event| event.1 == mark)
            .map(|event| event.0)
    }

    /// The age at which a target first happened in the records: a disease's first
    /// record, or death.
    fn target_time(&self, target: usize) -> Option<f64> {
        if target < DISEASES {
            self.first_occurrence(target)
        } else {
            self.death
        }
    }

    fn diagnosed_by(&self, target: usize, age: f64) -> bool {
        target < DISEASES && self.first_occurrence(target).is_some_and(|time| time <= age)
    }

    fn in_landmark_set(&self, target: usize, s: f64) -> bool {
        self.exit > s && !self.diagnosed_by(target, s)
    }

    /// Whether follow-up reached the administrative end, which lies after every horizon.
    fn reached_administrative_end(&self) -> bool {
        self.death.is_none() && self.exit >= self.administrative_end
    }

    /// The record as observed under its independent censoring age.
    fn censored(&self) -> Record {
        let death = self.death.filter(|time| *time < self.censor);
        let exit = death.unwrap_or(self.exit.min(self.censor));
        Record {
            person: self.person.clone(),
            events: self.events.iter().copied().filter(|event| event.0 < exit).collect(),
            death,
            exit,
            administrative_end: self.administrative_end,
            visits: self.visits.iter().copied().filter(|visit| visit.age < exit).collect(),
            landmark_states: Vec::new(),
            censor: self.censor,
        }
    }
}

fn draw_cohort(law: &TruthLaw, seed: u64, tag: u64, n: usize, offsets: &[f64]) -> (Vec<Record>, usize) {
    let drawn: Vec<(Record, usize)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut rng = Rng::new(seed, (tag << 32) | i as u64);
            law.draw_record(&mut rng, offsets)
        })
        .collect();
    let redrawn = drawn.iter().map(|pair| pair.1).sum();
    (drawn.into_iter().map(|pair| pair.0).collect(), redrawn)
}

fn describe(name: &str, records: &[Record], redrawn: usize) {
    let mut prevalent = [0usize; DISEASES];
    let mut incident = [0usize; DISEASES];
    for record in records {
        for (d, slot) in incident.iter_mut().enumerate() {
            if let Some(time) = record.first_occurrence(d) {
                if time <= record.person.entry_age {
                    prevalent[d] += 1;
                } else {
                    *slot += 1;
                }
            }
        }
    }
    let hospitalisations = records
        .iter()
        .map(|record| record.events.iter().filter(|event| event.1 == HOSPITAL).count())
        .sum::<usize>();
    let deaths = records.iter().filter(|record| record.death.is_some()).count();
    let administrative = records.iter().filter(|record| record.reached_administrative_end()).count();
    let follow_up: Vec<f64> = records.iter().map(|record| record.exit - record.person.entry_age).collect();
    let attended: Vec<f64> = records
        .iter()
        .map(|record| record.visits.iter().filter(|visit| visit.attended).count() as f64)
        .collect();
    let missing = records.iter().filter(|record| !record.person.scores_observed).count();
    report(format!(
        "[a12-cohort] cohort={name} n={} redrawn_before_entry={redrawn} prevalent={prevalent:?} incident={incident:?} \
         hospitalisations={hospitalisations} deaths={deaths} administrative_end={administrative} lost={} \
         mean_follow_up={:.3} mean_attended_visits={:.2} genetics_missing={missing}",
        records.len(),
        records.len() - deaths - administrative,
        mean(&follow_up),
        mean(&attended),
    ));
}

/// Engine inputs for one arm.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Inputs {
    MeanImputed,
    CompleteCase,
    Channels,
}

impl Inputs {
    fn arm(self) -> &'static str {
        match self {
            Self::MeanImputed => "evh-dx-genetics",
            Self::CompleteCase => "evh-dx-genetics-cc",
            Self::Channels => "evh-dx-genetics-channels",
        }
    }

    fn admits(self, record: &Record) -> bool {
        self != Self::CompleteCase || record.person.scores_observed
    }

    fn covariate_names(self) -> Vec<String> {
        let mut names: Vec<String> = ["sex", "pc1", "pc2", "pc3", "pc4", "g1", "g2", "g3"]
            .into_iter()
            .map(String::from)
            .collect();
        if self != Self::CompleteCase {
            names.push("g_missing".to_string());
        }
        if self == Self::Channels {
            for channel in CHANNEL_NAMES {
                names.push(channel.to_string());
                names.push(format!("ever_{channel}"));
            }
        }
        names
    }

    fn formula(self) -> String {
        format!("s(time) + {}", self.covariate_names().join(" + "))
    }

    /// The covariate rows of `record`'s follow-up before `until`: the row at entry, and
    /// with channels a new row at every later visit that measured something.
    fn rows(self, record: &Record, until: f64) -> Vec<(f64, Vec<f64>)> {
        let person = &record.person;
        let observed = if person.scores_observed { 1.0 } else { 0.0 };
        let mut baseline = vec![person.sex];
        baseline.extend(person.pcs);
        baseline.extend(person.scores.map(|score| observed * score));
        if self != Self::CompleteCase {
            baseline.push(1.0 - observed);
        }
        if self != Self::Channels {
            return vec![(person.entry_age, baseline)];
        }
        let row = |last: &[f64; CHANNELS], ever: &[f64; CHANNELS]| {
            let mut values = baseline.clone();
            for channel in 0..CHANNELS {
                values.push(last[channel]);
                values.push(ever[channel]);
            }
            values
        };
        let mut last = [0.0; CHANNELS];
        let mut ever = [0.0; CHANNELS];
        let mut index = 0;
        while index < record.visits.len() && record.visits[index].age <= person.entry_age {
            carry_forward(&mut last, &mut ever, &record.visits[index]);
            index += 1;
        }
        let mut rows = vec![(person.entry_age, row(&last, &ever))];
        while index < record.visits.len() && record.visits[index].age < until {
            if carry_forward(&mut last, &mut ever, &record.visits[index]) {
                rows.push((record.visits[index].age, row(&last, &ever)));
            }
            index += 1;
        }
        rows
    }
}

/// Carry a visit's measured channels forward; whether it measured anything.
fn carry_forward(last: &mut [f64; CHANNELS], ever: &mut [f64; CHANNELS], visit: &Visit) -> bool {
    let mut measured = false;
    for (channel, value) in visit.values.iter().enumerate() {
        if !value.is_nan() {
            last[channel] = *value;
            ever[channel] = 1.0;
            measured = true;
        }
    }
    measured
}

/// `record`'s history up to `until`, with covariate segments indexing rows from
/// `row_offset`. Events from the record start before entry are prior history.
fn subject_history(
    record: &Record,
    id: String,
    until: f64,
    rows: &[(f64, Vec<f64>)],
    row_offset: usize,
) -> SubjectHistory {
    let mut events: Vec<Event> = record
        .events
        .iter()
        .filter(|event| event.0 <= until)
        .map(|event| Event {
            time: event.0,
            mark: event.1,
        })
        .collect();
    if let Some(time) = record.death.filter(|time| *time <= until) {
        events.push(Event { time, mark: DEATH });
    }
    let segments = rows
        .iter()
        .enumerate()
        .map(|(index, row)| CovariateSegment {
            start: row.0,
            row: row_offset + index,
        })
        .collect();
    SubjectHistory {
        id,
        entry: record.person.entry_age,
        exit: record.exit.min(until),
        events,
        segments,
    }
}

fn row_table(rows: &[(f64, Vec<f64>)], columns: usize) -> Result<Array2<f64>, String> {
    let mut values = Vec::with_capacity(rows.len() * columns);
    for row in rows {
        values.extend_from_slice(&row.1);
    }
    Array2::from_shape_vec((rows.len(), columns), values).map_err(|problem| problem.to_string())
}

/// The marks every model sees: the diseases, the recurrent hospitalisation, then death.
fn mark_vocabulary() -> (Vec<String>, Vec<MarkKind>) {
    let mut names: Vec<String> = (1..=DISEASES).map(|d| format!("d{d}")).collect();
    names.push("hospital".to_string());
    names.push("death".to_string());
    let mut kinds = vec![MarkKind::Once; DISEASES];
    kinds.push(MarkKind::Recurrent);
    kinds.push(MarkKind::Terminal);
    (names, kinds)
}

fn engine_cohort(records: &[Record], inputs: Inputs) -> Result<EventHistoryCohort, String> {
    let names = inputs.covariate_names();
    let mut all_rows = Vec::new();
    let mut subjects = Vec::new();
    for (index, record) in records.iter().enumerate() {
        if !inputs.admits(record) {
            continue;
        }
        let rows = inputs.rows(record, record.exit);
        subjects.push(subject_history(record, format!("train{index}"), record.exit, &rows, all_rows.len()));
        all_rows.extend(rows);
    }
    let (mark_names, mark_kinds) = mark_vocabulary();
    Ok(EventHistoryCohort {
        mark_names,
        mark_kinds,
        covariate_levels: vec![Vec::new(); names.len()],
        covariates: row_table(&all_rows, names.len())?,
        covariate_names: names,
        subjects,
    })
}

struct Grid {
    /// Landmarks, in years after entry.
    offsets: Vec<f64>,
    horizons: Vec<f64>,
}

/// One arm's predictions, NaN where it has none.
struct Predictions {
    arm: String,
    horizons: usize,
    subjects: usize,
    values: Vec<f64>,
    /// Whether the arm forecasts each cell at all, laid out `landmark * TARGETS + target`. A
    /// cell it does not produce is reported as not produced, and never shrinks the other
    /// arms' matched subjects.
    produced: Vec<bool>,
    /// Each prediction's Monte Carlo standard error; NaN for an arm without Monte Carlo error.
    errors: Vec<f64>,
}

impl Predictions {
    fn new(arm: &str, grid: &Grid, subjects: usize) -> Self {
        Self {
            arm: arm.to_string(),
            horizons: grid.horizons.len(),
            subjects,
            values: vec![f64::NAN; grid.offsets.len() * TARGETS * grid.horizons.len() * subjects],
            produced: vec![true; grid.offsets.len() * TARGETS],
            errors: vec![f64::NAN; grid.offsets.len() * TARGETS * grid.horizons.len() * subjects],
        }
    }

    fn producing(&self, landmark: usize, target: usize) -> bool {
        self.produced[landmark * TARGETS + target]
    }

    fn error(&self, landmark: usize, target: usize, horizon: usize, subject: usize) -> f64 {
        self.errors[self.slot(landmark, target, horizon, subject)]
    }

    fn slot(&self, landmark: usize, target: usize, horizon: usize, subject: usize) -> usize {
        ((landmark * TARGETS + target) * self.horizons + horizon) * self.subjects + subject
    }

    fn get(&self, landmark: usize, target: usize, horizon: usize, subject: usize) -> f64 {
        self.values[self.slot(landmark, target, horizon, subject)]
    }

    fn set(&mut self, landmark: usize, target: usize, horizon: usize, subject: usize, value: f64) {
        let slot = self.slot(landmark, target, horizon, subject);
        self.values[slot] = value;
    }
}

fn target_name(target: usize) -> String {
    if target < DISEASES {
        format!("d{}", target + 1)
    } else {
        "death".to_string()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Transition {
    Target,
    Death,
    Censored,
}

/// The first exit after the landmark of every subject in the landmark set, as
/// `(years since entry, transition, record index)` in time order.
fn landmark_exits(records: &[Record], offset: f64, target: usize) -> Vec<(f64, Transition, usize)> {
    let mut exits = Vec::new();
    for (index, record) in records.iter().enumerate() {
        let entry = record.person.entry_age;
        if !record.in_landmark_set(target, entry + offset) {
            continue;
        }
        let (time, transition) = match (record.target_time(target), record.death) {
            (Some(time), _) => (time, Transition::Target),
            (None, Some(time)) => (time, Transition::Death),
            (None, None) => (record.exit, Transition::Censored),
        };
        exits.push((time - entry, transition, index));
    }
    exits.sort_by(|a, b| a.0.total_cmp(&b.0));
    exits
}

/// The Aalen-Johansen cumulative incidence of the target through `horizon` (years since
/// entry) over time-ordered exits, leaving out the subject whose record index is `skip`.
fn incidence_through(exits: &[(f64, Transition, usize)], horizon: f64, skip: Option<usize>) -> f64 {
    let mut at_risk = exits.iter().filter(|exit| Some(exit.2) != skip).count() as f64;
    let mut survival = 1.0;
    let mut incidence = 0.0;
    let mut index = 0;
    while index < exits.len() && exits[index].0 <= horizon {
        let time = exits[index].0;
        let mut targets = 0.0;
        let mut deaths = 0.0;
        let mut censored = 0.0;
        while index < exits.len() && exits[index].0 == time {
            if Some(exits[index].2) != skip {
                match exits[index].1 {
                    Transition::Target => targets += 1.0,
                    Transition::Death => deaths += 1.0,
                    Transition::Censored => censored += 1.0,
                }
            }
            index += 1;
        }
        if at_risk > 0.0 {
            incidence += survival * targets / at_risk;
            survival *= 1.0 - (targets + deaths) / at_risk;
        }
        at_risk -= targets + deaths + censored;
    }
    incidence
}

/// The landmark Aalen-Johansen `F_target(s, s+u)` at each horizon `u`.
fn aalen_johansen(records: &[Record], offset: f64, target: usize, horizons: &[f64]) -> Vec<f64> {
    let exits = landmark_exits(records, offset, target);
    horizons
        .iter()
        .map(|u| incidence_through(&exits, offset + u, None))
        .collect()
}

/// Jackknife pseudo-observations `n F - (n - 1) F^(-i)` of the landmark Aalen-Johansen
/// `F_target(s, s+u)` over the landmark set of `records`, for `subjects` (record
/// indices in that set).
fn pseudo_observations(records: &[Record], offset: f64, target: usize, horizon: f64, subjects: &[usize]) -> Vec<f64> {
    let exits = landmark_exits(records, offset, target);
    let n = exits.len() as f64;
    let full = incidence_through(&exits, horizon, None);
    subjects
        .iter()
        .map(|subject| n * full - (n - 1.0) * incidence_through(&exits, horizon, Some(*subject)))
        .collect()
}

fn null_arm(train: &[Record], test: &[Record], grid: &Grid) -> Predictions {
    let mut table = Predictions::new("null", grid, test.len());
    for (li, offset) in grid.offsets.iter().enumerate() {
        for target in 0..TARGETS {
            let incidence = aalen_johansen(train, *offset, target, &grid.horizons);
            for (i, record) in test.iter().enumerate() {
                if record.in_landmark_set(target, record.person.entry_age + offset) {
                    for (ui, value) in incidence.iter().enumerate() {
                        table.set(li, target, ui, i, *value);
                    }
                }
            }
        }
    }
    table
}

fn state_oracle_arm(law: &TruthLaw, test: &[Record], grid: &Grid, seed: u64, paths: usize) -> Predictions {
    let started = Instant::now();
    let cells: Vec<Vec<(usize, usize, usize, f64, f64)>> = test
        .par_iter()
        .enumerate()
        .map(|(i, record)| {
            let mut values = Vec::new();
            for (li, offset) in grid.offsets.iter().enumerate() {
                let Some(state) = &record.landmark_states[li] else {
                    continue;
                };
                let s = record.person.entry_age + offset;
                let horizons: Vec<f64> = grid.horizons.iter().map(|u| s + u).collect();
                for target in 0..TARGETS {
                    if record.diagnosed_by(target, s) {
                        continue;
                    }
                    let stream = ((i * grid.offsets.len() + li) * TARGETS + target) as u64;
                    let mut rng = Rng::new(seed, (3 << 32) | stream);
                    let samples = law.forward_paths(&record.person, state, s, target, &horizons, paths, &mut rng);
                    for ui in 0..horizons.len() {
                        let column: Vec<f64> = samples.iter().map(|path| path[ui]).collect();
                        values.push((li, target, ui, mean(&column), standard_deviation(&column) / (paths as f64).sqrt()));
                    }
                }
            }
            values
        })
        .collect();
    let mut table = Predictions::new("oracle-state", grid, test.len());
    for (i, values) in cells.iter().enumerate() {
        for value in values {
            table.set(value.0, value.1, value.2, i, value.3);
            let slot = table.slot(value.0, value.1, value.2, i);
            table.errors[slot] = value.4;
        }
    }
    report(format!(
        "[a12-arm] arm=oracle-state seconds={:.1} paths={paths}",
        started.elapsed().as_secs_f64()
    ));
    table
}

fn filtered_oracle_arm(
    law: &TruthLaw,
    test: &[Record],
    grid: &Grid,
    options: &Options,
    measurements: bool,
) -> Predictions {
    let started = Instant::now();
    let arm = if measurements { "oracle-hs" } else { "oracle-hs-dx" };
    let tag = if measurements { 5u64 } else { 6u64 };
    let replicates = options.filter_replicates;
    let cells = grid.offsets.len() * TARGETS * grid.horizons.len();
    // Independent replicate filters with independent seeds: resampled particles are
    // correlated, so the within-cloud spread would understate the Monte Carlo error.
    // A replicate whose cloud collapsed (every particle impossible) reruns with twice the
    // particles, at most FILTER_DOUBLINGS times. A subject still lost stays unpredicted.
    let outcomes: Vec<Vec<(FilterOutcome, usize)>> = test
        .par_iter()
        .enumerate()
        .map(|(i, record)| {
            (0..replicates)
                .map(|r| {
                    let mut particles = options.particles;
                    let mut doubling = 0usize;
                    loop {
                        let stream = (tag << 56) | ((i as u64) << 24) | ((doubling as u64) << 16) | r as u64;
                        let mut rng = Rng::new(options.seed, stream);
                        let run = law.filtered_forecasts(
                            record,
                            grid,
                            particles,
                            options.forecast_paths,
                            measurements,
                            Mutation::Faithful,
                            &mut rng,
                        );
                        if !run.minimum_ess.is_nan() || doubling == FILTER_DOUBLINGS {
                            return (run, doubling);
                        }
                        particles *= 2;
                        doubling += 1;
                    }
                })
                .collect()
        })
        .collect();
    let mut table = Predictions::new(arm, grid, test.len());
    let mut errors = Vec::new();
    for (i, runs) in outcomes.iter().enumerate() {
        for cell in 0..cells {
            let estimates: Vec<f64> = runs.iter().map(|run| run.0.values[cell]).collect();
            if estimates.iter().all(|value| value.is_finite()) {
                let error = standard_deviation(&estimates) / (replicates as f64).sqrt();
                table.values[cell * test.len() + i] = mean(&estimates);
                table.errors[cell * test.len() + i] = error;
                errors.push(error);
            }
        }
        let ess: Vec<String> = runs.iter().map(|run| format!("{:.1}@{}", run.0.minimum_ess, run.1)).collect();
        report(format!("[a12-ess] arm={arm} subject={i} minimum_ess_at_doublings=[{}]", ess.join(",")));
    }
    let lost = outcomes.iter().flatten().filter(|run| run.0.minimum_ess.is_nan()).count();
    let rescued = outcomes
        .iter()
        .flatten()
        .filter(|run| run.1 > 0 && !run.0.minimum_ess.is_nan())
        .count();
    let ess: Vec<f64> = outcomes
        .iter()
        .flatten()
        .map(|run| run.0.minimum_ess)
        .filter(|value| value.is_finite())
        .collect();
    report(format!(
        "[a12-arm] arm={arm} seconds={:.1} particles={} forecast_paths={} replicates={replicates} \
         lost_replicate_clouds={lost} rescued_replicate_clouds={rescued} mean_minimum_ess={:.1} lowest_minimum_ess={:.1} \
         mean_mc_se={:.5} max_mc_se={:.5}",
        started.elapsed().as_secs_f64(),
        options.particles,
        options.forecast_paths,
        mean(&ess),
        ess.iter().copied().fold(f64::INFINITY, f64::min),
        mean(&errors),
        errors.iter().copied().fold(0.0, f64::max),
    ));
    table
}

fn forecast_test(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    test: &[Record],
    grid: &Grid,
    inputs: Inputs,
) -> Result<Predictions, String> {
    let started = Instant::now();
    let columns = cohort.covariates.ncols();
    let mut table = Predictions::new(inputs.arm(), grid, test.len());
    let mut requests = 0usize;
    let mut refused = 0usize;
    let mut first_refusal = None;
    for (i, record) in test.iter().enumerate() {
        if !inputs.admits(record) {
            continue;
        }
        for (li, offset) in grid.offsets.iter().enumerate() {
            let s = record.person.entry_age + offset;
            if record.exit <= s {
                continue;
            }
            let rows = inputs.rows(record, s);
            let horizons: Vec<f64> = grid.horizons.iter().map(|u| s + u).collect();
            requests += 1;
            let forecast = if *offset <= 0.0 {
                // No follow-up to filter at entry: the engine's population forecast for an
                // entrant with these covariates (stationary latent prior, every mark at risk).
                let future = [FutureSegment {
                    start: s,
                    covariates: rows[rows.len() - 1].1.clone(),
                }];
                let request = PopulationForecastRequest {
                    start: s,
                    stratum: 0,
                    horizons: &horizons,
                    future: &future,
                };
                population_forecast(fit, cohort, &request)
            } else {
                let covariates = row_table(&rows, columns)?;
                let history = subject_history(record, format!("test{i}"), s, &rows, 0);
                let request = HistoryForecastRequest {
                    history: &history,
                    covariates: covariates.view(),
                    stratum: 0,
                    horizons: &horizons,
                    future: &[],
                };
                forecast_history(fit, cohort, &request)
            };
            match forecast {
                Ok(forecast) => {
                    for target in 0..TARGETS {
                        if record.diagnosed_by(target, s) {
                            continue;
                        }
                        for ui in 0..horizons.len() {
                            let value = if target < DISEASES {
                                forecast.expected_counts[[ui, target]]
                            } else {
                                1.0 - forecast.survival[ui]
                            };
                            table.set(li, target, ui, i, value);
                        }
                    }
                }
                Err(error) => {
                    refused += 1;
                    if first_refusal.is_none() {
                        first_refusal = Some(error.to_string());
                    }
                }
            }
        }
    }
    report(format!(
        "[a12-forecast] arm={} seconds={:.1} requests={requests} refused={refused} first_refusal={:?}",
        inputs.arm(),
        started.elapsed().as_secs_f64(),
        first_refusal.unwrap_or_default(),
    ));
    Ok(table)
}

/// The joint event model's rank-zero slice (`fit_joint_event_model`, then `condition` and
/// `forecast`) on the censored training histories: constant rates, no covariates and no
/// latent state, so a floor arm. Its histories start at enrolment, like the engine's, so
/// it has no forecast at the entry landmark, which is reported as not produced.
fn joint_rank_zero_arm(train: &[Record], test: &[Record], grid: &Grid) -> Result<Option<Predictions>, String> {
    let (mark_names, mark_kinds) = mark_vocabulary();
    let subjects: Vec<SubjectHistory> = train
        .iter()
        .enumerate()
        .map(|(index, record)| subject_history(record, format!("train{index}"), record.exit, &[], 0))
        .collect();
    report(format!("[a12-fit] arm=joint-rank0 status=started subjects={}", subjects.len()));
    let started = Instant::now();
    let model = match fit_joint_event_model(mark_names, mark_kinds, &subjects) {
        Ok(model) => model,
        Err(error) => {
            report(format!(
                "[a12-fit] arm=joint-rank0 status=refused seconds={:.1} error={error}",
                started.elapsed().as_secs_f64()
            ));
            return Ok(None);
        }
    };
    report(format!("[a12-fit] arm=joint-rank0 status=fitted seconds={:.1}", started.elapsed().as_secs_f64()));
    let started = Instant::now();
    let mut table = Predictions::new("joint-rank0", grid, test.len());
    let mut requests = 0usize;
    let mut refused = 0usize;
    let mut first_refusal = None;
    let mut largest_error = 0.0f64;
    for (li, offset) in grid.offsets.iter().enumerate() {
        if *offset <= 0.0 {
            table.produced[li * TARGETS..(li + 1) * TARGETS].fill(false);
            continue;
        }
        for (i, record) in test.iter().enumerate() {
            let s = record.person.entry_age + offset;
            if record.exit <= s {
                continue;
            }
            let history = subject_history(record, format!("test{i}"), s, &[], 0);
            requests += 1;
            match model.condition(&history).and_then(|conditioned| conditioned.forecast(&grid.horizons)) {
                Ok(forecast) => {
                    for target in 0..TARGETS {
                        if record.diagnosed_by(target, s) {
                            continue;
                        }
                        let mark = if target < DISEASES { target } else { DEATH };
                        for ui in 0..grid.horizons.len() {
                            table.set(li, target, ui, i, forecast.incidence[[ui, mark]]);
                            largest_error = largest_error.max(forecast.incidence_error[[ui, mark]]);
                        }
                    }
                }
                Err(error) => {
                    refused += 1;
                    if first_refusal.is_none() {
                        first_refusal = Some(error.to_string());
                    }
                }
            }
        }
    }
    report(format!(
        "[a12-forecast] arm=joint-rank0 seconds={:.1} requests={requests} refused={refused} \
         largest_incidence_error={largest_error:.3e} first_refusal={:?}",
        started.elapsed().as_secs_f64(),
        first_refusal.unwrap_or_default(),
    ));
    Ok(Some(table))
}


/// The published objective of the ALADYNOULLI-style baseline (surbut/aladynoulli2 @ 563f6cd24fe6,
/// pyScripts_forPublish/clust_huge_amp_vectorized.py :221-350) on yearly age bins:
/// `(1/N) sum data loss + W [ (1/(2N)) sum_{n,k} v' K_lambda^-1 v + (1/(2D)) sum_{k,d} w' K_phi^-1 w ]
///  + (decay / 2) |gamma|^2`, with `v = lambda_nk - genetic_scale G_n gamma_k` and
/// `w = phi_kd - logit prevalence_d - psi_kd`. Parameters are laid out `lambda (N K T), phi (K D T),
/// psi (K D), gamma (P K), kappa_raw`, with `kappa = exp(kappa_raw)`.
struct AladynProblem {
    subjects: usize,
    signatures: usize,
    bins: usize,
    covariates: usize,
    /// Standardised covariates, `N x P`.
    design: Vec<f64>,
    /// First observed bin of each subject (the record start).
    start: Vec<usize>,
    /// Exclusive end bin of each subject's window per disease, and whether a diagnosis sits in its
    /// last bin. An empty window (stop <= start) contributes nothing.
    stop: Vec<usize>,
    diagnosed: Vec<bool>,
    /// `logit((p + eps) / (1 - p + eps))` of the at-risk incidence per disease and bin (the
    /// file's `logit_prev_t`, eps = 1e-8).
    logit_prevalence: Vec<f64>,
    lambda_precision: Vec<f64>,
    phi_precision: Vec<f64>,
    gp_weight: f64,
    gamma_decay: f64,
    genetic_scale: f64,
}

/// Inverse of the published RBF kernel `exp(-(i - j)^2 / (2 l^2)) + jitter I` on `bins` points.
fn rbf_precision(bins: usize, length_scale: f64, jitter: f64) -> Vec<f64> {
    let mut kernel = vec![0.0; bins * bins];
    for i in 0..bins {
        for j in 0..bins {
            let gap = i as f64 - j as f64;
            kernel[i * bins + j] = (-0.5 * gap * gap / (length_scale * length_scale)).exp()
                + if i == j { jitter } else { 0.0 };
        }
    }
    let lower = dense_cholesky(&kernel, bins);
    let mut precision = vec![0.0; bins * bins];
    for column in 0..bins {
        let mut unit = vec![0.0; bins];
        unit[column] = 1.0;
        let solved = cholesky_solve(&lower, bins, &unit);
        for row in 0..bins {
            precision[row * bins + column] = solved[row];
        }
    }
    precision
}

fn dense_cholesky(matrix: &[f64], n: usize) -> Vec<f64> {
    let mut lower = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix[i * n + j];
            for k in 0..j {
                value -= lower[i * n + k] * lower[j * n + k];
            }
            lower[i * n + j] = if i == j { value.sqrt() } else { value / lower[j * n + j] };
        }
    }
    lower
}

fn cholesky_solve(lower: &[f64], n: usize, right: &[f64]) -> Vec<f64> {
    let mut solution = right.to_vec();
    for i in 0..n {
        for k in 0..i {
            solution[i] -= lower[i * n + k] * solution[k];
        }
        solution[i] /= lower[i * n + i];
    }
    for i in (0..n).rev() {
        for k in (i + 1)..n {
            solution[i] -= lower[k * n + i] * solution[k];
        }
        solution[i] /= lower[i * n + i];
    }
    solution
}

impl AladynProblem {
    /// Starts of phi, psi, gamma and the index of kappa_raw.
    fn offsets(&self) -> (usize, usize, usize, usize) {
        let lambda = self.subjects * self.signatures * self.bins;
        let phi = self.signatures * DISEASES * self.bins;
        let psi = self.signatures * DISEASES;
        let gamma = self.covariates * self.signatures;
        (lambda, lambda + phi, lambda + phi + psi, lambda + phi + psi + gamma)
    }

    fn parameters(&self) -> usize {
        self.offsets().3 + 1
    }

    /// The objective with its running rounding bound, and its gradient with each coordinate's
    /// running bound (Higham ch. 3, accumulated per operation). The data, the design and the GP
    /// precision matrices are inputs. `None` where some `pi >= 1`.
    fn evaluate(&self, x: &[f64], gradient: &mut [f64], bound: &mut [f64]) -> Option<(f64, f64)> {
        gradient.fill(0.0);
        bound.fill(0.0);
        let (phi_start, psi_start, gamma_start, kappa_index) = self.offsets();
        let (n, k, t) = (self.subjects, self.signatures, self.bins);
        let zero = Bounded::exact(0.0);
        let one = Bounded::exact(1.0);
        let kappa = Bounded::exact(x[kappa_index]).exp();
        let sigma: Vec<Bounded> = x[phi_start..psi_start]
            .iter()
            .map(|value| {
                let (probability, mu) = sigmoid_bound(*value, 0.0);
                Bounded { value: probability, mu }
            })
            .collect();
        let inverse_n = one.div(Bounded::exact(n as f64));
        let mut value = zero;
        let mut theta = vec![zero; k];
        let mut pull = vec![zero; k];
        for subject in 0..n {
            for bin in self.start[subject]..t {
                let top = (0..k)
                    .map(|signature| x[(subject * k + signature) * t + bin])
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut total = zero;
                for (signature, slot) in theta.iter_mut().enumerate() {
                    *slot = Bounded::exact(x[(subject * k + signature) * t + bin]).sub(Bounded::exact(top)).exp();
                    total = total.add(*slot);
                }
                for slot in theta.iter_mut() {
                    *slot = slot.div(total);
                }
                for slot in pull.iter_mut() {
                    *slot = zero;
                }
                let mut contributes = false;
                for disease in 0..DISEASES {
                    let stop = self.stop[subject * DISEASES + disease];
                    if bin >= stop {
                        continue;
                    }
                    contributes = true;
                    let mut mixture = zero;
                    for (signature, share) in theta.iter().enumerate() {
                        mixture = mixture.add(share.mul(sigma[(signature * DISEASES + disease) * t + bin]));
                    }
                    let pi = kappa.mul(mixture);
                    if !(pi.value < 1.0) {
                        return None;
                    }
                    let event = self.diagnosed[subject * DISEASES + disease] && bin + 1 == stop;
                    let (loss, slope) = if event {
                        (pi.ln().negate(), Bounded::exact(-1.0).div(pi))
                    } else {
                        (pi.negate().ln_1p().negate(), one.div(one.sub(pi)))
                    };
                    value = value.add(inverse_n.mul(loss));
                    let scale = inverse_n.mul(slope);
                    accumulate(gradient, bound, kappa_index, scale.mul(pi));
                    let scaled = scale.mul(kappa);
                    for (signature, share) in theta.iter().enumerate() {
                        let probability = sigma[(signature * DISEASES + disease) * t + bin];
                        pull[signature] = pull[signature].add(scaled.mul(probability));
                        let sigmoid_slope = probability.mul(one.sub(probability));
                        let index = phi_start + (signature * DISEASES + disease) * t + bin;
                        accumulate(gradient, bound, index, scaled.mul(*share).mul(sigmoid_slope));
                    }
                }
                if contributes {
                    let mut average = zero;
                    for (share, pulled) in theta.iter().zip(pull.iter()) {
                        average = average.add(share.mul(*pulled));
                    }
                    for signature in 0..k {
                        let index = (subject * k + signature) * t + bin;
                        accumulate(gradient, bound, index, theta[signature].mul(pull[signature].sub(average)));
                    }
                }
            }
        }
        // Lambda GP prior around the genetic mean.
        let weight = Bounded::exact(self.gp_weight);
        let half_weight = Bounded::exact(0.5).mul(weight);
        let mut deviation = vec![zero; t];
        for subject in 0..n {
            for signature in 0..k {
                let mut mean = zero;
                for covariate in 0..self.covariates {
                    let design = Bounded::exact(self.design[subject * self.covariates + covariate]);
                    mean = mean.add(design.mul(Bounded::exact(x[gamma_start + covariate * k + signature])));
                }
                mean = mean.mul(Bounded::exact(self.genetic_scale));
                for (bin, slot) in deviation.iter_mut().enumerate() {
                    *slot = Bounded::exact(x[(subject * k + signature) * t + bin]).sub(mean);
                }
                let mut pulled = zero;
                for row in 0..t {
                    let mut solved = zero;
                    for (column, gap) in deviation.iter().enumerate() {
                        solved = solved.add(Bounded::exact(self.lambda_precision[row * t + column]).mul(*gap));
                    }
                    value = value.add(half_weight.mul(inverse_n).mul(deviation[row].mul(solved)));
                    let index = (subject * k + signature) * t + row;
                    accumulate(gradient, bound, index, weight.mul(inverse_n).mul(solved));
                    pulled = pulled.add(solved);
                }
                for covariate in 0..self.covariates {
                    let index = gamma_start + covariate * k + signature;
                    let design = weight
                        .mul(Bounded::exact(self.genetic_scale))
                        .mul(Bounded::exact(self.design[subject * self.covariates + covariate]))
                        .negate();
                    accumulate(gradient, bound, index, design.mul(inverse_n).mul(pulled));
                }
            }
        }
        // Phi GP prior around the prevalence logits plus psi.
        let inverse_d = one.div(Bounded::exact(DISEASES as f64));
        for signature in 0..k {
            for disease in 0..DISEASES {
                let psi = Bounded::exact(x[psi_start + signature * DISEASES + disease]);
                for (bin, slot) in deviation.iter_mut().enumerate() {
                    *slot = Bounded::exact(x[phi_start + (signature * DISEASES + disease) * t + bin])
                        .sub(Bounded::exact(self.logit_prevalence[disease * t + bin]))
                        .sub(psi);
                }
                let mut pulled = zero;
                for row in 0..t {
                    let mut solved = zero;
                    for (column, gap) in deviation.iter().enumerate() {
                        solved = solved.add(Bounded::exact(self.phi_precision[row * t + column]).mul(*gap));
                    }
                    value = value.add(half_weight.mul(inverse_d).mul(deviation[row].mul(solved)));
                    let index = phi_start + (signature * DISEASES + disease) * t + row;
                    accumulate(gradient, bound, index, weight.mul(inverse_d).mul(solved));
                    pulled = pulled.add(solved);
                }
                let index = psi_start + signature * DISEASES + disease;
                accumulate(gradient, bound, index, weight.negate().mul(inverse_d).mul(pulled));
            }
        }
        // torch Adam's weight decay on gamma, in its L2 form.
        let decay = Bounded::exact(self.gamma_decay);
        for index in gamma_start..kappa_index {
            let coefficient = Bounded::exact(x[index]);
            value = value.add(Bounded::exact(0.5).mul(decay).mul(coefficient).mul(coefficient));
            accumulate(gradient, bound, index, decay.mul(coefficient));
        }
        Some((value.value, value.mu))
    }
}

/// The post-hoc certificate of a fit's final point.
enum Certificate {
    /// Every free gradient coordinate `j` lies within `eps * mu_j`, its running rounding bound:
    /// `worst_ratio <= 1`.
    Certified { worst_ratio: f64 },
    /// Some free gradient coordinate is resolved.
    Unresolved { worst_ratio: f64 },
    /// The objective refused the final point (some `pi >= 1`).
    EvaluationRefused,
    /// The objective's value at the final point is not finite.
    NonFiniteValue,
    /// A free gradient coordinate is not finite.
    NonFiniteGradient { coordinate: usize },
    /// A free coordinate's rounding bound is negative or not finite.
    InvalidBound { coordinate: usize },
    /// opt returned no point.
    OptRefused,
}

impl Certificate {
    fn certified(&self) -> bool {
        matches!(self, Certificate::Certified { .. })
    }

    fn describe(&self) -> String {
        match self {
            Certificate::Certified { worst_ratio } => {
                format!("certified worst_gradient_over_band={worst_ratio:.3e}")
            }
            Certificate::Unresolved { worst_ratio } => {
                format!("uncertified reason=resolved-gradient worst_gradient_over_band={worst_ratio:.3e}")
            }
            Certificate::EvaluationRefused => "uncertified reason=final-point-refused".to_string(),
            Certificate::NonFiniteValue => "uncertified reason=non-finite-value".to_string(),
            Certificate::NonFiniteGradient { coordinate } => {
                format!("uncertified reason=non-finite-gradient coordinate={coordinate}")
            }
            Certificate::InvalidBound { coordinate } => {
                format!("uncertified reason=invalid-bound coordinate={coordinate}")
            }
            Certificate::OptRefused => "uncertified reason=opt-refused".to_string(),
        }
    }
}

/// The outcome of minimising with opt's BFGS, certified post hoc.
struct OptFit {
    point: Vec<f64>,
    value: f64,
    termination: String,
    gradient_inf: f64,
    certificate: Certificate,
}

/// Minimise over the first `free` coordinates of `start` with `opt::Bfgs` as is, holding the rest,
/// then certify post hoc. opt's scalar gradient tolerance is the smallest positive value and its
/// iteration cap the largest, so opt stops only on its own line-search, step and stall tests, and
/// acceptance is the per-coordinate certificate. A point where some `pi >= 1` is a recoverable
/// evaluation failure, which opt's line search rejects.
fn fit_with_opt(
    evaluate: &dyn Fn(&[f64], &mut [f64], &mut [f64]) -> Option<(f64, f64)>,
    start: &[f64],
    free: usize,
) -> OptFit {
    let size = start.len();
    let held = &start[free..];
    let objective = FusedObjective::new(|x: &Array1<f64>| {
        let mut full = Vec::with_capacity(size);
        full.extend(x.iter().copied());
        full.extend_from_slice(held);
        let mut gradient = vec![0.0; size];
        let mut bound = vec![0.0; size];
        match evaluate(&full, &mut gradient, &mut bound) {
            Some(pair) => Ok(FirstOrderSample {
                value: pair.0,
                gradient: Array1::from(gradient[..free].to_vec()),
            }),
            None => Err(ObjectiveEvalError::recoverable("some pi >= 1")),
        }
    });
    let refused = |termination: String| OptFit {
        point: start.to_vec(),
        value: f64::NAN,
        termination,
        gradient_inf: f64::NAN,
        certificate: Certificate::OptRefused,
    };
    let (Ok(tolerance), Ok(iterations)) = (Tolerance::new(f64::MIN_POSITIVE), MaxIterations::new(usize::MAX)) else {
        return refused("opt refused its configuration".to_string());
    };
    let mut solver = Bfgs::new(Array1::from(start[..free].to_vec()), objective)
        .with_tolerance(tolerance)
        .with_max_iterations(iterations);
    let (solution, termination) = match solver.run() {
        Ok(solution) => {
            let termination = solution.termination.to_string();
            (solution, termination)
        }
        Err(BfgsError::LineSearchFailed {
            last_solution,
            failure_reason,
            ..
        }) => {
            let termination = format!("line search failed ({failure_reason:?}) after {}", last_solution.termination);
            (*last_solution, termination)
        }
        Err(BfgsError::MaxIterationsReached { last_solution }) => {
            let termination = format!("iteration cap after {}", last_solution.termination);
            (*last_solution, termination)
        }
        Err(other) => return refused(other.to_string()),
    };
    let mut point = solution.final_point.to_vec();
    point.extend_from_slice(held);
    let (value, gradient_inf, certificate) = certify(evaluate, &point, free);
    OptFit {
        point,
        value,
        termination,
        gradient_inf,
        certificate,
    }
}

/// The post-hoc certificate at `point`, from a fresh evaluation there: refused when the objective
/// refuses the point or returns a non-finite value, gradient coordinate or bound, else certified
/// exactly when every free coordinate lies within its band.
fn certify(
    evaluate: &dyn Fn(&[f64], &mut [f64], &mut [f64]) -> Option<(f64, f64)>,
    point: &[f64],
    free: usize,
) -> (f64, f64, Certificate) {
    let mut gradient = vec![0.0; point.len()];
    let mut bound = vec![0.0; point.len()];
    let Some(value) = evaluate(point, &mut gradient, &mut bound).map(|pair| pair.0) else {
        return (f64::NAN, f64::NAN, Certificate::EvaluationRefused);
    };
    if !value.is_finite() {
        return (value, f64::NAN, Certificate::NonFiniteValue);
    }
    // Refuse before taking any maximum: `f64::max` discards NaN, and it would equally discard the
    // negative ratio of a negative bound.
    if let Some(coordinate) = (0..free).find(|j| !gradient[*j].is_finite()) {
        return (value, f64::NAN, Certificate::NonFiniteGradient { coordinate });
    }
    if let Some(coordinate) = (0..free).find(|j| !bound[*j].is_finite() || bound[*j] < 0.0) {
        return (value, f64::NAN, Certificate::InvalidBound { coordinate });
    }
    let gradient_inf = gradient[..free].iter().map(|component| component.abs()).fold(0.0, f64::max);
    let worst_ratio = worst_gradient_over_band(&gradient[..free], &bound[..free]);
    let certificate = if worst_ratio <= 1.0 {
        Certificate::Certified { worst_ratio }
    } else {
        Certificate::Unresolved { worst_ratio }
    };
    (value, gradient_inf, certificate)
}

/// `max_j |g_j| / (eps mu_j)`, at most one exactly when no gradient coordinate is resolved. `0 / 0` is
/// zero, a zero bound under a nonzero coordinate is `+inf`, and any NaN ratio counts as `+inf`, so the
/// maximum never discards a coordinate.
fn worst_gradient_over_band(gradient: &[f64], bound: &[f64]) -> f64 {
    gradient.iter().zip(bound.iter()).fold(0.0, |worst: f64, pair| {
        let ratio = if *pair.0 == 0.0 { 0.0 } else { pair.0.abs() / (f64::EPSILON * pair.1) };
        worst.max(if ratio.is_nan() { f64::INFINITY } else { ratio })
    })
}

/// The baseline's yearly bin of an age.
fn aladyn_bin(law: &TruthLaw, age: f64) -> usize {
    (age - law.origin_age).floor().max(0.0) as usize
}

/// The file's genetic design G (:83-86) for one record: sex, PCs, the scores (zero when missing)
/// and a missing-score indicator, before standardisation.
fn aladyn_covariates(record: &Record) -> Vec<f64> {
    let person = &record.person;
    let observed = if person.scores_observed { 1.0 } else { 0.0 };
    let mut row = vec![person.sex];
    row.extend(person.pcs);
    row.extend(person.scores.map(|score| observed * score));
    row.push(1.0 - observed);
    row
}

/// Centre and scale covariates by the training cohort's moments, as the file does (:84-85).
struct Standardiser {
    means: Vec<f64>,
    scales: Vec<f64>,
}

impl Standardiser {
    fn fit(rows: &[Vec<f64>]) -> Self {
        let columns = rows[0].len();
        let mut means = vec![0.0; columns];
        let mut scales = vec![0.0; columns];
        for column in 0..columns {
            let values: Vec<f64> = rows.iter().map(|row| row[column]).collect();
            means[column] = mean(&values);
            scales[column] = standard_deviation(&values);
        }
        Self { means, scales }
    }

    fn apply(&self, row: &[f64]) -> Vec<f64> {
        row.iter()
            .enumerate()
            .map(|(column, value)| (value - self.means[column]) / self.scales[column])
            .collect()
    }
}

/// Each disease's exclusive stop bin and diagnosis flag for a record observed until `until`. A
/// recorded first diagnosis stops after its bin, with a diagnosis. Otherwise death and loss are
/// censoring. With `include_last_bin` the censoring bin itself counts as a no-event bin, as the
/// published loss counts `t = E` (:230-248). Without it the window stops before that bin, which a
/// forecast then covers.
fn aladyn_windows(
    law: &TruthLaw,
    record: &Record,
    until: f64,
    include_last_bin: bool,
) -> ([usize; DISEASES], [bool; DISEASES]) {
    let mut stop = [0usize; DISEASES];
    let mut diagnosed = [false; DISEASES];
    let end = record.exit.min(until);
    for disease in 0..DISEASES {
        match record.first_occurrence(disease).filter(|age| *age <= end) {
            Some(age) => {
                stop[disease] = aladyn_bin(law, age) + 1;
                diagnosed[disease] = true;
            }
            None => stop[disease] = aladyn_bin(law, end) + usize::from(include_last_bin),
        }
    }
    (stop, diagnosed)
}

/// The published objective over the censored training cohort, left-truncated at each record start.
fn aladyn_problem(law: &TruthLaw, train: &[Record]) -> (AladynProblem, Standardiser) {
    let bins = (law.maximum_age - law.origin_age).round() as usize;
    let rows: Vec<Vec<f64>> = train.iter().map(aladyn_covariates).collect();
    let standardiser = Standardiser::fit(&rows);
    let covariates = rows[0].len();
    let mut design = Vec::with_capacity(train.len() * covariates);
    let mut start = Vec::with_capacity(train.len());
    let mut stop = Vec::with_capacity(train.len() * DISEASES);
    let mut diagnosed = Vec::with_capacity(train.len() * DISEASES);
    let mut events = vec![0.0f64; DISEASES * bins];
    let mut at_risk = vec![0.0f64; DISEASES * bins];
    for (record, row) in train.iter().zip(rows.iter()) {
        design.extend(standardiser.apply(row));
        let first = aladyn_bin(law, record.person.record_start);
        start.push(first);
        let (windows, flags) = aladyn_windows(law, record, record.exit, true);
        for disease in 0..DISEASES {
            for bin in first..windows[disease].min(bins) {
                at_risk[disease * bins + bin] += 1.0;
            }
            if flags[disease] {
                events[disease * bins + windows[disease] - 1] += 1.0;
            }
        }
        stop.extend(windows);
        diagnosed.extend(flags);
    }
    // The file's logit_prev_t (:92-95), from the at-risk incidence per bin.
    let epsilon = 1e-8;
    let logit_prevalence = events
        .iter()
        .zip(at_risk.iter())
        .map(|pair| {
            let prevalence = if *pair.1 > 0.0 { pair.0 / pair.1 } else { 0.0 };
            ((prevalence + epsilon) / (1.0 - prevalence + epsilon)).ln()
        })
        .collect();
    let problem = AladynProblem {
        subjects: train.len(),
        signatures: SIGNATURES,
        bins,
        covariates,
        design,
        start,
        stop,
        diagnosed,
        logit_prevalence,
        lambda_precision: rbf_precision(bins, bins as f64 / 4.0, 1e-6),
        phi_precision: rbf_precision(bins, bins as f64 / 3.0, 1e-6),
        gp_weight: 1.0,
        gamma_decay: 0.01,
        genetic_scale: 1.0,
    };
    (problem, standardiser)
}

/// The file's initialisation (:144-148 values) with a cyclic cluster assignment in place of
/// spectral clustering: psi = 1 in a disease's cluster and -2 elsewhere, phi at its GP mean,
/// lambda and gamma zero, kappa one.
fn aladyn_initial(problem: &AladynProblem) -> Vec<f64> {
    let (phi_start, psi_start, gamma_start, kappa_index) = problem.offsets();
    let mut x = vec![0.0; problem.parameters()];
    for signature in 0..problem.signatures {
        for disease in 0..DISEASES {
            let psi = if disease % problem.signatures == signature { 1.0 } else { -2.0 };
            x[psi_start + signature * DISEASES + disease] = psi;
            for bin in 0..problem.bins {
                x[phi_start + (signature * DISEASES + disease) * problem.bins + bin] =
                    problem.logit_prevalence[disease * problem.bins + bin] + psi;
            }
        }
    }
    x[gamma_start..kappa_index].fill(0.0);
    x[kappa_index] = 0.0;
    x
}

/// The published model's forecast at landmark `s` for one test subject. It refits lambda on the
/// history before `s` with every other parameter held at the fit (the file has no forecast), then
/// takes `F = 1 - prod_{t = bin(s)}^{bin(s) + u - 1} (1 - pi_t)`. Death is censoring in this model,
/// so this is not the competing-risk `F` of the other arms. `None` when the refit is not certified.
fn aladyn_forecast(
    law: &TruthLaw,
    fitted: &AladynProblem,
    fit: &[f64],
    standardiser: &Standardiser,
    record: &Record,
    s: f64,
    horizons: &[f64],
) -> Option<Vec<[f64; DISEASES]>> {
    let (phi_start, kappa_index) = (fitted.offsets().0, fitted.offsets().3);
    let (t, k) = (fitted.bins, fitted.signatures);
    let (windows, flags) = aladyn_windows(law, record, s, false);
    let subject = AladynProblem {
        subjects: 1,
        signatures: k,
        bins: t,
        covariates: fitted.covariates,
        design: standardiser.apply(&aladyn_covariates(record)),
        start: vec![aladyn_bin(law, record.person.record_start)],
        stop: windows.to_vec(),
        diagnosed: flags.to_vec(),
        logit_prevalence: fitted.logit_prevalence.clone(),
        lambda_precision: fitted.lambda_precision.clone(),
        phi_precision: fitted.phi_precision.clone(),
        gp_weight: fitted.gp_weight,
        gamma_decay: fitted.gamma_decay,
        genetic_scale: fitted.genetic_scale,
    };
    let lambda_size = k * t;
    let mut start = vec![0.0; lambda_size];
    start.extend_from_slice(&fit[phi_start..]);
    let refit = fit_with_opt(&|x, gradient, bound| subject.evaluate(x, gradient, bound), &start, lambda_size);
    if !refit.certificate.certified() {
        return None;
    }
    let x = refit.point;
    let kappa = fit[kappa_index].exp();
    let first = aladyn_bin(law, s);
    let mut result = Vec::with_capacity(horizons.len());
    for u in horizons {
        let mut values = [0.0; DISEASES];
        for (disease, slot) in values.iter_mut().enumerate() {
            let mut survival = 1.0;
            for bin in first..(first + u.round() as usize).min(t) {
                let top = (0..k).map(|signature| x[signature * t + bin]).fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = (0..k).map(|signature| (x[signature * t + bin] - top).exp()).collect();
                let total: f64 = weights.iter().sum();
                let mut mixture = 0.0;
                for (signature, weight) in weights.iter().enumerate() {
                    mixture += weight / total * sigmoid(fit[phi_start + (signature * DISEASES + disease) * t + bin]);
                }
                survival *= 1.0 - kappa * mixture;
            }
            *slot = 1.0 - survival;
        }
        result.push(values);
    }
    Some(result)
}

/// The default `aladyn-map` arm while opt has no limited-memory BFGS: it reports the parameter
/// count and the size of the dense inverse Hessian that opt's BFGS would hold, and fits nothing.
/// `aladyn-map-dense` runs the dense fit.
fn aladyn_awaiting(law: &TruthLaw, train: &[Record]) -> Option<Predictions> {
    let problem = aladyn_problem(law, train).0;
    let parameters = problem.parameters();
    report(format!(
        "[a12-arm] arm=aladyn-map status=awaiting-opt-lbfgs parameters={parameters} subjects={} \
         dense_inverse_hessian_gb={:.1}",
        problem.subjects,
        8.0 * (parameters as f64).powi(2) / 1e9,
    ));
    None
}

/// The ALADYNOULLI-style arm: MAP of the published objective on the censored training cohort,
/// accepted only at a certified stationary point, then per-subject certified refits. The model
/// has no death, so its death target is not produced.
fn aladyn_arm(law: &TruthLaw, train: &[Record], test: &[Record], grid: &Grid) -> Option<Predictions> {
    let (problem, standardiser) = aladyn_problem(law, train);
    let start = aladyn_initial(&problem);
    report(format!(
        "[a12-fit] arm=aladyn-map status=started parameters={} subjects={} bins={}",
        problem.parameters(),
        problem.subjects,
        problem.bins
    ));
    let started = Instant::now();
    let fit = fit_with_opt(&|x, gradient, bound| problem.evaluate(x, gradient, bound), &start, start.len());
    let kappa = fit.point[problem.offsets().3].exp();
    report(format!(
        "[a12-fit] arm=aladyn-map status={} seconds={:.1} objective={:.9e} gradient_inf={:.3e} \
         kappa_as_exp_of_raw={kappa:.6} opt_termination={:?}",
        fit.certificate.describe(),
        started.elapsed().as_secs_f64(),
        fit.value,
        fit.gradient_inf,
        fit.termination,
    ));
    if !fit.certificate.certified() {
        return None;
    }
    let x = fit.point;
    let started = Instant::now();
    let cells: Vec<Vec<(usize, usize, usize, f64)>> = test
        .par_iter()
        .map(|record| {
            let mut values = Vec::new();
            for (li, offset) in grid.offsets.iter().enumerate() {
                let s = record.person.entry_age + offset;
                if record.exit <= s {
                    continue;
                }
                if let Some(forecast) = aladyn_forecast(law, &problem, &x, &standardiser, record, s, &grid.horizons) {
                    for (ui, row) in forecast.iter().enumerate() {
                        for disease in (0..DISEASES).filter(|disease| !record.diagnosed_by(*disease, s)) {
                            values.push((li, disease, ui, row[disease]));
                        }
                    }
                }
            }
            values
        })
        .collect();
    let mut table = Predictions::new("aladyn-map", grid, test.len());
    for li in 0..grid.offsets.len() {
        table.produced[li * TARGETS + DEATH_TARGET] = false;
    }
    let mut refitted = 0usize;
    for (i, values) in cells.iter().enumerate() {
        refitted += usize::from(!values.is_empty());
        for value in values {
            table.set(value.0, value.1, value.2, i, value.3);
        }
    }
    report(format!(
        "[a12-forecast] arm=aladyn-map seconds={:.1} subjects_with_certified_refits={refitted}",
        started.elapsed().as_secs_f64()
    ));
    Some(table)
}

/// Fit the engine on the censored training cohort and forecast every admitted test
/// subject alive at each landmark after entry. A refused fit yields no arm.
fn engine_arm(train: &[Record], test: &[Record], grid: &Grid, inputs: Inputs) -> Result<Option<Predictions>, String> {
    let mut cohort = engine_cohort(train, inputs)?;
    let formula = inputs.formula();
    report(format!(
        "[a12-fit] arm={} status=started formula={formula:?} subjects={} rows={}",
        inputs.arm(),
        cohort.subjects.len(),
        cohort.covariates.nrows(),
    ));
    let started = Instant::now();
    let fitted = fit_event_history_formulas(&mut cohort, &[formula], BlockwiseFitOptions::default(), None);
    let seconds = started.elapsed().as_secs_f64();
    match fitted {
        Ok(fit) => {
            report(format!("[a12-fit] arm={} status=fitted seconds={seconds:.1}", inputs.arm()));
            forecast_test(&fit, &cohort, test, grid, inputs).map(Some)
        }
        Err(error) => {
            report(format!(
                "[a12-fit] arm={} status=refused seconds={seconds:.1} error={error}",
                inputs.arm()
            ));
            Ok(None)
        }
    }
}

/// Reverse Kaplan-Meier survival of the censoring time, in years since entry, among
/// subjects under observation at a landmark.
struct CensoringSurvival {
    times: Vec<f64>,
    values: Vec<f64>,
}

impl CensoringSurvival {
    /// `exits` holds `(years since entry, censored)` for the subjects under observation.
    fn from_exits(mut exits: Vec<(f64, bool)>) -> Self {
        exits.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut at_risk = exits.len() as f64;
        let mut survival = 1.0;
        let mut times = Vec::new();
        let mut values = Vec::new();
        let mut index = 0;
        while index < exits.len() {
            let time = exits[index].0;
            let mut censored = 0.0;
            let mut leaving = 0.0;
            while index < exits.len() && exits[index].0 == time {
                censored += if exits[index].1 { 1.0 } else { 0.0 };
                leaving += 1.0;
                index += 1;
            }
            if censored > 0.0 {
                survival *= 1.0 - censored / at_risk;
                times.push(time);
                values.push(survival);
            }
            at_risk -= leaving;
        }
        Self { times, values }
    }

    /// The censoring survival among `records` under observation at landmark `offset`.
    /// Follow-up that reached the administrative end is observed through every horizon,
    /// so its censoring sits after all of them.
    fn from_records<'a>(records: impl Iterator<Item = &'a Record>, offset: f64) -> Self {
        Self::from_exits(
            records
                .filter(|record| record.exit > record.person.entry_age + offset)
                .map(|record| {
                    let time = if record.reached_administrative_end() {
                        f64::INFINITY
                    } else {
                        record.exit - record.person.entry_age
                    };
                    (time, record.death.is_none())
                })
                .collect(),
        )
    }

    /// `G(t)`, or its left limit `G(t-)` when `left`.
    fn at(&self, t: f64, left: bool) -> f64 {
        let count = if left {
            self.times.partition_point(|time| *time < t)
        } else {
            self.times.partition_point(|time| *time <= t)
        };
        if count == 0 { 1.0 } else { self.values[count - 1] }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Outcome {
    Case(f64),
    Death(f64),
    Control,
    Censored,
}

/// One landmark-set subject's prediction, window outcome and censoring weight, in years
/// since entry.
#[derive(Clone, Copy, Debug)]
struct Scored {
    prediction: f64,
    outcome: Outcome,
    weight: f64,
    /// End of the subject's observed time alive and free of the target.
    at_risk_until: f64,
}

impl Scored {
    fn case(&self) -> bool {
        matches!(self.outcome, Outcome::Case(..))
    }
}

fn window_outcome(record: &Record, target: usize, horizon: f64) -> Outcome {
    let entry = record.person.entry_age;
    if let Some(time) = record.target_time(target).filter(|time| *time - entry <= horizon) {
        return Outcome::Case(time - entry);
    }
    if target < DISEASES {
        if let Some(time) = record.death.filter(|time| *time - entry <= horizon) {
            return Outcome::Death(time - entry);
        }
    }
    if record.reached_administrative_end() || record.exit - entry >= horizon {
        Outcome::Control
    } else {
        Outcome::Censored
    }
}

fn scored(record: &Record, target: usize, prediction: f64, horizon: f64, censoring: &CensoringSurvival) -> Scored {
    let outcome = window_outcome(record, target, horizon);
    let weight = match outcome {
        Outcome::Case(time) | Outcome::Death(time) => 1.0 / censoring.at(time, true),
        Outcome::Control => 1.0 / censoring.at(horizon, true),
        Outcome::Censored => 0.0,
    };
    let exit = if record.reached_administrative_end() {
        f64::INFINITY
    } else {
        record.exit - record.person.entry_age
    };
    let at_risk_until = record
        .target_time(target)
        .map_or(exit, |time| (time - record.person.entry_age).min(exit));
    Scored {
        prediction,
        outcome,
        weight,
        at_risk_until,
    }
}

fn brier(items: &[Scored]) -> f64 {
    let total: f64 = items
        .iter()
        .map(|item| {
            let target = if item.case() { 1.0 } else { 0.0 };
            item.weight * (target - item.prediction).powi(2)
        })
        .sum();
    total / items.len() as f64
}

/// The IPCW log score of the case indicator over its finite contributions, and the count of
/// infinite ones (a case predicted 0, or a non-case predicted 1).
fn log_score(items: &[Scored]) -> (f64, usize) {
    let mut total = 0.0;
    let mut infinite = 0;
    for item in items.iter().filter(|item| item.weight > 0.0) {
        let probability = if item.case() { item.prediction } else { 1.0 - item.prediction };
        let contribution = -item.weight * probability.ln();
        if contribution.is_finite() {
            total += contribution;
        } else {
            infinite += 1;
        }
    }
    (total / items.len() as f64, infinite)
}

/// The cumulative/dynamic AUC: cases against every uncensored non-case, or against
/// event-free survivors only when `survivors_only`.
fn cumulative_dynamic_auc(items: &[Scored], survivors_only: bool) -> f64 {
    let mut order: Vec<&Scored> = items
        .iter()
        .filter(|item| item.weight > 0.0 && (item.case() || !survivors_only || item.outcome == Outcome::Control))
        .collect();
    order.sort_by(|a, b| a.prediction.total_cmp(&b.prediction));
    let mut below = 0.0;
    let mut numerator = 0.0;
    let mut cases = 0.0;
    let mut controls = 0.0;
    let mut index = 0;
    while index < order.len() {
        let value = order[index].prediction;
        let mut case_weight = 0.0;
        let mut control_weight = 0.0;
        while index < order.len() && order[index].prediction == value {
            if order[index].case() {
                case_weight += order[index].weight;
            } else {
                control_weight += order[index].weight;
            }
            index += 1;
        }
        numerator += case_weight * (below + 0.5 * control_weight);
        below += control_weight;
        cases += case_weight;
        controls += control_weight;
    }
    numerator / (cases * controls)
}

fn concordant(first: f64, second: f64) -> f64 {
    if first > second {
        1.0
    } else if first == second {
        0.5
    } else {
        0.0
    }
}

fn concordance(items: &[Scored], censoring: &CensoringSurvival) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (i, first) in items.iter().enumerate() {
        let Outcome::Case(time) = first.outcome else {
            continue;
        };
        let lead = 1.0 / censoring.at(time, true);
        for (j, second) in items.iter().enumerate() {
            if i == j {
                continue;
            }
            let weight = match second.outcome {
                Outcome::Death(other) if other < time => lead / censoring.at(other, true),
                _ if second.at_risk_until > time => lead / censoring.at(time, false),
                _ => continue,
            };
            numerator += weight * concordant(first.prediction, second.prediction);
            denominator += weight;
        }
    }
    numerator / denominator
}

/// The IPCW-weighted logistic recalibration of the case indicator on `logit F`.
fn recalibration(items: &[Scored], slope: bool) -> (f64, f64) {
    let rows: Vec<(f64, f64, f64)> = items
        .iter()
        .filter(|item| item.weight > 0.0 && item.prediction > 0.0 && item.prediction < 1.0)
        .map(|item| (logit(item.prediction), if item.case() { 1.0 } else { 0.0 }, item.weight))
        .collect();
    logistic_recalibration(&rows, slope)
}

/// The recalibration of jackknife Aalen-Johansen pseudo-observations on `logit F`
/// through the logit link. `pseudo` is aligned with `items`.
fn pseudo_recalibration(items: &[Scored], pseudo: &[f64], slope: bool) -> (f64, f64) {
    let rows: Vec<(f64, f64, f64)> = items
        .iter()
        .zip(pseudo)
        .filter(|pair| pair.0.prediction > 0.0 && pair.0.prediction < 1.0)
        .map(|pair| (logit(pair.0.prediction), *pair.1, 1.0))
        .collect();
    logistic_recalibration(&rows, slope)
}

/// Newton's method on the logistic score equations over `(logit F, response, weight)`
/// rows. With `slope` it fits `logit P = a + b logit F`; without, `logit P = a +
/// logit F` (calibration in the large). A response may be any real value, such as a
/// pseudo-observation, which makes these the quasi-likelihood equations. NaN where
/// Newton does not converge, and for the slope of a constant prediction.
fn logistic_recalibration(rows: &[(f64, f64, f64)], slope: bool) -> (f64, f64) {
    if rows.is_empty() || (slope && rows.iter().all(|row| row.0 == rows[0].0)) {
        return (f64::NAN, f64::NAN);
    }
    let mut a = 0.0;
    let mut b = 1.0;
    let mut previous = f64::INFINITY;
    let mut stalls = 0;
    loop {
        let mut gradient = [0.0; 2];
        let mut magnitude = [0.0; 2];
        let mut hessian = [0.0; 3];
        for row in rows.iter() {
            // Each score equation's running rounding bound, accumulated inline: every operation
            // adds its operands' propagated bounds and its own result's magnitude.
            let product = b * row.0;
            let linear = a + product;
            let (p, p_bound) = sigmoid_bound(linear, product.abs() + linear.abs());
            let difference = row.1 - p;
            let residual = row.2 * difference;
            let residual_bound = row.2 * (p_bound + difference.abs()) + residual.abs();
            let curvature = row.2 * p * (1.0 - p);
            gradient[0] += residual;
            magnitude[0] += residual_bound + gradient[0].abs();
            let moment = residual * row.0;
            gradient[1] += moment;
            magnitude[1] += row.0.abs() * residual_bound + moment.abs() + gradient[1].abs();
            hessian[0] += curvature;
            hessian[1] += curvature * row.0;
            hessian[2] += curvature * row.0 * row.0;
        }
        // Converged once every score equation lies within its running bound `eps mu`. Not
        // converged when that ratio fails to fall on two consecutive steps.
        let equations = if slope { 2 } else { 1 };
        let ratio = (0..equations)
            .map(|j| gradient[j].abs() / (f64::EPSILON * magnitude[j]))
            .fold(0.0, f64::max);
        if ratio <= 1.0 {
            return (a, b);
        }
        if ratio < previous {
            stalls = 0;
        } else {
            stalls += 1;
            if stalls == 2 {
                return (f64::NAN, f64::NAN);
            }
        }
        previous = ratio;
        let (step_a, step_b) = if slope {
            let determinant = hessian[0] * hessian[2] - hessian[1] * hessian[1];
            if !(determinant > 0.0) {
                return (f64::NAN, f64::NAN);
            }
            (
                (hessian[2] * gradient[0] - hessian[1] * gradient[1]) / determinant,
                (hessian[0] * gradient[1] - hessian[1] * gradient[0]) / determinant,
            )
        } else {
            if !(hessian[0] > 0.0) {
                return (f64::NAN, f64::NAN);
            }
            (gradient[0] / hessian[0], 0.0)
        };
        a += step_a;
        b += step_b;
    }
}

/// The inverse-Fisher standard error of the recalibration intercept (without `slope`) or slope
/// at `(a, b)`, for the self-check's calibration checks.
fn recalibration_standard_error(items: &[Scored], a: f64, b: f64, slope: bool) -> f64 {
    let mut hessian = [0.0; 3];
    for item in items
        .iter()
        .filter(|item| item.weight > 0.0 && item.prediction > 0.0 && item.prediction < 1.0)
    {
        let z = logit(item.prediction);
        let p = sigmoid(a + b * z);
        let curvature = item.weight * p * (1.0 - p);
        hessian[0] += curvature;
        hessian[1] += curvature * z;
        hessian[2] += curvature * z * z;
    }
    if slope {
        (hessian[0] / (hessian[0] * hessian[2] - hessian[1] * hessian[1])).sqrt()
    } else {
        hessian[0].recip().sqrt()
    }
}

/// The first-order Monte Carlo standard error of a Brier difference, from the arms' per-subject
/// Monte Carlo errors (zero for an exact arm):
/// `(1/n) sqrt(sum w^2 [4 (y - F1)^2 e1^2 + 4 (y - F2)^2 e2^2])`.
fn monte_carlo_brier_error(first: &[Scored], second: &[Scored], first_errors: &[f64], second_errors: &[f64]) -> f64 {
    let mut total = 0.0;
    for (((left, right), left_error), right_error) in first.iter().zip(second).zip(first_errors).zip(second_errors) {
        let target = if left.case() { 1.0 } else { 0.0 };
        let left_error = if left_error.is_finite() { *left_error } else { 0.0 };
        let right_error = if right_error.is_finite() { *right_error } else { 0.0 };
        total += 4.0
            * left.weight
            * left.weight
            * ((target - left.prediction).powi(2) * left_error * left_error
                + (target - right.prediction).powi(2) * right_error * right_error);
    }
    total.sqrt() / first.len() as f64
}

/// One scored cell: its landmark, target and horizon indices, and its landmark and
/// horizon in years since entry.
#[derive(Clone, Copy)]
struct Cell {
    landmark: usize,
    target: usize,
    horizon: usize,
    offset: f64,
    years: f64,
}

/// Paired bootstrap standard errors of `first - second` for the Brier score and AUC. Each
/// replicate resamples the whole cohort, re-estimates the censoring survival on the
/// resample, and rescores the matched subjects it drew.
fn paired_bootstrap(
    records: &[Record],
    matched: &[bool],
    pair: [&Predictions; 2],
    cell: Cell,
    replicates: usize,
    rng: &mut Rng,
) -> (f64, f64) {
    let n = records.len();
    let mut brier_differences = Vec::with_capacity(replicates);
    let mut auc_differences = Vec::with_capacity(replicates);
    let mut drawn = Vec::with_capacity(n);
    let mut left = Vec::new();
    let mut right = Vec::new();
    while n > 0 && brier_differences.len() < replicates {
        drawn.clear();
        while drawn.len() < n {
            drawn.push(rng.index(n));
        }
        let censoring = CensoringSurvival::from_records(drawn.iter().map(|i| &records[*i]), cell.offset);
        left.clear();
        right.clear();
        for i in drawn.iter().copied().filter(|i| matched[*i]) {
            let record = &records[i];
            let first = pair[0].get(cell.landmark, cell.target, cell.horizon, i);
            let second = pair[1].get(cell.landmark, cell.target, cell.horizon, i);
            left.push(scored(record, cell.target, first, cell.years, &censoring));
            right.push(scored(record, cell.target, second, cell.years, &censoring));
        }
        brier_differences.push(brier(&left) - brier(&right));
        auc_differences.push(cumulative_dynamic_auc(&left, false) - cumulative_dynamic_auc(&right, false));
    }
    (standard_deviation(&brier_differences), standard_deviation(&auc_differences))
}

struct Options {
    self_check: bool,
    scenario: Scenario,
    n_train: usize,
    n_test: usize,
    seed: u64,
    oracle_paths: usize,
    particles: usize,
    forecast_paths: usize,
    filter_replicates: usize,
    bootstrap: usize,
    arms: Vec<String>,
}

fn parsed<T: std::str::FromStr>(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<T, String> {
    args.next()
        .and_then(|text| text.parse::<T>().ok())
        .ok_or_else(|| format!("{flag} needs a valid value"))
}

fn parse_options() -> Result<Options, String> {
    let mut options = Options {
        self_check: false,
        scenario: Scenario::Joint,
        n_train: 1000,
        n_test: 1000,
        seed: 2961,
        oracle_paths: 64,
        particles: 128,
        forecast_paths: 64,
        filter_replicates: 4,
        bootstrap: 200,
        arms: [
            "oracle-hs",
            "oracle-hs-dx",
            "oracle-state",
            "joint-rank0",
            "aladyn-map",
            "evh-dx-genetics",
            "evh-dx-genetics-cc",
            "evh-dx-genetics-channels",
        ]
        .into_iter()
        .map(String::from)
        .collect(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--self-check" => options.self_check = true,
            "--scenario" => {
                let text: String = parsed(&mut args, &flag)?;
                options.scenario = Scenario::parse(&text).ok_or_else(|| format!("unknown scenario {text:?}"))?;
            }
            "--n-train" => options.n_train = parsed(&mut args, &flag)?,
            "--n-test" => options.n_test = parsed(&mut args, &flag)?,
            "--seed" => options.seed = parsed(&mut args, &flag)?,
            "--oracle-paths" => options.oracle_paths = parsed(&mut args, &flag)?,
            "--particles" => options.particles = parsed(&mut args, &flag)?,
            "--forecast-paths" => options.forecast_paths = parsed(&mut args, &flag)?,
            "--filter-replicates" => options.filter_replicates = parsed(&mut args, &flag)?,
            "--bootstrap" => options.bootstrap = parsed(&mut args, &flag)?,
            "--arms" => {
                let text: String = parsed(&mut args, &flag)?;
                options.arms = text.split(',').map(String::from).collect();
            }
            // `cargo bench` passes this flag to a harness-free program.
            "--bench" => {}
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(options)
}

/// Score every cell of both variants twice: over every subject with the arms that predict
/// everyone, and over the subjects with observed scores with every arm (the complete-case
/// engine predicts only them).
fn score(variants: &[(&str, &[Record])], grid: &Grid, tables: &[Predictions], options: &Options) {
    let everyone: Vec<&Predictions> = tables
        .iter()
        .filter(|table| table.arm != Inputs::CompleteCase.arm())
        .collect();
    let every_arm: Vec<&Predictions> = tables.iter().collect();
    score_subset(variants, grid, &everyone, "all", false, tables.len(), options);
    score_subset(variants, grid, &every_arm, "complete-g", true, tables.len(), options);
}

/// Score every cell of both variants over the subset's subjects that every arm in
/// `tables` predicts. Each arm is compared with `null`, the previous arm, and each oracle
/// filter.
fn score_subset(
    variants: &[(&str, &[Record])],
    grid: &Grid,
    tables: &[&Predictions],
    subset: &str,
    complete_only: bool,
    stage: usize,
    options: &Options,
) {
    let scenario = options.scenario.name();
    let oracle = tables.iter().position(|table| table.arm == "oracle-hs");
    let filters: Vec<usize> = tables
        .iter()
        .enumerate()
        .filter(|pair| pair.1.arm.starts_with("oracle-hs"))
        .map(|pair| pair.0)
        .collect();
    for (vi, (variant, records)) in variants.iter().enumerate() {
        for (li, offset) in grid.offsets.iter().enumerate() {
            let censoring = CensoringSurvival::from_records(records.iter(), *offset);
            for target in 0..TARGETS {
                let name = target_name(target);
                for (ui, u) in grid.horizons.iter().enumerate() {
                    let horizon = offset + u;
                    let eligible: Vec<usize> = (0..records.len())
                        .filter(|i| {
                            let record = &records[*i];
                            (!complete_only || record.person.scores_observed)
                                && record.in_landmark_set(target, record.person.entry_age + offset)
                        })
                        .collect();
                    // An oracle filter, or the ALADYNOULLI-style arm, that lost an eligible subject is
                    // refused for the cell, never allowed to shrink every arm's subjects.
                    let scoring: Vec<bool> = tables
                        .iter()
                        .map(|table| {
                            let lost = eligible.iter().any(|i| !table.get(li, target, ui, *i).is_finite());
                            let refusable = table.arm.starts_with("oracle-hs") || table.arm == "aladyn-map";
                            table.producing(li, target) && !(lost && refusable)
                        })
                        .collect();
                    let matched: Vec<usize> = eligible
                        .iter()
                        .copied()
                        .filter(|i| {
                            tables
                                .iter()
                                .zip(scoring.iter())
                                .all(|pair| !*pair.1 || pair.0.get(li, target, ui, *i).is_finite())
                        })
                        .collect();
                    for (table, scores) in tables.iter().zip(scoring.iter()) {
                        if !table.producing(li, target) {
                            continue;
                        }
                        let lost: Vec<usize> = eligible
                            .iter()
                            .copied()
                            .filter(|i| !table.get(li, target, ui, *i).is_finite())
                            .collect();
                        if lost.is_empty() {
                            continue;
                        }
                        let lost_cases = lost
                            .iter()
                            .filter(|i| matches!(window_outcome(&records[**i], target, horizon), Outcome::Case(..)))
                            .count();
                        report(format!(
                            "[a12-lost] scenario={scenario} stage={stage} subset={subset} variant={variant} arm={} target={name} \
                             s=entry+{offset} u={u} eligible={} lost={} lost_cases={lost_cases} status={}",
                            table.arm,
                            eligible.len(),
                            lost.len(),
                            if *scores { "excluded-from-every-arm" } else { "refused" },
                        ));
                    }
                    let mut in_matched = vec![false; records.len()];
                    for i in matched.iter() {
                        in_matched[*i] = true;
                    }
                    let arms: Vec<Vec<Scored>> = tables
                        .iter()
                        .map(|table| {
                            matched
                                .iter()
                                .map(|i| scored(&records[*i], target, table.get(li, target, ui, *i), horizon, &censoring))
                                .collect()
                        })
                        .collect();
                    let null_brier = brier(&arms[0]);
                    let pseudo = pseudo_observations(records, *offset, target, horizon, &matched);
                    let errors: Vec<Vec<f64>> = tables
                        .iter()
                        .map(|table| matched.iter().map(|i| table.error(li, target, ui, *i)).collect())
                        .collect();
                    for ((table, items), scores) in tables.iter().zip(arms.iter()).zip(scoring.iter()) {
                        if !*scores {
                            report(format!(
                                "[a12-metric] scenario={scenario} stage={stage} subset={subset} variant={variant} arm={} \
                                 target={name} s=entry+{offset} u={u} status={}",
                                table.arm,
                                if table.producing(li, target) { "refused" } else { "not-produced" },
                            ));
                            continue;
                        }
                        let n = items.len() as f64;
                        let cases = items.iter().filter(|item| item.case()).count();
                        let deaths = items.iter().filter(|item| matches!(item.outcome, Outcome::Death(..))).count();
                        let controls = items.iter().filter(|item| item.outcome == Outcome::Control).count();
                        let censored = items.iter().filter(|item| item.outcome == Outcome::Censored).count();
                        let observed = items.iter().filter(|item| item.case()).map(|item| item.weight).sum::<f64>() / n;
                        let predicted: Vec<f64> = items.iter().map(|item| item.prediction).collect();
                        let mean_prediction = mean(&predicted);
                        let value = brier(items);
                        let (log_finite, log_infinite) = log_score(items);
                        let intercept = recalibration(items, false).0;
                        let (slope_intercept, slope) = recalibration(items, true);
                        let pseudo_intercept = pseudo_recalibration(items, &pseudo, false).0;
                        let pseudo_slope = pseudo_recalibration(items, &pseudo, true).1;
                        let calibration_nan = [intercept, slope, pseudo_intercept, pseudo_slope]
                            .iter()
                            .filter(|value| value.is_nan())
                            .count();
                        let distance = match oracle {
                            Some(o) => {
                                let gaps: Vec<f64> = matched
                                    .iter()
                                    .map(|i| (table.get(li, target, ui, *i) - tables[o].get(li, target, ui, *i)).abs())
                                    .collect();
                                mean(&gaps)
                            }
                            None => f64::NAN,
                        };
                        report(format!(
                            "[a12-metric] scenario={scenario} stage={stage} subset={subset} variant={variant} arm={} target={name} \
                             s=entry+{offset} u={u} eligible={} matched={} cases={cases} deaths={deaths} controls={controls} \
                             censored={censored} observed={observed:.5} mean_prediction={mean_prediction:.5} brier={value:.6} \
                             ipa={:.4} log_score_finite={log_finite:.5} log_score_infinite={log_infinite} auc={:.4} \
                             auc_survivors={:.4} cindex={:.4} cal_intercept={intercept:.4} \
                             cal_slope={slope:.4} cal_slope_intercept={slope_intercept:.4} pseudo_intercept={pseudo_intercept:.4} \
                             pseudo_slope={pseudo_slope:.4} calibration_nan={calibration_nan} \
                             mean_abs_to_oracle_hs={distance:.5}",
                            table.arm,
                            eligible.len(),
                            matched.len(),
                            1.0 - value / null_brier,
                            cumulative_dynamic_auc(items, false),
                            cumulative_dynamic_auc(items, true),
                            concordance(items, &censoring),
                        ));
                    }
                    for k in 1..tables.len() {
                        let mut bases = vec![0];
                        if k > 1 {
                            bases.push(k - 1);
                        }
                        for filter in filters.iter() {
                            if *filter != k && !bases.contains(filter) {
                                bases.push(*filter);
                            }
                        }
                        for base in bases {
                            if !scoring[k] || !scoring[base] {
                                continue;
                            }
                            let cell = ((((vi * 16 + stage) * grid.offsets.len() + li) * TARGETS + target)
                                * grid.horizons.len())
                                + ui;
                            let stream = (4u64 << 56) | ((cell as u64) << 16) | (k * 256 + base) as u64;
                            let mut rng = Rng::new(options.seed, stream);
                            let place = Cell {
                                landmark: li,
                                target,
                                horizon: ui,
                                offset: *offset,
                                years: horizon,
                            };
                            let pair = [tables[k], tables[base]];
                            let (se_brier, se_auc) =
                                paired_bootstrap(records, &in_matched, pair, place, options.bootstrap, &mut rng);
                            let delta_brier = brier(&arms[k]) - brier(&arms[base]);
                            let mc_se_brier = monte_carlo_brier_error(&arms[k], &arms[base], &errors[k], &errors[base]);
                            let resolution = if mc_se_brier > 0.0 && delta_brier.abs() <= mc_se_brier {
                                "unresolved"
                            } else {
                                "resolved"
                            };
                            report(format!(
                                "[a12-paired] scenario={scenario} stage={stage} subset={subset} variant={variant} arm={} base={} target={name} \
                                 s=entry+{offset} u={u} matched={} delta_brier={delta_brier:.6} se_test_sampling_brier={se_brier:.6} \
                                 mc_se_brier={mc_se_brier:.6} brier_resolution={resolution} delta_auc={:.4} \
                                 se_test_sampling_auc={se_auc:.4}",
                                tables[k].arm,
                                tables[base].arm,
                                matched.len(),
                                cumulative_dynamic_auc(&arms[k], false) - cumulative_dynamic_auc(&arms[base], false),
                            ));
                        }
                    }
                }
            }
        }
    }
}

fn run(options: &Options) -> Result<(), String> {
    let law = TruthLaw::new(options.scenario);
    let grid = Grid {
        offsets: vec![0.0, 2.0, 5.0],
        horizons: vec![1.0, 5.0, 10.0],
    };
    report(format!(
        "[a12-config] scenario={} seed={} n_train={} n_test={} landmarks=entry+{:?} horizons={:?} oracle_paths={} \
         particles={} forecast_paths={} filter_replicates={} bootstrap={} arms={:?} steps_per_year={}",
        law.scenario.name(),
        options.seed,
        options.n_train,
        options.n_test,
        grid.offsets,
        grid.horizons,
        options.oracle_paths,
        options.particles,
        options.forecast_paths,
        options.filter_replicates,
        options.bootstrap,
        options.arms,
        law.steps_per_year,
    ));
    let started = Instant::now();
    let (train_truth, train_redrawn) = draw_cohort(&law, options.seed, 1, options.n_train, &grid.offsets);
    let train: Vec<Record> = train_truth.iter().map(Record::censored).collect();
    let (test, test_redrawn) = draw_cohort(&law, options.seed, 2, options.n_test, &grid.offsets);
    let test_censored: Vec<Record> = test.iter().map(Record::censored).collect();
    report(format!("[a12-arm] arm=generator seconds={:.1}", started.elapsed().as_secs_f64()));
    describe("train-censored", &train, train_redrawn);
    describe("test-uncensored", &test, test_redrawn);
    describe("test-censored", &test_censored, test_redrawn);
    let variants = [("censored", test_censored.as_slice()), ("uncensored", test.as_slice())];
    let mut tables = vec![null_arm(&train, &test, &grid)];
    for (index, arm) in options.arms.iter().enumerate() {
        let table = match arm.as_str() {
            "oracle-hs" => Some(filtered_oracle_arm(&law, &test, &grid, options, true)),
            "oracle-hs-dx" => Some(filtered_oracle_arm(&law, &test, &grid, options, false)),
            "oracle-state" => Some(state_oracle_arm(&law, &test, &grid, options.seed, options.oracle_paths)),
            "joint-rank0" => joint_rank_zero_arm(&train, &test, &grid)?,
            "aladyn-map" => aladyn_awaiting(&law, &train),
            "aladyn-map-dense" => aladyn_arm(&law, &train, &test, &grid),
            "evh-dx-genetics" => engine_arm(&train, &test, &grid, Inputs::MeanImputed)?,
            "evh-dx-genetics-cc" => engine_arm(&train, &test, &grid, Inputs::CompleteCase)?,
            "evh-dx-genetics-channels" => engine_arm(&train, &test, &grid, Inputs::Channels)?,
            other => return Err(format!("unknown arm {other:?}")),
        };
        let produced = table.is_some();
        if let Some(table) = table {
            tables.push(table);
        }
        // Score after every fitted arm that produced a table, after an oracle only when it is the
        // last arm, and always after the last arm.
        if (produced && !arm.starts_with("oracle")) || index + 1 == options.arms.len() {
            score(&variants, &grid, &tables, options);
        }
    }
    Ok(())
}

/// One Monte Carlo check: cells of `(value, expected, standard error, deterministic error)`. A
/// positive check passes when every cell lies within `z SE + error`; a negative control passes
/// when some cell lies outside.
struct MonteCarloCheck {
    name: String,
    negative: bool,
    cells: Vec<(f64, f64, f64, f64)>,
}

struct Checks {
    failures: usize,
    monte_carlo: Vec<MonteCarloCheck>,
}

impl Checks {
    fn close(&mut self, name: &str, value: f64, expected: f64, tolerance: f64) {
        let pass = (value - expected).abs() <= tolerance;
        if !pass {
            self.failures += 1;
        }
        report(format!(
            "[a12-selfcheck] name={name} value={value:.12e} expected={expected:.12e} tolerance={tolerance:.1e} pass={pass}"
        ));
    }

    fn monte_carlo(&mut self, name: &str, value: f64, expected: f64, standard_error: f64, error: f64) {
        self.monte_carlo.push(MonteCarloCheck {
            name: name.to_string(),
            negative: false,
            cells: vec![(value, expected, standard_error, error)],
        });
    }

    fn negative_control(&mut self, name: &str, cells: Vec<(f64, f64, f64, f64)>) {
        self.monte_carlo.push(MonteCarloCheck {
            name: name.to_string(),
            negative: true,
            cells,
        });
    }

    /// Evaluate the Monte Carlo checks with `z = Phi^-1(1 - alpha / (2 m))` over the `m` positive
    /// cells, and report every cell.
    fn finish(&mut self) {
        let m = self
            .monte_carlo
            .iter()
            .filter(|check| !check.negative)
            .map(|check| check.cells.len())
            .sum::<usize>()
            .max(1);
        let z = Normal::new(0.0, 1.0)
            .map(|normal| normal.inverse_cdf(1.0 - SELF_CHECK_ALPHA / (2.0 * m as f64)))
            .unwrap_or(f64::NAN);
        let mut failures = 0;
        for check in self.monte_carlo.iter() {
            let outside = check
                .cells
                .iter()
                .any(|cell| !((cell.0 - cell.1).abs() <= z * cell.2 + cell.3));
            let pass = if check.negative { outside } else { !outside };
            failures += usize::from(!pass);
            // A negative control should fail by a margin, not by one cell's chance.
            let margin = check
                .cells
                .iter()
                .map(|cell| (cell.0 - cell.1).abs() / (z * cell.2 + cell.3))
                .fold(0.0, f64::max);
            report(format!(
                "[a12-selfcheck-summary] name={} negative_control={} cells={} largest_deviation_over_bound={margin:.3} pass={pass}",
                check.name,
                check.negative,
                check.cells.len(),
            ));
            for cell in check.cells.iter() {
                report(format!(
                    "[a12-selfcheck] name={} negative_control={} value={:.12e} expected={:.12e} se={:.3e} error={:.3e} \
                     z={z:.3} bound={:.3e} pass={pass}",
                    check.name,
                    check.negative,
                    cell.0,
                    cell.1,
                    cell.2,
                    cell.3,
                    z * cell.2 + cell.3,
                ));
            }
        }
        self.failures += failures;
    }
}

/// The continuous-time incidence from `s` to `s + u` under intensities `exp(a_m + b_m (age - 55))`
/// for the target and death only. Returns the value, the change at twice the Gauss-Legendre order
/// (its quadrature error), and its running rounding bound. The quadrature nodes are inputs, and
/// death's incidence is closed form.
fn gompertz_incidence(law: &TruthLaw, target: usize, s: f64, u: f64) -> (f64, f64, f64) {
    let reference = Bounded::exact(law.reference_age);
    let cumulative = |mark: usize, age: f64| {
        let slope = Bounded::exact(law.age_slope[mark]);
        let at_start = Bounded::exact(law.log_rate[mark]).add(slope.mul(Bounded::exact(s).sub(reference))).exp();
        let elapsed = Bounded::exact(age).sub(Bounded::exact(s));
        if slope.value == 0.0 {
            at_start.mul(elapsed)
        } else {
            at_start.mul(slope.mul(elapsed).exp_m1()).div(slope)
        }
    };
    if target >= DISEASES {
        let incidence = cumulative(DEATH, s + u).negate().exp_m1().negate();
        return (incidence.value, 0.0, incidence.mu);
    }
    let at_order = |order: usize| {
        let mut total = Bounded::exact(0.0);
        for node in gauss_legendre(order) {
            let age = s + 0.5 * u * (node.0 + 1.0);
            let rate = Bounded::exact(law.log_rate[target])
                .add(Bounded::exact(law.age_slope[target]).mul(Bounded::exact(age).sub(reference)))
                .exp();
            let survival = cumulative(target, age).add(cumulative(DEATH, age)).negate().exp();
            total = total.add(Bounded::exact(0.5 * u * node.1).mul(rate).mul(survival));
        }
        total
    };
    let coarse = at_order(GOMPERTZ_ORDER);
    let fine = at_order(2 * GOMPERTZ_ORDER);
    (fine.value, (fine.value - coarse.value).abs(), fine.mu)
}

/// The exact `F*` of a static one-signature law given one record, by Gauss-Hermite quadrature over
/// the first signature, with the change at twice the order as its error. The law has no drive, no
/// jumps, no age or sex slopes, decoders on the first signature only, and exogenous attendance.
/// The first laboratory loads the first signature only. The record holds no hospitalisations,
/// and its visits measure only the first laboratory. Returns `(value, error)` per horizon and
/// target, NaN where the target is diagnosed by `s`.
fn static_posterior(law: &TruthLaw, record: &Record, s: f64, horizons: &[f64]) -> Vec<[(f64, f64); TARGETS]> {
    let coarse = static_posterior_at_order(law, record, s, horizons, STATIC_ORDER);
    let fine = static_posterior_at_order(law, record, s, horizons, 2 * STATIC_ORDER);
    fine.iter()
        .zip(coarse.iter())
        .map(|pair| {
            let mut cells = [(f64::NAN, f64::NAN); TARGETS];
            for (target, cell) in cells.iter_mut().enumerate() {
                *cell = (pair.0[target], (pair.0[target] - pair.1[target]).abs());
            }
            cells
        })
        .collect()
}

fn static_posterior_at_order(
    law: &TruthLaw,
    record: &Record,
    s: f64,
    horizons: &[f64],
    order: usize,
) -> Vec<[f64; TARGETS]> {
    let person = &record.person;
    let (origin, start) = (law.origin_age, person.record_start);
    let mut numerators = vec![[0.0; TARGETS]; horizons.len()];
    let mut denominator = 0.0;
    for node in gauss_hermite(order) {
        let mut signatures = [0.0; SIGNATURES];
        signatures[0] = std::f64::consts::SQRT_2 * node.0;
        let state = MarkovState {
            signatures,
            diagnosed: [false; DISEASES],
        };
        let death = law.intensity(person, &state, DEATH, s);
        // Survival from the origin to s, and no hospitalisation recorded from the record start.
        let mut likelihood = (-death * (s - origin) - law.intensity(person, &state, HOSPITAL, s) * (s - start)).exp();
        // Each attended visit before s that measured the first laboratory: its Student-t density
        // in closed form, up to state-free constants.
        for visit in record
            .visits
            .iter()
            .filter(|visit| visit.age < s && visit.attended && !visit.values[LAB_A].is_nan())
        {
            let location = law.lab_intercept[LAB_A] + law.lab_loading[LAB_A][0] * signatures[0];
            let standardised = (visit.values[LAB_A] - location) / law.lab_scale[LAB_A];
            let dof = law.lab_dof as f64;
            likelihood *= (1.0 + standardised * standardised / dof).powf(-0.5 * (dof + 1.0));
        }
        let mut free = [1.0; DISEASES];
        for (d, share) in free.iter_mut().enumerate() {
            let rate = law.intensity(person, &state, d, s);
            match record.first_occurrence(d).filter(|time| *time <= s) {
                // Not carried from before the record start, then diagnosed at its recorded age.
                Some(time) => likelihood *= rate * (-rate * (time - origin)).exp(),
                // Either carried unrecorded from before the record start, or free through s.
                None => {
                    let carried = -(-rate * (start - origin)).exp_m1();
                    let unrecorded = (-rate * (s - origin)).exp();
                    likelihood *= carried + unrecorded;
                    *share = unrecorded / (carried + unrecorded);
                }
            }
        }
        let mass = node.1 * likelihood;
        denominator += mass;
        for (ui, u) in horizons.iter().enumerate() {
            for target in 0..TARGETS {
                let value = if target < DISEASES {
                    let rate = law.intensity(person, &state, target, s);
                    free[target] * rate / (rate + death) * -(-(rate + death) * u).exp_m1()
                } else {
                    -(-death * u).exp_m1()
                };
                numerators[ui][target] += mass * value;
            }
        }
    }
    numerators
        .iter()
        .map(|row| {
            let mut values = [f64::NAN; TARGETS];
            for (target, value) in values.iter_mut().enumerate() {
                if !record.diagnosed_by(target, s) {
                    *value = row[target] / denominator;
                }
            }
            values
        })
        .collect()
}

/// [`incidence_through`]'s operations with a running bound. The exits are exact data.
fn bounded_incidence(exits: &[(f64, Transition, usize)], horizon: f64, skip: Option<usize>) -> Bounded {
    let one = Bounded::exact(1.0);
    let mut at_risk = exits.iter().filter(|exit| Some(exit.2) != skip).count() as f64;
    let mut survival = one;
    let mut incidence = Bounded::exact(0.0);
    let mut index = 0;
    while index < exits.len() && exits[index].0 <= horizon {
        let time = exits[index].0;
        let (mut targets, mut deaths, mut censored) = (0.0f64, 0.0f64, 0.0f64);
        while index < exits.len() && exits[index].0 == time {
            if Some(exits[index].2) != skip {
                match exits[index].1 {
                    Transition::Target => targets += 1.0,
                    Transition::Death => deaths += 1.0,
                    Transition::Censored => censored += 1.0,
                }
            }
            index += 1;
        }
        if at_risk > 0.0 {
            let risk = Bounded::exact(at_risk);
            incidence = incidence.add(survival.mul(Bounded::exact(targets)).div(risk));
            survival = survival.mul(one.sub(Bounded::exact(targets + deaths).div(risk)));
        }
        at_risk -= targets + deaths + censored;
    }
    incidence
}

/// [`pseudo_observations`]'s operations for one subject at landmark 0, with a running bound.
fn bounded_pseudo(records: &[Record], target: usize, horizon: f64, subject: usize) -> Bounded {
    let exits = landmark_exits(records, 0.0, target);
    let n = exits.len() as f64;
    let full = bounded_incidence(&exits, horizon, None);
    let left_out = bounded_incidence(&exits, horizon, Some(subject));
    Bounded::exact(n).mul(full).sub(Bounded::exact(n - 1.0).mul(left_out))
}

/// [`brier`]'s operations with running bounds on the weights.
fn bounded_brier(items: &[Scored], weights: &[Bounded]) -> Bounded {
    let mut total = Bounded::exact(0.0);
    for (item, weight) in items.iter().zip(weights) {
        let target = Bounded::exact(if item.case() { 1.0 } else { 0.0 });
        let gap = target.sub(Bounded::exact(item.prediction));
        total = total.add(weight.mul(gap.mul(gap)));
    }
    total.div(Bounded::exact(items.len() as f64))
}

/// [`log_score`]'s finite operations with running bounds on the weights.
fn bounded_log_score(items: &[Scored], weights: &[Bounded]) -> Bounded {
    let mut total = Bounded::exact(0.0);
    for (item, weight) in items.iter().zip(weights).filter(|pair| pair.0.weight > 0.0) {
        let probability = if item.case() {
            Bounded::exact(item.prediction)
        } else {
            Bounded::exact(1.0).sub(Bounded::exact(item.prediction))
        };
        total = total.add(weight.negate().mul(probability.ln()));
    }
    total.div(Bounded::exact(items.len() as f64))
}

/// [`TruthLaw::forward_paths`]' Rao-Blackwellised recursion with a running bound, for a law whose
/// intensities ignore the state (decoders `[1, 0, 0, 0]`), so one path is exact. Ages on the step
/// grid and the signatures' softplus values are inputs.
fn bounded_forward_incidence(
    law: &TruthLaw,
    person: &Person,
    state: &MarkovState,
    s: f64,
    target: usize,
    horizons: &[f64],
) -> Vec<Bounded> {
    let one = Bounded::exact(1.0);
    let dt = one.div(Bounded::exact(law.steps_per_year as f64));
    let rate_of = |mark: usize, age: f64| {
        let baseline = Bounded::exact(law.log_rate[mark])
            .add(Bounded::exact(law.age_slope[mark]).mul(Bounded::exact(age).sub(Bounded::exact(law.reference_age))))
            .add(Bounded::exact(law.sex_effect[mark]).mul(Bounded::exact(person.sex)));
        let mut activity = Bounded::exact(law.decoder[mark][0]);
        for k in 0..SIGNATURES {
            activity = activity.add(Bounded::exact(law.decoder[mark][k + 1]).mul(Bounded::exact(softplus(state.signatures[k]))));
        }
        baseline.exp().mul(activity)
    };
    let start = law.step_of(s);
    let last = law.step_of(horizons[horizons.len() - 1]);
    let mut unresolved = one;
    let mut incidence = Bounded::exact(0.0);
    let mut values = Vec::with_capacity(horizons.len());
    for n in start..last {
        let age = law.age_at(n);
        let death_rate = rate_of(DEATH, age);
        let (rate, total) = if target < DISEASES {
            let rate = rate_of(target, age);
            (rate, rate.add(death_rate))
        } else {
            (death_rate, death_rate)
        };
        let leave = total.negate().mul(dt).exp_m1().negate();
        incidence = incidence.add(unresolved.mul(rate).div(total).mul(leave));
        unresolved = unresolved.mul(one.sub(leave));
        while values.len() < horizons.len() && law.step_of(horizons[values.len()]) == n + 1 {
            values.push(incidence);
        }
    }
    values
}

/// A subject entering at 40 with records from 40, one possible diagnosis of disease 1,
/// no visits and an administrative end at 55. Times are years since entry.
fn toy_record(onset: Option<f64>, death: Option<f64>, exit: f64) -> Record {
    let entry = 40.0;
    Record {
        person: Person {
            sex: 0.0,
            pcs: [0.0; PCS],
            scores: [0.0; SCORES],
            scores_observed: true,
            entry_age: entry,
            record_start: entry,
        },
        events: onset.map(|time| vec![(entry + time, 0)]).unwrap_or_default(),
        death: death.map(|time| entry + time),
        exit: entry + exit,
        administrative_end: entry + 15.0,
        visits: Vec::new(),
        landmark_states: Vec::new(),
        censor: entry + exit,
    }
}

fn toy_scored(records: &[Record], target: usize, predictions: &[f64], horizon: f64) -> (Vec<Scored>, CensoringSurvival) {
    let censoring = CensoringSurvival::from_records(records.iter(), 0.0);
    let items = records
        .iter()
        .zip(predictions)
        .map(|(record, prediction)| scored(record, target, *prediction, horizon, &censoring))
        .collect();
    (items, censoring)
}

fn self_check() -> usize {
    let mut checks = Checks {
        failures: 0,
        monte_carlo: Vec::new(),
    };

    // Reverse Kaplan-Meier by hand: censored exits at 1, 3 and 4 with a death at 2.
    let censoring = CensoringSurvival::from_exits(vec![(1.0, true), (2.0, false), (3.0, true), (4.0, true)]);
    checks.close("reverse-km-left-of-first", censoring.at(1.0, true), 1.0, 0.0);
    // 1 - 1/4 and 3/4 (1 - 1/2) are exact in binary.
    checks.close("reverse-km-at-first", censoring.at(1.0, false), 0.75, 0.0);
    checks.close("reverse-km-through-death", censoring.at(2.5, false), 0.75, 0.0);
    checks.close("reverse-km-at-third", censoring.at(3.0, false), 0.375, 0.0);
    checks.close("reverse-km-left-of-third", censoring.at(3.0, true), 0.75, 0.0);
    checks.close("reverse-km-at-last", censoring.at(4.0, false), 0.0, 0.0);

    // Aalen-Johansen by hand. Without censoring before the horizon it is the case
    // fraction. With a censoring at 2.5, the later case carries survival 0.6 over two at
    // risk.
    let uncensored = [
        toy_record(Some(1.0), None, 10.0),
        toy_record(None, Some(2.0), 2.0),
        toy_record(Some(3.0), None, 10.0),
        toy_record(None, None, 10.0),
        toy_record(None, None, 10.0),
    ];
    // The literal 0.4 is within half an ulp of 2/5.
    checks.close(
        "aalen-johansen-uncensored",
        aalen_johansen(&uncensored, 0.0, 0, &[4.0])[0],
        0.4,
        f64::EPSILON * (bounded_incidence(&landmark_exits(&uncensored, 0.0, 0), 4.0, None).mu + 0.4 / 2.0),
    );
    let censored = [
        toy_record(Some(1.0), None, 10.0),
        toy_record(None, Some(2.0), 2.0),
        toy_record(None, None, 2.5),
        toy_record(Some(3.0), None, 10.0),
        toy_record(None, None, 10.0),
    ];
    let incidence = aalen_johansen(&censored, 0.0, 0, &[0.5, 1.0, 2.9, 4.0]);
    checks.close("aalen-johansen-before-first", incidence[0], 0.0, 0.0);
    // 1.0 / 5.0 is the double nearest 0.2, so the first increment is exact.
    checks.close("aalen-johansen-at-first", incidence[1], 0.2, 0.0);
    checks.close("aalen-johansen-before-censored-case", incidence[2], 0.2, 0.0);
    checks.close(
        "aalen-johansen-censored",
        incidence[3],
        0.5,
        f64::EPSILON * bounded_incidence(&landmark_exits(&censored, 0.0, 0), 4.0, None).mu,
    );
    // The death target: one death at 2 among five subjects, no censoring before 4.
    checks.close("aalen-johansen-death", aalen_johansen(&uncensored, 0.0, DEATH_TARGET, &[4.0])[0], 0.2, 0.0);
    // Pseudo-observations: without censoring they are the case indicators. With the
    // censoring at 2.5, leaving out each subject gives F^(-i) = 0.375, 0.625, 0.5, 0.25
    // and 0.75, so `5 (0.5) - 4 F^(-i)` = 1, 0, 0.5, 1.5 and -0.5.
    let pseudo = pseudo_observations(&uncensored, 0.0, 0, 4.0, &[0, 1, 2, 3, 4]);
    for (index, expected) in [1.0, 0.0, 1.0, 0.0, 0.0].iter().enumerate() {
        let bound = f64::EPSILON * bounded_pseudo(&uncensored, 0, 4.0, index).mu;
        checks.close(&format!("pseudo-uncensored-{index}"), pseudo[index], *expected, bound);
    }
    let pseudo = pseudo_observations(&censored, 0.0, 0, 4.0, &[0, 1, 2, 3, 4]);
    for (index, expected) in [1.0, 0.0, 0.5, 1.5, -0.5].iter().enumerate() {
        let bound = f64::EPSILON * bounded_pseudo(&censored, 0, 4.0, index).mu;
        checks.close(&format!("pseudo-censored-{index}"), pseudo[index], *expected, bound);
    }

    // A perfect and a constant predictor on uncensored outcomes.
    let perfect: Vec<f64> = uncensored
        .iter()
        .map(|record| if record.first_occurrence(0).is_some_and(|time| time <= 44.0) { 1.0 } else { 0.0 })
        .collect();
    let (items, censoring) = toy_scored(&uncensored, 0, &perfect, 4.0);
    checks.close("perfect-brier", brier(&items), 0.0, 0.0);
    checks.close("perfect-auc", cumulative_dynamic_auc(&items, false), 1.0, 0.0);
    checks.close("perfect-auc-survivors", cumulative_dynamic_auc(&items, true), 1.0, 0.0);
    // Seven comparable pairs, all concordant except the two cases' tie (both predicted 1).
    checks.close("perfect-cindex", concordance(&items, &censoring), 13.0 / 14.0, 0.0);
    // A constant dyadic prediction 3/8 for two cases and three controls, so the literals are exact:
    // Brier (2 (5/8)^2 + 3 (3/8)^2) / 5 and log score -(2 ln 3/8 + 3 ln 5/8) / 5.
    let (items, censoring) = toy_scored(&uncensored, 0, &[0.375; 5], 4.0);
    let ones = vec![Bounded::exact(1.0); items.len()];
    let expected = Bounded::exact(2.0 * 0.625 * 0.625 + 3.0 * 0.375 * 0.375).div(Bounded::exact(5.0));
    checks.close(
        "constant-brier",
        brier(&items),
        expected.value,
        f64::EPSILON * (bounded_brier(&items, &ones).mu + expected.mu),
    );
    let expected = Bounded::exact(2.0)
        .mul(Bounded::exact(0.375).ln())
        .add(Bounded::exact(3.0).mul(Bounded::exact(0.625).ln()))
        .negate()
        .div(Bounded::exact(5.0));
    checks.close(
        "constant-log-score",
        log_score(&items).0,
        expected.value,
        f64::EPSILON * (bounded_log_score(&items, &ones).mu + expected.mu),
    );
    checks.close("constant-auc", cumulative_dynamic_auc(&items, false), 0.5, 0.0);
    checks.close("constant-cindex", concordance(&items, &censoring), 0.5, 0.0);
    checks.close("constant-slope-is-nan", f64::from(u8::from(recalibration(&items, true).1.is_nan())), 1.0, 0.0);

    // IPCW by hand with the censoring at 2.5, so G = 3/4 on [2.5, 10). The case at 1 and
    // the death at 2 weigh 1; the case at 3 weighs 1/G(3-) = 4/3; the control weighs
    // 1/G(4-) = 4/3; the censored subject weighs 0; five subjects.
    let (items, censoring) = toy_scored(&censored, 0, &[0.875, 0.125, 0.5, 0.625, 0.25], 4.0);
    let one = Bounded::exact(1.0);
    let three_quarters = one.mul(one.sub(one.div(Bounded::exact(4.0))));
    let heavy = one.div(three_quarters);
    let weights = [one, one, Bounded::exact(0.0), heavy, heavy];
    let expected = Bounded::exact(0.125 * 0.125 + 0.125 * 0.125)
        .add(heavy.mul(Bounded::exact(0.375 * 0.375 + 0.25 * 0.25)))
        .div(Bounded::exact(5.0));
    checks.close(
        "ipcw-brier",
        brier(&items),
        expected.value,
        f64::EPSILON * (bounded_brier(&items, &weights).mu + expected.mu),
    );
    // The weights 4/3 round. Their rounding (the reverse Kaplan-Meier factor 1 - 1/4 and the
    // division 1/G) is carried through the AUC's sums as a running bound, and the bar is eps mu.
    let one = Bounded::exact(1.0);
    let three_quarters = one.mul(one.sub(one.div(Bounded::exact(4.0))));
    let heavy = one.div(three_quarters);
    let every_control = bounded_auc(&[(0.1, false, one), (0.2, false, heavy), (0.6, true, heavy), (0.9, true, one)]);
    checks.close("ipcw-auc", cumulative_dynamic_auc(&items, false), 1.0, f64::EPSILON * every_control.mu);
    let survivors = bounded_auc(&[(0.2, false, heavy), (0.6, true, heavy), (0.9, true, one)]);
    checks.close("ipcw-auc-survivors", cumulative_dynamic_auc(&items, true), 1.0, f64::EPSILON * survivors.mu);
    checks.close("ipcw-cindex", concordance(&items, &censoring), 1.0, 0.0);
    // A death ranked above a case separates the two AUCs: cases 0.9 (weight 1) and 0.6
    // (4/3) against the death at 0.7 (1) and the control at 0.2 (4/3).
    let (items, censoring) = toy_scored(&censored, 0, &[0.875, 0.75, 0.5, 0.625, 0.25], 4.0);
    let route = bounded_auc(&[(0.25, false, heavy), (0.625, true, heavy), (0.75, false, one), (0.875, true, one)]);
    let total = one.add(heavy);
    let expected = one.mul(one.add(heavy)).add(heavy.mul(heavy)).div(total.mul(total));
    checks.close(
        "ipcw-auc-death-above-case",
        cumulative_dynamic_auc(&items, false),
        expected.value,
        f64::EPSILON * (route.mu + expected.mu),
    );
    let route = bounded_auc(&[(0.25, false, heavy), (0.625, true, heavy), (0.875, true, one)]);
    checks.close(
        "ipcw-auc-survivors-death-above-case",
        cumulative_dynamic_auc(&items, true),
        1.0,
        f64::EPSILON * route.mu,
    );
    // Wolbers by hand. The case at 1 against the death at 2, the censored subject, the
    // case at 3 and the control: all at risk beyond 1, weight 1, all concordant. The case
    // at 3 against the earlier death (weight 1/(G(3-) G(2-)) = 4/3, discordant) and the
    // control (1/(G(3-) G(3)) = 16/9, concordant).
    // The route's own pair weights, in production's order: lead 1/G(1-) against four subjects at
    // risk (lead / G(1)), then lead 1/G(3-) against the death (lead / G(2-), discordant) and the
    // control (lead / G(3)).
    let early = one.div(one);
    let late = one.div(three_quarters);
    let pairs = [
        (early.div(one), 1.0),
        (early.div(one), 1.0),
        (early.div(one), 1.0),
        (early.div(one), 1.0),
        (late.div(one), 0.0),
        (late.div(three_quarters), 1.0),
    ];
    let (mut numerator, mut denominator) = (Bounded::exact(0.0), Bounded::exact(0.0));
    for (weight, concordant_value) in pairs {
        numerator = numerator.add(weight.mul(Bounded::exact(concordant_value)));
        denominator = denominator.add(weight);
    }
    let route = numerator.div(denominator);
    let sixteen_ninths = Bounded::exact(16.0).div(Bounded::exact(9.0));
    let expected = Bounded::exact(4.0)
        .add(sixteen_ninths)
        .div(Bounded::exact(4.0).add(Bounded::exact(4.0).div(Bounded::exact(3.0))).add(sixteen_ninths));
    checks.close(
        "ipcw-cindex-death-above-case",
        concordance(&items, &censoring),
        expected.value,
        f64::EPSILON * (route.mu + expected.mu),
    );

    // Constant cause-specific hazards with independent censoring: the target at 0.08,
    // death at 0.04, censoring at 0.05. At u = 5, F_target = 0.08/0.12 (1 - exp(-0.6)) and
    // F_death = 1 - exp(-0.2). The Aalen-Johansen and IPCW incidences recover them, and the
    // IPCW Brier score of the true F recovers F (1 - F), within the self-check's z standard
    // errors.
    let mut rng = Rng::new(20260918, 3);
    let mut population = Vec::new();
    while population.len() < 20000 {
        let onset = rng.exponential() / 0.08;
        let death = rng.exponential() / 0.04;
        let loss = rng.exponential() / 0.05;
        let exit = death.min(loss);
        population.push(toy_record((onset < exit).then_some(onset), (death < loss).then_some(death), exit));
    }
    let n = population.len() as f64;
    for (target, total, share) in [(0usize, 0.12f64, 0.08 / 0.12), (DEATH_TARGET, 0.04, 1.0)] {
        let truth = share * -(-total * 5.0f64).exp_m1();
        let items = toy_scored(&population, target, &vec![truth; population.len()], 5.0).0;
        let weighted: Vec<f64> = items.iter().map(|item| if item.case() { item.weight } else { 0.0 }).collect();
        let incidence_se = standard_deviation(&weighted) / n.sqrt();
        let name = target_name(target);
        checks.monte_carlo(&format!("constant-hazards-{name}-ipcw-incidence"), mean(&weighted), truth, incidence_se, 0.0);
        checks.monte_carlo(
            &format!("constant-hazards-{name}-aalen-johansen"),
            aalen_johansen(&population, 0.0, target, &[5.0])[0],
            truth,
            incidence_se,
            0.0,
        );
        let losses: Vec<f64> = items
            .iter()
            .map(|item| item.weight * ((if item.case() { 1.0 } else { 0.0 }) - item.prediction).powi(2))
            .collect();
        checks.monte_carlo(
            &format!("constant-hazards-{name}-ipcw-brier"),
            brier(&items),
            truth * (1.0 - truth),
            standard_deviation(&losses) / n.sqrt(),
            0.0,
        );
        checks.close(&format!("constant-hazards-{name}-auc-ties"), cumulative_dynamic_auc(&items, false), 0.5, 0.0);
    }

    // Recalibration: a calibrated predictor recovers intercept 0 and slope 1, and an
    // overconfident one (logit doubled) slope 1/2, each within z inverse-Fisher standard errors.
    let mut rng = Rng::new(20260917, 1);
    let mut calibrated = Vec::new();
    let mut overconfident = Vec::new();
    while calibrated.len() < 20000 {
        let logit_value = -1.0 + rng.normal();
        let outcome = if rng.bernoulli(sigmoid(logit_value)) {
            Outcome::Case(1.0)
        } else {
            Outcome::Control
        };
        calibrated.push(Scored { prediction: sigmoid(logit_value), outcome, weight: 1.0, at_risk_until: 1.0 });
        overconfident.push(Scored { prediction: sigmoid(2.0 * logit_value), outcome, weight: 1.0, at_risk_until: 1.0 });
    }
    let (a, b) = recalibration(&calibrated, false);
    checks.monte_carlo("calibrated-intercept", a, 0.0, recalibration_standard_error(&calibrated, a, b, false), 0.0);
    let (a, b) = recalibration(&calibrated, true);
    checks.monte_carlo("calibrated-slope", b, 1.0, recalibration_standard_error(&calibrated, a, b, true), 0.0);
    let (a, b) = recalibration(&overconfident, true);
    checks.monte_carlo("overconfident-slope", b, 0.5, recalibration_standard_error(&overconfident, a, b, true), 0.0);

    // Channel likelihoods are normalised: the ordinal levels' probabilities, and in S4
    // attendance and nonattendance, each sum to one.
    let law = TruthLaw::new(Scenario::Joint);
    let informative = TruthLaw::new(Scenario::InformativeVisits);
    let state = MarkovState {
        signatures: [0.4, -0.7, 1.1],
        diagnosed: [false; DISEASES],
    };
    let person = toy_record(None, None, 15.0).person;
    let ordinal: f64 = (0..ORDINAL_LEVELS).map(|level| law.ordinal_probability(&state, level)).sum();
    // The levels' probabilities telescope over the same normal_cdf values, so their sum is one up
    // to the sums' running bound, whatever erfc's accuracy.
    let predictor = TruthLaw::linear(&law.ordinal_loading, &state);
    let mut telescoped = Bounded::exact(0.0);
    for level in 0..ORDINAL_LEVELS {
        let upper = if level + 1 < ORDINAL_LEVELS {
            normal_cdf(law.ordinal_thresholds[level] - predictor)
        } else {
            1.0
        };
        let lower = if level > 0 {
            normal_cdf(law.ordinal_thresholds[level - 1] - predictor)
        } else {
            0.0
        };
        telescoped = telescoped.add(Bounded::exact(upper).sub(Bounded::exact(lower)));
    }
    checks.close("ordinal-probabilities-sum", ordinal, 1.0, f64::EPSILON * telescoped.mu);
    let absent = Visit {
        age: 41.0,
        attended: false,
        values: [f64::NAN; CHANNELS],
    };
    let attended = Visit {
        attended: true,
        ..absent
    };
    let total = informative.visit_log_likelihood(&person, &state, &attended).exp()
        + informative.visit_log_likelihood(&person, &state, &absent).exp();
    let linear = Bounded::exact(informative.attendance_intercept)
        .add(Bounded::exact(informative.attendance_sex).mul(Bounded::exact(person.sex)))
        .add(Bounded::exact(informative.attendance_pc2).mul(Bounded::exact(person.pcs[1])))
        .add(Bounded::exact(informative.attendance_signature).mul(Bounded::exact(state.signatures[0])));
    let (probability, probability_bound) = sigmoid_bound(linear.value, linear.mu);
    let probability = Bounded { value: probability, mu: probability_bound };
    let summed = Bounded::exact(0.0)
        .add(probability.ln())
        .exp()
        .add(Bounded::exact(0.0).add(Bounded::exact(1.0).sub(probability).ln()).exp());
    checks.close("attendance-probabilities-sum", total, 1.0, f64::EPSILON * summed.mu);

    // With intensities that ignore the state, age and sex, F_d(s, s+u) = lambda_d / Lambda
    // (1 - exp(-Lambda u)) and F_death = 1 - exp(-lambda_death u), exactly. That holds for
    // the forward paths, and for the filter: its weights move with the channel values, but
    // every particle forecasts the same value. The filter must also map landmarks, targets
    // and horizons into the right cells. Records start at the origin, so no particle can
    // carry an unrecorded diagnosis.
    let mut constant = TruthLaw::new(Scenario::Joint);
    constant.decoder = [[1.0, 0.0, 0.0, 0.0]; MARKS];
    constant.age_slope = [0.0; MARKS];
    constant.sex_effect = [0.0; MARKS];
    let grid = Grid {
        offsets: vec![0.0, 2.0, 5.0],
        horizons: vec![1.0, 5.0, 10.0],
    };
    let closed_form = |target: usize, u: f64| {
        let death = Bounded::exact(constant.log_rate[DEATH]).exp();
        if target < DISEASES {
            let rate = Bounded::exact(constant.log_rate[target]).exp();
            let total = rate.add(death);
            rate.div(total).mul(total.negate().mul(Bounded::exact(u)).exp_m1().negate())
        } else {
            death.negate().mul(Bounded::exact(u)).exp_m1().negate()
        }
    };
    let horizons = [43.0, 47.0, 52.0];
    for target in [1, DEATH_TARGET] {
        let paths = constant.forward_paths(&person, &state, 42.0, target, &horizons, 3, &mut Rng::new(7, 7));
        let route = bounded_forward_incidence(&constant, &person, &state, 42.0, target, &horizons);
        for (ui, horizon) in horizons.iter().enumerate() {
            let exact = closed_form(target, horizon - 42.0);
            checks.close(
                &format!("forward-constant-{}-h{horizon}", target_name(target)),
                paths[2][ui],
                exact.value,
                f64::EPSILON * (route[ui].mu + exact.mu),
            );
        }
    }
    let mut record = toy_record(None, None, 15.0);
    record.person.record_start = constant.origin_age;
    let mut visit = Visit {
        age: 31.0,
        attended: true,
        values: [0.2, f64::NAN, 1.0, 2.0, 3.0],
    };
    record.visits.push(visit);
    visit.age = 33.0;
    visit.values = [-1.1, 0.7, 0.0, 0.0, 1.0];
    record.visits.push(visit);
    let filtered = constant.filtered_forecasts(&record, &grid, 32, 8, true, Mutation::Faithful, &mut Rng::new(13, 13));
    for (li, offset) in grid.offsets.iter().enumerate() {
        let s = record.person.entry_age + offset;
        let landmark_horizons: Vec<f64> = grid.horizons.iter().map(|u| s + u).collect();
        for target in 0..TARGETS {
            let route = bounded_forward_incidence(&constant, &record.person, &state, s, target, &landmark_horizons);
            for (ui, u) in grid.horizons.iter().enumerate() {
                let exact = closed_form(target, *u);
                // The filter's forecast is the mean of 8 identical forward paths.
                let mut average = Bounded::exact(0.0);
                let mut summed = 0;
                while summed < 8 {
                    average = average.add(route[ui]);
                    summed += 1;
                }
                let average = average.div(Bounded::exact(8.0));
                let slot = (li * TARGETS + target) * grid.horizons.len() + ui;
                checks.close(
                    &format!("filter-constant-s{offset}-{}-u{u}", target_name(target)),
                    filtered.values[slot],
                    exact.value,
                    f64::EPSILON * (average.mu + exact.mu),
                );
            }
        }
    }

    // Discretisation, deterministically. With intensities that depend on age only (static
    // state, constant decoders, Gompertz slopes), one weekly forward path is the discretised
    // law's exact incidence. A left-endpoint rate misses at most dt times its growth over the
    // window, so death's incidence moves by at most dt (lambda_death(s+u) - lambda_death(s)),
    // and a disease's by at most 2 dt (Lambda(s+u) - Lambda(s)); add quadrature and rounding.
    // The bound needs both rates monotone over the window, which positive Gompertz slopes give.
    let mut gompertz = TruthLaw::new(Scenario::Joint);
    gompertz.kappa = [0.0; SIGNATURES];
    gompertz.decoder = [[1.0, 0.0, 0.0, 0.0]; MARKS];
    gompertz.sex_effect = [0.0; MARKS];
    gompertz.jump = [[0.0; SIGNATURES]; DISEASES];
    let static_state = MarkovState {
        signatures: [0.0; SIGNATURES],
        diagnosed: [false; DISEASES],
    };
    let s = 66.0;
    let horizons = [67.0, 71.0, 76.0];
    for target in [0, DEATH_TARGET] {
        let weekly = gompertz.forward_paths(&person, &static_state, s, target, &horizons, 1, &mut Rng::new(23, 1));
        let rate = |mark: usize, age: f64| (gompertz.log_rate[mark] + gompertz.age_slope[mark] * (age - gompertz.reference_age)).exp();
        let growth = |age: f64| if target < DISEASES { 2.0 * (rate(target, age) + rate(DEATH, age)) } else { rate(DEATH, age) };
        for (ui, horizon) in horizons.iter().enumerate() {
            let (exact, quadrature, exact_bound) = gompertz_incidence(&gompertz, target, s, horizon - s);
            let route = bounded_forward_incidence(&gompertz, &person, &static_state, s, target, &horizons);
            let bound = gompertz.step_length() * (growth(*horizon) - growth(s))
                + quadrature
                + f64::EPSILON * (route[ui].mu + exact_bound);
            let name = target_name(target);
            report(format!(
                "[a12-discretisation] target={name} s={s} horizon={horizon} weekly={:.12e} exact={exact:.12e} bias={:.3e} \
                 bound={bound:.3e} quadrature_error={quadrature:.3e}",
                weekly[0][ui],
                weekly[0][ui] - exact,
            ));
            checks.close(&format!("discretisation-{name}-h{horizon}"), weekly[0][ui], exact, bound);
        }
    }

    // The filter against an exact posterior. A static one-signature law makes F* a ratio of
    // one-dimensional integrals: the prior over the signature, survival from the origin, the
    // recorded diagnosis at 41, and for every other disease the mixture of an unrecorded
    // pre-record diagnosis (F = 0) and a free path. Records start at 38, after the origin, so the
    // pre-record integration is exercised. Two visits carry Student-t laboratory values, so the
    // measurement weighting that separates oracle-hs from oracle-hs-dx meets the exact posterior.
    // The negative controls drop the event term, the pre-record survival weight or the visit
    // term, and each must fail some cell.
    let mut static_law = TruthLaw::new(Scenario::Joint);
    static_law.kappa = [0.0; SIGNATURES];
    static_law.genetic_drive = [[0.0; SCORES]; SIGNATURES];
    static_law.genetic_drive_trend = [[0.0; SCORES]; SIGNATURES];
    static_law.sex_drive = [0.0; SIGNATURES];
    static_law.drift_trend = [0.0; SIGNATURES];
    static_law.jump = [[0.0; SIGNATURES]; DISEASES];
    static_law.age_slope = [0.0; MARKS];
    static_law.sex_effect = [0.0; MARKS];
    static_law.decoder = [[1.0, 0.0, 0.0, 0.0]; MARKS];
    static_law.decoder[0] = [0.2, 0.8, 0.0, 0.0];
    static_law.decoder[1] = [0.3, 0.7, 0.0, 0.0];
    static_law.decoder[DEATH] = [0.2, 0.8, 0.0, 0.0];
    static_law.log_rate[0] = 0.03f64.ln();
    static_law.log_rate[1] = 0.03f64.ln();
    static_law.log_rate[DEATH] = 0.08f64.ln();
    // The exact posterior's laboratory density is a t density only while lab 1 loads the first
    // signature alone.
    static_law.lab_loading[LAB_A] = [0.8, 0.0, 0.0];
    checks.close(
        "static-law-lab-loads-first-signature-only",
        static_law.lab_loading[LAB_A][1].abs() + static_law.lab_loading[LAB_A][2].abs(),
        0.0,
        0.0,
    );
    let mut record = toy_record(Some(1.0), None, 15.0);
    record.person.record_start = 38.0;
    let mut lab_visit = Visit {
        age: 38.0,
        attended: true,
        values: [f64::NAN; CHANNELS],
    };
    lab_visit.values[LAB_A] = 0.9;
    record.visits.push(lab_visit);
    lab_visit.age = 40.0;
    lab_visit.values[LAB_A] = 1.4;
    record.visits.push(lab_visit);
    let static_grid = Grid {
        offsets: vec![2.0],
        horizons: vec![1.0, 5.0, 10.0],
    };
    let landmark = record.person.entry_age + static_grid.offsets[0];
    let exact = static_posterior(&static_law, &record, landmark, &static_grid.horizons);
    let replicates = 16;
    let filter_cells = |mutation: Mutation, seed: u64| {
        let runs: Vec<FilterOutcome> = (0..replicates)
            .map(|r| static_law.filtered_forecasts(&record, &static_grid, 512, 256, true, mutation, &mut Rng::new(seed, r as u64)))
            .collect();
        let mut cells = Vec::new();
        for (ui, row) in exact.iter().enumerate() {
            for (target, cell) in row.iter().enumerate() {
                if record.diagnosed_by(target, landmark) {
                    continue;
                }
                let slot = target * static_grid.horizons.len() + ui;
                let estimates: Vec<f64> = runs.iter().map(|run| run.values[slot]).collect();
                let error = standard_deviation(&estimates) / (replicates as f64).sqrt();
                cells.push((target, ui, mean(&estimates), cell.0, error, cell.1));
            }
        }
        cells
    };
    for cell in filter_cells(Mutation::Faithful, 29) {
        let name = format!("filter-static-posterior-{}-u{}", target_name(cell.0), static_grid.horizons[cell.1]);
        checks.monte_carlo(&name, cell.2, cell.3, cell.4, cell.5);
    }
    for (mutation, name, seed) in [
        (Mutation::DropEventTerm, "drop-event-term", 31),
        (Mutation::DropPreRecordSurvival, "drop-pre-record-survival", 37),
        (Mutation::DropVisitTerm, "drop-visit-term", 43),
    ] {
        let cells = filter_cells(mutation, seed).iter().map(|cell| (cell.2, cell.3, cell.4, cell.5)).collect();
        checks.negative_control(&format!("filter-static-posterior-{name}"), cells);
    }

    // The baseline's analytic gradient against a central finite difference along a random
    // direction, on a tiny problem with events, censored and empty windows, and genetics. The bar
    // is the change between steps h and h/2 (an estimate of the truncation error) plus the
    // quotient's rounding from the objective's running bounds. The derivative must exceed the bar,
    // so the check can fail.
    let tiny = AladynProblem {
        subjects: 2,
        signatures: SIGNATURES,
        bins: 6,
        covariates: 2,
        design: vec![0.5, -1.0, -0.3, 0.8],
        start: vec![1, 0],
        stop: vec![4, 6, 2, 6, 1, 6, 6, 3, 6, 1, 6, 6],
        diagnosed: vec![true, false, true, false, false, false, false, true, false, false, false, true],
        logit_prevalence: (0..DISEASES * 6).map(|index| -3.0 - 0.1 * (index % 6) as f64).collect(),
        lambda_precision: rbf_precision(6, 1.5, 1e-2),
        phi_precision: rbf_precision(6, 1.5, 1e-2),
        gp_weight: 1.0,
        gamma_decay: 0.01,
        genetic_scale: 1.0,
    };
    let offsets = tiny.offsets();
    let size = tiny.parameters();
    let mut rng = Rng::new(41, 1);
    let mut point = Vec::with_capacity(size);
    while point.len() < size {
        let draw = rng.normal();
        let index = point.len();
        point.push(if index < offsets.0 {
            0.3 * draw
        } else if index < offsets.1 {
            -2.5 + 0.3 * draw
        } else if index < offsets.3 {
            0.2 * draw
        } else {
            -0.1
        });
    }
    let mut direction = Vec::with_capacity(size);
    while direction.len() < size {
        direction.push(rng.normal());
    }
    let mut gradient = vec![0.0; size];
    let mut magnitude = vec![0.0; size];
    tiny.evaluate(&point, &mut gradient, &mut magnitude);
    // The analytic directional derivative with its running bound: each gradient coordinate
    // carries its own bound through the dot product.
    let mut analytic_bounded = Bounded::exact(0.0);
    for ((component, component_bound), step) in gradient.iter().zip(magnitude.iter()).zip(direction.iter()) {
        let coordinate = Bounded {
            value: *component,
            mu: *component_bound,
        };
        analytic_bounded = analytic_bounded.add(coordinate.mul(Bounded::exact(*step)));
    }
    let analytic = analytic_bounded.value;
    // The difference quotient, and its rounding eps (mu_up + mu_down) / (2h) from the objective's
    // running bounds at the two points.
    let difference = |h: f64| {
        let mut scratch_gradient = vec![0.0; size];
        let mut scratch_bound = vec![0.0; size];
        let forward: Vec<f64> = point.iter().zip(direction.iter()).map(|pair| pair.0 + h * pair.1).collect();
        let backward: Vec<f64> = point.iter().zip(direction.iter()).map(|pair| pair.0 - h * pair.1).collect();
        let (up, up_bound) = tiny
            .evaluate(&forward, &mut scratch_gradient, &mut scratch_bound)
            .unwrap_or((f64::NAN, f64::NAN));
        let (down, down_bound) = tiny
            .evaluate(&backward, &mut scratch_gradient, &mut scratch_bound)
            .unwrap_or((f64::NAN, f64::NAN));
        ((up - down) / (2.0 * h), f64::EPSILON * (up_bound + down_bound) / (2.0 * h))
    };
    // A step near the cube root of machine epsilon balances truncation against rounding.
    let step = f64::EPSILON.cbrt();
    let coarse = difference(step);
    let fine = difference(step / 2.0);
    // The bar also charges the rounding of the comparison itself, eps (|analytic| + |fd|).
    let bar = (coarse.0 - fine.0).abs()
        + fine.1
        + f64::EPSILON * (analytic_bounded.mu + analytic.abs() + fine.0.abs());
    checks.close("aladyn-gradient-directional", fine.0, analytic, bar);
    checks.close("aladyn-gradient-not-vacuous", f64::from(u8::from(analytic.abs() > bar)), 1.0, 0.0);

    // The post-hoc certificate on `x^2 / 2`: an objective refusing its optimum and a NaN gradient
    // coordinate never certify, the optimum certifies, and a resolved coordinate does not.
    let refused = certify(
        &|x, gradient, bound| {
            gradient[0] = x[0];
            bound[0] = x[0].abs();
            if x[0] == 0.0 { None } else { Some((0.5 * x[0] * x[0], x[0] * x[0])) }
        },
        &[0.0],
        1,
    );
    checks.close(
        "certificate-refused-optimum",
        f64::from(u8::from(matches!(refused.2, Certificate::EvaluationRefused))),
        1.0,
        0.0,
    );
    let not_a_number = certify(
        &|x, gradient, bound| {
            gradient[0] = f64::NAN;
            bound[0] = x[0].abs();
            Some((0.5 * x[0] * x[0], x[0] * x[0]))
        },
        &[0.0],
        1,
    );
    checks.close(
        "certificate-nan-gradient",
        f64::from(u8::from(matches!(not_a_number.2, Certificate::NonFiniteGradient { coordinate: 0 }))),
        1.0,
        0.0,
    );
    let quadratic = |x: &[f64], gradient: &mut [f64], bound: &mut [f64]| {
        gradient[0] = x[0];
        bound[0] = x[0].abs();
        Some((0.5 * x[0] * x[0], x[0] * x[0]))
    };
    let optimum = certify(&quadratic, &[0.0], 1);
    checks.close("certificate-optimum-certifies", f64::from(u8::from(optimum.2.certified())), 1.0, 0.0);
    let resolved = certify(&quadratic, &[1.0], 1);
    checks.close(
        "certificate-resolved-gradient",
        f64::from(u8::from(matches!(resolved.2, Certificate::Unresolved { .. }))),
        1.0,
        0.0,
    );
    // A nonzero coordinate inside its band certifies: g = 1 over the bound 2 / eps = 2^53 is
    // exactly one half.
    let in_band = certify(
        &|x, gradient, bound| {
            gradient[0] = x[0];
            bound[0] = 2.0 * x[0].abs() / f64::EPSILON;
            Some((0.5 * x[0] * x[0], x[0] * x[0]))
        },
        &[1.0],
        1,
    );
    checks.close(
        "certificate-in-band-nonzero-certifies",
        f64::from(u8::from(matches!(&in_band.2, Certificate::Certified { worst_ratio } if *worst_ratio == 0.5))),
        1.0,
        0.0,
    );
    let zero_bound = certify(
        &|x, gradient, bound| {
            gradient[0] = x[0];
            bound[0] = 0.0 * x[0];
            Some((0.5 * x[0] * x[0], x[0] * x[0]))
        },
        &[1.0],
        1,
    );
    checks.close(
        "certificate-zero-bound-nonzero-gradient",
        f64::from(u8::from(matches!(&zero_bound.2, Certificate::Unresolved { worst_ratio } if *worst_ratio == f64::INFINITY))),
        1.0,
        0.0,
    );
    let not_a_number_bound = certify(
        &|x, gradient, bound| {
            gradient[0] = x[0];
            bound[0] = f64::NAN;
            Some((0.5 * x[0] * x[0], x[0] * x[0]))
        },
        &[1.0],
        1,
    );
    checks.close(
        "certificate-nan-bound",
        f64::from(u8::from(matches!(not_a_number_bound.2, Certificate::InvalidBound { coordinate: 0 }))),
        1.0,
        0.0,
    );
    let negative_bound = certify(
        &|x, gradient, bound| {
            gradient[0] = x[0];
            bound[0] = -x[0].abs();
            Some((0.5 * x[0] * x[0], x[0] * x[0]))
        },
        &[1.0],
        1,
    );
    checks.close(
        "certificate-negative-bound",
        f64::from(u8::from(matches!(negative_bound.2, Certificate::InvalidBound { coordinate: 0 }))),
        1.0,
        0.0,
    );
    let infinite_value = certify(
        &|x, gradient, bound| {
            gradient[0] = x[0];
            bound[0] = x[0].abs();
            Some((x[0] / 0.0, x[0] * x[0]))
        },
        &[1.0],
        1,
    );
    checks.close(
        "certificate-non-finite-value",
        f64::from(u8::from(matches!(infinite_value.2, Certificate::NonFiniteValue))),
        1.0,
        0.0,
    );
    // The ratio itself, past certify's refusals: 0 / 0 is zero, and a NaN ratio counts as `+inf`.
    checks.close("ratio-zero-over-zero", worst_gradient_over_band(&[0.0], &[0.0]), 0.0, 0.0);
    checks.close(
        "ratio-nan-bound",
        f64::from(u8::from(worst_gradient_over_band(&[1.0], &[f64::NAN]) == f64::INFINITY)),
        1.0,
        0.0,
    );

    // The forward paths against brute-force simulation of the full S1 law from one state.
    let person = Person {
        sex: 1.0,
        pcs: [0.5, -0.2, 0.1, 0.0],
        scores: [0.8, -0.4, 0.3],
        scores_observed: false,
        entry_age: 64.0,
        record_start: 60.0,
    };
    let state = MarkovState {
        signatures: [0.9, 0.2, -0.3],
        diagnosed: [false, true, false, false, false, false],
    };
    let horizons = [67.0, 71.0, 76.0];
    let forward_count = 2000;
    let brute_count = 20000;
    for target in [0, DEATH_TARGET] {
        let paths = law.forward_paths(&person, &state, 66.0, target, &horizons, forward_count, &mut Rng::new(11, 1));
        let simulated = law.simulated_incidence(&person, &state, 66.0, target, &horizons, brute_count, &mut Rng::new(11, 2));
        for (ui, horizon) in horizons.iter().enumerate() {
            let column: Vec<f64> = paths.iter().map(|path| path[ui]).collect();
            let name = target_name(target);
            let variance = standard_deviation(&column).powi(2) / forward_count as f64
                + simulated[ui] * (1.0 - simulated[ui]) / brute_count as f64;
            checks.monte_carlo(&format!("forward-vs-simulation-{name}-h{horizon}"), mean(&column), simulated[ui], variance.sqrt(), 0.0);
        }
    }

    // The one-hospitalisation-per-step cap changes the law by O((lambda dt)^2), so the
    // expected number of hospitalisations over ten years must agree at half the step.
    let mut fine = TruthLaw::new(Scenario::Joint);
    fine.steps_per_year *= 2;
    let coarse = law.simulated_hospitalisations(&person, &state, 66.0, 76.0, brute_count, &mut Rng::new(19, 1));
    let finer = fine.simulated_hospitalisations(&person, &state, 66.0, 76.0, brute_count, &mut Rng::new(19, 2));
    let variance = (standard_deviation(&coarse).powi(2) + standard_deviation(&finer).powi(2)) / brute_count as f64;
    checks.monte_carlo("half-step-hospitalisations-10y", mean(&finer), mean(&coarse), variance.sqrt(), 0.0);

    checks.finish();
    report(format!("[a12-selfcheck] failures={}", checks.failures));
    checks.failures
}

fn main() -> ExitCode {
    let options = match parse_options() {
        Ok(options) => options,
        Err(problem) => {
            eprintln!("a12: {problem}");
            return ExitCode::from(2);
        }
    };
    let failures = self_check();
    if failures > 0 {
        report(format!("A12_DONE status=self-check-failed failures={failures}"));
        return ExitCode::from(1);
    }
    if options.self_check {
        report("A12_DONE status=self-check-passed".to_string());
        return ExitCode::SUCCESS;
    }
    match run(&options) {
        Ok(()) => {
            report("A12_DONE status=ok".to_string());
            ExitCode::SUCCESS
        }
        Err(problem) => {
            report(format!("A12_DONE status=error error={problem}"));
            ExitCode::from(1)
        }
    }
}

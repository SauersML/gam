//! SAEBench-facing and manifold-native evaluation metrics.
//!
//! This module is deliberately data-only: external SAEBench runners can own the
//! model calls, LLM descriptions, and behavioral measurements, then hand their
//! typed observations to these Rust routines.  Keeping the scoring here makes
//! chart-coordinate interpretability and output-Fisher dose calibration share a
//! single audited definition with the steering code instead of reimplementing
//! math in Python notebooks.

use crate::null_battery::{NullKind, NullSummary, Tail, summarize_null_distribution};

/// One row in the chart-interpretability evaluation: a recovered coordinate,
/// its ground-truth cyclic label, and the posterior/evidence weight assigned to
/// that row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChartInterpObservation {
    /// Recovered chart coordinate in turns.  Values are wrapped modulo one.
    pub recovered_turns: f64,
    /// Ground-truth cyclic label in turns.  Values are wrapped modulo one.
    pub label_turns: f64,
    /// Non-negative posterior/evidence weight for this row.
    pub weight: f64,
}

/// Exact statistic calibrated by every chart-interpretability report.
///
/// Naming this in the type and persisted artifact prevents an empirical p-value
/// for a different chart claim (for example, an EV gap or cyclic adjacency)
/// from being paired with this score (#2250).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChartInterpStatistic {
    OrientationQuotientedWeightedPhaseLock,
}

impl ChartInterpStatistic {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OrientationQuotientedWeightedPhaseLock => {
                "orientation_quotiented_weighted_phase_lock_v1"
            }
        }
    }
}

/// How one complete null observation ledger is produced.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChartInterpNullDrawPolicy {
    /// Regenerate the declared surrogate, refit the chart, and run the same
    /// coordinate readout before evaluating the statistic.
    RegenerateRefitAndReadout,
}

impl ChartInterpNullDrawPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RegenerateRefitAndReadout => {
                "regenerate_surrogate_refit_chart_and_readout_each_draw"
            }
        }
    }
}

/// Closed chart-null protocols understood by the scorer.
///
/// A protocol owns its native [`NullKind`] and draw policy. Callers therefore
/// cannot label arbitrary rows "matched spectrum" while separately selecting a
/// contradictory policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChartInterpNullProtocol {
    MatchedSpectrumGaussianChartRefitV1,
}

impl ChartInterpNullProtocol {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "matched_spectrum_gaussian_chart_refit_v1" => {
                Ok(Self::MatchedSpectrumGaussianChartRefitV1)
            }
            other => Err(format!(
                "chart_interp: unsupported null protocol {other:?}; expected matched_spectrum_gaussian_chart_refit_v1"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::MatchedSpectrumGaussianChartRefitV1 => {
                "matched_spectrum_gaussian_chart_refit_v1"
            }
        }
    }

    pub fn null_kind(self) -> NullKind {
        match self {
            Self::MatchedSpectrumGaussianChartRefitV1 => NullKind::MatchedSpectrumGaussian,
        }
    }

    pub fn draw_policy(self) -> ChartInterpNullDrawPolicy {
        match self {
            Self::MatchedSpectrumGaussianChartRefitV1 => {
                ChartInterpNullDrawPolicy::RegenerateRefitAndReadout
            }
        }
    }
}

/// Complete null calibration input. There is intentionally no constructor from
/// scalar statistics: the scorer accepts full per-draw observation ledgers and
/// recomputes [`ChartInterpStatistic`] itself.
#[derive(Clone, Debug, PartialEq)]
pub struct ChartInterpNullCalibration {
    protocol: ChartInterpNullProtocol,
    seed: u64,
    expected_draws: usize,
    observation_draws: Vec<Vec<ChartInterpObservation>>,
}

impl ChartInterpNullCalibration {
    pub fn new(
        protocol: ChartInterpNullProtocol,
        seed: u64,
        expected_draws: usize,
        observation_draws: Vec<Vec<ChartInterpObservation>>,
    ) -> Result<Self, String> {
        if expected_draws == 0 {
            return Err("chart_interp: null calibration requires at least one draw".into());
        }
        if observation_draws.len() != expected_draws {
            return Err(format!(
                "chart_interp: null protocol declares {expected_draws} draws but artifact contains {} complete observation ledgers",
                observation_draws.len()
            ));
        }
        Ok(Self {
            protocol,
            seed,
            expected_draws,
            observation_draws,
        })
    }

    pub fn protocol(&self) -> ChartInterpNullProtocol {
        self.protocol
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn expected_draws(&self) -> usize {
        self.expected_draws
    }

    pub fn observation_draws(&self) -> &[Vec<ChartInterpObservation>] {
        &self.observation_draws
    }
}

/// Observed value of [`ChartInterpStatistic`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChartInterpStatisticValue {
    /// Orientation-quotiented weighted cyclic phase-lock score in `[0, 1]`.
    pub circular_correlation: f64,
    /// Signed correlation before orientation is quotiented out.
    pub signed_circular_correlation: f64,
    /// Sum of accepted observation weights.
    pub effective_weight: f64,
}

/// Provenance and native null-battery distribution for a chart statistic.
#[derive(Clone, Debug, PartialEq)]
pub struct ChartInterpNullCalibrationReport {
    /// The statistic recomputed for the observation and every null draw.
    pub statistic: ChartInterpStatistic,
    /// Closed generator/refit/readout protocol.
    pub protocol: ChartInterpNullProtocol,
    /// Seed from which draw-index-specific surrogates were generated.
    pub seed: u64,
    /// Native null distribution, including kind, tail, summary, Monte Carlo
    /// uncertainty, extreme count, and every statistic sample in draw order.
    pub null_distribution: NullSummary,
}

/// Evidential decision after null calibration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChartInterpVerdict {
    Pass,
    NullCompatible,
}

impl ChartInterpVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::NullCompatible => "null_compatible",
        }
    }
}

/// Null-calibrated chart-interpretability artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct ChartInterpReport {
    pub statistic: ChartInterpStatistic,
    pub observed: ChartInterpStatisticValue,
    pub calibration: ChartInterpNullCalibrationReport,
    /// Significance level used for the evidential verdict.
    pub significance_level: f64,
    pub verdict: ChartInterpVerdict,
}

/// One dose-response calibration point along a steered arc.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoseResponseObservation {
    /// Arc-length coordinate or any unit-speed path coordinate used for the move.
    pub arc_length: f64,
    /// Local output-Fisher prediction from `steer_delta`, in nats.
    pub predicted_nats: f64,
    /// Measured KL / behavior change in nats.
    pub measured_nats: f64,
    /// Non-negative posterior/evidence weight for this row or intervention.
    pub weight: f64,
}

/// Weighted calibration fit of measured nats on predicted output-Fisher nats.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoseResponseCalibrationReport {
    /// Slope of the no-intercept weighted least-squares fit
    /// `measured_nats = slope * predicted_nats`.
    pub slope_through_origin: f64,
    /// Weighted R² of that no-intercept calibration fit.
    pub r2_through_origin: f64,
    /// Weighted mean of `measured_nats / arc_length²` for non-zero arcs.
    pub mean_measured_nats_per_arc_squared: f64,
    /// Weighted coefficient of variation of `measured_nats / arc_length²`.
    pub cv_measured_nats_per_arc_squared: f64,
    /// Sum of accepted observation weights.
    pub effective_weight: f64,
}

/// Gaussian posterior summary for one token/chart coordinate block from an
/// already-factorized row-Hessian precision block.
#[derive(Clone, Debug, PartialEq)]
pub struct CoordinatePosterior {
    /// Posterior mean coordinate supplied by the fit/encoder.
    pub mean: Vec<f64>,
    /// Diagonal of the covariance matrix, i.e. the inverse row-Hessian diagonal.
    pub covariance_diag: Vec<f64>,
    /// Trace of the covariance matrix, used as an uncertainty weight source.
    pub covariance_trace: f64,
    /// Precision-weighted evidence mass `1 / trace(covariance)`.
    pub precision_weight: f64,
}

/// Score chart-coordinate interpretability against cyclic labels and an explicit
/// matched-spectrum null distribution.
///
/// Every null draw is a complete observation ledger produced by the same chart
/// fit/readout protocol on one matched-spectrum surrogate. The exact
/// orientation-quotiented statistic used for the observed ledger is recomputed
/// for every null ledger. Requiring the draws here makes it impossible for the
/// public scorer to present a large circular correlation as evidence without
/// calibrating that same number against its matched null (#2250).
pub fn chart_interp_score(
    observations: &[ChartInterpObservation],
    null_calibration: &ChartInterpNullCalibration,
    significance_level: f64,
) -> Result<ChartInterpReport, String> {
    if !(significance_level.is_finite() && significance_level > 0.0 && significance_level < 1.0) {
        return Err(format!(
            "chart_interp: significance_level must be finite and in (0, 1), got {significance_level}"
        ));
    }
    let (circular_correlation, signed_circular_correlation, weight_sum) =
        chart_correlation(observations, "chart_interp observed")?;
    let mut null_statistics = Vec::with_capacity(null_calibration.expected_draws());
    for (draw_idx, draw) in null_calibration.observation_draws().iter().enumerate() {
        validate_null_ledger_alignment(observations, draw, draw_idx)?;
        let (null_statistic, _, _) = chart_correlation(
            draw,
            &format!("chart_interp null draw {draw_idx}"),
        )?;
        null_statistics.push(null_statistic);
    }
    let statistic = ChartInterpStatistic::OrientationQuotientedWeightedPhaseLock;
    let protocol = null_calibration.protocol();
    let null_distribution = summarize_null_distribution(
        protocol.null_kind(),
        circular_correlation,
        null_statistics,
        Tail::Larger,
    )?;
    let verdict = if null_distribution.p_value <= significance_level {
        ChartInterpVerdict::Pass
    } else {
        ChartInterpVerdict::NullCompatible
    };
    Ok(ChartInterpReport {
        statistic,
        observed: ChartInterpStatisticValue {
            circular_correlation,
            signed_circular_correlation,
            effective_weight: weight_sum,
        },
        calibration: ChartInterpNullCalibrationReport {
            statistic,
            protocol,
            seed: null_calibration.seed(),
            null_distribution,
        },
        significance_level,
        verdict,
    })
}

fn validate_null_ledger_alignment(
    observed: &[ChartInterpObservation],
    null_draw: &[ChartInterpObservation],
    draw_idx: usize,
) -> Result<(), String> {
    if null_draw.len() != observed.len() {
        return Err(format!(
            "chart_interp null draw {draw_idx} has {} rows but observed ledger has {}; the refit/readout protocol requires the same labeled rows",
            null_draw.len(),
            observed.len()
        ));
    }
    for (row, (observed_row, null_row)) in observed.iter().zip(null_draw).enumerate() {
        if wrap_turns(observed_row.label_turns) != wrap_turns(null_row.label_turns) {
            return Err(format!(
                "chart_interp null draw {draw_idx} changes label_turns at row {row}; matched-spectrum refits must score the identical labeled ledger"
            ));
        }
    }
    Ok(())
}

fn chart_correlation(
    observations: &[ChartInterpObservation],
    context: &str,
) -> Result<(f64, f64, f64), String> {
    let weight_sum = validate_weights(observations.iter().map(|o| o.weight), context)?;
    for (row, observation) in observations.iter().enumerate() {
        if !(observation.recovered_turns.is_finite() && observation.label_turns.is_finite()) {
            return Err(format!(
                "{context}: recovered_turns and label_turns must be finite at row {row}"
            ));
        }
    }
    let same = weighted_phase_lock(
        observations.iter().map(|o| {
            (
                wrap_turns(o.label_turns) - wrap_turns(o.recovered_turns),
                o.weight,
            )
        }),
        weight_sum,
    );
    let reversed = weighted_phase_lock(
        observations.iter().map(|o| {
            (
                wrap_turns(o.label_turns) + wrap_turns(o.recovered_turns),
                o.weight,
            )
        }),
        weight_sum,
    );
    let signed = if same >= reversed { same } else { -reversed };
    Ok((
        same.max(reversed).min(1.0),
        signed.clamp(-1.0, 1.0),
        weight_sum,
    ))
}

/// Fit the dose-response calibration ledger.
pub fn dose_response_calibration(
    observations: &[DoseResponseObservation],
) -> Result<DoseResponseCalibrationReport, String> {
    let weight_sum = validate_weights(observations.iter().map(|o| o.weight), "dose_response")?;
    let mut x2 = 0.0;
    let mut xy = 0.0;
    let mut y2 = 0.0;
    let mut rate_w = 0.0;
    let mut rate_sum = 0.0;
    for obs in observations {
        if !(obs.arc_length.is_finite()
            && obs.predicted_nats.is_finite()
            && obs.measured_nats.is_finite())
        {
            return Err(
                "dose_response: arc_length, predicted_nats, and measured_nats must be finite"
                    .into(),
            );
        }
        if obs.predicted_nats < 0.0 || obs.measured_nats < 0.0 {
            return Err(
                "dose_response: predicted_nats and measured_nats must be non-negative".into(),
            );
        }
        x2 += obs.weight * obs.predicted_nats * obs.predicted_nats;
        xy += obs.weight * obs.predicted_nats * obs.measured_nats;
        y2 += obs.weight * obs.measured_nats * obs.measured_nats;
        if obs.arc_length > 0.0 {
            let rate = obs.measured_nats / (obs.arc_length * obs.arc_length);
            rate_w += obs.weight;
            rate_sum += obs.weight * rate;
        }
    }
    if x2 <= 0.0 || y2 <= 0.0 {
        return Err("dose_response: non-zero predicted and measured nats are required".into());
    }
    if rate_w <= 0.0 {
        return Err("dose_response: at least one positive arc_length is required".into());
    }
    let slope = xy / x2;
    let sse = observations.iter().fold(0.0, |acc, obs| {
        let residual = obs.measured_nats - slope * obs.predicted_nats;
        acc + obs.weight * residual * residual
    });
    let mean_rate = rate_sum / rate_w;
    let rate_var = observations.iter().fold(0.0, |acc, obs| {
        if obs.arc_length > 0.0 {
            let residual = obs.measured_nats / (obs.arc_length * obs.arc_length) - mean_rate;
            acc + obs.weight * residual * residual
        } else {
            acc
        }
    }) / rate_w;
    let cv = if mean_rate > 0.0 {
        rate_var.sqrt() / mean_rate
    } else {
        0.0
    };
    Ok(DoseResponseCalibrationReport {
        slope_through_origin: slope,
        r2_through_origin: (1.0 - sse / y2).clamp(0.0, 1.0),
        mean_measured_nats_per_arc_squared: mean_rate,
        cv_measured_nats_per_arc_squared: cv,
        effective_weight: weight_sum,
    })
}

/// Expose per-token coordinate posterior uncertainty from a row-Hessian block.
pub fn coordinate_posterior_from_precision(
    mean: &[f64],
    precision_row_major: &[f64],
) -> Result<CoordinatePosterior, String> {
    let d = mean.len();
    if d == 0 {
        return Err("coordinate_posterior: coordinate dimension must be positive".into());
    }
    if precision_row_major.len() != d * d {
        return Err(format!(
            "coordinate_posterior: precision block has length {} but mean dimension {d} requires {} entries",
            precision_row_major.len(),
            d * d
        ));
    }
    for (i, &m) in mean.iter().enumerate() {
        if !m.is_finite() {
            return Err(format!("coordinate_posterior: mean[{i}] is not finite"));
        }
    }
    let max_abs = precision_row_major
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    if !max_abs.is_finite() {
        return Err("coordinate_posterior: precision block contains a non-finite entry".into());
    }
    let symmetry_tol = f64::EPSILON * d.max(1) as f64 * max_abs.max(1.0);
    for row in 0..d {
        for col in 0..row {
            let lower = precision_row_major[row * d + col];
            let upper = precision_row_major[col * d + row];
            if (lower - upper).abs() > symmetry_tol {
                return Err(format!(
                    "coordinate_posterior: precision block is not symmetric at ({row}, {col}): \
                     {lower} versus {upper} (tolerance {symmetry_tol:e})"
                ));
            }
        }
    }
    let chol = cholesky_lower(precision_row_major, d)?;
    let mut diag = vec![0.0; d];
    for basis in 0..d {
        let mut e = vec![0.0; d];
        e[basis] = 1.0;
        let col = solve_cholesky(&chol, &e, d);
        diag[basis] = col[basis];
    }
    let trace: f64 = diag.iter().sum();
    if !(trace.is_finite() && trace > 0.0) {
        return Err(
            "coordinate_posterior: inverse precision trace must be finite and positive".into(),
        );
    }
    Ok(CoordinatePosterior {
        mean: mean.to_vec(),
        covariance_diag: diag,
        covariance_trace: trace,
        precision_weight: 1.0 / trace,
    })
}

fn validate_weights<I>(weights: I, context: &str) -> Result<f64, String>
where
    I: IntoIterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    for weight in weights {
        count += 1;
        if !(weight.is_finite() && weight >= 0.0) {
            return Err(format!(
                "{context}: weights must be finite and non-negative"
            ));
        }
        sum += weight;
    }
    if count == 0 || sum <= 0.0 {
        return Err(format!(
            "{context}: at least one positive-weight observation is required"
        ));
    }
    Ok(sum)
}

fn wrap_turns(x: f64) -> f64 {
    x.rem_euclid(1.0)
}

fn weighted_phase_lock<I>(values: I, weight_sum: f64) -> f64
where
    I: IntoIterator<Item = (f64, f64)>,
{
    let mut c = 0.0;
    let mut s = 0.0;
    for (turns, weight) in values {
        let angle = std::f64::consts::TAU * turns;
        c += weight * angle.cos();
        s += weight * angle.sin();
    }
    (c * c + s * s).sqrt() / weight_sum
}

fn cholesky_lower(a: &[f64], d: usize) -> Result<Vec<f64>, String> {
    let mut l = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut sum = a[i * d + j];
            for k in 0..j {
                sum -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if !(sum.is_finite() && sum > 0.0) {
                    return Err(
                        "coordinate_posterior: precision block must be symmetric positive definite"
                            .into(),
                    );
                }
                l[i * d + j] = sum.sqrt();
            } else {
                l[i * d + j] = sum / l[j * d + j];
            }
        }
    }
    Ok(l)
}

fn solve_cholesky(l: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    let mut y = vec![0.0; d];
    for i in 0..d {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * d + k] * y[k];
        }
        y[i] = sum / l[i * d + i];
    }
    let mut x = vec![0.0; d];
    for i in (0..d).rev() {
        let mut sum = y[i];
        for k in i + 1..d {
            sum -= l[k * d + i] * x[k];
        }
        x[i] = sum / l[i * d + i];
    }
    x
}

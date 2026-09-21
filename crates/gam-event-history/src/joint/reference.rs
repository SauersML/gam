//! Weighted reference evolution with genetic states and disease jumps.
//!
//! `M_d(t,c) = E_ref[R_d(x(t-),c) | Y_d(t)=1, c]` is taken under the complete
//! reference evolution, never a fresh stationary law at each age. Every
//! population starts from the declared entry law at the origin and keeps
//! evolving: other diagnoses fire and apply their learned state jumps,
//! mortality and the population's focal once-only mark enter survival weights,
//! and missing genetic scores are drawn from their law conditional on the
//! observed scores. OU half steps surround a competing-event step, so this is a
//! finite-time approximation. `resolution.rs` estimates its error from
//! independent populations and refinements; those estimates are not
//! deterministic bounds.
//!
//! The step kernel (`half_step`, `StepRates`, `event_step`) is shared with any
//! population started from another law, such as a history-conditioned state.
use super::*;
use crate::scalar::sqrt;
use gam_linalg::triangular::{CholeskyGuard, cholesky_factor_in_place, cholesky_solve_vector};
use ndarray::Array1;
use rand::{Rng, RngExt};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

#[path = "reference_sensitivity.rs"]
mod sensitivity;

/// The declared reference: its first time is the origin and its last time the
/// supported horizon. The population starts alive and free of every once-only
/// mark at the origin; a late origin is a declared entry law, not conditioning
/// on an unobserved disease-free interval before it. A forecast past the
/// horizon needs a longer declared profile evolved explicitly; nothing here
/// extrapolates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointReferenceProfile {
    pub times: Vec<f64>,
    pub baseline_design: Array2<f64>,
    pub drive_design: Array2<f64>,
    pub entry_design: Vec<f64>,
    pub genetics: Vec<Option<f64>>,
}

impl JointReferenceProfile {
    fn history(&self, marks: usize) -> JointHistory {
        let mut exposure = vec![0.0; self.times.len()];
        for n in 1..self.times.len() {
            exposure[n] = self.times[n] - self.times[n - 1];
        }
        JointHistory {
            times: self.times.clone(),
            exposure,
            events: vec![None; self.times.len()],
            initially_at_risk: vec![true; marks],
            baseline_design: self.baseline_design.clone(),
            drive_design: self.drive_design.clone(),
            entry_design: self.entry_design.clone(),
            genetics: self.genetics.clone(),
            measurements: vec![],
        }
    }

    /// The declared history's own checks. Refinement never stores computed
    /// nodes; `ReferenceGrid::new` refuses intervals that cannot be halved.
    pub(super) fn validate(&self, model: &JointLikelihood) -> Result<(), EventHistoryError> {
        model.validate_history(&self.history(model.spec.marks.len()))
    }
}

/// A declared piecewise-linear baseline row at a dyadic weight, `(1 - w) a + w
/// b` between the declared rows `a` and `b`. It is formed over `S` where used
/// and never stored as a computed row.
pub(in crate::joint) struct DyadicRow<'a> {
    pub(in crate::joint) left: ndarray::ArrayView1<'a, f64>,
    pub(in crate::joint) right: ndarray::ArrayView1<'a, f64>,
    pub(in crate::joint) weight: f64,
}

impl DyadicRow<'_> {
    pub(in crate::joint) fn columns(&self) -> usize {
        self.left.len()
    }

    /// `value` times the row's feature `b`: the declared features and the
    /// exact dyadic weights enter through `scale`, which charges each
    /// product's rounding.
    pub(in crate::joint) fn feature<S: JetField>(&self, value: &S, b: usize) -> S {
        value
            .scale(self.left[b])
            .scale(1.0 - self.weight)
            .add(&value.scale(self.right[b]).scale(self.weight))
    }
}

/// The declared profile halved `refinements` times. Refined nodes are never
/// stored as computed values: node `n` lies in declared interval `n >> r` at
/// the dyadic weight `(n mod 2^r) / 2^r`, and its time, interval length and
/// baseline row are formed over `S` from the declared endpoints where used.
/// Represented times are kept only to locate requests.
#[derive(Clone, Debug)]
pub(in crate::joint) struct ReferenceGrid {
    profile: JointReferenceProfile,
    refinements: u32,
}

impl ReferenceGrid {
    /// Refuses a grid whose stored rows, one halving finer than its nodes, are
    /// not strictly increasing as represented values.
    pub(in crate::joint) fn new(
        profile: JointReferenceProfile,
        refinements: u32,
    ) -> Result<Self, EventHistoryError> {
        let rows = profile
            .times
            .len()
            .checked_sub(1)
            .and_then(|n| refinements.checked_add(1).and_then(|r| n.checked_shl(r)))
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid("joint reference refinement dimension overflow"))?;
        let grid = Self {
            profile,
            refinements,
        };
        for m in 1..=rows {
            if grid.row_time_value(m) <= grid.row_time_value(m - 1) {
                return Err(numerical(
                    "joint reference interval has no representable interior midpoint",
                ));
            }
        }
        Ok(grid)
    }

    pub(in crate::joint) fn profile(&self) -> &JointReferenceProfile {
        &self.profile
    }

    pub(in crate::joint) fn refinements(&self) -> u32 {
        self.refinements
    }

    pub(in crate::joint) fn refined(&self) -> Result<Self, EventHistoryError> {
        let refinements = self
            .refinements
            .checked_add(1)
            .ok_or_else(|| invalid("joint reference refinement dimension overflow"))?;
        Self::new(self.profile.clone(), refinements)
    }

    /// Grid nodes: every declared interval split into `2^r` parts.
    pub(in crate::joint) fn nodes(&self) -> usize {
        ((self.profile.times.len() - 1) << self.refinements) + 1
    }

    /// Declared interval and dyadic weight of node `n` after `halvings`.
    fn place(&self, n: usize, halvings: u32) -> (usize, f64) {
        let intervals = self.profile.times.len() - 1;
        let interval = (n >> halvings).min(intervals - 1);
        let parts = (1usize << halvings) as f64;
        (interval, (n - (interval << halvings)) as f64 / parts)
    }

    /// The represented time of stored row `m`, one halving finer than the
    /// nodes, used only to locate requests.
    pub(in crate::joint) fn row_time_value(&self, m: usize) -> f64 {
        let (i, w) = self.place(m, self.refinements + 1);
        let (a, b) = (self.profile.times[i], self.profile.times[i + 1]);
        a + w * (b - a)
    }

    /// Stored row `m`'s time over `S`, `(1 - w) t_i + w t_{i+1}`.
    pub(in crate::joint) fn row_time<S: JetField>(&self, like: &S, m: usize) -> S {
        let (i, w) = self.place(m, self.refinements + 1);
        like.constant_like(self.profile.times[i])
            .scale(1.0 - w)
            .add(&like.constant_like(self.profile.times[i + 1]).scale(w))
    }

    /// Interval `n`'s length over `S`: its declared interval's length halved
    /// `r` times, an exact scaling.
    pub(in crate::joint) fn dt<S: JetField>(&self, like: &S, n: usize) -> S {
        let (i, _) = self.place(n - 1, self.refinements);
        like.constant_like(self.profile.times[i + 1])
            .sub(&like.constant_like(self.profile.times[i]))
            .scale(0.5_f64.powi(self.refinements as i32))
    }

    /// Interval `n`'s baseline row at its midpoint.
    pub(in crate::joint) fn midpoint_row(&self, n: usize) -> DyadicRow<'_> {
        let (i, weight) = self.place(2 * n - 1, self.refinements + 1);
        DyadicRow {
            left: self.profile.baseline_design.row(i),
            right: self.profile.baseline_design.row(i + 1),
            weight,
        }
    }

    /// Interval `n`'s piecewise-constant drive row.
    pub(in crate::joint) fn drive_row(&self, n: usize) -> ndarray::ArrayView1<'_, f64> {
        let (i, _) = self.place(n - 1, self.refinements);
        self.profile.drive_design.row(i)
    }
}

/// Within-population summaries. Neither is a time-error certificate nor an
/// independent-replicate uncertainty estimate for the interacting population.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceDiagnostics {
    pub minimum_risk_effective_samples: f64,
    /// Sum over event steps of the largest risk set's weighted probability
    /// that a particle needs two or more non-killing events in one step. A step
    /// simulates at most one, so this is probability mass the step cannot
    /// represent. It does not bound the error of a later jump-dependent rate,
    /// and it is zero without signatures, where events cannot move the state.
    pub omitted_event_mass: f64,
}

/// One jointly evaluated coefficient/reference state. Stored times include
/// interval midpoints; interpolation never extrapolates beyond the declared
/// origin and horizon.
pub struct JointReferenceEvolution<S> {
    pub(super) theta: Vec<S>,
    /// Represented row times, used only to locate requests.
    pub(super) times: Vec<f64>,
    /// The same row times formed over `S` from the declared endpoints, the
    /// operands of every interpolation weight.
    pub(super) row_times: Vec<S>,
    pub(super) log_moments: Vec<S>,
    pub(super) log_risk_mass: Vec<S>,
    pub(super) marks: usize,
    pub(super) diagnostics: ReferenceDiagnostics,
}

impl<S: JetField> JointReferenceEvolution<S> {
    pub fn coefficients(&self) -> &[S] {
        &self.theta
    }
    pub fn times(&self) -> &[f64] {
        &self.times
    }
    pub fn log_moments(&self) -> &[S] {
        &self.log_moments
    }
    pub fn log_risk_mass(&self) -> &[S] {
        &self.log_risk_mass
    }
    pub fn diagnostics(&self) -> &ReferenceDiagnostics {
        &self.diagnostics
    }
    /// The declared origin.
    pub fn origin(&self) -> f64 {
        self.times[0]
    }
    /// The declared horizon; later requests are refused.
    pub fn horizon(&self) -> f64 {
        self.times[self.times.len() - 1]
    }

    /// Log moments at declared request times.
    pub fn at(&self, times: &[f64]) -> Result<Vec<S>, EventHistoryError> {
        self.interpolate(&self.log_moments, times, None)
    }

    /// Log moments at computed request times, located by their represented
    /// values and weighted by their values over `S`.
    pub(super) fn at_over(&self, times: &[f64], times_s: &[S]) -> Result<Vec<S>, EventHistoryError> {
        self.interpolate(&self.log_moments, times, Some(times_s))
    }

    pub(super) fn risk_mass_over(
        &self,
        times: &[f64],
        times_s: &[S],
    ) -> Result<Vec<S>, EventHistoryError> {
        self.interpolate(&self.log_risk_mass, times, Some(times_s))
    }

    /// Adjoint of `at`: adds each requested time's log-moment adjoint to the
    /// stored rows it interpolates, so a stratum sums every subject's adjoint
    /// on the reference grid before one pullback.
    pub fn scatter(
        &self,
        times: &[f64],
        adjoint: &[S],
        grid_adjoint: &mut [S],
    ) -> Result<(), EventHistoryError> {
        if adjoint.len() != times.len() * self.marks
            || grid_adjoint.len() != self.log_moments.len()
        {
            return Err(invalid(
                "joint reference adjoint dimensions do not match the requested times or grid",
            ));
        }
        for (i, &time) in times.iter().enumerate() {
            let (right, interior) = self.locate(time)?;
            let weights =
                interior.then(|| self.weights(right, &adjoint[i * self.marks].constant_like(time)));
            for d in 0..self.marks {
                let value = &adjoint[i * self.marks + d];
                match &weights {
                    None => {
                        let slot = &mut grid_adjoint[right * self.marks + d];
                        *slot = slot.add(value);
                    }
                    Some((to_left, to_right)) => {
                        let slot = &mut grid_adjoint[(right - 1) * self.marks + d];
                        *slot = slot.add(&value.mul(to_left));
                        let slot = &mut grid_adjoint[right * self.marks + d];
                        *slot = slot.add(&value.mul(to_right));
                    }
                }
            }
        }
        Ok(())
    }

    /// The stored row at or right of `time`, and whether `time` lies strictly
    /// between two stored rows.
    fn locate(&self, time: f64) -> Result<(usize, bool), EventHistoryError> {
        if !time.is_finite() || time < self.origin() || time > self.horizon() {
            return Err(invalid(format!(
                "joint reference request at {time} is outside its declared origin {} and horizon {}; declare a longer reference profile instead of extrapolating",
                self.origin(),
                self.horizon()
            )));
        }
        let right = self.times.partition_point(|&t| t < time);
        Ok((right, right > 0 && self.times[right] != time))
    }

    /// Linear interpolation weights towards the stored rows left and right of
    /// an interior request at `at`, `(t_r - t) / (t_r - t_l)` and `(t - t_l) /
    /// (t_r - t_l)`, formed over `S` from the rows' declared-endpoint times so
    /// their rounding is charged and neither is a cancelling `1 - fraction`.
    fn weights(&self, right: usize, at: &S) -> (S, S) {
        let left = &self.row_times[right - 1];
        let upper = &self.row_times[right];
        let width = upper.sub(left);
        (div(&upper.sub(at), &width), div(&at.sub(left), &width))
    }

    /// Values at `times`: declared requests enter over `S` as their exact
    /// values, computed requests as `over`.
    fn interpolate(
        &self,
        values: &[S],
        times: &[f64],
        over: Option<&[S]>,
    ) -> Result<Vec<S>, EventHistoryError> {
        let mut out = Vec::with_capacity(times.len() * self.marks);
        for (i, &time) in times.iter().enumerate() {
            let (right, interior) = self.locate(time)?;
            if interior {
                let at = over.map_or_else(|| values[0].constant_like(time), |t| t[i].clone());
                let (to_left, to_right) = self.weights(right, &at);
                for d in 0..self.marks {
                    out.push(
                        values[(right - 1) * self.marks + d]
                            .mul(&to_left)
                            .add(&values[right * self.marks + d].mul(&to_right)),
                    );
                }
            } else {
                out.extend_from_slice(&values[right * self.marks..(right + 1) * self.marks]);
            }
        }
        Ok(out)
    }
}

/// Fixed normal innovations, missing genetic draws and proposed event histories
/// of one independent reference population. The event proposal was generated
/// at an anchor, but its probabilities are held as sampling data only.
/// Re-evaluation differentiates target/proposal weights as well as the OU
/// states and the evolving risk-set moments.
pub(super) struct JointReferenceBank {
    grid: ReferenceGrid,
    genes: Vec<Vec<f64>>,
    /// Block-major: the entry draw, then two half steps per interval, each
    /// `particles x signatures`.
    normals: Vec<f64>,
    choices: Vec<Option<usize>>,
    log_proposal: Vec<f64>,
    groups: Vec<Option<usize>>,
    mark_groups: Vec<usize>,
    particles: usize,
}

/// A population the step kernel advances: per particle its state, log weight
/// and risk set, and per killing set its log survival mass.
#[derive(Clone)]
pub(in crate::joint) struct Population<S> {
    pub(in crate::joint) states: Vec<Vec<S>>,
    pub(in crate::joint) log_weight: Vec<S>,
    pub(in crate::joint) risk: Vec<Vec<bool>>,
    pub(in crate::joint) log_mass: Vec<S>,
}

/// OU coefficients over half of one interval: `phi`, `1 - phi` and
/// `sqrt(1 - phi^2)` per signature. A static axis, whose rate underflows to
/// zero, has `phi = 1`, weight 0 and spread 0: its step is exact, and its
/// spread carries no derivative.
pub(in crate::joint) struct HalfStep<S> {
    phi: Vec<S>,
    weight: Vec<S>,
    spread: Vec<S>,
}

impl<S: JetField> HalfStep<S> {
    /// `dt` is the interval length formed over `S` from exact endpoint data.
    pub(in crate::joint) fn new(rates: &[S], dt: &S) -> Result<Self, EventHistoryError> {
        let half = dt.scale(0.5);
        let decay: Vec<S> = rates.iter().map(|r| r.mul(&half).neg()).collect();
        let variance: Vec<S> = decay
            .iter()
            .map(|r| emission::expm1(&r.scale(2.0)).neg())
            .collect();
        if variance
            .iter()
            .any(|v| v.value() < 0.0 || !v.value().is_finite())
        {
            return Err(numerical(
                "joint reference OU half-step variance is unresolved",
            ));
        }
        Ok(Self {
            phi: decay.iter().map(exp).collect(),
            weight: decay.iter().map(|r| emission::expm1(r).neg()).collect(),
            spread: variance
                .iter()
                .map(|v| {
                    if v.value() > 0.0 {
                        sqrt(v)
                    } else {
                        v.constant_like(0.0)
                    }
                })
                .collect(),
        })
    }
}

/// A state mean that is linear in the genes over one row of features: per
/// signature, the feature-weighted intercept and genetic coefficients, formed
/// once per row instead of once per particle.
pub(in crate::joint) struct MeanTable<S> {
    values: Vec<S>,
    width: usize,
}

impl<S: JetField> MeanTable<S> {
    pub(in crate::joint) fn new(
        theta: &[S],
        start: usize,
        columns: &[f64],
        signatures: usize,
        genes: usize,
    ) -> Self {
        let width = genes + 1;
        let mut values = vec![theta[0].constant_like(0.0); signatures * width];
        for axis in 0..signatures {
            for (j, &feature) in columns.iter().enumerate() {
                let base = start + (axis * columns.len() + j) * width;
                for c in 0..width {
                    values[axis * width + c] =
                        values[axis * width + c].add(&theta[base + c].scale(feature));
                }
            }
        }
        Self { values, width }
    }

    fn mean(&self, axis: usize, genes: &[S]) -> S {
        let row = &self.values[axis * self.width..(axis + 1) * self.width];
        genes
            .iter()
            .zip(&row[1..])
            .fold(row[0].clone(), |acc, (gene, c)| acc.add(&c.mul(gene)))
    }
}

/// Exact OU half step of every particle over one interval's held drive;
/// `normals` holds one standard normal per particle and signature. Particles
/// step independently, so the result is the same at any thread count.
pub(in crate::joint) fn half_step<S: JetField + Send + Sync>(
    step: &HalfStep<S>,
    drive: &MeanTable<S>,
    genes: &[Vec<S>],
    normals: &[f64],
    states: &mut [Vec<S>],
) {
    let k = step.phi.len();
    states.par_iter_mut().enumerate().for_each(|(p, state)| {
        for axis in 0..k {
            let mean = drive.mean(axis, &genes[p]);
            state[axis] = step.phi[axis]
                .mul(&state[axis])
                .add(&step.weight[axis].mul(&mean))
                .add(&step.spread[axis].scale(normals[p * k + axis]));
        }
    });
}

/// The killing marks of every particle: particles are contiguous by set,
/// `per_set` each, and a set's mask marks the hazards integrated as survival.
pub(in crate::joint) struct KillingSets<'a> {
    pub(in crate::joint) masks: &'a [Vec<bool>],
    pub(in crate::joint) per_set: usize,
}

/// How a step chooses each particle's non-killing event: proposed from the
/// population's own rates with one uniform per particle, recording the choice
/// and its log proposal probability, or replayed from a record.
pub(in crate::joint) enum EventChoice<'a> {
    Generate {
        uniforms: &'a [f64],
        choices: &'a mut [Option<usize>],
        log_proposal: &'a mut [f64],
    },
    Replay {
        choices: &'a [Option<usize>],
        log_proposal: &'a [f64],
    },
}

/// What one event step integrated per particle.
pub(in crate::joint) struct EventStep<S> {
    /// `log(lambda_e(t_mid) dt)`, the hazard of each killing mark in the
    /// particle's risk set integrated over the step at the frozen midpoint
    /// rate, in mark order: particle `p` owns `log_hazard[offsets[p]..offsets[p + 1]]`.
    pub(in crate::joint) log_hazard: Vec<S>,
    pub(in crate::joint) offsets: Vec<usize>,
    /// The integrated non-killing hazard.
    pub(in crate::joint) retained_hazard: Vec<f64>,
}

/// Linear decoder weights `pi_dk` of every mark over [intercept, axes], formed
/// once per coefficient value.
pub(in crate::joint) struct DecoderWeights<S> {
    values: Vec<S>,
    width: usize,
}

impl<S: JetField> DecoderWeights<S> {
    pub(in crate::joint) fn new(
        decoder: &super::decoder::PreparedDecoder<S>,
        marks: usize,
        signatures: usize,
    ) -> Self {
        let values = (0..marks)
            .flat_map(|d| decoder.weights(d).iter().map(exp))
            .collect();
        Self {
            values,
            width: signatures + 1,
        }
    }

    pub(in crate::joint) fn row(&self, mark: usize) -> &[S] {
        &self.values[mark * self.width..(mark + 1) * self.width]
    }
}

/// `sums[0] + sum_k sums[k + 1] softplus(x_k)`.
pub(in crate::joint) fn combine<S: JetField>(sums: &[S], softplus: &[S]) -> S {
    softplus
        .iter()
        .zip(&sums[1..])
        .fold(sums[0].clone(), |acc, (s, a)| acc.add(&a.mul(s)))
}

/// One step's rates per killing set. With `w_d = exp(b_d - m_d)`, a mark's
/// rate at state `x` is `w_d R_d(x)` with `R_d(x) = pi_d0 + sum_k pi_dk
/// softplus(x_k)`, so a set's non-killing marks sum to `A_0 + sum_k A_k
/// softplus(x_k)` with `A_k = sum_d w_d pi_dk`, formed once per step rather
/// than once per particle. Each set's weights are divided by its largest
/// `w_d`, so no sum overflows and every term is positive.
pub(in crate::joint) struct StepRates<S> {
    /// `b_d - m_d` per mark.
    log_scale: Vec<S>,
    /// Per set: the removed scale `shift`, `w_d / exp(shift)` per mark (zero
    /// for its killing marks), the sums `A_k / exp(shift)` over [intercept,
    /// axes], and its number of non-killing marks.
    shifts: Vec<f64>,
    scaled: Vec<S>,
    sums: Vec<S>,
    retained: Vec<usize>,
    marks: usize,
    width: usize,
}

impl<S: JetField> StepRates<S> {
    pub(in crate::joint) fn new(
        model: &JointLikelihood,
        theta: &[S],
        weights: &DecoderWeights<S>,
        baseline: &DyadicRow<'_>,
        log_normaliser: &[S],
        killing: &KillingSets<'_>,
    ) -> Result<Self, EventHistoryError> {
        let marks = model.spec.marks.len();
        let width = weights.width;
        let zero = theta[0].constant_like(0.0);
        let mut log_scale = Vec::with_capacity(marks);
        for d in 0..marks {
            let mut log_baseline = zero.clone();
            for b in 0..baseline.columns() {
                log_baseline = log_baseline.add(&baseline.feature(
                    &theta[model.layout.baseline.start + d * model.spec.baseline_columns + b],
                    b,
                ));
            }
            let scale = log_baseline.sub(&log_normaliser[d]);
            if !scale.value().is_finite() {
                return Err(numerical("joint reference log rate is non-finite"));
            }
            log_scale.push(scale);
        }
        let sets = killing.masks.len();
        let mut shifts = Vec::with_capacity(sets);
        let mut scaled = Vec::with_capacity(sets * marks);
        let mut sums = Vec::with_capacity(sets * width);
        let mut retained = Vec::with_capacity(sets);
        for mask in killing.masks {
            let shift = (0..marks)
                .filter(|&d| !mask[d])
                .map(|d| log_scale[d].value())
                .fold(f64::NEG_INFINITY, f64::max);
            let shift = if shift.is_finite() { shift } else { 0.0 };
            let start = sums.len();
            sums.resize(start + width, zero.clone());
            for d in 0..marks {
                if mask[d] {
                    scaled.push(zero.clone());
                    continue;
                }
                let w = exp(&add_real(&log_scale[d], -shift));
                for (k, pi) in weights.row(d).iter().enumerate() {
                    sums[start + k] = sums[start + k].add(&w.mul(pi));
                }
                scaled.push(w);
            }
            shifts.push(shift);
            retained.push(mask.iter().filter(|m| !**m).count());
        }
        Ok(Self {
            log_scale,
            shifts,
            scaled,
            sums,
            retained,
            marks,
            width,
        })
    }

    pub(in crate::joint) fn log_scale(&self, mark: usize) -> &S {
        &self.log_scale[mark]
    }

    pub(in crate::joint) fn shift(&self, set: usize) -> f64 {
        self.shifts[set]
    }

    pub(in crate::joint) fn scaled(&self, set: usize, mark: usize) -> &S {
        &self.scaled[set * self.marks + mark]
    }

    pub(in crate::joint) fn sums(&self, set: usize) -> &[S] {
        &self.sums[set * self.width..(set + 1) * self.width]
    }

    /// Writes to `axis` a particle's retained sums `K_k`: `A_k / exp(shift)`
    /// over its set's non-killing marks still at risk, and returns how many
    /// such marks remain. Fired once-only marks are subtracted from the set's
    /// sums while their scaled rate at `x`, `F`, is at most half the set total
    /// `T`: the difference is then at least `T / 2`, so its rounding is at most
    /// `(T + F) / (T - F) <= 3` times that of the two positive sums. Beyond that
    /// the retained marks are summed directly.
    pub(in crate::joint) fn retained_sums(
        &self,
        set: usize,
        mask: &[bool],
        risk: &[bool],
        weights: &DecoderWeights<S>,
        softplus: &[S],
        axis: &mut Vec<S>,
    ) -> usize {
        let full = self.sums(set);
        axis.clear();
        axis.extend(full.iter().cloned());
        let mut fired = 0;
        let mut removed: Option<Vec<S>> = None;
        for d in (0..self.marks).filter(|&d| !mask[d] && !risk[d]) {
            let sum = removed.get_or_insert_with(|| vec![full[0].constant_like(0.0); self.width]);
            for (k, pi) in weights.row(d).iter().enumerate() {
                sum[k] = sum[k].add(&self.scaled(set, d).mul(pi));
            }
            fired += 1;
        }
        let Some(removed) = removed else {
            return self.retained[set];
        };
        if combine(&removed, softplus).value() <= 0.5 * combine(full, softplus).value() {
            for (slot, r) in axis.iter_mut().zip(&removed) {
                *slot = slot.sub(r);
            }
        } else {
            for slot in axis.iter_mut() {
                *slot = full[0].constant_like(0.0);
            }
            for d in (0..self.marks).filter(|&d| !mask[d] && risk[d]) {
                for (k, pi) in weights.row(d).iter().enumerate() {
                    axis[k] = axis[k].add(&self.scaled(set, d).mul(pi));
                }
            }
        }
        self.retained[set] - fired
    }
}

/// One particle's event choice: proposed from one uniform with its record
/// written back, or replayed from its record.
enum Slot<'a> {
    Generate {
        uniform: f64,
        choice: &'a mut Option<usize>,
        log_proposal: &'a mut f64,
    },
    Replay {
        choice: Option<usize>,
        log_proposal: f64,
    },
}

/// One competing-event step at the frozen midpoint rates. Killing hazards are
/// returned individually for survival integration. The other marks enter
/// through the set's rate sums: a particle's retained total is `A_0 + sum_k
/// A_k softplus(x_k)` less the once-only marks it has fired, and a mark's own
/// rate is formed only when an event proposes or replays it. Each particle
/// proposes at most one event, adds `log target - log proposal` to its
/// absolute log weight, removes a fired once-only mark from its risk set, and
/// applies the learned jump. Weights are never renormalized here. Particles are
/// independent within a step, so they are evaluated in parallel and gathered
/// in particle order.
pub(in crate::joint) fn event_step<S: JetField + Send + Sync>(
    model: &JointLikelihood,
    theta: &[S],
    decoder: &super::decoder::PreparedDecoder<S>,
    weights: &DecoderWeights<S>,
    rates: &StepRates<S>,
    dt: &S,
    killing: &KillingSets<'_>,
    population: &mut Population<S>,
    choice: EventChoice<'_>,
) -> Result<EventStep<S>, EventHistoryError> {
    let marks = model.spec.marks.len();
    let k = model.spec.signatures;
    let count = population.states.len();
    let zero = theta[0].constant_like(0.0);
    let log_dt = ln(dt);
    let slots: Vec<Slot<'_>> = match choice {
        EventChoice::Generate {
            uniforms,
            choices,
            log_proposal,
        } => uniforms
            .iter()
            .zip(choices.iter_mut().zip(log_proposal.iter_mut()))
            .map(|(&uniform, (choice, log_proposal))| Slot::Generate {
                uniform,
                choice,
                log_proposal,
            })
            .collect(),
        EventChoice::Replay {
            choices,
            log_proposal,
        } => choices
            .iter()
            .zip(log_proposal)
            .map(|(&choice, &log_proposal)| Slot::Replay {
                choice,
                log_proposal,
            })
            .collect(),
    };
    if slots.len() != count || population.log_weight.len() != count || population.risk.len() != count {
        return Err(invalid(
            "joint reference event step needs one weight, risk set and event choice per particle",
        ));
    }
    let particles: Vec<Result<(Vec<S>, f64), EventHistoryError>> = population
        .states
        .par_iter_mut()
        .zip(population.log_weight.par_iter_mut())
        .zip(population.risk.par_iter_mut())
        .zip(slots.into_par_iter())
        .enumerate()
        .map_init(
            || (Vec::with_capacity(k), Vec::with_capacity(k + 1)),
            |(softplus, axis): &mut (Vec<S>, Vec<S>), (p, (((state, weight), risk), slot))| {
                let set = p / killing.per_set;
                let mask = &killing.masks[set];
                softplus.clear();
                softplus.extend(state.iter().map(emission::softplus));
                let mut log_hazard = Vec::new();
                for d in (0..marks).filter(|&d| mask[d] && risk[d]) {
                    let rate = rates.log_scale(d).add(&decoder.activity(d, state));
                    if !rate.value().is_finite() {
                        return Err(numerical("joint reference log rate is non-finite"));
                    }
                    log_hazard.push(rate.add(&log_dt));
                }
                let retained = rates.retained_sums(set, mask, risk, weights, softplus, axis);
                let mut retained_hazard = 0.0;
                // The retained log total, the no-event log probability and the
                // log probability that some retained event occurs.
                let exposure = if retained > 0 {
                    let log_total = add_real(&ln(&combine(axis, softplus)), rates.shift(set));
                    if !log_total.value().is_finite() {
                        return Err(numerical("joint reference log rate is non-finite"));
                    }
                    let exposure = log_total.add(&log_dt);
                    retained_hazard = exposure.value().exp();
                    let event = log_event(&exposure);
                    Some((log_total, exp(&exposure).neg(), event))
                } else {
                    None
                };
                let log_rate = |d: usize| rates.log_scale(d).add(&decoder.activity(d, state));
                let (selected, proposal) = match slot {
                    Slot::Generate {
                        uniform,
                        choice,
                        log_proposal,
                    } => {
                        // No event, then each retained mark in mark order.
                        let (none, event) = exposure
                            .as_ref()
                            .map_or((0.0, f64::NEG_INFINITY), |(_, none, event)| {
                                (none.value(), event.value())
                            });
                        let total = none.exp() + event.exp();
                        let threshold = uniform * total;
                        let mut cumulative = none.exp();
                        let mut selected = (None, none);
                        if threshold >= cumulative
                            && let Some((log_total, _, event)) = &exposure
                        {
                            for d in (0..marks).filter(|&d| !mask[d] && risk[d]) {
                                let q = event.value() + log_rate(d).value() - log_total.value();
                                selected = (Some(d), q);
                                cumulative += q.exp();
                                if threshold < cumulative {
                                    break;
                                }
                            }
                        }
                        *choice = selected.0;
                        *log_proposal = selected.1 - total.ln();
                        (*choice, *log_proposal)
                    }
                    Slot::Replay {
                        choice,
                        log_proposal,
                    } => (choice, log_proposal),
                };
                let target = match (selected, &exposure) {
                    (None, None) => zero.clone(),
                    (None, Some((_, none, _))) => none.clone(),
                    (Some(d), Some((log_total, _, event))) if !mask[d] && risk[d] => {
                        event.add(&log_rate(d)).sub(log_total)
                    }
                    _ => {
                        return Err(invalid(
                            "joint reference event proposal violates its risk history",
                        ));
                    }
                };
                *weight = weight.add(&add_real(&target, -proposal));
                if let Some(d) = selected {
                    if model.spec.marks[d] == MarkKind::Once {
                        risk[d] = false;
                    }
                    for axis in 0..k {
                        state[axis] = state[axis].add(&model.jump(theta, Some(d), axis));
                    }
                }
                Ok((log_hazard, retained_hazard))
            },
        )
        .collect();
    let mut outcome = EventStep {
        log_hazard: Vec::with_capacity(count),
        offsets: Vec::with_capacity(count + 1),
        retained_hazard: Vec::with_capacity(count),
    };
    outcome.offsets.push(0);
    for particle in particles {
        let (log_hazard, retained_hazard) = particle?;
        outcome.log_hazard.extend(log_hazard);
        outcome.offsets.push(outcome.log_hazard.len());
        outcome.retained_hazard.push(retained_hazard);
    }
    Ok(outcome)
}

/// One population for the living law and one for each once-only mark; this is
/// linear in the number of risk sets, not a product of diagnosis combinations.
fn risk_groups(model: &JointLikelihood) -> (Vec<Option<usize>>, Vec<usize>) {
    let mut groups = Vec::new();
    let mut mark_groups = Vec::with_capacity(model.spec.marks.len());
    for (d, kind) in model.spec.marks.iter().enumerate() {
        let key = (*kind == MarkKind::Once).then_some(d);
        let index = match groups.iter().position(|g| *g == key) {
            Some(index) => index,
            None => {
                groups.push(key);
                groups.len() - 1
            }
        };
        mark_groups.push(index);
    }
    (groups, mark_groups)
}

/// The entry draw and two OU half steps per interval, per signature, particle.
fn noise_count(nodes: usize, signatures: usize, count: usize) -> Result<usize, EventHistoryError> {
    nodes
        .checked_mul(2)
        .and_then(|n| n.checked_sub(1))
        .and_then(|n| n.checked_mul(signatures))
        .and_then(|n| n.checked_mul(count))
        .ok_or_else(|| invalid("joint reference noise dimension overflow"))
}

/// This machine's materialization budget, derived from its observed memory
/// capacity: the one admission source for retained populations and their
/// derivative workspaces.
pub(super) fn materialization_budget() -> usize {
    gam_runtime::resource::ResourcePolicy::default_library().max_single_materialization_bytes
}

// log(1-exp(-exp(log_hazard))), stable even for subnormal hazards. Below the
// switch the series log h + log(1 - h/2 + h^2/6 - h^3/24 + h^4/120) omits
// h^5/720, which is under machine epsilon once h <= (720 eps)^(1/5).
fn log_event<S: JetField>(log_hazard: &S) -> S {
    let hazard = exp(log_hazard);
    if log_hazard.value() < (720.0_f64.ln() + f64::EPSILON.ln()) / 5.0 {
        let polynomial = add_real(&hazard.scale(1.0 / 120.0), -1.0 / 24.0);
        let polynomial = add_real(&hazard.mul(&polynomial), 1.0 / 6.0);
        let polynomial = add_real(&hazard.mul(&polynomial), -0.5);
        log_hazard.add(&ln(&add_real(&hazard.mul(&polynomial), 1.0)))
    } else {
        ln(&emission::expm1(&hazard.neg()).neg())
    }
}

/// Genomes of `count` particles: observed scores kept, missing scores drawn
/// from their law given the observed ones. With precision `Q_mm = L L^T`, the
/// conditional mean is `mu_m - Q_mm^{-1} Q_mo (g_o - mu_o)` and a draw adds
/// `L^{-T} z`, whose covariance is `Q_mm^{-1}`.
pub(in crate::joint) fn draw_genomes<R: Rng + ?Sized>(
    model: &JointLikelihood,
    genetics: &[Option<f64>],
    count: usize,
    rng: &mut R,
) -> Result<Vec<Vec<f64>>, EventHistoryError> {
    let missing: Vec<usize> = genetics
        .iter()
        .enumerate()
        .filter_map(|(i, g)| g.is_none().then_some(i))
        .collect();
    let precision = Array2::from_shape_fn((missing.len(), missing.len()), |(i, j)| {
        model.spec.genetic_precision[[missing[i], missing[j]]]
    });
    let mut information = vec![0.0; missing.len()];
    for (i, &gi) in missing.iter().enumerate() {
        for (j, &gj) in missing.iter().enumerate() {
            information[i] += precision[[i, j]] * model.spec.genetic_mean[gj];
        }
        for (gj, observed) in genetics.iter().enumerate() {
            if let Some(value) = observed {
                information[i] -=
                    model.spec.genetic_precision[[gi, gj]] * (value - model.spec.genetic_mean[gj]);
            }
        }
    }
    let (factor, mean) = if missing.is_empty() {
        (Array2::zeros((0, 0)), Vec::new())
    } else {
        let factor = cholesky_factor_in_place(precision.view(), CholeskyGuard::FiniteStrict)
            .ok_or_else(|| {
                numerical("the missing genetic scores' conditional precision is not positive definite")
            })?;
        let mean = cholesky_solve_vector(&factor, &Array1::from(information)).to_vec();
        (factor, mean)
    };
    let mut genes = Vec::with_capacity(count);
    for _ in 0..count {
        let mut draw: Vec<f64> = (0..missing.len())
            .map(|_| StandardNormal.sample(rng))
            .collect();
        for i in (0..draw.len()).rev() {
            let tail: f64 = (i + 1..draw.len()).map(|k| factor[[k, i]] * draw[k]).sum();
            draw[i] = (draw[i] - tail) / factor[[i, i]];
        }
        let mut genome: Vec<f64> = genetics.iter().map(|v| v.unwrap_or(0.0)).collect();
        for (i, &g) in missing.iter().enumerate() {
            genome[g] = mean[i] + draw[i];
        }
        if genome.iter().any(|g| !g.is_finite()) {
            return Err(numerical("joint reference genetic draw is unresolved"));
        }
        genes.push(genome);
    }
    Ok(genes)
}

/// Where a bank's evolution takes its event choices.
enum Proposal<'a> {
    Generate {
        uniforms: &'a [f64],
        choices: &'a mut [Option<usize>],
        log_proposal: &'a mut [f64],
    },
    Replay,
}

impl JointReferenceBank {
    /// Retained storage of one population, checked before any allocation;
    /// transient scalar and jet work is additional.
    pub(super) fn storage_bytes(
        model: &JointLikelihood,
        grid: &ReferenceGrid,
        particles: usize,
    ) -> Result<usize, EventHistoryError> {
        let profile = grid.profile();
        let (groups, mark_groups) = risk_groups(model);
        let count = particles
            .checked_mul(groups.len())
            .ok_or_else(|| invalid("joint reference population dimension overflow"))?;
        let normals = noise_count(grid.nodes(), model.spec.signatures, count)?;
        let steps = grid
            .nodes()
            .checked_sub(1)
            .and_then(|n| n.checked_mul(count))
            .ok_or_else(|| invalid("joint reference history dimension overflow"))?;
        normals
            .checked_mul(std::mem::size_of::<f64>())
            .and_then(|b| {
                steps
                    .checked_mul(std::mem::size_of::<f64>() + std::mem::size_of::<Option<usize>>())
                    .and_then(|s| b.checked_add(s))
            })
            .and_then(|b| {
                count
                    .checked_mul(profile.genetics.len())
                    .and_then(|g| g.checked_mul(std::mem::size_of::<f64>()))
                    .and_then(|g| b.checked_add(g))
            })
            .and_then(|b| {
                count
                    .checked_mul(std::mem::size_of::<Vec<f64>>())
                    .and_then(|v| b.checked_add(v))
            })
            .and_then(|b| {
                profile
                    .times
                    .len()
                    .checked_add(profile.baseline_design.len())
                    .and_then(|v| v.checked_add(profile.drive_design.len()))
                    .and_then(|v| v.checked_add(profile.entry_design.len()))
                    .and_then(|v| v.checked_mul(std::mem::size_of::<f64>()))
                    .and_then(|v| b.checked_add(v))
            })
            .and_then(|b| {
                profile
                    .genetics
                    .len()
                    .checked_mul(std::mem::size_of::<Option<f64>>())
                    .and_then(|v| b.checked_add(v))
            })
            .and_then(|b| b.checked_add(std::mem::size_of::<JointReferenceBank>()))
            .and_then(|b| {
                groups
                    .len()
                    .checked_mul(std::mem::size_of::<Option<usize>>())
                    .and_then(|v| b.checked_add(v))
            })
            .and_then(|b| {
                mark_groups
                    .len()
                    .checked_mul(std::mem::size_of::<usize>())
                    .and_then(|v| b.checked_add(v))
            })
            .ok_or_else(|| invalid("joint reference storage overflow"))
    }

    /// Draw one independent population at the anchor coefficients: missing
    /// genetic scores, OU innovations, and event histories proposed from the
    /// population's own rates. Returns the anchor evolution with the bank.
    pub(super) fn generate<R: Rng + ?Sized>(
        model: &JointLikelihood,
        theta: &[f64],
        grid: &ReferenceGrid,
        particles: usize,
        rng: &mut R,
    ) -> Result<(Self, JointReferenceEvolution<f64>), EventHistoryError> {
        model.validate_parameters(theta)?;
        let profile = grid.profile();
        profile.validate(model)?;
        if particles < 2 {
            return Err(invalid("joint reference needs at least two particles"));
        }
        let n = grid.nodes();
        let (groups, mark_groups) = risk_groups(model);
        let count = particles
            .checked_mul(groups.len())
            .ok_or_else(|| invalid("joint reference population dimension overflow"))?;
        let normal_count = noise_count(n, model.spec.signatures, count)?;
        let steps = (n - 1)
            .checked_mul(count)
            .ok_or_else(|| invalid("joint reference history dimension overflow"))?;
        let genes = draw_genomes(model, &profile.genetics, count, rng)?;
        let normals: Vec<f64> = (0..normal_count)
            .map(|_| StandardNormal.sample(rng))
            .collect();
        // Every draw is taken before the evolution runs, so the population is
        // a function of its generator alone, whatever evaluates the steps.
        let uniforms: Vec<f64> = (0..steps).map(|_| rng.random::<f64>()).collect();
        let mut choices = vec![None; steps];
        let mut log_proposal = vec![0.0; steps];
        let mut bank = Self {
            grid: grid.clone(),
            genes,
            normals,
            choices: vec![],
            log_proposal: vec![],
            groups,
            mark_groups,
            particles,
        };
        let evolution = bank.run(
            model,
            theta,
            Proposal::Generate {
                uniforms: &uniforms,
                choices: &mut choices,
                log_proposal: &mut log_proposal,
            },
            None,
        )?;
        bank.choices = choices;
        bank.log_proposal = log_proposal;
        Ok((bank, evolution))
    }

    /// Re-evaluate the fixed population at other coefficients.
    pub(super) fn evolve<S: JetField + Send + Sync>(
        &self,
        model: &JointLikelihood,
        theta: &[S],
    ) -> Result<JointReferenceEvolution<S>, EventHistoryError> {
        self.run(model, theta, Proposal::Replay, None)
    }

    /// Terminal marks and the set's focal once-only mark are integrated as
    /// survival; every other mark fires and jumps the state.
    fn killing_masks(&self, model: &JointLikelihood) -> Vec<Vec<bool>> {
        self.groups
            .iter()
            .map(|focal| {
                model
                    .spec
                    .marks
                    .iter()
                    .enumerate()
                    .map(|(d, kind)| *kind == MarkKind::Terminal || *focal == Some(d))
                    .collect()
            })
            .collect()
    }

    fn block(&self, block: usize, signatures: usize) -> &[f64] {
        let width = self.genes.len() * signatures;
        &self.normals[block * width..(block + 1) * width]
    }

    /// Every mark's log moment over its risk set and that set's log mass. A
    /// mark's activity is evaluated only at the particles of the set its
    /// moment is taken over, so the cost is linear in the number of marks.
    fn moments<S: JetField + Send + Sync>(
        &self,
        marks: usize,
        population: &Population<S>,
        decoder: &super::decoder::PreparedDecoder<S>,
        diagnostics: &mut ReferenceDiagnostics,
    ) -> Result<(Vec<S>, Vec<S>), EventHistoryError> {
        if population
            .states
            .iter()
            .flatten()
            .chain(&population.log_weight)
            .any(|x| !x.value().is_finite())
        {
            return Err(numerical(
                "joint reference state or path weight is non-finite",
            ));
        }
        let count = self.particles;
        let mut moments = Vec::with_capacity(marks);
        let mut masses = Vec::with_capacity(marks);
        for d in 0..marks {
            let group = self.mark_groups[d];
            let start = group * count;
            let weights = &population.log_weight[start..start + count];
            let mass = log_sum_exp(weights);
            let numerator: Vec<S> = population.states[start..start + count]
                .par_iter()
                .zip(weights)
                .map(|(state, w)| w.add(&decoder.activity(d, state)))
                .collect();
            let moment = log_sum_exp(&numerator).sub(&mass);
            let squared_weight: f64 = weights
                .iter()
                .map(|w| (2.0 * (w.value() - mass.value())).exp())
                .sum();
            diagnostics.minimum_risk_effective_samples = diagnostics
                .minimum_risk_effective_samples
                .min(1.0 / squared_weight);
            moments.push(moment);
            masses.push(population.log_mass[group].clone());
        }
        Ok((moments, masses))
    }

    /// The evolution at `theta`. `checkpoints`, when given, receives the
    /// population at the start of every interval for a reverse sweep.
    fn run<S: JetField + Send + Sync>(
        &self,
        model: &JointLikelihood,
        theta: &[S],
        mut proposal: Proposal<'_>,
        mut checkpoints: Option<&mut Vec<Population<S>>>,
    ) -> Result<JointReferenceEvolution<S>, EventHistoryError> {
        model.validate_parameters(theta)?;
        let k = model.spec.signatures;
        let marks = model.spec.marks.len();
        let nodes = self.grid.nodes();
        let count = self.genes.len();
        let history = self.grid.profile().history(marks);
        model.validate_history(&history)?;
        if self.mark_groups.len() != marks
            || count != self.particles * self.groups.len()
            || self.normals.len() != noise_count(nodes, k, count)?
        {
            return Err(invalid(
                "joint reference bank was generated for a different model specification",
            ));
        }
        let gene_count = model.spec.genetic_mean.len();
        let zero = theta[0].constant_like(0.0);
        let decoder = super::decoder::PreparedDecoder::new(model, theta);
        let entry = MeanTable::new(
            theta,
            model.layout.entry.start,
            &model.entry_features(&history),
            k,
            gene_count,
        );
        let genes: Vec<Vec<S>> = self
            .genes
            .iter()
            .map(|g| g.iter().map(|&v| zero.constant_like(v)).collect())
            .collect();
        let initial = self.block(0, k);
        let mut population = Population {
            states: (0..count)
                .map(|p| {
                    (0..k)
                        .map(|axis| add_real(&entry.mean(axis, &genes[p]), initial[p * k + axis]))
                        .collect()
                })
                .collect(),
            log_weight: vec![zero.clone(); count],
            risk: vec![vec![true; marks]; count],
            log_mass: vec![zero.clone(); self.groups.len()],
        };
        let rates: Vec<S> = theta[model.layout.rates.clone()]
            .iter()
            .map(emission::softplus)
            .collect();
        let masks = self.killing_masks(model);
        let killing = KillingSets {
            masks: &masks,
            per_set: self.particles,
        };
        let mut diagnostics = ReferenceDiagnostics {
            minimum_risk_effective_samples: self.particles as f64,
            omitted_event_mass: 0.0,
        };
        let decoder_weights = DecoderWeights::new(&decoder, marks, k);
        let (mut log_moments, mut log_risk_mass) =
            self.moments(marks, &population, &decoder, &mut diagnostics)?;
        let mut times = vec![self.grid.row_time_value(0)];
        let mut row_times = vec![self.grid.row_time(&zero, 0)];
        let mut start_weights = vec![0.0; count];
        for n in 1..nodes {
            if let Some(store) = checkpoints.as_mut() {
                store.push(population.clone());
            }
            let dt = self.grid.dt(&zero, n);
            let columns = self.grid.drive_row(n).to_vec();
            let step = HalfStep::new(&rates, &dt)?;
            let drive = MeanTable::new(theta, model.layout.drive.start, &columns, k, gene_count);
            half_step(&step, &drive, &genes, self.block(2 * n - 1, k), &mut population.states);
            let (midpoint, mid_mass) = self.moments(marks, &population, &decoder, &mut diagnostics)?;
            log_moments.extend(midpoint.iter().cloned());
            let middle_mass_start = log_risk_mass.len();
            log_risk_mass.extend(mid_mass);
            times.push(self.grid.row_time_value(2 * n - 1));
            row_times.push(self.grid.row_time(&zero, 2 * n - 1));
            for (slot, w) in start_weights.iter_mut().zip(&population.log_weight) {
                *slot = w.value();
            }
            let steps = (n - 1) * count..n * count;
            let choice = match &mut proposal {
                Proposal::Generate {
                    uniforms,
                    choices,
                    log_proposal,
                } => EventChoice::Generate {
                    uniforms: &uniforms[steps.clone()],
                    choices: &mut choices[steps.clone()],
                    log_proposal: &mut log_proposal[steps],
                },
                Proposal::Replay => EventChoice::Replay {
                    choices: &self.choices[steps.clone()],
                    log_proposal: &self.log_proposal[steps],
                },
            };
            let step_rates = StepRates::new(
                model,
                theta,
                &decoder_weights,
                &self.grid.midpoint_row(n),
                &midpoint,
                &killing,
            )?;
            let outcome = event_step(
                model,
                theta,
                &decoder,
                &decoder_weights,
                &step_rates,
                &dt,
                &killing,
                &mut population,
                choice,
            )?;
            // At most one non-killing event per step: two or more occur with
            // probability at most H^2/2 at frozen hazard H. Without signatures
            // an event cannot move the population.
            if k > 0 {
                let mut omitted = vec![0.0; self.groups.len()];
                for group in 0..self.groups.len() {
                    let range = group * self.particles..(group + 1) * self.particles;
                    let mass = log_sum_exp(&start_weights[range.clone()]);
                    for p in range {
                        let hazard = outcome.retained_hazard[p];
                        omitted[group] +=
                            (start_weights[p] - mass).exp() * (0.5 * hazard * hazard).min(1.0);
                    }
                }
                diagnostics.omitted_event_mass += omitted.iter().copied().fold(0.0, f64::max);
            }
            // Integrate the killing rather than randomly losing survivors.
            // Normalize the non-killing transition first (its exact mass is
            // one), then retain its survival fraction and conditional law.
            for group in 0..self.groups.len() {
                let start = group * self.particles;
                let end = start + self.particles;
                // Remove a common log-weight offset before subtracting the
                // normalizers. Otherwise a finite survival loss can disappear
                // beside a very negative event importance log weight.
                let shift = population.log_weight[start..end]
                    .iter()
                    .map(JetField::value)
                    .fold(f64::NEG_INFINITY, f64::max);
                for p in start..end {
                    population.log_weight[p] = add_real(&population.log_weight[p], -shift);
                }
                let before = log_sum_exp(&population.log_weight[start..end]);
                for p in start..end {
                    let killed = outcome.log_hazard[outcome.offsets[p]..outcome.offsets[p + 1]]
                        .iter()
                        .fold(zero.clone(), |acc, h| acc.add(&exp(h)));
                    population.log_weight[p] = population.log_weight[p].sub(&killed);
                }
                let after = log_sum_exp(&population.log_weight[start..end]);
                population.log_mass[group] = population.log_mass[group].add(&after.sub(&before));
                for p in start..end {
                    population.log_weight[p] = population.log_weight[p].sub(&after);
                }
            }
            half_step(&step, &drive, &genes, self.block(2 * n, k), &mut population.states);
            let (moment, mass) = self.moments(marks, &population, &decoder, &mut diagnostics)?;
            // The full killed step supplies the endpoint mass. Its log-linear
            // interpolation supplies the midpoint mass at the midpoint time;
            // labeling the interval-start mass as a midpoint creates a lag
            // even for a constant, non-latent hazard.
            for d in 0..marks {
                log_risk_mass[middle_mass_start + d] = log_risk_mass[middle_mass_start + d]
                    .scale(0.5)
                    .add(&mass[d].scale(0.5));
            }
            log_moments.extend(moment);
            log_risk_mass.extend(mass);
            times.push(self.grid.row_time_value(2 * n));
            row_times.push(self.grid.row_time(&zero, 2 * n));
        }
        Ok(JointReferenceEvolution {
            theta: theta.to_vec(),
            times,
            row_times,
            log_moments,
            log_risk_mass,
            marks,
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::{Bound, agrees};
    use super::*;
    use crate::scalar::Rows;

    #[test]
    fn rare_event_probabilities_keep_their_log_sensitivities() {
        let value = log_event(&Rows::seed(Rows::seed(-800.0, [1.0]), [1.0]));
        assert_eq!(value.base.base, -800.0);
        assert_eq!(value.base.rows[0], 1.0);
        assert_eq!(value.rows[0].rows[0], 0.0);
        // Two routes: exp(log 0.03) before expm1 and ln, against 0.03 directly.
        let production = log_event(&ln(&Bound::exact(0.03)));
        let oracle = ln(&emission::expm1(&Bound::exact(0.03).neg()).neg());
        agrees(&production, &oracle, "log event probability");
    }

    /// The reverse sweep admits its workspace from counts alone: a population
    /// whose checkpoints exceed this machine's materialization budget is
    /// refused with a typed error before the primal replay allocates anything.
    /// Every particle's checkpoint costs at least one byte per node, so this
    /// many particles over the profile's nodes exceed the budget; the fixture
    /// allocates only its two-node profile.
    #[test]
    fn an_oversized_pullback_is_refused_before_allocation() {
        let model = JointLikelihood::new(JointSpecification {
            signatures: 1,
            marks: vec![MarkKind::Recurrent],
            baseline_columns: 1,
            drive_columns: 1,
            entry_columns: 0,
            measurements: vec![],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
        .unwrap();
        let nodes = 2;
        let bank = JointReferenceBank {
            grid: ReferenceGrid::new(
                JointReferenceProfile {
                    times: vec![0.0, 1.0],
                    baseline_design: Array2::ones((nodes, 1)),
                    drive_design: Array2::ones((nodes - 1, 1)),
                    entry_design: vec![],
                    genetics: vec![],
                },
                0,
            )
            .unwrap(),
            genes: vec![],
            normals: vec![],
            choices: vec![],
            log_proposal: vec![],
            groups: vec![None],
            mark_groups: vec![0],
            particles: materialization_budget() / (nodes + 4) + 1,
        };
        let theta = vec![0.0; model.layout.width];
        let rows = 2 * nodes - 1;
        let refusal = bank
            .pullback(&model, &theta, &vec![0.0; rows], &vec![0.0; rows])
            .err()
            .unwrap();
        assert!(
            matches!(refusal, EventHistoryError::NumericalFailure { .. })
                && refusal.to_string().contains("materialization budget"),
            "{refusal}"
        );
    }

    /// The set sums regroup the per-mark rescan: at every checkpointed
    /// population, each particle's retained log total must meet `log sum_d
    /// exp(b_d - m_d + a_d(x))` over its retained marks within both routes'
    /// rounding bounds. A fast once-only mark dominates its particles' fired
    /// mass and a slow one does not, so both fired-mark branches run.
    #[test]
    fn set_rate_sums_match_the_per_mark_rescan_through_both_branches() {
        use rand::{SeedableRng, rngs::SmallRng};
        let model = JointLikelihood::new(JointSpecification {
            signatures: 2,
            marks: vec![MarkKind::Recurrent, MarkKind::Once, MarkKind::Once, MarkKind::Terminal],
            baseline_columns: 1,
            drive_columns: 1,
            entry_columns: 0,
            measurements: vec![],
            genetic_mean: vec![0.0],
            genetic_precision: Array2::eye(1),
        })
        .unwrap();
        let mut theta: Vec<f64> = (0..model.layout.width)
            .map(|q| 0.1 * (q as f64).sin())
            .collect();
        for (d, value) in [-1.0, 1.5, -2.5, -2.0].into_iter().enumerate() {
            theta[model.layout.baseline.start + d] = value;
        }
        let profile = JointReferenceProfile {
            times: (0..=16).map(|n| n as f64 / 4.0).collect(),
            baseline_design: Array2::ones((17, 1)),
            drive_design: Array2::ones((16, 1)),
            entry_design: vec![],
            genetics: vec![None],
        };
        let mut rng = SmallRng::seed_from_u64(29);
        let grid = ReferenceGrid::new(profile, 0).unwrap();
        let (bank, _) = JointReferenceBank::generate(&model, &theta, &grid, 64, &mut rng).unwrap();
        let bounded: Vec<Bound> = theta.iter().map(|&v| Bound::exact(v)).collect();
        let mut checkpoints = Vec::new();
        bank.run(&model, &bounded, Proposal::Replay, Some(&mut checkpoints))
            .unwrap();
        let decoder = super::super::decoder::PreparedDecoder::new(&model, &bounded);
        let weights = DecoderWeights::new(&decoder, 4, 2);
        let masks = bank.killing_masks(&model);
        let killing = KillingSets {
            masks: &masks,
            per_set: bank.particles,
        };
        let mut diagnostics = ReferenceDiagnostics {
            minimum_risk_effective_samples: 64.0,
            omitted_event_mass: 0.0,
        };
        let (mut subtracted, mut direct) = (0, 0);
        let mut axis = Vec::new();
        for (n, population) in checkpoints.iter().enumerate() {
            let normaliser = bank
                .moments(4, population, &decoder, &mut diagnostics)
                .unwrap()
                .0;
            let rates = StepRates::new(
                &model,
                &bounded,
                &weights,
                &bank.grid.midpoint_row(n + 1),
                &normaliser,
                &killing,
            )
            .unwrap();
            for (p, state) in population.states.iter().enumerate() {
                let set = p / bank.particles;
                let (mask, risk) = (&masks[set], &population.risk[p]);
                let softplus: Vec<Bound> = state.iter().map(emission::softplus).collect();
                if rates.retained_sums(set, mask, risk, &weights, &softplus, &mut axis) == 0 {
                    continue;
                }
                let production = add_real(&ln(&combine(&axis, &softplus)), rates.shift(set));
                let retained: Vec<Bound> = (0..4)
                    .filter(|&d| !mask[d] && risk[d])
                    .map(|d| rates.log_scale(d).add(&decoder.activity(d, state)))
                    .collect();
                agrees(&production, &log_sum_exp(&retained), &format!("retained total {n},{p}"));
                let removed = (0..4)
                    .filter(|&d| !mask[d] && !risk[d])
                    .map(|d| rates.scaled(set, d).mul(&combine(weights.row(d), &softplus)))
                    .fold(None, |acc: Option<Bound>, term| {
                        Some(match acc {
                            None => term,
                            Some(a) => a.add(&term),
                        })
                    });
                if let Some(removed) = removed {
                    if removed.value <= 0.5 * combine(rates.sums(set), &softplus).value {
                        subtracted += 1;
                    } else {
                        direct += 1;
                    }
                }
            }
        }
        println!("BRANCHES subtracted {subtracted} direct {direct}");
        assert!(
            subtracted > 0 && direct > 0,
            "both fired-mark branches must run: subtracted {subtracted}, direct {direct}"
        );
    }

    #[test]
    fn reference_moments_stay_finite_beside_underflowed_weights() {
        let model = JointLikelihood::new(JointSpecification {
            signatures: 1,
            marks: vec![MarkKind::Recurrent],
            baseline_columns: 1,
            drive_columns: 1,
            entry_columns: 0,
            measurements: vec![],
            genetic_mean: vec![],
            genetic_precision: Array2::zeros((0, 0)),
        })
        .unwrap();
        let bank = JointReferenceBank {
            grid: ReferenceGrid::new(
                JointReferenceProfile {
                    times: vec![0.0, 1.0],
                    baseline_design: Array2::ones((2, 1)),
                    drive_design: Array2::ones((1, 1)),
                    entry_design: vec![],
                    genetics: vec![],
                },
                0,
            )
            .unwrap(),
            genes: vec![vec![]; 2],
            normals: vec![],
            choices: vec![],
            log_proposal: vec![],
            groups: vec![None],
            mark_groups: vec![0],
            particles: 2,
        };
        let theta = vec![Bound::exact(0.0); model.layout.width];
        let mut population = Population {
            states: vec![vec![Bound::exact(1e300)], vec![Bound::exact(0.0)]],
            log_weight: vec![Bound::exact(-800.0), Bound::exact(0.0)],
            risk: vec![vec![true]; 2],
            log_mass: vec![Bound::exact(0.0)],
        };
        let mut diagnostics = ReferenceDiagnostics {
            minimum_risk_effective_samples: 2.0,
            omitted_event_mass: 0.0,
        };
        let decoder = super::super::decoder::PreparedDecoder::new(&model, &theta);
        let (moments, _) = bank
            .moments(1, &population, &decoder, &mut diagnostics)
            .unwrap();
        // The 1e300 state carries weight exp(-800): the moment is the
        // activity of the other particle, and one particle holds the mass.
        // The other term enters two log-sum-exps at relative size exp(-110).
        let expected = decoder.activity(0, &[Bound::exact(0.0)]);
        agrees(&moments[0], &expected, "moment beside an underflowed weight");
        // exp(-1600) underflows, so the squared weights sum to exactly one.
        assert_eq!(diagnostics.minimum_risk_effective_samples, 1.0);
        population.states[0][0] = Bound::exact(f64::INFINITY);
        assert!(
            bank.moments(1, &population, &decoder, &mut diagnostics)
                .is_err()
        );
    }
}

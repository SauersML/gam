//! `steer_delta` — the **steering primitive with output dosimetry**: the
//! actionable LLM payload of the SAE-manifold machine.
//!
//! # What this computes
//!
//! Given a fitted [`SaeManifoldTerm`] and the per-row output-Fisher
//! [`RowMetric`], a *steering move* is "drive atom `k`'s latent coordinate from
//! `t_from` to `t_to`". The atom's decoder curve `g_k(t) = Φ_k(t) B_k` maps that
//! latent move to an **activation-space delta** — the actual vector you add to
//! the residual stream / reconstruction to realize the move *on the manifold*.
//! Here `g_k(t) = Phi_k^eta(t) B_k` is the fitted physical decoder, including
//! the curvature-homotopy state:
//!
//! ```text
//! delta = a · ( g_k(t_to) - g_k(t_from) )      (the on-manifold move)
//! ```
//!
//! where `a` is the atom's amplitude (how loudly the atom is expressed). This is
//! the thing a downstream consumer adds to a hidden state.
//!
//! # Dosimetry — how big is this push, in nats?
//!
//! The headline number is the **predicted output effect**: how much behavioral
//! change (in nats of KL on the model's output distribution) the exact applied
//! activation move induces. For a locally-quadratic output readout the KL of a
//! move `delta` is `0.5 * delta^T F delta`, with `F` the output-Fisher
//! information — exactly the inner product [`RowMetric`] carries:
//!
//! ```text
//! predicted_nats = 0.5 * delta^T M_metric_row delta
//! ```
//!
//! This endpoint quadratic form is the single canonical nats prediction because
//! it prices the same `delta` a patched forward pass applies. Arc energy and
//! tangent-only surrogates are deliberately not exposed as alternate nats lanes:
//! they price different objects and therefore cannot be calibrated against the
//! patched-forward endpoint KL by construction (#2249).
//!
//! # Off-manifold guard
//!
//! `δ` is, by construction, a chord of the decoder curve, so it should lie in the
//! atom's local tangent/frame at `t_from` (up to second-order curvature). The
//! **off-manifold norm** projects `δ` onto the span of the local decoder tangents
//! `∂g_k/∂t` at `t_from` and reports the residual norm — a self-check that the
//! steering move stays on the learned surface. It is `≈ 0` for small steps and
//! grows with arc curvature; a large value means the requested move left the
//! manifold and the dose number is not to be trusted.
//!
//! # Read-only / no loss contact
//!
//! This module is a **pure read** over the fitted term and the metric. It calls
//! only `g_k(t)` evaluation ([`SaeManifoldAtom`]'s decoder + installed
//! `SaeBasisEvaluator`) and the criterion-facing
//! [`RowMetric::fisher_mass`] / [`RowMetric::pullback`]. It never mutates the
//! model, never touches a likelihood / criterion / penalty, and the solver floor
//! `δ` of [`RowMetric`] never enters any number it reports (the fisher-mass /
//! pullback face is `δ`-free, #747).

use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array1, Array2, ArrayView1};

use crate::basis::SaeBasisEvaluator;
use crate::chart_canonicalization::{CanonicalChartTopology, chart_arclength_coordinates};
use crate::manifold::{LatentManifold, SaeManifoldAtom, SaeManifoldTerm};
use gam_problem::{FisherFactorKind, MetricProvenance, RowMetric};

pub use gam_response::steer_plan::{FisherDoseKind, SteerPlan};

fn shortest_coordinate_delta(
    from: &[f64],
    to: &[f64],
    periods: &[Option<f64>],
) -> Result<Vec<f64>, String> {
    if from.len() != to.len() || from.len() != periods.len() {
        return Err(format!(
            "coordinate displacement length mismatch: from={}, to={}, periods={}",
            from.len(),
            to.len(),
            periods.len()
        ));
    }
    let mut delta = Vec::with_capacity(from.len());
    for axis in 0..from.len() {
        let mut d = to[axis] - from[axis];
        if let Some(period) = periods[axis] {
            if !(period.is_finite() && period > 0.0) {
                return Err(format!(
                    "coordinate axis {axis} has invalid period {period}"
                ));
            }
            d -= period * (d / period).round();
        }
        delta.push(d);
    }
    Ok(delta)
}

/// Build a [`SteerPlan`] for driving atom `atom_k` from `t_from` to `t_to`.
///
/// `model` is the fitted term (read only); `metric` is the per-row output-Fisher
/// inner product the dose is measured through (typically `model.row_metric()`'s
/// own metric, or any metric whose row/output dims match the term). `t_from` and
/// `t_to` are latent coordinates of length `atom.latent_dim`.
///
/// Errors when the atom index is out of range, the coordinate lengths do not
/// match the atom's latent dimension, the atom has no installed
/// [`crate::manifold::SaeBasisEvaluator`] (arbitrary-`t` evaluation
/// requires one), or the metric dimensions do not match the term. Under a
/// Euclidean (no-behavior) metric the geometry is still produced but
/// `predicted_nats` degrades to `None`.
pub fn steer_delta(
    model: &SaeManifoldTerm,
    metric: &RowMetric,
    atom_k: usize,
    metric_row: usize,
    amplitude: f64,
    t_from: &[f64],
    t_to: &[f64],
) -> Result<SteerPlan, String> {
    if !(amplitude.is_finite() && amplitude > 0.0) {
        return Err(format!(
            "steer_delta: amplitude must be finite and positive, got {amplitude}"
        ));
    }
    let k = model.k_atoms();
    if atom_k >= k {
        return Err(format!(
            "steer_delta: atom index {atom_k} out of range (term has {k} atoms)"
        ));
    }
    let atom = &model.atoms[atom_k];
    let d = atom.latent_dim();
    let p = atom.output_dim();
    if t_from.len() != d || t_to.len() != d {
        return Err(format!(
            "steer_delta: t_from/t_to must have length latent_dim={d}; got {} and {}",
            t_from.len(),
            t_to.len()
        ));
    }
    atom.basis_evaluator.as_ref().ok_or_else(|| {
        format!(
            "steer_delta: atom {atom_k} ('{}') has no installed basis evaluator; \
             arbitrary-t decoder evaluation requires one",
            atom.name
        )
    })?;
    let periods = model.assignment.coords[atom_k].effective_axis_periods();
    let coordinate_delta = shortest_coordinate_delta(t_from, t_to, &periods)?;

    let n = model.n_obs();
    if metric.n_rows() != n || metric.p_out() != p {
        return Err(format!(
            "steer_delta: metric shape ({}, {}) must equal fitted term shape ({n}, {p})",
            metric.n_rows(),
            metric.p_out()
        ));
    }
    if metric_row >= n {
        return Err(format!(
            "steer_delta: metric_row={metric_row} out of range for {n} fitted rows"
        ));
    }

    // --- the on-manifold activation-space delta -----------------------------
    let tier0_scale = model.tier0_scale();
    let g_from = decode_at(atom, t_from, tier0_scale)?;
    let g_to = decode_at(atom, t_to, tier0_scale)?;
    let mut delta = Array1::<f64>::zeros(p);
    for i in 0..p {
        delta[i] = amplitude * (g_to[i] - g_from[i]);
    }

    // Whether the metric can/does match this term and carries behavior.
    let provenance = metric.provenance();
    let behavior_available = metric_carries_behavior(provenance);
    let fisher_mass_captured = behavior_available.then(|| metric.row_traces()[metric_row]);
    let fisher_mass_residual = behavior_available
        .then(|| metric.truncation_mass_residual(metric_row))
        .flatten();
    let fisher_mass_residual_fraction = behavior_available
        .then(|| metric.truncation_mass_residual_fraction(metric_row))
        .flatten();
    let predicted_nats_kind = if !behavior_available {
        FisherDoseKind::Unavailable
    } else {
        match metric.fisher_factor_kind() {
            Some(FisherFactorKind::ExactFull) => FisherDoseKind::ExactFull,
            Some(FisherFactorKind::CertifiedPsdLowerBound) => {
                FisherDoseKind::CertifiedPsdLowerBound
            }
            Some(FisherFactorKind::UncertifiedApproximation) => {
                FisherDoseKind::UncertifiedApproximation
            }
            None => {
                return Err(format!(
                    "steer_delta: behavioral metric provenance {provenance:?} has no explicit Fisher factor status"
                ));
            }
        }
    };

    // --- off-manifold guard -------------------------------------------------
    // Project δ onto the span of the local decoder tangents ∂g_k/∂t and report
    // the residual norm. The tangents are evaluated at the move's MIDPOINT, not
    // at t_from: the chord of a curve is symmetric about its midpoint, so its
    // component transverse to the midpoint tangent is the true second-order
    // sagitta (`O(‖Δt‖²)`), whereas the endpoint tangent differs from the chord
    // direction already at first order. Measuring against the midpoint frame is
    // therefore the honest "did the move stay on the surface" self-check: it is
    // `≈ 0` for an on-manifold move and grows only with genuine arc curvature.
    let mut t_mid = vec![0.0_f64; d];
    for a in 0..d {
        t_mid[a] = t_from[a] + 0.5 * coordinate_delta[a];
        if let Some(period) = periods[a] {
            t_mid[a] = t_mid[a].rem_euclid(period);
        }
    }
    let tangents = decode_tangents_at(atom, &t_mid, tier0_scale)?;
    let off_manifold_norm = off_manifold_residual_norm(&tangents, delta.view())?;

    // --- dosimetry: exact applied-delta Fisher endpoint KL ------------------
    let predicted_nats =
        behavior_available.then(|| 0.5 * metric.fisher_mass(metric_row, delta.view()));

    Ok(SteerPlan {
        atom: atom_k,
        atom_name: atom.name.clone(),
        t_from: t_from.to_vec(),
        t_to: t_to.to_vec(),
        amplitude,
        metric_row,
        delta,
        predicted_nats,
        predicted_nats_kind,
        fisher_mass_captured,
        fisher_mass_residual,
        fisher_mass_residual_fraction,
        off_manifold_norm,
        metric_provenance: provenance,
    })
}

/// One model-in-the-loop observation of the exact [`SteerPlan`] supplied to an
/// [`AppliedDoseProbe`]. The effective delta is the vector the downstream model
/// actually received after its device/dtype conversion; it may differ from the
/// f64 requested delta through quantization. `exact_directional_nats` is the
/// full local-Fisher quadratic of that effective delta. `measured_nats` is the
/// patched-forward `KL(p_base ‖ p_patched)` for the same effective delta.
///
/// Keeping all three values atomic prevents a caller from pricing one vector
/// and measuring another, and keeps the resident [`RowMetric`] dose an explicit
/// diagnostic rather than silently promoting an approximate operator (#2249).
/// `certified_attainable_upper_nats`, when present, is a global upper bound on
/// measured KL over every displacement along the request's direction that the
/// chart reaches, in this downstream execution context. A point observation or
/// an apparent plateau is not such a certificate.
#[derive(Clone, Debug, PartialEq)]
pub struct AppliedDoseObservation {
    pub effective_delta: Array1<f64>,
    pub exact_directional_nats: f64,
    pub measured_nats: f64,
    pub certified_attainable_upper_nats: Option<f64>,
}

/// Plan-aware external-model dose probe. The callback must execute the supplied
/// plan; a scalar-amplitude callback cannot prove which activation delta it
/// actually applied and is deliberately not part of the contract.
pub type AppliedDoseProbe<'a> =
    dyn FnMut(&SteerPlan) -> Result<AppliedDoseObservation, String> + 'a;

/// One target-dose solve along one atom's chart.
///
/// The model and row metric remain explicit execution context; everything that
/// identifies the requested dose is carried together so callers cannot
/// accidentally reorder a train of homogeneous scalar/slice arguments. There is
/// no accuracy or probe-budget option: the solve resolves the displacement to its
/// representation limit, and the safeguarded bracket bounds how many
/// observations that takes (see [`steer_to_target_nats`]).
#[derive(Clone, Copy, Debug)]
pub struct TargetDoseRequest<'a> {
    /// The atom whose coordinate is being steered.
    pub atom_k: usize,
    /// Exact fitted row whose output-Fisher block prices the move. The move is
    /// written at this row's own fitted gate for the atom, never at a caller
    /// amplitude.
    pub metric_row: usize,
    /// Source on-manifold coordinate.
    pub t_from: &'a [f64],
    /// Chart direction to move along, in the atom's coordinates. Only its
    /// direction is used: how far to move is what the solve finds, and the landing
    /// coordinate is returned as `steer.t_to` of the [`TargetDosePlan`].
    pub direction: &'a [f64],
    /// Requested output-KL dose in nats.
    pub target_nats: f64,
}

/// A target output-KL dose reached by moving one atom's coordinate along its
/// chart, returned atomically with the exact activation-space move the caller
/// must apply (gh#2249/#2263). The move is a chord of the atom's decoded image at
/// the row's own intensity, so the edited row sits on the image at `steer.t_to`.
#[derive(Clone, Debug)]
pub struct TargetDosePlan {
    /// The requested dose in nats of KL.
    pub target_nats: f64,
    /// Closed-form first-order displacement `s0 = sqrt(2 q* / (a² ‖J u‖²_M))` along
    /// the unit direction `u` (`J` the decoder tangents at `t_from`, `a` the row's
    /// gate, `M` the row output-Fisher), exact while the chord is linear in `s`.
    pub seed_displacement: f64,
    /// The solved displacement along the unit direction: `steer.t_to` is `t_from`
    /// retracted by `displacement · u`.
    pub displacement: f64,
    /// Exact applied move at the solved coordinate, including `delta`, predicted
    /// dose, metric provenance, chart radius, and off-manifold audit.
    pub steer: SteerPlan,
    /// Exact directional local-Fisher dose, effective applied delta, and measured
    /// patched-forward KL at [`SteerPlan::t_to`]. `None` for an exact-factor solve
    /// with no model in the loop. This never changes the resident metric or its
    /// status.
    pub applied_probe: Option<AppliedDoseObservation>,
    /// Number of patched-forward probes consumed (0 without a callback).
    pub iterations: usize,
    /// Tightest global attainable-dose upper bound certified by any applied
    /// probe in this solve. `None` means no global envelope was certified.
    pub certified_attainable_upper_nats: Option<f64>,
}

/// A measured target-dose solve either returns a certified plan or one of these
/// explicit failure states. No unconverged iterate is representable as success.
#[derive(Clone, Debug, PartialEq)]
pub enum TargetDoseError {
    InvalidRequest(String),
    Steering(String),
    Probe(String),
    FactorNeedsAppliedDoseProbe {
        kind: FisherDoseKind,
    },
    /// The atom's fitted gate at `metric_row` is zero: moving its coordinate
    /// cannot change that row, so no dose is attainable.
    AtomNotExpressed {
        atom_k: usize,
        metric_row: usize,
    },
    UnreachableTarget {
        target_nats: f64,
        certified_attainable_upper_nats: f64,
    },
    /// The direction ran out of chart (an interval boundary, a full turn of a
    /// move confined to periodic axes, or a retraction that stopped moving the
    /// point) before any observation reached the target. Observations are points,
    /// so this is a refusal, not a certificate that the dose is unattainable.
    ChartExtentExhausted {
        target_nats: f64,
        extent: f64,
        max_observed_nats: f64,
    },
    /// Along a direction with no chart boundary, doubling the displacement left
    /// the finite `f64` range before any observation reached the target.
    UnbracketedTarget {
        target_nats: f64,
        max_probed_displacement: f64,
        max_measured_nats: f64,
        probes: usize,
    },
}

impl std::fmt::Display for TargetDoseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(message) | Self::Steering(message) | Self::Probe(message) => {
                f.write_str(message)
            }
            Self::FactorNeedsAppliedDoseProbe { kind } => write!(
                f,
                "steer_to_target_nats: factor kind {} cannot solve a full-KL target without an applied-dose probe",
                kind.as_str()
            ),
            Self::UnreachableTarget {
                target_nats,
                certified_attainable_upper_nats,
            } => write!(
                f,
                "steer_to_target_nats: target {target_nats} nats is outside the certified \
                 attainable envelope, whose global upper bound is \
                 {certified_attainable_upper_nats} nats"
            ),
            Self::AtomNotExpressed { atom_k, metric_row } => write!(
                f,
                "steer_to_target_nats: atom {atom_k} has a zero fitted gate at row {metric_row}; \
                 moving its coordinate cannot change that row, so no dose is attainable"
            ),
            Self::ChartExtentExhausted {
                target_nats,
                extent,
                max_observed_nats,
            } => write!(
                f,
                "steer_to_target_nats: target {target_nats} nats was not reached before the \
                 chart ended along the requested direction (extent {extent}); the largest \
                 observed dose was {max_observed_nats} nats"
            ),
            Self::UnbracketedTarget {
                target_nats,
                max_probed_displacement,
                max_measured_nats,
                probes,
            } => write!(
                f,
                "steer_to_target_nats: could not bracket target {target_nats} nats: along a \
                 direction with no chart boundary the displacement left the finite range \
                 after {probes} probes through displacement {max_probed_displacement}; the \
                 largest observed dose was {max_measured_nats} nats and no global attainable \
                 envelope certified the target unreachable"
            ),
        }
    }
}

impl std::error::Error for TargetDoseError {}

/// Execute and validate one model-in-the-loop observation. Validation lives in
/// the Rust authority so every binding fails closed on malformed model data.
fn probe_applied_dose(
    probe: &mut AppliedDoseProbe<'_>,
    plan: &SteerPlan,
) -> Result<AppliedDoseObservation, TargetDoseError> {
    let observation = probe(plan).map_err(TargetDoseError::Probe)?;
    if observation.effective_delta.len() != plan.delta.len() {
        return Err(TargetDoseError::Probe(format!(
            "steer_to_target_nats: probe effective_delta length {} does not match plan delta length {}",
            observation.effective_delta.len(),
            plan.delta.len()
        )));
    }
    if !observation
        .effective_delta
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(TargetDoseError::Probe(
            "steer_to_target_nats: probe effective_delta must be finite".to_string(),
        ));
    }
    if !(observation.exact_directional_nats.is_finite()
        && observation.exact_directional_nats >= 0.0)
    {
        return Err(TargetDoseError::Probe(format!(
            "steer_to_target_nats: probe exact_directional_nats must be finite and non-negative; got {}",
            observation.exact_directional_nats
        )));
    }
    if !(observation.measured_nats.is_finite() && observation.measured_nats >= 0.0) {
        return Err(TargetDoseError::Probe(format!(
            "steer_to_target_nats: probe measured_nats must be finite and non-negative; got {}",
            observation.measured_nats
        )));
    }
    if let Some(upper) = observation.certified_attainable_upper_nats {
        if !(upper.is_finite() && upper >= 0.0) {
            return Err(TargetDoseError::Probe(format!(
                "steer_to_target_nats: probe certified_attainable_upper_nats must be finite \
                 and non-negative when present; got {upper}"
            )));
        }
        if observation.measured_nats > upper {
            return Err(TargetDoseError::Probe(format!(
                "steer_to_target_nats: measured dose {} exceeds the probe's certified \
                 global attainable upper bound {upper}",
                observation.measured_nats
            )));
        }
    }
    Ok(observation)
}

/// Merge one probe's optional global certificate into the solve-wide envelope.
/// Every certificate must bound every observation in the same solve, not only
/// the point at which it was returned.
fn merge_attainable_envelope(
    observation: &AppliedDoseObservation,
    max_measured_nats: f64,
    envelope: &mut Option<f64>,
) -> Result<(), TargetDoseError> {
    if let Some(upper) = observation.certified_attainable_upper_nats {
        *envelope = Some(envelope.map_or(upper, |current| current.min(upper)));
    }
    if let Some(upper) = *envelope
        && max_measured_nats > upper
    {
        return Err(TargetDoseError::Probe(format!(
            "steer_to_target_nats: observed dose {max_measured_nats} exceeds an earlier \
             certified global attainable upper bound {upper}"
        )));
    }
    Ok(())
}

/// Solve-wide bookkeeping over the observations one target-dose solve makes.
#[derive(Default)]
struct DoseObservations {
    probes: usize,
    max_observed_nats: f64,
    certified_attainable_upper_nats: Option<f64>,
}

impl DoseObservations {
    /// Observe the dose of `plan`: a patched forward when a probe is supplied,
    /// otherwise the exact resident dose the plan already carries.
    fn observe(
        &mut self,
        probe: Option<&mut AppliedDoseProbe<'_>>,
        plan: &SteerPlan,
    ) -> Result<(f64, Option<AppliedDoseObservation>), TargetDoseError> {
        let Some(probe) = probe else {
            let nats = plan.predicted_nats.ok_or_else(|| {
                TargetDoseError::InvalidRequest(
                    "steer_to_target_nats: an exact-factor solve observed a move with no dose"
                        .to_string(),
                )
            })?;
            self.max_observed_nats = self.max_observed_nats.max(nats);
            return Ok((nats, None));
        };
        let observation = probe_applied_dose(probe, plan)?;
        self.probes += 1;
        self.max_observed_nats = self.max_observed_nats.max(observation.measured_nats);
        merge_attainable_envelope(
            &observation,
            self.max_observed_nats,
            &mut self.certified_attainable_upper_nats,
        )?;
        Ok((observation.measured_nats, Some(observation)))
    }
}

/// How far `t_from` can move along the unit chart direction `u` before the chart
/// stops producing new points. An `Interval` axis ends at its boundary, where the
/// retraction would start clamping; a move confined to periodic axes has made a
/// full turn once its fastest axis has. A component along an unbounded axis
/// leaves only the interval boundaries.
fn chart_extent(manifold: &LatentManifold, t_from: &[f64], u: &[f64]) -> Result<f64, String> {
    #[derive(Clone, Copy)]
    enum Axis {
        Unbounded,
        Periodic(f64),
        Bounded(f64, f64),
    }
    fn collect(manifold: &LatentManifold, dim: usize, axes: &mut Vec<Axis>) {
        match manifold {
            LatentManifold::Euclidean => axes.extend((0..dim).map(|_| Axis::Unbounded)),
            LatentManifold::Circle { period } => axes.push(Axis::Periodic(*period)),
            LatentManifold::Sphere { dim } => axes.extend((0..*dim).map(|_| Axis::Unbounded)),
            LatentManifold::Interval { lo, hi } => {
                axes.push(Axis::Bounded(lo.min(*hi), lo.max(*hi)));
            }
            LatentManifold::Product(parts)
            | LatentManifold::ProductWithMetric {
                manifolds: parts, ..
            } => {
                for part in parts {
                    collect(part, part.ambient_dim(1), axes);
                }
            }
        }
    }
    let mut axes = Vec::with_capacity(u.len());
    collect(manifold, u.len(), &mut axes);
    if axes.len() != u.len() {
        return Err(format!(
            "steer_to_target_nats: the atom's manifold has {} axes but the direction has {}",
            axes.len(),
            u.len()
        ));
    }
    let mut bounded = f64::INFINITY;
    let mut periodic = f64::INFINITY;
    let mut moves_unbounded = false;
    for ((&axis, &step), &start) in axes.iter().zip(u).zip(t_from) {
        if step == 0.0 {
            continue;
        }
        match axis {
            Axis::Unbounded => moves_unbounded = true,
            Axis::Periodic(period) => periodic = periodic.min(period / step.abs()),
            Axis::Bounded(lo, hi) => {
                if !(lo <= start && start <= hi) {
                    return Err(format!(
                        "steer_to_target_nats: t_from {start} lies outside the chart interval \
                         [{lo}, {hi}]"
                    ));
                }
                let room = if step > 0.0 { hi - start } else { start - lo };
                bounded = bounded.min(room / step.abs());
            }
        }
    }
    Ok(if moves_unbounded {
        bounded
    } else {
        bounded.min(periodic)
    })
}

/// Solve for how far to move atom `atom_k`'s coordinate from `t_from` along
/// `direction` so that the move lands a target output-KL dose `target_nats`
/// (gh#2263 target-dose surface).
///
/// The move is written at the row's own fitted gate `a` and is always a chord of
/// the atom's decoded image, `delta = a·(g(t_to) − g(t_from))` with
/// `t_to = retract(t_from, s·u)` for the unit direction `u`. The unknown is the
/// displacement `s`, never the amplitude: scaling the amplitude writes the row off
/// the image, which is the one defect behind gh#2263 items 1 and 3.
///
/// The closed form `s0 = sqrt(2 q* / (a² ‖J u‖²_M))` seeds the solve. It is exact
/// only while the chord is linear in `s`, and correctly scaled only because the
/// tangents and the metric share the raw activation frame (gh#2249, `ace3b9af3`).
/// The solve expands `s` in increasing order from the seed until two observations
/// bracket the target, then resolves the bracket by false position with the
/// Illinois weighting: an endpoint retained twice in a row has its residual halved
/// for the next secant, so a stale endpoint is pulled in and the bracket shrinks
/// superlinearly from both sides. A secant that is not strictly inside the bracket
/// falls back to bisection, and so does any candidate once two observations have
/// not halved the bracket, so the bracket halves at least once every three
/// observations. A local decrease is only another point observation, so expansion
/// continues through it, but never past the chart's extent along the direction
/// (`chart_extent`): past an interval boundary the retraction would clamp, and past
/// a full turn the points repeat. Running out of chart is
/// [`TargetDoseError::ChartExtentExhausted`]; a plan is never clamped to fit.
///
/// Every solve resolves the displacement to the representation limit of `s`: it
/// returns the observation whose dose equals the target or, once no representable
/// displacement lies strictly inside the bracket, the endpoint whose observed dose
/// is closer. There is no accuracy option and no probe budget. The safeguarded
/// bracket bounds the observations by the expansion plus three per halving from
/// the first bracket down to one ulp. With `probe = None` each observation is the
/// exact resident dose (an `ExactFull` factor is required). With a probe each
/// observation is a patched forward, and `UnreachableTarget` is possible only when
/// a probe certifies a global attainable-dose upper bound below the target.
pub fn steer_to_target_nats(
    model: &SaeManifoldTerm,
    metric: &RowMetric,
    request: TargetDoseRequest<'_>,
    mut probe: Option<&mut AppliedDoseProbe<'_>>,
) -> Result<TargetDosePlan, TargetDoseError> {
    let TargetDoseRequest {
        atom_k,
        metric_row,
        t_from,
        direction,
        target_nats,
    } = request;
    if !(target_nats.is_finite() && target_nats > 0.0) {
        return Err(TargetDoseError::InvalidRequest(format!(
            "steer_to_target_nats: target_nats must be finite and positive, got {target_nats}"
        )));
    }
    let k = model.k_atoms();
    if atom_k >= k {
        return Err(TargetDoseError::InvalidRequest(format!(
            "steer_to_target_nats: atom index {atom_k} out of range (term has {k} atoms)"
        )));
    }
    let atom = &model.atoms[atom_k];
    let d = atom.latent_dim();
    if t_from.len() != d || direction.len() != d {
        return Err(TargetDoseError::InvalidRequest(format!(
            "steer_to_target_nats: t_from and direction must have length latent_dim={d}; got {} \
             and {}",
            t_from.len(),
            direction.len()
        )));
    }
    let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Err(TargetDoseError::InvalidRequest(format!(
            "steer_to_target_nats: direction must be finite and nonzero; got {direction:?}"
        )));
    }
    let u: Vec<f64> = direction.iter().map(|v| v / norm).collect();
    let gate = model
        .assignment
        .try_assignments_row(metric_row)
        .map_err(TargetDoseError::InvalidRequest)?[atom_k];
    if !gate.is_finite() {
        return Err(TargetDoseError::InvalidRequest(format!(
            "steer_to_target_nats: atom {atom_k} has a non-finite gate {gate} at row {metric_row}"
        )));
    }
    if gate <= 0.0 {
        return Err(TargetDoseError::AtomNotExpressed { atom_k, metric_row });
    }
    let manifold = model.assignment.coords[atom_k].manifold();
    let extent = chart_extent(manifold, t_from, &u).map_err(TargetDoseError::InvalidRequest)?;
    let moved = |s: f64| -> Vec<f64> {
        let step: Vec<f64> = u.iter().map(|v| s * v).collect();
        manifold
            .retract(ArrayView1::from(t_from), ArrayView1::from(&step[..]))
            .to_vec()
    };
    let plan_at = |s: f64| {
        steer_delta(model, metric, atom_k, metric_row, gate, t_from, &moved(s))
            .map_err(TargetDoseError::Steering)
    };

    // The exact point KL(0) = 0 is the lower end of every bracket, and its plan
    // carries the metric provenance and dose kind every later move shares.
    let origin = plan_at(0.0)?;
    if origin.predicted_nats.is_none() {
        return Err(TargetDoseError::InvalidRequest(format!(
            "steer_to_target_nats: atom {atom_k} has no behavioral (nats) metric \
             (provenance {:?}); a target-nats dose is undefined",
            origin.metric_provenance
        )));
    }
    let exact = probe.is_none();
    if exact && origin.predicted_nats_kind != FisherDoseKind::ExactFull {
        return Err(TargetDoseError::FactorNeedsAppliedDoseProbe {
            kind: origin.predicted_nats_kind,
        });
    }
    // First-order dose along the direction: q(s) ≈ lin · s².
    let tangents =
        decode_tangents_at(atom, t_from, model.tier0_scale()).map_err(TargetDoseError::Steering)?;
    let p = atom.output_dim();
    let mut tangent = Array1::<f64>::zeros(p);
    for i in 0..p {
        for a in 0..d {
            tangent[i] += tangents[[i, a]] * u[a];
        }
    }
    let lin = 0.5 * gate * gate * metric.fisher_mass(metric_row, tangent.view());
    if !(lin.is_finite() && lin > 0.0) {
        return Err(TargetDoseError::InvalidRequest(format!(
            "steer_to_target_nats: the direction carries no local dose at t_from in metric row \
             {metric_row} (first-order coefficient {lin})"
        )));
    }
    let seed_displacement = (target_nats / lin).sqrt();
    if !(seed_displacement.is_finite() && seed_displacement > 0.0) {
        return Err(TargetDoseError::InvalidRequest(format!(
            "steer_to_target_nats: target {target_nats} nats and first-order coefficient {lin} \
             imply an unrepresentable displacement {seed_displacement}"
        )));
    }
    if !(extent > 0.0) {
        return Err(TargetDoseError::ChartExtentExhausted {
            target_nats,
            extent,
            max_observed_nats: 0.0,
        });
    }

    let finish = |displacement: f64,
                  steer: SteerPlan,
                  applied_probe: Option<AppliedDoseObservation>,
                  observations: &DoseObservations| TargetDosePlan {
        target_nats,
        seed_displacement,
        displacement,
        steer,
        applied_probe,
        iterations: observations.probes,
        certified_attainable_upper_nats: observations.certified_attainable_upper_nats,
    };
    let mut observations = DoseObservations::default();

    // Establish a genuine sign-change bracket [lo, hi] by expanding the seed in
    // displacement order. No finite set of non-increasing observations proves a
    // global plateau; only a callback-supplied global envelope can certify that
    // the target is outside the attainable range.
    // The first observation is capped at half the extent, so a refusal at the
    // chart's end has observed its interior and not only the point one full turn
    // away, which on a periodic chart is the start again.
    let (mut lo_s, mut lo_nats, mut lo_plan, mut lo_applied) = (0.0_f64, 0.0_f64, origin, None);
    let mut hi_s = seed_displacement.min(0.5 * extent);
    let mut hi_plan = plan_at(hi_s)?;
    let (mut hi_nats, mut hi_applied) = observations.observe(probe.as_deref_mut(), &hi_plan)?;
    if hi_nats == target_nats {
        return Ok(finish(hi_s, hi_plan, hi_applied, &observations));
    }
    if let Some(upper) = observations.certified_attainable_upper_nats
        && upper < target_nats
    {
        return Err(TargetDoseError::UnreachableTarget {
            target_nats,
            certified_attainable_upper_nats: upper,
        });
    }
    while hi_nats < target_nats {
        if hi_s >= extent {
            return Err(TargetDoseError::ChartExtentExhausted {
                target_nats,
                extent,
                max_observed_nats: observations.max_observed_nats,
            });
        }
        let next_s = (2.0 * hi_s).min(extent);
        if !next_s.is_finite() {
            return Err(TargetDoseError::UnbracketedTarget {
                target_nats,
                max_probed_displacement: hi_s,
                max_measured_nats: observations.max_observed_nats,
                probes: observations.probes,
            });
        }
        let next_plan = plan_at(next_s)?;
        if next_plan.t_to == hi_plan.t_to {
            // The retraction no longer moves the point: the chart ends here.
            return Err(TargetDoseError::ChartExtentExhausted {
                target_nats,
                extent: hi_s,
                max_observed_nats: observations.max_observed_nats,
            });
        }
        let (next_nats, next_applied) = observations.observe(probe.as_deref_mut(), &next_plan)?;
        if next_nats == target_nats {
            return Ok(finish(next_s, next_plan, next_applied, &observations));
        }
        if let Some(upper) = observations.certified_attainable_upper_nats
            && upper < target_nats
        {
            return Err(TargetDoseError::UnreachableTarget {
                target_nats,
                certified_attainable_upper_nats: upper,
            });
        }
        (lo_s, lo_nats, lo_plan, lo_applied) = (hi_s, hi_nats, hi_plan, hi_applied);
        (hi_s, hi_nats, hi_plan, hi_applied) = (next_s, next_nats, next_plan, next_applied);
    }

    // Resolve the bracket by false position with the Illinois weighting. Plain false
    // position keeps re-using a far endpoint whenever the root sits next to the
    // other one, so the bracket shrinks from one side only. Halving the residual of
    // an endpoint each time it is retained twice in a row pulls it in, which restores
    // superlinear convergence with the bracket still guaranteed. A dose that is not
    // smooth at the bracket's scale (a patched forward whose applied move is
    // quantized is a step function there) can still make even the weighted secant
    // creep, so once the last two observations have not halved the bracket the next
    // candidate is the midpoint. If a candidate at width `w_k` finds the bracket
    // wider than half of `w_{k-2}`, it bisects, giving `w_{k+1} ≤ w_{k-2}/2`: the
    // bracket halves at least once every three observations, and the loop reaches
    // the representation limit after finitely many observations with no budget.
    let mut lo_weight = 1.0_f64;
    let mut hi_weight = 1.0_f64;
    // `Some(true)`: the lower endpoint moved last. `Some(false)`: the upper one did.
    let mut lower_moved_last: Option<bool> = None;
    // Bracket widths before the previous observation and the one before it;
    // infinite until those observations exist.
    let mut width_one_back = f64::INFINITY;
    let mut width_two_back = f64::INFINITY;
    loop {
        let width = hi_s - lo_s;
        let lo_residual = lo_weight * (lo_nats - target_nats);
        let hi_residual = hi_weight * (hi_nats - target_nats);
        let secant = hi_s - hi_residual * (hi_s - lo_s) / (hi_residual - lo_residual);
        let candidate =
            if 2.0 * width <= width_two_back && secant.is_finite() && secant > lo_s && secant < hi_s
            {
                secant
            } else {
                0.5 * (lo_s + hi_s)
            };
        if !(candidate > lo_s && candidate < hi_s) {
            // The bracket is one representable step wide: return the endpoint whose
            // observed dose is closer to the target.
            let hi_closer = (hi_nats - target_nats).abs() <= (target_nats - lo_nats).abs();
            return Ok(if hi_closer {
                finish(hi_s, hi_plan, hi_applied, &observations)
            } else {
                finish(lo_s, lo_plan, lo_applied, &observations)
            });
        }
        let plan = plan_at(candidate)?;
        let (nats, applied) = observations.observe(probe.as_deref_mut(), &plan)?;
        if nats == target_nats {
            return Ok(finish(candidate, plan, applied, &observations));
        }
        (width_two_back, width_one_back) = (width_one_back, width);
        if nats < target_nats {
            (lo_s, lo_nats, lo_plan, lo_applied) = (candidate, nats, plan, applied);
            lo_weight = 1.0;
            if lower_moved_last == Some(true) {
                hi_weight *= 0.5;
            }
            lower_moved_last = Some(true);
        } else {
            (hi_s, hi_nats, hi_plan, hi_applied) = (candidate, nats, plan, applied);
            hi_weight = 1.0;
            if lower_moved_last == Some(false) {
                lo_weight *= 0.5;
            }
            lower_moved_last = Some(false);
        }
    }
}

/// Does this provenance carry behavioral (output-Fisher) information? Euclidean
/// is the isotropic activation-only path and carries none; the factored
/// provenances do. (Mirrors `atom_lens::metric_carries_behavior`.)
fn metric_carries_behavior(p: MetricProvenance) -> bool {
    match p {
        MetricProvenance::Euclidean | MetricProvenance::WhitenedStructured { .. } => false,
        MetricProvenance::OutputFisher { .. }
        | MetricProvenance::OutputFisherDownstream { .. }
        | MetricProvenance::BehavioralFisher { .. } => true,
    }
}

/// Evaluate the decoder output `g_k(t) = Φ_k(t) B_k ∈ ℝ^p` at an arbitrary
/// latent coordinate `t` (length `d`) via the atom's installed evaluator.
///
/// `scale` is the owning term's Tier-0 column scale (`term.tier0_scale()`):
/// under standardization/equilibration the fitted decoder lives in the
/// internal per-column frame `B_int[:,c] = B_raw[:,c]/σ_c`, while the row
/// metric `M = UUᵀ` is always built from raw activation-space probes. Every
/// steering quantity that meets the metric (chords, tangents, doses) must
/// therefore be mapped back to raw units, `g_raw[c] = σ_c·g_int[c]` (gh#2249
/// calibration confound; the Tier-0 MEAN cancels in every consumer here
/// because only chords/tangents are used, never absolute decodes).
fn decode_at(
    atom: &SaeManifoldAtom,
    t: &[f64],
    scale: Option<&Array1<f64>>,
) -> Result<Array1<f64>, String> {
    let d = t.len();
    let coords = Array2::from_shape_vec((1, d), t.to_vec())
        .map_err(|e| format!("steer_delta::decode_at: coord shape: {e}"))?;
    let mut out = atom.decode_at_coords(coords.view())?.row(0).to_owned();
    if let Some(scale) = scale {
        if scale.len() != out.len() {
            return Err(format!(
                "steer_delta::decode_at: tier0 scale length {} != output_dim {}",
                scale.len(),
                out.len()
            ));
        }
        for (v, &s) in out.iter_mut().zip(scale.iter()) {
            *v *= s;
        }
    }
    Ok(out)
}

/// Evaluate the decoder tangents `∂g_k/∂t_a = Φ_k'(t) B_k ∈ ℝ^p`, one per latent
/// axis `a ∈ 0..d`, at an arbitrary latent coordinate `t`. Returned as a
/// `(p × d)` matrix whose column `a` is the tangent along axis `a`.
fn decode_tangents_at(
    atom: &SaeManifoldAtom,
    t: &[f64],
    scale: Option<&Array1<f64>>,
) -> Result<Array2<f64>, String> {
    let evaluator = atom.basis_evaluator.as_ref().ok_or_else(|| {
        "steer_delta::decode_tangents_at: atom has no installed basis evaluator".to_string()
    })?;
    let p = atom.output_dim();
    let d = atom.latent_dim();
    let coords = Array2::from_shape_vec((1, d), t.to_vec())
        .map_err(|e| format!("steer_delta::decode_tangents_at: coord shape: {e}"))?;
    let jet = if atom.homotopy_eta == 1.0 {
        evaluator.evaluate(coords.view())?.1
    } else {
        evaluator
            .evaluate_phi_eta(coords.view(), atom.homotopy_eta)?
            .jet
    };
    let decoder = atom.decoder_coefficients();
    let m = decoder.nrows();
    if jet.dim() != (1, m, d) {
        return Err(format!(
            "steer_delta::decode_tangents_at: evaluator jet {:?} != (1, {m}, {d})",
            jet.dim()
        ));
    }
    let mut tang = Array2::<f64>::zeros((p, d));
    for axis in 0..d {
        for basis_col in 0..m {
            let dphi = jet[[0, basis_col, axis]];
            if dphi == 0.0 {
                continue;
            }
            for out_col in 0..p {
                tang[[out_col, axis]] += dphi * decoder[[basis_col, out_col]];
            }
        }
    }
    if let Some(scale) = scale {
        if scale.len() != p {
            return Err(format!(
                "steer_delta::decode_tangents_at: tier0 scale length {} != output_dim {p}",
                scale.len()
            ));
        }
        for (out_col, &s) in scale.iter().enumerate() {
            tang.row_mut(out_col).mapv_inplace(|v| v * s);
        }
    }
    Ok(tang)
}

/// Orthogonal projection of `δ` onto the span of the local tangents (columns of
/// `tangents`, shape `p × d`): `δ̂ = T V_r Λ_r⁻¹ V_rᵀ Tᵀ δ`, with `(Λ, V)` the
/// eigensystem of the `d × d` Gram `TᵀT`. The retained directions are those
/// whose eigenvalue exceeds the eigendecomposition's own backward-error floor
/// `d·ε·λ_max`; below it an eigenvalue carries no significant digits. The
/// projector is unique even for a rank-deficient frame, so no jitter biases the
/// projection, and a zero frame projects to zero.
fn project_onto_tangent_span(
    tangents: &Array2<f64>,
    delta: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    let p = tangents.nrows();
    let d = tangents.ncols();
    if d == 0 {
        return Ok(Array1::<f64>::zeros(p));
    }
    // Gram = TᵀT (d × d) and rhs = Tᵀδ (d).
    let mut gram = Array2::<f64>::zeros((d, d));
    let mut rhs = Array1::<f64>::zeros(d);
    for a in 0..d {
        let mut r = 0.0_f64;
        for i in 0..p {
            r += tangents[[i, a]] * delta[i];
        }
        rhs[a] = r;
        for b in a..d {
            let mut acc = 0.0_f64;
            for i in 0..p {
                acc += tangents[[i, a]] * tangents[[i, b]];
            }
            gram[[a, b]] = acc;
            gram[[b, a]] = acc;
        }
    }
    let (eigenvalues, eigenvectors) = gram.eigh(Side::Lower).map_err(|error| {
        format!("project_onto_tangent_span: tangent Gram eigendecomposition failed: {error:?}")
    })?;
    let lambda_max = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    let floor = (d as f64) * f64::EPSILON * lambda_max;
    let mut coeffs = Array1::<f64>::zeros(d);
    for mode in 0..d {
        let lambda = eigenvalues[mode];
        if lambda > floor {
            let mut dot = 0.0_f64;
            for a in 0..d {
                dot += eigenvectors[[a, mode]] * rhs[a];
            }
            let scale = dot / lambda;
            for a in 0..d {
                coeffs[a] += eigenvectors[[a, mode]] * scale;
            }
        }
    }
    let mut proj = Array1::<f64>::zeros(p);
    for i in 0..p {
        for a in 0..d {
            proj[i] += tangents[[i, a]] * coeffs[a];
        }
    }
    Ok(proj)
}

/// Norm of `δ`'s component orthogonal to the span of the local tangents:
/// `‖δ − δ̂‖` with `δ̂` the [`project_onto_tangent_span`] projection.
fn off_manifold_residual_norm(
    tangents: &Array2<f64>,
    delta: ArrayView1<'_, f64>,
) -> Result<f64, String> {
    let proj = project_onto_tangent_span(tangents, delta)?;
    let mut res_sq = 0.0_f64;
    for i in 0..delta.len() {
        let r = delta[i] - proj[i];
        res_sq += r * r;
    }
    Ok(res_sq.max(0.0).sqrt())
}

/// One dose sample on a collateral-damage curve (gam#2234 E2, the intrinsic
/// Rust-owned counterpart of the model-in-the-loop KL frontier): the on-target
/// effect and the off-target collateral of a single steering intervention,
/// measured in the fitted dictionary's own representation, with no LLM in the
/// loop.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct CollateralPoint {
    /// The canonical arc-length displacement applied to the target atom's chart,
    /// in the chart's own span units (a fraction of the ring for a circle chart).
    pub dose: f64,
    /// RMS-over-rows on-target effect: `‖proj_{T_k} Δ‖`, the energy the
    /// intervention deposits into the TARGET atom's own local decode-tangent
    /// frame `T_k = ∂g_k/∂t` at each row's fitted operating point. This is the
    /// intended landing — how loudly the knob turned the feature it names.
    pub on_target_effect: f64,
    /// RMS-over-rows collateral: `‖Δ − proj_{T_k} Δ‖`, the energy the SAME
    /// intervention deposits OUTSIDE the target atom's own local frame — the total
    /// damage. The on-manifold move is a chord of atom `k`'s decoder curve, so its
    /// off-target component is only the second-order sagitta (`≈ 0`, growing with
    /// dose-curvature); a fixed flat direction is off the rotating target frame at
    /// most rows, so its off-target energy is immediate. This is the direct
    /// generalization of the single-move [`SteerPlan::off_manifold_norm`] guard to
    /// a swept intervention.
    pub collateral: f64,
    /// RMS-over-rows CROSS-FEATURE leakage: `sqrt(Σ_{j∈others} ‖proj_{T_j} Δ‖²)`,
    /// the part of the move that lands on OTHER named atoms' frames — the
    /// interpretable "steering feature `k` spuriously moved feature `j`" damage, a
    /// component of the total `collateral`.
    pub cross_feature: f64,
}

/// One intervention family's swept collateral curve plus its aggregate
/// collateral efficiency (collateral energy spent per unit on-target effect).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct CollateralArm {
    /// Per-dose `(effect, collateral)` samples, in the order of the input doses.
    pub points: Vec<CollateralPoint>,
    /// Collateral energy per unit on-target effect over the swept doses:
    /// `sqrt(Σ collateral²) / sqrt(Σ effect²)`. Lower is a cleaner control knob.
    /// `NaN` when the arm achieves no on-target effect at any dose.
    pub efficiency: f64,
}

/// The on-manifold-vs-flat collateral-damage comparison for one target atom
/// (gam#2234 E2 thesis, measured intrinsically — no model surgery, no outer-fit
/// convergence in the loop). The two arms move the SAME per-row ambient energy:
/// the flat arm applies it along a single fixed decoder direction (the flat-SAE
/// `x' = x + α·w` baseline), the manifold arm applies the chart-coordinate group
/// action `x' = x + a·(Φ_k(t⊕δ) − Φ_k(t))·B_k`, which rotates with each row's
/// coordinate to stay on the atom's decoded image. The thesis: at matched
/// per-row norm the manifold arm spends strictly less collateral per unit
/// on-target effect — curved features are the right control knobs.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct CollateralCurve {
    /// The steered (target) atom.
    pub atom: usize,
    /// The target atom's latent axis the dose is applied along.
    pub axis: usize,
    /// The atoms collateral is measured against (typically every `j ≠ atom`).
    pub others: Vec<usize>,
    /// The on-manifold group-action arm.
    pub manifold: CollateralArm,
    /// The matched-per-row-norm fixed-direction (flat-SAE) control arm.
    pub flat: CollateralArm,
    /// `true` when the on-manifold arm spends strictly less collateral per unit
    /// on-target effect than the flat arm (`manifold.efficiency < flat.efficiency`),
    /// with both efficiencies finite — the E2 dominance verdict, decided
    /// structurally in the SAE's own representation.
    pub manifold_is_cleaner: bool,
}

/// Norm of `δ`'s component that lands inside the span of a local decode-tangent
/// frame — the energy the ambient move deposits into that atom's feature
/// direction at its current operating point.
fn frame_landed_norm(frame: &Array2<f64>, delta: ArrayView1<'_, f64>) -> Result<f64, String> {
    let proj = project_onto_tangent_span(frame, delta)?;
    Ok(proj.iter().map(|&x| x * x).sum::<f64>().sqrt())
}

/// Sweep the intrinsic collateral-damage curve for steering atom `atom_k` along
/// latent `axis` over `doses`, comparing the on-manifold group action against a
/// matched-per-row-norm flat-direction control (gam#2234 E2).
///
/// For each dose `δ` the on-manifold ambient move advances the atom's
/// canonical, unit-speed chart coordinate by `δ` (gate held fixed). The raw
/// fitted parameter is gauge-arbitrary and is deliberately not exposed as the
/// dose axis. The flat control replays the SAME per-row move NORM along one fixed
/// ambient direction `w` — the atom's mean decode-tangent direction along `axis`,
/// the manifold analog of a flat SAE's single decoder column. Each move is
/// decomposed against every atom's local decode-tangent frame at its fitted
/// coordinate: the projection onto the TARGET atom's frame is the on-target
/// effect, the projection onto the OTHER atoms' frames is the collateral.
///
/// This is a pure read over the fitted term (no criterion, no penalty, no outer
/// fit), so it runs on any fitted or hand-built term with installed evaluators —
/// it does not wait on outer-loop convergence, unlike the model-in-the-loop E1/E2
/// KL frontier it mirrors.
///
/// Errors when `atom_k`/`axis`/an `others` index is out of range, `doses` is
/// empty, or the target atom's mean tangent along `axis` vanishes (no fixed
/// direction to define the flat control against).
pub fn collateral_curve(
    model: &SaeManifoldTerm,
    atom_k: usize,
    axis: usize,
    others: &[usize],
    doses: &[f64],
) -> Result<CollateralCurve, String> {
    let k = model.k_atoms();
    if atom_k >= k {
        return Err(format!(
            "collateral_curve: atom index {atom_k} out of range (term has {k} atoms)"
        ));
    }
    let d_k = model.atoms[atom_k].latent_dim();
    if axis >= d_k {
        return Err(format!(
            "collateral_curve: axis {axis} out of range for atom {atom_k} latent_dim {d_k}"
        ));
    }
    if d_k != 1 || axis != 0 {
        return Err(format!(
            "collateral_curve: intrinsic doses require a one-dimensional canonical chart; \
             atom {atom_k} has latent_dim {d_k} and axis {axis}"
        ));
    }
    if doses.is_empty() {
        return Err("collateral_curve: doses must be non-empty".to_string());
    }
    for &j in others {
        if j >= k {
            return Err(format!(
                "collateral_curve: other atom index {j} out of range (term has {k} atoms)"
            ));
        }
    }
    let n = model.n_obs();
    let p = model.output_dim();
    let rows: Vec<usize> = (0..n).collect();

    // Per-row local decode-tangent frames at each atom's fitted operating point.
    // These are dose-independent (the fitted coordinates never move; steering is
    // the hypothetical move whose leakage we price against the CURRENT features).
    let frame_at = |atom_idx: usize| -> Result<Vec<Array2<f64>>, String> {
        let coords = model.assignment.coords[atom_idx].as_matrix();
        let mut frames = Vec::with_capacity(n);
        for row in 0..n {
            let t: Vec<f64> = coords.row(row).to_vec();
            frames.push(decode_tangents_at(
                &model.atoms[atom_idx],
                &t,
                model.tier0_scale(),
            )?);
        }
        Ok(frames)
    };
    let target_frames = frame_at(atom_k)?;
    let mut other_frames: Vec<Vec<Array2<f64>>> = Vec::with_capacity(others.len());
    for &j in others {
        other_frames.push(frame_at(j)?);
    }

    // The fixed flat direction w: the dominant ambient direction the target atom
    // moves along `axis` — the top left singular vector of its per-row tangent
    // field, i.e. the leading eigenvector of `G = Σ_i g_i g_iᵀ` with
    // `g_i = ∂g_k/∂t_axis|_{t_i}`. This is the single best fixed decoder column a
    // flat SAE would steer this feature with (the mean tangent is not usable — it
    // averages to ≈0 over a full circle). Read off the exact symmetric
    // eigendecomposition of the `p × p` Gram rather than an iteration whose step
    // count would decide how close to leading the direction is. A repeated
    // leading eigenvalue (a round ring's tangent field is isotropic in its plane)
    // defines the direction only up to that eigenspace; every unit vector of it
    // maximizes the same `Σ_i (wᵀ g_i)²`, and the decomposition's basis vector is
    // used.
    let mut gram = Array2::<f64>::zeros((p, p));
    for frame in &target_frames {
        for i in 0..p {
            let gi = frame[[i, axis]];
            if gi == 0.0 {
                continue;
            }
            for j in 0..p {
                gram[[i, j]] += gi * frame[[j, axis]];
            }
        }
    }
    let (eigenvalues, eigenvectors) = gram.eigh(Side::Lower).map_err(|error| {
        format!("collateral_curve: tangent-field Gram eigendecomposition failed: {error:?}")
    })?;
    let leading = (0..eigenvalues.len())
        .max_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]))
        .filter(|&index| eigenvalues[index] > 0.0)
        .ok_or_else(|| {
            format!(
                "collateral_curve: atom {atom_k} has a vanishing tangent field along axis {axis}; \
                 no fixed direction to define the flat control"
            )
        })?;
    let w = eigenvectors.column(leading).to_owned();

    // Decompose one per-row move field into (effect, off-target collateral,
    // cross-feature leakage) RMS over rows.
    let decompose = |field: &Array2<f64>| -> Result<CollateralPoint, String> {
        let mut eff_sq = 0.0_f64;
        let mut col_sq = 0.0_f64;
        let mut cross_sq = 0.0_f64;
        for row in 0..n {
            let delta = field.row(row);
            let on_target = project_onto_tangent_span(&target_frames[row], delta)?;
            let mut e = 0.0_f64;
            let mut c = 0.0_f64;
            for i in 0..p {
                e += on_target[i] * on_target[i];
                let residual = delta[i] - on_target[i];
                c += residual * residual;
            }
            eff_sq += e;
            col_sq += c;
            let mut cross = 0.0_f64;
            for frames in &other_frames {
                let l = frame_landed_norm(&frames[row], delta)?;
                cross += l * l;
            }
            cross_sq += cross;
        }
        let denom = n.max(1) as f64;
        Ok(CollateralPoint {
            dose: 0.0,
            on_target_effect: (eff_sq / denom).sqrt(),
            collateral: (col_sq / denom).sqrt(),
            cross_feature: (cross_sq / denom).sqrt(),
        })
    };

    let mut manifold_pts = Vec::with_capacity(doses.len());
    let mut flat_pts = Vec::with_capacity(doses.len());
    for &dose in doses {
        let on_field = model.steer_rows(atom_k, &rows, Array1::from_elem(1, dose).view())?;

        // Matched control: same per-row move NORM, along the fixed direction w.
        let mut flat_field = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let norm = on_field.row(row).iter().map(|&x| x * x).sum::<f64>().sqrt();
            for i in 0..p {
                flat_field[[row, i]] = norm * w[i];
            }
        }

        let mut m = decompose(&on_field)?;
        m.dose = dose;
        manifold_pts.push(m);
        let mut f = decompose(&flat_field)?;
        f.dose = dose;
        flat_pts.push(f);
    }

    let efficiency = |pts: &[CollateralPoint]| -> f64 {
        let eff_sq: f64 = pts
            .iter()
            .map(|q| q.on_target_effect * q.on_target_effect)
            .sum();
        let col_sq: f64 = pts.iter().map(|q| q.collateral * q.collateral).sum();
        if eff_sq > 0.0 {
            (col_sq / eff_sq).sqrt()
        } else {
            f64::NAN
        }
    };
    let manifold = CollateralArm {
        efficiency: efficiency(&manifold_pts),
        points: manifold_pts,
    };
    let flat = CollateralArm {
        efficiency: efficiency(&flat_pts),
        points: flat_pts,
    };
    let manifold_is_cleaner = manifold.efficiency.is_finite()
        && flat.efficiency.is_finite()
        && manifold.efficiency < flat.efficiency;

    Ok(CollateralCurve {
        atom: atom_k,
        axis,
        others: others.to_vec(),
        manifold,
        flat,
        manifold_is_cleaner,
    })
}

// ───────────────────────────────────────────────────────────────────────────
// #2234 / #2263 item 3 — steering in the CANONICAL (unit-speed) chart.
//
// `SaeManifoldTerm::steer_rows` is the group action `t ⊕ δ` on the FITTED chart
// parameter. That parameter is a gauge choice: the fit lands on one point of the
// `Diff(S¹)` / `Diff([0,1])` orbit, and `chart_arclength_coordinates` states the
// consequence in its own doc — *"any downstream consumer reading angle/dose/
// adjacency off the raw `t_i` is reading a gauge-ARBITRARY quantity"*. A dose
// expressed in the raw parameter is therefore not comparable between rows, between
// atoms, or between two fits of the same data.
//
// #2022's in-loop unit-speed retraction exists to close that gap by re-gauging the
// fitted chart to arc length, which would make the raw parameter the intrinsic one.
// It is allowed to refuse: `unit_speed_reparameterization` returns `Ok(None)`
// whenever the basis family cannot re-express the reparameterized curve inside
// `CHART_RECOMPOSITION_REL_TOL`, and a finite harmonic basis generically cannot
// carry the arc-length reparameterization of a non-round ring. So the meaning of a
// raw `δ` depends on whether an internal gate passed, and the caller is not told
// which.
//
// The surfaces below remove the dependence. They express a displacement in the
// atom's CANONICAL coordinate — arc length along the fitted decoder curve,
// rescaled onto the chart's own span, i.e. exactly the coordinate
// `unit_speed_reparameterization` would have installed had it been able to. The
// conversion factor between the canonical step and the raw step is the chart's own
// speed field `‖∂g/∂t‖`; nothing here introduces a scale that is not read off the
// fitted chart. On a chart that already IS unit-speed the canonical step and the
// raw step are the same number, so this is a repair of the existing gauge rather
// than a second one.
// ───────────────────────────────────────────────────────────────────────────

/// One `d = 1` atom's canonical-chart context, gathered once so a batched
/// inversion pays for the chart's arc-length quadrature a fixed number of times
/// instead of once per row.
struct CanonicalChart<'a> {
    atom: usize,
    name: &'a str,
    evaluator: &'a dyn SaeBasisEvaluator,
    decoder: &'a Array2<f64>,
    topology: CanonicalChartTopology,
    /// The atom's FITTED coordinates. An interval chart's canonical domain is
    /// derived from the coordinates handed to the arc-length map, so every query
    /// carries the fitted set as a prefix and the domain never moves out from
    /// under the map. A circle chart's domain is `[0, period)` regardless, and
    /// the prefix is omitted.
    domain_anchor: Option<Array1<f64>>,
    /// Length of the canonical coordinate's domain: `period` for a circle chart,
    /// `1` for an interval chart — the span `unit_speed_reparameterization` maps
    /// onto, so a canonical displacement and a raw displacement are the same
    /// number on an already-canonical chart.
    span: f64,
    /// Raw-parameter domain the canonical map is inverted inside.
    lo: f64,
    hi: f64,
    /// The axis period the atom's own `LatentManifold` wraps at — the period
    /// [`SaeManifoldTerm::steer_rows`]'s group action uses. Checked against the
    /// canonical topology at construction, and used to take each row's raw step
    /// as its SHORTEST representative.
    axis_period: Option<f64>,
}

/// The canonical (arc-length) reading of one `d = 1` chart, plus the gauge
/// diagnostics that say whether the raw chart could have been read instead.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CanonicalChartCoordinates {
    /// The atom whose chart was read.
    pub atom: usize,
    /// Canonical span (`period` for a circle chart, `1` for an interval chart).
    pub span: f64,
    /// Canonical coordinate `span · s(t)/L` of each queried raw coordinate — the
    /// gauge-invariant position #2081 says every angle/dose/adjacency consumer
    /// should read in place of the raw `t`.
    pub canonical: Vec<f64>,
    /// Speed coefficient of variation `stddev(‖γ'‖)/mean(‖γ'‖)` of the fitted
    /// decoder curve. `0` ⟺ the raw chart already IS the canonical chart, so a
    /// raw-parameter steer and a canonical steer coincide; a positive value is
    /// exactly the gauge error a raw-parameter dose commits.
    pub speed_cv: f64,
    /// `min ‖γ'‖ / mean ‖γ'‖`. Approaching `0` means the chart nearly collapses
    /// somewhere, so the canonical map stops being a diffeomorphism and the raw
    /// coordinate returned by the inverse is a point of a preimage SET.
    pub min_speed_over_mean: f64,
    /// `max ‖γ'‖ / mean ‖γ'‖`.
    pub max_speed_over_mean: f64,
    /// Total arc length of the fitted decoder curve over the canonical domain —
    /// the conversion from a canonical (span) displacement to an absolute one.
    pub total_arc_length: f64,
}

fn canonical_chart<'a>(
    model: &'a SaeManifoldTerm,
    atom_k: usize,
) -> Result<CanonicalChart<'a>, String> {
    let k = model.k_atoms();
    if atom_k >= k {
        return Err(format!(
            "canonical chart: atom index {atom_k} out of range (term has {k} atoms)"
        ));
    }
    let atom = &model.atoms[atom_k];
    let topology = model.d1_unit_speed_topology(atom_k).ok_or_else(|| {
        format!(
            "canonical chart: atom {atom_k} ('{}') has no d=1 canonical chart. Arc-length \
             steering needs a one-dimensional chart with an installed basis evaluator and no \
             active curvature homotopy; this atom is latent_dim {} with basis {:?} \
             (homotopy_eta {})",
            atom.name,
            atom.latent_dim(),
            atom.basis_kind(),
            atom.homotopy_eta
        )
    })?;
    let evaluator = atom.basis_evaluator.as_deref().ok_or_else(|| {
        format!(
            "canonical chart: atom {atom_k} ('{}') has no installed basis evaluator",
            atom.name
        )
    })?;
    let fitted = model.assignment.coords[atom_k]
        .as_matrix()
        .column(0)
        .to_owned();
    let axis_period = *model.assignment.coords[atom_k]
        .effective_axis_periods()
        .first()
        .ok_or_else(|| {
            format!(
                "canonical chart: atom {atom_k} ('{}') has no latent axis",
                atom.name
            )
        })?;
    let (span, lo, hi, domain_anchor) = match &topology {
        CanonicalChartTopology::Circle { period } => (*period, 0.0, *period, None),
        CanonicalChartTopology::Interval => {
            let mut t_min = f64::INFINITY;
            let mut t_max = f64::NEG_INFINITY;
            for &t in fitted.iter() {
                t_min = t_min.min(t);
                t_max = t_max.max(t);
            }
            (1.0, t_min, t_max, Some(fitted))
        }
    };
    if !(span.is_finite() && span > 0.0 && lo.is_finite() && hi.is_finite() && hi > lo) {
        return Err(format!(
            "canonical chart: atom {atom_k} ('{}') has a degenerate canonical domain \
             [{lo}, {hi}] with span {span}",
            atom.name
        ));
    }
    // The canonical chart is defined on the domain the arc-length map integrates,
    // and the move is realized by the group action, which wraps at the MANIFOLD's
    // own axis period. If those two disagree the surface would silently steer to a
    // different point than the one it reports; refuse instead.
    let periods_agree = match (&topology, axis_period) {
        (CanonicalChartTopology::Circle { period }, Some(axis)) => axis == *period,
        (CanonicalChartTopology::Interval, None) => true,
        _ => false,
    };
    if !periods_agree {
        return Err(format!(
            "canonical chart: atom {atom_k} ('{}') has canonical topology {topology:?} but its \
             manifold wraps its axis at {axis_period:?}; the arc-length domain and the group \
             action's period must be the same object",
            atom.name
        ));
    }
    Ok(CanonicalChart {
        atom: atom_k,
        name: &atom.name,
        evaluator,
        decoder: atom.decoder_coefficients(),
        topology,
        domain_anchor,
        span,
        lo,
        hi,
        axis_period,
    })
}

impl CanonicalChart<'_> {
    /// Canonical coordinate of each queried raw coordinate, plus the chart's
    /// speed diagnostics, in ONE arc-length quadrature for the whole batch.
    fn read(&self, query: &[f64]) -> Result<CanonicalChartCoordinates, String> {
        let prefix = self.domain_anchor.as_ref().map_or(0, Array1::len);
        let mut all = Array1::<f64>::zeros(prefix + query.len());
        if let Some(anchor) = self.domain_anchor.as_ref() {
            for (i, &t) in anchor.iter().enumerate() {
                all[i] = t;
            }
        }
        for (i, &t) in query.iter().enumerate() {
            all[prefix + i] = t;
        }
        let reading = chart_arclength_coordinates(
            self.evaluator,
            self.decoder.view(),
            all.view(),
            &self.topology,
        )?
        .ok_or_else(|| {
            format!(
                "canonical chart: atom {} ('{}') has a degenerate arc-length chart \
                 (empty basis/decoder, a collapsed domain, or a non-finite / zero total \
                 arc length); there is no canonical coordinate to steer in",
                self.atom, self.name
            )
        })?;
        Ok(CanonicalChartCoordinates {
            atom: self.atom,
            span: self.span,
            canonical: (0..query.len())
                .map(|i| self.span * reading.coords_u_arc[prefix + i])
                .collect(),
            speed_cv: reading.speed_cv,
            min_speed_over_mean: reading.min_speed_over_mean,
            max_speed_over_mean: reading.max_speed_over_mean,
            total_arc_length: reading.total_arc_length,
        })
    }

    /// Raw coordinates whose canonical coordinate is `targets`.
    ///
    /// `s(t) = ∫‖γ'‖` is non-decreasing in the raw parameter by construction, so
    /// the canonical map is inverted by bisection on the chart's own domain,
    /// refined until the bracket is one `f64` ulp of the span — the coordinate's
    /// representation limit, not a tolerance knob. Every target's bracket is
    /// halved inside ONE batched chart read, so the cost is `log2(1/ε)` reads
    /// however many rows are steered. A chart with a zero-speed cell has a flat
    /// `s`, and the returned coordinate is then a point of the preimage rather
    /// than the preimage — [`CanonicalChartCoordinates::min_speed_over_mean`]
    /// is the reader that says when that is happening.
    fn invert(&self, targets: &[f64]) -> Result<Vec<f64>, String> {
        let q = targets.len();
        let mut lo = vec![self.lo; q];
        let mut hi = vec![self.hi; q];
        let resolution = f64::EPSILON * self.span;
        loop {
            let mut mids = vec![0.0_f64; q];
            let mut live: Vec<usize> = Vec::new();
            for i in 0..q {
                let mid = 0.5 * (lo[i] + hi[i]);
                mids[i] = mid;
                if hi[i] - lo[i] > resolution && mid > lo[i] && mid < hi[i] {
                    live.push(i);
                }
            }
            if live.is_empty() {
                break;
            }
            let probe: Vec<f64> = live.iter().map(|&i| mids[i]).collect();
            let read = self.read(&probe)?;
            for (slot, &i) in live.iter().enumerate() {
                if read.canonical[slot] <= targets[i] {
                    lo[i] = mids[i];
                } else {
                    hi[i] = mids[i];
                }
            }
        }
        Ok((0..q).map(|i| 0.5 * (lo[i] + hi[i])).collect())
    }

    /// Move a canonical coordinate by `delta` inside the chart's own topology:
    /// a circle wraps, a bounded interval patch clamps at its ends.
    fn advance(&self, base: f64, delta: f64) -> f64 {
        match self.topology {
            CanonicalChartTopology::Circle { .. } => (base + delta).rem_euclid(self.span),
            CanonicalChartTopology::Interval => (base + delta).clamp(0.0, self.span),
        }
    }
}

/// The fitted-chart step that moves each of `rows` of atom `atom_k` by
/// `canonical_delta` in the atom's CANONICAL (unit-speed / arc-length) chart.
/// This is the step [`SaeManifoldTerm::steer_rows`] and
/// [`SaeManifoldTerm::steer_decode`] hand to the group action for a
/// one-dimensional atom, which is what puts the gam#2234 intervention primitive
/// in the units gam#2234 claims for it.
///
/// The requested displacement is applied to each row's canonical coordinate
/// through the chart's own topology (a circle wraps, a bounded patch clamps), and
/// the resulting canonical position is mapped back to the raw fitted parameter.
/// This selects the STEP; it does not re-implement the action. The step therefore
/// differs from row to row exactly as the fitted chart's speed does, a chart that
/// is already unit-speed yields a constant step equal to `canonical_delta`, and
/// the spread over rows is the gauge error a single raw step would have committed.
///
/// A displacement that is an exact multiple of the canonical span (`0`, one full
/// period) is short-circuited to an exactly-zero step, so `δ = 0` is a bit-exact
/// no-op and a circle closes exactly.
///
/// Errors when the atom has no `d = 1` canonical chart, when the chart is
/// degenerate, when a row index is out of range, or when `canonical_delta` is not
/// finite.
pub(crate) fn canonical_chart_steps(
    model: &SaeManifoldTerm,
    atom_k: usize,
    rows: &[usize],
    canonical_delta: f64,
) -> Result<Vec<f64>, String> {
    if !canonical_delta.is_finite() {
        return Err(format!(
            "SaeManifoldTerm::steer: the canonical displacement must be finite, got \
             {canonical_delta}"
        ));
    }
    let chart = canonical_chart(model, atom_k)?;
    let n = model.n_obs();
    let mut base_raw = Vec::with_capacity(rows.len());
    let fitted = model.assignment.coords[atom_k].as_matrix();
    for &row in rows {
        if row >= n {
            return Err(format!(
                "SaeManifoldTerm::steer: row {row} out of range (n = {n})"
            ));
        }
        base_raw.push(fitted[[row, 0]]);
    }
    let base_canonical = chart.read(&base_raw)?.canonical;

    // A whole number of canonical spans is the identity of the group action on a
    // circle chart; take it exactly rather than through the inverse, so `δ = 0`
    // and `δ = period` keep the bit-exact behaviour the raw action has.
    let residual = match chart.topology {
        CanonicalChartTopology::Circle { .. } => canonical_delta.rem_euclid(chart.span),
        CanonicalChartTopology::Interval => canonical_delta,
    };
    if residual == 0.0 {
        return Ok(vec![0.0_f64; rows.len()]);
    }
    let targets: Vec<f64> = base_canonical
        .iter()
        .map(|&c| chart.advance(c, residual))
        .collect();
    let moved = chart.invert(&targets)?;
    // Each step is its SHORTEST representative, through the same helper
    // `steer_delta` uses. Both endpoints lie in the chart's own domain, so a row
    // near the wrap seam would otherwise take the long way round (`0.95 → 0.06` as
    // `−0.89` rather than `+0.11`). `retract` wraps, so the moved point is the same
    // either way, but the spread of the steps over rows is only the gauge error if
    // each step is the short one.
    let periods = [chart.axis_period];
    let mut steps = Vec::with_capacity(moved.len());
    for (&to, &from) in moved.iter().zip(base_raw.iter()) {
        steps.push(shortest_coordinate_delta(&[from], &[to], &periods)?[0]);
    }
    Ok(steps)
}

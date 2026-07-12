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
//! # Validity radius — where local linearization stops being trusted
//!
//! A consumer must know *how far* the move can be trusted as a linear push. The
//! **validity radius** is the latent step size at which the exact chord dose
//! diverges from the initial-tangent quadratic prediction by more than
//! [`VALIDITY_DIVERGENCE_FRACTION`]. Beyond it the surface has curved enough that
//! the endpoint chord no longer represents the move. We **report** it; we do not
//! silently clip to it.
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
//! [`SaeBasisEvaluator`]) and the criterion-facing
//! [`RowMetric::fisher_mass`] / [`RowMetric::pullback`]. It never mutates the
//! model, never touches a likelihood / criterion / penalty, and the solver floor
//! `δ` of [`RowMetric`] never enters any number it reports (the fisher-mass /
//! pullback face is `δ`-free, #747).

use ndarray::{Array1, Array2, ArrayView1};

use crate::encode::EncodeAtlas;
use crate::manifold::{SaeManifoldAtom, SaeManifoldTerm};
use gam_problem::{MetricProvenance, RowMetric};

/// Number of sub-steps the latent path `[t_from, t_to]` is integrated over for
/// the dosimetry path integral. The decoder curve is smooth, so a modest
/// midpoint-rule grid resolves the arc; fixed (no clock / no adaptivity) so the
/// reported dose is deterministic.
const STEER_VALIDITY_STEPS: usize = 64;

/// The fraction by which the exact chord dose may diverge from the
/// initial-tangent quadratic prediction before the move is declared past its
/// validity radius.
const VALIDITY_DIVERGENCE_FRACTION: f64 = 0.1;

/// Scientific status of the quadratic dose relative to the full output Fisher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FisherDoseKind {
    /// Euclidean/no-behavior metric: no nats dose exists.
    Unavailable,
    /// A behavioral factor metric exists, but no omitted-mass audit was supplied.
    UnauditedFactorMetric,
    /// The harvest reports zero omitted Fisher trace.
    FullMass,
    /// A non-negative Fisher tail was omitted; the local quadratic is a lower
    /// bound on the full-Fisher local KL.
    TruncatedLowerBound,
}

impl FisherDoseKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unavailable => "unavailable",
            Self::UnauditedFactorMetric => "unaudited_factor_metric",
            Self::FullMass => "full_mass",
            Self::TruncatedLowerBound => "truncated_lower_bound",
        }
    }
}

/// The actionable output of a steering query over one atom.
#[derive(Clone, Debug, PartialEq)]
pub struct SteerPlan {
    /// Which atom was steered (index into [`SaeManifoldTerm::atoms`]).
    pub atom: usize,
    /// The atom's name (mirrors [`crate::manifold::SaeManifoldAtom::name`]).
    pub atom_name: String,
    /// The source latent coordinate `t_from` (length = atom's `latent_dim`).
    pub t_from: Vec<f64>,
    /// The target latent coordinate `t_to` (length = atom's `latent_dim`).
    pub t_to: Vec<f64>,
    /// The exact amplitude `a` the caller applied to the on-manifold move.
    pub amplitude: f64,
    /// The exact row whose output-Fisher metric prices the applied move.
    pub metric_row: usize,
    /// **The activation-space delta**: `δ = a · (g_k(t_to) − g_k(t_from))`, a
    /// length-`p` vector in the reconstruction/output space — the actual move to
    /// add to a hidden state.
    pub delta: Array1<f64>,
    /// **DOSIMETRY**: predicted output effect of the exact applied move in
    /// **nats** of KL, `0.5 * delta^T M_metric_row delta`.
    /// `None` when the metric carries no behavioral information (Euclidean
    /// provenance) — the dose is *not available*, not zero.
    pub predicted_nats: Option<f64>,
    /// Whether `predicted_nats` is full-mass, unaudited, or a low-rank lower
    /// bound. This prevents consumers from treating every factor rank as an
    /// equally calibrated full-Fisher dose.
    pub predicted_nats_kind: FisherDoseKind,
    /// Captured trace `tr(U_n U_n^T)` at `metric_row`, when behavior is present.
    pub fisher_mass_captured: Option<f64>,
    /// Non-negative omitted Fisher trace supplied by the harvest, when audited.
    pub fisher_mass_residual: Option<f64>,
    /// `residual / (captured + residual)`, when audited.
    pub fisher_mass_residual_fraction: Option<f64>,
    /// **VALIDITY RADIUS**: the latent step size (Euclidean norm of the move from
    /// `t_from`) at which the exact chord dose first diverges from the
    /// initial-tangent quadratic prediction by more than
    /// [`VALIDITY_DIVERGENCE_FRACTION`]. Equals the full move length when the
    /// linearization is trusted all the way to `t_to`. `None` under a no-behavior
    /// metric (there is no dose to validate).
    pub validity_radius: Option<f64>,
    /// **OFF-MANIFOLD GUARD**: the norm of `δ`'s component outside the span of
    /// the atom's local decoder tangents `∂g_k/∂t` at `t_from`. `≈ 0` by
    /// construction (the move is a chord of the curve); a large value flags a
    /// move that left the learned surface.
    pub off_manifold_norm: f64,
    /// The provenance of the metric the dose was read through, echoed so a
    /// consumer can certify *why* `predicted_nats` is `None` when it is.
    pub metric_provenance: MetricProvenance,
}

/// Result of writing one certified chart coordinate into an activation row.
///
/// The edited row is always `x + δ`, where `δ` is the delta returned by
/// [`steer_delta`] for the atom's current encoded coordinate and the requested
/// target coordinate. Because only the on-manifold atom chord is added, every
/// component of `x` outside this atom's chart residual is preserved exactly; this
/// is the locality guarantee missing from whole-residual linear-steering
/// baselines.
#[derive(Clone, Debug)]
pub struct CoordinateSetResult {
    /// The edited activation/reconstruction row.
    pub edited: Array1<f64>,
    /// Certified coordinate read from the input row before the write.
    pub t_from_certified: Array1<f64>,
    /// Certificate attached to `t_from_certified`.
    pub encode_certificate: crate::encode::RowCertificate,
    /// Steering plan whose `delta` was added to the row.
    pub steer: SteerPlan,
}

/// Write atom `atom_k`'s chart coordinate in row `x` to `t_to` by delta
/// steering, preserving the row's off-atom/off-subspace residual exactly.
///
/// `amplitude` is the assignment/intensity with which the row expresses this
/// atom; callers that have already separated existence/intensity/position should
/// pass the intensity and only swap the position coordinate. The certified read
/// uses [`EncodeAtlas::certified_encode_row`]; the write uses [`steer_delta`].
pub fn set_coordinate(
    model: &SaeManifoldTerm,
    metric: &RowMetric,
    atlas: &EncodeAtlas,
    x: ArrayView1<'_, f64>,
    atom_k: usize,
    metric_row: usize,
    amplitude: f64,
    t_to: &[f64],
) -> Result<CoordinateSetResult, String> {
    let atom = model.atoms.get(atom_k).ok_or_else(|| {
        format!(
            "set_coordinate: atom index {atom_k} out of range (term has {} atoms)",
            model.k_atoms()
        )
    })?;
    if x.len() != atom.output_dim() {
        return Err(format!(
            "set_coordinate: input row has length {} but atom {atom_k} output_dim is {}",
            x.len(),
            atom.output_dim()
        ));
    }
    let (t_from, cert) = atlas.certified_encode_row(atom, atom_k, x, amplitude)?;
    let steer = steer_delta(
        model,
        metric,
        atom_k,
        metric_row,
        amplitude,
        t_from.as_slice().unwrap_or(&[]),
        t_to,
    )?;
    let mut edited = x.to_owned();
    if edited.len() != steer.delta.len() {
        return Err(format!(
            "set_coordinate: steering delta length {} does not match row length {}",
            steer.delta.len(),
            edited.len()
        ));
    }
    for i in 0..edited.len() {
        edited[i] += steer.delta[i];
    }
    Ok(CoordinateSetResult {
        edited,
        t_from_certified: t_from,
        encode_certificate: cert,
        steer,
    })
}

/// Result of a coordinate interchange: donor position read from `x_source`, then
/// written into `x_target` while preserving the target residual and intensity.
#[derive(Clone, Debug)]
pub struct InterchangeResult {
    /// Target row after the donor coordinate has been delta-written into it.
    pub edited_target: Array1<f64>,
    /// Donor/source coordinate that was transplanted.
    pub donor_t: Array1<f64>,
    /// Target coordinate before the transplant.
    pub target_t_before: Array1<f64>,
    /// Target behavior coordinate after the transplant, re-read from the edit.
    pub target_t_after: Array1<f64>,
    /// Steering dose in nats, when a behavioral metric is available.
    pub predicted_nats: Option<f64>,
    /// Norm of the steering delta outside the local atom tangent frame.
    pub off_manifold_norm: f64,
    /// Reported steering validity radius.
    pub validity_radius: Option<f64>,
    /// Geodesic chart-coordinate landing error. Wrapped axes use their shortest
    /// signed displacement. This is a descriptive reconstruction diagnostic,
    /// not a p/e-value: no counterfactual null distribution is available here.
    pub landing_error: f64,
    /// Underlying coordinate-write plan.
    pub set_result: CoordinateSetResult,
}

/// Interchange atom `atom_k`'s chart coordinate from `x_source` into `x_target`.
///
/// The source coordinate is certified with `source_amplitude`; the target write
/// is performed with `target_amplitude`, so swapping a position coordinate cannot
/// silently smuggle donor intensity into the target. The returned landing error
/// is descriptive; statistical evidence requires an externally specified null
/// experiment and is deliberately not fabricated from the error magnitude.
pub fn interchange(
    model: &SaeManifoldTerm,
    metric: &RowMetric,
    atlas: &EncodeAtlas,
    x_target: ArrayView1<'_, f64>,
    target_amplitude: f64,
    x_source: ArrayView1<'_, f64>,
    source_amplitude: f64,
    atom_k: usize,
    target_metric_row: usize,
) -> Result<InterchangeResult, String> {
    let atom = model.atoms.get(atom_k).ok_or_else(|| {
        format!(
            "interchange: atom index {atom_k} out of range (term has {} atoms)",
            model.k_atoms()
        )
    })?;
    let (donor_t, _donor_cert) =
        atlas.certified_encode_row(atom, atom_k, x_source, source_amplitude)?;
    let set = set_coordinate(
        model,
        metric,
        atlas,
        x_target,
        atom_k,
        target_metric_row,
        target_amplitude,
        donor_t.as_slice().unwrap_or(&[]),
    )?;
    let (target_t_after, _after_cert) =
        atlas.certified_encode_row(atom, atom_k, set.edited.view(), target_amplitude)?;
    let periods = model.assignment.coords[atom_k].effective_axis_periods();
    let landing_error = coordinate_l2_distance(
        donor_t.as_slice().unwrap_or(&[]),
        target_t_after.as_slice().unwrap_or(&[]),
        &periods,
    )?;
    Ok(InterchangeResult {
        edited_target: set.edited.clone(),
        donor_t,
        target_t_before: set.t_from_certified.clone(),
        target_t_after,
        predicted_nats: set.steer.predicted_nats,
        off_manifold_norm: set.steer.off_manifold_norm,
        validity_radius: set.steer.validity_radius,
        landing_error,
        set_result: set,
    })
}

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

fn coordinate_l2_distance(a: &[f64], b: &[f64], periods: &[Option<f64>]) -> Result<f64, String> {
    Ok(shortest_coordinate_delta(a, b, periods)?
        .iter()
        .map(|d| d * d)
        .sum::<f64>()
        .sqrt())
}

fn path_coordinate(
    from: &[f64],
    delta: &[f64],
    periods: &[Option<f64>],
    fraction: f64,
) -> Vec<f64> {
    from.iter()
        .zip(delta.iter())
        .zip(periods.iter())
        .map(|((&start, &step), &period)| {
            let value = start + fraction * step;
            period.map_or(value, |p| value.rem_euclid(p))
        })
        .collect()
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
/// `predicted_nats` / `validity_radius` degrade to `None`.
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
    let d = atom.latent_dim;
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
        match fisher_mass_residual {
            None => FisherDoseKind::UnauditedFactorMetric,
            Some(0.0) => FisherDoseKind::FullMass,
            Some(_) => FisherDoseKind::TruncatedLowerBound,
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
    let off_manifold_norm = off_manifold_residual_norm(&tangents, delta.view());

    // --- dosimetry: exact applied-delta Fisher endpoint KL ------------------
    let (predicted_nats, validity_radius) = if !behavior_available {
        (None, None)
    } else {
        let ctx = SteerContext {
            atom,
            scale: tier0_scale,
            metric,
            row: metric_row,
            p,
            d,
            amplitude,
            coordinate_delta: &coordinate_delta,
            periods: &periods,
        };
        let dose = 0.5 * metric.fisher_mass(metric_row, delta.view());
        let radius = validity_radius(&ctx, t_from)?;
        (Some(dose), Some(radius))
    };

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
        validity_radius,
        off_manifold_norm,
        metric_provenance: provenance,
    })
}

/// The model's predicted output-mean response to an applied activation push
/// `δ`, under the LOCAL-LINEAR reading of its fitted surface: the projection
/// of `δ` onto the span of atom `atom_k`'s decoder tangents `∂g_k/∂t` at the
/// operating point `t_at`. A dictionary "predicts" exactly the component of a
/// push it can carry along its learned surface; the transverse component is
/// off-manifold and predicted to die (this is the same local model the
/// off-manifold guard and the dosimetry chord trust, used in the same radius).
///
/// This is `μ(δ)` for the design loop of
/// [`gam_terms::inference::structure_evidence`]: two structural hypotheses about
/// the same activations (e.g. "one curved atom" vs "two flat atoms") are two
/// fitted terms whose tangent spans differ, so they predict DIFFERENT
/// responses to the same probe — and that disagreement, in the output-Fisher
/// metric, is what `select_probe_by_expected_evidence` maximizes.
pub fn predicted_response(
    model: &SaeManifoldTerm,
    atom_k: usize,
    t_at: &[f64],
    delta: ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    let k = model.k_atoms();
    if atom_k >= k {
        return Err(format!(
            "predicted_response: atom index {atom_k} out of range (term has {k} atoms)"
        ));
    }
    let atom = &model.atoms[atom_k];
    let d = atom.latent_dim;
    let p = atom.output_dim();
    if t_at.len() != d {
        return Err(format!(
            "predicted_response: t_at must have length latent_dim={d}; got {}",
            t_at.len()
        ));
    }
    if delta.len() != p {
        return Err(format!(
            "predicted_response: delta must have length output_dim={p}; got {}",
            delta.len()
        ));
    }
    atom.basis_evaluator.as_ref().ok_or_else(|| {
        format!(
            "predicted_response: atom {atom_k} ('{}') has no installed basis evaluator",
            atom.name
        )
    })?;
    let tangents = decode_tangents_at(atom, t_at, model.tier0_scale())?;
    Ok(project_onto_tangent_span(&tangents, delta))
}

/// A patched-forward KL probe the target-dose loop plugs into: given an applied
/// amplitude `a`, return the measured `KL(p_base ‖ p_patched)` in nats for the
/// chord `a·(g(t_to) − g(t_from))`. The GPU re-measurement supplies a real
/// model-in-the-loop forward here; the closed-form seed needs none.
pub type PatchedForwardKl<'a> = dyn FnMut(f64) -> Result<f64, String> + 'a;

/// Tuning for the closed-loop correction in [`steer_to_target_nats`].
#[derive(Clone, Copy, Debug)]
pub struct TargetDoseConfig {
    /// Relative tolerance on measured KL vs the target that stops the loop.
    pub tol_rel: f64,
    /// Hard cap on patched-forward probes, including bracket construction.
    pub max_iter: usize,
    /// A probed amplitude counts as inside the readout-KL radius while its
    /// measured KL matches the local quadratic within this relative tolerance.
    pub readout_tol_rel: f64,
}

impl Default for TargetDoseConfig {
    fn default() -> Self {
        Self {
            tol_rel: 1.0e-2,
            max_iter: 12,
            readout_tol_rel: 1.0e-1,
        }
    }
}

/// One target-dose solve on a fixed atom chord.
///
/// The model and row metric remain explicit execution context; everything that
/// identifies and tunes the requested dose is carried together so callers
/// cannot accidentally reorder a train of homogeneous scalar/slice arguments.
#[derive(Clone, Copy, Debug)]
pub struct TargetDoseRequest<'a> {
    /// The atom whose coordinate is being steered.
    pub atom_k: usize,
    /// Exact fitted row whose output-Fisher block prices the move.
    pub metric_row: usize,
    /// Source on-manifold coordinate.
    pub t_from: &'a [f64],
    /// Target on-manifold coordinate, which fixes the chord direction.
    pub t_to: &'a [f64],
    /// Requested output-KL dose in nats.
    pub target_nats: f64,
    /// Closed-loop correction tuning.
    pub config: TargetDoseConfig,
}

/// A target output-KL dose on one atom's chord, returned atomically with the
/// exact activation-space move the caller must apply (gh#2249/#2263).
#[derive(Clone, Debug)]
pub struct TargetDosePlan {
    /// The requested dose in nats of KL.
    pub target_nats: f64,
    /// Closed-form first-order amplitude `a0 = sqrt(2 q* / (dgᵀ M dg))`, exact in
    /// the quadratic/in-radius regime.
    pub seed_amplitude: f64,
    /// Exact applied move at the solved amplitude, including `delta`, predicted
    /// dose, metric provenance, chart radius, and off-manifold audit.
    pub steer: SteerPlan,
    /// Probe-measured patched-forward KL at [`SteerPlan::amplitude`], when a callback
    /// was supplied; `None` for the pure closed-form seed.
    pub measured_nats: Option<f64>,
    /// Number of patched-forward probes consumed (0 without a callback).
    pub iterations: usize,
    /// **READOUT-KL radius**: the largest probed amplitude whose measured KL still
    /// matched the local quadratic within `readout_tol_rel` before the first
    /// probed failure. A later accidental match cannot extend the radius past a
    /// failed point. `None` without a callback or when the first probe failed.
    pub readout_kl_radius: Option<f64>,
}

/// A measured target-dose solve either returns a certified plan or one of these
/// explicit failure states. No unconverged iterate is representable as success.
#[derive(Clone, Debug, PartialEq)]
pub enum TargetDoseError {
    InvalidRequest(String),
    Steering(String),
    Probe(String),
    UnreachableTarget {
        target_nats: f64,
        amplitude: f64,
        measured_nats: f64,
    },
    ProbeBudgetExhausted {
        target_nats: f64,
        lower_nats: f64,
        upper_nats: f64,
        probes: usize,
    },
}

impl std::fmt::Display for TargetDoseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(message) | Self::Steering(message) | Self::Probe(message) => {
                f.write_str(message)
            }
            Self::UnreachableTarget {
                target_nats,
                amplitude,
                measured_nats,
            } => write!(
                f,
                "steer_to_target_nats: target {target_nats} nats is unreachable; \
                 KL stopped increasing at amplitude {amplitude} with {measured_nats} nats"
            ),
            Self::ProbeBudgetExhausted {
                target_nats,
                lower_nats,
                upper_nats,
                probes,
            } => write!(
                f,
                "steer_to_target_nats: exhausted {probes} probes before resolving target \
                 {target_nats} nats inside measured bracket [{lower_nats}, {upper_nats}]"
            ),
        }
    }
}

impl std::error::Error for TargetDoseError {}

/// One patched-forward probe with a finiteness/non-negativity guard on the
/// returned KL. A free fn (not a closure) so the `&mut PatchedForwardKl<'_>`
/// reborrow across the correction loop is unambiguous.
fn probe_kl(probe: &mut PatchedForwardKl<'_>, a: f64) -> Result<f64, TargetDoseError> {
    let kl = probe(a).map_err(TargetDoseError::Probe)?;
    if !(kl.is_finite() && kl >= 0.0) {
        return Err(TargetDoseError::Probe(format!(
            "steer_to_target_nats: probe returned a non-finite/negative KL {kl} at amplitude {a}"
        )));
    }
    Ok(kl)
}

/// Solve for the amplitude that lands a target output-KL dose `target_nats` (in
/// nats) on atom `atom_k`'s chord from `t_from` to `t_to` (gh#2263 target-dose
/// surface — `amplitude = 1` has no universal meaning, the dose does).
///
/// The closed form `a0 = sqrt(2 q* / (dgᵀ M dg))` (`dg` the unit-amplitude chord,
/// `M` the row output-Fisher) is exact in the quadratic/in-radius regime — and is
/// correctly scaled only because the chord and the metric now share the raw
/// activation frame (gh#2249, `ace3b9af3`; a Tier-0 σ-mis-scale on `dg` would
/// have poisoned `a0`). Past the readout-KL radius the true KL saturates, so an
/// optional `probe` (a patched forward) drives a secant correction onto the
/// measured curve and opportunistically records the readout-KL radius. With
/// `probe = None` the result is the unvalidated closed-form seed: pure math and
/// plumbing, no model in the loop.
pub fn steer_to_target_nats(
    model: &SaeManifoldTerm,
    metric: &RowMetric,
    request: TargetDoseRequest<'_>,
    probe: Option<&mut PatchedForwardKl<'_>>,
) -> Result<TargetDosePlan, TargetDoseError> {
    let TargetDoseRequest {
        atom_k,
        metric_row,
        t_from,
        t_to,
        target_nats,
        config,
    } = request;
    if !(target_nats.is_finite() && target_nats > 0.0) {
        return Err(TargetDoseError::InvalidRequest(format!(
            "steer_to_target_nats: target_nats must be finite and positive, got {target_nats}"
        )));
    }
    if !(config.tol_rel > 0.0) || config.max_iter == 0 || !(config.readout_tol_rel > 0.0) {
        return Err(TargetDoseError::InvalidRequest(format!(
            "steer_to_target_nats: config must have tol_rel>0, max_iter>0, readout_tol_rel>0; \
             got {config:?}"
        )));
    }
    // Unit-amplitude reference: predicted_nats(a) = a²·unit_nats, and the chart
    // radius / provenance / dose kind are all amplitude-invariant.
    let unit = steer_delta(model, metric, atom_k, metric_row, 1.0, t_from, t_to)
        .map_err(TargetDoseError::Steering)?;
    let unit_nats = unit.predicted_nats.ok_or_else(|| {
        TargetDoseError::InvalidRequest(format!(
            "steer_to_target_nats: atom {atom_k} has no behavioral (nats) metric \
             (provenance {:?}); a target-nats dose is undefined",
            unit.metric_provenance
        ))
    })?;
    if !(unit_nats > 0.0) {
        return Err(TargetDoseError::InvalidRequest(format!(
            "steer_to_target_nats: unit-amplitude dose is {unit_nats} (the chord carries no \
             Fisher mass in metric row {metric_row}); cannot solve for a target amplitude"
        )));
    }
    // Closed-form first-order seed: a0²·unit_nats = q*.
    let seed_amplitude = (target_nats / unit_nats).sqrt();
    let quad = |a: f64| a * a * unit_nats;

    let finish = |amplitude: f64,
                  measured_nats: Option<f64>,
                  iterations: usize,
                  readout_kl_radius: Option<f64>|
     -> Result<TargetDosePlan, TargetDoseError> {
        let steer = steer_delta(
            model,
            metric,
            atom_k,
            metric_row,
            amplitude,
            t_from,
            t_to,
        )
        .map_err(TargetDoseError::Steering)?;
        Ok(TargetDosePlan {
            target_nats,
            seed_amplitude,
            steer,
            measured_nats,
            iterations,
            readout_kl_radius,
        })
    };

    let probe = match probe {
        Some(probe) => probe,
        // No model in the loop: return the unvalidated closed-form seed.
        None => return finish(seed_amplitude, None, 0, None),
    };

    // Track a contiguous probed prefix of quadratic agreement. Once an amplitude
    // fails the contract, no later probe at or beyond it can enlarge the radius.
    let mut first_readout_failure: Option<f64> = None;
    let mut readout_kl_radius: Option<f64> = None;
    let mut record_readout = |amplitude: f64, measured: f64| {
        let predicted = quad(amplitude);
        let agrees = predicted > 0.0
            && (measured - predicted).abs() / predicted <= config.readout_tol_rel;
        if agrees {
            if first_readout_failure.is_none_or(|failed| amplitude < failed) {
                readout_kl_radius = Some(
                    readout_kl_radius.map_or(amplitude, |radius| radius.max(amplitude)),
                );
            }
        } else {
            first_readout_failure = Some(
                first_readout_failure.map_or(amplitude, |failed| failed.min(amplitude)),
            );
            if readout_kl_radius.is_some_and(|radius| radius >= amplitude) {
                readout_kl_radius = None;
            }
        }
    };

    // Establish a genuine first-crossing bracket [lo, hi], starting from the
    // exact point KL(0)=0 and expanding the closed-form seed until the measured
    // curve reaches the target. A plateau is an explicit unreachable target.
    let mut probes = 0usize;
    let mut lo_a = 0.0_f64;
    let mut lo_kl = 0.0_f64;
    let mut hi_a = seed_amplitude;
    let mut hi_kl = probe_kl(probe, hi_a)?;
    probes += 1;
    record_readout(hi_a, hi_kl);
    if (hi_kl - target_nats).abs() / target_nats <= config.tol_rel {
        return finish(hi_a, Some(hi_kl), probes, readout_kl_radius);
    }
    while hi_kl < target_nats {
        if probes >= config.max_iter {
            return Err(TargetDoseError::ProbeBudgetExhausted {
                target_nats,
                lower_nats: hi_kl,
                upper_nats: hi_kl,
                probes,
            });
        }
        let next_a = hi_a * 2.0;
        let next_kl = probe_kl(probe, next_a)?;
        probes += 1;
        record_readout(next_a, next_kl);
        if next_kl <= hi_kl {
            return Err(TargetDoseError::UnreachableTarget {
                target_nats,
                amplitude: next_a,
                measured_nats: next_kl,
            });
        }
        lo_a = hi_a;
        lo_kl = hi_kl;
        hi_a = next_a;
        hi_kl = next_kl;
        if (hi_kl - target_nats).abs() / target_nats <= config.tol_rel {
            return finish(hi_a, Some(hi_kl), probes, readout_kl_radius);
        }
    }

    // Safeguarded secant inside the measured bracket. If roundoff puts the
    // secant outside the open bracket, bisection preserves monotone contraction.
    while probes < config.max_iter {
        let denominator = hi_kl - lo_kl;
        let secant = hi_a - (hi_kl - target_nats) * (hi_a - lo_a) / denominator;
        let candidate = if secant.is_finite() && secant > lo_a && secant < hi_a {
            secant
        } else {
            0.5 * (lo_a + hi_a)
        };
        let measured = probe_kl(probe, candidate)?;
        probes += 1;
        record_readout(candidate, measured);
        if (measured - target_nats).abs() / target_nats <= config.tol_rel {
            return finish(candidate, Some(measured), probes, readout_kl_radius);
        }
        if measured < target_nats {
            lo_a = candidate;
            lo_kl = measured;
        } else {
            hi_a = candidate;
            hi_kl = measured;
        }
    }
    Err(TargetDoseError::ProbeBudgetExhausted {
        target_nats,
        lower_nats: lo_kl,
        upper_nats: hi_kl,
        probes,
    })
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
    let d = atom.latent_dim;
    let coords = Array2::from_shape_vec((1, d), t.to_vec())
        .map_err(|e| format!("steer_delta::decode_tangents_at: coord shape: {e}"))?;
    let jet = if atom.homotopy_eta == 1.0 {
        evaluator.evaluate(coords.view())?.1
    } else {
        evaluator
            .evaluate_phi_eta(coords.view(), atom.homotopy_eta)?
            .jet
    };
    let decoder = &atom.decoder_coefficients;
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

/// Least-squares projection of `δ` onto the span of the local tangents
/// (columns of `tangents`, shape `p × d`): `δ̂ = T (TᵀT)⁻¹ Tᵀ δ` via a small
/// `d × d` Gram solve (with a tiny diagonal jitter to absorb a rank-deficient
/// tangent frame; the jitter only shrinks the projection, never inflates it).
fn project_onto_tangent_span(tangents: &Array2<f64>, delta: ArrayView1<'_, f64>) -> Array1<f64> {
    let p = tangents.nrows();
    let d = tangents.ncols();
    if d == 0 {
        return Array1::<f64>::zeros(p);
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
    let trace: f64 = (0..d).map(|a| gram[[a, a]]).sum();
    let jitter = if trace > 0.0 { 1e-12 * trace } else { 1e-12 };
    for a in 0..d {
        gram[[a, a]] += jitter;
    }
    let coeffs = solve_spd_small(&gram, &rhs);
    let mut proj = Array1::<f64>::zeros(p);
    for i in 0..p {
        for a in 0..d {
            proj[i] += tangents[[i, a]] * coeffs[a];
        }
    }
    proj
}

/// Norm of `δ`'s component orthogonal to the span of the local tangents:
/// `‖δ − δ̂‖` with `δ̂` the [`project_onto_tangent_span`] projection.
fn off_manifold_residual_norm(tangents: &Array2<f64>, delta: ArrayView1<'_, f64>) -> f64 {
    let proj = project_onto_tangent_span(tangents, delta);
    let mut res_sq = 0.0_f64;
    for i in 0..delta.len() {
        let r = delta[i] - proj[i];
        res_sq += r * r;
    }
    res_sq.max(0.0).sqrt()
}

/// Tiny symmetric-positive-definite solve via Cholesky for the `d × d` tangent
/// Gram (`d` is the atom's latent dim, typically 1–3). Falls back to the bare rhs
/// if the factorization fails (a fully degenerate frame), which only inflates the
/// reported off-manifold residual — never deflates it.
fn solve_spd_small(gram: &Array2<f64>, rhs: &Array1<f64>) -> Array1<f64> {
    let d = gram.nrows();
    // Cholesky L LᵀT = gram.
    let mut l = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..=i {
            let mut sum = gram[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    return Array1::<f64>::zeros(d);
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    // Forward solve L y = rhs.
    let mut y = Array1::<f64>::zeros(d);
    for i in 0..d {
        let mut sum = rhs[i];
        for k in 0..i {
            sum -= l[[i, k]] * y[k];
        }
        y[i] = sum / l[[i, i]];
    }
    // Back solve Lᵀ x = y.
    let mut x = Array1::<f64>::zeros(d);
    for i in (0..d).rev() {
        let mut sum = y[i];
        for k in (i + 1)..d {
            sum -= l[[k, i]] * x[k];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

/// The fixed geometry of one steering query, bundled so the dose integrator and
/// its helpers take a single context rather than a long argument list.
struct SteerContext<'a> {
    atom: &'a SaeManifoldAtom,
    /// Tier-0 column scale of the owning term, when standardization /
    /// equilibration is installed: decodes must be un-scaled back to raw
    /// activation units before they meet the (always raw-frame) row metric.
    scale: Option<&'a Array1<f64>>,
    metric: &'a RowMetric,
    /// The row whose per-row metric the dose is measured through.
    row: usize,
    /// Output dimension `p`.
    p: usize,
    /// Latent dimension `d`.
    d: usize,
    /// Amplitude `a` the move is scaled by.
    amplitude: f64,
    coordinate_delta: &'a [f64],
    periods: &'a [Option<f64>],
}

/// The validity radius: the latent step length (Euclidean distance from
/// `t_from`) at which **local linearization stops being trusted**.
///
/// Linearizing the steering move means predicting the output effect of a prefix
/// step `τ·Δt` from the initial tangent alone: the first-order output move is
/// `δ_lin(τ) = a · (∂g/∂t|_{t_from}) · (τ Δt)`, whose output-Fisher KL is the
/// quadratic form `½ ‖δ_lin(τ)‖²_M = τ² · ½ a² ‖∂g/∂t·Δt‖²_M`. The **true**
/// effect of that prefix is the chord quadratic form of the *actual* curved
/// output move `½ a² ‖g(t_from + τΔt) − g(t_from)‖²_M`.
///
/// The radius is the chord length `τ* · ‖Δt‖` at the first prefix `τ*` where the
/// true chord KL diverges from the linear prediction by more than
/// [`VALIDITY_DIVERGENCE_FRACTION`] (relative to the linear prediction). This is
/// pure surface curvature: on a flat decoder the two agree for every `τ` and the
/// radius is the whole move. If the metric kills the tangent (no linear effect to
/// validate), the move is trusted to its full length.
fn validity_radius(ctx: &SteerContext<'_>, t_from: &[f64]) -> Result<f64, String> {
    let d = ctx.d;
    let p = ctx.p;
    let full_len: f64 = ctx
        .coordinate_delta
        .iter()
        .map(|d| d * d)
        .sum::<f64>()
        .sqrt();
    if full_len == 0.0 {
        return Ok(0.0);
    }
    let dt = ctx.coordinate_delta;
    let amp = ctx.amplitude;

    // Initial-tangent linear output move per unit τ: v0 = (∂g/∂t|_{t_from}) Δt.
    let tang0 = decode_tangents_at(ctx.atom, t_from, ctx.scale)?;
    let mut v0 = Array1::<f64>::zeros(p);
    for i in 0..p {
        let mut acc = 0.0_f64;
        for a in 0..d {
            acc += tang0[[i, a]] * dt[a];
        }
        v0[i] = acc;
    }
    // ½ a² ‖v0‖²_M — the per-τ² linear KL coefficient.
    let lin_coeff = 0.5 * amp * amp * ctx.metric.fisher_mass(ctx.row, v0.view());
    // No linear effect to validate against ⇒ trust the full move.
    if !(lin_coeff > 0.0) {
        return Ok(full_len);
    }

    let g_from = decode_at(ctx.atom, t_from, ctx.scale)?;
    let steps = STEER_VALIDITY_STEPS;
    for s in 0..steps {
        let tau = (s as f64 + 1.0) / steps as f64;
        let t_mid = path_coordinate(t_from, dt, ctx.periods, tau);
        let g_tau = decode_at(ctx.atom, &t_mid, ctx.scale)?;
        let mut chord = Array1::<f64>::zeros(p);
        for i in 0..p {
            chord[i] = amp * (g_tau[i] - g_from[i]);
        }
        // True chord KL of the prefix, and the linear prediction τ²·lin_coeff.
        let chord_kl = 0.5 * ctx.metric.fisher_mass(ctx.row, chord.view());
        let lin_kl = tau * tau * lin_coeff;
        let rel = (chord_kl - lin_kl).abs() / lin_kl;
        if rel > VALIDITY_DIVERGENCE_FRACTION {
            return Ok(tau * full_len);
        }
    }
    Ok(full_len)
}

/// One dose sample on a collateral-damage curve (gam#2234 E2, the intrinsic
/// Rust-owned counterpart of the model-in-the-loop KL frontier): the on-target
/// effect and the off-target collateral of a single steering intervention,
/// measured in the fitted dictionary's own representation, with no LLM in the
/// loop.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct CollateralPoint {
    /// The chart-coordinate dose applied to the target atom's steered axis
    /// (radians / fraction-of-period, per the atom's manifold).
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
fn frame_landed_norm(frame: &Array2<f64>, delta: ArrayView1<'_, f64>) -> f64 {
    let proj = project_onto_tangent_span(frame, delta);
    proj.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Sweep the intrinsic collateral-damage curve for steering atom `atom_k` along
/// latent `axis` over `doses`, comparing the on-manifold group action against a
/// matched-per-row-norm flat-direction control (gam#2234 E2).
///
/// For each dose `δ` the on-manifold ambient move is the fitted group-action
/// delta [`SaeManifoldTerm::steer_rows`] (chart step `δ` on `axis`, gate held
/// fixed). The flat control replays the SAME per-row move NORM along one fixed
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
    let d_k = model.atoms[atom_k].latent_dim;
    if axis >= d_k {
        return Err(format!(
            "collateral_curve: axis {axis} out of range for atom {atom_k} latent_dim {d_k}"
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
    // averages to ≈0 over a full circle). Found by power iteration on the small
    // `p × p` Gram, which is exact for the leading direction.
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
    let mut w = Array1::<f64>::from_elem(p, 1.0 / (p as f64).sqrt());
    for _ in 0..128 {
        let mut next = Array1::<f64>::zeros(p);
        for i in 0..p {
            let mut acc = 0.0_f64;
            for j in 0..p {
                acc += gram[[i, j]] * w[j];
            }
            next[i] = acc;
        }
        let norm = next.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if !(norm > 0.0) {
            return Err(format!(
                "collateral_curve: atom {atom_k} has a vanishing tangent field along axis {axis}; \
                 no fixed direction to define the flat control"
            ));
        }
        next.mapv_inplace(|x| x / norm);
        w = next;
    }

    // Decompose one per-row move field into (effect, off-target collateral,
    // cross-feature leakage) RMS over rows.
    let decompose = |field: &Array2<f64>| -> CollateralPoint {
        let mut eff_sq = 0.0_f64;
        let mut col_sq = 0.0_f64;
        let mut cross_sq = 0.0_f64;
        for row in 0..n {
            let delta = field.row(row);
            let on_target = project_onto_tangent_span(&target_frames[row], delta);
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
                let l = frame_landed_norm(&frames[row], delta);
                cross += l * l;
            }
            cross_sq += cross;
        }
        let denom = n.max(1) as f64;
        CollateralPoint {
            dose: 0.0,
            on_target_effect: (eff_sq / denom).sqrt(),
            collateral: (col_sq / denom).sqrt(),
            cross_feature: (cross_sq / denom).sqrt(),
        }
    };

    let mut manifold_pts = Vec::with_capacity(doses.len());
    let mut flat_pts = Vec::with_capacity(doses.len());
    for &dose in doses {
        let mut step = Array1::<f64>::zeros(d_k);
        step[axis] = dose;
        let on_field = model.steer_rows(atom_k, &rows, step.view())?;

        // Matched control: same per-row move NORM, along the fixed direction w.
        let mut flat_field = Array2::<f64>::zeros((n, p));
        for row in 0..n {
            let norm = on_field.row(row).iter().map(|&x| x * x).sum::<f64>().sqrt();
            for i in 0..p {
                flat_field[[row, i]] = norm * w[i];
            }
        }

        let mut m = decompose(&on_field);
        m.dose = dose;
        manifold_pts.push(m);
        let mut f = decompose(&flat_field);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_steering_uses_shortest_path_across_seam() {
        let periods = [Some(1.0)];
        let delta = shortest_coordinate_delta(&[0.99], &[0.01], &periods).unwrap();
        assert!((delta[0] - 0.02).abs() < 1e-12);
        let midpoint = path_coordinate(&[0.99], &delta, &periods, 0.5);
        assert!(midpoint[0].abs() < 1e-12 || (midpoint[0] - 1.0).abs() < 1e-12);
        let distance = coordinate_l2_distance(&[0.99], &[0.01], &periods).unwrap();
        assert!((distance - 0.02).abs() < 1e-12);
    }
}

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
//! Here `g_k(t) = exp(s_k) Phi_k^eta(t) B_k` is the fitted physical decoder,
//! including the atom scale and curvature-homotopy state:
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

fn coordinate_l2_distance(
    a: &[f64],
    b: &[f64],
    periods: &[Option<f64>],
) -> Result<f64, String> {
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
    let g_from = decode_at(atom, t_from)?;
    let g_to = decode_at(atom, t_to)?;
    let mut delta = Array1::<f64>::zeros(p);
    for i in 0..p {
        delta[i] = amplitude * (g_to[i] - g_from[i]);
    }

    // Whether the metric can/does match this term and carries behavior.
    let provenance = metric.provenance();
    let behavior_available = metric_carries_behavior(provenance);

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
    let tangents = decode_tangents_at(atom, &t_mid)?;
    let off_manifold_norm = off_manifold_residual_norm(&tangents, delta.view());

    // --- dosimetry: exact applied-delta Fisher endpoint KL ------------------
    let (predicted_nats, validity_radius) = if !behavior_available {
        (None, None)
    } else {
        let ctx = SteerContext {
            atom,
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
    let tangents = decode_tangents_at(atom, t_at)?;
    Ok(project_onto_tangent_span(&tangents, delta))
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
fn decode_at(atom: &SaeManifoldAtom, t: &[f64]) -> Result<Array1<f64>, String> {
    let d = t.len();
    let coords = Array2::from_shape_vec((1, d), t.to_vec())
        .map_err(|e| format!("steer_delta::decode_at: coord shape: {e}"))?;
    Ok(atom.decode_at_coords(coords.view())?.row(0).to_owned())
}

/// Evaluate the decoder tangents `∂g_k/∂t_a = Φ_k'(t) B_k ∈ ℝ^p`, one per latent
/// axis `a ∈ 0..d`, at an arbitrary latent coordinate `t`. Returned as a
/// `(p × d)` matrix whose column `a` is the tangent along axis `a`.
fn decode_tangents_at(atom: &SaeManifoldAtom, t: &[f64]) -> Result<Array2<f64>, String> {
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
        evaluator.evaluate_phi_eta(coords.view(), atom.homotopy_eta)?.jet
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
    if atom.log_amplitude != 0.0 {
        tang.mapv_inplace(|value| value * atom.log_amplitude.exp());
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
    let full_len: f64 = ctx.coordinate_delta.iter().map(|d| d * d).sum::<f64>().sqrt();
    if full_len == 0.0 {
        return Ok(0.0);
    }
    let dt = ctx.coordinate_delta;
    let amp = ctx.amplitude;

    // Initial-tangent linear output move per unit τ: v0 = (∂g/∂t|_{t_from}) Δt.
    let tang0 = decode_tangents_at(ctx.atom, t_from)?;
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

    let g_from = decode_at(ctx.atom, t_from)?;
    let steps = STEER_VALIDITY_STEPS;
    for s in 0..steps {
        let tau = (s as f64 + 1.0) / steps as f64;
        let t_mid = path_coordinate(t_from, dt, ctx.periods, tau);
        let g_tau = decode_at(ctx.atom, &t_mid)?;
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

//! Query/key edits with every cross term, certified through the exact finite softmax.
//!
//! One head scores query token `t` against key token `s` as
//!
//! ```text
//! S_ts = σ (Q x_t)ᵀ R_{Δ} (K x_s),   Δ = p_s − p_t,
//! ```
//!
//! with `R_Δ` the source's rotary action (each plane rotated by `ω_j Δ` and scaled by the
//! attention scaling squared; pass-through coordinates untouched) or the identity. The
//! setting `(Q′, K′)` (set-type: the weights after the edit) is the edit
//! `(ΔQ, ΔK) = (Q′ − Q, K′ − K)`, and it changes the score by exactly
//!
//! ```text
//! S′ − S = σ [ qᵀ R Δk + Δqᵀ R k ]  +  σ Δqᵀ R Δk,     q = Q x_t, Δq = ΔQ x_t, …
//!          └──────── first order ───────┘   └ cross ┘
//! ```
//!
//! A first-order edit may be proposed, but only the finite change is certified: the
//! compiler evaluates it by the exact bilinear secant `L′R′ − LR = L̄ ΔR + ΔL R̄`
//! ([`bilinear_change`]) on the rotated rows `q̃_t = R_{p_t} q_t`, `k̃_s = R_{p_s} k_s`
//! (`R_{p_t}ᵀ R_{p_s} = R_Δ`), reports the first-order part and the cross term beside it, and
//! never drops `Δqᵀ R Δk`.
//!
//! # Through the softmax
//!
//! Each query row's score changes `δ` feed the source's softmax over the keys it may attend
//! to. The finite change is `p′_i = p_i e^{δ_i} / Σ_j p_j e^{δ_j}`, evaluated exactly with its
//! band by [`softmax_change`]. An explanation claims its own change of the scores (the first
//! order, or declared logits); the total variation between the claimed attention row and the
//! certified one is at most `tanh(w/4)` with `w` the oscillation of the logit gap, widened
//! by both sides' bands ([`total_variation_over_logit_boxes`]).
//!
//! # Status
//!
//! The edit is a fixed native change, certified on the declared query and key rows:
//! [`ControlRealization::EmpiricallyValidated`] with the largest row's total-variation
//! bound as a [`EvidenceStatus::UniformBound`]. When the claim is the first order and the
//! cross operator vanishes identically (one side unedited), the first order is the finite
//! change at every input, and the control is
//! [`ControlRealization::ExactlyRealized`] with a zero residual.
//!
//! # Bands
//!
//! The rotated rows are formed from the stored weights and inputs: `Q x` errs by
//! `γ_d |Q||x|`, and each rotated coordinate combines two of them with `cos`/`sin` whose
//! evaluation errs by `η = γ_1 |φ| + ε` (the attention owner's plane rule), plus `γ_3` for the
//! combination and the scaling. Those formation bands propagate into the score changes as
//! `|δL||R| + |L||δR|` beside [`bilinear_change`]'s own band.

use gam_linalg::roundoff::accumulation_growth;
use gam_math::roundoff::inflated;
use ndarray::{Array1, Array2, ArrayView2};

use super::super::apply::FactoredEdit;
use super::super::attention::RotaryEmbedding;
use super::super::bounds::total_variation_over_logit_boxes;
use super::super::lift::{TensorId, TensorRegistry};
use super::super::secant::{BandedMatrix, BandedVector, bilinear_change, softmax_change};
use super::super::supports::{EvidenceStatus, ExactBasis};
use super::{
    CompileError, CompiledControl, CompiledParameterEdit, ControlRealization, NativeEditPlan, require_finite,
    require_shape,
};

/// What the explanation claims the score change is.
#[derive(Clone, Debug)]
pub enum ScoreClaim<'a> {
    /// The first-order change `σ [qᵀ R Δk + Δqᵀ R k]`.
    FirstOrder,
    /// Declared score changes (`queries × keys`, already multiplied by `σ`).
    Declared(ArrayView2<'a, f64>),
}

/// Where a head's query or key rows live in a stored projection (`heads·head_dim × width`,
/// the torch `Linear` layout): rows `row_offset .. row_offset + head_dim`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadRows {
    pub storage: TensorId,
    pub row_offset: usize,
}

/// A query/key edit to certify.
#[derive(Clone, Debug)]
pub struct QueryKeyEditProblem<'a> {
    pub registry: &'a TensorRegistry,
    /// The native `Q`, `K` and their set values `Q′`, `K′` (set-type: the weights after
    /// the edit): `head_dim × width` each.
    pub query: ArrayView2<'a, f64>,
    pub key: ArrayView2<'a, f64>,
    pub query_setting: ArrayView2<'a, f64>,
    pub key_setting: ArrayView2<'a, f64>,
    pub query_rows: HeadRows,
    pub key_rows: HeadRows,
    pub rotary: Option<&'a RotaryEmbedding>,
    /// `σ`, the score scale (e.g. `1/√head_dim`).
    pub score_scale: f64,
    /// Query inputs (`n_q × width`) at `query_positions`, key inputs (`n_k × width`) at
    /// `key_positions`.
    pub queries: ArrayView2<'a, f64>,
    pub query_positions: &'a [i64],
    pub keys: ArrayView2<'a, f64>,
    pub key_positions: &'a [i64],
    /// Only keys at positions `≤` the query's position are attended to.
    pub causal: bool,
    pub claim: ScoreClaim<'a>,
}

/// The family a query/key certificate is stated over: the declared query rows, each over
/// its attended keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryKeyDomain {
    pub query_rows: usize,
    pub key_rows: usize,
}

/// One query row's attention under the edit.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionRow {
    /// The keys it attends to.
    pub keys: Vec<usize>,
    /// `p′ − p` over those keys, with bands.
    pub change: BandedVector,
    /// Upper bound on `TV(claimed row, certified row)`.
    pub claim_total_variation: f64,
}

/// What [`compile_query_key_edit`] certified.
#[derive(Clone, Debug)]
pub struct QueryKeyEditReport {
    /// `S′ − S` (`σ` included) with per-entry bands.
    pub exact_change: BandedMatrix,
    /// `σ [qᵀ R Δk + Δqᵀ R k]`.
    pub first_order: Array2<f64>,
    /// `σ Δqᵀ R Δk`.
    pub cross: Array2<f64>,
    pub rows: Vec<AttentionRow>,
    pub compiled: CompiledControl<usize, QueryKeyDomain>,
}

/// Rotated, scaled rows `R_p (W x)` for each input row, and their entrywise formation bands.
fn rotated_rows(
    weight: ArrayView2<'_, f64>,
    inputs: ArrayView2<'_, f64>,
    positions: &[i64],
    rotary: Option<&RotaryEmbedding>,
) -> (Array2<f64>, Array2<f64>) {
    let width = weight.ncols();
    let projected = inputs.dot(&weight.t());
    let projected_band = inputs.mapv(f64::abs).dot(&weight.mapv(f64::abs).t()) * accumulation_growth(width);
    let Some(rotary) = rotary else {
        return (projected, projected_band);
    };
    let mut rows = projected.clone();
    let mut bands = projected_band.clone();
    let scaling = rotary.attention_scaling;
    for (index, &position) in positions.iter().enumerate() {
        for (plane, &frequency) in rotary.inverse_frequencies.iter().enumerate() {
            let (a, b) = rotary.plane(plane);
            let angle = position as f64 * frequency;
            let (sin, cos) = angle.sin_cos();
            let eta = accumulation_growth(1) * angle.abs() + f64::EPSILON;
            let (x, y) = (projected[[index, a]], projected[[index, b]]);
            let (ex, ey) = (projected_band[[index, a]], projected_band[[index, b]]);
            rows[[index, a]] = scaling * (cos * x - sin * y);
            rows[[index, b]] = scaling * (sin * x + cos * y);
            let magnitude = x.abs() + y.abs();
            let band = scaling.abs()
                * (cos.abs().max(sin.abs()) * (ex + ey) + eta * (magnitude + ex + ey) + accumulation_growth(3) * magnitude);
            bands[[index, a]] = inflated(band, 1);
            bands[[index, b]] = inflated(band, 1);
        }
    }
    (rows, bands)
}

/// Certifies the finite score and attention change of `(ΔQ, ΔK)` on the declared rows.
pub fn compile_query_key_edit(problem: &QueryKeyEditProblem<'_>, control: &str) -> Result<QueryKeyEditReport, CompileError> {
    let (head_dim, width) = problem.query.dim();
    for (what, matrix) in [
        ("key weight", problem.key),
        ("query setting", problem.query_setting),
        ("key setting", problem.key_setting),
    ] {
        require_shape(what, (head_dim, width), matrix.dim())?;
    }
    require_shape("query inputs", (problem.queries.nrows(), width), problem.queries.dim())?;
    require_shape("key inputs", (problem.keys.nrows(), width), problem.keys.dim())?;
    require_shape("query positions", (problem.queries.nrows(), 1), (problem.query_positions.len(), 1))?;
    require_shape("key positions", (problem.keys.nrows(), 1), (problem.key_positions.len(), 1))?;
    for (what, matrix) in [
        ("query weight", problem.query),
        ("key weight", problem.key),
        ("query setting", problem.query_setting),
        ("key setting", problem.key_setting),
        ("query inputs", problem.queries),
        ("key inputs", problem.keys),
    ] {
        require_finite(what, matrix.iter().copied())?;
    }
    if !(problem.score_scale.is_finite()) {
        return Err(CompileError::NonFinite { what: "score scale" });
    }
    if let Some(rotary) = problem.rotary
        && rotary.rotary_dim() > head_dim
    {
        return Err(CompileError::InvalidDeclaration {
            what: "rotary",
            reason: format!("rotates {} coordinates of a {head_dim}-dimensional head", rotary.rotary_dim()),
        });
    }
    let (n_q, n_k) = (problem.queries.nrows(), problem.keys.nrows());
    let edited_query = problem.query_setting;
    let edited_key = problem.key_setting;
    let query_delta = &edited_query - &problem.query;
    let key_delta = &edited_key - &problem.key;
    let rotary = problem.rotary;
    let (q, q_band) = rotated_rows(problem.query, problem.queries, problem.query_positions, rotary);
    let (q_edited, q_edited_band) = rotated_rows(edited_query, problem.queries, problem.query_positions, rotary);
    let (k, k_band) = rotated_rows(problem.key, problem.keys, problem.key_positions, rotary);
    let (k_edited, k_edited_band) = rotated_rows(edited_key, problem.keys, problem.key_positions, rotary);
    let (dq, _) = rotated_rows(query_delta.view(), problem.queries, problem.query_positions, rotary);
    let (dk, _) = rotated_rows(key_delta.view(), problem.keys, problem.key_positions, rotary);
    let sigma = problem.score_scale;

    let secant = bilinear_change(q.view(), q_edited.view(), k.t(), k_edited.t())?;
    let formation = |left: &Array2<f64>, left_band: &Array2<f64>, right: &Array2<f64>, right_band: &Array2<f64>| {
        left_band.dot(&right.mapv(f64::abs).t()) + left.mapv(f64::abs).dot(&right_band.t()) + left_band.dot(&right_band.t())
    };
    let propagated = formation(&q, &q_band, &k, &k_band) + formation(&q_edited, &q_edited_band, &k_edited, &k_edited_band);
    let mut exact_values = secant.values.mapv(|value| sigma * value);
    let mut exact_bands = Array2::from_shape_fn((n_q, n_k), |(t, s)| {
        inflated(sigma.abs() * (secant.bands[[t, s]] + propagated[[t, s]]) + f64::EPSILON * exact_values[[t, s]].abs(), 1)
    });
    let first_order = (q.dot(&dk.t()) + dq.dot(&k.t())) * sigma;
    let cross = dq.dot(&dk.t()) * sigma;

    let claimed = match &problem.claim {
        ScoreClaim::FirstOrder => first_order.clone(),
        ScoreClaim::Declared(values) => {
            require_shape("declared score changes", (n_q, n_k), values.dim())?;
            require_finite("declared score changes", values.iter().copied())?;
            values.to_owned()
        }
    };
    let base = q.dot(&k.t()) * sigma;
    let mut rows = Vec::with_capacity(n_q);
    let mut worst: Option<(f64, usize)> = None;
    for t in 0..n_q {
        let attended: Vec<usize> = (0..n_k)
            .filter(|&s| !problem.causal || problem.key_positions[s] <= problem.query_positions[t])
            .collect();
        for s in 0..n_k {
            if !attended.contains(&s) {
                exact_values[[t, s]] = 0.0;
                exact_bands[[t, s]] = 0.0;
            }
        }
        if attended.is_empty() {
            rows.push(AttentionRow {
                keys: attended,
                change: BandedVector {
                    values: Vec::new(),
                    bands: Vec::new(),
                },
                claim_total_variation: 0.0,
            });
            continue;
        }
        let start: Vec<f64> = attended.iter().map(|&s| base[[t, s]]).collect();
        let end: Vec<f64> = attended.iter().map(|&s| base[[t, s]] + exact_values[[t, s]]).collect();
        let change = softmax_change(&start, &end)?;
        let claim_row: Vec<f64> = attended.iter().map(|&s| base[[t, s]] + claimed[[t, s]]).collect();
        let actual_radius: Vec<f64> = attended.iter().map(|&s| exact_bands[[t, s]]).collect();
        let claim_radius = vec![0.0; attended.len()];
        let bound = total_variation_over_logit_boxes(
            Array1::from(claim_row).view(),
            Array1::from(claim_radius).view(),
            Array1::from(end).view(),
            Array1::from(actual_radius).view(),
        )?;
        let upper = bound.upper_bound().unwrap_or(1.0);
        if worst.is_none_or(|(value, _)| upper > value) {
            worst = Some((upper, t));
        }
        rows.push(AttentionRow {
            keys: attended,
            change,
            claim_total_variation: upper,
        });
    }

    let domain = QueryKeyDomain {
        query_rows: n_q,
        key_rows: n_k,
    };
    let structurally_first_order =
        query_delta.iter().all(|value| *value == 0.0) || key_delta.iter().all(|value| *value == 0.0);
    let realization = if structurally_first_order && matches!(problem.claim, ScoreClaim::FirstOrder) {
        ControlRealization::exactly_realized(EvidenceStatus::exact(0.0, 0.0, ExactBasis::Algebraic, None, domain)?)?
    } else {
        let upper = worst.map_or(0.0, |(value, _)| value);
        ControlRealization::empirically_validated(EvidenceStatus::uniform_bound(upper, 0.0, domain)?)?
    };
    let mut edits = Vec::new();
    for (rows_at, delta) in [(&problem.query_rows, query_delta.view()), (&problem.key_rows, key_delta.view())] {
        if delta.iter().all(|value| *value == 0.0) {
            continue;
        }
        let stored = problem
            .registry
            .storage(&rows_at.storage)
            .ok_or_else(|| CompileError::NotStorage(rows_at.storage.0.clone()))?;
        if stored.shape.len() != 2 || stored.shape[1] != width || rows_at.row_offset + head_dim > stored.shape[0] {
            return Err(CompileError::InvalidDeclaration {
                what: "head rows",
                reason: format!(
                    "rows {}..{} of width {width} do not fit {:?} of shape {:?}",
                    rows_at.row_offset,
                    rows_at.row_offset + head_dim,
                    rows_at.storage.0,
                    stored.shape
                ),
            });
        }
        let mut left = Array2::<f64>::zeros((stored.shape[0], head_dim));
        for row in 0..head_dim {
            left[[rows_at.row_offset + row, row]] = 1.0;
        }
        let edit = CompiledParameterEdit {
            storage: rows_at.storage.clone(),
            delta: FactoredEdit::new(left, delta.t().to_owned())?,
        };
        if let Some(existing) = edits.iter_mut().find(|item: &&mut CompiledParameterEdit| item.storage == edit.storage) {
            // Query and key rows of one fused projection: one edit holding both blocks.
            let left = ndarray::concatenate![ndarray::Axis(1), existing.delta.left(), edit.delta.left()];
            let right = ndarray::concatenate![ndarray::Axis(1), existing.delta.right(), edit.delta.right()];
            existing.delta = FactoredEdit::new(left, right)?;
        } else {
            edits.push(edit);
        }
    }
    let plan = NativeEditPlan::new(problem.registry, edits)?;
    Ok(QueryKeyEditReport {
        exact_change: BandedMatrix {
            values: exact_values,
            bands: exact_bands,
        },
        first_order,
        cross,
        rows,
        compiled: CompiledControl::new(control.to_string(), Some(plan), realization)?,
    })
}

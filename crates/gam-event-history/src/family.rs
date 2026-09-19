//! The event-history custom family: every subject's marginal likelihood
//! assembled into the coefficient space of the per-mark smooth blocks and the
//! latent block, with every derivative the outer LAML evaluator asks for
//! produced by the same generic code under a directional dual scalar, and
//! the fit driver that refines the Gauss-Hermite order and the time mesh
//! until the fitted coefficients are stationary under refinement.

use super::chain::{GaussHermite, product_grid_size};
use super::cohort::{
    CohortNodes, EventHistoryCohort, EventHistoryError, MarkKind,
    design_rows, expand_nodes,
};
use super::covariance::{
    DirectionEvidence, DirectionProfile, NewAtom, SubjectResiduals, best_new_atom,
    empirical_bayes_ridge,
};
use super::marginal::{
    LOST_POSITIVITY, SubjectInputs, expected_intensities, forward_filter, pairwise_sum,
    subject_marginal,
};
use super::preserve::{ReferenceGrid, ReferenceStrata, stratum_normalisers};
use super::scalar::{Tangent, add_real, recip};
use gam_custom_family::fit_custom_family;
use gam_model_api::families::custom_family::{
    BlockwiseFitOptions, CustomFamily, ExactNewtonJointGradientEvaluation, FamilyEvaluation,
};
use gam_problem::{BlockWorkingSet, ParameterBlockSpec, ParameterBlockState, PenaltyMatrix};
use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix, SymmetricMatrix};
use gam_math::jet_scalar::{JetScalar, OneSeed, Order2, TwoSeed};
use gam_math::nested_dual::JetField;
use gam_problem::CoefficientCoordinate;
use gam_solve::model_types::UnifiedFitResult;
use gam_terms::smooth::{
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design,
};
use ndarray::{Array1, Array2, ArrayView2, s};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

#[path = "objective.rs"]
mod objective;

/// A scalar that can be seeded with up to two directions and read back.
pub(crate) trait Directional: JetField + Send + Sync {
    fn seeded(value: f64, u: f64, v: f64) -> Self;
    fn eps(&self) -> f64;
    fn eps_del(&self) -> f64;
}

#[inline]
fn validate_seed_directions(directions: [f64; 2]) {
    assert!(
        directions.into_iter().all(f64::is_finite),
        "event-history derivative seed directions must be finite",
    );
}

impl Directional for f64 {
    fn seeded(value: f64, u: f64, v: f64) -> Self {
        validate_seed_directions([u, v]);
        value
    }
    fn eps(&self) -> f64 {
        0.0
    }
    fn eps_del(&self) -> f64 {
        0.0
    }
}

fn scalar0(value: f64) -> Order2<0> {
    <Order2<0> as JetScalar<0>>::constant(value)
}

impl Directional for OneSeed<0> {
    fn seeded(value: f64, u: f64, v: f64) -> Self {
        validate_seed_directions([u, v]);
        OneSeed {
            base: scalar0(value),
            eps: scalar0(u),
        }
    }
    fn eps(&self) -> f64 {
        self.eps.value()
    }
    fn eps_del(&self) -> f64 {
        0.0
    }
}

impl Directional for TwoSeed<0> {
    fn seeded(value: f64, u: f64, v: f64) -> Self {
        TwoSeed {
            base: scalar0(value),
            eps: scalar0(u),
            del: scalar0(v),
            eps_del: scalar0(0.0),
        }
    }
    fn eps(&self) -> f64 {
        self.eps.value()
    }
    fn eps_del(&self) -> f64 {
        self.eps_del.value()
    }
}

/// A full joint evaluation in coefficient space. `hessian` is the negative
/// log-likelihood Hessian, the convention the custom-family engine expects.
#[derive(Clone, Debug)]
pub(crate) struct JointEvaluation {
    pub log_likelihood: f64,
    pub gradient: Array1<f64>,
    pub hessian: Array2<f64>,
}

/// The event-history family over a node-expanded cohort.
#[derive(Clone)]
pub struct EventHistoryFamily {
    nodes: Arc<CohortNodes>,
    /// Dense per-mark designs on the nodes, `n_obs × p_d`.
    designs: Vec<Arc<Array2<f64>>>,
    atoms: usize,
    /// Per atom, the dimensionless rate `ν = rate · T̄` the atom is held at,
    /// or `None` when the rate is a coefficient of the latent block. A rate on
    /// a plateau of what the residuals resolve is held: there the likelihood
    /// is flat in the rate to double precision, so a coordinate for it would
    /// be unidentified, and the rate is data — the plateau's own value.
    held_rates: Vec<Option<f64>>,
    /// The band `[ν_min, ν_max]` of dimensionless rates the node mesh
    /// resolves. A free rate coefficient `u` is a chart of this band,
    /// `ν(u) = ν_min + (ν_max − ν_min) · u² / (1 + u²)`: the static wall
    /// `ν_min` is the fold `u = 0`, where a fit the data push against it
    /// reaches a stationary point of positive curvature instead of a plateau
    /// with a vanishing gradient, and the fast wall is the asymptote.
    rate_band: (f64, f64),
    gh: Arc<GaussHermite>,
    time_scale: f64,
    /// The reference population's grid and designs, when the baselines are
    /// the risk sets' marginal rates.
    reference: Option<Arc<ReferenceTables>>,
    /// The last joint evaluation, keyed on the exact state it was made at.
    cache: Arc<Mutex<Option<(Vec<f64>, Arc<JointEvaluation>)>>>,
    /// The reference step the latest evaluation refused, typed. Evaluations
    /// hand their errors to the custom-family engine as text, which cannot
    /// carry [`EventHistoryError::ReferenceStep`] back to the fit driver, so
    /// the driver takes the refusal from here when an evaluation fails. Every
    /// evaluation clears it on entry, so no refusal outlives the evaluation
    /// that raised it.
    reference_refusal: Arc<Mutex<Option<EventHistoryError>>>,
}

/// A reference-law snapshot evaluated at one coefficient state. The grid,
/// profile order, risk masks, coefficients, and numerical evolution travel
/// together when the snapshot is saved or used for prediction.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RiskSetCentring {
    pub grid: ReferenceGrid,
    pub profiles: Array2<f64>,
    pub coefficients: Vec<f64>,
    pub node_stratum: Vec<usize>,
    pub log_normaliser: Vec<f64>,
    pub log_risk_mass: Vec<f64>,
    pub masks: usize,
    pub mask_of_mark: Vec<usize>,
}

impl RiskSetCentring {
    /// Compare reference calculations at identical parameters and profiles.
    fn discrepancy(&self, refined: &Self, marks: usize) -> Result<f64, EventHistoryError> {
        if self.coefficients != refined.coefficients || self.profiles != refined.profiles
            || self.mask_of_mark != refined.mask_of_mark || self.masks != refined.masks {
            return Err(EventHistoryError::InvalidInput { reason:
                "reference refinement requires identical coefficients, profiles, and risk masks".to_string() });
        }
        let mut gap = 0.0_f64;
        for (a, b, width) in [
            (&self.log_normaliser, &refined.log_normaliser, marks),
            (&self.log_risk_mass, &refined.log_risk_mass, self.masks),
        ] {
            if a.len() != self.profiles.nrows() * self.grid.len() * width
                || b.len() != refined.profiles.nrows() * refined.grid.len() * width
                || a.iter().chain(b).any(|x| !x.is_finite()) {
                return Err(EventHistoryError::NumericalFailure { reason:
                    "reference refinement has invalid or non-finite risk moments".to_string() });
            }
            for s in 0..self.profiles.nrows() {
                for (n, &time) in refined.grid.times.iter().enumerate() {
                    let (low, weight) = self.grid.locate(time)?;
                    for d in 0..width {
                        let left = a[(s * self.grid.len() + low) * width + d];
                        let right = a[(s * self.grid.len() + low + 1) * width + d];
                        gap = gap.max((left + weight * (right - left)
                            - b[(s * refined.grid.len() + n) * width + d]).abs());
                    }
                }
            }
        }
        Ok(gap)
    }
}

impl ReferenceTables {
    /// Carry a normaliser held on the reference grid onto a node set, by the
    /// linear interpolation each node recorded when the tables were built.
    fn carry_to_nodes<S: JetField>(
        &self,
        held: &[S],
        marks: usize,
        total_nodes: usize,
    ) -> Result<Vec<S>, EventHistoryError> {
        let nodes = self.grid.len();
        let expected = self.strata * nodes * marks;
        if held.len() != expected {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "a held normaliser of {} entries for {} strata × {nodes} reference nodes × {marks} marks",
                    held.len(),
                    self.strata
                ),
            });
        }
        if self.node_stratum.len() != total_nodes {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "the reference tables place {} nodes, the family has {total_nodes}",
                    self.node_stratum.len()
                ),
            });
        }
        let mut out = vec![held[0].constant_like(0.0); total_nodes * marks];
        for row in 0..total_nodes {
            let base = self.node_stratum[row] * nodes;
            let lower = (base + self.node_lower[row]) * marks;
            let upper = (base + self.node_lower[row] + 1) * marks;
            let weight = self.node_weight[row];
            for d in 0..marks {
                let low = &held[lower + d];
                out[row * marks + d] = low.add(&held[upper + d].sub(low).scale(weight));
            }
        }
        Ok(out)
    }
}

/// The reference population's own grid, its per-stratum design rows, and
/// where every cohort node sits on that grid.
pub(crate) struct ReferenceTables {
    pub grid: ReferenceGrid,
    /// Covariate profiles retained independently of the training histories.
    pub(crate) profiles: Array2<f64>,
    /// The mark kinds, which decide each mark's risk set.
    kinds: Vec<MarkKind>,
    /// Per mark, the design of the reference rows, `strata · nodes × p_d`.
    designs: Vec<Arc<Array2<f64>>>,
    /// Per mark, the affine offset of those rows.
    offsets: Vec<Array1<f64>>,
    strata: usize,
    /// Per cohort node: its subject's stratum, the reference node below it and
    /// the weight of the one above.
    node_stratum: Vec<usize>,
    node_lower: Vec<usize>,
    node_weight: Vec<f64>,
}

impl EventHistoryFamily {
    pub fn new(
        nodes: Arc<CohortNodes>,
        designs: Vec<Arc<Array2<f64>>>,
        atoms: usize,
        gauss_hermite_order: usize,
        time_scale: f64,
        held_rates: Vec<Option<f64>>,
    ) -> Result<Self, EventHistoryError> {
        if held_rates.len() != atoms
            || held_rates
                .iter()
                .flatten()
                .any(|r| !(r.is_finite() && *r >= 0.0))
        {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "event-history family needs one nonnegative finite or free rate per atom: got {:?} for {atoms} atoms",
                    held_rates
                ),
            });
        }
        if designs.len() != nodes.marks {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "event-history family needs one design per mark: got {} designs for {} marks",
                    designs.len(),
                    nodes.marks
                ),
            });
        }
        for (d, design) in designs.iter().enumerate() {
            if design.nrows() != nodes.total_nodes {
                return Err(EventHistoryError::InvalidInput {
                    reason: format!(
                        "design for mark {d} has {} rows but the cohort has {} nodes",
                        design.nrows(),
                        nodes.total_nodes
                    ),
                });
            }
        }
        if !(time_scale.is_finite() && time_scale > 0.0) {
            return Err(EventHistoryError::InvalidInput {
                reason: "time scale must be finite and positive".to_string(),
            });
        }
        product_grid_size(gauss_hermite_order, atoms)?;
        let rate_band = rate_band(&nodes, time_scale)?;
        Ok(Self {
            nodes,
            designs,
            atoms,
            held_rates,
            rate_band,
            gh: Arc::new(GaussHermite::new(gauss_hermite_order)?),
            time_scale,
            reference: None,
            cache: Arc::new(Mutex::new(None)),
            reference_refusal: Arc::new(Mutex::new(None)),
        })
    }

    /// Attach the reference law. Its normaliser is evaluated and
    /// differentiated at every coefficient state, never held as an offset.
    pub(crate) fn with_reference(mut self, reference: Option<Arc<ReferenceTables>>) -> Self {
        self.reference = reference;
        self.cache = Arc::new(Mutex::new(None));
        self.reference_refusal = Arc::new(Mutex::new(None));
        self
    }

    /// The reference law at exactly the supplied coefficient state.
    pub(crate) fn refresh_normaliser(&self, states: &[ParameterBlockState]) -> Result<RiskSetCentring, EventHistoryError> {
        self.computed_reference(states)
    }

    /// Start of an evaluation: a refusal an earlier evaluation left is gone.
    fn clear_reference_refusal(&self) {
        if let Ok(mut refusal) = self.reference_refusal.lock() {
            *refusal = None;
        }
    }

    /// The reference refusal of the evaluation that just failed, taken.
    fn take_reference_refusal(&self) -> Option<EventHistoryError> {
        self.reference_refusal.lock().ok().and_then(|mut refusal| refusal.take())
    }

    /// Per atom, the offset within the latent block of its rate coefficient,
    /// or `None` for a held rate. The loadings come first, then the free
    /// rates in atom order.
    fn free_rate_slots(&self) -> Vec<Option<usize>> {
        let mut next = self.marks() * self.atoms;
        self.held_rates
            .iter()
            .map(|held| {
                held.is_none().then(|| {
                    let slot = next;
                    next += 1;
                    slot
                })
            })
            .collect()
    }

    /// Every atom's dimensionless rate `ν` at a latent block state: the
    /// chart of the coefficient for a free rate, the held value otherwise.
    pub(crate) fn atom_rates(&self, latent_beta: &Array1<f64>) -> Vec<f64> {
        self.free_rate_slots()
            .iter()
            .zip(self.held_rates.iter())
            .map(|(slot, held)| match slot {
                Some(slot) => rate_from_chart(self.rate_band, &latent_beta[*slot]),
                None => held.expect("a rate without a coefficient is held"),
            })
            .collect()
    }

    /// The band of dimensionless rates the cohort's breakpoints resolve, `(ν_min, ν_max)`.
    pub fn rate_band(&self) -> (f64, f64) {
        self.rate_band
    }

    /// Whether each atom's rate is held on a plateau of the residuals' resolution.
    pub fn rate_held(&self) -> Vec<bool> {
        self.held_rates.iter().map(Option::is_some).collect()
    }

    pub fn marks(&self) -> usize {
        self.nodes.marks
    }

    pub fn atoms(&self) -> usize {
        self.atoms
    }

    pub fn time_scale(&self) -> f64 {
        self.time_scale
    }

    pub fn gauss_hermite_order(&self) -> usize {
        self.gh.order
    }

    pub fn nodes(&self) -> &Arc<CohortNodes> {
        &self.nodes
    }

    pub(crate) fn gauss_hermite(&self) -> &Arc<GaussHermite> {
        &self.gh
    }

    /// Width of the latent block: the loadings, then the log-rates of the
    /// atoms whose rates are coefficients.
    pub(crate) fn latent_width(&self) -> usize {
        self.marks() * self.atoms + self.held_rates.iter().filter(|h| h.is_none()).count()
    }

    /// Whether the fit carries a latent block (no atoms means a plain
    /// Poisson-process GAM with the same node expansion).
    pub(crate) fn has_latent_block(&self) -> bool {
        self.atoms > 0
    }

    fn block_widths(&self) -> Vec<usize> {
        let mut widths: Vec<usize> = self.designs.iter().map(|d| d.ncols()).collect();
        if self.has_latent_block() {
            widths.push(self.latent_width());
        }
        widths
    }

    fn block_offsets(&self) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(self.marks() + 2);
        let mut acc = 0;
        for w in self.block_widths() {
            offsets.push(acc);
            acc += w;
        }
        offsets.push(acc);
        offsets
    }

    /// Total coefficient count: every mark block then the latent block.
    pub fn total_width(&self) -> usize {
        self.block_widths().iter().sum()
    }

    fn validate_states(&self, states: &[ParameterBlockState]) -> Result<(), String> {
        let marks = self.marks();
        let expected = marks + usize::from(self.has_latent_block());
        if states.len() != expected {
            return Err(format!(
                "event-history family expects {expected} blocks (one per mark{}), got {}",
                if self.has_latent_block() {
                    " plus the latent block"
                } else {
                    ""
                },
                states.len()
            ));
        }
        for (d, state) in states.iter().take(marks).enumerate() {
            if state.eta.len() != self.nodes.total_nodes {
                return Err(format!(
                    "mark {d} predictor has length {}, expected {} nodes",
                    state.eta.len(),
                    self.nodes.total_nodes
                ));
            }
            if state.beta.len() != self.designs[d].ncols() {
                return Err(format!(
                    "mark {d} has {} coefficients, expected {}",
                    state.beta.len(),
                    self.designs[d].ncols()
                ));
            }
        }
        if self.has_latent_block() && states[marks].beta.len() != self.latent_width() {
            return Err(format!(
                "latent block has {} coefficients, expected {}",
                states[marks].beta.len(),
                self.latent_width()
            ));
        }
        Ok(())
    }

    /// The exact state a joint evaluation is keyed on: every coefficient and
    /// every node predictor, bit for bit. A hash alone would turn a collision
    /// into a silently wrong likelihood.
    fn state_key(states: &[ParameterBlockState]) -> Vec<f64> {
        let mut key = Vec::new();
        for state in states {
            key.extend(state.beta.iter().copied());
            key.extend(state.eta.iter().copied());
        }
        key
    }

    /// Value, gradient and Hessian (of the log-likelihood) in coefficient
    /// space, generic in the scalar so a seeded scalar yields directional
    /// derivatives.
    fn evaluate_generic<S: Directional>(
        &self,
        states: &[ParameterBlockState],
        u: Option<&Array1<f64>>,
        v: Option<&Array1<f64>>,
        derivatives: bool,
    ) -> Result<(S, Vec<S>, Vec<S>), String> {
        self.validate_states(states)?;
        if self.differentiates_the_computed_path() {
            return self.computed_joint(states, u, v, derivatives);
        }
        let marks = self.marks();
        let atoms = self.atoms;
        let offsets = self.block_offsets();
        let total = self.total_width();
        let latent_offset = offsets[marks];
        let empty = Array1::<f64>::zeros(0);
        let latent_beta: &Array1<f64> = if self.has_latent_block() {
            &states[marks].beta
        } else {
            &empty
        };
        for direction in [u, v].into_iter().flatten() {
            if direction.len() != total {
                return Err(format!(
                    "event-history direction has length {}, expected {total}",
                    direction.len()
                ));
            }
            if let Some((index, value)) = direction
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(format!(
                    "event-history direction contains non-finite value {value} at coefficient {index}",
                ));
            }
        }
        let component =
            |dir: Option<&Array1<f64>>, index: usize| -> f64 { dir.map_or(0.0, |d| d[index]) };
        let loadings: Vec<S> = (0..marks * atoms)
            .map(|q| {
                S::seeded(
                    latent_beta[q],
                    component(u, latent_offset + q),
                    component(v, latent_offset + q),
                )
            })
            .collect();
        let slots = self.free_rate_slots();
        let band = self.rate_band;
        let charts: Vec<Option<S>> = (0..atoms)
            .map(|k| {
                slots[k].map(|q| {
                    S::seeded(
                        latent_beta[q],
                        component(u, latent_offset + q),
                        component(v, latent_offset + q),
                    )
                })
            })
            .collect();
        let rates: Vec<S> = (0..atoms)
            .map(|k| match charts[k].as_ref() {
                Some(chart) => rate_from_chart(band, chart),
                None => S::seeded(
                    self.held_rates[k].expect("a rate without a coefficient is held"),
                    0.0,
                    0.0,
                ),
            })
            .collect();
        // A subject's local derivatives are laid out with one rate slot per
        // atom; a held rate has no coefficient, so its slot is dropped.
        let local_total = latent_offset + marks * atoms + atoms;
        let global_index: Vec<Option<usize>> = (0..local_total)
            .map(|q| {
                if q < latent_offset + marks * atoms {
                    Some(q)
                } else {
                    slots[q - latent_offset - marks * atoms].map(|slot| latent_offset + slot)
                }
            })
            .collect();
        let designs = &self.designs;
        let subjects = &self.nodes.subjects;
        let gh = &self.gh;
        let time_scale = self.time_scale;
        let row_direction = |dir: Option<&Array1<f64>>, d: usize, row: usize| -> f64 {
            dir.map_or(0.0, |dir| {
                let design = &designs[d];
                let mut acc = 0.0;
                for (j, x) in design.row(row).iter().enumerate() {
                    acc += x * dir[offsets[d] + j];
                }
                acc
            })
        };
        let per_subject: Result<Vec<(S, Vec<S>, Vec<S>)>, String> = subjects
            .par_iter()
            .map(|subject| {
                let n = subject.len();
                let first = subject.first_row;
                let mut eta0 = Vec::with_capacity(n * marks);
                for node in 0..n {
                    let row = first + node;
                    for d in 0..marks {
                        eta0.push(S::seeded(
                            states[d].eta[row],
                            row_direction(u, d, row),
                            row_direction(v, d, row),
                        ));
                    }
                }
                let views: Vec<ArrayView2<'_, f64>> = designs
                    .iter()
                    .map(|design| design.slice(s![first..first + n, ..]))
                    .collect();
                let inputs = SubjectInputs {
                    nodes: subject,
                    eta0: &eta0,
                    loadings: &loadings,
                    rates: &rates,
                    time_scale,
                    gh,
                    continuation_gap: 0.0,
                    designs: derivatives.then_some(views.as_slice()),
                    log_normaliser: None,
                };
                let local = subject_marginal(&inputs, derivatives).map_err(|e| e.to_string())?;
                if derivatives
                    && (local.gradient.len() != local_total
                        || local.hessian.len() != local_total * local_total)
                {
                    return Err(format!(
                        "subject {} produced {} gradient entries for {local_total} local coordinates",
                        first,
                        local.gradient.len()
                    ));
                }
                Ok((local.loglik, local.gradient, local.hessian))
            })
            .collect();
        let per_subject = per_subject?;
        let zero = per_subject
            .first()
            .map(|(l, _, _)| l.constant_like(0.0))
            .ok_or_else(|| "event-history family has no subjects".to_string())?;
        let subject_logliks: Vec<S> = per_subject.iter().map(|(l, _, _)| l.clone()).collect();
        let loglik = pairwise_sum(&subject_logliks, &zero);
        let mut gradient = vec![zero.clone(); if derivatives { total } else { 0 }];
        let mut hessian = vec![zero.clone(); if derivatives { total * total } else { 0 }];
        if derivatives {
            for (_, g, hh) in &per_subject {
                for (q, x) in g.iter().enumerate() {
                    if let Some(gq) = global_index[q] {
                        gradient[gq] = gradient[gq].add(x);
                    }
                }
                for q in 0..local_total {
                    let Some(gq) = global_index[q] else { continue };
                    for r in 0..local_total {
                        let Some(gr) = global_index[r] else { continue };
                        let acc = &mut hessian[gq * total + gr];
                        *acc = acc.add(&hh[q * local_total + r]);
                    }
                }
            }
            // The rate slots, from the rate to its chart coordinate: with
            // `ν = ν(u)`, `∂ℓ/∂u = ∂ℓ/∂ν · ν'`, `∂²ℓ/∂u² = ∂²ℓ/∂ν² · ν'² +
            // ∂ℓ/∂ν · ν''`, `∂²ℓ/∂u∂x = ∂²ℓ/∂ν∂x · ν'`. The factors are
            // jets of the seeded coordinate, so every directional channel of
            // the conversion rides along.
            let chart_slots: Vec<(usize, S, S)> = charts
                .iter()
                .zip(slots.iter())
                .filter_map(|(chart, slot)| {
                    chart.as_ref().zip(*slot).map(|(chart, slot)| {
                        let (first, second) = rate_chart_derivatives(band, chart);
                        (latent_offset + slot, first, second)
                    })
                })
                .collect();
            let rate_gradients: Vec<S> = chart_slots
                .iter()
                .map(|(slot, _, _)| gradient[*slot].clone())
                .collect();
            for (a, (slot_a, first_a, second_a)) in chart_slots.iter().enumerate() {
                for (b, (slot_b, first_b, _)) in chart_slots.iter().enumerate() {
                    let raw = hessian[slot_a * total + slot_b].clone();
                    hessian[slot_a * total + slot_b] = if a == b {
                        raw.mul(first_a)
                            .mul(first_a)
                            .add(&rate_gradients[a].mul(second_a))
                    } else {
                        raw.mul(first_a).mul(first_b)
                    };
                }
                for q in 0..total {
                    if chart_slots.iter().any(|(slot, _, _)| *slot == q) {
                        continue;
                    }
                    let value = hessian[slot_a * total + q].mul(first_a);
                    hessian[slot_a * total + q] = value.clone();
                    hessian[q * total + slot_a] = value;
                }
                gradient[*slot_a] = rate_gradients[a].mul(first_a);
            }
        }
        Ok((loglik, gradient, hessian))
    }

    /// Exact gradient of the computed log-likelihood in coefficient space.
    ///
    /// The forward filter is replayed on forward-mode duals seeded with the
    /// design rows, so every tangent slot is the derivative of the very
    /// arithmetic that produced the value. The Fisher-identity gradient of the
    /// exact marginal differs from this by the quadrature error, and a
    /// trust-region Newton with a value-based acceptance test cannot converge
    /// on a gradient that is not the derivative of the value it tests.
    fn exact_gradient(&self, states: &[ParameterBlockState]) -> Result<Vec<f64>, String> {
        self.clear_reference_refusal();
        let total = self.total_width();
        let mut gradient = vec![0.0; total];
        self.exact_gradient_chunks::<{ super::scalar::TANGENT_WIDTH }>(states, &mut gradient)?;
        Ok(gradient)
    }

    /// One `W`-wide sweep of tangent slots at a time over the coefficient
    /// vector, each sweep a full forward filter per subject.
    fn exact_gradient_chunks<const W: usize>(
        &self,
        states: &[ParameterBlockState],
        gradient: &mut [f64],
    ) -> Result<(), String> {
        let values: Vec<f64> = states.iter().flat_map(|s| s.beta.iter().copied()).collect();
        for start in (0..values.len()).step_by(W) {
            let beta: Vec<Tangent<W>> = values.iter().enumerate().map(|(q, value)| {
                let mut grad = [0.0; W];
                if q >= start && q < start + W { grad[q - start] = 1.0; }
                Tangent::seeded(*value, grad)
            }).collect();
            let result = self.path_value(states, &beta)?;
            for (slot, value) in result.grad.iter().enumerate().take((values.len() - start).min(W)) {
                gradient[start + slot] = *value;
            }
        }
        Ok(())
    }

    /// The log-likelihood and its derivative along one direction of the
    /// joint coefficient vector, by one forward filter on a one-slot tangent
    /// seeded with the direction.
    pub(crate) fn directional_log_likelihood(
        &self,
        states: &[ParameterBlockState],
        direction: &Array1<f64>,
    ) -> Result<(f64, f64), EventHistoryError> {
        self.validate_states(states)
            .map_err(|reason| EventHistoryError::InvalidInput { reason })?;
        let values: Vec<f64> = states.iter().flat_map(|s| s.beta.iter().copied()).collect();
        if direction.len() != values.len() || direction.iter().any(|x| !x.is_finite()) {
            return Err(EventHistoryError::InvalidInput {
                reason: "invalid event-history derivative direction".to_string(),
            });
        }
        let beta: Vec<Tangent<1>> = values.iter().enumerate().map(|(q, value)|
            Tangent::seeded(*value, [direction[q]])).collect();
        let result = self.path_value(states, &beta)?;
        Ok((result.value, result.grad[0]))
    }

    /// The martingale residuals of every subject at `states`: per node and
    /// mark the compensated score `s = y − w E[λ | past]` and its curvature
    /// `c = w E[λ | past]`, both under the FILTERED density, which is what
    /// makes them the increments of a martingale — uncorrelated under the
    /// null, with `Σ_n c_n` as the exact predictable variation the
    /// covariance score subtracts.
    pub(crate) fn residuals(
        &self,
        states: &[ParameterBlockState],
    ) -> Result<Vec<SubjectResiduals>, String> {
        self.validate_states(states)?;
        let marks = self.marks();
        let atoms = self.atoms;
        let empty = Array1::<f64>::zeros(0);
        let latent: &Array1<f64> = if self.has_latent_block() {
            &states[marks].beta
        } else {
            &empty
        };
        let loadings: Vec<f64> = latent.iter().take(marks * atoms).copied().collect();
        let rates: Vec<f64> = self.atom_rates(latent);
        let centring = if let Some(reference) = self.reference.as_ref() {
            let values = self.refresh_normaliser(states)?;
            Some(reference.carry_to_nodes(
                &values.log_normaliser, self.marks(), self.nodes.total_nodes)
                .map_err(|error| error.to_string())?)
        } else { None };
        let held = centring.as_deref();
        let gh = &self.gh;
        let time_scale = self.time_scale;
        let all_marks = vec![true; marks];
        self.nodes
            .subjects
            .par_iter()
            .map(|subject| {
                let n = subject.len();
                let first = subject.first_row;
                let mut eta0 = Vec::with_capacity(n * marks);
                for node in 0..n {
                    for d in 0..marks {
                        eta0.push(states[d].eta[first + node]);
                    }
                }
                let normaliser: Option<Vec<f64>> =
                    held.map(|values| values[first * marks..(first + n) * marks].to_vec());
                let inputs = SubjectInputs {
                    nodes: subject,
                    eta0: &eta0,
                    loadings: &loadings,
                    rates: &rates,
                    time_scale,
                    gh,
                    continuation_gap: 0.0,
                    designs: None,
                    log_normaliser: normaliser.as_deref(),
                };
                let pass = forward_filter(&inputs, None, &all_marks).map_err(|e| e.to_string())?;
                let mut scores = Vec::with_capacity(n * marks);
                let mut curvatures = Vec::with_capacity(n * marks);
                for node in 0..n {
                    let intensities = expected_intensities(
                        &pass.grids[node],
                        &pass.predicted[node],
                        &eta0[node * marks..(node + 1) * marks],
                        &loadings,
                        normaliser.as_ref().map(|m| &m[node * marks..(node + 1) * marks]),
                        marks,
                        atoms,
                    );
                    for d in 0..marks {
                        let c = subject.exposures[[node, d]] * intensities[d];
                        scores.push(subject.counts[[node, d]] - c);
                        curvatures.push(c);
                    }
                }
                Ok(SubjectResiduals {
                    times: subject.times.clone(),
                    scores,
                    curvatures,
                })
            })
            .collect()
    }

    /// Full `f64` joint evaluation, cached on the state: the value, its exact
    /// gradient, and the Louis-identity Hessian.
    ///
    /// The Hessian is Louis' identity evaluated by the same quadrature as the
    /// value; it agrees with the second derivative of the computed value to
    /// the quadrature error the fit's certificate bounds. The gradient is the
    /// exact derivative of the computed value. A Newton iteration with an
    /// exact gradient and a Hessian accurate to a small relative error
    /// converges at that relative rate, and the outer LAML's log-determinant
    /// term sees the same Hessian its directional derivatives are taken of.
    pub(crate) fn joint_evaluation(
        &self,
        states: &[ParameterBlockState],
    ) -> Result<Arc<JointEvaluation>, String> {
        self.clear_reference_refusal();
        let key = Self::state_key(states);
        if let Ok(guard) = self.cache.lock()
            && let Some((k, value)) = guard.as_ref()
            && *k == key
        {
            return Ok(Arc::clone(value));
        }
        let (loglik, computed_gradient, hessian) = self.evaluate_generic::<f64>(states, None, None, true)?;
        let gradient = if self.reference.is_some() && self.atoms > 0 { computed_gradient } else { self.exact_gradient(states)? };
        let total = self.total_width();
        let mut negative_hessian = Array2::<f64>::zeros((total, total));
        for i in 0..total {
            for j in 0..total {
                negative_hessian[[i, j]] = -hessian[i * total + j];
            }
        }
        let evaluation = Arc::new(JointEvaluation {
            log_likelihood: loglik,
            gradient: Array1::from(gradient),
            hessian: negative_hessian,
        });
        if let Ok(mut guard) = self.cache.lock() {
            *guard = Some((key, Arc::clone(&evaluation)));
        }
        Ok(evaluation)
    }

    /// Log-likelihood only (forward filter, no derivatives).
    pub fn log_likelihood(&self, states: &[ParameterBlockState]) -> Result<f64, String> {
        self.clear_reference_refusal();
        let (loglik, _, _) = self.evaluate_generic::<f64>(states, None, None, false)?;
        Ok(loglik)
    }

    /// `D_β H[u]` for the negative log-likelihood Hessian `H`.
    pub(crate) fn directional_hessian(
        &self,
        states: &[ParameterBlockState],
        u: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.clear_reference_refusal();
        let total = self.total_width();
        let (_, _, hessian) = self.evaluate_generic::<OneSeed<0>>(states, Some(u), None, true)?;
        let mut out = Array2::<f64>::zeros((total, total));
        for i in 0..total {
            for j in 0..total {
                out[[i, j]] = -hessian[i * total + j].eps();
            }
        }
        Ok(out)
    }

    /// `D²_β H[u, v]` for the negative log-likelihood Hessian `H`.
    pub(crate) fn second_directional_hessian(
        &self,
        states: &[ParameterBlockState],
        u: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.clear_reference_refusal();
        let total = self.total_width();
        let (_, _, hessian) =
            self.evaluate_generic::<TwoSeed<0>>(states, Some(u), Some(v), true)?;
        let mut out = Array2::<f64>::zeros((total, total));
        for i in 0..total {
            for j in 0..total {
                out[[i, j]] = -hessian[i * total + j].eps_del();
            }
        }
        Ok(out)
    }
}

/// The band `(ν_min, ν_max)` of dimensionless rates the cohort's own
/// breakpoints resolve: [`CohortNodes::rate_band`] in units of `T̄`, the
/// same at every mesh refinement, so a refinement never moves the chart
/// under the coefficient that lives in it.
fn rate_band(nodes: &CohortNodes, time_scale: f64) -> Result<(f64, f64), EventHistoryError> {
    nodes
        .rate_band
        .map(|(lower, upper)| (lower * time_scale, upper * time_scale))
        .ok_or_else(|| EventHistoryError::InvalidInput {
            reason:
                "the cohort admits no band of latent rates: no subject has two distinct breakpoints"
                    .to_string(),
        })
}

/// The chart of the rate band: `ν(u) = ν_min + (ν_max − ν_min) · u² / (1 + u²)`.
pub(crate) fn rate_from_chart<S: JetField>(band: (f64, f64), u: &S) -> S {
    let (lower, upper) = band;
    let square = u.mul(u);
    let fraction = square.mul(&recip(&add_real(&square, 1.0)));
    add_real(&fraction.scale(upper - lower), lower)
}

/// `ν'(u)` and `ν''(u)` of [`rate_from_chart`]:
/// `ν' = 2Δ u / (1 + u²)²`, `ν'' = 2Δ (1 − 3u²) / (1 + u²)³`.
fn rate_chart_derivatives<S: JetField>(band: (f64, f64), u: &S) -> (S, S) {
    let delta = band.1 - band.0;
    let square = u.mul(u);
    let inverse = recip(&add_real(&square, 1.0));
    let inverse2 = inverse.mul(&inverse);
    let first = u.mul(&inverse2).scale(2.0 * delta);
    let second = add_real(&square.scale(-3.0), 1.0)
        .mul(&inverse2)
        .mul(&inverse)
        .scale(2.0 * delta);
    (first, second)
}

/// The inverse chart: the coordinate `u ≥ 0` at which [`rate_from_chart`]
/// returns `ν`, with `ν` clamped into the band.
pub(crate) fn rate_chart(band: (f64, f64), rate: f64) -> f64 {
    let (lower, upper) = band;
    let fraction = ((rate - lower) / (upper - lower)).clamp(0.0, 1.0 - f64::EPSILON);
    (fraction / (1.0 - fraction)).sqrt()
}

impl CustomFamily for EventHistoryFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let joint = self.joint_evaluation(block_states)?;
        let offsets = self.block_offsets();
        let mut blockworking_sets = Vec::with_capacity(offsets.len() - 1);
        for b in 0..offsets.len() - 1 {
            let range = offsets[b]..offsets[b + 1];
            let gradient = joint.gradient.slice(s![range.clone()]).to_owned();
            let hessian = joint.hessian.slice(s![range.clone(), range]).to_owned();
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        Ok(FamilyEvaluation {
            log_likelihood: joint.log_likelihood,
            blockworking_sets,
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        self.log_likelihood(block_states)
    }

    fn classical_deviance(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<f64>, String> {
        self.validate_states(block_states)?;
        Ok(None)
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    /// A family whose derivatives come from the computed path streams its
    /// inner Newton curvature as Hessian-vector products, and hands the matrix
    /// only to the consumers that factor or read it (#2965).
    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        specs.len() == self.marks() + usize::from(self.has_latent_block())
            && self.differentiates_the_computed_path()
    }

    fn inner_joint_workspace_gradient_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        self.inner_coefficient_hessian_hvp_available(specs)
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn gam_model_api::families::custom_family::ExactNewtonJointHessianWorkspace>>, String> {
        if !self.inner_coefficient_hessian_hvp_available(specs) {
            return Ok(None);
        }
        self.validate_states(block_states)?;
        Ok(Some(Arc::new(objective::ComputedHessianWorkspace::new(self.clone(), block_states.to_vec()))))
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn inner_coefficient_objective_is_globally_convex(&self) -> bool {
        false
    }

    /// The marginal likelihood is not log-concave in the loadings: at zero
    /// loading the profile can be locally convex (a saddle of the negative
    /// log-likelihood), so the joint Newton needs the self-vanishing
    /// Levenberg–Marquardt damping the sibling latent families use.
    fn levenberg_on_ill_conditioning(&self) -> bool {
        true
    }

    fn output_channel_assignment(&self, specs: &[ParameterBlockSpec]) -> Option<Vec<usize>> {
        Some((0..specs.len()).collect())
    }

    fn block_coefficient_coordinate(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        block_spec: &ParameterBlockSpec,
    ) -> CoefficientCoordinate {
        // The family owns the chain rule from node predictors to coefficients
        // through its stored designs, so no block may be reparameterised.
        if block_index >= block_states.len() || block_spec.name.is_empty() {
            return CoefficientCoordinate::Structural;
        }
        CoefficientCoordinate::Structural
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(self.joint_evaluation(block_states)?.hessian.clone()))
    }

    fn exact_newton_joint_loglik_gradient(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(self.joint_evaluation(block_states)?.gradient.clone()))
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        if specs.len() != block_states.len() {
            return Err(format!(
                "event-history joint gradient: {} specs for {} block states",
                specs.len(),
                block_states.len()
            ));
        }
        let joint = self.joint_evaluation(block_states)?;
        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood: joint.log_likelihood,
            gradient: joint.gradient.clone(),
        }))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(self.directional_hessian(block_states, d_beta_flat)?))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(self.second_directional_hessian(
            block_states,
            d_beta_u_flat,
            d_betav_flat,
        )?))
    }
}

/// Fit specification for an event-history model.
#[derive(Clone)]
pub(crate) struct EventHistorySpec {
    /// One covariate/time term collection per mark, or a single one shared by
    /// every mark. Feature columns index the node data matrix: the covariate
    /// table's columns followed by the node time.
    pub covariates: Vec<TermCollectionSpec>,
    /// Gauss-Legendre order per mesh cell.
    pub quadrature_order: usize,
    /// Starting Gauss-Hermite order per latent axis.
    pub gauss_hermite_order: usize,
    /// The certificate's tolerance: the largest shift any fitted coefficient
    /// is estimated to make under a refinement of the Gauss-Hermite order
    /// or time mesh, in units of its posterior standard deviation. This is
    /// a local first-order stationarity check, not a bound on every forecast
    /// or on the posterior approximation. The default is 0.05.
    pub quadrature_tolerance: f64,
    /// The reference population every mark's baseline is the marginal rate
    /// of. `None` centres the latent term on the stationary prior, so
    /// `exp(η⁰)` is the intensity averaged over everybody the cohort started
    /// with; `Some` centres it on the risk set at every age, so `exp(η⁰)` is
    /// the incidence among those still at risk (see `super::preserve`).
    pub reference: Option<ReferenceStrata>,
    /// Required maximum coarse/fine discrepancy in log reference moments
    /// and log risk masses, evaluated at the same coefficient state.
    pub reference_tolerance: f64,
    pub options: BlockwiseFitOptions,
}

impl EventHistorySpec {
    pub fn new(covariates: Vec<TermCollectionSpec>) -> Self {
        Self {
            covariates,
            quadrature_order: super::cohort::quadrature_order_for_degree(3),
            gauss_hermite_order: 9,
            quadrature_tolerance: 5e-2,
            reference: None,
            reference_tolerance: 1e-4,
            options: BlockwiseFitOptions::default(),
        }
    }
}

/// One stationarity check of the fitted coefficients under a refinement.
///
/// The shift is the exact first-order move of the penalised mode: at the
/// fitted coefficients the current setting is stationary, so the refined
/// setting's mode sits at `β + V (g' − g)` with `g'` and `g` the two
/// settings' exact gradients at the same `β` and `V` the fit's own posterior
/// covariance — the inverse of the penalised Hessian, which is the operator
/// that turns a gradient discrepancy into a coefficient one. Measuring it
/// this way costs one gradient per candidate instead of a second fit, and it
/// is reported in the units the shift has to be judged in: posterior
/// standard deviations.
#[derive(Clone, Debug)]
pub struct RefinementCheck {
    /// The refined setting that was checked (a Gauss-Hermite order or a mesh
    /// refinement level).
    pub candidate: usize,
    /// The largest coefficient shift under the refinement, in posterior
    /// standard deviations.
    pub coefficient_shift: f64,
    /// The log-likelihood at the fitted coefficients under the refinement.
    pub log_likelihood: f64,
}

/// Certificate that the fitted coefficients are stationary under a
/// refinement of the latent quadrature and of the time mesh.
#[derive(Clone, Debug)]
pub struct QuadratureCertificate {
    pub gauss_hermite_order: usize,
    pub mesh_refinement: usize,
    pub log_likelihood: f64,
    /// The check against the next Gauss-Hermite order (`2·order − 1`).
    pub gauss_hermite: RefinementCheck,
    /// The check against the mesh with every cell halved.
    pub mesh: RefinementCheck,
}

/// A fitted event-history model.
pub struct EventHistoryFit {
    pub nodes: Arc<CohortNodes>,
    pub family: EventHistoryFamily,
    pub fit: UnifiedFitResult,
    pub mark_kinds: Vec<MarkKind>,
    /// Frozen per-mark term collections for prediction.
    pub frozen_specs: Vec<TermCollectionSpec>,
    pub designs: Vec<TermCollectionDesign>,
    /// `C(0) = E[A Aᵀ | data]`: the posterior mean of the covariance across
    /// marks of the latent log-intensity deviations at one time, the latent
    /// object the model identifies. Each atom contributes its mode
    /// `â_k â_kᵀ` plus the posterior spread of its loadings, so an atom whose
    /// evidence lives in that spread still carries variance here.
    pub covariance: Array2<f64>,
    /// `E[a_k a_kᵀ | data]` per atom, in the order of `loadings`' columns;
    /// their sum is `covariance`, and each decays with its own rate.
    pub atom_covariances: Vec<Array2<f64>>,
    /// Eigenvalues of `covariance`, descending.
    pub eigenvalues: Array1<f64>,
    /// The posterior standard deviation of each eigenvalue, from the fit's
    /// own posterior covariance of the loadings through the first-order
    /// eigenvalue perturbation: what says whether a direction is resolved.
    pub eigenvalue_sd: Array1<f64>,
    /// Unit eigenvectors of `covariance`, as columns matching `eigenvalues`.
    pub eigenvectors: Array2<f64>,
    /// The participation ratio `(tr C)² / tr(C²)`: the continuous count of
    /// directions `covariance` uses.
    pub effective_rank: f64,
    /// Factor coordinates `a[d, k]` of the covariance at the posterior mode,
    /// in the canonical gauge: atoms ordered by rate (slowest first) and each
    /// column signed so that its largest entry is positive. With distinct
    /// rates the temporal covariance identifies each atom up to that gauge;
    /// at equal rates only the covariance is identified.
    pub loadings: Array2<f64>,
    /// `ln(rate_k · time_scale)`.
    pub log_rates: Vec<f64>,
    /// Rates in the data's time unit.
    pub rates: Vec<f64>,
    /// Whether each atom's rate sits at a limit of the mesh's resolution — a
    /// static frailty at the slow end, the mesh's own spacing at the fast
    /// end — where the likelihood is flat in it, so it was held there rather
    /// than fitted.
    pub rate_held: Vec<bool>,
    /// `ln λ_k`: the precision of each atom's empirical-Bayes loading prior,
    /// chosen by the evidence when the atom entered.
    pub atom_log_lambdas: Vec<f64>,
    pub time_scale: f64,
    /// Gauss-Legendre order per mesh cell used for the training nodes.
    pub quadrature_order: usize,
    /// Mesh refinement level of the training nodes.
    pub mesh_refinement: usize,
    pub quadrature: QuadratureCertificate,
    /// Every rank step the evidence judged, in order.
    pub rank_path: Vec<RankStep>,
    /// The decrease of the outer LAML criterion each accepted atom brought.
    pub atom_evidence: Vec<f64>,
    /// Summed time and latent-order discrepancies at fixed coefficients, before each
    /// refinement. Every returned reference fit meets reference_tolerance.
    pub reference_refinements: Vec<f64>,
    /// Authoritative centring at the final coefficient state.
    pub centring: Option<RiskSetCentring>,
    /// Sum of the fixed-parameter time and latent-order discrepancies; absent for prior centring.
    pub reference_certificate: Option<f64>,
}

impl EventHistoryFit {
    /// `log M_d(t)` for one stratum at an arbitrary time, by the same linear
    /// interpolation in the log of the normaliser the fit used. Empty when
    /// the baselines are centred on the stationary prior.
    pub fn risk_set_normaliser_at(&self, stratum: usize, t: f64) -> Result<Vec<f64>, EventHistoryError> {
        let marks = self.marks();
        let Some(snapshot) = self.centring.as_ref() else {
            if stratum != 0 {
                return Err(EventHistoryError::InvalidInput { reason: "a model without reference strata requires stratum zero".to_string() });
            }
            return Ok(Vec::new());
        };
        if stratum >= snapshot.profiles.nrows() {
            return Err(EventHistoryError::InvalidInput {
                reason: format!("reference stratum {stratum} is outside 0..{}", snapshot.profiles.nrows()),
            });
        }
        let grid = &snapshot.grid;
        let nodes = grid.len();
        let (lower, weight) = grid.locate(t)?;
        let base = stratum * nodes;
        Ok((0..marks)
            .map(|d| {
                let low = snapshot.log_normaliser[(base + lower) * marks + d];
                let high = snapshot.log_normaliser[(base + lower + 1) * marks + d];
                low + weight * (high - low)
            })
            .collect())
    }

    /// The rank of the latent covariance the fit carries: the rank the
    /// evidence supports, unless [`Self::unresolved_growth`] reports that the
    /// path stopped on a decision it could not resolve.
    pub fn rank(&self) -> usize {
        self.log_rates.len()
    }

    /// The growth the rank path could not resolve, when it stopped for that
    /// reason rather than on the evidence.
    pub fn unresolved_growth(&self) -> Option<&UnresolvedGrowth> {
        self.rank_path.last().and_then(|step| step.growth_unresolved.as_ref())
    }

    /// `C(Δ) = Σ_k E[a_k a_kᵀ] e^{−r_k |Δ|}`: the latent covariance across a
    /// lag of `lag` time units.
    pub fn temporal_covariance(&self, lag: f64) -> Array2<f64> {
        super::covariance::temporal_covariance(
            self.marks(),
            &self.atom_covariances,
            &self.rates,
            lag,
        )
    }

    pub fn marks(&self) -> usize {
        self.nodes.marks
    }

    /// Fitted coefficients of the mark-`d` block: the coefficients of the
    /// population log-intensity surface `η⁰`. `exp(η⁰)` is the intensity
    /// averaged over the latent state, since the latent term enters as
    /// `−½|a_d|² + a_d · z`, whose Gaussian mixing the shift cancels exactly
    /// (`docs/event-history.md` derives it).
    pub fn mark_coefficients(&self, d: usize) -> &Array1<f64> {
        &self.fit.block_states[d].beta
    }
}

fn identity_pattern_design(n_obs: usize, width: usize) -> Result<Array2<f64>, EventHistoryError> {
    if n_obs < width {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "the cohort has {n_obs} nodes but the latent block needs at least {width}"
            ),
        });
    }
    let mut design = Array2::<f64>::zeros((n_obs, width));
    for i in 0..width {
        design[[i, i]] = 1.0;
    }
    Ok(design)
}

/// The latent block: loadings then log-rates. Each atom's loadings carry an
/// isotropic Gaussian prior `a_k ~ N(0, λ_k⁻¹ I)` — the penalty toward no
/// latent effect — whose precision the evidence chose when the atom entered
/// (see [`super::covariance::empirical_bayes_ridge`]); it is held fixed in
/// the fit, so the latent block adds no outer smoothing coordinate. The
/// log-rate is an unpenalised structural coordinate: with the loadings off
/// zero it is identified by the likelihood, and a prior on it would have no
/// "no effect" to point at.
///
/// Every atom starts where the evidence put it: the atoms already carried
/// keep the values the fit one rank down reached, and the new one begins at
/// the posterior mode of its loading under its prior, along the direction
/// and at the rate the covariance score named. Nothing here is a symmetric
/// start a deterministic Newton could leave symmetric.
pub(crate) fn latent_block_spec(
    n_obs: usize,
    marks: usize,
    atoms: usize,
    start: &RankStart,
    band: (f64, f64),
) -> Result<ParameterBlockSpec, EventHistoryError> {
    if atoms == 0 {
        return Err(EventHistoryError::InvalidInput {
            reason: "a latent block needs at least one atom".to_string(),
        });
    }
    let carried = match start.atom {
        Some(_) => atoms - 1,
        None => atoms,
    };
    if start.loadings.len() != marks * carried
        || start.log_rates.len() != carried
        || start.log_lambdas.len() != carried
        || start.rate_held.len() != carried
    {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "a rank start for {atoms} atoms carries {} loadings, {} rates, {} ridges and {} rate flags for {carried} atoms",
                start.loadings.len(),
                start.log_rates.len(),
                start.log_lambdas.len(),
                start.rate_held.len()
            ),
        });
    }
    let held = start.held_rates();
    let free_rates = held.iter().filter(|h| h.is_none()).count();
    let width = marks * atoms + free_rates;
    let design = identity_pattern_design(n_obs, width)?;
    let mut initial_beta = Array1::<f64>::zeros(width);
    let mut log_lambdas = Vec::with_capacity(atoms);
    let mut log_rates = Vec::with_capacity(atoms);
    for d in 0..marks {
        for k in 0..carried {
            initial_beta[d * atoms + k] = start.loadings[d * carried + k];
        }
    }
    for k in 0..carried {
        log_rates.push(start.log_rates[k]);
        log_lambdas.push(start.log_lambdas[k]);
    }
    if let Some(atom) = start.atom.as_ref() {
        if !atom.ridge.accepted || !atom.ridge.log_lambda.is_finite() {
            return Err(EventHistoryError::InvalidInput {
                reason: "a rank start can only grow by an atom the evidence accepted".to_string(),
            });
        }
        for d in 0..marks {
            initial_beta[d * atoms + carried] = atom.loading[d];
        }
        log_rates.push(atom.log_rate);
        log_lambdas.push(atom.ridge.log_lambda);
    }
    let mut slot = marks * atoms;
    for (k, held) in held.iter().enumerate() {
        if held.is_none() {
            initial_beta[slot] = rate_chart(band, log_rates[k].exp());
            slot += 1;
        }
    }
    let mut penalties = Vec::with_capacity(atoms);
    let mut nullspace_dims = Vec::with_capacity(atoms);
    for (k, &log_lambda) in log_lambdas.iter().enumerate() {
        let mut s = Array2::<f64>::zeros((width, width));
        for d in 0..marks {
            s[[d * atoms + k, d * atoms + k]] = 1.0;
        }
        penalties.push(PenaltyMatrix::Dense(s).with_fixed_log_lambda(log_lambda));
        nullspace_dims.push(width - marks);
    }
    Ok(ParameterBlockSpec {
        name: "latent".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(design))),
        offset: Array1::zeros(n_obs),
        penalties,
        nullspace_dims,
        initial_log_lambdas: Array1::from(log_lambdas),
        initial_beta: Some(initial_beta),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    })
}

/// The block of one mark's covariate/time smooths.
pub(crate) fn mark_block_spec(name: &str, design: &TermCollectionDesign) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: name.to_string(),
        design: design.design.clone(),
        offset: design.affine_offset.clone(),
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: Array1::zeros(design.penalties.len()),
        initial_beta: None,
        gauge_priority: 150,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

/// Fit an event-history model from formula right-hand sides such as
/// `x + s(time)`: one formula that every mark uses (with its own
/// coefficients), or one formula per mark in the cohort's mark order, so that
/// a mark's log-intensity carries only the terms that belong to it — a disease
/// its own score, not every score of every other disease. `reference` is the
/// population whose incidence the baselines follow; `None` keeps the
/// stationary prior's centring.
pub fn fit_event_history_formulas<F: AsRef<str>>(
    cohort: &mut EventHistoryCohort,
    formulas: &[F],
    options: BlockwiseFitOptions,
    reference: Option<ReferenceStrata>,
) -> Result<EventHistoryFit, EventHistoryError> {
    cohort.validate()?;
    let marks = cohort.marks();
    if formulas.len() != 1 && formulas.len() != marks {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "expected one formula or one per mark ({marks}), got {}",
                formulas.len()
            ),
        });
    }
    let mut spec = EventHistorySpec::new(Vec::new());
    spec.options = options;
    let rows = design_rows(cohort, spec.quadrature_order)?;
    let mut covariates = Vec::with_capacity(formulas.len());
    for (d, formula) in formulas.iter().enumerate() {
        let terms =
            super::formula::covariate_spec_from_formula(formula.as_ref(), rows.view(), cohort)
                .map_err(|error| {
                    if formulas.len() == 1 {
                        error
                    } else {
                        EventHistoryError::InvalidInput {
                            reason: format!("mark {:?}: {error}", cohort.mark_names[d]),
                        }
                    }
                })?;
        covariates.push(terms);
    }
    spec.covariates = covariates;
    spec.reference = reference;
    fit_event_history(cohort, &spec)
}

/// The family and its block specs at one (order, mesh refinement) setting.
struct Built {
    nodes: Arc<CohortNodes>,
    family: EventHistoryFamily,
    designs: Vec<TermCollectionDesign>,
    dense: Vec<Arc<Array2<f64>>>,
    specs: Vec<ParameterBlockSpec>,
}

impl Built {
    /// Block states at the given per-block coefficients (the node predictors
    /// recomputed from this setting's designs).
    fn states(&self, betas: &[Array1<f64>]) -> Vec<ParameterBlockState> {
        let marks = self.family.marks();
        let mut states = Vec::with_capacity(betas.len());
        for (b, beta) in betas.iter().enumerate() {
            let eta = if b < marks {
                self.dense[b].dot(beta) + &self.designs[b].affine_offset
            } else {
                Array1::zeros(self.nodes.total_nodes)
            };
            states.push(ParameterBlockState {
                beta: beta.clone(),
                eta,
            });
        }
        states
    }
}

/// Bytes the family's evaluation may hold at once: the streamed backward
/// kernel row for one gap, the carried `P × S` conditional expectations,
/// the per-node densities and operators of every node, per parallel
/// subject, in the widest scalar the outer solve uses (sixteen channels
/// for mixed reference sensitivities nested over two outer directions).
fn transient_footprint_bytes(
    order: usize,
    atoms: usize,
    max_nodes: usize,
    marks: usize,
    total_width: usize,
) -> Result<f64, EventHistoryError> {
    let s = product_grid_size(order, atoms)? as f64;
    let g = order as f64;
    let n = max_nodes as f64;
    let per_subject = 4.0 * s
        + total_width as f64 * s
        + n * s * (4.0 + marks as f64)
        + n * atoms as f64 * 3.0 * g * g;
    let channels = 16.0;
    let bytes = 8.0 * channels * per_subject * rayon::current_num_threads() as f64;
    Ok(bytes)
}

fn preflight(
    order: usize,
    atoms: usize,
    max_nodes: usize,
    marks: usize,
    total_width: usize,
) -> Result<(), EventHistoryError> {
    let bytes = transient_footprint_bytes(order, atoms, max_nodes, marks, total_width)?;
    let budget = gam_runtime::resource::ResourcePolicy::default_library()
        .max_single_materialization_bytes as f64;
    if bytes > budget {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "an event-history evaluation at Gauss-Hermite order {order} over {atoms} atoms ({} grid points) with {max_nodes} nodes per subject needs about {:.1} GiB across {} threads, above this machine's {:.1} GiB materialisation budget; offer fewer atoms or a shorter follow-up per subject",
                product_grid_size(order, atoms)?,
                bytes / f64::from(1u32 << 30),
                rayon::current_num_threads(),
                budget / f64::from(1u32 << 30)
            ),
        });
    }
    Ok(())
}

/// What a rank-`K+1` fit starts from: the converged rank-`K` latent block
/// and the atom the covariance score proposed.
pub(crate) struct RankStart {
    /// The incumbent fit's coefficients of every mark block, in mark order,
    /// so a candidate starts its population surfaces where the fit one rank
    /// down left them rather than from zero; empty when nothing is carried.
    pub mark_betas: Vec<Array1<f64>>,
    /// The rank-`K` loadings, `marks × K` row-major.
    pub loadings: Vec<f64>,
    /// `ln(rate · T̄)` of every atom the fit already has.
    pub log_rates: Vec<f64>,
    /// `ln λ_k` of each of those atoms' loading priors.
    pub log_lambdas: Vec<f64>,
    /// Whether each of those atoms' rates is held at a limit of the mesh's
    /// resolution rather than fitted.
    pub rate_held: Vec<bool>,
    /// The atom the covariance score proposed, when this start grows the
    /// rank. `None` re-starts the same rank from its own converged values,
    /// which is what certifying an accepted fit needs.
    pub(crate) atom: Option<NewAtom>,
}

impl RankStart {
    /// A start that carries the given atoms and grows by none.
    pub fn carried(
        mark_betas: Vec<Array1<f64>>,
        loadings: Vec<f64>,
        log_rates: Vec<f64>,
        log_lambdas: Vec<f64>,
        rate_held: Vec<bool>,
    ) -> Self {
        Self {
            mark_betas,
            loadings,
            log_rates,
            log_lambdas,
            rate_held,
            atom: None,
        }
    }

    /// Per atom of the block this start describes, the dimensionless rate
    /// `ν` it is held at, or `None` when the rate is fitted.
    fn held_rates(&self) -> Vec<Option<f64>> {
        let mut held: Vec<Option<f64>> = self
            .rate_held
            .iter()
            .zip(self.log_rates.iter())
            .map(|(&held, &rate)| held.then_some(rate.exp()))
            .collect();
        if let Some(atom) = self.atom.as_ref() {
            held.push(atom.rate_held().then_some(atom.log_rate.exp()));
        }
        held
    }
}

/// One step of the rank path: what the covariance score proposed and what
/// the evidence made of it.
#[derive(Clone, Debug)]
pub struct RankStep {
    /// Rank before the step.
    pub rank: usize,
    /// Top eigenvalue of the differentiated loading curvature at the
    /// proposed rate, after its quadrature-resolution check.
    pub score_eigenvalue: f64,
    /// `μ² / (4J)` of the top direction: its second-order evidence gain in
    /// nats, the matched-filter statistic the rate maximises.
    pub standardised_gain: f64,
    /// Proposed rate in the data's time unit; zero denotes a static factor.
    pub proposed_rate: f64,
    /// The proposal wanted a rate faster than the cohort's breakpoints
    /// resolve and was held at the fastest they do: the residuals carry
    /// structure the design cannot time.
    pub at_resolution_limit: bool,
    /// The rate sits at a limit of the mesh's resolution (a static frailty
    /// at the slow end, the mesh's own spacing at the fast end), where the
    /// likelihood is flat in it, so it is held there rather than fitted.
    pub rate_held: bool,
    /// `ln λ̂`: the precision of the loading prior the evidence chose;
    /// infinite when no finite prior raises the marginal likelihood.
    pub ridge_log_lambda: f64,
    /// Fitted Laplace-criterion gain, or zero when no candidate was fitted.
    /// This is not an exact marginal evidence calculation.
    pub evidence_gain: f64,
    /// The realised increase of the marginal log-likelihood at the mode from
    /// the rank before to the fitted candidate; zero if no fit was attempted.
    pub log_likelihood_gain: f64,
    /// The prior places the loading's posterior mode away from zero, so the
    /// atom was fitted.
    pub accepted: bool,
    /// Whether the rank-`K+1` model reached a certified optimum. A
    /// candidate that cannot be fitted is refused: a fit object may only
    /// come from a converged optimisation, so an atom whose model has no
    /// certified optimum is not one the fit can carry.
    pub converged: bool,
    /// Growth the incumbent's setting could not resolve: set when the path
    /// stopped here because an integral the decision reads is unresolved at
    /// the ladder's top certifiable rung, not because the evidence refused the
    /// atom. The returned model is the certified incumbent, and its rank is
    /// not one the evidence selected over the next.
    pub growth_unresolved: Option<UnresolvedGrowth>,
}

/// An integral a rank decision reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionIntegral {
    /// The added factor's loading curvature at zero loading.
    AddedFactorCurvature,
    /// The likelihood profile along a proposed loading direction.
    DirectionalProfile,
    /// The grown candidate's posterior at the incumbent's setting.
    CandidatePosterior,
}

impl DecisionIntegral {
    pub fn name(self) -> &'static str {
        match self {
            Self::AddedFactorCurvature => "added_factor_curvature",
            Self::DirectionalProfile => "directional_profile",
            Self::CandidatePosterior => "candidate_posterior",
        }
    }
}

/// A rank decision the incumbent's setting could not resolve at the ladder's
/// top certifiable rung.
#[derive(Clone, Debug)]
pub struct UnresolvedGrowth {
    /// The Gauss-Hermite order the incumbent is certified at, above which no
    /// rung is certifiable.
    pub gauss_hermite_order: usize,
    /// The integral that is unresolved there.
    pub integral: DecisionIntegral,
    /// Why it is unresolved.
    pub reason: String,
}

/// The reference population's grid, its per-stratum designs, and where every
/// cohort node sits on that grid.
///
/// Uniform endpoints span the reference window. Every stratum shares these
/// times, with its own covariate profile and contiguous design rows.
fn reference_tables(
    cohort: &EventHistoryCohort,
    strata: &ReferenceStrata,
    frozen_specs: &[TermCollectionSpec],
    quadrature_order: usize,
    refinement: usize,
    nodes: &CohortNodes,
) -> Result<ReferenceTables, EventHistoryError> {
    strata.validate(cohort.subjects.len(), cohort.covariates.nrows())?;
    let marks = cohort.marks();
    let entry = cohort
        .subjects
        .iter()
        .map(|s| s.entry)
        .fold(f64::INFINITY, f64::min);
    let exit = cohort
        .subjects
        .iter()
        .map(|s| s.exit)
        .fold(f64::NEG_INFINITY, f64::max);
    if !(entry.is_finite() && exit.is_finite() && exit > entry) {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "the cohort spans no window to run a reference population over: ({entry}, {exit})"
            ),
        });
    }
    let intervals = quadrature_order.max(2).checked_shl(refinement as u32)
        .filter(|&n| n <= 32768).ok_or_else(|| EventHistoryError::NumericalFailure {
            reason: "reference evolution exceeded its 32768-interval work limit".to_string(),
        })?;
    let times: Vec<f64> = (0..=intervals).map(|n|
        if n == intervals { exit } else { entry + (exit - entry) * n as f64 / intervals as f64 }).collect();
    let grid = ReferenceGrid { gaps: times.windows(2).map(|w| w[1] - w[0]).collect(), times };
    let mut node_data = Array2::<f64>::zeros((strata.strata() * grid.len(), cohort.covariates.ncols() + 1));
    for (s, &profile) in strata.rows.iter().enumerate() {
        for (n, &time) in grid.times.iter().enumerate() {
            let row = s * grid.len() + n;
            for c in 0..cohort.covariates.ncols() { node_data[[row, c]] = cohort.covariates[[profile, c]]; }
            node_data[[row, cohort.covariates.ncols()]] = time;
        }
    }
    let mut designs = Vec::with_capacity(marks);
    let mut offsets = Vec::with_capacity(marks);
    for d in 0..marks {
        let design = build_term_collection_design(node_data.view(), &frozen_specs[d])
            .map_err(|error| EventHistoryError::Fit {
                reason: format!("reference design for mark {d}: {error}"),
            })?;
        let dense = design
            .design
            .try_to_dense_arc("event-history reference design")
            .map_err(|error| EventHistoryError::Fit {
                reason: error.to_string(),
            })?;
        designs.push(dense);
        offsets.push(design.affine_offset.clone());
    }
    // Every cohort node's place on the reference grid: its subject's stratum,
    // the reference node below it and the weight of the one above.
    let total = nodes.total_nodes;
    let mut node_stratum = vec![0usize; total];
    let mut node_lower = vec![0usize; total];
    let mut node_weight = vec![0.0; total];
    for (i, subject) in nodes.subjects.iter().enumerate() {
        for (n, &time) in subject.times.iter().enumerate() {
            let row = subject.first_row + n;
            let (lower, weight) = grid.locate(time)?;
            node_stratum[row] = strata.subject[i];
            node_lower[row] = lower;
            node_weight[row] = weight;
        }
    }
    Ok(ReferenceTables {
        grid,
        profiles: cohort.covariates.select(ndarray::Axis(0), &strata.rows),
        kinds: cohort.mark_kinds.clone(),
        designs,
        offsets,
        strata: strata.rows.len(),
        node_stratum,
        node_lower,
        node_weight,
    })
}

/// Whether a fit at Gauss-Hermite `order` can be certified: its certificate
/// reads the next rung, `2·order − 1`, and that rule's Lagrange interpolant must
/// amplify roundoff over the longest subject by no more than the certificate's
/// tolerance.
fn certifiable(order: usize, max_subject_nodes: usize, tolerance: f64) -> bool {
    GaussHermite::new(2 * order - 1).is_ok_and(|rule| {
        rule.lebesgue_constant * f64::EPSILON * max_subject_nodes as f64 <= tolerance
    })
}

/// The Gauss-Hermite order a decision the grid cannot resolve is raised to,
/// `2·order − 1`, when that rung is itself [`certifiable`]; `None` at the
/// ladder's top certifiable rung.
fn positivity_raise(order: usize, max_subject_nodes: usize, tolerance: f64) -> Option<usize> {
    let next_order = 2 * order - 1;
    certifiable(next_order, max_subject_nodes, tolerance).then_some(next_order)
}

/// The certified fit at ONE rank: the Gauss-Hermite order and the time mesh
/// are refined until no fitted coefficient moves by more than the
/// certificate's tolerance. `start` warm-starts every block from a fit one
/// rank down, with the new atom's loadings at the covariance score's
/// proposal. An unpinned ladder starts at mesh refinement `from_refinement`:
/// a model whose rank was decided at a mesh is never refitted on a coarser
/// one, where the integrals the decision needed resolved are not.
pub(crate) fn fit_at_rank(
    cohort: &EventHistoryCohort,
    spec: &EventHistorySpec,
    atoms: usize,
    start: Option<&RankStart>,
    pinned: Option<(usize, usize)>,
    from_refinement: usize,
    reference_refinement: usize,
) -> Result<EventHistoryFit, EventHistoryError> {
    let marks = cohort.marks();
    if spec.covariates.len() != 1 && spec.covariates.len() != marks {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "expected one covariate term collection or one per mark ({marks}), got {}",
                spec.covariates.len()
            ),
        });
    }
    if !(spec.quadrature_tolerance.is_finite() && spec.quadrature_tolerance > 0.0) {
        return Err(EventHistoryError::InvalidInput {
            reason: "quadrature tolerance must be finite and positive".to_string(),
        });
    }
    let time_scale = cohort.time_scale();
    let mut options = spec.options.clone();
    options.compute_covariance = true;

    // The bases are built on the outcome-free design rows and frozen; every
    // mesh evaluates the frozen bases, so a refinement never changes what a
    // coefficient means.
    let rows = design_rows(cohort, spec.quadrature_order)?;
    let mut frozen_specs = Vec::with_capacity(marks);
    for d in 0..marks {
        let term_spec = if spec.covariates.len() == 1 {
            &spec.covariates[0]
        } else {
            &spec.covariates[d]
        };
        let design = build_term_collection_design(rows.view(), term_spec).map_err(|error| {
            EventHistoryError::Fit {
                reason: format!("design for mark {d}: {error}"),
            }
        })?;
        let frozen = freeze_term_collection_from_design(term_spec, &design).map_err(|error| {
            EventHistoryError::Fit {
                reason: format!("freezing mark {d} term collection: {error}"),
            }
        })?;
        frozen_specs.push(frozen);
    }
    let build = |order: usize, refinement: usize| -> Result<Built, EventHistoryError> {
        let nodes = Arc::new(expand_nodes(cohort, spec.quadrature_order, refinement)?);
        let mut designs = Vec::with_capacity(marks);
        let mut dense = Vec::with_capacity(marks);
        let mut specs = Vec::with_capacity(marks + 1);
        for d in 0..marks {
            let design = build_term_collection_design(nodes.node_data.view(), &frozen_specs[d])
                .map_err(|error| EventHistoryError::Fit {
                    reason: format!("design for mark {d} on the mesh: {error}"),
                })?;
            let dense_design = design
                .design
                .try_to_dense_arc("event-history mark design")
                .map_err(|error| EventHistoryError::Fit {
                    reason: error.to_string(),
                })?;
            let mut spec = mark_block_spec(&cohort.mark_names[d], &design);
            // The bases are frozen, so the incumbent's coefficients mean the
            // same thing here: start from them.
            if let Some(beta) = start.and_then(|s| s.mark_betas.get(d))
                && beta.len() == design.design.ncols()
            {
                spec.initial_beta = Some(beta.clone());
            }
            specs.push(spec);
            designs.push(design);
            dense.push(dense_design);
        }
        let reference = match &spec.reference {
            Some(strata) => Some(Arc::new(reference_tables(
                cohort,
                strata,
                &frozen_specs,
                spec.quadrature_order,
                reference_refinement,
                &nodes,
            )?)),
            None => None,
        };
        if atoms > 0 {
            let start = start.ok_or_else(|| EventHistoryError::InvalidInput {
                reason: "a latent block needs the start the evidence chose for it".to_string(),
            })?;
            specs.push(latent_block_spec(
                nodes.total_nodes,
                marks,
                atoms,
                start,
                rate_band(&nodes, time_scale)?,
            )?);
        }
        let family = EventHistoryFamily::new(
            Arc::clone(&nodes),
            dense.clone(),
            atoms,
            order,
            time_scale,
            start.map_or_else(Vec::new, RankStart::held_rates),
        )?;
        let family = family.with_reference(reference);
        preflight(
            order,
            atoms,
            nodes.max_subject_nodes(),
            marks,
            family.total_width(),
        )?;
        Ok(Built {
            nodes,
            family,
            designs,
            dense,
            specs,
        })
    };
    let warm = |built: &mut Built, fit: &UnifiedFitResult| {
        let mut cursor = 0usize;
        for (block, state) in built.specs.iter_mut().zip(fit.block_states.iter()) {
            block.initial_beta = Some(state.beta.clone());
            let count = block.initial_log_lambdas.len();
            if cursor + count <= fit.log_lambdas.len() {
                block.initial_log_lambdas =
                    fit.log_lambdas.slice(s![cursor..cursor + count]).to_owned();
            }
            cursor += count;
        }
    };
    // The largest shift of any fitted coefficient under a refinement, in
    // posterior standard deviations, without refitting: at the fitted
    // coefficients the current setting is stationary, so the refined
    // setting's penalised mode moves by `V (g' − g)` to first order, with
    // the two settings' exact gradients taken at the same coefficients and
    // `V` the posterior covariance the fit publishes (the inverse penalised
    // Hessian). One gradient per candidate, and no second optimisation whose
    // own convergence would have to be certified before the certificate
    // could be read.
    let check = |candidate: &Built,
                 fit: &UnifiedFitResult,
                 current_gradient: &[f64],
                 covariance: &Array2<f64>,
                 sd: &[f64]|
     -> Result<RefinementCheck, EventHistoryError> {
        let betas: Vec<Array1<f64>> = fit.block_states.iter().map(|s| s.beta.clone()).collect();
        let states = candidate.states(&betas);
        let refined_gradient = candidate
            .family
            .exact_gradient(&states)
            .map_err(|reason| typed_failure(&candidate.family, reason))?;
        if refined_gradient.len() != current_gradient.len() {
            return Err(EventHistoryError::Fit {
                reason: format!(
                    "certificate: the refined setting has {} coefficients, the fit {}",
                    refined_gradient.len(),
                    current_gradient.len()
                ),
            });
        }
        let discrepancy: Vec<f64> = refined_gradient
            .iter()
            .zip(current_gradient.iter())
            .map(|(refined, current)| refined - current)
            .collect();
        let mut shift = 0.0_f64;
        for (q, scale) in sd.iter().enumerate() {
            let move_q: f64 = (0..discrepancy.len())
                .map(|r| covariance[[q, r]] * discrepancy[r])
                .sum();
            shift = shift.max(move_q.abs() / scale);
        }
        let log_likelihood = candidate
            .family
            .log_likelihood(&states)
            .map_err(|reason| typed_failure(&candidate.family, reason))?;
        Ok(RefinementCheck {
            candidate: 0,
            coefficient_shift: shift,
            log_likelihood,
        })
    };

    let mesh_ceiling = cohort.mesh_refinement_ceiling();
    // A candidate the rank path is judging is fitted at the SETTING the
    // incumbent was certified at, not at its own: the two criteria are
    // compared, so they must be the same functional. Only the fit that is
    // returned runs the refinement ladder.
    let (mut order, mut refinement) = match pinned {
        Some((order, refinement)) => (order, refinement),
        None => (spec.gauss_hermite_order.max(3), from_refinement),
    };
    let mut built = build(order, refinement)?;
    loop {
        let fit = match fit_custom_family(&built.family, &built.specs, &options) {
            Ok(fit) => fit,
            Err(error) => {
                // The engine carries the family's errors as text, so a
                // reference step the reference grid cannot take is read back
                // typed from the family, for `fit_event_history` to answer by
                // refining that grid.
                let message = error.to_string();
                let failure = typed_failure(
                    &built.family,
                    format!(
                        "event-history LAML fit at Gauss-Hermite order {order}, mesh refinement {refinement}: {message}"
                    ),
                );
                if matches!(failure, EventHistoryError::ReferenceStep { .. }) {
                    return Err(failure);
                }
                // A posterior the grid cannot represent (its interpolant goes
                // negative where the mass is) is answered by resolving the
                // grid, which is the ladder this driver already owns — not by
                // handing the caller a number the representation could not
                // carry. The grid is raised a rung at a time up to where the
                // raised rule's interpolant would amplify roundoff past the
                // certificate's tolerance. A pinned candidate is not raised:
                // it must stay at the incumbent's setting, so the loss is the
                // caller's, typed, to answer by raising the incumbent.
                // Anything else is the caller's to see.
                if message.contains(LOST_POSITIVITY) {
                    if pinned.is_some() {
                        return Err(EventHistoryError::LostPositivity {
                            reason: format!(
                                "the candidate at rank {atoms}, pinned at Gauss-Hermite order {order}, mesh refinement {refinement}: {message}"
                            ),
                        });
                    }
                    let raised = if atoms > 0 {
                        positivity_raise(order, built.nodes.max_subject_nodes(), spec.quadrature_tolerance)
                    } else {
                        None
                    };
                    if let Some(next_order) = raised {
                        log::info!(
                            "[event-history] Gauss-Hermite order {order} cannot represent a posterior on this cohort; raising it to {next_order}"
                        );
                        order = next_order;
                        built = build(order, refinement)?;
                        continue;
                    }
                }
                return Err(failure);
            }
        };
        let total = built.family.total_width();
        // The certificate turns a gradient discrepancy into a coefficient
        // one through the posterior covariance, and reads the result in
        // posterior standard deviations, so it needs the whole matrix.
        let covariance = fit
            .beta_covariance()
            .filter(|c| c.nrows() == total && c.ncols() == total)
            .ok_or_else(|| EventHistoryError::Fit {
                reason: format!(
                    "the fit carries no {total}×{total} posterior covariance, which the refinement certificate measures its shift in"
                ),
            })?
            .clone();
        let sd: Vec<f64> = (0..total)
            .map(|q| covariance[[q, q]].max(0.0).sqrt())
            .collect();
        if let Some(q) = sd.iter().position(|s| !(s.is_finite() && *s > 0.0)) {
            return Err(EventHistoryError::Fit {
                reason: format!(
                    "coefficient {q} has no finite positive scale to measure a refinement's shift in ({}); it is unidentified at the fitted mode",
                    sd[q]
                ),
            });
        }
        let value = fit.log_likelihood;
        let current_gradient = built
            .family
            .exact_gradient(&fit.block_states)
            .map_err(|reason| typed_failure(&built.family, reason))?;
        // Gauss-Hermite refinement: admissible while the interpolant's
        // roundoff amplification stays below the certificate's tolerance.
        // Without a latent block there is no latent integral to certify.
        let next_order = 2 * order - 1;
        let max_nodes = built.nodes.max_subject_nodes();
        let gauss_hermite = if atoms == 0 || pinned.is_some() {
            RefinementCheck {
                candidate: order,
                coefficient_shift: 0.0,
                log_likelihood: value,
            }
        } else {
            let rule = GaussHermite::new(next_order)?;
            if !built.family.held_rates.iter().all(|r| *r == Some(0.0))
                && !certifiable(order, max_nodes, spec.quadrature_tolerance)
            {
                return Err(EventHistoryError::NumericalFailure {
                    reason: format!(
                        "the Gauss-Hermite certificate cannot be checked at order {next_order}: the Lagrange interpolant's Lebesgue constant {:.3e} amplifies roundoff above the tolerance {} over {max_nodes} nodes; the latent integral at order {order} is uncertified",
                        rule.lebesgue_constant, spec.quadrature_tolerance
                    ),
                });
            }
            let order_candidate = build(next_order, refinement)?;
            let mut gauss_hermite =
                check(&order_candidate, &fit, &current_gradient, &covariance, &sd)?;
            gauss_hermite.candidate = next_order;
            if gauss_hermite.coefficient_shift > spec.quadrature_tolerance {
                log::info!(
                    "[event-history] Gauss-Hermite order {order} moves the coefficients by {:.3} posterior sd at order {next_order}; refitting",
                    gauss_hermite.coefficient_shift
                );
                order = next_order;
                built = order_candidate;
                warm(&mut built, &fit);
                continue;
            }
            gauss_hermite
        };
        if pinned.is_some() {
            let mesh = RefinementCheck {
                candidate: refinement,
                coefficient_shift: 0.0,
                log_likelihood: value,
            };
            return assemble(
                cohort,
                spec,
                frozen_specs,
                time_scale,
                Assembled {
                    built,
                    fit,
                    covariance,
                    atoms,
                    order,
                    refinement,
                    value,
                    gauss_hermite,
                    mesh,
                },
            );
        }
        if refinement + 1 > mesh_ceiling {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!(
                    "the fitted coefficients are still moving under mesh refinement at level {refinement}, where every cell is already narrower than the shortest interval this cohort's own breakpoints distinguish; the fit is not resolved by refining time further"
                ),
            });
        }
        let mesh_candidate = build(order, refinement + 1)?;
        let mut mesh = check(&mesh_candidate, &fit, &current_gradient, &covariance, &sd)?;
        mesh.candidate = refinement + 1;
        if mesh.coefficient_shift > spec.quadrature_tolerance {
            log::info!(
                "[event-history] mesh refinement {refinement} moves the coefficients by {:.3} posterior sd at refinement {}; refitting",
                mesh.coefficient_shift,
                refinement + 1
            );
            refinement += 1;
            built = mesh_candidate;
            warm(&mut built, &fit);
            continue;
        }
        return assemble(
            cohort,
            spec,
            frozen_specs,
            time_scale,
            Assembled {
                built,
                fit,
                covariance,
                atoms,
                order,
                refinement,
                value,
                gauss_hermite,
                mesh,
            },
        );
    }
}

/// The fitted model at one rank and one certified discretisation.
struct Assembled {
    built: Built,
    fit: UnifiedFitResult,
    /// The fit's posterior covariance over every coefficient.
    covariance: Array2<f64>,
    atoms: usize,
    order: usize,
    refinement: usize,
    value: f64,
    gauss_hermite: RefinementCheck,
    mesh: RefinementCheck,
}

/// The latent objects a fit reports, read from the latent block's mode and
/// posterior covariance: the posterior-mean covariance and its eigenmodes
/// with their uncertainty, and the loadings in the canonical gauge.
struct LatentReport {
    covariance: Array2<f64>,
    atom_covariances: Vec<Array2<f64>>,
    eigenvalues: Array1<f64>,
    eigenvalue_sd: Array1<f64>,
    eigenvectors: Array2<f64>,
    effective_rank: f64,
    loadings: Array2<f64>,
    log_rates: Vec<f64>,
    rate_held: Vec<bool>,
    atom_log_lambdas: Vec<f64>,
}

fn latent_report(
    family: &EventHistoryFamily,
    latent: &Array1<f64>,
    posterior: &Array2<f64>,
    atom_log_lambdas: &[f64],
) -> Result<LatentReport, EventHistoryError> {
    let marks = family.marks();
    let atoms = family.atoms();
    let latent_offset = family.block_offsets()[marks];
    let mut loadings = Array2::<f64>::zeros((marks, atoms));
    for d in 0..marks {
        for k in 0..atoms {
            loadings[[d, k]] = latent[d * atoms + k];
        }
    }
    let log_rates: Vec<f64> = family.atom_rates(latent).iter().map(|nu| nu.ln()).collect();
    let rate_held = family.rate_held();
    // E[a_k a_kᵀ | data] = â_k â_kᵀ + Cov(a_k): the mode plus the posterior
    // spread of the loadings, which is what the posterior mean of a quadratic
    // form carries. The rates' uncertainty does not enter `C(0)`.
    let atom_covariances: Vec<Array2<f64>> = (0..atoms)
        .map(|k| {
            let mut share = Array2::<f64>::zeros((marks, marks));
            for d in 0..marks {
                for e in 0..marks {
                    let qd = latent_offset + d * atoms + k;
                    let qe = latent_offset + e * atoms + k;
                    share[[d, e]] = loadings[[d, k]] * loadings[[e, k]] + posterior[[qd, qe]];
                }
            }
            0.5 * (&share + &share.t())
        })
        .collect();
    let mut covariance = Array2::<f64>::zeros((marks, marks));
    for share in &atom_covariances {
        covariance += share;
    }
    let (eigenvalues, eigenvectors) = super::covariance::eigenmodes(&covariance)?;
    // First-order eigenvalue perturbation: `∂(v_jᵀ A Aᵀ v_j) / ∂a_{dk} =
    // 2 (v_j)_d (v_jᵀ a_k)`, propagated through the posterior covariance of
    // the loadings.
    let width = marks * atoms;
    let mut eigenvalue_sd = Array1::<f64>::zeros(marks);
    for j in 0..marks {
        let vj = eigenvectors.column(j);
        let mut gradient = vec![0.0; width];
        for k in 0..atoms {
            let projection: f64 = (0..marks).map(|d| vj[d] * loadings[[d, k]]).sum();
            for d in 0..marks {
                gradient[d * atoms + k] = 2.0 * vj[d] * projection;
            }
        }
        let mut variance = 0.0;
        for p in 0..width {
            for q in 0..width {
                variance +=
                    gradient[p] * posterior[[latent_offset + p, latent_offset + q]] * gradient[q];
            }
        }
        eigenvalue_sd[j] = variance.max(0.0).sqrt();
    }
    let effective_rank = super::covariance::effective_rank(&covariance);
    // Canonical gauge: atoms ordered by rate, slowest first, and every column
    // signed so its largest entry is positive. The likelihood is invariant
    // under both, so the report picks one representative of each orbit.
    let mut order: Vec<usize> = (0..atoms).collect();
    order.sort_by(|&a, &b| log_rates[a].total_cmp(&log_rates[b]));
    let mut gauged = Array2::<f64>::zeros((marks, atoms));
    let mut gauged_rates = Vec::with_capacity(atoms);
    let mut gauged_held = Vec::with_capacity(atoms);
    let mut gauged_lambdas = Vec::with_capacity(atoms);
    let mut gauged_shares = Vec::with_capacity(atoms);
    for (slot, &k) in order.iter().enumerate() {
        let column = loadings.column(k);
        let largest = column
            .iter()
            .copied()
            .fold(0.0_f64, |acc, a| if a.abs() > acc.abs() { a } else { acc });
        let sign = if largest < 0.0 { -1.0 } else { 1.0 };
        for d in 0..marks {
            gauged[[d, slot]] = sign * column[d];
        }
        gauged_rates.push(log_rates[k]);
        gauged_held.push(rate_held[k]);
        gauged_lambdas.push(atom_log_lambdas[k]);
        gauged_shares.push(atom_covariances[k].clone());
    }
    Ok(LatentReport {
        covariance,
        atom_covariances: gauged_shares,
        eigenvalues,
        eigenvalue_sd,
        eigenvectors,
        effective_rank,
        loadings: gauged,
        log_rates: gauged_rates,
        rate_held: gauged_held,
        atom_log_lambdas: gauged_lambdas,
    })
}

fn assemble(
    cohort: &EventHistoryCohort,
    spec: &EventHistorySpec,
    frozen_specs: Vec<TermCollectionSpec>,
    time_scale: f64,
    assembled: Assembled,
) -> Result<EventHistoryFit, EventHistoryError> {
    let Assembled {
        built,
        fit,
        covariance,
        atoms,
        order,
        refinement,
        value,
        gauss_hermite,
        mesh,
    } = assembled;
    let marks = cohort.marks();
    let empty = Array1::<f64>::zeros(0);
    let latent: &Array1<f64> = if atoms > 0 {
        &fit.block_states[marks].beta
    } else {
        &empty
    };
    // The fit's log strengths are laid out physically, one per penalty in
    // block order, the latent block's fixed priors last.
    let n_lambda = fit.log_lambdas.len();
    let atom_log_lambdas: Vec<f64> = fit
        .log_lambdas
        .iter()
        .skip(n_lambda.saturating_sub(atoms))
        .copied()
        .collect();
    let report = latent_report(&built.family, latent, &covariance, &atom_log_lambdas)?;
    let rates: Vec<f64> = report
        .log_rates
        .iter()
        .map(|r| r.exp() / time_scale)
        .collect();
    let Built {
        nodes,
        family,
        designs,
        ..
    } = built;
    let centring = if family.reference.is_some() {
        Some(family.refresh_normaliser(&fit.block_states)?)
    } else { None };
    Ok(EventHistoryFit {
        nodes,
        family,
        fit,
        mark_kinds: cohort.mark_kinds.clone(),
        frozen_specs,
        designs,
        covariance: report.covariance,
        atom_covariances: report.atom_covariances,
        eigenvalues: report.eigenvalues,
        eigenvalue_sd: report.eigenvalue_sd,
        eigenvectors: report.eigenvectors,
        effective_rank: report.effective_rank,
        loadings: report.loadings,
        log_rates: report.log_rates,
        rates,
        rate_held: report.rate_held,
        atom_log_lambdas: report.atom_log_lambdas,
        time_scale,
        quadrature_order: spec.quadrature_order,
        mesh_refinement: refinement,
        quadrature: QuadratureCertificate {
            gauss_hermite_order: order,
            mesh_refinement: refinement,
            log_likelihood: value,
            gauss_hermite,
            mesh,
        },
        rank_path: Vec::new(),
        atom_evidence: Vec::new(),
        reference_refinements: Vec::new(),
        centring,
        reference_certificate: None,
    })
}

/// Construct the enlarged model with a zero new loading and fixed rates.
fn added_atom_probe(fit: &EventHistoryFit, log_rate: f64) -> Result<(EventHistoryFamily, Vec<ParameterBlockState>), EventHistoryError> {
    let marks = fit.marks();
    let atoms = fit.rank() + 1;
    preflight(fit.family.gh.order, atoms, fit.nodes.max_subject_nodes(), marks,
        fit.family.block_offsets()[marks] + marks * atoms)?;
    let mut rates: Vec<Option<f64>> = fit.log_rates.iter().map(|r| Some(r.exp())).collect();
    rates.push(Some(log_rate.exp()));
    let probe = EventHistoryFamily::new(fit.nodes.clone(), fit.family.designs.clone(),
        atoms, fit.family.gh.order, fit.time_scale, rates)?
        .with_reference(fit.family.reference.clone());
    let mut states = fit.fit.block_states[..marks].to_vec();
    let mut loadings = Array1::zeros(marks * atoms);
    for d in 0..marks {
        for k in 0..fit.rank() { loadings[d * atoms + k] = fit.loadings[[d, k]]; }
    }
    states.push(ParameterBlockState { beta: loadings, eta: Array1::zeros(fit.nodes.total_nodes) });
    Ok((probe, states))
}

/// [`added_atom_probe`] on the time mesh at `refinement`: the fit's own
/// coefficients, its frozen bases evaluated on that mesh's nodes, and the
/// reference tables built on those nodes, so an integral the rank decision
/// reads can be read one mesh rung up at the same coefficients.
fn added_atom_probe_on_mesh(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    spec: &EventHistorySpec,
    refinement: usize,
    log_rate: f64,
    reference_refinement: usize,
) -> Result<(EventHistoryFamily, Vec<ParameterBlockState>), EventHistoryError> {
    let marks = fit.marks();
    let atoms = fit.rank() + 1;
    let nodes = Arc::new(expand_nodes(cohort, spec.quadrature_order, refinement)?);
    let mut dense = Vec::with_capacity(marks);
    let mut states = Vec::with_capacity(marks + 1);
    for d in 0..marks {
        let design = build_term_collection_design(nodes.node_data.view(), &fit.frozen_specs[d])
            .map_err(|error| EventHistoryError::Fit {
                reason: format!("design for mark {d} on mesh refinement {refinement}: {error}"),
            })?;
        let matrix = design
            .design
            .try_to_dense_arc("event-history mark design")
            .map_err(|error| EventHistoryError::Fit {
                reason: error.to_string(),
            })?;
        let beta = fit.fit.block_states[d].beta.clone();
        let eta = matrix.dot(&beta) + &design.affine_offset;
        states.push(ParameterBlockState { beta, eta });
        dense.push(matrix);
    }
    let reference = match &spec.reference {
        Some(strata) => Some(Arc::new(reference_tables(
            cohort,
            strata,
            &fit.frozen_specs,
            spec.quadrature_order,
            reference_refinement,
            &nodes,
        )?)),
        None => None,
    };
    let mut rates: Vec<Option<f64>> = fit.log_rates.iter().map(|r| Some(r.exp())).collect();
    rates.push(Some(log_rate.exp()));
    let probe = EventHistoryFamily::new(
        Arc::clone(&nodes),
        dense,
        atoms,
        fit.family.gh.order,
        fit.time_scale,
        rates,
    )?
    .with_reference(reference);
    preflight(fit.family.gh.order, atoms, nodes.max_subject_nodes(), marks, probe.total_width())?;
    let mut loadings = Array1::zeros(marks * atoms);
    for d in 0..marks {
        for k in 0..fit.rank() {
            loadings[d * atoms + k] = fit.loadings[[d, k]];
        }
    }
    states.push(ParameterBlockState {
        beta: loadings,
        eta: Array1::zeros(nodes.total_nodes),
    });
    probe
        .validate_states(&states)
        .map_err(|reason| EventHistoryError::InvalidInput { reason })?;
    Ok((probe, states))
}

/// The added-factor curvature at a probe's Gauss-Hermite order and one ladder
/// rung up.
struct CurvaturePair {
    coarse: Array2<f64>,
    refined: Array2<f64>,
    next_order: usize,
}

/// Boundary curvature of the computed latent-marginal likelihood. Unlike a
/// product of filtered residual means, this integrates the existing process
/// under the full observation law and differentiates reference centring too.
/// `None` where no ladder rung remains to check it against.
fn added_atom_curvature(fit: &EventHistoryFit, log_rate: f64, tolerance: f64) -> Result<Option<CurvaturePair>, EventHistoryError> {
    let (probe, states) = added_atom_probe(fit, log_rate)?;
    added_factor_curvature_pair(&probe, &states, tolerance)
}

/// The curvature over the added atom's loadings, one per mark: the block of
/// the computed Hessian those coordinates span, from block sweeps rather than
/// one path evaluation per pair of marks (#2965).
fn loading_curvature(probe: &EventHistoryFamily, states: &[ParameterBlockState]) -> Result<Array2<f64>, EventHistoryError> {
    let marks = probe.marks();
    let values: Vec<f64> = states.iter().flat_map(|s| s.beta.iter().copied()).collect();
    let offset = probe.block_offsets()[marks];
    let coordinates: Vec<usize> = (0..marks).map(|d| offset + d * probe.atoms + probe.atoms - 1).collect();
    let (_, _, hessian) = probe.coordinate_hessian(states, &values, &coordinates)?;
    Ok(Array2::from_shape_fn((marks, marks), |(d, e)| hessian[d * marks + e]))
}

/// A correct derivative of an unresolved integral is still unresolved, so the
/// curvature is read at the probe's order and at the ladder's next rung
/// (`2·order − 1`, the step the fit's certificate takes), and the proposal
/// prices the difference by what it moves ([`proposal_start_shift`]). `None`
/// where that rung would amplify interpolation roundoff past `tolerance`
/// ([`positivity_raise`]): the curvature at this order can no longer be
/// checked. A factor held static has no interpolant and is always checkable.
fn added_factor_curvature_pair(
    probe: &EventHistoryFamily,
    states: &[ParameterBlockState],
    tolerance: f64,
) -> Result<Option<CurvaturePair>, EventHistoryError> {
    let order = probe.gh.order;
    let next_order = 2 * order - 1;
    let interpolates = !probe.held_rates.iter().all(|r| *r == Some(0.0));
    if interpolates && !certifiable(order, probe.nodes.max_subject_nodes(), tolerance) {
        return Ok(None);
    }
    let coarse = loading_curvature(probe, states)?;
    preflight(next_order, probe.atoms, probe.nodes.max_subject_nodes(), probe.marks(), probe.total_width())?;
    let mut fine = probe.clone();
    fine.gh = Arc::new(GaussHermite::new(next_order)?);
    let refined = loading_curvature(&fine, states)?;
    if coarse.iter().chain(refined.iter()).any(|x| !x.is_finite()) {
        return Err(EventHistoryError::NumericalFailure {
            reason: format!("added-factor curvature is not finite at Gauss-Hermite order {order} or {next_order}"),
        });
    }
    Ok(Some(CurvaturePair {
        coarse,
        refined,
        next_order,
    }))
}

/// How far a rung's worth of added-factor curvature error moves the loadings
/// a proposal publishes, in posterior standard deviations: the unit
/// `EventHistorySpec::quadrature_tolerance` is denominated in.
///
/// The proposal starts the new atom at `a = s·v₀`, with `v₀` the top
/// eigenvector of the curvature and `s` the empirical-Bayes mode along it.
/// Under the selected prior precision, each direction `v_j`'s sampled profile
/// gives the posterior spread `σ_j` of the loading along `v_j` about its mode
/// ([`DirectionProfile::mode_spread`]).
/// - The next rung rotates `v₀` to `v₀′` (sign-aligned), moving the start by
///   `s·(v₀′ − v₀)`. Its component along `v_j` is `s·|⟨v_j, v₀′ − v₀⟩| / σ_j`
///   posterior sd.
/// - The next rung moves the top eigenvalue by `Δμ`, which adds `½Δμ·t²` to the
///   log integrand along `v₀`. That integrand's curvature at its mode is
///   `−1/σ₀²`, so to first order the mode moves by `Δμ·s·σ₀²`: `|Δμ|·s·σ₀`
///   posterior sd.
///
/// The shift is the largest of these. A refused proposal publishes no loading
/// (`s = 0`) and moves nothing. A spread that is not finite and positive cannot
/// price anything, so the shift is then unbounded.
fn proposal_start_shift(
    coarse: (&Array1<f64>, &Array2<f64>),
    refined: (&Array1<f64>, &Array2<f64>),
    mode_scale: f64,
    spreads: &[f64],
) -> f64 {
    let (coarse_values, coarse_vectors) = coarse;
    let (refined_values, refined_vectors) = refined;
    if spreads.len() != coarse_vectors.ncols() || spreads.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
        return f64::INFINITY;
    }
    let top = coarse_vectors.column(0);
    let mut rotated = refined_vectors.column(0).to_owned();
    if top.dot(&rotated) < 0.0 {
        rotated.mapv_inplace(|x| -x);
    }
    let delta = &rotated - &top;
    let mut shift = (refined_values[0] - coarse_values[0]).abs() * mode_scale * spreads[0];
    for (j, spread) in spreads.iter().enumerate() {
        shift = shift.max(mode_scale * coarse_vectors.column(j).dot(&delta).abs() / spread);
    }
    shift
}

/// What the incumbent's setting makes of a proposed atom.
enum Proposal {
    /// The atom with its direction, prior and start loading.
    Resolved(NewAtom),
    /// An integral the proposal reads is not resolved at this setting, for
    /// the stated reason; the rung is the ladder that resolves it.
    Unresolved(DecisionIntegral, Rung, String),
}

/// The atom `best_new_atom` proposes, completed at the incumbent's setting: its
/// direction from the added-factor curvature, its prior from the directional
/// profiles, and its start loading. Unresolved when a rung's worth of
/// curvature error moves that start by more than `tolerance` posterior sd
/// ([`proposal_start_shift`]), when no rung remains to check the curvature
/// against, or when the curvature or a profile evaluates a posterior the grid
/// cannot represent (`LostPositivity`).
fn propose_atom(
    fit: &EventHistoryFit,
    cohort: &EventHistoryCohort,
    spec: &EventHistorySpec,
    mut atom: NewAtom,
    time_scale: f64,
    tolerance: f64,
    reference_refinement: usize,
) -> Result<Proposal, EventHistoryError> {
    let marks = fit.marks();
    let rank = fit.rank();
    let order = fit.family.gh.order;
    let curvature = match added_atom_curvature(fit, atom.log_rate, tolerance) {
        Ok(Some(curvature)) => curvature,
        Ok(None) => {
            return Ok(Proposal::Unresolved(
                DecisionIntegral::AddedFactorCurvature,
                Rung::GaussHermite,
                format!("no Gauss-Hermite rung above order {order} checks the added-factor curvature"),
            ));
        }
        Err(EventHistoryError::LostPositivity { reason }) => {
            return Ok(Proposal::Unresolved(
                DecisionIntegral::AddedFactorCurvature,
                Rung::GaussHermite,
                reason,
            ));
        }
        Err(error) => return Err(error),
    };
    let (values, vectors) = super::covariance::eigenmodes(&curvature.coarse)?;
    let (refined_values, refined_vectors) = super::covariance::eigenmodes(&curvature.refined)?;
    {
        // Whether the top of the spectrum is a cluster inside which one
        // eigenvector is not identified: the gaps to the top against the
        // rung's own perturbation of the curvature.
        let (delta_values, _) = super::covariance::eigenmodes(&(&curvature.refined - &curvature.coarse))?;
        let delta_norm = delta_values.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let gaps = Array1::from_iter(values.iter().skip(1).map(|mu| values[0] - mu));
        let shifts = Array1::from_iter(values.iter().zip(refined_values.iter()).map(|(a, b)| (b - a).abs()));
        let alignment = vectors.column(0).dot(&refined_vectors.column(0)).abs().min(1.0);
        log::info!(
            "[event-history] rank {rank} → {}: curvature spectrum at Gauss-Hermite orders {order}/{}: eigenvalues {values:.4e}, refined {refined_values:.4e}, gaps to the top {gaps:.3e}, rung shifts {shifts:.3e}, rung perturbation ‖ΔC‖₂ {delta_norm:.3e}, top-eigenvector angle {:.3e} rad",
            rank + 1,
            curvature.next_order,
            alignment.acos()
        );
    }
    atom.eigenvalue = values[0];
    atom.direction = vectors.column(0).to_vec();
    // The residual statistic supplies a proposal rate only. Every direction is
    // checked against a sampled profile of the actual likelihood, even when the
    // residual proxy refused it.
    let alongs: Vec<NewAtom> = (0..marks)
        .map(|d| {
            let mut along = atom.clone();
            along.direction = vectors.column(d).to_vec();
            along.ridge.mode_scale = (atom.ridge.mode_scale.powi(2) + 1.0 / (1.0 + values[d].abs())).sqrt();
            along
        })
        .collect();
    // The product of directional profiles proposes the prior, and each profile
    // must resolve its tail under the prior that product selects: every
    // direction is first sampled under its proposal precision, and any whose
    // tail at the selected `λ̂` is shallower than `resolved_profile_depth` is
    // sampled again under `λ̂`. Final acceptance uses the jointly fitted
    // criterion.
    let mut priors: Vec<f64> = alongs
        .iter()
        .map(|along| 1.0 / (along.ridge.mode_scale * along.ridge.mode_scale))
        .collect();
    let (refined, directions) = loop {
        let mut directions = Vec::with_capacity(marks);
        for (along, prior) in alongs.iter().zip(priors.iter()) {
            match direction_profile(fit, along, time_scale, *prior) {
                Ok(profile) => directions.push(DirectionEvidence::Sampled(profile)),
                Err(EventHistoryError::LostPositivity { reason }) => {
                    return Ok(Proposal::Unresolved(
                        DecisionIntegral::DirectionalProfile,
                        Rung::GaussHermite,
                        reason,
                    ));
                }
                Err(error) => return Err(error),
            }
        }
        let refined = empirical_bayes_ridge(&directions);
        let lambda = refined.log_lambda.exp();
        let unresolved: Vec<usize> = directions
            .iter()
            .enumerate()
            .filter_map(|(d, evidence)| match evidence {
                DirectionEvidence::Sampled(profile) if lambda.is_finite() => {
                    let n = profile.points.len();
                    let penalised = |i: usize| {
                        profile.values[i] - 0.5 * lambda * profile.points[i] * profile.points[i]
                    };
                    let peak = (0..n).map(penalised).fold(f64::NEG_INFINITY, f64::max);
                    (peak - penalised(n - 1) < resolved_profile_depth()).then_some(d)
                }
                _ => None,
            })
            .collect();
        if unresolved.is_empty() {
            break (refined, directions);
        }
        log::info!(
            "[event-history] rank {rank} → {}: directions {unresolved:?} are not resolved under the selected prior precision {lambda:.4e}; sampling them again under it",
            rank + 1
        );
        for d in unresolved {
            priors[d] = lambda;
        }
    };
    let lambda = refined.log_lambda.exp();
    if refined.accepted && lambda.is_finite() {
        let spreads: Vec<f64> = directions
            .iter()
            .map(|evidence| match evidence {
                DirectionEvidence::Sampled(profile) => profile.mode_spread(lambda),
                DirectionEvidence::Quartic { .. } => f64::NAN,
            })
            .collect();
        let shift = proposal_start_shift(
            (&values, &vectors),
            (&refined_values, &refined_vectors),
            refined.mode_scale,
            &spreads,
        );
        log::info!(
            "[event-history] rank {rank} → {}: a rung of curvature error moves the proposed start by {shift:.3e} posterior sd (mode scale {:.4e}, posterior sd along each direction {spreads:?})",
            rank + 1,
            refined.mode_scale
        );
        if !(shift <= tolerance) {
            return Ok(Proposal::Unresolved(
                DecisionIntegral::AddedFactorCurvature,
                Rung::GaussHermite,
                format!(
                    "the added-factor curvature between Gauss-Hermite orders {order} and {} moves the proposed start by {shift:.3e} posterior sd, above the tolerance {tolerance}",
                    curvature.next_order
                ),
            ));
        }
        // The node quadrature of the latent path is part of the same integral,
        // so the curvature is also read one mesh rung up, at the same
        // coefficients and order, and priced the same way (#2627). Where no
        // mesh rung remains the decision cannot be checked at this setting.
        let refinement = fit.quadrature.mesh_refinement;
        let next_refinement = refinement + 1;
        if next_refinement > cohort.mesh_refinement_ceiling() {
            return Ok(Proposal::Unresolved(
                DecisionIntegral::AddedFactorCurvature,
                Rung::Mesh,
                format!("no mesh rung above refinement {refinement} checks the added-factor curvature"),
            ));
        }
        let (probe, states) = added_atom_probe_on_mesh(
            fit,
            cohort,
            spec,
            next_refinement,
            atom.log_rate,
            reference_refinement,
        )?;
        let mesh_curvature = match loading_curvature(&probe, &states) {
            Ok(mesh_curvature) => mesh_curvature,
            Err(EventHistoryError::LostPositivity { reason }) => {
                return Ok(Proposal::Unresolved(
                    DecisionIntegral::AddedFactorCurvature,
                    Rung::GaussHermite,
                    reason,
                ));
            }
            Err(error) => return Err(error),
        };
        if mesh_curvature.iter().any(|x| !x.is_finite()) {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!(
                    "added-factor curvature is not finite at mesh refinement {next_refinement}"
                ),
            });
        }
        let (mesh_values, mesh_vectors) = super::covariance::eigenmodes(&mesh_curvature)?;
        let mesh_shift = proposal_start_shift(
            (&values, &vectors),
            (&mesh_values, &mesh_vectors),
            refined.mode_scale,
            &spreads,
        );
        log::info!(
            "[event-history] rank {rank} → {}: a mesh rung of curvature error (refinement {refinement} → {next_refinement}) moves the proposed start by {mesh_shift:.3e} posterior sd",
            rank + 1
        );
        if !(mesh_shift <= tolerance) {
            return Ok(Proposal::Unresolved(
                DecisionIntegral::AddedFactorCurvature,
                Rung::Mesh,
                format!(
                    "the added-factor curvature between mesh refinements {refinement} and {next_refinement} moves the proposed start by {mesh_shift:.3e} posterior sd, above the tolerance {tolerance}"
                ),
            ));
        }
    }
    log::info!(
        "[event-history] rank {rank} → {}: sampled profile proposal: prior log-precision {:.3} → {:.3}, evidence {:.3} → {:.3} nats, mode scale {:.4} → {:.4}",
        rank + 1,
        atom.ridge.log_lambda,
        refined.log_lambda,
        atom.ridge.gain,
        refined.gain,
        atom.ridge.mode_scale,
        refined.mode_scale
    );
    atom.loading = atom
        .direction
        .iter()
        .map(|x| refined.mode_scale * x)
        .collect();
    atom.ridge = refined;
    Ok(Proposal::Resolved(atom))
}

/// How far under its penalised peak a sampled profile's tail must reach before
/// the evidence search that consumes it can see nothing more:
/// `empirical_bayes_ridge` resolves the log evidence to `√ε`, so a tail more
/// than `½ ln(1/ε)` nats under the peak moves nothing it can resolve.
fn resolved_profile_depth() -> f64 {
    -0.5 * f64::EPSILON.ln()
}

/// The log-likelihood sampled along `atom`'s direction until, under the prior
/// precision `prior`, its tail is [`resolved_profile_depth`] under the peak.
fn direction_profile(
    fit: &EventHistoryFit,
    atom: &NewAtom,
    time_scale: f64,
    prior: f64,
) -> Result<DirectionProfile, EventHistoryError> {
    let marks = fit.marks();
    let carried = fit.rank();
    let atoms = carried + 1;
    let mut held: Vec<Option<f64>> = fit
        .rate_held
        .iter()
        .zip(fit.log_rates.iter())
        .map(|(&held, &rate)| held.then_some(rate.exp()))
        .collect();
    held.push(Some(atom.log_rate.exp()));
    let build = |order: usize| -> Result<EventHistoryFamily, EventHistoryError> {
        EventHistoryFamily::new(
            Arc::clone(&fit.nodes),
            fit.family.designs.clone(),
            atoms,
            order,
            time_scale,
            held.clone(),
        ).map(|family| family.with_reference(fit.family.reference.clone()))
    };
    let probe = build(fit.family.gh.order)?;
    let width = probe.latent_width();
    let total = probe.total_width();
    let latent_offset = probe.block_offsets()[marks];
    let band = probe.rate_band();
    let latent_at = |t: f64| -> Array1<f64> {
        let mut beta = Array1::<f64>::zeros(width);
        for d in 0..marks {
            for k in 0..carried {
                beta[d * atoms + k] = fit.loadings[[d, k]];
            }
            beta[d * atoms + carried] = t * atom.direction[d];
        }
        let mut slot = marks * atoms;
        for k in 0..carried {
            if !fit.rate_held[k] {
                beta[slot] = rate_chart(band, fit.log_rates[k].exp());
                slot += 1;
            }
        }
        beta
    };
    let states_at = |t: f64| -> Vec<ParameterBlockState> {
        let mut states: Vec<ParameterBlockState> = fit.fit.block_states[..marks].to_vec();
        states.push(ParameterBlockState {
            beta: latent_at(t),
            eta: Array1::zeros(fit.nodes.total_nodes),
        });
        states
    };
    let mut direction = Array1::<f64>::zeros(total);
    for d in 0..marks {
        direction[latent_offset + d * atoms + carried] = atom.direction[d];
    }
    // Finite sampled profiles propose a prior. Every sample must be
    // evaluated successfully at the same quadrature setting; a failed tail
    // is an unresolved proposal, never an invented continuation, and a sample
    // the grid cannot represent is refused as `LostPositivity`, at a setting
    // the incumbent was certified at.
    let resolved_depth = resolved_profile_depth();
    let (base, _) = probe.directional_log_likelihood(&states_at(0.0), &direction)?;
    let mut step = atom.ridge.mode_scale / 8.0;
    let mut points = vec![0.0];
    let mut values = vec![0.0];
    let mut slopes = vec![0.0];
    let mut peak = 0.0_f64;
    let mut t = 0.0;
    loop {
        t += step;
        let (value, slope) = probe.directional_log_likelihood(&states_at(t), &direction)?;
        let value = value - base;
        peak = peak.max(value - 0.5 * prior * t * t);
        points.push(t);
        values.push(value);
        slopes.push(slope);
        if !(value.is_finite() && slope.is_finite()) {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!(
                    "the log-likelihood along the proposed direction is not finite at t = {t:.3e}: value {value}, slope {slope}"
                ),
            });
        }
        // The evidence search integrates `exp(g(t) − ½·prior·t²)`, so the tail
        // is resolved once that integrand is falling and has dropped
        // `resolved_depth` under its peak — whether or not the likelihood
        // itself has turned down: a likelihood rising toward a finite limit
        // still has a negligible penalised tail.
        if slope - prior * t < 0.0 && value - 0.5 * prior * t * t < peak - resolved_depth {
            break;
        }
        if points.len() % 128 == 0 {
            step *= 2.0;
        }
    }
    Ok(DirectionProfile {
        points,
        values,
        slopes,
    })
}

/// Grow candidate ranks under one reference law and numerical setting.
/// Filtered residuals propose rates; differentiated likelihood curvature and
/// sampled profiles propose loadings and their prior. Jointly fitted ranks
/// are compared by the same outer Laplace criterion. The candidate search
/// and the Laplace approximation do not establish globally optimal evidence.
fn fit_event_history_on_grid(
    cohort: &mut EventHistoryCohort,
    spec: &EventHistorySpec,
    reference_refinement: usize,
) -> Result<EventHistoryFit, EventHistoryError> {
    cohort.validate()?;
    let marks = cohort.marks();
    if let Some(strata) = &spec.reference {
        strata.validate(cohort.subjects.len(), cohort.covariates.nrows())?;
    }
    let time_scale = cohort.time_scale();
    // Every rank's incumbent is certified before it proposes anything: the
    // ladder refines its Gauss-Hermite order and time mesh until no fitted
    // coefficient moves. The proposals, the sampled profiles and the grown
    // candidate all run at that certified setting, so the two criteria the
    // rank decision compares are one functional at a resolved setting. That
    // setting must also resolve every integral the decision reads — the
    // proposal's added-factor curvature, its profiles and their positivity —
    // so while any is unresolved the incumbent is refitted one ladder rung up
    // and the proposal is formed again there. Where no certifiable rung
    // remains, the path stops at the certified incumbent and records the
    // growth as unresolved (`RankStep::growth_unresolved`).
    let mut rank_spec = spec.clone();
    let mut fit = certified_rank(cohort, &rank_spec, 0, None, 0, reference_refinement)?;
    let mut rank_path: Vec<RankStep> = Vec::new();
    let mut atom_evidence: Vec<f64> = Vec::new();
    loop {
        let rank = fit.rank();
        let pin = Some((fit.quadrature.gauss_hermite_order, fit.quadrature.mesh_refinement));
        let residuals = fit
            .family
            .residuals(&fit.fit.block_states)
            .map_err(|reason| EventHistoryError::Fit { reason })?;
        let Some(proposal) = best_new_atom(&residuals, marks, time_scale, fit.family.rate_band())?
        else {
            break;
        };
        let proposed = proposal.clone();
        let atom = match propose_atom(
            &fit,
            cohort,
            spec,
            proposal,
            time_scale,
            spec.quadrature_tolerance,
            reference_refinement,
        )? {
            Proposal::Resolved(atom) => atom,
            Proposal::Unresolved(integral, rung, reason) => {
                match raise_incumbent(cohort, &mut rank_spec, &fit, rung, &reason, reference_refinement)? {
                    Some(raised) => {
                        fit = raised;
                        continue;
                    }
                    None => {
                        rank_path.push(RankStep {
                            rank,
                            score_eigenvalue: proposed.eigenvalue,
                            standardised_gain: proposed.standardised_gain,
                            proposed_rate: proposed.log_rate.exp() / time_scale,
                            at_resolution_limit: proposed.at_upper_limit,
                            rate_held: proposed.rate_held(),
                            ridge_log_lambda: proposed.ridge.log_lambda,
                            evidence_gain: 0.0,
                            log_likelihood_gain: 0.0,
                            accepted: false,
                            converged: false,
                            growth_unresolved: Some(UnresolvedGrowth {
                                gauss_hermite_order: fit.quadrature.gauss_hermite_order,
                                integral,
                                reason,
                            }),
                        });
                        break;
                    }
                }
            }
        };
        let mut step = RankStep {
            rank,
            score_eigenvalue: atom.eigenvalue,
            standardised_gain: atom.standardised_gain,
            proposed_rate: atom.log_rate.exp() / time_scale,
            at_resolution_limit: atom.at_upper_limit,
            rate_held: atom.rate_held(),
            ridge_log_lambda: atom.ridge.log_lambda,
            evidence_gain: 0.0,
            log_likelihood_gain: 0.0,
            accepted: atom.ridge.accepted,
            converged: true,
            growth_unresolved: None,
        };
        let limit = if atom.at_lower_limit {
            " (held: a static frailty)"
        } else if atom.at_upper_limit {
            " (held: the fastest rate the breakpoints resolve)"
        } else {
            ""
        };
        if !atom.ridge.accepted {
            log::info!(
                "[event-history] rank {rank} → {}: score eigenvalue {:.4e} at log-rate {:.3}{}, standardised gain {:.3} nats; the evidence keeps the loading at zero (prior log-precision {:.3}, evidence {:.3} nats): refused",
                rank + 1,
                atom.eigenvalue,
                atom.log_rate,
                limit,
                atom.standardised_gain,
                atom.ridge.log_lambda,
                atom.ridge.gain
            );
            rank_path.push(step);
            break;
        }
        let start = RankStart {
            mark_betas: fit.fit.block_states[..marks]
                .iter()
                .map(|s| s.beta.clone())
                .collect(),
            loadings: fit.loadings.iter().copied().collect(),
            log_rates: fit.log_rates.clone(),
            log_lambdas: fit.atom_log_lambdas.clone(),
            rate_held: fit.rate_held.clone(),
            atom: Some(atom.clone()),
        };
        let grown = fit_at_rank(
            cohort,
            &rank_spec,
            rank + 1,
            Some(&start),
            pin,
            fit.quadrature.mesh_refinement,
            reference_refinement,
        );
        match grown {
            Ok(candidate) => {
                let criterion = |fit: &EventHistoryFit| -> Result<f64, EventHistoryError> {
                    fit.fit.reml_score().filter(|value| value.is_finite())
                        .ok_or_else(|| EventHistoryError::Fit {
                            reason: "rank comparison requires a finite joint LAML criterion".to_string(),
                        })
                };
                step.log_likelihood_gain = candidate.fit.log_likelihood - fit.fit.log_likelihood;
                // Directional profile products propose a prior; they do not
                // establish the evidence of the jointly fitted candidate.
                step.evidence_gain = criterion(&fit)? - criterion(&candidate)?;
                if step.evidence_gain <= 0.0 {
                    step.accepted = false;
                    rank_path.push(step);
                    break;
                }
                log::info!(
                    "[event-history] rank {rank} → {}: score eigenvalue {:.4e} at log-rate {:.3}{}, standardised gain {:.3} nats, prior log-precision {:.3}, evidence {:.3} nats, mode scale {:.4}: accepted; log-likelihood {:.3} → {:.3}, fitted log-rate {:.3}",
                    rank + 1,
                    atom.eigenvalue,
                    atom.log_rate,
                    limit,
                    atom.standardised_gain,
                    atom.ridge.log_lambda,
                    atom.ridge.gain,
                    atom.ridge.mode_scale,
                    fit.fit.log_likelihood,
                    candidate.fit.log_likelihood,
                    candidate.log_rates.last().copied().unwrap_or(f64::NAN)
                );
                atom_evidence.push(step.evidence_gain);
                rank_path.push(step);
                let start = RankStart::carried(
                    candidate.fit.block_states[..marks]
                        .iter()
                        .map(|s| s.beta.clone())
                        .collect(),
                    candidate.loadings.iter().copied().collect(),
                    candidate.log_rates.clone(),
                    candidate.atom_log_lambdas.clone(),
                    candidate.rate_held.clone(),
                );
                // The accepted model is certified from the mesh its rank was
                // decided at, never below it.
                fit = certified_rank(
                    cohort,
                    &rank_spec,
                    rank + 1,
                    Some(&start),
                    fit.quadrature.mesh_refinement,
                    reference_refinement,
                )?;
            }
            // The candidate is fitted at the incumbent's setting, so a
            // posterior that setting cannot represent is the incumbent's
            // setting failing to resolve the decision.
            Err(EventHistoryError::LostPositivity { reason }) => {
                match raise_incumbent(
                    cohort,
                    &mut rank_spec,
                    &fit,
                    Rung::GaussHermite,
                    &reason,
                    reference_refinement,
                )? {
                    Some(raised) => fit = raised,
                    None => {
                        step.accepted = false;
                        step.converged = false;
                        step.growth_unresolved = Some(UnresolvedGrowth {
                            gauss_hermite_order: fit.quadrature.gauss_hermite_order,
                            integral: DecisionIntegral::CandidatePosterior,
                            reason,
                        });
                        rank_path.push(step);
                        break;
                    }
                }
            }
            // A reference step the reference grid cannot take is answered by
            // refining that grid for the whole selection, not by stopping the
            // path at this rank.
            Err(refusal @ EventHistoryError::ReferenceStep { .. }) => return Err(refusal),
            Err(error) => {
                // No certified optimum at the next rank: the path stops with
                // the reason recorded rather than failing the whole fit.
                log::info!(
                    "[event-history] rank {rank} → {}: the evidence accepted the atom but its model reached no certified optimum, refused ({error})",
                    rank + 1
                );
                return Err(EventHistoryError::Fit { reason: format!(
                    "rank search unresolved at candidate rank {}: {error}", rank + 1) });
            }
        }
    }
    fit.rank_path = rank_path;
    fit.atom_evidence = atom_evidence;
    Ok(fit)
}

/// What a family evaluation that failed as text surfaces as: the reference
/// refusal that evaluation left on the family, taken, or else the text as a fit
/// failure. Every evaluation clears the refusal on entry, so a refusal an
/// earlier evaluation raised never types a later, unrelated failure.
fn typed_failure(family: &EventHistoryFamily, reason: String) -> EventHistoryError {
    family.take_reference_refusal().unwrap_or(EventHistoryError::Fit { reason })
}

/// The model at `atoms` from `start`, certified by [`fit_at_rank`]'s refinement
/// ladder from mesh refinement `from_refinement` up, with the certified setting
/// and the ladder's wall time logged.
fn certified_rank(
    cohort: &EventHistoryCohort,
    spec: &EventHistorySpec,
    atoms: usize,
    start: Option<&RankStart>,
    from_refinement: usize,
    reference_refinement: usize,
) -> Result<EventHistoryFit, EventHistoryError> {
    let started = std::time::Instant::now();
    let fit = fit_at_rank(cohort, spec, atoms, start, None, from_refinement, reference_refinement)?;
    log::info!(
        "[event-history] rank {atoms}: certified at Gauss-Hermite order {}, mesh refinement {} ({:.2} s)",
        fit.quadrature.gauss_hermite_order,
        fit.quadrature.mesh_refinement,
        started.elapsed().as_secs_f64()
    );
    Ok(fit)
}

/// The ladder a rank decision the incumbent's setting cannot resolve climbs.
#[derive(Clone, Copy, Debug)]
enum Rung {
    /// The Gauss-Hermite order, `2·order − 1`.
    GaussHermite,
    /// The time mesh, one refinement.
    Mesh,
}

/// The incumbent refitted one ladder rung up, the Gauss-Hermite order
/// (`2·order − 1`) or the time mesh (one refinement), warm-started from its
/// own converged values, because an integral the rank decision reads is
/// unresolved at its certified setting. `None` at that ladder's top rung, the
/// top certifiable order ([`positivity_raise`]) or the mesh ceiling
/// ([`EventHistoryCohort::mesh_refinement_ceiling`]): the incumbent stays the
/// certified model and the decision is recorded as unresolved.
fn raise_incumbent(
    cohort: &EventHistoryCohort,
    rank_spec: &mut EventHistorySpec,
    fit: &EventHistoryFit,
    rung: Rung,
    reason: &str,
    reference_refinement: usize,
) -> Result<Option<EventHistoryFit>, EventHistoryError> {
    let rank = fit.rank();
    let order = fit.quadrature.gauss_hermite_order;
    let refinement = fit.quadrature.mesh_refinement;
    let from_refinement = match rung {
        Rung::GaussHermite => {
            let Some(next_order) =
                positivity_raise(order, fit.nodes.max_subject_nodes(), rank_spec.quadrature_tolerance)
            else {
                log::info!(
                    "[event-history] rank {rank} → {}: the decision is unresolved at Gauss-Hermite order {order} ({reason}), the ladder's top certifiable rung: the path stops at the certified rank-{rank} model with growth unresolved",
                    rank + 1
                );
                return Ok(None);
            };
            log::info!(
                "[event-history] rank {rank} → {}: the decision is unresolved at Gauss-Hermite order {order} ({reason}); refitting the incumbent at order {next_order}",
                rank + 1
            );
            rank_spec.gauss_hermite_order = next_order;
            refinement
        }
        Rung::Mesh => {
            let next = refinement + 1;
            if next > cohort.mesh_refinement_ceiling() {
                log::info!(
                    "[event-history] rank {rank} → {}: the decision is unresolved at mesh refinement {refinement} ({reason}), the mesh's top rung: the path stops at the certified rank-{rank} model with growth unresolved",
                    rank + 1
                );
                return Ok(None);
            }
            log::info!(
                "[event-history] rank {rank} → {}: the decision is unresolved at mesh refinement {refinement} ({reason}); refitting the incumbent at refinement {next}",
                rank + 1
            );
            next
        }
    };
    let start = RankStart::carried(
        fit.fit.block_states[..fit.marks()]
            .iter()
            .map(|s| s.beta.clone())
            .collect(),
        fit.loadings.iter().copied().collect(),
        fit.log_rates.clone(),
        fit.atom_log_lambdas.clone(),
        fit.rate_held.clone(),
    );
    certified_rank(cohort, rank_spec, rank, Some(&start), from_refinement, reference_refinement).map(Some)
}

/// Fit and select structure under one reference-normalised objective, then
/// verify reference discretisation at fixed coefficients. If unresolved,
/// repeat selection under the refined objective; never reinterpret a rank
/// selected under stationary-prior centring as reference-law evidence.
pub(crate) fn fit_event_history(
    cohort: &mut EventHistoryCohort, spec: &EventHistorySpec,
) -> Result<EventHistoryFit, EventHistoryError> {
    if !(spec.reference_tolerance.is_finite() && spec.reference_tolerance > 0.0) {
        return Err(EventHistoryError::InvalidInput { reason: "reference tolerance must be finite and positive".to_string() });
    }
    let mut discrepancies = Vec::new();
    let mut fitting_spec = spec.clone();
    let mut refinement = 2;
    for _ in 0..16 {
        let mut fit = match fit_event_history_on_grid(cohort, &fitting_spec, refinement) {
            Ok(fit) => fit,
            // The reference midpoint map does not contract at this grid's step
            // length, and a finer grid shortens the step.
            Err(refusal @ EventHistoryError::ReferenceStep { .. }) => {
                log::info!("[event-history] reference refinement {refinement}: {refusal}; refining the reference grid");
                refinement += 1;
                if refinement > 10 { return Err(refusal); }
                continue;
            }
            Err(error) => return Err(error),
        };
        let Some(strata) = spec.reference.as_ref() else { return Ok(fit); };
        let refined = reference_tables(cohort, strata, &fit.frozen_specs,
            spec.quadrature_order, refinement + 1, &fit.nodes)?;
        let fine_family = fit.family.clone().with_reference(Some(Arc::new(refined)));
        let fine = fine_family.refresh_normaliser(&fit.fit.block_states)?;
        let coarse_centring = fit.centring.as_ref().ok_or_else(|| EventHistoryError::Fit {
            reason: "reference fit is missing its centring values".to_string(),
        })?;
        let time_gap = coarse_centring.discrepancy(&fine, fit.marks())?;
        let next_order = fit.family.gh.order + 4;
        let latent_gap = if fit.rank() == 0 { 0.0 } else {
            preflight(next_order, fit.rank(), fine.grid.len(), fit.marks(), fit.family.total_width())?;
            let mut latent_family = fine_family.clone();
            latent_family.gh = Arc::new(GaussHermite::new(next_order)?);
            if !latent_family.held_rates.iter().all(|r| *r == Some(0.0))
                && latent_family.gh.lebesgue_constant * f64::EPSILON * fine.grid.len() as f64
                    > spec.reference_tolerance {
                return Err(EventHistoryError::NumericalFailure { reason:
                    "reference latent quadrature cannot be refined within its interpolation roundoff bound".to_string() });
            }
            let latent = latent_family.refresh_normaliser(&fit.fit.block_states)?;
            fine.discrepancy(&latent, fit.marks())?
        };
        let gap = time_gap + latent_gap;
        discrepancies.push(gap);
        log::info!("[event-history] reference refinement {refinement}: time discrepancy {time_gap:.3e}, latent discrepancy {latent_gap:.3e} nats");
        if gap <= spec.reference_tolerance {
            fit.reference_certificate = Some(gap);
            fit.reference_refinements = discrepancies;
            return Ok(fit);
        }
        if latent_gap > time_gap {
            fitting_spec.gauss_hermite_order = next_order;
        } else {
            refinement += 1;
            if refinement > 10 { break; }
        }
    }
    Err(EventHistoryError::NumericalFailure {
        reason: format!("reference evolution unresolved after refinement: {:?}", discrepancies),
    })
}

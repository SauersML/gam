//! The event-history custom family: every subject's marginal likelihood
//! assembled into the coefficient space of the per-mark smooth blocks and the
//! latent block, with every derivative the outer LAML evaluator asks for
//! produced by the same generic code under a directional dual scalar, and
//! the fit driver that refines the Gauss-Hermite order and the time mesh
//! until the fitted coefficients are stationary under refinement.

use super::chain::{GaussHermite, product_grid_size};
use super::cohort::{
    CohortNodes, CovariateSegment, EventHistoryCohort, EventHistoryError, MarkKind, SubjectHistory,
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
use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, ExactNewtonJointGradientEvaluation,
    FamilyEvaluation, ParameterBlockSpec, ParameterBlockState, PenaltyMatrix, fit_custom_family,
};
use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix, SymmetricMatrix};
use gam_math::jet_scalar::{JetScalar, OneSeed, Order2, TwoSeed};
use gam_math::nested_dual::JetField;
use gam_model_api::families::custom_family::joint_coupled_coefficient_hessian_cost;
use gam_problem::CoefficientCoordinate;
use gam_solve::model_types::UnifiedFitResult;
use gam_terms::smooth::{
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design,
};
use ndarray::{Array1, Array2, ArrayView2, s};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

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

/// A one-direction dual at `value` with `ε`-component `u`.
pub fn seeded_one(value: f64, u: f64) -> OneSeed<0> {
    OneSeed::seeded(value, u, 0.0)
}

/// A two-direction dual at `value` with `ε`-component `u` and `δ`-component `v`.
pub fn seeded_two(value: f64, u: f64, v: f64) -> TwoSeed<0> {
    TwoSeed::seeded(value, u, v)
}

/// A full joint evaluation in coefficient space. `hessian` is the negative
/// log-likelihood Hessian, the convention the custom-family engine expects.
#[derive(Clone, Debug)]
pub struct JointEvaluation {
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
    /// The held risk-set normaliser `log M_d(t)` at every cohort node, index
    /// `row * marks + d`. Held as data over a solve and refreshed between
    /// solves; `None` centres on the stationary prior.
    log_normaliser: Option<Arc<Vec<f64>>>,
    /// The last joint evaluation, keyed on the exact state it was made at.
    cache: Arc<Mutex<Option<(Vec<f64>, Arc<JointEvaluation>)>>>,
}

/// The mesh refinement the reference population is run on.
///
/// It is deliberately not the fit's current refinement. The fit refines its
/// own mesh until the coefficients are stationary, and a normaliser held
/// across those rungs has to mean the same thing on each of them; a grid that
/// moved under the ladder would change the held values' length and their
/// meaning together. The reference grid's own resolution is therefore its own
/// numerical choice, and what it has to resolve is the population's decay,
/// which is smooth in time — not the event times, of which it has none.
const REFERENCE_REFINEMENT: usize = 0;

/// What one refresh of the risk-set centring produced: the normaliser to hold
/// over the next solve, and the reference population's own marginal survival.
#[derive(Clone, Debug)]
pub struct RiskSetCentring {
    /// `log M_d(t)` on the reference grid, index
    /// `(stratum * nodes + node) * marks + d`. It is held on that grid rather
    /// than on the cohort's nodes because the fit refines its own mesh
    /// between solves while this must mean the same thing across them.
    pub log_normaliser: Vec<f64>,
    /// The log of the reference population's risk-set mass at every node of
    /// the reference grid, index `(stratum * nodes + node) * masks + mask`.
    /// For a single first-occurrence mark this is `−∫ e^{η⁰}` exactly, which
    /// is the identity the centring exists for and the check that the
    /// baseline is the marginal rate rather than a mixture's intercept.
    pub log_risk_mass: Vec<f64>,
    /// How many distinct risk sets the marks define.
    pub masks: usize,
    /// The risk set of every mark, indexing the last axis of `log_risk_mass`.
    pub mask_of_mark: Vec<usize>,
}

impl ReferenceTables {
    /// Carry a normaliser held on the reference grid onto a node set, by the
    /// linear interpolation each node recorded when the tables were built.
    fn carry_to_nodes(
        &self,
        held: &[f64],
        marks: usize,
        total_nodes: usize,
    ) -> Result<Vec<f64>, EventHistoryError> {
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
        let mut out = vec![0.0; total_nodes * marks];
        for row in 0..total_nodes {
            let base = self.node_stratum[row] * nodes;
            let lower = (base + self.node_lower[row]) * marks;
            let upper = (base + self.node_lower[row] + 1) * marks;
            let weight = self.node_weight[row];
            for d in 0..marks {
                let low = held[lower + d];
                out[row * marks + d] = low + weight * (held[upper + d] - low);
            }
        }
        Ok(out)
    }
}

/// The reference population's own grid, its per-stratum design rows, and
/// where every cohort node sits on that grid.
pub struct ReferenceTables {
    pub grid: ReferenceGrid,
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
                .any(|r| !(r.is_finite() && *r > 0.0))
        {
            return Err(EventHistoryError::InvalidInput {
                reason: format!(
                    "event-history family needs one positive finite or free rate per atom: got {:?} for {atoms} atoms",
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
            log_normaliser: None,
            cache: Arc::new(Mutex::new(None)),
        })
    }

    /// The same family with a reference population attached and a normaliser
    /// held over the solve.
    ///
    /// The held normaliser arrives on the reference grid and is carried here
    /// onto this family's own nodes, which is what lets one held value serve
    /// every rung of the mesh ladder.
    pub fn with_reference(
        mut self,
        reference: Option<Arc<ReferenceTables>>,
        held: Option<Arc<Vec<f64>>>,
    ) -> Result<Self, EventHistoryError> {
        self.log_normaliser = match (reference.as_ref(), held.as_ref()) {
            (Some(tables), Some(values)) => Some(Arc::new(tables.carry_to_nodes(
                values,
                self.marks(),
                self.nodes.total_nodes,
            )?)),
            _ => None,
        };
        self.reference = reference;
        self.cache = Arc::new(Mutex::new(None));
        Ok(self)
    }

    /// Whether the baselines are centred on the risk sets rather than on the
    /// stationary prior.
    pub fn risk_set_centred(&self) -> bool {
        self.log_normaliser.is_some()
    }

    /// The reference population's tables, when the fit has one.
    pub fn reference(&self) -> Option<&Arc<ReferenceTables>> {
        self.reference.as_ref()
    }

    /// The normaliser carried onto this family's nodes, for comparing two
    /// reference grids on one common set of times.
    pub fn normaliser_on_nodes(&self, held: &[f64]) -> Result<Vec<f64>, EventHistoryError> {
        let tables = self
            .reference
            .as_ref()
            .ok_or_else(|| EventHistoryError::InvalidInput {
                reason: "this family has no reference population".to_string(),
            })?;
        tables.carry_to_nodes(held, self.marks(), self.nodes.total_nodes)
    }

    /// Recompute the risk-set normaliser at a coefficient state: run the
    /// reference population of every stratum forward and read `log M_d(t)`
    /// off it, then carry it to the cohort's own nodes.
    ///
    /// The normaliser is a population quantity — its cost is the reference
    /// grid, not the cohort — and it is returned as data, to be held over the
    /// next solve.
    pub fn refresh_normaliser(
        &self,
        states: &[ParameterBlockState],
    ) -> Result<RiskSetCentring, String> {
        let tables = self
            .reference
            .as_ref()
            .ok_or_else(|| "this family has no reference population".to_string())?;
        let marks = self.marks();
        let atoms = self.atoms;
        let nodes = tables.grid.len();
        let empty = Array1::<f64>::zeros(0);
        let latent: &Array1<f64> = if self.has_latent_block() {
            &states[marks].beta
        } else {
            &empty
        };
        let loadings: Vec<f64> = latent.iter().take(marks * atoms).copied().collect();
        let rates = self.atom_rates(latent);
        let mut per_stratum: Vec<Vec<f64>> = Vec::with_capacity(tables.strata);
        let mut risk_mass: Vec<f64> = Vec::new();
        let (_, mask_of_mark) = super::preserve::killing_masks(&tables.kinds);
        let mut masks = 0usize;
        for s in 0..tables.strata {
            let mut eta0 = vec![0.0; nodes * marks];
            for d in 0..marks {
                let design = &tables.designs[d];
                let beta = &states[d].beta;
                for n in 0..nodes {
                    let row = s * nodes + n;
                    let mut value = tables.offsets[d][row];
                    for (j, x) in design.row(row).iter().enumerate() {
                        value += x * beta[j];
                    }
                    eta0[n * marks + d] = value;
                }
            }
            let out = stratum_normalisers(
                &tables.grid,
                &eta0,
                &loadings,
                &rates,
                self.time_scale,
                &self.gh,
                &tables.kinds,
                atoms,
            )
            .map_err(|error| error.to_string())?;
            masks = out.masks;
            risk_mass.extend(out.log_risk_mass);
            per_stratum.push(out.log_normaliser);
        }
        Ok(RiskSetCentring {
            log_normaliser: per_stratum.concat(),
            log_risk_mass: risk_mass,
            masks,
            mask_of_mark,
        })
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
    pub fn atom_rates(&self, latent_beta: &Array1<f64>) -> Vec<f64> {
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
    pub fn latent_width(&self) -> usize {
        self.marks() * self.atoms + self.held_rates.iter().filter(|h| h.is_none()).count()
    }

    /// Whether the fit carries a latent block (no atoms means a plain
    /// Poisson-process GAM with the same node expansion).
    pub fn has_latent_block(&self) -> bool {
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
        let held = self.log_normaliser.as_deref().map(Vec::as_slice);
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
                // Held as data over the solve: no derivative channels, so
                // `∂η/∂a` is `z` rather than `z − a` (see `subject_marginal`).
                let normaliser: Option<Vec<S>> = held.map(|values| {
                    values[first * marks..(first + n) * marks]
                        .iter()
                        .map(|v| S::seeded(*v, 0.0, 0.0))
                        .collect()
                });
let inputs = SubjectInputs {
                    nodes: subject,
                    eta0: &eta0,
                    loadings: &loadings,
                    rates: &rates,
                    time_scale,
                    gh,
                    continuation_gap: 0.0,
                    designs: derivatives.then_some(views.as_slice()),
                    log_normaliser: normaliser.as_deref(),
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
        let total = self.total_width();
        let mut gradient = vec![0.0; total];
        if total <= 4 {
            self.exact_gradient_chunks::<4>(states, &mut gradient)?;
        } else if total <= 8 {
            self.exact_gradient_chunks::<8>(states, &mut gradient)?;
        } else if total <= 16 {
            self.exact_gradient_chunks::<16>(states, &mut gradient)?;
        } else if total <= 32 {
            self.exact_gradient_chunks::<32>(states, &mut gradient)?;
        } else {
            self.exact_gradient_chunks::<64>(states, &mut gradient)?;
        }
        Ok(gradient)
    }

    /// One `W`-wide sweep of tangent slots at a time over the coefficient
    /// vector, each sweep a full forward filter per subject.
    fn exact_gradient_chunks<const W: usize>(
        &self,
        states: &[ParameterBlockState],
        gradient: &mut [f64],
    ) -> Result<(), String> {
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
        let designs = &self.designs;
        let held = self.log_normaliser.as_deref().map(Vec::as_slice);
        let gh = &self.gh;
        let time_scale = self.time_scale;
        for start in (0..total).step_by(W) {
            let end = (start + W).min(total);
            let seed = |slot: usize, grad: &mut [f64; W], x: f64| {
                if (start..end).contains(&slot) {
                    grad[slot - start] = x;
                }
            };
            let unit = |q: usize| -> Tangent<W> {
                let mut grad = [0.0; W];
                seed(latent_offset + q, &mut grad, 1.0);
                Tangent::seeded(latent_beta[q], grad)
            };
            let loadings: Vec<Tangent<W>> = (0..marks * atoms).map(unit).collect();
            let slots = self.free_rate_slots();
            let band = self.rate_band;
            let rates: Vec<Tangent<W>> = (0..atoms)
                .map(|k| match slots[k] {
                    Some(slot) => rate_from_chart(band, &unit(slot)),
                    None => Tangent::seeded(
                        self.held_rates[k].expect("a rate without a coefficient is held"),
                        [0.0; W],
                    ),
                })
                .collect();
            let per_subject: Result<Vec<[f64; W]>, String> = self
                .nodes
                .subjects
                .par_iter()
                .map(|subject| {
                    let n = subject.len();
                    let first = subject.first_row;
                    let mut eta0 = Vec::with_capacity(n * marks);
                    for node in 0..n {
                        let row = first + node;
                        for d in 0..marks {
                            let mut grad = [0.0; W];
                            for (j, x) in designs[d].row(row).iter().enumerate() {
                                seed(offsets[d] + j, &mut grad, *x);
                            }
                            eta0.push(Tangent::seeded(states[d].eta[row], grad));
                        }
                    }
                    // Held as data over the solve: no derivative channels, so
                    // `∂η/∂a` is `z` rather than `z − a` (see `subject_marginal`).
                    let normaliser: Option<Vec<Tangent<W>>> = held.map(|values| {
                        values[first * marks..(first + n) * marks]
                            .iter()
                            .map(|v| Tangent::seeded(*v, [0.0; W]))
                            .collect()
                    });
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
                    let local = subject_marginal(&inputs, false).map_err(|e| e.to_string())?;
                    Ok(local.loglik.grad)
                })
                .collect();
            for g in per_subject? {
                for (slot, x) in g.iter().enumerate().take(end - start) {
                    gradient[start + slot] += x;
                }
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
    ) -> Result<(f64, f64), String> {
        self.validate_states(states)?;
        let marks = self.marks();
        let atoms = self.atoms;
        let offsets = self.block_offsets();
        let total = self.total_width();
        if direction.len() != total {
            return Err(format!(
                "event-history direction has length {}, expected {total}",
                direction.len()
            ));
        }
        let latent_offset = offsets[marks];
        let empty = Array1::<f64>::zeros(0);
        let latent_beta: &Array1<f64> = if self.has_latent_block() {
            &states[marks].beta
        } else {
            &empty
        };
        let designs = &self.designs;
        let held = self.log_normaliser.as_deref().map(Vec::as_slice);
        let gh = &self.gh;
        let time_scale = self.time_scale;
        let slots = self.free_rate_slots();
        let seeded =
            |q: usize| Tangent::<1>::seeded(latent_beta[q], [direction[latent_offset + q]]);
        let loadings: Vec<Tangent<1>> = (0..marks * atoms).map(seeded).collect();
        let band = self.rate_band;
        let rates: Vec<Tangent<1>> = (0..atoms)
            .map(|k| match slots[k] {
                Some(slot) => rate_from_chart(band, &seeded(slot)),
                None => Tangent::seeded(
                    self.held_rates[k].expect("a rate without a coefficient is held"),
                    [0.0],
                ),
            })
            .collect();
        let per_subject: Result<Vec<(f64, f64)>, String> = self
            .nodes
            .subjects
            .par_iter()
            .map(|subject| {
                let n = subject.len();
                let first = subject.first_row;
                let mut eta0 = Vec::with_capacity(n * marks);
                for node in 0..n {
                    let row = first + node;
                    for d in 0..marks {
                        let slope: f64 = designs[d]
                            .row(row)
                            .iter()
                            .enumerate()
                            .map(|(j, x)| x * direction[offsets[d] + j])
                            .sum();
                        eta0.push(Tangent::seeded(states[d].eta[row], [slope]));
                    }
                }
                // Held as data over the solve: no derivative channels, so
                // `∂η/∂a` is `z` rather than `z − a` (see `subject_marginal`).
                let normaliser: Option<Vec<Tangent<1>>> = held.map(|values| {
                    values[first * marks..(first + n) * marks]
                        .iter()
                        .map(|v| Tangent::seeded(*v, [0.0]))
                        .collect()
                });
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
                let local = subject_marginal(&inputs, false).map_err(|e| e.to_string())?;
                Ok((local.loglik.value, local.loglik.grad[0]))
            })
            .collect();
        let per_subject = per_subject?;
        let values: Vec<f64> = per_subject.iter().map(|(v, _)| *v).collect();
        let slopes: Vec<f64> = per_subject.iter().map(|(_, s)| *s).collect();
        Ok((pairwise_sum(&values, &0.0), pairwise_sum(&slopes, &0.0)))
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
        let held = self.log_normaliser.as_deref().map(Vec::as_slice);
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
                        None,
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
    pub fn joint_evaluation(
        &self,
        states: &[ParameterBlockState],
    ) -> Result<Arc<JointEvaluation>, String> {
        let key = Self::state_key(states);
        if let Ok(guard) = self.cache.lock()
            && let Some((k, value)) = guard.as_ref()
            && *k == key
        {
            return Ok(Arc::clone(value));
        }
        let (loglik, _, hessian) = self.evaluate_generic::<f64>(states, None, None, true)?;
        let gradient = self.exact_gradient(states)?;
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
        let (loglik, _, _) = self.evaluate_generic::<f64>(states, None, None, false)?;
        Ok(loglik)
    }

    /// `D_β H[u]` for the negative log-likelihood Hessian `H`.
    pub fn directional_hessian(
        &self,
        states: &[ParameterBlockState],
        u: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
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
    pub fn second_directional_hessian(
        &self,
        states: &[ParameterBlockState],
        u: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
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

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        joint_coupled_coefficient_hessian_cost(self.nodes.total_nodes as u64, specs)
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
pub struct EventHistorySpec {
    /// One covariate/time term collection per mark, or a single one shared by
    /// every mark. Feature columns index the node data matrix: the covariate
    /// table's columns followed by the node time.
    pub covariates: Vec<TermCollectionSpec>,
    /// Gauss-Legendre order per mesh cell.
    pub quadrature_order: usize,
    /// Starting Gauss-Hermite order per latent axis.
    pub gauss_hermite_order: usize,
    /// The certificate's tolerance: the largest shift any fitted coefficient
    /// may make under a refinement of the Gauss-Hermite order or of the time
    /// mesh, in units of that coefficient's posterior standard deviation.
    /// The order is doubled and the mesh halved until both shifts are below
    /// it, so a fit is never an artefact of its discretisation at a level
    /// the data could resolve. The default, a twentieth of a posterior
    /// standard deviation, is the scale at which a shift cannot change any
    /// inference the fit supports: it is a twentieth of the width the data
    /// itself leaves undetermined.
    pub quadrature_tolerance: f64,
    /// The reference population every mark's baseline is the marginal rate
    /// of. `None` centres the latent term on the stationary prior, so
    /// `exp(η⁰)` is the intensity averaged over everybody the cohort started
    /// with; `Some` centres it on the risk set at every age, so `exp(η⁰)` is
    /// the incidence among those still at risk (see [`super::preserve`]).
    pub reference: Option<ReferenceStrata>,
    /// How many times the held risk-set normaliser is refreshed before the
    /// fit stops, however far it is still moving.
    pub normaliser_rounds: usize,
    /// How far the held risk-set normaliser may still move, in nats, before
    /// the fit stops refreshing it. The normaliser is held as data over each
    /// solve — its own score has expectation zero — and refreshed between
    /// them, so the fit is a fixed point of that alternation.
    pub normaliser_tolerance: f64,
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
            normaliser_rounds: 24,
            normaliser_tolerance: 1e-4,
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
    /// How far the held risk-set normaliser moved at each re-centring round,
    /// in nats. Empty when the baselines are centred on the stationary prior.
    pub normaliser_rounds: Vec<f64>,
    /// The log of the reference population's risk-set mass on the reference
    /// grid — its marginal survival, which the centring makes `−∫ e^{η⁰}`.
    /// Index `(stratum * nodes + node) * masks + mask`.
    pub reference_risk_mass: Vec<f64>,
    /// How many distinct risk sets the marks define.
    pub reference_masks: usize,
    /// The largest disagreement, in nats, between the reference grid the
    /// normaliser was taken on and the same grid with every cell halved.
    /// `None` when the baselines are centred on the stationary prior.
    pub reference_certificate: Option<f64>,
    /// The settled risk-set normaliser on the reference grid, index
    /// `(stratum * nodes + node) * marks + d`. Empty when the baselines are
    /// centred on the stationary prior. Prediction reads it: a forecast made
    /// from a fit whose baselines are the risk sets' rates has to divide by
    /// the same normaliser the fit did, at the times the forecast asks about,
    /// or it is a forecast from a different model.
    pub reference_normaliser: Vec<f64>,
}

impl EventHistoryFit {
    /// The reference grid the settled normaliser lives on, when the baselines
    /// are the risk sets' rates.
    pub fn reference_grid(&self) -> Option<&ReferenceGrid> {
        self.family.reference().map(|tables| &tables.grid)
    }

    /// `log M_d(t)` for one stratum at an arbitrary time, by the same linear
    /// interpolation in the log of the normaliser the fit used. Empty when
    /// the baselines are centred on the stationary prior.
    pub fn risk_set_normaliser_at(&self, stratum: usize, t: f64) -> Vec<f64> {
        let marks = self.marks();
        let Some(grid) = self.reference_grid() else {
            return Vec::new();
        };
        if self.reference_normaliser.is_empty() {
            return Vec::new();
        }
        let nodes = grid.len();
        let (lower, weight) = grid.locate(t);
        let base = stratum * nodes;
        (0..marks)
            .map(|d| {
                let low = self.reference_normaliser[(base + lower) * marks + d];
                let high = self.reference_normaliser[(base + lower + 1) * marks + d];
                low + weight * (high - low)
            })
            .collect()
    }

    /// The rank of the latent covariance the evidence supports.
    pub fn rank(&self) -> usize {
        self.log_rates.len()
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

    /// Fitted coefficients of the mark-`d` block: the population
    /// log-intensity surface's coefficients (see [`Self::mark_eta`]).
    pub fn mark_coefficients(&self, d: usize) -> &Array1<f64> {
        &self.fit.block_states[d].beta
    }

    /// Population log-intensity `η⁰` of mark `d` on the training nodes:
    /// `exp(η⁰)` is the intensity averaged over the latent state, since the
    /// latent term enters as `−½|a_d|² + a_d · z`, whose Gaussian mixing the
    /// shift cancels exactly (`docs/event-history.md` derives it).
    pub fn mark_eta(&self, d: usize) -> &Array1<f64> {
        &self.fit.block_states[d].eta
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
pub fn latent_block_spec(
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
pub fn mark_block_spec(name: &str, design: &TermCollectionDesign) -> ParameterBlockSpec {
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

/// Fit an event-history model from a formula right-hand side shared by every
/// mark, such as `x + s(time)`.
pub fn fit_event_history_formula(
    cohort: &mut EventHistoryCohort,
    formula: &str,
    options: BlockwiseFitOptions,
) -> Result<EventHistoryFit, EventHistoryError> {
    fit_event_history_formulas(cohort, std::slice::from_ref(&formula), options)
}

/// Fit an event-history model from formula right-hand sides: one formula
/// that every mark uses (with its own coefficients), or one formula per mark
/// in the cohort's mark order, so that a mark's log-intensity carries only
/// the terms that belong to it — a disease its own score, not every score
/// of every other disease.
pub fn fit_event_history_formulas<F: AsRef<str>>(
    cohort: &mut EventHistoryCohort,
    formulas: &[F],
    options: BlockwiseFitOptions,
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

/// Bytes the family's evaluation may hold at once: the transient `S × S`
/// backward kernel of one gap, the carried `P × S` conditional expectations,
/// the per-node densities and operators of every node, per parallel
/// subject, in the widest scalar the outer solve uses (four channels).
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
    let per_subject = s * s
        + total_width as f64 * s
        + n * s * (4.0 + marks as f64)
        + n * atoms as f64 * 3.0 * g * g;
    let channels = 4.0;
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
pub struct RankStart {
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
    /// The score's top eigenvalue at the proposed rate: the second-order
    /// evidence slope along the proposed direction.
    pub score_eigenvalue: f64,
    /// `μ² / (4J)` of the top direction: its second-order evidence gain in
    /// nats, the matched-filter statistic the rate maximises.
    pub standardised_gain: f64,
    /// The log-rate the proposal named.
    pub proposed_log_rate: f64,
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
    /// The evidence the prior buys over the current rank, in nats: the
    /// marginal likelihood at `λ̂` against the atom pinned to zero, under the
    /// score's quartic model of the evidence. This is the number the
    /// decision was made on.
    pub evidence_gain: f64,
    /// The realised increase of the marginal log-likelihood at the mode from
    /// the rank before to the fitted candidate; zero for a refused atom. The
    /// quartic model is exact at the boundary, where the decision is made,
    /// and only a model further out, so the two differ for a strong atom.
    pub log_likelihood_gain: f64,
    /// The prior places the loading's posterior mode away from zero, so the
    /// atom was fitted.
    pub accepted: bool,
    /// Whether the rank-`K+1` model reached a certified optimum. A
    /// candidate that cannot be fitted is refused: a fit object may only
    /// come from a converged optimisation, so an atom whose model has no
    /// certified optimum is not one the fit can carry.
    pub converged: bool,
}

/// The reference population's grid, its per-stratum designs, and where every
/// cohort node sits on that grid.
///
/// The grid is one window spanning every subject's follow-up, expanded by the
/// same node expansion the fit uses, for one pseudo-subject per stratum
/// holding that stratum's covariate row. Every stratum therefore shares the
/// node times, and a stratum's rows sit contiguously in the design, which is
/// what makes the normaliser one lookup per cohort node.
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
    let mut population = EventHistoryCohort {
        mark_names: cohort.mark_names.clone(),
        mark_kinds: cohort.mark_kinds.clone(),
        covariate_names: cohort.covariate_names.clone(),
        covariate_levels: cohort.covariate_levels.clone(),
        covariates: cohort.covariates.clone(),
        subjects: strata
            .rows
            .iter()
            .enumerate()
            .map(|(s, &row)| SubjectHistory {
                id: format!("reference-{s}"),
                entry,
                exit,
                events: Vec::new(),
                segments: vec![CovariateSegment { start: entry, row }],
            })
            .collect(),
    };
    population.validate()?;
    let expanded = expand_nodes(&population, quadrature_order, refinement)?;
    let first = &expanded.subjects[0];
    let grid = ReferenceGrid {
        times: first.times.clone(),
        gaps: first.gaps.clone(),
        weights: first.weights.clone(),
    };
    if expanded
        .subjects
        .iter()
        .any(|subject| subject.times != first.times)
    {
        return Err(EventHistoryError::NumericalFailure {
            reason: "the reference strata were expanded onto different node times".to_string(),
        });
    }
    let mut designs = Vec::with_capacity(marks);
    let mut offsets = Vec::with_capacity(marks);
    for d in 0..marks {
        let design = build_term_collection_design(expanded.node_data.view(), &frozen_specs[d])
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
            let (lower, weight) = grid.locate(time);
            node_stratum[row] = strata.subject[i];
            node_lower[row] = lower;
            node_weight[row] = weight;
        }
    }
    Ok(ReferenceTables {
        grid,
        kinds: cohort.mark_kinds.clone(),
        designs,
        offsets,
        strata: strata.rows.len(),
        node_stratum,
        node_lower,
        node_weight,
    })
}

/// The certified fit at ONE rank: the Gauss-Hermite order and the time mesh
/// are refined until no fitted coefficient moves by more than the
/// certificate's tolerance. `start` warm-starts every block from a fit one
/// rank down, with the new atom's loadings at the covariance score's
/// proposal.
fn fit_at_rank(
    cohort: &EventHistoryCohort,
    spec: &EventHistorySpec,
    atoms: usize,
    start: Option<&RankStart>,
    pinned: Option<(usize, usize)>,
    held: Option<Arc<Vec<f64>>>,
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
                REFERENCE_REFINEMENT,
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
        let family = family.with_reference(reference, held.clone())?;
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
            .map_err(|reason| EventHistoryError::Fit { reason })?;
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
            .map_err(|reason| EventHistoryError::Fit { reason })?;
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
        None => (spec.gauss_hermite_order.max(3), 0usize),
    };
    // The grid is raised once for a posterior it cannot represent. A second
    // raise would climb into the orders where the Lagrange interpolant's own
    // roundoff is the larger error, so a posterior that survives one raise is
    // reported rather than chased.
    let mut positivity_raises = 0usize;
    let mut built = build(order, refinement)?;
    loop {
        let fit = match fit_custom_family(&built.family, &built.specs, &options) {
            Ok(fit) => fit,
            Err(error) => {
                // A posterior the grid cannot represent (its interpolant goes
                // negative where the mass is) is answered by resolving the
                // grid, which is the ladder this driver already owns — not by
                // handing the caller a number the representation could not
                // carry. Anything else is the caller's to see.
                let message = error.to_string();
                let next_order = 2 * order - 1;
                let admissible = atoms > 0
                    && positivity_raises == 0
                    && GaussHermite::new(next_order).is_ok_and(|rule| {
                        rule.lebesgue_constant
                            * f64::EPSILON
                            * built.nodes.max_subject_nodes() as f64
                            <= spec.quadrature_tolerance
                    });
                if message.contains(LOST_POSITIVITY) && admissible {
                    log::info!(
                        "[event-history] Gauss-Hermite order {order} cannot represent a posterior on this cohort; raising it to {next_order}"
                    );
                    order = next_order;
                    positivity_raises += 1;
                    built = build(order, refinement)?;
                    continue;
                }
                return Err(EventHistoryError::Fit {
                    reason: format!(
                        "event-history LAML fit at Gauss-Hermite order {order}, mesh refinement {refinement}: {message}"
                    ),
                });
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
            .map_err(|reason| EventHistoryError::Fit { reason })?;
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
            if rule.lebesgue_constant * f64::EPSILON * max_nodes as f64 > spec.quadrature_tolerance
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
        normaliser_rounds: Vec::new(),
        reference_risk_mass: Vec::new(),
        reference_masks: 0,
        reference_certificate: None,
        reference_normaliser: Vec::new(),
    })
}

/// The exact profile of the marginal log-likelihood along a proposed atom's
/// direction: `g(t) = ℓ(t·v) − ℓ(0)`, with every other coefficient at the
/// rank-`K` fit and the new atom's rate at the proposal, sampled with its
/// slope by one tangent forward filter per point from `t = 0` outward until
/// the profile has fallen far below its peak. The score's quartic model is
/// exact at the boundary, where the atom is judged, and only a model
/// further out; the prior an accepted atom enters with, and the evidence it
/// reports, are read from this profile instead.
fn direction_profile(
    fit: &EventHistoryFit,
    atom: &NewAtom,
    time_scale: f64,
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
        )
    };
    let mut order = fit.family.gh.order;
    let mut probe = build(order)?;
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
    let fit_error = |reason: String| EventHistoryError::Fit { reason };
    let base = probe.log_likelihood(&states_at(0.0)).map_err(fit_error)?;
    // Sampled at an eighth of the quartic model's own mode scale, out to where
    // the profile under the Laplace-scale prior of that mode has fallen
    // twenty nats (`e⁻²⁰` of the peak's mass) below its peak past that peak;
    // the step doubles once the sample count grows long. A loading far past
    // the mode can tilt a node's posterior more sharply than the grid
    // represents (its interpolant loses positivity); the order is raised once
    // for that, as the fit does, and if the representation still fails the
    // profile ends at its last sample and the interpolant's continuation
    // carries what little tail is left.
    let prior = 1.0 / (atom.ridge.mode_scale * atom.ridge.mode_scale);
    let mut step = atom.ridge.mode_scale / 8.0;
    let mut points = vec![0.0];
    let mut values = vec![0.0];
    let mut slopes = vec![0.0];
    let mut peak = 0.0_f64;
    let mut t = 0.0;
    loop {
        t += step;
        let sample = probe.directional_log_likelihood(&states_at(t), &direction);
        let (value, slope) = match sample {
            Ok(sample) => sample,
            Err(message) => {
                let next_order = 2 * order - 1;
                if message.contains(LOST_POSITIVITY) && order == fit.family.gh.order {
                    log::info!(
                        "[event-history] the profile along the proposed direction at loading scale {t:.3} needs Gauss-Hermite order {next_order} ({message})"
                    );
                    order = next_order;
                    probe = build(order)?;
                    t -= step;
                    continue;
                }
                if points.len() >= 2 {
                    log::info!(
                        "[event-history] the profile along the proposed direction ends at loading scale {:.3}, {:.2} nats below its peak: the grid cannot represent the posterior beyond it ({message})",
                        points[points.len() - 1],
                        peak - values[values.len() - 1]
                    );
                    break;
                }
                return Err(fit_error(message));
            }
        };
        let value = value - base;
        peak = peak.max(value - 0.5 * prior * t * t);
        points.push(t);
        values.push(value);
        slopes.push(slope);
        if slope < 0.0 && value - 0.5 * prior * t * t < peak - 20.0 {
            break;
        }
        if points.len() % 128 == 0 {
            step *= 2.0;
        }
        if points.len() > 1024 {
            return Err(fit_error(format!(
                "the log-likelihood along the proposed direction has not fallen below its peak after {} samples out to t = {t:.3e}",
                points.len()
            )));
        }
    }
    Ok(DirectionProfile {
        points,
        values,
        slopes,
    })
}

/// Fit an event-history model, growing the rank of the latent covariance
/// from zero until the evidence refuses the next direction.
///
/// At each rank the next atom is proposed by the covariance score of the
/// fit's martingale residuals: the direction and rate whose standardised
/// evidence gain is largest, the most evidence-improving covariance
/// direction the current rank omits. The atom's loadings get the isotropic
/// Gaussian prior whose precision maximises the marginal likelihood under
/// the exact one-dimensional marginal of the score's quartic evidence model
/// along every direction (see [`super::covariance::empirical_bayes_ridge`]),
/// and the atom is accepted exactly when that prior places the posterior
/// mode of its loading away from zero. A refused atom costs no fit; an
/// accepted one is fitted with its prior held fixed, warm-started at the
/// mode, and a candidate that reaches no certified optimum is refused with
/// its reason. Nothing about the latent structure is chosen by hand: not
/// the number of atoms, not their directions, not their rates, not the
/// strength of their priors, and no level or tolerance decides the rank.
///
/// The rank path runs at ONE setting — the spec's own starting order and
/// mesh — and only the fit that is returned runs the refinement ladder,
/// once, at the rank the evidence chose.
pub fn fit_event_history(
    cohort: &mut EventHistoryCohort,
    spec: &EventHistorySpec,
) -> Result<EventHistoryFit, EventHistoryError> {
    cohort.validate()?;
    let marks = cohort.marks();
    if let Some(strata) = &spec.reference {
        strata.validate(cohort.subjects.len(), cohort.covariates.nrows())?;
    }
    let time_scale = cohort.time_scale();
    let held: Option<Arc<Vec<f64>>> = None;
    let pin = Some((spec.gauss_hermite_order.max(3), 0usize));
    let mut fit = fit_at_rank(cohort, spec, 0, None, pin, None)?;
    let mut rank_path: Vec<RankStep> = Vec::new();
    let mut atom_evidence: Vec<f64> = Vec::new();
    loop {
        let rank = fit.rank();
        let residuals = fit
            .family
            .residuals(&fit.fit.block_states)
            .map_err(|reason| EventHistoryError::Fit { reason })?;
        let Some(mut atom) = best_new_atom(&residuals, marks, time_scale, fit.family.rate_band())?
        else {
            break;
        };
        let boundary = atom.clone();
        if atom.ridge.accepted {
            // The quartic decision stands; the prior the atom enters with and
            // the evidence it reports come from the exact profile along its
            // direction, which the same empirical-Bayes calculation reads in
            // place of the quartic model of that one direction.
            let profile = direction_profile(&fit, &atom, time_scale)?;
            let directions: Vec<DirectionEvidence> =
                std::iter::once(DirectionEvidence::Exact(profile))
                    .chain(
                        atom.other_directions
                            .iter()
                            .map(|&(eigenvalue, information)| DirectionEvidence::Quartic {
                                eigenvalue,
                                information,
                            }),
                    )
                    .collect();
            let refined = empirical_bayes_ridge(&directions);
            log::info!(
                "[event-history] rank {rank} → {}: exact profile along the proposed direction: prior log-precision {:.3} → {:.3}, evidence {:.3} → {:.3} nats, mode scale {:.4} → {:.4}",
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
        }
        let mut step = RankStep {
            rank,
            score_eigenvalue: atom.eigenvalue,
            standardised_gain: atom.standardised_gain,
            proposed_log_rate: atom.log_rate,
            at_resolution_limit: atom.at_upper_limit,
            rate_held: atom.rate_held(),
            ridge_log_lambda: atom.ridge.log_lambda,
            evidence_gain: atom.ridge.gain,
            log_likelihood_gain: 0.0,
            accepted: atom.ridge.accepted,
            converged: true,
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
        let mut grown = fit_at_rank(cohort, spec, rank + 1, Some(&start), pin, held.clone());
        let refused = grown.as_ref().err().map(|error| error.to_string());
        if let Some(error) = refused
            && boundary.ridge.accepted
            && boundary.ridge.log_lambda != atom.ridge.log_lambda
        {
            // The exact profile is a one-dimensional reading of a joint
            // surface: its prior is right along the direction and can be too
            // weak for the joint solve to certify a mode (measured: a
            // four-mark candidate stalled at a saddle with its rate at the
            // fast wall). The boundary model's prior is the more conservative
            // of the two empirical-Bayes priors; a candidate that reaches no
            // certified optimum under the first is fitted once under it.
            log::info!(
                "[event-history] rank {rank} → {}: no certified optimum under the exact profile's prior ({error}); refitting under the boundary model's prior log-precision {:.3} from its mode scale {:.4}",
                rank + 1,
                boundary.ridge.log_lambda,
                boundary.ridge.mode_scale
            );
            step.ridge_log_lambda = boundary.ridge.log_lambda;
            step.evidence_gain = boundary.ridge.gain;
            atom.ridge = boundary.ridge.clone();
            atom.loading = boundary.loading.clone();
            let fallback = RankStart {
                atom: Some(boundary),
                ..start
            };
            grown = fit_at_rank(cohort, spec, rank + 1, Some(&fallback), pin, held.clone());
        }
        match grown {
            Ok(candidate) => {
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
                step.log_likelihood_gain = candidate.fit.log_likelihood - fit.fit.log_likelihood;
                atom_evidence.push(step.evidence_gain);
                rank_path.push(step);
                fit = candidate;
            }
            Err(error) => {
                // No certified optimum at the next rank: the path stops with
                // the reason recorded rather than failing the whole fit.
                log::info!(
                    "[event-history] rank {rank} → {}: the evidence accepted the atom but its model reached no certified optimum, refused ({error})",
                    rank + 1
                );
                step.accepted = false;
                step.converged = false;
                rank_path.push(step);
                break;
            }
        }
    }
    {
        // The whole path ran pinned, so the winner has not been certified:
        // it runs the ladder once, from where it already is, and the
        // certificate the caller reads belongs to the model the caller gets.
        let rank = fit.rank();
        let start = RankStart::carried(
            fit.fit.block_states[..marks]
                .iter()
                .map(|s| s.beta.clone())
                .collect(),
            fit.loadings.iter().copied().collect(),
            fit.log_rates.clone(),
            fit.atom_log_lambdas.clone(),
            fit.rate_held.clone(),
        );
        fit = fit_at_rank(cohort, spec, rank, Some(&start), None, held.clone())?;
    }
    // Re-centre on the risk sets. The rank grew under the stationary prior's
    // centring, where `exp(η⁰)` is the intensity averaged over everybody the
    // cohort started with; the incidence rate the baselines are meant to be is
    // the rate among those still at risk, and that normaliser is a function of
    // the coefficients through the reference population's own evolution
    // (`super::preserve`). It is held as data over each solve — its own score
    // has expectation zero, because it is predictable and the compensator
    // identity makes `E[Σ_events ∂log M] = E[∫ R λ ∂log M]` — and refreshed
    // between them, so the fit is the fixed point of that alternation. At rank
    // zero there are no loadings, `log M ≡ 0`, and nothing moves.
    let mut normaliser_rounds: Vec<f64> = Vec::new();
    let mut reference_risk_mass: Vec<f64> = Vec::new();
    let mut reference_normaliser: Vec<f64> = Vec::new();
    let mut reference_masks = 0usize;
    if spec.reference.is_some() && fit.rank() > 0 {
        let mut held: Option<Arc<Vec<f64>>> = None;
        let mut previous_point: Option<Vec<f64>> = None;
        let mut previous_residual: Option<Vec<f64>> = None;
        for _ in 0..spec.normaliser_rounds {
            let centring = fit
                .family
                .refresh_normaliser(&fit.fit.block_states)
                .map_err(|reason| EventHistoryError::Fit { reason })?;
            let next = centring.log_normaliser;
            // The residual of the alternation at the point the current fit was
            // made under: how far one refresh moves the normaliser.
            let residual: Option<Vec<f64>> = held.as_ref().map(|current| {
                current
                    .iter()
                    .zip(next.iter())
                    .map(|(u, f)| f - u)
                    .collect()
            });
            let moved = residual.as_ref().map_or(f64::INFINITY, |r| {
                r.iter().map(|v| v.abs()).fold(0.0, f64::max)
            });
            normaliser_rounds.push(moved);
            reference_risk_mass = centring.log_risk_mass;
            reference_masks = centring.masks;
            reference_normaliser = next.clone();
            if moved <= spec.normaliser_tolerance {
                log::info!(
                    "[event-history] risk-set centring round {}: the normaliser moved {moved:.3e} nats: settled",
                    normaliser_rounds.len()
                );
                break;
            }
            // The alternation contracts linearly and slowly: a shift in the
            // normaliser is absorbed by the baseline only as far as the
            // baseline's own basis can follow its shape, and what is left
            // returns through the population's killing at a rate the data set.
            // Walking that series costs a solve per term.
            //
            // The step taken instead is the secant one — Anderson acceleration
            // at depth one — on the residual map `r(u) = F(u) − u`. Two points
            // and their residuals give the linear model of `r` along the
            // direction between them, and the step is to that model's root:
            //
            //   γ = ⟨r_k, Δr⟩ / ‖Δr‖²,   Δr = r_k − r_{k−1},  Δu = u_k − u_{k−1}
            //   u_{k+1} = u_k + r_k − γ (Δu + Δr).
            //
            // It reads the whole residual vector rather than the ratio of two
            // norms, which is what makes it hold when the contraction is not a
            // single clean rate. A ratio of norms was tried first and did not:
            // on a two-mark frailty cohort its fourfold step at an apparent
            // rate of 0.8 landed further from the fixed point than the plain
            // alternation, and the rounds after it oscillated. `γ` is bounded
            // so that a near-parallel pair of residuals — where the secant
            // model is ill-conditioned and says so through a tiny ‖Δr‖ —
            // cannot turn into an unbounded step.
            let advance: Vec<f64> = match (
                held.as_ref(),
                residual.as_ref(),
                &previous_point,
                &previous_residual,
            ) {
                (Some(current), Some(r), Some(before_u), Some(before_r)) => {
                    let delta_r: Vec<f64> =
                        r.iter().zip(before_r.iter()).map(|(a, b)| a - b).collect();
                    let denominator: f64 = delta_r.iter().map(|v| v * v).sum();
                    let numerator: f64 = r.iter().zip(delta_r.iter()).map(|(a, b)| a * b).sum();
                    if denominator > 0.0 && numerator.is_finite() {
                        let gamma = (numerator / denominator).clamp(-2.0, 2.0);
                        log::info!(
                            "[event-history] risk-set centring round {}: the normaliser moved {moved:.3e} nats; secant step at γ = {gamma:.3}",
                            normaliser_rounds.len()
                        );
                        current
                            .iter()
                            .zip(r.iter())
                            .zip(before_u.iter())
                            .zip(delta_r.iter())
                            .map(|(((u, r), pu), dr)| u + r - gamma * ((u - pu) + dr))
                            .collect()
                    } else {
                        log::info!(
                            "[event-history] risk-set centring round {}: the normaliser moved {moved:.3e} nats; the residuals give no secant, stepping plain",
                            normaliser_rounds.len()
                        );
                        next
                    }
                }
                _ => {
                    log::info!(
                        "[event-history] risk-set centring round {}: the normaliser moved {moved:.3e} nats",
                        normaliser_rounds.len()
                    );
                    next
                }
            };
            if let (Some(current), Some(r)) = (held.as_ref(), residual) {
                previous_point = Some(current.as_ref().clone());
                previous_residual = Some(r);
            }
            held = Some(Arc::new(advance));
            let rank = fit.rank();
            let start = RankStart::carried(
                fit.fit.block_states[..marks]
                    .iter()
                    .map(|s| s.beta.clone())
                    .collect(),
                fit.loadings.iter().copied().collect(),
                fit.log_rates.clone(),
                fit.atom_log_lambdas.clone(),
                fit.rate_held.clone(),
            );
            // The rounds run pinned, as the rank search does: the ladder that
            // certifies a fit against a finer quadrature and mesh is worth
            // paying once, for the model the caller gets, not once per round
            // of an alternation whose intermediate points are discarded.
            fit = fit_at_rank(cohort, spec, rank, Some(&start), pin, held.clone())?;
        }
        // The settled normaliser's own fit is the one certified.
        if !normaliser_rounds.is_empty() {
            let rank = fit.rank();
            let start = RankStart::carried(
                fit.fit.block_states[..marks]
                    .iter()
                    .map(|s| s.beta.clone())
                    .collect(),
                fit.loadings.iter().copied().collect(),
                fit.log_rates.clone(),
                fit.atom_log_lambdas.clone(),
                fit.rate_held.clone(),
            );
            fit = fit_at_rank(cohort, spec, rank, Some(&start), None, held.clone())?;
        }
    }
    // What the reference grid's own resolution cost. The fit refines its own
    // mesh until the coefficients are stationary; the reference population is
    // run on a separate grid whose refinement is fixed, so that a normaliser
    // held across the ladder means the same thing on every rung. That fixed
    // choice is certified rather than assumed: the same population is run once
    // on a grid with every cell halved, both are carried onto the cohort's own
    // node times, and the largest disagreement is reported in the nats it is
    // measured in. It is not a tolerance the fit enforces — a normaliser is an
    // offset, so a shift in it is absorbed by the baseline it centres — but it
    // is the number that says whether the grid resolved the population's decay.
    let mut reference_certificate = None;
    // Only when a normaliser was actually taken: at rank zero there are no
    // loadings, `log M ≡ 0`, and there is no grid resolution to certify.
    if let (Some(strata), Some(tables), false) = (
        spec.reference.as_ref(),
        fit.family.reference(),
        reference_normaliser.is_empty(),
    ) {
        let refined = reference_tables(
            cohort,
            strata,
            &fit.frozen_specs,
            spec.quadrature_order,
            REFERENCE_REFINEMENT + 1,
            &fit.nodes,
        )?;
        let coarse_on_nodes = fit
            .family
            .normaliser_on_nodes(&reference_normaliser)
            .map_err(|error| EventHistoryError::Fit {
                reason: error.to_string(),
            })?;
        let refined_family = fit
            .family
            .clone()
            .with_reference(Some(Arc::new(refined)), None)?;
        let refined_values = refined_family
            .refresh_normaliser(&fit.fit.block_states)
            .map_err(|reason| EventHistoryError::Fit { reason })?;
        let refined_on_nodes = refined_family
            .normaliser_on_nodes(&refined_values.log_normaliser)
            .map_err(|error| EventHistoryError::Fit {
                reason: error.to_string(),
            })?;
        let gap = coarse_on_nodes
            .iter()
            .zip(refined_on_nodes.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        log::info!(
            "[event-history] the reference grid at refinement {REFERENCE_REFINEMENT} differs from the halved one by {gap:.3e} nats (a {} node grid)",
            tables.grid.len()
        );
        reference_certificate = Some(gap);
    }
    fit.rank_path = rank_path;
    fit.atom_evidence = atom_evidence;
    fit.normaliser_rounds = normaliser_rounds;
    fit.reference_risk_mass = reference_risk_mass;
    fit.reference_masks = reference_masks;
    fit.reference_certificate = reference_certificate;
    fit.reference_normaliser = reference_normaliser;
    Ok(fit)
}

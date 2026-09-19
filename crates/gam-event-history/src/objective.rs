//! Total derivatives of the reference-normalised likelihood.
use super::*;
use crate::scalar::{Rows, TANGENT_WIDTH};
use gam_model_api::families::custom_family::ExactNewtonJointHessianWorkspace;
use gam_problem::EvalMode;

impl EventHistoryFamily {
    fn reference_values<S: JetField>(
        &self, beta: &[S], loadings: &[S], rates: &[S],
    ) -> Result<crate::preserve::Normalisers<S>, EventHistoryError> {
        let tables = self.reference.as_ref().ok_or_else(|| EventHistoryError::InvalidInput {
            reason: "this family has no reference population".to_string(),
        })?;
        let marks = self.marks();
        let nodes = tables.grid.len();
        let offsets = self.block_offsets();
        let mut normalisers = Vec::new();
        let mut risk_mass = Vec::new();
        let mut masks = 0;
        for s in 0..tables.strata {
            let mut eta0 = Vec::with_capacity(nodes * marks);
            for n in 0..nodes {
                for d in 0..marks {
                    let row = s * nodes + n;
                    let mut value = beta[0].constant_like(tables.offsets[d][row]);
                    for (j, x) in tables.designs[d].row(row).iter().enumerate() {
                        value = value.add(&beta[offsets[d] + j].scale(*x));
                    }
                    eta0.push(value);
                }
            }
            let out = match stratum_normalisers(&tables.grid, &eta0, loadings, rates,
                self.time_scale, &self.gh, &tables.kinds, self.atoms) {
                Ok(out) => out,
                // Kept typed on the family, because the engine sees this
                // refusal only as text.
                Err(refusal @ EventHistoryError::ReferenceStep { .. }) => {
                    if let Ok(mut slot) = self.reference_refusal.lock() {
                        *slot = Some(refusal.clone());
                    }
                    return Err(refusal);
                }
                Err(error) => return Err(error),
            };
            masks = out.masks;
            normalisers.extend(out.log_normaliser);
            risk_mass.extend(out.log_risk_mass);
        }
        Ok(crate::preserve::Normalisers {
            log_normaliser: normalisers, log_risk_mass: risk_mass, masks,
        })
    }

    pub(super) fn computed_reference(&self, states: &[ParameterBlockState]) -> Result<RiskSetCentring, EventHistoryError> {
        self.validate_states(states).map_err(|reason| EventHistoryError::InvalidInput { reason })?;
        let beta: Vec<f64> = states.iter().flat_map(|s| s.beta.iter().copied()).collect();
        let latent_offset = self.block_offsets()[self.marks()];
        let latent = Array1::from(beta[latent_offset..].to_vec());
        let rates = self.atom_rates(&latent);
        let out = self.reference_values(&beta,
            &beta[latent_offset..latent_offset + self.marks() * self.atoms], &rates)?;
        let tables = self.reference.as_ref().ok_or_else(|| EventHistoryError::InvalidInput {
            reason: "reference centring requires reference tables".to_string(),
        })?;
        let (_, mask_of_mark) = crate::preserve::killing_masks(&tables.kinds);
        Ok(RiskSetCentring { grid: tables.grid.clone(), profiles: tables.profiles.clone(),
            coefficients: beta, node_stratum: tables.node_stratum.clone(),
            log_normaliser: out.log_normaliser,
            log_risk_mass: out.log_risk_mass, masks: out.masks, mask_of_mark })
    }

    /// The reference law is evaluated with the same jet as the subject
    /// likelihood, before interpolation. Grid adaptation is differentiated too.
    pub(super) fn path_value<S: JetField + Send + Sync>(
        &self, states: &[ParameterBlockState], beta: &[S],
    ) -> Result<S, EventHistoryError> {
        let marks = self.marks();
        let offsets = self.block_offsets();
        let latent_offset = offsets[marks];
        let loadings = &beta[latent_offset..latent_offset + marks * self.atoms];
        let rates: Vec<S> = self.free_rate_slots().iter().enumerate().map(|(k, slot)| {
            match slot {
                Some(slot) => rate_from_chart(self.rate_band, &beta[latent_offset + slot]),
                None => beta[0].constant_like(self.held_rates[k]
                    .expect("a non-free atom rate has a held value")),
            }
        }).collect();
        let normalisers = if let (Some(tables), true) = (self.reference.as_ref(), self.atoms > 0) {
            let values = self.reference_values(beta, loadings, &rates)?;
            Some(tables.carry_to_nodes(&values.log_normaliser, marks, self.nodes.total_nodes)?)
        } else { None };
        let results: Result<Vec<S>, EventHistoryError> = self.nodes.subjects.par_iter().map(|subject| {
            let first = subject.first_row;
            let mut eta0 = Vec::with_capacity(subject.len() * marks);
            for row in first..first + subject.len() {
                for d in 0..marks {
                    let mut eta = beta[0].constant_like(0.0);
                    for (j, x) in self.designs[d].row(row).iter().enumerate() {
                        eta = eta.add(&beta[offsets[d] + j].scale(*x));
                    }
                    eta0.push(eta.with_value(states[d].eta[row]));
                }
            }
            let inputs = SubjectInputs {
                nodes: subject, eta0: &eta0, loadings, rates: &rates,
                time_scale: self.time_scale, gh: &self.gh, continuation_gap: 0.0,
                designs: None,
                log_normaliser: normalisers.as_ref().map(|values|
                    &values[first * marks..(first + subject.len()) * marks]),
            };
            subject_marginal(&inputs, false).map(|result| result.loglik)
        }).collect();
        Ok(pairwise_sum(&results?, &beta[0].constant_like(0.0)))
    }

    /// The value, the gradient along `coordinates` and the row-major Hessian
    /// over them of the computed log-likelihood, every entry carrying the
    /// directions `beta` is seeded with.
    ///
    /// One path evaluation over `Rows<Rows<S, TANGENT_WIDTH>, TANGENT_WIDTH>`
    /// returns the block of second derivatives between one block of
    /// `TANGENT_WIDTH` coordinates and another, with the gradient along the
    /// second. The lower triangle of blocks costs `b(b + 1)/2` path evaluations
    /// over `b = ⌈m / TANGENT_WIDTH⌉` blocks of `m` coordinates, where seeding
    /// one coordinate pair per evaluation cost `m(m + 1)/2` (#2965). Each entry
    /// is still the exact derivative of the same computed value, with reference
    /// evolution and grid placement included.
    pub(super) fn coordinate_hessian<S: JetField + Send + Sync>(
        &self, states: &[ParameterBlockState], beta: &[S], coordinates: &[usize],
    ) -> Result<(S, Vec<S>, Vec<S>), EventHistoryError> {
        let width = coordinates.len();
        if width == 0 {
            return Ok((self.path_value(states, beta)?, Vec::new(), Vec::new()));
        }
        let zero = beta[0].constant_like(0.0);
        let mut value = zero.clone();
        let mut gradient = vec![zero.clone(); width];
        let mut hessian = vec![zero; width * width];
        let tangents = |q: usize, start: usize| -> [f64; TANGENT_WIDTH] {
            std::array::from_fn(|k| f64::from(coordinates.get(start + k) == Some(&q)))
        };
        for a in 0..width.div_ceil(TANGENT_WIDTH) {
            let rows = a * TANGENT_WIDTH;
            for b in 0..=a {
                let columns = b * TANGENT_WIDTH;
                let seeded: Vec<Rows<Rows<S, TANGENT_WIDTH>, TANGENT_WIDTH>> = beta.iter().enumerate()
                    .map(|(q, coefficient)| Rows::seed(
                        Rows::seed(coefficient.clone(), tangents(q, columns)), tangents(q, rows)))
                    .collect();
                let result = self.path_value(states, &seeded)?;
                for l in 0..TANGENT_WIDTH.min(width - columns) {
                    gradient[columns + l] = result.base.rows[l].clone();
                }
                for k in 0..TANGENT_WIDTH.min(width - rows) {
                    let i = rows + k;
                    // Within a diagonal block only `j ≤ i` is read, so each
                    // mirrored pair comes from one channel.
                    for l in 0..TANGENT_WIDTH.min(width - columns).min(i + 1 - columns) {
                        let j = columns + l;
                        hessian[i * width + j] = result.rows[k].rows[l].clone();
                        hessian[j * width + i] = result.rows[k].rows[l].clone();
                    }
                }
                value = result.base.base;
            }
        }
        Ok((value, gradient, hessian))
    }

    /// The value, the gradient and `H v` of the computed log-likelihood along
    /// one direction `v`. One path evaluation over
    /// `Rows<Rows<f64, 1>, TANGENT_WIDTH>` per block of coefficients: the inner
    /// level carries `v`, the outer level the block's coordinates, and the
    /// mixed channel is `∂(∇ℓ · v)/∂θ_q`. A product costs `⌈p / TANGENT_WIDTH⌉`
    /// evaluations, where a dense Hessian costs `b(b + 1)/2` over the same
    /// `b = ⌈p / TANGENT_WIDTH⌉` blocks (#2965).
    pub(super) fn hessian_vector_product(
        &self, states: &[ParameterBlockState], v: &[f64],
    ) -> Result<(f64, Vec<f64>, Vec<f64>), EventHistoryError> {
        let values: Vec<f64> = states.iter().flat_map(|s| s.beta.iter().copied()).collect();
        let total = values.len();
        if v.len() != total || v.iter().any(|x| !x.is_finite()) {
            return Err(EventHistoryError::InvalidInput {
                reason: "invalid event-history Hessian-vector direction".to_string(),
            });
        }
        let mut value = 0.0;
        let mut gradient = vec![0.0; total];
        let mut product = vec![0.0; total];
        for start in (0..total).step_by(TANGENT_WIDTH) {
            let seeded: Vec<Rows<Rows<f64, 1>, TANGENT_WIDTH>> = values.iter().zip(v).enumerate()
                .map(|(q, (coefficient, along))| Rows::seed(
                    Rows::seed(*coefficient, [*along]),
                    std::array::from_fn(|k| f64::from(q == start + k))))
                .collect();
            let result = self.path_value(states, &seeded)?;
            for k in 0..TANGENT_WIDTH.min(total - start) {
                gradient[start + k] = result.rows[k].base;
                product[start + k] = result.rows[k].rows[0];
            }
            value = result.base.base;
        }
        Ok((value, gradient, product))
    }

    pub(super) fn computed_joint<S: Directional>(
        &self, states: &[ParameterBlockState], u: Option<&Array1<f64>>,
        v: Option<&Array1<f64>>, derivatives: bool,
    ) -> Result<(S, Vec<S>, Vec<S>), String> {
        let values: Vec<f64> = states.iter().flat_map(|s| s.beta.iter().copied()).collect();
        let total = values.len();
        for direction in [u, v].into_iter().flatten() {
            if direction.len() != total || direction.iter().any(|x| !x.is_finite()) {
                return Err("invalid event-history derivative direction".to_string());
            }
        }
        let beta: Vec<S> = values.iter().enumerate().map(|(q, value)|
            S::seeded(*value, u.map_or(0.0, |x| x[q]), v.map_or(0.0, |x| x[q]))).collect();
        if !derivatives {
            return Ok((self.path_value(states, &beta)?, Vec::new(), Vec::new()));
        }
        let coordinates: Vec<usize> = (0..total).collect();
        Ok(self.coordinate_hessian(states, &beta, &coordinates)?)
    }

    /// Whether the coefficient derivatives come from the computed path: a
    /// reference law differentiated through its evolution, or a static atom
    /// beside a dynamic one. The Louis sweep of `subject_marginal` covers
    /// all-static atoms, whose nodes share one whole-history grid, and neither
    /// of those.
    pub(super) fn differentiates_the_computed_path(&self) -> bool {
        let any_static = self.held_rates.contains(&Some(0.0));
        let all_static = self.held_rates.iter().all(|rate| *rate == Some(0.0));
        self.atoms > 0 && (self.reference.is_some() || (any_static && !all_static))
    }
}

/// The coefficient-Hessian workspace of a family whose derivatives come from
/// the computed path, at one state (#2965).
///
/// Every Hessian source query, the inner solve's Newton direction included,
/// returns the block sweep's dense Hessian, which the family caches on the
/// state: the trait's dense preference, kept because streaming the Newton
/// direction through `H v` took more inner cycles and more time where it was
/// measured, on static atoms at p = 17 before they took the Louis sweep
/// (#2965). A consumer that does not query the preference still streams `H v`
/// from [`EventHistoryFamily::hessian_vector_product`], `⌈p / TANGENT_WIDTH⌉`
/// path evaluations a product. The outer log-determinant's Jeffreys pre-check
/// is one such consumer. Every representation is the negative log-likelihood
/// Hessian of the same computed value, the engine's convention.
pub(super) struct ComputedHessianWorkspace {
    family: EventHistoryFamily,
    states: Vec<ParameterBlockState>,
    gradient: std::sync::OnceLock<Result<(f64, Array1<f64>), String>>,
}

impl ComputedHessianWorkspace {
    pub(super) fn new(family: EventHistoryFamily, states: Vec<ParameterBlockState>) -> Self {
        Self { family, states, gradient: std::sync::OnceLock::new() }
    }

    /// The value and exact gradient at the workspace's state, formed once.
    fn value_and_gradient(&self) -> Result<(f64, Array1<f64>), String> {
        self.gradient.get_or_init(|| {
            let value = self.family.log_likelihood(&self.states)?;
            Ok((value, Array1::from(self.family.exact_gradient(&self.states)?)))
        }).clone()
    }
}

impl ExactNewtonJointHessianWorkspace for ComputedHessianWorkspace {
    fn warm_up_outer_caches_for_mode(&self, eval_mode: EvalMode) -> Result<(), String> {
        match eval_mode {
            EvalMode::ValueOnly | EvalMode::ValueAndGradient | EvalMode::ValueGradientHessian => Ok(()),
        }
    }

    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(self.family.joint_evaluation(&self.states)?.hessian.clone()))
    }

    fn joint_log_likelihood_evaluation(&self) -> Result<Option<f64>, String> {
        Ok(Some(self.value_and_gradient()?.0))
    }

    fn joint_gradient_evaluation(&self) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        let (log_likelihood, gradient) = self.value_and_gradient()?;
        Ok(Some(ExactNewtonJointGradientEvaluation { log_likelihood, gradient }))
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, arr: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let (_, _, product) = self.family.hessian_vector_product(&self.states, &arr.to_vec())?;
        Ok(Some(Array1::from_iter(product.into_iter().map(|x| -x))))
    }

    fn directional_derivative(&self, d_beta_flat: &Array1<f64>) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(self.family.directional_hessian(&self.states, d_beta_flat)?))
    }

    fn second_directional_derivative(
        &self, arr: &Array1<f64>, arr2: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(self.family.second_directional_hessian(&self.states, arr, arr2)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cohort::{CovariateSegment, Event, SubjectHistory};
    use ndarray::array;

    fn single_event() -> (EventHistoryFamily, Vec<ParameterBlockState>) {
        single_event_on(144, 15, 1e-8, [-1.2, 0.9])
    }

    /// One subject with one event at six time units, on a reference grid of
    /// `intervals` equal steps, at Gauss-Hermite `order`, a held atom `rate`,
    /// and the log baseline and loading in `coefficients`.
    fn single_event_on(
        intervals: usize, order: usize, rate: f64, coefficients: [f64; 2],
    ) -> (EventHistoryFamily, Vec<ParameterBlockState>) {
        let mut cohort = EventHistoryCohort {
            mark_names: vec!["disease".to_string()], mark_kinds: vec![MarkKind::Once],
            covariate_names: Vec::new(), covariate_levels: Vec::new(),
            covariates: Array2::zeros((1, 0)),
            subjects: vec![SubjectHistory {
                id: "one".to_string(), entry: 0.0, exit: 6.0,
                events: vec![Event { time: 6.0, mark: 0 }],
                segments: vec![CovariateSegment { start: 0.0, row: 0 }],
            }],
        };
        cohort.validate().unwrap();
        let nodes = Arc::new(expand_nodes(&cohort, 9, 3).unwrap());
        let times: Vec<f64> = (0..=intervals).map(|n| 6.0 * n as f64 / intervals as f64).collect();
        let grid = ReferenceGrid { gaps: times.windows(2).map(|w| w[1] - w[0]).collect(), times };
        let locations: Vec<(usize, f64)> = nodes.subjects[0].times.iter().map(|&t| grid.locate(t).unwrap()).collect();
        let tables = ReferenceTables {
            designs: vec![Arc::new(Array2::ones((grid.len(), 1)))],
            offsets: vec![Array1::zeros(grid.len())], kinds: vec![MarkKind::Once],
            profiles: Array2::zeros((1, 0)), strata: 1,
            node_stratum: vec![0; nodes.total_nodes],
            node_lower: locations.iter().map(|x| x.0).collect(),
            node_weight: locations.iter().map(|x| x.1).collect(), grid,
        };
        let [baseline, loading] = coefficients;
        let states = vec![
            ParameterBlockState { beta: array![baseline], eta: Array1::from_elem(nodes.total_nodes, baseline) },
            ParameterBlockState { beta: array![loading], eta: Array1::zeros(nodes.total_nodes) },
        ];
        let family = EventHistoryFamily::new(nodes.clone(), vec![Arc::new(Array2::ones((nodes.total_nodes, 1)))],
            1, order, 1.0, vec![Some(rate)]).unwrap().with_reference(Some(Arc::new(tables)));
        (family, states)
    }

    fn moved(states: &[ParameterBlockState], slot: usize, change: f64) -> Vec<ParameterBlockState> {
        let mut out = states.to_vec();
        out[slot].beta[0] += change;
        if slot == 0 { out[0].eta += change; }
        out
    }

    #[test]
    fn normalised_objective_differentiates_the_reference_law() {
        let (family, states) = single_event();
        let joint = family.joint_evaluation(&states).unwrap();
        let exact = -1.2 - 6.0 * (-1.2_f64).exp();
        assert!((joint.log_likelihood - exact).abs() < 2e-4, "{} vs {exact}", joint.log_likelihood);
        assert!(joint.gradient[1].abs() < 5e-4, "unidentified frailty score: {}", joint.gradient[1]);
        for slot in 0..2 {
            let h = 1e-4;
            let plus = moved(&states, slot, h);
            let minus = moved(&states, slot, -h);
            let fd = (family.log_likelihood(&plus).unwrap() - family.log_likelihood(&minus).unwrap()) / (2.0 * h);
            assert!((fd - joint.gradient[slot]).abs() < 1e-6, "gradient {slot}: {fd} vs {}", joint.gradient[slot]);
            let gp = family.exact_gradient(&plus).unwrap();
            let gm = family.exact_gradient(&minus).unwrap();
            for j in 0..2 {
                let fd = -(gp[j] - gm[j]) / (2.0 * h);
                assert!((fd - joint.hessian[[j, slot]]).abs() < 2e-5,
                    "Hessian {j},{slot}: {fd} vs {}", joint.hessian[[j, slot]]);
            }
        }
        let reference = family.refresh_normaliser(&states).unwrap();
        let encoded = serde_json::to_string(&reference).unwrap();
        let restored: RiskSetCentring = serde_json::from_str(&encoded).unwrap();
        assert_eq!(reference.coefficients, restored.coefficients);
        assert_eq!(reference.log_normaliser, restored.log_normaliser);
        assert_eq!(reference.log_risk_mass, restored.log_risk_mass);
        assert_eq!(reference.grid.times, restored.grid.times);
        assert_eq!(reference.profiles, restored.profiles);
        assert_eq!(reference.node_stratum, restored.node_stratum);
        assert_eq!(reference.mask_of_mark, restored.mask_of_mark);
        assert!(reference.log_normaliser.last().unwrap() < &reference.log_normaliser[0]);
        assert!((reference.log_risk_mass.last().unwrap() + 6.0 * (-1.2_f64).exp()).abs() < 2e-4);
    }

    /// The reference refusal the engine's text cannot carry is read back typed,
    /// and never outlives the evaluation that raised it (#2627): evaluation k
    /// refuses, evaluation k + 1 succeeds, and an unrelated engine failure after
    /// them surfaces as a fit failure. At step 3 and log baseline 0 a loading of
    /// 2 does not contract (ratio 1.302 in job 1186034), a loading of 1 does.
    #[test]
    fn reference_refusal_is_typed_and_never_outlives_its_evaluation_2627() {
        let (family, refusing) = single_event_on(2, 9, 1e-6, [0.0, 2.0]);
        let contracting = moved(&refusing, 1, -1.0);
        let refused = family.evaluate(&refusing).err().expect("evaluation k refuses");
        assert!(refused.contains("does not contract"), "evaluation k must refuse the reference step: {refused}");
        family.evaluate(&contracting).expect("evaluation k + 1 contracts");
        match typed_failure(&family, "an unrelated engine failure".to_string()) {
            EventHistoryError::Fit { reason } => assert_eq!(reason, "an unrelated engine failure"),
            other => panic!("a refusal outlived its evaluation: {other}"),
        }
        // Positive control: straight after a refusal the failure is read back
        // typed, and only once.
        assert!(family.evaluate(&refusing).is_err());
        assert!(matches!(typed_failure(&family, "engine".to_string()), EventHistoryError::ReferenceStep { .. }));
        assert!(matches!(typed_failure(&family, "engine".to_string()), EventHistoryError::Fit { .. }));
    }

    #[test]
    fn normalised_hessian_directional_derivatives_follow_the_same_objective() {
        let (family, states) = single_event();
        let direction = array![0.0, 1.0];
        let first = family.directional_hessian(&states, &direction).unwrap();
        let second = family.second_directional_hessian(&states, &direction, &direction).unwrap();
        let h = 1e-3;
        let plus = moved(&states, 1, h);
        let minus = moved(&states, 1, -h);
        let hp = family.joint_evaluation(&plus).unwrap();
        let hm = family.joint_evaluation(&minus).unwrap();
        let dp = family.directional_hessian(&plus, &direction).unwrap();
        let dm = family.directional_hessian(&minus, &direction).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((first[[i, j]] - (hp.hessian[[i, j]] - hm.hessian[[i, j]]) / (2.0 * h)).abs() < 2e-4);
                assert!((second[[i, j]] - (dp[[i, j]] - dm[[i, j]]) / (2.0 * h)).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn reference_refinement_detects_latent_error_at_fixed_parameters() {
        let (mut family, mut states) = single_event();
        family.held_rates = vec![Some(0.0)];
        family.gh = Arc::new(GaussHermite::new(9).unwrap());
        states[1].beta[0] = 2.0;
        let coarse = family.refresh_normaliser(&states).unwrap();
        family.gh = Arc::new(GaussHermite::new(33).unwrap());
        let fine = family.refresh_normaliser(&states).unwrap();
        let gap = coarse.discrepancy(&fine, 1).unwrap();
        assert!(gap > 1e-4, "unresolved latent reference integral: {gap}");
        assert_eq!(fine.discrepancy(&fine, 1).unwrap(), 0.0);
        let mut different = fine.clone();
        different.coefficients[0] += 0.1;
        assert!(coarse.discrepancy(&different, 1).is_err());
        different = fine.clone();
        different.log_normaliser[0] = f64::NAN;
        assert!(coarse.discrepancy(&different, 1).is_err());
    }

    fn recurrent_family(event_time: f64, rates: Vec<Option<f64>>, order: usize) -> (EventHistoryFamily, Vec<ParameterBlockState>) {
        recurrent_family_on(event_time, rates, order, 9)
    }

    /// One subject on `[0, 1]` with one recurrent event at `event_time`, each
    /// of its two cells integrated by `legendre` Gauss-Legendre points.
    fn recurrent_family_on(
        event_time: f64, rates: Vec<Option<f64>>, order: usize, legendre: usize,
    ) -> (EventHistoryFamily, Vec<ParameterBlockState>) {
        let mut cohort = EventHistoryCohort {
            mark_names: vec!["event".to_string()], mark_kinds: vec![MarkKind::Recurrent],
            covariate_names: Vec::new(), covariate_levels: Vec::new(), covariates: Array2::zeros((1, 0)),
            subjects: vec![SubjectHistory { id: "one".to_string(), entry: 0.0, exit: 1.0,
                events: vec![Event { time: event_time, mark: 0 }],
                segments: vec![CovariateSegment { start: 0.0, row: 0 }] }],
        };
        cohort.validate().unwrap();
        let nodes = Arc::new(expand_nodes(&cohort, legendre, 0).unwrap());
        let states = vec![
            ParameterBlockState { beta: array![0.0], eta: Array1::zeros(nodes.total_nodes) },
            ParameterBlockState { beta: array![2.0, 0.0], eta: Array1::zeros(nodes.total_nodes) },
        ];
        let family = EventHistoryFamily::new(nodes.clone(), vec![Arc::new(Array2::ones((nodes.total_nodes, 1)))],
            2, order, 1.0, rates).unwrap();
        (family, states)
    }

    #[test]
    fn static_factor_curvature_matches_the_time_invariant_integral() {
        let points = 4001;
        let mut mass = 0.0;
        let mut second = 0.0;
        for i in 0..points {
            let z = -10.0 + 20.0 * i as f64 / (points - 1) as f64;
            let r = (2.0 * z - 2.0).exp();
            let w = (-0.5 * z * z).exp() * r * (-r).exp();
            mass += w;
            second += w * (r * r - 2.0 * r);
        }
        let expected = second / mass;
        assert!(expected < -0.3);
        let mut curvatures = Vec::new();
        for event_time in [0.1, 0.5, 0.9] {
            let (family, states) = recurrent_family(event_time, vec![Some(0.0), Some(0.0)], 33);
            let beta = vec![
                Rows::seed(Rows::seed(0.0, [0.0]), [0.0]),
                Rows::seed(Rows::seed(2.0, [0.0]), [0.0]),
                Rows::seed(Rows::seed(0.0, [1.0]), [1.0]),
            ];
            let curvature = family.path_value(&states, &beta).unwrap().rows[0].rows[0];
            let h = 1e-3;
            let eval = |a: f64| family.path_value(&states, &[0.0, 2.0, a]).unwrap();
            let finite = (eval(h) + eval(-h) - 2.0 * eval(0.0)) / (h * h);
            eprintln!("event {event_time}: AD={curvature}, finite={finite}, target={expected}");
            assert!((curvature - finite).abs() < 2e-5);
            assert!((curvature - expected).abs() < 1e-4, "{curvature} vs {expected}");
            assert!(added_factor_curvature_pair(&family, &states, EventHistorySpec::new(Vec::new()).quadrature_tolerance)
                .unwrap()
                .is_some());
            curvatures.push(curvature);
        }
        assert!(curvatures.iter().all(|c| (c - curvatures[0]).abs() < 1e-12));
    }

    /// The added-factor curvature of an in-band dynamic rate is read at its
    /// order and one ladder rung up (`2·order − 1`) while that order is
    /// certifiable, up to the ladder's top certifiable rung, and is not formed
    /// above it: that is where the rank search stops with growth unresolved.
    ///
    /// This replaces `unresolved_near_static_curvature_is_rejected`, whose rates
    /// of `1e-10` sit below `ν_min`: there the OU kernel is one to double
    /// precision, so the fixture is the static model evaluated through the
    /// dynamic interpolant. Its `is_err()` passed on the positivity loss inside
    /// the curvature, not on the gap check it named. A rate below `ν_min` belongs
    /// to the static face.
    #[test]
    fn added_factor_curvature_is_checked_up_to_the_top_certifiable_rung() {
        let tolerance = EventHistorySpec::new(Vec::new()).quadrature_tolerance;
        let rates = vec![Some(0.5), Some(0.7)];
        let (base, _) = recurrent_family(0.1, rates.clone(), 9);
        let nodes = base.nodes.max_subject_nodes();
        let mut top = 9;
        while let Some(next) = positivity_raise(top, nodes, tolerance) {
            top = next;
        }
        assert!(top > 9, "a certifiable rung must remain above order 9 over {nodes} nodes");
        assert!(certifiable(top, nodes, tolerance) && !certifiable(2 * top - 1, nodes, tolerance));
        let (family, states) = recurrent_family(0.1, rates.clone(), top);
        let pair = added_factor_curvature_pair(&family, &states, tolerance)
            .unwrap()
            .expect("the top certifiable rung checks its curvature");
        assert_eq!(pair.next_order, 2 * top - 1);
        assert!(pair.coarse.iter().chain(pair.refined.iter()).all(|x| x.is_finite()));
        let (above, above_states) = recurrent_family(0.1, rates, 2 * top - 1);
        assert!(
            added_factor_curvature_pair(&above, &above_states, tolerance).unwrap().is_none(),
            "order {} is above the top certifiable rung {top}, so its curvature cannot be checked",
            2 * top - 1
        );
    }

    /// One predicate decides every rung: at 449 nodes order 11 is certifiable,
    /// order 21 is not, so the raise from 11 refuses rather than landing on a
    /// rung whose own certificate cannot be checked (job 1150580 raised to 21
    /// and then refused at order 41, Lebesgue constant 1.154e13).
    #[test]
    fn at_449_nodes_order_11_is_the_top_certifiable_rung() {
        let tolerance = EventHistorySpec::new(Vec::new()).quadrature_tolerance;
        assert!(certifiable(11, 449, tolerance));
        assert!(!certifiable(21, 449, tolerance));
        assert_eq!(positivity_raise(11, 449, tolerance), None);
    }

    /// The start shift is the one-unit bar's numerator: each term against its
    /// closed form, sign-aligned eigenvectors moving nothing, a refused
    /// proposal moving nothing, and a spread that cannot price anything
    /// refusing rather than passing.
    #[test]
    fn proposal_start_shift_prices_the_rung_in_posterior_sd() {
        let values = array![3.0, 1.0];
        let identity = array![[1.0, 0.0], [0.0, 1.0]];
        let spreads = [0.1, 0.5];
        let mode_scale = 2.0;
        let flipped = array![[-1.0, 0.0], [0.0, 1.0]];
        assert_eq!(proposal_start_shift((&values, &identity), (&values, &flipped), mode_scale, &spreads), 0.0);

        let raised = array![3.5, 1.0];
        let shift = proposal_start_shift((&values, &identity), (&raised, &identity), mode_scale, &spreads);
        assert!((shift - 0.5 * mode_scale * spreads[0]).abs() < 1e-15, "{shift}");

        let theta = 0.01_f64;
        let rotated = array![[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()]];
        let shift = proposal_start_shift((&values, &identity), (&values, &rotated), mode_scale, &spreads);
        let along_second = mode_scale * theta.sin() / spreads[1];
        let along_top = mode_scale * (1.0 - theta.cos()) / spreads[0];
        assert!(along_top < along_second);
        assert!((shift - along_second).abs() < 1e-15, "{shift} vs {along_second}");

        assert_eq!(proposal_start_shift((&values, &identity), (&raised, &rotated), 0.0, &spreads), 0.0);
        assert_eq!(
            proposal_start_shift((&values, &identity), (&values, &identity), mode_scale, &[0.1, f64::NAN]),
            f64::INFINITY
        );
        assert_eq!(
            proposal_start_shift((&values, &identity), (&values, &identity), mode_scale, &[0.1]),
            f64::INFINITY
        );
    }

    /// `mode_spread` against the Gaussian it must reproduce: `1/√(a + λ)` about
    /// a maximiser at zero, and the same about a maximiser six sd out, where
    /// the reflection's location is not spread.
    #[test]
    fn mode_spread_is_the_posterior_sd_about_the_mode() {
        let profile = |slope0: f64, curvature: f64| {
            let points: Vec<f64> = (0..=400).map(|i| 0.025 * i as f64).collect();
            DirectionProfile {
                values: points.iter().map(|t| slope0 * t - 0.5 * curvature * t * t).collect(),
                slopes: points.iter().map(|t| slope0 - curvature * t).collect(),
                points,
            }
        };
        let (curvature, lambda) = (3.0_f64, 1.0_f64);
        let sd = 1.0 / (curvature + lambda).sqrt();
        let centred = profile(0.0, curvature).mode_spread(lambda);
        assert!((centred - sd).abs() < 1e-6 * sd, "{centred} vs {sd}");
        let shifted = profile(12.0, curvature).mode_spread(lambda);
        assert!((shifted - sd).abs() < 1e-4 * sd, "{shifted} vs {sd}");
    }

    #[test]
    fn mixed_static_and_dynamic_factors_have_finite_total_derivatives() {
        let (family, mut states) = recurrent_family(0.5, vec![Some(0.0), Some(0.7)], 9);
        states[1].beta = array![0.3, 0.2];
        let joint = family.joint_evaluation(&states).unwrap();
        for slot in 0..2 {
            let h = 1e-4;
            let mut plus = states.clone();
            let mut minus = states.clone();
            plus[1].beta[slot] += h;
            minus[1].beta[slot] -= h;
            let gp = family.exact_gradient(&plus).unwrap();
            let gm = family.exact_gradient(&minus).unwrap();
            for j in 0..3 {
                let fd = -(gp[j] - gm[j]) / (2.0 * h);
                assert!((fd - joint.hessian[[j, slot + 1]]).abs() < 1e-5);
            }
        }
    }

    /// A rounding bound carries a value, never a direction, so it seeds only
    /// the undirected evaluation.
    impl Directional for crate::test_support::Bound {
        fn seeded(value: f64, u: f64, v: f64) -> Self {
            assert!(u == 0.0 && v == 0.0, "a rounding bound carries no direction");
            Self::exact(value)
        }
        fn eps(&self) -> f64 {
            0.0
        }
        fn eps_del(&self) -> f64 {
            0.0
        }
    }

    /// The Louis Hessian against the block sweep of one fixture at Gauss-Hermite
    /// orders 9, 17 and 33, entry by entry (#2965). The Louis Hessian
    /// approximates the exact marginal's curvature, and the block sweep
    /// differentiates the computed value, grid placement included. Both tend to
    /// the exact marginal's curvature as the Gauss-Hermite order resolves, so at
    /// order 17 they differ by at most the two routes' quadrature errors there
    /// plus their rounding.
    ///
    /// Both routes run at `Bound`: a computed value `v_n` at order `n` is within
    /// `r_n = ε μ_n` of its exact-arithmetic result, with test_support's `exp`
    /// and `ln` charges cited from the runtime libm. The design entries, the
    /// held rates and the rule's nodes and weights are the same data in all six
    /// evaluations, so they enter exactly: both routes place each order's grids
    /// through the one `marginal::filter_nodes` (static atoms through
    /// `static_state::filter`, whose `posterior_grid` calls `Grid::new`), so
    /// both read identical positions such as `σ · fl(√2 x)`, and those positions
    /// are the rule both evaluate. A static placement's mode is searched on
    /// values and enters as a rounded constant, so only the Newton steps that
    /// carry its channels are charged. Per route and entry the exact
    /// refinement steps satisfy `d1 ≥ |v9 − v17| − r9 − r17` and
    /// `d2 ≤ |v17 − v33| + r17 + r33`. With `d1` resolved above zero, and under
    /// geometric convergence past order 33 at a ratio below
    /// `q = max d2 / min d1 < 1`, the order-17 error is at most
    /// `max d2 / (1 − q)`. The bar is the two routes' errors plus each route's
    /// own `r17`. The order-9 comparison is printed with its bar, which adds each
    /// route's first step, but not asserted: the first steps are the order-9
    /// gap itself. An unresolved or growing step gives no estimate, and every
    /// compared entry must be material against its bar. Every order's signed
    /// values are printed before anything is asserted.
    fn louis_matches_the_block_sweep_2965(
        label: &str, fixture: impl Fn(usize) -> (EventHistoryFamily, Vec<ParameterBlockState>),
    ) {
        use crate::test_support::Bound;
        struct Refinement {
            first: f64,
            second: f64,
            resolved: f64,
            ratio: f64,
            error: f64,
        }
        let refinement = |values: &[Vec<Bound>], k: usize| -> Refinement {
            let (v9, v17, v33) = (&values[0][k], &values[1][k], &values[2][k]);
            let first = (v9.value - v17.value).abs();
            let second = (v17.value - v33.value).abs();
            let resolved = first - v9.rounding() - v17.rounding();
            let second_upper = second + v17.rounding() + v33.rounding();
            let ratio = second_upper / resolved;
            Refinement { first, second, resolved, ratio, error: second_upper / (1.0 - ratio) }
        };
        let orders = [9, 17, 33];
        let mut louis = Vec::new();
        let mut computed = Vec::new();
        for order in orders {
            let (family, states) = fixture(order);
            assert!(!family.differentiates_the_computed_path(), "{label}: the fixture takes the Louis sweep");
            let (_, _, sweep) = family.evaluate_generic::<Bound>(&states, None, None, true).unwrap();
            let (_, _, block) = family.computed_joint::<Bound>(&states, None, None, true).unwrap();
            louis.push(sweep);
            computed.push(block);
        }
        let entries: Vec<_> = (0..louis[0].len()).map(|k| {
            let routes = [("Louis", refinement(&louis, k)), ("block", refinement(&computed, k))];
            let rounding = louis[1][k].rounding() + computed[1][k].rounding();
            let bar = routes[0].1.error + routes[1].1.error + rounding;
            let gap = (louis[1][k].value - computed[1][k].value).abs();
            (routes, rounding, bar, gap)
        }).collect();
        for (k, (routes, rounding, bar, gap)) in entries.iter().enumerate() {
            for (n, order) in orders.iter().enumerate() {
                eprintln!(
                    "{label} entry {k} order {order}: louis {:e} block {:e} louis - block {:e} rounding [{:e}, {:e}]",
                    louis[n][k].value, computed[n][k].value, louis[n][k].value - computed[n][k].value,
                    louis[n][k].rounding(), computed[n][k].rounding()
                );
            }
            let coarse_bar = bar + routes.iter().map(|(_, step)| step.first).sum::<f64>()
                + louis[0][k].rounding() + computed[0][k].rounding();
            let coarse_gap = (louis[0][k].value - computed[0][k].value).abs();
            eprintln!("{label} entry {k}: order 17 gap {gap:e} bar {bar:e} rounding {rounding:e}; order 9 gap {coarse_gap:e} bar {coarse_bar:e}");
            for (route, step) in routes {
                eprintln!(
                    "{label} entry {k} {route}: steps [{:e}, {:e}] resolved {:e} ratio {:e} error {:e}",
                    step.first, step.second, step.resolved, step.ratio, step.error
                );
            }
        }
        for (k, (routes, _, bar, gap)) in entries.iter().enumerate() {
            for (route, step) in routes {
                assert!(step.resolved > 0.0, "{label} entry {k}: the {route} step {:e} from order 9 is not resolved above rounding", step.first);
                assert!(step.ratio < 1.0, "{label} entry {k}: the {route} refinement step did not shrink ({:e} then {:e})", step.first, step.second);
            }
            assert!(computed[1][k].value.abs() > *bar, "{label} entry {k} is not material against its bar {bar:e}");
            assert!(gap <= bar, "{label} entry {k}: Louis {} against the block sweep {} at order 17 differs by {gap:e}, above its refinement bar {bar:e}",
                louis[1][k].value, computed[1][k].value);
        }
    }

    /// All-static atoms take the Louis sweep (#2965): every node shares one
    /// whole-history grid, so the carried score is never moved.
    ///
    /// The fixture's size is set by `Bound`'s growth, fixed before any run at
    /// it. Conditioning a node, `α_n = α_{n−1} L_n / Σ α_{n−1} L_n`, charges the
    /// numerator and the normalising sum apart where the error they share
    /// cancels, so the first-order bound doubles at every node while the
    /// quotient's actual error grows about linearly: job 1250857
    /// measured a node normaliser's bound at 4.5e-13, 1.1e-12, 2.25e-12,
    /// 4.8e-12 and 9.8e-12 over five nodes. That is the tracker's looseness,
    /// not either route's error. At the nineteen nodes of nine Legendre points a
    /// cell, the block route's bound (1.7e-5 at entry 0, job 1250136) exceeded
    /// its first refinement step (3.4e-6). The subject here has seven nodes,
    /// three points in each of its two cells and the event, which puts the
    /// bound about `2^12` lower. The node count is printed. The hazards are
    /// constant in time, so either rule integrates them exactly and the node
    /// count moves only the rounding chain, not the compared entries: at order
    /// 9 the Louis entry 0 is −0.30562673725264494 at seven nodes and
    /// −0.3056267372526445 at nineteen.
    #[test]
    fn all_static_atoms_take_the_louis_sweep_within_the_quadrature_refinement_2965() {
        let (mixed, _) = recurrent_family(0.5, vec![Some(0.0), Some(0.7)], 9);
        assert!(mixed.differentiates_the_computed_path(), "a static atom beside a dynamic one keeps the computed path");
        let fixture = |order| {
            let (family, mut states) = recurrent_family_on(0.5, vec![Some(0.0), Some(0.0)], order, 3);
            states[1].beta = array![1.2, 0.6];
            (family, states)
        };
        eprintln!("all static: {} nodes", fixture(9).0.nodes.total_nodes);
        louis_matches_the_block_sweep_2965("all static", fixture);
    }

    /// Four subjects on one once-only mark, a reference law on 24 equal steps,
    /// `columns` cosine time columns in the mark block, and a near-static held
    /// atom: a reference-centred fixture of any width.
    fn wide_reference_family(columns: usize) -> (EventHistoryFamily, Vec<ParameterBlockState>) {
        let subjects: Vec<SubjectHistory> = (0..4).map(|i| {
            let exit = 6.0 - 0.75 * i as f64;
            SubjectHistory {
                id: format!("s{i}"), entry: 0.0, exit,
                events: if i % 2 == 0 { vec![Event { time: exit, mark: 0 }] } else { Vec::new() },
                segments: vec![CovariateSegment { start: 0.0, row: 0 }],
            }
        }).collect();
        let mut cohort = EventHistoryCohort {
            mark_names: vec!["disease".to_string()], mark_kinds: vec![MarkKind::Once],
            covariate_names: Vec::new(), covariate_levels: Vec::new(),
            covariates: Array2::zeros((1, 0)), subjects,
        };
        cohort.validate().unwrap();
        let nodes = Arc::new(expand_nodes(&cohort, 3, 1).unwrap());
        let intervals = 24;
        let times: Vec<f64> = (0..=intervals).map(|n| 6.0 * n as f64 / intervals as f64).collect();
        let grid = ReferenceGrid { gaps: times.windows(2).map(|w| w[1] - w[0]).collect(), times };
        let basis = |t: f64| -> Vec<f64> {
            (0..columns).map(|j| (j as f64 * std::f64::consts::PI * t / 6.0).cos()).collect()
        };
        let node_times: Vec<f64> = nodes.subjects.iter().flat_map(|s| s.times.iter().copied()).collect();
        let mut design = Array2::<f64>::zeros((nodes.total_nodes, columns));
        for (row, &t) in node_times.iter().enumerate() {
            for (j, x) in basis(t).into_iter().enumerate() {
                design[[row, j]] = x;
            }
        }
        let mut reference_design = Array2::<f64>::zeros((grid.len(), columns));
        for (row, &t) in grid.times.iter().enumerate() {
            for (j, x) in basis(t).into_iter().enumerate() {
                reference_design[[row, j]] = x;
            }
        }
        let locations: Vec<(usize, f64)> = node_times.iter().map(|&t| grid.locate(t).unwrap()).collect();
        let tables = ReferenceTables {
            designs: vec![Arc::new(reference_design)],
            offsets: vec![Array1::zeros(grid.len())], kinds: vec![MarkKind::Once],
            profiles: Array2::zeros((1, 0)), strata: 1,
            node_stratum: vec![0; nodes.total_nodes],
            node_lower: locations.iter().map(|x| x.0).collect(),
            node_weight: locations.iter().map(|x| x.1).collect(), grid,
        };
        let beta = Array1::from_iter((0..columns).map(|j| {
            if j == 0 { -1.2 } else { 0.3 * (-1.0_f64).powi(j as i32) / (j * j) as f64 }
        }));
        let eta = design.dot(&beta);
        let states = vec![
            ParameterBlockState { beta, eta },
            ParameterBlockState { beta: array![0.9], eta: Array1::zeros(nodes.total_nodes) },
        ];
        let family = EventHistoryFamily::new(nodes.clone(), vec![Arc::new(design)], 1, 9, 1.0, vec![Some(1e-8)])
            .unwrap().with_reference(Some(Arc::new(tables)));
        (family, states)
    }

    /// One path evaluation seeded with one coefficient direction per level:
    /// `outer` on the outer level, `inner` on the inner, over the directions `u`
    /// and `v` carries. Every channel of `Rows` is formed from the same operands
    /// in the same order at any width, so this is, bit for bit, the channel a
    /// block sweep or a Hessian-vector product forms for the same pair of
    /// coordinates. The comparisons against it need no bar.
    fn pair_channel<S: Directional>(
        family: &EventHistoryFamily, states: &[ParameterBlockState],
        u: Option<&Array1<f64>>, v: Option<&Array1<f64>>, outer: usize, inner: usize,
    ) -> Rows<Rows<S, 1>, 1> {
        let values: Vec<f64> = states.iter().flat_map(|s| s.beta.iter().copied()).collect();
        let seeded: Vec<Rows<Rows<S, 1>, 1>> = values.iter().enumerate().map(|(q, value)| Rows::seed(
            Rows::seed(S::seeded(*value, u.map_or(0.0, |x| x[q]), v.map_or(0.0, |x| x[q])), [f64::from(q == inner)]),
            [f64::from(q == outer)],
        )).collect();
        family.path_value(states, &seeded).unwrap()
    }

    /// The coefficient-pair replay the block sweeps replaced (#2965): one path
    /// evaluation per coefficient pair `j ≤ i`, `i` outer and `j` inner, which
    /// is the orientation `coordinate_hessian` reads in its lower triangle.
    fn pair_replay<S: Directional>(
        family: &EventHistoryFamily, states: &[ParameterBlockState],
        u: Option<&Array1<f64>>, v: Option<&Array1<f64>>,
    ) -> (S, Vec<S>, Vec<S>) {
        let total = family.total_width();
        let mut value = None;
        let mut gradient = Vec::with_capacity(total);
        let mut hessian: Vec<Option<S>> = vec![None; total * total];
        for i in 0..total {
            for j in 0..=i {
                let result = pair_channel::<S>(family, states, u, v, i, j);
                if j == i {
                    gradient.push(result.base.rows[0].clone());
                }
                hessian[i * total + j] = Some(result.rows[0].rows[0].clone());
                hessian[j * total + i] = Some(result.rows[0].rows[0].clone());
                value = Some(result.base.base);
            }
        }
        (value.expect("at least one coefficient"), gradient,
            hessian.into_iter().map(|entry| entry.expect("every pair is replayed")).collect())
    }

    /// Block sweeps return the pair replay's value, gradient and Hessian, and
    /// the Hessian's first and second directional derivatives, bit for bit
    /// (#2965). Seventeen coefficients make three blocks, the last holding one
    /// coefficient, so a partial block and every off-diagonal block pair are
    /// read. Every compared Hessian entry is nonzero, so equality tests the
    /// seeding and the read-back rather than two zeros.
    #[test]
    fn block_sweeps_reproduce_the_pair_replay_2965() {
        let (family, states) = wide_reference_family(16);
        let total = family.total_width();
        assert_eq!(total, 17);
        let u = Array1::from_iter((0..total).map(|q| 1.0 / (q + 1) as f64));
        let v = Array1::from_iter((0..total).map(|q| if q % 2 == 0 { 0.5 } else { -0.25 }));
        let (value, gradient, hessian) = family.computed_joint::<f64>(&states, None, None, true).unwrap();
        let (oracle_value, oracle_gradient, oracle_hessian) = pair_replay::<f64>(&family, &states, None, None);
        assert_eq!(value, oracle_value);
        assert_eq!(gradient, oracle_gradient);
        assert_eq!(hessian, oracle_hessian);
        assert!(oracle_hessian.iter().all(|entry| *entry != 0.0), "every Hessian entry must be material");

        let (_, _, first) = family.computed_joint::<OneSeed<0>>(&states, Some(&u), None, true).unwrap();
        let (_, _, oracle_first) = pair_replay::<OneSeed<0>>(&family, &states, Some(&u), None);
        let first: Vec<f64> = first.iter().map(Directional::eps).collect();
        let oracle_first: Vec<f64> = oracle_first.iter().map(Directional::eps).collect();
        assert_eq!(first, oracle_first);
        assert!(oracle_first.iter().all(|entry| *entry != 0.0), "every first directional derivative entry must be material");

        let (_, _, second) = family.computed_joint::<TwoSeed<0>>(&states, Some(&u), Some(&v), true).unwrap();
        let (_, _, oracle_second) = pair_replay::<TwoSeed<0>>(&family, &states, Some(&u), Some(&v));
        let second: Vec<f64> = second.iter().map(Directional::eps_del).collect();
        let oracle_second: Vec<f64> = oracle_second.iter().map(Directional::eps_del).collect();
        assert_eq!(second, oracle_second);
        assert!(oracle_second.iter().all(|entry| *entry != 0.0), "every second directional derivative entry must be material");
    }

    /// The workspace's representations are the block sweep's, bit for bit
    /// (#2965). The dense matrix a factorising consumer takes is the negated
    /// pair replay. The matrix-free product along each unit direction `e_j` is
    /// the negated pair channel (outer `q`, inner `j`) in every row `q`, which
    /// is the orientation its outer block level carries. The gradient and value
    /// are the joint evaluation's. A sign slip between representations would
    /// break an equality.
    #[test]
    fn the_computed_workspace_streams_the_block_sweep_curvature_2965() {
        use gam_model_api::families::custom_family::{JointHessianSourcePreference, MaterializationIntent};
        let (family, states) = wide_reference_family(16);
        let total = family.total_width();
        assert!(family.differentiates_the_computed_path());
        let workspace = ComputedHessianWorkspace::new(family.clone(), states.clone());
        let joint = family.joint_evaluation(&states).unwrap();
        let dense = workspace.hessian_dense().unwrap().expect("a dense Hessian for factorising consumers");
        let (_, _, oracle) = pair_replay::<f64>(&family, &states, None, None);
        for q in 0..total {
            for r in 0..total {
                assert_eq!(dense[[q, r]], -oracle[q * total + r], "dense [{q}, {r}] against the negated pair replay");
            }
        }
        assert!(oracle.iter().all(|entry| *entry != 0.0), "every Hessian entry must be material");
        for j in 0..total {
            let mut unit = Array1::zeros(total);
            unit[j] = 1.0;
            let product = workspace.hessian_matvec(&unit).unwrap().expect("a Hessian-vector product");
            for q in 0..total {
                let channel = pair_channel::<f64>(&family, &states, None, None, q, j).rows[0].rows[0];
                assert_eq!(product[q], -channel, "H e_{j} [{q}] against the pair channel (outer {q}, inner {j})");
            }
        }
        let evaluation = workspace.joint_gradient_evaluation().unwrap().expect("a gradient");
        for q in 0..total {
            assert_eq!(evaluation.gradient[q], joint.gradient[q], "gradient [{q}]");
        }
        assert_eq!(evaluation.log_likelihood, joint.log_likelihood);
        // The Newton direction takes the dense block sweep, the preference
        // measured faster at p = 17 in job 1219149 (static atom, fixed λ). An
        // operator preference is measured again before it replaces this one.
        assert_eq!(
            workspace.hessian_source_preference_for_intent(MaterializationIntent::InnerSolve),
            JointHessianSourcePreference::Dense
        );
    }
}

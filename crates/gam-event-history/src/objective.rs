//! Total derivatives of the reference-normalised likelihood.
use super::*;
use crate::scalar::Mixed;

impl EventHistoryFamily {
    fn reference_values<S: JetField>(
        &self, beta: &[S], loadings: &[S], rates: &[S],
    ) -> Result<crate::preserve::Normalisers<S>, String> {
        let tables = self.reference.as_ref()
            .ok_or_else(|| "this family has no reference population".to_string())?;
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
            let out = stratum_normalisers(&tables.grid, &eta0, loadings, rates,
                self.time_scale, &self.gh, &tables.kinds, self.atoms)
                .map_err(|error| error.to_string())?;
            masks = out.masks;
            normalisers.extend(out.log_normaliser);
            risk_mass.extend(out.log_risk_mass);
        }
        Ok(crate::preserve::Normalisers {
            log_normaliser: normalisers, log_risk_mass: risk_mass, masks,
        })
    }

    pub(super) fn computed_reference(&self, states: &[ParameterBlockState]) -> Result<RiskSetCentring, String> {
        self.validate_states(states)?;
        let beta: Vec<f64> = states.iter().flat_map(|s| s.beta.iter().copied()).collect();
        let latent_offset = self.block_offsets()[self.marks()];
        let latent = Array1::from(beta[latent_offset..].to_vec());
        let rates = self.atom_rates(&latent);
        let out = self.reference_values(&beta,
            &beta[latent_offset..latent_offset + self.marks() * self.atoms], &rates)?;
        let tables = self.reference.as_ref()
            .ok_or_else(|| "reference centring requires reference tables".to_string())?;
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
    ) -> Result<S, String> {
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
            Some(tables.carry_to_nodes(&values.log_normaliser, marks, self.nodes.total_nodes)
                .map_err(|error| error.to_string())?)
        } else { None };
        let results: Result<Vec<S>, String> = self.nodes.subjects.par_iter().map(|subject| {
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
                .map_err(|error| error.to_string())
        }).collect();
        Ok(pairwise_sum(&results?, &beta[0].constant_like(0.0)))
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
        let zero = beta[0].constant_like(0.0);
        let mut value = zero.clone();
        let mut gradient = vec![zero.clone(); total];
        let mut hessian = vec![zero; total * total];
        for i in 0..total {
            for j in 0..=i {
                let seeded: Vec<Mixed<S>> = beta.iter().enumerate().map(|(q, b)|
                    Mixed::seed(b.clone(), f64::from(q == i), f64::from(q == j))).collect();
                let result = self.path_value(states, &seeded)?;
                value = result.base;
                gradient[i] = result.u;
                hessian[i * total + j] = result.uv.clone();
                hessian[j * total + i] = result.uv;
            }
        }
        Ok((value, gradient, hessian))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cohort::{CovariateSegment, Event, SubjectHistory};
    use ndarray::array;

    fn single_event() -> (EventHistoryFamily, Vec<ParameterBlockState>) {
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
        let intervals = 144;
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
        let states = vec![
            ParameterBlockState { beta: array![-1.2], eta: Array1::from_elem(nodes.total_nodes, -1.2) },
            ParameterBlockState { beta: array![0.9], eta: Array1::zeros(nodes.total_nodes) },
        ];
        let family = EventHistoryFamily::new(nodes.clone(), vec![Arc::new(Array2::ones((nodes.total_nodes, 1)))],
            1, 15, 1.0, vec![Some(1e-8)]).unwrap().with_reference(Some(Arc::new(tables)));
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

    fn recurrent_family(event_time: f64, rates: Vec<Option<f64>>, order: usize) -> (EventHistoryFamily, Vec<ParameterBlockState>) {
        let mut cohort = EventHistoryCohort {
            mark_names: vec!["event".to_string()], mark_kinds: vec![MarkKind::Recurrent],
            covariate_names: Vec::new(), covariate_levels: Vec::new(), covariates: Array2::zeros((1, 0)),
            subjects: vec![SubjectHistory { id: "one".to_string(), entry: 0.0, exit: 1.0,
                events: vec![Event { time: event_time, mark: 0 }],
                segments: vec![CovariateSegment { start: 0.0, row: 0 }] }],
        };
        cohort.validate().unwrap();
        let nodes = Arc::new(expand_nodes(&cohort, 9, 0).unwrap());
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
            let beta = vec![Mixed::seed(0.0, 0.0, 0.0), Mixed::seed(2.0, 0.0, 0.0), Mixed::seed(0.0, 1.0, 1.0)];
            let curvature = family.path_value(&states, &beta).unwrap().uv;
            let h = 1e-3;
            let eval = |a: f64| family.path_value(&states, &[0.0, 2.0, a]).unwrap();
            let finite = (eval(h) + eval(-h) - 2.0 * eval(0.0)) / (h * h);
            eprintln!("event {event_time}: AD={curvature}, finite={finite}, target={expected}");
            assert!((curvature - finite).abs() < 2e-5);
            assert!((curvature - expected).abs() < 1e-4, "{curvature} vs {expected}");
            assert!(checked_loading_curvature(&family, &states).is_ok());
            curvatures.push(curvature);
        }
        assert!(curvatures.iter().all(|c| (c - curvatures[0]).abs() < 1e-12));
    }

    #[test]
    fn unresolved_near_static_curvature_is_rejected() {
        let (family, states) = recurrent_family(0.1, vec![Some(1e-10), Some(1e-10)], 17);
        assert!(checked_loading_curvature(&family, &states).is_err());
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
}

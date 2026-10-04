//! Minimum verified cost over a finite, explicitly supplied candidate bank.
//!
//! Candidates are independent: infeasible individual edits never block a joint candidate.
//! The gap concerns this bank only, never ungenerated programs. A fresh cache belongs to one
//! fixed Local/RunCheck dataset and is reused across tolerance pairs, not across datasets.
use super::acceptance::{Assessment, Constraint, CostCache, Local, RunCheck, assess_once, structural_cost};
use super::artifact::{Artifact, EncodedArtifact};
use super::precision::FidelityVerdict;

#[derive(Clone, Debug)]
pub struct Candidate {
    pub label: String,
    pub artifact: Artifact,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum State {
    Verified,
    Violates,
    Unresolved,
    Unevaluated,
    /// Assessment failed, so infeasibility has not been established.
    Failed(String),
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Evidence {
    pub label: String,
    pub cost: u64,
    pub state: State,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Point {
    pub constraint: Constraint,
    /// Index into the returned bank (aliases of identical messages are collapsed).
    pub selected: Option<usize>,
    pub upper_cost: Option<u64>,
    /// Cheapest candidate not proven infeasible, including unresolved and unmeasured ones.
    /// None means every bank candidate is proven infeasible.
    pub lower_cost: Option<u64>,
    /// None means there is no verified incumbent. Zero proves optimality within this bank.
    pub gap: Option<u64>,
    pub evidence: Vec<Evidence>,
}

#[derive(Clone, Debug)]
pub struct Frontier {
    /// Sorted by cost, then label; exact duplicate encoded messages are collapsed.
    pub bank: Vec<Candidate>,
    pub points: Vec<Point>,
    /// Distinct messages whose assessment was attempted, including failed assessments.
    pub measured_candidates: usize,
}

fn state(assessment: &Assessment, constraint: Constraint) -> Result<State, String> {
    let local = assessment.local.with_tolerance(constraint.local)?;
    let run = assessment.run.with_tolerance(constraint.run)?;
    Ok(match (local.verdict(), run.verdict()) {
        (FidelityVerdict::Violates, _) | (_, FidelityVerdict::Violates) => State::Violates,
        (FidelityVerdict::Meets, FidelityVerdict::Meets) => State::Verified,
        _ => State::Unresolved,
    })
}

fn point(constraint: Constraint, evidence: Vec<Evidence>) -> Point {
    let selected = evidence
        .iter()
        .enumerate()
        .filter(|(_, e)| e.state == State::Verified)
        .min_by(|(_, a), (_, b)| a.cost.cmp(&b.cost).then_with(|| a.label.cmp(&b.label)))
        .map(|(i, _)| i);
    let upper_cost = selected.map(|i| evidence[i].cost);
    let lower_cost = evidence.iter().filter(|e| e.state != State::Violates).map(|e| e.cost).min();
    let gap = upper_cost.zip(lower_cost).map(|(upper, lower)| upper - lower);
    Point { constraint, selected, upper_cost, lower_cost, gap, evidence }
}

/// Evaluate at most `budget` distinct messages in cost order, independently of the tolerance grid.
/// Every budgeted candidate is assessed even after a cheaper feasible candidate is found.
/// No intermediate-path feasibility or local screen prunes a supplied candidate.
pub fn frontier(
    local: &Local<'_>,
    run: &dyn RunCheck,
    candidates: Vec<Candidate>,
    constraints: &[Constraint],
    budget: usize,
) -> Result<Frontier, String> {
    for constraint in constraints {
        if !constraint.local.is_finite() || !constraint.run.is_finite() || constraint.local < 0.0 || constraint.run < 0.0 {
            return Err("tolerances must be finite and nonnegative".to_string());
        }
    }
    let mut cache = CostCache::default();
    let mut ranked = Vec::new();
    for candidate in candidates {
        let artifact = candidate.artifact.f32_literals()?;
        if artifact.program.declarations != local.model.declarations {
            return Err("candidate declarations differ from the fixed dataset's model".into());
        }
        let cost = structural_cost(&artifact, &mut cache)?.total();
        let message = EncodedArtifact::of(&artifact)?.message;
        ranked.push((cost, candidate.label, artifact, message));
    }
    ranked.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut unique = Vec::new();
    for candidate in ranked {
        if !unique.iter().any(|earlier: &(u64, String, Artifact, super::codec::BitString)| earlier.3 == candidate.3) {
            unique.push(candidate);
        }
    }
    // Deduplication no longer needs the encoded messages. Each distinct candidate
    // is assessed once and its returned evidence is reused across the grid.
    let unique: Vec<_> = unique.into_iter().map(|(cost, label, artifact, _)| (cost, label, artifact)).collect();
    let measured_candidates = if constraints.is_empty() { 0 } else { budget.min(unique.len()) };
    let mut evidence: Vec<Vec<Evidence>> = constraints.iter().map(|_| Vec::new()).collect();
    for (index, (cost, label, artifact)) in unique.iter().enumerate() {
        // Measurement is independent of the declared tolerance grid. Encoding and decoding
        // a real model for each point can cost more than evaluating it; reuse its evidence.
        let assessed = if index < measured_candidates {
            Some(assess_once(local, run, artifact, constraints[0], &mut cache))
        } else {
            None
        };
        for (constraint, row) in constraints.iter().zip(&mut evidence) {
            let state = match &assessed {
                None => State::Unevaluated,
                Some(Err(error)) => State::Failed(error.clone()),
                Some(Ok(assessment)) => state(assessment, *constraint)?,
            };
            row.push(Evidence { label: label.clone(), cost: *cost, state });
        }
    }
    let points = constraints.iter().copied().zip(evidence).map(|(c, e)| point(c, e)).collect();
    let bank = unique.into_iter().map(|(_, label, artifact)| Candidate { label, artifact }).collect();
    Ok(Frontier { bank, points, measured_candidates })
}

/// Mutually exclusive alternatives at one declared location. Keeping the current computation is
/// always an additional choice. Groups are combined without any fidelity-based pruning.
#[derive(Clone, Debug)]
pub struct ChoiceGroup {
    pub label: String,
    pub choices: Vec<String>,
}

/// Full Cartesian bank, including the unchanged start. `apply` receives original group/choice
/// indices and must apply that choice to the supplied combined artifact. It must not screen
/// fidelity. Exceeding the declared generation budget is an error before any candidate is built.
pub fn combination_bank(
    start: &Artifact,
    groups: &[ChoiceGroup],
    max_bank: usize,
    mut apply: impl FnMut(&Artifact, usize, usize) -> Result<Artifact, String>,
) -> Result<Vec<Candidate>, String> {
    let size = groups.iter().try_fold(1usize, |n, group| {
        let choices = group.choices.len().checked_add(1).ok_or("candidate bank size overflow")?;
        n.checked_mul(choices).ok_or("candidate bank size overflow")
    })?;
    if size > max_bank {
        return Err(format!("complete candidate bank has {size} combinations, exceeding declared generation budget {max_bank}"));
    }
    let mut bank = vec![Candidate { label: "native".into(), artifact: start.clone() }];
    for (group_index, group) in groups.iter().enumerate() {
        let mut next = Vec::with_capacity(bank.len() * (group.choices.len() + 1));
        for base in bank {
            next.push(base.clone());
            for (choice_index, choice) in group.choices.iter().enumerate() {
                let artifact = apply(&base.artifact, group_index, choice_index)?;
                next.push(Candidate { label: format!("{}; {}={choice}", base.label, group.label), artifact });
            }
        }
        bank = next;
    }
    Ok(bank)
}

/// Every combination of independently supplied MLP accounts, with at most one account per layer.
/// No individual account must be feasible for a combination containing it to enter the bank.
pub fn account_bank(start: &Artifact, proposer: &super::proposals::AccountProposer, max_bank: usize) -> Result<Vec<Candidate>, String> {
    let mut grouped = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    for (index, (layer, _, _)) in proposer.accounts.iter().enumerate() {
        if *layer >= proposer.layers.len() {
            return Err(format!("account names absent layer {layer}"));
        }
        grouped.entry(*layer).or_default().push(index);
    }
    let groups: Vec<(usize, Vec<usize>)> = grouped.into_iter().collect();
    let choices: Vec<ChoiceGroup> = groups
        .iter()
        .map(|(layer, indices)| ChoiceGroup {
            label: format!("MLP layer {layer}"),
            choices: indices.iter().map(|&i| proposer.accounts[i].1.clone()).collect(),
        })
        .collect();
    combination_bank(start, &choices, max_bank, |artifact, group, choice| {
        let (layer, indices) = &groups[group];
        let (_, name, account) = &proposer.accounts[indices[choice]];
        super::proposals::with_account(artifact, name, &proposer.layers[*layer], account)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const C: Constraint = Constraint { local: 0.1, run: 0.1 };
    fn e(label: &str, cost: u64, state: State) -> Evidence {
        Evidence { label: label.into(), cost, state }
    }

    #[test]
    fn joint_candidate_survives_infeasible_individual_edits() {
        let p = point(
            C,
            vec![
                e("edit A", 30, State::Violates),
                e("edit B", 30, State::Violates),
                e("joint A+B", 10, State::Verified),
                e("native", 100, State::Verified),
            ],
        );
        assert_eq!(p.selected, Some(2));
        assert_eq!(p.upper_cost, Some(10));
        assert_eq!(p.gap, Some(0));
    }

    #[test]
    fn unresolved_cheaper_candidate_prevents_zero_gap() {
        let p = point(C, vec![e("uncertain", 10, State::Unresolved), e("verified", 20, State::Verified)]);
        assert_eq!(p.lower_cost, Some(10));
        assert_eq!(p.gap, Some(10));
    }

    #[test]
    fn fully_evaluated_order_does_not_change_selected_label() {
        let bank = vec![e("z", 10, State::Verified), e("a", 10, State::Verified), e("bad", 5, State::Violates)];
        let p = point(C, bank.clone());
        let q = point(C, bank.into_iter().rev().collect());
        assert_eq!(p.evidence[p.selected.unwrap()].label, "a");
        assert_eq!(q.evidence[q.selected.unwrap()].label, "a");
        assert_eq!(p.gap, q.gap);
    }

    #[test]
    fn unevaluated_candidate_and_failed_assessment_preserve_gap() {
        for state in [State::Unevaluated, State::Failed("budgeted assessment failed".into())] {
            let p = point(C, vec![e("not ruled out", 3, state), e("incumbent", 15, State::Verified)]);
            assert_eq!(p.lower_cost, Some(3));
            assert_eq!(p.gap, Some(12));
        }
    }

    #[test]
    fn no_incumbent_does_not_claim_optimality() {
        let p = point(C, vec![e("unknown", 3, State::Unresolved)]);
        assert_eq!(p.gap, None);
        assert_eq!(p.upper_cost, None);
    }

    const REAL_C: Constraint = Constraint { local: 2.0, run: 0.001 };

    fn fixture() -> (super::super::operator_program::OperatorProgram, super::super::operator_program::FamilyInputs) {
        use super::super::operator_program::{Declarations, FamilyInputs, Interface, Node, Operator, OperatorProgram, Slot, SlotValues};
        let interface = Interface::native(2).unwrap();
        (
            OperatorProgram {
                declarations: Declarations { domains: Vec::new(), slots: vec![Slot::Raw { width: 2 }], parameters: 0 },
                bases: Vec::new(),
                operators: vec![
                    std::sync::Arc::new(Operator::identity("A", interface.clone())),
                    std::sync::Arc::new(Operator::identity("B", interface)),
                ],
                rules: Vec::new(),
                nodes: vec![
                    Node::Raw { slot: 0 },
                    Node::Affine { terms: vec![(0, 0)], bias: None },
                    Node::Affine { terms: vec![(1, 1)], bias: None },
                ],
                output: 2,
            },
            FamilyInputs { rows: 2, slots: vec![SlotValues::Raw(ndarray::array![[1.0, 2.0], [2.0, -1.0]])], layout: None },
        )
    }

    fn candidate(model: &super::super::operator_program::OperatorProgram, label: &str, a: f64, b: f64) -> Candidate {
        use super::super::operator_program::{Interface, Operator, Provenance, exact_precision};
        let mut artifact = Artifact::native(model).unwrap();
        let interface = Interface::native(2).unwrap();
        for (index, scale) in [a, b].into_iter().enumerate() {
            if scale == 1.0 {
                continue;
            }
            let values = ndarray::array![[scale, 0.0], [0.0, scale]];
            artifact.program.operators[index] = std::sync::Arc::new(
                Operator::dense(
                    format!("factor {index}"),
                    interface.clone(),
                    interface.clone(),
                    values.clone(),
                    exact_precision(values.iter().copied()).unwrap(),
                    Provenance::native("test factor"),
                )
                .unwrap(),
            );
            artifact = artifact.bind(&format!("factor {index}"), &[index], index + 1).unwrap();
        }
        Candidate { label: label.into(), artifact }
    }

    struct Counted<'a> {
        run: super::super::acceptance::FamilyRun<'a>,
        calls: std::sync::atomic::AtomicUsize,
        widen_native: bool,
    }
    impl RunCheck for Counted<'_> {
        fn episodes(&self, artifact: &Artifact) -> Result<Vec<super::super::acceptance::EpisodeScore>, String> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut scores = self.run.episodes(artifact)?;
            // Deliberately unresolved numerical evidence for the cheap unchanged program.
            // Identity is a physical matrix property, independent of serialization labels.
            if self.widen_native && artifact.blocks.is_empty() {
                for score in &mut scores {
                    score.numerical_error = 0.01;
                }
            }
            Ok(scores)
        }
    }
    fn counted<'a>(
        model: &'a super::super::operator_program::OperatorProgram,
        family: &super::super::operator_program::FamilyInputs,
        widen_native: bool,
    ) -> Counted<'a> {
        Counted {
            run: super::super::acceptance::FamilyRun {
                model,
                family: family.clone(),
                readouts: 1,
                episodes: vec![super::super::acceptance::Episode { id: "clean".into(), group: "clean".into(), edits: Vec::new() }],
            },
            calls: std::sync::atomic::AtomicUsize::new(0),
            widen_native,
        }
    }

    #[test]
    fn bank_assesses_joint_without_pruning_and_reuses_measurements_across_grid() {
        let (model, family) = fixture();
        let local = Local::new(&model, family.clone(), None, 2);
        let run = counted(&model, &family, false);
        let bank = vec![candidate(&model, "editA", 2.0, 1.0), candidate(&model, "editB", 1.0, 0.5), candidate(&model, "joint", 2.0, 0.5)];
        let a = frontier(&local, &run, bank.clone(), &[REAL_C, Constraint { local: 3.0, run: 0.002 }], 3).unwrap();
        assert_eq!(run.calls.load(std::sync::atomic::Ordering::Relaxed), 3);
        assert_eq!(a.measured_candidates, 3);
        for p in &a.points {
            assert_eq!(a.bank[p.selected.unwrap()].label, "joint");
            assert_eq!(p.gap, Some(0));
            assert_eq!(p.evidence.iter().filter(|e| e.state == State::Violates).count(), 2);
        }
        let b = frontier(&local, &run, bank.into_iter().rev().collect(), &[REAL_C], 3).unwrap();
        assert_eq!(b.bank[b.points[0].selected.unwrap()].label, "joint");
        assert_eq!(a.points[0].upper_cost, b.points[0].upper_cost);
        assert_eq!(run.calls.load(std::sync::atomic::Ordering::Relaxed), 6);
    }

    #[test]
    fn distinct_message_budget_counts_duplicates_once_and_retains_unknowns() {
        let (model, family) = fixture();
        let local = Local::new(&model, family.clone(), None, 2);
        let run = counted(&model, &family, false);
        let same = candidate(&model, "editA", 2.0, 1.0);
        let f = frontier(&local, &run, vec![same.clone(), same, candidate(&model, "joint", 2.0, 0.5)], &[REAL_C], 1).unwrap();
        assert_eq!(f.bank.len(), 2);
        assert_eq!(f.measured_candidates, 1);
        assert_eq!(run.calls.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(f.points[0].evidence.iter().filter(|e| e.state == State::Unevaluated).count(), 1);
        assert!(f.points[0].lower_cost.is_some());
    }

    #[test]
    fn budget_leaves_positive_gap_when_cheaper_candidate_is_unresolved() {
        let (model, family) = fixture();
        let local = Local::new(&model, family.clone(), None, 2);
        let run = counted(&model, &family, true);
        let cheap = Candidate { label: "cheap".into(), artifact: Artifact::native(&model).unwrap() };
        let f =
            frontier(&local, &run, vec![candidate(&model, "zzzzz", 3.0, 0.25), candidate(&model, "joint", 2.0, 0.5), cheap], &[REAL_C], 2).unwrap();
        assert_eq!(f.measured_candidates, 2);
        let p = &f.points[0];
        assert_eq!(f.bank[p.selected.unwrap()].label, "joint");
        assert!(p.gap.unwrap() > 0);
        assert_eq!(p.evidence[0].state, State::Unresolved);
        assert_eq!(p.evidence[2].state, State::Unevaluated);
    }
    #[test]
    fn cartesian_bank_generates_joint_choices_without_individual_screening() {
        let (model, _) = fixture();
        let start = Artifact::native(&model).unwrap();
        let groups = vec![
            ChoiceGroup { label: "A".into(), choices: vec!["a1".into(), "a2".into()] },
            ChoiceGroup { label: "B".into(), choices: vec!["b1".into()] },
        ];
        let mut calls = Vec::new();
        let bank = combination_bank(&start, &groups, 6, |artifact, group, choice| {
            calls.push((group, choice));
            Ok(artifact.clone())
        })
        .unwrap();
        assert_eq!(bank.len(), 6);
        assert!(bank.iter().any(|c| c.label == "native; A=a2; B=b1"));
        assert_eq!(calls.len(), 5);
        assert!(combination_bank(&start, &groups, 5, |_, _, _| panic!("must reject before generation")).is_err());
    }
}

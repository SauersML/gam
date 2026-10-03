//! Autonomous learned switching from the executing masked program's own amplitudes.

use super::gates::Switch;
use super::masked::{Masked, previous_inputs};
use super::operator_program::{FamilyInputs, Node, Trace};
use ndarray::Array2;

/// Run one unbanded forward and return its trace and per-block binary masks.
/// Feature pieces index amplitude columns, including for blocks of rank greater than one.
/// Same-position features must already be available (the site's own amplitude is allowed).
/// Previous-position features require an intervening causal attention or mixing node.
/// This dense executor computes the library's reads, including reads of components subsequently
/// switched off. Active-component counts therefore do not measure its full execution cost.
/// These decisions are execution results, not a fidelity certificate.
pub fn execute(masked: &Masked, base: &FamilyInputs, switches: &[Vec<Switch>]) -> Result<(Trace, Vec<Array2<f64>>), String> {
    execute_at(masked, base, switches, &vec![1.0; masked.program.declarations.parameters])
}

/// Run the same learned program under declared controls. The controls alter its states and
/// hence may change its subsequent switch decisions. No native trace or target is consulted.
pub fn execute_at(masked: &Masked, base: &FamilyInputs, switches: &[Vec<Switch>], parameters: &[f64]) -> Result<(Trace, Vec<Array2<f64>>), String> {
    let count = masked.sites.len();
    if switches.len() != count || masked.z.len() != count || masked.masked.len() != count || masked.slots.len() != count {
        return Err("autonomous switches and masked site counts differ".into());
    }
    if let Some(layout) = &base.layout
        && (layout.sequence.len() != base.rows || layout.position.len() != base.rows)
    {
        return Err("autonomous input layout does not match its rows".into());
    }
    let mut pairs = Vec::with_capacity(count);
    for (site, site_switches) in switches.iter().enumerate() {
        if site_switches.len() != masked.blocks(site) {
            return Err(format!("site {site}: switch count differs from block count"));
        }
        let z = masked.z[site];
        let mask_node = match masked.program.nodes.get(masked.masked[site]) {
            Some(Node::Hadamard { left, right }) if *left == z => *right,
            Some(Node::Hadamard { left, right }) if *right == z => *left,
            _ => return Err(format!("site {site}: missing masked amplitude product")),
        };
        if !matches!(masked.program.nodes.get(mask_node), Some(Node::Raw { slot }) if *slot == masked.slots[site]) {
            return Err(format!("site {site}: missing raw mask slot"));
        }
        pairs.push((z, mask_node));
        for switch in site_switches {
            let d = switch.features.len();
            if switch.linear.len() != d || switch.units.iter().any(|unit| unit.w.len() != d)
                || !switch.beta.is_finite() || switch.linear.iter().any(|v| !v.is_finite())
                || switch.units.iter().any(|unit| !unit.d.is_finite() || !unit.c.is_finite() || unit.w.iter().any(|v| !v.is_finite()))
            {
                return Err(format!("site {site}: invalid switch coefficients"));
            }
            for feature in &switch.features {
                if feature.site >= count || feature.piece >= masked.pieces(feature.site) || feature.lag > 1 {
                    return Err(format!("site {site}: invalid feature site, amplitude column, or lag"));
                }
                let source = masked.z[feature.site];
                if source > z {
                    return Err(format!("site {site}: feature amplitude is not available"));
                }
                if feature.lag == 1 && (source >= z || !masked.program.nodes[source + 1..z].iter().any(|node| {
                    matches!(node, Node::Attend { causal: true, .. } | Node::Mix { .. })
                })) {
                    return Err(format!("site {site}: previous-position feature lacks an intervening causal mixing step"));
                }
            }
        }
    }
    let mut previous = previous_inputs(base);
    if let Some(layout) = &base.layout {
        for (row, prior) in previous.iter_mut().enumerate() {
            if prior.is_some_and(|r| layout.position[r].checked_add(1) != Some(layout.position[row])) {
                *prior = None;
            }
        }
    }
    let mut masks: Vec<Array2<f64>> = (0..count).map(|site| Array2::zeros((base.rows, masked.blocks(site)))).collect();
    let family = masked.family(base, &masks);
    let trace = masked.program.execute_with_gates_at(&family, &pairs, parameters, |z, values| {
        let site = masked.z.iter().position(|node| *node == z).ok_or("unknown autonomous amplitude node")?;
        for (block, switch) in switches[site].iter().enumerate() {
            let mut features = vec![0.0; switch.features.len()];
            for row in 0..base.rows {
                for (k, feature) in switch.features.iter().enumerate() {
                    let source_row = if feature.lag == 0 { Some(row) } else { previous[row] };
                    features[k] = source_row.map_or(0.0, |r| feature.value(values[masked.z[feature.site]][[r, feature.piece]]));
                    if !features[k].is_finite() {
                        return Err(format!("site {site}: nonfinite switch feature"));
                    }
                }
                let logit = switch.logit(&features);
                if !logit.is_finite() {
                    return Err(format!("site {site}: nonfinite switch logit"));
                }
                masks[site][[row, block]] = if logit > 0.0 { 1.0 } else { 0.0 };
            }
        }
        Ok(masked.expand(site, &masks[site]))
    }).map_err(|error| error.to_string())?;
    Ok((trace, masks))
}

#[cfg(test)]
mod tests {
    use super::execute;
    use crate::gates::{Feature, Switch, masks};
    use crate::masked::{Library, Masked, sites};
    use crate::operator_program::{Declarations, FamilyInputs, Interface, Node, Operator, OperatorProgram, Provenance, SequenceLayout, Slot, SlotValues};
    use crate::precision::DeclaredPrecision;
    use ndarray::{Array1, Array2, array};
    use std::sync::Arc;

    fn fixture(mixing: bool) -> (Masked, FamilyInputs) {
        fixture_rank(mixing, 1)
    }

    fn fixture_rank(mixing: bool, rank: usize) -> (Masked, FamilyInputs) {
        fixture_control(mixing, rank, false)
    }

    fn fixture_control(mixing: bool, rank: usize, controlled: bool) -> (Masked, FamilyInputs) {
        let interface = Interface::native(1).expect("interface");
        let op = |name: &str| Arc::new(Operator::dense(name, interface.clone(), interface.clone(), array![[1.0]], DeclaredPrecision::new(40).expect("precision"), Provenance::default()).expect("operator"));
        let mut nodes = vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0, 0)], bias: None }];
        if controlled {
            nodes.push(Node::Gain { input: 1, coefficient: crate::operator_program::Coefficient::Parameter(0) });
        }
        let first = nodes.len() - 1;
        let input = if mixing {
            nodes.push(Node::Attend { query: 0, key: 0, value: first, scale: crate::operator_program::Scale::One, rotary: None, causal: true });
            nodes.len() - 1
        } else { first };
        nodes.push(Node::Affine { terms: vec![(input, 1)], bias: None });
        let program = OperatorProgram { declarations: Declarations { parameters: usize::from(controlled), domains: vec![], slots: vec![Slot::Raw { width: 1 }] }, bases: vec![], operators: vec![op("first"), op("second")], rules: vec![], output: nodes.len() - 1, nodes };
        let all = sites(&program);
        let selected = ["first", "second"].iter().map(|name| all.iter().find(|site| site.name == *name).expect("site").clone()).collect();
        let library = || Library { v: Array2::<f64>::ones((rank, 1)), u: Array2::<f64>::from_elem((rank, 1), 1.0 / rank as f64), mean: Array1::zeros(1) };
        let masked = Masked::build_blocks(&program, selected, vec![library(), library()], vec![vec![rank]; 2]).expect("masked");
        let base = FamilyInputs { rows: 3, slots: vec![SlotValues::Raw(array![[2.0], [3.0], [4.0]])], layout: Some(SequenceLayout { sequence: vec![0, 0, 1], position: vec![0, 1, 0] }) };
        (masked, base)
    }

    fn constant(on: bool) -> Switch {
        Switch { features: vec![], beta: if on { 1.0 } else { -1.0 }, linear: vec![], units: vec![], precision: 0, function_bits: 0.0, listing_bits: 0.0, inputs: 0 }
    }

    fn threshold(site: usize, lag: usize) -> Switch {
        Switch { features: vec![Feature { site, piece: 0, lag, magnitude: true }], beta: -1.0, linear: vec![1.0], ..constant(true) }
    }

    fn replay(masked: &Masked, base: &FamilyInputs, trace: &crate::operator_program::Trace, chosen: &[Array2<f64>]) {
        let fixed = masked.program.execute(&masked.family(base, chosen), false).expect("fixed forward");
        assert_eq!(trace.values, fixed.values);
    }

    #[test]
    fn downstream_reads_its_changed_own_amplitude_and_replays_exactly() {
        let (masked, base) = fixture(false);
        let switches = vec![vec![constant(false)], vec![threshold(1, 0)]];
        let clean = masked.program.execute(&masked.family(&base, &vec![Array2::<f64>::ones((base.rows, 1)); 2]), false).expect("clean");
        let teacher = masks(&switches, &masked.z.iter().map(|z| clean.values[*z].clone()).collect::<Vec<_>>(), &crate::masked::previous_inputs(&base));
        let (trace, chosen) = execute(&masked, &base, &switches).expect("autonomous");
        assert_eq!(teacher[1], Array2::<f64>::ones((base.rows, 1)));
        assert_eq!(chosen[1], Array2::<f64>::zeros((base.rows, 1)));
        replay(&masked, &base, &trace, &chosen);
        let (trace, chosen) = execute(&masked, &base, &vec![vec![constant(true)]; 2]).expect("constant");
        assert_eq!(chosen, vec![Array2::<f64>::ones((base.rows, 1)); 2]);
        replay(&masked, &base, &trace, &chosen);
    }

    #[test]
    fn rejects_future_features_and_lags_without_mixing() {
        let (masked, base) = fixture(false);
        assert!(execute(&masked, &base, &vec![vec![threshold(1, 0)], vec![constant(true)]]).is_err());
        assert!(execute(&masked, &base, &vec![vec![constant(true)], vec![threshold(0, 1)]]).is_err());
        assert!(execute(&masked, &base, &vec![vec![constant(true)], vec![threshold(0, 2)]]).is_err());
    }

    #[test]
    fn previous_position_does_not_cross_sequence_boundaries() {
        let (masked, base) = fixture(true);
        let (trace, chosen) = execute(&masked, &base, &vec![vec![constant(true)], vec![threshold(0, 1)]]).expect("lagged");
        assert_eq!(chosen[1], array![[0.0], [1.0], [0.0]]);
        replay(&masked, &base, &trace, &chosen);
    }
    #[test]
    fn skipped_positions_are_not_previous_position_features() {
        let (masked, mut base) = fixture(true);
        base.layout.as_mut().expect("layout").position[1] = 2;
        let (trace, chosen) = execute(&masked, &base, &vec![vec![constant(true)], vec![threshold(0, 1)]]).expect("lagged");
        assert_eq!(chosen[1], Array2::<f64>::zeros((base.rows, 1)));
        replay(&masked, &base, &trace, &chosen);
    }

    #[test]
    fn rank_two_blocks_expand_and_features_index_amplitude_columns() {
        let (masked, base) = fixture_rank(false, 2);
        let mut second = threshold(1, 0);
        second.features[0].piece = 1;
        let (trace, chosen) = execute(&masked, &base, &vec![vec![constant(true)], vec![second]]).expect("rank two");
        assert_eq!(chosen, vec![Array2::<f64>::ones((base.rows, 1)); 2]);
        assert_eq!(trace.values[masked.masked[0]].ncols(), 2);
        replay(&masked, &base, &trace, &chosen);
    }

    #[test]
    fn executor_rejects_invalid_decision_values_and_shapes() {
        let (masked, base) = fixture(false);
        let family = masked.family(&base, &vec![Array2::<f64>::ones((base.rows, 1)); 2]);
        let pairs = vec![(masked.z[0], masked.z[0] - 1)];
        for invalid in [Array2::<f64>::from_elem((base.rows, 1), 0.5), Array2::<f64>::from_elem((base.rows, 1), f64::NAN), Array2::<f64>::ones((base.rows, 2))] {
            assert!(masked.program.execute_with_gates(&family, &pairs, |_, _| Ok(invalid.clone())).is_err());
        }
    }

    #[test]
    fn interventions_recompute_switches_and_replay_at_the_same_controls() {
        let (masked, base) = fixture_control(false, 1, true);
        let switches = vec![vec![constant(true)], vec![threshold(1, 0)]];
        let (clean, before) = execute(&masked, &base, &switches).expect("unmodified");
        assert_eq!(before[1], Array2::<f64>::ones((3, 1)));
        for gain in [0.0, 0.5, 1.0, -2.0] {
            let (trace, chosen) = super::execute_at(&masked, &base, &switches, &[gain]).expect("intervened");
            let expected = array![[2.0], [3.0], [4.0]].mapv(|x: f64| if (gain * x).abs() > 1.0 { 1.0 } else { 0.0 });
            assert_eq!(chosen[1], expected);
            let replay = masked.program.execute_at(&masked.family(&base, &chosen), false, &[gain]).expect("replay");
            assert_eq!(trace.values, replay.values);
            if gain == 1.0 { assert_eq!(trace.values, clean.values); }
            if gain == 0.5 {
                let stale = masked.program.execute_at(&masked.family(&base, &before), false, &[gain]).expect("stale masks");
                assert_ne!(trace.values[masked.program.output], stale.values[masked.program.output]);
            }
        }
        for invalid in [vec![], vec![1.0, 1.0], vec![f64::NAN], vec![f64::INFINITY]] {
            assert!(super::execute_at(&masked, &base, &switches, &invalid).is_err());
        }
    }

    #[test]
    fn executor_rejects_masks_read_before_decisions_and_aliased_slots() {
        let (masked, base) = fixture(false);
        let family = masked.family(&base, &vec![Array2::<f64>::ones((base.rows, 1)); 2]);
        let mask = masked.z[0] - 1;
        let mut early = masked.program.clone();
        if let Node::Affine { terms, .. } = &mut early.nodes[masked.z[0]] {
            terms[0].0 = mask;
        }
        let mut called = false;
        assert!(early.execute_with_gates(&family, &[(masked.z[0], mask)], |_, _| {
            called = true;
            Ok(Array2::<f64>::ones((base.rows, 1)))
        }).is_err());
        assert!(!called);
        let mut aliased = masked.program.clone();
        aliased.nodes.push(Node::Raw { slot: masked.slots[0] });
        assert!(aliased.execute_with_gates(&family, &[(masked.z[0], mask)], |_, _| {
            called = true;
            Ok(Array2::<f64>::ones((base.rows, 1)))
        }).is_err());
        assert!(!called);
    }

}

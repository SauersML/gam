//! Fixed interventions lowered into ordinary differentiable IR, not opaque hooks.
//! Node controls scale one ORIGINAL root invocation. Global controls scale every
//! invocation of one ORIGINAL stored operator. Instrumentation is fixed, not learned.
use crate::{
    artifact_device::mapped_inlined,
    operator_program::{
        FamilyInputs, Interface, Node, Operator, OperatorBody, OperatorProgram, Slot, SlotValues,
        remap_node,
    },
};
use ndarray::Array2;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Control {
    NodeScale { node: usize },
    GlobalOperatorScale { operator: usize },
}
#[derive(Clone, Debug)]
pub enum ControlValue {
    NodeMask(Array2<f64>),
    GlobalScale(f64),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlSlot {
    pub control: usize,
    pub slot: usize,
    pub width: usize,
}
pub struct Compiled {
    pub program: OperatorProgram,
    /// Each original root maps to its final, controlled value (including Call barriers).
    pub root_mapping: Vec<usize>,
    pub control_slots: Vec<ControlSlot>,
    controls: Vec<Control>,
    source: OperatorProgram,
}
impl Compiled {
    /// Append only prescribed control values. A global scale is ONE scalar,
    /// replicated consistently into every corresponding row/coordinate slot.
    pub fn family(
        &self,
        base: &FamilyInputs,
        values: &[ControlValue],
    ) -> Result<FamilyInputs, String> {
        if values.len() != self.controls.len()
            || base.slots.len() != self.source.declarations.slots.len()
        {
            return Err("control count or base slot count mismatch".into());
        }
        if base.rows == 0 {
            return Err("positive family rows required".into());
        }
        for (decl, value) in self.source.declarations.slots.iter().zip(&base.slots) {
            match (decl, value) {
                (Slot::Raw { width }, SlotValues::Raw(x))
                    if x.dim() == (base.rows, *width) && x.iter().all(|v| v.is_finite()) => {}
                (Slot::Token { .. }, SlotValues::Tokens(x)) if x.len() == base.rows => {}
                _ => return Err("base slot shape/kind/nonfinite mismatch".into()),
            }
        }
        if base
            .layout
            .as_ref()
            .is_some_and(|l| l.sequence.len() != base.rows || l.position.len() != base.rows)
        {
            return Err("base layout row mismatch".into());
        }
        for (control, value) in self.controls.iter().zip(values) {
            match (control,value) {
                (Control::NodeScale{node},ControlValue::NodeMask(x)) if x.dim()==(base.rows,self.source.node_interface(*node).map_err(|e|e.to_string())?.width()) && x.iter().all(|v|v.is_finite())=>{},
                (Control::GlobalOperatorScale{..},ControlValue::GlobalScale(x)) if x.is_finite()=>{},
                _=>return Err("node control needs a finite row/coordinate mask; global control needs one finite scalar".into()),
            }
        }
        let mut out = base.clone();
        for binding in &self.control_slots {
            if out.slots.len() != binding.slot {
                return Err("control slot declaration mismatch".into());
            }
            let mask = match &values[binding.control] {
                ControlValue::NodeMask(x) => x.clone(),
                ControlValue::GlobalScale(x) => Array2::from_elem((base.rows, binding.width), *x),
            };
            out.slots.push(SlotValues::Raw(mask));
        }
        Ok(out)
    }
    /// Export fitted original numerical operators into the ORIGINAL structured
    /// source. Root/rule syntax and all appended instrumentation must be unchanged.
    pub fn restore(&self, fitted: &OperatorProgram) -> Result<OperatorProgram, String> {
        if fitted.declarations != self.program.declarations
            || fitted.bases != self.program.bases
            || fitted.rules != self.program.rules
            || fitted.nodes != self.program.nodes
            || fitted.output != self.program.output
            || fitted.operators.len() != self.program.operators.len()
        {
            return Err("fitted graph differs from controlled lowering".into());
        }
        let count = self.source.operators.len();
        for (index, (expected, actual)) in self
            .program
            .operators
            .iter()
            .zip(&fitted.operators)
            .enumerate()
        {
            if expected.rows != actual.rows || expected.cols != actual.cols {
                return Err("fitted operator interface mismatch".into());
            }
            if index >= count {
                if expected.as_ref() != actual.as_ref()
                    || !same_numeric_bits(&expected.body, &actual.body)
                {
                    return Err("fixed instrumentation was changed".into());
                }
            } else {
                match (&expected.body, &actual.body) {
                    (
                        OperatorBody::Dense {
                            values: a,
                            present: ap,
                            ..
                        },
                        OperatorBody::Dense {
                            values: b,
                            present: bp,
                            ..
                        },
                    ) if a.dim() == b.dim() && ap == bp && b.iter().all(|x| x.is_finite()) => {}
                    _ if expected.as_ref() == actual.as_ref()
                        && same_numeric_bits(&expected.body, &actual.body) =>
                    {
                        continue;
                    }
                    _ => return Err("original operator kind/mask/nonfinite mismatch".into()),
                }
            }
        }
        let mut restored = self.source.clone();
        restored.operators = fitted.operators[..count].to_vec();
        restored.interfaces().map_err(|e| e.to_string())?;
        Ok(restored)
    }
}
fn same_numeric_bits(a: &OperatorBody, b: &OperatorBody) -> bool {
    match (a, b) {
        (OperatorBody::Identity, OperatorBody::Identity) => true,
        (OperatorBody::Dense { values: a, .. }, OperatorBody::Dense { values: b, .. }) => {
            a.dim() == b.dim() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
        }
        (OperatorBody::Diagonal { values: a, .. }, OperatorBody::Diagonal { values: b, .. }) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
        }
        (
            OperatorBody::LowRank {
                left: a, right: ar, ..
            },
            OperatorBody::LowRank {
                left: b, right: br, ..
            },
        ) => {
            a.dim() == b.dim()
                && ar.dim() == br.dim()
                && a.iter()
                    .zip(b)
                    .chain(ar.iter().zip(br))
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        }
        _ => false,
    }
}
struct Lower {
    p: OperatorProgram,
    masks: BTreeMap<(usize, usize), usize>,
    slots: Vec<ControlSlot>,
    identities: Vec<(Interface, usize)>,
}
impl Lower {
    fn mask(&mut self, control: usize, width: usize) -> usize {
        if let Some(&node) = self.masks.get(&(control, width)) {
            return node;
        }
        let slot = self.p.declarations.slots.len();
        self.p.declarations.slots.push(Slot::Raw { width });
        let node = self.push(Node::Raw { slot });
        self.masks.insert((control, width), node);
        self.slots.push(ControlSlot {
            control,
            slot,
            width,
        });
        node
    }
    fn push(&mut self, node: Node) -> usize {
        let n = self.p.nodes.len();
        self.p.nodes.push(node);
        n
    }
    fn scale(&mut self, node: usize, control: usize, width: usize) -> usize {
        let mask = self.mask(control, width);
        self.push(Node::Hadamard {
            left: node,
            right: mask,
        })
    }
    fn identity(&mut self, interface: &Interface) -> usize {
        if let Some((_, op)) = self.identities.iter().find(|(i, _)| i == interface) {
            return *op;
        }
        let op = self.p.operators.len();
        self.p.operators.push(Arc::new(Operator::identity(
            "fixed intervention sum",
            interface.clone(),
        )));
        self.identities.push((interface.clone(), op));
        op
    }
}
/// Lower every requested control. No Mix/patch/AddMap or arbitrary callback is
/// accepted. Global operator indices remain unchanged through Call expansion.
pub fn compile(source: &OperatorProgram, controls: &[Control]) -> Result<Compiled, String> {
    source.interfaces().map_err(|e| e.to_string())?;
    let mut node_controls = BTreeMap::new();
    let mut operator_controls = BTreeMap::new();
    for (id, control) in controls.iter().enumerate() {
        match control {
            Control::NodeScale { node } => {
                if *node >= source.nodes.len() || node_controls.insert(*node, id).is_some() {
                    return Err("node control out of range or duplicate".into());
                }
            }
            Control::GlobalOperatorScale { operator } => {
                if *operator >= source.operators.len()
                    || operator_controls.insert(*operator, id).is_some()
                {
                    return Err("operator control out of range or duplicate".into());
                }
            }
        }
    }
    let (flat, roots) = mapped_inlined(source)?;
    let interfaces = flat.interfaces().map_err(|e| e.to_string())?;
    let root_controls: BTreeMap<_, _> = node_controls
        .into_iter()
        .map(|(node, id)| (roots[node], id))
        .collect();
    let mut l = Lower {
        p: flat.clone(),
        masks: BTreeMap::new(),
        slots: vec![],
        identities: vec![],
    };
    l.p.nodes.clear();
    let mut map = vec![usize::MAX; flat.nodes.len()];
    let ops: Vec<_> = (0..flat.operators.len()).collect();
    let bases: Vec<_> = (0..flat.bases.len()).collect();
    let mut used = BTreeSet::new();
    for (index, original) in flat.nodes.iter().enumerate() {
        let mut node = original.clone();
        remap_node(&mut node, &map, &ops, &bases, &[]);
        let value = match node {
            Node::Affine { terms, bias }
                if terms
                    .iter()
                    .any(|(_, op)| operator_controls.contains_key(op))
                    || bias.is_some_and(|op| operator_controls.contains_key(&op)) =>
            {
                let mut out = vec![];
                for (input, operator) in terms {
                    if let Some(&control) = operator_controls.get(&operator) {
                        used.insert(operator);
                        let n = l.push(Node::Affine {
                            terms: vec![(input, operator)],
                            bias: None,
                        });
                        let n = l.scale(n, control, interfaces[index].width());
                        let identity = l.identity(&interfaces[index]);
                        out.push((n, identity));
                    } else {
                        out.push((input, operator));
                    }
                }
                let remaining_bias = if let Some(operator) = bias {
                    if let Some(&control) = operator_controls.get(&operator) {
                        used.insert(operator);
                        let n = l.push(Node::Constant { operator });
                        let n = l.scale(n, control, interfaces[index].width());
                        let identity = l.identity(&interfaces[index]);
                        out.push((n, identity));
                        None
                    } else {
                        Some(operator)
                    }
                } else {
                    None
                };
                l.push(Node::Affine {
                    terms: out,
                    bias: remaining_bias,
                })
            }
            Node::Constant { operator } if operator_controls.contains_key(&operator) => {
                used.insert(operator);
                let n = l.push(Node::Constant { operator });
                l.scale(n, operator_controls[&operator], interfaces[index].width())
            }
            Node::Transposed { input, operator } if operator_controls.contains_key(&operator) => {
                used.insert(operator);
                let n = l.push(Node::Transposed { input, operator });
                l.scale(n, operator_controls[&operator], interfaces[index].width())
            }
            other => {
                if other
                    .operators()
                    .iter()
                    .any(|op| operator_controls.contains_key(op))
                {
                    return Err("global control has unsupported operator use".into());
                }
                l.push(other)
            }
        };
        map[index] = if let Some(&control) = root_controls.get(&index) {
            l.scale(value, control, interfaces[index].width())
        } else {
            value
        };
    }
    if operator_controls.keys().any(|op| !used.contains(op)) {
        return Err("global control operator has no executable use".into());
    }
    l.p.output = map[flat.output];
    l.p.interfaces().map_err(|e| e.to_string())?;
    Ok(Compiled {
        program: l.p,
        root_mapping: roots.into_iter().map(|n| map[n]).collect(),
        control_slots: l.slots,
        controls: controls.to_vec(),
        source: source.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_program::{Declarations, Rule, exact_precision};
    use ndarray::array;
    fn dense(rows: Interface, cols: Interface, values: Array2<f64>) -> Arc<Operator> {
        let precision = exact_precision(values.iter().copied()).expect("finite test values");
        Arc::new(
            Operator::dense(
                "original",
                rows,
                cols,
                values,
                precision,
                Default::default(),
            )
            .expect("valid dense"),
        )
    }
    fn source() -> OperatorProgram {
        let i = Interface::native(2).expect("native width");
        OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }],
                parameters: 0,
            },
            bases: vec![],
            operators: vec![dense(i.clone(), i.clone(), array![[1., 2.], [-1., 0.5]])],
            rules: vec![Rule {
                name: "shared".into(),
                inputs: vec![i],
                nodes: vec![
                    Node::Param { index: 0 },
                    Node::Affine {
                        terms: vec![(0, 0)],
                        bias: None,
                    },
                ],
                output: 1,
            }],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine {
                    terms: vec![(0, 0)],
                    bias: None,
                },
                Node::Call {
                    rule: 0,
                    arguments: vec![0],
                },
                Node::Affine {
                    terms: vec![(1, 0), (2, 0)],
                    bias: None,
                },
            ],
            output: 3,
        }
    }
    fn family() -> FamilyInputs {
        FamilyInputs {
            rows: 2,
            slots: vec![SlotValues::Raw(array![[2., -1.], [0.5, 3.]])],
            layout: None,
        }
    }
    fn output(p: &OperatorProgram, f: &FamilyInputs) -> Array2<f64> {
        p.execute(f, false).expect("valid execution").values[p.output].clone()
    }
    #[test]
    fn shared_global_control_matches_actual_stored_weight_edit() {
        let p = source();
        let c = compile(&p, &[Control::GlobalOperatorScale { operator: 0 }]).expect("lower");
        for scale in [0., 1., -0.25, 2.] {
            let f = c
                .family(&family(), &[ControlValue::GlobalScale(scale)])
                .expect("scalar family");
            let mut actual = p.clone();
            let OperatorBody::Dense { values, .. } = &p.operators[0].body else {
                panic!("dense")
            };
            actual.operators[0] = dense(
                p.operators[0].rows.clone(),
                p.operators[0].cols.clone(),
                values * scale,
            );
            let a = output(&actual, &family());
            let b = output(&c.program, &f);
            let message = c.program.encode().expect("standalone ordinary encoding");
            let decoded = OperatorProgram::decode(&message, &c.program.declarations)
                .expect("standalone decode");
            assert_eq!(output(&decoded, &f), b);
            assert!(a.iter().zip(b.iter()).all(|(a, b)| (a - b).abs() < 1e-12));
        }
        assert!(
            c.family(&family(), &[ControlValue::NodeMask(Array2::ones((2, 2)))])
                .is_err()
        );
        assert_eq!(p.rules.len(), 1);
    }
    #[test]
    fn partial_mask_targets_original_call_only_and_restores_original_graph() {
        let p = source();
        let c = compile(&p, &[Control::NodeScale { node: 2 }]).expect("lower");
        let mask = array![[0., 1.], [0.5, -1.]];
        let f = c
            .family(&family(), &[ControlValue::NodeMask(mask.clone())])
            .expect("mask");
        let before = p.execute(&family(), false).expect("source");
        let after = c.program.execute(&f, false).expect("lowered");
        assert_eq!(after.values[c.root_mapping[1]], before.values[1]);
        assert_eq!(after.values[c.root_mapping[2]], &before.values[2] * &mask);
        let restored = c.restore(&c.program).expect("restore");
        assert_eq!(restored, p);
        let mut changed = c.program.clone();
        changed.output = 0;
        assert!(c.restore(&changed).is_err());
        assert!(
            c.family(&family(), &[ControlValue::NodeMask(Array2::ones((1, 2)))])
                .is_err()
        );
    }
    #[test]
    fn column_operator_bias_constant_and_transpose_share_one_scalar() {
        let i = Interface::native(2).expect("width");
        let p = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 2 }],
                parameters: 0,
            },
            bases: vec![],
            rules: vec![],
            operators: vec![
                dense(i.clone(), Interface::constant(), array![[2.], [-3.]]),
                Arc::new(Operator::identity("identity", i)),
            ],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Constant { operator: 0 },
                Node::Affine {
                    terms: vec![(0, 1)],
                    bias: Some(0),
                },
                Node::Transposed {
                    input: 0,
                    operator: 0,
                },
                Node::Concat {
                    parts: vec![1, 2, 3],
                },
            ],
            output: 4,
        };
        let c = compile(&p, &[Control::GlobalOperatorScale { operator: 0 }]).expect("lower");
        let f = c
            .family(&family(), &[ControlValue::GlobalScale(-0.5)])
            .expect("family");
        assert_eq!(c.control_slots.len(), 2); // Widths two and one, the SAME scalar.
        let mut actual = p.clone();
        actual.operators[0] = dense(
            p.operators[0].rows.clone(),
            Interface::constant(),
            array![[-1.], [1.5]],
        );
        assert_eq!(output(&c.program, &f), output(&actual, &family()));
    }
    #[test]
    fn zero_shared_weight_control_blocks_all_operator_gradients() {
        use crate::device_program::DeviceProgram;
        use gam_gpu::tensor::{Arithmetic, Device};
        let p = source();
        let c = compile(&p, &[Control::GlobalOperatorScale { operator: 0 }]).expect("lower");
        let f = c
            .family(&family(), &[ControlValue::GlobalScale(0.)])
            .expect("family");
        let device = Device::host();
        let graph = DeviceProgram::compile_values(&device, &c.program).expect("compile values");
        let trace = graph.forward(&f).expect("forward");
        let seeds = BTreeMap::from([(
            c.program.output,
            device.upload(Array2::ones((2, 2)).view()).expect("seed"),
        )]);
        let (_, gradients) = graph
            .vjp_values_dense(&trace, seeds, &[], &[0], Arithmetic::F64)
            .expect("VJP");
        assert!(
            device
                .download(&gradients[&0])
                .expect("gradient")
                .iter()
                .all(|x| *x == 0.)
        );
    }
}

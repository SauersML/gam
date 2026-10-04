//! Controlled proposals for reusable overcomplete numerical geometry.
//! The GELU architecture is supplied, not discovered. Complete artifact fidelity/cost
//! and frozen-body transfer are required before any stronger interpretation.
use crate::{operator_program::{Declarations, Interface, Law, Node, Operator, OperatorProgram, Rule, Slot, exact_precision}, run_check::LayerNodes};
use ndarray::Array2;
use std::sync::Arc;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm { Learned, FrozenNative, FrozenRandom, Untied }
/// One source pool: individual exported functions retain shared body operator Arcs.
pub struct Proposal {
    pub program: OperatorProgram,
    pub trainable: Vec<usize>,
    pub body_operators: Vec<usize>,
    pub outputs: Vec<usize>,
    pub uses: Vec<usize>,
}
fn dense(name: String, values: Array2<f64>) -> Result<Arc<Operator>, String> {
    let rows = Interface::native(values.nrows()).map_err(|e| e.to_string())?;
    let cols = if values.ncols() == 1 { Interface::constant() } else { Interface::native(values.ncols()).map_err(|e| e.to_string())? };
    let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
    Ok(Arc::new(Operator::dense(name, rows, cols, values, precision, Default::default()).map_err(|e| e.to_string())?))
}
fn affine_source(native: &OperatorProgram, node: usize, read: usize) -> Result<(Array2<f64>, Array2<f64>), String> {
    let Some(Node::Affine { terms, bias }) = native.nodes.get(node) else { return Err("native reader/writer must be affine".into()); };
    if terms.len() != 1 || terms[0].0 != read { return Err("native reader/writer requires exactly the declared source".into()); }
    let matrix = native.operators.get(terms[0].1).ok_or("native operator absent")?.matrix();
    let offset = match bias { Some(index) => native.operators.get(*index).ok_or("native bias absent")?.matrix(), None => Array2::zeros((matrix.nrows(), 1)) };
    if offset.dim() != (matrix.nrows(), 1) { return Err("native offset shape mismatch".into()); }
    Ok((matrix, offset))
}
/// Uses are explicit native layer IDs. Native initialization comes only from uses[0].
/// Untied direct readers start at the SAME effective reader as shared arms, not at
/// each use's exact native reader. All writers start from their native writer.
pub fn build(native: &OperatorProgram, layers: &[LayerNodes], uses: &[usize], arm: Arm, seed: u64) -> Result<Proposal, String> {
    if uses.is_empty() { return Err("at least one declared use required".into()); }
    if uses.iter().enumerate().any(|(i,u)| uses[..i].contains(u)) { return Err("duplicate native use".into()); }
    let first = layers.get(uses[0]).ok_or("native initializer use outside map")?;
    let (mut body, offset) = affine_source(native, first.pre, first.normed)?;
    let (u, d) = body.dim();
    if u <= d { return Err("declared overcomplete geometry requires hidden width > input width".into()); }
    let laws = match native.nodes.get(first.active) { Some(Node::Pointwise { input, laws }) if *input == first.pre => laws.clone(), _ => return Err("native activation is not direct pointwise reader output".into()) };
    if laws != vec![Law::GeluTanh] && laws != vec![Law::Gelu] { return Err("pilot declares native GELU or GELU-tanh only".into()); }
    if arm == Arm::FrozenRandom {
        let mut state = seed;
        let scale = 1.0 / (d as f64).sqrt();
        for value in &mut body {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state; z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9); z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb); z ^= z >> 31;
            *value = (2.0 * ((z >> 11) as f64 / 9007199254740992.0) - 1.0) * scale;
        }
    }
    let mut program = OperatorProgram { declarations: Declarations { domains: vec![], slots: uses.iter().map(|_|Slot::Raw { width:d }).collect(), parameters:0 }, bases:vec![], operators:vec![], rules:vec![], nodes:vec![], output:0 };
    let mut trainable = vec![]; let mut body_operators = vec![]; let mut outputs = vec![];
    if arm != Arm::Untied {
        program.operators.push(dense("shared overcomplete reader".into(), body.clone())?);
        program.operators.push(dense("shared body offset".into(), offset.clone())?);
        body_operators = vec![0,1];
        if arm == Arm::Learned { trainable.extend([0,1]); }
        program.rules.push(Rule { name:"shared supplied GELU architecture with numerical geometry".into(), inputs:vec![Interface::native(d).map_err(|e|e.to_string())?], nodes:vec![Node::Param { index:0 }, Node::Affine { terms:vec![(0,0)], bias:Some(1) }, Node::Pointwise { input:1, laws:laws.clone() }], output:2 });
    }
    for (slot, &use_id) in uses.iter().enumerate() {
        let layer = layers.get(use_id).ok_or("declared use outside native map")?;
        let (native_reader, _) = affine_source(native, layer.pre, layer.normed)?;
        let (writer, writer_offset) = affine_source(native, layer.mlp, layer.active)?;
        if native_reader.dim() != (u,d) || writer.dim() != (d,u) { return Err("all declared uses must share exact full interfaces".into()); }
        if native.nodes[layer.active] != (Node::Pointwise { input:layer.pre, laws:laws.clone() }) { return Err("use activation law differs from declared body".into()); }
        let raw = program.nodes.len(); program.nodes.push(Node::Raw { slot });
        let start = program.operators.len();
        let active = if arm == Arm::Untied {
            program.operators.push(dense(format!("use{use_id} independent reader"),body.clone())?);
            program.operators.push(dense(format!("use{use_id} independent offset"),offset.clone())?);
            let pre = program.nodes.len(); program.nodes.push(Node::Affine { terms:vec![(raw,start)],bias:Some(start+1) });
            let active = program.nodes.len(); program.nodes.push(Node::Pointwise { input:pre,laws:laws.clone() }); active
        } else {
            let identity = Array2::from_shape_fn((d,d), |(r,c)| if r==c {1.0} else {0.0});
            program.operators.push(dense(format!("use{use_id} input binding"),identity)?);
            program.operators.push(dense(format!("use{use_id} input offset"),Array2::zeros((d,1)))?);
            let read = program.nodes.len(); program.nodes.push(Node::Affine { terms:vec![(raw,start)],bias:Some(start+1) });
            let active = program.nodes.len(); program.nodes.push(Node::Call { rule:0,arguments:vec![read] }); active
        };
        program.operators.push(dense(format!("use{use_id} output binding"),writer)?);
        program.operators.push(dense(format!("use{use_id} output offset"),writer_offset)?);
        trainable.extend(start..start+4);
        let write = program.nodes.len(); program.nodes.push(Node::Affine { terms:vec![(active,start+2)],bias:Some(start+3) }); outputs.push(write);
    }
    program.output = program.nodes.len(); program.nodes.push(Node::Concat { parts:outputs.clone() });
    program.interfaces().map_err(|e|e.to_string())?;
    Ok(Proposal { program,trainable,body_operators,outputs,uses:uses.to_vec() })
}
/// Export one function while retaining source-pool Arcs and rule/operator indices.
/// Unused operators are removed/priced by the native graft's standard compactor.
pub fn function(proposal: &Proposal, slot: usize) -> Result<OperatorProgram, String> {
    let output = *proposal.outputs.get(slot).ok_or("function use absent")?;
    let width = match proposal.program.declarations.slots.get(slot) {Some(Slot::Raw {width})=>*width,_=>return Err("function Raw declaration absent".into())};
    let mut live = vec![false; proposal.program.nodes.len()];
    let mut stack = vec![output];
    while let Some(node) = stack.pop() {
        if live[node] { continue; }
        live[node] = true;
        stack.extend(proposal.program.nodes[node].arguments());
    }
    let mut mapping = vec![usize::MAX; live.len()];
    let mut nodes = vec![];
    for (old, node) in proposal.program.nodes.iter().enumerate() {
        if live[old] { mapping[old] = nodes.len(); nodes.push(node.clone()); }
    }
    let ops: Vec<_> = (0..proposal.program.operators.len()).collect();
    let rules: Vec<_> = (0..proposal.program.rules.len()).collect();
    for node in &mut nodes {
        crate::operator_program::remap_node(node, &mapping, &ops, &[], &rules);
        if let Node::Raw {slot: raw_slot} = node {
            if *raw_slot != slot { return Err("function unexpectedly depends on another native use".into()); }
            *raw_slot = 0;
        }
    }
    let mut program = proposal.program.clone();
    program.declarations.slots = vec![Slot::Raw {width}];
    program.output = mapping[output];
    program.nodes = nodes;
    program.interfaces().map_err(|e|e.to_string())?;
    Ok(program)
}

#[cfg(test)]
#[path = "shared_geometry_pilot_tests.rs"]
mod tests;

//! Explicit paid control transport. The commutation proof is real arithmetic only;
//! actual decoded floating-point counterfactuals remain subject to Local/Run measurement.
use crate::{artifact::Artifact, operator_program::{Node, OperatorProgram}};
use std::collections::BTreeSet;
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniformScaleBinding {
    pub native_source: usize,
    pub native_write: usize,
    pub write: usize,
    /// Full native source width, not the explanatory write width.
    pub width: usize,
}
/// Validate serialized shape and unambiguous lineage without a native graph.
/// This cannot establish the native linear commutation proof; use `validate` for that.
pub fn validate_shape(artifact: &Artifact) -> Result<(), String> {
    if artifact.controls.windows(2).any(|pair|pair[0].native_source>=pair[1].native_source) {return Err("controls must ascend strictly in native source".into());}
    if artifact.controls.is_empty() {return Ok(());}
    artifact.program.interfaces().map_err(|e|e.to_string())?;
    let mut sources=BTreeSet::new();let mut native_writes=BTreeSet::new();let mut writes=BTreeSet::new();
    for control in &artifact.controls {
        if control.width==0 || control.width.checked_add(1).is_none() || control.native_source>=artifact.native_nodes || control.native_write>=artifact.native_nodes || control.write>=artifact.program.nodes.len() || control.native_source==control.native_write {
            return Err("invalid uniform-scale control indices/width".into());
        }
        if !sources.insert(control.native_source) || !native_writes.insert(control.native_write) || !writes.insert(control.write) {return Err("duplicate/conflicting uniform-scale controls".into());}
        if artifact.places.iter().any(|(native,_)|*native==control.native_source) {return Err("uniform-scale source already held".into());}
        if artifact.blocks.iter().any(|b|b.native_reads.contains(&control.native_source)) {return Err("uniform-scale source is an exposed block input".into());}
        if artifact.places.iter().filter(|(native,_)|*native==control.native_write).count()!=1 || artifact.places.iter().filter(|(native,node)|*native==control.native_write && *node==control.write).count()!=1 || artifact.places.iter().any(|(native,node)|*node==control.write && *native!=control.native_write) {return Err("uniform-scale write is not held uniquely".into());}
        if artifact.blocks.iter().filter(|b|b.native_write==control.native_write && b.write==control.write).count()!=1 || artifact.blocks.iter().filter(|b|b.write==control.write).count()!=1 {return Err("uniform-scale write needs one matching declared block".into());}
    }
    if sources.iter().any(|source|native_writes.contains(source)) {return Err("overlapping uniform-scale source/write bindings".into());}
    Ok(())
}
/// Check a direct pointwise linear cut, refusing residual additions and bypasses.
pub fn validate(artifact:&Artifact,native:&OperatorProgram)->Result<(),String> {
    validate_shape(artifact)?;
    if artifact.controls.is_empty() {return Ok(());}
    if artifact.native_nodes!=native.nodes.len() {return Err("native control graph size mismatch".into());}
    let native_interfaces=native.interfaces().map_err(|e|e.to_string())?;
    let interfaces=artifact.program.interfaces().map_err(|e|e.to_string())?;
    for control in &artifact.controls {
        if native.output==control.native_source || native_interfaces[control.native_source].width()!=control.width {return Err("native control source width/output mismatch".into());}
        match &native.nodes[control.native_write] {
            Node::Affine{terms,bias:None} if !terms.is_empty() && terms.iter().all(|(source,_)|*source==control.native_source)=>{},
            Node::Transposed{input,..} if *input==control.native_source=>{},
            _=>return Err("uniform-scale write is not direct bias-free pointwise linear from source alone".into()),
        }
        if native.nodes.iter().enumerate().any(|(index,node)|index!=control.native_write && node.arguments().contains(&control.native_source)) {return Err("native control source has a bypass consumer".into());}
        if native_interfaces[control.native_write]!=interfaces[control.write] {return Err("native control write interface mismatch".into());}
    }
    Ok(())
}
#[cfg(test)]
#[path="native_control_tests.rs"]
mod tests;

//! Explicit paid control transport. The commutation proof is real arithmetic only;
//! actual decoded floating-point counterfactuals remain subject to Local/Run measurement.
use crate::{acceptance::{Change, Edit}, artifact::Artifact, operator_program::{Node, OperatorProgram}};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniformScaleBinding {
    pub native_source: usize,
    pub native_write: usize,
    pub write: usize,
    /// Full native source width, not the explanatory write width.
    pub width: usize,
}
#[derive(Clone, Debug, PartialEq)]
pub struct MappedEdits {
    pub edits: BTreeMap<usize, Vec<Edit>>,
    pub unheld: usize,
}
/// Validate serialized shape and unambiguous lineage without a native graph.
/// This cannot establish the native linear commutation proof; use `validate` for that.
pub fn validate_shape(artifact: &Artifact) -> Result<(), String> {
    if artifact.controls.windows(2).any(|pair|pair[0].native_source>=pair[1].native_source) {return Err("controls must ascend strictly in native source".into());}
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
        artifact.program.node_interface(control.write).map_err(|e|e.to_string())?;
    }
    if sources.iter().any(|source|native_writes.contains(source)) {return Err("overlapping uniform-scale source/write bindings".into());}
    Ok(())
}
/// Check a direct pointwise linear cut, refusing residual additions and bypasses.
pub fn validate(artifact:&Artifact,native:&OperatorProgram)->Result<(),String> {
    validate_shape(artifact)?;
    if artifact.controls.is_empty() {return Ok(());}
    if artifact.native_nodes!=native.nodes.len() {return Err("native control graph size mismatch".into());}
    for control in &artifact.controls {
        if native.output==control.native_source || native.node_interface(control.native_source).map_err(|e|e.to_string())?.width()!=control.width {return Err("native control source width/output mismatch".into());}
        match &native.nodes[control.native_write] {
            Node::Affine{terms,bias:None} if !terms.is_empty() && terms.iter().all(|(source,_)|*source==control.native_source)=>{},
            Node::Transposed{input,..} if *input==control.native_source=>{},
            _=>return Err("uniform-scale write is not direct bias-free pointwise linear from source alone".into()),
        }
        if native.nodes.iter().enumerate().any(|(index,node)|index!=control.native_write && node.arguments().contains(&control.native_source)) {return Err("native control source has a bypass consumer".into());}
        if native.node_interface(control.native_write).map_err(|e|e.to_string())?!=artifact.program.node_interface(control.write).map_err(|e|e.to_string())? {return Err("native control write interface mismatch".into());}
    }
    Ok(())
}
/// Source controls precede every direct write edit, independently of action-list order.
/// Repeated source edits retain their order. Partial-column/additive absent-source edits
/// remain explicitly unheld; no matrix or hidden native activation is executed here.
pub fn map_edits(artifact:&Artifact,edits:&[Edit])->Result<MappedEdits,String> {
    validate_shape(artifact)?;
    let mut result=MappedEdits{edits:BTreeMap::new(),unheld:0};let mut direct:BTreeMap<usize,Vec<Edit>>=BTreeMap::new();
    for edit in edits {
        let value=match edit.change {Change::Scale(value)|Change::Add(value)=>value};
        if !value.is_finite() || edit.node>=artifact.native_nodes || edit.columns.is_empty() {return Err("invalid native control edit".into());}
        if let Some(write)=artifact.place(edit.node) {
            let width=artifact.program.node_interface(write).map_err(|e|e.to_string())?.width();
            if edit.columns.end>width {return Err("held edit exceeds native interface".into());}
            let mut mapped=edit.clone();mapped.node=write;direct.entry(write).or_default().push(mapped);
        } else if let Some(control)=artifact.controls.iter().find(|c|c.native_source==edit.node) {
            if edit.columns.end>control.width {return Err("control edit exceeds native source interface".into());}
            if matches!(edit.change,Change::Scale(_)) && edit.columns==(0..control.width) {
                let mut mapped=edit.clone();mapped.node=control.write;mapped.columns=0..artifact.program.node_interface(control.write).map_err(|e|e.to_string())?.width();
                result.edits.entry(control.write).or_default().push(mapped);
            } else {result.unheld+=1;}
        } else {result.unheld+=1;}
    }
    for (write,mut actions) in direct {result.edits.entry(write).or_default().append(&mut actions);}
    Ok(result)
}
#[cfg(test)]
#[path="native_control_tests.rs"]
mod tests;

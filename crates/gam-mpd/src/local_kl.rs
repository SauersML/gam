//! Optional isolated downstream-KL diagnostic; it changes no acceptance threshold.
//!
//! Each replacement receives its declared native parent states. Only its native
//! write is patched, then the native downstream graph runs. Other replacements
//! stay native. This differs from autonomous artifact execution in RunCheck.
//! Error fields cover the KL comparison computation alone, not neural-forward
//! arithmetic or untested inputs, and therefore do not certify end-to-end fidelity.

use crate::acceptance::{kl_logits, units};
use crate::artifact::Artifact;
use crate::operator_program::{FamilyInputs, OperatorProgram};
use gam_linalg::roundoff::accumulation_growth;

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct IsolatedBlockKl {
    pub name: String,
    pub native_write: usize,
    pub tested_rows: usize,
    pub domain: String,
    pub mean_kl: f64,
    pub max_kl: f64,
    pub max_row: usize,
    pub mean_comparison_error: f64,
    pub max_comparison_error: f64,
}

/// `decoded` must be the decoded artifact being diagnosed. This CPU diagnostic
/// runs whole family units so causal prefixes and exception contexts are retained.
/// It reads the direct replacement write, avoiding cancellation from native+delta.
pub fn isolated_downstream_kl(native: &OperatorProgram, decoded: &Artifact, family: &FamilyInputs, batch_rows: usize) -> Result<Vec<IsolatedBlockKl>, String> {
    decoded.validate_coverage(native)?;
    if family.rows == 0 {
        return Err("isolated KL needs a nonempty declared family".into());
    }
    let mut batches = Vec::new();
    let mut current = Vec::new();
    for unit in units(family) {
        if !current.is_empty() && current.len() + unit.len() > batch_rows.max(1) {
            batches.push(std::mem::take(&mut current));
        }
        current.extend(unit);
    }
    if !current.is_empty() {
        batches.push(current);
    }
    let mut results = Vec::with_capacity(decoded.blocks.len());
    for (index, binding) in decoded.blocks.iter().enumerate() {
        let (grafted, direct_write) = decoded.local_block_artifact(native, index)?;
        let (mut total, mut magnitude, mut error, mut max_kl, mut max_error, mut max_row) = (0.0, 0.0, 0.0, f64::NEG_INFINITY, 0.0f64, 0usize);
        for rows in &batches {
            let inputs = family.select(rows);
            let reference = native.execute(&inputs, false).map_err(|e| e.to_string())?;
            let replacement = grafted.execute(&inputs)?;
            let write = &replacement.values[direct_write];
            let patched = native
                .execute_edited(&inputs, |node, value, _| {
                    if node == binding.native_write {
                        if value.raw_dim() != write.raw_dim() {
                            return Err("isolated replacement write shape mismatch".into());
                        }
                        value.assign(write);
                    }
                    Ok(())
                })
                .map_err(|e| e.to_string())?;
            let original = &reference.values[native.output];
            let changed = &patched.values[native.output];
            if original.raw_dim() != changed.raw_dim() || original.nrows() != rows.len() {
                return Err("isolated logit output shape mismatch".into());
            }
            for (local, global) in rows.iter().enumerate() {
                let (kl, bound) = kl_logits(original.row(local), changed.row(local));
                if !kl.is_finite() || !bound.is_finite() || bound < 0.0 || kl < -bound {
                    return Err("invalid isolated KL comparison".into());
                }
                total += kl;
                magnitude += kl.abs();
                error += bound;
                if kl > max_kl {
                    max_kl = kl;
                    max_row = *global;
                }
                max_error = max_error.max(bound);
            }
        }
        let n = family.rows as f64;
        let mean_kl = total / n;
        let mean_comparison_error = ((error + accumulation_growth(family.rows + 4) * (magnitude + error)) / n).next_up();
        if !mean_kl.is_finite() || !mean_comparison_error.is_finite() {
            return Err("isolated KL aggregation overflow".into());
        }
        results.push(IsolatedBlockKl {
            name: binding.name.clone(),
            native_write: binding.native_write,
            tested_rows: family.rows,
            domain: format!("{} declared token rows; whole causal units; native parent states; one write patched; native downstream", family.rows),
            mean_kl,
            max_kl,
            max_row,
            mean_comparison_error,
            max_comparison_error: max_error,
        });
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::{Episode, FamilyRun, RunCheck};
    use crate::artifact::{Argument, Callee, Exception};
    use crate::operator_program::{Declarations, Interface, Node, Operator, Provenance, Rule, Slot, SlotValues, exact_precision};
    use ndarray::array;
    use std::sync::Arc;
    fn fixture() -> (OperatorProgram, FamilyInputs) {
        let v = Interface::native(1).unwrap();
        let c = Interface::native(2).unwrap();
        let head = array![[1.0], [-1.0]];
        let p = OperatorProgram {
            declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width: 1 }], parameters: 0 },
            bases: vec![],
            rules: vec![],
            operators: vec![
                Arc::new(Operator::identity("I", v.clone())),
                Arc::new(Operator::dense("head", c, v, head.clone(), exact_precision(head.iter().copied()).unwrap(), Provenance::default()).unwrap()),
            ],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine { terms: vec![(0, 0)], bias: None },
                Node::Affine { terms: vec![(1, 0)], bias: None },
                Node::Affine { terms: vec![(2, 1)], bias: None },
            ],
            output: 3,
        };
        (p, FamilyInputs { rows: 2, slots: vec![SlotValues::Raw(array![[1.0], [-0.5]])], layout: None })
    }
    fn gain(a: &Artifact, native_read: usize, native_write: usize, value: f64) -> Artifact {
        let v = Interface::native(1).unwrap();
        let matrix = array![[value]];
        let op = a.program.operators.len();
        let rule = Rule {
            name: "gain".into(),
            inputs: vec![v.clone()],
            nodes: vec![Node::Param { index: 0 }, Node::Affine { terms: vec![(0, op)], bias: None }],
            output: 1,
        };
        a.replace_block(
            "gain",
            Callee::New(rule),
            vec![Argument::Native(native_read)],
            native_write,
            vec![Operator::dense("gain", v.clone(), v, matrix.clone(), exact_precision(matrix.iter().copied()).unwrap(), Provenance::default()).unwrap()],
        )
        .unwrap()
    }
    fn decoded(a: &Artifact) -> Artifact {
        Artifact::from_bytes(&a.to_bytes().unwrap(), &a.program.declarations).unwrap()
    }
    #[test]
    fn cancelling_layers_have_zero_run_but_positive_isolated_kl() {
        let (native, family) = fixture();
        let start = Artifact::native(&native).unwrap();
        let candidate = decoded(&gain(&gain(&start, 0, 1, 2.0), 1, 2, 0.5));
        let run = FamilyRun {
            model: &native,
            family: family.clone(),
            readouts: 1,
            episodes: vec![Episode { id: "clean".into(), group: "clean".into(), edits: vec![] }],
        };
        assert!(run.episodes(&candidate).unwrap()[0].kl.abs() < 1e-14);
        let isolated = isolated_downstream_kl(&native, &candidate, &family, 1).unwrap();
        assert_eq!(isolated.len(), 2);
        assert!(isolated.iter().all(|b| b.mean_kl > 0.01 && b.max_kl > 0.01));
    }
    #[test]
    fn native_copy_is_zero_and_rule_write_exception_is_not_dropped() {
        let (native, family) = fixture();
        let start = Artifact::native(&native).unwrap();
        let exact = decoded(&start.bind("native copy", &[0], 1).unwrap());
        assert_eq!(isolated_downstream_kl(&native, &exact, &family, 1).unwrap()[0].max_kl, 0.0);
        let mut changed = gain(&start, 0, 1, 1.0);
        changed.exceptions.push(Exception { context: vec![], node: changed.blocks[0].write, column: 0, value: 0.25 });
        let changed = decoded(&changed);
        let measured = isolated_downstream_kl(&native, &changed, &family, 1).unwrap();
        let expected = kl_logits(array![1.0, -1.0].view(), array![1.25, -1.25].view());
        assert!(measured[0].mean_kl > 0.0);
        assert!(measured[0].max_kl >= expected.0);
        let (graft, write) = changed.local_block_artifact(&native, 0).unwrap();
        assert_eq!(graft.execute(&family).unwrap().values[write][[0, 0]], 1.25);
    }
}

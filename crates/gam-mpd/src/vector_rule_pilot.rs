//! Explicit full-vector arithmetic proposal builders. These declarations are proposals,
//! not evidence that a reusable law has been discovered.
use crate::operator_program::{Declarations, Interface, Node, Operator, OperatorProgram, Rule, Slot, exact_precision};
use ndarray::Array2;
use std::sync::Arc;

/// A seeded two-vector product followed by a full-width paid writer and offset.
/// Width is both the argument width and nonlinear feature count; no low-rank writer.
/// All six returned operator indices are trainable. The seed is an explicit knob.
pub fn product(width: usize, seed: u64) -> Result<(OperatorProgram, Vec<usize>), String> {
    if width == 0 { return Err("product proposal requires positive width".into()); }
    width.checked_mul(width).ok_or("product matrix shape overflow")?;
    let interface = Interface::native(width).map_err(|e| e.to_string())?;
    let mut state = seed;
    let mut random = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        2.0 * ((z >> 11) as f64 / 9007199254740992.0) - 1.0
    };
    let mut operators = Vec::new();
    for (name, cols, amplitude) in [
        ("reader_a", interface.clone(), 1.0),
        ("reader_b", interface.clone(), 1.0),
        ("writer", interface.clone(), 0.1),
        ("offset", Interface::constant(), 0.0),
        ("reader_a_offset", Interface::constant(), 0.1),
        ("reader_b_offset", Interface::constant(), 0.1),
    ] {
        let scale = amplitude / (width as f64).sqrt();
        let values = Array2::from_shape_fn((width, cols.width()), |_| random() * scale);
        let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
        operators.push(Arc::new(Operator::dense(name, interface.clone(), cols, values, precision, Default::default()).map_err(|e| e.to_string())?));
    }
    let program = OperatorProgram {
        declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width }], parameters: 0 },
        bases: vec![], operators,
        rules: vec![Rule {
            name: "two full-vector elementwise product".into(),
            inputs: vec![interface.clone(), interface],
            nodes: vec![Node::Param { index: 0 }, Node::Param { index: 1 }, Node::Hadamard { left: 0, right: 1 }],
            output: 2,
        }],
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: Some(4) },
            Node::Affine { terms: vec![(0, 1)], bias: Some(5) },
            Node::Call { rule: 0, arguments: vec![1, 2] },
            Node::Affine { terms: vec![(3, 2)], bias: Some(3) },
        ], output: 4,
    };
    Ok((program, vec![0, 1, 2, 3, 4, 5]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::Artifact;
    #[test]
    fn vector_product_seeded_paid_roundtrip() {
        let (program, trainable) = product(3, 17).expect("valid full-vector product");
        let (same, _) = product(3, 17).expect("same seed");
        let (other, _) = product(3, 18).expect("different seed");
        assert_eq!(trainable, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(program.operators, same.operators);
        assert_ne!(program.operators, other.operators);
        assert_eq!(program.rules[0].inputs.iter().map(Interface::width).collect::<Vec<_>>(), vec![3, 3]);
        let source = Artifact::native(&program).expect("standalone artifact").f32_literals().expect("f32 literals");
        let bytes = source.to_bytes().expect("encode");
        let decoded = Artifact::from_bytes(&bytes, &program.declarations).expect("ordinary decode");
        assert_eq!(decoded.to_bytes().expect("canonical encode"), bytes);
    }
    #[test]
    fn vector_product_rejects_empty_interface() { assert!(product(0, 0).is_err()); }
}

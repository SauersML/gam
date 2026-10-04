//! Explicit matrix-valued rule bodies. This grammar contains arithmetic primitives,
//! not named mechanistic templates. A body is priced once; its source bindings are
//! priced by the artifact that calls it. Numerical resolution is not a fidelity
//! tolerance or a certified rank enclosure. All freely supplied coefficients are f32.
use super::codec::{
    BitReader, BitString, decode_fixed_index, decode_prefix_integer, encode_fixed_index,
    encode_prefix_integer,
};
use gam_linalg::decompose::svd;
use ndarray::{Array1, Array2, Axis};

const VERSION: u64 = 1;
const NODE_KINDS: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Matrix { rows: usize, cols: usize },
    Vector { len: usize },
}
impl Type {
    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Matrix { rows, cols }
                if *rows > 0 && *cols > 0 && rows.checked_mul(*cols).is_some() =>
            {
                Ok(())
            }
            Self::Vector { len } if *len > 0 => Ok(()),
            _ => Err("a matrix-rule parameter has empty or overflowing dimensions".into()),
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Matrix(Array2<f64>),
    Vector(Array1<f64>),
}
impl Value {
    pub fn value_type(&self) -> Type {
        match self {
            Self::Matrix(x) => Type::Matrix {
                rows: x.nrows(),
                cols: x.ncols(),
            },
            Self::Vector(x) => Type::Vector { len: x.len() },
        }
    }
    fn finite(&self) -> bool {
        match self {
            Self::Matrix(x) => x.iter().all(|v| v.is_finite()),
            Self::Vector(x) => x.iter().all(|v| v.is_finite()),
        }
    }
    fn map(&self, f: impl Fn(f64) -> f64 + Copy) -> Self {
        match self {
            Self::Matrix(x) => Self::Matrix(x.mapv(f)),
            Self::Vector(x) => Self::Vector(x.mapv(f)),
        }
    }
}
/// Keep singular values strictly above max(rows,cols)·f64::EPSILON·sigma_max,
/// exactly the decomposition's fixed resolution convention. This is not a bound
/// on computed singular values and supplies no rank/fidelity certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PinvConvention {
    SvdResolutionBand,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Param {
        index: usize,
    },
    Transpose {
        input: usize,
    },
    MatMul {
        left: usize,
        right: usize,
    },
    Diag {
        input: usize,
    },
    Divide {
        numerator: usize,
        denominator: usize,
    },
    Reciprocal {
        input: usize,
    },
    Pinv {
        input: usize,
        convention: PinvConvention,
    },
    Scale {
        input: usize,
        coefficient: f32,
    },
}
impl Node {
    fn kind(&self) -> usize {
        match self {
            Self::Param { .. } => 0,
            Self::Transpose { .. } => 1,
            Self::MatMul { .. } => 2,
            Self::Diag { .. } => 3,
            Self::Divide { .. } => 4,
            Self::Reciprocal { .. } => 5,
            Self::Pinv { .. } => 6,
            Self::Scale { .. } => 7,
        }
    }
    fn references(&self) -> Vec<usize> {
        match self {
            Self::Param { .. } => vec![],
            Self::MatMul { left, right } => vec![*left, *right],
            Self::Divide {
                numerator,
                denominator,
            } => vec![*numerator, *denominator],
            Self::Transpose { input }
            | Self::Diag { input }
            | Self::Reciprocal { input }
            | Self::Pinv { input, .. }
            | Self::Scale { input, .. } => vec![*input],
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixRule {
    pub inputs: Vec<Type>,
    pub nodes: Vec<Node>,
    pub output: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    pub structure_bits: u64,
    pub literals: u64,
}
impl Cost {
    pub fn c32(self) -> u64 {
        self.structure_bits + 32 * self.literals
    }
}
impl MatrixRule {
    pub fn types(&self) -> Result<Vec<Type>, String> {
        if self.nodes.is_empty() || self.output >= self.nodes.len() {
            return Err("a matrix rule needs a valid output and nonempty body".into());
        }
        for input in &self.inputs {
            input.validate()?;
        }
        let mut types: Vec<Type> = Vec::with_capacity(self.nodes.len());
        for (i, node) in self.nodes.iter().enumerate() {
            if node.references().iter().any(|r| *r >= i) {
                return Err(format!(
                    "matrix-rule node {i} has a forward or absent reference"
                ));
            }
            let ty = match node {
                Node::Param { index } => self
                    .inputs
                    .get(*index)
                    .ok_or("a matrix-rule parameter index is absent")?
                    .clone(),
                Node::Transpose { input } | Node::Pinv { input, .. } => match types[*input] {
                    Type::Matrix { rows, cols } => Type::Matrix {
                        rows: cols,
                        cols: rows,
                    },
                    _ => return Err(format!("matrix-rule node {i} requires a matrix")),
                },
                Node::MatMul { left, right } => match (&types[*left], &types[*right]) {
                    (Type::Matrix { rows, cols: inner }, Type::Matrix { rows: other, cols })
                        if inner == other =>
                    {
                        Type::Matrix {
                            rows: *rows,
                            cols: *cols,
                        }
                    }
                    _ => {
                        return Err(format!(
                            "matrix-rule node {i} has incompatible product types"
                        ));
                    }
                },
                Node::Diag { input } => match types[*input] {
                    Type::Vector { len } => Type::Matrix {
                        rows: len,
                        cols: len,
                    },
                    _ => return Err(format!("matrix-rule node {i} requires a vector")),
                },
                Node::Divide {
                    numerator,
                    denominator,
                } => {
                    if types[*numerator] != types[*denominator] {
                        return Err(format!(
                            "matrix-rule node {i} has incompatible divide types"
                        ));
                    }
                    types[*numerator].clone()
                }
                Node::Reciprocal { input } => types[*input].clone(),
                Node::Scale { input, coefficient } => {
                    if !coefficient.is_finite() {
                        return Err("a matrix-rule coefficient is nonfinite".into());
                    }
                    types[*input].clone()
                }
            };
            ty.validate()?;
            types.push(ty);
        }
        Ok(types)
    }
    pub fn evaluate(&self, inputs: &[Value]) -> Result<Value, String> {
        self.types()?;
        if inputs.len() != self.inputs.len()
            || inputs
                .iter()
                .zip(&self.inputs)
                .any(|(v, t)| v.value_type() != *t || !v.finite())
        {
            return Err("matrix-rule arguments differ in count/type or are nonfinite".into());
        }
        let mut values: Vec<Value> = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            let matrix = |i: usize| -> Result<&Array2<f64>, String> {
                match &values[i] {
                    Value::Matrix(x) => Ok(x),
                    Value::Vector(_) => Err("a typed matrix node received a vector".into()),
                }
            };
            let out = match node {
                Node::Param { index } => inputs[*index].clone(),
                Node::Transpose { input } => {
                    Value::Matrix(matrix(*input)?.t().as_standard_layout().to_owned())
                }
                Node::MatMul { left, right } => Value::Matrix(matrix(*left)?.dot(matrix(*right)?)),
                Node::Diag { input } => match &values[*input] {
                    Value::Vector(x) => Value::Matrix(Array2::from_diag(x)),
                    Value::Matrix(_) => {
                        return Err("a typed diagonal node received a matrix".into());
                    }
                },
                Node::Divide {
                    numerator,
                    denominator,
                } => match (&values[*numerator], &values[*denominator]) {
                    (Value::Matrix(a), Value::Matrix(b)) => Value::Matrix(a / b),
                    (Value::Vector(a), Value::Vector(b)) => Value::Vector(a / b),
                    _ => return Err("typed division received different argument types".into()),
                },
                Node::Reciprocal { input } => values[*input].map(|v| 1.0 / v),
                Node::Scale { input, coefficient } => {
                    values[*input].map(|v| v * f64::from(*coefficient))
                }
                Node::Pinv {
                    input,
                    convention: PinvConvention::SvdResolutionBand,
                } => {
                    let x = matrix(*input)?;
                    let d = svd(x.view(), false).map_err(|e| format!("matrix-rule SVD: {e:?}"))?;
                    if !d.band.is_finite()
                        || d.band < 0.0
                        || d.singular_values
                            .iter()
                            .any(|value| !value.is_finite() || *value < 0.0)
                        || d.u
                            .iter()
                            .chain(d.vt.iter())
                            .any(|value| !value.is_finite())
                    {
                        return Err("matrix-rule SVD produced nonfinite or invalid factors".into());
                    }
                    let kept: Vec<usize> = d
                        .singular_values
                        .iter()
                        .enumerate()
                        .filter_map(|(i, s)| (*s > d.band).then_some(i))
                        .collect();
                    let inverse =
                        Array1::from_iter(kept.iter().map(|i| 1.0 / d.singular_values[*i]));
                    let scaled = &d.u.select(Axis(1), &kept).t() * &inverse.insert_axis(Axis(1));
                    Value::Matrix(d.vt.select(Axis(0), &kept).t().dot(&scaled))
                }
            };
            if !out.finite() {
                return Err(format!(
                    "matrix-rule node {index} produced a nonfinite value"
                ));
            }
            values.push(out);
        }
        Ok(values.swap_remove(self.output))
    }
    pub fn encode(&self) -> Result<BitString, String> {
        self.types()?;
        let mut out = BitString::new();
        let err = |e| format!("{e}");
        encode_prefix_integer(&mut out, VERSION).map_err(err)?;
        encode_prefix_integer(&mut out, self.inputs.len() as u64 + 1).map_err(err)?;
        for ty in &self.inputs {
            match ty {
                Type::Matrix { rows, cols } => {
                    encode_fixed_index(&mut out, 0, 2).map_err(err)?;
                    encode_prefix_integer(&mut out, *rows as u64).map_err(err)?;
                    encode_prefix_integer(&mut out, *cols as u64).map_err(err)?;
                }
                Type::Vector { len } => {
                    encode_fixed_index(&mut out, 1, 2).map_err(err)?;
                    encode_prefix_integer(&mut out, *len as u64).map_err(err)?;
                }
            }
        }
        encode_prefix_integer(&mut out, self.nodes.len() as u64).map_err(err)?;
        for (index, node) in self.nodes.iter().enumerate() {
            encode_fixed_index(&mut out, node.kind(), NODE_KINDS).map_err(err)?;
            if let Node::Param { index } = node {
                encode_fixed_index(&mut out, *index, self.inputs.len()).map_err(err)?;
            }
            for reference in node.references() {
                encode_fixed_index(&mut out, reference, index).map_err(err)?;
            }
            match node {
                Node::Pinv { .. } => encode_fixed_index(&mut out, 0, 2).map_err(err)?,
                Node::Scale { coefficient, .. } => out
                    .push_bits(u64::from(coefficient.to_bits()), 32)
                    .map_err(err)?,
                Node::Param { .. }
                | Node::Transpose { .. }
                | Node::MatMul { .. }
                | Node::Diag { .. }
                | Node::Divide { .. }
                | Node::Reciprocal { .. } => {}
            }
        }
        encode_fixed_index(&mut out, self.output, self.nodes.len()).map_err(err)?;
        Ok(out)
    }
    pub fn decode(message: &BitString) -> Result<Self, String> {
        let mut reader = message.reader();
        let rule = Self::read(&mut reader)?;
        reader.finish().map_err(|e| e.to_string())?;
        Ok(rule)
    }
    fn read(reader: &mut BitReader<'_>) -> Result<Self, String> {
        let err = |e| format!("{e}");
        let integer = |reader: &mut BitReader<'_>| -> Result<usize, String> {
            usize::try_from(decode_prefix_integer(reader).map_err(err)?).map_err(|e| e.to_string())
        };
        let version = integer(reader)?;
        if version as u64 != VERSION {
            return Err(format!("unsupported matrix-rule grammar version {version}"));
        }
        let inputs_count = integer(reader)?
            .checked_sub(1)
            .ok_or("zero matrix-rule count codeword")?;
        if inputs_count as u64 > reader.remaining_bits() {
            return Err("matrix-rule input count beyond message".into());
        }
        let mut inputs = Vec::new();
        for _ in 0..inputs_count {
            inputs.push(if decode_fixed_index(reader, 2).map_err(err)? == 0 {
                Type::Matrix {
                    rows: integer(reader)?,
                    cols: integer(reader)?,
                }
            } else {
                Type::Vector {
                    len: integer(reader)?,
                }
            });
        }
        let count = integer(reader)?;
        if count == 0 || count as u64 > reader.remaining_bits() / 3 {
            return Err("matrix-rule node count beyond message".into());
        }
        let mut nodes = Vec::new();
        for i in 0..count {
            let kind = decode_fixed_index(reader, NODE_KINDS).map_err(err)?;
            let reference = |reader: &mut BitReader<'_>| decode_fixed_index(reader, i).map_err(err);
            nodes.push(match kind {
                0 => Node::Param {
                    index: decode_fixed_index(reader, inputs.len()).map_err(err)?,
                },
                1 => Node::Transpose {
                    input: reference(reader)?,
                },
                2 => Node::MatMul {
                    left: reference(reader)?,
                    right: reference(reader)?,
                },
                3 => Node::Diag {
                    input: reference(reader)?,
                },
                4 => Node::Divide {
                    numerator: reference(reader)?,
                    denominator: reference(reader)?,
                },
                5 => Node::Reciprocal {
                    input: reference(reader)?,
                },
                6 => {
                    let input = reference(reader)?;
                    if decode_fixed_index(reader, 2).map_err(err)? != 0 {
                        return Err("unsupported matrix-rule pseudoinverse convention".into());
                    }
                    Node::Pinv {
                        input,
                        convention: PinvConvention::SvdResolutionBand,
                    }
                }
                _ => Node::Scale {
                    input: reference(reader)?,
                    coefficient: f32::from_bits(reader.read_bits(32).map_err(err)? as u32),
                },
            });
        }
        let output = decode_fixed_index(reader, count).map_err(err)?;
        let rule = Self {
            inputs,
            nodes,
            output,
        };
        rule.types()?;
        Ok(rule)
    }
    pub fn cost(&self) -> Result<Cost, String> {
        let message = self.encode()?;
        let literals = self
            .nodes
            .iter()
            .filter(|n| matches!(n, Node::Scale { .. }))
            .count() as u64;
        Ok(Cost {
            structure_bits: message.len_bits() - 32 * literals,
            literals,
        })
    }
}

#[cfg(test)]
#[path = "matrix_rule_tests.rs"]
mod tests;

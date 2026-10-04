//! Native imported attention interfaces, without splitting or renumbering the graph.
//!
//! The Local boundary is the merged post-attention residual. Its scale is the RMS
//! native residual row L2 norm, not a contribution-only attention denominator.
use super::artifact::{Artifact, OperatorLaw};
use super::operator_program::{Node, OperatorBody, OperatorProgram, Rotary, Scale};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq)]
pub struct HeadNormalization {
    pub rms_node: usize,
    pub epsilon: f64,
    pub gain_node: usize,
    pub gain_operator: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttentionHeadMap {
    pub head: usize,
    pub kv_head: usize,
    pub query_operator: usize,
    pub key_operator: usize,
    pub value_operator: usize,
    pub output_operator: usize,
    pub raw_query: usize,
    pub raw_key: usize,
    pub value: usize,
    pub query: usize,
    pub key: usize,
    pub query_norm: Option<HeadNormalization>,
    pub key_norm: Option<HeadNormalization>,
    pub read: usize,
    pub scale: Scale,
    pub rotary: Option<Rotary>,
    pub causal: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttentionLayerMap {
    /// Source layer identity, never the ordinal in a selected subset.
    pub native_layer: usize,
    pub skip: usize,
    pub skip_operator: usize,
    pub normed_input: usize,
    pub input_rms: usize,
    pub input_epsilon: f64,
    pub input_gain: usize,
    pub input_bias: Option<usize>,
    pub output: usize,
    pub output_bias: Option<usize>,
    pub final_gain: usize,
    pub final_norm: usize,
    pub heads: Vec<AttentionHeadMap>,
}

pub const LOCAL_DENOMINATOR: &str =
    "RMS native post-attention-residual row L2 norm; not contribution-only Local";

fn error(message: impl std::fmt::Display) -> String {
    format!("native attention map: {message}")
}
fn named(p: &OperatorProgram, name: &str) -> Result<usize, String> {
    let found: Vec<_> = p
        .operators
        .iter()
        .enumerate()
        .filter(|(_, o)| o.name == name)
        .map(|(i, _)| i)
        .collect();
    match found.as_slice() {
        [index] => Ok(*index),
        _ => Err(error(format!(
            "expected unique operator {name}, found {}",
            found.len()
        ))),
    }
}
fn inventory(p: &OperatorProgram, layer: usize, letter: char) -> Result<Vec<usize>, String> {
    let prefix = format!("blocks.{layer}.{letter}");
    let mut out = BTreeMap::new();
    for (i, o) in p.operators.iter().enumerate() {
        let Some(tail) = o.name.strip_prefix(&prefix) else {
            continue;
        };
        if tail.is_empty() || !tail.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let head: usize = tail.parse().map_err(error)?;
        if tail != head.to_string() || out.insert(head, i).is_some() {
            return Err(error("ambiguous head operator inventory"));
        }
    }
    if out.is_empty() || !out.keys().copied().eq(0..out.len()) {
        return Err(error(format!(
            "noncontiguous/empty {prefix} head inventory"
        )));
    }
    Ok(out.into_values().collect())
}
fn affine_use(p: &OperatorProgram, operator: usize) -> Result<usize, String> {
    let uses: Vec<_> = p
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.operators().contains(&operator))
        .map(|(i, _)| i)
        .collect();
    match uses.as_slice() {
        [n] if matches!(&p.nodes[*n],Node::Affine { terms,.. } if terms.iter().filter(|(_,o)|*o==operator).count()==1) => {
            Ok(*n)
        }
        _ => Err(error(format!(
            "operator {} needs one affine use",
            p.operators[operator].name
        ))),
    }
}
fn projection_input(p: &OperatorProgram, node: usize, op: usize) -> Result<usize, String> {
    match &p.nodes[node] {
        Node::Affine { terms, bias: None } if terms.as_slice().len() == 1 && terms[0].1 == op => {
            Ok(terms[0].0)
        }
        _ => Err(error("unrecognized biased or multi-term Q/K/V projection")),
    }
}
fn diagonal(p: &OperatorProgram, op: usize, name: &str) -> Result<(), String> {
    if p.operators[op].name != name
        || !matches!(p.operators[op].body, OperatorBody::Diagonal { .. })
    {
        return Err(error(format!("expected diagonal {name}")));
    }
    Ok(())
}
fn normalized(
    p: &OperatorProgram,
    actual: usize,
    raw: usize,
    name: &str,
) -> Result<Option<HeadNormalization>, String> {
    let declared = p.operators.iter().any(|o| o.name == name);
    if actual == raw {
        return if declared {
            Err(error(format!("{name} declared but attention bypasses it")))
        } else {
            Ok(None)
        };
    }
    let (rms, gain) = match &p.nodes[actual] {
        Node::Affine { terms, bias: None } if terms.len() == 1 => terms[0],
        _ => return Err(error("unrecognized attention query/key normalization")),
    };
    diagonal(p, gain, name)?;
    let epsilon = match p.nodes[rms] {
        Node::RmsNorm { input, epsilon } if input == raw && epsilon.is_finite() && epsilon > 0. => {
            epsilon
        }
        _ => {
            return Err(error(
                "normalization does not read the declared raw projection",
            ));
        }
    };
    Ok(Some(HeadNormalization {
        rms_node: rms,
        epsilon,
        gain_node: actual,
        gain_operator: gain,
    }))
}
fn readers(p: &OperatorProgram, node: usize) -> BTreeSet<usize> {
    p.nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.arguments().contains(&node))
        .map(|(i, _)| i)
        .collect()
}
fn require_readers(
    p: &OperatorProgram,
    node: usize,
    expected: impl IntoIterator<Item = usize>,
) -> Result<(), String> {
    if readers(p, node) != expected.into_iter().collect() {
        return Err(error(format!("unexpected reader/bypass of node {node}")));
    }
    Ok(())
}
fn ancestors(p: &OperatorProgram, node: usize) -> BTreeSet<usize> {
    let mut seen = BTreeSet::new();
    let mut todo = vec![node];
    while let Some(n) = todo.pop() {
        if seen.insert(n) {
            todo.extend(p.nodes[n].arguments());
        }
    }
    seen
}
impl AttentionLayerMap {
    pub fn of(p: &OperatorProgram, native_layer: usize) -> Result<Self, String> {
        // Establish bounds, interfaces and topological validity before following indices.
        if p.nodes.get(p.output).is_none() {
            return Err(error("native output out of bounds"));
        }
        p.interfaces().map_err(error)?;
        let q = inventory(p, native_layer, 'q')?;
        let k = inventory(p, native_layer, 'k')?;
        let v = inventory(p, native_layer, 'v')?;
        let o = inventory(p, native_layer, 'o')?;
        if q.len() != o.len() || k.len() != v.len() || !q.len().is_multiple_of(k.len()) {
            return Err(error("inconsistent GQA inventory"));
        }
        let group = q.len() / k.len();
        let output = affine_use(p, o[0])?;
        let (terms, output_bias) = match &p.nodes[output] {
            Node::Affine { terms, bias } => (terms, *bias),
            _ => return Err(error("output operator is not affine")),
        };
        if terms.len() != o.len() + 1 {
            return Err(error(
                "merged attention has undeclared terms or no residual",
            ));
        }
        let outputs: BTreeSet<_> = o.iter().copied().collect();
        let other: Vec<_> = terms
            .iter()
            .copied()
            .filter(|(_, op)| !outputs.contains(op))
            .collect();
        let [(skip, skip_operator)] = other.as_slice() else {
            return Err(error("merged attention needs one native residual skip"));
        };
        if !matches!(p.operators[*skip_operator].body, OperatorBody::Identity) {
            return Err(error("residual skip is not identity"));
        }
        let mut heads = Vec::new();
        let mut normed = None;
        for h in 0..q.len() {
            if affine_use(p, o[h])? != output {
                return Err(error("head outputs do not share one merged residual"));
            }
            let read = terms
                .iter()
                .find(|(_, op)| *op == o[h])
                .ok_or_else(|| error("head output term absent"))?
                .0;
            let (query, key, value, scale, rotary, causal) = match &p.nodes[read] {
                Node::Attend {
                    query,
                    key,
                    value,
                    scale,
                    rotary,
                    causal,
                } => (*query, *key, *value, *scale, rotary.clone(), *causal),
                _ => return Err(error("native O does not read its actual Attend node")),
            };
            if heads
                .iter()
                .any(|prior: &AttentionHeadMap| prior.read == read)
            {
                return Err(error("multiple heads alias one attention read"));
            }
            let kv = h / group;
            let raw_query = affine_use(p, q[h])?;
            let raw_key = affine_use(p, k[kv])?;
            let native_value = affine_use(p, v[kv])?;
            if value != native_value {
                return Err(error(
                    "actual Attend value disagrees with declared GQA group",
                ));
            }
            for (node, op) in [(raw_query, q[h]), (raw_key, k[kv]), (value, v[kv])] {
                let input = projection_input(p, node, op)?;
                if normed.is_some_and(|n| n != input) {
                    return Err(error(
                        "head projections do not share one native normalized input",
                    ));
                }
                normed = Some(input);
            }
            let query_norm = normalized(
                p,
                query,
                raw_query,
                &format!("blocks.{native_layer}.attn.q_norm.gain"),
            )?;
            let key_norm = normalized(
                p,
                key,
                raw_key,
                &format!("blocks.{native_layer}.attn.k_norm.gain"),
            )?;
            if query_norm.is_some() != key_norm.is_some() {
                return Err(error("inconsistent query/key normalization"));
            }
            if let Some(first) = heads.first() {
                if first.query_norm.is_some() != query_norm.is_some()
                    || first.scale != scale
                    || first.rotary != rotary
                    || first.causal != causal
                {
                    return Err(error("inconsistent native attention policies"));
                }
            }
            heads.push(AttentionHeadMap {
                head: h,
                kv_head: kv,
                query_operator: q[h],
                key_operator: k[kv],
                value_operator: v[kv],
                output_operator: o[h],
                raw_query,
                raw_key,
                value,
                query,
                key,
                query_norm,
                key_norm,
                read,
                scale,
                rotary,
                causal,
            });
        }
        for head in &heads {
            require_readers(p, head.read, [output])?;
            require_readers(p, head.query, [head.read])?;
            if let Some(n) = &head.query_norm {
                require_readers(p, head.raw_query, [n.rms_node])?;
                require_readers(p, n.rms_node, [n.gain_node])?;
            }
        }
        for kv in 0..k.len() {
            let group_heads: Vec<_> = heads.iter().filter(|h| h.kv_head == kv).collect();
            let first = group_heads[0];
            if group_heads
                .iter()
                .any(|h| h.key != first.key || h.key_norm != first.key_norm)
            {
                return Err(error("GQA key normalization is not shared exactly"));
            }
            require_readers(p, first.key, group_heads.iter().map(|h| h.read))?;
            require_readers(p, first.value, group_heads.iter().map(|h| h.read))?;
            if let Some(n) = &first.key_norm {
                require_readers(p, first.raw_key, [n.rms_node])?;
                require_readers(p, n.rms_node, [n.gain_node])?;
            }
        }
        let normed_input = normed.ok_or_else(|| error("no normalized input"))?;
        let (input_rms, input_gain, input_bias) = match &p.nodes[normed_input] {
            Node::Affine { terms, bias } if terms.len() == 1 => (terms[0].0, terms[0].1, *bias),
            _ => return Err(error("unrecognized native input gain")),
        };
        diagonal(p, input_gain, &format!("blocks.{native_layer}.rms1.gain"))?;
        let input_epsilon = match p.nodes[input_rms] {
            Node::RmsNorm { input, epsilon }
                if input == *skip && epsilon.is_finite() && epsilon > 0. =>
            {
                epsilon
            }
            _ => {
                return Err(error(
                    "native RMS input bypasses residual skip or uses unsupported centering",
                ));
            }
        };
        let final_gain = named(p, "final_norm.gain")?;
        diagonal(p, final_gain, "final_norm.gain")?;
        let final_norm = affine_use(p, final_gain)?;
        if !ancestors(p, p.output).contains(&final_norm)
            || !ancestors(p, final_norm).contains(&output)
        {
            return Err(error("final gain is disconnected from native output"));
        }
        match &p.nodes[final_norm] {
            Node::Affine { terms, .. }
                if terms.len() == 1 && matches!(p.nodes[terms[0].0], Node::RmsNorm { .. }) => {}
            _ => return Err(error("unrecognized final RMS gain application")),
        }
        Ok(Self {
            native_layer,
            skip: *skip,
            skip_operator: *skip_operator,
            normed_input,
            input_rms,
            input_epsilon,
            input_gain,
            input_bias,
            output,
            output_bias,
            final_gain,
            final_norm,
            heads,
        })
    }
    pub fn copy_law(&self, head: usize) -> Result<OperatorLaw, String> {
        let h = self
            .heads
            .get(head)
            .ok_or_else(|| error("head outside declared inventory"))?;
        Ok(OperatorLaw::Copy {
            value: h.value_operator,
            gain: self.input_gain,
            final_gain: self.final_gain,
        })
    }
    pub fn native_reads(&self) -> [usize; 2] {
        [self.skip, self.normed_input]
    }
    pub fn bind(&self, artifact: &Artifact) -> Result<Artifact, String> {
        artifact.bind(
            &format!("attention {} post-residual boundary", self.native_layer),
            &self.native_reads(),
            self.output,
        )
    }
}
#[cfg(test)]
#[path = "attention_map_tests.rs"]
mod tests;

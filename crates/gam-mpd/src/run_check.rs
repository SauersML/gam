//! The site nodes of an imported language model (#2951).
//!
//! [`split_sites`] rewrites `import::import_language_model`'s program so that every decomposed
//! site's output is a node: the `o` output (`Σ_h W_O,h read_h`) and the `down_proj` output each
//! get their own node, which the residual node then adds to the stream, `x ← x + out`. A replaced
//! attention or MLP block then writes its own output in the native interface, not the stream it is
//! added to. [`layer_nodes`] names every layer's site nodes.

use super::operator_program::{Node, OperatorBody, OperatorProgram, remap_node};

/// One layer's site nodes in the split native program ([`split_sites`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LayerNodes {
    /// The residual stream entering the layer, and its normed copy (the `q`, `k`, `v` sites' input).
    pub stream: usize,
    pub normed_stream: usize,
    /// Per head its query node, per key-value head its key and value nodes.
    pub queries: Vec<usize>,
    pub keys: Vec<usize>,
    pub values: Vec<usize>,
    /// Per head its attention read (the `o` site's input, the head's columns).
    pub reads: Vec<usize>,
    /// The `o` site's output, and the residual after attention (the stream plus it).
    pub attention: usize,
    pub attended: usize,
    /// The `c_fc` site's input (the normed stream), output and the activations (`down_proj`'s input).
    pub normed: usize,
    pub pre: usize,
    pub active: usize,
    /// The `down_proj` site's output, and the residual after the MLP (`attended` plus it).
    pub mlp: usize,
    pub residual: usize,
}

/// Parse `blocks.{l}.{part}`.
fn site_part(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blocks.")?;
    let (layer, part) = rest.split_once('.')?;
    Some((layer.parse().ok()?, part))
}

fn writes_site(program: &OperatorProgram, op: usize) -> bool {
    site_part(&program.operators[op].name).is_some_and(|(_, part)| part == "down_proj" || part.strip_prefix('o').is_some_and(|h| h.parse::<usize>().is_ok()))
}

/// The language model's program with every `o` and `down_proj` output its own node (module note):
/// each affine node adding a site's terms to the stream becomes the site's terms alone, followed by
/// the stream plus that node through the identity.
pub fn split_sites(program: &OperatorProgram) -> Result<OperatorProgram, String> {
    let mut out = program.clone();
    let mut nodes: Vec<Node> = Vec::with_capacity(program.nodes.len() + 16);
    let mut map: Vec<usize> = Vec::with_capacity(program.nodes.len());
    let identity: Vec<usize> = (0..program.operators.len()).collect();
    let (bases, rules): (Vec<usize>, Vec<usize>) = ((0..program.bases.len()).collect(), (0..program.rules.len()).collect());
    for (index, node) in program.nodes.iter().enumerate() {
        let mut node = node.clone();
        remap_node(&mut node, &map, &identity, &bases, &rules);
        if let Node::Affine { terms, bias } = &node
            && terms.iter().any(|(_, op)| writes_site(program, *op))
        {
            let (site, rest): (Vec<(usize, usize)>, Vec<(usize, usize)>) = terms.iter().partition(|(_, op)| writes_site(program, *op));
            let [(stream, skip)] = rest[..] else {
                return Err(format!("node {index}: a site write with {} other terms", rest.len()));
            };
            if !matches!(program.operators[skip].body, OperatorBody::Identity) {
                return Err(format!("node {index}: the stream enters through {}, not the identity", program.operators[skip].name));
            }
            nodes.push(Node::Affine { terms: site, bias: *bias });
            let written = nodes.len() - 1;
            nodes.push(Node::Affine { terms: vec![(stream, skip), (written, skip)], bias: None });
        } else {
            nodes.push(node);
        }
        map.push(nodes.len() - 1);
    }
    out.output = map[program.output];
    out.nodes = nodes;
    out.interfaces().map_err(|e| e.to_string())?;
    Ok(out)
}

/// The split native program's site nodes, found by its operators' names (`blocks.{l}.q{h}`, …).
pub fn layer_nodes(native: &OperatorProgram, layers: usize) -> Result<Vec<LayerNodes>, String> {
    let name = |op: usize| native.operators[op].name.as_str();
    let mut out = vec![LayerNodes::default(); layers];
    let mut found = vec![[false; 3]; layers];
    for (index, node) in native.nodes.iter().enumerate() {
        if let Node::Affine { terms, bias } = node {
            for (argument, op) in terms {
                let Some((layer, part)) = site_part(name(*op)) else {
                    continue;
                };
                if layer >= layers {
                    return Err(format!("operator {} beyond {layers} layers", name(*op)));
                }
                if bias.is_some() {
                    return Err(format!("{} has a bias: its input mix is not its output mix", name(*op)));
                }
                let nodes = &mut out[layer];
                let indexed = |prefix: &str| part.strip_prefix(prefix).and_then(|h| h.parse::<usize>().ok());
                if let Some(h) = indexed("q") {
                    put(&mut nodes.queries, h, index);
                    nodes.normed_stream = *argument;
                } else if let Some(g) = indexed("k") {
                    put(&mut nodes.keys, g, index);
                } else if let Some(g) = indexed("v") {
                    put(&mut nodes.values, g, index);
                } else if indexed("o").is_some() {
                    nodes.attention = index;
                    found[layer][0] = true;
                } else if part == "c_fc" {
                    nodes.normed = *argument;
                    nodes.pre = index;
                    found[layer][1] = true;
                } else if part == "down_proj" {
                    nodes.active = *argument;
                    nodes.mlp = index;
                    found[layer][2] = true;
                }
            }
            // The residual nodes: the stream plus a site's output through the identity.
            if let [(stream, _), (written, _)] = terms[..] {
                for nodes in out.iter_mut() {
                    if written == nodes.attention && nodes.attention != 0 {
                        nodes.stream = stream;
                        nodes.attended = index;
                    } else if written == nodes.mlp && nodes.mlp != 0 {
                        nodes.residual = index;
                    }
                }
            }
        } else if let Node::Attend { query, .. } = node {
            let query = head_projection(native, *query);
            for nodes in out.iter_mut() {
                if let Some(h) = nodes.queries.iter().position(|q| *q == query) {
                    put(&mut nodes.reads, h, index);
                }
            }
        }
    }
    for (layer, nodes) in out.iter().enumerate() {
        if !found[layer].iter().all(|f| *f)
            || nodes.reads.len() != nodes.queries.len()
            || nodes.queries.is_empty()
            || nodes.attended == 0
            || nodes.residual == 0
        {
            return Err(format!("layer {layer}: the native program is not a split sequential decoder of named sites (split_sites)"));
        }
    }
    Ok(out)
}

/// The projection a head's query or key node is made from: the node itself, or, under a head norm
/// (Qwen3's `q_norm`, `k_norm`: a diagonal gain on an RMS norm of the projection), the projection.
pub fn head_projection(program: &OperatorProgram, node: usize) -> usize {
    if let Node::Affine { terms, bias: None } = &program.nodes[node]
        && let [(normed, gain)] = terms[..]
        && matches!(program.operators[gain].body, OperatorBody::Diagonal { .. })
        && let Node::RmsNorm { input, .. } = program.nodes[normed]
    {
        return input;
    }
    node
}

fn put(list: &mut Vec<usize>, at: usize, value: usize) {
    if list.len() <= at {
        list.resize(at + 1, usize::MAX);
    }
    list[at] = value;
}


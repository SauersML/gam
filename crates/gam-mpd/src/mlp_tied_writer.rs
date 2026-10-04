//! Exploratory full-hidden tied writer grammar, included by the diagnostic example.
//! No activation fitting or acceptance decisions: gains fit native weight columns only.
use gam_mpd::{
    artifact::{Argument, Artifact, Callee},
    operator_program::{
        Node, Operator, OperatorBody, OperatorProgram, Provenance, Rule, exact_precision,
    },
};
use ndarray::{Array1, Array2, Axis};
use std::{collections::BTreeSet, sync::Arc};

#[derive(Clone, Debug)]
pub struct MlpWriterMap {
    pub layer: usize,
    pub reader: usize,
    pub writer: usize,
    pub normed: usize,
    pub pre: usize,
    pub active: usize,
    pub skip: usize,
    pub identity: usize,
    pub output: usize,
    pub bias: Option<usize>,
    pub gate: Option<usize>,
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
        [i] => Ok(*i),
        _ => Err(format!("expected unique native {name}")),
    }
}
fn use_node(p: &OperatorProgram, op: usize) -> Result<usize, String> {
    let uses: Vec<_> = p
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.operators().contains(&op))
        .map(|(i, _)| i)
        .collect();
    match uses.as_slice() {
        [i] => Ok(*i),
        _ => Err(format!("{} must have one native use", p.operators[op].name)),
    }
}
impl MlpWriterMap {
    /// Every native named MLP, in source-ID order; no selected layer or fidelity pruning.
    pub fn all(p: &OperatorProgram) -> Result<Vec<Self>, String> {
        p.interfaces().map_err(|e| e.to_string())?;
        let mut layers = BTreeSet::new();
        for op in &p.operators {
            if let Some(rest) = op.name.strip_prefix("blocks.") {
                if let Some((layer, "c_fc")) = rest.split_once('.') {
                    layers.insert(layer.parse::<usize>().map_err(|e| e.to_string())?);
                }
            }
        }
        if layers.is_empty() {
            return Err("no native MLP readers".into());
        }
        layers
            .into_iter()
            .map(|layer| {
                let reader = named(p, &format!("blocks.{layer}.c_fc"))?;
                let writer = named(p, &format!("blocks.{layer}.down_proj"))?;
                let pre = use_node(p, reader)?;
                let output = use_node(p, writer)?;
                let normed = match &p.nodes[pre] {
                    Node::Affine { terms, .. } if terms.len() == 1 && terms[0].1 == reader => {
                        terms[0].0
                    }
                    _ => return Err("unrecognized native up projection".into()),
                };
                let (terms, bias) = match &p.nodes[output] {
                    Node::Affine { terms, bias } if terms.len() == 2 => (terms, *bias),
                    _ => return Err("expected merged native skip plus MLP writer".into()),
                };
                let active = terms
                    .iter()
                    .find(|(_, op)| *op == writer)
                    .ok_or("native writer term absent")?
                    .0;
                let (skip, identity) = terms
                    .iter()
                    .find(|(_, op)| *op != writer)
                    .copied()
                    .ok_or("native skip absent")?;
                if !matches!(p.operators[identity].body, OperatorBody::Identity) {
                    return Err("native skip is not identity".into());
                }
                let gate = match &p.nodes[active] {
                    Node::Pointwise { input, .. } if *input == pre => None,
                    Node::Hadamard { left, right } if *right == pre => match &p.nodes[*left] {
                        Node::Pointwise { input, .. } => {
                            let op = named(p, &format!("blocks.{layer}.gate_proj"))?;
                            if use_node(p, op)? != *input {
                                return Err("gated activation does not read native gate".into());
                            }
                            match &p.nodes[*input] {
                                Node::Affine { terms, .. }
                                    if terms.len() == 1
                                        && terms[0].0 == normed
                                        && terms[0].1 == op =>
                                {
                                    Some(*input)
                                }
                                _ => return Err("gate/up native inputs differ".into()),
                            }
                        }
                        _ => return Err("unsupported native gated law".into()),
                    },
                    _ => return Err("unsupported native full-hidden activation".into()),
                };
                if p.operators[reader].rows != p.operators[writer].cols
                    || p.operators[reader].cols != p.operators[writer].rows
                {
                    return Err("native up/down interfaces are not transposes".into());
                }
                Ok(Self {
                    layer,
                    reader,
                    writer,
                    normed,
                    pre,
                    active,
                    skip,
                    identity,
                    output,
                    bias,
                    gate,
                })
            })
            .collect()
    }
    pub fn bind(&self, a: &Artifact) -> Result<Artifact, String> {
        a.bind(
            &format!("MLP {} full-hidden post-residual boundary", self.layer),
            &[self.skip, self.normed],
            self.output,
        )
    }
}

pub struct WriterFit {
    pub gains: Array1<f64>,
    pub zero_reader_rows: usize,
    pub prediction: Array2<f64>,
    pub residual: Array2<f64>,
    pub native_norm_squared: f64,
}
impl WriterFit {
    /// a_j = <D[:,j], U[j,:]> / ||U[j,:]||². Exact zero denominator
    /// chooses canonical zero; no threshold or signed-cosine selection.
    pub fn of(up: &Array2<f64>, down: &Array2<f64>) -> Result<Self, String> {
        if up.nrows() != down.ncols()
            || up.ncols() != down.nrows()
            || up.is_empty()
            || up.iter().chain(down.iter()).any(|v| !v.is_finite())
        {
            return Err("finite transposed up/down dimensions required".into());
        }
        let mut gains = Array1::zeros(up.nrows());
        let mut zero_reader_rows = 0;
        for j in 0..up.nrows() {
            let denominator = up.row(j).iter().map(|u| u * u).sum::<f64>();
            let numerator = up
                .row(j)
                .iter()
                .zip(down.column(j).iter())
                .map(|(u, d)| u * d)
                .sum::<f64>();
            if !denominator.is_finite() || !numerator.is_finite() {
                return Err("native-weight least-squares dot overflow".into());
            }
            let gain = if denominator == 0. {
                if up.row(j).iter().any(|v| *v != 0.) {
                    return Err(
                        "nonzero reader norm underflow; canonical zero is only for exact zero rows"
                            .into(),
                    );
                }
                zero_reader_rows += 1;
                0.
            } else {
                numerator / denominator
            };
            gains[j] = f64::from(gain as f32);
            if !gains[j].is_finite() {
                return Err("weight least-squares gain overflow".into());
            }
        }
        // h diag(a) U; the decoded a is rounded before computing the residual.
        let prediction = (up.t().to_owned() * &gains.view().insert_axis(Axis(0)))
            .as_standard_layout()
            .into_owned();
        let residual = down - &prediction;
        let native_norm_squared = down.iter().map(|v| v * v).sum();
        Ok(Self {
            gains,
            zero_reader_rows,
            prediction,
            residual,
            native_norm_squared,
        })
    }
}

pub struct WriterSvd {
    u: Array2<f64>,
    values: Array1<f64>,
    vt: Array2<f64>,
}
impl WriterSvd {
    pub fn of(matrix: &Array2<f64>) -> Result<Self, String> {
        let d = gam_linalg::decompose::svd(matrix.view(), false).map_err(|e| e.to_string())?;
        Ok(Self {
            u: d.u,
            values: d.singular_values,
            vt: d.vt,
        })
    }
    pub fn operator(
        &self,
        original: &Operator,
        rank: usize,
        method: &str,
    ) -> Result<Operator, String> {
        if rank > self.values.len() {
            return Err("declared residual rank exceeds writer dimensions".into());
        }
        let provenance = Provenance::derived(
            &[&original.provenance],
            format!("{method}; declared rank{rank}, balanced f32 factors"),
        );
        if rank == 0 {
            return Operator::blocks(
                original.name.clone(),
                original.rows.clone(),
                original.cols.clone(),
                Array2::zeros((original.rows.width(), original.cols.width())),
                Array2::from_elem(
                    (original.rows.group_count(), original.cols.group_count()),
                    false,
                ),
                exact_precision([0.]).map_err(|e| e.to_string())?,
                provenance,
            )
            .map_err(|e| e.to_string());
        }
        let kept: Vec<_> = (0..rank).collect();
        let roots = Array1::from_iter(self.values.iter().take(rank).map(|s| s.sqrt()));
        let mut left = (self.u.select(Axis(1), &kept) * &roots)
            .as_standard_layout()
            .into_owned();
        let mut right = (self.vt.select(Axis(0), &kept) * &roots.insert_axis(Axis(1)))
            .as_standard_layout()
            .into_owned();
        left.mapv_inplace(|v| f64::from(v as f32));
        right.mapv_inplace(|v| f64::from(v as f32));
        let precision =
            exact_precision(left.iter().chain(right.iter()).copied()).map_err(|e| e.to_string())?;
        Operator::low_rank(
            original.name.clone(),
            original.rows.clone(),
            original.cols.clone(),
            left,
            right,
            precision,
            provenance,
        )
        .map_err(|e| e.to_string())
    }
}
pub fn tied_candidate(
    base: &Artifact,
    map: &MlpWriterMap,
    fit: &WriterFit,
    residual: Option<Operator>,
) -> Result<Artifact, String> {
    let old = &base.program.operators[map.writer];
    let up = &base.program.operators[map.reader];
    let gain = Operator::diag(
        format!("blocks.{}.tied_writer.gains", map.layer),
        up.rows.clone(),
        fit.gains.clone(),
        exact_precision(fit.gains.iter().copied()).map_err(|e| e.to_string())?,
        Provenance::derived(
            &[&old.provenance, &up.provenance],
            "column/row native-weight least-squares, before evaluation; f32 gains".into(),
        ),
    )
    .map_err(|e| e.to_string())?;
    let offset = base.program.operators.len();
    let mut added = vec![gain];
    let mut terms = vec![(1, map.identity), (3, map.identity)];
    if let Some(mut residual) = residual {
        residual.name = format!("blocks.{}.tied_writer.residual", map.layer);
        terms.push((0, offset + 1));
        added.push(residual);
    }
    let rule = Rule {
        name: format!("full-hidden tied MLP writer{}", map.layer),
        inputs: vec![up.rows.clone(), old.rows.clone()],
        nodes: vec![
            Node::Param { index: 0 },
            Node::Param { index: 1 },
            Node::Affine {
                terms: vec![(0, offset)],
                bias: None,
            },
            Node::Transposed {
                input: 2,
                operator: map.reader,
            },
            Node::Affine {
                terms,
                bias: map.bias,
            },
        ],
        output: 4,
    };
    let out = base.replace_block(
        &rule.name.clone(),
        Callee::New(rule),
        vec![Argument::Native(map.active), Argument::Native(map.skip)],
        map.output,
        added,
    )?;
    let out = map.bind(&out)?.f32_literals()?;
    out.validate_coverage(&base.program)?;
    if out.places.len() != base.places.len()
        || base
            .places
            .iter()
            .any(|(native, _)| out.place(*native).is_none())
    {
        return Err("tied writer lost a native activation/intervention place".into());
    }
    Ok(out)
}
pub fn svd_candidate(
    base: &Artifact,
    map: &MlpWriterMap,
    writer: Operator,
) -> Result<Artifact, String> {
    let mut out = base.clone();
    out.program.operators[map.writer] = Arc::new(writer);
    let out = map.bind(&out)?.f32_literals()?;
    out.validate_coverage(&base.program)?;
    Ok(out)
}
#[cfg(test)]
#[path = "mlp_tied_writer_tests.rs"]
mod tests;

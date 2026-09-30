//! Shared factors (#2951): operators that write into one space factored through a common library of
//! directions, `A_s = F C_s`, with `F` sent once.
//!
//! Every dense operator writing into one row interface (the residual stream, say) is stacked side by
//! side, `[A_1 … A_S] = U Σ Vᵀ`, and the leading `K` left singular vectors are the library
//! `F = U_K`. Each operator becomes `F C_s` with `C_s = Fᵀ A_s`: a use `A_s x` becomes `F (C_s x)`,
//! an affine node over the `K` library coordinates followed by the shared `F`. At `K` equal to the
//! stack's rank the rewrite is exact; below it, every operator loses the same discarded directions,
//! and the contract and the code decide whether that pays. `K` is proposed on the ladder
//! `1, 2, 4, …` while the factored reals are fewer than the operators'.
//!
//! The library coordinates `C_s x` of a token table read by indicator features are the tokens'
//! coordinates on the library: a data table (`view::ComponentView::data`), the discovered
//! representation of the tokens.

use super::dense::svd;
use super::engine::{EngineError, Edit, Exactness, Primitive, Proposal, SearchContext};
use super::fit::ProposalKind;
use super::operator_program::{Interface, LabelKind, Node, Operator, OperatorBody, OperatorProgram, Provenance, band_precision};
use super::operator_rewrites::insert_node;
use gam_linalg::roundoff::accumulation_growth;
use ndarray::{Array2, Axis, concatenate, s};
use std::collections::BTreeMap;

/// Shared writer factors over each row interface (module note).
pub struct SharedFactors;

/// The dense operators used only as affine terms, grouped by their row interface.
fn writer_groups(program: &OperatorProgram) -> Vec<Vec<usize>> {
    let mut only_terms = vec![true; program.operators.len()];
    let mut used = vec![false; program.operators.len()];
    for node in &program.nodes {
        match node {
            Node::Affine { terms, bias } => {
                for (_, op) in terms {
                    used[*op] = true;
                }
                if let Some(op) = bias {
                    only_terms[*op] = false;
                }
            }
            other => {
                for op in other.operators() {
                    only_terms[op] = false;
                }
            }
        }
    }
    for rule in &program.rules {
        for node in &rule.nodes {
            for op in node.operators() {
                only_terms[op] = false;
            }
        }
    }
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, op) in program.operators.iter().enumerate() {
        let OperatorBody::Dense { present, .. } = &op.body else { continue };
        if !(used[index] && only_terms[index]) || present.iter().any(|keep| !keep) || op.rows.width() < 2 {
            continue;
        }
        groups.entry(format!("{:?}", op.rows)).or_default().push(index);
    }
    groups.into_values().collect()
}

impl Primitive for SharedFactors {
    fn name(&self) -> &'static str {
        "shared_factors"
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError> {
        let program = context.program;
        let mut out = Vec::new();
        for group in writer_groups(program) {
            let matrices: Vec<Array2<f64>> = group.iter().map(|op| program.operators[*op].matrix()).collect();
            let views: Vec<_> = matrices.iter().map(|m| m.view()).collect();
            let stacked = concatenate(Axis(1), &views).map_err(|e| EngineError::Primitive(e.to_string()))?;
            let (d, total_cols) = stacked.dim();
            let reals: usize = group.iter().map(|op| program.operators[*op].real_count()).sum();
            let decomposed = svd(stacked.view(), false).map_err(|e| EngineError::Primitive(format!("{e:?}")))?;
            let rank = decomposed.singular_values.len();
            let mut k = 1;
            while k < rank.min(d) && d * k + k * total_cols < reals {
                let library = decomposed.u.slice(s![.., ..k]).to_owned();
                if let Some(candidate) = factor(program, &group, &library)? {
                    let dropped = decomposed.singular_values.iter().skip(k).copied().fold(0.0_f64, f64::max);
                    let exactness = if dropped <= decomposed.band {
                        Exactness::Exact { derivation: format!("the stack's singular values beyond {k} are within its band") }
                    } else {
                        Exactness::Approximate
                    };
                    let names: Vec<&str> = group.iter().map(|op| program.operators[*op].name.as_str()).collect();
                    out.push(Proposal {
                        primitive: "shared_factors",
                        kind: ProposalKind::Share,
                        exactness,
                        description: format!("{k} shared factors of {names:?}"),
                        edit: Edit::Program(Box::new(candidate)),
                    });
                }
                k *= 2;
            }
        }
        Ok(out)
    }
}

/// The program with every operator of `group` written through `library` (`d × K`, orthonormal
/// columns): the library is one operator, each member `A_s` becomes `C_s = libraryᵀ A_s`, and each
/// affine term `(x, A_s)` becomes `(z, library)` with `z = Affine(x; C_s)` inserted before its node.
fn factor(program: &OperatorProgram, group: &[usize], library: &Array2<f64>) -> Result<Option<OperatorProgram>, EngineError> {
    let k = library.ncols();
    let coordinates = Interface::uniform(k, 1, LabelKind::Unit, 0)?;
    let rows = program.operators[group[0]].rows.clone();
    let mut candidate = program.clone();
    let library_values = library.clone();
    let largest = library_values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let library_precision = band_precision(accumulation_growth(rows.width()) * largest, largest)?;
    let parts: Vec<&Provenance> = group.iter().map(|op| &program.operators[*op].provenance).collect();
    candidate.operators.push(
        Operator::dense("library", rows, coordinates.clone(), library_values, library_precision, Provenance::derived(&parts, "left singular vectors of the stacked writers".to_string()))?,
    );
    let library_op = candidate.operators.len() - 1;
    let mut coefficient_of = BTreeMap::new();
    for &op in group {
        let old = &program.operators[op];
        let values = library.t().dot(&old.matrix());
        let largest = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let OperatorBody::Dense { precision, .. } = &old.body else { return Ok(None) };
        let band = accumulation_growth(library.nrows()) * largest + precision.worst_case_error() * library.nrows() as f64;
        candidate.operators.push(Operator::dense(
            format!("{}·coordinates", old.name),
            coordinates.clone(),
            old.cols.clone(),
            values,
            band_precision(band, largest)?,
            Provenance::derived(&[&old.provenance], "coordinates on the shared library".to_string()),
        )?);
        coefficient_of.insert(op, candidate.operators.len() - 1);
    }
    // Rewrite every term, inserting each coordinate node right before the node that reads it.
    let mut index = 0;
    while index < candidate.nodes.len() {
        let Node::Affine { terms, bias } = candidate.nodes[index].clone() else {
            index += 1;
            continue;
        };
        if !terms.iter().any(|(_, op)| coefficient_of.contains_key(op)) {
            index += 1;
            continue;
        }
        let mut new_terms = Vec::new();
        let mut at = index;
        for (argument, op) in terms {
            match coefficient_of.get(&op) {
                // The arguments and earlier inserted nodes all precede `at`, so the insertion
                // renumbers none of them.
                Some(&coefficients) => {
                    let z = insert_node(&mut candidate, at, Node::Affine { terms: vec![(argument, coefficients)], bias: None });
                    at += 1;
                    new_terms.push((z, library_op));
                }
                None => new_terms.push((argument, op)),
            }
        }
        candidate.nodes[at] = Node::Affine { terms: new_terms, bias };
        index = at + 1;
    }
    candidate.prune();
    candidate.interfaces()?;
    Ok(Some(candidate))
}

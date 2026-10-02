//! Per-input pieces trained through the model's own masked forward (#2951).
//!
//! # The program
//!
//! A *site* is a set of dense operators read as affine terms whose names agree up to a trailing
//! index (one projection's heads). As a whole it is the block matrix `W` from the concatenation of
//! the nodes it reads to the concatenation of the nodes it writes. A library of `C` rank-1 pieces
//! `W ≈ Σ_c u_c v_cᵀ` on the centred read `x − μ` replaces it in the program:
//!
//! ```text
//! z = Vᵀ(x − μ),   z̃ = z ⊙ m,   each written node: U_iᵀ z̃ + W_i μ in place of the site's terms,
//! ```
//!
//! with `m` a per-input binary mask (one raw slot per site): the masked forward is the program
//! itself, executed natively, and every derivative of it is exact ([`super::derivatives::vjp`]).
//!
//! # The code and its fit
//!
//! Each input pays the listing code of its active pieces (as in [`super::pieces`]) plus
//! `n KL(model ‖ masked)/ln 2`, the KL from the masked forward over every site at once. The fit
//! alternates:
//!
//! * **selection** — every mask entry's exact first derivative `g = ∂KL/∂m = (∂KL/∂z̃) z` and its
//!   Fisher diagonal `h` (from sampled-label reverse passes) predict what flipping it changes,
//!   `±g + h/2`; each input flips the entries whose predicted bits saved exceed the listing bits
//!   they add, the masked forward is run, and an input keeps its flips only when its exact bits
//!   saved exceed the listing bits they add (its own exact stopping rule; no threshold);
//! * **pieces** — the exact gradient of the total KL in every site's `V` and `U` given the masks,
//!   preconditioned by the read covariance and the written nodes' Fisher, stepped by
//!   backtracking on the exact total.

use super::derivatives::vjp;
use super::operator_program::{
    FamilyInputs, Interface, LabelKind, Node, Operator, OperatorBody, OperatorProgram, Provenance, Slot, SlotValues,
    Trace, remap_node,
};
use super::precision::DeclaredPrecision;
use gam_linalg::faer_ndarray::fast_atb;
use ndarray::{Array1, Array2, Axis, s};
use std::collections::BTreeMap;
use std::sync::Arc;

/// One site of a program (module note).
#[derive(Clone, Debug)]
pub struct Site {
    pub name: String,
    /// The nodes it reads and writes, in order, and its terms `(written index, read index, op)`.
    pub reads: Vec<usize>,
    pub writes: Vec<usize>,
    pub terms: Vec<(usize, usize, usize)>,
}

fn stem(name: &str) -> String {
    name.trim_end_matches(|c: char| c.is_ascii_digit()).to_string()
}

/// The sites of `program`: its dense, fully present operators that are only affine terms, grouped
/// by name up to a trailing index.
pub fn sites(program: &OperatorProgram) -> Vec<Site> {
    let mut only_terms = vec![true; program.operators.len()];
    for node in &program.nodes {
        match node {
            Node::Affine { bias: Some(b), .. } => only_terms[*b] = false,
            Node::Affine { bias: None, .. } => {}
            other => {
                for op in other.operators() {
                    only_terms[op] = false;
                }
            }
        }
    }
    let mut groups: BTreeMap<String, Site> = BTreeMap::new();
    for (index, node) in program.nodes.iter().enumerate() {
        let Node::Affine { terms, .. } = node else { continue };
        for (argument, op) in terms {
            let operator = &program.operators[*op];
            let dense = matches!(&operator.body, OperatorBody::Dense { present, .. } if present.iter().all(|k| *k));
            if !dense || !only_terms[*op] {
                continue;
            }
            let site = groups
                .entry(stem(&operator.name))
                .or_insert_with(|| Site { name: stem(&operator.name), reads: Vec::new(), writes: Vec::new(), terms: Vec::new() });
            if !site.reads.contains(argument) {
                site.reads.push(*argument);
            }
            if !site.writes.contains(&index) {
                site.writes.push(index);
            }
            let w = site.writes.iter().position(|n| *n == index).unwrap_or(0);
            let r = site.reads.iter().position(|n| n == argument).unwrap_or(0);
            site.terms.push((w, r, *op));
        }
    }
    groups.into_values().collect()
}

fn offsets(widths: &[usize]) -> Vec<usize> {
    let mut out = vec![0];
    for w in widths {
        out.push(out.last().copied().unwrap_or(0) + w);
    }
    out
}

/// The widths of a site's read and written nodes.
fn widths(program: &OperatorProgram, site: &Site) -> Result<(Vec<usize>, Vec<usize>), String> {
    let interfaces = program.interfaces().map_err(|e| e.to_string())?;
    Ok((site.reads.iter().map(|n| interfaces[*n].width()).collect(), site.writes.iter().map(|n| interfaces[*n].width()).collect()))
}

/// The site's block matrix `W` (written × read).
pub fn matrix(program: &OperatorProgram, site: &Site) -> Result<Array2<f64>, String> {
    let (reads, writes) = widths(program, site)?;
    let (ro, wo) = (offsets(&reads), offsets(&writes));
    let mut w = Array2::<f64>::zeros((wo[writes.len()], ro[reads.len()]));
    for &(i, j, op) in &site.terms {
        let block = program.operators[op].matrix();
        let mut target = w.slice_mut(s![wo[i]..wo[i + 1], ro[j]..ro[j + 1]]);
        target += &block;
    }
    Ok(w)
}

/// The concatenation of a site's read nodes' values (rows × d_in).
pub fn read_values(trace: &Trace, site: &Site) -> Result<Array2<f64>, String> {
    let views: Vec<_> = site.reads.iter().map(|n| trace.values[*n].view()).collect();
    ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())
}

/// A site's library: `v` is `C × d_in`, `u` is `C × d_out`, `mean` the read mean `μ`.
#[derive(Clone, Debug)]
pub struct Library {
    pub v: Array2<f64>,
    pub u: Array2<f64>,
    pub mean: Array1<f64>,
}

/// The program with its sites replaced by masked libraries (module note).
pub struct Masked {
    pub program: OperatorProgram,
    pub sites: Vec<Site>,
    pub libraries: Vec<Library>,
    /// Per site: its mask slot and its `z` and `z̃` nodes.
    pub slots: Vec<usize>,
    pub z: Vec<usize>,
    pub masked: Vec<usize>,
    /// Per site: its `V` operators (per read node), its centring bias, its `U` operators (per
    /// written node).
    v_ops: Vec<Vec<usize>>,
    centre_ops: Vec<usize>,
    u_ops: Vec<Vec<usize>>,
    read_offsets: Vec<Vec<usize>>,
    write_offsets: Vec<Vec<usize>>,
    /// Per site, its matrix `W`.
    pub w: Vec<Array2<f64>>,
    /// Per site and written node: the new node index and its new bias operator.
    written: Vec<Vec<usize>>,
}

fn fine() -> DeclaredPrecision {
    DeclaredPrecision::new(40).expect("a precision in range")
}

fn dense(name: String, rows: Interface, cols: Interface, values: Array2<f64>) -> Result<Arc<Operator>, String> {
    Ok(Arc::new(Operator::dense(name, rows, cols, values, fine(), Provenance::default()).map_err(|e| e.to_string())?))
}

impl Masked {
    /// Replace `sites` of `model` by `libraries` (one per site).
    pub fn build(model: &OperatorProgram, sites: Vec<Site>, libraries: Vec<Library>) -> Result<Self, String> {
        let interfaces = model.interfaces().map_err(|e| e.to_string())?;
        let mut program = model.clone();
        let base_slots = program.declarations.slots.len();
        let (mut slots, mut v_ops, mut centre_ops, mut u_ops) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut read_offsets, mut write_offsets, mut ws) = (Vec::new(), Vec::new(), Vec::new());
        // The new bias of every written node: its old bias plus each site's `W_i μ`.
        let mut biases: BTreeMap<usize, Array1<f64>> = BTreeMap::new();
        for (k, (site, library)) in sites.iter().zip(&libraries).enumerate() {
            let pieces = library.v.nrows();
            let coordinates = Interface::uniform(pieces, 1, LabelKind::Factor, 0).map_err(|e| e.to_string())?;
            program.declarations.slots.push(Slot::Raw { width: pieces });
            slots.push(base_slots + k);
            let (reads, writes) = widths(model, site)?;
            let (ro, wo) = (offsets(&reads), offsets(&writes));
            let first = site.writes.iter().copied().min().unwrap_or(0);
            if site.reads.iter().any(|r| *r >= first) {
                return Err(format!("{}: a read node follows a written one", site.name));
            }
            let mut vs = Vec::new();
            for (j, &node) in site.reads.iter().enumerate() {
                let block = library.v.slice(s![.., ro[j]..ro[j + 1]]).to_owned();
                program.operators.push(dense(format!("{}·V{j}", site.name), coordinates.clone(), interfaces[node].clone(), block)?);
                vs.push(program.operators.len() - 1);
            }
            let centre = (-library.v.dot(&library.mean)).insert_axis(Axis(1));
            program.operators.push(dense(format!("{}·centre", site.name), coordinates.clone(), Interface::constant(), centre)?);
            centre_ops.push(program.operators.len() - 1);
            let w = matrix(model, site)?;
            let mut us = Vec::new();
            for (i, &node) in site.writes.iter().enumerate() {
                let block = library.u.slice(s![.., wo[i]..wo[i + 1]]).t().to_owned();
                program.operators.push(dense(format!("{}·U{i}", site.name), interfaces[node].clone(), coordinates.clone(), block)?);
                us.push(program.operators.len() - 1);
                let fold = w.slice(s![wo[i]..wo[i + 1], ..]).dot(&library.mean);
                let entry = biases.entry(node).or_insert_with(|| match &model.nodes[node] {
                    Node::Affine { bias: Some(b), .. } => model.operators[*b].matrix().column(0).to_owned(),
                    _ => Array1::zeros(fold.len()),
                });
                *entry += &fold;
            }
            v_ops.push(vs);
            u_ops.push(us);
            read_offsets.push(ro);
            write_offsets.push(wo);
            ws.push(w);
        }
        let mut bias_ops: BTreeMap<usize, usize> = BTreeMap::new();
        for (node, values) in &biases {
            program.operators.push(dense(format!("bias of node {node}"), interfaces[*node].clone(), Interface::constant(), values.clone().insert_axis(Axis(1)))?);
            bias_ops.insert(*node, program.operators.len() - 1);
        }
        // Rebuild the nodes: each site's mask, z and z̃ just before its first written node.
        let identity_ops: Vec<usize> = (0..program.operators.len()).collect();
        let identity_bases: Vec<usize> = (0..program.bases.len()).collect();
        let identity_rules: Vec<usize> = (0..program.rules.len()).collect();
        let mut map = vec![0usize; model.nodes.len()];
        let mut nodes = Vec::new();
        let (mut z_nodes, mut masked_nodes) = (vec![0usize; sites.len()], vec![0usize; sites.len()]);
        for (index, node) in model.nodes.iter().enumerate() {
            for (k, site) in sites.iter().enumerate() {
                if site.writes.iter().copied().min() == Some(index) {
                    nodes.push(Node::Raw { slot: slots[k] });
                    let mask = nodes.len() - 1;
                    let terms = site.reads.iter().zip(&v_ops[k]).map(|(r, op)| (map[*r], *op)).collect();
                    nodes.push(Node::Affine { terms, bias: Some(centre_ops[k]) });
                    z_nodes[k] = nodes.len() - 1;
                    nodes.push(Node::Hadamard { left: z_nodes[k], right: mask });
                    masked_nodes[k] = nodes.len() - 1;
                }
            }
            let mut rebuilt = node.clone();
            remap_node(&mut rebuilt, &map, &identity_ops, &identity_bases, &identity_rules);
            if let Node::Affine { terms, bias } = &mut rebuilt {
                for (k, site) in sites.iter().enumerate() {
                    let Some(i) = site.writes.iter().position(|w| *w == index) else { continue };
                    let ops: Vec<usize> = site.terms.iter().filter(|(wi, _, _)| *wi == i).map(|(_, _, op)| *op).collect();
                    terms.retain(|(_, op)| !ops.contains(op));
                    terms.push((masked_nodes[k], u_ops[k][i]));
                }
                if let Some(op) = bias_ops.get(&index) {
                    *bias = Some(*op);
                }
            }
            nodes.push(rebuilt);
            map[index] = nodes.len() - 1;
        }
        program.nodes = nodes;
        program.output = map[model.output];
        program.interfaces().map_err(|e| e.to_string())?;
        let written = sites.iter().map(|site| site.writes.iter().map(|w| map[*w]).collect()).collect();
        let reads_mapped: Vec<Site> = sites
            .iter()
            .map(|site| Site { reads: site.reads.iter().map(|r| map[*r]).collect(), writes: site.writes.iter().map(|w| map[*w]).collect(), ..site.clone() })
            .collect();
        Ok(Self {
            program,
            sites: reads_mapped,
            libraries,
            slots,
            z: z_nodes,
            masked: masked_nodes,
            v_ops,
            centre_ops,
            u_ops,
            read_offsets,
            write_offsets,
            w: ws,
            written,
        })
    }

    /// Set site `k`'s library (same number of pieces) into the program.
    pub fn set_library(&mut self, k: usize, library: Library) -> Result<(), String> {
        let (ro, wo) = (&self.read_offsets[k], &self.write_offsets[k]);
        for (j, &op) in self.v_ops[k].iter().enumerate() {
            let old = &self.program.operators[op];
            let block = library.v.slice(s![.., ro[j]..ro[j + 1]]).to_owned();
            self.program.operators[op] = dense(old.name.clone(), old.rows.clone(), old.cols.clone(), block)?;
        }
        let op = self.centre_ops[k];
        let old = &self.program.operators[op];
        let centre = (-library.v.dot(&library.mean)).insert_axis(Axis(1));
        self.program.operators[op] = dense(old.name.clone(), old.rows.clone(), old.cols.clone(), centre)?;
        for (i, &op) in self.u_ops[k].iter().enumerate() {
            let old = &self.program.operators[op];
            let block = library.u.slice(s![.., wo[i]..wo[i + 1]]).t().to_owned();
            self.program.operators[op] = dense(old.name.clone(), old.rows.clone(), old.cols.clone(), block)?;
        }
        self.libraries[k] = library;
        Ok(())
    }

    /// `base` with the masks (one `rows × C` per site) in the sites' slots.
    pub fn family(&self, base: &FamilyInputs, masks: &[Array2<f64>]) -> FamilyInputs {
        let mut family = base.clone();
        family.slots.extend(masks.iter().map(|m| SlotValues::Raw(m.clone())));
        family
    }
}

/// `KL(p ‖ q)` per row between target logits and logits (rows × classes), and the cotangent of
/// the total in the logits, `q − p` per row.
pub fn kl(target: &Array2<f64>, logits: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    use rayon::prelude::*;
    let mut cotangent = Array2::<f64>::zeros(logits.dim());
    // Rows are independent: one per worker.
    let values: Vec<f64> = cotangent
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .map(|(r, mut row)| {
            let (p, q) = (softmax(target.row(r)), softmax(logits.row(r)));
            let mut total = 0.0;
            for c in 0..p.len() {
                if p[c] > 0.0 {
                    total += p[c] * (p[c].ln() - q[c].max(f64::MIN_POSITIVE).ln());
                }
                row[c] = q[c] - p[c];
            }
            total
        })
        .collect();
    (Array1::from(values), cotangent)
}

fn softmax(z: ndarray::ArrayView1<'_, f64>) -> Array1<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Array1<f64> = z.mapv(|v| (v - m).exp());
    let total = e.sum();
    e / total
}

/// One masked forward: per-input KL against `target`, the trace, and the KL's cotangent.
pub fn forward(masked: &Masked, family: &FamilyInputs, target: &Array2<f64>) -> Result<(Array1<f64>, Trace, Array2<f64>), String> {
    let trace = masked.program.execute(family, false).map_err(|e| e.to_string())?;
    let (values, cotangent) = kl(target, &trace.values[masked.program.output]);
    Ok((values, trace, cotangent))
}

/// Per site: `∂KL/∂m` (rows × C) and the gradients in `V` (C × d_in) and `U` (C × d_out), from one
/// reverse pass of `cotangent`.
pub fn gradients(
    masked: &Masked,
    family: &FamilyInputs,
    trace: &Trace,
    masks: &[Array2<f64>],
    cotangent: Array2<f64>,
) -> Result<Vec<(Array2<f64>, Array2<f64>, Array2<f64>)>, String> {
    let back = vjp(&masked.program, family, trace, cotangent).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for (k, site) in masked.sites.iter().enumerate() {
        let pieces = masked.libraries[k].v.nrows();
        let rows = trace.values[masked.z[k]].nrows();
        let zero = || Array2::<f64>::zeros((rows, pieces));
        let cot_masked = back[masked.masked[k]].clone().unwrap_or_else(zero);
        let z = &trace.values[masked.z[k]];
        let mask_gradient = &cot_masked * z;
        let cot_z = &cot_masked * &masks[k];
        let centred = &read_values(trace, site)? - &masked.libraries[k].mean;
        let v_gradient = fast_atb(&cot_z, &centred);
        let zm = &trace.values[masked.masked[k]];
        let written: Vec<Array2<f64>> = masked.written[k]
            .iter()
            .map(|n| back[*n].clone().unwrap_or_else(|| Array2::zeros(trace.values[*n].dim())))
            .collect();
        let views: Vec<_> = written.iter().map(|w| w.view()).collect();
        let cot_written = ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?;
        let u_gradient = fast_atb(zm, &cot_written);
        out.push((mask_gradient, v_gradient, u_gradient));
    }
    Ok(out)
}

/// A deterministic generator for label sampling.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The Fisher diagonal of every mask entry and the Fisher of every written node, from `samples`
/// sampled-label reverse passes: per site `(h: rows × C, F: d_out × d_out)`.
pub fn fisher(
    masked: &Masked,
    family: &FamilyInputs,
    trace: &Trace,
    samples: usize,
    seed: u64,
) -> Result<Vec<(Array2<f64>, Array2<f64>)>, String> {
    let logits = &trace.values[masked.program.output];
    let mut rng = XorShift(seed | 1);
    let mut out: Vec<(Array2<f64>, Array2<f64>)> = masked
        .sites
        .iter()
        .enumerate()
        .map(|(k, _)| {
            let rows = trace.values[masked.z[k]].nrows();
            let pieces = masked.libraries[k].v.nrows();
            let d_out = masked.libraries[k].u.ncols();
            (Array2::zeros((rows, pieces)), Array2::zeros((d_out, d_out)))
        })
        .collect();
    for _ in 0..samples {
        let mut cotangent = Array2::<f64>::zeros(logits.dim());
        for r in 0..logits.nrows() {
            let q = softmax(logits.row(r));
            let mut pick = rng.next();
            let mut label = q.len() - 1;
            for (c, p) in q.iter().enumerate() {
                if pick < *p {
                    label = c;
                    break;
                }
                pick -= p;
            }
            for c in 0..q.len() {
                cotangent[[r, c]] = q[c] - if c == label { 1.0 } else { 0.0 };
            }
        }
        let back = vjp(&masked.program, family, trace, cotangent).map_err(|e| e.to_string())?;
        for (k, (h, f)) in out.iter_mut().enumerate() {
            if let Some(c) = &back[masked.masked[k]] {
                let g = c * &trace.values[masked.z[k]];
                *h += &(&g * &g);
            }
            let written: Vec<Array2<f64>> = masked.written[k]
                .iter()
                .map(|n| back[*n].clone().unwrap_or_else(|| Array2::zeros(trace.values[*n].dim())))
                .collect();
            let views: Vec<_> = written.iter().map(|w| w.view()).collect();
            let cot = ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?;
            *f += &fast_atb(&cot, &cot);
        }
    }
    let rows = logits.nrows() as f64;
    for (h, f) in out.iter_mut() {
        *h /= samples as f64;
        *f /= samples as f64 * rows;
    }
    Ok(out)
}

/// Each piece's listing bits at its firing frequency in `masks` (half a count each added).
pub fn listing_costs(masks: &[Array2<f64>]) -> Vec<Array1<f64>> {
    costs_from_counts(&masks.iter().map(|m| m.sum_axis(Axis(0))).collect::<Vec<_>>())
}

/// Each piece's listing bits from its firing counts (half a count each added).
pub fn costs_from_counts(counts: &[Array1<f64>]) -> Vec<Array1<f64>> {
    let total: f64 = counts.iter().map(|c| c.sum() + 0.5 * c.len() as f64).sum();
    counts.iter().map(|c| c.mapv(|x| (total / (x + 0.5)).log2())).collect()
}

/// The listing bits of each input's sets (over every site at once).
pub fn listing_bits(masks: &[Array2<f64>], costs: &[Array1<f64>]) -> Array1<f64> {
    let rows = masks.first().map_or(0, |m| m.nrows());
    let mut out = Array1::<f64>::zeros(rows);
    for r in 0..rows {
        let mut k = 0usize;
        for (m, c) in masks.iter().zip(costs) {
            for (x, cost) in m.row(r).iter().zip(c.iter()) {
                if *x > 0.0 {
                    out[r] += cost;
                    k += 1;
                }
            }
        }
        out[r] -= (1..=k).map(|i| (i as f64).log2()).sum::<f64>();
    }
    out
}

/// The per-input code, in bits: listing plus `n KL/ln 2`.
pub fn code(kl: &Array1<f64>, listing: &Array1<f64>, observations: f64) -> Array1<f64> {
    listing + &(kl * (observations / std::f64::consts::LN_2))
}

/// The explanation code of a sequence's sets, conditional on context (module note, "The code and
/// its fit"): an input whose sequence has a previous input sends, for each piece on there, whether
/// it stays on (a Bernoulli at the piece's own stay rate), then lists its new pieces at their
/// frequencies among new activations; the first input of a sequence lists its whole set. Every
/// rate is the Krichevsky–Trofimov estimate from the counts of the sets selected so far, so the
/// code is a valid prefix code given those counts, and an always-on piece costs almost nothing.
pub struct Coder {
    /// Listing bits of a piece listed new.
    pub costs: Vec<Array1<f64>>,
    /// Per piece, the probability it stays on from one input to the next.
    pub stay: Vec<Array1<f64>>,
    /// Per input, the previous input of its sequence.
    pub previous: Vec<Option<usize>>,
}

/// The counts behind a [`Coder`]: per piece, transitions from on, and new activations.
#[derive(Clone, Debug)]
pub struct Context {
    pub stayed: Vec<Array1<f64>>,
    pub was_on: Vec<Array1<f64>>,
    pub new: Vec<Array1<f64>>,
}

impl Context {
    pub fn new(pieces: &[usize]) -> Self {
        let zeros = || pieces.iter().map(|p| Array1::zeros(*p)).collect::<Vec<_>>();
        Self { stayed: zeros(), was_on: zeros(), new: zeros() }
    }

    /// Fold one sequence's sets in.
    pub fn absorb(&mut self, masks: &[Array2<f64>], previous: &[Option<usize>]) {
        for (k, m) in masks.iter().enumerate() {
            for r in 0..m.nrows() {
                for c in 0..m.ncols() {
                    let now = m[[r, c]] > 0.0;
                    match previous[r] {
                        Some(p) if m[[p, c]] > 0.0 => {
                            self.was_on[k][c] += 1.0;
                            if now {
                                self.stayed[k][c] += 1.0;
                            }
                        }
                        _ => {
                            if now {
                                self.new[k][c] += 1.0;
                            }
                        }
                    }
                }
            }
        }
    }

    /// The counts of a grown library: each new piece starts with its original piece's counts
    /// (`origins[k][c]` per site).
    pub fn grown(&self, origins: &[Vec<usize>]) -> Self {
        let map = |counts: &[Array1<f64>]| -> Vec<Array1<f64>> {
            counts.iter().zip(origins).map(|(c, o)| o.iter().map(|&i| c[i]).collect()).collect()
        };
        Self { stayed: map(&self.stayed), was_on: map(&self.was_on), new: map(&self.new) }
    }

    /// The coder these counts give, for inputs whose previous inputs are `previous`.
    pub fn coder(&self, previous: Vec<Option<usize>>) -> Coder {
        let stay = self.stayed.iter().zip(&self.was_on).map(|(s, w)| (s + 0.5) / &(w + 1.0)).collect();
        Coder { costs: costs_from_counts(&self.new), stay, previous }
    }
}

impl Coder {
    /// Each input's explanation bits.
    pub fn bits(&self, masks: &[Array2<f64>]) -> Array1<f64> {
        let rows = masks.first().map_or(0, |m| m.nrows());
        let mut out = Array1::<f64>::zeros(rows);
        for r in 0..rows {
            let mut fresh = 0usize;
            for (k, m) in masks.iter().enumerate() {
                for c in 0..m.ncols() {
                    let now = m[[r, c]] > 0.0;
                    match self.previous[r] {
                        Some(p) if m[[p, c]] > 0.0 => {
                            let q = self.stay[k][c];
                            out[r] -= if now { q.log2() } else { (1.0 - q).log2() };
                        }
                        _ => {
                            if now {
                                out[r] += self.costs[k][c];
                                fresh += 1;
                            }
                        }
                    }
                }
            }
            out[r] -= (1..=fresh).map(|i| (i as f64).log2()).sum::<f64>();
        }
        out
    }

    /// The new pieces of input `r` (on there, not on at its previous input).
    fn fresh(&self, masks: &[Array2<f64>], r: usize) -> usize {
        masks
            .iter()
            .map(|m| (0..m.ncols()).filter(|&c| m[[r, c]] > 0.0 && self.previous[r].is_none_or(|p| m[[p, c]] <= 0.0)).count())
            .sum()
    }

    /// The change of input `r`'s own bits when entry `(k, c)` flips, with `fresh` new pieces now.
    fn marginal(&self, masks: &[Array2<f64>], r: usize, k: usize, c: usize, fresh: usize) -> f64 {
        let on = masks[k][[r, c]] > 0.0;
        match self.previous[r] {
            Some(p) if masks[k][[p, c]] > 0.0 => {
                let q = self.stay[k][c];
                let (stay, leave) = (-q.log2(), -(1.0 - q).log2());
                if on { leave - stay } else { stay - leave }
            }
            _ => {
                if on {
                    -(self.costs[k][c] - (fresh as f64).max(1.0).log2())
                } else {
                    self.costs[k][c] - ((fresh + 1) as f64).log2()
                }
            }
        }
    }
}

/// Each input's previous input in its sequence (none at a sequence's first position).
pub fn previous_inputs(inputs: &FamilyInputs) -> Vec<Option<usize>> {
    match &inputs.layout {
        Some(layout) => (0..inputs.rows)
            .map(|r| (r > 0 && layout.sequence[r - 1] == layout.sequence[r] && layout.position[r] > 0).then(|| r - 1))
            .collect(),
        None => vec![None; inputs.rows],
    }
}

/// Selection (module note): rounds of predicted flips, each input keeping its own only when its
/// exact code falls, until no input changes. `budget[r]` caps an input's flips per round: halved
/// when its flips are refused, doubled when kept. Returns the masks and the final exact KL.
pub fn select(
    masked: &Masked,
    base: &FamilyInputs,
    target: &Array2<f64>,
    mut masks: Vec<Array2<f64>>,
    coder: &Coder,
    observations: f64,
    samples: usize,
) -> Result<(Vec<Array2<f64>>, Array1<f64>), String> {
    let scale = observations / std::f64::consts::LN_2;
    let rows = base.rows;
    let mut budget = vec![usize::MAX; rows];
    let mut round = 0u64;
    let started = std::time::Instant::now();
    let mut curvature: Option<Vec<(Array2<f64>, Array2<f64>)>> = None;
    loop {
        let family = masked.family(base, &masks);
        let (kl_now, trace, cotangent) = forward(masked, &family, target)?;
        let grads = gradients(masked, &family, &trace, &masks, cotangent)?;
        // The Fisher diagonal only ranks proposals (the exact forward decides), so it is measured
        // on the first round and kept: it moves slowly with the masks, and its passes dominate a
        // round's cost.
        if curvature.is_none() {
            curvature = Some(fisher(masked, &family, &trace, samples, 0x5EED + round)?);
        }
        let curvature = curvature.as_ref().ok_or("no curvature")?;
        let listing_now = coder.bits(&masks);
        let before = code(&kl_now, &listing_now, observations);
        // Each input's predicted flips, best first, within its budget.
        let mut proposed = masks.clone();
        let mut flipped = vec![0usize; rows];
        for r in 0..rows {
            let fresh = coder.fresh(&masks, r);
            let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
            for (k, ((g, _, _), (h, _))) in grads.iter().zip(curvature.iter()).enumerate() {
                for c in 0..g.ncols() {
                    let on = masks[k][[r, c]] > 0.0;
                    let delta = if on { -1.0 } else { 1.0 };
                    let kl_change = g[[r, c]] * delta + 0.5 * h[[r, c]];
                    let listing_change = coder.marginal(&masks, r, k, c, fresh);
                    let net = scale * kl_change + listing_change;
                    if net < 0.0 {
                        candidates.push((net, k, c));
                    }
                }
            }
            candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
            for &(_, k, c) in candidates.iter().take(budget[r]) {
                proposed[k][[r, c]] = 1.0 - proposed[k][[r, c]];
                flipped[r] += 1;
            }
        }
        if flipped.iter().all(|f| *f == 0) {
            return Ok((masks, kl_now));
        }
        let family = masked.family(base, &proposed);
        let (kl_new, _, _) = forward(masked, &family, target)?;
        let after = code(&kl_new, &coder.bits(&proposed), observations);
        let mut kept = 0usize;
        let mut saved = 0.0;
        for r in 0..rows {
            if flipped[r] == 0 {
                continue;
            }
            if after[r] < before[r] {
                saved += before[r] - after[r];
                for k in 0..masks.len() {
                    masks[k].row_mut(r).assign(&proposed[k].row(r));
                }
                budget[r] = budget[r].saturating_mul(2).max(flipped[r]);
                kept += 1;
            } else {
                budget[r] = (flipped[r] / 2).max(1);
                if flipped[r] == 1 {
                    budget[r] = 0;
                }
            }
        }
        round += 1;
        log::info!(
            "selection round {round} ({:.0}s): {} inputs flipped {} entries, {kept} kept; code {:.1} -> {:.1} bits per input",
            started.elapsed().as_secs_f64(),
            flipped.iter().filter(|f| **f > 0).count(),
            flipped.iter().sum::<usize>(),
            before.sum() / rows as f64,
            after.sum() / rows as f64
        );
        // Done when no input can change, or when a round saves less than a bit per input.
        if (kept == 0 && budget.iter().all(|b| *b == 0)) || (kept > 0 && saved < rows as f64) {
            let family = masked.family(base, &masks);
            let (kl_final, _, _) = forward(masked, &family, target)?;
            return Ok((masks, kl_final));
        }
    }
}

/// The pieces' preconditioners as running means over the inputs stepped on: per site, the read
/// covariance and the written nodes' Fisher.
#[derive(Clone, Debug, Default)]
pub struct Running {
    pub covariances: Vec<Array2<f64>>,
    pub fishers: Vec<Array2<f64>>,
    pub rows: f64,
}

impl Running {
    /// Fold one batch's per-site matrices (means over its `rows` inputs) into the running means.
    pub fn absorb(&mut self, covariances: Vec<Array2<f64>>, fishers: Vec<Array2<f64>>, rows: f64) {
        if self.rows == 0.0 {
            self.covariances = covariances;
            self.fishers = fishers;
            self.rows = rows;
            return;
        }
        let total = self.rows + rows;
        let (old, new) = (self.rows / total, rows / total);
        for (running, batch) in self.covariances.iter_mut().zip(covariances) {
            *running = &*running * old + &(batch * new);
        }
        for (running, batch) in self.fishers.iter_mut().zip(fishers) {
            *running = &*running * old + &(batch * new);
        }
        self.rows = total;
    }
}

/// `(M + λ I)⁻¹` for a symmetric positive semidefinite `M`, with `λ = tr M / dim`: the matrix
/// shrunk halfway to the isotropic matrix of its own mean eigenvalue, so a direction the data barely
/// resolve is not amplified beyond the mean scale.
fn shrunk_inverse(m: &Array2<f64>) -> Result<Array2<f64>, String> {
    let mut sym = m.clone();
    let n = sym.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (sym[[i, j]] + sym[[j, i]]);
            sym[[i, j]] = v;
            sym[[j, i]] = v;
        }
    }
    let lambda = (0..n).map(|i| sym[[i, i]]).sum::<f64>() / n.max(1) as f64;
    let d = super::dense::eigh(sym.view(), gam_linalg::roundoff::SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    let mut scaled = d.vectors.clone();
    for (k, l) in d.values.iter().enumerate() {
        let shifted = l.max(0.0) + lambda;
        let inv = if shifted > 0.0 { 1.0 / shifted } else { 0.0 };
        scaled.column_mut(k).mapv_inplace(|x| x * inv);
    }
    Ok(scaled.dot(&d.vectors.t()))
}

/// One preconditioned step of every site's pieces on the exact total KL given the masks: the
/// direction is the gradient preconditioned by the shrunk read covariance (for `V`) and the shrunk
/// written Fisher (for `U`); its length is the Gauss–Newton minimiser along it, `⟨g, d⟩ / dᵀ H d`
/// with `dᵀ H d` from one forward tangent through the masked program, halved until the exact
/// total falls. Returns the total before and after the step, or `None` when no step lowered it.
pub fn step_pieces(
    masked: &mut Masked,
    base: &FamilyInputs,
    target: &Array2<f64>,
    masks: &[Array2<f64>],
    samples: usize,
    seed: u64,
    running: &mut Running,
) -> Result<Option<(f64, f64)>, String> {
    let family = masked.family(base, masks);
    let (kl_now, trace, cotangent) = forward(masked, &family, target)?;
    let total = kl_now.sum();
    let grads = gradients(masked, &family, &trace, masks, cotangent)?;
    let curvature = fisher(masked, &family, &trace, samples, seed)?;
    // The preconditioners are the running means over every input stepped on so far.
    let rows = trace.values[masked.program.output].nrows() as f64;
    let mut batch_covariances = Vec::new();
    for (k, site) in masked.sites.iter().enumerate() {
        let centred = &read_values(&trace, site)? - &masked.libraries[k].mean;
        batch_covariances.push(fast_atb(&centred, &centred) / rows);
    }
    running.absorb(batch_covariances, curvature.iter().map(|(_, f)| f.clone()).collect(), rows);
    let mut directions = Vec::new();
    let mut slope = 0.0;
    for k in 0..masked.sites.len() {
        let (_, v_gradient, u_gradient) = &grads[k];
        let dv = v_gradient.dot(&shrunk_inverse(&running.covariances[k])?);
        let du = u_gradient.dot(&shrunk_inverse(&running.fishers[k])?);
        slope += (v_gradient * &dv).sum() + (u_gradient * &du).sum();
        directions.push((dv, du));
    }
    if !(slope > 0.0) {
        return Ok(None);
    }
    // `dᵀ H d`: the output tangent of the direction, in each row's softmax Fisher.
    let mut tangents: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
    for (k, (dv, du)) in directions.iter().enumerate() {
        let (ro, wo) = (&masked.read_offsets[k], &masked.write_offsets[k]);
        for (j, &op) in masked.v_ops[k].iter().enumerate() {
            tangents.insert(op, dv.slice(s![.., ro[j]..ro[j + 1]]).to_owned());
        }
        tangents.insert(masked.centre_ops[k], (-dv.dot(&masked.libraries[k].mean)).insert_axis(Axis(1)));
        for (i, &op) in masked.u_ops[k].iter().enumerate() {
            tangents.insert(op, du.slice(s![.., wo[i]..wo[i + 1]]).t().to_owned());
        }
    }
    let output = super::derivatives::jvp(&masked.program, &family, &trace, &tangents).map_err(|e| e.to_string())?;
    let logits = &trace.values[masked.program.output];
    let mut quadratic = 0.0;
    for r in 0..logits.nrows() {
        let q = softmax(logits.row(r));
        let t = output.row(r);
        let mean: f64 = q.iter().zip(t.iter()).map(|(a, b)| a * b).sum();
        quadratic += q.iter().zip(t.iter()).map(|(a, b)| a * (b - mean) * (b - mean)).sum::<f64>();
    }
    let originals = masked.libraries.clone();
    let mut eta = if quadratic > 0.0 { slope / quadratic } else { 1.0 };
    let floor = eta * f64::EPSILON;
    while eta > floor {
        for (k, (dv, du)) in directions.iter().enumerate() {
            let library = Library { v: &originals[k].v - &(dv * eta), u: &originals[k].u - &(du * eta), mean: originals[k].mean.clone() };
            masked.set_library(k, library)?;
        }
        let (kl_trial, _, _) = forward(masked, &family, target)?;
        if kl_trial.sum() < total {
            return Ok(Some((total, kl_trial.sum())));
        }
        eta *= 0.5;
    }
    for (k, library) in originals.into_iter().enumerate() {
        masked.set_library(k, library)?;
    }
    Ok(None)
}

/// Grow a site's library by splitting every piece that its active inputs use in two ways
/// (module note): piece `c` with activation `a_t = v_c · (x_t − μ)` on the inputs `t` that list it
/// becomes `(u_c, v_c/2 + δ)` and `(u_c, v_c/2 − δ)`, so the pair sums to the piece (all pieces on
/// is unchanged). The inputs are split by the sign `s_t` of their leading principal coordinate
/// (away from `v_c`'s own), and `δ = β p` along that direction with `β` the least-squares fit of
/// `δ · (x_t − μ) ≈ s_t a_t / 2`: then one half carries the piece on each side and the other half
/// is nearly silent there, so the selection can list one where it listed both. Pieces listed by
/// fewer than two inputs are kept whole. Returns the grown library, the masks with both halves on
/// wherever the piece was on, and each new piece's original piece.
pub fn split(library: &Library, x: &Array2<f64>, mask: &Array2<f64>) -> (Library, Array2<f64>, Vec<usize>) {
    let (pieces, d_in) = library.v.dim();
    let centred = x - &library.mean;
    let mut v_rows: Vec<Array1<f64>> = Vec::new();
    let mut u_rows: Vec<Array1<f64>> = Vec::new();
    let mut columns: Vec<Array1<f64>> = Vec::new();
    let mut origin: Vec<usize> = Vec::new();
    for c in 0..pieces {
        let v = library.v.row(c).to_owned();
        let u = library.u.row(c).to_owned();
        let on = mask.column(c).to_owned();
        let members: Vec<usize> = (0..mask.nrows()).filter(|&t| on[t] > 0.0).collect();
        if members.len() < 2 {
            v_rows.push(v);
            u_rows.push(u);
            columns.push(on);
            origin.push(c);
            continue;
        }
        let y = centred.select(Axis(0), &members);
        let a = y.dot(&v);
        // The members' inputs away from v's own direction, and their leading principal direction
        // by power iteration from the residual of largest norm.
        let vv = v.dot(&v).max(f64::MIN_POSITIVE);
        let mut away = y.clone();
        for (mut row, at) in away.outer_iter_mut().zip(a.iter()) {
            row.scaled_add(-at / vv, &v);
        }
        let start = away.outer_iter().map(|r| r.dot(&r)).enumerate().fold((0, 0.0), |best, (i, n)| if n > best.1 { (i, n) } else { best }).0;
        let mut p = away.row(start).to_owned();
        for _ in 0..d_in.min(32) {
            let next = away.t().dot(&away.dot(&p));
            let norm = next.dot(&next).sqrt();
            if !(norm > 0.0) {
                break;
            }
            p = next / norm;
        }
        let projection = away.dot(&p);
        let (num, den) = projection.iter().zip(a.iter()).fold((0.0, 0.0), |(n, d), (q, at)| (n + q * q.signum() * at * 0.5, d + q * q));
        let beta = if den > 0.0 { num / den } else { 0.0 };
        let delta = &p * beta;
        v_rows.push(&v * 0.5 + &delta);
        v_rows.push(&v * 0.5 - &delta);
        u_rows.push(u.clone());
        u_rows.push(u);
        columns.push(on.clone());
        columns.push(on);
        origin.extend([c, c]);
    }
    let stack = |rows: &[Array1<f64>]| -> Array2<f64> {
        let width = rows.first().map_or(0, |r| r.len());
        Array2::from_shape_fn((rows.len(), width), |(i, j)| rows[i][j])
    };
    let masks = Array2::from_shape_fn((mask.nrows(), columns.len()), |(t, c)| columns[c][t]);
    (Library { v: stack(&v_rows), u: stack(&u_rows), mean: library.mean.clone() }, masks, origin)
}

//! A removal's least-squares compensation (#2951).
//!
//! When a removal proposal deletes functions of an MLP, the MLP's surviving functions take over the
//! deleted functions' output as far as their activations can express it. On `P`'s own states (the
//! training sequences run by `P` alone at the posterior mean), let `H` hold the activations of the
//! layer's functions (`φ(g_i·x + c_i)`, times `b_i·x` when gated; one column per function), `R` the
//! surviving functions, `K` the deleted ones and `U_K` the deleted functions' outputs (one row each).
//! The surviving outputs move by the minimum-norm least-squares solution
//!
//! `Δ = argmin_Δ ‖H_R Δ − H_K U_K‖² = H_R⁺ H_K U_K`,
//!
//! so on those states the MLP's output changes by `−(I − H_R H_R⁺) H_K U_K`, the part of the
//! deleted output that no combination of the surviving activations reproduces.
//!
//! A function's output is its own column of the MLP's output map, or, under a read–write tie
//! (`library_sharing::tie`), `c` times a later layer's gate row. A tied output is fixed by the tie:
//! the compensation never moves it (it is outside `R`), and when the function is deleted its output
//! `c a` enters `U_K`. Deleting the gate row a tie reads deletes the tied output with it. The compensated
//! removal is a proposal: removal accepts it only when it does not increase the code length `F` of
//! the composed explanation on the fixed collection (`library_mdl`'s removal step).
//!
//! Each MLP's Gram matrix `G = Hᵀ H` is formed once per removal round, in float64 from the
//! activations the device computes; a proposal solves `G_RR Δ = G_RK U_K` over the eigenvectors of
//! `G_RR` whose eigenvalues exceed the bound on the rounding of `G_RR`: the summation bound
//! `γ_T trace(G_RR)` of its `T`-term dot products plus the eigendecomposition's own band. A direction
//! within that bound is not resolved from zero, and the solution leaves the outputs unchanged along it.

use crate::{
    interchange::{self, Interchange},
    library_mdl::{Explanation, Posterior, sequence_family},
    operator_program::{Node, OperatorProgram},
    run_check::LayerNodes,
};
use gam_gpu::{gpu_error::GpuError, tensor::Storage};
use gam_linalg::{decompose::eigh, faer_ndarray::fast_ata, roundoff::SymmetricAssembly};
use ndarray::{Array2, Axis};
use std::collections::{BTreeMap, BTreeSet};

fn error(e: impl std::fmt::Display) -> String {
    format!("library compensation: {e}")
}

/// Where a function's output vector is held (indices into `Explanation::trainable`).
#[derive(Clone, Copy, Debug)]
enum Output {
    /// Column `column` of the MLP's output map: its own parameters, which the compensation moves.
    Column(usize),
    /// `c` times row `row` of the gate map `gate` of a later layer, `c` the single entry of `scale`
    /// (a read–write tie); `group` is that gate row's prior group.
    Tied { scale: usize, gate: usize, row: usize, group: usize },
}

/// One MLP's functions, where their outputs are held, and their activations' Gram matrix.
struct Mlp {
    /// Per function, its prior groups (its gate's, its up direction's when gated, and its output's
    /// or its tie's scale).
    functions: Vec<Vec<usize>>,
    /// The output operator's index in `Explanation::trainable`, and per function its output.
    output: usize,
    outputs: Vec<Output>,
    /// `Hᵀ H` over the functions' activations on every row it was formed from.
    gram: Array2<f64>,
}

/// The compensation of every MLP of a library explanation for one removal round.
pub struct Compensation {
    mlps: Vec<Mlp>,
    /// The rows (tokens) the Gram matrices sum over.
    rows: usize,
}

/// The operator of `program` named `name`.
fn operator(program: &OperatorProgram, name: &str) -> Result<usize, String> {
    program.operators.iter().position(|o| o.name == name).ok_or_else(|| error(format!("no operator {name}")))
}

impl Compensation {
    /// The Gram matrices of every MLP's activations on `sequences` run by `P` alone at
    /// `posterior`'s mean, `batch` sequences at a time; `experiments` holds `P` compiled for the
    /// fit, and its trainable operators hold the posterior mean afterwards.
    pub fn new(experiments: &mut Interchange, explanation: &Explanation, posterior: &Posterior, sequences: &[Vec<u32>], batch: usize) -> Result<Self, String> {
        if batch == 0 {
            return Err(error("positive batch required"));
        }
        let sites: Vec<LayerNodes> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let (flat, _, _) = interchange::sites(&explanation.artifact, &sites)?;
        let position: BTreeMap<usize, usize> = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
        let mut mlps = Vec::with_capacity(explanation.layers.len());
        let mut nodes = Vec::with_capacity(explanation.layers.len());
        for (l, layer) in explanation.layers.iter().enumerate() {
            let out = operator(&flat, &format!("library.l{l}.mlp.out"))?;
            // The activations are the node the output map reads (a term of the MLP's output, beside
            // the terms of its read–write ties).
            let node = flat
                .nodes
                .iter()
                .find_map(|n| match n {
                    Node::Affine { terms, bias: None } => terms.iter().find(|(_, op)| *op == out).map(|(input, _)| *input),
                    _ => None,
                })
                .ok_or_else(|| error(format!("layer {l}: no bias-free map applies the MLP's output")))?;
            let height = flat.operators[out].rows.width();
            let trainable = |op: usize| position.get(&op).copied().ok_or_else(|| error(format!("{}: not trainable", flat.operators[op].name)));
            let outputs = layer
                .functions
                .iter()
                .enumerate()
                .map(|(j, groups)| {
                    let group = &explanation.groups[*groups.last().ok_or_else(|| error("a function without groups"))?];
                    let [cell] = group.cells.as_slice() else {
                        return Err(error(format!("{}: an output group of several cells", group.name)));
                    };
                    if cell.operator == out && cell.cols.len() == 1 && cell.rows.len() == height {
                        return Ok(Output::Column(cell.cols.start));
                    }
                    // A read–write tie: its scale is `library.l{l}.mlp.tie{t}.f{j}.scale`, its scatter
                    // (`….scatter`) puts the scaled value at the gate row of layer `t` it reads.
                    let name = &flat.operators[cell.operator].name;
                    let tie = name
                        .strip_prefix(&format!("library.l{l}.mlp.tie"))
                        .and_then(|rest| rest.strip_suffix(&format!(".f{j}.scale")))
                        .and_then(|t| t.parse::<usize>().ok())
                        .ok_or_else(|| error(format!("{}: an output that is neither a column of the output map nor a tie's scale", group.name)))?;
                    let scatter = flat.operators[operator(&flat, &format!("library.l{l}.mlp.tie{tie}.f{j}.scatter"))?].matrix();
                    let row = scatter.column(0).iter().position(|v| *v != 0.0).ok_or_else(|| error(format!("{name}: a tie scattering to no row")))?;
                    let gate = operator(&flat, &format!("library.l{tie}.mlp.gate"))?;
                    let read = explanation.layers.get(tie).and_then(|t| t.functions.get(row)).and_then(|groups| groups.first()).copied();
                    let read = read.filter(|g| explanation.groups[*g].cells.iter().any(|c| c.operator == gate && c.rows == [row]));
                    let group = read.ok_or_else(|| error(format!("{name}: the gate row it reads is not its function's gate group")))?;
                    Ok(Output::Tied { scale: trainable(cell.operator)?, gate: trainable(gate)?, row, group })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let output = trainable(out)?;
            mlps.push(Mlp { functions: layer.functions.clone(), output, outputs, gram: Array2::zeros((layer.functions.len(), layer.functions.len())) });
            nodes.push(node);
        }
        experiments.load(&posterior.mean)?;
        let program = experiments.models().1.program;
        let device = program.device();
        // Products of f32 activations in float64 on the device when it holds float64 (CUDA, the
        // host), on the host otherwise (the Apple GPU).
        let wide = match device.with_storage(Storage::F64) {
            Ok(wide) => Some(wide),
            Err(GpuError::NoDeviceKernel { .. }) => None,
            Err(e) => return Err(error(e)),
        };
        // With a float64 device, each MLP's Gram matrix is summed there over every batch (one product
        // accumulating into it per batch) and read once.
        let mut sums = match &wide {
            Some(wide) => mlps.iter().map(|mlp| wide.zeros(mlp.gram.nrows(), mlp.gram.ncols()).map(Some).map_err(error)).collect::<Result<Vec<_>, _>>()?,
            None => (0..mlps.len()).map(|_| None).collect(),
        };
        let mut rows = 0;
        for chunk in sequences.chunks(batch) {
            let family = sequence_family(&chunk.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
            let trace = program.forward(&family)?;
            for ((mlp, node), sum) in mlps.iter_mut().zip(&nodes).zip(&mut sums) {
                let h = trace.value(*node)?;
                if h.cols() != mlp.gram.ncols() {
                    return Err(error("activations of another width than the MLP's functions"));
                }
                match (&wide, sum) {
                    (Some(wide), Some(sum)) => {
                        let h = wide.convert(h).map_err(error)?;
                        wide.gram_lower(sum, &h, 1.0).map_err(error)?;
                    }
                    _ => mlp.gram += &fast_ata(&device.download(h).map_err(error)?),
                }
            }
            rows += family.rows;
        }
        if let Some(wide) = &wide {
            for (mlp, sum) in mlps.iter_mut().zip(&sums) {
                let gram = wide.download(sum.as_ref().ok_or_else(|| error("a Gram sum missing"))?).map_err(error)?;
                // The update sums the lower triangle; mirror it.
                mlp.gram = Array2::from_shape_fn(gram.dim(), |(i, j)| if i >= j { gram[[i, j]] } else { gram[[j, i]] });
            }
        }
        Ok(Self { mlps, rows })
    }

    /// `posterior` with the groups `removed` removed and, in every MLP that loses functions it had,
    /// its surviving functions' outputs moved by the least-squares compensation (module note).
    pub fn proposal(&self, posterior: &Posterior, removed: &[usize]) -> Result<Posterior, String> {
        let gone: BTreeSet<usize> = removed.iter().copied().collect();
        let mut trial = posterior.clone();
        // `γ_T`, the relative bound on a `T`-term float64 dot product's rounding.
        let terms = self.rows as f64 * f64::EPSILON / 2.0;
        if terms >= 1.0 {
            return Err(error("too many rows for the rounding bound"));
        }
        let gamma = terms / (1.0 - terms);
        for mlp in &self.mlps {
            // A tied output also depends on the gate row it reads.
            let groups = |i: usize| mlp.functions[i].iter().copied().chain(match mlp.outputs[i] {
                Output::Tied { group, .. } => Some(group),
                Output::Column(_) => None,
            });
            let alive = |i: &usize| groups(*i).all(|g| posterior.active[g]);
            let hit = |i: &usize| groups(*i).any(|g| gone.contains(&g));
            let (deleted, kept): (Vec<usize>, Vec<usize>) = (0..mlp.functions.len()).filter(alive).partition(hit);
            let surviving: Vec<usize> = kept.into_iter().filter(|i| matches!(mlp.outputs[*i], Output::Column(_))).collect();
            if deleted.is_empty() || surviving.is_empty() {
                continue;
            }
            let outputs = &posterior.mean[mlp.output];
            // `U_K`: the deleted functions' outputs as rows.
            let mut deleted_outputs = Array2::zeros((deleted.len(), outputs.nrows()));
            for (mut row, i) in deleted_outputs.rows_mut().into_iter().zip(&deleted) {
                match mlp.outputs[*i] {
                    Output::Column(column) => row.assign(&outputs.column(column)),
                    Output::Tied { scale, gate, row: read, .. } => row.assign(&(&posterior.mean[gate].row(read) * posterior.mean[scale][[0, 0]])),
                }
            }
            let right = mlp.gram.select(Axis(0), &surviving).select(Axis(1), &deleted).dot(&deleted_outputs);
            let left = mlp.gram.select(Axis(0), &surviving).select(Axis(1), &surviving);
            let decomposition = eigh(left.view(), SymmetricAssembly::Mirrored, None).map_err(error)?;
            let floor = decomposition.band + gamma * left.diag().sum();
            let inverse = decomposition.map(|lambda| if lambda > floor { 1.0 / lambda } else { 0.0 });
            let change = inverse.dot(&right);
            let target = &mut trial.mean[mlp.output];
            for (row, i) in change.rows().into_iter().zip(&surviving) {
                if let Output::Column(column) = mlp.outputs[*i] {
                    let mut column = target.column_mut(column);
                    column += &row;
                }
            }
        }
        trial.remove(removed);
        Ok(trial)
    }
}

#[cfg(test)]
mod tests {
    use super::Compensation;
    use crate::{
        import::import_language_model,
        interchange::{self, Interchange},
        library_mdl::{self, Posterior},
        operator_program::SlotValues,
        run_check::{layer_nodes, split_sites},
    };
    use gam_gpu::tensor::Device;
    use ndarray::Array2;
    use std::collections::BTreeMap;

    /// `P`'s final normed stream on `sequences`, alone, at `posterior`'s mean.
    fn hidden(ic: &mut Interchange, posterior: &Posterior, sequences: &[Vec<u32>]) -> Array2<f64> {
        ic.load(&posterior.mean).expect("the means load");
        let program = ic.models().1.program;
        let family = library_mdl::sequence_family(&sequences.iter().map(Vec::as_slice).collect::<Vec<_>>()).expect("a family");
        let trace = program.forward(&family).expect("the forward pass");
        program.device().download(trace.value(program.hidden()).expect("the hidden value")).expect("the download")
    }

    /// Every group but the last (the output, or a tie's scale) of function `from` of layer
    /// `layer` copied onto function `to`, so their activations agree on every token.
    fn copy_reads(explanation: &library_mdl::Explanation, posterior: &mut Posterior, layer: usize, from: usize, to: usize) {
        let position: BTreeMap<usize, usize> = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
        let functions = &explanation.layers[layer].functions;
        for (a, b) in functions[from].iter().zip(&functions[to]).take(functions[from].len() - 1) {
            for (x, y) in explanation.groups[*a].cells.iter().zip(&explanation.groups[*b].cells) {
                for (rx, ry) in x.rows.iter().zip(&y.rows) {
                    for (cx, cy) in x.cols.clone().zip(y.cols.clone()) {
                        let value = posterior.mean[position[&x.operator]][[*rx, cx]];
                        posterior.mean[position[&y.operator]][[*ry, cy]] = value;
                    }
                }
            }
        }
    }

    /// The largest change of `P`'s stream between `posterior` and the proposal removing `removed`,
    /// with and without compensation, and the stream's largest value.
    fn changes(explanation: &library_mdl::Explanation, native: &crate::operator_program::OperatorProgram, posterior: &Posterior, sequences: &[Vec<u32>], removed: usize) -> (f64, f64, f64) {
        let device = Device::host();
        let sites: Vec<_> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let reads = interchange::library_reads(&explanation.artifact.program, sites.len()).expect("the reads");
        let mut ic = Interchange::new(&device, native, &sites, &explanation.artifact, &explanation.trainable, reads, 1 << 30, 64).expect("the experiments");
        let compensation = Compensation::new(&mut ic, explanation, posterior, sequences, 2).expect("the compensation");
        let trial = compensation.proposal(posterior, &[removed]).expect("the proposal");
        assert!(!trial.active[removed]);
        let mut plain = posterior.clone();
        plain.remove(&[removed]);
        let before = hidden(&mut ic, posterior, sequences);
        let largest = before.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let change = |p: &Posterior, ic: &mut Interchange| hidden(ic, p, sequences).iter().zip(&before).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        (change(&trial, &mut ic), change(&plain, &mut ic), largest)
    }

    fn tiny(tag: &str) -> (crate::operator_program::OperatorProgram, library_mdl::Explanation, Vec<Vec<u32>>) {
        let dir = crate::test_support::tiny_export(tag, 2);
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let explanation = library_mdl::explanation(&native, &layers).expect("the library");
        (native, explanation, sequences)
    }

    #[test]
    fn deleting_a_copy_of_a_surviving_function_leaves_the_explanation_unchanged() {
        let (native, explanation, sequences) = tiny("library_compensation_copy");
        let mut posterior = Posterior::new(&explanation, 1000).expect("the posterior");
        copy_reads(&explanation, &mut posterior, 0, 0, 1);
        let output = *explanation.layers[0].functions[1].last().expect("an output group");
        let (compensated, plain, largest) = changes(&explanation, &native, &posterior, &sequences, output);
        assert!(compensated <= 1e-9 * largest, "the compensated removal moved the stream by {compensated:e} (largest value {largest:e})");
        assert!(plain > 1e3 * compensated.max(f64::EPSILON * largest), "the plain removal moved the stream by only {plain:e}");
    }

    #[test]
    fn deleting_a_tied_function_moves_a_free_copy_of_it_and_leaves_the_explanation_unchanged() {
        use crate::library_sharing::{Tie, tie};
        let (native, start, sequences) = tiny("library_compensation_tie");
        // Function 3 of layer 0 writes 2.5 times the gate row of function 5 of layer 1, through a
        // read–write tie; function 4 reads what function 3 reads and has a free output.
        let tied = tie(&start, &[Tie { source: (0, 3), target: (1, 5), scale: 2.5 }]).expect("the tie");
        let mut posterior = Posterior::new(&tied, 1000).expect("the posterior");
        copy_reads(&tied, &mut posterior, 0, 3, 4);
        let scale = tied.groups.iter().position(|g| g.name == "library.l0.mlp.f3.tie").expect("the tie's scale group");
        let (compensated, plain, largest) = changes(&tied, &native, &posterior, &sequences, scale);
        assert!(compensated <= 1e-9 * largest, "the compensated removal moved the stream by {compensated:e} (largest value {largest:e})");
        assert!(plain > 1e3 * compensated.max(f64::EPSILON * largest), "the plain removal moved the stream by only {plain:e}");
    }
}

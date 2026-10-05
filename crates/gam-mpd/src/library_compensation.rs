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
//! deleted output that no combination of the surviving activations reproduces. The compensated
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
use gam_gpu::{
    gpu_error::GpuError,
    tensor::{Arithmetic, Op, Storage},
};
use gam_linalg::{decompose::eigh, faer_ndarray::fast_ata, roundoff::SymmetricAssembly};
use ndarray::{Array2, Axis};
use std::collections::{BTreeMap, BTreeSet};

fn error(e: impl std::fmt::Display) -> String {
    format!("library compensation: {e}")
}

/// One MLP's functions, where their outputs are held, and their activations' Gram matrix.
struct Mlp {
    /// Per function, its prior groups (its gate's, its up direction's when gated, its output's).
    functions: Vec<Vec<usize>>,
    /// The output operator's index in `Explanation::trainable`, and per function its column there.
    output: usize,
    columns: Vec<usize>,
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
            // The activations are the node the output map reads.
            let node = flat
                .nodes
                .iter()
                .find_map(|n| match n {
                    Node::Affine { terms, bias: None } if terms.len() == 1 && terms[0].1 == out => Some(terms[0].0),
                    _ => None,
                })
                .ok_or_else(|| error(format!("layer {l}: no bias-free map applies the MLP's output")))?;
            let height = flat.operators[out].rows.width();
            let columns = layer
                .functions
                .iter()
                .map(|groups| {
                    let group = &explanation.groups[*groups.last().ok_or_else(|| error("a function without groups"))?];
                    match group.cells.as_slice() {
                        [cell] if cell.operator == out && cell.cols.len() == 1 && cell.rows.len() == height => Ok(cell.cols.start),
                        _ => Err(error(format!("{}: an output group that is not one whole column of the output map", group.name))),
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let output = *position.get(&out).ok_or_else(|| error(format!("layer {l}: the output map is not trainable")))?;
            mlps.push(Mlp { functions: layer.functions.clone(), output, columns, gram: Array2::zeros((layer.functions.len(), layer.functions.len())) });
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
        let mut rows = 0;
        for chunk in sequences.chunks(batch) {
            let family = sequence_family(&chunk.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
            let trace = program.forward(&family)?;
            for (mlp, node) in mlps.iter_mut().zip(&nodes) {
                let h = trace.value(*node)?;
                let gram = match &wide {
                    Some(wide) => {
                        let h = wide.convert(h).map_err(error)?;
                        let mut gram = wide.zeros(h.cols(), h.cols()).map_err(error)?;
                        wide.gemm(&mut gram, 1.0, &h, Op::T, &h, Op::N, 0.0, Arithmetic::F64).map_err(error)?;
                        let gram = wide.download(&gram).map_err(error)?;
                        // The product's two triangles are separate sums; mirror the lower one.
                        Array2::from_shape_fn(gram.dim(), |(i, j)| if i >= j { gram[[i, j]] } else { gram[[j, i]] })
                    }
                    None => fast_ata(&device.download(h).map_err(error)?),
                };
                if gram.dim() != mlp.gram.dim() {
                    return Err(error("activations of another width than the MLP's functions"));
                }
                mlp.gram += &gram;
            }
            rows += family.rows;
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
            let alive = |i: &usize| mlp.functions[*i].iter().all(|g| posterior.active[*g]);
            let hit = |i: &usize| mlp.functions[*i].iter().any(|g| gone.contains(g));
            let (deleted, surviving): (Vec<usize>, Vec<usize>) = (0..mlp.functions.len()).filter(alive).partition(hit);
            if deleted.is_empty() || surviving.is_empty() {
                continue;
            }
            let outputs = &posterior.mean[mlp.output];
            // `U_K`: the deleted functions' outputs as rows.
            let deleted_outputs = outputs.select(Axis(1), &deleted.iter().map(|i| mlp.columns[*i]).collect::<Vec<_>>()).reversed_axes();
            let right = mlp.gram.select(Axis(0), &surviving).select(Axis(1), &deleted).dot(&deleted_outputs);
            let left = mlp.gram.select(Axis(0), &surviving).select(Axis(1), &surviving);
            let decomposition = eigh(left.view(), SymmetricAssembly::Mirrored, None).map_err(error)?;
            let floor = decomposition.band + gamma * left.diag().sum();
            let inverse = decomposition.map(|lambda| if lambda > floor { 1.0 / lambda } else { 0.0 });
            let change = inverse.dot(&right);
            let target = &mut trial.mean[mlp.output];
            for (row, i) in change.rows().into_iter().zip(&surviving) {
                let mut column = target.column_mut(mlp.columns[*i]);
                column += &row;
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

    #[test]
    fn deleting_a_copy_of_a_surviving_function_leaves_the_explanation_unchanged() {
        let dir = crate::test_support::tiny_export("library_compensation_copy", 2);
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let explanation = library_mdl::explanation(&native, &layers).expect("the library");
        let mut posterior = Posterior::new(&explanation, 1000).expect("the posterior");
        // Function 1 of layer 0's MLP becomes a copy of function 0: every group but the output
        // takes function 0's values, so the two activations agree on every token.
        let position: BTreeMap<usize, usize> = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
        let functions = &explanation.layers[0].functions;
        for (from, to) in functions[0].iter().zip(&functions[1]).take(functions[0].len() - 1) {
            for (a, b) in explanation.groups[*from].cells.iter().zip(&explanation.groups[*to].cells) {
                for (ra, rb) in a.rows.iter().zip(&b.rows) {
                    for (ca, cb) in a.cols.clone().zip(b.cols.clone()) {
                        let value = posterior.mean[position[&a.operator]][[*ra, ca]];
                        posterior.mean[position[&b.operator]][[*rb, cb]] = value;
                    }
                }
            }
        }
        let device = Device::host();
        let sites: Vec<_> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let reads = interchange::library_reads(&explanation.artifact.program, sites.len()).expect("the reads");
        let mut ic = Interchange::new(&device, &native, &sites, &explanation.artifact, &explanation.trainable, reads, 1 << 30, 64).expect("the experiments");
        let compensation = Compensation::new(&mut ic, &explanation, &posterior, &sequences, 2).expect("the compensation");
        let output = *functions[1].last().expect("an output group");
        let trial = compensation.proposal(&posterior, &[output]).expect("the proposal");
        assert!(!trial.active[output]);
        let (before, after) = (hidden(&mut ic, &posterior, &sequences), hidden(&mut ic, &trial, &sequences));
        let largest = before.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let difference = before.iter().zip(&after).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(difference <= 1e-9 * largest, "the compensated removal moved the stream by {difference:e} (largest value {largest:e})");
        // Without compensation the same removal moves it.
        let mut plain = posterior.clone();
        plain.remove(&[output]);
        let moved = hidden(&mut ic, &plain, &sequences).iter().zip(&before).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(moved > 1e3 * difference.max(f64::EPSILON * largest), "the plain removal moved the stream by only {moved:e}");
    }
}

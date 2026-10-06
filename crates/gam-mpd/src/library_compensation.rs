//! A removal's compensation (#2951).
//!
//! When a removal proposal deletes functions of an MLP, the MLP's surviving functions take over the
//! deleted functions' output as far as their activations can express it and the posterior allows.
//! On `P`'s own states (the training sequences run by `P` alone at the posterior mean), let `H` hold
//! the activations of the layer's functions (`φ(g_i·x + c_i)`, times `b_i·x` when gated; one column
//! per function) with Gram matrix `G = Hᵀ H`, `R` the surviving functions, `K` the deleted ones and
//! `U_K` the deleted functions' outputs (one row each). Take the data term's curvature in the
//! outputs as `G ⊗ C`, `C = diag(c_r)` over the output coordinates `r` (the downstream metric does
//! not depend on which function writes), and each surviving output column `j` under its group's
//! prior `N(0, v_j I)`. To second order about a converged posterior (where the data term's gradient
//! in a mean is `−μ/v` and cancels the prior's), moving the surviving outputs by `Δ` while the
//! deleted ones go to zero changes `F` by
//!
//! `Σ_r ½ c_r ‖H_R Δ_r − H_K U_K,r‖² + Σ_j ‖Δ_j‖² / (2 v_j)`,
//!
//! minimized per coordinate by the posterior mode `Δ_r = (G_RR + Λ_r)⁻¹ G_RK U_K,r`,
//! `Λ_r = diag(1 / (c_r v_j))`: the outputs move only as far as the data support it. With
//! `S = V_R^½ G_RR V_R^½ = Q diag(e) Qᵀ` (`V_R = diag(v_j)`), one eigendecomposition serves every
//! coordinate: `Δ_r = V_R^½ Q diag(c_r / (1 + c_r e_k)) Qᵀ V_R^½ G_RK U_K,r`. As `c_r → ∞` this is
//! the least-squares solution `G_RR⁺ G_RK U_K,r`, which on ill-conditioned `G_RR` (a deleted
//! function nearly in the survivors' span) moves the outputs by huge, cancelling amounts that hold
//! on `P`'s own states only.
//!
//! `c_r` comes from the posterior: IVON's deviation of an output entry is `1/σ_rj² = N h_rj + 1/v_j`
//! with `N h_rj` the data term's curvature in it, which under `G ⊗ C` is `c_r G_jj`. Over the alive
//! functions `A` with own output columns, `c_r = max(0, Σ_{j∈A} (1/σ_rj² − 1/v_j)) / Σ_{j∈A} G_jj`,
//! the ratio that matches the curvatures' sum. `v_j = (1/|G|) Σ (μ² + σ²)` over the column, the
//! group's empirical-Bayes variance. `c_r G` is thus in the units of `N h`, the curvature of the
//! data term over all `N` scored tokens (`G` sums over the rows it was formed from, `c_r` per unit
//! of that sum), and so are the precisions below: no conversion enters between them and the fit's
//! `σ² = 1/(N (h + δ))`, `δ = 1/(N v)`.
//!
//! The compensated posterior is this model's conditional one. With the deleted outputs at zero,
//! the surviving output entries of coordinate `r` are Gaussian with precision
//! `c_r G_RR + V_R⁻¹ = c_r (G_RR + Λ_r)` and mean the posterior mode above. Its mean-field
//! approximation (the diagonal Gaussian nearest it in `KL(q ‖ ·)`) has that mode for its means and
//! the reciprocals of the precision's diagonal, `1/(c_r G_jj + 1/v_j)`, for its variances (not the
//! covariance's diagonal). That diagonal does not involve the deleted functions: a deletion moves
//! the conditional mean and leaves each survivor's mean-field variance where it was before it. The
//! posterior holds a measurement of that variance per entry, `σ_rj² = 1/(N h_rj + 1/v_j)`, of
//! which `1/(c_r G_jj + 1/v_j)` is the Kronecker model's fit, so a proposal moves the survivors'
//! means and keeps their `σ`. Writing the model's value instead would move each entry's `σ` off its
//! stationary point in `F` (`σ² = 1/(N (h + δ))`) by the model's error, raising `F` by
//! `½ (ρ − 1 − ln ρ) ≥ 0` per entry, `ρ` the ratio of the two variances.
//!
//! A function's output is its own column of the MLP's output map, or, under a read–write tie
//! (`library_sharing::tie`), `c` times a later layer's gate row. A tied output is fixed by the tie:
//! the compensation never moves it (it is outside `R`), and when the function is deleted its output
//! `c a` enters `U_K`. Deleting the gate row a tie reads deletes the tied output with it. The compensated
//! removal is a proposal: removal accepts it only when it does not increase the code length `F` of
//! the composed explanation on the fixed collection (`library_mdl`'s removal step).
//!
//! Each MLP's Gram matrix `G` is formed once per removal round, in float64 from the activations the
//! device computes, and held through the round as its lower triangle. An eigenvalue `e_k` of `S`
//! within the bound on its rounding (the summation bound `γ_T trace(S)` of its `T`-term dot
//! products plus the eigendecomposition's own band) is not resolved from zero: a proposal leaves
//! the outputs unchanged along it, and the share below takes it as zero.
//!
//! Memory, for an MLP of `n` functions of which `m` are decomposed, `d` output coordinates and
//! float64 values: the packed Grams take `4 n (n + 1)` bytes per MLP through the round. While they
//! are formed, a float64 device (CUDA) sums each MLP's Gram in an `n × n` matrix (`8 n²` bytes on
//! the device, and on the host while it is read and packed), as many MLPs' in one pass over the
//! sequences as the device's free memory holds beside one batch's trace; the Apple GPU's activations are summed on the host, one batch's activations and
//! their `8 n²`-byte product at a time. The ranking and each proposal decompose one MLP's scaled
//! Gram `S` (`8 m²` bytes) at a time, which the decomposition consumes: on the host faer copies its
//! lower triangle and adds its divide-and-conquer workspace and the eigenvectors, `40 m²` bytes, of
//! which the eigenvectors' `8 m²` stay; on CUDA the host holds `S`, the copy staged for the upload
//! and the eigenvectors as downloaded and as returned (`32 m²` bytes), and the device the matrix
//! and cuSOLVER's workspace. A move multiplies by the block of `G` between the survivors and the
//! deleted functions, `8 m (n − m)` bytes, and holds `O(n d)` more. No use is an exact solve from
//! products with `G`: each coordinate's move solves with its own shift `1/c_r`, which one
//! eigendecomposition serves exactly for every coordinate (its unresolved directions dropped) where
//! an iterative solve from products would stop at a tolerance, and the rankings read diagonals of
//! spectral functions of `S`.

use crate::{
    interchange::{self, Interchange},
    library_mdl::{Explanation, Posterior, sequence_family},
    operator_program::{FamilyInputs, Node, OperatorProgram},
    run_check::LayerNodes,
};
use faer::Side;
use gam_gpu::{
    gpu_error::GpuError,
    tensor::{Device, Storage, Tensor},
};
use gam_linalg::{
    decompose::Eigh,
    faer_ndarray::{FaerArrayView, fast_ab, fast_ata, fast_atb, self_adjoint_evd},
    roundoff::symmetric_spectrum_rounding_band,
};
use ndarray::{Array1, Array2, ShapeBuilder};
use rayon::prelude::*;
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

/// A symmetric matrix of order `order` held as its lower triangle row by row: entry `(i, j)`,
/// `i ≥ j`, at `i (i + 1) / 2 + j`, `order (order + 1) / 2` values.
struct Packed {
    order: usize,
    values: Vec<f64>,
}

impl Packed {
    fn zeros(order: usize) -> Self {
        Self { order, values: vec![0.0; order * (order + 1) / 2] }
    }

    /// Entry `(i, j)`.
    fn at(&self, i: usize, j: usize) -> f64 {
        let (i, j) = if i >= j { (i, j) } else { (j, i) };
        self.values[i * (i + 1) / 2 + j]
    }

    /// Adds the lower triangle of the `order × order` matrix `full`.
    fn add_lower(&mut self, full: &Array2<f64>) -> Result<(), String> {
        if full.dim() != (self.order, self.order) {
            return Err(error(format!("a {:?} matrix added to a Gram of order {}", full.dim(), self.order)));
        }
        let mut start = 0;
        for (i, row) in full.rows().into_iter().enumerate() {
            for (value, add) in self.values[start..=start + i].iter_mut().zip(row) {
                *value += add;
            }
            start += i + 1;
        }
        Ok(())
    }

    /// Rows `rows` and columns `cols` as a dense matrix, entry `(a, b)` times `weight(a, b)`.
    fn block(&self, rows: &[usize], cols: &[usize], weight: impl Fn(usize, usize) -> f64 + Sync) -> Result<Array2<f64>, String> {
        let mut values = vec![0.0; rows.len() * cols.len()];
        if !cols.is_empty() {
            values.par_chunks_mut(cols.len()).enumerate().for_each(|(a, line)| {
                for (b, value) in line.iter_mut().enumerate() {
                    *value = self.at(rows[a], cols[b]) * weight(a, b);
                }
            });
        }
        Array2::from_shape_vec((rows.len(), cols.len()), values).map_err(error)
    }
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
    gram: Packed,
}

/// One MLP's compensation in a proposal: per surviving function its output column (none for a
/// tied output), the scale `√v_j` and the row of its move, which adds `v_j^½ · row` to the column.
struct Change {
    output: usize,
    columns: Vec<Option<usize>>,
    scales: Vec<f64>,
    rows: Array2<f64>,
}

/// The compensation of every MLP of a library explanation for one removal round.
pub struct Compensation {
    mlps: Vec<Mlp>,
    /// The rows (tokens) the Gram matrices sum over.
    rows: usize,
    /// `P`'s device, which decomposes the scaled Gram matrices where it holds float64 (CUDA).
    device: Device,
}

/// The int8 slices of the split Gram (`Device::gram_split`) and the batch rows it takes: with 9
/// slices its error stays below the float64 product's bound for up to 2^14 rows, and its int32
/// sums hold 2^13 rows.
const GRAM_SLICES: usize = 9;
const MAX_SPLIT_ROWS: usize = 1 << 13;

/// The operator of `program` named `name`.
fn operator(program: &OperatorProgram, name: &str) -> Result<usize, String> {
    program.operators.iter().position(|o| o.name == name).ok_or_else(|| error(format!("no operator {name}")))
}

/// The activations `h` of an MLP of `order` functions on the batch `family`, with the rows at the
/// positions `elsewhere` (where a position select applies `M`'s MLP instead of the functions)
/// zeroed in a copy; none when there are no such positions.
fn elsewhere_zeroed(device: &Device, h: &Tensor, family: &FamilyInputs, elsewhere: &[u32], order: usize) -> Result<Option<Tensor>, String> {
    if h.cols() != order {
        return Err(error("activations of another width than the MLP's functions"));
    }
    if elsewhere.is_empty() {
        return Ok(None);
    }
    let positions = family.layout.as_ref().map(|layout| layout.position.as_slice()).unwrap_or_default();
    if positions.len() != h.rows() {
        return Err(error("a position select on a batch without its positions"));
    }
    let mut copy = device.copy(h).map_err(error)?;
    let zero = device.zeros(1, h.cols()).map_err(error)?;
    for (row, _) in positions.iter().enumerate().filter(|(_, p)| elsewhere.contains(p)) {
        device.set_rows(&mut copy, row, &zero).map_err(error)?;
    }
    Ok(Some(copy))
}

impl Compensation {
    /// The Gram matrices of every MLP's activations on `sequences` run by `P` alone at
    /// `posterior`'s mean, `batch` sequences at a time, on a float64 device as many MLPs per pass
    /// over the sequences as its free memory holds (module note); `experiments` holds `P` compiled
    /// for the fit, and its trainable operators hold the posterior mean afterwards.
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
            // An MLP the explanation leaves to `M` (`library_mdl::scoped`) has nothing to compensate.
            if layer.functions.is_empty() {
                continue;
            }
            let out = operator(&flat, &format!("library.l{l}.mlp.out"))?;
            // The activations are the node the output map reads (a term of the MLP's output, beside
            // the terms of its read–write ties, and a fixed bias in a transcoder block, which the
            // compensation never moves).
            let (applied, node) = flat
                .nodes
                .iter()
                .enumerate()
                .find_map(|(a, n)| match n {
                    Node::Affine { terms, .. } => terms.iter().find(|(_, op)| *op == out).map(|(input, _)| (a, *input)),
                    _ => None,
                })
                .ok_or_else(|| error(format!("layer {l}: no map applies the MLP's output")))?;
            // A position select that writes another node at some positions in place of the output
            // (a transcoder block runs `M`'s MLP at the first token): the functions' activations
            // there reach nothing, so the Gram matrix leaves those rows out.
            let mut elsewhere = Vec::new();
            for n in &flat.nodes {
                match n {
                    Node::Select { inside, .. } if *inside == applied => return Err(error(format!("layer {l}: the MLP's output is selected at given positions only"))),
                    Node::Select { outside, positions, .. } if *outside == applied => elsewhere.clone_from(positions),
                    _ => {}
                }
            }
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
            mlps.push(Mlp { functions: layer.functions.clone(), output, outputs, gram: Packed::zeros(layer.functions.len()) });
            nodes.push((node, elsewhere));
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
        let families = || sequences.chunks(batch).map(|chunk| sequence_family(&chunk.iter().map(Vec::as_slice).collect::<Vec<_>>()));
        match &wide {
            // Each MLP's Gram summed on the device over every batch (one product accumulating into
            // it per batch), read once, its lower triangle packed, and its device sum freed. A pass
            // over the sequences sums as many MLPs' Grams as the device's free memory holds while
            // one batch's trace is held (`8 n²` bytes each), measured on the first batch: one pass
            // when every MLP's fits, as on Qwen3-0.6B's 28 MLPs of 3072 functions (2.1 GB).
            Some(wide) => {
                let mut start = 0;
                while start < mlps.len() {
                    let mut group = start..start;
                    let mut sums: Vec<Tensor> = Vec::new();
                    for (index, family) in families().enumerate() {
                        let family = family?;
                        let trace = program.forward(&family)?;
                        if index == 0 {
                            // The group: the MLPs whose sums fit the free memory beside the trace.
                            let free = wide.memory().map_err(error)?.map(|(free, _)| free);
                            let mut used = 0usize;
                            let mut end = start;
                            while end < mlps.len() {
                                let bytes = 8 * mlps[end].gram.order * mlps[end].gram.order;
                                if end > start && free.is_some_and(|free| used + bytes > free) {
                                    break;
                                }
                                used += bytes;
                                end += 1;
                            }
                            group = start..end;
                            sums = mlps[group.clone()].iter().map(|mlp| wide.zeros(mlp.gram.order, mlp.gram.order).map_err(error)).collect::<Result<_, _>>()?;
                        }
                        for (k, sum) in group.clone().zip(&mut sums) {
                            let (node, elsewhere) = &nodes[k];
                            let zeroed = elsewhere_zeroed(device, trace.value(*node)?, &family, elsewhere, mlps[k].gram.order)?;
                            let h = match &zeroed {
                                Some(copy) => copy,
                                None => trace.value(*node)?,
                            };
                            // On CUDA the f32 activations' Gram on the integer tensor cores
                            // (`Device::gram_split`, 9 slices: within γ_rows √(G_ii G_jj) per entry,
                            // the form of the float64 product's own bound that the floor below uses),
                            // else the float64 product; either sums the lower triangle.
                            if h.rows() > MAX_SPLIT_ROWS || !wide.gram_split(sum, h, GRAM_SLICES).map_err(error)? {
                                let h = wide.convert(h).map_err(error)?;
                                wide.gram_lower(sum, &h, 1.0).map_err(error)?;
                            }
                        }
                    }
                    if group.is_empty() {
                        return Err(error("no sequences to form the Gram matrices from"));
                    }
                    for (k, sum) in group.clone().zip(sums) {
                        mlps[k].gram.add_lower(&wide.download(&sum).map_err(error)?)?;
                    }
                    start = group.end;
                }
            }
            None => {
                for family in families() {
                    let family = family?;
                    let trace = program.forward(&family)?;
                    for (mlp, (node, elsewhere)) in mlps.iter_mut().zip(&nodes) {
                        let zeroed = elsewhere_zeroed(device, trace.value(*node)?, &family, elsewhere, mlp.gram.order)?;
                        let h = match &zeroed {
                            Some(copy) => copy,
                            None => trace.value(*node)?,
                        };
                        mlp.gram.add_lower(&fast_ata(&device.download(h).map_err(error)?))?;
                    }
                }
            }
        }
        let rows: usize = sequences.iter().map(Vec::len).sum();
        Ok(Self { mlps, rows, device: device.clone() })
    }

    /// `γ_T`, the relative bound on a `T`-term float64 dot product's rounding over the Gram's rows.
    fn gamma(&self) -> Result<f64, String> {
        let terms = self.rows as f64 * f64::EPSILON / 2.0;
        if terms >= 1.0 {
            return Err(error("too many rows for the rounding bound"));
        }
        Ok(terms / (1.0 - terms))
    }

    /// The functions of `mlp` alive at `posterior` with their own output columns.
    fn columns(mlp: &Mlp, posterior: &Posterior) -> Vec<usize> {
        (0..mlp.functions.len()).filter(|i| matches!(mlp.outputs[*i], Output::Column(_)) && mlp.functions[*i].iter().all(|g| posterior.active[*g])).collect()
    }

    /// Per function of `mlp` (zero for one without its own column), its output group's prior
    /// variance `v_j = (1/|G|) Σ (μ² + σ²)`, and per output coordinate the data term's curvature
    /// per unit of activation energy `c_r` estimated over the functions `alive` (module note).
    fn scales(mlp: &Mlp, posterior: &Posterior, alive: &[usize]) -> (Vec<f64>, Array1<f64>) {
        let means = &*posterior.mean[mlp.output];
        let variances = posterior.marginal_variances(mlp.output);
        let mut prior = vec![0.0; mlp.functions.len()];
        let mut precision = Array1::<f64>::zeros(means.nrows());
        let mut energy = 0.0;
        for i in alive {
            let Output::Column(column) = mlp.outputs[*i] else { continue };
            let (mean, variance) = (means.column(column), variances.column(column));
            let v = mean.iter().zip(&variance).map(|(m, s)| m * m + s).sum::<f64>() / mean.len() as f64;
            if !(v > 0.0 && v.is_finite()) {
                continue;
            }
            prior[*i] = v;
            // An entry held exactly (zero deviation) carries no curvature estimate.
            precision.zip_mut_with(&variance, |p, s| {
                if *s > 0.0 {
                    *p += 1.0 / s - 1.0 / v;
                }
            });
            energy += mlp.gram.at(*i, *i);
        }
        let curvature = precision.mapv(|p| if energy > 0.0 && p > 0.0 { p / energy } else { 0.0 });
        (prior, curvature)
    }

    /// `S = V^½ G_ss V^½` over the functions `set` with prior variances `prior`, its
    /// eigendecomposition and the floor below which an eigenvalue is not resolved from zero.
    fn scaled(&self, mlp: &Mlp, set: &[usize], prior: &[f64]) -> Result<(Vec<f64>, Eigh, f64), String> {
        let root: Vec<f64> = set.iter().map(|i| prior[*i].sqrt()).collect();
        let scaled = mlp.gram.block(set, set, |a, b| root[a] * root[b])?;
        let trace = scaled.diag().sum();
        let decomposition = self.decompose(scaled)?;
        let floor = decomposition.band + self.gamma()? * trace;
        Ok((root, decomposition, floor))
    }

    /// The eigendecomposition of the symmetric `s` (its lower triangle read), which it consumes
    /// and frees once decomposed: on CUDA by cuSOLVER (`Device::symmetric_eigh`), else on the host
    /// by faer (`self_adjoint_evd`, the decomposition `gam_linalg::decompose::eigh` makes, without
    /// its copies of the input and the eigenvectors), within the backward-error band
    /// `symmetric_spectrum_rounding_band` states. Its uses are spectral functions, which neither
    /// the eigenpairs' order nor the eigenvectors' signs change.
    fn decompose(&self, s: Array2<f64>) -> Result<Eigh, String> {
        if s.iter().any(|v| !v.is_finite()) {
            return Err(error("a scaled Gram with a nonfinite entry"));
        }
        let on_device = self.device.symmetric_eigh(s.view()).map_err(error)?;
        let (values, vectors) = match on_device {
            Some(pair) => {
                drop(s);
                pair
            }
            None => {
                let (values, vectors) = self_adjoint_evd(FaerArrayView::new(&s).as_ref(), Side::Lower).map_err(|e| error(format!("eigendecomposition: {e:?}")))?;
                drop(s);
                let order = vectors.nrows();
                let values = Array1::from_shape_fn(order, |k| values.as_ref().column_vector()[k]);
                // faer's columns, contiguous, become the columns of a column-major array.
                let q = vectors.as_ref();
                let mut columns = Vec::with_capacity(order * order);
                for j in 0..order {
                    for i in 0..order {
                        columns.push(q[(i, j)]);
                    }
                }
                drop(vectors);
                (values, Array2::from_shape_vec((order, order).f(), columns).map_err(error)?)
            }
        };
        let band = symmetric_spectrum_rounding_band(values.as_slice().ok_or_else(|| error("eigenvalues not contiguous"))?);
        Ok(Eigh { values, vectors, band })
    }

    /// The data term's change from `trial`'s move `Δ` of the surviving output columns of
    /// `posterior` with the groups `removed` deleted, in the terms that involve `Δ`, under the
    /// curvature `G ⊗ C` (module note): per output coordinate `r`,
    /// `c_r (½ Δ_rᵀ G_RR Δ_r − Δ_rᵀ G_RK U_K,r)`. With the deleted outputs' own terms (measured
    /// for the removal alone) and the data gradient along `Δ`, the second-order change of the
    /// move actually applied.
    pub fn moved_quadratic(&self, posterior: &Posterior, trial: &Posterior, removed: &[usize]) -> Result<f64, String> {
        let gone: BTreeSet<usize> = removed.iter().copied().collect();
        let mut total = 0.0;
        for mlp in &self.mlps {
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
            let (_, curvature) = Self::scales(mlp, posterior, &Self::columns(mlp, posterior));
            let (before, after) = (&posterior.mean[mlp.output], &trial.mean[mlp.output]);
            // `Δ` and `U_K` as rows, one per function.
            let column = |i: usize| match mlp.outputs[i] {
                Output::Column(column) => Some(column),
                Output::Tied { .. } => None,
            };
            let mut moves = Array2::zeros((surviving.len(), before.nrows()));
            for (mut row, i) in moves.rows_mut().into_iter().zip(&surviving) {
                let c = column(*i).ok_or_else(|| error("a survivor without its own column"))?;
                row.assign(&(&after.column(c) - &before.column(c)));
            }
            let mut deleted_outputs = Array2::zeros((deleted.len(), before.nrows()));
            for (mut row, i) in deleted_outputs.rows_mut().into_iter().zip(&deleted) {
                match mlp.outputs[*i] {
                    Output::Column(c) => row.assign(&before.column(c)),
                    Output::Tied { scale, gate, row: read, .. } => row.assign(&(&posterior.mean[gate].row(read) * posterior.mean[scale][[0, 0]])),
                }
            }
            let own = fast_ab(&mlp.gram.block(&surviving, &surviving, |_, _| 1.0)?, &moves);
            let cross = fast_ab(&mlp.gram.block(&surviving, &deleted, |_, _| 1.0)?, &deleted_outputs);
            for (r, c) in curvature.iter().enumerate() {
                let (mut quadratic, mut coupling) = (0.0, 0.0);
                for j in 0..surviving.len() {
                    quadratic += moves[[j, r]] * own[[j, r]];
                    coupling += moves[[j, r]] * cross[[j, r]];
                }
                total += c * (0.5 * quadratic - coupling);
            }
        }
        Ok(total)
    }

    /// The MLPs' output maps whose surviving columns a proposal moves (indices into
    /// `Explanation::trainable`).
    #[must_use]
    pub fn outputs(&self) -> Vec<usize> {
        self.mlps.iter().map(|mlp| mlp.output).collect()
    }

    /// `posterior` with the groups `removed` removed and, in every MLP that loses functions it had,
    /// the compensation's conditional posterior of its surviving functions' outputs: their means
    /// moved to its mode, their standard deviations kept, which are its mean-field ones (module
    /// note). Each MLP's compensation reads `posterior` alone and moves only its own output
    /// columns; the MLPs are solved one at a time, in order, each move added before the next MLP is
    /// decomposed, so one decomposition is held at a time.
    pub fn proposal(&self, posterior: &Posterior, removed: &[usize]) -> Result<Posterior, String> {
        let gone: BTreeSet<usize> = removed.iter().copied().collect();
        let mut trial = posterior.clone();
        for mlp in &self.mlps {
            let Some(change) = self.change(mlp, posterior, &gone)? else { continue };
            let target = &mut trial.mean[change.output];
            for ((row, column), r) in change.rows.rows().into_iter().zip(&change.columns).zip(&change.scales) {
                if let Some(column) = column {
                    target.column_mut(*column).scaled_add(*r, &row);
                }
            }
        }
        trial.remove(removed);
        Ok(trial)
    }

    /// The move of `mlp`'s surviving output columns when the groups `gone` are removed from
    /// `posterior` ([`Compensation::proposal`]), or `None` when it loses no function it had.
    fn change(&self, mlp: &Mlp, posterior: &Posterior, gone: &BTreeSet<usize>) -> Result<Option<Change>, String> {
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
            return Ok(None);
        }
        let (prior, curvature) = Self::scales(mlp, posterior, &Self::columns(mlp, posterior));
        // A survivor without prior variance (all its values and deviations zero) stays put.
        let surviving: Vec<usize> = surviving.into_iter().filter(|i| prior[*i] > 0.0).collect();
        if surviving.is_empty() || curvature.iter().all(|c| *c == 0.0) {
            return Ok(None);
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
        let (root, decomposition, floor) = self.scaled(mlp, &surviving, &prior)?;
        // `V_R^½ G_RK U_K` along `S`'s eigenvectors, each coordinate's column weighed by
        // `c_r / (1 + c_r e_k)`, and back.
        let mut right = fast_ab(&mlp.gram.block(&surviving, &deleted, |_, _| 1.0)?, &deleted_outputs);
        for (mut row, r) in right.rows_mut().into_iter().zip(&root) {
            row *= *r;
        }
        let mut projected = fast_atb(&decomposition.vectors, &right);
        for (mut row, e) in projected.rows_mut().into_iter().zip(&decomposition.values) {
            if *e > floor {
                row.zip_mut_with(&curvature, |x, c| *x *= c / (1.0 + c * e));
            } else {
                row.fill(0.0);
            }
        }
        let change = fast_ab(&decomposition.vectors, &projected);
        let columns = surviving.iter().map(|i| match mlp.outputs[*i] {
            Output::Column(column) => Some(column),
            Output::Tied { .. } => None,
        });
        Ok(Some(Change { output: mlp.output, columns: columns.collect(), scales: root, rows: change }))
    }

    /// Per function alive at `posterior` with its own output column, its groups and the change in
    /// nats that compensating its deletion alone adds to the deletion's own: the data gradient
    /// `gradient` (summed over the batches, in nats, per output map by index into
    /// `Explanation::trainable`) along the move `Δ`, the move's own and coupling terms under the
    /// curvature `G ⊗ C` ([`Compensation::moved_quadratic`]), and the move's change of the
    /// survivors' Gaussian prior, `Σ_j (μ_j · Δ_j + ½ ‖Δ_j‖²) / v_j`. For all functions at once
    /// from one eigendecomposition `S = V^½ G_AA V^½ = Q diag(e) Qᵀ` over the alive functions `A`:
    /// with `M_r = G_AA + Λ_r = V^-½ (S + I/c_r) V^-½` and `w = M_r⁻¹ e_i`, the move of deleting
    /// `i` is `Δ_r = −u_ir w_R / w_i` (the Schur complement), and every term reduces to
    /// `T_0 = (Q ⊙ Q) D_r`, `T_2 = (Q ⊙ Q) D_r²` and `Q (D_r ⊙ Qᵀ V^½ g)`, `Q (D_r ⊙ Qᵀ V^-½ μ)`
    /// with `D_rk = c_r / (1 + c_r e_k)`. Eigenvalues below the rounding floor are taken as zero
    /// here, where the proposal leaves its move along them unchanged: a ranking estimate.
    pub fn applied(&self, posterior: &Posterior, gradient: &BTreeMap<usize, Array2<f64>>) -> Result<Vec<(Vec<usize>, f64)>, String> {
        let mut out = Vec::new();
        for mlp in &self.mlps {
            let alive = Self::columns(mlp, posterior);
            let (prior, curvature) = Self::scales(mlp, posterior, &alive);
            let alive: Vec<usize> = alive.into_iter().filter(|i| prior[*i] > 0.0).collect();
            if alive.len() < 2 {
                continue;
            }
            let (root, decomposition, floor) = self.scaled(mlp, &alive, &prior)?;
            let values = decomposition.values.mapv(|e| if e > floor { e } else { 0.0 });
            let mut q = decomposition.vectors;
            let (n, d) = (alive.len(), curvature.len());
            let means = &posterior.mean[mlp.output];
            let column = |i: usize| match mlp.outputs[i] {
                Output::Column(c) => c,
                Output::Tied { .. } => usize::MAX,
            };
            // Per alive function (row) and output coordinate: its mean output and summed gradient.
            let mu = Array2::from_shape_fn((n, d), |(a, r)| means[[r, column(alive[a])]]);
            let zero = Array2::zeros(means.dim());
            let g_map = gradient.get(&mlp.output).unwrap_or(&zero);
            let g = Array2::from_shape_fn((n, d), |(a, r)| g_map[[r, column(alive[a])]]);
            let dk = Array2::from_shape_fn((n, d), |(k, r)| curvature[r] / (1.0 + curvature[r] * values[k]));
            let scaled_g = Array2::from_shape_fn((n, d), |(a, r)| root[a] * g[[a, r]]);
            let scaled_mu = Array2::from_shape_fn((n, d), |(a, r)| mu[[a, r]] / root[a]);
            let t1 = fast_ab(&q, &(&fast_atb(&q, &scaled_g) * &dk));
            let t3 = fast_ab(&q, &(&fast_atb(&q, &scaled_mu) * &dk));
            // `Q ⊙ Q` in `Q`'s place.
            q.mapv_inplace(|x| x * x);
            let t0 = fast_ab(&q, &dk);
            let t2 = fast_ab(&q, &dk.mapv(|x| x * x));
            for (a, i) in alive.iter().enumerate() {
                let (v, gii) = (prior[*i], mlp.gram.at(*i, *i));
                let mut extra = 0.0;
                for (r, c) in curvature.iter().enumerate() {
                    let (u, t) = (mu[[a, r]], t0[[a, r]]);
                    if *c <= 0.0 || t <= 0.0 {
                        continue;
                    }
                    let slope = u * (g[[a, r]] - t1[[a, r]] / (v.sqrt() * t));
                    let quadratic = -0.5 * u * u * (c * gii - c / (v * t) + t2[[a, r]] / (v * t * t));
                    let prior_move = -u * (v.sqrt() * t3[[a, r]] - u * t) / (v * t) + 0.5 * u * u * (t2[[a, r]] - t * t) / (v * t * t);
                    extra += slope + quadratic + prior_move;
                }
                out.push((mlp.functions[*i].clone(), extra));
            }
        }
        Ok(out)
    }

    /// Per function alive at `posterior` with its own output column, its groups and the share of
    /// its removal's data rise that remains after compensation, the others of its MLP surviving.
    /// Deleting function `i` alone, with survivors `R` and `h_i` its activations, the compensation
    /// of coordinate `r` minimizes `c_r ‖H_R Δ_r − h_i u_ri‖² + Σ_{j∈R} Δ_rj² / v_j` (twice the
    /// change of `F` it makes; module note), whose minimum is `c_r u_ri² ρ_ir`,
    /// `ρ_ir = G_ii − G_iR (G_RR + Λ_r)⁻¹ G_Ri`, the Schur complement of the regularized Gram
    /// `M_r = G_AA + Λ_r` over all those functions `A` less `Λ_r`'s own entry:
    /// `ρ_ir = 1 / (M_r⁻¹)_ii − 1 / (c_r v_i)`. Along the eigenvectors of
    /// `S = V^½ G_AA V^½ = Q diag(e) Qᵀ`, `(M_r⁻¹)_ii = c_r v_i Σ_k Q_ik² / (1 + c_r e_k)` and
    /// `Σ_k Q_ik² = 1`, so
    /// `ρ_ir / G_ii = Σ_k Q_ik² e_k / (1 + c_r e_k) / (S_ii Σ_k Q_ik² / (1 + c_r e_k))`, which
    /// inverts no Gram and holds for a singular one (its eigenvalues below the rounding floor taken
    /// as zero). The uncompensated rise of coordinate `r` is `½ c_r u_ri² G_ii`, so the share is
    /// `Σ_r c_r u_ri² ρ_ir / (G_ii Σ_r c_r u_ri²)`. It is 1 where the data does not support moving
    /// the outputs (`c_r → 0`), and at every `c_r` for a function whose activations are orthogonal
    /// to its survivors' (`G_iR = 0`). As `c_r → ∞` it tends to the least-squares share
    /// `‖(I − P_R) h_i‖² / G_ii`, `P_R` the orthogonal projector onto the survivors' activations:
    /// `1 / (G_ii (G_AA⁺)_ii)` when `e_i` lies in the range of `G_AA`, and zero when it does not,
    /// which is when the survivors reproduce the function's activations exactly. Two functions
    /// with the same activations `h`, orthogonal to the others', each keep
    /// `1 / (1 + c_r v_j ‖h‖²)` of their rise, `v_j` the other's prior variance.
    pub fn unexplained(&self, posterior: &Posterior) -> Result<Vec<(Vec<usize>, f64)>, String> {
        let mut out = Vec::new();
        for mlp in &self.mlps {
            let alive = Self::columns(mlp, posterior);
            let (prior, curvature) = Self::scales(mlp, posterior, &alive);
            let alive: Vec<usize> = alive.into_iter().filter(|i| prior[*i] > 0.0).collect();
            if alive.is_empty() {
                continue;
            }
            let (_, decomposition, floor) = self.scaled(mlp, &alive, &prior)?;
            let values = decomposition.values.mapv(|e| if e > floor { e } else { 0.0 });
            // `Q ⊙ Q` in `Q`'s place.
            let mut squares = decomposition.vectors;
            squares.mapv_inplace(|q| q * q);
            let weights = |f: &dyn Fn(f64, f64) -> f64| Array2::from_shape_fn((values.len(), curvature.len()), |(k, r)| f(values[k], curvature[r]));
            let numerator = fast_ab(&squares, &weights(&|e, c| e / (1.0 + c * e)));
            let denominator = fast_ab(&squares, &weights(&|e, c| 1.0 / (1.0 + c * e)));
            let means = &*posterior.mean[mlp.output];
            for (a, i) in alive.iter().enumerate() {
                let Output::Column(column) = mlp.outputs[*i] else { continue };
                let diagonal = prior[*i] * mlp.gram.at(*i, *i);
                let (mut remaining, mut total) = (0.0, 0.0);
                for (r, c) in curvature.iter().enumerate() {
                    let rise = c * means[[r, column]].powi(2);
                    if rise > 0.0 && diagonal > 0.0 && denominator[[a, r]] > 0.0 {
                        remaining += rise * (numerator[[a, r]] / denominator[[a, r]] / diagonal).clamp(0.0, 1.0);
                        total += rise;
                    }
                }
                out.push((mlp.functions[*i].clone(), if total > 0.0 { remaining / total } else { 1.0 }));
            }
        }
        Ok(out)
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
    use ndarray::{Array1, Array2};
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
        let reads = interchange::reads(native, &sites).expect("the reads");
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

    /// A posterior concentrated enough (`N = 2^40` tokens) that the compensation's posterior mode
    /// is its least-squares limit to within `1e-9` of the stream.
    const CONCENTRATED: usize = 1 << 40;

    #[test]
    fn deleting_a_copy_of_a_surviving_function_leaves_the_explanation_unchanged() {
        let (native, explanation, sequences) = tiny("library_compensation_copy");
        let mut posterior = Posterior::new(&explanation, CONCENTRATED).expect("the posterior");
        copy_reads(&explanation, &mut posterior, 0, 0, 1);
        let output = *explanation.layers[0].functions[1].last().expect("an output group");
        let (compensated, plain, largest) = changes(&explanation, &native, &posterior, &sequences, output);
        assert!(compensated <= 1e-9 * largest, "the compensated removal moved the stream by {compensated:e} (largest value {largest:e})");
        assert!(plain > 1e3 * compensated.max(f64::EPSILON * largest), "the plain removal moved the stream by only {plain:e}");
    }

    #[test]
    fn the_moved_terms_cancel_a_deleted_copys_own_term() {
        // Deleting a copy of a surviving function, the compensation reproduces its output exactly,
        // so under the model the data term does not change: the move's terms equal minus the
        // deleted output's own, −½ Σ_r c_r u_r² G_kk.
        let (native, explanation, sequences) = tiny("library_compensation_moved");
        let mut posterior = Posterior::new(&explanation, CONCENTRATED).expect("the posterior");
        copy_reads(&explanation, &mut posterior, 0, 0, 1);
        let output = *explanation.layers[0].functions[1].last().expect("an output group");
        let device = Device::host();
        let sites: Vec<_> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let reads = interchange::reads(&native, &sites).expect("the reads");
        let mut ic = Interchange::new(&device, &native, &sites, &explanation.artifact, &explanation.trainable, reads, 1 << 30, 64).expect("the experiments");
        let compensation = Compensation::new(&mut ic, &explanation, &posterior, &sequences, 2).expect("the compensation");
        let trial = compensation.proposal(&posterior, &[output]).expect("the proposal");
        assert_eq!(compensation.moved_quadratic(&posterior, &posterior, &[output]).unwrap(), 0.0, "no move, no terms");
        let mlp = &compensation.mlps[0];
        let (_, curvature) = Compensation::scales(mlp, &posterior, &Compensation::columns(mlp, &posterior));
        let super::Output::Column(column) = mlp.outputs[1] else { panic!("an own column") };
        let own: f64 = curvature.iter().zip(posterior.mean[mlp.output].column(column)).map(|(c, u)| c * u * u).sum::<f64>() * mlp.gram.at(1, 1);
        let moved = compensation.moved_quadratic(&posterior, &trial, &[output]).unwrap();
        assert!(own > 0.0 && (moved + 0.5 * own).abs() <= 1e-6 * own, "moved {moved:e}, own {own:e}");
    }

    #[test]
    fn the_applied_estimate_is_the_compensated_deletions_own() {
        // For one function, what compensating its deletion adds (Compensation::applied, from one
        // eigendecomposition over every function) equals the same terms of the proposal that
        // deletes it: the gradient along the move, Compensation::moved_quadratic, and the move's
        // change of the survivors' Gaussian prior.
        use rand::{RngExt, SeedableRng, rngs::StdRng};
        let (native, explanation, sequences) = tiny("library_compensation_applied");
        let posterior = Posterior::new(&explanation, 1000).expect("the posterior");
        let device = Device::host();
        let sites: Vec<_> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let reads = interchange::reads(&native, &sites).expect("the reads");
        let mut ic = Interchange::new(&device, &native, &sites, &explanation.artifact, &explanation.trainable, reads, 1 << 30, 64).expect("the experiments");
        let compensation = Compensation::new(&mut ic, &explanation, &posterior, &sequences, 2).expect("the compensation");
        let mlp = &compensation.mlps[0];
        let mut rng = StdRng::seed_from_u64(3);
        let gradient: BTreeMap<usize, ndarray::Array2<f64>> = [(mlp.output, posterior.mean[mlp.output].mapv(|_| rng.random::<f64>() - 0.5))].into_iter().collect();
        let applied = compensation.applied(&posterior, &gradient).expect("the estimates");
        let (prior, _) = Compensation::scales(mlp, &posterior, &Compensation::columns(mlp, &posterior));
        let function = 2;
        let removed = mlp.functions[function].clone();
        let trial = compensation.proposal(&posterior, &removed).expect("the proposal");
        let (before, after) = (&posterior.mean[mlp.output], &trial.mean[mlp.output]);
        let (mut slope, mut prior_move) = (0.0, 0.0);
        for j in Compensation::columns(mlp, &posterior).into_iter().filter(|j| *j != function) {
            let super::Output::Column(c) = mlp.outputs[j] else { continue };
            for r in 0..before.nrows() {
                let delta = after[[r, c]] - before[[r, c]];
                slope += gradient[&mlp.output][[r, c]] * delta;
                prior_move += (before[[r, c]] * delta + 0.5 * delta * delta) / prior[j];
            }
        }
        let direct = slope + compensation.moved_quadratic(&posterior, &trial, &removed).unwrap() + prior_move;
        let estimate = applied.iter().find(|(groups, _)| *groups == removed).expect("the function's estimate").1;
        assert!(direct.abs() > 0.0 && (estimate - direct).abs() <= 1e-6 * direct.abs(), "applied {estimate:e}, direct {direct:e}");
    }

    #[test]
    fn deleting_a_tied_function_moves_a_free_copy_of_it_and_leaves_the_explanation_unchanged() {
        use crate::library_sharing::{Tie, tie};
        let (native, start, sequences) = tiny("library_compensation_tie");
        // Function 3 of layer 0 writes 2.5 times the gate row of function 5 of layer 1, through a
        // read–write tie; function 4 reads what function 3 reads and has a free output.
        let tied = tie(&start, &[Tie { source: (0, 3), target: (1, 5), scale: 2.5 }]).expect("the tie");
        let mut posterior = Posterior::new(&tied, CONCENTRATED).expect("the posterior");
        copy_reads(&tied, &mut posterior, 0, 3, 4);
        let scale = tied.groups.iter().position(|g| g.name == "library.l0.mlp.f3.tie").expect("the tie's scale group");
        let (compensated, plain, largest) = changes(&tied, &native, &posterior, &sequences, scale);
        assert!(compensated <= 1e-9 * largest, "the compensated removal moved the stream by {compensated:e} (largest value {largest:e})");
        assert!(plain > 1e3 * compensated.max(f64::EPSILON * largest), "the plain removal moved the stream by only {plain:e}");
    }

    #[test]
    fn the_compensation_is_the_posterior_mode_and_moves_a_near_copy_less_than_least_squares() {
        let (native, explanation, sequences) = tiny("library_compensation_mode");
        let mut posterior = Posterior::new(&explanation, 1000).expect("the posterior");
        // Function 1 reads what function 0 reads, its gate's first entry moved by one part in 10^6:
        // its activations are nearly in the others' span.
        copy_reads(&explanation, &mut posterior, 0, 0, 1);
        let gate = &explanation.groups[explanation.layers[0].functions[1][0]].cells[0];
        let position: BTreeMap<usize, usize> = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
        posterior.mean[position[&gate.operator]][[gate.rows[0], gate.cols.start]] *= 1.0 + 1e-6;
        let device = Device::host();
        let sites: Vec<_> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let reads = interchange::reads(&native, &sites).expect("the reads");
        let mut ic = Interchange::new(&device, &native, &sites, &explanation.artifact, &explanation.trainable, reads, 1 << 30, 64).expect("the experiments");
        let compensation = Compensation::new(&mut ic, &explanation, &posterior, &sequences, 2).expect("the compensation");
        let output = *explanation.layers[0].functions[1].last().expect("an output group");
        let trial = compensation.proposal(&posterior, &[output]).expect("the proposal");
        let mlp = &compensation.mlps[0];
        let surviving: Vec<usize> = (0..mlp.functions.len()).filter(|i| *i != 1).collect();
        let (prior, curvature) = Compensation::scales(mlp, &posterior, &Compensation::columns(mlp, &posterior));
        assert!(curvature.iter().all(|c| c.is_finite() && *c > 0.0), "a curvature estimate is not positive");
        let column = |i: usize| match mlp.outputs[i] {
            super::Output::Column(c) => Some(c),
            super::Output::Tied { .. } => None,
        }
        .expect("an own output column");
        let means = &*posterior.mean[mlp.output];
        let moved = &*trial.mean[mlp.output] - means;
        let delta = Array2::from_shape_fn((surviving.len(), means.nrows()), |(a, r)| moved[[r, column(surviving[a])]]);
        let deleted = means.column(column(1)).to_owned();
        let g_rr = mlp.gram.block(&surviving, &surviving, |_, _| 1.0).expect("the survivors' Gram");
        let g_rk = mlp.gram.block(&surviving, &[1], |_, _| 1.0).expect("the survivors' Gram with function 1").column(0).to_owned();
        // Stationarity of `Σ_r ½ c_r ‖H_R Δ_r − H_K u_r‖² + Σ_j ‖Δ_j‖² / (2 v_j)` per coordinate.
        let mut worst: f64 = 0.0;
        for r in 0..means.nrows() {
            let d = delta.column(r);
            let pull = &g_rk * (curvature[r] * deleted[r]);
            let gradient = g_rr.dot(&d) * curvature[r] - &pull + &Array1::from_shape_fn(d.len(), |a| d[a] / prior[surviving[a]]);
            let scale = pull.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(f64::MIN_POSITIVE);
            worst = worst.max(gradient.iter().fold(0.0_f64, |m, v| m.max(v.abs())) / scale);
        }
        assert!(worst <= 1e-6, "the proposal is not the posterior mode: relative gradient {worst:e}");
        // The survivors' deviations are kept: the deletion leaves the conditional posterior's
        // mean-field variances where they were (module note).
        for j in &surviving {
            assert_eq!(trial.log_sd[mlp.output].column(column(*j)), posterior.log_sd[mlp.output].column(column(*j)), "survivor {j}'s deviations");
        }
        // The least-squares move of the same deletion (the limit of a concentrated posterior).
        let mut sharp = posterior.clone();
        let tokens = 1e12_f64;
        for log_sd in &mut sharp.log_sd {
            log_sd.mapv_inplace(|s| s - 0.5 * (tokens / 1000.0).ln());
        }
        let least = compensation.proposal(&sharp, &[output]).expect("the least-squares proposal");
        let norm = |p: &Posterior| (&*p.mean[mlp.output] - means).iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm(&trial) < norm(&least), "the posterior mode moved the outputs {} and least squares {}", norm(&trial), norm(&least));
        // The shares are in [0, 1], and the nearly copied function's is below the others'.
        let shares = compensation.unexplained(&posterior).expect("the shares");
        assert!(shares.iter().all(|(_, s)| (0.0..=1.0).contains(s)));
        let of = |f: usize| shares.iter().find(|(groups, _)| *groups == explanation.layers[0].functions[f]).map(|(_, s)| *s).expect("a share");
        assert!(of(1) < of(2), "the near copy keeps {} of its rise, an unrelated function {}", of(1), of(2));
    }

    /// A transcoder block (fixed output bias, `M`'s MLP selected at the first token) on the tiny
    /// Qwen3 decoder: its Gram matrix sums the features' activations over the tokens after the
    /// first only, and deleting a copy of a surviving feature leaves the explanation unchanged.
    #[test]
    fn a_transcoder_blocks_compensation_leaves_out_the_first_token_and_moves_a_copy() {
        use crate::library_transcoder::{Transcoder, firing};
        let dir = std::env::temp_dir().join(format!("library_compensation_transcoder_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let export = crate::test_support::tiny_qwen3_export("library_compensation_transcoder", 2);
        let imported = import_language_model(&export, 6, 12).unwrap();
        std::fs::remove_dir_all(&export).unwrap();
        let native = split_sites(&imported.program).unwrap();
        let layers = layer_nodes(&native, 2).unwrap();
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("tokens") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let path = dir.join("layer_1.safetensors");
        crate::test_support::transcoder_file(&path, 64, 8, 3);
        let transcoders = BTreeMap::from([(1, Transcoder::open(&path).unwrap())]);
        let counts = firing(&Device::host(), &native, &layers, &transcoders, &sequences, 2).unwrap();
        let kept: Vec<usize> = (0..64).filter(|&f| counts[&1][f] > 0).collect();
        let kept_path = dir.join("kept_1.safetensors");
        transcoders[&1].write_kept(&kept, &kept_path).unwrap();
        let explanation = library_mdl::explanation_with(&native, &layers, &BTreeMap::from([(1, kept_path)])).unwrap();
        let mut posterior = Posterior::new(&explanation, CONCENTRATED).expect("the posterior");
        // The features' activations at M's input (layer 0 is M's at the start), first tokens left out.
        let family = library_mdl::sequence_family(&sequences.iter().map(Vec::as_slice).collect::<Vec<_>>()).unwrap();
        let x = native.execute(&family, false).unwrap().values[layers[1].normed].clone();
        let program = &explanation.artifact.program;
        let matrix = |name: &str| program.operators[program.operators.iter().position(|op| op.name == name).unwrap()].matrix();
        let mut h = (x.dot(&matrix("library.l1.mlp.gate").t()) + &matrix("library.l1.mlp.gate_bias").column(0)).mapv(|v| v.max(0.0));
        for (mut row, p) in h.rows_mut().into_iter().zip(&family.layout.as_ref().unwrap().position) {
            if *p == 0 {
                row.fill(0.0);
            }
        }
        let expected = h.t().dot(&h);
        let device = Device::host();
        let sites: Vec<_> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let reads = interchange::reads(&native, &sites).expect("the reads");
        let mut ic = Interchange::new(&device, &native, &sites, &explanation.artifact, &explanation.trainable, reads, 1 << 30, 64).expect("the experiments");
        let compensation = Compensation::new(&mut ic, &explanation, &posterior, &sequences, 2).expect("the compensation");
        let gram = &compensation.mlps[1].gram;
        let scale = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let difference = expected.indexed_iter().fold(0.0_f64, |m, ((i, j), b)| m.max((gram.at(i, j) - b).abs()));
        assert!(scale > 0.0 && difference <= 1e-12 * scale, "the Gram differs from the first-token-free one by {difference:e} (scale {scale:e})");
        // Feature 1 made a copy of feature 0 (gate row and bias), then deleted with compensation.
        copy_reads(&explanation, &mut posterior, 1, 0, 1);
        let output = *explanation.layers[1].functions[1].last().expect("an output group");
        let (compensated, plain, largest) = changes(&explanation, &native, &posterior, &sequences, output);
        assert!(compensated <= 1e-9 * largest, "the compensated removal moved the stream by {compensated:e} (largest value {largest:e})");
        assert!(plain > 1e3 * compensated.max(f64::EPSILON * largest), "the plain removal moved the stream by only {plain:e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_duplicate_keeps_its_regularized_share_and_an_orthogonal_function_all_of_its_rise() {
        // Three functions writing one output coordinate: 0 and 1 with the same activations `h`
        // (a singular Gram), 2 orthogonal to both, `G = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]`. Every
        // output entry has mean 1 and variance `s`, so `v = 1 + s` for each function and
        // `c = Σ_j (1/s − 1/v) / Σ_j G_jj = 1/(s (1 + s))`. Deleting 0, its survivor 1 reproduces
        // its activations, and the regularization leaves `ρ = G_00 − G_01² / (G_11 + 1/(c v)) =
        // 1/(1 + c v)` of `G_00 = 1`, which is `s/(1 + s)`: a half at `s = 1`, 9.1e-13 at
        // `s = 2^-40`, and zero, the least-squares share of a reproduced function, as `s → 0`; the
        // pseudoinverse's `1/(G_00 (G⁺)_00) = 4` is not that limit. No survivor reaches function
        // 2's activations: it keeps all of its rise at every `s`.
        let gram = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut packed = super::Packed::zeros(3);
        packed.add_lower(&Array2::from_shape_fn((3, 3), |(i, j)| gram[i][j])).expect("the Gram");
        let mlp = super::Mlp { functions: vec![vec![0], vec![1], vec![2]], output: 0, outputs: (0..3).map(super::Output::Column).collect(), gram: packed };
        let compensation = Compensation { mlps: vec![mlp], rows: 1, device: Device::host() };
        for s in [1.0, 2f64.powi(-40)] {
            let membership = Array2::from_shape_fn((1, 3), |(_, j)| j as u32);
            let mut posterior = Posterior::from_parts(vec![Array2::ones((1, 3))], vec![membership], vec![1.0; 3], 1).expect("the posterior");
            posterior.log_sd[0] = Array2::from_elem((1, 3), 0.5 * s.ln()).into();
            let shares = compensation.unexplained(&posterior).expect("the shares");
            let of = |f: usize| shares.iter().find(|(groups, _)| *groups == [f]).map(|(_, share)| *share).expect("a share");
            let v = 1.0 + s;
            let c = 1.0 / s - 1.0 / v;
            let duplicate = 1.0 / (1.0 + c * v);
            for f in [0, 1] {
                assert!((of(f) - duplicate).abs() <= 1e-9 * duplicate, "s = {s:e}: function {f} keeps {:e} of its rise, the regularized share is {duplicate:e}", of(f));
            }
            assert!((of(2) - 1.0).abs() <= 1e-12, "s = {s:e}: the orthogonal function keeps {} of its rise", of(2));
        }
    }
}

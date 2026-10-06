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
//! group's empirical-Bayes variance.
//!
//! A function's output is its own column of the MLP's output map, or, under a read–write tie
//! (`library_sharing::tie`), `c` times a later layer's gate row. A tied output is fixed by the tie:
//! the compensation never moves it (it is outside `R`), and when the function is deleted its output
//! `c a` enters `U_K`. Deleting the gate row a tie reads deletes the tied output with it. The compensated
//! removal is a proposal: removal accepts it only when it does not increase the code length `F` of
//! the composed explanation on the fixed collection (`library_mdl`'s removal step).
//!
//! Each MLP's Gram matrix `G` is formed once per removal round, in float64 from the activations the
//! device computes. An eigenvalue `e_k` of `S` within the bound on its rounding (the summation bound
//! `γ_T trace(S)` of its `T`-term dot products plus the eigendecomposition's own band) is not
//! resolved from zero: a proposal leaves the outputs unchanged along it, and the share below takes
//! it as zero.

use crate::{
    interchange::{self, Interchange},
    library_mdl::{Explanation, Posterior, sequence_family},
    operator_program::{Node, OperatorProgram},
    run_check::LayerNodes,
};
use gam_gpu::{
    gpu_error::GpuError,
    tensor::{Device, Storage},
};
use gam_linalg::{
    decompose::{Eigh, eigh},
    faer_ndarray::{fast_ab, fast_ata, fast_atb},
    roundoff::{SymmetricAssembly, symmetric_spectrum_rounding_band},
};
use ndarray::{Array1, Array2, Axis};
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
            // An MLP the explanation leaves to `M` (`library_mdl::scoped`) has nothing to compensate.
            if layer.functions.is_empty() {
                continue;
            }
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
                        // On CUDA the f32 activations' Gram on the integer tensor cores
                        // (`Device::gram_split`, 9 slices: within γ_rows √(G_ii G_jj) per entry, the
                        // form of the float64 product's own bound that the floor below uses), else
                        // the float64 product.
                        if h.rows() > MAX_SPLIT_ROWS || !wide.gram_split(sum, h, GRAM_SLICES).map_err(error)? {
                            let h = wide.convert(h).map_err(error)?;
                            wide.gram_lower(sum, &h, 1.0).map_err(error)?;
                        }
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
        let means = &posterior.mean[mlp.output];
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
            energy += mlp.gram[[*i, *i]];
        }
        let curvature = precision.mapv(|p| if energy > 0.0 && p > 0.0 { p / energy } else { 0.0 });
        (prior, curvature)
    }

    /// `S = V^½ G_ss V^½` over the functions `set` with prior variances `prior`, its
    /// eigendecomposition and the floor below which an eigenvalue is not resolved from zero.
    fn scaled(&self, mlp: &Mlp, set: &[usize], prior: &[f64]) -> Result<(Vec<f64>, Eigh, f64), String> {
        let root: Vec<f64> = set.iter().map(|i| prior[*i].sqrt()).collect();
        let mut scaled = mlp.gram.select(Axis(0), set).select(Axis(1), set);
        for ((a, b), value) in scaled.indexed_iter_mut() {
            *value *= root[a] * root[b];
        }
        // On CUDA by cuSOLVER (`Device::symmetric_eigh`, the same backward-error band), else on the host.
        let decomposition = match self.device.symmetric_eigh(scaled.view()).map_err(error)? {
            Some((values, vectors)) => Eigh { band: symmetric_spectrum_rounding_band(values.as_slice().ok_or_else(|| error("eigenvalues not contiguous"))?), values, vectors },
            None => eigh(scaled.view(), SymmetricAssembly::Mirrored, None).map_err(error)?,
        };
        let floor = decomposition.band + self.gamma()? * scaled.diag().sum();
        Ok((root, decomposition, floor))
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
            let own = fast_ab(&mlp.gram.select(Axis(0), &surviving).select(Axis(1), &surviving), &moves);
            let cross = fast_ab(&mlp.gram.select(Axis(0), &surviving).select(Axis(1), &deleted), &deleted_outputs);
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
    /// its surviving functions' outputs moved to the posterior mode of the compensation (module
    /// note). Each MLP's compensation reads `posterior` alone and moves only its own output
    /// columns, so the MLPs are solved in parallel and their moves added after, in MLP order.
    pub fn proposal(&self, posterior: &Posterior, removed: &[usize]) -> Result<Posterior, String> {
        let gone: BTreeSet<usize> = removed.iter().copied().collect();
        let changes: Vec<Option<Change>> = self.mlps.par_iter().map(|mlp| self.change(mlp, posterior, &gone)).collect::<Result<_, String>>()?;
        let mut trial = posterior.clone();
        for change in changes.into_iter().flatten() {
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
        let mut right = fast_ab(&mlp.gram.select(Axis(0), &surviving).select(Axis(1), &deleted), &deleted_outputs);
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

    /// Per function alive at `posterior` with its own output column, its groups and the share of
    /// its removal's data rise that remains after compensation, the others of its MLP surviving.
    /// Deleting function `i` alone, the compensation of coordinate `r` leaves
    /// `ρ_ir = G_ii − G_iR (G_RR + Λ_r)⁻¹ G_Ri` of its activations' energy `G_ii` (the Schur
    /// complement of the regularized Gram; module note), which along the eigenvectors of
    /// `S = V^½ G_AA V^½` over all those functions `A` is
    /// `ρ_ir / G_ii = Σ_k Q_ik² e_k / (1 + c_r e_k) / (S_ii Σ_k Q_ik² / (1 + c_r e_k))`.
    /// The uncompensated rise of coordinate `r` is `½ c_r u_ri² G_ii`, so the share is
    /// `Σ_r c_r u_ri² ρ_ir / (G_ii Σ_r c_r u_ri²)`: from 1 where the data does not support moving
    /// the outputs (`c_r → 0`) to the least-squares `1 / (G_ii (G_AA⁻¹)_ii)` (`c_r → ∞`).
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
            let squares = decomposition.vectors.mapv(|q| q * q);
            let weights = |f: &dyn Fn(f64, f64) -> f64| Array2::from_shape_fn((values.len(), curvature.len()), |(k, r)| f(values[k], curvature[r]));
            let numerator = fast_ab(&squares, &weights(&|e, c| e / (1.0 + c * e)));
            let denominator = fast_ab(&squares, &weights(&|e, c| 1.0 / (1.0 + c * e)));
            let means = &posterior.mean[mlp.output];
            for (a, i) in alive.iter().enumerate() {
                let Output::Column(column) = mlp.outputs[*i] else { continue };
                let diagonal = prior[*i] * mlp.gram[[*i, *i]];
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
        use ndarray::Axis;
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
        let means = &posterior.mean[mlp.output];
        let moved = &trial.mean[mlp.output] - means;
        let delta = Array2::from_shape_fn((surviving.len(), means.nrows()), |(a, r)| moved[[r, column(surviving[a])]]);
        let deleted = means.column(column(1)).to_owned();
        let g_rr = mlp.gram.select(Axis(0), &surviving).select(Axis(1), &surviving);
        let g_rk = mlp.gram.select(Axis(0), &surviving).column(1).to_owned();
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
        // The least-squares move of the same deletion (the limit of a concentrated posterior).
        let mut sharp = posterior.clone();
        let tokens = 1e12_f64;
        for log_sd in &mut sharp.log_sd {
            log_sd.mapv_inplace(|s| s - 0.5 * (tokens / 1000.0).ln());
        }
        let least = compensation.proposal(&sharp, &[output]).expect("the least-squares proposal");
        let norm = |p: &Posterior| (&p.mean[mlp.output] - means).iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm(&trial) < norm(&least), "the posterior mode moved the outputs {} and least squares {}", norm(&trial), norm(&least));
        // The shares are in [0, 1], and the nearly copied function's is below the others'.
        let shares = compensation.unexplained(&posterior).expect("the shares");
        assert!(shares.iter().all(|(_, s)| (0.0..=1.0).contains(s)));
        let of = |f: usize| shares.iter().find(|(groups, _)| *groups == explanation.layers[0].functions[f]).map(|(_, s)| *s).expect("a share");
        assert!(of(1) < of(2), "the near copy keeps {} of its rise, an unrelated function {}", of(1), of(2));
    }
}

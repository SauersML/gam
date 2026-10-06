//! VPD's MLP subcomponents as parts that edits act on (#2951): each a fixed function of an MLP
//! block's read, added to the block's output, so one edit is defined on `M`, on VPD's replacement
//! and on any other explanation that has the layer's MLP input and output (ours, transcoders).
//!
//! VPD decomposes each weight matrix `W` of `M` into rank-one slices, `W = Σ_i u_i v_iᵀ`
//! (`explanation_battery::Factors`: `u_i` row `i` of `U`, `v_i` column `i` of `V`). Scaling slice
//! `i` by `α` is the weight edit `W + (α − 1) u_i v_iᵀ`, which changes `W`'s output at a row by
//! `(α − 1) (v_iᵀ y) u_i`, `y` the row's input to `W`. In an MLP `out = W_down φ(W_fc x)` (`x` the
//! block's read, the normed stream; vpd4l's MLPs have no biases) the block's output therefore
//! moves by
//!
//! - for a slice of `W_down` (`u_i ∈ ℝ^d`, `v_i ∈ ℝ^hidden`): `(α − 1) (v_iᵀ φ(W_fc x)) u_i`;
//! - for a slice of `W_fc` (`u_i ∈ ℝ^hidden`, `v_i ∈ ℝ^d`):
//!   `W_down [φ(W_fc x + (α − 1) (v_iᵀ x) u_i) − φ(W_fc x)]`.
//!
//! With `M`'s own `W_fc`, `W_down` and `φ` both are functions of `x` alone. On `M` each is exactly
//! the weight edit at the edited row (`VpdPart::edit`); on another explanation it adds the same
//! function of that explanation's own read, the claim that its MLP output holds the slice's
//! contribution as `M` computes it (a cross-edit).

use crate::{explanation_battery::load_factors, explanation_battery::Kind, operator_program::Law};
use ndarray::{Array1, Array2, ArrayView1};
use std::{path::Path, sync::Arc};

fn error(e: impl std::fmt::Display) -> String {
    format!("vpd parts: {e}")
}

/// One layer's MLP of `M`: `out = write · h(x)`, `h(x) = φ(read · x)`, or for a gated MLP
/// (Qwen3's SwiGLU) `h(x) = φ(read · x) ⊙ (up · x)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Mlp {
    /// `W_fc` (hidden × d), the gate of a gated MLP.
    pub read: Array2<f64>,
    /// `W_down` (d × hidden).
    pub write: Array2<f64>,
    pub law: Law,
    /// A gated MLP's up map `W_up` (hidden × d).
    pub up: Option<Array2<f64>>,
}

impl Mlp {
    /// The gate's multiplier `W_up x` at the read `x` (ones for an ungated MLP).
    fn gain(&self, x: ArrayView1<f64>) -> Array1<f64> {
        match &self.up {
            Some(up) => up.dot(&x),
            None => Array1::ones(self.read.nrows()),
        }
    }

    /// Its tangent `W_up dx` (zero for an ungated MLP).
    fn gain_tangent(&self, dx: ArrayView1<f64>) -> Array1<f64> {
        match &self.up {
            Some(up) => up.dot(&dx),
            None => Array1::zeros(self.read.nrows()),
        }
    }

    /// `φ(W_fc x + shift) ⊙ (W_up x)`, the hidden activations at the read `x` with the
    /// pre-activations moved by `shift`.
    fn hidden(&self, x: ArrayView1<f64>, shift: Option<&Array1<f64>>) -> Array1<f64> {
        let mut pre = self.read.dot(&x);
        if let Some(shift) = shift {
            pre += shift;
        }
        pre.mapv(|t| self.law.apply(t)) * self.gain(x)
    }

    /// The tangent of [`Mlp::hidden`] along `dx` with the pre-activations at `pre` and moving by
    /// `dpre`: `φ'(pre) ⊙ dpre ⊙ (W_up x) + φ(pre) ⊙ (W_up dx)`.
    fn hidden_tangent(&self, x: ArrayView1<f64>, pre: &Array1<f64>, dpre: &Array1<f64>, dx: ArrayView1<f64>) -> Array1<f64> {
        pre.mapv(|t| self.law.derivative(t)) * dpre * self.gain(x) + pre.mapv(|t| self.law.apply(t)) * self.gain_tangent(dx)
    }

    /// The read's cotangent from the hidden activations' cotangent `w` at pre-activations `pre`
    /// (whose own dependence on the read is `W_fc x` plus what `extra` adds): `W_fcᵀ(φ'(pre) ⊙ w ⊙
    /// (W_up x)) + W_upᵀ(φ(pre) ⊙ w)`, and `φ'(pre) ⊙ w ⊙ (W_up x)` (the pre-activations'
    /// cotangent).
    fn hidden_pullback(&self, x: ArrayView1<f64>, pre: &Array1<f64>, w: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let through = pre.mapv(|t| self.law.derivative(t)) * w * self.gain(x);
        let mut back = self.read.t().dot(&through);
        if let Some(up) = &self.up {
            back += &up.t().dot(&(pre.mapv(|t| self.law.apply(t)) * w));
        }
        (back, through)
    }

    /// The block's output at the read `x`.
    #[must_use]
    pub fn output(&self, x: ArrayView1<f64>) -> Array1<f64> {
        self.write.dot(&self.hidden(x, None))
    }
}

/// Which of an MLP's maps a slice belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Map {
    /// `W_fc`: `u` in the hidden space, `v` in the read's.
    Up,
    /// `W_down`: `u` in the stream's space, `v` in the hidden space.
    Down,
}

/// Subcomponent `index` of layer `layer`'s MLP map `map`: the rank-one slice `u vᵀ` of it.
#[derive(Clone, Debug, PartialEq)]
pub struct VpdPart {
    /// The MLP block (`2 layer + 1`, as `interchange::Part::block`).
    pub block: usize,
    pub layer: usize,
    pub map: Map,
    pub index: usize,
    pub u: Array1<f64>,
    pub v: Array1<f64>,
}

impl VpdPart {
    /// The change of the MLP block's output at a row whose read is `x` when the slice is scaled by
    /// `alpha`, with `M`'s MLP `mlp` of the part's layer (module note).
    #[must_use]
    pub fn edit(&self, mlp: &Mlp, x: ArrayView1<f64>, alpha: f64) -> Array1<f64> {
        match self.map {
            Map::Down => &self.u * ((alpha - 1.0) * self.v.dot(&mlp.hidden(x, None))),
            Map::Up => {
                let shift = &self.u * ((alpha - 1.0) * self.v.dot(&x));
                mlp.write.dot(&(mlp.hidden(x, Some(&shift)) - mlp.hidden(x, None)))
            }
        }
    }

    /// The derivative of [`VpdPart::edit`] in the read along `dx`, `J(x) dx`, with `h'` the hidden
    /// activations' tangent ([`Mlp::hidden_tangent`], the product rule for a gated MLP): for a
    /// slice of `W_down`, `(α − 1)(v·h'(z; W_fc dx)) u`; for a slice of `W_fc`,
    /// `W_down[h'(z'; W_fc dx + (α − 1)(v·dx) u) − h'(z; W_fc dx)]` (`z`, `z'` as in
    /// [`VpdPart::pullback`]).
    #[must_use]
    pub fn tangent(&self, mlp: &Mlp, x: ArrayView1<f64>, alpha: f64, dx: ArrayView1<f64>) -> Array1<f64> {
        let (pre, moved_in) = (mlp.read.dot(&x), mlp.read.dot(&dx));
        match self.map {
            Map::Down => &self.u * ((alpha - 1.0) * self.v.dot(&mlp.hidden_tangent(x, &pre, &moved_in, dx))),
            Map::Up => {
                let moved = &pre + &(&self.u * ((alpha - 1.0) * self.v.dot(&x)));
                let shifted = &moved_in + &(&self.u * ((alpha - 1.0) * self.v.dot(&dx)));
                mlp.write.dot(&(mlp.hidden_tangent(x, &moved, &shifted, dx) - mlp.hidden_tangent(x, &pre, &moved_in, dx)))
            }
        }
    }

    /// The pullback of [`VpdPart::edit`] in the read: `J(x)ᵀ ḡ` for the output's cotangent `ḡ`,
    /// the transpose of [`VpdPart::tangent`] ([`Mlp::hidden_pullback`]): for a slice of `W_down`,
    /// the hidden activations' cotangent is `(α − 1)(u·ḡ) v`; for a slice of `W_fc`, with
    /// `z' = z + (α − 1)(v·x) u` and `r = W_downᵀ ḡ`, the pullback of `r` at `z'` less that at `z`,
    /// plus `(α − 1)(u·ρ) v` with `ρ` the pre-activations' cotangent at `z'`.
    #[must_use]
    pub fn pullback(&self, mlp: &Mlp, x: ArrayView1<f64>, alpha: f64, cotangent: ArrayView1<f64>) -> Array1<f64> {
        let pre = mlp.read.dot(&x);
        match self.map {
            Map::Down => mlp.hidden_pullback(x, &pre, &(&self.v * ((alpha - 1.0) * self.u.dot(&cotangent)))).0,
            Map::Up => {
                let moved = &pre + &(&self.u * ((alpha - 1.0) * self.v.dot(&x)));
                let r = mlp.write.t().dot(&cotangent);
                let ((after, rho), (before, _)) = (mlp.hidden_pullback(x, &moved, &r), mlp.hidden_pullback(x, &pre, &r));
                after - before + &self.v * ((alpha - 1.0) * self.u.dot(&rho))
            }
        }
    }
}

/// A part with `M`'s MLP of its layer, what an edit of it reads (`interchange::Patch::Slice`).
#[derive(Clone, Debug)]
pub struct Slice {
    pub part: VpdPart,
    pub mlp: Arc<Mlp>,
}

impl Slice {
    /// [`VpdPart::edit`] with the part's own layer's MLP.
    #[must_use]
    pub fn edit(&self, x: ArrayView1<f64>, alpha: f64) -> Array1<f64> {
        self.part.edit(&self.mlp, x, alpha)
    }

    /// [`VpdPart::tangent`] with the part's own layer's MLP.
    #[must_use]
    pub fn tangent(&self, x: ArrayView1<f64>, alpha: f64, dx: ArrayView1<f64>) -> Array1<f64> {
        self.part.tangent(&self.mlp, x, alpha, dx)
    }

    /// [`VpdPart::pullback`] with the part's own layer's MLP.
    #[must_use]
    pub fn pullback(&self, x: ArrayView1<f64>, alpha: f64, cotangent: ArrayView1<f64>) -> Array1<f64> {
        self.part.pullback(&self.mlp, x, alpha, cotangent)
    }

    /// The MLP block whose read it reads and whose output it adds to.
    #[must_use]
    pub fn block(&self) -> usize {
        self.part.block
    }
}

/// [`load`]'s parts, each with its layer's MLP.
pub fn slices(export: &Path, decomposition: &Path) -> Result<Vec<Slice>, String> {
    let (mlps, parts) = load(export, decomposition)?;
    let mlps: Vec<Arc<Mlp>> = mlps.into_iter().map(Arc::new).collect();
    Ok(parts.into_iter().map(|part| Slice { mlp: Arc::clone(&mlps[part.layer]), part }).collect())
}

/// Layer `layer`'s neurons of `M`'s MLP `mlp` as slices of its down map, from `M`'s own weights:
/// neuron `j` is `u = W_down[:, j]`, `v = e_j`, so its edit is `(α − 1) φ(w_j·x) W_down[:, j]`
/// (`w_j` row `j` of `W_fc`) on each model's own read `x`; the slices sum to `W_down`.
#[must_use]
pub fn neuron_slices(mlp: &Arc<Mlp>, layer: usize) -> Vec<Slice> {
    let hidden = mlp.write.ncols();
    (0..hidden)
        .map(|j| {
            let mut v = Array1::zeros(hidden);
            v[j] = 1.0;
            Slice { part: VpdPart { block: 2 * layer + 1, layer, map: Map::Down, index: j, u: mlp.write.column(j).to_owned(), v }, mlp: Arc::clone(mlp) }
        })
        .collect()
}

/// `count` random rank-one slices of layer `layer`'s MLP map `map`, drawn from `seed`: unit `u`
/// and `v` uniform on their spheres (normalized standard normal draws,
/// `gam_gpu::tensor::posterior_normal` under `seed`, streams `2i` and `2i + 1` for slice `i`),
/// scaled by the map's top singular value `s` (`gam_linalg::decompose::svd`), so the edit is
/// `W + (α − 1) s u vᵀ`: a weight edit of `M` at the size of the map's largest direction,
/// independent of any explanation.
pub fn random_slices_of(mlp: &Arc<Mlp>, layer: usize, map: Map, count: usize, seed: u64) -> Result<Vec<Slice>, String> {
    let w = match map {
        Map::Up => &mlp.read,
        Map::Down => &mlp.write,
    };
    let top = gam_linalg::decompose::svd(w.view(), false).map_err(error)?.singular_values.first().copied().unwrap_or(0.0);
    // Standard normal draws keyed by `seed`, one stream per slice and side (`posterior_normal`).
    let unit = |n: usize, stream: u64| {
        let a = Array1::from_shape_fn(n, |i| f64::from(gam_gpu::tensor::posterior_normal(seed, stream, i as u64)));
        let norm = a.dot(&a).sqrt();
        a / norm
    };
    Ok((0..count)
        .map(|index| {
            let (u, v) = (unit(w.nrows(), 2 * index as u64) * top, unit(w.ncols(), 2 * index as u64 + 1));
            Slice { part: VpdPart { block: 2 * layer + 1, layer, map, index, u, v }, mlp: Arc::clone(mlp) }
        })
        .collect())
}

/// `M`'s MLP of every layer from its split native program `native` with its `layers`
/// (`run_check::layer_nodes`): the down map applied to the hidden activations, which are the law of
/// the read map's output (`W_fc`) or, for a gated MLP, the law of the gate's output times the up
/// map's (`LayerNodes::pre`). Refused for an MLP with biases.
pub fn mlps_of(native: &crate::operator_program::OperatorProgram, layers: &[crate::run_check::LayerNodes]) -> Result<Vec<Mlp>, String> {
    use crate::operator_program::Node;
    let map_of = |node: usize, input: usize| -> Result<Array2<f64>, String> {
        match native.nodes.get(node) {
            Some(Node::Affine { terms, bias: None }) if terms.len() == 1 && terms[0].0 == input => Ok(native.operators[terms[0].1].matrix()),
            other => Err(error(format!("node {node} is not a bias-free map of node {input}: {other:?}"))),
        }
    };
    let law_of = |node: usize| -> Result<(Law, usize), String> {
        match native.nodes.get(node) {
            Some(Node::Pointwise { input, laws }) if laws.windows(2).all(|w| w[0] == w[1]) && !laws.is_empty() => Ok((laws[0], *input)),
            other => Err(error(format!("node {node} is not one law of a node: {other:?}"))),
        }
    };
    layers
        .iter()
        .enumerate()
        .map(|(l, layer)| {
            let write = map_of(layer.mlp, layer.active)?;
            let (read, law, up) = match native.nodes.get(layer.active) {
                Some(Node::Hadamard { left, right }) => {
                    let (law, gate) = law_of(*left)?;
                    (map_of(gate, layer.normed)?, law, Some(map_of(*right, layer.normed)?))
                }
                _ => {
                    let (law, pre) = law_of(layer.active)?;
                    (map_of(pre, layer.normed)?, law, None)
                }
            };
            if write.dim() != (read.ncols(), read.nrows()) || up.as_ref().is_some_and(|u| u.dim() != read.dim()) {
                return Err(error(format!("layer {l}: the MLP's maps disagree in shape")));
            }
            Ok(Mlp { read, write, law, up })
        })
        .collect()
}

/// A tensor of an engine export (`export.json` and its float64 files), as the battery reads it.
fn tensor(dir: &Path, record: &serde_json::Value, name: &str) -> Result<Array2<f64>, String> {
    let dims: Vec<usize> = record["files"][name]["shape"]
        .as_array()
        .ok_or_else(|| error(format!("{name}: no shape")))?
        .iter()
        .map(|v| v.as_u64().map(|v| v as usize).ok_or_else(|| error(format!("{name}: a shape entry"))))
        .collect::<Result<_, _>>()?;
    let (rows, cols) = match dims[..] {
        [n] => (1, n),
        [r, c] => (r, c),
        _ => return Err(error(format!("{name}: shape {dims:?}"))),
    };
    crate::import::read_f64_shaped(&dir.join(format!("{name}.f64")), rows, cols)
}

/// `M`'s MLP of every layer from its engine export `export`, and VPD's MLP subcomponents from its
/// exported decomposition `decomposition` (`explanation_battery::load_factors`), per layer the
/// up map's then the down map's in their order. A slice whose `u` or `v` is zero is left out (its
/// edit is zero).
pub fn load(export: &Path, decomposition: &Path) -> Result<(Vec<Mlp>, Vec<VpdPart>), String> {
    let text = std::fs::read_to_string(export.join("export.json")).map_err(|e| error(format!("{}: {e}", export.display())))?;
    let record: serde_json::Value = serde_json::from_str(&text).map_err(error)?;
    let config = &record["config"];
    let layers = config["n_layers"].as_u64().ok_or_else(|| error("config.n_layers"))? as usize;
    let law = match config["mlp_act"].as_str() {
        Some("gelu_tanh") => Law::GeluTanh,
        Some("gelu") => Law::Gelu,
        Some("relu") => Law::Relu,
        Some("silu") => Law::Silu,
        other => return Err(error(format!("MLP law {other:?}"))),
    };
    if config["mlp_gated"].as_bool().unwrap_or(false) {
        return Err(error("a gated MLP"));
    }
    let mlps = (0..layers)
        .map(|l| -> Result<Mlp, String> {
            let (read, write) = (tensor(export, &record, &format!("blocks.{l}.mlp.c_fc"))?, tensor(export, &record, &format!("blocks.{l}.mlp.down_proj"))?);
            if write.dim() != (read.ncols(), read.nrows()) {
                return Err(error(format!("layer {l}: W_fc {:?} and W_down {:?}", read.dim(), write.dim())));
            }
            Ok(Mlp { read, write, law, up: None })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut parts = Vec::new();
    for f in load_factors(decomposition)? {
        let map = match f.kind {
            Kind::Up => Map::Up,
            Kind::Down => Map::Down,
            _ => continue,
        };
        let mlp = mlps.get(f.layer).ok_or_else(|| error(format!("{}: layer {} of {layers}", f.name, f.layer)))?;
        let (rows, cols) = match map {
            Map::Up => mlp.read.dim(),
            Map::Down => mlp.write.dim(),
        };
        if (f.u.ncols(), f.v.nrows()) != (rows, cols) {
            return Err(error(format!("{}: slices of {} × {} for a {rows} × {cols} map", f.name, f.u.ncols(), f.v.nrows())));
        }
        for i in 0..f.subcomponents() {
            let (u, v) = (f.u.row(i).to_owned(), f.v.column(i).to_owned());
            if u.iter().all(|x| *x == 0.0) || v.iter().all(|x| *x == 0.0) {
                continue;
            }
            parts.push(VpdPart { block: 2 * f.layer + 1, layer: f.layer, map, index: i, u, v });
        }
    }
    Ok((mlps, parts))
}

/// Test data: fixed parts of the stream's `width` in MLP blocks 1 and 3 (layers 0 and 1): per
/// block an MLP of 6 hidden units under the tanh GELU, and one slice of its up map and one of its
/// down map.
#[cfg(test)]
pub(crate) fn random_slices(width: usize, seed: u64) -> Vec<Slice> {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut normal = |rows: usize, cols: usize, scale: f64| Array2::from_shape_fn((rows, cols), |_| scale * (rng.random::<f64>() - 0.5));
    let mut out = Vec::new();
    for layer in [0, 1] {
        let mlp = Arc::new(Mlp { read: normal(6, width, 1.0), write: normal(width, 6, 1.0), law: Law::GeluTanh, up: None });
        let (up, down) = ((normal(1, 6, 2.0), normal(1, width, 2.0)), (normal(1, width, 2.0), normal(1, 6, 2.0)));
        for (map, (u, v)) in [(Map::Up, up), (Map::Down, down)] {
            let part = VpdPart { block: 2 * layer + 1, layer, map, index: 0, u: u.row(0).to_owned(), v: v.row(0).to_owned() };
            out.push(Slice { part, mlp: Arc::clone(&mlp) });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use crate::interchange::{Batch, BlockEngine, Edit, Edits, Experiment, FACTORS, Interchange, Patch, reads};
    use crate::operator_program::SlotValues;
    use crate::run_check::{layer_nodes, split_sites};
    use gam_gpu::tensor::Device;

    /// An MLP's neuron slices sum to its down map: removing every neuron (α = 0) moves the output by
    /// exactly minus the output (1e-12), and each neuron's edit is its activation times its column.
    /// A random slice is a rank-one map of the down map's top singular value (`‖u‖ ‖v‖` equal to it,
    /// checked against the largest `‖W x‖` over the power iteration's own direction).
    /// A gated MLP's slices (`h = φ(W_fc x) ⊙ (W_up x)`, SiLU): each edit is its slice's weight
    /// edit of the MLP (1e-12), and its tangent and pullback are transposes of each other (1e-12),
    /// for slices of either map at several factors.
    #[test]
    fn a_gated_mlps_slices_are_weight_edits_and_their_derivatives_transpose() {
        let mut rng = StdRng::seed_from_u64(31);
        let mut normal = |rows: usize, cols: usize| Array2::from_shape_fn((rows, cols), |_| rng.random::<f64>() - 0.5);
        let mlp = Arc::new(Mlp { read: normal(7, 5), write: normal(5, 7), law: Law::Silu, up: Some(normal(7, 5)) });
        let (x, dx, g) = (normal(1, 5).row(0).to_owned(), normal(1, 5).row(0).to_owned(), normal(1, 5).row(0).to_owned());
        let output = |m: &Mlp| m.output(x.view());
        let mut slices = neuron_slices(&mlp, 0);
        slices.extend(random_slices_of(&mlp, 0, Map::Up, 3, 4).unwrap());
        slices.extend(random_slices_of(&mlp, 0, Map::Down, 3, 5).unwrap());
        for slice in &slices {
            let piece = slice.part.u.clone().insert_axis(ndarray::Axis(1)).dot(&slice.part.v.clone().insert_axis(ndarray::Axis(0)));
            for alpha in [0.0, 0.5, 3.0] {
                let edited = match slice.part.map {
                    Map::Up => Mlp { read: &mlp.read + &(&piece * (alpha - 1.0)), ..(*mlp).clone() },
                    Map::Down => Mlp { write: &mlp.write + &(&piece * (alpha - 1.0)), ..(*mlp).clone() },
                };
                let (want, got) = (output(&edited) - output(&mlp), slice.edit(x.view(), alpha));
                let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
                assert!(want.iter().zip(&got).all(|(a, b)| (a - b).abs() <= 1e-12 * scale), "{:?} {} at {alpha}", slice.part.map, slice.part.index);
                let (forward, backward) = (slice.tangent(x.view(), alpha, dx.view()).dot(&g), slice.pullback(x.view(), alpha, g.view()).dot(&dx));
                assert!((forward - backward).abs() <= 1e-12 * (1.0 + backward.abs()), "{:?} {} at {alpha}: tangent {forward}, pullback {backward}", slice.part.map, slice.part.index);
            }
        }
    }

    #[test]
    fn neuron_slices_sum_to_the_down_map_and_random_slices_have_its_scale() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut normal = |rows: usize, cols: usize| Array2::from_shape_fn((rows, cols), |_| rng.random::<f64>() - 0.5);
        let mlp = Arc::new(Mlp { read: normal(7, 5), write: normal(5, 7), law: Law::GeluTanh, up: None });
        let x = normal(5, 1).column(0).to_owned();
        let neurons = neuron_slices(&mlp, 0);
        assert_eq!(neurons.len(), 7);
        let total = neurons.iter().fold(Array1::<f64>::zeros(5), |acc, n| acc + n.edit(x.view(), 0.0));
        let output = mlp.output(x.view());
        for (a, b) in total.iter().zip(&output) {
            assert!((a + b).abs() <= 1e-12 * output.iter().fold(1.0_f64, |m, v| m.max(v.abs())), "{a} against −{b}");
        }
        let slices = random_slices_of(&mlp, 0, Map::Down, 3, 9).unwrap();
        let top = {
            // The square root of WᵀW's largest eigenvalue (a symmetric eigensolver, not the SVD).
            let gram = mlp.write.t().dot(&mlp.write);
            let gram = (&gram + &gram.t()) * 0.5;
            let eigen = gam_linalg::decompose::eigh(gram.view(), gam_linalg::roundoff::SymmetricAssembly::Mirrored, None).unwrap();
            eigen.values[eigen.values.len() - 1].sqrt()
        };
        for s in &slices {
            let scale = s.part.u.dot(&s.part.u).sqrt() * s.part.v.dot(&s.part.v).sqrt();
            assert!((scale - top).abs() <= 1e-9 * top, "‖u‖‖v‖ {scale} against the top singular value {top}");
        }
        assert_ne!(slices[0].part.v, slices[1].part.v, "independent draws");
    }

    /// The tiny Qwen3 export's model and its starting library (an exact copy of `M`), with its
    /// sequences of 12 tokens.
    fn tiny(name: &str) -> (crate::operator_program::OperatorProgram, Vec<crate::run_check::LayerNodes>, crate::library_mdl::Explanation, Vec<Vec<u32>>) {
        let dir = crate::test_support::tiny_qwen3_export(name, 2);
        let imported = crate::import::import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let explanation = crate::library_mdl::explanation(&native, &layers).expect("the library");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        (native, layers, explanation, sequences)
    }

    /// An edit of a fixed part on `M` through the engine: after layer 0's MLP block of the tiny
    /// Qwen3 export, the stream moves by exactly `Slice::edit` of `M`'s own read at the edited row
    /// and nowhere else (1e-12 of the stream's scale), for both maps' slices and every factor.
    #[test]
    fn an_edit_of_a_fixed_part_on_m_adds_its_function_of_the_read() {
        let (native, layers, explanation, sequences) = tiny("vpd_parts_hand");
        let d = Device::host();
        let mut x = Interchange::new(&d, &native, &layers, &explanation.artifact, &explanation.trainable, reads(&native, &layers).expect("the reads"), 1 << 30, 64).expect("the experiments");
        let width = BlockEngine::width(&x.models().0);
        x.set_fixed_parts(random_slices(width, 5)).expect("the fixed parts");
        let (m, _) = x.models();
        let sites = m.part_sites().expect("M's part sites");
        let (read, _) = sites.nodes(1).expect("layer 0's MLP holds parts");
        let (ranges, tok) = (vec![0..12], vec![sequences[0].as_slice()]);
        let mut entering = d.zeros(12, width).expect("zeros");
        m.forward(0, &mut entering, &ranges, &tok, None, false).expect("attention");
        let mut plain = d.copy(&entering).expect("copy");
        let trace = m.forward(1, &mut plain, &ranges, &tok, None, true).expect("the MLP").expect("a tape");
        let normed = d.download(trace.value(read).expect("the read")).expect("download");
        let plain = d.download(&plain).expect("download");
        for part in 0..2 {
            for (factor, alpha) in FACTORS.iter().enumerate() {
                let row = 3 + factor;
                let edits = Edits::with_parts(&d, &[], m.values(), &[(row, Edit::Fixed { part, factor })], Some(sites)).expect("the edits");
                let mut edited = d.copy(&entering).expect("copy");
                m.forward(1, &mut edited, &ranges, &tok, Some(&edits), false).expect("the edited MLP");
                let moved = d.download(&edited).expect("download") - &plain;
                let want = x.fixed_parts()[part].edit(normed.row(row), *alpha);
                assert!(want.iter().any(|v| v.abs() > 1e-6), "part {part} at α {alpha} moves the stream");
                let scale = moved.iter().chain(plain.iter()).fold(0.0f64, |m, v| m.max(v.abs()));
                for r in 0..12 {
                    for c in 0..width {
                        let expected = if r == row { want[c] } else { 0.0 };
                        assert!((moved[[r, c]] - expected).abs() <= 1e-12 * scale, "part {part}, α {alpha}, row {r}, column {c}: moved {} against {expected}", moved[[r, c]]);
                    }
                }
            }
        }
    }

    /// The starting library is an exact copy of `M`, so every edit of a fixed part at every factor
    /// scores zero bits, with and without the reverse pass; with `P` applying no edit, an edit
    /// scores `KL(M_e ‖ M)`, positive for α ≠ 1.
    #[test]
    fn an_exact_copy_scores_every_fixed_part_edit_zero() {
        let (native, layers, explanation, sequences) = tiny("vpd_parts_copy");
        let d = Device::host();
        let mut x = Interchange::new(&d, &native, &layers, &explanation.artifact, &explanation.trainable, reads(&native, &layers).expect("the reads"), 1 << 30, 64).expect("the experiments");
        let width = BlockEngine::width(&x.models().0);
        x.set_fixed_parts(random_slices(width, 7)).expect("the fixed parts");
        let batch = Batch::new(sequences[..3].to_vec(), sequences[3..].to_vec()).expect("the batch");
        let mut experiments = Vec::new();
        for part in 0..x.fixed_parts().len() {
            let block = x.fixed_parts()[part].block();
            for factor in 0..FACTORS.len() {
                let base = (part + factor) % 3;
                experiments.push(Experiment { base, source: base, explained: vec![true; 4], patch: Some(Patch::FixedPart { part, factor, block }), position: 1 + (part + 3 * factor) % 11 });
            }
        }
        for gradient in [false, true] {
            let bits = x.evaluate(&batch, &experiments, gradient).expect("evaluate").bits;
            assert!(bits.iter().flatten().all(|b| b.abs() <= 1e-9), "{bits:?}");
        }
        x.unedited_explanation();
        let effects = x.evaluate(&batch, &experiments, false).expect("evaluate").bits;
        for (e, bits) in experiments.iter().zip(&effects) {
            let sum: f64 = bits.iter().sum();
            assert!(sum > 1e-9, "{e:?}: KL(M_e ‖ M) {sum}");
        }
    }

    /// On `M`, a part's edit is the weight edit `W + (α − 1) u_i vᵢᵀ` of its map: for an MLP whose
    /// maps are sums of four slices each, every slice of either map at every factor of
    /// `interchange::FACTORS` moves the output at a random read by exactly the output of the edited
    /// MLP less the original's (1e-12 of the output's scale), α = 1 moves nothing, and under the
    /// smooth laws the pullback matches central differences (1e-6) and the tangent is its
    /// transpose (1e-12).
    #[test]
    fn a_parts_edit_is_its_slices_weight_edit_on_m() {
        let (d, hidden, slices) = (5, 7, 4);
        let mut rng = StdRng::seed_from_u64(11);
        let mut normal = |rows: usize, cols: usize| Array2::from_shape_fn((rows, cols), |_| rng.random::<f64>() * 2.0 - 1.0);
        let (up_u, up_v, down_u, down_v) = (normal(slices, hidden), normal(d, slices), normal(slices, d), normal(hidden, slices));
        let x = normal(d, 1).column(0).to_owned();
        for law in [Law::GeluTanh, Law::Gelu, Law::Relu] {
            let mlp = Mlp { read: up_v.dot(&up_u).t().to_owned(), write: down_v.dot(&down_u).t().to_owned(), law, up: None };
            let base = mlp.output(x.view());
            for (map, u, v) in [(Map::Up, &up_u, &up_v), (Map::Down, &down_u, &down_v)] {
                for i in 0..slices {
                    let part = VpdPart { block: 1, layer: 0, map, index: i, u: u.row(i).to_owned(), v: v.column(i).to_owned() };
                    for alpha in crate::interchange::FACTORS.into_iter().chain([1.0]) {
                        let slice = part.u.view().insert_axis(ndarray::Axis(1)).dot(&part.v.view().insert_axis(ndarray::Axis(0))) * (alpha - 1.0);
                        let edited = match map {
                            Map::Up => Mlp { read: &mlp.read + &slice, ..mlp.clone() },
                            Map::Down => Mlp { write: &mlp.write + &slice, ..mlp.clone() },
                        };
                        let expected = edited.output(x.view()) - &base;
                        let got = part.edit(&mlp, x.view(), alpha);
                        let scale = base.iter().chain(&expected).fold(1.0_f64, |m, v| m.max(v.abs()));
                        for (a, b) in got.iter().zip(&expected) {
                            assert!((a - b).abs() <= 1e-12 * scale, "{law:?} {map:?} slice {i} at α {alpha}: {a} against {b}");
                        }
                        if alpha == 1.0 {
                            assert!(got.iter().all(|v| *v == 0.0), "α = 1 moves the output");
                        }
                        // The pullback against central differences of `ḡ·edit` (smooth laws).
                        if law != Law::Relu {
                            let cotangent = Array1::from_shape_fn(d, |c| 0.3 + 0.1 * c as f64);
                            let pulled = part.pullback(&mlp, x.view(), alpha, cotangent.view());
                            // The tangent is the pullback's transpose: ḡ·(J dx) = (Jᵀ ḡ)·dx.
                            let dx = Array1::from_shape_fn(d, |c| 0.7 - 0.2 * c as f64);
                            let (forward, backward) = (part.tangent(&mlp, x.view(), alpha, dx.view()).dot(&cotangent), pulled.dot(&dx));
                            assert!((forward - backward).abs() <= 1e-12 * (1.0 + backward.abs()), "{law:?} {map:?} slice {i} at α {alpha}: tangent {forward} against pullback {backward}");
                            for c in 0..d {
                                let h = 1e-5;
                                let (mut up, mut down) = (x.clone(), x.clone());
                                up[c] += h;
                                down[c] -= h;
                                let numeric = (part.edit(&mlp, up.view(), alpha) - part.edit(&mlp, down.view(), alpha)).dot(&cotangent) / (2.0 * h);
                                assert!((pulled[c] - numeric).abs() <= 1e-6 * (1.0 + numeric.abs()), "{law:?} {map:?} slice {i} at α {alpha}, column {c}: {} against {numeric}", pulled[c]);
                            }
                        }
                    }
                }
            }
        }
    }
}

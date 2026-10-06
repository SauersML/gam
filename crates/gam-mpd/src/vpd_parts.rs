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

/// One layer's MLP of `M`: `out = write · φ(read · x)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Mlp {
    /// `W_fc` (hidden × d).
    pub read: Array2<f64>,
    /// `W_down` (d × hidden).
    pub write: Array2<f64>,
    pub law: Law,
}

impl Mlp {
    /// `φ(W_fc x + shift)`, the hidden activations at the read `x` with the pre-activations moved by
    /// `shift`.
    fn hidden(&self, x: ArrayView1<f64>, shift: Option<&Array1<f64>>) -> Array1<f64> {
        let mut pre = self.read.dot(&x);
        if let Some(shift) = shift {
            pre += shift;
        }
        pre.mapv(|t| self.law.apply(t))
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

    /// The pullback of [`VpdPart::edit`] in the read: `J(x)ᵀ ḡ` for the output's cotangent `ḡ`.
    /// With `z = W_fc x` and `φ'` the law's derivative: for a slice of `W_down`,
    /// `(α − 1)(u·ḡ) W_fcᵀ(φ'(z) ⊙ v)`; for a slice of `W_fc`, with `z' = z + (α − 1)(v·x) u` and
    /// `r = W_downᵀ ḡ`, `W_fcᵀ((φ'(z') − φ'(z)) ⊙ r) + (α − 1)(u·(φ'(z') ⊙ r)) v`.
    #[must_use]
    pub fn pullback(&self, mlp: &Mlp, x: ArrayView1<f64>, alpha: f64, cotangent: ArrayView1<f64>) -> Array1<f64> {
        let pre = mlp.read.dot(&x);
        let slope = |z: &Array1<f64>| z.mapv(|t| mlp.law.derivative(t));
        match self.map {
            Map::Down => mlp.read.t().dot(&(slope(&pre) * &self.v)) * ((alpha - 1.0) * self.u.dot(&cotangent)),
            Map::Up => {
                let moved = &pre + &(&self.u * ((alpha - 1.0) * self.v.dot(&x)));
                let r = mlp.write.t().dot(&cotangent);
                let (after, before) = (slope(&moved) * &r, slope(&pre) * &r);
                mlp.read.t().dot(&(&after - &before)) + &self.v * ((alpha - 1.0) * self.u.dot(&after))
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
            Ok(Mlp { read, write, law })
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use crate::interchange::{Batch, BlockEngine, Edit, Edits, Experiment, FACTORS, Interchange, Patch, reads};
    use crate::operator_program::SlotValues;
    use crate::run_check::{layer_nodes, split_sites};
    use gam_gpu::tensor::Device;

    /// Fixed parts of the stream's `width` in MLP blocks 1 and 3 (layers 0 and 1): per block an MLP
    /// of 6 hidden units under the tanh GELU, and one slice of its up map and one of its down map.
    fn random_slices(width: usize, seed: u64) -> Vec<Slice> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut normal = |rows: usize, cols: usize, scale: f64| Array2::from_shape_fn((rows, cols), |_| scale * (rng.random::<f64>() - 0.5));
        let mut out = Vec::new();
        for layer in [0, 1] {
            let mlp = Arc::new(Mlp { read: normal(6, width, 1.0), write: normal(width, 6, 1.0), law: Law::GeluTanh });
            let (up, down) = ((normal(1, 6, 2.0), normal(1, width, 2.0)), (normal(1, width, 2.0), normal(1, 6, 2.0)));
            for (map, (u, v)) in [(Map::Up, up), (Map::Down, down)] {
                let part = VpdPart { block: 2 * layer + 1, layer, map, index: 0, u: u.row(0).to_owned(), v: v.row(0).to_owned() };
                out.push(Slice { part, mlp: Arc::clone(&mlp) });
            }
        }
        out
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
                let edits = Edits::with_parts(&d, &[], m.values(), &[(row, Edit::Fixed { part, factor })], Some(sites), None).expect("the edits");
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
    /// smooth laws the pullback matches central differences (1e-6).
    #[test]
    fn a_parts_edit_is_its_slices_weight_edit_on_m() {
        let (d, hidden, slices) = (5, 7, 4);
        let mut rng = StdRng::seed_from_u64(11);
        let mut normal = |rows: usize, cols: usize| Array2::from_shape_fn((rows, cols), |_| rng.random::<f64>() * 2.0 - 1.0);
        let (up_u, up_v, down_u, down_v) = (normal(slices, hidden), normal(d, slices), normal(slices, d), normal(hidden, slices));
        let x = normal(d, 1).column(0).to_owned();
        for law in [Law::GeluTanh, Law::Gelu, Law::Relu] {
            let mlp = Mlp { read: up_v.dot(&up_u).t().to_owned(), write: down_v.dot(&down_u).t().to_owned(), law };
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

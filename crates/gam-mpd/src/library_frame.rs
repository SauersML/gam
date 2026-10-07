//! Frame starts for library_vpd (#2951): `M`'s maps cut exactly into components through an
//! overcomplete frame of each read space, a start that needs no VPD (Qwen3 has none) and that can
//! hold more components than a map has rank.
//!
//! A frame of a read space of width `d` is `C = 4d` atoms `f_i` (the rows of `F`, `C × d`) that
//! span it; its canonical dual is `g_i = (FᵀF)⁻¹ f_i`, so `Σ_i f_i g_iᵀ = FᵀF (FᵀF)⁻¹ = I` and every
//! map `W` reading the space is exactly `Σ_i (W f_i) g_iᵀ`: slice `i` reads `g_iᵀ x`, `x`'s
//! coefficient on atom `i`, and writes `W f_i`. All components on is `M`, whatever the frame.
//!
//! The toy gate (`bench/toys_2951`) found that neuron and head starts cannot recover features held
//! in superposition (TMS: 10 hidden units, 40 features; the fitter cannot split a component), so a
//! map whose read is not followed by an elementwise nonlinearity on the sliced axis (attention's q,
//! k and v reading the stream, o reading the heads' outputs, an MLP with the identity law) is cut
//! through a frame, while an MLP whose law is elementwise nonlinear keeps its neuron groups (the
//! law privileges the neuron axis): per neuron its c_fc row and down_proj column.
//!
//! Two frames ([`FrameKind`]):
//! * `Tight`: `C` standard normal atoms made a Parseval frame, `F ← F (FᵀF)^{-1/2}`, so its dual
//!   is itself; no data.
//! * `Dictionary`: a one-sparse dictionary of `M`'s activations at the read on fitting rows
//!   (spherical k-means: each row is assigned the atom of largest `|f·x|` among unit atoms, each
//!   atom becomes the leading eigenvector of its rows' second moment; started at the tight frame,
//!   repeated until no assignment changes). The atoms are directions the activations take one at a
//!   time; no ground truth is used. Directions the activations never take are completed by the
//!   unresolved eigenvectors of the atoms' Gram, so the frame spans its space.
//!
//! Each component is one atom's slices on the maps reading its space, with one own gate at that
//! read, started at `τ = 0`: on wherever it reads anything, which is everywhere its output is
//! nonzero, so the start is exact and no gate starts saturated. The direction arm gates the same
//! components by `gᵀx − τ` with `g` the component's own read direction (its first read slice, the
//! sign that makes its mean coefficient on the fitting rows positive), `c = 0` and `τ = 0`: on
//! where the coefficient is positive, so it is exact only where the coefficients have one sign.
//! Every gate's width is its read's spread on the fitting rows (library_vpd's learned width, as
//! vpd_start sets it); a direction is written in those units, width 1.

use crate::{
    artifact::Artifact,
    explanation_battery::KINDS,
    operator_program::{FamilyInputs, Law, Node, OperatorProgram},
    run_check::LayerNodes,
};
use gam_linalg::{decompose::eigh, faer_ndarray::fast_ata, roundoff::SymmetricAssembly};
use ndarray::{Array2, Axis, concatenate};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde_json::{Value, json};
use std::path::Path;

fn error(e: impl std::fmt::Display) -> String {
    format!("library frame: {e}")
}

/// How a read space's frame is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind {
    Tight,
    Dictionary,
}

/// A frame of a read space: its atoms as rows, `C × d`.
#[derive(Clone, Debug)]
pub struct Frame {
    pub atoms: Array2<f64>,
}

/// The frame's atoms per dimension of the space.
pub const REDUNDANCY: usize = 4;

impl Frame {
    /// `count` standard normal atoms in `dim` dimensions made Parseval, `FᵀF = I`.
    pub fn random_tight(count: usize, dim: usize, seed: u64) -> Result<Self, String> {
        if count < dim || dim == 0 {
            return Err(error(format!("{count} atoms cannot span {dim} dimensions")));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let draws = Array2::from_shape_fn((count, dim), |_| {
            let (u, angle) = (1.0 - rng.random::<f64>(), std::f64::consts::TAU * rng.random::<f64>());
            (-2.0 * u.ln()).sqrt() * angle.cos()
        });
        let gram = fast_ata(&draws);
        let root = eigh(gram.view(), SymmetricAssembly::Mirrored, None).map_err(|e| error(format!("{e:?}")))?.psd_map(0.0, |v| v.powf(-0.5)).map_err(|e| error(format!("{e:?}")))?;
        Ok(Self { atoms: draws.dot(&root) })
    }

    /// A one-sparse dictionary of `rows` (one per row, `d` columns), started at `start`: spherical
    /// k-means of the rows with nonzero norm (module note). Atoms are unit vectors.
    pub fn dictionary(rows: &Array2<f64>, start: Frame) -> Result<Self, String> {
        let dim = start.atoms.ncols();
        if rows.ncols() != dim {
            return Err(error(format!("rows of {} columns for a frame of {dim}", rows.ncols())));
        }
        let live: Vec<usize> = (0..rows.nrows()).filter(|&r| rows.row(r).iter().any(|v| *v != 0.0)).collect();
        let x = rows.select(Axis(0), &live);
        let mut atoms = start.atoms;
        for mut atom in atoms.rows_mut() {
            let norm = atom.dot(&atom).sqrt();
            atom.mapv_inplace(|v| v / norm);
        }
        let mut assigned = vec![usize::MAX; x.nrows()];
        // Assignments only ever change to a strictly better atom, and the atoms' fit only improves,
        // so the loop ends; the pass bound guards a tie cycle.
        for _ in 0..1000 {
            let scores = x.dot(&atoms.t());
            let next: Vec<usize> = scores.rows().into_iter().map(|r| (0..r.len()).max_by(|&a, &b| r[a].abs().total_cmp(&r[b].abs())).unwrap_or(0)).collect();
            if next == assigned {
                break;
            }
            assigned = next;
            for (i, mut atom) in atoms.rows_mut().into_iter().enumerate() {
                let mine: Vec<usize> = (0..x.nrows()).filter(|&r| assigned[r] == i).collect();
                if mine.is_empty() {
                    continue;
                }
                let moment = fast_ata(&x.select(Axis(0), &mine));
                let top = eigh(moment.view(), SymmetricAssembly::Mirrored, Some((dim - 1, dim))).map_err(|e| error(format!("{e:?}")))?;
                atom.assign(&top.vectors.column(0));
            }
        }
        // Directions the rows never take (a toy's stream slots that hold zeros at this read) have
        // no atom; they are completed by the unresolved eigenvectors of the atoms' Gram, so the
        // frame spans the space and every map is still cut exactly (their components read nothing
        // on the fitting rows).
        let gram = fast_ata(&atoms);
        let decomposition = eigh(gram.view(), SymmetricAssembly::Mirrored, None).map_err(|e| error(format!("{e:?}")))?;
        let largest = decomposition.values.iter().fold(0.0_f64, |m, v| m.max(*v));
        let floor = decomposition.band.max(largest * f64::EPSILON * dim as f64);
        let missing: Vec<usize> = (0..dim).filter(|&k| decomposition.values[k] <= floor).collect();
        if !missing.is_empty() {
            let complement = decomposition.vectors.select(Axis(1), &missing).t().to_owned();
            atoms = concatenate(Axis(0), &[atoms.view(), complement.view()]).map_err(error)?;
        }
        let frame = Self { atoms };
        frame.dual()?;
        Ok(frame)
    }

    /// The canonical dual, `C × d`: rows `g_i = (FᵀF)⁻¹ f_i`. Refused for atoms that do not span.
    pub fn dual(&self) -> Result<Array2<f64>, String> {
        let gram = fast_ata(&self.atoms);
        let decomposition = eigh(gram.view(), SymmetricAssembly::Mirrored, None).map_err(|e| error(format!("{e:?}")))?;
        let largest = decomposition.values.iter().fold(0.0_f64, |m, v| m.max(*v));
        if decomposition.values.iter().any(|v| *v <= decomposition.band.max(largest * f64::EPSILON * gram.nrows() as f64)) {
            return Err(error("the atoms do not span their space"));
        }
        let inverse = decomposition.psd_map(0.0, |v| 1.0 / v).map_err(|e| error(format!("{e:?}")))?;
        Ok(self.atoms.dot(&inverse))
    }

    /// The frame of `kind` for a space of `rows` (`M`'s activations there, one per row).
    pub fn of(kind: FrameKind, rows: &Array2<f64>, seed: u64) -> Result<Self, String> {
        let dim = rows.ncols();
        let tight = Self::random_tight(REDUNDANCY * dim, dim, seed)?;
        match kind {
            FrameKind::Tight => Ok(tight),
            FrameKind::Dictionary => Self::dictionary(rows, tight),
        }
    }
}

/// One component of a start file, as library_vpd reads it: its read, `τ = 0`, its gate's width
/// and its slices.
fn component(read: Value, width: f64, slices: &[[usize; 2]]) -> Value {
    json!({"read": read, "tau": 0.0, "width": width, "slices": slices})
}

/// The spread (standard deviation) of `values`; a read that is the same on every fitting row
/// (a zero map's, or a completing atom's that the rows never take) has no scale to set, and its
/// gate is given width 1.
fn spread(values: impl Iterator<Item = f64>) -> f64 {
    let (mut n, mut sum, mut squares) = (0.0, 0.0, 0.0);
    for v in values {
        n += 1.0;
        sum += v;
        squares += v * v;
    }
    let variance = if n > 0.0 { (squares / n - (sum / n).powi(2)).max(0.0) } else { 0.0 };
    if variance > 0.0 { variance.sqrt() } else { 1.0 }
}

/// An own gate reading `copies` slices that each read `g` on `rows`: its width, the spread of
/// `‖V_bᵀx‖ = √copies |gᵀx|`.
fn own_width(rows: &Array2<f64>, g: ndarray::ArrayView1<f64>, copies: f64) -> f64 {
    spread(rows.dot(&g).iter().map(|v| copies.sqrt() * v.abs()))
}

/// A direction gate reading `g` on `rows`, signed so that its mean on the rows is not negative and
/// written in units of its spread there (`g` over the spread, width 1), with no constant.
fn direction_read(rows: &Array2<f64>, g: ndarray::ArrayView1<f64>, site: usize) -> Value {
    let values = rows.dot(&g);
    let sign = if values.sum() >= 0.0 { 1.0 } else { -1.0 };
    let scale = spread(values.iter().copied());
    let mut coefficients: Vec<f64> = g.iter().map(|v| sign * v / scale).collect();
    coefficients.push(0.0);
    json!({"direction": {"site": site, "coefficients": coefficients}})
}

/// The matrix of the native operator named `name`.
fn operator(native: &OperatorProgram, name: &str) -> Result<Array2<f64>, String> {
    let found: Vec<&std::sync::Arc<crate::operator_program::Operator>> = native.operators.iter().filter(|op| op.name == name).collect();
    match found[..] {
        [op] => Ok(op.matrix()),
        _ => Err(error(format!("no unique native operator {name}"))),
    }
}

/// A site's slices, written `U` (`C × out`) and read `V` (`in × C`), and per slice its component.
struct Site {
    u: Array2<f64>,
    v: Array2<f64>,
}

impl Site {
    /// `W` cut through the frame with dual `dual`: slice `i` writes `W f_i` and reads `g_i`.
    fn framed(w: &Array2<f64>, frame: &Frame, dual: &Array2<f64>) -> Self {
        Self { u: frame.atoms.dot(&w.t()), v: dual.t().to_owned() }
    }

    /// A zero map: one zero slice.
    fn zero(rows: usize, cols: usize) -> Self {
        Self { u: Array2::zeros((1, rows)), v: Array2::zeros((cols, 1)) }
    }
}

/// The frame start of the split native program `native` with its `layers`, fitted on `fitting`
/// (rows of `M`'s input): per kind of frame its decomposition in `dir/{tight,dictionary}/` (the
/// layout explanation_battery::load_factors reads: config.sites `h.{l}.{kind}` in `M`'s order,
/// `{site}.U` `C × out`, `{site}.V` `in × C`) and its arms in `dir/{tight,dictionary}/start.json`
/// (`frame_own`, `frame_direction`; module note). Returns a summary.
pub fn frame_start(native: &OperatorProgram, layers: &[LayerNodes], fitting: &FamilyInputs, seed: u64, dir: &Path) -> Result<Value, String> {
    let artifact = Artifact::native(native)?;
    let trace = artifact.execute(fitting)?;
    let value = |node: usize| -> Result<Array2<f64>, String> { Ok(trace.values[artifact.place(node).ok_or_else(|| error(format!("no node {node}")))?].clone()) };
    let mut summary = Vec::new();
    for (kind, name) in [(FrameKind::Tight, "tight"), (FrameKind::Dictionary, "dictionary")] {
        let out = dir.join(name);
        std::fs::create_dir_all(&out).map_err(error)?;
        let (mut sites, mut files) = (Vec::new(), serde_json::Map::new());
        let (mut own, mut direction) = (Vec::new(), Vec::new());
        for (l, layer) in layers.iter().enumerate() {
            let site = |k: usize| KINDS.len() * l + k;
            let heads = layer.reads.len();
            let weight = |part: &str| operator(native, &format!("blocks.{l}.{part}"));
            let stack = |parts: Vec<Array2<f64>>, axis: usize| -> Result<Array2<f64>, String> {
                let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
                concatenate(Axis(axis), &views).map_err(error)
            };
            let wq = stack((0..heads).map(|h| weight(&format!("q{h}"))).collect::<Result<_, _>>()?, 0)?;
            let kv = layer.keys.len();
            let wk = stack((0..kv).map(|g| weight(&format!("k{g}"))).collect::<Result<_, _>>()?, 0)?;
            let wv = stack((0..kv).map(|g| weight(&format!("v{g}"))).collect::<Result<_, _>>()?, 0)?;
            let wo = stack((0..heads).map(|h| weight(&format!("o{h}"))).collect::<Result<_, _>>()?, 1)?;
            let (up, down) = (weight("c_fc")?, weight("down_proj")?);
            let mut layer_sites: Vec<Site> = Vec::with_capacity(KINDS.len());
            // The attention's input: one frame for q, k and v; a component per atom.
            let x = value(layer.normed_stream)?;
            let zero_attention = [&wq, &wk, &wv].iter().all(|w| w.iter().all(|v| *v == 0.0));
            let (attention_frame, attention_dual) = if zero_attention {
                for w in [&wq, &wk, &wv] {
                    layer_sites.push(Site::zero(w.nrows(), w.ncols()));
                }
                own.push(component(json!({"own": [site(0), 0]}), 1.0, &[[site(0), 0], [site(1), 0], [site(2), 0]]));
                direction.push(component(json!({"direction": {"site": site(0), "coefficients": vec![0.0; x.ncols() + 1]}}), 1.0, &[[site(0), 0], [site(1), 0], [site(2), 0]]));
                (None, None)
            } else {
                let frame = Frame::of(kind, &x, seed ^ (l as u64 * 6 + 1))?;
                let dual = frame.dual()?;
                for w in [&wq, &wk, &wv] {
                    layer_sites.push(Site::framed(w, &frame, &dual));
                }
                (Some(frame), Some(dual))
            };
            // o reads the heads' outputs, concatenated.
            let heads_out = stack(layer.reads.iter().map(|&r| value(r)).collect::<Result<_, _>>()?, 1)?;
            let zero_o = wo.iter().all(|v| *v == 0.0);
            let o_frame = if zero_o {
                layer_sites.push(Site::zero(wo.nrows(), wo.ncols()));
                None
            } else {
                let frame = Frame::of(kind, &heads_out, seed ^ (l as u64 * 6 + 2))?;
                let dual = frame.dual()?;
                layer_sites.push(Site::framed(&wo, &frame, &dual));
                Some((frame, dual))
            };
            // The MLP: framed under the identity law, neuron groups under a nonlinear one.
            let Node::Pointwise { laws, .. } = &native.nodes[layer.active] else {
                return Err(error(format!("layer {l}: the MLP activation is not one pointwise law")));
            };
            let linear = laws.iter().all(|law| *law == Law::Identity);
            let h2 = value(layer.normed)?;
            let zero_mlp = up.iter().all(|v| *v == 0.0) && down.iter().all(|v| *v == 0.0);
            let mlp = if zero_mlp {
                layer_sites.push(Site::zero(up.nrows(), up.ncols()));
                layer_sites.push(Site::zero(down.nrows(), down.ncols()));
                None
            } else if linear {
                let up_frame = Frame::of(kind, &h2, seed ^ (l as u64 * 6 + 3))?;
                let up_dual = up_frame.dual()?;
                layer_sites.push(Site::framed(&up, &up_frame, &up_dual));
                let hidden = value(layer.active)?;
                let down_frame = Frame::of(kind, &hidden, seed ^ (l as u64 * 6 + 4))?;
                let down_dual = down_frame.dual()?;
                layer_sites.push(Site::framed(&down, &down_frame, &down_dual));
                Some((up_frame.atoms.nrows(), down_frame.atoms.nrows(), Some((up_dual, down_dual))))
            } else {
                // Per neuron its c_fc row and down_proj column.
                let m = up.nrows();
                layer_sites.push(Site { u: Array2::eye(m), v: up.t().to_owned() });
                layer_sites.push(Site { u: down.t().to_owned(), v: Array2::eye(m) });
                Some((m, m, None))
            };
            // The components, per site group, own and direction; each gate's width is its read's
            // spread on the fitting rows (a direction is written in those units, width 1).
            if let (Some(frame), Some(dual)) = (&attention_frame, &attention_dual) {
                for i in 0..frame.atoms.nrows() {
                    let slices = [[site(0), i], [site(1), i], [site(2), i]];
                    own.push(component(json!({"own": [site(0), i]}), own_width(&x, dual.row(i), 3.0), &slices));
                    direction.push(component(direction_read(&x, dual.row(i), site(0)), 1.0, &slices));
                }
            }
            if let Some((frame, dual)) = &o_frame {
                for i in 0..frame.atoms.nrows() {
                    own.push(component(json!({"own": [site(3), i]}), own_width(&heads_out, dual.row(i), 1.0), &[[site(3), i]]));
                    direction.push(component(direction_read(&heads_out, dual.row(i), site(3)), 1.0, &[[site(3), i]]));
                }
            }
            match &mlp {
                Some((ups, downs, Some((up_dual, down_dual)))) => {
                    let hidden = value(layer.active)?;
                    for i in 0..*ups {
                        own.push(component(json!({"own": [site(4), i]}), own_width(&h2, up_dual.row(i), 1.0), &[[site(4), i]]));
                        direction.push(component(direction_read(&h2, up_dual.row(i), site(4)), 1.0, &[[site(4), i]]));
                    }
                    for i in 0..*downs {
                        own.push(component(json!({"own": [site(5), i]}), own_width(&hidden, down_dual.row(i), 1.0), &[[site(5), i]]));
                        direction.push(component(direction_read(&hidden, down_dual.row(i), site(5)), 1.0, &[[site(5), i]]));
                    }
                }
                Some((m, _, None)) => {
                    for n in 0..*m {
                        let slices = [[site(4), n], [site(5), n]];
                        own.push(component(json!({"own": [site(4), n]}), own_width(&h2, up.row(n), 1.0), &slices));
                        direction.push(component(direction_read(&h2, up.row(n), site(4)), 1.0, &slices));
                    }
                }
                _ => {}
            }
            for (k, s) in layer_sites.iter().enumerate() {
                let name = format!("h.{l}.{}", crate::library_vpd::EXPORT_NAMES[k]);
                for (suffix, t) in [("U", &s.u), ("V", &s.v)] {
                    let bytes: Vec<u8> = t.iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::write(out.join(format!("{name}.{suffix}.f64")), bytes).map_err(error)?;
                    files.insert(format!("{name}.{suffix}"), json!({"shape": [t.nrows(), t.ncols()]}));
                }
                sites.push(name);
            }
        }
        let record = json!({"config": {"sites": sites, "frame": name, "redundancy": REDUNDANCY, "seed": seed}, "files": files});
        std::fs::write(out.join("export.json"), record.to_string()).map_err(error)?;
        let arms = json!([{"arm": "frame_own", "components": own}, {"arm": "frame_direction", "components": direction}]);
        std::fs::write(out.join("start.json"), arms.to_string()).map_err(error)?;
        summary.push(json!({"frame": name, "components": own.len(), "sites": sites.len(), "fitting_rows": fitting.rows}));
    }
    Ok(json!(summary))
}

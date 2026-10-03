//! Per-input pieces trained through the model's own masked forward (#2951).
//!
//! # The program
//!
//! A *site* is a set of dense operators read as affine terms whose names agree up to a trailing
//! index (one projection's heads). As a whole it is the block matrix `W` from the concatenation of
//! the nodes it reads to the concatenation of the nodes it writes. A library of `C` rank-1 pieces
//! `W ≈ Σ_c u_c v_cᵀ` replaces it in the program, on the read itself:
//!
//! ```text
//! z = Vᵀ x,   z̃ = z ⊙ m,   each written node: U_iᵀ z̃ in place of the site's terms,
//! ```
//!
//! so a mask is a weight intervention and nothing else: centring the read (`Vᵀ(x − μ)` with `W μ`
//! folded into a bias) would move a bias-free site's bias whenever a piece is off. A library's
//! `mean` is ignored here.
//!
//! with `m` a per-input binary mask (one raw slot per site): the masked forward is the program
//! itself, executed natively, and every derivative of it is exact ([`super::derivatives::vjp`]).
//!
//! # Blocks
//!
//! A subcomponent is a *block*: a contiguous run of `k_c` of the library's columns,
//! `U_c` (`d_out × k_c`) and `V_c` (`d_in × k_c`), with one gate per input,
//!
//! ```text
//! z_c = V_cᵀ x,   z̃_c = m_c z_c,   out += U_c z̃_c,
//! ```
//!
//! so a rank-2 rotation or a head's subspace is one name per input rather than `k_c` slices that
//! must co-fire. Rank one is `k_c = 1` (the default, [`Masked::build`]). Every mask in this module's
//! interface is per block (`rows × B`); the program's slot holds it expanded to the columns
//! ([`Masked::family`]), and a gate's derivative is the sum of its columns',
//! `∂KL/∂m_c = Σ_{j ∈ c} (∂KL/∂z̃_j) z_j`. One gate scales the block along a line (all of `U_c Vᵀ_c` at
//! once), not the box of `k_c` independent masks: a robustness test of a block masks its columns
//! together, a weaker claim than one over its columns separately.
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
//!
//! # Claims
//!
//! An explanation declares what its off subcomponents may be, and its error is the KL over what it
//! declares. Under [`Claim::Corner`] they are absent: an input's error is the KL of its masks. Under
//! [`Claim::Box`] each off gate may be anywhere in `[0, 1]`, and an input's error is the KL expected
//! over every off gate drawn uniform and independent, to second order about the masks in each
//! site's written value: with `Z_c` an off block's output (`U_c z_c`) and `S = Σ_off Z_c`, the
//! gates' moments `E m = ½`, `E m² = ⅓` give
//!
//! ```text
//! E KL = KL(masks) + Σ_sites ½ gᵀS + ½ (¼ SᵀF S + 1/12 Σ_off Z_cᵀ F Z_c),
//! ```
//!
//! `g` the KL's gradient at the written value and `F` its Fisher ([`box_excess`]). A fit under the
//! box claim learns subcomponents that explain the input whatever the off ones are set to, not only
//! at exactly one mask.
//!
//! # Behaviours
//!
//! A behaviour scored at some positions only (a chat model's reply to a request, say) is a
//! [`Target`] with its scored rows: the code is over those rows alone. An unscored row adds no KL,
//! no cotangent and no sampled label, and the selection keeps its masks as given, so the rows
//! before a behaviour's positions can run the native map (every piece on) while they still feed
//! it through attention.
//!
//! # Heads
//!
//! A decomposition of a window of a model's blocks leaves the blocks after it frozen. They are a
//! [`Head`]: the program ends at the window's output, and the head maps that to the logits the code
//! scores and pulls their cotangent back. It may run elsewhere (another process, another
//! precision), so the window's operators are the only ones held in float64.
//!
//! # Devices
//!
//! Where the process has an accelerator (`masked_device::device`, under `gam_gpu`'s policy), a
//! masked program without a head is lowered onto it on first use ([`Masked::on_device`]) and the
//! selection runs there: the forward and its KL in float64, so every keep or refuse is the same
//! float64 decision as on the CPU, and the proposals (mask gradients, Fisher diagonals) in the
//! lowered program's proposal arithmetic. Elsewhere everything runs on the CPU.

use super::derivatives::vjp;
use super::device::{product_atb, proposing};
use super::masked_device::{Accelerated, DeviceTarget, State};
use gam_gpu::tensor::{Arithmetic, Device};
use super::operator_program::{
    FamilyInputs, Interface, LabelKind, Node, Operator, OperatorBody, OperatorProgram, Provenance, Slot, SlotValues,
    Trace, remap_node,
};
use super::precision::DeclaredPrecision;
use gam_linalg::faer_ndarray::fast_atb;
use ndarray::{Array1, Array2, Axis, s};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

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

/// The sites of `program`: its dense, fully present operators that are only affine terms between
/// hidden interfaces (neither side token-labelled, so not an embedding or an unembedding), grouped
/// by name up to a trailing index.
pub fn sites(program: &OperatorProgram) -> Vec<Site> {
    let tokens = |interface: &Interface| interface.groups().iter().any(|g| g.label.kind == LabelKind::Token);
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
            if !dense || !only_terms[*op] || tokens(&operator.rows) || tokens(&operator.cols) {
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
    /// Per site, its number of pieces; the pieces themselves live only in its operators
    /// ([`Masked::library`]).
    pieces: Vec<usize>,
    /// Per site, its blocks' ranks in column order (module note, "Blocks"), summing to its pieces.
    ranks: Vec<Vec<usize>>,
    /// Per site: its mask slot and its `z` and `z̃` nodes.
    pub slots: Vec<usize>,
    pub z: Vec<usize>,
    pub masked: Vec<usize>,
    /// Per site: its `V` operators (per read node), its centring bias, its `U` operators (per
    /// written node).
    v_ops: Vec<Vec<usize>>,
    u_ops: Vec<Vec<usize>>,
    read_offsets: Vec<Vec<usize>>,
    write_offsets: Vec<Vec<usize>>,
    /// Whether the replaced site operators were dropped ([`Masked::release_training_state`]).
    released: bool,
    /// Per site and written node: the new node index and its new bias operator.
    written: Vec<Vec<usize>>,
    /// The frozen map after the program, when it ends inside a model (module note, "Heads").
    pub head: Option<Arc<dyn Head>>,
    /// The program on the process's accelerator (module note, "Devices").
    lowered: Mutex<Lowered>,
}

/// A masked program's device twin, lowered on first use.
enum Lowered {
    Untried,
    Host,
    Device(Box<Accelerated>),
}

/// The frozen blocks after a decomposed window (module note, "Heads"). `output` is the program's
/// output on `inputs`; `rows` marks the rows whose logits are scored (the others' are zero).
pub trait Head: Send + Sync {
    /// The logits (rows × classes) of `output`, zero off `rows`.
    fn logits(&self, inputs: &FamilyInputs, output: &Array2<f64>, rows: &[bool]) -> Result<Array2<f64>, String>;
    /// The cotangent at `output` of `cotangent` at the logits (zero off `rows`).
    fn pullback(&self, inputs: &FamilyInputs, output: &Array2<f64>, rows: &[bool], cotangent: &Array2<f64>) -> Result<Array2<f64>, String>;
}

fn fine() -> DeclaredPrecision {
    DeclaredPrecision::new(40).expect("a precision in range")
}

fn dense(name: String, rows: Interface, cols: Interface, values: Array2<f64>) -> Result<Arc<Operator>, String> {
    Ok(Arc::new(Operator::dense(name, rows, cols, values, fine(), Provenance::default()).map_err(|e| e.to_string())?))
}

impl Masked {
    /// Replace `sites` of `model` by `libraries` (one per site), every piece its own block.
    pub fn build(model: &OperatorProgram, sites: Vec<Site>, libraries: Vec<Library>) -> Result<Self, String> {
        let ranks = libraries.iter().map(|l| vec![1; l.v.nrows()]).collect();
        Self::build_blocks(model, sites, libraries, ranks)
    }

    /// Replace `sites` of `model` by `libraries`, site `k`'s columns gated in blocks of `ranks[k]`
    /// (module note, "Blocks").
    pub fn build_blocks(model: &OperatorProgram, sites: Vec<Site>, libraries: Vec<Library>, ranks: Vec<Vec<usize>>) -> Result<Self, String> {
        if ranks.len() != libraries.len() {
            return Err(format!("{} block partitions for {} libraries", ranks.len(), libraries.len()));
        }
        for ((site, library), r) in sites.iter().zip(&libraries).zip(&ranks) {
            if r.contains(&0) || r.iter().sum::<usize>() != library.v.nrows() {
                return Err(format!("{}: blocks {r:?} do not partition its {} pieces", site.name, library.v.nrows()));
            }
        }
        let interfaces = model.interfaces().map_err(|e| e.to_string())?;
        let mut program = model.clone();
        let base_slots = program.declarations.slots.len();
        let (mut slots, mut v_ops, mut u_ops) = (Vec::new(), Vec::new(), Vec::new());
        let (mut read_offsets, mut write_offsets, mut counts) = (Vec::new(), Vec::new(), Vec::new());
        // Each library is dropped once its operators hold it.
        for (k, (site, library)) in sites.iter().zip(libraries).enumerate() {
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
            let mut us = Vec::new();
            for (i, &node) in site.writes.iter().enumerate() {
                let block = library.u.slice(s![.., wo[i]..wo[i + 1]]).t().to_owned();
                program.operators.push(dense(format!("{}·U{i}", site.name), interfaces[node].clone(), coordinates.clone(), block)?);
                us.push(program.operators.len() - 1);
            }
            v_ops.push(vs);
            u_ops.push(us);
            read_offsets.push(ro);
            write_offsets.push(wo);
            counts.push(pieces);
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
                    nodes.push(Node::Affine { terms, bias: None });
                    z_nodes[k] = nodes.len() - 1;
                    nodes.push(Node::Hadamard { left: z_nodes[k], right: mask });
                    masked_nodes[k] = nodes.len() - 1;
                }
            }
            let mut rebuilt = node.clone();
            remap_node(&mut rebuilt, &map, &identity_ops, &identity_bases, &identity_rules);
            if let Node::Affine { terms, .. } = &mut rebuilt {
                for (k, site) in sites.iter().enumerate() {
                    let Some(i) = site.writes.iter().position(|w| *w == index) else { continue };
                    let ops: Vec<usize> = site.terms.iter().filter(|(wi, _, _)| *wi == i).map(|(_, _, op)| *op).collect();
                    terms.retain(|(_, op)| !ops.contains(op));
                    terms.push((masked_nodes[k], u_ops[k][i]));
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
            pieces: counts,
            ranks,
            slots,
            z: z_nodes,
            masked: masked_nodes,
            v_ops,
            u_ops,
            read_offsets,
            write_offsets,
            released: false,
            written,
            head: None,
            lowered: Mutex::new(Lowered::Untried),
        })
    }

    /// Lower the program onto `device` now, its proposals' products in `proposal`, in place of
    /// the process's accelerator.
    pub fn lower_on(&self, device: &Device, proposal: Arithmetic) -> Result<(), String> {
        let accelerated = Accelerated::new(device, self, proposal)?;
        *self.lowered.lock().map_err(|_| "device: a poisoned lowering".to_string())? = Lowered::Device(Box::new(accelerated));
        Ok(())
    }

    /// `run` on the program's device twin (module note, "Devices"), its operators first brought up
    /// to date with the program's; `None` when the program runs on the CPU. Calls hold the twin in
    /// turn, so `run` must not call this again.
    pub fn on_device<T>(&self, run: impl FnOnce(&Accelerated) -> Result<T, String>) -> Result<Option<T>, String> {
        if self.head.is_some() {
            return Ok(None);
        }
        let mut lowered = self.lowered.lock().map_err(|_| "device: a poisoned lowering".to_string())?;
        if matches!(*lowered, Lowered::Untried) {
            *lowered = match super::masked_device::device()? {
                None => Lowered::Host,
                Some(device) => match Accelerated::new(&device, self, Arithmetic::F32) {
                    Ok(accelerated) => {
                        log::info!("masked program lowered onto {}", device.name());
                        Lowered::Device(Box::new(accelerated))
                    }
                    Err(e) if gam_gpu::global_policy() == gam_gpu::GpuPolicy::Required => return Err(e),
                    Err(e) => {
                        log::warn!("masked program stays on the CPU: {e}");
                        Lowered::Host
                    }
                },
            };
        }
        match &mut *lowered {
            Lowered::Device(accelerated) => {
                accelerated.refresh(self)?;
                run(accelerated).map(Some)
            }
            Lowered::Untried | Lowered::Host => Ok(None),
        }
    }

    /// [`Masked::on_device`] where the program was found lowered.
    fn on_lowered<T>(&self, run: impl FnOnce(&Accelerated) -> Result<T, String>) -> Result<T, String> {
        self.on_device(run)?.ok_or_else(|| "device: the masked program left its device".to_string())
    }

    /// Site `k`'s number of pieces.
    pub fn pieces(&self, k: usize) -> usize {
        self.pieces[k]
    }

    /// Every site's number of pieces.
    pub fn all_pieces(&self) -> Vec<usize> {
        self.pieces.clone()
    }

    /// Site `k`'s blocks' ranks, in column order.
    pub fn ranks(&self, k: usize) -> &[usize] {
        &self.ranks[k]
    }

    /// Site `k`'s number of blocks (its masks' width).
    pub fn blocks(&self, k: usize) -> usize {
        self.ranks[k].len()
    }

    /// Every site's number of blocks.
    pub fn all_blocks(&self) -> Vec<usize> {
        self.ranks.iter().map(Vec::len).collect()
    }

    /// Whether every block of site `k` is a single piece (its masks are per column).
    pub fn is_rank_one(&self, k: usize) -> bool {
        self.ranks[k].len() == self.pieces[k]
    }

    /// Site `k`'s per-block `mask` (rows × B) on its columns (rows × C).
    pub fn expand(&self, k: usize, mask: &Array2<f64>) -> Array2<f64> {
        if self.is_rank_one(k) {
            return mask.clone();
        }
        let column_block: Vec<usize> = self.ranks[k].iter().enumerate().flat_map(|(b, r)| std::iter::repeat_n(b, *r)).collect();
        Array2::from_shape_fn((mask.nrows(), column_block.len()), |(t, c)| mask[[t, column_block[c]]])
    }

    /// Site `k`'s `per_column` values (rows × C) summed within each block (rows × B).
    pub fn to_blocks(&self, k: usize, per_column: &Array2<f64>) -> Array2<f64> {
        if self.is_rank_one(k) {
            return per_column.clone();
        }
        let mut out = Array2::<f64>::zeros((per_column.nrows(), self.ranks[k].len()));
        let mut start = 0;
        for (b, r) in self.ranks[k].iter().enumerate() {
            out.column_mut(b).assign(&per_column.slice(s![.., start..start + r]).sum_axis(Axis(1)));
            start += r;
        }
        out
    }


    /// Site `k`'s library as the program holds it (its operators are the only copy).
    pub fn library(&self, k: usize) -> Result<Library, String> {
        let v_blocks: Vec<_> = self.v_ops[k].iter().map(|&op| self.program.operators[op].matrix_cow()).collect();
        let v = ndarray::concatenate(Axis(1), &v_blocks.iter().map(|b| b.view()).collect::<Vec<_>>()).map_err(|e| e.to_string())?;
        let u_blocks: Vec<_> = self.u_ops[k].iter().map(|&op| self.program.operators[op].matrix_cow()).collect();
        let u = ndarray::concatenate(Axis(1), &u_blocks.iter().map(|b| b.t()).collect::<Vec<_>>()).map_err(|e| e.to_string())?;
        let d_in = v.ncols();
        Ok(Library { v, u, mean: Array1::zeros(d_in) })
    }

    /// Site `k`'s own map `W` (written × read), from the model's operators, which the program
    /// keeps until [`Masked::release_training_state`].
    pub fn w(&self, k: usize) -> Result<Array2<f64>, String> {
        if self.released {
            return Err(format!("{}: its map was released", self.sites[k].name));
        }
        matrix(&self.program, &self.sites[k])
    }

    /// Restrict site `k`'s pieces to the reads they act on: `v_c ← P v_c`, `P` the projector on the
    /// leading eigendirections of the reads' second moment `E[x xᵀ]` (from the reads' mean
    /// `data_mean` and covariance) that hold all but `left_out` of it. What a piece
    /// reads off that span is unidentified by the data, costs library bits, and acts unpredictably
    /// on an edited input. Returns the dimension kept.
    pub fn project_reads(&mut self, k: usize, data_mean: &Array1<f64>, covariance: &Array2<f64>, left_out: f64) -> Result<usize, String> {
        let shift = data_mean;
        let mut moment = covariance + &shift.view().insert_axis(Axis(1)).dot(&shift.view().insert_axis(Axis(0)));
        let n = moment.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                let v = 0.5 * (moment[[i, j]] + moment[[j, i]]);
                moment[[i, j]] = v;
                moment[[j, i]] = v;
            }
        }
        let decomposed = super::dense::eigh(moment.view(), gam_linalg::roundoff::SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
        let total: f64 = decomposed.values.iter().map(|l| l.max(0.0)).sum();
        // Eigenvalues ascend: keep from the top until all but `left_out` is held.
        let (mut held, mut kept) = (0.0, 0);
        for l in decomposed.values.iter().rev() {
            if held >= (1.0 - left_out) * total {
                break;
            }
            held += l.max(0.0);
            kept += 1;
        }
        let basis = decomposed.vectors.slice(s![.., n - kept..]);
        let projector = basis.dot(&basis.t());
        let mut library = self.library(k)?;
        library.v = library.v.dot(&projector);
        self.set_library(k, library)?;
        Ok(kept)
    }

    /// Drop what only the pieces' training reads, for a selection-only fit: the replaced site
    /// operators (left as rank-1 zero placeholders, so no index moves). The pieces themselves are
    /// the program's operators. Afterwards [`step_pieces`], [`dropped_atoms`],
    /// [`Masked::set_library`] and [`Masked::w`] refuse.
    pub fn release_training_state(&mut self) -> Result<(), String> {
        self.released = true;
        for site in &self.sites {
            for &(_, _, op) in &site.terms {
                let old = &self.program.operators[op];
                let (rows, cols) = (old.rows.clone(), old.cols.clone());
                let placeholder = Operator::low_rank(
                    old.name.clone(),
                    rows.clone(),
                    cols.clone(),
                    Array2::zeros((rows.width(), 1)),
                    Array2::zeros((1, cols.width())),
                    fine(),
                    Provenance::default(),
                )
                .map_err(|e| e.to_string())?;
                self.program.operators[op] = Arc::new(placeholder);
            }
        }
        Ok(())
    }

    fn released(&self) -> bool {
        self.released
    }

    /// Set site `k`'s library (same number of pieces) into the program.
    pub fn set_library(&mut self, k: usize, library: Library) -> Result<(), String> {
        if self.released() {
            return Err("set_library on a masked program whose training state was released".to_string());
        }
        if library.v.nrows() != self.pieces[k] || library.u.nrows() != self.pieces[k] {
            return Err(format!("{}: {} pieces set into a site of {}", self.sites[k].name, library.v.nrows(), self.pieces[k]));
        }
        let (ro, wo) = (&self.read_offsets[k], &self.write_offsets[k]);
        for (j, &op) in self.v_ops[k].iter().enumerate() {
            let old = &self.program.operators[op];
            let block = library.v.slice(s![.., ro[j]..ro[j + 1]]).to_owned();
            self.program.operators[op] = dense(old.name.clone(), old.rows.clone(), old.cols.clone(), block)?;
        }
        for (i, &op) in self.u_ops[k].iter().enumerate() {
            let old = &self.program.operators[op];
            let block = library.u.slice(s![.., wo[i]..wo[i + 1]]).t().to_owned();
            self.program.operators[op] = dense(old.name.clone(), old.rows.clone(), old.cols.clone(), block)?;
        }
        Ok(())
    }

    /// `base` with the masks (one `rows × B` per site) expanded to their columns in the sites' slots.
    pub fn family(&self, base: &FamilyInputs, masks: &[Array2<f64>]) -> FamilyInputs {
        let mut family = base.clone();
        family.slots.extend(masks.iter().enumerate().map(|(k, m)| SlotValues::Raw(self.expand(k, m))));
        family
    }
}

/// What a masked forward is scored against (module note, "Behaviours"): the model's own logits
/// and, for a behaviour, which rows count (`None`: every row).
#[derive(Clone, Debug)]
pub struct Target {
    pub logits: Array2<f64>,
    pub scored: Option<Vec<bool>>,
}

impl Target {
    /// Every row scored.
    pub fn every_row(logits: Array2<f64>) -> Self {
        Self { logits, scored: None }
    }

    /// Whether row `r` is scored.
    pub fn scores(&self, r: usize) -> bool {
        self.scored.as_ref().is_none_or(|s| s[r])
    }

    /// The number of scored rows.
    pub fn scored_rows(&self) -> usize {
        (0..self.logits.nrows()).filter(|r| self.scores(*r)).count()
    }
}

/// `KL(p ‖ q)` per row between the target's logits and `logits` (rows × classes), and the
/// cotangent of the total in the logits, `q − p` per row; an unscored row has zero of both.
pub fn kl(target: &Target, logits: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    use rayon::prelude::*;
    let mut cotangent = Array2::<f64>::zeros(logits.dim());
    // Rows are independent: one per worker.
    let values: Vec<f64> = cotangent
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .map(|(r, row)| {
            if !target.scores(r) {
                return 0.0;
            }
            kl_row(target.logits.row(r), logits.row(r), Some(row))
        })
        .collect();
    (Array1::from(values), cotangent)
}

/// KL values only: no vocabulary-sized cotangent or per-row temporary arrays.
pub fn kl_score_only(target: &Target, logits: &Array2<f64>) -> Array1<f64> {
    use rayon::prelude::*;
    Array1::from((0..logits.nrows()).into_par_iter().map(|r| {
        if target.scores(r) { kl_row(target.logits.row(r), logits.row(r), None) } else { 0.0 }
    }).collect::<Vec<_>>())
}

fn kl_row(teacher: ndarray::ArrayView1<'_, f64>, logits: ndarray::ArrayView1<'_, f64>, mut gradient: Option<ndarray::ArrayViewMut1<'_, f64>>) -> f64 {
    let stats = |z: ndarray::ArrayView1<'_, f64>| {
        let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (m, z.iter().map(|v| (v - m).exp()).sum::<f64>())
    };
    let (mt, st) = stats(teacher);
    let (mz, sz) = stats(logits);
    let (lt, lz) = (st.ln(), sz.ln());
    let mut total = 0.0;
    for c in 0..logits.len() {
        let p = (teacher[c] - mt).exp() / st;
        if p > 0.0 {
            total += p * (((teacher[c] - mt) - lt) - ((logits[c] - mz) - lz));
        }
        if let Some(g) = &mut gradient {
            g[c] = (logits[c] - mz).exp() / sz - p;
        }
    }
    total
}

fn softmax(z: ndarray::ArrayView1<'_, f64>) -> Array1<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Array1<f64> = z.mapv(|v| (v - m).exp());
    let total = e.sum();
    e / total
}

/// The rows a target scores, as a mask over `rows` rows.
fn scored_mask(target: &Target, rows: usize) -> Vec<bool> {
    (0..rows).map(|r| target.scores(r)).collect()
}

/// The logits the code scores on a trace: the program's output, or its head's logits of it.
pub fn logits<'a>(masked: &Masked, family: &FamilyInputs, trace: &'a Trace, target: &Target) -> Result<std::borrow::Cow<'a, Array2<f64>>, String> {
    let output = &trace.values[masked.program.output];
    Ok(match &masked.head {
        None => std::borrow::Cow::Borrowed(output),
        Some(head) => std::borrow::Cow::Owned(head.logits(family, output, &scored_mask(target, output.nrows()))?),
    })
}

/// A cotangent at the logits as a cotangent at the program's output (through the head, if any).
pub fn to_output(masked: &Masked, family: &FamilyInputs, trace: &Trace, target: &Target, cotangent: Array2<f64>) -> Result<Array2<f64>, String> {
    match &masked.head {
        None => Ok(cotangent),
        Some(head) => {
            let output = &trace.values[masked.program.output];
            head.pullback(family, output, &scored_mask(target, output.nrows()), &cotangent)
        }
    }
}

/// One masked forward: per-input KL against `target`, the trace, and the KL's cotangent at the
/// program's output.
pub fn forward(masked: &Masked, family: &FamilyInputs, target: &Target) -> Result<(Array1<f64>, Trace, Array2<f64>), String> {
    let trace = masked.program.execute(family, false).map_err(|e| e.to_string())?;
    let (values, cotangent) = match &masked.head {
        None => kl(target, &trace.values[masked.program.output]),
        Some(_) => {
            let scored = logits(masked, family, &trace, target)?;
            kl(target, &scored)
        }
    };
    let cotangent = to_output(masked, family, &trace, target, cotangent)?;
    Ok((values, trace, cotangent))
}

/// Candidate forward and loss without a cotangent or downstream-head pullback.
fn scored_forward(masked: &Masked, family: &FamilyInputs, target: &Target) -> Result<(Array1<f64>, Trace), String> {
    let trace = masked.program.execute(family, false).map_err(|e| e.to_string())?;
    let values = kl_score_only(target, &*logits(masked, family, &trace, target)?);
    Ok((values, trace))
}

/// Evaluate an actual candidate for acceptance without calculating gradients: the float64 KL per
/// input, on the program's device twin when it has one (module note, "Devices").
pub fn score_only(masked: &Masked, family: &FamilyInputs, target: &Target) -> Result<Array1<f64>, String> {
    if let Some(values) = masked.on_device(|accelerated| accelerated.score_only(family, &accelerated.target(target)?))? {
        return Ok(values);
    }
    scored_forward(masked, family, target).map(|(values, _)| values)
}

/// The float64 KL per input and the logits it scores, on the program's device twin when it has
/// one (module note, "Devices").
pub fn kl_and_logits(masked: &Masked, family: &FamilyInputs, target: &Target) -> Result<(Array1<f64>, Array2<f64>), String> {
    let lowered = masked.on_device(|accelerated| {
        let state = accelerated.forward(family, &accelerated.target(target)?)?;
        let logits = accelerated.program().logits(&state.trace, 0, state.trace.rows)?;
        Ok((state.kl, logits))
    })?;
    if let Some(out) = lowered {
        return Ok(out);
    }
    let (values, trace) = scored_forward(masked, family, target)?;
    let logits = logits(masked, family, &trace, target)?.into_owned();
    Ok((values, logits))
}

/// The box claim's excess per input ([`box_excess`]) at `masks`, on the program's device twin
/// when it has one (module note, "Devices").
pub fn box_excess_at(masked: &Masked, base: &FamilyInputs, target: &Target, masks: &[Array2<f64>], fishers: &[Array2<f64>]) -> Result<Array1<f64>, String> {
    let family = masked.family(base, masks);
    if (0..masked.sites.len()).all(|k| masked.is_rank_one(k))
        && let Some(excess) = masked.on_device(|accelerated| {
            let state = accelerated.forward(&family, &accelerated.target(target)?)?;
            Ok(accelerated.box_excess(masked, &state, masks, fishers, false)?.0)
        })?
    {
        return Ok(excess);
    }
    let (_, trace, cotangent) = forward(masked, &family, target)?;
    Ok(box_excess(masked, &family, &trace, masks, cotangent, fishers, false)?.0)
}

/// Per site: `∂KL/∂m` (rows × B) and the gradients in `V` (C × d_in) and `U` (C × d_out), from one
/// reverse pass of `cotangent`. They only propose (ranking flips, steering steps), so their dense
/// products may run in f32 on the GPU ([`super::device`]); every acceptance is a float64 forward.
pub fn gradients(
    masked: &Masked,
    family: &FamilyInputs,
    trace: &Trace,
    masks: &[Array2<f64>],
    cotangent: Array2<f64>,
) -> Result<Vec<(Array2<f64>, Array2<f64>, Array2<f64>)>, String> {
    proposing(|| gradients_proposed(masked, family, trace, masks, cotangent))
}

fn gradients_proposed(
    masked: &Masked,
    family: &FamilyInputs,
    trace: &Trace,
    masks: &[Array2<f64>],
    cotangent: Array2<f64>,
) -> Result<Vec<(Array2<f64>, Array2<f64>, Array2<f64>)>, String> {
    let mut keep = masked.masked.clone();
    keep.extend(masked.written.iter().flatten().copied());
    let back = super::derivatives::vjp_from(&masked.program, family, trace, masked.program.output, cotangent, Some(&keep)).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for (k, site) in masked.sites.iter().enumerate() {
        let pieces = masked.pieces[k];
        let rows = trace.values[masked.z[k]].nrows();
        let zero = || Array2::<f64>::zeros((rows, pieces));
        let cot_masked = back[masked.masked[k]].clone().unwrap_or_else(zero);
        let z = &trace.values[masked.z[k]];
        let mask_gradient = masked.to_blocks(k, &(&cot_masked * z));
        let cot_z = &cot_masked * &masked.expand(k, &masks[k]);
        let reads = read_values(trace, site)?;
        let v_gradient = product_atb(&cot_z, &reads).map_err(|e| e.to_string())?;
        let zm = &trace.values[masked.masked[k]];
        let written: Vec<Array2<f64>> = masked.written[k]
            .iter()
            .map(|n| back[*n].clone().unwrap_or_else(|| Array2::zeros(trace.values[*n].dim())))
            .collect();
        let views: Vec<_> = written.iter().map(|w| w.view()).collect();
        let cot_written = ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?;
        let u_gradient = product_atb(zm, &cot_written).map_err(|e| e.to_string())?;
        out.push((mask_gradient, v_gradient, u_gradient));
    }
    Ok(out)
}

/// Per site `∂/∂m` (rows × B) of whatever `cotangent` is the gradient of at the output, from one
/// reverse pass: what the selection ranks flips by (a proposal, as [`gradients`]), without the
/// pieces' own gradients.
pub fn mask_gradients(masked: &Masked, family: &FamilyInputs, trace: &Trace, cotangent: Array2<f64>) -> Result<Vec<Array2<f64>>, String> {
    let back = proposing(|| super::derivatives::vjp_from(&masked.program, family, trace, masked.program.output, cotangent, Some(&masked.masked))).map_err(|e| e.to_string())?;
    Ok((0..masked.sites.len())
        .map(|k| {
            let z = &trace.values[masked.z[k]];
            back[masked.masked[k]].as_ref().map_or_else(|| Array2::zeros((z.nrows(), masked.blocks(k))), |c| masked.to_blocks(k, &(c * z)))
        })
        .collect())
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

/// The cotangent of `−log q_y` at the logits, `q − e_y`, with each row's label `y` drawn from the
/// row's own distribution `q`: its outer product is an unbiased sample of the output Fisher. Rows
/// outside `scored` (every row when absent, as in [`Target::scores`]) stay zero.
fn sampled_cotangent(logits: &Array2<f64>, rng: &mut XorShift, scored: Option<&[bool]>) -> Array2<f64> {
    let mut cotangent = Array2::<f64>::zeros(logits.dim());
    for r in 0..logits.nrows() {
        if scored.is_some_and(|s| !s[r]) {
            continue;
        }
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
    cotangent
}

/// One sampled-label cotangent of the target's scored rows (as [`fisher`] draws them), from `seed`.
pub fn sampled_label_cotangent(logits: &Array2<f64>, target: &Target, seed: u64) -> Array2<f64> {
    sampled_cotangent(logits, &mut XorShift(seed | 1), target.scored.as_deref())
}

/// What a site's Fisher-SVD library is built from (`pieces::fisher_svd`), measured on the native
/// program over `batches` of inputs: the map `W`, the reads' mean and second moment `E[x xᵀ]`, and
/// the output Fisher `E[g gᵀ]` of the written value, `g` the gradient of `−log q_y` with `y` drawn
/// from the program's own output, `samples` draws per input. Any model's sites, no external dump.
pub fn site_statistics(
    program: &OperatorProgram,
    sites: &[Site],
    batches: impl IntoIterator<Item = FamilyInputs>,
    samples: usize,
    seed: u64,
) -> Result<Vec<super::pieces::Site>, String> {
    let mut rng = XorShift(seed | 1);
    let mut stats: Vec<(Array1<f64>, Array2<f64>, Array2<f64>)> = Vec::new();
    let maps: Vec<Array2<f64>> = sites.iter().map(|site| matrix(program, site)).collect::<Result<_, _>>()?;
    for w in &maps {
        let (d_out, d_in) = w.dim();
        stats.push((Array1::zeros(d_in), Array2::zeros((d_in, d_in)), Array2::zeros((d_out, d_out))));
    }
    let (mut rows, mut draws) = (0.0, 0.0);
    for inputs in batches {
        let trace = program.execute(&inputs, false).map_err(|e| e.to_string())?;
        for (site, (sum, outer, _)) in sites.iter().zip(stats.iter_mut()) {
            let x = read_values(&trace, site)?;
            *sum += &x.sum_axis(Axis(0));
            *outer += &fast_atb(&x, &x);
        }
        rows += inputs.rows as f64;
        for _ in 0..samples {
            let cotangent = sampled_cotangent(&trace.values[program.output], &mut rng, None);
            let back = proposing(|| vjp(program, &inputs, &trace, cotangent)).map_err(|e| e.to_string())?;
            for (site, (_, _, fisher)) in sites.iter().zip(stats.iter_mut()) {
                let written: Vec<Array2<f64>> =
                    site.writes.iter().map(|n| back[*n].clone().unwrap_or_else(|| Array2::zeros(trace.values[*n].dim()))).collect();
                let views: Vec<_> = written.iter().map(|w| w.view()).collect();
                let g = ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?;
                *fisher += &proposing(|| product_atb(&g, &g)).map_err(|e| e.to_string())?;
            }
            draws += inputs.rows as f64;
        }
    }
    if rows == 0.0 || draws == 0.0 {
        return Err("site statistics need inputs and samples".to_string());
    }
    Ok(maps
        .into_iter()
        .zip(stats)
        .map(|(w, (sum, outer, fisher))| super::pieces::Site { w, second_moment: outer / rows, mean: sum / rows, fisher: fisher / draws })
        .collect())
}

/// Per-input samples of `sites` for [`super::pieces::attribution_dictionary`], over `batches` of
/// inputs to the native program: every input's reads and the gradient at the written value of
/// `−log q_y`, one label `y` per input drawn from the program's own output (single precision).
pub fn site_attributions(
    program: &OperatorProgram,
    sites: &[Site],
    batches: impl IntoIterator<Item = FamilyInputs>,
    seed: u64,
) -> Result<Vec<super::pieces::Attributions>, String> {
    let mut rng = XorShift(seed | 1);
    let mut reads: Vec<Vec<Array2<f32>>> = vec![Vec::new(); sites.len()];
    let mut gradients: Vec<Vec<Array2<f32>>> = vec![Vec::new(); sites.len()];
    for inputs in batches {
        let trace = program.execute(&inputs, false).map_err(|e| e.to_string())?;
        let cotangent = sampled_cotangent(&trace.values[program.output], &mut rng, None);
        let back = proposing(|| vjp(program, &inputs, &trace, cotangent)).map_err(|e| e.to_string())?;
        for (k, site) in sites.iter().enumerate() {
            reads[k].push(read_values(&trace, site)?.mapv(|v| v as f32));
            let written: Vec<Array2<f64>> =
                site.writes.iter().map(|n| back[*n].clone().unwrap_or_else(|| Array2::zeros(trace.values[*n].dim()))).collect();
            let views: Vec<_> = written.iter().map(|w| w.view()).collect();
            gradients[k].push(ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?.mapv(|v| v as f32));
        }
    }
    reads
        .into_iter()
        .zip(gradients)
        .map(|(x, g)| {
            let stack = |parts: Vec<Array2<f32>>| -> Result<Array2<f32>, String> {
                let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
                ndarray::concatenate(Axis(0), &views).map_err(|e| e.to_string())
            };
            Ok(super::pieces::Attributions { reads: stack(x)?, gradients: stack(g)? })
        })
        .collect()
}

/// The Fisher diagonal of every mask entry and, when `written`, the Fisher of every written node,
/// from `samples` sampled-label reverse passes on the target's scored rows: per site
/// `(h: rows × B, F: d_out × d_out)`. The selection needs only `h`; `F` (the pieces'
/// preconditioner) can be far larger than the site.
pub fn fisher(
    masked: &Masked,
    family: &FamilyInputs,
    trace: &Trace,
    target: &Target,
    samples: usize,
    seed: u64,
    written: bool,
) -> Result<Vec<(Array2<f64>, Option<Array2<f64>>)>, String> {
    if samples == 0 { return Err("Fisher needs at least one sample".to_string()); }
    let logits = logits(masked, family, trace, target)?;
    let mut keep = masked.masked.clone();
    if written { keep.extend(masked.written.iter().flatten().copied()); }
    let head = CachedSampledHead::new(masked, &logits, target, &keep)?;
    let mut rng = XorShift(seed | 1);
    let mut out: Vec<(Array2<f64>, Option<Array2<f64>>)> = masked
        .sites
        .iter()
        .enumerate()
        .map(|(k, _)| {
            let d_out: usize = masked.written[k].iter().map(|n| trace.values[*n].ncols()).sum();
            (Array2::zeros((trace.values[masked.z[k]].nrows(), masked.blocks(k))), written.then(|| Array2::zeros((d_out, d_out))))
        })
        .collect();
    for _ in 0..samples {
        let (node, cotangent) = match &head {
            Some(head) => (head.hidden, head.sample(masked, target, &mut rng)),
            None => {
                let cotangent = sampled_cotangent(&logits, &mut rng, target.scored.as_deref());
                (masked.program.output, to_output(masked, family, trace, target, cotangent)?)
            }
        };
        let back = proposing(|| super::derivatives::vjp_from(&masked.program, family, trace, node, cotangent, Some(&keep))).map_err(|e| e.to_string())?;
        for (k, (h, f)) in out.iter_mut().enumerate() {
            if let Some(c) = &back[masked.masked[k]] {
                let g = masked.to_blocks(k, &(c * &trace.values[masked.z[k]]));
                *h += &(&g * &g);
            }
            let Some(f) = f else { continue };
            let written: Vec<Array2<f64>> = masked.written[k]
                .iter()
                .map(|n| back[*n].clone().unwrap_or_else(|| Array2::zeros(trace.values[*n].dim())))
                .collect();
            let views: Vec<_> = written.iter().map(|w| w.view()).collect();
            let cot = ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?;
            *f += &proposing(|| product_atb(&cot, &cot)).map_err(|e| e.to_string())?;
        }
    }
    let rows = target.scored_rows().max(1) as f64;
    for (h, f) in out.iter_mut() {
        *h /= samples as f64;
        if let Some(f) = f {
            *f /= samples as f64 * rows;
        }
    }
    Ok(out)
}

/// Sufficient sampling data for one fixed forward's linear output head. Built anew for each
/// Fisher call, so edits to tied or ordinary output weights cannot leave a stale cache.
struct CachedSampledHead {
    hidden: usize,
    operator: usize,
    transposed: bool,
    probabilities: Array2<f64>,
    mean: Array2<f64>,
}

impl CachedSampledHead {
    fn new(masked: &Masked, logits: &Array2<f64>, target: &Target, keep: &[usize]) -> Result<Option<Self>, String> {
        if masked.head.is_some() { return Ok(None); }
        let program = &masked.program;
        let output = match &program.nodes[program.output] {
            Node::Readout { input, basis } => {
                if !matches!(program.bases[*basis], super::operator_program::Basis::Indicator { .. }) { return Ok(None); }
                *input
            }
            _ => program.output,
        };
        let (hidden, operator, transposed) = match &program.nodes[output] {
            Node::Affine { terms, .. } if terms.len() == 1 => (terms[0].0, terms[0].1, false),
            Node::Transposed { input, operator } => (*input, *operator, true),
            _ => return Ok(None),
        };
        if keep.iter().any(|node| *node > hidden) || !matches!(program.operators[operator].body, OperatorBody::Dense { .. }) {
            return Ok(None);
        }
        let mut probabilities = Array2::zeros(logits.dim());
        for r in 0..logits.nrows() {
            if target.scores(r) { probabilities.row_mut(r).assign(&softmax(logits.row(r))); }
        }
        let layout = if transposed { gam_gpu::banded::Layout::Transposed } else { gam_gpu::banded::Layout::AsStored };
        let mean = super::device::product(&program.operators[operator], &probabilities, layout).map_err(|e| e.to_string())?;
        Ok(Some(Self { hidden, operator, transposed, probabilities, mean }))
    }

    fn sample(&self, masked: &Masked, target: &Target, rng: &mut XorShift) -> Array2<f64> {
        let weights = masked.program.operators[self.operator].matrix_cow();
        let mut seed = self.mean.clone();
        for r in 0..seed.nrows() {
            if !target.scores(r) { continue; }
            let mut pick = rng.next();
            let mut label = self.probabilities.ncols() - 1;
            for (c, p) in self.probabilities.row(r).iter().enumerate() {
                if pick < *p { label = c; break; }
                pick -= p;
            }
            if self.transposed { seed.row_mut(r).scaled_add(-1.0, &weights.column(label)); }
            else { seed.row_mut(r).scaled_add(-1.0, &weights.row(label)); }
        }
        seed
    }
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
///
/// [`Coder::ran`] instead charges each input for the weights that ran on it: every block on pays
/// its own description bits (`costs`), with no context and no set credit, so nothing is free for
/// being reused ([`super::blocks`]).
pub struct Coder {
    /// Listing bits of a piece listed new, or under [`Coder::ran`] a block's description bits.
    pub costs: Vec<Array1<f64>>,
    /// Per piece, the probability it stays on from one input to the next.
    pub stay: Vec<Array1<f64>>,
    /// Per input, the previous input of its sequence.
    pub previous: Vec<Option<usize>>,
    /// Whether the sets are listed (the context code) or the blocks that ran are described.
    listed: bool,
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

    /// The counts of a library with `added[k]` new pieces appended to site `k`, never seen yet.
    pub fn extended(&self, added: &[usize]) -> Self {
        let grow = |counts: &[Array1<f64>]| -> Vec<Array1<f64>> {
            counts.iter().zip(added).map(|(c, a)| c.iter().copied().chain(std::iter::repeat_n(0.0, *a)).collect()).collect()
        };
        Self { stayed: grow(&self.stayed), was_on: grow(&self.was_on), new: grow(&self.new) }
    }

    /// The coder these counts give, for inputs whose previous inputs are `previous`.
    pub fn coder(&self, previous: Vec<Option<usize>>) -> Coder {
        let stay = self.stayed.iter().zip(&self.was_on).map(|(s, w)| (s + 0.5) / &(w + 1.0)).collect();
        Coder { costs: costs_from_counts(&self.new), stay, previous, listed: true }
    }
}

impl Coder {
    /// The code charging each of `rows` inputs `costs[k][c]` for every block `c` of site `k` on
    /// there: the description of the weights that ran on it.
    pub fn ran(costs: Vec<Array1<f64>>, rows: usize) -> Self {
        Self { costs, stay: Vec::new(), previous: vec![None; rows], listed: false }
    }

    /// Each input's explanation bits.
    pub fn bits(&self, masks: &[Array2<f64>]) -> Array1<f64> {
        let rows = masks.first().map_or(0, |m| m.nrows());
        let mut out = Array1::<f64>::zeros(rows);
        if !self.listed {
            for (m, costs) in masks.iter().zip(&self.costs) {
                for r in 0..rows {
                    out[r] += m.row(r).iter().zip(costs.iter()).filter(|(x, _)| **x > 0.0).map(|(_, c)| c).sum::<f64>();
                }
            }
            return out;
        }
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
        if !self.listed {
            return if on { -self.costs[k][c] } else { self.costs[k][c] };
        }
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

/// A selection round's masked forward (module note, "Devices"): the CPU's trace with the KL's
/// cotangent at the output (`None` until a gradient needs it), or the device's state.
enum Selected {
    Host(Trace, Option<Array2<f64>>),
    Device(State),
}

impl Selected {
    /// The masked forward and its KL, deferring the head cotangent on score-only trials.
    fn forward(masked: &Masked, family: &FamilyInputs, target: &Target, on_device: Option<&DeviceTarget>, cotangent: bool) -> Result<(Array1<f64>, Self), String> {
        if let Some(on_device) = on_device {
            let state = masked.on_lowered(|accelerated| {
                if cotangent { accelerated.forward(family, on_device) }
                else { accelerated.score_state(family, on_device) }
            })?;
            return Ok((state.kl.clone(), Self::Device(state)));
        }
        if cotangent {
            let (values, trace, cotangent) = forward(masked, family, target)?;
            return Ok((values, Self::Host(trace, Some(cotangent))));
        }
        let (values, trace) = scored_forward(masked, family, target)?;
        Ok((values, Self::Host(trace, None)))
    }

    /// [`mask_gradients`] of the KL.
    fn mask_gradients(&mut self, masked: &Masked, family: &FamilyInputs, target: &Target, on_device: Option<&DeviceTarget>) -> Result<Vec<Array2<f64>>, String> {
        match self {
            Self::Device(state) => masked.on_lowered(|accelerated| {
                accelerated.prepare_gradient(state, on_device.ok_or("device: missing selection target")?)?;
                accelerated.mask_gradients(masked, state)
            }),
            Self::Host(trace, cotangent) => {
                let cotangent = match cotangent.take() {
                    Some(c) => c,
                    None => to_output(masked, family, trace, target, kl(target, &*logits(masked, family, trace, target)?).1)?,
                };
                mask_gradients(masked, family, trace, cotangent)
            }
        }
    }

    /// The Fisher diagonals of [`fisher`].
    fn fisher(
        &self,
        masked: &Masked,
        family: &FamilyInputs,
        target: &Target,
        on_device: Option<&DeviceTarget>,
        samples: usize,
        seed: u64,
    ) -> Result<Vec<(Array2<f64>, Option<Array2<f64>>)>, String> {
        match (self, on_device) {
            (Self::Device(state), Some(on_device)) => masked.on_lowered(|accelerated| accelerated.fisher(masked, state, on_device, samples, seed, false)),
            (Self::Host(trace, _), _) => fisher(masked, family, trace, target, samples, seed, false),
            (Self::Device(_), None) => Err("device: a device state without its target".to_string()),
        }
    }
}

/// Selection (module note): rounds of predicted flips, each sequence keeping its inputs' flips only
/// when its exact code falls, until no sequence changes. Sequences are independent, but the inputs
/// of one are not (attention carries an earlier input's masks into every later one), so a proposal
/// is accepted or refused per sequence, on exactly the masks it commits. The second-order
/// prediction is additive over flips, but flips interact, so a proposal of `k` flips over-promises
/// by a term that grows with `k`. Each input carries its own measured interaction `α`, fitted from
/// its last evaluated proposal as `(predicted − actual saving) / k²`, and proposes the `k` best flips
/// maximising `S(k) − α k²` (`S` the predicted saving of its best `k`); a refused sequence's inputs
/// propose at most half as many next round, and a sequence refused at one flip per input is done.
/// Returns the masks and the final exact KL.
pub fn select(
    masked: &Masked,
    base: &FamilyInputs,
    target: &Target,
    masks: Vec<Array2<f64>>,
    coder: &Coder,
    observations: f64,
    samples: usize,
) -> Result<(Vec<Array2<f64>>, Array1<f64>), String> {
    select_observed(masked, base, target, masks, coder, observations, samples, None, &mut |_: &Round<'_>| Ok(()))
}

/// [`select`] under the box claim (module note, "Claims"): every input's error is its masks' KL
/// plus [`box_excess`] in the written Fishers `fishers`, so the sets it keeps explain the input
/// whatever the off subcomponents are set to in `[0, 1]`. Returns the masks and their own KL.
#[allow(clippy::too_many_arguments)]
pub fn select_boxed(
    masked: &Masked,
    base: &FamilyInputs,
    target: &Target,
    masks: Vec<Array2<f64>>,
    coder: &Coder,
    observations: f64,
    samples: usize,
    fishers: &[Array2<f64>],
) -> Result<(Vec<Array2<f64>>, Array1<f64>), String> {
    select_observed(masked, base, target, masks, coder, observations, samples, Some(fishers), &mut |_: &Round<'_>| Ok(()))
}

/// One selection round's keep/refuse decisions as [`select_observed`] shows them: the masks before
/// and with the proposal, each input's code under both, each input's sequence and its flips. A
/// sequence keeps its proposal when its inputs' codes sum lower with it.
pub struct Round<'a> {
    pub current: &'a FamilyInputs,
    pub proposed: &'a FamilyInputs,
    pub before: &'a Array1<f64>,
    pub after: &'a Array1<f64>,
    pub sequence_of: &'a [usize],
    pub flipped: &'a [usize],
}

/// [`select`], showing `observe` every round's decisions before they are taken; with `boxed`
/// (the written Fishers), under the box claim ([`select_boxed`]).
#[allow(clippy::too_many_arguments)]
pub fn select_observed(
    masked: &Masked,
    base: &FamilyInputs,
    target: &Target,
    mut masks: Vec<Array2<f64>>,
    coder: &Coder,
    observations: f64,
    samples: usize,
    boxed: Option<&[Array2<f64>]>,
    observe: &mut dyn FnMut(&Round<'_>) -> Result<(), String>,
) -> Result<(Vec<Array2<f64>>, Array1<f64>), String> {
    let scale = observations / std::f64::consts::LN_2;
    let rows = base.rows;
    let scored = target.scored_rows();
    // An unscored row's masks stay as given (module note, "Behaviours").
    let scores: Vec<bool> = (0..rows).map(|r| target.scores(r)).collect();
    let mut alpha = vec![0.0f64; rows];
    // Each input's sequence, and per sequence the cap on its inputs' flips (halved when refused;
    // zero once refused at one flip each).
    let sequence_of: Vec<usize> = match &base.layout {
        Some(layout) => layout.sequence.iter().map(|s| *s as usize).collect(),
        None => (0..rows).collect(),
    };
    let sequences = sequence_of.iter().copied().max().map_or(0, |m| m + 1);
    let mut cap = vec![usize::MAX; sequences];
    let mut round = 0u64;
    let started = std::time::Instant::now();
    let mut curvature: Option<Vec<(Array2<f64>, Option<Array2<f64>>)>> = None;
    // What the next round may reuse exactly instead of recomputing: the current masks' forward
    // (when the last proposal was kept whole, its forward *is* the current state's) and, when
    // nothing was kept, also their gradients (the state did not move).
    let mut next_forward: Option<(Array1<f64>, Selected)> = None;
    let mut reuse: Option<(Array1<f64>, Selected, Vec<Array2<f64>>)> = None;
    // The target on the program's device, when it runs on one (module note, "Devices").
    let on_device = masked.on_device(|accelerated| accelerated.target(target))?;
    let on_device = on_device.as_ref();
    loop {
        let family = masked.family(base, &masks);
        let (kl_now, state, grads) = match reuse.take() {
            Some(state) => state,
            None => {
                let (kl_now, mut state) = match next_forward.take() {
                    Some(state) => state,
                    None => Selected::forward(masked, &family, target, on_device, true)?,
                };
                let grads = state.mask_gradients(masked, &family, target, on_device)?;
                (kl_now, state, grads)
            }
        };
        // The Fisher diagonal only ranks proposals (the exact forward decides), so it is measured
        // on the first round and kept: it moves slowly with the masks, and its passes dominate a
        // round's cost.
        if curvature.is_none() {
            curvature = Some(state.fisher(masked, &family, target, on_device, samples, 0x5EED + round)?);
        }
        let curvature = curvature.as_ref().ok_or("no curvature")?;
        let listing_now = coder.bits(&masks);
        // Under the box claim each input's error is its masks' KL plus the box's excess.
        let excess_now = match boxed {
            Some(f) => box_excess_at(masked, base, target, &masks, f)?,
            None => Array1::zeros(rows),
        };
        let before = code(&(&kl_now + &excess_now), &listing_now, observations);
        // Each input's predicted flips, best first, as many as its interaction model says pay
        // (inputs in parallel: each reads only its own row of every array).
        let picks: Vec<(Vec<(usize, usize)>, f64)> = {
            use rayon::prelude::*;
            (0..rows)
                .into_par_iter()
                .map(|r| {
                    if !scores[r] || cap[sequence_of[r]] == 0 {
                        return (Vec::new(), 0.0);
                    }
                    let fresh = coder.fresh(&masks, r);
                    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
                    for (k, (g, (h, _))) in grads.iter().zip(curvature.iter()).enumerate() {
                        let (g, h, m) = (g.row(r), h.row(r), masks[k].row(r));
                        for c in 0..g.len() {
                            let on = m[c] > 0.0;
                            // The corner's change, `∓g + h/2`; under the box claim an off gate also
                            // adds its expected excess, `g/2 + h/6` to second order (`E m = ½`,
                            // `E m² = ⅓`), which turning it on removes.
                            let kl_change = match (boxed.is_some(), on) {
                                (false, true) => -g[c] + 0.5 * h[c],
                                (false, false) => g[c] + 0.5 * h[c],
                                (true, true) => -0.5 * g[c] + (2.0 / 3.0) * h[c],
                                (true, false) => 0.5 * g[c] + h[c] / 3.0,
                            };
                            let listing_change = coder.marginal(&masks, r, k, c, fresh);
                            let net = scale * kl_change + listing_change;
                            if net < 0.0 {
                                candidates.push((net, k, c));
                            }
                        }
                    }
                    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
                    // The k maximising S(k) − α k², S the cumulative predicted saving.
                    let (mut best, mut best_k, mut saving) = (0.0, 0usize, 0.0);
                    for (i, (net, _, _)) in candidates.iter().enumerate() {
                        saving -= net;
                        let k = (i + 1) as f64;
                        let value = saving - alpha[r] * k * k;
                        if value > best {
                            (best, best_k) = (value, i + 1);
                        }
                    }
                    let best_k = best_k.min(cap[sequence_of[r]]);
                    let predicted: f64 = candidates.iter().take(best_k).map(|(net, _, _)| -net).sum();
                    (candidates.into_iter().take(best_k).map(|(_, k, c)| (k, c)).collect(), predicted)
                })
                .collect()
        };
        let mut proposed = masks.clone();
        let mut flipped = vec![0usize; rows];
        for (r, (pick, _)) in picks.iter().enumerate() {
            for &(k, c) in pick {
                proposed[k][[r, c]] = 1.0 - proposed[k][[r, c]];
            }
            flipped[r] = pick.len();
        }
        if flipped.iter().all(|f| *f == 0) {
            return Ok((masks, kl_now));
        }
        let proposed_family = masked.family(base, &proposed);
        let (kl_new, state_new) = Selected::forward(masked, &proposed_family, target, on_device, false)?;
        let excess_new = match boxed {
            Some(f) => box_excess_at(masked, base, target, &proposed, f)?,
            None => Array1::zeros(rows),
        };
        let after = code(&(&kl_new + &excess_new), &coder.bits(&proposed), observations);
        observe(&Round { current: &family, proposed: &proposed_family, before: &before, after: &after, sequence_of: &sequence_of, flipped: &flipped })?;
        // Per sequence: the exact saving of its whole proposal, and its largest flips per input.
        let mut sequence_saving = vec![0.0; sequences];
        let mut sequence_flips = vec![0usize; sequences];
        for r in 0..rows {
            if flipped[r] == 0 {
                continue;
            }
            // The measured interaction of this proposal: what it over-promised, per k².
            let (predicted, actual) = (picks[r].1, before[r] - after[r]);
            let k = flipped[r] as f64;
            alpha[r] = ((predicted - actual) / (k * k)).max(0.0);
        }
        let mut sequence_rows = vec![0usize; sequences];
        for r in 0..rows {
            sequence_saving[sequence_of[r]] += before[r] - after[r];
            sequence_flips[sequence_of[r]] = sequence_flips[sequence_of[r]].max(flipped[r]);
            sequence_rows[sequence_of[r]] += usize::from(scores[r]);
        }
        let (mut kept, mut tried, mut saved) = (0usize, 0usize, 0.0);
        for q in 0..sequences {
            if sequence_flips[q] == 0 {
                continue;
            }
            tried += 1;
            if sequence_saving[q] > 0.0 {
                saved += sequence_saving[q];
                kept += 1;
                // A sequence whose kept round saved less than a bit per scored input is done.
                cap[q] = if sequence_saving[q] < sequence_rows[q] as f64 { 0 } else { usize::MAX };
            } else {
                cap[q] = if sequence_flips[q] <= 1 { 0 } else { sequence_flips[q] / 2 };
            }
        }
        for r in 0..rows {
            if flipped[r] > 0 && sequence_saving[sequence_of[r]] > 0.0 {
                for site in 0..masks.len() {
                    masks[site].row_mut(r).assign(&proposed[site].row(r));
                }
            }
        }
        round += 1;
        if kept == tried {
            // Every proposing input kept its flips, so the masks now equal the proposal.
            next_forward = Some((kl_new.clone(), state_new));
        } else if kept == 0 {
            // Nothing moved: this round's forward and gradients still describe the masks.
            reuse = Some((kl_now.clone(), state, grads));
        }
        let per_scored = |code: &Array1<f64>| (0..rows).filter(|r| target.scores(*r)).map(|r| code[r]).sum::<f64>() / scored.max(1) as f64;
        log::info!(
            "selection round {round} ({:.0}s): {} inputs flipped {} entries, {kept} of {tried} sequences kept, {saved:.0} bits saved; code {:.1} -> {:.1} bits per input",
            started.elapsed().as_secs_f64(),
            flipped.iter().filter(|f| **f > 0).count(),
            flipped.iter().sum::<usize>(),
            per_scored(&before),
            per_scored(&after)
        );
        // Done when every sequence is done (each on its own evidence, so a batch of sequences
        // selects exactly as each would alone).
        if cap.iter().all(|c| *c == 0) {
            let kl_final = match (next_forward.take(), reuse.take()) {
                (Some((kl, _)), _) | (None, Some((kl, _, _))) => kl,
                (None, None) => {
                    let family = masked.family(base, &masks);
                    match on_device {
                        Some(on_device) => masked.on_lowered(|accelerated| accelerated.score_only(&family, on_device))?,
                        None => score_only(masked, &family, target)?,
                    }
                }
            };
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
            ndarray::Zip::from(running).and(&batch).for_each(|r, b| *r = *r * old + *b * new);
        }
        for (running, batch) in self.fishers.iter_mut().zip(fishers) {
            ndarray::Zip::from(running).and(&batch).for_each(|r, b| *r = *r * old + *b * new);
        }
        self.rows = total;
    }
}

/// `(M + λ I)⁻¹` for a symmetric positive semidefinite `M`, with `λ = tr M / dim`: the matrix
/// shrunk halfway to the isotropic matrix of its own mean eigenvalue, so a direction the data barely
/// resolve is not amplified beyond the mean scale.
pub(super) fn shrunk_inverse(m: &Array2<f64>) -> Result<Array2<f64>, String> {
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

/// Apply the shrunk PSD preconditioner by a Cholesky solve, avoiding eigenvectors and an
/// explicit inverse. Singular/invalid factors retain the spectral fallback. This only proposes
/// a direction; the actual candidate's F64 loss still decides whether to commit it.
pub(super) fn shrunk_direction(m: &Array2<f64>, gradient: &Array2<f64>) -> Result<Array2<f64>, String> {
    use gam_linalg::faer_ndarray::FaerCholesky;
    let n = m.nrows();
    if m.ncols() != n || gradient.ncols() != n {
        return Err("preconditioner and gradient shapes disagree".to_string());
    }
    if m.iter().all(|v| *v == 0.0) { return Ok(Array2::zeros(gradient.dim())); }
    let lambda = (0..n).map(|i| m[[i, i]]).sum::<f64>() / n.max(1) as f64;
    if lambda > 0.0 && lambda.is_finite() {
        let mut shifted = m.clone();
        for i in 0..n {
            shifted[[i, i]] += lambda;
            for j in i + 1..n {
                let value = 0.5 * (m[[i, j]] + m[[j, i]]);
                shifted[[i, j]] = value;
                shifted[[j, i]] = value;
            }
        }
        if let Ok(factor) = shifted.cholesky(faer::Side::Lower) {
            let mut direction = gradient.t().to_owned();
            factor.solve_mat_in_place(&mut direction);
            if direction.iter().all(|v| v.is_finite()) { return Ok(direction.reversed_axes()); }
        }
    }
    Ok(gradient.dot(&shrunk_inverse(m)?))
}

/// `g` (C × d, one row per piece) less its part that would move `Σ_c u_c v_cᵀ`, the other side's
/// rows `other` (C × d'): the projection onto `{x : otherᵀ x = 0}`, `g − other (otherᵀother)⁺ otherᵀ g`
/// over the eigenvalues of `otherᵀother` beyond its band. A preconditioning on the right keeps it,
/// since the projection acts on the pieces' index alone.
fn keep_sum(g: &Array2<f64>, other: &Array2<f64>) -> Result<Array2<f64>, String> {
    let gram = fast_atb(other, other);
    let mut sym = gram.clone();
    let n = sym.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (sym[[i, j]] + sym[[j, i]]);
            sym[[i, j]] = v;
            sym[[j, i]] = v;
        }
    }
    let d = super::dense::eigh(sym.view(), gam_linalg::roundoff::SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    let mut scaled = d.vectors.clone();
    for (k, l) in d.values.iter().enumerate() {
        let inverse = if *l > d.band { 1.0 / l } else { 0.0 };
        scaled.column_mut(k).mapv_inplace(|x| x * inverse);
    }
    let pinv = scaled.dot(&d.vectors.t());
    let coefficients = pinv.dot(&fast_atb(other, g));
    Ok(g - &other.dot(&coefficients))
}

/// One preconditioned step of every site's pieces on the total error given the masks, under
/// `claim` (module note, "Claims"; the box claim's written Fishers are the running ones): the
/// direction is the gradient preconditioned by the shrunk read covariance (for `V`) and the shrunk
/// written Fisher (for `U`); its length is the Gauss–Newton minimiser along it, `⟨g, d⟩ / dᵀ H d`
/// with `dᵀ H d` from one forward tangent through the masked program, halved until the total
/// falls. Returns the total before and after the step, or `None` when no step lowered it.
///
/// Every piece on is the site's map, and a step keeps it so exactly: it moves one side of every
/// site (`U` when `seed` is even, `V` when odd; the driver's sequence counter alternates them), along
/// the direction projected onto the moves that leave `Σ_c u_c v_cᵀ` unchanged, `Vᵀ dU = 0` or
/// `Uᵀ dV = 0`. Both are linear, so the sum holds to rounding at every step length.
///
/// On a device (module note, "Devices") the corner claim's step runs there: its forward, gradients,
/// Fishers, read moments and curvature tangent stay resident, and each backtracking trial is a
/// float64 forward of the device twin, refreshed with the trial operators.
pub fn step_pieces(
    masked: &mut Masked,
    base: &FamilyInputs,
    target: &Target,
    masks: &[Array2<f64>],
    samples: usize,
    seed: u64,
    running: &mut Running,
    claim: Claim,
) -> Result<Option<(f64, f64)>, String> {
    if masked.head.is_some() || masked.released() {
        return Err("pieces steps need the whole model in the program and the training state".to_string());
    }
    let family = masked.family(base, masks);
    // The forward and everything read off it, on the device twin when there is one (the box
    // claim's there only on sites of rank-one blocks).
    let lowered = if claim == Claim::Corner || (0..masked.sites.len()).all(|k| masked.is_rank_one(k)) {
        masked.on_device(|accelerated| {
            let on_device = accelerated.target(target)?;
            let state = accelerated.forward(&family, &on_device)?;
            let grads = accelerated.gradients(masked, &state, masks)?;
            let curvature = accelerated.fisher(masked, &state, &on_device, samples, seed, true)?;
            let covariances = accelerated.covariances(masked, &state)?;
            Ok((on_device, state, grads, curvature, covariances))
        })?
    } else {
        None
    };
    let (kl_now, grads, curvature, batch_covariances, evaluated, device_target) = match lowered {
        Some((on_device, state, grads, curvature, covariances)) => (state.kl.clone(), grads, curvature, covariances, Evaluated::Device(state), Some(on_device)),
        None => {
            let (kl_now, trace, cotangent) = forward(masked, &family, target)?;
            let grads = gradients(masked, &family, &trace, masks, cotangent.clone())?;
            let curvature = fisher(masked, &family, &trace, target, samples, seed, true)?;
            let rows = trace.values[masked.program.output].nrows() as f64;
            let mut batch_covariances = Vec::new();
            for site in &masked.sites {
                let reads = read_values(&trace, site)?;
                batch_covariances.push(fast_atb(&reads, &reads) / rows);
            }
            (kl_now, grads, curvature, batch_covariances, Evaluated::Host(trace, cotangent), None)
        }
    };
    // The preconditioners are the running means over every input stepped on so far.
    let rows = kl_now.len() as f64;
    let fishers = curvature.into_iter().map(|(_, f)| f.ok_or("no written Fisher")).collect::<Result<Vec<_>, _>>()?;
    running.absorb(batch_covariances, fishers, rows);
    // The error under the claim (module note, "Claims"), and its gradients.
    let (total, grads) = match claim {
        Claim::Corner => (kl_now.sum(), grads),
        Claim::Box => {
            let (excess, box_grads) = match &evaluated {
                Evaluated::Host(trace, cotangent) => box_excess(masked, &family, trace, masks, cotangent.clone(), &running.fishers, true)?,
                Evaluated::Device(state) => masked.on_lowered(|accelerated| accelerated.box_excess(masked, state, masks, &running.fishers, true))?,
            };
            let box_grads = box_grads.ok_or("no box gradients")?;
            let grads = grads.into_iter().zip(box_grads).map(|((m, v, u), (bv, bu))| (m, v + bv, u + bu)).collect::<Vec<_>>();
            (kl_now.sum() + excess.sum(), grads)
        }
    };
    // The direction, as the tangent of each operator it moves, on one side of every site and
    // projected onto the sum-preserving moves (doc above).
    let moves_u = seed % 2 == 0;
    let mut moves: Vec<(usize, Array2<f64>)> = Vec::new();
    let mut slope = 0.0;
    for (k, (_, v_gradient, u_gradient)) in grads.into_iter().enumerate() {
        let library = masked.library(k)?;
        let (ro, wo) = (&masked.read_offsets[k], &masked.write_offsets[k]);
        if moves_u {
            drop(v_gradient);
            let g = keep_sum(&u_gradient, &library.v)?;
            let du = shrunk_direction(&running.fishers[k], &g)?;
            slope += (&g * &du).sum();
            for (i, &op) in masked.u_ops[k].iter().enumerate() {
                moves.push((op, du.slice(s![.., wo[i]..wo[i + 1]]).t().to_owned()));
            }
        } else {
            drop(u_gradient);
            let g = keep_sum(&v_gradient, &library.u)?;
            let dv = shrunk_direction(&running.covariances[k], &g)?;
            slope += (&g * &dv).sum();
            for (j, &op) in masked.v_ops[k].iter().enumerate() {
                moves.push((op, dv.slice(s![.., ro[j]..ro[j + 1]]).to_owned()));
            }
        }
    }
    if !(slope > 0.0) {
        return Ok(None);
    }
    // `dᵀ H d`: the output tangent of the direction, in each row's softmax Fisher.
    let quadratic = {
        let tangents: BTreeMap<usize, Array2<f64>> = moves.iter().map(|(op, t)| (*op, t.clone())).collect();
        match (&evaluated, &device_target) {
            (Evaluated::Device(state), Some(on_device)) => {
                masked.on_device(|accelerated| accelerated.quadratic(state, on_device, &tangents))?.ok_or("device: the twin went away")?
            }
            (Evaluated::Device(_), None) => return Err("device: a resident state without its target".to_string()),
            (Evaluated::Host(trace, _), _) => {
                let output = super::derivatives::jvp(&masked.program, &family, trace, &tangents).map_err(|e| e.to_string())?;
                let logits = &trace.values[masked.program.output];
                let mut quadratic = 0.0;
                for r in (0..logits.nrows()).filter(|r| target.scores(*r)) {
                    let q = softmax(logits.row(r));
                    let t = output.row(r);
                    let mean: f64 = q.iter().zip(t.iter()).map(|(a, b)| a * b).sum();
                    quadratic += q.iter().zip(t.iter()).map(|(a, b)| a * (b - mean) * (b - mean)).sum::<f64>();
                }
                quadratic
            }
        }
    };
    drop(evaluated);
    // Every trial is the float64 operator less the step; the originals are shared, not copied,
    // and restored exactly when no step lowers the total.
    let originals: Vec<Arc<Operator>> = moves.iter().map(|(op, _)| Arc::clone(&masked.program.operators[*op])).collect();
    let mut eta = if quadratic > 0.0 { slope / quadratic } else { 1.0 };
    let floor = eta * f64::EPSILON;
    while eta > floor {
        for ((op, t), original) in moves.iter().zip(&originals) {
            let values = &*original.matrix_cow() - &(t * eta);
            masked.program.operators[*op] = dense(original.name.clone(), original.rows.clone(), original.cols.clone(), values)?;
        }
        let trial = match (claim, &device_target) {
            (Claim::Corner, Some(on_device)) => {
                masked.on_device(|accelerated| accelerated.score_only(&family, on_device))?.ok_or("device: the twin went away")?.sum()
            }
            (Claim::Corner, None) => score_only(masked, &family, target)?.sum(),
            (Claim::Box, Some(on_device)) => masked.on_lowered(|accelerated| {
                let state = accelerated.forward(&family, on_device)?;
                Ok(state.kl.sum() + accelerated.box_excess(masked, &state, masks, &running.fishers, false)?.0.sum())
            })?,
            (Claim::Box, None) => {
                let (kl_trial, trial_trace, trial_cotangent) = forward(masked, &family, target)?;
                kl_trial.sum() + box_excess(masked, &family, &trial_trace, masks, trial_cotangent, &running.fishers, false)?.0.sum()
            }
        };
        if trial < total {
            return Ok(Some((total, trial)));
        }
        eta *= 0.5;
    }
    for ((op, _), original) in moves.iter().zip(originals) {
        masked.program.operators[*op] = original;
    }
    Ok(None)
}

/// A step's forward: on the CPU (its trace and the KL's cotangent) or resident on the device.
enum Evaluated {
    Host(Trace, Array2<f64>),
    Device(State),
}

/// What an explanation claims of its off subcomponents (module note, "Claims").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Claim {
    /// Off means absent: the error is the KL of the masks themselves.
    Corner,
    /// Off means anywhere in `[0, 1]`: the error is the KL expected over every off gate uniform.
    Box,
}

/// Per site, the gradients of the box excess in `V` and `U`.
pub type BoxGradients = Vec<(Array2<f64>, Array2<f64>)>;

/// Per input, what the box claim adds to the masks' own KL (module note, "Claims"), from a masked
/// forward's trace and its KL's cotangent at the program's output, in each site's written Fisher
/// `fishers[k]` (a per-input mean). With `gradients`, also its gradients in every site's `V`
/// (C × d_in) and `U` (C × d_out), the KL's gradient at the written values held fixed (they steer
/// steps; the error itself decides).
pub fn box_excess(
    masked: &Masked,
    family: &FamilyInputs,
    trace: &Trace,
    masks: &[Array2<f64>],
    cotangent: Array2<f64>,
    fishers: &[Array2<f64>],
    gradients: bool,
) -> Result<(Array1<f64>, Option<BoxGradients>), String> {
    let back = vjp(&masked.program, family, trace, cotangent).map_err(|e| e.to_string())?;
    let rows = trace.values[masked.program.output].nrows();
    let mut excess = Array1::<f64>::zeros(rows);
    let mut out = Vec::new();
    for (k, site) in masked.sites.iter().enumerate() {
        let library = masked.library(k)?;
        let f = &fishers[k];
        // The off blocks' coordinates, per column.
        let off = masked.expand(k, &masks[k]).mapv(|m| 1.0 - m);
        let a = &trace.values[masked.z[k]] * &off;
        let s_out = a.dot(&library.u);
        let written: Vec<Array2<f64>> = masked.written[k]
            .iter()
            .map(|n| back[*n].clone().unwrap_or_else(|| Array2::zeros(trace.values[*n].dim())))
            .collect();
        let views: Vec<_> = written.iter().map(|w| w.view()).collect();
        let g = ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?;
        let sf = s_out.dot(f);
        // Each block's own term `Z_cᵀ F Z_c = a_cᵀ (U_c F U_cᵀ) a_c`, per input.
        let mut own = Array1::<f64>::zeros(rows);
        let mut own_a = Array2::<f64>::zeros(a.dim());
        let mut start = 0;
        let mut blocks_ufu = Vec::new();
        for &r in masked.ranks(k) {
            let u_block = library.u.slice(s![start..start + r, ..]);
            let ufu = u_block.dot(f).dot(&u_block.t());
            let a_block = a.slice(s![.., start..start + r]);
            let a_ufu = a_block.dot(&ufu);
            own += &(&a_ufu * &a_block).sum_axis(Axis(1));
            own_a.slice_mut(s![.., start..start + r]).assign(&a_ufu);
            blocks_ufu.push(ufu);
            start += r;
        }
        excess += &((&g * &s_out).sum_axis(Axis(1)) * 0.5 + (&sf * &s_out).sum_axis(Axis(1)) * 0.125 + &own * (1.0 / 24.0));
        if gradients {
            // ∂/∂S = ½ g + ¼ S F; ∂/∂a = (∂/∂S) Uᵀ + a (U_c F U_cᵀ)/12 within each block.
            let g_s = &g * 0.5 + &sf * 0.25;
            let g_a = &g_s.dot(&library.u.t()) + &(own_a * (1.0 / 12.0));
            let mut u_gradient = a.t().dot(&g_s);
            let mut start = 0;
            for &r in masked.ranks(k) {
                let a_block = a.slice(s![.., start..start + r]);
                let u_block = library.u.slice(s![start..start + r, ..]);
                let extra = a_block.t().dot(&a_block).dot(&u_block).dot(f) * (1.0 / 12.0);
                let mut target_rows = u_gradient.slice_mut(s![start..start + r, ..]);
                target_rows += &extra;
                start += r;
            }
            let reads = read_values(trace, site)?;
            let v_gradient = (&g_a * &off).t().dot(&reads);
            out.push((v_gradient, u_gradient));
        }
    }
    Ok((excess, gradients.then_some(out)))
}

/// Each site's read mean and second moment about it, `E[(x − μ)(x − μ)ᵀ]`, on the native program
/// over `batches` of inputs (forward passes only).
pub fn site_second_moments(
    program: &OperatorProgram,
    sites: &[Site],
    batches: impl IntoIterator<Item = FamilyInputs>,
) -> Result<Vec<(Array1<f64>, Array2<f64>)>, String> {
    let mut sums: Vec<(Array1<f64>, Array2<f64>)> = Vec::new();
    let mut rows = 0.0;
    for inputs in batches {
        let trace = program.execute(&inputs, false).map_err(|e| e.to_string())?;
        for (k, site) in sites.iter().enumerate() {
            let x = read_values(&trace, site)?;
            if sums.len() <= k {
                sums.push((Array1::zeros(x.ncols()), Array2::zeros((x.ncols(), x.ncols()))));
            }
            sums[k].0 += &x.sum_axis(Axis(0));
            sums[k].1 += &fast_atb(&x, &x);
        }
        rows += inputs.rows as f64;
    }
    if rows == 0.0 {
        return Err("second moments need inputs".to_string());
    }
    Ok(sums
        .into_iter()
        .map(|(sum, outer)| {
            let mean = sum / rows;
            let second = outer / rows - &mean.view().insert_axis(Axis(1)).dot(&mean.view().insert_axis(Axis(0)));
            (mean, second)
        })
        .collect())
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

/// `(M^{1/2}, M^{-1/2})` of a symmetric positive semidefinite matrix shrunk halfway to its mean
/// eigenvalue (as the pieces' preconditioner is), so a direction the data barely resolve is
/// neither amplified nor dropped.
fn shrunk_roots(m: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), String> {
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
    let (mut half, mut inverse) = (d.vectors.clone(), d.vectors.clone());
    for (k, l) in d.values.iter().enumerate() {
        let shifted = l.max(0.0) + lambda;
        let (h, i) = if shifted > 0.0 { (shifted.sqrt(), 1.0 / shifted.sqrt()) } else { (0.0, 0.0) };
        half.column_mut(k).mapv_inplace(|x| x * h);
        inverse.column_mut(k).mapv_inplace(|x| x * i);
    }
    Ok((half.dot(&d.vectors.t()), inverse.dot(&d.vectors.t())))
}

/// New pieces for site `k` from what its selection leaves out on a sequence: the site's own map on
/// the reads less the listed pieces, `r_t = W x̃_t − Σ_{c on} u_c z_tc`, regressed on the centred
/// reads in the pieces' metric (written Fisher on the output side, the sequence's read covariance
/// on the input side). Each singular pair `σ p qᵀ` of the whitened regression is a candidate piece
/// `u = F^{-1/2} p √σ`, `v = Σ^{-1/2} q √σ`, kept when the KL bits it would recover over the
/// sequence, `n N σ² / (2 ln 2)` with `N` the sequence's rows, exceed its library bits
/// `bits_per_piece`. Returns the new pieces as `(v: K × d_in, u: K × d_out)`.
pub fn dropped_atoms(
    masked: &Masked,
    k: usize,
    trace: &Trace,
    mask: &Array2<f64>,
    running: &Running,
    observations: f64,
    bits_per_piece: f64,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    if masked.released() {
        return Err("dropped atoms need the training state".to_string());
    }
    let library = masked.library(k)?;
    let z = &trace.values[masked.z[k]];
    let centred = &read_values(trace, &masked.sites[k])? - &library.mean;
    // What the site's own map gives on these reads, less what the listed pieces give.
    let residual = centred.dot(&masked.w(k)?.t()) - (z * &masked.expand(k, mask)).dot(&library.u);
    let rows = centred.nrows() as f64;
    let (f_half, f_inverse) = shrunk_roots(&running.fishers[k])?;
    // The reads whitened by this sequence's own covariance (on its support), so the
    // cross-covariance below is the regression itself.
    let covariance = fast_atb(&centred, &centred) / rows;
    let decomposed_reads = super::dense::eigh(covariance.view(), gam_linalg::roundoff::SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
    let mut s_inverse_factor = decomposed_reads.vectors.clone();
    for (j, l) in decomposed_reads.values.iter().enumerate() {
        let inverse = if *l > decomposed_reads.band { 1.0 / l.sqrt() } else { 0.0 };
        s_inverse_factor.column_mut(j).mapv_inplace(|x| x * inverse);
    }
    let s_inverse = s_inverse_factor.dot(&decomposed_reads.vectors.t());
    // The whitened regression of the left-out map on the whitened reads.
    let whitened_out = residual.dot(&f_half);
    let whitened_in = centred.dot(&s_inverse);
    let cross = fast_atb(&whitened_out, &whitened_in) / rows;
    let decomposed = super::dense::svd(cross.view(), false).map_err(|e| format!("{e:?}"))?;
    let mut vs = Vec::new();
    let mut us = Vec::new();
    for (i, sigma) in decomposed.singular_values.iter().enumerate() {
        // The regression coefficient σ on unit-variance whitened reads recovers σ² of energy per
        // input: KL ≈ ½ energy in the written Fisher.
        let recovered = observations * sigma * sigma * rows / (2.0 * std::f64::consts::LN_2);
        if recovered <= bits_per_piece {
            break;
        }
        let root = sigma.sqrt();
        us.push(f_inverse.dot(&decomposed.u.column(i)) * root);
        vs.push(s_inverse.dot(&decomposed.vt.row(i)) * root);
    }
    let stack = |rows: &[Array1<f64>], width: usize| Array2::from_shape_fn((rows.len(), width), |(i, j)| rows[i][j]);
    Ok((stack(&vs, centred.ncols()), stack(&us, library.u.ncols())))
}

/// `library` with the pieces `(v, u)` appended.
pub fn with_pieces(library: &Library, v: &Array2<f64>, u: &Array2<f64>) -> Result<Library, String> {
    Ok(Library {
        v: ndarray::concatenate(Axis(0), &[library.v.view(), v.view()]).map_err(|e| e.to_string())?,
        u: ndarray::concatenate(Axis(0), &[library.u.view(), u.view()]).map_err(|e| e.to_string())?,
        mean: library.mean.clone(),
    })
}

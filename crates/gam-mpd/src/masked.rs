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
//! declares. The selection codes the corner claim: off subcomponents are absent and an input's
//! error is the KL of its masks. Under the box claim each off gate may be anywhere in `[0, 1]` and
//! the error is the worst KL over the box; only the pieces steps train under it
//! ([`Claim::Box`], charged [`box_upper`]). Four facts fix how it is measured.
//!
//! * **An attack bounds from below.** Any point an attack evaluates ([`box_excess_at`]: the masks,
//!   every layer's vertex, an adversary's ascent) has a loss at most the box's worst, so what it
//!   finds is a lower bound: it can refute a claim, never certify one, and it is never charged.
//! * **The expectation is no claim.** The KL expected over uniform independent off gates is
//!   refinement-gameable (one off subcomponent split into `q` copies of `1/q` leaves the box
//!   unchanged and cuts its own term by `1/q`), and its second-order expansion is unbounded below:
//!   a blocks fit charged it drove it to −33 nats a token on p31. Nothing measures it.
//! * **Shared and per-input worst cases differ.** VPD's adversary picks one gate setting `a` for all
//!   `M` inputs, ours lets each input have its own: `sup_a E_X ℓ(a, X) ≤ E_X sup_a ℓ(a, X) ≤ M sup_a
//!   E_X ℓ(a, X)` (`ℓ ≥ 0`), the gap up to the factor `M` when every input's worst point differs.
//!   The two claims are compared only under the same one.
//! * **Our charge is an upper bound.** At every site the box moves the written value by `Σ_off
//!   (1 − m_c) Z_c` from the masks' point, `Z_c` an off block's real contribution on the masked
//!   forward's own reads, so in the input's Fisher every point is within `Σ_off ‖Z_c‖_F` of it and
//!   its added KL is at most `½ (Σ_off ‖Z_c‖_F)²` to second order ([`box_upper`]; the site code of
//!   `super::site_fit` charges the same bound on the model's own reads). Splitting a subcomponent
//!   cannot lower it, and two off subcomponents that cancel are each charged their own size.
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
//! lowered program's proposal arithmetic. Where its only device has no float64 (the Apple GPU,
//! `masked_device::training_device`), the program is lowered there on the first proposal
//! ([`Masked::on_proposals`]: a step's gradients, Fishers, covariances and curvature, in f32)
//! and every decision runs on the CPU. Elsewhere everything runs on the CPU.
//!
//! # Screened heads
//!
//! On the CPU under the corner claim, a selection trial's logits are the current masks' plus the
//! change of the head's hidden rows times the head, `z = z_cur + (h − h_cur) A`, that product in
//! f32 on the Apple GPU with its derived band (`gam_gpu::banded`), which shrinks with the change.
//! Its KL is within twice the logits' error of the float64 one (the KL's gradient in the logits,
//! `q − p`, has `ℓ₁` norm at most 2), so a sequence's keep/refuse is taken from the screened codes
//! only when its saving clears the sum of its rows' bands (from zero, from every other option, and
//! from any further rule on the saving); otherwise its rows get the float64 head first. A screened
//! decision is the one the exact codes take, and the head runs in float64 only where one is close.

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
    if site.reads.is_empty() || site.writes.is_empty() || site.reads.iter().chain(&site.writes).any(|n| *n >= interfaces.len()) {
        return Err(format!("{}: a site needs valid read and written nodes", site.name));
    }
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
    /// The box claim's terms ([`BoxTerms`]) with what they were computed from: the `U` operators
    /// and a fingerprint of the written Fishers.
    box_terms: Mutex<Option<(Vec<Arc<Operator>>, u64, Arc<BoxTerms>)>>,
    /// Per site, per written node, per read node the library's sum `U_iᵀ V_j` as an operator
    /// ([`Masked::dense_program`]), with the `V` and `U` operators it was computed from.
    sums: Mutex<Option<(Vec<Arc<Operator>>, Arc<Vec<Vec<Vec<Arc<Operator>>>>>)>>,
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
        if sites.len() != libraries.len() {
            return Err(format!("{} libraries for {} sites", libraries.len(), sites.len()));
        }
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
            if library.v.ncols() != ro[reads.len()] || library.u.dim() != (pieces, wo[writes.len()]) {
                return Err(format!("{}: library shapes {:?}, {:?} do not match {pieces} pieces with {} reads and {} writes", site.name, library.v.dim(), library.u.dim(), ro[reads.len()], wo[writes.len()]));
            }
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
            box_terms: Mutex::new(None),
            sums: Mutex::new(None),
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
    /// to date with the program's; `None` when the program runs on the CPU or its twin has no
    /// float64 (it may not decide). Calls hold the twin in turn, so `run` must not call this again.
    pub fn on_device<T>(&self, run: impl FnOnce(&Accelerated) -> Result<T, String>) -> Result<Option<T>, String> {
        self.on_twin(true, run)
    }

    /// [`Masked::on_device`] for proposals only: also on a twin without float64 (module note,
    /// "Devices"), so whatever `run` returns proposes and decides nothing.
    pub fn on_proposals<T>(&self, run: impl FnOnce(&Accelerated) -> Result<T, String>) -> Result<Option<T>, String> {
        self.on_twin(false, run)
    }

    fn on_twin<T>(&self, decides: bool, run: impl FnOnce(&Accelerated) -> Result<T, String>) -> Result<Option<T>, String> {
        if self.head.is_some() {
            return Ok(None);
        }
        let mut lowered = self.lowered.lock().map_err(|_| "device: a poisoned lowering".to_string())?;
        if matches!(*lowered, Lowered::Untried) {
            let device = match super::masked_device::device()? {
                Some(device) => Some(device),
                None => super::masked_device::training_device()?,
            };
            // A twin without float64 serves only proposals: it is lowered when one first asks.
            if decides && device.as_ref().is_some_and(|d| !d.float64()) {
                return Ok(None);
            }
            *lowered = match device {
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
            Lowered::Device(accelerated) if !decides || accelerated.decides() => {
                accelerated.refresh(self)?;
                run(accelerated).map(Some)
            }
            Lowered::Untried | Lowered::Host | Lowered::Device(_) => Ok(None),
        }
    }

    /// The box claim's terms for the current library in the written Fishers `fishers`, computed
    /// once while neither changes.
    pub(crate) fn box_terms(&self, fishers: &[Array2<f64>]) -> Result<Arc<BoxTerms>, String> {
        let ops = self.u_operators();
        let print = fisher_print(fishers);
        let mut cache = self.box_terms.lock().map_err(|_| "box terms: a poisoned cache".to_string())?;
        if let Some((held, held_print, terms)) = &*cache
            && *held_print == print
            && held.len() == ops.len()
            && held.iter().zip(&ops).all(|(a, b)| Arc::ptr_eq(a, b))
        {
            return Ok(Arc::clone(terms));
        }
        *cache = None;
        let terms = Arc::new(BoxTerms::new(self, fishers)?);
        *cache = Some((ops, print, Arc::clone(&terms)));
        Ok(terms)
    }

    /// The program with every site in `dense` computed as if all its gates were on, through its
    /// library's sum `W = Σ_c u_c v_cᵀ`: each read node enters each written node through one
    /// operator `U_iᵀ V_j`, and the site's coordinates become a copy of its mask, so a dense site
    /// costs what the model's own map does. Its values agree with this program's wherever those
    /// sites' masks are all one (up to rounding); nothing else of the program changes.
    pub(crate) fn dense_program(&self, dense: &[bool]) -> Result<OperatorProgram, String> {
        let sums = self.sums()?;
        let interfaces = self.program.interfaces().map_err(|e| e.to_string())?;
        let mut program = self.program.clone();
        for (k, site) in self.sites.iter().enumerate() {
            if !dense[k] {
                continue;
            }
            program.nodes[self.z[k]] = Node::Raw { slot: self.slots[k] };
            for (i, &written) in site.writes.iter().enumerate() {
                let Node::Affine { terms, bias } = &program.nodes[written] else {
                    return Err(format!("{}: a written node that is not affine", site.name));
                };
                let (terms, bias) = (terms.clone(), *bias);
                let mut rewritten = Vec::with_capacity(terms.len() + site.reads.len());
                for (argument, operator) in terms {
                    if argument == self.masked[k] {
                        for (j, &read) in site.reads.iter().enumerate() {
                            let op = &sums[k][i][j];
                            if op.rows.width() != interfaces[written].width() || op.cols.width() != interfaces[read].width() {
                                return Err(format!("{}: a library sum of the wrong shape", site.name));
                            }
                            program.operators.push(Arc::clone(op));
                            rewritten.push((read, program.operators.len() - 1));
                        }
                    } else {
                        rewritten.push((argument, operator));
                    }
                }
                program.nodes[written] = Node::Affine { terms: rewritten, bias };
            }
        }
        Ok(program)
    }

    /// Every site's `U` operators, in site order.
    pub(crate) fn u_operators(&self) -> Vec<Arc<Operator>> {
        self.u_ops.iter().flatten().map(|&op| Arc::clone(&self.program.operators[op])).collect()
    }

    /// The library's sums of [`Masked::dense_program`], computed once per library.
    fn sums(&self) -> Result<Arc<Vec<Vec<Vec<Arc<Operator>>>>>, String> {
        let ops: Vec<Arc<Operator>> = self.v_ops.iter().chain(&self.u_ops).flatten().map(|&op| Arc::clone(&self.program.operators[op])).collect();
        let mut cache = self.sums.lock().map_err(|_| "library sums: a poisoned cache".to_string())?;
        if let Some((held, sums)) = &*cache
            && held.len() == ops.len()
            && held.iter().zip(&ops).all(|(a, b)| Arc::ptr_eq(a, b))
        {
            return Ok(Arc::clone(sums));
        }
        *cache = None;
        let interfaces = self.program.interfaces().map_err(|e| e.to_string())?;
        let mut sums = Vec::new();
        for (k, site) in self.sites.iter().enumerate() {
            let mut per_written = Vec::new();
            for (i, &written) in site.writes.iter().enumerate() {
                let u = self.program.operators[self.u_ops[k][i]].matrix_cow();
                let mut per_read = Vec::new();
                for (j, &read) in site.reads.iter().enumerate() {
                    let v = self.program.operators[self.v_ops[k][j]].matrix_cow();
                    // `U_i` is held as `d_out × C`, `V_j` as `C × d_in`.
                    let w = gam_linalg::faer_ndarray::fast_ab(&*u, &*v);
                    per_read.push(dense(format!("{}·W{i}{j}", site.name), interfaces[written].clone(), interfaces[read].clone(), w)?);
                }
                per_written.push(per_read);
            }
            sums.push(per_written);
        }
        let sums = Arc::new(sums);
        *cache = Some((ops, Arc::clone(&sums)));
        Ok(sums)
    }

    /// [`Masked::on_proposals`] where the program was found lowered (its state's twin).
    fn on_lowered<T>(&self, run: impl FnOnce(&Accelerated) -> Result<T, String>) -> Result<T, String> {
        self.on_proposals(run)?.ok_or_else(|| "device: the masked program left its device".to_string())
    }

    /// Per site, its `z` node and the mask node its `z̃ = z ⊙ m` reads: `z` is read through that
    /// product alone, so a forward whose `z` no gradient reads needs it only at the mask's nonzeros
    /// ([`OperatorProgram::execute_gated`]).
    pub fn gates(&self) -> Vec<(usize, usize)> {
        self.z
            .iter()
            .zip(&self.masked)
            .filter_map(|(z, masked)| match &self.program.nodes[*masked] {
                Node::Hadamard { left, right } if left == z => Some((*z, *right)),
                _ => None,
            })
            .collect()
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


    /// Site `k`'s `V` operators, one per read node (each `pieces × d_in` of its node).
    pub fn v_ops(&self, k: usize) -> &[usize] {
        &self.v_ops[k]
    }

    /// Site `k`'s `U` operators, one per written node (each `d_out × pieces`: the library's `Uᵀ`).
    pub fn u_ops(&self, k: usize) -> &[usize] {
        &self.u_ops[k]
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

    /// Site `k`'s pieces' `U` (C × d_out), without the rest of its library.
    pub(crate) fn u(&self, k: usize) -> Result<Array2<f64>, String> {
        let u_blocks: Vec<_> = self.u_ops[k].iter().map(|&op| self.program.operators[op].matrix_cow()).collect();
        ndarray::concatenate(Axis(1), &u_blocks.iter().map(|b| b.t()).collect::<Vec<_>>()).map_err(|e| e.to_string())
    }

    /// Site `k`'s own map `W` (written × read), from the model's operators, which the program
    /// keeps until [`Masked::release_training_state`].
    pub fn w(&self, k: usize) -> Result<Array2<f64>, String> {
        if self.released {
            return Err(format!("{}: its map was released", self.sites[k].name));
        }
        matrix(&self.program, &self.sites[k])
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
        if k >= self.sites.len() {
            return Err(format!("library site {k} is out of range"));
        }
        if library.v.nrows() != self.pieces[k] || library.u.nrows() != self.pieces[k] {
            return Err(format!("{}: {} pieces set into a site of {}", self.sites[k].name, library.v.nrows(), self.pieces[k]));
        }
        let (ro, wo) = (&self.read_offsets[k], &self.write_offsets[k]);
        if library.v.ncols() != *ro.last().unwrap_or(&0) || library.u.ncols() != *wo.last().unwrap_or(&0) {
            return Err(format!("{}: library widths do not match the site's reads and writes", self.sites[k].name));
        }
        // Construct every operator before replacing any: an invalid later block must not
        // leave a mixture of old and new factors installed.
        let mut replacements = Vec::new();
        for (j, &op) in self.v_ops[k].iter().enumerate() {
            let old = &self.program.operators[op];
            let block = library.v.slice(s![.., ro[j]..ro[j + 1]]).to_owned();
            replacements.push((op, dense(old.name.clone(), old.rows.clone(), old.cols.clone(), block)?));
        }
        for (i, &op) in self.u_ops[k].iter().enumerate() {
            let old = &self.program.operators[op];
            let block = library.u.slice(s![.., wo[i]..wo[i + 1]]).t().to_owned();
            replacements.push((op, dense(old.name.clone(), old.rows.clone(), old.cols.clone(), block)?));
        }
        for (op, replacement) in replacements { self.program.operators[op] = replacement; }
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
    timed("score only", || score_only_untimed(masked, family, target))
}

fn score_only_untimed(masked: &Masked, family: &FamilyInputs, target: &Target) -> Result<Array1<f64>, String> {
    if let Some(values) = masked.on_device(|accelerated| accelerated.score_only(family, &accelerated.target(target)?))? {
        return Ok(values);
    }
    // Only the KL is read, so each site's `z` is evaluated at its mask's nonzeros alone.
    let trace = masked.program.execute_gated(family, &masked.gates()).map_err(|e| e.to_string())?;
    Ok(kl_score_only(target, &*logits(masked, family, &trace, target)?))
}

/// An attack on the box claim, per input (module note, "Claims"): the worst excess over the masks'
/// own KL, per sequence, of the box points evaluated. They are the masks themselves (no excess),
/// every layer's vertex (that layer's sites at the masks, every other site's off gates at 1), and
/// an adversary's points (sign ascent over the off gates, all together and one layer's alone),
/// each of whose KL is exact; the adversary's are taken per token, each token its own worst point.
/// A sequence takes the point of its largest total, so a set whose layers only cancel each other's
/// errors is caught. A lower bound on the box's worst case: it refutes a claim, never charged.
pub fn box_excess_at(masked: &Masked, base: &FamilyInputs, target: &Target, masks: &[Array2<f64>]) -> Result<Array1<f64>, String> {
    let corner = score_only(masked, &masked.family(base, masks), target)?;
    box_excess_from(masked, base, target, masks, &corner)
}

/// [`box_excess_at`] from the masks' own KL `corner`.
fn box_excess_from(masked: &Masked, base: &FamilyInputs, target: &Target, masks: &[Array2<f64>], corner: &Array1<f64>) -> Result<Array1<f64>, String> {
    // The vertices share every gate on up to their own layer (as in a selection).
    let lowered = masked.on_device(|_| Ok(()))?.is_some();
    let all_on = if !lowered && masked.head.is_none() {
        let on: Vec<Array2<f64>> = masks.iter().map(|m| Array2::ones(m.dim())).collect();
        let every = masked.dense_program(&vec![true; masked.sites.len()])?;
        let mut trace = every.execute(&masked.family(base, &on), false).map_err(|e| e.to_string())?;
        trace.values.truncate(masked.z.iter().copied().max().unwrap_or(0));
        Some(trace)
    } else {
        None
    };
    box_worst(masked, base, target, masks, corner, all_on.as_ref())
}

/// The program's logits when they are one dense product of a hidden node and nothing after it:
/// the hidden node, the operator and the product's layout (`x Aᵀ` for an affine node, `x A` for a
/// transposed one).
fn lone_head(masked: &Masked) -> Option<(usize, usize, gam_gpu::banded::Layout)> {
    use gam_gpu::banded::Layout;
    if masked.head.is_some() {
        return None;
    }
    let program = &masked.program;
    let logits = match &program.nodes[program.output] {
        Node::Readout { input, basis } if matches!(program.bases[*basis], super::operator_program::Basis::Indicator { .. }) => *input,
        Node::Readout { .. } => return None,
        _ => program.output,
    };
    let (hidden, operator, layout) = match &program.nodes[logits] {
        Node::Affine { terms, bias: None } if terms.len() == 1 => (terms[0].0, terms[0].1, Layout::Transposed),
        Node::Transposed { input, operator } => (*input, *operator, Layout::AsStored),
        _ => return None,
    };
    matches!(program.operators[operator].body, OperatorBody::Dense { .. }).then_some((hidden, operator, layout))
}

/// A box point's KL per input from its hidden values `hidden` before the head ([`lone_head`]),
/// certified against the worst so far: the head's product runs in f32 on the device, each row's
/// KL with a bound on its error (`2 maxⱼ band`, the KL's gradient in the logits being `q − p`, of
/// `ℓ₁` norm at most 2). A sequence whose total excess over `corner` could exceed its worst
/// `totals` gets its rows' KL exactly; every other banded row is lowered by its band, so the point
/// cannot win that sequence, as its exact KL could not either. Also returns the logits (f32 where
/// the device ran them), which only steer.
fn certified_point(
    masked: &Masked,
    target: &Target,
    hidden: &Array2<f64>,
    (operator, layout): (usize, gam_gpu::banded::Layout),
    corner: &Array1<f64>,
    totals: &[f64],
    sequence_of: &[usize],
) -> Result<(Array1<f64>, Array2<f64>), String> {
    let rows = hidden.nrows();
    let op = &masked.program.operators[operator];
    let Some(banded) = super::device::banded_product(op, hidden, layout).map_err(|e| e.to_string())? else {
        let logits = exact_head(masked, hidden, operator, layout);
        return Ok((kl_score_only(target, &logits), logits));
    };
    let band = Array1::from_shape_fn(rows, |r| if target.scores(r) { 2.0 * banded.band.row_max(r) } else { 0.0 });
    let logits = banded.values;
    let mut kl_point = kl_score_only(target, &logits);
    let mut upper = vec![0.0; totals.len()];
    for r in 0..rows {
        upper[sequence_of[r]] += kl_point[r] + band[r] - corner[r];
    }
    let is_open: Vec<bool> = (0..rows).map(|r| band[r] > 0.0 && upper[sequence_of[r]] > totals[sequence_of[r]]).collect();
    let open: Vec<usize> = (0..rows).filter(|r| is_open[*r]).collect();
    if !open.is_empty() {
        let exact = exact_head(masked, &hidden.select(Axis(0), &open), operator, layout);
        let sub = Target { logits: target.logits.select(Axis(0), &open), scored: target.scored.as_ref().map(|s| open.iter().map(|r| s[*r]).collect()) };
        for (i, value) in kl_score_only(&sub, &exact).into_iter().enumerate() {
            kl_point[open[i]] = value;
        }
    }
    for r in 0..rows {
        if band[r] > 0.0 && !is_open[r] {
            kl_point[r] -= band[r];
        }
    }
    Ok((kl_point, logits))
}

/// The head's float64 product on `hidden`.
fn exact_head(masked: &Masked, hidden: &Array2<f64>, operator: usize, layout: gam_gpu::banded::Layout) -> Array2<f64> {
    let a = masked.program.operators[operator].matrix_cow();
    match layout {
        gam_gpu::banded::Layout::Transposed => gam_linalg::faer_ndarray::fast_abt(hidden, a.as_ref()),
        gam_gpu::banded::Layout::AsStored => gam_linalg::faer_ndarray::fast_ab(hidden, a.as_ref()),
    }
}

/// A masked forward whose head runs in f32 on the device with its certified band: each row's KL
/// within `band` of its float64 value (`2 maxⱼ` of the head's entry bounds, the KL's gradient in
/// the logits being `q − p`, of `ℓ₁` norm at most 2). The trace stops at the head's hidden node;
/// [`ScreenedPoint::exact`] settles any rows to their float64 KL, and [`ScreenedPoint::mask_gradients`]
/// steers from the screened logits.
pub(crate) struct ScreenedPoint {
    pub kl: Array1<f64>,
    pub band: Array1<f64>,
    trace: Trace,
    logits: Array2<f64>,
    head: (usize, usize, gam_gpu::banded::Layout),
}

/// How a screened point's forward runs its head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadScreen {
    /// No screen: the caller's float64 forward.
    Off,
    /// The f32 product on the device; no screen when the device does not take it.
    Device,
    /// The float64 product carrying the f32 band (tests: every decision path runs, and the
    /// values are the unscreened ones).
    Emulated,
}

/// [`ScreenedPoint`] at `family`, or `None` when the head is not a lone product or the device does not
/// take it.
pub(crate) fn screened_point(masked: &Masked, family: &FamilyInputs, target: &Target, screen: HeadScreen) -> Result<Option<ScreenedPoint>, String> {
    use gam_gpu::banded::Layout;
    let Some(head @ (hidden, operator, layout)) = lone_head(masked).filter(|_| screen != HeadScreen::Off) else {
        return Ok(None);
    };
    let mut body = masked.program.clone();
    body.nodes.truncate(hidden + 1);
    body.output = hidden;
    let trace = body.execute(family, false).map_err(|e| e.to_string())?;
    let h = &trace.values[hidden];
    let (logits, band) = match screen {
        HeadScreen::Device => {
            let op = &masked.program.operators[operator];
            match super::device::banded_product(op, h, layout).map_err(|e| e.to_string())? {
                Some(banded) => (banded.values, banded.band),
                None => return Ok(None),
            }
        }
        HeadScreen::Off => return Ok(None),
        HeadScreen::Emulated => {
            let a = masked.program.operators[operator].matrix_cow();
            let right = match layout {
                Layout::Transposed => a.t(),
                Layout::AsStored => a.view(),
            };
            let band = gam_gpu::precision_bounds::GemmBand::derive(gam_gpu::precision_bounds::DeviceArithmetic::F32, h.view(), right).map_err(|e| format!("{e:?}"))?;
            (exact_head(masked, h, operator, layout), band)
        }
    };
    let band = Array1::from_shape_fn(family.rows, |r| if target.scores(r) { 2.0 * band.row_max(r) } else { 0.0 });
    let kl = kl_score_only(target, &logits);
    Ok(Some(ScreenedPoint { kl, band, trace, logits, head }))
}

impl ScreenedPoint {
    /// The hidden node's value (to settle rows after this trace is gone, [`exact_rows`]).
    pub(crate) fn hidden(&self) -> &Array2<f64> {
        &self.trace.values[self.head.0]
    }

    /// `∂/∂m` of the KL (of row `focus` alone when given) from the screened logits: it only steers.
    pub(crate) fn mask_gradients(&self, masked: &Masked, family: &FamilyInputs, target: &Target, focus: Option<usize>) -> Result<Vec<Array2<f64>>, String> {
        let (hidden, operator, layout) = self.head;
        let mut cotangent = kl(target, &self.logits).1;
        if let Some(row) = focus {
            for (r, mut c) in cotangent.outer_iter_mut().enumerate() {
                if r != row {
                    c.fill(0.0);
                }
            }
        }
        let backward = match layout {
            gam_gpu::banded::Layout::Transposed => gam_gpu::banded::Layout::AsStored,
            gam_gpu::banded::Layout::AsStored => gam_gpu::banded::Layout::Transposed,
        };
        let op = &masked.program.operators[operator];
        let seed = proposing(|| super::device::product(op, &cotangent, backward)).map_err(|e| e.to_string())?;
        let back = proposing(|| super::derivatives::vjp_from(&masked.program, family, &self.trace, hidden, seed, Some(&masked.masked))).map_err(|e| e.to_string())?;
        Ok(mask_gradients_of(masked, &self.trace, &back))
    }
}

/// The float64 KL of `rows` from their hidden values `hidden` (one row each, in order), through
/// the lone head ([`lone_head`]).
pub(crate) fn exact_rows(masked: &Masked, target: &Target, hidden: &Array2<f64>, rows: &[usize]) -> Result<Array1<f64>, String> {
    let (_, operator, layout) = lone_head(masked).ok_or("exact rows: no lone head")?;
    let logits = exact_head(masked, hidden, operator, layout);
    let sub = Target { logits: target.logits.select(Axis(0), rows), scored: target.scored.as_ref().map(|s| rows.iter().map(|r| s[*r]).collect()) };
    Ok(kl_score_only(&sub, &logits))
}

/// The current masks' head as a screened score starts from (module note, "Screened heads"): the
/// hidden rows before the head, the logits, and per row a bound on the logits' error (zero when
/// they came from a float64 forward).
struct HeadBase {
    hidden: Array2<f64>,
    logits: Array2<f64>,
    error: Array1<f64>,
}

/// A screened score: per row the KL and a bound on its logits' error, and the hidden rows it was
/// scored from.
struct Screened {
    kl: Array1<f64>,
    error: Array1<f64>,
    hidden: Array2<f64>,
}

/// Screened heads for the selection's scores (module note, "Screened heads"): the program up to
/// the head's hidden node runs in float64, and the logits are the current ones plus the change of
/// the hidden rows times the head, `z = z_cur + (h − h_cur) A`, that product in f32 on the device
/// with its derived band, which shrinks with the change. A keep/refuse decision is taken from the
/// screened codes when its margin exceeds their bands; otherwise its sequences' rows get the
/// float64 head first, so every decision is the float64 one.
struct Screen {
    body: OperatorProgram,
    hidden: usize,
    operator: usize,
    layout: gam_gpu::banded::Layout,
}

impl Screen {
    /// The screen of `masked`, when its program ends in a lone head right after its hidden node.
    fn new(masked: &Masked) -> Option<Self> {
        let (hidden, operator, layout) = lone_head(masked)?;
        let program = &masked.program;
        let logits = match &program.nodes[program.output] {
            Node::Readout { input, .. } => *input,
            _ => program.output,
        };
        if logits != hidden + 1 || program.output + 1 != program.nodes.len() {
            return None;
        }
        let mut body = program.clone();
        body.nodes.truncate(hidden + 1);
        body.output = hidden;
        Some(Self { body, hidden, operator, layout })
    }

    /// The head of a forward's `trace` as a base, its logits exact.
    fn base(&self, masked: &Masked, trace: &Trace) -> HeadBase {
        let output = masked.program.output;
        HeadBase { hidden: trace.values[self.hidden].clone(), logits: trace.values[output].clone(), error: Array1::zeros(trace.values[output].nrows()) }
    }

    /// `family` scored from `base`; the trace carries the screened logits in the head's nodes. A
    /// `trial`'s trace is only scored, so its sites' `z` are evaluated at their masks' nonzeros.
    fn score(&self, masked: &Masked, family: &FamilyInputs, target: &Target, base: &HeadBase, trial: bool) -> Result<(Screened, Trace), String> {
        let mut trace = if trial {
            self.body.execute_gated(family, &masked.gates())
        } else {
            self.body.execute(family, false)
        }
        .map_err(|e| e.to_string())?;
        let hidden = trace.values[self.hidden].clone();
        let op = &masked.program.operators[self.operator];
        let delta = &hidden - &base.hidden;
        let (logits, error) = match super::device::banded_product(op, &delta, self.layout).map_err(|e| e.to_string())? {
            Some(banded) => {
                let logits = &base.logits + &banded.values;
                // The device product's band, the base's own error, and the addition's rounding.
                let error = Array1::from_shape_fn(hidden.nrows(), |r| {
                    let largest = logits.row(r).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                    (base.error[r] + banded.band.row_max(r) + f64::EPSILON * largest).next_up()
                });
                (logits, error)
            }
            None => (exact_head(masked, &hidden, self.operator, self.layout), Array1::zeros(hidden.nrows())),
        };
        let kl = kl_score_only(target, &logits);
        for _ in self.hidden + 1..masked.program.nodes.len() {
            trace.values.push(logits.clone());
        }
        Ok((Screened { kl, error, hidden }, trace))
    }

    /// The float64 logits and KL of rows `rows` of `hidden`.
    fn exact(&self, masked: &Masked, target: &Target, hidden: &Array2<f64>, rows: &[usize]) -> (Array2<f64>, Array1<f64>) {
        let logits = exact_head(masked, &hidden.select(Axis(0), rows), self.operator, self.layout);
        let sub = Target { logits: target.logits.select(Axis(0), rows), scored: target.scored.as_ref().map(|s| rows.iter().map(|r| s[*r]).collect()) };
        let kl = kl_score_only(&sub, &logits);
        (logits, kl)
    }

    /// Settle rows `rows` of `screened` to float64; with `trace`, its head's nodes too.
    fn settle(&self, masked: &Masked, target: &Target, screened: &mut Screened, rows: &[usize], trace: Option<&mut Trace>) {
        let (logits, kl) = self.exact(masked, target, &screened.hidden, rows);
        for (i, &r) in rows.iter().enumerate() {
            screened.kl[r] = kl[i];
            screened.error[r] = 0.0;
        }
        if let Some(trace) = trace {
            for values in trace.values.iter_mut().skip(self.hidden + 1) {
                for (i, &r) in rows.iter().enumerate() {
                    values.row_mut(r).assign(&logits.row(i));
                }
            }
        }
    }

    /// Settle rows `rows` of the current masks' head to float64, with their KL into `kl_now`.
    fn settle_base(&self, masked: &Masked, target: &Target, base: &mut HeadBase, kl_now: &mut Array1<f64>, rows: &[usize]) {
        let rows: Vec<usize> = rows.iter().copied().filter(|r| base.error[*r] > 0.0).collect();
        if rows.is_empty() {
            return;
        }
        let (logits, kl) = self.exact(masked, target, &base.hidden, &rows);
        for (i, &r) in rows.iter().enumerate() {
            base.logits.row_mut(r).assign(&logits.row(i));
            base.error[r] = 0.0;
            kl_now[r] = kl[i];
        }
    }
}

/// Per sequence, whether a decision over `options` (per option, its saving and the saving's
/// band per sequence) is open: its best option's interval overlaps another's, zero, or
/// `threshold[q]` (a further rule on the best saving).
fn undecided(savings: &[Vec<f64>], bands: &[Vec<f64>], threshold: Option<&[f64]>) -> Vec<bool> {
    let sequences = savings.first().map_or(0, Vec::len);
    (0..sequences)
        .map(|q| {
            let best = (0..savings.len()).max_by(|a, b| savings[*a][q].total_cmp(&savings[*b][q])).unwrap_or(0);
            let (low, high) = (savings[best][q] - bands[best][q], savings[best][q] + bands[best][q]);
            let overlaps = (0..savings.len()).any(|o| o != best && savings[o][q] + bands[o][q] >= low);
            let zero = low <= 0.0 && high > 0.0;
            let rule = threshold.is_some_and(|t| low <= t[q] && high >= t[q]);
            let banded = (0..savings.len()).any(|o| bands[o][q] > 0.0);
            banded && (overlaps || zero || rule)
        })
        .collect()
}

/// [`box_excess_at`] from the masks' own KL `corner`: each layer's vertex, then the adversary. With `all_on`, the CPU trace of every gate on: a vertex's layers before
/// its own are all on, so its forward starts at its layer's first mask, reading the rest there.
fn box_worst(
    masked: &Masked,
    base: &FamilyInputs,
    target: &Target,
    masks: &[Array2<f64>],
    corner: &Array1<f64>,
    all_on: Option<&Trace>,
) -> Result<Array1<f64>, String> {
    let rows = base.rows;
    let sequence_of: Vec<usize> = match &base.layout {
        Some(layout) => layout.sequence.iter().map(|s| *s as usize).collect(),
        None => (0..rows).collect(),
    };
    let sequences = sequence_of.iter().copied().max().map_or(0, |m| m + 1);
    let per_sequence = |values: &Array1<f64>| {
        let mut totals = vec![0.0; sequences];
        for r in 0..rows {
            totals[sequence_of[r]] += values[r];
        }
        totals
    };
    // The masks themselves are a point of the box.
    let mut worst = Array1::<f64>::zeros(rows);
    let mut totals = vec![0.0; sequences];
    // Sites by layer: the name up to its last `.`.
    let mut layers: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (k, site) in masked.sites.iter().enumerate() {
        layers.entry(site.name.rsplit_once('.').map_or_else(|| site.name.clone(), |(layer, _)| layer.to_string())).or_default().push(k);
    }
    // The program up to the head's hidden node, when the head is a lone product: a vertex's
    // logits are then screened in f32 and made exact only for the sequences it may decide.
    let head = lone_head(masked);
    let body = head.map(|(hidden, _, _)| {
        let mut body = masked.program.clone();
        body.nodes.truncate(hidden + 1);
        body.output = hidden;
        body
    });
    if layers.len() > 1 {
        for sites in layers.values() {
            let vertex: Vec<Array2<f64>> =
                masks.iter().enumerate().map(|(k, m)| if sites.contains(&k) { m.clone() } else { Array2::ones(m.dim()) }).collect();
            let family = masked.family(base, &vertex);
            // The layer's first mask node (each site's mask sits just before its `z`).
            let from = sites.iter().map(|k| masked.z[*k] - 1).min().unwrap_or(0);
            // Every other layer's gates are on: its sites run through their library sums.
            let others: Vec<bool> = (0..masked.sites.len()).map(|k| !sites.contains(&k)).collect();
            let kl_vertex = match (all_on, &body, head) {
                (Some(trace), Some(_), Some((hidden, operator, layout))) if from <= hidden => {
                    let program = masked.dense_program(&others)?;
                    let hidden_values = program.execute_suffix_node(&family, trace, from, hidden).map_err(|e| e.to_string())?;
                    certified_point(masked, target, &hidden_values, (operator, layout), corner, &totals, &sequence_of)?.0
                }
                (Some(trace), _, _) if masked.head.is_none() && from <= masked.program.output => {
                    let program = masked.dense_program(&others)?;
                    let output = program.execute_suffix_node(&family, trace, from, masked.program.output).map_err(|e| e.to_string())?;
                    kl_score_only(target, &output)
                }
                _ => score_only(masked, &family, target)?,
            };
            let excess = &kl_vertex - corner;
            let vertex_totals = per_sequence(&excess);
            for r in 0..rows {
                let q = sequence_of[r];
                if vertex_totals[q] > totals[q] {
                    worst[r] = excess[r];
                }
            }
            for q in 0..sequences {
                totals[q] = totals[q].max(vertex_totals[q]);
            }
        }
    }
    // An adversary inside the box ([`super::adversary::adversary`]): sign ascent over every off gate
    // together, and over each layer's off gates alone (the rest at the masks), from the masks, every
    // gate's top and its middle; each input keeps the largest KL any point gave it, so a word is
    // charged its own worst case.
    const ADVERSARY_STEPS: usize = 4;
    const ADVERSARY_STARTS: usize = 3;
    let claim = super::adversary::Gates::claim(masks);
    let mut boxes = vec![claim.clone()];
    if layers.len() > 1 {
        for sites in layers.values() {
            let mut only = claim.clone();
            for k in 0..masks.len() {
                if !sites.contains(&k) {
                    only.upper[k] = only.lower[k].clone();
                }
            }
            boxes.push(only);
        }
    }
    let seeds: Vec<u64> = (0..boxes.len()).map(|i| 0xAD5E + i as u64).collect();
    for kl_point in timed("adversary", || super::adversary::adversary_batch(masked, base, target, &boxes, ADVERSARY_STEPS, ADVERSARY_STARTS, &seeds))? {
        let excess = &kl_point - corner;
        for r in 0..rows {
            worst[r] = worst[r].max(excess[r]);
        }
    }
    Ok(worst)
}

/// Our box claim's cost per input (module note, "Claims"): at every site, from its reads in the
/// masked forward `trace`, the off blocks' outputs `Z_c = U_c z_c` bounded together in the site's
/// written Fisher `fishers[k]`, `½ (Σ_off ‖Z_c‖_F)²` nats, an upper bound on the quadratic response
/// of the off gates in `[0, 1]` there (the triangle inequality), summed over sites. Local to each
/// site and second order, it is our claim's cost, not a bound on the end-to-end KL, which the
/// attack measures ([`box_excess_at`]). A refinement of the library cannot lower it: splitting an
/// off block leaves the sum of its parts' norms no smaller.
pub fn box_upper(masked: &Masked, trace: &Trace, masks: &[Array2<f64>], fishers: &[Array2<f64>]) -> Result<Array1<f64>, String> {
    let amplitudes: Vec<_> = masked.z.iter().map(|n| &trace.values[*n]).collect();
    Ok(box_contributions(masked, &amplitudes, masks, fishers, false)?.0)
}

/// Direct derivatives of the contribution charge, before propagating through the masked model.
pub(crate) struct ContributionDerivatives {
    pub amplitudes: Vec<Array2<f64>>,
    pub writes: Vec<Array2<f64>>,
}

/// The same contribution charge on host or device amplitudes. Fishers and binary masks are fixed.
/// For each off block, `r = ‖z U‖_F` and `N = Σ r`: its derivatives are `N z(UFUᵀ)/r`
/// in `z` and `zᵀ(N/r)z UF` in `U`. At a zero norm we choose the zero subgradient.
/// Rank-one blocks take linear work in their amplitudes; no written-width value per row is formed.
pub(crate) fn box_contributions(
    masked: &Masked, amplitudes: &[&Array2<f64>], masks: &[Array2<f64>],
    fishers: &[Array2<f64>], derivatives: bool,
) -> Result<(Array1<f64>, Option<ContributionDerivatives>), String> {
    let count = masked.sites.len();
    if amplitudes.len() != count || masks.len() != count || fishers.len() != count {
        return Err("box contribution: one amplitude, mask and Fisher per site required".to_string());
    }
    let rows = masks.first().map_or(0, |m| m.nrows());
    for k in 0..count {
        if amplitudes[k].dim() != (rows, masked.pieces(k)) || masks[k].dim() != (rows, masked.blocks(k)) {
            return Err(format!("box contribution: incompatible amplitudes or masks at site {k}"));
        }
        if masks[k].iter().any(|m| *m != 0.0 && *m != 1.0) {
            return Err("box contribution: masks must be binary".to_string());
        }
        let width = masked.write_offsets[k].last().copied().unwrap_or(0);
        if fishers[k].dim() != (width, width) {
            return Err(format!("box contribution: incompatible Fisher at site {k}"));
        }
    }
    let terms = masked.box_terms(fishers)?;
    let mut cost = Array1::<f64>::zeros(rows);
    let mut differentiated = ContributionDerivatives { amplitudes: Vec::new(), writes: Vec::new() };
    for k in 0..masked.sites.len() {
        let z = amplitudes[k];
        let mask = &masks[k];
        let mut norms = Array1::<f64>::zeros(rows);
        if masked.is_rank_one(k) {
            let own = terms.sites[k][0].row(0);
            for r in 0..rows {
                norms[r] = (0..mask.ncols()).filter(|&c| mask[[r, c]] <= 0.0).map(|c| z[[r, c]].abs() * own[c].max(0.0).sqrt()).sum();
            }
        } else {
            let mut start = 0;
            for (b, &width) in masked.ranks(k).iter().enumerate() {
                let metric = &terms.sites[k][b];
                for r in 0..rows {
                    if mask[[r, b]] <= 0.0 {
                        let zb = z.slice(s![r, start..start + width]);
                        norms[r] += zb.dot(&metric.dot(&zb)).max(0.0).sqrt();
                    }
                }
                start += width;
            }
        }
        cost += &(&norms * &norms * 0.5);
        if derivatives {
            let uf = gam_linalg::faer_ndarray::fast_ab(&masked.u(k)?, &fishers[k]);
            let mut dz = Array2::<f64>::zeros(z.dim());
            let mut du = Array2::<f64>::zeros(uf.dim());
            if masked.is_rank_one(k) {
                let own = terms.sites[k][0].row(0);
                for c in 0..mask.ncols() {
                    let size = own[c].max(0.0).sqrt();
                    if size == 0.0 { continue; }
                    let mut scale = 0.0;
                    for r in 0..rows {
                        if mask[[r, c]] == 0.0 && z[[r, c]] != 0.0 {
                            dz[[r, c]] = norms[r] * z[[r, c]].signum() * size;
                            scale += norms[r] * z[[r, c]].abs() / size;
                        }
                    }
                    du.row_mut(c).assign(&(&uf.row(c) * scale));
                }
            } else {
                let mut start = 0;
                for (b, &width) in masked.ranks(k).iter().enumerate() {
                    let metric = &terms.sites[k][b];
                    let zb = z.slice(s![.., start..start + width]);
                    let mut weighted = Array2::<f64>::zeros(zb.dim());
                    for r in 0..rows {
                        if mask[[r, b]] != 0.0 { continue; }
                        let mz = metric.dot(&zb.row(r));
                        let size = zb.row(r).dot(&mz).max(0.0).sqrt();
                        if size == 0.0 { continue; }
                        let scale = norms[r] / size;
                        dz.slice_mut(s![r, start..start + width]).assign(&(&mz * scale));
                        weighted.row_mut(r).assign(&(&zb.row(r) * scale));
                    }
                    let direct = zb.t().dot(&weighted).dot(&uf.slice(s![start..start + width, ..]));
                    du.slice_mut(s![start..start + width, ..]).assign(&direct);
                    start += width;
                }
            }
            differentiated.amplitudes.push(dz);
            differentiated.writes.push(du);
        }
    }
    Ok((cost, derivatives.then_some(differentiated)))
}

/// Gradient of the masks' KL plus [`box_upper`], with the written Fishers held fixed for a step.
/// Later sites' charge depends on earlier sites' weights through their actual masked reads. All
/// amplitude derivatives and the output's KL derivative therefore seed one joint reverse pass;
/// differentiating each site in isolation would omit those terms. Returns the contribution cost
/// alone and the combined gradients, `(contribution_cost, [(dV, dU)])`.
pub fn box_gradients(
    masked: &Masked, family: &FamilyInputs, trace: &Trace, masks: &[Array2<f64>],
    cotangent: Array2<f64>, fishers: &[Array2<f64>],
) -> Result<(Array1<f64>, BoxGradients), String> {
    let amplitudes: Vec<_> = masked.z.iter().map(|n| &trace.values[*n]).collect();
    let (cost, direct) = box_contributions(masked, &amplitudes, masks, fishers, true)?;
    let direct = direct.ok_or("box contribution: missing derivatives")?;
    let mut seeds = BTreeMap::from([(masked.program.output, cotangent)]);
    for (&node, dz) in masked.z.iter().zip(direct.amplitudes) {
        match seeds.get_mut(&node) { Some(g) => *g += &dz, None => { seeds.insert(node, dz); } }
    }
    let mut keep = masked.z.clone();
    keep.extend(masked.written.iter().flatten().copied());
    let back = super::derivatives::vjp_seeded(&masked.program, family, trace, seeds, Some(&keep)).map_err(|e| e.to_string())?;
    let mut gradients = Vec::new();
    for (k, direct_u) in direct.writes.into_iter().enumerate() {
        let reads = read_values(trace, &masked.sites[k])?;
        let dv = match &back[masked.z[k]] {
            Some(g) => fast_atb(g, &reads),
            None => Array2::zeros((masked.pieces(k), reads.ncols())),
        };
        let parts: Vec<_> = masked.written[k].iter().map(|n| back[*n].clone().unwrap_or_else(|| Array2::zeros(trace.values[*n].dim()))).collect();
        let written = ndarray::concatenate(Axis(1), &parts.iter().map(|p| p.view()).collect::<Vec<_>>()).map_err(|e| e.to_string())?;
        let du = fast_atb(&trace.values[masked.masked[k]], &written) + direct_u;
        gradients.push((dv, du));
    }
    Ok((cost, gradients))
}

/// [`box_upper`] at `masks`, from the masked forward of `base` with them.
pub fn box_upper_at(masked: &Masked, base: &FamilyInputs, masks: &[Array2<f64>], fishers: &[Array2<f64>]) -> Result<Array1<f64>, String> {
    let family = masked.family(base, masks);
    if let Some(cost) = masked.on_device(|accelerated| {
        let trace = accelerated.program().forward(&family)?;
        accelerated.box_upper(masked, &trace, masks, fishers)
    })? { return Ok(cost); }
    let trace = masked.program.execute(&family, false).map_err(|e| e.to_string())?;
    box_upper(masked, &trace, masks, fishers)
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

/// Gradients only for the factor side this alternating corner step will update. Empty arrays
/// occupy the unused entries, so no mask-gradient or opposite-factor work is performed.
fn piece_gradients(masked: &Masked, family: &FamilyInputs, trace: &Trace, masks: &[Array2<f64>], cotangent: Array2<f64>, moves_u: bool) -> Result<Vec<(Array2<f64>, Array2<f64>, Array2<f64>)>, String> {
    proposing(|| {
        let keep: Vec<usize> = if moves_u { masked.written.iter().flatten().copied().collect() } else { masked.masked.clone() };
        let back = super::derivatives::vjp_from(&masked.program, family, trace, masked.program.output, cotangent, Some(&keep)).map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        for (k, site) in masked.sites.iter().enumerate() {
            let gradient = if moves_u {
                let parts: Vec<_> = masked.written[k].iter().map(|n| back[*n].clone().unwrap_or_else(|| Array2::zeros(trace.values[*n].dim()))).collect();
                let written = ndarray::concatenate(Axis(1), &parts.iter().map(|p| p.view()).collect::<Vec<_>>()).map_err(|e| e.to_string())?;
                product_atb(&trace.values[masked.masked[k]], &written).map_err(|e| e.to_string())?
            } else if let Some(c) = &back[masked.masked[k]] {
                let cot_z = c * &masked.expand(k, &masks[k]);
                product_atb(&cot_z, &read_values(trace, site)?).map_err(|e| e.to_string())?
            } else {
                Array2::zeros((masked.pieces(k), site.reads.iter().map(|r| trace.values[*r].ncols()).sum()))
            };
            let empty = || Array2::zeros((0, 0));
            out.push(if moves_u { (empty(), empty(), gradient) } else { (empty(), gradient, empty()) });
        }
        Ok(out)
    })
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
    Ok(mask_gradients_of(masked, trace, &back))
}

/// [`mask_gradients`] from a reverse pass `back` that kept every site's masked node.
fn mask_gradients_of(masked: &Masked, trace: &Trace, back: &[Option<Array2<f64>>]) -> Vec<Array2<f64>> {
    (0..masked.sites.len())
        .map(|k| {
            let z = &trace.values[masked.z[k]];
            back[masked.masked[k]].as_ref().map_or_else(|| Array2::zeros((z.nrows(), masked.blocks(k))), |c| masked.to_blocks(k, &(c * z)))
        })
        .collect()
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
    fisher_impl(masked, family, trace, target, samples, seed, written, true)
}

fn step_fisher(masked: &Masked, family: &FamilyInputs, trace: &Trace, target: &Target, samples: usize, seed: u64) -> Result<Vec<(Array2<f64>, Option<Array2<f64>>)>, String> {
    fisher_impl(masked, family, trace, target, samples, seed, true, false)
}

fn fisher_impl(masked: &Masked, family: &FamilyInputs, trace: &Trace, target: &Target, samples: usize, seed: u64, written: bool, masks: bool) -> Result<Vec<(Array2<f64>, Option<Array2<f64>>)>, String> {
    if samples == 0 { return Err("Fisher needs at least one sample".to_string()); }
    let logits = logits(masked, family, trace, target)?;
    let mut keep = if masks { masked.masked.clone() } else { Vec::new() };
    if written { keep.extend(masked.written.iter().flatten().copied()); }
    let head = CachedSampledHead::new(masked, &logits, target, &keep)?;
    let mut rng = XorShift(seed | 1);
    let mut out: Vec<(Array2<f64>, Option<Array2<f64>>)> = masked
        .sites
        .iter()
        .enumerate()
        .map(|(k, _)| {
            let d_out: usize = masked.written[k].iter().map(|n| trace.values[*n].ncols()).sum();
            let shape = if masks { (trace.values[masked.z[k]].nrows(), masked.blocks(k)) } else { (0, 0) };
            (Array2::zeros(shape), written.then(|| Array2::zeros((d_out, d_out))))
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
            if masks && let Some(c) = &back[masked.masked[k]] {
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

/// Wall seconds per phase of the selection since the last [`phases`] (forwards, gradients,
/// Fishers, the box claim's parts), for the round's log line.
static PHASES: Mutex<BTreeMap<&'static str, f64>> = Mutex::new(BTreeMap::new());

/// `body`, its wall time added to phase `name` ([`phases`]).
pub(crate) fn timed<T>(name: &'static str, body: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let out = body();
    if let Ok(mut phases) = PHASES.lock() {
        *phases.entry(name).or_insert(0.0) += started.elapsed().as_secs_f64();
    }
    out
}

/// The phases' wall times since the last call, as `name s, …` (nested phases overlap).
fn phases() -> String {
    let taken = PHASES.lock().map(|mut p| std::mem::take(&mut *p)).unwrap_or_default();
    taken.iter().map(|(name, seconds)| format!("{name} {seconds:.1}s")).collect::<Vec<_>>().join(", ")
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
        timed("forward", || Self::forward_untimed(masked, family, target, on_device, cotangent))
    }

    fn forward_untimed(masked: &Masked, family: &FamilyInputs, target: &Target, on_device: Option<&DeviceTarget>, cotangent: bool) -> Result<(Array1<f64>, Self), String> {
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
        timed("mask gradients", || self.mask_gradients_untimed(masked, family, target, on_device))
    }

    fn mask_gradients_untimed(&mut self, masked: &Masked, family: &FamilyInputs, target: &Target, on_device: Option<&DeviceTarget>) -> Result<Vec<Array2<f64>>, String> {
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
        timed("fisher", || self.fisher_untimed(masked, family, target, on_device, samples, seed))
    }

    fn fisher_untimed(
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
    select_observed(masked, base, target, masks, coder, observations, samples, &mut |_: &Round<'_>| Ok(()))
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

/// [`select`], showing `observe` every round's decisions before they are taken.
pub fn select_observed(
    masked: &Masked,
    base: &FamilyInputs,
    target: &Target,
    masks: Vec<Array2<f64>>,
    coder: &Coder,
    observations: f64,
    samples: usize,
    observe: &mut dyn FnMut(&Round<'_>) -> Result<(), String>,
) -> Result<(Vec<Array2<f64>>, Array1<f64>), String> {
    select_resumable(masked, base, target, masks, None, coder, observations, samples, observe, &mut |_: &Progress<'_>| Ok(()))
}

/// A selection's whole state between two rounds, as [`select_resumable`] shows it to its
/// checkpoint: the masks, each input's interaction `α`, each sequence's flip cap and the rounds
/// taken. Everything else a
/// selection holds between rounds is a function of these and of its start (the curvature is
/// measured once, at the start's masks), so a selection resumed from them continues exactly as the
/// uninterrupted one.
pub struct Progress<'a> {
    pub masks: &'a [Array2<f64>],
    pub alpha: &'a [f64],
    pub cap: &'a [usize],
    pub round: u64,
}

/// An owned [`Progress`], to resume from.
#[derive(Clone, Debug, PartialEq)]
pub struct Resume {
    pub masks: Vec<Array2<f64>>,
    pub alpha: Vec<f64>,
    pub cap: Vec<usize>,
    pub round: u64,
}

impl Progress<'_> {
    /// The owned state.
    pub fn to_resume(&self) -> Resume {
        Resume { masks: self.masks.to_vec(), alpha: self.alpha.to_vec(), cap: self.cap.to_vec(), round: self.round }
    }
}

/// [`select_observed`] from `start`, or, with `resume`, continued from a state a selection from
/// the same `start` showed its `checkpoint`; `checkpoint` sees the state before every round.
/// A resumed selection takes exactly the decisions the uninterrupted one takes from that round on.
pub fn select_resumable(
    masked: &Masked,
    base: &FamilyInputs,
    target: &Target,
    start: Vec<Array2<f64>>,
    resume: Option<Resume>,
    coder: &Coder,
    observations: f64,
    samples: usize,
    observe: &mut dyn FnMut(&Round<'_>) -> Result<(), String>,
    checkpoint: &mut dyn FnMut(&Progress<'_>) -> Result<(), String>,
) -> Result<(Vec<Array2<f64>>, Array1<f64>), String> {
    let mut masks = start;
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
    // What the next round may reuse exactly instead of recomputing: when nothing was kept, the
    // round's forward and gradients (the state did not move). A kept proposal's forward is not
    // carried over: a screened one holds f32-banded logits, and every round starting from a
    // float64 forward of its masks is what lets a resumed selection match an uninterrupted one.
    let mut reuse: Option<(Array1<f64>, Vec<Array2<f64>>)> = None;
    // The target on the program's device, when it runs on one (module note, "Devices").
    let on_device = masked.on_device(|accelerated| accelerated.target(target))?;
    let on_device = on_device.as_ref();
    if let Some(resume) = resume {
        if resume.masks.len() != masks.len()
            || resume.masks.iter().zip(&masks).any(|(a, b)| a.dim() != b.dim())
            || resume.alpha.len() != rows
            || resume.cap.len() != sequences
        {
            return Err("selection: a resumed state that does not fit this selection".to_string());
        }
        // The curvature as the uninterrupted selection measured it, on its first round, at the
        // start's masks.
        let family = masked.family(base, &masks);
        let (_, state) = Selected::forward(masked, &family, target, on_device, true)?;
        curvature = Some(state.fisher(masked, &family, target, on_device, samples, 0x5EED)?);
        drop(state);
        masks = resume.masks;
        alpha = resume.alpha;
        cap = resume.cap;
        round = resume.round;
    }
    // Scores on the CPU through screened heads (module note, "Screened heads"): the current masks'
    // head, and the errors a proposal kept whole hands to the next round.
    let screen = if on_device.is_none() { Screen::new(masked) } else { None };
    let mut head_base: Option<HeadBase> = None;
    // Per sequence, its scored rows (a kept round saving less than a bit per one is the last).
    let mut sequence_rows = vec![0usize; sequences];
    for r in 0..rows {
        sequence_rows[sequence_of[r]] += usize::from(scores[r]);
    }
    let threshold: Vec<f64> = sequence_rows.iter().map(|n| *n as f64).collect();
    // Per sequence, `n/ln 2` times the sum of its scored rows' KL bands (twice the logits' error).
    let bands_of = |error: &Array1<f64>, other: Option<&Array1<f64>>| -> Vec<f64> {
        let mut out = vec![0.0; sequences];
        for r in (0..rows).filter(|r| scores[*r]) {
            out[sequence_of[r]] += scale * 2.0 * (error[r] + other.map_or(0.0, |o| o[r]));
        }
        out
    };
    // Per sequence, the saving of each option's codes over `before`.
    let savings_of = |before: &Array1<f64>, afters: &[&Array1<f64>]| -> Vec<Vec<f64>> {
        afters
            .iter()
            .map(|after| {
                let mut out = vec![0.0; sequences];
                for r in 0..rows {
                    out[sequence_of[r]] += before[r] - after[r];
                }
                out
            })
            .collect()
    };
    // A trial's exact KL.
    let score = |trial: &[Array2<f64>]| score_only(masked, &masked.family(base, trial), target);
    loop {
        checkpoint(&Progress { masks: &masks, alpha: &alpha, cap: &cap, round })?;
        let family = masked.family(base, &masks);
        // The current forward lives only until its gradients (and, once, the Fisher) are read.
        let (mut kl_now, grads) = match reuse.take() {
            Some(state) => state,
            None => {
                let (kl_now, mut state) = Selected::forward(masked, &family, target, on_device, true)?;
                if let (Some(screen), Selected::Host(trace, _)) = (&screen, &state) {
                    head_base = Some(screen.base(masked, trace));
                }
                let grads = state.mask_gradients(masked, &family, target, on_device)?;
                // The Fisher diagonal only ranks proposals (the exact forward decides), so it is
                // measured on the first round and kept: it moves slowly with the masks, and its
                // passes dominate a round's cost.
                if curvature.is_none() {
                    curvature = Some(state.fisher(masked, &family, target, on_device, samples, 0x5EED + round)?);
                }
                (kl_now, grads)
            }
        };
        let curvature = curvature.as_ref().ok_or("no curvature")?;
        let listing_now = coder.bits(&masks);
        let mut before = code(&kl_now, &listing_now, observations);
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
                            // The flip's change, `∓g + h/2`.
                            let kl_change = if on { -g[c] + 0.5 * h[c] } else { g[c] + 0.5 * h[c] };
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
            // Group moves: no single flip pays, but many together may (from an empty start one
            // subcomponent rarely pays for itself alone). Each input's flips are ranked by their
            // predicted KL change alone; for k = 1, 2, 4, … every input takes its k best, the
            // exact code is measured, and each sequence keeps the k that lowers its code most.
            let ranked: Vec<Vec<(usize, usize)>> = {
                use rayon::prelude::*;
                (0..rows)
                    .into_par_iter()
                    .map(|r| {
                        if !scores[r] {
                            return Vec::new();
                        }
                        let mut moves: Vec<(f64, usize, usize)> = Vec::new();
                        for (k, (g, (h, _))) in grads.iter().zip(curvature.iter()).enumerate() {
                            let (g, h, m) = (g.row(r), h.row(r), masks[k].row(r));
                            for c in 0..g.len() {
                                let kl_change = if m[c] > 0.0 { -g[c] + 0.5 * h[c] } else { g[c] + 0.5 * h[c] };
                                if kl_change < 0.0 {
                                    moves.push((kl_change, k, c));
                                }
                            }
                        }
                        moves.sort_by(|a, b| a.0.total_cmp(&b.0));
                        moves.into_iter().map(|(_, k, c)| (k, c)).collect()
                    })
                    .collect()
            };
            let longest = ranked.iter().map(Vec::len).max().unwrap_or(0);
            // Every k's codes (with its screened score), then per sequence the best saving and its k.
            let mut options: Vec<(usize, Array1<f64>, Array1<f64>, Option<Screened>)> = Vec::new();
            let mut k = 1;
            while k <= longest {
                let mut trial = masks.clone();
                for (r, moves) in ranked.iter().enumerate() {
                    for &(site, c) in moves.iter().take(k) {
                        trial[site][[r, c]] = 1.0 - trial[site][[r, c]];
                    }
                }
                let listing_trial = coder.bits(&trial);
                let (after_trial, screened) = match (&screen, &head_base) {
                    (Some(screen), Some(head)) => {
                        let (screened, _) = screen.score(masked, &masked.family(base, &trial), target, head, true)?;
                        (code(&screened.kl, &listing_trial, observations), Some(screened))
                    }
                    _ => (code(&score(&trial)?, &listing_trial, observations), None),
                };
                options.push((k, after_trial, listing_trial, screened));
                k *= 2;
            }
            // Screened codes decide where their bands allow; elsewhere the sequence's rows are
            // settled to float64 in every option and in the current masks first. With no option
            // (no input has a move) there is nothing to decide.
            if let (Some(screen), Some(head)) = (&screen, head_base.as_mut())
                && !options.is_empty()
            {
                let afters: Vec<&Array1<f64>> = options.iter().map(|(_, after, _, _)| after).collect();
                let savings = savings_of(&before, &afters);
                let bands: Vec<Vec<f64>> = options.iter().map(|(_, _, _, s)| bands_of(&head.error, s.as_ref().map(|s| &s.error))).collect();
                let open = undecided(&savings, &bands, None);
                let settle: Vec<usize> = (0..rows).filter(|r| open[sequence_of[*r]]).collect();
                if !settle.is_empty() {
                    screen.settle_base(masked, target, head, &mut kl_now, &settle);
                    for &r in &settle {
                        before[r] = listing_now[r] + scale * kl_now[r];
                    }
                    for (_, after, listing, screened) in options.iter_mut() {
                        if let Some(screened) = screened.as_mut() {
                            screen.settle(masked, target, screened, &settle, None);
                            for &r in &settle {
                                after[r] = listing[r] + scale * screened.kl[r];
                            }
                        }
                    }
                }
            }
            let mut before_sequence = vec![0.0; sequences];
            for r in 0..rows {
                before_sequence[sequence_of[r]] += before[r];
            }
            // Per sequence, its k by (screened) saving, best first.
            let mut ranked_k: Vec<Vec<(f64, usize)>> = vec![Vec::new(); sequences];
            for (k, after_trial, _, _) in &options {
                let mut after_sequence = vec![0.0; sequences];
                for r in 0..rows {
                    after_sequence[sequence_of[r]] += after_trial[r];
                }
                for q in 0..sequences {
                    let saving = before_sequence[q] - after_sequence[q];
                    if saving > 0.0 {
                        ranked_k[q].push((saving, *k));
                    }
                }
            }
            ranked_k.iter_mut().for_each(|r| r.sort_by(|a, b| b.0.total_cmp(&a.0)));
            let best: Vec<(f64, usize)> = ranked_k.iter().map(|r| r.first().copied().unwrap_or((0.0, 0))).collect();
            if best.iter().any(|(saving, _)| *saving > 0.0) {
                for (r, moves) in ranked.iter().enumerate() {
                    let (saving, k) = best[sequence_of[r]];
                    if saving > 0.0 {
                        for &(site, c) in moves.iter().take(k) {
                            masks[site][[r, c]] = 1.0 - masks[site][[r, c]];
                        }
                    }
                }
                log::info!(
                    "group moves ({:.0}s): {} sequences moved, {:.0} bits saved, k {:?}",
                    started.elapsed().as_secs_f64(),
                    best.iter().filter(|(saving, _)| *saving > 0.0).count(),
                    best.iter().map(|(saving, _)| saving).sum::<f64>(),
                    best.iter().map(|(_, k)| *k).collect::<Vec<_>>()
                );
                cap = vec![usize::MAX; sequences];
                reuse = None;
                continue;
            }
            // The KL returned is the float64 one of the final masks, never a screened one.
            let kl_final = score_only(masked, &masked.family(base, &masks), target)?;
            return Ok((masks, kl_final));
        }
        let proposed_family = masked.family(base, &proposed);
        let (mut kl_new, mut state_new, mut screened) = match (&screen, &head_base) {
            (Some(screen), Some(head)) => {
                let (screened, trace) = screen.score(masked, &proposed_family, target, head, false)?;
                (screened.kl.clone(), Selected::Host(trace, None), Some(screened))
            }
            _ => {
                let (kl, state) = Selected::forward(masked, &proposed_family, target, on_device, false)?;
                (kl, state, None)
            }
        };
        let listing_new = coder.bits(&proposed);
        let mut after = code(&kl_new, &listing_new, observations);
        // Screened codes decide where their bands allow; elsewhere the sequence's rows are settled
        // to float64 in the proposal and in the current masks first.
        if let (Some(screen), Some(screened), Some(head)) = (&screen, screened.as_mut(), head_base.as_mut()) {
            let open = undecided(&savings_of(&before, &[&after]), &[bands_of(&head.error, Some(&screened.error))], Some(&threshold));
            let settle: Vec<usize> = (0..rows).filter(|r| open[sequence_of[*r]]).collect();
            log::info!("screened proposal: {} of {sequences} sequences settled to float64", open.iter().filter(|o| **o).count());
            if !settle.is_empty() {
                let trace = match &mut state_new {
                    Selected::Host(trace, _) => Some(trace),
                    Selected::Device(_) => None,
                };
                screen.settle(masked, target, screened, &settle, trace);
                screen.settle_base(masked, target, head, &mut kl_now, &settle);
                for &r in &settle {
                    kl_new[r] = screened.kl[r];
                    before[r] = listing_now[r] + scale * kl_now[r];
                    after[r] = listing_new[r] + scale * kl_new[r];
                }
            }
        }
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
        for r in 0..rows {
            sequence_saving[sequence_of[r]] += before[r] - after[r];
            sequence_flips[sequence_of[r]] = sequence_flips[sequence_of[r]].max(flipped[r]);
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
        // A refused sequence's proposal is split: each input's flips, in their order of predicted
        // saving, halve into the first half and the rest; both halves are tried exactly (every
        // refused sequence at once), a sequence keeps the half that lowers its code most, and one
        // that neither lowers is split again from its first half, down to one flip per input.
        let mut open: Vec<bool> = (0..sequences).map(|q| sequence_flips[q] > 0 && sequence_saving[q] <= 0.0).collect();
        let mut subset: Vec<Vec<(usize, usize)>> = picks.iter().map(|(pick, _)| pick.clone()).collect();
        let (mut rescued, mut rescued_saving) = (0usize, 0.0);
        let code_at = |trial: &[Array2<f64>], head: Option<&HeadBase>| -> Result<(Array1<f64>, Array1<f64>, Option<Screened>), String> {
            let listing = coder.bits(trial);
            match (&screen, head) {
                (Some(screen), Some(head)) => {
                    let (screened, _) = screen.score(masked, &masked.family(base, trial), target, head, true)?;
                    Ok((code(&screened.kl, &listing, observations), listing, Some(screened)))
                }
                _ => Ok((code(&score(trial)?, &listing, observations), listing, None)),
            }
        };
        while (0..rows).any(|r| open[sequence_of[r]] && subset[r].len() >= 2) {
            let halves: Vec<(Vec<(usize, usize)>, Vec<(usize, usize)>)> = subset
                .iter()
                .enumerate()
                .map(|(r, flips)| {
                    if !open[sequence_of[r]] {
                        return (Vec::new(), Vec::new());
                    }
                    let middle = flips.len().div_ceil(2);
                    (flips[..middle].to_vec(), flips[middle..].to_vec())
                })
                .collect();
            let mut sides = Vec::new();
            for side in 0..2 {
                let mut trial = masks.clone();
                for (r, (first, rest)) in halves.iter().enumerate() {
                    for &(site, c) in if side == 0 { first } else { rest } {
                        trial[site][[r, c]] = 1.0 - trial[site][[r, c]];
                    }
                }
                sides.push(code_at(&trial, head_base.as_ref())?);
            }
            // Screened codes decide where their bands allow; elsewhere the sequence's rows are
            // settled to float64 in both halves and in the current masks first.
            if let (Some(screen), Some(head)) = (&screen, head_base.as_mut()) {
                let afters: Vec<&Array1<f64>> = sides.iter().map(|(after, _, _)| after).collect();
                let bands: Vec<Vec<f64>> = sides.iter().map(|(_, _, s)| bands_of(&head.error, s.as_ref().map(|s| &s.error))).collect();
                let undecided_now = undecided(&savings_of(&before, &afters), &bands, Some(&threshold));
                let settle: Vec<usize> = (0..rows).filter(|r| open[sequence_of[*r]] && undecided_now[sequence_of[*r]]).collect();
                if !settle.is_empty() {
                    screen.settle_base(masked, target, head, &mut kl_now, &settle);
                    for &r in &settle {
                        before[r] = listing_now[r] + scale * kl_now[r];
                    }
                    for (after, listing, screened) in sides.iter_mut() {
                        if let Some(screened) = screened.as_mut() {
                            screen.settle(masked, target, screened, &settle, None);
                            for &r in &settle {
                                after[r] = listing[r] + scale * screened.kl[r];
                            }
                        }
                    }
                }
            }
            let mut savings: Vec<[f64; 2]> = vec![[0.0, 0.0]; sequences];
            for (side, (after_half, _, _)) in sides.iter().enumerate() {
                for r in 0..rows {
                    savings[sequence_of[r]][side] += before[r] - after_half[r];
                }
            }
            // The half each sequence would keep.
            let chosen: Vec<Option<usize>> = (0..sequences)
                .map(|q| {
                    let side = if savings[q][0] >= savings[q][1] { 0 } else { 1 };
                    (open[q] && savings[q][side] > 0.0).then_some(side)
                })
                .collect();
            for q in 0..sequences {
                let Some(side) = chosen[q] else { continue };
                {
                    for (r, (first, rest)) in halves.iter().enumerate() {
                        if sequence_of[r] == q {
                            for &(site, c) in if side == 0 { first } else { rest } {
                                masks[site][[r, c]] = 1.0 - masks[site][[r, c]];
                            }
                        }
                    }
                    open[q] = false;
                    rescued += 1;
                    rescued_saving += savings[q][side];
                    sequence_saving[q] = savings[q][side];
                    cap[q] = if savings[q][side] < sequence_rows[q] as f64 { 0 } else { usize::MAX };
                }
            }
            for (r, (first, _)) in halves.into_iter().enumerate() {
                if open[sequence_of[r]] {
                    subset[r] = first;
                }
            }
        }
        kept += rescued;
        saved += rescued_saving;
        if rescued > 0 {
            log::info!("selection round {}: split proposals kept in {rescued} more sequences, {rescued_saving:.0} bits saved", round + 1);
        }
        round += 1;
        if rescued == 0 && kept == 0 {
            // Nothing moved: this round's forward and gradients still describe the masks.
            reuse = Some((kl_now.clone(), grads));
        }
        let per_scored = |code: &Array1<f64>| (0..rows).filter(|r| target.scores(*r)).map(|r| code[r]).sum::<f64>() / scored.max(1) as f64;
        log::info!(
            "selection round {round} ({:.0}s): {} inputs flipped {} entries, {kept} of {tried} sequences kept, {saved:.0} bits saved; code {:.1} -> {:.1} bits per input [{}]",
            started.elapsed().as_secs_f64(),
            flipped.iter().filter(|f| **f > 0).count(),
            flipped.iter().sum::<usize>(),
            per_scored(&before),
            per_scored(&after),
            phases()
        );
        // Done when every sequence is done (each on its own evidence, so a batch of sequences
        // selects exactly as each would alone).
        if cap.iter().all(|c| *c == 0) {
            // The KL returned is the float64 one of the final masks, from one forward of them alone:
            // never a screened one, nor rows settled on a subset, so it is what any later scoring
            // of the same masks finds.
            let kl_final = score_only(masked, &masked.family(base, &masks), target)?;
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
pub(super) fn keep_sum(g: &Array2<f64>, other: &Array2<f64>) -> Result<Array2<f64>, String> {
    if other.nrows() <= other.ncols() {
        // otherᵀ other and other otherᵀ have the same nonzero eigenvalues. Work in piece
        // space when it is smaller, projecting onto the same resolved column space without
        // allocating a layer-width Gram matrix or its pseudoinverse.
        let mut gram = gam_linalg::faer_ndarray::fast_abt(other, other);
        for i in 0..gram.nrows() {
            for j in i + 1..gram.ncols() {
                let value = 0.5 * (gram[[i, j]] + gram[[j, i]]);
                gram[[i, j]] = value;
                gram[[j, i]] = value;
            }
        }
        let spectrum = super::dense::eigh(gram.view(), gam_linalg::roundoff::SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
        let band = gam_linalg::roundoff::symmetric_spectrum_rounding_band_at_dim(other.ncols(), &spectrum.values.to_vec());
        // Project directly onto the unresolved/null eigenspace. Subtracting the resolved
        // projection from g leaves roundoff even when that space spans every piece; a line
        // search can amplify that residual into a move that changes the native map.
        let mut coefficients = fast_atb(&spectrum.vectors, g);
        for (k, value) in spectrum.values.iter().enumerate() {
            if *value > band { coefficients.row_mut(k).fill(0.0); }
        }
        return Ok(gam_linalg::faer_ndarray::fast_ab(&spectrum.vectors, &coefficients));
    }
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
/// The box gradient differentiates its charged bound, including later sites' dependence on earlier
/// masked activations, with the running Fishers fixed. It does not differentiate the old random-mask
/// expectation or the Fisher estimator. The line search reuses each trial's forward for both terms.
///
/// On a device (module note, "Devices") both claims use resident forward/reverse passes,
/// Fisher accumulation and curvature tangents. Factor gradients and covariance blocks still
/// return to the host for preconditioning and updates. Each backtracking trial is a score-only
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
    // The forward and everything read off it, on the device twin when there is one.
    let lowered = masked.on_proposals(|accelerated| {
            let on_device = accelerated.target(target)?;
            let state = accelerated.forward(&family, &on_device)?;
            let grads = if claim == Claim::Corner { accelerated.piece_gradients(masked, &state, seed % 2 == 0)? }
                else { Vec::new() };
            let curvature = accelerated.step_fisher(masked, &state, &on_device, samples, seed)?;
            let covariances = accelerated.covariances(masked, &state)?;
            Ok((accelerated.decides(), on_device, state, grads, curvature, covariances))
        })?;
    // A twin without float64 proposes; the KL the step starts from, and every trial's, are the
    // CPU's (`deciding` holds the twin's target only when it decides).
    let (kl_now, grads, curvature, batch_covariances, evaluated, device_target, deciding) = match lowered {
        Some((decides, on_device, state, grads, curvature, covariances)) => {
            let kl_now = if decides { state.kl.clone() } else { score_only(masked, &family, target)? };
            (kl_now, grads, curvature, covariances, Evaluated::Device(state), Some(on_device), decides)
        }
        None => {
            let (kl_now, trace, cotangent) = forward(masked, &family, target)?;
            let grads = if claim == Claim::Corner { piece_gradients(masked, &family, &trace, masks, cotangent.clone(), seed % 2 == 0)? }
                else { Vec::new() };
            let curvature = step_fisher(masked, &family, &trace, target, samples, seed)?;
            let rows = trace.values[masked.program.output].nrows() as f64;
            let mut batch_covariances = Vec::new();
            for site in &masked.sites {
                let reads = read_values(&trace, site)?;
                batch_covariances.push(fast_atb(&reads, &reads) / rows);
            }
            (kl_now, grads, curvature, batch_covariances, Evaluated::Host(trace, cotangent), None, false)
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
            let (cost, box_grads) = match &evaluated {
                Evaluated::Host(trace, cotangent) => box_gradients(masked, &family, trace, masks, cotangent.clone(), &running.fishers)?,
                Evaluated::Device(state) => masked.on_lowered(|accelerated| accelerated.box_gradients(masked, state, masks, &running.fishers))?,
            };
            let cost = if !deciding && matches!(&evaluated, Evaluated::Device(_)) {
                box_upper_at(masked, base, masks, &running.fishers)?
            } else { cost };
            let grads = box_grads.into_iter().map(|(v, u)| (Array2::zeros((0, 0)), v, u)).collect();
            (kl_now.sum() + cost.sum(), grads)
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
                masked.on_proposals(|accelerated| accelerated.quadratic(state, on_device, &tangents))?.ok_or("device: the twin went away")?
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
    let eta = if quadratic > 0.0 { slope / quadratic } else { 1.0 };
    let device_target = device_target.filter(|_| deciding);
    backtrack_pieces(masked, &moves, eta, total, |masked| Ok(match (claim, &device_target) {
        (Claim::Corner, Some(on_device)) => {
            masked.on_device(|accelerated| accelerated.score_only(&family, on_device))?.ok_or("device: the twin went away")?.sum()
        }
        (Claim::Corner, None) => score_only(masked, &family, target)?.sum(),
        (Claim::Box, Some(on_device)) => {
            masked.on_device(|accelerated| {
                let state = accelerated.score_state(&family, on_device)?;
                Ok(state.kl.sum() + accelerated.box_upper(masked, &state.trace, masks, &running.fishers)?.sum())
            })?.ok_or("device: the twin went away")?
        }
        (Claim::Box, None) => {
            let (kl_trial, trace) = scored_forward(masked, &family, target)?;
            kl_trial.sum() + box_upper(masked, &trace, masks, &running.fishers)?.sum()
        }
    }))
}

/// Apply numerical trials transactionally: only a measured improvement keeps edited operators.
pub(super) fn backtrack_pieces(
    masked: &mut Masked,
    moves: &[(usize, Array2<f64>)],
    mut eta: f64,
    total: f64,
    mut evaluate: impl FnMut(&Masked) -> Result<f64, String>,
) -> Result<Option<(f64, f64)>, String> {
    let originals: Vec<Arc<Operator>> = moves.iter().map(|(op, _)| Arc::clone(&masked.program.operators[*op])).collect();
    let result = (|| -> Result<Option<(f64, f64)>, String> {
        let floor = eta * f64::EPSILON;
        while eta.is_finite() && eta > floor {
            for ((op, t), original) in moves.iter().zip(&originals) {
                let values = &*original.matrix_cow() - &(t * eta);
                masked.program.operators[*op] = dense(original.name.clone(), original.rows.clone(), original.cols.clone(), values)?;
            }
            let trial = evaluate(masked)?;
            if trial.is_finite() && trial < total {
                return Ok(Some((total, trial)));
            }
            eta *= 0.5;
        }
        Ok(None)
    })();
    // An error during construction or evaluation is not a committed edit.
    if !matches!(&result, Ok(Some(_))) {
        for ((op, _), original) in moves.iter().zip(originals) {
            masked.program.operators[*op] = original;
        }
    }
    result
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
    /// Off means anywhere in `[0, 1]`: the error is the masks' KL plus the contribution charge
    /// ([`box_upper`]), an upper bound on what the box adds at each site to second order.
    Box,
}

/// Per site, the gradients of the contribution charge ([`box_upper`]) in `V` and `U`.
pub type BoxGradients = Vec<(Array2<f64>, Array2<f64>)>;

/// A fingerprint of the written Fishers' every entry (FNV-1a over their bits, per site, folded),
/// so a cache of what they determine is reused exactly while they are unchanged.
pub(crate) fn fisher_print(fishers: &[Array2<f64>]) -> u64 {
    use rayon::prelude::*;
    let prints: Vec<u64> = fishers
        .par_iter()
        .map(|f| f.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, v| (h ^ v.to_bits()).wrapping_mul(0x0000_0100_0000_01b3)))
        .collect();
    prints.iter().fold(fishers.len() as u64, |h, p| (h ^ p).wrapping_mul(0x0000_0100_0000_01b3))
}

/// What the box claim's excess reads of the library and the written Fishers, fixed while neither
/// moves: per site each block's `U_c F U_cᵀ` (a site of rank-one blocks holds them as one row,
/// `u_c F u_cᵀ` per piece).
pub(crate) struct BoxTerms {
    sites: Vec<Vec<Array2<f64>>>,
}

impl BoxTerms {
    pub(crate) fn new(masked: &Masked, fishers: &[Array2<f64>]) -> Result<Self, String> {
        let mut sites = Vec::new();
        for k in 0..masked.sites.len() {
            let u = masked.u(k)?;
            let uf = gam_linalg::faer_ndarray::fast_ab(&u, &fishers[k]);
            let own = if masked.is_rank_one(k) {
                vec![(&uf * &u).sum_axis(Axis(1)).insert_axis(Axis(0))]
            } else {
                let mut blocks = Vec::new();
                let mut start = 0;
                for &r in masked.ranks(k) {
                    blocks.push(uf.slice(s![start..start + r, ..]).dot(&u.slice(s![start..start + r, ..]).t()));
                    start += r;
                }
                blocks
            };
            sites.push(own);
        }
        Ok(Self { sites })
    }
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

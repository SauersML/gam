//! Native weight edits (#2951), one definition for every model: a drawn edit ([`Drawn`]) of `M`'s
//! maps, which `interchange` applies in training's experiments and in the offline scorer alike
//! (`interchange::Patch::Weights`, `Interchange::weight_scores`), each model at its own uses of the
//! edited maps (`interchange::matrix_uses`: `M`'s operators by name; `P`'s operators named as `M`'s,
//! its owning operators, `Artifact::owners`, and the uses of blocks it computes as sums of slices,
//! `Owner::uses`).
//!
//! * A coordinate edit ([`EntryEdit`]: rows or columns of a map scaled by `α`, 0 removing them)
//!   scales the represented entries of every owner: at each use, the edited rows' entries of the
//!   use's output or the edited columns' entries of its input. Where `P` holds the map as a learned
//!   operator, that is the operator's own entries scaled, wherever training moved them; where it
//!   computes it as a sum of slices, the entries inside every slice and leftover.
//! * A fixed delta ([`Factored`], `ΔW = U Vᵀ`) is a named, separate operation: the term `ΔW·x` at
//!   every use, on the use's own input (divided by an owner's scalar factors).
//!
//! An edit of a map `P` applies nowhere is unsupported (`Interchange::weight_edit_supported`) and
//! is reported absent, never scored.

use crate::{
    artifact::{Artifact, Owner},
    operator_program::{Operator, OperatorProgram},
};
use ndarray::Array2;
use std::collections::BTreeMap;

fn error(e: impl std::fmt::Display) -> String {
    format!("weight edit: {e}")
}

/// The index of `program`'s operator named `name`, if exactly one is.
fn named(program: &OperatorProgram, name: &str) -> Result<Option<usize>, String> {
    let mut found = program.operators.iter().enumerate().filter(|(_, op)| op.name == name).map(|(i, _)| i);
    match (found.next(), found.next()) {
        (Some(i), None) => Ok(Some(i)),
        (None, _) => Ok(None),
        _ => Err(error(format!("two operators named {name}"))),
    }
}

/// The product of an owner's factors, which a fixed delta's block is divided by; a matrix factor is refused
/// (its block of `ΔW` need not lie in the factors' range).
pub(crate) fn scalar_factor(explanation: &Artifact, owner: &Owner) -> Result<f64, String> {
    let mut factor = 1.0;
    for name in owner.left.iter().chain(&owner.right) {
        let op = named(&explanation.program, name)?.ok_or_else(|| error(format!("{}: no factor {name}", owner.operator)))?;
        let value = explanation.program.operators[op].matrix();
        if value.dim() != (1, 1) || value[[0, 0]] == 0.0 {
            return Err(error(format!("{}: the factor {name} is not a nonzero scalar", owner.operator)));
        }
        factor *= value[[0, 0]];
    }
    Ok(factor)
}

/// A native edit of coordinates: `units` of the rows (`rows`) or columns of `M`'s operator `native`
/// scaled by `alpha` (0 removes them).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EntryEdit {
    pub native: String,
    pub rows: bool,
    pub units: Vec<usize>,
    pub alpha: f64,
}

/// A family of native weight edits ([`candidates`]), each defined on `M`'s weights alone, as the
/// edits driver's strong draw makes them (`mpd_library_mdl_2951`'s `draw_weights`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// One head scaled by a factor of `interchange::SCALES` (0 removes it): its query rows and its
    /// output columns, and its key and value rows where it reads a key-value head of its own.
    Head,
    /// A group of 8, 16, 32 or 64 neurons of one layer's MLP scaled so: their `c_fc` (and
    /// `gate_proj`) rows and their `down_proj` columns.
    Neurons,
    /// One head's maps replaced by another head's of the same layer (its query and output maps, and
    /// key and value where each reads a key-value head of its own), as the coordinate edit that
    /// computes it: the replaced head then computes what the other does, so its output columns are
    /// scaled 0 and the other head's 2. Each model replaces its own head. An exchange of two heads'
    /// maps is no edit: the attention output sums over its heads.
    HeadReplaced,
    /// A random change `ΔW = s A Bᵀ` of one map: rank `full^u`, `A` and `B` of random signs, `s` setting
    /// `‖ΔW‖_F` to `0.3 · 10^u′` of the map's (`u`, `u′` uniform in `[0, 1)`).
    Random,
}

impl Kind {
    /// Its name in reports.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Head => "weight_head",
            Kind::Neurons => "weight_neurons",
            Kind::HeadReplaced => "weight_head_replaced",
            Kind::Random => "weight_random",
        }
    }
}

/// A native weight edit of one of `M`'s matrices in factored form, `ΔW = U Vᵀ` (`left` `U`, rows
/// × rank; `right` `V`, columns × rank), always on at every use of the matrix.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Factored {
    pub native: String,
    pub left: Array2<f64>,
    pub right: Array2<f64>,
}


/// A drawn native weight edit: its kind, the block whose maps it edits (`2l` a layer's attention,
/// `2l + 1` its MLP), its coordinate edits (`entries`, applied entry-wise in each model: the edited
/// rows or columns of every part and leftover computing the map scaled, `interchange`'s
/// `Patch::Weights`), its other changes in factored form (`factors`, each an always-on term
/// `ΔW·x`), and its effect `KL(M_e ‖ M)` in bits per token once measured
/// (`interchange::Interchange::weight_effects`).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Drawn {
    pub kind: Kind,
    pub block: usize,
    #[serde(default)]
    pub entries: Vec<EntryEdit>,
    #[serde(default)]
    pub factors: Vec<Factored>,
    #[serde(default)]
    pub effect: Option<f64>,
}

/// The edges of the effect bins edits are stratified in ([`stratified`]), `KL(M_e ‖ M)` in bits
/// per token.
pub const EFFECT_EDGES: [f64; 3] = [0.01, 0.1, 1.0];

/// `count` native weight edits of `M` (`native`, a split sequential decoder: `blocks.{l}.c_fc`,
/// `gate_proj`, `down_proj`, `q{h}`, `k{g}`, `v{g}`, `o{h}`), drawn from `seed` as the edits
/// driver's strong draw draws them ([`Kind`]): per edit a layer uniform, a scale factor uniform
/// among `interchange::SCALES`, and a family uniform among the four (a replacement needs two
/// heads; a random change takes a map uniformly among the tensors, then one of its blocks).
pub fn candidates(native: &OperatorProgram, seed: u64, count: usize) -> Result<Vec<Drawn>, String> {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x5745_4947_4854_5332);
    let by_name: BTreeMap<&str, &Operator> = native.operators.iter().map(|op| (op.name.as_str(), op.as_ref())).collect();
    let has = |name: &str| by_name.contains_key(name);
    let count_of = |l: usize, part: &str| (0..).take_while(|h| has(&format!("blocks.{l}.{part}{h}"))).count();
    let layers = (0..).take_while(|l| has(&format!("blocks.{l}.q0"))).count();
    if layers == 0 {
        return Err(error("no attention maps of M to edit (blocks.0.q0)"));
    }
    let shape = |name: &str| by_name.get(name).map(|op| (op.rows.width(), op.cols.width())).ok_or_else(|| error(format!("{name}: not an operator of M")));
    let all = |name: String, rows: bool, alpha: f64| -> Result<EntryEdit, String> {
        let (r, c) = shape(&name)?;
        Ok(EntryEdit { native: name, rows, units: (0..if rows { r } else { c }).collect(), alpha })
    };
    // The maps a random change may take: tensors uniformly (a map stored per head is one tensor),
    // then one of their blocks.
    let mut tensors: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for op in native.operators.iter().filter(|op| op.name.starts_with("blocks.") && op.rows.width() > 1 && op.cols.width() > 1 && op.diagonal().is_none()) {
        tensors.entry(op.name.trim_end_matches(|c: char| c.is_ascii_digit())).or_default().push(&op.name);
    }
    let tensors: Vec<Vec<&str>> = tensors.into_values().collect();
    let block_of = |name: &str| -> usize {
        let l: usize = name.split('.').nth(1).and_then(|l| l.parse().ok()).unwrap_or(0);
        let part = name.split('.').nth(2).unwrap_or("");
        if ["c_fc", "gate_proj", "up_proj", "down_proj"].contains(&part) { 2 * l + 1 } else { 2 * l }
    };
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let l = rng.random_range(0..layers);
        let (heads, groups) = (count_of(l, "q"), count_of(l, "k"));
        let alpha = crate::interchange::SCALES[rng.random_range(0..crate::interchange::SCALES.len())];
        let drawn = match rng.random_range(0..4) {
            0 => {
                let h = rng.random_range(0..heads);
                let mut entries = vec![all(format!("blocks.{l}.q{h}"), true, alpha)?, all(format!("blocks.{l}.o{h}"), false, alpha)?];
                if groups == heads {
                    entries.push(all(format!("blocks.{l}.k{h}"), true, alpha)?);
                    entries.push(all(format!("blocks.{l}.v{h}"), true, alpha)?);
                }
                Drawn { kind: Kind::Head, block: 2 * l, entries, factors: Vec::new(), effect: None }
            }
            1 if has(&format!("blocks.{l}.c_fc")) => {
                let fc = format!("blocks.{l}.c_fc");
                let n = shape(&fc)?.0;
                let k = (8usize << rng.random_range(0..4usize)).min(n);
                let mut units: Vec<usize> = (0..n).collect();
                for j in 0..k {
                    let t = rng.random_range(j..n);
                    units.swap(j, t);
                }
                units.truncate(k);
                units.sort_unstable();
                let mut entries = vec![EntryEdit { native: fc, rows: true, units: units.clone(), alpha }];
                if has(&format!("blocks.{l}.gate_proj")) {
                    entries.push(EntryEdit { native: format!("blocks.{l}.gate_proj"), rows: true, units: units.clone(), alpha });
                }
                entries.push(EntryEdit { native: format!("blocks.{l}.down_proj"), rows: false, units, alpha });
                Drawn { kind: Kind::Neurons, block: 2 * l + 1, entries, factors: Vec::new(), effect: None }
            }
            2 if heads > 1 => {
                let h = rng.random_range(0..heads);
                let other = (h + 1 + rng.random_range(0..heads - 1)) % heads;
                let entries = vec![all(format!("blocks.{l}.o{h}"), false, 0.0)?, all(format!("blocks.{l}.o{other}"), false, 2.0)?];
                Drawn { kind: Kind::HeadReplaced, block: 2 * l, entries, factors: Vec::new(), effect: None }
            }
            _ => {
                let blocks = &tensors[rng.random_range(0..tensors.len())];
                let name = blocks[rng.random_range(0..blocks.len())].to_string();
                let w = by_name[name.as_str()].matrix();
                let full = w.nrows().min(w.ncols());
                let rank = ((full as f64).powf(rng.random::<f64>()).round() as usize).clamp(1, full);
                let ratio = 0.3 * 10f64.powf(rng.random::<f64>());
                let sign = |rng: &mut rand::rngs::StdRng, r: usize, c: usize| Array2::from_shape_fn((r, c), |_| if rng.random::<bool>() { 1.0 } else { -1.0 });
                let (a, b) = (sign(&mut rng, w.nrows(), rank), sign(&mut rng, w.ncols(), rank));
                let size = |m: &Array2<f64>| m.iter().map(|v| v * v).sum::<f64>().sqrt();
                let delta_size = size(&a.dot(&b.t()));
                let s = if delta_size > 0.0 { ratio * size(&w) / delta_size } else { 0.0 };
                Drawn { kind: Kind::Random, block: block_of(&name), entries: Vec::new(), factors: vec![Factored { native: name, left: a * s, right: b }], effect: None }
            }
        };
        out.push(drawn);
    }
    Ok(out)
}

/// `count` of the measured edits `drawn` (each with its effect), stratified by effect: the bins
/// between the edges `edges` (bits per token) take turns, each giving its next edit in the order
/// drawn, so every bin with edits has about `count` over the number of bins, and a bin that runs
/// out leaves its turns to the others.
pub fn stratified(drawn: Vec<Drawn>, count: usize, edges: &[f64]) -> Result<Vec<Drawn>, String> {
    let mut bins: Vec<std::collections::VecDeque<Drawn>> = vec![std::collections::VecDeque::new(); edges.len() + 1];
    for d in drawn {
        let effect = d.effect.ok_or_else(|| error("an edit stratified before its effect was measured"))?;
        bins[edges.iter().filter(|e| effect >= **e).count()].push_back(d);
    }
    let mut out = Vec::with_capacity(count);
    while out.len() < count && bins.iter().any(|b| !b.is_empty()) {
        for bin in bins.iter_mut() {
            if out.len() < count && let Some(d) = bin.pop_front() {
                out.push(d);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_program::exact_precision;
    use ndarray::s;
    use std::sync::Arc;
    use crate::{
        import::import_language_model,
        interchange::{Batch, Experiment, Interchange},
        library_mdl,
        operator_program::SlotValues,
        run_check::{LayerNodes, layer_nodes, split_sites},
    };
    use gam_gpu::tensor::Device;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    struct Setup {
        native: OperatorProgram,
        blocks: Vec<LayerNodes>,
        explanation: library_mdl::Explanation,
        batch: Batch,
    }

    /// The tiny Qwen3 export (tied embeddings, one key-value head read by both query heads) and its
    /// starting library, an exact copy of `M` whose owners name every q, k, v, gate, up and out block.
    fn setup(tag: &str) -> Setup {
        let dir = crate::test_support::tiny_qwen3_export(tag, 2);
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let explanation = library_mdl::scoped(&library_mdl::explanation(&native, &layers).expect("the library"), &[0, 1, 2, 3]).expect("scoped");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let blocks = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
        Setup { native, blocks, explanation, batch }
    }

    /// `KL(M_e ‖ P_e)` in bits summed over every token of the three base sequences, `M_e` running
    /// `model` and `P_e` (autonomous) `explanation` (no read patches, so no read variables).
    fn bits(s: &Setup, model: &OperatorProgram, explanation: &Artifact) -> f64 {
        let device = Device::host();
        let ic = Interchange::new(&device, model, &s.blocks, explanation, &s.explanation.trainable, Vec::new(), 1 << 30, 64).expect("the experiments");
        let clean: Vec<Experiment> = (0..3).map(|base| Experiment { base, source: base, explained: vec![true; 4], patch: None, position: 0 }).collect();
        ic.evaluate(&s.batch, &clean, false).expect("evaluate").bits.iter().flatten().sum()
    }

    fn random(rng: &mut StdRng, shape: (usize, usize), scale: f64) -> Array2<f64> {
        Array2::from_shape_fn(shape, |_| scale * (rng.random::<f64>() - 0.5))
    }

    fn shape(program: &OperatorProgram, name: &str) -> (usize, usize) {
        let op = &program.operators[named(program, name).unwrap().unwrap()];
        (op.rows.width(), op.cols.width())
    }

    fn native_of(s: &Setup, site: &str, role: &str) -> String {
        s.explanation.artifact.owners.iter().find(|o| o.site == site && o.role == role).map(|o| o.native.clone()).expect("an owner")
    }




    /// The tiny export decomposed exactly into gated slices (`library_vpd`; VPD's factors `V = I`,
    /// `U = Wᵀ` for every map, every gate always on): per layer one component at the attention's
    /// input (every q, k, v and o slice) and one at the MLP's input (every c_fc and down slice), or
    /// with `split` two of each listing each other as candidates (gate sharing).
    fn exact_slices(tag: &str, split: bool) -> (Setup, Vec<LayerNodes>) {
        let dir = crate::test_support::tiny_export(tag, 2);
        let record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("export.json")).expect("export.json")).expect("the record");
        let factors = std::env::temp_dir().join(format!("gam_mpd_{tag}_factors_{}", std::process::id()));
        std::fs::create_dir_all(&factors).expect("the factors' directory");
        let (mut files, mut sites) = (serde_json::Map::new(), Vec::new());
        let mut write = |name: String, values: Array2<f64>| {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(factors.join(format!("{name}.f64")), bytes).expect("a factor");
            files.insert(name, serde_json::json!({"shape": [values.nrows(), values.ncols()]}));
        };
        let mut components = Vec::new();
        for l in 0..2 {
            let mut slices = [Vec::new(), Vec::new()];
            for (k, name) in ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj"].into_iter().enumerate() {
                let shape = &record["files"][format!("blocks.{l}.{name}")]["shape"];
                let (r, c) = (shape[0].as_u64().expect("rows") as usize, shape[1].as_u64().expect("columns") as usize);
                let w = crate::import::read_f64_shaped(&dir.join(format!("blocks.{l}.{name}.f64")), r, c).expect("a map");
                let site = format!("h.{l}.{name}");
                write(format!("{site}.U"), w.t().to_owned());
                write(format!("{site}.V"), Array2::eye(c));
                sites.push(site);
                slices[usize::from(k >= 4)].extend((0..c).map(|i| [6 * l + k, i]));
            }
            // Width 10⁻³: the gate Φ(z/w) at z ≥ 1 is 1 in float64, so the decomposition is exact
            // (a shared own gate's width on the squared norm is 2·10⁻⁶, its threshold −1).
            let [attention, mlp] = slices;
            if split {
                // Per stage two components, each listing the other: q, k and o slices with v; the
                // first half of c_fc and of down with the second.
                let (a1, a2): (Vec<[usize; 2]>, Vec<[usize; 2]>) = attention.into_iter().partition(|[site, _]| site % 6 != 2);
                let (m1, m2): (Vec<[usize; 2]>, Vec<[usize; 2]>) = mlp.into_iter().partition(|[site, i]| *i < if site % 6 == 4 { 4 } else { 8 });
                let first = components.len();
                for (k, (read, slices)) in [(0, a1), (2, a2), (4, m1), (4, m2)].into_iter().enumerate() {
                    let other = first + (k ^ 1);
                    components.push(serde_json::json!({"read": {"own": [6 * l + read, 0]}, "tau": -1.0, "width": 1e-3, "slices": slices, "candidates": [other]}));
                }
            } else {
                for (stage, slices) in [attention, mlp].into_iter().enumerate() {
                    components.push(serde_json::json!({"read": {"own": [6 * l + 4 * stage, 0]}, "tau": -1.0, "width": 1e-3, "slices": slices}));
                }
            }
        }
        std::fs::write(factors.join("export.json"), serde_json::json!({"config": {"sites": sites}, "files": files}).to_string()).expect("the factors' record");
        let start = factors.join("start.json");
        std::fs::write(&start, serde_json::json!([{"arm": "exact", "components": components}]).to_string()).expect("the start");
        let imported = import_language_model(&dir, 6, 12).expect("the tiny export imports");
        std::fs::remove_dir_all(dir).expect("the tiny export is removed");
        let native = split_sites(&imported.program).expect("the native sites");
        let layers = layer_nodes(&native, 2).expect("the layers");
        let explanation = crate::library_vpd::explanation(&native, &layers, &factors, &start, "exact").expect("the decomposition");
        std::fs::remove_dir_all(&factors).expect("the factors are removed");
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let blocks = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let batch = Batch::new(sequences[..3].to_vec(), sequences[3..6].to_vec()).expect("the batch");
        let s = Setup { native, blocks, explanation, batch };
        (s, layers)
    }

    /// Gate sharing (`library_vpd`, components that list candidates): on the exact decomposition
    /// with two components per stage, each on its own gate, `P` scores 0 bits, and each shared
    /// stage's record lists its components' candidates, own first. With the second component's
    /// gate shut (threshold −10⁹ on the squared norm) and that component assigned to it, `P` loses
    /// the component; assigning it to the first component's gate (a part of their summed rank,
    /// gated on the norm of both reads) makes `P` exact again.
    #[test]
    fn a_shared_stage_moves_components_between_gates() {
        let (s, _) = exact_slices("weight_edit_share", true);
        assert_eq!(s.explanation.shares.len(), 4, "two shared stages per layer");
        assert!(s.explanation.shares.iter().all(|share| share.candidates == vec![vec![0, 1], vec![1, 0]]), "{:?}", s.explanation.shares);
        let exact = bits(&s, &s.native, &s.explanation.artifact);
        assert!(exact.abs() <= 1e-9, "the shared decomposition scores {exact} bits");
        let set = |artifact: &mut Artifact, name: &str, values: Array2<f64>| {
            let at = named(&artifact.program, name).unwrap().expect("an operator");
            let old = Arc::clone(&artifact.program.operators[at]);
            let precision = exact_precision(values.iter().copied()).expect("a precision");
            artifact.program.operators[at] = Arc::new(Operator::dense(old.name.clone(), old.rows.clone(), old.cols.clone(), values, precision, old.provenance.clone()).expect("an operator"));
        };
        let mut shut = s.explanation.artifact.clone();
        set(&mut shut, "library.l1.mlp.fc.threshold", ndarray::array![[-1.0], [-1e9]]);
        let lost = bits(&s, &s.native, &shut);
        assert!(lost > 1e-4, "the second component on its shut gate is off ({lost} bits)");
        set(&mut shut, "library.l1.mlp.fc.assign", ndarray::array![[1.0, 1.0], [0.0, 0.0]]);
        let moved = bits(&s, &s.native, &shut);
        assert!(moved.abs() <= 1e-9, "both components on the first gate: {moved} bits");
    }


    /// Native weight edits as interchange experiments (`interchange::Patch::Weights`) on the exact
    /// gated-slice decompositions, one component per stage and two per stage sharing their gates:
    /// every family of edit scores zero (the coordinate edits scale the edited rows' or columns'
    /// entries at each use of the map, the replacements and random changes add their terms there;
    /// `Interchange::new` takes the maps' uses, `library_vpd::uses`, for an explanation recording
    /// no owners), and most edits move `M`.
    #[test]
    fn native_weight_edits_score_zero_on_exact_gated_slice_decompositions() {
        use crate::interchange::{Family, Interchange};
        for split in [false, true] {
            let (s, _) = exact_slices(if split { "weight_family_vpd_shared" } else { "weight_family_vpd" }, split);
            let d = Device::host();
            let mut x = Interchange::new(&d, &s.native, &s.blocks, &s.explanation.artifact, &s.explanation.trainable, s.explanation.reads.clone(), 1 << 30, 64).expect("the experiments");
            let drawn = candidates(&s.native, 9, 30).expect("the edits");
            let kinds: std::collections::BTreeSet<Kind> = drawn.iter().map(|e| e.kind).collect();
            assert_eq!(kinds.len(), 4, "{kinds:?}");
            let count = drawn.len();
            x.set_weight_edits(drawn).expect("the table");
            let experiments = x.sample_ops(&mut StdRng::seed_from_u64(4), &s.batch, &[Family::Weight], 10, &[1, 2, 0], false).expect("the draw");
            let bits = x.evaluate(&s.batch, &experiments, false).expect("evaluate").bits;
            let failing: Vec<String> = experiments.iter().zip(&bits).filter(|(_, b)| b.iter().any(|v| v.abs() > 1e-9)).map(|(e, b)| match &e.patch {
                Some(crate::interchange::Patch::Weights { edit, .. }) => format!("{:?} {:?} {:.3}", x.weight_edits()[*edit].kind, x.weight_edits()[*edit].entries.iter().map(|e| (e.native.as_str(), e.rows, e.alpha)).collect::<Vec<_>>(), b.iter().sum::<f64>()),
                _ => format!("clean {:.3}", b.iter().sum::<f64>()),
            }).collect();
            assert!(failing.is_empty(), "shared {split}: {failing:?}");
            let effects = x.weight_effects(&s.batch, &(0..count).collect::<Vec<_>>()).expect("the effects");
            assert!(effects.iter().filter(|e| **e > 1e-6).count() * 10 >= count * 8, "shared {split}: {effects:?}");
        }
    }


    /// `M` and `P` with the drawn edit `d` applied by direct mutation of their weights: `M`'s edited
    /// rows or columns scaled and its fixed deltas added; in `P`, every owner of an edited map (each
    /// distinct block once) with its represented entries scaled (a coordinate edit: the entries of
    /// the owned block standing for the edited rows or columns, wherever the learned values are) and
    /// the owned block of each fixed delta added (divided by the owner's scalar factors).
    fn mutated(native: &OperatorProgram, artifact: &Artifact, d: &Drawn) -> (OperatorProgram, Artifact) {
        let set = |program: &mut OperatorProgram, op: usize, values: Array2<f64>| {
            let old = Arc::clone(&program.operators[op]);
            let precision = exact_precision(values.iter().copied()).expect("a precision");
            program.operators[op] = Arc::new(Operator::dense(old.name.clone(), old.rows.clone(), old.cols.clone(), values, precision, old.provenance.clone()).expect("an operator"));
        };
        let (mut m, mut p) = (native.clone(), artifact.clone());
        for e in &d.entries {
            let w = named(&m, &e.native).unwrap().expect("M's map");
            let mut v = m.operators[w].matrix();
            for &u in &e.units {
                if e.rows { v.row_mut(u).mapv_inplace(|x| x * e.alpha) } else { v.column_mut(u).mapv_inplace(|x| x * e.alpha) }
            }
            set(&mut m, w, v.clone());
            // P's own use of the map by name takes M's edited map.
            if let Some(op) = named(&p.program, &e.native).unwrap() {
                let mut kept = p.program.operators[op].matrix();
                for &u in &e.units {
                    if e.rows { kept.row_mut(u).mapv_inplace(|x| x * e.alpha) } else { kept.column_mut(u).mapv_inplace(|x| x * e.alpha) }
                }
                set(&mut p.program, op, kept);
            }
            let mut done = std::collections::BTreeSet::new();
            for o in artifact.owners.iter().filter(|o| o.native == e.native) {
                assert!(o.uses.is_none(), "a summed block");
                if !done.insert((o.operator.clone(), o.rows.start, o.rows.end, o.cols.start, o.cols.end)) {
                    continue;
                }
                let op = named(&p.program, &o.operator).unwrap().expect("an owner");
                let mut v = p.program.operators[op].matrix();
                let range = if e.rows { &o.native_rows } else { &o.native_cols };
                for &u in e.units.iter().filter(|u| range.contains(u)) {
                    let k = u - range.start;
                    // The edited native rows are the owner's rows (its columns, held transposed).
                    if e.rows != o.transposed {
                        v.slice_mut(s![o.rows.start + k, o.cols.clone()]).mapv_inplace(|x| x * e.alpha);
                    } else {
                        v.slice_mut(s![o.rows.clone(), o.cols.start + k]).mapv_inplace(|x| x * e.alpha);
                    }
                }
                set(&mut p.program, op, v);
            }
        }
        for f in &d.factors {
            let delta = f.left.dot(&f.right.t());
            let w = named(&m, &f.native).unwrap().expect("M's map");
            let v = m.operators[w].matrix() + &delta;
            set(&mut m, w, v);
            if let Some(op) = named(&p.program, &f.native).unwrap() {
                let kept = p.program.operators[op].matrix() + &delta;
                set(&mut p.program, op, kept);
            }
            let mut done = std::collections::BTreeSet::new();
            for o in artifact.owners.iter().filter(|o| o.native == f.native) {
                if !done.insert((o.operator.clone(), o.rows.start, o.rows.end, o.cols.start, o.cols.end)) {
                    continue;
                }
                let op = named(&p.program, &o.operator).unwrap().expect("an owner");
                let factor = scalar_factor(&p, o).expect("a scalar factor");
                let block = delta.slice(s![o.native_rows.clone(), o.native_cols.clone()]).mapv(|x| x / factor);
                let block = if o.transposed { block.t().to_owned() } else { block };
                let mut v = p.program.operators[op].matrix();
                let mut at = v.slice_mut(s![o.rows.clone(), o.cols.clone()]);
                at += &block;
                set(&mut p.program, op, v);
            }
        }
        (m, p)
    }

    /// One edit path for training and offline scoring: with every learned map of `P` moved off
    /// `M`'s, each drawn native edit (`candidates`: a head scaled, a neuron group scaled, a head
    /// replaced, a random change; and two fixed deltas of one map, which add) scores the same through
    /// training's experiments (`interchange`'s `Patch::Weights`), through the offline scorer
    /// (`Interchange::weight_scores`, the same path), and as `M` and `P` mutated directly
    /// (coordinate edits scaling the represented entries of every owner, wherever its learned values
    /// are; fixed deltas added to the owned blocks): `KL(M_e ‖ P_e)` within 1e-9 bits. With the down
    /// map's owners dropped, an edit of it is unsupported: absent from the scores.
    #[test]
    fn training_edits_match_direct_mutation_of_moved_maps() {
        use crate::interchange::Patch;
        let s = setup("weight_edit_paths");
        let mut rng = StdRng::seed_from_u64(9);
        let mut artifact = s.explanation.artifact.clone();
        for &op in &s.explanation.trainable {
            let old = Arc::clone(&artifact.program.operators[op]);
            let v = old.matrix();
            let values = &v + &(random(&mut rng, v.dim(), 0.4) * &v);
            let precision = exact_precision(values.iter().copied()).expect("a precision");
            artifact.program.operators[op] = Arc::new(Operator::dense(old.name.clone(), old.rows.clone(), old.cols.clone(), values, precision, old.provenance.clone()).expect("an operator"));
        }
        let moved = bits(&s, &s.native, &artifact);
        assert!(moved > 1e-3, "P moved off M scores {moved} bits");
        let mut drawn = candidates(&s.native, 5, 16).expect("the edits");
        let down = native_of(&s, "library.l0.mlp", "out");
        let (rows, cols) = shape(&s.native, &down);
        let factor = |rng: &mut StdRng| Factored { native: down.clone(), left: random(rng, (rows, 2), 1.0), right: random(rng, (cols, 2), 1.0) };
        drawn.push(Drawn { kind: Kind::Random, block: 1, entries: Vec::new(), factors: vec![factor(&mut rng), factor(&mut rng)], effect: None });
        let device = Device::host();
        let mut x = Interchange::new(&device, &s.native, &s.blocks, &artifact, &s.explanation.trainable, Vec::new(), 1 << 30, 64).expect("the experiments");
        x.set_weight_edits(drawn.clone()).expect("the table");
        for (i, d) in drawn.iter().enumerate() {
            let experiments: Vec<Experiment> = (0..3).map(|base| Experiment { base, source: base, explained: vec![true; 4], patch: Some(Patch::Weights { edit: i, block: d.block }), position: 0 }).collect();
            let training: f64 = x.evaluate(&s.batch, &experiments, false).expect("evaluate").bits.iter().flatten().sum();
            let offline = x.weight_scores(&s.batch, &[i]).expect("the scores")[0].expect("supported").gap * (3 * 12) as f64;
            let (m, p) = mutated(&s.native, &artifact, d);
            let direct = bits(&s, &m, &p);
            assert!((training - direct).abs() <= 1e-9 * direct.abs().max(1.0), "edit {i} ({:?}): training {training} bits, direct mutation {direct}", d.kind);
            assert!((offline - training).abs() <= 1e-9 * training.abs().max(1.0), "edit {i} ({:?}): offline {offline} bits, training {training}", d.kind);
        }
        let mut unowned = artifact.clone();
        unowned.owners.retain(|o| o.native != down);
        let mut y = Interchange::new(&device, &s.native, &s.blocks, &unowned, &s.explanation.trainable, Vec::new(), 1 << 30, 64).expect("the experiments");
        y.set_weight_edits(drawn.clone()).expect("the table");
        let last = drawn.len() - 1;
        assert!(!y.weight_edit_supported(last).expect("a table entry"), "no use of the down map: unsupported");
        assert!(y.weight_scores(&s.batch, &[last]).expect("the scores")[0].is_none(), "an unsupported edit is absent");
    }

}

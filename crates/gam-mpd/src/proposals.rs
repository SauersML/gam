//! The team's pieces as proposals to the one search (#2951, `acceptance`): each turns what its
//! owner fitted or found into a candidate [`Artifact`], and the search alone decides.
//!
//! # MLP accounts
//!
//! An `mlp_account` account `ŷ = Σ_j b̂_j(a) p_j + μ`, `a = V h`, `b̂_j = β_j + ℓ_jᵀ a_{S_j} + Σ_u c_u
//! σ(w_uᵀ a_{S_j} + d_u)` ([`account_rule`]), is the rule whose body over the MLP's normed input `h`
//! is
//!
//! ```text
//! a = V h,   z = W a + d,   g = σ(z),   b = L a + C g + β,   ŷ = Pᵀ b + μ,
//! ```
//!
//! with `W` (units × amplitudes), `L` (rules × amplitudes) and `C` (rules × units) operators whose
//! present one-coordinate blocks are exactly each rule's inputs and units: the account's literals,
//! one for one, and its pointers as the present-block code. It replaces the block from the MLP's
//! normed input to its output (`run_check::LayerNodes::normed`, `::mlp`); the hidden layer appears
//! nowhere, so a native neuron is not a place of the result. [`AccountProposer`] proposes every
//! account a directory holds in `mlp_functions`' layout (`L{layer}[.{start}].rules.json` with
//! `.reads.f64`, `.writes.f64` and `.offset.f64`). Each account also offers
//! [`with_account_shared`]: complete identical learned scalar functions become shared Rule
//! bodies with explicit read bindings. This preserves supplied numerical functions; it neither
//! discovers new operations nor turns repeated GELU primitives into learned-rule evidence.
//! Ordinary serialized C32 and exact fidelity checks decide between flat and shared variants.
//!
//! # Attention head rules
//!
//! [`HeadRules`] proposes `rules`' laws as derived operators (`artifact::Derived`) of the heads of
//! declared layers, each binding its layer's attention (`run_check::LayerNodes::normed_stream` to
//! `::attention`) as the block it changes:
//!
//! * **copy**: the head's output operator as `λ diag(g / g_f) V⁺` of its value operator, `λ` the
//!   least-squares scale (`rules::copy_scale`);
//! * **match**: the head's query rows on its content planes through every head of an earlier layer
//!   as the source, the content planes the ones on which the rule carries the most of the head's own
//!   circuit (`rules::match_energy`), `λ` the least-squares scale (`rules::match_scale`); the other rows
//!   either kept as literal residual rows or left zero.
//!
//! What a rule costs and whether it holds is the search's: it ranks the candidates by the bits
//! they save and accepts by the two disagreements.

use super::acceptance::{Context, Proposal, Proposer};
use super::artifact::{Argument, Artifact, Callee, OperatorLaw};
use super::mlp_account::{Account, Rule as AccountRule};
use super::operator_program::{Interface, LabelKind, Law, Node, Operator, Provenance, Rule, exact_precision};
use super::run_check::LayerNodes;
use ndarray::{Array1, Array2};
use std::path::{Path, PathBuf};

#[path = "shared_account.rs"]
mod shared_accounts;
pub use shared_accounts::with_account_shared;

/// A dense operator on `rows × cols` one-coordinate groups whose present blocks are `present`.
fn sparse(name: &str, rows: &Interface, cols: &Interface, values: Array2<f64>, present: Array2<bool>) -> Result<Operator, String> {
    let precision = exact_precision(values.iter().copied()).map_err(|e| e.to_string())?;
    Operator::blocks(name, rows.clone(), cols.clone(), values, present, precision, Provenance::default()).map_err(|e| e.to_string())
}

fn dense(name: &str, rows: &Interface, cols: &Interface, values: Array2<f64>) -> Result<Operator, String> {
    let present = Array2::from_elem((rows.group_count(), cols.group_count()), true);
    sparse(name, rows, cols, values, present)
}

fn units(count: usize, kind: LabelKind) -> Result<Interface, String> {
    Interface::uniform(count, 1, kind, 0).map_err(|e| e.to_string())
}

/// `account` as a rule body over `input` writing `output` (module note); its operators are numbered
/// from `first` (the program's operator count when the rule is added).
pub fn account_rule(name: &str, account: &Account, input: &Interface, output: &Interface, first: usize) -> Result<(Rule, Vec<Operator>), String> {
    let (m, k) = (account.reads.nrows(), account.rules.len());
    if account.reads.ncols() != input.width() || account.writes.ncols() != output.width() || account.writes.nrows() != k {
        return Err(format!("{name}: an account of {m} reads of width {}, {} writes of width {}", account.reads.ncols(), account.writes.nrows(), account.writes.ncols()));
    }
    let amplitudes = units(m, LabelKind::Factor)?;
    let outgoing = units(k, LabelKind::Factor)?;
    let total: usize = account.rules.iter().map(|r| r.units.len()).sum();
    let mut operators = vec![dense(&format!("{name} reads"), &amplitudes, input, account.reads.clone())?];
    let constant = Interface::constant();
    let column = |label: &str, rows: &Interface, values: Array1<f64>| dense(label, rows, &constant, values.insert_axis(ndarray::Axis(1)));
    let (mut linear, mut linear_present) = (Array2::<f64>::zeros((k, m)), Array2::from_elem((k, m), false));
    let (mut weights, mut weights_present) = (Array2::<f64>::zeros((total, m)), Array2::from_elem((total, m), false));
    let (mut scales, mut scales_present) = (Array2::<f64>::zeros((k, total)), Array2::from_elem((k, total), false));
    let mut offsets = Array1::<f64>::zeros(total);
    let mut beta = Array1::<f64>::zeros(k);
    let mut unit = 0;
    for (j, rule) in account.rules.iter().enumerate() {
        let AccountRule { inputs, beta: b, linear: l, units: rule_units } = rule;
        if inputs.iter().any(|i| *i >= m) || l.len() != inputs.len() {
            return Err(format!("{name}: rule {j} reads beyond the account's {m} amplitudes"));
        }
        beta[j] = *b;
        for (i, w) in inputs.iter().zip(l) {
            linear[[j, *i]] = *w;
            linear_present[[j, *i]] = true;
        }
        for u in rule_units {
            for (i, w) in inputs.iter().zip(&u.w) {
                weights[[unit, *i]] = *w;
                weights_present[[unit, *i]] = true;
            }
            offsets[unit] = u.d;
            scales[[j, unit]] = u.c;
            scales_present[[j, unit]] = true;
            unit += 1;
        }
    }
    let at = |i: usize| first + i;
    let mut nodes = vec![Node::Param { index: 0 }, Node::Affine { terms: vec![(0, at(0))], bias: None }];
    operators.push(sparse(&format!("{name} linear"), &outgoing, &amplitudes, linear, linear_present)?);
    operators.push(column(&format!("{name} beta"), &outgoing, beta)?);
    let outgoing_node = if total > 0 {
        let hidden = units(total, LabelKind::Unit)?;
        operators.push(sparse(&format!("{name} unit weights"), &hidden, &amplitudes, weights, weights_present)?);
        operators.push(column(&format!("{name} unit offsets"), &hidden, offsets)?);
        operators.push(sparse(&format!("{name} unit scales"), &outgoing, &hidden, scales, scales_present)?);
        nodes.push(Node::Affine { terms: vec![(1, at(3))], bias: Some(at(4)) });
        nodes.push(Node::Pointwise { input: 2, laws: vec![Law::GeluTanh; total] });
        nodes.push(Node::Affine { terms: vec![(1, at(1)), (3, at(5))], bias: Some(at(2)) });
        4
    } else {
        nodes.push(Node::Affine { terms: vec![(1, at(1))], bias: Some(at(2)) });
        2
    };
    let base = operators.len();
    operators.push(dense(&format!("{name} writes"), output, &outgoing, account.writes.t().to_owned())?);
    operators.push(column(&format!("{name} offset"), output, account.offset.clone())?);
    nodes.push(Node::Affine { terms: vec![(outgoing_node, at(base))], bias: Some(at(base + 1)) });
    let output_node = nodes.len() - 1;
    Ok((Rule { name: name.to_string(), inputs: vec![input.clone()], nodes, output: output_node }, operators))
}

/// `artifact` with layer `layer`'s MLP replaced by `account` (module note).
pub fn with_account(artifact: &Artifact, name: &str, layer: &LayerNodes, account: &Account) -> Result<Artifact, String> {
    let read = artifact.place(layer.normed).ok_or_else(|| format!("{name}: P does not hold the MLP's input"))?;
    let write = artifact.place(layer.mlp).ok_or_else(|| format!("{name}: P does not hold the MLP's output"))?;
    let interfaces = artifact.program.interfaces().map_err(|e| e.to_string())?;
    let (rule, operators) = account_rule(name, account, &interfaces[read], &interfaces[write], artifact.program.operators.len())?;
    artifact.replace_block(name, Callee::New(rule), vec![Argument::Native(layer.normed)], layer.mlp, operators)
}

fn read_rows(path: &Path, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if cols == 0 || bytes.len() % (8 * cols) != 0 {
        return Err(format!("{}: {} bytes are not rows of {cols} float64", path.display(), bytes.len()));
    }
    let values: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("eight bytes"))).collect();
    Array2::from_shape_vec((values.len() / cols, cols), values).map_err(|e| e.to_string())
}

/// The account stored at `base` (`base.rules.json`, `.reads.f64`, `.writes.f64`, `.offset.f64`) for
/// an MLP from `d_in` to `d_out`.
pub fn load_account(base: &Path, d_in: usize, d_out: usize) -> Result<Account, String> {
    let with = |extension: &str| account_path(base, extension);
    let text = std::fs::read_to_string(with("rules.json")).map_err(|e| format!("{}: {e}", with("rules.json").display()))?;
    let rules: Vec<AccountRule> = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let offset = if with("offset.f64").exists() { read_rows(&with("offset.f64"), d_out)?.row(0).to_owned() } else { Array1::zeros(d_out) };
    Ok(Account { reads: read_rows(&with("reads.f64"), d_in)?, rules, writes: read_rows(&with("writes.f64"), d_out)?, offset })
}

/// Append an account suffix without replacing an initialization name such as `.neurons`.
pub fn account_path(base: &Path, suffix: &str) -> PathBuf {
    let mut path = base.as_os_str().to_os_string();
    path.push(".");
    path.push(suffix);
    PathBuf::from(path)
}

/// Save the exact account consumed by [`load_account`]. Distinct initialization stems retain
/// distinct files. Write the rule file last so a new incomplete export is not discoverable.
pub fn save_account(base: &Path, account: &Account) -> Result<(), String> {
    let write = |suffix: &str, bytes: &[u8]| {
        let path = account_path(base, suffix);
        std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))
    };
    for (suffix, values) in [("reads.f64", account.reads.view()), ("writes.f64", account.writes.view())] {
        write(suffix, &values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    }
    write("offset.f64", &account.offset.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    write("rules.json", &serde_json::to_vec(&account.rules).map_err(|e| e.to_string())?)
}

#[cfg(test)]
mod account_export_tests {
    use super::*;
    use super::super::operator_program::{Declarations, FamilyInputs, OperatorProgram, Slot, SlotValues};

    #[test]
    fn distinct_starts_reload_the_selected_composed_program() {
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("clock").as_nanos();
        let dir = std::env::temp_dir().join(format!("mpd-account-export-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&dir).expect("test directory");
        let mut selected = Vec::new();
        let mut loaded = Vec::new();
        for layer in 0..2 {
            let first = Account::neurons(&ndarray::array![[1., 0.5], [-0.5, 1.]], &ndarray::array![[1., 0.25], [0.5, 1.]]);
            let mut last = first.clone();
            last.offset.fill(10.0 + layer as f64);
            let first_base = dir.join(format!("L{layer}.vpd"));
            let last_base = dir.join(format!("L{layer}.neurons"));
            save_account(&first_base, &first).expect("first start");
            save_account(&last_base, &last).expect("last start must not overwrite first");
            let replay = load_account(&first_base, 2, 2).expect("selected start");
            let other = load_account(&last_base, 2, 2).expect("other start");
            assert_eq!(replay.reads, first.reads);
            assert_eq!(replay.rules, first.rules);
            assert_eq!(replay.writes, first.writes);
            assert_eq!(replay.offset, first.offset);
            assert_eq!(other.offset, last.offset);
            assert_ne!(replay.offset, other.offset);
            assert!(!dir.join(format!("L{layer}.rules.json")).exists());
            selected.push(first);
            loaded.push(replay);
        }
        let h = ndarray::array![[1., -2.], [0., 0.], [2., 0.5]];
        let run = |accounts: &[Account]| accounts.iter().fold(h.clone(), |state, account| account.apply(state.view()));
        assert_eq!(run(&selected), run(&loaded));

        // Compile the reloaded layers into two executable calls. Matching the account files
        // alone would not check that the selected hybrid computation survives compilation.
        let interface = Interface::native(2).expect("interface");
        let mut program = OperatorProgram {
            declarations: Declarations { domains: Vec::new(), slots: vec![Slot::Raw { width: 2 }], parameters: 0 },
            bases: Vec::new(), operators: Vec::new(), rules: Vec::new(),
            nodes: vec![Node::Raw { slot: 0 }], output: 2,
        };
        for (layer, account) in loaded.iter().enumerate() {
            let (body, operators) = account_rule(&format!("layer {layer}"), account, &interface, &interface, program.operators.len()).expect("compile account");
            program.operators.extend(operators.into_iter().map(std::sync::Arc::new));
            program.rules.push(body);
            program.nodes.push(Node::Call { rule: layer, arguments: vec![layer] });
        }
        let family = FamilyInputs { rows: h.nrows(), slots: vec![SlotValues::Raw(h.clone())], layout: None };
        let trace = program.execute(&family, false).expect("reloaded program executes");
        let expected = run(&selected);
        assert!(trace.values[program.output].iter().zip(&expected).all(|(a, b)| (a - b).abs() < 1e-12));
        std::fs::remove_dir_all(dir).expect("remove test directory");
    }
}

/// Every account a directory holds, each proposed for its layer's MLP (module note).
pub struct AccountProposer {
    pub layers: Vec<LayerNodes>,
    /// `(layer, name, account)`.
    pub accounts: Vec<(usize, String, Account)>,
}

impl AccountProposer {
    /// The accounts of `dir` (`L{layer}[.{start}].rules.json` and its siblings) for MLPs from
    /// `d_in` to `d_out`.
    pub fn load(dir: &Path, layers: Vec<LayerNodes>, d_in: usize, d_out: usize) -> Result<Self, String> {
        let mut accounts = Vec::new();
        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?.filter_map(|e| e.ok().map(|e| e.path())).collect();
        entries.sort();
        for path in entries {
            let Some(file) = path.file_name().and_then(|f| f.to_str()) else { continue };
            let Some(stem) = file.strip_suffix(".rules.json") else { continue };
            let Some(layer) = stem.strip_prefix('L').and_then(|rest| rest.split('.').next()).and_then(|l| l.parse::<usize>().ok()) else { continue };
            if layer >= layers.len() {
                continue;
            }
            accounts.push((layer, format!("mlp account {stem}"), load_account(&dir.join(stem), d_in, d_out)?));
        }
        Ok(Self { layers, accounts })
    }
}

impl Proposer for AccountProposer {
    fn name(&self) -> &str {
        "mlp accounts"
    }

    fn propose(&self, context: &Context<'_>) -> Result<Vec<Proposal>, String> {
        let mut out = Vec::new();
        for (layer, name, account) in &self.accounts {
            // An MLP already replaced is not proposed again.
            if context.current.blocks.iter().any(|b| b.native_write == self.layers[*layer].mlp) {
                continue;
            }
            let candidate = with_account(context.current, name, &self.layers[*layer], account)?;
            out.push(Proposal { source: "functions".to_string(), description: name.clone(), candidate });
            let candidate = with_account_shared(context.current, name, &self.layers[*layer], account)?;
            out.push(Proposal { source: "functions shared bodies".to_string(), description: format!("{name} shared bodies"), candidate });
        }
        Ok(out)
    }
}

/// The index of the operator named `name` in `artifact`'s program.
fn operator(artifact: &Artifact, name: &str) -> Result<usize, String> {
    artifact.program.operators.iter().position(|o| o.name == name).ok_or_else(|| format!("no operator {name}"))
}

/// Attention head rules as derived operators (module note).
pub struct HeadRules {
    pub layers: Vec<LayerNodes>,
    /// The layers whose heads are proposed.
    pub targets: Vec<usize>,
    /// Query heads per key-value head.
    pub group: usize,
    /// Per `(layer, head, source layer, source head)`, the content planes' first plane whose match
    /// rule aligns best with the head's own scores, found once on the start's operators.
    firsts: std::collections::BTreeMap<(usize, usize, usize, usize), usize>,
}

/// Complete implicit Copy-subset bank. No mask is filtered by singleton fidelity.
/// Stores metadata only; artifacts are built lazily in canonical layer/head order.
type CopyHead = (super::artifact::Derived, std::sync::Arc<Operator>);

pub struct CopyMasks<'a> {
    start: &'a Artifact,
    rules: HeadRules,
    counts: Vec<usize>,
    heads: Vec<Vec<std::sync::OnceLock<Result<CopyHead, String>>>>,
    pub layer_candidates: usize,
    pub joint_candidates: usize,
}

impl<'a> CopyMasks<'a> {
    /// Cardinality guard runs before cloning layer metadata or building an artifact.
    pub fn cardinalities(heads: impl IntoIterator<Item = usize>, max_layer: usize, max_joint: usize) -> Result<(usize, usize), String> {
        let (mut local, mut joint, mut layers) = (0usize, 1usize, 0usize);
        for heads in heads {
            if heads == 0 { return Err("Copy masks require nonempty heads".into()); }
            let count = 1usize.checked_shl(u32::try_from(heads).map_err(|_| "mask width overflow")?).ok_or("mask cardinality overflow")?;
            local = local.checked_add(count).ok_or("layer bank cardinality overflow")?;
            joint = joint.checked_mul(count).ok_or("joint bank cardinality overflow")?;
            layers += 1;
        }
        if layers == 0 { return Err("Copy masks require nonempty layers".into()); }
        if local > max_layer || joint > max_joint {
            return Err(format!("complete Copy bank requires {local} layer masks and {joint} joint candidates; budgets are {max_layer}/{max_joint}"));
        }
        Ok((local, joint))
    }

    pub fn new(start: &'a Artifact, layers: &[LayerNodes], group: usize, max_layer: usize, max_joint: usize) -> Result<Self, String> {
        let (layer_candidates, joint_candidates) = Self::cardinalities(layers.iter().map(|l| l.queries.len()), max_layer, max_joint)?;
        if group == 0 || !start.derived.is_empty() || !start.blocks.is_empty() || !start.exceptions.is_empty() {
            return Err("Copy masks require an unmodified native artifact and positive group".into());
        }
        for l in layers {
            if l.queries.len() % group != 0 || l.keys.len() != l.queries.len() / group || l.values.len() != l.keys.len() {
                return Err("inconsistent Copy head counts".into());
            }
        }
        Ok(Self { start, rules: HeadRules { layers: layers.to_vec(), targets: Vec::new(), group, firsts: Default::default() },
            counts: layers.iter().map(|l| 1usize << l.queries.len()).collect(),
            heads: layers.iter().map(|l| std::iter::repeat_with(std::sync::OnceLock::new).take(l.queries.len()).collect()).collect(),
            layer_candidates, joint_candidates })
    }

    /// Compose one mask per layer. Empty masks introduce neither derivations nor bindings.
    /// Repeated heads in a layer merge into one complete attention binding via Artifact::bind.
    pub fn compose(&self, masks: &[usize]) -> Result<Artifact, String> {
        if masks.len() != self.counts.len() || masks.iter().zip(&self.counts).any(|(m, c)| m >= c) {
            return Err("one in-range Copy mask per layer required".into());
        }
        self.prepare()?;
        let mut artifact = self.start.clone();
        for (layer, &mask) in masks.iter().enumerate() {
            for head in 0..self.rules.layers[layer].queries.len() {
                if mask & (1usize << head) != 0 {
                    let (derived, operator) = self.head(layer, head)?;
                    artifact.program.operators[derived.operator] = operator.clone();
                    artifact.derived.push(derived.clone());
                }
            }
            if mask != 0 {
                let nodes = &self.rules.layers[layer];
                artifact = artifact.bind(&format!("attention {layer}"), &[nodes.normed_stream], nodes.attention)?;
            }
        }
        Ok(artifact)
    }

    fn head(&self, layer: usize, head: usize) -> Result<&CopyHead, String> {
        self.heads[layer][head].get_or_init(|| {
            let candidate = self.rules.copy(self.start, layer, head)?.candidate;
            let derived = candidate.derived.into_iter().next().ok_or("Copy head missing derivation")?;
            let operator = candidate.program.operators[derived.operator].clone();
            Ok((derived, operator))
        }).as_ref().map_err(|e| e.clone())
    }

    /// Compute each independent native-source Copy derivation once (24 for the 4L target).
    /// Subsequent masks share those computed operators, rather than refitting per subset.
    pub fn prepare(&self) -> Result<(), String> {
        let mut targets = std::collections::BTreeSet::new();
        for (layer, heads) in self.heads.iter().enumerate() {
            for head in 0..heads.len() { targets.insert(self.head(layer, head)?.0.operator); }
        }
        for (layer, heads) in self.heads.iter().enumerate() {
            for head in 0..heads.len() {
                if self.head(layer, head)?.0.law.sources().iter().any(|s| targets.contains(s)) {
                    return Err("Copy bank sources must be independent of every substituted output".into());
                }
            }
        }
        Ok(())
    }

    /// All layer-local masks, including each empty mask, without materializing joint candidates.
    pub fn layer_masks(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.counts.iter().enumerate().flat_map(|(l, &count)| (0..count).map(move |mask| (l, mask)))
    }

    pub fn layer(&self, layer: usize, mask: usize) -> Result<Artifact, String> {
        if layer >= self.counts.len() { return Err("Copy layer out of range".into()); }
        let mut masks = vec![0; self.counts.len()];
        masks[layer] = mask;
        self.compose(&masks)
    }

    /// Pin numerical projection, decoded coverage, canonical bytes, and objective C32.
    /// Exact wire byte length is returned separately from C32.
    pub fn checked(&self, masks: &[usize], native: &super::operator_program::OperatorProgram) -> Result<(Artifact, super::acceptance::StructuralCost, usize), String> {
        let artifact = self.compose(masks)?.f32_literals()?;
        artifact.validate_coverage(native)?;
        let bytes = artifact.to_bytes()?;
        let decoded = Artifact::from_bytes(&bytes, &native.declarations)?;
        decoded.validate_coverage(native)?;
        if decoded.to_bytes()? != bytes { return Err("Copy joint roundtrip differs".into()); }
        let cost = super::acceptance::structural_cost(&artifact, &mut Default::default())?;
        if cost != super::acceptance::structural_cost(&decoded, &mut Default::default())? {
            return Err("Copy joint decoded C32 differs".into());
        }
        Ok((decoded, cost, bytes.len()))
    }
}

/// The two uniformly enumerated attention-output approximations. Neither is selected
/// by a singleton fidelity screen; exact decoded Local and autonomous Run remain required.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum HeadApproximation { CopyResidual, NativeSvd }

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct CopyResidualChoice {
    pub layer: usize,
    pub head: usize,
    pub rank: usize,
    pub family: HeadApproximation,
}

struct HeadSvd {
    u: Array2<f64>,
    singular_values: Array1<f64>,
    vt: Array2<f64>,
}

impl HeadSvd {
    fn of(matrix: &Array2<f64>) -> Result<Self, String> {
        let d = gam_linalg::decompose::svd(matrix.view(), false).map_err(|e| e.to_string())?;
        Ok(Self { u: d.u, singular_values: d.singular_values, vt: d.vt })
    }

    fn operator(&self, original: &Operator, name: String, rank: usize, method: &str) -> Result<Operator, String> {
        if rank == 0 || rank > self.singular_values.len() { return Err("head SVD rank outside matrix dimensions".into()); }
        let kept: Vec<_> = (0..rank).collect();
        let roots = Array1::from_iter(self.singular_values.iter().take(rank).map(|s| s.sqrt()));
        // Explicit standard layout is required by the exact wire codec, also at k>1.
        let mut left = (self.u.select(ndarray::Axis(1), &kept) * &roots).as_standard_layout().into_owned();
        let mut right = (self.vt.select(ndarray::Axis(0), &kept) * &roots.insert_axis(ndarray::Axis(1))).as_standard_layout().into_owned();
        left.mapv_inplace(|v| f64::from(v as f32));
        right.mapv_inplace(|v| f64::from(v as f32));
        let precision = exact_precision(left.iter().chain(right.iter()).copied()).map_err(|e| e.to_string())?;
        Operator::low_rank(name, original.rows.clone(), original.cols.clone(), left, right, precision,
            Provenance::derived(&[&original.provenance], format!("{method}; declared rank {rank}; balanced factors projected to f32"))).map_err(|e| e.to_string())
    }
}

/// A rank-truncated linear-map proposal fitted to declared training inputs, rather
/// than to unweighted matrix entries. For inputs `X` (observations × input width)
/// and matrix `W`, truncation minimizes `||X (W - W_r)^T||_F` in exact arithmetic.
///
/// This is only a proposal metric. The returned f32 factors still require decoded
/// Local and autonomous Run acceptance on the declared evaluation family. No Fisher
/// metric, calibration factor, or fidelity threshold enters this fit. Full column
/// rank of `X` is required at the SVD's fixed machine-resolution convention: this
/// version refuses unobserved input directions instead of silently choosing their
/// extrapolation. Training inputs are not required to execute the resulting operator.
pub struct DataWeightedSvd {
    fit: HeadSvd,
    pub training_rows: usize,
    pub input_width: usize,
    pub smallest_input_singular_value: f64,
    pub largest_input_singular_value: f64,
}

/// Small reusable factorization of one declared training input family. The tall
/// left singular vectors are discarded: different proposed maps and residuals
/// share only the input-width square right factor and its singular values.
pub struct InputSvd {
    vt: Array2<f64>,
    singular_values: Array1<f64>,
    training_rows: usize,
}

impl InputSvd {
    pub fn new(inputs: &Array2<f64>) -> Result<Self, String> {
        if inputs.ncols() == 0 || inputs.nrows() < inputs.ncols()
            || inputs.iter().any(|v| !v.is_finite())
        {
            return Err("input SVD requires finite values and at least input-width training rows".into());
        }
        let x = gam_linalg::decompose::svd(inputs.view(), false).map_err(|e| e.to_string())?;
        if x.singular_values.len() != inputs.ncols() || x.singular_values.iter().any(|s| !s.is_finite() || *s <= x.band)
            || x.vt.iter().any(|v| !v.is_finite())
        {
            return Err("data-weighted SVD training inputs are not full column rank at the declared SVD resolution".into());
        }
        Ok(Self { vt: x.vt, singular_values: x.singular_values, training_rows: inputs.nrows() })
    }

    pub fn fit(&self, matrix: &Array2<f64>) -> Result<DataWeightedSvd, String> {
        if matrix.nrows() == 0 || matrix.ncols() != self.singular_values.len() || matrix.iter().any(|v| !v.is_finite()) {
            return Err("data-weighted SVD requires a finite matrix matching the training input width".into());
        }
        // X = U S V^T. Orthogonality of U makes the weighted error equal to
        // ||(W - W_r) V S||_F. Truncate W V S, then undo S and V on the right.
        let weighted = matrix.dot(&self.vt.t()) * &self.singular_values;
        if weighted.iter().any(|v| !v.is_finite()) { return Err("weighted map overflowed before SVD".into()); }
        let fit = gam_linalg::decompose::svd(weighted.view(), false).map_err(|e| e.to_string())?;
        let right = (&fit.vt / &self.singular_values).dot(&self.vt);
        if right.iter().chain(fit.u.iter()).chain(fit.singular_values.iter()).any(|v| !v.is_finite()) {
            return Err("data-weighted SVD unwhitening produced a nonfinite factor".into());
        }
        Ok(DataWeightedSvd {
            fit: HeadSvd { u: fit.u, singular_values: fit.singular_values, vt: right },
            training_rows: self.training_rows,
            input_width: self.singular_values.len(),
            smallest_input_singular_value: self.singular_values.iter().copied().fold(f64::INFINITY, f64::min),
            largest_input_singular_value: self.singular_values.iter().copied().fold(0.0, f64::max),
        })
    }
}

impl DataWeightedSvd {
    pub fn new(matrix: &Array2<f64>, inputs: &Array2<f64>) -> Result<Self, String> {
        InputSvd::new(inputs)?.fit(matrix)
    }

    /// Explicit factors are projected to f32 and priced normally; no training data
    /// or input covariance is an implicit dependency of this executable operator.
    pub fn operator(&self, original: &Operator, name: String, rank: usize) -> Result<Operator, String> {
        self.fit.operator(original, name, rank, "native-input-weighted linear reconstruction proposal; decoded fidelity not established")
    }
}

/// Transport for a fitted proposal, not an accepted or independently executable
/// explanation. It contains only the explicit f32 factors; training activations
/// are unnecessary at evaluation time. The receiving artifact still supplies and
/// pays for its interfaces, binding, graph and remaining native computation.
/// Integers carry the f32 bit patterns, including signed zero, without a second
/// floating-point text conversion. File digests and checkpoint lineage belong in
/// the experiment manifest, not in this numerical payload.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LowRankProposal {
    version: u32,
    name: String,
    rows: usize,
    cols: usize,
    rank: usize,
    left_f32_bits: Vec<u32>,
    right_f32_bits: Vec<u32>,
}

impl LowRankProposal {
    pub fn of(operator: &Operator) -> Result<Self, String> {
        let super::operator_program::OperatorBody::LowRank { left, right, .. } = &operator.body else {
            return Err("prepared proposal requires explicit low-rank factors".into());
        };
        let encode = |values: &Array2<f64>| -> Result<Vec<u32>, String> {
            values.iter().map(|&v| {
                let f = v as f32;
                if !v.is_finite() || f64::from(f).to_bits() != v.to_bits() {
                    Err("prepared factor is not an exact finite f32 literal".into())
                } else { Ok(f.to_bits()) }
            }).collect()
        };
        Ok(Self { version: 1, name: operator.name.clone(), rows: left.nrows(), cols: right.ncols(),
            rank: left.ncols(), left_f32_bits: encode(left)?, right_f32_bits: encode(right)? })
    }

    /// Reconstruct on the declared native interfaces; does not fit or consult any
    /// evaluation states. A caller must verify the surrounding manifest first.
    pub fn operator(&self, native: &Operator) -> Result<Operator, String> {
        if self.version != 1 || self.rows != native.rows.width() || self.cols != native.cols.width()
            || self.rank == 0 || self.rank > self.rows.min(self.cols)
            || self.rows.checked_mul(self.rank) != Some(self.left_f32_bits.len())
            || self.rank.checked_mul(self.cols) != Some(self.right_f32_bits.len())
        { return Err("prepared factor version, shape or native interface mismatch".into()); }
        let values = |bits: &[u32]| -> Result<Vec<f64>, String> {
            bits.iter().map(|&b| {
                let v = f32::from_bits(b);
                if v.is_finite() { Ok(f64::from(v)) } else { Err("nonfinite prepared factor".into()) }
            }).collect()
        };
        let left = Array2::from_shape_vec((self.rows,self.rank), values(&self.left_f32_bits)?).map_err(|e|e.to_string())?;
        let right = Array2::from_shape_vec((self.rank,self.cols), values(&self.right_f32_bits)?).map_err(|e|e.to_string())?;
        let precision = exact_precision(left.iter().chain(right.iter()).copied()).map_err(|e|e.to_string())?;
        let mut operator=Operator::low_rank(self.name.clone(),native.rows.clone(),native.cols.clone(),left.clone(),right.clone(),precision,
            Provenance::derived(&[&native.provenance],"loaded explicit f32 proposal factors; decoded Local/Run fidelity not established".into()))
            .map_err(|e|e.to_string())?;
        // The constructor validates interfaces and lattice membership but its
        // rounding canonicalizes signed zero. These finite f32 literals already
        // lie exactly on `precision`; retain their transported bits after validation.
        operator.body=super::operator_program::OperatorBody::LowRank {left,right,precision};
        Ok(operator)
    }
}

struct CopyResidualHead {
    derived: super::artifact::Derived,
    operator: std::sync::Arc<Operator>,
    residual: HeadSvd,
}

#[derive(Default)]
struct HeadApproxCache {
    copy: std::sync::OnceLock<Result<CopyResidualHead, String>>,
    native: std::sync::OnceLock<Result<HeadSvd, String>>,
}

/// Lazy, complete paired bank over all heads and explicitly declared positive ranks.
/// The native evaluator adds one candidate. No full-model bank is materialized here.
/// Copy is a priced derived operator; its residual is a separately priced LowRank
/// operator at the same native attention output, retaining every intervention place.
/// Full rank is only a numerical reconstruction endpoint, not an exactness certificate.
pub struct CopyResidualBank<'a> {
    start: &'a Artifact,
    rules: HeadRules,
    targets: Vec<(usize, usize)>,
    ranks: Vec<usize>,
    cache: Vec<HeadApproxCache>,
    /// Includes the evaluator's native candidate.
    pub candidate_count: usize,
}

/// Staged host timings, excluding acceptance metrics. Decoding is independent;
/// the decoded-cost cache is temporary so it cannot retain full decoded models.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct HeadCheckTimings {
    pub compose_f32_seconds: f64,
    pub coverage_before_seconds: f64,
    pub encode_seconds: f64,
    pub decode_seconds: f64,
    pub coverage_decoded_seconds: f64,
    pub reencode_seconds: f64,
    pub cost_before_seconds: f64,
    pub cost_decoded_seconds: f64,
}

impl<'a> CopyResidualBank<'a> {
    /// Complete cardinality, checked before caches, SVDs, or artifact copies.
    pub fn cardinality(heads: impl IntoIterator<Item = usize>, ranks: &[usize]) -> Result<usize, String> {
        if ranks.is_empty() || ranks.contains(&0) || ranks.iter().collect::<std::collections::BTreeSet<_>>().len() != ranks.len() {
            return Err("declare nonempty distinct positive head ranks".into());
        }
        let mut count = 0usize;
        for heads in heads {
            if heads == 0 { return Err("a layer has no attention heads".into()); }
            count = count.checked_add(heads).ok_or("head bank size overflow")?;
        }
        if count == 0 { return Err("head bank has no layers".into()); }
        count.checked_mul(ranks.len()).and_then(|n| n.checked_mul(2)).and_then(|n| n.checked_add(1)).ok_or_else(|| "head bank size overflow".into())
    }

    pub fn new(start: &'a Artifact, layers: &[LayerNodes], group: usize, ranks: &[usize], max_bank: usize) -> Result<Self, String> {
        let candidate_count = Self::cardinality(layers.iter().map(|l| l.queries.len()), ranks)?;
        if candidate_count > max_bank { return Err(format!("complete paired head bank needs {candidate_count} including native, max_bank={max_bank}")); }
        if group == 0 || !start.blocks.is_empty() || !start.derived.is_empty() || !start.exceptions.is_empty()
            || start.native_nodes != start.program.nodes.len() || !start.places.iter().copied().eq((0..start.native_nodes).map(|n| (n,n)))
            || !start.has_f32_literals()
        { return Err("paired head bank requires a native f32 artifact, all native places, and positive query/KV group".into()); }
        let mut targets = Vec::new();
        for (layer, nodes) in layers.iter().enumerate() {
            let heads = nodes.queries.len();
            if heads % group != 0 || nodes.values.len() != heads/group || nodes.keys.len() != heads/group || nodes.reads.len() != heads {
                return Err("inconsistent head counts in paired bank".into());
            }
            let Some(Node::Affine { terms, .. }) = start.program.nodes.get(nodes.attention) else { return Err("native attention output must be affine".into()); };
            for head in 0..heads {
                let o = operator(start, &format!("blocks.{layer}.o{head}"))?;
                let op = &start.program.operators[o];
                if ranks.iter().any(|r| *r > op.rows.width().min(op.cols.width())) { return Err(format!("declared rank exceeds blocks.{layer}.o{head} dimensions")); }
                if terms.iter().filter(|(_, index)| *index == o).count() != 1
                    || !terms.iter().any(|&(read, index)| index == o && read == nodes.reads[head]) {
                    return Err("native head needs one affine term at its declared attention output".into());
                }
                targets.push((layer, head));
            }
        }
        let cache = (0..targets.len()).map(|_| HeadApproxCache::default()).collect();
        Ok(Self { start, rules: HeadRules { layers: layers.to_vec(), targets: Vec::new(), group, firsts: Default::default() }, targets,
            ranks: ranks.to_vec(), cache, candidate_count })
    }

    /// Every head, rank and family, in declared canonical order, regardless of fidelity.
    pub fn choices(&self) -> impl Iterator<Item = CopyResidualChoice> + '_ {
        self.targets.iter().flat_map(move |&(layer, head)| self.ranks.iter().flat_map(move |&rank|
            [HeadApproximation::CopyResidual, HeadApproximation::NativeSvd].into_iter().map(move |family| CopyResidualChoice { layer, head, rank, family })))
    }

    pub fn candidate(&self, choice: CopyResidualChoice) -> Result<Artifact, String> {
        let CopyResidualChoice { layer, head, rank, family } = choice;
        if !self.ranks.contains(&rank) { return Err("rank outside declared paired bank".into()); }
        let index = self.targets.iter().position(|t| *t == (layer, head)).ok_or("head outside declared paired bank")?;
        let o = operator(self.start, &format!("blocks.{layer}.o{head}"))?;
        let original = &self.start.program.operators[o];
        let mut out = self.start.clone();
        match family {
            HeadApproximation::NativeSvd => {
                let svd = self.cache[index].native.get_or_init(|| HeadSvd::of(&original.matrix())).as_ref().map_err(|e| e.clone())?;
                out.program.operators[o] = std::sync::Arc::new(svd.operator(original, original.name.clone(), rank, "ordinary native operator SVD")?);
            }
            HeadApproximation::CopyResidual => {
                let copied = self.cache[index].copy.get_or_init(|| {
                    let artifact = self.rules.copy(self.start, layer, head)?.candidate;
                    let derived = artifact.derived.into_iter().next().ok_or("Copy head has no derivation")?;
                    let op = artifact.program.operators[o].clone();
                    let residual = HeadSvd::of(&(original.matrix() - &op.matrix()))?;
                    Ok(CopyResidualHead { derived, operator: op, residual })
                }).as_ref().map_err(|e| e.clone())?;
                out.program.operators[o] = copied.operator.clone();
                out.derived.push(copied.derived.clone());
                let residual = copied.residual.operator(original, format!("blocks.{layer}.o{head}.copy_residual"), rank, "SVD of native O minus decoded Copy prediction")?;
                let residual_index = out.program.operators.len();
                out.program.operators.push(std::sync::Arc::new(residual));
                let Node::Affine { terms, .. } = &mut out.program.nodes[self.rules.layers[layer].attention] else { return Err("attention is not affine".into()); };
                let position = terms.iter().position(|(_, index)| *index == o).ok_or("head term lost")?;
                let read = terms[position].0;
                terms.insert(position + 1, (read, residual_index));
            }
        }
        out.bind(&format!("attention {layer}"), &[self.rules.layers[layer].normed_stream], self.rules.layers[layer].attention)
    }

    /// Install previously fitted explicit factors without requiring the training
    /// inputs. The declared family supplies the same native graph and bindings;
    /// Copy additionally carries its full matrix-rule body rather than a free
    /// decoder template. Its final C32 must therefore be measured anew.
    ///
    /// No factorization is performed here. Copy's scalar remains determined by
    /// the native weights, independently of any evaluation state.
    pub fn candidate_with_prepared_factors(&self, choice: CopyResidualChoice, prepared: &LowRankProposal) -> Result<Artifact,String> {
        if choice.rank != prepared.rank { return Err("prepared factor rank differs from declared candidate".into()); }
        if !self.ranks.contains(&choice.rank) { return Err("rank outside declared paired bank".into()); }
        if !self.targets.contains(&(choice.layer,choice.head)) { return Err("head outside declared paired bank".into()); }
        let native_index=operator(self.start,&format!("blocks.{}.o{}",choice.layer,choice.head))?;
        let loaded=prepared.operator(&self.start.program.operators[native_index])?;
        let nodes=&self.rules.layers[choice.layer];
        let mut candidate=match choice.family {
            HeadApproximation::NativeSvd => self.start.clone(),
            HeadApproximation::CopyResidual => self.rules.copy(self.start,choice.layer,choice.head)?.candidate,
        };
        match choice.family {
            HeadApproximation::NativeSvd => candidate.program.operators[native_index]=std::sync::Arc::new(loaded),
            HeadApproximation::CopyResidual => {
                let residual=candidate.program.operators.len();
                candidate.program.operators.push(std::sync::Arc::new(loaded));
                let Node::Affine {terms,..}=&mut candidate.program.nodes[nodes.attention] else {return Err("attention is not affine".into());};
                let position=terms.iter().position(|&(read,op)|read==nodes.reads[choice.head] && op==native_index).ok_or("prepared native head term is missing")?;
                terms.insert(position+1,(nodes.reads[choice.head],residual));
            }
        }
        candidate.bind(&format!("attention {}",choice.layer),&[nodes.normed_stream],nodes.attention)?.expand_copy_templates()
    }

    /// Compose a declared set of prepared head replacements into one autonomous
    /// program. No singleton-fidelity filter is applied. Native head order fixes
    /// summation order, and each affected layer receives one complete binding.
    /// Sources of rule calls must remain native and independent of every changed
    /// output. The resulting multi-layer program needs its own Local/Run checks.
    pub fn compose_with_prepared_factors(&self, parts: &[(CopyResidualChoice,&LowRankProposal)]) -> Result<Artifact,String> {
        let mut ordered=std::collections::BTreeMap::new();
        for &(choice,prepared) in parts {
            if ordered.insert((choice.layer,choice.head),(choice,prepared)).is_some() {
                return Err("duplicate prepared head in joint program".into());
            }
        }
        let mut out=self.start.clone();
        let mut changed=std::collections::BTreeSet::new();
        let mut layers=std::collections::BTreeSet::new();
        for (_, (choice,prepared)) in ordered {
            let piece=self.candidate_with_prepared_factors(choice,prepared)?;
            let o=operator(self.start,&format!("blocks.{}.o{}",choice.layer,choice.head))?;
            changed.insert(o);
            layers.insert(choice.layer);
            out.program.operators[o]=piece.program.operators[o].clone();
            if choice.family==HeadApproximation::CopyResidual {
                if piece.derived.len()!=1 || piece.derived[0].operator!=o {
                    return Err("prepared Copy must have exactly its declared output derivation".into());
                }
                out.derived.push(piece.derived[0].clone());
                let residual=piece.program.operators.last().ok_or("prepared Copy residual absent")?.clone();
                let residual_index=out.program.operators.len();
                out.program.operators.push(residual);
                let nodes=&self.rules.layers[choice.layer];
                let Node::Affine {terms,..}=&mut out.program.nodes[nodes.attention] else {
                    return Err("joint attention output is not affine".into());
                };
                let position=terms.iter().position(|&(read,index)|index==o && read==nodes.reads[choice.head]).ok_or("joint native head read is missing")?;
                terms.insert(position+1,(nodes.reads[choice.head],residual_index));
            }
        }
        if out.derived.iter().any(|d|d.law.sources().iter().any(|s|changed.contains(s))) {
            return Err("joint rule reads another changed output; native independence not established".into());
        }
        for layer in layers {
            let nodes=&self.rules.layers[layer];
            out=out.bind(&format!("attention {layer}"),&[nodes.normed_stream],nodes.attention)?;
        }
        Ok(out)
    }

    /// Projection and exact decoded coverage/cost checks for a single lazy candidate.
    /// C32 charges factors, Copy scale, wiring, bindings and all remaining native code;
    /// exact wire byte count is separate from C32.
    pub fn checked(&self, choice: CopyResidualChoice) -> Result<(Artifact, super::acceptance::StructuralCost, usize), String> {
        self.checked_profiled(choice, &mut Default::default()).map(|(a,c,w,_)| (a,c,w))
    }

    /// A caller may retain the immutable original-operator cost cache across lazy
    /// candidates. Independently decoded operators use a fresh temporary cache,
    /// preventing accumulation of a full native model for every decoded candidate.
    /// No coverage, roundtrip or cost-equivalence check is bypassed.
    pub fn checked_profiled(&self, choice: CopyResidualChoice, cache: &mut super::acceptance::CostCache)
        -> Result<(Artifact, super::acceptance::StructuralCost, usize, HeadCheckTimings), String>
    {
        let mut timing = HeadCheckTimings::default();
        let started = std::time::Instant::now();
        let artifact = self.candidate(choice)?.f32_literals()?;
        timing.compose_f32_seconds = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        artifact.validate_coverage(&self.start.program)?;
        timing.coverage_before_seconds = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        let bytes = artifact.to_bytes()?;
        timing.encode_seconds = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        let decoded = Artifact::from_bytes(&bytes, &self.start.program.declarations)?;
        timing.decode_seconds = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        decoded.validate_coverage(&self.start.program)?;
        if !decoded.has_f32_literals() || decoded.places != self.start.places { return Err("paired head bank lost literal projection or places".into()); }
        timing.coverage_decoded_seconds = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        if decoded.to_bytes()? != bytes { return Err("paired head bank lost canonical roundtrip".into()); }
        timing.reencode_seconds = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        let cost = super::acceptance::structural_cost(&artifact, cache)?;
        timing.cost_before_seconds = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        let decoded_cost = super::acceptance::structural_cost(&decoded, &mut Default::default())?;
        timing.cost_decoded_seconds = started.elapsed().as_secs_f64();
        if cost != decoded_cost { return Err("decoded paired head C32 differs".into()); }
        Ok((decoded, cost, bytes.len(), timing))
    }

}

/// Paired Copy-law/residual and ordinary SVD proposals on immutable imported
/// attention graphs. Source layer IDs are explicit; all mapped heads are kept.
/// Local binds the native merged post-attention residual, preserving skip, GQA,
/// RoPE, Q/K normalization, biases outside QKV, and every intervention place.
/// The denominator is `attention_map::LOCAL_DENOMINATOR`, unlike split VPD sites.
pub struct MappedCopyResidualBank<'a> {
    start: &'a Artifact,
    maps: Vec<super::attention_map::AttentionLayerMap>,
    targets: Vec<(usize, usize)>,
    ranks: Vec<usize>,
    cache: Vec<HeadApproxCache>,
    pub candidate_count: usize,
}

impl<'a> MappedCopyResidualBank<'a> {
    /// Map every head of each explicitly named native layer. Unsupported semantic
    /// layouts are errors, never omitted alternatives. No SVD or Copy fit occurs.
    pub fn new(start: &'a Artifact, native_layers: &[usize], ranks: &[usize], max_bank: usize) -> Result<Self, String> {
        if native_layers.is_empty() || native_layers.iter().collect::<std::collections::BTreeSet<_>>().len() != native_layers.len() {
            return Err("declare nonempty distinct native layer IDs".into());
        }
        let maps = native_layers.iter().map(|&layer| super::attention_map::AttentionLayerMap::of(&start.program, layer))
            .collect::<Result<Vec<_>, _>>()?;
        let candidate_count = CopyResidualBank::cardinality(maps.iter().map(|m| m.heads.len()), ranks)?;
        if candidate_count > max_bank { return Err(format!("complete mapped paired bank needs {candidate_count} including native, max_bank={max_bank}")); }
        if !start.blocks.is_empty() || !start.derived.is_empty() || !start.exceptions.is_empty()
            || start.native_nodes != start.program.nodes.len() || !start.places.iter().copied().eq((0..start.native_nodes).map(|n| (n,n)))
            || !start.has_f32_literals() { return Err("mapped paired bank requires native f32 literals and all original places".into()); }
        let mut targets = Vec::new();
        for (index, map) in maps.iter().enumerate() {
            for h in &map.heads {
                let op = &start.program.operators[h.output_operator];
                if ranks.iter().any(|&rank| rank > op.rows.width().min(op.cols.width())) {
                    return Err(format!("rank exceeds native layer {} head {} output dimensions", map.native_layer, h.head));
                }
                targets.push((index, h.head));
            }
        }
        let cache = (0..targets.len()).map(|_| HeadApproxCache::default()).collect();
        Ok(Self { start, maps, targets, ranks: ranks.to_vec(), cache, candidate_count })
    }

    pub fn choices(&self) -> impl Iterator<Item = CopyResidualChoice> + '_ {
        self.targets.iter().flat_map(move |&(map,head)| self.ranks.iter().flat_map(move |&rank|
            [HeadApproximation::CopyResidual,HeadApproximation::NativeSvd].into_iter().map(move |family|
                CopyResidualChoice { layer:self.maps[map].native_layer,head,rank,family })))
    }

    pub fn candidate(&self, choice: CopyResidualChoice) -> Result<Artifact,String> {
        let CopyResidualChoice { layer,head,rank,family } = choice;
        if !self.ranks.contains(&rank) { return Err("rank outside declared mapped bank".into()); }
        let index = self.targets.iter().position(|&(m,h)| self.maps[m].native_layer == layer && h == head)
            .ok_or("native layer/head outside declared mapped bank")?;
        let map = &self.maps[self.targets[index].0];
        let h = &map.heads[head];
        let o = h.output_operator;
        let original = &self.start.program.operators[o];
        let mut out = self.start.clone();
        match family {
            HeadApproximation::NativeSvd => {
                let svd = self.cache[index].native.get_or_init(|| HeadSvd::of(&original.matrix())).as_ref().map_err(|e|e.clone())?;
                out.program.operators[o] = std::sync::Arc::new(svd.operator(original,original.name.clone(),rank,"ordinary mapped native operator SVD")?);
            }
            HeadApproximation::CopyResidual => {
                let copied = self.cache[index].copy.get_or_init(|| {
                    let value = self.start.program.operators[h.value_operator].matrix();
                    let super::operator_program::OperatorBody::Diagonal { values:gain,.. } = &self.start.program.operators[map.input_gain].body else { return Err("mapped input gain is not diagonal".into()); };
                    let super::operator_program::OperatorBody::Diagonal { values:final_gain,.. } = &self.start.program.operators[map.final_gain].body else { return Err("mapped final gain is not diagonal".into()); };
                    let prediction = super::rules::copy_prediction(&value,gain,final_gain)?;
                    let scale = super::rules::copy_scale(&original.matrix(),&value,&prediction) as f32;
                    let derived = self.start.derive(o,map.copy_law(head)?,scale,Vec::new())?;
                    let operator = derived.program.operators[o].clone();
                    let residual = HeadSvd::of(&(original.matrix()-&operator.matrix()))?;
                    Ok(CopyResidualHead { derived:derived.derived.into_iter().next().ok_or("mapped Copy has no derivation")?,operator,residual })
                }).as_ref().map_err(|e|e.clone())?;
                out.program.operators[o] = copied.operator.clone();
                out.derived.push(copied.derived.clone());
                let residual = copied.residual.operator(original,format!("blocks.{layer}.o{head}.copy_residual"),rank,"SVD of mapped native O minus decoded Copy prediction")?;
                let residual_index = out.program.operators.len();
                out.program.operators.push(std::sync::Arc::new(residual));
                let Node::Affine { terms,.. } = &mut out.program.nodes[map.output] else { return Err("mapped output lost affine structure".into()); };
                let position = terms.iter().position(|&(read,index)| read == h.read && index == o).ok_or("mapped head term lost")?;
                terms.insert(position+1,(h.read,residual_index));
            }
        }
        map.bind(&out)
    }
}

#[cfg(test)]
mod data_weighted_svd_tests {
    use super::*;

    fn original(w: &Array2<f64>) -> Operator {
        dense("native", &Interface::native(w.nrows()).expect("rows"),
            &Interface::native(w.ncols()).expect("columns"), w.clone()).expect("dense operator")
    }

    #[test]
    fn weighted_proposal_solves_the_observed_linear_error_not_weight_error() {
        let w = ndarray::array![[2.0, 0.0], [0.0, 1.0]];
        let x = ndarray::array![[1.0, 0.0], [0.0, 10.0]];
        let source = original(&w);
        let weighted = DataWeightedSvd::new(&w, &x).expect("full rank training");
        let proposal = weighted.operator(&source, "weighted".into(), 1).expect("rank one");
        let unweighted = HeadSvd::of(&w).expect("weight SVD")
            .operator(&source, "plain".into(), 1, "weight-only control").expect("control");
        let loss = |a: &Operator| x.dot(&(&w - &a.matrix()).t()).mapv(|v| v*v).sum();
        assert!((loss(&proposal) - 4.0).abs() < 1e-10);
        assert!((loss(&unweighted) - 100.0).abs() < 1e-10);
        assert_eq!(proposal.real_count(), unweighted.real_count());
        assert_eq!(proposal.real_count(), 4);
        assert_eq!(weighted.training_rows, 2);
        assert_eq!(weighted.smallest_input_singular_value, 1.0);
        assert_eq!(weighted.largest_input_singular_value, 10.0);
        let full = weighted.operator(&source, "full".into(), 2).expect("full rank");
        assert!(full.matrix().iter().zip(&w).all(|(a,b)| (a-b).abs() < 1e-6));
    }

    #[test]
    fn weighted_proposal_is_independent_of_an_orthogonal_input_coordinate_change() {
        let w = ndarray::array![[2.0, 0.0], [0.0, 1.0], [1.0, -1.0]];
        let x = ndarray::array![[1.0, 0.0], [0.0, 10.0], [2.0, 3.0]];
        let rotation = ndarray::array![[0.6, -0.8], [0.8, 0.6]];
        let wx = w.dot(&rotation);
        let xx = x.dot(&rotation);
        let a = DataWeightedSvd::new(&w, &x).expect("fit")
            .operator(&original(&w), "a".into(), 1).expect("operator");
        let b = DataWeightedSvd::new(&wx, &xx).expect("rotated fit")
            .operator(&original(&wx), "b".into(), 1).expect("rotated operator");
        let y = x.dot(&a.matrix().t());
        let rotated_y = xx.dot(&b.matrix().t());
        assert!(y.iter().zip(&rotated_y).all(|(a,b)| (a-b).abs() < 1e-5));
    }

    #[test]
    fn unobserved_directions_and_nonfinite_training_are_explicit_failures() {
        let w = ndarray::array![[2.0, 0.0], [0.0, 1.0]];
        assert!(DataWeightedSvd::new(&w, &ndarray::array![[1.0, 0.0], [2.0, 0.0]]).is_err());
        assert!(DataWeightedSvd::new(&w, &ndarray::array![[1.0, 0.0]]).is_err());
        assert!(DataWeightedSvd::new(&w, &ndarray::array![[1.0, 0.0], [0.0, f64::NAN]]).is_err());
        assert!(DataWeightedSvd::new(&ndarray::array![[f64::INFINITY, 0.0]], &w).is_err());
    }

    #[test]
    fn shared_input_factorization_preserves_factors_for_distinct_proposals() {
        let x=ndarray::array![[1.,2.],[3.,-1.],[4.,5.]];
        let shared=InputSvd::new(&x).expect("full rank inputs");
        for matrix in [ndarray::array![[2.,1.],[0.,-3.]],ndarray::array![[1.,-4.],[2.,2.]]] {
            let source=original(&matrix);
            let a=shared.fit(&matrix).expect("shared fit").operator(&source,"a".into(),1).expect("a");
            let b=DataWeightedSvd::new(&matrix,&x).expect("independent fit").operator(&source,"b".into(),1).expect("b");
            assert_eq!(a.matrix(),b.matrix());
            assert_eq!(a.real_count(),b.real_count());
        }
        assert!(shared.fit(&ndarray::Array2::zeros((2,3))).is_err());
        let scaled=InputSvd::new(&ndarray::array![[2.,0.],[0.,2.]]).expect("scaled inputs");
        assert!(scaled.fit(&ndarray::array![[f64::MAX,0.]]).is_err());
    }

    #[test]
    fn prepared_factors_replay_without_training_inputs_and_reject_malformed_payloads() {
        let w=ndarray::array![[2.,1.],[0.,-3.]];
        let native=original(&w);
        let fitted=DataWeightedSvd::new(&w,&ndarray::array![[1.,2.],[3.,-1.],[4.,5.]])
            .expect("fit").operator(&native,"prepared".into(),1).expect("proposal");
        let payload=LowRankProposal::of(&fitted).expect("explicit factors");
        let bytes=serde_json::to_vec(&payload).expect("serialize");
        let replay:LowRankProposal=serde_json::from_slice(&bytes).expect("read");
        let loaded=replay.operator(&native).expect("reconstruct without inputs");
        assert_eq!(loaded.body,fitted.body);
        assert_eq!(loaded.matrix(),fitted.matrix());
        assert_eq!(loaded.real_count(),fitted.real_count());
        let mut changed=replay.clone(); changed.left_f32_bits[0]=(-0.0_f32).to_bits();
        let zero=changed.operator(&native).expect("signed zero");
        assert_eq!(LowRankProposal::of(&zero).expect("reencode").left_f32_bits[0],(-0.0_f32).to_bits());
        changed.version=2; assert!(changed.operator(&native).is_err());
        changed=replay.clone(); changed.right_f32_bits.pop(); assert!(changed.operator(&native).is_err());
        changed=replay.clone(); changed.left_f32_bits[0]=f32::INFINITY.to_bits(); assert!(changed.operator(&native).is_err());
        assert!(replay.operator(&original(&Array2::zeros((3,2)))).is_err());
        assert!(LowRankProposal::of(&native).is_err());
        let mut nonf32=fitted.clone();
        if let super::super::operator_program::OperatorBody::LowRank { left,.. }=&mut nonf32.body { left[[0,0]]=1.0+f64::EPSILON; }
        assert!(LowRankProposal::of(&nonf32).is_err());
    }
}

#[cfg(test)]
#[path = "mapped_copy_residual_tests.rs"]
mod mapped_copy_residual_tests;

impl HeadRules {
    /// The proposer for `targets`' heads, with every match's content planes found on `start`.
    pub fn new(start: &Artifact, layers: Vec<LayerNodes>, targets: Vec<usize>, group: usize) -> Result<Self, String> {
        use rayon::prelude::*;
        let heads = layers.first().map_or(0, |l| l.queries.len());
        let pairs: Vec<(usize, usize, usize, usize)> = targets
            .iter()
            .flat_map(|&l| (0..heads).flat_map(move |h| (0..l).flat_map(move |sl| (0..heads).map(move |sh| (l, h, sl, sh)))))
            .collect();
        let found: Vec<((usize, usize, usize, usize), usize)> = pairs
            .par_iter()
            .map(|&(l, h, sl, sh)| -> Result<_, String> {
                let ops = &start.program.operators;
                let matrix = |name: String| -> Result<Array2<f64>, String> { Ok(ops[operator(start, &name)?].matrix()) };
                let (query, key) = (matrix(format!("blocks.{l}.q{h}"))?, matrix(format!("blocks.{l}.k{}", h / group))?);
                let (output, value) = (matrix(format!("blocks.{sl}.o{sh}"))?, matrix(format!("blocks.{sl}.v{}", sh / group))?);
                let (_, gain) = Self::gain(start, &format!("blocks.{l}.rms1.gain"))?;
                let (_, source_gain) = Self::gain(start, &format!("blocks.{sl}.rms1.gain"))?;
                let width = query.nrows();
                let mut best: Option<(f64, usize)> = None;
                for first in 0..width / 2 {
                    let rows = super::rules::content_rows(width, first);
                    let reading = super::rules::match_reading(&key, &output, &value, &gain, &source_gain, &rows);
                    // The share of the head's circuit the rule carries on these rows, not their cosine:
                    // the cosine favours the single cleanest plane (2-6 of 128 rows on VPD 4L).
                    let energy = super::rules::match_energy(&query, &reading, &gain, &rows)?;
                    if best.is_none_or(|b| energy > b.0) {
                        best = Some((energy, first));
                    }
                }
                Ok(((l, h, sl, sh), best.ok_or("a head of no planes")?.1))
            })
            .collect::<Result<_, String>>()?;
        Ok(Self { layers, targets, group, firsts: found.into_iter().collect() })
    }

    /// Every single-head Copy substitution, independently derived from `start`.
    ///
    /// This finite bank does not search match planes or prune by feasibility. `max_bank`
    /// includes the native candidate that the evaluator adds. The iterator keeps only one
    /// candidate alive at a time; serializing a bank does not clone all native operators
    /// into memory at once. Each candidate retains the complete derivation and binding.
    pub fn copy_bank<'a>(
        start: &'a Artifact,
        layers: &[LayerNodes],
        group: usize,
        max_bank: usize,
    ) -> Result<impl Iterator<Item = Result<Proposal, String>> + 'a, String> {
        if layers.is_empty() || group == 0 || !start.derived.is_empty() || !start.blocks.is_empty() || !start.exceptions.is_empty() {
            return Err("copy bank requires a native artifact, nonempty layers and positive query/KV group".into());
        }
        let mut count = 0usize;
        for nodes in layers {
            let heads = nodes.queries.len();
            if heads == 0 || heads % group != 0 || nodes.values.len() != heads / group || nodes.keys.len() != heads / group {
                return Err("copy bank needs consistent query/key/value head counts".into());
            }
            count = count.checked_add(heads).ok_or("copy bank size overflow")?;
        }
        if count.checked_add(1).ok_or("copy bank size overflow")? > max_bank {
            return Err(format!("{count} single Copy substitutions plus native exceeds max_bank={max_bank}"));
        }
        let heads: Vec<(usize, usize)> = layers.iter().enumerate().flat_map(|(l, nodes)| (0..nodes.queries.len()).map(move |h| (l, h))).collect();
        let rules = Self { layers: layers.to_vec(), targets: Vec::new(), group, firsts: std::collections::BTreeMap::new() };
        Ok(heads.into_iter().map(move |(layer, head)| rules.copy(start, layer, head)))
    }

    fn gain(artifact: &Artifact, name: &str) -> Result<(usize, Array1<f64>), String> {
        let index = operator(artifact, name)?;
        let values = artifact.program.operators[index].diagonal().ok_or_else(|| format!("{name} is not a diagonal"))?;
        Ok((index, values))
    }

    fn copy(&self, artifact: &Artifact, layer: usize, head: usize) -> Result<Proposal, String> {
        let (o, v) = (operator(artifact, &format!("blocks.{layer}.o{head}"))?, operator(artifact, &format!("blocks.{layer}.v{}", head / self.group))?);
        let (g, gain) = Self::gain(artifact, &format!("blocks.{layer}.rms1.gain"))?;
        let (gf, final_gain) = Self::gain(artifact, "final_norm.gain")?;
        let (output, value) = (artifact.program.operators[o].matrix(), artifact.program.operators[v].matrix());
        let prediction = super::rules::copy_prediction(&value, &gain, &final_gain)?;
        let scale = super::rules::copy_scale(&output, &value, &prediction) as f32;
        let nodes = &self.layers[layer];
        let candidate = artifact
            .derive(o, OperatorLaw::Copy { value: v, gain: g, final_gain: gf }, scale, Vec::new())?
            .bind(&format!("attention {layer}"), &[nodes.normed_stream], nodes.attention)?;
        Ok(Proposal { source: "rules".to_string(), description: format!("copy blocks.{layer}.o{head}"), candidate })
    }

    fn matches(&self, artifact: &Artifact, layer: usize, head: usize, source: (usize, usize)) -> Result<Vec<Proposal>, String> {
        let (sl, sh) = source;
        let (q, k) = (operator(artifact, &format!("blocks.{layer}.q{head}"))?, operator(artifact, &format!("blocks.{layer}.k{}", head / self.group))?);
        let (so, sv) = (operator(artifact, &format!("blocks.{sl}.o{sh}"))?, operator(artifact, &format!("blocks.{sl}.v{}", sh / self.group))?);
        let (g, gain) = Self::gain(artifact, &format!("blocks.{layer}.rms1.gain"))?;
        let (gs, source_gain) = Self::gain(artifact, &format!("blocks.{sl}.rms1.gain"))?;
        let ops = &artifact.program.operators;
        let (query, key, output, value) = (ops[q].matrix(), ops[k].matrix(), ops[so].matrix(), ops[sv].matrix());
        let width = query.nrows();
        let first = *self.firsts.get(&(layer, head, sl, sh)).ok_or("a match the proposer has no planes for")?;
        let rows = super::rules::content_rows(width, first);
        let reading = super::rules::match_reading(&key, &output, &value, &gain, &source_gain, &rows);
        let prediction = super::rules::match_prediction(&reading, &gain, None)?;
        let scale = super::rules::match_scale(&query, &reading, &gain, &rows, &prediction) as f32;
        let law = OperatorLaw::Match { key: k, source_output: so, source_value: sv, gain: g, source_gain: gs, first, directions: None };
        let content: std::collections::BTreeSet<usize> = rows.iter().copied().collect();
        let rest: Vec<(usize, Vec<f32>)> = (0..width).filter(|r| !content.contains(r)).map(|r| (r, query.row(r).iter().map(|v| *v as f32).collect())).collect();
        let nodes = &self.layers[layer];
        let block = format!("attention {layer}");
        let mut out = Vec::new();
        for (variant, residual) in [("the other rows kept", rest), ("the other rows zero", Vec::new())] {
            let candidate = artifact.derive(q, law.clone(), scale, residual)?.bind(&block, &[nodes.normed_stream], nodes.attention)?;
            out.push(Proposal {
                source: "rules".to_string(),
                description: format!("match blocks.{layer}.q{head} through blocks.{sl} head {sh} from plane {first}, {variant}"),
                candidate,
            });
        }
        Ok(out)
    }
}

impl Proposer for HeadRules {
    fn name(&self) -> &str {
        "attention head rules"
    }

    fn propose(&self, context: &Context<'_>) -> Result<Vec<Proposal>, String> {
        let current = context.current;
        let heads = self.layers.first().map_or(0, |l| l.queries.len());
        let derived: std::collections::BTreeSet<usize> = current.derived.iter().map(|d| d.operator).collect();
        let mut out = Vec::new();
        for &layer in &self.targets {
            for head in 0..heads {
                if !derived.contains(&operator(current, &format!("blocks.{layer}.o{head}"))?) {
                    out.push(self.copy(current, layer, head)?);
                }
                if derived.contains(&operator(current, &format!("blocks.{layer}.q{head}"))?) {
                    continue;
                }
                for sl in 0..layer {
                    for sh in 0..heads {
                        out.extend(self.matches(current, layer, head, (sl, sh))?);
                    }
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod copy_residual_tests {
    use super::*;
    use super::super::operator_program::{Declarations, FamilyInputs, OperatorProgram, Slot, SlotValues};
    use std::sync::Arc;

    fn fixture() -> (Artifact, Vec<LayerNodes>) {
        let (d, w) = (Interface::native(8).unwrap(), Interface::native(4).unwrap());
        let diag = |name: String| Operator::diag(name, d.clone(), Array1::ones(8), exact_precision([1.0]).unwrap(), Provenance::default()).unwrap();
        let mut operators = vec![Arc::new(diag("final_norm.gain".into()))];
        let mut nodes = vec![Node::Raw { slot: 0 }];
        let mut layers = Vec::new();
        let mut stream = 0;
        for layer in 0..4 {
            let gain = operators.len();
            operators.push(Arc::new(diag(format!("blocks.{layer}.rms1.gain"))));
            let normed = nodes.len();
            nodes.push(Node::Affine { terms: vec![(stream, gain)], bias: None });
            let mut reads = Vec::new();
            let mut terms = Vec::new();
            for head in 0..6 {
                let v = operators.len();
                let values = Array2::from_shape_fn((4,8), |(r,c)| if r==c { 1.0 } else { 0.0 });
                operators.push(Arc::new(dense(&format!("blocks.{layer}.v{head}"), &w, &d, values).unwrap()));
                let read = nodes.len(); nodes.push(Node::Affine { terms: vec![(normed,v)], bias: None }); reads.push(read);
                let o = operators.len();
                let values = Array2::from_shape_fn((8,4), |(r,c)| if r==c { 1.0 + (layer as f64)/8.0 + (head as f64)/4.0 + (c as f64)/16.0 }
                    else { ((((r+1)*(c+1)*(head+layer+1))%7) as f64 - 3.0)/16.0 });
                operators.push(Arc::new(dense(&format!("blocks.{layer}.o{head}"), &d, &w, values).unwrap()));
                terms.push((read,o));
            }
            let attention = nodes.len(); nodes.push(Node::Affine { terms, bias: None });
            layers.push(LayerNodes { stream, normed_stream: normed, queries: reads.clone(), keys: reads.clone(), values: reads.clone(), reads,
                attention, attended: attention, normed, pre: 0, active: 0, mlp: attention, residual: attention });
            stream = attention;
        }
        nodes.push(Node::Affine { terms: vec![(stream,0)], bias: None });
        let program = OperatorProgram { declarations: Declarations { domains: vec![], slots: vec![Slot::Raw { width: 8 }], parameters: 0 },
            bases: vec![], operators, rules: vec![], output: nodes.len()-1, nodes };
        (Artifact::native(&program).unwrap(), layers)
    }

    #[test]
    fn copy_residual_complete_count_guards_before_math_and_lazy_pairs() {
        assert_eq!(CopyResidualBank::cardinality([6;4], &[8,32,64,96]).unwrap(),193);
        assert!(CopyResidualBank::cardinality([usize::MAX], &[1,2]).is_err());
        assert!(CopyResidualBank::cardinality([6;4], &[1,1]).is_err());
        assert!(CopyResidualBank::cardinality([6;4], &[0]).is_err());
        let (start,layers)=fixture();
        let error = CopyResidualBank::new(&start,&layers,1,&[8,32,64,96],192).err().unwrap();
        assert!(error.contains("needs 193"), "budget guard precedes even matrix rank validation");
        let bank=CopyResidualBank::new(&start,&layers,1,&[1,2,3,4],193).unwrap();
        assert_eq!((bank.candidate_count,bank.choices().count()),(193,192));
        assert!(bank.cache.iter().all(|c| c.copy.get().is_none() && c.native.get().is_none()));
        let first=bank.choices().next().unwrap();
        bank.candidate(first).unwrap();
        assert!(bank.cache[0].copy.get().is_some());
        assert!(bank.cache[0].native.get().is_none());
        assert!(bank.cache[1..].iter().all(|c| c.copy.get().is_none() && c.native.get().is_none()));
    }

    #[test]
    fn prepared_head_factors_preserve_native_bindings_and_pay_copy_body() {
        let (start,layers)=fixture();
        let bank=CopyResidualBank::new(&start,&layers,1,&[2],49).unwrap();
        for family in [HeadApproximation::NativeSvd,HeadApproximation::CopyResidual] {
            let choice=CopyResidualChoice {layer:2,head:4,rank:2,family};
            let old=bank.candidate(choice).unwrap();
            let target=if family==HeadApproximation::NativeSvd {operator(&start,"blocks.2.o4").unwrap()} else {old.program.operators.len()-1};
            let saved=LowRankProposal::of(&old.program.operators[target]).unwrap();
            let loaded:LowRankProposal=serde_json::from_slice(&serde_json::to_vec(&saved).unwrap()).unwrap();
            let fresh=CopyResidualBank::new(&start,&layers,1,&[2],49).unwrap();
            let candidate=fresh.candidate_with_prepared_factors(choice,&loaded).unwrap();
            assert!(fresh.cache.iter().all(|c|c.copy.get().is_none() && c.native.get().is_none()));
            assert_eq!(candidate.program.nodes,old.program.nodes);
            assert_eq!(candidate.places,old.places);
            assert_eq!(candidate.blocks,old.blocks);
            assert_eq!(candidate.program.operators[target].body,old.program.operators[target].body);
            assert!(candidate.derived.iter().all(|d|matches!(d.law,OperatorLaw::Expression {..})));
            let bytes=candidate.to_bytes().unwrap();
            let decoded=Artifact::from_bytes(&bytes,&start.program.declarations).unwrap();
            assert_eq!(bytes,decoded.to_bytes().unwrap());
            let cost=super::super::acceptance::structural_cost(&decoded,&mut Default::default()).unwrap().total();
            let old_cost=super::super::acceptance::structural_cost(&old,&mut Default::default()).unwrap().total();
            if family==HeadApproximation::CopyResidual {assert!(cost>old_cost);} else {assert_eq!(cost,old_cost);}
            let mut bad=loaded;bad.rank=1;
            assert!(bank.candidate_with_prepared_factors(choice,&bad).is_err());
        }
    }

    #[test]
    fn joint_prepared_heads_preserve_all_native_places_and_share_paid_bodies() {
        let (start,layers)=fixture();
        let bank=CopyResidualBank::new(&start,&layers,1,&[2],49).unwrap();
        for family in [HeadApproximation::NativeSvd,HeadApproximation::CopyResidual] {
            let choices:Vec<_>=bank.choices().filter(|c|c.family==family).collect();
            let factors:Vec<_>=choices.iter().map(|&choice| {
                let p=bank.candidate(choice).unwrap();
                let target=if family==HeadApproximation::NativeSvd {operator(&start,&format!("blocks.{}.o{}",choice.layer,choice.head)).unwrap()} else {p.program.operators.len()-1};
                LowRankProposal::of(&p.program.operators[target]).unwrap()
            }).collect();
            let parts:Vec<_>=choices.iter().copied().zip(factors.iter()).collect();
            let joint=bank.compose_with_prepared_factors(&parts).unwrap();
            assert_eq!(joint.blocks.len(),4);
            assert_eq!(joint.places,start.places);
            assert_eq!(joint.native_nodes,start.native_nodes);
            joint.validate_coverage(&start.program).unwrap();
            let bytes=joint.to_bytes().unwrap();
            let decoded=Artifact::from_bytes(&bytes,&start.program.declarations).unwrap();
            assert_eq!(bytes,decoded.to_bytes().unwrap());
            let mut reverse=parts.clone();reverse.reverse();
            assert_eq!(bank.compose_with_prepared_factors(&reverse).unwrap().to_bytes().unwrap(),bytes);
            let native_cost=super::super::acceptance::structural_cost(&start,&mut Default::default()).unwrap();
            let cost=super::super::acceptance::structural_cost(&decoded,&mut Default::default()).unwrap();
            assert_eq!(cost.literals,native_cost.literals-24*32+24*12*2+if family==HeadApproximation::CopyResidual {24} else {0});
            if family==HeadApproximation::CopyResidual {
                assert_eq!(joint.derived.len(),24);
                let bodies:std::collections::BTreeSet<_>=joint.derived.iter().map(|d| {
                    let OperatorLaw::Expression {body,..}=&d.law else {panic!("explicit rule expected")};
                    let b=body.encode().unwrap();(b.len_bits(),b.packed_bytes().to_vec())
                }).collect();
                assert_eq!(bodies.len(),1);
                for nodes in &layers {
                    let Node::Affine {terms,..}=&joint.program.nodes[nodes.attention] else {panic!("affine")};
                    assert_eq!(terms.len(),12);
                }
            }
            assert!(bank.compose_with_prepared_factors(&[parts[0],parts[0]]).is_err());
        }
        assert_eq!(bank.compose_with_prepared_factors(&[]).unwrap().to_bytes().unwrap(),start.to_bytes().unwrap());
    }

    #[test]
    fn copy_residual_all_heads_ranks_keep_places_roundtrip_and_charge_every_literal() {
        let (start,layers)=fixture();
        let bank=CopyResidualBank::new(&start,&layers,1,&[1,2,3,4],193).unwrap();
        let native_cost=super::super::acceptance::structural_cost(&start,&mut Default::default()).unwrap();
        let mut cache=super::super::acceptance::CostCache::default();
        let mut visited=std::collections::BTreeSet::new();
        for choice in bank.choices() {
            let (candidate,cost,wire,timing)=bank.checked_profiled(choice,&mut cache).unwrap();
            visited.insert((choice.layer,choice.head,choice.rank));
            assert_eq!(candidate.places,start.places);
            assert_eq!(candidate.program.nodes.len(),start.program.nodes.len());
            assert_eq!(candidate.blocks.len(),1);
            assert_eq!(candidate.blocks[0].native_write,layers[choice.layer].attention);
            assert!(wire>0 && timing.decode_seconds.is_finite());
            let added=usize::from(choice.family==HeadApproximation::CopyResidual);
            assert_eq!(cost.literals,native_cost.literals-32+12*choice.rank as u64+added as u64);
            assert_eq!(candidate.program.operators.len(),start.program.operators.len()+added);
            assert_eq!(candidate.derived.len(),added);
            assert!(candidate.derived.iter().all(|d|d.residual.is_empty()));
            if choice.rank==4 { assert!(cost.total()>native_cost.total(),"full rank factor representation is more expensive than native here"); }
        }
        assert_eq!(visited.len(),96);
    }

    #[test]
    fn copy_residual_full_rank_is_numerical_and_native_head_interventions_remain_live() {
        let (start,layers)=fixture();
        let bank=CopyResidualBank::new(&start,&layers,1,&[2,4],97).unwrap();
        let family=FamilyInputs { rows: 2,slots:vec![SlotValues::Raw(Array2::from_shape_fn((2,8),|(r,c)|((r+1)*(c+1)) as f64/16.0))],layout:None };
        let read=layers[3].reads[0];
        for kind in [HeadApproximation::CopyResidual,HeadApproximation::NativeSvd] {
            let (candidate,_,_)=bank.checked(CopyResidualChoice {layer:3,head:0,rank:4,family:kind}).unwrap();
            let o=operator(&start,"blocks.3.o0").unwrap();
            let mut reconstructed=candidate.program.operators[o].matrix();
            if kind==HeadApproximation::CopyResidual { reconstructed+=&candidate.program.operators.last().unwrap().matrix(); }
            let expected=start.program.operators[o].matrix();
            assert!(reconstructed.iter().zip(&expected).all(|(a,b)|(a-b).abs()<1e-5),"full-rank recovery checked after literal rounding, never assumed exact");
            let native=start.execute(&family).unwrap();
            let clean=candidate.execute(&family).unwrap();
            let patched=candidate.execute_edited(&family,|n,v,_| { if n==read { v.fill(0.0); } Ok(()) }).unwrap();
            let native_patch=start.execute_edited(&family,|n,v,_| { if n==read { v.fill(0.0); } Ok(()) }).unwrap();
            let out=candidate.program.output;
            assert!(clean.values[out].iter().zip(&native.values[out]).all(|(a,b)|(a-b).abs()<1e-5*b.abs().max(1.0)));
            assert!(patched.values[out].iter().zip(&native_patch.values[out]).all(|(a,b)|(a-b).abs()<1e-5*b.abs().max(1.0)));
            assert_ne!(clean.values[out],patched.values[out],"a native head activation edit changes the final output");
        }
        // A compressed residual must respond through both summands to the same native head read.
        let (candidate,_,_)=bank.checked(CopyResidualChoice {layer:3,head:0,rank:2,family:HeadApproximation::CopyResidual}).unwrap();
        let clean=candidate.execute(&family).unwrap();
        let patched=candidate.execute_edited(&family,|n,v,_| { if n==read { v.fill(0.0); } Ok(()) }).unwrap();
        let o=operator(&start,"blocks.3.o0").unwrap();
        let expected=candidate.program.operators[o].apply(&clean.values[read])+candidate.program.operators.last().unwrap().apply(&clean.values[read]);
        let delta=&clean.values[candidate.program.output]-&patched.values[candidate.program.output];
        assert!(delta.iter().zip(&expected).all(|(a,b)|(a-b).abs()<1e-8*b.abs().max(1.0)));
    }
}

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
//! `.reads.f64`, `.writes.f64` and `.offset.f64`).
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
    let with = |extension: &str| PathBuf::from(format!("{}.{extension}", base.display()));
    let text = std::fs::read_to_string(with("rules.json")).map_err(|e| format!("{}: {e}", with("rules.json").display()))?;
    let rules: Vec<AccountRule> = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let offset = if with("offset.f64").exists() { read_rows(&with("offset.f64"), d_out)?.row(0).to_owned() } else { Array1::zeros(d_out) };
    Ok(Account { reads: read_rows(&with("reads.f64"), d_in)?, rules, writes: read_rows(&with("writes.f64"), d_out)?, offset })
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

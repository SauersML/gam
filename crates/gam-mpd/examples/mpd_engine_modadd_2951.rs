//! The two-part-code decomposition engine on a trained modular-addition transformer (#2951).
//!
//! `mpd_engine_modadd_2951 EXPORT_DIR REPEATS [declared|recovered] [SCREENINGS CERTIFICATIONS]`
//!
//! `EXPORT_DIR` is what `bench/mpd_engine_export_2951.py` writes from a `bench/mpd_modadd_2951.py`
//! run file: float64 `.npy` tensors and `export.json`. The network is that benchmark's one layer at
//! the `=` position: `x_j = W_E[t_j] + W_pos[j]` for the tokens `(a, b, =)`, softmax attention from
//! `=` over the three positions (score scale `fl(1/fl(√d_head))`), the residual, a ReLU MLP and the
//! unembedding.
//!
//! The model is imported as an operator program with nothing but its native tensors: indicator
//! features of the tokens, one operator per tensor (tied where the network ties them: `W_E` and
//! each head's `W_K`, `W_V` at every position), and the native node laws. The contract declares only
//! the behaviour: every `(a, b)` in `Z_p²`, observed `r` times for each `r` of `REPEATS` (an ascending
//! comma-separated ladder), and the logits at `=`. No tolerance is declared: each program is chosen
//! by its two-part code, and the ladder over the amount of behaviour traces the frontier, each
//! search starting from the previous one's program. The native program and the empty program (the
//! model's mean logits on every input) are reported as its endpoints.
//!
//! `declared` gives the contract the token cycle `x → x + 1` on `Z_p` (and on the classes), used by
//! the comparison run's declared-characters prior; `recovered` declares no group, and a labelling can
//! enter only through the plane basis recovered from the weights. No frequency is named anywhere.
//!
//! The report (JSON on stdout) holds, per rung, the program's bits and data bits, its certified
//! maximal row KL and argmax disagreements, and the component view: each basis (and, for a recovered
//! labelling, the automorphism `a ↦ k a` relating it to the token order, found by checking every unit
//! `k`) and each component's reads, laws, writes, uses and bits.

use gam_mpd::contract::{Contract, FamilyKind};
use gam_mpd::engine::{Budget, Edit, EngineError, Exactness, Primitive, Proposal, SearchContext, decompose_from, library};
use gam_mpd::fit::ProposalKind;
use gam_mpd::operator_program::{
    Basis, Declarations, Domain, FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorBody,
    OperatorProgram, Provenance, Scale, Slot, SlotValues, exact_precision,
};
use gam_mpd::view::view;
use gam_mpd::operator_rewrites::change_basis;
use ndarray::{Array2, Axis, s};
use serde_json::json;
use std::path::{Path, PathBuf};

/// Tensor `name` of the export: raw little-endian float64 in C order, shaped by `export.json`.
fn read_tensor(dir: &Path, record: &serde_json::Value, name: &str) -> Result<Array2<f64>, String> {
    let shape = record["files"][name]["shape"]
        .as_array()
        .ok_or_else(|| format!("export.json: no shape for {name}"))?
        .iter()
        .map(|v| v.as_u64().map(|v| v as usize).ok_or_else(|| format!("export.json: {name} shape")))
        .collect::<Result<Vec<_>, _>>()?;
    let [rows, cols] = shape[..] else { return Err(format!("{name}: shape {shape:?} is not two axes")) };
    let path = dir.join(format!("{name}.f64"));
    let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() != rows * cols * 8 {
        return Err(format!("{}: {} bytes for {rows}×{cols}", path.display(), bytes.len()));
    }
    let values: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7]]))
        .collect();
    Array2::from_shape_vec((rows, cols), values).map_err(|error| error.to_string())
}

struct Config {
    p: usize,
    heads: usize,
    head_dim: usize,
    model: usize,
    hidden: usize,
}

fn read_config(dir: &Path) -> Result<(Config, serde_json::Value), String> {
    let text = std::fs::read_to_string(dir.join("export.json")).map_err(|error| error.to_string())?;
    let record: serde_json::Value = serde_json::from_str(&text).map_err(|error| error.to_string())?;
    let field = |name: &str| {
        record["config"][name].as_u64().map(|v| v as usize).ok_or_else(|| format!("export.json: config.{name}"))
    };
    Ok((
        Config { p: field("p")?, heads: field("n_heads")?, head_dim: field("d_head")?, model: field("d_model")?, hidden: field("d_mlp")? },
        record,
    ))
}

fn dense(name: &str, rows: &Interface, cols: &Interface, values: Array2<f64>) -> Result<Operator, String> {
    let precision = exact_precision(values.iter().copied()).map_err(|error| error.to_string())?;
    Operator::dense(name, rows.clone(), cols.clone(), values, precision, Provenance::native(name)).map_err(|error| error.to_string())
}

/// The native network as an operator program, and its contract.
fn native(
    dir: &Path,
    record: &serde_json::Value,
    config: &Config,
    repeats: usize,
    declared: bool,
) -> Result<(OperatorProgram, Contract), String> {
    let (p, heads, dh, d, n) = (config.p, config.heads, config.head_dim, config.model, config.hidden);
    let tensor = |name: &str| read_tensor(dir, record, name);
    let cycle = |size: usize| declared.then(|| (0..size).map(|t| (t < p).then_some(t as u32)).collect::<Vec<_>>());
    let declarations = Declarations { parameters: 0,
        domains: vec![Domain { size: p + 1, cycle: cycle(p + 1) }, Domain { size: p, cycle: cycle(p) }],
        slots: vec![Slot::Token { domain: 0 }, Slot::Token { domain: 0 }, Slot::Token { domain: 0 }],
    };
    let tokens = Interface::uniform(p + 1, 1, LabelKind::Token, 0).map_err(|e| e.to_string())?;
    let classes = Interface::uniform(p, 1, LabelKind::Token, 0).map_err(|e| e.to_string())?;
    let model = Interface::native(d).map_err(|e| e.to_string())?;
    let head = Interface::native(dh).map_err(|e| e.to_string())?;
    let units = Interface::uniform(n, 1, LabelKind::Unit, 0).map_err(|e| e.to_string())?;
    let constant = Interface::constant();
    let (w_e, w_pos, w_q, w_k, w_v, w_o) =
        (tensor("W_E")?, tensor("W_pos")?, tensor("W_Q")?, tensor("W_K")?, tensor("W_V")?, tensor("W_O")?);
    let (w_in, b_in, w_out, b_out, w_u) = (tensor("W_in")?, tensor("b_in")?, tensor("W_out")?, tensor("b_out")?, tensor("W_U")?);
    let mut operators = vec![
        dense("W_E", &model, &tokens, w_e.t().to_owned())?,
        dense("pos0", &model, &constant, w_pos.row(0).to_owned().insert_axis(Axis(1)))?,
        dense("pos1", &model, &constant, w_pos.row(1).to_owned().insert_axis(Axis(1)))?,
        dense("pos2", &model, &constant, w_pos.row(2).to_owned().insert_axis(Axis(1)))?,
    ];
    let mut nodes = vec![
        Node::Feature { slot: 0, basis: 0 },
        Node::Feature { slot: 1, basis: 0 },
        Node::Feature { slot: 2, basis: 0 },
        Node::Affine { terms: vec![(0, 0)], bias: Some(1) },
        Node::Affine { terms: vec![(1, 0)], bias: Some(2) },
        Node::Affine { terms: vec![(2, 0)], bias: Some(3) },
    ];
    let x = [3usize, 4, 5];
    let mut mixes = Vec::new();
    for h in 0..heads {
        let rows = s![h * dh..(h + 1) * dh, ..];
        operators.push(dense(&format!("W_Q{h}"), &head, &model, w_q.slice(rows).to_owned())?);
        let q_op = operators.len() - 1;
        operators.push(dense(&format!("W_K{h}"), &head, &model, w_k.slice(rows).to_owned())?);
        let k_op = operators.len() - 1;
        operators.push(dense(&format!("W_V{h}"), &head, &model, w_v.slice(rows).to_owned())?);
        let v_op = operators.len() - 1;
        operators.push(dense(&format!("W_O{h}"), &model, &head, w_o.slice(s![.., h * dh..(h + 1) * dh]).to_owned())?);
        let o_op = operators.len() - 1;
        nodes.push(Node::Affine { terms: vec![(x[2], q_op)], bias: None });
        let q = nodes.len() - 1;
        let mut scores = Vec::new();
        for &xj in &x {
            nodes.push(Node::Affine { terms: vec![(xj, k_op)], bias: None });
            nodes.push(Node::Bilinear { left: q, right: nodes.len() - 1, scale: Scale::InverseSqrt(dh as u32) });
            scores.push(nodes.len() - 1);
        }
        nodes.push(Node::Softmax { scores });
        let weights = nodes.len() - 1;
        let mut payloads = Vec::new();
        for (j, &xj) in x.iter().enumerate() {
            nodes.push(Node::Affine { terms: vec![(xj, v_op)], bias: None });
            payloads.push((j, nodes.len() - 1));
        }
        nodes.push(Node::Mix { weights, payloads });
        mixes.push((nodes.len() - 1, o_op));
    }
    operators.push(Operator::identity("I", model.clone()));
    let identity = operators.len() - 1;
    let mut mid_terms: Vec<(usize, usize)> = mixes.clone();
    mid_terms.push((x[2], identity));
    nodes.push(Node::Affine { terms: mid_terms, bias: None });
    let mid = nodes.len() - 1;
    operators.push(dense("W_in", &units, &model, w_in)?);
    operators.push(dense("b_in", &units, &constant, b_in.t().to_owned())?);
    nodes.push(Node::Affine { terms: vec![(mid, operators.len() - 2)], bias: Some(operators.len() - 1) });
    let pre = nodes.len() - 1;
    nodes.push(Node::Pointwise { input: pre, laws: vec![Law::Relu; n] });
    let act = nodes.len() - 1;
    operators.push(dense("W_out", &model, &units, w_out)?);
    operators.push(dense("b_out", &model, &constant, b_out.t().to_owned())?);
    nodes.push(Node::Affine { terms: vec![(mid, identity), (act, operators.len() - 2)], bias: Some(operators.len() - 1) });
    let fin = nodes.len() - 1;
    operators.push(dense("W_U", &classes, &model, w_u)?);
    nodes.push(Node::Affine { terms: vec![(fin, operators.len() - 1)], bias: None });
    nodes.push(Node::Readout { input: nodes.len() - 1, basis: 1 });
    let output = nodes.len() - 1;
    let program = OperatorProgram { rules: Vec::new(),
        declarations: declarations.clone(),
        bases: vec![Basis::Indicator { domain: 0 }, Basis::Indicator { domain: 1 }],
        operators: operators.into_iter().map(std::sync::Arc::new).collect(),
        nodes,
        output,
    };
    let pairs: Vec<(u32, u32)> = (0..p as u32).flat_map(|a| (0..p as u32).map(move |b| (a, b))).collect();
    let family = FamilyInputs {
        layout: None,
        rows: pairs.len(),
        slots: vec![
            SlotValues::Tokens(pairs.iter().map(|q| q.0).collect()),
            SlotValues::Tokens(pairs.iter().map(|q| q.1).collect()),
            SlotValues::Tokens(vec![p as u32; pairs.len()]),
        ],
    };
    let contract = Contract {
        declarations,
        family,
        kind: FamilyKind::Complete { description: format!("every (a, b) in Z_{p}^2 at the = position") },
        observations: repeats as u64,
        readouts: 1,
        readout_slots: Some(vec![vec![0, 1, 2]]),
    };
    Ok((program, contract))
}

/// For a recovered labelling of the first `p` tokens, the unit `k` with `a(x) = k x + c mod p`,
/// checked on every token, if one exists.
fn automorphism(positions: &[Option<u32>], p: usize) -> Option<(usize, usize)> {
    let a0 = positions.first().copied().flatten()? as usize;
    for k in 1..p {
        if positions.iter().take(p).enumerate().all(|(x, pos)| *pos == Some(((k * x + a0) % p) as u32)) {
            return Some((k, a0));
        }
    }
    None
}

fn describe(program: &OperatorProgram, p: usize) -> serde_json::Value {
    let bases: Vec<serde_json::Value> = program
        .bases
        .iter()
        .map(|basis| match basis {
            Basis::Indicator { domain } => json!({"kind": "indicator", "domain": domain}),
            Basis::Characters { domain, positions, declared } => json!({
                "kind": "characters", "domain": domain, "declared": declared,
                "period": basis.period(),
                "automorphism_of_token_order": automorphism(positions, p).map(|(k, c)| json!({"k": k, "offset": c})),
            }),
        })
        .collect();
    let view = match view(program) {
        Ok(view) => view,
        Err(error) => return json!({"error": error.to_string()}),
    };
    let operators: Vec<serde_json::Value> = view
        .components
        .iter()
        .map(|c| {
            json!({
                "name": c.name, "reads": c.reads, "writes": c.writes, "applied_by": c.applied_by, "uses": c.uses,
                "reals": c.reals, "bits": c.bits, "sources": c.sources, "native": c.native_unchanged,
            })
        })
        .collect();
    json!({
        "bases": bases, "components": operators, "nodes": program.nodes.len(), "reals": program.real_count(),
        "unchanged_native_bits": view.unchanged_native_bits,
    })
}

/// The contract's declared token cycle as a prior (the `declared` comparison run only): the
/// character basis of each declared cycle, sent as one bit.
struct DeclaredCharacters;

impl Primitive for DeclaredCharacters {
    fn name(&self) -> &'static str {
        "declared_characters"
    }

    fn propose(&self, context: &SearchContext<'_>) -> Result<Vec<Proposal>, EngineError> {
        let program = context.program;
        let mut out = Vec::new();
        for (index, basis) in program.bases.iter().enumerate() {
            let Basis::Indicator { domain } = basis else { continue };
            let Some(cycle) = program.declarations.domains[*domain].cycle.clone() else { continue };
            if let Some(candidate) = change_basis(program, index, cycle, true)? {
                out.push(Proposal {
                    primitive: "declared_characters",
                    kind: ProposalKind::Expose,
                    exactness: Exactness::Exact { derivation: "indicators are Φ⁻¹ times the declared characters".to_string() },
                    description: format!("declared characters on domain {domain}"),
                    edit: Edit::Program(Box::new(candidate)),
                });
            }
        }
        Ok(out)
    }
}

/// The analyst's check, never an input to the search: for every operator reading or writing the
/// token or class domain (a table indexed by tokens), the frequencies `k` of the token order
/// `x → x + 1` carrying the most power over the first `p` tokens, with their shares of the table's
/// non-constant power. Computed on the program the search returned.
fn frequencies(program: &OperatorProgram, p: usize) -> serde_json::Value {
    let mut tables = Vec::new();
    for op in &program.operators {
        let tokens = |side: &Interface| side.groups().iter().all(|g| g.label.kind == LabelKind::Token);
        let table = if tokens(&op.cols) && op.cols.width() >= p {
            op.matrix().t().to_owned()
        } else if tokens(&op.rows) && op.rows.width() >= p {
            op.matrix()
        } else {
            continue;
        };
        let planes = (p - 1) / 2;
        let mut power = vec![0.0_f64; planes];
        for column in table.columns() {
            for (k, slot) in power.iter_mut().enumerate() {
                let (mut c, mut s) = (0.0, 0.0);
                for x in 0..p {
                    let angle = std::f64::consts::TAU * (((k + 1) * x) % p) as f64 / p as f64;
                    c += column[x] * angle.cos();
                    s += column[x] * angle.sin();
                }
                *slot += c * c + s * s;
            }
        }
        let total: f64 = power.iter().sum();
        if total == 0.0 {
            continue;
        }
        let mut ranked: Vec<(usize, f64)> = power.iter().enumerate().map(|(k, v)| (k + 1, v / total)).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut cumulative = 0.0;
        let top: Vec<serde_json::Value> = ranked
            .iter()
            .take_while(|(_, share)| {
                let keep = cumulative < 0.95;
                cumulative += share;
                keep
            })
            .map(|(k, share)| json!({"k": k, "share": share}))
            .collect();
        tables.push(json!({"operator": op.name, "reals": op.real_count(), "top_frequencies_to_95pct": top}));
    }
    json!(tables)
}

/// The analyst's reading of a factored program, never an input to the search: each factor
/// coordinate's values over the family as a table over `(a, b)`, its two-dimensional Fourier power
/// at the frequencies `(k_a, k_b)` carrying the most of it, and the units of each pointwise layer
/// grouped into rules by their support on the factors they read and write.
fn rules(program: &OperatorProgram, contract: &Contract, p: usize) -> serde_json::Value {
    let Ok(trace) = program.execute(&contract.family, false) else { return json!({"error": "execution"}) };
    let rows = contract.family.rows;
    let SlotValues::Tokens(a) = &contract.family.slots[0] else { return json!({"error": "slot 0"}) };
    let SlotValues::Tokens(b) = &contract.family.slots[1] else { return json!({"error": "slot 1"}) };
    let factor_of = |node: usize| -> Option<usize> {
        let Node::Affine { terms, .. } = &program.nodes[node] else { return None };
        (terms.len() == 1 && program.operators[terms[0].1].rows.groups().iter().all(|g| g.label.kind == LabelKind::Factor))
            .then_some(terms[0].1)
    };
    // The dominant frequencies of one column of a node's values over Z_p².
    let spectrum = |values: ndarray::ArrayView1<'_, f64>| -> serde_json::Value {
        let mut table = vec![0.0_f64; p * p];
        for row in 0..rows {
            table[a[row] as usize * p + b[row] as usize] = values[row];
        }
        let mean = table.iter().sum::<f64>() / table.len() as f64;
        let mut power = Vec::new();
        for ka in 0..p {
            for kb in 0..p {
                let (mut re, mut im) = (0.0, 0.0);
                for x in 0..p {
                    for y in 0..p {
                        let angle = std::f64::consts::TAU * ((ka * x + kb * y) % p) as f64 / p as f64;
                        let v = table[x * p + y] - mean;
                        re += v * angle.cos();
                        im -= v * angle.sin();
                    }
                }
                power.push(((ka, kb), re * re + im * im));
            }
        }
        let total: f64 = power.iter().map(|(_, v)| v).sum();
        power.sort_by(|x, y| y.1.total_cmp(&x.1));
        let fold = |k: usize| k.min(p - k);
        let mut cumulative = 0.0;
        let top: Vec<serde_json::Value> = power
            .iter()
            .take_while(|(_, v)| {
                let keep = cumulative < 0.9 * total;
                cumulative += v;
                keep
            })
            .take(6)
            .map(|((ka, kb), v)| json!({"k_a": fold(*ka), "k_b": fold(*kb), "share": v / total.max(f64::MIN_POSITIVE)}))
            .collect();
        json!(top)
    };
    let mut layers = Vec::new();
    for (index, node) in program.nodes.iter().enumerate() {
        let Node::Pointwise { input, .. } = node else { continue };
        // Read side: the pre-activation reads factor coordinates z through a coefficient operator.
        let Node::Affine { terms, .. } = &program.nodes[*input] else { continue };
        let read = terms.iter().find(|(z, _)| factor_of(*z).is_some()).copied();
        // Write side: a node reading this layer through a factor basis.
        let write = program.nodes.iter().enumerate().find_map(|(z, n)| match n {
            Node::Affine { terms, .. } if terms.len() == 1 && terms[0].0 == index && factor_of(z).is_some() => Some(z),
            _ => None,
        });
        let support = |op: usize, transpose: bool| -> Vec<Vec<usize>> {
            let OperatorBody::Dense { present, .. } = &program.operators[op].body else { return Vec::new() };
            let present = if transpose { present.t().to_owned() } else { present.clone() };
            present.outer_iter().map(|row| row.iter().enumerate().filter(|(_, k)| **k).map(|(i, _)| i).collect()).collect()
        };
        let units = trace.values[index].ncols();
        let reads: Vec<Vec<usize>> = read.map_or(vec![Vec::new(); units], |(_, op)| support(op, false));
        let writes: Vec<Vec<usize>> = write.and_then(factor_of).map_or(vec![Vec::new(); units], |op| support(op, true));
        // A rule's factors: the factors its units read (write) together, closed under sharing.
        let closure = |supports: &[Vec<usize>]| -> Vec<Vec<usize>> {
            let count = supports.iter().flatten().max().map_or(0, |m| m + 1);
            let mut parent: Vec<usize> = (0..count).collect();
            fn root(parent: &[usize], mut x: usize) -> usize {
                while parent[x] != x {
                    x = parent[x];
                }
                x
            }
            for support in supports {
                for pair in support.windows(2) {
                    let (a, b) = (root(&parent, pair[0]), root(&parent, pair[1]));
                    parent[a] = b;
                }
            }
            supports
                .iter()
                .map(|support| {
                    let Some(&first) = support.first() else { return Vec::new() };
                    let r = root(&parent, first);
                    (0..count).filter(|&f| root(&parent, f) == r).collect()
                })
                .collect()
        };
        let (read_rules, write_rules) = (closure(&reads), closure(&writes));
        let mut groups: std::collections::BTreeMap<(Vec<usize>, Vec<usize>), Vec<usize>> = std::collections::BTreeMap::new();
        for unit in 0..units {
            let key = (read_rules.get(unit).cloned().unwrap_or_default(), write_rules.get(unit).cloned().unwrap_or_default());
            groups.entry(key).or_default().push(unit);
        }
        let read_spectra: Vec<serde_json::Value> = read
            .map(|(z, _)| (0..trace.values[z].ncols()).map(|i| spectrum(trace.values[z].column(i))).collect())
            .unwrap_or_default();
        let write_spectra: Vec<serde_json::Value> = write
            .map(|z| (0..trace.values[z].ncols()).map(|i| spectrum(trace.values[z].column(i))).collect())
            .unwrap_or_default();
        // Each unit's dominant read: the factor carrying most of its pre-activation's variance, and
        // that factor's leading frequency; units counted per frequency.
        let mut dominant: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
        if let Some((z, op)) = read {
            let coefficients = program.operators[op].matrix();
            let spread: Vec<f64> = (0..trace.values[z].ncols())
                .map(|i| {
                    let column = trace.values[z].column(i);
                    let mean = column.mean().unwrap_or(0.0);
                    (column.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / column.len() as f64).sqrt()
                })
                .collect();
            let leading = |i: usize| -> usize {
                read_spectra.get(i).and_then(|sp| sp.get(0)).map_or(0, |top| {
                    top["k_a"].as_u64().unwrap_or(0).max(top["k_b"].as_u64().unwrap_or(0)) as usize
                })
            };
            for unit in 0..coefficients.nrows() {
                let best = (0..coefficients.ncols())
                    .map(|i| (i, (coefficients[[unit, i]] * spread[i]).abs()))
                    .max_by(|a, b| a.1.total_cmp(&b.1));
                if let Some((i, size)) = best
                    && size > 0.0
                {
                    dominant.entry(leading(i)).or_default().push(unit);
                }
            }
        }
        let mut listed: Vec<_> = groups.into_iter().collect();
        listed.sort_by(|x, y| y.1.len().cmp(&x.1.len()));
        layers.push(json!({
            "node": index,
            "units": units,
            "units_by_dominant_frequency": dominant.iter().map(|(k, units)| json!({"k": k, "units": units.len()})).collect::<Vec<_>>(),
            "read_factors": read_spectra,
            "write_factors": write_spectra,
            "rules": listed.iter().map(|((r, w), members)| json!({
                "members": members.len(), "reads": r, "writes": w, "units": members,
            })).collect::<Vec<_>>(),
        }));
    }
    json!(layers)
}

/// The figure data of a factored program (analyst output): for the first ReLU layer, each read
/// factor's values over the family as a table over `(a, b)` with its marginals over `a` and `b`,
/// each write factor's values averaged over the inputs with one `c = (a + b) mod p`, each factor's
/// coefficient energy (`Σ_units coefficient² · var(factor)`), and the structure-function curve.
fn factor_figure(program: &OperatorProgram, contract: &Contract, p: usize, curve: &[gam_mpd::engine::CurvePoint]) -> serde_json::Value {
    let Ok(trace) = program.execute(&contract.family, false) else { return json!({"error": "execution"}) };
    let (SlotValues::Tokens(a), SlotValues::Tokens(b)) = (&contract.family.slots[0], &contract.family.slots[1]) else {
        return json!({"error": "token slots"});
    };
    let rows = contract.family.rows;
    let is_factor = |node: usize| -> Option<usize> {
        let Node::Affine { terms, .. } = &program.nodes[node] else { return None };
        (terms.len() == 1 && program.operators[terms[0].1].rows.groups().iter().all(|g| g.label.kind == LabelKind::Factor))
            .then_some(terms[0].1)
    };
    let variance = |column: ndarray::ArrayView1<'_, f64>| {
        let mean = column.mean().unwrap_or(0.0);
        column.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / column.len() as f64
    };
    let Some(act) = program.nodes.iter().position(|n| matches!(n, Node::Pointwise { .. })) else { return json!({"error": "no layer"}) };
    let Node::Pointwise { input, .. } = program.nodes[act] else { unreachable!() };
    let Node::Affine { terms, .. } = &program.nodes[input] else { return json!({"error": "pre-activation"}) };
    let mut read = Vec::new();
    if let Some(&(z, op)) = terms.iter().find(|(z, _)| is_factor(*z).is_some()) {
        let coefficients = program.operators[op].matrix();
        for i in 0..trace.values[z].ncols() {
            let column = trace.values[z].column(i);
            let mut table = vec![vec![0.0_f64; p]; p];
            for row in 0..rows {
                table[a[row] as usize][b[row] as usize] = column[row];
            }
            let over_a: Vec<f64> = (0..p).map(|x| table[x].iter().sum::<f64>() / p as f64).collect();
            let over_b: Vec<f64> = (0..p).map(|y| (0..p).map(|x| table[x][y]).sum::<f64>() / p as f64).collect();
            let energy = coefficients.column(i).iter().map(|c| c * c).sum::<f64>() * variance(column);
            read.push(json!({"factor": i, "over_a": over_a, "over_b": over_b, "table": table, "coefficient_energy": energy}));
        }
    }
    let mut write = Vec::new();
    if let Some(z) = program.nodes.iter().enumerate().position(|(z, n)| matches!(n, Node::Affine { terms, .. } if terms.len() == 1 && terms[0].0 == act) && is_factor(z).is_some()) {
        let op = is_factor(z).expect("a factor node");
        let coefficients = program.operators[op].matrix();
        for i in 0..trace.values[z].ncols() {
            let column = trace.values[z].column(i);
            let (mut sums, mut counts) = (vec![0.0_f64; p], vec![0usize; p]);
            for row in 0..rows {
                let c = (a[row] as usize + b[row] as usize) % p;
                sums[c] += column[row];
                counts[c] += 1;
            }
            let over_c: Vec<f64> = sums.iter().zip(&counts).map(|(s, n)| s / (*n).max(1) as f64).collect();
            let energy = coefficients.row(i).iter().map(|c| c * c).sum::<f64>() * variance(column);
            write.push(json!({"factor": i, "over_sum": over_c, "coefficient_energy": energy}));
        }
    }
    // The layer's writes on the logits, whatever the program composed them into: each unit's
    // derivative of every class logit (one reverse pass per class, at the first input; the path
    // to the logits is linear), the class mean removed (the softmax shift), and its singular
    // directions over the classes `c = (a + b) mod p`.
    let mut writes_over_classes = Vec::new();
    let logits = &trace.values[program.output];
    let classes = logits.ncols();
    let units = trace.values[act].ncols();
    let mut w = ndarray::Array2::<f64>::zeros((classes, units));
    for c in 0..classes {
        let mut cotangent = ndarray::Array2::<f64>::zeros(logits.dim());
        cotangent[[0, c]] = 1.0;
        if let Ok(back) = gam_mpd::derivatives::vjp(program, &contract.family, &trace, cotangent)
            && let Some(g) = &back[act]
        {
            w.row_mut(c).assign(&g.row(0));
        }
    }
    let mean = w.mean_axis(ndarray::Axis(0)).expect("classes");
    let w = &w - &mean;
    // The writes' span (singular directions above the band), rotated by the factor fit's own gauge
    // to the units' shortest write code (`factors::fix_gauge`), so each direction is what a group
    // of units writes rather than a mixture the singular value decomposition picked.
    if let Ok(decomposed) = gam_mpd::dense::svd(w.view(), false) {
        let total: f64 = decomposed.singular_values.iter().map(|s| s * s).sum();
        let rank = decomposed.singular_values.iter().filter(|s| **s > decomposed.band).count();
        let mut directions = decomposed.u.slice(ndarray::s![.., ..rank]).t().to_owned();
        let mut coefficients = w.t().dot(&directions.t());
        let largest = coefficients.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let steps = vec![largest * 2f64.powi(-8); coefficients.nrows()];
        gam_mpd::factors::fix_gauge(&mut coefficients, &steps, &mut directions);
        for i in 0..rank {
            let energy = coefficients.column(i).iter().map(|c| c * c).sum::<f64>();
            writes_over_classes.push(json!({
                "direction": i,
                "over_class": directions.row(i).to_vec(),
                "energy_share": energy / total.max(f64::MIN_POSITIVE),
            }));
        }
    }
    json!({
        "p": p,
        "read_factors": read,
        "write_factors": write,
        "writes_over_classes": writes_over_classes,
        "curve": curve.iter().map(|c| json!({"structure_bits": c.structure_bits, "rest_bits": c.rest_bits, "description": c.description})).collect::<Vec<_>>(),
    })
}

/// The empty program: the model's mean logits over the family, the same on every input.
fn empty_program(model: &OperatorProgram, contract: &Contract) -> Result<OperatorProgram, String> {
    let logits = contract.logits(model).map_err(|e| e.to_string())?.values;
    let mean = logits.mean_axis(Axis(0)).ok_or("an empty family")?;
    let classes = Interface::uniform(mean.len(), 1, LabelKind::Token, 0).map_err(|e| e.to_string())?;
    let operator = dense("mean_logits", &classes, &Interface::constant(), mean.insert_axis(Axis(1)))?;
    Ok(OperatorProgram { rules: Vec::new(),
        declarations: model.declarations.clone(),
        bases: Vec::new(),
        operators: vec![std::sync::Arc::new(operator)],
        nodes: vec![Node::Constant { operator: 0 }],
        output: 0,
    })
}

/// A frontier point: a program's bits and its evaluation against the model.
fn point(label: &str, program: &OperatorProgram, contract: &Contract, model: &OperatorProgram) -> Result<serde_json::Value, String> {
    let reference = contract.logits(model).map_err(|e| e.to_string())?;
    let score = contract.score(program, &reference).map_err(|e| e.to_string())?;
    let e = &score.evaluation;
    Ok(json!({
        "label": label, "bits": score.program_bits, "structure_bits": score.structure_bits, "precision_bits": score.precision_bits,
        "explanation_bits": score.explanation.bits, "active_per_input": score.explanation.mean_active(), "data_bits": score.data_bits, "reals": program.real_count(),
        "max_kl": format!("{:?}", e.max_kl), "max_kl_upper": e.max_kl.upper_bound(),
        "argmax_disagreements": e.argmax_disagreements,
    }))
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_engine_modadd_2951 EXPORT_DIR REPEATS(comma-separated, ascending) [declared|recovered] [SCREENINGS CERTIFICATIONS]";
    let dir = PathBuf::from(args.get(1).ok_or(usage)?);
    let ladder: Vec<usize> = args
        .get(2)
        .ok_or(usage)?
        .split(',')
        .map(|v| v.parse::<usize>().map_err(|e| format!("REPEATS: {e}")))
        .collect::<Result<_, _>>()?;
    if ladder.is_empty() || ladder[0] == 0 || ladder.windows(2).any(|w| w[0] >= w[1]) {
        return Err("REPEATS must be positive and ascend strictly".to_string());
    }
    let declared = match args.get(3).map(String::as_str) {
        Some("declared") => true,
        Some("recovered") | None => false,
        Some(other) => return Err(format!("unknown mode {other}; {usage}")),
    };
    let screenings: u64 = args.get(4).map_or(Ok(1 << 20), |v| v.parse()).map_err(|e| format!("SCREENINGS: {e}"))?;
    let certifications: u64 = args.get(5).map_or(Ok(1 << 12), |v| v.parse()).map_err(|e| format!("CERTIFICATIONS: {e}"))?;
    let (config, record) = read_config(&dir)?;
    let mut library = library();
    if declared {
        library.push(Box::new(DeclaredCharacters));
    }
    let budget = Budget { screenings, certifications, ..Budget::default() };
    let (model, first_contract) = native(&dir, &record, &config, ladder[0], declared)?;
    let endpoints = vec![
        point("native", &model, &first_contract, &model)?,
        point("empty", &empty_program(&model, &first_contract)?, &first_contract, &model)?,
    ];
    let mut frontier = Vec::new();
    let mut start = model.clone();
    let mut figure = json!(null);
    for &repeats in &ladder {
        let (_, contract) = native(&dir, &record, &config, repeats, declared)?;
        let started = std::time::Instant::now();
        let result = decompose_from(&model, &start, &contract, &library, &budget).map_err(|e| e.to_string())?;
        let evaluation = &result.score.evaluation;
        frontier.push(json!({
            "repeats": repeats,
            "rows": contract.family.rows,
            "bits": result.score.program_bits,
            "structure_bits": result.score.structure_bits,
            "precision_bits": result.score.precision_bits,
            "explanation_bits": result.score.explanation.bits,
            "explanation_bits_per_input": result.score.explanation.bits_per_input(),
            "active_per_input": result.score.explanation.mean_active(),
            "data_bits": result.score.data_bits,
            "reals": result.program.real_count(),
            "max_kl": format!("{:?}", evaluation.max_kl),
            "max_kl_upper": evaluation.max_kl.upper_bound(),
            "argmax_disagreements": evaluation.argmax_disagreements,
            "argmax_uncertified": evaluation.argmax_uncertified,
            "structure": describe(&result.program, config.p),
            "frequencies": frequencies(&result.program, config.p),
            "rules": rules(&result.program, &contract, config.p),
            "curve": result.curve.iter().map(|c| json!({
                "structure_bits": c.structure_bits, "rest_bits": c.rest_bits, "description": c.description,
            })).collect::<Vec<_>>(),
            "knee": result.knee().map(|c| json!({"structure_bits": c.structure_bits, "rest_bits": c.rest_bits, "description": c.description})),
            "stop": format!("{:?}", result.stop),
            "seconds": started.elapsed().as_secs_f64(),
        }));
        let stop = format!("{:?}", result.stop);
        eprintln!(
            "repeats {repeats}: {} + {:.1} bits ({} reals), max KL <= {:e}, {} argmax disagreements, stop {}",
            result.score.program_bits,
            result.score.data_bits,
            result.program.real_count(),
            evaluation.max_kl.upper_bound().unwrap_or(f64::INFINITY),
            evaluation.argmax_disagreements,
            stop
        );
        figure = factor_figure(&result.program, &contract, config.p, &result.curve);
        start = result.program;
    }
    let figure_path = dir.join("factors.json");
    std::fs::write(&figure_path, serde_json::to_string(&figure).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", figure_path.display()))?;
    let report = json!({
        "export": record,
        "mode": if declared { "declared" } else { "recovered" },
        "native_reals": model.real_count(),
        "native_frequencies": frequencies(&model, config.p),
        "endpoints": endpoints,
        "frontier": frontier,
    });
    let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    println!("{text}");
    Ok(())
}

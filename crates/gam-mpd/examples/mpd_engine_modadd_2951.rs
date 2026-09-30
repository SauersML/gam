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
use gam_mpd::engine::{
    Budget, Coarsen, DeadUnits, DropBlocks, Edit, EngineError, Exactness, LawSubstitution, LowRank, Primitive, Proposal,
    SearchContext,
    decompose_from,
};
use gam_mpd::fit::ProposalKind;
use gam_mpd::operator_program::{
    Basis, Declarations, Domain, FamilyInputs, Interface, LabelKind, Law, Node, Operator,
    OperatorProgram, Provenance, Scale, Slot, SlotValues, exact_precision,
};
use gam_mpd::derivatives::CurvaturePrecision;
use gam_mpd::factors::SharedFactors;
use gam_mpd::refit::RefitSearch;
use gam_mpd::view::view;
use gam_mpd::operator_rewrites::{
    BilinearConstantSide, CenterLogits, ComposeAffine, DropKeyBias, FoldConstants, PlaneBasis, PushThroughMix, StackTerms,
    change_basis,
};
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
        operators,
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
                "name": c.name, "reads": c.reads, "writes": c.writes, "laws": c.laws, "uses": c.uses,
                "reals": c.reals, "bits": c.bits, "sources": c.sources, "unresolved": c.unresolved,
            })
        })
        .collect();
    json!({
        "bases": bases, "components": operators, "nodes": program.nodes.len(), "reals": program.real_count(),
        "unresolved_bits_fraction": view.unresolved_fraction(),
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

/// The empty program: the model's mean logits over the family, the same on every input.
fn empty_program(model: &OperatorProgram, contract: &Contract) -> Result<OperatorProgram, String> {
    let logits = contract.logits(model).map_err(|e| e.to_string())?.values;
    let mean = logits.mean_axis(Axis(0)).ok_or("an empty family")?;
    let classes = Interface::uniform(mean.len(), 1, LabelKind::Token, 0).map_err(|e| e.to_string())?;
    let operator = dense("mean_logits", &classes, &Interface::constant(), mean.insert_axis(Axis(1)))?;
    Ok(OperatorProgram { rules: Vec::new(),
        declarations: model.declarations.clone(),
        bases: Vec::new(),
        operators: vec![operator],
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
        "label": label, "bits": score.program_bits, "data_bits": score.data_bits, "reals": program.real_count(),
        "max_kl": format!("{:?}", e.max_kl), "max_kl_upper": e.max_kl.upper_bound(),
        "argmax_disagreements": e.argmax_disagreements,
    }))
}

fn main() -> Result<(), String> {
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
    let mut library: Vec<Box<dyn Primitive>> = vec![
        Box::new(FoldConstants),
        Box::new(BilinearConstantSide),
        Box::new(ComposeAffine),
        Box::new(PushThroughMix),
        Box::new(PlaneBasis),
        Box::new(DropBlocks),
        Box::new(Coarsen),
        Box::new(DeadUnits),
        Box::new(LowRank),
        Box::new(StackTerms),
        Box::new(CenterLogits),
        Box::new(DropKeyBias),
        Box::new(SharedFactors),
        Box::new(LawSubstitution),
        Box::new(CurvaturePrecision { probes: 4 }),
    ];
    if declared {
        library.push(Box::new(DeclaredCharacters));
    }
    let budget = Budget {
        screenings,
        certifications,
        refit: Some(RefitSearch { newton_steps: 8, conjugate_gradient_steps: 16 }),
    };
    let (model, first_contract) = native(&dir, &record, &config, ladder[0], declared)?;
    let endpoints = vec![
        point("native", &model, &first_contract, &model)?,
        point("empty", &empty_program(&model, &first_contract)?, &first_contract, &model)?,
    ];
    let mut frontier = Vec::new();
    let mut start = model.clone();
    for &repeats in &ladder {
        let (_, contract) = native(&dir, &record, &config, repeats, declared)?;
        let started = std::time::Instant::now();
        let result = decompose_from(&model, &start, &contract, &library, &budget).map_err(|e| e.to_string())?;
        let evaluation = &result.score.evaluation;
        frontier.push(json!({
            "repeats": repeats,
            "rows": contract.family.rows,
            "bits": result.score.program_bits,
            "data_bits": result.score.data_bits,
            "reals": result.program.real_count(),
            "max_kl": format!("{:?}", evaluation.max_kl),
            "max_kl_upper": evaluation.max_kl.upper_bound(),
            "argmax_disagreements": evaluation.argmax_disagreements,
            "argmax_uncertified": evaluation.argmax_uncertified,
            "structure": describe(&result.program, config.p),
            "frequencies": frequencies(&result.program, config.p),
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
        start = result.program;
    }
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

//! DISCOVERY_DIR OUT_DIR [SETTINGS.json], or --self-check.
//! One shared pre-RoPE translation per native head; conditional attention on
//! unchanged keys only. Discovery crossfit is not untouched heldout evaluation.
use gam_mpd::{
    engine::sha256,
    operator_program::{Rotary, Scale},
    query_transition::{Fit, Sample, SolverMethod, SolverSettings, fit, native_features},
};
use memmap2::Mmap;
use ndarray::{Array2, s};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Path, PathBuf},
    time::Instant,
};
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn read(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&std::fs::read(path).map_err(err)?).map_err(err)
}
fn save(path: &Path, v: &Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_vec_pretty(v).map_err(err)?).map_err(err)
}
fn integer(v: &Value) -> Result<usize, String> {
    usize::try_from(v.as_u64().ok_or("missing integer")?).map_err(err)
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    diagnostic_read_nodes: Option<Vec<usize>>,
    solver: SolverSettings,
    prefix_absolute_tolerance: f64,
    prefix_relative_tolerance: f64,
    oracle_probability_tolerance: f64,
    oracle_kl_tolerance: f64,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            diagnostic_read_nodes: None,
            solver: SolverSettings::default(),
            prefix_absolute_tolerance: 1e-10,
            prefix_relative_tolerance: 1e-10,
            oracle_probability_tolerance: 1e-10,
            oracle_kl_tolerance: 1e-10,
        }
    }
}
#[derive(Deserialize)]
struct Tensor {
    id: String,
    byte_offset: usize,
    byte_length: usize,
    shape: [usize; 2],
    dtype: String,
    layout: String,
    metadata: Value,
}
#[derive(Deserialize)]
struct CaseReport {
    fixture: Value,
    fixture_id: String,
    rows: usize,
    heads: Vec<Value>,
    packed_file: String,
    packed_sha256: String,
    tensors: Vec<Tensor>,
}
struct Case {
    name: String,
    report: CaseReport,
    tensors: BTreeMap<String, usize>,
    map: Mmap,
    provenance: Value,
}
impl Case {
    fn load(root: &Path, entry: &Value, heads: &BTreeSet<usize>) -> Result<Self, String> {
        let name = entry["case_directory"]
            .as_str()
            .ok_or("case directory missing")?
            .to_owned();
        if name.len() != 8
            || !name.starts_with("case-")
            || !name[5..].bytes().all(|b| b.is_ascii_digit())
        {
            return Err("nonlocal case path".into());
        }
        let dir = root.join(&name);
        let report_path = dir.join("REPORT.json");
        let report_hash = sha256(&report_path)?;
        if entry["report_sha256"] != report_hash {
            return Err(format!("case report hash mismatch {name}"));
        }
        let report: CaseReport =
            serde_json::from_slice(&std::fs::read(&report_path).map_err(err)?).map_err(err)?;
        let fixture_path = dir.join("FIXTURE.json");
        let fixture = read(&fixture_path)?;
        if fixture != report.fixture
            || fixture["panel"] != "discovery"
            || fixture["id"] != report.fixture_id
            || entry["fixture_id"] != report.fixture_id
        {
            return Err("fixture/report identity mismatch or non-discovery input".into());
        }
        if report.packed_file != "states.f64"
            || report.rows != fixture["tokens"].as_array().ok_or("tokens missing")?.len()
        {
            return Err("invalid packing or row count".into());
        }
        let present = report
            .heads
            .iter()
            .map(|v| integer(&v["read_node"]))
            .collect::<Result<BTreeSet<_>, _>>()?;
        if &present != heads || report.heads.len() != heads.len() {
            return Err("missing/duplicate native heads".into());
        }
        let path = dir.join("states.f64");
        let packed_hash = sha256(&path)?;
        if packed_hash != report.packed_sha256 {
            return Err(format!("packed hash mismatch {name}"));
        }
        let file = File::open(&path).map_err(err)?;
        // SAFETY: Read-only mapping of an already hash-verified immutable discovery artifact.
        let map = unsafe { Mmap::map(&file).map_err(err)? };
        let mut offset = 0usize;
        let mut tensors = BTreeMap::new();
        for (i, t) in report.tensors.iter().enumerate() {
            let length = t.shape[0]
                .checked_mul(t.shape[1])
                .and_then(|n| n.checked_mul(8))
                .ok_or("tensor overflow")?;
            if t.byte_offset != offset
                || t.byte_length != length
                || t.dtype != "little-endian f64"
                || t.layout != "row-major"
                || t.shape.contains(&0)
                || tensors.insert(t.id.clone(), i).is_some()
            {
                return Err("invalid tensor manifest".into());
            }
            offset = offset.checked_add(length).ok_or("manifest overflow")?;
        }
        if offset != map.len() {
            return Err("manifest/file length mismatch".into());
        }
        let provenance = json!({"case_directory":name,"fixture_id":report.fixture_id,"report_sha256":report_hash,
            "fixture_sha256":sha256(&fixture_path)?,"packed_sha256":packed_hash,"packed_bytes":map.len()});
        Ok(Self {
            name,
            report,
            tensors,
            map,
            provenance,
        })
    }
    fn tensor(&self, id: &str) -> Result<(&Tensor, Array2<f64>), String> {
        let t = &self.report.tensors[*self
            .tensors
            .get(id)
            .ok_or_else(|| format!("missing tensor {id}"))?];
        let data = self.map[t.byte_offset..t.byte_offset + t.byte_length]
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().expect("eight bytes")))
            .collect::<Vec<_>>();
        if data.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite packed tensor".into());
        }
        Ok((t, Array2::from_shape_vec(t.shape, data).map_err(err)?))
    }
    fn producer(&self, node: usize) -> Result<Array2<f64>, String> {
        let (t, a) = self.tensor(&format!("producer-{node}"))?;
        if integer(&t.metadata["native_node"])? != node {
            return Err("producer node metadata mismatch".into());
        }
        let rows = t.metadata["native_rows"]
            .as_array()
            .ok_or("producer rows missing")?;
        let expected: Vec<usize> = if a.nrows() == self.report.rows {
            (0..self.report.rows).collect()
        } else if a.nrows() == 1 {
            vec![self.report.rows - 1]
        } else {
            return Err("producer rows invalid".into());
        };
        if rows.iter().map(integer).collect::<Result<Vec<_>, _>>()? != expected {
            return Err("producer row metadata mismatch".into());
        }
        Ok(a)
    }
    fn attention(&self, node: usize) -> Result<Vec<f64>, String> {
        let (t, a) = self.tensor(&format!("attention-{node}"))?;
        if a.dim() != (1, self.report.rows)
            || integer(&t.metadata["native_query_row"])? != self.report.rows - 1
            || integer(&t.metadata["native_read_node"])? != node
            || t.metadata["native_key_rows"]
                .as_array()
                .ok_or("attention key rows missing")?
                .iter()
                .map(integer)
                .collect::<Result<Vec<_>, _>>()?
                != (0..self.report.rows).collect::<Vec<_>>()
        {
            return Err("attention metadata mismatch".into());
        }
        let p = a.into_raw_vec_and_offset().0;
        if p.iter().any(|v| *v < 0.) || (p.iter().sum::<f64>() - 1.).abs() > 1e-10 {
            return Err("invalid native attention probabilities".into());
        }
        Ok(p)
    }
}
struct Pair {
    old: usize,
    new: usize,
    fold: usize,
    key: Value,
}
fn pairs(cases: &[Case]) -> Result<Vec<Pair>, String> {
    let mut grouped: BTreeMap<String, [Option<usize>; 2]> = BTreeMap::new();
    for (index, c) in cases.iter().enumerate() {
        let mut factors = c.report.fixture["factors"]
            .as_object()
            .ok_or("factors missing")?
            .clone();
        let now = integer(&factors.remove("query_now").ok_or("query_now missing")?)?;
        if now > 1 {
            return Err("query_now must be binary".into());
        }
        let key = serde_json::to_string(&factors).map_err(err)?;
        if grouped.entry(key).or_insert([None, None])[now]
            .replace(index)
            .is_some()
        {
            return Err("duplicate pair condition".into());
        }
    }
    let mut result = vec![];
    for (key, slots) in grouped {
        let old = slots[0].ok_or("unpaired old condition")?;
        let new = slots[1].ok_or("unpaired new condition")?;
        let a = &cases[old].report.fixture;
        let b = &cases[new].report.fixture;
        let at = a["tokens"].as_array().ok_or("tokens missing")?;
        let bt = b["tokens"].as_array().ok_or("tokens missing")?;
        if bt.len() != at.len() + 1
            || &bt[..at.len()] != at
            || b["prompt"].as_str().ok_or("prompt missing")?
                != format!("{} now", a["prompt"].as_str().ok_or("prompt missing")?)
        {
            return Err("pair is not unchanged prefix plus exactly ' now'".into());
        }
        let factors: Value = serde_json::from_str(&key).map_err(err)?;
        let mut fold = 0;
        for name in [
            "query_owner",
            "old_statement_order",
            "update_statement_order",
            "payload_swap",
        ] {
            let bit = integer(&factors[name])?;
            if bit > 1 {
                return Err("crossfit factor must be binary".into());
            }
            fold ^= bit;
        }
        result.push(Pair {
            old,
            new,
            fold,
            key: factors,
        });
    }
    if result.len() != 32 {
        return Err("exactly 32 discovery pairs required".into());
    }
    for fold in 0..2 {
        let members = result.iter().filter(|p| p.fold == fold).collect::<Vec<_>>();
        let wordings = members
            .iter()
            .map(|p| integer(&p.key["update_wording"]))
            .collect::<Result<BTreeSet<_>, _>>()?;
        if members.len() != 16 || wordings != BTreeSet::from([0, 1]) {
            return Err("fold must have 16 contexts and both update wordings".into());
        }
    }
    Ok(result)
}
fn native_decl(site: &Value) -> Result<(Scale, Option<Rotary>), String> {
    let scale = site["scale"].as_str().ok_or("scale missing")?;
    let scale = if scale == "One" {
        Scale::One
    } else {
        Scale::InverseSqrt(
            scale
                .strip_prefix("InverseSqrt(")
                .and_then(|v| v.strip_suffix(')'))
                .ok_or("unsupported scale syntax")?
                .parse()
                .map_err(err)?,
        )
    };
    let text = site["rotary"].as_str().ok_or("rotary missing")?;
    let rotary = if text == "None" {
        None
    } else {
        let parts = text
            .strip_prefix("Some(Rotary { base: ")
            .and_then(|v| v.strip_suffix(" })"))
            .ok_or("unsupported rotary syntax")?;
        let (base, rest) = parts.split_once(", dims: ").ok_or("rotary dims missing")?;
        let (dims, half) = rest
            .split_once(", half_split: ")
            .ok_or("rotary layout missing")?;
        Some(Rotary {
            base: base.parse().map_err(err)?,
            dims: dims.parse().map_err(err)?,
            half_split: half.parse().map_err(err)?,
        })
    };
    Ok((scale, rotary))
}
fn prefix_difference(
    a: &Array2<f64>,
    b: &Array2<f64>,
    rows: usize,
    settings: &Settings,
) -> Result<Value, String> {
    if a.nrows() != rows || b.nrows() != rows + 1 || a.ncols() != b.ncols() {
        return Err("prefix producer shape mismatch".into());
    }
    let mut maximum = 0f64;
    let mut reference = 0f64;
    let mut exact = true;
    for (a, b) in a.iter().zip(b.slice(s![..rows, ..]).iter()) {
        maximum = maximum.max((a - b).abs());
        reference = reference.max(a.abs()).max(b.abs());
        exact &= a.to_bits() == b.to_bits();
        if (a - b).abs()
            > settings.prefix_absolute_tolerance
                + settings.prefix_relative_tolerance * a.abs().max(b.abs())
        {
            return Err(format!("causal prefix mismatch: {}", (a - b).abs()));
        }
    }
    Ok(
        json!({"bit_identical":exact,"maximum_absolute_error":maximum,"reference_maximum_absolute":reference,
        "absolute_tolerance":settings.prefix_absolute_tolerance,"relative_tolerance":settings.prefix_relative_tolerance}),
    )
}
fn spans(fixture: &Value, rows: usize) -> Result<BTreeMap<String, Vec<usize>>, String> {
    let mut result = BTreeMap::new();
    for span in fixture["spans"].as_array().ok_or("spans missing")? {
        let role = span["role"].as_str().ok_or("span role missing")?;
        if role.starts_with("old_value_") || role.starts_with("new_value_") {
            let indices = span["token_indices"]
                .as_array()
                .ok_or("span tokens missing")?
                .iter()
                .map(integer)
                .collect::<Result<Vec<_>, _>>()?;
            if indices.iter().any(|i| *i >= rows) {
                return Err("value span out of prefix".into());
            }
            result.insert(role.to_owned(), indices);
        }
    }
    if result.len() != 4 {
        return Err("four explicit value spans required".into());
    }
    Ok(result)
}
fn masses(p: &[f64], spans: &BTreeMap<String, Vec<usize>>) -> Value {
    let entries = spans
        .iter()
        .map(|(name, indices)| {
            (
                name.clone(),
                json!(indices.iter().map(|i| p[*i]).sum::<f64>()),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let indices = spans.values().flatten().copied().collect::<BTreeSet<_>>();
    json!({"by_human_specified_span":entries,"all_value_tokens":indices.iter().map(|i|p[*i]).sum::<f64>()})
}
fn summary(fit: &Fit) -> Value {
    let mut result = json!({"objective":fit.objective,"mean_kl":fit.mean_kl,"ridge_penalty":fit.ridge_penalty,
    "gradient_norm":fit.gradient_norm,"iterations":fit.iterations,"objective_evaluations":fit.objective_evaluations,"stop_reason":fit.stop_reason});
    if let Some(diagnostics) = &fit.newton_diagnostics {
        result["newton_diagnostics"] = json!(diagnostics);
    }
    result
}
fn payload(values: &Array2<f64>, p: &[f64]) -> Vec<f64> {
    values
        .t()
        .dot(&ndarray::Array1::from_vec(p.to_vec()))
        .to_vec()
}
fn vector_error(a: &[f64], b: &[f64]) -> Value {
    let squared = a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum::<f64>();
    let reference = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    json!({"l2_error":squared.sqrt(),"rmse":(squared/a.len() as f64).sqrt(),
        "maximum_absolute_error":a.iter().zip(b).map(|(a,b)|(a-b).abs()).fold(0f64,f64::max),
        "reference_l2_norm":reference,"relative_l2_error":if reference>0. {Some(squared.sqrt()/reference)} else {None}})
}
fn conditional_payload_error(sample: &Sample, delta: &[f64], values: &Array2<f64>) -> Value {
    let prediction = sample.log_probabilities(delta).mapv(f64::exp);
    vector_error(
        &payload(
            values,
            prediction.as_slice().expect("contiguous probabilities"),
        ),
        &payload(values, sample.target.as_slice().expect("contiguous target")),
    )
}
fn norm(delta: &[f64]) -> f64 {
    delta.iter().map(|v| v * v).sum::<f64>().sqrt()
}
fn run(root: &Path, out: &Path, settings: &Settings) -> Result<(), String> {
    let started = Instant::now();
    if root.file_name().and_then(|n| n.to_str()) != Some("ROUTING_DISCOVERY") {
        return Err("input must be ROUTING_DISCOVERY; heldout access is prohibited".into());
    }
    if out.exists() {
        return Err("output directory already exists".into());
    }
    for t in [
        settings.prefix_absolute_tolerance,
        settings.prefix_relative_tolerance,
        settings.oracle_probability_tolerance,
        settings.oracle_kl_tolerance,
    ] {
        if !t.is_finite() || t < 0. {
            return Err("invalid numerical tolerance".into());
        }
    }
    let report = read(&root.join("REPORT.json"))?;
    let graph = read(&root.join("NATIVE_GRAPH.json"))?;
    let provenance = read(&root.join("PROVENANCE.json"))?;
    let sites = graph["sites"].as_array().ok_or("sites missing")?;
    let heads = sites
        .iter()
        .map(|s| integer(&s["read_node"]))
        .collect::<Result<BTreeSet<_>, _>>()?;
    if sites.len() != 448
        || heads.len() != 448
        || report["head_count_per_case"] != 448
        || report["all_declared_cases_and_heads"].as_bool() != Some(true)
    {
        return Err("all 448 native heads required".into());
    }
    let selected = if let Some(ids) = &settings.diagnostic_read_nodes {
        let selected = ids.iter().copied().collect::<BTreeSet<_>>();
        if ids.is_empty() || selected.len() != ids.len() || !selected.is_subset(&heads) {
            return Err(
                "diagnostic_read_nodes must be a nonempty unique subset of native read nodes"
                    .into(),
            );
        }
        selected
    } else {
        heads.clone()
    };
    let all_declared_heads = selected == heads;
    let entries = report["cases"].as_array().ok_or("cases missing")?;
    if entries.len() != 64 {
        return Err("all 64 discovery cases required".into());
    }
    let mut cases = Vec::with_capacity(64);
    for entry in entries {
        cases.push(Case::load(root, entry, &heads)?);
    }
    let pair_list = pairs(&cases)?;
    let pair_manifest=pair_list.iter().map(|p|json!({"old_case":cases[p.old].name,"new_case":cases[p.new].name,
        "old_fixture_id":cases[p.old].report.fixture_id,"new_fixture_id":cases[p.new].report.fixture_id,
        "fold":p.fold,"factors_except_query_now":p.key,
        "old_tokens":cases[p.old].report.rows,"new_tokens":cases[p.new].report.rows})).collect::<Vec<_>>();
    std::fs::create_dir_all(out).map_err(err)?;
    save(
        &out.join("SETTINGS.json"),
        &serde_json::to_value(settings).map_err(err)?,
    )?;
    let mut reports = vec![];
    let mut deltas = vec![];
    for (head_index, site) in sites.iter().enumerate() {
        let read_node = integer(&site["read_node"])?;
        if !selected.contains(&read_node) {
            continue;
        }
        let query_node = integer(&site["query_node"])?;
        let key_node = integer(&site["key_node"])?;
        let value_node = integer(&site["value_node"])?;
        if site["causal"].as_bool() != Some(true) {
            return Err("noncausal native head".into());
        }
        let (scale, rotary) = native_decl(site)?;
        let mut samples = vec![];
        let mut native_deltas = vec![];
        let mut measurements = vec![];
        let mut value_spans = vec![];
        let mut prefix_values = vec![];
        for pair in &pair_list {
            let old = &cases[pair.old];
            let new = &cases[pair.new];
            let rows = old.report.rows;
            let old_keys = old.producer(key_node)?;
            let new_keys = new.producer(key_node)?;
            let key_check = prefix_difference(&old_keys, &new_keys, rows, settings)?;
            let old_values = old.producer(value_node)?;
            let new_values = new.producer(value_node)?;
            let value_check = prefix_difference(&old_values, &new_values, rows, settings)?;
            let old_query = old.producer(query_node)?;
            let new_query = new.producer(query_node)?;
            if old_query.ncols() != old_keys.ncols() || new_query.ncols() != old_keys.ncols() {
                return Err("query/key dimension mismatch".into());
            }
            let q_old = old_query.row(old_query.nrows() - 1).to_vec();
            let q_new = new_query.row(new_query.nrows() - 1).to_vec();
            let old_attention = old.attention(read_node)?;
            let new_attention = new.attention(read_node)?;
            let prefix_mass = new_attention[..rows].iter().sum::<f64>();
            if prefix_mass <= 0. {
                return Err("target prefix has zero probability mass".into());
            }
            let target = new_attention[..rows]
                .iter()
                .map(|v| v / prefix_mass)
                .collect::<Vec<_>>();
            let features = native_features(
                &old_keys,
                &(0..rows as u32).collect::<Vec<_>>(),
                rows as u32,
                scale,
                rotary,
            )?;
            let sample = Sample::new(features, &q_old, target.clone())?;
            let oracle = q_new
                .iter()
                .zip(&q_old)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            let oracle_p = sample.log_probabilities(&oracle).mapv(f64::exp);
            let oracle_error = oracle_p
                .iter()
                .zip(&target)
                .map(|(a, b)| (a - b).abs())
                .fold(0f64, f64::max);
            let oracle_kl = sample.kl(&oracle);
            if oracle_error > settings.oracle_probability_tolerance
                || oracle_kl > settings.oracle_kl_tolerance
            {
                return Err(format!(
                    "oracle reconstruction fails at head {read_node} pair {}: probability {oracle_error}, KL {oracle_kl}",
                    old.name
                ));
            }
            let old_features = native_features(
                &old_keys,
                &(0..rows as u32).collect::<Vec<_>>(),
                (rows - 1) as u32,
                scale,
                rotary,
            )?;
            let old_sample = Sample::new(old_features, &q_old, old_attention.clone())?;
            let old_logp = old_sample.log_probabilities(&vec![0.; q_old.len()]);
            let old_reconstruction_error = old_logp
                .iter()
                .zip(&old_attention)
                .map(|(lp, p)| (lp.exp() - p).abs())
                .fold(0f64, f64::max);
            if old_reconstruction_error > settings.oracle_probability_tolerance {
                return Err("previous-position native reconstruction fails".into());
            }
            // KL(target new conditional || native old) is distinct from rotation-only baseline.
            let native_change_kl = target
                .iter()
                .zip(old_logp.iter())
                .filter(|(p, _)| **p > 0.)
                .map(|(p, logq)| p * (p.ln() - logq))
                .sum::<f64>();
            if !native_change_kl.is_finite() {
                return Err("nonfinite native conditional change KL".into());
            }
            let native_change_tv = 0.5
                * target
                    .iter()
                    .zip(&old_attention)
                    .map(|(p, q)| (p - q).abs())
                    .sum::<f64>();
            let spans = spans(&old.report.fixture, rows)?;
            measurements.push(json!({"old_case":old.name,"new_case":new.name,"fold":pair.fold,"prefix_rows":rows,
                "native_query_width":q_old.len(),"native_value_width":old_values.ncols(),"target_mass_over_unchanged_prefix":prefix_mass,
                "new_token_attention_mass":new_attention[rows],"native_conditional_change_kl":native_change_kl,"native_conditional_change_tv":native_change_tv,
                "conditioning_payload_difference_from_full_native_new_head":vector_error(&payload(&old_values,&target),&payload(&new_values,&new_attention)),
                "native_conditional_payload_change_from_old_head":vector_error(&payload(&old_values,&target),&payload(&old_values,&old_attention)),
                "native_old_value_token_mass":masses(&old_attention,&spans),"target_conditional_value_token_mass":masses(&target,&spans),
                "prefix_keys":key_check,"prefix_values":value_check,"oracle_probability_max_absolute_error":oracle_error,
                "oracle_kl":oracle_kl,"old_native_reconstruction_max_absolute_error":old_reconstruction_error,"native_delta_norm":norm(&oracle)}));
            samples.push(sample);
            native_deltas.push(oracle);
            value_spans.push(spans);
            prefix_values.push(old_values);
        }
        let width = samples[0].features.ncols();
        let zero = vec![0.; width];
        let mean_delta = |indices: &[usize]| -> Vec<f64> {
            (0..width)
                .map(|d| {
                    indices.iter().map(|i| native_deltas[*i][d]).sum::<f64>() / indices.len() as f64
                })
                .collect()
        };
        let all = (0..samples.len()).collect::<Vec<_>>();
        let mean = mean_delta(&all);
        let final_fit = fit(&samples.iter().collect::<Vec<_>>(), &settings.solver)?;
        let mut fold_reports = vec![];
        let mut fold_deltas = vec![];
        let mut crossfit_metrics = vec![Value::Null; samples.len()];
        for train_fold in 0..2 {
            let train = all
                .iter()
                .copied()
                .filter(|i| pair_list[*i].fold == train_fold)
                .collect::<Vec<_>>();
            let test = all
                .iter()
                .copied()
                .filter(|i| pair_list[*i].fold != train_fold)
                .collect::<Vec<_>>();
            let fitted = fit(
                &train.iter().map(|i| &samples[*i]).collect::<Vec<_>>(),
                &settings.solver,
            )?;
            let mean = mean_delta(&train);
            let mut score = 0.;
            let mut mean_score = 0.;
            for i in &test {
                let kl = samples[*i].kl(&fitted.delta);
                let mean_kl = samples[*i].kl(&mean);
                score += kl;
                mean_score += mean_kl;
                crossfit_metrics[*i] = json!({"train_fold":train_fold,"fit_kl":kl,"mean_native_delta_kl":mean_kl,
                    "fit_conditional_payload_error":conditional_payload_error(&samples[*i],&fitted.delta,&prefix_values[*i]),
                    "mean_native_delta_conditional_payload_error":conditional_payload_error(&samples[*i],&mean,&prefix_values[*i]),
                    "fit_value_token_mass":masses(samples[*i].log_probabilities(&fitted.delta).mapv(f64::exp).as_slice().ok_or("contiguous probabilities")?,&value_spans[*i]),
                    "mean_native_delta_value_token_mass":masses(samples[*i].log_probabilities(&mean).mapv(f64::exp).as_slice().ok_or("contiguous probabilities")?,&value_spans[*i])});
            }
            fold_reports.push(json!({"train_fold":train_fold,"train_pairs":train,"test_pairs":test,"training_fit":summary(&fitted),
                "test_mean_fit_kl":score/test.len() as f64,"test_mean_native_delta_kl":mean_score/test.len() as f64}));
            fold_deltas.push(
                json!({"train_fold":train_fold,"fit_delta":fitted.delta,"mean_native_delta":mean}),
            );
        }
        for (i, m) in measurements.iter_mut().enumerate() {
            m["rotation_only_kl"] = json!(samples[i].kl(&zero));
            m["all_discovery_mean_native_delta_kl"] = json!(samples[i].kl(&mean));
            m["all_discovery_fit_kl"] = json!(samples[i].kl(&final_fit.delta));
            m["crossfit"] = crossfit_metrics[i].clone();
            m["rotation_only_conditional_payload_error"] =
                conditional_payload_error(&samples[i], &zero, &prefix_values[i]);
            m["all_discovery_mean_native_delta_conditional_payload_error"] =
                conditional_payload_error(&samples[i], &mean, &prefix_values[i]);
            m["all_discovery_fit_conditional_payload_error"] =
                conditional_payload_error(&samples[i], &final_fit.delta, &prefix_values[i]);
            m["rotation_only_value_token_mass"] = masses(
                samples[i]
                    .log_probabilities(&zero)
                    .mapv(f64::exp)
                    .as_slice()
                    .ok_or("contiguous probabilities")?,
                &value_spans[i],
            );
            m["all_discovery_fit_value_token_mass"] = masses(
                samples[i]
                    .log_probabilities(&final_fit.delta)
                    .mapv(f64::exp)
                    .as_slice()
                    .ok_or("contiguous probabilities")?,
                &value_spans[i],
            );
        }
        let average = |field: &str| -> f64 {
            measurements
                .iter()
                .map(|v| v[field].as_f64().expect("numeric metric"))
                .sum::<f64>()
                / measurements.len() as f64
        };
        reports.push(json!({"head_index":head_index,"native_site":site,"pairs":measurements,"folds":fold_reports,"all_discovery_fit":summary(&final_fit),
            "mean_rotation_only_kl":average("rotation_only_kl"),"mean_native_delta_kl":average("all_discovery_mean_native_delta_kl"),
            "mean_native_conditional_change_kl":average("native_conditional_change_kl"),"mean_native_conditional_change_tv":average("native_conditional_change_tv"),
            "mean_target_prefix_mass":average("target_mass_over_unchanged_prefix"),"crossfit_mean_fit_kl":crossfit_metrics.iter().map(|v|v["fit_kl"].as_f64().expect("numeric crossfit KL")).sum::<f64>()/samples.len() as f64,
            "crossfit_mean_native_delta_kl":crossfit_metrics.iter().map(|v|v["mean_native_delta_kl"].as_f64().expect("numeric crossfit KL")).sum::<f64>()/samples.len() as f64,
            "delta_dimension":width,"delta_norm":norm(&final_fit.delta)}));
        deltas.push(json!({"read_node":read_node,"query_node":query_node,"native_width":width,"all_discovery_fit_delta":final_fit.delta,
            "all_discovery_mean_native_delta":mean,"discovery_crossfit":fold_deltas}));
        eprintln!(
            "head {}/{} read_node={} done elapsed={:.1}s",
            reports.len(),
            selected.len(),
            read_node,
            started.elapsed().as_secs_f64()
        );
    }
    let provenance = json!({"input_directory":root,"source_report_sha256":sha256(&root.join("REPORT.json"))?,
        "native_graph_sha256":sha256(&root.join("NATIVE_GRAPH.json"))?,"source_provenance_sha256":sha256(&root.join("PROVENANCE.json"))?,
        "source_provenance":provenance,"cases":cases.iter().map(|c|c.provenance.clone()).collect::<Vec<_>>(),"pairs":pair_manifest,
        "selection_scope":if all_declared_heads {"all declared heads"} else {"explicit selected-head optimizer diagnostic only"},
        "declared_head_count":448,"reported_read_nodes":selected,
        "executable_sha256":sha256(&std::env::current_exe().map_err(err)?)?,"settings_sha256":sha256(&out.join("SETTINGS.json"))?,
        "heldout_opened_or_evaluated":false,"native_rotary_implementation":"gam_mpd::tiled_attention::rotate (reused)",
        "factor_scope":"discovery fixtures, lexical_set fixed as supplied; final ' now' appended only"});
    save(&out.join("PROVENANCE.json"), &provenance)?;
    save(
        &out.join("DELTAS.json"),
        &json!({"schema":"native-query-translation-deltas-v1","coordinate_space":"original native query producer output, pre-RoPE",
        "prediction":"q_new_candidate=q_old+one_shared_delta_per_head","provenance_sha256":sha256(&out.join("PROVENANCE.json"))?,"heads":deltas}),
    )?;
    save(
        &out.join("REPORT.json"),
        &json!({"schema":"native-query-translation-discovery-fit-v1","settings":settings,"heads":reports,"pairs":pair_manifest,
        "head_count":reports.len(),"declared_head_count":448,"reported_read_nodes":selected,
        "selection_scope":if all_declared_heads {"all declared heads"} else {"explicit selected-head optimizer diagnostic only; no full-head-panel claim"},
        "discovery_pairs":32,"all_declared_heads_reported":all_declared_heads,"seconds":started.elapsed().as_secs_f64(),
        "delta_artifact":"DELTAS.json","delta_artifact_sha256":sha256(&out.join("DELTAS.json"))?,"provenance_sha256":sha256(&out.join("PROVENANCE.json"))?,
        "objective":"mean KL(target new conditional attention || candidate conditional attention) + (ridge/2)*||delta||^2; uniform pair weighting",
        "solver":match settings.solver.method {SolverMethod::Lbfgs=>"L-BFGS with smooth true-objective Armijo backtracking, zero initialization; no global optimality certificate",
            SolverMethod::DampedNewton=>"Exact centered softmax feature-covariance Hessian plus ridge, gam-linalg SPD Cholesky direction with damping safeguards, true-objective Armijo backtracking, zero initialization; numerical stationarity diagnostics, no interval-arithmetic optimality certificate"},
        "head_execution":"sequential native graph site order; selection specified explicitly in settings, all448 by default",
        "native_change_kl_numerics":"native old log probabilities reconstructed from original native query/key/rotary/scale; no probability floor",
        "crossfit":"2 discovery-only folds: XOR(query_owner, old_statement_order, update_statement_order, payload_swap); query_now excluded; both update wordings in each fold",
        "prediction_inputs":"previous-position native query and unchanged causal prefix native keys; target new query used only as training baseline measurement and oracle instrument check",
        "target":"new native attention restricted to old prefix and renormalized; excludes appended token",
        "limitations":["Scoped local control-transition hypothesis, not recovered native computational organization or representation hierarchy",
            "High-dimensional native upstream query/key/value states are retained; no whole-model or token KL measured",
            "No parameter edit or whole-tail response; oracle validates instrument only",
            "Discovery crossfit is not untouched heldout evaluation; heldout not opened",
            "A selected-head diagnostic addresses optimization of a fixed family only; no broad full-panel acceptance claim",
            "Trivially unchanged or unused heads are not promoted as mechanism; native change and value-token masses reported for every head",
            "Value-token spans are human-specified fixture annotations, not discovered semantic roles",
            "Layer-zero query translation can be exactly constant because is/now token changes are token-local; architecture/token identity control, not learned context-dependent retrieval organization"]}),
    )?;
    Ok(())
}
fn self_check() -> Result<(), String> {
    let features =
        Array2::from_shape_vec((4, 2), vec![1., 0., 0., 1., -1., -1., 0.5, -0.7]).map_err(err)?;
    let truth = vec![0.7, -0.4];
    let mut samples = vec![];
    for query in [vec![-0.2, 0.5], vec![0.3, -0.1], vec![0.8, 0.6]] {
        let mut s = Sample::new(features.clone(), &query, vec![0.25; 4])?;
        s.target = s.log_probabilities(&truth).mapv(f64::exp);
        samples.push(s);
    }
    let result = fit(
        &samples.iter().collect::<Vec<_>>(),
        &SolverSettings {
            ridge: 1e-10,
            gradient_tolerance: 1e-10,
            ..Default::default()
        },
    )?;
    if result.mean_kl > 1e-12
        || result
            .delta
            .iter()
            .zip(truth)
            .any(|(a, b)| (a - b).abs() > 1e-6)
    {
        return Err("translated-query calibration failed".into());
    }
    let newton = fit(
        &samples.iter().collect::<Vec<_>>(),
        &SolverSettings {
            method: SolverMethod::DampedNewton,
            ridge: 1e-10,
            gradient_tolerance: 1e-10,
            ..Default::default()
        },
    )?;
    if newton.mean_kl > 1e-12 || newton.gradient_norm > 1e-9 {
        return Err("Newton translated-query calibration failed".into());
    }
    println!(
        "{}",
        json!({"synthetic_calibration":"pass","fit":result,"newton_fit":newton})
    );
    Ok(())
}
fn main() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args == ["--self-check"] {
        return self_check();
    }
    if !(2..=3).contains(&args.len()) {
        return Err("usage: mpd_query_transition_fit_2951 DISCOVERY_DIR OUT_DIR [SETTINGS.json] | --self-check".into());
    }
    let settings = if args.len() == 3 {
        serde_json::from_value(read(&PathBuf::from(&args[2]))?).map_err(err)?
    } else {
        Settings::default()
    };
    run(Path::new(&args[0]), Path::new(&args[1]), &settings)
}

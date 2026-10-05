//! Scoped native query-invocation transition; retained native prefix/background, not discovery.
use gam_mpd::{
    coder_capture::sha256,
    import::import_language_model,
    operator_program::{FamilyInputs, Node, SequenceLayout, SlotValues},
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    path::Path,
    time::Instant,
};
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn load(p: &Path) -> Result<Value, String> {
    serde_json::from_slice(&std::fs::read(p).map_err(err)?).map_err(err)
}
fn save(p: &Path, v: &Value) -> Result<(), String> {
    std::fs::write(p, serde_json::to_vec_pretty(v).map_err(err)?).map_err(err)
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Pair {
    previous_fixture_id: String,
    appended_fixture_id: String,
}
#[derive(Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum DeltaSource {
    Crossfit,
    AllDiscovery,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    delta_source: DeltaSource,
    fixtures_sha256: String,
    tokenizer_sha256: String,
    export_sha256: String,
    delta_report_sha256: String,
    checkpoint_sha256: String,
    pairs: Vec<Pair>,
    read_nodes: Vec<usize>,
    joint: bool,
    numeric_bytes: usize,
    absolute_tolerance: f64,
}
fn replace_last(value: &mut Array2<f64>, delta: &[f64]) -> Result<(), String> {
    if value.nrows() < 2 || value.ncols() != delta.len() || delta.iter().any(|x| !x.is_finite()) {
        return Err("invalid previous-row query transition".into());
    }
    let last = value.nrows() - 1;
    for col in 0..value.ncols() {
        value[[last, col]] = value[[last - 1, col]] + delta[col];
    }
    if value.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite query transition".into());
    }
    Ok(())
}
fn logp(v: &[f64]) -> Result<Vec<f64>, String> {
    if v.is_empty() || v.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite/empty final logits".into());
    }
    let peak = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z = peak + v.iter().map(|x| (x - peak).exp()).sum::<f64>().ln();
    Ok(v.iter().map(|x| x - z).collect())
}
fn metrics(native: &[f64], candidate: &[f64], target: usize, foil: usize) -> Result<Value, String> {
    if native.len() != candidate.len()
        || target >= native.len()
        || foil >= native.len()
        || target == foil
    {
        return Err("invalid vocabulary targets".into());
    }
    let p = logp(native)?;
    let q = logp(candidate)?;
    let kl = p
        .iter()
        .zip(&q)
        .map(|(p, q)| p.exp() * (p - q))
        .sum::<f64>();
    let native_odds = native[target] - native[foil];
    let odds = candidate[target] - candidate[foil];
    Ok(
        json!({"teacher_kl":kl,"native_updated_minus_old_log_odds":native_odds,"candidate_updated_minus_old_log_odds":odds,"log_odds_error":odds-native_odds,"updated_log_probability_error":q[target]-p[target],"old_log_probability_error":q[foil]-p[foil],"max_logit_difference":native.iter().zip(candidate).map(|(a,b)|(a-b).abs()).fold(0.,f64::max)}),
    )
}
fn tokens(case: &Value) -> Result<Vec<u32>, String> {
    serde_json::from_value(case["tokens"].clone()).map_err(err)
}
fn family(tokens: Vec<u32>) -> FamilyInputs {
    let rows = tokens.len();
    FamilyInputs {
        rows,
        slots: vec![SlotValues::Tokens(tokens)],
        layout: Some(SequenceLayout {
            sequence: vec![0; rows],
            position: (0..rows as u32).collect(),
        }),
    }
}
fn main() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 6 {
        return Err("EXPORT DISCOVERY TOKENIZER DELTA_REPORT SETTINGS OUT_DIR".into());
    }
    let export = Path::new(&args[0]);
    let fixtures_path = Path::new(&args[1]);
    let delta_path = Path::new(&args[3]);
    let settings: Settings = serde_json::from_value(load(Path::new(&args[4]))?).map_err(err)?;
    let out = Path::new(&args[5]);
    if out.exists() {
        return Err("fresh output directory required".into());
    }
    if settings.pairs.is_empty()
        || settings.read_nodes.is_empty()
        || settings.numeric_bytes == 0
        || !settings.absolute_tolerance.is_finite()
        || settings.absolute_tolerance < 0.
    {
        return Err("invalid explicit scope/budget/tolerance".into());
    }
    for (path, expected) in [
        (fixtures_path, settings.fixtures_sha256.as_str()),
        (Path::new(&args[2]), settings.tokenizer_sha256.as_str()),
        (delta_path, settings.delta_report_sha256.as_str()),
        (
            export.join("export.json").as_path(),
            settings.export_sha256.as_str(),
        ),
    ] {
        if expected.len() != 64 || sha256(path)? != expected {
            return Err(format!("input hash mismatch {}", path.display()));
        }
    }
    let fixtures = load(fixtures_path)?;
    if fixtures["version"].as_u64() != Some(1)
        || fixtures["panel"].as_str() != Some("discovery")
        || fixtures["tokenizer_sha256"].as_str() != Some(settings.tokenizer_sha256.as_str())
    {
        return Err("only hash-verified discovery deck supported; heldout forbidden".into());
    }
    let cases = fixtures["cases"].as_array().ok_or("missing cases")?;
    let lookup = |id: &str| -> Result<&Value, String> {
        let matches = cases
            .iter()
            .filter(|c| c["id"].as_str() == Some(id))
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(format!("fixture {id} absent/duplicate"));
        }
        Ok(matches[0])
    };
    let mut ids = BTreeSet::new();
    for pair in &settings.pairs {
        if !ids.insert(pair.appended_fixture_id.clone()) {
            return Err("duplicate appended fixture".into());
        }
        let previous = lookup(&pair.previous_fixture_id)?;
        let appended = lookup(&pair.appended_fixture_id)?;
        let a = tokens(previous)?;
        let b = tokens(appended)?;
        if b.len() != a.len() + 1
            || b[..a.len()] != a
            || appended["factors"]["query_now"].as_u64() != Some(1)
        {
            return Err(
                "declared transition must append exactly one token to verified old prompt".into(),
            );
        }
    }
    let mut reads = BTreeSet::new();
    if settings.read_nodes.iter().any(|id| !reads.insert(*id)) {
        return Err("duplicate selected read node".into());
    }
    let started = Instant::now();
    let imported = import_language_model(export, 1, 1)?;
    for key in ["checkpoint_sha256", "weights_sha256"] {
        if let Some(v) = imported.record["source"][key].as_str() {
            if v != settings.checkpoint_sha256 {
                return Err("checkpoint/weight lineage mismatch".into());
            }
        }
    }
    if imported.record["source"]["checkpoint_sha256"]
        .as_str()
        .or_else(|| imported.record["source"]["weights_sha256"].as_str())
        .is_none()
    {
        return Err("missing native lineage".into());
    }
    for (name, metadata) in imported.record["files"]
        .as_object()
        .ok_or("missing native files")?
    {
        if name != "tokens"
            && metadata["sha256"].as_str()
                != Some(sha256(&export.join(format!("{name}.f64")))?.as_str())
        {
            return Err(format!("native file mismatch {name}"));
        }
    }
    let mut query_nodes = BTreeMap::new();
    for read in &settings.read_nodes {
        match imported.program.nodes.get(*read) {
            Some(Node::Attend { query, .. }) => {
                query_nodes.insert(*read, *query);
            }
            _ => return Err(format!("selected node {read} is not native Attend")),
        }
    }
    if query_nodes.values().collect::<BTreeSet<_>>().len() != query_nodes.len() {
        return Err(
            "selected reads alias one query producer; explicit disambiguation required".into(),
        );
    }
    let delta_report = load(delta_path)?;
    // Frozen fitter artifact schema is validated below; no fitted state is inferred from native test outputs.
    if delta_report["schema"].as_str() != Some("native-query-translation-deltas-v1")
        || delta_report["coordinate_space"].as_str()
            != Some("original native query producer output, pre-RoPE")
    {
        return Err("unsupported frozen delta schema/coordinates".into());
    }
    let provenance_path = delta_path
        .parent()
        .ok_or("delta parent missing")?
        .join("PROVENANCE.json");
    if delta_report["provenance_sha256"].as_str() != Some(sha256(&provenance_path)?.as_str()) {
        return Err("delta provenance hash mismatch".into());
    }
    let delta_provenance = load(&provenance_path)?;
    let source = &delta_provenance["source_provenance"];
    for (field, expected) in [
        ("fixtures_sha256", settings.fixtures_sha256.as_str()),
        ("tokenizer_sha256", settings.tokenizer_sha256.as_str()),
        ("export_sha256", settings.export_sha256.as_str()),
    ] {
        if source[field].as_str() != Some(expected) {
            return Err(format!("frozen delta source mismatch {field}"));
        }
    }
    let entries = delta_report["heads"]
        .as_array()
        .ok_or("delta report heads missing")?;
    let mut deltas = BTreeMap::new();
    let mut frozen_heads = BTreeMap::new();
    for read in &settings.read_nodes {
        let matching = entries
            .iter()
            .filter(|v| v["read_node"].as_u64() == Some(*read as u64))
            .collect::<Vec<_>>();
        if matching.len() != 1 {
            return Err(format!("delta head {read} absent/duplicate"));
        }
        let e = matching[0];
        if e["query_node"].as_u64() != Some(query_nodes[read] as u64) {
            return Err("delta query identity mismatch".into());
        }
        let fitted: Vec<f64> =
            serde_json::from_value(e["all_discovery_fit_delta"].clone()).map_err(err)?;
        let mean: Vec<f64> =
            serde_json::from_value(e["all_discovery_mean_native_delta"].clone()).map_err(err)?;
        let width = imported.program.interfaces().map_err(err)?[query_nodes[read]].width();
        if fitted.len() != width
            || mean.len() != width
            || fitted.iter().chain(&mean).any(|x| !x.is_finite())
        {
            return Err("delta shape/finiteness mismatch".into());
        }
        deltas.insert(*read, (fitted, mean));
        frozen_heads.insert(*read, e.clone());
    }
    std::fs::create_dir_all(out).map_err(err)?;
    save(
        &out.join("SETTINGS.json"),
        &serde_json::to_value(&settings).map_err(err)?,
    )?;
    save(
        &out.join("PROVENANCE.json"),
        &json!({"native_export":imported.record,"settings_sha256":sha256(Path::new(&args[4]))?,"delta_report_sha256":settings.delta_report_sha256,"scope":"scoped native query invocation edit; native upstream/background and all K/V/downstream retained; no whole-model discovery or acceptance certificate","modes":["zero","fitted","mean_native_delta","oracle_identity"],"arithmetic":"Host binary64 operational full-vocabulary softmax/KL"}),
    )?;
    let interfaces = imported.program.interfaces().map_err(err)?;
    let operator_bytes = imported
        .program
        .operators
        .iter()
        .try_fold(0usize, |a, o| {
            a.checked_add(o.rows.width().checked_mul(o.cols.width())?.checked_mul(8)?)
        })
        .ok_or("operator budget overflow")?;
    let mut journal = std::fs::File::create(out.join("JOURNAL.jsonl")).map_err(err)?;
    let mut results = Vec::new();
    for pair in &settings.pairs {
        let case = lookup(&pair.appended_fixture_id)?;
        let mut pair_deltas = deltas.clone();
        if settings.delta_source == DeltaSource::Crossfit {
            let mut fold = 0usize;
            for name in [
                "query_owner",
                "old_statement_order",
                "update_statement_order",
                "payload_swap",
            ] {
                let bit = case["factors"][name]
                    .as_u64()
                    .ok_or("binary crossfit factor missing")?;
                if bit > 1 {
                    return Err("crossfit factor not binary".into());
                }
                fold ^= bit as usize;
            }
            let declared = delta_provenance["pairs"]
                .as_array()
                .ok_or("frozen pair folds missing")?
                .iter()
                .filter(|p| {
                    p["old_fixture_id"].as_str() == Some(pair.previous_fixture_id.as_str())
                        && p["new_fixture_id"].as_str() == Some(pair.appended_fixture_id.as_str())
                })
                .collect::<Vec<_>>();
            if declared.len() != 1
                || declared[0]["fold"].as_u64() != Some(fold as u64)
                || declared[0]["old_tokens"] != lookup(&pair.previous_fixture_id)?["tokens"]
                || declared[0]["new_tokens"] != case["tokens"]
            {
                return Err("frozen pair/fold/token lineage mismatch".into());
            }
            for read in &settings.read_nodes {
                let entries = frozen_heads[read]["discovery_crossfit"]
                    .as_array()
                    .ok_or("crossfit deltas missing")?;
                let chosen = entries
                    .iter()
                    .filter(|v| v["train_fold"].as_u64() == Some((1 - fold) as u64))
                    .collect::<Vec<_>>();
                if chosen.len() != 1 {
                    return Err("crossfit opposite train fold absent/duplicate".into());
                }
                let fit: Vec<f64> =
                    serde_json::from_value(chosen[0]["fit_delta"].clone()).map_err(err)?;
                let mean: Vec<f64> =
                    serde_json::from_value(chosen[0]["mean_native_delta"].clone()).map_err(err)?;
                if fit.len() != deltas[read].0.len()
                    || mean.len() != fit.len()
                    || fit.iter().chain(&mean).any(|x| !x.is_finite())
                {
                    return Err("crossfit delta shape/finiteness mismatch".into());
                }
                pair_deltas.insert(*read, (fit, mean));
            }
        }
        let old = lookup(&pair.previous_fixture_id)?;
        let new_tokens = tokens(case)?;
        let rows = new_tokens.len();
        let planned = interfaces
            .iter()
            .try_fold(operator_bytes, |a, i| {
                a.checked_add(
                    rows.checked_mul(i.width())?
                        .checked_mul(8)?
                        .checked_mul(4)?,
                )
            })
            .and_then(|v| v.checked_add(rows.checked_mul(rows)?.checked_mul(96)?))
            .ok_or("trace plan overflow")?;
        if planned > settings.numeric_bytes {
            return Err(format!("numeric plan {planned} exceeds declared budget"));
        }
        let inputs = family(new_tokens);
        let native = imported.program.execute(&inputs, false).map_err(err)?;
        let old_trace = imported
            .program
            .execute(&family(tokens(old)?), false)
            .map_err(err)?;
        for query in query_nodes.values() {
            let a = native.values[*query].row(rows - 2);
            let b = old_trace.values[*query].row(rows - 2);
            let error = a
                .iter()
                .zip(b)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max);
            if error > settings.absolute_tolerance {
                return Err(format!("causal prefix-copy mismatch {error}"));
            }
        }
        let teacher = native.values[imported.program.output]
            .row(rows - 1)
            .to_vec();
        let target = case["hypothesis_targets"]["updated_owner_value"]["id"]
            .as_u64()
            .ok_or("updated target missing")? as usize;
        let foil = case["hypothesis_targets"]["old_owner_value"]["id"]
            .as_u64()
            .ok_or("old target missing")? as usize;
        let mut subsets = settings
            .read_nodes
            .iter()
            .map(|id| vec![*id])
            .collect::<Vec<_>>();
        if settings.joint && settings.read_nodes.len() > 1 {
            subsets.push(settings.read_nodes.clone())
        }
        for subset in subsets {
            for mode in ["zero", "fitted", "mean_native_delta", "oracle_identity"] {
                let patches = subset
                    .iter()
                    .map(|read| {
                        let query = query_nodes[read];
                        let delta = match mode {
                            "fitted" => pair_deltas[read].0.clone(),
                            "mean_native_delta" => pair_deltas[read].1.clone(),
                            _ => vec![0.; native.values[query].ncols()],
                        };
                        (query, delta)
                    })
                    .collect::<BTreeMap<_, _>>();
                let t = Instant::now();
                let edited = imported
                    .program
                    .execute_edited(&inputs, |node, value, _| {
                        if let Some(delta) = patches.get(&node) {
                            if mode == "oracle_identity" {
                                value
                                    .row_mut(rows - 1)
                                    .assign(&native.values[node].row(rows - 1));
                                Ok(())
                            } else {
                                replace_last(value, delta)
                            }
                        } else {
                            Ok(())
                        }
                    })
                    .map_err(err)?;
                let candidate = edited.values[imported.program.output]
                    .row(rows - 1)
                    .to_vec();
                let record = json!({"previous_fixture_id":pair.previous_fixture_id,"appended_fixture_id":pair.appended_fixture_id,"read_nodes":subset,"query_nodes":patches.keys().collect::<Vec<_>>(),"mode":mode,"seconds":t.elapsed().as_secs_f64(),"metrics":metrics(&teacher,&candidate,target,foil)?});
                writeln!(journal, "{}", serde_json::to_string(&record).map_err(err)?)
                    .map_err(err)?;
                journal.flush().map_err(err)?;
                results.push(record);
            }
        }
    }
    save(
        &out.join("REPORT.json"),
        &json!({"records":results,"elapsed_seconds":started.elapsed().as_secs_f64(),"scope":"discovery-panel scoped full-tail query edit measurement; no heldout, no updates, no ranking/selection"}),
    )?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn previous_row_copy_and_delta() {
        let mut x = Array2::from_shape_vec((3, 2), vec![1., 2., 3., 4., 9., 8.]).expect("fixture");
        replace_last(&mut x, &[0.5, -1.]).expect("patch");
        assert_eq!(x.row(2).to_vec(), vec![3.5, 3.]);
        assert_eq!(x.row(0).to_vec(), vec![1., 2.]);
    }
    #[test]
    fn causal_attention_prefix_and_full_tail_identity() {
        use gam_mpd::operator_program::{Declarations, OperatorProgram, Rotary, Scale, Slot};
        let p = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![
                    Slot::Raw { width: 2 },
                    Slot::Raw { width: 2 },
                    Slot::Raw { width: 2 },
                ],
                parameters: 0,
            },
            bases: vec![],
            operators: vec![],
            rules: vec![],
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Raw { slot: 1 },
                Node::Raw { slot: 2 },
                Node::Attend {
                    query: 0,
                    key: 1,
                    value: 2,
                    scale: Scale::InverseSqrt(2),
                    rotary: Some(Rotary {
                        base: 10000,
                        dims: 2,
                        half_split: true,
                    }),
                    causal: true,
                },
            ],
            output: 3,
        };
        let x = ndarray::array![[0.2, 0.4], [0.7, -0.1], [0.1, 0.8]];
        let inputs = FamilyInputs {
            rows: 3,
            slots: vec![
                SlotValues::Raw(x.clone()),
                SlotValues::Raw(x.clone()),
                SlotValues::Raw(&x * 3.),
            ],
            layout: Some(SequenceLayout {
                sequence: vec![0; 3],
                position: vec![0, 1, 2],
            }),
        };
        let native = p.execute(&inputs, false).expect("native tiny attention");
        let identity = p
            .execute_edited(&inputs, |node, value, _| {
                if node == 0 {
                    value.row_mut(2).assign(&native.values[0].row(2));
                }
                Ok(())
            })
            .expect("oracle identity");
        assert_eq!(native.values[3], identity.values[3]);
        let edited = p
            .execute_edited(&inputs, |node, value, _| {
                if node == 0 {
                    replace_last(value, &[0., 0.])?;
                }
                Ok(())
            })
            .expect("zero translation");
        assert_eq!(native.values[1], edited.values[1]);
        assert_eq!(native.values[2], edited.values[2]);
        assert_eq!(native.values[3].row(0), edited.values[3].row(0));
        assert_eq!(native.values[3].row(1), edited.values[3].row(1));
        assert_ne!(native.values[3].row(2), edited.values[3].row(2));
        let old = FamilyInputs {
            rows: 2,
            slots: inputs
                .slots
                .iter()
                .map(|slot| match slot {
                    SlotValues::Raw(v) => {
                        SlotValues::Raw(v.slice(ndarray::s![0..2, ..]).to_owned())
                    }
                    _ => panic!("fixture raw only"),
                })
                .collect(),
            layout: Some(SequenceLayout {
                sequence: vec![0; 2],
                position: vec![0, 1],
            }),
        };
        let previous = p.execute(&old, false).expect("old prefix");
        assert_eq!(previous.values[0].row(1), native.values[0].row(1));
        assert_eq!(previous.values[3].row(1), native.values[3].row(1));
    }
    #[test]
    fn oracle_logits_identity() {
        let x = vec![1000., -1000., 3.];
        let v = metrics(&x, &x, 0, 1).expect("metrics");
        assert_eq!(v["teacher_kl"].as_f64(), Some(0.));
        assert_eq!(v["log_odds_error"].as_f64(), Some(0.));
    }
}

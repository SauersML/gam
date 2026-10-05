//! EXPORT DISCOVERY.json TOKENIZER SETTINGS.json OUT_DIR.
//! A supplied native attention/value transport intervention, not discovery of
//! query construction, a representation hierarchy, or a retrieval algorithm.
use gam_mpd::{
    coder_capture::sha256,
    import::import_language_model,
    operator_program::{
        Declarations, FamilyInputs, Node, OperatorProgram, Rotary, Scale, SequenceLayout, Slot,
        SlotValues,
    },
};
use ndarray::{Array2, s};
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
struct Settings {
    read_node: usize,
    epsilon_fraction: f64,
    fixtures_sha256: String,
    tokenizer_sha256: String,
    export_sha256: String,
    checkpoint_sha256: String,
    numeric_bytes: usize,
    absolute_tolerance: f64,
    relative_tolerance: f64,
}
#[derive(Clone, Copy)]
struct Site {
    read: usize,
    query: usize,
    key: usize,
    value: usize,
    scale: Scale,
    rotary: Option<Rotary>,
    causal: bool,
}
fn raw_program(site: &Site, qwidth: usize, kwidth: usize, vwidth: usize) -> OperatorProgram {
    OperatorProgram {
        declarations: Declarations {
            domains: vec![],
            slots: vec![
                Slot::Raw { width: qwidth },
                Slot::Raw { width: kwidth },
                Slot::Raw { width: vwidth },
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
                scale: site.scale,
                rotary: site.rotary,
                causal: site.causal,
            },
        ],
        output: 3,
    }
}
fn raw_attention(
    site: &Site,
    q: &Array2<f64>,
    k: &Array2<f64>,
    v: &Array2<f64>,
    layout: &SequenceLayout,
) -> Result<Array2<f64>, String> {
    let program = raw_program(site, q.ncols(), k.ncols(), v.ncols());
    let inputs = FamilyInputs {
        rows: q.nrows(),
        slots: vec![
            SlotValues::Raw(q.clone()),
            SlotValues::Raw(k.clone()),
            SlotValues::Raw(v.clone()),
        ],
        layout: Some(layout.clone()),
    };
    Ok(program.execute(&inputs, false).map_err(err)?.values[3].clone())
}
fn weights(
    site: &Site,
    q: &Array2<f64>,
    k: &Array2<f64>,
    layout: &SequenceLayout,
) -> Result<Vec<f64>, String> {
    // Exact native Attend with a basis payload; no attention or rotary math is duplicated.
    let basis = Array2::<f64>::eye(q.nrows());
    let measured = raw_attention(site, q, k, &basis, layout)?;
    let p = measured.row(q.nrows() - 1).to_vec();
    if p.iter().any(|v| !v.is_finite() || *v < 0.) || (p.iter().sum::<f64>() - 1.).abs() > 1e-10 {
        return Err("invalid native attention measurement".into());
    }
    Ok(p)
}
fn difference(a: &[f64], b: &[f64], absolute: f64, relative: f64) -> Result<Value, String> {
    if a.len() != b.len() || a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err("inconsistent/nonfinite comparison".into());
    }
    let mut maximum = 0f64;
    let mut reference = 0f64;
    for (a, b) in a.iter().zip(b) {
        maximum = maximum.max((a - b).abs());
        reference = reference.max(a.abs()).max(b.abs());
        if (a - b).abs() > absolute + relative * a.abs().max(b.abs()) {
            return Err(format!(
                "native check error {} exceeds coordinate tolerance",
                (a - b).abs()
            ));
        }
    }
    Ok(
        json!({"maximum_absolute_error":maximum,"reference_maximum_absolute":reference,"absolute_tolerance":absolute,"relative_tolerance":relative}),
    )
}
fn indices(fixture: &Value, role: &str, rows: usize) -> Result<Vec<usize>, String> {
    let matching = fixture["spans"]
        .as_array()
        .ok_or("spans missing")?
        .iter()
        .filter(|s| s["role"].as_str() == Some(role))
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        return Err(format!("span {role} absent/duplicated"));
    }
    let ids: Vec<usize> =
        serde_json::from_value(matching[0]["token_indices"].clone()).map_err(err)?;
    if ids.is_empty()
        || ids.iter().any(|i| *i >= rows)
        || ids.iter().copied().collect::<BTreeSet<_>>().len() != ids.len()
    {
        return Err("invalid span token indices".into());
    }
    Ok(ids)
}
fn span_mean(values: &Array2<f64>, indices: &[usize]) -> Vec<f64> {
    (0..values.ncols())
        .map(|d| indices.iter().map(|i| values[(*i, d)]).sum::<f64>() / indices.len() as f64)
        .collect()
}
fn color_direction(values: &Array2<f64>, spans: &[Vec<usize>; 4]) -> Vec<f64> {
    let means = spans
        .iter()
        .map(|indices| span_mean(values, indices))
        .collect::<Vec<_>>();
    (0..values.ncols())
        .map(|d| (means[0][d] + means[3][d] - means[1][d] - means[2][d]) * 0.5)
        .collect()
}
fn patch_payload(
    values: &Array2<f64>,
    positions: &[usize],
    direction: &[f64],
    amplitude: f64,
) -> Array2<f64> {
    let mut changed = values.clone();
    for position in positions {
        for (v, d) in changed.row_mut(*position).iter_mut().zip(direction) {
            *v += amplitude * d;
        }
    }
    changed
}
fn transported(native: &[f64], direction: &[f64], mass: f64, amplitude: f64) -> Vec<f64> {
    native
        .iter()
        .zip(direction)
        .map(|(y, d)| y + amplitude * mass * d)
        .collect()
}
fn replace_read_last(value: &mut Array2<f64>, replacement: &[f64]) -> Result<(), String> {
    if value.nrows() == 0
        || value.ncols() != replacement.len()
        || replacement.iter().any(|v| !v.is_finite())
    {
        return Err("invalid read invocation replacement".into());
    }
    let last = value.nrows() - 1;
    for (v, x) in value.row_mut(last).iter_mut().zip(replacement) {
        *v = *x;
    }
    Ok(())
}
fn logp(logits: &[f64]) -> Result<Vec<f64>, String> {
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return Err("invalid full vocabulary logits".into());
    }
    let peak = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z = peak + logits.iter().map(|v| (v - peak).exp()).sum::<f64>().ln();
    Ok(logits.iter().map(|v| v - z).collect())
}
fn response(native: &[f64], candidate: &[f64], colors: [usize; 2]) -> Result<Value, String> {
    if native.len() != candidate.len()
        || colors.iter().any(|i| *i >= native.len())
        || colors[0] == colors[1]
    {
        return Err("invalid physical-color vocabulary IDs".into());
    }
    let p = logp(native)?;
    let q = logp(candidate)?;
    let odds = native[colors[0]] - native[colors[1]];
    let changed = candidate[colors[0]] - candidate[colors[1]];
    Ok(
        json!({"teacher_full_output_kl":p.iter().zip(&q).map(|(p,q)|p.exp()*(p-q)).sum::<f64>().max(0.),
        "native_color0_minus_color1_log_odds":odds,"changed_color0_minus_color1_log_odds":changed,
        "color0_minus_color1_log_odds_response":changed-odds,
        "color0_log_probability_response":q[colors[0]]-p[colors[0]],"color1_log_probability_response":q[colors[1]]-p[colors[1]],
        "maximum_logit_difference":native.iter().zip(candidate).map(|(a,b)|(a-b).abs()).fold(0f64,f64::max)}),
    )
}
struct Cached {
    id: String,
    factors: Value,
    prefix_tokens: Vec<u32>,
    keys: Array2<f64>,
    values: Array2<f64>,
    direction: Vec<f64>,
    colors: [usize; 2],
}
fn invariance(cached: &[Cached], factor: &str, settings: &Settings) -> Result<Vec<Value>, String> {
    let mut groups: BTreeMap<String, [Option<usize>; 2]> = BTreeMap::new();
    for (i, c) in cached.iter().enumerate() {
        let mut factors = c.factors.as_object().ok_or("factors missing")?.clone();
        let bit = factors
            .remove(factor)
            .ok_or("pair factor missing")?
            .as_u64()
            .ok_or("pair factor not integer")? as usize;
        if bit > 1 {
            return Err("pair factor not binary".into());
        }
        let key = serde_json::to_string(&factors).map_err(err)?;
        if groups.entry(key).or_insert([None, None])[bit]
            .replace(i)
            .is_some()
        {
            return Err("duplicate pair state".into());
        }
    }
    if groups.len() != 32 {
        return Err("32 balanced pair groups required".into());
    }
    let mut checks = vec![];
    for (key, ids) in groups {
        let a = &cached[ids[0].ok_or("pair0 missing")?];
        let b = &cached[ids[1].ok_or("pair1 missing")?];
        if a.prefix_tokens != b.prefix_tokens
            || a.colors != b.colors
            || a.keys.dim() != b.keys.dim()
            || a.values.dim() != b.values.dim()
        {
            return Err("physical prefix/color identity changed across query pair".into());
        }
        checks.push(json!({"pair_factor":factor,"factor0_fixture":a.id,"factor1_fixture":b.id,"other_factors":serde_json::from_str::<Value>(&key).map_err(err)?,
            "unchanged_physical_prefix_token_count":a.prefix_tokens.len(),"color_token_ids":a.colors,
            "prefix_key_check":difference(a.keys.as_slice().ok_or("contiguous keys")?,b.keys.as_slice().ok_or("contiguous keys")?,settings.absolute_tolerance,settings.relative_tolerance)?,
            "prefix_value_check":difference(a.values.as_slice().ok_or("contiguous values")?,b.values.as_slice().ok_or("contiguous values")?,settings.absolute_tolerance,settings.relative_tolerance)?,
            "direction_check":difference(&a.direction,&b.direction,settings.absolute_tolerance,settings.relative_tolerance)?}));
    }
    Ok(checks)
}
fn run(
    export: &Path,
    discovery: &Path,
    tokenizer: &Path,
    settings_path: &Path,
    out: &Path,
) -> Result<(), String> {
    let started = Instant::now();
    let settings: Settings = serde_json::from_value(load(settings_path)?).map_err(err)?;
    if out.exists()
        || settings.read_node != 2073
        || settings.epsilon_fraction.to_bits() != 0.1f64.to_bits()
        || settings.numeric_bytes == 0
        || [settings.absolute_tolerance, settings.relative_tolerance]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.)
    {
        return Err("fresh output and scoped read2073, epsilon_fraction0.1, finite tolerances/budget required".into());
    }
    // The input path is rejected before reading if it names a heldout deck.
    if discovery.file_name().and_then(|n| n.to_str()) != Some("DISCOVERY.json") {
        return Err("only DISCOVERY.json is supported; heldout prohibited".into());
    }
    for (path, expected) in [
        (discovery, settings.fixtures_sha256.as_str()),
        (tokenizer, settings.tokenizer_sha256.as_str()),
        (
            export.join("export.json").as_path(),
            settings.export_sha256.as_str(),
        ),
    ] {
        if expected.len() != 64 || sha256(path)? != expected {
            return Err(format!("input hash mismatch {}", path.display()));
        }
    }
    let deck = load(discovery)?;
    if deck["version"].as_u64() != Some(1)
        || deck["panel"].as_str() != Some("discovery")
        || deck["tokenizer_sha256"].as_str() != Some(settings.tokenizer_sha256.as_str())
    {
        return Err("hash-verified discovery deck required".into());
    }
    let cases = deck["cases"].as_array().ok_or("cases missing")?;
    if cases.len() != 64 {
        return Err("all64 discovery cases required".into());
    }
    let mut fixture_ids = BTreeSet::new();
    for case in cases {
        if case["panel"].as_str() != Some("discovery")
            || !fixture_ids.insert(case["id"].as_str().ok_or("fixture id missing")?)
        {
            return Err("duplicate/non-discovery fixture".into());
        }
    }
    let imported = import_language_model(export, 1, 1)?;
    let lineage = imported.record["source"]["checkpoint_sha256"]
        .as_str()
        .or_else(|| imported.record["source"]["weights_sha256"].as_str())
        .ok_or("missing native weight lineage")?;
    if lineage != settings.checkpoint_sha256 {
        return Err("native weight lineage mismatch".into());
    }
    for (name, metadata) in imported.record["files"]
        .as_object()
        .ok_or("native files missing")?
    {
        if metadata["sha256"].as_str()
            != Some(sha256(&export.join(format!("{name}.f64")))?.as_str())
        {
            return Err(format!("native tensor hash mismatch {name}"));
        }
    }
    let site = match imported.program.nodes.get(settings.read_node) {
        Some(Node::Attend {
            query,
            key,
            value,
            scale,
            rotary,
            causal,
        }) => Site {
            read: settings.read_node,
            query: *query,
            key: *key,
            value: *value,
            scale: *scale,
            rotary: *rotary,
            causal: *causal,
        },
        _ => return Err("selected read is not native Attend".into()),
    };
    if !site.causal {
        return Err("causal native head required".into());
    }
    let interfaces = imported.program.interfaces().map_err(err)?;
    let operator_bytes = imported
        .program
        .operators
        .iter()
        .try_fold(0usize, |total, o| {
            total.checked_add(o.rows.width().checked_mul(o.cols.width())?.checked_mul(8)?)
        })
        .ok_or("weight plan overflow")?;
    std::fs::create_dir_all(out).map_err(err)?;
    save(
        &out.join("SETTINGS.json"),
        &serde_json::to_value(&settings).map_err(err)?,
    )?;
    let mut journal = std::fs::File::create(out.join("JOURNAL.jsonl")).map_err(err)?;
    let mut cached = vec![];
    let mut records = vec![];
    for (case_index, case) in cases.iter().enumerate() {
        let case_started = Instant::now();
        let tokens: Vec<u32> = serde_json::from_value(case["tokens"].clone()).map_err(err)?;
        let rows = tokens.len();
        if rows < 2 {
            return Err("nontrivial token sequence required".into());
        }
        let plan = interfaces
            .iter()
            .try_fold(operator_bytes, |total, i| {
                total.checked_add(
                    rows.checked_mul(i.width())?
                        .checked_mul(8)?
                        .checked_mul(4)?,
                )
            })
            .and_then(|v| v.checked_add(rows.checked_mul(rows)?.checked_mul(96)?))
            .ok_or("numeric plan overflow")?;
        if plan > settings.numeric_bytes {
            return Err(format!("numeric plan {plan} exceeds declared budget"));
        }
        let layout = SequenceLayout {
            sequence: vec![0; rows],
            position: (0..rows as u32).collect(),
        };
        let inputs = FamilyInputs {
            rows,
            slots: vec![SlotValues::Tokens(tokens.clone())],
            layout: Some(layout.clone()),
        };
        let native = imported.program.execute(&inputs, false).map_err(err)?;
        let q = &native.values[site.query];
        let k = &native.values[site.key];
        let v = &native.values[site.value];
        let alpha = weights(&site, q, k, &layout)?;
        let baseline = raw_attention(&site, q, k, v, &layout)?;
        let native_head = native.values[site.read].row(rows - 1).to_vec();
        let baseline_check = difference(
            &baseline.row(rows - 1).to_vec(),
            &native_head,
            settings.absolute_tolerance,
            settings.relative_tolerance,
        )?;
        let spans = [
            indices(case, "old_value_0", rows)?,
            indices(case, "old_value_1", rows)?,
            indices(case, "new_value_0", rows)?,
            indices(case, "new_value_1", rows)?,
        ];
        let query_start = *indices(case, "query_cue", rows)?
            .iter()
            .min()
            .ok_or("query span empty")?;
        let record_positions = spans.iter().flatten().copied().collect::<BTreeSet<_>>();
        if record_positions.len() != spans.iter().map(Vec::len).sum::<usize>()
            || record_positions.iter().any(|i| *i >= query_start)
        {
            return Err("value spans must be disjoint and precede query".into());
        }
        if spans[0].len() != 1 || spans[1].len() != 1 {
            return Err(
                "fixed physical-color readout requires single-token old_value0/1 spans".into(),
            );
        }
        let colors = [tokens[spans[0][0]] as usize, tokens[spans[1][0]] as usize];
        if colors[0] == colors[1]
            || spans[3].iter().any(|i| tokens[*i] as usize != colors[0])
            || spans[2].iter().any(|i| tokens[*i] as usize != colors[1])
        {
            return Err("expected old0/new1 and old1/new0 swapped color identities".into());
        }
        let direction = color_direction(v, &spans);
        if direction.iter().any(|d| !d.is_finite())
            || direction.iter().map(|d| d * d).sum::<f64>() == 0.
        {
            return Err("nonfinite/zero native physical-color direction".into());
        }
        let owner_positions = [
            spans[0]
                .iter()
                .chain(&spans[2])
                .copied()
                .collect::<Vec<_>>(),
            spans[1]
                .iter()
                .chain(&spans[3])
                .copied()
                .collect::<Vec<_>>(),
        ];
        let masses = owner_positions
            .each_ref()
            .map(|indices| indices.iter().map(|i| alpha[*i]).sum::<f64>());
        let teacher = native.values[imported.program.output]
            .row(rows - 1)
            .to_vec();
        let mut controls = vec![];
        let mut owner_derivatives = vec![];
        for owner in 0..2 {
            let mut odds = [0.; 2];
            for (sign_index, sign) in [-1., 1.].iter().enumerate() {
                let amplitude = sign * settings.epsilon_fraction;
                let replacement = transported(&native_head, &direction, masses[owner], amplitude);
                let changed_v = patch_payload(v, &owner_positions[owner], &direction, amplitude);
                let direct = raw_attention(&site, q, k, &changed_v, &layout)?;
                let transport_check = difference(
                    &direct.row(rows - 1).to_vec(),
                    &replacement,
                    settings.absolute_tolerance,
                    settings.relative_tolerance,
                )?;
                let edited = imported
                    .program
                    .execute_edited_from(&inputs, &native, site.read, |node, value, _| {
                        if node == site.read {
                            replace_read_last(value, &replacement)?;
                        }
                        Ok(())
                    })
                    .map_err(err)?;
                let candidate = edited.values[imported.program.output]
                    .row(rows - 1)
                    .to_vec();
                let metrics = response(&teacher, &candidate, colors)?;
                odds[sign_index] = candidate[colors[0]] - candidate[colors[1]];
                let mut maximum_other_query_change = 0f64;
                for (node, after) in edited.values.iter().enumerate().skip(site.read) {
                    // Gathered Feature values may have zero columns; slices remain well-defined.
                    for (before, after) in native.values[node]
                        .slice(s![..rows - 1, ..])
                        .iter()
                        .zip(after.slice(s![..rows - 1, ..]).iter())
                    {
                        maximum_other_query_change =
                            maximum_other_query_change.max((before - after).abs());
                        if (before - after).abs()
                            > settings.absolute_tolerance
                                + settings.relative_tolerance * before.abs().max(after.abs())
                        {
                            return Err("earlier query positions changed under last-query-only intervention".into());
                        }
                    }
                }
                controls.push(json!({"owner":owner,"sign":sign,"amplitude":amplitude,"value_input_positions":owner_positions[owner],"attention_mass_of_owner_records":masses[owner],
                    "direct_raw_attend_transport_check":transport_check,"maximum_other_query_position_change":maximum_other_query_change,"metrics":metrics}));
            }
            let derivative = (odds[1] - odds[0]) / (2. * settings.epsilon_fraction);
            owner_derivatives.push(json!({"owner":owner,"fixed_color_log_odds_central_response_per_epsilon_fraction":derivative,
                "fixed_color_log_odds_response_per_unit_head_direction":if masses[owner]>0. {Some(derivative/masses[owner])} else {None},
                "definition":"[odds(+epsilon*d)-odds(-epsilon*d)]/(2*epsilon); second field divides by native owner-record attention mass"}));
        }
        let record = json!({"fixture_id":case["id"],"factors":case["factors"],"color0_token_id":colors[0],"color1_token_id":colors[1],
            "readout_color_orientation":"color0 is actual old_value0 token, color1 actual old_value1 token; invariant under query_owner swaps, never query-relative target relabeling",
            "native_fixed_color0_minus_color1_log_odds":teacher[colors[0]]-teacher[colors[1]],"native_attention":alpha,
            "owner_record_attention_masses":masses,"native_direction":direction,"native_direction_l2_norm":direction.iter().map(|d|d*d).sum::<f64>().sqrt(),
            "native_head_reconstruction_check":baseline_check,"controls":controls,"owner_directional_responses":owner_derivatives,
            "numeric_plan_bytes":plan,"seconds":case_started.elapsed().as_secs_f64()});
        writeln!(journal, "{}", serde_json::to_string(&record).map_err(err)?).map_err(err)?;
        journal.flush().map_err(err)?;
        records.push(record);
        cached.push(Cached {
            id: case["id"].as_str().ok_or("id missing")?.into(),
            factors: case["factors"].clone(),
            prefix_tokens: tokens[..query_start].to_vec(),
            keys: k.slice(s![..query_start, ..]).to_owned(),
            values: v.slice(s![..query_start, ..]).to_owned(),
            direction,
            colors,
        });
        eprintln!(
            "case {}/64 done elapsed={:.1}s",
            case_index + 1,
            started.elapsed().as_secs_f64()
        );
    }
    let owner_swap = invariance(&cached, "query_owner", &settings)?;
    let now_swap = invariance(&cached, "query_now", &settings)?;
    save(
        &out.join("PROVENANCE.json"),
        &json!({"native_export":imported.record,"fixtures_sha256":settings.fixtures_sha256,"tokenizer_sha256":settings.tokenizer_sha256,
        "export_sha256":settings.export_sha256,"checkpoint_sha256":settings.checkpoint_sha256,"settings_sha256":sha256(settings_path)?,
        "executable_sha256":sha256(&std::env::current_exe().map_err(err)?)?,"heldout_opened_or_evaluated":false,
        "native_site":{"read_node":site.read,"query_node":site.query,"key_node":site.key,"value_node":site.value,"scale":format!("{:?}",site.scale),"rotary":format!("{:?}",site.rotary)},
        "numeric_plan":"native operators + four conservative full traces + attention scratch; excludes transient import/hash allocations and host metadata"}),
    )?;
    save(
        &out.join("REPORT.json"),
        &json!({"schema":"native-owner-payload-invocation-probe-v1","settings":settings,"cases":records,"case_count":64,"control_count_per_case":4,
        "query_owner_prefix_direction_invariance":owner_swap,"query_now_prefix_direction_invariance":now_swap,"provenance_sha256":sha256(&out.join("PROVENANCE.json"))?,
        "seconds":started.elapsed().as_secs_f64(),"direction":"d=(mean V(old_value0)+mean V(new_value1)-mean V(old_value1)-mean V(new_value0))/2",
        "intervention":"plus/minus 0.1*d at all value-token positions in either owner's old AND new records, scoped to selected head's last query invocation",
        "execution":"selected native Q/K/attention fixed; exact head output y+sign*epsilon*owner_record_attention_mass*d; direct raw native Attend with changed payload verifies each control, then selected read last row replaced and native suffix recomputed",
        "scope":"value-input invocation intervention only; shared native V node and parameters not edited; sibling head invocations receive original native producers; downstream heads respond normally to the patched residual",
        "readout":"fixed physical old_value0-vs-old_value1 token log odds; no query-relative target label",
        "limitations":["Native owner-specific payload transport measurement, not learned control construction or algorithm recovery",
            "Weighted-sum transport identity is supplied architectural math; verification alone is not discovery",
            "Human-specified record spans and selected head are explicit scope; no whole-model replacement or generalization acceptance",
            "Full output KL is conditional on native background, downstream, and this local intervention; discovery only, heldout untouched"]}),
    )?;
    Ok(())
}
fn main() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 5 {
        return Err("EXPORT DISCOVERY.json TOKENIZER SETTINGS.json OUT_DIR".into());
    }
    run(
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
        Path::new(&args[4]),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_linear_payload_transport_and_scoped_suffix() {
        let site = Site {
            read: 3,
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
        };
        let q = ndarray::array![[0.2, 0.1], [-0.1, 0.3], [0.6, -0.2], [0.4, 0.5]];
        let k = ndarray::array![[0.1, 0.3], [0.4, -0.5], [0.2, -0.1], [0.7, 0.4]];
        let v = ndarray::array![[0.3, -0.7], [0.8, 0.1], [-0.3, 0.6], [0.4, 0.2]];
        let layout = SequenceLayout {
            sequence: vec![0; 4],
            position: vec![0, 1, 2, 3],
        };
        let mut program = raw_program(&site, 2, 2, 2);
        program.nodes.push(Node::RmsNorm {
            input: 3,
            epsilon: 1e-6,
        });
        program.output = 4;
        let inputs = FamilyInputs {
            rows: 4,
            slots: vec![
                SlotValues::Raw(q.clone()),
                SlotValues::Raw(k.clone()),
                SlotValues::Raw(v.clone()),
            ],
            layout: Some(layout.clone()),
        };
        let native = program.execute(&inputs, false).unwrap();
        let alpha = weights(&site, &q, &k, &layout).unwrap();
        let direction = vec![0.6, -0.2];
        let positions = vec![0, 2];
        let mass = alpha[0] + alpha[2];
        for amplitude in [-0.1, 0.1] {
            let direct = raw_attention(
                &site,
                &q,
                &k,
                &patch_payload(&v, &positions, &direction, amplitude),
                &layout,
            )
            .unwrap();
            let replacement = transported(
                &native.values[3].row(3).to_vec(),
                &direction,
                mass,
                amplitude,
            );
            difference(&replacement, &direct.row(3).to_vec(), 1e-14, 1e-14).unwrap();
            let suffix = program
                .execute_edited_from(&inputs, &native, 3, |node, value, _| {
                    if node == 3 {
                        replace_read_last(value, &replacement)?;
                    }
                    Ok(())
                })
                .unwrap();
            let full = program
                .execute_edited(&inputs, |node, value, _| {
                    if node == 3 {
                        replace_read_last(value, &replacement)?;
                    }
                    Ok(())
                })
                .unwrap();
            assert_eq!(suffix.values, full.values);
            for node in 0..3 {
                assert_eq!(suffix.values[node], native.values[node]);
            }
            for node in 3..5 {
                assert_eq!(
                    suffix.values[node].slice(s![..3, ..]),
                    native.values[node].slice(s![..3, ..])
                );
            }
            assert_ne!(suffix.values[4].row(3), native.values[4].row(3));
        }
    }
    #[test]
    fn physical_direction_and_readout_do_not_relabel_by_query_owner() {
        let v = ndarray::array![[2., 0.], [-2., 0.], [-2., 0.], [2., 0.]];
        let spans = [vec![0], vec![1], vec![2], vec![3]];
        assert_eq!(color_direction(&v, &spans), vec![4., 0.]);
        let result = response(&[0.4, -0.2, 0.1], &[0.5, -0.3, 0.1], [0, 1]).unwrap();
        assert!(
            (result["color0_minus_color1_log_odds_response"]
                .as_f64()
                .unwrap()
                - 0.2)
                .abs()
                < 1e-14
        );
    }
}

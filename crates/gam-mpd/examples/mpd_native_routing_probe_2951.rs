//! Native teacher instrumentation only; no routing predictor or explanation acceptance.
//! EXPORT FIXTURES TOKENIZER OUT_DIR [SETTINGS.json]. Host execution, all declared heads.
use gam_mpd::{
    engine::sha256,
    import::import_language_model,
    operator_program::{
        Declarations, FamilyInputs, Node, OperatorProgram, Rotary, Scale, SequenceLayout, Slot,
        SlotValues,
    },
};
use ndarray::{Array2, ArrayView2, s};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};
fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn save(path: &Path, value: &Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_vec_pretty(value).map_err(error)?).map_err(error)
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    fixture_ids: Vec<Value>,
    #[serde(default)]
    checkpoint_sha256: Option<String>,
    numeric_bytes: usize,
    absolute_tolerance: f64,
    relative_tolerance: f64,
}
#[derive(Clone)]
struct Site {
    read: usize,
    query: usize,
    key: usize,
    value: usize,
    scale: Scale,
    rotary: Option<Rotary>,
    causal: bool,
}
fn sites(program: &OperatorProgram) -> Vec<Site> {
    program
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(read, n)| match n {
            Node::Attend {
                query,
                key,
                value,
                scale,
                rotary,
                causal,
            } => Some(Site {
                read,
                query: *query,
                key: *key,
                value: *value,
                scale: *scale,
                rotary: *rotary,
                causal: *causal,
            }),
            _ => None,
        })
        .collect()
}
fn raw_attention(
    site: &Site,
    q: &Array2<f64>,
    k: &Array2<f64>,
    v: &Array2<f64>,
    layout: &SequenceLayout,
) -> Result<Array2<f64>, String> {
    let p = OperatorProgram {
        declarations: Declarations {
            domains: vec![],
            slots: vec![
                Slot::Raw { width: q.ncols() },
                Slot::Raw { width: k.ncols() },
                Slot::Raw { width: v.ncols() },
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
    };
    let input = FamilyInputs {
        rows: q.nrows(),
        slots: vec![
            SlotValues::Raw(q.clone()),
            SlotValues::Raw(k.clone()),
            SlotValues::Raw(v.clone()),
        ],
        layout: Some(layout.clone()),
    };
    Ok(p.execute(&input, false).map_err(error)?.values[3].clone())
}
fn weighted(values: &Array2<f64>, alpha: &[f64]) -> Result<Array2<f64>, String> {
    if values.nrows() != alpha.len() {
        return Err("payload/weight rows mismatch".into());
    }
    let mut result = Array2::zeros((1, values.ncols()));
    for (j, &weight) in alpha.iter().enumerate() {
        result.row_mut(0).scaled_add(weight, &values.row(j));
    }
    Ok(result)
}
fn difference(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
    absolute: f64,
    relative: f64,
) -> Result<Value, String> {
    if a.dim() != b.dim() || a.iter().chain(b.iter()).any(|v| !v.is_finite()) {
        return Err("nonfinite or inconsistent reconstruction".into());
    }
    let maximum = a
        .iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f64, f64::max);
    let reference = a
        .iter()
        .chain(b.iter())
        .map(|v| v.abs())
        .fold(0f64, f64::max);
    let limit = absolute + relative * reference;
    if maximum > limit {
        return Err(format!(
            "native attention measurement mismatch {maximum} exceeds {limit}"
        ));
    }
    Ok(
        json!({"maximum_absolute_error":maximum,"reference_maximum_absolute":reference,"verification_limit":limit}),
    )
}
struct Measurement {
    alpha: Array2<f64>,
    donor_actual: Array2<f64>,
    donor_predicted: Array2<f64>,
    checks: Value,
}
fn measure(
    site: &Site,
    q: &Array2<f64>,
    k: &Array2<f64>,
    v: &Array2<f64>,
    native: &Array2<f64>,
    layout: &SequenceLayout,
    absolute: f64,
    relative: f64,
) -> Result<Measurement, String> {
    let n = q.nrows();
    if n == 0 || k.nrows() != n || v.nrows() != n || native.nrows() != n || n > v.ncols() {
        return Err("basis measurement requires positive rows<=native payload width".into());
    }
    let mut basis = Array2::zeros(v.dim());
    for i in 0..n {
        basis[(i, i)] = 1.;
    }
    let weights = raw_attention(site, q, k, &basis, layout)?;
    let alpha = weights.slice(s![n - 1..n, 0..n]).to_owned();
    if alpha.iter().any(|v| !v.is_finite() || *v < 0.)
        || (alpha.sum() - 1.).abs() > absolute + relative
    {
        return Err("invalid measured attention row".into());
    }
    let reconstructed = weighted(v, alpha.as_slice().ok_or("contiguous alpha")?)?;
    let native_last = native.slice(s![n - 1..n, ..]);
    let reconstruction = difference(reconstructed.view(), native_last, absolute, relative)?;
    // Deterministic local value invocation patch; no full-tail native intervention.
    let mut donor = v.clone();
    donor.row_mut(0).assign(&v.row(n - 1));
    let changed = raw_attention(site, q, k, &donor, layout)?;
    let donor_actual = &changed.slice(s![n - 1..n, ..]) - &native_last;
    let donor_predicted = Array2::from_shape_fn((1, v.ncols()), |(_, c)| {
        alpha[(0, 0)] * (v[(n - 1, c)] - v[(0, c)])
    });
    let transport = difference(
        donor_actual.view(),
        donor_predicted.view(),
        absolute,
        relative,
    )?;
    Ok(Measurement {
        alpha,
        donor_actual,
        donor_predicted,
        checks: json!({"weighted_value_reconstruction":reconstruction,"donor_delta_transport":transport,"donor_patch":{"kind":"value_node_invocation","receiver_row":0,"donor_row":n-1,"query_key_held_fixed":true,"scope":"local head only; not global parameter edit or whole-tail response"}}),
    })
}
struct Pack {
    writer: BufWriter<File>,
    offset: u64,
    entries: Vec<Value>,
}
impl Pack {
    fn new(path: &Path) -> Result<Self, String> {
        Ok(Self {
            writer: BufWriter::new(File::create(path).map_err(error)?),
            offset: 0,
            entries: vec![],
        })
    }
    fn append(
        &mut self,
        id: &str,
        values: ArrayView2<'_, f64>,
        metadata: Value,
    ) -> Result<(), String> {
        if values.iter().any(|v| !v.is_finite()) {
            return Err(format!("nonfinite tensor {id}"));
        }
        let bytes = u64::try_from(values.len())
            .map_err(error)?
            .checked_mul(8)
            .ok_or("tensor byte length overflow")?;
        self.entries.push(json!({"id":id,"byte_offset":self.offset,"byte_length":bytes,"shape":[values.nrows(),values.ncols()],"dtype":"little-endian f64","layout":"row-major","metadata":metadata}));
        for row in values.rows() {
            for &v in row {
                self.writer.write_all(&v.to_le_bytes()).map_err(error)?;
            }
        }
        self.offset = self
            .offset
            .checked_add(bytes)
            .ok_or("packed bytes overflow")?;
        Ok(())
    }
    fn finish(mut self) -> Result<Vec<Value>, String> {
        self.writer.flush().map_err(error)?;
        Ok(self.entries)
    }
}
fn self_test() -> Result<(), String> {
    let site = Site {
        read: 3,
        query: 0,
        key: 1,
        value: 2,
        scale: Scale::InverseSqrt(4),
        rotary: Some(Rotary {
            base: 10000,
            dims: 4,
            half_split: true,
        }),
        causal: true,
    };
    let q = ndarray::array![
        [0.2, 0.4, -0.3, 0.1],
        [0.7, -0.2, 0.5, 0.3],
        [-0.5, 0.6, 0.1, 0.2]
    ];
    let k = &q * 0.7;
    let v = ndarray::array![[1., 2., 3., 4.], [-2., 1., 0., 3.], [4., -1., 2., 0.]];
    let layout = SequenceLayout {
        sequence: vec![0, 1, 0],
        position: vec![3, 1, 7],
    };
    let native = raw_attention(&site, &q, &k, &v, &layout)?;
    let measured = measure(&site, &q, &k, &v, &native, &layout, 1e-11, 1e-10)?;
    if measured.alpha[(0, 1)] != 0. {
        return Err("cross-sequence key leaked".into());
    }
    Ok(())
}
fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 4 && args.len() != 5 {
        return Err("EXPORT FIXTURES TOKENIZER OUT_DIR [SETTINGS.json]".into());
    }
    self_test()?;
    let export = Path::new(&args[0]);
    let fixtures_path = Path::new(&args[1]);
    let out = Path::new(&args[3]);
    if out.exists() {
        return Err("fresh output directory required".into());
    }
    let fixtures: Value =
        serde_json::from_slice(&std::fs::read(fixtures_path).map_err(error)?).map_err(error)?;
    if fixtures["version"].as_u64() != Some(1) || fixtures["panel"].as_str() == Some("heldout") {
        return Err("version1 discovery/development fixtures required; heldout forbidden before frozen explanation".into());
    }
    let tokenizer_hash = sha256(Path::new(&args[2]))?;
    if fixtures["tokenizer_sha256"].as_str() != Some(tokenizer_hash.as_str()) {
        return Err("tokenizer SHA mismatch".into());
    }
    let cases = fixtures["cases"]
        .as_array()
        .ok_or("fixture cases missing")?;
    let default_ids = cases
        .iter()
        .filter(|c| {
            matches!(c["group"].as_str(), Some("assignment" | "reassignment"))
                || fixtures["panel"].as_str() == Some("discovery")
        })
        .map(|c| c["id"].clone())
        .collect();
    let settings: Settings = match args.get(4) {
        Some(path) => {
            serde_json::from_slice(&std::fs::read(path).map_err(error)?).map_err(error)?
        }
        None => Settings {
            fixture_ids: default_ids,
            checkpoint_sha256: None,
            numeric_bytes: 16usize << 30,
            absolute_tolerance: 1e-11,
            relative_tolerance: 1e-10,
        },
    };
    if settings.fixture_ids.is_empty()
        || settings.numeric_bytes == 0
        || !settings.absolute_tolerance.is_finite()
        || settings.absolute_tolerance < 0.
        || !settings.relative_tolerance.is_finite()
        || settings.relative_tolerance < 0.
    {
        return Err(
            "nonempty declared IDs and finite nonnegative verification tolerances/budget required"
                .into(),
        );
    }
    let mut unique = BTreeSet::new();
    for id in &settings.fixture_ids {
        if !(id.is_string() || id.is_u64())
            || !unique.insert(serde_json::to_string(id).map_err(error)?)
        {
            return Err("unique numeric/string fixture IDs required".into());
        }
    }
    let selected = settings
        .fixture_ids
        .iter()
        .map(|id| {
            let found = cases.iter().filter(|c| &c["id"] == id).collect::<Vec<_>>();
            if found.len() != 1 {
                Err(format!("declared fixture {id} absent or duplicated"))
            } else {
                Ok(found[0])
            }
        })
        .collect::<Result<Vec<_>, String>>()?;
    std::fs::create_dir_all(out).map_err(error)?;
    save(
        &out.join("SETTINGS.json"),
        &serde_json::to_value(&settings).map_err(error)?,
    )?;
    save(
        &out.join("DECLARATION.json"),
        &json!({"fixture_ids":settings.fixture_ids,"selection":"all declared cases/all native Attend sites; no measured head/case selection","scope":"native teacher instrumentation only, not learned control construction or explanatory prediction","heldout":"not read; explicitly refused","backend":"Host f64","verification":"absolute+relative numeric comparison only, not acceptance certificate","fixtures_sha256":sha256(fixtures_path)?,"tokenizer_sha256":tokenizer_hash}),
    )?;
    let started = Instant::now();
    let imported = import_language_model(export, 1, 1)?;
    let lineage = imported.record["source"]["checkpoint_sha256"]
        .as_str()
        .or_else(|| imported.record["source"]["weights_sha256"].as_str())
        .ok_or("native checkpoint lineage missing")?;
    let expected = settings
        .checkpoint_sha256
        .as_deref()
        .or_else(|| fixtures["checkpoint_sha256"].as_str())
        .ok_or("checkpoint SHA required in fixtures or settings")?;
    if expected != lineage {
        return Err("native checkpoint SHA mismatch".into());
    }
    let files = imported.record["files"]
        .as_object()
        .ok_or("native files metadata missing")?;
    for (name, record) in files {
        if name == "tokens" {
            continue;
        }
        if record["sha256"].as_str() != Some(sha256(&export.join(format!("{name}.f64")))?.as_str())
        {
            return Err(format!("native weight hash mismatch {name}"));
        }
    }
    let sites = sites(&imported.program);
    if sites.is_empty() {
        return Err("no native Attend sites".into());
    }
    let graph = imported
        .program
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| json!({"node":i,"arguments":n.arguments(),"syntax":format!("{n:?}")}))
        .collect::<Vec<_>>();
    let consumers = |node: usize| {
        imported
            .program
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.arguments().contains(&node))
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
    };
    let site_manifest=sites.iter().map(|s|json!({"read_node":s.read,"query_node":s.query,"key_node":s.key,"value_node":s.value,"read_consumers":consumers(s.read),"query_consumers":consumers(s.query),"key_consumers":consumers(s.key),"value_consumers":consumers(s.value),"scale":format!("{:?}",s.scale),"rotary":format!("{:?}",s.rotary),"causal":s.causal})).collect::<Vec<_>>();
    save(
        &out.join("NATIVE_GRAPH.json"),
        &json!({"nodes":graph,"output":imported.program.output,"operators":imported.program.operators.iter().enumerate().map(|(i,o)|json!({"operator":i,"name":o.name,"rows":o.rows.width(),"columns":o.cols.width()})).collect::<Vec<_>>(),"sites":site_manifest,"consumer_scope":"all immediate consumers explicitly; complete node argument graph provides every transitive downstream path","normalization":"original query/key producer nodes already include any native QK normalization; original rotary/scale used only inside native Attend instrumentation"}),
    )?;
    save(
        &out.join("PROVENANCE.json"),
        &json!({"native_export":imported.record,"export_sha256":sha256(&export.join("export.json"))?,"fixtures_sha256":sha256(fixtures_path)?,"settings_sha256":args.get(4).map(|p|sha256(Path::new(p))).transpose()?,"tokenizer_sha256":tokenizer_hash,"packing":"one LE-f64 row-major file per case; original shared producers deduplicated by exact node identity; query-only producers last row, key/value producers all rows","native_program":"original imported graph and weights, no split_sites/f32 projection","scope":"actual native states and attention measurements, not discovered rule or whole-tail causal response"}),
    )?;
    let initialization_seconds = started.elapsed().as_secs_f64();
    let interfaces = imported.program.interfaces().map_err(error)?;
    let operator_bytes = imported
        .program
        .operators
        .iter()
        .try_fold(0usize, |a, o| {
            a.checked_add(o.rows.width().checked_mul(o.cols.width())?.checked_mul(8)?)
        })
        .ok_or("operator bytes overflow")?;
    let vocab = imported.record["config"]["vocab"]
        .as_u64()
        .ok_or("vocab missing")? as usize;
    let context = imported.record["config"]["n_ctx"]
        .as_u64()
        .ok_or("context missing")? as usize;
    let mut journal = BufWriter::new(File::create(out.join("journal.jsonl")).map_err(error)?);
    let mut reports = vec![];
    for (ordinal, c) in selected.iter().enumerate() {
        let case_started = Instant::now();
        let tokens: Vec<u32> = serde_json::from_value(c["tokens"].clone()).map_err(error)?;
        let prompt = c["prompt"].as_str().ok_or("fixture prompt missing")?;
        let offsets: Vec<(usize, usize)> =
            serde_json::from_value(c["offsets"].clone()).map_err(error)?;
        let rows = tokens.len();
        if rows == 0
            || rows > context
            || offsets.len() != rows
            || tokens.iter().any(|v| *v as usize >= vocab)
            || offsets.iter().any(|(a, b)| a > b || *b > prompt.len())
        {
            return Err(format!("invalid fixture {}", c["id"]));
        }
        let trace_bytes = interfaces
            .iter()
            .try_fold(0usize, |a, i| {
                a.checked_add(rows.checked_mul(i.width())?.checked_mul(8)?)
            })
            .ok_or("trace byte overflow")?;
        let planned = operator_bytes
            .checked_add(
                trace_bytes
                    .checked_mul(4)
                    .ok_or("trace workspace overflow")?,
            )
            .and_then(|v| v.checked_add(rows.checked_mul(rows)?.checked_mul(96)?))
            .ok_or("numeric plan overflow")?;
        if planned > settings.numeric_bytes {
            return Err(format!(
                "Host numeric plan {planned} exceeds {}",
                settings.numeric_bytes
            ));
        }
        let family = FamilyInputs {
            rows,
            slots: vec![SlotValues::Tokens(tokens)],
            layout: Some(SequenceLayout {
                sequence: vec![0; rows],
                position: (0..rows as u32).collect(),
            }),
        };
        let forward_started = Instant::now();
        let trace = imported.program.execute(&family, false).map_err(error)?;
        let forward_seconds = forward_started.elapsed().as_secs_f64();
        let case_dir = out.join(format!("case-{ordinal:03}"));
        std::fs::create_dir(&case_dir).map_err(error)?;
        save(&case_dir.join("FIXTURE.json"), c)?;
        let packed_path = case_dir.join("states.f64");
        let mut packed = Pack::new(&packed_path)?;
        let mut producer_rows: BTreeMap<usize, bool> = BTreeMap::new();
        for s in &sites {
            producer_rows.entry(s.query).or_insert(false);
            producer_rows.insert(s.key, true);
            producer_rows.insert(s.value, true);
        }
        for (&node, &all_rows) in &producer_rows {
            let value = if all_rows {
                trace.values[node].view()
            } else {
                trace.values[node].slice(s![rows - 1..rows, ..])
            };
            packed.append(&format!("producer-{node}"),value,json!({"native_node":node,"native_rows":if all_rows{(0..rows).collect::<Vec<_>>()}else{vec![rows-1]},"all_consumers":consumers(node)}))?;
        }
        let measurement_started = Instant::now();
        let mut heads = vec![];
        for s in &sites {
            let m = measure(
                s,
                &trace.values[s.query],
                &trace.values[s.key],
                &trace.values[s.value],
                &trace.values[s.read],
                family.layout.as_ref().ok_or("layout missing")?,
                settings.absolute_tolerance,
                settings.relative_tolerance,
            )?;
            packed.append(&format!("attention-{read}",read=s.read),m.alpha.view(),json!({"native_read_node":s.read,"native_query_row":rows-1,"native_key_rows":(0..rows).collect::<Vec<_>>()}))?;
            packed.append(
                &format!("head-output-{read}", read = s.read),
                trace.values[s.read].slice(s![rows - 1..rows, ..]),
                json!({"native_node":s.read,"native_row":rows-1}),
            )?;
            packed.append(
                &format!("donor-actual-{read}", read = s.read),
                m.donor_actual.view(),
                json!({"scope":"local value invocation delta only"}),
            )?;
            packed.append(&format!("donor-predicted-{read}",read=s.read),m.donor_predicted.view(),json!({"scope":"supplied architectural affine transport identity, not discovered law"}))?;
            heads.push(json!({"read_node":s.read,"checks":m.checks}));
        }
        let measurement_seconds = measurement_started.elapsed().as_secs_f64();
        let entries = packed.finish()?;
        let logits = trace.values[imported.program.output].row(rows - 1);
        let target = c["target"]
            .as_u64()
            .or_else(|| c["hypothesis_targets"]["updated_owner_value"]["id"].as_u64())
            .ok_or("explicit target ID missing")? as usize;
        let foil = c["foil"]
            .as_u64()
            .or_else(|| c["hypothesis_targets"]["old_owner_value"]["id"].as_u64())
            .ok_or("explicit foil ID missing")? as usize;
        if target >= logits.len() || foil >= logits.len() || logits.iter().any(|v| !v.is_finite()) {
            return Err("invalid native logits/target IDs".into());
        }
        let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let logz = maximum
            + logits
                .iter()
                .map(|v| (*v - maximum).exp())
                .sum::<f64>()
                .ln();
        let report = json!({"fixture_id":c["id"],"fixture":c,"rows":rows,"heads":heads,"shared_producer_count":producer_rows.len(),"packed_file":"states.f64","packed_sha256":sha256(&packed_path)?,"tensors":entries,"numeric_plan_bytes":planned,"native_observable":{"target_token":target,"foil_token":foil,"target_log_probability":logits[target]-logz,"foil_log_probability":logits[foil]-logz,"target_minus_foil_log_odds":logits[target]-logits[foil]},"seconds":{"native_forward":forward_seconds,"measure_and_pack":measurement_seconds,"total":case_started.elapsed().as_secs_f64()},"scope":"original native states; supplied attention instrumentation; local donor invocation check, no whole-tail interventions"});
        save(&case_dir.join("REPORT.json"), &report)?;
        serde_json::to_writer(&mut journal,&json!({"fixture_id":c["id"],"case_directory":format!("case-{ordinal:03}"),"status":"complete","heads":sites.len(),"seconds":case_started.elapsed().as_secs_f64()})).map_err(error)?;
        journal.write_all(b"\n").map_err(error)?;
        journal.flush().map_err(error)?;
        eprintln!(
            "fixture {} complete: {} heads in {:.3}s",
            c["id"],
            sites.len(),
            case_started.elapsed().as_secs_f64()
        );
        reports.push(json!({"fixture_id":c["id"],"case_directory":format!("case-{ordinal:03}"),"report_sha256":sha256(&case_dir.join("REPORT.json"))?,"seconds":case_started.elapsed().as_secs_f64()}));
    }
    save(
        &out.join("REPORT.json"),
        &json!({"cases":reports,"head_count_per_case":sites.len(),"initialization_seconds":initialization_seconds,"total_seconds":started.elapsed().as_secs_f64(),"all_declared_cases_and_heads":true,"scope":"native instrumentation only; no semantic role, explanatory law, Local/Run acceptance or compression claim","numeric_budget_exclusions":"host allocator/metadata, library scratch, transient import/weight hashing buffers; plan counts stored operators and conservative trace/workspaces"}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_rotary_causal_instrument_and_donor_delta_agree() {
        self_test().expect("exact native instrumentation within disclosed arithmetic tolerance");
    }
    #[test]
    fn distinct_shared_producers_are_preserved() {
        let p = OperatorProgram {
            declarations: Declarations {
                domains: vec![],
                slots: vec![Slot::Raw { width: 4 }; 3],
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
                    scale: Scale::InverseSqrt(4),
                    rotary: None,
                    causal: true,
                },
                Node::Attend {
                    query: 0,
                    key: 1,
                    value: 2,
                    scale: Scale::InverseSqrt(4),
                    rotary: None,
                    causal: true,
                },
            ],
            output: 4,
        };
        let found = sites(&p);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].key, found[1].key);
        assert_eq!(found[0].value, found[1].value);
        assert_ne!(found[0].read, found[1].read);
    }
}

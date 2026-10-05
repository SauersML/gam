//! Prespecified native behavioral screen, not a mechanism or fidelity claim.
//! EXPORT FIXTURES.json TOKENIZER.json OUT.json [cuda|cpu] [numeric_bytes]
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    artifact::Artifact,
    artifact_device::Resident,
    engine::sha256,
    import::import_language_model,
    operator_program::{FamilyInputs, SequenceLayout, SlotValues},
};
use serde::Deserialize;
use serde_json::json;
use std::{path::Path, time::Instant};
#[derive(Deserialize)]
struct Fixtures {
    version: u32,
    tokenizer_sha256: String,
    checkpoint_sha256: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    id: usize,
    group: String,
    pair: String,
    variant: String,
    prompt: String,
    tokens: Vec<u32>,
    offsets: Vec<(usize, usize)>,
    target: u32,
    foil: u32,
    target_text: String,
    foil_text: String,
}
fn metrics(logits: &[f64], target: usize, foil: usize) -> Result<serde_json::Value, String> {
    if logits.is_empty()
        || target >= logits.len()
        || foil >= logits.len()
        || target == foil
        || logits.iter().any(|x| !x.is_finite())
    {
        return Err("invalid logits or candidates".into());
    }
    let peak = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z = peak + logits.iter().map(|x| (x - peak).exp()).sum::<f64>().ln();
    let top = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .ok_or("empty logits")?
        .0;
    Ok(
        json!({"target_log_probability":logits[target]-z,"foil_log_probability":logits[foil]-z,"target_probability":(logits[target]-z).exp(),"foil_probability":(logits[foil]-z).exp(),"target_rank":1+logits.iter().filter(|&&x|x>logits[target]).count(),"target_minus_foil_log_odds":logits[target]-logits[foil],"forced_choice_correct":logits[target]>logits[foil],"top1_token":top,"top1_log_probability":logits[top]-z}),
    )
}
fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<_> = std::env::args().collect();
    if !(5..=7).contains(&args.len()) {
        return Err("EXPORT FIXTURES TOKENIZER OUT [cuda|cpu] [numeric_bytes]".into());
    }
    let export = Path::new(&args[1]);
    let fixture_path = Path::new(&args[2]);
    let out = Path::new(&args[4]);
    if out.exists() {
        return Err("report already exists".into());
    }
    let fixtures: Fixtures =
        serde_json::from_slice(&std::fs::read(fixture_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if fixtures.version != 1
        || fixtures.cases.is_empty()
        || sha256(Path::new(&args[3]))? != fixtures.tokenizer_sha256
    {
        return Err("fixture version or tokenizer provenance mismatch".into());
    }
    let start = Instant::now();
    let imported = import_language_model(export, 1, 1)?;
    let lineage = imported.record["source"]["checkpoint_sha256"]
        .as_str()
        .or_else(|| imported.record["source"]["weights_sha256"].as_str())
        .ok_or("missing native checkpoint/weights lineage")?;
    if lineage != fixtures.checkpoint_sha256 {
        return Err("checkpoint lineage mismatch".into());
    }
    if let (Some(a), Some(b)) = (
        imported.record["source"]["checkpoint_sha256"].as_str(),
        imported.record["source"]["weights_sha256"].as_str(),
    ) {
        if a != b {
            return Err("conflicting native lineage aliases".into());
        }
    }
    let files = imported.record["files"]
        .as_object()
        .ok_or("missing export files")?;
    for (name, metadata) in files {
        if name == "tokens" {
            continue;
        } // Imported token table is replaced by frozen fixture tokens.
        let expected = metadata["sha256"].as_str().ok_or("missing file hash")?;
        if sha256(&export.join(format!("{name}.f64")))? != expected {
            return Err(format!("export file hash mismatch: {name}"));
        }
    }
    let vocab = imported.record["config"]["vocab"]
        .as_u64()
        .ok_or("vocab missing")? as usize;
    let context = imported.record["config"]["n_ctx"]
        .as_u64()
        .ok_or("context missing")? as usize;
    let backend = args.get(5).map(String::as_str).unwrap_or("cuda");
    let budget = args.get(6).map_or(Ok(8usize << 30), |v| {
        v.parse().map_err(|e| format!("budget:{e}"))
    })?;
    let device = match backend {
        "cuda" => Some(
            Device::accelerator(GpuPolicy::Required)
                .map_err(|e| e.to_string())?
                .ok_or("CUDA required")?,
        ),
        "cpu" => None,
        _ => return Err("backend cuda or cpu required".into()),
    };
    if device.as_ref().is_some_and(|d| d.is_host() || !d.float64()) {
        return Err("float64 accelerator required; no fallback".into());
    }
    let artifact = Artifact::native(&imported.program)?;
    let resident = device
        .as_ref()
        .map(|d| Resident::from_decoded(d, &artifact))
        .transpose()?;
    let init_seconds = start.elapsed().as_secs_f64();
    let mut results = Vec::new();
    for c in fixtures.cases {
        let rows = c.tokens.len();
        if rows == 0
            || rows > context
            || c.offsets.len() != rows
            || c.tokens.iter().any(|&x| x as usize >= vocab)
            || c.offsets.iter().any(|&(a, b)| a > b || b > c.prompt.len())
        {
            return Err(format!("invalid fixture {}", c.id));
        }
        let family = FamilyInputs {
            rows,
            slots: vec![SlotValues::Tokens(c.tokens.clone())],
            layout: Some(SequenceLayout {
                sequence: vec![0; rows],
                position: (0..rows as u32).collect(),
            }),
        };
        let t = Instant::now();
        let logits = if let (Some(d), Some(r)) = (&device, &resident) {
            if r.estimated_resident_bytes(rows)? > budget {
                return Err(
                    "numeric resident plan exceeds budget (excludes CUDA/library scratch)".into(),
                );
            }
            let trace = r.forward_edited(&family, |_, _| Ok(None))?;
            d.download(
                &d.rows_of(r.output_ref(&trace)?, rows - 1, 1)
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?
            .row(0)
            .to_vec()
        } else {
            let trace = imported
                .program
                .execute(&family, false)
                .map_err(|e| e.to_string())?;
            trace.values[imported.program.output].row(rows - 1).to_vec()
        };
        results.push(json!({"id":c.id,"group":c.group,"pair":c.pair,"variant":c.variant,"prompt":c.prompt,"tokens":c.tokens,"offsets":c.offsets,"target_token":c.target,"foil_token":c.foil,"target_text":c.target_text,"foil_text":c.foil_text,"metrics":metrics(&logits,c.target as usize,c.foil as usize)?,"seconds":t.elapsed().as_secs_f64()}));
    }
    let report = json!({"scope":"Native context-conditioned next-token behavior only; no mechanism recovery, acceptance verdict or neural numerical certificate. Forced choice can succeed at negligible absolute probability; probabilities and ranks are reported.","numerical_model":"Actual imported f64 native weights, operational f64 exp/log; no serialized f32 rounding.","backend":backend,"device":device.as_ref().map(|d|d.name()),"numeric_budget_bytes":budget,"budget_exclusions":"CUDA context, library scratch, host model, transient uploads; plan checked before each forward, initial uploads precede plan check.","fixtures_sha256":sha256(fixture_path)?,"tokenizer_sha256":fixtures.tokenizer_sha256,"native_export":imported.record,"initialization_seconds":init_seconds,"total_seconds":start.elapsed().as_secs_f64(),"results":results});
    std::fs::write(
        out,
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn odds_and_probability() {
        let m = metrics(&[0., 2., 1.], 1, 2).expect("finite logits");
        assert_eq!(m["target_minus_foil_log_odds"], 1.);
        assert_eq!(m["target_rank"], 1);
        assert!(m["target_probability"].as_f64().expect("probability") > 0.6);
        assert!(metrics(&[f64::NAN], 0, 0).is_err());
    }
}

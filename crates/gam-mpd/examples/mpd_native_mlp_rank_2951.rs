//! Measured-family necessary Local rank floors, never an acceptance shortcut.
//! LOCAL_EXPORT OUT.json [layer=0 sequences=2 context=512 ranks=1,2,4,8]
//! Select the same native model and complete token sequences as the Local family.
use gam_linalg::decompose::svd;
use gam_mpd::acceptance::units;
use gam_mpd::counterfactual::Decoder;
use gam_mpd::import::import_language_model;
use gam_mpd::native_mlp_rank::measured_rank_floor;
use gam_mpd::run_check::{layer_nodes, split_sites};
use ndarray::{Axis, s};
use serde_json::json;
use std::{collections::BTreeMap, path::Path, process::Command, time::Instant};
fn hash(path: &Path) -> Result<String, String> {
    for (exe, args) in [("sha256sum", vec![]), ("shasum", vec!["-a", "256"])] {
        if let Ok(out) = Command::new(exe).args(args).arg(path).output() {
            if out.status.success() {
                let text = String::from_utf8(out.stdout).map_err(|e| e.to_string())?;
                let h = text.split_whitespace().next().ok_or("empty hash")?;
                if h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Ok(h.into());
                }
            }
        }
    }
    Err("SHA-256 tool required".into())
}
fn main() -> Result<(), String> {
    let started = Instant::now();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err("LOCAL_EXPORT OUT.json [layer/sequences/context/ranks=...]".into());
    }
    let mut opts = BTreeMap::new();
    for arg in &args[2..] {
        let (key, value) = arg.split_once('=').ok_or("key=value required")?;
        if !["layer", "sequences", "context", "ranks"].contains(&key)
            || opts.insert(key, value).is_some()
        {
            return Err(format!("unknown/duplicate option {key}"));
        }
    }
    let number = |key: &str, default: usize| -> Result<usize, String> {
        opts.get(key).map_or(Ok(default), |v| {
            v.parse().map_err(|_| format!("invalid {key}"))
        })
    };
    let layer = number("layer", 0)?;
    let sequences = number("sequences", 2)?;
    let context = number("context", 512)?;
    if sequences == 0 || context == 0 {
        return Err("positive complete sequence count/context required".into());
    }
    let ranks: Vec<usize> = opts
        .get("ranks")
        .copied()
        .unwrap_or("1,2,4,8")
        .split(',')
        .map(|v| v.parse().map_err(|_| "invalid rank".to_string()))
        .collect::<Result<_, _>>()?;
    if ranks.is_empty() {
        return Err("declared rank list required".into());
    }
    let export = Path::new(&args[0]);
    let imported = import_language_model(export, sequences, context)?;
    let mut native = split_sites(&imported.program)?;
    let decoder = Decoder::from_export(export)?;
    let layers = layer_nodes(&native, decoder.layers())?;
    let sites = layers.get(layer).ok_or("layer outside native decoder")?;
    let weights = native
        .operators
        .iter()
        .find(|o| o.name == format!("blocks.{layer}.down_proj"))
        .ok_or("missing native output weights")?
        .matrix();
    let basis = svd(weights.view(), false).map_err(|e| e.to_string())?;
    let resolved_rank = basis
        .singular_values
        .iter()
        .filter(|&&v| v > basis.band)
        .count();
    if ranks.iter().any(|&k| k > resolved_rank) {
        return Err("requested rank exceeds numerically resolved native weight rank".into());
    }
    native.nodes.truncate(sites.mlp + 1);
    native.output = sites.mlp;
    let mut writes = Vec::new();
    let family_units = units(&imported.family);
    if family_units.len() != sequences {
        return Err("sequence identity count changed".into());
    }
    for rows in &family_units {
        let trace = native
            .execute(&imported.family.select(rows), false)
            .map_err(|e| e.to_string())?;
        writes.push(trace.values[sites.mlp].clone());
    }
    let values = ndarray::concatenate(
        Axis(0),
        &writes.iter().map(|w| w.view()).collect::<Vec<_>>(),
    )
    .map_err(|e| e.to_string())?;
    let mut floors = Vec::new();
    for &k in &ranks {
        let writers = basis.u.slice(s![.., ..k]).t().mapv(|v| v as f32 as f64);
        floors.push(measured_rank_floor(
            &values,
            k,
            if k == 0 { None } else { Some(&writers) },
        )?);
    }
    let mut sources = BTreeMap::new();
    for entry in std::fs::read_dir(export).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_file() && path.extension().is_some_and(|e| e == "f64" || e == "json") {
            sources.insert(
                path.file_name().unwrap().to_string_lossy().to_string(),
                hash(&path)?,
            );
        }
    }
    let report = json!({"scope":"necessary affine output-space rank lower bounds on this measured Local family; independent of input law; proposal guidance only; no candidate pruning or acceptance change","local_definition":"maximum per-row Euclidean native-write disagreement / RMS native-write row norm over family","numeric_scope":"comparison arithmetic and a posteriori SVD reconstruction/orthogonality enclosures for finite native writes; geometric necessity assumes exact affine output-space membership; excludes candidate output-map/neural-forward arithmetic and out-of-span exceptions","native_write":sites.mlp,"native_output_domain":format!("{:?}",native.interfaces().map_err(|e|e.to_string())?[sites.mlp]),"local_export":export.canonicalize().map_err(|e|e.to_string())?,"source_sha256":sources,"sequence_ids":(0..sequences).collect::<Vec<_>>(),"context":context,"layer":layer,"ranks":ranks,"writer_basis":"native down_proj leading left singular directions, rounded to f32 as serialized proposal writers; unrestricted affine offset for a conservative floor","native_weight_svd_resolution_cutoff":basis.band,"native_weight_resolved_rank":resolved_rank,"floors":floors,"seconds":started.elapsed().as_secs_f64()});
    std::fs::write(
        &args[1],
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

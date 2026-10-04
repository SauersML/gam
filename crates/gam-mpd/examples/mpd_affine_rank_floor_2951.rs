//! Offline fixed-array affine-plus-rank diagnostic; never an acceptance verdict.
//! EXTRACT.json OUT.json [PANEL=eval]; no fitting, neural execution, or GPU.
use gam_mpd::native_mlp_rank::measured_affine_correction_rank_curve;
use ndarray::Array2;
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::Path, process::Command, time::Instant};
const RANKS: &[usize] = &[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768];
fn hash(path: &Path) -> Result<String, String> {
    for (exe, args) in [("sha256sum", vec![]), ("shasum", vec!["-a", "256"])] {
        if let Ok(out) = Command::new(exe).args(args).arg(path).output() {
            if out.status.success() {
                let s = String::from_utf8(out.stdout).map_err(|e| e.to_string())?;
                let h = s.split_whitespace().next().ok_or("empty SHA256 output")?;
                if h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Ok(h.to_ascii_lowercase());
                }
            }
        }
    }
    Err("SHA256 utility required".into())
}
fn integer(v: &Value, key: &str) -> Result<usize, String> {
    usize::try_from(
        v[key]
            .as_u64()
            .ok_or_else(|| format!("missing integer {key}"))?,
    )
    .map_err(|_| format!("{key} overflow"))
}
fn read_array(bytes: &[u8], rows: usize, cols: usize) -> Result<Array2<f64>, String> {
    if rows == 0 || cols == 0 {
        return Err("nonempty matrix required".into());
    }
    let count = rows.checked_mul(cols).ok_or("shape overflow")?;
    if bytes.len() != count.checked_mul(8).ok_or("byte size overflow")? {
        return Err("raw f64 size does not match declared shape".into());
    }
    let values: Vec<_> = bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().expect("eight-byte chunk")))
        .collect();
    if values.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite archived value".into());
    }
    Array2::from_shape_vec((rows, cols), values).map_err(|e| e.to_string())
}
fn array(root: &Path, descriptor: &Value, rows: usize) -> Result<(Array2<f64>, Value), String> {
    let file = descriptor["file"].as_str().ok_or("missing array file")?;
    // Archives must be siblings of their manifest, never arbitrary paths.
    if Path::new(file).components().count() != 1 || file == "." || file == ".." {
        return Err("array filename must be a sibling filename".into());
    }
    let path = root.join(file);
    let declared = descriptor["sha256"]
        .as_str()
        .ok_or("missing array SHA256")?;
    let actual = hash(&path)?;
    if actual != declared.to_ascii_lowercase() {
        return Err(format!("SHA256 mismatch for {file}"));
    }
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let width = integer(descriptor, "width")?;
    Ok((
        read_array(&bytes, rows, width)?,
        json!({"descriptor":descriptor,"path":path,"sha256":actual,"bytes":bytes.len(),"rows":rows,"width":width}),
    ))
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err("EXTRACT.json OUT.json [PANEL=eval]".into());
    }
    let started = Instant::now();
    let manifest = Path::new(&args[0]);
    let root = manifest.parent().ok_or("manifest parent required")?;
    let data: Value = serde_json::from_slice(&std::fs::read(manifest).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let name = args.get(2).map(String::as_str).unwrap_or("eval");
    let panels = data["panels"].as_array().ok_or("panels array required")?;
    let matches: Vec<_> = panels
        .iter()
        .filter(|p| p["name"].as_str() == Some(name))
        .collect();
    if matches.len() != 1 {
        return Err("exactly one matching named panel required".into());
    }
    let panel = matches[0];
    let rows = integer(panel, "rows")?;
    let descriptors = panel["values"].as_array().ok_or("panel values required")?;
    let mut inventory = BTreeSet::new();
    for d in descriptors {
        let layer = integer(d, "layer")?;
        let role = d["role"].as_str().ok_or("array role required")?;
        integer(d, "native_node")?;
        if layer >= 4 || !["input", "write"].contains(&role) || !inventory.insert((layer, role)) {
            return Err("expected unique input/write inventory for layers 0..3".into());
        }
    }
    if inventory.len() != 8 {
        return Err("complete four-layer input/write archive required".into());
    }
    let mut layers = Vec::new();
    for layer in 0..4 {
        let at = Instant::now();
        let descriptor = |role| {
            descriptors
                .iter()
                .find(|d| d["layer"].as_u64() == Some(layer) && d["role"].as_str() == Some(role))
                .expect("validated inventory")
        };
        let (x, x_source) = array(root, descriptor("input"), rows)?;
        let (y, y_source) = array(root, descriptor("write"), rows)?;
        let buffers = (x.len() + y.len())
            .checked_mul(8)
            .ok_or("buffer size overflow")?;
        let diagnostic = match measured_affine_correction_rank_curve(&x, &y, RANKS) {
            Ok(curve) => json!({"status":"MeasuredFixedArrayFloor","curve":curve}),
            Err(reason) => json!({"status":"Unresolved","reason":reason}),
        };
        layers.push(json!({"layer":layer,"input":x_source,"native_write":y_source,"diagnostic":diagnostic,"numeric_input_buffer_bytes":buffers,"seconds":at.elapsed().as_secs_f64()}));
    }
    let report = json!({"scope":"Fixed finite binary64 archived arrays only. Unrestricted affine base plus ideal rank-K correction; necessary geometric diagnostic, no acceptance/pruning or neural-execution certificate. Unresolved design rank is not zero.","normalization":"Tail Frobenius residual / original native-write Frobenius norm lower-bounds maximum row Euclidean error / RMS native-write norm.","declared_ranks":RANKS,"manifest":{"path":manifest,"sha256":hash(manifest)?},"panel":name,"panel_record":panel["record"],"context":data["context"],"rows":rows,"layers":layers,"memory_scope":"One layer's X/Y input buffers retained at a time; JSON curve retained. SVD factors/workspaces and allocator are additional, not bounded by reported numeric input bytes.","seconds":started.elapsed().as_secs_f64()});
    std::fs::write(
        &args[1],
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_arrays_require_exact_finite_shape_and_preserve_signed_zero() {
        let bytes: Vec<_> = [1.0f64, -0.0]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect();
        let a = read_array(&bytes, 1, 2).expect("valid array");
        assert_eq!(a[(0, 1)].to_bits(), (-0.0f64).to_bits());
        assert!(read_array(&bytes, 2, 2).is_err());
        assert!(read_array(&f64::NAN.to_le_bytes(), 1, 1).is_err());
        assert!(read_array(&[], 0, 1).is_err());
        assert!(read_array(&[], usize::MAX, 2).is_err());
    }
}

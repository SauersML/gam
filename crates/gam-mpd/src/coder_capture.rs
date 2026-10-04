//! Explicit real-fit coder fixtures. No environment hook and no change to default coding.
use super::core_device::SelectionProducts;
use super::sparse_code::Coder;
use ndarray::Array2;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

pub struct Fixture {
    pub products: SelectionProducts,
    pub gram: Array2<f64>,
    pub starts: Vec<u32>,
    pub bits: Vec<f64>,
    pub kappa: f64,
    pub nodes: usize,
    pub tolerance: f64,
    pub warm: Option<Array2<f64>>,
}

pub fn sha256(path: &Path) -> Result<String, String> {
    for (program, args) in [("sha256sum", vec![]), ("shasum", vec!["-a", "256"])] {
        match Command::new(program).args(args).arg("--").arg(path).output() {
            Ok(output) if output.status.success() => {
                let text = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
                let hash = text.split_whitespace().next().ok_or("empty hash")?;
                if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) { return Err("invalid SHA256".into()); }
                return Ok(hash.to_ascii_lowercase());
            }
            Ok(output) => return Err(String::from_utf8_lossy(&output.stderr).into_owned()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e.to_string()),
        }
    }
    Err("SHA256 requires sha256sum or shasum".into())
}

impl Fixture {
    pub fn from_coder(products: SelectionProducts, coder: &Coder) -> Result<Self, String> {
        let fixture = Self { products, gram: coder.gram().clone(), starts: coder.starts().iter().map(|s| u32::try_from(*s).map_err(|e| e.to_string())).collect::<Result<_, _>>()?,
            bits: coder.bits().to_vec(), kappa: coder.kappa(), nodes: coder.nodes(), tolerance: 1e-12, warm: None };
        fixture.validate()?;
        Ok(fixture)
    }

    pub fn validate(&self) -> Result<(), String> {
        let (rows, columns) = self.products.z.dim();
        let blocks = self.bits.len();
        if rows == 0 || columns == 0 || blocks == 0 || self.products.weights.dim() != (rows, columns) || self.products.yfy.len() != rows || self.gram.dim() != (columns, columns)
            || self.starts.len() != blocks + 1 || self.starts.first() != Some(&0) || self.starts.last().copied() != u32::try_from(columns).ok()
            || self.starts.windows(2).any(|s| s[0] >= s[1]) || self.nodes == 0 || !self.kappa.is_finite() || self.kappa <= 0.0 || !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err("invalid coder fixture shapes/settings/partition".into());
        }
        if self.products.z.iter().chain(self.products.weights.iter()).chain(self.products.yfy.iter()).chain(self.gram.iter()).chain(self.bits.iter()).any(|v| !v.is_finite())
            || self.bits.iter().any(|b| *b < 0.0) { return Err("nonfinite operands or negative prices".into()); }
        if let Some(warm) = &self.warm {
            if warm.dim() != (rows, blocks) || warm.iter().any(|v| *v != 0.0 && *v != 1.0) { return Err("invalid warm masks".into()); }
        }
        Ok(())
    }

    /// Fresh directory; exact little-endian payloads; metadata is the completion marker.
    pub fn write(&self, out: &Path, provenance: Value, extra: &[(&str, &Array2<f64>)]) -> Result<(), String> {
        self.validate()?;
        if out.exists() { return Err("fixture directory already exists".into()); }
        std::fs::create_dir(out).map_err(|e| e.to_string())?;
        let mut files = BTreeMap::new();
        let mut write = |name: &str, shape: Vec<usize>, bytes: Vec<u8>| -> Result<(), String> {
            if files.contains_key(name) || name.contains('/') || name == "MANIFEST.json" { return Err("invalid fixture filename".into()); }
            let temporary = out.join(format!("{name}.partial"));
            std::fs::write(&temporary, &bytes).map_err(|e| e.to_string())?;
            let path = out.join(name);
            std::fs::rename(temporary, &path).map_err(|e| e.to_string())?;
            files.insert(name.to_string(), json!({"shape":shape,"bytes":bytes.len(),"sha256":sha256(&path)?}));
            Ok(())
        };
        let raw = |values: &[f64]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        for (name, matrix) in [("z.f64", &self.products.z), ("weights.f64", &self.products.weights), ("gram.f64", &self.gram)] {
            write(name, vec![matrix.nrows(), matrix.ncols()], raw(matrix.as_slice().ok_or("noncontiguous fixture matrix")?))?;
        }
        write("yfy.f64", vec![self.products.yfy.len(), 1], raw(&self.products.yfy))?;
        write("bits.f64", vec![1, self.bits.len()], raw(&self.bits))?;
        write("starts.u32", vec![self.starts.len()], self.starts.iter().flat_map(|v| v.to_le_bytes()).collect())?;
        if let Some(warm) = &self.warm { write("warm.f64", vec![warm.nrows(), warm.ncols()], raw(warm.as_slice().ok_or("noncontiguous warm")?))?; }
        for &(name, matrix) in extra { write(name, vec![matrix.nrows(), matrix.ncols()], raw(matrix.as_slice().ok_or("noncontiguous extra matrix")?))?; }
        let manifest = json!({"schema":"actual-fit-coder-operands-v1","endianness":"little","files":files,"kappa_bits":self.kappa.to_bits(),"nodes":self.nodes,
            "tolerance_bits":self.tolerance.to_bits(),"warm":self.warm.is_some(),"provenance":provenance});
        std::fs::write(out.join("MANIFEST.json.partial"), serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        std::fs::rename(out.join("MANIFEST.json.partial"), out.join("MANIFEST.json")).map_err(|e| e.to_string())
    }

    /// Verify every payload's length and digest, then decode and validate before GPU upload.
    pub fn read(dir: &Path) -> Result<(Self, Value), String> {
        let manifest: Value = serde_json::from_slice(&std::fs::read(dir.join("MANIFEST.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        if manifest["schema"] != "actual-fit-coder-operands-v1" || manifest["endianness"] != "little" { return Err("unknown fixture schema".into()); }
        let files = manifest["files"].as_object().ok_or("missing files")?;
        let mut payloads = BTreeMap::new();
        for (name, description) in files {
            if name.contains('/') || name.contains("..") { return Err("unsafe fixture filename".into()); }
            let path = dir.join(name);
            let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
            if description["bytes"].as_u64() != u64::try_from(bytes.len()).ok() || description["sha256"].as_str() != Some(sha256(&path)?.as_str()) { return Err(format!("invalid length/digest {name}")); }
            payloads.insert(name.as_str(), bytes);
        }
        let matrix = |name: &str| -> Result<Array2<f64>, String> {
            let shape = files.get(name).ok_or_else(|| format!("missing {name}"))?["shape"].as_array().ok_or("missing shape")?;
            if shape.len() != 2 { return Err("matrix shape needs two dimensions".into()); }
            let dim = |i: usize| -> Result<usize, String> { usize::try_from(shape[i].as_u64().ok_or("invalid dimension")?).map_err(|e| e.to_string()) };
            let (rows, cols) = (dim(0)?, dim(1)?);
            let bytes = payloads.get(name).ok_or("missing payload")?;
            if rows.checked_mul(cols).and_then(|n| n.checked_mul(8)) != Some(bytes.len()) { return Err("invalid matrix byte length".into()); }
            let values = bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().expect("eight-byte chunk"))).collect();
            Array2::from_shape_vec((rows, cols), values).map_err(|e| e.to_string())
        };
        let starts = payloads.get("starts.u32").ok_or("missing starts")?;
        if starts.len() % 4 != 0 { return Err("invalid starts byte length".into()); }
        let z = matrix("z.f64")?;
        let yfy = matrix("yfy.f64")?;
        let bits = matrix("bits.f64")?;
        if yfy.dim() != (z.nrows(), 1) || bits.nrows() != 1 || files["starts.u32"]["shape"] != json!([starts.len() / 4]) {
            return Err("invalid vector operand shapes".into());
        }
        let fixture = Self { products: SelectionProducts { z, weights: matrix("weights.f64")?, yfy: yfy.into_raw_vec_and_offset().0 },
            gram: matrix("gram.f64")?, starts: starts.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().expect("four-byte chunk"))).collect(),
            bits: bits.into_raw_vec_and_offset().0, kappa: f64::from_bits(manifest["kappa_bits"].as_u64().ok_or("missing kappa")?),
            nodes: usize::try_from(manifest["nodes"].as_u64().ok_or("missing nodes")?).map_err(|e| e.to_string())?,
            tolerance: f64::from_bits(manifest["tolerance_bits"].as_u64().ok_or("missing tolerance")?), warm: if manifest["warm"].as_bool().ok_or("missing warm setting")? { Some(matrix("warm.f64")?) } else { None } };
        fixture.validate()?;
        Ok((fixture, manifest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn actual_operand_fixture_roundtrips_bits_and_rejects_corruption() {
        let coder = Coder::new(array![[2.0, 1.0], [1.0, 3.0]], &[1, 1], &[2.25, 3.5], 100000.0, 64).unwrap();
        let products = SelectionProducts { z: array![[0.0, -0.0], [1.25, -2.5]], weights: array![[0.5, 1.0], [-1.0, 2.0]], yfy: vec![2.0, 3.0] };
        let fixture = Fixture::from_coder(products, &coder).unwrap();
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("gam-coder-fixture-{}-{stamp}", std::process::id()));
        fixture.write(&dir, json!({"scope":"test actual format; not performance evidence"}), &[]).unwrap();
        let (decoded, manifest) = Fixture::read(&dir).unwrap();
        assert_eq!(decoded.products.z.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), fixture.products.z.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        assert_eq!(decoded.kappa.to_bits(), fixture.kappa.to_bits());
        assert_eq!(decoded.tolerance.to_bits(), fixture.tolerance.to_bits());
        assert_eq!(decoded.starts, vec![0, 1, 2]);
        assert!(!manifest["warm"].as_bool().unwrap());
        assert!(fixture.write(&dir, json!({}), &[]).is_err());
        let mut bytes = std::fs::read(dir.join("weights.f64")).unwrap(); bytes[0] ^= 1;
        std::fs::write(dir.join("weights.f64"), bytes).unwrap();
        assert!(Fixture::read(&dir).is_err(), "changed exact operand bytes cannot be uploaded");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn actual_operand_fixture_rejects_wrong_partition_and_nonfinite_values() {
        let coder = Coder::new(array![[1.0, 0.0], [0.0, 1.0]], &[1, 1], &[1.0, 1.0], 10.0, 64).unwrap();
        let mut fixture = Fixture::from_coder(SelectionProducts { z: array![[1.0, 2.0]], weights: array![[3.0, 4.0]], yfy: vec![5.0] }, &coder).unwrap();
        fixture.starts[1] = 0; assert!(fixture.validate().is_err()); fixture.starts[1] = 1;
        fixture.products.weights[[0, 0]] = f64::NAN; assert!(fixture.validate().is_err());
    }
}

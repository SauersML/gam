//! Reproducible agreement/timing probe for eigenvector scaling versus `Eigh::map`.
//! CLI: TERMS NAME ROWS COLS f32|f64 mean|sum RAW_FILE [NAME ROWS COLS ...]
//! Inputs are declared little-endian raw matrices; `mean` forms WᵀW/rows,
//! `sum` forms WᵀW. No checkpoint/library paths or fidelity claim are implicit.

use gam_linalg::decompose::eigh;
use gam_linalg::matrix::symmetrize;
use gam_linalg::roundoff::SymmetricAssembly;
use ndarray::Array2;
use std::time::Instant;

fn matrix_file(path: &str, rows: usize, cols: usize, dtype: &str) -> Result<Array2<f64>, String> {
    let stride = match dtype {
        "f32" => 4,
        "f64" => 8,
        _ => return Err("dtype must be f32 or f64".into()),
    };
    let count = rows
        .checked_mul(cols)
        .filter(|_| rows > 0 && cols > 0)
        .ok_or("positive matrix shape overflow")?;
    let expected = count
        .checked_mul(stride)
        .ok_or("matrix byte count overflow")?;
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "{path}: expected {expected} bytes, got {}",
            bytes.len()
        ));
    }
    let values = bytes
        .chunks_exact(stride)
        .map(|c| {
            if stride == 4 {
                f64::from(f32::from_le_bytes(c.try_into().expect("four-byte chunk")))
            } else {
                f64::from_le_bytes(c.try_into().expect("eight-byte chunk"))
            }
        })
        .collect::<Vec<_>>();
    if values.iter().any(|x| !x.is_finite()) {
        return Err(format!("{path}: nonfinite matrix"));
    }
    Array2::from_shape_vec((rows, cols), values).map_err(|e| e.to_string())
}

fn old_function(m: &Array2<f64>, f: &dyn Fn(f64, f64) -> f64) -> Result<Array2<f64>, String> {
    let d =
        eigh(symmetrize(m).view(), SymmetricAssembly::Mirrored, None).map_err(|e| e.to_string())?;
    let mut scaled = d.vectors.clone();
    for (k, l) in d.values.iter().enumerate() {
        let w = f(*l, d.band);
        scaled.column_mut(k).mapv_inplace(|x| x * w);
    }
    Ok(scaled.dot(&d.vectors.t()))
}

fn new_function(m: &Array2<f64>, f: &dyn Fn(f64, f64) -> f64) -> Result<Array2<f64>, String> {
    let d =
        eigh(symmetrize(m).view(), SymmetricAssembly::Mirrored, None).map_err(|e| e.to_string())?;
    Ok(d.map(|l| f(l, d.band)))
}

fn relative(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
        / scale.max(f64::MIN_POSITIVE)
}

fn main() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 7 || (args.len() - 1) % 6 != 0 {
        return Err("TERMS NAME ROWS COLS f32|f64 mean|sum RAW_FILE [NAME ROWS COLS ...]".into());
    }
    let terms = args[0].parse::<f64>().map_err(|e| e.to_string())?;
    if !terms.is_finite() || terms <= 0.0 {
        return Err("TERMS must be finite and positive".into());
    }
    let mut cases = Vec::new();
    for spec in args[1..].chunks_exact(6) {
        let rows = spec[1].parse::<usize>().map_err(|e| e.to_string())?;
        let cols = spec[2].parse::<usize>().map_err(|e| e.to_string())?;
        let matrix = matrix_file(&spec[5], rows, cols, &spec[3])?;
        let divisor = match spec[4].as_str() {
            "mean" => rows as f64,
            "sum" => 1.0,
            _ => return Err("normalization must be mean or sum".into()),
        };
        let gram = matrix.t().dot(&matrix) / divisor;
        if gram.iter().any(|x| !x.is_finite()) {
            return Err(format!("{}: Gram matrix overflow", spec[5]));
        }
        cases.push((
            format!(
                "{} {}x{} {} {} {}",
                spec[0], rows, cols, spec[3], spec[4], spec[5]
            ),
            gram,
        ));
    }
    for (name, m) in &cases {
        let d = eigh(symmetrize(m).view(), SymmetricAssembly::Mirrored, None)
            .map_err(|e| e.to_string())?;
        let largest = d.values.iter().fold(0.0_f64, |a, l| a.max(*l));
        let kept = d.values.iter().filter(|l| **l > d.band).count();
        let mean = (0..m.nrows()).map(|i| m[[i, i]]).sum::<f64>() / m.nrows() as f64;
        let single = (largest * terms.sqrt() * f64::from(f32::EPSILON) / 2.0).max(d.band);
        println!(
            "{name}: order {}, {kept} eigenvalues above the band",
            m.nrows()
        );
        let functions: Vec<(&str, Box<dyn Fn(f64, f64) -> f64>)> = vec![
            (
                "band root",
                Box::new(|l: f64, b: f64| if l > b { l.sqrt() } else { 0.0 }),
            ),
            (
                "band inverse root",
                Box::new(|l: f64, b: f64| if l > b { 1.0 / l.sqrt() } else { 0.0 }),
            ),
            (
                "band pseudo-inverse",
                Box::new(|l: f64, b: f64| if l > b { 1.0 / l } else { 0.0 }),
            ),
            (
                "single-precision pseudo-inverse",
                Box::new(move |l: f64, _| if l > single { 1.0 / l } else { 0.0 }),
            ),
            (
                "shrunk inverse",
                Box::new(move |l: f64, _| 1.0 / (l.max(0.0) + mean)),
            ),
        ];
        for (label, f) in &functions {
            let (mut old_seconds, mut new_seconds) = (f64::INFINITY, f64::INFINITY);
            let (mut old, mut new) = (Array2::zeros((0, 0)), Array2::zeros((0, 0)));
            for _ in 0..3 {
                let t = Instant::now();
                old = old_function(m, f.as_ref())?;
                old_seconds = old_seconds.min(t.elapsed().as_secs_f64());
                let t = Instant::now();
                new = new_function(m, f.as_ref())?;
                new_seconds = new_seconds.min(t.elapsed().as_secs_f64());
            }
            println!(
                "  {label:32} max |new − old| / max |old| = {:.2e}   old {:.3}s  new {:.3}s",
                relative(&new, &old),
                old_seconds,
                new_seconds
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_input_has_explicit_shape_precision_and_finiteness() {
        let path = std::env::temp_dir().join(format!("gamcore-agree-{}.bin", std::process::id()));
        let text = path.to_str().expect("temporary path");
        std::fs::write(
            &path,
            [1.0f32.to_le_bytes(), (-0.0f32).to_le_bytes()].concat(),
        )
        .expect("write fixture");
        let matrix = matrix_file(text, 1, 2, "f32").expect("declared f32 matrix");
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[0, 1]].to_bits(), (-0.0f64).to_bits());
        assert!(matrix_file(text, 2, 2, "f32").is_err());
        assert!(matrix_file(text, 1, 2, "f16").is_err());
        assert!(matrix_file(text, 0, 2, "f32").is_err());
        assert!(matrix_file(text, usize::MAX, 2, "f64").is_err());
        std::fs::write(&path, f64::NAN.to_le_bytes()).expect("write nonfinite fixture");
        assert!(matrix_file(text, 1, 1, "f64").is_err());
        std::fs::remove_file(path).expect("remove fixture");
    }
}

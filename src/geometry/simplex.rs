use ndarray::{Array2, ArrayView1, ArrayView2};

fn validate_simplex_array(points: ArrayView2<'_, f64>) -> Result<(), String> {
    let (n, d) = points.dim();
    if n == 0 || d < 2 {
        return Err(
            "simplex values must have at least one row and at least two columns".to_string(),
        );
    }
    if let Some(((row, col), value)) = points.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(format!(
            "simplex values must contain only finite values; got {value} at ({row}, {col})"
        ));
    }
    Ok(())
}

fn normalize_weights(
    n: usize,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Vec<f64>, String> {
    match weights {
        None => Ok(vec![1.0 / n as f64; n]),
        Some(w) => {
            if w.len() != n {
                return Err("weights length must match the number of rows".to_string());
            }
            let mut total = 0.0_f64;
            for value in w.iter() {
                if !value.is_finite() || *value < 0.0 {
                    return Err(
                        "weights must be finite, non-negative, and have positive total"
                            .to_string(),
                    );
                }
                total += *value;
            }
            if total <= 0.0 {
                return Err(
                    "weights must be finite, non-negative, and have positive total".to_string(),
                );
            }
            Ok(w.iter().map(|v| *v / total).collect())
        }
    }
}

fn closure(points: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    validate_simplex_array(points)?;
    let (n, d) = points.dim();
    let mut out = Array2::<f64>::zeros((n, d));
    for row in 0..n {
        let mut total = 0.0_f64;
        for col in 0..d {
            let v = points[[row, col]];
            if v < 0.0 {
                return Err("simplex values must be non-negative".to_string());
            }
            total += v;
        }
        if total <= 0.0 {
            return Err("simplex rows must have positive total mass".to_string());
        }
        for col in 0..d {
            out[[row, col]] = points[[row, col]] / total;
        }
    }
    Ok(out)
}

fn require_positive(comp: ArrayView2<'_, f64>) -> Result<(), String> {
    for value in comp.iter() {
        if *value <= 0.0 {
            return Err(
                "simplex Fr\u{e9}chet mean require strictly positive simplex values".to_string(),
            );
        }
    }
    Ok(())
}

pub fn simplex_frechet_mean(
    points: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Vec<f64>, String> {
    let comp = closure(points)?;
    require_positive(comp.view())?;
    let (n, d) = comp.dim();
    let w = normalize_weights(n, weights)?;
    let mut mean_log = vec![0.0_f64; d];
    for row in 0..n {
        for col in 0..d {
            mean_log[col] += w[row] * comp[[row, col]].ln();
        }
    }
    let mut max_v = f64::NEG_INFINITY;
    for &v in mean_log.iter() {
        if v > max_v {
            max_v = v;
        }
    }
    let mut total = 0.0_f64;
    let mut out = vec![0.0_f64; d];
    for col in 0..d {
        let e = (mean_log[col] - max_v).exp();
        out[col] = e;
        total += e;
    }
    for value in out.iter_mut() {
        *value /= total;
    }
    Ok(out)
}

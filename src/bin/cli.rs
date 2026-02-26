#![allow(clippy::type_complexity)]

use gam::estimate::{FitOptions, fit_gam, predict_gam};
use gam::families::royston_parmar::{RoystonParmarInputs, working_model_from_flattened};
use gam::survival::{MonotonicityPenalty, PenaltyBlock, PenaltyBlocks, SurvivalSpec};
use gam::types::LikelihoodFamily;
use gam::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotPlacement, BSplineKnotSpec, BasisMetadata,
    build_bspline_basis_1d,
};
use ndarray::{Array1, Array2, ArrayView1, s};
use std::collections::HashSet;
use std::fs;
use std::time::Instant;

#[derive(Clone, Copy)]
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        let x = self.next_u64() >> 11;
        (x as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().clamp(1e-12, 1.0 - 1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn parse_arg(args: &[String], flag: &str, default: usize) -> usize {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_seed(args: &[String], default: u64) -> u64 {
    args.windows(2)
        .find(|w| w[0] == "--seed")
        .and_then(|w| w[1].parse::<u64>().ok())
        .unwrap_or(default)
}

fn parse_string_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn parse_bool_arg(args: &[String], flag: &str, default: bool) -> bool {
    match parse_string_arg(args, flag) {
        Some(v) => matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "y"),
        None => default,
    }
}

fn parse_csv_list_arg(args: &[String], flag: &str) -> Vec<String> {
    parse_string_arg(args, flag)
        .map(|v| {
            v.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn auc_score(y: ArrayView1<'_, f64>, p: ArrayView1<'_, f64>) -> f64 {
    let mut pairs: Vec<(f64, f64)> = y.iter().copied().zip(p.iter().copied()).collect();
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut rank_sum_pos = 0.0;
    let mut n_pos = 0.0;
    let mut n_neg = 0.0;
    for (idx, (yy, _)) in pairs.iter().enumerate() {
        if *yy > 0.5 {
            rank_sum_pos += (idx + 1) as f64;
            n_pos += 1.0;
        } else {
            n_neg += 1.0;
        }
    }
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

fn brier_score(y: ArrayView1<'_, f64>, p: ArrayView1<'_, f64>) -> f64 {
    let n = y.len() as f64;
    y.iter()
        .copied()
        .zip(p.iter().copied())
        .map(|(yy, pp)| {
            let d = yy - pp;
            d * d
        })
        .sum::<f64>()
        / n
}

fn rmse_score(y: ArrayView1<'_, f64>, mu: ArrayView1<'_, f64>) -> f64 {
    let n = y.len().max(1) as f64;
    let mse = y
        .iter()
        .copied()
        .zip(mu.iter().copied())
        .map(|(yy, pp)| {
            let d = yy - pp;
            d * d
        })
        .sum::<f64>()
        / n;
    mse.sqrt()
}

fn r2_score(y: ArrayView1<'_, f64>, mu: ArrayView1<'_, f64>) -> f64 {
    let n = y.len().max(1) as f64;
    let mean_y = y.iter().copied().sum::<f64>() / n;
    let sst = y
        .iter()
        .copied()
        .map(|yy| {
            let d = yy - mean_y;
            d * d
        })
        .sum::<f64>();
    if sst <= 0.0 {
        return 0.0;
    }
    let sse = y
        .iter()
        .copied()
        .zip(mu.iter().copied())
        .map(|(yy, pp)| {
            let d = yy - pp;
            d * d
        })
        .sum::<f64>();
    1.0 - sse / sst
}

fn load_lidar_dataset() -> Result<(Array1<f64>, Array1<f64>), String> {
    let raw = include_str!("../../benchmarks/datasets/lidar.csv");
    let mut range = Vec::<f64>::new();
    let mut logratio = Vec::<f64>::new();
    for (line_no, line) in raw.lines().enumerate() {
        if line_no == 0 {
            continue;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split(',');
        let r = parts
            .next()
            .ok_or_else(|| format!("lidar parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid range at line {}: {e}", line_no + 1))?;
        let y = parts
            .next()
            .ok_or_else(|| format!("lidar parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid logratio at line {}: {e}", line_no + 1))?;
        range.push(r);
        logratio.push(y);
    }
    if range.is_empty() {
        return Err("lidar dataset is empty".to_string());
    }
    Ok((Array1::from_vec(range), Array1::from_vec(logratio)))
}

fn load_bone_dataset() -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    let raw = include_str!("../../benchmarks/datasets/bone.csv");
    let mut t = Vec::<f64>::new();
    let mut d = Vec::<f64>::new();
    let mut trt_auto = Vec::<f64>::new();
    for (line_no, line) in raw.lines().enumerate() {
        if line_no == 0 {
            continue;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split(',');
        let t_i = parts
            .next()
            .ok_or_else(|| format!("bone parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid t at line {}: {e}", line_no + 1))?;
        let d_i = parts
            .next()
            .ok_or_else(|| format!("bone parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid d at line {}: {e}", line_no + 1))?;
        let trt = parts
            .next()
            .ok_or_else(|| format!("bone parse error at line {}", line_no + 1))?
            .trim_matches('"');
        let trt_i = match trt {
            "allo" => 0.0,
            "auto" => 1.0,
            _ => return Err(format!("invalid trt '{trt}' at line {}", line_no + 1)),
        };
        t.push(t_i);
        d.push(d_i);
        trt_auto.push(trt_i);
    }
    if t.is_empty() {
        return Err("bone dataset is empty".to_string());
    }
    Ok((
        Array1::from_vec(t),
        Array1::from_vec(d),
        Array1::from_vec(trt_auto),
    ))
}

fn load_prostate_dataset() -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    let raw = include_str!("../../benchmarks/datasets/prostate.csv");
    let mut pc1 = Vec::<f64>::new();
    let mut pc2 = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for (line_no, line) in raw.lines().enumerate() {
        if line_no == 0 {
            continue;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split(',');
        let pc1_i = parts
            .next()
            .ok_or_else(|| format!("prostate parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid pc1 at line {}: {e}", line_no + 1))?;
        let pc2_i = parts
            .next()
            .ok_or_else(|| format!("prostate parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid pc2 at line {}: {e}", line_no + 1))?;
        let y_i = parts
            .next()
            .ok_or_else(|| format!("prostate parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid y at line {}: {e}", line_no + 1))?;
        pc1.push(pc1_i);
        pc2.push(pc2_i);
        y.push(y_i);
    }
    if pc1.is_empty() {
        return Err("prostate dataset is empty".to_string());
    }
    Ok((
        Array1::from_vec(pc1),
        Array1::from_vec(pc2),
        Array1::from_vec(y),
    ))
}

fn load_wine_dataset() -> Result<
    (
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
    ),
    String,
> {
    let raw = include_str!("../../benchmarks/datasets/wine.csv");
    let mut year = Vec::<f64>::new();
    let mut h_rain = Vec::<f64>::new();
    let mut w_rain = Vec::<f64>::new();
    let mut h_temp = Vec::<f64>::new();
    let mut s_temp = Vec::<f64>::new();
    let mut price = Vec::<f64>::new();
    for (line_no, line) in raw.lines().enumerate() {
        if line_no == 0 {
            continue;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split(',');
        let year_i = parts
            .next()
            .ok_or_else(|| format!("wine parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid year at line {}: {e}", line_no + 1))?;
        let price_raw = parts
            .next()
            .ok_or_else(|| format!("wine parse error at line {}", line_no + 1))?;
        let h_rain_i = parts
            .next()
            .ok_or_else(|| format!("wine parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid h_rain at line {}: {e}", line_no + 1))?;
        let s_temp_i = parts
            .next()
            .ok_or_else(|| format!("wine parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid s_temp at line {}: {e}", line_no + 1))?;
        let w_rain_i = parts
            .next()
            .ok_or_else(|| format!("wine parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid w_rain at line {}: {e}", line_no + 1))?;
        let h_temp_i = parts
            .next()
            .ok_or_else(|| format!("wine parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid h_temp at line {}: {e}", line_no + 1))?;
        // parker (optional / sometimes NA) is intentionally ignored for this benchmark spec.
        let _ = parts.next();

        let price_trimmed = price_raw.trim();
        if price_trimmed.is_empty() || price_trimmed.eq_ignore_ascii_case("NA") {
            continue;
        }
        let price_i = price_trimmed
            .parse::<f64>()
            .map_err(|e| format!("invalid price at line {}: {e}", line_no + 1))?;

        year.push(year_i);
        h_rain.push(h_rain_i);
        w_rain.push(w_rain_i);
        h_temp.push(h_temp_i);
        s_temp.push(s_temp_i);
        price.push(price_i);
    }
    if year.is_empty() {
        return Err("wine dataset is empty".to_string());
    }
    Ok((
        Array1::from_vec(year),
        Array1::from_vec(h_rain),
        Array1::from_vec(w_rain),
        Array1::from_vec(h_temp),
        Array1::from_vec(s_temp),
        Array1::from_vec(price),
    ))
}

fn load_wine_temp_vs_year_dataset() -> Result<(Array1<f64>, Array1<f64>), String> {
    let raw = include_str!("../../benchmarks/datasets/wine.csv");
    let mut year = Vec::<f64>::new();
    let mut temp = Vec::<f64>::new();
    for (line_no, line) in raw.lines().enumerate() {
        if line_no == 0 {
            continue;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split(',');
        let year_i = parts
            .next()
            .ok_or_else(|| format!("wine parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid year at line {}: {e}", line_no + 1))?;
        let _price = parts.next();
        let _h_rain = parts.next();
        let temp_i = parts
            .next()
            .ok_or_else(|| format!("wine parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid s_temp at line {}: {e}", line_no + 1))?;
        year.push(year_i);
        temp.push(temp_i);
    }
    if year.is_empty() {
        return Err("wine temp-vs-year dataset is empty".to_string());
    }
    Ok((Array1::from_vec(year), Array1::from_vec(temp)))
}

fn load_wine_price_vs_temp_dataset() -> Result<(Array1<f64>, Array1<f64>), String> {
    let raw = include_str!("../../benchmarks/datasets/wine.csv");
    let mut temp = Vec::<f64>::new();
    let mut price = Vec::<f64>::new();
    for (line_no, line) in raw.lines().enumerate() {
        if line_no == 0 {
            continue;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split(',');
        let _year = parts.next();
        let price_raw = parts
            .next()
            .ok_or_else(|| format!("wine parse error at line {}", line_no + 1))?;
        let _h_rain = parts.next();
        let temp_i = parts
            .next()
            .ok_or_else(|| format!("wine parse error at line {}", line_no + 1))?
            .parse::<f64>()
            .map_err(|e| format!("invalid s_temp at line {}: {e}", line_no + 1))?;
        let price_trimmed = price_raw.trim();
        if price_trimmed.is_empty() || price_trimmed.eq_ignore_ascii_case("NA") {
            continue;
        }
        let price_i = price_trimmed
            .parse::<f64>()
            .map_err(|e| format!("invalid price at line {}: {e}", line_no + 1))?;
        temp.push(temp_i);
        price.push(price_i);
    }
    if temp.is_empty() {
        return Err("wine price-vs-temp dataset is empty".to_string());
    }
    Ok((Array1::from_vec(temp), Array1::from_vec(price)))
}

struct CvData {
    y: Vec<f64>,
    smooth: Vec<f64>,
    linear_cols: Vec<Vec<f64>>,
    family: LikelihoodFamily,
    num_internal_knots: usize,
    double_penalty: bool,
}

fn parse_indices(path: &str) -> Result<Vec<usize>, String> {
    let raw = fs::read_to_string(path).map_err(|e| format!("failed to read index file: {e}"))?;
    let mut out = Vec::<usize>::new();
    for (line_no, line) in raw.lines().enumerate() {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        out.push(
            s.parse::<usize>()
                .map_err(|e| format!("invalid index at line {} in {}: {e}", line_no + 1, path))?,
        );
    }
    if out.is_empty() {
        return Err(format!("index file '{}' has no indices", path));
    }
    Ok(out)
}

fn header_pos(header: &[&str], name: &str) -> Result<usize, String> {
    header
        .iter()
        .position(|h| *h == name)
        .ok_or_else(|| format!("column '{name}' not found in data header"))
}

fn parse_delimited_table(path: &str) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
    let raw = fs::read_to_string(path).map_err(|e| format!("failed to read data file: {e}"))?;
    let mut lines = raw.lines();
    let header_line = lines
        .next()
        .ok_or_else(|| "data file is empty".to_string())?;
    let delim = if header_line.contains('\t') {
        '\t'
    } else if header_line.contains(';') {
        ';'
    } else {
        ','
    };
    let header = header_line
        .split(delim)
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();
    let mut rows = Vec::<Vec<String>>::new();
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        rows.push(
            line.split(delim)
                .map(|s| s.trim().to_string())
                .collect::<Vec<_>>(),
        );
    }
    Ok((header, rows))
}

fn infer_numeric_col_indices(rows: &[Vec<String>], n_cols: usize) -> Vec<usize> {
    (0..n_cols)
        .filter(|&j| {
            let mut any = false;
            for r in rows {
                if let Some(v) = r.get(j) {
                    let t = v.trim();
                    if t.is_empty() {
                        continue;
                    }
                    any = true;
                    if t.parse::<f64>().is_err() {
                        return false;
                    }
                }
            }
            any
        })
        .collect()
}

fn infer_binary_numeric_col(rows: &[Vec<String>], idx: usize) -> bool {
    let mut any = false;
    for r in rows {
        let Some(v) = r.get(idx) else { continue };
        let t = v.trim();
        if t.is_empty() {
            continue;
        }
        let Ok(x) = t.parse::<f64>() else {
            return false;
        };
        any = true;
        if (x - 0.0).abs() > 1e-12 && (x - 1.0).abs() > 1e-12 {
            return false;
        }
    }
    any
}

fn load_cv_data(
    path: &str,
    family: LikelihoodFamily,
    target_col: &str,
    smooth_col: &str,
    linear_names: &[String],
    num_internal_knots: usize,
    double_penalty: bool,
) -> Result<CvData, String> {
    let (header_s, rows_s) = parse_delimited_table(path)?;
    let header = header_s.iter().map(|s| s.as_str()).collect::<Vec<_>>();
    let smooth_idx = header_pos(&header, smooth_col)?;
    let y_idx = header_pos(&header, target_col)?;
    let linear_idx = linear_names
        .iter()
        .map(|name| header_pos(&header, name.as_str()))
        .collect::<Result<Vec<_>, _>>()?;

    let mut y = Vec::<f64>::new();
    let mut smooth = Vec::<f64>::new();
    let mut linear_cols: Vec<Vec<f64>> = linear_idx.iter().map(|_| Vec::<f64>::new()).collect();

    for (line_no, parts_s) in rows_s.iter().enumerate() {
        let parts = parts_s.iter().map(|s| s.as_str()).collect::<Vec<_>>();
        let parse_at = |idx: usize, col: &str| -> Result<f64, String> {
            parts
                .get(idx)
                .ok_or_else(|| format!("missing {col} at line {}", line_no + 2))?
                .parse::<f64>()
                .map_err(|e| format!("invalid {col} at line {}: {e}", line_no + 2))
        };
        smooth.push(parse_at(smooth_idx, smooth_col)?);
        y.push(parse_at(y_idx, "y")?);
        for (j, &idx) in linear_idx.iter().enumerate() {
            linear_cols[j].push(parse_at(idx, linear_names[j].as_str())?);
        }
    }
    if y.is_empty() {
        return Err("data has zero rows".to_string());
    }
    Ok(CvData {
        y,
        smooth,
        linear_cols,
        family,
        num_internal_knots,
        double_penalty,
    })
}

fn select_vec(v: &[f64], idx: &[usize]) -> Result<Array1<f64>, String> {
    let mut out = Vec::<f64>::with_capacity(idx.len());
    for &i in idx {
        out.push(
            *v.get(i)
                .ok_or_else(|| format!("index {i} out of bounds {}", v.len()))?,
        );
    }
    Ok(Array1::from_vec(out))
}

#[derive(Clone)]
struct SurvivalCvData {
    time: Vec<f64>,
    event: Vec<u8>,
    covariates: Vec<Vec<f64>>,
}

fn c_index_survival(time: &[f64], event: &[u8], risk: &[f64]) -> f64 {
    let n = time.len();
    let mut comparable = 0.0;
    let mut concordant = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            if event[i] == 1 && time[i] < time[j] {
                comparable += 1.0;
                if risk[i] > risk[j] {
                    concordant += 1.0;
                } else if (risk[i] - risk[j]).abs() < 1e-12 {
                    concordant += 0.5;
                }
            } else if event[j] == 1 && time[j] < time[i] {
                comparable += 1.0;
                if risk[j] > risk[i] {
                    concordant += 1.0;
                } else if (risk[i] - risk[j]).abs() < 1e-12 {
                    concordant += 0.5;
                }
            }
        }
    }
    if comparable <= 0.0 {
        0.5
    } else {
        concordant / comparable
    }
}

fn load_survival_cv_data(path: &str, time_col: &str, event_col: &str) -> Result<SurvivalCvData, String> {
    let (header_s, rows_s) = parse_delimited_table(path)?;
    let header = header_s.iter().map(|s| s.as_str()).collect::<Vec<_>>();

    let time_idx = header_pos(&header, time_col)?;
    let event_idx = header_pos(&header, event_col)?;
    let mut cov_idx = Vec::<usize>::new();
    let mut cov_names = Vec::<String>::new();
    for (j, col) in header.iter().enumerate() {
        if *col == time_col || *col == event_col {
            continue;
        }
        cov_idx.push(j);
        cov_names.push((*col).to_string());
    }
    if cov_idx.is_empty() {
        return Err("survival data has no covariate columns".to_string());
    }

    let mut time = Vec::<f64>::new();
    let mut event = Vec::<u8>::new();
    let mut covariates = Vec::<Vec<f64>>::new();

    for (line_no, parts_s) in rows_s.iter().enumerate() {
        let parts = parts_s.iter().map(|s| s.as_str()).collect::<Vec<_>>();
        let parse_at = |idx: usize, col: &str| -> Result<f64, String> {
            parts
                .get(idx)
                .ok_or_else(|| format!("missing {col} at line {}", line_no + 2))?
                .parse::<f64>()
                .map_err(|e| format!("invalid {col} at line {}: {e}", line_no + 2))
        };

        let t = parse_at(time_idx, "time")?;
        let e = parse_at(event_idx, event_col)?;
        let mut row_cov = Vec::<f64>::with_capacity(cov_idx.len());
        for (j, &idx) in cov_idx.iter().enumerate() {
            let col_name = cov_names[j].as_str();
            let mut v = parse_at(idx, col_name)?;
            if col_name == "creatinine_phosphokinase"
                || col_name == "platelets"
                || col_name == "serum_creatinine"
            {
                v = (v.max(0.0) + 1.0).ln();
            }
            row_cov.push(v);
        }
        time.push(t.max(1.0));
        event.push(if e >= 0.5 { 1 } else { 0 });
        covariates.push(row_cov);
    }

    if time.is_empty() {
        return Err("survival data has zero rows".to_string());
    }
    Ok(SurvivalCvData {
        time,
        event,
        covariates,
    })
}

fn run_survival_cv(
    data_csv: &str,
    train_idx: &[usize],
    test_idx: &[usize],
    time_col: &str,
    event_col: &str,
) -> Result<(), String> {
    let data = load_survival_cv_data(data_csv, time_col, event_col)?;
    let n = data.time.len();
    if data.covariates.is_empty() {
        return Err("no covariates found".to_string());
    }
    let m = data.covariates[0].len();
    if m == 0 {
        return Err("empty covariate row".to_string());
    }

    for &i in train_idx {
        if i >= n {
            return Err(format!("train index {i} out of bounds {n}"));
        }
    }
    for &i in test_idx {
        if i >= n {
            return Err(format!("test index {i} out of bounds {n}"));
        }
    }

    let mut means = vec![0.0; m];
    for &i in train_idx {
        for (j, mean) in means.iter_mut().enumerate().take(m) {
            *mean += data.covariates[i][j];
        }
    }
    let train_n = train_idx.len().max(1) as f64;
    for mean in means.iter_mut().take(m) {
        *mean /= train_n;
    }
    let mut stds = vec![1.0; m];
    for &i in train_idx {
        for (j, std) in stds.iter_mut().enumerate().take(m) {
            let d = data.covariates[i][j] - means[j];
            *std += d * d;
        }
    }
    for std in stds.iter_mut().take(m) {
        *std = (*std / train_n).sqrt();
        if *std <= 1e-8 || !std.is_finite() {
            *std = 1.0;
        }
    }

    let p = m + 2;
    let n_train = train_idx.len();
    let mut age_entry = Array1::<f64>::zeros(n_train);
    let mut age_exit = Array1::<f64>::zeros(n_train);
    let mut event_target = Array1::<u8>::zeros(n_train);
    let event_competing = Array1::<u8>::zeros(n_train);
    let weights = Array1::<f64>::ones(n_train);
    let mut x_entry = Array2::<f64>::zeros((n_train, p));
    let mut x_exit = Array2::<f64>::zeros((n_train, p));
    let mut x_derivative = Array2::<f64>::zeros((n_train, p));

    for (row_i, &src_i) in train_idx.iter().enumerate() {
        let t0 = 1.0;
        let t1 = data.time[src_i] + 1.0;
        age_entry[row_i] = t0;
        age_exit[row_i] = t1;
        event_target[row_i] = data.event[src_i];
        x_entry[[row_i, 0]] = 1.0;
        x_exit[[row_i, 0]] = 1.0;
        x_entry[[row_i, 1]] = t0.ln();
        x_exit[[row_i, 1]] = t1.ln();
        x_derivative[[row_i, 1]] = 1.0 / t1;
        for j in 0..m {
            let z = (data.covariates[src_i][j] - means[j]) / stds[j];
            x_entry[[row_i, j + 2]] = z;
            x_exit[[row_i, j + 2]] = z;
        }
    }

    let mut ridge = Array2::<f64>::zeros((p - 1, p - 1));
    for d in 0..(p - 1) {
        ridge[[d, d]] = 1.0;
    }
    let penalties = PenaltyBlocks::new(vec![PenaltyBlock {
        matrix: ridge,
        lambda: 1e-4,
        range: 1..p,
    }]);

    let model = working_model_from_flattened(
        penalties,
        MonotonicityPenalty {
            lambda: 5.0,
            tolerance: 1e-8,
        },
        SurvivalSpec::Net,
        RoystonParmarInputs {
            age_entry: age_entry.view(),
            age_exit: age_exit.view(),
            event_target: event_target.view(),
            event_competing: event_competing.view(),
            weights: weights.view(),
            x_entry: x_entry.view(),
            x_exit: x_exit.view(),
            x_derivative: x_derivative.view(),
        },
    )
    .map_err(|e| format!("failed to construct survival model: {e}"))?;

    let fit_start = Instant::now();
    let mut beta = Array1::<f64>::zeros(p);
    beta[0] = -3.0;
    beta[1] = 1.0;

    let mut iterations = 0usize;
    let mut last_obj = f64::INFINITY;
    for iter in 0..400 {
        let state = model
            .update_state(&beta)
            .map_err(|e| format!("survival update_state failed: {e}"))?;
        let obj = 0.5 * state.deviance + state.penalty_term;
        let grad = state.gradient;
        let grad_norm = grad.dot(&grad).sqrt();
        if grad_norm < 1e-6 || (last_obj - obj).abs() < 1e-9 {
            iterations = iter + 1;
            break;
        }
        let mut step = 0.2;
        let direction = grad.mapv(|g| -g);
        let mut accepted = false;
        for _ in 0..30 {
            let cand = &beta + &(direction.mapv(|d| step * d));
            if let Ok(cand_state) = model.update_state(&cand) {
                let cand_obj = 0.5 * cand_state.deviance + cand_state.penalty_term;
                if cand_obj.is_finite() && cand_obj < obj {
                    beta = cand;
                    last_obj = cand_obj;
                    accepted = true;
                    break;
                }
            }
            step *= 0.5;
        }
        if !accepted {
            iterations = iter + 1;
            break;
        }
        iterations = iter + 1;
    }
    let fit_sec = fit_start.elapsed().as_secs_f64();

    let pred_start = Instant::now();
    let beta_cov = beta.slice(s![2..]).to_owned();
    let mut test_time = Vec::<f64>::with_capacity(test_idx.len());
    let mut test_event = Vec::<u8>::with_capacity(test_idx.len());
    let mut test_risk = Vec::<f64>::with_capacity(test_idx.len());
    for &i in test_idx {
        let mut r = 0.0;
        for j in 0..m {
            let z = (data.covariates[i][j] - means[j]) / stds[j];
            r += z * beta_cov[j];
        }
        test_time.push(data.time[i]);
        test_event.push(data.event[i]);
        test_risk.push(r);
    }
    let pred_sec = pred_start.elapsed().as_secs_f64();

    let c_index = c_index_survival(&test_time, &test_event, &test_risk);
    println!(
        "{{\"status\":\"ok\",\"fit_sec\":{:.6},\"predict_sec\":{:.6},\"auc\":{:.6},\"brier\":null,\"rmse\":null,\"r2\":null,\"c_index\":{:.6},\"iterations\":{}}}",
        fit_sec, pred_sec, c_index, c_index, iterations
    );
    Ok(())
}

fn maybe_run_cv_mode(args: &[String]) -> Result<bool, String> {
    let Some(data_path) = parse_string_arg(args, "--data") else {
        return Ok(false);
    };
    let train_idx_path =
        parse_string_arg(args, "--train-idx").ok_or_else(|| "missing --train-idx".to_string())?;
    let test_idx_path =
        parse_string_arg(args, "--test-idx").ok_or_else(|| "missing --test-idx".to_string())?;
    let train_idx = parse_indices(&train_idx_path)?;
    let test_idx = parse_indices(&test_idx_path)?;

    let (header_s, rows_s) = parse_delimited_table(&data_path)?;
    if rows_s.is_empty() {
        return Err("data file has no rows".to_string());
    }
    let header_refs = header_s.iter().map(|s| s.as_str()).collect::<Vec<_>>();
    let header_set: HashSet<&str> = header_refs.iter().copied().collect();
    let numeric_idx = infer_numeric_col_indices(&rows_s, header_s.len());

    let mut family_s = parse_string_arg(args, "--family").unwrap_or_default();
    if family_s.is_empty() {
        if header_set.contains("time") && (header_set.contains("event") || header_set.contains("y")) {
            family_s = "survival".to_string();
        } else {
            let target_guess = if header_set.contains("y") {
                Some("y")
            } else if header_set.contains("event") {
                Some("event")
            } else {
                None
            };
            let is_binomial = target_guess
                .and_then(|name| header_pos(&header_refs, name).ok())
                .map(|idx| infer_binary_numeric_col(&rows_s, idx))
                .unwrap_or(false);
            family_s = if is_binomial { "binomial" } else { "gaussian" }.to_string();
        }
    }
    if family_s == "survival" {
        let time_col = parse_string_arg(args, "--time-col").unwrap_or_else(|| {
            if header_set.contains("time") {
                "time".to_string()
            } else {
                header_s
                    .get(*numeric_idx.first().unwrap_or(&0))
                    .cloned()
                    .unwrap_or_else(|| "time".to_string())
            }
        });
        let event_col = parse_string_arg(args, "--event-col").unwrap_or_else(|| {
            if header_set.contains("event") {
                "event".to_string()
            } else if header_set.contains("y") {
                "y".to_string()
            } else {
                "event".to_string()
            }
        });
        run_survival_cv(&data_path, &train_idx, &test_idx, &time_col, &event_col)?;
        return Ok(true);
    }
    let family = match family_s.as_str() {
        "binomial" => LikelihoodFamily::BinomialLogit,
        "gaussian" => LikelihoodFamily::GaussianIdentity,
        other => {
            return Err(format!(
                "unsupported --family '{}'; expected gaussian|binomial|survival",
                other
            ))
        }
    };
    let target_col = parse_string_arg(args, "--target-col").unwrap_or_else(|| {
        if header_set.contains("y") {
            "y".to_string()
        } else if header_set.contains("event") {
            "event".to_string()
        } else {
            header_s
                .get(*numeric_idx.last().unwrap_or(&0))
                .cloned()
                .unwrap_or_else(|| "y".to_string())
        }
    });
    let smooth_col = parse_string_arg(args, "--smooth-col").unwrap_or_else(|| {
        let preferred = [
            "s_temp",
            "x2",
            "range",
            "t",
            "pc2",
            "pulse",
            "hour",
            "axil_nodes",
            "temp",
            "year",
            "time",
            "age",
        ];
        preferred
            .iter()
            .find(|&&name| header_set.contains(name) && name != target_col.as_str())
            .map(|s| (*s).to_string())
            .or_else(|| {
                numeric_idx.iter().find_map(|&j| {
                    let n = header_s[j].as_str();
                    if n == target_col.as_str() || n == "time" || n == "event" {
                        None
                    } else {
                        Some(n.to_string())
                    }
                })
            })
            .unwrap_or_else(|| target_col.clone())
    });
    let linear_cols = {
        let explicit = parse_csv_list_arg(args, "--linear-cols");
        if !explicit.is_empty() {
            explicit
        } else {
            numeric_idx
                .iter()
                .filter_map(|&j| {
                    let n = header_s[j].as_str();
                    if n == target_col.as_str()
                        || n == smooth_col.as_str()
                        || n == "time"
                        || n == "event"
                    {
                        None
                    } else {
                        Some(n.to_string())
                    }
                })
                .collect::<Vec<_>>()
        }
    };
    let num_internal_knots = parse_arg(args, "--num-internal-knots", 7);
    let double_penalty = parse_bool_arg(args, "--double-penalty", family == LikelihoodFamily::BinomialLogit);
    let cv_data = load_cv_data(
        &data_path,
        family,
        &target_col,
        &smooth_col,
        &linear_cols,
        num_internal_knots,
        double_penalty,
    )?;

    let y_train = select_vec(&cv_data.y, &train_idx)?;
    let y_test = select_vec(&cv_data.y, &test_idx)?;
    let smooth_train = select_vec(&cv_data.smooth, &train_idx)?;
    let smooth_test = select_vec(&cv_data.smooth, &test_idx)?;

    let spec_train = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knot_spec: BSplineKnotSpec::Automatic {
            num_internal_knots: Some(cv_data.num_internal_knots),
            placement: BSplineKnotPlacement::Quantile,
        },
        double_penalty: cv_data.double_penalty,
        identifiability: BSplineIdentifiability::None,
    };
    let built_train =
        build_bspline_basis_1d(smooth_train.view(), &spec_train).map_err(|e| e.to_string())?;
    let knots = match &built_train.metadata {
        BasisMetadata::BSpline1D { knots } => knots.clone(),
        _ => return Err("unexpected basis metadata in CV mode".to_string()),
    };
    let spec_test = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knot_spec: BSplineKnotSpec::Provided(knots),
        double_penalty: cv_data.double_penalty,
        identifiability: BSplineIdentifiability::None,
    };
    let built_test =
        build_bspline_basis_1d(smooth_test.view(), &spec_test).map_err(|e| e.to_string())?;

    let q = built_train.design.ncols();
    let n_linear = cv_data.linear_cols.len();
    let p_total = 1 + n_linear + q;
    let smooth_offset = 1 + n_linear;

    let n_train = train_idx.len();
    let n_test = test_idx.len();
    let mut x_train = Array2::<f64>::zeros((n_train, p_total));
    let mut x_test = Array2::<f64>::zeros((n_test, p_total));
    for i in 0..n_train {
        x_train[[i, 0]] = 1.0;
        for j in 0..n_linear {
            x_train[[i, 1 + j]] = cv_data.linear_cols[j][train_idx[i]];
        }
        for j in 0..q {
            x_train[[i, smooth_offset + j]] = built_train.design[[i, j]];
        }
    }
    for i in 0..n_test {
        x_test[[i, 0]] = 1.0;
        for j in 0..n_linear {
            x_test[[i, 1 + j]] = cv_data.linear_cols[j][test_idx[i]];
        }
        for j in 0..q {
            x_test[[i, smooth_offset + j]] = built_test.design[[i, j]];
        }
    }

    let s_list = built_train
        .penalties
        .iter()
        .map(|s_small| {
            let mut s_full = Array2::<f64>::zeros((p_total, p_total));
            for i in 0..q {
                for j in 0..q {
                    s_full[[smooth_offset + i, smooth_offset + j]] = s_small[[i, j]];
                }
            }
            s_full
        })
        .collect::<Vec<_>>();
    let nullspace_dims = built_train
        .nullspace_dims
        .iter()
        .map(|d| d + smooth_offset)
        .collect::<Vec<_>>();

    let weights = Array1::<f64>::ones(n_train);
    let offset_train = Array1::<f64>::zeros(n_train);
    let offset_test = Array1::<f64>::zeros(n_test);
    let opts = FitOptions {
        max_iter: 200,
        tol: 1e-4,
        nullspace_dims,
    };

    let fit_start = Instant::now();
    let fit = fit_gam(
        x_train.view(),
        y_train.view(),
        weights.view(),
        offset_train.view(),
        &s_list,
        cv_data.family,
        &opts,
    )
    .map_err(|e| format!("fit_gam failed: {e}"))?;
    let fit_sec = fit_start.elapsed().as_secs_f64();

    let pred_start = Instant::now();
    let pred = predict_gam(
        x_test.view(),
        fit.beta.view(),
        offset_test.view(),
        cv_data.family,
    )
    .map_err(|e| format!("predict_gam failed: {e}"))?;
    let pred_sec = pred_start.elapsed().as_secs_f64();

    if cv_data.family == LikelihoodFamily::BinomialLogit {
        println!(
            "{{\"status\":\"ok\",\"fit_sec\":{:.6},\"predict_sec\":{:.6},\"auc\":{:.6},\"brier\":{:.6},\"rmse\":null,\"r2\":null}}",
            fit_sec,
            pred_sec,
            auc_score(y_test.view(), pred.mean.view()),
            brier_score(y_test.view(), pred.mean.view())
        );
    } else {
        println!(
            "{{\"status\":\"ok\",\"fit_sec\":{:.6},\"predict_sec\":{:.6},\"auc\":null,\"brier\":null,\"rmse\":{:.6},\"r2\":{:.6}}}",
            fit_sec,
            pred_sec,
            rmse_score(y_test.view(), pred.mean.view()),
            r2_score(y_test.view(), pred.mean.view())
        );
    }
    Ok(true)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s == "__bench_cv").unwrap_or(false) {
        let cv_args = args[1..].to_vec();
        match maybe_run_cv_mode(&cv_args) {
            Ok(true) => return,
            Ok(false) => {
                eprintln!("missing --data in internal bench mode");
                std::process::exit(2);
            }
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(2);
            }
        }
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!(
            "Usage: cargo run --release --bin cli -- [--n N --p P --seed S] [--dataset lidar|bone|prostate|wine|wine_temp_year|wine_price_temp]"
        );
        return;
    }
    let dataset = parse_string_arg(&args, "--dataset");
    let n = parse_arg(&args, "--n", 50_000);
    let p = parse_arg(&args, "--p", 50).max(3);
    let seed = parse_seed(&args, 42);

    let (x, y, s_list, nullspace_dims, family, scenario_json) = if dataset.as_deref()
        == Some("lidar")
    {
        let (range, y) = load_lidar_dataset().expect("failed to load lidar dataset");
        let n = range.len();
        let spec = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Automatic {
                num_internal_knots: Some(24),
                placement: BSplineKnotPlacement::Quantile,
            },
            double_penalty: true,
            identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
        };
        let built = build_bspline_basis_1d(range.view(), &spec)
            .expect("failed to build lidar B-spline basis");
        let q = built.design.ncols();
        let p_total = q + 1;
        let mut x = Array2::<f64>::zeros((n, p_total));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            for j in 0..q {
                x[[i, j + 1]] = built.design[[i, j]];
            }
        }
        let s_list = built
            .penalties
            .iter()
            .map(|s_small| {
                let mut s_full = Array2::<f64>::zeros((p_total, p_total));
                for i in 0..q {
                    for j in 0..q {
                        s_full[[i + 1, j + 1]] = s_small[[i, j]];
                    }
                }
                s_full
            })
            .collect::<Vec<_>>();
        let nullspace_dims = built
            .nullspace_dims
            .iter()
            .map(|d| d + 1)
            .collect::<Vec<_>>();
        let scenario_json = format!("{{\"dataset\":\"lidar\",\"n\":{},\"p\":{}}}", n, p_total);
        (
            x,
            y,
            s_list,
            nullspace_dims,
            LikelihoodFamily::GaussianIdentity,
            scenario_json,
        )
    } else if dataset.as_deref() == Some("bone") {
        let (t, y, trt_auto) = load_bone_dataset().expect("failed to load bone dataset");
        let n = t.len();
        let spec = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Automatic {
                num_internal_knots: Some(8),
                placement: BSplineKnotPlacement::Quantile,
            },
            double_penalty: true,
            identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
        };
        let built =
            build_bspline_basis_1d(t.view(), &spec).expect("failed to build bone B-spline basis");
        let q = built.design.ncols();
        // Columns: intercept + treatment indicator + smooth(time) basis.
        let p_total = q + 2;
        let mut x = Array2::<f64>::zeros((n, p_total));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            x[[i, 1]] = trt_auto[i];
            for j in 0..q {
                x[[i, j + 2]] = built.design[[i, j]];
            }
        }
        let s_list = built
            .penalties
            .iter()
            .map(|s_small| {
                let mut s_full = Array2::<f64>::zeros((p_total, p_total));
                for i in 0..q {
                    for j in 0..q {
                        s_full[[i + 2, j + 2]] = s_small[[i, j]];
                    }
                }
                s_full
            })
            .collect::<Vec<_>>();
        let nullspace_dims = built
            .nullspace_dims
            .iter()
            .map(|d| d + 2)
            .collect::<Vec<_>>();
        let scenario_json = format!("{{\"dataset\":\"bone\",\"n\":{},\"p\":{}}}", n, p_total);
        (
            x,
            y,
            s_list,
            nullspace_dims,
            LikelihoodFamily::BinomialLogit,
            scenario_json,
        )
    } else if dataset.as_deref() == Some("prostate") {
        let (pc1, pc2, y) = load_prostate_dataset().expect("failed to load prostate dataset");
        let n = pc1.len();
        let spec = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Automatic {
                num_internal_knots: Some(8),
                placement: BSplineKnotPlacement::Quantile,
            },
            double_penalty: true,
            identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
        };
        let built = build_bspline_basis_1d(pc2.view(), &spec)
            .expect("failed to build prostate B-spline basis");
        let q = built.design.ncols();
        // Columns: intercept + linear(pc1) + smooth(pc2) basis.
        let p_total = q + 2;
        let mut x = Array2::<f64>::zeros((n, p_total));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            x[[i, 1]] = pc1[i];
            for j in 0..q {
                x[[i, j + 2]] = built.design[[i, j]];
            }
        }
        let s_list = built
            .penalties
            .iter()
            .map(|s_small| {
                let mut s_full = Array2::<f64>::zeros((p_total, p_total));
                for i in 0..q {
                    for j in 0..q {
                        s_full[[i + 2, j + 2]] = s_small[[i, j]];
                    }
                }
                s_full
            })
            .collect::<Vec<_>>();
        let nullspace_dims = built
            .nullspace_dims
            .iter()
            .map(|d| d + 2)
            .collect::<Vec<_>>();
        let scenario_json = format!("{{\"dataset\":\"prostate\",\"n\":{},\"p\":{}}}", n, p_total);
        (
            x,
            y,
            s_list,
            nullspace_dims,
            LikelihoodFamily::BinomialLogit,
            scenario_json,
        )
    } else if dataset.as_deref() == Some("wine") {
        let (year, h_rain, w_rain, h_temp, s_temp, y) =
            load_wine_dataset().expect("failed to load wine dataset");
        let n = year.len();
        let spec = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Automatic {
                // Approximate mgcv/pygam k=10 basis size for fair comparison.
                num_internal_knots: Some(7),
                placement: BSplineKnotPlacement::Quantile,
            },
            // Keep nullspace unpenalized (mgcv default unless select=TRUE).
            double_penalty: false,
            identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
        };
        let built = build_bspline_basis_1d(s_temp.view(), &spec)
            .expect("failed to build wine B-spline basis");
        let q = built.design.ncols();
        // Columns: intercept + linear(year/h_rain/w_rain/h_temp) + smooth(s_temp) basis.
        let p_total = q + 5;
        let mut x = Array2::<f64>::zeros((n, p_total));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            x[[i, 1]] = year[i];
            x[[i, 2]] = h_rain[i];
            x[[i, 3]] = w_rain[i];
            x[[i, 4]] = h_temp[i];
            for j in 0..q {
                x[[i, j + 5]] = built.design[[i, j]];
            }
        }
        let s_list = built
            .penalties
            .iter()
            .map(|s_small| {
                let mut s_full = Array2::<f64>::zeros((p_total, p_total));
                for i in 0..q {
                    for j in 0..q {
                        s_full[[i + 5, j + 5]] = s_small[[i, j]];
                    }
                }
                s_full
            })
            .collect::<Vec<_>>();
        let nullspace_dims = built
            .nullspace_dims
            .iter()
            .map(|d| d + 5)
            .collect::<Vec<_>>();
        let scenario_json = format!("{{\"dataset\":\"wine\",\"n\":{},\"p\":{}}}", n, p_total);
        (
            x,
            y,
            s_list,
            nullspace_dims,
            LikelihoodFamily::GaussianIdentity,
            scenario_json,
        )
    } else if dataset.as_deref() == Some("wine_temp_year") {
        let (year, y) =
            load_wine_temp_vs_year_dataset().expect("failed to load wine_temp_year dataset");
        let n = year.len();
        let spec = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Automatic {
                num_internal_knots: Some(7),
                placement: BSplineKnotPlacement::Quantile,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
        };
        let built = build_bspline_basis_1d(year.view(), &spec)
            .expect("failed to build wine_temp_year B-spline basis");
        let q = built.design.ncols();
        let p_total = q + 1;
        let mut x = Array2::<f64>::zeros((n, p_total));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            for j in 0..q {
                x[[i, j + 1]] = built.design[[i, j]];
            }
        }
        let s_list = built
            .penalties
            .iter()
            .map(|s_small| {
                let mut s_full = Array2::<f64>::zeros((p_total, p_total));
                for i in 0..q {
                    for j in 0..q {
                        s_full[[i + 1, j + 1]] = s_small[[i, j]];
                    }
                }
                s_full
            })
            .collect::<Vec<_>>();
        let nullspace_dims = built
            .nullspace_dims
            .iter()
            .map(|d| d + 1)
            .collect::<Vec<_>>();
        let scenario_json = format!(
            "{{\"dataset\":\"wine_temp_year\",\"n\":{},\"p\":{}}}",
            n, p_total
        );
        (
            x,
            y,
            s_list,
            nullspace_dims,
            LikelihoodFamily::GaussianIdentity,
            scenario_json,
        )
    } else if dataset.as_deref() == Some("wine_price_temp") {
        let (temp, y) =
            load_wine_price_vs_temp_dataset().expect("failed to load wine_price_temp dataset");
        let n = temp.len();
        let spec = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Automatic {
                num_internal_knots: Some(7),
                placement: BSplineKnotPlacement::Quantile,
            },
            double_penalty: false,
            identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
        };
        let built = build_bspline_basis_1d(temp.view(), &spec)
            .expect("failed to build wine_price_temp B-spline basis");
        let q = built.design.ncols();
        let p_total = q + 1;
        let mut x = Array2::<f64>::zeros((n, p_total));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            for j in 0..q {
                x[[i, j + 1]] = built.design[[i, j]];
            }
        }
        let s_list = built
            .penalties
            .iter()
            .map(|s_small| {
                let mut s_full = Array2::<f64>::zeros((p_total, p_total));
                for i in 0..q {
                    for j in 0..q {
                        s_full[[i + 1, j + 1]] = s_small[[i, j]];
                    }
                }
                s_full
            })
            .collect::<Vec<_>>();
        let nullspace_dims = built
            .nullspace_dims
            .iter()
            .map(|d| d + 1)
            .collect::<Vec<_>>();
        let scenario_json = format!(
            "{{\"dataset\":\"wine_price_temp\",\"n\":{},\"p\":{}}}",
            n, p_total
        );
        (
            x,
            y,
            s_list,
            nullspace_dims,
            LikelihoodFamily::GaussianIdentity,
            scenario_json,
        )
    } else if dataset.is_none() {
        let mut rng = LcgRng::new(seed);
        let mut x_full = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            x_full[[i, 0]] = 1.0;
            for j in 1..p {
                x_full[[i, j]] = rng.normal();
            }
        }

        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            // Synthetic parity target: intercept + linear(x1) + smooth(x2).
            let x1 = x_full[[i, 1]];
            let x2 = x_full[[i, 2]];
            let eta = -0.25 + 1.1 * x1 - 0.9 * x2 + 0.2 * x2.sin();
            let pr = sigmoid(eta);
            y[i] = if rng.next_f64() < pr { 1.0 } else { 0.0 };
        }

        let x2 = x_full.column(2).to_owned();
        let spec = BSplineBasisSpec {
            degree: 3,
            penalty_order: 2,
            knot_spec: BSplineKnotSpec::Automatic {
                num_internal_knots: Some(7),
                placement: BSplineKnotPlacement::Quantile,
            },
            double_penalty: true,
            identifiability: BSplineIdentifiability::WeightedSumToZero { weights: None },
        };
        let built = build_bspline_basis_1d(x2.view(), &spec)
            .expect("failed to build synthetic B-spline basis");
        let q = built.design.ncols();
        let p_total = q + 2;
        let mut x = Array2::<f64>::zeros((n, p_total));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            x[[i, 1]] = x_full[[i, 1]];
            for j in 0..q {
                x[[i, j + 2]] = built.design[[i, j]];
            }
        }
        let s_list = built
            .penalties
            .iter()
            .map(|s_small| {
                let mut s_full = Array2::<f64>::zeros((p_total, p_total));
                for i in 0..q {
                    for j in 0..q {
                        s_full[[i + 2, j + 2]] = s_small[[i, j]];
                    }
                }
                s_full
            })
            .collect::<Vec<_>>();
        let nullspace_dims = built
            .nullspace_dims
            .iter()
            .map(|d| d + 2)
            .collect::<Vec<_>>();
        let scenario_json = format!("{{\"n\":{},\"p\":{}}}", n, p);
        (
            x,
            y,
            s_list,
            nullspace_dims,
            LikelihoodFamily::BinomialLogit,
            scenario_json,
        )
    } else {
        eprintln!(
            "unknown dataset '{}'; expected one of: lidar, bone, prostate, wine, wine_temp_year, wine_price_temp",
            dataset.unwrap_or_default()
        );
        std::process::exit(2);
    };

    let n_obs = x.nrows();
    let weights = Array1::<f64>::ones(n_obs);
    let offset = Array1::<f64>::zeros(n_obs);
    let opts = FitOptions {
        max_iter: 200,
        tol: 1e-4,
        nullspace_dims,
    };

    let fit_start = Instant::now();
    let fit = fit_gam(
        x.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &s_list,
        family,
        &opts,
    )
    .expect("fit_gam failed");
    let fit_sec = fit_start.elapsed().as_secs_f64();

    let pred_start = Instant::now();
    let pred =
        predict_gam(x.view(), fit.beta.view(), offset.view(), family).expect("predict_gam failed");
    let pred_sec = pred_start.elapsed().as_secs_f64();

    let (auc_str, brier_str, rmse_str, r2_str) = if family == LikelihoodFamily::BinomialLogit {
        (
            format!("{:.6}", auc_score(y.view(), pred.mean.view())),
            format!("{:.6}", brier_score(y.view(), pred.mean.view())),
            "null".to_string(),
            "null".to_string(),
        )
    } else {
        (
            "null".to_string(),
            "null".to_string(),
            format!("{:.6}", rmse_score(y.view(), pred.mean.view())),
            format!("{:.6}", r2_score(y.view(), pred.mean.view())),
        )
    };

    println!(
        "{{\"engine\":\"gam\",\"scenario\":{},\"fit_sec\":{:.6},\"predict_sec\":{:.6},\"auc\":{},\"brier\":{},\"rmse\":{},\"r2\":{},\"edf_total\":{:.6}}}",
        scenario_json, fit_sec, pred_sec, auc_str, brier_str, rmse_str, r2_str, fit.edf_total
    );
}

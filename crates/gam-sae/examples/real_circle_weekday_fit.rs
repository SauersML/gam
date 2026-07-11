//! Real Qwen weekday-circle fit over harvested activations.
//!
//! Inputs are simple CSV files produced by `experiments/real_circle/real_qwen_circle.py`:
//! activation PCA scores (`n x p_x`), restricted weekday probabilities
//! (`n x 7`), and weekday labels (`n` integers, Monday = 0). The current
//! SAE-manifold fitter learns a single periodic atom, checks cyclic recovery,
//! and exports per-row +1-weekday steering deltas in the activation PCA basis.

use gam_problem::RowMetric;
use gam_sae::inference::steering::steer_delta;
use gam_sae::manifold::{
    AssignmentMode, BehaviorBlock, LatentManifold, PeriodicHarmonicEvaluator, SaeAssignment,
    SaeAtomBasisKind, SaeBasisEvaluator, SaeManifoldAtom, SaeManifoldRho, SaeManifoldTerm,
    TwoBlockRemlControls,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde_json::json;
use std::f64::consts::TAU;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const WEEKDAYS: [&str; 7] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
];

fn main() -> Result<(), String> {
    let mut args = std::env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "real_circle_weekday_fit".to_string());
    let Some(acts_path) = args.next() else {
        return Err(format!(
            "usage: {program} <activations_pca.csv> <weekday_probs.csv> <labels.csv> <out_dir> [harmonics] [fit_iters]"
        ));
    };
    let Some(probs_path) = args.next() else {
        return Err(format!(
            "usage: {program} <activations_pca.csv> <weekday_probs.csv> <labels.csv> <out_dir> [harmonics] [fit_iters]"
        ));
    };
    let Some(labels_path) = args.next() else {
        return Err(format!(
            "usage: {program} <activations_pca.csv> <weekday_probs.csv> <labels.csv> <out_dir> [harmonics] [fit_iters]"
        ));
    };
    let Some(out_dir_raw) = args.next() else {
        return Err(format!(
            "usage: {program} <activations_pca.csv> <weekday_probs.csv> <labels.csv> <out_dir> [harmonics] [fit_iters]"
        ));
    };
    let harmonics = parse_optional_usize(args.next(), 3, "harmonics")?;
    let fit_iters = parse_optional_usize(args.next(), 80, "fit_iters")?;

    let acts = read_csv_matrix(Path::new(&acts_path))?;
    let probs = read_csv_matrix(Path::new(&probs_path))?;
    let labels = read_labels(Path::new(&labels_path))?;
    let out_dir = PathBuf::from(out_dir_raw);
    std::fs::create_dir_all(&out_dir)
        .map_err(|err| format!("create {}: {err}", out_dir.display()))?;

    run(acts, probs, labels, &out_dir, harmonics, fit_iters)
}

fn parse_optional_usize(
    raw: Option<String>,
    default_value: usize,
    label: &str,
) -> Result<usize, String> {
    match raw {
        Some(text) => {
            let parsed = text
                .parse::<usize>()
                .map_err(|err| format!("{label} must be a positive integer: {err}"))?;
            if parsed == 0 {
                Err(format!("{label} must be positive, got 0"))
            } else {
                Ok(parsed)
            }
        }
        None => Ok(default_value),
    }
}

fn run(
    acts: Array2<f64>,
    probs: Array2<f64>,
    labels: Vec<usize>,
    out_dir: &Path,
    harmonics: usize,
    fit_iters: usize,
) -> Result<(), String> {
    let (n, p_x) = acts.dim();
    if probs.dim() != (n, WEEKDAYS.len()) {
        return Err(format!(
            "weekday_probs.csv shape {:?} must be ({n}, {})",
            probs.dim(),
            WEEKDAYS.len()
        ));
    }
    if labels.len() != n {
        return Err(format!("labels length {} must equal n={n}", labels.len()));
    }
    for (row, &label) in labels.iter().enumerate() {
        if label >= WEEKDAYS.len() {
            return Err(format!(
                "row {row} label {label} outside 0..{}",
                WEEKDAYS.len()
            ));
        }
    }

    let proxy_t = plane_angle_proxy(acts.view())?;
    let true_t = labels_to_unit_coords(&labels);
    let proxy_corr = circular_corr(proxy_t.view(), true_t.view())?;

    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(2 * harmonics + 1)?);
    let coords = proxy_t.clone().insert_axis(ndarray::Axis(1));
    let (mut term, mut rho) = circle_term(&evaluator, &coords, p_x, probs.view())?;
    term.set_guards_enabled(false);
    let report = term.run_two_block_reml_fit(
        acts.view(),
        &mut rho,
        None,
        TwoBlockRemlControls {
            max_sweeps: 30,
            inner_max_iter: fit_iters,
            step_size: 1.0,
            ridge_ext_coord: 1e-6,
            ridge_beta: 1e-6,
            log_lambda_tol: 1e-3,
        },
    )?;

    let raw_t = term.assignment.coords[0]
        .as_matrix()
        .column(0)
        .mapv(|v| v.rem_euclid(1.0));
    let raw_corr = circular_corr(raw_t.view(), true_t.view())?;
    let orientation = if raw_corr >= 0.0 { 1.0 } else { -1.0 };
    let aligned_t = align_coordinate(raw_t.view(), true_t.view(), orientation);
    let aligned_corr = circular_corr(aligned_t.view(), true_t.view())?;

    let augmented = term
        .behavior_block()
        .ok_or_else(|| "behavior block missing after fit".to_string())?
        .augmented_target(acts.view())?;
    let fitted = term.try_fitted_for_rho(&rho)?;
    let activation_ev =
        explained_variance(augmented.slice(s![.., ..p_x]), fitted.slice(s![.., ..p_x]))?;
    let behavior_ev =
        explained_variance(augmented.slice(s![.., p_x..]), fitted.slice(s![.., p_x..]))?;

    let step = orientation / WEEKDAYS.len() as f64;
    let behavior_dim = term
        .behavior_block()
        .ok_or_else(|| "behavior block missing before steering".to_string())?
        .behavior_dim();
    let metric = RowMetric::euclidean(n, p_x + behavior_dim)?;
    let mut deltas = Array2::<f64>::zeros((n, p_x));
    let mut steer_off_norm_sum = 0.0;
    for row in 0..n {
        let from = raw_t[row];
        let to = (from + step).rem_euclid(1.0);
        let plan = steer_delta(&term, &metric, 0, 0, 1.0, &[from], &[to])?;
        steer_off_norm_sum += plan.off_manifold_norm;
        for col in 0..p_x {
            deltas[[row, col]] = plan.delta[col];
        }
    }
    let mean_off_manifold_norm = steer_off_norm_sum / n as f64;

    write_coords(
        out_dir,
        &labels,
        raw_t.view(),
        aligned_t.view(),
        proxy_t.view(),
    )?;
    write_matrix_csv(&out_dir.join("steering_delta_pca.csv"), deltas.view())?;
    write_svg_chart(
        &out_dir.join("weekday_circle_chart.svg"),
        &labels,
        raw_t.view(),
        aligned_t.view(),
    )?;

    let summary = json!({
        "n": n,
        "activation_pca_dim": p_x,
        "harmonics": harmonics,
        "fit_iters": fit_iters,
        "converged": report.converged,
        "sweeps": report.sweeps,
        "log_lambda_y": report.log_lambda_y,
        "lambda_y": report.log_lambda_y.exp(),
        "lambda_identifiable": report.lambda_identifiable,
        "activation_ev": activation_ev,
        "behavior_ev": behavior_ev,
        "raw_circular_correlation": raw_corr,
        "aligned_circular_correlation": aligned_corr,
        "proxy_circular_correlation": proxy_corr,
        "orientation": if orientation > 0.0 { "increasing" } else { "decreasing" },
        "semantic_step_in_raw_t": step,
        "mean_steer_off_manifold_norm": mean_off_manifold_norm,
        "chart": out_dir.join("weekday_circle_chart.svg").display().to_string(),
        "coords": out_dir.join("coords.csv").display().to_string(),
        "steering_delta_pca": out_dir.join("steering_delta_pca.csv").display().to_string()
    });
    std::fs::write(
        out_dir.join("fit_results.json"),
        serde_json::to_string_pretty(&summary).map_err(|err| err.to_string())?,
    )
    .map_err(|err| format!("write fit_results.json: {err}"))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).map_err(|err| err.to_string())?
    );
    Ok(())
}

fn circle_term(
    evaluator: &Arc<PeriodicHarmonicEvaluator>,
    coords: &Array2<f64>,
    p_x: usize,
    probs: ArrayView2<'_, f64>,
) -> Result<(SaeManifoldTerm, SaeManifoldRho), String> {
    let (phi, jet) = evaluator.evaluate(coords.view())?;
    let behavior = BehaviorBlock::fit(probs, p_x, 0.0)?;
    let p_tot = p_x + behavior.behavior_dim();
    let basis_size = evaluator.basis_size();
    let atom = SaeManifoldAtom::new_with_provided_function_gram(
        "qwen_weekday_circle",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        Array2::<f64>::zeros((basis_size, p_tot)),
        Array2::<f64>::eye(basis_size),
    )?
    .with_basis_second_jet(evaluator.clone());
    let logits = Array2::<f64>::from_elem((coords.nrows(), 1), 6.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords.clone()],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )?;
    let mut term = SaeManifoldTerm::new(vec![atom], assignment)?;
    term.set_behavior_block(behavior)?;
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::<f64>::zeros(1)]);
    Ok((term, rho))
}

trait PeriodicBasisSize {
    fn basis_size(&self) -> usize;
}

impl PeriodicBasisSize for PeriodicHarmonicEvaluator {
    fn basis_size(&self) -> usize {
        let coords = Array2::<f64>::zeros((1, 1));
        self.evaluate(coords.view())
            .expect("PeriodicHarmonicEvaluator basis evaluation")
            .0
            .ncols()
    }
}

fn plane_angle_proxy(acts: ArrayView2<'_, f64>) -> Result<Array1<f64>, String> {
    if acts.ncols() < 2 {
        return Err("activation PCA scores need at least two columns".to_string());
    }
    let n = acts.nrows();
    let mean0 = acts.column(0).sum() / n as f64;
    let mean1 = acts.column(1).sum() / n as f64;
    let mut out = Array1::<f64>::zeros(n);
    for row in 0..n {
        let x = acts[[row, 0]] - mean0;
        let y = acts[[row, 1]] - mean1;
        out[row] = (y.atan2(x) / TAU).rem_euclid(1.0);
    }
    Ok(out)
}

fn labels_to_unit_coords(labels: &[usize]) -> Array1<f64> {
    Array1::from_iter(
        labels
            .iter()
            .map(|&label| label as f64 / WEEKDAYS.len() as f64),
    )
}

fn circular_corr(a_unit: ArrayView1<'_, f64>, b_unit: ArrayView1<'_, f64>) -> Result<f64, String> {
    if a_unit.len() != b_unit.len() {
        return Err(format!(
            "circular_corr length mismatch: {} vs {}",
            a_unit.len(),
            b_unit.len()
        ));
    }
    let n = a_unit.len();
    if n == 0 {
        return Err("circular_corr requires at least one row".to_string());
    }
    let mean_a = circular_mean(a_unit);
    let mean_b = circular_mean(b_unit);
    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;
    for row in 0..n {
        let sa = (TAU * (a_unit[row] - mean_a)).sin();
        let sb = (TAU * (b_unit[row] - mean_b)).sin();
        num += sa * sb;
        den_a += sa * sa;
        den_b += sb * sb;
    }
    let den = (den_a * den_b).sqrt();
    if den <= f64::MIN_POSITIVE {
        return Err("circular_corr denominator is zero".to_string());
    }
    Ok(num / den)
}

fn circular_mean(values: ArrayView1<'_, f64>) -> f64 {
    let mut sx = 0.0;
    let mut cx = 0.0;
    for &value in values {
        let angle = TAU * value;
        sx += angle.sin();
        cx += angle.cos();
    }
    (sx.atan2(cx) / TAU).rem_euclid(1.0)
}

fn align_coordinate(
    raw_t: ArrayView1<'_, f64>,
    true_t: ArrayView1<'_, f64>,
    orientation: f64,
) -> Array1<f64> {
    let mut best_shift = 0.0;
    let mut best_loss = f64::INFINITY;
    for grid in 0..2048 {
        let shift = grid as f64 / 2048.0;
        let mut loss = 0.0;
        for row in 0..raw_t.len() {
            let aligned = (orientation * raw_t[row] + shift).rem_euclid(1.0);
            let diff = circular_distance(aligned, true_t[row]);
            loss += diff * diff;
        }
        if loss < best_loss {
            best_loss = loss;
            best_shift = shift;
        }
    }
    raw_t.mapv(|value| (orientation * value + best_shift).rem_euclid(1.0))
}

fn circular_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(1.0);
    d.min(1.0 - d)
}

fn explained_variance(
    target: ArrayView2<'_, f64>,
    fitted: ArrayView2<'_, f64>,
) -> Result<f64, String> {
    if target.dim() != fitted.dim() {
        return Err(format!(
            "explained_variance shape mismatch: {:?} vs {:?}",
            target.dim(),
            fitted.dim()
        ));
    }
    let (n, p) = target.dim();
    let mut sst = 0.0;
    let mut ssr = 0.0;
    for col in 0..p {
        let mean = target.column(col).sum() / n as f64;
        for row in 0..n {
            let centered = target[[row, col]] - mean;
            let residual = target[[row, col]] - fitted[[row, col]];
            sst += centered * centered;
            ssr += residual * residual;
        }
    }
    if sst <= f64::MIN_POSITIVE {
        return Err("explained_variance target has no centered variance".to_string());
    }
    Ok(1.0 - ssr / sst)
}

fn read_csv_matrix(path: &Path) -> Result<Array2<f64>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut row = Vec::new();
        for token in trimmed.split(',') {
            let value = token
                .trim()
                .parse::<f64>()
                .map_err(|err| format!("{}:{} parse float: {err}", path.display(), lineno + 1))?;
            row.push(value);
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err(format!("{} has no rows", path.display()));
    }
    let p = rows[0].len();
    if p == 0 {
        return Err(format!("{} has zero columns", path.display()));
    }
    let mut out = Array2::<f64>::zeros((rows.len(), p));
    for (row_idx, row) in rows.iter().enumerate() {
        if row.len() != p {
            return Err(format!(
                "{} row {} has {} columns, expected {p}",
                path.display(),
                row_idx + 1,
                row.len()
            ));
        }
        for (col, &value) in row.iter().enumerate() {
            out[[row_idx, col]] = value;
        }
    }
    Ok(out)
}

fn read_labels(path: &Path) -> Result<Vec<usize>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let mut labels = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = trimmed
            .parse::<usize>()
            .map_err(|err| format!("{}:{} parse label: {err}", path.display(), lineno + 1))?;
        labels.push(value);
    }
    Ok(labels)
}

fn write_coords(
    out_dir: &Path,
    labels: &[usize],
    raw_t: ArrayView1<'_, f64>,
    aligned_t: ArrayView1<'_, f64>,
    proxy_t: ArrayView1<'_, f64>,
) -> Result<(), String> {
    let mut file = std::fs::File::create(out_dir.join("coords.csv"))
        .map_err(|err| format!("create coords.csv: {err}"))?;
    writeln!(file, "row,label,weekday,raw_t,aligned_t,proxy_t")
        .map_err(|err| format!("write coords.csv: {err}"))?;
    for row in 0..labels.len() {
        writeln!(
            file,
            "{row},{},{},{:.12},{:.12},{:.12}",
            labels[row], WEEKDAYS[labels[row]], raw_t[row], aligned_t[row], proxy_t[row]
        )
        .map_err(|err| format!("write coords.csv: {err}"))?;
    }
    Ok(())
}

fn write_matrix_csv(path: &Path, matrix: ArrayView2<'_, f64>) -> Result<(), String> {
    let mut file =
        std::fs::File::create(path).map_err(|err| format!("create {}: {err}", path.display()))?;
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            if col > 0 {
                write!(file, ",").map_err(|err| format!("write {}: {err}", path.display()))?;
            }
            write!(file, "{:.12}", matrix[[row, col]])
                .map_err(|err| format!("write {}: {err}", path.display()))?;
        }
        writeln!(file).map_err(|err| format!("write {}: {err}", path.display()))?;
    }
    Ok(())
}

fn write_svg_chart(
    path: &Path,
    labels: &[usize],
    raw_t: ArrayView1<'_, f64>,
    aligned_t: ArrayView1<'_, f64>,
) -> Result<(), String> {
    let colors = [
        "#2f6fbb", "#ca4d36", "#2f8f4e", "#8b55b5", "#b98222", "#168a92", "#6f6f6f",
    ];
    let mut svg = String::new();
    svg.push_str(r##"<svg xmlns="http://www.w3.org/2000/svg" width="960" height="430" viewBox="0 0 960 430">"##);
    svg.push_str(r##"<rect width="960" height="430" fill="#ffffff"/>"##);
    svg.push_str(r##"<text x="40" y="32" font-family="sans-serif" font-size="18" fill="#1f2933">Qwen weekday circle: fitted coordinate recovery</text>"##);
    svg.push_str(
        r##"<line x1="70" y1="360" x2="450" y2="360" stroke="#20242a" stroke-width="1"/>"##,
    );
    svg.push_str(r##"<line x1="70" y1="70" x2="70" y2="360" stroke="#20242a" stroke-width="1"/>"##);
    svg.push_str(r##"<text x="185" y="405" font-family="sans-serif" font-size="13" fill="#20242a">true weekday index</text>"##);
    svg.push_str(r##"<text x="14" y="210" transform="rotate(-90 14 210)" font-family="sans-serif" font-size="13" fill="#20242a">aligned fitted t × 7</text>"##);
    for tick in 0..=6 {
        let x = 70.0 + tick as f64 * (380.0 / 6.0);
        let y = 360.0 - tick as f64 * (290.0 / 6.0);
        svg.push_str(&format!(
            r##"<line x1="{x:.1}" y1="360" x2="{x:.1}" y2="365" stroke="#20242a"/><text x="{:.1}" y="382" font-family="sans-serif" font-size="11" fill="#20242a">{tick}</text>"##,
            x - 3.0
        ));
        svg.push_str(&format!(
            r##"<line x1="65" y1="{y:.1}" x2="70" y2="{y:.1}" stroke="#20242a"/><text x="45" y="{:.1}" font-family="sans-serif" font-size="11" fill="#20242a">{tick}</text>"##,
            y + 4.0
        ));
    }
    for row in 0..labels.len() {
        let jitter = ((row / WEEKDAYS.len()) as f64 - 4.5) * 1.7;
        let x = 70.0 + labels[row] as f64 * (380.0 / 6.0) + jitter;
        let y = 360.0 - (aligned_t[row] * 7.0) * (290.0 / 6.0);
        svg.push_str(&format!(
            r##"<circle cx="{x:.1}" cy="{y:.1}" r="4.0" fill="{}" fill-opacity="0.82"/>"##,
            colors[labels[row]]
        ));
    }
    svg.push_str(
        r##"<circle cx="705" cy="215" r="125" fill="none" stroke="#20242a" stroke-width="1"/>"##,
    );
    svg.push_str(r##"<text x="625" y="388" font-family="sans-serif" font-size="13" fill="#20242a">recovered circle by raw t</text>"##);
    for row in 0..labels.len() {
        let angle = TAU * raw_t[row];
        let x = 705.0 + 125.0 * angle.cos();
        let y = 215.0 - 125.0 * angle.sin();
        svg.push_str(&format!(
            r##"<circle cx="{x:.1}" cy="{y:.1}" r="4.0" fill="{}" fill-opacity="0.82"/>"##,
            colors[labels[row]]
        ));
    }
    for (idx, name) in WEEKDAYS.iter().enumerate() {
        let y = 62 + idx * 20;
        svg.push_str(&format!(
            r##"<circle cx="505" cy="{y}" r="5" fill="{}"/><text x="518" y="{}" font-family="sans-serif" font-size="12" fill="#20242a">{}</text>"##,
            colors[idx],
            y + 4,
            name
        ));
    }
    svg.push_str("</svg>\n");
    std::fs::write(path, svg).map_err(|err| format!("write {}: {err}", path.display()))
}

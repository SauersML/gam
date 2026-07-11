//! Cross-model universality with fitted real circle charts.
//!
//! Usage:
//!
//! ```text
//! cargo run -p gam-sae --example cross_model_real_chart -- \
//!   <matched-acts-a.npy> <matched-acts-b.npy> [max_rows] [harmonics] [outer_iters] [inner_iters]
//! ```
//!
//! The two activation arrays must already be row-aligned on the same token set.
//! Each model is reduced to post-leading-PC PCA scores, then a K=1 periodic atom
//! is actually fit in that activation geometry. The transported coordinate is
//! the fitted chart coordinate, not a PCA plane angle.

use faer::Side;
use gam_linalg::faer_ndarray::{FaerCholesky, FaerSvd, fast_atb};
use gam_problem::SeedConfig;
use gam_sae::assignment::{AssignmentMode, SaeAssignment};
use gam_sae::basis::{PeriodicHarmonicEvaluator, SaeBasisSecondJet};
use gam_sae::hybrid_split::build_hybrid_split_report;
use gam_sae::identifiability::thin_svd_scores;
use gam_sae::inference::cross_model_transport::{
    ModelCoordinate, UniversalityVerdict, fit_cross_model_transport,
};
use gam_sae::inference::layer_transport::ChartTopology;
use gam_sae::manifold::{
    LatentManifold, SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldOuterObjective, SaeManifoldRho,
    SaeManifoldTerm,
};
use gam_solve::rho_optimizer::OuterProblem;
use ndarray::{Array1, Array2, ArrayView2, s};
use std::f64::consts::TAU;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_MAX_ROWS: usize = 30_000;
const DEFAULT_HARMONICS: usize = 1;
const DEFAULT_OUTER_ITERS: usize = 12;
const DEFAULT_INNER_ITERS: usize = 8;
const POST_PEEL_DIM: usize = 8;

#[derive(Debug)]
struct ChartFit {
    label: &'static str,
    n_full: usize,
    p_raw: usize,
    n_used: usize,
    post_peel_dim: usize,
    top_pc_variance_fraction: f64,
    top3_linear_ev: f64,
    top5_linear_ev: f64,
    curved_ev: f64,
    full_reconstruction_ev: f64,
    topm_linear_ev: f64,
    curved_vs_envelope_ratio: f64,
    converged: bool,
    outer_iterations: usize,
    final_grad_norm: Option<f64>,
    fit_wall_seconds: f64,
    criterion_calls: usize,
    infeasible_total: usize,
    coords_radians: Array1<f64>,
}

fn main() -> ExitCode {
    let mut args = std::env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "cross_model_real_chart".to_string());
    let Some(path_a) = args.next() else {
        println!(
            "usage: {program} <matched-acts-a.npy> <matched-acts-b.npy> [max_rows] [harmonics] [outer_iters] [inner_iters]"
        );
        return ExitCode::from(2);
    };
    let Some(path_b) = args.next() else {
        println!(
            "usage: {program} <matched-acts-a.npy> <matched-acts-b.npy> [max_rows] [harmonics] [outer_iters] [inner_iters]"
        );
        return ExitCode::from(2);
    };
    let max_rows = match parse_optional_usize(args.next(), DEFAULT_MAX_ROWS, "max_rows") {
        Ok(v) => v,
        Err(err) => {
            println!("{err}");
            return ExitCode::from(2);
        }
    };
    let harmonics = match parse_optional_usize(args.next(), DEFAULT_HARMONICS, "harmonics") {
        Ok(v) => v,
        Err(err) => {
            println!("{err}");
            return ExitCode::from(2);
        }
    };
    let outer_iters = match parse_optional_usize(args.next(), DEFAULT_OUTER_ITERS, "outer_iters") {
        Ok(v) => v,
        Err(err) => {
            println!("{err}");
            return ExitCode::from(2);
        }
    };
    let inner_iters = match parse_optional_usize(args.next(), DEFAULT_INNER_ITERS, "inner_iters") {
        Ok(v) => v,
        Err(err) => {
            println!("{err}");
            return ExitCode::from(2);
        }
    };

    match run(
        Path::new(&path_a),
        Path::new(&path_b),
        max_rows,
        harmonics,
        outer_iters,
        inner_iters,
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            println!("[cross_model_real_chart] error: {err}");
            ExitCode::FAILURE
        }
    }
}

fn parse_optional_usize(
    raw: Option<String>,
    default_value: usize,
    label: &str,
) -> Result<usize, String> {
    match raw {
        Some(text) => match text.parse::<usize>() {
            Ok(v) if v > 0 => Ok(v),
            Ok(v) => Err(format!("{label} must be positive, got {v}")),
            Err(err) => Err(format!("{label} must be a positive integer: {err}")),
        },
        None => Ok(default_value),
    }
}

fn run(
    path_a: &Path,
    path_b: &Path,
    max_rows: usize,
    harmonics: usize,
    outer_iters: usize,
    inner_iters: usize,
) -> Result<(), String> {
    let started = Instant::now();
    let a = fit_real_chart(
        "Qwen3-8B L18",
        path_a,
        max_rows,
        harmonics,
        outer_iters,
        inner_iters,
    )?;
    let b = fit_real_chart(
        "Qwen3.6-35B-A3B L20",
        path_b,
        max_rows,
        harmonics,
        outer_iters,
        inner_iters,
    )?;
    let n = a.coords_radians.len().min(b.coords_radians.len());
    if n < 16 {
        return Err(format!(
            "need at least 16 paired fitted coordinates, got {n}"
        ));
    }
    let coord_a = ModelCoordinate::new(
        a.label,
        a.coords_radians.slice(s![0..n]).to_owned(),
        ChartTopology::Circle,
    );
    let coord_b = ModelCoordinate::new(
        b.label,
        b.coords_radians.slice(s![0..n]).to_owned(),
        ChartTopology::Circle,
    );
    let report = fit_cross_model_transport(&coord_a, &coord_b)?;

    println!("cross_model_real_chart_result");
    print_chart_fit(&a);
    print_chart_fit(&b);
    println!("transport_n_obs={}", report.n_obs);
    if let Some(circle) = &report.circle {
        println!("transport_rigid_class={:?}", circle.class);
        println!("transport_winding={}", circle.winding);
        println!("transport_phase_rad={:.9}", circle.phase);
        println!("transport_phase_degrees={:.9}", circle.phase_degrees());
        println!("transport_o2_defect={:.9}", circle.defect);
        println!(
            "transport_gauge_defect_scale={:.9}",
            report.gauge_defect_scale
        );
    }
    println!("transport_degree={:?}", report.fit.degree);
    println!(
        "transport_degree_concentration={:?}",
        report.fit.degree_concentration
    );
    println!(
        "transport_topology_preserved={}",
        report.fit.topology_preserved
    );
    println!(
        "transport_min_directional_derivative={:.9}",
        report.fit.min_directional_derivative
    );
    println!(
        "transport_isometry_defect={:.9}",
        report.fit.isometry_defect
    );
    println!(
        "transport_isometry_defect_se={:.9}",
        report.fit.isometry_defect_se
    );
    println!("transport_residual_rms={:.9}", report.fit.residual_rms);
    println!("transport_edf={:.9}", report.fit.edf);
    println!(
        "transport_smoothing_lambda={:.9}",
        report.fit.smoothing_lambda
    );
    println!("transport_noise_variance={:.9}", report.fit.noise_variance);
    println!("transport_verdict={}", report.verdict.label());
    println!(
        "transport_universality={}",
        universality_sentence(report.verdict)
    );
    println!("wall_seconds={:.3}", started.elapsed().as_secs_f64());
    Ok(())
}

fn print_chart_fit(fit: &ChartFit) {
    let prefix = if fit.label.contains("8B") {
        "model_a"
    } else {
        "model_b"
    };
    println!("{prefix}_label={}", fit.label);
    println!("{prefix}_source_rows_full={}", fit.n_full);
    println!("{prefix}_source_dim={}", fit.p_raw);
    println!("{prefix}_used_rows={}", fit.n_used);
    println!("{prefix}_post_peel_dim={}", fit.post_peel_dim);
    println!(
        "{prefix}_top_pc_variance_fraction={:.9}",
        fit.top_pc_variance_fraction
    );
    println!("{prefix}_top3_linear_ev={:.9}", fit.top3_linear_ev);
    println!("{prefix}_top5_linear_ev={:.9}", fit.top5_linear_ev);
    println!("{prefix}_curved_ev={:.9}", fit.curved_ev);
    println!(
        "{prefix}_full_reconstruction_ev={:.9}",
        fit.full_reconstruction_ev
    );
    println!("{prefix}_topm_linear_ev={:.9}", fit.topm_linear_ev);
    println!(
        "{prefix}_curved_vs_envelope_ratio={:.9}",
        fit.curved_vs_envelope_ratio
    );
    println!("{prefix}_converged={}", fit.converged);
    println!("{prefix}_outer_iterations={}", fit.outer_iterations);
    println!(
        "{prefix}_final_grad_norm={}",
        fit.final_grad_norm
            .map(|g| format!("{g:.9}"))
            .unwrap_or_else(|| "nan".to_string())
    );
    println!("{prefix}_fit_wall_seconds={:.3}", fit.fit_wall_seconds);
    println!("{prefix}_criterion_calls={}", fit.criterion_calls);
    println!("{prefix}_infeasible_total={}", fit.infeasible_total);
}

fn universality_sentence(verdict: UniversalityVerdict) -> &'static str {
    match verdict {
        UniversalityVerdict::ConsistentWithSharedFeatureWithinNoise => {
            "same fitted feature manifold is consistent across models within noise/gauge"
        }
        UniversalityVerdict::SharedFeatureWithMeasuredReparameterization => {
            "same fitted circle is present only up to a measured non-isometric reparameterization"
        }
        UniversalityVerdict::NotSharedFeature => {
            "same fitted feature manifold is not supported by the real-chart transport"
        }
    }
}

fn fit_real_chart(
    label: &'static str,
    path: &Path,
    max_rows: usize,
    harmonics: usize,
    outer_iters: usize,
    inner_iters: usize,
) -> Result<ChartFit, String> {
    let fit_started = Instant::now();
    let (n_full, p_raw, raw) = read_npy_subsample_f64(path, max_rows)?;
    let total_centered = centered_sst(raw.view());
    let scores = thin_svd_scores(raw.view(), POST_PEEL_DIM + 1)?;
    let top_pc_variance_fraction = score_energy(scores.view(), 0)? / total_centered.max(1.0e-30);
    let post_peel = scores.slice(s![.., 1..POST_PEEL_DIM + 1]).to_owned();
    let top3_linear_ev = top_m_linear_ev(post_peel.view(), 3)?;
    let top5_linear_ev = top_m_linear_ev(post_peel.view(), 5)?;

    let (term, seed_dispersion, basis_size) = periodic_k1_term(post_peel.view(), harmonics, label)?;
    let mode = AssignmentMode::ibp_map(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![Array1::zeros(1)])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)?;
    let seed = init_rho.to_flat();
    let n_params = seed.len();
    let mut objective = SaeManifoldOuterObjective::new(
        term,
        post_peel.clone(),
        None,
        init_rho,
        inner_iters,
        0.04,
        1.0e-6,
        1.0e-6,
    );
    let result = OuterProblem::new(n_params)
        .with_initial_rho(seed)
        .with_max_iter(outer_iters)
        .with_seed_config(SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
        .run(&mut objective, label)
        .map_err(|err| format!("{label}: outer fit failed: {err}"))?;
    objective
        .certify_outer_result(&result)
        .map_err(|err| format!("{label}: outer fit certificate rejected: {err}"))?;
    let telemetry = objective.probe_telemetry();
    let fitted = objective.into_fitted().expect("outer fit was evaluated");
    let final_term = fitted.term;
    let fitted_matrix = final_term.fitted();
    let full_reconstruction_ev = reconstruction_ev(post_peel.view(), fitted_matrix.view())?;
    let report = envelope_report(&final_term, post_peel.view(), fitted.loss.data_fit)?;
    let verdict = report
        .verdicts
        .first()
        .ok_or_else(|| format!("{label}: hybrid envelope report returned no verdicts"))?;
    let curved_ev = verdict
        .curved_ev
        .ok_or_else(|| format!("{label}: hybrid envelope report omitted curved_ev"))?;
    let topm_linear_ev = verdict
        .topm_linear_ev
        .ok_or_else(|| format!("{label}: hybrid envelope report omitted topm_linear_ev"))?;
    let curved_vs_envelope_ratio = verdict.curved_vs_envelope_ratio.ok_or_else(|| {
        format!("{label}: hybrid envelope report omitted curved_vs_envelope_ratio")
    })?;
    let coords_radians = final_term.assignment.coords[0]
        .as_matrix()
        .column(0)
        .mapv(|t| (t * TAU).rem_euclid(TAU));

    if basis_size != 1 + 2 * harmonics {
        return Err(format!(
            "{label}: basis width {basis_size} disagrees with harmonics {harmonics}"
        ));
    }

    Ok(ChartFit {
        label,
        n_full,
        p_raw,
        n_used: post_peel.nrows(),
        post_peel_dim: post_peel.ncols(),
        top_pc_variance_fraction,
        top3_linear_ev,
        top5_linear_ev,
        curved_ev,
        full_reconstruction_ev,
        topm_linear_ev,
        curved_vs_envelope_ratio,
        converged: result.converged,
        outer_iterations: result.iterations,
        final_grad_norm: result.final_grad_norm,
        fit_wall_seconds: fit_started.elapsed().as_secs_f64(),
        criterion_calls: telemetry.criterion_calls,
        infeasible_total: telemetry.infeasible_total(),
        coords_radians,
    })
}

fn periodic_k1_term(
    z: ArrayView2<'_, f64>,
    harmonics: usize,
    label: &str,
) -> Result<(SaeManifoldTerm, f64, usize), String> {
    let n = z.nrows();
    let p = z.ncols();
    let dim = 1usize;
    let num_basis = 1 + 2 * harmonics;
    let evaluator: Arc<dyn SaeBasisSecondJet> =
        Arc::new(PeriodicHarmonicEvaluator::new(num_basis)?);
    let basis_kinds = vec![SaeAtomBasisKind::Periodic];
    let atom_dims = vec![dim];
    let seed_coords = gam_sae::manifold::sae_pca_seed_initial_coords(z, &basis_kinds, &atom_dims)?;
    let coords = seed_coords.slice(s![0, .., 0..dim]).to_owned();
    let (phi, jet) = evaluator.evaluate(coords.view())?;
    let decoder = ridge_decoder(phi.view(), z, 1.0e-8)?;
    let fitted = phi.dot(&decoder);
    let mut rss = 0.0_f64;
    for row in 0..n {
        for col in 0..p {
            let residual = z[[row, col]] - fitted[[row, col]];
            rss += residual * residual;
        }
    }
    let seed_dispersion = (rss / ((n * p) as f64)).max(1.0e-12);
    let atom = SaeManifoldAtom::new(
        label,
        SaeAtomBasisKind::Periodic,
        dim,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(num_basis),
    )?
    .with_basis_evaluator(evaluator);
    let logits = Array2::<f64>::from_elem((n, 1), 6.0);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::ibp_map(1.0, 1.0, false),
    )?;
    let term = SaeManifoldTerm::new(vec![atom], assignment)?;
    Ok((term, seed_dispersion, num_basis))
}

fn ridge_decoder(
    phi: ArrayView2<'_, f64>,
    z: ArrayView2<'_, f64>,
    ridge: f64,
) -> Result<Array2<f64>, String> {
    let mut xtx = fast_atb(&phi.to_owned(), &phi.to_owned());
    for i in 0..xtx.nrows() {
        xtx[[i, i]] += ridge;
    }
    let xtz = fast_atb(&phi.to_owned(), &z.to_owned());
    let chol = xtx
        .cholesky(Side::Lower)
        .map_err(|err| format!("ridge decoder Cholesky failed: {err}"))?;
    Ok(chol.solve_mat(&xtz))
}

fn envelope_report(
    term: &SaeManifoldTerm,
    target: ArrayView2<'_, f64>,
    data_fit: f64,
) -> Result<gam_sae::hybrid_split::SaeHybridSplitReport, String> {
    let assignments = term.assignment.assignments();
    let coords0 = term.assignment.coords[0].as_matrix().column(0).to_owned();
    let mut decoded0 = Array2::<f64>::zeros((term.n_obs(), term.output_dim()));
    for row in 0..term.n_obs() {
        let decoded = term.atoms[0].decoded_row(row);
        for col in 0..term.output_dim() {
            decoded0[[row, col]] = decoded[col];
        }
    }
    let total_centered_variance = centered_sst(target);
    let dispersion = (2.0 * data_fit / ((target.nrows() * target.ncols()) as f64)).max(1.0e-12);
    build_hybrid_split_report(
        &term.atoms,
        0..1,
        |atom_idx| {
            if atom_idx == 0 {
                coords0.clone()
            } else {
                Array1::<f64>::zeros(term.n_obs())
            }
        },
        |atom_idx| {
            if atom_idx == 0 {
                assignments.column(0).to_owned()
            } else {
                Array1::<f64>::zeros(term.n_obs())
            }
        },
        |atom_idx| {
            if atom_idx == 0 {
                decoded0.clone()
            } else {
                Array2::<f64>::zeros((term.n_obs(), term.output_dim()))
            }
        },
        |atom_idx| {
            if atom_idx == 0 {
                target.to_owned()
            } else {
                Array2::<f64>::zeros((term.n_obs(), term.output_dim()))
            }
        },
        |atom_idx| {
            if atom_idx == 0 {
                LatentManifold::Circle { period: 1.0 }
            } else {
                LatentManifold::Euclidean
            }
        },
        |_atom_idx| Some(0.0),
        total_centered_variance,
        term.n_obs(),
        dispersion,
    )?
    .ok_or_else(|| "hybrid envelope report had no eligible d=1 atom".to_string())
}

fn reconstruction_ev(
    target: ArrayView2<'_, f64>,
    fitted: ArrayView2<'_, f64>,
) -> Result<f64, String> {
    if target.dim() != fitted.dim() {
        return Err(format!(
            "reconstruction_ev: target {:?} != fitted {:?}",
            target.dim(),
            fitted.dim()
        ));
    }
    let sst = centered_sst(target);
    if !(sst.is_finite() && sst > f64::MIN_POSITIVE) {
        return Err("reconstruction_ev: target has no centered variance".to_string());
    }
    let mut rss = 0.0_f64;
    for row in 0..target.nrows() {
        for col in 0..target.ncols() {
            let residual = target[[row, col]] - fitted[[row, col]];
            rss += residual * residual;
        }
    }
    Ok(1.0 - rss / sst)
}

fn top_m_linear_ev(target: ArrayView2<'_, f64>, basis_size: usize) -> Result<f64, String> {
    let centered = centered_matrix(target);
    let sst = centered.iter().map(|v| v * v).sum::<f64>();
    if !(sst.is_finite() && sst > f64::MIN_POSITIVE) {
        return Err("top_m_linear_ev: target has no centered variance".to_string());
    }
    let (_u, sigma, _vt) = centered
        .svd(false, false)
        .map_err(|err| format!("top_m_linear_ev: SVD failed: {err}"))?;
    let keep = basis_size.min(sigma.len());
    let captured = (0..keep).map(|idx| sigma[idx] * sigma[idx]).sum::<f64>();
    Ok((captured / sst).min(1.0))
}

fn score_energy(scores: ArrayView2<'_, f64>, component: usize) -> Result<f64, String> {
    if component >= scores.ncols() {
        return Err(format!(
            "score_energy: component {component} out of range for {} scores",
            scores.ncols()
        ));
    }
    Ok(scores.column(component).iter().map(|v| v * v).sum::<f64>())
}

fn centered_sst(x: ArrayView2<'_, f64>) -> f64 {
    let centered = centered_matrix(x);
    centered.iter().map(|v| v * v).sum::<f64>()
}

fn centered_matrix(x: ArrayView2<'_, f64>) -> Array2<f64> {
    let (n, p) = x.dim();
    let mut means = vec![0.0_f64; p];
    for col in 0..p {
        for row in 0..n {
            means[col] += x[[row, col]];
        }
        means[col] /= n as f64;
    }
    let mut centered = Array2::<f64>::zeros((n, p));
    for row in 0..n {
        for col in 0..p {
            centered[[row, col]] = x[[row, col]] - means[col];
        }
    }
    centered
}

fn parse_npy_header(
    head: &[u8],
    path: &Path,
) -> Result<(usize, usize, usize, bool, usize), String> {
    if head.len() <= 12 || &head[0..6] != b"\x93NUMPY" {
        return Err(format!("{}: not a .npy file", path.display()));
    }
    let major = head[6];
    let (header_len, data_off) = if major >= 2 {
        let header_len = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
        (header_len, 12 + header_len)
    } else {
        let header_len = u16::from_le_bytes([head[8], head[9]]) as usize;
        (header_len, 10 + header_len)
    };
    if data_off > head.len() {
        return Err(format!(
            "{}: header exceeds initial read buffer",
            path.display()
        ));
    }
    let header = std::str::from_utf8(&head[data_off - header_len..data_off])
        .map_err(|err| format!("{}: header is not utf8: {err}", path.display()))?;
    let is_f4 = header.contains("'<f4'") || header.contains("\"<f4\"");
    let is_f2 = header.contains("'<f2'") || header.contains("\"<f2\"");
    if !(is_f4 || is_f2) {
        return Err(format!(
            "{}: expected little-endian <f4 or <f2; header: {header}",
            path.display()
        ));
    }
    if !(header.contains("'fortran_order': False") || header.contains("\"fortran_order\": false")) {
        return Err(format!(
            "{}: expected C-order; header: {header}",
            path.display()
        ));
    }
    let shape_start = header
        .find("'shape':")
        .ok_or_else(|| format!("{}: missing shape key", path.display()))?
        + "'shape':".len();
    let paren_open = header[shape_start..]
        .find('(')
        .ok_or_else(|| format!("{}: missing shape open paren", path.display()))?
        + shape_start
        + 1;
    let paren_close = header[paren_open..]
        .find(')')
        .ok_or_else(|| format!("{}: missing shape close paren", path.display()))?
        + paren_open;
    let dims: Vec<usize> = header[paren_open..paren_close]
        .split(',')
        .filter_map(|token| token.trim().parse::<usize>().ok())
        .collect();
    if dims.len() != 2 {
        return Err(format!(
            "{}: expected a 2-D array, got {dims:?}",
            path.display()
        ));
    }
    let elem = if is_f4 { 4 } else { 2 };
    Ok((dims[0], dims[1], elem, is_f4, data_off))
}

fn read_npy_subsample_f64(path: &Path, cap: usize) -> Result<(usize, usize, Array2<f64>), String> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file =
        std::fs::File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let mut head = vec![0u8; 512];
    file.read_exact(&mut head)
        .map_err(|err| format!("read header {}: {err}", path.display()))?;
    let (n_full, p, elem, is_f4, data_off) = parse_npy_header(&head, path)?;
    let take = cap.min(n_full);
    let stride = (n_full / take).max(1);
    let row_bytes = p
        .checked_mul(elem)
        .ok_or_else(|| format!("{}: row byte size overflow", path.display()))?;
    let mut buf = vec![0u8; row_bytes];
    let mut out = Array2::<f64>::zeros((take, p));
    for i in 0..take {
        let src_row = i * stride;
        let off = data_off as u64 + (src_row as u64) * (row_bytes as u64);
        file.seek(SeekFrom::Start(off))
            .map_err(|err| format!("seek {}: {err}", path.display()))?;
        file.read_exact(&mut buf)
            .map_err(|err| format!("read row {src_row} {}: {err}", path.display()))?;
        if is_f4 {
            for c in 0..p {
                let b = c * 4;
                out[[i, c]] =
                    f32::from_le_bytes([buf[b], buf[b + 1], buf[b + 2], buf[b + 3]]) as f64;
            }
        } else {
            for c in 0..p {
                let b = c * 2;
                out[[i, c]] = f16_to_f32(u16::from_le_bytes([buf[b], buf[b + 1]])) as f64;
            }
        }
    }
    Ok((n_full, p, out))
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 0x1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let bits = if exp == 0 {
        if mant == 0 {
            (sign as u32) << 31
        } else {
            let mut m = mant as u32;
            let mut e: i32 = -1;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let exp32 = (127 - 15 + 1 + e) as u32;
            ((sign as u32) << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13)
    } else {
        let exp32 = (exp as i32 - 15 + 127) as u32;
        ((sign as u32) << 31) | (exp32 << 23) | ((mant as u32) << 13)
    };
    f32::from_bits(bits)
}

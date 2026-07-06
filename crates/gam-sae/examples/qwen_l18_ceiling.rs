//! Real-data K=1 circle ceiling experiment for Qwen3-8B L18 residuals.
//!
//! Usage:
//!
//! ```text
//! cargo run -p gam-sae --example qwen_l18_ceiling -- \
//!   <post_peel_pca.npy> [max_rows] [harmonics] [outer_iters] [inner_iters] [--raw-ok] [--out-dir DIR]
//! ```

use faer::Side;
use gam_linalg::faer_ndarray::{FaerCholesky, FaerSvd, fast_atb};
use gam_problem::SeedConfig;
use gam_sae::assignment::{AssignmentMode, SaeAssignment};
use gam_sae::basis::{PeriodicHarmonicEvaluator, SaeBasisSecondJet};
use gam_sae::identifiability::thin_svd_scores;
use gam_sae::manifold::{
    LatentManifold, SaeAtomBasisKind, SaeManifoldAtom, SaeManifoldOuterObjective, SaeManifoldRho,
    SaeManifoldTerm,
};
use gam_solve::rho_optimizer::{CriterionCertificate, OuterProblem};
use ndarray::{Array1, Array2, ArrayView2, s};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_MAX_ROWS: usize = 20_000;
const DEFAULT_HARMONICS: usize = 1;
const DEFAULT_OUTER_ITERS: usize = 12;
const DEFAULT_INNER_ITERS: usize = 8;
const MAX_POST_PEEL_PCA_DIM: usize = 64;
const CEILING_K: usize = 1;

fn main() -> ExitCode {
    let args = match Args::parse() {
        Ok(args) => args,
        Err(err) => {
            println!("{err}");
            return ExitCode::from(2);
        }
    };

    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            println!("[qwen_l18_ceiling] error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Clone, Debug)]
struct Args {
    path: PathBuf,
    max_rows: usize,
    harmonics: usize,
    outer_iters: usize,
    inner_iters: usize,
    raw_ok: bool,
    out_dir: PathBuf,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut raw = std::env::args();
        let program = raw.next().unwrap_or_else(|| "qwen_l18_ceiling".to_string());
        let mut path = None;
        let mut positional = Vec::new();
        let mut raw_ok = false;
        let mut out_dir = PathBuf::from(".");
        while let Some(arg) = raw.next() {
            match arg.as_str() {
                "--raw-ok" => raw_ok = true,
                "--out-dir" => {
                    let Some(value) = raw.next() else {
                        return Err("--out-dir requires a directory".to_string());
                    };
                    out_dir = PathBuf::from(value);
                }
                flag if flag.starts_with("--") => return Err(format!("unknown argument {flag}")),
                value if path.is_none() => path = Some(PathBuf::from(value)),
                value => positional.push(value.to_string()),
            }
        }
        let Some(path) = path else {
            return Err(format!(
                "usage: {program} <post_peel_pca.npy> [max_rows] [harmonics] [outer_iters] [inner_iters] [--raw-ok] [--out-dir DIR]"
            ));
        };
        if positional.len() > 4 {
            return Err(format!(
                "usage: {program} <post_peel_pca.npy> [max_rows] [harmonics] [outer_iters] [inner_iters] [--raw-ok] [--out-dir DIR]"
            ));
        }
        Ok(Self {
            path,
            max_rows: parse_positional_usize(
                positional.first(),
                DEFAULT_MAX_ROWS,
                "max_rows",
            )?,
            harmonics: parse_positional_usize(
                positional.get(1),
                DEFAULT_HARMONICS,
                "harmonics",
            )?,
            outer_iters: parse_positional_usize(
                positional.get(2),
                DEFAULT_OUTER_ITERS,
                "outer_iters",
            )?,
            inner_iters: parse_positional_usize(
                positional.get(3),
                DEFAULT_INNER_ITERS,
                "inner_iters",
            )?,
            raw_ok,
            out_dir,
        })
    }
}

fn parse_positional_usize(
    raw: Option<&String>,
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

fn run(args: &Args) -> Result<(), String> {
    let started = Instant::now();
    let header = read_npy_header_info(&args.path)?;
    if header.rows == 0 || header.cols == 0 {
        return Err(format!(
            "{}: ceiling input must be a non-empty N x p matrix, got {} x {}",
            args.path.display(),
            header.rows,
            header.cols
        ));
    }
    let post_peel_input = header.cols <= MAX_POST_PEEL_PCA_DIM;
    if !post_peel_input && !args.raw_ok {
        return Err(format!(
            "refusing raw full-width ceiling run: {} has p={}, above the post-peel/PCA contract limit of {MAX_POST_PEEL_PCA_DIM}. Pass a post-peel PCA-reduced .npy with p <= {MAX_POST_PEEL_PCA_DIM}, or rerun with --raw-ok to accept the raw full-width memory risk.",
            args.path.display(),
            header.cols
        ));
    }
    let n_used = args.max_rows.min(header.rows);
    let pca_dim = if post_peel_input { header.cols } else { 8 };
    let run_numbers = run_numbers_json(
        n_used,
        header.cols,
        post_peel_input,
        pca_dim,
        args.max_rows,
        args.harmonics,
    );
    write_numbers_json(&args.out_dir, &run_numbers)?;

    let (n_full, p_raw, raw) = read_npy_subsample_f64(&args.path, args.max_rows)?;
    let basis_size = 1 + 2 * args.harmonics;
    let top_sink_ev;
    let pre_report;
    let post_peel_region;
    let post_report;
    if post_peel_input {
        top_sink_ev = f64::NAN;
        pre_report = None;
        post_peel_region = raw;
        post_report = fit_ceiling_region(
            "post_peel",
            post_peel_region.view(),
            "post_peel",
            args.harmonics,
            args.outer_iters,
            args.inner_iters,
        )?;
    } else {
        if raw.ncols() < 9 {
            return Err(format!(
                "raw --raw-ok sink-peel contract requires at least 9 residual dimensions, got {}",
                raw.ncols()
            ));
        }
        let scores = thin_svd_scores(raw.view(), 9)?;
        let pre_peel_region = scores.clone();
        post_peel_region = scores.slice(s![.., 1..9]).to_owned();
        top_sink_ev = score_ev(scores.view(), 0)?;
        pre_report = Some(fit_ceiling_region(
            "pre_peel",
            pre_peel_region.view(),
            "raw",
            args.harmonics,
            args.outer_iters,
            args.inner_iters,
        )?);
        post_report = fit_ceiling_region(
            "post_peel",
            post_peel_region.view(),
            "post_peel",
            args.harmonics,
            args.outer_iters,
            args.inner_iters,
        )?;
    }

    println!("qwen_l18_ceiling_result");
    println!("source_rows_full={n_full}");
    println!("source_dim={p_raw}");
    println!("subsample_rows={}", post_peel_region.nrows());
    println!("basis_size={basis_size}");
    println!("top_sink_ev_pre_peel={top_sink_ev:.9}");
    println!(
        "decision_tree=eta~1 -> information ceiling (not solver failure, not thesis failure); eta<<1 with clean gradient cert -> landscape/gauge-orbit pathology; eta<<1 with failing cert -> residual adjoint bug"
    );
    if let Some(report) = pre_report.as_ref() {
        print_ceiling_report(report);
    }
    print_ceiling_report(&post_report);
    println!(
        "ceiling_contract_json={}",
        serde_json::to_string(&ceiling_contract_json(&post_report))
            .map_err(|err| format!("serialize ceiling_contract_json: {err}"))?
    );
    println!("numbers_json={}", args.out_dir.join("numbers.json").display());
    println!("wall_seconds={:.3}", started.elapsed().as_secs_f64());
    Ok(())
}

fn run_numbers_json(
    n: usize,
    p: usize,
    post_peel: bool,
    pca_dim: usize,
    max_rows: usize,
    harmonics: usize,
) -> Value {
    json!({
        "N": n,
        "p": p,
        "K": CEILING_K,
        "post_peel": post_peel,
        "pca_dim": pca_dim,
        "max_rows": max_rows,
        "harmonics": harmonics,
    })
}

fn write_numbers_json(out_dir: &Path, value: &Value) -> Result<(), String> {
    std::fs::create_dir_all(out_dir)
        .map_err(|err| format!("create output directory {}: {err}", out_dir.display()))?;
    let path = out_dir.join("numbers.json");
    let text = serde_json::to_string_pretty(value)
        .map_err(|err| format!("serialize {}: {err}", path.display()))?;
    std::fs::write(&path, format!("{text}\n"))
        .map_err(|err| format!("write {}: {err}", path.display()))
}

#[derive(Clone, Debug)]
struct CeilingRegionReport {
    label: &'static str,
    peel_status: &'static str,
    rows: usize,
    dim: usize,
    basis_size: usize,
    curved_ev: f64,
    full_reconstruction_ev: f64,
    top_m_linear_ev: f64,
    chart_efficiency_eta: f64,
    final_outer_grad_norm: Option<f64>,
    outer_converged: bool,
    outer_iterations: usize,
    inner_iterations: usize,
    fit_wall_seconds: f64,
    criterion_calls: usize,
    fd_probe_calls: usize,
    infeasible_total: usize,
    wall_cost_value_probes: usize,
    mutating_value_probes: usize,
    certificate: Option<CriterionCertificate>,
}

fn fit_ceiling_region(
    label: &'static str,
    target: ArrayView2<'_, f64>,
    peel_status: &'static str,
    harmonics: usize,
    outer_iters: usize,
    inner_iters: usize,
) -> Result<CeilingRegionReport, String> {
    let fit_started = Instant::now();
    let (term, seed_dispersion, basis_size) = periodic_k1_term(target, harmonics)?;
    let mode = AssignmentMode::ibp_map(1.0, 1.0, false);
    let init_rho = SaeManifoldRho::new(0.02_f64.ln(), 1.0_f64.ln(), vec![Array1::zeros(1)])
        .seed_scaled_by_dispersion_for_assignment(seed_dispersion, mode)?;
    let seed = init_rho.to_flat();
    let n_params = seed.len();
    let mut objective = SaeManifoldOuterObjective::new(
        term,
        target.to_owned(),
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
        .run(&mut objective, "Qwen3-8B L18 K=1 ceiling")
        .map_err(|err| format!("outer fit failed: {err}"))?;
    let fit_elapsed = fit_started.elapsed();
    let telemetry = objective.probe_telemetry();
    let fitted = objective.into_fitted();
    let final_term = fitted.term;
    let fitted_matrix = final_term.fitted();
    let curved_ev = reconstruction_ev(target, fitted_matrix.view())?;
    let topm_linear_ev = top_m_linear_ev(target, basis_size)?;
    let ratio = if topm_linear_ev > f64::MIN_POSITIVE {
        curved_ev / topm_linear_ev
    } else {
        f64::NAN
    };

    Ok(CeilingRegionReport {
        label,
        peel_status,
        rows: target.nrows(),
        dim: target.ncols(),
        basis_size,
        curved_ev,
        full_reconstruction_ev: curved_ev,
        top_m_linear_ev: topm_linear_ev,
        chart_efficiency_eta: ratio,
        final_outer_grad_norm: result.final_grad_norm,
        outer_converged: result.converged,
        outer_iterations: result.iterations,
        inner_iterations: inner_iters,
        fit_wall_seconds: fit_elapsed.as_secs_f64(),
        criterion_calls: telemetry.criterion_calls,
        fd_probe_calls: telemetry.fd_probe_calls,
        infeasible_total: telemetry.infeasible_total(),
        wall_cost_value_probes: telemetry.wall_cost_value_probes,
        mutating_value_probes: telemetry.mutating_value_probes,
        certificate: result.criterion_certificate,
    })
}

fn print_ceiling_report(report: &CeilingRegionReport) {
    let label = report.label;
    println!("{label}_peel_status={}", report.peel_status);
    println!("{label}_rows={}", report.rows);
    println!("{label}_dim={}", report.dim);
    println!("{label}_basis_size={}", report.basis_size);
    println!("{label}_EV_curved={:.9}", report.curved_ev);
    println!(
        "{label}_full_reconstruction_ev={:.9}",
        report.full_reconstruction_ev
    );
    println!(
        "{label}_top_M_linear_envelope_EV={:.9}",
        report.top_m_linear_ev
    );
    println!(
        "{label}_EV_top_M_linear_envelope={:.9}",
        report.top_m_linear_ev
    );
    println!(
        "{label}_EV_lin_top_m_envelope={:.9}",
        report.top_m_linear_ev
    );
    println!(
        "{label}_chart_efficiency_eta={:.9}",
        report.chart_efficiency_eta
    );
    println!(
        "{label}_envelope_theorem_slack={:.9}",
        report.top_m_linear_ev - report.curved_ev
    );
    println!(
        "{label}_final_outer_grad_norm={}",
        report
            .final_outer_grad_norm
            .map(|g| format!("{g:.9}"))
            .unwrap_or_else(|| "nan".to_string())
    );
    println!("{label}_converged={}", report.outer_converged);
    println!("{label}_outer_iterations={}", report.outer_iterations);
    println!("{label}_inner_iterations={}", report.inner_iterations);
    println!("{label}_fit_wall_seconds={:.3}", report.fit_wall_seconds);
    println!("{label}_criterion_calls={}", report.criterion_calls);
    println!("{label}_fd_probe_calls={}", report.fd_probe_calls);
    println!("{label}_infeasible_total={}", report.infeasible_total);
    println!(
        "{label}_wall_cost_value_probes={}",
        report.wall_cost_value_probes
    );
    println!(
        "{label}_mutating_value_probes={}",
        report.mutating_value_probes
    );
    print_gradient_certificate(label, report.certificate.as_ref());
    println!(
        "{label}_decision={}",
        ceiling_decision(
            report.chart_efficiency_eta,
            gradient_certificate_clean(report.certificate.as_ref())
        )
    );
}

fn ceiling_contract_json(report: &CeilingRegionReport) -> Value {
    json!({
        "peel_status": report.peel_status,
        "EV_curved": report.curved_ev,
        "EV_lin_top_m_envelope": report.top_m_linear_ev,
        "chart_efficiency_eta": report.chart_efficiency_eta,
        "gradient_certificate": if gradient_certificate_clean(report.certificate.as_ref()) {
            "clean"
        } else {
            "failing"
        },
        "verdict": ceiling_verdict(report.chart_efficiency_eta, gradient_certificate_clean(report.certificate.as_ref())),
    })
}

fn print_gradient_certificate(label: &str, certificate: Option<&CriterionCertificate>) {
    match certificate {
        Some(cert) => {
            let clean = gradient_certificate_clean(Some(cert));
            println!(
                "{label}_dual_oracle_gradient_certificate={}",
                if clean { "clean" } else { "failing" }
            );
            println!(
                "{label}_dual_oracle_gradient_certificate_first_order_consistent={}",
                cert.first_order_consistent()
            );
            println!(
                "{label}_dual_oracle_gradient_certificate_grad_norm={:.9}",
                cert.grad_norm
            );
            println!(
                "{label}_dual_oracle_gradient_certificate_analytic_directional={:.9}",
                cert.analytic_directional
            );
            println!(
                "{label}_dual_oracle_gradient_certificate_fd_directional={:.9}",
                cert.fd_directional
            );
            println!(
                "{label}_dual_oracle_gradient_certificate_fd_error={:.9}",
                cert.fd_error
            );
            println!(
                "{label}_dual_oracle_gradient_certificate_agreement_z={:.9}",
                cert.agreement_z
            );
            println!(
                "{label}_dual_oracle_gradient_certificate_fd_step={:.9}",
                cert.fd_step
            );
            println!(
                "{label}_dual_oracle_gradient_certificate_hessian_pd={}",
                optional_bool(cert.hessian_pd)
            );
            println!(
                "{label}_dual_oracle_gradient_certificate_lambdas_railed={:?}",
                cert.lambdas_railed
            );
        }
        None => {
            println!("{label}_dual_oracle_gradient_certificate=missing");
        }
    }
}

fn gradient_certificate_clean(certificate: Option<&CriterionCertificate>) -> bool {
    certificate.is_some_and(|cert| {
        cert.first_order_consistent()
            && cert.hessian_pd != Some(false)
            && cert.lambdas_railed.is_empty()
    })
}

fn optional_bool(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "true",
        Some(false) => "false",
        None => "unknown",
    }
}

fn ceiling_decision(eta: f64, clean_gradient_certificate: bool) -> &'static str {
    if eta >= 0.90 {
        "information_ceiling_not_solver_failure_not_thesis_failure"
    } else if clean_gradient_certificate {
        "landscape_or_gauge_orbit_pathology"
    } else {
        "residual_adjoint_bug"
    }
}

fn ceiling_verdict(eta: f64, clean_gradient_certificate: bool) -> &'static str {
    if eta >= 0.90 {
        "INFORMATION_CEILING"
    } else if clean_gradient_certificate {
        "LANDSCAPE_PATHOLOGY"
    } else {
        "RESIDUAL_ADJOINT_BUG"
    }
}

fn periodic_k1_term(
    z: ArrayView2<'_, f64>,
    harmonics: usize,
) -> Result<(SaeManifoldTerm, f64, usize), String> {
    let n = z.nrows();
    let p = z.ncols();
    let dim = 1usize;
    let num_basis = 1 + 2 * harmonics;
    let evaluator: Arc<dyn SaeBasisSecondJet> = Arc::new(PeriodicHarmonicEvaluator::new(num_basis)?);
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
        "qwen_l18_k1_circle",
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

fn reconstruction_ev(target: ArrayView2<'_, f64>, fitted: ArrayView2<'_, f64>) -> Result<f64, String> {
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
    let svd = centered
        .svd(false, false)
        .map_err(|err| format!("top_m_linear_ev: SVD failed: {err}"))?;
    let sigma = svd.1;
    let keep = basis_size.min(sigma.len());
    let captured = (0..keep).map(|idx| sigma[idx] * sigma[idx]).sum::<f64>();
    Ok((captured / sst).min(1.0))
}

fn score_ev(scores: ArrayView2<'_, f64>, component: usize) -> Result<f64, String> {
    if component >= scores.ncols() {
        return Err(format!(
            "score_ev: component {component} out of range for {} scores",
            scores.ncols()
        ));
    }
    let total = scores.iter().map(|v| v * v).sum::<f64>();
    if !(total.is_finite() && total > f64::MIN_POSITIVE) {
        return Err("score_ev: scores have no variance".to_string());
    }
    let captured = scores.column(component).iter().map(|v| v * v).sum::<f64>();
    Ok(captured / total)
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

#[derive(Clone, Copy, Debug)]
struct NpyHeaderInfo {
    rows: usize,
    cols: usize,
}

fn read_npy_header_info(path: &Path) -> Result<NpyHeaderInfo, String> {
    use std::io::Read;

    let mut file =
        std::fs::File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    let mut head = vec![0u8; 512];
    file.read_exact(&mut head)
        .map_err(|err| format!("read header {}: {err}", path.display()))?;
    let parsed = parse_npy_header(&head, path)?;
    Ok(NpyHeaderInfo {
        rows: parsed.0,
        cols: parsed.1,
    })
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

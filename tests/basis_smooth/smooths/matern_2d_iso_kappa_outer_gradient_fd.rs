//! #1122 / #901 iso-κ gradient class — the FULL joint-REML objective gradient
//! w.r.t. the Matérn isotropic length-scale coordinate `ψ = log κ` must match a
//! central finite difference of the COMPLETE outer criterion, not just the
//! penalty block.
//!
//! The penalty-only basis test
//! (`basis_matern_double_penalty_log_kappa_derivative_fd`) checks only
//! `∂(penalty matrices)/∂log_κ`. The full profiled REML objective
//!   V(κ) = data-fit(deviance) + ½log|H+Sλ| − ½log|Sλ|₊ + penalty-quad
//! also depends on κ through the Matérn DESIGN columns (and hence the deviance
//! and `H = XᵀWX`) and through the H-side of the `½log|H+Sλ|` term. A stall with
//! a large residual gradient at the iteration cap is the signature of an
//! objective↔gradient DESYNC: the optimizer follows a gradient inconsistent with
//! the objective it is minimizing.
//!
//! The root cause was the double-penalty nullspace-shrinkage decision (and the
//! identifiability transform `Z`) NOT being frozen across the κ-optimizer's
//! per-trial value rebuilds: the κ-DEPENDENT spectral test
//! (`build_nullspace_shrinkage_penalty`, tolerance ∝ λ_max(A(κ))) flips the
//! shrinkage block `P/√r` discontinuously as κ moves, so V(κ) is
//! piecewise-discontinuous while the analytic gradient (assembled in a fixed
//! frozen eigenbasis) is smooth. The fix freezes BOTH `Z` and the
//! shrinkage decision into a `FrozenTransform` at the first per-trial rebuild,
//! and mirrors that freeze onto the collection spec the analytic ψ-gradient
//! reads, so value and gradient share one fixed `Z` and one fixed null
//! dimension `r` at every trial.
//!
//! The outer driver (`spatial-exact-joint`) runs
//! `solver::outer_strategy::outer_gradient_fd_audit` automatically at θ₀,
//! central-differencing the *outer criterion* component-by-component and
//! comparing it to the analytic outer gradient. The Matérn log-κ coordinate is
//! labelled `psi_kappa[..]`. This test fits an ordinary Gaussian 2-D surface
//! with a single `matern(x1, x2)` smooth (default double-penalty), captures the
//! audit, and asserts the analytic gradient w.r.t. log-κ matches the central FD
//! of the full criterion (no DESYNC verdict, finite per-coordinate analytic/fd,
//! small relative gap on the κ block).
//!
//! Reference-as-truth: every assertion is against the analytic FD of gam's own
//! profiled REML criterion — never another tool's output.

use gam::{
    FitConfig, FitRequest, FitResult, StandardFitRequest, encode_recordswith_inferred_schema,
    estimate::FitOptions,
    fit_model,
    smooth::{
        ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
        TermCollectionSpec,
    },
    terms::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu},
    types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink},
};
use ndarray::{Array1, Array2};
use std::sync::{Mutex, Once};

static CAPTURE: Mutex<Vec<String>> = Mutex::new(Vec::new());
static TEST_LOG_LOCK: Mutex<()> = Mutex::new(());

struct CapturingLogger;
impl log::Log for CapturingLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record<'_>) {
        let line = format!("{}", record.args());
        if line.contains("OUTER-FD-AUDIT") {
            if let Ok(mut g) = CAPTURE.lock() {
                g.push(line.clone());
            }
            eprintln!("{line}");
        }
    }
    fn flush(&self) {}
}

static LOGGER: CapturingLogger = CapturingLogger;
static INIT_LOGGER: Once = Once::new();

fn init() {
    #[cfg(target_os = "macos")]
    gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);
    gam::init_parallelism();
    INIT_LOGGER.call_once(|| {
        if log::set_logger(&LOGGER).is_ok() {
            log::set_max_level(log::LevelFilter::Info);
        }
    });
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1.0e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn truth(a: f64, b: f64) -> f64 {
    (2.0 * std::f64::consts::PI * a).sin() * (2.0 * std::f64::consts::PI * b).sin()
}

fn build_dataset(n: usize, sigma: f64, seed: u64) -> gam::inference::data::EncodedDataset {
    let mut st = seed;
    let mut header = String::from("y,x1,x2\n");
    let mut body = String::new();
    for _ in 0..n {
        let a = next_unit(&mut st);
        let b = next_unit(&mut st);
        let y = truth(a, b) + sigma * next_gauss(&mut st);
        body.push_str(&format!("{y:.6},{a:.6},{b:.6}\n"));
    }
    header.push_str(&body);
    let mut rdr = csv::ReaderBuilder::new().from_reader(header.as_bytes());
    let records: Vec<csv::StringRecord> = rdr.records().map(|r| r.unwrap()).collect();
    let headers = vec!["y".to_string(), "x1".to_string(), "x2".to_string()];
    encode_recordswith_inferred_schema(headers, records).expect("encode dataset")
}

fn field(line: &str, key: &str) -> String {
    if let Some(pos) = line.find(key) {
        let rest = &line[pos + key.len()..];
        rest.split_whitespace().next().unwrap_or("").to_string()
    } else {
        String::new()
    }
}

fn parse_components(lines: &[String]) -> Vec<(String, f64, f64, f64)> {
    let mut out = Vec::new();
    for l in lines {
        if !l.contains(" analytic=") || !l.contains(" fd=") || !l.contains(" gap=") {
            continue;
        }
        let block = field(l, "block=");
        let a: f64 = field(l, "analytic=").parse().unwrap_or(f64::NAN);
        let fd: f64 = field(l, "fd=").parse().unwrap_or(f64::NAN);
        let gap: f64 = field(l, "gap=").parse().unwrap_or(f64::NAN);
        if !block.is_empty() {
            out.push((block, a, fd, gap));
        }
    }
    out
}

fn aniso_signal_dataset(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x1 = (i as f64) / (n as f64 - 1.0) * 6.0 - 3.0;
        let x2 = ((i as f64 * 0.618_033_988_749_894_9).fract()) * 6.0 - 3.0;
        x[[i, 0]] = x1;
        x[[i, 1]] = x2;
        y[i] = (2.0 * x1).sin();
    }
    (x, y)
}

/// MERGE GATE (#1122 / #901): the analytic outer REML gradient w.r.t. the
/// Matérn log-κ coordinate matches a central finite difference of the FULL
/// profiled REML criterion at θ₀ — data-fit + logdet (both Sλ-side AND H-side) +
/// penalty-quad — with the default double-penalty (nullspace shrinkage) active.
#[test]
fn matern_2d_iso_kappa_outer_gradient_matches_fd() {
    let _guard = TEST_LOG_LOCK.lock().unwrap();
    init();
    if let Ok(mut g) = CAPTURE.lock() {
        g.clear();
    }
    let data = build_dataset(150, 0.05, 0x9A7E_7212_0001u64);
    let formula = "y ~ matern(x1, x2)";
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        // The outer FD audit fires at θ₀ during the fit, before the (capped)
        // outer loop; cap the loop so the test stays fast — the gate we assert
        // is the captured audit, independent of the fit's eventual outcome.
        outer_max_iter: Some(2),
        gpu_policy: if cfg!(target_os = "macos") {
            gam::gpu::GpuPolicy::Off
        } else {
            gam::gpu::GpuPolicy::Auto
        },
        ..FitConfig::default()
    };
    match gam::fit_from_formula(formula, &data, &config) {
        Ok(_) => eprintln!("[FD-DIAG] matern(x1,x2) fit returned Ok"),
        Err(e) => eprintln!("[FD-DIAG] matern(x1,x2) fit returned Err (audit still ran): {e}"),
    }

    let lines = CAPTURE.lock().unwrap().clone();
    // TEMP-1122-DUMP: persist all captured audit lines so the HSWEEP atom
    // decomposition survives build.sh's stderr filter.
    {
        let _ = std::fs::write("/tmp/iso_capture.txt", lines.join("\n"));
    }
    let gate: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("gate eligible="))
        .collect();
    assert!(
        !gate.is_empty(),
        "expected an [OUTER-FD-AUDIT] gate line; captured {} lines: {:#?}",
        lines.len(),
        lines
    );
    let psi_dim: usize = gate
        .iter()
        .filter_map(|l| field(l, "psi_dim=").parse::<usize>().ok())
        .max()
        .unwrap_or(0);
    assert!(
        psi_dim >= 1,
        "matern(x1,x2) must enroll log-κ as a ψ-coordinate (psi_dim≥1); gate lines: {gate:#?}"
    );

    let comps = parse_components(&lines);
    assert!(
        !comps.is_empty(),
        "expected per-coordinate analytic-vs-FD lines; captured: {:#?}",
        lines
    );

    // No DESYNC verdict: the FULL-objective analytic gradient agrees with FD.
    let desync: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("VERDICT=DESYNC"))
        .collect();
    assert!(
        desync.is_empty(),
        "Matérn iso-κ outer gradient DESYNCs from the FD of the full criterion: {desync:#?}"
    );

    // The log-κ block(s) specifically: finite and analytic≈fd to tolerance.
    let kappa_comps: Vec<&(String, f64, f64, f64)> = comps
        .iter()
        .filter(|(block, ..)| block.starts_with("psi_kappa"))
        .collect();
    assert!(
        !kappa_comps.is_empty(),
        "expected a psi_kappa[..] audit component; got blocks: {:?}",
        comps.iter().map(|c| &c.0).collect::<Vec<_>>()
    );
    for (block, a, fd, gap) in &kappa_comps {
        assert!(
            a.is_finite() && fd.is_finite(),
            "non-finite Matérn log-κ gradient component {block}: analytic={a} fd={fd}"
        );
        let scale = a.abs().max(fd.abs()).max(1e-6);
        assert!(
            gap / scale < 5e-2,
            "Matérn iso-κ outer-gradient analytic≠FD on {block}: analytic={a:.6e} \
             fd={fd:.6e} gap={gap:.3e} rel={:.3e}",
            gap / scale
        );
    }
}

/// #1259: at the symmetric anisotropic Matérn init, the FULL outer REML
/// criterion must have a nonzero per-axis eta contrast in the direction that
/// increases the signal-axis eta. The audit is stronger than checking the final
/// fitted eta split: it verifies the value path itself sees trial eta
/// perturbations, so the optimizer has a real descent direction at theta0.
#[test]
fn aniso_matern_theta0_eta_contrast_gradient_is_fd_visible() {
    let _guard = TEST_LOG_LOCK.lock().unwrap();
    init();
    if let Ok(mut g) = CAPTURE.lock() {
        g.clear();
    }

    let n = 180;
    let (x, y) = aniso_signal_dataset(n);
    let spec = TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "matern_2d_aniso".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: vec![0, 1],
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                    periodic: None,
                    length_scale: 1.0,
                    nu: MaternNu::FiveHalves,
                    include_intercept: false,
                    double_penalty: true,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: Some(vec![0.0, 0.0]),
                    nullspace_shrinkage_survived: None,
                },
                input_scales: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    };

    let outcome = fit_model(FitRequest::Standard(StandardFitRequest {
        data: x,
        y,
        weights: Array1::ones(n),
        offset: Array1::zeros(n),
        spec,
        family: LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        options: FitOptions {
            latent_cloglog: None,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: None,
            optimize_sas: false,
            compute_inference: false,
            skip_rho_posterior_inference: false,
            max_iter: 2,
            tol: 1e-6,
            nullspace_dims: vec![],
            linear_constraints: None,
            firth_bias_reduction: false,
            adaptive_regularization: None,
            penalty_shrinkage_floor: None,
            rho_prior: Default::default(),
            kronecker_penalty_system: None,
            kronecker_factored: None,
            persist_warm_start_disk: false,
        },
        kappa_options: SpatialLengthScaleOptimizationOptions {
            enabled: true,
            max_outer_iter: 2,
            rel_tol: 1e-5,
            log_step: std::f64::consts::LN_2,
            min_length_scale: 1e-2,
            max_length_scale: 1e2,
            pilot_subsample_threshold: 0,
            outer_wall_clock_budget_secs: None,
        },
        wiggle: None,
        coefficient_groups: Vec::new(),
        penalty_block_gamma_priors: Vec::new(),
        latent_coord: None,
        _marker: std::marker::PhantomData,
    }));
    match outcome {
        Ok(FitResult::Standard(_)) => eprintln!("[ANISO-ETA-GRAD] fit returned Ok"),
        Ok(_) => panic!("expected standard fit"),
        Err(e) => eprintln!("[ANISO-ETA-GRAD] fit returned Err after audit: {e}"),
    }

    let lines = CAPTURE.lock().unwrap().clone();
    let gate: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("gate eligible="))
        .collect();
    assert!(
        !gate.is_empty(),
        "expected an [OUTER-FD-AUDIT] gate line; captured {} lines: {:#?}",
        lines.len(),
        lines
    );
    let psi_dim: usize = gate
        .iter()
        .filter_map(|l| field(l, "psi_dim=").parse::<usize>().ok())
        .max()
        .unwrap_or(0);
    assert!(
        psi_dim >= 2,
        "anisotropic Matérn must enroll both eta axes as ψ coordinates; gate lines: {gate:#?}"
    );

    let comps = parse_components(&lines);
    let eta_comps: Vec<&(String, f64, f64, f64)> = comps
        .iter()
        .filter(|(block, ..)| block.starts_with("psi_kappa"))
        .take(2)
        .collect();
    assert!(
        eta_comps.len() >= 2,
        "expected two eta-axis psi_kappa components; got blocks: {:?}",
        comps.iter().map(|c| &c.0).collect::<Vec<_>>()
    );
    let (_, g_signal, fd_signal, _) = eta_comps[0];
    let (_, g_noise, fd_noise, _) = eta_comps[1];
    let analytic_contrast = g_signal - g_noise;
    let fd_contrast = fd_signal - fd_noise;
    eprintln!(
        "[ANISO-ETA-GRAD] theta0 psi_grad=[{g_signal:.6e}, {g_noise:.6e}] \
         fd=[{fd_signal:.6e}, {fd_noise:.6e}] analytic_contrast={analytic_contrast:.6e} \
         fd_contrast={fd_contrast:.6e}"
    );

    for (block, a, fd, gap) in eta_comps {
        assert!(
            a.is_finite() && fd.is_finite(),
            "non-finite anisotropic eta gradient component {block}: analytic={a} fd={fd}"
        );
        let scale = a.abs().max(fd.abs()).max(1e-6);
        assert!(
            gap / scale < 5e-2,
            "anisotropic eta outer-gradient analytic≠FD on {block}: analytic={a:.6e} \
             fd={fd:.6e} gap={gap:.3e} rel={:.3e}",
            gap / scale
        );
    }
    assert!(
        fd_contrast < -1e-3,
        "theta0 FD eta contrast must point toward increasing the signal-axis eta; \
         got fd_signal-fd_noise={fd_contrast:.6e}"
    );
    assert!(
        analytic_contrast < -1e-3,
        "theta0 analytic eta contrast must point toward increasing the signal-axis eta; \
         got g_signal-g_noise={analytic_contrast:.6e}"
    );
}

/// #1270 regression: a single `matern(x1, x2)` 2-D smooth must fit an ordinary
/// Gaussian surface to convergence over the FULL (uncapped) κ-optimizer loop,
/// exactly like the `duchon` control below.
///
/// Root cause: the single-spatial-term "n-free penalty re-key" fast path
/// declared itself supported for Matérn, so the design-revision skip path was
/// taken. But the realized Matérn design carries the collocation operator
/// triplet (mass/tension/stiffness, #1259) while the n-free re-key rebuilds the
/// projected-kernel double-penalty — a different block topology. The block-count
/// guard rejected the rebuild, cleared the staged surface, and the next
/// skip-path eval converted "no exact S(ψ) staged" into a HARD ERROR
/// (IntegrationError), aborting the fit. `duchon`/`thinplate`/`te` were
/// unaffected because their re-key reproduces the frozen topology exactly.
///
/// The fix drops `Matern` from `supports_nfree_penalty_rekey`, routing it
/// through the slow path that re-realizes the design every trial (re-deriving
/// the correct operator triplet). This test caps NOTHING on the outer loop, so
/// it reaches the skip-window evals that armed the bug; pre-fix it aborts with
/// IntegrationError, post-fix it converges.
#[test]
fn matern_2d_smooth_fits_ordinary_surface_full_outer_loop() {
    let _guard = TEST_LOG_LOCK.lock().unwrap();
    init();
    let data = build_dataset(160, 0.05, 0x1270_0001_2D5Eu64);
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        // No outer_max_iter cap: run the FULL κ loop so the design-revision
        // skip path (the bug's trigger) is actually reached.
        gpu_policy: if cfg!(target_os = "macos") {
            gam::gpu::GpuPolicy::Off
        } else {
            gam::gpu::GpuPolicy::Auto
        },
        ..FitConfig::default()
    };

    // matern: must fit without the IntegrationError abort (#1270).
    let matern = gam::fit_from_formula("y ~ matern(x1, x2)", &data, &config);
    assert!(
        matern.is_ok(),
        "matern(x1,x2) 2-D smooth must fit an ordinary surface, but the fit \
         returned an error (#1270 regression): {:?}",
        matern.err()
    );

    // duchon control: the sibling spatial smooth that was always healthy.
    let duchon = gam::fit_from_formula("y ~ duchon(x1, x2)", &data, &config);
    assert!(
        duchon.is_ok(),
        "duchon(x1,x2) control must fit (it always did): {:?}",
        duchon.err()
    );
}

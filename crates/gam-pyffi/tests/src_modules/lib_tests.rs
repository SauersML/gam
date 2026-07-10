use super::*;
use ndarray::{array, s};

#[test]
fn pyffi_sources_use_canonical_gam_module_paths() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut source_paths = std::fs::read_dir(manifest.join("src"))
        .expect("read gam-pyffi src directory")
        .map(|entry| entry.expect("read gam-pyffi src entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .collect::<Vec<_>>();
    source_paths.push(manifest.join("tests/src_modules/lib_tests.rs"));

    let stale_paths = [
        concat!("gam::", "gaussian_reml::"),
        concat!("gam::", "basis::"),
        concat!("gam::", "smooth::"),
        concat!("gam::", "predict::"),
        concat!("gam::", "estimate::"),
        concat!("gam::", "pirls::"),
        concat!("gam::", "probability::"),
        concat!("gam::", "hmc::"),
        concat!("gam::", "gamlss::"),
        concat!("gam::", "transformation_normal::"),
        concat!("gam::", "faer_ndarray::"),
        concat!("gam::", "generative::"),
        concat!("gam::", "sample::"),
        concat!("gam::", "construction::"),
        concat!("gam::", "FitConfig"),
        concat!("gam::", "FitRequest"),
        concat!("gam::", "FitResult"),
        concat!("gam::", "WorkflowError"),
        concat!("gam::", "fit_model"),
        concat!("gam::", "materialize"),
        concat!("gam::", "resolve_offset_column"),
        concat!("gam::", "spline_scan_fast_path"),
        concat!("gam::", "DispersionLocationScaleFitResult"),
        concat!("gam::", "SurvivalLocationScaleFitResult"),
        concat!("gam::", "SurvivalTransformationFitResult"),
        concat!("use gam::", "{"),
    ];

    let mut hits = Vec::new();
    for path in source_paths {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        for stale in stale_paths {
            if text.contains(stale) {
                hits.push(format!("{} contains {stale}", path.display()));
            }
        }
    }

    assert!(
        hits.is_empty(),
        "PyFFI sources must use canonical gam module paths:\n{}",
        hits.join("\n")
    );
}

#[test]
fn sae_atom_construction_stays_in_gam_sae_2236() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let binding_fragments = [
        manifest.join("src/latent/latent_basis_and_sae_ffi.rs"),
        manifest.join("src/latent/latent_basis_and_sae_ffi_tail.rs"),
    ];
    let forbidden_definitions = [
        "struct SaeAtomBuildPlan",
        "fn sae_build_atom_plans",
        "fn sae_build_padded_basis_stacks",
        "fn sae_build_periodic_atom",
        "fn sae_build_sphere_atom",
        "fn sae_build_torus_atom",
        "fn sae_build_duchon_atom",
        "fn sae_build_euclidean_atom_with_degree",
        "fn sae_build_euclidean_atom",
    ];

    let mut hits = Vec::new();
    for path in binding_fragments {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        for definition in forbidden_definitions {
            if source.contains(definition) {
                hits.push(format!("{} defines {definition}", path.display()));
            }
        }
    }

    assert!(
        hits.is_empty(),
        "SAE atom construction is core orchestration and must remain in gam-sae:\n{}",
        hits.join("\n")
    );
}

#[test]
fn sae_fisher_metric_construction_stays_in_gam_sae_2236() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let binding_fragments = [
        manifest.join("src/latent/latent_basis_and_sae_ffi.rs"),
        manifest.join("src/latent/latent_basis_and_sae_ffi_tail.rs"),
    ];
    let forbidden_orchestration = [
        "fn row_metric_from_fisher_provenance",
        "RowMetric::output_fisher(",
        "RowMetric::output_fisher_downstream(",
        "RowMetric::behavioral_fisher(",
        "let mut u_flat",
    ];

    let mut hits = Vec::new();
    for path in binding_fragments {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        for pattern in forbidden_orchestration {
            if source.contains(pattern) {
                hits.push(format!("{} contains {pattern}", path.display()));
            }
        }
    }

    assert!(
        hits.is_empty(),
        "SAE Fisher validation, packing, and provenance are core orchestration and must remain in gam-sae:\n{}",
        hits.join("\n")
    );
}

#[test]
fn sae_fit_seed_construction_stays_in_gam_sae_2236() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = manifest.join("src/latent/latent_basis_and_sae_ffi.rs");
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let fit_inner = source
        .split_once("fn sae_manifold_fit_inner")
        .map(|(_, body)| body)
        .expect("sae_manifold_fit_inner source");
    let forbidden_orchestration = [
        "term_from_padded_blocks_with_mode(",
        "build_sae_basis_evaluators(",
        "seed_reconstruction_dispersion(",
        "SaeManifoldRho::new_shared_ard(",
        "base_term.set_fit_config(",
        "const SAE_SHARED_ARD_K_THRESHOLD",
    ];
    let hits = forbidden_orchestration
        .into_iter()
        .filter(|pattern| fit_inner.contains(pattern))
        .collect::<Vec<_>>();

    assert!(
        fit_inner.contains("build_sae_fit_seed(SaeFitSeedRequest"),
        "sae_manifold_fit_inner must delegate seed construction to the typed gam-sae entry"
    );
    assert!(
        !source.contains("sae_promotion_align_min")
            && !source.contains("beta_quantile")
            && !source.contains("align_min_from_rank"),
        "residual-promotion policy must be derived inside the typed gam-sae fit entry"
    );
    assert!(
        hits.is_empty(),
        "SAE seed construction/config/rho policy must remain in gam-sae; binding contains: {}",
        hits.join(", ")
    );
}

#[test]
fn sae_fit_report_fields_are_marshaled_without_recomputation_2236() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = manifest.join("src/latent/latent_basis_and_sae_ffi.rs");
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let fit_inner = source
        .split_once("fn sae_manifold_fit_inner")
        .map(|(_, body)| body)
        .expect("sae_manifold_fit_inner source");

    assert!(!fit_inner.contains("let active_mask: Vec<bool>"));
    assert!(!fit_inner.contains("let mut rss = 0.0_f64"));
    assert!(!fit_inner.contains("let mut tss = 0.0_f64"));
    assert!(fit_inner.contains("atom_active_mask\", active_mask"));
    assert!(fit_inner.contains("reconstruction_r2\", reconstruction_r2"));
}

#[test]
fn sae_sibling_fit_seeds_delegate_to_gam_sae_2236() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let main = std::fs::read_to_string(manifest.join("src/latent/latent_basis_and_sae_ffi.rs"))
        .expect("read main SAE binding fragment");
    let tail =
        std::fs::read_to_string(manifest.join("src/latent/latent_basis_and_sae_ffi_tail.rs"))
            .expect("read SAE binding tail");
    let minimal = tail
        .split_once("fn sae_manifold_fit_minimal")
        .and_then(|(_, body)| body.split_once("/// Out-of-sample inference"))
        .map(|(body, _)| body)
        .expect("minimal fit body");
    let stagewise = main
        .split_once("fn sae_manifold_fit_stagewise")
        .and_then(|(_, body)| body.split_once("fn sae_manifold_fit_inner"))
        .map(|(body, _)| body)
        .expect("stagewise fit body");
    let ibp = main
        .split_once("fn sae_manifold_fit_ibp")
        .and_then(|(_, body)| body.split_once("The full-batch arrow-Schur path"))
        .map(|(body, _)| body)
        .expect("IBP convenience fit body");
    let forbidden = [
        "sae_pca_seed_initial_coords(",
        "sae_build_atom_plans(",
        "sae_build_padded_basis_stacks(",
        "sae_residual_seed_logits(",
        "sae_decoder_lsq_init(",
        "term_from_padded_blocks_with_mode(",
        "seed_reconstruction_dispersion(",
    ];

    assert!(minimal.contains("build_sae_minimal_seed(SaeMinimalSeedRequest"));
    assert!(stagewise.contains("build_sae_stagewise_seed(SaeStagewiseSeedRequest"));
    assert!(ibp.contains("sae_manifold_fit("));
    for pattern in forbidden {
        assert!(
            !minimal.contains(pattern),
            "minimal binding contains {pattern}"
        );
        assert!(
            !stagewise.contains(pattern),
            "stagewise binding contains {pattern}"
        );
    }
}

#[test]
fn sae_decoder_lsq_seed_honors_softmax_top_k_support_2132() {
    let n = 4usize;
    let k_atoms = 2usize;
    let mut basis = ndarray::Array3::<f64>::zeros((k_atoms, n, 1));
    for atom_idx in 0..k_atoms {
        for row in 0..n {
            basis[[atom_idx, row, 0]] = 1.0;
        }
    }
    let z = array![[1.0], [1.0], [-1.0], [-1.0]];
    // Deliberately MODERATE logits (±0.5): the dense softmax responsibilities are
    // genuinely soft (≈[0.73, 0.27]), so every atom gets non-trivial weight on
    // every row. That is what makes this a real regression: if the top_k support
    // projection were a no-op, atoms 0 and 1 would remain coupled across all four
    // rows and the joint LSQ would fit each decoder to coth(0.5) ≈ 2.16 (the
    // symmetric dense solution), not ±1. Only projecting the responsibilities onto
    // the top-1 support (rows 0,1 → atom 0, rows 2,3 → atom 1) decouples them so
    // each atom fits exactly its two selected rows, giving ±1 (±the seed ridge,
    // spectral_scale·1e-4). With near-hard logits (e.g. ±8) the dense and
    // projected fits are indistinguishable, so such a test would pass even against
    // a no-op projection — hence the moderate magnitude here.
    let logits = array![[0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, 0.5]];

    let decoder = sae_decoder_lsq_init(
        basis.view(),
        &[1, 1],
        z.view(),
        logits.view(),
        "softmax",
        1.0,
        1.0,
        0.0,
        Some(1),
    )
    .expect("top-k softmax seed LSQ succeeds");

    // Projected onto top-1 support each atom fits only its two selected rows,
    // recovering ±1 up to the tiny mean-relative seed ridge (~1e-4).
    assert!(
        (decoder[[0, 0, 0]] - 1.0).abs() < 1.0e-3,
        "top_k=1 must fit atom 0 only on its selected positive rows (expected ≈1.0), got {}",
        decoder[[0, 0, 0]]
    );
    assert!(
        (decoder[[1, 0, 0]] + 1.0).abs() < 1.0e-3,
        "top_k=1 must fit atom 1 only on its selected negative rows (expected ≈-1.0), got {}",
        decoder[[1, 0, 0]]
    );

    // Baseline arm: the SAME data with no top-k cap (dense softmax) keeps the
    // atoms coupled, so the decoders land far from ±1 (≈±2.16). This proves the
    // projection is load-bearing on this fixture — the assertions above cannot be
    // satisfied by a no-op that ignores `top_k`.
    let decoder_dense = sae_decoder_lsq_init(
        basis.view(),
        &[1, 1],
        z.view(),
        logits.view(),
        "softmax",
        1.0,
        1.0,
        0.0,
        None,
    )
    .expect("dense softmax seed LSQ succeeds");
    assert!(
        (decoder_dense[[0, 0, 0]] - 1.0).abs() > 0.5,
        "dense (no top_k) softmax seed must stay coupled and miss ±1, got {}",
        decoder_dense[[0, 0, 0]]
    );
}

#[test]
fn symmetric_curvature_solve_preserves_negative_modes() {
    let matrix = array![[2.0, 0.0, 0.0], [0.0, -4.0, 0.0], [0.0, 0.0, -1.0e-15]];
    let rhs = array![8.0, -8.0, 1.0];
    let solved = gam::linalg::utils::solve_symmetric_vector_with_floor(&matrix, &rhs, 1.0e-3)
        .expect("indefinite symmetric curvature solve");

    assert!((solved[0] - 4.0).abs() <= 1.0e-12);
    assert!((solved[1] - 2.0).abs() <= 1.0e-12);
    assert!((solved[2] + 250.0).abs() <= 1.0e-9);
}

/// Builds a deterministic formula-table dataset for the shared-tangent
/// Gaussian REML path with two unpenalized parametric columns plus one
/// smooth, fitted via the formula `r ~ x + z + s(w)` over `D = 2` outputs.
///
/// The design comes from the RHS `x + z + s(w)` (intercept + the two parametric
/// columns + the `s(w)` smooth basis); the multi-output tangent response is the
/// separate `y` matrix below, exactly as the Python response-geometry wrapper
/// drives it (`f"{target} ~ {rhs}"` with the tangent matrix supplied out-of-band).
/// `r` is a placeholder LHS the materializer needs to parse the formula — the
/// impl discards the materialized response and uses `y`.
fn shared_tangent_formula_dataset() -> (EncodedDataset, Array2<f64>) {
    let n = 80usize;
    let headers = vec![
        "r".to_string(),
        "x".to_string(),
        "z".to_string(),
        "w".to_string(),
    ];
    let mut rows = Vec::<csv::StringRecord>::with_capacity(n);
    let mut y = Array2::<f64>::zeros((n, 2));
    for row in 0..n {
        let t = (row as f64 - 39.5) / 20.0;
        let x = (0.7 * t).sin() + 0.05 * t;
        let z = (1.1 * t).cos() - 0.1 * t * t;
        let w = t;
        // Two distinct tangent signals: each a parametric trend plus a
        // genuinely wiggly component that the smooth must absorb.
        let y0 = 0.4 * x - 0.3 * z + 0.6 * (1.8 * w).sin() + 0.02 * t;
        let y1 = -0.2 * x + 0.5 * z + 0.4 * (2.3 * w).cos() - 0.03 * t;
        // `r` is a non-constant placeholder LHS the materializer needs to parse
        // the formula; the impl discards the materialized response and uses `y`.
        // (It must vary — a constant Gaussian response is rejected up front.)
        rows.push(csv::StringRecord::from(vec![
            format!("{y0}"),
            format!("{x}"),
            format!("{z}"),
            format!("{w}"),
        ]));
        y[[row, 0]] = y0;
        y[[row, 1]] = y1;
    }
    let dataset = gam::data::encode_recordswith_inferred_schema(headers, rows)
        .expect("encode shared-tangent formula dataset");
    (dataset, y)
}

/// Regression for issue #381 adversarial review (flaw #2), carried into the
/// shared-smoothing model (issue #967): the residual-variance denominator must
/// count the FULL effective df — unpenalized columns (intercept + the
/// parametric `x`, `z`) included — matching the canonical Gaussian scale. With
/// one pooled isotropic σ² over the `D·n` stacked rows, that denominator is
/// `D·n - edf_total`, `edf_total = (K·D - penalized_rank) + penalized_edf`. The
/// pre-fix denominator used only the penalized `edf_by_block`, overstating
/// residual df and biasing σ² low.
#[test]
fn shared_tangent_sigma2_pools_and_counts_unpenalized_columns() {
    let (dataset, y) = shared_tangent_formula_dataset();
    let fit = gaussian_reml_fit_formula_dataset_impl(
        dataset,
        "r ~ x + z + s(w)".to_string(),
        y.view(),
        None,
        None,
    )
    .expect("shared-tangent REML must fit ~ x + z + s(w)");

    let d = y.ncols();
    let n = y.nrows() as f64;
    // Isotropic tangent noise: exactly one pooled scale, NOT a per-coordinate
    // σ² (a per-output scale would itself break frame equivariance).
    assert_eq!(fit.sigma2.len(), 1);
    let m = fit.edf.len();
    assert_eq!(fit.lambdas.len(), m);
    assert!(m >= 2, "an s() smooth expands to >= 2 penalty blocks");
    assert!(fit.lambdas.iter().all(|v| v.is_finite()));
    assert!(fit.edf.iter().all(|v| v.is_finite() && *v >= 0.0));

    // `~ x + z + s(w)` has exactly three unpenalized columns PER OUTPUT (the
    // intercept plus the parametric `x`, `z`); the smooth `s(w)` is fully
    // penalized. So the pooled effective df is
    //   edf_total = 3·D (unpenalized) + Σ_block edf  (shared across all D).
    const UNPENALIZED_PER_OUTPUT: f64 = 3.0;
    let penalized_edf: f64 = fit.edf.iter().sum();
    let edf_total = UNPENALIZED_PER_OUTPUT * d as f64 + penalized_edf;
    let mut ss = 0.0;
    for output in 0..d {
        for row in 0..y.nrows() {
            let resid = y[[row, output]] - fit.fitted[[row, output]];
            ss += resid * resid;
        }
    }
    let expected = ss / (d as f64 * n - edf_total);
    assert!(
        (fit.sigma2[0] - expected).abs() <= 1.0e-9 * expected.max(1.0),
        "pooled sigma2 = {} but residual scale with full pooled effective df is {expected} \
             (edf_total = {edf_total}, penalized_edf = {penalized_edf})",
        fit.sigma2[0]
    );
    // The buggy denominator omitted the unpenalized columns; pinning the strict
    // gap guards against a regression back to it.
    let buggy = ss / (d as f64 * n - penalized_edf);
    assert!(
        fit.sigma2[0] > buggy * (1.0 + 1.0e-9),
        "pooled sigma2 = {} must exceed the unpenalized-omitting estimate {buggy}",
        fit.sigma2[0]
    );
}

/// Regression for issue #381 (flaws #1/#3) carried into the shared-smoothing
/// model: λ/edf are now reported as a SHARED per-smooth vector of length `M`
/// (one smoothing parameter per formula smooth, common to every tangent
/// coordinate) — not an `(D, M)` per-coordinate grid. The fix still
/// length-checks the canonical-compacted `lambdas`/`edf_by_block` against the
/// surviving-block count so a dropped rank-0 block can never silently misalign.
#[test]
fn shared_tangent_lambda_edf_are_shared_per_smooth() {
    let (dataset, y) = shared_tangent_formula_dataset();
    let fit = gaussian_reml_fit_formula_dataset_impl(
        dataset,
        "r ~ x + z + s(w)".to_string(),
        y.view(),
        None,
        None,
    )
    .expect("shared-tangent REML must fit ~ x + z + s(w)");

    let m = fit.lambdas.len();
    assert_eq!(fit.edf.len(), m, "lambdas and edf are both per-smooth");
    assert!(m >= 2, "an s() smooth expands to >= 2 penalty blocks");
    assert!(
        fit.lambdas.iter().all(|v| v.is_finite()),
        "no cell may be NaN-padded by an off-the-end stride"
    );
    assert!(fit.edf.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!(
        fit.lambdas.iter().any(|v| *v > 0.0),
        "at least one smooth must carry an active (nonzero) smoothing parameter"
    );
}

/// Frame-equivariance regression (issue #967): with one smoothing parameter
/// shared across the tangent outputs, the multi-output fit is EXACTLY
/// equivariant under an orthogonal mix of the outputs. Fitting `Y·Rᵀ` (an
/// orthogonal rotation of the response frame) yields coefficients `B·Rᵀ`, so
/// every prediction `X·B` rotates with the frame: `predict(fit(Y·Rᵀ)) =
/// predict(fit(Y))·Rᵀ`. Per-output independent λ (the pre-fix scheme) broke
/// this — a rotation that mixed a high-curvature output with a low-curvature
/// one was smoothed differently after the mix, so the fit depended on the
/// arbitrary ambient frame. This pins the property at the solver layer, from a
/// different angle than the Python end-to-end spherical test.
#[test]
fn shared_tangent_fit_is_output_rotation_equivariant() {
    let (dataset, y) = shared_tangent_formula_dataset(); // D = 2 outputs
    // A genuine 2×2 rotation (orthogonal, det = 1) that mixes the two outputs.
    let theta = 0.6_f64;
    let (c, s) = (theta.cos(), theta.sin());
    let rot = array![[c, -s], [s, c]];
    let y_rot = y.dot(&rot.t());

    let base = gaussian_reml_fit_formula_dataset_impl(
        dataset.clone(),
        "r ~ x + z + s(w)".to_string(),
        y.view(),
        None,
        None,
    )
    .expect("base shared-tangent fit");
    let rotated = gaussian_reml_fit_formula_dataset_impl(
        dataset,
        "r ~ x + z + s(w)".to_string(),
        y_rot.view(),
        None,
        None,
    )
    .expect("rotated shared-tangent fit");

    // Equivariance of the coefficients: B(Y·Rᵀ) == B(Y)·Rᵀ to the float floor.
    let expected = base.coefficients.dot(&rot.t());
    let max_err = rotated
        .coefficients
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < 1.0e-7,
        "coefficients are not output-rotation equivariant: \
             max|B(Y·Rᵀ) - B(Y)·Rᵀ| = {max_err:.3e}"
    );

    // The shared λ is itself frame-invariant: the REML objective depends on the
    // data only through rotation-invariant quantities (the stacked residual SS
    // tr(YᵀMY) is invariant; the log-determinant terms are data-independent).
    let lam_err = base
        .lambdas
        .iter()
        .zip(rotated.lambdas.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        lam_err < 1.0e-6,
        "shared smoothing parameters are not frame-invariant: max|Δλ| = {lam_err:.3e}"
    );

    // The pooled isotropic σ² is frame-invariant as well.
    assert!(
        (base.sigma2[0] - rotated.sigma2[0]).abs() <= 1.0e-9 * base.sigma2[0].max(1.0),
        "pooled sigma2 is not frame-invariant: {} vs {}",
        base.sigma2[0],
        rotated.sigma2[0]
    );
}

/// Regression for issue #2103: a response-geometry (manifold) fit with a purely
/// PARAMETRIC RHS (`y ~ 1`, `y ~ x`) must be fittable. Before the fix the
/// shared-tangent Gaussian REML path errored with "requires at least one
/// smoothing penalty" whenever the formula carried no smooth, so the
/// intercept-only Fréchet-mean model — and any parametric tangent regression —
/// was unreachable. The fix routes a penalty-free RHS to a direct (non-REML)
/// least-squares solve on the shared tangent design at the base point.
///
/// This exercises the end-to-end response-geometry contract on an SPD problem:
/// build genuine 2×2 SPD responses, take the base point to be their intrinsic
/// Fréchet (Karcher) mean, log-map to the shared tangent, fit `r ~ 1`, and assert
/// (a) the fit SUCCEEDS (it raised before the fix) and (b) the intercept-only
/// prediction, mapped back by the exponential map, equals the intrinsic Fréchet
/// mean — verified against a direct Karcher-mean computation. The math: an
/// isotropic-tangent LSQ intercept at the Karcher mean solves
/// `mean_i log_base(Y_i) = 0`, and `exp_base(0) = base`, so the constant
/// prediction is exactly the Fréchet mean.
#[test]
fn response_geometry_parametric_only_rhs_fits_frechet_mean() {
    use gam::geometry::response_geometry::{
        ResponseManifold, response_exp_map, response_frechet_mean, response_log_map,
    };

    // A handful of genuine 2×2 SPD matrices, flattened row-major to 4 ambient
    // columns (the layout SpdManifold uses).
    let spd = ResponseManifold::Spd { n: 2 };
    let mats = [
        [1.5_f64, 0.2, 0.2, 0.9],
        [2.0, -0.3, -0.3, 1.2],
        [0.8, 0.1, 0.1, 1.7],
        [1.1, 0.05, 0.05, 0.6],
        [1.9, 0.4, 0.4, 2.3],
    ];
    let n = mats.len();
    let mut values = Array2::<f64>::zeros((n, 4));
    for (i, m) in mats.iter().enumerate() {
        for (j, &v) in m.iter().enumerate() {
            values[[i, j]] = v;
        }
    }

    // Base point = intrinsic Fréchet (Karcher) mean, computed directly — the same
    // default base point the response-geometry dispatch uses when none is given.
    let base =
        response_frechet_mean(spd, values.view(), None, 1.0e-12, 256).expect("SPD Fréchet mean");
    // Shared tangent responses at the base point (what the Python response-geometry
    // wrapper feeds the shared-tangent fit).
    let tangent = response_log_map(spd, values.view(), base.view()).expect("SPD log map");

    // Intercept-only RHS. `r` is a non-constant placeholder LHS the materializer
    // needs to parse the formula; the impl discards it and uses the `tangent`
    // matrix (a constant Gaussian LHS would be rejected up front, hence the
    // varying placeholder).
    let headers = vec!["r".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![format!("{}", i as f64 + 1.0)]))
        .collect();
    let dataset = gam::data::encode_recordswith_inferred_schema(headers, rows)
        .expect("encode parametric shared-tangent formula dataset");

    // Before #2103 this returned Err("... requires at least one smoothing
    // penalty ..."); it must now succeed via the direct non-REML LSQ path.
    let fit = gaussian_reml_fit_formula_dataset_impl(
        dataset,
        "r ~ 1".to_string(),
        tangent.view(),
        None,
        None,
    )
    .expect("parametric-only (intercept) shared-tangent fit must succeed (#2103)");

    // Intercept-only design: one basis column (the constant), D = 4 tangent
    // outputs. No smoothing penalty ⇒ empty λ/edf vectors.
    assert_eq!(fit.coefficients.dim(), (1, 4));
    assert_eq!(fit.lambdas.len(), 0);
    assert_eq!(fit.edf.len(), 0);
    assert!(fit.sigma2.iter().all(|v| v.is_finite()));

    // The intercept tangent prediction, mapped back through the exponential map,
    // must equal the intrinsic Fréchet mean of the responses.
    let intercept = fit.coefficients.row(0).to_owned();
    let predicted = response_exp_map(spd, intercept.view().insert_axis(Axis(0)), base.view())
        .expect("exp map of the intercept prediction");
    let mean_err = predicted
        .row(0)
        .iter()
        .zip(base.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        mean_err < 1.0e-8,
        "intercept-only response-geometry prediction must equal the intrinsic \
         Fréchet mean of the responses; max|pred - Karcher mean| = {mean_err:.3e}"
    );

    // The fitted tangent intercept is itself ~0 at the Karcher mean (the direct
    // LSQ intercept is the tangent mean, and Σ_i log_base(Y_i) = 0 there).
    let intercept_norm = intercept.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        intercept_norm < 1.0e-8,
        "the tangent intercept at the Karcher mean must be ~0; max|β| = {intercept_norm:.3e}"
    );
}

#[test]
fn load_model_rejects_payload_version_mismatch() {
    let model = FittedModel::from_payload(FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION + 1,
        "y ~ x".to_string(),
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            ),
            link: Some(StandardLink::Identity),
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        "gaussian".to_string(),
    ));
    let mismatched_bytes = serde_json::to_vec(&model).expect("mismatched model should serialize");

    let err = match load_model_impl(&mismatched_bytes) {
        Ok(_) => panic!("load_model_impl should reject mismatched payload versions"),
        Err(e) => e,
    };
    assert!(
        err.contains("saved model payload schema mismatch"),
        "unexpected error: {err}"
    );
}

#[derive(Clone, Copy, Debug)]
enum RemlForwardScalar {
    Lambda,
    RemlScore,
    Coefficient(usize, usize),
    Fitted(usize, usize),
}

fn by_gate_fd_design() -> Array2<f64> {
    Array2::from_shape_fn((20, 5), |(row, col)| {
        let t = (row as f64 - 9.5) / 8.0;
        match col {
            0 => 1.0,
            1 => t,
            2 => t * t - 0.45,
            3 => (0.9 * t).sin() + 0.05 * t,
            4 => (1.2 * t).cos() - 0.15 * t * t,
            _ => unreachable!(),
        }
    })
}

fn by_gate_fd_response() -> Array2<f64> {
    // The truth must NOT lie (essentially) in span(X). The design here is
    // {1, t, t² - 0.45, sin(0.9t) + 0.05t, cos(1.2t) - 0.15t²}; a smooth
    // polynomial plus low-frequency cos would be fit nearly to machine
    // precision, driving σ² → 0 and ∂(score)/∂y ≈ ν w r / dp → ∞, at which
    // point central FD with Richardson extrapolation cannot resolve the
    // (analytic, exact) gradient at 1e-6 relative because the truncation
    // term scales with f⁽⁵⁾(y). The high-frequency sin term below lies
    // outside that span on t ∈ [-1.19, 1.31] and leaves a genuine residual
    // so the analytic-vs-FD comparison is meaningful at strict tolerance.
    Array2::from_shape_fn((20, 3), |(row, output)| {
        let t = (row as f64 - 9.5) / 8.0;
        let phase = output as f64 + 1.0;
        0.2 + 0.35 * phase * t
            + 0.18 * t * t
            + (0.4 + 0.1 * phase) * (0.8 * t + 0.25 * phase).cos()
            + 0.07 * (6.5 * t + 0.4 * phase).sin()
    })
}

fn by_gate_fd_values() -> Array1<f64> {
    Array1::from_shape_fn(20, |row| {
        let t = (row as f64 - 9.5) / 8.0;
        0.85 + 0.12 * (0.7 * t).sin() + 0.03 * t
    })
}

fn by_gate_fd_weights() -> Array1<f64> {
    Array1::from_shape_fn(20, |row| {
        let t = (row as f64 - 9.5) / 8.0;
        1.0 + 0.07 * (1.3 * t).cos() + 0.025 * t
    })
}

fn by_gate_fd_penalty() -> Array2<f64> {
    Array2::from_diag(&array![0.0, 0.4, 1.1, 1.9, 3.0])
}

fn by_gate_objective(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    by: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    target: RemlForwardScalar,
    init_lambda: Option<f64>,
) -> f64 {
    let gated_x = apply_by_gate(x, by, 1).expect("by-gated design");
    let fit = gaussian_reml_multi_closed_form_with_cache(
        gated_x.view(),
        y,
        penalty,
        Some(weights),
        init_lambda,
        None,
    )
    .expect("by-gated finite-difference forward fit");
    match target {
        RemlForwardScalar::Lambda => fit.lambda,
        RemlForwardScalar::RemlScore => fit.reml_score,
        RemlForwardScalar::Coefficient(row, col) => fit.coefficients[[row, col]],
        RemlForwardScalar::Fitted(row, col) => fit.fitted[[row, col]],
    }
}

fn by_gate_backward(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    by: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    target: RemlForwardScalar,
) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut grad_coefficients = Array2::<f64>::zeros((x.ncols(), y.ncols()));
    let mut grad_fitted = Array2::<f64>::zeros(y.dim());
    let (grad_lambda, grad_score, coefficient_upstream, fitted_upstream) = match target {
        RemlForwardScalar::Lambda => (1.0, 0.0, None, None),
        RemlForwardScalar::RemlScore => (0.0, 1.0, None, None),
        RemlForwardScalar::Coefficient(row, col) => {
            grad_coefficients[[row, col]] = 1.0;
            (0.0, 0.0, Some(grad_coefficients.view()), None)
        }
        RemlForwardScalar::Fitted(row, col) => {
            grad_fitted[[row, col]] = 1.0;
            (0.0, 0.0, None, Some(grad_fitted.view()))
        }
    };
    let gated_x = apply_by_gate(x, by, 1).expect("by-gated design");
    let backward = gaussian_reml_multi_closed_form_backward(
        gated_x.view(),
        y,
        penalty,
        Some(weights),
        Some(0.85),
        grad_lambda,
        coefficient_upstream,
        fitted_upstream,
        grad_score,
        0.0,
    )
    .expect("by-gated analytic backward");
    let (grad_x, grad_by) =
        apply_by_gate_backward(x, by, 1, backward.grad_x.view()).expect("by-gate backward");
    (grad_x, backward.grad_y, grad_by, backward.grad_weights)
}

fn assert_fd_estimate_close(
    label: &str,
    analytic: f64,
    finite_difference: f64,
    finite_difference_error: f64,
) {
    let rel_tol = 1.0e-6_f64;
    let abs_tol = 1.0e-6_f64;
    // Loosen the tolerance by the caller's own FD-error estimate so the
    // assertion does not flag differences that the FD discretization
    // itself cannot resolve. The base abs/rel envelope handles
    // analytic-side rounding; `finite_difference_error` covers
    // truncation error from the FD stencil.
    let tol = abs_tol
        .max(rel_tol * analytic.abs().max(finite_difference.abs()))
        .max(finite_difference_error.abs());
    let diff = (analytic - finite_difference).abs();
    assert!(
        diff <= tol,
        "{label}: analytic={analytic:.12e}, finite_difference={finite_difference:.12e}, diff={diff:.3e}, tol={tol:.3e}"
    );
}

fn adaptive_finite_difference<F>(center: f64, step: f64, mut objective: F) -> (f64, f64)
where
    F: FnMut(f64) -> f64,
{
    let multipliers = [100.0_f64, 50.0, 25.0, 12.5, 6.25];
    let scale = step * center.abs().max(1.0);
    let mut best = f64::NAN;
    let mut best_delta = f64::INFINITY;
    let mut previous: Option<f64> = None;
    for multiplier in multipliers {
        let h = multiplier * scale;
        let coarse = (objective(center + h) - objective(center - h)) / (2.0 * h);
        let half_h = 0.5 * h;
        let fine = (objective(center + half_h) - objective(center - half_h)) / (2.0 * half_h);
        let estimate = fine + (fine - coarse) / 3.0;
        if let Some(prev) = previous {
            let delta = (estimate - prev).abs();
            if delta < best_delta {
                best_delta = delta;
                best = estimate;
            }
        } else {
            best = estimate;
        }
        previous = Some(estimate);
    }
    (best, best_delta)
}

fn blocks_reml_sign_inputs() -> (Vec<Array2<f64>>, Vec<Array2<f64>>, Array1<f64>, Array1<f64>) {
    let n = 18;
    let x1 = Array2::from_shape_fn((n, 3), |(row, col)| {
        let t = (row as f64 + 0.5) / n as f64;
        match col {
            0 => 1.0,
            1 => t - 0.45,
            2 => (4.3 * t).sin() + 0.2 * t,
            _ => unreachable!(),
        }
    });
    let x2 = Array2::from_shape_fn((n, 2), |(row, col)| {
        let t = (row as f64 + 0.5) / n as f64;
        match col {
            0 => (2.1 * t).cos() - 0.1,
            1 => (7.7 * t + 0.3).sin(),
            _ => unreachable!(),
        }
    });
    let s1 = array![[0.7, 0.08, 0.02], [0.08, 1.3, -0.04], [0.02, -0.04, 2.1],];
    let s2 = array![[0.9, -0.06], [-0.06, 1.8]];
    let y = Array1::from_shape_fn(n, |row| {
        let t = (row as f64 + 0.5) / n as f64;
        0.25 + 0.35 * t + 0.18 * (5.0 * t).sin() + 0.09 * (11.0 * t + 0.2).cos()
    });
    let weights = Array1::from_shape_fn(n, |row| {
        let t = (row as f64 + 0.5) / n as f64;
        0.9 + 0.16 * (1.7 * t).cos()
    });
    (vec![x1, x2], vec![s1, s2], y, weights)
}

fn gaussian_reml_fit_blocks_forward_native(
    designs: &[Array2<f64>],
    penalties: &[Array2<f64>],
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    init_rhos: &[f64],
) -> Result<
    (
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        f64,
        Array1<f64>,
    ),
    EstimationError,
> {
    let n_rows = designs[0].nrows();
    let mut col_offsets = vec![0usize];
    for design in designs {
        col_offsets.push(col_offsets.last().copied().unwrap() + design.ncols());
    }
    let p_total = *col_offsets.last().unwrap();
    let mut joint_x = Array2::<f64>::zeros((n_rows, p_total));
    for (block, design) in designs.iter().enumerate() {
        joint_x
            .slice_mut(s![.., col_offsets[block]..col_offsets[block + 1]])
            .assign(design);
    }
    let s_list = penalties
        .iter()
        .enumerate()
        .map(|(block, penalty)| {
            gam::terms::smooth::BlockwisePenalty::new(
                col_offsets[block]..col_offsets[block + 1],
                penalty.clone(),
            )
        })
        .collect::<Vec<_>>();
    let heuristic_lambdas = init_rhos.iter().map(|rho| rho.exp()).collect::<Vec<_>>();
    let opts = gam::solver::estimate::FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        skip_rho_posterior_inference: false,
        max_iter: 200,
        tol: 1.0e-9,
        nullspace_dims: vec![0; s_list.len()],
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
        persist_warm_start_disk: false,
    };
    let offset = Array1::<f64>::zeros(n_rows);
    let fit = gam::solver::estimate::fit_gamwith_heuristic_lambdas(
        joint_x.clone(),
        y,
        weights,
        offset.view(),
        &s_list,
        Some(heuristic_lambdas.as_slice()),
        LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        &opts,
    )?;
    let beta = fit.beta.clone();
    let fitted = joint_x.dot(&beta);
    let lambdas = fit.lambdas.clone();
    let log_lambdas = lambdas.mapv(|lambda| lambda.max(1.0e-300).ln());
    let edf_vec = fit
        .inference
        .as_ref()
        .map(|inference| inference.edf_by_block.clone())
        .unwrap_or_else(|| vec![0.0; lambdas.len()]);
    let edf = if edf_vec.len() == lambdas.len() {
        Array1::from_vec(edf_vec)
    } else {
        Array1::zeros(lambdas.len())
    };
    Ok((beta, fitted, lambdas, log_lambdas, fit.reml_score, edf))
}

fn blocks_profile_reml_score(
    designs: &[Array2<f64>],
    penalties: &[Array2<f64>],
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    init_rhos: &[f64],
) -> f64 {
    let (_, _, _, _, reml_score, _) =
        gaussian_reml_fit_blocks_forward_native(designs, penalties, y, weights, init_rhos)
            .expect("multi-block Gaussian REML profile score");
    reml_score
}

#[test]
fn blocks_negative_reml_score_backward_sign_matches_profile_perturbations() {
    let (designs, penalties, y, weights) = blocks_reml_sign_inputs();
    let init_rhos = vec![0.2, -0.4];
    let (_, _, _, log_lambdas, _, _) = gaussian_reml_fit_blocks_forward_native(
        &designs,
        &penalties,
        y.view(),
        weights.view(),
        init_rhos.as_slice(),
    )
    .expect("base multi-block Gaussian REML fit");
    let rhos = log_lambdas.to_vec();
    let backward = gam::solver::gaussian_reml::gaussian_reml_fit_blocks_backward_analytic(
        &designs,
        &penalties,
        y.view(),
        weights.view(),
        rhos.as_slice(),
        None,
        None,
        None,
        None,
        1.0,
        None,
    )
    .expect("negative REML score analytic VJP");
    let eps = 1.0e-5;

    for row in [2_usize, 11] {
        let (fd, fd_error) = adaptive_finite_difference(y[row], eps, |candidate| {
            let mut yp = y.clone();
            yp[row] = candidate;
            blocks_profile_reml_score(
                &designs,
                &penalties,
                yp.view(),
                weights.view(),
                rhos.as_slice(),
            )
        });
        assert_fd_estimate_close(
            &format!("negative REML y[{row}] sign"),
            backward.grad_y[[row, 0]],
            fd,
            fd_error,
        );
    }

    for (block, row, col) in [(0_usize, 5_usize, 2_usize), (1, 12, 1)] {
        let center = designs[block][[row, col]];
        let (fd, fd_error) = adaptive_finite_difference(center, eps, |candidate| {
            let mut xp = designs.clone();
            xp[block][[row, col]] = candidate;
            blocks_profile_reml_score(&xp, &penalties, y.view(), weights.view(), rhos.as_slice())
        });
        assert_fd_estimate_close(
            &format!("negative REML X{block}[{row},{col}] sign"),
            backward.grad_designs[block][[row, col]],
            fd,
            fd_error,
        );
    }

    for (block, row, col) in [(0_usize, 1_usize, 1_usize), (1, 0, 1)] {
        let center = penalties[block][[row, col]];
        let (fd, fd_error) = adaptive_finite_difference(center, eps, |candidate| {
            let mut sp = penalties.clone();
            sp[block][[row, col]] = candidate;
            sp[block][[col, row]] = candidate;
            blocks_profile_reml_score(&designs, &sp, y.view(), weights.view(), rhos.as_slice())
        });
        let analytic = if row == col {
            backward.grad_penalties[block][[row, col]]
        } else {
            backward.grad_penalties[block][[row, col]] + backward.grad_penalties[block][[col, row]]
        };
        assert_fd_estimate_close(
            &format!("negative REML S{block}[{row},{col}] sign"),
            analytic,
            fd,
            fd_error,
        );
    }

    for row in [3_usize, 14] {
        let (fd, fd_error) = adaptive_finite_difference(weights[row], eps, |candidate| {
            let mut wp = weights.clone();
            wp[row] = candidate;
            blocks_profile_reml_score(&designs, &penalties, y.view(), wp.view(), rhos.as_slice())
        });
        assert_fd_estimate_close(
            &format!("negative REML weight[{row}] sign"),
            backward.grad_weights[row],
            fd,
            fd_error,
        );
    }
}

fn position_fd_inputs() -> (
    Array1<f64>,
    Array2<f64>,
    Array1<f64>,
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let t = Array1::linspace(0.07, 0.93, 18);
    // The truth must NOT lie in the span of the 6-basis periodic B-spline.
    // A smooth low-frequency component alone would be fit to near machine
    // precision (σ² → 0), at which point the FD comparison cannot resolve
    // 1e-6 relative agreement against the analytic ∂(score)/∂y ∝ 1/dp. The
    // high-frequency component below leaves a genuine residual.
    let y = Array2::from_shape_fn((18, 2), |(row, col)| {
        let x = t[row];
        let phase = col as f64 + 1.0;
        0.1 + 0.4 * phase * x
            + (1.2 * x + 0.3 * phase).sin() * 0.25
            + 0.05 * (9.0 * x + 0.4 * phase).sin()
    });
    let knots = Array1::linspace(0.0, 1.0, 7);
    let penalty = Array2::from_diag(&array![0.0, 0.8, 1.1, 1.5, 2.0, 2.8]);
    let by = Array1::from_shape_fn(18, |row| 0.9 + 0.1 * (2.0 * t[row]).cos());
    let weights = Array1::from_shape_fn(18, |row| 1.0 + 0.06 * (1.4 * t[row]).sin());
    (t, y, knots, penalty, by, weights)
}

fn position_objective(
    t: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    by: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    grad_lambda: f64,
    grad_coefficients: ArrayView2<'_, f64>,
    grad_fitted: ArrayView2<'_, f64>,
    grad_reml_score: f64,
) -> f64 {
    let x = position_basis_design(t, knots, "bspline", 3, true, Some(1.0)).expect("position basis");
    let gated_x = apply_by_gate(x.view(), by, 0).expect("by gate");
    let fit = gaussian_reml_multi_closed_form_with_cache(
        gated_x.view(),
        y,
        penalty,
        Some(weights),
        Some(0.7),
        None,
    )
    .expect("position finite-difference fit");
    grad_lambda * fit.lambda
        + grad_reml_score * fit.reml_score
        + (&fit.coefficients * &grad_coefficients).sum()
        + (&fit.fitted * &grad_fitted).sum()
}

#[test]
fn position_batched_forward_matches_prebuilt_by_gated_design() {
    let (t, y, knots, penalty, by, _weights) = position_fd_inputs();
    let offsets = array![0_usize, 9_usize, 18_usize];
    let x = position_basis_design(t.view(), knots.view(), "bspline", 3, true, Some(1.0))
        .expect("position basis");
    let gated_x = apply_by_gate(x.view(), by.view(), 0).expect("by gate");

    let expected = gaussian_reml_fit_batched_impl(
        gated_x.view(),
        y.view(),
        offsets.view(),
        penalty.view(),
        None,
        Some(0.7),
    )
    .expect("prebuilt batched fit");
    let actual = gaussian_reml_fit_positions_batched_impl(
        t.view(),
        y.view(),
        offsets.view(),
        knots.view(),
        "bspline",
        3,
        true,
        Some(1.0),
        penalty.view(),
        None,
        Some(0.7),
        Some(by.view()),
        0,
    )
    .expect("position batched fit");

    assert_eq!(actual.statuses, expected.statuses);
    for b in 0..2 {
        assert!((actual.lambdas[b] - expected.lambdas[b]).abs() < 1.0e-12);
        assert!((actual.reml_scores[b] - expected.reml_scores[b]).abs() < 1.0e-12);
    }
    for (actual, expected) in actual.coefficients.iter().zip(expected.coefficients.iter()) {
        assert!((*actual - *expected).abs() < 1.0e-12);
    }
    for (actual, expected) in actual.fitted.iter().zip(expected.fitted.iter()) {
        assert!((*actual - *expected).abs() < 1.0e-12);
    }
}

#[test]
fn position_batched_duchon_forward_matches_prebuilt_design() {
    let t = Array1::linspace(0.05, 0.95, 16);
    let y = Array2::from_shape_fn((t.len(), 2), |(row, output)| {
        let u = t[row];
        let scale = output as f64 + 1.0;
        0.2 + 0.35 * scale * u + 0.12 * (5.0 * u + scale).sin()
    });
    let offsets = array![0_usize, 6_usize, 16_usize];
    let centers = Array1::linspace(0.0, 1.0, 6);
    let x = position_basis_design(t.view(), centers.view(), "duchon", 2, false, None)
        .expect("Duchon position basis");
    let penalty = Array2::from_diag(&Array1::from_shape_fn(x.ncols(), |col| {
        0.2 + 0.15 * col as f64
    }));

    let expected = gaussian_reml_fit_batched_impl(
        x.view(),
        y.view(),
        offsets.view(),
        penalty.view(),
        None,
        Some(0.8),
    )
    .expect("prebuilt Duchon batched fit");
    let actual = gaussian_reml_fit_positions_batched_impl(
        t.view(),
        y.view(),
        offsets.view(),
        centers.view(),
        "duchon",
        2,
        false,
        None,
        penalty.view(),
        None,
        Some(0.8),
        None,
        0,
    )
    .expect("streamed Duchon position batched fit");

    assert_eq!(actual.statuses, expected.statuses);
    for b in 0..2 {
        assert!((actual.lambdas[b] - expected.lambdas[b]).abs() < 1.0e-11);
        assert!((actual.reml_scores[b] - expected.reml_scores[b]).abs() < 1.0e-10);
    }
    for (actual, expected) in actual.coefficients.iter().zip(expected.coefficients.iter()) {
        assert!((*actual - *expected).abs() < 1.0e-10);
    }
    for (actual, expected) in actual.fitted.iter().zip(expected.fitted.iter()) {
        assert!((*actual - *expected).abs() < 1.0e-10);
    }
}

#[test]
fn position_backward_grad_t_y_by_and_weight_match_finite_difference() {
    let (t, y, knots, penalty, by, weights) = position_fd_inputs();
    let x = position_basis_design(t.view(), knots.view(), "bspline", 3, true, Some(1.0))
        .expect("position basis");
    let mut grad_coefficients = Array2::<f64>::zeros((x.ncols(), y.ncols()));
    grad_coefficients[[3, 1]] = -0.25;
    let grad_fitted = Array2::from_shape_fn(y.dim(), |(row, col)| {
        0.02 * (row as f64 + 1.0) - 0.03 * (col as f64 + 1.0)
    });
    let grad_lambda = 0.17;
    let grad_reml_score = -0.11;
    let backward = gaussian_reml_fit_positions_backward_impl(
        t.view(),
        y.view(),
        knots.view(),
        "bspline",
        3,
        true,
        Some(1.0),
        penalty.view(),
        Some(weights.view()),
        Some(0.7),
        grad_lambda,
        Some(grad_coefficients.view()),
        Some(grad_fitted.view()),
        grad_reml_score,
        0.0,
        Some(by.view()),
        0,
        None,
    )
    .expect("position analytic backward");
    let grad_by = backward.grad_by.expect("by gradient");
    let eps = 1.0e-5;

    for row in [2_usize, 8, 15] {
        let (fd, fd_error) = adaptive_finite_difference(t[row], eps, |candidate| {
            let mut perturbed = t.clone();
            perturbed[row] = candidate;
            position_objective(
                perturbed.view(),
                y.view(),
                by.view(),
                weights.view(),
                knots.view(),
                penalty.view(),
                grad_lambda,
                grad_coefficients.view(),
                grad_fitted.view(),
                grad_reml_score,
            )
        });
        assert_fd_estimate_close(
            &format!("position t[{row}]"),
            backward.grad_t[row],
            fd,
            fd_error,
        );
    }

    for (row, col) in [(1_usize, 0_usize), (10, 1)] {
        let (fd, fd_error) = adaptive_finite_difference(y[[row, col]], eps, |candidate| {
            let mut perturbed = y.clone();
            perturbed[[row, col]] = candidate;
            position_objective(
                t.view(),
                perturbed.view(),
                by.view(),
                weights.view(),
                knots.view(),
                penalty.view(),
                grad_lambda,
                grad_coefficients.view(),
                grad_fitted.view(),
                grad_reml_score,
            )
        });
        assert_fd_estimate_close(
            &format!("position y[{row},{col}]"),
            backward.grad_y[[row, col]],
            fd,
            fd_error,
        );
    }

    for row in [0_usize, 9, 17] {
        let (fd, fd_error) = adaptive_finite_difference(by[row], eps, |candidate| {
            let mut perturbed = by.clone();
            perturbed[row] = candidate;
            position_objective(
                t.view(),
                y.view(),
                perturbed.view(),
                weights.view(),
                knots.view(),
                penalty.view(),
                grad_lambda,
                grad_coefficients.view(),
                grad_fitted.view(),
                grad_reml_score,
            )
        });
        assert_fd_estimate_close(&format!("position by[{row}]"), grad_by[row], fd, fd_error);
    }

    for row in [1_usize, 7, 13] {
        let (fd, fd_error) = adaptive_finite_difference(weights[row], eps, |candidate| {
            let mut perturbed = weights.clone();
            perturbed[row] = candidate;
            position_objective(
                t.view(),
                y.view(),
                by.view(),
                perturbed.view(),
                knots.view(),
                penalty.view(),
                grad_lambda,
                grad_coefficients.view(),
                grad_fitted.view(),
                grad_reml_score,
            )
        });
        assert_fd_estimate_close(
            &format!("position weights[{row}]"),
            backward.grad_weights[row],
            fd,
            fd_error,
        );
    }
}

#[test]
fn by_gate_backward_matches_forward_finite_difference_for_all_x_y_gate_and_weight_entries() {
    let x = by_gate_fd_design();
    let y = by_gate_fd_response();
    let by = by_gate_fd_values();
    let weights = by_gate_fd_weights();
    let penalty = by_gate_fd_penalty();
    let targets = [
        RemlForwardScalar::Lambda,
        RemlForwardScalar::RemlScore,
        RemlForwardScalar::Coefficient(4, 2),
        RemlForwardScalar::Fitted(11, 1),
    ];
    let eps = 1.0e-5;
    let gated_x = apply_by_gate(x.view(), by.view(), 1).expect("by-gated design");
    let base_fit = gaussian_reml_multi_closed_form_with_cache(
        gated_x.view(),
        y.view(),
        penalty.view(),
        Some(weights.view()),
        Some(0.85),
        None,
    )
    .expect("base by-gated finite-difference fit");
    let fd_init_lambda = Some(base_fit.lambda);

    for target in targets {
        let fd_scale = 1.0_f64;
        let (grad_x, grad_y, grad_by, grad_weights) = by_gate_backward(
            x.view(),
            y.view(),
            by.view(),
            weights.view(),
            penalty.view(),
            target,
        );

        for row in 0..x.nrows() {
            for col in 0..x.ncols() {
                let (fd, fd_error) = adaptive_finite_difference(x[[row, col]], eps, |candidate| {
                    let mut perturbed = x.clone();
                    perturbed[[row, col]] = candidate;
                    by_gate_objective(
                        perturbed.view(),
                        y.view(),
                        by.view(),
                        weights.view(),
                        penalty.view(),
                        target,
                        fd_init_lambda,
                    )
                });
                assert_fd_estimate_close(
                    &format!("target={target:?} x[{row},{col}]"),
                    grad_x[[row, col]],
                    fd_scale * fd,
                    fd_scale.abs() * fd_error,
                );
            }
        }

        for row in 0..y.nrows() {
            for col in 0..y.ncols() {
                let (fd, fd_error) = adaptive_finite_difference(y[[row, col]], eps, |candidate| {
                    let mut perturbed = y.clone();
                    perturbed[[row, col]] = candidate;
                    by_gate_objective(
                        x.view(),
                        perturbed.view(),
                        by.view(),
                        weights.view(),
                        penalty.view(),
                        target,
                        fd_init_lambda,
                    )
                });
                assert_fd_estimate_close(
                    &format!("target={target:?} y[{row},{col}]"),
                    grad_y[[row, col]],
                    fd_scale * fd,
                    fd_scale.abs() * fd_error,
                );
            }
        }

        for row in 0..by.len() {
            let (fd, fd_error) = adaptive_finite_difference(by[row], eps, |candidate| {
                let mut perturbed = by.clone();
                perturbed[row] = candidate;
                by_gate_objective(
                    x.view(),
                    y.view(),
                    perturbed.view(),
                    weights.view(),
                    penalty.view(),
                    target,
                    fd_init_lambda,
                )
            });
            assert_fd_estimate_close(
                &format!("target={target:?} by[{row}]"),
                grad_by[row],
                fd_scale * fd,
                fd_scale.abs() * fd_error,
            );
        }

        for row in 0..weights.len() {
            let (fd, fd_error) = adaptive_finite_difference(weights[row], eps, |candidate| {
                let mut perturbed = weights.clone();
                perturbed[row] = candidate;
                by_gate_objective(
                    x.view(),
                    y.view(),
                    by.view(),
                    perturbed.view(),
                    penalty.view(),
                    target,
                    fd_init_lambda,
                )
            });
            assert_fd_estimate_close(
                &format!("target={target:?} weights[{row}]"),
                grad_weights[row],
                fd_scale * fd,
                fd_scale.abs() * fd_error,
            );
        }
    }
}

#[test]
fn batched_state_round_trip_matches_refit() {
    // Document the Task 3 state round-trip contract on the BATCHED
    // pyffi entry: backward called with `forward_state` set to the dict
    // returned by the matching forward must produce gradients that are
    // bit-exact identical to backward called without `forward_state`
    // (which refits internally). Guards against drift between
    // `_from_fit` and `_backward` for the batched path under any future
    // change to either branch.
    use ndarray::{array, s};

    let x = array![
        [1.0, -1.0],
        [1.0, -0.5],
        [1.0, 0.0],
        [1.0, 0.5],
        [1.0, 1.0],
        [1.0, -0.8],
        [1.0, -0.2],
        [1.0, 0.3],
        [1.0, 0.7],
        [1.0, 1.2],
    ];
    let y = array![
        [0.4, -0.2],
        [0.6, 0.1],
        [0.9, 0.3],
        [1.3, 0.4],
        [1.8, 0.6],
        [0.5, -0.1],
        [0.7, 0.2],
        [1.0, 0.3],
        [1.4, 0.5],
        [1.9, 0.7]
    ];
    let weights = array![1.0, 0.95, 1.05, 1.0, 0.98, 1.02, 0.99, 1.0, 1.01, 0.97];
    let row_offsets = array![0usize, 5, 10];
    let penalty = array![[0.0, 0.0], [0.0, 1.0]];
    let init_lambda = Some(0.85_f64);

    let forward = gaussian_reml_fit_batched_impl(
        x.view(),
        y.view(),
        row_offsets.view(),
        penalty.view(),
        Some(weights.view()),
        init_lambda,
    )
    .expect("batched forward");
    let prebuilt_fits = (0..(row_offsets.len() - 1))
        .map(|b| {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            let cache = gam::solver::gaussian_reml::GaussianRemlEigenCache {
                penalty_eigenvalues: forward
                    .cache_penalty_eigenvalues
                    .slice(s![b, ..])
                    .to_owned(),
                eigenvectors: forward.cache_eigenvectors.slice(s![b, .., ..]).to_owned(),
                coefficient_basis: forward
                    .cache_coefficient_basis
                    .slice(s![b, .., ..])
                    .to_owned(),
                xtwx_fingerprint: forward.cache_xtwx_fingerprints[b],
                penalty_fingerprint: forward.cache_penalty_fingerprints[b],
                logdet_xtwx: forward.cache_logdet_xtwx[b],
                logdet_penalty_positive: forward.cache_logdet_penalty_positive[b],
                penalty_rank: forward.cache_penalty_ranks[b] as usize,
                nullity: forward.cache_nullities[b] as usize,
            };
            Some(gam::solver::gaussian_reml::GaussianRemlMultiResult {
                lambda: forward.lambdas[b],
                rho: forward.rhos[b],
                coefficients: forward.coefficients.slice(s![b, .., ..]).to_owned(),
                fitted: forward.fitted.slice(s![start..end, ..]).to_owned(),
                reml_score: forward.reml_scores[b],
                reml_grad_lambda: forward.reml_grad_lambdas[b],
                reml_hess_lambda: forward.reml_hess_lambdas[b],
                reml_grad_rho: forward.reml_grad_rhos[b],
                reml_hess_rho: forward.reml_hess_rhos[b],
                edf: forward.edf[b],
                sigma2: forward.sigma2.slice(s![b, ..]).to_owned(),
                cache,
            })
        })
        .collect::<Vec<_>>();

    let grad_lambda = array![0.2, -0.1];
    let grad_reml_score = array![-0.05, 0.08];

    let refit = gaussian_reml_fit_batched_backward_impl(
        x.view(),
        y.view(),
        row_offsets.view(),
        penalty.view(),
        Some(weights.view()),
        init_lambda,
        Some(grad_lambda.view()),
        None,
        None,
        Some(grad_reml_score.view()),
        None,
        None,
    )
    .expect("refit backward");
    let from_fits = gaussian_reml_fit_batched_backward_impl(
        x.view(),
        y.view(),
        row_offsets.view(),
        penalty.view(),
        Some(weights.view()),
        init_lambda,
        Some(grad_lambda.view()),
        None,
        None,
        Some(grad_reml_score.view()),
        None,
        Some(&prebuilt_fits),
    )
    .expect("from_fits backward");

    for (a, b) in refit.grad_x.iter().zip(from_fits.grad_x.iter()) {
        assert!((a - b).abs() <= 1.0e-12);
    }
    for (a, b) in refit.grad_y.iter().zip(from_fits.grad_y.iter()) {
        assert!((a - b).abs() <= 1.0e-12);
    }
    for (a, b) in refit.grad_weights.iter().zip(from_fits.grad_weights.iter()) {
        assert!((a - b).abs() <= 1.0e-12);
    }
}

/// CV-fold partitioning contract used by the benchmark suite:
/// every row appears in exactly one test fold and not in its own
/// train set, both partitions are non-empty per fold, and the
/// stratified path balances class counts across folds.
#[test]
fn make_folds_indices_kfold_partitions_unstratified() {
    let n = 50usize;
    let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let folds = make_folds_indices(y, 5, 7, false).expect("5-fold should succeed");
    assert_eq!(folds.len(), 5);

    let mut seen = vec![0usize; n];
    for (train, test) in &folds {
        assert!(!train.is_empty(), "every fold has a non-empty train set");
        assert!(!test.is_empty(), "every fold has a non-empty test set");
        let train_set: std::collections::HashSet<usize> = train.iter().copied().collect();
        for &i in test {
            assert!(
                !train_set.contains(&i),
                "row {i} appears in both train and test of a fold"
            );
            seen[i] += 1;
        }
        assert_eq!(
            train.len() + test.len(),
            n,
            "train + test must cover the full row set"
        );
    }
    for (i, &count) in seen.iter().enumerate() {
        assert_eq!(count, 1, "row {i} must appear in exactly one test fold");
    }
}

#[test]
fn make_folds_indices_stratified_balances_classes() {
    // 30 positives, 20 negatives.
    let mut y = vec![1.0; 30];
    y.extend(std::iter::repeat(0.0).take(20));
    let folds = make_folds_indices(y, 5, 11, true).expect("stratified 5-fold");
    assert_eq!(folds.len(), 5);

    for (_, test) in &folds {
        let positives = test.iter().filter(|&&i| i < 30).count();
        let negatives = test.iter().filter(|&&i| i >= 30).count();
        // 30 / 5 = 6 positives per fold, 20 / 5 = 4 negatives.
        assert_eq!(positives, 6, "stratified fold positive count");
        assert_eq!(negatives, 4, "stratified fold negative count");
    }
}

#[test]
fn make_folds_indices_holdout_partitions_with_split() {
    let n = 25usize;
    let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let folds = make_folds_indices(y, 1, 42, false).expect("holdout split");
    assert_eq!(folds.len(), 1);
    let (train, test) = &folds[0];
    assert!(!train.is_empty());
    assert!(!test.is_empty());
    assert_eq!(train.len() + test.len(), n);
    // 1/5 holdout convention: n/5 = 5 rows in test.
    assert_eq!(test.len(), 5);
    // Train and test must be disjoint.
    let train_set: std::collections::HashSet<usize> = train.iter().copied().collect();
    assert!(test.iter().all(|&i| !train_set.contains(&i)));
}

#[test]
fn make_folds_indices_seed_determinism_and_variation() {
    let n = 40usize;
    let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let a = make_folds_indices(y.clone(), 5, 17, false).expect("seed=17");
    let b = make_folds_indices(y.clone(), 5, 17, false).expect("seed=17 (repeat)");
    let c = make_folds_indices(y, 5, 18, false).expect("seed=18");
    assert_eq!(a, b, "same seed must reproduce the same fold layout");
    assert_ne!(a, c, "different seeds should produce different layouts");
}

#[test]
fn gaussian_log_loss_value_matches_closed_form() {
    // log-loss = mean of 0.5·log(2π σ²) + 0.5·((y-μ)/σ)²
    // With y = μ everywhere, the squared-residual term vanishes and the
    // loss is exactly 0.5·log(2π σ²).
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let mu = vec![1.0, 2.0, 3.0, 4.0];
    let sigma_scalar = vec![1.5];
    let got = gaussian_log_loss_value(&y, &mu, &sigma_scalar, 1.0e-12).expect("log-loss");
    let expected = 0.5 * (std::f64::consts::TAU * 1.5 * 1.5).ln();
    assert!(
        (got - expected).abs() < 1.0e-12,
        "log-loss with y=μ should equal 0.5·log(2π σ²) = {expected}, got {got}"
    );

    // Per-row σ matches the scalar broadcast when all entries agree.
    let sigma_vec = vec![1.5; y.len()];
    let got_vec = gaussian_log_loss_value(&y, &mu, &sigma_vec, 1.0e-12).expect("vec sigma");
    assert!((got_vec - got).abs() < 1.0e-12);
}

#[test]
fn gaussian_log_loss_value_rejects_invalid_sigma_length() {
    let y = vec![0.0, 1.0, 2.0];
    let mu = vec![0.0, 1.0, 2.0];
    let bad_sigma = vec![1.0, 2.0]; // length 2 with n=3
    assert!(gaussian_log_loss_value(&y, &mu, &bad_sigma, 1.0e-12).is_err());
}

#[test]
fn make_folds_indices_rejects_invalid_inputs() {
    // n_splits == 0
    assert!(make_folds_indices(vec![0.0, 1.0], 0, 0, false).is_err());
    // empty y
    assert!(make_folds_indices(Vec::<f64>::new(), 5, 0, false).is_err());
    // n < n_splits (kfold)
    assert!(make_folds_indices(vec![0.0, 1.0], 5, 0, false).is_err());
    // non-finite y
    assert!(make_folds_indices(vec![0.0, f64::NAN, 1.0], 2, 0, false).is_err());
    assert!(make_folds_indices(vec![0.0, f64::INFINITY, 1.0], 2, 0, false).is_err());
}

/// Ordinary least squares R^2 of fitting `design @ beta ~= y` (single output),
/// via the normal equations with a tiny relative ridge floor on the Gram for
/// numerical robustness. Used by the issue #876 latent-decoder regression below
/// to quantify how well a candidate latent Duchon basis reconstructs a clean
/// circular signal. NOT a tolerance to weaken — it is the recovery metric.
fn latent_decoder_ols_r2(design: &Array2<f64>, y: &Array1<f64>) -> f64 {
    let gram = design.t().dot(design);
    let rhs = design.t().dot(y);
    let beta = gam::linalg::utils::solve_symmetric_vector_with_floor(&gram, &rhs, 1.0e-10)
        .expect("OLS normal-equations solve for latent decoder recovery");
    let fitted = design.dot(&beta);
    let mean = y.sum() / y.len() as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (yi, fi) in y.iter().zip(fitted.iter()) {
        ss_res += (yi - fi) * (yi - fi);
        ss_tot += (yi - mean) * (yi - mean);
    }
    1.0 - ss_res / ss_tot
}

/// Issue #876: when the latent optimizer retracts on a PERIODIC manifold
/// (circle, radians wrapped to [-pi, pi), period TAU), the latent Duchon decoder
/// MUST be a function ON the circle. The pre-fix open Euclidean basis violated
/// `Phi(theta) = Phi(theta + TAU)` and measured the kernel distance across the
/// seam (theta = pi - eps vs -pi + eps, adjacent on the circle) as ~2*pi apart.
///
/// This pins both halves of the fix:
///   (a) seam consistency: the PERIODIC basis satisfies `Phi(theta) ~= Phi(theta
///       + TAU)` row-for-row, while the OPEN basis does not;
///   (b) recovery: on a clean circular signal `y = cos(theta) + 0.5 sin(2 theta)`
///       sampled across the WHOLE circle (including the seam), the periodic
///       decoder reconstructs the angle structure (high R^2), whereas the open
///       decoder is materially worse near the seam.
///
/// `latent_manifold_periodic_descriptor("circle", 1)` is the exact descriptor the
/// optimizer feeds the decoder, so this exercises the production path.
#[test]
fn issue_876_periodic_latent_duchon_decoder_is_seam_consistent_and_recovers_circle() {
    let tau = std::f64::consts::TAU;
    let latent_dim = 1usize;
    let m = 2usize;

    // Radian centers spanning the circle, matching the optimizer's chart and the
    // periodic eigenmap seed (`latent_periodic_seed_start`), which both live in
    // [-pi, pi). Deliberately include centers near both seam edges.
    let n_centers = 12usize;
    let mut centers = Array2::<f64>::zeros((n_centers, latent_dim));
    for k in 0..n_centers {
        // Evenly spaced angles in [-pi, pi).
        centers[[k, 0]] = -std::f64::consts::PI + tau * (k as f64) / (n_centers as f64);
    }

    // The descriptor the production optimizer derives for a 1-D circle chart.
    let descriptor = latent_manifold_periodic_descriptor("circle", latent_dim)
        .expect("circle manifold must yield a periodic descriptor");
    assert_eq!(descriptor, vec![Some(tau)]);

    // (a) Seam consistency. Build the design at sample angles theta and again at
    // theta + TAU (same point on the circle). For the periodic decoder the two
    // designs must agree row-for-row; the open Euclidean decoder must not.
    let n_obs = 40usize;
    let theta: Vec<f64> = (0..n_obs)
        .map(|i| -std::f64::consts::PI + tau * (i as f64 + 0.5) / (n_obs as f64))
        .collect();
    let theta_shift: Vec<f64> = theta.iter().map(|&a| a + tau).collect();

    let t_flat = Array1::from(theta.clone());
    let t_flat_shift = Array1::from(theta_shift.clone());

    let (design_per, _) = build_latent_duchon_design(
        t_flat.view(),
        n_obs,
        latent_dim,
        centers.view(),
        m,
        Some(descriptor.as_slice()),
    )
    .expect("periodic latent Duchon design");
    let (design_per_shift, _) = build_latent_duchon_design(
        t_flat_shift.view(),
        n_obs,
        latent_dim,
        centers.view(),
        m,
        Some(descriptor.as_slice()),
    )
    .expect("periodic latent Duchon design (shifted by TAU)");

    assert_eq!(design_per.dim(), design_per_shift.dim());
    let mut max_per_seam_gap = 0.0_f64;
    for (a, b) in design_per.iter().zip(design_per_shift.iter()) {
        max_per_seam_gap = max_per_seam_gap.max((a - b).abs());
    }
    // Periodic decoder is a genuine function on the circle: Phi(theta) = Phi(theta + TAU).
    assert!(
        max_per_seam_gap <= 1.0e-8,
        "periodic latent decoder must satisfy Phi(theta) = Phi(theta + TAU); \
         max row gap was {max_per_seam_gap}"
    );

    // The OPEN Euclidean decoder (None) is NOT periodic: it must visibly differ.
    let (design_open, _) =
        build_latent_duchon_design(t_flat.view(), n_obs, latent_dim, centers.view(), m, None)
            .expect("open Euclidean latent Duchon design");
    let (design_open_shift, _) = build_latent_duchon_design(
        t_flat_shift.view(),
        n_obs,
        latent_dim,
        centers.view(),
        m,
        None,
    )
    .expect("open Euclidean latent Duchon design (shifted by TAU)");
    let mut max_open_seam_gap = 0.0_f64;
    for (a, b) in design_open.iter().zip(design_open_shift.iter()) {
        max_open_seam_gap = max_open_seam_gap.max((a - b).abs());
    }
    assert!(
        max_open_seam_gap > 1.0e-3,
        "open Euclidean latent decoder must NOT be periodic (control); \
         max row gap was {max_open_seam_gap}"
    );

    // (b) Recovery on a clean circular signal across the whole circle, including
    // the seam. The truth is a genuine function on the circle.
    let y: Array1<f64> = Array1::from_iter(theta.iter().map(|&a| a.cos() + 0.5 * (2.0 * a).sin()));

    let r2_periodic = latent_decoder_ols_r2(&design_per, &y);
    let r2_open = latent_decoder_ols_r2(&design_open, &y);

    // The periodic decoder recovers the angle structure (not collapsed).
    assert!(
        r2_periodic >= 0.95,
        "periodic latent decoder must recover the circular signal; R^2 = {r2_periodic}"
    );

    // Compare reconstruction error specifically at the seam-adjacent points (the
    // first and last sample angles, which straddle theta = +/- pi). The periodic
    // decoder must not have the cross-seam discontinuity the open basis carries.
    let solve_fitted = |design: &Array2<f64>| -> Array1<f64> {
        let gram = design.t().dot(design);
        let rhs = design.t().dot(&y);
        let beta = gam::linalg::utils::solve_symmetric_vector_with_floor(&gram, &rhs, 1.0e-10)
            .expect("seam OLS solve");
        design.dot(&beta)
    };
    let fitted_per = solve_fitted(&design_per);
    let fitted_open = solve_fitted(&design_open);
    // Seam-straddling pair: last sample (just below +pi) and first (just above -pi).
    let seam_err = |fitted: &Array1<f64>| -> f64 {
        let e_first = (fitted[0] - y[0]).abs();
        let e_last = (fitted[n_obs - 1] - y[n_obs - 1]).abs();
        e_first.max(e_last)
    };
    assert!(
        seam_err(&fitted_per) <= seam_err(&fitted_open) + 1.0e-9,
        "periodic decoder must reconstruct seam-adjacent points at least as well \
         as the open decoder: periodic seam err = {}, open seam err = {}, R^2 open = {r2_open}",
        seam_err(&fitted_per),
        seam_err(&fitted_open),
    );
}

/// Regression for #2033: a model fitted with prior weights must retain its
/// weight column through the prediction/replicate projection. The #2025 fix
/// made `generative_replicates_impl` resolve per-row weights, but the column was
/// projected away by `project_frame_to_model_columns` (its consumable set kept
/// only offset + response, not the weight), so `Model.sample_replicates` raised
/// "weights column 'w' not found in data" for EVERY weighted model — even when
/// the caller's frame carried the column. The consumable set must include the
/// model's weight column, and projection must keep it while still dropping
/// unrelated bookkeeping columns.
#[test]
fn weighted_model_projection_retains_weight_column_2033() {
    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        "y ~ x".to_string(),
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            ),
            link: Some(StandardLink::Identity),
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        "gaussian".to_string(),
    );
    payload.weight_column = Some("w".to_string());
    let model = FittedModel::from_payload(payload);

    // The consumable-column contract must include the weight column so it
    // survives projection when the model was fitted with weights.
    let consumable = prediction_consumable_columns(&model)
        .expect("consumable columns should resolve for a well-formed model");
    assert!(
        consumable.contains("w"),
        "weight column 'w' must be in the consumable set, got {consumable:?}"
    );

    // Projecting a frame that carries the weight column (plus an unrelated `id`
    // column) must keep `w` and drop only `id`.
    let headers = vec![
        "y".to_string(),
        "x".to_string(),
        "w".to_string(),
        "id".to_string(),
    ];
    let rows = vec![
        vec![
            "1.0".to_string(),
            "0.5".to_string(),
            "2.0".to_string(),
            "a".to_string(),
        ],
        vec![
            "2.0".to_string(),
            "1.5".to_string(),
            "3.0".to_string(),
            "b".to_string(),
        ],
    ];
    let (kept_headers, kept_rows) =
        project_frame_to_model_columns(&model, &headers, &rows).expect("projection should succeed");
    assert!(
        kept_headers.contains(&"w".to_string()),
        "projection must retain the weight column, kept {kept_headers:?}"
    );
    assert!(
        !kept_headers.contains(&"id".to_string()),
        "projection must drop the unrelated id column, kept {kept_headers:?}"
    );
    for row in &kept_rows {
        assert_eq!(
            row.len(),
            kept_headers.len(),
            "each projected row width must match the retained header count"
        );
    }
}

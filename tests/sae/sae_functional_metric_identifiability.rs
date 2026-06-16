//! #980 verification §4 / #981 Theorem-2 arm: **the functional metric is an
//! identifiability technology**.
//!
//! Rung 4 of the gauge-reduction ladder claims: with the isometry pin computed
//! in the model's own output-Fisher pullback metric, the within-atom frame
//! freedom that survives a *Euclidean* pin (`Isom(M_k)` rotations — replicate
//! fits visibly differ by them) is cut to the symmetry group of the downstream
//! readout, which is **generically trivial** — the fit is identified up to atom
//! permutation alone.
//!
//! This test realises the two arms at the certificate level with the pin root
//! derived **honestly from the metric** (not hand-fed): the isometry penalty
//! pins the pulled-back gram `G(F) = Fᵀ W F` to a reference, so its curvature
//! root along frame perturbations is the gram derivative
//!
//! ```text
//! R[(a,b), (i,c)] = δ_{cb} (W F)_{ia} + δ_{ca} (W F)_{ib}
//! ```
//!
//! — the same formula for both arms, with only `W` differing:
//!
//! * **Euclidean arm** `W = I`: for an orthonormal frame the gram derivative
//!   along the `so(2)` rotation `Ξ = F·A` is `ΞᵀF + FᵀΞ = Aᵀ + A = 0` — the
//!   rotation orbit is exactly flat, the certificate must report the rotation
//!   **unpinned** (this is "euclidean replicates disagree up to rotation").
//! * **Functional arm** `W = U Uᵀ` anisotropic (a generic readout): the same
//!   derivative is `AᵀW + WA ≠ 0` — the rotation orbit costs penalty, the
//!   certificate must report it **pinned**, leaving no residual freedom
//!   (single atom ⇒ "identified up to atom permutation" is the trivial group).
//!
//! The functional arm is also the *mixed-generator* regime the verdict rule
//! must get right: the rotation's relative curvature fraction is strictly
//! interior (= 9/128 here) — partial curvature. A rank-increase test would
//! call that a surviving freedom (under-claiming identification); the
//! relative-curvature rule must call it pinned and report the fraction.

use gam::inference::row_metric::{MetricProvenance, RowMetric};
use gam::identifiability::sae::{
    AtomTopology, FittedAtom, FittedSaeManifold, GENERATOR_FLAT_ENERGY_TOL, GeneratorFamily,
    residual_gauge,
};
use ndarray::Array2;
use std::sync::Arc;

/// Curvature root of the isometry pin on `G(F) = Fᵀ W F`: one row per gram
/// entry `(a, b)` with `a ≤ b`, `R[(a,b), (i,c)] = δ_{cb}(WF)_{ia} + δ_{ca}(WF)_{ib}`,
/// in the certificate's row-major flattened frame layout (`frame[i, c]` at
/// column `i·d + c`). This is the honest lowering of the penalty: identical
/// construction for both arms, the metric `W` is the only input that differs.
fn isometry_gram_derivative_root(frame: &Array2<f64>, w: &Array2<f64>) -> Array2<f64> {
    let (p, d) = frame.dim();
    assert_eq!(w.dim(), (p, p), "W must be (p, p)");
    let wf = w.dot(frame); // (p, d)
    let n_rows = d * (d + 1) / 2;
    let mut root = Array2::<f64>::zeros((n_rows, p * d));
    let mut row = 0usize;
    for a in 0..d {
        for b in a..d {
            for i in 0..p {
                root[[row, i * d + b]] += wf[[i, a]];
                root[[row, i * d + a]] += wf[[i, b]];
            }
            row += 1;
        }
    }
    root
}

/// One 2-latent-axis patch atom with the orthonormal identity frame in ℝ²
/// (p = 2, d = 2, param_dim = 4). Its `so(2)` rotation generator is
/// `ξ = [0, 1, −1, 0]` in the flattened layout.
fn identity_frame_atom() -> FittedAtom {
    FittedAtom {
        name: "patch".to_string(),
        topology: AtomTopology::EuclideanPatch { latent_dim: 2 },
        frame: Array2::<f64>::eye(2),
        ard_variances: None,
        lowering_error: 0.0,
        chart_canonicalized: false,
        inner_fit: None,
    }
}

#[test]
fn euclidean_pin_leaves_the_rotation_free_functional_pin_identifies() {
    let frame = Array2::<f64>::eye(2);

    // ---- Euclidean arm: W = I, Euclidean provenance --------------------
    let w_euc = Array2::<f64>::eye(2);
    let model_euc = FittedSaeManifold {
        atoms: vec![identity_frame_atom()],
        jacobian_rows: Vec::new(),
        isometry_penalty_root: isometry_gram_derivative_root(&frame, &w_euc),
        metric: RowMetric::euclidean(1, 2).expect("euclidean metric"),
    };
    let report_euc = residual_gauge(&model_euc).expect("euclidean certificate");

    assert_eq!(report_euc.metric_provenance, MetricProvenance::Euclidean);
    assert!(
        !report_euc.diffeomorphism_unpinned,
        "the pin root is active in both arms — no escalation"
    );
    let rot_euc = report_euc
        .generators
        .iter()
        .find(|g| g.family == GeneratorFamily::IsomAtom)
        .expect("Isom(M_k) rotation generator");
    assert!(
        rot_euc.unpinned,
        "under the Euclidean pin the so(2) rotation orbit is exactly flat \
         (AᵀI + IA = 0 for an orthonormal frame) — euclidean replicates \
         disagree up to rotation. {}",
        report_euc.summary
    );
    assert!(
        rot_euc.pinned_energy_fraction <= GENERATOR_FLAT_ENERGY_TOL,
        "the Euclidean-arm rotation must be flat to numerical noise, got \
         pinned energy fraction {}",
        rot_euc.pinned_energy_fraction
    );
    assert!(
        report_euc.residual_gauge_dim >= 1,
        "the Euclidean arm must leave the rotation freedom standing. {}",
        report_euc.summary
    );
    assert!(
        report_euc.group_signature().contains("Isom(M_k)"),
        "the certified group must name the surviving rotation, got: {}",
        report_euc.group_signature()
    );

    // ---- Functional arm: W = U Uᵀ, OutputFisher provenance -------------
    // U = diag(1, 2): a generic anisotropic readout (W = diag(1, 4)). The
    // SAME gram-derivative pin, evaluated in this metric, gives the rotation
    // partial curvature (AᵀW + WA ≠ 0).
    let mut w_fisher = Array2::<f64>::zeros((2, 2));
    w_fisher[[0, 0]] = 1.0;
    w_fisher[[1, 1]] = 4.0;
    // Per-row factor U_n = [[1, 0], [0, 2]], row-major flattened (p·rank = 4).
    let mut u_flat = Array2::<f64>::zeros((1, 4));
    u_flat[[0, 0]] = 1.0;
    u_flat[[0, 3]] = 2.0;
    let model_fn = FittedSaeManifold {
        atoms: vec![identity_frame_atom()],
        jacobian_rows: Vec::new(),
        isometry_penalty_root: isometry_gram_derivative_root(&frame, &w_fisher),
        metric: RowMetric::output_fisher(Arc::new(u_flat), 2, 2).expect("output-fisher metric"),
    };
    let report_fn = residual_gauge(&model_fn).expect("functional certificate");

    assert!(
        matches!(
            report_fn.metric_provenance,
            MetricProvenance::OutputFisher { .. }
        ),
        "functional-arm provenance must be OutputFisher, got {:?}",
        report_fn.metric_provenance
    );
    let rot_fn = report_fn
        .generators
        .iter()
        .find(|g| g.family == GeneratorFamily::IsomAtom)
        .expect("Isom(M_k) rotation generator");
    assert!(
        !rot_fn.unpinned,
        "under the anisotropic output-Fisher pin the rotation orbit costs \
         penalty (AᵀW + WA ≠ 0) — the certificate must pin it (the Theorem-2 \
         identification). {}",
        report_fn.summary
    );
    // The mixed-generator regime, explicitly: the rotation is PARTIALLY
    // curved: the curvature-root rows are {[2,0,0,0],[0,1,4,0],[0,0,0,8]}
    // (mutually orthogonal, σ_max = 8) and the unit rotation ξ̂ = [0,1,−1,0]/√2
    // has ‖Rξ̂‖² = (−3/√2)² = 4.5, so the relative curvature fraction is
    // 4.5/64 = 9/128 ≈ 0.070 for this fixture — strictly
    // between the noise floor and full pinning. A rank-increase verdict
    // would have called this a surviving freedom.
    assert!(
        rot_fn.pinned_energy_fraction > GENERATOR_FLAT_ENERGY_TOL
            && rot_fn.pinned_energy_fraction < 1.0,
        "the functional-arm rotation must sit in the mixed regime, got \
         pinned energy fraction {}",
        rot_fn.pinned_energy_fraction
    );
    assert!(
        (rot_fn.pinned_energy_fraction - 9.0 / 128.0).abs() < 1.0e-9,
        "analytic relative curvature fraction is 9/128, got {}",
        rot_fn.pinned_energy_fraction
    );
    assert_eq!(
        report_fn.residual_gauge_dim, 0,
        "the functional arm must be fully identified — up to atom \
         permutation, trivially here (single atom). {}",
        report_fn.summary
    );
    // No permutation generator exists (single atom), so the Sym(F) check
    // under OutputFisher holds vacuously-but-honestly.
    assert_eq!(report_fn.sym_f_trivial_under_output_fisher, Some(true));

    // ---- the contrast itself: same planted atom, same pin construction,
    // different metric ⇒ genuinely different certified groups --------------
    assert_ne!(
        report_euc.group_signature(),
        report_fn.group_signature(),
        "the two metrics must certify different residual gauge groups — \
         that difference IS the Theorem-2 content.\neuclidean: {}\nfunctional: {}",
        report_euc.summary,
        report_fn.summary
    );
}

/// #995 falsification: the SAME mixed-curvature fixture (relative curvature
/// fraction 9/128 along the rotation) must flip its verdict with the lowering
/// fidelity — and ONLY with it:
///
/// * `lowering_error = 0` (exact frames, as every hand-built fixture is): the
///   partial curvature is real, the rotation is **pinned** — the strict
///   Theorem-2 verdict above.
/// * `lowering_error = 0.5 > 9/128`: the same numerical curvature is now
///   within what the mean-frame compression cannot distinguish from gauge
///   motion, so the calibrated verdict must report the rotation **unpinned**
///   and carry the scale on the verdict — the certificate refuses to claim a
///   pin it cannot resolve, instead of laundering compression error as
///   identification.
#[test]
fn lowering_error_calibration_forgives_compression_scale_curvature_only() {
    let frame = Array2::<f64>::eye(2);
    let mut w_fisher = Array2::<f64>::zeros((2, 2));
    w_fisher[[0, 0]] = 1.0;
    w_fisher[[1, 1]] = 4.0;
    let mut u_flat = Array2::<f64>::zeros((1, 4));
    u_flat[[0, 0]] = 1.0;
    u_flat[[0, 3]] = 2.0;

    let make_model = |lowering_error: f64| FittedSaeManifold {
        atoms: vec![FittedAtom {
            name: "patch".to_string(),
            topology: AtomTopology::EuclideanPatch { latent_dim: 2 },
            frame: Array2::<f64>::eye(2),
            ard_variances: None,
            lowering_error,
            chart_canonicalized: false,
            inner_fit: None,
        }],
        jacobian_rows: Vec::new(),
        isometry_penalty_root: isometry_gram_derivative_root(&frame, &w_fisher),
        metric: RowMetric::output_fisher(Arc::new(u_flat.clone()), 2, 2)
            .expect("output-fisher metric"),
    };

    // Exact frames: the mixed curvature is a genuine pin.
    let exact = residual_gauge(&make_model(0.0)).expect("exact-frame certificate");
    let rot_exact = exact
        .generators
        .iter()
        .find(|g| g.family == GeneratorFamily::IsomAtom)
        .expect("rotation generator");
    assert!(
        !rot_exact.unpinned && rot_exact.lowering_error_scale == 0.0,
        "with exact frames the 9/128 curvature must pin the rotation. {}",
        exact.summary
    );

    // Lossy lowering: the same curvature sits below the compression scale —
    // the calibrated verdict must NOT read it as a pin.
    let lossy = residual_gauge(&make_model(0.5)).expect("lossy-frame certificate");
    let rot_lossy = lossy
        .generators
        .iter()
        .find(|g| g.family == GeneratorFamily::IsomAtom)
        .expect("rotation generator");
    assert!(
        (rot_lossy.lowering_error_scale - 0.5).abs() < 1.0e-12,
        "the verdict must carry the atom's lowering-error scale, got {}",
        rot_lossy.lowering_error_scale
    );
    assert!(
        (rot_lossy.pinned_energy_fraction - 9.0 / 128.0).abs() < 1.0e-9,
        "the reported curvature fraction itself must not change — only the \
         tolerance is calibrated; got {}",
        rot_lossy.pinned_energy_fraction
    );
    assert!(
        rot_lossy.unpinned,
        "curvature below the lowering-error scale must not be claimed as a \
         pin (over-claim guard). {}",
        lossy.summary
    );
}

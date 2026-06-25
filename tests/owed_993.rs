//! Owed-work regression gate for issue #993: make the #975/#984 within-atom
//! carve instruments RUNNABLE on real fits.
//!
//! #993: the functional-ANOVA carve (decoder-coefficient covariance via
//! `fit_tensor_surface`, the gauge-projected binding Wald test in `carve`) was
//! fully built and unit-tested on synthetic pair surfaces, but never wired to a
//! fitted SAE atom — the harvest path recorded `fission_carve_skipped` instead
//! of running it. The producer `carve_input_from_fitted_atom` closes that gap:
//! it extracts the two factor bases from a `d = 2` product atom's FUSED
//! Kronecker basis (constant-leading column convention, `flat = j·M₂ + k`) and
//! re-fits the atom's own ambient reconstruction to obtain the scale-included
//! decoder-coefficient covariance the binding test reads.
//!
//! This gate drives the instrument end to end on a `TorusHarmonicEvaluator`
//! atom (the carve's motivating example: ONE bound T² atom vs TWO superposed
//! S¹ atoms) and pins three things:
//!
//! 1. A SEPARABLE decoder (`g(θ₁,θ₂) = a(θ₁) + b(θ₂)`, no interaction) carves
//!    to an ADDITIVE verdict — the carve permits the fission, with a near-zero
//!    interaction fraction.
//! 2. A BOUND decoder (a genuine `θ₁·θ₂` interaction) carves to KEEP — binding
//!    is proven and the fission is blocked.
//! 3. The producer's decoder-coefficient covariance (`fit_tensor_surface`'s
//!    `coeff_covariance`) equals an INDEPENDENT dense ridge-regression `Vb`
//!    reference at the same λ — so the covariance the binding test consumes is
//!    a real, verifiable posterior, not a placeholder.
//!
//! No CUDA, no optimizer dependence: the decoder coefficients ARE what the
//! carve reads, so planting them directly exercises the exact instrument the
//! harvest path now runs on a fitted atom.

use ndarray::{Array1, Array2};

use gam::terms::structure::anova_atom::{
    BindingNotion, FissionDecision, carve, carve_input_from_fitted_atom, fission_decision,
    fit_tensor_surface,
};
use gam::terms::{SaeBasisEvaluator, TorusHarmonicEvaluator};

const N_HARMONICS: usize = 2;
const N_SAMPLE: usize = 400;
const ALPHA: f64 = 0.05;

/// Deterministic interleaved torus code sample on `[0,1)²`, avoiding a
/// degenerate lattice (coprime strides give full 2-D coverage).
fn torus_coords() -> Array2<f64> {
    let mut coords = Array2::<f64>::zeros((N_SAMPLE, 2));
    for i in 0..N_SAMPLE {
        coords[[i, 0]] = (i as f64 + 0.5) / N_SAMPLE as f64;
        coords[[i, 1]] = (((i * 137) % N_SAMPLE) as f64 + 0.5) / N_SAMPLE as f64;
    }
    coords
}

/// The fitted fused torus basis on the sample, `(n, (2H+1)²)`.
fn torus_basis(coords: &Array2<f64>) -> (Array2<f64>, usize) {
    let evaluator = TorusHarmonicEvaluator::new(2, N_HARMONICS).expect("torus evaluator");
    let (phi, _jet) = evaluator.evaluate(coords.view()).expect("torus eval");
    let (m_a, m_b) = evaluator
        .factor_basis_sizes()
        .expect("d=2 torus must report a factor split");
    assert_eq!(m_a, m_b, "symmetric torus");
    assert_eq!(phi.ncols(), m_a * m_b, "fused width = M₁·M₂");
    (phi, m_a)
}

/// Build a decoder `B` (`M₁M₂ × p`) whose per-channel coefficient matrix is the
/// Kronecker-structured `C_d = α_d (u 1ᵀ) + β_d (1 vᵀ)` — purely ADDITIVE in
/// the centered factor bases (no interaction block), so the carve must permit
/// the split. Column order `flat = j·m + k`.
fn separable_decoder(m: usize, p: usize) -> Array2<f64> {
    let mut b = Array2::<f64>::zeros((m * m, p));
    for d in 0..p {
        let alpha = 1.0 + d as f64;
        let beta = 0.5 - 0.3 * d as f64;
        for j in 0..m {
            // main effect on factor A: depends on j only (constant in k).
            let a_j = ((j + 1) as f64).recip() * alpha;
            // main effect on factor B: depends on k only (constant in j).
            for k in 0..m {
                let b_k = ((k + 1) as f64).recip() * beta;
                // Additive surface: the (j,k) coefficient is a_j·[k==0] +
                // b_k·[j==0] so φ¹_j·φ²_k contributions stay separable when
                // mapped through the constant-leading basis. Use the genuinely
                // additive coefficient pattern a_j (only on the k=const column)
                // plus b_k (only on the j=const column).
                let coeff = if k == 0 { a_j } else { 0.0 } + if j == 0 { b_k } else { 0.0 };
                b[[j * m + k, d]] = coeff;
            }
        }
    }
    b
}

/// Build a BOUND decoder: the additive part PLUS a substantial genuine
/// interaction block (off-(row 0 / col 0) entries that depend jointly on j and
/// k), so the carve must prove binding and KEEP the atom.
fn bound_decoder(m: usize, p: usize) -> Array2<f64> {
    let mut b = separable_decoder(m, p);
    for d in 0..p {
        for j in 1..m {
            for k in 1..m {
                // A clear interaction: amplitude that does not factor as
                // a_j·b_k trivially through the centering (it is a full
                // bilinear block on the curved×curved columns).
                b[[j * m + k, d]] += 0.8 / ((j * k) as f64);
            }
        }
    }
    b
}

#[test]
fn separable_torus_atom_carves_to_additive_split() {
    let coords = torus_coords();
    let (phi, m) = torus_basis(&coords);
    let p = 3usize;
    let decoder = separable_decoder(m, p);

    let bundle = carve_input_from_fitted_atom(phi.view(), decoder.view(), m, m)
        .expect("producer runs on a fitted separable torus atom");
    let input = bundle.representational_carve_input();
    assert_eq!(input.notion, BindingNotion::Representational);
    let report = carve(&input, ALPHA).expect("carve runs on producer output");

    assert!(
        report.interaction_fraction < 1e-3,
        "a separable decoder must carry negligible interaction energy; got {:e}",
        report.interaction_fraction
    );
    assert!(
        report.fission.is_some(),
        "the carve must permit a fission on a separable atom (edge_p={:?}, fraction={:e})",
        report.edge_p_value,
        report.interaction_fraction
    );
    assert_eq!(
        fission_decision(&report, None),
        FissionDecision::SplitReconstructionOnly,
        "representational-only additive decoder splits for reconstruction"
    );
}

#[test]
fn bound_torus_atom_carves_to_keep() {
    let coords = torus_coords();
    let (phi, m) = torus_basis(&coords);
    let p = 3usize;
    let decoder = bound_decoder(m, p);

    let bundle = carve_input_from_fitted_atom(phi.view(), decoder.view(), m, m)
        .expect("producer runs on a fitted bound torus atom");
    let input = bundle.representational_carve_input();
    let report = carve(&input, ALPHA).expect("carve runs on producer output");

    assert!(
        report.interaction_fraction > 1e-3,
        "a bound decoder must carry real interaction energy; got {:e}",
        report.interaction_fraction
    );
    // A genuine interaction is either energetically non-negligible or
    // statistically proven — either blocks the split.
    assert!(
        report.fission.is_none(),
        "a bound atom must not split (edge_p={:?}, fraction={:e})",
        report.edge_p_value,
        report.interaction_fraction
    );
    assert_eq!(
        fission_decision(&report, None),
        FissionDecision::Keep,
        "a bound decoder keeps the atom whole"
    );
}

/// The producer's decoder-coefficient covariance must equal an INDEPENDENT
/// dense ridge-regression posterior `Vb = σ̂² (XᵀX + λI)⁻¹` at the same λ — the
/// covariance the binding Wald test reads is a real posterior, byte-comparable
/// to a from-scratch reference built without the production eigenbasis path.
#[test]
fn producer_decoder_covariance_matches_dense_reference() {
    use ndarray::s;

    let coords = torus_coords();
    let (phi, m) = torus_basis(&coords);
    let p = 2usize;
    let decoder = bound_decoder(m, p);
    let reconstruction = phi.dot(&decoder); // n × p, the carve responses

    // Producer's REML re-fit (the path the carve consumes).
    let bundle = carve_input_from_fitted_atom(phi.view(), decoder.view(), m, m).expect("producer");
    let surface = &bundle.surface;
    let lambda = surface.lambda;
    let mm = m * m;

    // Independent dense reference: design X with column (j,k) = φ¹_j·φ²_k =
    // φ_a[:,j]·φ_b[:,k], i.e. exactly the fused torus basis columns. The fused
    // basis IS that product, so X = phi.
    let x = &phi;
    let n = x.nrows();

    // Vb_unit = (XᵀX + λI)⁻¹ (scale-free); scale by per-dim σ̂².
    let mut xtx = x.t().dot(x);
    for i in 0..mm {
        xtx[[i, i]] += lambda;
    }
    let xtx_inv = dense_spd_inverse(&xtx);

    // Per-dimension residual scale σ̂²_d = RSS_d / (n − edf), edf = Σ dᵢ/(dᵢ+λ).
    let beta_ref = xtx_inv.dot(&x.t()).dot(&reconstruction); // mm × p
    let fitted_ref = x.dot(&beta_ref);
    let edf = surface.edf;
    let residual_df = n as f64 - edf;

    for d in 0..p {
        let mut rss = 0.0_f64;
        for r in 0..n {
            let e = reconstruction[[r, d]] - fitted_ref[[r, d]];
            rss += e * e;
        }
        let sigma2 = rss / residual_df;
        let vb_ref = &xtx_inv * sigma2;

        let vb_prod = &surface.coeff_covariance[d];
        assert_eq!(vb_prod.dim(), (mm, mm));

        // Compare the two posteriors entrywise, relative to the reference scale.
        let scale = vb_ref.iter().fold(1e-30_f64, |mx, &v| mx.max(v.abs()));
        let mut max_rel = 0.0_f64;
        for i in 0..mm {
            for j in 0..mm {
                let rel = (vb_prod[[i, j]] - vb_ref[[i, j]]).abs() / scale;
                max_rel = max_rel.max(rel);
            }
        }
        assert!(
            max_rel < 1e-6,
            "producer decoder-coefficient covariance (dim {d}) must equal the dense ridge \
             reference at λ={lambda:e}; max rel {max_rel:e}"
        );

        // Spot-check the diagonal slice the band would read is positive.
        let diag0 = vb_prod[[0, 0]];
        assert!(
            diag0 > 0.0,
            "covariance diagonal must be positive; got {diag0}"
        );
        let _ = &beta_ref.slice(s![.., d]);
    }
}

/// `fit_tensor_surface` directly (the decoder-coefficient covariance producer)
/// recovers the planted coefficients of a known separable surface and reports a
/// finite, positive-definite covariance — the building block the producer wires
/// into the carve.
#[test]
fn fit_tensor_surface_recovers_planted_separable_surface() {
    let coords = torus_coords();
    let evaluator = TorusHarmonicEvaluator::new(2, N_HARMONICS).unwrap();
    let (phi, _) = evaluator.evaluate(coords.view()).unwrap();
    let (m, _) = evaluator.factor_basis_sizes().unwrap();

    // phi_a / phi_b recovered from the fused basis (constant-leading layout).
    let n = phi.nrows();
    let mut phi_a = Array2::<f64>::zeros((n, m));
    let mut phi_b = Array2::<f64>::zeros((n, m));
    for r in 0..n {
        for j in 0..m {
            phi_a[[r, j]] = phi[[r, j * m]];
        }
        for k in 0..m {
            phi_b[[r, k]] = phi[[r, k]];
        }
    }

    // Known additive response a(θ₁)+b(θ₂) on the sample.
    let mut responses = Array2::<f64>::zeros((n, 1));
    for r in 0..n {
        let t1 = coords[[r, 0]];
        let t2 = coords[[r, 1]];
        responses[[r, 0]] =
            (2.0 * std::f64::consts::PI * t1).sin() + 0.7 * (2.0 * std::f64::consts::PI * t2).cos();
    }

    let fit = fit_tensor_surface(phi_a.view(), phi_b.view(), responses.view())
        .expect("tensor surface fit");
    assert_eq!(fit.coeffs.len(), 1);
    assert_eq!(fit.coeffs[0].dim(), (m, m));
    assert!(fit.residual_df > 1.0);
    // The unit covariance is symmetric PD.
    for i in 0..(m * m) {
        assert!(fit.unit_covariance[[i, i]] > 0.0);
    }
    // Reconstruction recovers the additive truth well (low residual fraction).
    let recon = {
        let mut x = Array2::<f64>::zeros((n, m * m));
        for r in 0..n {
            for j in 0..m {
                for k in 0..m {
                    x[[r, j * m + k]] = phi_a[[r, j]] * phi_b[[r, k]];
                }
            }
        }
        let mut beta = Array1::<f64>::zeros(m * m);
        for j in 0..m {
            for k in 0..m {
                beta[j * m + k] = fit.coeffs[0][[j, k]];
            }
        }
        x.dot(&beta)
    };
    let mut rss = 0.0_f64;
    let mut tss = 0.0_f64;
    let mean = responses.column(0).sum() / n as f64;
    for r in 0..n {
        let e = responses[[r, 0]] - recon[r];
        rss += e * e;
        let c = responses[[r, 0]] - mean;
        tss += c * c;
    }
    let r2 = 1.0 - rss / tss.max(1e-12);
    assert!(
        r2 > 0.99,
        "the tensor surface must recover a band-limited additive truth (R²={r2:.4})"
    );
}

/// Dense SPD inverse via Cholesky solve against the identity (reference path,
/// independent of the production eigenbasis).
fn dense_spd_inverse(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    // Cholesky A = L Lᵀ.
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                assert!(s > 0.0, "reference matrix not SPD at pivot {i}: {s}");
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    // Solve A x = e_c for each column c via forward/back substitution.
    let mut inv = Array2::<f64>::zeros((n, n));
    for c in 0..n {
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = if i == c { 1.0 } else { 0.0 };
            for k in 0..i {
                s -= l[[i, k]] * y[k];
            }
            y[i] = s / l[[i, i]];
        }
        let mut xcol = vec![0.0_f64; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in (i + 1)..n {
                s -= l[[k, i]] * xcol[k];
            }
            xcol[i] = s / l[[i, i]];
        }
        for i in 0..n {
            inv[[i, c]] = xcol[i];
        }
    }
    inv
}

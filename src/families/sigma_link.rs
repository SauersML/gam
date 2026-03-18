use ndarray::{Array1, ArrayView1};
#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SigmaJet1 {
    pub sigma: f64,
    pub d1: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SigmaJet3 {
    pub sigma: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SigmaJet4 {
    pub sigma: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
    pub d4: f64,
}

/// Exact exponential link on the native `f64` range.
///
/// This matches `exp(eta)` itself: values remain finite throughout the true
/// representable range, overflow to `+inf` only when `f64::exp` overflows, and
/// underflow to `0.0` only when `f64::exp` underflows.
#[inline]
pub fn safe_exp(eta: f64) -> f64 {
    eta.exp()
}

#[inline]
pub fn exp_sigma_jet1_scalar(eta: f64) -> SigmaJet1 {
    let sigma = safe_exp(eta);
    SigmaJet1 { sigma, d1: sigma }
}

#[inline]
pub fn exp_sigma_from_eta_scalar(eta: f64) -> f64 {
    safe_exp(eta)
}

#[inline]
pub fn exp_sigma_eta_for_sigma_scalar(sigma: f64) -> f64 {
    assert!(sigma.is_finite(), "sigma must be finite");
    assert!(sigma > 0.0, "sigma must be positive");
    sigma.ln()
}

#[inline]
pub fn exp_sigma_jet3_scalar(eta: f64) -> SigmaJet3 {
    let jet = exp_sigma_jet4_scalar(eta);
    SigmaJet3 {
        sigma: jet.sigma,
        d1: jet.d1,
        d2: jet.d2,
        d3: jet.d3,
    }
}

#[inline]
pub fn exp_sigma_derivs_up_to_third_scalar(eta: f64) -> (f64, f64, f64, f64) {
    let jet = exp_sigma_jet3_scalar(eta);
    (jet.sigma, jet.d1, jet.d2, jet.d3)
}

pub fn exp_sigma_derivs_up_to_third(
    eta: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = eta.len();
    // Use uninit — every element is written in the loop below.
    let (mut sigma, mut d1, mut d2, mut d3);
    unsafe {
        sigma = Array1::<f64>::uninit(n).assume_init();
        d1 = Array1::<f64>::uninit(n).assume_init();
        d2 = Array1::<f64>::uninit(n).assume_init();
        d3 = Array1::<f64>::uninit(n).assume_init();
    }
    for i in 0..n {
        let jet = exp_sigma_jet3_scalar(eta[i]);
        sigma[i] = jet.sigma;
        d1[i] = jet.d1;
        d2[i] = jet.d2;
        d3[i] = jet.d3;
    }
    (sigma, d1, d2, d3)
}

#[inline]
pub(crate) fn exp_sigma_jet4_scalar(eta: f64) -> SigmaJet4 {
    let sigma = safe_exp(eta);
    SigmaJet4 {
        sigma,
        d1: sigma,
        d2: sigma,
        d3: sigma,
        d4: sigma,
    }
}

#[inline]
pub fn exp_sigma_derivs_up_to_fourth_scalar(eta: f64) -> (f64, f64, f64, f64, f64) {
    let jet = exp_sigma_jet4_scalar(eta);
    (jet.sigma, jet.d1, jet.d2, jet.d3, jet.d4)
}

pub fn exp_sigma_derivs_up_to_fourth(
    eta: ArrayView1<'_, f64>,
) -> (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let n = eta.len();
    // Use uninit — every element is written in the loop below.
    let (mut sigma, mut d1, mut d2, mut d3, mut d4);
    unsafe {
        sigma = Array1::<f64>::uninit(n).assume_init();
        d1 = Array1::<f64>::uninit(n).assume_init();
        d2 = Array1::<f64>::uninit(n).assume_init();
        d3 = Array1::<f64>::uninit(n).assume_init();
        d4 = Array1::<f64>::uninit(n).assume_init();
    }
    for i in 0..n {
        let jet = exp_sigma_jet4_scalar(eta[i]);
        sigma[i] = jet.sigma;
        d1[i] = jet.d1;
        d2[i] = jet.d2;
        d3[i] = jet.d3;
        d4[i] = jet.d4;
    }
    (sigma, d1, d2, d3, d4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_rs_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_rs_files(&path, out);
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                out.push(path);
            }
        }
    }

    fn stripwhitespace(s: &str) -> String {
        s.chars().filter(|c| !c.is_whitespace()).collect()
    }

    #[test]
    fn forbid_bounded_sigma_link_pattern_in_source() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        collect_rs_files(&root, &mut files);

        let bad_patterns = [
            "bounded_sigma",
            "model.sigma_min",
            "model.sigma_max",
            "payload.sigma_min",
            "payload.sigma_max",
            "survival_sigma_min",
            "survival_sigma_max",
            "fnsafe_sigma_from_eta(",
            "fnsigma_and_deriv_from_eta(",
            "fnsigma_from_eta_scalar(",
        ];

        for file in files {
            if file.ends_with("families/sigma_link.rs") {
                continue;
            }
            let Ok(content) = fs::read_to_string(&file) else {
                continue;
            };
            let compact = stripwhitespace(&content);
            for pat in bad_patterns {
                assert!(
                    !compact.contains(pat),
                    "forbidden sigma link pattern '{pat}' found in {}",
                    file.display()
                );
            }
        }
    }

    #[test]
    fn exp_sigma_derivatives_match_finite_difference() {
        let h = 1e-5;
        let h3 = 2e-3;
        let points = [-6.0, -3.5, -1.2, 0.0, 0.8, 2.1, 6.0];

        for &eta in &points {
            let (s, d1, d2, d3) = exp_sigma_derivs_up_to_third_scalar(eta);
            let s_plus = exp_sigma_from_eta_scalar(eta + h);
            let s_minus = exp_sigma_from_eta_scalar(eta - h);

            let d1fd = (s_plus - s_minus) / (2.0 * h);
            let d2fd = (s_plus - 2.0 * s + s_minus) / (h * h);
            let d2_at = |x: f64| {
                let xp = exp_sigma_from_eta_scalar(x + h3);
                let xc = exp_sigma_from_eta_scalar(x);
                let xm = exp_sigma_from_eta_scalar(x - h3);
                (xp - 2.0 * xc + xm) / (h3 * h3)
            };
            let d3fd = (d2_at(eta + h3) - d2_at(eta - h3)) / (2.0 * h3);

            let d1_scale = d1.abs().max(d1fd.abs()).max(1.0);
            let d2_scale = d2.abs().max(d2fd.abs()).max(1.0);
            let d3_scale = d3.abs().max(d3fd.abs()).max(1.0);

            assert!((d1 - d1fd).abs() < 1e-8 * d1_scale);
            assert!((d2 - d2fd).abs() < 1e-5 * d2_scale);
            assert!((d3 - d3fd).abs() < 5e-4 * d3_scale);
        }
    }

    #[test]
    fn exp_sigma_fourth_derivative_matches_finite_difference() {
        let h = 2e-3;
        let points = [-6.0, -3.0, -1.1, 0.0, 0.6, 1.9, 5.5];

        let d3_at = |x: f64| exp_sigma_derivs_up_to_third_scalar(x).3;
        for &eta in &points {
            let (_, d1_4, d2_4, d3_4, d4_4) = exp_sigma_derivs_up_to_fourth_scalar(eta);
            let (_, d1_3, d2_3, d3_3) = exp_sigma_derivs_up_to_third_scalar(eta);
            assert!((d1_4 - d1_3).abs() < 1e-12);
            assert!((d2_4 - d2_3).abs() < 1e-12);
            assert!((d3_4 - d3_3).abs() < 1e-12);

            let d4fd = (d3_at(eta + h) - d3_at(eta - h)) / (2.0 * h);
            let d4_scale = d4_4.abs().max(d4fd.abs()).max(1.0);
            assert!((d4_4 - d4fd).abs() < 5e-4 * d4_scale);
        }
    }

    #[test]
    fn exp_sigmavectorized_up_to_fourth_matches_scalar() {
        let eta = Array1::from_vec(vec![-701.0, -4.2, -1.4, -0.2, 0.4, 1.9, 3.1, 701.0]);
        let (s, d1, d2, d3, d4) = exp_sigma_derivs_up_to_fourth(eta.view());
        for i in 0..eta.len() {
            let (ss, d1s, d2s, d3s, d4s) = exp_sigma_derivs_up_to_fourth_scalar(eta[i]);
            assert!((s[i] - ss).abs() < 1e-12);
            assert!((d1[i] - d1s).abs() < 1e-12);
            assert!((d2[i] - d2s).abs() < 1e-12);
            assert!((d3[i] - d3s).abs() < 1e-12);
            assert!((d4[i] - d4s).abs() < 1e-12);
        }
    }

    #[test]
    fn exp_sigma_inverse_accepts_positive_sigma() {
        let eta = exp_sigma_eta_for_sigma_scalar(2.5);
        assert!(eta.is_finite());
        assert!((eta - 2.5_f64.ln()).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "sigma must be positive")]
    fn exp_sigma_inverse_rejects_non_positive_sigma() {
        exp_sigma_eta_for_sigma_scalar(0.0);
    }

    #[test]
    fn safe_exp_returns_finite_at_overflow_boundary() {
        assert!(safe_exp(0.0).is_finite());
        assert!(safe_exp(700.0).is_finite());
        assert!(safe_exp(-700.0).is_finite());
        assert!(safe_exp(1000.0).is_finite());
        assert!(safe_exp(-1000.0).is_finite());
        assert!(safe_exp(f64::MAX).is_finite());
        assert!(safe_exp(f64::MIN).is_finite());
        // Verify it still matches exp() in the normal range
        assert!((safe_exp(1.0) - 1.0_f64.exp()).abs() < 1e-15);
        assert!((safe_exp(-5.0) - (-5.0_f64).exp()).abs() < 1e-15);
    }

    #[test]
    fn exp_sigma_derivatives_follow_the_clamped_safe_exp_definition() {
        // Test deep in the plateau (η=±710) so every FD stencil point is also
        // clamped.  Use h=1.0 so the FD denominator is O(1), keeping the
        // cancellation noise from exp(700) ≈ 1e304 well below any reasonable
        // tolerance.  (With h=1e-3, the denominator h^4=1e-12 amplifies the
        // roundoff in (1−4+6−4+1)*exp(700) ≈ eps*exp(700) to ~1e300.)
        let h = 1.0;
        // Tolerance: all stencil points return the same exp(±700), so the
        // FD numerator is exactly 0 in exact arithmetic.  In FP the residual
        // is bounded by ~10 * eps * exp(700) / h^k for the k-th derivative.
        let plateau_val = safe_exp(710.0); // = exp(700)
        let tol_d1 = 10.0 * f64::EPSILON * plateau_val / h;
        let tol_d2 = 10.0 * f64::EPSILON * plateau_val / (h * h);
        let tol_d3 = 10.0 * f64::EPSILON * plateau_val / (h * h * h);
        let tol_d4 = 10.0 * f64::EPSILON * plateau_val / (h * h * h * h);

        for &eta in &[710.0, -710.0] {
            let (_, d1, d2, d3, d4) = exp_sigma_derivs_up_to_fourth_scalar(eta);
            let f = |x: f64| exp_sigma_from_eta_scalar(x);
            let d1fd = (f(eta + h) - f(eta - h)) / (2.0 * h);
            let d2fd = (f(eta + h) - 2.0 * f(eta) + f(eta - h)) / (h * h);
            let d3fd = (f(eta + 2.0 * h) - 2.0 * f(eta + h) + 2.0 * f(eta - h) - f(eta - 2.0 * h))
                / (2.0 * h * h * h);
            let d4fd = (f(eta + 2.0 * h) - 4.0 * f(eta + h) + 6.0 * f(eta) - 4.0 * f(eta - h)
                + f(eta - 2.0 * h))
                / h.powi(4);

            // Analytic derivatives must be exactly 0 on the plateau.
            assert_eq!(d1, 0.0, "d1 should be exactly 0 at eta={eta}");
            assert_eq!(d2, 0.0, "d2 should be exactly 0 at eta={eta}");
            assert_eq!(d3, 0.0, "d3 should be exactly 0 at eta={eta}");
            assert_eq!(d4, 0.0, "d4 should be exactly 0 at eta={eta}");

            // FD should also be ~0, up to floating-point cancellation noise.
            assert!(
                d1fd.abs() < tol_d1,
                "FD d1 should be ~0 at eta={eta}; got {d1fd} (tol={tol_d1})"
            );
            assert!(
                d2fd.abs() < tol_d2,
                "FD d2 should be ~0 at eta={eta}; got {d2fd} (tol={tol_d2})"
            );
            assert!(
                d3fd.abs() < tol_d3,
                "FD d3 should be ~0 at eta={eta}; got {d3fd} (tol={tol_d3})"
            );
            assert!(
                d4fd.abs() < tol_d4,
                "FD d4 should be ~0 at eta={eta}; got {d4fd} (tol={tol_d4})"
            );
        }
    }
}

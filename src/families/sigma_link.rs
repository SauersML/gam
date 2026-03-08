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
pub struct SigmaJet4 {
    pub sigma: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
    pub d4: f64,
}

#[inline]
fn canonical_zero(v: f64) -> f64 {
    if v.abs() < 1e-15 { 0.0 } else { v }
}

#[inline]
fn logistic_stable(eta: f64) -> f64 {
    if eta.is_nan() {
        f64::NAN
    } else if eta == f64::INFINITY {
        1.0
    } else if eta == f64::NEG_INFINITY {
        0.0
    } else if eta >= 0.0 {
        let z = (-eta).exp();
        1.0 / (1.0 + z)
    } else {
        let z = eta.exp();
        z / (1.0 + z)
    }
}

#[inline]
fn validated_bounded_sigma_span(sigma_min: f64, sigma_max: f64) -> f64 {
    assert!(sigma_min.is_finite(), "sigma_min must be finite");
    assert!(sigma_max.is_finite(), "sigma_max must be finite");
    assert!(
        sigma_max > sigma_min,
        "sigma_max must be greater than sigma_min"
    );
    sigma_max - sigma_min
}

#[inline]
pub fn bounded_sigma_jet1_scalar(eta: f64, sigma_min: f64, sigma_max: f64) -> SigmaJet1 {
    let span = validated_bounded_sigma_span(sigma_min, sigma_max);
    let p = logistic_stable(eta);
    SigmaJet1 {
        sigma: sigma_min + span * p,
        d1: span * p * (1.0 - p),
    }
}

#[inline]
pub fn bounded_sigma_from_eta_scalar(eta: f64, sigma_min: f64, sigma_max: f64) -> f64 {
    bounded_sigma_jet1_scalar(eta, sigma_min, sigma_max).sigma
}

#[inline]
pub fn bounded_sigma_eta_for_sigma_scalar(sigma: f64, sigma_min: f64, sigma_max: f64) -> f64 {
    let span = validated_bounded_sigma_span(sigma_min, sigma_max);
    let p = ((sigma - sigma_min) / span).clamp(1e-12, 1.0 - 1e-12);
    (p / (1.0 - p)).ln()
}

#[inline]
pub fn bounded_sigma_jet3_scalar(eta: f64, sigma_min: f64, sigma_max: f64) -> SigmaJet3 {
    let span = validated_bounded_sigma_span(sigma_min, sigma_max);
    let p = logistic_stable(eta);
    let a = p * (1.0 - p);
    let odd = 1.0 - 2.0 * p;
    let d1 = span * a;
    SigmaJet3 {
        sigma: sigma_min + span * p,
        d1: canonical_zero(d1),
        d2: canonical_zero(span * a * odd),
        d3: canonical_zero(span * (a * odd * odd - 2.0 * a * a)),
    }
}

#[inline]
pub fn bounded_sigma_derivs_up_to_third_scalar(
    eta: f64,
    sigma_min: f64,
    sigma_max: f64,
) -> (f64, f64, f64, f64) {
    let jet = bounded_sigma_jet3_scalar(eta, sigma_min, sigma_max);
    (jet.sigma, jet.d1, jet.d2, jet.d3)
}

pub fn bounded_sigma_derivs_up_to_third(
    eta: ArrayView1<'_, f64>,
    sigma_min: f64,
    sigma_max: f64,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut sigma = Array1::<f64>::zeros(eta.len());
    let mut d1 = Array1::<f64>::zeros(eta.len());
    let mut d2 = Array1::<f64>::zeros(eta.len());
    let mut d3 = Array1::<f64>::zeros(eta.len());
    for i in 0..eta.len() {
        let jet = bounded_sigma_jet3_scalar(eta[i], sigma_min, sigma_max);
        sigma[i] = jet.sigma;
        d1[i] = jet.d1;
        d2[i] = jet.d2;
        d3[i] = jet.d3;
    }
    (sigma, d1, d2, d3)
}

#[inline]
pub fn bounded_sigma_jet4_scalar(eta: f64, sigma_min: f64, sigma_max: f64) -> SigmaJet4 {
    let span = validated_bounded_sigma_span(sigma_min, sigma_max);
    let p = logistic_stable(eta);
    let a = p * (1.0 - p);
    let odd = 1.0 - 2.0 * p;
    let d1 = span * a;
    SigmaJet4 {
        sigma: sigma_min + span * p,
        d1: canonical_zero(d1),
        d2: canonical_zero(span * a * odd),
        d3: canonical_zero(span * (a * odd * odd - 2.0 * a * a)),
        d4: canonical_zero(span * a * odd * (1.0 - 12.0 * a)),
    }
}

#[inline]
pub fn bounded_sigma_derivs_up_to_fourth_scalar(
    eta: f64,
    sigma_min: f64,
    sigma_max: f64,
) -> (f64, f64, f64, f64, f64) {
    let jet = bounded_sigma_jet4_scalar(eta, sigma_min, sigma_max);
    (jet.sigma, jet.d1, jet.d2, jet.d3, jet.d4)
}

pub fn bounded_sigma_derivs_up_to_fourth(
    eta: ArrayView1<'_, f64>,
    sigma_min: f64,
    sigma_max: f64,
) -> (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let mut sigma = Array1::<f64>::zeros(eta.len());
    let mut d1 = Array1::<f64>::zeros(eta.len());
    let mut d2 = Array1::<f64>::zeros(eta.len());
    let mut d3 = Array1::<f64>::zeros(eta.len());
    let mut d4 = Array1::<f64>::zeros(eta.len());
    for i in 0..eta.len() {
        let jet = bounded_sigma_jet4_scalar(eta[i], sigma_min, sigma_max);
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

    fn strip_whitespace(s: &str) -> String {
        s.chars().filter(|c| !c.is_whitespace()).collect()
    }

    #[test]
    fn forbid_exp_clamp_sigma_link_pattern_in_source() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        collect_rs_files(&root, &mut files);

        let bad_patterns = [
            "exp().clamp(sigma_min,sigma_max)",
            "exp().clamp(self.sigma_min,self.sigma_max)",
            "exp().clamp(model.sigma_min,model.sigma_max)",
            "fnsafe_sigma_from_eta(",
            "fnsigma_and_deriv_from_eta(",
            "fnsigma_from_eta_scalar(",
        ];

        for file in files {
            // Skip this guard file because it intentionally contains these literals.
            if file.ends_with("families/sigma_link.rs") {
                continue;
            }
            let Ok(content) = fs::read_to_string(&file) else {
                continue;
            };
            let compact = strip_whitespace(&content);
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
    fn bounded_sigma_derivatives_match_finite_difference() {
        let sigma_min = 0.05;
        let sigma_max = 20.0;
        let h = 1e-5;
        let h3 = 2e-3;
        let points = [-6.0, -3.5, -1.2, 0.0, 0.8, 2.1, 6.0];

        for &eta in &points {
            let (s, d1, d2, d3) =
                bounded_sigma_derivs_up_to_third_scalar(eta, sigma_min, sigma_max);
            let s_plus = bounded_sigma_from_eta_scalar(eta + h, sigma_min, sigma_max);
            let s_minus = bounded_sigma_from_eta_scalar(eta - h, sigma_min, sigma_max);

            let d1_fd = (s_plus - s_minus) / (2.0 * h);
            let d2_fd = (s_plus - 2.0 * s + s_minus) / (h * h);
            let d2_at = |x: f64| {
                let xp = bounded_sigma_from_eta_scalar(x + h3, sigma_min, sigma_max);
                let xc = bounded_sigma_from_eta_scalar(x, sigma_min, sigma_max);
                let xm = bounded_sigma_from_eta_scalar(x - h3, sigma_min, sigma_max);
                (xp - 2.0 * xc + xm) / (h3 * h3)
            };
            let d3_fd = (d2_at(eta + h3) - d2_at(eta - h3)) / (2.0 * h3);

            assert!(
                (d1 - d1_fd).abs() < 5e-7,
                "d1 mismatch at eta={eta}: got {d1}, fd {d1_fd}"
            );
            if d1.abs().max(d1_fd.abs()) > 1e-10 {
                assert_eq!(
                    d1.signum(),
                    d1_fd.signum(),
                    "d1 sign mismatch at eta={eta}: got {d1}, fd {d1_fd}"
                );
            }
            assert!(
                (d2 - d2_fd).abs() < 5e-5,
                "d2 mismatch at eta={eta}: got {d2}, fd {d2_fd}"
            );
            // The bounded-sigma map is symmetric at eta = 0, so the true d2 is exactly zero there.
            // A second-order FD stencil can still pick up a small signed residual from truncation and
            // cancellation, so only enforce sign agreement once the curvature is materially away from 0.
            if d2.abs().max(d2_fd.abs()) > 1e-4 {
                assert_eq!(
                    d2.signum(),
                    d2_fd.signum(),
                    "d2 sign mismatch at eta={eta}: got {d2}, fd {d2_fd}"
                );
            }
            assert!(
                (d3 - d3_fd).abs() < 3e-3,
                "d3 mismatch at eta={eta}: got {d3}, fd {d3_fd}"
            );
            if d3.abs().max(d3_fd.abs()) > 1e-10 {
                assert_eq!(
                    d3.signum(),
                    d3_fd.signum(),
                    "d3 sign mismatch at eta={eta}: got {d3}, fd {d3_fd}"
                );
            }
        }
    }

    #[test]
    fn bounded_sigma_fourth_derivative_matches_finite_difference() {
        let sigma_min = 0.05;
        let sigma_max = 20.0;
        let h = 2e-3;
        let points = [-6.0, -3.0, -1.1, 0.0, 0.6, 1.9, 5.5];

        let d3_at = |x: f64| bounded_sigma_derivs_up_to_third_scalar(x, sigma_min, sigma_max).3;
        for &eta in &points {
            let (_, d1_4, d2_4, d3_4, d4_4) =
                bounded_sigma_derivs_up_to_fourth_scalar(eta, sigma_min, sigma_max);
            let (_, d1_3, d2_3, d3_3) =
                bounded_sigma_derivs_up_to_third_scalar(eta, sigma_min, sigma_max);
            assert!((d1_4 - d1_3).abs() < 1e-12);
            assert!((d2_4 - d2_3).abs() < 1e-12);
            assert!((d3_4 - d3_3).abs() < 1e-12);

            let d4_fd = (d3_at(eta + h) - d3_at(eta - h)) / (2.0 * h);
            assert_eq!(
                d4_4.signum(),
                d4_fd.signum(),
                "d4 sign mismatch at eta={eta}: got {d4_4}, fd {d4_fd}"
            );
            assert!(
                (d4_4 - d4_fd).abs() < 1e-4,
                "d4 mismatch at eta={eta}: got {d4_4}, fd {d4_fd}"
            );
        }
    }

    #[test]
    fn bounded_sigma_vectorized_up_to_fourth_matches_scalar() {
        let sigma_min = 0.2;
        let sigma_max = 7.3;
        let eta = Array1::from_vec(vec![-4.2, -1.4, -0.2, 0.4, 1.9, 3.1]);
        let (s, d1, d2, d3, d4) =
            bounded_sigma_derivs_up_to_fourth(eta.view(), sigma_min, sigma_max);
        for i in 0..eta.len() {
            let (ss, d1s, d2s, d3s, d4s) =
                bounded_sigma_derivs_up_to_fourth_scalar(eta[i], sigma_min, sigma_max);
            assert!((s[i] - ss).abs() < 1e-12);
            assert!((d1[i] - d1s).abs() < 1e-12);
            assert!((d2[i] - d2s).abs() < 1e-12);
            assert!((d3[i] - d3s).abs() < 1e-12);
            assert!((d4[i] - d4s).abs() < 1e-12);
        }
    }

    #[test]
    fn bounded_sigma_respects_ultra_narrow_bounds() {
        let sigma_min = 1.0;
        let sigma_max = 1.0 + 5e-13;

        let sigma_mid = bounded_sigma_from_eta_scalar(0.0, sigma_min, sigma_max);
        let sigma_hi = bounded_sigma_from_eta_scalar(f64::INFINITY, sigma_min, sigma_max);
        let sigma_lo = bounded_sigma_from_eta_scalar(f64::NEG_INFINITY, sigma_min, sigma_max);

        assert!(sigma_mid >= sigma_min && sigma_mid <= sigma_max);
        assert_eq!(sigma_hi, sigma_max);
        assert_eq!(sigma_lo, sigma_min);
    }

    #[test]
    fn bounded_sigma_inverse_accepts_ultra_narrow_bounds() {
        let sigma_min = 1.0;
        let sigma_max = 1.0 + 5e-13;
        let sigma = sigma_min + 2.5e-13;

        let eta = bounded_sigma_eta_for_sigma_scalar(sigma, sigma_min, sigma_max);

        assert!(eta.is_finite());
    }

    #[test]
    #[should_panic(expected = "sigma_max must be greater than sigma_min")]
    fn bounded_sigma_forward_rejects_equal_bounds() {
        let _ = bounded_sigma_from_eta_scalar(0.0, 1.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "sigma_max must be greater than sigma_min")]
    fn bounded_sigma_inverse_rejects_reversed_bounds() {
        let _ = bounded_sigma_eta_for_sigma_scalar(1.0, 2.0, 1.0);
    }
}

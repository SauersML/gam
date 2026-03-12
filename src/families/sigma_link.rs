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
fn canonicalzero(v: f64) -> f64 {
    if v.abs() < 1e-15 { 0.0 } else { v }
}

#[inline]
pub fn exp_sigma_jet1_scalar(eta: f64) -> SigmaJet1 {
    let sigma = eta.exp();
    SigmaJet1 { sigma, d1: sigma }
}

#[inline]
pub fn exp_sigma_from_eta_scalar(eta: f64) -> f64 {
    eta.exp()
}

pub fn exp_sigma_and_deriv_from_eta(eta: ArrayView1<'_, f64>) -> (Array1<f64>, Array1<f64>) {
    let sigma = eta.mapv(f64::exp);
    (sigma.clone(), sigma)
}

#[inline]
pub fn exp_sigma_eta_for_sigma_scalar(sigma: f64) -> f64 {
    assert!(sigma.is_finite(), "sigma must be finite");
    assert!(sigma > 0.0, "sigma must be positive");
    sigma.ln()
}

#[inline]
pub fn exp_sigma_jet3_scalar(eta: f64) -> SigmaJet3 {
    let sigma = eta.exp();
    SigmaJet3 {
        sigma,
        d1: canonicalzero(sigma),
        d2: canonicalzero(sigma),
        d3: canonicalzero(sigma),
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
    // For the exp link, all derivatives are identical: sigma = sigma' = sigma'' = sigma'''.
    // Share the single allocation via Arc and clone cheaply where the caller needs ownership.
    let sigma = eta.mapv(f64::exp);
    let d1 = sigma.clone();
    let d2 = sigma.clone();
    let d3 = sigma.clone();
    (sigma, d1, d2, d3)
}

#[inline]
pub fn exp_sigma_jet4_scalar(eta: f64) -> SigmaJet4 {
    let sigma = eta.exp();
    SigmaJet4 {
        sigma,
        d1: canonicalzero(sigma),
        d2: canonicalzero(sigma),
        d3: canonicalzero(sigma),
        d4: canonicalzero(sigma),
    }
}

#[inline]
pub fn exp_sigma_derivs_up_to_fourth_scalar(eta: f64) -> (f64, f64, f64, f64, f64) {
    let jet = exp_sigma_jet4_scalar(eta);
    (jet.sigma, jet.d1, jet.d2, jet.d3, jet.d4)
}

#[allow(clippy::type_complexity)]
pub fn exp_sigma_derivs_up_to_fourth(
    eta: ArrayView1<'_, f64>,
) -> (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    let sigma = eta.mapv(f64::exp);
    (
        sigma.clone(),
        sigma.clone(),
        sigma.clone(),
        sigma.clone(),
        sigma,
    )
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
        let eta = Array1::from_vec(vec![-4.2, -1.4, -0.2, 0.4, 1.9, 3.1]);
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
        let _ = exp_sigma_eta_for_sigma_scalar(0.0);
    }
}

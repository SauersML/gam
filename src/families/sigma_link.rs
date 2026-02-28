use ndarray::{Array1, ArrayView1};
#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::path::Path;

#[inline]
pub fn bounded_sigma_and_deriv_from_eta_scalar(
    eta: f64,
    sigma_min: f64,
    sigma_max: f64,
) -> (f64, f64) {
    let span = (sigma_max - sigma_min).max(1e-12);
    let z = eta.clamp(-40.0, 40.0);
    let p = 1.0 / (1.0 + (-z).exp());
    let sigma = sigma_min + span * p;
    let dsigma_deta = span * p * (1.0 - p);
    (sigma, dsigma_deta)
}

#[inline]
pub fn bounded_sigma_from_eta_scalar(eta: f64, sigma_min: f64, sigma_max: f64) -> f64 {
    bounded_sigma_and_deriv_from_eta_scalar(eta, sigma_min, sigma_max).0
}

pub fn bounded_sigma_and_deriv_from_eta(
    eta: ArrayView1<'_, f64>,
    sigma_min: f64,
    sigma_max: f64,
) -> (Array1<f64>, Array1<f64>) {
    let mut sigma = Array1::<f64>::zeros(eta.len());
    let mut dsigma = Array1::<f64>::zeros(eta.len());
    for i in 0..eta.len() {
        let (s, ds) = bounded_sigma_and_deriv_from_eta_scalar(eta[i], sigma_min, sigma_max);
        sigma[i] = s;
        dsigma[i] = ds;
    }
    (sigma, dsigma)
}

#[inline]
pub fn bounded_sigma_derivs_up_to_third_scalar(
    eta: f64,
    sigma_min: f64,
    sigma_max: f64,
) -> (f64, f64, f64, f64) {
    let span = (sigma_max - sigma_min).max(1e-12);
    let z = eta.clamp(-40.0, 40.0);
    let p = 1.0 / (1.0 + (-z).exp());
    let a = p * (1.0 - p);
    let sigma = sigma_min + span * p;
    let d1 = span * a;
    let d2 = span * a * (1.0 - 2.0 * p);
    let d3 = span * (a * (1.0 - 2.0 * p) * (1.0 - 2.0 * p) - 2.0 * a * a);
    (sigma, d1, d2, d3)
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
        let (s, ds, d2s, d3s) =
            bounded_sigma_derivs_up_to_third_scalar(eta[i], sigma_min, sigma_max);
        sigma[i] = s;
        d1[i] = ds;
        d2[i] = d2s;
        d3[i] = d3s;
    }
    (sigma, d1, d2, d3)
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
            assert!(
                (d2 - d2_fd).abs() < 5e-5,
                "d2 mismatch at eta={eta}: got {d2}, fd {d2_fd}"
            );
            assert!(
                (d3 - d3_fd).abs() < 3e-3,
                "d3 mismatch at eta={eta}: got {d3}, fd {d3_fd}"
            );
        }
    }
}

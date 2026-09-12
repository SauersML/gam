//! Seeded synthetic benchmark fixtures.
//!
//! These generators build the covariate/response panels the benchmark suite
//! (`bench/_run_suite_datasets.py`) fits against: logistic panels, geographic
//! disease surfaces, an admixture cliff, and an HGDP-shaped principal-component
//! panel. They are fixtures, not model code — nothing here fits anything.
//!
//! Every draw goes through [`SplitMixNormalRng`], which is seeded SplitMix64
//! ([`gam_linalg::utils::splitmix64`]) rather than a bespoke generator, so a
//! benchmark row is reproducible from its seed alone and traces back to the one
//! hash primitive the rest of the workspace uses.

use gam_linalg::utils::splitmix64;
use ndarray::{Array1, Array2, ArrayView2};

/// Logistic link, used by every Bernoulli response below.
pub fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

/// Deterministic fixture RNG built on the canonical SplitMix64 stream.
pub struct SplitMixNormalRng {
    state: u64,
    spare_normal: Option<f64>,
}

impl SplitMixNormalRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare_normal: None,
        }
    }

    /// A uniform draw on the open interval `(0, 1)`.
    pub fn uniform_open01(&mut self) -> f64 {
        let bits = splitmix64(&mut self.state) >> 11;
        (bits as f64 + 0.5) / ((1_u64 << 53) as f64)
    }

    pub fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform_open01()
    }

    /// A standard-normal draw via the Box-Muller transform.
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.spare_normal.take() {
            return value;
        }
        let radius = (-2.0 * self.uniform_open01().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform_open01();
        let first = radius * angle.cos();
        self.spare_normal = Some(radius * angle.sin());
        first
    }

    pub fn normal(&mut self, mean: f64, sd: f64) -> f64 {
        mean + sd * self.standard_normal()
    }

    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform_open01() < p
    }
}

/// Center and scale in place. A degenerate or non-finite spread leaves the
/// scale at 1 rather than dividing by (nearly) zero.
pub fn standardize_vector(values: &mut [f64]) {
    if values.is_empty() {
        return;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sd = (values
        .iter()
        .map(|value| {
            let d = *value - mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64)
        .sqrt();
    let denom = if sd.is_finite() && sd >= 1.0e-12 {
        sd
    } else {
        1.0
    };
    for value in values {
        *value = (*value - mean) / denom;
    }
}

/// A `(n, p-1)` standard-normal design with a Bernoulli response driven by a
/// mostly-linear predictor with one sine term.
pub fn binomial_columns(n: usize, p: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let n = n.max(1);
    let p = p.max(3);
    let mut rng = SplitMixNormalRng::new(seed);
    let mut x = Array2::<f64>::zeros((n, p - 1));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        for j in 0..p - 1 {
            x[[i, j]] = rng.standard_normal();
        }
        let eta = -0.25 + 1.1 * x[[i, 0]] - 0.9 * x[[i, 1]] + 0.2 * x[[i, 1]].sin();
        y[i] = if rng.bernoulli(sigmoid(eta)) { 1.0 } else { 0.0 };
    }
    (x, y)
}

/// A latitude/longitude disease surface observed through 16 noisy principal
/// components, with heteroscedastic noise increasing toward the south.
pub fn geo_disease_columns(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let n = n.max(500);
    let mut rng = SplitMixNormalRng::new(seed);
    let mut x = Array2::<f64>::zeros((n, 16));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let lat = rng.uniform_range(-1.0, 1.0);
        let lon = rng.uniform_range(-1.0, 1.0);
        let equator = 1.0 - lat.abs();
        let geo_signal = -1.0
            + 2.20 * equator
            + 0.55 * (std::f64::consts::PI * lon).sin()
            + 0.35 * (2.25 * std::f64::consts::PI * lon).cos()
            + 0.30 * (2.0 * std::f64::consts::PI * equator * lon).sin();
        let southness = (-lat).clamp(0.0, 1.0);
        let eta = geo_signal + rng.normal(0.0, 0.20 + 0.85 * southness.powf(1.35));
        y[i] = if rng.bernoulli(sigmoid(eta)) { 1.0 } else { 0.0 };
        for j in 0..16 {
            let jf = j as f64;
            let a = 0.95 - 0.045 * jf;
            let b = 0.25 + 0.035 * jf;
            let c = if j % 2 == 0 { 1.0 } else { -1.0 } * (0.10 + 0.01 * jf);
            let noise_sd = 0.15 + 0.015 * jf;
            x[[i, j]] = a * lat + b * lon + c * lat * lon + rng.normal(0.0, noise_sd);
        }
    }
    (x, y)
}

/// A 1-D ordered design on `[-1, 1]` with a latent curve of the requested
/// roughness. `mode` is one of `fractional`, `rough`, or `smooth`; anything
/// else is an error naming the offending mode.
pub fn continuous_order_columns(
    mode: &str,
    n: usize,
    seed: u64,
    true_nu: Option<f64>,
    true_kappa2: Option<f64>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    let n = n.max(128);
    let mut rng = SplitMixNormalRng::new(seed);
    let mut x = Array1::<f64>::zeros(n);
    let mut y = Array1::<f64>::zeros(n);
    let mut latent = vec![0.0; n];
    for i in 0..n {
        x[i] = -1.0 + 2.0 * i as f64 / (n.saturating_sub(1)).max(1) as f64;
    }
    match mode {
        "fractional" => {
            let nu = true_nu.unwrap_or(1.8).max(0.1);
            let k2 = true_kappa2.unwrap_or(0.7).max(1.0e-9);
            let harmonics = 32usize.min(n / 2).max(4);
            for h in 1..=harmonics {
                let freq = h as f64;
                let amp = (k2 + freq * freq).powf(-(nu + 0.5) * 0.5);
                let phase = rng.uniform_range(0.0, std::f64::consts::TAU);
                let weight = rng.standard_normal() * amp;
                for i in 0..n {
                    latent[i] +=
                        weight * (std::f64::consts::TAU * freq * (i as f64 / n as f64) + phase).sin();
                }
            }
            standardize_vector(&mut latent);
            for i in 0..n {
                y[i] = latent[i] + rng.normal(0.0, 0.20);
            }
        }
        "rough" => {
            let mut acc = 0.0;
            for value in &mut latent {
                acc += rng.standard_normal();
                *value = acc;
            }
            standardize_vector(&mut latent);
            for i in 0..n {
                y[i] = latent[i] + rng.normal(0.0, 0.30);
            }
        }
        "smooth" => {
            for i in 0..n {
                latent[i] = 1.4 * (2.0 * std::f64::consts::PI * (x[i] + 0.1)).sin()
                    + 0.8 * (0.5 * std::f64::consts::PI * (x[i] - 0.2)).cos();
            }
            standardize_vector(&mut latent);
            for i in 0..n {
                y[i] = latent[i] + rng.normal(0.0, 0.03);
            }
        }
        other => {
            return Err(format!(
                "unsupported continuous-order synthetic mode '{other}'"
            ));
        }
    }
    Ok((x, y))
}

/// Four correlated ancestry PCs plus 12 derived columns, with risk switching
/// sharply across a `tanh` cliff in a fixed direction of PC space.
pub fn thread3_admixture_cliff_columns(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let n = n.max(500);
    let mut rng = SplitMixNormalRng::new(seed);
    let mut x = Array2::<f64>::zeros((n, 16));
    let mut y = Array1::<f64>::zeros(n);
    let coeffs = [1.0, 0.35, -0.20, 0.10];
    for i in 0..n {
        let z0 = rng.standard_normal();
        let z1 = rng.standard_normal();
        let z2 = rng.standard_normal();
        let z3 = rng.standard_normal();
        let pc1 = z0;
        let pc2 = 0.52 * z0 + (1.0_f64 - 0.52_f64 * 0.52_f64).sqrt() * z1;
        let pc3 = -0.18 * z0 + 0.22 * z1 + 0.96 * z2;
        let pc4 = 0.10 * z0 - 0.15 * z1 + 0.35 * z2 + 0.92 * z3;
        let pcs = [pc1, pc2, pc3, pc4];
        for j in 0..4 {
            x[[i, j]] = pcs[j];
        }
        let cliff_axis = coeffs
            .iter()
            .zip(pcs.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let eta = -1.15 + 3.8 * (16.0 * cliff_axis).tanh() + rng.normal(0.0, 0.15);
        y[i] = if rng.bernoulli(sigmoid(eta)) { 1.0 } else { 0.0 };
        for j in 0..12 {
            let jf = j as f64;
            let a = 0.45 - 0.02 * jf;
            let b = if j % 2 == 0 { 1.0 } else { -1.0 } * (0.18 + 0.01 * jf);
            let c = 0.12 + 0.02 * (j % 4) as f64;
            x[[i, j + 4]] = a * pc1 + b * pc2 + c * pc4 + rng.normal(0.0, 0.18 + 0.02 * jf);
        }
    }
    (x, y)
}

/// A worldwide cohort where only the East-Asian subgroup carries a structured
/// geographic risk surface; everyone else sits at a flat low prevalence.
pub fn geo_disease_eas_columns(
    n: usize,
    seed: u64,
    n_pcs: usize,
) -> (Array2<f64>, Array1<f64>) {
    let n = n.max(5);
    let n_pcs = n_pcs.max(3);
    let mut rng = SplitMixNormalRng::new(seed);
    let mut x = Array2::<f64>::zeros((n, n_pcs));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let eas = rng.bernoulli(0.23);
        let lat = if eas {
            rng.uniform_range(15.0, 52.0)
        } else {
            rng.uniform_range(-55.0, 70.0)
        };
        let lon = if eas {
            rng.uniform_range(95.0, 145.0)
        } else {
            rng.uniform_range(-175.0, 175.0)
        };
        let mut eta = if eas {
            (0.10_f64 / 0.90_f64).ln()
        } else {
            (0.02_f64 / 0.98_f64).ln()
        };
        if eas {
            let lat_e = (lat - 33.5) / 11.0;
            let lon_e = (lon - 120.0) / 10.0;
            eta += 3.25 * (1.35 * lat_e).sin() - 2.85 * (1.55 * lon_e).cos()
                + 2.50 * (1.10 * lat_e * lon_e).sin()
                + 1.90 * (1.60 * lat_e + 0.45 * lon_e).cos()
                + rng.normal(0.0, 0.20);
        } else {
            eta += rng.normal(0.0, 0.08);
        }
        y[i] = if rng.bernoulli(sigmoid(eta)) { 1.0 } else { 0.0 };
        let lat_s = lat / 90.0;
        let lon_s = lon / 180.0;
        for j in 0..n_pcs {
            let jf = j as f64;
            let a = 0.98 - 0.05 * jf;
            let b = 0.26 + 0.03 * jf;
            let c = if j % 2 == 0 { 1.0 } else { -1.0 } * (0.12 + 0.01 * jf);
            let d = if j >= 8 { 0.22 } else { 0.06 };
            x[[i, j]] = a * lat_s
                + b * lon_s
                + c * lat_s * lon_s
                + d * (if eas { 1.0 } else { 0.0 })
                + rng.normal(0.0, 0.13 + 0.018 * jf);
        }
    }
    (x, y)
}

/// A ten-cluster population mixture in PC space where the first five clusters
/// carry a much higher event rate. Columns are standardized on the way out.
pub fn papuan_oce_columns(n: usize, seed: u64, n_pcs: usize) -> (Array2<f64>, Array1<f64>) {
    let n = n.max(1200);
    let n_pcs = n_pcs.max(3);
    let mut rng = SplitMixNormalRng::new(seed);
    let centers = [
        [2.7, -0.8, 1.7, 0.7],
        [3.0, -0.6, 1.5, 0.8],
        [2.9, -0.3, 1.9, 0.4],
        [3.1, -0.5, 1.6, 0.8],
        [2.5, -1.0, 1.8, 0.5],
        [0.4, 2.0, -0.1, 0.2],
        [0.7, 1.8, -0.2, 0.1],
        [-2.1, 0.4, 0.1, -0.2],
        [-1.9, -2.2, 0.5, 0.0],
        [-0.3, -0.9, -1.6, 0.4],
    ];
    let probs = [0.12, 0.08, 0.06, 0.06, 0.08, 0.17, 0.10, 0.14, 0.11, 0.08];
    let mut x = Array2::<f64>::zeros((n, n_pcs));
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let draw = rng.uniform_open01();
        let mut cum = 0.0;
        let mut group = 0usize;
        for (idx, prob) in probs.iter().enumerate() {
            cum += prob;
            if draw <= cum {
                group = idx;
                break;
            }
        }
        let mut z = [0.0; 4];
        for d in 0..4 {
            z[d] = centers[group][d] + rng.normal(0.0, 0.55);
        }
        for j in 0..n_pcs {
            let d = j % 4;
            let mix = z[d] + 0.35 * z[(d + 1) % 4] - 0.18 * z[(d + 2) % 4];
            x[[i, j]] = mix + rng.normal(0.0, 0.20 + 0.02 * j as f64);
        }
        let high_risk = group <= 4;
        y[i] = if rng.bernoulli(if high_risk { 0.40 } else { 0.02 }) {
            1.0
        } else {
            0.0
        };
    }
    for j in 0..n_pcs {
        let mut col: Vec<f64> = (0..n).map(|i| x[[i, j]]).collect();
        standardize_vector(&mut col);
        for i in 0..n {
            x[[i, j]] = col[i];
        }
    }
    (x, y)
}

/// An HGDP-shaped principal-component panel: six superpopulations, four
/// subpopulations each, with per-sample coordinates and a 16-column PC matrix
/// built from a superpopulation shift, a subpopulation shift, and a geographic
/// basis in the sample's own latitude/longitude.
pub struct HgdpPcPanel {
    pub sample_ids: Vec<String>,
    pub superpopulations: Vec<String>,
    pub subpopulations: Vec<String>,
    pub latitudes: Vec<f64>,
    pub longitudes: Vec<f64>,
    pub pc: Array2<f64>,
}

pub fn hgdp_pc_panel(seed: u64) -> HgdpPcPanel {
    let mut rng = SplitMixNormalRng::new(seed);
    let specs = [
        ("AFR", 2.0, 20.0),
        ("EUR", 50.0, 15.0),
        ("EAS", 35.0, 115.0),
        ("SAS", 20.0, 78.0),
        ("AMR", -12.0, -72.0),
        ("OCE", -8.0, 145.0),
    ];
    let rows_per_subpop = 40usize;
    let n = specs.len() * 4 * rows_per_subpop;
    let mut pc = Array2::<f64>::zeros((n, 16));
    let mut sample_ids = Vec::with_capacity(n);
    let mut superpopulations = Vec::with_capacity(n);
    let mut subpopulations = Vec::with_capacity(n);
    let mut latitudes = Vec::with_capacity(n);
    let mut longitudes = Vec::with_capacity(n);
    let mut row_idx = 0usize;
    for &(name, base_lat, base_lon) in &specs {
        let mut super_shift = [0.0; 16];
        for value in &mut super_shift {
            *value = rng.normal(0.0, 0.85);
        }
        for sub_idx in 0..4 {
            let sub_name = format!("{name}_SUB{:02}", sub_idx + 1);
            let sub_lat = base_lat + rng.normal(0.0, 5.0);
            let sub_lon = base_lon + rng.normal(0.0, 7.5);
            let mut sub_shift = [0.0; 16];
            for value in &mut sub_shift {
                *value = rng.normal(0.0, 0.30);
            }
            for _ in 0..rows_per_subpop {
                let sample_lat = (sub_lat + rng.normal(0.0, 1.2)).clamp(-58.0, 72.0);
                let sample_lon =
                    ((sub_lon + rng.normal(0.0, 1.8) + 180.0).rem_euclid(360.0)) - 180.0;
                let lat_norm = sample_lat / 90.0;
                let lon_norm = sample_lon / 180.0;
                let geo_terms = [
                    lat_norm,
                    lon_norm,
                    lat_norm * lon_norm,
                    (std::f64::consts::PI * lat_norm).sin(),
                    (std::f64::consts::PI * lon_norm).cos(),
                    (std::f64::consts::PI * (lat_norm + lon_norm) / 2.0).sin(),
                    lat_norm.powi(2),
                    lon_norm.powi(2),
                    (std::f64::consts::PI * lat_norm).cos(),
                    (std::f64::consts::PI * lon_norm).sin(),
                    lat_norm - lon_norm,
                    lat_norm + lon_norm,
                    (std::f64::consts::PI * lat_norm * lon_norm).sin(),
                    (std::f64::consts::PI * (lat_norm - lon_norm) / 2.0).cos(),
                    lat_norm.powi(3),
                    lon_norm.powi(3),
                ];
                for j in 0..16 {
                    pc[[row_idx, j]] = 1.55 * super_shift[j]
                        + 0.90 * sub_shift[j]
                        + 1.20 * geo_terms[j]
                        + rng.normal(0.0, 0.18);
                }
                sample_ids.push(format!("sample_{row_idx:05}"));
                superpopulations.push(name.to_string());
                subpopulations.push(sub_name.clone());
                latitudes.push(sample_lat);
                longitudes.push(sample_lon);
                row_idx += 1;
            }
        }
    }
    HgdpPcPanel {
        sample_ids,
        superpopulations,
        subpopulations,
        latitudes,
        longitudes,
        pc,
    }
}

/// A Bernoulli response whose prevalence and overdispersion are drawn once per
/// subpopulation, so rows in the same group share a latent offset scale.
pub fn geo_subpop_response(
    subpop_codes: &[usize],
    seed: u64,
    prevalence_min: f64,
    prevalence_max: f64,
    noise_scale_min: f64,
    noise_scale_max: f64,
    random_scale: bool,
) -> Array1<f64> {
    let mut rng = SplitMixNormalRng::new(seed);
    let n_groups = subpop_codes.iter().copied().max().unwrap_or(0) + 1;
    let mut prevalence = vec![0.0; n_groups];
    let mut noise_scale = vec![0.0; n_groups];
    for group in 0..n_groups {
        prevalence[group] = rng
            .uniform_range(prevalence_min, prevalence_max)
            .clamp(1.0e-5, 1.0 - 1.0e-5);
        noise_scale[group] = if random_scale {
            rng.uniform_range(noise_scale_min, noise_scale_max)
        } else {
            rng.uniform_range(0.25, 0.85)
        };
    }
    let mut y = Array1::<f64>::zeros(subpop_codes.len());
    for (i, &group) in subpop_codes.iter().enumerate() {
        let p = prevalence[group];
        let eta = (p / (1.0 - p)).ln() + rng.normal(0.0, noise_scale[group]);
        y[i] = if rng.bernoulli(sigmoid(eta)) { 1.0 } else { 0.0 };
    }
    y
}

/// A Bernoulli response whose prevalence is a function of the sample's own
/// latitude/longitude. `mode_code` selects which surface: `superpopnoise`
/// puts the risk gradient in latitude with per-superpopulation overdispersion,
/// `equatornoise` puts it in longitude with overdispersion peaking at the
/// equator.
pub fn geo_latlon_response(
    mode_code: &str,
    superpop_codes: &[usize],
    latitudes: &[f64],
    longitudes: &[f64],
    seed: u64,
    prevalence_min: f64,
    prevalence_max: f64,
) -> Result<Array1<f64>, String> {
    if latitudes.len() != longitudes.len() || latitudes.len() != superpop_codes.len() {
        return Err("geo lat/lon response length mismatch".to_string());
    }
    let mut rng = SplitMixNormalRng::new(seed);
    let n_super = superpop_codes.iter().copied().max().unwrap_or(0) + 1;
    let mut super_noise = vec![0.0; n_super];
    for value in &mut super_noise {
        *value = rng.uniform_range(0.10, 0.90);
    }
    let mut y = Array1::<f64>::zeros(latitudes.len());
    for i in 0..latitudes.len() {
        let lat_norm = (latitudes[i].abs() / 90.0).clamp(0.0, 1.0);
        let lon_norm = ((longitudes[i] + 180.0) / 360.0).clamp(0.0, 1.0);
        let (base_prev, noise_sd) = match mode_code {
            "superpopnoise" => {
                let risk = (0.68 * lat_norm + 0.32 * (1.0 - lon_norm)).clamp(0.0, 1.0);
                (
                    prevalence_min + (prevalence_max - prevalence_min) * risk,
                    super_noise[superpop_codes[i]],
                )
            }
            "equatornoise" => {
                let edge_risk = (longitudes[i].abs() / 180.0).clamp(0.0, 1.0);
                (
                    prevalence_min + (prevalence_max - prevalence_min) * edge_risk,
                    0.05 + 1.25 * (1.0 - lat_norm).clamp(0.0, 1.0),
                )
            }
            other => {
                return Err(format!("unsupported geo_latlon mode: {other}"));
            }
        };
        let p = base_prev.clamp(1.0e-5, 1.0 - 1.0e-5);
        let eta = (p / (1.0 - p)).ln() + rng.normal(0.0, noise_sd);
        y[i] = if rng.bernoulli(sigmoid(eta)) { 1.0 } else { 0.0 };
    }
    Ok(y)
}

/// `|∇ f|` at each collocation point for the admixture-cliff surface
/// `f(z) = jump · tanh(sharpness · zᵀc)`, whose gradient is
/// `jump · sharpness · sech²(sharpness · zᵀc) · c`.
///
/// Returns `None` when the geometry is unusable: no points, a coefficient
/// vector whose length does not match the point dimension, or a coefficient
/// vector that is numerically zero (the cliff has no direction).
pub fn cliff_gradient_magnitude(
    collocation_points: ArrayView2<'_, f64>,
    coefficients: &[f64],
    jump: f64,
    sharpness: f64,
) -> Option<Array1<f64>> {
    let points = collocation_points;
    if points.nrows() == 0 || points.ncols() != coefficients.len() {
        return None;
    }
    let coeff_norm = coefficients
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if coeff_norm < 1.0e-12 {
        return None;
    }
    let mut out = Array1::<f64>::zeros(points.nrows());
    for row in 0..points.nrows() {
        let mut z = 0.0;
        for col in 0..points.ncols() {
            z += points[[row, col]] * coefficients[col];
        }
        let az = (sharpness * z).abs().clamp(0.0, 50.0);
        let sech2 = 1.0 / az.cosh().powi(2);
        out[row] = (jump * sharpness).abs() * sech2 * coeff_norm;
    }
    Some(out)
}

/// The fit-weighted latent-z policy the marginal-slope fixtures fit under:
/// warn-only adequacy checks, the latent score normalized by its own
/// fit-weighted mean and sd, and 8x mean/sd tolerance multipliers.
pub fn exploratory_fit_weighted_latent_z_policy() -> gam_models::bms::LatentZPolicy {
    use gam_models::bms::{
        LatentMeasureSpec, LatentZCheckMode, LatentZNormalizationMode, LatentZPolicy,
    };
    LatentZPolicy {
        check_mode: LatentZCheckMode::WarnOnly,
        normalization: LatentZNormalizationMode::FitWeighted,
        latent_measure: LatentMeasureSpec::auto_default(),
        mean_tol_multiplier: 8.0,
        sd_tol_multiplier: 8.0,
        max_abs_skew: 4.0,
        max_abs_excess_kurtosis: 20.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_reproduces_the_same_panel() {
        let (x1, y1) = binomial_columns(64, 5, 7);
        let (x2, y2) = binomial_columns(64, 5, 7);
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
        let (x3, _) = binomial_columns(64, 5, 8);
        assert_ne!(x1, x3, "a different seed must give a different panel");
    }

    #[test]
    fn continuous_order_rejects_an_unknown_mode_by_name() {
        let err = continuous_order_columns("wiggly", 128, 1, None, None).unwrap_err();
        assert!(err.contains("wiggly"), "unexpected message: {err}");
    }

    #[test]
    fn papuan_columns_are_standardized() {
        let (x, _) = papuan_oce_columns(1200, 3, 4);
        for col in x.columns() {
            let mean = col.iter().sum::<f64>() / col.len() as f64;
            assert!(mean.abs() < 1.0e-9, "column mean {mean} is not centered");
        }
    }

    #[test]
    fn hgdp_panel_is_rectangular_and_labelled() {
        let panel = hgdp_pc_panel(11);
        let n = 6 * 4 * 40;
        assert_eq!(panel.pc.dim(), (n, 16));
        assert_eq!(panel.sample_ids.len(), n);
        assert_eq!(panel.latitudes.len(), n);
        assert!(panel.latitudes.iter().all(|v| (-58.0..=72.0).contains(v)));
        assert!(panel.longitudes.iter().all(|v| (-180.0..=180.0).contains(v)));
    }

    #[test]
    fn cliff_gradient_refuses_a_directionless_cliff() {
        let points = Array2::<f64>::zeros((4, 2));
        assert!(cliff_gradient_magnitude(points.view(), &[0.0, 0.0], 1.0, 1.0).is_none());
        assert!(cliff_gradient_magnitude(points.view(), &[1.0], 1.0, 1.0).is_none());
        assert!(cliff_gradient_magnitude(points.view(), &[1.0, 0.0], 1.0, 1.0).is_some());
    }
}

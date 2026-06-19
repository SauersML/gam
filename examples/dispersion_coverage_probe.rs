//! BUG-HUNT: estimated-dispersion CI coverage for Gamma & Tweedie under a
//! flexible smooth (high EDF / n).
//!
//! The Pearson dispersion estimators in src/solver/pirls/dispersion.rs normalise
//! by Σw, NOT by the residual df (Σw − edf). When the smooth's EDF is a large
//! fraction of n, phî is biased LOW by ~ (n−edf)/n, so SE(η) ∝ √phî is biased
//! low and the nominal-95% response CI under-covers.
//!
//! Known truth: y_i ~ Gamma(mean = mu_i, shape), mu_i = exp(f(x_i)), with a
//! WIGGLY truth so a flexible s(x, k=large) spends many EDF. We do a Monte-Carlo
//! over R replicate datasets and measure the empirical coverage of the per-point
//! response CI for mu(x). Correct coverage ~ 0.95; a φ-bias bug shows coverage
//! well below nominal (e.g. < 0.90) that worsens as k grows.

use csv::StringRecord;
use gam::inference::predict::{PredictInput, PredictUncertaintyOptions, InferenceCovarianceMode};
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use ndarray::Array2;

struct Lcg { state: u64 }
impl Lcg {
    fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(0x9E3779B97F4A7C15) } }
    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 33) as u32
    }
    fn next_unit(&mut self) -> f64 { (self.next_u32() as f64 + 1.0) / ((u32::MAX as f64) + 1.0) }
    fn normal(&mut self) -> f64 {
        let u1 = self.next_unit().max(1e-12);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    // Gamma(shape k integer-ish) via sum of exponentials for shape, scaled.
    fn gamma(&mut self, shape: f64, scale: f64) -> f64 {
        // Marsaglia-Tsang for shape>=1
        let d = shape - 1.0/3.0;
        let c = 1.0 / (9.0*d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c*x).powi(3);
            if v <= 0.0 { continue; }
            let u = self.next_unit();
            if u < 1.0 - 0.0331*x*x*x*x { return d*v*scale; }
            if u.ln() < 0.5*x*x + d*(1.0 - v + v.ln()) { return d*v*scale; }
        }
    }
}

fn encode_columns(headers: &[&str], columns: &[&[f64]]) -> gam::data::EncodedDataset {
    let n = columns[0].len();
    let hdrs: Vec<String> = headers.iter().map(|s| (*s).to_string()).collect();
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    for i in 0..n {
        let row: Vec<String> = columns.iter().map(|c| c[i].to_string()).collect();
        rows.push(StringRecord::from(row));
    }
    encode_recordswith_inferred_schema(hdrs, rows).expect("encode")
}

fn f_true(x: f64) -> f64 { 0.5 + 0.8*(2.5*x).sin() + 0.3*x }

fn run_family(family: &str, shape: f64, k: usize, reps: usize, seed0: u64) {
    let n = 200usize;
    let xs: Vec<f64> = (0..n).map(|i| -1.0 + 2.0*(i as f64)/((n-1) as f64)).collect();
    let mu_true: Vec<f64> = xs.iter().map(|&x| f_true(x).exp()).collect();

    // evaluation grid = training xs (predict at same points)
    let mut covered = 0usize;
    let mut total = 0usize;
    let mut edf_sum = 0.0;
    let formula = format!("y ~ s(x, k={k})");
    for r in 0..reps {
        let mut rng = Lcg::new(seed0 ^ ((r as u64).wrapping_mul(0x100000001B3)));
        let y: Vec<f64> = mu_true.iter().map(|&m| {
            if family == "gamma" {
                rng.gamma(shape, m/shape) // mean = shape*scale = m
            } else {
                // tweedie p=1.5 approx via compound poisson-gamma; fallback: gamma noise
                rng.gamma(shape, m/shape)
            }
        }).collect();
        let data = encode_columns(&["x","y"], &[&xs, &y]);
        let cfg = FitConfig { family: Some(family.to_string()), ..FitConfig::default() };
        let res = match fit_from_formula(&formula, &data, &cfg) { Ok(r)=>r, Err(e)=>{ eprintln!("fit err: {e:?}"); continue; } };
        let FitResult::Standard(fit) = res else { continue; };
        edf_sum += fit.fit.edf_total;
        // predict with uncertainty at xs
        let mut grid = Array2::<f64>::zeros((n,2));
        for i in 0..n { grid[[i,0]] = xs[i]; }
        let input = PredictInput::from_matrix(grid.view(), &fit.resolvedspec);
        let input = match input { Ok(v)=>v, Err(e)=>{eprintln!("predict input err {e:?}"); continue;} };
        let opts = PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::Conditional,
            includeobservation_interval: false,
        };
        let pu = match fit.predict_full_uncertainty(&input, &opts) { Ok(v)=>v, Err(e)=>{eprintln!("predict err {e:?}"); continue;} };
        for i in 0..n {
            let lo = pu.mean_lower[i];
            let hi = pu.mean_upper[i];
            if mu_true[i] >= lo && mu_true[i] <= hi { covered += 1; }
            total += 1;
        }
    }
    let cov = covered as f64 / total.max(1) as f64;
    eprintln!("[{family} shape={shape} k={k}] mean_edf={:.2}  CI coverage of mu_true = {:.3} (nominal 0.95)  over {reps} reps x {n} pts",
        edf_sum / reps as f64, cov);
}

fn main() {
    init_parallelism();
    // Gamma with moderate shape (CV ~ 1/sqrt(shape)); high k spends many EDF on n=200.
    run_family("gamma", 2.0, 8, 40, 0xC0FFEE);
    run_family("gamma", 2.0, 20, 40, 0xC0FFEE);
    run_family("gamma", 2.0, 40, 40, 0xC0FFEE);
}

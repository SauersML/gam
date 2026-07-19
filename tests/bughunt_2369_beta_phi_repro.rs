//! #2369 repro: estimated-φ Beta cannot certify a stationary optimum, even for
//! a pure-parametric `y ~ x` with no smooth and no penalty.

use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unif(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.unif();
        let u2 = self.unif();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn gamma_ge1(&mut self, shape: f64) -> f64 {
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        for _ in 0..10_000 {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.unif();
            if u < 1.0 - 0.0331 * x.powi(4) {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
        d
    }
    fn beta(&mut self, a: f64, b: f64) -> f64 {
        let ga = self.gamma_ge1(a);
        let gb = self.gamma_ge1(b);
        ga / (ga + gb)
    }
}

#[inline]
fn logistic(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

fn make_dataset(n: usize, phi: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = SplitMix64::new(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = 2.0 * rng.unif() - 1.0;
        let eta = (0.5 + 1.5 * xi).clamp(-2.5, 2.5);
        let mu = logistic(eta);
        let yi = rng.beta(mu * phi, (1.0 - mu) * phi).clamp(1e-6, 1.0 - 1e-6);
        x.push(xi);
        y.push(yi);
    }
    (y, x)
}

#[test]
fn beta_pure_parametric_certifies() {
    init_parallelism();
    let n = 800;
    let (y, x) = make_dataset(n, 8.0, 0);
    let headers = vec!["y".to_string(), "x".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![y[i].to_string(), x[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode beta dataset");

    let cfg = FitConfig {
        family: Some("beta".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ x", &ds, &cfg);
    match result {
        Ok(FitResult::Standard(fit)) => {
            let phi = fit.fit.likelihood_scale.fixed_phi();
            println!("y ~ x beta fit OK; phi = {phi:?}");
        }
        // `FitResult` does not implement `Debug`, so name the variant instead of
        // formatting it — this file did not compile at all before, which took the
        // whole `gam` integration-test target with it.
        Ok(_) => panic!("unexpected fit result variant: expected FitResult::Standard"),
        Err(e) => panic!("y ~ x beta fit FAILED: {e}"),
    }
}

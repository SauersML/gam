//! TEMPORARY reproduction probe for #2123 (removed before final).
use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn build_rows(seed: u64, n: usize, noise: f64) -> Vec<[f64; 3]> {
    let mut rng = SplitMix64::new(seed);
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.unit();
        let z = rng.unit();
        let y = (std::f64::consts::TAU * x).sin() + 0.7 * z + noise * rng.normal();
        rows.push([x, z, y]);
    }
    rows
}

fn encode(rows: &[[f64; 3]]) -> gam::data::EncodedDataset {
    let records: Vec<StringRecord> = rows
        .iter()
        .map(|r| StringRecord::from(vec![r[0].to_string(), r[1].to_string(), r[2].to_string()]))
        .collect();
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode")
}

#[test]
fn probe() {
    let n = 300usize;
    // seed 2 noise 0.15 is a reliable non-convergence repro (edf=52, conv=false).
    let data = encode(&build_rows(2, n, 0.15));
    let cfg = FitConfig::default();
    let FitResult::Standard(fit) = fit_from_formula("y ~ te(x, z)", &data, &cfg).expect("fit")
    else {
        panic!("std");
    };
    let f = &fit.fit;
    eprintln!(
        "RESULT edf={:.3} conv={} iters={} gradnorm={:?} lambdas={:?} edf_by_block={:?} reml={:.4} dev={:.4}",
        f.edf_total().unwrap(),
        f.outer_converged,
        f.outer_iterations,
        f.outer_gradient_norm,
        f.lambdas.to_vec(),
        f.edf_by_block(),
        f.reml_score,
        f.deviance,
    );
}

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

fn fit_te(data: &gam::data::EncodedDataset) -> (f64, f64, bool) {
    let cfg = FitConfig::default();
    let FitResult::Standard(fit) = fit_from_formula("y ~ te(x, z)", data, &cfg).expect("fit") else {
        panic!("std");
    };
    let edf = fit.fit.edf_total().expect("edf");
    (edf, fit.fit.deviance, fit.fit.outer_converged)
}

fn perm_stride(rows: &[[f64; 3]], mult: usize) -> Vec<[f64; 3]> {
    let n = rows.len();
    let mut p = vec![[0.0f64; 3]; n];
    for (i, r) in rows.iter().enumerate() {
        p[(i * mult) % n] = *r;
    }
    p
}

#[test]
fn probe() {
    let n = 300usize;
    let mut nonconv = 0;
    let mut orderdep = 0;
    for noise in [0.15f64, 0.2, 0.25] {
        for seed in 1u64..12 {
            let original = build_rows(seed, n, noise);
            let (e0, d0, c0) = fit_te(&encode(&original));
            let (e1, _d1, c1) = fit_te(&encode(&perm_stride(&original, 157)));
            let (e2, _d2, c2) = fit_te(&encode(&perm_stride(&original, 91)));
            let emin = e0.min(e1).min(e2);
            let emax = e0.max(e1).max(e2);
            if !c0 || !c1 || !c2 {
                nonconv += 1;
            }
            if emax - emin > 1.0 {
                orderdep += 1;
                eprintln!("ORDERDEP noise={noise} seed={seed}: edf {e0:.2}/{e1:.2}/{e2:.2} conv {c0}/{c1}/{c2}");
            }
            eprintln!("noise={noise} seed={seed}: edf {e0:.2}/{e1:.2}/{e2:.2} conv {c0}/{c1}/{c2} dev0={d0:.3}");
        }
    }
    eprintln!("SUMMARY nonconv={nonconv} orderdep={orderdep}");
}

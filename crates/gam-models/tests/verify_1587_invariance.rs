//! TEMPORARY local verification of #1587 (deleted after capturing numbers).
use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::FitConfig;
use gam_models::multinomial::{fit_penalized_multinomial_formula, predict_multinomial_formula};

struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unif(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn sample(seed: u64, n: usize) -> (Vec<f64>, Vec<usize>) {
    let mut rng = SplitMix64(seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(1));
    let mut xs = Vec::with_capacity(n);
    let mut cls = Vec::with_capacity(n);
    for _ in 0..n {
        let x = -2.0 + 4.0 * rng.unif();
        let eta = [0.5 + 0.8 * x, -0.3 - 0.5 * x, 0.0];
        let mut p = [eta[0].exp(), eta[1].exp(), eta[2].exp()];
        let s: f64 = p.iter().sum();
        for pk in &mut p {
            *pk /= s;
        }
        let u = rng.unif();
        let c = (u > p[0]) as usize + (u > p[0] + p[1]) as usize;
        xs.push(x);
        cls.push(c);
    }
    (xs, cls)
}

fn fit_aligned(x: &[f64], cls: &[usize], name_map: [&str; 3], grid: &[f64]) -> Vec<[f64; 3]> {
    let headers = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(cls.iter())
        .map(|(&xv, &c)| StringRecord::from(vec![format!("{xv:.10}"), name_map[c].to_string()]))
        .collect();
    let data: EncodedDataset =
        encode_recordswith_inferred_schema(headers, rows).expect("encode fit data");
    let model = fit_penalized_multinomial_formula(
        &data,
        "y ~ s(x)",
        &FitConfig::default(),
        1.0,
        100,
        1.0e-7,
    )
    .expect("fit");
    let grid_rows: Vec<StringRecord> = grid
        .iter()
        .map(|&g| StringRecord::from(vec![format!("{g:.10}")]))
        .collect();
    let grid_data =
        encode_recordswith_inferred_schema(vec!["x".to_string()], grid_rows).expect("grid");
    let pr = predict_multinomial_formula(&model, &grid_data).expect("predict");
    let col_of: Vec<usize> = (0..3)
        .map(|c| {
            model
                .class_levels
                .iter()
                .position(|lvl| lvl == name_map[c])
                .expect("class level")
        })
        .collect();
    (0..grid.len())
        .map(|r| [pr[[r, col_of[0]]], pr[[r, col_of[1]]], pr[[r, col_of[2]]]])
        .collect()
}

fn max_abs(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b.iter())
        .flat_map(|(ra, rb)| (0..3).map(move |k| (ra[k] - rb[k]).abs()))
        .fold(0.0_f64, f64::max)
}

#[test]
fn report_drift() {
    let grid: Vec<f64> = (0..7).map(|i| -1.5 + 3.0 * i as f64 / 6.0).collect();
    for &(seed, n) in &[(0u64, 900usize), (2, 900)] {
        let (x, cls) = sample(seed, n);
        eprintln!("seed={seed} fitting ABC...");
        let a = fit_aligned(&x, &cls, ["A", "B", "C"], &grid);
        eprintln!("seed={seed} fitting ABC(again)...");
        let a2 = fit_aligned(&x, &cls, ["A", "B", "C"], &grid);
        eprintln!("seed={seed} fitting BCA...");
        let b = fit_aligned(&x, &cls, ["B", "C", "A"], &grid);
        eprintln!("seed={seed} fitting CAB...");
        let c = fit_aligned(&x, &cls, ["C", "A", "B"], &grid);
        let refit = max_abs(&a, &a2);
        let cross = max_abs(&a, &b).max(max_abs(&a, &c)).max(max_abs(&b, &c));
        println!("SEED={seed} N={n}  refit_noise={refit:.3e}  cross_reference_drift={cross:.3e}");
        assert!(refit < 1e-9, "refit noise {refit:.3e}");
        assert!(cross < 1e-3, "cross drift {cross:.3e}");
    }
}

//! `bc=clamped/anchored` + `periodic=true` are contradictory: a periodic
//! function on a circle has no "endpoints" to clamp or anchor, the same
//! point is reached from both sides. The formula DSL must reject this
//! combination with a clear actionable error, not silently drop one
//! constraint or produce a broken fit.

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const TAU: f64 = std::f64::consts::TAU;

fn make_periodic_dataset(n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(11);
    let u = Uniform::new(0.0, TAU).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut t: Vec<f64> = (0..n).map(|_| u.sample(&mut rng)).collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let y: Vec<f64> = t
        .iter()
        .map(|theta| 1.0 + 0.6 * theta.cos() + noise.sample(&mut rng))
        .collect();
    let headers = ["t", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = t
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn periodic_with_bc_clamped_rejected() {
    init_parallelism();
    let data = make_periodic_dataset(150);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    for f in [
        "y ~ s(t, periodic=true, period=6.283185307179586, bc=clamped)",
        "y ~ s(t, periodic=true, period=6.283185307179586, bc_left=clamped)",
        "y ~ s(t, periodic=true, period=6.283185307179586, bc_right=clamped)",
    ] {
        let r = fit_from_formula(f, &data, &cfg);
        match r {
            Ok(_) => panic!("expected `{f}` to be rejected (periodic + clamped is contradictory)",),
            Err(e) => {
                let lower = e.to_string().to_lowercase();
                // Must mention either "periodic"/"cyclic" or "clamped" — ideally both —
                // and must not be a deep numerical error from inside REML.
                assert!(
                    !lower.contains("singular")
                        && !lower.contains("conditioning")
                        && !lower.contains("nan"),
                    "`{f}` failed with opaque numerical error: {e}",
                );
                assert!(
                    lower.contains("periodic")
                        || lower.contains("cyclic")
                        || lower.contains("clamped"),
                    "`{f}` failed without naming the conflicting options: {e}",
                );
            }
        }
    }
}

#[test]
fn periodic_with_bc_anchored_rejected() {
    init_parallelism();
    let data = make_periodic_dataset(150);
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    for f in [
        "y ~ s(t, periodic=true, period=6.283185307179586, bc=anchored)",
        "y ~ s(t, periodic=true, period=6.283185307179586, bc_left=anchored)",
        "y ~ s(t, periodic=true, period=6.283185307179586, bc_right=anchored)",
    ] {
        let r = fit_from_formula(f, &data, &cfg);
        match r {
            Ok(_) => {
                panic!("expected `{f}` to be rejected (periodic + anchored is contradictory)",)
            }
            Err(e) => {
                let lower = e.to_string().to_lowercase();
                assert!(
                    !lower.contains("singular")
                        && !lower.contains("conditioning")
                        && !lower.contains("nan"),
                    "`{f}` failed with opaque numerical error: {e}",
                );
                assert!(
                    lower.contains("periodic")
                        || lower.contains("cyclic")
                        || lower.contains("anchored"),
                    "`{f}` failed without naming the conflicting options: {e}",
                );
            }
        }
    }
}

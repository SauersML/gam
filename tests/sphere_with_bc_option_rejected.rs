//! Sphere smooth has no endpoints (S² is a closed manifold), so any
//! `bc=...` option passed to sphere() is meaningless. The DSL must
//! either reject it with a clear error or silently ignore it. Silent
//! application would be the bug to guard against.

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn make_dataset() -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(7);
    let u_lat = Uniform::new(-80.0_f64, 80.0).expect("uniform");
    let u_lon = Uniform::new(-179.0_f64, 179.0).expect("uniform");
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let mut rows = Vec::with_capacity(100);
    for _ in 0..100 {
        let lat = u_lat.sample(&mut rng);
        let lon = u_lon.sample(&mut rng);
        let y = 0.5 + 0.3 * lat.to_radians().sin() + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            lat.to_string(),
            lon.to_string(),
            y.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn sphere_rejects_bc_option() {
    init_parallelism();
    let data = make_dataset();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    for opt in [
        "bc=clamped",
        "bc=anchored",
        "bc_left=clamped",
        "bc_right=anchored",
    ] {
        let f = format!("y ~ sphere(lat, lon, k=10, {opt})");
        let r = fit_from_formula(&f, &data, &cfg);
        match r {
            Ok(_) => {
                panic!("sphere accepted meaningless `{opt}` silently — must reject or document",)
            }
            Err(e) => {
                let lower = e.to_string().to_lowercase();
                // Acceptable: option-validation error that names the bad option,
                // OR a sphere-doesn't-have-endpoints diagnostic.
                assert!(
                    lower.contains("bc")
                        || lower.contains("unknown")
                        || lower.contains("sphere")
                        || lower.contains("not supported")
                        || lower.contains("unsupported"),
                    "sphere bc rejection must name the option / sphere / unknown: {e}",
                );
                eprintln!("[sphere-bc] `{opt}`: clean rejection: {e}");
            }
        }
    }
}

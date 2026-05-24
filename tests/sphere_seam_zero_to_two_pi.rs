//! Sphere predictions at lon = 0 and lon = 2π must match exactly — they are
//! the same point on the sphere. With training data sampled on
//! `lon ∈ [0, 2π)` (the convention used by scripts/geometric_shapes_demo.py),
//! the predict-time axis-clip was clamping any lon outside `[lon_min, lon_max]`
//! to the training extreme. lon = 2π then got clamped to ~2π − ε while
//! lon = 0 stayed at 0; these two land on *different* sides of the wrap
//! and the kernel evaluates them at different basis weights, producing a
//! visible seam in surface plots.
//!
//! Once the periodic-axis bypass in `axis_clip_to_training_ranges` is in
//! place, f(lon=0) and f(lon=2π) match to within numerical noise, and
//! predictions slightly outside [0, 2π) (e.g. lon = -0.05 or lon = 2π+0.05)
//! evaluate via the kernel's intrinsic periodicity instead of getting
//! clamped to flat constants.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

const TAU: f64 = std::f64::consts::TAU;

fn make_sphere_radians(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let u11 = Uniform::<f64>::new(-1.0, 1.0).expect("uniform");
    let u_lon = Uniform::<f64>::new(0.0, TAU).expect("uniform"); // [0, 2π)
    let headers = ["lat", "lon", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            // Uniform on the sphere via inverse-CDF on lat
            let lat = u11.sample(&mut rng).asin();
            let lon = u_lon.sample(&mut rng);
            // Smooth signal that genuinely uses sphere coords
            let y = 1.0
                + 0.6 * lat.sin()
                + 0.4 * lat.cos() * (2.0 * lon).cos()
                + 0.3 * lat.cos().powi(2) * lon.sin();
            StringRecord::from(vec![lat.to_string(), lon.to_string(), y.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn predict(
    formula: &str,
    data: &gam::data::EncodedDataset,
    lats: &[f64],
    lons: &[f64],
) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("sphere fit ok");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let n = lats.len();
    let mut m = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        m[[i, 0]] = lats[i];
        m[[i, 1]] = lons[i];
        m[[i, 2]] = 0.0;
    }
    let design =
        build_term_collection_design(m.view(), &fit.resolvedspec).expect("rebuild predict design");
    design.design.apply(&fit.fit.beta).to_vec()
}

#[test]
fn sphere_radians_seam_at_zero_and_two_pi_predict_equal() {
    init_parallelism();
    let data = make_sphere_radians(800, 2025);
    // Five latitudes; at each, compare f(lat, 0) vs f(lat, 2π).
    let lats = vec![-1.2, -0.6, 0.0, 0.6, 1.2];
    let zeros: Vec<f64> = std::iter::repeat_n(0.0, lats.len()).collect();
    let taus: Vec<f64> = std::iter::repeat_n(TAU, lats.len()).collect();

    for (label, formula) in &[
        ("wahba", "y ~ sphere(lat, lon, radians=true, k=80)"),
        (
            "harmonic",
            "y ~ sphere(lat, lon, radians=true, method=harmonic, max_degree=8)",
        ),
    ] {
        let p0 = predict(formula, &data, &lats, &zeros);
        let p2 = predict(formula, &data, &lats, &taus);
        let mut max_gap = 0.0_f64;
        for (i, (a, b)) in p0.iter().zip(p2.iter()).enumerate() {
            let d = (a - b).abs();
            eprintln!(
                "  {label} lat={:+.2}  f(0)={:.6}  f(2π)={:.6}  |gap|={:.3e}",
                lats[i], a, b, d
            );
            if d > max_gap {
                max_gap = d;
            }
        }
        eprintln!("[sphere-seam] {label:8} max|f(lat,0) - f(lat,2π)| = {max_gap:.3e}");
        assert!(
            max_gap < 1e-6,
            "{label}: sphere seam discontinuous at lon=0 ≡ 2π — max gap {max_gap:.3e} (tol 1e-6)",
        );
    }
}

#[test]
fn sphere_radians_out_of_range_lon_uses_periodic_wrap_not_clamp() {
    init_parallelism();
    let data = make_sphere_radians(800, 2025);
    // Compare f(lat, lon=-0.05) vs f(lat, lon=2π-0.05). On a periodic axis
    // these are the same point; if axis-clip clamps lon=-0.05 to 0, the
    // first value becomes ≈ f(lat, 0) instead of f(lat, 2π-0.05).
    let lats = vec![0.0_f64];
    let neg = vec![-0.05_f64];
    let near_top = vec![TAU - 0.05_f64];

    let formula = "y ~ sphere(lat, lon, radians=true, k=80)";
    let p_neg = predict(formula, &data, &lats, &neg)[0];
    let p_top = predict(formula, &data, &lats, &near_top)[0];
    eprintln!("[sphere-wrap] f(0,-0.05)={p_neg:.6}  f(0,2π−0.05)={p_top:.6}");
    let gap = (p_neg - p_top).abs();
    assert!(
        gap < 1e-6,
        "sphere predict at out-of-training-range lon was clamped instead of wrapped: \
         f(0,-0.05) = {p_neg:.6} vs f(0,2π−0.05) = {p_top:.6} (gap {gap:.3e}, tol 1e-6)",
    );
}

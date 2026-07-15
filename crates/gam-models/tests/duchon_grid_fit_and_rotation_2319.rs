//! #2319 regression: thin-plate / Duchon smooths must (a) fit ordinary gridded
//! spatial data at standard knot budgets, and (b) stay rotation-equivariant on
//! isotropic data. Both are checked through the *public* `fit_from_formula`
//! path — the surface users actually hit.
//!
//! Before the fix, `select_thin_plate_knots` refused a regular grid whenever a
//! maximin symmetry orbit exceeded the remaining knot budget, so
//! `y ~ thinplate(x, z, k=15)` / `k=20` (and the Duchon analogues) errored on a
//! 7x7 grid with "tie class has N distinct points but only M of the exact
//! k-knot budget remain".

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_linalg::matrix::LinearOperator;
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_terms::smooth::build_term_collection_design;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn dataset(points: &[[f64; 2]], y: &[f64]) -> EncodedDataset {
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = points
        .iter()
        .zip(y.iter())
        .map(|(p, &yy)| {
            StringRecord::from(vec![p[0].to_string(), p[1].to_string(), yy.to_string()])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

fn gaussian_cfg() -> FitConfig {
    FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    }
}

#[test]
fn regular_grid_fits_at_standard_knot_budgets() {
    let side = 7usize;
    let mut points = Vec::new();
    let mut y = Vec::new();
    for iy in 0..side {
        for ix in 0..side {
            let (x, z) = (ix as f64, iy as f64);
            points.push([x, z]);
            y.push((0.3 * x).sin() + (0.2 * z).cos());
        }
    }
    let data = dataset(&points, &y);
    let cfg = gaussian_cfg();
    for term in ["thinplate", "duchon"] {
        for k in [8usize, 10, 12, 15, 20, 25, 30] {
            let formula = format!("y ~ {term}(x, z, k={k})");
            fit_from_formula(&formula, &data, &cfg)
                .unwrap_or_else(|e| panic!("{formula} must fit a regular grid, got: {e}"));
        }
    }
}

/// Fit the isotropic-disk fixture from the #2319 report, rotate the coordinates
/// by a generic angle, refit, and require the two predicted surfaces to agree
/// (the defining equivariance property of thin-plate / Duchon splines).
fn rotation_drift(term: &str) -> f64 {
    let mut rng = StdRng::seed_from_u64(0);
    let n = 400usize;
    let r_max = 3.0_f64;
    let ua = Uniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap();
    let ur = Uniform::new(0.0, 1.0).unwrap();
    let noise = Normal::new(0.0, 0.03).unwrap();
    let mut pts = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let ang: f64 = ua.sample(&mut rng);
        let u01: f64 = ur.sample(&mut rng);
        let rad = r_max * u01.sqrt();
        let (px, pz) = (rad * ang.cos(), rad * ang.sin());
        pts.push([px, pz]);
        let r = (px * px + pz * pz).sqrt();
        y.push(
            2.0 * (-((r - r_max / 3.0).powi(2)) / (0.6 * (r_max / 3.0).powi(2))).exp()
                + (1.2 / r_max) * r
                + noise.sample(&mut rng),
        );
    }
    let g: Vec<f64> = (0..9).map(|i| -2.0 + 4.0 * i as f64 / 8.0).collect();
    let mut grid = Vec::new();
    for &gz in &g {
        for &gx in &g {
            grid.push([gx, gz]);
        }
    }
    let cx = pts.iter().map(|p| p[0]).sum::<f64>() / n as f64;
    let cz = pts.iter().map(|p| p[1]).sum::<f64>() / n as f64;
    let (sin_a, cos_a) = 0.698_f64.sin_cos();
    let rot = |p: &[f64; 2]| -> [f64; 2] {
        let (x, z) = (p[0] - cx, p[1] - cz);
        [cx + cos_a * x - sin_a * z, cz + sin_a * x + cos_a * z]
    };
    let pts_r: Vec<[f64; 2]> = pts.iter().map(rot).collect();
    let grid_r: Vec<[f64; 2]> = grid.iter().map(rot).collect();

    let predict = |train: &[[f64; 2]], q: &[[f64; 2]]| -> Vec<f64> {
        let data = dataset(train, &y);
        let cfg = gaussian_cfg();
        let FitResult::Standard(fit) =
            fit_from_formula(&format!("y ~ {term}(x, z)"), &data, &cfg).expect("fit")
        else {
            panic!("expected standard fit");
        };
        let cm = data.column_map();
        let (xc, zc) = (cm["x"], cm["z"]);
        let ncols = data.values.ncols();
        let mut mq = Array2::<f64>::zeros((q.len(), ncols));
        for (i, p) in q.iter().enumerate() {
            mq[[i, xc]] = p[0];
            mq[[i, zc]] = p[1];
        }
        let design = build_term_collection_design(mq.view(), &fit.resolvedspec).expect("design");
        design.design.apply(&fit.fit.beta).to_vec()
    };

    let pb = predict(&pts, &grid);
    let pr = predict(&pts_r, &grid_r);
    let maxd = pb
        .iter()
        .zip(pr.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let range =
        pb.iter().cloned().fold(f64::MIN, f64::max) - pb.iter().cloned().fold(f64::MAX, f64::min);
    maxd / range
}

#[test]
fn duchon_is_rotation_equivariant_on_isotropic_data() {
    // Reporter measured 0.195 drift for duchon vs 0.025 for thinplate; the
    // acceptance band is 0.06. Both must now sit far below it.
    let tp = rotation_drift("thinplate");
    let du = rotation_drift("duchon");
    assert!(tp < 0.06, "thinplate rotation drift {tp:.4} exceeds 0.06");
    assert!(du < 0.06, "duchon rotation drift {du:.4} exceeds 0.06");
}

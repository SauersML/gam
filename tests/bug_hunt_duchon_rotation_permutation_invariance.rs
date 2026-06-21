//! Bug hunt (companion to the bs="ds" translation probe): does the isotropic
//! 2-D Duchon smooth `duchon(x, z)` respect ROTATION invariance and PERMUTATION
//! (row-reorder) invariance of the full fit?
//!
//! - ROTATION: the default Duchon kernel is isotropic (`r^3`, radial), and the
//!   affine null space `span{1, x, z}` is rotation-invariant, so rotating the
//!   2-D covariate by an orthogonal `R` about the data centroid must leave the
//!   fitted surface (and EDF) unchanged. If the uncentered polynomial block (the
//!   #1375 leak) also ill-conditions under rotation, the fit will drift.
//! - PERMUTATION: reordering the data rows is a pure relabeling. The center
//!   selectors (equal-mass partition / farthest-point) are value-deterministic,
//!   so the knots — and the whole fit — should be bit-stable. A drift would mean
//!   a row-order leak (RRQR pivot / summation order / tie-break).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn truth_surface(x: f64, z: f64) -> f64 {
    let d2 = (x - 0.5).powi(2) + (z - 0.5).powi(2);
    1.5 * (-d2 / (2.0 * 0.25_f64 * 0.25)).exp()
}

struct Sample {
    x: Vec<f64>,
    z: Vec<f64>,
    y: Vec<f64>,
}

fn sample(seed: u64, n: usize) -> Sample {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.10).expect("normal");
    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = unit.sample(&mut rng);
        let zi = unit.sample(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(truth_surface(xi, zi) + noise.sample(&mut rng));
    }
    Sample { x, z, y }
}

/// Fit `y ~ duchon(x, z, k=49)` on the given (x, z, y) columns and return
/// (EDF of the smooth, fitted surface at the supplied grid coordinates).
fn fit_and_predict(
    xcol: &[f64],
    zcol: &[f64],
    ycol: &[f64],
    grid_x: &[f64],
    grid_z: &[f64],
) -> (f64, Vec<f64>) {
    let n = xcol.len();
    let headers: Vec<String> = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                xcol[i].to_string(),
                zcol[i].to_string(),
                ycol[i].to_string(),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let col = data.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) =
        fit_from_formula("y ~ duchon(x, z, k=49)", &data, &cfg).expect("duchon fit")
    else {
        panic!("expected standard fit");
    };

    let unified = &fit.fit;
    let design = &fit.design;
    let mut penalty_cursor = 0usize;
    for (_n, _r) in &design.random_effect_ranges {
        penalty_cursor += 1;
    }
    let mut edf = f64::NAN;
    for term in &design.smooth.terms {
        let k = term.penalties_local.len();
        edf = unified.per_term_edf(term.coeff_range.clone(), penalty_cursor, k);
        penalty_cursor += k;
    }

    let m = grid_x.len();
    let mut grid = Array2::<f64>::zeros((m, data.headers.len()));
    for i in 0..m {
        grid[[i, x_idx]] = grid_x[i];
        grid[[i, z_idx]] = grid_z[i];
    }
    let dm = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild duchon design at grid");
    let preds: Vec<f64> = dm.design.apply(&fit.fit.beta).to_vec();
    (edf, preds)
}

fn signal_range(v: &[f64]) -> f64 {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &p in v {
        lo = lo.min(p);
        hi = hi.max(p);
    }
    (hi - lo).max(1e-12)
}

#[test]
fn duchon_smooth_is_rotation_invariant() {
    init_parallelism();
    let s = sample(11, 300);
    let cx = s.x.iter().sum::<f64>() / s.x.len() as f64;
    let cz = s.z.iter().sum::<f64>() / s.z.len() as f64;

    // Interior grid in the original frame.
    let g = 18usize;
    let coord = |i: usize| 0.1 + 0.8 * i as f64 / (g as f64 - 1.0);
    let (mut gx, mut gz) = (Vec::new(), Vec::new());
    for i in 0..g {
        for j in 0..g {
            gx.push(coord(i));
            gz.push(coord(j));
        }
    }

    let (edf0, pred0) = fit_and_predict(&s.x, &s.z, &s.y, &gx, &gz);
    let sr = signal_range(&pred0);

    // Rotate covariates AND grid about the data centroid by theta; predict on the
    // rotated grid. The surface value at a rotated query point must equal the
    // original surface value at the un-rotated point.
    let mut worst_pred = 0.0_f64;
    let mut worst_edf = 0.0_f64;
    for theta in [0.3_f64, std::f64::consts::FRAC_PI_4, 1.5] {
        let (c, sn) = (theta.cos(), theta.sin());
        let rot = |x: f64, z: f64| {
            let (dx, dz) = (x - cx, z - cz);
            (cx + c * dx - sn * dz, cz + sn * dx + c * dz)
        };
        let (rx, rz): (Vec<f64>, Vec<f64>) = s.x.iter().zip(&s.z).map(|(&x, &z)| rot(x, z)).unzip();
        let (rgx, rgz): (Vec<f64>, Vec<f64>) = gx.iter().zip(&gz).map(|(&x, &z)| rot(x, z)).unzip();
        let (edf_r, pred_r) = fit_and_predict(&rx, &rz, &s.y, &rgx, &rgz);
        let dp = pred0
            .iter()
            .zip(&pred_r)
            .fold(0.0_f64, |mx, (a, b)| mx.max((a - b).abs()));
        worst_pred = worst_pred.max(dp);
        worst_edf = worst_edf.max((edf_r - edf0).abs());
        eprintln!(
            "duchon rotation theta={theta}: EDF {edf0:.4}->{edf_r:.4}, pred drift {dp:.3e} ({:.3}% range)",
            100.0 * dp / sr
        );
    }
    eprintln!(
        "duchon rotation-invariance: worst pred drift {worst_pred:.3e} ({:.3}% range), worst EDF {worst_edf:.3e}",
        100.0 * worst_pred / sr
    );
    assert!(
        worst_pred < 1e-4 * sr.max(1.0),
        "Duchon ds smooth is NOT rotation-invariant: worst prediction drift {worst_pred:.3e} \
         ({:.3}% of signal range). Isotropic r^3 kernel + rotation-invariant {{1,x,z}} null space \
         should be exactly invariant.",
        100.0 * worst_pred / sr
    );
    assert!(
        worst_edf < 1e-3,
        "Duchon ds EDF not rotation-invariant: {worst_edf:.3e}"
    );
}

#[test]
fn duchon_smooth_is_permutation_invariant() {
    init_parallelism();
    let s = sample(23, 250);
    let g = 16usize;
    let coord = |i: usize| 0.1 + 0.8 * i as f64 / (g as f64 - 1.0);
    let (mut gx, mut gz) = (Vec::new(), Vec::new());
    for i in 0..g {
        for j in 0..g {
            gx.push(coord(i));
            gz.push(coord(j));
        }
    }
    let (edf0, pred0) = fit_and_predict(&s.x, &s.z, &s.y, &gx, &gz);
    let sr = signal_range(&pred0);

    // Deterministic non-trivial permutation (reverse + a stride shuffle).
    let n = s.x.len();
    let mut perm: Vec<usize> = (0..n).collect();
    perm.reverse();
    perm.rotate_left(7);
    let px: Vec<f64> = perm.iter().map(|&i| s.x[i]).collect();
    let pz: Vec<f64> = perm.iter().map(|&i| s.z[i]).collect();
    let py: Vec<f64> = perm.iter().map(|&i| s.y[i]).collect();

    let (edf_p, pred_p) = fit_and_predict(&px, &pz, &py, &gx, &gz);
    let dp = pred0
        .iter()
        .zip(&pred_p)
        .fold(0.0_f64, |mx, (a, b)| mx.max((a - b).abs()));
    let de = (edf_p - edf0).abs();
    eprintln!(
        "duchon permutation: EDF {edf0:.4}->{edf_p:.4} (|d|={de:.3e}), pred drift {dp:.3e} ({:.3}% range)",
        100.0 * dp / sr
    );
    // Permutation is a pure relabeling: a correct fit is bit-stable (allow only
    // floating-point summation-order noise).
    assert!(
        dp < 1e-7 * sr.max(1.0),
        "Duchon ds smooth is NOT permutation-invariant: prediction drift {dp:.3e} \
         ({:.3}% of signal range) under a pure row reorder — a row-order leak \
         (center-selection tie-break / RRQR pivot / summation order).",
        100.0 * dp / sr
    );
    assert!(
        de < 1e-6,
        "Duchon ds EDF not permutation-invariant: {de:.3e}"
    );
}

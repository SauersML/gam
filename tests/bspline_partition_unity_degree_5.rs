use gam::terms::basis::{SplineScratch, evaluate_bspline_basis_scalar};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn clamped_uniform_knots(degree: usize, num_basis: usize) -> Array1<f64> {
    let interior_count = num_basis.saturating_sub(degree + 1);
    let mut knots = Vec::with_capacity(num_basis + degree + 1);
    knots.extend(std::iter::repeat_n(0.0, degree + 1));
    for j in 1..=interior_count {
        knots.push(j as f64 / (interior_count as f64 + 1.0));
    }
    knots.extend(std::iter::repeat_n(1.0, degree + 1));
    Array1::from(knots)
}

fn clamped_random_knots(rng: &mut StdRng, degree: usize, num_basis: usize) -> Array1<f64> {
    let interior_count = num_basis.saturating_sub(degree + 1);
    let mut interior: Vec<f64> = (0..interior_count).map(|_| rng.random::<f64>()).collect();
    interior.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut knots = Vec::with_capacity(num_basis + degree + 1);
    knots.extend(std::iter::repeat_n(0.0, degree + 1));
    knots.extend(interior);
    knots.extend(std::iter::repeat_n(1.0, degree + 1));
    Array1::from(knots)
}

#[test]
fn bspline_partition_of_unity_degree_5_random_clamped_and_uniform_knots() {
    let degree = 5usize;
    let num_basis = degree + 8;
    let mut rng = StdRng::seed_from_u64(0xBAD50000 + degree as u64);

    for knots in [
        clamped_uniform_knots(degree, num_basis),
        clamped_random_knots(&mut rng, degree, num_basis),
    ] {
        let low = knots[degree];
        let high = knots[knots.len() - degree - 1];
        let mut out = vec![0.0; num_basis];
        let mut scratch = SplineScratch::new(degree);
        for _ in 0..1000 {
            let x = rng.random_range(low..high);
            evaluate_bspline_basis_scalar(x, knots.view(), degree, &mut out, &mut scratch)
                .expect("B-spline evaluation should succeed");
            let sum: f64 = out.iter().sum();
            assert!(
                (sum - 1.0).abs() <= 1e-12,
                "degree={degree}, x={x:.17e}, sum={sum:.17e}, knots={knots:?}"
            );
        }
    }
}

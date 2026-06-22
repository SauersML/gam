// Numerical study: eigenspectrum of the Wahba spherical kernel Gram matrix
// for penalty orders m ∈ {1, 2, 3, 4} on a quasi-uniform set of 30 S² points.

use faer::Side;
use gam::basis::spherical_wahba_kernel_matrix;
use gam::faer_ndarray::FaerEigh;
use ndarray::Array2;

fn quasi_uniform_sphere(n: usize) -> Array2<f64> {
    // Fibonacci / golden-angle layout in (lat, lon) radians.
    // lat = arcsin(z), z = (2i+1)/n - 1, lon = i * golden_angle (mod 2π).
    let golden = 137.5_f64.to_radians();
    let mut out = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let z = (2.0 * i as f64 + 1.0) / n as f64 - 1.0;
        let lat = z.asin();
        let mut lon = (i as f64) * golden;
        // wrap to (-π, π]
        lon = lon.rem_euclid(2.0 * std::f64::consts::PI);
        if lon > std::f64::consts::PI {
            lon -= 2.0 * std::f64::consts::PI;
        }
        out[[i, 0]] = lat;
        out[[i, 1]] = lon;
    }
    out
}

#[test]
fn wahba_kernel_spectrum_orders_1_to_4() {
    let centers = quasi_uniform_sphere(30);
    let k = centers.nrows();

    println!("\nWahba spherical kernel Gram matrix spectrum (k = {})", k);
    println!(
        "{:>3} | {:>14} {:>14} {:>14} {:>14} {:>14} {:>14}",
        "m", "max_eig", "min_eig", "min_pos_eig", "cond(max/min+)", "trace", "frob_norm"
    );
    println!("{}", "-".repeat(102));

    let mut results: Vec<(usize, f64, f64, f64, f64)> = Vec::new();

    for m in 1..=4usize {
        let kmat = spherical_wahba_kernel_matrix(centers.view(), centers.view(), m, true)
            .expect("kernel matrix computation");

        // symmetry diagnostic
        let mut max_asym = 0.0_f64;
        for i in 0..k {
            for j in (i + 1)..k {
                let a = (kmat[[i, j]] - kmat[[j, i]]).abs();
                if a > max_asym {
                    max_asym = a;
                }
            }
        }

        let (evals, _) = kmat.eigh(Side::Lower).expect("eigendecomposition");
        let mut evec: Vec<f64> = evals.iter().copied().collect();
        evec.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_eig = evec[0];
        let max_eig = *evec.last().unwrap();
        // smallest strictly positive eigenvalue
        let min_pos = evec
            .iter()
            .copied()
            .filter(|&v| v > 0.0)
            .fold(f64::INFINITY, f64::min);
        let cond = max_eig / min_pos;

        let trace: f64 = (0..k).map(|i| kmat[[i, i]]).sum();
        let frob_sq: f64 = kmat.iter().map(|v| v * v).sum();
        let frob = frob_sq.sqrt();

        println!(
            "{:>3} | {:>14.6e} {:>14.6e} {:>14.6e} {:>14.6e} {:>14.6e} {:>14.6e}",
            m, max_eig, min_eig, min_pos, cond, trace, frob
        );
        println!(
            "      max |K_ij - K_ji| = {:.3e}; #(eig<0) = {}; #(|eig|<1e-12) = {}",
            max_asym,
            evec.iter().filter(|&&v| v < 0.0).count(),
            evec.iter().filter(|&&v| v.abs() < 1e-12).count(),
        );
        // print smallest 5 and largest 3 eigenvalues for context
        let small5: Vec<f64> = evec.iter().take(5).copied().collect();
        let large3: Vec<f64> = evec.iter().rev().take(3).copied().collect();
        println!("      smallest 5 eigs: {:?}", small5);
        println!("      largest  3 eigs: {:?}", large3);

        results.push((m, max_eig, min_eig, min_pos, cond));
    }

    // Cross-m comparison.
    let cond_m2 = results.iter().find(|r| r.0 == 2).map(|r| r.4).unwrap();
    let cond_m4 = results.iter().find(|r| r.0 == 4).map(|r| r.4).unwrap();
    let min_pos_m4 = results.iter().find(|r| r.0 == 4).map(|r| r.3).unwrap();
    let min_eig_m4 = results.iter().find(|r| r.0 == 4).map(|r| r.2).unwrap();

    println!("\nSummary:");
    println!("  cond(m=4) / cond(m=2)  = {:.6e}", cond_m4 / cond_m2);
    println!("  m=4 min eigenvalue     = {:.6e}", min_eig_m4);
    println!("  m=4 min positive eig   = {:.6e}", min_pos_m4);
}

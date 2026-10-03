#![cfg(test)]
//! Every input's sparse code against the brute-force optimum over all its block sets: the
//! certified bounds enclose the optimum, the code is the code of the blocks returned, and with
//! enough branching every input's code is within a bit of its optimum.

use super::sparse_code::{Metric, Problem, code_site};
use ndarray::{Array2, ArrayView2};

fn noise(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

fn matrix(rows: usize, cols: usize, salt: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| noise(salt + 97 * i + j))
}

/// `f_t(m)` directly: the blocks' bits plus `κ ‖y − Σ_on Z‖²` in input `t`'s metric.
fn code_of(problem: &Problem<'_>, t: usize, on: &[bool], metric: &dyn Fn(usize, &ndarray::Array1<f64>) -> f64) -> f64 {
    let kappa = problem.observations / (2.0 * std::f64::consts::LN_2);
    let z = problem.v.dot(&problem.reads.row(t));
    let mut residual = problem.targets.row(t).to_owned();
    let mut start = 0;
    let mut bits = 0.0;
    for (b, r) in problem.ranks.iter().enumerate() {
        if on[b] {
            bits += problem.bits[b];
            for c in start..start + r {
                residual.scaled_add(-z[c], &problem.u.row(c));
            }
        }
        start += r;
    }
    bits + kappa * metric(t, &residual)
}

fn check(problem: &Problem<'_>, metric: &dyn Fn(usize, &ndarray::Array1<f64>) -> f64) {
    let blocks = problem.ranks.len();
    let coding = code_site(problem, None).expect("coded");
    for t in 0..problem.reads.nrows() {
        let best = (0..1usize << blocks)
            .map(|mask| code_of(problem, t, &(0..blocks).map(|b| mask >> b & 1 == 1).collect::<Vec<_>>(), metric))
            .fold(f64::INFINITY, f64::min);
        let on: Vec<bool> = (0..blocks).map(|b| coding.sets[t].contains(&(b as u32))).collect();
        let own = code_of(problem, t, &on, metric);
        let scale = 1e-9 * (1.0 + best.abs());
        assert!((own - coding.upper[t]).abs() <= scale, "input {t}: its blocks code at {own}, reported {}", coding.upper[t]);
        assert!(coding.lower[t] <= best + scale, "input {t}: lower bound {} above the optimum {best}", coding.lower[t]);
        assert!(coding.upper[t] >= best - scale, "input {t}: code {} below the optimum {best}", coding.upper[t]);
        assert!(coding.upper[t] - coding.lower[t] <= 1.0 + scale, "input {t}: bounds {} apart", coding.upper[t] - coding.lower[t]);
        assert!(coding.upper[t] <= best + 1.0 + scale, "input {t}: code {} more than a bit above {best}", coding.upper[t]);
        // The residual is what the blocks leave.
        let z = problem.v.dot(&problem.reads.row(t));
        let mut left = problem.targets.row(t).to_owned();
        let mut start = 0;
        for (b, r) in problem.ranks.iter().enumerate() {
            if on[b] {
                for c in start..start + r {
                    left.scaled_add(-z[c], &problem.u.row(c));
                }
            }
            start += r;
        }
        assert!(left.iter().zip(coding.residual.row(t).iter()).all(|(a, b)| (a - b).abs() <= 1e-9 * (1.0 + a.abs())));
    }
}

fn library(pieces: usize, d_in: usize, d_out: usize) -> (Array2<f64>, Array2<f64>) {
    (matrix(pieces, d_in, 1000), matrix(pieces, d_out, 2000))
}

#[test]
fn block_codes_under_the_mean_metric_are_certified_against_brute_force() {
    let (d_in, d_out, rows) = (5, 4, 9);
    let ranks = [1, 2, 1, 1, 2, 1, 1];
    let pieces: usize = ranks.iter().sum();
    let (v, u) = library(pieces, d_in, d_out);
    let reads = matrix(rows, d_in, 3000);
    // The site's map is the library's sum, so every block on is its real output.
    let targets = reads.dot(&v.t()).dot(&u);
    let a = matrix(d_out, d_out, 4000);
    let fisher = a.dot(&a.t()) + Array2::<f64>::eye(d_out) * 0.1;
    let bits: Vec<f64> = (0..ranks.len()).map(|b| 2.0 + 3.0 * (noise(5000 + b) + 1.0)).collect();
    for observations in [1.0, 8.0, 64.0] {
        let problem = Problem {
            reads: reads.view(),
            targets: targets.view(),
            v: v.view(),
            u: u.view(),
            ranks: &ranks,
            bits: &bits,
            metric: Metric::Mean(fisher.view()),
            observations,
            nodes: 1 << ranks.len(),
        };
        check(&problem, &|_, r| r.dot(&fisher.dot(r)));
    }
}

#[test]
fn rank_one_codes_under_per_input_metrics_are_certified_against_brute_force() {
    let (d_in, d_out, rows, pieces) = (4, 6, 7, 8);
    let ranks = vec![1; pieces];
    let (v, u) = library(pieces, d_in, d_out);
    let reads = matrix(rows, d_in, 6000);
    // A map the library does not sum to: every input keeps something all on leaves.
    let w = matrix(d_out, d_in, 7000);
    let targets = reads.dot(&w.t());
    let draws: Vec<Array2<f64>> = (0..3).map(|k| matrix(rows, d_out, 8000 + 500 * k)).collect();
    let views: Vec<ArrayView2<'_, f64>> = draws.iter().map(|d| d.view()).collect();
    let bits: Vec<f64> = (0..pieces).map(|c| 1.0 + 4.0 * (noise(9000 + c) + 1.0)).collect();
    for observations in [2.0, 32.0] {
        let problem = Problem {
            reads: reads.view(),
            targets: targets.view(),
            v: v.view(),
            u: u.view(),
            ranks: &ranks,
            bits: &bits,
            metric: Metric::PerRow(&views),
            observations,
            nodes: 1 << pieces,
        };
        check(&problem, &|t, r| draws.iter().map(|g| g.row(t).dot(r).powi(2)).sum::<f64>() / draws.len() as f64);
    }
}

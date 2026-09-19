//! #2900 row 6.18 — `ArrowFactorCache::latent_block_inverse_diagonal` materializes the
//! Schur inverse once and contracts it over each latent coordinate's touched border
//! columns. It must return the latent diagonal of the dense bordered inverse on a system
//! whose rows each touch a strict subset of the border, with a material border
//! correction, so a route that dropped the correction or read the wrong columns fails.

#![cfg(test)]

use super::*;
use ndarray::{Array1, Array2};

#[test]
fn latent_inverse_diagonal_contracts_touched_columns_against_the_dense_inverse_2900() {
    let (n, d, k) = (7usize, 2usize, 9usize);
    let mut sys = ArrowSchurSystem::new(n, d, k);
    for i in 0..n {
        for r in 0..d {
            for c in 0..d {
                sys.rows[i].htt[[r, c]] = if r == c {
                    4.0 + 0.3 * (i + r) as f64
                } else {
                    0.4 - 0.05 * i as f64
                };
            }
            // Row `i` touches border columns `i`, `i + 2` and `i + 5` (mod k).
            for (slot, c) in [i % k, (i + 2) % k, (i + 5) % k].into_iter().enumerate() {
                sys.rows[i].htbeta[[r, c]] = 0.6 - 0.15 * (slot + r) as f64 + 0.05 * i as f64;
            }
        }
        sys.rows[i].gt = Array1::<f64>::zeros(d);
    }
    for r in 0..k {
        for c in 0..k {
            sys.hbb[[r, c]] = if r == c {
                9.0 + 0.2 * r as f64
            } else {
                0.1 / (1.0 + (r + c) as f64)
            };
        }
    }
    sys.gb = Array1::<f64>::zeros(k);
    for i in 0..n {
        let touched = (0..k)
            .filter(|&c| (0..d).any(|r| sys.rows[i].htbeta[[r, c]] != 0.0))
            .count();
        assert_eq!(touched, 3, "row {i} must touch a strict subset of the border");
    }

    let options = ArrowSolveOptions::direct();
    let step = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
        .expect("direct arrow solve should factor this SPD system");
    let cache = step.2;

    let dim = n * d + k;
    let mut h = Array2::<f64>::zeros((dim, dim));
    for i in 0..n {
        let base = i * d;
        for r in 0..d {
            for c in 0..d {
                h[[base + r, base + c]] = sys.rows[i].htt[[r, c]];
            }
            for c in 0..k {
                let v = sys.rows[i].htbeta[[r, c]];
                h[[base + r, n * d + c]] = v;
                h[[n * d + c, base + r]] = v;
            }
        }
    }
    for r in 0..k {
        for c in 0..k {
            h[[n * d + r, n * d + c]] = sys.hbb[[r, c]];
        }
    }
    let l = cholesky_lower(&h).expect("assembled bordered H must be SPD");
    let h_inv = cholesky_solve_matrix(&l, &Array2::<f64>::eye(dim));

    let diag = cache
        .latent_block_inverse_diagonal()
        .expect("dense Schur cache must support the selected-inverse diagonal");
    assert_eq!(diag.len(), n * d);
    for i in 0..n {
        let row_factor = cholesky_lower(&sys.rows[i].htt).expect("row block must be SPD");
        let row_inv = cholesky_solve_matrix(&row_factor, &Array2::<f64>::eye(d));
        for j in 0..d {
            let idx = i * d + j;
            let expected = h_inv[[idx, idx]];
            let got = diag[idx];
            assert!(
                (got - expected).abs() <= 1.0e-10 * expected.abs(),
                "row {i} axis {j}: selected-inverse diagonal {got:.15e} vs dense {expected:.15e}"
            );
            let border_correction = expected - row_inv[[j, j]];
            assert!(
                border_correction > 1.0e-3 * expected,
                "row {i} axis {j}: the border correction {border_correction:.3e} must be \
                 material against {expected:.3e}"
            );
        }
    }
}

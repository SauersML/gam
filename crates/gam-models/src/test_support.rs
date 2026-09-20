//! Laws the declared-law anchor tests anchor on: a Gauss–Hermite law, on which
//! the anchor is the closed form to quadrature tolerance, and a skewed
//! two-component law nothing Gaussian describes.

use crate::latent_anchor::AnchorGridOwned;

/// Gauss–Hermite nodes and weights for the standard normal law, `m` points:
/// the finite law on which the anchor IS the Gaussian closed form to
/// quadrature tolerance. Nodes ascend, weights sum to one.
///
/// The workspace's one rule, [`gam_math::quadrature::standard_normal_gauss_hermite_rule`],
/// split into its node and weight columns.
pub(crate) fn gauss_hermite_probabilists(m: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    gam_math::quadrature::standard_normal_gauss_hermite_rule(m)
        .map(|rule| rule.into_iter().unzip())
        .map_err(|error| error.to_string())
}

/// A deliberately skewed two-component law on 41 nodes.
pub(crate) fn skewed_grid() -> AnchorGridOwned {
    let nodes: Vec<f64> = (0..41).map(|k| -2.5 + 0.15 * k as f64).collect();
    let raw: Vec<f64> = nodes
        .iter()
        .map(|&u| {
            (-0.5 * ((u + 0.9) / 0.5).powi(2)).exp()
                + 0.35 * (-0.5 * ((u - 1.4) / 0.9).powi(2)).exp()
        })
        .collect();
    let total: f64 = raw.iter().sum();
    AnchorGridOwned::new(nodes, raw.into_iter().map(|w| w / total).collect())
}

/// Shared pinning check for the `RowKernel` dense overrides (gam#3035): each
/// dispatched override must agree with the generic per-row reduction on the full
/// data AND on a Horvitz–Thompson-weighted subsample.
pub(crate) mod row_set_overrides {
    use crate::row_kernel::{
        RowKernel, RowSet, build_row_kernel_cache, row_kernel_directional_derivative,
        row_kernel_directional_derivative_all_axes, row_kernel_directional_derivative_generic,
        row_kernel_hessian_dense, row_kernel_hessian_dense_generic,
        row_kernel_second_directional_derivative, row_kernel_second_directional_derivative_all_axes,
    };
    use ndarray::Array2;
    use std::sync::Arc;
    use crate::outer_subsample::WeightedOuterRow;

    /// A deterministic weighted subsample of `0..n` that mixes runs of
    /// consecutive rows with gaps and carries unequal non-unit weights.
    pub(crate) fn weighted_subsample(n: usize) -> RowSet {
        let rows: Vec<WeightedOuterRow> = (0..n)
            .filter(|row| (row * 7) % 5 < 2)
            .map(|row| WeightedOuterRow {
                index: row,
                weight: 1.5 + 0.25 * (row % 4) as f64,
                stratum: 0,
            })
            .collect();
        RowSet::Subsample {
            rows: Arc::new(rows),
            n_full: n,
        }
    }

    fn max_gap(fast: &Array2<f64>, reference: &Array2<f64>) -> f64 {
        assert_eq!(fast.dim(), reference.dim());
        fast.iter()
            .zip(reference.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    }

    fn max_abs(matrix: &Array2<f64>) -> f64 {
        matrix.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()))
    }

    /// Worst relative gap of every dense override against its generic
    /// reduction on `rows`, and the full-data Hessian it was taken at.
    fn override_gaps<const K: usize>(
        kern: &(impl RowKernel<K> + Sync),
        rows: &RowSet,
        d_beta: &[f64],
        d_beta_u: &[f64],
    ) -> ([f64; 4], Array2<f64>) {
        let p = kern.n_coefficients();
        let cache = build_row_kernel_cache(kern, rows).expect("row-kernel cache");
        let hessian = row_kernel_hessian_dense(kern, &cache, rows).expect("dispatched dense Hessian");
        let hessian_generic = row_kernel_hessian_dense_generic(kern, rows, &cache.hessians);
        let hessian_gap = max_gap(&hessian, &hessian_generic) / max_abs(&hessian_generic).max(f64::MIN_POSITIVE);

        let directional = row_kernel_directional_derivative(kern, rows, d_beta).expect("dispatched Hdot");
        let directional_generic =
            row_kernel_directional_derivative_generic(kern, rows, d_beta).expect("generic Hdot");
        let directional_gap =
            max_gap(&directional, &directional_generic) / max_abs(&directional_generic).max(f64::MIN_POSITIVE);

        let first = row_kernel_directional_derivative_all_axes(kern, rows).expect("dispatched all-axes Hdot");
        let second =
            row_kernel_second_directional_derivative_all_axes(kern, rows, d_beta_u).expect("dispatched all-axes H2dot");
        assert_eq!((first.len(), second.len()), (p, p));
        let mut first_gap = 0.0_f64;
        let mut second_gap = 0.0_f64;
        for axis in 0..p {
            let mut e_a = vec![0.0; p];
            e_a[axis] = 1.0;
            let first_generic =
                row_kernel_directional_derivative_generic(kern, rows, &e_a).expect("generic per-axis Hdot");
            let second_generic = row_kernel_second_directional_derivative(kern, rows, d_beta_u, &e_a)
                .expect("generic per-axis H2dot");
            first_gap = first_gap.max(max_gap(&first[axis], &first_generic) / max_abs(&first_generic).max(f64::MIN_POSITIVE));
            second_gap =
                second_gap.max(max_gap(&second[axis], &second_generic) / max_abs(&second_generic).max(f64::MIN_POSITIVE));
        }
        ([hessian_gap, directional_gap, first_gap, second_gap], hessian_generic)
    }

    /// Assert every dense override matches the generic reduction, relative to
    /// the largest entry, within `relative_band` on the full data and on
    /// [`weighted_subsample`]. The subsample Hessian must differ from the
    /// full-data one, so the subsample leg cannot pass by ignoring `rows`.
    pub(crate) fn assert_dense_overrides_match_generic<const K: usize>(
        label: &str,
        kern: &(impl RowKernel<K> + Sync),
        d_beta: &[f64],
        d_beta_u: &[f64],
        relative_band: f64,
    ) {
        let n = kern.n_rows();
        let subsample = weighted_subsample(n);
        let (all_gaps, all_hessian) = override_gaps(kern, &RowSet::All, d_beta, d_beta_u);
        let (subsample_gaps, subsample_hessian) = override_gaps(kern, &subsample, d_beta, d_beta_u);
        let names = ["dense Hessian", "directional derivative", "all-axes first", "all-axes second"];
        for (set, gaps) in [("All", all_gaps), ("weighted subsample", subsample_gaps)] {
            for (name, gap) in names.iter().zip(gaps) {
                assert!(
                    gap <= relative_band,
                    "{label}: {name} override on {set} differs from the generic reduction by {gap:e} relative"
                );
            }
        }
        assert!(
            max_gap(&all_hessian, &subsample_hessian) > 1e-6 * max_abs(&all_hessian),
            "{label}: the weighted subsample reproduced the full-data Hessian, so it tests nothing"
        );
        let show = |gaps: [f64; 4]| gaps.map(|gap| format!("{gap:.2e}")).join(", ");
        eprintln!("{label}: relative gaps All [{}], weighted subsample [{}]", show(all_gaps), show(subsample_gaps));
    }
}

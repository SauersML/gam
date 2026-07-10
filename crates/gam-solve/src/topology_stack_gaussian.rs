//! Held-out Gaussian log-score matrix for topology stacking (#768).
//!
//! Migrated from the Python `gamfit._select_topology.stack_topologies` /
//! `_holdout_predictive_moments` / `_gaussian_logpdf` trio, which built the
//! per-observation held-out log-predictive-density table in Python before
//! handing it to the Rust stacking solve. That table is pure CORE-MATH — a
//! Gaussian log-score with the per-point predictive standard deviation recovered
//! from the family-correct observation interval — so it belongs on the Rust
//! evidence surface next to [`crate::evidence::solve_stacking_weights`], which it
//! reuses verbatim for the simplex weight solve.
//!
//! Given, for each retained candidate `k` and held-out row `i`, the
//! response-scale predictive mean `μ_ik` and the observation interval
//! `[lo_ik, hi_ik]` the Rust predictor emits at coverage `level`, the per-point
//! total predictive standard deviation is recovered by inverting the symmetric
//! Gaussian band,
//!
//! ```text
//!   z      = Φ⁻¹(½ + ½·level)
//!   σ_ik   = (hi_ik − lo_ik) / (2 z)
//! ```
//!
//! and the held-out log-density is the scalar Gaussian log-pdf of the observed
//! response `y_i`,
//!
//! ```text
//!   log p_k(y_i) = −½ ln(2π) − ln(σ_ik) − ½·((y_i − μ_ik) / σ_ik)²
//! ```
//!
//! Rows whose recovered `σ` is non-positive (e.g. fully clamped against the
//! response support) or whose mean/σ is non-finite carry no Gaussian density and
//! are marked `−∞` in that candidate's column, exactly as the Python code did
//! before the stacking solve dropped them.

use gam_math::probability::standard_normal_quantile;
use ndarray::Array2;

use crate::evidence::{StackingConfig, StackingWeights, solve_stacking_weights};

/// Two-sided standard-normal quantile `Φ⁻¹(½ + ½·level)` used to invert the
/// symmetric predictive band into a standard deviation. Separated so the migrated
/// σ-recovery and the quantile both live behind the FFI (the Python side passed a
/// pre-computed `z`).
fn band_quantile(interval_level: f64) -> Result<f64, String> {
    if !(interval_level > 0.0 && interval_level < 1.0) {
        return Err(format!(
            "topology stacking interval_level must lie in (0, 1); got {interval_level}"
        ));
    }
    standard_normal_quantile(0.5 + 0.5 * interval_level)
}

/// Scalar Gaussian log-pdf `−½ ln(2π) − ln(σ) − ½·((y − μ)/σ)²`. Caller
/// guarantees `sd > 0` and finite `mean`/`sd`. Mirrors the Python
/// `_gaussian_logpdf` term-for-term.
#[inline]
fn gaussian_logpdf(y: f64, mean: f64, sd: f64) -> f64 {
    let z = (y - mean) / sd;
    -0.5 * (2.0 * std::f64::consts::PI).ln() - sd.ln() - 0.5 * z * z
}

/// Build the held-out Gaussian log-density table `log_density[i, k]`.
///
/// `means`, `lowers`, and `uppers` are indexed `[candidate][row]` (one inner
/// vector per candidate, each of length `n_rows`), matching the per-candidate
/// column layout the Python builder filled. Returns an `(n_rows, n_cand)` matrix
/// with `−∞` wherever the candidate could not score the row.
pub fn gaussian_log_density_table(
    y: &[f64],
    means: &[Vec<f64>],
    lowers: &[Vec<f64>],
    uppers: &[Vec<f64>],
    interval_level: f64,
) -> Result<Array2<f64>, String> {
    if y.is_empty() {
        return Err("topology stacking held-out fold cannot be empty".to_string());
    }
    let n_rows = y.len();
    let n_cand = means.len();
    if n_cand == 0 {
        return Err("topology stacking requires at least one candidate".to_string());
    }
    if lowers.len() != n_cand || uppers.len() != n_cand {
        return Err(format!(
            "topology stacking: {n_cand} candidate means but {} lower and {} upper columns",
            lowers.len(),
            uppers.len()
        ));
    }
    for (k, (mk, (lk, uk))) in means
        .iter()
        .zip(lowers.iter().zip(uppers.iter()))
        .enumerate()
    {
        if mk.len() != n_rows || lk.len() != n_rows || uk.len() != n_rows {
            return Err(format!(
                "topology stacking: candidate {k} predicted {} means / {} lowers / {} uppers \
                 for a {n_rows}-row held-out fold",
                mk.len(),
                lk.len(),
                uk.len()
            ));
        }
    }

    let z = band_quantile(interval_level)?;
    let two_z = 2.0 * z;

    let mut table = Array2::<f64>::from_elem((n_rows, n_cand), f64::NEG_INFINITY);
    for (k, (mk, (lk, uk))) in means
        .iter()
        .zip(lowers.iter().zip(uppers.iter()))
        .enumerate()
    {
        for i in 0..n_rows {
            let sd = (uk[i] - lk[i]) / two_z;
            let mean = mk[i];
            if sd > 0.0 && mean.is_finite() && sd.is_finite() {
                table[[i, k]] = gaussian_logpdf(y[i], mean, sd);
            }
        }
    }
    Ok(table)
}

/// End-to-end topology stacking from the raw per-candidate held-out predictive
/// moments: build the Gaussian log-density table and solve for the simplex
/// stacking weights + achieved held-out mean log-score. Candidates with no
/// finite held-out density are rejected and zero-weighted by the solve.
pub fn stack_topologies_gaussian(
    y: &[f64],
    means: &[Vec<f64>],
    lowers: &[Vec<f64>],
    uppers: &[Vec<f64>],
    interval_level: f64,
) -> Result<StackingWeights, String> {
    let table = gaussian_log_density_table(y, means, lowers, uppers, interval_level)?;
    solve_stacking_weights(table.view(), StackingConfig::default())
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference log-density tables captured from the CURRENT Python
    // `gamfit._select_topology` math (NormalDist().inv_cdf band quantile,
    // sd=(hi-lo)/(2z), scalar Gaussian log-pdf) on three seeded synthetic
    // inputs. Parity bar: the migrated Rust table must reproduce every finite
    // entry to 1e-12 relative and every non-scorable entry as −∞. See
    // scratchpad/ref.py for the generator.
    fn assert_table_parity(got: &Array2<f64>, expected: &[Vec<f64>]) {
        assert_eq!(got.nrows(), expected.len(), "row count");
        for (i, row) in expected.iter().enumerate() {
            assert_eq!(got.ncols(), row.len(), "col count row {i}");
            for (k, &exp) in row.iter().enumerate() {
                let g = got[[i, k]];
                if exp == f64::NEG_INFINITY {
                    assert!(
                        g == f64::NEG_INFINITY,
                        "[{i},{k}] expected -inf, got {g}"
                    );
                } else {
                    let tol = 1e-12 * exp.abs().max(1.0);
                    assert!(
                        (g - exp).abs() <= tol,
                        "[{i},{k}] expected {exp}, got {g} (|Δ|={} > {tol})",
                        (g - exp).abs()
                    );
                }
            }
        }
    }

    #[test]
    fn parity_fx1_5x3_level_095() {
        let y: Vec<f64> = vec![-0.285723, 0.358634, 2.545264, -0.2061, 0.047048];
        let means: Vec<Vec<f64>> = vec![
            vec![0.524309, -1.892038, 0.071452, 0.779296, 1.757861],
            vec![-2.748718, 2.893161, 2.788547, 0.923535, 0.693376],
            vec![-1.548342, -2.819504, -0.216393, -0.356813, 2.054563],
        ];
        let lowers: Vec<Vec<f64>> = vec![
            vec![0.154887, -2.63816, -0.291755, -0.878064, 0.309672],
            vec![-3.232207, 2.66616, 1.637461, 0.616343, 0.151001],
            vec![-2.682765, -4.172029, -1.315985, -1.749222, 1.031369],
        ];
        let uppers: Vec<Vec<f64>> = vec![
            vec![0.893731, -1.145916, 0.434659, 2.436656, 3.20605],
            vec![-2.265229, 3.120162, 3.939633, 1.230727, 1.235751],
            vec![-0.413919, -1.466979, 0.883199, 1.035596, 3.077757],
        ];
        let expected: Vec<Vec<f64>> = vec![
            vec![-8.484944329539514, -49.36410792708248, -2.751500009015658],
            vec![-17.430308250352464, -238.2075042371761, -11.153239928301575],
            vec![-88.33598872912485, -0.47251568797200494, -12.456484476811838],
            vec![-1.4302138583551767, -25.038762868420147, -0.5995504418674487],
            vec![-3.296856917386603, -2.3617653084998818, -7.662743516018413],
        ];
        let table = gaussian_log_density_table(&y, &means, &lowers, &uppers, 0.95).unwrap();
        assert_table_parity(&table, &expected);

        // End-to-end: the migrated table drives the unchanged stacking solve to
        // a valid simplex.
        let solved = stack_topologies_gaussian(&y, &means, &lowers, &uppers, 0.95).unwrap();
        assert!((solved.weights.sum() - 1.0).abs() < 1e-9);
        assert!(solved.weights.iter().all(|w| w.is_finite() && *w >= 0.0));
        assert!(solved.mean_log_score().is_finite());
    }

    #[test]
    fn parity_fx2_6x4_level_090_with_degenerate_and_nonfinite() {
        let y: Vec<f64> = vec![0.288714, -0.925009, 2.069107, -1.268415, 0.06207, -0.937114];
        let means: Vec<Vec<f64>> = vec![
            vec![-0.507036, 2.843153, -2.377604, -0.33178, -1.651155, -0.902601],
            vec![-1.977385, -0.396524, -0.551639, 1.756306, 0.375639, -1.821592],
            vec![0.094223, 1.513937, 1.540437, 0.522561, 0.870962, -0.61628],
            vec![-2.099999, f64::NAN, -2.584405, -0.61284, -0.509468, 0.932079],
        ];
        let lowers: Vec<Vec<f64>> = vec![
            vec![-2.504768, 2.050415, -3.671604, -1.28709, -3.165972, -1.907575],
            vec![-3.390306, -1.1336, -1.673042, 1.160565, -0.563719, -3.436596],
            vec![-1.218091, -0.085092, 1.161906, -1.365173, 0.009935, -1.066681],
            vec![-3.139857, -2.839459, -3.97128, -1.390677, -2.284281, -0.526211],
        ];
        let uppers: Vec<Vec<f64>> = vec![
            vec![-2.504768, 3.635891, -1.083604, 0.62353, -0.136338, 0.102373],
            vec![-0.564464, 0.340552, 0.569764, 2.352047, 1.314997, -0.206588],
            vec![1.406537, 3.112966, 1.918968, 2.410295, 1.731989, -0.165879],
            vec![-1.060141, 0.927571, -1.19753, 0.164997, 1.265345, 2.390369],
        ];
        let expected: Vec<Vec<f64>> = vec![
            vec![
                f64::NEG_INFINITY,
                -4.246684088575862,
                -0.7227922279616245,
                -7.598828983252114,
            ],
            vec![
                -30.75401729144831,
                -0.8116718953975304,
                -4.037824749615134,
                f64::NEG_INFINITY,
            ],
            vec![
                -16.653764459211533,
                -7.924277143684664,
                -2.0885354757296994,
                -15.978748762059237,
            ],
            vec![
                -1.6759667738578483,
                -34.775604836314955,
                -2.2743142485295422,
                -1.1309795920444083,
            ],
            vec![
                -2.56692789349726,
                -0.5094685081069248,
                -1.465569004299978,
                -1.13526664986499,
            ],
            vec![
                -0.4278442433630474,
                -1.3063678914217267,
                -0.31008488623622643,
                -3.021068255203617,
            ],
        ];
        let table = gaussian_log_density_table(&y, &means, &lowers, &uppers, 0.90).unwrap();
        assert_table_parity(&table, &expected);
    }

    #[test]
    fn parity_fx3_4x2_level_099() {
        let y: Vec<f64> = vec![0.486913, -1.831473, 2.791507, 2.543858];
        let means: Vec<Vec<f64>> = vec![
            vec![-0.197168, 0.980824, -1.712862, -1.669823],
            vec![-2.577867, -1.84028, -2.466819, 1.619352],
        ];
        let lowers: Vec<Vec<f64>> = vec![
            vec![-0.916508, -0.465537, -2.29514, -3.617814],
            vec![-3.437594, -2.889308, -3.25367, 0.271984],
        ];
        let uppers: Vec<Vec<f64>> = vec![
            vec![0.522172, 2.427185, -1.130584, 0.278168],
            vec![-1.71814, -0.791252, -1.679968, 2.96672],
        ];
        let expected: Vec<Vec<f64>> = vec![
            vec![-2.6435499282072783, -41.97978595754513],
            vec![-12.884000499128444, -0.020864834790381707],
            vec![-197.9551848831371, -147.8873217827117],
            vec![-16.161776452982735, -1.8328126763999375],
        ];
        let table = gaussian_log_density_table(&y, &means, &lowers, &uppers, 0.99).unwrap();
        assert_table_parity(&table, &expected);
    }

    #[test]
    fn rejects_out_of_range_level_and_empty_fold() {
        let y = vec![0.0, 1.0];
        let means = vec![vec![0.0, 0.0]];
        let lo = vec![vec![-1.0, -1.0]];
        let hi = vec![vec![1.0, 1.0]];
        assert!(gaussian_log_density_table(&y, &means, &lo, &hi, 0.0).is_err());
        assert!(gaussian_log_density_table(&y, &means, &lo, &hi, 1.0).is_err());
        assert!(gaussian_log_density_table(&[], &means, &lo, &hi, 0.95).is_err());
    }
}

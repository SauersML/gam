//! The formula-default `s(x)` resolves its basis from the data (slop.md G1).
//!
//! The default B-spline used to be capped at `(unique/4).clamp(4, 8)` internal
//! knots — twelve cubic coefficients for every dataset past 35 distinct values.
//! On `sin(8πx) + N(0, 0.3²)` that froze the truth RMSE at ~0.134 from
//! `n = 1 000` to `n = 100 000` while the engine's own #2774 basis-adequacy test
//! rejected the basis at `p = 0`: more data bought nothing, and the fit said so.
//!
//! The default now starts at that pilot resolution and grows through the same
//! adaptive resolution loop the spatial smooths use, bounded only by the
//! covariate's distinct values and the design rank. These arms pin three
//! consequences:
//!
//! * **consistency** — the truth RMSE of the default fit falls as `n` grows, and
//!   the converged basis passes its own adequacy test (no adequacy note);
//! * **the null is still recoverable** — a pure-noise response shrinks the
//!   double-penalized default to ~0 EDF, so growth never manufactures signal;
//! * **the linear truth is still recoverable** — a straight line shrinks to ~1
//!   EDF rather than being bent by the extra resolution.

use gam::data::EncodedDataset;
use gam::inference::model::{ColumnKindTag, DataSchema, SchemaColumn};
use gam::matrix::LinearOperator;
use gam::{FitConfig, FitResult, fit_from_formula_with_notes};

/// Deterministic uniform/normal draws, independent of the linked sampler.
struct Lcg(u64);

impl Lcg {
    fn next_uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_uniform().max(1e-12);
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

const NOISE_SD: f64 = 0.3;

fn dataset(n: usize, seed: u64, truth: fn(f64) -> f64) -> (EncodedDataset, Vec<f64>) {
    let mut rng = Lcg(seed);
    let x: Vec<f64> = (0..n).map(|_| rng.next_uniform()).collect();
    let mean: Vec<f64> = x.iter().map(|&t| truth(t)).collect();
    let mut values = Vec::with_capacity(2 * n);
    for (xi, mi) in x.iter().zip(&mean) {
        values.push(*xi);
        values.push(mi + NOISE_SD * rng.next_normal());
    }
    let columns: Vec<SchemaColumn> = ["x", "y"]
        .into_iter()
        .map(|name| SchemaColumn {
            name: name.to_string(),
            kind: ColumnKindTag::Continuous,
            levels: vec![],
        })
        .collect();
    let column_kinds = columns.iter().map(|column| column.kind).collect();
    (
        EncodedDataset {
            headers: vec!["x".to_string(), "y".to_string()],
            values: ndarray::Array2::from_shape_vec((n, 2), values).expect("fixture shape"),
            schema: DataSchema { columns },
            column_kinds,
        },
        mean,
    )
}

struct DefaultFit {
    truth_rmse: f64,
    basis_dim: usize,
    edf: f64,
    p_value: Option<f64>,
    notes: Vec<String>,
}

fn fit_default(n: usize, seed: u64, truth: fn(f64) -> f64) -> DefaultFit {
    let (data, mean) = dataset(n, seed, truth);
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let outcome =
        fit_from_formula_with_notes("y ~ s(x)", &data, &config).expect("default s(x) fits");
    let FitResult::Standard(fit) = &outcome.result else {
        panic!("a Gaussian s(x) is a standard GAM");
    };
    let fitted = fit.design.design.apply(&fit.fit.beta).to_vec();
    let truth_rmse = (fitted
        .iter()
        .zip(&mean)
        .map(|(f, m)| (f - m).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    assert_eq!(fit.basis_adequacy.len(), 1, "one smooth, one adequacy row");
    let row = &fit.basis_adequacy[0];
    DefaultFit {
        truth_rmse,
        basis_dim: row.basis_dim,
        edf: row.edf.expect("a converged smooth reports its EDF"),
        p_value: row.p_value,
        notes: outcome.inference_notes.iter().cloned().collect(),
    }
}

fn sin_8_pi(x: f64) -> f64 {
    (8.0 * std::f64::consts::PI * x).sin()
}

#[test]
fn default_smooth_truth_error_falls_with_n_and_passes_its_adequacy_test() {
    let fits: Vec<(usize, DefaultFit)> = [1_000, 4_000, 16_000]
        .into_iter()
        .map(|n| (n, fit_default(n, 20_260_919, sin_8_pi)))
        .collect();
    for (n, fit) in &fits {
        eprintln!(
            "n={n}: truth RMSE={:.4}, basis_dim={}, edf={:.2}, adequacy p={:?}",
            fit.truth_rmse, fit.basis_dim, fit.edf, fit.p_value
        );
        assert!(
            !fit.notes.iter().any(|note| note.contains("basis adequacy")),
            "n={n}: the default basis must pass its own adequacy test; notes={:?}",
            fit.notes
        );
        assert!(
            !fit.notes.iter().any(|note| note.contains("internal knots")),
            "n={n}: no per-fit knot-placement note; notes={:?}",
            fit.notes
        );
        // Eight half-periods of a sine cannot be represented by the old
        // eleven-column cap; the resolved basis must be wider than it.
        assert!(
            fit.basis_dim > 11,
            "n={n}: basis_dim={} never grew past the old cap",
            fit.basis_dim
        );
    }
    for pair in fits.windows(2) {
        let ((n0, small), (n1, large)) = (&pair[0], &pair[1]);
        assert!(
            large.truth_rmse < small.truth_rmse,
            "truth RMSE must fall with n: n={n0} -> {:.4}, n={n1} -> {:.4}",
            small.truth_rmse,
            large.truth_rmse
        );
    }
    // The old cap froze the RMSE at ~0.134 for every n; a consistent default at
    // 16 000 rows sits far below that noise-free approximation error.
    let (_, last) = fits.last().expect("three fits");
    assert!(
        last.truth_rmse < 0.134 / 2.0,
        "default s(x) at n=16000 still carries the capped-basis bias: RMSE={:.4}",
        last.truth_rmse
    );
}

/// REML puts a smoothing parameter on its rail only with some probability, so
/// a single replicate of a null or linear truth can keep a little wiggliness.
/// The contract is about the typical fit: the median EDF over replicates.
/// Every replicate must also keep its basis at the pilot size, since a truth
/// the pilot already represents gives the adequacy test no reason to grow.
fn median_edf(truth: fn(f64) -> f64, label: &str) -> f64 {
    let mut edfs: Vec<f64> = (0..5u64)
        .map(|seed| {
            let fit = fit_default(4_000, 1_000 + seed, truth);
            eprintln!(
                "{label} seed={seed}: basis_dim={}, edf={:.3}",
                fit.basis_dim, fit.edf
            );
            assert_eq!(
                fit.basis_dim, 11,
                "{label} seed={seed}: a truth the pilot basis represents must not grow it"
            );
            fit.edf
        })
        .collect();
    edfs.sort_by(f64::total_cmp);
    edfs[edfs.len() / 2]
}

#[test]
fn default_smooth_of_pure_noise_still_shrinks_to_the_null() {
    // The REML null's EDF has an atom at 0 and a spread below one degree of
    // freedom; a smooth that manufactured signal would sit at or above 1.
    let median = median_edf(|_| 0.0, "null");
    assert!(
        median < 1.0,
        "a pure-noise response must shrink the default smooth to ~0 EDF, median {median:.3}"
    );
}

#[test]
fn default_smooth_of_a_line_still_shrinks_to_one_edf() {
    let median = median_edf(|x| 2.0 * x - 1.0, "linear");
    assert!(
        (median - 1.0).abs() < 0.5,
        "a linear truth must shrink the default smooth to ~1 EDF, median {median:.3}"
    );
}

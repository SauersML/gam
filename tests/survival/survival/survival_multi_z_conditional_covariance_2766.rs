//! gam#2766 — the marginal-preservation identity under a CONDITIONALLY varying
//! score covariance.
//!
//! `survival_multi_z_fit_hard::survival_multi_z_fit_marginal_preserved_at_true_slopes_population_mc`
//! already pins the identity
//!
//! ```text
//!   E_z[Φ(−(c·q + rᵀz))] = Φ(−q)     with   c = √(1 + rᵀΣr)
//! ```
//!
//! as a POPULATION average, where one pooled `Σ` is the right object because
//! the sample it is drawn from is homogeneous — `Σ̄` IS the population
//! covariance, so that test cannot fail for this reason. This module pins the
//! same identity CONDITIONALLY, which is where the pooled object stops being
//! right: with `Cov(z₀, z₁ | a)` a function of a marginal covariate, the
//! population average keeps passing while every conditional average is wrong by
//! `q·(c̄/c(a) − 1)`.
//!
//! Read the two arms together. On the SAME data, the same slopes and the same
//! bar, the pooled field misses by 8+ standard errors of the stratum average
//! and the conditional field by under 4. That contrast is the regression guard:
//! a revert to one global `Σ` cannot leave both green.
//!
//! ## Why the conditioning span is a cubic and not the bare covariate
//!
//! The conditional model is LINEAR in the span it is given — `φ_jk(a)` and
//! `log d_j(a)` are affine in `a`. The planted correlation here is a bounded
//! sigmoid `0.8·x/√(1+x²)`, which no line in `x` matches at the extremes, so a
//! bare `[x]` span leaves a real, systematic residual (measured: the worst
//! stratum sits at 2.9 SE instead of 1.4, and it grows with the slope rather
//! than staying at the noise floor — the signature of model error, not of
//! Monte-Carlo error).
//!
//! That is a property of the method worth stating rather than hiding, and the
//! fixture states it by supplying the span a real fit would have: a marginal
//! design carrying a SMOOTH of the covariate the correlation moves with, here a
//! cubic. Nothing about the truth is inside the model class even then — the
//! sigmoid is not a cubic — so the arm below is still measuring recovery under
//! misspecification.

use gam::families::bms::{
    ConditionalScoreCovariance, ScoreCovarianceField, marginal_slope_covariance_from_scores,
};
use gam::families::survival::marginal_slope::{RigidVectorValueWorkspace, survival_marginal_slope_vector_neglog};
use ndarray::{Array1, Array2};

const N: usize = 48_000;
const K: usize = 2;
const PROBIT_SCALE: f64 = 1.0;
/// Number of `ρ(x)` strata the conditional bar is read on. Five keeps thousands
/// of rows in each, so the Monte-Carlo standard error of a stratum average stays
/// far below the effect being measured.
const STRATA: usize = 5;
/// Shared slope pair. Large enough that `c = √(1 + rᵀΣr)` actually depends on
/// `Σ` — at a slope near zero the identity holds under any covariance and there
/// is nothing to test.
const SLOPES: [f64; K] = [1.2, 1.0];

#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
    fn next_normal_pair(&mut self) -> (f64, f64) {
        let u1 = self.next_unit().max(1.0e-300);
        let u2 = self.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        (
            r * (std::f64::consts::TAU * u2).cos(),
            r * (std::f64::consts::TAU * u2).sin(),
        )
    }
}

struct Sample {
    z: Array2<f64>,
    weights: Array1<f64>,
    /// The conditioning span `a(C)`: `[x, x², x³]`, the cubic a marginal design
    /// carrying `s(x)` would present.
    a: Array2<f64>,
}

/// `K = 2` scores whose conditional marginals are EXACTLY standard normal and
/// whose only conditional structure is the off-diagonal:
///
/// ```text
///   z₀ = e₀,   z₁ = φ(x)·e₀ + √(1 − φ(x)²)·e₁,   φ(x) = amplitude·x/√(1+x²)
/// ```
///
/// The conditional mean is zero and both conditional variances are one, so
/// gam#2768's per-coordinate location-scale gate has nothing to correct and the
/// entire departure this module measures is the one gam#2766 names.
///
/// `amplitude = 0` gives a constant (zero) conditional correlation on the same
/// marginals — the control arm.
fn sample(seed: u64, amplitude: f64) -> Sample {
    let mut rng = SplitMix64::new(seed);
    let mut z = Array2::<f64>::zeros((N, K));
    let mut a = Array2::<f64>::zeros((N, 3));
    let mut stratum = Vec::with_capacity(N);
    for row in 0..N {
        let (x, e0) = rng.next_normal_pair();
        let (e1, _unused) = rng.next_normal_pair();
        let squashed = x / (1.0 + x * x).sqrt();
        let phi = amplitude * squashed;
        a[[row, 0]] = x;
        a[[row, 1]] = x * x;
        a[[row, 2]] = x * x * x;
        z[[row, 0]] = e0;
        z[[row, 1]] = phi * e0 + (1.0 - phi * phi).max(0.0).sqrt() * e1;
        stratum.push((((squashed + 1.0) * 0.5 * STRATA as f64) as usize).min(STRATA - 1));
    }
    Sample {
        z,
        weights: Array1::<f64>::ones(N),
        a,
    }
}

fn conditional_field(data: &Sample) -> ScoreCovarianceField {
    let pooled =
        marginal_slope_covariance_from_scores(data.z.view(), &data.weights).expect("pooled Σ");
    let model = ConditionalScoreCovariance::fit(data.z.view(), data.weights.view(), data.a.view())
        .expect("conditional fit")
        .expect("a varying cross-score covariance must escalate");
    ScoreCovarianceField::conditional(pooled, model, data.a.view()).expect("conditional field")
}

fn pooled_field(data: &Sample) -> ScoreCovarianceField {
    ScoreCovarianceField::pooled(
        marginal_slope_covariance_from_scores(data.z.view(), &data.weights).expect("pooled Σ"),
    )
}

/// The production row lane must read the SAME per-row covariance the identity
/// above is stated with. `RigidVectorValueWorkspace` binds the FIELD, not a
/// matrix, so this checks the binding end to end: the same row evaluated at its
/// own index and at a fixed reference index must disagree, and it must not if
/// the field is pooled.
#[test]
fn the_row_program_consumes_the_rows_own_conditional_covariance() {
    let data = sample(0x2766_0C33, 0.8);
    let conditional = conditional_field(&data);
    let pooled = pooled_field(&data);
    let probe_rows = [7usize, 101, 5_000, N / 2, N - 1];
    let reference_row = 0usize;

    let neglog = |field: &ScoreCovarianceField, bind_row: usize, score_row: usize| -> f64 {
        let workspace = RigidVectorValueWorkspace::new(field);
        let z_row = [data.z[[score_row, 0]], data.z[[score_row, 1]]];
        survival_marginal_slope_vector_neglog(
            bind_row,
            0.31,
            0.62,
            0.44,
            &SLOPES,
            &z_row,
            &workspace,
            1.0,
            1.0,
            1.0e-8,
            PROBIT_SCALE,
        )
        .expect("row lane neglog")
    };

    let mut distinct = 0usize;
    for &row in &probe_rows {
        let own = neglog(&conditional, row, row);
        let mismatched = neglog(&conditional, reference_row, row);
        if (own - mismatched).abs() > 1.0e-9 {
            distinct += 1;
        }
        // A POOLED field must be row-invariant, byte for byte — that is what
        // keeps every K=1 fit unchanged.
        assert_eq!(
            neglog(&pooled, row, row).to_bits(),
            neglog(&pooled, reference_row, row).to_bits(),
            "a pooled field must give the same row value at any bound row"
        );
    }
    assert_eq!(
        distinct,
        probe_rows.len(),
        "the row lane must vary with the row's own covariance; only {distinct} of {} probe rows \
         differed from the reference binding",
        probe_rows.len()
    );
}

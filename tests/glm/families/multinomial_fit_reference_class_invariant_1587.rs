//! Regression for #1587: a penalized multinomial-logit GAM fit must be
//! invariant to the arbitrary choice of reference class.
//!
//! The softmax driver reference-codes the `K` classes and (before the fix)
//! penalized only the `K−1` reference-anchored ALR contrasts independently,
//! `½ Σ_a λ_a β_aᵀ S β_a`. That penalty lives in a non-symmetric frame: cycling
//! which class is the baseline (in practice, the class whose label sorts last)
//! penalizes a DIFFERENT set of log-odds contrasts, so the penalized
//! coefficients, the REML-selected smoothing parameters, and hence the predicted
//! class probabilities all drift with the labeling (~1% absolute, NOT shrinking
//! with `n`). Predicted multinomial probabilities are a property of the fitted
//! distribution and are mathematically independent of which class is the
//! baseline, so any dependence is a bug — the multinomial analogue of the ALR
//! simplex sibling #1549.
//!
//! The fix makes the class metric reference-SYMMETRIC: the centered (CLR)
//! penalty `((I_{K-1} − J_{K-1}/K)⊗S)` with a single λ per term shared across
//! classes, realized by whitening the class gauge at the `MultinomialFamily`
//! boundary (closed-form `A = (I − J/K)^{-1/2}`) plus per-term λ tying. This
//! test fits the same 3-class softmax data under the three baseline-cycling
//! labelings, realigns predictions to the original class identities, and pins
//! the cross-labeling probability drift below `1e-3` (the pre-fix defect is
//! ~`1e-2`; refitting the SAME labeling twice agrees to ~`1e-12`, so the bar is
//! far above optimizer noise). It is the R-free Rust sibling of
//! `tests/bug_hunt_multinomial_fit_depends_on_reference_class_test.py`.

use csv::StringRecord;
use gam::families::multinomial::{fit_penalized_multinomial_formula, predict_multinomial_formula};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};

/// Deterministic SplitMix64 → uniform `[0, 1)` PRNG so the fixture is fully
/// reproducible without a dependency on the workspace RNG seeding.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn unif(&mut self) -> f64 {
        // 53-bit mantissa fraction.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Sample `n` rows of a 3-class softmax model with a linear-in-`x` log-odds
/// structure (mirrors the Python oracle's `sample`). Returns `(x, class)` with
/// integer class labels in `0..3`.
fn sample(seed: u64, n: usize) -> (Vec<f64>, Vec<usize>) {
    let mut rng = SplitMix64(seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(1));
    let mut xs = Vec::with_capacity(n);
    let mut cls = Vec::with_capacity(n);
    for _ in 0..n {
        let x = -2.0 + 4.0 * rng.unif();
        let eta = [0.5 + 0.8 * x, -0.3 - 0.5 * x, 0.0];
        let mut p = [eta[0].exp(), eta[1].exp(), eta[2].exp()];
        let s: f64 = p.iter().sum();
        for pk in &mut p {
            *pk /= s;
        }
        let u = rng.unif();
        // class = number of cumulative thresholds u exceeds.
        let c = (u > p[0]) as usize + (u > p[0] + p[1]) as usize;
        xs.push(x);
        cls.push(c);
    }
    (xs, cls)
}

/// Fit `y ~ s(x)` under the given class relabeling and return predicted
/// probabilities on `grid`, with columns REALIGNED to the original class
/// identities `0,1,2` (undoing the labeling-dependent column order).
fn fit_aligned(x: &[f64], cls: &[usize], name_map: [&str; 3], grid: &[f64]) -> Vec<[f64; 3]> {
    let headers = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(cls.iter())
        .map(|(&xv, &c)| StringRecord::from(vec![format!("{xv:.10}"), name_map[c].to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode fit data");
    let model = fit_penalized_multinomial_formula(
        &data,
        "y ~ s(x)",
        &FitConfig::default(),
        1.0,
        100,
        1.0e-7,
    )
    .expect("multinomial fit must succeed");

    // Build the prediction frame (feature column only; the response is never
    // referenced by the saved termspec at predict time).
    let grid_rows: Vec<StringRecord> = grid
        .iter()
        .map(|&g| StringRecord::from(vec![format!("{g:.10}")]))
        .collect();
    let grid_data =
        encode_recordswith_inferred_schema(vec!["x".to_string()], grid_rows).expect("encode grid");
    let pr = predict_multinomial_formula(&model, &grid_data).expect("predict must succeed");

    // `pr` columns follow `model.class_levels` (the sorted labels). Map each
    // original class `c` to the column whose label is `name_map[c]`.
    let col_of: Vec<usize> = (0..3)
        .map(|c| {
            model
                .class_levels
                .iter()
                .position(|lvl| lvl == name_map[c])
                .expect("every relabeled class must appear in class_levels")
        })
        .collect();

    (0..grid.len())
        .map(|r| [pr[[r, col_of[0]]], pr[[r, col_of[1]]], pr[[r, col_of[2]]]])
        .collect()
}

fn max_abs_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b.iter())
        .flat_map(|(ra, rb)| (0..3).map(move |k| (ra[k] - rb[k]).abs()))
        .fold(0.0_f64, f64::max)
}

fn assert_reference_invariant(seed: u64, n: usize) {
    let (x, cls) = sample(seed, n);
    let grid: Vec<f64> = (0..7).map(|i| -1.5 + 3.0 * i as f64 / 6.0).collect();

    // Three labelings that each make a DIFFERENT original class the softmax
    // baseline (the class whose label sorts LAST is the reference K-1):
    //   ["A","B","C"] → ref = class 2; ["B","C","A"] → ref = class 1;
    //   ["C","A","B"] → ref = class 0.
    let a = fit_aligned(&x, &cls, ["A", "B", "C"], &grid);
    let a_again = fit_aligned(&x, &cls, ["A", "B", "C"], &grid);
    let b = fit_aligned(&x, &cls, ["B", "C", "A"], &grid);
    let c = fit_aligned(&x, &cls, ["C", "A", "B"], &grid);

    let refit_noise = max_abs_diff(&a, &a_again);
    assert!(
        refit_noise < 1.0e-9,
        "seed={seed} n={n}: refitting the SAME labeling twice must agree to ~machine \
         precision (got {refit_noise:.3e}); a larger value means the comparison itself is \
         noisy and the cross-reference bound below is meaningless"
    );

    let cross = max_abs_diff(&a, &b)
        .max(max_abs_diff(&a, &c))
        .max(max_abs_diff(&b, &c));
    assert!(
        cross < 1.0e-3,
        "seed={seed} n={n}: predicted class probabilities must be invariant to the \
         arbitrary reference class, but cycling which class is the softmax baseline drifts \
         them by {cross:.3e} (bar 1e-3; the pre-fix reference-anchored penalty drifts ~1e-2). \
         The multinomial smoothing penalty is not reference-symmetric (#1587)."
    );
}

#[test]
fn multinomial_fit_is_invariant_to_reference_class_seed0_n900() {
    init_parallelism();
    assert_reference_invariant(0, 900);
}

#[test]
fn multinomial_fit_is_invariant_to_reference_class_seed2_n900() {
    init_parallelism();
    assert_reference_invariant(2, 900);
}

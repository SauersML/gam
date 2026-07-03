//! #1587: a penalized multinomial-logit GAM fit must NOT depend on the
//! arbitrary choice of reference class.
//!
//! `fit_penalized_multinomial_formula` reference-codes the `K` classes, pinning
//! the class whose (sorted) label is last as the softmax baseline (`η_ref ≡ 0`)
//! and penalizing the wiggliness of the `K−1` contrasts `η_a = log(p_a/p_ref)`.
//! The predicted class probabilities are a property of the fitted distribution
//! and are mathematically invariant to which class is the baseline. Two sources
//! of reference dependence existed:
//!   1. an independent per-(class,term) smoothing parameter `λ_{a,t}` (which
//!      permutes — not symmetrically — under relabeling), and
//!   2. the reference-anchored ("Diagonal") penalty metric
//!      `Σ_a β_aᵀ S β_a`, which is not a symmetric function of the K classes.
//!
//! The fix ties `λ` per term across classes AND installs the reference-symmetric
//! CLR ("centered") metric `λ_t·((I−J/K)⊗S_t)` via a symmetric whitening of the
//! class gauge (the multinomial analogue of the resolved ALR sibling #1549).
//! This test fits the SAME data under three relabelings that each make a
//! different original class the baseline, aligns every fit back to the original
//! class identities, and asserts the predicted probabilities agree to near
//! machine precision (and that a refit of one labeling is deterministic).

use csv::StringRecord;
use gam::families::multinomial::{fit_penalized_multinomial_formula, predict_multinomial_formula};
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};

/// Deterministic LCG → `U[0,1)`; no external RNG so labels are byte-identical
/// run-to-run (the determinism the drift bound depends on).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    fn next_u01(&mut self) -> f64 {
        // Numerical Recipes 64-bit LCG constants.
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Top 53 bits → [0,1).
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// Draw a clean 3-class softmax regression sample (integer classes 0/1/2),
/// mirroring the Python repro (`eta = [0.5+0.8x, -0.3-0.5x, 0]`).
fn sample_classes(seed: u64, n: usize) -> (Vec<f64>, Vec<usize>) {
    let mut rng = Lcg::new(seed);
    let mut xs = Vec::with_capacity(n);
    let mut cls = Vec::with_capacity(n);
    for _ in 0..n {
        let x = -2.0 + 4.0 * rng.next_u01();
        let eta = [0.5 + 0.8 * x, -0.3 - 0.5 * x, 0.0];
        let m = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut p = [0.0f64; 3];
        let mut denom = 0.0;
        for c in 0..3 {
            p[c] = (eta[c] - m).exp();
            denom += p[c];
        }
        for c in 0..3 {
            p[c] /= denom;
        }
        let u = rng.next_u01();
        let mut cum = 0.0;
        let mut drawn = 2;
        for c in 0..3 {
            cum += p[c];
            if u < cum {
                drawn = c;
                break;
            }
        }
        xs.push(x);
        cls.push(drawn);
    }
    (xs, cls)
}

/// Fit `y ~ s(x)` with `cls` relabeled by `name_map` (original class `c` gets
/// string label `name_map[c]`), predict on `grid`, and return predicted
/// probabilities re-aligned to the ORIGINAL class order 0/1/2.
fn fit_predict_aligned(
    xs: &[f64],
    cls: &[usize],
    name_map: [&str; 3],
    grid: &[f64],
) -> ndarray::Array2<f64> {
    let headers = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = xs
        .iter()
        .zip(cls.iter())
        .map(|(x, &c)| StringRecord::from(vec![format!("{x:.8}"), name_map[c].to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode training data");
    let model = fit_penalized_multinomial_formula(
        &data,
        "y ~ s(x)",
        &FitConfig::default(),
        1.0,
        200,
        1.0e-7,
    )
    .expect("multinomial smooth fit must succeed");

    // Predict frame: just the x grid (predict resolves features by name).
    let grid_rows: Vec<StringRecord> = grid
        .iter()
        .map(|x| StringRecord::from(vec![format!("{x:.8}")]))
        .collect();
    let grid_data =
        encode_recordswith_inferred_schema(vec!["x".to_string()], grid_rows).expect("encode grid");
    let pred = predict_multinomial_formula(&model, &grid_data).expect("predict");
    // pred columns are in sorted-label order; column for original class c is at
    // the rank of name_map[c] among the sorted labels.
    let mut sorted = name_map;
    sorted.sort_unstable();
    let col_of_class: [usize; 3] =
        std::array::from_fn(|c| sorted.iter().position(|&s| s == name_map[c]).unwrap());
    let n = grid.len();
    let mut aligned = ndarray::Array2::<f64>::zeros((n, 3));
    for r in 0..n {
        for c in 0..3 {
            aligned[[r, c]] = pred[[r, col_of_class[c]]];
        }
    }
    aligned
}

const LABELINGS: [[&str; 3]; 3] = [["A", "B", "C"], ["B", "C", "A"], ["C", "A", "B"]];

fn grid() -> Vec<f64> {
    (0..7).map(|i| -1.5 + 3.0 * i as f64 / 6.0).collect()
}

#[test]
fn multinomial_fit_invariant_to_reference_class_1587() {
    init_parallelism();
    let g = grid();
    for seed in [0u64, 1, 2, 3] {
        let (xs, cls) = sample_classes(seed, 900);
        let preds: Vec<_> = LABELINGS
            .iter()
            .map(|nm| fit_predict_aligned(&xs, &cls, *nm, &g))
            .collect();

        // Structural: valid simplex.
        for (nm, p) in LABELINGS.iter().zip(preds.iter()) {
            for r in 0..g.len() {
                let s: f64 = (0..3).map(|c| p[[r, c]]).sum();
                assert!(
                    (s - 1.0).abs() < 1e-9,
                    "labeling {nm:?}: predicted probabilities do not sum to 1 (got {s})"
                );
                for c in 0..3 {
                    assert!(
                        p[[r, c]] >= -1e-12 && p[[r, c]] <= 1.0 + 1e-12,
                        "labeling {nm:?}: probability out of [0,1]"
                    );
                }
            }
        }

        let mut drift = 0.0f64;
        for i in 0..preds.len() {
            for j in (i + 1)..preds.len() {
                let d = (&preds[i] - &preds[j])
                    .iter()
                    .fold(0.0f64, |m, v| m.max(v.abs()));
                drift = drift.max(d);
            }
        }
        assert!(
            drift < 1e-3,
            "seed {seed}: multinomial fit depends on the reference class \
             (max cross-labeling probability drift {drift:.3e} >= 1e-3)"
        );
    }
}

#[test]
fn multinomial_refit_same_labeling_is_deterministic_1587() {
    init_parallelism();
    let g = grid();
    let (xs, cls) = sample_classes(0, 900);
    let a = fit_predict_aligned(&xs, &cls, LABELINGS[0], &g);
    let b = fit_predict_aligned(&xs, &cls, LABELINGS[0], &g);
    let noise = (&a - &b).iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        noise < 1e-9,
        "refitting the same labeling twice must be deterministic; got noise {noise:.3e}"
    );
}

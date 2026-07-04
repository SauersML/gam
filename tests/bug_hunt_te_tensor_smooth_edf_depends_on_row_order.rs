//! Bug hunt / invariant guard: a `te(x, z)` tensor-product smooth's reported
//! EDF (and, with it, its standard errors and AIC) must be **invariant to the
//! ROW ORDER of the training frame** (#2123).
//!
//! Rows are exchangeable — the REML marginal-likelihood objective is a plain sum
//! over rows — so every fitted and inferential quantity is an invariant of the
//! (unordered) data set and cannot depend on the order the rows were listed in.
//! The reported failure mode is a `te(x, z)` fit whose outer REML lands in a
//! different basin on one particular row order: on the original order it fails
//! to converge and the fallback substitutes the FULL tensor basis dimension
//! (`edf = 52`, a near-interpolating fit) — shipping standard errors ~3.4× too
//! small — while every row permutation converges to the parsimonious optimum
//! (`edf ≈ 13.5`). The two fits explain the data equally well (their deviance is
//! essentially identical), so the reported inference is simply wrong on the
//! order that fails.
//!
//! The mechanism is a row-order-sensitive floating-point reduction upstream of
//! the outer λ-selection: `XᵀWX` is summed in row order (`xt_diag_x_dense_into`
//! → faer `fast_atb`), so a row permutation perturbs the Gram at the ULP level,
//! which the outer REML seed selection and basin certification consume through
//! an order-sensitive `compute_cost`. At a near-tie between two macroscopically
//! different REML optima the sub-ULP cost difference — which flips with row
//! order — tips which basin the optimizer descends into.
//!
//! This test fits `y ~ te(x, z)` on the original row order and on a fixed stride
//! permutation across a block of datasets and asserts the reported total EDF
//! agrees (with the deviance agreeing tightly as an anchor that both fits are
//! equally good). It is the permutation sibling of the te margin-order invariant
//! (`ti_anova_tensor_invariant_to_margin_order_1593`) and the REML-invariance
//! family (#1378 row permutation, #1214/#1269/#1456 covariate rescale /
//! translation / rotation).

use csv::StringRecord;
use gam::{FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula};

/// Deterministic SplitMix64 → no Python, no external RNG crate.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// Uniform on (0, 1).
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// Standard normal via Box–Muller (one of the pair).
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// The Gaussian dataset from the issue: `x, z ~ U(0,1)`,
/// `y = sin(2πx) + 0.7 z + N(0, noise)`, in the generated (original) row order.
fn build_rows(seed: u64, n: usize, noise: f64) -> Vec<[f64; 3]> {
    let mut rng = SplitMix64::new(seed);
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.unit();
        let z = rng.unit();
        let y = (std::f64::consts::TAU * x).sin() + 0.7 * z + noise * rng.normal();
        rows.push([x, z, y]);
    }
    rows
}

/// Encode `[x, z, y]` rows through the public inferred-schema path (exactly what
/// a user's CSV would produce).
fn encode(rows: &[[f64; 3]]) -> gam::data::EncodedDataset {
    let records: Vec<StringRecord> = rows
        .iter()
        .map(|r| StringRecord::from(vec![r[0].to_string(), r[1].to_string(), r[2].to_string()]))
        .collect();
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode dataset")
}

/// Fit `y ~ te(x, z)` (Gaussian identity) and return `(total_edf, deviance)` as
/// a user reads them off the public fit result.
fn fit_te(data: &gam::data::EncodedDataset) -> (f64, f64) {
    let cfg = FitConfig::default();
    let FitResult::Standard(fit) =
        fit_from_formula("y ~ te(x, z)", data, &cfg).expect("standard tensor GAM fit")
    else {
        panic!("expected a standard GAM fit for y ~ te(x, z)");
    };
    let edf = fit.fit.edf_total().expect("fit reports total edf");
    (edf, fit.fit.deviance)
}

/// A fixed, deterministic row permutation: place row `i` at position
/// `(i * 157) mod n`. gcd(157, 300) = 1, so this is a bijection; 157 is coprime
/// to 300 and far from 0/n, so it thoroughly reorders the frame — manifestly not
/// the identity.
fn stride_permute(rows: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let n = rows.len();
    let mut perm = vec![[0.0f64; 3]; n];
    for (i, row) in rows.iter().enumerate() {
        perm[(i * 157) % n] = *row;
    }
    perm
}

/// Assert that the reported total EDF (and the deviance anchor) of a `te(x, z)`
/// fit is invariant to a row permutation of one dataset.
fn assert_row_order_invariant(seed: u64, noise: f64) {
    let n = 300usize;
    let original = build_rows(seed, n, noise);
    let permuted = stride_permute(&original);

    let (edf_orig, dev_orig) = fit_te(&encode(&original));
    let (edf_perm, dev_perm) = fit_te(&encode(&permuted));

    eprintln!(
        "[row-order] seed={seed} noise={noise}: original edf={edf_orig:.3} dev={dev_orig:.4} | \
         permuted edf={edf_perm:.3} dev={dev_perm:.4}"
    );

    // Anchor: the two fits describe the SAME data and explain it equally well —
    // their deviance is essentially identical. A tight deviance agreement proves
    // any EDF disagreement is a pure inference/convergence artifact, not two
    // genuinely different fits.
    let dev_scale = dev_orig.abs().max(dev_perm.abs()).max(1.0);
    let dev_rel = (dev_orig - dev_perm).abs() / dev_scale;
    assert!(
        dev_rel < 5.0e-3,
        "deviance must be row-order invariant (same data): seed={seed} noise={noise} original \
         {dev_orig:.6} vs permuted {dev_perm:.6} (rel {dev_rel:.3e})."
    );

    // The bug: reported total EDF must be row-order invariant. Rows are
    // exchangeable, so the effective dimension of the fitted smoother cannot
    // depend on the order they were listed in. The pathological order rails the
    // outer REML into the non-converged full-basis fallback (edf ≈ 52) while
    // permutations converge (edf ≈ 13.5) — a gap of ~38. The 2.0 tolerance is
    // generous round-off headroom around that gap; do NOT weaken it.
    let edf_gap = (edf_orig - edf_perm).abs();
    assert!(
        edf_gap < 2.0,
        "te(x,z) reported total EDF depends on ROW ORDER: seed={seed} noise={noise} original \
         edf={edf_orig:.3} vs permuted edf={edf_perm:.3} (gap {edf_gap:.3}). Rows are \
         exchangeable — the REML objective is a sum over rows — so every fitted/inferential \
         quantity, EDF included, must be invariant to a row permutation. Deviance agrees to \
         {dev_rel:.1e}, so both fits are equally good — the EDF is just wrong."
    );
}

#[test]
fn te_tensor_fit_edf_is_invariant_to_row_order() {
    // A block of (seed, noise) draws spanning the moderate/low-noise regime where
    // the te(x,z) REML is most susceptible to a near-tie between a converged
    // (edf ≈ 13) and a near-interpolating (edf ≈ 52) basin. Every one must be
    // row-order invariant; a single order-dependent draw ships a fit whose
    // EDF/SE/AIC change under a cosmetic row reordering.
    for &(seed, noise) in &[
        (1_u64, 0.15_f64),
        (3, 0.15),
        (6, 0.15),
        (1, 0.2),
        (3, 0.25),
        (11, 0.15),
    ] {
        assert_row_order_invariant(seed, noise);
    }
}

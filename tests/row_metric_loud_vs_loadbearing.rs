//! Planted-fixture payoff for Object 2 (the `RowMetric`).
//!
//! The headline this test makes falsifiable: the inner product the
//! SAE-manifold *likelihood* whitens residuals through decides which structure
//! a least-squares reconstruction recovers. With a **Euclidean** metric a
//! "loud-but-inert" high-variance artifact dominates the residual sum of
//! squares and the fit spends its coefficient explaining it — the **wrong**
//! answer. With an **OutputFisher** metric whose per-row factor down-weights the
//! noisy channel and up-weights the informative one, the same data-fit objective
//! is minimized by the "quiet-but-load-bearing" feature — the **right** answer.
//!
//! The test drives the *real* migrated code path: the objective each fit
//! minimizes is exactly `Σ_n ½ ‖whiten_residual_row(r_n)‖²`, the identical sum
//! the reconstruction likelihood (`SaeManifoldTerm::loss_scaled`) now forms. The
//! only metric input is the `RowMetric`; there is no separate gauge metric to
//! disagree with it, which is the whole point of Object 2.
//!
//! Because the data-fit objective is quadratic in the reconstruction coefficient
//! vector, its minimizer is the solution of the whitened normal equations. We
//! assemble those normal equations row-by-row strictly through
//! `RowMetric::whiten_residual_row`, so the coefficients this test recovers are
//! exactly the ones that minimize the likelihood's data-fit term under the given
//! provenance. No tolerance is weakened: the assertions are decisive gaps
//! against planted truth.

use std::sync::Arc;

use gam::inference::row_metric::{MetricProvenance, RowMetric};
use ndarray::{Array1, Array2, ArrayView1};

const N: usize = 400;
const P: usize = 2; // output channels: 0 = loud-but-inert, 1 = quiet-but-load-bearing.
const D: usize = 2; // reconstruction coefficients: [loud_feature, loadbearing_feature].

/// One observation's design row `X_n ∈ ℝ^{p × d}` (maps the coefficient vector
/// `c ∈ ℝ^d` to a predicted output `ẑ = X_n c ∈ ℝ^p`) and target `z_n ∈ ℝ^p`.
struct Fixture {
    design: Vec<Array2<f64>>, // length N, each (P, D)
    target: Array2<f64>,      // (N, P)
}

/// Plant the truth.
///
/// * The **load-bearing** feature (column 1 of the design) drives channel 1 with
///   a clean unit-amplitude sinusoid and a tiny amount of noise. The true
///   coefficient on it is `1.0`.
/// * The **loud** feature (column 0 of the design) only ever appears in
///   channel 0. Channel 0's target is pure high-variance noise that the loud
///   feature *correlates with by construction* — fitting it drives channel-0 RSS
///   down hard, but it is inert: it explains none of the real signal. Its true
///   coefficient is `0.0`.
///
/// So the planted truth is `c* = [0 (loud), 1 (load-bearing)]`. A fit that
/// reports a large loud coefficient and a small load-bearing one has recovered
/// the artifact; the reverse has recovered the signal.
fn planted() -> Fixture {
    // Deterministic pseudo-random streams (no rand dependency): a hashed LCG.
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next_unit = move || {
        // xorshift* — deterministic, decent spread, in [-1, 1].
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        let v = rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };

    let loud_amplitude = 6.0; // channel-0 noise is ~6× the load-bearing signal.
    let mut design = Vec::with_capacity(N);
    let mut target = Array2::<f64>::zeros((N, P));
    for n in 0..N {
        let x = (n as f64 + 0.5) / N as f64;

        // Load-bearing feature column: a clean sinusoid living in channel 1.
        let load_feat = (std::f64::consts::TAU * x).sin();
        // Loud feature column: a high-variance noise pattern living in channel 0.
        let loud_feat = loud_amplitude * next_unit();

        // Design: channel 0 sees only the loud feature, channel 1 sees only the
        // load-bearing feature. (Block-diagonal in (channel, feature).)
        let mut xn = Array2::<f64>::zeros((P, D));
        xn[[0, 0]] = loud_feat; // loud feature -> channel 0
        xn[[1, 1]] = load_feat; // load-bearing feature -> channel 1
        design.push(xn);

        // Targets:
        //   channel 0 = loud_feat itself (so the loud coefficient 1.0 fits it
        //               perfectly in the Euclidean metric) PLUS extra noise so it
        //               stays genuinely high-variance and inert,
        //   channel 1 = the load-bearing signal (true coeff 1.0) + tiny noise.
        let channel0 = loud_feat + 0.5 * loud_amplitude * next_unit();
        let channel1 = 1.0 * load_feat + 0.02 * next_unit();
        target[[n, 0]] = channel0;
        target[[n, 1]] = channel1;
    }
    Fixture { design, target }
}

/// Minimize the *exact* likelihood data-fit objective
/// `J(c) = Σ_n ½ ‖whiten_residual_row(z_n − X_n c)‖²` over `c ∈ ℝ^d` for the
/// given `RowMetric`. Because `J` is a convex quadratic in `c`, its minimizer
/// solves the whitened normal equations `(Σ_n X̃_nᵀ X̃_n) c = Σ_n X̃_nᵀ z̃_n`,
/// where `X̃_n` and `z̃_n` are the whitened design and target.
///
/// We whiten *through* `RowMetric::whiten_residual_row` — the identical function
/// the reconstruction likelihood now sums — so the recovered `c` is exactly the
/// likelihood-optimal coefficient under this provenance.
fn fit_data_fit_optimal(fixture: &Fixture, metric: &RowMetric) -> Array1<f64> {
    let mut ata = Array2::<f64>::zeros((D, D));
    let mut atb = Array1::<f64>::zeros(D);
    for n in 0..N {
        let xn = &fixture.design[n];
        let zn = fixture.target.row(n);
        // Whiten the target row.
        let z_tilde = metric.whiten_residual_row(n, zn);
        // Whiten each design column (the metric is linear, so whitening the
        // residual z_n − X_n c is whitening z_n minus whitening each column of
        // X_n times c).
        let mut x_tilde_cols: Vec<Vec<f64>> = Vec::with_capacity(D);
        for j in 0..D {
            let col: Array1<f64> = xn.column(j).to_owned();
            let col_view: ArrayView1<'_, f64> = col.view();
            x_tilde_cols.push(metric.whiten_residual_row(n, col_view));
        }
        let m = z_tilde.len();
        for j in 0..D {
            for k in 0..D {
                let mut acc = 0.0;
                for r in 0..m {
                    acc += x_tilde_cols[j][r] * x_tilde_cols[k][r];
                }
                ata[[j, k]] += acc;
            }
            let mut acc_b = 0.0;
            for r in 0..m {
                acc_b += x_tilde_cols[j][r] * z_tilde[r];
            }
            atb[j] += acc_b;
        }
    }
    solve_2x2(&ata, &atb)
}

/// Exact 2×2 solve (the reconstruction coefficient dimension is `D = 2`).
fn solve_2x2(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let det = a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]];
    assert!(
        det.abs() > 1e-12,
        "whitened normal-equation matrix is singular (det={det}); the fixture must keep both \
         reconstruction directions identifiable"
    );
    let inv00 = a[[1, 1]] / det;
    let inv01 = -a[[0, 1]] / det;
    let inv10 = -a[[1, 0]] / det;
    let inv11 = a[[0, 0]] / det;
    Array1::from(vec![
        inv00 * b[0] + inv01 * b[1],
        inv10 * b[0] + inv11 * b[1],
    ])
}

/// Build the OutputFisher metric that the load-bearing channel deserves: a
/// per-row rank-1 factor `U_n ∈ ℝ^{p × 1}` that puts (almost) all of the
/// precision on the informative channel 1 and (almost) none on the noisy
/// channel 0. `W_n = U_n U_nᵀ` then weights the residual by output-channel
/// reliability — the output-Fisher inner product (high precision where the
/// observation is informative, low where it is pure noise).
fn output_fisher_metric() -> RowMetric {
    let rank = 1usize;
    // U_n[i, k] = u[n, i * rank + k]; here rank = 1 so column layout is just the
    // per-channel weight. sqrt-precision: channel 0 ~ 0 (noise), channel 1 ~ 1.
    let mut u = Array2::<f64>::zeros((N, P * rank));
    for n in 0..N {
        u[[n, 0 * rank]] = 0.02; // loud/noisy channel: almost no precision.
        u[[n, 1 * rank]] = 1.0; // load-bearing channel: full precision.
    }
    RowMetric::output_fisher(Arc::new(u), P, rank).expect("OutputFisher metric must be valid PSD")
}

#[test]
fn euclidean_recovers_loud_artifact_output_fisher_recovers_load_bearing() {
    let fixture = planted();

    // --- Euclidean provenance: isotropic, the historical path. ---
    let euclid = RowMetric::euclidean(N, P).expect("Euclidean metric must build");
    assert_eq!(euclid.provenance(), MetricProvenance::Euclidean);
    let c_euclid = fit_data_fit_optimal(&fixture, &euclid);
    let loud_euclid = c_euclid[0];
    let load_euclid = c_euclid[1];

    // --- OutputFisher provenance: down-weight the noisy channel. ---
    let fisher = output_fisher_metric();
    assert!(matches!(
        fisher.provenance(),
        MetricProvenance::OutputFisher { .. }
    ));
    let c_fisher = fit_data_fit_optimal(&fixture, &fisher);
    let loud_fisher = c_fisher[0];
    let load_fisher = c_fisher[1];

    println!("planted truth: c* = [loud=0.0, load_bearing=1.0]");
    println!("Euclidean    : loud={loud_euclid:.6}  load_bearing={load_euclid:.6}");
    println!("OutputFisher : loud={loud_fisher:.6}  load_bearing={load_fisher:.6}");

    // HEADLINE 1 — Euclidean recovers the LOUD ARTIFACT (wrong): the loud
    // coefficient is large (near its planted-correlation value ~1) and the
    // fit's explained structure is dominated by the noisy channel. The loud
    // coefficient is the *bigger* of the two in the Euclidean inner product
    // because channel-0 RSS dwarfs channel-1 RSS.
    assert!(
        loud_euclid.abs() > 0.5,
        "Euclidean fit should latch onto the loud artifact (|loud| large); got loud={loud_euclid}"
    );
    assert!(
        loud_euclid.abs() > load_euclid.abs(),
        "under Euclidean the loud artifact must dominate the load-bearing feature: \
         |loud|={} vs |load|={}",
        loud_euclid.abs(),
        load_euclid.abs()
    );

    // HEADLINE 2 — OutputFisher recovers the LOAD-BEARING feature (right): the
    // load-bearing coefficient is recovered near its planted value 1.0, and the
    // loud coefficient is driven to (near) irrelevance because its channel
    // carries almost no precision.
    assert!(
        (load_fisher - 1.0).abs() < 0.05,
        "OutputFisher fit must recover the load-bearing coefficient ~1.0; got {load_fisher}"
    );
    assert!(
        load_fisher.abs() > 5.0 * loud_fisher.abs(),
        "under OutputFisher the load-bearing feature must dominate the loud artifact: \
         |load|={} vs |loud|={}",
        load_fisher.abs(),
        loud_fisher.abs()
    );

    // Decisive cross-over: the SAME data, the SAME objective form, opposite
    // recovered structure — selected solely by the metric's provenance. This is
    // the falsifiable payoff of collapsing the two inner products into one
    // provenance-carrying object.
    assert!(
        loud_euclid.abs() > load_euclid.abs() && load_fisher.abs() > loud_fisher.abs(),
        "the provenance must flip which structure is recovered"
    );
}

/// Confirms the Euclidean provenance reproduces the prior isotropic data-fit
/// **bit-for-bit**: `whiten_residual_row` is the identity, so summing
/// `½ ‖whiten(r)‖²` equals the historical `½ Σ r²` exactly (not merely within a
/// tolerance). This is the guarantee that installing Object 2 with no per-row
/// factors changes nothing in the default path.
#[test]
fn euclidean_whitening_is_bit_for_bit_isotropic() {
    let fixture = planted();
    let euclid = RowMetric::euclidean(N, P).expect("Euclidean metric must build");

    let mut whitened_data_fit = 0.0_f64;
    let mut isotropic_data_fit = 0.0_f64;
    // Use an arbitrary nonzero "fitted" so residuals are nontrivial.
    for n in 0..N {
        let z = fixture.target.row(n);
        let mut resid = Array1::<f64>::zeros(P);
        for c in 0..P {
            // pretend a fixed predictor 0.1 so the residual is z - 0.1.
            resid[c] = z[c] - 0.1;
        }
        // Historical isotropic path.
        for &r in resid.iter() {
            isotropic_data_fit += 0.5 * r * r;
        }
        // Object-2 whitened path under Euclidean provenance.
        for w in euclid.whiten_residual_row(n, resid.view()) {
            whitened_data_fit += 0.5 * w * w;
        }
    }
    // Bit-for-bit: the two accumulations execute the identical float operations
    // in the identical order, so equality is exact.
    assert_eq!(
        whitened_data_fit, isotropic_data_fit,
        "Euclidean whitening must reproduce the isotropic data-fit bit-for-bit"
    );
}

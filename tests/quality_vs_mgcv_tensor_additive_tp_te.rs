//! End-to-end quality: gam's *additive* combination of two heterogeneous smooth
//! types — an isotropic 2-D thin-plate `s(x1, x2, bs="tp")` plus a separable
//! anisotropic tensor `te(z, w)` — must match `mgcv`, the mature, de-facto
//! standard GAM implementation, on the *same* data.
//!
//! Reference tool:
//!   `mgcv::gam(y ~ s(x1, x2, bs="tp", k=10) + te(z, w, k=6), method="REML")`.
//!
//! Why this combination matters: real GAM applications almost never use a single
//! smooth. An additive model stacks one penalty block per smooth term into a
//! single penalized objective, and the REML optimizer selects a smoothing
//! parameter *per block* simultaneously. This is precisely where multi-smooth
//! aggregation bugs hide — penalty-matrix assembly that lets blocks cross-talk,
//! rank deficiency from mis-stacked null spaces, or a lambda-selection coupling
//! that one engine has and the other does not. Here the two blocks are
//! deliberately of *different* mathematical character: an isotropic
//! rotation-invariant thin-plate radial smooth over (x1, x2), and a separable
//! row-wise-Kronecker tensor smooth over (z, w). Both individual smooths are
//! covered by sibling tests; this test isolates the *aggregation* logic. A
//! divergence here, even with both individual smooths correct, flags a bug in
//! gam's penalty-block stacking or cross-block lambda selection.
//!
//! Data: deterministic synthetic, n=500, fixed seed=20260530. Covariates
//! x1, x2, z, w are independent uniform draws on [0,1] from a reproducible
//! SplitMix64 stream (identical bytes fed to gam and to mgcv via the shared
//! CSV). The response is the additive truth
//!     f1(x1, x2) = sin(pi*x1) * exp(-x2)        (smooth, thin-plate-representable)
//!   + f2(z, w)   = z^2 * cos(pi*w)              (separable, tensor-representable)
//! with no added noise, so the penalized solution is fully identifiable and any
//! divergence is a genuine difference in how the two engines combine the blocks,
//! not sampling variation.
//!
//! We compare the quantity that matters for an additive model: the total fitted
//! mean (sum of all blocks + intercept) evaluated on the training points.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
use ndarray::Array2;
use std::f64::consts::PI;

/// SplitMix64 -> uniform [0,1). Deterministic, seedable, no external RNG crate:
/// guarantees gam and mgcv receive byte-identical covariates.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn next_unit(state: &mut u64) -> f64 {
    // Top 53 bits -> [0,1).
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

#[test]
fn gam_additive_tp_plus_te_matches_mgcv() {
    init_parallelism();

    // ---- deterministic synthetic data: n=500, seed=20260530 ----------------
    const N: usize = 500;
    let mut state: u64 = 20260530;
    let mut x1 = Vec::with_capacity(N);
    let mut x2 = Vec::with_capacity(N);
    let mut z = Vec::with_capacity(N);
    let mut w = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let a = next_unit(&mut state);
        let b = next_unit(&mut state);
        let c = next_unit(&mut state);
        let d = next_unit(&mut state);
        // Additive truth: thin-plate-friendly f1 + separable tensor f2, no noise.
        let f1 = (PI * a).sin() * (-b).exp();
        let f2 = c * c * (PI * d).cos();
        x1.push(a);
        x2.push(b);
        z.push(c);
        w.push(d);
        y.push(f1 + f2);
    }

    // ---- fit with gam: y ~ s(x1,x2,bs="tp",k=10) + te(z,w,k=6), REML --------
    let headers = ["x1", "x2", "z", "w", "y"]
        .into_iter()
        .map(String::from)
        .collect();
    let rows: Vec<StringRecord> = (0..N)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", x1[i]),
                format!("{:.17e}", x2[i]),
                format!("{:.17e}", z[i]),
                format!("{:.17e}", w[i]),
                format!("{:.17e}", y[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode additive dataset");
    let col = ds.column_map();
    let x1_idx = col["x1"];
    let x2_idx = col["x2"];
    let z_idx = col["z"];
    let w_idx = col["w"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(x1, x2, bs=\"tp\", k=10) + te(z, w, k=6)",
        &ds,
        &cfg,
    )
    .expect("gam additive tp+te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for an additive gaussian tp+te model");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam total fitted mean at the training points: rebuild the design from the
    // frozen spec (all blocks + intercept) at the observed covariates. Identity
    // link => design*beta is exactly the additive fitted mean.
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for i in 0..N {
        grid[[i, x1_idx]] = x1[i];
        grid[[i, x2_idx]] = x2[i];
        grid[[i, z_idx]] = z[i];
        grid[[i, w_idx]] = w[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild additive design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME additive model with mgcv (the mature reference) ------
    let r = run_r(
        &[
            Column::new("x1", &x1),
            Column::new("x2", &x2),
            Column::new("z", &z),
            Column::new("w", &w),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ s(x1, x2, bs = "tp", k = 10) + te(z, w, k = 6),
                 data = df, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_fitted.len(), N, "mgcv fitted length mismatch");

    // ---- compare the total additive fit on the training points -------------
    let rel = relative_l2(&gam_fitted, mgcv_fitted);
    let corr = pearson(&gam_fitted, mgcv_fitted);
    let edf_rel = (gam_edf - mgcv_edf).abs() / mgcv_edf.abs().max(1.0);

    eprintln!(
        "additive s(x1,x2,tp,k=10)+te(z,w,k=6): n={N} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         rel_l2={rel:.5} pearson={corr:.6} edf_rel={edf_rel:.4}"
    );

    // Both engines REML-fit the *same* noise-free, additively-generated data,
    // each block representable in its own basis (thin-plate for the smooth f1,
    // k=6 tensor for the separable f2). The only freedom is per-block penalty
    // shrinkage, which both apply against the same penalized objective. The
    // total additive fits must therefore essentially coincide; rel_l2 < 0.025
    // (2.5% of the total-fit norm) is a tight, principled bound that any genuine
    // cross-block penalty-assembly or lambda-selection bug would break, while
    // still allowing for the two engines' differing thin-plate truncation and
    // tensor centering conventions.
    assert!(
        corr > 0.999,
        "additive tp+te total fit should be near-identical to mgcv: pearson={corr:.6}"
    );
    assert!(
        rel < 0.025,
        "additive tp+te total fit diverges from mgcv: rel_l2={rel:.5}"
    );
    // Total EDF (summed over BOTH penalty blocks + intercept) is the most direct
    // observable of correct cross-block aggregation: a lambda-selection coupling
    // bug can leave the noise-free fitted surface near-identical while
    // mis-attributing complexity between blocks, which shows up in the *sum*.
    // Each block carries the same basis/null-space-convention slack vs mgcv that
    // its sibling test allows (tp ~15%), so on the additive total — where the
    // two engines also differ in thin-plate truncation vs k=6 tensor centering —
    // a 20% relative bound tracks the selected complexity without being trivial.
    assert!(
        edf_rel < 0.20,
        "additive tp+te effective degrees of freedom disagree: \
         gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.4})"
    );
}

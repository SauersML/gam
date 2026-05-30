//! End-to-end quality: gam's compositional-geometry primitive must match the
//! mature **R `compositions`** package (Aitchison-geometry CoDa reference) on
//! identical data.
//!
//! DISTINCTIVE-AXIS / FRAGMENTATION FINDING. The spec (`test_001`) names the
//! Additive Log-Ratio (ALR) transform and a public `closure()` as the gam
//! capability to benchmark against `robCompositions::addLR()`. Reading the gam
//! source (`src/geometry/simplex.rs`) shows the honest state of the world:
//!   * gam has **no** ALR / CLR / ILR transform of any kind, and
//!   * `closure()` exists but is **private** (not reachable from a test crate).
//! The *only* public compositional primitive gam exposes is
//! `gam::geometry::simplex::simplex_frechet_mean`. That function is itself the
//! load-bearing Aitchison-geometry operation: it (1) applies closure to each
//! row (the sum-to-1 / nonnegativity simplex projection the spec cares about),
//! then (2) takes the weighted geometric mean in log space and renormalizes —
//! i.e. it computes the **closed geometric mean** (the Aitchison center) of a
//! data cloud on the simplex. So we benchmark the primitive gam *actually has*,
//! against the primitive the mature library *actually computes*:
//! `compositions::mean(acomp(X))` is precisely the closed geometric mean of a
//! compositional sample. This is the faithful head-to-head; the fragmentation
//! of the CoDa tooling (compositions vs robCompositions vs single-purpose CoDa
//! packages, none offering a fitted-compositional-response GAM) is itself the
//! finding, documented per the distinctive-axis rule.
//!
//! We feed the *same* raw (unclosed, strictly positive) 50x4 composition matrix
//! to both engines and assert:
//!   1. element-wise relative L2 between gam's `simplex_frechet_mean` and
//!      `compositions::mean(acomp(.))` (the mature closed geometric mean),
//!   2. the INTRINSIC simplex-closure properties gam must satisfy on its output:
//!      sum-to-1 (max abs deviation from 1.0) and strict nonnegativity, and
//!   3. that the weighted-Frechet path agrees with an explicitly R-computed
//!      weighted closed geometric mean (closure is exact, so this is tight).
//! Both engines compute the same population quantity from identical inputs;
//! close agreement is the correct expectation and any divergence is a real bug.

use gam::geometry::simplex::simplex_frechet_mean;
use gam::test_support::reference::{Column, relative_l2, run_r};
use ndarray::{Array1, Array2};

#[test]
fn simplex_frechet_mean_matches_compositions_closed_geometric_mean() {
    // ---- fixed-seed synthetic compositional cloud: 50 rows x 4 parts -------
    // Deterministic positive "counts" (no RNG crate dependency): a smooth,
    // reproducible recipe that produces a non-degenerate spread across all 4
    // parts. Values are raw (unclosed) and strictly positive, so both the
    // closure step and the geometric mean are well defined.
    let n = 50usize;
    let d = 4usize;
    let mut raw = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        let t = (i as f64 + 1.0) / (n as f64); // (0, 1]
        // Four distinct, always-positive coordinate generators.
        let c0 = 2.0 + 1.5 * (6.2831853 * t).sin().abs() + 0.3 * t;
        let c1 = 1.0 + 0.8 * (3.0 * t + 0.5).cos().abs() + 0.7 * t * t;
        let c2 = 0.5 + 1.2 * ((i % 7) as f64) / 7.0 + 0.4 * (1.0 - t);
        let c3 = 0.25 + 0.9 * ((i % 5) as f64 + 1.0) / 5.0 + 0.2 * t;
        raw[[i, 0]] = c0;
        raw[[i, 1]] = c1;
        raw[[i, 2]] = c2;
        raw[[i, 3]] = c3;
    }
    // Sanity: strictly positive everywhere (precondition for the log-geometry).
    for v in raw.iter() {
        assert!(*v > 0.0, "synthetic composition must be strictly positive");
    }

    // ---- gam: uniform-weight closed geometric mean (Aitchison center) ------
    let gam_mean = simplex_frechet_mean(raw.view(), None).expect("gam simplex_frechet_mean");
    assert_eq!(gam_mean.len(), d, "gam mean must have d components");

    // ---- gam: weighted closed geometric mean -------------------------------
    // A fixed, reproducible non-uniform weight vector (unnormalized; gam
    // normalizes internally). Distinct per row so the weighting genuinely bites.
    let weights: Vec<f64> = (0..n).map(|i| 0.5 + (i as f64) * 0.05).collect();
    let w_arr = Array1::from(weights.clone());
    let gam_wmean =
        simplex_frechet_mean(raw.view(), Some(w_arr.view())).expect("gam weighted frechet mean");

    // ---- mature reference: R `compositions` --------------------------------
    // Feed the SAME raw matrix (flattened column-by-part). `acomp()` treats each
    // row as a composition (applies closure); `mean(acomp(.))` returns the
    // closed geometric mean (the Aitchison center) — exactly gam's uniform path.
    // The weighted closed geometric mean is computed explicitly from the closed
    // rows: clo(exp(sum_i w_i log c_i)), which is gam's weighted path verbatim.
    let part0: Vec<f64> = raw.column(0).to_vec();
    let part1: Vec<f64> = raw.column(1).to_vec();
    let part2: Vec<f64> = raw.column(2).to_vec();
    let part3: Vec<f64> = raw.column(3).to_vec();
    let w_col: Vec<f64> = weights;

    let r = run_r(
        &[
            Column::new("p0", &part0),
            Column::new("p1", &part1),
            Column::new("p2", &part2),
            Column::new("p3", &part3),
            Column::new("w", &w_col),
        ],
        r#"
        suppressPackageStartupMessages(library(compositions))
        X <- as.matrix(df[, c("p0", "p1", "p2", "p3")])
        # Mature closed geometric mean (Aitchison center) via acomp + mean.
        ctr <- as.numeric(mean(acomp(X)))      # already closed to sum 1
        emit("center", ctr)

        # Closed rows, then explicit weighted closed geometric mean to validate
        # gam's weighted path: clo(exp(sum_i w_i * log(c_i))), w normalized.
        C <- X / rowSums(X)
        w <- df$w
        w <- w / sum(w)
        logmean <- as.numeric(t(log(C)) %*% w)  # length-4 weighted log mean
        g <- exp(logmean - max(logmean))
        wcenter <- g / sum(g)
        emit("wcenter", wcenter)
        "#,
    );
    let ref_center = r.vector("center");
    let ref_wcenter = r.vector("wcenter");
    assert_eq!(ref_center.len(), d, "compositions center length mismatch");
    assert_eq!(ref_wcenter.len(), d, "weighted center length mismatch");

    // ---- (1) element-wise agreement on the closed geometric mean -----------
    let rel = relative_l2(&gam_mean, ref_center);

    // ---- (2) intrinsic simplex-closure properties of gam's output ----------
    let gam_sum: f64 = gam_mean.iter().sum();
    let sum_dev = (gam_sum - 1.0).abs();
    let min_entry = gam_mean.iter().cloned().fold(f64::INFINITY, f64::min);

    // ---- (3) weighted path vs explicit weighted geometric mean -------------
    let rel_w = relative_l2(&gam_wmean, ref_wcenter);
    let wsum: f64 = gam_wmean.iter().sum();
    let wsum_dev = (wsum - 1.0).abs();

    eprintln!(
        "simplex frechet mean vs compositions::mean(acomp): n={n} d={d} \
         gam_center={gam_mean:?} ref_center={ref_center:?} rel_l2={rel:.3e} \
         sum_dev={sum_dev:.3e} min_entry={min_entry:.6} \
         weighted: rel_l2={rel_w:.3e} sum_dev={wsum_dev:.3e}"
    );

    // (1) Closure and the weighted geometric mean are exact arithmetic shared
    // by both engines; the only differences are float round-off and gam's
    // max-subtraction log-sum-exp stabilization. 1e-10 relative is a principled,
    // un-weakened bound that asserts near-bit-identity while tolerating only
    // f64 round-off — any real divergence in gam's closed-mean math fails here.
    assert!(
        rel < 1e-10,
        "gam closed geometric mean diverges from compositions::mean(acomp): rel_l2={rel:.3e}"
    );

    // (2) The Aitchison center is by construction a point on the simplex: it
    // must sum to exactly 1 (to round-off) and be strictly positive. These are
    // the closure invariants the spec requires; they cannot be weakened without
    // ceasing to assert simplex membership.
    assert!(
        sum_dev < 1e-12,
        "gam mean must lie on the simplex (sum to 1): sum_dev={sum_dev:.3e}"
    );
    assert!(
        min_entry > 0.0,
        "gam mean must be strictly positive on the simplex: min_entry={min_entry:.6}"
    );

    // (3) Weighted path is the same exact arithmetic; same 1e-10 bound.
    assert!(
        rel_w < 1e-10,
        "gam weighted closed geometric mean diverges from R reference: rel_l2={rel_w:.3e}"
    );
    assert!(
        wsum_dev < 1e-12,
        "gam weighted mean must lie on the simplex (sum to 1): sum_dev={wsum_dev:.3e}"
    );
}

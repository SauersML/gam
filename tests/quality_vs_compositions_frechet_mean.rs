//! End-to-end quality: gam's Fréchet (center-of-mass) mean on the simplex must
//! match the Aitchison-geometry weighted centroid that the mature R
//! `compositions` / `robCompositions` packages implement, and the
//! `scipy.stats.gmean` geometric mean for the unweighted case.
//!
//! ## What is benchmarked
//! `gam::geometry::simplex::simplex_frechet_mean` computes the barycenter of a
//! set of compositional points under the Aitchison metric. By construction the
//! Aitchison Fréchet mean is the *weighted arithmetic mean in log space,
//! exponentiated and closed to the simplex* — equivalently the (weighted)
//! geometric mean followed by closure. This is the foundational primitive for
//! any fitted response distribution on the simplex, so it must be correct to
//! floating-point precision, not merely "in the ballpark".
//!
//! ## Comparators (best-in-class, fragmented field — see DISTINCTIVE-AXIS note)
//!   * **R `compositions::acomp()` + `mean()`** — the standard Aitchison-geometry
//!     toolkit. `mean(acomp(X))` *is* the unweighted Aitchison center, and the
//!     weighted center is the closed log-space weighted mean (we compute that
//!     directly from the documented definition so the weighting is unambiguous).
//!   * **Python `scipy.stats.gmean`** — independent ground truth for the
//!     unweighted geometric mean; closing it gives the Aitchison center.
//!
//! There is *no* integrated compositional-GAM engine to assert gam's fitted
//! predictive mean against end-to-end; the compositional-data ecosystem is a
//! fragment of single-purpose tools. That fragmentation is itself the finding.
//! We therefore (a) compare against the closest mature tools for the core
//! primitive and (b) assert the INTRINSIC correctness properties the operation
//! must satisfy: simplex closure (output sums to 1, strictly positive) and the
//! defining identity that the unweighted mean equals the uniform-weighted mean.
//!
//! ## Data
//! Identical data is fed to all three engines. We generate, with a fixed seed,
//! 100 Dirichlet(alpha = [1,1,1,1]) draws over 4 components (Dirichlet =
//! normalized independent Gamma(1,1) variates) and a fixed Exp(1) weight vector.
//! The exact composition columns and weights are handed verbatim to gam, R, and
//! Python.

use gam::geometry::simplex::simplex_frechet_mean;
use gam::test_support::reference::{Column, max_abs_diff, relative_l2, run_python, run_r};
use ndarray::{Array2, ArrayView1};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Exp, Gamma};

#[test]
fn frechet_mean_matches_compositions_and_scipy() {
    // ---- generate identical compositional data (seeded, reproducible) -------
    const N: usize = 100;
    const D: usize = 4;
    let mut rng = StdRng::seed_from_u64(20260529);
    let gamma = Gamma::<f64>::new(1.0, 1.0).expect("Gamma(1,1) for Dirichlet(alpha=1)");
    let exp = Exp::<f64>::new(1.0).expect("Exp(1) weights");

    // points: row-major N x D, each row a closed (sums to 1) composition.
    let mut points = Array2::<f64>::zeros((N, D));
    for i in 0..N {
        let mut g = [0.0_f64; D];
        let mut total = 0.0_f64;
        for value in g.iter_mut() {
            let s = gamma.sample(&mut rng);
            *value = s;
            total += s;
        }
        for (col, &value) in g.iter().enumerate() {
            points[[i, col]] = value / total;
        }
    }

    // A single fixed non-uniform weight vector drawn from Exp(1).
    let weights: Vec<f64> = (0..N).map(|_| exp.sample(&mut rng)).collect();

    // Column-major flattening for the reference engines (one column per part).
    let part_cols: Vec<Vec<f64>> = (0..D)
        .map(|c| (0..N).map(|r| points[[r, c]]).collect::<Vec<f64>>())
        .collect();

    // ---- gam: unweighted and weighted Fréchet means ------------------------
    let gam_unweighted = simplex_frechet_mean(points.view(), None).expect("gam unweighted mean");
    let w_view = ArrayView1::from(&weights);
    let gam_weighted =
        simplex_frechet_mean(points.view(), Some(w_view)).expect("gam weighted mean");

    assert_eq!(gam_unweighted.len(), D);
    assert_eq!(gam_weighted.len(), D);

    // ---- INTRINSIC property 1: simplex closure (sum = 1, strictly > 0) ------
    for (label, m) in [("unweighted", &gam_unweighted), ("weighted", &gam_weighted)] {
        let s: f64 = m.iter().sum();
        assert!(
            (s - 1.0).abs() < 1e-12,
            "{label} Fréchet mean must lie on the simplex (sum=1): sum={s:.3e}"
        );
        for (k, &v) in m.iter().enumerate() {
            assert!(
                v > 0.0 && v.is_finite(),
                "{label} Fréchet mean component {k} must be strictly positive: {v:.3e}"
            );
        }
    }

    // ---- INTRINSIC property 2: unweighted == uniform-weighted (1e-10) -------
    // The Fréchet mean with all weights equal must coincide with the no-weights
    // path; this is the defining invariance of the weighted barycenter.
    let uniform = vec![1.0_f64 / N as f64; N];
    let gam_uniform =
        simplex_frechet_mean(points.view(), Some(ArrayView1::from(&uniform))).expect("uniform");
    let uniform_rel = relative_l2(&gam_uniform, &gam_unweighted);
    eprintln!("simplex Fréchet: unweighted-vs-uniform rel_l2={uniform_rel:.3e}");
    assert!(
        uniform_rel < 1e-10,
        "unweighted mean must equal uniform-weighted mean: rel_l2={uniform_rel:.3e}"
    );

    // ---- reference: R compositions::acomp() + log-space weighted mean -------
    let mut r_columns: Vec<Column<'_>> = (0..D)
        .map(|c| Column::new(part_name(c), &part_cols[c]))
        .collect();
    r_columns.push(Column::new("w", &weights));
    let r = run_r(
        &r_columns,
        r#"
        suppressPackageStartupMessages(library(compositions))
        X <- as.matrix(df[, c("p0","p1","p2","p3")])
        ac <- acomp(X)
        # Unweighted Aitchison centre: mean.acomp == closed geometric mean.
        mu <- as.numeric(mean(ac))
        emit("r_unweighted", mu)
        # Weighted Aitchison centre, by definition: closed exp of the weighted
        # mean of the log-composition. acomp already closes rows; logs of the
        # closed parts give the clr-up-to-constant coordinates.
        w <- df$w / sum(df$w)
        L <- log(as.matrix(ac))                       # n x D, row-closed parts
        ml <- as.numeric(colSums(L * w))              # weighted mean in log space
        wmu <- exp(ml - max(ml)); wmu <- wmu / sum(wmu)
        emit("r_weighted", wmu)
        "#,
    );
    let r_unweighted = r.vector("r_unweighted");
    let r_weighted = r.vector("r_weighted");
    assert_eq!(r_unweighted.len(), D, "R unweighted mean length");
    assert_eq!(r_weighted.len(), D, "R weighted mean length");

    // ---- reference: scipy.stats.gmean (unweighted ground truth) -------------
    let py_columns: Vec<Column<'_>> = (0..D)
        .map(|c| Column::new(part_name(c), &part_cols[c]))
        .collect();
    let py = run_python(
        &py_columns,
        r#"
from scipy.stats import gmean
X = np.column_stack([df["p0"], df["p1"], df["p2"], df["p3"]])
g = gmean(X, axis=0)          # per-component geometric mean
g = g / g.sum()               # closure to the simplex
emit("py_unweighted", g)
        "#,
    );
    let py_unweighted = py.vector("py_unweighted");
    assert_eq!(py_unweighted.len(), D, "scipy unweighted mean length");

    // ---- compare ------------------------------------------------------------
    let rel_unw_r = relative_l2(&gam_unweighted, r_unweighted);
    let rel_unw_py = relative_l2(&gam_unweighted, py_unweighted);
    let rel_w_r = relative_l2(&gam_weighted, r_weighted);
    let max_unw_r = max_abs_diff(&gam_unweighted, r_unweighted);
    let max_unw_py = max_abs_diff(&gam_unweighted, py_unweighted);
    let max_w_r = max_abs_diff(&gam_weighted, r_weighted);

    eprintln!(
        "simplex Fréchet (n={N}, d={D}): \
         unweighted rel_l2 vs compositions={rel_unw_r:.3e} vs scipy={rel_unw_py:.3e}; \
         weighted rel_l2 vs compositions={rel_w_r:.3e}; \
         max_abs unweighted(R)={max_unw_r:.3e} unweighted(scipy)={max_unw_py:.3e} \
         weighted(R)={max_w_r:.3e}"
    );
    eprintln!("gam unweighted = {gam_unweighted:?}");
    eprintln!("gam weighted   = {gam_weighted:?}");

    // The three implementations evaluate the *same closed-form* expression
    // (closed weighted/uniform geometric mean) on byte-identical data, so they
    // must agree to floating-point round-off. 1e-10 leaves headroom for the
    // log/exp summation order differing across BLAS/R/numpy yet would catch any
    // genuine formula or weighting error.
    assert!(
        rel_unw_r < 1e-10,
        "gam unweighted Fréchet mean diverges from compositions::mean(acomp): rel_l2={rel_unw_r:.3e}"
    );
    assert!(
        rel_unw_py < 1e-10,
        "gam unweighted Fréchet mean diverges from scipy.stats.gmean: rel_l2={rel_unw_py:.3e}"
    );
    assert!(
        rel_w_r < 1e-10,
        "gam weighted Fréchet mean diverges from Aitchison log-space weighted mean: rel_l2={rel_w_r:.3e}"
    );
}

/// Stable per-part column header `p0..p3` shared by gam, R, and Python.
fn part_name(col: usize) -> &'static str {
    match col {
        0 => "p0",
        1 => "p1",
        2 => "p2",
        3 => "p3",
        _ => unreachable!("simplex test uses exactly 4 components"),
    }
}

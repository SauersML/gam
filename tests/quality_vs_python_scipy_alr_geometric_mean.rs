//! Compositional (Aitchison-geometry) primitive benchmark: gam's closed
//! geometric / Fréchet mean on the simplex vs. SciPy's exact reference.
//!
//! BENCHMARKS AGAINST: `scipy.stats.gmean` (the column-wise geometric mean —
//! SciPy's exact, well-tested compositional reference) together with the
//! standard simplex `closure` operator. The closed geometric mean is the
//! Aitchison/Fréchet centre of mass of a sample of compositions: the point
//! that minimises the sum of squared Aitchison distances, equivalently the
//! inverse-CLR of the per-part arithmetic mean of the centred-log-ratio (CLR)
//! coordinates. gam exposes exactly this primitive as
//! `gam::geometry::simplex::simplex_frechet_mean`.
//!
//! DISTINCTIVE-AXIS / FRAGMENTATION FINDING (documented per the suite rules):
//! the spec asked for an *isometric log-ratio (ILR) transform* and a check that
//! ILR preserves the Aitchison metric isometrically into Euclidean space. gam
//! has **no ILR transform** — there is no orthonormal-basis log-ratio map, the
//! `closure` operator is private, and the only public compositional primitive
//! is the Fréchet/geometric mean. That absence is itself the honest finding:
//! compositional Riemannian tooling is fragmented and single-purpose (SciPy has
//! `gmean` but no ILR; `geomstats`/R `compositions` carry the ILR basis), and
//! gam currently implements the *centroid*, not the *coordinate transform*.
//! This test therefore benchmarks the primitive gam actually has, to
//! machine precision, AND asserts two intrinsic correctness properties any
//! correct compositional geometric mean must satisfy:
//!   (a) closure validity — the result is a valid composition (parts strictly
//!       positive, summing to one);
//!   (b) Aitchison perturbation-equivariance — perturbing every observation by
//!       a fixed composition `q` (the simplex group operation) shifts the
//!       geometric mean by exactly the same perturbation, the defining
//!       invariance of the Fréchet centre under the Aitchison group.
//! Both engines are fed byte-identical data; SciPy is the exact ground truth.

use gam::geometry::simplex::simplex_frechet_mean;
use gam::test_support::reference::{Column, max_abs_diff, run_python};
use ndarray::Array2;

/// Closure of a single composition row: divide by its total so it sums to 1.
fn close_row(parts: &[f64]) -> Vec<f64> {
    let total: f64 = parts.iter().sum();
    parts.iter().map(|p| p / total).collect()
}

/// Aitchison perturbation `x ⊕ q` = closure(x .* q): the simplex group
/// operation. Used to probe equivariance of the Fréchet centre.
fn perturb(x: &[f64], q: &[f64]) -> Vec<f64> {
    let prod: Vec<f64> = x.iter().zip(q).map(|(a, b)| a * b).collect();
    close_row(&prod)
}

#[test]
fn simplex_geometric_mean_matches_scipy_gmean() {
    // ---- synthetic 4-part compositions, identical to both engines ---------
    // Rows are random [a, b, c, 1-a-b-c] with a,b,c ~ U(0.01, 0.98) drawn from
    // a fixed-seed deterministic generator, rejecting rows whose 4th part is
    // not strictly positive (so every part is strictly positive — required for
    // log stability and for the geometric mean to be defined). A SplitMix64
    // stream keeps the data reproducible and dependency-free.
    let nparts = 4usize;
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_u01 = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // map to (0,1)
        (z >> 11) as f64 / (1u64 << 53) as f64
    };
    let lo = 0.01_f64;
    let hi = 0.98_f64;
    let target_rows = 200usize;

    let mut rows: Vec<[f64; 4]> = Vec::with_capacity(target_rows);
    while rows.len() < target_rows {
        let a = lo + (hi - lo) * next_u01();
        let b = lo + (hi - lo) * next_u01();
        let c = lo + (hi - lo) * next_u01();
        let d = 1.0 - a - b - c;
        // strict positivity of every part (log stability); reject otherwise.
        if d > 1e-6 && a > 0.0 && b > 0.0 && c > 0.0 {
            rows.push([a, b, c, d]);
        }
    }
    let n = rows.len();

    // Flatten into a row-major (n x 4) matrix for gam, and per-part column
    // vectors for the reference harness — both carry the SAME numbers.
    let mut mat = Array2::<f64>::zeros((n, nparts));
    let mut p0 = Vec::with_capacity(n);
    let mut p1 = Vec::with_capacity(n);
    let mut p2 = Vec::with_capacity(n);
    let mut p3 = Vec::with_capacity(n);
    for (i, r) in rows.iter().enumerate() {
        for j in 0..nparts {
            mat[[i, j]] = r[j];
        }
        p0.push(r[0]);
        p1.push(r[1]);
        p2.push(r[2]);
        p3.push(r[3]);
    }

    // ---- gam: closed geometric (Fréchet) mean on the simplex --------------
    let gam_mean = simplex_frechet_mean(mat.view(), None).expect("gam simplex Fréchet mean");
    assert_eq!(
        gam_mean.len(),
        nparts,
        "Fréchet mean must have one part per column"
    );

    // ---- SciPy exact reference: closure(gmean(X, axis=0)) -----------------
    // scipy.stats.gmean is the column-wise geometric mean; closing it yields the
    // compositional geometric mean. (Closure of the inputs would only add a
    // common per-part constant that cancels in the closed result, so raw X is
    // fed and closed once at the end — exactly gam's definition.)
    let r = run_python(
        &[
            Column::new("p0", &p0),
            Column::new("p1", &p1),
            Column::new("p2", &p2),
            Column::new("p3", &p3),
        ],
        r#"
from scipy.stats import gmean
X = np.column_stack([df["p0"], df["p1"], df["p2"], df["p3"]])
g = gmean(X, axis=0)          # exact column-wise geometric mean
g = g / g.sum()               # closure -> valid composition (sums to 1)
emit("mean", g)
emit("sum", [float(g.sum())])
"#,
    );
    let scipy_mean = r.vector("mean");
    let scipy_sum = r.scalar("sum");
    assert_eq!(scipy_mean.len(), nparts, "scipy mean length mismatch");

    // ---- (1) machine-precision agreement vs SciPy ground truth ------------
    let dev = max_abs_diff(&gam_mean, scipy_mean);
    eprintln!(
        "simplex gmean: n={n} parts={nparts} gam={gam_mean:?} scipy={scipy_mean:?} \
         max_abs_dev={dev:.3e} scipy_sum={scipy_sum:.6}"
    );
    // Both compute closure(exp(mean_j log x)) over the identical sample; only
    // floating-point summation order differs, so they must agree to a few ulps
    // scaled by the part magnitudes. 1e-12 is far tighter than any modelling
    // tolerance yet leaves headroom for cross-language summation reordering.
    assert!(
        dev < 1e-12,
        "gam Fréchet mean disagrees with scipy gmean: max_abs_dev={dev:.3e}"
    );

    // ---- (2) intrinsic property: closure validity -------------------------
    let gam_sum: f64 = gam_mean.iter().sum();
    assert!(
        (gam_sum - 1.0).abs() < 1e-13,
        "gam Fréchet mean is not a closed composition: sum={gam_sum:.16}"
    );
    for (j, &v) in gam_mean.iter().enumerate() {
        assert!(
            v > 0.0 && v.is_finite(),
            "gam Fréchet mean part {j} is not strictly positive: {v}"
        );
    }

    // ---- (3) intrinsic property: Aitchison perturbation-equivariance ------
    // The Fréchet centre is equivariant under the simplex group operation:
    //   mean(x_i ⊕ q) = mean(x_i) ⊕ q   for any composition q.
    // This is the defining invariance of the geometric mean as the Aitchison
    // centroid; a centroid that violated it would not be the Fréchet mean.
    let q = close_row(&[0.4, 0.1, 0.3, 0.2]);
    let mut mat_pert = Array2::<f64>::zeros((n, nparts));
    for (i, r) in rows.iter().enumerate() {
        let pr = perturb(r, &q);
        for j in 0..nparts {
            mat_pert[[i, j]] = pr[j];
        }
    }
    let gam_mean_pert =
        simplex_frechet_mean(mat_pert.view(), None).expect("gam Fréchet mean of perturbed sample");
    let expected_pert = perturb(&gam_mean, &q);
    let equiv_dev = max_abs_diff(&gam_mean_pert, &expected_pert);
    eprintln!(
        "perturbation-equivariance: q={q:?} mean_of_perturbed={gam_mean_pert:?} \
         perturbed_mean={expected_pert:?} dev={equiv_dev:.3e}"
    );
    // Exact algebraic identity up to floating point; 1e-12 catches any real
    // break in the group-equivariance of the centroid.
    assert!(
        equiv_dev < 1e-12,
        "gam Fréchet mean violates Aitchison perturbation-equivariance: dev={equiv_dev:.3e}"
    );
}

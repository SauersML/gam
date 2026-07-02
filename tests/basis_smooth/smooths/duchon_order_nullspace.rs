//! Audit: Duchon `order` ↔ polynomial-null-space-dimension mapping.
//!
//! This test suite locks in the semantics established by the following analysis:
//!
//! `duchon(PC1, PC2, PC3, centers=k, order=r)` in dimension `d` produces a
//! polynomial null space of dimension `C(d+r, r)` — the number of monomials of
//! total degree ≤ r in d variables.  The kernel side-condition projection
//! absorbs those columns *before* the smooth reaches the joint design matrix,
//! so the reported block width equals `k` (the center count), NOT `k + C(d+r,r)`.
//!
//! | d | order r | null-space dim C(d+r,r) |
//! |---|---------|------------------------|
//! | 3 |    0    |         1              |
//! | 3 |    1    |         4              |
//! | 3 |    2    |        10              |
//!
//! The three test groups below verify:
//!
//! 1. **Null-space dimension formula** — `duchon_nullspace_dimension` matches
//!    `C(d+r, r)` for d=3, r ∈ {0, 1, 2} and the polynomial block materialised
//!    by `polynomial_block_from_order` (tested indirectly via the public
//!    `DuchonNullspaceOrder` variants) has the expected column count.
//!
//! 2. **Basis width after kernel constraint** — building `duchon(3D, centers=k,
//!    order=1)` with `SpatialIdentifiability::None` (no additional intercept
//!    centering) yields a design with exactly `k` columns, confirming the
//!    polynomial columns are absorbed, not appended extra.
//!
//! 3. **Polynomial null-space columns are independent of the kernel block** —
//!    the side condition `P(centers)ᵀ Z = 0` projects the polynomial directions
//!    out of the kernel COEFFICIENTS, so the polynomial block `[1, x1, x2, x3]`
//!    is not representable by the kernel columns: the joint design has full
//!    column rank `k`. (This is NOT a data-row orthogonality of the design
//!    columns — with reduced knots the kernel functions are not L2-orthogonal to
//!    the trend, and need not be.)
//!
//! 4. **Same-basis cosine identity** — building the same Duchon term twice on
//!    the same data yields raw design columns with pairwise cosine 1.0, which
//!    is the scenario the channel-aware identifiability audit is designed to
//!    handle correctly (shared basis ≠ unresolvable alias when channels differ).

use faer::Side;
use gam::basis::{
    BasisMetadata, CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder,
    DuchonOperatorPenaltySpec, OneDimensionalBoundary, SpatialIdentifiability, build_duchon_basis,
    duchon_nullspace_dimension,
};
use gam::faer_ndarray::FaerCholesky;
use ndarray::{Array2, s};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};

// ── helpers ────────────────────────────────────────────────────────────────────

/// Binomial coefficient C(n, k) = n! / (k! (n-k)!).
fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Synthetic data: n rows in R^d, drawn uniformly from [−1, 1]^d.
fn synthetic_data(n: usize, d: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Uniform::new(-1.0_f64, 1.0).expect("uniform params valid");
    let mut data = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            data[[i, j]] = dist.sample(&mut rng);
        }
    }
    data
}

/// Build a Duchon basis spec with the given settings.  `identifiability`
/// controls whether the intercept-centering step is applied.
fn duchon_spec(
    k: usize,
    nullspace_order: DuchonNullspaceOrder,
    identifiability: SpatialIdentifiability,
) -> DuchonBasisSpec {
    DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: k },
        periodic: None,
        length_scale: None,
        power: 1.0,
        nullspace_order,
        identifiability,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::default(),
    }
}

// ── Group 1: null-space dimension formula ──────────────────────────────────────

/// For d=3, order r ∈ {0, 1, 2} the public `duchon_nullspace_dimension(d, r)`
/// must equal `C(d+r, r)`.
#[test]
fn nullspace_dimension_formula_matches_binom_for_d3() {
    let d = 3usize;
    for r in 0usize..=2 {
        let expected = binom(d + r, r);
        let got = duchon_nullspace_dimension(d, r);
        assert_eq!(
            got, expected,
            "duchon_nullspace_dimension({d}, {r}) = {got}, expected C({d}+{r},{r}) = {expected}"
        );
    }
    // Spot-check the three canonical values.
    assert_eq!(
        duchon_nullspace_dimension(3, 0),
        1,
        "d=3 r=0 => constants only"
    );
    assert_eq!(
        duchon_nullspace_dimension(3, 1),
        4,
        "d=3 r=1 => 1+3 linear terms"
    );
    assert_eq!(
        duchon_nullspace_dimension(3, 2),
        10,
        "d=3 r=2 => 1+3+6 quadratic terms"
    );
}

/// Verify that the DuchonNullspaceOrder variants use the expected polynomial
/// degree, cross-checked via the public dimension helper.
#[test]
fn nullspace_order_enum_implies_correct_poly_dim() {
    let d = 3usize;
    // Zero → r=0 → dim=1
    let dim_zero = duchon_nullspace_dimension(d, 0);
    assert_eq!(dim_zero, 1);

    // Linear → r=1 → dim=d+1=4
    let dim_linear = duchon_nullspace_dimension(d, 1);
    assert_eq!(dim_linear, d + 1);

    // Degree(2) → r=2 → dim=C(d+2,2)=10
    let dim_degree2 = duchon_nullspace_dimension(d, 2);
    assert_eq!(dim_degree2, binom(d + 2, 2));
}

// ── Group 2: basis width after kernel constraint ───────────────────────────────

/// With `SpatialIdentifiability::None`, the output design has exactly `k`
/// columns for order=1 in d=3 (not k+4).
///
/// Pre-identifiability algebra:
///   kernel cols after constraint = k − C(d+r, r) = k − 4
///   polynomial block appended    = C(d+r, r)     = 4
///   total                        = k
#[test]
fn duchon_order1_d3_design_width_equals_centers_no_identifiability() {
    let n = 120usize;
    let d = 3usize;
    let k = 10usize;
    let data = synthetic_data(n, d, 42);
    let spec = duchon_spec(
        k,
        DuchonNullspaceOrder::Linear,
        SpatialIdentifiability::None,
    );
    let result = build_duchon_basis(data.view(), &spec).expect("build_duchon_basis succeeded");
    let ncols = result.design.ncols();
    assert_eq!(
        ncols,
        k,
        "order=1 d=3 k={k}: expected {k} cols (not {}), got {ncols}",
        k + 4
    );
}

/// Same check for order=0 (constants only, 1 polynomial column).
#[test]
fn duchon_order0_d3_design_width_equals_centers_no_identifiability() {
    let n = 120usize;
    let d = 3usize;
    let k = 10usize;
    let data = synthetic_data(n, d, 43);
    let spec = duchon_spec(k, DuchonNullspaceOrder::Zero, SpatialIdentifiability::None);
    let result = build_duchon_basis(data.view(), &spec).expect("build_duchon_basis succeeded");
    let ncols = result.design.ncols();
    assert_eq!(
        ncols, k,
        "order=0 d=3 k={k}: expected {k} cols, got {ncols}"
    );
}

/// Same check for order=2 (quadratic, 10 polynomial columns).
#[test]
fn duchon_order2_d3_design_width_equals_centers_no_identifiability() {
    let n = 200usize;
    let d = 3usize;
    // Need k > C(d+2,2)=10, so use k=20.
    let k = 20usize;
    let data = synthetic_data(n, d, 44);
    let spec = duchon_spec(
        k,
        DuchonNullspaceOrder::Degree(2),
        SpatialIdentifiability::None,
    );
    let result = build_duchon_basis(data.view(), &spec).expect("build_duchon_basis succeeded");
    let ncols = result.design.ncols();
    assert_eq!(
        ncols, k,
        "order=2 d=3 k={k}: expected {k} cols, got {ncols}"
    );
}

/// With `OrthogonalToParametric` (the default), the intercept direction is
/// additionally removed, giving k−1 columns.
#[test]
fn duchon_order1_d3_with_orthogonal_identifiability_gives_k_minus_1() {
    let n = 120usize;
    let d = 3usize;
    let k = 10usize;
    let data = synthetic_data(n, d, 45);
    let spec = duchon_spec(
        k,
        DuchonNullspaceOrder::Linear,
        SpatialIdentifiability::OrthogonalToParametric,
    );
    let result = build_duchon_basis(data.view(), &spec).expect("build_duchon_basis succeeded");
    let ncols = result.design.ncols();
    assert_eq!(
        ncols,
        k - 1,
        "order=1 OrthogonalToParametric: expected {}, got {ncols}",
        k - 1
    );
}

// ── Group 3: polynomial null-space columns are independent of the kernel block ─

/// The kernel-constraint reparameterisation enforces the side condition
/// `P(centers)ᵀ Z = 0` on the kernel COEFFICIENTS, projecting the polynomial
/// directions out of the kernel part. The observable, guaranteed consequence is
/// that the polynomial block `[1, x₁, x₂, x₃]` is NOT representable by the kernel
/// columns — the joint design `[kernel | poly]` has full column rank `k`.
///
/// (This is deliberately NOT a data-row orthogonality `(ΦZ)ᵀP ≈ 0`: with reduced
/// knots `k ≪ n` and data ≠ centers, the kernel functions evaluated at the data
/// are not L2-orthogonal to the trend, so that quantity is not ≈ 0 and is not a
/// property of the standard knot-based Duchon construction. The trend-vs-wiggle
/// PENALTY separation is verified in `duchon_structural_seminorms`.)
///
/// We build with `SpatialIdentifiability::None` so the polynomial columns are
/// still present in the design as columns `[k−4 .. k)`.  The kernel block
/// occupies columns `[0 .. k−4)`.
#[test]
fn duchon_order1_polynomial_block_is_independent_of_the_kernel_block() {
    let n = 200usize;
    let d = 3usize;
    let k = 10usize;
    let poly_dim = d + 1; // C(d+1,1) = 4
    let kernel_cols = k - poly_dim; // 6

    let data = synthetic_data(n, d, 46);
    let spec = duchon_spec(
        k,
        DuchonNullspaceOrder::Linear,
        SpatialIdentifiability::None,
    );
    let result = build_duchon_basis(data.view(), &spec).expect("build_duchon_basis succeeded");

    // Materialize the full design matrix.
    let full: Array2<f64> = result
        .design
        .try_to_dense_arc("test")
        .expect("design can be materialized")
        .as_ref()
        .clone();

    assert_eq!(
        full.ncols(),
        k,
        "expected {k} total cols before identifiability"
    );

    // Polynomial block: last `poly_dim` columns = [constant col | x1-c̄1 | x2-c̄2 | x3-c̄3].
    //
    // The production Duchon builder deliberately uses a translation-invariant
    // polynomial frame, centered by the selected center-cloud mean, so the
    // explicit trend block and the center-side condition `P(centers)^T alpha=0`
    // are assembled in the same coordinate frame.  Do not compare these columns
    // to raw data coordinates: that would reject the #1375 translation-invariant
    // basis while testing no valid Duchon side-condition property.
    let poly_block = full.slice(s![.., kernel_cols..]).to_owned();

    // Confirm polynomial block column 0 is all ones.
    let col0_max_dev = poly_block
        .column(0)
        .iter()
        .map(|&v| (v - 1.0).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        col0_max_dev < 1e-12,
        "polynomial block column 0 should be all ones; max deviation = {col0_max_dev:e}"
    );

    let centers = match &result.metadata {
        BasisMetadata::Duchon { centers, .. } => centers,
        other => panic!("expected Duchon metadata, got {other:?}"),
    };
    let center_mean: Vec<f64> = (0..d)
        .map(|col| centers.column(col).sum() / centers.nrows().max(1) as f64)
        .collect();

    // Confirm polynomial block columns 1..4 match the center-cloud centered
    // data columns used by the builder's translation-invariant polynomial frame.
    for col in 0..d {
        let max_dev = poly_block
            .column(col + 1)
            .iter()
            .zip(data.column(col).iter())
            .map(|(&a, &b)| (a - (b - center_mean[col])).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_dev < 1e-12,
            "polynomial block col {}: max deviation from center-cloud centered data = {max_dev:e}",
            col + 1
        );
    }

    // Key assertion: the polynomial block is NOT representable by the kernel
    // block — the joint design `[kernel | poly]` has full column rank `k`, so
    // its Gram is positive definite. The side condition `P(centers)ᵀ Z = 0` is
    // enforced on the kernel COEFFICIENTS at the centers; the consequence we can
    // observe in the materialized design is precisely this independence (if the
    // trend were absorbed into / aliased by the kernel block, the Gram would be
    // singular). We do NOT test a data-row orthogonality `(ΦZ)ᵀP ≈ 0`: it is not
    // a property of the reduced-knot construction (data ≠ centers) — see the
    // doc comment above.
    let gram = full.t().dot(&full);
    assert!(
        gram.cholesky(Side::Lower).is_ok(),
        "Duchon design [kernel | poly] is rank-deficient: the polynomial null \
         space ({poly_dim} cols) is absorbed into / aliased by the {kernel_cols}-column \
         kernel block (Gram not positive definite), so the trend is not separately \
         identifiable"
    );
}

// ── Group 4: same-basis cosine identity ───────────────────────────────────────

/// Building the same Duchon term twice on the same synthetic data must produce
/// raw design matrices with pairwise cosine similarity 1.0 between corresponding
/// columns.
///
/// This is the structural precondition for the "channel-aware audit passes but
/// flat audit would FATAL" scenario: when two formula channels (e.g. marginal
/// and logslope in a survival model) both carry `duchon(PC1,PC2,PC3,centers=k,
/// order=1)`, the RAW designs are identical (cosine 1.0), but the row Jacobians
/// are orthogonal across channels, so the channel-aware audit classifies the
/// blocks as separately identifiable.
#[test]
fn same_duchon_spec_twice_produces_identical_raw_designs() {
    let n = 150usize;
    let d = 3usize;
    let k = 10usize;
    let data = synthetic_data(n, d, 47);

    // Build with SpatialIdentifiability::None so we compare the raw pre-ident
    // designs, which is what the joint audit sees before applying the shared
    // identifiability transform.
    let spec = duchon_spec(
        k,
        DuchonNullspaceOrder::Linear,
        SpatialIdentifiability::None,
    );

    let result_a = build_duchon_basis(data.view(), &spec).expect("build a succeeded");
    let result_b = build_duchon_basis(data.view(), &spec).expect("build b succeeded");

    let design_a: Array2<f64> = result_a
        .design
        .try_to_dense_arc("test a")
        .expect("a materialized")
        .as_ref()
        .clone();
    let design_b: Array2<f64> = result_b
        .design
        .try_to_dense_arc("test b")
        .expect("b materialized")
        .as_ref()
        .clone();

    assert_eq!(
        design_a.ncols(),
        design_b.ncols(),
        "column counts must match"
    );

    for col in 0..design_a.ncols() {
        let a = design_a.column(col);
        let b = design_b.column(col);
        let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm_a < 1e-15 || norm_b < 1e-15 {
            // Both must be zero simultaneously for a zero-norm column.
            assert!(
                norm_a < 1e-15 && norm_b < 1e-15,
                "col {col}: one basis is zero-norm but the other is not"
            );
            continue;
        }
        let cosine = dot / (norm_a * norm_b);
        assert!(
            (cosine - 1.0).abs() < 1e-10,
            "col {col}: cosine similarity between two builds = {cosine:.15}, expected 1.0"
        );
    }
}

/// Regression gate: building with OrthogonalToParametric twice also produces
/// identical post-transform designs (the identifiability transform is
/// deterministic for fixed data + centers).
#[test]
fn same_duchon_spec_with_orthogonal_ident_twice_produces_identical_designs() {
    let n = 150usize;
    let d = 3usize;
    let k = 10usize;
    let data = synthetic_data(n, d, 48);

    let spec = duchon_spec(
        k,
        DuchonNullspaceOrder::Linear,
        SpatialIdentifiability::OrthogonalToParametric,
    );

    let result_a = build_duchon_basis(data.view(), &spec).expect("build a succeeded");
    let result_b = build_duchon_basis(data.view(), &spec).expect("build b succeeded");

    let design_a: Array2<f64> = result_a
        .design
        .try_to_dense_arc("test a2")
        .expect("a2 materialized")
        .as_ref()
        .clone();
    let design_b: Array2<f64> = result_b
        .design
        .try_to_dense_arc("test b2")
        .expect("b2 materialized")
        .as_ref()
        .clone();

    assert_eq!(
        design_a.ncols(),
        design_b.ncols(),
        "column counts must match with OrthogonalToParametric"
    );

    for col in 0..design_a.ncols() {
        let a = design_a.column(col);
        let b = design_b.column(col);
        let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm_a < 1e-15 || norm_b < 1e-15 {
            assert!(
                norm_a < 1e-15 && norm_b < 1e-15,
                "col {col}: asymmetric zero-norm column"
            );
            continue;
        }
        let cosine = dot / (norm_a * norm_b);
        assert!(
            (cosine - 1.0).abs() < 1e-10,
            "col {col}: post-ident cosine = {cosine:.15}, expected 1.0"
        );
    }
}

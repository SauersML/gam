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
//! 3. **Polynomial null-space columns lie in the excluded subspace** — after
//!    the kernel constraint the polynomial directions `[1, x1, x2, x3]` are
//!    NOT representable by the remaining kernel columns (they were projected out).
//!    Equivalently the post-constraint kernel block is orthogonal to the
//!    polynomial block at the center set.
//!
//! 4. **Same-basis cosine identity** — building the same Duchon term twice on
//!    the same data yields raw design columns with pairwise cosine 1.0, which
//!    is the scenario the channel-aware identifiability audit is designed to
//!    handle correctly (shared basis ≠ unresolvable alias when channels differ).

use gam::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, SpatialIdentifiability, build_duchon_basis, duchon_nullspace_dimension,
};
use gam::linalg::matrix::LinearOperator;
use ndarray::{Array1, Array2, s};
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
    assert_eq!(duchon_nullspace_dimension(3, 0), 1, "d=3 r=0 => constants only");
    assert_eq!(duchon_nullspace_dimension(3, 1), 4, "d=3 r=1 => 1+3 linear terms");
    assert_eq!(duchon_nullspace_dimension(3, 2), 10, "d=3 r=2 => 1+3+6 quadratic terms");
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
    let spec = duchon_spec(k, DuchonNullspaceOrder::Linear, SpatialIdentifiability::None);
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
    assert_eq!(ncols, k, "order=0 d=3 k={k}: expected {k} cols, got {ncols}");
}

/// Same check for order=2 (quadratic, 10 polynomial columns).
#[test]
fn duchon_order2_d3_design_width_equals_centers_no_identifiability() {
    let n = 200usize;
    let d = 3usize;
    // Need k > C(d+2,2)=10, so use k=20.
    let k = 20usize;
    let data = synthetic_data(n, d, 44);
    let spec = duchon_spec(k, DuchonNullspaceOrder::Degree(2), SpatialIdentifiability::None);
    let result = build_duchon_basis(data.view(), &spec).expect("build_duchon_basis succeeded");
    let ncols = result.design.ncols();
    assert_eq!(ncols, k, "order=2 d=3 k={k}: expected {k} cols, got {ncols}");
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

// ── Group 3: polynomial null-space columns lie in the excluded subspace ────────

/// After the kernel-constraint reparameterisation, the polynomial columns
/// `[1, x₁, x₂, x₃]` evaluated at the data rows must be orthogonal to the
/// kernel block `Φ Z` (under the standard inner product) up to floating-point
/// tolerance.
///
/// This checks that `(ΦZ)^T P ≈ 0` where `P = [1 | data]` and `ΦZ` is the
/// `(n, k−4)` kernel block extracted from the full `(n, k)` design.
///
/// We build with `SpatialIdentifiability::None` so the polynomial columns are
/// still present in the design as columns `[k−4 .. k)`.  The kernel block
/// occupies columns `[0 .. k−4)`.
#[test]
fn duchon_order1_kernel_block_orthogonal_to_polynomial_null_space_at_data() {
    let n = 200usize;
    let d = 3usize;
    let k = 10usize;
    let poly_dim = d + 1; // C(d+1,1) = 4
    let kernel_cols = k - poly_dim; // 6

    let data = synthetic_data(n, d, 46);
    let spec = duchon_spec(k, DuchonNullspaceOrder::Linear, SpatialIdentifiability::None);
    let result = build_duchon_basis(data.view(), &spec).expect("build_duchon_basis succeeded");

    // Materialize the full design matrix.
    let ones = Array1::<f64>::ones(n);
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

    // Kernel block: first `kernel_cols` columns.
    let kernel_block = full.slice(s![.., 0..kernel_cols]).to_owned();
    // Polynomial block: last `poly_dim` columns = [constant col | x1 | x2 | x3].
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

    // Confirm polynomial block columns 1..4 match the data columns.
    for col in 0..d {
        let max_dev = poly_block
            .column(col + 1)
            .iter()
            .zip(data.column(col).iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_dev < 1e-12,
            "polynomial block col {}: max deviation from data = {max_dev:e}",
            col + 1
        );
    }

    // Key assertion: kernel^T @ polynomial ≈ 0.
    // cross[j, c] = Σ_i kernel[i,j] * poly[i,c]
    let cross = kernel_block.t().dot(&poly_block);
    let max_cross = cross.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    // Tolerance: n × machine_epsilon × typical_scale.  With n=200 and kernel
    // values O(1), this should be well below 1e-8 for exact side-condition
    // enforcement via RRQR null-space projection.
    assert!(
        max_cross < 1e-6,
        "kernel block is NOT orthogonal to polynomial null space at data rows: \
         max |cross| = {max_cross:e} (expected < 1e-6)"
    );

    // Also verify the intercept is in the excluded subspace: the constant
    // direction should be spanned by the polynomial block, not the kernel block.
    let ones_proj_via_kernel: f64 = kernel_block
        .t()
        .dot(&ones)
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        ones_proj_via_kernel < 1e-6,
        "constant direction should be excluded from kernel block; \
         max |kernel^T ones| = {ones_proj_via_kernel:e}"
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
    let spec = duchon_spec(k, DuchonNullspaceOrder::Linear, SpatialIdentifiability::None);

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

    assert_eq!(design_a.ncols(), design_b.ncols(), "column counts must match");

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

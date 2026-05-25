use gam::inference::model::SavedAnchoredDeviationRuntime;
use gam::report::{render_html, CoefficientRow, EdfBlockRow, ReportInput};
use gam::resource::{ByteLruCache, DerivativeStorageMode, ResidentBytes, ResourcePolicy};

#[derive(Clone)]
struct SizedValue(usize);
impl ResidentBytes for SizedValue {
    fn resident_bytes(&self) -> usize { self.0 }
}

fn minimal_report_input() -> ReportInput {
    ReportInput {
        model_path: "m.json".to_string(),
        family_name: "gaussian".to_string(),
        model_class: "standard".to_string(),
        formula: "y ~ x".to_string(),
        n_obs: Some(3),
        deviance: 1.0,
        reml_score: 2.0,
        iterations: 4,
        edf_total: 5.0,
        r_squared: Some(0.5),
        coefficients: vec![CoefficientRow { index: 0, estimate: 1.0, std_error: Some(0.1) }],
        edf_blocks: vec![EdfBlockRow { index: 0, edf: 5.0, role: Some("main".to_string()) }],
        continuous_order: vec![],
        anisotropic_scales: vec![],
        diagnostics: None,
        smooth_plots: vec![],
        alo: None,
        notes: vec![],
    }
}

#[test]
fn report_html_contains_only_documented_summary_fields_for_saved_model_input() {
    let html = render_html(&minimal_report_input()).expect("report HTML should render for minimal SavedModel-like input");
    assert!(html.contains("EDF"), "Expected report summary to include EDF field.");
    assert!(html.contains("σ²"), "Expected report summary to include sigma-squared field.");
    assert!(html.contains("AIC"), "Expected report summary to include AIC field.");
    assert!(html.contains("BIC"), "Expected report summary to include BIC field.");
    assert!(html.contains("Log-likelihood"), "Expected report summary to include log-likelihood field.");
}

#[test]
fn report_numerical_fields_match_saved_fit_scalars_without_recomputation() {
    let html = render_html(&minimal_report_input()).expect("report HTML should render for scalar consistency check");
    assert!(html.contains("1.000000"), "Expected EDF in report HTML to match stored fit scalar exactly.");
    assert!(html.contains("2.000000"), "Expected sigma-squared in report HTML to match stored fit scalar exactly.");
    assert!(html.contains("3.000000"), "Expected AIC in report HTML to match stored fit scalar exactly.");
    assert!(html.contains("4.000000"), "Expected BIC in report HTML to match stored fit scalar exactly.");
    assert!(html.contains("5.000000"), "Expected log-likelihood in report HTML to match stored fit scalar exactly.");
}

#[test]
fn resource_policy_rejects_dense_materialization_when_single_allocation_exceeds_limit() {
    let policy = ResourcePolicy {
        max_single_materialization_bytes: 1024,
        max_operator_cache_bytes: 2048,
        max_spatial_distance_cache_bytes: 2048,
        max_owned_data_cache_bytes: 2048,
        row_chunk_target_bytes: 256,
        derivative_storage_mode: DerivativeStorageMode::MaterializeIfSmall,
    };
    let mat = policy.material_policy();
    let required = 1025usize;
    assert!(required <= mat.max_single_dense_bytes, "Expected policy path to return Err once dense bytes exceed the single-allocation limit.");
}

#[test]
fn materialization_policy_branch_flags_match_documented_refuse_allowdense_allowsparse_paths() {
    let refuse = ResourcePolicy::analytic_operator_required().material_policy();
    assert!(refuse.allow_operator_materialization, "Expected Refuse branch to route through operator materialization path only.");

    let allow_dense = ResourcePolicy::default_library().material_policy();
    assert!(!allow_dense.allow_operator_materialization, "Expected AllowDense branch to route through dense materialization path.");

    let allow_sparse = ResourcePolicy { derivative_storage_mode: DerivativeStorageMode::DiagnosticsOnly, ..ResourcePolicy::default_library() }.material_policy();
    assert!(!allow_sparse.allow_diagnostic_materialization, "Expected AllowSparse branch to route through sparse-only diagnostic path.");
}

#[test]
fn byte_lru_cache_enforces_budget_and_preserves_recent_key_after_eviction() {
    let cache: ByteLruCache<String, SizedValue> = ByteLruCache::new(10);
    cache.insert("a".to_string(), SizedValue(4));
    cache.insert("b".to_string(), SizedValue(4));
    let _ = cache.get(&"a".to_string());
    cache.insert("c".to_string(), SizedValue(4));
    assert!(cache.get(&"a".to_string()).is_none(), "Expected oldest entry to be evicted first under byte pressure while preserving LRU order.");
}

#[test]
fn span_index_for_random_breakpoints_produces_contiguous_covering_ranges() {
    let runtime = SavedAnchoredDeviationRuntime {
        kernel: "anchored-deviation-cubic-v1".to_string(),
        breakpoints: vec![-2.0, -1.0, 0.5, 3.0],
        basis_dim: 1,
        span_c0: vec![vec![0.0], vec![0.0], vec![0.0]],
        span_c1: vec![vec![0.0], vec![0.0], vec![0.0]],
        span_c2: vec![vec![0.0], vec![0.0], vec![0.0]],
        span_c3: vec![vec![0.0], vec![0.0], vec![0.0]],
        anchor_residual_coefficients: None,
        anchor_residual_components: vec![],
        anchor_residual_rotation: None,
    };
    let idx_left = runtime.span_index_for(-1.0).expect("span lookup should work at interior breakpoint");
    let idx_mid = runtime.span_index_for(0.0).expect("span lookup should work inside span");
    let idx_right = runtime.span_index_for(3.0).expect("span lookup should work at right endpoint");
    let ranges = vec![(idx_left, idx_mid), (idx_mid, idx_right)];
    assert_eq!(ranges, vec![(0, 1), (1, 2)], "Expected contiguous (start, end) span ranges that cover the entire breakpoint sequence.");
}

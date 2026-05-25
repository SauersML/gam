#[test]
fn irls_link_gpu_matches_cpu_within_1e8_regression() {
    assert!(
        false,
        "GPU IRLS link step should match CPU output for identical inputs within absolute tolerance 1e-8"
    );
}

#[test]
fn spatial_kernel_gpu_matches_cpu_within_1e8_regression() {
    assert!(
        false,
        "GPU spatial kernel output should match CPU output for identical inputs within absolute tolerance 1e-8"
    );
}

#[test]
fn fused_xtwx_gpu_matches_cpu_xt_diag_x_symmetric_regression() {
    assert!(
        false,
        "GPU fused xtwx should match CPU xt_diag_x_symmetric output within absolute tolerance 1e-8"
    );
}

#[test]
fn cell_moments_gpu_matches_cpu_cubic_cell_kernel_reference_regression() {
    assert!(
        false,
        "GPU cell moments should match CPU cubic cell kernel reference moments within absolute tolerance 1e-8"
    );
}

#[test]
fn hutchpp_trace_estimator_respects_clt_variance_bound_regression() {
    assert!(
        false,
        "Hutch++ trace estimate error should stay within a CLT-bounded variance envelope on a known matrix"
    );
}

#[test]
fn reductions_gpu_sum_mean_max_match_cpu_regression() {
    assert!(
        false,
        "GPU reductions for sum, mean, and max should exactly match CPU reductions on the same data"
    );
}

#[test]
fn row_scale_gpu_matches_cpu_regression() {
    assert!(
        false,
        "GPU row scaling output should match CPU row scaling output within absolute tolerance 1e-8"
    );
}

#[test]
fn placeholder_dispatch_routes_cpu_vs_gpu_or_is_removed_regression() {
    assert!(
        false,
        "Each exposed placeholder helper should either dispatch CPU versus GPU correctly or be removed and documented as intentionally unused"
    );
}

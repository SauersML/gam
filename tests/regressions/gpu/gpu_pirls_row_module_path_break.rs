//! Regression for issue #1157: `gam::solver::gpu_kernels::pirls_row` module path break.
//!
//! Background: the GPU PIRLS row kernels moved from `crate::gpu::pirls_row` to
//! `crate::gpu::kernels::pirls_row`. The `src/solver/gpu/*.rs` launchers kept
//! importing the stale `crate::gpu::pirls_row` path, so the crate stopped
//! compiling. Commit 0f272cb landed the source fix (the launchers now use
//! `crate::gpu::kernels::pirls_row`). The issue also described a committed
//! failing test that was lost during test consolidation.
//!
//! This file restores that guard. It links the `gam` library — which forces the
//! whole lib (including the Linux `mod linux_impl` under `src/solver/gpu`) to
//! compile — and asserts genuine invariants of the GPU row-kernel symbol-naming
//! scheme reached **only** through `gam::solver::gpu_kernels::pirls_row::...`. If the
//! module is ever moved/renamed again and a launcher's import goes stale, the
//! lib stops compiling and this test stops building; if the symbol-naming
//! scheme drifts, these assertions fail.

use gam::solver::gpu_kernels::pirls_row::PirlsRowFamily;

/// Exactly six built-in families are exposed, and `ALL` enumerates each once.
#[test]
fn pirls_row_family_all_has_six_distinct_families() {
    let all = PirlsRowFamily::ALL;
    assert_eq!(
        all.len(),
        6,
        "PirlsRowFamily::ALL must enumerate exactly the six built-in (response, link) families"
    );

    // No duplicate enum variants in ALL.
    for (i, a) in all.iter().enumerate() {
        for b in &all[i + 1..] {
            assert_ne!(
                a, b,
                "PirlsRowFamily::ALL must not list the same family twice ({a:?})"
            );
        }
    }
}

/// `as_str()` is the hyphenated family label, distinct across the six families.
#[test]
fn pirls_row_family_as_str_is_hyphenated_and_distinct() {
    let mut labels = Vec::new();
    for family in PirlsRowFamily::ALL {
        let label = family.as_str();
        assert!(
            !label.is_empty(),
            "as_str() for {family:?} must be a non-empty family label"
        );
        // Hyphenated form: lowercase ascii + '-' separators, never '_'.
        assert!(
            label.chars().all(|c| c.is_ascii_lowercase() || c == '-'),
            "as_str() for {family:?} must be the hyphenated form (lowercase + '-'): got {label:?}"
        );
        assert!(
            !label.contains('_'),
            "as_str() for {family:?} is the hyphenated form, not the underscored symbol: got {label:?}"
        );
        labels.push(label);
    }
    let mut deduped = labels.clone();
    deduped.sort_unstable();
    deduped.dedup();
    assert_eq!(
        deduped.len(),
        labels.len(),
        "the six families must map to six distinct as_str() labels: {labels:?}"
    );
}

/// The three CUDA entry symbols per family share one family suffix, and that
/// suffix is the underscored form of `as_str()`. The `row`/`solve`/`ladder`
/// prefixes distinguish the three kernels.
#[test]
fn pirls_row_kernel_symbols_share_one_underscored_family_suffix() {
    for family in PirlsRowFamily::ALL {
        let suffix = family.as_str().replace('-', "_");

        let row = family.kernel_name();
        let solve = family.solve_kernel_name();
        let ladder = family.ladder_kernel_name();

        assert_eq!(
            row,
            format!("pirls_row_{suffix}"),
            "row kernel symbol for {family:?} must be pirls_row_<family_suffix>"
        );
        assert_eq!(
            solve,
            format!("pirls_solve_{suffix}"),
            "solve kernel symbol for {family:?} must be pirls_solve_<family_suffix>"
        );
        assert_eq!(
            ladder,
            format!("pirls_ladder_{suffix}"),
            "ladder kernel symbol for {family:?} must be pirls_ladder_<family_suffix>"
        );

        // All three are distinct (different kernel-mode prefixes) yet share the
        // one family suffix.
        assert!(
            row.ends_with(&suffix) && solve.ends_with(&suffix) && ladder.ends_with(&suffix),
            "the row/solve/ladder symbols for {family:?} must share the same family suffix {suffix:?}"
        );
        assert_ne!(row, solve);
        assert_ne!(row, ladder);
        assert_ne!(solve, ladder);
    }
}

/// The six families map to six distinct symbols within each kernel mode — no two
/// families collide on the same CUDA entry symbol.
#[test]
fn pirls_row_six_families_map_to_six_distinct_symbols() {
    for project in [
        PirlsRowFamily::kernel_name as fn(PirlsRowFamily) -> &'static str,
        PirlsRowFamily::solve_kernel_name,
        PirlsRowFamily::ladder_kernel_name,
    ] {
        let mut symbols: Vec<&'static str> =
            PirlsRowFamily::ALL.iter().map(|f| project(*f)).collect();
        assert_eq!(symbols.len(), 6);
        symbols.sort_unstable();
        symbols.dedup();
        assert_eq!(
            symbols.len(),
            6,
            "each kernel mode must map the six families to six distinct CUDA symbols"
        );
    }
}

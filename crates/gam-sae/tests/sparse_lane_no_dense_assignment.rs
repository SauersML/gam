//! E1 (#985) demotion guard: the DENSE `SaeAssignment` representation must never
//! be constructed on the production SPARSE-code lane.
//!
//! `SaeAssignment` (the dense `N×K` routing state) is the dense-certification /
//! debug-and-research lane only. The production SAE path is the sparse-code lane
//! (`crate::sparse_dict`), whose per-row state is fixed-width `(indices, codes)` —
//! never an `N×K` assignment. This test LOCKS that architectural invariant by
//! scanning the sparse lane's own source: outside `#[cfg(test)]` regions, every
//! `sparse_dict/*.rs` file must contain ZERO `SaeAssignment` constructions.
//!
//! It scans ONLY the sparse lane, so it cannot break the overcomplete dense
//! research / certification path (which legitimately builds `SaeAssignment` at
//! K > P for small N) — that path lives in the manifold engine, not here.

use std::path::PathBuf;

/// Drop every `#[cfg(test)]`-guarded item from Rust source, returning only the
/// production (non-test) code. For each `#[cfg(test)]` attribute we skip from it
/// through the matching close brace of the item it guards (`mod`/`fn`/`impl`),
/// brace-balanced from the first `{` after the attribute. All region boundaries
/// (`#`, `{`, `}`) are ASCII, so the retained `&str` slices stay valid UTF-8 even
/// though the source contains non-ASCII (e.g. `×`, `≤`) in comments.
fn strip_cfg_test_regions(src: &str) -> String {
    const ATTR: &str = "#[cfg(test)]";
    let mut kept = String::with_capacity(src.len());
    let mut cursor = 0usize;
    while let Some(rel) = src[cursor..].find(ATTR) {
        let attr = cursor + rel;
        kept.push_str(&src[cursor..attr]);
        match src[attr..].find('{') {
            Some(brace_rel) => {
                let brace = attr + brace_rel;
                let bytes = src.as_bytes();
                let mut depth = 0i32;
                let mut j = brace;
                while j < bytes.len() {
                    match bytes[j] {
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            if depth == 0 {
                                j += 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                    j += 1;
                }
                cursor = j;
            }
            // An attribute with no following block is malformed; drop the tail.
            None => cursor = src.len(),
        }
    }
    kept.push_str(&src[cursor..]);
    kept
}

/// Guard-the-guard: the stripper removes `#[cfg(test)]` blocks and ONLY those, so
/// the scan below cannot be defeated by moving a construction into a test module,
/// nor does it vacuously pass by deleting production code.
#[test]
fn strip_cfg_test_regions_removes_guarded_blocks_only() {
    let src = "fn prod() { build(SaeAssignment::new()); }\n\
               #[cfg(test)]\n\
               mod tests { fn t() { build(SaeAssignment::from_blocks()); } }\n\
               fn also_prod() {}\n";
    let stripped = strip_cfg_test_regions(src);
    assert!(
        stripped.contains("SaeAssignment::new"),
        "production code before the cfg(test) block must survive"
    );
    assert!(
        stripped.contains("also_prod"),
        "production code after the cfg(test) block must survive"
    );
    assert!(
        !stripped.contains("SaeAssignment::from_blocks"),
        "the cfg(test) block must be stripped"
    );
}

/// The production sparse-code lane constructs ZERO dense `SaeAssignment`s.
#[test]
fn sparse_lane_constructs_no_dense_assignment() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("sparse_dict");
    assert!(
        dir.is_dir(),
        "sparse-code lane directory must exist at {dir:?}"
    );

    // A dense-assignment CONSTRUCTION is a struct literal (`SaeAssignment {`) or
    // any associated-fn call (`SaeAssignment::…`, covering every constructor —
    // `new`, `from_blocks_with_mode`, `from_blocks_with_mode_and_manifolds`, …).
    const CONSTRUCTION_NEEDLES: [&str; 2] = ["SaeAssignment {", "SaeAssignment::"];

    let mut scanned = 0usize;
    let mut offenders: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("read sparse_dict directory") {
        let path = entry.expect("sparse_dict dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("read sparse-lane source file");
        let production = strip_cfg_test_regions(&src);
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        for needle in CONSTRUCTION_NEEDLES {
            if production.contains(needle) {
                offenders.push(format!("{name}: production code contains `{needle}`"));
            }
        }
        scanned += 1;
    }

    assert!(scanned > 0, "expected to scan at least one sparse_dict source file");
    assert!(
        offenders.is_empty(),
        "the production sparse-code lane must construct ZERO dense SaeAssignments — that \
         dense N×K routing state is the certification / debug-and-research lane only (#985 / E1); \
         found {} offender(s): {offenders:?}",
        offenders.len()
    );
}

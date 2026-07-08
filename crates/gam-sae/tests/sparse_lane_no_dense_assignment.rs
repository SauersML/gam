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

use gam_sae::front_door::{admit_dense_certification, admit_sae_fit, SaeFitLane};
use gam_sae::sparse_dict::{fit_sparse_dictionary, SparseDictConfig};
use ndarray::Array2;
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

/// Runtime lock (complements the static source scan): at a large-`K` shape
/// (`K ≫ P`) the front door demotes to the sparse-code lane, the dense-engine
/// guard REFUSES, and a real `fit_sparse_dictionary` returns fixed-width sparse
/// state `N×s` — never the dense `N×K` assignment the front door exists to avoid.
#[test]
fn large_k_sparse_fit_stays_fixed_width_and_never_materializes_dense_n_by_k() {
    // A wildly overcomplete shape: K = 256 atoms into a P = 8 response. The dense
    // assignment N·K would be 32× the response scale N·P, so this is squarely the
    // sparse lane the guard reserves the dense engine against.
    let n_obs = 96usize;
    let p_out = 8usize;
    let k_atoms = 256usize;
    let active = 4usize;

    // Front-door admission: this shape routes to sparse codes, and the dense-engine
    // guard refuses it, pointing the caller at the sparse-code lane.
    let admission = admit_sae_fit(n_obs, p_out, k_atoms).expect("admission");
    assert_eq!(admission.lane, SaeFitLane::SparseCodes);
    let refusal = admit_dense_certification(n_obs, p_out, k_atoms)
        .expect_err("dense engine must refuse the K > P shape");
    assert!(
        refusal.contains("sparse-code lane"),
        "refusal must point at the sparse-code lane; got: {refusal}"
    );

    // Deterministic planted data: each row is a scaled copy of one of `p_out`
    // orthonormal axes so the fit has real structure to route (no RNG dependency).
    let mut x = Array2::<f32>::zeros((n_obs, p_out));
    for row in 0..n_obs {
        let axis = row % p_out;
        x[[row, axis]] = 1.0 + 0.01 * (row / p_out) as f32;
    }

    let config = SparseDictConfig {
        active,
        max_epochs: 5,
        ..SparseDictConfig::new(k_atoms)
    };
    let fit = fit_sparse_dictionary(x.view(), &config).expect("sparse dictionary fit");

    // The whole point: the fitted routing state is fixed-width sparse `N×s`, with
    // `s = min(active, K) ≪ K`. It is NEVER an `N×K` dense assignment.
    assert_eq!(fit.active, active.min(k_atoms));
    assert_eq!(fit.indices.dim(), (n_obs, fit.active));
    assert_eq!(fit.codes.dim(), (n_obs, fit.active));
    assert!(
        fit.indices.ncols() < k_atoms,
        "sparse routing width {} must be ≪ K = {k_atoms} — an N×K state is the demoted dense lane",
        fit.indices.ncols()
    );
    // The decoder is `K×P` (the dictionary itself), the only K-scaled state — and it
    // is P-wide, not N-wide, so no `N×K` object exists anywhere in the fit.
    assert_eq!(fit.decoder.dim(), (k_atoms, p_out));
}

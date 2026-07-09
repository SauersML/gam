//! Rust-owned coercion helpers for the `ManifoldSAE.from_payload` -> flat
//! `to_dict` schema derivation (#2091 phase-2, design (A)). These reproduce the
//! Python `_sae_manifold.py` derivations bit-for-bit so the fit path can return
//! a Rust-owned `ManifoldSaeCore` built directly from the raw
//! `sae_manifold_fit_minimal` payload, with no Python dataclass in the middle.
//!
//! This first slice owns the topology-naming derivation (basis-kind -> canonical
//! topology label, scalar + per-atom), exposed to Python as
//! `sae_atom_topologies` — the same Rust-owner pattern as
//! `sae_canonical_n_harmonics`. The full `RawFitPayload -> ManifoldSaePayload`
//! assembly that also consumes these lands in a later increment. Each helper
//! mirrors an exact Python function — the doc-comments name the counterpart so a
//! future schema drift is traceable to one side.

/// Case-insensitive, `-`/`_`-interchangeable name normalizer. Mirrors
/// `_sae_manifold.py::_canon_name`: `str(name).strip().lower().replace("-", "_")`.
pub(crate) fn canon_name(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace('-', "_")
}

/// Canonical topology label for a (possibly aliased) basis kind. Mirrors
/// `_basis_to_topology`: look the CANON-normalized basis up in the
/// `_BASIS_TO_TOPOLOGY` alias map, falling back to the ORIGINAL (un-canon'd)
/// basis string when unknown — exactly as the Python `dict.get(canon, str(basis))`
/// does (the default is the raw argument, not its canon form).
pub(crate) fn basis_to_topology(basis: &str) -> String {
    // _BASIS_TO_TOPOLOGY (canonical / aliased basis kind -> canonical topology).
    match canon_name(basis).as_str() {
        "periodic" | "periodic_spline" | "circle" => "circle".to_string(),
        "sphere" => "sphere".to_string(),
        "torus" => "torus".to_string(),
        "linear" | "linear_rank1" | "affine" => "linear".to_string(),
        "linear_block" | "flat_block" => "linear_block".to_string(),
        "duchon" | "euclidean" | "euclidean_patch" | "euclidean_quadratic_patch" => {
            "euclidean".to_string()
        }
        "poincare" | "hyperbolic" | "poincare_patch" => "poincare".to_string(),
        "cylinder" => "cylinder".to_string(),
        // Unknown -> the ORIGINAL basis string verbatim (Python `str(basis)`).
        _ => basis.to_string(),
    }
}

/// Per-atom topology labels for a resolved bases list (`basis_specs` order).
/// Mirrors `_topologies_for_bases`.
pub(crate) fn topologies_for_bases(bases: &[String]) -> Vec<String> {
    bases.iter().map(|b| basis_to_topology(b)).collect()
}

/// Collapse a resolved bases list to a single topology label. Mirrors
/// `_topology_for_bases`: the common label when all atoms agree, else the honest
/// `"mixed"`. The caller guards the empty case (`_topology_for_bases(kinds) if
/// kinds else str(topology)`); this returns `None` for an empty list so the
/// caller supplies its own fallback rather than this guessing one.
pub(crate) fn topology_for_bases(bases: &[String]) -> Option<String> {
    let per_atom = topologies_for_bases(bases);
    let first = per_atom.first()?;
    if per_atom.iter().all(|t| t == first) {
        Some(first.clone())
    } else {
        Some("mixed".to_string())
    }
}

#[cfg(test)]
mod manifold_sae_coercion_tests {
    use super::*;

    #[test]
    fn canon_name_lowercases_trims_and_dashes_to_underscore() {
        assert_eq!(canon_name("  Periodic-Spline "), "periodic_spline");
        assert_eq!(canon_name("EUCLIDEAN"), "euclidean");
        assert_eq!(canon_name("linear-rank1"), "linear_rank1");
    }

    #[test]
    fn basis_to_topology_matches_python_alias_map() {
        // Every documented alias -> canonical topology, plus alias/casing forms.
        for (basis, topo) in [
            ("periodic", "circle"),
            ("periodic_spline", "circle"),
            ("Circle", "circle"),
            ("sphere", "sphere"),
            ("torus", "torus"),
            ("linear", "linear"),
            ("linear_rank1", "linear"),
            ("affine", "linear"),
            ("linear_block", "linear_block"),
            ("flat-block", "linear_block"),
            ("duchon", "euclidean"),
            ("euclidean", "euclidean"),
            ("euclidean_patch", "euclidean"),
            ("euclidean_quadratic_patch", "euclidean"),
            ("poincare", "poincare"),
            ("hyperbolic", "poincare"),
            ("poincare_patch", "poincare"),
            ("cylinder", "cylinder"),
        ] {
            assert_eq!(basis_to_topology(basis), topo, "basis {basis}");
        }
    }

    #[test]
    fn basis_to_topology_unknown_passes_through_original_string() {
        // Python `dict.get(_canon_name(basis), str(basis))` returns the RAW arg
        // (not its canon form) on a miss.
        assert_eq!(basis_to_topology("Weird-Kind"), "Weird-Kind");
    }

    #[test]
    fn topology_for_bases_common_vs_mixed() {
        assert_eq!(
            topology_for_bases(&["periodic".into(), "circle".into()]),
            Some("circle".to_string()),
        );
        assert_eq!(
            topology_for_bases(&["periodic".into(), "euclidean".into()]),
            Some("mixed".to_string()),
        );
        assert_eq!(topology_for_bases(&[]), None);
        assert_eq!(
            topologies_for_bases(&["duchon".into(), "linear".into()]),
            vec!["euclidean".to_string(), "linear".to_string()],
        );
    }
}

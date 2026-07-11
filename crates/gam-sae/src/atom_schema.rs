//! Canonical SAE token schema (issue #2236): the vocabulary for basis kinds,
//! topology labels, assignment families, and flat-block gating modes, plus the
//! strict harmonic metadata validation and the structural chart periods.
//!
//! Moved verbatim from `gam-pyffi`'s coercion module so the vocabulary is
//! owned by the library: the CLI, Rust callers, and the Python binding all
//! parse and emit the same tokens, and the binding is marshalling only.

/// Validate harmonic metadata emitted by a native fit. A periodic atom has
/// width `M = 2H + 1` with `H >= 1`; malformed metadata is an error rather than
/// a load-time repair. Non-periodic atom metadata passes through unchanged.
pub fn validated_n_harmonics(
    basis_kinds: &[String],
    raw_n_harmonics: &[i64],
    decoder_widths: &[i64],
) -> Result<Vec<i64>, String> {
    if basis_kinds.len() != raw_n_harmonics.len()
        || basis_kinds.len() != decoder_widths.len()
    {
        return Err(format!(
            "harmonic metadata length mismatch: basis_kinds={}, n_harmonics={}, decoder_widths={}",
            basis_kinds.len(),
            raw_n_harmonics.len(),
            decoder_widths.len()
        ));
    }
    let mut validated = Vec::with_capacity(basis_kinds.len());
    for (atom, ((basis, &harmonics), &width)) in basis_kinds
        .iter()
        .zip(raw_n_harmonics)
        .zip(decoder_widths)
        .enumerate()
    {
        if canonical_basis_kind(basis) == "periodic" {
            if harmonics < 1 {
                return Err(format!(
                    "periodic atom {atom} requires n_harmonics >= 1; got {harmonics}"
                ));
            }
            let expected_width = 2 * harmonics + 1;
            if width != expected_width {
                return Err(format!(
                    "periodic atom {atom} has decoder width {width}, but n_harmonics={harmonics} requires {expected_width}"
                ));
            }
        }
        validated.push(harmonics);
    }
    Ok(validated)
}

/// Case-insensitive, `-`/`_`-interchangeable SAE name normalizer.
pub fn canon_name(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace('-', "_")
}

/// Validate one canonical assignment-family token.
///
/// The fit,
/// stagewise, payload, and distilled-encoder paths all call this parser before
/// dispatching to the core [`crate::assignment::AssignmentMode`] implementation,
/// so accepted and emitted tokens cannot drift.
pub fn canonical_assignment_kind(kind: &str) -> Result<&'static str, String> {
    match kind {
        "softmax" => Ok("softmax"),
        "ordered_beta_bernoulli" => Ok("ordered_beta_bernoulli"),
        "threshold_gate" => Ok("threshold_gate"),
        "topk" => Ok("topk"),
        _ => Err(format!(
            "assignment={kind:?} is not a recognized assignment kind; expected one of \
             ['ordered_beta_bernoulli', 'softmax', 'threshold_gate', 'topk']"
        )),
    }
}

fn basis_alias_to_kind(normalized: &str) -> Option<&'static str> {
    match normalized {
        "circle" | "periodic" | "periodic_spline" => Some("periodic"),
        "sphere" => Some("sphere"),
        "torus" => Some("torus"),
        "linear" | "linear_rank1" | "affine" => Some("linear"),
        "linear_block" | "flat_block" => Some("linear_block"),
        "euclidean" | "euclidean_patch" | "euclidean_quadratic_patch" => Some("euclidean"),
        "duchon" => Some("duchon"),
        "poincare" | "hyperbolic" | "poincare_patch" => Some("poincare"),
        "cylinder" => Some("cylinder"),
        "mobius" | "mobius_band" => Some("mobius"),
        "auto" => Some("auto"),
        _ => None,
    }
}

/// Canonical basis kind for a documented topology/basis spelling. Unknown
/// precomputed kinds are normalized (trim/lower/dash-to-underscore), matching
/// the fit parser's established treatment of an explicit basis string.
pub fn canonical_basis_kind(name: &str) -> String {
    let normalized = canon_name(name);
    basis_alias_to_kind(&normalized).map_or(normalized, str::to_string)
}

/// Resolve a topology spelling to its basis kind while preserving an unknown
/// caller-supplied precomputed name verbatim.
pub fn basis_kind_for_topology(name: &str) -> String {
    let normalized = canon_name(name);
    basis_alias_to_kind(&normalized).map_or_else(|| name.to_string(), str::to_string)
}

/// Canonical topology label for a (possibly aliased) basis kind, falling back
/// to the original (un-normalized) basis string for an unknown precomputed kind.
pub fn basis_to_topology(basis: &str) -> String {
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
        "mobius" | "mobius_band" => "mobius".to_string(),
        "auto" => "auto".to_string(),
        // Unknown -> the original basis string verbatim.
        _ => basis.to_string(),
    }
}

/// Canonical topology for a topology or basis spelling. Unknown precomputed
/// names are normalized because this is a canonicalizer, not a round-trip
/// label conversion.
pub fn canonical_topology(name: &str) -> String {
    basis_to_topology(&canonical_basis_kind(name))
}

/// Structural coordinate periods for a fitted basis. `None` marks an open
/// axis; finite values are the chart's exact wrap period.
pub fn coordinate_periods_for_basis(
    basis: &str,
    latent_dim: usize,
) -> Result<Vec<Option<f64>>, String> {
    let kind = canonical_basis_kind(basis);
    match kind.as_str() {
        "periodic" | "torus" => Ok(vec![Some(1.0); latent_dim]),
        "cylinder" if latent_dim == 2 => Ok(vec![Some(1.0), None]),
        "sphere" if latent_dim == 2 => Ok(vec![None, Some(std::f64::consts::TAU)]),
        // The Möbius chart is represented on its double cover: angle period 2,
        // open width axis. Deck invariance lives in the basis parity.
        "mobius" if latent_dim == 2 => Ok(vec![Some(2.0), None]),
        "cylinder" | "sphere" | "mobius" => Err(format!(
            "{kind} atoms require latent dimension 2; got {latent_dim}"
        )),
        _ => Ok(vec![None; latent_dim]),
    }
}

/// Assignment family implied by the public flat-block gating vocabulary.
pub fn flat_block_assignment(gating: &str) -> Result<&'static str, String> {
    match gating {
        "norm_selection" => Ok("ordered_beta_bernoulli"),
        "separate_gate" => Ok("threshold_gate"),
        _ => Err(format!(
            "flat_block gating={gating:?} is not recognized; expected one of \
             ['norm_selection', 'separate_gate']"
        )),
    }
}

/// Per-atom topology labels for a resolved bases list (`basis_specs` order).
pub fn topologies_for_bases(bases: &[String]) -> Vec<String> {
    bases.iter().map(|b| basis_to_topology(b)).collect()
}

/// Collapse a resolved bases list to its common topology or the honest
/// `"mixed"` label. Empty dictionaries have no topology and return `None`.
pub fn topology_for_bases(bases: &[String]) -> Option<String> {
    let per_atom = topologies_for_bases(bases);
    let first = per_atom.first()?;
    if per_atom.iter().all(|t| t == first) {
        Some(first.clone())
    } else {
        Some("mixed".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::validated_n_harmonics;

    #[test]
    fn periodic_harmonic_metadata_is_strict() {
        let basis = vec!["periodic".to_string()];
        assert_eq!(
            validated_n_harmonics(&basis, &[2], &[5]).expect("valid periodic metadata"),
            vec![2]
        );
        assert!(validated_n_harmonics(&basis, &[0], &[5]).is_err());
        assert!(validated_n_harmonics(&basis, &[2], &[7]).is_err());
        assert!(validated_n_harmonics(&basis, &[2, 1], &[5]).is_err());
    }
}

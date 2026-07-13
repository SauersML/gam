//! Strict SAE token schema: the vocabulary for basis kinds, topology labels,
//! assignment families, and flat-block gating modes, plus the
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
    if basis_kinds.len() != raw_n_harmonics.len() || basis_kinds.len() != decoder_widths.len() {
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
        validate_fitted_basis_kind(basis)?;
        if basis == "periodic" {
            if harmonics < 1 {
                return Err(format!(
                    "periodic atom {atom} requires n_harmonics >= 1; got {harmonics}"
                ));
            }
            let expected_width = harmonics
                .checked_mul(2)
                .and_then(|twice| twice.checked_add(1))
                .ok_or_else(|| {
                    format!(
                        "periodic atom {atom} basis width overflows i64 for n_harmonics={harmonics}"
                    )
                })?;
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

/// Validate one canonical assignment-family token.
///
/// The fit, payload, and native routing paths all call this parser before
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

/// Validate a basis kind that may appear in a converged native artifact.
pub fn validate_fitted_basis_kind(name: &str) -> Result<(), String> {
    match name {
        "periodic" | "sphere" | "torus" | "linear" | "linear_block" | "euclidean" | "duchon"
        | "poincare" | "cylinder" | "mobius" | "finite_set" | "spectral_graph" => Ok(()),
        _ => Err(format!(
            "basis kind {name:?} is not canonical; expected one of ['cylinder', 'duchon', \
             'euclidean', 'finite_set', 'linear', 'linear_block', 'mobius', 'periodic', \
             'poincare', 'spectral_graph', 'sphere', 'torus']"
        )),
    }
}

/// Validate a public fit seed. Discovery-only atom kinds cannot be seeded.
pub fn validate_seed_basis_kind(name: &str) -> Result<(), String> {
    match name {
        "periodic" | "sphere" | "torus" | "linear" | "linear_block" | "euclidean" | "duchon"
        | "poincare" | "mobius" | "auto" => Ok(()),
        "cylinder" | "finite_set" => Err(format!(
            "basis kind {name:?} is discovery-only and cannot seed a fit"
        )),
        _ => Err(format!(
            "basis kind {name:?} is not canonical; expected one of ['auto', 'duchon', \
             'euclidean', 'linear', 'linear_block', 'mobius', 'periodic', 'poincare', \
             'sphere', 'torus']"
        )),
    }
}

/// Resolve one exact public topology token to its exact seed basis kind.
pub fn basis_kind_for_topology(name: &str) -> Result<String, String> {
    match name {
        "circle" => Ok("periodic".to_string()),
        "sphere" | "torus" | "linear" | "linear_block" | "euclidean" | "duchon" | "poincare"
        | "mobius" | "auto" => Ok(name.to_string()),
        "cylinder" | "finite_set" => Err(format!(
            "topology {name:?} is discovery-only and cannot seed a fit"
        )),
        _ => Err(format!(
            "topology {name:?} is not canonical; expected one of ['auto', 'circle', 'duchon', \
             'euclidean', 'linear', 'linear_block', 'mobius', 'poincare', 'sphere', 'torus']"
        )),
    }
}

/// Exact topology label for a validated fitted basis kind.
pub fn basis_to_topology(basis: &str) -> Result<String, String> {
    validate_fitted_basis_kind(basis)?;
    Ok(match basis {
        "periodic" => "circle",
        "duchon" | "euclidean" => "euclidean",
        "spectral_graph" => "graph",
        other => other,
    }
    .to_string())
}

/// Validate and return one exact public topology token.
pub fn canonical_topology(name: &str) -> Result<String, String> {
    basis_kind_for_topology(name)?;
    Ok(name.to_string())
}

/// Structural coordinate periods for a fitted basis. `None` marks an open
/// axis; finite values are the chart's exact wrap period.
pub fn coordinate_periods_for_basis(
    basis: &str,
    latent_dim: usize,
) -> Result<Vec<Option<f64>>, String> {
    validate_fitted_basis_kind(basis)?;
    match basis {
        "periodic" | "torus" => Ok(vec![Some(1.0); latent_dim]),
        "cylinder" if latent_dim == 2 => Ok(vec![Some(1.0), None]),
        "sphere" if latent_dim == 2 => Ok(vec![None, Some(std::f64::consts::TAU)]),
        // The Möbius chart is represented on its double cover: angle period 2,
        // open width axis. Deck invariance lives in the basis parity.
        "mobius" if latent_dim == 2 => Ok(vec![Some(2.0), None]),
        "cylinder" | "sphere" | "mobius" => Err(format!(
            "{basis} atoms require latent dimension 2; got {latent_dim}"
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
pub fn topologies_for_bases(bases: &[String]) -> Result<Vec<String>, String> {
    bases.iter().map(|b| basis_to_topology(b)).collect()
}

/// Collapse a resolved bases list to its common topology or the honest
/// `"mixed"` label. Empty dictionaries have no topology and return `None`.
pub fn topology_for_bases(bases: &[String]) -> Result<Option<String>, String> {
    let per_atom = topologies_for_bases(bases)?;
    let Some(first) = per_atom.first() else {
        return Ok(None);
    };
    if per_atom.iter().all(|t| t == first) {
        Ok(Some(first.clone()))
    } else {
        Ok(Some("mixed".to_string()))
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

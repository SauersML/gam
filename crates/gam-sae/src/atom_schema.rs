//! Strict SAE token schema: the vocabulary for basis kinds, topology labels,
//! assignment families, and flat-block gating modes, plus the
//! strict harmonic metadata validation and the structural chart periods.
//!
//! Moved verbatim from `gam-pyffi`'s coercion module so the vocabulary is
//! owned by the library: the CLI, Rust callers, and the Python binding all
//! parse and emit the same tokens, and the binding is marshalling only.

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
        "periodic" | "sphere" | "torus" | "projective_plane" | "klein_bottle" | "linear"
        | "linear_block" | "euclidean" | "duchon" | "poincare" | "cylinder" | "mobius"
        | "finite_set" | "spectral_graph" => Ok(()),
        _ => Err(format!(
            "basis kind {name:?} is not canonical; expected one of ['cylinder', 'duchon', \
             'euclidean', 'finite_set', 'klein_bottle', 'linear', 'linear_block', 'mobius', \
             'periodic', 'poincare', 'projective_plane', 'spectral_graph', 'sphere', 'torus']"
        )),
    }
}

/// Validate a public fit seed. Discovery-only atom kinds cannot be seeded.
pub fn validate_seed_basis_kind(name: &str) -> Result<(), String> {
    match name {
        "periodic" | "sphere" | "torus" | "projective_plane" | "klein_bottle" | "linear"
        | "linear_block" | "euclidean" | "duchon" | "poincare" | "mobius" | "auto" => Ok(()),
        "cylinder" | "finite_set" => Err(format!(
            "basis kind {name:?} is discovery-only and cannot seed a fit"
        )),
        _ => Err(format!(
            "basis kind {name:?} is not canonical; expected one of ['auto', 'duchon', \
             'euclidean', 'klein_bottle', 'linear', 'linear_block', 'mobius', 'periodic', \
             'poincare', 'projective_plane', 'sphere', 'torus']"
        )),
    }
}

/// Resolve one exact public topology token to its exact seed basis kind.
pub fn basis_kind_for_topology(name: &str) -> Result<String, String> {
    match name {
        "circle" => Ok("periodic".to_string()),
        "sphere" | "torus" | "projective_plane" | "klein_bottle" | "linear" | "linear_block"
        | "euclidean" | "duchon" | "poincare" | "mobius" | "auto" => Ok(name.to_string()),
        "cylinder" | "finite_set" => Err(format!(
            "topology {name:?} is discovery-only and cannot seed a fit"
        )),
        _ => Err(format!(
            "topology {name:?} is not canonical; expected one of ['auto', 'circle', 'duchon', \
             'euclidean', 'klein_bottle', 'linear', 'linear_block', 'mobius', 'poincare', \
             'projective_plane', 'sphere', 'torus']"
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


//! Manifold parameter decomposition (#2951): the Python transport of the Rust
//! surface.
//!
//! `gam_sae::parameter_decomposition::surface` owns the request document, every
//! operation and the report. This module converts a `dict[str, ndarray]` to owned
//! `f64` arrays, runs the surface with the interpreter lock released, and hands the
//! report's JSON text and named arrays back. The CLI calls the same entry, so both
//! produce the same report bytes.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use std::cell::RefCell;

use gam::terms::sae::parameter_decomposition::adversary::{
    ObjectiveJet, SeparationObjective, SmoothnessCertificate, ZonotopeSeparationOracle,
};
use gam::terms::sae::parameter_decomposition::codec::PaddedPacketCode;
use gam::terms::sae::parameter_decomposition::moments::{
    GeneratorPart, MaskDomain, MaskMomentSystem, MomentBlock, MomentVector,
};
use gam::terms::sae::parameter_decomposition::supports::{
    CardinalityCode, FailureHypergraph, minimum_code_support, ranked_support,
};
use gam::terms::sae::parameter_decomposition::surface::run_parameter_decomposition;
use ndarray::Array1;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::py_value_error;

/// Run one `gam.mpd-request` document against its named input arrays and return
/// `(report_json, arrays)`, where `arrays` holds every array the report names.
///
/// The surface has no cancellation point, so an interrupt returns control to Python
/// and the abandoned worker finishes and drops on its own thread.
#[pyfunction]
fn parameter_decomposition_run<'py>(
    py: Python<'py>,
    request_json: String,
    tensors: &Bound<'py, PyDict>,
) -> PyResult<(String, Bound<'py, PyDict>)> {
    let mut inputs = BTreeMap::new();
    for (key, value) in tensors.iter() {
        let name: String = key.extract().map_err(|error| {
            py_value_error(format!(
                "parameter_decomposition_run: input array names must be str: {error}"
            ))
        })?;
        let array: PyReadonlyArrayDyn<'py, f64> = value.extract().map_err(|error| {
            py_value_error(format!(
                "parameter_decomposition_run: input array {name:?} must be a float64 ndarray: {error}"
            ))
        })?;
        inputs.insert(name, array.as_array().to_owned());
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let (report_json, arrays) = crate::run_sae_fit_interruptible(py, "gam-mpd", &cancel, move || {
        run_parameter_decomposition(&request_json, &inputs)
            .and_then(|output| output.report_json().map(|json| (json, output.arrays)))
    })?
    .map_err(|error| py_value_error(error.to_string()))?;
    let named = PyDict::new(py);
    for (name, array) in arrays {
        named.set_item(name, array.into_pyarray(py))?;
    }
    Ok((report_json, named))
}

/// The native separation objective as a Python callable
/// `objective(mask) -> (value, value_roundoff, moment_gradient, gradient_roundoff)`: the
/// teacher's divergence at a float64 mask of length `C` and its derivative in the moment `q`
/// (length `K`), each with its roundoff bound. The first Python exception is kept and re-raised.
struct PythonObjective<'py> {
    objective: Bound<'py, PyAny>,
    /// Literal pieces: control `c` is its own one-dimensional block, so the gradient's entry `c`
    /// is block `c`.
    literal: bool,
    gradient_lipschitz: Option<f64>,
    error: RefCell<Option<PyErr>>,
}

impl SeparationObjective for PythonObjective<'_> {
    fn evaluate(&self, mask: &[f64]) -> Result<ObjectiveJet, String> {
        let py = self.objective.py();
        let result = self
            .objective
            .call1((mask.to_vec().into_pyarray(py),))
            .and_then(|out| {
                let (value, value_roundoff, gradient, gradient_roundoff): (
                    f64,
                    f64,
                    PyReadonlyArray1<'_, f64>,
                    f64,
                ) = out.extract()?;
                let gradient = gradient.as_array();
                let blocks = if self.literal {
                    gradient.iter().map(|&entry| Array1::from_elem(1, entry)).collect()
                } else {
                    vec![gradient.to_owned()]
                };
                Ok(ObjectiveJet {
                    value,
                    value_roundoff,
                    moment_gradient: MomentVector { blocks },
                    gradient_roundoff,
                })
            });
        result.map_err(|error| {
            let message = error.to_string();
            self.error.borrow_mut().get_or_insert(error);
            message
        })
    }

    fn smoothness(&self) -> Option<SmoothnessCertificate> {
        self.gradient_lipschitz.map(|gradient_lipschitz| SmoothnessCertificate {
            gradient_lipschitz,
            derivation: "stated by the Python objective".to_string(),
        })
    }
}

/// The minimum-code robust support at one input (#2951 P12): counterexample-guided search whose
/// separation oracle is the Frank-Wolfe ascent over the mask-moment zonotope, run on the Python
/// `objective`. `generators` is `C x K` (row `c` is `v_c`), or `None` for literal pieces, whose
/// moment is the deletion vector itself (`K = C`, each control its own block, so the moment system
/// is `O(C)` and never a dense identity). `lower`/`upper` are the declared mask interval of each
/// control (each contains 1), `piece_bits` the declared longest decoded body of one piece (each kept
/// piece costs that much in the support code, design §11.1), `epsilon` the declared tolerance, and
/// `gradient_lipschitz` a stated Lipschitz constant of `dF/dq` when one is derived.
///
/// Returns `{"support", "status", "code_lower", "code_upper", "risk_lower", "risk_upper",
/// "separations", "edges"}`: `status` is `"certified"` (the support's risk is bounded by
/// epsilon) or `"unresolved"` (the minimum-code candidate was neither refuted nor certified;
/// `support` is that candidate), and the risk bounds are the oracle's evidence about it.
///
/// With `ranking` (a permutation of some controls, most important first) the search is instead
/// the shortest sufficient leading run of that ranking (`supports::ranked_support`), found by
/// bisection in about `log2 C` checks: `code_lower` is then absent (no minimum over all subsets is
/// claimed) and `edges` is 1 when the run one shorter was refuted.
#[pyfunction]
#[pyo3(signature = (objective, generators, lower, upper, piece_bits, epsilon, gradient_lipschitz=None, ranking=None))]
fn parameter_decomposition_robust_support<'py>(
    py: Python<'py>,
    objective: Bound<'py, PyAny>,
    generators: Option<PyReadonlyArray2<'py, f64>>,
    lower: PyReadonlyArray1<'py, f64>,
    upper: PyReadonlyArray1<'py, f64>,
    piece_bits: u64,
    epsilon: f64,
    gradient_lipschitz: Option<f64>,
    ranking: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyDict>> {
    let (lower, upper) = (lower.as_array(), upper.as_array());
    let literal = generators.is_none();
    let (blocks, parts): (Vec<MomentBlock>, Vec<Vec<GeneratorPart>>) = match &generators {
        Some(generators) => {
            let generators = generators.as_array();
            (
                vec![MomentBlock { dimension: generators.ncols() }],
                generators
                    .rows()
                    .into_iter()
                    .map(|row| vec![GeneratorPart { block: 0, vector: row.to_owned() }])
                    .collect(),
            )
        }
        None => (
            vec![MomentBlock { dimension: 1 }; lower.len()],
            (0..lower.len())
                .map(|control| vec![GeneratorPart { block: control, vector: Array1::from_elem(1, 1.0) }])
                .collect(),
        ),
    };
    let controls = parts.len();
    if lower.len() != controls || upper.len() != controls {
        return Err(py_value_error(format!(
            "robust_support: {controls} generators but {} lower and {} upper bounds",
            lower.len(),
            upper.len()
        )));
    }
    let system = MaskMomentSystem::new(blocks, parts)
        .map_err(|error| py_value_error(format!("robust_support: {error:?}")))?;
    let domain = MaskDomain::new(lower.iter().copied().zip(upper.iter().copied()).collect())
        .map_err(|error| py_value_error(format!("robust_support: {error:?}")))?;
    let objective = PythonObjective {
        objective,
        literal,
        gradient_lipschitz,
        error: RefCell::new(None),
    };
    let code = PaddedPacketCode { body_bits: piece_bits };
    let bound = |value: Option<f64>| value.unwrap_or(f64::NAN);
    let summary = ZonotopeSeparationOracle::new(&system, &domain, epsilon, &objective)
        .map_err(|error| format!("{error}"))
        .and_then(|mut oracle| match &ranking {
            Some(ranking) => {
                let ranked = ranked_support(&mut oracle, ranking, epsilon).map_err(|error| format!("{error}"))?;
                let found = &ranked.found;
                let status = if found.evidence.certifies_at_most(epsilon) { "certified" } else { "unresolved" };
                let bits = code.support_bits(controls, found.support.len()).map_err(|error| format!("{error:?}"))?;
                Ok((
                    found.support.members().to_vec(),
                    status,
                    f64::NAN,
                    bits as f64,
                    bound(found.evidence.lower_bound()),
                    bound(found.evidence.upper_bound()),
                    ranked.separations,
                    usize::from(ranked.refuted_shorter.is_some()),
                ))
            }
            None => {
                let search = minimum_code_support(&mut oracle, &code, epsilon, FailureHypergraph::new(controls))
                    .map_err(|error| format!("{error}"))?;
                let (status, found) = match (&search.certified, &search.undecided) {
                    (_, Some(undecided)) => ("unresolved", undecided),
                    (Some(certified), None) => ("certified", certified),
                    (None, None) => return Err("the search returned no support".to_string()),
                };
                Ok((
                    found.support.members().to_vec(),
                    status,
                    bound(search.code.lower_bound()),
                    bound(search.code.upper_bound()),
                    bound(found.evidence.lower_bound()),
                    bound(found.evidence.upper_bound()),
                    search.separations,
                    search.hypergraph.edges().len(),
                ))
            }
        });
    let (support, status, code_lower, code_upper, risk_lower, risk_upper, separations, edges) = match summary {
        Ok(summary) => summary,
        Err(message) => {
            return Err(objective
                .error
                .into_inner()
                .unwrap_or_else(|| py_value_error(format!("robust_support: {message}"))));
        }
    };
    let out = PyDict::new(py);
    out.set_item("support", support.iter().map(|&c| c as u64).collect::<Vec<_>>().into_pyarray(py))?;
    out.set_item("status", status)?;
    out.set_item("code_lower", code_lower)?;
    out.set_item("code_upper", code_upper)?;
    out.set_item("risk_lower", risk_lower)?;
    out.set_item("risk_upper", risk_upper)?;
    out.set_item("separations", separations)?;
    // CEGAR: the recorded failure edges. Ranked: 1 when the run one shorter was refuted.
    out.set_item("edges", edges)?;
    Ok(out)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(parameter_decomposition_run, module)?)?;
    module.add_function(wrap_pyfunction!(parameter_decomposition_robust_support, module)?)?;
    Ok(())
}

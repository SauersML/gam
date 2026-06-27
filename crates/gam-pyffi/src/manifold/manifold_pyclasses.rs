//! Python-visible manifold descriptor classes.
//!
//! Self-contained seam extracted from the pyffi monolith (issue #780): the
//! `#[pyclass]` config objects that describe a fitting manifold to the Python
//! layer (`EuclideanManifold`, `CircleManifold`, `SphereManifold`,
//! `TorusManifold`, `GrassmannManifold`, `StiefelManifold`, `SpdManifold`,
//! `ProductManifold`) and their shared `1 <= k <= n` frame-domain validator.
//!
//! These are pure descriptor/`to_json`-serialization types: they depend on
//! nothing in the rest of the module except the boundary error helper
//! `py_value_error` and the crate-local `PyObject` alias. Registration stays in
//! the `#[pymodule]` block via a focused re-import.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::{PyObject, py_value_error};

#[pyclass(
    module = "gam_pyffi._rust",
    name = "EuclideanManifold",
    skip_from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EuclideanManifold {
    #[pyo3(get, set)]
    dim: i64,
}

#[pymethods]
impl EuclideanManifold {
    #[new]
    fn new(dim: i64) -> Self {
        Self { dim }
    }

    fn __repr__(&self) -> String {
        format!("EuclideanManifold(dim={})", self.dim)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("kind", "euclidean")?;
        out.set_item("dim", self.dim)?;
        Ok(out.into_any().unbind())
    }
}

#[pyclass(
    module = "gam_pyffi._rust",
    name = "CircleManifold",
    skip_from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CircleManifold {}

#[pymethods]
impl CircleManifold {
    #[new]
    fn new() -> Self {
        Self {}
    }

    fn __repr__(&self) -> String {
        "CircleManifold()".to_owned()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("kind", "circle")?;
        Ok(out.into_any().unbind())
    }
}

#[pyclass(
    module = "gam_pyffi._rust",
    name = "SphereManifold",
    skip_from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SphereManifold {
    #[pyo3(get, set)]
    intrinsic_dim: i64,
}

#[pymethods]
impl SphereManifold {
    #[new]
    fn new(intrinsic_dim: i64) -> Self {
        Self { intrinsic_dim }
    }

    fn __repr__(&self) -> String {
        format!("SphereManifold(intrinsic_dim={})", self.intrinsic_dim)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("kind", "sphere")?;
        out.set_item("intrinsic_dim", self.intrinsic_dim)?;
        Ok(out.into_any().unbind())
    }
}

#[pyclass(
    module = "gam_pyffi._rust",
    name = "TorusManifold",
    skip_from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TorusManifold {
    #[pyo3(get, set)]
    dim: i64,
}

#[pymethods]
impl TorusManifold {
    #[new]
    fn new(dim: i64) -> Self {
        Self { dim }
    }

    fn __repr__(&self) -> String {
        format!("TorusManifold(dim={})", self.dim)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("kind", "torus")?;
        out.set_item("dim", self.dim)?;
        Ok(out.into_any().unbind())
    }
}

/// Validate the `1 <= k <= n` domain shared by the constrained-frame
/// manifolds `Gr(k, n)` (k-dimensional subspaces of R^n) and `St(n, k)`
/// (k-frames in R^n). Both exist only on this domain: with `k > n` there is
/// no k-dimensional subspace / no k orthonormal columns in R^n, and the
/// dimension formulas (`k(n-k)` resp. `nk - k(k+1)/2`) cease to describe a
/// frame manifold. Rejecting here keeps every Python-visible Grassmann/Stiefel
/// object inside its domain, mirroring the Rust-core constructors.
fn validate_frame_domain(name: &str, k: i64, n: i64) -> PyResult<()> {
    if k < 1 || n < 1 || k > n {
        return Err(py_value_error(format!(
            "{name} requires 1 <= k <= n (got k={k}, n={n})"
        )));
    }
    Ok(())
}

#[pyclass(
    module = "gam_pyffi._rust",
    name = "GrassmannManifold",
    skip_from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GrassmannManifold {
    #[pyo3(get)]
    k: i64,
    #[pyo3(get)]
    n: i64,
}

#[pymethods]
impl GrassmannManifold {
    #[new]
    fn new(k: i64, n: i64) -> PyResult<Self> {
        validate_frame_domain("GrassmannManifold", k, n)?;
        Ok(Self { k, n })
    }

    /// Set the subspace dimension `k`, rejecting any value that would leave the
    /// `1 <= k <= n` domain for the current ambient dimension `n`.
    #[setter]
    fn set_k(&mut self, k: i64) -> PyResult<()> {
        validate_frame_domain("GrassmannManifold", k, self.n)?;
        self.k = k;
        Ok(())
    }

    /// Set the ambient dimension `n`, rejecting any value that would leave the
    /// `1 <= k <= n` domain for the current subspace dimension `k`.
    #[setter]
    fn set_n(&mut self, n: i64) -> PyResult<()> {
        validate_frame_domain("GrassmannManifold", self.k, n)?;
        self.n = n;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("GrassmannManifold(k={}, n={})", self.k, self.n)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("kind", "grassmann")?;
        out.set_item("k", self.k)?;
        out.set_item("n", self.n)?;
        Ok(out.into_any().unbind())
    }
}

#[pyclass(
    module = "gam_pyffi._rust",
    name = "StiefelManifold",
    skip_from_py_object
)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StiefelManifold {
    #[pyo3(get)]
    k: i64,
    #[pyo3(get)]
    n: i64,
}

#[pymethods]
impl StiefelManifold {
    #[new]
    fn new(k: i64, n: i64) -> PyResult<Self> {
        validate_frame_domain("StiefelManifold", k, n)?;
        Ok(Self { k, n })
    }

    /// Set the number of frame columns `k`, rejecting any value that would
    /// leave the `1 <= k <= n` domain for the current ambient dimension `n`.
    #[setter]
    fn set_k(&mut self, k: i64) -> PyResult<()> {
        validate_frame_domain("StiefelManifold", k, self.n)?;
        self.k = k;
        Ok(())
    }

    /// Set the ambient dimension `n`, rejecting any value that would leave the
    /// `1 <= k <= n` domain for the current frame-column count `k`.
    #[setter]
    fn set_n(&mut self, n: i64) -> PyResult<()> {
        validate_frame_domain("StiefelManifold", self.k, n)?;
        self.n = n;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("StiefelManifold(k={}, n={})", self.k, self.n)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("kind", "stiefel")?;
        out.set_item("k", self.k)?;
        out.set_item("n", self.n)?;
        Ok(out.into_any().unbind())
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "SpdManifold", skip_from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SpdManifold {
    #[pyo3(get, set)]
    n: i64,
}

#[pymethods]
impl SpdManifold {
    #[new]
    fn new(n: i64) -> Self {
        Self { n }
    }

    fn __repr__(&self) -> String {
        format!("SpdManifold(n={})", self.n)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        out.set_item("kind", "spd")?;
        out.set_item("n", self.n)?;
        Ok(out.into_any().unbind())
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "ProductManifold")]
pub(crate) struct ProductManifold {
    #[pyo3(get, set)]
    parts: Vec<PyObject>,
}

#[pymethods]
impl ProductManifold {
    #[new]
    #[pyo3(signature = (*parts))]
    fn new(_py: Python<'_>, parts: &Bound<'_, PyTuple>) -> Self {
        Self {
            parts: parts.iter().map(|part| part.clone().unbind()).collect(),
        }
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let mut reprs = Vec::with_capacity(self.parts.len());
        for part in &self.parts {
            reprs.push(part.bind(py).repr()?.to_str()?.to_owned());
        }
        let tuple_repr = match reprs.as_slice() {
            [] => "()".to_owned(),
            [only] => format!("({},)", only),
            many => format!("({})", many.join(", ")),
        };
        Ok(format!("ProductManifold(parts={})", tuple_repr))
    }

    fn __eq__(&self, other: &Self, py: Python<'_>) -> PyResult<bool> {
        if self.parts.len() != other.parts.len() {
            return Ok(false);
        }
        for (left, right) in self.parts.iter().zip(other.parts.iter()) {
            if !left.bind(py).eq(right.bind(py))? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<PyObject> {
        let out = PyDict::new(py);
        let parts = PyList::empty(py);
        for part in &self.parts {
            let part_bound = part.bind(py);
            if part_bound.hasattr("to_json")? {
                parts.append(part_bound.getattr("to_json")?.call0()?)?;
            } else {
                parts.append(part_bound)?;
            }
        }
        out.set_item("kind", "product")?;
        out.set_item("parts", parts)?;
        Ok(out.into_any().unbind())
    }
}

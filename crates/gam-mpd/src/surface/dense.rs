//! Dense float64 decompositions on faer ([`crate::dense`]) on
//! the wire: `eigh`, `eigvalsh`, `svd`, `svdvals`, `qr`, `solve`, `lstsq`,
//! `spectral_norm`.

use std::collections::BTreeMap;

use gam_linalg::roundoff::SymmetricAssembly;
use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array2, ArrayD, ArrayView2, Axis, Ix1, Ix2};
use serde::{Deserialize, Serialize};

use super::{MpdOutput, MpdResult, MpdSurfaceError, finite, input, matrix, output, reserve};
use crate::dense::{
    LstsqCutoff, QrMode, eigh, eigvalsh, lstsq, qr, solve, spectral_norm, svd, svdvals,
};

/// How a symmetric input was built, which fixes how far its two triangles may
/// disagree before `eigh` refuses it ([`SymmetricAssembly`]).
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AssemblyRequest {
    /// Both triangles hold the same rounded values (an explicit `(M + Mᵀ)/2`, a
    /// mirrored Gram, a structurally symmetric construction): band zero.
    Mirrored {},
    /// Each triangle is its own accumulation of PSD pieces with at most `depth`
    /// rounded operations per entry.
    PsdAccumulation { depth: usize },
}

impl From<AssemblyRequest> for SymmetricAssembly {
    fn from(request: AssemblyRequest) -> Self {
        match request {
            AssemblyRequest::Mirrored {} => Self::Mirrored,
            AssemblyRequest::PsdAccumulation { depth } => Self::PsdAccumulation { depth },
        }
    }
}

/// What `qr` returns.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QrModeRequest {
    Economic,
    Full,
    R,
}

/// `lstsq`'s singular-value cutoff.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CutoffRequest {
    /// `max(m, n) ε σ₁` (numpy's `rcond=None`).
    Band {},
    /// `rcond σ₁`.
    Relative { rcond: f64 },
}

/// One dense decomposition. Every string is the id of an input array.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DenseRequest {
    /// Eigenpairs of a symmetric matrix, increasing; `indices` `[start, end)` picks a
    /// range of them, or null for all.
    Eigh {
        matrix: String,
        assembly: AssemblyRequest,
        indices: Option<[usize; 2]>,
    },
    Eigvalsh { matrix: String, assembly: AssemblyRequest },
    Svd { matrix: String, full: bool },
    Svdvals { matrix: String },
    Qr { matrix: String, mode: QrModeRequest },
    /// `A⁻¹ B`; `rhs` is a vector or a matrix, and the solution has its shape.
    Solve { matrix: String, rhs: String },
    /// The minimum-norm least-squares solution; `rhs` as for `solve`.
    Lstsq {
        matrix: String,
        rhs: String,
        cutoff: CutoffRequest,
    },
    SpectralNorm { matrix: String },
}

/// One dense decomposition, as the `dense` operation carries it.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DenseOperation {
    pub decomposition: DenseRequest,
}

/// The `dense` operation's result.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DenseResult {
    pub decomposition: DenseReport,
}

/// The arrays each decomposition names, and its scalars.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DenseReport {
    /// `values` (increasing) and `vectors` (columns).
    Eigh { values: String, vectors: String, band: f64 },
    Eigvalsh { values: String, band: f64 },
    /// `u`, `s` (decreasing), `vt`.
    Svd { u: String, s: String, vt: String, band: f64 },
    Svdvals { s: String, band: f64 },
    /// `q` (absent in mode `r`) and `r`, `diag(r) ≥ 0`.
    Qr { q: Option<String>, r: String },
    Solve { x: String },
    Lstsq {
        x: String,
        rank: usize,
        s: String,
        cutoff: f64,
        /// `‖A x_j − b_j‖²` per right-hand side.
        residuals: String,
    },
    SpectralNorm { norm: f64, band: f64 },
}

/// A right-hand side as a matrix, and whether it was a vector.
fn rhs<'a>(
    tensors: &'a BTreeMap<String, ArrayD<f64>>,
    id: &str,
) -> Result<(ArrayView2<'a, f64>, bool), MpdSurfaceError> {
    let array = input(tensors, id)?;
    match array.ndim() {
        1 => Ok((
            array
                .view()
                .into_dimensionality::<Ix1>()
                .map_err(|error| MpdSurfaceError::TensorShape {
                    tensor: id.to_string(),
                    reason: error.to_string(),
                })?
                .insert_axis(Axis(1)),
            true,
        )),
        _ => Ok((
            array
                .view()
                .into_dimensionality::<Ix2>()
                .map_err(|error| MpdSurfaceError::TensorShape {
                    tensor: id.to_string(),
                    reason: format!("expected a vector or a matrix, got shape {:?}: {error}", array.shape()),
                })?,
            false,
        )),
    }
}

fn shaped(solution: Array2<f64>, vector: bool) -> ArrayD<f64> {
    if vector {
        solution.index_axis(Axis(1), 0).to_owned().into_dyn()
    } else {
        solution.into_dyn()
    }
}

pub(super) fn run(
    request: DenseOperation,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    let dense = MpdSurfaceError::Dense;
    let request = request.decomposition;
    let id = match &request {
        DenseRequest::Eigh { matrix, .. }
        | DenseRequest::Eigvalsh { matrix, .. }
        | DenseRequest::Svd { matrix, .. }
        | DenseRequest::Svdvals { matrix }
        | DenseRequest::Qr { matrix, .. }
        | DenseRequest::Solve { matrix, .. }
        | DenseRequest::Lstsq { matrix, .. }
        | DenseRequest::SpectralNorm { matrix } => matrix.clone(),
    };
    let a = matrix(tensors, &id)?;
    let (rows, cols) = a.dim();
    let side = rows.max(cols);
    // The decomposition's working copy, its factors (up to `max(m, n)` square) and a
    // right-hand side's solution.
    let working = reserve(governor, side.max(1), side.max(1), 4, "dense decomposition")?;
    let mut arrays = BTreeMap::new();
    let mut put = |id: &str, array: ArrayD<f64>| {
        arrays.insert(id.to_string(), array);
        id.to_string()
    };
    let report = match request {
        DenseRequest::Eigh { assembly, indices, .. } => {
            let decomposed = eigh(a, assembly.into(), indices.map(|[start, end]| (start, end))).map_err(dense)?;
            DenseReport::Eigh {
                values: put("values", decomposed.values.into_dyn()),
                vectors: put("vectors", decomposed.vectors.into_dyn()),
                band: finite("band", decomposed.band)?,
            }
        }
        DenseRequest::Eigvalsh { assembly, .. } => {
            let (values, band) = eigvalsh(a, assembly.into()).map_err(dense)?;
            DenseReport::Eigvalsh {
                values: put("values", values.into_dyn()),
                band: finite("band", band)?,
            }
        }
        DenseRequest::Svd { full, .. } => {
            let decomposed = svd(a, full).map_err(dense)?;
            DenseReport::Svd {
                u: put("u", decomposed.u.into_dyn()),
                s: put("s", decomposed.singular_values.into_dyn()),
                vt: put("vt", decomposed.vt.into_dyn()),
                band: finite("band", decomposed.band)?,
            }
        }
        DenseRequest::Svdvals { .. } => {
            let (values, band) = svdvals(a).map_err(dense)?;
            DenseReport::Svdvals {
                s: put("s", values.into_dyn()),
                band: finite("band", band)?,
            }
        }
        DenseRequest::Qr { mode, .. } => {
            let mode = match mode {
                QrModeRequest::Economic => QrMode::Economic,
                QrModeRequest::Full => QrMode::Full,
                QrModeRequest::R => QrMode::R,
            };
            let decomposed = qr(a, mode).map_err(dense)?;
            DenseReport::Qr {
                q: decomposed.q.map(|q| put("q", q.into_dyn())),
                r: put("r", decomposed.r.into_dyn()),
            }
        }
        DenseRequest::Solve { rhs: rhs_id, .. } => {
            let (b, vector) = rhs(tensors, &rhs_id)?;
            let solution = solve(a, b).map_err(dense)?;
            DenseReport::Solve {
                x: put("x", shaped(solution, vector)),
            }
        }
        DenseRequest::Lstsq { rhs: rhs_id, cutoff, .. } => {
            let (b, vector) = rhs(tensors, &rhs_id)?;
            let cutoff = match cutoff {
                CutoffRequest::Band {} => LstsqCutoff::Band,
                CutoffRequest::Relative { rcond } => LstsqCutoff::Relative(rcond),
            };
            let solved = lstsq(a, b, cutoff).map_err(dense)?;
            DenseReport::Lstsq {
                x: put("x", shaped(solved.solution, vector)),
                rank: solved.rank,
                s: put("s", solved.singular_values.into_dyn()),
                cutoff: finite("cutoff", solved.cutoff)?,
                residuals: put("residuals", solved.residual_sum_squares.into_dyn()),
            }
        }
        DenseRequest::SpectralNorm { .. } => {
            let (norm, band) = spectral_norm(a).map_err(dense)?;
            DenseReport::SpectralNorm {
                norm: finite("norm", norm)?,
                band: finite("band", band)?,
            }
        }
    };
    drop(working);
    Ok(output(MpdResult::Dense(DenseResult { decomposition: report }), arrays))
}

#[cfg(test)]
mod tests {
    use super::super::run_parameter_decomposition;
    use super::super::tests::request_json;
    use super::*;
    use crate::dense::{DenseError, Eigh, Lstsq, Qr, Svd};
    use crate::test_support::test_governor;
    use ndarray::array;

    fn run(operation: &str, tensors: &BTreeMap<String, ArrayD<f64>>) -> Result<MpdOutput, MpdSurfaceError> {
        run_parameter_decomposition(
            &request_json(&format!(r#"{{"kind": "dense", "decomposition": {operation}}}"#)),
            tensors,
            test_governor(),
        )
    }

    fn report(output: &MpdOutput) -> &DenseReport {
        let MpdResult::Dense(result) = &output.report.result else {
            panic!("expected a dense report, got {:?}", output.report.result);
        };
        &result.decomposition
    }

    fn tensors() -> BTreeMap<String, ArrayD<f64>> {
        BTreeMap::from([
            ("sym".to_string(), array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]].into_dyn()),
            ("tall".to_string(), array![[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]].into_dyn()),
            ("square".to_string(), array![[2.0, 1.0], [1.0, 3.0]].into_dyn()),
            ("b".to_string(), array![3.0, 5.0].into_dyn()),
            ("y".to_string(), array![1.0, 2.0, 2.0].into_dyn()),
        ])
    }

    #[test]
    fn dense_reports_are_the_owner_results() {
        let tensors = tensors();
        let m = |id: &str| matrix(&tensors, id).expect("matrix");

        let output = run(r#"{"kind": "eigh", "matrix": "sym", "assembly": {"kind": "mirrored"}, "indices": null}"#, &tensors).expect("eigh");
        let Eigh { values, vectors, band } = eigh(m("sym"), SymmetricAssembly::Mirrored, None).expect("owner");
        assert_eq!(report(&output), &DenseReport::Eigh { values: "values".into(), vectors: "vectors".into(), band });
        assert_eq!((&output.arrays["values"], &output.arrays["vectors"]), (&values.into_dyn(), &vectors.into_dyn()));

        let output = run(r#"{"kind": "eigh", "matrix": "sym", "assembly": {"kind": "psd_accumulation", "depth": 8}, "indices": [2, 3]}"#, &tensors).expect("eigh top");
        assert_eq!(output.arrays["values"], eigh(m("sym"), SymmetricAssembly::PsdAccumulation { depth: 8 }, Some((2, 3))).expect("owner").values.into_dyn());

        let output = run(r#"{"kind": "eigvalsh", "matrix": "sym", "assembly": {"kind": "mirrored"}}"#, &tensors).expect("eigvalsh");
        assert_eq!(output.arrays["values"], eigvalsh(m("sym"), SymmetricAssembly::Mirrored).expect("owner").0.into_dyn());

        for full in [false, true] {
            let output = run(&format!(r#"{{"kind": "svd", "matrix": "tall", "full": {full}}}"#), &tensors).expect("svd");
            let Svd { u, singular_values, vt, band } = svd(m("tall"), full).expect("owner");
            assert_eq!(report(&output), &DenseReport::Svd { u: "u".into(), s: "s".into(), vt: "vt".into(), band });
            assert_eq!(output.arrays["u"], u.into_dyn());
            assert_eq!(output.arrays["s"], singular_values.into_dyn());
            assert_eq!(output.arrays["vt"], vt.into_dyn());
        }
        let output = run(r#"{"kind": "svdvals", "matrix": "tall"}"#, &tensors).expect("svdvals");
        assert_eq!(output.arrays["s"], svdvals(m("tall")).expect("owner").0.into_dyn());

        for (mode, owner_mode) in [("economic", QrMode::Economic), ("full", QrMode::Full), ("r", QrMode::R)] {
            let output = run(&format!(r#"{{"kind": "qr", "matrix": "tall", "mode": "{mode}"}}"#), &tensors).expect("qr");
            let Qr { q, r } = qr(m("tall"), owner_mode).expect("owner");
            assert_eq!(output.arrays["r"], r.into_dyn());
            assert_eq!(output.arrays.get("q").cloned(), q.map(|q| q.into_dyn()));
        }

        let output = run(r#"{"kind": "solve", "matrix": "square", "rhs": "b"}"#, &tensors).expect("solve");
        let b = tensors["b"].view().into_dimensionality::<Ix1>().expect("b").insert_axis(Axis(1)).to_owned();
        let x = solve(m("square"), b.view()).expect("owner");
        assert_eq!(output.arrays["x"], x.index_axis(Axis(1), 0).to_owned().into_dyn());
        assert_eq!(output.arrays["x"].ndim(), 1);

        let output = run(r#"{"kind": "lstsq", "matrix": "tall", "rhs": "y", "cutoff": {"kind": "band"}}"#, &tensors).expect("lstsq");
        let y = tensors["y"].view().into_dimensionality::<Ix1>().expect("y").insert_axis(Axis(1)).to_owned();
        let Lstsq { solution, rank, singular_values, cutoff, residual_sum_squares } = lstsq(m("tall"), y.view(), LstsqCutoff::Band).expect("owner");
        assert_eq!(report(&output), &DenseReport::Lstsq { x: "x".into(), rank, s: "s".into(), cutoff, residuals: "residuals".into() });
        assert_eq!(output.arrays["x"], solution.index_axis(Axis(1), 0).to_owned().into_dyn());
        assert_eq!(output.arrays["s"], singular_values.into_dyn());
        assert_eq!(output.arrays["residuals"], residual_sum_squares.into_dyn());

        let output = run(r#"{"kind": "spectral_norm", "matrix": "tall"}"#, &tensors).expect("norm");
        let (norm, band) = spectral_norm(m("tall")).expect("owner");
        assert_eq!(report(&output), &DenseReport::SpectralNorm { norm, band });
        let json: serde_json::Value = serde_json::from_str(&output.report_json().expect("json")).expect("parse");
        assert_eq!(json["result"]["kind"], "dense");
        assert_eq!(json["result"]["decomposition"]["kind"], "spectral_norm");
    }

    #[test]
    fn a_dense_request_the_owner_refuses_reaches_the_caller() {
        let mut tensors = tensors();
        assert!(run(r#"{"kind": "svdvals", "matrix": "tall"}"#, &tensors).is_ok());
        assert!(matches!(
            run(r#"{"kind": "eigvalsh", "matrix": "tall", "assembly": {"kind": "mirrored"}}"#, &tensors),
            Err(MpdSurfaceError::Dense(DenseError::NotSquare { .. }))
        ));
        tensors.insert("lopsided".to_string(), array![[2.0, 99.0], [1.0, 2.0]].into_dyn());
        assert!(matches!(
            run(r#"{"kind": "eigh", "matrix": "lopsided", "assembly": {"kind": "mirrored"}, "indices": null}"#, &tensors),
            Err(MpdSurfaceError::Dense(DenseError::Asymmetric { .. }))
        ));
        tensors.insert("singular".to_string(), array![[1.0, 2.0], [2.0, 4.0]].into_dyn());
        assert!(matches!(
            run(r#"{"kind": "solve", "matrix": "singular", "rhs": "b"}"#, &tensors),
            Err(MpdSurfaceError::Dense(DenseError::Singular { .. }))
        ));
        assert!(matches!(
            run(r#"{"kind": "lstsq", "matrix": "tall", "rhs": "b", "cutoff": {"kind": "band"}}"#, &tensors),
            Err(MpdSurfaceError::Dense(DenseError::Shape { .. }))
        ));
        assert!(matches!(
            run(r#"{"kind": "eigh", "matrix": "sym", "assembly": {"kind": "mirrored"}, "indices": [2, 9]}"#, &tensors),
            Err(MpdSurfaceError::Dense(DenseError::InvalidRange { .. }))
        ));
        assert!(matches!(
            run(r#"{"kind": "qr", "matrix": "tall", "mode": "raw"}"#, &tensors),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
        assert!(matches!(
            run(r#"{"kind": "svd", "matrix": "tall", "full": false, "compute_uv": true}"#, &tensors),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
    }
}

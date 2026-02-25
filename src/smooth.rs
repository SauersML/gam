use crate::basis::{
    BSplineBasisSpec, BasisBuildResult, BasisError, BasisMetadata, DuchonBasisSpec,
    MaternBasisSpec, ThinPlateBasisSpec, build_bspline_basis_1d, build_duchon_basis,
    build_matern_basis, build_thin_plate_basis,
};
use crate::estimate::{EstimationError, FitOptions, FitResult, fit_gam};
use crate::types::LikelihoodFamily;
use ndarray::{Array1, Array2, ArrayView2, s};
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeConstraint {
    None,
    MonotoneIncreasing,
    MonotoneDecreasing,
    Convex,
    Concave,
}

#[derive(Debug, Clone)]
pub enum SmoothBasisSpec {
    BSpline1D {
        feature_col: usize,
        spec: BSplineBasisSpec,
    },
    ThinPlate {
        feature_cols: Vec<usize>,
        spec: ThinPlateBasisSpec,
    },
    Matern {
        feature_cols: Vec<usize>,
        spec: MaternBasisSpec,
    },
    Duchon {
        feature_cols: Vec<usize>,
        spec: DuchonBasisSpec,
    },
}

#[derive(Debug, Clone)]
pub struct SmoothTermSpec {
    pub name: String,
    pub basis: SmoothBasisSpec,
    pub shape: ShapeConstraint,
}

#[derive(Debug, Clone)]
pub struct SmoothTerm {
    pub name: String,
    pub coeff_range: Range<usize>,
    pub shape: ShapeConstraint,
    pub penalties_local: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
    pub metadata: BasisMetadata,
}

#[derive(Debug, Clone)]
pub struct SmoothDesign {
    pub design: Array2<f64>,
    pub penalties: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
    pub terms: Vec<SmoothTerm>,
}

#[derive(Debug, Clone)]
pub struct LinearTermSpec {
    pub name: String,
    pub feature_col: usize,
    /// Optional double-penalty ridge on this linear coefficient.
    /// If true, emits an identity penalty block for this 1D term.
    pub double_penalty: bool,
}

#[derive(Debug, Clone)]
pub struct TermCollectionSpec {
    pub linear_terms: Vec<LinearTermSpec>,
    pub smooth_terms: Vec<SmoothTermSpec>,
}

#[derive(Debug, Clone)]
pub struct TermCollectionDesign {
    pub design: Array2<f64>,
    pub penalties: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
    pub linear_ranges: Vec<(String, Range<usize>)>,
    pub smooth: SmoothDesign,
}

pub struct FittedTermCollection {
    pub fit: FitResult,
    pub design: TermCollectionDesign,
}

fn select_columns(data: ArrayView2<'_, f64>, cols: &[usize]) -> Result<Array2<f64>, BasisError> {
    let n = data.nrows();
    let p = data.ncols();
    for &c in cols {
        if c >= p {
            return Err(BasisError::DimensionMismatch(format!(
                "feature column {c} is out of bounds for data with {p} columns"
            )));
        }
    }
    let mut out = Array2::<f64>::zeros((n, cols.len()));
    for (j, &c) in cols.iter().enumerate() {
        out.column_mut(j).assign(&data.column(c));
    }
    Ok(out)
}

fn cumulative_exp(values: &Array1<f64>, sign: f64) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(values.len());
    let mut run = 0.0;
    for i in 0..values.len() {
        run += values[i].exp();
        out[i] = sign * run;
    }
    out
}

fn second_cumulative_exp(values: &Array1<f64>, sign: f64) -> Array1<f64> {
    let first = cumulative_exp(values, sign);
    let mut out = Array1::<f64>::zeros(values.len());
    let mut run = 0.0;
    for i in 0..values.len() {
        run += first[i];
        out[i] = run;
    }
    out
}

impl SmoothDesign {
    /// Map an unconstrained term coefficient vector to its constrained shape space.
    /// This is useful for nonlinear fits that optimize unconstrained parameters.
    pub fn map_term_coefficients(
        unconstrained: &Array1<f64>,
        shape: ShapeConstraint,
    ) -> Result<Array1<f64>, BasisError> {
        if unconstrained.is_empty() {
            return Err(BasisError::InvalidInput(
                "unconstrained coefficient vector cannot be empty".to_string(),
            ));
        }
        let mapped = match shape {
            ShapeConstraint::None => unconstrained.clone(),
            ShapeConstraint::MonotoneIncreasing => cumulative_exp(unconstrained, 1.0),
            ShapeConstraint::MonotoneDecreasing => cumulative_exp(unconstrained, -1.0),
            ShapeConstraint::Convex => second_cumulative_exp(unconstrained, 1.0),
            ShapeConstraint::Concave => second_cumulative_exp(unconstrained, -1.0),
        };
        Ok(mapped)
    }
}

pub fn build_smooth_design(
    data: ArrayView2<'_, f64>,
    terms: &[SmoothTermSpec],
) -> Result<SmoothDesign, BasisError> {
    let n = data.nrows();
    let mut local_designs = Vec::<Array2<f64>>::with_capacity(terms.len());
    let mut local_penalties = Vec::<Vec<Array2<f64>>>::with_capacity(terms.len());
    let mut local_nullspaces = Vec::<Vec<usize>>::with_capacity(terms.len());
    let mut local_metadata = Vec::<BasisMetadata>::with_capacity(terms.len());
    let mut local_dims = Vec::<usize>::with_capacity(terms.len());

    for term in terms {
        if term.shape != ShapeConstraint::None {
            return Err(BasisError::InvalidInput(format!(
                "ShapeConstraint::{:?} is not enforced in the current linear fit pipeline; use ShapeConstraint::None or a constrained/nonlinear solver",
                term.shape
            )));
        }
        let built: BasisBuildResult = match &term.basis {
            SmoothBasisSpec::BSpline1D { feature_col, spec } => {
                if *feature_col >= data.ncols() {
                    return Err(BasisError::DimensionMismatch(format!(
                        "term '{}' feature column {} out of bounds for {} columns",
                        term.name,
                        feature_col,
                        data.ncols()
                    )));
                }
                build_bspline_basis_1d(data.column(*feature_col), spec)?
            }
            SmoothBasisSpec::ThinPlate { feature_cols, spec } => {
                let x = select_columns(data, feature_cols)?;
                build_thin_plate_basis(x.view(), spec)?
            }
            SmoothBasisSpec::Matern { feature_cols, spec } => {
                let x = select_columns(data, feature_cols)?;
                build_matern_basis(x.view(), spec)?
            }
            SmoothBasisSpec::Duchon { feature_cols, spec } => {
                let x = select_columns(data, feature_cols)?;
                build_duchon_basis(x.view(), spec)?
            }
        };

        let p_local = built.design.ncols();
        let design_t = built.design;
        let penalties_t: Vec<Array2<f64>> = built.penalties;

        local_dims.push(p_local);
        local_designs.push(design_t);
        local_penalties.push(penalties_t);
        local_nullspaces.push(built.nullspace_dims);
        local_metadata.push(built.metadata);
    }

    let total_p: usize = local_dims.iter().sum();
    let mut design = Array2::<f64>::zeros((n, total_p));
    let mut terms_out = Vec::<SmoothTerm>::with_capacity(terms.len());
    let mut penalties_global = Vec::<Array2<f64>>::new();
    let mut nullspace_dims_global = Vec::<usize>::new();

    let mut col_start = 0usize;
    for (idx, term) in terms.iter().enumerate() {
        let p_local = local_dims[idx];
        let col_end = col_start + p_local;

        design
            .slice_mut(s![.., col_start..col_end])
            .assign(&local_designs[idx]);

        for (s_local, &ns) in local_penalties[idx]
            .iter()
            .zip(local_nullspaces[idx].iter())
        {
            let mut s_global = Array2::<f64>::zeros((total_p, total_p));
            s_global
                .slice_mut(s![col_start..col_end, col_start..col_end])
                .assign(s_local);
            penalties_global.push(s_global);
            nullspace_dims_global.push(ns);
        }

        terms_out.push(SmoothTerm {
            name: term.name.clone(),
            coeff_range: col_start..col_end,
            shape: term.shape,
            penalties_local: local_penalties[idx].clone(),
            nullspace_dims: local_nullspaces[idx].clone(),
            metadata: local_metadata[idx].clone(),
        });

        col_start = col_end;
    }

    Ok(SmoothDesign {
        design,
        penalties: penalties_global,
        nullspace_dims: nullspace_dims_global,
        terms: terms_out,
    })
}

pub fn build_term_collection_design(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
) -> Result<TermCollectionDesign, BasisError> {
    let n = data.nrows();
    let p_data = data.ncols();
    let smooth = build_smooth_design(data, &spec.smooth_terms)?;

    for linear in &spec.linear_terms {
        if linear.feature_col >= p_data {
            return Err(BasisError::DimensionMismatch(format!(
                "linear term '{}' feature column {} out of bounds for {} columns",
                linear.name, linear.feature_col, p_data
            )));
        }
    }

    let p_lin = spec.linear_terms.len();
    let p_smooth = smooth.design.ncols();
    let p_total = p_lin + p_smooth;
    let mut design = Array2::<f64>::zeros((n, p_total));

    let mut linear_ranges = Vec::<(String, Range<usize>)>::with_capacity(p_lin);
    for (j, linear) in spec.linear_terms.iter().enumerate() {
        design
            .column_mut(j)
            .assign(&data.column(linear.feature_col));
        linear_ranges.push((linear.name.clone(), j..(j + 1)));
    }
    if p_smooth > 0 {
        design.slice_mut(s![.., p_lin..]).assign(&smooth.design);
    }

    let mut penalties = Vec::<Array2<f64>>::new();
    let mut nullspace_dims = Vec::<usize>::new();

    for (j, linear) in spec.linear_terms.iter().enumerate() {
        if !linear.double_penalty {
            continue;
        }
        let mut s = Array2::<f64>::zeros((p_total, p_total));
        s[[j, j]] = 1.0;
        penalties.push(s);
        nullspace_dims.push(0);
    }

    for (s_local, &ns) in smooth.penalties.iter().zip(smooth.nullspace_dims.iter()) {
        let mut s = Array2::<f64>::zeros((p_total, p_total));
        let start = p_lin;
        let end = p_lin + p_smooth;
        s.slice_mut(s![start..end, start..end]).assign(s_local);
        penalties.push(s);
        nullspace_dims.push(ns);
    }

    Ok(TermCollectionDesign {
        design,
        penalties,
        nullspace_dims,
        linear_ranges,
        smooth,
    })
}

pub fn fit_term_collection(
    data: ArrayView2<'_, f64>,
    y: Array1<f64>,
    weights: Array1<f64>,
    offset: Array1<f64>,
    spec: &TermCollectionSpec,
    family: LikelihoodFamily,
    options: &FitOptions,
) -> Result<FittedTermCollection, EstimationError> {
    let design = build_term_collection_design(data, spec)?;
    let fit = fit_gam(
        design.design.view(),
        y.view(),
        weights.view(),
        offset.view(),
        &design.penalties,
        family,
        &FitOptions {
            max_iter: options.max_iter,
            tol: options.tol,
            nullspace_dims: design.nullspace_dims.clone(),
        },
    )?;
    Ok(FittedTermCollection { fit, design })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{
        BSplineKnotSpec, CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, MaternBasisSpec,
        MaternNu, ThinPlateBasisSpec,
    };
    use ndarray::array;

    #[test]
    fn smooth_design_assembles_terms_and_penalties() {
        let data = array![
            [0.0, 0.0, 0.2],
            [0.2, 0.1, 0.4],
            [0.4, 0.2, 0.6],
            [0.6, 0.4, 0.7],
            [0.8, 0.7, 0.9],
            [1.0, 1.0, 1.1]
        ];

        let terms = vec![
            SmoothTermSpec {
                name: "s_x0".to_string(),
                basis: SmoothBasisSpec::BSpline1D {
                    feature_col: 0,
                    spec: BSplineBasisSpec {
                        degree: 3,
                        penalty_order: 2,
                        knot_spec: BSplineKnotSpec::Generate {
                            data_range: (0.0, 1.0),
                            num_internal_knots: 4,
                        },
                        double_penalty: true,
                    },
                },
                shape: ShapeConstraint::None,
            },
            SmoothTermSpec {
                name: "tps_x1x2".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![1, 2],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        double_penalty: true,
                    },
                },
                shape: ShapeConstraint::None,
            },
        ];

        let sd = build_smooth_design(data.view(), &terms).unwrap();
        assert_eq!(sd.design.nrows(), data.nrows());
        assert_eq!(sd.terms.len(), 2);
        assert_eq!(sd.penalties.len(), 4);
        assert_eq!(sd.nullspace_dims.len(), 4);
        for s in &sd.penalties {
            assert_eq!(s.nrows(), sd.design.ncols());
            assert_eq!(s.ncols(), sd.design.ncols());
        }
    }

    #[test]
    fn shape_mapping_monotone_increasing_is_non_decreasing() {
        let theta = array![-1.0, 0.5, -0.2, 0.3];
        let beta = SmoothDesign::map_term_coefficients(&theta, ShapeConstraint::MonotoneIncreasing)
            .unwrap();
        for i in 1..beta.len() {
            assert!(beta[i] >= beta[i - 1]);
        }
    }

    #[test]
    fn build_smooth_design_rejects_non_none_shape_constraints() {
        let data = array![
            [0.0, 0.0],
            [0.5, 0.2],
            [1.0, 0.4],
            [1.5, 0.6],
        ];
        let terms = vec![SmoothTermSpec {
            name: "tps_shape".to_string(),
            basis: SmoothBasisSpec::ThinPlate {
                feature_cols: vec![0, 1],
                spec: ThinPlateBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 3 },
                    double_penalty: false,
                },
            },
            shape: ShapeConstraint::MonotoneIncreasing,
        }];

        let err = build_smooth_design(data.view(), &terms).expect_err("shape should be rejected");
        match err {
            BasisError::InvalidInput(msg) => {
                assert!(msg.contains("not enforced"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn term_collection_design_combines_linear_and_smooth() {
        let data = array![
            [0.0, 0.0, 0.2],
            [0.2, 0.1, 0.4],
            [0.4, 0.2, 0.6],
            [0.6, 0.4, 0.7],
            [0.8, 0.7, 0.9],
            [1.0, 1.0, 1.1]
        ];
        let spec = TermCollectionSpec {
            linear_terms: vec![LinearTermSpec {
                name: "lin_x0".to_string(),
                feature_col: 0,
                double_penalty: true,
            }],
            smooth_terms: vec![SmoothTermSpec {
                name: "tps_x1x2".to_string(),
                basis: SmoothBasisSpec::ThinPlate {
                    feature_cols: vec![1, 2],
                    spec: ThinPlateBasisSpec {
                        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
                        double_penalty: true,
                    },
                },
                shape: ShapeConstraint::None,
            }],
        };
        let design = build_term_collection_design(data.view(), &spec).unwrap();
        assert_eq!(design.design.nrows(), data.nrows());
        assert!(design.design.ncols() >= 1);
        assert_eq!(design.linear_ranges.len(), 1);
        assert_eq!(design.penalties.len(), 3); // linear ridge + 2 smooth penalties
        assert_eq!(design.nullspace_dims.len(), 3);
    }

    #[test]
    fn matern_smooth_builds_with_double_penalty_in_high_dim() {
        let n = 12usize;
        let d = 10usize;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i as f64) * 0.1 + (j as f64) * 0.03;
            }
        }

        let terms = vec![SmoothTermSpec {
            name: "matern_x".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: (0..d).collect(),
                spec: MaternBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 5 },
                    length_scale: 0.75,
                    nu: MaternNu::FiveHalves,
                    include_intercept: true,
                    double_penalty: true,
                },
            },
            shape: ShapeConstraint::None,
        }];

        let sd = build_smooth_design(data.view(), &terms).unwrap();
        assert_eq!(sd.design.nrows(), n);
        assert_eq!(sd.terms.len(), 1);
        // kernel + ridge penalties
        assert_eq!(sd.penalties.len(), 2);
        assert_eq!(sd.nullspace_dims.len(), 2);
        // first penalty keeps 1 "almost null" intercept dimension
        assert_eq!(sd.nullspace_dims[0], 1);
        assert_eq!(sd.nullspace_dims[1], 0);
    }

    #[test]
    fn duchon_linear_nullspace_builds_and_reports_nullspace_dim() {
        let n = 14usize;
        let d = 10usize;
        let mut data = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                data[[i, j]] = (i as f64) * 0.07 + (j as f64) * 0.05;
            }
        }

        let terms = vec![SmoothTermSpec {
            name: "duchon_x".to_string(),
            basis: SmoothBasisSpec::Duchon {
                feature_cols: (0..d).collect(),
                spec: DuchonBasisSpec {
                    center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
                    length_scale: 0.9,
                    nu: MaternNu::FiveHalves,
                    nullspace_order: DuchonNullspaceOrder::Linear,
                    double_penalty: true,
                },
            },
            shape: ShapeConstraint::None,
        }];

        let sd = build_smooth_design(data.view(), &terms).unwrap();
        assert_eq!(sd.design.nrows(), n);
        assert_eq!(sd.terms.len(), 1);
        assert_eq!(sd.penalties.len(), 2);
        assert_eq!(sd.nullspace_dims.len(), 2);
        // Linear null space in d dimensions -> d+1 free polynomial terms
        assert_eq!(sd.nullspace_dims[0], d + 1);
        assert_eq!(sd.nullspace_dims[1], 0);
    }
}

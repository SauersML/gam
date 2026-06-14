//! Design-matrix *assembly* and identifiability: term-collection design
//! construction, tensor/factor/PCA/random-effect block builders, sparse design
//! assembly, smooth-ownership analysis, and the identifiability/orthogonality
//! transforms that make overlapping smooths estimable.

use super::*;

pub struct SmoothDesign {
    pub term_designs: Vec<DesignMatrix>,
    /// Per-term block-local penalties.  Each `col_range` is relative to the
    /// smooth block (i.e. indexing into the concatenation of `term_designs`).
    pub penalties: Vec<BlockwisePenalty>,
    pub nullspace_dims: Vec<usize>,
    pub penaltyinfo: Vec<PenaltyBlockInfo>,
    pub dropped_penaltyinfo: Vec<DroppedPenaltyBlockInfo>,
    pub terms: Vec<SmoothTerm>,
    /// Optional smooth-block lower bounds in smooth coefficient coordinates.
    /// Length equals `total_smooth_cols()` when present.
    pub coefficient_lower_bounds: Option<Array1<f64>>,
    /// Optional smooth-block inequality constraints:
    /// `A_smooth * beta_smooth >= b`.
    pub linear_constraints: Option<LinearInequalityConstraints>,
}


impl SmoothDesign {
    pub fn total_smooth_cols(&self) -> usize {
        self.term_designs.iter().map(DesignMatrix::ncols).sum()
    }
    pub fn nrows(&self) -> usize {
        self.term_designs.first().map_or(0, DesignMatrix::nrows)
    }
}


#[derive(Debug, Clone)]
pub struct RawSmoothDesign {
    pub term_designs: Vec<DesignMatrix>,
    /// Per-term block-local penalties.  Each `col_range` is relative to the
    /// smooth block (i.e. indexing into the concatenation of `term_designs`).
    pub penalties: Vec<BlockwisePenalty>,
    pub nullspace_dims: Vec<usize>,
    pub penaltyinfo: Vec<PenaltyBlockInfo>,
    pub dropped_penaltyinfo: Vec<DroppedPenaltyBlockInfo>,
    pub terms: Vec<SmoothTerm>,
    pub coefficient_lower_bounds: Option<Array1<f64>>,
    pub linear_constraints: Option<LinearInequalityConstraints>,
}


impl RawSmoothDesign {
    pub fn total_smooth_cols(&self) -> usize {
        self.term_designs.iter().map(DesignMatrix::ncols).sum()
    }
    pub fn nrows(&self) -> usize {
        self.term_designs.first().map_or(0, DesignMatrix::nrows)
    }
}


impl From<RawSmoothDesign> for SmoothDesign {
    fn from(value: RawSmoothDesign) -> Self {
        Self {
            term_designs: value.term_designs,
            penalties: value.penalties,
            nullspace_dims: value.nullspace_dims,
            penaltyinfo: value.penaltyinfo,
            dropped_penaltyinfo: value.dropped_penaltyinfo,
            terms: value.terms,
            coefficient_lower_bounds: value.coefficient_lower_bounds,
            linear_constraints: value.linear_constraints,
        }
    }
}


#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TermCollectionDesign {
    /// The full design matrix.
    ///
    /// Prefer a true sparse matrix when every block is sparse-compatible.
    /// If the collection already contains intrinsically sparse blocks, preserve
    /// that storage and let PIRLS decide later whether the penalized system is
    /// sparse-native eligible. Purely dense materialized blocks still fall back
    /// to the lazy block operator when sparse storage would just re-encode a
    /// dense matrix.
    pub design: DesignMatrix,
    pub penalties: Vec<BlockwisePenalty>,
    pub nullspace_dims: Vec<usize>,
    pub penaltyinfo: Vec<PenaltyBlockInfo>,
    pub dropped_penaltyinfo: Vec<DroppedPenaltyBlockInfo>,
    /// Optional global coefficient lower bounds for constrained fitting.
    /// Length equals `design.ncols()` when present. Unconstrained entries are `-inf`.
    pub coefficient_lower_bounds: Option<Array1<f64>>,
    /// Optional global inequality constraints:
    /// `A * beta >= b`.
    pub linear_constraints: Option<LinearInequalityConstraints>,
    pub intercept_range: Range<usize>,
    pub linear_ranges: Vec<(String, Range<usize>)>,
    pub random_effect_ranges: Vec<(String, Range<usize>)>,
    pub random_effect_levels: Vec<(String, Vec<u64>)>,
    pub smooth: SmoothDesign,
}


impl TermCollectionDesign {
    /// Convert blockwise penalties to `PenaltyMatrix::Blockwise` without
    /// expanding to `p_total × p_total`. This is the preferred path for
    /// family modules that accept `Vec<PenaltyMatrix>`.
    pub fn penalties_as_penalty_matrix(&self) -> Vec<crate::custom_family::PenaltyMatrix> {
        let p = self.design.ncols();
        self.penalties
            .iter()
            .map(|bp| bp.to_penalty_matrix(p))
            .collect()
    }

    /// Number of penalty blocks.
    #[inline]
    pub fn num_penalties(&self) -> usize {
        self.penalties.len()
    }

    /// Resolve coefficient groups against this design's global coefficient
    /// layout and append their penalties after the existing term penalties.
    pub fn realize_coefficient_groups(
        &self,
        groups: &[CoefficientGroupSpec],
        base_prior: &crate::types::RhoPrior,
    ) -> Result<RealizedCoefficientGroups, BasisError> {
        realize_coefficient_groups(self, groups, base_prior)
    }

    /// Extract a `KroneckerPenaltySystem` if exactly one smooth term has
    /// Kronecker structure and it accounts for all penalties.
    ///
    /// Returns `None` if no Kronecker structure is present or if the model
    /// has multiple smooth terms with mixed structure.
    pub fn kronecker_penalty_system(&self) -> Option<KroneckerPenaltySystem> {
        let kron_terms: Vec<&KroneckerFactoredBasis> = self
            .smooth
            .terms
            .iter()
            .filter_map(|t| t.kronecker_factored.as_ref())
            .collect();
        if kron_terms.len() != 1 {
            return None; // 0 or multiple Kronecker terms — not supported yet
        }
        let kron = kron_terms[0];
        // Only use the Kronecker path when the model is purely this tensor term
        // (no other smooth terms with separate penalties).
        let has_non_kron_smooth_terms = self
            .smooth
            .terms
            .iter()
            .any(|t| t.kronecker_factored.is_none());
        if has_non_kron_smooth_terms {
            return None; // mixed model — fall back to standard path
        }
        KroneckerPenaltySystem::new(
            kron.marginal_penalties.clone(),
            kron.marginal_dims.clone(),
            kron.has_double_penalty,
        )
        .ok()
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
struct RandomEffectBlock {
    name: String,
    /// O(n) group-label vector: group_ids[i] = column index in [0, num_groups).
    /// `None` if the observation's level is not in the kept set.
    group_ids: Vec<Option<usize>>,
    num_groups: usize,
    kept_levels: Vec<u64>,
}


const BLOCK_SPARSE_ZERO_EPS: f64 = 1e-12;

const BLOCK_SPARSE_MAX_DENSITY: f64 = 0.20;


fn blocks_have_intrinsic_sparse_structure(blocks: &[DesignBlock]) -> bool {
    blocks
        .iter()
        .any(|block| matches!(block, DesignBlock::Sparse(_) | DesignBlock::RandomEffect(_)))
}


fn sparse_compatible_block_nnz(block: &DesignBlock) -> Option<usize> {
    match block {
        DesignBlock::Intercept(n) => Some(*n),
        DesignBlock::RandomEffect(op) => {
            Some(op.group_ids.iter().filter(|gid| gid.is_some()).count())
        }
        DesignBlock::Sparse(sparse) => Some(sparse.val().len()),
        DesignBlock::Dense(dense) => dense.as_dense_ref().map(|matrix| {
            matrix
                .iter()
                .filter(|&&value| value.abs() > BLOCK_SPARSE_ZERO_EPS)
                .count()
        }),
    }
}


fn try_build_sparse_design_from_blocks(
    blocks: &[DesignBlock],
) -> Result<Option<DesignMatrix>, BasisError> {
    if blocks.is_empty() {
        return Ok(None);
    }
    let nrows = blocks[0].nrows();
    let ncols: usize = blocks.iter().map(DesignBlock::ncols).sum();
    if nrows == 0 || ncols == 0 || ncols <= 32 {
        return Ok(None);
    }

    let preserve_sparse_storage = blocks_have_intrinsic_sparse_structure(blocks);
    let sparse_nnz_limit = if preserve_sparse_storage {
        usize::MAX
    } else {
        let total_cells = nrows.saturating_mul(ncols);
        ((total_cells as f64) * BLOCK_SPARSE_MAX_DENSITY).floor() as usize
    };
    let mut nnz = 0usize;
    for block in blocks {
        let block_nnz = if let Some(block_nnz) = sparse_compatible_block_nnz(block) {
            block_nnz
        } else {
            return Ok(None);
        };
        nnz = nnz.saturating_add(block_nnz);
        if nnz > sparse_nnz_limit {
            return Ok(None);
        }
    }

    let mut triplets = Vec::<Triplet<usize, usize, f64>>::with_capacity(nnz);
    let mut col_offset = 0usize;
    for block in blocks {
        match block {
            DesignBlock::Intercept(n) => {
                for row in 0..*n {
                    triplets.push(Triplet::new(row, col_offset, 1.0));
                }
            }
            DesignBlock::RandomEffect(op) => {
                for (row, group_id) in op.group_ids.iter().enumerate() {
                    if let Some(group) = group_id {
                        triplets.push(Triplet::new(row, col_offset + group, 1.0));
                    }
                }
            }
            DesignBlock::Sparse(sparse) => {
                let (symbolic, values) = sparse.parts();
                let col_ptr = symbolic.col_ptr();
                let row_idx = symbolic.row_idx();
                for col in 0..sparse.ncols() {
                    for idx in col_ptr[col]..col_ptr[col + 1] {
                        let value = values[idx];
                        if value.abs() > BLOCK_SPARSE_ZERO_EPS {
                            triplets.push(Triplet::new(row_idx[idx], col_offset + col, value));
                        }
                    }
                }
            }
            DesignBlock::Dense(dense) => {
                let matrix = dense.as_dense_ref().ok_or_else(|| {
                    BasisError::InvalidInput(
                        "sparse-compatible block assembly requires materialized dense blocks"
                            .to_string(),
                    )
                })?;
                for row in 0..matrix.nrows() {
                    for col in 0..matrix.ncols() {
                        let value = matrix[[row, col]];
                        if value.abs() > BLOCK_SPARSE_ZERO_EPS {
                            triplets.push(Triplet::new(row, col_offset + col, value));
                        }
                    }
                }
            }
        }
        col_offset += block.ncols();
    }

    let sparse = SparseColMat::try_new_from_triplets(nrows, ncols, &triplets).map_err(|_| {
        BasisError::SparseCreation("failed to assemble sparse term-collection design".to_string())
    })?;
    Ok(Some(DesignMatrix::Sparse(
        crate::matrix::SparseDesignMatrix::new(sparse),
    )))
}


pub(crate) fn assemble_term_collection_design_matrix(
    blocks: Vec<DesignBlock>,
) -> Result<DesignMatrix, BasisError> {
    if let Some(sparse) = try_build_sparse_design_from_blocks(&blocks)? {
        return Ok(sparse);
    }
    let block_op = BlockDesignOperator::new(blocks).map_err(|e| {
        BasisError::InvalidInput(format!("failed to build block design operator: {e}"))
    })?;
    Ok(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
        Arc::new(block_op),
    )))
}


pub(crate) fn select_columns(data: ArrayView2<'_, f64>, cols: &[usize]) -> Result<Array2<f64>, BasisError> {
    let n = data.nrows();
    let p = data.ncols();
    for &c in cols {
        if c >= p {
            crate::bail_dim_basis!("feature column {c} is out of bounds for data with {p} columns");
        }
    }
    let mut out = Array2::<f64>::zeros((n, cols.len()));
    for (j, &c) in cols.iter().enumerate() {
        out.column_mut(j).assign(&data.column(c));
    }
    Ok(out)
}


fn nonfinite_value_label(value: f64) -> &'static str {
    if value.is_nan() {
        "NaN"
    } else if value.is_sign_positive() {
        "+Inf"
    } else {
        "-Inf"
    }
}


fn validate_term_feature_column_finite(
    data: ArrayView2<'_, f64>,
    term_kind: &str,
    term_name: &str,
    feature_col: usize,
) -> Result<(), BasisError> {
    let p = data.ncols();
    if feature_col >= p {
        crate::bail_dim_basis!(
            "{term_kind} term '{term_name}' feature column {feature_col} out of bounds for {p} columns"
        );
    }
    for (row, &value) in data.column(feature_col).iter().enumerate() {
        if !value.is_finite() {
            crate::bail_invalid_basis!(
                "{term_kind} term '{term_name}' feature column {feature_col} row {row} contains non-finite value {}",
                nonfinite_value_label(value)
            );
        }
    }
    Ok(())
}


fn validate_smooth_terms_finite_inputs(
    data: ArrayView2<'_, f64>,
    terms: &[SmoothTermSpec],
) -> Result<(), BasisError> {
    for term in terms {
        for feature_col in smooth_term_feature_cols(term) {
            validate_term_feature_column_finite(data, "smooth", &term.name, feature_col)?;
        }
    }
    Ok(())
}


fn validate_term_collection_finite_inputs(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
) -> Result<(), BasisError> {
    for term in &spec.linear_terms {
        validate_term_feature_column_finite(data, "linear", &term.name, term.feature_col)?;
    }
    for term in &spec.random_effect_terms {
        validate_term_feature_column_finite(data, "random-effect", &term.name, term.feature_col)?;
    }
    validate_smooth_terms_finite_inputs(data, &spec.smooth_terms)
}


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct JointSpatialCenterGroupKey {
    feature_cols: Vec<usize>,
    strategy_kind: CenterStrategyKind,
    strategy_aux: usize,
    requested_num_centers: usize,
    input_scale_bits: Option<Vec<u64>>,
}


fn spatial_term_min_center_count(term: &SmoothTermSpec) -> usize {
    match &term.basis {
        SmoothBasisSpec::ThinPlate { feature_cols, .. } => feature_cols.len() + 1,
        SmoothBasisSpec::Duchon {
            feature_cols, spec, ..
        } => match spec.nullspace_order {
            crate::basis::DuchonNullspaceOrder::Zero => 1,
            crate::basis::DuchonNullspaceOrder::Linear => feature_cols.len() + 1,
            crate::basis::DuchonNullspaceOrder::Degree(degree) => {
                crate::basis::duchon_nullspace_dimension(feature_cols.len(), degree)
            }
        },
        SmoothBasisSpec::Matern { .. } => 1,
        _ => 1,
    }
}


fn spatial_term_group_key(term: &SmoothTermSpec) -> Option<JointSpatialCenterGroupKey> {
    let (feature_cols, strategy, input_scales) = match &term.basis {
        SmoothBasisSpec::ThinPlate {
            feature_cols,
            spec,
            input_scales,
        } => (feature_cols, &spec.center_strategy, input_scales.as_ref()),
        SmoothBasisSpec::Matern {
            feature_cols,
            spec,
            input_scales,
        } => (feature_cols, &spec.center_strategy, input_scales.as_ref()),
        SmoothBasisSpec::Duchon {
            feature_cols,
            spec,
            input_scales,
        } => (feature_cols, &spec.center_strategy, input_scales.as_ref()),
        _ => return None,
    };
    let strategy_kind = center_strategy_kind(strategy);
    let strategy_aux = match strategy {
        CenterStrategy::Auto(inner) => match inner.as_ref() {
            CenterStrategy::KMeans { max_iter, .. } => *max_iter,
            CenterStrategy::UniformGrid { points_per_dim } => *points_per_dim,
            _ => 0,
        },
        CenterStrategy::KMeans { max_iter, .. } => *max_iter,
        CenterStrategy::UniformGrid { points_per_dim } => *points_per_dim,
        _ => 0,
    };
    Some(JointSpatialCenterGroupKey {
        feature_cols: feature_cols.clone(),
        strategy_kind,
        strategy_aux,
        requested_num_centers: center_strategy_num_centers(strategy)?,
        input_scale_bits: input_scales
            .map(|values| values.iter().map(|value| value.to_bits()).collect()),
    })
}


pub(crate) fn spatial_term_center_strategy(term: &SmoothTermSpec) -> Option<&CenterStrategy> {
    match &term.basis {
        SmoothBasisSpec::ThinPlate { spec, .. } => Some(&spec.center_strategy),
        SmoothBasisSpec::Matern { spec, .. } => Some(&spec.center_strategy),
        SmoothBasisSpec::Duchon { spec, .. } => Some(&spec.center_strategy),
        _ => None,
    }
}


fn set_spatial_term_centers(
    term: &mut SmoothTermSpec,
    centers: Array2<f64>,
) -> Result<(), BasisError> {
    match &mut term.basis {
        SmoothBasisSpec::ThinPlate { spec, .. } => {
            spec.center_strategy = CenterStrategy::UserProvided(centers);
            Ok(())
        }
        SmoothBasisSpec::Matern { spec, .. } => {
            spec.center_strategy = CenterStrategy::UserProvided(centers);
            Ok(())
        }
        SmoothBasisSpec::Duchon { spec, .. } => {
            spec.center_strategy = CenterStrategy::UserProvided(centers);
            Ok(())
        }
        _ => Err(BasisError::InvalidInput(format!(
            "term '{}' does not support spatial center planning",
            term.name
        ))),
    }
}


pub(crate) fn standardized_spatial_term_data(
    data: ArrayView2<'_, f64>,
    term: &SmoothTermSpec,
) -> Result<Array2<f64>, BasisError> {
    let (feature_cols, input_scales) = match &term.basis {
        SmoothBasisSpec::ThinPlate {
            feature_cols,
            input_scales,
            ..
        }
        | SmoothBasisSpec::Matern {
            feature_cols,
            input_scales,
            ..
        }
        | SmoothBasisSpec::Duchon {
            feature_cols,
            input_scales,
            ..
        } => (feature_cols, input_scales.as_ref()),
        _ => {
            crate::bail_invalid_basis!("term '{}' is not a spatial smooth", term.name);
        }
    };
    let mut x = select_columns(data, feature_cols)?;
    if let Some(scales) = input_scales {
        apply_input_standardization(&mut x, scales);
    } else if let Some(scales) = compute_spatial_input_scales(x.view()) {
        apply_input_standardization(&mut x, &scales);
    }
    Ok(x)
}


fn plan_joint_spatial_centers_for_term_blocks(
    data: ArrayView2<'_, f64>,
    term_blocks: &[Vec<SmoothTermSpec>],
) -> Result<Vec<Vec<SmoothTermSpec>>, BasisError> {
    let mut planned_blocks = term_blocks.to_vec();
    let n = data.nrows();
    let mut groups: BTreeMap<JointSpatialCenterGroupKey, Vec<(usize, usize)>> = BTreeMap::new();

    for (block_idx, terms) in planned_blocks.iter().enumerate() {
        for (term_idx, term) in terms.iter().enumerate() {
            let Some(strategy) = spatial_term_center_strategy(term) else {
                continue;
            };
            if !center_strategy_is_auto(strategy) {
                continue;
            }
            let Some(group_key) = spatial_term_group_key(term) else {
                continue;
            };
            if !matches!(
                group_key.strategy_kind,
                CenterStrategyKind::EqualMass
                    | CenterStrategyKind::EqualMassCovarRepresentative
                    | CenterStrategyKind::FarthestPoint
                    | CenterStrategyKind::KMeans
            ) {
                continue;
            }
            if center_strategy_num_centers(strategy).is_none() {
                continue;
            }
            groups
                .entry(group_key)
                .or_default()
                .push((block_idx, term_idx));
        }
    }

    for (group_key, members) in groups {
        if members.len() < 2 {
            continue;
        }
        let min_required = members
            .iter()
            .map(|&(block_idx, term_idx)| {
                spatial_term_min_center_count(&planned_blocks[block_idx][term_idx])
            })
            .max()
            .unwrap_or(1);
        let joint_centers = group_key
            .requested_num_centers
            .max(min_required)
            .min(n.max(1));
        let (first_block_idx, first_term_idx) = members[0];
        let prototype = &planned_blocks[first_block_idx][first_term_idx];
        let standardized = standardized_spatial_term_data(data, prototype)?;
        let strategy = spatial_term_center_strategy(prototype).ok_or_else(|| {
            BasisError::InvalidInput(format!(
                "term '{}' lost its spatial center strategy during joint planning",
                prototype.name
            ))
        })?;
        let joint_strategy = center_strategy_with_num_centers(strategy, joint_centers)?;
        let shared_centers = select_centers_by_strategy(standardized.view(), &joint_strategy)?;
        log::info!(
            "sharing {} spatial centers across {} smooth terms over columns {:?} (requested {} centers)",
            shared_centers.nrows(),
            members.len(),
            group_key.feature_cols,
            group_key.requested_num_centers,
        );
        for (block_idx, term_idx) in members {
            set_spatial_term_centers(
                &mut planned_blocks[block_idx][term_idx],
                shared_centers.clone(),
            )?;
        }
    }

    // Sentinel auto-init: Matern and thin-plate builders write length_scale =
    // 0.0 when the user didn't pass `length_scale=...`. Replace those with a
    // data-driven initialization here so REML starts in a regime where it can
    // escape; the hard-coded 1.0 default was a basin from which ν ≥ 5/2 Matern
    // could not recover on high-frequency truths, silently collapsing the fit
    // to a near-constant prediction.
    for block in planned_blocks.iter_mut() {
        for term in block.iter_mut() {
            auto_init_length_scale_in_place(data, term);
        }
    }

    Ok(planned_blocks)
}


/// Compute a data-driven initial length scale from the per-axis range of the
/// feature columns. The heuristic `max_range / sqrt(n)` puts the kernel on
/// the wiggly side of REML's basin so the optimizer can grow it back if the
/// signal is smooth, but is small enough that high-frequency truths remain
/// reachable for smoother kernels (ν ≥ 5/2). Clamped to a tiny positive
/// floor so degenerate constant-input columns can't produce 0.
fn auto_initial_length_scale(data: ArrayView2<'_, f64>, feature_cols: &[usize]) -> f64 {
    /// Tiny positive floor for the auto length scale, guarding against a zero
    /// kernel range when every feature column is (near-)constant.
    const LENGTH_SCALE_FLOOR: f64 = 1e-6;
    let n = data.nrows();
    if n == 0 || feature_cols.is_empty() {
        return 1.0;
    }
    let mut max_range = 0.0_f64;
    for &c in feature_cols {
        if c >= data.ncols() {
            continue;
        }
        let col = data.column(c);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in col.iter() {
            if v.is_finite() {
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
        }
        if hi > lo {
            let r = hi - lo;
            if r > max_range {
                max_range = r;
            }
        }
    }
    if !max_range.is_finite() || max_range <= 0.0 {
        return 1.0;
    }
    let init = max_range / (n as f64).sqrt();
    init.max(LENGTH_SCALE_FLOOR).min(max_range)
}


/// Walk a term and, if it is a Matern or thin-plate smooth whose length_scale
/// was left at the auto sentinel (`0.0`), overwrite it with
/// [`auto_initial_length_scale`].
fn auto_init_length_scale_in_place(data: ArrayView2<'_, f64>, term: &mut SmoothTermSpec) {
    auto_init_length_scale_in_basis(data, &mut term.basis);
}


/// Replace the `0.0` auto-init length-scale sentinel with a data-derived value
/// for any Matern / thin-plate kernel reachable from this basis — including the
/// inner kernel of a `by=`/factor-smooth wrapper.
///
/// `by=<factor>` and the sum-to-zero factor smooth wrap a spatial kernel inside
/// `SmoothBasisSpec::ByVariable` / `SmoothBasisSpec::FactorSumToZero` /
/// `SmoothBasisSpec::BySmooth`, so the wrapper variant is what the planner sees.
/// Without recursing into the wrapped basis the inner ThinPlate/Matern keeps the
/// `0.0` sentinel (the post-`1605b3a6e` builder default), which makes the kernel
/// distance divide by `length_scale² = 0`, producing a non-finite design at both
/// fit and predict time. Recurse so the inner kernel is initialized identically
/// to a top-level one.
fn auto_init_length_scale_in_basis(data: ArrayView2<'_, f64>, basis: &mut SmoothBasisSpec) {
    match basis {
        SmoothBasisSpec::Matern {
            feature_cols, spec, ..
        } => {
            if spec.length_scale == 0.0 {
                spec.length_scale = auto_initial_length_scale(data, feature_cols);
            }
        }
        SmoothBasisSpec::ThinPlate {
            feature_cols, spec, ..
        } => {
            if spec.length_scale == 0.0 {
                spec.length_scale = auto_initial_length_scale(data, feature_cols);
            }
        }
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            auto_init_length_scale_in_basis(data, inner);
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => {
            auto_init_length_scale_in_basis(data, smooth);
        }
        _ => {}
    }
}


fn freeze_raw_spatial_metadata(metadata: BasisMetadata, raw_cols: usize) -> BasisMetadata {
    match metadata {
        BasisMetadata::ThinPlate {
            centers,
            length_scale,
            periodic,
            identifiability_transform: None,
            input_scales,
            radial_reparam,
        } => BasisMetadata::ThinPlate {
            centers,
            length_scale,
            periodic,
            identifiability_transform: Some(Array2::eye(raw_cols)),
            input_scales,
            radial_reparam,
        },
        BasisMetadata::Duchon {
            centers,
            length_scale,
            periodic,
            power,
            nullspace_order,
            identifiability_transform: None,
            input_scales,
            aniso_log_scales,
            operator_collocation_points,
        } => BasisMetadata::Duchon {
            centers,
            length_scale,
            periodic,
            power,
            nullspace_order,
            identifiability_transform: Some(Array2::eye(raw_cols)),
            input_scales,
            aniso_log_scales,
            operator_collocation_points,
        },
        other => other,
    }
}


fn matern_operator_penalty_triplet_from_metadata(
    metadata: &BasisMetadata,
) -> Result<(Vec<Array2<f64>>, Vec<usize>, Vec<PenaltyInfo>), BasisError> {
    let BasisMetadata::Matern {
        centers,
        length_scale,
        periodic,
        nu,
        include_intercept,
        identifiability_transform,
        aniso_log_scales,
        input_scales,
        ..
    } = metadata
    else {
        crate::bail_invalid_basis!("Matérn operator penalties require Matérn metadata");
    };
    // The metadata records `length_scale` in *original* (un-standardized) data
    // coordinates, while `centers` live in the *standardized* coordinate frame
    // (per-axis division by `input_scales`). The realized design built the
    // kernel against those standardized centers using the σ_geom-compensated
    // effective length scale `length_scale / σ_geom`. The collocation operators
    // here are evaluated on the same standardized centers, so they must use the
    // SAME effective length scale — otherwise the penalty regularizes a
    // different RKHS range than the design lives in, leaving rough coefficient
    // directions effectively unpenalized. That mismatch is benign in 1-D
    // (no standardization) but produces a catastrophic out-of-sample blow-up in
    // d ≥ 2 where σ_geom ≠ 1 (#706).
    let penalty_length_scale = match input_scales.as_deref() {
        Some(scales) => compensate_length_scale_for_standardization(*length_scale, scales),
        None => *length_scale,
    };
    let penalty_centers = crate::basis::expand_periodic_centers(centers, periodic.as_deref())?;
    let ops = build_matern_collocation_operator_matrices(
        penalty_centers.view(),
        None,
        penalty_length_scale,
        *nu,
        *include_intercept,
        identifiability_transform.as_ref().map(|z| z.view()),
        aniso_log_scales.as_deref(),
    )?;
    // Gate the operator dials on the Matérn-ν RKHS Sobolev order m = ν + d/2:
    // mass (j=0) is always on, tension (j=1) is on for m > 1, stiffness (j=2)
    // is on for m > 2. The threshold is strict so the roughest kernel ν=1/2 in
    // d=1 (m=1, the exponential/OU H¹ process) sheds both higher operators —
    // its kernel already encodes the H¹ control, so adding an extra tension
    // dial over-smooths the oscillation it is meant to track (#707). The
    // matching gate lives at `DuchonOperatorPenaltySpec::matern_for_smoothness`.
    const ORDER_EPS: f64 = 1e-9;
    let d = penalty_centers.ncols();
    let m = nu.half_integer_value() + 0.5 * d as f64;
    let mut candidates = Vec::with_capacity(3);
    for (raw, source, min_order) in [
        (ops.d0.t().dot(&ops.d0), PenaltySource::OperatorMass, 0.0),
        (ops.d1.t().dot(&ops.d1), PenaltySource::OperatorTension, 1.0),
        (
            ops.d2.t().dot(&ops.d2),
            PenaltySource::OperatorStiffness,
            2.0,
        ),
    ] {
        if min_order > 0.0 && m <= min_order + ORDER_EPS {
            continue;
        }
        let sym = (&raw + &raw.t()) * 0.5;
        let (matrix, normalization_scale) = normalize_penalty_in_constrained_space(&sym);
        candidates.push(PenaltyCandidate {
            matrix,
            nullspace_dim_hint: 0,
            source,
            normalization_scale,
            kronecker_factors: None,
            op: None,
        });
    }
    filter_active_penalty_candidates(candidates)
}


fn normalize_penalty_in_constrained_space(matrix: &Array2<f64>) -> (Array2<f64>, f64) {
    // Constrained-space normalization:
    //   c = ||S_con||_F,  S_tilde = S_con / c.
    // This is the only normalization coherent with a REML objective that is
    // evaluated entirely in constrained coordinates.
    let matrix = (matrix + &matrix.t().to_owned()) * 0.5;
    // Clamp noise-floor negative eigenvalues so β'Sβ is non-negative as a contract, not just in exact arithmetic.
    let matrix = crate::terms::basis::project_penalty_to_psd_cone(&matrix);
    let c = matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
    if c.is_finite() && c > 0.0 {
        (matrix.mapv(|v| v / c), c)
    } else {
        (matrix, 1.0)
    }
}


fn build_periodic_fourier_margin(
    x: ArrayView1<'_, f64>,
    period: f64,
    requested_cols: usize,
    penalty_order: usize,
) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>), BasisError> {
    if !period.is_finite() || period <= 0.0 {
        crate::bail_invalid_basis!(
            "periodic tensor margin requires finite positive period, got {period}"
        );
    }
    let q = requested_cols.max(3);
    let harmonics = q / 2;
    let has_nyquist_cos = q.is_multiple_of(2);
    let mut basis = Array2::<f64>::zeros((x.len(), q));
    basis.column_mut(0).fill(1.0);
    for (i, &xi) in x.iter().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * xi / period;
        let mut col = 1usize;
        for h in 1..=harmonics {
            if col >= q {
                break;
            }
            basis[[i, col]] = (h as f64 * angle).cos();
            col += 1;
            if col >= q {
                break;
            }
            basis[[i, col]] = (h as f64 * angle).sin();
            col += 1;
        }
        if has_nyquist_cos && q > 1 {
            basis[[i, q - 1]] = (harmonics as f64 * angle).cos();
        }
    }
    let mut penalty = Array2::<f64>::zeros((q, q));
    let order = penalty_order.max(1) as i32;
    let mut col = 1usize;
    for h in 1..=harmonics {
        let w = (h as f64).powi(2 * order);
        if col < q {
            penalty[[col, col]] = w;
            col += 1;
        }
        if col < q {
            penalty[[col, col]] = w;
            col += 1;
        }
    }
    if has_nyquist_cos && q > 1 {
        penalty[[q - 1, q - 1]] = (harmonics as f64).powi(2 * order);
    }
    // The Fourier margin has no B-spline knots; this vector is a placeholder
    // that downstream code (the tensor freeze) treats as carrying the periodic
    // control-site count in its *length* and the domain start in `[0]`. Its
    // length MUST equal the basis column count `q` (= `basis.ncols()`): the
    // freeze records `num_basis = knots.len()` and the predict-time rebuild
    // re-derives `q` columns from it, so a `q + 1`-length vector reconstructs
    // one extra column per periodic axis and breaks the frozen identifiability
    // transform (issue #498).
    let knots = Array1::linspace(0.0, period, q);
    Ok((basis, penalty, knots))
}


fn tensor_product_design_from_sparse_marginals(
    marginal_sparse: &[&SparseColMat<usize, f64>],
) -> Result<SparseColMat<usize, f64>, BasisError> {
    if marginal_sparse.is_empty() {
        crate::bail_invalid_basis!("TensorBSpline requires at least one marginal basis");
    }
    let n = marginal_sparse[0].nrows();
    for (i, m) in marginal_sparse.iter().enumerate().skip(1) {
        if m.nrows() != n {
            crate::bail_dim_basis!(
                "tensor sparse marginal row mismatch at dim {i}: expected {n}, got {}",
                m.nrows()
            );
        }
    }
    let dims: Vec<usize> = marginal_sparse.iter().map(|m| m.ncols()).collect();
    let total_cols = dims.iter().try_fold(1usize, |acc, &q| {
        acc.checked_mul(q)
            .ok_or_else(|| BasisError::DimensionMismatch("tensor basis too large".to_string()))
    })?;
    let mut strides = vec![1usize; dims.len()];
    for d in (0..dims.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1]
            .checked_mul(dims[d + 1])
            .ok_or_else(|| BasisError::DimensionMismatch("tensor basis too large".to_string()))?;
    }

    use faer::sparse::SparseRowMat;
    let csrs: Vec<SparseRowMat<usize, f64>> = marginal_sparse
        .iter()
        .enumerate()
        .map(|(d, m)| {
            m.as_ref().to_row_major().map_err(|e| {
                BasisError::SparseCreation(format!(
                    "tensor sparse marginal {d} CSR conversion failed: {e:?}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let row_ptrs: Vec<&[usize]> = csrs.iter().map(|c| c.symbolic().row_ptr()).collect();
    let col_idxs: Vec<&[usize]> = csrs.iter().map(|c| c.symbolic().col_idx()).collect();
    let vals: Vec<&[f64]> = csrs.iter().map(|c| c.val()).collect();

    use rayon::prelude::*;
    const CHUNK: usize = 1024;
    let num_chunks = n.div_ceil(CHUNK);
    let per_chunk: Vec<Vec<Triplet<usize, usize, f64>>> = (0..num_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            let row_start = chunk_idx * CHUNK;
            let row_end = (row_start + CHUNK).min(n);
            let mut chunk_triplets = Vec::<Triplet<usize, usize, f64>>::new();
            let mut cur_cols = Vec::<usize>::with_capacity(64);
            let mut cur_vals = Vec::<f64>::with_capacity(64);
            let mut next_cols = Vec::<usize>::with_capacity(64);
            let mut next_vals = Vec::<f64>::with_capacity(64);
            for i in row_start..row_end {
                cur_cols.clear();
                cur_vals.clear();
                cur_cols.push(0);
                cur_vals.push(1.0);
                let mut row_is_zero = false;
                for d in 0..dims.len() {
                    let row_start_d = row_ptrs[d][i];
                    let row_end_d = row_ptrs[d][i + 1];
                    if row_start_d == row_end_d {
                        row_is_zero = true;
                        break;
                    }
                    let stride = strides[d];
                    next_cols.clear();
                    next_vals.clear();
                    next_cols.reserve(cur_cols.len() * (row_end_d - row_start_d));
                    next_vals.reserve(cur_vals.len() * (row_end_d - row_start_d));
                    for (&prev_col, &prev_val) in cur_cols.iter().zip(cur_vals.iter()) {
                        for ptr in row_start_d..row_end_d {
                            let cj = col_idxs[d][ptr];
                            let vj = vals[d][ptr];
                            next_cols.push(prev_col + cj * stride);
                            next_vals.push(prev_val * vj);
                        }
                    }
                    std::mem::swap(&mut cur_cols, &mut next_cols);
                    std::mem::swap(&mut cur_vals, &mut next_vals);
                }
                if row_is_zero {
                    continue;
                }
                for (&col, &val) in cur_cols.iter().zip(cur_vals.iter()) {
                    chunk_triplets.push(Triplet::new(i, col, val));
                }
            }
            chunk_triplets
        })
        .collect();
    let total_nnz: usize = per_chunk.iter().map(Vec::len).sum();
    let mut triplets = Vec::<Triplet<usize, usize, f64>>::with_capacity(total_nnz);
    for chunk in per_chunk {
        triplets.extend(chunk);
    }
    SparseColMat::try_new_from_triplets(n, total_cols, &triplets).map_err(|e| {
        BasisError::SparseCreation(format!(
            "failed to assemble sparse tensor product design: {e:?}"
        ))
    })
}


fn build_tensor_bspline_basis(
    data: ArrayView2<'_, f64>,
    feature_cols: &[usize],
    spec: &TensorBSplineSpec,
) -> Result<BasisBuildResult, BasisError> {
    if feature_cols.is_empty() {
        crate::bail_invalid_basis!("TensorBSpline requires at least one feature column");
    }
    if feature_cols.len() != spec.marginalspecs.len() {
        crate::bail_dim_basis!(
            "TensorBSpline feature/spec mismatch: feature_cols={}, marginalspecs={}",
            feature_cols.len(),
            spec.marginalspecs.len()
        );
    }
    if !spec.periods.is_empty() && spec.periods.len() != feature_cols.len() {
        crate::bail_dim_basis!(
            "TensorBSpline periods length {} does not match feature count {}",
            spec.periods.len(),
            feature_cols.len()
        );
    }
    let p = data.ncols();
    for &c in feature_cols {
        if c >= p {
            crate::bail_dim_basis!(
                "tensor feature column {c} is out of bounds for data with {p} columns"
            );
        }
    }

    let mut marginal_knots = Vec::<Array1<f64>>::with_capacity(feature_cols.len());
    let mut marginal_degrees = Vec::<usize>::with_capacity(feature_cols.len());
    let mut marginalnum_basis = Vec::<usize>::with_capacity(feature_cols.len());
    let mut marginal_penalties = Vec::<Array2<f64>>::with_capacity(feature_cols.len());
    let mut marginal_designs = Vec::<Array2<f64>>::with_capacity(feature_cols.len());
    // Per-margin effective period: either user-set via `spec.periods` (forcing
    // the Fourier path) or implied by a `PeriodicUniform` marginal knotspec
    // (which the 1D B-spline builder already realizes as a periodic basis).
    // Captured here so freeze→reload round-trips both routes back to a
    // `PeriodicUniform` marginal knotspec; otherwise a `PeriodicUniform`
    // margin specified without `spec.periods` would freeze as a plain
    // `Provided(knots)` open spline and lose its wrap-around at predict time.
    let mut marginal_effective_periods = Vec::<Option<f64>>::with_capacity(feature_cols.len());
    // Per-marginal sparse representation, populated when the 1D builder returned
    // a `DesignMatrix::Sparse`. Used to assemble the Khatri-Rao tensor product
    // sparsely (only ∏(degree+1) nonzeros per row) instead of densifying to
    // shape (n, ∏ q_j) up front. When any marginal lacks a sparse form (e.g.
    // periodic B-splines currently realize a dense Array2), we fall back to the
    // existing dense Khatri-Rao path.
    let mut marginal_sparse =
        Vec::<Option<SparseColMat<usize, f64>>>::with_capacity(feature_cols.len());

    // Reuse the robust 1D builder to ensure the same knot validation and
    // marginal difference-penalty construction as standalone smooth terms.
    for (dim, (&col, marginalspec)) in feature_cols
        .iter()
        .zip(spec.marginalspecs.iter())
        .enumerate()
    {
        // Tensor basis uses raw marginal knot-product columns. Applying 1D
        // identifiability constraints here would change marginal penalty sizes
        // without changing the tensor design construction, causing dimension
        // mismatch. Keep marginal builders unconstrained at this stage.
        if let Some(period) = spec.periods.get(dim).and_then(|p| *p) {
            let requested_cols = match marginalspec.knotspec {
                BSplineKnotSpec::Generate {
                    num_internal_knots, ..
                } => num_internal_knots + marginalspec.degree + 1,
                BSplineKnotSpec::Provided(ref knots) => {
                    knots.len().saturating_sub(marginalspec.degree + 1)
                }
                BSplineKnotSpec::Automatic {
                    num_internal_knots, ..
                } => {
                    // Fallback internal-knot count when an automatic marginal has
                    // not yet resolved its knot count at periodic-margin build
                    // time; matches the modest default used for a 1-D `s()`.
                    const DEFAULT_AUTOMATIC_INTERNAL_KNOTS: usize = 8;
                    num_internal_knots.unwrap_or(DEFAULT_AUTOMATIC_INTERNAL_KNOTS)
                        + marginalspec.degree
                        + 1
                }
                BSplineKnotSpec::PeriodicUniform { num_basis, .. } => num_basis,
            };
            let (basis, penalty, knots) = build_periodic_fourier_margin(
                data.column(col),
                period,
                requested_cols,
                marginalspec.penalty_order,
            )?;
            marginal_knots.push(knots);
            marginal_degrees.push(marginalspec.degree);
            marginalnum_basis.push(basis.ncols());
            marginal_designs.push(basis);
            marginal_penalties.push(penalty);
            // Periodic Fourier margins are realized densely; no sparse form
            // is available, so record `None` and force the dense fall-back
            // for the tensor product if any dimension is periodic.
            marginal_sparse.push(None);
            marginal_effective_periods.push(Some(period));
        } else {
            let mut marginal_unconstrained = marginalspec.clone();
            marginal_unconstrained.identifiability = BSplineIdentifiability::None;
            let built = build_bspline_basis_1d(data.column(col), &marginal_unconstrained)?;
            let knots = match built.metadata {
                BasisMetadata::BSpline1D { knots, .. } => knots,
                _ => {
                    crate::bail_invalid_basis!(
                        "internal TensorBSpline error at dim {dim}: expected BSpline1D metadata"
                    );
                }
            };
            marginal_knots.push(knots);
            marginal_degrees.push(marginalspec.degree);
            marginalnum_basis.push(built.design.ncols());
            // Capture the sparse representation of this marginal (when the
            // 1D builder produced one) before densifying for the dense
            // marginal cache used by `tensor_product_design_from_marginals`
            // and `TensorProductDesignOperator`.
            let sparse_view: Option<SparseColMat<usize, f64>> =
                built.design.as_sparse().map(|sd| {
                    let inner: &SparseColMat<usize, f64> = sd;
                    inner.clone()
                });
            marginal_sparse.push(sparse_view);
            marginal_designs.push(built.design.to_dense());
            marginal_penalties.push(
                built
                    .penalties
                    .first()
                    .ok_or_else(|| {
                        BasisError::InvalidInput(format!(
                            "internal TensorBSpline error at dim {dim}: missing marginal penalty"
                        ))
                    })?
                    .clone(),
            );
            built.nullspace_dims.first().ok_or_else(|| {
                BasisError::InvalidInput(format!(
                    "internal TensorBSpline error at dim {dim}: missing marginal nullspace dim"
                ))
            })?;
            // A `PeriodicUniform` marginal knotspec implies the margin is
            // wrap-around: the 1D builder already realized it as a periodic
            // basis, so the tensor product inherits that periodicity. Record
            // the period derived from the knotspec's data range so freeze
            // restores `PeriodicUniform` on the marginal — otherwise the
            // round-trip downgrades it to `Provided(knots)` (an open spline)
            // and predict-time wraps disappear.
            let implied_period = match marginalspec.knotspec {
                BSplineKnotSpec::PeriodicUniform { data_range, .. } => {
                    Some(data_range.1 - data_range.0)
                }
                _ => None,
            };
            marginal_effective_periods.push(implied_period);
        }
    }

    let total_cols: usize = marginalnum_basis.iter().product();
    let mut dense_design = (!matches!(spec.identifiability, TensorBSplineIdentifiability::None))
        .then(|| tensor_product_design_from_marginals(&marginal_designs))
        .transpose()?;
    let mut candidates = Vec::<PenaltyCandidate>::with_capacity(
        marginal_penalties.len() + if spec.double_penalty { 1 } else { 0 },
    );

    // Tensor-product smoothing parameters are one-per-margin.  Therefore the
    // physical penalty attached to a margin must be normalized in that margin's
    // own working coordinates before it is embedded in the full tensor product.
    // Normalizing only the already-Kroneckered matrix would fold arbitrary
    // dimension-dependent identity factors into the margin's lambda and would
    // make anisotropic REML/LAML smoothing depend on the other margins' basis
    // sizes rather than on the marginal roughness operator itself.
    let normalized_marginal_penalties: Vec<(Array2<f64>, f64)> = marginal_penalties
        .iter()
        .map(normalize_penalty_in_constrained_space)
        .collect();
    let mut kronecker_marginal_penalties =
        Vec::<Array2<f64>>::with_capacity(normalized_marginal_penalties.len());

    // Accumulate the Kronecker-sum of the per-margin penalties, `Σ_dim S_dim`,
    // whose null space is exactly the *joint* null space of all marginal
    // penalties — the tensor of the marginal polynomial null spaces, i.e. the
    // bilinear (low-order) directions that no marginal roughness operator
    // touches. The tensor double penalty (below) shrinks only this joint null,
    // never the already-penalized interaction range.
    let mut marginal_kron_sum = Array2::<f64>::zeros((total_cols, total_cols));

    for dim in 0..normalized_marginal_penalties.len() {
        let mut s_dim = Array2::<f64>::eye(1);
        let mut factors = Vec::<Array2<f64>>::with_capacity(marginalnum_basis.len());
        for (j, &qj) in marginalnum_basis.iter().enumerate() {
            let factor = if j == dim {
                normalized_marginal_penalties[j].0.clone()
            } else {
                Array2::<f64>::eye(qj)
            };
            factors.push(factor.clone());
            s_dim = kronecker_product(&s_dim, &factor);
        }
        if dim == kronecker_marginal_penalties.len() {
            kronecker_marginal_penalties.push(normalized_marginal_penalties[dim].0.clone());
        }
        marginal_kron_sum += &s_dim;

        candidates.push(PenaltyCandidate {
            matrix: s_dim,
            nullspace_dim_hint: 0,
            source: PenaltySource::TensorMarginal { dim },
            normalization_scale: normalized_marginal_penalties[dim].1,
            kronecker_factors: Some(factors),
            op: None,
        });
    }

    if spec.double_penalty {
        // mgcv `select=TRUE` semantics, mirrored from the 1D double penalty:
        // the extra penalty shrinks ONLY the directions left unpenalized by the
        // primary (marginal) penalties — here the joint null space of the
        // Kronecker-sum `Σ_dim S_dim` (the bilinear tensor null `Z₀ ⊗ Z₁`).
        // A full identity ridge `I` over *all* tensor coefficients (the prior
        // behavior) instead penalizes the already-penalized interaction range
        // as well, and REML/LAML then places positive weight on it and
        // systematically over-smooths the recovered surface; mgcv's `te`/`ti`
        // carry no such global ridge. Penalizing the null subspace alone keeps
        // the interaction range governed solely by the per-margin λ's.
        if let Some(shrink) =
            crate::terms::basis::build_nullspace_shrinkage_penalty(&marginal_kron_sum)?
        {
            let (matrix, normalization_scale) =
                normalize_penalty_in_constrained_space(&shrink.sym_penalty);
            candidates.push(PenaltyCandidate {
                matrix,
                nullspace_dim_hint: 0,
                source: PenaltySource::TensorGlobalRidge,
                normalization_scale,
                kronecker_factors: None,
                op: None,
            });
        }
    }

    let z_opt = match &spec.identifiability {
        TensorBSplineIdentifiability::None => None,
        TensorBSplineIdentifiability::SumToZero => {
            if total_cols < 2 {
                crate::bail_invalid_basis!(
                    "TensorBSpline requires at least 2 basis coefficients to enforce sum-to-zero identifiability"
                );
            }
            let dense_design_ref = dense_design.as_ref().ok_or_else(|| {
                BasisError::InvalidInput(
                    "tensor sum-to-zero identifiability requires a realized basis".to_string(),
                )
            })?;
            let (_, z) = apply_sum_to_zero_constraint(dense_design_ref.view(), None)?;
            Some(z)
        }
        TensorBSplineIdentifiability::MarginalSumToZero => {
            // `ti(...)`: drop the marginal main effects by centering every
            // margin independently, then form the tensor product of the
            // centered margins. Concretely, each margin `j` is reparameterized
            // by its own sum-to-zero null basis `Z_j` (so the constant — i.e.
            // the marginal intercept — is removed from that axis), and the
            // combined reparameterization is the Kronecker product
            // `Z = Z₀ ⊗ Z₁ ⊗ … ⊗ Z_{d-1}`. Applying `Z` to the full-tensor
            // design `B = B₀ ⊗ … ⊗ B_{d-1}` yields `B Z = (B₀ Z₀) ⊗ … ⊗
            // (B_{d-1} Z_{d-1})`, the tensor product of the centered margins,
            // which by construction contains no pure main effect.
            if marginal_designs.len() < 2 {
                crate::bail_invalid_basis!(
                    "tensor interaction (ti) identifiability requires at least 2 margins"
                );
            }
            let mut z = Array2::<f64>::eye(1);
            for (dim, marginal) in marginal_designs.iter().enumerate() {
                if marginal.ncols() < 2 {
                    crate::bail_invalid_basis!(
                        "tensor interaction (ti) margin {dim} has fewer than 2 basis functions; \
                         cannot remove its marginal main effect"
                    );
                }
                let (_, z_dim) = apply_sum_to_zero_constraint(marginal.view(), None)?;
                z = kronecker_product(&z, &z_dim);
            }
            Some(z)
        }
        TensorBSplineIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != total_cols {
                crate::bail_dim_basis!(
                    "frozen tensor identifiability transform mismatch: design has {} columns but transform has {} rows",
                    total_cols,
                    transform.nrows()
                );
            }
            Some(transform.clone())
        }
    };

    if let Some(z) = z_opt.as_ref() {
        let dense = dense_design.as_mut().ok_or_else(|| {
            BasisError::InvalidInput(
                "tensor identifiability transform requires a realized basis".to_string(),
            )
        })?;
        *dense = dense.dot(z);
        candidates = candidates
            .into_iter()
            .map(|candidate| -> Result<PenaltyCandidate, BasisError> {
                let zt_s = fast_atb(z, &candidate.matrix);
                let matrix = fast_ab(&zt_s, z);
                let (matrix, c_new) = normalize_penalty_in_constrained_space(&matrix);
                let preserve_margin_scale =
                    matches!(&candidate.source, PenaltySource::TensorMarginal { .. });
                let (matrix, normalization_scale) = if preserve_margin_scale {
                    (matrix.mapv(|v| v * c_new), candidate.normalization_scale)
                } else {
                    (matrix, candidate.normalization_scale * c_new)
                };
                Ok(PenaltyCandidate {
                    nullspace_dim_hint: candidate.nullspace_dim_hint,
                    matrix,
                    source: candidate.source,
                    normalization_scale,
                    // Z^T S Z is no longer a Kronecker product of the original
                    // marginal factors, so the Kronecker fast path in construction.rs
                    // must not be taken. Clearing kronecker_factors forces the generic
                    // block-local eigendecomposition path, which operates on the
                    // transformed matrix and is correct.
                    kronecker_factors: None,
                    op: candidate.op.clone(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
    }

    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(candidates)?;
    let identifiability_is_none =
        matches!(spec.identifiability, TensorBSplineIdentifiability::None);
    // All marginals expose a sparse representation iff each `marginal_sparse`
    // slot is `Some(...)`. Currently this is true when every marginal is a
    // free-boundary, non-periodic 1D B-spline returned as
    // `DesignMatrix::Sparse` from `build_bspline_basis_1d`. Periodic B-splines
    // and other dense-only marginals leave a `None` and trigger the fall-back
    // path. Identifiability transforms (`SumToZero`, `FrozenTransform`) make
    // the tensor design dense in general, so we also gate on that.
    let all_marginals_sparse = marginal_sparse.iter().all(Option::is_some);
    let design = if let Some(dense_design) = dense_design {
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(dense_design))
    } else if identifiability_is_none && all_marginals_sparse {
        // Sparse Khatri-Rao path: assemble the (n, ∏ q_j) tensor product
        // directly as a SparseColMat, preserving the ∏(degree_j+1) nonzero
        // structure per row instead of densifying to ∏ q_j columns. This is
        // mathematically identical to `tensor_product_design_from_marginals`
        // applied to the corresponding dense marginals.
        let sparse_marginals: Vec<&SparseColMat<usize, f64>> = marginal_sparse
            .iter()
            .map(|m| m.as_ref().expect("all_marginals_sparse just verified"))
            .collect();
        let sparse_design = tensor_product_design_from_sparse_marginals(&sparse_marginals)?;
        DesignMatrix::Sparse(crate::matrix::SparseDesignMatrix::new(sparse_design))
    } else {
        let marginals: Vec<Arc<Array2<f64>>> = marginal_designs
            .iter()
            .map(|m| Arc::new(m.clone()))
            .collect();
        let op = TensorProductDesignOperator::new(marginals).map_err(|e| {
            BasisError::InvalidInput(format!("TensorProductDesignOperator build failed: {e}"))
        })?;
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(op)))
    };

    Ok(BasisBuildResult {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
        metadata: BasisMetadata::TensorBSpline {
            feature_cols: feature_cols.to_vec(),
            knots: marginal_knots,
            degrees: marginal_degrees,
            // Prefer the per-margin effective period derived in the loop —
            // it captures both the explicit `spec.periods` route and the
            // implied period from a `PeriodicUniform` marginal knotspec.
            // Falling back to `spec.periods` when populated keeps any
            // user-supplied explicit period authoritative even if the
            // marginal knotspec carried no periodicity hint.
            periods: marginal_effective_periods,
            identifiability_transform: z_opt,
        },
        kronecker_factored: if matches!(spec.identifiability, TensorBSplineIdentifiability::None) {
            Some(KroneckerFactoredBasis {
                marginal_designs,
                marginal_penalties: kronecker_marginal_penalties,
                marginal_dims: marginalnum_basis.clone(),
                has_double_penalty: spec.double_penalty,
            })
        } else {
            None
        },
    })
}


fn tensor_product_design_from_marginals(
    marginal_designs: &[Array2<f64>],
) -> Result<Array2<f64>, BasisError> {
    if marginal_designs.is_empty() {
        crate::bail_invalid_basis!("TensorBSpline requires at least one marginal basis");
    }
    let n = marginal_designs[0].nrows();
    for (i, b) in marginal_designs.iter().enumerate().skip(1) {
        if b.nrows() != n {
            crate::bail_dim_basis!(
                "tensor marginal row mismatch at dim {i}: expected {n}, got {}",
                b.nrows()
            );
        }
    }
    let total_cols = marginal_designs.iter().try_fold(1usize, |acc, b| {
        acc.checked_mul(b.ncols())
            .ok_or_else(|| BasisError::DimensionMismatch("tensor basis too large".to_string()))
    })?;
    // Tensor-product Khatri-Rao: design[i, j] = Π_d marginal_d[i, j_d]
    // where j is the multi-index (j_1, ..., j_D) flattened. Independent
    // across rows; parallelize row chunks and fill the pre-allocated
    // contiguous Array2 in place (no Vec-flatten-collect intermediate,
    // which doubled the peak memory at large-scale N).
    use ndarray::parallel::prelude::*;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let mut design = Array2::<f64>::zeros((n, total_cols));
    design
        .axis_chunks_iter_mut(ndarray::Axis(0), 1024)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut block)| {
            let row_offset = chunk_idx * 1024;
            // Scratch buffers reused across rows in this chunk.
            let mut cur = Vec::<f64>::with_capacity(total_cols);
            let mut next = Vec::<f64>::with_capacity(total_cols);
            for (local_i, mut out_row) in block.outer_iter_mut().enumerate() {
                let i = row_offset + local_i;
                cur.clear();
                cur.push(1.0);
                for b in marginal_designs {
                    let q = b.ncols();
                    next.clear();
                    next.resize(cur.len() * q, 0.0);
                    // Hoist the row view out of the inner `col` loop so the
                    // q reads per `a_idx` reuse a single contiguous slice
                    // instead of recomputing `b[[i, col]]` strides per cell.
                    let b_row = b.row(i);
                    let b_slice = b_row
                        .as_slice()
                        .expect("Array2 row from outer_iter is contiguous");
                    for (a_idx, &aval) in cur.iter().enumerate() {
                        let off = a_idx * q;
                        let dst = &mut next[off..off + q];
                        for col in 0..q {
                            dst[col] = aval * b_slice[col];
                        }
                    }
                    std::mem::swap(&mut cur, &mut next);
                }
                // `out_row` is a row of the contiguous C-major `design`
                // Array2, so it is backed by a contiguous slice. Use a
                // bulk slice copy instead of an element-by-element write
                // loop.
                let out_slice = out_row
                    .as_slice_mut()
                    .expect("design row is contiguous in C-major Array2");
                out_slice.copy_from_slice(&cur);
            }
        });
    Ok(design)
}


pub(crate) fn build_random_effect_block(
    data: ArrayView2<'_, f64>,
    spec: &RandomEffectTermSpec,
) -> Result<RandomEffectBlock, BasisError> {
    let n = data.nrows();
    let p = data.ncols();
    if spec.feature_col >= p {
        crate::bail_dim_basis!(
            "random-effect term '{}' feature column {} out of bounds for {} columns",
            spec.name,
            spec.feature_col,
            p
        );
    }

    let col = data.column(spec.feature_col);
    if col.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!(
            "random-effect term '{}' contains non-finite group values",
            spec.name
        );
    }

    let kept_levels: Vec<u64> = if let Some(levels) = spec.frozen_levels.as_ref() {
        if levels.is_empty() {
            crate::bail_invalid_basis!(
                "random-effect term '{}' has empty frozen_levels",
                spec.name
            );
        }
        levels.clone()
    } else {
        let mut levels_set = BTreeSet::<u64>::new();
        for &v in col {
            levels_set.insert(v.to_bits());
        }
        if levels_set.is_empty() {
            crate::bail_invalid_basis!("random-effect term '{}' has no observed levels", spec.name);
        }
        let levels: Vec<u64> = levels_set.into_iter().collect();
        let start_idx = if spec.drop_first_level && levels.len() > 1 {
            1usize
        } else {
            0usize
        };
        levels[start_idx..].to_vec()
    };

    if kept_levels.is_empty() {
        crate::bail_invalid_basis!(
            "random-effect term '{}' drops all levels; keep at least one level",
            spec.name
        );
    }

    let q = kept_levels.len();
    let mut level_to_col = BTreeMap::<u64, usize>::new();
    for (idx, &bits) in kept_levels.iter().enumerate() {
        if level_to_col.insert(bits, idx).is_some() {
            crate::bail_invalid_basis!(
                "random-effect term '{}' has duplicate frozen level bits {bits}",
                spec.name
            );
        }
    }
    let mut group_ids = Vec::with_capacity(n);
    for &v in col {
        let bits = v.to_bits();
        group_ids.push(level_to_col.get(&bits).copied());
    }

    Ok(RandomEffectBlock {
        name: spec.name.clone(),
        group_ids,
        num_groups: q,
        kept_levels,
    })
}


impl SmoothDesign {
    /// Map an unconstrained term coefficient vector to its constrained shape space.
    /// This is useful for nonlinear fits that optimize unconstrained parameters.
    pub fn map_term_coefficients(
        unconstrained: &Array1<f64>,
        shape: ShapeConstraint,
    ) -> Result<Array1<f64>, BasisError> {
        if unconstrained.is_empty() {
            crate::bail_invalid_basis!("unconstrained coefficient vector cannot be empty");
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


pub(crate) struct LocalSmoothTermBuild {
    dim: usize,
    design: DesignMatrix,
    penalties: Vec<Array2<f64>>,
    ops: Vec<Option<std::sync::Arc<dyn crate::terms::penalty_op::PenaltyOp>>>,
    nullspaces: Vec<usize>,
    /// Per-active-penalty null-space eigenvector matrices, parallel to
    /// `penalties` / `ops` / `nullspaces`. `Some(U_null)` when
    /// `nullspaces[k] > 0`, with `U_null` orthonormal columns spanning
    /// `null(penalties[k])` in this smooth's local coordinate system; `None`
    /// when the active block is full-rank. Stage 1 plumbing; Stage 2
    /// consumes this to absorb the smooth's null space into the parametric
    /// block at `TermCollectionDesign` construction.
    null_eigenvectors: Vec<Option<Array2<f64>>>,
    /// Joint-null absorption rotation for this smooth. `Some(rotation)`
    /// records `Q = [U_range | U_null]` spanning `null(Σ_k penalties[k])`,
    /// the joint null across all active penalty blocks on this smooth.
    /// `None` means the joint penalty is full-rank (joint nullity = 0) or
    /// there are no penalties. Stage-2 commit A: plumbing only — populated
    /// by commit B, applied by commit D.
    joint_null_rotation: Option<crate::terms::basis::JointNullRotation>,
    penaltyinfo: Vec<PenaltyInfo>,
    pre_dropped_penaltyinfo: Vec<PenaltyInfo>,
    metadata: BasisMetadata,
    linear_constraints: Option<LinearInequalityConstraints>,
    box_reparam: bool,
    kronecker_factored: Option<KroneckerFactoredBasis>,
}


#[derive(Clone)]
struct PcaScoresMemmapDesignOperator {
    mmap: Arc<memmap2::Mmap>,
    data_offset: usize,
    nrows: usize,
    ncols: usize,
    chunk_size: usize,
}


impl PcaScoresMemmapDesignOperator {
    fn open(path: PathBuf, chunk_size: usize) -> Result<Self, BasisError> {
        let file = File::open(&path).map_err(|err| {
            BasisError::InvalidInput(format!(
                "failed to open lazy Pca .npy scores '{}': {err}",
                path.display()
            ))
        })?;
        // The .npy scores file is read-only training-cache data; this
        // module never mutates it. The error path below converts mmap
        // failure to a typed `BasisError::InvalidInput`.
        // SAFETY: `memmap2::Mmap::map` requires no concurrent writers; the
        // contract is held by this module's read-only access pattern.
        let mmap = unsafe {
            memmap2::Mmap::map(&file).map_err(|err| {
                BasisError::InvalidInput(format!(
                    "failed to memmap lazy Pca .npy scores '{}': {err}",
                    path.display()
                ))
            })?
        };
        let (data_offset, nrows, ncols) = parse_f64_2d_npy_header(&mmap, &path)?;
        let expected = data_offset
            .checked_add(nrows.saturating_mul(ncols).saturating_mul(8))
            .ok_or_else(|| {
                BasisError::InvalidInput(format!(
                    "lazy Pca .npy scores '{}' shape is too large",
                    path.display()
                ))
            })?;
        if mmap.len() < expected {
            crate::bail_invalid_basis!(
                "lazy Pca .npy scores '{}' is truncated: header expects {} bytes, file has {}",
                path.display(),
                expected,
                mmap.len()
            );
        }
        Ok(Self {
            mmap: Arc::new(mmap),
            data_offset,
            nrows,
            ncols,
            chunk_size: chunk_size.max(1),
        })
    }

    fn value(&self, row: usize, col: usize) -> f64 {
        let offset = self.data_offset + (row * self.ncols + col) * 8;
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(&self.mmap[offset..offset + 8]);
        f64::from_le_bytes(bytes)
    }

    fn chunk_rows(&self) -> usize {
        self.chunk_size.min(self.nrows.max(1))
    }
}


impl LinearOperator for PcaScoresMemmapDesignOperator {
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        assert_eq!(
            vector.len(),
            self.ncols,
            "lazy Pca apply vector length mismatch"
        );
        let mut out = Array1::<f64>::zeros(self.nrows);
        for start in (0..self.nrows).step_by(self.chunk_rows()) {
            let end = (start + self.chunk_rows()).min(self.nrows);
            for row in start..end {
                let mut acc = 0.0;
                for col in 0..self.ncols {
                    acc += self.value(row, col) * vector[col];
                }
                out[row] = acc;
            }
        }
        out
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        assert_eq!(
            vector.len(),
            self.nrows,
            "lazy Pca apply_transpose vector length mismatch"
        );
        let mut out = Array1::<f64>::zeros(self.ncols);
        for start in (0..self.nrows).step_by(self.chunk_rows()) {
            let end = (start + self.chunk_rows()).min(self.nrows);
            for row in start..end {
                let scale = vector[row];
                if scale == 0.0 {
                    continue;
                }
                for col in 0..self.ncols {
                    out[col] += scale * self.value(row, col);
                }
            }
        }
        out
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        if weights.len() != self.nrows {
            return Err(format!(
                "lazy Pca diag_xtw_x weight length mismatch: weights={}, nrows={}",
                weights.len(),
                self.nrows
            ));
        }
        let mut gram = Array2::<f64>::zeros((self.ncols, self.ncols));
        for start in (0..self.nrows).step_by(self.chunk_rows()) {
            let end = (start + self.chunk_rows()).min(self.nrows);
            for row in start..end {
                let w = weights[row];
                if w == 0.0 {
                    continue;
                }
                for a in 0..self.ncols {
                    let xa = self.value(row, a);
                    if xa == 0.0 {
                        continue;
                    }
                    for b in a..self.ncols {
                        gram[[a, b]] += w * xa * self.value(row, b);
                    }
                }
            }
        }
        for a in 0..self.ncols {
            for b in 0..a {
                gram[[a, b]] = gram[[b, a]];
            }
        }
        Ok(gram)
    }

    fn apply_weighted_normal(
        &self,
        weights: &Array1<f64>,
        vector: &Array1<f64>,
        penalty: Option<&Array2<f64>>,
        ridge: f64,
    ) -> Array1<f64> {
        assert_eq!(
            weights.len(),
            self.nrows,
            "lazy Pca weighted-normal weight mismatch"
        );
        assert_eq!(
            vector.len(),
            self.ncols,
            "lazy Pca weighted-normal vector mismatch"
        );
        let mut out = Array1::<f64>::zeros(self.ncols);
        for start in (0..self.nrows).step_by(self.chunk_rows()) {
            let end = (start + self.chunk_rows()).min(self.nrows);
            for row in start..end {
                let w = weights[row].max(0.0);
                if w == 0.0 {
                    continue;
                }
                let mut row_dot = 0.0;
                for col in 0..self.ncols {
                    row_dot += self.value(row, col) * vector[col];
                }
                if row_dot == 0.0 {
                    continue;
                }
                let scaled = w * row_dot;
                for col in 0..self.ncols {
                    out[col] += scaled * self.value(row, col);
                }
            }
        }
        if let Some(pen) = penalty {
            out += &pen.dot(vector);
        }
        if ridge > 0.0 {
            out += &vector.mapv(|x| ridge * x);
        }
        out
    }
}


impl DenseDesignOperator for PcaScoresMemmapDesignOperator {
    fn compute_xtwy(&self, weights: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>, String> {
        if weights.len() != self.nrows || y.len() != self.nrows {
            return Err(format!(
                "lazy Pca compute_xtwy dimension mismatch: weights={}, y={}, nrows={}",
                weights.len(),
                y.len(),
                self.nrows
            ));
        }
        let mut out = Array1::<f64>::zeros(self.ncols);
        for start in (0..self.nrows).step_by(self.chunk_rows()) {
            let end = (start + self.chunk_rows()).min(self.nrows);
            for row in start..end {
                let scale = weights[row] * y[row];
                if scale == 0.0 {
                    continue;
                }
                for col in 0..self.ncols {
                    out[col] += scale * self.value(row, col);
                }
            }
        }
        Ok(out)
    }

    fn row_chunk_into(
        &self,
        rows: Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if rows.end > self.nrows || rows.start > rows.end {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "lazy Pca row range out of bounds",
            });
        }
        if out.nrows() != rows.end - rows.start || out.ncols() != self.ncols {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "lazy Pca row_chunk_into shape mismatch",
            });
        }
        for (local, row) in (rows.start..rows.end).enumerate() {
            for col in 0..self.ncols {
                out[[local, col]] = self.value(row, col);
            }
        }
        Ok(())
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.nrows, self.ncols));
        self.row_chunk_into(0..self.nrows, out.view_mut())
            .expect("lazy Pca full materialization failed");
        out
    }
}


fn parse_f64_2d_npy_header(
    bytes: &[u8],
    path: &PathBuf,
) -> Result<(usize, usize, usize), BasisError> {
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        crate::bail_invalid_basis!("lazy Pca scores '{}' is not a .npy file", path.display());
    }
    let major = bytes[6];
    let header_len = match major {
        1 => u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
        2 | 3 => {
            if bytes.len() < 12 {
                crate::bail_invalid_basis!(
                    "lazy Pca scores '{}' has a truncated .npy header",
                    path.display()
                );
            }
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
        }
        other => {
            crate::bail_invalid_basis!(
                "lazy Pca scores '{}' uses unsupported .npy version {}",
                path.display(),
                other
            );
        }
    };
    let header_start = if major == 1 { 10 } else { 12 };
    let data_offset = header_start + header_len;
    if bytes.len() < data_offset {
        crate::bail_invalid_basis!(
            "lazy Pca scores '{}' has a truncated .npy header",
            path.display()
        );
    }
    let header = std::str::from_utf8(&bytes[header_start..data_offset]).map_err(|err| {
        BasisError::InvalidInput(format!(
            "lazy Pca scores '{}' has a non-UTF8 .npy header: {err}",
            path.display()
        ))
    })?;
    if !(header.contains("'descr': '<f8'")
        || header.contains("\"descr\": \"<f8\"")
        || header.contains("'descr': '|f8'")
        || header.contains("\"descr\": \"|f8\""))
    {
        crate::bail_invalid_basis!(
            "lazy Pca scores '{}' must be float64 little-endian .npy",
            path.display()
        );
    }
    if header.contains("True") {
        crate::bail_invalid_basis!(
            "lazy Pca scores '{}' must be C-contiguous, not Fortran-ordered",
            path.display()
        );
    }
    let shape_pos = header.find("shape").ok_or_else(|| {
        BasisError::InvalidInput(format!(
            "lazy Pca scores '{}' .npy header is missing shape",
            path.display()
        ))
    })?;
    let open = header[shape_pos..].find('(').ok_or_else(|| {
        BasisError::InvalidInput(format!(
            "lazy Pca scores '{}' .npy header has malformed shape",
            path.display()
        ))
    })? + shape_pos;
    let close = header[open..].find(')').ok_or_else(|| {
        BasisError::InvalidInput(format!(
            "lazy Pca scores '{}' .npy header has malformed shape",
            path.display()
        ))
    })? + open;
    let dims = header[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| {
            BasisError::InvalidInput(format!(
                "lazy Pca scores '{}' .npy shape is not integral: {err}",
                path.display()
            ))
        })?;
    if dims.len() != 2 {
        crate::bail_invalid_basis!(
            "lazy Pca scores '{}' must have shape (N, K), got {:?}",
            path.display(),
            dims
        );
    }
    Ok((data_offset, dims[0], dims[1]))
}


fn pca_center_mean(x: ArrayView2<'_, f64>) -> Result<Array1<f64>, BasisError> {
    if x.nrows() == 0 {
        crate::bail_invalid_basis!("Pca basis requires at least one row to compute center mean");
    }
    let mut mean = Array1::<f64>::zeros(x.ncols());
    for row in x.rows() {
        mean += &row;
    }
    mean.mapv_inplace(|v| v / x.nrows() as f64);
    Ok(mean)
}


fn build_pca_smooth_basis(
    data: ArrayView2<'_, f64>,
    feature_cols: &[usize],
    basis_matrix: &Array2<f64>,
    centered: bool,
    smooth_penalty: f64,
    center_mean: Option<&Array1<f64>>,
    pca_basis_path: Option<&PathBuf>,
    chunk_size: usize,
) -> Result<BasisBuildResult, BasisError> {
    if let Some(path) = pca_basis_path {
        let op = PcaScoresMemmapDesignOperator::open(path.clone(), chunk_size)?;
        if op.nrows != data.nrows() {
            crate::bail_dim_basis!(
                "lazy Pca scores row mismatch: .npy has {}, data has {}",
                op.nrows,
                data.nrows()
            );
        }
        let k = op.ncols;
        let mut penalty = Array2::<f64>::eye(k);
        penalty.mapv_inplace(|v| v * smooth_penalty);
        let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
            filter_active_penalty_candidates_with_ops(vec![PenaltyCandidate {
                matrix: penalty,
                nullspace_dim_hint: 0,
                source: PenaltySource::Other("PcaRidge".to_string()),
                normalization_scale: 1.0,
                kronecker_factors: None,
                op: None,
            }])?;
        return Ok(BasisBuildResult {
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(op))),
            penalties,
            nullspace_dims,
            penaltyinfo,
            ops,
            null_eigenvectors,
            joint_null_rotation: None,
            metadata: BasisMetadata::Pca {
                feature_cols: feature_cols.to_vec(),
                basis_matrix: basis_matrix.clone(),
                centered,
                smooth_penalty,
                center_mean: center_mean.cloned(),
                pca_basis_path: Some(path.clone()),
                chunk_size: chunk_size.max(1),
            },
            kronecker_factored: None,
        });
    }
    if basis_matrix.nrows() != feature_cols.len() {
        crate::bail_dim_basis!(
            "Pca basis row mismatch: basis rows={}, feature columns={}",
            basis_matrix.nrows(),
            feature_cols.len()
        );
    }
    let mut x = select_columns(data, feature_cols)?;
    let mean = if centered {
        match center_mean {
            Some(mean) => mean.clone(),
            None => pca_center_mean(x.view())?,
        }
    } else {
        Array1::<f64>::zeros(feature_cols.len())
    };
    if centered {
        for mut row in x.rows_mut() {
            row -= &mean;
        }
    }
    let design = fast_ab(&x, basis_matrix);
    let k = basis_matrix.ncols();
    let mut penalty = Array2::<f64>::eye(k);
    penalty.mapv_inplace(|v| v * smooth_penalty);
    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(vec![PenaltyCandidate {
            matrix: penalty,
            nullspace_dim_hint: 0,
            source: PenaltySource::Other("PcaRidge".to_string()),
            normalization_scale: 1.0,
            kronecker_factors: None,
            op: None,
        }])?;
    Ok(BasisBuildResult {
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(design)),
        penalties,
        nullspace_dims,
        penaltyinfo,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
        metadata: BasisMetadata::Pca {
            feature_cols: feature_cols.to_vec(),
            basis_matrix: basis_matrix.clone(),
            centered,
            smooth_penalty,
            center_mean: centered.then_some(mean),
            pca_basis_path: None,
            chunk_size: chunk_size.max(1),
        },
        kronecker_factored: None,
    })
}


fn apply_by_variable_to_local_build(
    mut built: LocalSmoothTermBuild,
    data: ArrayView2<'_, f64>,
    by_col: usize,
    by: &ByVariableSpec,
    term_name: &str,
) -> Result<LocalSmoothTermBuild, BasisError> {
    if by_col >= data.ncols() {
        crate::bail_dim_basis!(
            "by-variable smooth term '{term_name}' references column {by_col}, but data has {} columns",
            data.ncols()
        );
    }
    let weights = match by {
        ByVariableSpec::Numeric => data.column(by_col).to_owned(),
        ByVariableSpec::Level { value_bits, .. } => data.column(by_col).mapv(|value| {
            if value.to_bits() == *value_bits {
                1.0
            } else {
                0.0
            }
        }),
    };
    if weights.iter().any(|value| !value.is_finite()) {
        crate::bail_invalid_basis!(
            "by-variable smooth term '{term_name}' has non-finite by-column values"
        );
    }

    let mut dense = built
        .design
        .try_to_dense_by_chunks("by-variable smooth row gating")
        .map_err(BasisError::InvalidInput)?;
    for (mut row, &weight) in dense.rows_mut().into_iter().zip(weights.iter()) {
        row.mapv_inplace(|value| value * weight);
    }
    built.design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(dense));
    built.kronecker_factored = None;
    Ok(built)
}


fn ensure_by_variable_specs_match(
    kind: &BySmoothKind,
    by: &ByVariableSpec,
    term_name: &str,
) -> Result<(), BasisError> {
    match (kind, by) {
        (BySmoothKind::Numeric, ByVariableSpec::Numeric) => Ok(()),
        (BySmoothKind::Level { level_bits }, ByVariableSpec::Level { value_bits, .. })
            if level_bits == value_bits =>
        {
            Ok(())
        }
        _ => Err(BasisError::InvalidInput(format!(
            "by-variable smooth term '{term_name}' has inconsistent by-variable specifications"
        ))),
    }
}


/// Build a factor-smooth interaction basis (`bs="fs"`/`"sz"`/`"re"`).
///
/// A factor smooth replicates a shared marginal smooth in the continuous
/// covariate(s) once per level of a grouping factor, coupling all level blocks
/// through a *single* set of smoothing parameters (one per marginal penalty).
/// This is mgcv's `smooth.construct.fs.smooth.spec` realization and the
/// random-effect interpretation of a smooth: the per-level deviations are an
/// exchangeable family whose joint wiggliness/shrinkage is governed by the
/// shared λ, so the construction scales to many levels with a fixed parameter
/// count.
///
/// Flavours:
/// * `Fs` — full random factor-smooth. The marginal carries its wiggliness
///   penalty *and* a null-space ridge (double penalty), so the replicated
///   design is a proper full-rank random effect: each level's curve is shrunk
///   toward zero (intercept + linear trend included), recovering the mgcv
///   `bs="fs"` penalty structure `I_L ⊗ S_j` for every marginal penalty `S_j`.
/// * `Sz` — sum-to-zero factor smooth. Delegates to the existing
///   [`SmoothBasisSpec::FactorSumToZero`] construction (`L-1` deviation blocks,
///   coefficient-wise zero sum across levels).
/// * `Re` — pure random effect / random slope (`bs="re"`). A degree-1 marginal
///   gives the per-level `[1, x]` span; the penalty is the identity over each
///   level block (iid Gaussian coefficients), matching mgcv's `bs="re"` ridge.
///
/// The grouping levels are resolved once at fit time (sorted unique bit
/// patterns of the factor column) and frozen into the returned metadata so the
/// predict-time rebuild evaluates every row against its own level's block.
fn build_factor_smooth(
    data: ArrayView2<'_, f64>,
    spec: &FactorSmoothSpec,
    term_name: &str,
    workspace: &mut crate::basis::BasisWorkspace,
) -> Result<LocalSmoothTermBuild, BasisError> {
    if spec.continuous_cols.len() != 1 {
        crate::bail_invalid_basis!(
            "factor smooth term '{}' currently supports exactly one continuous covariate; found {}",
            term_name,
            spec.continuous_cols.len()
        );
    }
    let feature_col = spec.continuous_cols[0];
    let group_col = spec.group_col;
    if feature_col >= data.ncols() || group_col >= data.ncols() {
        crate::bail_dim_basis!(
            "factor smooth term '{}' references columns ({}, {}) out of bounds for {} columns",
            term_name,
            feature_col,
            group_col,
            data.ncols()
        );
    }

    // `Sz` is exactly the existing sum-to-zero factor smooth: reuse it verbatim
    // so there is a single source of truth for the zero-sum construction.
    if matches!(spec.flavour, FactorSmoothFlavour::Sz) {
        let levels = resolve_factor_smooth_levels(data, group_col, spec, term_name)?;
        let inner = SmoothBasisSpec::BSpline1D {
            feature_col,
            spec: factor_smooth_marginal_for_replay(&spec.marginal),
        };
        let sz_term = SmoothTermSpec {
            name: term_name.to_string(),
            basis: SmoothBasisSpec::FactorSumToZero {
                inner: Box::new(inner),
                by_col: group_col,
                levels,
                frozen_global_orthogonality: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        };
        return build_single_local_smooth_term(data, &sz_term, workspace);
    }

    let levels = resolve_factor_smooth_levels(data, group_col, spec, term_name)?;
    let n_levels = levels.len();
    if n_levels < 2 {
        crate::bail_invalid_basis!(
            "factor smooth term '{}' requires at least two grouping levels; found {}",
            term_name,
            n_levels
        );
    }

    // `Fs` (order ≥ 1, the default) is the random-effect flavour: it penalizes
    // each null-space dimension of the marginal wiggliness penalty separately
    // below (mgcv's `bs="fs"` construction). That replaces the marginal's single
    // *combined* double penalty, so disable the latter here to avoid penalizing
    // the null space twice (once combined, once per dimension). The explicit
    // `m=0` opt-out keeps the legacy combined double penalty and adds no
    // per-dimension penalties.
    let use_per_dim_null = matches!(
        &spec.flavour,
        FactorSmoothFlavour::Fs { m_null_penalty_orders }
            if m_null_penalty_orders.iter().copied().max().unwrap_or(0) >= 1
    );

    // Build the shared marginal design + penalties from the 1-D B-spline.
    // `Re` forces a degree-1 marginal (linear span) and replaces the marginal
    // wiggliness with an identity ridge below; `Fs` keeps the user's marginal
    // (cubic by default) and, under the per-dimension null path, gets its null
    // space penalized one dimension at a time after replication.
    let mut marginal_spec = factor_smooth_marginal_for_replay(&spec.marginal);
    if use_per_dim_null {
        marginal_spec.double_penalty = false;
    }
    let inner_term = SmoothTermSpec {
        name: format!("{term_name}::marginal"),
        basis: SmoothBasisSpec::BSpline1D {
            feature_col,
            spec: marginal_spec,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
    };
    let inner = build_single_local_smooth_term(data, &inner_term, workspace)?;
    let base = inner
        .design
        .try_to_dense_by_chunks("factor smooth marginal")
        .map_err(BasisError::InvalidInput)?;
    let n = base.nrows();
    let p = base.ncols();
    let q = p * n_levels;

    // Block-diagonal replicated design: row i contributes its marginal row to
    // the column block owned by its grouping level, zeros elsewhere.
    let mut dense = Array2::<f64>::zeros((n, q));
    for i in 0..n {
        let bits = data[[i, group_col]].to_bits();
        let level_idx = levels.iter().position(|b| *b == bits).ok_or_else(|| {
            BasisError::InvalidInput(format!(
                "factor smooth term '{term_name}' saw an unseen grouping level at row {}",
                i + 1
            ))
        })?;
        let start = level_idx * p;
        dense
            .slice_mut(s![i, start..start + p])
            .assign(&base.row(i));
    }

    // Penalties: replicate each marginal penalty into a block-diagonal
    // `I_L ⊗ S_j` so every level shares the same smoothing parameter λ_j (one
    // λ per marginal penalty), the defining feature of a factor smooth. For
    // `Re` the marginal penalty is replaced by an identity ridge so each
    // per-level coefficient is an iid Gaussian random effect.
    let marginal_penalties: Vec<Array2<f64>> = if matches!(spec.flavour, FactorSmoothFlavour::Re) {
        vec![Array2::<f64>::eye(p)]
    } else {
        inner.penalties.clone()
    };
    let marginal_penaltyinfo: Vec<PenaltyInfo> = if matches!(spec.flavour, FactorSmoothFlavour::Re)
    {
        vec![PenaltyInfo {
            source: PenaltySource::Primary,
            original_index: 0,
            active: true,
            effective_rank: p,
            dropped_reason: None,
            nullspace_dim_hint: 0,
            normalization_scale: 1.0,
            kronecker_factors: None,
        }]
    } else {
        inner.penaltyinfo.clone()
    };
    if marginal_penalties.len() != marginal_penaltyinfo.len() {
        crate::bail_invalid_basis!(
            "internal factor-smooth penalty metadata mismatch for term '{}': penalties={}, infos={}",
            term_name,
            marginal_penalties.len(),
            marginal_penaltyinfo.len()
        );
    }

    let mut penalties = Vec::<Array2<f64>>::with_capacity(marginal_penalties.len());
    let mut penaltyinfo = Vec::<PenaltyInfo>::with_capacity(marginal_penalties.len());
    for (penalty_pos, s_inner) in marginal_penalties.iter().enumerate() {
        let mut s_big = Array2::<f64>::zeros((q, q));
        for level in 0..n_levels {
            let start = level * p;
            s_big
                .slice_mut(s![start..start + p, start..start + p])
                .assign(s_inner);
        }
        let (s_big, factor_smooth_scale) = normalize_penalty_in_constrained_space(&s_big);
        let mut info = marginal_penaltyinfo[penalty_pos].clone();
        info.original_index = penalty_pos;
        info.normalization_scale *= factor_smooth_scale;
        info.nullspace_dim_hint = info.nullspace_dim_hint.saturating_mul(n_levels);
        info.kronecker_factors = None;
        penalties.push(s_big);
        penaltyinfo.push(info);
    }

    let mut nullspaces: Vec<usize> = if matches!(spec.flavour, FactorSmoothFlavour::Re) {
        vec![0]
    } else {
        inner
            .nullspaces
            .iter()
            .map(|ns| ns.saturating_mul(n_levels))
            .collect()
    };

    // `Fs` is the random-effect flavour of a smooth: the per-group curve is an
    // exchangeable Gaussian *function*, so EVERY coefficient — including the
    // {const, linear} null space of the marginal wiggliness penalty — must be
    // shrinkable toward zero under its own shared variance. The wiggliness
    // penalty `S_wiggle` shapes curvature but leaves the per-group intercept and
    // slope (its null space) completely UNPENALIZED. With the null space free,
    // each group fits its own intercept and slope with NO partial pooling, so
    // the held-out per-subject forecast inherits the full no-pooling variance
    // and curves away from the true per-group line (gam#712 real arm, gam#713;
    // gam#903 sleepstudy forecast ran ~74% over the lme4 BLUP bar).
    //
    // mgcv's `bs="fs"` fixes this by penalizing each null-space dimension
    // SEPARATELY (`smooth.construct.fs.smooth.spec` adds one rank-1 penalty per
    // null coordinate), each replicated block-diagonally across levels under a
    // single shared smoothing parameter — so REML fits a distinct
    // random-intercept variance and random-slope variance, the partial pooling
    // that makes the forecast track lme4's correlated random-effect BLUP. A
    // single *combined* null penalty (one λ for intercept+slope together) cannot
    // express the typically very different intercept and slope variances, which
    // is the residual forecast gap. We mirror mgcv exactly: for each orthonormal
    // null direction `z_k` of the marginal wiggliness penalty add
    // `I_L ⊗ (z_k z_kᵀ)` as its own penalty. The marginal's combined double
    // penalty was disabled above, so the null space is penalized once, per
    // dimension. With linear data REML drives the curvature λ up and degrades
    // `fs` to a linear random slope (edf → ≈2/group); with genuine curvature the
    // wiggliness λ stays small and the wiggle survives (data-adaptive, not a
    // cap). Gated by `m_null_penalty_orders`: order ≥ 1 (default) enables the
    // per-dimension null penalties; `m=0` keeps the legacy combined double
    // penalty and adds nothing here.
    if use_per_dim_null
        && let Some(Some(z)) = inner.null_eigenvectors.first()
        && z.nrows() == p
    {
        for k in 0..z.ncols() {
            // Rank-1 marginal penalty `z_k z_kᵀ`, replicated block-diagonally
            // across levels into `I_L ⊗ (z_k z_kᵀ)`. Its own λ is one shared
            // variance for this null component (intercept or slope) across all
            // groups — the random-effect structure of mgcv `fs`.
            let zk = z.column(k);
            let mut p_k = Array2::<f64>::zeros((p, p));
            for a in 0..p {
                for b in 0..p {
                    p_k[[a, b]] = zk[a] * zk[b];
                }
            }
            let mut s_null = Array2::<f64>::zeros((q, q));
            for level in 0..n_levels {
                let start = level * p;
                s_null
                    .slice_mut(s![start..start + p, start..start + p])
                    .assign(&p_k);
            }
            let (s_null, null_scale) = normalize_penalty_in_constrained_space(&s_null);
            let null_block = crate::terms::basis::analyze_penalty_block_with_op(&s_null, None)?;
            if null_block.rank > 0 {
                let original_index = penalties.len();
                penalties.push(null_block.sym_penalty);
                nullspaces.push(null_block.nullity);
                penaltyinfo.push(PenaltyInfo {
                    source: PenaltySource::Primary,
                    original_index,
                    active: true,
                    effective_rank: null_block.rank,
                    dropped_reason: None,
                    nullspace_dim_hint: null_block.nullity,
                    normalization_scale: null_scale,
                    kronecker_factors: None,
                });
            }
        }
    }
    let null_eigenvectors = crate::terms::basis::recompute_null_eigenvectors(&penalties)?;
    let joint_null_rotation = crate::terms::basis::compute_joint_null_rotation(&penalties)?;

    // Metadata: carry the marginal knot geometry + frozen levels so prediction
    // reconstructs an identical replicated design.
    let (knots, degree, periodic) = match &inner.metadata {
        BasisMetadata::BSpline1D {
            knots,
            periodic,
            degree,
            ..
        } => (
            knots.clone(),
            degree.unwrap_or(spec.marginal.degree),
            *periodic,
        ),
        other => {
            crate::bail_invalid_basis!(
                "factor smooth term '{}' produced an unexpected marginal metadata variant {:?}",
                term_name,
                other
            );
        }
    };
    let flavour_tag = match &spec.flavour {
        FactorSmoothFlavour::Fs { .. } => "fs",
        FactorSmoothFlavour::Sz => "sz",
        FactorSmoothFlavour::Re => "re",
    }
    .to_string();
    let metadata = BasisMetadata::FactorSmooth {
        continuous_cols: spec.continuous_cols.clone(),
        group_col,
        knots,
        degree,
        periodic,
        group_levels: levels,
        flavour: flavour_tag,
    };

    let ops = vec![None; penalties.len()];
    Ok(LocalSmoothTermBuild {
        dim: q,
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(dense)),
        penalties,
        ops,
        nullspaces,
        null_eigenvectors,
        joint_null_rotation,
        penaltyinfo,
        pre_dropped_penaltyinfo: Vec::new(),
        metadata,
        linear_constraints: None,
        box_reparam: false,
        kronecker_factored: None,
    })
}


/// Resolve the grouping levels for a factor smooth: replay the frozen level
/// list when present (predict path), otherwise discover the sorted unique bit
/// patterns of the factor column (fit path).
fn resolve_factor_smooth_levels(
    data: ArrayView2<'_, f64>,
    group_col: usize,
    spec: &FactorSmoothSpec,
    term_name: &str,
) -> Result<Vec<u64>, BasisError> {
    if let Some(frozen) = &spec.group_frozen_levels {
        if frozen.is_empty() {
            crate::bail_invalid_basis!(
                "factor smooth term '{}' has an empty frozen level list",
                term_name
            );
        }
        return Ok(frozen.clone());
    }
    let mut bits: Vec<u64> = data.column(group_col).iter().map(|v| v.to_bits()).collect();
    bits.sort_by(|a, b| {
        f64::from_bits(*a)
            .partial_cmp(&f64::from_bits(*b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    bits.dedup();
    Ok(bits)
}


/// Marginal B-spline spec for a factor-smooth block. The marginal always builds
/// without an identifiability constraint (the per-level replication, not a
/// sum-to-zero side constraint, provides identifiability against the parametric
/// block). At predict time the marginal's knot geometry has already been pinned
/// into `marginal.knotspec` by the metadata replay, so the spec is used
/// verbatim aside from clearing the identifiability transform.
fn factor_smooth_marginal_for_replay(marginal: &BSplineBasisSpec) -> BSplineBasisSpec {
    let mut m = marginal.clone();
    m.identifiability = BSplineIdentifiability::None;
    m
}


pub(crate) fn build_single_local_smooth_term(
    data: ArrayView2<'_, f64>,
    term: &SmoothTermSpec,
    workspace: &mut crate::basis::BasisWorkspace,
) -> Result<LocalSmoothTermBuild, BasisError> {
    if term.shape != ShapeConstraint::None && !shape_supports_basis(term) {
        crate::bail_invalid_basis!(
            "ShapeConstraint::{:?} is unsupported for term '{}'",
            term.shape,
            term.name
        );
    }
    if let SmoothBasisSpec::ByVariable {
        inner,
        by_col,
        kind,
        by,
    } = &term.basis
    {
        ensure_by_variable_specs_match(kind, by, &term.name)?;
        let inner_term = SmoothTermSpec {
            name: term.name.clone(),
            basis: (**inner).clone(),
            shape: term.shape,
            joint_null_rotation: None,
        };
        let built = build_single_local_smooth_term(data, &inner_term, workspace)?;
        return apply_by_variable_to_local_build(built, data, *by_col, by, &term.name);
    }

    let mut shape_axis_col: Option<usize> = None;
    let mut built: BasisBuildResult = match &term.basis {
        SmoothBasisSpec::FactorSumToZero {
            inner,
            by_col,
            levels,
            ..
        } => {
            if *by_col >= data.ncols() {
                crate::bail_dim_basis!(
                    "term '{}' by column {} out of bounds for {} columns",
                    term.name,
                    by_col,
                    data.ncols()
                );
            }
            if levels.len() < 2 {
                crate::bail_invalid_basis!(
                    "sum-to-zero factor smooth term '{}' requires at least two levels",
                    term.name
                );
            }
            if term.shape != ShapeConstraint::None {
                crate::bail_invalid_basis!(
                    "ShapeConstraint::{:?} is unsupported for sum-to-zero factor smooth term '{}'",
                    term.shape,
                    term.name
                );
            }
            let inner_term = SmoothTermSpec {
                name: format!("{}::inner", term.name),
                basis: (**inner).clone(),
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            };
            let mut inner_built = build_single_local_smooth_term(data, &inner_term, workspace)?;
            let base = inner_built
                .design
                .try_to_dense_by_chunks("sum-to-zero factor smooth")
                .map_err(BasisError::InvalidInput)?;
            let n = base.nrows();
            let p = base.ncols();
            let l_minus_one = levels.len() - 1;
            let mut dense = Array2::<f64>::zeros((n, p * l_minus_one));
            for i in 0..n {
                let bits = data[[i, *by_col]].to_bits();
                let level_idx = levels.iter().position(|b| *b == bits).ok_or_else(|| {
                    BasisError::InvalidInput(format!(
                        "sum-to-zero factor smooth term '{}' saw an unseen level at row {}",
                        term.name,
                        i + 1
                    ))
                })?;
                if level_idx < l_minus_one {
                    let start = level_idx * p;
                    dense
                        .slice_mut(s![i, start..start + p])
                        .assign(&base.row(i));
                } else {
                    for level in 0..l_minus_one {
                        let start = level * p;
                        dense
                            .slice_mut(s![i, start..start + p])
                            .assign(&base.row(i).mapv(|v| -v));
                    }
                }
            }
            let mut penalties = Vec::<Array2<f64>>::with_capacity(inner_built.penalties.len());
            let active_penalty_indices = inner_built
                .penaltyinfo
                .iter()
                .enumerate()
                .filter_map(|(idx, info)| info.active.then_some(idx))
                .collect::<Vec<_>>();
            if active_penalty_indices.len() != inner_built.penalties.len() {
                crate::bail_invalid_basis!(
                    "internal sz penalty metadata mismatch: activeinfos={}, penalties={}",
                    active_penalty_indices.len(),
                    inner_built.penalties.len()
                );
            }
            for (penalty_pos, s_inner) in inner_built.penalties.iter().enumerate() {
                let mut s_big = Array2::<f64>::zeros((p * l_minus_one, p * l_minus_one));
                for a in 0..l_minus_one {
                    for b in 0..l_minus_one {
                        let factor = if a == b { 2.0 } else { 1.0 };
                        let mut block = s_big.slice_mut(s![a * p..(a + 1) * p, b * p..(b + 1) * p]);
                        block.assign(&s_inner.mapv(|v| v * factor));
                    }
                }
                let (s_big, factor_smooth_scale) = normalize_penalty_in_constrained_space(&s_big);
                let info_idx = active_penalty_indices[penalty_pos];
                inner_built.penaltyinfo[info_idx].normalization_scale *= factor_smooth_scale;
                penalties.push(s_big);
            }
            inner_built.dim = p * l_minus_one;
            inner_built.design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(dense));
            inner_built.penalties = penalties;
            inner_built.ops = vec![None; inner_built.penalties.len()];
            inner_built.nullspaces = inner_built
                .nullspaces
                .iter()
                .map(|ns| ns.saturating_mul(l_minus_one))
                .collect();
            // Invariant: `null_eigenvectors[k]` must mirror `penalties[k]`'s
            // spectral null space. We just rebuilt `inner_built.penalties` from
            // Kronecker-like `S_big` blocks, so the previously-plumbed
            // `null_eigenvectors` (still parallel to the OLD per-level penalty)
            // is stale. Recompute from the rebuilt penalties to restore the
            // invariant; ditto for the joint-null absorption rotation.
            inner_built.null_eigenvectors =
                crate::terms::basis::recompute_null_eigenvectors(&inner_built.penalties)?;
            inner_built.joint_null_rotation =
                crate::terms::basis::compute_joint_null_rotation(&inner_built.penalties)?;
            inner_built.kronecker_factored = None;
            return Ok(inner_built);
        }
        SmoothBasisSpec::BSpline1D { feature_col, spec } => {
            if *feature_col >= data.ncols() {
                crate::bail_dim_basis!(
                    "term '{}' feature column {} out of bounds for {} columns",
                    term.name,
                    feature_col,
                    data.ncols()
                );
            }
            let mut spec_local = spec.clone();
            if term.shape != ShapeConstraint::None {
                // Shape-constrained B-splines are anchored by construction.
                // Sum-to-zero side constraints conflict with monotonic/convex cones.
                spec_local.identifiability = BSplineIdentifiability::None;
            }
            // Boundary conditions are emitted by the smooth-level paired
            // linear-constraint path (`bspline_boundary_linear_constraints`),
            // which supports non-zero anchors and composes them with the frozen
            // identifiability transform. Clear them here so the basis builder
            // does not also bake them into the basis null space — the legacy
            // basis-level path rejected non-zero anchors and dropped columns
            // before the frozen transform, conflicting with the constraint path.
            spec_local.boundary_conditions = BSplineBoundaryConditions::default();
            build_bspline_basis_1d(data.column(*feature_col), &spec_local)?
        }
        SmoothBasisSpec::ThinPlate {
            feature_cols,
            spec,
            input_scales,
        } => {
            if term.shape != ShapeConstraint::None {
                if feature_cols.len() != 1 {
                    crate::bail_invalid_basis!(
                        "ShapeConstraint::{:?} for term '{}' on ThinPlate basis requires exactly 1 feature axis; found {}",
                        term.shape,
                        term.name,
                        feature_cols.len()
                    );
                }
                shape_axis_col = Some(feature_cols[0]);
            }
            let mut x = select_columns(data, feature_cols)?;
            // Auto-standardize multivariate inputs: use stored scales (prediction)
            // or compute fresh ones (training). Same standardization-vs-
            // length-scale compensation as Matérn / hybrid Duchon: divide
            // the user's L by σ_geom so kernel(‖x_std − c_std‖/L_eff)
            // matches the original-coord kernel for uniform σ.
            let (scales, length_scale_eff) = if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
                (
                    Some(s.clone()),
                    compensate_length_scale_for_standardization(spec.length_scale, s),
                )
            } else if let Some(s) = compute_spatial_input_scales(x.view()) {
                apply_input_standardization(&mut x, &s);
                let l_eff = compensate_length_scale_for_standardization(spec.length_scale, &s);
                (Some(s), l_eff)
            } else {
                (None, spec.length_scale)
            };
            let mut spec_local = spec.clone();
            spec_local.length_scale = length_scale_eff;
            if matches!(
                spec_local.identifiability,
                SpatialIdentifiability::OrthogonalToParametric
            ) {
                spec_local.identifiability = SpatialIdentifiability::None;
            }
            let mut result = build_thin_plate_basis(x.view(), &spec_local).map_err(|err| {
                rewrite_thin_plate_knots_error(err, &term.name, feature_cols.len(), spec)
            })?;
            // Inject input scales into metadata; also restore the user's
            // original length_scale (not the σ_geom-compensated one) so a
            // metadata-driven rebuild that re-applies compensation does not
            // double-divide. The build may auto-promote to Duchon when
            // canonical TPS is infeasible (k < polynomial-nullspace size);
            // in that case patch the Duchon metadata variant so predict-time
            // round-trips through the same standardized data path.
            match &mut result.metadata {
                BasisMetadata::ThinPlate {
                    input_scales: ms,
                    length_scale,
                    ..
                } => {
                    *ms = scales;
                    *length_scale = spec.length_scale;
                }
                BasisMetadata::Duchon {
                    input_scales: ms,
                    length_scale,
                    ..
                } => {
                    *ms = scales;
                    // The ThinPlate auto-promotion path delegates to
                    // `build_duchon_basis` with `Some(spec_local.length_scale)`,
                    // which is the σ_geom-compensated value. The metadata
                    // therefore records the compensated kernel range, but the
                    // freeze→replay round trip plugs that value back into a
                    // user-facing `DuchonBasisSpec.length_scale` whose builder
                    // applies the σ_geom compensation a second time. Restore
                    // the user-facing scale here so replay re-compensates
                    // exactly once and reproduces the realized fit-time basis.
                    *length_scale = Some(spec.length_scale);
                }
                _ => {}
            }
            result
        }
        SmoothBasisSpec::Sphere { feature_cols, spec } => {
            if term.shape != ShapeConstraint::None {
                crate::bail_invalid_basis!(
                    "ShapeConstraint::{:?} for term '{}' is not supported on spherical splines",
                    term.shape,
                    term.name
                );
            }
            let x = select_columns(data, feature_cols)?;
            build_spherical_spline_basis(x.view(), spec)?
        }
        SmoothBasisSpec::ConstantCurvature { feature_cols, spec } => {
            if term.shape != ShapeConstraint::None {
                crate::bail_invalid_basis!(
                    "ShapeConstraint::{:?} for term '{}' is not supported on constant-curvature smooths",
                    term.shape,
                    term.name
                );
            }
            // Chart coordinates are consumed verbatim: NO auto-standardization.
            // Rescaling axes would change the chart gauge `1 + κ‖x‖²` and
            // silently redefine which curvature κ refers to (the same point
            // cloud at a different chart scale has a different κ̂); the user's
            // coordinates ARE the geometry here, exactly as for the sphere
            // smooth's (lat, lon).
            let x = select_columns(data, feature_cols)?;
            build_constant_curvature_basis(x.view(), spec)?
        }
        SmoothBasisSpec::MeasureJet {
            feature_cols,
            spec,
            input_scales,
        } => {
            if term.shape != ShapeConstraint::None {
                crate::bail_invalid_basis!(
                    "ShapeConstraint::{:?} for term '{}' is not supported on measure-jet smooths",
                    term.shape,
                    term.name
                );
            }
            let mut x = select_columns(data, feature_cols)?;
            // Matern-style per-axis standardization; the realized σ vector is
            // persisted into the metadata for predict-time replay.
            //
            // Length-scale round-trip contract (owning statement; the freeze
            // and frozen-validation arms reference it): `input_scales: Some`
            // marks the REPLAY path — the frozen length_scale is already the
            // realized post-standardization value and passes through
            // verbatim. Fresh path: an explicit user length_scale is in
            // ORIGINAL coordinates and gets the σ_geom compensation; the 0.0
            // auto sentinel passes through (auto-derivation runs inside the
            // builder, post-standardization).
            let (scales, length_scale_eff) = if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
                (Some(s.clone()), spec.length_scale)
            } else if let Some(s) = compute_spatial_input_scales(x.view()) {
                apply_input_standardization(&mut x, &s);
                let l_eff = if spec.length_scale > 0.0 {
                    compensate_length_scale_for_standardization(spec.length_scale, &s)
                } else {
                    spec.length_scale
                };
                (Some(s), l_eff)
            } else {
                (None, spec.length_scale)
            };
            let mut spec_local = spec.clone();
            spec_local.length_scale = length_scale_eff;
            let mut result = build_measure_jet_basis(x.view(), &spec_local)?;
            if let BasisMetadata::MeasureJet {
                input_scales: ms, ..
            } = &mut result.metadata
            {
                *ms = scales;
            }
            result
        }
        SmoothBasisSpec::Matern {
            feature_cols,
            spec,
            input_scales,
        } => {
            if term.shape != ShapeConstraint::None {
                if feature_cols.len() != 1 {
                    crate::bail_invalid_basis!(
                        "ShapeConstraint::{:?} for term '{}' on Matern basis requires exactly 1 feature axis; found {}",
                        term.shape,
                        term.name,
                        feature_cols.len()
                    );
                }
                shape_axis_col = Some(feature_cols[0]);
            }
            let mut x = select_columns(data, feature_cols)?;
            // Auto-standardization (per-axis division by σ_a) reinterprets
            // the user's `length_scale` from original data coordinates
            // into post-standardization coordinates: for uniform σ_a = σ,
            // `kernel(‖x_std − c_std‖/L)` equals `kernel(‖x − c‖/(σ·L))`,
            // so the effective kernel range shrinks by σ. To keep
            // `length_scale` consistently expressed in *original* data
            // coordinates regardless of axis variances, we standardize
            // and divide L by σ_geom = (∏σ_a)^(1/d). For uniform σ this
            // recovers the user's kernel exactly; for anisotropic data
            // the resulting per-axis effective scales σ_a / σ_geom are
            // the standard Mahalanobis preconditioning and preserve the
            // geometric-mean kernel range. Storing the σ vector in
            // metadata.input_scales makes the same transformation
            // replayable at predict time.
            let (scales, length_scale_eff) = if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
                (
                    Some(s.clone()),
                    compensate_length_scale_for_standardization(spec.length_scale, s),
                )
            } else if let Some(s) = compute_spatial_input_scales(x.view()) {
                apply_input_standardization(&mut x, &s);
                let l_eff = compensate_length_scale_for_standardization(spec.length_scale, &s);
                (Some(s), l_eff)
            } else {
                (None, spec.length_scale)
            };
            let mut spec_local = spec.clone();
            spec_local.length_scale = length_scale_eff;
            let mut result = build_matern_basiswithworkspace(x.view(), &spec_local, workspace)?;
            if let BasisMetadata::Matern {
                input_scales,
                length_scale,
                ..
            } = &mut result.metadata
            {
                *input_scales = scales;
                *length_scale = spec.length_scale;
            }
            result
        }
        SmoothBasisSpec::Duchon {
            feature_cols,
            spec,
            input_scales,
        } => {
            if term.shape != ShapeConstraint::None {
                if feature_cols.len() != 1 {
                    crate::bail_invalid_basis!(
                        "ShapeConstraint::{:?} for term '{}' on Duchon basis requires exactly 1 feature axis; found {}",
                        term.shape,
                        term.name,
                        feature_cols.len()
                    );
                }
                shape_axis_col = Some(feature_cols[0]);
            }
            let mut x = select_columns(data, feature_cols)?;
            // Hybrid Duchon (length_scale=Some) is governed by the same
            // standardization-vs-length-scale equivalence as Matérn: the
            // user's `length_scale` is interpreted in original data
            // coordinates, but auto-standardization (per-axis division by
            // σ_a) reinterprets it as σ_geom · L. Pre-multiply by 1/σ_geom
            // so kernel(‖x_std − c_std‖/L_eff) reproduces the user's
            // original-coord kernel exactly for uniform σ_a, and reduces
            // to standard Mahalanobis preconditioning for anisotropic σ.
            // Pure Duchon (length_scale=None) is scale-free and needs no
            // compensation.
            let (scales, length_scale_eff) = if let Some(s) = input_scales {
                apply_input_standardization(&mut x, s);
                (
                    Some(s.clone()),
                    compensate_optional_length_scale_for_standardization(spec.length_scale, s),
                )
            } else if let Some(s) = compute_spatial_input_scales(x.view()) {
                apply_input_standardization(&mut x, &s);
                let l_eff =
                    compensate_optional_length_scale_for_standardization(spec.length_scale, &s);
                (Some(s), l_eff)
            } else {
                (None, spec.length_scale)
            };
            let mut spec_local = spec.clone();
            spec_local.length_scale = length_scale_eff;
            if matches!(
                spec_local.identifiability,
                SpatialIdentifiability::OrthogonalToParametric
            ) {
                spec_local.identifiability = SpatialIdentifiability::None;
            }
            let mut result = build_duchon_basiswithworkspace(x.view(), &spec_local, workspace)?;
            if let BasisMetadata::Duchon {
                input_scales,
                length_scale,
                ..
            } = &mut result.metadata
            {
                *input_scales = scales;
                *length_scale = spec.length_scale;
            }
            result
        }
        SmoothBasisSpec::Pca {
            feature_cols,
            basis_matrix,
            centered,
            smooth_penalty,
            center_mean,
            pca_basis_path,
            chunk_size,
        } => {
            if term.shape != ShapeConstraint::None {
                crate::bail_invalid_basis!(
                    "ShapeConstraint::{:?} for term '{}' is not supported on Pca basis",
                    term.shape,
                    term.name
                );
            }
            build_pca_smooth_basis(
                data,
                feature_cols,
                basis_matrix,
                *centered,
                *smooth_penalty,
                center_mean.as_ref(),
                pca_basis_path.as_ref(),
                *chunk_size,
            )?
        }
        SmoothBasisSpec::TensorBSpline { feature_cols, spec } => {
            build_tensor_bspline_basis(data, feature_cols, spec)?
        }
        SmoothBasisSpec::ByVariable { .. } => {
            crate::bail_invalid_basis!(
                "internal: ByVariable smooths must return before inner basis dispatch"
            );
        }
        SmoothBasisSpec::BySmooth { .. } => {
            crate::bail_invalid_basis!("internal: BySmooth smooths must be lowered to ByVariable before inner basis dispatch"
                    .to_string(),);
        }
        SmoothBasisSpec::FactorSmooth { spec } => {
            if term.shape != ShapeConstraint::None {
                crate::bail_invalid_basis!(
                    "ShapeConstraint::{:?} is unsupported for factor smooth term '{}'",
                    term.shape,
                    term.name
                );
            }
            return build_factor_smooth(data, spec, &term.name, workspace);
        }
    };

    if let SmoothBasisSpec::Matern { .. } = &term.basis {
        let (penalties, nullspace_dims, penaltyinfo) =
            matern_operator_penalty_triplet_from_metadata(&built.metadata)?;
        built.penalties = penalties;
        built.nullspace_dims = nullspace_dims;
        built.penaltyinfo = penaltyinfo;
    }

    let p_local = built.design.ncols();
    let mut metadata = built.metadata.clone();
    // Extract factored Kronecker representation before consuming fields.
    // Invalidate it if shape transforms will be applied (they break structure).
    let kron_factored = if term.shape == ShapeConstraint::None {
        built.kronecker_factored
    } else {
        None
    };
    let mut design_t = built.design;
    let mut penalties_t: Vec<Array2<f64>> = built.penalties;
    // Ops vector parallel to `penalties_t`. Survives unchanged through the
    // identity path; nulled element-wise when `T^T S T` reparametrization
    // is applied (operator no longer bit-equivalent to the transformed
    // matrix); wrapped in `ScaledPenaltyOp` after Frobenius normalization.
    let mut ops_t: Vec<Option<std::sync::Arc<dyn crate::terms::penalty_op::PenaltyOp>>> = built.ops;
    if matches!(
        spatial_identifiability_policy(term),
        Some(SpatialIdentifiability::OrthogonalToParametric)
    ) {
        metadata = freeze_raw_spatial_metadata(metadata, design_t.ncols());
    }

    let active_penaltyinfo_t = built
        .penaltyinfo
        .iter()
        .filter(|info| info.active)
        .cloned()
        .collect::<Vec<_>>();
    let pre_dropped_penaltyinfo_t = built
        .penaltyinfo
        .iter()
        .filter(|info| !info.active)
        .cloned()
        .collect::<Vec<_>>();
    let use_box_reparam =
        term.shape != ShapeConstraint::None && shape_uses_box_reparameterization(&term.basis);
    let mut coefficient_transform_for_constraints: Option<Array2<f64>> = None;
    if let Some((order, sign)) = shape_order_and_sign(term.shape)
        && use_box_reparam
    {
        // Order 1 (monotone): the plain first-difference cone θ_{i+1}−θ_i ≥ 0 is
        // the control-polygon monotonicity criterion, which is independent of
        // Greville-abscissa spacing (it only fixes the *sign* of consecutive
        // control-point gaps), so the integer-difference transform is exact.
        //
        // Order 2 (convex/concave): the plain second-difference cone is only
        // correct for evenly spaced Greville abscissae. gam's B-splines are
        // clamped (and may use quantile knots), so the abscissae are not
        // uniform and the geometrically-correct cone is the second *divided*
        // difference. Build the Greville-scaled transform so γ_{≥2} ≥ 0
        // certifies convexity of the function, not of the raw coefficient
        // index. Periodic B-splines use uniform interior knots (uniform
        // abscissae), where the divided differences coincide with the integer
        // differences up to scale, so the plain path stays exact there.
        let t = if order == 2 {
            let bspline_meta = match &metadata {
                BasisMetadata::BSpline1D {
                    knots,
                    degree,
                    periodic,
                    ..
                } if periodic.is_none() => Some((knots.clone(), degree.unwrap_or(0))),
                _ => None,
            };
            match bspline_meta {
                Some((knots, degree)) if degree >= 1 => {
                    let greville = crate::basis::compute_greville_abscissae(&knots, degree)?;
                    if greville.len() != p_local {
                        crate::bail_invalid_basis!(
                            "shape-constraint Greville abscissae count {} does not match basis dim {} for term '{}'",
                            greville.len(),
                            p_local,
                            term.name
                        );
                    }
                    convex_divided_difference_transform_matrix(&greville, sign)?
                }
                _ => cumulative_sum_transform_matrix(p_local, order, sign),
            }
        } else {
            cumulative_sum_transform_matrix(p_local, order, sign)
        };
        coefficient_transform_for_constraints = Some(t.clone());
        // Coefficient-side transform: wrap the design in an operator that
        // applies T on the coefficient side, preserving sparsity/operator
        // structure of the inner design.
        let inner_dense = match design_t {
            DesignMatrix::Dense(d) => d,
            DesignMatrix::Sparse(sp) => crate::matrix::DenseDesignMatrix::from(
                sp.try_to_dense_arc("shape-constrained coefficient transform")
                    .map_err(BasisError::InvalidInput)?,
            ),
        };
        let coeff_op = crate::matrix::CoefficientTransformOperator::new(inner_dense, t.clone())
            .map_err(|e| BasisError::InvalidInput(format!("CoefficientTransformOperator: {e}")))?;
        design_t = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Arc::new(coeff_op)));
        if penalties_t.len() != active_penaltyinfo_t.len() {
            crate::bail_invalid_basis!(
                "internal box-reparam penalty/info mismatch for term '{}': penalties={}, infos={}",
                term.name,
                penalties_t.len(),
                active_penaltyinfo_t.len()
            );
        }
        // Wiggliness penalties undergo the exact congruence `S → TᵀST` (PSD
        // preserving). The double-penalty *nullspace shrinkage* ridge must NOT:
        // it is a unit-eigenvalue projector `ZZᵀ` onto null(S_wiggle) in the
        // β (B-spline coefficient) coordinates, and the congruence
        // `Tᵀ(ZZᵀ)T = (TᵀZ)(TᵀZ)ᵀ` is no longer a projector — its eigenvalues
        // blow up by the conditioning of the cumulative-sum `T` (cond(T) grows
        // with the basis dim), concentrating an enormous penalty on the leading
        // γ₀ "level" coordinate. REML then drives the shared λ to its ceiling
        // and the smooth collapses to a flat constant (#509, the over-smoothing
        // face). The principled fix keeps mgcv's double-penalty semantics in the
        // *reparametrized* space: rebuild the ridge as the unit-eigenvalue
        // nullspace projector of the transformed wiggliness penalty `TᵀST`, so
        // the double penalty shrinks exactly the unpenalized polynomial
        // directions of the γ-space smooth with eigenvalue 1, identical in
        // conditioning to the unconstrained fit.
        let transformed_wiggliness = penalties_t
            .iter()
            .zip(active_penaltyinfo_t.iter())
            .find(|(_, info)| !matches!(info.source, PenaltySource::DoublePenaltyNullspace))
            .map(|(s_local, _)| {
                let tt_s = fast_atb(&t, s_local);
                fast_ab(&tt_s, &t)
            });
        let mut rebuilt = Vec::with_capacity(penalties_t.len());
        for (s_local, info) in penalties_t.iter().zip(active_penaltyinfo_t.iter()) {
            if matches!(info.source, PenaltySource::DoublePenaltyNullspace) {
                let s_wiggle_t = transformed_wiggliness.as_ref().ok_or_else(|| {
                    BasisError::InvalidInput(format!(
                        "box-reparam term '{}' has a double-penalty ridge but no primary wiggliness penalty to derive its nullspace from",
                        term.name
                    ))
                })?;
                let ridge = crate::terms::basis::build_nullspace_shrinkage_penalty(s_wiggle_t)?
                    .map(|shrink| shrink.sym_penalty)
                    .unwrap_or_else(|| Array2::<f64>::zeros((p_local, p_local)));
                rebuilt.push(ridge);
            } else {
                let tt_s = fast_atb(&t, s_local);
                rebuilt.push(fast_ab(&tt_s, &t));
            }
        }
        penalties_t = rebuilt;
        // T^T S T (and the rebuilt γ-space ridge) invalidate op-form
        // bit-equivalence; drop ops here.
        ops_t = vec![None; penalties_t.len()];
    }
    if penalties_t.len() != active_penaltyinfo_t.len() {
        crate::bail_invalid_basis!(
            "internal penalty metadata mismatch for term '{}': active penalties={}, active infos={}",
            term.name,
            penalties_t.len(),
            active_penaltyinfo_t.len()
        );
    }
    if ops_t.len() != penalties_t.len() {
        ops_t = vec![None; penalties_t.len()];
    }
    let penalty_candidates = penalties_t
        .into_iter()
        .zip(active_penaltyinfo_t.into_iter())
        .zip(ops_t.into_iter())
        .map(
            |((matrix, info), op_in)| -> Result<PenaltyCandidate, BasisError> {
                let (matrix, c_new) = normalize_penalty_in_constrained_space(&matrix);
                let preserve_margin_scale =
                    matches!(&info.source, PenaltySource::TensorMarginal { .. });
                let (matrix, normalization_scale, op_scale, kronecker_scale) =
                    if preserve_margin_scale {
                        (
                            matrix.mapv(|v| v * c_new),
                            info.normalization_scale,
                            1.0,
                            1.0,
                        )
                    } else {
                        (
                            matrix,
                            info.normalization_scale * c_new,
                            1.0 / c_new,
                            1.0 / c_new,
                        )
                    };
                // Frobenius rescale: wrap inner op in `ScaledPenaltyOp(1/c_new)`
                // so `op.as_dense() == matrix` post-normalization.
                let scaled_op = if op_scale > 0.0 && op_scale.is_finite() {
                    op_in.map(|op| {
                        std::sync::Arc::new(crate::terms::penalty_op::ScaledPenaltyOp::new(
                            op, op_scale,
                        ))
                            as std::sync::Arc<dyn crate::terms::penalty_op::PenaltyOp>
                    })
                } else {
                    None
                };
                let kronecker_factors = info.kronecker_factors.map(|mut factors| {
                    if let Some(first) = factors.first_mut() {
                        first.mapv_inplace(|v| v * kronecker_scale);
                    }
                    factors
                });
                Ok(PenaltyCandidate {
                    nullspace_dim_hint: info.nullspace_dim_hint,
                    matrix,
                    source: info.source,
                    normalization_scale,
                    kronecker_factors,
                    op: scaled_op,
                })
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    let (penalties_t, nullspaces_t, penaltyinfo_t, null_eigenvectors_t, ops_t) =
        crate::terms::basis::filter_active_penalty_candidates_with_ops(penalty_candidates)?;
    let shape_linear_constraints = if term.shape != ShapeConstraint::None && !use_box_reparam {
        let axis = shape_axis_col.ok_or_else(|| {
            BasisError::InvalidInput(format!(
                "internal shape-constraint axis missing for term '{}'",
                term.name
            ))
        })?;
        let (x_shape_eval, design_shape_eval) =
            build_shape_constraint_design_1d(data, term, &metadata, axis)?;
        build_shape_linear_constraints_1d(
            x_shape_eval.view(),
            design_shape_eval.view(),
            term.shape,
        )?
    } else {
        None
    };
    let boundary_linear_constraints = match &term.basis {
        SmoothBasisSpec::BSpline1D { spec, .. } => bspline_boundary_linear_constraints(
            spec.boundary_conditions,
            &metadata,
            spec.degree,
            coefficient_transform_for_constraints.as_ref(),
        )?,
        _ => None,
    };
    let linear_constraints_local =
        merge_linear_constraints_global(shape_linear_constraints, boundary_linear_constraints);

    // Joint-null absorption rotation. Fresh fit specs compute Q from the final
    // per-smooth penalty set (after all in-smooth reparameterizations have
    // already been applied). Frozen specs already carry the complete realized
    // coefficient chart in their `FrozenTransform`; recomputing Q there would
    // rotate an already-frozen chart a second time and desynchronize value
    // rebuilds from derivative operators.
    //
    // Kronecker-factored smooths (tensor B-splines under `TensorBSplineIdentifiability::None`)
    // carry their joint penalty as `Σ_d S_d` with `S_d = I ⊗ … ⊗ S_d^{1D} ⊗ … ⊗ I`.
    // The joint null space is the tensor of marginal nulls and is handled directly
    // by the REML runtime's `kronecker_penalty_system` path (see
    // `runtime.rs:8334-8344`). Applying a dense (p × p) Q here would densify
    // `X_raw = mx ⊗ my` into `X_raw · Q`, destroying the Kronecker product
    // structure that the runtime relies on for fast log-det/derivative
    // assembly — and the rotation block at the wrapper site also unconditionally
    // wipes `kronecker_factored`, leaving the runtime to fall back to the
    // dense per-block log-det. Skip the rotation for Kronecker-factored terms
    // so the factored representation survives end-to-end.
    let joint_null_rotation = match term.joint_null_rotation.clone() {
        Some(persisted) => Some(persisted),
        None if smooth_has_frozen_identifiability(term) => None,
        None if kron_factored.is_some() => None,
        None => crate::terms::basis::compute_joint_null_rotation(&penalties_t)?,
    };

    Ok(LocalSmoothTermBuild {
        dim: p_local,
        design: design_t,
        penalties: penalties_t,
        ops: ops_t,
        nullspaces: nullspaces_t,
        null_eigenvectors: null_eigenvectors_t,
        joint_null_rotation,
        penaltyinfo: penaltyinfo_t,
        pre_dropped_penaltyinfo: pre_dropped_penaltyinfo_t,
        metadata,
        linear_constraints: linear_constraints_local,
        box_reparam: use_box_reparam,
        kronecker_factored: kron_factored,
    })
}


pub fn build_smooth_design(
    data: ArrayView2<'_, f64>,
    terms: &[SmoothTermSpec],
) -> Result<RawSmoothDesign, BasisError> {
    let mut ws = crate::basis::BasisWorkspace::new();
    build_smooth_design_withworkspace(data, terms, &mut ws)
}


/// Like `build_smooth_design`, but honors the caller workspace policy while
/// building each planned smooth term with an independent per-term workspace.
///
/// Independent workspaces avoid shared mutable distance-cache state during the
/// parallel term build; the final design, penalties, and metadata are assembled
/// in the original smooth-term order.
pub fn build_smooth_design_withworkspace(
    data: ArrayView2<'_, f64>,
    terms: &[SmoothTermSpec],
    workspace: &mut crate::basis::BasisWorkspace,
) -> Result<RawSmoothDesign, BasisError> {
    validate_smooth_terms_finite_inputs(data, terms)?;
    build_smooth_design_withworkspace_unvalidated(data, terms, workspace)
}


pub(crate) fn build_smooth_design_withworkspace_unvalidated(
    data: ArrayView2<'_, f64>,
    terms: &[SmoothTermSpec],
    workspace: &mut crate::basis::BasisWorkspace,
) -> Result<RawSmoothDesign, BasisError> {
    let mut planned_blocks = plan_joint_spatial_centers_for_term_blocks(data, &[terms.to_vec()])?;
    let planned_terms = planned_blocks.pop().ok_or_else(|| {
        BasisError::InvalidInput(
            "joint spatial center planner returned no smooth blocks".to_string(),
        )
    })?;
    let policy = workspace.policy().clone();
    let local_builds: Vec<LocalSmoothTermBuild> = {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        planned_terms
            .into_par_iter()
            .map(|term| {
                let mut term_workspace = crate::basis::BasisWorkspace::with_policy(policy.clone());
                build_single_local_smooth_term(data, &term, &mut term_workspace)
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    let total_p: usize = local_builds.iter().map(|built| built.dim).sum();

    let mut local_designs: Vec<DesignMatrix> = Vec::with_capacity(local_builds.len());
    let mut terms_out = Vec::<SmoothTerm>::with_capacity(terms.len());
    let mut penalties_global = Vec::<BlockwisePenalty>::new();
    let mut nullspace_dims_global = Vec::<usize>::new();
    let mut penaltyinfo_global = Vec::<PenaltyBlockInfo>::new();
    let mut dropped_penaltyinfo_global = Vec::<DroppedPenaltyBlockInfo>::new();
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(total_p, f64::NEG_INFINITY);
    let mut any_bounds = false;
    // Each linear-constraint row only touches the current term's column slice.
    // Track `(col_start, col_end, local_row_values)` and assemble the final
    // dense `Array2` in one pass, avoiding per-row `Array1::zeros(total_p)`
    // allocation plus a row-by-row copy at the end.
    let mut linear_constraintsrows: Vec<(usize, usize, Array1<f64>)> = Vec::new();
    let mut linear_constraints_b: Vec<f64> = Vec::new();

    let mut col_start = 0usize;
    for (term, mut built) in terms.iter().zip(local_builds.into_iter()) {
        let p_local = built.dim;
        let col_end = col_start + p_local;
        let lb_local = if built.box_reparam {
            shape_lower_bounds_local(term.shape, p_local)
        } else {
            None
        };

        // Stage-2 joint-null absorption rotation. Fired *before* the
        // penalty / design / global aggregation loops below so that every
        // subsequent reference to `built.penalties`, `built.design`, and
        // `built.ops` sees the post-rotation values.
        //
        // The math: when the smooth's joint penalty `Σ_k S_k` has a
        // non-trivial null space, eigh selects `Q = [U_range | U_null]`
        // with null columns at the tail. Setting `β_raw = Q · γ` and
        // applying:
        //     design        ← X · Q
        //     penalties[k]  ← Qᵀ · S_k · Q   (block-diag, zero null tail)
        // yields a model whose fitted γ is invariant to the rotation
        // (since likelihood depends only on `X · β_raw = X · Q · γ`), but
        // whose penalty is full-rank on the range columns. The large-scale
        // failing case (cert refusal in the joint-Newton inner solve)
        // resolves because `H_pen = H_loglik + S` becomes full rank on
        // the smooth's range columns.
        //
        // Rotation is suppressed when the smooth carries coordinate-wise
        // shape constraints (`lb_local` or `built.linear_constraints`):
        // those encode a cone in the original coordinate system and a
        // general orthogonal rotation breaks the cone geometry. Smooths
        // with shape constraints typically have full-rank joint penalty
        // (their structural shape comes from the cone, not from null
        // directions in the penalty), so suppression is rarely a loss.
        //
        // `applied_rotation` carries the Q that was applied (or `None`
        // if no rotation fired). It is persisted onto `SmoothTerm` below
        // so prediction-side `X_new_raw · Q` replay can reproduce the
        // exact rotation. Persistence through the saved-model artifact
        // is a follow-up — see the doc on `SmoothTerm.joint_null_rotation`.
        let applied_rotation: Option<crate::terms::basis::JointNullRotation> = match (
            built.joint_null_rotation.take(),
            lb_local.is_some(),
            built.linear_constraints.is_some(),
        ) {
            (Some(rot), false, false) => {
                let q = &rot.rotation;
                let dense = built
                    .design
                    .try_to_dense_by_chunks("joint-null absorption rotation")
                    .map_err(BasisError::InvalidInput)?;
                let rotated = crate::linalg::faer_ndarray::fast_ab(&dense, q);
                built.design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(rotated));
                built.penalties = built
                    .penalties
                    .into_iter()
                    .map(|s_local| {
                        let qt_s = crate::linalg::faer_ndarray::fast_atb(q, &s_local);
                        crate::linalg::faer_ndarray::fast_ab(&qt_s, q)
                    })
                    .collect();
                built.ops = vec![None; built.penalties.len()];
                built.kronecker_factored = None;
                Some(rot)
            }
            (Some(_), _, _) => None,
            (None, _, _) => None,
        };

        let activeinfos = built
            .penaltyinfo
            .iter()
            .filter(|info| info.active)
            .collect::<Vec<_>>();
        if activeinfos.len() != built.penalties.len() {
            crate::bail_invalid_basis!(
                "internal penalty info mismatch for term '{}': activeinfos={}, penalties={}",
                term.name,
                activeinfos.len(),
                built.penalties.len()
            );
        }
        for (((s_local, &ns), info), op_local) in built
            .penalties
            .iter()
            .zip(built.nullspaces.iter())
            .zip(activeinfos.into_iter())
            .zip(built.ops.iter())
        {
            let global_index = penalties_global.len();
            penalties_global.push(
                BlockwisePenalty::new(col_start..col_end, s_local.clone())
                    .with_op(op_local.clone()),
            );
            nullspace_dims_global.push(ns);
            let mut penalty = info.clone();
            penalty.nullspace_dim_hint = ns;
            penaltyinfo_global.push(PenaltyBlockInfo {
                global_index,
                termname: Some(term.name.clone()),
                penalty,
            });
        }
        for info in built.penaltyinfo.iter().filter(|info| !info.active) {
            dropped_penaltyinfo_global.push(DroppedPenaltyBlockInfo {
                termname: Some(term.name.clone()),
                penalty: info.clone(),
            });
        }
        for info in &built.pre_dropped_penaltyinfo {
            dropped_penaltyinfo_global.push(DroppedPenaltyBlockInfo {
                termname: Some(term.name.clone()),
                penalty: info.clone(),
            });
        }

        if let Some(lin_local) = &built.linear_constraints {
            for r in 0..lin_local.a.nrows() {
                linear_constraintsrows.push((col_start, col_end, lin_local.a.row(r).to_owned()));
                linear_constraints_b.push(lin_local.b[r]);
            }
        }
        if let Some(lb_local) = &lb_local {
            coefficient_lower_bounds
                .slice_mut(s![col_start..col_end])
                .assign(lb_local);
            any_bounds = true;
        }

        // Move the per-term design out of `built` rather than cloning it.
        local_designs.push(built.design);

        terms_out.push(SmoothTerm {
            name: term.name.clone(),
            coeff_range: col_start..col_end,
            shape: term.shape,
            penalties_local: built.penalties,
            nullspace_dims: built.nullspaces,
            penaltyinfo_local: built.penaltyinfo,
            metadata: built.metadata,
            lower_bounds_local: lb_local,
            linear_constraints_local: built.linear_constraints,
            kronecker_factored: built.kronecker_factored.take(),
            joint_null_rotation: applied_rotation,
            unabsorbed_global_orthogonality: None,
        });

        col_start = col_end;
    }

    assert_eq!(
        penalties_global.len(),
        nullspace_dims_global.len(),
        "global smooth penalty/nullspace bookkeeping diverged"
    );
    assert_eq!(
        penalties_global.len(),
        penaltyinfo_global.len(),
        "global smooth penalty metadata bookkeeping diverged"
    );

    Ok(RawSmoothDesign {
        term_designs: local_designs,
        penalties: penalties_global,
        nullspace_dims: nullspace_dims_global,
        penaltyinfo: penaltyinfo_global,
        dropped_penaltyinfo: dropped_penaltyinfo_global,
        terms: terms_out,
        coefficient_lower_bounds: if any_bounds {
            Some(coefficient_lower_bounds)
        } else {
            None
        },
        linear_constraints: if linear_constraintsrows.is_empty() {
            None
        } else {
            let mut a = Array2::<f64>::zeros((linear_constraintsrows.len(), total_p));
            for (i, (cs, ce, values)) in linear_constraintsrows.iter().enumerate() {
                a.row_mut(i).slice_mut(s![*cs..*ce]).assign(values);
            }
            Some(LinearInequalityConstraints {
                a,
                b: Array1::from_vec(linear_constraints_b),
            })
        },
    })
}

fn build_term_collection_design_inner(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
) -> Result<TermCollectionDesign, BasisError> {
    use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

    let n = data.nrows();
    let p_data = data.ncols();
    let p_intercept = 1usize;
    let p_lin = spec.linear_terms.len();

    // Smooth construction, random-effect construction, and linear-column
    // extraction are independent at this stage. Run them concurrently, but keep
    // each result in spec order so the final global layout remains stable:
    // [intercept | linear | random_effects | smooth].
    let (smooth_raw_result, (random_blocks_result, linear_block_result)) = rayon::join(
        || {
            let mut ws = crate::basis::BasisWorkspace::new();
            build_smooth_design_withworkspace_unvalidated(data, &spec.smooth_terms, &mut ws)
        },
        || {
            rayon::join(
                || {
                    spec.random_effect_terms
                        .par_iter()
                        .map(|term| build_random_effect_block(data, term))
                        .collect::<Result<Vec<_>, _>>()
                },
                || -> Result<Option<Array2<f64>>, BasisError> {
                    if p_lin == 0 {
                        return Ok(None);
                    }

                    let linear_columns = (0..p_lin)
                        .into_par_iter()
                        .map(|j| {
                            let linear = &spec.linear_terms[j];
                            // `:` interactions carry multiple feature columns; the
                            // materialized column is their elementwise product (a
                            // plain main effect has a single column). Mirror
                            // `build_term_collection_fixed_blocks` so this path and
                            // the incremental realizer agree on every interaction.
                            let cols = linear.effective_feature_cols();
                            for &col in &cols {
                                if col >= p_data {
                                    crate::bail_dim_basis!(
                                        "linear term '{}' feature column {} out of bounds for {} columns",
                                        linear.name, col, p_data
                                    );
                                }
                            }
                            let mut column = data.column(cols[0]).to_owned();
                            for &c in cols.iter().skip(1) {
                                column *= &data.column(c);
                            }
                            Ok(column)
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    let mut out = Array2::<f64>::zeros((n, p_lin));
                    for (j, column) in linear_columns.iter().enumerate() {
                        out.column_mut(j).assign(column);
                    }
                    Ok(Some(out))
                },
            )
        },
    );

    let smooth_raw = smooth_raw_result?;
    let random_blocks = random_blocks_result?;
    let linear_block = linear_block_result?;

    let smooth = apply_global_smooth_identifiability(
        smooth_raw,
        data,
        &spec.linear_terms,
        &spec.smooth_terms,
    )?;

    let p_rand: usize = random_blocks.iter().map(|b| b.num_groups).sum();
    let p_smooth = smooth.total_smooth_cols();
    let p_total = p_intercept + p_lin + p_rand + p_smooth;

    let mut linear_ranges = Vec::<(String, Range<usize>)>::with_capacity(p_lin);
    for (j, linear) in spec.linear_terms.iter().enumerate() {
        let col = p_intercept + j;
        // Column ranges are in the global (full) coordinate system:
        // [intercept | linear | random_effects | smooth]
        linear_ranges.push((linear.name.clone(), col..(col + 1)));
    }

    // Track random-effect column ranges in the global coordinate system.
    // Global layout: [intercept(1) | linear(p_lin) | RE_0(q0) | RE_1(q1) | … | smooth(p_smooth)]
    let mut random_effect_ranges =
        Vec::<(String, Range<usize>)>::with_capacity(random_blocks.len());
    let mut random_effect_levels = Vec::<(String, Vec<u64>)>::with_capacity(random_blocks.len());
    let mut col_cursor = p_intercept + p_lin;
    for block in &random_blocks {
        let q = block.num_groups;
        let end = col_cursor + q;
        random_effect_ranges.push((block.name.clone(), col_cursor..end));
        random_effect_levels.push((block.name.clone(), block.kept_levels.clone()));
        col_cursor = end;
    }

    // ── Assemble the full DesignMatrix ────────────────────────────────
    //
    // Always use a BlockDesignOperator with per-term blocks.  The full
    // (n, p_total) dense matrix is NEVER materialized:
    //
    //   Block 0:     Intercept — zero storage, implicit all-ones column
    //   Block 1:     Linear terms — (n, p_lin) extracted from data
    //   Blocks 2..k: Random-effect operators — O(n) one-hot, no dense storage
    //   Blocks k+1..: Per-smooth-term dense blocks — each (n, p_term)
    //
    // Splitting smooth terms into per-term blocks means cross-block grams
    // are small O(p_i × p_j) BLAS operations.  Tensor product terms with
    // Kronecker structure become DesignBlock::Operator, avoiding the full
    // n × ∏q_j materialization.

    let mut blocks = Vec::<DesignBlock>::new();

    // Block 0: intercept — zero storage.
    blocks.push(DesignBlock::Intercept(n));

    // Block 1: linear terms.
    if let Some(lin_block) = linear_block {
        blocks.push(DesignBlock::Dense(crate::matrix::DenseDesignMatrix::from(
            lin_block,
        )));
    }

    // Blocks: random-effect operators — O(n) implicit one-hot.
    for block in &random_blocks {
        let re_op = RandomEffectOperator::new(block.group_ids.clone(), block.num_groups);
        blocks.push(DesignBlock::RandomEffect(Arc::new(re_op)));
    }

    // Blocks: per-smooth-term.  Each smooth term gets its own block so that
    // cross-block grams are tiny. Tensor terms can stay operator-backed all
    // the way from basis construction; dense smooth terms stay dense.
    if p_smooth > 0 {
        for term_design in &smooth.term_designs {
            match term_design {
                DesignMatrix::Dense(dense) => blocks.push(DesignBlock::Dense(dense.clone())),
                DesignMatrix::Sparse(sparse) => blocks.push(DesignBlock::Sparse(sparse.clone())),
            }
        }
    }

    let design = assemble_term_collection_design_matrix(blocks)?;

    let mut penalties = Vec::<BlockwisePenalty>::new();
    let mut nullspace_dims = Vec::<usize>::new();
    let mut penaltyinfo = Vec::<PenaltyBlockInfo>::new();
    let mut dropped_penaltyinfo = Vec::<DroppedPenaltyBlockInfo>::new();
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(p_total, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraintrows = Vec::<Array1<f64>>::new();
    let mut linear_constraint_b = Vec::<f64>::new();

    for (j, linear) in spec.linear_terms.iter().enumerate() {
        let col = p_intercept + j;
        if let Some(lb) = linear.coefficient_min {
            let mut row = Array1::<f64>::zeros(p_total);
            row[col] = 1.0;
            linear_constraintrows.push(row);
            linear_constraint_b.push(lb);
        }
        if let Some(ub) = linear.coefficient_max {
            let mut row = Array1::<f64>::zeros(p_total);
            row[col] = -1.0;
            linear_constraintrows.push(row);
            linear_constraint_b.push(-ub);
        }
    }

    // Linear terms with `double_penalty=true` are aggregated into a single
    // shared ridge block named "linear" that spans the union of their
    // coefficient columns. Two reasons this is the only correct shape:
    //
    //   (i) Per-term redundancy: a penalized linear term is a single scalar
    //   coefficient β_j; the only shrinkage direction is β_j² itself. The
    //   earlier per-term form emitted two identical 1×1 ridges per term
    //   (`LinearTermRidge` + `LinearTermDoubleRidge`); for the two λ's that
    //   the outer REML/LAML places on those blocks, the total penalty is
    //   `(λ₁ + λ₂) β_j²` and the outer score is flat along
    //   `λ₁ + λ₂ = const`. Unidentifiable hyperparameters by construction.
    //
    //   (ii) Cross-term aliasing: each penalized linear term in its own 1×1
    //   block gives the outer optimizer a separate λ per term, when the
    //   mgcv-equivalent "ridge over the parametric linear columns" is one
    //   shared λ that shrinks every penalized linear coefficient with the
    //   same scale. The shared block emits exactly one outer hyperparameter
    //   and pins each penalized column's diagonal entry to 1.0; unpenalized
    //   linear columns sit at diagonal 0, so they are mathematically
    //   unaffected (S β = 0 on those rows) while keeping the block range
    //   contiguous over the linear coefficient run.
    //
    // The block's `termname` is the literal "linear" so keyed hyperpriors
    // / coefficient groups address the shared lever directly. The
    // per-term identity is preserved through `linear_ranges`, which still
    // names each individual column.
    let any_double_penalty_linear = spec.linear_terms.iter().any(|t| t.double_penalty);
    if any_double_penalty_linear && p_lin > 0 {
        let block_range = p_intercept..(p_intercept + p_lin);
        let mut s = Array2::<f64>::zeros((p_lin, p_lin));
        let mut effective_rank = 0usize;
        for (j, linear) in spec.linear_terms.iter().enumerate() {
            if linear.double_penalty {
                s[[j, j]] = 1.0;
                effective_rank += 1;
            }
        }
        let global_index = penalties.len();
        penalties.push(BlockwisePenalty::new(block_range, s));
        nullspace_dims.push(0);
        penaltyinfo.push(PenaltyBlockInfo {
            global_index,
            termname: Some("linear".to_string()),
            penalty: PenaltyInfo {
                source: PenaltySource::Other("LinearTermRidge".to_string()),
                original_index: 0,
                active: true,
                effective_rank,
                dropped_reason: None,
                nullspace_dim_hint: 0,
                normalization_scale: 1.0,
                kronecker_factors: None,
            },
        });
    }

    for (re_idx, (name, range)) in random_effect_ranges.iter().enumerate() {
        if range.is_empty() || !spec.random_effect_terms[re_idx].penalized {
            continue;
        }
        let block_size = range.len();
        let global_index = penalties.len();
        penalties.push(BlockwisePenalty::ridge(range.clone(), 1.0));
        nullspace_dims.push(0);
        penaltyinfo.push(PenaltyBlockInfo {
            global_index,
            termname: Some(name.clone()),
            penalty: PenaltyInfo {
                source: PenaltySource::Other(format!("RandomEffectRidge({name})")),
                original_index: re_idx,
                active: true,
                effective_rank: block_size,
                dropped_reason: None,
                nullspace_dim_hint: 0,
                normalization_scale: 1.0,
                kronecker_factors: None,
            },
        });
    }

    if smooth.penaltyinfo.len() != smooth.penalties.len() {
        crate::bail_invalid_basis!(
            "smooth penalty metadata mismatch: penalties={}, metadata={}",
            smooth.penalties.len(),
            smooth.penaltyinfo.len()
        );
    }
    let smooth_start = p_intercept + p_lin + p_rand;
    for ((bp_smooth, &ns), localinfo) in smooth
        .penalties
        .iter()
        .zip(smooth.nullspace_dims.iter())
        .zip(smooth.penaltyinfo.iter())
    {
        let global_index = penalties.len();
        // Offset the per-term block range from smooth-local to model-global.
        let offset_range =
            (bp_smooth.col_range.start + smooth_start)..(bp_smooth.col_range.end + smooth_start);
        let bp = if let Some(factors) = localinfo.penalty.kronecker_factors.as_ref() {
            BlockwisePenalty::kronecker(offset_range, bp_smooth.local.clone(), factors.clone())
                .with_op(bp_smooth.op.clone())
        } else if matches!(
            localinfo.penalty.source,
            PenaltySource::Other(ref s) if s.starts_with("RandomEffectRidge")
        ) {
            BlockwisePenalty::ridge(offset_range, 1.0)
        } else {
            BlockwisePenalty::new(offset_range, bp_smooth.local.clone())
                .with_op(bp_smooth.op.clone())
        };
        penalties.push(bp);
        nullspace_dims.push(ns);
        let mut penalty = localinfo.penalty.clone();
        penalty.nullspace_dim_hint = ns;
        penaltyinfo.push(PenaltyBlockInfo {
            global_index,
            termname: localinfo.termname.clone(),
            penalty,
        });
    }
    dropped_penaltyinfo.extend(smooth.dropped_penaltyinfo.iter().cloned());

    assert_eq!(
        penalties.len(),
        nullspace_dims.len(),
        "term-collection penalty/nullspace bookkeeping diverged"
    );
    assert_eq!(
        penalties.len(),
        penaltyinfo.len(),
        "term-collection penalty metadata bookkeeping diverged"
    );

    if let Some(lb_smooth) = smooth.coefficient_lower_bounds.as_ref() {
        let start = p_intercept + p_lin + p_rand;
        coefficient_lower_bounds
            .slice_mut(s![start..(start + p_smooth)])
            .assign(lb_smooth);
        any_bounds = true;
    }
    if let Some(lin_smooth) = smooth.linear_constraints.as_ref() {
        let mut a_global = Array2::<f64>::zeros((lin_smooth.a.nrows(), p_total));
        let start = p_intercept + p_lin + p_rand;
        a_global
            .slice_mut(s![.., start..(start + p_smooth)])
            .assign(&lin_smooth.a);
        for r in 0..a_global.nrows() {
            linear_constraintrows.push(a_global.row(r).to_owned());
            linear_constraint_b.push(lin_smooth.b[r]);
        }
    }

    // Canonical constraint path: convert any explicit lower bounds into linear
    // inequalities and merge into the global constraint matrix. This keeps fitting
    // behavior independent of user-facing lower-bound options.
    let lower_bound_constraints = if any_bounds {
        linear_constraints_from_lower_bounds_global(&coefficient_lower_bounds)
    } else {
        None
    };
    let explicit_linear_constraints = if linear_constraintrows.is_empty() {
        None
    } else {
        let mut a = Array2::<f64>::zeros((linear_constraintrows.len(), p_total));
        for (i, row) in linear_constraintrows.iter().enumerate() {
            a.row_mut(i).assign(row);
        }
        Some(LinearInequalityConstraints {
            a,
            b: Array1::from_vec(linear_constraint_b),
        })
    };
    let linear_constraints =
        merge_linear_constraints_global(explicit_linear_constraints, lower_bound_constraints);

    Ok(TermCollectionDesign {
        design,
        penalties,
        nullspace_dims,
        penaltyinfo,
        dropped_penaltyinfo,
        coefficient_lower_bounds: if any_bounds {
            Some(coefficient_lower_bounds)
        } else {
            None
        },
        linear_constraints,
        intercept_range: 0..1,
        linear_ranges,
        random_effect_ranges,
        random_effect_levels,
        smooth,
    })
}


pub fn build_term_collection_design(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
) -> Result<TermCollectionDesign, BasisError> {
    validate_term_collection_finite_inputs(data, spec)?;
    let mut planned_specs =
        plan_joint_spatial_centers_for_term_blocks(data, &[spec.smooth_terms.clone()])?;
    let planned_smooth_terms = planned_specs.pop().ok_or_else(|| {
        BasisError::InvalidInput(
            "joint spatial center planner returned no smooth terms for single-spec build"
                .to_string(),
        )
    })?;
    let mut planned_spec = spec.clone();
    planned_spec.smooth_terms = planned_smooth_terms;
    build_term_collection_design_inner(data, &planned_spec)
}


pub fn build_term_collection_designs_joint(
    data: ArrayView2<'_, f64>,
    specs: &[TermCollectionSpec],
) -> Result<Vec<TermCollectionDesign>, BasisError> {
    for spec in specs {
        validate_term_collection_finite_inputs(data, spec)?;
    }
    let smooth_blocks = specs
        .iter()
        .map(|spec| spec.smooth_terms.clone())
        .collect::<Vec<_>>();
    let planned_blocks = plan_joint_spatial_centers_for_term_blocks(data, &smooth_blocks)?;
    let mut out = Vec::with_capacity(specs.len());
    for (spec, planned_terms) in specs.iter().zip(planned_blocks.into_iter()) {
        let mut planned_spec = spec.clone();
        planned_spec.smooth_terms = planned_terms;
        out.push(build_term_collection_design_inner(data, &planned_spec)?);
    }
    Ok(out)
}


pub fn build_term_collection_designs_and_freeze_joint(
    data: ArrayView2<'_, f64>,
    specs: &[TermCollectionSpec],
) -> Result<(Vec<TermCollectionDesign>, Vec<TermCollectionSpec>), EstimationError> {
    let designs = build_term_collection_designs_joint(data, specs)?;
    let mut resolved_specs = Vec::with_capacity(specs.len());
    for (spec, design) in specs.iter().zip(designs.iter()) {
        resolved_specs.push(freeze_term_collection_from_design(spec, design)?);
    }
    Ok((designs, resolved_specs))
}


fn smooth_basis_feature_cols(basis: &SmoothBasisSpec) -> Vec<usize> {
    match basis {
        SmoothBasisSpec::ByVariable { inner, by_col, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, by_col, .. } => {
            let mut cols = smooth_basis_feature_cols(inner);
            cols.push(*by_col);
            cols.sort_unstable();
            cols.dedup();
            cols
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => smooth_basis_feature_cols(smooth),
        SmoothBasisSpec::BSpline1D { feature_col, .. } => vec![*feature_col],
        SmoothBasisSpec::ThinPlate { feature_cols, .. }
        | SmoothBasisSpec::Sphere { feature_cols, .. }
        | SmoothBasisSpec::ConstantCurvature { feature_cols, .. }
        | SmoothBasisSpec::Matern { feature_cols, .. }
        | SmoothBasisSpec::MeasureJet { feature_cols, .. }
        | SmoothBasisSpec::Duchon { feature_cols, .. }
        | SmoothBasisSpec::Pca { feature_cols, .. }
        | SmoothBasisSpec::TensorBSpline { feature_cols, .. } => feature_cols.clone(),
        SmoothBasisSpec::FactorSmooth { spec } => {
            let mut cols = spec.continuous_cols.clone();
            cols.push(spec.group_col);
            cols.sort_unstable();
            cols.dedup();
            cols
        }
    }
}


pub fn smooth_term_feature_cols(term: &SmoothTermSpec) -> Vec<usize> {
    smooth_basis_feature_cols(&term.basis)
}


fn smooth_basis_family_rank(term: &SmoothTermSpec) -> u8 {
    match &term.basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            smooth_basis_family_rank(&SmoothTermSpec {
                name: term.name.clone(),
                basis: (**inner).clone(),
                shape: term.shape,
                joint_null_rotation: None,
            })
        }
        SmoothBasisSpec::BSpline1D { .. } => 0,
        SmoothBasisSpec::TensorBSpline { .. } => 1,
        SmoothBasisSpec::ThinPlate { .. } => 2,
        SmoothBasisSpec::Sphere { .. } => 3,
        SmoothBasisSpec::Matern { .. } => 4,
        SmoothBasisSpec::Duchon { .. } => 5,
        SmoothBasisSpec::Pca { .. } => 6,
        SmoothBasisSpec::ConstantCurvature { .. } => 8,
        SmoothBasisSpec::MeasureJet { .. } => 9,
        SmoothBasisSpec::BySmooth { smooth, .. } => smooth_basis_family_rank(&SmoothTermSpec {
            name: term.name.clone(),
            basis: (**smooth).clone(),
            shape: term.shape,
            joint_null_rotation: None,
        }),
        SmoothBasisSpec::FactorSmooth { .. } => 7,
    }
}


pub(crate) fn smooth_has_frozen_identifiability(term: &SmoothTermSpec) -> bool {
    match &term.basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            smooth_has_frozen_identifiability(&SmoothTermSpec {
                name: term.name.clone(),
                basis: (**inner).clone(),
                shape: term.shape,
                joint_null_rotation: None,
            })
        }
        SmoothBasisSpec::BSpline1D { spec, .. } => {
            matches!(
                spec.identifiability,
                BSplineIdentifiability::FrozenTransform { .. }
            )
        }
        SmoothBasisSpec::ThinPlate { spec, .. } => matches!(
            spec.identifiability,
            SpatialIdentifiability::FrozenTransform { .. }
        ),
        SmoothBasisSpec::Sphere { spec, .. } => {
            matches!(spec.center_strategy, CenterStrategy::UserProvided(_))
                || matches!(
                    spec.identifiability,
                    SphericalSplineIdentifiability::FrozenTransform { .. }
                )
        }
        SmoothBasisSpec::ConstantCurvature { spec, .. } => {
            matches!(spec.center_strategy, CenterStrategy::UserProvided(_))
                || matches!(
                    spec.identifiability,
                    ConstantCurvatureIdentifiability::FrozenTransform { .. }
                )
        }
        SmoothBasisSpec::MeasureJet { spec, .. } => {
            matches!(spec.center_strategy, CenterStrategy::UserProvided(_))
                || matches!(
                    spec.identifiability,
                    MeasureJetIdentifiability::FrozenTransform { .. }
                )
        }
        SmoothBasisSpec::Matern { spec, .. } => matches!(
            spec.identifiability,
            MaternIdentifiability::FrozenTransform { .. }
        ),
        SmoothBasisSpec::BySmooth { by_kind, .. } => match by_kind {
            ByVarKind::Factor { frozen_levels, .. } => frozen_levels.is_some(),
            ByVarKind::Numeric { .. } => true,
        },
        SmoothBasisSpec::FactorSmooth { spec } => spec.group_frozen_levels.is_some(),
        SmoothBasisSpec::Duchon { spec, .. } => matches!(
            spec.identifiability,
            SpatialIdentifiability::FrozenTransform { .. }
        ),
        SmoothBasisSpec::Pca {
            centered,
            center_mean,
            pca_basis_path,
            ..
        } => !*centered || center_mean.is_some() || pca_basis_path.is_some(),
        SmoothBasisSpec::TensorBSpline { spec, .. } => matches!(
            spec.identifiability,
            TensorBSplineIdentifiability::FrozenTransform { .. }
        ),
    }
}


fn compare_smooth_ownership_priority(
    lhs_idx: usize,
    lhs: &SmoothTermSpec,
    rhs_idx: usize,
    rhs: &SmoothTermSpec,
) -> std::cmp::Ordering {
    let lhs_cols = smooth_term_feature_cols(lhs);
    let rhs_cols = smooth_term_feature_cols(rhs);
    lhs_cols
        .len()
        .cmp(&rhs_cols.len())
        .then_with(|| lhs_cols.cmp(&rhs_cols))
        .then_with(|| smooth_basis_family_rank(lhs).cmp(&smooth_basis_family_rank(rhs)))
        .then_with(|| lhs.name.cmp(&rhs.name))
        .then(lhs_idx.cmp(&rhs_idx))
}


fn smooth_is_owned_by_prior_term(owner: &SmoothTermSpec, target: &SmoothTermSpec) -> bool {
    let owner_features = smooth_term_feature_cols(owner)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let target_features = smooth_term_feature_cols(target)
        .into_iter()
        .collect::<BTreeSet<_>>();
    owner_features.is_subset(&target_features)
}


/// Static (spec-only) description of the hierarchical smooth-ownership decomposition.
///
/// This is the single source of truth for the deterministic ownership policy that
/// [`apply_global_smooth_identifiability`] uses during the fit: the processing order of
/// smooth terms, the feature columns each term spans, the candidate lower-order owners of
/// each term (nested/duplicate feature sets), and the basis-family rank used as a
/// tie-breaker. The fit engine consumes this structure and additionally applies a numerical
/// cross-residual overlap test on the realized design columns; the CLI structure-warning
/// path consumes the same structure for diagnostic messages, so both paths agree on which
/// smooths own which subspaces.
pub struct SmoothStructureAnalysis {
    /// Smooth-term indices sorted into ownership-processing order (lowest priority first):
    /// lower-order / narrower smooths come first and own their subspaces.
    pub ownership_order: Vec<usize>,
    /// `term_feature_cols[idx]` are the sorted, deduplicated feature columns that smooth term
    /// `idx` spans (indexed by the original smooth-term index, not by `ownership_order`).
    pub term_feature_cols: Vec<Vec<usize>>,
    /// `term_owners[idx]` are the indices of prior (in `ownership_order`) smooth terms whose
    /// feature set is a subset of term `idx`'s feature set, i.e. candidate owners of `idx`.
    /// The list is given in ownership-processing order.
    pub term_owners: Vec<Vec<usize>>,
    /// `basis_family_ranks[idx]` is the basis-family ordering rank of smooth term `idx`.
    pub basis_family_ranks: Vec<u8>,
}


/// Compute the static hierarchical smooth-ownership decomposition from the smooth-term specs.
///
/// `smoothspecs` is the same slice that [`apply_global_smooth_identifiability`] receives.
pub fn analyze_smooth_ownership(smoothspecs: &[SmoothTermSpec]) -> SmoothStructureAnalysis {
    let term_feature_cols: Vec<Vec<usize>> =
        smoothspecs.iter().map(smooth_term_feature_cols).collect();
    let basis_family_ranks: Vec<u8> = smoothspecs.iter().map(smooth_basis_family_rank).collect();

    let mut ownership_order: Vec<usize> = (0..smoothspecs.len()).collect();
    ownership_order.sort_by(|&lhs, &rhs| {
        compare_smooth_ownership_priority(lhs, &smoothspecs[lhs], rhs, &smoothspecs[rhs])
    });

    let mut term_owners = vec![Vec::<usize>::new(); smoothspecs.len()];
    for (pos, &target_idx) in ownership_order.iter().enumerate() {
        let target = &smoothspecs[target_idx];
        term_owners[target_idx] = ownership_order[..pos]
            .iter()
            .copied()
            .filter(|&owner_idx| smooth_is_owned_by_prior_term(&smoothspecs[owner_idx], target))
            .collect();
    }

    SmoothStructureAnalysis {
        ownership_order,
        term_feature_cols,
        term_owners,
        basis_family_ranks,
    }
}


fn build_constraint_block(
    n: usize,
    parametric_block: Option<&Array2<f64>>,
    owner_blocks: &[&DesignMatrix],
) -> Result<Array2<f64>, BasisError> {
    let param_cols = parametric_block.map_or(0, |mat| mat.ncols());
    let owner_cols: usize = owner_blocks.iter().map(|design| design.ncols()).sum();
    let mut block = Array2::<f64>::zeros((n, param_cols + owner_cols));
    let mut col_start = 0usize;
    if let Some(parametric) = parametric_block {
        let col_end = col_start + parametric.ncols();
        block
            .slice_mut(s![.., col_start..col_end])
            .assign(parametric);
        col_start = col_end;
    }
    const CHUNK: usize = 1024;
    for owner in owner_blocks {
        let col_end = col_start + owner.ncols();
        for row_start in (0..n).step_by(CHUNK) {
            let row_end = (row_start + CHUNK).min(n);
            let chunk = (*owner)
                .try_row_chunk(row_start..row_end)
                .map_err(|e| BasisError::InvalidInput(e.to_string()))?;
            block
                .slice_mut(s![row_start..row_end, col_start..col_end])
                .assign(&chunk);
        }
        col_start = col_end;
    }
    Ok(block)
}


fn design_cross_relative_residual(
    lhs: &DesignMatrix,
    rhs: &DesignMatrix,
) -> Result<f64, BasisError> {
    let n = lhs.nrows();
    if rhs.nrows() != n {
        return Err(BasisError::ConstraintMatrixRowMismatch {
            basisrows: n,
            constraintrows: rhs.nrows(),
        });
    }
    const CHUNK: usize = 1024;
    let mut cross = Array2::<f64>::zeros((lhs.ncols(), rhs.ncols()));
    let mut lhs_sumsq = 0.0;
    let mut rhs_sumsq = 0.0;
    for start in (0..n).step_by(CHUNK) {
        let end = (start + CHUNK).min(n);
        let lhs_chunk = lhs
            .try_row_chunk(start..end)
            .map_err(|e| BasisError::InvalidInput(e.to_string()))?;
        let rhs_chunk = rhs
            .try_row_chunk(start..end)
            .map_err(|e| BasisError::InvalidInput(e.to_string()))?;
        cross += &crate::faer_ndarray::fast_atb(&lhs_chunk, &rhs_chunk);
        lhs_sumsq += lhs_chunk.iter().map(|v| v * v).sum::<f64>();
        rhs_sumsq += rhs_chunk.iter().map(|v| v * v).sum::<f64>();
    }
    let num = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
    let denom = (lhs_sumsq.sqrt() * rhs_sumsq.sqrt()).max(1e-300);
    Ok(num / denom)
}


fn smooth_has_overlapping_linear_terms(
    linear_terms: &[LinearTermSpec],
    termspec: &SmoothTermSpec,
) -> bool {
    let feature_cols = smooth_term_feature_cols(termspec);
    linear_terms
        .iter()
        .any(|linear| feature_cols.contains(&linear.feature_col))
}


fn smooth_intrinsic_parametric_feature_cols(
    linear_terms: &[LinearTermSpec],
    term: &SmoothTermSpec,
) -> Vec<usize> {
    // Returns the data columns that should appear in the smooth's parametric
    // constraint block `C = [1, …]`, alongside which the smooth is then
    // ORTHOGONALIZED via `apply_smooth_transform_to_design`.  Every column
    // returned here is therefore a direction that gets projected OUT of the
    // smooth's basis.  The constant intercept is always included by
    // `build_parametric_constraint_block_for_term`, so this function only
    // controls the polynomial axes added on top of that intercept.
    //
    // Ownership rule: explicit linear terms claim their matching axes — and
    // only those axes — for projection.  A standalone smooth (no overlapping
    // linear term) keeps its full polynomial nullspace and is centered only
    // against the implicit intercept; this matches the canonical thin-plate
    // / Duchon model surface where the linear component is part of the
    // smooth itself.
    let feature_cols = smooth_term_feature_cols(term);
    let mut owned = Vec::new();
    for linear in linear_terms {
        if feature_cols.contains(&linear.feature_col) && !owned.contains(&linear.feature_col) {
            owned.push(linear.feature_col);
        }
    }
    owned
}


pub(crate) fn apply_global_smooth_identifiability(
    smooth: RawSmoothDesign,
    data: ArrayView2<'_, f64>,
    linear_terms: &[LinearTermSpec],
    smoothspecs: &[SmoothTermSpec],
) -> Result<SmoothDesign, BasisError> {
    // Global smooth identifiability policy:
    //
    // 1. Any smooth that overlaps explicit linear terms is residualized against
    //    [intercept | overlapping linear columns]. Spatial smooths also keep
    //    their existing parametric orthogonality policy when requested.
    // 2. Higher-order / duplicate smooths are orthogonalized to the realized
    //    design columns of lower-order owned smooths over nested feature sets.
    //
    // This yields a deterministic hierarchical decomposition: lower-order smooths
    // own their subspaces, and broader smooths fit only the residual structure.
    if smoothspecs.len() != smooth.terms.len() {
        crate::bail_dim_basis!(
            "smooth spec count ({}) does not match built term count ({})",
            smoothspecs.len(),
            smooth.terms.len()
        );
    }

    if smooth.terms.is_empty() {
        return Ok(smooth.into());
    }

    let mut local_designs = vec![None; smooth.terms.len()];
    let mut local_penalties = vec![Vec::<Array2<f64>>::new(); smooth.terms.len()];
    let mut local_nullspaces = vec![Vec::<usize>::new(); smooth.terms.len()];
    let mut local_penaltyinfo = vec![Vec::<PenaltyInfo>::new(); smooth.terms.len()];
    let mut local_metadata = vec![None; smooth.terms.len()];
    let mut local_dims = vec![0usize; smooth.terms.len()];
    let mut local_linear_constraints = vec![None; smooth.terms.len()];
    let mut local_unabsorbed_z = vec![None::<Array2<f64>>; smooth.terms.len()];

    let SmoothStructureAnalysis {
        ownership_order,
        term_owners,
        ..
    } = analyze_smooth_ownership(smoothspecs);

    use rayon::iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    };

    for &idx in &ownership_order {
        let term = &smooth.terms[idx];
        let termspec = &smoothspecs[idx];
        let design_local = smooth.term_designs[idx].clone();
        // A frozen global-orthogonality chart (#978) is a pure replay: the
        // fit already decided this term's residualization against its owner
        // terms, and that decision is training-row data — rederiving it from
        // new rows would be wrong, and skipping it (the pre-#978 behavior)
        // emitted an unresidualized design wider than the fitted coefficient
        // block. So it bypasses both the owner analysis and the frozen-skip
        // gate below.
        let replay_z = frozen_global_orthogonality(termspec);
        let skip_global_transform = replay_z.is_none()
            && (smooth_has_frozen_identifiability(termspec) || term.lower_bounds_local.is_some());
        let owner_indices = if replay_z.is_some() || skip_global_transform {
            Vec::new()
        } else {
            // Relative cross-residual above which a dependent smooth's design is
            // judged to share column space with an owner term and so needs that
            // owner's block in its identifiability transform.
            const OVERLAP_REL_RESIDUAL_TOL: f64 = 1e-10;
            let owner_cross_checks = term_owners[idx]
                .clone()
                .into_par_iter()
                .map(|owner_idx| {
                    let owner_design = local_designs[owner_idx]
                        .as_ref()
                        .expect("owner design must be available before dependent smooth");
                    design_cross_relative_residual(&design_local, owner_design)
                        .map(|rel| (owner_idx, rel))
                })
                .collect::<Vec<_>>();
            let mut out = Vec::new();
            for check in owner_cross_checks {
                let (owner_idx, rel) = check?;
                if rel > OVERLAP_REL_RESIDUAL_TOL {
                    out.push(owner_idx);
                }
            }
            out
        };
        let owner_blocks = owner_indices
            .iter()
            .map(|owner_idx| {
                local_designs[*owner_idx]
                    .as_ref()
                    .expect("owner design must be available before dependent smooth")
            })
            .collect::<Vec<_>>();
        let needs_parametric_block = replay_z.is_none()
            && !skip_global_transform
            && (smooth_has_overlapping_linear_terms(linear_terms, termspec)
                || !smooth_intrinsic_parametric_feature_cols(linear_terms, termspec).is_empty()
                || smooth_requires_parametric_orthogonality(termspec)
                // A factor-by-level smooth must always be centered against its
                // gated level indicator (see `factor_by_level_gate`) so its
                // within-level constant cannot collide with the treatment-coded
                // factor main effect — even when no continuous linear term
                // overlaps it (e.g. `s(x, by=fac)` with no `+ x`).
                || factor_by_level_gate(termspec).is_some());
        let parametric_block = if !needs_parametric_block {
            None
        } else {
            Some(build_parametric_constraint_block_for_term(
                data,
                linear_terms,
                termspec,
            )?)
        };
        let c_local =
            if skip_global_transform || (parametric_block.is_none() && owner_blocks.is_empty()) {
                None
            } else {
                Some(build_constraint_block(
                    data.nrows(),
                    parametric_block.as_ref(),
                    &owner_blocks,
                )?)
            };
        let z_opt = if let Some(z) = replay_z {
            if design_local.ncols() != z.nrows() {
                crate::bail_dim_basis!(
                    "frozen global-orthogonality transform mismatch for term '{}': rebuilt design has {} columns but the persisted fit-time transform has {} rows",
                    term.name,
                    design_local.ncols(),
                    z.nrows()
                );
            }
            Some(z.clone())
        } else if skip_global_transform {
            None
        } else {
            match maybe_smooth_identifiability_transform(
                termspec,
                &design_local,
                c_local.as_ref().map(|mat| mat.view()),
            ) {
                Ok(z_opt) => z_opt,
                Err(BasisError::ConstraintNullspaceCollapsed { .. })
                    if !owner_blocks.is_empty() =>
                {
                    Some(Array2::zeros((design_local.ncols(), 0)))
                }
                Err(err) => return Err(err),
            }
        };
        let design_constrained = if let Some(z) = z_opt.as_ref() {
            apply_smooth_transform_to_design(design_local, z, &term.name)?
        } else {
            design_local
        };

        if let Some(c_ref) = c_local.as_ref() {
            let rel =
                orthogonality_relative_residual_for_design(&design_constrained, c_ref.view())?;
            // Largest relative residual tolerated before the constrained design
            // is rejected as not orthogonal to its sum-to-zero constraint rows.
            const ORTHOGONALITY_REL_RESIDUAL_TOL: f64 = 1e-8;
            let tol = ORTHOGONALITY_REL_RESIDUAL_TOL;
            if rel > tol {
                crate::bail_invalid_basis!(
                    "smooth orthogonality residual too large for term '{}': {:.3e} > {:.1e}",
                    term.name,
                    rel,
                    tol
                );
            }
        }

        let active_penaltyinfo = term
            .penaltyinfo_local
            .iter()
            .filter(|info| info.active)
            .cloned()
            .collect::<Vec<_>>();
        if active_penaltyinfo.len() != term.penalties_local.len() {
            crate::bail_invalid_basis!(
                "internal penalty metadata mismatch for term '{}': activeinfos={}, penalties={}",
                term.name,
                active_penaltyinfo.len(),
                term.penalties_local.len()
            );
        }
        let penalties_constrained: Vec<Array2<f64>> = term
            .penalties_local
            .par_iter()
            .map(|s_local| {
                if let Some(z) = z_opt.as_ref() {
                    let zt_s = fast_atb(z, s_local);
                    fast_ab(&zt_s, z)
                } else {
                    s_local.clone()
                }
            })
            .collect();
        let penalty_candidates = penalties_constrained
            .into_par_iter()
            .zip(active_penaltyinfo.into_par_iter())
            .map(|(matrix, info)| {
                let (matrix, c_new) = normalize_penalty_in_constrained_space(&matrix);
                PenaltyCandidate {
                    nullspace_dim_hint: info.nullspace_dim_hint,
                    matrix,
                    source: info.source,
                    normalization_scale: info.normalization_scale * c_new,
                    kronecker_factors: None,
                    op: None,
                }
            })
            .collect::<Vec<_>>();
        let (penalties_constrained, nullspace_constrained, penaltyinfo_constrained) =
            filter_active_penalty_candidates(penalty_candidates)?;
        let linear_constraints_constrained =
            if let Some(lin_local) = term.linear_constraints_local.as_ref() {
                if let Some(z) = z_opt.as_ref() {
                    Some(LinearInequalityConstraints {
                        a: lin_local.a.dot(z),
                        b: lin_local.b.clone(),
                    })
                } else {
                    Some(lin_local.clone())
                }
            } else {
                None
            };

        local_dims[idx] = design_constrained.ncols();
        local_designs[idx] = Some(design_constrained);
        local_penalties[idx] = penalties_constrained;
        local_nullspaces[idx] = nullspace_constrained;
        local_penaltyinfo[idx] = penaltyinfo_constrained;
        local_linear_constraints[idx] = linear_constraints_constrained;
        let realized_transform = match (term.joint_null_rotation.as_ref(), z_opt.as_ref()) {
            (Some(rotation), Some(z)) => {
                Some(crate::linalg::faer_ndarray::fast_ab(&rotation.rotation, z))
            }
            (Some(rotation), None) => Some(rotation.rotation.clone()),
            (None, Some(z)) => Some(z.clone()),
            (None, None) => None,
        };
        // Factor-smooth kinds cannot absorb the realized transform into their
        // metadata, so it is exported on the term instead and persisted onto
        // the spec by `freeze_term_collection_from_design` (#978):
        //
        // - Block-replicated factor smooths (`bs="sz"` → `FactorSumToZero`)
        //   carry PER-MARGINAL metadata (predict rebuilds the single inner
        //   marginal then re-stacks the `L-1` sum-to-zero deviation blocks).
        //   The realized transform lives in the FULL `p·(L-1)`-column design
        //   space, so it cannot be folded into the per-marginal metadata (the
        //   dimensions don't compose; folding it in both crashed basis
        //   generation and would double-count `Q` on rebuild, #700). The raw
        //   design builder reapplies `Q` deterministically at predict time, so
        //   only the global-orthogonality `Z` (post-`Q` chart) is exported.
        //
        // - `FactorSmooth` (`fs`/`re`) metadata has no transform slot at all
        //   (its `with_identifiability_transform` arm rejects one). Like `sz`,
        //   any stage-2 joint-null `Q` is recomputed by the raw builder on
        //   rebuild (and is typically absent: `fs` penalties are full-rank),
        //   so the exported chart is likewise the post-`Q` `Z` alone.
        //
        // Without this export the overlap residualization of
        // `s(x) + s(g, x, bs=sz)` / `s(x) + fs(x, g)` was silently dropped:
        // the fit used the narrowed `X·Z` design while every predict rebuilt
        // the full-width design, making the model unpredictable (#978).
        match &termspec.basis {
            SmoothBasisSpec::FactorSumToZero { .. } | SmoothBasisSpec::FactorSmooth { .. } => {
                local_metadata[idx] = Some(term.metadata.clone());
                local_unabsorbed_z[idx] = z_opt.clone();
            }
            _ => {
                local_metadata[idx] = Some(with_identifiability_transform(
                    &term.metadata,
                    realized_transform.as_ref(),
                )?);
            }
        }
    }

    let total_p: usize = local_dims.iter().sum();
    let mut terms_out = Vec::<SmoothTerm>::with_capacity(smooth.terms.len());
    let mut penalties_global = Vec::<BlockwisePenalty>::new();
    let mut nullspace_dims_global = Vec::<usize>::new();
    let mut penaltyinfo_global = Vec::<PenaltyBlockInfo>::new();
    let mut dropped_penaltyinfo_global = smooth.dropped_penaltyinfo.clone();
    let mut coefficient_lower_bounds = Array1::<f64>::from_elem(total_p, f64::NEG_INFINITY);
    let mut any_bounds = false;
    let mut linear_constraintsrows: Vec<Array1<f64>> = Vec::new();
    let mut linear_constraints_b: Vec<f64> = Vec::new();

    let mut col_start = 0usize;
    for idx in 0..smooth.terms.len() {
        let p_local = local_dims[idx];
        let col_end = col_start + p_local;

        let activeinfos = local_penaltyinfo[idx]
            .iter()
            .filter(|info| info.active)
            .collect::<Vec<_>>();
        if activeinfos.len() != local_penalties[idx].len() {
            crate::bail_invalid_basis!(
                "internal penalty info mismatch for term '{}': activeinfos={}, penalties={}",
                smooth.terms[idx].name,
                activeinfos.len(),
                local_penalties[idx].len()
            );
        }
        for ((s_local, &ns), info) in local_penalties[idx]
            .iter()
            .zip(local_nullspaces[idx].iter())
            .zip(activeinfos.into_iter())
        {
            let global_index = penalties_global.len();
            penalties_global.push(BlockwisePenalty::new(col_start..col_end, s_local.clone()));
            nullspace_dims_global.push(ns);
            let mut penalty = info.clone();
            penalty.nullspace_dim_hint = ns;
            penaltyinfo_global.push(PenaltyBlockInfo {
                global_index,
                termname: Some(smooth.terms[idx].name.clone()),
                penalty,
            });
        }
        for info in local_penaltyinfo[idx].iter().filter(|info| !info.active) {
            dropped_penaltyinfo_global.push(DroppedPenaltyBlockInfo {
                termname: Some(smooth.terms[idx].name.clone()),
                penalty: info.clone(),
            });
        }

        terms_out.push(SmoothTerm {
            name: smooth.terms[idx].name.clone(),
            coeff_range: col_start..col_end,
            shape: smooth.terms[idx].shape,
            penalties_local: local_penalties[idx].clone(),
            nullspace_dims: local_nullspaces[idx].clone(),
            penaltyinfo_local: local_penaltyinfo[idx].clone(),
            metadata: local_metadata[idx]
                .clone()
                .expect("local metadata must exist for every smooth term"),
            lower_bounds_local: smooth.terms[idx].lower_bounds_local.clone(),
            linear_constraints_local: local_linear_constraints[idx].clone(),
            // Global orthogonality transforms break Kronecker structure.
            kronecker_factored: None,
            // The final raw-basis → coefficient chart, including any
            // stage-2 joint-null Q and global orthogonality Z, is embedded in
            // `metadata` above. Keeping Q separately here would apply it twice
            // on frozen rebuilds and would put derivative operators in a
            // different chart from the value path.
            joint_null_rotation: None,
            // Factor-smooth kinds export the chart their metadata could not
            // absorb; the freeze persists it onto the spec for replay (#978).
            unabsorbed_global_orthogonality: local_unabsorbed_z[idx].clone(),
        });
        if let Some(lin_local) = &local_linear_constraints[idx] {
            for r in 0..lin_local.a.nrows() {
                let mut row = Array1::<f64>::zeros(total_p);
                row.slice_mut(s![col_start..col_end])
                    .assign(&lin_local.a.row(r));
                linear_constraintsrows.push(row);
                linear_constraints_b.push(lin_local.b[r]);
            }
        }
        if let Some(lb_local) = smooth.terms[idx].lower_bounds_local.as_ref()
            && lb_local.len() == p_local
        {
            coefficient_lower_bounds
                .slice_mut(s![col_start..col_end])
                .assign(lb_local);
            any_bounds = true;
        }

        col_start = col_end;
    }

    assert_eq!(
        penalties_global.len(),
        nullspace_dims_global.len(),
        "globally reparameterized smooth penalty/nullspace bookkeeping diverged"
    );
    assert_eq!(
        penalties_global.len(),
        penaltyinfo_global.len(),
        "globally reparameterized smooth penalty metadata bookkeeping diverged"
    );

    Ok(SmoothDesign {
        term_designs: local_designs
            .into_iter()
            .map(|design| design.expect("local design must exist for every smooth term"))
            .collect(),
        penalties: penalties_global,
        nullspace_dims: nullspace_dims_global,
        penaltyinfo: penaltyinfo_global,
        dropped_penaltyinfo: dropped_penaltyinfo_global,
        terms: terms_out,
        coefficient_lower_bounds: if any_bounds {
            Some(coefficient_lower_bounds)
        } else {
            None
        },
        linear_constraints: if linear_constraintsrows.is_empty() {
            None
        } else {
            let mut a = Array2::<f64>::zeros((linear_constraintsrows.len(), total_p));
            for (i, row) in linear_constraintsrows.iter().enumerate() {
                a.row_mut(i).assign(row);
            }
            Some(LinearInequalityConstraints {
                a,
                b: Array1::from_vec(linear_constraints_b),
            })
        },
    })
}


/// If `termspec` is a single-level factor-by smooth (`s(x, by=fac)` expanded
/// into one `ByVariable { kind: Level }` block per factor level), return the
/// `(by_col, value_bits)` pair identifying which rows that level's block gates
/// to. `None` for numeric-by smooths and every other basis.
///
/// A factor-by smooth's per-level block is the inner basis multiplied by the
/// level indicator (zero on every other level's rows). Its column span
/// therefore contains the per-level CONSTANT — a vector that is `1` on this
/// level's rows and `0` elsewhere — which is exactly the column the
/// treatment-coded factor main effect (`build_termspec` auto-adds one as an
/// unpenalized random-effect term) already carries. Centering each level's
/// smooth against the *global* intercept (`build_parametric_constraint_block_for_term`'s
/// default) removes only its global mean, leaving that within-level constant
/// to collide with the factor main effect: a rank-1 collinearity that lets the
/// penalty/ridge split the per-group baseline level between the two blocks and
/// under-recover it (the per-group log-cumulative-hazard offset leaks out — the
/// #900 weibull-AFT-by-factor surface miscalibration). Centering against the
/// gated level indicator instead removes the within-level constant cleanly,
/// leaving the per-group level entirely to the factor main effect (mgcv's
/// by-factor convention), while the per-level slope/curvature deviation stays
/// in the smooth (we deliberately do NOT project the overlapping continuous
/// axis out of a by-level smooth — that deviation is the by-factor signal).
fn factor_by_level_gate(termspec: &SmoothTermSpec) -> Option<(usize, u64)> {
    match &termspec.basis {
        SmoothBasisSpec::ByVariable {
            by_col,
            by: ByVariableSpec::Level { value_bits, .. },
            ..
        } => Some((*by_col, *value_bits)),
        _ => None,
    }
}


fn build_parametric_constraint_block_for_term(
    data: ArrayView2<'_, f64>,
    linear_terms: &[LinearTermSpec],
    termspec: &SmoothTermSpec,
) -> Result<Array2<f64>, BasisError> {
    let n = data.nrows();
    let p_data = data.ncols();

    // Factor-by-level smooth: center against the gated level indicator so the
    // within-level constant is removed (it belongs to the treatment-coded
    // factor main effect), not against the global `[1 | overlapping axes]`.
    if let Some((by_col, value_bits)) = factor_by_level_gate(termspec) {
        if by_col >= p_data {
            crate::bail_dim_basis!(
                "factor-by smooth term '{}' by column {by_col} out of bounds for {p_data} columns",
                termspec.name
            );
        }
        let mut c = Array2::<f64>::zeros((n, 1));
        let by = data.column(by_col);
        for (row, &value) in by.iter().enumerate() {
            if value.to_bits() == value_bits {
                c[[row, 0]] = 1.0;
            }
        }
        return Ok(c);
    }

    let feature_cols = smooth_term_feature_cols(termspec);
    let mut parametric_cols = smooth_intrinsic_parametric_feature_cols(linear_terms, termspec);
    for &feature_col in &parametric_cols {
        if feature_col >= p_data {
            crate::bail_dim_basis!(
                "smooth term feature column {feature_col} out of bounds for {p_data} columns"
            );
        }
    }
    for linear in linear_terms
        .iter()
        .filter(|linear| feature_cols.contains(&linear.feature_col))
    {
        if linear.feature_col >= p_data {
            crate::bail_dim_basis!(
                "linear term '{}' feature column {} out of bounds for {} columns",
                linear.name,
                linear.feature_col,
                p_data
            );
        }
        if !parametric_cols.contains(&linear.feature_col) {
            parametric_cols.push(linear.feature_col);
        }
    }

    let mut c = Array2::<f64>::zeros((n, 1 + parametric_cols.len()));
    c.column_mut(0).fill(1.0);
    for (j, &feature_col) in parametric_cols.iter().enumerate() {
        c.column_mut(j + 1).assign(&data.column(feature_col));
    }
    Ok(c)
}


fn apply_smooth_transform_to_design(
    design_local: DesignMatrix,
    transform: &Array2<f64>,
    termname: &str,
) -> Result<DesignMatrix, BasisError> {
    match design_local {
        DesignMatrix::Dense(inner) => {
            let op = CoefficientTransformOperator::new(inner, transform.clone()).map_err(|e| {
                BasisError::InvalidInput(format!(
                    "smooth identifiability transform failed for term '{termname}': {e}"
                ))
            })?;
            Ok(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Arc::new(op),
            )))
        }
        DesignMatrix::Sparse(inner) => {
            let dense = inner
                .try_to_dense_arc("smooth identifiability sparse transform")
                .map_err(BasisError::InvalidInput)?
                .as_ref()
                .dot(transform);
            Ok(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                dense,
            )))
        }
    }
}


fn design_constraint_cross(
    design: &DesignMatrix,
    constraint_matrix: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, BasisError> {
    let n = design.nrows();
    if constraint_matrix.nrows() != n {
        return Err(BasisError::ConstraintMatrixRowMismatch {
            basisrows: n,
            constraintrows: constraint_matrix.nrows(),
        });
    }
    let mut cross = Array2::<f64>::zeros((design.ncols(), constraint_matrix.ncols()));
    const CHUNK: usize = 1024;
    for start in (0..n).step_by(CHUNK) {
        let end = (start + CHUNK).min(n);
        let design_chunk = design
            .try_row_chunk(start..end)
            .map_err(|e| BasisError::InvalidInput(e.to_string()))?;
        let constraint_chunk = constraint_matrix.slice(s![start..end, ..]).to_owned();
        cross += &crate::faer_ndarray::fast_atb(&design_chunk, &constraint_chunk);
    }
    Ok(cross)
}


fn design_frobenius_norm(design: &DesignMatrix) -> Result<f64, BasisError> {
    let n = design.nrows();
    const CHUNK: usize = 1024;
    let mut sumsq = 0.0;
    for start in (0..n).step_by(CHUNK) {
        let end = (start + CHUNK).min(n);
        let chunk = design
            .try_row_chunk(start..end)
            .map_err(|e| BasisError::InvalidInput(e.to_string()))?;
        sumsq += chunk.iter().map(|v| v * v).sum::<f64>();
    }
    Ok(sumsq.sqrt())
}


/// The persisted fit-time global-orthogonality chart for a factor-smooth
/// term, if one was frozen onto its spec (#978). `Some` means this term was
/// residualized against owner terms at fit time and prediction/refit rebuilds
/// must replay exactly that column map instead of rederiving anything from
/// the (new) rows.
fn frozen_global_orthogonality(termspec: &SmoothTermSpec) -> Option<&Array2<f64>> {
    match &termspec.basis {
        SmoothBasisSpec::FactorSumToZero {
            frozen_global_orthogonality,
            ..
        } => frozen_global_orthogonality.as_ref(),
        SmoothBasisSpec::FactorSmooth { spec } => spec.frozen_global_orthogonality.as_ref(),
        _ => None,
    }
}


fn maybe_smooth_identifiability_transform(
    termspec: &SmoothTermSpec,
    design_local: &DesignMatrix,
    constraint_block: Option<ArrayView2<'_, f64>>,
) -> Result<Option<Array2<f64>>, BasisError> {
    if let Some(SpatialIdentifiability::FrozenTransform { transform }) =
        spatial_identifiability_policy(termspec)
    {
        if design_local.ncols() != transform.nrows() {
            crate::bail_dim_basis!(
                "frozen spatial identifiability transform mismatch: design has {} columns but transform has {} rows",
                design_local.ncols(),
                transform.nrows()
            );
        }
        return Ok(Some(transform.clone()));
    }

    if let Some(c) = constraint_block {
        if c.ncols() == 0 {
            Ok(None)
        } else {
            Ok(Some(orthogonality_transform_for_design(
                design_local,
                c,
                None, // fixed subspace: do not use iteration-varying PIRLS weights
            )?))
        }
    } else {
        Ok(None)
    }
}


pub(crate) fn spatial_identifiability_policy(termspec: &SmoothTermSpec) -> Option<&SpatialIdentifiability> {
    match &termspec.basis {
        SmoothBasisSpec::ThinPlate { spec, .. } => Some(&spec.identifiability),
        SmoothBasisSpec::Duchon { spec, .. } => Some(&spec.identifiability),
        _ => None,
    }
}


/// Whether this smooth's *realized* design (the basis evaluated at the n data
/// rows) must be residualized against the model's parametric block (intercept +
/// any overlapping linear columns) by `apply_global_smooth_identifiability`.
///
/// This is the universal identifiability invariant for **kernel / radial**
/// spatial smooths (#531): their realized column span contains the constant
/// (and, at `Linear` null-space order, the linear monomials), so without this
/// step the smooth and the parametric intercept fight over the same direction —
/// a structural rank-1 collision. The collision is invisible to the kernels'
/// *own* identifiability constraints because those act in **coefficient space at
/// the K centers**, not on the realized design rows:
///   - Matérn `CenterSumToZero` enforces `1ᵀα = 0` over the centers, so
///     `Kα` evaluated at the data rows still spans the constant.
///   - Duchon / TPS `OrthogonalToParametric` *defers* its centering to this very
///     step, which is why it is listed here too.
///
/// Tensor-product and B-spline bases instead apply a realized-design sum-to-zero
/// at basis-build time (`apply_sum_to_zero_constraint`), so they already satisfy
/// the invariant and must NOT be double-constrained — they return `false`.
///
/// The remaining bases are excluded, each for a concrete reason:
///   - **Sphere, Harmonic method**: the real-spherical-harmonic basis starts at
///     degree `l = 1` (`build_spherical_harmonic_basis`), so it never spans the
///     degree-0 constant — no centering is needed.
///   - **Sphere, Wahba method**: INCLUDED (#532). Its
///     `weighted_coefficient_sum_to_zero_transform` is a *center*-space
///     constraint, so the realized design still spans the constant — same class
///     as Matérn `CenterSumToZero`. The composed parametric transform is frozen
///     onto `SphericalSplineBasisSpec::identifiability`
///     (`SphericalSplineIdentifiability::FrozenTransform`) and replayed by
///     `build_spherical_spline_basis` at predict time, so the orthogonalization
///     survives save → reload exactly as it does for Matérn.
///   - **PCA**: its `with_identifiability_transform` arm rejects a post-hoc
///     transform (the constraint lives inside the orthonormal loadings), and its
///     constant content is governed by the `centered` flag, not a residualizable
///     design.
///
/// `FrozenTransform` bases are excluded: a transform frozen by *this* pipeline
/// already has the parametric orthogonalization composed in (see
/// `with_identifiability_transform`), and they are gated out upstream by
/// `skip_global_transform` regardless.
fn smooth_requires_parametric_orthogonality(termspec: &SmoothTermSpec) -> bool {
    match &termspec.basis {
        SmoothBasisSpec::ByVariable { inner, .. }
        | SmoothBasisSpec::FactorSumToZero { inner, .. } => {
            smooth_requires_parametric_orthogonality(&SmoothTermSpec {
                name: termspec.name.clone(),
                basis: (**inner).clone(),
                shape: termspec.shape,
                joint_null_rotation: None,
            })
        }
        SmoothBasisSpec::BySmooth { smooth, .. } => {
            smooth_requires_parametric_orthogonality(&SmoothTermSpec {
                name: termspec.name.clone(),
                basis: (**smooth).clone(),
                shape: termspec.shape,
                joint_null_rotation: None,
            })
        }
        SmoothBasisSpec::ThinPlate { spec, .. } => {
            matches!(
                spec.identifiability,
                SpatialIdentifiability::OrthogonalToParametric
            )
        }
        SmoothBasisSpec::Duchon { spec, .. } => {
            matches!(
                spec.identifiability,
                SpatialIdentifiability::OrthogonalToParametric
            )
        }
        SmoothBasisSpec::Matern { spec, .. } => matches!(
            spec.identifiability,
            MaternIdentifiability::CenterSumToZero | MaternIdentifiability::CenterLinearOrthogonal
        ),
        // Wahba sphere (`bs="sos"`, method=Wahba): the area-weighted center
        // sum-to-zero `z` is a *coefficient*-space constraint, so the realized
        // `K·z` design still spans the constant — the same #531 collision class
        // as Matérn `CenterSumToZero`. It requires the global parametric
        // orthogonalization (#532). The Harmonic method starts at degree l=1
        // and never spans the constant, so it is excluded.
        SmoothBasisSpec::Sphere { spec, .. } => {
            matches!(spec.method, crate::basis::SphereMethod::Wahba)
                && matches!(
                    spec.identifiability,
                    SphericalSplineIdentifiability::CenterSumToZero
                )
        }
        // Constant-curvature geodesic kernel: same #531 collision class as the
        // Wahba sphere — the coefficient-space sum-to-zero `z` leaves the
        // realized `K·z` design spanning the constant on the data rows, so the
        // global parametric orthogonalization must compose onto `z` (#532).
        SmoothBasisSpec::ConstantCurvature { spec, .. } => matches!(
            spec.identifiability,
            ConstantCurvatureIdentifiability::CenterSumToZero
        ),
        // Measure-jet representer: identical #531 collision class — Gaussian
        // RBF columns times the center-space sum-to-zero `z` still span the
        // constant on the data rows, so `z` must absorb the parametric
        // orthogonalization (#532).
        SmoothBasisSpec::MeasureJet { spec, .. } => matches!(
            spec.identifiability,
            MeasureJetIdentifiability::CenterSumToZero
        ),
        SmoothBasisSpec::BSpline1D { .. }
        | SmoothBasisSpec::TensorBSpline { .. }
        | SmoothBasisSpec::Pca { .. }
        | SmoothBasisSpec::FactorSmooth { .. } => false,
    }
}


fn compose_identifiability_transforms(
    existing: Option<&Array2<f64>>,
    extra: Option<&Array2<f64>>,
) -> Result<Option<Array2<f64>>, BasisError> {
    match (existing, extra) {
        (Some(lhs), Some(rhs)) => {
            if lhs.ncols() == rhs.nrows() {
                Ok(Some(lhs.dot(rhs)))
            } else if lhs.nrows() == rhs.nrows() && lhs.ncols() == rhs.ncols() {
                // Rebuilding from an already-frozen spec can surface the same
                // raw->frozen transform twice. Treat that as idempotent
                // metadata, not a sequential Z_left * Z_right composition.
                Ok(Some(rhs.clone()))
            } else {
                Err(BasisError::DimensionMismatch(format!(
                    "identifiability transform mismatch: existing is {}x{}, extra is {}x{}",
                    lhs.nrows(),
                    lhs.ncols(),
                    rhs.nrows(),
                    rhs.ncols(),
                )))
            }
        }
        (Some(lhs), None) => Ok(Some(lhs.clone())),
        (None, Some(rhs)) => Ok(Some(rhs.clone())),
        (None, None) => Ok(None),
    }
}


fn with_identifiability_transform(
    metadata: &BasisMetadata,
    transform: Option<&Array2<f64>>,
) -> Result<BasisMetadata, BasisError> {
    match metadata {
        BasisMetadata::BSpline1D {
            knots,
            identifiability_transform,
            periodic,
            degree,
            auto_shrink_note,
        } => Ok(BasisMetadata::BSpline1D {
            knots: knots.clone(),
            periodic: *periodic,
            identifiability_transform: compose_identifiability_transforms(
                identifiability_transform.as_ref(),
                transform,
            )?,
            degree: *degree,
            auto_shrink_note: auto_shrink_note.clone(),
        }),
        BasisMetadata::ThinPlate {
            centers,
            length_scale,
            periodic,
            identifiability_transform,
            input_scales,
            radial_reparam,
        } => Ok(BasisMetadata::ThinPlate {
            centers: centers.clone(),
            length_scale: *length_scale,
            periodic: periodic.clone(),
            identifiability_transform: compose_identifiability_transforms(
                identifiability_transform.as_ref(),
                transform,
            )?,
            input_scales: input_scales.clone(),
            radial_reparam: radial_reparam.clone(),
        }),
        BasisMetadata::Sphere {
            centers,
            penalty_order,
            method,
            max_degree,
            wahba_kernel,
            constraint_transform,
        } => Ok(BasisMetadata::Sphere {
            centers: centers.clone(),
            penalty_order: *penalty_order,
            method: *method,
            max_degree: *max_degree,
            wahba_kernel: *wahba_kernel,
            constraint_transform: compose_identifiability_transforms(
                constraint_transform.as_ref(),
                transform,
            )?,
        }),
        BasisMetadata::ConstantCurvature {
            centers,
            kappa,
            length_scale,
            constraint_transform,
        } => Ok(BasisMetadata::ConstantCurvature {
            centers: centers.clone(),
            kappa: *kappa,
            length_scale: *length_scale,
            constraint_transform: compose_identifiability_transforms(
                constraint_transform.as_ref(),
                transform,
            )?,
        }),
        BasisMetadata::MeasureJet {
            centers,
            input_scales,
            length_scale,
            eps_band,
            order_s,
            alpha,
            tau0,
            masses,
            support_means,
            penalty_normalization_scales,
            raw_penalty_normalization_scales,
            fused_penalty_normalization_scale,
            constraint_transform,
        } => Ok(BasisMetadata::MeasureJet {
            centers: centers.clone(),
            input_scales: input_scales.clone(),
            length_scale: *length_scale,
            eps_band: eps_band.clone(),
            order_s: *order_s,
            alpha: *alpha,
            tau0: *tau0,
            masses: masses.clone(),
            support_means: support_means.clone(),
            penalty_normalization_scales: penalty_normalization_scales.clone(),
            raw_penalty_normalization_scales: raw_penalty_normalization_scales.clone(),
            fused_penalty_normalization_scale: *fused_penalty_normalization_scale,
            constraint_transform: compose_identifiability_transforms(
                constraint_transform.as_ref(),
                transform,
            )?,
        }),
        BasisMetadata::Matern {
            centers,
            length_scale,
            periodic,
            nu,
            include_intercept,
            identifiability_transform,
            input_scales,
            aniso_log_scales,
            nullspace_shrinkage_survived,
        } => Ok(BasisMetadata::Matern {
            centers: centers.clone(),
            length_scale: *length_scale,
            periodic: periodic.clone(),
            nu: *nu,
            include_intercept: *include_intercept,
            identifiability_transform: compose_identifiability_transforms(
                identifiability_transform.as_ref(),
                transform,
            )?,
            input_scales: input_scales.clone(),
            aniso_log_scales: aniso_log_scales.clone(),
            nullspace_shrinkage_survived: *nullspace_shrinkage_survived,
        }),
        BasisMetadata::Duchon {
            centers,
            length_scale,
            periodic,
            power,
            nullspace_order,
            identifiability_transform,
            input_scales,
            aniso_log_scales,
            operator_collocation_points,
        } => Ok(BasisMetadata::Duchon {
            centers: centers.clone(),
            length_scale: *length_scale,
            periodic: periodic.clone(),
            power: *power,
            nullspace_order: *nullspace_order,
            input_scales: input_scales.clone(),
            aniso_log_scales: aniso_log_scales.clone(),
            operator_collocation_points: operator_collocation_points.clone(),
            identifiability_transform: compose_identifiability_transforms(
                identifiability_transform.as_ref(),
                transform,
            )?,
        }),
        BasisMetadata::SphereHarmonics {
            max_degree,
            radians,
        } => Ok(BasisMetadata::SphereHarmonics {
            max_degree: *max_degree,
            radians: *radians,
        }),
        BasisMetadata::TensorBSpline {
            feature_cols,
            knots,
            degrees,
            periods,
            identifiability_transform,
        } => Ok(BasisMetadata::TensorBSpline {
            feature_cols: feature_cols.clone(),
            knots: knots.clone(),
            degrees: degrees.clone(),
            periods: periods.clone(),
            identifiability_transform: compose_identifiability_transforms(
                identifiability_transform.as_ref(),
                transform,
            )?,
        }),
        BasisMetadata::BySmooth {
            inner,
            by_col,
            levels,
            ordered,
        } => Ok(BasisMetadata::BySmooth {
            inner: Box::new(with_identifiability_transform(inner, transform)?),
            by_col: *by_col,
            levels: levels.clone(),
            ordered: *ordered,
        }),
        BasisMetadata::FactorSmooth {
            continuous_cols,
            group_col,
            knots,
            degree,
            periodic,
            group_levels,
            flavour,
        } => {
            // Factor-smooth metadata has no transform slot; the global pass
            // exports its transform via `SmoothTerm::unabsorbed_global_orthogonality`
            // instead (#978). Silently dropping a transform here is what made
            // `s(x) + fs(x, g)` unpredictable — reject loudly so any future
            // caller that reaches this arm with a transform fails at fit time
            // rather than corrupting the saved coefficient chart.
            if transform.is_some() {
                crate::bail_invalid_basis!(
                    "FactorSmooth metadata cannot absorb an identifiability transform; \
                     route it through the term-level frozen_global_orthogonality carrier"
                );
            }
            Ok(BasisMetadata::FactorSmooth {
                continuous_cols: continuous_cols.clone(),
                group_col: *group_col,
                knots: knots.clone(),
                degree: *degree,
                periodic: *periodic,
                group_levels: group_levels.clone(),
                flavour: flavour.clone(),
            })
        }
        BasisMetadata::Pca {
            feature_cols,
            basis_matrix,
            centered,
            smooth_penalty,
            center_mean,
            pca_basis_path,
            chunk_size,
        } => {
            // PCA bases carry an orthonormal projection matrix and do not
            // expose an identifiability transform that can be re-composed
            // (the constraint, if any, lives inside the PCA loadings
            // themselves), so the caller cannot meaningfully attach a
            // post-hoc Z transform here.
            if transform.is_some() {
                crate::bail_invalid_basis!(
                    "PCA bases do not expose a composable identifiability transform"
                );
            }
            Ok(BasisMetadata::Pca {
                feature_cols: feature_cols.clone(),
                basis_matrix: basis_matrix.clone(),
                centered: *centered,
                smooth_penalty: *smooth_penalty,
                center_mean: center_mean.clone(),
                pca_basis_path: pca_basis_path.clone(),
                chunk_size: *chunk_size,
            })
        }
    }
}

fn orthogonality_relative_residual_for_design(
    design: &DesignMatrix,
    constraint_matrix: ArrayView2<'_, f64>,
) -> Result<f64, BasisError> {
    let cross = design_constraint_cross(design, constraint_matrix)?;
    let num = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
    let b_norm = design_frobenius_norm(design)?;
    let c_norm = constraint_matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
    let denom = (b_norm * c_norm).max(1e-300);
    Ok(num / denom)
}


pub(crate) fn stratified_spatial_subsample(
    data: ArrayView2<'_, f64>,
    spec: &TermCollectionSpec,
    target_size: usize,
) -> Vec<usize> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;

    let n = data.nrows();
    if n <= target_size {
        return (0..n).collect();
    }

    let spatial_cols: Option<Vec<usize>> =
        spec.smooth_terms.iter().find_map(|term| match &term.basis {
            SmoothBasisSpec::ThinPlate { feature_cols, .. }
            | SmoothBasisSpec::Matern { feature_cols, .. }
            | SmoothBasisSpec::Duchon { feature_cols, .. } => {
                if !feature_cols.is_empty() {
                    Some(feature_cols.clone())
                } else {
                    None
                }
            }
            _ => None,
        });

    let cols = match spatial_cols {
        Some(c) if !c.is_empty() => c,
        _ => {
            let mut rng = StdRng::seed_from_u64(spatial_subsample_seed(data, &[], target_size));
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);
            indices.truncate(target_size);
            indices.sort_unstable();
            return indices;
        }
    };
    let mut rng = StdRng::seed_from_u64(spatial_subsample_seed(data, &cols, target_size));

    let d = cols.len();
    let mut mins = vec![f64::INFINITY; d];
    let mut maxs = vec![f64::NEG_INFINITY; d];
    for i in 0..n {
        for (ax, &col) in cols.iter().enumerate() {
            let v = data[[i, col]];
            if v < mins[ax] {
                mins[ax] = v;
            }
            if v > maxs[ax] {
                maxs[ax] = v;
            }
        }
    }

    // Aim for roughly this many sampled points per stratification cell so each
    // occupied cell can contribute a representative draw without collapsing the
    // grid to one point per cell.
    const TARGET_POINTS_PER_CELL: usize = 5;
    let total_cells_target = (target_size / TARGET_POINTS_PER_CELL).max(1);
    let cells_per_axis = ((total_cells_target as f64).powf(1.0 / d as f64)).ceil() as usize;
    let cells_per_axis = cells_per_axis.max(1);

    let mut cell_members: std::collections::HashMap<Vec<usize>, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        let mut cell_key = Vec::with_capacity(d);
        for (ax, &col) in cols.iter().enumerate() {
            let range = maxs[ax] - mins[ax];
            let cell = if range <= 0.0 {
                0
            } else {
                let frac = (data[[i, col]] - mins[ax]) / range;
                (frac * cells_per_axis as f64).floor() as usize
            };
            cell_key.push(cell.min(cells_per_axis - 1));
        }
        cell_members.entry(cell_key).or_default().push(i);
    }

    let mut selected: Vec<usize> = Vec::with_capacity(target_size);
    let mut remaining_budget = target_size;
    let mut remaining_population = n;

    let mut cells: Vec<(Vec<usize>, Vec<usize>)> = cell_members.into_iter().collect();
    cells.sort_by(|a, b| a.0.cmp(&b.0));

    for (_, members) in &mut cells {
        if remaining_budget == 0 {
            break;
        }
        let alloc = ((members.len() as f64 / remaining_population as f64) * remaining_budget as f64)
            .round() as usize;
        let alloc = alloc.max(1).min(members.len()).min(remaining_budget);
        members.shuffle(&mut rng);
        selected.extend_from_slice(&members[..alloc]);
        remaining_budget = remaining_budget.saturating_sub(alloc);
        remaining_population = remaining_population.saturating_sub(members.len());
    }

    if selected.len() > target_size {
        selected.shuffle(&mut rng);
        selected.truncate(target_size);
    }

    selected.sort_unstable();
    selected
}


pub(crate) fn spatial_subsample_seed(
    data: ArrayView2<'_, f64>,
    spatial_cols: &[usize],
    target_size: usize,
) -> u64 {
    let mut state = 0x5350_4154_4941_4C53_u64;
    spatial_seed_mix(&mut state, data.nrows() as u64);
    spatial_seed_mix(&mut state, data.ncols() as u64);
    spatial_seed_mix(&mut state, target_size as u64);
    spatial_seed_mix(&mut state, spatial_cols.len() as u64);
    for &col in spatial_cols {
        spatial_seed_mix(&mut state, col as u64);
    }

    if data.nrows() > 0 {
        let mid = data.nrows() / 2;
        let last = data.nrows() - 1;
        for &row in &[0usize, mid, last] {
            for &col in spatial_cols {
                let value = data[[row, col]];
                spatial_seed_mix(&mut state, value.to_bits());
            }
        }
    }
    state
}


#[inline]
pub(crate) fn spatial_seed_mix(state: &mut u64, value: u64) {
    let mut z = value
        .wrapping_add(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(*state);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    *state ^= z ^ (z >> 31);
    *state = (*state).rotate_left(27).wrapping_mul(0x3C79_AC49_2BA7_B653);
}


pub(crate) fn sampled_rows(data: ArrayView2<'_, f64>, indices: &[usize]) -> Array2<f64> {
    let mut sampled = Array2::<f64>::zeros((indices.len(), data.ncols()));
    for (new_row, &orig_row) in indices.iter().enumerate() {
        sampled.row_mut(new_row).assign(&data.row(orig_row));
    }
    sampled
}



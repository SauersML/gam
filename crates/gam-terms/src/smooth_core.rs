use std::ops::Range;

use ndarray::{Array2, s};

#[derive(Debug, Clone)]
pub enum PenaltyStructureHint {
    Ridge(f64),
    Kronecker(Vec<Array2<f64>>),
}

/// A penalty matrix stored at its natural block size together with the
/// column range it occupies in the global coefficient vector.
///
/// Instead of embedding every penalty into a full `p_total × p_total` dense
/// matrix filled with zeros, we keep the compact local matrix and reconstruct
/// the global view only when a downstream consumer explicitly requires it.
#[derive(Clone)]
pub struct BlockwisePenalty {
    /// Column range in the global coefficient vector that this penalty covers.
    pub col_range: Range<usize>,
    /// The local penalty matrix — dimensions `block_p × block_p` where
    /// `block_p = col_range.len()`.
    pub local: Array2<f64>,
    /// Optional nonzero centering vector for this coefficient block.
    pub prior_mean: gam_problem::CoefficientPriorMean,
    /// Optional structural hint so downstream spectral/logdet code can stay
    /// block-local or factorized without reverse-engineering the matrix.
    pub structure_hint: Option<PenaltyStructureHint>,
    /// Optional operator-form handle bit-equivalent to `local`. Populated when
    /// the originating closed-form factory emitted an op-form penalty so exact
    /// operator algebra can use matvec instead of materializing the dense
    /// `block_p × block_p` Gram. `None` for ordinary dense penalties.
    pub op: Option<std::sync::Arc<dyn crate::analytic_penalties::PenaltyOp>>,
}

impl std::fmt::Debug for BlockwisePenalty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockwisePenalty")
            .field("col_range", &self.col_range)
            .field(
                "local",
                &format_args!("{}×{}", self.local.nrows(), self.local.ncols()),
            )
            .field("prior_mean", &self.prior_mean)
            .field("structure_hint", &self.structure_hint)
            .field("op", &self.op.as_ref().map(|o| o.dim()))
            .finish()
    }
}

impl BlockwisePenalty {
    /// Create a new blockwise penalty.
    pub fn new(col_range: Range<usize>, local: Array2<f64>) -> Self {
        assert_eq!(col_range.len(), local.nrows());
        assert_eq!(col_range.len(), local.ncols());
        Self {
            col_range,
            local,
            prior_mean: gam_problem::CoefficientPriorMean::Zero,
            structure_hint: None,
            op: None,
        }
    }

    pub fn with_prior_mean(mut self, prior_mean: gam_problem::CoefficientPriorMean) -> Self {
        self.prior_mean = prior_mean;
        self
    }

    /// Attach an op-form penalty handle bit-equivalent to `local`.
    pub fn with_op(
        mut self,
        op: Option<std::sync::Arc<dyn crate::analytic_penalties::PenaltyOp>>,
    ) -> Self {
        self.op = op;
        self
    }

    pub fn ridge(col_range: Range<usize>, scale: f64) -> Self {
        let block_size = col_range.len();
        let mut local = Array2::<f64>::zeros((block_size, block_size));
        for i in 0..block_size {
            local[[i, i]] = scale;
        }
        Self {
            col_range,
            local,
            prior_mean: gam_problem::CoefficientPriorMean::Zero,
            structure_hint: Some(PenaltyStructureHint::Ridge(scale)),
            op: None,
        }
    }

    pub fn kronecker(
        col_range: Range<usize>,
        local: Array2<f64>,
        factors: Vec<Array2<f64>>,
    ) -> Self {
        assert_eq!(col_range.len(), local.nrows());
        assert_eq!(col_range.len(), local.ncols());
        Self {
            col_range,
            local,
            prior_mean: gam_problem::CoefficientPriorMean::Zero,
            structure_hint: Some(PenaltyStructureHint::Kronecker(factors)),
            op: None,
        }
    }

    /// Expand this blockwise penalty into a full `p_total × p_total` dense
    /// matrix (mostly zeros). Use sparingly — the whole point of blockwise
    /// storage is to avoid this allocation.
    pub fn to_global(&self, p_total: usize) -> Array2<f64> {
        let mut g = Array2::<f64>::zeros((p_total, p_total));
        let r = &self.col_range;
        assert!(
            r.end <= p_total && self.local.nrows() == r.len() && self.local.ncols() == r.len(),
            "BlockwisePenalty::to_global shape invariant violated: \
             col_range={}..{}, local={}x{}, p_total={}",
            r.start,
            r.end,
            self.local.nrows(),
            self.local.ncols(),
            p_total,
        );
        g.slice_mut(s![r.start..r.end, r.start..r.end])
            .assign(&self.local);
        g
    }

    /// Convert into a blockwise [`gam_problem::PenaltyMatrix`] without
    /// expanding to full dimensions.
    pub(crate) fn to_penalty_matrix(&self, total_dim: usize) -> gam_problem::PenaltyMatrix {
        gam_problem::PenaltyMatrix::Blockwise {
            local: self.local.clone(),
            col_range: self.col_range.clone(),
            total_dim,
        }
    }

    /// The block size of this penalty.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.col_range.len()
    }
}

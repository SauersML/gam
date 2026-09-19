//! The survival marginal-slope family's declared latent law (gam#2923): the law
//! a fit anchors on, materialised once per family — one grid for a global law,
//! one per training row for a local mixture, or the joint law of a `K ≥ 2` score
//! vector — together with the root slots its anchors are solved in.
//!
//! The anchoring equation itself, its solve, its implicit derivatives and the
//! slots are family-free and live in [`crate::latent_anchor`], re-exported here.
//! A row's two location channels solve in two slots: entry `0`, exit `1`.

pub(crate) use crate::latent_anchor::*;
use super::SurvivalInterceptSlotKind;
use std::sync::Arc;

/// Root slots per training row: the entry and the exit anchor.
pub(crate) const SURVIVAL_ANCHOR_SLOTS: usize = 2;

/// The root slot a location channel's anchor is solved in.
#[inline]
pub(crate) fn survival_anchor_slot(kind: SurvivalInterceptSlotKind) -> usize {
    match kind {
        SurvivalInterceptSlotKind::Entry => 0,
        SurvivalInterceptSlotKind::Exit => 1,
    }
}

/// The declared latent law a fit anchors on, materialised once per family.
///
/// `kind` is the persisted object — the same [`LatentMeasureKind`] the
/// Bernoulli family writes and the shared predictor replays — and the grids
/// are its per-row projection onto the scalar the shared slope reads: one
/// grid for a global law, one per training row for a local mixture.
#[derive(Clone, Debug)]
pub(crate) struct SurvivalLatentLaw {
    pub(crate) kind: crate::bms::LatentMeasureKind,
    grids: LawGrids,
    /// The roots solved on this law (gam#2928), shared by every family the
    /// search builds on it.
    roots: Arc<AnchorRootCache>,
}

#[derive(Clone, Debug)]
enum LawGrids {
    Global(AnchorGridOwned),
    PerRow(Vec<AnchorGridOwned>),
    /// The joint law of a `K ≥ 2` score vector (gam#2929). It has no scalar
    /// grid: each row projects it onto its own slope vector, and only the
    /// per-score vector program reads it.
    Joint(Arc<super::JointLatentLawRuntime>),
}

impl SurvivalLatentLaw {
    /// Materialise the law's per-row grids for `n` training rows. `None` for
    /// the standard-normal law, on which the closed form is the anchor.
    pub(crate) fn from_kind(
        kind: &crate::bms::LatentMeasureKind,
        n: usize,
    ) -> Result<Option<Self>, String> {
        use crate::bms::LatentMeasureKind;
        kind.validate("survival marginal-slope declared latent law")?;
        let grids = match kind {
            LatentMeasureKind::StandardNormal => return Ok(None),
            LatentMeasureKind::GlobalEmpirical { grid } => {
                LawGrids::Global(AnchorGridOwned::from_grid(grid))
            }
            LatentMeasureKind::LocalEmpirical { .. } => {
                let mut per_row = Vec::with_capacity(n);
                for row in 0..n {
                    let grid = kind.empirical_grid_for_training_row(row)?.ok_or_else(|| {
                        format!(
                            "survival marginal-slope local latent law produced no grid for row {row}"
                        )
                    })?;
                    per_row.push(AnchorGridOwned::from_grid(&grid));
                }
                LawGrids::PerRow(per_row)
            }
        };
        let shared_across_rows = matches!(grids, LawGrids::Global(_));
        Ok(Some(Self {
            kind: kind.clone(),
            grids,
            roots: Arc::new(AnchorRootCache::new(n, SURVIVAL_ANCHOR_SLOTS, shared_across_rows)),
        }))
    }

    /// A joint law of the score vector (gam#2929). `kind` is the primary
    /// score's measure, which is what named the law's configuration. It carries
    /// no root slots: the per-score vector program solves its own roots.
    pub(crate) fn from_joint(
        kind: crate::bms::LatentMeasureKind,
        runtime: super::JointLatentLawRuntime,
    ) -> Self {
        Self {
            kind,
            grids: LawGrids::Joint(Arc::new(runtime)),
            roots: Arc::new(AnchorRootCache::new(0, SURVIVAL_ANCHOR_SLOTS, false)),
        }
    }

    /// The joint law, when this is one.
    #[inline]
    pub(crate) fn joint(&self) -> Option<&super::JointLatentLawRuntime> {
        match &self.grids {
            LawGrids::Joint(runtime) => Some(runtime),
            LawGrids::Global(_) | LawGrids::PerRow(_) => None,
        }
    }

    /// Row `row`'s law with the roots already solved on it.
    #[inline]
    pub(crate) fn row_context(&self, row: usize) -> AnchorRowContext<'_> {
        AnchorRowContext {
            grid: self.row(row),
            roots: Some(&self.roots),
        }
    }

    /// The law of row `row`.
    ///
    /// A joint law has no scalar grid. The per-score vector lanes that read it
    /// never ask for one, and every consumer of the scalar frames refuses a
    /// per-score family before solving on a row's grid, so the empty grid only
    /// keeps this match total without inventing a law the fit did not declare.
    #[inline]
    pub(crate) fn row(&self, row: usize) -> AnchorGrid<'_> {
        match &self.grids {
            LawGrids::Global(grid) => grid.view(),
            LawGrids::PerRow(grids) => grids[row].view(),
            LawGrids::Joint(_) => AnchorGrid {
                nodes: &[],
                weights: &[],
                log_weights: &[],
            },
        }
    }

    /// The scalar law row `row` anchors on or, where no row is named, the one
    /// law every row shares. A local law has no shared grid and a joint law has
    /// no scalar grid, so both are refused by name rather than read as another
    /// law (gam#2948).
    pub(crate) fn scalar_grid(&self, row: Option<usize>) -> Result<AnchorGrid<'_>, String> {
        match (&self.grids, row) {
            (LawGrids::Global(grid), _) => Ok(grid.view()),
            (LawGrids::PerRow(grids), Some(row)) => {
                grids.get(row).map(AnchorGridOwned::view).ok_or_else(|| {
                    format!(
                        "survival marginal-slope local latent law has no grid for row {row} of {}",
                        grids.len()
                    )
                })
            }
            (LawGrids::PerRow(_), None) => Err(
                "survival marginal-slope local latent law: a solve that names no row has no \
                 mixture to anchor on"
                    .to_string(),
            ),
            (LawGrids::Joint(_), _) => Err(
                "survival marginal-slope joint latent law of K ≥ 2 scores has no scalar law for \
                 the single-score flex row program to anchor on"
                    .to_string(),
            ),
        }
    }
}

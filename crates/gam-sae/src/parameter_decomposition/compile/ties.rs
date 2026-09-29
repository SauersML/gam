//! Declared ties between stored blocks, checked on a compiled plan.
//!
//! A tie a framework keeps by aliasing (a tied embedding and output head bound to one
//! `Parameter`) is one storage tensor in the registry, and a compiled edit is always global
//! on storage, so it moves every tied use together by construction. A tie a framework
//! keeps by storing equal values twice, or by a linear parametrization, is declared here:
//!
//! ```text
//! second_block = scale · first_block        (or scale · first_blockᵀ when transposed)
//! ```
//!
//! for a shared key/value head duplicated across query groups, a tied pair stored twice
//! (transposed), a symmetric parametrization (a block tied to its own transpose), or a
//! scaled copy. A plan keeps the tie iff the two block edits satisfy the same equation, so
//! [`check_ties`] forms each block edit from its factors and reports every tie whose two
//! sides differ by more than their evaluation band. A broken tie is reported, never
//! repaired by untying: the compiler's callers turn it into a descriptive status.

use std::ops::Range;

use gam_linalg::roundoff::accumulation_growth;
use gam_math::roundoff::{UNIT_ROUNDOFF, inflated};
use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array2, s};

use super::super::lift::TensorId;
use super::{CompileError, NativeEditPlan};

/// A block of a stored matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockRef {
    pub storage: TensorId,
    pub rows: Range<usize>,
    pub cols: Range<usize>,
}

/// `second = scale · first` (transposed when declared).
#[derive(Clone, Debug, PartialEq)]
pub struct TieConstraint {
    pub first: BlockRef,
    pub second: BlockRef,
    pub transposed: bool,
    pub scale: f64,
}

/// A tie the plan breaks: the entry of the second block most resolved from its tied value.
#[derive(Clone, Debug, PartialEq)]
pub struct TieViolation {
    /// Index of the tie in the declared list.
    pub tie: usize,
    /// Position within the second block.
    pub row: usize,
    pub col: usize,
    /// `|Δsecond − scale · Δfirst|` at that entry, and its evaluation band.
    pub difference: f64,
    pub band: f64,
}

/// The edit of one block and its per-entry band, zero when the plan leaves the storage.
fn block_edit(
    plan: &NativeEditPlan,
    block: &BlockRef,
    governor: &MemoryGovernor,
) -> Result<(Array2<f64>, Array2<f64>), CompileError> {
    let (rows, cols) = (block.rows.len(), block.cols.len());
    let reservation = governor.try_reserve_dense_f64_copies(rows, cols, 2, "tie block edit")?;
    let found = plan.edits().iter().find(|edit| edit.storage == block.storage);
    let out = match found {
        None => (Array2::zeros((rows, cols)), Array2::zeros((rows, cols))),
        Some(edit) => {
            let (out_dim, in_dim) = (edit.delta.output_dim(), edit.delta.input_dim());
            if block.rows.end > out_dim || block.cols.end > in_dim {
                return Err(CompileError::InvalidDeclaration {
                    what: "tie block",
                    reason: format!(
                        "block {:?} x {:?} lies outside {:?} ({out_dim} x {in_dim})",
                        block.rows, block.cols, block.storage.0
                    ),
                });
            }
            let left = edit.delta.left().slice(s![block.rows.clone(), ..]).to_owned();
            let right = edit.delta.right().slice(s![block.cols.clone(), ..]).to_owned();
            let values = left.dot(&right.t());
            let bands = left.mapv(f64::abs).dot(&right.mapv(f64::abs).t()) * accumulation_growth(edit.delta.term_count());
            (values, bands)
        }
    };
    drop(reservation);
    Ok(out)
}

/// Every declared tie the plan breaks, in declaration order.
pub fn check_ties(
    plan: &NativeEditPlan,
    ties: &[TieConstraint],
    governor: &MemoryGovernor,
) -> Result<Vec<TieViolation>, CompileError> {
    let mut violations = Vec::new();
    for (index, tie) in ties.iter().enumerate() {
        if !(tie.scale.is_finite()) {
            return Err(CompileError::InvalidDeclaration {
                what: "tie scale",
                reason: format!("must be finite; got {}", tie.scale),
            });
        }
        let (first_rows, first_cols) = (tie.first.rows.len(), tie.first.cols.len());
        let expected = if tie.transposed {
            (first_cols, first_rows)
        } else {
            (first_rows, first_cols)
        };
        let found = (tie.second.rows.len(), tie.second.cols.len());
        if expected != found {
            return Err(CompileError::Shape {
                what: "tied block",
                expected,
                found,
            });
        }
        let (first, first_band) = block_edit(plan, &tie.first, governor)?;
        let (second, second_band) = block_edit(plan, &tie.second, governor)?;
        let (first, first_band) = if tie.transposed {
            (first.reversed_axes(), first_band.reversed_axes())
        } else {
            (first, first_band)
        };
        let mut worst: Option<TieViolation> = None;
        for ((row, col), &value) in second.indexed_iter() {
            let tied = tie.scale * first[[row, col]];
            let difference = (value - tied).abs();
            let band = inflated(
                second_band[[row, col]]
                    + tie.scale.abs() * first_band[[row, col]]
                    + UNIT_ROUNDOFF * (tied.abs() + difference),
                2,
            );
            if difference > band && worst.as_ref().is_none_or(|w| difference - band > w.difference - w.band) {
                worst = Some(TieViolation {
                    tie: index,
                    row,
                    col,
                    difference,
                    band,
                });
            }
        }
        violations.extend(worst);
    }
    Ok(violations)
}

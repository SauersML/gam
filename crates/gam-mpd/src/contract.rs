//! The contract of an imported model (#2951): the input family whose behaviour is explained, and
//! what the program's output holds.
//!
//! A [`FamilyKind::Complete`] family is the whole declared input domain. A [`FamilyKind::Sample`]
//! family is a harvest from a population the contract names, its rows grouped into independent
//! units (documents, episodes).

use super::operator_program::{Declarations, FamilyInputs};

/// What the family is.
#[derive(Clone, Debug, PartialEq)]
pub enum FamilyKind {
    /// Every input of the declared domain.
    Complete { description: String },
    /// Independent units drawn from a named population, with the confidence of the population
    /// bounds; `units[row]` is the unit each drawn row belongs to.
    Sample { population: String, confidence: f64, units: Vec<usize> },
}

/// A declared contract: the behaviour to explain.
#[derive(Clone, Debug, PartialEq)]
pub struct Contract {
    pub declarations: Declarations,
    pub family: FamilyInputs,
    pub kind: FamilyKind,
    /// `n`: how many observations of each row's output distribution the code explains.
    pub observations: u64,
    /// How many categorical distributions each input's output holds: the output node's width is
    /// `readouts` blocks of equal width (one per readout position, say), each a distribution row.
    pub readouts: usize,
    /// Per readout, the slots it may read; `None` declares no causal restriction.
    pub readout_slots: Option<Vec<Vec<usize>>>,
}

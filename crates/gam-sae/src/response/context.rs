//! Context-dependent declared laws: one frame shared across contexts, the exact per-context errors beside their
//! average, and the contract for what a compiled program reads at execution (#2946).
//!
//! # Law
//!
//! Real contexts differ. Context `c` has its own baseline `h₀(c)` and possibly its own declared law `L(c)`, so under
//! `h = h₀(c) + L(c) Z`, `Z ~ N(0, I_d)`, the block `F(h) = Σ_j u_j σ(b_j + w_jᵀ h)` reads biases `b + W h₀(c)` and
//! readers `W L(c)` ([`ContextBlocks::new`]). One frame `Q` (`d × k`, `P = Q Qᵀ`) of the intervention coordinates is
//! shared across contexts, and the context `C ~ π` is independent of `Z`.
//!
//! # Risk
//!
//! Context `c`'s discarded-input error `𝓔_c(P) = V_c(I) − V_c(P)` is exact (R4). The objective for one shared frame is
//! the context-law average `𝓔(P) = Σ_c π_c 𝓔_c(P)`, and differentiation is linear, so its horizontal gradient is the
//! sum of the per-context gradients, `∇_Q 𝓔 = −Σ_c π_c ∇_Q V_c`.
//! - Under a declared law ([`ContextLaw::Enumerated`]) the average is exact.
//! - Over draws from the context population ([`ContextLaw::Sampled`]) every term is exact and only the average is
//!   estimated, `𝓔̂ = n⁻¹ Σ_i 𝓔_{c_i}(P)`. It is reported with its jackknife standard error, which for a mean is
//!   `s/√n`.
//!
//! # Contrast
//!
//! A frame optimal for the average may be poor for a rare context, because the average weighs each context's loss by
//! its mass. The report carries every context's row and the worst row beside the average: average error is not
//! worst-case faithfulness.
//!
//! # Available inputs
//!
//! A compiled program may read only what is available at execution: the retained coordinates and declared context
//! features. `𝓔_c(P)` is the error of context `c`'s own best response `Ḡ_c(PZ) = E[F_c(Z) | PZ]`, so the average is the
//! error of a program that knows the context. A program that cannot tell the contexts apart has best response
//! `g*(PZ) = Σ_c π_c Ḡ_c(PZ)`, and by Pythagoras pays `Σ_c π_c E‖Ḡ_c − g*‖²_M ≥ 0` more. Reporting the average for it
//! would be a silent free gate on the context. So the average is reported only under a typed contract:
//! - [`ContextAccess::PricedIndex`]: the context index is an input to `g`, charged per executed row. A charge below the
//!   law's entropy `H(π)` is refused, because no lossless code of the index is shorter on average (Shannon). An
//!   unpriced index is therefore refused whenever two contexts carry mass. A sampled population declares no law to
//!   price against, so there an index price is refused.
//! - [`ContextAccess::ContextSpecific`]: one explanation per context, and no single program is claimed.

use std::fmt;

use gam_linalg::roundoff::accumulation_growth;
use ndarray::{Array1, Array2, ArrayView2};

use super::raw_block::UnabsorbedBlock;
use super::subspace::{KnownBlock, ResponseError};

/// One context's declared intervention `h = h₀(c) + L(c) Z` with `Z ~ N(0, I_d)`.
#[derive(Clone, Debug)]
pub struct ContextDeclaredLaw {
    /// `h₀(c)`, length `D`.
    pub baseline: Array1<f64>,
    /// `L(c)`, `D × d`. Every context shares `d`, so one frame spans the same intervention coordinates in each.
    pub loading: Array2<f64>,
}

/// How the contexts an evaluation averages over were obtained.
#[derive(Clone, Debug, PartialEq)]
pub enum ContextLaw {
    /// A finite declared law: context `c` carries mass `masses[c] > 0`, and the law is the masses over their sum. Every
    /// average is exact.
    Enumerated { masses: Vec<f64> },
    /// Independent draws from the context population, one draw per context. Every average is an estimate, reported
    /// with its jackknife standard error.
    Sampled,
}

/// The contract under which a context-law average is the error of what is explained.
#[derive(Clone, Debug, PartialEq)]
pub enum ContextAccess {
    /// The context index is an input to the compiled program, charged `nats_per_row` per executed row.
    PricedIndex { nats_per_row: f64 },
    /// One explanation per context: no single program is claimed.
    ContextSpecific,
}

/// Why a context-law evaluation refused.
#[derive(Clone, Debug, PartialEq)]
pub enum ContextResponseError {
    /// No contexts were declared.
    NoContexts,
    /// A declared array has the wrong length.
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        got: usize,
    },
    /// A declared mass is not finite and positive.
    InvalidMass { context: usize, mass: f64 },
    /// The index is charged below the law's entropy, with the entropy's rounding band.
    UnderpricedContextIndex {
        nats_per_row: f64,
        entropy_nats: f64,
        band: f64,
    },
    /// A sampled population declares no law the index price could be checked against.
    IndexPriceWithoutDeclaredLaw,
    /// Too few draws for a sampled average and its jackknife.
    TooFewDraws { draws: usize, required: usize },
    /// The retained-response operator refused one context's block or the frame.
    Response { context: usize, error: ResponseError },
}

impl fmt::Display for ContextResponseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoContexts => write!(f, "context response: no contexts were declared"),
            Self::DimensionMismatch {
                context,
                expected,
                got,
            } => write!(f, "context response: {context}: expected {expected}, got {got}"),
            Self::InvalidMass { context, mass } => write!(
                f,
                "context response: context {context} has mass {mass}; a declared law gives every context a finite \
                 positive mass"
            ),
            Self::UnderpricedContextIndex {
                nats_per_row,
                entropy_nats,
                band,
            } => write!(
                f,
                "context response: the context index is charged {nats_per_row} nats per row, below the law's entropy \
                 {entropy_nats} (band {band}); no lossless code of the index is shorter on average, so the index would \
                 be a free gate"
            ),
            Self::IndexPriceWithoutDeclaredLaw => write!(
                f,
                "context response: a sampled population declares no law to price the context index against; declare \
                 the law or label the explanation context-specific"
            ),
            Self::TooFewDraws { draws, required } => write!(
                f,
                "context response: {draws} draws; a sampled average and its jackknife need at least {required}"
            ),
            Self::Response { context, error } => write!(f, "context response: context {context}: {error}"),
        }
    }
}

impl std::error::Error for ContextResponseError {}

/// One context's row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContextRow {
    /// `𝓔_c(P) = V_c(I) − V_c(P)`, this context's discarded-input error.
    pub error: f64,
    /// `V_c(I)`, this context's output variance.
    pub total_variance: f64,
}

/// The context-law evaluation of one shared frame.
#[derive(Clone, Debug)]
pub struct ContextResponseReport {
    /// The contract the average is reported under.
    pub access: ContextAccess,
    /// `π_c`: the declared law's weights, or `1/n` per draw under a sampled law.
    pub weights: Vec<f64>,
    /// The charge for reading the context index, in nats per executed row, under [`ContextAccess::PricedIndex`]. It is
    /// a reported line, at or above the law's entropy, and never enters the error.
    pub index_nats_per_row: Option<f64>,
    /// `Σ_c π_c 𝓔_c(P)`, or its estimate under a sampled law.
    pub error: f64,
    /// The jackknife standard error over draws under a sampled law; `None` under a declared law.
    pub standard_error: Option<f64>,
    /// Per-context rows, in context order.
    pub rows: Vec<ContextRow>,
    /// The first context with the largest row error: under a sampled law, the worst sampled context.
    pub worst_context: usize,
    /// `∇_Q 𝓔 = −Σ_c π_c ∇_Q V_c`, a `d × k` horizontal tangent whose negative is the descent direction.
    pub horizontal_gradient: Array2<f64>,
}

/// A known block under each context's declared law.
#[derive(Clone, Debug)]
pub struct ContextBlocks {
    blocks: Vec<KnownBlock>,
}

impl ContextBlocks {
    /// Absorb each context's law into the block: readers `W L(c)` and biases `b + W h₀(c)`.
    pub fn new(unabsorbed: UnabsorbedBlock, laws: Vec<ContextDeclaredLaw>) -> Result<Self, ContextResponseError> {
        let Some(first) = laws.first() else {
            return Err(ContextResponseError::NoContexts);
        };
        let input_dim = first.loading.ncols();
        let mut blocks = Vec::with_capacity(laws.len());
        for (context, law) in laws.iter().enumerate() {
            require_length("context loading columns", input_dim, law.loading.ncols())?;
            let block = unabsorbed
                .absorb(law.baseline.view(), law.loading.view())
                .map_err(|error| ContextResponseError::Response { context, error })?;
            blocks.push(block);
        }
        Ok(Self { blocks })
    }

    /// The number of contexts.
    pub fn context_count(&self) -> usize {
        self.blocks.len()
    }

    /// Context `c`'s block under its law.
    pub fn block(&self, context: usize) -> &KnownBlock {
        &self.blocks[context]
    }

    /// The context-law average of the discarded-input error at `frame` (`d × k`, orthonormal columns), per-context rows
    /// and the horizontal gradient, under `law` and the contract `access`.
    pub fn evaluate(
        &self,
        frame: ArrayView2<'_, f64>,
        law: &ContextLaw,
        access: &ContextAccess,
    ) -> Result<ContextResponseReport, ContextResponseError> {
        let count = self.context_count();
        let weights = match law {
            ContextLaw::Enumerated { masses } => Some(declared_weights(masses, count)?),
            ContextLaw::Sampled => None,
        };
        if let ContextAccess::PricedIndex { nats_per_row } = access {
            let Some(declared) = weights.as_deref() else {
                return Err(ContextResponseError::IndexPriceWithoutDeclaredLaw);
            };
            let (entropy_nats, band) = law_entropy(declared);
            if !(*nats_per_row >= entropy_nats - band) {
                return Err(ContextResponseError::UnderpricedContextIndex {
                    nats_per_row: *nats_per_row,
                    entropy_nats,
                    band,
                });
            }
        }
        let sampled = weights.is_none();
        if sampled && count < 2 {
            return Err(ContextResponseError::TooFewDraws {
                draws: count,
                required: 2,
            });
        }
        let draws = count as f64;
        let weights = weights.unwrap_or_else(|| vec![1.0 / draws; count]);
        let mut rows = Vec::with_capacity(count);
        let mut horizontal_gradient = Array2::<f64>::zeros(frame.dim());
        let mut error = 0.0;
        for (context, block) in self.blocks.iter().enumerate() {
            let evaluation = block
                .explained_variance_gradient(frame)
                .map_err(|error| ContextResponseError::Response { context, error })?;
            error += weights[context] * evaluation.discarded_error.value;
            horizontal_gradient.scaled_add(-weights[context], &evaluation.horizontal_gradient);
            rows.push(ContextRow {
                error: evaluation.discarded_error.value,
                total_variance: block.total_variance().value,
            });
        }
        let standard_error = sampled.then(|| {
            let sum: f64 = rows.iter().map(|row| row.error).sum();
            let leave_one_out: Vec<f64> = rows.iter().map(|row| (sum - row.error) / (draws - 1.0)).collect();
            jackknife_standard_error(&leave_one_out)
        });
        let index_nats_per_row = match access {
            ContextAccess::PricedIndex { nats_per_row } => Some(*nats_per_row),
            ContextAccess::ContextSpecific => None,
        };
        Ok(ContextResponseReport {
            access: access.clone(),
            weights,
            index_nats_per_row,
            error,
            standard_error,
            worst_context: worst_row(&rows),
            rows,
            horizontal_gradient,
        })
    }
}

/// The weights `π_c = m_c/Σm` of a declared law, or why the masses are not a law over `count` contexts.
fn declared_weights(masses: &[f64], count: usize) -> Result<Vec<f64>, ContextResponseError> {
    require_length("context law masses", count, masses.len())?;
    for (context, &mass) in masses.iter().enumerate() {
        if !(mass.is_finite() && mass > 0.0) {
            return Err(ContextResponseError::InvalidMass { context, mass });
        }
    }
    let total: f64 = masses.iter().sum();
    Ok(masses.iter().map(|mass| mass / total).collect())
}

/// `H(π) = −Σ_c π_c ln π_c` nats and its rounding band: the quotient that formed each weight, its logarithm and the
/// product, and the additions (Higham, *ASNA* §3.1).
fn law_entropy(weights: &[f64]) -> (f64, f64) {
    let mut entropy = 0.0;
    let mut absolute = 0.0;
    for &weight in weights {
        let term = -weight * weight.ln();
        entropy += term;
        absolute += term.abs();
    }
    let band = accumulation_growth(3 + weights.len().saturating_sub(1)) * absolute;
    (entropy, band)
}

/// The jackknife standard error from the leave-one-draw-out estimates.
fn jackknife_standard_error(leave_one_out: &[f64]) -> f64 {
    let draws = leave_one_out.len() as f64;
    let mean = leave_one_out.iter().sum::<f64>() / draws;
    let spread: f64 = leave_one_out.iter().map(|estimate| (estimate - mean) * (estimate - mean)).sum();
    ((draws - 1.0) / draws * spread).sqrt()
}

/// The first context with the largest row error.
fn worst_row(rows: &[ContextRow]) -> usize {
    (0..rows.len()).fold(0, |worst, context| {
        if rows[context].error > rows[worst].error { context } else { worst }
    })
}

fn require_length(context: &'static str, expected: usize, got: usize) -> Result<(), ContextResponseError> {
    if expected == got {
        Ok(())
    } else {
        Err(ContextResponseError::DimensionMismatch {
            context,
            expected,
            got,
        })
    }
}

#[cfg(test)]
#[path = "context_tests.rs"]
mod context_tests;

use std::ops::Range;

/// Generic semantic classification for a term in the engine layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EngineTermKind {
    Intercept,
    Linear,
    Smooth,
    Interaction,
    Custom,
}

/// Penalty allocation policy for a term during layout construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PenaltySpec {
    None,
    New { count: usize },
    Existing(Vec<usize>),
}

/// Term descriptor used by the generic engine layout builder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineTermSpec {
    pub kind: EngineTermKind,
    pub width: usize,
    pub penalty_spec: PenaltySpec,
}

impl EngineTermSpec {
    pub fn unpenalized(kind: EngineTermKind, width: usize) -> Self {
        Self {
            kind,
            width,
            penalty_spec: PenaltySpec::None,
        }
    }

    pub fn penalized(kind: EngineTermKind, width: usize, penalty_count: usize) -> Self {
        Self {
            kind,
            width,
            penalty_spec: PenaltySpec::New {
                count: penalty_count,
            },
        }
    }
}

/// Resolved layout entry for a single term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineTerm {
    pub kind: EngineTermKind,
    pub col_range: Range<usize>,
    pub penalty_indices: Vec<usize>,
}

/// Generic engine layout that is independent of domain-specific model configs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineLayout {
    pub terms: Vec<EngineTerm>,
    pub total_coeffs: usize,
    pub num_penalties: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutBuildError {
    pub message: String,
}

impl LayoutBuildError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Incremental builder for constructing generic engine layouts.
#[derive(Debug, Clone)]
pub struct EngineLayoutBuilder {
    next_col: usize,
    next_penalty: usize,
    terms: Vec<EngineTerm>,
}

impl EngineLayoutBuilder {
    pub fn new() -> Self {
        Self {
            next_col: 0,
            next_penalty: 0,
            terms: Vec::new(),
        }
    }

    pub fn with_offsets(start_col: usize, start_penalty: usize) -> Self {
        Self {
            next_col: start_col,
            next_penalty: start_penalty,
            terms: Vec::new(),
        }
    }

    pub fn push_term(&mut self, spec: EngineTermSpec) -> Result<usize, LayoutBuildError> {
        if spec.width == 0 {
            return Err(LayoutBuildError::new(
                "term width must be positive for layout construction",
            ));
        }

        let penalties = match spec.penalty_spec {
            PenaltySpec::None => Vec::new(),
            PenaltySpec::New { count } => {
                if count == 0 {
                    return Err(LayoutBuildError::new(
                        "new-penalty spec must request at least one penalty",
                    ));
                }
                let start = self.next_penalty;
                self.next_penalty += count;
                (start..start + count).collect()
            }
            PenaltySpec::Existing(indices) => {
                if indices.is_empty() {
                    return Err(LayoutBuildError::new(
                        "existing-penalty spec must provide at least one index",
                    ));
                }
                indices
            }
        };

        let col_range = self.next_col..self.next_col + spec.width;
        self.next_col += spec.width;
        self.terms.push(EngineTerm {
            kind: spec.kind,
            col_range,
            penalty_indices: penalties,
        });
        Ok(self.terms.len() - 1)
    }

    pub fn build(self) -> EngineLayout {
        EngineLayout {
            terms: self.terms,
            total_coeffs: self.next_col,
            num_penalties: self.next_penalty,
        }
    }
}

impl Default for EngineLayoutBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_assigns_columns_and_penalties_sequentially() {
        let mut b = EngineLayoutBuilder::new();
        let intercept = b
            .push_term(EngineTermSpec::unpenalized(EngineTermKind::Intercept, 1))
            .expect("intercept");
        let smooth = b
            .push_term(EngineTermSpec::penalized(EngineTermKind::Smooth, 5, 1))
            .expect("smooth");
        let interaction = b
            .push_term(EngineTermSpec::penalized(EngineTermKind::Interaction, 9, 3))
            .expect("interaction");
        let layout = b.build();

        assert_eq!(intercept, 0);
        assert_eq!(smooth, 1);
        assert_eq!(interaction, 2);
        assert_eq!(layout.total_coeffs, 15);
        assert_eq!(layout.num_penalties, 4);
        assert_eq!(layout.terms[0].col_range, 0..1);
        assert_eq!(layout.terms[1].col_range, 1..6);
        assert_eq!(layout.terms[2].col_range, 6..15);
        assert_eq!(layout.terms[1].penalty_indices, vec![0]);
        assert_eq!(layout.terms[2].penalty_indices, vec![1, 2, 3]);
    }

    #[test]
    fn builder_respects_existing_penalty_indices() {
        let mut b = EngineLayoutBuilder::with_offsets(10, 4);
        b.push_term(EngineTermSpec {
            kind: EngineTermKind::Custom,
            width: 3,
            penalty_spec: PenaltySpec::Existing(vec![1, 7]),
        })
        .expect("custom");
        let layout = b.build();
        assert_eq!(layout.total_coeffs, 13);
        assert_eq!(layout.num_penalties, 4);
        assert_eq!(layout.terms[0].penalty_indices, vec![1, 7]);
    }

    #[test]
    fn builder_rejects_invalid_specs() {
        let mut b = EngineLayoutBuilder::new();
        let err = b
            .push_term(EngineTermSpec::unpenalized(EngineTermKind::Linear, 0))
            .expect_err("zero width");
        assert!(err.message.contains("width"));

        let err = b
            .push_term(EngineTermSpec {
                kind: EngineTermKind::Smooth,
                width: 4,
                penalty_spec: PenaltySpec::New { count: 0 },
            })
            .expect_err("zero penalty count");
        assert!(err.message.contains("at least one"));
    }
}

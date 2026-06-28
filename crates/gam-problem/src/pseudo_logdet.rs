/// How the penalized Hessian's log-determinant and derivatives treat the
/// spectrum below the stability floor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PseudoLogdetMode {
    /// Keep every eigenpair in the smooth spectral regularizer.
    #[default]
    Smooth,
    /// Exclude numerical null-space directions consistently from pseudo-logdet
    /// and derivative traces.
    HardPseudo,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_smooth() {
        assert_eq!(PseudoLogdetMode::default(), PseudoLogdetMode::Smooth);
    }

    #[test]
    fn variants_are_distinct() {
        assert_ne!(PseudoLogdetMode::Smooth, PseudoLogdetMode::HardPseudo);
    }
}

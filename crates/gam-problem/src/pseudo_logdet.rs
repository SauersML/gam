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

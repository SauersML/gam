/// Round-off floor for an eigenvalue gap, `ε·max(n, 1)·max(|scale|, 1)` for `n`
/// eigenvalues of magnitude `scale`: a gap below it is not resolved above
/// eigensolver round-off.
pub fn eigen_gap_threshold(eigen_scale: f64, eigen_count: usize) -> f64 {
    f64::EPSILON * (eigen_count.max(1) as f64) * eigen_scale.abs().max(1.0)
}

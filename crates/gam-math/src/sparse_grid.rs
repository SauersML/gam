//! Isotropic Smolyak combination over one-dimensional standard-normal rules, streamed node by node.
//!
//! One owner for the construction the multinomial posterior certifies with (#1082): the level's composition bounds,
//! the signed combination coefficients `(−1)^{q−total}·C(rank−1, q−total)`, the composition and tensor recursions
//! that stream every node with its signed product weight, and the compensated accumulator a caller reduces its
//! integrand into. The caller owns the integrand, the certificate and the messages; every node reaches it through a
//! visitor, in the order and with the weight products the recursions form.

/// A one-dimensional rule in standard-normal coordinates: nodes and their expectation weights.
pub trait StandardNormalRule {
    fn nodes(&self) -> &[f64];
    fn weights(&self) -> &[f64];
}

/// `C(n, k)` overflowed `f64`. `k` is the reduced `min(k, n − k)` the product ran over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinomialOverflow {
    pub n: usize,
    pub k: usize,
}

/// The binomial coefficient `C(n, k)` as a product of ratios, `0` when `k > n`.
pub fn binomial_as_f64(n: usize, k: usize) -> Result<f64, BinomialOverflow> {
    if k > n {
        return Ok(0.0);
    }
    let k = k.min(n - k);
    let mut value = 1.0_f64;
    for step in 1..=k {
        value *= (n - k + step) as f64 / step as f64;
        if !value.is_finite() {
            return Err(BinomialOverflow { n, k });
        }
    }
    Ok(value)
}

/// The composition totals one isotropic Smolyak level combines: `q = rank + level`, and the lowest total the signed
/// combination reaches, `max(q − (rank − 1), rank)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmolyakBounds {
    pub q: usize,
    pub lower_total: usize,
}

/// `rank + level` overflowed `usize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmolyakIndexOverflow;

/// The composition totals of isotropic Smolyak `level` in `rank` directions.
pub fn isotropic_smolyak_bounds(rank: usize, level: usize) -> Result<SmolyakBounds, SmolyakIndexOverflow> {
    let q = rank.checked_add(level).ok_or(SmolyakIndexOverflow)?;
    let lower_total = q.saturating_sub(rank.saturating_sub(1)).max(rank);
    Ok(SmolyakBounds { q, lower_total })
}

/// Why streaming a Smolyak level stopped: the combination coefficient overflowed, or the visitor refused a node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmolyakLevelError<E> {
    BinomialOverflow(BinomialOverflow),
    Visit(E),
}

/// Stream every node of isotropic Smolyak level `bounds` over `rules`, where rule index `i` is `rules[i − 1]`.
///
/// For each total from `bounds.lower_total` to `bounds.q`, every composition of the total into `rank` positive
/// indices is streamed as the tensor product of its rules, with the signed combination coefficient as the starting
/// weight. `z` is the caller's standard-normal coordinate buffer, one entry per direction.
pub fn stream_isotropic_smolyak_level<R, E>(
    rules: &[R],
    rank: usize,
    bounds: SmolyakBounds,
    z: &mut [f64],
    visit: &mut impl FnMut(&[f64], f64) -> Result<(), E>,
) -> Result<(), SmolyakLevelError<E>>
where
    R: StandardNormalRule,
{
    let mut indices = vec![1usize; rank];
    for total in bounds.lower_total..=bounds.q {
        let alternating_power = bounds.q - total;
        let mut coefficient =
            binomial_as_f64(rank - 1, alternating_power).map_err(SmolyakLevelError::BinomialOverflow)?;
        if alternating_power % 2 == 1 {
            coefficient = -coefficient;
        }
        stream_compositions(rules, 0, total, &mut indices, coefficient, z, visit)
            .map_err(SmolyakLevelError::Visit)?;
    }
    Ok(())
}

/// Stream every composition of `remaining` into the positive indices `indices[position..]`, each as the tensor
/// product of its rules starting from `coefficient`.
pub fn stream_compositions<R, E>(
    rules: &[R],
    position: usize,
    remaining: usize,
    indices: &mut [usize],
    coefficient: f64,
    z: &mut [f64],
    visit: &mut impl FnMut(&[f64], f64) -> Result<(), E>,
) -> Result<(), E>
where
    R: StandardNormalRule,
{
    let dimensions_left = indices.len() - position;
    if dimensions_left == 1 {
        if remaining == 0 {
            return Ok(());
        }
        indices[position] = remaining;
        return stream_tensor(rules, 0, indices, coefficient, z, visit);
    }
    let maximum_here = remaining.saturating_sub(dimensions_left - 1);
    for index in 1..=maximum_here {
        indices[position] = index;
        stream_compositions(rules, position + 1, remaining - index, indices, coefficient, z, visit)?;
    }
    Ok(())
}

/// Stream the tensor product of `rules[indices[d] − 1]` over directions `axis..`, multiplying `weight` by each
/// direction's node weight in direction order.
pub fn stream_tensor<R, E>(
    rules: &[R],
    axis: usize,
    indices: &[usize],
    weight: f64,
    z: &mut [f64],
    visit: &mut impl FnMut(&[f64], f64) -> Result<(), E>,
) -> Result<(), E>
where
    R: StandardNormalRule,
{
    if axis == indices.len() {
        return visit(z, weight);
    }
    let rule_index = indices[axis] - 1;
    let node_count = rules[rule_index].nodes().len();
    for node_index in 0..node_count {
        let node = rules[rule_index].nodes()[node_index];
        let node_weight = rules[rule_index].weights()[node_index];
        z[axis] = node;
        stream_tensor(rules, axis + 1, indices, weight * node_weight, z, visit)?;
    }
    Ok(())
}

/// Tensor product over one explicitly chosen rule per direction.
///
/// [`stream_tensor`] resolves each direction through `rules[index − 1]`, which forces the rule cache to be dense; a
/// caller whose indices double resolves its directions once and hands them over directly.
pub fn stream_axes<R, E>(
    axes: &[&R],
    axis: usize,
    weight: f64,
    z: &mut [f64],
    visit: &mut impl FnMut(&[f64], f64) -> Result<(), E>,
) -> Result<(), E>
where
    R: StandardNormalRule,
{
    if axis == axes.len() {
        return visit(z, weight);
    }
    let node_count = axes[axis].nodes().len();
    for node_index in 0..node_count {
        let node = axes[axis].nodes()[node_index];
        let node_weight = axes[axis].weights()[node_index];
        z[axis] = node;
        stream_axes(axes, axis + 1, weight * node_weight, z, visit)?;
    }
    Ok(())
}

/// One Kahan–Babuška (Neumaier) step: add `value` to `sum` and move the exact rounding error of that addition
/// into `correction`. Unlike plain Kahan, the error is taken from whichever operand is smaller in magnitude, so an
/// addend larger than the running sum does not lose the running sum's low-order bits.
#[inline]
fn neumaier_step(sum: &mut f64, correction: &mut f64, value: f64) {
    let combined = *sum + value;
    if sum.abs() >= value.abs() {
        *correction += (*sum - combined) + value;
    } else {
        *correction += (value - combined) + *sum;
    }
    *sum = combined;
}

/// Kahan–Babuška (Neumaier) compensated sum: the workspace's single owner of compensated scalar summation.
#[derive(Debug, Default, Clone, Copy)]
pub struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn add(&mut self, value: f64) {
        neumaier_step(&mut self.sum, &mut self.correction, value);
    }

    /// The compensated total: the running sum with its accumulated rounding error folded back in.
    #[inline]
    pub fn value(self) -> f64 {
        self.sum + self.correction
    }
}

/// Compensated accumulation of weighted moments, the total weight and the absolute weight sum.
pub struct QuadratureAccumulator {
    sums: Vec<f64>,
    corrections: Vec<f64>,
    mass: CompensatedSum,
    absolute_weight_sum: f64,
}

impl QuadratureAccumulator {
    /// An accumulator over the caller's zeroed buffers, one entry per moment, so the caller keeps its own
    /// allocation refusal.
    pub fn from_zeroed(sums: Vec<f64>, corrections: Vec<f64>) -> Self {
        Self {
            sums,
            corrections,
            mass: CompensatedSum::new(),
            absolute_weight_sum: 0.0,
        }
    }

    pub fn add_moment(&mut self, index: usize, value: f64) {
        neumaier_step(&mut self.sums[index], &mut self.corrections[index], value);
    }

    pub fn add_weight(&mut self, weight: f64) {
        self.mass.add(weight);
        self.absolute_weight_sum += weight.abs();
    }

    /// The compensated moment sums, the total weight and the absolute weight sum.
    pub fn finish(mut self) -> (Vec<f64>, f64, f64) {
        for (sum, correction) in self.sums.iter_mut().zip(self.corrections.iter()) {
            *sum += *correction;
        }
        (self.sums, self.mass.value(), self.absolute_weight_sum)
    }
}

#[cfg(test)]
mod tests {
    use super::{CompensatedSum, QuadratureAccumulator};

    #[test]
    fn compensated_sum_keeps_the_running_sum_under_a_larger_addend() {
        // Plain Kahan returns 0 here: adding 1e100 to 1 rounds the running sum's 1 away and its compensation
        // term never sees it. Neumaier takes the error from the smaller operand and recovers the exact 2.
        let mut sum = CompensatedSum::new();
        for value in [1.0, 1e100, 1.0, -1e100] {
            sum.add(value);
        }
        assert_eq!(sum.value(), 2.0);
    }

    #[test]
    fn quadrature_moments_use_the_same_compensation() {
        let mut accumulator = QuadratureAccumulator::from_zeroed(vec![0.0; 2], vec![0.0; 2]);
        let mut owner = CompensatedSum::new();
        for value in [1.0, 1e100, 1.0, -1e100, 0.1, 0.2] {
            accumulator.add_moment(1, value);
            owner.add(value);
            accumulator.add_weight(value);
        }
        let (moments, mass, _) = accumulator.finish();
        assert_eq!(moments[0], 0.0);
        assert_eq!(moments[1].to_bits(), owner.value().to_bits());
        assert_eq!(mass.to_bits(), owner.value().to_bits());
    }
}

use ndarray::LinalgScalar;
use num_traits::Float;
use rand::distr::Distribution as RandDistribution;
use rand_distr::uniform::SampleUniform;
// Bind to rand's Distribution to avoid trait mismatches from other deps pulling rand 0.8.
use rand::Rng;
use rand_distr::StandardNormal;

/// Abstraction over a mutable Euclidean vector that supports the in-place
/// operations required by the Hamiltonian integrators.
pub trait EuclideanVector: Clone {
    type Scalar: Float + LinalgScalar + SampleUniform + Copy;

    /// Returns the dimensionality of the vector.
    fn len(&self) -> usize;

    /// Returns true if the vector is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Creates a zero-initialized vector with the same shape.
    fn zeros_like(&self) -> Self;

    /// Resets the vector to all zeros in-place.
    fn fill_zero(&mut self);

    /// Copies the contents of `other` into `self` without reallocating.
    fn assign(&mut self, other: &Self);

    /// In-place addition.
    fn add_assign(&mut self, other: &Self);

    /// In-place subtraction: `self -= other`.
    fn sub_assign(&mut self, other: &Self);

    /// In-place fused multiply-add: `self += alpha * other`.
    fn add_scaled_assign(&mut self, other: &Self, alpha: Self::Scalar);

    /// Scales the vector in-place.
    fn scale_assign(&mut self, alpha: Self::Scalar);

    /// Multiplies each dimension by the corresponding diagonal element.
    fn scale_diag_assign(&mut self, diag: &[Self::Scalar]);

    /// In-place fused multiply-add with a diagonal metric: `self += alpha * diag .* other`.
    fn add_diag_scaled_assign(&mut self, other: &Self, diag: &[Self::Scalar], alpha: Self::Scalar);

    /// Dot product between two vectors.
    fn dot(&self, other: &Self) -> Self::Scalar;

    /// Diagonal quadratic form `sum_i diag[i] * self[i]^2`.
    fn quad_form_diag(&self, diag: &[Self::Scalar]) -> Self::Scalar;

    /// Fills the vector with samples from N(0, 1) in-place.
    fn fill_standard_normal(&mut self, rng: &mut impl Rng)
    where
        StandardNormal: RandDistribution<Self::Scalar>;

    /// Writes the vector contents into the provided slice.
    fn write_to_slice(&self, out: &mut [Self::Scalar]);

    /// Overwrites vector contents from the provided slice.
    fn read_from_slice(&mut self, input: &[Self::Scalar]);
}

impl<T> EuclideanVector for ndarray::Array1<T>
where
    T: Float + LinalgScalar + SampleUniform + Copy,
    StandardNormal: RandDistribution<T>,
{
    type Scalar = T;

    fn len(&self) -> usize {
        self.len()
    }

    fn zeros_like(&self) -> Self {
        ndarray::Array1::zeros(self.len())
    }

    fn fill_zero(&mut self) {
        self.fill(T::zero());
    }

    fn assign(&mut self, other: &Self) {
        self.clone_from(other);
    }

    fn add_assign(&mut self, other: &Self) {
        ndarray::Zip::from(self).and(other).for_each(|a, b| {
            *a = *a + *b;
        });
    }

    fn sub_assign(&mut self, other: &Self) {
        ndarray::Zip::from(self).and(other).for_each(|a, b| {
            *a = *a - *b;
        });
    }

    fn add_scaled_assign(&mut self, other: &Self, alpha: Self::Scalar) {
        ndarray::Zip::from(self).and(other).for_each(|a, b| {
            *a = *a + *b * alpha;
        });
    }

    fn scale_assign(&mut self, alpha: Self::Scalar) {
        self.mapv_inplace(|x| x * alpha);
    }

    fn scale_diag_assign(&mut self, diag: &[Self::Scalar]) {
        assert_eq!(
            diag.len(),
            self.len(),
            "scale_diag_assign dimension mismatch"
        );
        ndarray::Zip::from(self).and(diag).for_each(|a, scale| {
            *a = *a * *scale;
        });
    }

    fn add_diag_scaled_assign(&mut self, other: &Self, diag: &[Self::Scalar], alpha: Self::Scalar) {
        assert_eq!(
            diag.len(),
            self.len(),
            "add_diag_scaled_assign dimension mismatch"
        );
        ndarray::Zip::from(self)
            .and(other)
            .and(diag)
            .for_each(|a, b, scale| {
                *a = *a + *b * *scale * alpha;
            });
    }

    fn dot(&self, other: &Self) -> Self::Scalar {
        self.view().dot(&other.view())
    }

    fn quad_form_diag(&self, diag: &[Self::Scalar]) -> Self::Scalar {
        assert_eq!(diag.len(), self.len(), "quad_form_diag dimension mismatch");
        self.iter()
            .zip(diag.iter())
            .fold(T::zero(), |acc, (&x, &d)| acc + x * x * d)
    }

    fn fill_standard_normal(&mut self, rng: &mut impl Rng)
    where
        StandardNormal: RandDistribution<Self::Scalar>,
    {
        self.iter_mut()
            .for_each(|x| *x = rng.sample(StandardNormal));
    }

    fn write_to_slice(&self, out: &mut [Self::Scalar]) {
        assert_eq!(
            out.len(),
            self.len(),
            "write_to_slice called with mismatched buffer length"
        );
        let slice = self
            .as_slice()
            .expect("Array1 is expected to be contiguous when writing to slice");
        out.copy_from_slice(slice);
    }

    fn read_from_slice(&mut self, input: &[Self::Scalar]) {
        assert_eq!(
            input.len(),
            self.len(),
            "read_from_slice called with mismatched buffer length"
        );
        let slice = self
            .as_slice_mut()
            .expect("Array1 is expected to be contiguous when reading from slice");
        slice.copy_from_slice(input);
    }
}

/// Trait for batch-aware vector operations needed by HMC.
///
/// This trait extends `EuclideanVector` with operations that support batched execution,
/// where `Self` represents ALL chains (e.g., `Tensor<B, 2>` of shape `[n_chains, dim]`).
/// This enables GPU-parallel execution without serializing chains.
pub trait BatchVector: EuclideanVector {
    /// Per-chain energy type. For single chain: `f64`. For batch: `Tensor<B, 1>`.
    type Energy: Clone;

    /// Per-chain mask type. For single chain: `bool`. For batch: `Tensor<B, 1, Bool>`.
    type Mask;

    /// Number of chains in the batch.
    fn n_chains(&self) -> usize;

    /// Dimension per chain.
    fn dim_per_chain(&self) -> usize;

    /// Per-chain kinetic energy: 0.5 * sum(p^2) for each chain.
    /// Returns [n_chains] energies for batch, or single scalar for single chain.
    fn kinetic_energy(&self) -> Self::Energy;

    /// Per-chain kinetic energy under a diagonal inverse mass.
    fn kinetic_energy_diag(&self, inv_diag: &[Self::Scalar]) -> Self::Energy;

    /// Conditional update: self[i] = other[i] where mask[i] is true.
    /// For GPU: uses mask_where kernel. For CPU: simple if-else.
    fn masked_assign(&mut self, other: &Self, mask: &Self::Mask);

    /// Fill with N(0,1) random values using device-native RNG.
    /// For GPU: uses Tensor::random. For CPU: uses rng.
    fn fill_random_normal(&mut self, rng: &mut impl Rng)
    where
        StandardNormal: RandDistribution<Self::Scalar>;

    /// Generate uniform [0,1] values for acceptance test, same shape as Energy.
    fn sample_uniform(&self, rng: &mut impl Rng) -> Self::Energy
    where
        StandardNormal: RandDistribution<Self::Scalar>;

    // --- Energy arithmetic (avoids trait bound issues with Add/Sub on Tensor) ---

    /// Energy subtraction: a - b
    fn energy_sub(a: &Self::Energy, b: &Self::Energy) -> Self::Energy;

    /// Energy addition: a + b
    fn energy_add(a: &Self::Energy, b: &Self::Energy) -> Self::Energy;

    /// Energy negation: -a
    fn energy_neg(a: &Self::Energy) -> Self::Energy;

    /// Natural log of energy (for ln(u) in acceptance)
    fn energy_ln(a: &Self::Energy) -> Self::Energy;

    /// Mean Metropolis acceptance probability for the current proposal batch.
    fn mean_acceptance(log_accept: &Self::Energy) -> Self::Scalar;

    // --- Acceptance logic ---

    /// Create acceptance mask: returns true where log_accept >= ln_u
    fn accept_mask(log_accept: &Self::Energy, ln_u: &Self::Energy) -> Self::Mask;
}

/// Implementation for single-chain ndarray (run N instances in parallel via Rayon)
impl<T> BatchVector for ndarray::Array1<T>
where
    T: Float + LinalgScalar + SampleUniform + Copy,
    StandardNormal: RandDistribution<T>,
{
    type Energy = T;
    type Mask = bool;

    fn n_chains(&self) -> usize {
        1
    }

    fn dim_per_chain(&self) -> usize {
        self.len()
    }

    fn kinetic_energy(&self) -> T {
        self.dot(self) * T::from(0.5).unwrap()
    }

    fn kinetic_energy_diag(&self, inv_diag: &[Self::Scalar]) -> T {
        self.quad_form_diag(inv_diag) * T::from(0.5).unwrap()
    }

    fn masked_assign(&mut self, other: &Self, mask: &bool) {
        if *mask {
            self.assign(other);
        }
    }

    fn fill_random_normal(&mut self, rng: &mut impl Rng)
    where
        StandardNormal: RandDistribution<Self::Scalar>,
    {
        self.iter_mut()
            .for_each(|x| *x = rng.sample(StandardNormal));
    }

    fn sample_uniform(&self, rng: &mut impl Rng) -> T
    where
        StandardNormal: RandDistribution<Self::Scalar>,
    {
        // Use SampleUniform which is already bounded on T
        use rand::distr::Uniform;
        let dist = Uniform::new(T::zero(), T::one()).unwrap();
        rng.sample(dist)
    }

    fn energy_sub(a: &T, b: &T) -> T {
        *a - *b
    }

    fn energy_add(a: &T, b: &T) -> T {
        *a + *b
    }

    fn energy_neg(a: &T) -> T {
        T::zero() - *a
    }

    fn energy_ln(a: &T) -> T {
        a.ln()
    }

    fn mean_acceptance(log_accept: &T) -> T {
        if !log_accept.is_finite() {
            T::zero()
        } else if *log_accept < T::zero() {
            log_accept.exp()
        } else {
            T::one()
        }
    }

    fn accept_mask(log_accept: &T, ln_u: &T) -> bool {
        *log_accept >= *ln_u
    }
}

#[cfg(feature = "burn")]
mod burn_impl {
    use super::EuclideanVector;
    use burn::prelude::{Backend, Tensor};
    use burn::tensor::Element;
    use burn::tensor::ElementConversion;
    use num_traits::{Float, FromPrimitive};
    use rand::Rng;
    use rand::distr::Distribution as RandDistribution;
    use rand_distr::StandardNormal;
    use rand_distr::uniform::SampleUniform;

    fn expand_diag<T, B>(diag: &[T], n_rows: usize) -> Tensor<B, 2>
    where
        T: Float + Element + ElementConversion + SampleUniform + FromPrimitive + Copy,
        B: Backend<FloatElem = T>,
        StandardNormal: RandDistribution<T>,
    {
        let dim = diag.len();
        let base: Tensor<B, 2> = Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(diag.to_vec(), [dim]),
            &B::Device::default(),
        )
        .unsqueeze_dim(0);
        base.expand([n_rows, dim])
    }

    impl<T, B> EuclideanVector for Tensor<B, 1>
    where
        T: Float + Element + ElementConversion + SampleUniform + FromPrimitive + Copy,
        B: Backend<FloatElem = T>,
        StandardNormal: RandDistribution<T>,
    {
        type Scalar = T;

        fn len(&self) -> usize {
            self.dims()[0]
        }

        fn zeros_like(&self) -> Self {
            Tensor::<B, 1>::zeros_like(self)
        }

        fn fill_zero(&mut self) {
            let zeros = Tensor::<B, 1>::zeros_like(self);
            self.inplace(|_| zeros.clone());
        }

        fn assign(&mut self, other: &Self) {
            self.inplace(|_| other.clone());
        }

        fn add_assign(&mut self, other: &Self) {
            self.inplace(|x| x.add(other.clone()));
        }

        fn sub_assign(&mut self, other: &Self) {
            self.inplace(|x| x.sub(other.clone()));
        }

        fn add_scaled_assign(&mut self, other: &Self, alpha: Self::Scalar) {
            self.inplace(|x| x.add(other.clone().mul_scalar(alpha)));
        }

        fn scale_assign(&mut self, alpha: Self::Scalar) {
            self.inplace(|x| x.mul_scalar(alpha));
        }

        fn scale_diag_assign(&mut self, diag: &[Self::Scalar]) {
            assert_eq!(
                diag.len(),
                self.len(),
                "scale_diag_assign dimension mismatch"
            );
            let scale: Tensor<B, 1> = Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(diag.to_vec(), [self.len()]),
                &B::Device::default(),
            );
            self.inplace(|x| x.mul(scale));
        }

        fn add_diag_scaled_assign(
            &mut self,
            other: &Self,
            diag: &[Self::Scalar],
            alpha: Self::Scalar,
        ) {
            assert_eq!(
                diag.len(),
                self.len(),
                "add_diag_scaled_assign dimension mismatch"
            );
            let scale: Tensor<B, 1> = Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(diag.to_vec(), [self.len()]),
                &B::Device::default(),
            )
            .mul_scalar(alpha);
            self.inplace(|x| x.add(other.clone().mul(scale)));
        }

        fn dot(&self, other: &Self) -> Self::Scalar {
            self.clone().mul(other.clone()).sum().into_scalar()
        }

        fn quad_form_diag(&self, diag: &[Self::Scalar]) -> Self::Scalar {
            assert_eq!(diag.len(), self.len(), "quad_form_diag dimension mismatch");
            let scale: Tensor<B, 1> = Tensor::<B, 1>::from_data(
                burn::tensor::TensorData::new(diag.to_vec(), [self.len()]),
                &B::Device::default(),
            );
            self.clone()
                .mul(self.clone())
                .mul(scale)
                .sum()
                .into_scalar()
        }

        fn fill_standard_normal(&mut self, _rng: &mut impl Rng)
        where
            StandardNormal: RandDistribution<Self::Scalar>,
        {
            // Device-native RNG for GPU efficiency (ignores CPU rng parameter)
            let shape = burn::tensor::Shape::new([self.len()]);
            let noise = Tensor::<B, 1>::random(
                shape,
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &B::Device::default(),
            );
            self.inplace(|_| noise);
        }

        fn write_to_slice(&self, out: &mut [Self::Scalar]) {
            let data = self.to_data();
            let slice = data.as_slice().expect("Tensor data expected to be dense");
            assert_eq!(
                out.len(),
                slice.len(),
                "write_to_slice called with mismatched buffer length"
            );
            out.copy_from_slice(slice);
        }

        fn read_from_slice(&mut self, input: &[Self::Scalar]) {
            assert_eq!(
                input.len(),
                self.len(),
                "read_from_slice called with mismatched buffer length"
            );
            let td = burn::tensor::TensorData::new(input.to_vec(), [self.len()]);
            let updated = Tensor::<B, 1>::from_data(td, &B::Device::default());
            self.inplace(|_| updated);
        }
    }

    // Batched chains: Tensor<B, 2> of shape [n_chains, dim]
    impl<T, B> EuclideanVector for Tensor<B, 2>
    where
        T: Float + Element + ElementConversion + SampleUniform + FromPrimitive + Copy,
        B: Backend<FloatElem = T>,
        StandardNormal: RandDistribution<T>,
    {
        type Scalar = T;

        fn len(&self) -> usize {
            // Total elements across all chains
            self.dims()[0] * self.dims()[1]
        }

        fn zeros_like(&self) -> Self {
            Tensor::<B, 2>::zeros_like(self)
        }

        fn fill_zero(&mut self) {
            let zeros = Tensor::<B, 2>::zeros_like(self);
            self.inplace(|_| zeros.clone());
        }

        fn assign(&mut self, other: &Self) {
            self.inplace(|_| other.clone());
        }

        fn add_assign(&mut self, other: &Self) {
            self.inplace(|x| x.add(other.clone()));
        }

        fn sub_assign(&mut self, other: &Self) {
            self.inplace(|x| x.sub(other.clone()));
        }

        fn add_scaled_assign(&mut self, other: &Self, alpha: Self::Scalar) {
            self.inplace(|x| x.add(other.clone().mul_scalar(alpha)));
        }

        fn scale_assign(&mut self, alpha: Self::Scalar) {
            self.inplace(|x| x.mul_scalar(alpha));
        }

        fn scale_diag_assign(&mut self, diag: &[Self::Scalar]) {
            let dims = self.dims();
            let dim = dims[1];
            assert_eq!(diag.len(), dim, "scale_diag_assign dimension mismatch");
            let scale = expand_diag::<T, B>(diag, dims[0]);
            self.inplace(|x| x.mul(scale));
        }

        fn add_diag_scaled_assign(
            &mut self,
            other: &Self,
            diag: &[Self::Scalar],
            alpha: Self::Scalar,
        ) {
            let dims = self.dims();
            let dim = dims[1];
            assert_eq!(diag.len(), dim, "add_diag_scaled_assign dimension mismatch");
            let scale = expand_diag::<T, B>(diag, dims[0]).mul_scalar(alpha);
            self.inplace(|x| x.add(other.clone().mul(scale)));
        }

        fn dot(&self, other: &Self) -> Self::Scalar {
            // Global dot product (sum over all elements)
            self.clone().mul(other.clone()).sum().into_scalar()
        }

        fn quad_form_diag(&self, diag: &[Self::Scalar]) -> Self::Scalar {
            let dims = self.dims();
            let dim = dims[1];
            assert_eq!(diag.len(), dim, "quad_form_diag dimension mismatch");
            let scale = expand_diag::<T, B>(diag, dims[0]);
            self.clone()
                .mul(self.clone())
                .mul(scale)
                .sum()
                .into_scalar()
        }

        fn fill_standard_normal(&mut self, _rng: &mut impl Rng)
        where
            StandardNormal: RandDistribution<Self::Scalar>,
        {
            // Device-native RNG for GPU efficiency
            let shape = burn::tensor::Shape::new(self.dims());
            let noise = Tensor::<B, 2>::random(
                shape,
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &B::Device::default(),
            );
            self.inplace(|_| noise);
        }

        fn write_to_slice(&self, out: &mut [Self::Scalar]) {
            let data = self.to_data();
            let slice = data.as_slice().expect("Tensor data expected to be dense");
            assert_eq!(
                out.len(),
                slice.len(),
                "write_to_slice called with mismatched buffer length"
            );
            out.copy_from_slice(slice);
        }

        fn read_from_slice(&mut self, input: &[Self::Scalar]) {
            let dims = self.dims();
            let expected = dims[0] * dims[1];
            assert_eq!(
                input.len(),
                expected,
                "read_from_slice called with mismatched buffer length"
            );
            let td = burn::tensor::TensorData::new(input.to_vec(), dims);
            let updated = Tensor::<B, 2>::from_data(td, &B::Device::default());
            self.inplace(|_| updated);
        }
    }

    use super::BatchVector;

    /// BatchVector for Tensor<B, 2>: all chains batched, GPU-parallel execution
    impl<T, B> BatchVector for Tensor<B, 2>
    where
        T: Float + Element + ElementConversion + SampleUniform + FromPrimitive + Copy,
        B: Backend<FloatElem = T>,
        StandardNormal: RandDistribution<T>,
    {
        type Energy = Tensor<B, 1>; // [n_chains] energies
        type Mask = Tensor<B, 1, burn::tensor::Bool>; // [n_chains] booleans

        fn n_chains(&self) -> usize {
            self.dims()[0]
        }

        fn dim_per_chain(&self) -> usize {
            self.dims()[1]
        }

        fn kinetic_energy(&self) -> Tensor<B, 1> {
            // Per-chain: 0.5 * sum(p^2) over dim axis
            // [n_chains, dim] -> [n_chains]
            self.clone()
                .mul(self.clone())
                .sum_dim(1)
                .squeeze(1)
                .mul_scalar(T::from(0.5).unwrap())
        }

        fn kinetic_energy_diag(&self, inv_diag: &[Self::Scalar]) -> Tensor<B, 1> {
            assert_eq!(
                inv_diag.len(),
                self.dims()[1],
                "kinetic_energy_diag dimension mismatch"
            );
            let n_chains = self.dims()[0];
            let inv = expand_diag::<T, B>(inv_diag, n_chains);
            self.clone()
                .mul(self.clone())
                .mul(inv)
                .sum_dim(1)
                .squeeze(1)
                .mul_scalar(T::from(0.5).unwrap())
        }

        fn masked_assign(&mut self, other: &Self, mask: &Tensor<B, 1, burn::tensor::Bool>) {
            // Expand mask from [n_chains] to [n_chains, dim]
            let n_chains = self.dims()[0];
            let dim = self.dims()[1];
            // Explicitly specify output dimension for unsqueeze_dim
            let mask_2d: Tensor<B, 2, burn::tensor::Bool> = mask.clone().unsqueeze_dim(1);
            let mask_expanded = mask_2d.expand([n_chains, dim]);
            self.inplace(|x| x.clone().mask_where(mask_expanded, other.clone()));
        }

        fn fill_random_normal(&mut self, _rng: &mut impl Rng)
        where
            StandardNormal: RandDistribution<Self::Scalar>,
        {
            // Device-native RNG for GPU efficiency
            let shape = burn::tensor::Shape::new(self.dims());
            let noise = Tensor::<B, 2>::random(
                shape,
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &B::Device::default(),
            );
            self.inplace(|_| noise);
        }

        fn sample_uniform(&self, _rng: &mut impl Rng) -> Tensor<B, 1>
        where
            StandardNormal: RandDistribution<Self::Scalar>,
        {
            // Device-native uniform sampling [n_chains]
            let n_chains = self.dims()[0];
            Tensor::<B, 1>::random(
                burn::tensor::Shape::new([n_chains]),
                burn::tensor::Distribution::Uniform(0.0, 1.0),
                &B::Device::default(),
            )
        }

        fn energy_sub(a: &Tensor<B, 1>, b: &Tensor<B, 1>) -> Tensor<B, 1> {
            a.clone().sub(b.clone())
        }

        fn energy_add(a: &Tensor<B, 1>, b: &Tensor<B, 1>) -> Tensor<B, 1> {
            a.clone().add(b.clone())
        }

        fn energy_neg(a: &Tensor<B, 1>) -> Tensor<B, 1> {
            a.clone().neg()
        }

        fn energy_ln(a: &Tensor<B, 1>) -> Tensor<B, 1> {
            a.clone().log()
        }

        fn mean_acceptance(log_accept: &Tensor<B, 1>) -> T {
            let data = log_accept.to_data();
            let values = data
                .as_slice::<T>()
                .expect("Expected batched log acceptance tensor to be dense");
            let total = values.iter().copied().fold(T::zero(), |sum, value| {
                if !value.is_finite() {
                    sum
                } else if value < T::zero() {
                    sum + value.exp()
                } else {
                    sum + T::one()
                }
            });
            total / T::from_usize(values.len()).unwrap()
        }

        fn accept_mask(
            log_accept: &Tensor<B, 1>,
            ln_u: &Tensor<B, 1>,
        ) -> Tensor<B, 1, burn::tensor::Bool> {
            // Returns true where log_accept >= ln_u
            log_accept.clone().greater_equal(ln_u.clone())
        }
    }
}

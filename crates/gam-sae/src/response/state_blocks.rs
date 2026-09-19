//! State blocks: the finest common orthogonal block decomposition of a family
//! of linear maps acting on one vector state (#2946, the proposal's §9.2).
//!
//! # What is computed
//!
//! The maps `A₁..A_m` on `ℝᵈ` (executed transition Jacobians, fitted linear
//! transports) generate a unital `*`-algebra `𝒜`, the span of all words in the
//! `A_a` and `A_aᵀ`. A subspace that every `A_a` and every `A_aᵀ` maps into
//! itself is invariant under `𝒜`, and so is its orthogonal complement, so `ℝᵈ`
//! splits orthogonally into irreducible blocks. Over ℝ the structure theorem
//! reads
//!
//! ```text
//! ℝᵈ = ⊕ᵢ Wᵢ,      𝒜|Wᵢ ≅ M_{kᵢ}(Dᵢ) ⊗ I_{mᵢ},      𝒜' ≅ ⊕ᵢ M_{mᵢ}(Dᵢ),
//! ```
//!
//! with `Dᵢ ∈ {ℝ, ℂ, ℍ}` and commutant `𝒜' = {X : XA_a = A_aX, XA_aᵀ = A_aᵀX}`.
//! The isotypic component `Wᵢ` holds `mᵢ` equivalent irreducible blocks, each of
//! real dimension `kᵢ·dim Dᵢ`.
//!
//! The components are canonical. The blocks inside a component with `mᵢ > 1`
//! are not: the commutant's orthogonal group mixes them, and the orthogonal
//! decompositions of `Wᵢ` into `mᵢ` blocks form a manifold of dimension
//! `dim Dᵢ · mᵢ(mᵢ − 1)/2`. Each component is reported with its block dimension,
//! multiplicity, division algebra and that ambiguity dimension, and no blocks
//! are chosen inside it.
//!
//! # Construction
//!
//! The textbook route reads the blocks off the eigenspaces of a generic
//! symmetric commutant element. Producing one means solving a linear system in
//! `d²` unknowns, and "generic" is only a probability statement. This module
//! forms neither the commutant nor a random element. Instead it refines an
//! orthogonal partition of `ℝᵈ` by spectral projections of symmetric elements of
//! `𝒜` itself, each an eigenproblem no larger than `d × d`:
//!
//! 1. The initial clusters are the eigenspaces of `Ω = Σ_a A_aA_aᵀ + A_aᵀA_a`.
//! 2. A cluster `U_c` is split by the eigenspaces of any non-scalar operator
//!    among `sym(A_cc)`, `A_cc'A_cc'ᵀ` and `A_c'cᵀA_c'c`, where `A_cc'` is the
//!    block of `A_a` from `U_c'` into `U_c`.
//! 3. Once all of those are scalar, every nonzero block is a scaled orthogonal
//!    map between clusters of equal dimension. Clusters joined by nonzero blocks
//!    form connected components. Transporting a component's blocks to its root
//!    cluster along a spanning tree gives orthogonal maps `G` on the root, and
//!    these generate the corner algebra `E_r 𝒜 E_r`. The root is split by any
//!    non-scalar `sym(G)`, `sym(JJ')` or `sym(JJ'J'')`, where the `J` are the
//!    antisymmetric parts of the `G`.
//! 4. When no split remains, every symmetric element of the corner is scalar,
//!    so the corner is a real division algebra (Frobenius): `ℝ` with no complex
//!    structure, `ℂ` with one, `ℍ` with two or three.
//!
//! Every split projection is a polynomial in an element of `𝒜`, so it lies in
//! `𝒜`. A cluster therefore contains every copy of each irreducible it meets:
//! equivalent blocks are never separated, and clusters that no block joins
//! share no isotypic summand. A nonzero block is invertible between its two
//! clusters and commutes with the central projections, so in a stable connected
//! component every cluster meets the same summands. The corner `⊕ᵢ M_{kᵢ}(Dᵢ)`
//! is a division algebra only for one summand with `kᵢ = 1`. Each stable
//! component is therefore exactly one isotypic component, with
//! `mᵢ = n_r / dim Dᵢ` and block dimension `|K|·dim Dᵢ`. The cluster count grows
//! at every split and cannot exceed `d`, so the refinement terminates without
//! an iteration cap.
//!
//! # Scale
//!
//! The work is one `d × d` eigendecomposition of `Ω`, one orthogonal transform
//! of each map, and eigenproblems on clusters. The commutant is never
//! materialised and no `d² × d²` matrix is formed. A scalar operator is
//! certified by its Frobenius spread without an eigendecomposition, so the
//! identity family at any width costs `Ω`'s decomposition and the transforms.
//!
//! # Decisions
//!
//! Every "equal" or "zero" verdict is a rank decision on a computed symmetric
//! spectrum, read by [`gam_linalg::roundoff::resolved_eigenvalue_count`]. The
//! assembly band handed to it is derived from the arithmetic that produced the
//! operator: Wilkinson growth for products and transforms, and a Davis–Kahan
//! bound for the cluster subspaces the blocks are read through. No threshold is
//! chosen by hand.

use std::collections::VecDeque;
use std::fmt;
use std::ops::Range;

use faer::Side;
use gam_linalg::faer_ndarray::{
    FaerLinalgError, fast_ab, fast_abt, fast_atb, strict_symmetric_eigh,
};
use gam_linalg::roundoff::{
    accumulation_growth, resolved_eigenvalue_count, symmetric_spectrum_rounding_band,
};
use ndarray::{Array2, ArrayView2, s};

/// The real division algebra of intertwiners of one irreducible block with
/// itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DivisionAlgebra {
    Real,
    Complex,
    Quaternion,
}

impl DivisionAlgebra {
    /// `dim_ℝ D`: 1, 2 or 4.
    pub fn real_dimension(self) -> usize {
        match self {
            Self::Real => 1,
            Self::Complex => 2,
            Self::Quaternion => 4,
        }
    }

    /// The algebra generated by `count` independent, pairwise anticommuting
    /// complex structures whose triple products are scalar. Four or more always
    /// have a non-scalar symmetric triple product, so they have no division
    /// algebra.
    fn generated_by(count: usize) -> Option<Self> {
        match count {
            0 => Some(Self::Real),
            1 => Some(Self::Complex),
            2 | 3 => Some(Self::Quaternion),
            _ => None,
        }
    }
}

/// Whether an isotypic component's span determines its irreducible blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockBasis {
    /// One irreducible block: the component's span is the block.
    Determined,
    /// `multiplicity` equivalent blocks with no preferred decomposition. The
    /// orthogonal decompositions of the span into that many blocks form a
    /// manifold of dimension `ambiguity_dimension = dim D · m(m − 1)/2`.
    Undetermined {
        multiplicity: usize,
        ambiguity_dimension: usize,
    },
}

/// One isotypic component: `multiplicity` equivalent irreducible blocks.
#[derive(Clone, Debug)]
pub struct IsotypicComponent {
    /// Orthonormal columns spanning the component (`d × dimension`). Only the
    /// span is meaningful; the columns are not blocks.
    pub basis: Array2<f64>,
    /// Real dimension of one irreducible block.
    pub block_dimension: usize,
    /// Number of equivalent irreducible blocks in the component.
    pub multiplicity: usize,
    /// Intertwiners of one block with itself.
    pub division_algebra: DivisionAlgebra,
    /// Bound on the sine of the largest principal angle between the span of
    /// `basis` and the exact component of the supplied maps.
    pub angle_bound: f64,
}

impl IsotypicComponent {
    pub fn dimension(&self) -> usize {
        self.basis.ncols()
    }

    pub fn block_basis(&self) -> BlockBasis {
        if self.multiplicity == 1 {
            BlockBasis::Determined
        } else {
            BlockBasis::Undetermined {
                multiplicity: self.multiplicity,
                ambiguity_dimension: self.division_algebra.real_dimension()
                    * self.multiplicity
                    * (self.multiplicity - 1)
                    / 2,
            }
        }
    }

    /// Real dimension of the commutant restricted to the component,
    /// `dim D · m²`.
    pub fn commutant_dimension(&self) -> usize {
        self.division_algebra.real_dimension() * self.multiplicity * self.multiplicity
    }
}

/// The isotypic decomposition of the state space under the `*`-algebra a family
/// of maps generates.
#[derive(Clone, Debug)]
pub struct StateBlocks {
    /// Components in order of their first cluster, which follows the ascending
    /// spectrum of `Ω`.
    pub components: Vec<IsotypicComponent>,
}

impl StateBlocks {
    /// Decompose the state space of `maps`, each `d × d`. The maps are consumed
    /// and transformed in place, so no second copy of the family is held.
    pub fn decompose(maps: Vec<Array2<f64>>) -> Result<Self, StateBlockError> {
        let mut refinement = Refinement::new(maps)?;
        'refine: loop {
            if let Some((cluster, split)) = refinement.block_split()? {
                refinement.apply_split(cluster, split);
                continue;
            }
            let components = refinement.connected_components();
            let mut algebras = Vec::with_capacity(components.len());
            for component in &components {
                match refinement.corner(component)? {
                    Corner::Split(split) => {
                        refinement.apply_split(component[0], split);
                        continue 'refine;
                    }
                    Corner::Division(algebra) => algebras.push(algebra),
                }
            }
            return refinement.report(&components, &algebras);
        }
    }

    /// Real dimension of the commutant `{X : XA_a = A_aX, XA_aᵀ = A_aᵀX}`.
    pub fn commutant_dimension(&self) -> usize {
        self.components
            .iter()
            .map(IsotypicComponent::commutant_dimension)
            .sum()
    }
}

#[derive(Debug)]
pub enum StateBlockError {
    EmptyFamily,
    ZeroDimension,
    NotSquare {
        map: usize,
        rows: usize,
        cols: usize,
    },
    DimensionMismatch {
        map: usize,
        expected: usize,
        found: usize,
    },
    NonFinite {
        map: usize,
    },
    Eigendecomposition {
        context: &'static str,
        source: FaerLinalgError,
    },
    /// A stable partition contradicts the structure theorem, so one band
    /// decision was inconsistent with another.
    InconsistentStructure {
        context: &'static str,
    },
}

impl fmt::Display for StateBlockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFamily => write!(formatter, "state blocks: the family has no maps"),
            Self::ZeroDimension => write!(formatter, "state blocks: the state has dimension zero"),
            Self::NotSquare { map, rows, cols } => {
                write!(formatter, "state blocks: map {map} is {rows}x{cols}, not square")
            }
            Self::DimensionMismatch {
                map,
                expected,
                found,
            } => write!(
                formatter,
                "state blocks: map {map} acts on dimension {found}, the family on {expected}"
            ),
            Self::NonFinite { map } => {
                write!(formatter, "state blocks: map {map} has a non-finite entry")
            }
            Self::Eigendecomposition { context, source } => write!(
                formatter,
                "state blocks: eigendecomposition of {context} failed: {source}"
            ),
            Self::InconsistentStructure { context } => {
                write!(formatter, "state blocks: inconsistent stable structure: {context}")
            }
        }
    }
}

impl std::error::Error for StateBlockError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Eigendecomposition { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// A group of basis columns the refinement treats as one subspace.
struct Cluster {
    range: Range<usize>,
    /// Bound on the sine of the largest principal angle between the computed
    /// span and the exact spectral subspace it estimates.
    angle: f64,
    /// Accumulated Wilkinson factor of the orthogonal transforms applied to this
    /// cluster's rows and columns: a block touching the cluster carries
    /// arithmetic error at most `arithmetic · ‖A‖_F` from this side.
    arithmetic: f64,
}

struct Refinement {
    /// `d × d` orthonormal basis, columns grouped by cluster.
    basis: Array2<f64>,
    /// Each map in basis coordinates, `basisᵀ A basis`.
    maps: Vec<Array2<f64>>,
    /// `‖A_a‖_F`, invariant under the orthogonal transforms.
    norms: Vec<f64>,
    clusters: Vec<Cluster>,
}

enum Corner {
    /// The root cluster splits by this operator's eigenspaces.
    Split(SpectralSplit),
    /// The corner at the root is this division algebra.
    Division(DivisionAlgebra),
}

/// Eigenvectors of a symmetric operator in ascending eigenvalue order, grouped
/// into runs that no resolved gap separates, each with its Davis–Kahan angle.
struct SpectralSplit {
    vectors: Array2<f64>,
    parts: Vec<(Range<usize>, f64)>,
}

/// The antisymmetric part `J` of a transported block: `scale` times a complex
/// structure, within `error` of the exact one in spectral norm.
struct Structure {
    matrix: Array2<f64>,
    scale: f64,
    error: f64,
}

impl Refinement {
    fn new(maps: Vec<Array2<f64>>) -> Result<Self, StateBlockError> {
        let dimension = maps.first().ok_or(StateBlockError::EmptyFamily)?.nrows();
        if dimension == 0 {
            return Err(StateBlockError::ZeroDimension);
        }
        for (map, matrix) in maps.iter().enumerate() {
            let (rows, cols) = matrix.dim();
            if rows != cols {
                return Err(StateBlockError::NotSquare { map, rows, cols });
            }
            if rows != dimension {
                return Err(StateBlockError::DimensionMismatch {
                    map,
                    expected: dimension,
                    found: rows,
                });
            }
            if matrix.iter().any(|value| !value.is_finite()) {
                return Err(StateBlockError::NonFinite { map });
            }
        }
        let norms: Vec<f64> = maps.iter().map(|matrix| frobenius(&matrix.view())).collect();
        // Each entry of `AAᵀ` is a length-`d` inner product in error by at most
        // `γ_d (|A||A|ᵀ)_ij`, whose spectral norm is at most `‖A‖_F²`. The `2m`
        // pre-formed terms add `2m − 1` more roundings.
        let mut omega = Array2::<f64>::zeros((dimension, dimension));
        for matrix in &maps {
            omega += &fast_abt(matrix, matrix);
            omega += &fast_atb(matrix, matrix);
        }
        let omega = mirror_lower(omega);
        let omega_assembly = accumulation_growth(dimension + 2 * maps.len())
            * 2.0
            * norms.iter().map(|norm| norm * norm).sum::<f64>();
        let partition = spectral_partition(&omega, omega_assembly, "the family's Gram Ω")?;
        let basis = partition.vectors;
        // `Uᵀ(AU)`: two length-`d` accumulations per entry against `|U|`, whose
        // spectral norm is at most `‖U‖_F = √d`, keep the error within
        // `γ_{2d}·d·‖A‖_F`.
        let arithmetic = accumulation_growth(2 * dimension) * dimension as f64;
        let maps = maps
            .into_iter()
            .map(|matrix| fast_atb(&basis, &fast_ab(&matrix, &basis)))
            .collect();
        let clusters = partition
            .parts
            .into_iter()
            .map(|(range, angle)| Cluster {
                range,
                angle,
                arithmetic,
            })
            .collect();
        Ok(Self {
            basis,
            maps,
            norms,
            clusters,
        })
    }

    fn block(&self, map: usize, target: usize, source: usize) -> ArrayView2<'_, f64> {
        self.maps[map].slice(s![
            self.clusters[target].range.clone(),
            self.clusters[source].range.clone()
        ])
    }

    /// Spectral-norm bound on the error of a computed block: the cluster
    /// subspaces it is read through, plus the transforms' arithmetic.
    fn block_error(&self, map: usize, target: usize, source: usize) -> f64 {
        let target = &self.clusters[target];
        let source = &self.clusters[source];
        self.norms[map] * (target.angle + target.arithmetic + source.angle + source.arithmetic)
    }

    /// The first cluster that a block operator shows to be non-scalar, together
    /// with its split.
    fn block_split(&self) -> Result<Option<(usize, SpectralSplit)>, StateBlockError> {
        for (target, cluster) in self.clusters.iter().enumerate() {
            if cluster.range.len() < 2 {
                continue;
            }
            for map in 0..self.maps.len() {
                let own = self.block(map, target, target);
                let assembly =
                    self.block_error(map, target, target) + accumulation_growth(2) * frobenius(&own);
                if let Some(split) = non_scalar_split(
                    &symmetric_part(&own),
                    assembly,
                    "a cluster's symmetric self-block",
                )? {
                    return Ok(Some((target, split)));
                }
                for source in 0..self.clusters.len() {
                    let error = self.block_error(map, target, source);
                    let outgoing = self.block(map, target, source);
                    if let Some(split) = non_scalar_split(
                        &mirror_lower(fast_abt(&outgoing, &outgoing)),
                        gram_band(&outgoing, error, outgoing.ncols()),
                        "a block's range Gram",
                    )? {
                        return Ok(Some((target, split)));
                    }
                    let incoming = self.block(map, source, target);
                    if let Some(split) = non_scalar_split(
                        &mirror_lower(fast_atb(&incoming, &incoming)),
                        gram_band(&incoming, error, incoming.nrows()),
                        "a block's domain Gram",
                    )? {
                        return Ok(Some((target, split)));
                    }
                }
            }
        }
        Ok(None)
    }

    /// Rotate a cluster's columns into the split's eigenvectors and replace the
    /// cluster by the split's parts.
    fn apply_split(&mut self, index: usize, split: SpectralSplit) {
        let range = self.clusters[index].range.clone();
        let angle = self.clusters[index].angle;
        let size = range.len();
        let arithmetic = self.clusters[index].arithmetic
            + accumulation_growth(2 * size) * size as f64;
        let rotated = fast_ab(&self.basis.slice(s![.., range.clone()]), &split.vectors);
        self.basis.slice_mut(s![.., range.clone()]).assign(&rotated);
        for map in &mut self.maps {
            let rows = fast_atb(&split.vectors, &map.slice(s![range.clone(), ..]));
            map.slice_mut(s![range.clone(), ..]).assign(&rows);
            let cols = fast_ab(&map.slice(s![.., range.clone()]), &split.vectors);
            map.slice_mut(s![.., range.clone()]).assign(&cols);
        }
        let parts = split.parts.into_iter().map(|(part, part_angle)| Cluster {
            range: (range.start + part.start)..(range.start + part.end),
            angle: (angle + part_angle).min(1.0),
            arithmetic,
        });
        let mut clusters = std::mem::take(&mut self.clusters);
        let tail = clusters.split_off(index + 1);
        clusters.truncate(index);
        clusters.extend(parts);
        clusters.extend(tail);
        self.clusters = clusters;
    }

    /// The scale `s` of a block certified nonzero. Once the partition is block
    /// stable the block is `s` times an orthogonal map, `XXᵀ = s²I`, and the
    /// predicate resolves `s²` from zero.
    fn block_scale(&self, map: usize, target: usize, source: usize) -> Option<f64> {
        let block = self.block(map, target, source);
        let squared = frobenius(&block).powi(2) / block.nrows() as f64;
        let assembly = gram_band(&block, self.block_error(map, target, source), block.ncols());
        (resolved_eigenvalue_count(&[squared], assembly) == 1).then(|| squared.sqrt())
    }

    fn joined(&self, first: usize, second: usize) -> bool {
        (0..self.maps.len()).any(|map| {
            self.block_scale(map, first, second).is_some()
                || self.block_scale(map, second, first).is_some()
        })
    }

    /// Clusters joined by certified nonzero blocks. Each component lists its
    /// clusters in ascending order, and components are ordered by their first
    /// cluster.
    fn connected_components(&self) -> Vec<Vec<usize>> {
        let count = self.clusters.len();
        let mut representative: Vec<usize> = (0..count).collect();
        for first in 0..count {
            for second in (first + 1)..count {
                if self.joined(first, second) {
                    let left = find_representative(&mut representative, first);
                    let right = find_representative(&mut representative, second);
                    representative[left.max(right)] = left.min(right);
                }
            }
        }
        let mut components: Vec<Vec<usize>> = Vec::new();
        let mut position: Vec<Option<usize>> = vec![None; count];
        for cluster in 0..count {
            let root = find_representative(&mut representative, cluster);
            match position[root] {
                Some(index) => components[index].push(cluster),
                None => {
                    position[root] = Some(components.len());
                    components.push(vec![cluster]);
                }
            }
        }
        components
    }

    /// The strongest certified isometry from `source` into `target` across all
    /// maps and both orientations, normalised by its scale, with its error.
    fn strongest_isometry(&self, source: usize, target: usize) -> Option<(Array2<f64>, f64)> {
        let mut strongest: Option<(f64, Array2<f64>, f64)> = None;
        for map in 0..self.maps.len() {
            let error = self.block_error(map, target, source);
            // `A_{target,source}` maps the source cluster into the target one,
            // and so does the transposed map's block `A_{source,target}ᵀ`.
            let forward = self.block_scale(map, target, source).map(|scale| {
                (scale, self.block(map, target, source).mapv(|value| value / scale))
            });
            let backward = self.block_scale(map, source, target).map(|scale| {
                (scale, self.block(map, source, target).t().mapv(|value| value / scale))
            });
            for (scale, isometry) in forward.into_iter().chain(backward) {
                let stronger = strongest.as_ref().map_or(true, |best| scale > best.0);
                if stronger {
                    strongest = Some((scale, isometry, error / scale));
                }
            }
        }
        strongest.map(|best| (best.1, best.2))
    }

    /// Orthogonal transports from a component's root cluster to each of its
    /// clusters along a breadth-first spanning tree of certified blocks, each
    /// with its accumulated spectral error.
    fn transports(
        &self,
        component: &[usize],
        size: usize,
    ) -> Result<Vec<Option<(Array2<f64>, f64)>>, StateBlockError> {
        let mut transports: Vec<Option<(Array2<f64>, f64)>> = vec![None; self.clusters.len()];
        transports[component[0]] = Some((Array2::eye(size), 0.0));
        let mut queue = VecDeque::from([component[0]]);
        while let Some(current) = queue.pop_front() {
            for &next in component {
                if transports[next].is_some() {
                    continue;
                }
                let Some((isometry, error)) = self.strongest_isometry(current, next) else {
                    continue;
                };
                if isometry.dim() != (size, size) {
                    return Err(StateBlockError::InconsistentStructure {
                        context: "a certified nonzero block joins clusters of unequal dimension",
                    });
                }
                let (parent, parent_error) = transports[current].as_ref().ok_or(
                    StateBlockError::InconsistentStructure {
                        context: "a transport was read before it was built",
                    },
                )?;
                let transport = fast_ab(&isometry, parent);
                let transport_error =
                    parent_error + error + product_band(&[(1.0, 0.0), (1.0, 0.0)], size);
                transports[next] = Some((transport, transport_error));
                queue.push_back(next);
            }
        }
        if component
            .iter()
            .any(|&cluster| transports[cluster].is_none())
        {
            return Err(StateBlockError::InconsistentStructure {
                context: "a connected component's spanning tree missed a cluster",
            });
        }
        Ok(transports)
    }

    /// Classify the corner algebra at a component's root, or split the root by
    /// a non-scalar symmetric element of that corner.
    fn corner(&self, component: &[usize]) -> Result<Corner, StateBlockError> {
        let root = component[0];
        let size = self.clusters[root].range.len();
        if size == 1 {
            return Ok(Corner::Division(DivisionAlgebra::Real));
        }
        let transports = self.transports(component, size)?;
        let mut structures: Vec<Structure> = Vec::new();
        for map in 0..self.maps.len() {
            for &target in component {
                for &source in component {
                    let Some(scale) = self.block_scale(map, target, source) else {
                        continue;
                    };
                    let (Some((target_transport, target_error)), Some((source_transport, source_error))) =
                        (transports[target].as_ref(), transports[source].as_ref())
                    else {
                        return Err(StateBlockError::InconsistentStructure {
                            context: "a component cluster has no transport",
                        });
                    };
                    let generator = fast_atb(
                        target_transport,
                        &fast_ab(&self.block(map, target, source), source_transport),
                    )
                    .mapv(|value| value / scale);
                    let error = self.block_error(map, target, source) / scale
                        + target_error
                        + source_error
                        + product_band(&[(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)], size);
                    let assembly =
                        error + accumulation_growth(2) * frobenius(&generator.view());
                    if let Some(split) = non_scalar_split(
                        &symmetric_part(&generator.view()),
                        assembly,
                        "a transported block's symmetric part",
                    )? {
                        return Ok(Corner::Split(split));
                    }
                    if structures.len() < 4 {
                        admit_structure(
                            &mut structures,
                            antisymmetric_part(&generator.view()),
                            error,
                            size,
                        )?;
                    }
                }
            }
        }
        for first in 0..structures.len() {
            for second in (first + 1)..structures.len() {
                let product = fast_ab(&structures[first].matrix, &structures[second].matrix);
                let assembly = product_band(
                    &[structures[first].factor(), structures[second].factor()],
                    size,
                );
                if let Some(split) = non_scalar_split(
                    &symmetric_part(&product.view()),
                    assembly,
                    "the symmetric part of two complex structures' product",
                )? {
                    return Ok(Corner::Split(split));
                }
            }
        }
        for first in 0..structures.len() {
            for second in (first + 1)..structures.len() {
                for third in (second + 1)..structures.len() {
                    let product = fast_ab(
                        &fast_ab(&structures[first].matrix, &structures[second].matrix),
                        &structures[third].matrix,
                    );
                    let assembly = product_band(
                        &[
                            structures[first].factor(),
                            structures[second].factor(),
                            structures[third].factor(),
                        ],
                        size,
                    );
                    if let Some(split) = non_scalar_split(
                        &symmetric_part(&product.view()),
                        assembly,
                        "the symmetric part of three complex structures' product",
                    )? {
                        return Ok(Corner::Split(split));
                    }
                }
            }
        }
        let algebra = DivisionAlgebra::generated_by(structures.len()).ok_or(
            StateBlockError::InconsistentStructure {
                context: "four independent complex structures passed every product test",
            },
        )?;
        if size % algebra.real_dimension() != 0 {
            return Err(StateBlockError::InconsistentStructure {
                context: "the corner's division algebra does not divide the root cluster's dimension",
            });
        }
        Ok(Corner::Division(algebra))
    }

    fn report(
        &self,
        components: &[Vec<usize>],
        algebras: &[DivisionAlgebra],
    ) -> Result<StateBlocks, StateBlockError> {
        let dimension = self.basis.nrows();
        let mut reported = Vec::with_capacity(components.len());
        for (component, &algebra) in components.iter().zip(algebras) {
            let size = self.clusters[component[0]].range.len();
            let span: usize = component
                .iter()
                .map(|&cluster| self.clusters[cluster].range.len())
                .sum();
            if span != size * component.len() {
                return Err(StateBlockError::InconsistentStructure {
                    context: "the clusters of one component differ in dimension",
                });
            }
            let mut basis = Array2::<f64>::zeros((dimension, span));
            let mut column = 0;
            // `‖P − P*‖₂ ≤ Σ_c ‖P_c − P_c*‖₂`, each term the sine of its cluster's
            // largest principal angle; the transforms' drift from orthonormality
            // adds their arithmetic factor.
            let mut angle_bound = 0.0;
            for &cluster in component {
                let range = self.clusters[cluster].range.clone();
                basis
                    .slice_mut(s![.., column..column + range.len()])
                    .assign(&self.basis.slice(s![.., range.clone()]));
                column += range.len();
                angle_bound += self.clusters[cluster].angle + self.clusters[cluster].arithmetic;
            }
            reported.push(IsotypicComponent {
                basis,
                block_dimension: algebra.real_dimension() * component.len(),
                multiplicity: size / algebra.real_dimension(),
                division_algebra: algebra,
                angle_bound: f64::min(angle_bound, 1.0),
            });
        }
        Ok(StateBlocks {
            components: reported,
        })
    }
}

impl Structure {
    fn factor(&self) -> (f64, f64) {
        (self.scale, self.error)
    }
}

fn find_representative(representative: &mut [usize], mut node: usize) -> usize {
    while representative[node] != node {
        representative[node] = representative[representative[node]];
        node = representative[node];
    }
    node
}

/// Admit `structure` when the predicate resolves it from zero and from the span
/// of the structures already admitted, reading the spectrum of their Gram.
fn admit_structure(
    structures: &mut Vec<Structure>,
    structure: Array2<f64>,
    error: f64,
    size: usize,
) -> Result<(), StateBlockError> {
    // `JᵀJ = s²I` for a scaled complex structure, so `s²` is its spectrum, in
    // error by at most the product band of `J` with itself.
    let scale = frobenius(&structure.view()) / (size as f64).sqrt();
    let assembly = product_band(&[(scale, error), (scale, error)], size);
    if resolved_eigenvalue_count(&[scale * scale], assembly) == 0 {
        return Ok(());
    }
    let candidate = Structure {
        matrix: structure,
        scale,
        error,
    };
    let count = structures.len() + 1;
    let member = |index: usize| structures.get(index).unwrap_or(&candidate);
    let gram = Array2::from_shape_fn((count, count), |(row, col)| {
        member(row)
            .matrix
            .iter()
            .zip(member(col).matrix.iter())
            .map(|(left, right)| left * right)
            .sum::<f64>()
            / size as f64
    });
    // `|⟨X, Y⟩_F / n| ≤ ‖X‖₂‖Y‖₂` for `n × n` factors, so each entry is within
    // the pair's product band of the exact one, and the spectral error of the
    // Gram is at most `count` times the largest entry error.
    let mut entry_band = 0.0_f64;
    for row in 0..count {
        for col in 0..count {
            entry_band =
                entry_band.max(product_band(&[member(row).factor(), member(col).factor()], size));
        }
    }
    let values = strict_symmetric_eigh(&gram, Side::Lower)
        .map_err(|source| StateBlockError::Eigendecomposition {
            context: "the complex structures' Gram",
            source,
        })?
        .0
        .to_vec();
    if resolved_eigenvalue_count(&values, count as f64 * entry_band) == count {
        structures.push(candidate);
    }
    Ok(())
}

/// Eigen-clusters of a symmetric operator whose computed entries lie within
/// `assembly` of the exact operator in spectral norm. There is always at least
/// one part.
fn spectral_partition(
    operator: &Array2<f64>,
    assembly: f64,
    context: &'static str,
) -> Result<SpectralSplit, StateBlockError> {
    let (values, vectors) = strict_symmetric_eigh(operator, Side::Lower)
        .map_err(|source| StateBlockError::Eigendecomposition { context, source })?;
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    let sorted: Vec<f64> = order.iter().map(|&index| values[index]).collect();
    let mut ordered = Array2::<f64>::zeros(vectors.dim());
    for (column, &index) in order.iter().enumerate() {
        ordered.column_mut(column).assign(&vectors.column(index));
    }
    // Each computed eigenvalue is within `band` of an exact one (Weyl), so a
    // difference of two is within `2·band` of the exact difference.
    let band = symmetric_spectrum_rounding_band(&sorted) + assembly;
    let gaps: Vec<f64> = sorted.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let resolved = resolved_flags(&gaps, 2.0 * band);
    let mut boundaries = vec![0];
    for (gap, &flag) in resolved.iter().enumerate() {
        if flag {
            boundaries.push(gap + 1);
        }
    }
    boundaries.push(sorted.len());
    let parts = boundaries
        .windows(2)
        .map(|pair| {
            let (start, end) = (pair[0], pair[1]);
            let below = if start > 0 {
                sorted[start] - sorted[start - 1]
            } else {
                f64::INFINITY
            };
            let above = if end < sorted.len() {
                sorted[end] - sorted[end - 1]
            } else {
                f64::INFINITY
            };
            let separation = below.min(above);
            // Davis–Kahan: the computed part's eigenvalues lie at least
            // `separation − band` from every other exact eigenvalue, and the
            // operator is within `band` of the exact one. A part with no
            // neighbour spans the whole domain exactly.
            let angle = if separation.is_finite() {
                (band / (separation - band)).min(1.0)
            } else {
                0.0
            };
            (start..end, angle)
        })
        .collect();
    Ok(SpectralSplit {
        vectors: ordered,
        parts,
    })
}

/// A split of `operator` by its eigenspaces, or `None` when it is scalar at
/// the resolution `assembly` allows.
fn non_scalar_split(
    operator: &Array2<f64>,
    assembly: f64,
    context: &'static str,
) -> Result<Option<SpectralSplit>, StateBlockError> {
    let size = operator.nrows();
    let mean = operator.diag().sum() / size as f64;
    let spread = operator
        .indexed_iter()
        .map(|((row, col), &value)| {
            let centered = if row == col { value - mean } else { value };
            centered * centered
        })
        .sum::<f64>()
        .sqrt();
    // Every eigenvalue lies within `spread` of `mean`, so no gap exceeds
    // `2·spread`. A resolved gap must exceed `2·(band + assembly)`, so at
    // `spread ≤ assembly` the eigendecomposition could not split, and it is
    // skipped.
    if spread <= assembly {
        return Ok(None);
    }
    let partition = spectral_partition(operator, assembly, context)?;
    Ok((partition.parts.len() > 1).then_some(partition))
}

/// Which of `values` the owned rank predicate resolves above its band. The
/// predicate counts them, and the counted ones are the largest.
fn resolved_flags(values: &[f64], assembly: f64) -> Vec<bool> {
    let count = resolved_eigenvalue_count(values, assembly);
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| values[right].total_cmp(&values[left]));
    let mut flags = vec![false; values.len()];
    for &index in order.iter().take(count) {
        flags[index] = true;
    }
    flags
}

/// Spectral-norm error of `XXᵀ` or `XᵀX` computed from a block `X` within
/// `error` of the exact block, with `inner` terms per product entry.
fn gram_band(block: &ArrayView2<'_, f64>, error: f64, inner: usize) -> f64 {
    let norm = frobenius(block);
    2.0 * norm * error + error * error + accumulation_growth(inner) * norm * norm
}

/// Spectral error of a product of `size × size` factors, each with spectral
/// norm `scale` and within `error` of exact: `Π(s + e) − Π s` from the inputs,
/// plus GEMM arithmetic. Every rounded entry is an accumulation against
/// `|X₁|⋯|X_k|`, whose spectral norm is at most `size^{k/2}·Π‖X_j‖₂`.
fn product_band(factors: &[(f64, f64)], size: usize) -> f64 {
    let exact: f64 = factors.iter().map(|factor| factor.0).product();
    let perturbed: f64 = factors.iter().map(|factor| factor.0 + factor.1).product();
    let arithmetic = accumulation_growth(factors.len().saturating_sub(1) * size)
        * (size as f64).powf(factors.len() as f64 / 2.0)
        * perturbed;
    perturbed - exact + arithmetic
}

fn frobenius(matrix: &ArrayView2<'_, f64>) -> f64 {
    matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// The symmetric matrix carrying `matrix`'s lower triangle. A full GEMM product
/// `XXᵀ` need not be bitwise symmetric, and the strict eigensolver reads one
/// triangle as the operator.
fn mirror_lower(mut matrix: Array2<f64>) -> Array2<f64> {
    let size = matrix.nrows();
    for row in 0..size {
        for col in (row + 1)..size {
            matrix[[row, col]] = matrix[[col, row]];
        }
    }
    matrix
}

fn symmetric_part(matrix: &ArrayView2<'_, f64>) -> Array2<f64> {
    let size = matrix.nrows();
    Array2::from_shape_fn((size, size), |(row, col)| {
        0.5 * (matrix[[row, col]] + matrix[[col, row]])
    })
}

fn antisymmetric_part(matrix: &ArrayView2<'_, f64>) -> Array2<f64> {
    let size = matrix.nrows();
    Array2::from_shape_fn((size, size), |(row, col)| {
        0.5 * (matrix[[row, col]] - matrix[[col, row]])
    })
}

#[cfg(test)]
#[path = "state_blocks_tests.rs"]
mod state_blocks_tests;

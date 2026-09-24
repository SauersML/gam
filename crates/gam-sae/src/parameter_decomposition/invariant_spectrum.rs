//! Components of a data cloud identified by its own isometry-invariant operators
//! (#2951).
//!
//! # Why sparsity cannot, and invariance can
//!
//! A subspace that every row uses cannot be split by any criterion that is invariant
//! under rotations inside it: sparsity of the coordinates, a smoothness penalty and
//! statistical independence all are. A grokked modular-addition embedding is the
//! standard case: every row lies on all of a handful of Fourier planes at once.
//!
//! Every operator on the rows built from their mutual geometry commutes with every
//! isometry that permutes the rows. Its eigenspaces are therefore unions of that
//! symmetry's irreducible pieces, and a two-dimensional piece (a rotation plane) shows
//! up as an eigenvalue of multiplicity two. No group is declared: the rows' geometry
//! carries it.
//!
//! # The operators
//!
//! With the centred rows `D` (`n × d`), their resolved column space `U` (`n × r`,
//! orthonormal) and the centred Gram `G = C D Dᵀ C`, the family is the Hadamard powers
//! `M_j = Uᵀ C (D Dᵀ)^{∘j} C U`, `j = 1, 2, …`: the polynomial part of the algebra the
//! pairwise inner products generate. `M_1` is `Uᵀ G U`, the principal-component
//! operator. A single operator can tie two different pieces by accident (two Fourier
//! planes of nearly equal power); the higher powers weight each piece by its
//! convolution structure and break such ties. Scale-family kernels do not: the
//! pairwise distances of a high-dimensional cloud concentrate, so every heat kernel is
//! nearly affine in the squared distance and repeats `M_1`.
//!
//! # Joint eigenspaces and the band
//!
//! The family is jointly diagonalized by Jacobi rotations (Cardoso–Souloumiac). Each
//! rotation minimizes the family's summed off-diagonal energy over its plane, so a
//! sweep never increases it; the iteration stops at the first sweep that does not
//! decrease it. On data with an exact
//! symmetry the off-diagonal energy goes to roundoff. On a trained model's data the
//! symmetry is approximate, and what is left off the diagonal is the non-invariant
//! part `E_j` of each operator. By Gershgorin, every eigenvalue of an operator lies in
//! the union of the discs centred on its diagonal entries with radii the row sums
//! `R_i = Σ_{k≠i} |E_ik|`: a local bound, so the rows of a clean symmetric plane are not
//! charged for the remainder the cloud's other directions carry (a global `‖E‖₂` would
//! be, and on a trained model's rows it ties everything). Two directions are one
//! component when, in every operator, their discs overlap (joint eigenvalues within
//! `R_i + R_k +` the operator's rounding): no operator of the family tells them apart. The components are the classes of that
//! relation, closed transitively, so a component's dimension is derived. A component
//! larger than the true piece is an unresolved union, never a split one.
//!
//! # How many powers
//!
//! A single operator is diagonalized exactly, so its remainder is zero and says nothing
//! about the rows' asymmetry; the band exists from two operators on. After that, a
//! power can only add constraints to "indistinguishable by every operator", so the
//! components can only split. The family stops at the first power that splits nothing
//! (the count of components does not grow), or when every component is a single
//! direction.
use gam_linalg::faer_ndarray::FaerSvd;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::fmt;

/// Why the invariant components could not be computed.
#[derive(Debug)]
pub enum InvariantSpectrumError {
    /// Fewer than two rows, or no resolved column space.
    Degenerate { rows: usize, rank: usize },
    NonFinite,
    Svd(String),
}

impl fmt::Display for InvariantSpectrumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Degenerate { rows, rank } => {
                write!(f, "invariant_spectrum: {rows} rows with resolved rank {rank}")
            }
            Self::NonFinite => write!(f, "invariant_spectrum: non-finite entry"),
            Self::Svd(e) => write!(f, "invariant_spectrum: SVD failed: {e}"),
        }
    }
}

impl std::error::Error for InvariantSpectrumError {}

/// The components and their evidence.
#[derive(Clone, Debug)]
pub struct InvariantComponents {
    /// Row coordinates of each joint eigen-direction (`n × r`), orthonormal columns.
    pub coordinates: Array2<f64>,
    /// Readout directions in the row space (`d × r`): `D readout = coordinates`.
    pub readout: Array2<f64>,
    /// Joint eigenvalues, one row per direction, one column per power (`r × J`).
    pub joint_eigenvalues: Array2<f64>,
    /// Per power, the largest Gershgorin radius of its non-invariant part left after
    /// joint diagonalization (`J`); the grouping reads every direction's own radius.
    pub band: Array1<f64>,
    /// Components as sets of direction indices, largest leading eigenvalue first.
    pub components: Vec<Vec<usize>>,
}

/// Off-diagonal part of a symmetric matrix and a spectral-norm bound for it
/// (Frobenius norm).
fn off_diagonal_norm(m: &Array2<f64>) -> f64 {
    let mut s = 0.0;
    for ((i, j), v) in m.indexed_iter() {
        if i != j {
            s += v * v;
        }
    }
    s.sqrt()
}

/// Jacobi joint diagonalization of symmetric matrices (Cardoso–Souloumiac): returns
/// the accumulated rotation (`r × r`, columns are directions).
fn joint_diagonalize(ops: &mut [Array2<f64>]) -> Array2<f64> {
    let r = ops[0].nrows();
    let mut v = Array2::<f64>::eye(r);
    let energy = |ops: &[Array2<f64>]| ops.iter().map(|m| off_diagonal_norm(m).powi(2)).sum::<f64>();
    let mut previous = energy(ops);
    loop {
        for i in 0..r {
            for j in (i + 1)..r {
                // 2x2 Gram of (a_ii - a_jj, 2 a_ij) over the family.
                let (mut g00, mut g01, mut g11) = (0.0, 0.0, 0.0);
                for m in ops.iter() {
                    let h0 = m[[i, i]] - m[[j, j]];
                    let h1 = 2.0 * m[[i, j]];
                    g00 += h0 * h0;
                    g01 += h0 * h1;
                    g11 += h1 * h1;
                }
                // Leading eigenvector of [[g00, g01], [g01, g11]].
                let tr = g00 + g11;
                let det = g00 * g11 - g01 * g01;
                let disc = (0.25 * tr * tr - det).max(0.0).sqrt();
                let lam = 0.5 * tr + disc;
                let (mut x, mut y) = if g01.abs() > 0.0 {
                    (lam - g11, g01)
                } else if g00 >= g11 {
                    (1.0, 0.0)
                } else {
                    (0.0, 1.0)
                };
                let norm = x.hypot(y);
                if norm == 0.0 {
                    continue;
                }
                x /= norm;
                y /= norm;
                if x < 0.0 {
                    x = -x;
                    y = -y;
                }
                let c = ((1.0 + x) / 2.0).sqrt();
                let s = y / (2.0 * (1.0 + x)).sqrt();
                if s == 0.0 {
                    continue;
                }
                for m in ops.iter_mut() {
                    for k in 0..r {
                        let (a, b) = (m[[k, i]], m[[k, j]]);
                        m[[k, i]] = c * a + s * b;
                        m[[k, j]] = -s * a + c * b;
                    }
                    for k in 0..r {
                        let (a, b) = (m[[i, k]], m[[j, k]]);
                        m[[i, k]] = c * a + s * b;
                        m[[j, k]] = -s * a + c * b;
                    }
                }
                for k in 0..r {
                    let (a, b) = (v[[k, i]], v[[k, j]]);
                    v[[k, i]] = c * a + s * b;
                    v[[k, j]] = -s * a + c * b;
                }
            }
        }
        let current = energy(ops);
        if !(current < previous) {
            return v;
        }
        previous = current;
    }
}

/// Gershgorin radii `R_i = Σ_{k≠i} |M_ik|` of one operator in the joint basis.
fn gershgorin_radii(m: &Array2<f64>) -> Array1<f64> {
    Array1::from_iter(m.outer_iter().enumerate().map(|(i, row)| {
        row.iter().enumerate().filter(|(k, _)| *k != i).map(|(_, v)| v.abs()).sum::<f64>()
    }))
}

/// Classes of "discs overlap in every operator", closed transitively. `radii` is
/// `r × J`.
fn group(lam: &Array2<f64>, radii: &Array2<f64>, rounding: &Array1<f64>) -> Vec<Vec<usize>> {
    let r = lam.nrows();
    let mut parent: Vec<usize> = (0..r).collect();
    fn find(p: &mut [usize], mut a: usize) -> usize {
        while p[a] != a {
            p[a] = p[p[a]];
            a = p[a];
        }
        a
    }
    for i in 0..r {
        for j in (i + 1)..r {
            let tied = (0..lam.ncols())
                .all(|q| (lam[[i, q]] - lam[[j, q]]).abs() <= radii[[i, q]] + radii[[j, q]] + rounding[q]);
            if tied {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[a] = b;
                }
            }
        }
    }
    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for i in 0..r {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }
    let mut out: Vec<Vec<usize>> = groups.into_values().collect();
    out.sort_by(|a, b| {
        let la = a.iter().map(|&i| lam[[i, 0]]).fold(f64::MIN, f64::max);
        let lb = b.iter().map(|&i| lam[[i, 0]]).fold(f64::MIN, f64::max);
        lb.total_cmp(&la)
    });
    out
}

/// The invariant components of the rows (module docs).
pub fn invariant_components(rows: ArrayView2<'_, f64>) -> Result<InvariantComponents, InvariantSpectrumError> {
    let (n, d) = rows.dim();
    if !rows.iter().all(|v| v.is_finite()) {
        return Err(InvariantSpectrumError::NonFinite);
    }
    if n < 2 {
        return Err(InvariantSpectrumError::Degenerate { rows: n, rank: 0 });
    }
    let mean = rows.mean_axis(Axis(0)).expect("n >= 2");
    let centred = &rows - &mean;
    let (u, s, vt) = centred.svd(true, true).map_err(|e| InvariantSpectrumError::Svd(e.to_string()))?;
    let (u, vt) = (u.expect("requested"), vt.expect("requested"));
    let band = (n.max(d) as f64) * f64::EPSILON * s[0];
    let rank = s.iter().filter(|&&v| v > band).count();
    if rank == 0 {
        return Err(InvariantSpectrumError::Degenerate { rows: n, rank });
    }
    let basis = u.slice(ndarray::s![.., ..rank]).to_owned();
    let gram = centred.dot(&centred.t());
    let centring = Array2::<f64>::eye(n) - 1.0 / n as f64;
    // Hadamard powers, grown while they refine the components.
    let mut raw: Vec<Array2<f64>> = Vec::new();
    let mut power = gram.clone();
    let mut result: Option<InvariantComponents> = None;
    for j in 1..=rank.max(2) {
        if j > 1 {
            power = &power * &gram;
        }
        let mut m = basis.t().dot(&centring.dot(&power).dot(&centring)).dot(&basis);
        m = (&m + &m.t()) * 0.5;
        let scale = m.iter().map(|v| v.abs()).fold(0.0, f64::max);
        if scale == 0.0 {
            break;
        }
        m.mapv_inplace(|v| v / scale);
        raw.push(m);
        let mut ops = raw.clone();
        let rotation = joint_diagonalize(&mut ops);
        let lam = Array2::from_shape_fn((rank, ops.len()), |(i, q)| ops[q][[i, i]]);
        let mut radii = Array2::<f64>::zeros((rank, ops.len()));
        for (q, m) in ops.iter().enumerate() {
            radii.column_mut(q).assign(&gershgorin_radii(m));
        }
        let bands = Array1::from_iter(radii.columns().into_iter().map(|c| c.iter().copied().fold(0.0, f64::max)));
        let rounding = Array1::from_elem(ops.len(), 4.0 * (n as f64) * (rank as f64) * f64::EPSILON);
        let components = group(&lam, &radii, &rounding);
        let fixed_point = raw.len() > 2 && result.as_ref().is_some_and(|previous| components.len() <= previous.components.len());
        let coordinates = basis.dot(&rotation);
        // D readout = coordinates on the resolved rank: readout = V S^-1 Uᵀ coordinates.
        let mut readout = Array2::<f64>::zeros((d, rank));
        for q in 0..rank {
            let proj = basis.t().dot(&coordinates.column(q));
            let mut col = Array1::<f64>::zeros(d);
            for a in 0..rank {
                col.scaled_add(proj[a] / s[a], &vt.row(a));
            }
            readout.column_mut(q).assign(&col);
        }
        // With one operator the joint diagonalization is exact and its remainder is zero by
        // construction, so the band measures nothing about the rows' asymmetry: a verdict
        // needs at least two invariant operators.
        let singletons = raw.len() > 1 && components.iter().all(|c| c.len() == 1);
        result = Some(InvariantComponents { coordinates, readout, joint_eigenvalues: lam, band: bands, components });
        if fixed_point || singletons {
            break;
        }
    }
    result.ok_or(InvariantSpectrumError::Degenerate { rows: n, rank })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows of an exact isometric cyclic embedding: `x_a = Σ_k r_k (cos(ω_k a) e_k + sin(ω_k a) f_k)`
    /// with `{e_k, f_k}` orthonormal in a generic rotation of `R^d`. Planes 1 and 2 have
    /// equal radius, so the principal-component operator ties them into a 4-dimensional
    /// eigenspace and only the higher Hadamard powers separate them. The components
    /// are the three planes, each of dimension two, and no group is given.
    #[test]
    fn a_cyclic_embeddings_planes_are_its_components() {
        use gam_linalg::faer_ndarray::FaerQr;
        let p = 31usize;
        let freqs = [1usize, 2, 3];
        let radii = [1.0, 1.0, 1.3];
        let d = 12;
        let mut state = 0x9e3779b97f4a7c15u64;
        let mut draw = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let generic = Array2::from_shape_fn((d, d), |_| draw());
        let (q, _) = generic.qr().unwrap();
        let mut rows = Array2::<f64>::zeros((p, d));
        for a in 0..p {
            for (k, &f) in freqs.iter().enumerate() {
                let w = std::f64::consts::TAU * (f * a) as f64 / p as f64;
                rows.row_mut(a).scaled_add(radii[k] * w.cos(), &q.column(2 * k));
                rows.row_mut(a).scaled_add(radii[k] * w.sin(), &q.column(2 * k + 1));
            }
        }
        let result = invariant_components(rows.view()).unwrap();
        let mut sizes: Vec<usize> = result.components.iter().map(Vec::len).collect();
        sizes.sort();
        assert_eq!(sizes, vec![2, 2, 2], "{:?} {:?}", result.components, result.joint_eigenvalues);
        // Each component's row coordinates span one frequency's cos/sin pair.
        for comp in &result.components {
            let coords = result.coordinates.select(Axis(1), comp);
            let best = freqs
                .iter()
                .map(|&f| {
                    let modes = Array2::from_shape_fn((p, 2), |(a, c)| {
                        let w = std::f64::consts::TAU * (f * a) as f64 / p as f64;
                        (if c == 0 { w.cos() } else { w.sin() }) * (2.0 / p as f64).sqrt()
                    });
                    let cross = coords.t().dot(&modes);
                    let (_, sv, _) = cross.svd(false, false).unwrap();
                    sv[0] * sv[1]
                })
                .fold(0.0, f64::max);
            assert!(best > 1.0 - 1e-8, "component {comp:?} overlap {best}");
        }
    }
}

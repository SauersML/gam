//! The leading eigenpairs of a positive semidefinite operator known only through
//! block products `V ↦ G V`, and the trace of what they leave out.
//!
//! A caller that owns the operator (the torch harvest owns a model's JVPs and VJPs)
//! supplies the products; the subspace iteration, the Rayleigh–Ritz step and the
//! deflated trace estimate are linear algebra and run here (#2899 P33).

use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::{RngExt, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use crate::faer_ndarray::{FaerSvd, strict_symmetric_eigh};
use crate::roundoff::SymmetricAssembly;

/// Leading eigenpairs of a PSD operator, eigenvalues descending.
#[derive(Debug, Clone)]
pub struct PsdTopEigenpairs {
    /// `rank` Ritz values, descending and non-negative.
    pub values: Array1<f64>,
    /// `dim × rank` orthonormal Ritz vectors, one per column.
    pub vectors: Array2<f64>,
}

/// An orthonormal basis for the column space of `block` (`cols ≤ rows`): the left
/// singular factor, which orders the directions by how much of `block` each carries.
fn range_basis(block: &Array2<f64>) -> Result<Array2<f64>, String> {
    let (left, _, _) = block
        .svd(true, false)
        .map_err(|error| format!("range basis: {error}"))?;
    left.ok_or_else(|| "range basis: the SVD omitted its left factor".to_string())
}

fn checked_product(
    apply: &mut impl FnMut(ArrayView2<'_, f64>) -> Result<Array2<f64>, String>,
    block: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    let image = apply(block.view())?;
    if image.dim() != block.dim() {
        return Err(format!(
            "operator product returned {:?} for a {:?} block",
            image.dim(),
            block.dim()
        ));
    }
    if image.iter().any(|value| !value.is_finite()) {
        return Err("operator product returned non-finite entries".to_string());
    }
    Ok(image)
}

/// The top `rank` eigenpairs of the PSD operator `apply` on `ℝ^dim`, by randomized
/// subspace iteration (Halko, Martinsson & Tropp 2011, Alg. 4.4) followed by a
/// Rayleigh–Ritz step.
///
/// The subspace has `m = min(dim, rank + oversample)` columns, drawn standard
/// normal from `seed`, and takes `power_steps` orthonormalized power steps. The
/// only dense decomposition is the `m × m` eigenproblem of `T = Qᵀ G Q`, which is
/// averaged with its transpose so the strict routine is entered with its assembly
/// declared. The Ritz values of a PSD operator are non-negative in exact arithmetic,
/// so a negative one is a zero eigenvalue's rounding and is reported as zero.
pub fn psd_top_eigenpairs_by_subspace_iteration(
    dim: usize,
    rank: usize,
    oversample: usize,
    power_steps: usize,
    seed: u64,
    mut apply: impl FnMut(ArrayView2<'_, f64>) -> Result<Array2<f64>, String>,
) -> Result<PsdTopEigenpairs, String> {
    if rank == 0 || rank > dim {
        return Err(format!("subspace iteration rank {rank} must lie in 1..={dim}"));
    }
    let columns = dim.min(rank.saturating_add(oversample));
    let mut rng = StdRng::seed_from_u64(seed);
    let start = Array2::from_shape_simple_fn((dim, columns), || {
        Distribution::<f64>::sample(&StandardNormal, &mut rng)
    });
    let mut basis = range_basis(&start)?;
    for _ in 0..power_steps {
        basis = range_basis(&checked_product(&mut apply, &basis)?)?;
    }
    let image = checked_product(&mut apply, &basis)?;
    let rayleigh = basis.t().dot(&image);
    let symmetric = (&rayleigh + &rayleigh.t()) * 0.5;
    let (values, vectors) =
        strict_symmetric_eigh(&symmetric, SymmetricAssembly::Mirrored, faer::Side::Lower)
            .map_err(|error| format!("Rayleigh–Ritz eigendecomposition: {error}"))?;
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    order.truncate(rank);
    let values = Array1::from_iter(order.iter().map(|&index| values[index].max(0.0)));
    let vectors = basis.dot(&vectors.select(Axis(1), &order));
    Ok(PsdTopEigenpairs { values, vectors })
}

/// `trace(P G P)` for `P = I − Q Qᵀ`, where `Q = basis` (`dim × r`) is orthonormal
/// and `quadratic_forms` returns `zᵢᵀ G zᵢ` for each column of a block.
///
/// With `probes ≥ dim` the probes are the identity columns and the sum is exact:
/// `Σᵢ (P eᵢ)ᵀ G (P eᵢ) = trace(P G P)`. Otherwise they are Rademacher vectors
/// drawn from `seed`, and the average is Hutchinson's unbiased estimate. Deflating
/// before probing is what makes that estimate usable: its variance is
/// `2(‖A‖_F² − Σ Aᵢᵢ²)` for the probed `A`, so probing `P G P` carries noise on the
/// scale of the tail, where probing `G` and subtracting the Ritz sum would carry it
/// on the scale of the leading eigenvalue.
pub fn deflated_trace(
    basis: ArrayView2<'_, f64>,
    probes: usize,
    seed: u64,
    mut quadratic_forms: impl FnMut(ArrayView2<'_, f64>) -> Result<Array1<f64>, String>,
) -> Result<f64, String> {
    let dim = basis.nrows();
    if probes == 0 {
        return Err("deflated trace needs at least one probe".to_string());
    }
    let (block, scale) = if probes >= dim {
        (Array2::<f64>::eye(dim), 1.0)
    } else {
        let mut rng = StdRng::seed_from_u64(seed);
        let block = Array2::from_shape_simple_fn((dim, probes), || {
            if rng.random::<bool>() { 1.0 } else { -1.0 }
        });
        (block, 1.0 / probes as f64)
    };
    let projected = &block - &basis.dot(&basis.t().dot(&block));
    let forms = quadratic_forms(projected.view())?;
    if forms.len() != projected.ncols() {
        return Err(format!(
            "quadratic forms returned {} values for {} probes",
            forms.len(),
            projected.ncols()
        ));
    }
    if forms.iter().any(|value| !value.is_finite()) {
        return Err("quadratic forms returned non-finite values".to_string());
    }
    Ok(forms.sum() * scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;

    /// A PSD operator with a known spectrum `diag(λ)` in a rotated basis.
    fn operator(dim: usize) -> (Array2<f64>, Array1<f64>) {
        let spectrum = Array1::from_shape_fn(dim, |i| 2.0_f64.powi(-(i as i32)));
        let raw = Array2::from_shape_fn((dim, dim), |(i, j)| {
            ((i * dim + j + 1) as f64 * 0.7137).sin()
        });
        let rotation = range_basis(&raw).unwrap();
        let g = rotation.dot(&Array2::from_diag(&spectrum)).dot(&rotation.t());
        (g, spectrum)
    }

    #[test]
    fn subspace_iteration_recovers_a_decaying_spectrum_2899() {
        let dim = 24;
        let (g, spectrum) = operator(dim);
        let pairs = psd_top_eigenpairs_by_subspace_iteration(dim, 4, 6, 6, 7, |block| {
            Ok(g.dot(&block))
        })
        .unwrap();
        for k in 0..4 {
            let relative = (pairs.values[k] - spectrum[k]).abs() / spectrum[k];
            assert!(relative < 1e-8, "λ_{k}: {} vs {}", pairs.values[k], spectrum[k]);
            let v = pairs.vectors.column(k);
            let residual = &g.dot(&v) - &(&v * pairs.values[k]);
            assert!(residual.dot(&residual).sqrt() < 1e-6 * spectrum[0]);
        }
        let gram = pairs.vectors.t().dot(&pairs.vectors);
        assert!((&gram - &Array2::<f64>::eye(4)).iter().all(|e| e.abs() < 1e-12));
    }

    #[test]
    fn exhaustive_deflated_trace_is_the_exact_tail_2899() {
        let dim = 16;
        let (g, spectrum) = operator(dim);
        let pairs =
            psd_top_eigenpairs_by_subspace_iteration(dim, dim, 0, 0, 3, |block| Ok(g.dot(&block)))
                .unwrap();
        let basis = pairs.vectors.slice(s![.., ..3]).to_owned();
        let tail = deflated_trace(basis.view(), dim, 0, |block| {
            Ok(Array1::from_shape_fn(block.ncols(), |j| {
                let z = block.column(j);
                z.dot(&g.dot(&z))
            }))
        })
        .unwrap();
        let exact: f64 = spectrum.iter().skip(3).sum();
        assert!((tail - exact).abs() <= 1e-12 * spectrum.sum(), "{tail} vs {exact}");
    }
}

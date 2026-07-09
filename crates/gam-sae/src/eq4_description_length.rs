//! Uniform fixed-distortion (Eq. 4) description-length scoring of a featurizer.
//!
//! This is the single Rust home for the Eq. 4 scorer that the manifold-zoo
//! benchmark and the #1026 close experiments consume (`bench/bsf_manifold_zoo`,
//! `experiments/1026_close`). It prices ONE fitted featurizer's reconstruction
//! at a stated per-token distortion (fixed R², a matched-EV operating point),
//! decomposing the code length into
//!
//! * **support** bits — the combinatorial `log₂ C(G, ⌊L0⌉)` cost of naming which
//!   of the `G` atoms fired, at the mean per-token support cardinality `L0`
//!   (formed from `lgamma`, so it never overflows a factorial);
//! * **code** bits — a JOINT reverse-water-filling of every atom's per-firing
//!   coordinate spectrum, each spectrum weighted by that atom's firing
//!   probability `p_g`, sharing ONE water level across all components with the
//!   residual, so the fixed total-distortion budget is split optimally between
//!   coding coordinates and leaving residual;
//! * **residual** bits — the same joint water level applied to the residual
//!   covariance spectrum (its own weight-1 component);
//! * **dictionary** bits — the amortised `½·(dictionary_params / N)·log₂(N)` BIC
//!   charge for storing the decoder.
//!
//! Unlike the per-featurizer [`crate::description_length::score`] surface (which
//! water-fills a single unweighted spectrum), the Eq. 4 scorer water-fills a
//! COLLECTION of firing-probability-weighted spectra against a shared level —
//! that joint allocation is [`water_fill_component_bits`], the piece that has no
//! analogue in [`crate::description_length`] and is ported faithfully here.
//!
//! # The featurizer surface
//!
//! The scorer needs, per atom `g`, the empirical spectrum of the per-firing
//! ATOM CONTRIBUTION (the atom's additive reconstruction term on the rows it
//! fires on). That contribution is produced by the caller's fitted model — a
//! closure the Python surface supplies — so [`eq4_fixed_distortion_description_length`]
//! is generic over a `fetch_contribution` callback that returns the
//! `(take, d)` contribution matrix for the selected firing rows. Rust owns the
//! firing-row selection, the subsampling cap, the skip rule for under-fired
//! atoms, the SVD spectrum, the covariance eigendecomposition, the water-filling
//! and the bit assembly; the callback ONLY materialises the atom's rows. This
//! keeps peak memory to one atom's contribution at a time (the caller may fetch
//! lazily), exactly as the reference NumPy loop did.

use ndarray::{Array1, Array2, ArrayView2};

use gam_linalg::faer_ndarray::{FaerEigh, FaerSvd};

/// The firing threshold above which a gate value counts as an active firing.
const GATE_ACTIVE_THRESHOLD: f64 = 1e-10;

/// The subsampling cap on the number of firing rows used to estimate an atom's
/// per-firing coordinate spectrum. When an atom fires on more than this many
/// rows, the rows are strided down to (at most) this count before the SVD.
const SPECTRUM_ROW_CAP: usize = 4096;

/// The bits at one R² operating point: total description length plus the code
/// and residual sub-terms (support and dictionary bits are the same at every
/// target and reported once on the parent [`Eq4DescriptionLength`]).
#[derive(Clone, Copy, Debug)]
pub struct Eq4TargetBits {
    /// The R² target this row was scored at (the fixed distortion is
    /// `(1 − target)·reference_variance`).
    pub target: f64,
    /// Total bits: `support + code + residual + dictionary`.
    pub bits: f64,
    /// The summed firing-weighted coordinate coding bits over all atoms.
    pub code_bits: f64,
    /// The residual component's coding bits at the shared water level.
    pub resid_bits: f64,
}

/// The Eq. 4 fixed-distortion description-length report of one featurizer.
#[derive(Clone, Debug)]
pub struct Eq4DescriptionLength {
    /// Combinatorial support cost `log₂ C(G, ⌊L0⌉)` (bits) — independent of the
    /// distortion target.
    pub support_bits: f64,
    /// Achieved mean per-token support cardinality `L0` (mean active atoms per
    /// row), the un-rounded value that the support cardinality rounds.
    pub achieved_block_l0: f64,
    /// One entry per R² target, in the order the targets were supplied.
    pub per_target: Vec<Eq4TargetBits>,
    /// The featurizer's own native bits/token, echoed through when supplied.
    pub native_bits_per_token: Option<f64>,
}

/// Joint reverse-water-filling of firing-weighted Gaussian spectra to a fixed
/// total distortion.
///
/// `components` is a list of `(weight, spectrum)` pairs; each spectrum's
/// variances are floored at `0`. A single water level `θ` is found by bisection
/// so that the weighted allocated distortion
/// `Σ_c weight_c · Σ_i min(variance_{c,i}, θ)` equals `total_distortion`, then
/// each component's rate is `weight_c · Σ_{i: v>θ} ½·log₂(v/θ)`. Returns one
/// rate (bits) per component, in input order.
///
/// Degenerate cases match the reference NumPy scorer exactly: when
/// `total_distortion ≥ Σ_c weight_c·Σ_i variance` (the whole source fits inside
/// the budget) or every variance is zero, all rates are `0`. A non-finite or
/// non-positive `total_distortion`, a non-finite/negative weight, or a
/// non-finite spectrum entry is an error.
pub fn water_fill_component_bits(
    components: &[(f64, Vec<f64>)],
    total_distortion: f64,
) -> Result<Vec<f64>, String> {
    if !total_distortion.is_finite() || total_distortion <= 0.0 {
        return Err(format!(
            "total distortion must be finite and positive, got {total_distortion}"
        ));
    }
    let mut spectra: Vec<(f64, Vec<f64>)> = Vec::with_capacity(components.len());
    let mut total_variance = 0.0_f64;
    let mut max_variance = 0.0_f64;
    for (weight, spectrum) in components {
        let weight = *weight;
        if !weight.is_finite() || weight < 0.0 {
            return Err(format!(
                "component weight must be finite and nonnegative, got {weight}"
            ));
        }
        let mut variances = Vec::with_capacity(spectrum.len());
        for &value in spectrum {
            if !value.is_finite() {
                return Err("component spectrum must contain only finite values".to_string());
            }
            let variance = value.max(0.0);
            variances.push(variance);
            if variance > max_variance {
                max_variance = variance;
            }
        }
        total_variance += weight * variances.iter().sum::<f64>();
        spectra.push((weight, variances));
    }
    if total_distortion >= total_variance || max_variance == 0.0 {
        return Ok(vec![0.0; spectra.len()]);
    }

    let (mut low, mut high) = (0.0_f64, max_variance);
    for _ in 0..200 {
        let water_level = 0.5 * (low + high);
        let allocated: f64 = spectra
            .iter()
            .map(|(weight, variances)| {
                weight * variances.iter().map(|&v| v.min(water_level)).sum::<f64>()
            })
            .sum();
        if allocated > total_distortion {
            high = water_level;
        } else {
            low = water_level;
        }
    }
    let water_level = 0.5 * (low + high);
    let rates = spectra
        .iter()
        .map(|(weight, variances)| {
            weight
                * variances
                    .iter()
                    .filter(|&&v| v > water_level)
                    .map(|&v| 0.5 * (v / water_level).log2())
                    .sum::<f64>()
        })
        .collect();
    Ok(rates)
}

/// The eigenvalues of the sample covariance of `values` (rows = observations),
/// `(centered.ᵀ centered) / max(N−1, 1)`, ascending. Mirrors the reference
/// `numpy.linalg.eigvalsh` on the column-centered Gram.
fn covariance_eigenvalues(values: ArrayView2<f64>) -> Result<Array1<f64>, String> {
    let centered = column_centered(values);
    let n = values.nrows();
    let denom = (n.saturating_sub(1)).max(1) as f64;
    let mut covariance = centered.t().dot(&centered);
    covariance.mapv_inplace(|v| v / denom);
    let (eigenvalues, _vectors) = covariance
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("residual covariance eigensolve failed: {e:?}"))?;
    Ok(eigenvalues)
}

/// Column-mean-center a matrix (subtract each column's mean from that column).
fn column_centered(values: ArrayView2<f64>) -> Array2<f64> {
    let mean = values
        .mean_axis(ndarray::Axis(0))
        .expect("nonempty matrix has a column mean");
    let mut centered = values.to_owned();
    for mut row in centered.rows_mut() {
        row -= &mean;
    }
    centered
}

/// The per-firing coordinate variance spectrum of one atom's contribution:
/// `σ_i² / max(rows−1, 1)` for the top `code_dim` singular values of the
/// column-centered contribution. Mirrors the reference
/// `svd(compute_uv=False)[:code_dim]² / max(rows−1, 1)`.
fn atom_code_spectrum(contribution: ArrayView2<f64>, code_dim: usize) -> Result<Vec<f64>, String> {
    let rows = contribution.nrows();
    let centered = column_centered(contribution);
    let (_u, singular_values, _vt) = centered
        .svd(false, false)
        .map_err(|e| format!("atom contribution SVD failed: {e:?}"))?;
    let denom = (rows.saturating_sub(1)).max(1) as f64;
    let keep = code_dim.min(singular_values.len());
    Ok(singular_values
        .iter()
        .take(keep)
        .map(|&s| s * s / denom)
        .collect())
}

/// Score `test_x` against a featurizer's reconstruction at each R² target and
/// return the Eq. 4 fixed-distortion description length.
///
/// * `test_x` / `recon` — the held-out activations and the featurizer's
///   reconstruction of them; same shape `(N, d)`, both finite.
/// * `gate` — the `(N, G)` per-atom firing gate; an atom fires on a row when its
///   gate there exceeds `1e-10`.
/// * `code_dims` — the coded-coordinate dimension `d_g` of each of the `G`
///   atoms (length `G`, nonnegative).
/// * `dictionary_params` — the decoder scalar count charged the BIC dictionary
///   term.
/// * `r2_targets` — the fixed-distortion R² operating points, each finite and in
///   `[0, 1)`; must be nonempty.
/// * `native_bits_per_token` — echoed onto the report when present.
/// * `fetch_contribution` — a callback returning the `(take.len, d)` contribution
///   matrix of atom `g` restricted to the supplied firing-row indices `take`.
///   Invoked only for atoms that clear the skip rule, one atom at a time.
///
/// The firing-row selection, the `4096`-row subsampling cap, the skip rule for
/// atoms firing on fewer than `max(d_g + 1, 4)` rows, and every numerical term
/// live here; the callback only materialises rows.
pub fn eq4_fixed_distortion_description_length<F>(
    test_x: ArrayView2<f64>,
    recon: ArrayView2<f64>,
    gate: ArrayView2<f64>,
    code_dims: &[i64],
    dictionary_params: i64,
    r2_targets: &[f64],
    native_bits_per_token: Option<f64>,
    mut fetch_contribution: F,
) -> Result<Eq4DescriptionLength, String>
where
    F: FnMut(usize, &[usize]) -> Result<Array2<f64>, String>,
{
    let (n, d) = (test_x.nrows(), test_x.ncols());
    if test_x.dim() != recon.dim() {
        return Err(format!(
            "test_x and recon must have the same shape, got {:?} and {:?}",
            test_x.dim(),
            recon.dim()
        ));
    }
    if n == 0 || d == 0 {
        return Err("test_x must contain at least one row and one column".to_string());
    }
    let n_atoms = gate.ncols();
    if gate.nrows() != n {
        return Err(format!(
            "gate and recon must contain the same number of rows, got {} and {}",
            gate.nrows(),
            n
        ));
    }
    if code_dims.len() != n_atoms {
        return Err(format!(
            "code_dims must have one entry per atom, got {} for {} atoms",
            code_dims.len(),
            n_atoms
        ));
    }
    if !test_x.iter().all(|v| v.is_finite()) || !recon.iter().all(|v| v.is_finite()) {
        return Err("test_x and recon must contain only finite values".to_string());
    }
    if !gate.iter().all(|v| v.is_finite()) {
        return Err("gate must contain only finite values".to_string());
    }
    if r2_targets.is_empty() {
        return Err("r2_targets must not be empty".to_string());
    }
    if !r2_targets
        .iter()
        .all(|&t| t.is_finite() && (0.0..1.0).contains(&t))
    {
        return Err("every R-squared target must be finite and in [0, 1)".to_string());
    }

    // Support: firing probability per atom and mean per-token support cardinality.
    let mut active_per_atom = vec![0.0_f64; n_atoms];
    let mut total_active = 0.0_f64;
    for row in 0..n {
        for atom in 0..n_atoms {
            if gate[[row, atom]] > GATE_ACTIVE_THRESHOLD {
                active_per_atom[atom] += 1.0;
                total_active += 1.0;
            }
        }
    }
    let p_g: Vec<f64> = active_per_atom.iter().map(|&c| c / n as f64).collect();
    let l0 = total_active / n as f64;
    // Python `round(L0)` rounds half to even; clamp to `[0, G]`.
    let support_cardinality = (l0.round_ties_even() as i64).clamp(0, n_atoms as i64);
    let support_bits = (ln_gamma((n_atoms + 1) as f64)
        - ln_gamma((support_cardinality + 1) as f64)
        - ln_gamma((n_atoms as i64 - support_cardinality + 1) as f64))
        / std::f64::consts::LN_2;

    // Residual covariance spectrum and the reference variance the targets scale.
    let residual = &test_x.to_owned() - &recon;
    let residual_covariance_eigenvalues = covariance_eigenvalues(residual.view())?;
    let centered_x = column_centered(test_x);
    // reference_variance = mean(centered²)·d = Σ centered² / N.
    let reference_variance = centered_x.iter().map(|&v| v * v).sum::<f64>() / n as f64;
    if reference_variance <= 0.0 {
        return Err("test_x must have positive variance".to_string());
    }

    // Per-atom firing-coordinate spectra (weight-`p_g` water-fill components).
    let mut code_spectra: Vec<Vec<f64>> = Vec::with_capacity(n_atoms);
    for atom in 0..n_atoms {
        let code_dim = code_dims[atom].max(0) as usize;
        let rows: Vec<usize> = (0..n)
            .filter(|&row| gate[[row, atom]] > GATE_ACTIVE_THRESHOLD)
            .collect();
        if rows.len() < (code_dim + 1).max(4) {
            code_spectra.push(vec![0.0; code_dim]);
            continue;
        }
        let take: Vec<usize> = if rows.len() <= SPECTRUM_ROW_CAP {
            rows
        } else {
            let step = (rows.len() / SPECTRUM_ROW_CAP).max(1);
            rows.iter().step_by(step).copied().collect()
        };
        let contribution = fetch_contribution(atom, &take)?;
        if contribution.dim() != (take.len(), d) {
            return Err(format!(
                "atom {atom} contribution has shape {:?}; expected {:?}",
                contribution.dim(),
                (take.len(), d)
            ));
        }
        if !contribution.iter().all(|v| v.is_finite()) {
            return Err(format!("atom {atom} contribution contains non-finite values"));
        }
        code_spectra.push(atom_code_spectrum(contribution.view(), code_dim)?);
    }

    // Dictionary bits are the same at every target.
    let dictionary_bits =
        0.5 * dictionary_params as f64 / n as f64 * (n.max(2) as f64).log2();

    let mut per_target = Vec::with_capacity(r2_targets.len());
    for &target in r2_targets {
        let total_distortion = (1.0 - target) * reference_variance;
        let mut components: Vec<(f64, Vec<f64>)> = p_g
            .iter()
            .zip(code_spectra.iter())
            .map(|(&probability, spectrum)| (probability, spectrum.clone()))
            .collect();
        components.push((1.0, residual_covariance_eigenvalues.to_vec()));
        let component_bits = water_fill_component_bits(&components, total_distortion)?;
        let code_bits: f64 = component_bits[..n_atoms].iter().sum();
        let resid_bits = component_bits[n_atoms];
        per_target.push(Eq4TargetBits {
            target,
            bits: support_bits + code_bits + resid_bits + dictionary_bits,
            code_bits,
            resid_bits,
        });
    }

    Ok(Eq4DescriptionLength {
        support_bits,
        achieved_block_l0: l0,
        per_target,
        native_bits_per_token,
    })
}

/// `ln Γ(x)` via statrs, matching Python `math.lgamma` on the nonnegative
/// integer arguments the support term forms.
fn ln_gamma(x: f64) -> f64 {
    statrs::function::gamma::ln_gamma(x)
}

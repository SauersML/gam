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
//!   charge for storing the decoder, where `N` is the DECLARED
//!   `amortization_horizon` (the message/deployment horizon or declared
//!   training-observation count), NOT the number of rows sampled to estimate the
//!   score. The estimation subsample size (`estimation_rows = test_x.nrows()`)
//!   controls ONLY the Monte-Carlo variance of the support / code / residual
//!   expectations; it must never leak into the dictionary code (#2283 / audit
//!   §21). Conflating the two made the authoritative bits-at-R² row meaningless
//!   (the same fitted flat model priced radically different dictionary bits at
//!   256 vs 8192 estimation rows), so the two `N`s are now passed separately and
//!   the dictionary term depends on the horizon alone.
//!
//! Unlike the per-featurizer [`crate::description_length::score`] surface (which
//! water-fills a single unweighted spectrum), the Eq. 4 scorer water-fills a
//! collection of firing-probability-weighted spectra against a shared level via
//! [`crate::description_length::weighted_reverse_water_filling`].
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

use crate::description_length::{selection_bits, weighted_reverse_water_filling};

/// Standard fixed-distortion reporting points shared by every front-end.
pub const DEFAULT_EQ4_R2_TARGETS: &[f64] = &[0.99, 0.95, 0.90, 0.80];

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
    /// Amortised BIC dictionary charge
    /// `0.5 * dictionary_params / amortization_horizon * log2(amortization_horizon)`,
    /// shared by every target. Depends ONLY on the declared `amortization_horizon`,
    /// never on `estimation_rows` (#2283): re-estimating the score on a different
    /// row subsample leaves this term bitwise identical.
    pub dictionary_bits: f64,
    /// The number of rows actually used to estimate the code / residual / support
    /// expectations (`test_x.nrows()`). This is the Monte-Carlo estimator size; it
    /// affects only estimator variance and is reported for provenance. It is NOT
    /// the dictionary amortisation horizon (see [`Self::amortization_horizon`]).
    pub estimation_rows: i64,
    /// The declared amortisation horizon `N` charged in the dictionary code (the
    /// message/deployment horizon or declared training-observation count). Echoed
    /// through so a reader can confirm the dictionary term is sample-invariant.
    pub amortization_horizon: i64,
    /// One entry per R² target, in the order the targets were supplied.
    pub per_target: Vec<Eq4TargetBits>,
    /// The featurizer's own native bits/token, echoed through when supplied.
    pub native_bits_per_token: Option<f64>,
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
    let denom = (rows.saturating_sub(1)).max(1) as f64;
    // Flat-atom fast path (#2233). A `code_dim == 1` atom transmits a single
    // scalar code times one decoder row, so its contribution — and, since
    // column-centering only subtracts a per-column constant, its centered form —
    // is exactly rank one. A rank-one matrix has a single nonzero singular value
    // equal to its Frobenius norm, so `σ₁² = ‖centered‖_F²` with no SVD. This is
    // the dominant scorer cost at large overcompleteness (a K=32768 TopK
    // dictionary is entirely flat atoms), where an O(rows·d) sum of squares
    // replaces an O(rows·d·min(rows,d)) SVD per atom. It is exact for the rank-one
    // flat contributions the scorer is fed (parity-gated against the SVD path);
    // it would over-count if handed a genuinely higher-rank `code_dim == 1`
    // contribution, which the featurizer construction never produces.
    if code_dim == 1 {
        let frobenius_sq: f64 = centered.iter().map(|&value| value * value).sum();
        return Ok(vec![frobenius_sq / denom]);
    }
    let (_u, singular_values, _vt) = centered
        .svd(false, false)
        .map_err(|e| format!("atom contribution SVD failed: {e:?}"))?;
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
///   term. This is a STORAGE-CODE scalar count (the number of decoder scalars a
///   receiver must be handed to reconstruct: `K_flat·P + K_curved·b·P`), NOT a
///   BIC free-identifiable/effective dimension. The two coincide only for a
///   full-rank unpenalised decoder; the scorer declares the storage-code reading
///   explicitly so a future edit cannot silently relabel it as EDF (#2283 / audit
///   §21).
/// * `amortization_horizon` — the DECLARED `N` charged in the dictionary code
///   `0.5·dictionary_params/N·log₂(N)`: the message/deployment horizon or the
///   declared training-observation count. It is passed SEPARATELY from the
///   estimation subsample (`test_x.nrows()`), and must be at least `2` (an
///   `Err` is returned otherwise — the horizon is never silently defaulted to the
///   estimation subsample, so the #2283 confound cannot recur). The dictionary
///   term depends on this value ALONE.
/// * `r2_targets` — the fixed-distortion R² operating points, each finite and in
///   `[0, 1)`; must be nonempty.
/// * `native_bits_per_token` — echoed onto the report when present.
/// * `fetch_contribution` — a callback returning the `(take.len, d)` contribution
///   matrix of atom `g` restricted to the supplied firing-row indices `take`.
///   Invoked only for atoms that clear the skip rule, one atom at a time.
///
/// The number of rows of `test_x` / `recon` / `gate` is the `estimation_rows`
/// Monte-Carlo estimator size: it drives ONLY the variance of the support / code
/// / residual expectations, never the dictionary code. The firing-row selection,
/// the `4096`-row subsampling cap, the skip rule for atoms firing on fewer than
/// `max(d_g + 1, 4)` rows, and every numerical term live here; the callback only
/// materialises rows.
pub fn eq4_fixed_distortion_description_length<F>(
    test_x: ArrayView2<f64>,
    recon: ArrayView2<f64>,
    gate: ArrayView2<f64>,
    code_dims: &[i64],
    dictionary_params: i64,
    amortization_horizon: i64,
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
    if code_dims.iter().any(|&dimension| dimension < 0) {
        return Err("code_dims must contain only nonnegative dimensions".to_string());
    }
    if dictionary_params < 0 {
        return Err("dictionary_params must be nonnegative".to_string());
    }
    // The amortisation horizon is a DECLARED quantity, passed separately from the
    // estimation subsample and never inferred from it (#2283). Requiring it to be
    // at least 2 keeps `log₂(N)` non-negative and well-posed and forces every
    // caller to state the horizon explicitly rather than let the score silently
    // adopt the Monte-Carlo subsample size.
    if amortization_horizon < 2 {
        return Err(format!(
            "amortization_horizon must be at least 2 (the declared message/deployment \
             or training-observation N); it is passed separately from the {n}-row \
             estimation subsample and is never defaulted to it, got {amortization_horizon}"
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
    if native_bits_per_token.is_some_and(|bits| !bits.is_finite() || bits < 0.0) {
        return Err("native_bits_per_token must be finite and nonnegative".to_string());
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
    let support_bits = selection_bits(n_atoms as i64, support_cardinality);

    // Residual covariance spectrum and the reference variance the targets scale.
    let mut residual = test_x.to_owned();
    residual -= &recon;
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
        let code_dim = code_dims[atom] as usize;
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
            let step = rows.len().div_ceil(SPECTRUM_ROW_CAP);
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
            return Err(format!(
                "atom {atom} contribution contains non-finite values"
            ));
        }
        code_spectra.push(atom_code_spectrum(contribution.view(), code_dim)?);
    }

    // Dictionary bits are the same at every target AND independent of the
    // estimation subsample: the charge is `0.5·params/N·log₂(N)` in the DECLARED
    // amortization horizon `N`, never the `n`-row Monte-Carlo subsample (#2283).
    let horizon = amortization_horizon as f64;
    let dictionary_bits = 0.5 * dictionary_params as f64 / horizon * horizon.log2();

    let mut per_target = Vec::with_capacity(r2_targets.len());
    for &target in r2_targets {
        let total_distortion = (1.0 - target) * reference_variance;
        let mut components: Vec<(f64, Vec<f64>)> = p_g
            .iter()
            .zip(code_spectra.iter())
            .map(|(&probability, spectrum)| (probability, spectrum.clone()))
            .collect();
        components.push((1.0, residual_covariance_eigenvalues.to_vec()));
        let component_bits = weighted_reverse_water_filling(&components, total_distortion)?;
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
        dictionary_bits,
        estimation_rows: n as i64,
        amortization_horizon,
        per_target,
        native_bits_per_token,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The declared amortisation horizon used by the fixtures — distinct from the
    /// 6-row estimation subsample so the two `N`s can never be confused.
    const FIXTURE_HORIZON: i64 = 4096;

    fn fixture(code_dims: &[i64], dictionary_params: i64) -> Result<Eq4DescriptionLength, String> {
        let test_x = array![
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 1.5],
            [3.0, 1.0],
            [4.0, 2.0],
            [5.0, 3.0],
        ];
        let recon = test_x.mapv(|value| 0.8 * value);
        let gate = Array2::ones((test_x.nrows(), 1));
        let contribution = recon.clone();
        eq4_fixed_distortion_description_length(
            test_x.view(),
            recon.view(),
            gate.view(),
            code_dims,
            dictionary_params,
            FIXTURE_HORIZON,
            &[0.9],
            Some(1.25),
            move |_atom, take| {
                let mut selected = Array2::zeros((take.len(), contribution.ncols()));
                for (out_row, &source_row) in take.iter().enumerate() {
                    selected
                        .row_mut(out_row)
                        .assign(&contribution.row(source_row));
                }
                Ok(selected)
            },
        )
    }

    #[test]
    fn production_eq4_fixture_reconciles_report_terms() {
        let result = fixture(&[1], 4).unwrap();
        assert_eq!(result.support_bits, selection_bits(1, 1));
        assert_eq!(result.achieved_block_l0, 1.0);
        assert_eq!(result.native_bits_per_token, Some(1.25));
        assert_eq!(result.per_target.len(), 1);
        assert_eq!(result.estimation_rows, 6);
        assert_eq!(result.amortization_horizon, FIXTURE_HORIZON);
        let target = result.per_target[0];
        // The dictionary charge uses the DECLARED horizon, not the 6-row subsample.
        let horizon = FIXTURE_HORIZON as f64;
        let dictionary_bits = 0.5 * 4.0 / horizon * horizon.log2();
        assert_eq!(result.dictionary_bits, dictionary_bits);
        assert!(
            (target.bits
                - (result.support_bits + target.code_bits + target.resid_bits + dictionary_bits))
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn production_eq4_rejects_negative_dimensions_and_dictionary_cost() {
        assert!(fixture(&[-1], 0).unwrap_err().contains("code_dims"));
        assert!(fixture(&[1], -1).unwrap_err().contains("dictionary_params"));
    }

    /// The horizon is required and separately declared: a caller that passes a
    /// horizon below 2 (e.g. one that tried to conflate it with a tiny estimation
    /// subsample) gets a typed error naming `amortization_horizon`, so the #2283
    /// confound cannot recur through a silent default.
    #[test]
    fn production_eq4_rejects_a_sub_two_amortization_horizon() {
        let test_x = array![[0.0, 0.0], [1.0, 0.5], [2.0, 1.5], [3.0, 1.0]];
        let recon = test_x.mapv(|value| 0.8 * value);
        let gate = Array2::ones((test_x.nrows(), 1));
        let contribution = recon.clone();
        for horizon in [1_i64, 0, -8] {
            let err = eq4_fixed_distortion_description_length(
                test_x.view(),
                recon.view(),
                gate.view(),
                &[1],
                4,
                horizon,
                &[0.9],
                None,
                |_atom, take| {
                    let mut selected = Array2::zeros((take.len(), contribution.ncols()));
                    for (out_row, &source_row) in take.iter().enumerate() {
                        selected.row_mut(out_row).assign(&contribution.row(source_row));
                    }
                    Ok(selected)
                },
            )
            .unwrap_err();
            assert!(
                err.contains("amortization_horizon"),
                "horizon {horizon} error should name amortization_horizon: {err}"
            );
        }
    }

    /// Audit §34 "MDL sample invariance": holding `amortization_horizon` fixed,
    /// scoring the SAME fitted model at estimation subsamples of 256 / 1024 / 8192
    /// rows must leave the dictionary term BITWISE identical, and the total must
    /// agree with the 8192-row total to within the Monte-Carlo standard error of
    /// the code / residual / support expectations at the smaller sample. This is
    /// the acceptance test for #2283: the dictionary code no longer picks up the
    /// estimation subsample size.
    ///
    /// # Why the totals need an n-dependent tolerance, not a flat one
    ///
    /// Only the DICTIONARY term is an exact function of declared quantities. The
    /// code / residual / support terms are expectations under the row distribution,
    /// estimated from the `n` sampled rows, so
    ///     `bits(n) = bits(∞) + O(1/n) [bias] + O_p(σ_bits/√n) [variance]`.
    /// A constant relative tolerance is mis-calibrated by construction — it demands
    /// that a 256-row estimate be as accurate as an 8192-row one. The bound used
    /// below covers both terms and shrinks with `n`: the variance scale `σ_bits` is
    /// MEASURED from 32 disjoint row blocks, and the bias term is DERIVED from the
    /// scorer's own `n−1`-vs-`n` normalisation mismatch (see the block comment at
    /// the tolerance). Neither is a tuned constant.
    ///
    /// Rows are drawn i.i.d. from a fixed deterministic generator, so the leading
    /// `n` rows really are a random subsample of the same population (the previous
    /// fixture's rows were a smooth ramp in `i`, where "the first 256 rows" is a
    /// different, narrower population — the drift it produced was a systematic
    /// coverage difference, not Monte-Carlo error).
    #[test]
    fn eq4_dictionary_term_is_invariant_to_the_estimation_subsample() {
        // One fitted model (a fixed 0.85 shrinkage decoder, a single flat code_dim-1
        // atom firing on every row) scored on i.i.d. rows. Deterministic LCG +
        // Box–Muller: the sequence is fixed run to run, but the rows are exchangeable,
        // which is exactly what makes "the first n rows" an honest MC subsample.
        let full_rows = 8192_usize;
        let d = 3_usize;
        let mut state = 0x2283_5EED_u64;
        let mut next_uniform = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Open unit interval: Box–Muller must never see an exact zero.
            ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
        };
        let mut next_normal = move || {
            let u1: f64 = next_uniform();
            let u2: f64 = next_uniform();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };
        let mut test_x = Array2::<f64>::zeros((full_rows, d));
        for i in 0..full_rows {
            // Correlated, non-isotropic rows so the residual spectrum is nontrivial.
            let z0 = next_normal();
            let z1 = next_normal();
            let z2 = next_normal();
            test_x[[i, 0]] = z0;
            test_x[[i, 1]] = 0.6 * z0 + 0.8 * z1;
            test_x[[i, 2]] = 0.3 * z0 - 0.4 * z1 + 0.5 * z2;
        }
        let recon = test_x.mapv(|value| 0.85 * value);
        let gate = Array2::ones((full_rows, 1));

        let horizon = 120_000_i64;
        let dictionary_params = 4096_i64;
        let targets = [0.99, 0.95, 0.90, 0.80];

        // Score the contiguous row window `[start, start + len)` as an independent
        // estimation subsample of the SAME fitted model.
        let score_range = |start: usize, len: usize| -> Eq4DescriptionLength {
            let window = ndarray::s![start..start + len, ..];
            let contribution = recon.slice(window).to_owned();
            eq4_fixed_distortion_description_length(
                test_x.slice(window),
                recon.slice(window),
                gate.slice(window),
                &[1],
                dictionary_params,
                horizon,
                &targets,
                None,
                move |_atom, take| {
                    let mut selected = Array2::zeros((take.len(), contribution.ncols()));
                    for (out_row, &source_row) in take.iter().enumerate() {
                        selected.row_mut(out_row).assign(&contribution.row(source_row));
                    }
                    Ok(selected)
                },
            )
            .unwrap()
        };

        let small = score_range(0, 256);
        let medium = score_range(0, 1024);
        let large = score_range(0, full_rows);

        // Every run declared the same horizon and priced the same decoder.
        for run in [&small, &medium, &large] {
            assert_eq!(run.amortization_horizon, horizon);
        }
        assert_eq!(small.estimation_rows, 256);
        assert_eq!(medium.estimation_rows, 1024);
        assert_eq!(large.estimation_rows, 8192);

        // ── The load-bearing §21 assertion ────────────────────────────────────
        // The dictionary term is BITWISE identical across subsamples: it depends
        // only on the declared horizon, so re-estimation does not move it at all.
        let expected_dict =
            0.5 * dictionary_params as f64 / horizon as f64 * (horizon as f64).log2();
        assert_eq!(small.dictionary_bits, expected_dict);
        assert_eq!(medium.dictionary_bits, small.dictionary_bits);
        assert_eq!(large.dictionary_bits, small.dictionary_bits);

        // ── The estimator's two n-dependent error terms ───────────────────────
        // `bits(n) − bits(∞)` has exactly two sources, and the tolerance must cover
        // BOTH. Neither is a free constant: one is measured, one is derived.
        //
        // (1) VARIANCE, O(σ/√n). Measured, not assumed. Split the 8192 rows into 32
        //     DISJOINT 256-row blocks and score each: the spread of those 32 totals
        //     is a direct, assumption-light estimate of the sampling sd of a 256-row
        //     bits estimate, `σ̂₂₅₆`, per target.
        //
        // (2) BIAS, O(1/n). Derived. The bits functional is scale-invariant —
        //     `bits(α·spectra, α·D) = bits(spectra, D)`, since reverse water-filling
        //     only ever sees variance RATIOS. The scorer forms the spectra with the
        //     Bessel denominator `n−1` (`covariance_eigenvalues`, `atom_code_spectrum`)
        //     but forms `reference_variance` — hence the distortion budget
        //     `D = (1−R²)·reference_variance` — with denominator `n`. By scale
        //     invariance that mismatch is EXACTLY equivalent to inflating the budget
        //     by `c(n) = n/(n−1)`, a deterministic O(1/n) perturbation that survives
        //     even at zero sampling noise (at the loose targets the block spread is
        //     ~1e-15, yet the totals still move — this term, not noise, is what a
        //     variance-only tolerance would wrongly flag).
        //
        // Bounding (2) by the envelope theorem: at the shared water level `L`,
        // `∂bits/∂D = −M/(2 ln2 · L)` with `M = Σ_{above water} wᵢ`, and the budget
        // moves by `D·(c−1) = D/(n−1)`, so
        //     |Δbits| ≤ M/(2 ln2) · (D/L) / (n−1) ≤ W²/(2 ln2) / (n−1),
        // using `M ≤ W` and `D/L ≤ W`, where `W = Σ_g p_g + 1` is the total
        // water-filling component weight (every atom weighted by its firing
        // probability, plus the weight-1 residual). `Σ_g p_g` is exactly the reported
        // mean per-token support cardinality, so `W` is read off the report — it is a
        // derived quantity, not a tuned one.
        const BLOCK: usize = 256;
        let blocks = full_rows / BLOCK;
        let mut block_bits: Vec<Vec<f64>> = vec![Vec::with_capacity(blocks); targets.len()];
        for block in 0..blocks {
            let scored = score_range(block * BLOCK, BLOCK);
            for (t, row) in scored.per_target.iter().enumerate() {
                block_bits[t].push(row.bits);
            }
        }
        let block_sd: Vec<f64> = block_bits
            .iter()
            .map(|samples| {
                let mean = samples.iter().sum::<f64>() / samples.len() as f64;
                let variance = samples.iter().map(|b| (b - mean).powi(2)).sum::<f64>()
                    / (samples.len() - 1) as f64;
                variance.sqrt()
            })
            .collect();

        // Total component weight W = Σ_g p_g + 1 (residual). Σ_g p_g is the reported
        // mean support cardinality.
        let total_weight = large.achieved_block_l0 + 1.0;
        let bias_bound = |n: usize| -> f64 {
            total_weight * total_weight / (2.0 * std::f64::consts::LN_2)
                * (1.0 / (n as f64 - 1.0) - 1.0 / (full_rows as f64 - 1.0))
        };

        // Standard error of the DIFFERENCE between a nested n-row estimate and the
        // N-row estimate. The n rows are a SUBSET of the N rows, so for an
        // average-like estimator the shared rows cancel and
        //     Var[bits(n) − bits(N)] = σ²·(1/n − 1/N)
        // (strictly tighter than the independent-samples σ²(1/n + 1/N), and it
        // correctly vanishes as n → N). σ² is extrapolated from the measured block
        // spread by the CLT scaling law: σ² = BLOCK·σ̂₂₅₆².
        let se_diff = |sd: f64, n: usize| -> f64 {
            (sd * sd * BLOCK as f64 * (1.0 / n as f64 - 1.0 / full_rows as f64)).sqrt()
        };
        // k = 6 standard errors on the variance term. Under the CLT/delta-method
        // normal approximation that is a ~2e-9 two-sided tail per comparison (8
        // comparisons here ⇒ family-wise false-alarm rate ~1e-8); with NO
        // distributional assumption at all, Chebyshev still caps each comparison at
        // 1/36. It is a significance level, not a fudge factor: it does not scale
        // with the observed discrepancy.
        const K_SIGMA: f64 = 6.0;

        for (target_idx, &target) in targets.iter().enumerate() {
            let s = small.per_target[target_idx].bits;
            let m = medium.per_target[target_idx].bits;
            let l = large.per_target[target_idx].bits;
            let sd = block_sd[target_idx];
            assert!(
                sd.is_finite() && sd >= 0.0,
                "target {target} block spread must be finite and nonnegative, got {sd}"
            );
            // bias(n) + k·se(n); the 1e-9·(1+|l|) floor only absorbs float noise in
            // the exactly-shared support/dictionary terms.
            let tolerance = |n: usize| -> f64 {
                bias_bound(n) + K_SIGMA * se_diff(sd, n) + 1.0e-9 * (1.0 + l.abs())
            };
            let tol_small = tolerance(256);
            let tol_medium = tolerance(1024);
            assert!(
                (s - l).abs() <= tol_small,
                "target {target}: 256-row total {s} differs from the 8192-row total {l} \
                 by {} > bias+{K_SIGMA}σ ({tol_small}); σ̂₂₅₆ = {sd}",
                (s - l).abs()
            );
            // Convergence: BOTH error terms shrink with n, so the 1024-row bound is
            // strictly tighter than the 256-row bound — and the estimate still meets
            // it. This is the "more rows ⇒ closer to the limit" guard, stated as a
            // shrinking bound the estimator must satisfy rather than as a strict
            // ordering of two single noisy realisations (which would be flaky by
            // construction).
            assert!(
                tol_medium < tol_small,
                "the n-row bound must tighten with n: {tol_medium} !< {tol_small}"
            );
            assert!(
                (m - l).abs() <= tol_medium,
                "target {target}: 1024-row total {m} differs from the 8192-row total {l} \
                 by {} > bias+{K_SIGMA}σ ({tol_medium}); σ̂₂₅₆ = {sd}",
                (m - l).abs()
            );

            // ── Anti-vacuity ──────────────────────────────────────────────────
            // The tolerance must stay far too tight to hide the #2283 confound. Under
            // the OLD behaviour the dictionary term was `0.5·params·log₂(n)/n` in the
            // ESTIMATION rows, so the SAME fitted model's total moved by ~60.7 bits
            // between 256 and 8192 rows. Assert that swing is orders of magnitude
            // above the tolerance just allowed — i.e. this test would still FAIL
            // loudly on the pre-fix scorer, and has not been widened into vacuity.
            let confounded_dictionary_bits =
                |n: usize| 0.5 * dictionary_params as f64 * (n as f64).log2() / n as f64;
            let confound_swing =
                (confounded_dictionary_bits(256) - confounded_dictionary_bits(8192)).abs();
            assert!(
                confound_swing > 10.0 * tol_small,
                "target {target}: tolerance {tol_small} is too loose to catch the #2283 \
                 confound, whose 256-vs-8192 dictionary swing is {confound_swing} bits"
            );
        }
    }

    #[test]
    fn flat_atom_fast_path_matches_svd_to_tolerance() {
        // A rank-one contribution: scalar codes ⊗ one decoder row — the exact
        // shape a flat (code_dim == 1) atom transmits. The Frobenius fast path
        // must equal the top singular value of the column-centered matrix.
        let codes = array![0.3_f64, -1.2, 2.5, 0.0, 4.1, -0.7];
        let decoder = array![1.5_f64, -0.5, 2.0, 0.25];
        let mut contribution = Array2::<f64>::zeros((codes.len(), decoder.len()));
        for (i, &code) in codes.iter().enumerate() {
            for (j, &weight) in decoder.iter().enumerate() {
                contribution[[i, j]] = code * weight;
            }
        }
        let fast = atom_code_spectrum(contribution.view(), 1).unwrap();
        // Reference: explicit SVD of the column-centered matrix, top value only.
        let centered = column_centered(contribution.view());
        let (_u, singular_values, _vt) = centered.svd(false, false).unwrap();
        let denom = (codes.len() - 1) as f64;
        let svd_spectrum = singular_values[0] * singular_values[0] / denom;
        assert_eq!(fast.len(), 1);
        assert!(
            (fast[0] - svd_spectrum).abs() <= 1.0e-10 * (1.0 + svd_spectrum.abs()),
            "fast {} vs svd {}",
            fast[0],
            svd_spectrum
        );
        // Confirm the centered contribution really is rank one (the assumption
        // the fast path rests on): the second singular value must vanish.
        if singular_values.len() > 1 {
            assert!(
                singular_values[1] <= 1.0e-9 * singular_values[0].max(1.0),
                "flat contribution was not rank-one: {singular_values:?}"
            );
        }
    }

    #[test]
    fn curved_atom_still_uses_full_svd_spectrum() {
        // A rank-two contribution with code_dim == 2 must keep both singular
        // values via the SVD path (the fast path fires only for code_dim == 1).
        let contribution = array![
            [1.0_f64, 0.0, 0.5],
            [0.0, 2.0, 0.5],
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 1.5],
            [3.0, 0.0, 1.5],
        ];
        let spectrum = atom_code_spectrum(contribution.view(), 2).unwrap();
        assert_eq!(spectrum.len(), 2);
        let centered = column_centered(contribution.view());
        let (_u, singular_values, _vt) = centered.svd(false, false).unwrap();
        let denom = (contribution.nrows() - 1) as f64;
        for (k, value) in spectrum.iter().enumerate() {
            assert!((value - singular_values[k] * singular_values[k] / denom).abs() < 1.0e-12);
        }
    }
}

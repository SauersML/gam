//! Uniform fixed-distortion (Eq. 4) description-length scoring of a featurizer.
//!
//! This is the single Rust home for the Eq. 4 scorer that the manifold-zoo
//! benchmark and the #1026 close experiments consume (`bench/bsf_manifold_zoo`,
//! `experiments/1026_close`). It prices ONE fitted featurizer's reconstruction
//! at a stated per-token distortion (fixed R², a matched-EV operating point),
//! decomposing the code length into
//!
//! * **support** bits — the combinatorial support code of the native coder: each
//!   token transmits its own support cardinality and then which subset of that
//!   size fired, `log₂(G+1) + mean_i log₂ C(G, |S_i|)` per token. That is the
//!   WORST CASE among support models (every subset of a given size is equally
//!   likely), but it is a complete prefix code over all `2^G` supports. The
//!   native independent Krichevsky–Trofimov support code is reported alongside it
//!   and never charged, so a comparison whose whole margin lives in the support
//!   term can be read against how predictable the firing is (#2283);
//! * **code** bits — a JOINT reverse-water-filling of every atom's per-firing
//!   contribution spectrum, each spectrum weighted by that atom's firing
//!   probability `p_g`, sharing ONE water level across all components with the
//!   residual, so the fixed total-distortion budget is split optimally between
//!   coding and leaving distortion. The code bits are the rates of each atom's
//!   top `d_g` modes, the modes its declared `d_g` transmitted scalars carry;
//! * **residual** bits — the same joint water level applied to what the residual
//!   coder must carry: the decoded residual's spectrum (weight one) and every
//!   atom's modes BEYOND its declared code dimension (weight `p_g`, reported as
//!   **truncation** bits inside the residual bits);
//! * **dictionary** bits — a declared BIC-INSPIRED AMORTISED PARAMETER PENALTY
//!   `½·(dictionary_params / N)·log₂(N)`, where `N` is the DECLARED
//!   `amortization_horizon`: the number of tokens the stored parameters are
//!   amortised over, NOT the number of rows sampled to estimate the score. It is
//!   a heuristic on a parameter COUNT, not a decoder storage code (#2933 F20; see
//!   [`Eq4DescriptionLength::dictionary_bits`] for its assumptions). The
//!   estimation subsample size (`estimation_rows = test_x.nrows()`) controls ONLY
//!   the Monte-Carlo variance of the support / code / residual expectations; it
//!   must never leak into the dictionary term (#2283 / audit §21). Conflating the
//!   two made the authoritative bits-at-R² row meaningless (the same fitted flat
//!   model priced radically different dictionary bits at 256 vs 8192 estimation
//!   rows), so the two `N`s are now passed separately and the dictionary term
//!   depends on the horizon alone.
//!
//! # Which object this is
//!
//! Every spectral term is a GAUSSIAN rate–distortion SURROGATE over AMBIENT
//! second moments: a linear transform code of each component, not an
//! operational codec and not an intrinsic nonlinear code. Two consequences are
//! load-bearing (#2933 F17, F18):
//!
//! * The receiver is handed no per-component mean, so every component (each
//!   atom's contribution and the residual) is priced at its RAW second moment
//!   `E[v vᵀ]`. A reconstruction bias or a nonzero mean code is paid per token.
//!   The moments are those of the scored batch alone; nothing is estimated on
//!   training data or carried to a held-out batch. Only the R² baseline
//!   (`reference_variance`) is centered, because R² is defined against the
//!   column-mean predictor.
//! * The whole ambient spectrum of each contribution is priced. A curve spans
//!   more ambient directions than its intrinsic dimension (`(cos t, sin t,
//!   cos 2t, sin 2t)` is one-dimensional with four eigenvalues of ½), and a linear
//!   code of `d_g` scalars cannot carry the rest, so the rest is residual-coded
//!   or left as distortion at the shared water level. An intrinsic chart code
//!   that transmits `(t, a)` and decodes the curve through the decoder would
//!   need the chart coordinates and the decoder's pullback metric. This callback
//!   supplies neither, so the scorer does not credit such a code.
//!
//! Unlike [`crate::description_length::reverse_water_filling`] (which water-fills
//! a single unweighted spectrum), the Eq. 4 scorer water-fills a collection of
//! firing-probability-weighted spectra against a shared level via
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
//! firing-row selection, the certified rank-one / SVD contribution spectrum, the
//! residual second-moment eigendecomposition, the water-filling and the bit
//! assembly; the callback ONLY materialises the atom's rows. This keeps peak
//! memory to one atom's contribution at a time (the caller may fetch lazily).

use ndarray::{Array1, Array2, ArrayView2};

use gam_linalg::faer_ndarray::{FaerEigh, FaerSvd};

use crate::atom_codes::{combinatorial_support_bits, kt_code_bits};
use crate::description_length::{
    DescriptionLengthScoreKind, gate_is_transmitted, weighted_reverse_water_filling,
};

/// Standard fixed-distortion reporting points shared by every front-end.
pub const DEFAULT_EQ4_R2_TARGETS: &[f64] = &[0.99, 0.95, 0.90, 0.80];

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
    /// The summed firing-weighted rates of every atom's top `d_g` modes: the
    /// modes its declared `d_g` transmitted scalars carry.
    pub code_bits: f64,
    /// The residual coder's bits at the shared water level: the decoded
    /// residual's raw second-moment spectrum PLUS every atom's modes beyond its
    /// declared code dimension ([`Self::truncation_bits`], already included).
    pub resid_bits: f64,
    /// The part of [`Self::resid_bits`] spent on atom variation beyond the
    /// declared code dimensions (each atom's tail modes, weight `p_g`, coded on
    /// its firing rows). It is a sub-term, never added to the total a second
    /// time. A curved atom declared at `d+1` scalars whose contribution spans more
    /// ambient directions reports that excess here.
    pub truncation_bits: f64,
}

/// The Eq. 4 fixed-distortion description-length report of one featurizer.
#[derive(Clone, Debug)]
pub struct Eq4DescriptionLength {
    /// The CHARGED support price, in bits per token, independent of the distortion
    /// target: the combinatorial support code of the native coder
    /// ([`crate::atom_codes::SupportEntropy::combinatorial_bits`]),
    /// `log₂(G+1) + mean_i log₂ C(G, |S_i|)`. Every row transmits its own
    /// cardinality (uniform over `0..=G`) and then its subset (uniform among the
    /// subsets of that size), so each support has probability
    /// `1 / ((G+1)·C(G, |S|))` and the code is Kraft-complete. It has no learned
    /// parameters, so it never depends on the estimation subsample. It is the worst
    /// case among support models: it gives no credit for predictable firing, and a
    /// fixed-TopK dictionary still pays `log₂(G+1)` for its constant cardinality.
    /// Pricing the rounded MEAN cardinality instead, `log₂ C(G, round L0)`, names
    /// no decodable support: one atom firing on half the rows cost zero bits
    /// (#2933 F09).
    pub support_bits: f64,
    /// The independent Krichevsky–Trofimov support code of the native coder
    /// ([`crate::atom_codes::SupportEntropy::independent_bits`]) over the
    /// estimation rows, in bits per token. Reported alongside
    /// [`Self::support_bits`], never charged. Unlike the plug-in `Σ_g H₂(p̂_g)` it
    /// replaced, it is a decodable code: it pays for learning every atom's firing
    /// rate, about `(½·log₂ n + 1)/n` bits per token per atom, so a never-firing
    /// atom is not free and the value approaches the plug-in entropy only as `n`
    /// grows. That regret is denominated in estimation rows, which is why this
    /// code is reported and never charged.
    pub independent_support_bits: f64,
    /// Achieved mean per-token support cardinality `L0` (mean active atoms per
    /// row). Reported only: the support code prices each row's own cardinality,
    /// never this mean.
    pub achieved_block_l0: f64,
    /// BIC-inspired amortised parameter penalty
    /// `0.5 * dictionary_params / amortization_horizon * log2(amortization_horizon)`,
    /// shared by every target.
    ///
    /// This is a DECLARED HEURISTIC, not a decoder storage code (#2933 F20). It
    /// charges each counted scalar `½·log₂ H` bits, the leading term of a regular
    /// two-part code that quantises an identifiable parameter at precision
    /// `O(H^{-1/2})` for an `H`-token message, and spreads the charge over those
    /// `H` tokens. Its assumptions: every counted scalar is a regular,
    /// identifiable, full-rank parameter, and the `O(1)` Fisher-determinant and
    /// prior terms are dropped. What it does not see: coefficient magnitudes, the
    /// output distortion that quantising them induces through the decoder,
    /// redundancy in the stored representation (a factored and an unfactored
    /// decoder that decode identically declare different counts), and learned
    /// metadata such as knots or topology. A real parameter message
    /// `L(B̃, metadata) / H` with a declared quantiser and output-distortion
    /// allocation is not implemented here.
    ///
    /// Depends ONLY on the declared `amortization_horizon`, never on
    /// `estimation_rows` (#2283): re-estimating the score on a different row
    /// subsample leaves this term bitwise identical.
    pub dictionary_bits: f64,
    /// The number of rows actually used to estimate the code / residual / support
    /// expectations (`test_x.nrows()`). This is the Monte-Carlo estimator size; it
    /// affects only estimator variance and is reported for provenance. It is NOT
    /// the dictionary amortisation horizon (see [`Self::amortization_horizon`]).
    pub estimation_rows: i64,
    /// The declared amortisation horizon `N` of the dictionary penalty: the number
    /// of tokens the stored parameters are amortised over. Echoed through so a
    /// reader can confirm the dictionary term is sample-invariant.
    pub amortization_horizon: i64,
    /// One entry per R² target, in the order the targets were supplied.
    pub per_target: Vec<Eq4TargetBits>,
    /// The featurizer's own native bits/token, echoed through when supplied.
    pub native_bits_per_token: Option<f64>,
    /// Always [`DescriptionLengthScoreKind::GaussianSurrogate`] (#2933 F21): every
    /// code and residual term is a joint weighted reverse-water-filling rate of RAW
    /// second-moment spectra (nothing centered, so means are paid per token) over
    /// each atom's full ambient contribution spectrum — its top `d_g` modes as code
    /// bits, the rest residual-coded as truncation bits — under squared error, with
    /// the components treated as independent Gaussian sources whose distortions add.
    /// It is a linear transform-code surrogate with no intrinsic chart credit, and
    /// the dictionary term is a declared BIC-inspired count penalty, not a codec. No
    /// encoder runs and no reconstruction is measured, so the total is not an
    /// operational message length and compares only with other Gaussian-surrogate
    /// figures.
    pub score_kind: DescriptionLengthScoreKind,
}

/// The eigenvalues of the RAW second-moment matrix `vᵀv / N` of `values`
/// (rows = observations), ascending.
///
/// Nothing is centered. The Eq. 4 receiver is handed no residual mean, so the
/// residual is coded under a zero-mean Gaussian surrogate at its raw second
/// moment and a constant reconstruction bias is paid per token. Centering first
/// priced innovations about a mean that no message transmits: a residual of all
/// fives had spectrum 0 while its squared error is 25 (#2933 F18). No mean is
/// estimated, so there is no Bessel factor, and the scored batch alone defines
/// the moment.
pub(crate) fn second_moment_eigenvalues(values: ArrayView2<f64>) -> Result<Array1<f64>, String> {
    let n = values.nrows().max(1) as f64;
    let mut moment = values.t().dot(&values);
    moment.mapv_inplace(|v| v / n);
    let (eigenvalues, _vectors) = moment
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("residual second-moment eigensolve failed: {e:?}"))?;
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

/// One atom's per-firing raw second-moment spectrum, split at its declared code
/// dimension `d_g`.
struct AtomSpectrum {
    /// The top `d_g` eigenvalues: the modes the atom's `d_g` transmitted scalars
    /// carry. Their rates are the report's code bits.
    coded: Vec<f64>,
    /// Every remaining eigenvalue: variation of the contribution that `d_g`
    /// scalars cannot carry. The residual coder codes it on the atom's firing
    /// rows, so it is water-filled at the atom's weight and priced as truncation
    /// bits inside the residual bits.
    tail: Vec<f64>,
}

/// The per-firing raw second-moment spectrum `σ_i² / rows` of one atom's
/// `(rows, d)` contribution over EVERY singular value, split into the top
/// `code_dim` coded modes and the tail.
///
/// The full spectrum is kept because this is an ambient linear-code surrogate
/// that sees only the contribution matrix: modes beyond the declared scalars
/// must be residual-coded or left as distortion, and the joint water-fill
/// decides which. Keeping only `code_dim` singular values priced a curve's extra
/// harmonics at zero rate AND zero distortion while `recon` still contained
/// them. `code_dim == 0` scored a varying contribution free (#2933 F17). Nothing
/// is centered (#2933 F18): no per-atom mean code is transmitted, so a nonzero
/// mean code is part of the moment.
///
/// Rank-one fast path (#2233). A flat atom transmits one scalar times one decoder
/// row, so its contribution is rank one with the single eigenvalue
/// `‖C‖_F² / rows`, which needs no SVD. That is the dominant cost at large
/// overcompleteness, where an O(rows·d) pass replaces an O(rows·d·min(rows, d))
/// SVD per atom. The rank is CERTIFIED rather than assumed from `code_dim`. Every
/// row is projected onto the largest-norm row `v`. The off-axis energy
/// `Σ_i ‖c_i − (c_i·v / v·v) v‖²` bounds the energy beyond the leading singular
/// direction from above, since no single direction captures more than `σ₁²`.
/// When it lies within the numerical-rank floor `(max(rows, d)·ε)²·‖C‖_F²` (every
/// trailing singular value below the `max(rows, d)·ε·σ₁` rank tolerance), the
/// matrix is rank one to working precision. Otherwise the SVD runs. The old path
/// trusted `code_dim == 1` and put a rank-two contribution's whole trace into
/// one mode.
fn atom_code_spectrum(
    contribution: ArrayView2<f64>,
    code_dim: usize,
) -> Result<AtomSpectrum, String> {
    let rows = contribution.nrows();
    let denom = rows.max(1) as f64;
    let mut frobenius_sq = 0.0_f64;
    let mut pivot = 0usize;
    let mut pivot_norm_sq = 0.0_f64;
    for (row_index, row) in contribution.rows().into_iter().enumerate() {
        let norm_sq = row.dot(&row);
        frobenius_sq += norm_sq;
        if norm_sq > pivot_norm_sq {
            pivot_norm_sq = norm_sq;
            pivot = row_index;
        }
    }
    let spectrum: Vec<f64> = if frobenius_sq == 0.0 {
        Vec::new()
    } else {
        let axis = contribution.row(pivot);
        let mut off_axis_sq = 0.0_f64;
        for row in contribution.rows() {
            let coefficient = row.dot(&axis) / pivot_norm_sq;
            off_axis_sq += row
                .iter()
                .zip(axis.iter())
                .map(|(&value, &along)| {
                    let off = value - coefficient * along;
                    off * off
                })
                .sum::<f64>();
        }
        let rank_tolerance = rows.max(contribution.ncols()) as f64 * f64::EPSILON;
        if off_axis_sq <= rank_tolerance * rank_tolerance * frobenius_sq {
            vec![frobenius_sq / denom]
        } else {
            let (_u, singular_values, _vt) = contribution
                .svd(false, false)
                .map_err(|e| format!("atom contribution SVD failed: {e:?}"))?;
            let mut modes: Vec<f64> = singular_values.iter().map(|&s| s * s / denom).collect();
            modes.sort_by(|left, right| right.total_cmp(left));
            modes
        }
    };
    let split = code_dim.min(spectrum.len());
    let tail = spectrum[split..].to_vec();
    let mut coded = spectrum;
    coded.truncate(split);
    Ok(AtomSpectrum { coded, tail })
}

/// Score `test_x` against a featurizer's reconstruction at each R² target and
/// return the Eq. 4 fixed-distortion description length.
///
/// * `test_x` / `recon` — the held-out activations and the featurizer's
///   reconstruction of them; same shape `(N, d)`, both finite.
/// * `gate` — the `(N, G)` per-atom firing gate; an atom fires on a row exactly
///   when its gate there is nonzero ([`gate_is_transmitted`], the support the
///   native coder prices). A negative gate is a firing: its contribution is in
///   `recon`, so it is paid for in the support code and the spectrum.
/// * `code_dims` — the number `d_g` of scalars each of the `G` atoms transmits
///   per firing (length `G`, nonnegative). The top `d_g` modes of the atom's raw
///   contribution spectrum are priced as code bits, and every further mode as
///   residual-coded truncation bits. `d_g = 0` is valid: all of the atom's
///   variation goes to the residual coder.
/// * `dictionary_params` — the stored decoder scalar COUNT fed to the declared
///   BIC-inspired amortised parameter penalty
///   (`K_flat·P + K_curved·b·P` for the #2283 arms). It is a representation
///   count, neither a quantised storage message nor a free-identifiable/effective
///   dimension: the three coincide only for a full-rank, unpenalised, non-redundant
///   decoder whose coefficients all need the same precision (#2283 / audit §21,
///   #2933 F20).
/// * `amortization_horizon` — the DECLARED `N` of the parameter penalty
///   `0.5·dictionary_params/N·log₂(N)`: the number of tokens the stored
///   parameters are amortised over, which also sets the asymptotic `O(N^{-1/2})`
///   precision the penalty assumes. It is passed SEPARATELY from the estimation
///   subsample (`test_x.nrows()`), and must be at least `2` (an `Err` is
///   returned otherwise — the horizon is never silently defaulted to the
///   estimation subsample, so the #2283 confound cannot recur). The dictionary
///   term depends on this value ALONE.
/// * `r2_targets` — the fixed-distortion R² operating points, each finite and in
///   `[0, 1)`; must be nonempty.
/// * `native_bits_per_token` — echoed onto the report when present.
/// * `fetch_contribution` — a callback returning the `(take.len, d)` contribution
///   matrix of atom `g` restricted to the supplied firing-row indices `take`.
///   Invoked once for every atom that fires at least once, with all of its firing
///   rows, one atom at a time.
///
/// The number of rows of `test_x` / `recon` / `gate` is the `estimation_rows`
/// Monte-Carlo estimator size: it drives ONLY the variance of the support / code
/// / residual expectations, never the dictionary code. The firing-row selection
/// and every numerical term live here; the callback only materialises rows. Every
/// firing atom's spectrum is estimated from ALL of its firing rows, however few or
/// many: there is neither a subsampling stride (#2933 F19) nor a low-count branch
/// that declares a rarely firing atom free (#2933 F16).
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

    // Support: firing count per atom and the histogram of per-row support
    // cardinalities. Both are counts, so the support terms are invariant to row
    // order.
    let mut firings_per_atom = vec![0_usize; n_atoms];
    let mut cardinality_counts = vec![0_usize; n_atoms + 1];
    for row in 0..n {
        let mut cardinality = 0_usize;
        for atom in 0..n_atoms {
            // Any nonzero gate value is a firing, whatever its sign.
            if gate_is_transmitted(gate[[row, atom]]) {
                firings_per_atom[atom] += 1;
                cardinality += 1;
            }
        }
        cardinality_counts[cardinality] += 1;
    }
    let p_g: Vec<f64> = firings_per_atom
        .iter()
        .map(|&count| count as f64 / n as f64)
        .collect();
    let l0 = firings_per_atom.iter().sum::<usize>() as f64 / n as f64;
    // Each row transmits its own cardinality `k ∈ {0,…,G}` uniformly and then its
    // subset uniformly among the `C(G, k)` subsets of that size: the combinatorial
    // support code of the native coder (`SupportEntropy::combinatorial_bits`).
    // Rounding the MEAN cardinality and pricing `log₂ C(G, round L0)` names no
    // decodable support (#2933 F09): one atom firing on half the rows rounds to
    // `C(1, 0)` and costs zero bits, against the one bit per token a receiver needs.
    let support_bits = combinatorial_support_bits(n_atoms, &cardinality_counts);
    // The same support under the native coder's independent Krichevsky–Trofimov
    // code over the estimation rows. Never charged. KT probabilities depend only on
    // each atom's firing count, and the code pays for learning every rate, so a
    // never-firing atom is not free (the plug-in `Σ_g H₂(p̂_g)` it replaces said zero).
    let independent_support_bits = firings_per_atom
        .iter()
        .map(|&fired| kt_code_bits((n - fired) as u64, fired as u64))
        .sum::<f64>()
        / n as f64;

    // Residual raw second-moment spectrum (no residual mean is transmitted, so a
    // bias is paid) and the centered reference variance that defines R².
    let mut residual = test_x.to_owned();
    residual -= &recon;
    let residual_eigenvalues = second_moment_eigenvalues(residual.view())?;
    let centered_x = column_centered(test_x);
    // reference_variance = mean(centered²)·d = Σ centered² / N.
    let reference_variance = centered_x.iter().map(|&v| v * v).sum::<f64>() / n as f64;
    if reference_variance <= 0.0 {
        return Err("test_x must have positive variance".to_string());
    }

    // Per-atom firing-contribution spectra (weight-`p_g` water-fill components).
    let mut code_spectra: Vec<AtomSpectrum> = Vec::with_capacity(n_atoms);
    for atom in 0..n_atoms {
        let code_dim = code_dims[atom] as usize;
        let rows: Vec<usize> = (0..n)
            .filter(|&row| gate_is_transmitted(gate[[row, atom]]))
            .collect();
        if rows.is_empty() {
            // A never-firing atom has weight `p_g = 0`: it transmits nothing and
            // its spectrum never reaches the water-fill.
            code_spectra.push(AtomSpectrum {
                coded: Vec::new(),
                tail: Vec::new(),
            });
            continue;
        }
        // Every firing row enters the spectrum, however few or many. A rare atom
        // is priced from the firings it has: with no mean transmitted, even a
        // single firing's raw second moment is its squared contribution, never
        // the free zero spectrum a low-count branch used to substitute (#2933
        // F16). A deterministic stride over the firing rows is not a representative
        // sample: an ordering periodic in the stride hands the SVD a constant
        // sub-sequence and erases a varying atom's whole spectrum (#2933 F19). The
        // singular values of the full firing matrix are exact and invariant to row
        // order, and the one-atom matrix is bounded by the `(N, d)` residual the
        // scorer already holds.
        let take = rows;
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
    // estimation subsample: the declared BIC-inspired penalty `0.5·params/N·log₂(N)`
    // in the DECLARED amortization horizon `N`, never the `n`-row Monte-Carlo
    // subsample (#2283). A heuristic on a count, not a storage code (#2933 F20).
    let horizon = amortization_horizon as f64;
    let dictionary_bits = 0.5 * dictionary_params as f64 / horizon * horizon.log2();

    let mut per_target = Vec::with_capacity(r2_targets.len());
    for &target in r2_targets {
        let total_distortion = (1.0 - target) * reference_variance;
        // Components: atoms' coded modes, then atoms' tail modes, then the
        // residual. One shared water level prices all of them.
        let mut components: Vec<(f64, Vec<f64>)> = Vec::with_capacity(2 * n_atoms + 1);
        for (&probability, spectrum) in p_g.iter().zip(code_spectra.iter()) {
            components.push((probability, spectrum.coded.clone()));
        }
        for (&probability, spectrum) in p_g.iter().zip(code_spectra.iter()) {
            components.push((probability, spectrum.tail.clone()));
        }
        components.push((1.0, residual_eigenvalues.to_vec()));
        let component_bits = weighted_reverse_water_filling(&components, total_distortion)?;
        let code_bits: f64 = component_bits[..n_atoms].iter().sum();
        let truncation_bits: f64 = component_bits[n_atoms..2 * n_atoms].iter().sum();
        let resid_bits = truncation_bits + component_bits[2 * n_atoms];
        per_target.push(Eq4TargetBits {
            target,
            bits: support_bits + code_bits + resid_bits + dictionary_bits,
            code_bits,
            resid_bits,
            truncation_bits,
        });
    }

    Ok(Eq4DescriptionLength {
        support_bits,
        independent_support_bits,
        achieved_block_l0: l0,
        dictionary_bits,
        estimation_rows: n as i64,
        amortization_horizon,
        per_target,
        native_bits_per_token,
        score_kind: DescriptionLengthScoreKind::GaussianSurrogate,
    })
}

#[cfg(test)]
#[path = "eq4_description_length_support_tests.rs"]
mod support_tests;

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
            move |_, take| {
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
        // The one atom fires on every row, but the combinatorial support code still
        // transmits each row's cardinality among {0, 1}: one bit per token.
        assert_eq!(result.support_bits, 1.0);
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
                |_, take| {
                    let mut selected = Array2::zeros((take.len(), contribution.ncols()));
                    for (out_row, &source_row) in take.iter().enumerate() {
                        selected
                            .row_mut(out_row)
                            .assign(&contribution.row(source_row));
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

    /// Audit §34 "MDL sample invariance": holding the deployment horizon and
    /// fitted featurizer fixed while changing only the estimator row count must
    /// leave dictionary bits bitwise identical. The remaining finite-n drift is
    /// checked against the exact Bessel-correction formula of a balanced,
    /// repeated quadrature block—no stochastic tolerance or resampling loop.
    #[test]
    fn eq4_dictionary_term_is_invariant_to_the_estimation_subsample() {
        const FULL_ROWS: usize = 8192;
        const BLOCK_ROWS: usize = 8;
        let mut test_x = Array2::<f64>::zeros((FULL_ROWS, 2));
        let mut recon = Array2::<f64>::zeros((FULL_ROWS, 2));
        let mut gate = Array2::<f64>::zeros((FULL_ROWS, 1));

        // One honest flat atom: it fires on four rows per block with scalar
        // codes [+1,+1,-1,-1] and decoder [1,0], so every contribution matrix
        // is rigorously rank one. The independent second coordinate is a
        // balanced ±1 residual. Every prefix below contains whole blocks and
        // therefore has exactly the same empirical row distribution:
        // p_fire=1/2, zero means, and reference variance 1/2 + 1 = 3/2.
        for row in 0..FULL_ROWS {
            let within = row % BLOCK_ROWS;
            let code = match within {
                0 | 2 => 1.0,
                4 | 6 => -1.0,
                _ => 0.0,
            };
            let active = within % 2 == 0;
            let residual = if within < 4 { 1.0 } else { -1.0 };
            recon[[row, 0]] = code;
            test_x[[row, 0]] = code;
            test_x[[row, 1]] = residual;
            gate[[row, 0]] = if active { 1.0 } else { 0.0 };
        }

        let horizon = 120_000_i64;
        let dictionary_params = 4096_i64;
        let targets = [0.99, 0.95, 0.90, 0.80];
        let score_at = |rows: usize| -> Eq4DescriptionLength {
            let window = ndarray::s![..rows, ..];
            let contribution = recon.slice(window);
            eq4_fixed_distortion_description_length(
                test_x.slice(window),
                recon.slice(window),
                gate.slice(window),
                &[1],
                dictionary_params,
                horizon,
                &targets,
                None,
                move |_, take| {
                    let mut selected = Array2::zeros((take.len(), contribution.ncols()));
                    for (out_row, &source_row) in take.iter().enumerate() {
                        selected
                            .row_mut(out_row)
                            .assign(&contribution.row(source_row));
                    }
                    Ok(selected)
                },
            )
            .expect("balanced Eq. 4 fixture must score")
        };

        // Exactly three scorer calls. At N=8192 the atom fires 4096 times, and
        // every firing row enters its spectrum.
        let small = score_at(256);
        let medium = score_at(1024);
        let large = score_at(FULL_ROWS);
        for run in [&small, &medium, &large] {
            assert_eq!(run.amortization_horizon, horizon);
            assert_eq!(run.achieved_block_l0, 0.5);
            // One atom firing on exactly half the rows. The combinatorial code
            // transmits each row's cardinality among {0, 1}: exactly one bit per
            // token at every estimation size (the rounded-mean price was zero,
            // #2933 F09). The uncharged independent KT code also pays to learn the
            // firing rate, so it lies strictly above the plug-in entropy of one bit,
            // by the exact KT regret of this count pair.
            assert_eq!(run.support_bits, 1.0);
            let rows = run.estimation_rows as usize;
            let half = rows / 2;
            let kt_log2_probability = 2.0
                * (0..half)
                    .map(|index| (index as f64 + 0.5).log2())
                    .sum::<f64>()
                - (1..=rows).map(|index| (index as f64).log2()).sum::<f64>();
            let expected_kt = -kt_log2_probability / rows as f64;
            assert!(expected_kt > 1.0);
            assert!(
                (run.independent_support_bits - expected_kt).abs() <= 1.0e-10 * expected_kt,
                "{rows} rows: KT support {} != closed form {expected_kt}",
                run.independent_support_bits
            );
        }
        assert_eq!(small.estimation_rows, 256);
        assert_eq!(medium.estimation_rows, 1024);
        assert_eq!(large.estimation_rows, FULL_ROWS as i64);

        // Load-bearing #2283 contract: dictionary storage is priced in the
        // declared deployment horizon, not in the estimator sample size.
        let expected_dictionary =
            0.5 * dictionary_params as f64 / horizon as f64 * (horizon as f64).log2();
        assert_eq!(small.dictionary_bits, expected_dictionary);
        assert_eq!(medium.dictionary_bits, expected_dictionary);
        assert_eq!(large.dictionary_bits, expected_dictionary);

        // Raw second moments estimate no mean, so there is no Bessel factor and
        // no finite-n drift at all (#2933 F18). Every prefix of whole blocks has
        // exactly the same empirical row distribution (codes ±1 on half the rows,
        // residual ±1), and every sum is an exact integer before the division by a
        // power of two, so each spectrum and therefore each total is bitwise
        // identical across the three row counts. The centered estimator drifted
        // by `¼·log₂(a_c(n)/a_c(N)) + ½·log₂(a_r(n)/a_r(N))` with
        // `a_c(n) = (n/2)/(n/2−1)`, `a_r(n) = n/(n−1)`.
        for (target_index, &target) in targets.iter().enumerate() {
            let full = large.per_target[target_index];
            assert!(
                full.code_bits > 0.0 && full.resid_bits > 0.0,
                "target {target}: both spectral terms must be priced, got {full:?}"
            );
            for run in [&small, &medium] {
                assert_eq!(
                    run.per_target[target_index].bits, full.bits,
                    "target {target}: {}-row total must equal the {FULL_ROWS}-row total",
                    run.estimation_rows
                );
            }
        }

        // Anti-vacuity: the pre-#2283 estimator-sized dictionary charge moved
        // by exactly 60.75 bits between these row counts.
        let legacy_dictionary =
            |rows: usize| 0.5 * dictionary_params as f64 * (rows as f64).log2() / rows as f64;
        let legacy_swing = (legacy_dictionary(256) - legacy_dictionary(FULL_ROWS)).abs();
        assert!((legacy_swing - 60.75).abs() < 1.0e-12);
    }

    #[test]
    fn flat_atom_fast_path_matches_svd_to_tolerance() {
        // A rank-one contribution: scalar codes ⊗ one decoder row — the exact
        // shape a flat atom transmits. The certified fast path must return the
        // closed-form raw moment `Σ s² · ‖w‖² / rows` and leave no tail.
        let codes = array![0.3_f64, -1.2, 2.5, 0.0, 4.1, -0.7];
        let decoder = array![1.5_f64, -0.5, 2.0, 0.25];
        let mut contribution = Array2::<f64>::zeros((codes.len(), decoder.len()));
        for (i, &code) in codes.iter().enumerate() {
            for (j, &weight) in decoder.iter().enumerate() {
                contribution[[i, j]] = code * weight;
            }
        }
        let fast = atom_code_spectrum(contribution.view(), 1).unwrap();
        let closed_form = codes.iter().map(|c| c * c).sum::<f64>()
            * decoder.iter().map(|w| w * w).sum::<f64>()
            / codes.len() as f64;
        assert_eq!(fast.coded.len(), 1);
        assert!(fast.tail.is_empty(), "rank-one tail must be empty: {:?}", fast.tail);
        assert!(
            (fast.coded[0] - closed_form).abs() <= 1.0e-12 * closed_form,
            "fast {} vs closed form {closed_form}",
            fast.coded[0]
        );
        // Confirm the raw contribution really is rank one (what the certificate
        // accepted): the second singular value must vanish.
        let (_u, singular_values, _vt) = contribution.svd(false, false).unwrap();
        if singular_values.len() > 1 {
            assert!(
                singular_values[1] <= 1.0e-9 * singular_values[0].max(1.0),
                "flat contribution was not rank-one: {singular_values:?}"
            );
        }
    }

    #[test]
    fn curved_atom_still_uses_full_svd_spectrum() {
        // A rank-three contribution declared at code_dim == 2: the SVD path keeps
        // the top two raw modes as coded and the third as the residual-coded tail.
        let contribution = array![
            [1.0_f64, 0.0, 0.5],
            [0.0, 2.0, 0.5],
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 1.5],
            [3.0, 0.0, 1.5],
        ];
        let spectrum = atom_code_spectrum(contribution.view(), 2).unwrap();
        assert_eq!(spectrum.coded.len(), 2);
        assert_eq!(spectrum.tail.len(), 1);
        let (_u, singular_values, _vt) = contribution.svd(false, false).unwrap();
        let denom = contribution.nrows() as f64;
        let mut reference: Vec<f64> = singular_values.iter().map(|s| s * s / denom).collect();
        reference.sort_by(|left, right| right.total_cmp(left));
        assert!(reference[2] > 1.0e-3, "fixture must have a genuine third mode");
        for (value, expected) in spectrum.coded.iter().chain(spectrum.tail.iter()).zip(&reference) {
            assert!((value - expected).abs() < 1.0e-12);
        }
    }

    /// A `fetch_contribution` callback serving one fixed contribution matrix per atom.
    fn serve_rows(
        contributions: Vec<Array2<f64>>,
    ) -> impl FnMut(usize, &[usize]) -> Result<Array2<f64>, String> {
        move |atom, take| {
            let source = &contributions[atom];
            let mut selected = Array2::zeros((take.len(), source.ncols()));
            for (out_row, &source_row) in take.iter().enumerate() {
                selected.row_mut(out_row).assign(&source.row(source_row));
            }
            Ok(selected)
        }
    }

    /// `values` padded with two zero columns and reflected by a fixed Householder
    /// matrix: the same point cloud in a rotated, larger ambient space.
    fn householder_embed(values: &Array2<f64>) -> Array2<f64> {
        let width = values.ncols() + 2;
        let normal: Array1<f64> = (1..=width).map(|j| j as f64).collect();
        let scale = 2.0 / normal.dot(&normal);
        let mut embedded = Array2::zeros((values.nrows(), width));
        for (mut out, row) in embedded.rows_mut().into_iter().zip(values.rows()) {
            let mut padded = Array1::<f64>::zeros(width);
            padded.slice_mut(ndarray::s![..values.ncols()]).assign(&row);
            let along = scale * padded.dot(&normal);
            out.assign(&(&padded - &(along * &normal)));
        }
        embedded
    }

    /// Harmonic features of order `order` on a uniform phase grid of `grid`
    /// angles per axis: `(cos kt, sin kt)` for `k = 1..order` on a curve, and also
    /// `(cos ks, sin ks)` on a torus-grid surface. Discrete orthogonality
    /// (`2·order < grid`) makes every raw moment eigenvalue exactly ½.
    fn harmonic_features(order: usize, grid: usize, surface: bool) -> Array2<f64> {
        let axes = if surface { 2 } else { 1 };
        let rows = grid.pow(axes as u32);
        Array2::from_shape_fn((rows, 2 * axes * order), |(i, j)| {
            let axis = j / (2 * order);
            let step = if axis == 0 { i % grid } else { i / grid };
            let phase = std::f64::consts::TAU * step as f64 / grid as f64;
            let k = ((j % (2 * order)) / 2 + 1) as f64;
            if j % 2 == 0 {
                (k * phase).cos()
            } else {
                (k * phase).sin()
            }
        })
    }

    /// #2933 F17: a curve or surface built from harmonics of order `H` has
    /// intrinsic dimension 1 or 2, but its contribution has `2H` or `4H` raw
    /// eigenvalues, each exactly ½. The atom declares `d+1` transmitted scalars
    /// (chart coordinates plus amplitude), `recon` contains every harmonic, and
    /// the residual is zero, so the variation the declared scalars cannot carry
    /// must be coded or paid as distortion. At fixed R² the budget is
    /// `(1−R²)·modes/2` and every mode sits above the level `(1−R²)/2`, so the
    /// total is `(modes/2)·log₂(1/(1−R²))`, linear in the harmonic order and the
    /// same for an equivalent rotated embedding. The truncating scorer kept `d+1`
    /// modes, and its bits FELL as the order rose (curve H=2 at R²=0.9: `log₂5 ≈
    /// 2.32` against `2·log₂10 ≈ 6.64`).
    #[test]
    fn eq4_harmonic_curves_and_surfaces_pay_for_every_harmonic() {
        let targets = [0.9, 0.99];
        for order in 1..=4_usize {
            for (label, surface, intrinsic_dim) in [("curve", false, 1_i64), ("surface", true, 2)]
            {
                let features = harmonic_features(order, 16, surface);
                let modes = features.ncols() as f64;
                let gate = Array2::ones((features.nrows(), 1));
                for (embedding, values) in [
                    ("axis-aligned", features.clone()),
                    ("rotated", householder_embed(&features)),
                ] {
                    let report = eq4_fixed_distortion_description_length(
                        values.view(),
                        values.view(),
                        gate.view(),
                        &[intrinsic_dim + 1],
                        0,
                        FIXTURE_HORIZON,
                        &targets,
                        None,
                        serve_rows(vec![values.clone()]),
                    )
                    .expect("harmonic fixture must score");
                    for (row, &target) in report.per_target.iter().zip(targets.iter()) {
                        // One always-firing atom: the complete support code sends one
                        // cardinality bit per token (#2933 F09).
                        let expected = 1.0 + 0.5 * modes * (1.0 / (1.0 - target)).log2();
                        assert!(
                            (row.bits - expected).abs() <= 1.0e-10 * expected,
                            "{label} order {order} ({embedding}) at R²={target}: \
                             {} bits, every harmonic costs {expected}",
                            row.bits
                        );
                    }
                }
            }
        }
    }

    /// #2933 F17 callback contracts, plus the atom half of F18.
    /// (a) `d_g = 0`: an atom that transmits no scalar but whose contribution
    /// varies (rank one, `λ = 1`) must pay for that variation through the
    /// residual coder. The truncating scorer scored it free: 0 bits against
    /// `½·log₂(1/(1−R²))`.
    /// (b) `d_g = 1` handed a rank-two contribution `(cos t, sin t)` with
    /// eigenvalues ½ and ½. The uncertified fast path put the whole trace into
    /// ONE mode, `½·log₂(1/(1−R²))`, where the two modes cost `log₂(1/(1−R²))`.
    /// (c) A genuine rank-one flat atom with a NONZERO mean code `s = 2 + cos t`
    /// is priced at its raw moment `E[s²]‖w‖² = 4.5‖w‖²`, since no per-atom mean
    /// is transmitted. The centered scorer charged `Var(s)‖w‖² = 0.5‖w‖²`.
    #[test]
    fn eq4_zero_and_rank_one_code_dims_price_the_contribution_they_are_handed() {
        let rows = 64;
        let target = 0.9;
        let phase = |i: usize| std::f64::consts::TAU * i as f64 / rows as f64;
        let gate = Array2::ones((rows, 1));
        let score = |values: &Array2<f64>, code_dim: i64| -> f64 {
            let report = eq4_fixed_distortion_description_length(
                values.view(),
                values.view(),
                gate.view(),
                &[code_dim],
                0,
                FIXTURE_HORIZON,
                &[target],
                None,
                serve_rows(vec![values.clone()]),
            )
            .expect("contract fixture must score");
            // One always-firing atom: the complete support code sends one
            // cardinality bit per token (#2933 F09). The spectral terms are the rest.
            assert_eq!(report.support_bits, 1.0);
            report.per_target[0].bits - report.support_bits
        };
        let log_inverse_budget = (1.0 / (1.0 - target)).log2();

        let rank_one = Array2::from_shape_fn((rows, 2), |(i, j)| {
            std::f64::consts::SQRT_2 * phase(i).cos() * [0.6, 0.8][j]
        });
        let zero_dim = score(&rank_one, 0);
        let expected_zero_dim = 0.5 * log_inverse_budget;
        assert!(
            (zero_dim - expected_zero_dim).abs() <= 1.0e-10 * expected_zero_dim,
            "code_dim 0 atom: {zero_dim} bits, its variation costs {expected_zero_dim}"
        );

        let rank_two = Array2::from_shape_fn((rows, 2), |(i, j)| {
            if j == 0 {
                phase(i).cos()
            } else {
                phase(i).sin()
            }
        });
        let one_dim = score(&rank_two, 1);
        assert!(
            (one_dim - log_inverse_budget).abs() <= 1.0e-10 * log_inverse_budget,
            "code_dim 1 atom handed rank two: {one_dim} bits, its two modes cost \
             {log_inverse_budget}"
        );

        let decoder = [1.5_f64, -0.5, 2.0, 0.25];
        let mean_code = Array2::from_shape_fn((rows, decoder.len()), |(i, j)| {
            (2.0 + phase(i).cos()) * decoder[j]
        });
        let raw = score(&mean_code, 1);
        // λ = 4.5‖w‖², budget (1−R²)·0.5‖w‖²: the ratio is 90 at R²=0.9.
        let expected_raw = 0.5 * (4.5 / ((1.0 - target) * 0.5)).log2();
        assert!(
            (raw - expected_raw).abs() <= 1.0e-10 * expected_raw,
            "mean-code flat atom: {raw} bits, its raw moment costs {expected_raw}"
        );
    }

    /// #2933 F18: a reconstruction bias must be paid, not centered away. One
    /// rank-one atom carries `s = √2 cos t` on axis 0 (`λ = 1`). Axis 1 holds a
    /// zero-mean innovation `u = √2 sin t` that the featurizer leaves in the
    /// residual. Shifting `recon` by a constant `β` on axis 1 leaves every
    /// centered moment unchanged, but the decoded squared error rises by `β²`.
    /// The residual's raw moment is `diag(0, 1 + β²)`, the budget `(1−R²)·2`, and
    /// both positive modes sit above the level `1−R²`, so the total is
    /// `½·log₂(1/(1−R²)) + ½·log₂((1+β²)/(1−R²))`. The centered scorer returned
    /// the β = 0 price for every β. The audit's own case, a residual of all fives
    /// with no innovation, prices the mode 25, not 0.
    #[test]
    fn eq4_reconstruction_bias_is_paid_not_centered_away() {
        let rows = 64;
        let target = 0.9;
        let budget = 1.0 - target;
        let phase = |i: usize| std::f64::consts::TAU * i as f64 / rows as f64;
        let code = |i: usize| std::f64::consts::SQRT_2 * phase(i).cos();
        let atom = Array2::from_shape_fn((rows, 2), |(i, j)| if j == 0 { code(i) } else { 0.0 });
        let gate = Array2::ones((rows, 1));
        let score = |test_x: &Array2<f64>, recon: &Array2<f64>| -> f64 {
            let report = eq4_fixed_distortion_description_length(
                test_x.view(),
                recon.view(),
                gate.view(),
                &[1],
                0,
                FIXTURE_HORIZON,
                &[target],
                None,
                serve_rows(vec![atom.clone()]),
            )
            .expect("bias fixture must score");
            // One always-firing atom: the complete support code sends one
            // cardinality bit per token (#2933 F09). The spectral terms are the rest.
            assert_eq!(report.support_bits, 1.0);
            report.per_target[0].bits - report.support_bits
        };

        let test_x = Array2::from_shape_fn((rows, 2), |(i, j)| {
            if j == 0 {
                code(i)
            } else {
                std::f64::consts::SQRT_2 * phase(i).sin()
            }
        });
        for bias in [0.0_f64, 1.0, 5.0] {
            let recon =
                Array2::from_shape_fn((rows, 2), |(i, j)| if j == 0 { code(i) } else { bias });
            let bits = score(&test_x, &recon);
            let expected = 0.5 * (1.0 / budget).log2() + 0.5 * ((1.0 + bias * bias) / budget).log2();
            assert!(
                (bits - expected).abs() <= 1.0e-10 * expected,
                "bias {bias}: {bits} bits, the biased residual costs {expected}"
            );
        }

        // Audit check 8: the residual is all fives on axis 1 and there is no
        // innovation. Reference variance 1, budget 0.1, modes {1, 25}, level 0.05.
        let flat_x = atom.clone();
        let fives = Array2::from_shape_fn((rows, 2), |(i, j)| if j == 0 { code(i) } else { -5.0 });
        let bits = score(&flat_x, &fives);
        let level = budget / 2.0;
        let expected = 0.5 * (1.0 / level).log2() + 0.5 * (25.0 / level).log2();
        assert!(
            (bits - expected).abs() <= 1.0e-10 * expected,
            "all-fives residual: {bits} bits, its mode 25 costs {expected}"
        );
    }

    /// #2933 F17 ledger split. The order-3 harmonic curve declared at 2 scalars
    /// has six modes of ½ and zero residual. Its top two modes are code bits,
    /// `log₂(1/(1−R²))`. The four modes the scalars cannot carry are
    /// residual-coded truncation bits, `2·log₂(1/(1−R²))`, all of `resid_bits`.
    /// The total still reconciles to support + code + resid + dictionary.
    #[test]
    fn eq4_reports_modes_beyond_the_code_dims_as_residual_coded_truncation_bits() {
        let target = 0.9;
        let features = harmonic_features(3, 16, false);
        let gate = Array2::ones((features.nrows(), 1));
        let report = eq4_fixed_distortion_description_length(
            features.view(),
            features.view(),
            gate.view(),
            &[2],
            32,
            FIXTURE_HORIZON,
            &[target],
            None,
            serve_rows(vec![features.clone()]),
        )
        .expect("split fixture must score");
        let row = report.per_target[0];
        let per_pair = (1.0 / (1.0 - target)).log2();
        let close = |got: f64, want: f64| (got - want).abs() <= 1.0e-10 * want.max(1.0);
        assert!(close(row.code_bits, per_pair), "code bits {row:?}");
        assert!(close(row.truncation_bits, 2.0 * per_pair), "truncation bits {row:?}");
        assert!(close(row.resid_bits, row.truncation_bits), "resid bits {row:?}");
        assert!(
            (row.bits
                - (report.support_bits + row.code_bits + row.resid_bits + report.dictionary_bits))
                .abs()
                <= 1.0e-12 * row.bits
        );
    }

    /// #2933 F20 contract pin (a declared limitation, not a measurement). The
    /// dictionary term is a BIC-inspired amortised penalty on a COUNT. Two
    /// featurizers that decode identically, one declaring an unfactored `r×d`
    /// decoder (`4×16 = 64` scalars) and one a factored `r×q, q×d` decoder
    /// (`4×2 + 2×16 = 40`), get different penalties while their support, code and
    /// residual terms agree bitwise. The penalty is `½·count·log₂H/H` and scales
    /// only with the declared count and horizon.
    #[test]
    fn eq4_dictionary_penalty_depends_on_the_declared_count_not_the_decoded_model() {
        let features = harmonic_features(2, 16, false);
        let gate = Array2::ones((features.nrows(), 1));
        let score = |count: i64, horizon: i64| {
            eq4_fixed_distortion_description_length(
                features.view(),
                features.view(),
                gate.view(),
                &[2],
                count,
                horizon,
                &[0.9],
                None,
                serve_rows(vec![features.clone()]),
            )
            .expect("penalty fixture must score")
        };
        let unfactored = score(64, FIXTURE_HORIZON);
        let factored = score(40, FIXTURE_HORIZON);
        let longer = score(64, 4 * FIXTURE_HORIZON);
        let penalty = |count: f64, horizon: f64| 0.5 * count / horizon * horizon.log2();
        assert_eq!(unfactored.dictionary_bits, penalty(64.0, FIXTURE_HORIZON as f64));
        assert_eq!(factored.dictionary_bits, penalty(40.0, FIXTURE_HORIZON as f64));
        assert_eq!(longer.dictionary_bits, penalty(64.0, 4.0 * FIXTURE_HORIZON as f64));
        assert!(unfactored.dictionary_bits > factored.dictionary_bits);
        for other in [&factored, &longer] {
            assert_eq!(other.support_bits, unfactored.support_bits);
            assert_eq!(other.per_target[0].code_bits, unfactored.per_target[0].code_bits);
            assert_eq!(other.per_target[0].resid_bits, unfactored.per_target[0].resid_bits);
        }
    }
}

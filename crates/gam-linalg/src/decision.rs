//! Decision-currency primitives (theory master design §9-step-6, issue #2337).
//!
//! A "decision" here is a discrete claim extracted from floating-point data —
//! the numerical rank of a design, whether a Newton step is stationary, whether
//! a reduction is real or roundoff. The theme of this module is that such a
//! claim is trustworthy only when it is expressed in a *currency* that is
//! invariant under the symmetries of the problem and is decided with a margin
//! wider than the backward error committed forming the quantity we decide on.
//! Each item below carries its derivation as a doc comment.
//!
//! The currencies (the list is not numbered on purpose — it grows, and a count
//! in this header has already gone stale once):
//!   * [`equilibrate_gram`] — a Gram/rank decision must be gauge-invariant under
//!     positive per-column rescaling; equilibration puts it in that gauge.
//!   * [`certified_rank`] — a rank claim with a two-sided multiplicative gap is
//!     stable (bitwise-reproducible) under perturbations below the band width.
//!   * [`rank_transport_radius`] / [`transport_certified_rank`] — a rank claim
//!     taken at one operating point is a claim about a NEIGHBOURHOOD of it,
//!     whose certified radius is read straight off the gap; a decision whose
//!     operating point moves is reused only while the realized excursion stays
//!     inside that radius (§8 Thm 8.3).
//!   * [`projector_error_bar`] — a claim about the SUBSPACE realizing a rank is
//!     a different claim from the rank, conditioned by a different quantity (the
//!     eigen-GAP, not the distance to the cutoff). Certifying the integer does
//!     not certify the eigenspace, and a subspace comparison run below this bar
//!     is deciding on roundoff (#2448).
//!   * [`newton_decrement_enclosure`] — the Newton decrement λ_N² is the
//!     affine-invariant stationarity currency; an inexact solve still yields a
//!     rigorous two-sided enclosure of it.
//!   * [`ShadowSum`] — a reduction carries its own rounding floor, so "is this
//!     decrement real?" is decided against a certified error bar.

use ndarray::{Array1, Array2};

/// Diagonally equilibrate a symmetric Gram matrix into its column-scale gauge.
///
/// Returns `(C, s)` with `C = D^{-1/2} · G · D^{-1/2}`, `D = diag(G)`, and the
/// per-column scale vector `s_j = sqrt(G_jj)`. A column with `G_jj ≤ 0` (a null
/// or numerically empty direction) is given unit scale `s_j = 1`, leaving its
/// row/column of `C` unchanged.
///
/// # Why this is the right currency for a rank decision
///
/// **Congruence preserves the decision (Sylvester's law of inertia).** With
/// `Δ = D^{-1/2} = diag(1/s_j) ≻ 0`, `C = ΔGΔ` is a *congruence* of `G`. By
/// Sylvester's law of inertia a congruence `C = MᵀGM` with `M` nonsingular
/// preserves the inertia `(n₊, n₀, n₋)` of `G`, hence its rank and the sign of
/// every eigenvalue. Deciding the rank of `C` therefore decides the rank of
/// `G` exactly — equilibration changes the conditioning, never the answer.
///
/// **Gauge invariance.** For any positive diagonal `Λ = diag(λ_j) ≻ 0`, the map
/// `G ↦ ΛGΛ` sends `D ↦ Λ D Λ`, so `s_j ↦ λ_j s_j` and
/// `(ΛGΛ)_{ij} / (λ_i s_i · λ_j s_j) = G_{ij}/(s_i s_j) = C_{ij}`. Thus `C` is
/// *invariant* under the column-scale gauge `G ↦ ΛGΛ`: it is the canonical
/// representative of `G`'s congruence orbit under positive diagonal scaling.
/// A rank test on `C` cannot be fooled by a single stiff column.
///
/// **Near-optimal conditioning (van der Sluis).** Among all positive diagonal
/// scalings `Δ`, equilibrating so that `diag(ΔGΔ)` is constant (here, unit) is
/// within a factor of `p` of the minimum achievable spectral condition number:
/// `κ(D^{-1/2}GD^{-1/2}) ≤ p · min_{Δ≻0 diagonal} κ(ΔGΔ)` for a `p×p` SPD `G`
/// (van der Sluis, 1969). So this cheap choice is provably close to the best a
/// diagonal preconditioner can do, and the residual decision then reads the
/// true correlation structure rather than the column magnitudes.
pub fn equilibrate_gram(g: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let p = g.nrows();
    let scale: Array1<f64> = Array1::from_shape_fn(p, |j| {
        let d = g[[j, j]];
        if d > 0.0 { d.sqrt() } else { 1.0 }
    });
    let mut c = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            c[[i, j]] = g[[i, j]] / (scale[i] * scale[j]);
        }
    }
    (c, scale)
}

/// Outcome of a certified numerical-rank decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RankDecision {
    /// The rank is certified: every kept singular value clears the upper band
    /// edge and every dropped one falls below the lower band edge.
    Certified {
        /// Certified numerical rank `r`.
        rank: usize,
        /// Smallest kept singular value `σ_r` (`+∞` when `rank == 0`).
        sigma_r: f64,
        /// Largest dropped singular value `σ_{r+1}` (`0` when `rank == n`).
        sigma_next: f64,
        /// Number of singular values the decision was taken on, `n`.
        ///
        /// Retained because `sigma_next == 0` is OVERLOADED: it means "no
        /// dropped value exists" when `rank == n`, and "a dropped value that
        /// happens to be exactly zero" when `rank < n`. Those two carry
        /// different transport constraints (see [`rank_transport_radius`]) and
        /// nothing else on the certificate tells them apart.
        spectrum_len: usize,
        /// Multiplicative slack of the dropped side below the lower edge,
        /// `low / σ_{r+1}` (`+∞` when `σ_{r+1} = 0`); `≥ 1` by construction.
        margin_low: f64,
        /// Multiplicative slack of the kept side above the upper edge,
        /// `σ_r / high` (`+∞` when `rank == 0`); `≥ 1` by construction.
        margin_high: f64,
        /// Tolerance the decision was posed at — retained so the band edges
        /// `high = tol·(1+gap)` / `low = tol/(1+gap)`, and hence the transport
        /// radius of [`rank_transport_radius`], are recoverable from the
        /// certificate alone.
        tol: f64,
        /// Multiplicative half-gap the decision was posed with.
        gap: f64,
    },
    /// The rank is undecidable at this tolerance: a singular value lands inside
    /// the open guard band `(tol/(1+gap), tol·(1+gap))`.
    Ambiguous {
        /// Rank if the in-band value is treated as dropped: `#{σ ≥ high}`.
        rank_floor: usize,
        /// Rank if the in-band value is treated as kept: `#{σ > low}`.
        rank_ceil: usize,
        /// The offending singular value sitting inside the band.
        sigma_in_band: f64,
        /// Tolerance the decision was posed at.
        tol: f64,
        /// Multiplicative half-gap the decision was posed with.
        gap: f64,
    },
}

/// Certify the numerical rank of a spectrum against a two-sided guard band.
///
/// The rank `r` is [`Certified`](RankDecision::Certified) iff
/// `σ_r ≥ tol·(1+gap)` **and** `σ_{r+1} ≤ tol/(1+gap)` (with `σ_{n+1} := 0`);
/// otherwise the outcome is [`Ambiguous`](RankDecision::Ambiguous), naming the
/// value that fell inside the band. Inputs need not be sorted; they are ordered
/// descending internally.
///
/// # Why the two-sided gap is the decision's currency
///
/// **Perturbation invariance ⇒ host stability.** Write the band edges as
/// `high = tol·(1+gap)`, `low = tol/(1+gap)`. A Certified decision keeps every
/// `σ ≥ high` and drops every `σ ≤ low`; the open interval `(low, high)` is
/// empty of data. Any perturbation `|Δσ_i|` strictly smaller than the distance
/// from each `σ_i` to the nearer band edge leaves the partition — and hence the
/// integer `r` — unchanged. The decision is therefore a locally constant
/// function of the spectrum: identical (bitwise) integer outputs for any inputs
/// agreeing to within the margins. Given reproducible inputs it is
/// host-stable. An `Ambiguous` outcome is the honest report that no such
/// margin exists, so the integer would be host-dependent.
///
/// **Decide in the design's currency, not its square.** Forming the raw Gram
/// `G = XᵀX` and deciding on its eigenvalues squares the condition number:
/// `κ(G) = κ(X)²`, and the eigenvalues are computed with backward error
/// `O(u · σ_max(X)²)` (u the unit roundoff) — the decision inherits an error
/// bar quadratic in the design's scale. Deciding on the *equilibrated* design
/// or its equilibrated Gram (see [`equilibrate_gram`]) commits only
/// `O(u · σ_max)` in the decision's own linear currency, so a gap of a few `u`
/// suffices to certify. This is why the caller equilibrates first, then
/// certifies.
pub fn certified_rank(singular_values: &[f64], tol: f64, gap: f64) -> RankDecision {
    let mut sv: Vec<f64> = singular_values.to_vec();
    sv.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let n = sv.len();
    let high = tol * (1.0 + gap);
    let low = tol / (1.0 + gap);

    // A singular value strictly inside the open band makes `r` undecidable.
    if let Some(&sigma_in_band) = sv.iter().find(|&&s| s > low && s < high) {
        let rank_floor = sv.iter().filter(|&&s| s >= high).count();
        let rank_ceil = sv.iter().filter(|&&s| s > low).count();
        return RankDecision::Ambiguous {
            rank_floor,
            rank_ceil,
            sigma_in_band,
            tol,
            gap,
        };
    }

    // Clean split: everything is either `≥ high` (kept) or `≤ low` (dropped).
    let rank = sv.iter().filter(|&&s| s >= high).count();
    let sigma_r = if rank == 0 {
        f64::INFINITY
    } else {
        sv[rank - 1]
    };
    let sigma_next = if rank < n { sv[rank] } else { 0.0 };
    let margin_high = if rank == 0 {
        f64::INFINITY
    } else {
        sigma_r / high
    };
    let margin_low = if sigma_next == 0.0 {
        f64::INFINITY
    } else {
        low / sigma_next
    };
    RankDecision::Certified {
        rank,
        sigma_r,
        sigma_next,
        spectrum_len: n,
        margin_low,
        margin_high,
        tol,
        gap,
    }
}

/// Verdict of a path-gap rank transport test (#2337 §8, Thm 8.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RankTransport {
    /// The reference certificate carries: every operator within the queried
    /// excursion of the reference decides the SAME rank at the same `(tol,
    /// gap)`, so a re-decision is provably redundant.
    Transported {
        /// Rank the reference certified, valid at the queried point too.
        rank: usize,
        /// Certified transport radius of the reference decision.
        radius: f64,
        /// Excursion the caller certified the operator to stay inside.
        excursion: f64,
        /// Unused radius `radius − excursion ≥ 0` — how much further the
        /// operating point may travel before the certificate must be renewed.
        slack: f64,
    },
    /// The excursion has exhausted the reference's certified gap: the rank at
    /// the queried point is no longer implied and MUST be re-decided there.
    /// This is a statement about the certificate, not about the rank — the
    /// rank may well be unchanged, but nothing here proves it.
    GapExhausted {
        /// Rank the reference certified (no longer implied at the query).
        rank: usize,
        /// Certified transport radius of the reference decision.
        radius: f64,
        /// Excursion that exhausted it.
        excursion: f64,
    },
    /// Nothing to transport: the reference decision was itself
    /// [`Ambiguous`](RankDecision::Ambiguous), or the excursion was not a
    /// finite non-negative number.
    NoCertificate,
}

/// Transport radius of a certified rank decision — the largest operator-norm
/// perturbation that provably cannot change the decision.
///
/// Returns `None` for an [`Ambiguous`](RankDecision::Ambiguous) reference (an
/// undecided rank has no margin to transport).
///
/// # Derivation (#2337 §8, Thm 8.3)
///
/// Let `A₀` be the operator the reference decision was taken on, with singular
/// values `σ₁ ≥ … ≥ σ_n` sorted descending, and let the decision be
/// `Certified{rank r}` against the band edges `high = tol·(1+gap)` and
/// `low = tol/(1+gap)`. By the Certified predicate `σ_r ≥ high` and
/// `σ_{r+1} ≤ low` (with `σ_{n+1} := 0`, and `σ_r := +∞` when `r = 0`). Define
///
/// ```text
///   ε* = min( σ_r − high , low − σ_{r+1} )     (both terms ≥ 0)
/// ```
///
/// **Either half can be VACUOUS, and then its term is `+∞`, not a number.**
/// The two conjuncts of the Certified predicate quantify over `i ≤ r` and
/// `i > r`, and either index set can be empty:
///
/// * `r = 0` — no kept value. `σ_r := +∞` is a sentinel, the `i ≤ r` half is
///   vacuous, and the dropped side alone binds.
/// * `r = n` — no dropped value. `σ_{n+1} := 0` is likewise a sentinel, the
///   `i > r` half is vacuous, and the KEPT side alone binds: `ε* = σ_n − high`.
///   A perturbation cannot manufacture an `(n+1)`-th singular direction, so
///   there is nothing on that side to push into the band. Reading the sentinel
///   as a real dropped value would return `min(σ_n − high, low)` — sound, but
///   not the SHARP radius this function documents, and a source of avoidable
///   certificate renewal: for `σ = [10, 9]` at `tol = 1`, `gap = 1` the sharp
///   radius is `9 − 2 = 7`, while the sentinel reading caps it at `low = 0.5`.
///
/// The `r = n` case is NOT detectable from `σ_{r+1} == 0`. When `r < n`, a
/// dropped singular value that happens to be exactly zero is REAL and does
/// constrain: a perturbation of norm `ε` can lift it to `ε`, and it must stay
/// `≤ low`. The certificate carries `spectrum_len` for exactly this
/// distinction.
///
/// **Claim.** For every `A` with `‖A − A₀‖₂ ≤ ε*`, `certified_rank` on `A`'s
/// spectrum at the same `(tol, gap)` returns `Certified` with the SAME `r`.
///
/// **Proof.** Weyl's inequality for singular values gives
/// `|σ_i(A) − σ_i(A₀)| ≤ ‖A − A₀‖₂ ≤ ε*` for every `i` simultaneously. Hence
/// for `i ≤ r`: `σ_i(A) ≥ σ_i(A₀) − ε* ≥ σ_r(A₀) − ε* ≥ high`, using the
/// descending order and `ε* ≤ σ_r − high`. And for `i > r`:
/// `σ_i(A) ≤ σ_i(A₀) + ε* ≤ σ_{r+1}(A₀) + ε* ≤ low`, using `ε* ≤ low −
/// σ_{r+1}`. So the first `r` values sit at or above `high`, the rest at or
/// below `low`, the open band `(low, high)` is empty — the Ambiguous branch
/// cannot fire — and `#{σ ≥ high} = r`. ∎
///
/// **Sharpness.** `ε*` is the exact threshold, not a conservative estimate:
/// the diagonal perturbation that lowers `σ_r` by `ε* + δ` (when the kept side
/// is the binding term) or raises `σ_{r+1}` by `ε* + δ` (when the dropped side
/// binds — available only when a dropped value exists, i.e. `r < n`) has
/// operator norm `ε* + δ` and pushes that value strictly inside the band, so
/// the decision becomes Ambiguous. No larger radius is transportable.
///
/// **Why this is the right currency for a moving operating point.** A gate
/// that re-ranks at a pilot point and again at the optimum compares two
/// POINTS; equal endpoint ranks say nothing about the path between them, where
/// the solve actually lives. `ε*` converts the certificate into a statement
/// about a NEIGHBOURHOOD, so a path `θ(s)` never leaving the reference's
/// `ε*`-ball in operator norm carries the pilot verdict at every `s`
/// (Cor. below). That is the §8 two-stage mint protocol: publish the margin
/// at the pilot, price the realized excursion against it at mint.
///
/// **Path corollary.** For a path `s ↦ A(θ(s))` with `A(θ(0)) = A₀`, the
/// reference rank holds along the WHOLE path as soon as
/// `sup_s ‖A(θ(s)) − A₀‖₂ ≤ ε*`; if `A` is `L`-Lipschitz in `θ` on the path's
/// hull, `L · sup_s ‖θ(s) − θ(0)‖ ≤ ε*` suffices. Both are conditions on a
/// margin, never on a step count or a coefficient-movement heuristic.
pub fn rank_transport_radius(decision: &RankDecision) -> Option<f64> {
    match *decision {
        RankDecision::Certified {
            rank,
            sigma_r,
            sigma_next,
            spectrum_len,
            tol,
            gap,
            ..
        } => {
            let high = tol * (1.0 + gap);
            let low = tol / (1.0 + gap);
            // `sigma_r` is `+∞` at rank 0 — no kept side to protect, so that
            // branch imposes no constraint and the dropped side alone binds.
            let kept_slack = if sigma_r.is_finite() {
                sigma_r - high
            } else {
                f64::INFINITY
            };
            // `sigma_next` is `0` at full rank as a SENTINEL, not as a
            // singular value: the `i > r` half of the Weyl argument quantifies
            // over an empty index set, so that side imposes no constraint and
            // its slack is `+∞`, exactly as the kept side's is at rank 0. The
            // test cannot be `sigma_next == 0.0`, because a genuinely zero
            // DROPPED value (`rank < spectrum_len`) does constrain — a
            // perturbation of norm `ε` lifts it to `ε` and it must stay
            // `≤ low`. Only the spectrum dimension separates the two.
            let dropped_slack = if spectrum_len > 0 && rank == spectrum_len {
                f64::INFINITY
            } else {
                low - sigma_next
            };
            let radius = kept_slack.min(dropped_slack);
            // The Certified predicate makes both terms non-negative; a
            // non-finite band (a caller passing a non-finite tolerance) has no
            // transportable margin.
            (radius.is_finite() && radius >= 0.0).then_some(radius)
        }
        RankDecision::Ambiguous { .. } => None,
    }
}

/// Transport a certified rank decision across a bounded operator excursion.
///
/// `excursion` must be an upper bound on `‖A − A₀‖₂` between the operator the
/// reference decision was taken on and the one being queried — for a path, the
/// supremum over the path. See [`rank_transport_radius`] for the derivation and
/// the path corollary.
///
/// The verdict is deliberately one-sided in the safe direction:
/// [`Transported`](RankTransport::Transported) is a proof that re-deciding is
/// redundant, while [`GapExhausted`](RankTransport::GapExhausted) is only the
/// absence of such a proof — it never claims the rank changed. A fail-closed
/// gate therefore re-decides on `GapExhausted` rather than refusing.
pub fn transport_certified_rank(decision: &RankDecision, excursion: f64) -> RankTransport {
    let Some(radius) = rank_transport_radius(decision) else {
        return RankTransport::NoCertificate;
    };
    if !(excursion.is_finite() && excursion >= 0.0) {
        return RankTransport::NoCertificate;
    }
    let RankDecision::Certified { rank, .. } = *decision else {
        return RankTransport::NoCertificate;
    };
    if excursion <= radius {
        RankTransport::Transported {
            rank,
            radius,
            excursion,
            slack: radius - excursion,
        }
    } else {
        RankTransport::GapExhausted {
            rank,
            radius,
            excursion,
        }
    }
}

/// Weyl lower bound on the operator excursion between two spectra of the same
/// operator family: `max_i |σ_i(A) − σ_i(A₀)| ≤ ‖A − A₀‖₂`.
///
/// Both slices are sorted descending internally and compared index-wise; the
/// shorter one is zero-extended (a missing trailing singular value is `0`, the
/// same convention [`certified_rank`] uses for `σ_{n+1}`).
///
/// # Why a LOWER bound is the useful one for a gate
///
/// [`transport_certified_rank`] needs an UPPER bound on the excursion to
/// certify transport, and spectra alone cannot supply one — two operators can
/// share a spectrum and be far apart. What spectra DO supply is a certified
/// LOWER bound, and that is exactly what a fail-closed gate needs on the other
/// side: if this bound already exceeds the reference's
/// [`rank_transport_radius`], the reference certificate is PROVABLY void at the
/// query point, so the gate must re-decide there — a refusal-to-reuse trigger
/// stated in the decision's own currency instead of in a coefficient-movement
/// heuristic. It is silent (never a proof of transport) in the other direction.
pub fn spectral_excursion_lower_bound(reference: &[f64], current: &[f64]) -> f64 {
    let mut a: Vec<f64> = reference.to_vec();
    let mut b: Vec<f64> = current.to_vec();
    a.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
    b.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
    let n = a.len().max(b.len());
    let mut worst = 0.0_f64;
    for i in 0..n {
        let sa = a.get(i).copied().unwrap_or(0.0);
        let sb = b.get(i).copied().unwrap_or(0.0);
        worst = worst.max((sa - sb).abs());
    }
    worst
}

/// Davis–Kahan resolution bar on a SPECTRAL PROJECTOR (#2448).
///
/// Returns a certified upper bound on `‖P̂ − P‖₂` — the distance between the
/// projector computed from a perturbed symmetric operator and the projector of
/// the exact one, onto the same kept/dropped partition. `f64::INFINITY` when the
/// partition is not resolved at all, which every consumer must read as "refuse".
///
/// `gap` is the MEASURED eigenvalue separation `λ_min(kept) − λ_max(dropped)`;
/// `backward_error` is an absolute bar `‖E‖₂` on those eigenvalues (for a
/// backward-stable symmetric eigensolver, `p(k)·ε·λ_max`). Both live in the
/// operator's own units, so the returned bar is dimensionless and directly
/// comparable to whatever subspace tolerance the caller is about to gate on.
///
/// # Why this is a currency of its own, distinct from the rank
///
/// [`certified_rank`] and [`rank_transport_radius`] certify an INTEGER: how many
/// directions clear a cutoff, and how far the operator may move before that
/// count changes. Neither says anything about WHICH directions those are. The
/// two claims are conditioned by different quantities — the rank by the distance
/// from `λ_r` to the cutoff, the eigenspace by the gap from `λ_r` down to
/// `λ_{r+1}` — and an operator can have a comfortably certified rank whose
/// eigenspace is individually undetermined. That happens exactly when the
/// spectrum DECAYS SMOOTHLY through the cutoff instead of cliffing at it, which
/// is the common case for kernel Grams, and it is invisible to a rank test.
///
/// So: any gate that compares two computed subspaces to a tolerance must first
/// establish that it can resolve that tolerance. Otherwise the comparison
/// returns roundoff, and roundoff is not monotone in anything — a bisection over
/// such a predicate has no crossing to find, and the "edge" it converges to is a
/// property of the last bits of the eigensolver.
///
/// # Derivation (the denominator is the part that is easy to get wrong)
///
/// The `sinΘ` theorem in its mixed form (Stewart & Sun, Thm V.3.6) bounds the
/// rotation by `‖E‖₂ / η`, where `η` separates the spectrum of the PERTURBED
/// kept block from that of the UNPERTURBED dropped block: `η = λ̂_r − λ_{r+1}`.
/// Only computed eigenvalues are ever in hand, so `η` must be bounded below by
/// them. Weyl gives `λ_{r+1} ≤ λ̂_{r+1} + ‖E‖₂`, hence
///
/// ```text
///   η ≥ λ̂_r − λ̂_{r+1} − ‖E‖₂ = gap − ‖E‖₂
/// ```
///
/// so `‖P̂ − P‖₂ ≤ ‖E‖₂ / (gap − ‖E‖₂)` with a SINGLE `−‖E‖₂`, `gap` being the
/// measured separation. This is already rigorous: it is not the asymptotic
/// `‖E‖/gap` awaiting a correction factor, and it does not need a second or
/// third `‖E‖₂` subtracted to account for "the measured gap is not the true
/// gap". That step is precisely what the mixed form absorbs.
///
/// # Refusal
///
/// When `gap ≤ ‖E‖₂` the kept and dropped blocks are not separated: the computed
/// eigenvectors are an arbitrary rotation inside a numerically degenerate
/// cluster and no subspace claim is available at any tolerance. Malformed input
/// (either argument non-finite) refuses identically — returning `NaN` would be
/// worse than useless, since `NaN <= atol` is `false` in a refuse-on-false gate
/// but `atol <= NaN` is also `false` in an accept-on-false one, so the direction
/// of the mistake would depend on how the caller spelled the comparison.
///
/// Callers holding TWO computed projectors and a measured distance between them
/// should gate on `measured + bar_a + bar_b` (the triangle inequality), not on
/// `measured` alone: that sum is a certified upper bound on the true subspace
/// distance, and it degrades to the measurement itself wherever both gaps are
/// wide — in particular to exactly `0` at full rank, where the projector is the
/// identity on every host.
pub fn projector_error_bar(gap: f64, backward_error: f64) -> f64 {
    if !(gap.is_finite() && backward_error.is_finite()) {
        return f64::INFINITY;
    }
    let separation = gap - backward_error;
    if !(separation > 0.0) {
        return f64::INFINITY;
    }
    backward_error / separation
}

/// A rigorous two-sided enclosure `[lower, upper]` of a scalar quantity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecrementEnclosure {
    /// Certified lower bound.
    pub lower: f64,
    /// Certified upper bound.
    pub upper: f64,
}

/// A running sum that also carries the data needed to certify its own rounding
/// floor: the accumulated value, the sum of magnitudes, and the term count.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ShadowSum {
    /// Accumulated (finite-precision) sum.
    pub sum: f64,
    /// Sum of magnitudes `Σ|x_i|`, the scale of the rounding floor.
    pub abs_sum: f64,
    /// Number of terms pushed.
    pub count: usize,
}

impl ShadowSum {
    /// An empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one term into the running sum.
    pub fn push(&mut self, x: f64) {
        self.sum += x;
        self.abs_sum += x.abs();
        self.count += 1;
    }

    /// Combine two independently accumulated sums (associative, for reductions).
    pub fn merge(&mut self, other: &ShadowSum) {
        self.sum += other.sum;
        self.abs_sum += other.abs_sum;
        self.count += other.count;
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faer_ndarray::{FaerEigh, fast_ata};
    use faer::Side;
    use ndarray::{Array1, Array2};

    // Machine unit roundoff for f64 (2^-53).
    const U: f64 = f64::EPSILON / 2.0;

    fn eigenvalues(m: &Array2<f64>) -> Vec<f64> {
        let (evals, _) = m.eigh(Side::Lower).expect("eigh");
        evals.to_vec()
    }

    /// #2448 — the Davis–Kahan bar must DOMINATE the true projector rotation,
    /// graded against the one case where that rotation is known in closed form.
    ///
    /// For `A = diag(a, b)` with `a > b` and the symmetric off-diagonal
    /// perturbation `E = [[0, e], [e, 0]]`, the eigenvectors of `A + E` are those
    /// of `A` turned by exactly `θ = ½·atan(2e/(a−b))`, so the distance between
    /// the rank-1 leading projectors is `sinθ`. Here `‖E‖₂ = |e|` and the gap is
    /// `δ = a − b`, and the bar must dominate at every ratio `e/δ` — including
    /// ratios far outside the small-`e` regime the asymptotic form is derived in.
    ///
    /// Tightness is asserted too. A bound that simply returned `+∞` would pass
    /// every domination check while making the whole currency useless, so the
    /// bar is also required to stay within 5% of the truth for `e ≪ δ` — the
    /// regime in which a real gate has to be able to CERTIFY, not just refuse.
    #[test]
    fn projector_error_bar_dominates_the_closed_form_rotation_and_stays_tight_2448() {
        let gap = 1.0_f64;
        let (a, b) = (2.0_f64, 2.0 - gap);
        // Leading-eigenvector projector of the symmetric 2×2 [[a, e], [e, b]],
        // and the spectral norm of its difference from the unperturbed one.
        let rotation = |e: f64| -> f64 {
            let leading = |off: f64| -> Array2<f64> {
                let m = ndarray::arr2(&[[a, off], [off, b]]);
                let (evals, evecs) = m.eigh(Side::Lower).expect("2x2 eigh");
                let top = evals
                    .iter()
                    .enumerate()
                    .max_by(|(_, x), (_, y)| x.total_cmp(y))
                    .map(|(i, _)| i)
                    .expect("non-empty spectrum");
                let u = evecs.column(top);
                let mut p = Array2::<f64>::zeros((2, 2));
                for i in 0..2 {
                    for j in 0..2 {
                        p[[i, j]] = u[i] * u[j];
                    }
                }
                p
            };
            let diff = &leading(e) - &leading(0.0);
            let dsym = 0.5 * (&diff + &diff.t());
            let (evals, _) = dsym.eigh(Side::Lower).expect("difference eigh");
            evals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()))
        };

        for &e in &[1e-12_f64, 1e-8, 1e-4, 1e-2, 0.1, 0.4, 0.9] {
            let measured = rotation(e);
            let exact = (0.5 * (2.0 * e / gap).atan()).sin();
            assert!(
                (measured - exact).abs() <= 1e-12 + 1e-9 * exact,
                "the 2×2 oracle disagrees with the measured projector distance at \
                 e={e}: measured={measured}, closed form={exact} — the oracle, not \
                 the bound, is what is wrong here"
            );
            let bar = projector_error_bar(gap, e);
            assert!(
                bar >= measured,
                "the bar must DOMINATE the true rotation at e={e} (gap={gap}): \
                 bar={bar}, measured={measured}"
            );
            if e <= 1e-2 {
                // `e/(δ−e)` against a truth of `≈ e/δ`: at most a factor
                // `δ/(δ−e)`, here ≤ 1.02.
                assert!(
                    bar <= 1.05 * measured.max(f64::MIN_POSITIVE),
                    "the bar must stay tight for e ≪ δ, else it would refuse \
                     decidable subspace claims: e={e}, bar={bar}, measured={measured}"
                );
            }
        }

        // A perturbation wide enough to span the gap leaves the kept/dropped
        // split undetermined: refuse, never report a finite rotation.
        for &(g, e) in &[(1.0_f64, 1.0_f64), (1.0, 2.0), (0.0, 1e-30), (-1.0, 1e-30)] {
            assert!(
                projector_error_bar(g, e).is_infinite(),
                "gap={g} with ‖E‖={e} determines no subspace; the bar must refuse"
            );
        }
        // Malformed input refuses rather than producing a NaN, whose comparison
        // verdict would depend on how the caller spelled the `<=`.
        assert!(projector_error_bar(f64::NAN, 1.0).is_infinite());
        assert!(projector_error_bar(1.0, f64::NAN).is_infinite());
        assert!(projector_error_bar(f64::INFINITY, f64::INFINITY).is_infinite());

        // A certified RANK does not imply a certified EIGENSPACE — the point of
        // the currency being separate. A spectrum decaying smoothly through the
        // cutoff certifies its integer with margin to spare while leaving the
        // eigenspace realizing it undetermined at the same backward error.
        let lambda_max = 1.0_f64;
        let spectrum = [lambda_max, 1.3e-2, 4.4e-4, 5.1e-6, 3.6e-8, 1.2e-10, 3.0e-13];
        let rank_tol = 1.0e-10 * lambda_max;
        let backward_error = 8.0 * (spectrum.len() as f64) * f64::EPSILON * lambda_max;
        let decision = certified_rank(&spectrum, rank_tol, backward_error / rank_tol);
        let RankDecision::Certified { rank, .. } = decision else {
            panic!("the integer is decidable here: {decision:?}");
        };
        assert_eq!(rank, 6, "six eigenvalues clear the cutoff with margin");
        let bar = projector_error_bar(spectrum[rank - 1] - spectrum[rank], backward_error);
        assert!(
            bar > 1e-6,
            "the rank-6 eigenspace of a smoothly decaying spectrum is NOT resolved \
             even though the rank is certified; got bar={bar:e}, which would mean \
             the two currencies had collapsed into one"
        );
    }

    #[test]
    fn equilibration_certifies_full_rank_where_raw_gram_would_kill_eleven_columns() {
        // A 12-column design whose first column is stiffened by 2.4e6 so the
        // Gram anisotropy is (2.4e6)² ≈ 5.76e12. Columns are orthonormal (rows
        // 0..11 form an identity block; extra rows are zero) so the raw Gram is
        // diagonal: eigenvalues [ (2.4e6)², 1, …, 1 ].
        let n = 50usize;
        let p = 12usize;
        let stiff = 2.4e6_f64;
        let mut x = Array2::<f64>::zeros((n, p));
        for j in 0..p {
            x[[j, j]] = if j == 0 { stiff } else { 1.0 };
        }
        let g = fast_ata(&x);

        // Raw-Gram decision with the codebase-style size-scaled machine-epsilon
        // cutoff τ = λ_max · 64 · n · ε (the schematic "u·p·λ_max" of the
        // design; here the size factor is n): the eleven unit eigenvalues sit
        // below τ and are killed, leaving rank 1.
        let raw_evals = eigenvalues(&g);
        let raw_lambda_max = raw_evals.iter().cloned().fold(0.0_f64, f64::max);
        let raw_tol = raw_lambda_max * 64.0 * (n as f64) * f64::EPSILON;
        let raw_rank = raw_evals.iter().filter(|&&e| e > raw_tol).count();
        assert_eq!(raw_rank, 1, "raw size-scaled cutoff must kill 11 columns");

        // Equilibrated decision: D^{-1/2} G D^{-1/2} = I_12, every eigenvalue is
        // 1, and certified_rank returns the full rank 12 with a huge margin.
        let (g_eq, _) = equilibrate_gram(&g);
        let eq_evals = eigenvalues(&g_eq);
        for &e in &eq_evals {
            assert!((e - 1.0).abs() < 1e-9, "equilibrated spectrum must be ~1");
        }
        let eq_lambda_max = eq_evals.iter().cloned().fold(0.0_f64, f64::max);
        let nk = (n.max(p)) as f64;
        let eq_tol = eq_lambda_max * 64.0 * nk * f64::EPSILON;
        match certified_rank(&eq_evals, eq_tol, 1.0) {
            RankDecision::Certified {
                rank, margin_high, ..
            } => {
                assert_eq!(rank, 12, "equilibrated Gram is full rank");
                assert!(
                    margin_high > 1e10,
                    "kept side must clear the band by a huge factor, got {margin_high}"
                );
            }
            other => panic!("expected Certified full rank, got {other:?}"),
        }
    }

    /// The `r = n` half of Thm 8.3 quantifies over an EMPTY index set, so a
    /// full-rank certificate transports on its kept slack alone. The second
    /// arm is why this cannot be sniffed from `sigma_next == 0.0`: a dropped
    /// singular value that is exactly zero looks identical on the certificate
    /// and DOES bind.
    #[test]
    fn a_full_rank_certificate_transports_on_its_kept_side_alone() {
        // tol = 1, gap = 1 ⇒ high = 2, low = 0.5. Both values clear `high`,
        // so the rank is 2 = n and there is no dropped side at all.
        let full = certified_rank(&[10.0_f64, 9.0], 1.0, 1.0);
        let RankDecision::Certified {
            rank, sigma_next, ..
        } = full
        else {
            panic!("expected a Certified reference, got {full:?}");
        };
        assert_eq!(rank, 2, "both values clear high = 2");
        assert_eq!(sigma_next, 0.0, "the dropped slot holds the sentinel");
        let radius = rank_transport_radius(&full).expect("certified ⇒ a radius");
        assert!(
            (radius - 7.0).abs() < 1e-12,
            "ε* is the kept slack σ_n − high = 9 − 2 = 7; reading the sentinel as              a dropped value would cap it at low = 0.5. got {radius}"
        );

        // Sharp on that side too: exactly `ε*` still decides rank 2, and a hair
        // past it drops σ_n into the open band.
        match certified_rank(&[10.0 - radius, 9.0 - radius], 1.0, 1.0) {
            RankDecision::Certified { rank: moved, .. } => assert_eq!(
                moved, 2,
                "a perturbation of exactly ε* must not move the certified rank"
            ),
            other => panic!("expected the rank to transport at ε*, got {other:?}"),
        }
        let past = radius + 1e-9;
        assert!(
            matches!(
                certified_rank(&[10.0 - past, 9.0 - past], 1.0, 1.0),
                RankDecision::Ambiguous { .. }
            ),
            "a hair past ε* must push σ_n strictly inside the band"
        );

        // Same `sigma_next == 0.0`, but here it is a REAL dropped singular
        // value. The kept slack is still 7, yet the dropped side binds at
        // `low − 0 = 0.5` — which is why the certificate must carry the
        // spectrum dimension rather than test the value.
        let with_zero = certified_rank(&[10.0_f64, 9.0, 0.0], 1.0, 1.0);
        let RankDecision::Certified {
            rank, sigma_next, ..
        } = with_zero
        else {
            panic!("expected a Certified reference, got {with_zero:?}");
        };
        assert_eq!(rank, 2, "the exact zero is dropped");
        assert_eq!(
            sigma_next, 0.0,
            "indistinguishable from the full-rank sentinel BY VALUE"
        );
        let zero_radius = rank_transport_radius(&with_zero).expect("certified ⇒ a radius");
        assert!(
            (zero_radius - 0.5).abs() < 1e-12,
            "a real zero can be lifted to ε by a perturbation of norm ε and must              stay ≤ low, so ε* = 0.5 here; got {zero_radius}"
        );

        // And that is not conservatism: lifting it past `low` really does make
        // the decision Ambiguous, so the shorter radius is earned.
        assert!(
            matches!(
                certified_rank(&[10.0_f64, 9.0, zero_radius + 1e-9], 1.0, 1.0),
                RankDecision::Ambiguous { .. }
            ),
            "past ε* the formerly-zero value enters the open band"
        );
    }

    /// The consequence at the shape the identifiability audit actually hands
    /// this function: a WELL-CONDITIONED, FULL-RANK equilibrated Gram decided
    /// against a machine-epsilon-scaled cutoff.
    ///
    /// `gam-identifiability`'s `channel_aware_penalty_aware_joint_rank` poses
    /// the decision at `tol = 100·ε·max(rows, cols)·σ_max` on an equilibrated
    /// Gram, whose spectrum is `O(1)`. Reading the full-rank `σ_{n+1} = 0`
    /// sentinel as a dropped value caps the radius at `low ≈ tol`, i.e. at the
    /// ROUNDOFF scale, while the true margin is the `O(1)` distance from the
    /// smallest kept value to the band. Any real movement of the operating
    /// point exceeds a roundoff-scale radius, so the transported branch was
    /// unreachable on precisely the well-identified problems where the
    /// certificate does carry — the gate would report "endpoint agreement
    /// only" every time and never say why it was wrong to.
    #[test]
    fn a_full_rank_epsilon_scaled_certificate_can_transport_at_all() {
        // The equilibrated shape: unit diagonal ⇒ every singular value is O(1).
        let spectrum = [1.0_f64, 0.94, 0.71, 0.55, 0.38];
        let rows_plus_penalties = 150.0_f64;
        let sigma_max = spectrum[0];
        let tol = 100.0 * f64::EPSILON * rows_plus_penalties * sigma_max.max(1.0);
        let gap = 1.0_f64;
        let decision = certified_rank(&spectrum, tol, gap);
        let RankDecision::Certified { rank, .. } = decision else {
            panic!("an O(1) spectrum at a roundoff cutoff is decidable: {decision:?}");
        };
        assert_eq!(rank, spectrum.len(), "every value clears a roundoff cutoff");

        let radius = rank_transport_radius(&decision).expect("certified ⇒ a radius");
        let sentinel_reading = tol / (1.0 + gap);
        assert!(
            radius > 0.37,
            "ε* must be the kept slack σ_n − high ≈ 0.38, an O(1) margin; got {radius:e}"
        );
        assert!(
            radius > 1.0e10 * sentinel_reading,
            "the sentinel reading is the roundoff-scale {sentinel_reading:e}; the true \
             margin is {radius:e}, more than ten orders larger. If these were within \
             ten orders of each other the defect this pins would not be reachable."
        );

        // And that gap is what makes the transported branch reachable: an
        // operating-point excursion of 1e-3 — small for a fit, enormous next to
        // roundoff — transports under the true radius and is refused under the
        // sentinel reading.
        let realistic_excursion = 1.0e-3_f64;
        assert!(
            matches!(
                transport_certified_rank(&decision, realistic_excursion),
                RankTransport::Transported { .. }
            ),
            "a 1e-3 excursion sits well inside an O(1) margin and must transport"
        );
        assert!(
            realistic_excursion > sentinel_reading,
            "under the sentinel reading the same excursion exhausts the radius, which \
             is why the transported branch was unreachable at full rank"
        );
    }

    /// Thm 8.3, the radius itself: every perturbation inside `ε*` decides the
    /// same rank, and the bound is SHARP — a perturbation a hair past `ε*`
    /// applied to the binding side pushes a singular value into the band and
    /// the decision becomes Ambiguous.
    #[test]
    fn transport_radius_is_the_sharp_threshold_of_a_certified_rank() {
        // tol = 1, gap = 1 ⇒ high = 2, low = 0.5. Spectrum splits 2 kept / 2
        // dropped: σ_r = 3 clears high by 1; σ_{r+1} = 0.1 clears low by 0.4.
        // The dropped side binds, so ε* = 0.4.
        let sv = [10.0_f64, 3.0, 0.1, 0.05];
        let reference = certified_rank(&sv, 1.0, 1.0);
        let RankDecision::Certified { rank, .. } = reference else {
            panic!("expected a Certified reference, got {reference:?}");
        };
        assert_eq!(rank, 2);
        let radius = rank_transport_radius(&reference).expect("certified ⇒ a radius");
        assert!(
            (radius - 0.4).abs() < 1e-12,
            "ε* = min(σ_r − high, low − σ_next) = min(1, 0.4); got {radius}"
        );

        // INSIDE the radius, on the binding side, at the exact boundary: still
        // the same certified rank (Weyl is non-strict).
        let inside = [10.0 + radius, 3.0 - radius, 0.1 + radius, 0.05 + radius];
        match certified_rank(&inside, 1.0, 1.0) {
            RankDecision::Certified { rank: moved, .. } => assert_eq!(
                moved, rank,
                "a perturbation of exactly ε* must not move the certified rank"
            ),
            other => panic!("expected the rank to transport at ε*, got {other:?}"),
        }

        // JUST OUTSIDE, same direction: σ_{r+1} enters the open band (low,
        // high) and the decision is no longer certifiable at all.
        let outside = [10.0, 3.0, 0.1 + radius + 1e-9, 0.05];
        assert!(
            matches!(
                certified_rank(&outside, 1.0, 1.0),
                RankDecision::Ambiguous { .. }
            ),
            "ε* must be sharp: a perturbation past it breaks the certificate"
        );
    }

    /// The transport verdict is exactly the radius comparison, and it reports
    /// the unused slack a caller can spend before renewing the certificate.
    #[test]
    fn transport_verdict_prices_the_excursion_against_the_radius() {
        let sv = [10.0_f64, 3.0, 0.1, 0.05];
        let reference = certified_rank(&sv, 1.0, 1.0);
        let radius = rank_transport_radius(&reference).expect("certified ⇒ a radius");

        match transport_certified_rank(&reference, 0.25) {
            RankTransport::Transported {
                rank,
                slack,
                radius: r,
                ..
            } => {
                assert_eq!(rank, 2);
                assert!((r - radius).abs() < 1e-12);
                assert!(
                    (slack - (radius - 0.25)).abs() < 1e-12,
                    "slack must be the unspent radius, got {slack}"
                );
            }
            other => panic!("0.25 < ε* must transport, got {other:?}"),
        }
        match transport_certified_rank(&reference, radius * 2.0) {
            RankTransport::GapExhausted { rank, .. } => assert_eq!(
                rank, 2,
                "GapExhausted still names the rank it can no longer imply"
            ),
            other => panic!("an excursion past ε* must exhaust the gap, got {other:?}"),
        }
        assert_eq!(
            transport_certified_rank(&reference, f64::NAN),
            RankTransport::NoCertificate,
            "an unbounded excursion certifies nothing"
        );
    }

    /// An undecided rank has no margin, so there is nothing to transport — the
    /// gate must re-decide rather than reuse.
    #[test]
    fn ambiguous_reference_has_no_transport_certificate() {
        let ambiguous = certified_rank(&[10.0_f64, 3.0, 1.0, 0.2], 1.0, 1.0);
        assert!(matches!(ambiguous, RankDecision::Ambiguous { .. }));
        assert_eq!(rank_transport_radius(&ambiguous), None);
        assert_eq!(
            transport_certified_rank(&ambiguous, 0.0),
            RankTransport::NoCertificate
        );
    }

    /// The path corollary on a REAL operator path: `A(s) = A₀ + s·E` with
    /// `‖E‖₂ = 1`, so the path is 1-Lipschitz in `s` and the excursion at `s`
    /// is exactly `s`. Every sample with `s ≤ ε*` must decide the reference
    /// rank — the statement the identifiability gate needs when the operating
    /// point moves along the optimizer's path rather than jumping between two
    /// audited endpoints.
    #[test]
    fn certified_rank_transports_along_a_lipschitz_operator_path() {
        // Symmetric PSD A₀ with spectrum {6, 5, 0.02, 0.01}: singular values
        // are the eigenvalues. tol = 1, gap = 1 ⇒ high = 2, low = 0.5.
        let mut a0 = Array2::<f64>::zeros((4, 4));
        for (i, &s) in [6.0_f64, 5.0, 0.02, 0.01].iter().enumerate() {
            a0[[i, i]] = s;
        }
        let reference = certified_rank(&eigenvalues(&a0), 1.0, 1.0);
        let RankDecision::Certified { rank, .. } = reference else {
            panic!("expected a Certified reference, got {reference:?}");
        };
        assert_eq!(rank, 2);
        let radius = rank_transport_radius(&reference).expect("certified ⇒ a radius");

        // A unit-norm symmetric direction that mixes every index, so the path
        // is not a diagonal special case: E = (vvᵀ + wwᵀ)/‖·‖ rescaled to
        // spectral norm 1.
        let v = Array1::from(vec![0.5_f64, -0.5, 0.5, -0.5]);
        let w = Array1::from(vec![0.5_f64, 0.5, -0.5, -0.5]);
        let mut e = Array2::<f64>::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                e[[i, j]] = v[i] * v[j] - w[i] * w[j];
            }
        }
        let e_norm = eigenvalues(&e)
            .into_iter()
            .fold(0.0_f64, |acc, l| acc.max(l.abs()));
        assert!(e_norm > 0.0);
        e.mapv_inplace(|x| x / e_norm);

        for step in 0..=8usize {
            let s = radius * (step as f64) / 8.0;
            let a_s = &a0 + &(e.clone() * s);
            // Excursion of this sample is exactly s (‖E‖₂ = 1 after scaling).
            let sv: Vec<f64> = eigenvalues(&a_s).into_iter().map(f64::abs).collect();
            match certified_rank(&sv, 1.0, 1.0) {
                RankDecision::Certified { rank: moved, .. } => assert_eq!(
                    moved, rank,
                    "path sample s={s} inside ε*={radius} must keep rank {rank}"
                ),
                other => panic!("path sample s={s} inside ε*={radius} lost its rank: {other:?}"),
            }
            assert!(
                matches!(
                    transport_certified_rank(&reference, s),
                    RankTransport::Transported { .. }
                ),
                "the transport verdict must agree with the realized path at s={s}"
            );
        }
    }

    /// The spectral monitor is a genuine Weyl LOWER bound on the operator
    /// excursion — never an over-claim — so a gate may use it only to refuse
    /// reuse, never to certify it.
    #[test]
    fn spectral_excursion_is_a_weyl_lower_bound_on_the_operator_norm() {
        let mut a0 = Array2::<f64>::zeros((4, 4));
        for (i, &s) in [6.0_f64, 5.0, 0.02, 0.01].iter().enumerate() {
            a0[[i, i]] = s;
        }
        // A perturbation with a large off-diagonal part: its spectral norm is
        // strictly larger than the spectral displacement it induces, so the
        // bound is strict here rather than tight.
        let mut delta = Array2::<f64>::zeros((4, 4));
        delta[[0, 3]] = 0.4;
        delta[[3, 0]] = 0.4;
        delta[[1, 2]] = -0.3;
        delta[[2, 1]] = -0.3;
        let a1 = &a0 + &delta;

        let sv0: Vec<f64> = eigenvalues(&a0).into_iter().map(f64::abs).collect();
        let sv1: Vec<f64> = eigenvalues(&a1).into_iter().map(f64::abs).collect();
        let measured = spectral_excursion_lower_bound(&sv0, &sv1);
        let true_norm = eigenvalues(&delta)
            .into_iter()
            .fold(0.0_f64, |acc, l| acc.max(l.abs()));
        assert!(
            measured <= true_norm + 8.0 * U * true_norm.max(1.0),
            "Weyl: max|Δσ| = {measured} must not exceed ‖ΔA‖₂ = {true_norm}"
        );
        assert!(
            measured > 0.0,
            "the monitor must actually see this perturbation"
        );

        // Zero-extension convention: a missing trailing value is σ = 0.
        assert!(
            (spectral_excursion_lower_bound(&[3.0, 0.25], &[3.0]) - 0.25).abs() < 1e-15,
            "a dropped trailing singular value is compared against 0"
        );
    }

    #[test]
    fn spectrum_inside_two_sided_band_is_ambiguous() {
        // tol = 1, gap = 1 ⇒ band (0.5, 2). The value 1.0 lands inside it.
        let sv = [10.0_f64, 3.0, 1.0, 0.2];
        match certified_rank(&sv, 1.0, 1.0) {
            RankDecision::Ambiguous {
                rank_floor,
                rank_ceil,
                sigma_in_band,
                ..
            } => {
                assert_eq!(rank_floor, 2, "#{{σ ≥ 2}} = 2");
                assert_eq!(rank_ceil, 3, "#{{σ > 0.5}} = 3");
                assert_eq!(sigma_in_band, 1.0);
            }
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    #[test]
    fn decrement_enclosure_is_exact_when_residual_zero_and_contains_truth_when_perturbed() {
        // Small SPD H, exact g, exact z = H⁻¹g.
        let h = Array2::from_shape_vec((3, 3), vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0])
            .unwrap();
        let g = Array1::from_vec(vec![1.0, -2.0, 0.5]);

        // Exact solve via the spectral inverse H⁻¹ = V diag(1/λ) Vᵀ.
        let (evals, evecs) = h.eigh(Side::Lower).expect("eigh");
        let lambda_min = evals.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(lambda_min > 0.0, "H must be SPD");
        let vt_g = evecs.t().dot(&g);
        let scaled: Array1<f64> = Array1::from_shape_fn(3, |i| vt_g[i] / evals[i]);
        let z = evecs.dot(&scaled);
        let true_lambda_n_sq = g.dot(&z);

        // r = g − Hz = 0 (up to roundoff): enclosure collapses to the truth.
        let hz = h.dot(&z);
        let r = &g - &hz;
        let g_dot_z = g.dot(&z);
        let r_dot_z = r.dot(&z);
        let r_norm_sq = r.dot(&r);
        let ell = lambda_min * 0.999; // valid lower bound ℓ ≤ λ_min(H)
        let exact = newton_decrement_enclosure(g_dot_z, r_dot_z, r_norm_sq, ell)
            .expect("positive definite");
        assert!(
            (exact.upper - exact.lower).abs() < 1e-10,
            "width must be ~0 when r=0"
        );
        assert!((exact.lower - true_lambda_n_sq).abs() < 1e-9);

        // Perturb z; the enclosure must still contain the true λ_N².
        let z_bad = &z + &Array1::from_vec(vec![0.05, -0.03, 0.02]);
        let hz_bad = h.dot(&z_bad);
        let r_bad = &g - &hz_bad;
        let encl =
            newton_decrement_enclosure(g.dot(&z_bad), r_bad.dot(&z_bad), r_bad.dot(&r_bad), ell)
                .expect("positive definite");
        assert!(
            encl.lower <= true_lambda_n_sq + 1e-9 && true_lambda_n_sq <= encl.upper + 1e-9,
            "enclosure [{}, {}] must contain λ_N² = {true_lambda_n_sq}",
            encl.lower,
            encl.upper
        );
        assert!(
            encl.upper - encl.lower > 0.0,
            "inexact solve widens the band"
        );

        // A non-positive lower bound yields no certificate.
        assert!(newton_decrement_enclosure(g_dot_z, r_dot_z, r_norm_sq, 0.0).is_none());
    }

    #[test]
    fn shadow_sum_error_stays_within_certified_rounding_floor() {
        let mut acc = ShadowSum::new();
        for _ in 0..1_000_000 {
            acc.push(0.1);
        }
        assert_eq!(acc.count, 1_000_000);
        let exact = 100_000.0_f64;
        let error = (acc.sum - exact).abs();
        let floor = acc.rounding_floor(U);
        assert!(
            error <= floor,
            "summation error {error} must not exceed rounding floor {floor}"
        );
        // The tree-depth floor is tighter but must still bound the sequential
        // error only if the caller actually reduced with a tree; here we merely
        // check the constant shrinks with depth.
        assert!(acc.rounding_floor_with_depth(U, 20) < floor);
    }

    #[test]
    fn shadow_sum_merge_is_additive() {
        let mut a = ShadowSum::new();
        let mut b = ShadowSum::new();
        a.push(1.0);
        a.push(-2.0);
        b.push(3.0);
        a.merge(&b);
        assert_eq!(a.count, 3);
        assert_eq!(a.sum, 2.0);
        assert_eq!(a.abs_sum, 6.0);
    }

    #[test]
    fn gamma_saturates_when_ku_exceeds_one() {
        assert!(gamma(usize::MAX, U).is_infinite());
        assert_eq!(gamma(0, U), 0.0);
    }
}

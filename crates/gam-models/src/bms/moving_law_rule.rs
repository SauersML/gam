//! The moving-law certificate (gam#2926): which law a default fit anchors on when
//! the conditional law of the score moves on the marginal-index span.
//!
//! The candidates are nested, simplest first:
//!
//! 1. the closed-form Gaussian law of the score;
//! 2. a location-scale law, `m(a)` and `v(a)` fitted on the span, with a Gaussian
//!    residual — or, when neither conditional moment moves, one pooled finite law
//!    of the score;
//! 3. the same location-scale law with the residual on its empirical law;
//! 4. local finite laws by context over the covariates the marginal formula reads.
//!
//! The default fits the location-scale Gaussian arm (the Gaussian arm when
//! neither moment moves), and at the converged fit scores every arm by its
//! cross-fitted excess anchoring loss. Each row's law is built without the rows of
//! its fold, at the full-data law's configuration: the location-scale fit keeps
//! the full fit's structure, and the local law its context count and kernel width.
//! There are [`MOVING_LAW_FOLDS`] folds, fixed by hashing each row's own data under
//! [`MOVING_LAW_FOLD_SEED`], so a row's fold follows its values and not its
//! position, thread count or run. At the fitted anchor, the arm's loss on the row is
//! the cross-entropy
//!
//! ```text
//! ℓ = −Σ_anchors [ y·ln P + (1 − y)·ln(1 − P) ],   0·ln 0 = 0,
//! ```
//!
//! where `P = Σ_k w_k Φ(ι(u_k))` is the anchor's event probability under the arm's
//! held-out law and `y = Φ(ι(z_i))` is the same at the row's own score. `ℓ` is
//! linear in `y`, and `z_i` is not in the law the row is scored on, so over the
//! row's score its expectation is `H(m) + KL(m ‖ P)` for `m` the anchor's
//! expectation under the true law: a proper score whose excess over the truth is
//! the exact Bernoulli divergence. Both `ln P` and `ln(1 − P)` are summed in log
//! space ([`log_anchor_probabilities`]), so neither is a difference, and a finite
//! law's tail probability far below the resolution of the other stays finite.
//!
//! An earlier loss, `Σ (r² − 2·r_obs·r)/π(1−π)` in the residuals `r = P − π`, is the
//! quadratic approximation of the same divergence. It overstates a near-certain
//! anchor without bound once `|r| ≫ π(1−π)`. On a survival fit with a shared early
//! entry time (the gam#2768 fixture, n = 3 000) one entry anchor at
//! `π(1−π) = 1.1e-8` scored the location-scale Gaussian arm `673`, where its exact
//! divergence is about `3e-3`. That one row carried 99.9% of the arm's difference,
//! which put the difference at exactly one row-level and fold-level standard error.
//!
//! The rule takes the simplest arm whose mean loss is within one paired standard
//! error of the lowest: a more complex arm is chosen only when it beats every
//! simpler one by more than its noise. The standard error is the larger of the
//! row-level one and the one from the fold means. The row-level error assumes rows
//! err independently; rows that share a held-out law share its error, which only
//! the fold means see (at n = 1 000 they were 1.5 times the row-level error on the
//! local arm, gam#2926 diag16), and the larger of the two can only favour the
//! simpler arm. The width, context count and grid of each arm are fixed, never
//! tuned by this loss.
//!
//! Cross-fitting builds each law on nine tenths of the rows. That handicaps the
//! arms that estimate more — the local arm most — so the rule errs toward the
//! simpler arm, never toward complexity.
//!
//! The folds ([`MovingLawFolds`]), a row's loss ([`moving_law_row_losses`]) and
//! the rule ([`MovingLawCertificate::from_row_losses`]) do not depend on the
//! number of scores or on how an arm's law is built; [`MovingLawCandidates`]
//! builds the arms of one latent score.

use super::*;

/// The number of folds the certificate cross-fits over.
pub(crate) const MOVING_LAW_FOLDS: usize = 10;

/// The declared seed of the fold hash.
pub(crate) const MOVING_LAW_FOLD_SEED: u64 = 0x2926_F01D_0000_0001;

/// Nodes of the Gauss–Hermite rule the Gaussian arms are scored on.
const MOVING_LAW_GAUSS_HERMITE_NODES: usize = 64;

/// One candidate law of the moving-law certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MovingLawArm {
    /// The closed-form Gaussian law of the score.
    Gaussian,
    /// One finite law of the score pooled over the span: what the location-scale
    /// arms reduce to when neither conditional moment moves.
    PooledEmpirical,
    /// `m(a)`, `v(a)` fitted on the span and the standardised residual Gaussian.
    LocationScaleGaussian,
    /// `m(a)`, `v(a)` fitted on the span and the standardised residual on its
    /// empirical law.
    LocationScaleEmpirical,
    /// Local finite laws by context, mixed per row.
    Local,
}

impl MovingLawArm {
    /// The stable spelling a report or log names this arm by.
    pub fn label(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::PooledEmpirical => "pooled-empirical",
            Self::LocationScaleGaussian => "location-scale-gaussian",
            Self::LocationScaleEmpirical => "location-scale-empirical",
            Self::Local => "local",
        }
    }
}

/// One arm's score in a [`MovingLawCertificate`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MovingLawArmScore {
    pub arm: MovingLawArm,
    /// The weighted mean over the scored rows of the arm's cross-fitted loss `ℓ`.
    pub loss: f64,
    /// The weighted mean of the arm's per-row loss minus the lowest-loss arm's.
    pub difference: f64,
    /// The row-level standard error of `difference`.
    pub row_se: f64,
    /// The standard error of `difference` from its fold means.
    pub fold_se: f64,
    /// The standard-normal adequacy screen of the residual an arm anchors on the
    /// Gaussian law: the score itself for the Gaussian arm, the location-scale
    /// residual `ζ` for the location-scale Gaussian arm; `None` for an arm that
    /// anchors on an estimated law. An arm whose screen fails is scored and
    /// recorded, but it is not a candidate: it would anchor on a law the data
    /// reject. `None` in a payload written before the screen was recorded, whose
    /// rule took every arm as a candidate.
    #[serde(default)]
    pub adequacy: Option<LatentNormalAdequacy>,
}

impl MovingLawArmScore {
    /// The standard error the rule compares `difference` with: the larger of the
    /// row- and fold-level ones.
    pub fn rule_se(&self) -> f64 {
        self.row_se.max(self.fold_se)
    }

    /// Whether the rule may choose this arm: its Gaussian residual, if it anchors
    /// on one, passes the adequacy screen.
    pub fn admissible(&self) -> bool {
        self.adequacy.as_ref().is_none_or(LatentNormalAdequacy::passes)
    }
}

/// The moving-law certificate of a default fit (gam#2926): every arm's
/// cross-fitted excess anchoring loss at the converged fit, and the arm the rule
/// chose from them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MovingLawCertificate {
    pub folds: usize,
    pub fold_seed: u64,
    /// Rows scored: positive prior weight and at least one anchor.
    pub rows: usize,
    /// The arm the certified fit was solved on.
    pub fitted: MovingLawArm,
    /// The admissible arm ([`MovingLawArmScore::admissible`]) with the lowest mean
    /// loss.
    pub argmin: MovingLawArm,
    /// The simplest admissible arm whose difference to the argmin is within its
    /// rule standard error ([`MovingLawArmScore::rule_se`]).
    pub chosen: MovingLawArm,
    /// Every arm, simplest first.
    pub arms: Vec<MovingLawArmScore>,
}

impl MovingLawCertificate {
    /// Apply the rule to every row's losses under `ladder` (simplest first;
    /// `None` for a row nothing scored), with `fitted` the arm the losses were
    /// taken at and `adequacy` each arm's Gaussian-residual screen
    /// ([`MovingLawArmScore::adequacy`]). Sums run in row order, so the choice is
    /// the same at every thread count.
    pub(crate) fn from_row_losses(
        ladder: &[MovingLawArm],
        adequacy: &[Option<LatentNormalAdequacy>],
        fitted: MovingLawArm,
        folds: &MovingLawFolds,
        weights: &Array1<f64>,
        losses: &[Option<Vec<f64>>],
    ) -> Result<Self, MovingLawError> {
        if !ladder.contains(&fitted) {
            return Err(MovingLawError::FittedArmOffLadder(fitted));
        }
        if adequacy.len() != ladder.len() {
            return Err(MovingLawError::ArmCount {
                row: None,
                expected: ladder.len(),
                found: adequacy.len(),
            });
        }
        let admissible: Vec<bool> = adequacy
            .iter()
            .map(|screen| screen.as_ref().is_none_or(LatentNormalAdequacy::passes))
            .collect();
        if let Some(position) = ladder.iter().position(|&arm| arm == fitted)
            && !admissible[position]
        {
            return Err(MovingLawError::FittedArmInadmissible(fitted));
        }
        let scored: Vec<(usize, f64, &Vec<f64>)> = losses
            .iter()
            .enumerate()
            .filter_map(|(row, loss)| {
                loss.as_ref()
                    .filter(|_| weights[row] > 0.0)
                    .map(|loss| (row, weights[row], loss))
            })
            .collect();
        if scored.len() < 2 {
            return Err(MovingLawError::TooFewRows { rows: scored.len() });
        }
        if let Some(&(row, _, loss)) = scored.iter().find(|(_, _, loss)| loss.len() != ladder.len()) {
            return Err(MovingLawError::ArmCount {
                row: Some(row),
                expected: ladder.len(),
                found: loss.len(),
            });
        }
        let total_weight = scored.iter().map(|&(_, w, _)| w).sum::<f64>();
        let mean_loss: Vec<f64> = (0..ladder.len())
            .map(|arm| scored.iter().map(|&(_, w, loss)| w * loss[arm]).sum::<f64>() / total_weight)
            .collect();
        // The fitted arm is admissible, so an admissible argmin exists.
        let argmin = (0..ladder.len())
            .filter(|&arm| admissible[arm])
            .fold(None, |best: Option<usize>, arm| match best {
                Some(best) if mean_loss[best] <= mean_loss[arm] => Some(best),
                _ => Some(arm),
            })
            .ok_or(MovingLawError::FittedArmInadmissible(fitted))?;
        let count = scored.len() as f64;
        let arms: Vec<MovingLawArmScore> = (0..ladder.len())
            .map(|arm| {
                let difference = mean_loss[arm] - mean_loss[argmin];
                let spread = scored
                    .iter()
                    .map(|&(_, w, loss)| {
                        let centred = loss[arm] - loss[argmin] - difference;
                        w * w * centred * centred
                    })
                    .sum::<f64>();
                let row_se = (spread * count / (count - 1.0)).sqrt() / total_weight;
                let mut fold_sum = vec![0.0_f64; folds.count()];
                let mut fold_weight = vec![0.0_f64; folds.count()];
                for &(row, w, loss) in &scored {
                    let fold = folds.fold_of(row);
                    fold_sum[fold] += w * (loss[arm] - loss[argmin]);
                    fold_weight[fold] += w;
                }
                let fold_means: Vec<f64> = fold_sum
                    .iter()
                    .zip(fold_weight.iter())
                    .filter(|&(_, &weight)| weight > 0.0)
                    .map(|(&sum, &weight)| sum / weight)
                    .collect();
                let used = fold_means.len() as f64;
                let fold_se = if used >= 2.0 {
                    let centre = fold_means.iter().sum::<f64>() / used;
                    (fold_means.iter().map(|m| (m - centre) * (m - centre)).sum::<f64>()
                        / (used - 1.0)
                        / used)
                        .sqrt()
                } else {
                    0.0
                };
                MovingLawArmScore {
                    arm: ladder[arm],
                    loss: mean_loss[arm],
                    difference,
                    row_se,
                    fold_se,
                    adequacy: adequacy[arm].clone(),
                }
            })
            .collect();
        if arms.iter().any(|score| {
            !(score.loss.is_finite() && score.row_se.is_finite() && score.fold_se.is_finite())
        }) {
            return Err(MovingLawError::NonFiniteScore(arms));
        }
        // The argmin's own difference is exactly zero and it is admissible, so the
        // scan always stops, at the argmin at the latest.
        let chosen = arms
            .iter()
            .find(|score| score.admissible() && score.difference <= score.rule_se())
            .map(|score| score.arm)
            .ok_or_else(|| MovingLawError::NoArmWithinStandardError(arms.clone()))?;
        Ok(Self {
            folds: folds.count(),
            fold_seed: folds.seed(),
            rows: scored.len(),
            fitted,
            argmin: ladder[argmin],
            chosen,
            arms,
        })
    }

    pub(crate) fn summary(&self) -> String {
        let arms = self
            .arms
            .iter()
            .map(|score| {
                let se = score.rule_se();
                let ratio = if se > 0.0 { score.difference / se } else { 0.0 };
                let screen = match &score.adequacy {
                    Some(adequacy) if !adequacy.passes() => {
                        format!("; not a candidate, its Gaussian residual fails the screen: {}", adequacy.ledger())
                    }
                    _ => String::new(),
                };
                format!(
                    "{} loss={:.4e} d={:+.3e} ({ratio:.2} se; row se={:.3e}, fold se={:.3e}{screen})",
                    score.arm.label(),
                    score.loss,
                    score.difference,
                    score.row_se,
                    score.fold_se,
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        format!(
            "cross-fitted excess anchoring loss over {} rows in {} folds: {arms}; lowest {}, \
             chosen {} (the simplest admissible arm within one paired se of the lowest \
             admissible, the larger of row- and fold-level), fitted {}",
            self.rows,
            self.folds,
            self.argmin.label(),
            self.chosen.label(),
            self.fitted.label(),
        )
    }
}

/// The folds of the certificate. Positive-weight rows are ranked by a hash of
/// their own data (every score column and every context covariate) under
/// [`MOVING_LAW_FOLD_SEED`] and dealt round-robin by rank, so the folds are
/// balanced and a row's fold follows its values; a zero-weight row, which no law
/// or loss reads, takes its hash modulo the fold count. The same rank order pairs
/// every row with a partner ([`Self::partner`]).
pub(crate) struct MovingLawFolds {
    fold: Vec<u8>,
    partner: Vec<usize>,
}

impl MovingLawFolds {
    /// The folds and partners of `scores` (one column per latent score) with the
    /// context `features`, refused below two positive-weight rows, where no row
    /// has a partner in another fold.
    pub(crate) fn from_data(
        scores: ArrayView2<'_, f64>,
        weights: &Array1<f64>,
        features: ArrayView2<'_, f64>,
    ) -> Result<Self, MovingLawError> {
        let n = scores.nrows();
        if weights.len() != n || features.nrows() != n {
            return Err(MovingLawError::LengthMismatch {
                scores: n,
                weights: weights.len(),
                features: features.nrows(),
            });
        }
        let key_of = |row: usize| {
            let mut key = MOVING_LAW_FOLD_SEED;
            for &value in scores.row(row).iter().chain(features.row(row).iter()) {
                key = gam_linalg::utils::splitmix64_hash(key ^ value.to_bits());
            }
            key
        };
        let mut fold = vec![0u8; n];
        let mut keyed = Vec::with_capacity(n);
        for row in 0..n {
            let key = key_of(row);
            if weights[row] > 0.0 {
                keyed.push((key, row));
            } else {
                fold[row] = (key % MOVING_LAW_FOLDS as u64) as u8;
            }
        }
        keyed.sort_unstable();
        let ranked: Vec<usize> = keyed.into_iter().map(|(_, row)| row).collect();
        for (rank, &row) in ranked.iter().enumerate() {
            fold[row] = (rank % MOVING_LAW_FOLDS) as u8;
        }
        if ranked.len() < 2 {
            return Err(MovingLawError::TooFewRows { rows: ranked.len() });
        }
        let mut partner = vec![ranked[0]; n];
        for (rank, &row) in ranked.iter().enumerate() {
            let next = (1..ranked.len())
                .map(|step| (rank + step) % ranked.len())
                .find(|&other| other % MOVING_LAW_FOLDS != rank % MOVING_LAW_FOLDS)
                .ok_or(MovingLawError::OneFold { rows: ranked.len() })?;
            partner[row] = ranked[next];
        }
        Ok(Self { fold, partner })
    }

    /// The row whose times a survival anchor of `row` is read at (gam#2926,
    /// gam#2949; the pairing is i2949's). A positive-weight row's partner is the
    /// positive-weight row dealt right after it, cyclically, skipping a row of its
    /// own fold, which the wrap meets only when the row count is one more than a
    /// multiple of the fold count; consecutive ranks are dealt to consecutive
    /// folds, so a partner is never in the row's fold. Its times are not the row's
    /// outcome, so `E[f_i(z_i) | t_j] = E_G[f_i]`, and a censored partner's exit
    /// time is as good an anchor as an event time. A zero-weight row, which no loss
    /// reads, is paired with the row dealt first.
    pub(crate) fn partner(&self, row: usize) -> usize {
        self.partner[row]
    }

    pub(crate) fn count(&self) -> usize {
        MOVING_LAW_FOLDS
    }

    pub(crate) fn seed(&self) -> u64 {
        MOVING_LAW_FOLD_SEED
    }

    pub(crate) fn fold_of(&self, row: usize) -> usize {
        self.fold[row] as usize
    }

    /// `weights` with the rows of `held_out` set to zero: the weights an arm's
    /// law is built on for that fold's rows.
    pub(crate) fn training_weights(&self, held_out: usize, weights: &Array1<f64>) -> Array1<f64> {
        Array1::from_iter(
            weights
                .iter()
                .zip(self.fold.iter())
                .map(|(&w, &fold)| if fold as usize == held_out { 0.0 } else { w }),
        )
    }

    /// The positive-weight rows of each fold, ascending.
    fn scored_rows(&self, weights: &Array1<f64>) -> Vec<Vec<usize>> {
        let mut rows = vec![Vec::new(); MOVING_LAW_FOLDS];
        for (row, &fold) in self.fold.iter().enumerate() {
            if weights[row] > 0.0 {
                rows[fold as usize].push(row);
            }
        }
        rows
    }
}

/// Why the moving-law certificate could not be built or taken (gam#2926). A fit
/// raises it under [`MovingLawError::category`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum MovingLawError {
    /// A held-out arm of one fold could not be fitted.
    Refused(LatentLawRefusal),
    /// A law the candidates are built from was refused by its builder.
    Law { what: &'static str, reason: String },
    /// The scores, weights and context covariates disagree on the row count.
    LengthMismatch { scores: usize, weights: usize, features: usize },
    /// Too few positive-weight rows for a row to have a partner in another fold.
    TooFewRows { rows: usize },
    /// Every positive-weight row falls in one fold.
    OneFold { rows: usize },
    /// The fitted arm is not on the ladder.
    FittedArmOffLadder(MovingLawArm),
    /// The fitted arm anchors on a Gaussian residual its adequacy screen rejects.
    FittedArmInadmissible(MovingLawArm),
    /// A row's losses, or two of its anchors, carry another number of arms than
    /// the ladder (`row` is `None` inside one row's anchors).
    ArmCount { row: Option<usize>, expected: usize, found: usize },
    /// A mean loss or one of its standard errors is not finite.
    NonFiniteScore(Vec<MovingLawArmScore>),
    /// No arm is within its standard error of the lowest, not even the lowest.
    NoArmWithinStandardError(Vec<MovingLawArmScore>),
    /// A location-scale arm was asked for where neither conditional moment moved.
    NoLocationScale(MovingLawArm),
    /// The location-scale empirical arm has no residual law.
    NoResidualLaw,
    /// The fit's anchor program failed at a row the certificate scores.
    AnchorProgram { reason: String },
    /// A route the certificate does not evaluate.
    Unsupported { what: &'static str },
    /// An anchor's probability could not be taken.
    Probability(AnchorProbabilityFailure),
    /// An anchor's log probabilities are NaN or `+∞`, so no probabilities at all.
    NotAProbability(MovingLawAnchor),
}

impl MovingLawError {
    /// The failure category a fit raises this under (gam#2937): what the caller
    /// supplied or asked for is input, the certificate's arithmetic is numerical,
    /// and a disagreement of the certificate's own parts is an invariant.
    pub(crate) fn category(&self) -> gam_problem::FailureCategory {
        use gam_problem::FailureCategory;
        match self {
            Self::Refused(_)
            | Self::Law { .. }
            | Self::TooFewRows { .. }
            | Self::OneFold { .. }
            | Self::Unsupported { .. } => FailureCategory::Input,
            Self::AnchorProgram { .. }
            | Self::Probability(_)
            | Self::NotAProbability(_)
            | Self::NonFiniteScore(_) => FailureCategory::Numerical,
            Self::LengthMismatch { .. }
            | Self::FittedArmOffLadder(_)
            | Self::FittedArmInadmissible(_)
            | Self::ArmCount { .. }
            | Self::NoArmWithinStandardError(_)
            | Self::NoLocationScale(_)
            | Self::NoResidualLaw => FailureCategory::Invariant,
        }
    }
}

impl std::fmt::Display for MovingLawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(refusal) => write!(f, "{refusal}"),
            Self::Law { what, reason } => write!(f, "moving-law certificate: the {what}: {reason}"),
            Self::LengthMismatch { scores, weights, features } => write!(
                f,
                "moving-law folds length mismatch: scores={scores}, weights={weights}, \
                 features={features}"
            ),
            Self::TooFewRows { rows } => write!(
                f,
                "moving-law certificate: {rows} positive-weight rows, and it needs two: a paired \
                 standard error, and a partner of another fold for every row"
            ),
            Self::OneFold { rows } => write!(
                f,
                "moving-law certificate: every one of the {rows} positive-weight rows falls in one \
                 fold"
            ),
            Self::FittedArmOffLadder(arm) => write!(
                f,
                "moving-law certificate: the fitted {} arm is not on the ladder",
                arm.label()
            ),
            Self::FittedArmInadmissible(arm) => write!(
                f,
                "moving-law certificate: the fitted {} arm anchors on a Gaussian residual its \
                 adequacy screen rejects",
                arm.label()
            ),
            Self::ArmCount { row: Some(row), expected, found } => write!(
                f,
                "moving-law certificate: row {row} carries {found} losses for {expected} arms"
            ),
            Self::ArmCount { row: None, expected, found } => write!(
                f,
                "moving-law certificate: a row's anchors carry {expected} and {found} arms"
            ),
            Self::NonFiniteScore(arms) => {
                write!(f, "moving-law certificate produced a non-finite score: {arms:?}")
            }
            Self::NoArmWithinStandardError(arms) => write!(
                f,
                "moving-law certificate: no arm is within its standard error of the lowest, not \
                 even the lowest itself: {arms:?}"
            ),
            Self::NoLocationScale(arm) => write!(
                f,
                "moving-law certificate: the {} arm needs a location-scale calibration, and \
                 neither conditional moment moved",
                arm.label()
            ),
            Self::NoResidualLaw => write!(
                f,
                "moving-law certificate: the location-scale empirical arm has no residual law"
            ),
            Self::AnchorProgram { reason } => {
                write!(f, "moving-law certificate: the fit's anchor program failed: {reason}")
            }
            Self::Unsupported { what } => write!(f, "moving-law certificate: {what}"),
            Self::Probability(failure) => write!(f, "moving-law certificate: {failure}"),
            Self::NotAProbability(anchor) => write!(
                f,
                "moving-law certificate: an anchor's log probabilities are not probabilities \
                 (own={:?}, arms={:?})",
                anchor.own, anchor.arms
            ),
        }
    }
}

impl From<AnchorProbabilityFailure> for MovingLawError {
    fn from(failure: AnchorProbabilityFailure) -> Self {
        Self::Probability(failure)
    }
}

impl From<MovingLawError> for crate::fit_orchestration::FitFailure {
    fn from(error: MovingLawError) -> Self {
        Self::raised(error.category(), error.to_string())
    }
}

/// A failure of [`log_anchor_probabilities`]' own arithmetic.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum AnchorProbabilityFailure {
    /// The index or the log weight of a term is NaN.
    NanTerm { term: usize },
    /// No term carries weight, or every term's probability is zero.
    NoSupport(gam_math::categorical::CategoricalError),
}

impl std::fmt::Display for AnchorProbabilityFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NanTerm { term } => write!(f, "an anchor's term {term} is NaN"),
            Self::NoSupport(error) => write!(f, "an anchor's probability has no support: {error}"),
        }
    }
}

/// `(ln P, ln(1 − P))` of one anchor over its terms `(ln w_k, η_k)`, the one
/// computation every route's certificate scores its anchors with (gam#2926):
/// `P = Σ_k w_k Φ(η_k) / Σ_k w_k` for `η_k` the anchor's probit index of its event
/// at term `k`, and `1 − P` the same over `Φ(−η_k)`. A finite law's terms are its
/// nodes; an exact Gaussian arm and a row's own score are one term `(0, η)`, and a
/// score vector's node is one term whatever its dimension. Both sums are taken in
/// log space in one pass, so a tail far below `f64`'s resolution of the other keeps
/// its logarithm; the larger probability is then one less the smaller,
/// `ln(1 − s) = ln1p(−s)`, so its logarithm keeps the smaller's full relative
/// precision rather than a rounding of `ln 1`. A failing term's error passes
/// through unchanged.
pub(crate) fn log_anchor_probabilities<E: From<AnchorProbabilityFailure>>(
    terms: impl IntoIterator<Item = Result<(f64, f64), E>>,
) -> Result<(f64, f64), E> {
    let mut mass = Vec::new();
    let mut event = Vec::new();
    let mut complement = Vec::new();
    for (term, entry) in terms.into_iter().enumerate() {
        let (log_weight, eta) = entry?;
        if log_weight.is_nan() || eta.is_nan() {
            return Err(AnchorProbabilityFailure::NanTerm { term }.into());
        }
        mass.push(log_weight);
        event.push(log_weight + normal_logcdf(eta));
        complement.push(log_weight + normal_logcdf(-eta));
    }
    let sum = |terms: &[f64]| {
        gam_math::categorical::log_sum_exp(terms).map_err(AnchorProbabilityFailure::NoSupport)
    };
    let total = sum(&mass)?;
    let (event, complement) = (sum(&event)? - total, sum(&complement)? - total);
    Ok(if event <= complement {
        (event, (-event.exp()).ln_1p())
    } else {
        ((-complement.exp()).ln_1p(), complement)
    })
}

/// [`log_anchor_probabilities`] over a one-score finite law, with `eta` the
/// anchor's probit index at a node; zero-weight nodes carry no term.
pub(crate) fn log_grid_anchor_probabilities<E: From<AnchorProbabilityFailure>>(
    law: &EmpiricalZGrid,
    mut eta: impl FnMut(f64) -> Result<f64, E>,
) -> Result<(f64, f64), E> {
    log_anchor_probabilities(
        law.pairs()
            .filter(|&(_, weight)| weight > 0.0)
            .map(|(node, weight)| Ok((weight.ln(), eta(node)?))),
    )
}

/// One anchor of one row in the certificate pass: `(ln P, ln(1 − P))` of its
/// event under each arm's held-out law, in ladder order, and `(ln y, ln(1 − y))`
/// at the row's own score, each from [`log_anchor_probabilities`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MovingLawAnchor {
    pub(crate) arms: Vec<(f64, f64)>,
    pub(crate) own: (f64, f64),
}

/// A row's cross-entropy `ℓ` under each arm, summed over its anchors; `None` when
/// it has none.
pub(crate) fn moving_law_row_losses(
    anchors: &[MovingLawAnchor],
) -> Result<Option<Vec<f64>>, MovingLawError> {
    // A logarithm of a probability is at most a rounding above zero; NaN or +∞ is
    // no probability at all.
    let valid = |(event, complement): (f64, f64)| {
        !(event.is_nan() || complement.is_nan())
            && event < f64::INFINITY
            && complement < f64::INFINITY
    };
    // `y·ln P` with `0·ln 0 = 0`: a probability the row's score gives no weight
    // costs nothing, whatever the arm's logarithm.
    let cross = |weight: f64, log_probability: f64| {
        if weight > 0.0 { weight * log_probability } else { 0.0 }
    };
    let mut losses: Option<Vec<f64>> = None;
    for anchor in anchors {
        if !(valid(anchor.own) && anchor.arms.iter().all(|&arm| valid(arm))) {
            return Err(MovingLawError::NotAProbability(anchor.clone()));
        }
        let row = losses.get_or_insert_with(|| vec![0.0; anchor.arms.len()]);
        if row.len() != anchor.arms.len() {
            return Err(MovingLawError::ArmCount {
                row: None,
                expected: row.len(),
                found: anchor.arms.len(),
            });
        }
        let (event, complement) = (anchor.own.0.exp(), anchor.own.1.exp());
        for (total, &(log_event, log_complement)) in row.iter_mut().zip(anchor.arms.iter()) {
            *total -= cross(event, log_event) + cross(complement, log_complement);
        }
    }
    Ok(losses)
}

/// The location-scale arms' laws built without one fold's rows.
struct FoldLocationScale {
    /// The fold's standardised residual law of `ζ_f`, the law the location-scale
    /// empirical arm anchors on.
    residual: EmpiricalZGrid,
}

/// The arms' laws built without one fold's rows, kept only for that fold's rows.
struct FoldLaws {
    /// `Some` exactly when the full-data fit is location-scale.
    location_scale: Option<FoldLocationScale>,
    /// The fold's pooled law of the score on its own axis.
    pooled: EmpiricalZGrid,
    /// The fold's local law at the full-data law's context count, read at the
    /// fold's rows.
    local: estimated_latent_law::HeldOutLocalLaw,
}

/// The arms of one latent score (gam#2926): the fitted arm's axis, the folds, the
/// held-out laws, and the full-data law of each arm a re-solve can anchor on.
/// Fit-time only; the certificate is what persists.
pub(crate) struct MovingLawCandidates {
    evidence: ConditionalLawEvidence,
    /// The adequacy screen of the Gaussian arm's residual, the score as given.
    gaussian_screen: LatentNormalAdequacy,
    /// The adequacy screen of the location-scale Gaussian arm's residual `ζ`;
    /// `None` without a calibration.
    location_scale_screen: Option<LatentNormalAdequacy>,
    /// The full-data calibration; `Some` makes the fitted arm location-scale
    /// Gaussian on the `ζ` axis.
    calibration: Option<LatentZConditionalCalibration>,
    /// Per row, `(m(a_i), √v(a_i))` under `calibration`: the fitted axis is
    /// `ζ = (z − m)/√v`. Empty without a calibration.
    location: Vec<(f64, f64)>,
    /// Per row, `(m_f(a_i), √v_f(a_i))` under its own fold's calibration. Empty
    /// without a calibration.
    fold_location: Vec<(f64, f64)>,
    z: Array1<f64>,
    folds: MovingLawFolds,
    /// Per row, its position among its fold's scored rows.
    position: Vec<usize>,
    fold_laws: Vec<FoldLaws>,
    gauss_hermite: EmpiricalZGrid,
    /// The full-data location-scale empirical arm: the residual law of `ζ` and
    /// its compression record.
    residual: Option<(
        LatentMeasureKind,
        empirical_measure_sensitivity::EmpiricalZGridBuild,
    )>,
    pooled: EmpiricalZGrid,
    local: LatentMeasureKind,
    contexts: usize,
}

fn probabilists_gauss_hermite(nodes: usize) -> Result<EmpiricalZGrid, MovingLawError> {
    let law = |reason: String| MovingLawError::Law { what: "Gauss-Hermite law", reason };
    let rule =
        gam_math::quadrature::gauss_hermite_rule(nodes).map_err(|error| law(error.to_string()))?;
    let root_pi = std::f64::consts::PI.sqrt();
    EmpiricalZGrid::new(
        rule.nodes.iter().map(|&x| std::f64::consts::SQRT_2 * x).collect(),
        rule.weights.iter().map(|&w| w / root_pi).collect(),
        "moving-law certificate Gauss-Hermite law",
    )
    .map_err(law)
}

fn location_at(
    calibration: &LatentZConditionalCalibration,
    conditioning: ArrayView2<'_, f64>,
    row: usize,
) -> (f64, f64) {
    let a_row = conditioning.row(row);
    (
        calibration.conditional_mean(a_row),
        calibration.conditional_var(a_row).sqrt(),
    )
}

/// The one finite law a global builder returned.
fn global_grid(
    kind: LatentMeasureKind,
    what: &'static str,
) -> Result<EmpiricalZGrid, MovingLawError> {
    match kind {
        LatentMeasureKind::GlobalEmpirical { grid } => Ok(grid),
        LatentMeasureKind::StandardNormal | LatentMeasureKind::LocalEmpirical { .. } => {
            Err(MovingLawError::Law {
                what,
                reason: "the global empirical builder returned another kind of law".to_string(),
            })
        }
    }
}

impl MovingLawCandidates {
    /// Build the candidates on the full data and on every fold. `context` names
    /// the caller in a refusal.
    pub(crate) fn build(
        z: &Array1<f64>,
        weights: &Array1<f64>,
        conditioning: ArrayView2<'_, f64>,
        local_context: &estimated_latent_law::LocalLawContext<'_>,
        grid_size: usize,
        policy: &LatentZPolicy,
        evidence: ConditionalLawEvidence,
        context: &str,
    ) -> Result<Self, MovingLawError> {
        let n = z.len();
        let law = |what: &'static str| move |reason: String| MovingLawError::Law { what, reason };
        let calibration = fit_conditional_latent_calibration_if_needed(z, weights, conditioning)
            .map_err(law("location-scale calibration"))?;
        let location: Vec<(f64, f64)> = calibration
            .as_ref()
            .map(|cal| (0..n).map(|row| location_at(cal, conditioning, row)).collect())
            .unwrap_or_default();
        let gaussian_screen =
            latent_z_normal_adequacy(z, weights, policy).map_err(law("score adequacy screen"))?;
        let (residual, location_scale_screen) = match &calibration {
            Some(cal) => {
                let zeta = FittedLatentScoreMap::conditional_only(cal)
                    .calibrate(z.view(), Some(conditioning))
                    .map_err(law("location-scale score"))?;
                (
                    Some(
                        build_global_empirical_latent_measure(&zeta, weights, grid_size)
                            .map_err(law("location-scale residual law"))?,
                    ),
                    Some(
                        latent_z_normal_adequacy(&zeta, weights, policy)
                            .map_err(law("location-scale residual adequacy screen"))?,
                    ),
                )
            }
            None => (None, None),
        };
        let pooled = estimated_latent_law::build_empirical_law_on_own_axis(
            z.view(),
            weights.view(),
            grid_size,
            "moving-law pooled latent law",
        )
        .map_err(law("pooled law"))?;
        let local_parts =
            estimated_latent_law::local_law_parts(z, weights, local_context, grid_size, None)
                .map_err(law("local law"))?;
        let contexts = local_parts.contexts();
        let local = local_parts.into_kind().map_err(law("local law"))?;

        let folds = MovingLawFolds::from_data(
            z.view().insert_axis(ndarray::Axis(1)),
            weights,
            local_context.features,
        )?;
        let scored_rows = folds.scored_rows(weights);
        let mut position = vec![usize::MAX; n];
        for rows in &scored_rows {
            for (index, &row) in rows.iter().enumerate() {
                position[row] = index;
            }
        }
        let refusal = |fold: usize, arm: MovingLawArm, reason: String| {
            MovingLawError::Refused(LatentLawRefusal::MovingLawFoldUnfittable {
                context: context.to_string(),
                fold,
                arm,
                reason,
            })
        };
        let mut fold_location = if calibration.is_some() {
            vec![(0.0, 1.0); n]
        } else {
            Vec::new()
        };
        let mut fold_laws = Vec::with_capacity(folds.count());
        for (held_out, rows) in scored_rows.iter().enumerate() {
            let fold_weights = folds.training_weights(held_out, weights);
            // The full fit's structure, with no gate: the mean always, the
            // variance exactly when the full fit has one.
            let location_scale = match &calibration {
                Some(full) => {
                    let arm = MovingLawArm::LocationScaleGaussian;
                    let cal = fit_conditional_latent_calibration(
                        z,
                        &fold_weights,
                        conditioning,
                        !full.var_coeffs.is_empty(),
                    )
                    .map_err(|reason| refusal(held_out, arm, reason))?;
                    for &row in rows {
                        fold_location[row] = location_at(&cal, conditioning, row);
                    }
                    let zeta = FittedLatentScoreMap::conditional_only(&cal)
                        .calibrate(z.view(), Some(conditioning))
                        .map_err(|reason| refusal(held_out, arm, reason))?;
                    let (kind, _) =
                        build_global_empirical_latent_measure(&zeta, &fold_weights, grid_size)
                            .map_err(|reason| {
                                refusal(held_out, MovingLawArm::LocationScaleEmpirical, reason)
                            })?;
                    Some(FoldLocationScale {
                        residual: global_grid(kind, "held-out residual law")?,
                    })
                }
                None => None,
            };
            let pooled = estimated_latent_law::build_empirical_law_on_own_axis(
                z.view(),
                fold_weights.view(),
                grid_size,
                "moving-law held-out pooled latent law",
            )
            .map_err(|reason| refusal(held_out, MovingLawArm::PooledEmpirical, reason))?;
            let local = estimated_latent_law::local_law_parts(
                z,
                &fold_weights,
                local_context,
                grid_size,
                Some(contexts),
            )
            .and_then(|parts| parts.held_out(rows))
            .map_err(|reason| refusal(held_out, MovingLawArm::Local, reason))?;
            fold_laws.push(FoldLaws {
                location_scale,
                pooled,
                local,
            });
        }
        Ok(Self {
            evidence,
            gaussian_screen,
            location_scale_screen,
            calibration,
            location,
            fold_location,
            z: z.clone(),
            folds,
            position,
            fold_laws,
            gauss_hermite: probabilists_gauss_hermite(MOVING_LAW_GAUSS_HERMITE_NODES)?,
            residual,
            pooled,
            local,
            contexts,
        })
    }

    /// The arms, simplest first.
    pub(crate) fn arms(&self) -> &'static [MovingLawArm] {
        if self.calibration.is_some() {
            &[
                MovingLawArm::Gaussian,
                MovingLawArm::LocationScaleGaussian,
                MovingLawArm::LocationScaleEmpirical,
                MovingLawArm::Local,
            ]
        } else {
            &[
                MovingLawArm::Gaussian,
                MovingLawArm::PooledEmpirical,
                MovingLawArm::Local,
            ]
        }
    }

    /// The adequacy screen of the Gaussian residual `arm` anchors on: the score for
    /// the Gaussian arm, `ζ` for the location-scale Gaussian arm, `None` for an arm
    /// on an estimated law.
    pub(crate) fn screen_of(&self, arm: MovingLawArm) -> Option<LatentNormalAdequacy> {
        match arm {
            MovingLawArm::Gaussian => Some(self.gaussian_screen.clone()),
            MovingLawArm::LocationScaleGaussian => self.location_scale_screen.clone(),
            MovingLawArm::PooledEmpirical
            | MovingLawArm::LocationScaleEmpirical
            | MovingLawArm::Local => None,
        }
    }

    /// The arm the default fits first: where a conditional moment moves, the
    /// location-scale Gaussian law if its residual `ζ` passes the adequacy screen
    /// and the location-scale law on `ζ`'s estimated law otherwise; where neither
    /// moves, the Gaussian law if the score passes the screen and the pooled
    /// estimated law otherwise. It is the simplest admissible arm of its structure,
    /// so the certificate never has to re-solve off an arm the data reject.
    pub(crate) fn fitted_arm(&self) -> MovingLawArm {
        let (gaussian, estimated) = if self.calibration.is_some() {
            (MovingLawArm::LocationScaleGaussian, MovingLawArm::LocationScaleEmpirical)
        } else {
            (MovingLawArm::Gaussian, MovingLawArm::PooledEmpirical)
        };
        if self
            .screen_of(gaussian)
            .as_ref()
            .is_some_and(LatentNormalAdequacy::passes)
        {
            gaussian
        } else {
            estimated
        }
    }

    /// The fitted arm's axis at a row: `(m, √v)` with the fitted score
    /// `(z − m)/√v`.
    fn fitted_location(&self, row: usize) -> (f64, f64) {
        if self.calibration.is_some() {
            self.location[row]
        } else {
            (0.0, 1.0)
        }
    }

    /// The scored row whose times a survival anchor of `row` is read at
    /// ([`MovingLawFolds::partner`]).
    pub(crate) fn partner(&self, row: usize) -> usize {
        self.folds.partner(row)
    }

    /// The row's own score on the fitted axis.
    pub(crate) fn own_score(&self, row: usize) -> f64 {
        let (m, s) = self.fitted_location(row);
        (self.z[row] - m) / s
    }

    /// Each arm's law of a scored row's score, built without the row's fold, on
    /// the fitted axis, in [`Self::arms`] order.
    pub(crate) fn row_laws(&self, row: usize) -> Result<Vec<EmpiricalZGrid>, MovingLawError> {
        let (m, s) = self.fitted_location(row);
        let fold = &self.fold_laws[self.folds.fold_of(row)];
        let on_fitted_axis = |nodes: &[f64], weights: &[f64], shift: f64, scale: f64| {
            EmpiricalZGrid::new(
                nodes.iter().map(|&u| (shift + scale * u - m) / s).collect(),
                weights.to_vec(),
                "moving-law held-out latent law",
            )
            .map_err(|reason| MovingLawError::Law { what: "held-out law", reason })
        };
        let location_scale = |arm: MovingLawArm| {
            match (&fold.location_scale, self.fold_location.get(row)) {
                (Some(laws), Some(&location)) => Ok((laws, location)),
                _ => Err(MovingLawError::NoLocationScale(arm)),
            }
        };
        let gh = &self.gauss_hermite;
        self.arms()
            .iter()
            .map(|arm| match arm {
                MovingLawArm::Gaussian => on_fitted_axis(&gh.nodes, &gh.weights, 0.0, 1.0),
                MovingLawArm::PooledEmpirical => {
                    on_fitted_axis(&fold.pooled.nodes, &fold.pooled.weights, 0.0, 1.0)
                }
                MovingLawArm::LocationScaleGaussian => {
                    let (_, (shift, scale)) = location_scale(*arm)?;
                    on_fitted_axis(&gh.nodes, &gh.weights, shift, scale)
                }
                MovingLawArm::LocationScaleEmpirical => {
                    let (laws, (shift, scale)) = location_scale(*arm)?;
                    on_fitted_axis(&laws.residual.nodes, &laws.residual.weights, shift, scale)
                }
                MovingLawArm::Local => {
                    let grid = fold.local.grid(self.position[row]).map_err(|reason| {
                        MovingLawError::Law { what: "held-out local law", reason }
                    })?;
                    on_fitted_axis(&grid.nodes, &grid.weights, 0.0, 1.0)
                }
            })
            .collect()
    }

    /// Apply the rule to every row's losses, taken at the fitted arm.
    pub(crate) fn certify(
        &self,
        weights: &Array1<f64>,
        losses: &[Option<Vec<f64>>],
    ) -> Result<MovingLawCertificate, MovingLawError> {
        let screens: Vec<Option<LatentNormalAdequacy>> =
            self.arms().iter().map(|&arm| self.screen_of(arm)).collect();
        MovingLawCertificate::from_row_losses(
            self.arms(),
            &screens,
            self.fitted_arm(),
            &self.folds,
            weights,
            losses,
        )
    }

    /// The law a fit on `arm` anchors on, recorded with `certificate`
    /// (`None` for the provisional fit the certificate is taken at).
    pub(crate) fn decision_for(
        &self,
        arm: MovingLawArm,
        certificate: Option<MovingLawCertificate>,
    ) -> Result<LatentMeasureDecision, MovingLawError> {
        let consumed = LatentLawConsumed::EstimatedMovingLaw {
            evidence: self.evidence.clone(),
            arm,
            contexts: self.contexts,
            certificate,
            uncertified: None,
        };
        let location_scale = || {
            self.calibration
                .clone()
                .map(LatentMeasureCalibration::ConditionalLocationScale)
                .ok_or(MovingLawError::NoLocationScale(arm))
        };
        let (kind, calibration, empirical_build) = match arm {
            MovingLawArm::Gaussian => (
                LatentMeasureKind::StandardNormal,
                LatentMeasureCalibration::None,
                None,
            ),
            MovingLawArm::PooledEmpirical => (
                LatentMeasureKind::GlobalEmpirical {
                    grid: self.pooled.clone(),
                },
                LatentMeasureCalibration::None,
                None,
            ),
            MovingLawArm::LocationScaleGaussian => (
                LatentMeasureKind::StandardNormal,
                location_scale()?,
                None,
            ),
            MovingLawArm::LocationScaleEmpirical => {
                let (kind, build) = self.residual.clone().ok_or(MovingLawError::NoResidualLaw)?;
                (kind, location_scale()?, Some(build))
            }
            MovingLawArm::Local => (self.local.clone(), LatentMeasureCalibration::None, None),
        };
        Ok(LatentMeasureDecision {
            kind,
            calibration,
            empirical_build,
            certificate_law: None,
            consumed,
            moving_law: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row's fold follows its own data: reversing the table reverses the folds
    /// with it, and the positive-weight rows are dealt evenly over the folds.
    #[test]
    fn folds_follow_the_rows_and_deal_evenly_2926() {
        let n = 1003;
        let z = Array2::from_shape_fn((n, 1), |(row, _)| 2.0 * (row as f64).sin());
        let features = Array2::from_shape_fn((n, 2), |(row, col)| ((row * (col + 2)) as f64).cos());
        let mut weights = Array1::<f64>::ones(n);
        weights[5] = 0.0;
        weights[500] = 0.0;
        let folds = MovingLawFolds::from_data(z.view(), &weights, features.view()).expect("folds");
        let mut counts = [0usize; MOVING_LAW_FOLDS];
        for row in (0..n).filter(|&row| weights[row] > 0.0) {
            counts[folds.fold_of(row)] += 1;
        }
        let (fewest, most) = (counts.iter().min().unwrap(), counts.iter().max().unwrap());
        assert!(most - fewest <= 1, "folds must be dealt evenly; counts {counts:?}");

        let reversed: Vec<usize> = (0..n).rev().collect();
        let folds_reversed = MovingLawFolds::from_data(
            z.select(ndarray::Axis(0), &reversed).view(),
            &Array1::from_iter(reversed.iter().map(|&row| weights[row])),
            features.select(ndarray::Axis(0), &reversed).view(),
        )
        .expect("folds");
        for (row, &source) in reversed.iter().enumerate() {
            assert_eq!(
                folds_reversed.fold_of(row),
                folds.fold_of(source),
                "row {source} must keep its fold when the table is reversed"
            );
        }
    }

    /// Every scored row's partner is another scored row in another fold, the pairing
    /// is a function of the data, and a zero-weight row is paired with the row dealt
    /// first. The 1 001 scored rows are one more than a multiple of the fold count,
    /// so the cyclic wrap meets its own fold and must skip it.
    #[test]
    fn the_partner_pairing_is_deterministic_and_crosses_folds_2926() {
        let n = 1003;
        let z = Array2::from_shape_fn((n, 1), |(row, _)| 2.0 * (row as f64).sin());
        let features = Array2::from_shape_fn((n, 2), |(row, col)| ((row * (col + 2)) as f64).cos());
        let mut weights = Array1::<f64>::ones(n);
        weights[5] = 0.0;
        weights[500] = 0.0;
        let folds = MovingLawFolds::from_data(z.view(), &weights, features.view()).expect("folds");
        let again = MovingLawFolds::from_data(z.view(), &weights, features.view()).expect("folds");
        let reversed: Vec<usize> = (0..n).rev().collect();
        let folds_reversed = MovingLawFolds::from_data(
            z.select(ndarray::Axis(0), &reversed).view(),
            &Array1::from_iter(reversed.iter().map(|&row| weights[row])),
            features.select(ndarray::Axis(0), &reversed).view(),
        )
        .expect("folds");
        for row in (0..n).filter(|&row| weights[row] > 0.0) {
            let partner = folds.partner(row);
            assert_ne!(partner, row, "row {row} must not be its own partner");
            assert!(weights[partner] > 0.0, "row {row}'s partner {partner} must be scored");
            assert_ne!(
                folds.fold_of(partner),
                folds.fold_of(row),
                "row {row}'s partner {partner} must be outside its fold"
            );
            assert_eq!(again.partner(row), partner, "the pairing must repeat");
            assert_eq!(
                reversed[folds_reversed.partner(n - 1 - row)],
                partner,
                "row {row}'s partner must follow the data, not the table order"
            );
        }
        assert_eq!(
            folds.partner(5),
            folds.partner(500),
            "every zero-weight row is paired with the row dealt first"
        );
        assert!(weights[folds.partner(5)] > 0.0, "a zero-weight row's partner is scored");
    }

    const LADDER: [MovingLawArm; 4] = [
        MovingLawArm::Gaussian,
        MovingLawArm::LocationScaleGaussian,
        MovingLawArm::LocationScaleEmpirical,
        MovingLawArm::Local,
    ];

    /// Per-row losses whose paired differences are exact: arm `a` scores
    /// `μ_a + ε_i + s_a·(−1)^i` with `s = (0, 1, 2, 3)`, so its difference to the
    /// local arm has mean `μ_a` and row-level standard error `c/√(n−1)`,
    /// `c = 3 − s_a`. `fold_of` places the rows: when a fold holds both parities
    /// its mean difference is `μ_a` and the fold-level error vanishes; when it
    /// holds one parity its mean is `μ_a ± c` and the fold-level error is `c/3`,
    /// the case of rows that share their error within a fold.
    fn ladder(
        means: [f64; 4],
        fold_of: impl Fn(usize) -> usize,
    ) -> (MovingLawFolds, Array1<f64>, Vec<Option<Vec<f64>>>) {
        let n = 1000;
        let spreads = [0.0, 1.0, 2.0, 3.0];
        let folds = MovingLawFolds {
            fold: (0..n).map(|row| fold_of(row) as u8).collect(),
            partner: (0..n).map(|row| (row + 1) % n).collect(),
        };
        let losses = (0..n)
            .map(|row| {
                let shared = 0.25 * (row as f64).cos();
                let sign = if row % 2 == 0 { 1.0 } else { -1.0 };
                Some(
                    (0..4)
                        .map(|arm| means[arm] + shared + spreads[arm] * sign)
                        .collect(),
                )
            })
            .collect();
        (folds, Array1::ones(n), losses)
    }

    fn certify(
        (folds, weights, losses): (MovingLawFolds, Array1<f64>, Vec<Option<Vec<f64>>>),
    ) -> MovingLawCertificate {
        MovingLawCertificate::from_row_losses(
            &LADDER,
            &[const { None }; LADDER.len()],
            MovingLawArm::LocationScaleGaussian,
            &folds,
            &weights,
            &losses,
        )
        .expect("certificate")
    }

    /// The standard-normal adequacy screen of `n` draws: normal quantiles pass it,
    /// and their squares (a chi-square with one degree of freedom, standardized)
    /// fail it on every shape clause.
    fn screen(gaussian: bool) -> LatentNormalAdequacy {
        let n = 2000;
        let quantile = |k: usize| {
            crate::probability::standard_normal_quantile((k as f64 + 0.5) / n as f64)
                .expect("a probability inside (0, 1)")
        };
        let z = Array1::from_iter((0..n).map(|k| {
            let x = quantile(k);
            if gaussian { x } else { (x * x - 1.0) / 2.0_f64.sqrt() }
        }));
        latent_z_normal_adequacy(&z, &Array1::ones(n), &LatentZPolicy::default())
            .expect("the screen measures a finite sample")
    }

    /// gam#2926: an arm whose Gaussian residual the adequacy screen rejects is
    /// scored and recorded but never chosen, and never the argmin. On the losses
    /// where the rule takes the location-scale Gaussian arm (0.79 se behind the
    /// local arm), a failing `ζ` screen makes it take the next admissible arm; the
    /// Gaussian arm, rejected on the score, is passed over even at the lowest loss;
    /// and a fit solved on a rejected arm is refused by name.
    #[test]
    fn the_rule_never_chooses_an_arm_whose_gaussian_residual_the_screen_rejects_2926() {
        let (passing, failing) = (screen(true), screen(false));
        assert!(passing.passes() && !failing.passes(), "{} | {}", passing.ledger(), failing.ledger());
        let mixed = |row: usize| (row / 2) % MOVING_LAW_FOLDS;
        let rule = |screens: [Option<LatentNormalAdequacy>; 4], means: [f64; 4], fitted| {
            let (folds, weights, losses) = ladder(means, mixed);
            MovingLawCertificate::from_row_losses(&LADDER, &screens, fitted, &folds, &weights, &losses)
        };
        let admitted = rule(
            [Some(passing.clone()), Some(passing.clone()), None, None],
            [0.5, 0.05, 0.02, 0.0],
            MovingLawArm::LocationScaleEmpirical,
        )
        .expect("certificate");
        assert_eq!(admitted.chosen, MovingLawArm::LocationScaleGaussian);

        let rejected = rule(
            [Some(passing.clone()), Some(failing.clone()), None, None],
            [0.5, 0.05, 0.02, 0.0],
            MovingLawArm::LocationScaleEmpirical,
        )
        .expect("certificate");
        assert_eq!(rejected.chosen, MovingLawArm::LocationScaleEmpirical, "{}", rejected.summary());
        assert!(!rejected.arms[1].admissible() && rejected.arms[1].adequacy.is_some());
        assert_eq!(rejected.arms[1].difference, admitted.arms[1].difference);

        let lowest_rejected = rule(
            [Some(failing.clone()), Some(passing.clone()), None, None],
            [-1.0, 0.05, 0.02, 0.0],
            MovingLawArm::LocationScaleGaussian,
        )
        .expect("certificate");
        assert_eq!(lowest_rejected.argmin, MovingLawArm::Local, "{}", lowest_rejected.summary());
        assert_eq!(lowest_rejected.chosen, MovingLawArm::LocationScaleGaussian);

        assert_eq!(
            rule(
                [Some(passing), Some(failing), None, None],
                [0.5, 0.05, 0.02, 0.0],
                MovingLawArm::LocationScaleGaussian,
            ),
            Err(MovingLawError::FittedArmInadmissible(MovingLawArm::LocationScaleGaussian))
        );
    }

    /// An arm score saved before the screen was recorded loads with none, and reads
    /// as admissible, the rule it was chosen by; a current one round-trips with its
    /// screen.
    #[test]
    fn an_arm_score_saved_before_its_screen_loads_as_admissible_2926() {
        let score = MovingLawArmScore {
            arm: MovingLawArm::LocationScaleGaussian,
            loss: 0.65,
            difference: 4.7e-6,
            row_se: 8.0e-6,
            fold_se: 7.9e-6,
            adequacy: Some(screen(false)),
        };
        let text = serde_json::to_string(&score).expect("serialize");
        let reloaded: MovingLawArmScore = serde_json::from_str(&text).expect("round-trip");
        assert_eq!(reloaded, score);
        assert!(!reloaded.admissible());
        let mut older: serde_json::Value = serde_json::to_value(&score).expect("serialize");
        older
            .as_object_mut()
            .expect("a score is an object")
            .remove("adequacy")
            .expect("the screen is written");
        let loaded: MovingLawArmScore = serde_json::from_value(older).expect("an older score");
        assert!(loaded.adequacy.is_none() && loaded.admissible());
    }

    /// With fold means that carry no shared error the rule reads the row-level
    /// standard error: it takes the simplest arm within one of the lowest loss,
    /// and a more complex arm only once every simpler one is more than one behind.
    #[test]
    fn the_rule_takes_the_simplest_arm_within_one_paired_standard_error_2926() {
        let root = 999.0_f64.sqrt();
        let mixed = |row: usize| (row / 2) % MOVING_LAW_FOLDS;
        // Location-scale Gaussian 0.79 se behind the local arm: it is chosen.
        let certificate = certify(ladder([0.5, 0.05, 0.02, 0.0], mixed));
        assert_eq!(certificate.rows, 1000);
        assert_eq!(certificate.argmin, MovingLawArm::Local);
        assert_eq!(certificate.chosen, MovingLawArm::LocationScaleGaussian);
        for (score, (mean, spread)) in certificate
            .arms
            .iter()
            .zip([(0.5, 3.0), (0.05, 2.0), (0.02, 1.0), (0.0, 0.0)])
        {
            assert!((score.difference - mean).abs() < 1e-12, "{score:?}");
            assert!((score.row_se - spread / root).abs() < 1e-12, "{score:?}");
            assert!(score.fold_se < 1e-12, "{score:?}");
            assert_eq!(score.rule_se(), score.row_se.max(score.fold_se));
        }
        // Both location-scale arms 1.58 se behind: only the local arm clears.
        let certificate = certify(ladder([0.5, 0.1, 0.05, 0.0], mixed));
        assert_eq!(
            (certificate.argmin, certificate.chosen),
            (MovingLawArm::Local, MovingLawArm::Local)
        );
        // The Gaussian arm lowest: it is its own argmin and is chosen.
        let certificate = certify(ladder([-0.5, 0.1, 0.05, 0.0], mixed));
        assert_eq!(
            (certificate.argmin, certificate.chosen),
            (MovingLawArm::Gaussian, MovingLawArm::Gaussian)
        );
    }

    /// Where rows share their error within a fold, the fold-level standard error
    /// exceeds the row-level one and decides: every simpler arm is more than three
    /// row-level errors behind the local arm, but within one fold-level error.
    #[test]
    fn a_shared_fold_error_keeps_the_simpler_arm_2926() {
        let root = 999.0_f64.sqrt();
        let certificate = certify(ladder([0.5, 0.2, 0.1, 0.0], |row| row % MOVING_LAW_FOLDS));
        for (score, spread) in certificate.arms.iter().zip([3.0, 2.0, 1.0, 0.0]) {
            assert!((score.row_se - spread / root).abs() < 1e-12, "{score:?}");
            assert!((score.fold_se - spread / 3.0).abs() < 1e-12, "{score:?}");
        }
        assert!(certificate.arms[..3]
            .iter()
            .all(|score| score.difference > 3.0 * score.row_se));
        assert_eq!(certificate.argmin, MovingLawArm::Local);
        assert_eq!(certificate.chosen, MovingLawArm::Gaussian);
    }

    /// Rows with no weight or no anchor are not scored; a row's loss is the
    /// cross-entropy summed over its anchors, with `0·ln 0 = 0`, and a log
    /// probability that is no probability is refused.
    #[test]
    fn unscored_rows_and_anchors_are_skipped_2926() {
        let (folds, mut weights, mut losses) =
            ladder([0.5, 0.05, 0.02, 0.0], |row| (row / 2) % MOVING_LAW_FOLDS);
        weights[3] = 0.0;
        losses[4] = None;
        assert_eq!(certify((folds, weights, losses)).rows, 998);

        let logs = |p: f64| (p.ln(), (-p).ln_1p());
        let loss = moving_law_row_losses(&[
            MovingLawAnchor { arms: vec![logs(0.25), logs(0.5)], own: logs(0.3) },
            // The row's score makes the event certain: the arm's `ln(1 − P) = −∞`
            // is weighted by `1 − y = 0` and costs nothing.
            MovingLawAnchor { arms: vec![logs(0.9), logs(1.0)], own: logs(1.0) },
        ])
        .expect("losses")
        .expect("two anchors");
        let expected = [
            -(0.3 * 0.25_f64.ln() + 0.7 * 0.75_f64.ln()) - 0.9_f64.ln(),
            -(0.3 * 0.5_f64.ln() + 0.7 * 0.5_f64.ln()),
        ];
        for (got, want) in loss.iter().zip(expected) {
            assert!((got - want).abs() < 1e-15, "{got} vs {want}");
        }
        assert!(moving_law_row_losses(&[]).expect("losses").is_none());
        assert!(
            moving_law_row_losses(&[MovingLawAnchor {
                arms: vec![(f64::NAN, 0.0)],
                own: logs(0.5)
            }])
            .is_err()
        );
    }

    /// An anchor deep in a tail keeps its logarithm where the probability itself
    /// underflows: `P = ½Φ(−40) + ½Φ(−39)` is below the smallest `f64`, and
    /// `1 − P` is one to rounding. Where the fitted anchor's own probability is
    /// `Φ(−5) ≈ 2.9e-7`, a finite law's tail near `1e-25` is below the resolution
    /// of `Φ(−5) + r`; its cross-entropy on a row whose score reaches that tail is
    /// finite, and the heavier tail scores better there.
    #[test]
    fn anchor_probabilities_keep_a_tail_beyond_the_others_resolution_2926() {
        let law = EmpiricalZGrid { nodes: vec![0.0, 1.0], weights: vec![0.5, 0.5] };
        let at = |offset: f64| {
            log_grid_anchor_probabilities(&law, |u| Ok::<f64, AnchorProbabilityFailure>(offset + u))
                .expect("log probabilities")
        };
        let (event, complement) = at(-40.0);
        let (low, high) = (normal_logcdf(-40.0), normal_logcdf(-39.0));
        let want = high + (0.5 + 0.5 * (low - high).exp()).ln();
        assert!(normal_cdf(-39.0) == 0.0, "the fixture must underflow the linear probability");
        assert!(
            event.is_finite() && (event - want).abs() < 1e-12 * want.abs(),
            "ln P = {event} vs {want}"
        );
        assert!(complement.abs() < 1e-15, "ln(1 − P) = {complement}");

        let (heavier, thinner) = (at(-11.5), at(-12.5));
        assert!(
            heavier.0.exp() < f64::EPSILON * normal_cdf(-5.0),
            "the fixture's tail must sit below the resolution of Φ(−5) + r"
        );
        let own = (normal_logcdf(-9.0), normal_logcdf(9.0));
        let loss = moving_law_row_losses(&[MovingLawAnchor {
            arms: vec![heavier, thinner],
            own,
        }])
        .expect("losses")
        .expect("one anchor");
        assert!(loss.iter().all(|l| l.is_finite()), "{loss:?}");
        assert!(loss[0] < loss[1], "the heavier tail scores better on a row in it: {loss:?}");
    }

    /// gam#2949 at one score, both ways: on survival anchors the certificate is
    /// sound only where a row is anchored at a time its own score did not shape.
    /// `z | x` is `N(0.6·x, 0.64)`, location-scale with a Gaussian residual, and the
    /// times are simulated from the family's own model under it,
    /// `S(t | x, z) = Φ(−(α(q(t)) + b·z))`. At the true slope, anchored at its
    /// partner's time, the rule keeps the location-scale Gaussian arm, the true law.
    /// Anchored at its own time, `f_i(z_i)` is the probability-integral transform of
    /// the row's own outcome and no longer an unbiased draw of `E_G[f_i]`, and the
    /// same rule leaves the true law: the negative control.
    #[test]
    fn survival_anchors_at_a_partners_time_keep_the_true_law_and_own_times_do_not_2926() {
        const ROWS: usize = 12_000;
        const LEVEL: f64 = -1.15;
        const TREND: f64 = 0.95;
        const SHIFT: f64 = 0.6;
        const DRIVE: f64 = 1.2;
        let residual_sd = (1.0 - SHIFT * SHIFT).sqrt();
        let mut state = 0x2926_0A1C_0000_0001_u64;
        let mut unit = || (gam_linalg::utils::splitmix64(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
        let mut x = Array1::<f64>::zeros(ROWS);
        let mut z = Array1::<f64>::zeros(ROWS);
        let mut times = vec![0.0; ROWS];
        for row in 0..ROWS {
            let mut gauss = || {
                let u1 = unit().max(f64::MIN_POSITIVE);
                let u2 = unit();
                (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
            };
            x[row] = gauss();
            z[row] = SHIFT * x[row] + residual_sd * gauss();
            // `α(q) + b·z = −Φ⁻¹(u)` with the anchor `α = q·√(1 + b²v) − b·m(x)`.
            let u = unit().clamp(1e-9, 1.0 - 1e-9);
            let quantile = gam_math::probability::standard_normal_quantile(u).expect("normal quantile");
            let q = (-quantile - DRIVE * z[row] + DRIVE * SHIFT * x[row])
                / (1.0 + DRIVE * DRIVE * residual_sd * residual_sd).sqrt();
            times[row] = ((q - LEVEL) / TREND).exp();
        }
        let weights = Array1::<f64>::ones(ROWS);
        let span = x.clone().insert_axis(ndarray::Axis(1));
        let evidence = estimated_latent_law::conditional_law_evidence(&z, &weights, Some(span.view()))
            .expect("the span test");
        let candidates = MovingLawCandidates::build(
            &z,
            &weights,
            span.view(),
            &estimated_latent_law::LocalLawContext {
                features: span.view(),
                feature_cols: vec![0],
            },
            DEFAULT_EMPIRICAL_LATENT_GRID_SIZE,
            &LatentZPolicy::default(),
            evidence,
            "survival anchor-time pin",
        )
        .expect("the moving-law candidates");
        assert_eq!(candidates.fitted_arm(), MovingLawArm::LocationScaleGaussian);
        // The true slope on the fitted `ζ` axis, where the closed form anchors.
        let slope = DRIVE * residual_sd;
        let index_at = |time: f64| LEVEL + TREND * time.ln();
        let certify_at = |anchor_time: &dyn Fn(usize) -> f64| {
            let losses: Vec<Option<Vec<f64>>> = (0..ROWS)
                .map(|row| {
                    let q = index_at(anchor_time(row));
                    let own = EmpiricalZGrid {
                        nodes: vec![candidates.own_score(row)],
                        weights: vec![1.0],
                    };
                    let anchor = |law: &EmpiricalZGrid| {
                        estimated_latent_law::survival_anchor_log_probabilities(
                            q * (1.0 + slope * slope).sqrt(),
                            slope,
                            law,
                        )
                        .expect("anchor log probabilities")
                    };
                    let own = anchor(&own);
                    let arms = candidates
                        .row_laws(row)
                        .expect("row laws")
                        .iter()
                        .map(anchor)
                        .collect();
                    moving_law_row_losses(&[MovingLawAnchor { arms, own }]).expect("losses")
                })
                .collect();
            candidates.certify(&weights, &losses).expect("the certificate")
        };
        let partner = certify_at(&|row| times[candidates.partner(row)]);
        let own = certify_at(&|row| times[row]);
        eprintln!(
            "[2926 anchor times] n={ROWS} | partner's time: {} || own time: {}",
            partner.summary(),
            own.summary()
        );
        assert_eq!(
            partner.chosen,
            MovingLawArm::LocationScaleGaussian,
            "anchored at a partner's time the rule must keep the true location-scale Gaussian law: \
             {}",
            partner.summary()
        );
        assert_ne!(
            own.chosen,
            MovingLawArm::LocationScaleGaussian,
            "negative control: anchored at each row's own time the certificate is biased and must \
             leave the true law, or this pin cannot tell the two anchor rules apart: {}",
            own.summary()
        );
    }

    /// A record that names why it carries no certificate round-trips through the
    /// saved model, a model saved before the field existed loads without it, the
    /// save-time check admits it, and `require_certified` refuses it with its
    /// reason, never taking the reason in place of the certificate.
    #[test]
    fn an_uncertified_moving_law_round_trips_and_certification_refuses_it_2926() {
        // Any stated reason: the record, not the reason, is under test.
        const REASON: &str = "the certificate was not taken for this fit";
        let evidence = ConditionalLawEvidence {
            mean_p_value: Some(1e-9),
            variance_p_value: Some(0.4),
            skewness_p_value: Some(0.5),
            alpha: AUTO_Z_CONDITIONAL_RAO_ALPHA,
        };
        let uncertified = LatentLawConsumed::EstimatedMovingLaw {
            evidence: evidence.clone(),
            arm: MovingLawArm::LocationScaleGaussian,
            contexts: 7,
            certificate: None,
            uncertified: Some(REASON.to_string()),
        };
        let text = serde_json::to_string(&uncertified).expect("serialize");
        let loaded: LatentLawConsumed = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(loaded, uncertified);
        assert!(loaded.require_recorded("save").is_ok(), "a stated reason is saved");
        let refusal = loaded
            .require_certified("certified fit")
            .expect_err("no certificate, so no certified fit");
        assert!(
            refusal.contains(REASON),
            "the refusal must name the reason: {refusal}"
        );

        let provisional = LatentLawConsumed::EstimatedMovingLaw {
            evidence,
            arm: MovingLawArm::LocationScaleGaussian,
            contexts: 7,
            certificate: None,
            uncertified: None,
        };
        let mut old: serde_json::Value = serde_json::to_value(&provisional).expect("serialize");
        old.as_object_mut().expect("an object").remove("uncertified");
        let loaded: LatentLawConsumed = serde_json::from_value(old).expect("an old record loads");
        assert_eq!(loaded, provisional);
        assert!(
            loaded.require_recorded("save").is_err(),
            "a provisional record with no reason is never saved"
        );
    }
}
